# Qwen3.8 MTP3 rollout 与在线训练

本分支实现单步 MTP teacher forcing 与三步 MTP rollout。训练只有一个辅助 CE 目标，系数 0.2；推理时将同一个原生 head 自回归展开三步。此前误加的三深度训练循环与配置已删除

## 训练与部署约定

下表记录已完成 GPU 验收的三轮配置。`dev_csl2` 的 launcher 保留两轮、24K/32K 的 context 默认值，MTP 默认是三步推理和单步训练；三轮 40K 用下面的显式参数复现

Qwen3.8 27B 使用 Qwen3.5 架构定义，checkpoint 只有一个原生 MTP layer。权重始终保留 `mtp.layers.0`。按 slime 的[在线 MTP SFT](https://thudm.github.io/slime/advanced/speculative-decoding.html) 约定，teacher forcing 使用 target hidden `h[t]` 和真实 token `x[t+1]`，监督 `x[t+2]`；推理的 draft 步数独立设置

| 项目 | 配置 |
|---|---|
| rollout | NEXTN，steps=3，top-k=1，draft tokens=4 |
| trainer | 原生 MTP layer=1，单步 teacher forcing |
| MTP loss | 单个 CE 目标系数 0.2；默认 per-token，由 MCore 按 step response token 统一归一化，MTP 与 policy 均补偿 CP |
| 梯度 | MTP 输入的 target hidden、embedding 和 lm-head stop-gradient；主干继续接受原有 RL 梯度 |
| actor | 两台训练节点、16 张 H20，TP4、PP2、CP2、SP，distributed FlashQLA GDN |
| PP 与重算 | 37 / 27 层，R27；MTP 单独重算，CE 投影也重算 |
| rollout 硬件 | 两台 rollout 节点、16 张 H20，四个 TP4 engine，CUDA graphs 开启 |
| 精度与 payload | BF16 trainer、FP8 rollout、T=1、top-p=1、top-k=-1、sampled logprob + top-20 |
| 正式批量 | 16 prompt groups × 16 samples = 256 轨迹 |
| 三轮 context | 24576 / 32768 / 40960，保留原有 token history、TRLOO、反馈和过滤流程 |

`MTP_STEPS=0/1/2/3` 只控制 rollout。trainer 使用 `--enable-mtp-training --mtp-num-layers 1 --mtp-loss-scaling-factor 0.2`，没有多深度训练参数

原有 PP33/31 三轮配置的显存余量较小。MTP 在输出 stage 增加 head 和 embedding 副本，因此将四个 decoder layer 移到前一个 stage。MTP 自身的 config 单独复制为 uniform/1 重算，decoder 仍使用 block 重算

## 正确性修复

- `full_loss_masks` 已按 next-token target 对齐。MTP 预测 `x[t+2]`，raw labels 移两次、现有 mask 移一次，遵守 packed sequence 和 CP 边界
- 沿 CP 汇总有效 token 数用于日志，空分片参与通信并保留零梯度。非 per-token 模式按全 CP 的 MTP token 数求均值；per-token 模式采用 policy 相同的 step denominator，两条路径均补偿 CP
- 推理预算取 `min(turn_cap, serving_context - draft_tokens)`，只在服务的全局 context 边界预留四个位置；前两轮的 24K / 32K 预算完整保留
- 应用 [FA3 context headroom 修复 #35985](https://github.com/sgl-project/sglang/pull/35985) 的逻辑，扩大 CUDA-graph page table；该 PR 在查询时仍未合并
- 应用 [Mamba ghost node 与 track step 修复 #35821](https://github.com/sgl-project/sglang/pull/35821)，适配当前 allocator 的 `free` 接口；继续使用 Triton GDN 和 `extra_buffer`
- 当前 SGLang 的分布式更新由 target loader 接收，原路径会跳过 `mtp.*`。补丁将同一次 NCCL 接收中的 MTP 张量交给原生 draft loader，并保留 target / draft 的共享 embedding 与 lm-head
- 启用 MTP training 时关闭 draft CPU backup，使用每次 actor 更新传来的 head

完整社区问题与冻结权重吞吐比较见 [MTP rollout 分析](mtp_rollout.md)。这些修复的范围是当前 H20 / FA3 / Triton 栈，仍需持续观察长时间运行

## 多轮 packing 与 MTP

`PACK_MULTI_TURN_TRAJECTORIES=1 ENABLE_MTP_TRAINING=1` 使用单步 teacher forcing，rollout 仍由 `MTP_STEPS` 独立控制。默认保持 packing 关闭；开启后遵守[多轮轨迹合并合同](trajectory_packing.md#等价性合同)，包括精确预测前缀、零 dropout 和其它参数约束

`tokens` 保存实际 causal history，`target_tokens` 保存原轮次的监督答案。模板修复可能将原结束 token 后移，MTP 标签需要保留原预测位置上的答案；head 的 teacher 输入和 attention 历史仍必须来自 `tokens`。`get_batch` 将独立标签按输入相同的 CP 切分、序列 padding 和 microbatch padding 布局整理为 `mtp_labels`；trainer 将其移两位，现有 next-token mask 移一位。轮次之间保留共享上下文，独立 trajectory 之间仍由 packed sequence 边界隔离

per-token 模式将 MTP CE 求和，并使用 policy 相同的 step 分母。packing 保留原来的逐轮 `sum(max(mask.sum(), 1))`，因此过滤轮和空轮不改变辅助 loss 的相对系数。非 per-token 模式原先按 microbatch 的 MTP 均值累积，合并会改变其权重；该组合明确拒绝，不自动切换训练目标。其它模型的 MTP 实现未接入此合同，仍不开放这一组合

`train/mtp_loss` 日志仍沿用 microbatch CE 均值再平均的统计方式，区别于 per-token backward 的 step 归一化。packing 会改变各 microbatch 的组成，不能直接用该日志标量判断 packing 前后的训练目标是否等价；回归比较的是统一 step 分母下的 CE 与梯度

CPU 回归覆盖 CP1/2/4、THD/BSHD 的标签布局与 padding、终止 token 修复、参数检查及原有 RL 等价性。两卡原生 head 回归使用真实 converter 和 `get_batch`，比较逐轮 microbatch 与同批合并轨迹的单步 CE、12 个 head 参数梯度及主干梯度隔离，覆盖不同轮长、过滤轮、补齐轮和全空轨迹。BF16 比较采用明确容差，不代表位级等价；27B GDN 主干已有的 packing 数值差异仍以[轨迹合并验证](trajectory_packing.md#验证与边界)为准

真实 27B train-only 使用留存的 24 轮、8 条轨迹，输入总量从 673543 降为 275815 tokens，最长 40958；237108 个有效监督 token 的 step 分母为 237109，保留一个被过滤轮次的 clamp。人工复核 4 处模板修复：实际历史读入换行，MTP 标签仍监督原 `<|im_end|>`

两台训练节点的 TP4/PP2/CP2、SP、PP37/27、R27、BF16 distributed FlashQLA 完成一次非零 RL advantage 下的联合更新，actor train 66.54s、MTP loss 0.30694、grad norm 0.18024；作业 `raysubmit_nWKMZL7RFeXu48Cd` 成功。两节点 445 个运行源码 hash 一致，Ray 实际执行包的四个关键文件再次核对通过。此项检查从不可变 MTP actor 输入初始化，未保存 checkpoint、未创建 SGLang；不将单次 train-only 时间宣称为端到端提速，也未重跑 packing 下的 refit/恢复流程

独立只读审查认可输入/标签分离、CP2 布局和 per-token 归一化，未发现当前正确性错误。按审查补齐 DP→actor→真实 trainer forward 的标签回归，并直接 hook 原生 head 的 embedding 输入；旧 trainer 标签传递的进程内反事实被回归准确拒绝。27B 的 split/packed GDN 数值等价仍未宣称成立

本次源码、CPU/GPU 日志、真实 replay 与独立审查证据保存在 [packing 与 MTP 验证](../../local_artifacts/qwen38/mtp_packing_20260906/)，其中 `manual_samples.txt` 可查看终止 token 修复，`replay_audit.json`、`train_result.json` 和 `ray_bundle_manifest.json` 保存数据与执行核对

## 节点本地 checkpoint 的共享 embedding

原 actor 输入 checkpoint 没有 MTP 参数。新的 BF16 conversion 显式使用 `--mtp-num-layers 1`，包含全部 12 个 Megatron MTP 参数张量；完整输入 checkpoint 的三份副本 SHA256 一致

MCore 在第一个 PP stage 保存共享 embedding，最后一个 stage 的 MTP 也需要读取它。节点本地目录只保留各自的 storage files，直接恢复会在输出 stage 报缺少 `__0_0.distcp` 等文件

恢复前自动生成 `${CHECKPOINT_SAVE_PATH}.mtp_resume` 加载缓存：本地 storage 与 rollout state 使用链接，只从其它 actor 节点读取缺失 embedding 的序列化字节段，再改写缓存中对应的 DCP storage location。当前模型补齐四段约 2.54GB 数据，原始 checkpoint 继续作为保存、归档和转换的来源

每个 DCP reader 使用所在节点的 storage map，跨 rank 的 tensor key、shape 和 chunk offset 保持一致。传输检查 checkpoint metadata、字节长度和摘要；失败的构建保留已有缓存，多源读取支持超时后切换

## 单步训练验收

| 验收 | 结果 |
|---|---|
| 两卡 CP2 packed | per-token 与非 per-token 两种模式的单个预测目标、标签/mask oracle、直接 CE 参考梯度、空 mask、主干梯度隔离全部通过 |
| 两卡 head 更新 | 三个 optimizer step 的单步 loss 为 5.627 / 5.203 / 4.805，12 个 head 参数均有非零梯度 |
| 真实 40K 恢复 | 从新单步训练 checkpoint 恢复模型与 optimizer，TP4/PP2/CP2，最长 40956 tokens 的三轮 replay 完成；trainer 41.87s、MTP loss 0.12623、grad norm 0.15822 |
| 40K 作业 | `raysubmit_jQfqxU5NZ4GCdVy5` 成功；仅 RAM 内更新，源 checkpoint tracker 保持 0，没有新增保存 |
| 修正后完整环路 | 32 轨迹、96 turn samples、935545 response tokens；92 completed、4 truncated、0 aborted，最长 39656 tokens |
| 数据与源码 | sampled logprob、21 列支持集、loss mask 通过；四节点 1205 个源码文件 hash 一致，Ray 实际执行包的四个关键训练文件也一致 |
| 联合 RL/MTP 更新 | 96 microbatches，trainer 385.29s，单步 MTP loss 0.21495、PPO KL 0.002797、grad norm 0.15394 |
| 保存与 refit | checkpoint 0 的 32 个 storage 文件完整，初始与最终回传均覆盖四个 TP4 engine 的 22 个 MTP 张量 |
| 保存后的 head | 7/12 个张量、11990242 个元素出现 BF16 可见变化，所有张量有限；两卡梯度测试独立确认全部 12 个参数接受梯度 |

单轨迹 replay 的 RL advantage 为零，这项检查验证长 context 的单步 MTP 前后向与 optimizer 恢复；32 轨迹的完整环路已独立验证非零 RL advantage 下的联合更新、保存和 rollout refit

修正后完整环路作业为 `qwen38-mtp3-full-loop-smoke-20260906.021610`。真实 rollout 耗时 184.32s；训练日志中的 6475.8 tokens/s 包含 context tokens，不能当作生成吞吐。人工复核了首轮不完整回答、带错误反馈的长上下文重写，以及获得正 reward 的修复样本；TVM-FFI 指令和三轮反馈仍进入实际上下文

合入 `dev_csl2` 时保留两轮 context 默认、轨迹打包与诊断接口；合并后的定向 CPU 测试和两种归一化模式的两卡 CP2 回归通过。上述三轮完整训练数字来自合并前的源码快照，合并检查未重新运行四节点完整训练

显存约每五秒采样一次，两台 actor 节点的采样最大 resident 分别为 60995MiB、85748MiB。监控在任务退出阶段因 SSH 查询超时结束，数字属于采样最大值；不视为分配器精确峰值，也不据此宣称正式 256 轨迹的容量上限

独立审查确认单步 teacher forcing 语义成立，并指出生产 per-token 模式未在原测试中覆盖。核对 slime 的完整 response-token 计数与 MCore SUM/finalize 后，补测在修复前复现梯度仅为参考的一半，修复后两种模式均通过。`0.2` 保持为单步 CE 系数，CP 补偿与 policy 的分布式归一化约定一致

此前 32/256 轨迹的完整环路使用了误加的三深度训练目标。这些运行确实验证了 checkpoint、通信和 refit 机制，但其 loss、acceptance、训练时间与 checkpoint 均不能作为单步 teacher forcing 方案的结果。旧原始产物保留，修正后的验证从不可变 BF16 输入重新开始

不受训练目标变化影响的已有检查包括：CPU context/kernel 24 passed（2 个服务测试未启用）、runtime patch 9 passed、bridge 与 zero-token 11 passed、加载缓存 7 passed；40K serving 边界的并发 1/3/4/8 与四个 TP rank 的 NCCL MTP norm 变化/恢复均通过

此前冻结权重的 [MTP 1/2/3 rollout 比较](mtp_rollout.md) 仍可用于推理选型；它与在线 MTP 训练深度无关。训练带来的额外 acceptance 与吞吐收益需要相同输入条件下的对照，当前不据这些短检查宣称长期收益

## 使用

### v4_1 三轮 packed TRLOO 正式启动

用户指定的 v4_1 三轮训练从不可变 BF16 MTP 输入初始化，启用 trajectory packing，`OVERLONG_PENALTY_FACTOR=0` 使 reward 函数直接跳过长度惩罚。两台节点训练、两台节点 rollout，每台 8 张 H20，MTP3 推理与单步 MTP teacher forcing 系数 0.2；其余使用上表的三轮 40K、256 轨迹、PP37/27 与 R27 配置

作业 `raysubmit_SqMVX9e1pARFJJzq` 提交至专用 Ray 集群，[W&B quc7eiwz](https://wandb.ai/shuailin_chen/slime/runs/quc7eiwz)记录训练。配置、启动命令、源码与数据哈希、真实 prompt 抽样统一保存在[本次配置 review](../../local_artifacts/qwen38/trloo_v4_1_packed_mtp_20260906/config_review.md)，实际执行使用同目录 `runtime_repo` 快照；该记录不代表作业此刻仍在运行

checkpoint 写入两台 actor 的节点本地实验目录，每 20 step 保存并保留最近两代完整 checkpoint；目录需满足清理器的实验路径约束，部署位置见 [RUNTIME.md](../../RUNTIME.md)。当时的冻结源码未读取首批留样上限，运行侧采用作业内保留器限制为首批加最新一批，详见配置 review。维护源码现已直接支持限量保存与最新批次原子替换，入口为 [RolloutManager](../../slime/ray/rollout.py)

首批真实留样核对了 768 轮 → 256 条轨迹，输入 token 约 1,396 万 → 603 万，最长 40,956，未产生长度惩罚 metadata。actor 完成两个 optimizer step，MTP loss 分别为 0.30735 / 0.30499，grad norm 为 0.14463 / 0.15450，未达到首次 checkpoint 保存点

首次训练后 refit 成功，但恢复长请求的 prefill 时，两个 rollout engine 在 input-logprob 的 `logits.float()` 分配 1.89 GiB 临时张量失败并退出。下一次回传访问退出 engine 的 `/pause_generation`，作业于 09:20:19 UTC 失败。[OOM 因果证据与修复候选](../../local_artifacts/qwen38/trloo_v4_1_packed_mtp_20260906/oom_diagnosis.md)记录默认 2048 行全词表 FP32 logits 的峰值、Ray 环境变量传递要求，以及保留 MTP3/CUDA Graph 的较小 chunk 与静态显存预算方案；诊断时尚未做候选方案的 GPU 回归，后续验收见下文；首轮 refit RPC 成功不能代替恢复生成检查

### 维护入口

维护 launcher 对 matched serving 默认使用 `SGLANG_LOGPROB_CHUNK_SIZE=256`、`SGLANG_MEM_FRACTION_STATIC=0.80`，并把 logprob chunk 开关和大小显式传入 Ray runtime_env。该配置针对上述 refit 后历史输出重算的全词表 FP32 logits 峰值，不改 MTP3、CUDA Graph、训练批量或上下文上限

修复后的两节点四引擎隔离检查完成 128 条长请求、两次原生 MTP norm 权重的 NCCL 回传与恢复，共约 210 万输出 token；四引擎均存活，sampled/top-20 有限，流式样本已有 behavior logprob 保持不变。压力请求强制忽略 EOS，不用于生成质量结论；隔离检查只回传一个 MTP 参数，正式 actor 全量回传单独验收。配置、进程环境、显存采样、脚本修正与正式重启证据由[refit 显存修复配置](../../local_artifacts/qwen38/trloo_v4_1_refit_fix_20260906/config_review.md)统一维护

正式修复作业为 `raysubmit_KXWmmCHVJCkDVGeZ`、[W&B ql06a0fc](https://wandb.ai/shuailin_chen/slime/runs/ql06a0fc)。首步 MTP loss=0.31339、grad norm=0.14291；全量 actor/MTP 回传后四引擎均完成恢复 prefill，后续 68 秒内每个引擎的生成 token 计数持续增长，未复现 OOM，当时第二步训练已开始。该回执验证首个训练后 refit，后续 checkpoint 与恢复结果单独记录

该作业 step31 起的 rollout_time 突增对应 fully-async 完成队列库存耗尽，客户端评测耗时明显上升，MTP acceptance 与模型请求耗时保持稳定。实际等待函数已通过隔离 CPU Ray actor 复现心跳查询阻塞结果回收的问题；真实后端慢任务也有贡献，两者占比及最初触发点尚未确定。[rollout 等待分析](../../local_artifacts/qwen38/trloo_v4_1_refit_fix_20260906/timing_analysis/report.md)统一保存逐 step 指标、生产样例、复现与修复建议，本次分析没有修改或重启正式训练

维护源码现已将结果 Future 与独立心跳 task 解耦，心跳超时或失败只停止该请求的心跳，结果返回不再经过默认线程池或控制查询。14 项 CPU 回归及已有 HTTP retry 回归通过；真实 Ray 对照中，结果就绪后的额外等待从约 806ms 降至 2.1ms，取消与异常路径通过。[结果等待修复](../../local_artifacts/qwen38/trloo_v4_1_refit_fix_20260906/result_wait_fix/report.md)保存源码 hash、验证、独立审查、结果下载采样，以及对用户提供的 KernelGym 输入生成分析的判断；源码验收时未在线替换，部署在后续 step40 恢复中完成

按用户指令停止旧作业后，从已完成的 `iter_0000039` 恢复，新作业为 `raysubmit_qgM4kabv9iZYm6nB`、[W&B ffe0dbz3](https://wandb.ai/shuailin_chen/slime/runs/ffe0dbz3)。32 个 checkpoint 分片固定到独立保存目录，恢复模型、optimizer、调度器和数据源状态，下一次编号为 40；仅结果等待修复进入新源码快照。四节点实际 Ray 包核对通过，actor/MTP 回传后四引擎持续生成，rollout40 已开始。启动验收时尚未完成恢复后的第一个 optimizer step；[恢复配置与证据](../../local_artifacts/qwen38/trloo_v4_1_resume40_waitfix_20260907/config_review.md)统一记录路径、参数、加载日志及 KernelGym 部署等待，不据初始冷启动时间推算稳态吞吐

该恢复作业后续完成到零基日志编号 119，并在训练编号 120 期间按用户指令停止。停止前已完成的两代 checkpoint 为 `iter_0000099`、`iter_0000119`；step100 的两节点分片另行合并导出，用原生 MTP3 完成 KernelBench L1–L3 三轮评测。分数、导出格式修复和验证边界见 [KernelBench 评测](kernelbench_eval.md#v4_1-三轮-trloo-step100)，训练保持停止

代码随 `dev_csl2` 维护，GPU 实验证据与隔离 SGLang 副本使用当前工作区 `local_artifacts/qwen38/` 入口；早期单步／三深度产物的可用性见文末。旧实验 worktree 已移除；其历史 HEAD `9ff07faf` 由 tag `archive/qwen38-mtp3-train-20260911` 和独立 Git bundle 保留，源码及未跟踪文件见[归档清单](../../local_artifacts/worktree_cleanup/20260911_152306/source_manifest.json)。模型文件及缺失产物通过同文件系统重命名迁入，已有同 inode 文件保留当前目录项，逐项对应见[迁移清单](../../local_artifacts/worktree_cleanup/20260911_152306/artifact_migration.json)。先同步选定 checkout 的代码、数据与配置并核对 hash，在各自运行容器内准备隔离的 SGLang 副本，机器和端口映射见 [RUNTIME.md](../../RUNTIME.md)；本地路径迁移不代表远端部署已更新或重新通过 GPU 验收

```bash
python scripts/patch_sglang_qwen_mtp.py \
  --overlay-dir local_artifacts/qwen38/mtp_runtime
```

在 Ray head 的运行容器内启动维护入口；默认四节点、MTP3 rollout 和单步 teacher forcing。共享 Ray 已由其它会话使用时，协调 GPU 资源后设置 `REUSE_RAY_CLUSTER=1`

```bash
MAX_TURNS=3 MAX_CONTEXT_LEN=40960 \
TURN_MAX_CONTEXT_LENS="24576 32768 40960" \
bash examples/kernel_agent/run.t1.qwen3.8.27B.fasync.sh
```

`ENABLE_MTP_TRAINING=0` 可关闭辅助训练，`MTP_STEPS=0` 可关闭 speculative rollout。关闭辅助训练时自动选择原来的非 MTP actor 输入和 PP33/31 默认分层。`PACK_MULTI_TURN_TRAJECTORIES=1` 可与默认的 MTP 辅助训练同时开启；组合要求 Qwen 原生 spec 和 `--calculate-per-token-loss`，launcher 已使用该归一化方式

恢复时设置 `RESUME_FROM_SAVE=1` 与已有 `CHECKPOINT_SAVE_PATH`，加载缓存自动准备。隔离恢复测试可同时设置 `LOAD_DEBUG_ROLLOUT_DATA` 和 `DISABLE_CHECKPOINT_SAVE=1`，完成一次 RAM 内更新后退出

## 代码与证据

- [单步 MTP trainer](../../slime_plugins/models/qwen3_5_mtp.py)、[模型接入](../../slime/backends/megatron_utils/model_provider.py)、[两卡 CP2 回归](../../tests/test_qwen3_5_mtp_training.py)
- [SGLang 补丁](../../scripts/patch_sglang_qwen_mtp.py)、[engine 参数](../../slime/backends/sglang_utils/sglang_engine.py)、[context 预算](../../examples/kernel_agent/generate_with_cuda_agent.py)
- [节点本地恢复缓存](../../scripts/prepare_mtp_resume_view.py)、[训练入口](../../examples/kernel_agent/run.t1.qwen3.8.27B.fasync.sh)
- 单步训练产物原目录 `local_artifacts/qwen38/mtp1_train_20260906/`：`source_manifest_*.json`、`ray_bundle_manifest.json`、`train1_source.tar.gz`、`mtp1_cp2_both_modes.log`、`per_token_before_fix.log`、`final_review_verdict.txt`、`review_response.md`
- 完整环路的本地摘要与样本：`full_loop_metrics.json`、`full_loop_data_audit.json`、`full_loop_checkpoint_audit.json`、`full_loop_log_audit.json`、`mtp_checkpoint_delta.json`、`full_loop_manual_samples.txt`、`partial_memory_summary.json`；恢复证据为 `resume40k_metrics.json` 和 `resume40k_audit.json`
- 运行容器中的 `train40k_engine.log`、`full_loop_engine.log`、`resume40k_engine.log` 和 `full_loop_rollout_0.pt` 保存原始训练证据；两台 actor 分别保留 `full_loop_checkpoints/` 分片，部署位置见 [RUNTIME.md](../../RUNTIME.md)
- 此前基础设施与三深度实验原目录 `local_artifacts/qwen38/mtp3_train_20260905/`：`refit_*.json`、`actor_checkpoint_manifest_*.json`、`production_data_audit.json`、`production_checkpoint_audit.json`；该目录的训练产物不属于当前单步方案
- [DFlash2 接入分析](dflash2.md) 单独维护外部 draft 的运行时差距、训练接入与社区问题

上述 `mtp1_train_20260906` 与 `mtp3_train_20260905` 入口在本次文档核查时仍是指向已移除 worktree 的失效链接。[迁移清单](../../local_artifacts/worktree_cleanup/20260911_152306/artifact_migration.json)保留文件名与当时的迁移记录，但不能替代原始内容；相关早期验证沿用历史记录，本次未重新读取这些产物或复跑 GPU 验证
