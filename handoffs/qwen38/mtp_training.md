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
| actor | node69 + node70，TP4、PP2、CP2、SP，distributed FlashQLA GDN |
| PP 与重算 | 37 / 27 层，R27；MTP 单独重算，CE 投影也重算 |
| rollout 硬件 | node53 + node64，四个 TP4 engine，CUDA graphs 开启 |
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

显存约每五秒采样一次，actor 的采样最大 resident 为 node69 60995MiB、node70 85748MiB。监控在任务退出阶段因 SSH 查询超时结束，数字属于采样最大值；不视为分配器精确峰值，也不据此宣称正式 256 轨迹的容量上限

独立审查确认单步 teacher forcing 语义成立，并指出生产 per-token 模式未在原测试中覆盖。核对 slime 的完整 response-token 计数与 MCore SUM/finalize 后，补测在修复前复现梯度仅为参考的一半，修复后两种模式均通过。`0.2` 保持为单步 CE 系数，CP 补偿与 policy 的分布式归一化约定一致

此前 32/256 轨迹的完整环路使用了误加的三深度训练目标。这些运行确实验证了 checkpoint、通信和 refit 机制，但其 loss、acceptance、训练时间与 checkpoint 均不能作为单步 teacher forcing 方案的结果。旧原始产物保留，修正后的验证从不可变 BF16 输入重新开始

不受训练目标变化影响的已有检查包括：CPU context/kernel 24 passed（2 个服务测试未启用）、runtime patch 9 passed、bridge 与 zero-token 11 passed、加载缓存 7 passed；40K serving 边界的并发 1/3/4/8 与四个 TP rank 的 NCCL MTP norm 变化/恢复均通过

此前冻结权重的 [MTP 1/2/3 rollout 比较](mtp_rollout.md) 仍可用于推理选型；它与在线 MTP 训练深度无关。训练带来的额外 acceptance 与吞吐收益需要相同输入条件下的对照，当前不据这些短检查宣称长期收益

## 使用

代码随 `dev_csl2` 维护，开发分支为 `feature/qwen38-mtp3-train`，GPU 实验证据保留在 `/nfs/FM/chenshuailin/projects/kernel_agents/slime-qwen38-mtp3-train`。先同步选定 checkout 的代码、数据与配置并核对 hash，在各自 `:23538` 容器内准备隔离的 SGLang 副本

```bash
python scripts/patch_sglang_qwen_mtp.py \
  --overlay-dir local_artifacts/qwen38/mtp_runtime
```

在 node70 容器启动维护入口；默认四节点、MTP3 rollout 和单步 teacher forcing。共享 Ray 已由其它会话使用时，协调 GPU 资源后设置 `REUSE_RAY_CLUSTER=1`

```bash
MAX_TURNS=3 MAX_CONTEXT_LEN=40960 \
TURN_MAX_CONTEXT_LENS="24576 32768 40960" \
bash examples/kernel_agent/run.t1.qwen3.8.27B.fasync.sh
```

`ENABLE_MTP_TRAINING=0` 可关闭辅助训练，`MTP_STEPS=0` 可关闭 speculative rollout。关闭辅助训练时自动选择原来的非 MTP actor 输入和 PP33/31 默认分层。轨迹打包仍保留原有不兼容检查：使用 `PACK_MULTI_TURN_TRAJECTORIES=1` 时必须显式设置 `ENABLE_MTP_TRAINING=0`；不将 packing 与 MTP 辅助训练的组合宣称为已验证

恢复时设置 `RESUME_FROM_SAVE=1` 与已有 `CHECKPOINT_SAVE_PATH`，加载缓存自动准备。隔离恢复测试可同时设置 `LOAD_DEBUG_ROLLOUT_DATA` 和 `DISABLE_CHECKPOINT_SAVE=1`，完成一次 RAM 内更新后退出

## 代码与证据

- [单步 MTP trainer](../../slime_plugins/models/qwen3_5_mtp.py)、[模型接入](../../slime/backends/megatron_utils/model_provider.py)、[两卡 CP2 回归](../../tests/test_qwen3_5_mtp_training.py)
- [SGLang 补丁](../../scripts/patch_sglang_qwen_mtp.py)、[engine 参数](../../slime/backends/sglang_utils/sglang_engine.py)、[context 预算](../../examples/kernel_agent/generate_with_cuda_agent.py)
- [节点本地恢复缓存](../../scripts/prepare_mtp_resume_view.py)、[训练入口](../../examples/kernel_agent/run.t1.qwen3.8.27B.fasync.sh)
- [单步训练产物](../../local_artifacts/qwen38/mtp1_train_20260906/)：`source_manifest_*.json`、`ray_bundle_manifest.json`、`train1_source.tar.gz`、`mtp1_cp2_both_modes.log`、`per_token_before_fix.log`、`final_review_verdict.txt`、`review_response.md`
- 完整环路的本地摘要与样本：`full_loop_metrics.json`、`full_loop_data_audit.json`、`full_loop_checkpoint_audit.json`、`full_loop_log_audit.json`、`mtp_checkpoint_delta.json`、`full_loop_manual_samples.txt`、`partial_memory_summary.json`；恢复证据为 `resume40k_metrics.json` 和 `resume40k_audit.json`
- node70 同路径的 `train40k_engine.log`、`full_loop_engine.log`、`resume40k_engine.log` 和 `full_loop_rollout_0.pt` 保存原始训练证据；checkpoint storage 分布在 node69/node70 的 `full_loop_checkpoints/`
- [此前基础设施与三深度实验原始产物](../../local_artifacts/qwen38/mtp3_train_20260905/)：`refit_*.json`、`actor_checkpoint_manifest_*.json`、`production_data_audit.json`、`production_checkpoint_audit.json`；该目录的训练产物不属于当前单步方案
- [DFlash2 接入分析](dflash2.md) 单独维护外部 draft 的运行时差距、训练接入与社区问题
