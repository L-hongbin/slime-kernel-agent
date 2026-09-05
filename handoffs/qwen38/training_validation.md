# Qwen3.8 训练验证与故障机制

本文保留已完成的隔离实验、容量检查和因果证据；配置选择与 lineage 统一见[训练配置](training.md)。日期和作业标识用于定位留存证据，不表示现场进程状态

## Distributed GDN 与 SP/P2P

三组均使用同一份 32-sample fixed rollout、10 个 microbatch 和相同训练参数；actor 拓扑均为 `TP4/PP2/CP2/DP1`，共 16 张 H20，训练为 BF16，GDN recurrence backend 为 FlashQLA。变量只有 GDN rank layout 与 SP/PP 通信方式：

| 配置 | SP / PP 通信 | actor time | actor throughput | 相对 baseline 吞吐 | 相对 baseline 时间 |
|---|---|---:|---:|---:|---:|
| baseline：replicated GDN | SP on / 原始 PP P2P | 67.352s | 2,197.63 tok/s | — | — |
| distributed GDN | SP off | 46.198s | 3,203.93 tok/s | **+45.79%** | **-31.41%** |
| distributed GDN + SP | SP on / rank-ordered PP P2P | 44.416s | 3,332.47 tok/s | **+51.64%** | **-34.05%** |

主要收益来自消除 replicated GDN 的 TP×CP 完整-head 重复计算：distributed GDN 本身约为 baseline 的 `1.458×` 吞吐；修复后的 SP 再在 distributed 基础上增加 4.01% 吞吐、减少 3.86% actor time，最终约为 baseline 的 `1.516×` 吞吐。以上是同一 replay 的单次严格 A/B，不外推为所有长度分布下的固定比例

这里有一个容易误判的 Qwen3.5 特例：HF `Qwen3_5RMSNorm` checkpoint 本身保存 zero-centered gamma，forward 再使用 `1 + gamma`；因此 fused GDN input norm 在 HF↔MCore 间必须原值复制，不能像传统 one-centered HF RMSNorm 再减/加 1。真实 `TELayerNormColumnParallelLinear` 探针确认 `zero_centered_gamma=true`，其输出与手工 `(gamma + 1)` 前向逐元素一致。曾按传统语义额外减 1 的 `_1pfix` base 将首层 gamma 均值从 `-0.03337` 推到 `-1.03335`，在线 run 出现训练熵 `9.8689`、train/rollout logprob 差 `12.1612`，已立即停止并拒绝。GDN `out_norm` 的 HF 权重仍是传统 one-centered 语义，原有 `-1/+1` 转换继续保留

SP 停滞由 PP batched-P2P 与 TP SP collective 的顺序环触发；更换 FlashQLA/FLA 或把 CP2 改成 CP1 均未解除，SP off 可以完成。固定版本的 rank-ordered P2P 修复后，三步在线更新、save 和 resume-prep 分别验证通过。中途一次保存失败来自磁盘写满，不能作为 SP 失败证据；该失败 checkpoint 已拒绝使用

证据入口：`local_artifacts/qwen38/distributed_gdn/formal_gate_20260828.md`；三组 fixed replay 的 node70 日志尾名依次为 `20260827.165314.log`、`20260828.014136.log`、`20260828.013652.log`；三步在线 job 为 `raysubmit_PZSR87XjWudwqNgv`，save gate 为 `qwen38-nospec-full-loop-smoke-20260828.025352`

## 精确 token history

仅把第一轮 decode 成 `reasoning_content/content` 再套模板，虽然能修正 Qwen3.8 的消息结构，但在 448 条真实 rollout 上只能完整匹配 277/448，聚合 token 复用约 77.6%；decode→re-encode 不可逆和模板 trim 仍会破坏前缀。因此最终没有走字段重渲染，而是让第二轮直接续接第一轮的 `prompt_ids + response_ids`，再以 token 级方式追加 verifier feedback 和新的 assistant thinking prefix

边界处理覆盖四种组合：已有/缺少 `</think>` × 有/无末尾 `<|im_end|>`。若第一轮缺少 `</think>`，先在 token 边界插入 `\n</think>\n\n`，必要时把末尾 `<|im_end|>` 移到闭标签之后。每条 trajectory 使用唯一 `session_id`，两轮都携带相同 `X-SMG-Routing-Key`，由 `consistent_hashing` 保证落到同一个 engine；因此不仅语义正确，也能实际复用 SGLang radix cache

使用原 448 条 Qwen3.8 rollout dump 和真实 tokenizer 的离线审计结果：

| 指标 | 结果 |
|---|---:|
| 原生成 response tokens | 8,363,173 |
| 完整 prompt+response 精确前缀 | 324/448 |
| 聚合精确前缀 | 9,390,585 / 9,390,709（99.99868%） |
| 生成 response 复用下界 | 8,363,049 / 8,363,173 |
| 缺少 `</think>` 并被修复 | 249/448 |
| 原始 feedback JSON 超过 8,000 字符 | 57/448（12.72%） |
| 第二轮 prompt mean / p50 / p95 / max | 22,072.66 / 24,151.5 / 25,088 / 27,792 |
| 第二轮 prompt 达到 32K | 0/448 |
| 剩余 response budget p50 / p05 / min | 8,616.5 / 7,680 / 4,976 |

只有 124 个末尾 `<|im_end|>` 因补 `</think>` 而移动，之前的全部 token 保持精确；448/448 的 longest-common-prefix 断言通过

## 32K 容量与层分配

trainer 使用完全相同的 protected replay：512 个 turn sample、5,880,342 total tokens、3,002,461 loss tokens、365 个 dynamic microbatch；最大总长度 32,766，4 条不低于 32K。单次严格结果如下：

| PP layers / recompute | actor time | throughput | step time | 相对 33/31 R29 |
|---|---:|---:|---:|---:|
| 33/31、R29 | 849.087s | 6,925.49 tok/s | 858.127s | baseline |
| 33/31、R27 | **839.515s** | **7,004.45 tok/s** | **848.372s** | actor -1.13%，吞吐 +1.14% |
| 32/32、R27 | 842.619s | 6,978.65 tok/s | 851.282s | actor -0.76%，吞吐 +0.77% |

保留 33/31 而不是按显存表象改成 32/32：R27 下 node69/PP0 的 180s `nvidia-smi` 采样约 56.3--58.3GB（工具原始 MiB 数值按十进制展示）、平均 SM 96.36%，node70/PP1 约 78.1--79.7GB、平均 SM 95.17%。32/32 虽略缩小 SM utilization 差，却把同一采样窗口的 PP1 峰值推到约 79.6--81.2GB，并使 actor 变慢 0.37%。PP1 的额外 loss/DPPO 固定开销使“显存平衡”和“计算平衡”不是同一个目标；继续把层从 PP1 移给本来已略忙的 PP0（如 35/29）会恶化计算平衡，所以没有运行。这个短窗口没有覆盖后述第二个 CP subgroup 的 32K 高水位，不能用来给容量留余量

随后又对 33/31 R27 同 replay 运行一次覆盖完整 GPU actor 生命周期的约 1 秒级物理显存监控，365/365 microbatch、Ray `SUCCEEDED`，actor 837.638s、7,020.15 tok/s、step 847.182s，loss/grad 有限。以下统一按工具原始 MiB 换算二进制 GiB：node69 八卡峰值为 57.84--59.96GiB；node70 的 local GPU 0--3 为 76.53--77.76GiB，local GPU 4--7 为 86.94--88.45GiB。Ray 节点/GPU 排序和 Megatron 默认 `tp-cp-ep-dp-pp` rank order 共同确认，后四卡对应 PP1 的第二个 CP subgroup；这一映射已由代码和 rank 公式确认，但约 11GiB 高峰的具体临时 tensor 尚未单独归因，不能写成“CP1 工作量更大”。四卡同时落在同一 rank subgroup，因此先按真实 physical high-water 处理，而不是采样噪声。最紧卡在 97,871MiB（95.58GiB）总量下只余 7,299MiB（约 7.13GiB），R27 的 32K train-only 容量门禁通过，但没有继续放松到 R25 的安全依据；正式 coupled step 仍需覆盖 initial sync、完整训练和 post-update sync，并复核真实在线 batch 下的峰值

`finalize=none` 会保留失败或退化的后续轮，改变 TRLOO credit assignment。一次非配对抽样中它增加了 loss tokens，不能把取消 small-group drop 当成吞吐收益；部署保留 `positive`

## 40K 三轮容量

三轮 R29 full-loop 峰值为 node70 94,922/97,871 MiB，余量 2,949 MiB；R31 峰值 92,230 MiB，余量 5,641 MiB，因此该部署选择 R31。下面是独立三轮 worktree 的已保存验证记录

- Rollout-only gate on node53 completed two real three-turn trajectories. It exercised totals of 24,576, 32,768, and 38,444 tokens on one trajectory, enforced all three caps, preserved token history, and rendered both inserted-close-thinking and already-closed boundaries correctly.
- Train-only replay on node70 + node69 completed forward, backward, and optimizer with a real 38,444-token sample at both R31 and R29. The n=1 replay has zero TRLOO advantage by construction but finite entropy, KL, and rollout/train logprob alignment.
- R29 full-loop smoke `qwen38-nospec-full-loop-smoke-20260831.215141` passed: rollout 522.0s, actor 383.5s, step 918.7s, loss -0.002637, entropy 0.17915, PPO KL 0.001796, grad norm 0.13293. Its 94,922 MiB node70 peak was rejected as too close to physical capacity.
- R31 full-loop smoke `qwen38-nospec-full-loop-smoke-20260831.222236` passed: rollout 695.1s, actor 396.5s, step 1,110.6s, loss -0.005190, entropy 0.19986, PPO KL 0.002129, grad norm 0.09384. Initial and post-update pause/flush/refit/continue succeeded on both TP4 engines. Peak memory was 92,230 MiB on node70, 48,051 MiB on node69, and 96,430 MiB on node53; no OOM, abort, or generation-guard timeout occurred.
- The R31 full-loop dump contained 96 turn samples in 32 complete trajectories. All token/logprob/top-k/loss-mask dimensions and rewards were finite and consistent. Of 64 inter-turn transitions, 41 retained the entire prior token sequence and 23 retained every token except the final `<|im_end|>`, replacing it with the deliberate `</think><|im_end|>` repair; aggregate retained prefix was 99.998504%. Manual decoded inspection confirmed correct Qwen boundaries and feedback tied to the actual validator failure. Gate dumps were deleted after audit.

## Prompt/backend 冲突造成整组零 reward

DataV4 首轮要求 TVM-FFI，但一版 tool-response 模板要求 pybind 的 `binding_registry.h`。256 条第二轮样本中，219 条在 precheck 失败，211 条直接命中该冲突。模型遵循冲突指令后才被 checker 拒绝，调 reward filter 或反馈预算不能修复这个原因

修复为 TVM-FFI short 模板、pre-Ray 模板校验和 generate-time backstop。正式 rollout 0 的该 marker 失败由 211 降至 0；precheck failure 从 219 降至 14。新旧 workload 受随机采样影响，3,329.3s→1,351.0s 的 wall 变化不单独作为参数性能估计。真实四轨迹 gate 为 8/8 precheck passed，后续轮 3/4 compiled/correct

证据：[模板 gate 报告](../../local_artifacts/qwen38/tvm_ffi_prompt_gate_20260829/report.md)，旧 job `raysubmit_QqRF4tqiRr8bFqvh`，修复后 job `raysubmit_t1Uk5zUQNc6mAxGD`

## Retract/refit 的 behavior provenance

SGLang 保留中断前 `output_ids`、sampled logprob 和 top-k；新权重重新 prefill 后继续生成。旧 token 不重新计算 behavior 分布，也不从 loss mask 中删除。flush/refit 失败时 abort parked requests，再 continue 解锁并保留原异常。waiting queue 非空时的 flush 依赖 `scripts/patch_sglang_retract_flush.py`，不能放松其他 active-state 检查

### 独立 TP4 检查

node53 使用正式 `/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.8-27B-FP8`、TP4、FA3 attention、Triton linear/Mamba backend 和非 eager 模式完成了真实 API 闭环：

| 检查 | 结果 |
|---|---|
| `retract -> flush -> same-checkpoint update -> continue` | 请求在 pause/flush/update 期间保持 pending；最终 `finish=length`、`num_retractions=1` |
| sampled logprob | 1,536/1,536 行完整 |
| predictive top-k | 1,536/1,536 行，每行 K=20 |
| streaming prefix 精确保留 | pause 前捕获的前 64 个 token、64 个 sampled logprob 和 64×20 top-k 在最终 response 中逐项完全相同 |
| 异常清理 | retract 后注入 abort+continue，请求数秒内以 `finish_reason=abort` 返回，无悬挂 |
| 相同权重普通 greedy 重复 | 256/256 token 完全相同，证明无 retract 时运行确定 |
| 相同权重 retract greedy | 第 0--70 token 相同，从 token 71 起允许分叉；pause 前捕获阈值为 64 token |

最后一行不是 prefix 丢失：中断前已返回的 token/logprob/top-k 已由 streaming 对比证明 byte-for-byte 保留。分叉发生在 resume 后的新 suffix，说明 GDN chunkwise re-prefill 重建的数值状态可以让未来 greedy 选择变化；partial rollout 的定义本来就允许中断后由新状态继续采样。训练 correctness 依赖每个 token 的真实 behavior distribution，而不是要求 suffix 与“不发生 pause”的反事实轨迹相同

24-GPU coupled smoke `qwen38-nospec-full-loop-smoke-20260830.012724` 完成 64 条 Sample、26 个 microbatch 和在途同步，actor 72.9s，step 394.45s。32-GPU smoke `qwen38-nospec-full-loop-smoke-20260830.064703` 也通过 correctness/synchronization，但遗留 scheduler 把有效 request cap 降到 108，其短批性能不作为正式基线

正式四引擎 job 在清理旧 scheduler 后每 engine 恢复 2,482,315 token slots 与 request cap 128，active prompt groups 保持 16。其 actor step0 为 895.11s，并行 rollout1 为 895.00s，配平结论只适用于这批长度分布。最终 response 的单值 weight-version 不能给出混合轨迹各版本 token 占比；prefix/logprob/top-k 保留由独立 streaming 测试证明，不能把同步前收齐的 capture 误称为 mixed evidence

## 单轮 serving 与长尾的已证边界

受控 gate 固定同一 8 个 prompt、medium、单样本单轮、16K、T=1、top-p 0.95、top-k 20、FP8 TP4、FA3、Triton GDN/mamba、CUDA Graph 和并发 1；唯一变量是 speculative path

| Arm | mean / median tokens | precheck | 完整三段 | unique-token range | rollout LP |
|---|---:|---:|---:|---:|---:|
| DSpark | 3,945.6 / 600.5 | 0/8 | 0/8 | 1.18%–47.53% | 8/8 finite |
| No-spec | 7,033.9 / 8,762.5 | 7/8 | 8/8 | 7.97%–20.75% | 8/8 finite |
| Native MTP | 7,391.8 / 6,702.0 | 7/8 | 7/8 | 5.87%–24.33% | 8/8 finite |

DSpark 有 6 条过早结束，另 2 条进入约 1% unique-token 的长重复；no-spec 和 MTP 恢复连贯实现。MTP wall time 384.3s、38.47 token/GPU/s，相对 no-spec 的 545.0s、25.81 token/GPU/s 快 1.49×，spec accept rate/length 为 0.7952/3.3850；59,134 个 response token 的 selected-token logprob 均有限且长度对齐。No-spec low follow-up 的 mean/median 为 7,123.4/7,896.5、8/8 precheck；相对 medium 的长度差置信区间跨 0，不能支持 effort 的系统性长度排序

该 gate 没有训练 actor、在线权重同步或 optimizer，只证明生成健康和 return-logprob。后续 MTP full-loop 虽以有限 loss/grad 完成 step 0/1，却在约 90 分钟持续负载下由 node53 scheduler 的 `copy_done.synchronize` 触发 CUDA illegal memory access，且没有 checkpoint；两轮 ExactGraph rollout-only 成功不足以证明稳定。当前生产因此使用 no-spec，MTP/DSpark 仅诊断。SGLang 0.5.16 的 MTP 必须使用 `extra_buffer`，不支持 speculative decoding 配 `extra_buffer_lazy`

Gate dumps 位于 `experiments/FAsync.{DSpark,NoSpec,MTP}.tvm_ffi.Qwen3.8-27B.BF16Train.FP8Rollout.CTX16384/reasoning_effort_length_8/`；原始 MTP full-loop、TIS replay 和 return-logprob smoke 已归档到 `local_artifacts/qwen38/`

### 网络与 FLA

| 对照 | actor time | token/s/GPU | 结论 |
|---|---:|---:|---|
| 污染网络正式 step 0/1/2 | 1,123.91 / 1,807.53 / 2,430.26s | 1,551 / 1,149 / 1,002 | node69 `mlx5_5` 出现 port error，不能代表模型性能 |
| 健康网络正式 step 0/1/2/3 | 591.16 / 563.33 / 609.29 / 497.99s | 3,164 / 3,383 / 3,407 / 3,346 | 修复后未复现逐 step 下降或分钟级 gap |
| 同一 113-shape train-only replay | 1,561.43 → 573.81s | 1,116.57 → 3,038.36 | 显式传播健康 HCA 后闭合网络因果分支 |
| 32-sample FLA 冷缓存 replay | 345.22 → 184.37s | 531.74 → 995.65 | 去除未被 kernel body 使用的 `NB` autotune/constexpr 特化有效，但只是独立性能 bug |

已验证结论：

- 极端长尾的主要已证根因是 launcher 漏传 HCA 白名单，导致 replay 使用不稳定 `mlx5_5`；当前所有 worker 都显式排除它
- FLA norm/conv 的无用 `NB` 特化会为新长度重复编译和 autotune，补丁有效，但不能解释所有历史运行
- 最长优先只是顺序选择：热缓存原始顺序比部分热缓存最长优先更快，不能称为修复
- FlashQLA hot varlen/auto-CP 隔离测试只有毫秒级，且历史慢 batch 未进入 gate-dependent auto-CP 分支；没有根因证据
- 将 CUDA `cu_seqlens` slice 改为 host offset 的候选在健康网络上使首 microbatch 卡住超过 160s，已完整撤回；原同步点参与当前 TP/CP/P2P 排序
- 132/145/165 等 microbatch 数来自 first-fit 动态打包；global batch 始终是 256，数量变化本身不是异常

数据、context、temperature 与 reward 的对照由[统一评测](kernelbench_eval.md)维护；非单位温度的生成退化、低/中 effort 的长度证据在 `local_artifacts/qwen38/reasoning_effort_overlimit_20260824/` 和 `local_artifacts/qwen38/ctx24_dynamic_filter_20260825/analysis.md`

## FP8 train 的未验证边界

当前没有本模型 FP8 actor 的完整通过记录。后续如评估 TE FP8，应在 distributed GDN 上重新检查实际模块覆盖、相同 token 的 logprob/DPPO 指标、长样本显存、optimizer save/resume 和在线同步。replicated GDN 的 69.6%/45% 参数覆盖估算、TP2/PP4 计算上界和其他模型的 FP8 吞吐不再作为本模型的建议配置或容量承诺
