# TVM-FFI Reward 侧加速：Level 1 / Level 2 耗时基线

## 结论

TVM-FFI reward 的主要可见长尾来自 correct 样本的 performance 测量，而不是 correctness 校验：

- **Level 1 correct 样本**：整次 reward `18.84s` mean，其中 candidate worker `14.28s`；worker 内 correctness trials 只有 `0.61s`，100 次正式计时的 candidate CUDA 执行为 `5.32s`，另有 `8.36s` 无法由旧 dump 继续拆分。
- **Level 2 correct 样本**：整次 reward `48.07s` mean，其中 candidate worker `29.84s`；worker 内 correctness trials 只有 `1.42s`，100 次正式计时的 candidate CUDA 执行为 **`22.31s`**，未分解 worker 时间为 `6.11s`。
- candidate worker 之外还有 reference timing、服务端排队、workflow/API 和轮询。这个部分在 Level 2 correct 样本中为 `18.23s` mean，不能简单称为“排队时间”。

现有 dump 可以证明 Level 2 应优先优化 performance，但不能支持“其余时间主要是编译”的结论。后续必须用精确 stage timer 重新回放固定 response，才能继续判断 compile、load、warmup 和 profiling 各自的收益。

## 统计口径

数据来自同配置的 2026-06-21 TVM-FFI Qwen3.6-27B Level 1 / Level 2 评测，correctness 使用 TF32-off、5 trials，performance 使用 100 trials。

字段含义：

- `env_time`：客户端看到的整次 reward 墙钟，包括本地 precheck、服务端 workflow、candidate kernel task、必要时单独执行的 reference timing、排队和轮询。
- `kg_kernel_total_s`：candidate kernel task 的 worker 内时间。它不包含后续独立的 reference timing task。
- correctness trials：`sum(correctness_trial_s)`，是 5 次数值正确性校验的实测墙钟。
- 100 次正式计时的 candidate CUDA 执行：`num_perf_trials × kernel_runtime`。`kernel_runtime` 是单次 candidate forward 的 CUDA-event 均值；这里不能加 `reference_runtime`，因为 reference timing 是独立 task。
- worker 外时间：`env_time - kg_kernel_total_s`。它包含 reference timing、queue、workflow/API 和轮询，现有 dump 无法在这几项之间精确拆分。

上述 CUDA 执行时间只是完整 performance 阶段的下界。完整阶段还包含 performance 输入准备、3 次 warmup、每次 trial 的 Python 循环、CUDA event 创建/record、kernel launch、`torch.cuda.synchronize`，以及额外最多 10 次 profiler iteration 和 profiler 处理。旧 dump 只保留了 `kernel_runtime`，没有保留完整 performance wall timer。

## “未分解 candidate-worker 时间”是什么

它不是 KernelGym 输出的阶段指标，而是为了让 candidate worker 时间对账而计算的减法项：

```text
未分解 candidate-worker 时间
  = kg_kernel_total_s
  - sum(correctness_trial_s)
  - num_perf_trials × kernel_runtime
```

按评测 pipeline 的执行路径，它混合了：

1. reference 源码加载、初始化输入生成与 reference model 构造；这些用于 correctness，但不是独立 reference performance task。
2. TVM-FFI candidate 编译、动态库加载、session 打开和 candidate model 构造。
3. correctness trial 计时窗口之外的 seed、输入和 CUDA 同步开销，以及 decoy/coverage 检查。
4. performance 前的输入准备和 CUDA 同步、3 次 warmup。
5. 100 次计时循环中的 Python/CUDA event 创建、kernel launch 和 synchronize 墙钟开销；`kernel_runtime` 只覆盖 CUDA-event 区间。
6. 额外最多 10 次 profiler iteration、torch profiler 本身以及 coverage 计算。

因此这个值只能叫“未分解 worker 时间”，不能叫编译时间，也不能据此直接选择编译优化方案。本次历史 dump 没有保存这些子阶段的 exact timer；需要重新做 reward-only 回放才能拆开。

## Level 1

### Compiled 样本的 candidate worker

| 指标 | mean / median / p90 |
|---|---:|
| candidate worker 总时间 | 9.15 / 3.72 / 17.07s |
| correctness trials | 0.69 / 0.43 / 0.95s |
| 100 次正式计时的 candidate CUDA 执行 | 3.26 / 0.55 / 4.36s |
| 未分解 candidate-worker 时间 | 5.19 / 2.32 / 13.80s |

按 worker 时间占比，correctness 为 `7.6%`，已观测的正式计时 CUDA 执行部分为 `35.7%`，未分解部分为 `56.8%`。真实 performance 占比高于 `35.7%`，因为 warmup、profiling 和计时循环开销仍在未分解部分。各占比是跨样本求和后得到的，不应从 mean/median 直接相除。

### Correct 样本的端到端分解

| 指标 | mean / median / p90 |
|---|---:|
| 客户端整次 reward | **18.84 / 15.15 / 33.14s** |
| worker 外：reference timing + queue/workflow/API | 4.55 / 2.63 / 11.29s |
| candidate worker 总时间 | 14.28 / 12.04 / 19.98s |
| ├─ correctness trials | 0.61 / 0.47 / 0.92s |
| ├─ 100 次正式计时的 candidate CUDA 执行 | 5.32 / 0.99 / 9.48s |
| └─ 未分解 candidate-worker 时间 | 8.36 / 8.00 / 14.97s |

Mean 可以对账为 `18.84 ≈ 4.55 + 14.28`，以及 `14.28 ≈ 0.61 + 5.32 + 8.36`。Median 和 p90 对应的未必是同一个样本，不能逐列相加。

## Level 2

### Compiled 样本的 candidate worker

| 指标 | mean / median / p90 |
|---|---:|
| candidate worker 总时间 | 8.28 / 1.24 / 15.72s |
| correctness trials | 0.85 / 0.48 / 1.50s |
| 100 次正式计时的 candidate CUDA 执行 | 5.29 / 0.00 / 8.24s |
| 未分解 candidate-worker 时间 | 2.14 / 0.86 / 5.29s |

正式计时 CUDA 执行的中位数为 0，是因为 incorrect 样本不进入 performance。按 worker 时间占比，correctness 为 `10.2%`，已观测的正式计时 CUDA 执行部分为 **`63.9%`**，未分解部分为 `25.8%`。因此真实 performance 占比至少为 `63.9%`。

### Correct 样本的端到端分解

| 指标 | mean / median / p90 |
|---|---:|
| 客户端整次 reward | **48.07 / 14.91 / 128.76s** |
| worker 外：reference timing + queue/workflow/API | 18.23 / 2.87 / 20.80s |
| candidate worker 总时间 | 29.84 / 10.02 / 69.55s |
| ├─ correctness trials | **1.42 / 0.62 / 3.06s** |
| ├─ 100 次正式计时的 candidate CUDA 执行 | **22.31 / 5.42 / 56.51s** |
| └─ 未分解 candidate-worker 时间 | 6.11 / 3.43 / 12.79s |

Mean 可以对账为 `48.07 ≈ 18.23 + 29.84`，以及 `29.84 ≈ 1.42 + 22.31 + 6.11`。这正是“correct 样本整次调用 48.07s”和“correctness 阶段 1.42s”差距悬殊的原因：两者分别是端到端时间和其中一个很小的阶段。

Reference cache 也明显影响 worker 外时间：

| correct 样本的 worker 外时间 | Level 1 mean / median / p90 | Level 2 mean / median / p90 |
|---|---:|---:|
| reference cache hit | 3.30 / 2.57 / 4.68s | 8.13 / 2.53 / 4.88s |
| reference cache miss | 11.71 / 12.03 / 19.83s | **41.86 / 4.92 / 190.67s** |

所以 `env_time - kg_kernel_total_s` 不能全部归因于 queue；cache miss 时独立 reference timing 是重要组成部分。

## 加速优先级

1. **P0：用现有精确 timer 做 reward-only 回放。** 固定 L1/L2 response，直接记录 reference timing、compile、load/session、model/input setup、correctness、warmup、100 trials、profiling 和 queue。旧 dump 的减法项只用于确定边界，不再用于猜根因。
2. **P0（Level 2）：验证 performance trials 100→25 或自适应 early-stop。** 只按已观测的 100 次 CUDA 执行部分线性下降推演，worker 容量约提高 L1 `1.37×`、L2 `1.92×`；这是不计 Python/event 循环随 trials 一起减少的保守估计。实验必须同时比较 runtime 方差、reward 和 fast@ 阈值判定。
3. **P1：保证 reference cache 命中。** 对同一题的多 sample / 多 turn 复用 reference runtime，避免 cache miss 的独立 reference timing 长尾。
4. **P1：再决定是否拆 worker 池或增加 worker。** 先用精确 queue timer 证明 head-of-line blocking，不能再用 `env_time - kg_kernel_total_s` 直接代替 queue。

## 证据与复现

- 统计脚本：`tools/analyze_reward_timing.py`
- L1 dump：`experiments/Eval.TVMFFI.Qwen3.6-27B.Qwen3.6-27B.ctx32768.resp32768.turn3.n8/Qwen3.6-27B.level1.kg21_tf32off_correctness.20260621.093912/dumps/rollout_data/eval_0.pt`
- L2 dump：同目录下 `Qwen3.6-27B.level2.kg21_tf32off_correctness.20260621.093912/dumps/rollout_data/eval_0.pt`

复算需要能导入 torch 的环境；当前可在 `.22` 的本仓库执行：

```bash
base=$PWD/experiments/Eval.TVMFFI.Qwen3.6-27B.Qwen3.6-27B.ctx32768.resp32768.turn3.n8
python3 tools/analyze_reward_timing.py \
  --run "Level-1=$base/Qwen3.6-27B.level1.kg21_tf32off_correctness.20260621.093912/dumps/rollout_data/eval_0.pt" \
  --run "Level-2=$base/Qwen3.6-27B.level2.kg21_tf32off_correctness.20260621.093912/dumps/rollout_data/eval_0.pt"
```
