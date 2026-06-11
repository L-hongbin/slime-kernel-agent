# 训练 RL Step 效率结论与下一步

<!-- > 面向同事快速接手：先读「关键结论」。后面的表格是支撑证据；
> 更细的配置、源码路径和历史 A/B 放在「技术注释」里，保留但不打断主线。 -->

## 关键结论

<!-- 当前权威数据来自 bf16 主 run： -->
**数据来源**：[`checkpoints/Qwen3.6-27B/20260609_134303.t1.27B.bf16.TP4.PP2.CP2.tis.eagle.colocate.offload.ctx16384.gradf32.H20/run.log`](https://wandb.ai/shuailin_chen/slime/runs/ttl1ctxe?nw=nwuser1106982578)

**训练配置**：TP4 x PP2 x CP2 x DP2，colocate + `--optimizer-cpu-offload`

<!-- 这是 `round_robin` router 下的 41 个完整 train step（step 0..40），比旧 FP8 step-0 分析可靠得多。 -->
**主要结论**：
<!-- 1. **稳态非 save step 约 15.7 分钟。**
   非 save step 的 `step_time` median/mean 为 **943s / 953s** -->

1. **当前主要瓶颈是两个大块：actor train 和 rollout。**
   - actor train：**约 498s（8.3 分钟）**，现在是最大单项
   - rollout：**约 390s（6.5 分钟）**，是 `train_wait_time` 的主体

<!-- 2. **checkpoint save 是新的周期性尾巴。**
   `--save-interval 10` 的 save step（10/20/30/40）会多出约 **384s** 的 `save_model_time`，
   并计入 `train_wait_time`。save step 的 `step_time` median/mean 为 **1315s / 1385s** -->

2. **offload / 权重同步不是大头。**
   - sleep/offload + update_weights + wake_up + data_preprocess 合计约 **22s**，只占 step_time 的 **约 2.3%**
   - 因此，async 模式的收益更多来自于 rollout 和 train 的 overlap，而不是 offload/weightsync开销的减少

<!-- 5. **router 继续固定用 `round_robin`。**
   历史 FP8 A/B 已显示 `consistent_hashing` 会放大 backlog、502、disconnect 和 drain；
   本轮 bf16 `round_robin` 全程 0 个 502 / router server error / disconnect。 -->

## 单步时间分解

`perf/step_time = perf/train_wait_time + perf/train_time`。

`train_wait_time` 包含 rollout 等待、abort/drain、权重同步、rollout engine offload、
train model wake-up、data_preprocess，以及 save step 上的 checkpoint save 阻塞。

| 指标 | 全部 41 step median / mean | 非 save step 37 个 median / mean | save step 4 个 median / mean | 结论 |
| --- | ---: | ---: | ---: | --- |
| **step_time** | **949s / 995s** | **943s / 953s** | **1315s / 1385s** | 端到端 RL step |
| train_wait_time | 430s / 495s | 420s / 453s | 827s / 885s | rollout 是主体；save step 含 ckpt 阻塞 |
| rollout_time | 390s / 425s | 390s / 421s | 410s / 465s | rollout perf 可按 step 对齐；save step 10 rollout 长尾明显 |
| actor_train_time | **498s / 500s** | **498s / 500s** | 498s / 499s | 最大单项 |
| actor_train_tflops | 30.5 / 30.4 | 30.5 / 30.4 | 30.4 / 30.5 | 稳态但偏低 |
| actor_train_tok_per_s | 5929 / 5911 | 5929 / 5910 | 5918 / 5917 | 稳态 |
| save_model_time | - | 0 | **385s / 384s** | 每 10 step 周期性尾巴 |

## Checkpoint Save 尾巴

<!-- `--save-interval 10` 的 save step 被 checkpoint 明显拉高。`rollout_time` 本身在这些 step 也有日志；
上一版表里写 `-` 只是没有按 save/non-save 拆开，不是没有这个指标。 -->

| step | rollout_time | step_time | train_wait_time | save_model_time |
| ---: | ---: | ---: | ---: | ---: |
| 10 | 664s | 1636s | 1089s | 385s |
| 20 | 430s | 1301s | 846s | 382s |
| 30 | 377s | 1273s | 797s | 384s |
| 40 | 391s | 1328s | 808s | 386s |

<!-- 判断：`--async-save` 名义上异步，但 `save_model_time ~= 384s` 仍然计入 save step 的
`train_wait_time`。这说明主循环仍在某个阶段等待 checkpoint 相关工作完成。

step 10 尤其差：`train_wait_time=1089s`，除 save 本身外，abort 后还多等约 119s。
这更像首个 save 与异步落盘/上一轮后台任务叠加，不应归因给常规 abort/drain。 -->

## Offload / 权重同步

非 save step 上可明确归因的切换与同步开销很小：

| 项 | median 时间 | 占典型 step_time（约 949s） |
| --- | ---: | ---: |
| sleep/offload | 3.8s | 0.40% |
| update_weights | 16.1s | 1.71% |
| wake_up | 1.7s | 0.18% |
| data_preprocess | 0.4s | 0.04% |
| **合计** | **约 22s** | **约 2.3%** |

判断：这条线即使完全优化掉，收益也明显小于 actor train、rollout 和 checkpoint save。

**因此，async模式的收益更多来自于rollout和train的overlap，而不是offload/weightsync开销的减少。**

## Rollout 状态

41 个 rollout 的稳态指标：

| 指标 | median | mean | min / max |
| --- | ---: | ---: | ---: |
| rollout_time | 390.3s | 424.9s | 266.9 / 699.9 |
| tokens_per_gpu_per_sec | 209.6 | 203.4 | 116.2 / 290.2 |
| longest_sample_tokens_per_sec | 38.4 | 37.7 | 21.5 / 56.4 |
| response_len mean | 10205 | 10170 | 9106 / 11671 |
| response_len median | 10221 | 10069 | 8918 / 11783 |
| response_len max | 15024 | 15026 | 14881 / 15143 |
| truncated_ratio | 3% | 约 4% | 1% / 19% |
| prefix_cache_hit_rate | 0.57 | 0.56 | 0.34 / 0.71 |
| avg_cached_tokens_per_sample | 766.6 | 759.9 | 460.8 / 963.0 |
| spec_accept_length | 3.24 | 3.24 | 3.15 / 3.31 |
| spec_accept_rate | 0.75 | 0.75 | 0.72 / 0.77 |

<!-- Router 健康度：

| 事件 / 队列 | bf16 主 run |
| --- | ---: |
| `502 Bad Gateway` | 0 |
| router `server error` | 0 |
| `circuit_breaker` warning | 0 |
| `Request is disconnected` | 0 |
| max `queue-req` | 25 |
| max `pending-token` | 12123 | -->

<!-- 判断：`round_robin` 下 router 健康。`run.log` 中没有搜到实际 `circuit_breaker` warning；
只出现了 router 参数里的 `disable_circuit_breaker=False`。rollout 仍有长尾（max 700s），但当前主要不是 router 错误导致。 -->

## 训练侧状态

| 指标 | 稳态 median / mean |
| --- | ---: |
| actor_train_time | 498s / 500s |
| train_time | 498s / 500s |
| actor_train_tflops | 30.5 / 30.4 |
| actor_train_tok_per_s | 5929 / 5911 |
| save_model_time（save step） | 约 384s（每 10 step） |

<!-- actor train 现在稳定占 8.3 分钟。step 0 的 581s 是 warmup outlier，不应代表稳态。

日志中还有 26 次：

```text
recompute_method == 'block' is not supported for MTP yet. Skipping recompute.
```

这表示 MTP 模块跳过 block recompute；普通 transformer 层仍按 `block` 重算。
当前 125 microbatches、`max-tokens-per-gpu 8192`、`recompute full block 25 层` 偏保守。
要提升训练吞吐，应该先测 actor train 峰值显存，再决定能否减少 recompute 或增大 token batch。

注意：日志里的 `Memory-Usage before/after offload/wake_up/update_weights` 是边界瞬时值，不是训练峰值。
测 actor train 峰值应外部用 `nvidia-smi -l 1` 对齐 actor_train 窗口，或在训练循环里打
`torch.cuda.max_memory_allocated()`。 -->

## 技术注释

<details>
<summary>主 run 配置</summary>

- 启动脚本：`scripts/train_drkernel/t1.27b.bf16.tis.tp4.cp2.pp2.eagle.colocate.offload.gradf32.sh`
- 运行区间：`2026-06-09 13:43:39` 到 `2026-06-10 01:27:08`，约 11h43m
- 模型 / 精度：Qwen3.6-27B，bf16 权重，`--accumulate-allreduce-grads-in-fp32`
- MTP：`--enable-mtp-training`、`--mtp-num-layers 1`、`--mtp-loss-scaling-factor 0.2`
- 解码：EAGLE 投机解码
- 算法：`--advantage-estimator rloo` + `--use-tis`
- 硬件 / 拓扑：4 节点 H20，TP4 x PP2 x CP2 x DP2，colocate + `--optimizer-cpu-offload`
- 训练批次：`rollout_batch_size=16`、`n_samples_per_prompt=16`、`global_batch_size=256`、`over_sampling_batch_size=32`
- 上下文：`--rollout-max-context-len 16384`，prompt/response 上限各 16383
- Rollout：8 个 SGLang engine，每 engine 4 GPU，`SGLANG_MAX_RUNNING_REQUESTS=32`
- Router：`--router-policy round_robin`
- 重计算：`--recompute-granularity full --recompute-method block --recompute-num-layers 25`
- PP layout：`--decoder-last-pipeline-num-layers 31`
- Token batch：`--max-tokens-per-gpu 8192`、`--log-probs-max-tokens-per-gpu 16384`
- Checkpoint：`--save-interval 10` + `--async-save`

复核工具：

```bash
python tools/summarize_run_perf.py checkpoints/Qwen3.6-27B/20260609_134303.t1.27B.bf16.TP4.PP2.CP2.tis.eagle.colocate.offload.ctx16384.gradf32.H20/run.log
```

</details>

<details>
<summary>为什么 rollout_time 不是进度条 100% 的时间</summary>

`perf/rollout_time` 不只到进度条 100%。它包含 `Finish rollout` 之后等待 pending generate tasks 结束的时间。
因此旧 FP8 run 中 `Finish rollout -> rollout perf` 的长 gap 会被统计进 rollout 侧等待。

源码上的 drain 点仍在 `slime/rollout/sglang_rollout.py::abort()`：
`while state.pendings` 等所有 pending `generate_and_rm_group` 完成。

本轮 bf16 run 只是 pending 长尾很短，所以这条路径不再是瓶颈。若日后 response 又变长或 truncated_ratio 又升高，
这段可能重新出现，届时再考虑 partial rollout 或给 `abort()` 加 pending/done/分段日志。

</details>

<details>
<summary>consistent_hashing 的历史 A/B 结论</summary>

这部分来自更早的 FP8 run，只用于解释为什么训练固定用 `round_robin`。

| 指标 | consistent_hashing rollout 0 | round_robin rollout 0 |
| --- | ---: | ---: |
| rollout_time | 860.6s | **458.0s** |
| tokens_per_gpu_per_sec | 104.6 | **204.6** |
| prefix_cache_hit_rate | **0.866** | 0.669 |
| `502 Bad Gateway` | 9 | **0** |
| `circuit_breaker` warning | 146 | **2** |
| `Request is disconnected` | 178 | **0** |
| max `queue-req` | 97 | **25** |
| max `pending-token` | 130037 | **5477** |

`consistent_hashing` 慢的根因不是随机 hash 不均匀。256 个随机 UUID 均匀打到 8 个 worker，
单 worker 期望约 32、标准差约 5.3，不该自然到 90+。

更可能的根因是它不按实时负载改路由：某 key 一旦映射到某 worker，router 不会因为该 worker 队列高、
pending token 高、长样本堆积或 circuit breaker 抖动而转走请求，于是局部 backlog 放大为
circuit breaker / 502 / disconnect / drain。

源码路径：`slime/rollout/sglang_rollout.py` 给每个 sample 分配唯一 UUID `session_id`，
仅在 `router_policy == "consistent_hashing"` 时作为 `X-SMG-Routing-Key` 传给 router。

DrKernel 是大批单轮长生成，不是真正的稳定多轮 session，因此亲和路由反而有害。
除非 workload 变成真正多轮 agent 且需要稳定 session prefix cache，否则不要再用 `consistent_hashing`。

</details>

<details>
<summary>训练效果指标只作健康度参考</summary>

41 个 rollout 的 kernel 评测每 step `num_evaluated=256`。

稳态均值：

- correctness：约 0.52
- fast@1：约 0.21
- fast@1.5：约 0.09
- fast@2：约 0.04
- speedup_mean：约 1.22

这些是训练效果指标，不是效率瓶颈指标。

</details>
