# V4 Train-Time Estimate and Qwen Calibration

## Engineering Conclusion

For the current DeepSeek-V4-Flash LoRA port, use a **train forward/backward-only** estimate of:

| Case | Step time at 256 examples, 12K tokens/example | 300 train steps |
| --- | ---: | ---: |
| Tuned MoE path, no activation recompute | 11.3-15.3 min/step | 56-77 h (2.3-3.2 days) |
| Tuned MoE path, full activation recompute | 17.0-23.0 min/step | 85-115 h (3.5-4.8 days) |
| Early MoE path, no activation recompute | 15.3-26.0 min/step | 77-130 h (3.2-5.4 days) |
| Early MoE path, full activation recompute | 23.0-39.0 min/step | 115-195 h (4.8-8.1 days) |

These numbers exclude rollout, reward/environment time, weight sync, checkpoint save, cold start, and failure recovery. They are based on the validated 16-actor-GPU baseline from the Qwen3.6-27B run and current V4 PP2/EP8 train-only evidence. The front table uses the same numbers as the detailed MoE table below; avoid mixing these exact values with separately rounded planning bands. If we later move V4 actor training to 24 GPUs, do not linearly accept the estimate until the new PP/EP/DP layout has a train-only timing run; the first-order ideal scaling factor would be `16/24`, but MoE all-to-all and pipeline layout can erase part of that gain.

The previous 5-8 min/step estimate was too optimistic because it treated MoE like dense compute and did not discount effective throughput for expert all-to-all, small expert GEMMs, token permutation, router work, and launch/host overhead. Community and vendor guidance consistently treats MoE communication and compute efficiency as first-order bottlenecks, not small corrections.

## What This Estimate Covers

This handoff only estimates the actor training forward/backward section. It does not estimate the full RL loop wall clock.

The batch assumption is the same global batch size as `examples/kernel_agent/run.t1.qwen3.6.27B.fasync.sh`:

| Quantity | Value |
| --- | ---: |
| Global batch size | 256 examples |
| Average sequence length requested for this estimate | 12,000 tokens |
| Tokens per train step | 3,072,000 tokens |

If another note interprets "12K" as binary 12,288 tokens, multiply the V4 FLOPs and wall-clock estimates by `1.024`. The rounded planning bands do not materially change.

The FLOP model is a parameter-matmul model. It includes active transformer matmul weights and LM head matmul. It intentionally does not claim to model every quadratic attention score, norm, routing, permutation, optimizer, or framework overhead term. Those effects are handled by calibration against the prior Qwen run and by the MoE efficiency penalty below. A V4 train-only timing run remains the required final calibration.

## RL Step Multiplier

The FLOP tables below estimate `perf/actor_train_time`, not the whole slime RL step. In the current slime actor path, one RL step normally does:

1. a separate log-probs forward pass timed as `perf/log_probs_time`;
2. rollout-data postprocessing such as `sequence_mis`, which can zero loss masks after log-probs are already computed;
3. the optimizer train pass timed as `perf/actor_train_time`, which does one training forward/backward and optional recompute.

So the whole step is not just one forward/backward. In the Qwen3.6-27B warm steps, the measured multiplier was:

```text
median(perf/step_time / perf/actor_train_time) = 1.173
mean(perf/step_time / perf/actor_train_time)   = 1.179
```

Use `1.18x` to convert actor-train-only estimates into fixed-step RL wall clock when rollout is overlapped and not the bottleneck. For the current 3-train-node / 24-H20 layout, the tuned-MoE full-recompute line becomes:

```text
16-GPU actor train-only: 17.0-23.0 min/step
ideal 24-GPU actor train-only: 17.0-23.0 * 16/24 = 11.3-15.3 min/step
RL step wall clock with Qwen multiplier: 11.3-15.3 * 1.18 = 13.4-18.1 min/step
300 steps: 67-90.5 h = 2.8-3.8 days
```

Rejected samples need a separate interpretation. If a group is dropped by the rollout dynamic filter before train data is built, it mainly increases rollout work. If `sequence_mis` rejects a sample after log-probs, the sample remains in the training tensors with a zeroed loss mask, so it still consumes log-probs and actor-train compute. Therefore:

```text
fixed 300 optimizer steps: no extra multiplier beyond processed tokens and the 1.18x step multiplier
target fixed useful/accepted-token budget: multiply by processed_tokens / effective_tokens
```

The prior Qwen run showed severe `sequence_mis` rejection in some steps, e.g. `reject_rate=0.960` with `mis_effective_token_frac=0.0369`, and `reject_rate=0.980` with `mis_effective_token_frac=0.0180`. Those values do not make a fixed 300-step run 27-56x longer, but they do make useful-token efficiency 27-56x worse if the target is defined in accepted/effective tokens rather than optimizer steps.

## Qwen3.6-27B Calibration

Use the prior Qwen3.6-27B full-parameter training run as the local hardware/software calibration point.

Local evidence:

| Item | Path / value |
| --- | --- |
| Run script | `examples/kernel_agent/run.t1.qwen3.6.27B.fasync.sh` |
| Actor GPUs | 16 H20 GPUs (`ACTOR_NUM_NODES=2`, `ACTOR_GPUS=16`) |
| Main log | `/nfs/FM/chenshuailin/projects/kernel_agents/slime-dev-csl-2/experiments/FAsync.tvm_ffi.Qwen3.6-27B.CTX16384/logs/20260624.121733.log` |
| Warm train steps parsed | 23 |
| Median actor train time | 1443.985 s |
| Mean actor train time | 1451.147 s |
| Median trained tokens/step | 2,531,251 |
| Mean trained tokens/step | 2,521,626 |
| Median logged actor train TFLOPS | 17.60 TF/s/GPU |

The important correction is that the actual Qwen train batch was not a fixed `256 * 4096` token batch. Rollout responses were long; the parsed train data is about 2.5M tokens/step. Any estimate that uses the nominal context cap or a short average sequence will underpredict Qwen step time.

Qwen checkpoint matmul parameter counts:

| Component | Weight count |
| --- | ---: |
| Transformer layer matmuls, excluding embeddings | 24.7228B |
| LM head | 1.2714B |
| MTP projection | 0.0524B |
| Forward parameter-matmul FLOPs/token | about 52.0 GF/token |

Qwen was full-parameter training, not LoRA. For full-parameter training:

```text
no-recompute F+B ~= 3F
partial recompute extra ~= recomputed_layer_fraction * layer_forward
```

The Qwen run used full recompute for 29 of 64 layers. That gives:

```text
F ~= 52.0 GF/token
F+B+partial-recompute ~= 3 * 52.0 + (29/64) * (2 * 24.7228)
                       ~= 178 GF/token
```

With the median parsed token count:

```text
2.531M tokens/step * 178 GF/token ~= 450 PF/step
450 PF / 1444 s ~= 0.312 PF/s cluster effective
```

The run log's median `actor_train_tflops` is 17.60 TF/s/GPU. Multiplying by 16 actor GPUs gives 0.282 PF/s cluster effective. The FLOP-model-derived 0.312 PF/s is close enough for an engineering estimate, so the calibrated dense-like effective throughput on this stack is:

```text
0.28-0.31 PF/s for 16 H20 actor GPUs
```

## LoRA Backward Correction

Do not use `backward = 2 * forward` for frozen-base LoRA as a blanket rule.

For full-parameter training, backward is roughly two forward-equivalent matmul passes: one for activation gradients and one for weight gradients. Therefore full-parameter no-recompute F+B is about `3F`.

For LoRA with frozen base weights, the expensive base weight-gradient pass is absent. The backward pass still needs activation gradients through the frozen base, plus small adapter gradients. Therefore LoRA no-recompute F+B is closer to:

```text
LoRA no-recompute F+B ~= 2F
LoRA full-recompute F+B planning bound ~= 3F
```

Muon affects the optimizer/update section, but this handoff excludes optimizer time and focuses only on actor train forward/backward.

## V4 Active FLOP Model

V4 HF config and safetensor metadata:

| Item | Value |
| --- | ---: |
| Layers | 43 |
| Hidden size | 4096 |
| Routed experts | 256 |
| Routed experts per token | 6 |
| Shared experts | 1 |
| Expert intermediate size | 2048 |
| Active matmul weights/token | about 12.705B |
| Non-expert active weights | about 6.212B |
| Routed active weights/token | about 6.493B |

The V4 parameter-matmul forward estimate is:

```text
F ~= 2 * 12.705B = 25.41 GF/token
LoRA no-recompute F+B ~= 50.82 GF/token
LoRA full-recompute F+B planning bound ~= 76.23 GF/token
```

The full-recompute line is a conservative planning bound. A stricter formula would be `2 * F_total + F_recomputed_transformer`, excluding the LM head and any non-recomputed terminal projection from the recompute pass. Excluding only the V4 LM head saves about `2 * 129,280 * 4096 ~= 1.06 GF/token`, roughly 1.4% of the 76.23 GF/token bound, so the planning bands do not move.

At 256 examples and 12,000 tokens/example:

```text
tokens/step = 256 * 12,000 = 3,072,000
no-recompute FLOPs/step ~= 3.072M * 50.82G = 156 PF
full-recompute FLOPs/step ~= 3.072M * 76.23G = 234 PF
```

If V4 behaved like the calibrated Qwen dense-like path, the train-only bound would be:

| Case | Dense-like 16-GPU estimate |
| --- | ---: |
| LoRA no recompute | 8.4-9.3 min/step |
| LoRA full recompute | 12.6-13.9 min/step |

That dense-like bound is not the expected runtime. It is only the lower reference before applying the MoE efficiency penalty.

The Qwen-to-V4 transfer is not architecture-exact. The calibrated `0.28-0.31 PF/s` is parameter-model FLOPs divided by Qwen wall time, so it absorbs Qwen attention, host, and framework overhead into the denominator. V4-Flash has different attention/MLA structure and sparse indexing at a different average context length; this can make the dense-like baseline slightly pessimistic or otherwise non-comparable. The final calibration still has to come from a V4 train-only timing run.

## MoE Efficiency Penalty

MoE should be estimated with lower effective throughput than dense because sparsity lowers per-token compute relative to model size and exposes communication/dispatch overhead.

External evidence:

| Source | Relevant point |
| --- | --- |
| NVIDIA Megatron Core MoE docs | Megatron recommends minimizing TP/EP/PP because model parallelism adds communication overhead; EP and TP should stay within a node when possible because they are communication-intensive; for large MoE such as DeepSeek-V3, EP communication may exceed NVLink bandwidth and should use all-to-all overlap. |
| Megatron Core MoE technical report | Before optimization, EP all-to-all typically consumes 20-60% of training time depending on model, EP size, and topology. DeepEP/HybridEP plus FWD-BWD overlap can reduce this to under 10% in DeepSeek-V3 training, but only in an optimized stack. The same report calls out small kernels, host overhead, grouped GEMM, fusion, CUDA Graphs, and Sync-Free MoE as necessary to address compute inefficiency. |
| DeepSeek-V3 technical report | DeepSeek-V3 is a sparse MoE with 671B total parameters and 37B active parameters/token; its training design includes node-limited routing and communication overlap. DeepSeek explicitly states communication bandwidth is a critical MoE bottleneck and that its overlap implementation consumes 20 of 132 H800 SMs for communication, limiting compute throughput. |
| Tutel MLSys 2023 | Tutel reports large speedups from adaptive all-to-all, hierarchical all-to-all, and fast encode/decode. That is evidence that MoE system overhead can dominate enough that specialized dispatch and layout work materially changes wall clock. |

Use two throughput bands for planning:

| Band | Effective throughput vs calibrated dense-like path | 16-GPU cluster throughput |
| --- | ---: | ---: |
| Tuned MoE | 60-75% | 0.17-0.23 PF/s |
| Early MoE | 35-55% | 0.10-0.17 PF/s |

The tuned band assumes DeepEP or equivalent dispatch, grouped expert GEMM, reasonable token balance, no pathological cross-node EP traffic, and enough overlap that all-to-all is no longer the dominant wall-clock term. The early band is the safer estimate before V4 forward/backward profiling proves those conditions.

Applying those bands to the V4 LoRA FLOPs:

| Case | FLOPs/step | Tuned MoE | Early MoE |
| --- | ---: | ---: | ---: |
| No activation recompute | 156 PF | 11.3-15.3 min/step | 15.3-26.0 min/step |
| Full activation recompute | 234 PF | 17.0-23.0 min/step | 23.0-39.0 min/step |

For 300 train steps:

| Case | Tuned MoE | Early MoE |
| --- | ---: | ---: |
| No activation recompute | 56-77 h | 77-130 h |
| Full activation recompute | 85-115 h | 115-195 h |

## Validation Plan

The next train-only timing milestone should measure V4 forward/backward directly before entering full rollout:

1. Run a debug train-only replay with the same global batch size and a controlled 12K-token average.
2. Log actual tokens/step, actor train time, actor train TFLOPS, MoE all-to-all time, expert GEMM time, token permutation time, and recompute mode.
3. Compare measured step time to the table above and update this handoff with the measured MoE throughput multiplier.
4. If V4 is outside the early band, treat it as a performance bug until profiling explains the gap.

## References

- NVIDIA Megatron Core MoE docs: <https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html>
- Megatron Core MoE technical report: <https://arxiv.org/html/2603.07685v1>
- DeepSeek-V3 technical report: <https://arxiv.org/html/2412.19437v1>
- Tutel MLSys 2023 abstract: <https://proceedings.mlsys.org/paper_files/paper/2023/hash/5616d34cf8ff73942cfd5aa922842556-Abstract-mlsys2023.html>
