# Speedup survey: Qwen3.6-27B hybrid on A800-80GB (2026-05-27)

Companion to `handoffs/in_progress/handoff_drkernel_w8a8_rollout.md`. Captures
a research-agent literature scan + a ranked decision matrix for techniques we
have NOT yet tried or that are not already covered by a dedicated completed
handoff. Anchor: current production wall 1:20:33 BF16 / 1:15:21
W8A8 MLP-only = **1.07× wall** (per-token tok/s ~1.13× in slime, 1.52× in pure
sglang ignore_eos).

## Anchored facts (don't re-derive)

- **Hardware**: 8× NVIDIA A800-80GB. SM 8.0 (Ampere). BF16 312 TFLOPS, INT8
  624 TOPS. **No FP8 tensor cores** (FP8 needs SM 8.9/H100). NVSwitch-attached
  in the 8-GPU box; TP all-reduce ~5–15 μs at our message sizes.
- **Model**: Qwen3.6-27B hybrid. 64 layers = 16 full_attention + 48
  linear_attention (Gated DeltaNet / Mamba). MLP 70 % of Linear bytes
  (17.11 B params), linear_attn 23 %, self_attn 7 %. SSM scan runs in fp32
  per config (`mamba_ssm_dtype: float32`), unaffected by weight quant.
- **Serving**: sglang 0.5.12.post1, TP=2, flashinfer attention, ctx=65536,
  max-running-requests=64. Practical DP×TP=4×2 across the 8-GPU box (4
  engines).
- **Workload**: RL rollout via slime. 100 prompts × 8 samples × 3 turns.
  Mean response ~6 K tokens. Real reward (KernelGym) takes seconds-to-minutes
  per call. **Reward + idle = ~47 % of production wall** (mock-KG
  decomposition, see W8A8_ROLLOUT.md §Efficiency).

## Ranked recommendations

Ranked by **ROI × low implementation cost × low risk** for this exact setup.

### Tier 1 — first to try

These came from `docs/en/advanced/sglang-config.md`, the short PD
disaggregation note, and the fully-async rollout example. They are mostly
orthogonal to W8A8/spec-decoding: they change how requests are scheduled and how GPUs are
partitioned, not the kernel math itself.

#### 1. PD disaggregation / heterogeneous SGLang server groups

`--sglang-config` can launch different server groups for `prefill` and
`decode`, with per-group GPU counts, TP sizes, and overrides. The docs
explicitly recommend PD disaggregation for multi-turn/agentic RL because
prefill is compute-heavy while decode is memory-bandwidth-heavy; one fixed
regular engine shape is often not optimal for both.

For this run, the conservative interpretation is **experiment, not guaranteed
win**. We are already at regular DP×TP=4×2 across the 8-GPU box, so a same-8GPU
PD layout must trade off fewer regular decode engines against better
prefill/decode specialization. The main cases where it can still help:

- prompts or turn histories get longer, making prefill/KV growth a bigger
  fraction of wall;
- scaling beyond 8 rollout GPUs, where prefill and decode can be sized
  independently instead of cloning the same TP=2 engine;
- current all-at-once eval floods SGLang with long multi-turn requests, so
  decode engines stay occupied while prefill work queues behind them.

Candidate smoke configs:

```yaml
sglang:
  - name: actor
    model_path: /path/to/Qwen3.6-27B-W8A8
    server_groups:
      - worker_type: prefill
        num_gpus: 2
        num_gpus_per_engine: 2
        overrides:
          chunked_prefill_size: 4096
      - worker_type: decode
        num_gpus: 6
        num_gpus_per_engine: 2
        overrides:
          context_length: 65536
          max_running_requests: 64
```

Also test the inverse if decode remains dominant: keep all 8 GPUs as regular
TP=2 engines and only move to PD when extra rollout GPUs are available.

**Estimated gain**: same-8GPU likely **0.95–1.15× wall** depending on prompt
length and queueing; with extra rollout GPUs, PD is a cleaner scaling path than
blindly increasing TP. Treat as a scheduling/topology smoke test.

#### 2. Fully-async rollout across train/rollout boundaries

`examples/fully_async` uses `train_async.py` plus
`slime.rollout.fully_async_rollout.generate_rollout_fully_async`. A background
worker keeps up to `args.sglang_server_concurrency * num_engines` groups in
flight and returns completed groups to training, so the next rollout does not
wait on the slowest trajectories from the previous one.

This is most useful for production training loops, not standalone eval: the
example currently says no evaluation mode. It fits DrKernel because custom
generate/RM hooks still go through the standard plug-in paths.

**Estimated gain**: can recover idle train/rollout gaps and long-tail waits;
for pure eval it is not directly applicable.

#### 3. Per-group overrides: smaller context where safe

`sglang-config` allows overrides per model/server group, so prefill/decode
groups can use different `context_length`, `mem_fraction_static`,
`chunked_prefill_size`, and CUDA-graph settings. This matters because the
recent BF16/SmoothQuant run lost KV capacity mainly from larger resident
weights and lower `mem_fraction_static`; any avoidable context reservation
directly reduces the maximum token pool.

Do not globally cut `ctx=65536` until we have a prompt+response length
histogram. But if eval/train splits have different tails, serve them with
separate configs instead of forcing the worst-case context on every request.

### Tier 2 — fallbacks if Tier 1 blocked

#### 4. INT4 W4A16 (AWQ-Marlin) for MLP only

MLP is 70 % of Linear bytes / 17.1 GB W8A8 → ~8.6 GB W4A16. Frees ~8 GB
weight memory per rank which sglang re-allocates to KV cache. Marlin
kernel holds near-ideal 3.87–4× speedup at batch 16–32 (our regime).

**Estimated gain**: **1.15–1.30× decode over current W8A8 baseline**
(combined ~1.5–1.7× over BF16). Edge narrows as batch grows.

**Implementation cost**: medium. AWQ calibration (~1 day script). sglang
already loads via compressed-tensors. Restrict to MLP-only initially —
**AWQ on hybrid Mamba's GDN projection matrices has no published
benchmark**; quality risk if we extend to non-MLP.

**Quality risk**: AWQ on Qwen3 reports 1.8 % average accuracy drop vs 2.7 %
for GPTQ-INT4 (ACL 2025 "Give me BF16" paper). Adds 1–2 pp on top of our
current 1–2 pp from Hadamard.

**Sources**:
- [AutoAWQ](https://github.com/casper-hansen/AutoAWQ)
- [Marlin paper](https://arxiv.org/abs/2408.11743)
- [Red Hat Marlin writeup](https://developers.redhat.com/articles/2024/04/17/how-marlin-pushes-boundaries-mixed-precision-llm-inference)
- [ACL 2025 "Give me BF16"](https://aclanthology.org/2025.acl-long.1304.pdf)

#### 5. FP8 KV cache (`--kv-cache-dtype fp8_e5m2`)

flashinfer supports `fp8_e5m2` KV cache on SM 7.5+, so A800-compatible.
Halves the KV cache footprint → bigger KV pool at the same
`mem-fraction-static` → more concurrent decoding.

**Estimated gain**: **1.15–1.30× decode** from larger batch + reduced
long-tail KV pressure.

**Critical caveat**: vllm bug #26646 reports FP8 KV "not supported in
this architecture" for Qwen3-Next; user tried fp8_e4m3, fp8_e5m2,
fp8_inc, all failed. **sglang status on Qwen3.5/3.6 with FP8 KV is
undocumented**. Must self-test on a smoke before assuming it works.

**Sources**:
- [FlashInfer FP8 KV docs](https://docs.flashinfer.ai/generated/flashinfer.decode.single_decode_with_kv_cache.html)
- [sglang quantized KV docs](https://docs.sglang.io/advanced_features/quantized_kv_cache.html)
- [vllm FP8 KV blog 2026-04-22](https://vllm.ai/blog/2026-04-22-fp8-kvcache)
- [vllm Qwen3-Next KV quant bug #26646](https://github.com/vllm-project/vllm/issues/26646)

### Tier 3 — research-grade / high cost

#### 6. FlashQLA GDN kernels (Alibaba, 2026-04)

TileLang-based replacement for fla-org/flash-linear-attention's Triton
kernels on Qwen3-Next/Qwen3.5/Qwen3.6. Reports 2–3× forward / 2× backward
**on Hopper (SM 90+ with WGMMA/TMA)**. Ampere fallback is much closer to
the Triton baseline.

**For our workload**: Decode mamba is 7.3 % of W8A8 step time. Even 3×
decode-mamba on Hopper → 1.05× decode wall ceiling; on Ampere likely
sub-1.05×. **Plausibly no measurable wall gain on A800.**

**Sources**:
- [FlashQLA GitHub](https://github.com/QwenLM/FlashQLA)
- [Alibaba blog](https://www.alibabacloud.com/blog/603084)
- [MarkTechPost coverage](https://www.marktechpost.com/2026/04/29/qwen-team-releases-flashqla-a-high-performance-linear-attention-kernel-library-that-achieves-up-to-3x-speedup-on-nvidia-hopper-gpus/)

#### 7. W4A8 QServe / LiquidGEMM

A800 sweet spot in theory: 4-bit weights (Marlin-class bandwidth) + 8-bit
activations (INT8 tensor cores). QServe reports 2.4× over TRT-LLM on
Qwen1.5-72B on A100. LiquidGEMM 1.12–1.63× over W4A16/W8A8/FP8.

**Implementation cost**: high. sglang's compressed-tensors W4A8 support
is limited; QServe ships its own runtime. Multi-week integration. Defer.

**Sources**:
- [QServe paper](https://arxiv.org/abs/2405.04532), [omniserve code](https://github.com/mit-han-lab/omniserve)
- [LiquidGEMM paper](https://arxiv.org/abs/2509.01229)

### Tier 4 — small or risky

#### 8. sglang piecewise CUDA graph

SGLang's piecewise CUDA graph is mainly an extend/prefill optimization, while
our DrKernel rollout wall is dominated by long decode plus KernelGym reward and
queueing. For this workload, expect **0–3 % wall** unless the run is visibly
prefill/extend-bound. Prefer first ensuring ordinary CUDA graph batch coverage
with `--sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)`.

**Known correctness bug on Qwen3-Next** (sglang issue #17330: "Output
token count abnormally increases when --enable-piecewise-cuda-graph is
set"). Do not enable `--sglang-enable-piecewise-cuda-graph` on Qwen3.6 without
a token-count regression smoke and KernelBench score check.

**Source**: [sglang PCG bug #17330](https://github.com/sgl-project/sglang/issues/17330)

#### 9. Prompt-side compression

Cut KernelGym problem statements from ~5683 → ~4000 tokens. Reduces
prefill + KV pressure. **Gain modest (1.02–1.05× wall)** because
prefill is a small fraction of our wall. **Risk**: directly changes the
eval semantics; requires correctness validation.

### Crossed off (already at limit or not applicable)

| | reason |
|---|---|
| **Plain DP × TP refactor** | We're already at DP=4 × TP=2 across 8 GPUs (4 engines × 2 GPUs each); no further plain DP gain. `--sglang-config` PD disaggregation is a separate topology experiment because it changes prefill/decode allocation. |
| **Per-group W8A8** | sglang's `compressed_tensors_w8a8_int8` scheme is per-channel/per-tensor only; we explicitly hard-fail on per-group. Defer until sglang adds it. |
| **FP8 weight/activation tensor cores** | A800 = SM 8.0, no FP8 hardware. Software FP8 is slower than INT8. |
| **LayerSkip / SkipDecode** | Requires layer-dropout fine-tune step on the 27B; lower-cost SpecDec routes are tracked in `handoff_specdec_drkernel.md`. |
| **TP=4 instead of TP=2** | Our profile shows allreduce at ~5 % of W8A8 step. TP=4 doubles allreduce; net loss for decode latency. |
| **TP=1** | Model weights (54 GB BF16 or 30 GB W8A8) + scratch + KV does not fit per A800-80GB. |

## Action items / next steps

1. **Try one PD-disaggregation config**:
   start with 2 GPUs prefill TP=2 and 6 GPUs decode TP=2, then compare against
   regular DP×TP=4×2 on completed samples/min and SGLang full-token usage.
2. For production training, **evaluate fully-async rollout** with
   `train_async.py` and
   `--rollout-function-path slime.rollout.fully_async_rollout.generate_rollout_fully_async`.
   Do not use it for standalone eval yet; the example documents no eval mode.
3. If PD/async changes raise effective decode concurrency, **revisit FP8 KV
   cache** (item #5). The bigger KV pool could matter more under higher
   concurrency.

## Sources

(Inlined per-item above; this list is the cross-reference.)

| topic | URL |
|---|---|
| RollPacker | https://arxiv.org/html/2509.21009v1 |
| APRIL (slime-integrated) | https://arxiv.org/html/2509.18521v1 |
| slime sglang-config | `docs/en/advanced/sglang-config.md` |
| slime PD disaggregation | `docs/en/advanced/pd-disaggregation.md` |
| slime fully-async rollout | `examples/fully_async/README.md`, `slime/rollout/fully_async_rollout.py` |
| Marlin | https://arxiv.org/abs/2408.11743 |
| Red Hat Marlin | https://developers.redhat.com/articles/2024/04/17/how-marlin-pushes-boundaries-mixed-precision-llm-inference |
| AutoAWQ | https://github.com/casper-hansen/AutoAWQ |
| GPTQModel | https://github.com/ModelCloud/GPTQModel |
| QServe | https://arxiv.org/abs/2405.04532 |
| LiquidGEMM | https://arxiv.org/abs/2509.01229 |
| FlashQLA | https://github.com/QwenLM/FlashQLA |
| FlashInfer FP8 KV | https://docs.flashinfer.ai/generated/flashinfer.decode.single_decode_with_kv_cache.html |
| sglang quantized KV docs | https://docs.sglang.io/advanced_features/quantized_kv_cache.html |
| vllm FP8 KV blog | https://vllm.ai/blog/2026-04-22-fp8-kvcache |
| vllm Qwen3-Next KV quant bug | https://github.com/vllm-project/vllm/issues/26646 |
| sglang PCG bug on Qwen3-Next | https://github.com/sgl-project/sglang/issues/17330 |
| LayerSkip | https://arxiv.org/abs/2404.16710 |
| ACL 2025 "Give me BF16" | https://aclanthology.org/2025.acl-long.1304.pdf |
