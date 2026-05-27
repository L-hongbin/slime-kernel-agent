# Speedup survey: Qwen3.6-27B hybrid on A800-80GB (2026-05-27)

Companion to `handoffs/in_progress/HANDOFF_DRKERNEL_W8A8_ROLLOUT.md`. Captures
a research-agent literature scan + a ranked decision matrix for techniques we
have NOT yet tried. Anchor: current production wall 1:20:33 BF16 / 1:15:21
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
- **Already deployed / tried**: W8A8 INT8 RTN (1.07–1.31× wall); Hadamard
  rotation pre-processor (+1–2 pp quality); SmoothQuant non_linear_attn
  (accuracy run in progress); `--mamba-scheduler-strategy extra_buffer`
  (regression).

## Ranked recommendations

Ranked by **ROI × low implementation cost × low risk** for this exact setup.

### Tier 1 — first to try

#### 1. MTP / NEXTN speculative decoding in sglang

Qwen3.6/Qwen3.5/Qwen3-Next models ship a native multi-token-prediction
head that shares embedding + lm_head with the main model. sglang exposes
the verifier via `--speculative-algorithm NEXTN --speculative-num-steps 3
--speculative-eagle-topk N --speculative-num-draft-tokens 4`.

Crucially, the NEXTN path handles GDN/Mamba SSM-state rollback after
rejected drafts — which is what breaks naive ngram speculative decoding on
hybrid models (open vllm bug #39273: "Output token count abnormally
increases on Qwen3-Next with ngram speculative"). Co-trained MTP head =
no separate draft model to keep in sync with the policy.

**Estimated wall gain**: 1.25–1.6× pure decode (lower than published
H100 numbers because temperature sampling reduces accept rate at our RL
rollout temperature). Stacked on W8A8's 1.27× decode → **roughly
1.6–2.0× combined decode**, diluted to **~1.3–1.5× wall** by the 47 %
reward dilution.

**Implementation cost**: low. Few-hour spike to add the flags.
**Caveats** to verify on a smoke before flipping production:
- sglang's compressed-tensors loader must accept W8A8 on the MTP head
  weights. If not, MTP forward stays BF16 and verifier savings shrink.
- RL rollout uses temperature sampling (not greedy); paper claims
  "preserving identical training curves" but verify on a 20×8 smoke.

**Sources**:
- [Qwen3-Next sglang docs](https://docs.sglang.io/basic_usage/qwen3.html)
- [sglang Qwen3-Next/Qwen3.5 tracking issue #18590](https://github.com/sgl-project/sglang/issues/18590)
- [vllm ngram-on-GDN bug #39273](https://github.com/vllm-project/vllm/issues/39273)
- [sglang NEXTN PR #3582](https://github.com/sgl-project/sglang/pull/3582)
- [EAGLE-3 lmsys blog](https://www.lmsys.org/blog/2025-12-01-eagle3-vertex/)
- [Qwen3.6 MTP GB10 benchmark](https://docai.hu/en/blog/qwen36-mtp-gb10)

#### 2. Reward prefetch / inter-turn pipelining

Pipeline `reward(N) || decode(N+1)` instead of serializing them. The 47 %
non-inference wall is our largest single dilution source; even partial
overlap recovers meaningful wall.

If reward+idle is 47 % and we can hide reward behind the next turn's
decode for ~half the turns (turns 1 and 2 can prefetch; turn 3's reward
can't), recover **~30 % of the reward wall = 1.15–1.25× wall**. Stacks
multiplicatively with #1.

**Implementation cost**: low-medium. Refactor lives almost entirely
inside `slime_plugins/drkernel/rollout.py`. The slime async scaffolding
exists; need to verify whether the multi-turn drkernel rollout currently
issues `reward.await()` before next turn (likely yes — pipelining is the
fix).

**Risk**: None semantically. Changes timing only. Reproducibility seeds
behave the same.

**Sources**:
- [AReaL parallel-reward-service paper](https://arxiv.org/html/2505.24298v1)
- [ROLL Flash](https://arxiv.org/html/2510.11345v1)

### Tier 2 — fallbacks if Tier 1 blocked

#### 3. DAS suffix-tree drafter (only if MTP blocked)

Training-free spec-decode drafter from a sliding window of recent
rollouts, with length-aware budget (more drafting on long trajectories).
RL-friendly because the drafter auto-tracks policy drift; no head retrain
needed. Up to 50 % rollout-time reduction on math/code reasoning per the
paper.

**Use when**: MTP/NEXTN path doesn't work (e.g. compressed-tensors loader
rejects W8A8 on MTP head weights). Otherwise prefer #1.

**Cost**: medium-high. No sglang integration exists; ~3–5 days to wire a
suffix-tree drafter into sglang's NEXTN draft interface.

**Sources**:
- [DAS paper](https://arxiv.org/html/2511.13841)
- [Together AI blog on DAS](https://www.together.ai/blog/distribution-aware-speculative-decoding)

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

+0.5–8.2 % decode wall in published numbers. For our batch=32 regime,
likely 1.02–1.05× decode → ~1.00–1.03× wall after reward dilution.
**Known correctness bug on Qwen3-Next** (sglang issue #17330: "Output
token count abnormally increases when --enable-piecewise-cuda-graph is
set"). Do not enable on Qwen3.6 without a token-count regression smoke.

**Source**: [sglang PCG bug #17330](https://github.com/sgl-project/sglang/issues/17330)

#### 9. Prompt-side compression

Cut KernelGym problem statements from ~5683 → ~4000 tokens. Reduces
prefill + KV pressure. **Gain modest (1.02–1.05× wall)** because
prefill is a small fraction of our wall. **Risk**: directly changes the
eval semantics; requires correctness validation.

### Crossed off (already at limit or not applicable)

| | reason |
|---|---|
| **DP × TP refactor** | We're already at DP=4 × TP=2 across 8 GPUs (4 engines × 2 GPUs each); no further DP gain. |
| **Per-group W8A8** | sglang's `compressed_tensors_w8a8_int8` scheme is per-channel/per-tensor only; we explicitly hard-fail on per-group. Defer until sglang adds it. |
| **FP8 weight/activation tensor cores** | A800 = SM 8.0, no FP8 hardware. Software FP8 is slower than INT8. |
| **LayerSkip / SkipDecode** | Requires layer-dropout fine-tune step on the 27B; MTP head ships in the ckpt already (use #1 instead). |
| **TP=4 instead of TP=2** | Our profile shows allreduce at ~5 % of W8A8 step. TP=4 doubles allreduce; net loss for decode latency. |
| **TP=1** | Model weights (54 GB BF16 or 30 GB W8A8) + scratch + KV does not fit per A800-80GB. |

## Stacked expectation if Tier 1 lands

If both **#1 (MTP/NEXTN)** and **#2 (reward prefetch)** land cleanly:
**~1.5–1.8× wall over current W8A8 production** (BF16 → 1.07× → ~1.6–1.9×),
with no quality regression beyond what MTP introduces for sampled
decoding.

## Action items / next steps

1. **Verify MTP head + W8A8 ckpt compatibility** in sglang's
   compressed-tensors loader. Either load the Qwen3.6-27B-W8A8-RTN-local-mlp
   ckpt with `--speculative-algorithm NEXTN` and see if it boots, or
   read the loader source to confirm the MTP weights are NOT in the
   `ignore` list (they should NOT be, since they're not under `mlp.`,
   `self_attn.`, or `linear_attn.`).
2. **20×8 smoke** with NEXTN enabled vs the current W8A8 baseline.
   Compare:
   - Wall and per-token tok/s (decode speedup ratio)
   - Correct T3 score (quality preserved)
   - Acceptance rate from the sglang decode log
3. **Audit `slime_plugins/drkernel/rollout.py`** for the multi-turn
   loop: where the reward await blocks the next-turn generate submit.
   That's the seam to add async prefetch.
4. After #1 and #2 land, **revisit FP8 KV cache** (item #5). The bigger
   KV pool + speculative decode together push effective concurrency
   higher than either alone.

## Sources

(Inlined per-item above; this list is the cross-reference.)

| topic | URL |
|---|---|
| MTP / NEXTN on Qwen3.5+ | https://docs.sglang.io/basic_usage/qwen3.html |
| Qwen3-Next sglang tracking | https://github.com/sgl-project/sglang/issues/18590 |
| ngram-on-GDN bug | https://github.com/vllm-project/vllm/issues/39273 |
| sglang NEXTN PR | https://github.com/sgl-project/sglang/pull/3582 |
| EAGLE-3 / lmsys | https://www.lmsys.org/blog/2025-12-01-eagle3-vertex/ |
| Qwen3.6 MTP GB10 | https://docai.hu/en/blog/qwen36-mtp-gb10 |
| DAS paper | https://arxiv.org/html/2511.13841 |
| Together AI DAS blog | https://www.together.ai/blog/distribution-aware-speculative-decoding |
| AReaL | https://arxiv.org/html/2505.24298v1 |
| ROLL Flash | https://arxiv.org/html/2510.11345v1 |
| RollPacker | https://arxiv.org/html/2509.21009v1 |
| APRIL (slime-integrated) | https://arxiv.org/html/2509.18521v1 |
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
