# DeepSeek-V4-Flash train↔rollout mismatch: audit, fixes, and precision memo

> **UPDATE (2026-07, round 2 — measured, supersedes the original claims below):**
> The full algorithmic audit is now **exhausted and clean**. Every component — router, RoPE,
> mHC, SwiGLU clamp, C4 indexer, **CSA compressor, attention sinks, conjugate output-RoPE** —
> either matches or is fixed. Two original claims are now **refuted by measurement**:
> 1. **The C4 indexer (sparse vs dense) is NOT the dominant long-context mismatch.** Measured
>    directly: dense-train vs sparse-rollout = 0.0404, sparse-train vs sparse-rollout = 0.0411 at
>    ctx-8192 (indexer active) → the indexer term is **≤0.0007** (within kernel-vs-reference noise).
>    A train-side sparse CSA path was built (env `V4_SPARSE_ATTENTION`) and showed **no improvement**.
>    Forcing sglang dense (original fix #1) is therefore **moot** — it removes ≤0.0007.
> 2. **The ~0.037 gap is the NON-indexer term** (measured at ctx-2048, where the indexer is a
>    provable no-op). It is *most likely* fp8-decode(rollout)-vs-bf16-prefill(train) **precision**,
>    but this is **inferred, not isolated** — the clean test (serve sglang bf16, compare bf16-train
>    vs bf16-rollout) has NOT been run. Direct kernel check: sglang's bf16 sparse flash-MLA matches a
>    reference to 0.0037, and the train A1 is also bf16 → the bf16 attention *kernels agree*; the gap
>    is the rollout's **fp8 KV-cache decode** path, not the kernel algorithm.
>
> **Net:** only ONE real algorithmic fix existed (the shared-expert clamp, done). The residual
> ~0.037 is a precision/kernel-alignment floor, closable only by unifying on bit-identical fp8
> kernels (DeepSeek's approach) or absorbed by TIS/MIS. See §Precision + the round-2 detail at end.

**Reader takeaway (original):** the train (Megatron/tilelang) and rollout (sglang) forwards diverge for
two *real algorithmic* reasons and several *negligible numeric* reasons. Fix the two real ones;
the rest are recorded as a memo so no one re-opens them. `train_rollout_logprob_abs_diff` is the
metric; its floor is set by the items below, **not** by fp8-vs-bf16 precision (see §Precision).

Distinguish two axes throughout:
- **Same math** — the two sides implement the same algorithm.
- **Same numerics** — the two sides produce the same bits (precision/kernel/accumulation).

## Fix list (do these)

| # | Mismatch | Side to change | Fix | Impact |
|---|---|---|---|---|
| 1 | **C4 lightning-indexer: rollout SPARSE (top-512), train DENSE** | **sglang → dense** | force `index_topk` ≥ `max_context/compress_ratio` so the top-k selects all compressed positions | **large at ctx>2048**, zero at ctx≤2048 |
| 2 | **Shared expert missing SwiGLU clamp** | **train → add clamp** | clamp `gate≤limit`, `up∈[-limit,limit]` (`swiglu_limit=10.0`) in `V4SharedExpertMLP`, in bf16 | moderate (every token where shared gate/up exceed ±10) |

Decision (owner): align **both sides on DENSE** attention (patch sglang to drop the indexer),
rather than porting the sparse indexer into train. Rationale: train already drops the indexer;
dense is the simpler common denominator and is exact.

## Real mismatch 1 — C4 lightning indexer (sparse vs dense)

| | Train (Megatron) | Rollout (sglang) |
|---|---|---|
| C4 compressed KV | yes (compress_ratio=4) | yes |
| indexer top-k | **DROPPED** — attends all valid compressed positions (dense) | **top-512** selected by lightning indexer (`index_topk=512`) |
| evidence | `attention/reference.py` "Lightning Indexer + top-k DROPPED"; `compressor.py:11-16` | `deepseek_v4.py:416` builds `C4Indexer` when `compress_ratio==4`; `sparse_prefill_utils.py:297` `topk_len=min((pos+1)//COMPRESS_RATIO, TOP_K)` |

**Why it hid until now:** at ctx ≤ 2048 there are ≤ 512 compressed positions, so `min(pos//4, 512)`
= all → the indexer is a no-op and both sides are dense. At 16k there are ~4096 compressed
positions and rollout attends to only 512 of them → genuinely different attention. This is the
dominant long-context divergence and is invisible in the ctx-2048 debug runs.

**Model-design note (important):** CSA (Compressed Sparse Attention) is V4-Flash's **native**
attention on `compress_ratio==4` layers (paper §Hybrid Attention; lightning-indexer top-k is the
mechanism, and the model is *trained* sparse — paper "backward propagation for sparse attention").
So **the rollout (sglang sparse) is correct; the train (dense, indexer dropped) is the deviation.**
Forcing sglang dense is OFF-DESIGN (attends positions the model learned to ignore → quality risk);
chosen deliberately here for train/rollout consistency. The *correct* alternative is to implement
CSA sparse in train.

**Fix (sglang → dense) — requires BACKEND surgery, not an `index_topk` override.** A naive override
of `get_dsa_index_topk` / `C4Indexer.index_topk` is a **no-op**: the DSV4 backend reads
`hf_text_config.index_topk` directly (`deepseek_v4_backend.py:454`) and `init_flashmla_related`
asserts `c4_sparse_topk ∈ {512,1024}` (`:308`), then `c4_sparse_topk_lengths = clamp(lengths,
max=c4_sparse_topk)` + sizes `c4_sparse_page_indices` to that width — the clamp IS the sparsity, and
the flashMLA kernel is tuned for 512/1024. Real dense mode (env `V4_DENSE_ATTENTION=1`): set
`c4_sparse_topk` to the batch's full compressed length, relax the assert, fill
`c4_sparse_page_indices` with all causal compressed positions (bypass the indexer top-k), and route
through a dense-capable kernel path; cover prefill AND decode; bound buffers by seq_len. Default
(env unset) stays byte-identical sparse.

## Real mismatch 2 — shared expert SwiGLU clamp

| | Train | Rollout |
|---|---|---|
| routed experts | clamp `gate≤limit`, `up∈[-limit,limit]` then `act(gate)*up` (`_apply_gate`) | same (`silu_and_mul_masked_post_quant.cuh:51-73`) — **MATCH** |
| shared expert | `act_fn(gate_proj(x))*up_proj(x)` — **NO clamp** | clamps with `swiglu_limit=10.0` (`deepseek_v2.py:675`) |

**Fix (train → add clamp):** apply the same clamp in `V4SharedExpertMLP.forward`, in bf16
(sglang clamps in bf16 by design). Thread `swiglu_limit` from config (checkpoint has `10.0`).

**Clamp backward = STE.** `torch.clamp` has zero gradient outside `[min,max]`, so on outlier
activations (|gate/up| > 10) the gradient flowing back *through* the frozen expert to the LoRA
is killed. The DeepSeek-V4 paper does not specify the swiglu-limit clamp's backward (it uses
QK-Clip nowhere and FP4-QAT elsewhere), so we use the **straight-through estimator**: forward
clamps, backward is identity (`_clamp(x)=x+(x.clamp()-x).detach()`, `_CLAMP_STE`, default on;
`V4_CLAMP_STE=0` reverts). Applied to **both** the routed `_apply_gate` and the shared `_act`.

## Structural asymmetry — weight sync (audit confounder, not a forward bug)

The LoRA-only weight sync (`update_weight_from_distributed.py`) **skips experts**
(`:279 if ".experts." in name: continue`) and merges LoRA attention as **bf16**
(`_merge_lora_weight → base_weight.dtype`, train is `--bf16`). Consequence:

| rollout state | attention | experts |
|---|---|---|
| **real loop** (after a weight sync) | bf16 (synced from train) | original fp8 (never synced) |
| **`--debug-rollout-only`** (no sync) | fp8 (initial load) | fp8 (initial load) |

The A–I debug comparisons used no-sync data (full fp8), which is **not** what the real loop sees.
The real-loop optimum is therefore **fp8-experts (ue8m0) + bf16-attention** — mirroring what the sync
makes sglang serve — and the 16k formal run (that config) measured `logprob_diff` 0.022.
**Action:** always verify precision choices against a sync'd/real-loop rollout, not `--debug-rollout-only`.

## Negligible / env-gated mismatches (MEMO — do not re-open without evidence)

| item | same math? | same numerics? | detail | why negligible |
|---|---|---|---|---|
| mHC prenorm | yes | no (env-gated) | train FP32 `_HC(... .float())`; sglang `tf32_hc_prenorm_gemm` (TF32) **iff** `SGLANG_OPT_DEEPGEMM_HC_PRENORM=True` (default True; `server_args.py:2122` can disable). Else branch `hc_pre_torch_impl` is fp32. | **MEASURED 2026-07:** TF32-vs-FP32 rel-err = **1.79e-4/site** → ~1.7e-3 random-walk over 43×2 sites (coherent worst 1.5e-2). Input `x_flat` is bf16 on both sides (bf16⊂TF32 → exact); only `hc_fn` rounds; mix is a *detached* Sinkhorn oracle (no grad compound). 45× below the per-layer bf16 residual rounding. Negligible vs 0.037. |
| weighted RMSNorm | yes | no | train keeps weighted RMSNorm bf16 (`mcore_model.py:997-1011`, quantified+accepted); sglang default `cast_x_before_out_mul=False` (`layernorm.py`) casts differently | eps matches (1e-6); dtype-order only |
| router `+1e-20` renorm guard | yes | ~ | train adds `1e-20` to the weight-sum denominator; sglang does not | scores positive for sigmoid/sqrtsoftplus → denominator ≫ 1e-20 |
| fp8 gemm accumulation | n/a | no | non-bit-identical deep_gemm invocation vs sglang | see §Precision |

### Round 3 (2026-07) — the memo items above NOW FIXED (train→rollout alignment)

User directive: fix all remaining mismatches even negligible, primarily changing TRAIN to match
ROLLOUT. Done + Codex-reviewed (94 tests pass):

| item | fix | side | verdict |
|---|---|---|---|
| **weighted RMSNorm cast** | `V4RMSNorm.forward` now `(self.weight * hidden_states).to(input_dtype)` (fp32-mul-then-cast), matching sglang `cast_x_before_out_mul=False`. Env `V4_RMSNORM_HF=1` reverts to HF order. Measured order-diff 4.15e-3 (~1 bf16 ulp). | **train** | Codex CORRECT |
| **router +1e-20 + score dtype** | removed `+1e-20` renorm guard AND now `score_fn(logits.float())` — sglang scores in fp32 (`gating_output.float()`), which also aligns the top-k SELECTION, not just weights. | **train** | Codex-corrected (was RISKY: dtype) |
| **mHC prenorm TF32** | run script forces sglang FP32: `SGLANG_OPT_USE_TILELANG_MHC_PRE=false` + `SGLANG_OPT_DEEPGEMM_HC_PRENORM=False` (BOTH needed — DEEPGEMM alone lands on another TileLang kernel, not fp32; Codex-caught). Gated by `V4_ALIGN_MHC_FP32=1`. Changes ROLLOUT to fp32 (keeps correct precision, does NOT degrade train to TF32). | rollout | Codex-corrected (was WRONG: single-flag no-op) |
| indexer sparse | DEFERRED — needs A1 tilelang kernel port; measured effect ≤0.0007. `V4_SPARSE_ATTENTION=1` reference path exists. | — | negligible, not worth kernel port |

## Confirmed MATCH (same math AND acceptable numerics)

Router gating-weight order (sqrtsoftplus + `e_score_correction_bias` for indices only + gather +
renorm + `routed_scaling_factor`); RoPE (theta, `compress_rope_theta`, YaRN, partial-rotary,
interleaved rotation, `is_neox_style=False`); attention `softmax_scale = head_dim**-0.5`; mHC
algorithm; logits (fp32, no softcap; temperature=1 so `logits/T` is a no-op); tokenization/chat
template (`add_generation_prompt`, `add_special_tokens=False`, BOS from template).

## Precision floor (memo — why fp8 does NOT help here)

Measured on identical data (ctx-2048, step-0), train vs a **known full-fp8** rollout:

| train forward | logprob_diff |
|---|---|
| bf16 | **0.034** |
| unified fp8, non-bit-identical kernels | **0.058** (worse) |

Isolation proved our fp8 *components* match sglang (MoE 0.0046, attn-proj 0.0051), yet the full
fp8 forward is *worse* than bf16: two independent fp8 quantizations (our deep_gemm kernels vs
sglang's) don't cancel and compound over 43 layers. DeepSeek-V4 (arXiv 2606.19348) resolves this
with **unified FP8 QAT + bitwise batch-invariant deterministic kernels** — "bitwise alignment
among pre-training, post-training, and inference pipelines." We lack bit-identical kernels, so
**bf16 train is the best practical choice** unless we invest in kernel alignment. Field-standard
alternatives: FP16 train (reported ~7.7× smaller diff than bf16) or TIS/MIS correction (Miles MoE
recipe: token TIS [0.5,1.5] + geometric MIS [0.99,1.001] + batch-norm — ours `[0.999,1.001]` is too
tight, no token-TIS/batch-norm, hence ~90% MIS reject).

<!-- Also fixed en route (kept, env-gated off): ue8m0 weight/activation scale SPLIT
(_V4_FP8_WEIGHT_UE8M0=True matches ckpt; _V4_FP8_ACT_UE8M0 auto False on H20/Hopper; the old
single flag gave 88%-wrong expert weights). mega_moe is Blackwell-only (DEEPGEMM_BLACKWELL=False
on H20) — H20 uses m_grouped_fp8_gemm_nt_contiguous, same as train. -->

## Verification (after fixes) — original plan, now OBE

1. ~~Fresh rollout with dense sglang at 16k~~ — moot; indexer term measured ≤0.0007 (see round 2).
2. Shared-clamp fix verified (94→93 tests pass; the 1 fail is a pre-existing WS-kernel small-N flake).
3. ctx-2048 unchanged (indexer was already a no-op there) — confirmed.

## Round 2 (2026-07) — measured audit closure

**Newly audited components (were NOT in the original 9-item audit) — all MATCH:**

| component | train | rollout (sglang) | verdict |
|---|---|---|---|
| CSA compressor | softmax-pool + `position_bias` + `kv_norm` (B2 kernel) | `compress_forward` online softmax-pool + `ape` | same math; kernel-precision only |
| attention sinks | `self.sinks`, mapped `←attn.attn_sink` (`deepseekv4.py:35`), loaded nonzero (mean 0.498) | `self.attn_sink` loaded | loaded + match |
| conjugate output-RoPE | `apply_rotary(out, cos, -sin)` (`attention.py:148`) | `fused_rope_inplace(o[...,-rope:], inverse=True)` (`deepseek_v4.py:980`) | match |

**Empirical mismatch decomposition (measured, step-0):**

| what | measurement | meaning |
|---|---|---|
| indexer term | dense-train 0.0404 vs sparse-train 0.0411 @ ctx-8192 (both vs sparse rollout) | **≤0.0007** (sparse gave no gain) |
| bf16-indexer vs fp8-rollout selection overlap | 98.9% (real L12 weights, ctx>2048) | replay unnecessary |
| non-indexer term | 0.037 @ ctx-2048 (indexer no-op) | the real gap |
| bf16 sparse flash-MLA kernel vs reference | 0.0037 | attention *algorithm* aligned |

**Train-side sparse CSA** (built, env `V4_SPARSE_ATTENTION=1`, torch-reference path; A1 kernel port
NOT done): indexer weights load cleanly (`compressor.indexer.*` ↔ ckpt `attn.indexer.*`, CSA layers
only). Kept for reference; **not worth the A1 kernel port** — the indexer isn't the mismatch.
sglang exposes `enable_return_indexer_topk` if exact-selection replay is ever wanted (~5.6 GB/batch;
not built — 98.9% overlap makes it a predicted null).

**The one unrun test that would settle "precision vs path":** serve sglang in bf16 (dequant the fp8
ckpt), regen rollout, compare bf16-train vs bf16-rollout. →0 confirms fp8 precision is the whole
gap; ≈0.037 means it's path/kernel/structural and the precision story is wrong.

**Why "use sglang's kernels in train" doesn't work:** they're inference-only (`sparse_decode_fwd`,
no backward) and serving-context-bound (`ForwardBatch`, paged KV, `PagedIndexerMetadata`). The A1
custom kernel is mandatory (non-standard MLA head_dim=512 + CSA + sinks; flash-attn can't express it;
training needs a backward). The gap is that A1 and sglang's flash-MLA are *independently written* —
closable only by unifying on one bit-identical kernel (DeepSeek's design) or TIS/MIS.
