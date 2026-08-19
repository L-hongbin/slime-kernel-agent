# M1 — V4-Flash torch forward parity vs HF eager

**Status: PASS.** The M0 torch `V4Model` forward is numerically equivalent to HF
`DeepseekV4ForCausalLM` (eager) on the tiny config, across **all three attention
layer types** (sliding / CSA / HCA) and **both MoE routers** (hash + top-k). Every
non-MoE seam matches at the bf16 op-floor; the only above-floor divergence is a
6/256-token top-k router boundary that is **inherent to bf16** (HF's own bf16 run
flips the same way), not a bug in our module. This closes the #1 project risk: the
3-kernel V4 scaffold (A1 attention / B2 compression / B1 mHC) plus the torch seams
(partial RoPE, output conjugate RoPE, grouped `o_a`/`o_b`, the mHC `post·out +
comb.T@stream` mix, MoE) are aligned with the reference.

Re-run: `CUDA_VISIBLE_DEVICES=0 python -m custom_kernels.deepseek_v4.megatron.m1_parity`
(harness: `megatron/m1_parity.py`). Exits 0 on PASS, 1 on any real divergence.

## Hardened validation (folded from the M1 codex review)

The harness runs five passes; the verdict requires the two real variants to PASS and
both negative controls to BREAK:

| pass | what it proves | result |
|---|---|---|
| baseline | sinks=0, position_bias=0 (HF `_init_weights`) | **PASS** |
| **nonzero** | shared sinks + compressor/indexer `position_bias` set nonzero in HF *and* M0 (identical values) — exercises the sink path and the gated-pool `position_bias` path away from 0 | **PASS** (rel ≤ 0.0194, cos ≥ 0.9998) |
| neg-control: zero RoPE sin | drops the output conjugate-RoPE `sin` in M0 | BREAKS (rel ~0.5, cos 0.82–0.95) ✓ |
| neg-control: permute `o_a` rows | shuffles every grouped `o_a_proj` weight in M0 | BREAKS (rel ~1.4, cos ~0) ✓ |
| CSA stress: `index_topk=4 < 32` | HF CSA does a *real* top-k while A1 stays dense | divergence at the CSA layer + downstream, **labeled dense-approx, NOT a failure** |

Two negative controls on different seams (RoPE and `o_a`) confirm the harness has
teeth on more than one path. Each non-MoE seam is now gated on **max_abs / p99 /
worst-token** in addition to global rel/cos, so a localized 1-token or 1-head error
cannot pass a diluted global metric.

### Nonzero-sink / position_bias parity holds

The nonzero variant matches at the bf16 floor on every non-MoE seam (rel ≤ 0.0194,
cos ≥ 0.9998, max_abs ≤ 0.14, p99 ≤ 0.051). The A1 sink path was also checked in
isolation vs the reference (rel ~0.002 at sink ∈ {0, 0.05, 0.5, 2.0}) and HF's
`eager_attention_forward` sink handling is **bit-identical** to the A1 reference
(rel 0.0 at sink=2.0). Conclusion: the nonzero sink and `position_bias` paths are
correct, not silently passing at zero.

### The L2 top-k flip is pure bf16 rounding (confirmed)

The 3-way router-index diagnostic (hf_fp32 / hf_bf16 / m0) on the flipped tokens
shows: `router-in rel(flipped) = 0.017` (the router INPUT matches M0↔HF-bf16 at the
same 1.7% bf16 floor as all tokens) and `margin_med = 0.016` (the flipped tokens are
near-ties between the kth and k+1th expert). Router input matches at floor + tiny
margins ⇒ the flip is bf16 discreteness, **not** a divergent router-input seam bug.
Hash routers (L0, L1) have **0 flips** (frozen `tid2eid` lookup, dtype-independent).

The gate has teeth — zeroing the RoPE `sin` drives every seam to cos 0.82–0.95 and
exits nonzero; permuting `o_a` drives cos to ~0.

## How parity is measured

Three forwards on the same `input_ids` (B=2, S=128, no cache, contiguous positions):

| run | dtype | role |
|---|---|---|
| HF-fp32 | fp32 | reference truth (`DeepseekV4ForCausalLM.model`, eager) |
| HF-bf16 | bf16 (mHC + hc_head fp32) | **same-dtype** reference — isolates op divergence from the fp32→bf16 gap; PASS is judged on this |
| M0 | bf16 (mHC + hc_head fp32) | our `V4Model`, HF weights copied in via `load_state_dict` (88 keys, exact) |

Weights are HF's `_init_weights` values (not random); `tid2eid` gets valid random
expert ids. M0 mirrors HF-bf16's dtype layout exactly: only `attn_hc`/`ffn_hc`/
`hc_head` are fp32, RMSNorms stay bf16 (a pure-eager HF with fp32 norms feeding bf16
linears crashes, so the faithful same-dtype reference keeps norms bf16 in both).

Metric: global relative error `rel = ||got−ref||/||ref||` (not elementwise
`diff/|ref|`, which explodes on near-zero entries) plus cosine. PASS floor:
upstream (non-MoE) seams `rel ≤ 0.02` and `cos ≥ 0.9995` vs HF-bf16; MoE-downstream
seams (`post_ffn_hc`, final) judged on `cos ≥ 0.997` because top-k flips inflate
max_abs on a few tokens while leaving cosine ~1; the rigorous MoE-op check is the
matched-router-token diagnostic.

## Per-seam result (M0 vs HF-bf16, same dtype)

| layer | type | seam | rel | cos | max_abs |
|---|---|---|---|---|---|
| L0 | sliding / hash_moe | attn pre-o_proj | 0.0051 | 0.99999 | 0.023 |
| L0 | sliding / hash_moe | post-attn HC | 0.0062 | 0.99998 | 0.031 |
| L0 | sliding / hash_moe | post-FFN HC | 0.0124 | 0.99992 | 0.078 |
| L1 | CSA / hash_moe | compressor out | 0.0156 | 0.99988 | 0.099 |
| L1 | CSA / hash_moe | attn pre-o_proj | 0.0123 | 0.99992 | 0.044 |
| L1 | CSA / hash_moe | post-attn HC | 0.0142 | 0.99990 | 0.109 |
| L1 | CSA / hash_moe | post-FFN HC | 0.0185 | 0.99983 | 0.164 |
| L2 | HCA / moe | compressor out | 0.0124 | 0.99992 | 0.051 |
| L2 | HCA / moe | attn pre-o_proj | 0.0137 | 0.99991 | 0.066 |
| L2 | HCA / moe | post-attn HC | 0.0185 | 0.99983 | 0.156 |
| L2 | HCA / moe | post-FFN HC | 0.063 | 0.998 | 2.76 ← top-k flips |
| — | — | FINAL hidden | 0.060 | 0.998 | 1.88 ← top-k flips |

The vs-HF-fp32 column (informational) is slightly looser because it also carries the
fp32→bf16 dtype gap, whose floor is HF-bf16-vs-HF-fp32 = rel 0.044 / cos 0.999.

## The only above-floor divergence: top-k router boundary (irreducible)

`post_ffn_hc`/final at L2 carry `max_abs ≈ 2`, but it is concentrated on **6 of 256
tokens** — exactly the tokens where bf16 rounding flips which expert the `top_k`
router selects vs the fp32 forward. A different expert emits a totally different
vector, so those few tokens get a large per-token error while cosine stays 0.998.

Evidence this is routing discreteness, not a seam bug:

- **Hash routers (L0, L1): 0/256 flips.** Hash routing is a frozen `tid2eid`
  lookup, dtype-independent → its MoE output matches at the bf16 floor (rel ≤ 0.02,
  max_abs ≤ 0.06).
- **Top-k router (L2): 6/256 flips.** On the **250 tokens where the router agrees**,
  the MoE output matches at the bf16 floor (cos 0.99967, max_abs 0.072). The entire
  large divergence is the 6 flipped tokens.
- The flip is **symmetric**: it appears the same way against HF-bf16 (same dtype),
  so it is the model's own bf16 behavior, not our approximation. Forcing the router
  to fp32 reduces but cannot eliminate it (the router *input* is still bf16, so
  borderline tokens still tie-break differently from an fp32 forward).

The MoE math is HF's verbatim (`DeepseekV4SparseMoeBlock` reused). No fix is needed
for parity; this is the standard bf16 MoE routing-boundary effect.

## Documented caveats

- **CSA = dense-over-compressed.** A1 drops the Lightning Indexer top-k. At tiny seq
  the compressed length (CSA 32, HCA 1) is ≤ `index_topk=512`, so HF's top-k selects
  every causally-valid entry and HF CSA == our dense path **exactly** (verified:
  L1 CSA compressor + attention at bf16 floor). At long seq the dropped top-k is the
  deliberate dense **approximation** (per the kernel inventory), not bit-exact.
  sliding/HCA have no top-k and are bit-for-bit at any length.
- **A1/B2 are bf16-only** → M0 runs bf16; HF-fp32 is truth, HF-bf16 is the
  same-dtype op reference. Do not attempt an fp32 M0 forward.
- **tiny config is structural**: `hidden=4096`, `head_dim=512`, `hc_mult=4`,
  `hc_sinkhorn_iters=20`, `hc_eps=rms_norm_eps=1e-6` are FIXED (A1 fixes hd=512; B1
  hardcodes H=4/HIDDEN=4096/SINKHORN=20/EPS=1e-6). Only layers/heads/experts/vocab/
  seq are shrunk. The harness asserts these.

## Coverage

Layer types exercised: sliding_attention, compressed_sparse_attention,
heavily_compressed_attention. Routers exercised: hash_moe, moe (top-k). All three
kernels fire (A1 ×3, B2 CSA + HCA, B1 ×6). Final `last_hidden_state` compared end
to end.
