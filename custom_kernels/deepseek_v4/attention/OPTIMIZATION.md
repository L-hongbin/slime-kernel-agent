# TileLang Attention Review

## Correctness Findings

Bottom line: under the stated contract, I do not see a forward/backward math bug. The sink, MQA shared-KV gradient, masks, and padding are internally consistent with `reference.py`.

1. **Sink handling looks correct.**
   `kernel.py:123-132` merges the per-head sink into the online softmax only at the end:
   `lse = log(exp(sink) + sum_real exp(score))`, while the output numerator still excludes the sink. That matches `reference.py:114-120`.
   Backward also uses the sink-inclusive `Lse`, and `dsink = -p_sink * delta` at `kernel.py:367-371` is the right denominator-only gradient.

2. **MQA dKV accumulation is algebraically correct.**
   In `kernel.py:292-294`, `dKV = dk + dv` is accumulated into the single shared KV tensor. Since `K == V` and the reference broadcasts one KV head to all `H=64` query heads, summing all head contributions with `atomic_add` is mathematically correct.

3. **Mask and banding rules match the reference.**
   Raw mask:
   `key < S`, `key <= q`, `q - key < window`.
   Compressed mask:
   `(w + 1) * m <= q + 1`, equivalent to `w < (q + 1) // m`.
   These match `reference.py:53-66`.

4. **Padding is safe for default block sizes.**
   Padded raw KV is excluded by `base + j < S`; padded compressed KV by `w < Tcomp`. Padded query rows may compute forward values, but they are sliced away, and `dO` is zero there in backward, so they do not contribute gradients.

5. **Real correctness risks are contract/API risks, not current math bugs.**
   - `Sp = ceil(S, max(block_M, block_N)) * max(...)` only guarantees both divisibilities when the larger block size is a multiple of the smaller. If arbitrary block sizes are exposed, use `lcm(block_M, block_N)` or assert compatibility.
   - Inputs are silently coerced through `q.new_zeros`; mixed q/k dtypes are not really supported.
   - Sinks are assumed finite fp32. `NEG = -1e30` is fine for bf16 model logits, but not for pathological `-inf`/NaN sinks.
   - Gradients are nondeterministic because dKV uses fp32 atomics over 64 heads.

## Performance Diagnosis

The backward ratio is not only atomics. Current backward does about **7 GEMM-equivalent operations per valid tile**:

- dQ kernel: `QK`, `dO K`, `dQ` = 3
- dKV kernel: `KQ`, `K dO`, `dK`, `dV` = 4

Forward does 2: `QK` and `PV`. So a 3.5x backward/forward floor is already implied before atomic overhead. To reach flash-style 2-2.5x, the duplicated recompute between dQ and dKV has to be reduced, not just the atomics.

## Forward Optimizations, Prioritized

| Priority | Change | Expected win | Risk |
|---|---|---:|---|
| P0 | Avoid wrapper copies when already aligned: use `q` directly when `S == Sp`; avoid concatenating raw/comp KV by passing separate raw/comp tensors or requiring a prepacked KV buffer. | Sparse cases: 10-40% wall-time; dense: 0-10%. | Low-medium; adds variants and contiguity checks. |
| P1 | Use Hopper pipelining: try fwd `num_stages=2`, `threads=256` so `T.Pipelined` can overlap K/V tile loads with WGMMA/TMA. Current default is `stages=1`, `threads=128`. | 15-35% fwd. | Medium; smem becomes ~192KB for `BM=BN=64`, occupancy may stay 1 CTA/SM. |
| P1 | Check/force WGMMA shared layouts. `K_sh` is used as `B^T` for `QK` and as `B` for `PV`; one shared layout may be suboptimal for one GEMM. Try separate `Kq_sh` and `V_sh` with explicit `T.annotate_layout` / WGMMA swizzle. | 20-50% if one GEMM is currently degraded. | High; extra smem/global load, layout inference fragility. |
| P2 | Autotune tile shapes for hd512: especially `(BM,BN)=(64,128)`, `(32,128)`, `(32,64)`, with `threads=256`. | 10-30%. | Medium; `BN=128` uses much more smem and may hurt sparse-window overfetch. |
| P2 | Add dense-causal and sliding-specialized paths that skip per-element mask branches for fully valid tiles; only edge/diagonal tiles need masks. | 5-15%. | Low-medium; more code paths. |
| P3 | Convert online softmax to base-2: scale logits by `log2(e)`, use `T.exp2`, store/convert LSE consistently. | 5-10%. | Medium; must update backward exactly and revalidate tolerances. |
| P3 | Improve block scheduling/L2 locality across heads using TileLang threadblock swizzle or grid order so adjacent CTAs reuse the same shared MQA KV tiles. | 5-20%. | Low-medium; benefit depends on whether K traffic is material vs tensor-core saturation. |

## Backward Optimizations, Prioritized

| Priority | Change | Expected win | Risk |
|---|---|---:|---|
| P0 | Split raw and compressed dKV kernels. For compressed dKV, use the tight lower query bound `q >= (w_min + 1) * m - 1` instead of looping from query block 0 for every compressed KV block. | Compressed dKV up to ~2x; full bwd 10-40%. | Medium; prior comment notes TVM analyzer issues, likely easier if raw/comp are separate kernels. |
| P0 | Head-group dKV accumulation: one CTA handles `G=4/8` heads for a KV tile, accumulates into one `dkv_acc`, then does one atomic tile add. | Reduces atomic traffic by `G`; dKV 1.3-2.5x if atomics dominate. | Medium; less parallelism and longer CTAs. |
| P1 | Deterministic two-stage dKV: write per-head-group partials `[B, H/G, KV, D]`, then reduce groups in a second kernel. | dKV 1.5-3x; removes atomic nondeterminism. | High memory: for `S=8192`, `G=8` scratch is ~134MB fp32; scales with KV. |
| P1 | Use vectorized atomics if retaining atomics: `atomic_addx4` along contiguous `D=512`. | 5-20% dKV atomic issue-rate improvement. | Low-medium; still contended, just fewer instructions. |
| P2 | Move toward a 5-GEMM flash backward design: compute `QK` and `dO·K` once per tile, then produce dQ and dK/dV partials from the same `P,dS`. | Needed to reach ~2.5x bwd/fwd. | High; requires partial reductions or atomics for one side. |
| P2 | Tune dQ separately: `BM=32, BN=64, stages=2` may fit `Q + dO + 2*K` in smem and reduce register pressure. | 10-25% dQ. | Medium; smaller M may reduce WGMMA efficiency. |
| P3 | Separate dk and dv partial reductions only if the reducer path is adopted. Combine final `dk + dv` after reduction. | 5-15% possible scheduling win. | Medium; doubles partial streams if done naively. |

## Recommended Order

1. First fix measurement overhead and shape guards: skip unnecessary `q_pad/kvt` copies and assert/lcm block compatibility.
2. Tune forward with `stages=2`, `threads=256`, then test `BN=128`.
3. Inspect generated CUDA/SASS to confirm both forward GEMMs are WGMMA and whether the shared `K_sh` layout is compromising either `QK` or `PV`.
4. For backward, split dKV raw/comp and add tight compressed `q_lo`.
5. Then reduce atomics with head grouping; only move to scratch-based deterministic reduction if atomics still dominate.
6. Treat a fused/5-GEMM backward as the larger redesign required to get from ~3.5-4x toward flash-style ~2.5x.
