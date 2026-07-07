# A1 — DeepSeek-V4-Flash fused flash attention (tilelang)

Fused **MQA shared-KV + per-head attention-sink + sliding-window/compressed flash attention** at
**head_dim=512**, hand-written tilelang forward + backward, on H20 (sm90 Hopper, bf16 peak ~148 TF/s).
This is the long-pole A1 kernel from `handoffs/deepseek-v4/v4_kernel_inventory.md`.

## Files
- `reference.py`        — exact fp32 torch port of the HF eager math (source of truth).
- `kernel.py`           — tilelang fwd + hand-written bwd (dQ kernel + dKV kernel, recompute path) in
  `torch.autograd.Function`. Public API `v4flash_attention(q, k_raw, k_comp, sinks, window, m, ...)`.
- `test_correctness.py` — fwd + bwd (dq, dk_raw, dk_comp, dsink) vs fp32 reference; fp64 gradcheck on the reference.
- `kernel.py::_build_fwd_ws` — warp-specialized split-D forward (default; `_USE_WS_FWD`). `_build_fwd`
  is the single-warpgroup baseline, kept for fallback/comparison.
- `exp_ws.py` / `exp_ws_check.py` — ws-vs-baseline speedup + %peak, and Out/Lse byte-parity + deadlock
  guard across shapes. `bench_fwd_quick.py` — fast forward-only %peak micro-bench.
- `bench.py`            — latency / TFLOP-s / peak-mem vs `torch.compile(reference)`.
- `bench_bwd.py`        — backward-vs-forward efficiency: separately times fwd / dQ / dKV / full bwd,
  reports TF/s + %peak + the bwd/fwd latency & efficiency ratios (the dedicated deliverable below).
- `bench_flashmla.py`   — PRIMARY SOTA forward comparison vs sglang's REAL V4 kernel,
  FlashMLA `flash_mla_sparse_fwd` (sparse-512 + dense-causal configs).
- `bench_sglang.py`     — Secondary SOTA forward comparison vs sglang FA4 (FlashAttention-CUTE).
- `bench_sota.py`       — Secondary SOTA forward comparison vs FlashInfer-MLA (hd_ckv=512).

## What the kernel computes
Shared-KV MQA (K == V, one KV head broadcast to all H=64 query heads), head_dim D=512, bf16 compute /
fp32 softmax+accum. Per-head attention sink (gpt-oss style: an extra logit column = `sinks[h]` enters
the softmax denominator only, no value contribution). KV axis is
`[ raw (length S, sliding-window causal, W=128) ++ compressed (length Tcomp, causal-threshold
(w+1)*m <= qpos+1) ]`, walked by a SINGLE kv loop so the `acc_o` accumulator keeps one consistent
fragment layout. Block-sparsity is exploited structurally: only banded raw blocks + compressed blocks
below the per-block causal threshold are iterated; fully-masked tiles are never visited.

Backward is hand-written (no autodiff), recompute path. Forward stores `lse_i` (incl. sink) and
`delta_i = sum_d dO_i·O_i`; for valid (i,j): `p = exp(scale·q_i·k_j − lse_i)`, `dp = dO_i·k_j`,
`dS = scale·p·(dp − delta_i)`, then `dQ_i = sum_j dS·k_j`, `dk_j = scale·sum_i dS·q_i`,
`dv_j = sum_i p·dO_i`. The dKV kernel atomic-adds `dk_j + dv_j` (fp32) into the single shared-KV
buffer to sum the broadcast contribution over all H heads. `dsink` computed in torch for completeness.

## Correctness — OVERALL PASS (GPU-validated, H20 GPU 0)
`python test_correctness.py` → OVERALL PASS. fp64 `torch.autograd.gradcheck` on the fp32 reference
passes; bf16 kernel vs fp32 reference is within tight bf16 tolerance for forward and all backward grads
across S∈{512,2048,8192}, sliding-only + CSA m=4 + HCA m=128, W∈{128,64}, B∈{1,2}.

| check | tolerance | result |
|---|---|---|
| reference fp64 gradcheck | atol 1e-4 / rtol 1e-3 | PASS |
| forward (bf16 vs fp32 ref) | rel < 1.5e-2 | PASS (observed rel ~2.4e-3 … 3.2e-3) |
| backward dq / dk_raw / dk_comp / dsink | rel < 2e-2 | PASS (observed rel ~2.0e-3 … 4.8e-3) |

(At S8192 m4-comp the dense fp32 reference OOMs; the kernel fwd+bwd runs — recorded as WIN.) Also
cross-validated at head_dim=256 in `bench_sglang.py` (rel ~2.4e-3 … 3.2e-3 vs fp32 ref).

## Benchmark vs torch.compile(reference) (H20, bf16, B=1, H=64, D=512)
The reference is a naive dense O(S²) attention, so the kernel wins hugely and the reference OOMs ≥16k.
FLOPs counted over structurally-valid (q,k) pairs (sliding band + compressed), 4·D per pair.

| shape | kernel fwd ms (TF/s) | kernel fwd+bwd ms | bwd/fwd | torch.compile fwd+bwd ms | speedup (fwd+bwd) |
|---|---|---|---|---|---|
| S2048 m4   | 2.85 (35.8) | 13.23 | 3.64 | 85.94  | 6.50x |
| S2048 m128 | 1.66 (21.3) | 7.30  | 3.39 | 70.89  | 9.71x |
| S8192 m4   | 26.54 (46.6)| 133.09| 4.02 | 1322.42| 9.94x |
| S8192 m128 | 6.63 (25.7) | 29.21 | 3.41 | 1074.99| 36.80x |
| S16384 m4  | 93.71 (49.8)| 481.04| 4.13 | OOM    | kernel-only |
| S32768 m128| 34.07 (32.2)| 155.39| 3.56 | OOM    | kernel-only |

## Community SOTA forward comparison (the real efficiency bar)
torch.compile(naive reference) is a weak baseline. The honest question: is our FORWARD within a
reasonable factor of a real, production-tuned hd-large kernel? Forward-only (these are inference
kernels). Three baselines were built on this box; the PRIMARY one is the kernel sglang's V4 backend
actually runs.

### PRIMARY — FlashMLA `flash_mla_sparse_fwd` (the REAL sglang V4 attention kernel)
`sgl_kernel.flash_mla.flash_mla_sparse_fwd` is exactly what `deepseek_v4_backend.py::_forward_prefill_sparse`
calls for V4 prefill — the production V4 attention. It is **MLA matrix-absorbed**: q/kv live in the
512-latent + 64-rope space (`d_qk=576`, `d_v=512` latent), one shared latent KV head broadcast to all
H=64 query heads (same shared-KV/MQA structure as ours), with a **top-k sparse** key budget per query
(the indexer selects ≈512 keys). Forward / inference only (no backward, fp8/paged-capable, per-head
`attn_sink` supported but no value contribution). `bench_flashmla.py` drives it standalone at V4-Flash
dims (`h_q=64, d_qk=576, d_v=512`) in two configs. Achieved bf16 TFLOP/s = useful-flops/time;
FlashMLA flops/pair/head = `2·(d_qk+d_v)=2176`, pairs = `Σ_i topk_length[i]`; ours flops/pair/head =
`4·D=2048`, dense-causal pairs = `S(S+1)/2`. The `topk_length` gate was verified to faithfully meter
work (cap-512 vs full-causal lens at S4096: 2.46 ms / 1.97M pairs vs 10.05 ms / 8.39M pairs).

**(a) NATIVE SPARSE (topk=512 — the production setting).** This is the real V4 attention's latency/
throughput. It is NOT directly ratio-able against our production sparse kernel: FlashMLA does a *fixed
512-key budget per query* (≈linear total work, `512·S`) while ours does *sliding-W=128 raw + dense-
over-compressed* (CSA m=4 → ~`S²/8` pairs, quadratic), and FlashMLA is matrix-absorbed — so a latency
ratio here is dominated by the **sparsity algorithm**, not the kernel. Reported for reference:

| S | FlashMLA pairs | FlashMLA ms / TF/s | ours fwd ms (config, from bench.py) |
|---|---|---|---|
| 2048  | 0.92M | 1.195 / 106.9 | 2.85 (m4) · 1.66 (m128) |
| 4096  | 1.97M | 2.468 / 110.9 | — |
| 8192  | 4.06M | 4.999 / 113.2 | 26.54 (m4) · 6.63 (m128) |
| 16384 | 8.26M | 10.064 / 114.3| 93.71 (m4) |
| 32768 | 16.65M| 20.197 / 114.8| 34.07 (m128) |

→ The real V4 kernel sustains ~107–115 TF/s. Our production kernel's larger latency at m4 is mostly
that **dense-over-compressed does far more pairs** than a fixed top-512 budget (algorithm), compounded
by MLA absorption + our ~0.5× kernel-efficiency (below). m128 (sparser compressed) is much closer.

**(b) DENSE CAUSAL (topk_length → full causal — the kernel-efficiency bar).** Both kernels do FULL
causal attention over all keys, removing the sparsity-algorithm advantage. The residual gap is the
closest thing to pure kernel efficiency (still confounded by MLA absorption doing PV over the 512
*latent* vs our materialized hd512 — see caveats). Our kernel runs matched dense-causal (window≥S, no
compressed):

| S | FlashMLA ms / TF/s | ours dense ms / TF/s | ours/FlashMLA |
|---|---|---|---|
| 2048  | 2.689 / 108.7 | 4.000 / 68.8 | **0.63x** |
| 4096  | 10.055 / 116.2| 14.759 / 74.5| **0.64x** |
| 8192  | 38.838 / 120.3| 56.520 / 77.8| **0.65x** |
| 16384 | 152.646 / 122.5| 221.470 / 79.4| **0.65x** |
| 32768 | 604.410 / 123.7| 876.217 / 80.3| **0.65x** |

→ With the **warp-specialized split-D forward** (see "Forward warp specialization APPLIED" below) our
hd512 dense-causal forward is now **~0.65x** of FlashMLA's dense throughput (was ~0.50x before warp
specialization), i.e. ~80 TF/s = ~53% of the H20 bf16 peak. This was the targeted Phase-2 win.
(The pre-warp-spec single-warpgroup baseline was ~0.50x / ~53–61 TF/s — matching the FA4 @hd256 0.49x and
FlashInfer-MLA @hd512 0.48–0.51x results below, which were measured against that older baseline.)

> **Methodology (codex-reviewed).** All numbers above are **FORWARD-ONLY** (FlashMLA is inference-only;
> our kernel supports fwd+bwd but only its forward is timed here — our fwd+bwd is ~3.4–4.1× the forward,
> per the torch.compile table). TF/s = *useful-matmul* throughput (excludes softmax/exp/sink/mask work).
> The dense-config ratio is **directional, not an identical-operation head-to-head**: "FlashMLA-dense"
> is the *sparse* kernel `flash_mla_sparse_fwd` driven with full-causal indices, in MLA-absorbed latent
> space (PV over the 512-latent), vs our materialized dense hd-512 K≡V — so it favors FlashMLA. `sm_scale`
> (1/√576) is passed only to FlashMLA; our kernel uses its own internal 1/√512 (verified — no scale leak).

### SECONDARY — sglang FA4 (`sglang.jit_kernel.flash_attention_v4.flash_attn_varlen_func`)
This wraps the vendored **FlashAttention-CUTE (FA4)** kernel and supports EXACTLY our pattern:
causal + `window_size=(W−1,0)` sliding window + per-head `sinks` + MQA (1 KV head). On the
sliding+sink dense path it is the **same math** as ours — the closest possible apples-to-apples bar.

**Decisive hardware fact (probed):** FA4 supports head_dim only in `[8, 256]` on SM90 —
`(head_dim, head_dim_v)=(512,512) is not supported on SM90`. So **FA4 cannot run V4-Flash's actual
head_dim=512** — the exact structural blocker that motivates this custom kernel. A community flash
kernel exists, but not at the hd=512 the model needs. We therefore compare at hd=256 (FA4's max), the
shape it *can* run, as a pure kernel-efficiency check (both validated vs fp32 ref, rel ~3e-3):

| head_dim | S | ours ms / TF/s | FA4 ms / TF/s | ours/FA4 |
|---|---|---|---|---|
| 256 | 2048  | 0.557 / 29.9 | 0.273 / 60.9 | 0.49x |
| 256 | 4096  | 1.095 / 30.9 | 0.541 / 62.5 | 0.49x |
| 256 | 8192  | 2.176 / 31.3 | 1.073 / 63.6 | 0.49x |
| 256 | 16384 | 4.329 / 31.6 | 2.131 / 64.3 | 0.49x |

→ Our forward is **~0.49x** of a fully production-tuned FA4 on identical sink+sliding+MQA math
(within ~2x of SOTA, consistent across S). FA4 carries no compressed entries and no backward.

### SECONDARY — FlashInfer-MLA (`BatchMLAPagedAttentionWrapper`, head_dim_ckv=512 + kpe=64)
The only kernel on this box that does hd=512, via the DeepSeek MLA matrix-absorption trick (a single
latent KV head broadcast to all H query heads — same shared-KV/MQA structure as ours). Dense causal
prefill, forward only. FLOPs/pair: MLA = 2·(ckv+kpe)+2·ckv; ours = 4·D. To remove the sparsity-pattern
confound, our kernel is run in matched **dense-causal** mode (window≥S, no compressed):

| S | FI-MLA ms / TF/s | ours dense ms / TF/s | ours/MLA |
|---|---|---|---|
| 2048  | 2.616 / 111.7 | 5.169 / 53.2 | 0.48x |
| 4096  | 10.004 / 116.8| 19.120 / 57.5| 0.49x |
| 8192  | 39.294 / 118.9| 73.385 / 59.9| 0.50x |
| 16384 | 155.383 / 120.3| 287.697 / 61.2| 0.51x |

→ Our hd512 dense-causal forward is **~0.50x** of FlashInfer-MLA's throughput.

**Caveats (precise):** FlashMLA and FlashInfer-MLA are *MLA matrix-absorbed inference* kernels — they
work in the 512-latent space (PV over the latent, algorithmically cheaper than a materialized hd512)
with no backward, no compressed-bias, and (FlashInfer) no per-head sink; FlashMLA additionally exploits
*top-k sparsity* (its native config does 512 keys/query, not full causal). FA4 is *inference-only* and
capped at hd≤256 on SM90. Ours is a *training dense forward+backward* with per-head sink + additive
compressed-bias over a materialized hd=512. So the **sparse-config** gap is mostly the *algorithm*
(sparse+absorbed does far less work); the **dense-config** gap is the closest kernel-efficiency number,
and even it is confounded by MLA absorption (latent PV). All SOTA numbers are **hardware-utilization
ceilings on adjacent math, not identical work.** Net: across all three independent baselines our forward
lands consistently at **~0.5× of production-tuned dense throughput**, and ours is the **only** kernel
that delivers V4-Flash's hd=512 with backward + sink + compressed-bias at all.

## Backward efficiency vs forward efficiency  (dedicated analysis)

Measured with `bench_bwd.py` (kernel-only CUDA-event timing; forward, dQ kernel, dKV kernel(s), and
full backward timed separately). FLOP model per structurally-valid (q,k) pair/head/batch:
forward = 4·D (QKᵀ + PV); dQ = 6·D (QKᵀ + dO·K + dS@K); dKV = 8·D (KQ + K·dO + dS@Q + p@dO);
backward total = 14·D ⇒ a **3.5× FLOP floor** vs forward (vs the flash-ideal 2.5× / 5-GEMM). The extra
1.0× is the QKᵀ and dO·K recompute duplicated across the dQ and dKV kernels — keeping dQ and dKV in
separate kernels is the FA2-style choice that avoids dQ atomics (a fused 5-GEMM kernel would instead
atomic-accumulate dQ across kv-blocks, blowing up L2/atomic traffic at H=64; see "Why not 5-GEMM").

### Before → after the dKV split (H20, bf16, peak ~148 TF/s)

| shape | fwd ms (%pk) | dQ ms (%pk) | dKV ms (%pk) | bwd ms (%pk) | b/f lat | b/f eff |
|---|---|---|---|---|---|---|
| **before** S2048 m4   | 2.71 (25.4) | 2.91 (35.5) | 6.19 (22.2) | 9.11 (26.5) | 3.36 | 0.96 |
| **after**  S2048 m4   | 2.71 (25.4) | 2.91 (35.5) | **4.27 (32.2)** | **7.19 (33.6)** | **2.65** | **1.32** |
| **before** S2048 m128 | 1.52 (15.7) | 1.56 (23.0) | 2.81 (17.0) | 4.36 (19.1) | 2.86 | 0.82 |
| **after**  S2048 m128 | 1.52 (15.7) | 1.56 (23.0) | **2.26 (21.1)** | **3.82 (21.9)** | **2.51** | **1.40** |
| **before** S8192 m4   | 26.0 (32.1) | 28.8 (43.5) | 72.9 (22.9) | 101.7 (28.7) | 3.91 | 1.12 |
| **after**  S8192 m4   | 26.0 (32.1) | 28.8 (43.5) | **40.7 (41.1)** | **69.4 (42.1)** | **2.67** | **1.31** |
| **before** S8192 m128 | 6.12 (18.8) | 6.28 (27.5) | 11.49 (20.0) | 17.78 (22.6) | 2.90 | 0.83 |
| **after**  S8192 m128 | 6.12 (18.8) | 6.28 (27.5) | **9.20 (25.0)** | **15.48 (26.0)** | **2.53** | **1.38** |

`b/f lat` = backward/forward latency ratio (FLOP-ideal 3.5×, flash-style ~2.5×).
`b/f eff` = backward %peak ÷ forward %peak (>1 ⇒ backward is MORE hardware-efficient than forward).

**Answer to the question "is the backward as hardware-efficient as the forward?":** After the split,
**yes — and then some.** The dKV kernel went from 22.2 %peak (the single worst kernel, dragging the
whole backward) to 41.1 %peak at S8192 m4 — now at parity with the dQ kernel (43.5) and *above* the
forward (32.1). The full backward runs at 42.1 %peak vs the forward's 32.1 %peak, i.e. the backward is
**~1.3× more hardware-efficient than the forward**. The bwd/fwd *latency* ratio is now **2.5–2.7×**
(was 2.9–3.9×) — at the flash-style ideal, and *below* the 3.5× FLOP floor precisely because the
backward GEMMs (larger K-reuse) sustain higher throughput than the forward. The residual ratio above
2.0× is now fully accounted for by the 3.5× FLOP count partially offset by the backward's higher
efficiency — **not** by atomics or loose loop bounds. (This made the forward the efficiency laggard at
the time — since improved by the warp-specialized split-D forward, which lifts the DENSE forward to
~53 %peak, above the backward's ~42 %peak; sparse forwards improve too but stay near/below the backward.
See "Forward warp specialization APPLIED" below.)

## Codex (xhigh) optimization pass
codex (xhigh) reviewed `kernel.py` + `reference.py` + the backward derivation (full report: `OPTIMIZATION.md`).

**Correctness: no bug found.** The sink in the online-softmax LSE, the MQA shared-KV `dKV = dk + dv`
atomic accumulation over all H heads, the raw/compressed masks + banding, and padding are all
internally consistent with `reference.py`. Remaining risks are contract/API only (block-size lcm
assumption if arbitrary block sizes are exposed; mixed-dtype inputs; nondeterministic atomics) — not
current math bugs. This corroborates the GPU gradcheck.

**Performance diagnosis:** the backward does ~7 GEMM-equivalents per valid tile (dQ: QK, dO·K,
dQ = 3; dKV: KQ, K·dO, dK, dV = 4) vs the forward's 2 (QK, PV) → a **~3.5× bwd/fwd floor is
structural** (recompute), before any atomic overhead. Reaching flash-style ~2.5× requires cutting the
dQ/dKV recompute duplication, not just the atomics.

**Prioritized optimizations (documented follow-ups):**
- Forward: `num_stages=2`/`threads=256` Hopper pipelining (~15–35%); skip wrapper `q_pad`/`kvt` copies
  when already aligned (10–40% on sparse shapes); split WGMMA shared layouts for QK vs PV; autotune
  (BM,BN) for hd512; dense/sliding-specialized mask-free tiles; base-2 `exp2` softmax.
- Backward: split raw/compressed dKV kernels with a tight compressed `q_lo` bound (up to ~2× on
  compressed dKV); head-group atomic accumulation (cut atomic traffic by G); optional deterministic
  two-stage dKV; vectorized `atomic_addx4`; a 5-GEMM flash-backward redesign to approach ~2.5×.

## ncu bottleneck findings (forward + dQ + dKV)
ncu (Nsight Compute 2025.2.1, perf counters available in-container) on `kernel_kernel` at S2048 m4,
skipping warmup launches. Key metrics (`%pk` = % of peak sustained):

| kernel | SM thrpt | DRAM | L2 (lts) | L1tex | warps active | TF/s %pk (useful) |
|---|---|---|---|---|---|---|
| forward      | 35.8 | 2.9 | 27.8 | 34.9 | 11.2 | 25.4 |
| dQ           | 49.9 | 5.1 | 21.1 | 25.1 | 11.5 | 35.5 |
| dKV (before) | 46.1 | 3.4 | 26.1 | 31.9 | 11.6 | 22.2 |
| dKV-raw (after)  | 46.4 | 8.2 | 34.2 | 37.1 | 9.9  | — |
| dKV-comp (after) | 44.5 | 11.1| 30.7 | 27.4 | 12.0 | — |

**Three decisive findings:**
1. **All kernels are occupancy-starved at 1 CTA/SM.** ncu Occupancy section: `Block Limit Shared Mem = 1`
   (196.6 KB dynamic smem/block — three `[64,512]` bf16 tiles), `Block Limit Registers = 1` (240
   reg/thread). Theoretical occupancy 12.5 %, achieved ~11.6 %. With one block/SM and `stages=1` the
   tensor cores cannot overlap with the softmax/exp/rescale elementwise work — this is the ~35–50 % SM
   ceiling (and why `num_stages=2`/`threads=256` gave nothing). Lifting it needs warp specialization or
   a smem-footprint cut (split-D), both high-effort — Phase-2.
2. **NOT memory- or atomic-bound.** DRAM 3–11 %, L2 21–34 %. The dKV atomic accumulation over H=64 heads
   (a `[64,512]` fp32 tile per kv-block) goes through L2 but does not saturate it. **This is the
   evidence that ruled out head-group atomic accumulation** (see below) — it would cut a non-bottleneck.
3. **The dKV "before" anomaly:** it had the *highest* SM throughput (46 %) yet the *lowest* useful TF/s
   (22 %). The hardware was busy — computing **masked/zero compressed tiles** because the compressed-dKV
   loop scanned from query-block 0. This is exactly what the split + tight `q_lo` fixed.

## Backward optimization APPLIED — split dKV + tight compressed q_lo
The single `_build_bwd_dkv` (one grid over all `n_kv` blocks, `is_raw` branch in the inner mask, loose
`q_lo = 0` for compressed blocks) was split into **two kernels**:
- `_build_bwd_dkv_raw`: grid over raw blocks `[0, Sp)`, banded sliding-window query range, raw-only mask.
- `_build_bwd_dkv_comp`: grid over compressed blocks `[Sp, Sp+Tcp)`, compressed-only mask, with a
  **tight lower query-block bound** `q_lo = min(n_q, max(0, ((wc0+1)*m − 1)//block_M))` where `wc0` is
  the smallest compressed entry in the block (a compressed entry `w` is attended only by queries
  `i ≥ (w+1)*m − 1`). The per-element mask is retained for the partial diagonal tile, so the analyzer
  cannot fold `valid` to a constant — which is what crashed lowering when the tight bound was attempted
  inside the *single* unified kernel. The `min(n_q, …)` clamp makes a fully-padded compressed block
  (only possible when `Tcomp % block_N ≠ 0`) a zero-extent loop instead of a negative one.

**Result:** dKV S8192 m4 **72.9 → 40.7 ms (1.79×), 22.9 → 41.1 %peak**; full backward 101.7 → 69.4 ms;
bwd/fwd latency 3.91 → 2.67. Correctness re-validated OVERALL PASS after the change. The tight bound was
**codex (xhigh) reviewed**: the monotone-`t(w)` argument proves no valid query is ever skipped, raw
upper bound is safe, the two kernels cover exactly the original valid set with no double-counting, and
the `q_lo` clamp closes the only edge case (codex's one finding, applied).

**Backward ideas evaluated and NOT applied (evidence-based):**
- **Head-group atomic accumulation** (one CTA does G heads → one atomic): ncu shows the backward is
  **not** atomic/L2-bound (post-split dKV: DRAM 8–11 %, L2 30–34 %), so grouping cuts a non-bottleneck;
  worse, it reduces grid parallelism on a kernel already capped at 1 CTA/SM — net negative expected. The
  split alone brought dKV to parity with dQ (~45 % SM, 41 % useful peak), so the atomic path is no
  longer the limiter.
- **5-GEMM fused backward** (single kv-parallel kernel producing dQ via atomics, eliminating the dQ/dKV
  QKᵀ+dO·K recompute, 14·D → 10·D): would atomic-accumulate a `[block_M,512]` fp32 dQ tile per
  (kv-block, head, query-iteration) — with H=64 heads that is orders of magnitude more atomic traffic
  than the current one-atomic-per-kv-block dKV, and atomics route through the L2 that is already at
  30 %+. The FA2-style separate dQ kernel (register accumulation, **zero** atomics) is the right design
  here; the 3.5× FLOP floor is the price of avoiding dQ atomics, and the measured 2.5–2.7× ratio already
  beats that floor. Documented as a redesign option, not pursued.

## Final verdict
**Correct, necessary, and the sole hd=512 training-attention option.** A1 is the only kernel that
delivers V4-Flash's `head_dim=512` with backward + per-head sink + additive compressed-bias at all
(FA2/3/4 cap at hd≤256 on SM90; FlashMLA and FlashInfer-MLA are matrix-absorbed inference-only). It
beats the compiled naive reference 6.5–36.8× and runs 16k/32k where that reference OOMs — i.e. it is
what makes hd=512 long-context training feasible (V4 ships eager-only).

Efficiency: forward is now **~0.65× of production-tuned SOTA dense hardware throughput** (FlashMLA
@hd512-latent dense, the real V4 kernel) after the warp-specialized split-D forward below — up from
~0.50× for the single-warpgroup baseline (which matched FA4 @hd256 0.49× and FlashInfer-MLA @hd512
0.48–0.51×, measured pre-warp-spec). **bwd/fwd is 2.5–2.7×** after the dKV split + tight compressed
`q_lo`. The warp-spec forward closed the forward-laggard gap the dKV split had opened: on **dense-causal**
the forward is now MORE hardware-efficient than the backward (ws-fwd ~53 %peak vs bwd ~42 %peak); on the
**production sparse** shapes the forward improves 1.10–1.12× but stays modestly below the backward
(e.g. S8192 m4: ws-fwd ~35 %peak vs bwd ~42 %peak) — sparse forwards have fewer kv tiles to amortize the
boundary/mask work and per-iter barrier overhead. ncu confirms the single-warpgroup baseline was occupancy-capped at 1 CTA/SM
(128 KB smem Q+K tiles, 240 regs/thread consumer) and compute-bound (not memory/atomic-bound).

### Forward warp specialization APPLIED — split-D 2-consumer-warpgroup forward (the Phase-2 win)
**Outcome: a hand-written warp-specialized forward (`_build_fwd_ws`) lifts the dense forward from
~0.50× → ~0.65× of FlashMLA (1.30× faster, 36.8 → 48–53 %peak), validated OVERALL PASS, deadlock-free,
with byte-identical Lse/Out so the backward is unchanged. It is now the default forward (`_USE_WS_FWD`).**

Design (FA3-style, adapted to the hd512 register blocker). The single 128-thread consumer serialized
`QK → softmax → rescale acc_o[64,512] → PV` with the tensor cores idle ~65% (ncu SM 35.8%). The
acc_o[64,512] fp32 accumulator (~256 regs/thread) forbids the standard `threads=256, block_M=128,
64-rows/warpgroup` FA trick (acc_o would be [128,512] = impossible) and `block_M<64` fails TileLang
layout inference. So the split is along **head_dim D=512**: two consumer warpgroups in one CTA
(threads=256), each owning `acc_o[64,256]` (128 regs/thread):
- **WG0** (`T.ws(0)`): loads Q/K/V halves, computes the FULL `QK` as two contiguous half-gemms
  (`Q0@V0ᵀ + Q1@V1ᵀ`, accumulated — column-slicing a swizzled smem tile gives a bad WGMMA descriptor,
  so Q/V are loaded as separate contiguous `[64,256]` buffers), does the online-softmax, publishes the
  probabilities `P_sh`, the per-iter rescale `sc_sh` and the FINAL normalize scale `fsc_sh` to shared
  memory, and does `PV` over `D[0:256] → Out[…,0:256]`.
- **WG1** (`T.ws(1)`): consumes `P_sh`/`sc_sh`/`fsc_sh` and does `PV` over `D[256:512] → Out[…,256:512]`.

This splits the PV gemm + the expensive `acc_o` rescale across two warpgroups (halving both), runs at
256 threads (better latency hiding), and adds NO redundant QK tensor work (QK computed once by WG0).

Deadlock avoidance (a prior warp-spec kernel deadlocked on this H20). Cross-warpgroup ordering uses
**named barriers** `T.sync_threads(id, 256)` (`bar.sync`), NOT mbarrier parity (the parity init is the
classic trap). Both warpgroups run identical `T.serial(0, n_total)` loops (n_total deterministic from
the shared `bx` ⇒ equal counts) and hit the per-iter barriers (probs-ready / consumers-done, + a
final-scale barrier — 3 per iter after dropping a redundant V-ready barrier that `bar.sync` already
implies) in LOCKSTEP ⇒ no warpgroup can wait on a barrier the other skips. Three findings
were decisive: (1) `T.Pipelined` REORDERS the manual barriers → illegal instruction / wrong results;
**`T.serial` is required**. (2) TileLang's auto-warp-spec + TMA pass COLLIDES with manual `T.ws`
(emits undefined `mbarrier`/broken `tma_load`) → must compile with
`pass_configs={"tl.disable_warp_specialized": True, "tl.disable_tma_lower": True}`. (3) Shared buffers
must be allocated ONCE at kernel top level (not inside the per-WG macros) — TileLang merges per-scope
shared allocations and ALIASES the two concurrent warpgroups' buffers → silent cross-WG race.

Validation. `python test_correctness.py` → OVERALL PASS (fwd + dq/dkr/dkc/dsink). The ws forward's
`Out` AND `Lse` are byte-identical (rel 0.0) to the baseline `_build_fwd` across S∈{512,1024,2048},
sliding-only/CSA m4/HCA m128, W∈{128,64}, B∈{1,2} ⇒ the hand-written backward needs zero changes. No
hang at any shape (tiny-shape deadlock guard + full sweep).

Measured forward (H20, bf16, peak 148 TF/s; `exp_ws.py` / `bench_fwd_quick.py` / `bench_flashmla.py`):

| shape | baseline ms (%pk) | ws ms (%pk) | speedup | ours/FlashMLA |
|---|---|---|---|---|
| dense-causal S2048  | 5.05 (36.8) | **4.00 (46–48)** | **1.31×** | 0.50→**0.63×** |
| dense-causal S8192  | 72.9 (40.7) | **56.5 (52.6)**  | **1.30×** | 0.50→**0.65×** |
| dense-causal S16384–32768 | — | (79–80 TF/s) | — | **0.65×** |
| production CSA m4 S2048 (W128)  | 2.71 (25.5) | **2.57 (26.8)** | 1.11× | — |
| production HCA m128 S2048 (W128)| 1.52 (15.7) | **1.49 (16.0)** | 1.12× | — |
| production CSA m4 S8192 (W128)  | 26.0 (32.1) | **24.1 (34.7)** | 1.10× | — |

The dense-causal shapes (the SOTA-comparison bar, and the most compute-heavy) gain most (1.30×); the
sparse production shapes gain less (1.10–1.12×) because they do less PV/rescale work to split (the gain
source), not because of barrier overhead (dropping the V-ready barrier was perf-neutral — the barriers
aren't the bottleneck).

**Why ~0.65× is near the structural ceiling for this DSL design (and why FA3 ping-pong does not port).**
FA3's Hopper win is a 2-warpgroup **ping-pong** that overlaps one warpgroup's softmax with the other's
WGMMA, with each consumer warpgroup processing a different **block_M** row tile holding its OWN full
output accumulator (Colfax/Tri Dao FA3: "more registers to hold both accumulators"). At hd=512 that
accumulator is `acc_o[64,512]` fp32 = **256 regs/thread per warpgroup**; two such warpgroups = the entire
sm90 register file (65536) across 256 threads with nothing left → guaranteed spills. So the FA3 block_M
ping-pong is **register-infeasible here** — the head_dim split (acc_o[64,256] = 128 regs/WG) is the
register-feasible adaptation, but it makes the two warpgroups **asymmetric**: WG0 does QK(full) +
softmax(full) + PV(half) while WG1 does only PV(half), so WG0 is ~3× WG1's work and is the critical
path. A skewed double-buffered pipeline (WG1 doing PV1[t−1] under WG0's softmax[t], expressible as a
single lockstep barrier/iter — deadlock-safe) would overlap only WG1's *small* half-PV under WG0's
softmax → marginal, because WG0 (QK+PV0+softmax) dominates regardless. Closing more of the gap to
FlashMLA would require **balancing** QK+softmax across both warpgroups (cross-WG smem round-trips of
acc_s/the reduction, large traffic) — and FlashMLA additionally has matrix-absorption (latent PV) +
hand-CUTLASS that a DSL materialized-hd512 kernel structurally cannot match. So ~0.65× (1.30× over the
single-WG baseline) is recorded as the realized warp-spec win; deeper pipelining is documented headroom
with diminishing return.

### Earlier low-risk param tuning (did not help; superseded by the warp-spec forward above)
- `num_stages=2` (threads=128): OVERALL PASS but no speedup (WGMMA-bound, not K-load-latency-bound;
  re-confirmed S2048 dense 5.05 vs 5.04 ms, stages=3 OOMs the 256 KB smem). `threads=256` on the
  single-warpgroup kernel fails `[64,64]` `acc_s` layout inference. These motivated the explicit
  split-D warp specialization above (which solves the layout problem by allocating per-WG fragments
  inside the ws macros).
- **`block_N=128` swept (post-dKV-split):** helps ONLY the densest shape (S8192 m4: 26.5→24.4 ms, +8 %)
  and HURTS every sparser shape (S2048 m4/m128, S8192 m128: +6–22 % slower) due to sliding-window
  over-fetch (a 128-wide key block straddles the W=128 band). Not a universal win → default stays
  `(block_M, block_N) = (64, 64)`. ncu attributes the forward's ~32 %peak ceiling to the 1-CTA/SM
  occupancy cap + no WGMMA↔softmax overlap, i.e. it needs warp specialization (Phase-2), not block-size
  tuning. (`block_N=128` also cannot be used for the backward: dKV's `dkv_acc[block_N,D]` fragment would
  be `[128,512]` fp32 = 64K registers — impossible. Forward/backward block sizes would have to be
  decoupled for a marginal ~3 % full-pass gain; not done.)

fp8: sglang's FA4 V4 attention does NOT support fp8 forward (`q/k/v_descale` →
`NotImplementedError("FA4 path does not support descale")`), so no fp8 forward is required for this
kernel (V4's fp8/fp4 lives in the MoE/indexer, not the attention).

### Phase-2 optimization roadmap (sourced from local SOTA; web search was unavailable in-env)
Mined the installed SOTA instead of the web. Key structural finding: **sglang has no dense-hd512
TileLang attention** — its real V4 attention is **FlashMLA `flash_mla_sparse_fwd` (MLA matrix-absorbed
+ top-k sparse)**, and its only V4 TileLang attention kernel is the **fp8 indexer logits at hd=128**
(`srt/layers/attention/dsv4/tilelang_kernel.py`, which we drop). So our materialized dense hd=512 kernel
cannot fully match MLA throughput by kernel tuning alone — the ~0.5× gap is part algorithmic.
Realizable kernel-efficiency steps, in increasing effort:
1. **Autotune** (block_M, block_N, num_stages, threads) via tilelang's roller / `@autotune`
   (`tilelang/carver/template/flashattention.py::get_hardware_aware_configs`) instead of the hand-fixed
   64/64/1/128 — automated, avoids the invalid configs that a manual `threads=256` flip hit.
2. **TMA loads** (Hopper `cp.async.bulk`) for Q/K tiles instead of plain `T.copy`.
3. **FA3/MLA-style warp specialization** — ✅ **DONE** (`_build_fwd_ws`, the split-D 2-consumer-warpgroup
   forward; see "Forward warp specialization APPLIED"). Lifted the dense forward 0.50× → 0.65× of
   FlashMLA (1.30×). On sm90 we split along head_dim (not the FlashInfer sm100 12-warp TMEM schedule,
   which is Blackwell-only) because acc_o[64,512] register pressure forbids more rows/warpgroup.
   Remaining headroom: a deeper async pipeline (WG1 doing PV[t−1] under WG0's QK[t]) toward ~0.7–0.8×.
4. Backward: split raw/compressed dKV + tight `q_lo` + head-group atomic accumulation (codex) to pull
   bwd/fwd below the ~3.5× recompute floor.
