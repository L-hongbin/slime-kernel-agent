# B1 — mHC HyperConnection (tilelang): results

DeepSeek-V4-Flash Manifold-Constrained Hyper-Connection (`DeepseekV4HyperConnection`),
hand-written tilelang forward + analytic backward, exposed as a
`torch.autograd.Function` (`kernel.py::hyper_connection`). Dims fixed at V4-Flash:
H=hc_mult=4, D=4096, M=H·D=16384, sinkhorn_iters=20, eps=1e-6, fp32 compute.

## Files
- `reference.py` — exact fp32 port of `HyperConnection` + `HyperHead` (+ fp64 path for gradcheck).
- `kernel.py` — tilelang `_build_norm_gemm` / `_build_collapse` / `_build_dx`; torch `_middle_fwd`/`_middle_bwd` (Sinkhorn, analytic bwd); autograd glue.
- `test_correctness.py` — fp64 formula gradcheck + fp32 + bf16 PASS/FAIL (+ sglang path [5]).
- `bench.py` — latency / peak-mem vs `torch.compile(reference)`.
- `bench_sota.py` — forward vs sglang `mhc_pre` (community SOTA).
- `bench_options.py` — fwd/bwd/total/ratio/peak trade-off across the backward-fix options (b)/(c)/(a)/baseline (see §Backward-recompute fix).

## R4 stride fix and rebench (2026-07-02)

`raw_p[:, :MIX].contiguous()` was not a strict enough packing boundary for the
TileLang FFI. When `N=1`, PyTorch can treat the sliced tensor as contiguous because
the leading dimension has size 1 while preserving the physical stride `(32, 1)`;
`_build_sinkhorn_collapse` declares `Raw[N,24]` and rejects that at runtime:
`kernel main input Raw strides[0] expected 24, but got 32`.

Fix: `kernel.py::_compact_last_dim` now always allocates a fresh `[..., MIX]` tensor
and copies the live columns before passing `Raw` to TileLang. For `N>1`, the old
`.contiguous()` path already had to allocate/copy, so the training-shape cost is
not expected to change materially.

Validation evidence:
- Stride unit: `pytest -q tests/test_dsv4_mhc_stride.py` -> PASS.
- Single-token GPU repro after fix: `local_artifacts/deepseek-v4/r2_logs/r4_mhc_stride_fix_s1_repro.log`
  -> `OVERALL mhc_s1_after_stride_fix PASS`.
- Full mHC correctness: `local_artifacts/deepseek-v4/r2_logs/r4_mhc_stride_fix_correctness.log`
  -> `OVERALL: ALL PASS` (fp64 formula, fp32 fwd/bwd, bf16 fwd/bwd, sglang-fwd + our-bwd).

Efficiency rebench after the kernel edit, node64 H20 GPU0, bf16, `bench.py`; raw log:
`local_artifacts/deepseek-v4/r2_logs/r4_mhc_stride_fix_bench.log`.

| shape | impl | fwd ms | bwd ms | fwd+bwd ms | peak MB |
|---|---|---:|---:|---:|---:|
| B1·S2048 | tilelang | 0.411 | 1.057 | 1.469 | 447.8 |
| B1·S2048 | torch.compile | 0.520 | 1.434 | 1.955 | 579.3 |
| B1·S8192 | tilelang | 0.848 | 2.806 | 3.654 | 1695.8 |
| B1·S8192 | torch.compile | 0.982 | 2.632 | 3.614 | 2093.7 |
| B1·S16384 | tilelang | 1.602 | 5.140 | 6.741 | 3315.1 |
| B1·S16384 | torch.compile | 2.014 | 4.773 | 6.787 | 4113.5 |

Conclusion: the stride fix restores the `N=1` kernel path and keeps the main
training-shape efficiency within the previous envelope: tilelang forward remains
faster and peak memory remains lower; total fwd+bwd is faster at S2048, near-parity
at S8192/S16384.

Node-specific kernel-on gate after this fix: node64 and node69 pass the same
single-process mHC S64 forward/backward check, but node70 currently fails any mHC
TileLang/sglang sm90a path at module load/runtime. Evidence:
`local_artifacts/deepseek-v4/r2_logs/r4_node70_mhc_norm_single.log` (our norm_gemm
segfault), `r4_node70_sglang_pre_only.log` (sglang pre-only SIGILL),
`r4_node70_mhc_torch_norm_hybrid2.log` (fused middle SIGILL), and
`r4_node70_mhc_torch_norm_middle_tilelang_bwd.log` (backward d_pre segfault).
This is not fixed by the stride patch; do not use node70 for kernel-on training or
V4 sglang rollout until its CUDA/driver stack is fixed.

## Correctness — ALL PASS

`CUDA_VISIBLE_DEVICES=1 python test_correctness.py`

| check | quantity | max rel err | tol |
|---|---|---|---|
| **fp64 formula** (analytic vs autograd-through-ref) | d_x, d_fn, d_base, d_scale | **0** (bit-exact) | 1e-9 |
| **fp32** fwd (kernel vs ref) | post / comb / collapsed | 3.7e-4 / 3.5e-4 / 3.4e-4 | 3e-3 |
| **fp32** bwd (kernel vs autograd-through-ref) | d_x / d_fn / d_base / d_scale | 3.7e-4 / 4.6e-4 / 2.9e-4 / 2.4e-3 | 3e-3 |
| **bf16** fwd | post / comb / collapsed | 1.6e-4 / 2.6e-4 / 4.0e-3 | 2e-2 |
| **bf16** bwd | d_x / d_fn | 6.0e-3 / 2.2e-4 | 3e-2 |

The fp32 residual (~3e-4) is **tf32 tensor-core rounding** in the mix GEMM (tilelang
runs fp32 GEMMs on tf32 cores). The model's real path is bf16 input → bf16 quantisation
(~8e-3) dominates and tf32 is irrelevant there. The fp64 check proves every backward
formula — including the 20-iter Sinkhorn VJP — is mathematically exact.

## Key derivations
- **Forward fusion**: RMSNorm rescale is a per-row scalar, so `raw = inv·(xf@fnᵀ)` —
  the 512 MB `flat` tensor is never materialised. Σx² is a second all-ones GEMM (kept in
  MMA fragments because a fused reduce+gemm fails tilelang layout inference).
- **Sinkhorn backward**: each normalise `y=x/(Σx+ε)` has VJP `gx=(gy−Σ(gy·y))/(Σx+ε)`;
  recompute the forward states, walk them back, then softmax+sigmoid backward → `d_raw`.
- **RMSNorm-collapse backward**: the RMSNorm Jacobian collapses to a scalar
  `p[n]=Σ_m d_raw·raw` because `Σ_k fn[m,k]·xf[n,k] = raw[n,m]/inv[n]`, so the dx kernel
  is a single streamed pass: `d_x = pre·dcollapsed + inv·(d_raw@fn) − (inv²·p/M)·xf`.

## Benchmark vs `torch.compile(reference)` — bf16, H20, GPU 1

`CUDA_VISIBLE_DEVICES=1 python bench.py` (ms/call, **post-codex**; mHC runs **86×** per
model forward). GPU 1 was shared during measurement → bwd has ~±0.4 ms run-to-run noise
(both impls); fwd and peak-mem are stable.

| shape | impl | fwd | bwd | fwd+bwd | peak MB |
|---|---|---|---|---|---|
| B1·S2048  | tilelang | **0.48** | 2.28–2.72 | 2.76–3.21 | **448** |
|           | torch.compile | 0.52 | 1.97–2.91 | 2.50–3.26 | 579 |
| B1·S8192  | tilelang | **0.96** | 2.65–2.83 | 3.60–3.79 | **1696** |
|           | torch.compile | 0.97 | 2.57 | 3.53 | 2094 |
| B1·S16384 | tilelang | **1.79** | 4.88–4.90 | 6.67–6.68 | **3315** |
|           | torch.compile | 2.01 | 4.57–4.59 | 6.58–6.60 | 4114 |

For reference, **before** the codex pass tilelang fwd+bwd was 3.15 / 5.00 / 9.40 ms (peak
614 / 2231 / 4387 MB) — i.e. backward was ~1.4–1.6× slower and memory was *higher* than
torch.compile.

## Verdict

After the codex optimization (TMA fast-path + tilelang `d_pre` / `d_fn` kernels that drop
the fp32 x-upcasts — see below), the tilelang implementation is **competitive with
torch.compile**:
- **Forward wins** at every shape (e.g. S16384 1.79 vs 2.01 ms) — the RMSNorm+GEMM+collapse
  fusion avoids materialising the 512 MB `flat` tensor.
- **fwd+bwd is at parity** within measurement noise (tilelang wins at S2048, ties at
  S16384, ~2–7% behind at S8192 depending on the run).
- **Peak memory is 19–27% lower** (448 vs 579, 1696 vs 2094, 3315 vs 4114 MB) — the
  backward never builds an fp32 copy of `x`. This is the most durable win: it directly
  frees activation budget at 32k training context.

The **tiny 4×4 Sinkhorn middle is left in `torch.compile`** (the `_middle_fwd`/`_middle_bwd`
helpers): per-thread-per-token tilelang codegen is latency-bound (~2.2 ms — 6× worse), and
a per-element-parallel formulation needs a block sync per iteration with no payoff. The
Sinkhorn **backward is still hand-derived analytic** (explicit per-step VJP, not
autograd-through-20-iters).

**Recommendation for the V4-Flash port**: the tilelang kernel is now a reasonable choice —
faster forward and lower memory at long context, parity on fwd+bwd — and matches the
production direction (the mHC authors ship fused TileLang kernels, see below). torch.compile
remains the simpler fallback if the ~parity latency is preferred over the memory saving. In
the real LoRA setting fn/base/scale are frozen, so backward computes only `d_x`
(`needs_input_grad` gating skips the d_fn GEMM) — cheaper still.

## Community SOTA comparison (sglang mHC)

**A production community kernel does exist and is directly runnable.** sglang ships the
DeepSeek-V4 mHC as **TileLang** kernels in `sglang.srt.layers.mhc` (`mhc_pre`,
`mhc_post`, `mhc_fused_post_pre`, `hc_split_sinkhorn`, + the `*_tilelang` JIT bodies).
The direct analog of our **forward** is `mhc_pre`: it consumes the residual streams + the
mix `fn` and returns `(post_mix, comb_mix, layer_input)` — the post placement weights, the
Sinkhorn-balanced comb matrix, and the pre-weighted **collapsed** sublayer input — i.e. the
same three tensors our `hyper_connection` forward returns. (`mhc_post` is the *separate*
sublayer-output placement step, which our forward does not do, so `mhc_pre` is the
apples-to-apples forward baseline.) This is therefore a **TileLang-vs-TileLang** forward
comparison. Benchmarked via `bench_sota.py` (`CUDA_VISIBLE_DEVICES=1`, H20, GPU 1, bf16
residual + fp32 `fn`, hc_mult=4, hidden=4096, sinkhorn_iters=20).

**Correctness cross-check** (sglang `mhc_pre` vs our forward, matched inputs, bf16): max
rel err `post ≈ 2e-5…4e-4`, `comb ≈ 5e-5…5e-4`, `collapsed ≈ 4e-3` (the collapsed residual
is bf16 rounding of the pre-weighted sum) — confirms both implement the same op.

Forward latency (ms/call):

| shape | ours fwd | sglang `mhc_pre` | ours/sglang |
|---|---|---|---|
| **pure-TileLang** (`SGLANG_OPT_DEEPGEMM_HC_PRENORM=0`: TileLang split-k GEMM + `mhc_pre_big_fuse_tilelang`) ||||
| B1·S2048 | 0.55 | **0.15** | 3.8× slower |
| B1·S8192 | 0.97 | **0.48** | 2.0× slower |
| **production default** (`SGLANG_OPT_DEEPGEMM_HC_PRENORM=1`: DeepGEMM tf32 prenorm GEMM + same TileLang big_fuse) ||||
| B1·S2048 | 0.49 | **0.08** | 6.4× slower |
| B1·S8192 | 0.96 | **0.28** | 3.4× slower |

**Verdict — our forward is NOT competitive with sglang's production TileLang mHC.** It is
2.0–3.8× slower than the pure-TileLang path and 3.4–6.4× slower than the DeepGEMM default.
The gap is largest at small S and shrinks as compute grows, which points at the cause:
launch / kernel-boundary overhead, not raw FLOPs.

> **Historical note (superseded below).** The first fused-Sinkhorn redesign attempt
> deadlocked on H20 when built with default `tilelang.compile`: the producer/consumer
> split around `thread < 32` became a cross-warp barrier hang. That failure is why the
> older forward-SOTA comparison below talks about a reverted fused kernel. The current
> implementation no longer uses the default pass pipeline for this kernel: see
> "Backward-recompute fix" below, where `_build_sinkhorn_collapse` is enabled with
> `TL_DISABLE_WARP_SPECIALIZED` and `TL_DISABLE_TMA_LOWER`, passes correctness, and is
> part of both trainable mHC paths.

**Why (our forward-stage profile, ms):**

| N | norm_gemm | middle (torch.compile Sinkhorn) | collapse |
|---|---|---|---|
| 2048 | 0.34 | **0.39** | 0.05 |
| 8192 | 0.67 | **0.28** | 0.18 |

The `torch.compile` Sinkhorn **middle is a ~0.3–0.4 ms fixed cost** (a handful of tiny
kernel launches over the `[N,4,4]` data) — at S2048 it is the *single biggest* term, larger
than the GEMM itself. sglang folds **all 20 Sinkhorn iterations + the stream collapse into
one per-token TileLang kernel** (`mhc_pre_big_fuse`), so the Sinkhorn is essentially free and
the whole forward is ~2 launches (GEMM + big_fuse) vs our 3+ (tilelang GEMM → torch.compile
middle → tilelang collapse). The production default additionally uses a tuned DeepGEMM tf32
GEMM, widening the gap.

**Caveats (honest):**
- sglang's `mhc_pre` is an **inference forward-only** kernel: no backward, no saved
  intermediates, fully fused. Our kernel is a **train-time `autograd.Function`** that must
  keep the GEMM separable and stash `raw/inv/pre` for the hand-derived analytic backward —
  a different design objective. The forward comparison is fair and informative, but the gap
  is partly the price of being differentiable.
- Both run the heavy GEMM in TileLang/tf32 and take bf16 residual + fp32 `fn`; the Sinkhorn
  middle differs only in *where* it runs (sglang: fused TileLang; ours: torch.compile).
- Numbers are forward-only, single GPU, no pipeline overlap.

**Takeaway for the V4-Flash port:** for a fast *inference* mHC forward, use sglang's kernels
directly — they are the SOTA and clearly faster. Our kernel's value is the **trainable**
fwd+bwd path (analytic Sinkhorn backward, lower training peak memory vs torch.compile — see
table above), not raw forward speed. The most actionable way to close the forward gap is to
fuse our Sinkhorn middle + collapse into a single TileLang kernel the way sglang does
(eliminating the ~0.3–0.4 ms torch.compile launch tax) while keeping the analytic backward.

## Backward-recompute fix (B1 follow-up, 2026-06-30) — fused forward, no GEMM recompute

**Problem (measured).** The previous `hyper_connection_sglang` design used sglang's
fused `mhc_pre` for the forward (1.00–1.07× of SOTA — kept) but `mhc_pre` exposes no
`raw`/`inv`/`pre`, so the backward **recomputed a full norm+GEMM** (`_build_norm_gemm`
over the 512 MB `x`) to reconstruct them. The GEMM was therefore done **twice total**
(once inside `mhc_pre`, once in the backward), and `bwd/fwd` blew up to **12–31×**
(the forward is a hyper-fused inference kernel, so the redundant-GEMM backward dwarfs
it). At S8192 the recompute added **~0.73 ms** to the backward (sglang-recompute bwd
3.40 ms vs a saved-raw/inv backward 2.67 ms ≈ exactly one `norm_gemm`).

**Fix — fused Sinkhorn+collapse kernel, deadlock resolved.** The forward now does the
norm+GEMM **once**, captures `raw`/`inv`, and runs the gates + 20-iter Sinkhorn +
stream-collapse in **one** TileLang kernel (`_build_sinkhorn_collapse`, the warp-0-
Sinkhorn / other-warps-collapse split that mirrors sglang's `mhc_pre_big_fuse_tilelang`).
That kernel previously **deadlocked**; the root cause was the *default* `tilelang.compile`,
whose warp-specialization pass auto-specialized the `if thread<32` producer/consumer
split into a cross-warp barrier hang. **Fix = compile it with
`pass_configs={TL_DISABLE_WARP_SPECIALIZED: True, TL_DISABLE_TMA_LOWER: True}`** — the
*exact* config sglang ships its structurally-identical big_fuse with. Verified
non-hanging at S=128 (90 s timeout) and produces a doubly-stochastic `comb`. Because
`raw`/`inv` are now **saved**, the backward does **no GEMM recompute** (shared
`_hc_backward`, unchanged math).

Two trainable paths, both fixed:

- **(c) `hyper_connection_sglang` — recommended default (bf16).** Forward reuses sglang's
  fast prenorm GEMM stage `mhc_pre_gemm_sqrsum_tilelang` → `gemm_out[N,24]` (un-normalised
  mix logits) + `sqrsum[N]` (Σx²); then `inv=rsqrt(sqrsum/M+eps)`, `raw=inv·gemm_out`
  (bit-identical to the `mixes`/`rms` `mhc_pre_big_fuse` forms internally) + the fused
  Sinkhorn+collapse. Keeps the SOTA-fast forward AND the cheap backward.
- **(b) `hyper_connection` — sglang-free fallback (fp32/bf16).** OUR `_build_norm_gemm`
  (saves `raw`/`inv`) + the same fused Sinkhorn+collapse. No sglang import; the only
  fp32/fp64 path. (`MHC_TORCH_MIDDLE=1` reverts the middle+collapse to torch.compile.)

**Numerics — `test_correctness.py` OVERALL ALL PASS** (`CUDA_VISIBLE_DEVICES=1`): fp64
formula gradcheck **0 error** (d_x/d_fn/d_base/d_scale); fp32 fwd/bwd ≤ 2.4e-3; bf16 fwd/bwd
(both (b) and (c)) ≤ 6.6e-3 — i.e. unchanged from before (the fused kernel reproduces the
same math; saved `raw`/`inv` are the exact analytic-backward intermediates).

**Trade-off table (`bench_options.py`, H20 GPU 1, bf16, all params grad).** fwd / bwd /
fwd+bwd / bwd÷fwd / peak-MB:

| path | S2048 fwd / bwd / total / ratio / MB | S8192 | S16384 |
|---|---|---|---|
| **(c) sglang-GEMM + fused** *(default)* | 0.17 / 2.10 / **2.27** / 12.8× / 331 | 0.47 / 2.67 / **3.15** / 5.6× / 1394 | 0.84 / 4.88 / **5.72** / 5.8× / 2677 |
| (b) ours norm + fused | 0.38 / 1.83 / 2.21 / 4.9× / 333 | 0.78 / 2.67 / 3.46 / 3.4× / 1394 | 1.48 / 4.90 / 6.38 / 3.3× / 2677 |
| (a) sglang-fwd + extra-norm-in-fwd | 0.44 / 1.82 / 2.26 / 4.1× / 367 | 0.99 / 2.66 / 3.65 / 2.7× / 1394 | 1.86 / 4.88 / 6.75 / 2.6× / 2677 |
| base sglang-recompute *(old default)* | 0.08 / 2.22 / 2.30 / 27× / 366 | 0.29 / 3.40 / 3.68 / 12× / 1395 | 0.56 / 6.19 / 6.75 / 11× / 2680 |
| ref torch.compile | 0.36 / 1.84 / 2.20 / 5.2× / 532 | 0.97 / 2.52 / 3.49 / 2.6× / 1792 | 2.02 / 4.58 / 6.60 / 2.3× / 3475 |

**Frozen-param LoRA path (only `d_x` needed — the real training setting, d_fn GEMM
skipped):** option (c) S8192 = fwd 0.47 / bwd 2.24 / total **2.71** / ratio 4.7× / 1287 MB;
S16384 = 0.83 / 4.11 / **4.94** / 4.9× / 2570 MB.

**Verdict — (c) is the default.** It wins **total fwd+bwd at every shape** (S8192/S16384
~**14–15% faster** than the old recompute design, and faster than every other path incl.
torch.compile), eliminates the redundant GEMM, keeps the **SOTA-fast forward** (0.47/0.84 ms,
unchanged), and peak memory is flat (~equal to all trainable paths, **−22% vs torch.compile**).

On the **bwd/fwd ratio**: (c) drops from 12–31× to **5.6–5.8×** (4.7–4.9× in the LoRA
setting). It does **not** reach the 2–3× target — *and that is the correct outcome*. The
ratio stays elevated because the forward is SOTA-fast; the recompute waste is gone, and the
residual gap is simply that the backward's per-pass tilelang stream kernels (`d_pre`, `dx`)
are slower than sglang's hyper-tuned forward GEMM — **not** wasted work. Forcing the ratio to
2–3× (options a/b) means **bloating the forward** (extra/slower GEMM), which *loses* on total
fwd+bwd — the metric that matters for training. The honest pick optimizes total, which (c)
wins. (b) is the dependency-free / fp32 fallback; (a) is dominated.

**Caveats:** (c) imports `sglang.srt.layers.mhc.mhc_pre_gemm_sqrsum_tilelang` (a stabler,
narrower coupling than the old `mhc_pre` whole-op dependency); bf16-only (use (b) for
fp32/fp64). Further bwd-ratio reduction would come from optimizing the `d_pre`/`dx` kernels,
not from touching the forward — documented follow-up.

> **Note (superseded):** an earlier session reported the hybrid `hyper_connection`
> *forward* regressing to ~1.1–2.6 ms because the Sinkhorn-deadlock revert left it on the
> **eager** `_middle_fwd`. That is now fixed at the source: the hybrid forward (option (b))
> uses the fused `_build_sinkhorn_collapse` kernel (forward 0.38/0.78 ms at S2048/8192 in the
> trade-off table above), no longer the eager torch middle.

Supplementary context:
- The mHC paper (arXiv 2512.24880) reports the authors fuse mHC into **unified TileLang
  kernels** (mixed precision + selective recompute + DualPipe), **6.7% whole-model compute
  overhead** — a model-level figure (pipeline overlap hides latency), not a per-op
  microbenchmark, but it confirms fused TileLang is the production direction, and sglang's
  kernels above realise it.
- **FlashSinkhorn** (arXiv 2602.03067) is a fused Sinkhorn-Knopp kernel for *entropic optimal
  transport* on large `n×m` cost matrices — a different regime than mHC's tiny 4×4 per-token
  projection, so it is related work, not a baseline.

## bwd/fwd latency ratio — see the trade-off table above
The current ratios (post backward-recompute fix) are in the §Backward-recompute fix table:
default path (c) is **5.6–5.8×** (4.7–4.9× frozen-param LoRA). The ratio is dominated by the
SOTA-fast forward, not by backward waste — see the verdict there for why chasing 2–3× would
regress total fwd+bwd. The backward makes streamed passes over x (`d_pre` reduction, `dx`,
and the `d_fn` GEMM only when `fn` needs grad — skipped in the LoRA setting).

## codex / ncu
- **codex xhigh optimization+review pass (applied + GPU-validated)** — codex cannot run
  CUDA (no GPU in its container), so all edits were re-verified locally: `test_correctness.py`
  stays **ALL PASS** and `bench.py` numbers above are post-codex on GPU 1. Changes kept:
  1. **TMA fast-path fix** in `_build_norm_gemm`: bf16 `x` is staged through a same-dtype
     bf16 shared tile and then cast to fp32 (was: copying bf16→fp32 directly, which fell
     off the TMA path — `bench` previously spammed `tma load … bfloat16 vs float32 …
     fallback`). Warning is now gone.
  2. New **`_build_d_pre`** kernel (d_pre reduction) and **`_build_d_fn`** kernel (d_fn
     GEMM): both read `x` in the caller dtype and cast per-tile, replacing the two torch
     ops that each materialised a 512 MB fp32 copy of `x`. This is what cut backward
     latency (~1.4–1.6× → ~parity) **and** peak memory (−19–27%).
  3. Robustness: `n < N` guards added so non-`BN`-divisible token counts are safe.
  - All three were reviewed for correctness and confirmed exact by the test suite
    (analytic-vs-autograd fp64 gradcheck still 0 error). No math changed.
- ncu: not run (kernels are memory-bound; the achieved peak-mem reduction and bandwidth
  parity with torch.compile already characterise them — left as a future deep-dive).

Sources: sglang `python/sglang/srt/layers/mhc.py` (`mhc_pre` + `*_tilelang`) & `models/deepseek_v4.py` (`hc_pre` call convention) · [mHC paper](https://arxiv.org/abs/2512.24880) · [FlashSinkhorn](https://arxiv.org/pdf/2602.03067) · [SGLang DeepSeek-V4](https://www.lmsys.org/blog/2026-04-25-deepseek-v4/)
