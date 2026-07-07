"""tilelang kernels for the DeepSeek-V4-Flash mHC ``HyperConnection`` (fwd + a
hand-written analytic backward), exposed as a ``torch.autograd.Function``.

Design (see ``RESULTS.md`` for the derivation and benchmarks)
=============================================================
Let ``x = hidden_streams`` reshaped to ``xf[N, M]`` with ``N=B*S`` and
``M=H*D=16384``.  The model computes ``flat = rmsnorm(xf)`` then
``raw = flat @ fn.T``.  Because the (unweighted) RMSNorm rescale is a *per-row
scalar* ``inv[n] = rsqrt(mean_k xf^2 + eps)``, we never have to materialise the
512 MB ``flat`` tensor::

        raw[n, m] = inv[n] * (xf[n, :] . fn[m, :])

That identity is the whole forward win: one streamed pass over ``x`` produces
both ``inv`` and the (un-normalised) mix logits via a tensor-core GEMM; the sum
of squares is computed as a *second* GEMM against an all-ones matrix so the
kernel stays entirely in MMA fragments (a fused `reduce`+`gemm` cannot share a
register layout in tilelang).

The ``[N,4,4]`` Sinkhorn-Knopp middle (sigmoid / softmax / 20 iters) + the
stream-collapse are run in the FORWARD by a single fused tilelang kernel
(``_build_sinkhorn_collapse``, a per-token warp-0-Sinkhorn / other-warps-collapse
split mirroring sglang's ``mhc_pre_big_fuse_tilelang``; compiled with
``TL_DISABLE_WARP_SPECIALIZED`` — the config that fixes the cross-warp deadlock the
default ``tilelang.compile`` produced).  This replaces the earlier torch.compile
middle (~0.3-0.4 ms launch tax) + separate torch collapse.  The torch
``_middle_fwd``/``_middle_bwd`` remain: ``_middle_fwd`` is the ``MHC_TORCH_MIDDLE=1``
/ fp64 fallback, and the Sinkhorn **backward** stays the hand-derived ``_middle_bwd``
(explicit per-step Jacobian-vector products — NOT autograd through 20 iters; a
per-thread-per-token tilelang Sinkhorn was ~2.2 ms, far worse).

Backward (all hand-derived, verified to fp64 machine precision against autograd
through the reference — no autograd-through-Sinkhorn, no tilelang autodiff):

  * ``collapse``:  d_pre[n,h] = Σ_d dcollapsed[n,d]·x[n,h,d]      (tilelang reduce)
                   the x-grad collapse term is folded into the dx kernel.
  * ``middle`` :   recompute the Sinkhorn forward saving every state, then walk
                   it back.  Each normalise ``y=x/(Σx+eps)`` has Jacobian-vector
                   product ``gx = (gy − Σ_axis(gy·y)) / (Σx+eps)``.  Then softmax
                   backward → ``d_raw[N,24]``.
  * ``dx``     :   d_x[n,k] = pre[n,h]·dcollapsed[n,d]
                            + inv[n]·(d_raw[n,:]·fn[:,k])
                            − (inv[n]²·p[n]/M)·xf[n,k],   p[n]=Σ_m d_raw[n,m]·raw[n,m]
                   (the RMSNorm Jacobian collapses to the tiny scalar ``p`` because
                   Σ_k fn[m,k]·xf[n,k] = raw[n,m]/inv[n].)
  * param grads ``d_fn, d_base, d_scale`` — ``d_fn`` is a tilelang GEMM when
    requested; ``d_base`` / ``d_scale`` are tiny torch reductions off ``d_raw``.

Compute is fp32 throughout (the model marks mHC ``_keep_in_fp32_modules``);
``x`` / ``collapsed`` / ``d_x`` carry the caller dtype (bf16 in training).
"""

import functools
import math
import os
import torch

os.environ.setdefault("TILELANG_CACHE_DIR", "/tmp/tilelang_cache")
os.environ.setdefault("TILELANG_TMP_DIR", "/tmp/tilelang_cache/tmp")

import tilelang
import tilelang.language as T

H = 4
MIX = (2 + H) * H  # 24
MIXP = 32  # padded mix width (tensor-core alignment)
HIDDEN = 4096
SINKHORN_ITERS = 20
HC_EPS = 1.0e-6
RMS_EPS = 1.0e-6
NSTATE = 1 + 2 * (SINKHORN_ITERS - 1)  # 39 Sinkhorn normalise steps
NHIST = NSTATE + 1  # 40 stored states (incl. comb0)
_ONES_CACHE = {}


# =============================== forward kernels ===============================


@functools.cache
def _build_norm_gemm(N, M, in_dtype, BN=64, BK=128):
    """raw[N,MIXP] = inv[n] * (xf @ fn.T) ; inv[N] = rsqrt(mean xf^2 + eps).

    Single streamed pass over x.  ``Fn`` and ``Ones`` are padded to MIXP rows
    (extra fn rows zero; Ones is all-ones so the sum-of-squares is broadcast
    across the output columns, letting us read ``sc[i,m]`` with the same
    fragment layout as ``g[i,m]``)."""

    @T.prim_func
    def main(
        X: T.Tensor([N, M], in_dtype),
        Fn: T.Tensor([MIXP, M], "float32"),
        Ones: T.Tensor([MIXP, M], "float32"),
        Raw: T.Tensor([N, MIXP], "float32"),
        Inv: T.Tensor([N], "float32"),
    ):
        with T.Kernel(T.ceildiv(N, BN), threads=128) as bx:
            # Stage low-precision x through same-dtype shared memory first so
            # TileLang can use the fast copy path, then cast for fp32 GEMM math.
            if in_dtype == "float32":
                Xs = T.alloc_shared([BN, BK], "float32")
            else:
                Xq = T.alloc_shared([BN, BK], in_dtype)
                Xs = T.alloc_shared([BN, BK], "float32")
            Fs = T.alloc_shared([MIXP, BK], "float32")
            Os = T.alloc_shared([MIXP, BK], "float32")
            Xsq = T.alloc_shared([BN, BK], "float32")
            g = T.alloc_fragment([BN, MIXP], "float32")
            sc = T.alloc_fragment([BN, MIXP], "float32")
            T.clear(g)
            T.clear(sc)
            for ko in T.Pipelined(T.ceildiv(M, BK), num_stages=2):
                if in_dtype == "float32":
                    T.copy(X[bx * BN, ko * BK], Xs)
                else:
                    T.copy(X[bx * BN, ko * BK], Xq)
                    for i, k in T.Parallel(BN, BK):
                        Xs[i, k] = T.Cast("float32", Xq[i, k])
                T.copy(Fn[0, ko * BK], Fs)
                T.copy(Ones[0, ko * BK], Os)
                for i, k in T.Parallel(BN, BK):
                    Xsq[i, k] = Xs[i, k] * Xs[i, k]
                T.gemm(Xs, Fs, g, transpose_B=True)
                T.gemm(Xsq, Os, sc, transpose_B=True)
            for i, m in T.Parallel(BN, MIXP):
                n = bx * BN + i
                if n < N:
                    Raw[n, m] = T.rsqrt(sc[i, m] / M + RMS_EPS) * g[i, m]
                    if m == 0:
                        Inv[n] = T.rsqrt(sc[i, m] / M + RMS_EPS)

    return tilelang.compile(main, out_idx=[3, 4])


def _middle_fwd(raw, base, scale, want_states=False):
    """Sigmoid / 2·sigmoid / softmax+Sinkhorn(20) — torch (see module docstring
    for why this stays out of tilelang).  ``raw`` is ``[N,MIX]``.

    Returns ``(pre[N,H], post[N,H], comb[N,H,H])`` and, if ``want_states``, also
    the list of Sinkhorn states + ``comb0`` needed by the analytic backward."""
    N = raw.shape[0]
    s0, s1, s2 = scale.unbind(0)
    pre = torch.sigmoid(raw[:, :H] * s0 + base[:H]) + HC_EPS
    post = 2.0 * torch.sigmoid(raw[:, H : 2 * H] * s1 + base[H : 2 * H])
    cl = raw[:, 2 * H :].view(N, H, H) * s2 + base[2 * H :].view(H, H)
    c0 = torch.softmax(cl, dim=-1) + HC_EPS
    states = [c0]
    c = c0 / (c0.sum(dim=-2, keepdim=True) + HC_EPS)  # step0: column norm
    if want_states:
        states.append(c)
    for _ in range(SINKHORN_ITERS - 1):
        c = c / (c.sum(dim=-1, keepdim=True) + HC_EPS)  # row norm
        if want_states:
            states.append(c)
        c = c / (c.sum(dim=-2, keepdim=True) + HC_EPS)  # column norm
        if want_states:
            states.append(c)
    if want_states:
        return pre, post, c, states, c0
    return pre, post, c


def _middle_bwd(raw, base, scale, d_pre, d_post, d_comb):
    """Analytic backward of ``_middle_fwd`` -> ``d_raw[N,MIX]``.

    Recomputes the Sinkhorn forward (saving every state) then walks it back.
    Each normalise ``y = x/(Σ_axis x + eps)`` has VJP
    ``gx = (gy − Σ_axis(gy·y)) / (Σ_axis x + eps)``.  No autograd is used."""
    N = raw.shape[0]
    s0, s1, s2 = scale.unbind(0)
    pre, post, _comb, states, c0 = _middle_fwd(raw, base, scale, want_states=True)
    # pre / post (sigmoid) backward
    sig_pre = pre - HC_EPS
    d_pre_w = d_pre * sig_pre * (1.0 - sig_pre) * s0
    sig_post = post / 2.0
    d_post_w = d_post * 2.0 * sig_post * (1.0 - sig_post) * s1
    # Sinkhorn backward (states[t] produced from states[t-1] by a normalise)
    gy = d_comb.clone()
    Tn = len(states)  # 40
    for t in range(Tn - 1, 0, -1):
        # axis: t==1 -> col(-2); else even t -> row(-1), odd t -> col(-2)
        axis = -2 if t == 1 else (-1 if t % 2 == 0 else -2)
        prev = states[t - 1]
        denom = prev.sum(dim=axis, keepdim=True) + HC_EPS
        y = states[t]
        gy = (gy - (gy * y).sum(dim=axis, keepdim=True)) / denom
    # softmax backward (gy is grad on comb0 = softmax(logits)+eps)
    sm = c0 - HC_EPS
    dz = sm * (gy - (gy * sm).sum(dim=-1, keepdim=True))
    d_comb_w = (dz * s2).reshape(N, H * H)
    return torch.cat([d_pre_w, d_post_w, d_comb_w], dim=-1)


# torch.compile fuses the ~40 tiny Sinkhorn ops into a handful of kernels; this is
# the fastest formulation of the middle found (see module docstring / RESULTS.md).
try:  # pragma: no cover - environment dependent
    _middle_fwd_c = torch.compile(_middle_fwd)
    _middle_bwd_c = torch.compile(_middle_bwd)
except Exception:
    _middle_fwd_c, _middle_bwd_c = _middle_fwd, _middle_bwd


@functools.cache
def _build_collapse(N, D, in_dtype, BN=16, BD=256):
    """collapsed[n,d] = Σ_h pre[n,h]·x[n,h,d].  Memory-bound second pass over x.
    H=4 is unrolled (no per-element ``alloc_local`` / serial loop — that pattern
    is ~50× slower in tilelang); ``pre`` is staged through shared memory."""
    M = H * D

    @T.prim_func
    def main(
        X: T.Tensor([N, M], in_dtype),
        Pre: T.Tensor([N, H], "float32"),
        Out: T.Tensor([N, D], in_dtype),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(D, BD), threads=256) as (bn, bd):
            ps = T.alloc_shared([BN, H], "float32")
            T.copy(Pre[bn * BN, 0], ps)
            for i, dd in T.Parallel(BN, BD):
                n = bn * BN + i
                d = bd * BD + dd
                if (n < N) and (d < D):
                    Out[n, d] = T.Cast(
                        in_dtype,
                        ps[i, 0] * T.Cast("float32", X[n, d])
                        + ps[i, 1] * T.Cast("float32", X[n, D + d])
                        + ps[i, 2] * T.Cast("float32", X[n, 2 * D + d])
                        + ps[i, 3] * T.Cast("float32", X[n, 3 * D + d]),
                    )

    return tilelang.compile(main, out_idx=[2])


@functools.cache
def _build_sinkhorn_collapse(N, D, in_dtype, threads=96):
    """Fused forward middle+collapse: one block per token (mirrors sglang's
    ``mhc_pre_big_fuse_tilelang``).  Consumes the rms-applied mix logits
    ``Raw[n,:]`` (= inv·(xf@fnᵀ), produced by ``_build_norm_gemm``) and emits the
    four forward tensors in a single launch::

        pre[n,h]   = sigmoid(raw[n,h]·s0 + base[h]) + eps
        post[n,h]  = 2·sigmoid(raw[n,H+h]·s1 + base[H+h])
        comb[n]    = Sinkhorn-Knopp(softmax(raw[n,2H:]·s2 + base[2H:]), 20 iters)
        coll[n,d]  = Σ_h pre[n,h]·x[n,h,d]

    Work split inside the block: warp 0 (threads <32) runs the entire 4×4
    softmax + 20-iter Sinkhorn in fragments with warp-level reductions (this is
    the trick that makes a per-token Sinkhorn fast — NOT one thread per token);
    the remaining warps compute ``pre`` and stream the memory-bound collapse over
    the hidden dim.  Eliminates the torch.compile middle + the separate collapse
    launch (3+ launches -> norm_gemm + this = 2, like sglang)."""
    hb = math.gcd(512, D)

    @T.prim_func
    def main(
        X: T.Tensor([N, H, D], in_dtype),
        Raw: T.Tensor([N, MIX], "float32"),
        Base: T.Tensor([MIX], "float32"),
        Scale: T.Tensor([3], "float32"),
        Pre: T.Tensor([N, H], "float32"),
        Post: T.Tensor([N, H], "float32"),
        Comb: T.Tensor([N, H, H], "float32"),
        Coll: T.Tensor([N, D], in_dtype),
    ):
        with T.Kernel(N, threads=threads) as i:
            mixes = T.alloc_shared([MIX], "float32")
            T.copy(Raw[i, :], mixes)

            if T.get_thread_binding() < 32:
                # ---- warp 0: post + Sinkhorn-balanced comb (4×4, in fragments)
                cm = T.alloc_fragment([H, H], "float32")
                for j in T.Parallel(H):
                    Post[i, j] = 2.0 * T.sigmoid(mixes[j + H] * Scale[1] + Base[j + H])
                for j, k in T.Parallel(H, H):
                    cm[j, k] = mixes[j * H + k + 2 * H] * Scale[2] + Base[j * H + k + 2 * H]
                row_sum = T.alloc_fragment([H], "float32")
                col_sum = T.alloc_fragment([H], "float32")
                row_max = T.alloc_fragment([H], "float32")
                # softmax(dim=-1) + eps  == comb0
                T.reduce_max(cm, row_max, dim=1)
                for j, k in T.Parallel(H, H):
                    cm[j, k] = T.exp(cm[j, k] - row_max[j])
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(H, H):
                    cm[j, k] = cm[j, k] / row_sum[j] + HC_EPS
                # step 0: column normalise
                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(H, H):
                    cm[j, k] = cm[j, k] / (col_sum[k] + HC_EPS)
                for _ in T.serial(SINKHORN_ITERS - 1):
                    T.reduce_sum(cm, row_sum, dim=1)  # row normalise
                    for j, k in T.Parallel(H, H):
                        cm[j, k] = cm[j, k] / (row_sum[j] + HC_EPS)
                    T.reduce_sum(cm, col_sum, dim=0)  # column normalise
                    for j, k in T.Parallel(H, H):
                        cm[j, k] = cm[j, k] / (col_sum[k] + HC_EPS)
                for j, k in T.Parallel(H, H):
                    Comb[i, j, k] = cm[j, k]
            else:
                # ---- warps 1+: pre weights + memory-bound stream collapse
                pre_s = T.alloc_shared([H], "float32")
                for j in T.Parallel(H):
                    pre_s[j] = T.sigmoid(mixes[j] * Scale[0] + Base[j]) + HC_EPS
                    Pre[i, j] = pre_s[j]
                for ho in T.Pipelined(D // hb, num_stages=2):
                    xs = T.alloc_shared([H, hb], in_dtype)
                    xl = T.alloc_fragment([H, hb], "float32")
                    T.copy(X[i, 0, ho * hb], xs)
                    T.copy(xs, xl)
                    ol = T.alloc_fragment([hb], "float32")
                    T.clear(ol)
                    for h in T.serial(H):
                        for d in T.Parallel(hb):
                            ol[d] += pre_s[h] * xl[h, d]
                    T.copy(ol, Coll[i, ho * hb])

    # NOTE: the warp-0-Sinkhorn / other-warps-collapse split MUST be compiled with
    # TL_DISABLE_WARP_SPECIALIZED. This is the exact config sglang ships its
    # identical `mhc_pre_big_fuse_tilelang` with. The 2026-06-30 deadlock was caused
    # by the default `tilelang.compile`, whose warp-specialization pass tried to
    # auto-specialize this producer/consumer split into a cross-warp barrier hang.
    # Disabling that pass (+ the TMA lowering, also disabled by sglang here) makes
    # the kernel run to completion. Verified non-hanging at S=128 with a 90s timeout.
    return tilelang.compile(
        main,
        out_idx=[4, 5, 6, 7],
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        },
    )


# Set MHC_TORCH_MIDDLE=1 to force the torch.compile Sinkhorn middle + torch
# collapse (the pre-fusion fallback). Default uses the fused tilelang kernel.
_USE_TORCH_MIDDLE = os.environ.get("MHC_TORCH_MIDDLE", "0") == "1"


def _fused_middle_collapse(x3d, raw, base_f, scale_f, N, D, in_dtype, ds):
    """Run the gates + 20-iter Sinkhorn + stream-collapse from the rms-applied mix
    logits ``raw[N,MIX]`` in ONE tilelang launch (``_build_sinkhorn_collapse``,
    the validated non-deadlocking fused kernel). Returns ``(pre, post, comb,
    collapsed)`` exactly as the torch ``_middle_fwd`` + collapse would.

    The whole point: the caller has already produced ``raw`` (and ``inv``) with a
    SINGLE norm+GEMM pass and saves them for the analytic backward, so the
    backward never recomputes the GEMM. The fused kernel replaces the
    torch.compile middle (~0.3-0.4 ms launch tax) + the separate torch collapse.

    Falls back to the torch path for non-tilelang dtypes (e.g. fp64) or when
    ``MHC_TORCH_MIDDLE=1``."""
    tl_dtypes = ("bfloat16", "float16", "float32")
    if _USE_TORCH_MIDDLE or ds not in tl_dtypes:
        pre, post, comb = _middle_fwd_c(raw, base_f, scale_f)
        collapsed = (pre.unsqueeze(-1) * x3d.float()).sum(dim=1).to(in_dtype)
        return pre, post, comb, collapsed
    pre, post, comb, coll = _build_sinkhorn_collapse(N, D, ds)(x3d, raw.contiguous(), base_f, scale_f)
    return pre, post, comb.view(N, H, H), coll


# =============================== backward kernels ==============================


@functools.cache
def _build_d_pre(N, M, D, in_dtype, BD=256):
    """d_pre[n,h] = Σ_d dcollapsed[n,d]·x[n,h,d].

    This replaces the torch einsum that materialised ``x.float()``.  One block
    handles one token and reduces over D in fp32 fragments while reading x in
    the caller dtype.
    """

    @T.prim_func
    def main(
        X: T.Tensor([N, M], in_dtype),
        dColl: T.Tensor([N, D], "float32"),
        dPre: T.Tensor([N, H], "float32"),
    ):
        with T.Kernel(N, threads=128) as n:
            dc = T.alloc_fragment([BD], "float32")
            prod = T.alloc_fragment([H, BD], "float32")
            part = T.alloc_fragment([H], "float32")
            acc = T.alloc_fragment([H], "float32")
            for h in T.Parallel(H):
                acc[h] = 0.0
            for do in T.serial(T.ceildiv(D, BD)):
                for d in T.Parallel(BD):
                    dd = do * BD + d
                    if dd < D:
                        dc[d] = dColl[n, dd]
                    else:
                        dc[d] = 0.0
                for h, d in T.Parallel(H, BD):
                    dd = do * BD + d
                    if dd < D:
                        prod[h, d] = T.Cast("float32", X[n, h * D + dd]) * dc[d]
                    else:
                        prod[h, d] = 0.0
                T.reduce_sum(prod, part, dim=1, clear=True)
                for h in T.Parallel(H):
                    acc[h] += part[h]
            for h in T.Parallel(H):
                dPre[n, h] = acc[h]

    return tilelang.compile(main, out_idx=[2])


@functools.cache
def _build_dx(N, M, D, in_dtype, BN=64, BK=128):
    """d_x[n,k] = pre[n,h]·dcollapsed[n,d] + inv[n]·(d_raw[n,:]·fn[:,k]) − pp[n]·xf[n,k]
    with k = h·D + d (BK divides D so each k-tile lies in one stream h) and
    pp[n] = inv[n]²·p[n]/M.  Streamed pass over x; GEMM over the 32-wide mix."""

    @T.prim_func
    def main(
        X: T.Tensor([N, M], in_dtype),
        Fn: T.Tensor([MIXP, M], "float32"),
        dRawP: T.Tensor([N, MIXP], "float32"),
        Inv: T.Tensor([N], "float32"),
        Pp: T.Tensor([N], "float32"),
        Pre: T.Tensor([N, H], "float32"),
        dColl: T.Tensor([N, D], "float32"),
        dX: T.Tensor([N, M], in_dtype),
    ):
        with T.Kernel(T.ceildiv(M, BK), T.ceildiv(N, BN), threads=128) as (bk, bn):
            Fs = T.alloc_shared([MIXP, BK], "float32")
            dr = T.alloc_shared([BN, MIXP], "float32")
            dflat = T.alloc_fragment([BN, BK], "float32")
            k0 = bk * BK
            h = k0 // D  # constant within the tile
            d0 = k0 - h * D
            T.copy(Fn[0, k0], Fs)
            T.copy(dRawP[bn * BN, 0], dr)
            T.clear(dflat)
            T.gemm(dr, Fs, dflat, transpose_B=False)  # [BN,BK] = dRawP @ Fn[:,tile]
            for i, kk in T.Parallel(BN, BK):
                n = bn * BN + i
                if n < N:
                    xv = T.Cast("float32", X[n, k0 + kk])
                    val = Pre[n, h] * dColl[n, d0 + kk] + Inv[n] * dflat[i, kk] - Pp[n] * xv
                    dX[n, k0 + kk] = T.Cast(in_dtype, val)

    return tilelang.compile(main, out_idx=[7])


@functools.cache
def _build_d_fn(N, M, in_dtype, BN=64, BK=128):
    """d_fn = (d_raw * inv[:, None]).T @ x.

    Torch's matmul path first builds ``x.float()``.  This GEMM reads x directly
    in caller dtype, casts the tile in shared memory, and accumulates fp32.
    """

    @T.prim_func
    def main(
        X: T.Tensor([N, M], in_dtype),
        dRawP: T.Tensor([N, MIXP], "float32"),
        Inv: T.Tensor([N], "float32"),
        dFn: T.Tensor([MIX, M], "float32"),
    ):
        with T.Kernel(T.ceildiv(M, BK), threads=128) as bk:
            Xs = T.alloc_shared([BN, BK], "float32")
            dr = T.alloc_shared([BN, MIXP], "float32")
            acc = T.alloc_fragment([MIXP, BK], "float32")
            k0 = bk * BK
            T.clear(acc)
            for no in T.Pipelined(T.ceildiv(N, BN), num_stages=2):
                for i, k in T.Parallel(BN, BK):
                    n = no * BN + i
                    kk = k0 + k
                    if (n < N) and (kk < M):
                        Xs[i, k] = T.Cast("float32", X[n, kk])
                    else:
                        Xs[i, k] = 0.0
                for i, m in T.Parallel(BN, MIXP):
                    n = no * BN + i
                    if n < N:
                        dr[i, m] = dRawP[n, m] * Inv[n]
                    else:
                        dr[i, m] = 0.0
                T.gemm(dr, Xs, acc, transpose_A=True)
            for m, k in T.Parallel(MIX, BK):
                kk = k0 + k
                if kk < M:
                    dFn[m, kk] = acc[m, k]

    return tilelang.compile(main, out_idx=[3])


# =============================== glue / autograd ===============================


def _pad_fn(fn):
    """fn[MIX,M] -> fnp[MIXP,M] with zero rows."""
    fnp = fn.new_zeros(MIXP, fn.shape[1])
    fnp[:MIX] = fn
    return fnp.contiguous()


def _ones(device):
    key = (device.type, device.index)
    ones = _ONES_CACHE.get(key)
    if ones is None or ones.device != device:
        ones = torch.ones(MIXP, H * HIDDEN, device=device, dtype=torch.float32)
        _ONES_CACHE[key] = ones
    return ones


def _compact_last_dim(t: torch.Tensor, width: int) -> torch.Tensor:
    """Return a freshly packed `[..., width]` tensor with strict dense strides."""
    out = t.new_empty((*t.shape[:-1], width))
    out.copy_(t[..., :width])
    return out


def _hc_backward(x2d, fnp, raw, inv, pre, scale, base, shape, d_post, d_comb, d_collapsed, needs_input_grad):
    """Shared analytic backward used by both the hybrid (`HyperConnectionFn`)
    and the sglang-forward (`HyperConnectionSglangFn`) paths.

    Inputs are the (saved or recomputed) forward intermediates ``raw[N,MIX]``,
    ``inv[N]``, ``pre[N,H]`` plus the streamed activation ``x2d[N,M]`` and the
    padded mix matrix ``fnp[MIXP,M]``.  The backward is independent of *how* the
    forward produced (post, comb, collapsed); it only needs ``raw, inv, pre``
    (+ params), so the two forward paths share this code verbatim."""
    B, S, D, N, M, in_dtype, ds = shape
    d_post = d_post.reshape(N, H).float().contiguous()
    d_comb = d_comb.reshape(N, H, H).float().contiguous()
    d_coll = d_collapsed.reshape(N, D).float().contiguous()

    # d_pre[n,h] = Σ_d dcollapsed[n,d] * x[n,h,d]
    d_pre = _build_d_pre(N, M, D, ds)(x2d, d_coll)

    # middle backward (hand-derived analytic Sinkhorn VJP) -> d_raw[N,MIX]
    d_raw = _middle_bwd_c(raw, base.contiguous(), scale.contiguous(), d_pre, d_post, d_comb)

    need_x, need_fn, need_base, need_scale = needs_input_grad
    d_raw_p = None
    if need_x or need_fn:
        d_raw_p = torch.nn.functional.pad(d_raw, (0, MIXP - MIX)).contiguous()  # [N,MIXP]

    # per-token scalars for the dx kernel
    d_x = None
    if need_x:
        p = (d_raw * raw).sum(-1)  # Σ_m d_raw·raw
        pp = (inv * inv * p / M).contiguous()
        d_x = _build_dx(N, M, D, ds)(x2d, fnp, d_raw_p, inv.contiguous(), pp, pre.contiguous(), d_coll).view(
            B, S, H, D
        )

    # parameter grads (tiny — analytic reductions off d_raw). In the V4-Flash
    # LoRA setting fn/base/scale are frozen, so these are usually skipped (the
    # only required grad is d_x, for cross-layer flow). d_fn is the one heavy
    # term (a [MIX,N]·[N,M] GEMM); it is computed only when requested.
    s0, s1, s2 = scale.unbind(0)
    d_fn = _build_d_fn(N, M, ds)(x2d, d_raw_p, inv.contiguous()) if need_fn else None
    d_base = (
        torch.cat(
            [
                d_raw[:, :H].sum(0) / s0,
                d_raw[:, H : 2 * H].sum(0) / s1,
                d_raw[:, 2 * H :].sum(0) / s2,
            ]
        )
        if need_base
        else None
    )
    d_scale = (
        torch.stack(
            [
                (d_raw[:, :H] * raw[:, :H]).sum() / s0,
                (d_raw[:, H : 2 * H] * raw[:, H : 2 * H]).sum() / s1,
                (d_raw[:, 2 * H :] * raw[:, 2 * H :]).sum() / s2,
            ]
        )
        if need_scale
        else None
    )
    return d_x, d_fn, d_base, d_scale


class HyperConnectionFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_streams, fn, base, scale):
        B, S, Hh, D = hidden_streams.shape
        assert Hh == H and D == HIDDEN, (hidden_streams.shape,)
        N, M = B * S, H * D
        in_dtype = hidden_streams.dtype
        ds = {torch.float32: "float32", torch.bfloat16: "bfloat16", torch.float16: "float16"}[in_dtype]
        x2d = hidden_streams.reshape(N, M).contiguous()
        fn32 = fn.float().contiguous()
        fnp = _pad_fn(fn32)
        ones = _ones(fn.device)

        base_f = base.float().contiguous()
        scale_f = scale.float().contiguous()
        raw_p, inv = _build_norm_gemm(N, M, ds)(x2d, fnp, ones)
        raw = _compact_last_dim(raw_p, MIX)
        # Option (b): ONE norm+GEMM pass (above) produces raw/inv, which are SAVED
        # for the analytic backward; the gates + 20-iter Sinkhorn + stream-collapse
        # are then done by the fused `_build_sinkhorn_collapse` tilelang kernel (now
        # compiled with TL_DISABLE_WARP_SPECIALIZED — the fix for the 2026-06-30
        # deadlock). Because raw/inv are saved, the backward does NOT recompute the
        # GEMM. (`MHC_TORCH_MIDDLE=1` falls back to the torch.compile middle.)
        pre, post, comb, collapsed = _fused_middle_collapse(
            x2d.view(N, H, D), raw, base_f, scale_f, N, D, in_dtype, ds
        )

        ctx.save_for_backward(x2d, fnp, raw, inv, pre, scale_f, base_f)
        ctx.shape = (B, S, D, N, M, in_dtype, ds)
        return (post.view(B, S, H), comb.view(B, S, H, H), collapsed.view(B, S, D))

    @staticmethod
    def backward(ctx, d_post, d_comb, d_collapsed):
        x2d, fnp, raw, inv, pre, scale, base = ctx.saved_tensors
        return _hc_backward(
            x2d, fnp, raw, inv, pre, scale, base, ctx.shape, d_post, d_comb, d_collapsed, ctx.needs_input_grad
        )


def hyper_connection(hidden_streams, fn, base, scale):
    """tilelang mHC HyperConnection. Returns (post, comb, collapsed)."""
    return HyperConnectionFn.apply(hidden_streams, fn, base, scale)


# ============== sglang-forward + our-analytic-backward variant =================
# Uses sglang's production fused TileLang `mhc_pre` for the FORWARD (closing the
# 2–6.4× forward-speed gap vs our hybrid), and our VALIDATED analytic backward
# (`_hc_backward`). sglang's `mhc_pre` does NOT expose the pre-gate mix logits
# (`raw`/`inv`) or the collapse weights (`pre`), so the backward recomputes them
# lazily: one `_build_norm_gemm` pass (raw, inv) + a direct sigmoid for `pre`
# (`pre` is independent of the 20-iter Sinkhorn). The forward therefore stays
# purely sglang's fast path; the recompute is paid only when grad is requested.
#
# sglang's `mhc_pre` computes the SAME math as `reference.py` (RMSNorm rescale +
# fn-GEMM mix logits → sigmoid pre/post + softmax→20-iter-Sinkhorn comb →
# stream-collapse), so the forward matches the reference within tf32/bf16
# rounding (validated in `test_correctness.py` / `bench_sota.py`). The backward
# is the gradient of THAT reference math (independent of how the forward was
# computed), so the fp64 analytic gradcheck stays exact.
#
# Constraint: sglang's `mhc_pre` asserts a bf16 residual + fp32 `fn` — this path
# is bf16-only (the real training/inference dtype). For fp32/fp64 use the hybrid
# `hyper_connection` above.


class HyperConnectionSglangFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_streams, fn, base, scale):
        # Option (c): reuse sglang's fast TileLang prenorm GEMM stage
        # (`mhc_pre_gemm_sqrsum_tilelang`) to produce the un-normalised mix logits
        # `gemm_out[N,MIX]` and the per-token Σx² `sqrsum[N]`. From those we form
        # raw = inv·gemm_out and inv = rsqrt(Σx²/M + eps) — exactly the `mixes`/`rms`
        # sglang's `mhc_pre_big_fuse` computes internally — and SAVE them. Then the
        # gates + Sinkhorn + collapse run in our fused `_build_sinkhorn_collapse`
        # (a clone of sglang's big_fuse). The forward stays the fast sglang GEMM +
        # one fused kernel (~mhc_pre), but raw/inv are captured for free, so the
        # backward does NOT recompute the GEMM (the whole point of this rewrite).
        from sglang.srt.layers.mhc import mhc_pre_gemm_sqrsum_tilelang

        B, S, Hh, D = hidden_streams.shape
        assert Hh == H and D == HIDDEN, (hidden_streams.shape,)
        N, M = B * S, H * D
        in_dtype = hidden_streams.dtype
        assert in_dtype == torch.bfloat16, (
            "sglang GEMM stage requires a bf16 residual; use hyper_connection() " f"for fp32/fp64 (got {in_dtype})"
        )
        ds = "bfloat16"
        x2d = hidden_streams.reshape(N, M).contiguous()
        fn32 = fn.float().contiguous()
        base_f = base.float().contiguous()
        scale_f = scale.float().contiguous()

        # sglang prenorm GEMM stage: gemm_out[N,MIX] = xf@fnᵀ (fp32, tf32-core),
        # sqrsum[N] = Σ_k xf[n,k]². (Non-split-k path; valid for any N.)
        gemm_out = torch.empty(N, MIX, dtype=torch.float32, device=x2d.device)
        sqrsum = torch.empty(N, dtype=torch.float32, device=x2d.device)
        mhc_pre_gemm_sqrsum_tilelang(x2d, fn32, gemm_out, sqrsum, MIX, M)
        inv = torch.rsqrt(sqrsum / M + RMS_EPS)
        raw = (inv[:, None] * gemm_out).contiguous()  # rms-applied mix logits

        pre, post, comb, collapsed = _fused_middle_collapse(
            x2d.view(N, H, D), raw, base_f, scale_f, N, D, in_dtype, ds
        )

        # Save raw/inv/pre so the analytic backward needs no GEMM recompute.
        ctx.save_for_backward(x2d, _pad_fn(fn32), raw, inv, pre, scale_f, base_f)
        ctx.shape = (B, S, D, N, M, in_dtype, ds)
        return (post.reshape(B, S, H), comb.reshape(B, S, H, H), collapsed.reshape(B, S, D))

    @staticmethod
    def backward(ctx, d_post, d_comb, d_collapsed):
        x2d, fnp, raw, inv, pre, scale, base = ctx.saved_tensors
        # raw/inv/pre were captured in the forward — NO GEMM recompute here.
        return _hc_backward(
            x2d, fnp, raw, inv, pre, scale, base, ctx.shape, d_post, d_comb, d_collapsed, ctx.needs_input_grad
        )


def hyper_connection_sglang(hidden_streams, fn, base, scale):
    """mHC HyperConnection with sglang's fused TileLang `mhc_pre` forward and our
    analytic backward. bf16-only. Returns (post, comb, collapsed)."""
    return HyperConnectionSglangFn.apply(hidden_streams, fn, base, scale)
