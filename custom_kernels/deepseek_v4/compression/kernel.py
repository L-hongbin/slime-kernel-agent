"""TileLang kernels for the DeepSeek-V4-Flash compression pool (B2).

Fused **per-channel windowed gated-softmax pool + RMSNorm** over head_dim, with a
hand-written backward (no tilelang autodiff). bf16 (or fp32) compute, fp32 softmax +
accumulation, matching ``reference.py``.

Core op (per row n of N = B*n_win, per channel d of D, over window slots k of K):
    w[k,d]   = softmax_k( gate[n,k,d] )            (fp32, per-channel over K)
    p[n,d]   = sum_k w[k,d] * kv[n,k,d]            (pooled, fp32)
    out[n,d] = weight[d] * p[n,d] * rsqrt(mean_d(p^2) + eps)   (RMSNorm over D)

Backward (derived analytically, see RESULTS.md):
    a_d  = dout_d * weight_d ;  r = rsqrt(mean_d p^2 + eps) ;  S1 = sum_d a_d p_d
    dp_d = r*a_d - (r^3 p_d / D) * S1
    dkv[k,d]   = dp_d * w[k,d]
    dgate[k,d] = dp_d * w[k,d] * (kv[k,d] - p_d)
    dweight_d  = sum_n dout[n,d] * p[n,d] * r[n]     (done in torch)

The generic autograd.Function operates on a windowed tensor [N, K, D]. HCA uses that
path directly. CSA also has a fused raw-input path that folds the Ca/Cb overlap gather
and +position_bias into TileLang, avoiding materialized new_kv/new_gate tensors.
"""

import functools
import os

import torch

os.environ.setdefault("TILELANG_CACHE_DIR", "/tmp/tilelang_cache")
os.environ.setdefault("TILELANG_TMP_DIR", "/tmp/tilelang_cache/tmp")

import tilelang
import tilelang.language as T

from reference import csa_build_overlap  # noqa: E402  (local import, run from this dir)


NEG = -1e30


# --------------------------------------------------------------------------------------
# Column-tile width: keep the [K, BD] working tile small enough for registers.
# --------------------------------------------------------------------------------------
def _pick_bd(K: int, D: int) -> int:
    target = max(1, 8192 // K)
    bd = min(D, target)
    while D % bd != 0:
        bd -= 1
    return bd


# --------------------------------------------------------------------------------------
# Forward kernel.  Outputs: Out [N, D] (rmsnormed), Pooled [N, D] (fp32, saved for bwd).
# --------------------------------------------------------------------------------------
@functools.cache
def _build_fwd(N: int, K: int, D: int, in_dtype: str, eps: float, threads: int = 128):
    BD = _pick_bd(K, D)
    n_tiles = D // BD
    acc = "float32"

    @tilelang.jit(out_idx=[-2, -1])
    def _make():
        @T.prim_func
        def main(
            KV: T.Tensor([N, K, D], in_dtype),  # type: ignore
            G: T.Tensor([N, K, D], in_dtype),  # type: ignore
            Wt: T.Tensor([D], in_dtype),  # type: ignore
            Out: T.Tensor([N, D], in_dtype),  # type: ignore
            Pooled: T.Tensor([N, D], acc),  # type: ignore
        ):
            with T.Kernel(N, threads=threads) as bn:
                p_s = T.alloc_shared([D], acc)
                g_t = T.alloc_fragment([K, BD], acc)
                kv_t = T.alloc_fragment([K, BD], acc)
                mx = T.alloc_fragment([BD], acc)
                denom = T.alloc_fragment([BD], acc)
                num = T.alloc_fragment([BD], acc)

                for i in T.serial(n_tiles):
                    T.copy(G[bn, :, i * BD : (i + 1) * BD], g_t)
                    T.copy(KV[bn, :, i * BD : (i + 1) * BD], kv_t)
                    T.reduce_max(g_t, mx, dim=0, clear=True)
                    for k, j in T.Parallel(K, BD):
                        g_t[k, j] = T.exp(g_t[k, j] - mx[j])
                    T.reduce_sum(g_t, denom, dim=0, clear=True)
                    for k, j in T.Parallel(K, BD):
                        kv_t[k, j] = g_t[k, j] * kv_t[k, j]
                    T.reduce_sum(kv_t, num, dim=0, clear=True)
                    for j in T.Parallel(BD):
                        p_s[i * BD + j] = num[j] / denom[j]

                # RMSNorm over D
                p_f = T.alloc_fragment([D], acc)
                sq = T.alloc_fragment([D], acc)
                ss = T.alloc_fragment([1], acc)
                T.copy(p_s, p_f)
                for d in T.Parallel(D):
                    sq[d] = p_f[d] * p_f[d]
                T.reduce_sum(sq, ss, dim=0, clear=True)
                rinv = T.alloc_fragment([1], acc)
                rinv[0] = T.rsqrt(ss[0] / D + eps)
                for d in T.Parallel(D):
                    Pooled[bn, d] = p_f[d]
                    Out[bn, d] = T.cast(Wt[d], acc) * p_f[d] * rinv[0]

        return main

    return _make()


# --------------------------------------------------------------------------------------
# Backward kernel.  Outputs: dKV [N, K, D], dG [N, K, D].
# Inputs: KV, G, Wt, P (saved pooled fp32), dOut.
# --------------------------------------------------------------------------------------
@functools.cache
def _build_bwd(N: int, K: int, D: int, in_dtype: str, eps: float, threads: int = 128):
    BD = _pick_bd(K, D)
    n_tiles = D // BD
    acc = "float32"

    @tilelang.jit(out_idx=[-2, -1])
    def _make():
        @T.prim_func
        def main(
            KV: T.Tensor([N, K, D], in_dtype),  # type: ignore
            G: T.Tensor([N, K, D], in_dtype),  # type: ignore
            Wt: T.Tensor([D], in_dtype),  # type: ignore
            P: T.Tensor([N, D], acc),  # type: ignore
            dOut: T.Tensor([N, D], in_dtype),  # type: ignore
            dKV: T.Tensor([N, K, D], in_dtype),  # type: ignore
            dG: T.Tensor([N, K, D], in_dtype),  # type: ignore
        ):
            with T.Kernel(N, threads=threads) as bn:
                p_f = T.alloc_fragment([D], acc)
                sq = T.alloc_fragment([D], acc)
                ap = T.alloc_fragment([D], acc)
                ss = T.alloc_fragment([1], acc)
                s1 = T.alloc_fragment([1], acc)
                rinv = T.alloc_fragment([1], acc)
                k2 = T.alloc_fragment([1], acc)

                # full-D reductions -> scalars r and k2 = r^3 * S1 / D
                for d in T.Parallel(D):
                    p_f[d] = P[bn, d]
                    sq[d] = p_f[d] * p_f[d]
                    ap[d] = (T.cast(dOut[bn, d], acc) * T.cast(Wt[d], acc)) * p_f[d]
                T.reduce_sum(sq, ss, dim=0, clear=True)
                T.reduce_sum(ap, s1, dim=0, clear=True)
                rinv[0] = T.rsqrt(ss[0] / D + eps)
                k2[0] = rinv[0] * rinv[0] * rinv[0] * s1[0] / D

                # per column tile: recompute softmax, compute dp tile, write dKV / dG
                g_t = T.alloc_fragment([K, BD], acc)
                kv_t = T.alloc_fragment([K, BD], acc)
                mx = T.alloc_fragment([BD], acc)
                denom = T.alloc_fragment([BD], acc)
                pt = T.alloc_fragment([BD], acc)
                dpt = T.alloc_fragment([BD], acc)
                dkv_t = T.alloc_fragment([K, BD], acc)
                dg_t = T.alloc_fragment([K, BD], acc)
                for i in T.serial(n_tiles):
                    for j in T.Parallel(BD):
                        pt[j] = P[bn, i * BD + j]
                        dpt[j] = (
                            rinv[0] * (T.cast(dOut[bn, i * BD + j], acc) * T.cast(Wt[i * BD + j], acc)) - k2[0] * pt[j]
                        )
                    T.copy(G[bn, :, i * BD : (i + 1) * BD], g_t)
                    T.copy(KV[bn, :, i * BD : (i + 1) * BD], kv_t)
                    T.reduce_max(g_t, mx, dim=0, clear=True)
                    for k, j in T.Parallel(K, BD):
                        g_t[k, j] = T.exp(g_t[k, j] - mx[j])
                    T.reduce_sum(g_t, denom, dim=0, clear=True)
                    for k, j in T.Parallel(K, BD):
                        w = g_t[k, j] / denom[j]
                        dkv_t[k, j] = dpt[j] * w
                        dg_t[k, j] = dpt[j] * w * (kv_t[k, j] - pt[j])
                    T.copy(dkv_t, dKV[bn, :, i * BD : (i + 1) * BD])
                    T.copy(dg_t, dG[bn, :, i * BD : (i + 1) * BD])

        return main

    return _make()


_TL_DTYPE = {torch.float32: "float32", torch.bfloat16: "bfloat16", torch.float16: "float16"}


# --------------------------------------------------------------------------------------
# autograd.Function: generic windowed pool on [N, K, D].
# --------------------------------------------------------------------------------------
class _GatedPoolCore(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, gate, weight, eps):
        assert kv.is_contiguous() and gate.is_contiguous()
        N, K, D = kv.shape
        dt = _TL_DTYPE[kv.dtype]
        fwd = _build_fwd(N, K, D, dt, float(eps))
        out, pooled = fwd(kv, gate, weight)
        ctx.save_for_backward(kv, gate, weight, pooled)
        ctx.eps = float(eps)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        kv, gate, weight, pooled = ctx.saved_tensors
        N, K, D = kv.shape
        dt = _TL_DTYPE[kv.dtype]
        grad_out = grad_out.contiguous()
        bwd = _build_bwd(N, K, D, dt, ctx.eps)
        dkv, dgate = bwd(kv, gate, weight, pooled, grad_out)
        # dweight = sum_n dout[n,:] * pooled[n,:] * r[n].  ``weight`` is the kv_norm gain,
        # frozen under LoRA -> skip the param-grad when autograd says it isn't needed
        # (needs_input_grad[2] == weight); only dkv/dgate feed the upstream proj grads.
        if ctx.needs_input_grad[2]:
            r = torch.rsqrt(pooled.pow(2).mean(-1, keepdim=True) + ctx.eps)  # [N,1]
            dweight = (grad_out.to(torch.float32) * pooled * r).sum(0).to(weight.dtype)
        else:
            dweight = None
        return dkv, dgate, dweight, None


def gated_pool(kv: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """kv, gate: [..., K, D]; weight: [D]. Returns [..., D]. Flattens leading dims to N."""
    *lead, K, D = kv.shape
    N = 1
    for s in lead:
        N *= s
    out = _GatedPoolCore.apply(kv.reshape(N, K, D).contiguous(), gate.reshape(N, K, D).contiguous(), weight, eps)
    return out.reshape(*lead, D)


# --------------------------------------------------------------------------------------
# Fused CSA path: read raw projected [B, S, 2D] tensors and fold the Ca/Cb overlap plus
# position_bias into the TileLang kernels. This removes the torch-side new_kv/new_gate
# materialization used by csa_build_overlap().
# --------------------------------------------------------------------------------------
@functools.cache
def _build_csa_fwd(
    B: int,
    S: int,
    R: int,
    D: int,
    in_dtype: str,
    eps: float,
    threads: int = 128,
):
    usable = (S // R) * R
    n_win = usable // R
    D2 = 2 * D
    K = 2 * R
    BD = _pick_bd(K, D)
    n_tiles = D // BD
    acc = "float32"

    @tilelang.jit(out_idx=[-2, -1])
    def _make():
        @T.prim_func
        def main(
            KV: T.Tensor([B, S, D2], in_dtype),  # type: ignore
            G: T.Tensor([B, S, D2], in_dtype),  # type: ignore
            PB: T.Tensor([R, D2], in_dtype),  # type: ignore
            Wt: T.Tensor([D], in_dtype),  # type: ignore
            Out: T.Tensor([B, n_win, D], in_dtype),  # type: ignore
            Pooled: T.Tensor([B, n_win, D], acc),  # type: ignore
        ):
            with T.Kernel(n_win, B, threads=threads) as (bw, bb):
                p_s = T.alloc_shared([D], acc)
                g_t = T.alloc_fragment([K, BD], acc)
                kv_t = T.alloc_fragment([K, BD], acc)
                mx = T.alloc_fragment([BD], acc)
                denom = T.alloc_fragment([BD], acc)
                num = T.alloc_fragment([BD], acc)

                for i in T.serial(n_tiles):
                    for k, j in T.Parallel(K, BD):
                        d = i * BD + j
                        if k < R:
                            if bw > 0:
                                s = (bw - 1) * R + k
                                g_t[k, j] = T.cast(G[bb, s, d], acc) + T.cast(PB[k, d], acc)
                                kv_t[k, j] = T.cast(KV[bb, s, d], acc)
                            else:
                                g_t[k, j] = NEG
                                kv_t[k, j] = 0.0
                        else:
                            r = k - R
                            s = bw * R + r
                            d2 = D + d
                            g_t[k, j] = T.cast(G[bb, s, d2], acc) + T.cast(PB[r, d2], acc)
                            kv_t[k, j] = T.cast(KV[bb, s, d2], acc)
                    T.reduce_max(g_t, mx, dim=0, clear=True)
                    for k, j in T.Parallel(K, BD):
                        g_t[k, j] = T.exp(g_t[k, j] - mx[j])
                    T.reduce_sum(g_t, denom, dim=0, clear=True)
                    for k, j in T.Parallel(K, BD):
                        kv_t[k, j] = g_t[k, j] * kv_t[k, j]
                    T.reduce_sum(kv_t, num, dim=0, clear=True)
                    for j in T.Parallel(BD):
                        p_s[i * BD + j] = num[j] / denom[j]

                p_f = T.alloc_fragment([D], acc)
                sq = T.alloc_fragment([D], acc)
                ss = T.alloc_fragment([1], acc)
                T.copy(p_s, p_f)
                for d in T.Parallel(D):
                    sq[d] = p_f[d] * p_f[d]
                T.reduce_sum(sq, ss, dim=0, clear=True)
                rinv = T.alloc_fragment([1], acc)
                rinv[0] = T.rsqrt(ss[0] / D + eps)
                for d in T.Parallel(D):
                    Pooled[bb, bw, d] = p_f[d]
                    Out[bb, bw, d] = T.cast(Wt[d], acc) * p_f[d] * rinv[0]

        return main

    return _make()


@functools.cache
def _build_csa_bwd(
    B: int,
    S: int,
    R: int,
    D: int,
    in_dtype: str,
    eps: float,
    threads: int = 128,
):
    usable = (S // R) * R
    n_win = usable // R
    D2 = 2 * D
    K = 2 * R
    BD = _pick_bd(K, D)
    n_tiles = D // BD
    acc = "float32"

    @tilelang.jit(out_idx=[-3, -2, -1])
    def _make():
        @T.prim_func
        def main(
            KV: T.Tensor([B, S, D2], in_dtype),  # type: ignore
            G: T.Tensor([B, S, D2], in_dtype),  # type: ignore
            PB: T.Tensor([R, D2], in_dtype),  # type: ignore
            Wt: T.Tensor([D], in_dtype),  # type: ignore
            P: T.Tensor([B, n_win, D], acc),  # type: ignore
            dOut: T.Tensor([B, n_win, D], in_dtype),  # type: ignore
            dKV: T.Tensor([B, n_win, R, D2], in_dtype),  # type: ignore
            dG: T.Tensor([B, n_win, R, D2], in_dtype),  # type: ignore
            DWP: T.Tensor([B, n_win, D], acc),  # type: ignore  per-row dweight partial
        ):
            with T.Kernel(n_win, B, threads=threads) as (bw, bb):
                p_f = T.alloc_fragment([D], acc)
                sq = T.alloc_fragment([D], acc)
                ap = T.alloc_fragment([D], acc)
                ss = T.alloc_fragment([1], acc)
                s1 = T.alloc_fragment([1], acc)
                rinv = T.alloc_fragment([1], acc)
                k2 = T.alloc_fragment([1], acc)

                for d in T.Parallel(D):
                    p_f[d] = P[bb, bw, d]
                    sq[d] = p_f[d] * p_f[d]
                    ap[d] = (T.cast(dOut[bb, bw, d], acc) * T.cast(Wt[d], acc)) * p_f[d]
                T.reduce_sum(sq, ss, dim=0, clear=True)
                T.reduce_sum(ap, s1, dim=0, clear=True)
                rinv[0] = T.rsqrt(ss[0] / D + eps)
                k2[0] = rinv[0] * rinv[0] * rinv[0] * s1[0] / D

                # Per-row dweight partial (= dOut * pooled * r), fused with the r we
                # already compute here; torch then does one small sum over rows. Avoids
                # a torch-side rsqrt(pow.mean) recompute + fp32 materialization.
                for d in T.Parallel(D):
                    DWP[bb, bw, d] = T.cast(dOut[bb, bw, d], acc) * p_f[d] * rinv[0]

                g_t = T.alloc_fragment([K, BD], acc)
                kv_t = T.alloc_fragment([K, BD], acc)
                mx = T.alloc_fragment([BD], acc)
                denom = T.alloc_fragment([BD], acc)
                pt = T.alloc_fragment([BD], acc)
                dpt = T.alloc_fragment([BD], acc)
                dpd = T.alloc_fragment([BD], acc)
                dkv_t = T.alloc_fragment([K, BD], acc)
                dg_t = T.alloc_fragment([K, BD], acc)

                for i in T.serial(n_tiles):
                    for j in T.Parallel(BD):
                        d = i * BD + j
                        pt[j] = P[bb, bw, d]
                        dpt[j] = rinv[0] * (T.cast(dOut[bb, bw, d], acc) * T.cast(Wt[d], acc)) - k2[0] * pt[j]

                    for k, j in T.Parallel(K, BD):
                        d = i * BD + j
                        if k < R:
                            if bw > 0:
                                s = (bw - 1) * R + k
                                g_t[k, j] = T.cast(G[bb, s, d], acc) + T.cast(PB[k, d], acc)
                                kv_t[k, j] = T.cast(KV[bb, s, d], acc)
                            else:
                                g_t[k, j] = NEG
                                kv_t[k, j] = 0.0
                        else:
                            r = k - R
                            s = bw * R + r
                            d2 = D + d
                            g_t[k, j] = T.cast(G[bb, s, d2], acc) + T.cast(PB[r, d2], acc)
                            kv_t[k, j] = T.cast(KV[bb, s, d2], acc)

                    T.reduce_max(g_t, mx, dim=0, clear=True)
                    for k, j in T.Parallel(K, BD):
                        g_t[k, j] = T.exp(g_t[k, j] - mx[j])
                    T.reduce_sum(g_t, denom, dim=0, clear=True)
                    # Fold dpt/denom into one per-column scalar so w = g_t*dpd avoids a
                    # per-element division (K*BD divides -> BD divides + multiplies).
                    for j in T.Parallel(BD):
                        dpd[j] = dpt[j] / denom[j]
                    for k, j in T.Parallel(K, BD):
                        wkv = g_t[k, j] * dpd[j]
                        dkv_t[k, j] = wkv
                        dg_t[k, j] = wkv * (kv_t[k, j] - pt[j])

                    for k, j in T.Parallel(K, BD):
                        d = i * BD + j
                        if k < R:
                            if bw > 0:
                                dKV[bb, bw - 1, k, d] = dkv_t[k, j]
                                dG[bb, bw - 1, k, d] = dg_t[k, j]
                        else:
                            r = k - R
                            d2 = D + d
                            dKV[bb, bw, r, d2] = dkv_t[k, j]
                            dG[bb, bw, r, d2] = dg_t[k, j]

                    # The final raw Ca half is unused by the CSA overlap and must not
                    # retain uninitialized output-buffer values.
                    if bw == n_win - 1:
                        for r, j in T.Parallel(R, BD):
                            d = i * BD + j
                            dKV[bb, bw, r, d] = 0.0
                            dG[bb, bw, r, d] = 0.0

        return main

    return _make()


class _CSAFusedPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv_seq, gate_seq, position_bias, weight, eps, compress_rate):
        assert kv_seq.is_contiguous() and gate_seq.is_contiguous() and position_bias.is_contiguous()
        B, S, D2 = kv_seq.shape
        assert D2 % 2 == 0
        D = D2 // 2
        R = int(compress_rate)
        usable = (S // R) * R
        n_win = usable // R
        if n_win == 0:
            ctx.empty = True
            ctx.input_shape = kv_seq.shape
            ctx.save_for_backward(weight)
            return kv_seq.new_empty((B, 0, D))

        dt = _TL_DTYPE[kv_seq.dtype]
        fwd = _build_csa_fwd(B, S, R, D, dt, float(eps))
        out, pooled = fwd(kv_seq, gate_seq, position_bias, weight)
        ctx.save_for_backward(kv_seq, gate_seq, position_bias, weight, pooled)
        ctx.eps = float(eps)
        ctx.compress_rate = R
        ctx.usable = usable
        ctx.empty = False
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.empty:
            (weight,) = ctx.saved_tensors
            # weight (kv_norm gain) frozen under LoRA -> gate its grad on needs_input_grad[3]
            # for consistency with the non-empty path (no leak today, autograd drops it when
            # frozen, but keep the two paths symmetric).
            dweight = torch.zeros_like(weight) if ctx.needs_input_grad[3] else None
            return None, None, None, dweight, None, None

        kv_seq, gate_seq, position_bias, weight, pooled = ctx.saved_tensors
        B, S, D2 = kv_seq.shape
        D = D2 // 2
        R = ctx.compress_rate
        usable = ctx.usable
        dt = _TL_DTYPE[kv_seq.dtype]
        grad_out = grad_out.contiguous()
        bwd = _build_csa_bwd(B, S, R, D, dt, ctx.eps)
        dkv_used, dgate_used, dwp = bwd(kv_seq, gate_seq, position_bias, weight, pooled, grad_out)
        # Accumulate in fp32 inside the reduction (sum(dtype=...)) instead of
        # materializing a full fp32 copy of dgate_used first — the .to(float32)
        # round-trip was the dominant backward cost (see RESULTS.md / prof_comp).
        # position_bias is frozen under LoRA -> skip its grad when not needed
        # (needs_input_grad[2] == position_bias); dgate_used is still used for dgate.
        if ctx.needs_input_grad[2]:
            dpos = dgate_used.sum(dim=(0, 1), dtype=torch.float32).to(position_bias.dtype)
        else:
            dpos = None

        if usable == S:
            dkv = dkv_used.reshape(B, usable, D2)
            dgate = dgate_used.reshape(B, usable, D2)
        else:
            dkv = torch.zeros_like(kv_seq)
            dgate = torch.zeros_like(gate_seq)
            dkv[:, :usable] = dkv_used.reshape(B, usable, D2)
            dgate[:, :usable] = dgate_used.reshape(B, usable, D2)

        # dweight = sum_n dOut[n]*pooled[n]*r[n]; the per-row product (dwp, fp32) is
        # emitted by the kernel using the r it already computes, so torch only sums.
        # weight (kv_norm gain) is frozen under LoRA -> skip when not needed.
        if ctx.needs_input_grad[3]:
            dweight = dwp.sum(dim=(0, 1)).to(weight.dtype)
        else:
            dweight = None
        return dkv, dgate, dpos, dweight, None, None


# --------------------------------------------------------------------------------------
# Fused HCA path: read the raw projected [B, S, D] sequence and fold the non-overlapping
# window gather + position_bias into TileLang, so no [B, n_win, W, D] gate intermediate is
# materialized in torch (the same in-kernel-gather trick used for CSA). HCA windows are
# non-overlapping (window size W = compress_rate), so slot k of window bw reads token
# s = bw*W + k directly; there is no Ca/Cb overlap and no -inf masking.
# --------------------------------------------------------------------------------------
@functools.cache
def _build_hca_fwd(B: int, S: int, R: int, D: int, in_dtype: str, eps: float, threads: int = 128):
    usable = (S // R) * R
    n_win = usable // R
    K = R
    BD = _pick_bd(K, D)
    n_tiles = D // BD
    acc = "float32"

    @tilelang.jit(out_idx=[-2, -1])
    def _make():
        @T.prim_func
        def main(
            KV: T.Tensor([B, S, D], in_dtype),  # type: ignore
            G: T.Tensor([B, S, D], in_dtype),  # type: ignore
            PB: T.Tensor([K, D], in_dtype),  # type: ignore
            Wt: T.Tensor([D], in_dtype),  # type: ignore
            Out: T.Tensor([B, n_win, D], in_dtype),  # type: ignore
            Pooled: T.Tensor([B, n_win, D], acc),  # type: ignore
        ):
            with T.Kernel(n_win, B, threads=threads) as (bw, bb):
                p_s = T.alloc_shared([D], acc)
                g_t = T.alloc_fragment([K, BD], acc)
                kv_t = T.alloc_fragment([K, BD], acc)
                mx = T.alloc_fragment([BD], acc)
                denom = T.alloc_fragment([BD], acc)
                num = T.alloc_fragment([BD], acc)

                for i in T.serial(n_tiles):
                    for k, j in T.Parallel(K, BD):
                        d = i * BD + j
                        s = bw * R + k
                        g_t[k, j] = T.cast(G[bb, s, d], acc) + T.cast(PB[k, d], acc)
                        kv_t[k, j] = T.cast(KV[bb, s, d], acc)
                    T.reduce_max(g_t, mx, dim=0, clear=True)
                    for k, j in T.Parallel(K, BD):
                        g_t[k, j] = T.exp(g_t[k, j] - mx[j])
                    T.reduce_sum(g_t, denom, dim=0, clear=True)
                    for k, j in T.Parallel(K, BD):
                        kv_t[k, j] = g_t[k, j] * kv_t[k, j]
                    T.reduce_sum(kv_t, num, dim=0, clear=True)
                    for j in T.Parallel(BD):
                        p_s[i * BD + j] = num[j] / denom[j]

                p_f = T.alloc_fragment([D], acc)
                sq = T.alloc_fragment([D], acc)
                ss = T.alloc_fragment([1], acc)
                T.copy(p_s, p_f)
                for d in T.Parallel(D):
                    sq[d] = p_f[d] * p_f[d]
                T.reduce_sum(sq, ss, dim=0, clear=True)
                rinv = T.alloc_fragment([1], acc)
                rinv[0] = T.rsqrt(ss[0] / D + eps)
                for d in T.Parallel(D):
                    Pooled[bb, bw, d] = p_f[d]
                    Out[bb, bw, d] = T.cast(Wt[d], acc) * p_f[d] * rinv[0]

        return main

    return _make()


@functools.cache
def _build_hca_bwd(B: int, S: int, R: int, D: int, in_dtype: str, eps: float, threads: int = 128):
    usable = (S // R) * R
    n_win = usable // R
    K = R
    BD = _pick_bd(K, D)
    n_tiles = D // BD
    acc = "float32"

    @tilelang.jit(out_idx=[-3, -2, -1])
    def _make():
        @T.prim_func
        def main(
            KV: T.Tensor([B, S, D], in_dtype),  # type: ignore
            G: T.Tensor([B, S, D], in_dtype),  # type: ignore
            PB: T.Tensor([K, D], in_dtype),  # type: ignore
            Wt: T.Tensor([D], in_dtype),  # type: ignore
            P: T.Tensor([B, n_win, D], acc),  # type: ignore
            dOut: T.Tensor([B, n_win, D], in_dtype),  # type: ignore
            dKV: T.Tensor([B, n_win, K, D], in_dtype),  # type: ignore
            dG: T.Tensor([B, n_win, K, D], in_dtype),  # type: ignore
            DWP: T.Tensor([B, n_win, D], acc),  # type: ignore  per-row dweight partial
        ):
            with T.Kernel(n_win, B, threads=threads) as (bw, bb):
                p_f = T.alloc_fragment([D], acc)
                sq = T.alloc_fragment([D], acc)
                ap = T.alloc_fragment([D], acc)
                ss = T.alloc_fragment([1], acc)
                s1 = T.alloc_fragment([1], acc)
                rinv = T.alloc_fragment([1], acc)
                k2 = T.alloc_fragment([1], acc)

                for d in T.Parallel(D):
                    p_f[d] = P[bb, bw, d]
                    sq[d] = p_f[d] * p_f[d]
                    ap[d] = (T.cast(dOut[bb, bw, d], acc) * T.cast(Wt[d], acc)) * p_f[d]
                T.reduce_sum(sq, ss, dim=0, clear=True)
                T.reduce_sum(ap, s1, dim=0, clear=True)
                rinv[0] = T.rsqrt(ss[0] / D + eps)
                k2[0] = rinv[0] * rinv[0] * rinv[0] * s1[0] / D

                # Per-row dweight partial (= dOut * pooled * r); torch then sums over rows.
                for d in T.Parallel(D):
                    DWP[bb, bw, d] = T.cast(dOut[bb, bw, d], acc) * p_f[d] * rinv[0]

                g_t = T.alloc_fragment([K, BD], acc)
                kv_t = T.alloc_fragment([K, BD], acc)
                mx = T.alloc_fragment([BD], acc)
                denom = T.alloc_fragment([BD], acc)
                pt = T.alloc_fragment([BD], acc)
                dpt = T.alloc_fragment([BD], acc)
                dpd = T.alloc_fragment([BD], acc)
                dkv_t = T.alloc_fragment([K, BD], acc)
                dg_t = T.alloc_fragment([K, BD], acc)

                for i in T.serial(n_tiles):
                    for j in T.Parallel(BD):
                        d = i * BD + j
                        pt[j] = P[bb, bw, d]
                        dpt[j] = rinv[0] * (T.cast(dOut[bb, bw, d], acc) * T.cast(Wt[d], acc)) - k2[0] * pt[j]

                    for k, j in T.Parallel(K, BD):
                        d = i * BD + j
                        s = bw * R + k
                        g_t[k, j] = T.cast(G[bb, s, d], acc) + T.cast(PB[k, d], acc)
                        kv_t[k, j] = T.cast(KV[bb, s, d], acc)

                    T.reduce_max(g_t, mx, dim=0, clear=True)
                    for k, j in T.Parallel(K, BD):
                        g_t[k, j] = T.exp(g_t[k, j] - mx[j])
                    T.reduce_sum(g_t, denom, dim=0, clear=True)
                    # Fold dpt/denom into one per-column scalar (see CSA bwd note).
                    for j in T.Parallel(BD):
                        dpd[j] = dpt[j] / denom[j]
                    for k, j in T.Parallel(K, BD):
                        wkv = g_t[k, j] * dpd[j]
                        dkv_t[k, j] = wkv
                        dg_t[k, j] = wkv * (kv_t[k, j] - pt[j])

                    for k, j in T.Parallel(K, BD):
                        d = i * BD + j
                        dKV[bb, bw, k, d] = dkv_t[k, j]
                        dG[bb, bw, k, d] = dg_t[k, j]

        return main

    return _make()


class _HCAFusedPool(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv_seq, gate_seq, position_bias, weight, eps, compress_rate):
        assert kv_seq.is_contiguous() and gate_seq.is_contiguous() and position_bias.is_contiguous()
        B, S, D = kv_seq.shape
        R = int(compress_rate)
        usable = (S // R) * R
        n_win = usable // R
        if n_win == 0:
            ctx.empty = True
            ctx.save_for_backward(weight)
            return kv_seq.new_empty((B, 0, D))

        dt = _TL_DTYPE[kv_seq.dtype]
        fwd = _build_hca_fwd(B, S, R, D, dt, float(eps))
        out, pooled = fwd(kv_seq, gate_seq, position_bias, weight)
        ctx.save_for_backward(kv_seq, gate_seq, position_bias, weight, pooled)
        ctx.eps = float(eps)
        ctx.compress_rate = R
        ctx.usable = usable
        ctx.empty = False
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.empty:
            (weight,) = ctx.saved_tensors
            # weight (kv_norm gain) frozen under LoRA -> gate its grad on needs_input_grad[3]
            # for consistency with the non-empty path (no leak today, autograd drops it when
            # frozen, but keep the two paths symmetric).
            dweight = torch.zeros_like(weight) if ctx.needs_input_grad[3] else None
            return None, None, None, dweight, None, None

        kv_seq, gate_seq, position_bias, weight, pooled = ctx.saved_tensors
        B, S, D = kv_seq.shape
        R = ctx.compress_rate
        usable = ctx.usable
        dt = _TL_DTYPE[kv_seq.dtype]
        grad_out = grad_out.contiguous()
        bwd = _build_hca_bwd(B, S, R, D, dt, ctx.eps)
        dkv_used, dgate_used, dwp = bwd(kv_seq, gate_seq, position_bias, weight, pooled, grad_out)
        # fp32 accumulation folded into the reduction (see CSA backward note).
        # position_bias frozen under LoRA -> skip its grad when not needed (idx 2).
        if ctx.needs_input_grad[2]:
            dpos = dgate_used.sum(dim=(0, 1), dtype=torch.float32).to(position_bias.dtype)
        else:
            dpos = None

        if usable == S:
            dkv = dkv_used.reshape(B, usable, D)
            dgate = dgate_used.reshape(B, usable, D)
        else:
            dkv = torch.zeros_like(kv_seq)
            dgate = torch.zeros_like(gate_seq)
            dkv[:, :usable] = dkv_used.reshape(B, usable, D)
            dgate[:, :usable] = dgate_used.reshape(B, usable, D)

        # dweight per-row product emitted by the kernel (see CSA backward note).
        # weight (kv_norm gain) frozen under LoRA -> skip its grad when not needed (idx 3).
        if ctx.needs_input_grad[3]:
            dweight = dwp.sum(dim=(0, 1)).to(weight.dtype)
        else:
            dweight = None
        return dkv, dgate, dpos, dweight, None, None


# --------------------------------------------------------------------------------------
# HCA / CSA wrappers.
# --------------------------------------------------------------------------------------
def hca_compress_unfused(kv_seq, gate_seq, position_bias, weight, eps, compress_rate: int = 128):
    """Pre-fusion path: torch builds the windowed [B, n_win, W, D] gate (+position_bias)
    before the pool kernel. Kept for before/after benchmarks."""
    B, S, D = kv_seq.shape
    usable = (S // compress_rate) * compress_rate
    n_win = usable // compress_rate
    kv = kv_seq[:, :usable].reshape(B, n_win, compress_rate, D)
    gate = gate_seq[:, :usable].reshape(B, n_win, compress_rate, D) + position_bias.to(gate_seq.dtype)
    return gated_pool(kv, gate, weight, eps)


def hca_compress_fused(kv_seq, gate_seq, position_bias, weight, eps, compress_rate: int = 128):
    if compress_rate <= 0:
        raise ValueError(f"compress_rate must be positive, got {compress_rate}")
    if kv_seq.dtype != gate_seq.dtype:
        raise TypeError(f"kv_seq/gate_seq dtype mismatch: {kv_seq.dtype} vs {gate_seq.dtype}")
    if kv_seq.shape != gate_seq.shape:
        raise ValueError(f"kv_seq/gate_seq shape mismatch: {tuple(kv_seq.shape)} vs {tuple(gate_seq.shape)}")
    position_bias = position_bias.to(dtype=gate_seq.dtype).contiguous()
    return _HCAFusedPool.apply(
        kv_seq.contiguous(),
        gate_seq.contiguous(),
        position_bias,
        weight.contiguous(),
        eps,
        int(compress_rate),
    )


def hca_compress(kv_seq, gate_seq, position_bias, weight, eps, compress_rate: int = 128):
    return hca_compress_fused(kv_seq, gate_seq, position_bias, weight, eps, compress_rate)


def csa_compress_unfused(kv_seq, gate_seq, position_bias, weight, eps, compress_rate: int = 4):
    new_kv, new_gate = csa_build_overlap(kv_seq, gate_seq, position_bias, compress_rate)
    return gated_pool(new_kv.contiguous(), new_gate.contiguous(), weight, eps)


def csa_compress_fused(kv_seq, gate_seq, position_bias, weight, eps, compress_rate: int = 4):
    if compress_rate <= 0:
        raise ValueError(f"compress_rate must be positive, got {compress_rate}")
    if kv_seq.dtype != gate_seq.dtype:
        raise TypeError(f"kv_seq/gate_seq dtype mismatch: {kv_seq.dtype} vs {gate_seq.dtype}")
    if kv_seq.shape != gate_seq.shape:
        raise ValueError(f"kv_seq/gate_seq shape mismatch: {tuple(kv_seq.shape)} vs {tuple(gate_seq.shape)}")
    if kv_seq.shape[-1] % 2 != 0:
        raise ValueError(f"CSA projection dim must be even, got {kv_seq.shape[-1]}")
    position_bias = position_bias.to(dtype=gate_seq.dtype).contiguous()
    return _CSAFusedPool.apply(
        kv_seq.contiguous(),
        gate_seq.contiguous(),
        position_bias,
        weight.contiguous(),
        eps,
        int(compress_rate),
    )


def csa_compress(kv_seq, gate_seq, position_bias, weight, eps, compress_rate: int = 4):
    return csa_compress_fused(kv_seq, gate_seq, position_bias, weight, eps, compress_rate)
