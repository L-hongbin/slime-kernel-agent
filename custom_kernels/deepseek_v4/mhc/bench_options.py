"""Trade-off bench for the mHC backward-recompute fix (B1 task).

Measures fwd / bwd / fwd+bwd / bwd-over-fwd ratio / peak-MB for the candidate
trainable paths, at V4-Flash training shapes (bf16):

  (b)  hyper_connection         : OUR norm+GEMM (saves raw/inv) + fused tilelang
                                  Sinkhorn+collapse; backward has NO GEMM recompute.
  (c)  hyper_connection_sglang  : sglang's fast prenorm GEMM stage captures raw/inv
                                  + fused tilelang Sinkhorn+collapse; NO recompute.
  (a)  sglang-fwd + extra-norm  : sglang mhc_pre forward + a SECOND our-norm_gemm in
                                  the forward purely to capture raw/inv (no recompute,
                                  but two GEMMs in the forward) — measured for the
                                  trade-off table; expected to be dominated.
  base sglang-recompute         : the OLD design (sglang mhc_pre fwd, backward
                                  recomputes the full norm+GEMM). Re-implemented here
                                  so the baseline is measured on the same GPU/run.
  ref  torch.compile(reference) : non-tilelang reference, for context.

Run: CUDA_VISIBLE_DEVICES=1 python bench_options.py
"""

import os
import time

os.environ.setdefault("TILELANG_CACHE_DIR", "/tmp/tilelang_cache")
os.environ.setdefault("TILELANG_TMP_DIR", "/tmp/tilelang_cache/tmp")

import kernel as K
import reference as R
import torch

DEV = "cuda"
H, D, MIX, M = K.H, K.HIDDEN, K.MIX, K.HIDDEN * K.H
RMS_EPS = K.RMS_EPS
HC_EPS = K.HC_EPS
SINKHORN_ITERS = K.SINKHORN_ITERS


def _bench(fn, iters=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t) / iters * 1e3
    peak = torch.cuda.max_memory_allocated() / 1e6
    return ms, peak


def _inputs(N, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    res = torch.randn(N, H, D, device=DEV, dtype=torch.bfloat16, generator=g)
    fn = torch.randn(MIX, M, device=DEV, dtype=torch.float32, generator=g) / M**0.5
    base = torch.randn(MIX, device=DEV, dtype=torch.float32, generator=g) * 0.1
    scale = torch.rand(3, device=DEV, dtype=torch.float32, generator=g) + 0.5
    return res, fn, base, scale


# ---- baseline: sglang mhc_pre forward + backward that RECOMPUTES norm+GEMM ----
class _SglangRecompute(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hs, fn, base, scale):
        from sglang.srt.layers.mhc import mhc_pre

        B, S, Hh, Dd = hs.shape
        N = B * S
        residual = hs.reshape(N, H, D).contiguous()
        fn32 = fn.float().contiguous()
        base_f = base.float().contiguous()
        scale_f = scale.float().contiguous()
        post, comb, coll = mhc_pre(
            residual=residual,
            fn=fn32,
            hc_scale=scale_f,
            hc_base=base_f,
            rms_eps=RMS_EPS,
            hc_pre_eps=HC_EPS,
            hc_sinkhorn_eps=HC_EPS,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=SINKHORN_ITERS,
        )
        ctx.save_for_backward(residual.reshape(N, M), fn32, base_f, scale_f)
        ctx.shape = (B, S, D, N, M, torch.bfloat16, "bfloat16")
        return post.reshape(B, S, H), comb.reshape(B, S, H, H), coll.reshape(B, S, D)

    @staticmethod
    def backward(ctx, d_post, d_comb, d_coll):
        x2d, fn32, base, scale = ctx.saved_tensors
        B, S, Dd, N, Mm, in_dtype, ds = ctx.shape
        fnp = K._pad_fn(fn32)
        ones = K._ones(fn32.device)
        raw_p, inv = K._build_norm_gemm(N, Mm, ds)(x2d, fnp, ones)
        raw = raw_p[:, :MIX].contiguous()
        pre = torch.sigmoid(raw[:, :H] * scale[0] + base[:H]) + HC_EPS
        return K._hc_backward(
            x2d, fnp, raw, inv, pre, scale, base, ctx.shape, d_post, d_comb, d_coll, ctx.needs_input_grad
        )


# ---- option (a): sglang mhc_pre fwd + extra our-norm_gemm in fwd (save raw/inv) --
class _SglangExtraNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hs, fn, base, scale):
        from sglang.srt.layers.mhc import mhc_pre

        B, S, Hh, Dd = hs.shape
        N = B * S
        residual = hs.reshape(N, H, D).contiguous()
        x2d = residual.reshape(N, M)
        fn32 = fn.float().contiguous()
        fnp = K._pad_fn(fn32)
        ones = K._ones(fn32.device)
        base_f = base.float().contiguous()
        scale_f = scale.float().contiguous()
        post, comb, coll = mhc_pre(
            residual=residual,
            fn=fn32,
            hc_scale=scale_f,
            hc_base=base_f,
            rms_eps=RMS_EPS,
            hc_pre_eps=HC_EPS,
            hc_sinkhorn_eps=HC_EPS,
            hc_post_mult_value=2.0,
            sinkhorn_repeat=SINKHORN_ITERS,
        )
        # EXTRA second GEMM purely to capture raw/inv for the backward.
        raw_p, inv = K._build_norm_gemm(N, M, "bfloat16")(x2d, fnp, ones)
        raw = raw_p[:, :MIX].contiguous()
        pre = torch.sigmoid(raw[:, :H] * scale_f[0] + base_f[:H]) + HC_EPS
        ctx.save_for_backward(x2d, fnp, raw, inv, pre, scale_f, base_f)
        ctx.shape = (B, S, D, N, M, torch.bfloat16, "bfloat16")
        return post.reshape(B, S, H), comb.reshape(B, S, H, H), coll.reshape(B, S, D)

    @staticmethod
    def backward(ctx, d_post, d_comb, d_coll):
        x2d, fnp, raw, inv, pre, scale, base = ctx.saved_tensors
        return K._hc_backward(
            x2d, fnp, raw, inv, pre, scale, base, ctx.shape, d_post, d_comb, d_coll, ctx.needs_input_grad
        )


def _measure(name, run, hs, fn, base, scale, gp, gc, gl):
    def fwd():
        return run(hs, fn, base, scale)

    def fwdbwd():
        hs.grad = fn.grad = base.grad = scale.grad = None
        p, c, l = run(hs, fn, base, scale)
        ((p.float() * gp).sum() + (c.float() * gc).sum() + (l.float() * gl).sum()).backward()

    f_ms, _ = _bench(fwd)
    fb_ms, peak = _bench(fwdbwd)
    b_ms = fb_ms - f_ms
    ratio = b_ms / f_ms if f_ms > 0 else float("nan")
    print(
        f"   {name:<26} fwd {f_ms:7.3f}  bwd {b_ms:7.3f}  fwd+bwd {fb_ms:7.3f}  "
        f"bwd/fwd {ratio:5.2f}x  peak {peak:8.1f}"
    )
    return dict(fwd=f_ms, bwd=b_ms, total=fb_ms, ratio=ratio, peak=peak)


def main():
    from sglang.srt.environ import envs

    print("torch", torch.__version__, "| GPU", torch.cuda.get_device_name(0))
    print("SGLANG_OPT_DEEPGEMM_HC_PRENORM =", envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get())
    print("MHC_TORCH_MIDDLE =", os.environ.get("MHC_TORCH_MIDDLE", "0"))

    ref_compiled = torch.compile(R.hyper_connection_forward)

    for N in (2048, 8192, 16384):
        res, fn, base, scale = _inputs(N)
        hs = res.view(1, N, H, D).clone().requires_grad_(True)
        fn_g = fn.clone().requires_grad_(True)
        base_g = base.clone().requires_grad_(True)
        scale_g = scale.clone().requires_grad_(True)
        p0, c0, l0 = K.hyper_connection_sglang(hs, fn_g, base_g, scale_g)
        gp = torch.randn_like(p0.float())
        gc = torch.randn_like(c0.float())
        gl = torch.randn_like(l0.float())

        print(f"\nN={N} (B1·S{N}) bf16")
        runs = [
            ("(c) sglang-GEMM + fused", K.hyper_connection_sglang),
            ("(b) ours norm+fused", K.hyper_connection),
            ("(a) sglang-fwd+extra-norm", _SglangExtraNorm.apply),
            ("base sglang-recompute", _SglangRecompute.apply),
            ("ref torch.compile", ref_compiled),
        ]
        for nm, run in runs:
            try:
                _measure(nm, run, hs, fn_g, base_g, scale_g, gp, gc, gl)
            except Exception as e:
                print(f"   {nm:<26} FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
