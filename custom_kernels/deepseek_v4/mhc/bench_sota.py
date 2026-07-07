"""Forward-only community-SOTA comparison: our tilelang mHC vs sglang's
production tilelang mHC.

sglang ships the DeepSeek-V4 mHC as TileLang kernels in
``sglang.srt.layers.mhc``.  The direct analog of our ``hyper_connection``
*forward* is ``mhc_pre``: it consumes the residual streams + the mix ``fn`` and
produces ``(post_mix, comb_mix, layer_input)`` — i.e. the post placement
weights, the Sinkhorn-balanced comb matrix, and the pre-weighted *collapsed*
sublayer input.  That is exactly the three things our forward returns
(``post, comb, collapsed``).  ``mhc_post`` is the separate sublayer-output
placement step (no analog in our forward), so the apples-to-apples forward
baseline is ``mhc_pre``.

We exercise sglang's **pure TileLang** path (``SGLANG_OPT_DEEPGEMM_HC_PRENORM=0``
-> TileLang split-k GEMM + ``mhc_pre_big_fuse_tilelang``) and, when available,
its production default (DeepGEMM tf32 prenorm GEMM + the same TileLang big_fuse).

Caveats (documented honestly):
  * sglang runs the mix GEMM in bf16/tf32 and the Sinkhorn entirely inside
    TileLang (one kernel/token); our forward leaves the tiny 4x4x20 Sinkhorn in
    torch.compile and the GEMM in tilelang fp32(tf32-core).  So this is
    TileLang-vs-TileLang for the heavy GEMM+collapse, but the Sinkhorn middle
    differs in *where* it runs.
  * sglang fuses everything into 1-2 launches (it is an inference kernel with no
    backward); ours saves tensors for an analytic backward.  Forward-only here.
  * Precision: both take bf16 residual + fp32 fn.  sglang's RMSNorm reads the
    bf16 residual; ours does too.

Run: CUDA_VISIBLE_DEVICES=1 python bench_sota.py
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
RMS_EPS = R.RMS_NORM_EPS
HC_EPS = R.HC_EPS
SINKHORN_ITERS = R.HC_SINKHORN_ITERS
POST_MULT = 2.0


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
    # residual / hidden streams, bf16 (training/inference dtype)
    res = torch.randn(N, H, D, device=DEV, dtype=torch.bfloat16, generator=g)
    fn = torch.randn(MIX, M, device=DEV, dtype=torch.float32, generator=g) / M**0.5
    base = torch.randn(MIX, device=DEV, dtype=torch.float32, generator=g) * 0.1
    scale = torch.rand(3, device=DEV, dtype=torch.float32, generator=g) + 0.5
    return res, fn, base, scale


def _correctness(N, mhc_pre):
    """Sanity: sglang mhc_pre vs our forward at matched inputs (bf16)."""
    res, fn, base, scale = _inputs(N, seed=123)
    # ours: hidden_streams [B,S,H,D]
    post_o, comb_o, coll_o = K.hyper_connection(res.view(1, N, H, D), fn, base, scale)
    post_o = post_o.view(N, H).float()
    comb_o = comb_o.view(N, H, H).float()
    coll_o = coll_o.view(N, D).float()
    # sglang mhc_pre
    post_s, comb_s, y_s = mhc_pre(
        residual=res,
        fn=fn,
        hc_scale=scale,
        hc_base=base,
        rms_eps=RMS_EPS,
        hc_pre_eps=HC_EPS,
        hc_sinkhorn_eps=HC_EPS,
        hc_post_mult_value=POST_MULT,
        sinkhorn_repeat=SINKHORN_ITERS,
    )
    post_s = post_s.reshape(N, H).float()
    comb_s = comb_s.reshape(N, H, H).float()
    y_s = y_s.reshape(N, D).float()

    def rel(a, b):
        return (a - b).abs().max().item() / (b.abs().max().item() + 1e-9)

    return {
        "post": rel(post_o, post_s),
        "comb": rel(comb_o, comb_s),
        "collapsed": rel(coll_o, y_s),
    }


def _fwdbwd(run, hs, fn, base, scale, gp, gc, gl):
    hs.grad = fn.grad = base.grad = scale.grad = None
    p, c, l = run(hs, fn, base, scale)
    ((p.float() * gp).sum() + (c.float() * gc).sum() + (l.float() * gl).sum()).backward()


def main():
    from sglang.srt.environ import envs
    from sglang.srt.layers.mhc import mhc_pre

    print("torch", torch.__version__, "| GPU", torch.cuda.get_device_name(0))
    print("SGLANG_OPT_DEEPGEMM_HC_PRENORM =", envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get())
    print()

    for N in (2048, 8192):
        res, fn, base, scale = _inputs(N)
        hs = res.view(1, N, H, D)

        ours = lambda: K.hyper_connection(hs, fn, base, scale)
        new = lambda: K.hyper_connection_sglang(hs, fn, base, scale)
        sgl = lambda: mhc_pre(
            residual=res,
            fn=fn,
            hc_scale=scale,
            hc_base=base,
            rms_eps=RMS_EPS,
            hc_pre_eps=HC_EPS,
            hc_sinkhorn_eps=HC_EPS,
            hc_post_mult_value=POST_MULT,
            sinkhorn_repeat=SINKHORN_ITERS,
        )

        # correctness sanity at this N (hybrid vs sglang mhc_pre)
        errs = _correctness(N, mhc_pre)
        print(
            f"N={N} (B1·S{N}) bf16  max-rel-err vs hybrid: "
            f"post {errs['post']:.2e}  comb {errs['comb']:.2e}  "
            f"collapsed {errs['collapsed']:.2e}"
        )

        # ---- FORWARD-only latency ----
        o_ms, o_pk = _bench(ours)
        n_ms, n_pk = _bench(new)
        s_ms, s_pk = _bench(sgl)
        print(f"   FWD ms:  hybrid {o_ms:7.3f}  |  new(sgl-fwd) {n_ms:7.3f}  " f"|  sglang mhc_pre {s_ms:7.3f}")
        print(f"            new/sglang {n_ms / s_ms:5.2f}×   hybrid/sglang {o_ms / s_ms:5.2f}×")

        # ---- FORWARD+BACKWARD latency (trainable paths only) ----
        hs_g = res.view(1, N, H, D).clone().requires_grad_(True)
        fn_g = fn.clone().requires_grad_(True)
        base_g = base.clone().requires_grad_(True)
        scale_g = scale.clone().requires_grad_(True)
        p0, c0, l0 = K.hyper_connection_sglang(hs_g, fn_g, base_g, scale_g)
        gp = torch.randn_like(p0.float())
        gc = torch.randn_like(c0.float())
        gl = torch.randn_like(l0.float())
        ofb, _ = _bench(lambda: _fwdbwd(K.hyper_connection, hs_g, fn_g, base_g, scale_g, gp, gc, gl))
        nfb, nfb_pk = _bench(lambda: _fwdbwd(K.hyper_connection_sglang, hs_g, fn_g, base_g, scale_g, gp, gc, gl))
        print(f"   FWD+BWD ms:  hybrid {ofb:7.3f}  |  new(sgl-fwd) {nfb:7.3f}   " f"(sglang mhc_pre has no backward)")
        print()


if __name__ == "__main__":
    main()
