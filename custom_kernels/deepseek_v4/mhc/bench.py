"""Benchmark the tilelang mHC HyperConnection vs torch.compile(reference).

Reports per-call latency (ms) and peak memory (MB) for forward, backward, and
forward+backward at V4-Flash training shapes (bf16), plus the per-call count
reminder: mHC runs 2× per layer × 43 layers = 86 calls per model forward.

Run: CUDA_VISIBLE_DEVICES=1 python bench.py
"""

import time

import kernel as K
import reference as R
import torch

DEV = "cuda"
H, D, MIX, M = K.H, K.HIDDEN, K.MIX, K.HIDDEN * K.H
SITES = 86  # mHC invocations per model forward pass


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


def _inputs(B, S, dtype, rg):
    x = torch.randn(B, S, H, D, device=DEV, dtype=dtype, requires_grad=rg)
    fn = (torch.randn(MIX, M, device=DEV, dtype=torch.float32) / M**0.5).requires_grad_(rg)
    base = (torch.randn(MIX, device=DEV, dtype=torch.float32) * 0.1).requires_grad_(rg)
    scale = (torch.rand(3, device=DEV, dtype=torch.float32) + 0.5).requires_grad_(rg)
    return x, fn, base, scale


def main():
    ref_compiled = torch.compile(R.hyper_connection_forward)
    print(
        f"{'shape':>12} {'impl':>14} {'fwd ms':>9} {'bwd ms':>9} {'fwd+bwd':>9} " f"{'peak MB':>9} {'×86 fb ms':>10}"
    )
    for B, S in [(1, 2048), (1, 8192), (1, 16384)]:
        x, fn, base, scale = _inputs(B, S, torch.bfloat16, rg=True)
        impls = {"tilelang": K.hyper_connection, "torch.compile": ref_compiled}
        out = {k: v(x, fn, base, scale) for k, v in impls.items()}
        gp = torch.randn_like(out["tilelang"][0])
        gc = torch.randn_like(out["tilelang"][1])
        gl = torch.randn_like(out["tilelang"][2].float())

        def fwd(run):
            return run(x, fn, base, scale)

        def fwdbwd(run):
            x.grad = fn.grad = base.grad = scale.grad = None
            p, c, l = run(x, fn, base, scale)
            ((p.float() * gp.float()).sum() + (c * gc).sum() + (l.float() * gl).sum()).backward()

        for name, run in impls.items():
            f_ms, _ = _bench(lambda: fwd(run))
            fb_ms, peak = _bench(lambda: fwdbwd(run))
            print(
                f"{f'B{B}S{S}':>12} {name:>14} {f_ms:9.3f} {fb_ms - f_ms:9.3f} "
                f"{fb_ms:9.3f} {peak:9.1f} {fb_ms * SITES:10.1f}"
            )
        print()


if __name__ == "__main__":
    main()
