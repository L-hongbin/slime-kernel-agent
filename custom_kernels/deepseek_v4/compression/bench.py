"""Benchmark the V4-Flash compression pool: tilelang vs torch.compile(reference).

Measures fwd, bwd, fwd+bwd latency and peak memory at training shapes for HCA (W=128)
and CSA (overlap, ratio=4), head_dim=512, bf16.

Run:  CUDA_VISIBLE_DEVICES=2 python bench.py
"""

import reference as ref
import torch
from kernel import csa_compress, csa_compress_unfused, hca_compress

DEV = "cuda"
D = 512
EPS = 1e-6
DT = torch.bfloat16


def _bench(fn, *args, iters=50, warmup=10, do_bwd=False):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        if do_bwd:
            for a in args:
                if a.grad is not None:
                    a.grad = None
            out = fn(*args)
            out.backward(torch.ones_like(out))
        else:
            with torch.no_grad():
                fn(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        if do_bwd:
            for a in args:
                if a.grad is not None:
                    a.grad = None
            out = fn(*args)
            out.backward(torch.ones_like(out))
        else:
            with torch.no_grad():
                fn(*args)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    peak = torch.cuda.max_memory_allocated() / 1e6
    return ms, peak


def make_inputs(kind, seq, B=1, req_grad=False):
    rate = 128 if kind == "HCA" else 4
    proj = D if kind == "HCA" else 2 * D
    # build proper leaves (detach so requires_grad tensors are graph leaves, not Mul nodes)
    kv = (torch.randn(B, seq, proj, device=DEV, dtype=DT) * 0.5).detach().requires_grad_(req_grad)
    gate = (torch.randn(B, seq, proj, device=DEV, dtype=DT) * 0.5).detach().requires_grad_(req_grad)
    pb = (torch.randn(rate, proj, device=DEV, dtype=DT) * 0.3).detach().requires_grad_(req_grad)
    w = (1.0 + 0.1 * torch.randn(D, device=DEV, dtype=DT)).detach().requires_grad_(req_grad)
    return kv, gate, pb, w, rate


def main():
    print(f"device={torch.cuda.get_device_name(0)}  dtype={DT}  head_dim={D}\n")
    header = (
        f"{'case':<22}{'mode':<10}{'variant':<10}"
        f"{'tilelang ms':>13}{'compile ms':>13}{'speedup':>10}{'tl_mem MB':>12}{'co_mem MB':>12}"
    )
    for kind in ("HCA", "CSA"):
        rate = 128 if kind == "HCA" else 4
        ref_fn = ref.hca_compress_ref if kind == "HCA" else ref.csa_compress_ref
        variants = (
            [("generic", hca_compress)]
            if kind == "HCA"
            else [
                ("fused", csa_compress),
                ("unfused", csa_compress_unfused),
            ]
        )
        compiled = torch.compile(ref_fn, fullgraph=False)
        for seq in (8192, 32768):
            n_win = seq // rate
            print(f"\n# {kind}  seq={seq}  n_win={n_win}")
            print(header)
            for mode, do_bwd in (("fwd", False), ("fwd+bwd", True)):
                for variant, tl_fn in variants:
                    rg = do_bwd
                    kv, gate, pb, w, _ = make_inputs(kind, seq, req_grad=rg)
                    tl = lambda: tl_fn(kv, gate, pb, w, EPS, rate)
                    co = lambda: compiled(kv, gate, pb, w, EPS, rate)
                    # warm compile
                    try:
                        o = co()
                        if do_bwd:
                            o.backward(torch.ones_like(o))
                            for a in (kv, gate, pb, w):
                                a.grad = None
                    except Exception as e:
                        print("  torch.compile failed:", str(e)[:120])
                        co = None
                    tl_ms, tl_mem = _bench(tl, iters=50, do_bwd=do_bwd)
                    if co is not None:
                        co_ms, co_mem = _bench(co, iters=50, do_bwd=do_bwd)
                        sp = co_ms / tl_ms
                        print(
                            f"{kind+' '+str(seq):<22}{mode:<10}{variant:<10}"
                            f"{tl_ms:>13.4f}{co_ms:>13.4f}{sp:>9.2f}x{tl_mem:>12.1f}{co_mem:>12.1f}"
                        )
                    else:
                        print(
                            f"{kind+' '+str(seq):<22}{mode:<10}{variant:<10}"
                            f"{tl_ms:>13.4f}{'n/a':>13}{'n/a':>10}{tl_mem:>12.1f}{'n/a':>12}"
                        )


if __name__ == "__main__":
    main()
