"""Benchmark the DeepSeek-V4-Flash core attention (A1) tilelang kernel.

Run:  CUDA_VISIBLE_DEVICES=0 python bench.py

Compares the tilelang kernel (fwd, bwd, fwd+bwd) against torch.compile(reference)
at training shapes, reporting latency / TFLOP-s / peak memory, and flags shapes
where the eager/compiled reference OOMs but the kernel survives.

FLOPs are counted over the structurally-valid (query, key) pairs (sliding band +
compressed causal-threshold), 4*D per pair forward (QKᵀ + PV), so TFLOP/s reflect
useful work, not dense O(S^2).
"""

import sys

import torch

sys.path.insert(0, ".")
import kernel as K  # noqa: E402
from reference import attention_reference  # noqa: E402

DEV = "cuda"
H, D = 64, 512


def valid_pairs(S, Tcomp, W, m):
    """Per-(head,batch) count of non-masked (query,key) pairs."""
    i = torch.arange(S, dtype=torch.float64)
    raw = torch.clamp(i + 1, max=W).sum().item()  # sliding band
    comp = 0.0
    if Tcomp > 0:
        thr = torch.clamp((i + 1) // m, max=Tcomp)
        comp = thr.sum().item()
    return raw + comp


from _bench_common import cuda_time  # noqa: E402


def peak_mem_mb(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


def bench_shape(B, S, m, with_comp, W):
    Tcomp = (S // m) if with_comp else 0
    P = valid_pairs(S, Tcomp, W, m)
    fwd_flops = B * H * 4 * D * P  # QKᵀ + PV
    tag = f"B{B} S{S:>5} m{m:>3} comp{int(with_comp)} W{W}"

    q = torch.randn(B, H, S, D, device=DEV, dtype=torch.bfloat16) * 0.3
    kr = torch.randn(B, 1, S, D, device=DEV, dtype=torch.bfloat16) * 0.3
    kc = (torch.randn(B, 1, Tcomp, D, device=DEV, dtype=torch.bfloat16) * 0.3) if with_comp else None
    sinks = torch.randn(H, device=DEV, dtype=torch.float32)

    res = {"tag": tag, "P": P, "fwd_tflops_ref_count": fwd_flops}

    # ---- tilelang kernel ----
    def k_fwd():
        return K.v4flash_attention(q, kr, kc, sinks, W, m)

    res["k_fwd_ms"] = cuda_time(k_fwd, iters=20, warmup=5)
    res["k_fwd_tflops"] = fwd_flops / (res["k_fwd_ms"] * 1e-3) / 1e12
    res["k_fwd_mem"] = peak_mem_mb(k_fwd)

    qg = q.clone().requires_grad_(True)
    krg = kr.clone().requires_grad_(True)
    kcg = None if kc is None else kc.clone().requires_grad_(True)
    sg = sinks.clone().requires_grad_(True)
    g = torch.randn(B, H, S, D, device=DEV, dtype=torch.bfloat16)

    def k_fb():
        for t in (qg, krg, kcg, sg):
            if t is not None and t.grad is not None:
                t.grad = None
        o = K.v4flash_attention(qg, krg, kcg, sg, W, m)
        o.backward(g)

    res["k_fwdbwd_ms"] = cuda_time(k_fb, iters=20, warmup=5)
    res["k_fwdbwd_mem"] = peak_mem_mb(k_fb)
    res["k_bwd_ms"] = res["k_fwdbwd_ms"] - res["k_fwd_ms"]
    res["bwd_fwd_ratio"] = res["k_bwd_ms"] / res["k_fwd_ms"]

    # ---- torch.compile(reference) ----
    res["ref"] = {}
    try:
        cref = torch.compile(attention_reference, dynamic=False)

        def r_fwd():
            return cref(q.float(), kr.float(), None if kc is None else kc.float(), sinks, W, m)

        res["ref"]["fwd_ms"] = cuda_time(r_fwd, iters=10, warmup=3)
        res["ref"]["fwd_tflops"] = fwd_flops / (res["ref"]["fwd_ms"] * 1e-3) / 1e12
        res["ref"]["fwd_mem"] = peak_mem_mb(r_fwd)

        qr = q.float().clone().requires_grad_(True)
        krr = kr.float().clone().requires_grad_(True)
        kcr = None if kc is None else kc.float().clone().requires_grad_(True)
        sr = sinks.clone().requires_grad_(True)
        gf = g.float()

        def r_fb():
            for t in (qr, krr, kcr, sr):
                if t is not None and t.grad is not None:
                    t.grad = None
            o = cref(qr, krr, kcr, sr, W, m)
            o.backward(gf)

        res["ref"]["fwdbwd_ms"] = cuda_time(r_fb, iters=10, warmup=3)
        res["ref"]["fwdbwd_mem"] = peak_mem_mb(r_fb)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        res["ref"]["oom"] = True
    except Exception as e:  # compile failures etc.
        res["ref"]["err"] = f"{type(e).__name__}: {str(e)[:60]}"
    torch.cuda.empty_cache()
    return res


def main():
    W = 128
    cases = [
        (1, 2048, 4, True, W),
        (1, 2048, 128, True, W),
        (1, 8192, 4, True, W),
        (1, 8192, 128, True, W),
        (1, 16384, 4, True, W),
        (1, 32768, 128, True, W),
    ]
    rows = []
    for c in cases:
        print("benchmarking", c, flush=True)
        rows.append(bench_shape(*c))

    print("\n=== tilelang kernel ===")
    hdr = f"{'shape':28s} {'fwd ms':>8} {'fwd TF/s':>9} {'bwd ms':>8} {'fb ms':>8} {'bwd/fwd':>7} {'fwd MB':>8} {'fb MB':>8}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['tag']:28s} {r['k_fwd_ms']:8.3f} {r['k_fwd_tflops']:9.1f} "
            f"{r['k_bwd_ms']:8.3f} {r['k_fwdbwd_ms']:8.3f} {r['bwd_fwd_ratio']:7.2f} "
            f"{r['k_fwd_mem']:8.0f} {r['k_fwdbwd_mem']:8.0f}"
        )

    print("\n=== torch.compile(reference) ===")
    hdr2 = f"{'shape':28s} {'fwd ms':>8} {'fwd TF/s':>9} {'fb ms':>8} {'fwd MB':>8} {'fb MB':>8}  status"
    print(hdr2)
    print("-" * len(hdr2))
    for r in rows:
        ref = r["ref"]
        if ref.get("oom"):
            print(f"{r['tag']:28s} {'OOM':>8} {'--':>9} {'OOM':>8} {'--':>8} {'--':>8}  OOM (kernel survives)")
        elif ref.get("err"):
            print(f"{r['tag']:28s}  ERR: {ref['err']}")
        else:
            print(
                f"{r['tag']:28s} {ref['fwd_ms']:8.3f} {ref['fwd_tflops']:9.1f} "
                f"{ref['fwdbwd_ms']:8.3f} {ref['fwd_mem']:8.0f} {ref['fwdbwd_mem']:8.0f}  ok"
            )

    print("\n=== speedup (torch.compile / kernel) ===")
    for r in rows:
        ref = r["ref"]
        if ref.get("oom") or ref.get("err"):
            print(f"{r['tag']:28s} reference unavailable (OOM/err) -> kernel-only win")
        else:
            sf = ref["fwd_ms"] / r["k_fwd_ms"]
            sfb = ref["fwdbwd_ms"] / r["k_fwdbwd_ms"]
            print(f"{r['tag']:28s} fwd x{sf:5.2f}  fwd+bwd x{sfb:5.2f}")


if __name__ == "__main__":
    main()
