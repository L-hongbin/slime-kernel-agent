"""Backward-efficiency vs forward-efficiency benchmark for the A1 kernel.

Run:  CUDA_VISIBLE_DEVICES=0 python bench_bwd.py

Separately times the forward, the dQ kernel, the dKV kernel(s), and the full
backward, then reports for each: latency (ms), achieved TFLOP/s, and % of the
H20 bf16 tensor-core peak (~148 TF/s). Reports the bwd/fwd LATENCY ratio AND the
bwd/fwd EFFICIENCY (%peak) ratio so we can tell whether the backward is as
hardware-efficient as the forward, and whether the latency ratio is justified by
the FLOP count or inflated by atomics / loose loop bounds.

FLOP model (per structurally-valid (query,key) pair, per head, per batch):
  forward  : QKᵀ (2D) + PV (2D)                      = 4·D
  dQ kernel: QKᵀ (2D) + dO·K (2D) + dS@K (2D)         = 6·D
  dKV krnl : KQ  (2D) + K·dO (2D) + dS@Q (2D) + p@dO (2D) = 8·D
  backward total                                      = 14·D   (= 3.5× fwd)
The ideal flash backward is 5 GEMM / 2.5× fwd; the extra 1.0× here is the QKᵀ
and dO·K recompute duplicated across the dQ and dKV kernels (MQA shared-KV makes
a single fused dQ+dKV kernel need atomics on one side).
"""

import sys
import torch

sys.path.insert(0, ".")
import kernel as K  # noqa: E402

DEV = "cuda"
H, D = 64, 512
PEAK_TFLOPS = 148.0  # H20 bf16 tensor-core peak


def valid_pairs(S, Tcomp, W, m):
    i = torch.arange(S, dtype=torch.float64)
    raw = torch.clamp(i + 1, max=W).sum().item()
    comp = 0.0
    if Tcomp > 0:
        thr = torch.clamp((i + 1) // m, max=Tcomp)
        comp = thr.sum().item()
    return raw + comp


from _bench_common import cuda_time  # noqa: E402


def _prep(B, S, m, with_comp, W, block_M=64, block_N=64):
    """Build padded inputs + compiled kernels, mirroring the autograd wrapper."""
    Tcomp = (S // m) if with_comp else 0
    q = torch.randn(B, H, S, D, device=DEV, dtype=torch.bfloat16) * 0.3
    kr = torch.randn(B, 1, S, D, device=DEV, dtype=torch.bfloat16) * 0.3
    kc = (torch.randn(B, 1, Tcomp, D, device=DEV, dtype=torch.bfloat16) * 0.3) if with_comp else None
    sinks = torch.randn(H, device=DEV, dtype=torch.float32)

    mb = max(block_M, block_N)
    Sp = K._ceil(S, mb) * mb
    Tcp = K._ceil(Tcomp, block_N) * block_N if Tcomp > 0 else 0
    KV = Sp + Tcp

    q_pad = q.new_zeros(B, H, Sp, D)
    q_pad[:, :, :S] = q
    kvt = q.new_zeros(B, 1, KV, D)
    kvt[:, :, :S] = kr
    if Tcomp > 0:
        kvt[:, :, Sp : Sp + Tcomp] = kc
    sinks_f = sinks.float().contiguous()

    args = (B, H, S, Tcomp, Sp, Tcp, W, m, D, block_M, block_N)
    key = args + (str(torch.bfloat16),)
    fwd = K._get(K._FWD_CACHE, K._build_fwd, key, *args)
    dq_fn = K._get(K._DQ_CACHE, K._build_bwd_dq, args, *args)
    dkv_raw_fn = K._get(K._DKV_RAW_CACHE, K._build_bwd_dkv_raw, args, *args)
    dkv_comp_fn = K._get(K._DKV_COMP_CACHE, K._build_bwd_dkv_comp, args, *args) if Tcp > 0 else None

    out_pad, lse_pad = fwd(q_pad, kvt, sinks_f)
    dO = q_pad.new_zeros(B, H, Sp, D)
    dO[:, :, :S] = torch.randn(B, H, S, D, device=DEV, dtype=torch.bfloat16)
    delta = (dO.float() * out_pad.float()).sum(-1).contiguous()

    return dict(
        B=B,
        S=S,
        m=m,
        Tcomp=Tcomp,
        W=W,
        Sp=Sp,
        Tcp=Tcp,
        KV=KV,
        q_pad=q_pad,
        kvt=kvt,
        sinks_f=sinks_f,
        out_pad=out_pad,
        lse_pad=lse_pad,
        dO=dO,
        delta=delta,
        fwd=fwd,
        dq_fn=dq_fn,
        dkv_raw_fn=dkv_raw_fn,
        dkv_comp_fn=dkv_comp_fn,
    )


def bench_shape(B, S, m, with_comp, W):
    P = valid_pairs(S, Tcomp=(S // m if with_comp else 0), W=W, m=m)
    npairs = B * H * P  # total valid (q,k) pairs across heads & batch
    fwd_flops = 4 * D * npairs
    dq_flops = 6 * D * npairs
    dkv_flops = 8 * D * npairs
    bwd_flops = 14 * D * npairs

    g = _prep(B, S, m, with_comp, W)
    dKV = torch.zeros(B, 1, g["KV"], D, device=DEV, dtype=torch.float32)

    def f_fwd():
        g["fwd"](g["q_pad"], g["kvt"], g["sinks_f"])

    def f_dq():
        g["dq_fn"](g["q_pad"], g["kvt"], g["dO"], g["lse_pad"], g["delta"])

    def f_dkv():
        dKV.zero_()
        g["dkv_raw_fn"](g["q_pad"], g["kvt"], g["dO"], g["lse_pad"], g["delta"], dKV)
        if g["dkv_comp_fn"] is not None:
            g["dkv_comp_fn"](g["q_pad"], g["kvt"], g["dO"], g["lse_pad"], g["delta"], dKV)

    def f_bwd():
        g["dq_fn"](g["q_pad"], g["kvt"], g["dO"], g["lse_pad"], g["delta"])
        dKV.zero_()
        g["dkv_raw_fn"](g["q_pad"], g["kvt"], g["dO"], g["lse_pad"], g["delta"], dKV)
        if g["dkv_comp_fn"] is not None:
            g["dkv_comp_fn"](g["q_pad"], g["kvt"], g["dO"], g["lse_pad"], g["delta"], dKV)

    fwd_ms = cuda_time(f_fwd)
    dq_ms = cuda_time(f_dq)
    dkv_ms = cuda_time(f_dkv)
    bwd_ms = cuda_time(f_bwd)

    def tf(flops, ms):
        return flops / (ms * 1e-3) / 1e12

    r = dict(
        tag=f"B{B} S{S:>5} m{m:>3} comp{int(with_comp)} W{W}",
        fwd_ms=fwd_ms,
        dq_ms=dq_ms,
        dkv_ms=dkv_ms,
        bwd_ms=bwd_ms,
        fwd_tf=tf(fwd_flops, fwd_ms),
        dq_tf=tf(dq_flops, dq_ms),
        dkv_tf=tf(dkv_flops, dkv_ms),
        bwd_tf=tf(bwd_flops, bwd_ms),
    )
    r["fwd_pk"] = 100 * r["fwd_tf"] / PEAK_TFLOPS
    r["dq_pk"] = 100 * r["dq_tf"] / PEAK_TFLOPS
    r["dkv_pk"] = 100 * r["dkv_tf"] / PEAK_TFLOPS
    r["bwd_pk"] = 100 * r["bwd_tf"] / PEAK_TFLOPS
    r["lat_ratio"] = bwd_ms / fwd_ms
    # backward %peak / forward %peak; >1 = backward is MORE hardware-efficient.
    r["eff_ratio"] = r["bwd_pk"] / r["fwd_pk"] if r["fwd_pk"] else float("nan")
    del g, dKV
    torch.cuda.empty_cache()
    return r


def main():
    W = 128
    cases = [
        (1, 2048, 4, True, W),
        (1, 2048, 128, True, W),
        (1, 8192, 4, True, W),
        (1, 8192, 128, True, W),
    ]
    rows = [bench_shape(*c) for c in cases]

    print(f"\n=== A1 backward vs forward efficiency (H20, bf16 peak {PEAK_TFLOPS} TF/s) ===")
    hdr = (
        f"{'shape':26s} | {'fwd ms':>7} {'TF/s':>6} {'%pk':>5} | "
        f"{'dQ ms':>7} {'TF/s':>6} {'%pk':>5} | {'dKV ms':>7} {'TF/s':>6} {'%pk':>5} | "
        f"{'bwd ms':>7} {'TF/s':>6} {'%pk':>5} | {'b/f lat':>7} {'b/f eff':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['tag']:26s} | {r['fwd_ms']:7.3f} {r['fwd_tf']:6.1f} {r['fwd_pk']:5.1f} | "
            f"{r['dq_ms']:7.3f} {r['dq_tf']:6.1f} {r['dq_pk']:5.1f} | "
            f"{r['dkv_ms']:7.3f} {r['dkv_tf']:6.1f} {r['dkv_pk']:5.1f} | "
            f"{r['bwd_ms']:7.3f} {r['bwd_tf']:6.1f} {r['bwd_pk']:5.1f} | "
            f"{r['lat_ratio']:7.2f} {r['eff_ratio']:7.2f}"
        )
    print("\nb/f lat = backward/forward latency ratio (FLOP-ideal 3.5×; flash-style ~2.5×).")
    print("b/f eff = backward %peak / forward %peak (>1 = backward MORE HW-efficient than forward).")


if __name__ == "__main__":
    main()
