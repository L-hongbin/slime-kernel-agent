"""Community-SOTA forward comparison: our tilelang compression pool vs sglang's
production DeepSeek-V4 compressor kernel (`sglang.jit_kernel.dsv4.compress_forward`).

FORWARD ONLY — sglang's `compress_forward` is an inference kernel (no backward).

What each side computes (head_dim=512, bf16, B=1):
  ours   : windowed gated-softmax pool + RMSNorm, dense training layout.
           CSA reads raw [B,S,2D] and gathers Ca/Cb + position_bias in-kernel;
           HCA reads windowed [B,n_win,128,D].
  sglang : windowed gated-softmax pool ONLY (NO RMSNorm — that lives in the
           separate `compress_norm_rope_store` kernel), paged/ragged inference
           layout. CSA input row layout is | kv_overlap | kv | score_overlap |
           score | (4*head_dim wide); HCA is | kv | score | (2*head_dim wide).

So this is a HARDWARE-EFFICIENCY reference, not identical work:
  - ours does MORE (fused RMSNorm over head_dim),
  - sglang reads MORE bytes for CSA (materialized overlap rows: 4*D vs our 2*D),
  - sglang runs a ragged/paged gather via a precomputed plan; ours gathers from a
    dense [B,S,2D] tensor. We time the compute kernel only (plan pre-generated).

Run:  CUDA_VISIBLE_DEVICES=2 python bench_sota.py
"""

import sys

import torch

# our kernels (run from this dir)
from kernel import csa_compress, hca_compress

# sglang production compressor
sys.path.insert(0, "/sgl-workspace/sglang/python/sglang/jit_kernel/tests/deepseek_v4")
from common import make_legacy_context, make_state_pool, to_seq_extend
from sglang.jit_kernel.dsv4 import compress_forward

DEV = "cuda"
D = 512
EPS = 1e-6
DT = torch.bfloat16
SEQS = (8192, 32768)


def _bench(fn, iters=50, warmup=10):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(warmup):
        with torch.no_grad():
            fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        with torch.no_grad():
            fn()
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    peak = torch.cuda.max_memory_allocated() / 1e6
    return ms, peak


def _ours_inputs(kind, seq):
    rate = 128 if kind == "HCA" else 4
    proj = D if kind == "HCA" else 2 * D
    kv = (torch.randn(1, seq, proj, device=DEV, dtype=DT) * 0.5).detach()
    gate = (torch.randn(1, seq, proj, device=DEV, dtype=DT) * 0.5).detach()
    pb = (torch.randn(rate, proj, device=DEV, dtype=DT) * 0.3).detach()
    w = (1.0 + 0.1 * torch.randn(D, device=DEV, dtype=DT)).detach()
    fn = hca_compress if kind == "HCA" else csa_compress
    return lambda: fn(kv, gate, pb, w, EPS, rate)


def _sglang_inputs(kind, seq):
    rate = 128 if kind == "HCA" else 4
    mult = 2 if kind == "HCA" else 4  # input row width = mult * head_dim
    ape_win = rate if kind == "HCA" else 8  # CSA window = 2*rate = 8
    ctx = make_legacy_context(bs=1, compress_ratio=rate, head_dim=D)
    seq_lens_cpu, extend_lens_cpu, num_q = to_seq_extend([(seq, seq)])
    kv_in = torch.randn(num_q, D * mult, device=DEV, dtype=DT)
    ape = torch.randn(ape_win, D, device=DEV, dtype=DT)
    pool = make_state_pool(ctx.num_pages, rate, D).to(DT)
    plan = ctx.make_prefill_plan(seq_lens_cpu, extend_lens_cpu, num_q)  # setup, timed-out
    return lambda: compress_forward(pool, kv_in, ape, plan, head_dim=D, compress_ratio=rate)


def _flops(kind, seq):
    """Approx FLOPs for the windowed gated-softmax pool (both sides share this core).
    Per compress event, per channel d (D), over K window slots:
      softmax (max+exp+sum+div) ~ 4K, weighted sum (mul+add) ~ 2K  => ~6K.
    RMSNorm (ours only) adds ~3 per channel, negligible vs 6K. Used only as a
    rough throughput yardstick — the two kernels do not do identical work."""
    rate = 128 if kind == "HCA" else 4
    K = rate if kind == "HCA" else 8
    events = seq // rate
    return events * D * 6 * K


def main():
    print(f"device={torch.cuda.get_device_name(0)}  dtype={DT}  head_dim={D}")
    print("FORWARD ONLY. ours = pool+RMSNorm; sglang = pool only (norm/rope external).\n")
    header = (
        f"{'case':<22}{'ours ms':>10}{'sgl ms':>10}{'ours/sgl':>10}"
        f"{'ours TF/s':>11}{'sgl TF/s':>11}{'ours MB':>10}{'sgl MB':>10}"
    )
    for kind in ("CSA", "HCA"):
        rate = 128 if kind == "HCA" else 4
        print(f"# {kind}  (rate={rate})")
        print(header)
        for seq in SEQS:
            n_win = seq // rate
            ours = _ours_inputs(kind, seq)
            sgl = _sglang_inputs(kind, seq)
            o_ms, o_mem = _bench(ours)
            s_ms, s_mem = _bench(sgl)
            fl = _flops(kind, seq)
            o_tf = fl / (o_ms * 1e-3) / 1e12
            s_tf = fl / (s_ms * 1e-3) / 1e12
            label = f"{kind} {seq} (nw={n_win})"
            print(
                f"{label:<22}{o_ms:>10.4f}{s_ms:>10.4f}{o_ms/s_ms:>9.2f}x"
                f"{o_tf:>11.3f}{s_tf:>11.3f}{o_mem:>10.1f}{s_mem:>10.1f}"
            )
        print()


if __name__ == "__main__":
    main()
