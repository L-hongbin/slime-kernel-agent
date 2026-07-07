"""Efficiency benchmark for the V4 FP8 MoE expert compute path.

Compares V4GroupedExperts.forward_dispatched in two modes at realistic EP8 shapes:
  * fp8 grouped (V4_FP8_EXPERT_GEMM=1): deep_gemm m_grouped_fp8_gemm_nt_contiguous
    (the kernel sglang serves these experts with) + bf16 grad_x, no bf16 weight
    retained.
  * bf16 loop  (baseline): per-expert F.linear on the dequantized bf16 weight.

Reports forward ms, backward ms, and PEAK activation memory (the fp8 path's
motivation: it must not retain the bf16 weights that OOM'd ctx-16k). Run on one
H20; re-run and record in FP8_MOE_RESULTS.md when the kernel path changes.

  python custom_kernels/deepseek_v4/megatron/bench_fp8_moe.py
"""

import os
import time

import torch

from custom_kernels.deepseek_v4.megatron.mcore_model import V4GroupedExperts


class _Cfg:
    # DeepSeek-V4-Flash MoE dims; EP8 -> 32 local experts / rank.
    num_local_experts = 32
    hidden_size = 4096
    intermediate_size = 2048
    hidden_act = "silu"
    swiglu_limit = 7.0


def _mk_experts(dev):
    e = V4GroupedExperts(_Cfg())
    e = e.to(dev)
    e.gate_up_proj.data = e.gate_up_proj.data.bfloat16()
    e.down_proj.data = e.down_proj.data.bfloat16()
    with torch.no_grad():
        e.gate_up_proj.normal_(0, 0.02)
        e.down_proj.normal_(0, 0.02)
    e.gate_up_proj.requires_grad_(False)
    e.down_proj.requires_grad_(False)
    e.quantize_frozen_experts_fp8()
    return e


def _counts(total_tokens, n_experts, dev):
    # near-uniform routed load with mild variance (not 128-aligned, realistic)
    base = total_tokens // n_experts
    c = torch.full((n_experts,), base, dtype=torch.long)
    c[: total_tokens - base * n_experts] += 1
    # perturb a bit so segments are uneven
    for i in range(0, n_experts, 4):
        c[i] = max(0, c[i] - 37)
        if i + 1 < n_experts:
            c[i + 1] = c[i + 1] + 37
    return c.to(dev)


def _time(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / iters
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    return ms, peak_gb


def _run(e, x, tpe, gemm):
    os.environ["V4_FP8_EXPERT_GEMM"] = "1" if gemm else "0"

    def fwd():
        with torch.no_grad():
            return e.forward_dispatched(x, tpe)

    def fwd_bwd():
        xi = x.detach().requires_grad_(True)
        out = e.forward_dispatched(xi, tpe)
        out.sum().backward()
        return xi.grad

    f_ms, f_gb = _time(fwd)
    fb_ms, fb_gb = _time(fwd_bwd)
    return f_ms, fb_ms, fb_gb


def main():
    dev = "cuda"
    torch.manual_seed(0)
    e = _mk_experts(dev)
    print(f"{'total_tok':>10} {'mode':>10} {'fwd_ms':>9} {'fwd+bwd_ms':>11} {'peak_GB':>9}")
    for total in (8192, 16384, 32768):
        tpe = _counts(total, _Cfg.num_local_experts, dev)
        x = torch.randn(total, _Cfg.hidden_size, dtype=torch.bfloat16, device=dev)
        for gemm, name in ((True, "fp8_gemm"), (False, "bf16_loop")):
            f, fb, gb = _run(e, x, tpe, gemm)
            print(f"{total:>10} {name:>10} {f:>9.2f} {fb:>11.2f} {gb:>9.2f}")


if __name__ == "__main__":
    main()
