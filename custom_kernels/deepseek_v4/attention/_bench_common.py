"""Shared CUDA-event timing helper for the A1 attention benches.

Single source for the measurement method so all benches (bench.py, bench_bwd,
bench_fwd_quick, bench_sota, bench_flashmla, bench_sglang) time identically.
"""

import torch


def cuda_time(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = torch.cuda.Event(True)
    en = torch.cuda.Event(True)
    st.record()
    for _ in range(iters):
        fn()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / iters  # ms
