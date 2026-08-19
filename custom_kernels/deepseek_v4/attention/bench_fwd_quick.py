"""Fast forward-only micro-bench reporting %peak, for warp-spec iteration.
Run: CUDA_VISIBLE_DEVICES=0 python bench_fwd_quick.py
"""

import sys
import torch

sys.path.insert(0, ".")
import kernel as K  # noqa

DEV = "cuda"
H, D = 64, 512
PEAK = 148.0  # H20 bf16 TF/s


def valid_pairs(S, Tcomp, W, m):
    i = torch.arange(S, dtype=torch.float64)
    raw = torch.clamp(i + 1, max=W).sum().item()
    comp = 0.0
    if Tcomp > 0:
        thr = torch.clamp((i + 1) // m, max=Tcomp)
        comp = thr.sum().item()
    return raw + comp


from _bench_common import cuda_time  # noqa: E402


def run(B, S, m, with_comp, W, bM=64, bN=64):
    Tcomp = (S // m) if with_comp else 0
    P = valid_pairs(S, Tcomp, W, m)
    flops = B * H * 4 * D * P
    q = torch.randn(B, H, S, D, device=DEV, dtype=torch.bfloat16) * 0.3
    kr = torch.randn(B, 1, S, D, device=DEV, dtype=torch.bfloat16) * 0.3
    kc = (torch.randn(B, 1, Tcomp, D, device=DEV, dtype=torch.bfloat16) * 0.3) if with_comp else None
    sinks = torch.randn(H, device=DEV, dtype=torch.float32)
    fn = lambda: K.v4flash_attention(q, kr, kc, sinks, W, m, bM, bN)
    fn()
    ms = cuda_time(fn)
    tf = flops / (ms * 1e-3) / 1e12
    tag = f"B{B} S{S} m{m} comp{int(with_comp)} W{W} bM{bM} bN{bN}"
    print(f"{tag:40s} {ms:8.3f} ms  {tf:6.1f} TF/s  {100*tf/PEAK:5.1f}%peak", flush=True)
    torch.cuda.empty_cache()
    return ms, tf


if __name__ == "__main__":
    cases = [
        (1, 2048, 4, False, 2048),  # dense-causal (SOTA bar)
        (1, 2048, 4, True, 128),  # production CSA m4
        (1, 2048, 128, True, 128),  # production HCA m128
        (1, 8192, 4, True, 128),  # large CSA m4
        (1, 8192, 4, False, 8192),  # large dense-causal
    ]
    print(f"GPU: {torch.cuda.get_device_name(0)}  (peak {PEAK} TF/s)")
    for c in cases:
        run(*c)
