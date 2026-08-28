"""Community-SOTA forward comparison vs sglang's DeepSeek-V4 flash attention.

Baseline: `sglang.jit_kernel.flash_attention_v4.flash_attn_varlen_func`, which
wraps the vendored FlashAttention-CUTE (FA4) kernel. It supports EXACTLY our
sink + sliding-window MQA pattern (causal, window_size=(W-1,0), per-head
`sinks`), so on the sliding+sink dense path it is the same math as ours.

KEY HARDWARE FACT (probed on this H20 / SM90 box):
  FA4 supports head_dim only in [8, 256] on SM90 -> it CANNOT run V4-Flash's
  actual head_dim=512. (`(head_dim, head_dim_v)=(512,512) is not supported on
  SM90`.) This is the exact structural blocker that motivates the custom kernel:
  no community flash kernel does hd=512 on Hopper.

So we compare two regimes:
  (1) HEAD-TO-HEAD at head_dim=256 (FA4's max), identical sink+sliding-window+MQA
      math, forward only -> a true apples-to-apples efficiency check: how close is
      our tilelang kernel to a fully production-tuned flash kernel on the same
      shape it CAN run?
  (2) head_dim=512 (what the model needs): FA4 is unavailable (error captured);
      only FlashInfer-MLA (see bench_sota.py) reaches hd512.

Forward only (FA4 is an inference kernel). FLOPs counted over the sliding band
(valid query/key pairs), 4*D per pair (QK^T + PV) -> achieved bf16 TFLOP/s.
"""

import sys
import torch

sys.path.insert(0, ".")
import kernel as K  # noqa: E402
from reference import attention_reference  # noqa: E402
from sglang.jit_kernel.flash_attention_v4 import flash_attn_varlen_func  # noqa: E402

DEV = "cuda"
H = 64


def valid_pairs_sliding(S, W):
    i = torch.arange(S, dtype=torch.float64)
    return torch.clamp(i + 1, max=W).sum().item()


from _bench_common import cuda_time  # noqa: E402


def fa4_fn(q_v, kr_v, sinks_bf, S, W, scale):
    """q_v [S,H,D], kr_v [S,1,D] (K==V); sliding-window causal + per-head sink."""
    cu = torch.tensor([0, S], device=DEV, dtype=torch.int32)

    def run():
        return flash_attn_varlen_func(
            q_v,
            kr_v,
            kr_v,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=S,
            max_seqlen_k=S,
            causal=True,
            window_size=(W - 1, 0),
            sinks=sinks_bf,
            softmax_scale=scale,
        )

    return run


def head_to_head(D, S, W):
    scale = D**-0.5
    P = valid_pairs_sliding(S, W)
    flops = H * 4 * D * P

    # inputs (shared between both kernels; K==V shared-KV MQA)
    q = torch.randn(H, S, D, device=DEV, dtype=torch.bfloat16) * 0.3  # [H,S,D]
    kr = torch.randn(1, S, D, device=DEV, dtype=torch.bfloat16) * 0.3  # [1,S,D]
    sinks = torch.randn(H, device=DEV, dtype=torch.float32)

    # ---- our tilelang kernel: [B=1,H,S,D] ----
    q_k = q.unsqueeze(0)  # [1,H,S,D]
    kr_k = kr.unsqueeze(0)  # [1,1,S,D]

    def our_run():
        return K.v4flash_attention(q_k, kr_k, None, sinks, W, 4)

    our_ms = cuda_time(our_run)
    our_tf = flops / (our_ms * 1e-3) / 1e12

    # ---- sglang FA4: varlen [S,H,D], [S,1,D] ----
    q_v = q.transpose(0, 1).contiguous()  # [S,H,D]
    kr_v = kr.transpose(0, 1).contiguous()  # [S,1,D]
    sinks_bf = sinks.to(torch.bfloat16)
    run = fa4_fn(q_v, kr_v, sinks_bf, S, W, scale)
    fa4_ms = cuda_time(run)
    fa4_tf = flops / (fa4_ms * 1e-3) / 1e12

    # ---- cross-check: both vs fp32 reference (sliding-only, no comp) ----
    # The dense reference materializes [H,S,S]; only feasible at small S.
    if S <= 4096:
        out_ref = attention_reference(q_k.float(), kr_k.float(), None, sinks, W, 4)  # [1,H,S,D]
        our_o = our_run().float()
        fa4_o = run().float().transpose(0, 1).unsqueeze(0)  # [S,H,D]->[1,H,S,D]

        def rel(a, b):
            return ((a - b).abs().max() / (b.abs().max() + 1e-9)).item()

        our_rel = rel(our_o, out_ref)
        fa4_rel = rel(fa4_o, out_ref)
    else:
        our_rel = fa4_rel = float("nan")

    return dict(
        D=D,
        S=S,
        W=W,
        P=P,
        our_ms=our_ms,
        our_tf=our_tf,
        fa4_ms=fa4_ms,
        fa4_tf=fa4_tf,
        ratio=our_tf / fa4_tf,
        our_rel=our_rel,
        fa4_rel=fa4_rel,
    )


def main():
    W = 128
    print("### (1) HEAD-TO-HEAD at head_dim=256 (FA4 max), sink + sliding-window + MQA, fwd only")
    print(
        f"{'D':>4} {'S':>7} {'P/1e6':>8} | {'ours ms':>8} {'ours TF/s':>9} | "
        f"{'FA4 ms':>8} {'FA4 TF/s':>9} | {'ours/FA4':>9} | {'our_rel':>9} {'fa4_rel':>9}"
    )
    print("-" * 100)
    for S in (2048, 4096, 8192, 16384):
        r = head_to_head(256, S, W)
        print(
            f"{r['D']:>4} {r['S']:>7} {r['P']/1e6:>8.2f} | {r['our_ms']:>8.3f} {r['our_tf']:>9.1f} | "
            f"{r['fa4_ms']:>8.3f} {r['fa4_tf']:>9.1f} | {r['ratio']:>8.2f}x | "
            f"{r['our_rel']:>9.2e} {r['fa4_rel']:>9.2e}"
        )
        torch.cuda.empty_cache()

    print("\n### (2) head_dim=512 (the dim V4-Flash actually uses): is FA4 available?")
    D, S = 512, 2048
    q = torch.randn(S, H, D, device=DEV, dtype=torch.bfloat16) * 0.3
    kr = torch.randn(S, 1, D, device=DEV, dtype=torch.bfloat16) * 0.3
    sinks_bf = torch.randn(H, device=DEV, dtype=torch.bfloat16)
    cu = torch.tensor([0, S], device=DEV, dtype=torch.int32)
    try:
        flash_attn_varlen_func(
            q,
            kr,
            kr,
            cu_seqlens_q=cu,
            cu_seqlens_k=cu,
            max_seqlen_q=S,
            max_seqlen_k=S,
            causal=True,
            window_size=(W - 1, 0),
            sinks=sinks_bf,
            softmax_scale=D**-0.5,
        )
        print("  FA4 ran at hd=512 (unexpected)")
    except Exception as e:
        print(f"  FA4 UNAVAILABLE at hd=512: {type(e).__name__}: {str(e)[:140]}")


if __name__ == "__main__":
    main()
