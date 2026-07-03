"""Exact torch reference for the DeepSeek-V4-Flash core attention (A1).

This is a faithful port of `eager_attention_forward` +
`DeepseekV4Attention.forward` from
`transformers/models/deepseek_v4/modeling_deepseek_v4.py`, restricted to the
TRAINING use case the tilelang kernel targets:

  * no KV cache, a single forward over a packed sequence of length S;
  * RoPE is already applied to q / k OUTSIDE this function (we take post-RoPE
    tensors), and the conjugate `-sin` rotation on the output is likewise
    outside;
  * shared-KV MQA: K and V are the *same* tensor, broadcast to all H heads
    (`repeat_kv` with `n_rep = H`);
  * the indexer top-k is DROPPED (dense-over-compressed), so the compressed
    region's mask is purely the causal-threshold additive bias the HCA
    compressor builds (`entry_idx >= (pos+1)//m -> -inf`).

The KV axis is `[ raw_tokens (length S) ++ compressed_entries (length Tcomp) ]`:

  * raw region (length S): sliding-window causal -- query i attends raw key j
    iff `j <= i` and `i - j <= W-1` (W = sliding_window).
  * compressed region (length Tcomp): query i attends compressed entry w iff
    `w < (i+1)//m` (m = compress_rate; CSA m=4, HCA m=128).

Per-head attention sink (gpt-oss style): a constant extra logit column equal to
`sinks[h]` is appended before the softmax (row-max subtracted in fp32 for
stability) and DROPPED before multiplying by V -- so it only enters the
softmax denominator.

The reference computes everything in fp32 and is the source of truth for the
correctness tests.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_additive_mask(
    s_raw: int,
    t_comp: int,
    window: int,
    m: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dense additive attention mask `[S, S + Tcomp]` (0 valid, -inf masked).

    Matches the structural masks the model builds: sliding-window causal over
    the raw region and causal-threshold over the compressed region.
    """
    q = torch.arange(s_raw, device=device)
    # --- raw region: sliding-window causal ---
    kr = torch.arange(s_raw, device=device)
    raw_valid = (kr[None, :] <= q[:, None]) & (q[:, None] - kr[None, :] <= window - 1)
    # --- compressed region: entry w valid iff w < (i+1)//m ---
    if t_comp > 0:
        wc = torch.arange(t_comp, device=device)
        thr = (q + 1) // m  # [S]
        comp_valid = wc[None, :] < thr[:, None]
        valid = torch.cat([raw_valid, comp_valid], dim=1)
    else:
        valid = raw_valid
    mask = torch.zeros(s_raw, s_raw + t_comp, device=device, dtype=dtype)
    mask.masked_fill_(~valid, float("-inf"))
    return mask


def attention_reference(
    q: torch.Tensor,
    k_raw: torch.Tensor,
    k_comp: torch.Tensor | None,
    sinks: torch.Tensor,
    window: int,
    m: int,
    return_lse: bool = False,
):
    """Reference forward.

    Args:
        q:      [B, H, S, D]   post-RoPE queries.
        k_raw:  [B, 1, S, D]   post-RoPE raw KV (K == V).
        k_comp: [B, 1, Tcomp, D] or None   post-RoPE compressed KV (K == V).
        sinks:  [H]            per-head sink logit.
        window: sliding window W.
        m:      compress rate.

    Returns:
        out: [B, H, S, D]  attention output (before the conjugate-RoPE / o_proj,
             which live outside the kernel).
        (optionally) lse: [B, H, S] log-sum-exp incl. the sink column.
    """
    B, H, S, D = q.shape
    scaling = D**-0.5
    # fp32 for bf16/fp16 inputs; keep fp64 so float64 gradcheck sees a fp64 fn.
    compute_dtype = torch.promote_types(q.dtype, torch.float32)

    qf = q.to(compute_dtype)
    if k_comp is not None and k_comp.shape[2] > 0:
        kv = torch.cat([k_raw, k_comp], dim=2)
    else:
        kv = k_raw
    kvf = kv.to(compute_dtype)
    t_comp = 0 if k_comp is None else k_comp.shape[2]

    # repeat_kv: broadcast the single KV head to all H heads.
    kf = kvf.expand(B, H, kvf.shape[2], D)

    attn = torch.matmul(qf, kf.transpose(2, 3)) * scaling  # [B,H,S,KV]
    mask = build_additive_mask(S, t_comp, window, m, q.device, compute_dtype)
    attn = attn + mask[None, None]

    sink = sinks.to(compute_dtype).reshape(1, -1, 1, 1).expand(B, H, S, 1)
    combined = torch.cat([attn, sink], dim=-1)  # [B,H,S,KV+1]
    row_max = combined.max(dim=-1, keepdim=True).values
    combined = combined - row_max
    probs = F.softmax(combined, dim=-1, dtype=compute_dtype)
    scores = probs[..., :-1]  # drop the sink column
    out = torch.matmul(scores, kf)  # [B,H,S,D]

    if return_lse:
        # lse over the combined logits (incl. sink), in the unshifted frame.
        lse = row_max.squeeze(-1) + torch.log(torch.exp(combined).sum(dim=-1))  # [B,H,S]
        return out, lse
    return out
