"""Exact torch reference for the DeepSeek-V4-Flash compression pool (B2 kernel).

Source of truth: ``transformers/models/deepseek_v4/modeling_deepseek_v4.py``
- ``DeepseekV4HCACompressor.forward``  (lines 394-426)
- ``DeepseekV4CSACompressor.forward``  (lines 613-668)
- ``DeepseekV4RMSNorm.forward``        (lines 55-60)

The recurring core of both compressors (training / stateless, ``cache=None``) is a
**per-channel windowed gated-softmax pool followed by RMSNorm over head_dim**:

    compressed = kv_norm( (kv_window * gate_window.softmax(dim=window, fp32)).sum(window) )

where ``gate_window`` already includes ``+ position_bias`` and the softmax is over the
*window* axis, independently for each of the ``head_dim`` channels.

RoPE and the linear projections (kv_proj / gate_proj) are applied OUTSIDE this module
(reused from the base model), so they are not part of the reference here. The CSA
indexer / top-k is dropped at 32k context and is not implemented.

Everything here is computed in fp32 and is the numerical ground truth that the tilelang
kernel (bf16 compute / fp32 softmax+accum) is checked against.
"""

from __future__ import annotations

import torch


# --------------------------------------------------------------------------------------
# RMSNorm (DeepseekV4RMSNorm, weighted, over the last dim) -- fp32 truth.
# --------------------------------------------------------------------------------------
def rms_norm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    in_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x.to(in_dtype)) if in_dtype != torch.float32 else (weight.to(torch.float32) * x)


# --------------------------------------------------------------------------------------
# Core pool: kv, gate are [..., K, D]; softmax over K per channel, weighted sum, RMSNorm.
# This is the part the tilelang kernel replaces. `gate` already includes position_bias
# (and any -inf masking for empty / out-of-window slots).
# --------------------------------------------------------------------------------------
def gated_pool_ref(kv: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """kv, gate: [..., K, D] (fp32). Returns [..., D] = RMSNorm(sum_K softmax_K(gate) * kv)."""
    w = gate.softmax(dim=-2, dtype=torch.float32).to(kv.dtype)
    pooled = (kv * w).sum(dim=-2)
    return rms_norm_ref(pooled, weight, eps)


def gated_pool_pooled_ref(kv: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Just the pooled (pre-RMSNorm) tensor, for diagnostics."""
    w = gate.softmax(dim=-2, dtype=torch.float32).to(kv.dtype)
    return (kv * w).sum(dim=-2)


# --------------------------------------------------------------------------------------
# HCA compressor pool (stateless). Non-overlapping windows of size W = compress_rate.
#   kv_seq, gate_seq: [B, S, D]; position_bias: [W, D]; weight: [D]
# Returns compressed: [B, n_win, D]  (pre-RoPE; RoPE applied outside).
# --------------------------------------------------------------------------------------
def hca_compress_ref(
    kv_seq: torch.Tensor,
    gate_seq: torch.Tensor,
    position_bias: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    compress_rate: int = 128,
) -> torch.Tensor:
    B, S, D = kv_seq.shape
    usable = (S // compress_rate) * compress_rate
    n_win = usable // compress_rate
    kv = kv_seq[:, :usable].view(B, n_win, compress_rate, D)
    gate = gate_seq[:, :usable].view(B, n_win, compress_rate, D) + position_bias.to(gate_seq.dtype)
    return gated_pool_ref(kv, gate, weight, eps)


# --------------------------------------------------------------------------------------
# CSA compressor pool (stateless). Overlapping Ca/Cb layout, window = 2*compress_rate.
#   kv_seq, gate_seq: [B, S, 2*D]; position_bias: [ratio, 2*D]; weight: [D]
# Builds new_kv/new_gate [B, n_win, 2*ratio, D] exactly as modeling:644-657, then pools.
# Returns compressed: [B, n_win, D] (pre-RoPE).
# --------------------------------------------------------------------------------------
def csa_build_overlap(
    kv_seq: torch.Tensor,
    gate_seq: torch.Tensor,
    position_bias: torch.Tensor,
    compress_rate: int = 4,
):
    B, S, D2 = kv_seq.shape
    D = D2 // 2
    ratio = compress_rate
    usable = (S // ratio) * ratio
    n_win = usable // ratio
    chunk_kv = kv_seq[:, :usable].view(B, n_win, ratio, D2)
    chunk_gate = gate_seq[:, :usable].view(B, n_win, ratio, D2) + position_bias.to(gate_seq.dtype)

    new_kv = chunk_kv.new_zeros((B, n_win, 2 * ratio, D))
    new_gate = chunk_gate.new_full((B, n_win, 2 * ratio, D), float("-inf"))
    # second half (slots ratio..2ratio-1) = current window's Cb = [..., D:]
    new_kv[:, :, ratio:] = chunk_kv[..., D:]
    new_gate[:, :, ratio:] = chunk_gate[..., D:]
    # first half (slots 0..ratio-1) of window w = previous window's Ca = [..., :D]
    if n_win > 1:
        new_kv[:, 1:, :ratio] = chunk_kv[:, :-1, :, :D]
        new_gate[:, 1:, :ratio] = chunk_gate[:, :-1, :, :D]
    # window 0's first half stays zero-kv / -inf-gate (softmax weight 0).
    return new_kv, new_gate


def csa_compress_ref(
    kv_seq: torch.Tensor,
    gate_seq: torch.Tensor,
    position_bias: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    compress_rate: int = 4,
) -> torch.Tensor:
    new_kv, new_gate = csa_build_overlap(kv_seq, gate_seq, position_bias, compress_rate)
    return gated_pool_ref(new_kv, new_gate, weight, eps)
