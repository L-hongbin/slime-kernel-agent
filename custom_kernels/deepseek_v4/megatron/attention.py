"""V4-Flash attention module (M0 scaffold) -- owns the A1 kernel call + its seams.

This is the custom core_attention-equivalent the plan calls for: it does NOT go
through mcore ``MultiLatentAttention`` / ``DotProductAttention`` (both assert
``attention_bias is None``; V4 needs the per-head sink + structural compressed
mask).  It mirrors ``DeepseekV4Attention.forward`` (HF:789) exactly:

    q_a_proj -> q_a_norm -> q_b_proj -> [B,H,S,D] -> q_b_norm -> partial RoPE
    kv_proj  -> kv_norm  -> [B,1,S,D] -> partial RoPE
    (CSA/HCA) compressor -> compressed_kv  -> concat onto A1's k_comp
    A1 v4flash_attention(q, k_raw, k_comp, sinks, window, m)   # sink + masks inside
    conjugate -sin RoPE on the output rope slice (HF:856)
    grouped o_a_proj (block-diagonal bmm) -> o_b_proj

The sliding-window causal mask (raw region) and the causal-threshold mask
(compressed region) are applied STRUCTURALLY inside A1; no torch attention_mask is
built.  The per-head ``sinks`` are passed to A1 (gpt-oss style, denom-only).
"""

import os

import torch
from torch import nn
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4GroupedLinear

from . import _kernels
from .compressor import COMPRESSOR_CLASSES
from .rope import V4RMSNorm, V4UnweightedRMSNorm, apply_rotary_pos_emb

# Set V4_ATTENTION_TORCH=1 to use the exact torch reference path when TileLang/TVM
# codegen is not stable on a target node. The reference computes in fp32, so cast
# back to q dtype to preserve the module contract seen by the output projections.
if os.environ.get("V4_ATTENTION_TORCH", "0") == "1":
    from custom_kernels.deepseek_v4.attention.reference import attention_reference as _attention_reference

    def _v4flash_attention(q, k_raw, k_comp, sinks, window, m):
        return _attention_reference(q, k_raw, k_comp, sinks, window, m).to(q.dtype)

else:
    _v4flash_attention = _kernels.v4flash_attention


# V4_SPARSE_ATTENTION=1: run the CSA lightning indexer + attend only to its top-k
# compressed KV entries (the model's NATIVE sparse attention, dropped in the M0
# dense-over-compressed path). Matches the sparse rollout. Currently routed through
# the torch reference (exact); the A1 tilelang top-k mask is the perf follow-up.
_SPARSE_ATTENTION = os.environ.get("V4_SPARSE_ATTENTION", "0") == "1"
if _SPARSE_ATTENTION:
    from custom_kernels.deepseek_v4.attention.reference import attention_reference as _sparse_attention_reference


class V4Attention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.rope_layer_type = "main" if self.layer_type == "sliding_attention" else "compress"
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.sliding_window = config.sliding_window
        self.scaling = self.head_dim**-0.5
        # compress_rate fed to A1's structural compressed mask (0 / unused for sliding).
        self.compress_rate = (
            config.compress_rates.get(self.layer_type, 1) if self.layer_type != "sliding_attention" else 1
        )

        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_a_norm = V4RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.head_dim, bias=False)
        self.q_b_norm = V4UnweightedRMSNorm(eps=config.rms_norm_eps)
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = V4RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.o_a_proj = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            config.o_groups,
        )
        self.o_b_proj = nn.Linear(config.o_groups * config.o_lora_rank, config.hidden_size, bias=False)
        self.sinks = nn.Parameter(torch.zeros(self.num_heads))
        self.compressor = (
            COMPRESSOR_CLASSES[self.layer_type](config) if self.layer_type != "sliding_attention" else None
        )

    def forward(self, hidden_states: torch.Tensor, position_embeddings: dict) -> torch.Tensor:
        # hidden_states: [B, S, hidden]  (HC-collapsed + input_layernorm'd by the layer)
        input_shape = hidden_states.shape[:-1]  # (B, S)
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings[self.rope_layer_type]

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)  # [B,H,S,D]
        q = self.q_b_norm(q)
        q = apply_rotary_pos_emb(q, cos, sin)

        kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)  # [B,1,S,D]
        kv = apply_rotary_pos_emb(kv, cos, sin)

        k_comp = None
        if self.compressor is not None:
            k_comp = self.compressor(hidden_states)  # [B,1,T,head_dim] post-RoPE, or T==0
            if k_comp.shape[2] == 0:
                k_comp = None

        # A1: shared-KV MQA (k_raw==v), per-head sink, structural sliding-window +
        # causal-threshold compressed masks.  Returns [B,H,S,D] pre-o_proj.
        comp_topk_mask = None
        if _SPARSE_ATTENTION and k_comp is not None and hasattr(self.compressor, "indexer"):
            # CSA lightning indexer -> top-k compressed selection [B,S,k], then a
            # [B,S,Tcomp] bool mask (scatter; -1 sentinel -> discarded slot 0).
            B, S = input_shape
            pos = torch.arange(S, device=hidden_states.device).unsqueeze(0).expand(B, -1)
            tki = self.compressor.indexer(hidden_states, q_residual, pos, None, self.layer_idx)  # [B,S,k]
            t_comp = k_comp.shape[2]
            scat = torch.zeros(B, S, t_comp + 1, dtype=torch.bool, device=hidden_states.device)
            scat.scatter_(2, (tki.clamp(min=-1) + 1).long(), True)
            comp_topk_mask = scat[:, :, 1:]
        if comp_topk_mask is not None:
            attn_output = _sparse_attention_reference(
                q.contiguous(),
                kv.contiguous(),
                k_comp.contiguous(),
                self.sinks,
                self.sliding_window,
                self.compress_rate,
                comp_topk_mask=comp_topk_mask,
            ).to(
                q.dtype
            )  # [B,H,S,D]
        else:
            attn_output = _v4flash_attention(
                q.contiguous(),
                kv.contiguous(),
                None if k_comp is None else k_comp.contiguous(),
                self.sinks,
                self.sliding_window,
                self.compress_rate,
            )  # [B,H,S,D]

        # HF eager hands back [B,S,H,D] (it transposes 1<->2 at the end); A1 gives
        # [B,H,S,D], so transpose to match HF's layout before the conjugate RoPE.
        attn_output = attn_output.transpose(1, 2)  # [B,S,H,D]

        # Conjugate -sin RoPE on the output's rope slice (HF:856). apply_rotary_pos_emb
        # wants [B,S,H,D] -> transpose(1,2) to [B,H,S,D], rotate, transpose back.
        attn_output = apply_rotary_pos_emb(attn_output.transpose(1, 2), cos, -sin).transpose(1, 2)  # [B,S,H,D]

        # Grouped low-rank output projection (HF:858-860).
        grouped = attn_output.reshape(*input_shape, self.config.o_groups, -1)
        grouped = self.o_a_proj(grouped).flatten(2)
        output = self.o_b_proj(grouped)
        return output


__all__ = ["V4Attention"]
