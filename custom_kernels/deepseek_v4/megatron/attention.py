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

import torch
from torch import nn
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4GroupedLinear

from . import _kernels
from .compressor import COMPRESSOR_CLASSES
from .cp_utils import (
    CP_HALO,
    assert_local_len_aligned,
    compressor_drop_windows,
    cp_allgather_compressed,
    cp_halo_exchange,
    get_cp_info,
)
from .rope import V4RMSNorm, V4UnweightedRMSNorm, apply_rotary_pos_emb

# Production always uses the validated TileLang kernel. Diagnostic harnesses
# inject attention_reference explicitly instead of changing model behavior via
# inherited process environment.
_v4flash_attention = _kernels.v4flash_attention


class V4Attention(nn.Module):
    def __init__(self, config, layer_idx: int, *, layer_type_override: str | None = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        # ``layer_type_override`` lets a caller build a variant decoupled from the
        # per-layer schedule in ``config.layer_types``.  The V4 MTP head uses
        # ``"sliding_attention"`` here: its attention has NO compressor/indexer
        # (sglang builds the NextN attention with ``compress_ratio=0``, which uses the
        # main RoPE + sliding window and skips the compressor), matching the fact that
        # the checkpoint's ``mtp.0.attn.*`` subtree has no compressor/indexer tensors.
        self.layer_type = layer_type_override if layer_type_override is not None else config.layer_types[layer_idx]
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
        # hidden_states: [B, S, hidden]  (HC-collapsed + input_layernorm'd by the layer).
        # Under context parallelism (CP2) S is the rank's LOCAL contiguous shard length
        # l_local; the receptive field is completed by a left-halo exchange + a
        # compressed all-gather (see cp_utils).  cp_size==1 takes the original path,
        # bit-identical to pre-CP behavior (no CP op is constructed).
        input_shape = hidden_states.shape[:-1]  # (B, l_local)
        hidden_shape = (*input_shape, -1, self.head_dim)
        cos, sin = position_embeddings[self.rope_layer_type]

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(*hidden_shape).transpose(1, 2)  # [B,H,S,D]
        q = self.q_b_norm(q)
        q = apply_rotary_pos_emb(q, cos, sin)  # LOCAL queries; cos/sin carry global positions under CP

        cp_info = get_cp_info()
        if cp_info.enabled:
            attn_output = self._forward_attn_cp(hidden_states, position_embeddings, cos, sin, q, cp_info, input_shape)
        else:
            attn_output = self._forward_attn_local(hidden_states, hidden_shape, cos, sin, q, q_residual, input_shape)

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

    def _forward_attn_local(self, hidden_states, hidden_shape, cos, sin, q, q_residual, input_shape):
        """Non-CP (cp_size==1) attention core -- the original whole-sequence path.

        Returns the pre-o_proj attention output ``[B,H,S,D]``.  Kept byte-identical to
        the pre-CP forward so ``cp_size==1`` is unchanged."""
        kv = self.kv_norm(self.kv_proj(hidden_states)).view(*hidden_shape).transpose(1, 2)  # [B,1,S,D]
        kv = apply_rotary_pos_emb(kv, cos, sin)

        k_comp = None
        if self.compressor is not None:
            k_comp = self.compressor(hidden_states)  # [B,1,T,head_dim] post-RoPE, or T==0
            if k_comp.shape[2] == 0:
                k_comp = None

        # A1: shared-KV MQA (k_raw==v), per-head sink, structural sliding-window +
        # causal-threshold compressed masks.  Returns [B,H,S,D] pre-o_proj.
        # The maintained trainer path is dense over the compressed region.
        return _v4flash_attention(
            q.contiguous(),
            kv.contiguous(),
            None if k_comp is None else k_comp.contiguous(),
            self.sinks,
            self.sliding_window,
            self.compress_rate,
            comp_topk_mask=None,
        )  # [B,H,S,D]

    def _forward_attn_cp(self, hidden_states, position_embeddings, cos, sin, q, cp_info, input_shape):
        """Context-parallel (cp_size>1) attention core (design CP2 dataflow §4).

        halo exchange -> local kv_proj over ``[halo || local]`` -> local compressor with
        global position offset + boundary-window drop -> all-gather the owned compressed
        windows into the global comp axis -> ONE A1 kernel call with ``q_pos0`` /
        ``raw_halo``.  The halo rows' gradient flows back through the SAME
        ``cp_halo_exchange`` (it feeds both kv_proj and the compressor), and the global
        ``dk_comp`` reduce-scatters to owners inside ``cp_allgather_compressed`` -- no
        hand-routed boundary gradients.  Returns ``[B,H,l_local,D]`` pre-o_proj.
        """
        B, l_local = input_shape
        assert_local_len_aligned(l_local, cp_info)
        this_halo = cp_info.local_halo(CP_HALO)  # 128 for rank>0, 0 for rank0
        global_start = cp_info.global_start(l_local)  # rank * l_local (contiguous partition)

        # ONE shared halo exchange feeding BOTH kv_proj and the compressor (design M5).
        hidden_haloed = cp_halo_exchange(hidden_states, cp_info)  # [B, this_halo+l_local, H]
        kv_len = hidden_haloed.shape[1]
        kv_shape = (B, kv_len, -1, self.head_dim)
        kv = self.kv_norm(self.kv_proj(hidden_haloed)).view(*kv_shape).transpose(1, 2)  # [B,1,this_halo+l_local,D]
        if this_halo > 0:
            # k_raw spans global positions [global_start-halo, global_start+l_local);
            # rank0 (this_halo==0) reuses the local (== global) cos/sin.
            cos_kv, sin_kv = position_embeddings[self.rope_layer_type + "_haloed"]
        else:
            cos_kv, sin_kv = cos, sin
        kv = apply_rotary_pos_emb(kv, cos_kv, sin_kv)

        k_comp = None
        if self.compressor is not None:
            drop = compressor_drop_windows(this_halo, self.compress_rate)  # halo // m (CSA 32 / HCA 1)
            k_comp_local = self.compressor(
                hidden_haloed, position_offset=global_start - this_halo, drop_windows=drop
            )  # [B,1,l_local/m,head_dim] -- the windows this rank OWNS
            if k_comp_local.shape[2] > 0:
                # All-gather owned windows -> global comp axis (rank order); the causal
                # threshold inside A1 masks each rank's future (other-rank) windows.
                k_comp = cp_allgather_compressed(k_comp_local, cp_info)  # [B,1,S/m,head_dim]

        return _v4flash_attention(
            q.contiguous(),
            kv.contiguous(),
            None if k_comp is None else k_comp.contiguous(),
            self.sinks,
            self.sliding_window,
            self.compress_rate,
            comp_topk_mask=None,
            q_pos0=global_start,
            raw_halo=this_halo,
        )  # [B,H,l_local,D]


__all__ = ["V4Attention"]
