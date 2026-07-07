"""V4-Flash custom decoder block + model (M0 scaffold).

This is the custom block the plan mandates instead of mcore's
``get_gpt_decoder_block_spec`` / ``TransformerLayer``: those carry a single
``[S,B,H]`` stream with ordinary BDA residuals and don't pass ``input_ids``.  V4
instead threads a ``[B, S, hc_mult, hidden]`` stack of parallel residual streams
through every layer, mixes it in/out at two mHC sites (attn + ffn), needs
``input_ids`` at the hash-MoE router, and collapses the stream with ``hc_head``
only before the final norm.

Per-layer the residual update is (HF:1129-1139, mHC comb consumed TRANSPOSED):

    post, comb, collapsed = attn_hc(hidden_streams)
    attn_out = self_attn(input_layernorm(collapsed))
    hidden_streams = post[...,None]*attn_out[...,None,:] + comb.transpose(-1,-2) @ hidden_streams
    post, comb, collapsed = ffn_hc(hidden_streams)
    mlp_out = mlp(post_attention_layernorm(collapsed), input_ids)
    hidden_streams = post[...,None]*mlp_out[...,None,:]  + comb.transpose(-1,-2) @ hidden_streams

mHC ``(post, comb, collapsed)`` come from the B1 kernel (``hyper_connection``);
the ``post·out + comb.T@stream`` mix stays here in torch.

MoE (routers + grouped experts + shared expert + clamped SwiGLU) and the final
``hc_head`` collapse reuse the HF modules verbatim (group-C ops; not kernels).
"""

import os

import torch
from torch import nn
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperHead, DeepseekV4SparseMoeBlock

from . import _kernels
from .attention import V4Attention
from .rope import DeepseekV4RotaryEmbedding, V4RMSNorm

# Default B1 path: the dtype-agnostic hybrid (no sglang coupling for the scaffold).
# Set V4_MHC_TORCH=1 to use the exact torch reference path when TileLang/TVM codegen
# is not stable on a target node; this is slower but keeps the training chain testable.
if os.environ.get("V4_MHC_TORCH", "0") == "1":
    from custom_kernels.deepseek_v4.mhc.reference import hyper_connection_forward as _HC
else:
    _HC = _kernels.hyper_connection


class V4HyperConnection(nn.Module):
    """Owns the mHC (fn, base, scale) params; calls the B1 kernel.

    Returns ``(post, comb, collapsed)`` exactly like ``DeepseekV4HyperConnection``
    (HF:864); the residual mix is applied by the decoder layer.
    """

    def __init__(self, config):
        super().__init__()
        self.hc_mult = config.hc_mult
        mix = (2 + self.hc_mult) * self.hc_mult
        self.fn = nn.Parameter(torch.empty(mix, self.hc_mult * config.hidden_size))
        self.base = nn.Parameter(torch.zeros(mix))
        self.scale = nn.Parameter(torch.ones(3))
        nn.init.normal_(self.fn, mean=0.0, std=config.initializer_range)

    def forward(self, hidden_streams: torch.Tensor):
        # B1 expects fn/base/scale in fp32 (mHC is _keep_in_fp32_modules).
        post, comb, collapsed = _HC(hidden_streams, self.fn.float(), self.base.float(), self.scale.float())
        return post, comb, collapsed


class V4DecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = V4Attention(config, layer_idx)
        self.mlp = DeepseekV4SparseMoeBlock(config, layer_idx)
        self.input_layernorm = V4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = V4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn_hc = V4HyperConnection(config)
        self.ffn_hc = V4HyperConnection(config)

    def forward(self, hidden_states, position_embeddings, input_ids):
        # hidden_states: [B, S, hc_mult, hidden]
        dtype = hidden_states.dtype

        post, comb, collapsed = self.attn_hc(hidden_states)
        attn_output = self.self_attn(self.input_layernorm(collapsed), position_embeddings)
        hidden_states = post.to(dtype).unsqueeze(-1) * attn_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_states
        )

        post, comb, collapsed = self.ffn_hc(hidden_states)
        mlp_output = self.mlp(self.post_attention_layernorm(collapsed), input_ids=input_ids)
        hidden_states = post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(
            comb.to(dtype).transpose(-1, -2), hidden_states
        )
        return hidden_states


class V4Model(nn.Module):
    """Embeddings -> custom V4 decoder layers (HC stream) -> hc_head -> norm.

    Returns ``last_hidden_state`` like ``DeepseekV4Model`` (HF:1237); no LM head
    (M0 only needs the hidden states for the no-NaN forward and, later, parity).
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList([V4DecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = V4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.hc_head = DeepseekV4HyperHead(config)

    def forward(self, input_ids, position_ids=None):
        B, S = input_ids.shape
        inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)

        # [B, S, hc_mult, hidden] stream (HF:1292).
        hidden_states = inputs_embeds.unsqueeze(2).expand(-1, -1, self.config.hc_mult, -1).contiguous()
        position_embeddings = {
            "main": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="main"),
            "compress": self.rotary_emb(inputs_embeds, position_ids=position_ids, layer_type="compress"),
        }

        for layer in self.layers:
            hidden_states = layer(hidden_states, position_embeddings, input_ids)

        # collapse the hc streams (hc_head) then final norm (HF:1309).
        hidden_states = self.norm(self.hc_head(hidden_states))
        return hidden_states


__all__ = ["V4Model", "V4DecoderLayer", "V4HyperConnection"]
