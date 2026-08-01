"""V4-Flash CSA / HCA compressor modules (M0 scaffold).

Each module owns the torch-side glue around the B2 pool kernel:

  * ``kv_proj`` / ``gate_proj`` linears + ``position_bias`` parameter + ``kv_norm``.
  * HCA: non-overlapping windows; the pool core is ``hca_compress`` (B2).
  * CSA: the Ca/Cb overlap layout (HF:644-657) is folded INTO the fused
    ``csa_compress`` kernel (it reads the raw ``[B,S,2*head_dim]`` projections and
    builds the overlap in-kernel), so no torch ``new_kv``/``new_gate`` here.
  * RoPE on the compressed entries at their deterministic absolute positions
    ``w * compress_rate`` (stateless / no-cache training path; HF:421-424).
  * The Lightning Indexer + top-k is DROPPED (dense-over-compressed; tiny seq makes
    it a no-op, see dsv4_kernel_inventory.md).  So no ``block_bias`` is materialised:
    the causal-threshold mask over compressed entries is applied STRUCTURALLY inside
    the A1 attention kernel (``(w+1)*m <= qpos+1``), which equals HF's
    ``block_bias`` at tiny seq.

Returns ``compressed_kv`` shaped ``[B, 1, T, head_dim]`` (post-RoPE, K==V), ready
for the attention module to concatenate onto A1's ``k_comp`` argument.

Math truth: ``DeepseekV4HCACompressor`` (HF:362) / ``DeepseekV4CSACompressor`` (HF:579).
"""

import os

import torch
from torch import nn

from . import _kernels
from .rope import DeepseekV4RotaryEmbedding, V4RMSNorm, apply_rotary_pos_emb


# Set V4_COMPRESS_TORCH=1 to use the exact torch reference path when TileLang/TVM
# codegen is not stable on a target node. This is slower but keeps the training
# chain testable without changing the default performance path.
if os.environ.get("V4_COMPRESS_TORCH", "0") == "1":
    from custom_kernels.deepseek_v4.compression.reference import csa_compress_ref as _csa_compress
    from custom_kernels.deepseek_v4.compression.reference import hca_compress_ref as _hca_compress
else:
    _csa_compress = _kernels.csa_compress
    _hca_compress = _kernels.hca_compress


def _rope_compressed(compressed, rotary_emb, compress_rate, n_windows, batch, layer_type="compress"):
    """RoPE the compressed entries at absolute positions ``w * compress_rate``.

    Stateless training path: ``first_window_position = 0`` (HF:408)."""
    positions = torch.arange(n_windows, device=compressed.device)
    positions = (positions * compress_rate).unsqueeze(0).expand(batch, -1)
    cos, sin = rotary_emb(compressed, position_ids=positions, layer_type=layer_type)
    return apply_rotary_pos_emb(compressed.unsqueeze(1), cos, sin).squeeze(1)


class V4HCACompressor(nn.Module):
    """Heavily Compressed Attention compressor (HF:362). Non-overlapping windows."""

    rope_layer_type = "compress"

    def __init__(self, config):
        super().__init__()
        self.compress_rate = config.compress_rates["heavily_compressed_attention"]
        self.head_dim = config.head_dim
        self.eps = config.rms_norm_eps
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(self.compress_rate, self.head_dim))
        self.kv_norm = V4RMSNorm(self.head_dim, eps=self.eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, S, hidden]  (already HC-collapsed by the layer)
        batch, S, _ = hidden_states.shape
        kv = self.kv_proj(hidden_states)  # [B, S, head_dim]
        gate = self.gate_proj(hidden_states)
        n_windows = S // self.compress_rate
        if n_windows == 0:
            return kv.new_zeros((batch, 1, 0, self.head_dim))

        # B2 fused HCA pool: in-kernel windowed gather + position_bias + softmax +
        # RMSNorm over head_dim.  weight = kv_norm.weight (RMSNorm gain).
        compressed = _hca_compress(
            kv.contiguous(),
            gate.contiguous(),
            self.position_bias,
            self.kv_norm.weight,
            self.eps,
            self.compress_rate,
        )  # [B, n_windows, head_dim]
        compressed = _rope_compressed(
            compressed, self.rotary_emb, self.compress_rate, n_windows, batch, self.rope_layer_type
        )
        return compressed.unsqueeze(1)  # [B, 1, T, head_dim]


class V4CSACompressor(nn.Module):
    """Compressed Sparse Attention compressor (HF:579). Ca/Cb overlap, width 2*m.

    The indexer (HF:448) is instantiated for state-dict completeness but its top-k
    is NOT applied (dropped per the inventory; at tiny seq it is a no-op).
    """

    rope_layer_type = "compress"

    def __init__(self, config):
        super().__init__()
        self.compress_rate = config.compress_rates["compressed_sparse_attention"]
        self.head_dim = config.head_dim
        self.eps = config.rms_norm_eps
        self.kv_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, 2 * self.head_dim, bias=False)
        self.position_bias = nn.Parameter(torch.zeros(self.compress_rate, 2 * self.head_dim))
        self.kv_norm = V4RMSNorm(self.head_dim, eps=self.eps)
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        # Indexer kept for parity / state-dict mapping (M3+); not used in M0/M1.
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Indexer

        self.indexer = DeepseekV4Indexer(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, S, _ = hidden_states.shape
        kv = self.kv_proj(hidden_states)  # [B, S, 2*head_dim]
        gate = self.gate_proj(hidden_states)
        n_windows = S // self.compress_rate
        if n_windows == 0:
            return kv.new_zeros((batch, 1, 0, self.head_dim))

        # B2 fused CSA pool: reads raw [B,S,2D] projections and folds the Ca/Cb
        # overlap (+position_bias) in-kernel, then softmax-pools + RMSNorm.
        compressed = _csa_compress(
            kv.contiguous(),
            gate.contiguous(),
            self.position_bias,
            self.kv_norm.weight,
            self.eps,
            self.compress_rate,
        )  # [B, n_windows, head_dim]
        compressed = _rope_compressed(
            compressed, self.rotary_emb, self.compress_rate, n_windows, batch, self.rope_layer_type
        )
        return compressed.unsqueeze(1)  # [B, 1, T, head_dim]


COMPRESSOR_CLASSES = {
    "sliding_attention": None,
    "compressed_sparse_attention": V4CSACompressor,
    "heavily_compressed_attention": V4HCACompressor,
}

__all__ = ["V4HCACompressor", "V4CSACompressor", "COMPRESSOR_CLASSES"]
