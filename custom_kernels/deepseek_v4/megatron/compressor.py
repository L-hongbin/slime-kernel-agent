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

import torch
from torch import nn

from . import _kernels
from .rope import DeepseekV4RotaryEmbedding, V4RMSNorm, apply_rotary_pos_emb


# Production always uses the validated TileLang kernels. Diagnostic harnesses
# that need the torch reference inject csa_compress_ref/hca_compress_ref
# explicitly instead of changing model behavior through inherited process env.
_csa_compress = _kernels.csa_compress
_hca_compress = _kernels.hca_compress


def _rope_compressed(
    compressed, rotary_emb, compress_rate, n_windows, batch, layer_type="compress", position_offset=0
):
    """RoPE the compressed entries at absolute positions ``w * compress_rate``.

    Stateless training path: ``first_window_position = 0`` (HF:408).

    ``position_offset`` (CP2): the global token position of the compressor input's row
    0.  Under context parallelism the input is ``[halo || local]`` whose first row sits
    at global position ``global_start - halo``, so window ``j`` gets absolute RoPE
    position ``j * compress_rate + position_offset``.  0 for the non-CP / rank-0 case
    (bit-identical to the zero-based path)."""
    positions = torch.arange(n_windows, device=compressed.device)
    positions = (positions * compress_rate + position_offset).unsqueeze(0).expand(batch, -1)
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

    def forward(self, hidden_states: torch.Tensor, *, position_offset: int = 0, drop_windows: int = 0) -> torch.Tensor:
        # hidden_states: [B, S, hidden]  (already HC-collapsed by the layer).  Under CP2
        # S = halo + l_local and the caller passes ``position_offset`` (global position
        # of row 0) + ``drop_windows`` (leading halo windows to drop); defaults (0, 0)
        # are the bit-identical non-CP path.
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
            compressed,
            self.rotary_emb,
            self.compress_rate,
            n_windows,
            batch,
            self.rope_layer_type,
            position_offset=position_offset,
        )
        # CP2: drop the first ``drop_windows`` (halo) windows -- owned by the left
        # neighbour.  RoPE is applied at absolute positions first, so the kept windows
        # already carry their global-position rotation.
        if drop_windows:
            compressed = compressed[:, drop_windows:]
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
        # Official HF indexer structure + official DeepGEMM fp8_mqa_logits
        # scoring on GPU (torch fallback keeps CPU/parity paths HF-exact).
        from custom_kernels.deepseek_v4.megatron.indexer import V4Indexer

        self.indexer = V4Indexer(config)

    def forward(self, hidden_states: torch.Tensor, *, position_offset: int = 0, drop_windows: int = 0) -> torch.Tensor:
        # Under CP2 S = halo + l_local; ``position_offset`` / ``drop_windows`` handle the
        # global RoPE position + boundary-window drop (defaults 0/0 == non-CP path).  The
        # dropped windows also absorb the CSA w==0 overlap special case (design B2).
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
            compressed,
            self.rotary_emb,
            self.compress_rate,
            n_windows,
            batch,
            self.rope_layer_type,
            position_offset=position_offset,
        )
        if drop_windows:
            compressed = compressed[:, drop_windows:]
        return compressed.unsqueeze(1)  # [B, 1, T, head_dim]


COMPRESSOR_CLASSES = {
    "sliding_attention": None,
    "compressed_sparse_attention": V4CSACompressor,
    "heavily_compressed_attention": V4HCACompressor,
}

__all__ = ["V4HCACompressor", "V4CSACompressor", "COMPRESSOR_CLASSES"]
