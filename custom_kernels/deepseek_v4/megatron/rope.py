"""V4-Flash RoPE + norms, ported 1:1 from HF ``modeling_deepseek_v4.py``.

Kept torch-side (the kernel inventory classifies interleaved partial RoPE and the
unweighted/weighted RMSNorms as cheap elementwise / trivial ops — group C).  The
math here must match HF bit-for-bit for the M1 parity gate, so this is a faithful
copy of:

  * ``rotate_half``            (HF:335)
  * ``apply_rotary_pos_emb``   (HF:342)  -- interleaved, trailing rope slice
  * ``DeepseekV4RotaryEmbedding.forward`` (HF:153)  -- half-size cos/sin
  * ``DeepseekV4RMSNorm`` / ``DeepseekV4UnweightedRMSNorm`` (HF:46 / :66)
"""

import torch
from torch import nn

from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4RotaryEmbedding,
    apply_rotary_pos_emb,
    rotate_half,
)


class V4RMSNorm(nn.Module):
    """Weighted RMSNorm over the last dim (HF DeepseekV4RMSNorm)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # Match sglang's RMSNorm (cast_x_before_out_mul=False, its default): apply the
        # weight in fp32 and cast the PRODUCT to the I/O dtype, rather than HF's
        # `weight * x.to(bf16)` (cast the normalized x first). Aligns train→rollout
        # and differs from HF eager by about one bf16 ulp.
        return (self.weight * hidden_states).to(input_dtype)


class V4UnweightedRMSNorm(nn.Module):
    """Unweighted RMSNorm over the last dim (HF DeepseekV4UnweightedRMSNorm)."""

    def __init__(self, eps: float = 1.0e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)


__all__ = [
    "V4RMSNorm",
    "V4UnweightedRMSNorm",
    "DeepseekV4RotaryEmbedding",
    "apply_rotary_pos_emb",
    "rotate_half",
]
