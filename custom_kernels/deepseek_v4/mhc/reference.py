"""Exact torch reference for the DeepSeek-V4-Flash mHC (Manifold-Constrained
Hyper-Connection) modules, ported verbatim from
``transformers/models/deepseek_v4/modeling_deepseek_v4.py``:

  * ``DeepseekV4UnweightedRMSNorm`` (modeling:66-72)
  * ``DeepseekV4HyperConnection``  (modeling:864-940)
  * ``DeepseekV4HyperHead``        (modeling:943-959)

This file is the numerical source of truth for ``kernel.py`` and the tests.
The reference always runs the mHC math in fp32 (the model marks these modules
``_keep_in_fp32_modules_strict``); the public forward casts ``collapsed`` back
to the input dtype, exactly like the model.

V4-Flash fixed dims: hidden D=4096, hc_mult H=4, hc_sinkhorn_iters=20,
hc_eps=1e-6, rms_norm_eps=1e-6.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---- fixed V4-Flash mHC config -------------------------------------------------
HC_MULT = 4  # H : number of parallel residual streams
HIDDEN = 4096  # D
HC_SINKHORN_ITERS = 20
HC_EPS = 1.0e-6
RMS_NORM_EPS = 1.0e-6
MIX = (2 + HC_MULT) * HC_MULT  # (2+H)*H = 24 : pre[H] + post[H] + comb[H*H]


def unweighted_rmsnorm(x: torch.Tensor, eps: float = RMS_NORM_EPS) -> torch.Tensor:
    """``DeepseekV4UnweightedRMSNorm`` (modeling:66-72).

    Normalises over the last axis, computing the variance in fp32 and rescaling
    back to ``x``'s dtype. The caller feeds an already-fp32 tensor here.
    """
    input_dtype = x.dtype
    # Model passes an already-fp32 tensor here, so computing the variance in the
    # tensor's own dtype matches the model bit-for-bit while also letting an
    # fp64 gradcheck stay in fp64 (the model's literal `.float()` would have
    # silently capped it at fp32).
    acc = x if input_dtype in (torch.float32, torch.float64) else x.float()
    var = acc.square().mean(-1, keepdim=True)
    return x * torch.rsqrt(var + eps).to(input_dtype)


def _cast(t: torch.Tensor, cdtype):
    """Cast to the mHC compute dtype. The model hardcodes fp32 (``cdtype`` left
    at its default); the tests raise it to fp64 for a numerically clean
    gradcheck of the analytic backward."""
    return t.to(cdtype) if cdtype is not None else t.float()


def hyper_connection_forward(
    hidden_streams: torch.Tensor,  # [B, S, H, D]
    fn: torch.Tensor,  # [MIX, H*D]
    base: torch.Tensor,  # [MIX]
    scale: torch.Tensor,  # [3]
    hc_mult: int = HC_MULT,
    sinkhorn_iters: int = HC_SINKHORN_ITERS,
    eps: float = HC_EPS,
    rms_eps: float = RMS_NORM_EPS,
    cdtype=None,
):
    """Verbatim port of ``DeepseekV4HyperConnection.forward`` (modeling:912-940).

    ``cdtype`` selects the internal compute dtype: ``None`` reproduces the model
    exactly (fp32); pass ``torch.float64`` for a clean gradcheck.

    Returns ``(post, comb, collapsed)``:
      * ``post``      [B, S, H]    cdtype  (block-output placement weights)
      * ``comb``      [B, S, H, H] cdtype  (doubly-stochastic residual mixer)
      * ``collapsed`` [B, S, D]    input dtype (stream-collapsed sublayer input)
    """
    hc = hc_mult
    in_dtype = hidden_streams.dtype
    flat = unweighted_rmsnorm(_cast(hidden_streams.flatten(start_dim=2), cdtype), rms_eps)
    pre_w, post_w, comb_w = F.linear(flat, _cast(fn, cdtype)).split([hc, hc, hc * hc], dim=-1)
    pre_b, post_b, comb_b = _cast(base, cdtype).split([hc, hc, hc * hc])
    pre_scale, post_scale, comb_scale = _cast(scale, cdtype).unbind(0)

    pre = torch.sigmoid(pre_w * pre_scale + pre_b) + eps
    post = 2 * torch.sigmoid(post_w * post_scale + post_b)
    comb_logits = comb_w.view(*comb_w.shape[:-1], hc, hc) * comb_scale + comb_b.view(hc, hc)
    comb = torch.softmax(comb_logits, dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    collapsed = (pre.unsqueeze(-1) * hidden_streams).sum(dim=2).to(in_dtype)
    return post, comb, collapsed


def hyper_head_forward(
    x: torch.Tensor,  # [B, S, H, D]
    hc_fn: torch.Tensor,  # [H, H*D]
    hc_base: torch.Tensor,  # [H]
    hc_scale: torch.Tensor,  # [1]
    eps: float = HC_EPS,
    rms_eps: float = RMS_NORM_EPS,
) -> torch.Tensor:
    """Verbatim port of ``DeepseekV4HyperHead.forward`` (modeling:955-959).

    Returns the collapsed stream ``[B, S, D]`` in the input dtype.
    """
    in_dtype = x.dtype
    flat = unweighted_rmsnorm(x.flatten(2).float(), rms_eps)
    mixes = F.linear(flat, hc_fn.float())
    pre = torch.sigmoid(mixes * hc_scale.float() + hc_base.float()) + eps
    return (pre.unsqueeze(-1) * x).sum(dim=2).to(in_dtype)


# ---- nn.Module wrappers (handy for torch.compile baselines & param init) -------


class HyperConnectionRef(torch.nn.Module):
    def __init__(
        self,
        hidden=HIDDEN,
        hc_mult=HC_MULT,
        sinkhorn_iters=HC_SINKHORN_ITERS,
        eps=HC_EPS,
        rms_eps=RMS_NORM_EPS,
        dtype=torch.float32,
        device="cuda",
        seed: int = 0,
    ):
        super().__init__()
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.eps = eps
        self.rms_eps = rms_eps
        mix = (2 + hc_mult) * hc_mult
        g = torch.Generator(device=device).manual_seed(seed)
        # init scale matching the magnitudes the trained model would carry:
        # fn ~ small (it acts on a 16384-wide normalised vector), scale O(1).
        self.fn = torch.nn.Parameter(
            torch.randn(mix, hc_mult * hidden, generator=g, device=device, dtype=dtype) / (hc_mult * hidden) ** 0.5
        )
        self.base = torch.nn.Parameter(torch.randn(mix, generator=g, device=device, dtype=dtype) * 0.1)
        self.scale = torch.nn.Parameter(torch.rand(3, generator=g, device=device, dtype=dtype) + 0.5)

    def forward(self, hidden_streams):
        return hyper_connection_forward(
            hidden_streams, self.fn, self.base, self.scale, self.hc_mult, self.sinkhorn_iters, self.eps, self.rms_eps
        )


class HyperHeadRef(torch.nn.Module):
    def __init__(
        self,
        hidden=HIDDEN,
        hc_mult=HC_MULT,
        eps=HC_EPS,
        rms_eps=RMS_NORM_EPS,
        dtype=torch.float32,
        device="cuda",
        seed: int = 1,
    ):
        super().__init__()
        self.eps = eps
        self.rms_eps = rms_eps
        g = torch.Generator(device=device).manual_seed(seed)
        self.hc_fn = torch.nn.Parameter(
            torch.randn(hc_mult, hc_mult * hidden, generator=g, device=device, dtype=dtype) / (hc_mult * hidden) ** 0.5
        )
        self.hc_base = torch.nn.Parameter(torch.randn(hc_mult, generator=g, device=device, dtype=dtype) * 0.1)
        self.hc_scale = torch.nn.Parameter(torch.rand(1, generator=g, device=device, dtype=dtype) + 0.5)

    def forward(self, x):
        return hyper_head_forward(x, self.hc_fn, self.hc_base, self.hc_scale, self.eps, self.rms_eps)
