"""Adapter around DeepSeek's official TileKernels mHC autograd ops.

Exposes ``hyper_connection_official(hidden_streams, fn, base, scale) ->
(post, comb, collapsed)`` with the signature and return convention expected by
the local DS-V4 decoder.

Equivalence config (validated; do NOT change silently):
- ``post_mult_value=2.0``  (our reference hardcodes 2*sigmoid; official default 1.0)
- ``sinkhorn repeat=20``   (= hc_sinkhorn_iters; official default 10)
- all eps = 1e-6 (norm/pre/sinkhorn)
- official arg order is (fn, SCALE, BASE) — swapped vs ours (fn, BASE, SCALE)
- official post_mix carries a trailing singleton dim -> squeeze(-1)

Built from the split ops with ``fuse_grad_acc=False`` so plain autograd
accumulates the residual grad. The residual must be bf16 because the official
norm kernel requires it; fp32/fp64 diagnostics call
``mhc.reference.hyper_connection_forward`` directly.

Requires ``tile_kernels`` importable — the launcher extends the actors'
PYTHONPATH with the TileKernels repo (TileLang 0.1.8 suffices for these ops;
only the unused multilayer_recompute needs 0.1.9).
"""

from __future__ import annotations

import torch

_POST_MULT = 2.0
_SINKHORN_ITERS = 20
_EPS = 1e-6
_N_SPLITS = 16

_ops = None


def _contiguous_grad(grad: torch.Tensor) -> torch.Tensor:
    """TileKernels Sinkhorn backward requires contiguous grad_output."""
    return grad.contiguous()


def _load_ops():
    global _ops
    if _ops is None:
        from tile_kernels.modeling.mhc.ops.norm_fn import mhc_pre_norm_fn
        from tile_kernels.modeling.mhc.ops.pre_apply_mix import mhc_pre_apply_mix
        from tile_kernels.modeling.mhc.ops.pre_split_mixes import mhc_pre_split_mixes
        from tile_kernels.modeling.mhc.ops.sinkhorn import sinkhorn_normalize

        _ops = (mhc_pre_norm_fn, mhc_pre_split_mixes, sinkhorn_normalize, mhc_pre_apply_mix)
    return _ops


def hyper_connection_official(hidden_streams: torch.Tensor, fn: torch.Tensor, base: torch.Tensor, scale: torch.Tensor):
    """TileKernels-official mHC with our (post, comb, collapsed) convention.

    hidden_streams: [B, S, hc_mult, hidden] bf16; fn/base/scale fp32 (same
    dtypes the DS-V4 hyper-connection caller already passes).
    """
    assert hidden_streams.dtype == torch.bfloat16, (
        "official TileKernels mHC is bf16-only (norm_fn kernel asserts); "
        f"got {hidden_streams.dtype} — call mhc.reference.hyper_connection_forward "
        "directly for fp32/fp64 diagnostics"
    )
    norm_fn, split_mixes, sinkhorn, apply_mix = _load_ops()
    hc_mult = hidden_streams.shape[-2]
    mixes = norm_fn(hidden_streams, fn, None, _EPS, fuse_grad_acc=False, n_splits=_N_SPLITS)
    pre_mix, post_mix, comb_mix = split_mixes(mixes, scale, base, hc_mult, _POST_MULT, _EPS)
    comb = sinkhorn(comb_mix, repeat=_SINKHORN_ITERS, eps=_EPS)
    if comb.requires_grad:
        # The decoder consumes comb.transpose(-1, -2), whose autograd gradient
        # has a non-unit last stride. The official backward kernel asserts a
        # contiguous grad_output, so normalize it at our adapter boundary.
        comb.register_hook(_contiguous_grad)
    collapsed = apply_mix(hidden_streams, pre_mix)
    return post_mix.squeeze(-1), comb, collapsed
