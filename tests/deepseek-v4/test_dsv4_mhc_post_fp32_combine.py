"""Tests for the fixed fp32 DS-V4 mHC post-combine schedule."""

from __future__ import annotations

import inspect

import pytest
import torch
from custom_kernels.deepseek_v4.megatron import mcore_model as mm

NUM_GPUS = 0
HC, HIDDEN = 4, 7168
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _bits(tensor):
    return tensor.contiguous().view(torch.int16)


def _inputs(n, seed=11):
    torch.manual_seed(seed)
    branch = torch.randn(n, HIDDEN, device=DEV).to(torch.bfloat16)
    hidden = torch.randn(n, HC, HIDDEN, device=DEV).to(torch.bfloat16)
    post = torch.sigmoid(torch.randn(n, HC, device=DEV, dtype=torch.float32)) * 2
    comb = torch.softmax(torch.randn(n, HC, HC, device=DEV, dtype=torch.float32), dim=-1)
    return branch, hidden, post, comb


def _reference(post, comb, branch, hidden):
    return (
        post.float().unsqueeze(-1) * branch.float().unsqueeze(-2)
        + torch.matmul(comb.float().transpose(-1, -2), hidden.float())
    ).to(torch.bfloat16)


def _legacy(post, comb, branch, hidden):
    return post.to(torch.bfloat16).unsqueeze(-1) * branch.unsqueeze(-2) + torch.matmul(
        comb.to(torch.bfloat16).transpose(-1, -2), hidden
    )


def test_fixed_combine_matches_fp32_single_round_reference():
    branch, hidden, post, comb = _inputs(16)
    actual = mm._v4_hc_post_combine(post, comb, branch, hidden, torch.bfloat16)
    assert torch.equal(_bits(actual), _bits(_reference(post, comb, branch, hidden)))
    assert not torch.equal(_bits(actual), _bits(_legacy(post, comb, branch, hidden)))


def test_decoder_layer_uses_fixed_helper_at_both_sites():
    source = inspect.getsource(mm.V4DecoderLayer.forward)
    assert source.count("_v4_hc_post_combine(") == 2
    assert "post.to(dtype)" not in source


def test_fixed_combine_gradients_are_finite():
    branch, hidden, post, comb = _inputs(4, seed=101)
    leaves = [tensor.detach().requires_grad_() for tensor in (branch, hidden, post, comb)]
    mm._v4_hc_post_combine(*leaves, torch.bfloat16).float().sum().backward()
    for tensor in leaves:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad.float()).all()
        assert tensor.grad.abs().sum() > 0


def test_fixed_combine_is_more_accurate_than_legacy():
    branch, hidden, post, comb = _inputs(64, seed=77)
    oracle = post.double().unsqueeze(-1) * branch.double().unsqueeze(-2) + torch.matmul(
        comb.double().transpose(-1, -2), hidden.double()
    )
    fixed = mm._v4_hc_post_combine(post, comb, branch, hidden, torch.bfloat16)
    legacy = _legacy(post, comb, branch, hidden)
    fixed_error = (fixed.double() - oracle).norm()
    legacy_error = (legacy.double() - oracle).norm()
    assert fixed_error < legacy_error


@pytest.mark.skipif(not torch.cuda.is_available(), reason="serving mHC needs CUDA")
def test_fixed_combine_matches_serving_schedule():
    serving = pytest.importorskip("sglang.srt.layers.mhc")
    serving.is_dsa_prefill_cp_round_robin_split = lambda: False
    branch, hidden, post, comb = _inputs(1, seed=24)
    trainer = mm._v4_hc_post_combine(post, comb, branch, hidden, torch.bfloat16)
    rollout = serving.mhc_post(branch, hidden, post.unsqueeze(-1), comb).view_as(trainer)
    differing = (_bits(trainer) != _bits(rollout)).float().mean()
    # Isolated bf16 boundary flips remain possible because the two paths use
    # different fp32 matmul accumulation orders (3/28672 elements for this seed).
    assert differing <= 1.2e-4


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
