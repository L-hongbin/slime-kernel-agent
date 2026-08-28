"""Tests for the fixed serving-aligned DS-V4 shared-expert activation."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from custom_kernels.deepseek_v4.megatron.lora import apply_v4_lora
from custom_kernels.deepseek_v4.megatron.m0_smoke import tiny_config
from custom_kernels.deepseek_v4.megatron.mcore_model import V4SharedExpertMLP
from torch import nn

NUM_GPUS = 0
LIMIT = 10.0
_SHARED_LEAVES = ("gate_proj", "up_proj", "down_proj")


def _mk_mlp():
    torch.manual_seed(0)
    mlp = V4SharedExpertMLP(tiny_config())
    assert mlp.limit == LIMIT
    return mlp


def _aligned_reference(gate, up):
    return (F.silu(gate.clamp(max=LIMIT).float()) * up.clamp(-LIMIT, LIMIT).float()).to(gate.dtype)


def _legacy_reference(gate, up):
    return F.silu(gate.clamp(max=LIMIT)) * up.clamp(-LIMIT, LIMIT)


def test_activation_is_fixed_to_single_round_serving_boundary():
    mlp = _mk_mlp()
    gate = torch.randn(64, 512, dtype=torch.bfloat16)
    up = torch.randn(64, 512, dtype=torch.bfloat16)
    with torch.no_grad():
        actual = mlp._act(gate, up)
    assert torch.equal(actual, _aligned_reference(gate, up))
    assert not torch.equal(actual, _legacy_reference(gate, up))


def test_alignment_requires_silu():
    cfg = tiny_config()
    cfg.hidden_act = "gelu"
    mlp = V4SharedExpertMLP(cfg)
    gate = torch.randn(4, 8, dtype=torch.bfloat16)
    up = torch.randn(4, 8, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="silu"):
        mlp._act(gate, up)


class _Holder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        layer = nn.Module()
        layer.mlp = nn.Module()
        layer.mlp.shared_experts = V4SharedExpertMLP(cfg)
        self.layers = nn.ModuleList([layer])


def test_adapter_gradients_and_ste_clamp_are_preserved():
    cfg = tiny_config()
    holder = apply_v4_lora(
        _Holder(cfg),
        dim=4,
        alpha=8,
        dropout=0.0,
        shared_expert=True,
    ).bfloat16()
    mlp = holder.layers[0].mlp.shared_experts
    with torch.no_grad():
        for leaf in _SHARED_LEAVES:
            getattr(mlp, leaf).linear_out.weight.normal_(0, 0.05)
    x = torch.randn(2, 5, cfg.hidden_size, dtype=torch.bfloat16)
    mlp(x).float().sum().backward()
    for leaf in _SHARED_LEAVES:
        adapter = getattr(mlp, leaf)
        assert adapter.linear_in.weight.grad is not None
        assert adapter.linear_out.weight.grad is not None
        assert torch.isfinite(adapter.linear_in.weight.grad).all()
        assert torch.isfinite(adapter.linear_out.weight.grad).all()

    gate = torch.full((4, 8), 12.0, dtype=torch.bfloat16, requires_grad=True)
    up = torch.full((4, 8), -12.0, dtype=torch.bfloat16, requires_grad=True)
    mlp._act(gate, up).float().sum().backward()
    assert gate.grad is not None and gate.grad.abs().sum() > 0
    assert up.grad is not None and up.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + sglang JIT kernel")
def test_activation_byte_matches_serving_kernel():
    try:
        from sglang.jit_kernel.dsv4.moe import silu_and_mul_clamp
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"sglang JIT kernel unavailable: {exc}")
    mlp = _mk_mlp().cuda()
    gate = torch.randn(256, 2048, device="cuda", dtype=torch.bfloat16)
    up = torch.randn(256, 2048, device="cuda", dtype=torch.bfloat16)
    gate_up = torch.cat([gate, up], dim=-1).contiguous()
    reference = gate.new_empty(gate.shape)
    silu_and_mul_clamp(gate_up, reference, float(mlp.limit))
    assert torch.equal(mlp._act(gate, up), reference)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
