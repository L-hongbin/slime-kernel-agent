"""Contracts for the fixed V4 attention kernel path and its test reference."""

import importlib
from pathlib import Path

import pytest
import torch

NUM_GPUS = 0

REPO = Path(__file__).resolve().parents[2]
ATTENTION = REPO / "custom_kernels" / "deepseek_v4" / "megatron" / "attention.py"
ACTCKPT_HARNESS = REPO / "scripts" / "dsv4" / "diagnostics" / "parity" / "test_act_ckpt_reentrant_memory.py"
RUNTIME_SURFACES = [
    ATTENTION,
    REPO / "slime" / "backends" / "megatron_utils" / "actor.py",
    REPO / "scripts" / "dsv4" / "full_loop_smoke.sh",
    REPO / "scripts" / "dsv4" / "train_smoke.sh",
    ACTCKPT_HARNESS,
]


def test_attention_production_path_is_fixed_to_kernel():
    removed_toggle = "V4_ATTENTION_" + "TORCH"
    for path in RUNTIME_SURFACES:
        assert removed_toggle not in path.read_text(), f"stale runtime toggle in {path}"

    assert "_v4flash_attention = _kernels.v4flash_attention" in ATTENTION.read_text()


def test_activation_memory_harness_injects_reference_explicitly():
    source = ACTCKPT_HARNESS.read_text()
    assert "attention._v4flash_attention = attention_reference_for_module" in source
    assert ").to(q.dtype)" in source


def test_removed_environment_toggle_cannot_change_backend(monkeypatch):
    removed_toggle = "V4_ATTENTION_" + "TORCH"
    monkeypatch.setenv(removed_toggle, "1")

    from custom_kernels.deepseek_v4.megatron import attention

    attention = importlib.reload(attention)
    assert attention._v4flash_attention is attention._kernels.v4flash_attention


def test_attention_torch_reference_remains_callable():
    from custom_kernels.deepseek_v4.attention.reference import attention_reference

    q = torch.randn(1, 3, 5, 4, requires_grad=True)
    k_raw = torch.randn(1, 1, 5, 4, requires_grad=True)
    k_comp = torch.randn(1, 1, 2, 4, requires_grad=True)
    sinks = torch.randn(3, requires_grad=True)
    out = attention_reference(q, k_raw, k_comp, sinks, 4, 2)
    out.sum().backward()

    assert torch.isfinite(out).all()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k_raw, k_comp, sinks))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
