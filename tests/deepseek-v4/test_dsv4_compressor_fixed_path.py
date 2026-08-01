"""Contracts for the fixed V4 compressor kernel path and its test reference."""

import importlib
from pathlib import Path

import pytest
import torch

NUM_GPUS = 0

REPO = Path(__file__).resolve().parents[2]
COMPRESSOR = REPO / "custom_kernels" / "deepseek_v4" / "megatron" / "compressor.py"
ACTCKPT_HARNESS = REPO / "scripts" / "dsv4" / "diagnostics" / "parity" / "test_act_ckpt_reentrant_memory.py"
RUNTIME_SURFACES = [
    COMPRESSOR,
    REPO / "slime" / "backends" / "megatron_utils" / "actor.py",
    REPO / "scripts" / "dsv4" / "full_loop_smoke.sh",
    REPO / "scripts" / "dsv4" / "train_smoke.sh",
    ACTCKPT_HARNESS,
]


def test_compressor_production_path_is_fixed_to_kernels():
    removed_toggle = "V4_COMPRESS_" + "TORCH"
    for path in RUNTIME_SURFACES:
        assert removed_toggle not in path.read_text(), f"stale runtime toggle in {path}"

    source = COMPRESSOR.read_text()
    assert "_csa_compress = _kernels.csa_compress" in source
    assert "_hca_compress = _kernels.hca_compress" in source


def test_activation_memory_harness_injects_references_explicitly():
    source = ACTCKPT_HARNESS.read_text()
    assert "compressor._csa_compress = csa_compress_ref" in source
    assert "compressor._hca_compress = hca_compress_ref" in source


def test_removed_environment_toggle_cannot_change_backend(monkeypatch):
    removed_toggle = "V4_COMPRESS_" + "TORCH"
    monkeypatch.setenv(removed_toggle, "1")

    from custom_kernels.deepseek_v4.megatron import compressor

    compressor = importlib.reload(compressor)
    assert compressor._csa_compress is compressor._kernels.csa_compress
    assert compressor._hca_compress is compressor._kernels.hca_compress


@pytest.mark.parametrize(
    ("name", "shape", "rate"),
    [("hca_compress_ref", (1, 8, 4), 4), ("csa_compress_ref", (1, 8, 8), 2)],
)
def test_compressor_torch_references_remain_callable(name, shape, rate):
    from custom_kernels.deepseek_v4.compression import reference

    kv = torch.randn(shape, requires_grad=True)
    gate = torch.randn(shape, requires_grad=True)
    position_bias = torch.randn(rate, shape[-1], requires_grad=True)
    weight = torch.randn(4, requires_grad=True)
    out = getattr(reference, name)(kv, gate, position_bias, weight, 1e-6, rate)
    out.sum().backward()

    assert torch.isfinite(out).all()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (kv, gate, position_bias, weight))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
