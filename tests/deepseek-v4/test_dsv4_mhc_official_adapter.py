"""Contracts for the local adapter around official TileKernels mHC."""

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from custom_kernels.deepseek_v4.mhc import official as official_mhc


NUM_GPUS = 0


def test_sinkhorn_backward_gradient_is_made_contiguous():
    grad = torch.randn(2, 4, 4).transpose(-1, -2)
    assert not grad.is_contiguous() and grad.stride(-1) != 1

    normalized = official_mhc._contiguous_grad(grad)

    assert normalized.is_contiguous()
    assert normalized.stride(-1) == 1
    assert torch.equal(normalized, grad)


def test_official_adapter_preserves_local_call_contract(monkeypatch):
    hidden = torch.zeros((1, 2, 4, 3), dtype=torch.bfloat16)
    fn = torch.tensor([1.0])
    base = torch.tensor([2.0])
    scale = torch.tensor([3.0])
    mixes = torch.tensor([4.0])
    pre_mix = torch.tensor([5.0])
    post_mix = torch.arange(8.0).reshape(1, 2, 4, 1)
    comb_mix = torch.tensor([6.0])
    comb = torch.randn((1, 2, 4, 4), requires_grad=True)
    collapsed = torch.randn((1, 2, 3))
    calls = []

    def norm_fn(residual, mhc_fn, norm_weight, eps, *, fuse_grad_acc, n_splits):
        calls.append(("norm", residual, mhc_fn, norm_weight, eps, fuse_grad_acc, n_splits))
        return mixes

    def split_mixes(input_mixes, mhc_scale, mhc_base, hc_mult, post_mult, eps):
        calls.append(("split", input_mixes, mhc_scale, mhc_base, hc_mult, post_mult, eps))
        return pre_mix, post_mix, comb_mix

    def sinkhorn(input_mix, *, repeat, eps):
        calls.append(("sinkhorn", input_mix, repeat, eps))
        return comb

    def apply_mix(residual, input_mix):
        calls.append(("apply", residual, input_mix))
        return collapsed

    monkeypatch.setattr(official_mhc, "_ops", (norm_fn, split_mixes, sinkhorn, apply_mix))

    post_result, comb_result, collapsed_result = official_mhc.hyper_connection_official(hidden, fn, base, scale)

    assert [call[0] for call in calls] == ["norm", "split", "sinkhorn", "apply"]
    assert calls[0][1] is hidden and calls[0][2] is fn
    assert calls[0][3:] == (None, 1e-6, False, 16)
    assert calls[1][1] is mixes and calls[1][2] is scale and calls[1][3] is base
    assert calls[1][4:] == (4, 2.0, 1e-6)
    assert calls[2][1] is comb_mix and calls[2][2:] == (20, 1e-6)
    assert calls[3][1] is hidden and calls[3][2] is pre_mix
    assert torch.equal(post_result, post_mix.squeeze(-1))
    assert comb_result is comb
    assert collapsed_result is collapsed


def test_official_adapter_rejects_non_bf16_residual():
    with pytest.raises(AssertionError, match="reference.hyper_connection_forward"):
        official_mhc.hyper_connection_official(
            torch.zeros((1, 1, 4, 3), dtype=torch.float32),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
        )


def test_ds_v4_decoder_defaults_to_official_mhc():
    decoder_path = REPO_ROOT / "custom_kernels" / "deepseek_v4" / "megatron" / "decoder.py"
    source = decoder_path.read_text()
    removed_env = "V4_MHC_" "TORCH"

    assert "from custom_kernels.deepseek_v4.mhc.official import hyper_connection_official as _HC" in source
    assert removed_env not in source
    assert "mhc.reference" not in source
    assert "_HC = _kernels.hyper_connection" not in source


def test_full_loop_exports_tilekernels_for_train_actors():
    launcher_path = REPO_ROOT / "scripts" / "dsv4" / "_dsv4_launch_core.sh"
    source = launcher_path.read_text()

    assert "TILEKERNELS_DIR=${TILEKERNELS_DIR:-/nfs/FM/chenshuailin/projects/kernel_agents/TileKernels}" in source
    assert 'export PYTHONPATH="${REPO}:/root/Megatron-LM:${TILEKERNELS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"' in source


def test_train_smoke_exports_tilekernels_without_mhc_runtime_switch():
    launcher_path = REPO_ROOT / "scripts" / "dsv4" / "train_smoke.sh"
    source = launcher_path.read_text()
    removed_env = "V4_MHC_" "TORCH"

    assert "TILEKERNELS_DIR=${TILEKERNELS_DIR:-/nfs/FM/chenshuailin/projects/kernel_agents/TileKernels}" in source
    assert 'export PYTHONPATH="${REPO}:/root/Megatron-LM:${TILEKERNELS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"' in source
    assert '"PYTHONPATH": "${REPO}:/root/Megatron-LM:${TILEKERNELS_DIR}"' in source
    assert removed_env not in source


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
