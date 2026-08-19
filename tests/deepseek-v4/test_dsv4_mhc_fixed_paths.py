"""Production mHC paths are fixed: official train and optimized TF32 rollout."""

from pathlib import Path

import pytest


NUM_GPUS = 0
REPO = Path(__file__).resolve().parents[2]


def test_train_path_is_fixed_to_official_mhc():
    decoder = (REPO / "custom_kernels/deepseek_v4/megatron/decoder.py").read_text()
    removed_env = "V4_MHC_" "TORCH"
    assert "hyper_connection_official as _HC" in decoder
    assert removed_env not in decoder
    assert "mhc.reference" not in decoder
    assert "_HC = _kernels.hyper_connection" not in decoder


def test_train_mhc_always_uses_full_autograd():
    model = (REPO / "custom_kernels/deepseek_v4/megatron/mcore_model.py").read_text()
    full_loop = (REPO / "scripts/dsv4/_dsv4_launch_core.sh").read_text()
    train_smoke = (REPO / "scripts/dsv4/train_smoke.sh").read_text()
    removed_env = "V4_MHC_" "MIXING_ORACLE"

    assert removed_env not in model
    assert removed_env not in full_loop
    assert removed_env not in train_smoke
    assert "post.detach()" not in model
    assert "comb.detach()" not in model


def test_rollout_launcher_keeps_tf32_optimizations_enabled():
    text = (REPO / "scripts/dsv4/run.t1.deepseek_v4_flash.rl.sh").read_text()
    assert "SGLANG_OPT_USE_TILELANG_MHC_PRE:-true" in text
    assert "SGLANG_OPT_DEEPGEMM_HC_PRENORM=False" not in text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
