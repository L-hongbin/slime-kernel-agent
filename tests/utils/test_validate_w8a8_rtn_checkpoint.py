import pytest
import torch
from safetensors.torch import save_file
from scripts.quantize.utils.validate_checkpoint import check_w8a8_checkpoint

from slime.backends.megatron_utils.megatron_to_hf.processors.quantizer_compressed_tensors import quantize_layer_int8


def _write_single_safetensors_checkpoint(path, tensors):
    path.mkdir()
    save_file(tensors, path / "model.safetensors")


def test_validate_w8a8_checkpoint_accepts_real_rtn(tmp_path):
    torch.manual_seed(0)
    weight_name = "model.layers.0.mlp.gate_proj.weight"
    w = torch.randn(32, 128, dtype=torch.bfloat16)
    q, scale, zp = quantize_layer_int8(w, group_size=None, strategy="channel", sym=True)
    assert zp is None

    ref_dir = tmp_path / "ref"
    q_dir = tmp_path / "w8a8"
    _write_single_safetensors_checkpoint(ref_dir, {weight_name: w})
    _write_single_safetensors_checkpoint(
        q_dir,
        {
            weight_name: q,
            weight_name.replace(".weight", ".weight_scale"): scale,
        },
    )

    checks = check_w8a8_checkpoint(q_dir, reference_checkpoint=ref_dir)

    assert len(checks) == 1
    assert checks[0].unique_count > 3
    assert checks[0].saturated_frac < 0.20
    assert checks[0].rel_l2 is not None
    assert checks[0].rel_l2 < 0.08


def test_validate_w8a8_checkpoint_accepts_1d_channel_scale(tmp_path):
    torch.manual_seed(2)
    weight_name = "model.layers.0.mlp.gate_proj.weight"
    w = torch.randn(16, 64, dtype=torch.bfloat16)
    q, scale, zp = quantize_layer_int8(w, group_size=None, strategy="channel", sym=True)
    assert zp is None

    ref_dir = tmp_path / "ref"
    q_dir = tmp_path / "w8a8"
    _write_single_safetensors_checkpoint(ref_dir, {weight_name: w})
    _write_single_safetensors_checkpoint(
        q_dir,
        {
            weight_name: q,
            weight_name.replace(".weight", ".weight_scale"): scale.squeeze(1),
        },
    )

    checks = check_w8a8_checkpoint(q_dir, reference_checkpoint=ref_dir)

    assert len(checks) == 1
    assert checks[0].rel_l2 is not None
    assert checks[0].rel_l2 < 0.08


def test_validate_w8a8_checkpoint_rejects_ternary_sign_quantization(tmp_path):
    torch.manual_seed(1)
    weight_name = "model.layers.0.mlp.gate_proj.weight"
    w = torch.randn(32, 128, dtype=torch.bfloat16)
    # This mimics the broken checkpoint failure mode: almost all values are
    # sign-only saturation, not real INT8 RTN levels.
    scale = (w.to(torch.float32).abs().amax(dim=1, keepdim=True) / 127).clamp(min=1e-8)
    q = torch.where(w.to(torch.float32) >= 0, 127, -128).to(torch.int8)

    ref_dir = tmp_path / "ref"
    q_dir = tmp_path / "w8a8"
    _write_single_safetensors_checkpoint(ref_dir, {weight_name: w})
    _write_single_safetensors_checkpoint(
        q_dir,
        {
            weight_name: q,
            weight_name.replace(".weight", ".weight_scale"): scale,
        },
    )

    with pytest.raises(AssertionError, match="looks ternary|saturated_frac|rel_l2"):
        check_w8a8_checkpoint(q_dir, reference_checkpoint=ref_dir)
