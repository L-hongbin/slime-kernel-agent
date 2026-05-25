"""Tests for the W8A8 INT8 RTN path in quantizer_compressed_tensors.

Runs on CPU. INT4 path uses a CUDA kernel so it is not exercised here.
"""

import pytest
import torch

from slime.backends.megatron_utils.megatron_to_hf.processors.quantizer_compressed_tensors import (
    quantize_layer_int8,
    quantize_params_compressed_tensors,
)


def _make_w8a8_int8_config(strategy, symmetric=True, group_size=None, ignore=None):
    weights = {
        "num_bits": 8,
        "type": "int",
        "symmetric": symmetric,
        "strategy": strategy,
        "dynamic": False,
    }
    if strategy == "group":
        weights["group_size"] = group_size
    return {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": weights,
                "input_activations": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "token",
                    "dynamic": True,
                },
                "output_activations": None,
            }
        },
        "ignore": ignore or [],
    }


def _dequant_int8_sym(q, scale, in_features):
    out_features, num_groups = scale.shape
    eff_group = in_features // num_groups
    return (
        q.to(torch.float32).view(out_features, num_groups, eff_group) * scale.unsqueeze(-1).to(torch.float32)
    ).view(out_features, in_features)


def _dequant_int8_asym(q, scale, zp, in_features):
    out_features, num_groups = scale.shape
    eff_group = in_features // num_groups
    qf = q.to(torch.float32).view(out_features, num_groups, eff_group)
    return ((qf - zp.to(torch.float32).unsqueeze(-1)) * scale.unsqueeze(-1).to(torch.float32)).view(
        out_features, in_features
    )


def test_quantize_layer_int8_per_channel_sym():
    torch.manual_seed(0)
    w = torch.randn(64, 256, dtype=torch.bfloat16)
    q, s, zp = quantize_layer_int8(w, group_size=None, strategy="channel", sym=True)

    assert q.shape == (64, 256)
    assert q.dtype == torch.int8
    assert s.shape == (64, 1)
    assert s.dtype == torch.float32
    assert zp is None

    deq = _dequant_int8_sym(q, s, in_features=256)
    err = (deq - w.to(torch.float32)).abs().max().item()
    ref = w.to(torch.float32).abs().max().item()
    # Per-channel symmetric INT8 quant error bound: ~max/127 per channel
    assert err < ref / 50, f"dequant error {err} too large vs ref {ref}"


def test_quantize_layer_int8_per_tensor_sym():
    torch.manual_seed(0)
    w = torch.randn(32, 128, dtype=torch.bfloat16)
    # "tensor" strategy emits a single global scalar scale.
    q, s, zp = quantize_layer_int8(w, group_size=None, strategy="tensor", sym=True)

    assert q.shape == (32, 128)
    assert q.dtype == torch.int8
    assert s.shape == (1,), f"per-tensor scale must be shape (1,), got {tuple(s.shape)}"
    assert s.dtype == torch.float32
    assert zp is None

    deq = q.to(torch.float32) * s.to(torch.float32)
    err = (deq - w.to(torch.float32)).abs().max().item()
    ref = w.to(torch.float32).abs().max().item()
    # Per-tensor is coarser than per-channel; bound ~max/127
    assert err < ref / 60, f"per-tensor dequant error {err} too large vs ref {ref}"


def test_quantize_layer_int8_per_tensor_asym_scalar_zp():
    torch.manual_seed(1)
    w = torch.randn(8, 64, dtype=torch.bfloat16) + 0.5
    q, s, zp = quantize_layer_int8(w, group_size=None, strategy="tensor", sym=False)

    assert q.dtype == torch.uint8
    assert s.shape == (1,)
    assert zp.shape == (1,)
    assert zp.dtype == torch.int32
    assert 0 <= int(zp.item()) <= 255


def test_quantize_layer_int8_sym_clamp_range():
    # Symmetric INT8 emits values in [-127, 127] (not [-128, 127]) since scale = absmax/127.
    torch.manual_seed(2)
    w = torch.randn(4, 32, dtype=torch.bfloat16) * 3.0
    q, _, _ = quantize_layer_int8(w, group_size=None, strategy="channel", sym=True)
    assert int(q.min()) >= -127, f"sym INT8 must not emit -128, got min={int(q.min())}"
    assert int(q.max()) <= 127


def test_quantize_layer_int8_handles_noncontiguous_input():
    # Megatron weight slices can be non-contiguous; .view() would crash, .reshape() copes.
    torch.manual_seed(3)
    full = torch.randn(16, 256, dtype=torch.bfloat16)
    w = full[::2, :]  # strided slice -> non-contiguous
    assert not w.is_contiguous()
    q, s, _ = quantize_layer_int8(w, group_size=None, strategy="channel", sym=True)
    assert q.shape == (8, 256)
    assert q.is_contiguous()
    assert s.shape == (8, 1)


def test_quantize_layer_int8_fp16_input():
    torch.manual_seed(4)
    w = torch.randn(8, 64, dtype=torch.float16)
    q, s, _ = quantize_layer_int8(w, group_size=None, strategy="channel", sym=True)
    assert q.dtype == torch.int8
    assert s.dtype == torch.float32


def test_quantize_params_w8a8_int8_auto_format_for_int8():
    # When format is omitted and num_bits=8, the dispatch must default to int-quantized
    # (raw int8 .weight) regardless of whether input_activations is set.
    cfg = _make_w8a8_int8_config(strategy="channel", symmetric=True)
    del cfg["format"]
    cfg["config_groups"]["group_0"]["input_activations"] = None  # weight-only-ish config
    params = [("model.layers.0.mlp.gate_proj.weight", torch.randn(8, 32, dtype=torch.bfloat16))]
    out = dict(quantize_params_compressed_tensors(params, cfg))
    assert out["model.layers.0.mlp.gate_proj.weight"].dtype == torch.int8
    assert out["model.layers.0.mlp.gate_proj.weight_scale"].shape == (8, 1)


def test_quantize_layer_int8_per_group_sym():
    torch.manual_seed(0)
    w = torch.randn(16, 512, dtype=torch.bfloat16)
    q, s, zp = quantize_layer_int8(w, group_size=128, strategy="group", sym=True)

    assert q.shape == (16, 512)
    assert q.dtype == torch.int8
    assert s.shape == (16, 4), f"per-group scale must be (out, in/group_size), got {tuple(s.shape)}"
    assert s.dtype == torch.float32
    assert zp is None

    deq = _dequant_int8_sym(q, s, in_features=512)
    err = (deq - w.to(torch.float32)).abs().max().item()
    ref = w.to(torch.float32).abs().max().item()
    # Per-group sym is at least as accurate as per-channel
    assert err < ref / 50


def test_quantize_layer_int8_per_channel_asym():
    torch.manual_seed(1)
    # Shift to make distribution asymmetric so zp matters
    w = torch.randn(8, 256, dtype=torch.bfloat16) + 2.0
    q, s, zp = quantize_layer_int8(w, group_size=None, strategy="channel", sym=False)

    assert q.shape == (8, 256)
    assert q.dtype == torch.uint8
    assert s.shape == (8, 1)
    assert zp is not None
    assert zp.shape == (8, 1)
    assert zp.dtype == torch.int32
    # Zero point must lie in [0, 255]
    assert int(zp.min()) >= 0
    assert int(zp.max()) <= 255

    deq = _dequant_int8_asym(q, s, zp, in_features=256)
    err = (deq - w.to(torch.float32)).abs().max().item()
    ref = w.to(torch.float32).abs().max().item()
    assert err < ref / 50


def test_quantize_layer_int8_rejects_unaligned_group():
    w = torch.randn(8, 250, dtype=torch.bfloat16)  # 250 not divisible by 128
    with pytest.raises(ValueError, match="not divisible by group_size"):
        quantize_layer_int8(w, group_size=128, strategy="group", sym=True)


def test_quantize_params_w8a8_int8_per_channel():
    torch.manual_seed(0)
    cfg = _make_w8a8_int8_config(strategy="channel", symmetric=True)
    params = [
        ("model.layers.0.mlp.gate_proj.weight", torch.randn(64, 128, dtype=torch.bfloat16)),
        ("model.layers.0.input_layernorm.weight", torch.randn(128, dtype=torch.bfloat16)),  # 1D, skip
    ]
    out = quantize_params_compressed_tensors(params, cfg)
    names = [n for n, _ in out]

    # Linear weight is replaced by .weight (int8) + .weight_scale; LayerNorm passes through
    assert "model.layers.0.mlp.gate_proj.weight" in names
    assert "model.layers.0.mlp.gate_proj.weight_scale" in names
    assert "model.layers.0.mlp.gate_proj.weight_packed" not in names, "must use int-quantized .weight naming"
    assert "model.layers.0.mlp.gate_proj.weight_shape" not in names, "no weight_shape in int-quantized layout"
    assert "model.layers.0.input_layernorm.weight" in names

    by_name = dict(out)
    qw = by_name["model.layers.0.mlp.gate_proj.weight"]
    s = by_name["model.layers.0.mlp.gate_proj.weight_scale"]
    assert qw.dtype == torch.int8
    assert qw.shape == (64, 128)
    assert s.shape == (64, 1)
    assert s.dtype == torch.float32


def test_quantize_params_w8a8_int8_ignore_lm_head_and_visual():
    cfg = _make_w8a8_int8_config(
        strategy="channel",
        symmetric=True,
        ignore=["lm_head", "re:.*visual.*"],
    )
    w = torch.randn(8, 32, dtype=torch.bfloat16)
    params = [
        ("lm_head.weight", w),
        ("model.visual.merger.mlp.0.weight", w),
        ("model.layers.0.mlp.gate_proj.weight", w),
    ]
    out = quantize_params_compressed_tensors(params, cfg)
    out_names = [n for n, _ in out]

    # Ignored Linear layers should pass through as raw .weight (still bf16)
    assert "lm_head.weight" in out_names
    assert "lm_head.weight_scale" not in out_names
    assert "model.visual.merger.mlp.0.weight" in out_names
    assert "model.visual.merger.mlp.0.weight_scale" not in out_names
    assert dict(out)["lm_head.weight"].dtype == torch.bfloat16
    # Non-ignored Linear gets quantized
    assert "model.layers.0.mlp.gate_proj.weight_scale" in out_names
    assert dict(out)["model.layers.0.mlp.gate_proj.weight"].dtype == torch.int8


def test_quantize_params_rejects_w8a8_int8_per_group():
    # W8A8 (input_activations set) + per-group is not loadable by sglang's
    # compressed_tensors_w8a8_int8 scheme — must hard-fail rather than emit a
    # silently-unloadable checkpoint.
    cfg = _make_w8a8_int8_config(strategy="group", symmetric=True, group_size=64)
    params = [("model.layers.0.mlp.up_proj.weight", torch.randn(16, 256, dtype=torch.bfloat16))]
    with pytest.raises(NotImplementedError, match="per-group weight strategy"):
        quantize_params_compressed_tensors(params, cfg)


def test_quantize_params_allows_int8_weight_only_per_group():
    # Without input_activations the config is a weight-only flavor and per-group
    # int-quantized output is the expected layout for compatible loaders.
    cfg = _make_w8a8_int8_config(strategy="group", symmetric=True, group_size=64)
    cfg["config_groups"]["group_0"]["input_activations"] = None
    params = [("model.layers.0.mlp.up_proj.weight", torch.randn(16, 256, dtype=torch.bfloat16))]
    out = dict(quantize_params_compressed_tensors(params, cfg))

    assert out["model.layers.0.mlp.up_proj.weight"].dtype == torch.int8
    assert out["model.layers.0.mlp.up_proj.weight_scale"].shape == (16, 256 // 64)


def test_quantize_params_w8a8_int8_asym_emits_zero_point():
    cfg = _make_w8a8_int8_config(strategy="channel", symmetric=False)
    params = [("model.layers.0.mlp.gate_proj.weight", torch.randn(8, 32, dtype=torch.bfloat16) + 1.0)]
    out = dict(quantize_params_compressed_tensors(params, cfg))

    assert out["model.layers.0.mlp.gate_proj.weight"].dtype == torch.uint8
    assert "model.layers.0.mlp.gate_proj.weight_zero_point" in out
    assert out["model.layers.0.mlp.gate_proj.weight_zero_point"].dtype == torch.int32


def test_quantize_params_rejects_int8_pack_quantized():
    # The INT4 packed kernel is INT8-incapable; raise a clear error instead of silently mis-quantizing.
    cfg = _make_w8a8_int8_config(strategy="group", symmetric=True, group_size=64)
    cfg["format"] = "pack-quantized"
    cfg["config_groups"]["group_0"]["input_activations"] = None
    params = [("model.layers.0.mlp.gate_proj.weight", torch.randn(8, 128, dtype=torch.bfloat16))]
    with pytest.raises(NotImplementedError, match="pack-quantized num_bits=8"):
        quantize_params_compressed_tensors(params, cfg)
