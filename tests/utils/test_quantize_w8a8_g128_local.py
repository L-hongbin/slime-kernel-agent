import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.quantize.producers.rtn_w8a8_g128 import (
    build_quantization_config,
    quantize_checkpoint,
    quantize_layer_blockwise_int8,
)
from scripts.quantize.utils.validate_checkpoint import check_w8a8_checkpoint


def _sglang_blockwise_is_ignored(name, ignored_layers):
    def module_path_match(ignored, prefix):
        if ignored == prefix:
            return True
        if prefix.startswith(ignored + "."):
            return True
        return ("." + ignored + ".") in ("." + prefix + ".")

    return any(module_path_match(ignored, name) for ignored in ignored_layers)


def test_quantize_layer_blockwise_int8_writes_scale_grid():
    torch.manual_seed(0)
    w = torch.randn(20, 48, dtype=torch.bfloat16)

    q, scale = quantize_layer_blockwise_int8(w, block_size=(8, 16))

    assert q.shape == w.shape
    assert q.dtype == torch.int8
    assert scale.shape == (3, 3)
    assert scale.dtype == torch.float32
    assert int(q.min()) >= -127
    assert int(q.max()) <= 127


def test_g128_non_linear_attn_mtp_scope_matches_eagle_nonla(tmp_path):
    cfg = build_quantization_config("non_linear_attn_mtp", block_size=(128, 128))
    ignored = cfg["ignored_layers"]

    quantized = [
        "model.language_model.layers.0.mlp.gate_up_proj",
        "model.language_model.layers.0.self_attn.qkv_proj",
        "mtp.layers.0.mlp.gate_up_proj",
        "mtp.layers.0.self_attn.qkv_proj",
    ]
    not_quantized = [
        "model.language_model.layers.5.linear_attn.in_proj_qkvz",
        "mtp.layers.0.linear_attn.in_proj_qkvz",
        "mtp.fc",
        "lm_head",
    ]
    for name in quantized:
        assert not _sglang_blockwise_is_ignored(name, ignored), name
    for name in not_quantized:
        assert _sglang_blockwise_is_ignored(name, ignored), name


def test_quantize_checkpoint_writes_blockwise_g128_checkpoint(tmp_path):
    torch.manual_seed(1)
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({"architectures": ["TinyForCausalLM"]}))
    (src / "tokenizer_config.json").write_text("{}")

    tensors = {
        "model.language_model.layers.0.mlp.gate_proj.weight": torch.randn(16, 32, dtype=torch.bfloat16),
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(16, 32, dtype=torch.bfloat16),
        "model.language_model.layers.0.linear_attn.in_proj_qkvz.weight": torch.randn(16, 32, dtype=torch.bfloat16),
        "mtp.layers.0.mlp.gate_proj.weight": torch.randn(16, 32, dtype=torch.bfloat16),
        "mtp.layers.0.self_attn.q_proj.weight": torch.randn(16, 32, dtype=torch.bfloat16),
        "mtp.fc.weight": torch.randn(16, 32, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(16, 32, dtype=torch.bfloat16),
    }
    save_file(tensors, src / "model.safetensors")

    quantized_count = quantize_checkpoint(src, dst, target="non_linear_attn_mtp", block_size=(8, 16))

    assert quantized_count == 4
    checks = check_w8a8_checkpoint(dst, reference_checkpoint=src)
    assert len(checks) == 4
    assert max(check.rel_l2 for check in checks if check.rel_l2 is not None) < 0.08

    with safe_open(dst / "model.safetensors", framework="pt", device="cpu") as f:
        assert f.get_tensor("model.language_model.layers.0.mlp.gate_proj.weight").dtype == torch.int8
        assert f.get_tensor("model.language_model.layers.0.mlp.gate_proj.weight_scale_inv").shape == (2, 2)
        assert f.get_tensor("mtp.layers.0.self_attn.q_proj.weight").dtype == torch.int8
        assert f.get_tensor("model.language_model.layers.0.linear_attn.in_proj_qkvz.weight").dtype == torch.bfloat16
        assert f.get_tensor("mtp.fc.weight").dtype == torch.bfloat16
        assert "mtp.fc.weight_scale_inv" not in f.keys()

    config = json.loads((dst / "config.json").read_text())
    qcfg = config["quantization_config"]
    assert qcfg["quant_method"] == "blockwise_int8"
    assert qcfg["activation_scheme"] == "dynamic"
    assert qcfg["weight_block_size"] == [8, 16]
    assert "linear_attn" in qcfg["ignored_layers"]
    assert "mtp.fc" in qcfg["ignored_layers"]
