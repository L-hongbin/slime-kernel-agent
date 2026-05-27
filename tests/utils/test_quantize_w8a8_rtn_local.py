import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from scripts.quantize.producers.rtn_w8a8 import quantize_checkpoint, should_quantize_weight
from scripts.quantize.utils.validate_checkpoint import check_w8a8_checkpoint


def test_should_quantize_weight_matches_text_side_linear_scope():
    assert should_quantize_weight("model.language_model.layers.0.mlp.gate_proj.weight", (8, 16), "all-linear")
    assert should_quantize_weight("model.language_model.layers.0.self_attn.q_proj.weight", (8, 16), "all-linear")
    assert should_quantize_weight("model.language_model.layers.0.mlp.down_proj.weight", (8, 16), "mlp")
    assert not should_quantize_weight("model.language_model.layers.0.self_attn.q_proj.weight", (8, 16), "mlp")
    assert not should_quantize_weight("model.language_model.embed_tokens.weight", (100, 16), "all-linear")
    assert not should_quantize_weight("lm_head.weight", (100, 16), "all-linear")
    assert not should_quantize_weight("model.visual.blocks.0.attn.qkv.weight", (8, 16), "all-linear")
    assert not should_quantize_weight("model.language_model.layers.0.input_layernorm.weight", (16,), "all-linear")


def test_quantize_checkpoint_writes_valid_w8a8_checkpoint(tmp_path):
    torch.manual_seed(0)
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    (src / "config.json").write_text(json.dumps({"architectures": ["TinyForCausalLM"]}))
    (src / "tokenizer_config.json").write_text("{}")

    tensors = {
        "model.language_model.layers.0.mlp.gate_proj.weight": torch.randn(8, 16, dtype=torch.bfloat16),
        "model.language_model.layers.0.input_layernorm.weight": torch.randn(16, dtype=torch.bfloat16),
        "model.language_model.embed_tokens.weight": torch.randn(32, 16, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(32, 16, dtype=torch.bfloat16),
        "model.visual.blocks.0.attn.qkv.weight": torch.randn(8, 16, dtype=torch.bfloat16),
    }
    save_file(tensors, src / "model.safetensors")

    quantized_count = quantize_checkpoint(src, dst, target="all-linear")

    assert quantized_count == 1
    checks = check_w8a8_checkpoint(dst, reference_checkpoint=src)
    assert len(checks) == 1
    assert checks[0].unique_count > 3
    assert checks[0].saturated_frac < 0.20
    assert checks[0].rel_l2 is not None
    assert checks[0].rel_l2 < 0.08

    with safe_open(dst / "model.safetensors", framework="pt", device="cpu") as f:
        assert f.get_tensor("model.language_model.layers.0.mlp.gate_proj.weight").dtype == torch.int8
        assert f.get_tensor("model.language_model.layers.0.mlp.gate_proj.weight_scale").shape == (8, 1)
        assert f.get_tensor("lm_head.weight").dtype == torch.bfloat16
        assert f.get_tensor("model.language_model.embed_tokens.weight").dtype == torch.bfloat16
        assert f.get_tensor("model.visual.blocks.0.attn.qkv.weight").dtype == torch.bfloat16

    config = json.loads((dst / "config.json").read_text())
    assert config["quantization_config"]["quant_method"] == "compressed-tensors"
    assert config["quantization_config"]["format"] == "int-quantized"
