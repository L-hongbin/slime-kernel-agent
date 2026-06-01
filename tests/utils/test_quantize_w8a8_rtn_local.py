import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from scripts.quantize.producers.rtn_w8a8 import build_quantization_config, quantize_checkpoint, should_quantize_weight
from scripts.quantize.utils.mtp_checkpoint import (
    MTP_SHARD_NAME,
    inject_mtp_tensors,
    inject_rotated_mtp_tensors,
    rotate_qwen35_mtp_tensors,
)
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


def test_quantize_checkpoint_accepts_string_paths_and_writes_valid_w8a8_checkpoint(tmp_path):
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

    quantized_count = quantize_checkpoint(str(src), str(dst), target="all-linear")

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


def _sglang_is_ignored(name, ignore):
    """Mirror sglang's compressed-tensors ignore matching.

    See srt/layers/quantization/compressed_tensors/utils.py
    `_is_equal_or_regex_match`: `re:`-prefixed targets use re.match (anchored at
    the START, NOT fullmatch/search); plain strings use exact equality.
    """
    import re

    for target in ignore:
        if target.startswith("re:"):
            if re.match(target[3:], name):
                return True
        elif target == name:
            return True
    return False


def test_mtp_draft_head_ignored_under_sglang_match_semantics():
    """Regression for the EAGLE-on-W8A8 accept-rate collapse.

    sglang builds the MTP/EAGLE draft model with prefix "mtp"
    (Qwen3_5ForCausalLMMTP), so its quant-aware Linear modules are named
    `mtp.layers.0.*` -- "mtp" at the START with no leading dot. sglang matches
    `ignore` with re.match (anchored), so the old pattern `re:.*\\.mtp\\..*`
    (which requires a literal dot before "mtp") NEVER matched the draft modules.
    With targets=["Linear"] they were then treated as W8A8 quant targets, but the
    checkpoint keeps mtp weights in BF16 with no weight_scale -> garbage draft
    logits and ~0.10 EAGLE accept rate (vs ~0.7-0.99 in BF16). The fix uses
    `re:.*mtp\\..*` (no required leading dot).

    The draft head must be ignored for non-MTP target scopes; the target-model
    body must stay quantized except where the scope intentionally excludes it.
    """
    draft_modules = [
        "mtp.fc",
        "mtp.layers.0.self_attn.qkv_proj",
        "mtp.layers.0.self_attn.o_proj",
        "mtp.layers.0.mlp.gate_up_proj",
        "mtp.layers.0.mlp.down_proj",
        "mtp.layers.0.linear_attn.in_proj_qkvz",
    ]
    body_modules = [
        "model.language_model.layers.0.mlp.gate_up_proj",
        "model.language_model.layers.0.self_attn.qkv_proj",
        "model.language_model.layers.5.linear_attn.in_proj_qkvz",
    ]
    for target in ("all-linear", "mlp", "non_linear_attn"):
        ignore = build_quantization_config(target)["ignore"]
        for name in draft_modules:
            assert _sglang_is_ignored(name, ignore), f"[{target}] draft not ignored: {name}"
        for name in body_modules:
            scope_excluded = (target == "mlp" and (".self_attn." in name or ".linear_attn." in name)) or (
                target == "non_linear_attn" and ".linear_attn." in name
            )
            assert (
                _sglang_is_ignored(name, ignore) == scope_excluded
            ), f"[{target}] body {name}: ignored != scope_excluded ({scope_excluded})"


def _sglang_is_quantized(name, cfg):
    """Mirror sglang's compressed-tensors decision: quantized iff a `targets`
    entry matches AND no `ignore` entry matches (both via re.match semantics)."""
    import re

    def match(pat, n):
        if pat.startswith("re:"):
            return re.match(pat[3:], n) is not None
        if pat == "Linear":
            return True
        return pat == n

    ignore = cfg["ignore"]
    targets = cfg["config_groups"]["group_0"]["targets"]
    if any(match(t, name) for t in ignore):
        return False
    return any(match(t, name) for t in targets)


def test_non_linear_attn_mtp_quantizes_draft_head_mlp_and_self_attn():
    """non_linear_attn_mtp must quantize the MTP draft head's mlp + self_attn.

    Counterpart to the BF16-draft targets: here the draft head is intentionally
    W8A8-quantized (the producer writes weight_scale for mtp.layers.* mlp/
    self_attn), so its live sglang module names `mtp.layers.0.{self_attn,mlp}.*`
    must be quant targets and NOT in ignore. mtp.fc (the [embed;hidden] proj,
    neither mlp nor self_attn) and mamba linear_attn stay BF16.
    """
    cfg = build_quantization_config("non_linear_attn_mtp")

    quantized = [
        "mtp.layers.0.self_attn.qkv_proj",
        "mtp.layers.0.self_attn.o_proj",
        "mtp.layers.0.mlp.gate_up_proj",
        "mtp.layers.0.mlp.down_proj",
        "model.language_model.layers.0.self_attn.qkv_proj",
        "model.language_model.layers.0.mlp.gate_up_proj",
    ]
    not_quantized = [
        "model.language_model.layers.5.linear_attn.in_proj_qkvz",
        "model.language_model.layers.5.linear_attn.out_proj",
        "mtp.fc",
        "lm_head",
    ]
    for name in quantized:
        assert _sglang_is_quantized(name, cfg), f"should be quantized: {name}"
    for name in not_quantized:
        assert not _sglang_is_quantized(name, cfg), f"should stay BF16: {name}"


def test_should_quantize_weight_non_linear_attn_mtp_scope():
    """Producer writes int8+scale for body+MTP mlp/self_attn, nothing else."""
    q = [
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.layers.0.self_attn.o_proj.weight",
        "mtp.layers.0.mlp.gate_proj.weight",
        "mtp.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight",
    ]
    not_q = [
        "mtp.fc.weight",
        "mtp.norm.weight",
        "model.language_model.layers.5.linear_attn.in_proj_qkvz.weight",
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
    ]
    for name in q:
        assert should_quantize_weight(name, (8, 16), "non_linear_attn_mtp"), name
    for name in not_q:
        shape = (16,) if "norm" in name else (8, 16)
        assert not should_quantize_weight(name, shape, "non_linear_attn_mtp"), name


def test_mlp_mtp_quantizes_only_body_and_draft_head_mlp():
    """mlp_mtp is the MTP-aware counterpart of mlp-only quantization."""
    cfg = build_quantization_config("mlp_mtp")

    quantized = [
        "mtp.layers.0.mlp.gate_up_proj",
        "mtp.layers.0.mlp.down_proj",
        "model.language_model.layers.0.mlp.gate_up_proj",
        "model.language_model.layers.0.mlp.down_proj",
    ]
    not_quantized = [
        "mtp.layers.0.self_attn.qkv_proj",
        "mtp.layers.0.self_attn.o_proj",
        "mtp.fc",
        "model.language_model.layers.0.self_attn.qkv_proj",
        "model.language_model.layers.5.linear_attn.in_proj_qkvz",
        "lm_head",
    ]
    for name in quantized:
        assert _sglang_is_quantized(name, cfg), f"should be quantized: {name}"
    for name in not_quantized:
        assert not _sglang_is_quantized(name, cfg), f"should stay BF16: {name}"

    q = [
        "mtp.layers.0.mlp.gate_proj.weight",
        "mtp.layers.0.mlp.down_proj.weight",
        "model.language_model.layers.0.mlp.gate_proj.weight",
    ]
    not_q = [
        "mtp.layers.0.self_attn.q_proj.weight",
        "mtp.fc.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.5.linear_attn.in_proj_qkvz.weight",
        "lm_head.weight",
    ]
    for name in q:
        assert should_quantize_weight(name, (8, 16), "mlp_mtp"), name
    for name in not_q:
        assert not should_quantize_weight(name, (8, 16), "mlp_mtp"), name


def test_injected_mtp_tensors_are_quantized_by_mtp_targets(tmp_path):
    torch.manual_seed(0)
    src = tmp_path / "src"
    smoothed = tmp_path / "smoothed"
    dst = tmp_path / "dst"
    src.mkdir()
    smoothed.mkdir()
    for root in (src, smoothed):
        (root / "config.json").write_text(json.dumps({"architectures": ["TinyForCausalLM"]}))

    source_tensors = {
        "model.language_model.layers.0.mlp.gate_proj.weight": torch.randn(8, 16, dtype=torch.bfloat16),
        "mtp.fc.weight": torch.randn(8, 16, dtype=torch.bfloat16),
        "mtp.lm_head.weight": torch.randn(32, 16, dtype=torch.bfloat16),
        "mtp.layers.0.mlp.gate_proj.weight": torch.randn(8, 16, dtype=torch.bfloat16),
        "mtp.layers.0.self_attn.q_proj.weight": torch.randn(8, 16, dtype=torch.bfloat16),
    }
    smoothed_tensors = {
        "model.language_model.layers.0.mlp.gate_proj.weight": source_tensors[
            "model.language_model.layers.0.mlp.gate_proj.weight"
        ],
    }
    save_file(source_tensors, src / "model.safetensors")
    save_file(smoothed_tensors, smoothed / "model.safetensors")

    assert inject_mtp_tensors(smoothed, src) == 4

    with safe_open(smoothed / "model-mtp.safetensors", framework="pt", device="cpu") as f:
        assert f.get_tensor("mtp.lm_head.weight").dtype == torch.bfloat16
        assert f.get_tensor("mtp.layers.0.mlp.gate_proj.weight").dtype == torch.bfloat16
        assert f.get_tensor("mtp.layers.0.self_attn.q_proj.weight").dtype == torch.bfloat16
        assert f.get_tensor("mtp.fc.weight").dtype == torch.bfloat16

    quantized_count = quantize_checkpoint(smoothed, dst, target="non_linear_attn_mtp")
    assert quantized_count == 3

    with safe_open(dst / "model-mtp.safetensors", framework="pt", device="cpu") as f:
        assert f.get_tensor("mtp.layers.0.mlp.gate_proj.weight").dtype == torch.int8
        assert f.get_tensor("mtp.layers.0.mlp.gate_proj.weight_scale").dtype == torch.float32
        assert f.get_tensor("mtp.layers.0.self_attn.q_proj.weight").dtype == torch.int8
        assert f.get_tensor("mtp.layers.0.self_attn.q_proj.weight_scale").dtype == torch.float32
        assert f.get_tensor("mtp.lm_head.weight").dtype == torch.bfloat16
        assert f.get_tensor("mtp.fc.weight").dtype == torch.bfloat16
        assert "mtp.lm_head.weight_scale" not in f.keys()
        assert "mtp.fc.weight_scale" not in f.keys()


def test_rotate_qwen35_mtp_tensors_fuses_norms_with_identity_transform():
    h = 4
    tensors = {
        "mtp.fc.weight": torch.ones(h, 2 * h, dtype=torch.float32),
        "mtp.pre_fc_norm_embedding.weight": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "mtp.pre_fc_norm_hidden.weight": torch.tensor([5.0, 6.0, 7.0, 8.0]),
        "mtp.layers.0.input_layernorm.weight": torch.tensor([1.0, 2.0, 3.0, 4.0]),
        "mtp.layers.0.self_attn.q_proj.weight": torch.ones(h, h),
        "mtp.layers.0.self_attn.k_proj.weight": torch.ones(h, h) * 2,
        "mtp.layers.0.self_attn.v_proj.weight": torch.ones(h, h) * 3,
        "mtp.layers.0.self_attn.o_proj.weight": torch.ones(h, h),
        "mtp.layers.0.post_attention_layernorm.weight": torch.tensor([2.0, 3.0, 4.0, 5.0]),
        "mtp.layers.0.mlp.gate_proj.weight": torch.ones(h, h),
        "mtp.layers.0.mlp.up_proj.weight": torch.ones(h, h) * 2,
        "mtp.layers.0.mlp.down_proj.weight": torch.ones(h, h),
        "mtp.norm.weight": torch.tensor([2.0, 4.0, 6.0, 8.0]),
    }

    rotated = rotate_qwen35_mtp_tensors(
        tensors,
        transform=torch.eye(h, dtype=torch.float64),
        body_final_norm=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        final_norm_mode="ratio",
        device="cpu",
    )

    assert torch.allclose(
        rotated["mtp.fc.weight"][:, :h], tensors["mtp.fc.weight"][:, :h] * torch.arange(2, 6).float()
    )
    assert torch.allclose(
        rotated["mtp.fc.weight"][:, h:], tensors["mtp.fc.weight"][:, h:] * torch.arange(6, 10).float()
    )
    assert torch.allclose(rotated["mtp.pre_fc_norm_embedding.weight"], torch.zeros(h))
    assert torch.allclose(rotated["mtp.pre_fc_norm_hidden.weight"], torch.zeros(h))
    assert torch.allclose(
        rotated["mtp.layers.0.self_attn.q_proj.weight"],
        tensors["mtp.layers.0.self_attn.q_proj.weight"] * torch.arange(2, 6).float(),
    )
    assert torch.allclose(
        rotated["mtp.layers.0.mlp.gate_proj.weight"],
        tensors["mtp.layers.0.mlp.gate_proj.weight"] * torch.tensor([3.0, 4.0, 5.0, 6.0]),
    )
    assert torch.allclose(rotated["mtp.layers.0.input_layernorm.weight"], torch.zeros(h))
    assert torch.allclose(rotated["mtp.layers.0.post_attention_layernorm.weight"], torch.zeros(h))
    assert torch.allclose(rotated["mtp.norm.weight"], torch.tensor([0.5, 2.0 / 3.0, 0.75, 0.8]))


def _tiny_mtp_tensors(h=4):
    torch.manual_seed(7)
    return {
        "mtp.fc.weight": torch.randn(h, 2 * h, dtype=torch.float64),
        "mtp.pre_fc_norm_embedding.weight": torch.tensor([1.0, 1.5, 2.0, 2.5], dtype=torch.float64),
        "mtp.pre_fc_norm_hidden.weight": torch.tensor([0.5, 0.75, 1.25, 1.75], dtype=torch.float64),
        "mtp.layers.0.input_layernorm.weight": torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64),
        "mtp.layers.0.self_attn.q_proj.weight": torch.randn(h, h, dtype=torch.float64),
        "mtp.layers.0.self_attn.k_proj.weight": torch.randn(h, h, dtype=torch.float64),
        "mtp.layers.0.self_attn.v_proj.weight": torch.randn(h, h, dtype=torch.float64),
        "mtp.layers.0.self_attn.o_proj.weight": torch.randn(h, h, dtype=torch.float64),
        "mtp.layers.0.post_attention_layernorm.weight": torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float64),
        "mtp.layers.0.mlp.gate_proj.weight": torch.randn(h, h, dtype=torch.float64),
        "mtp.layers.0.mlp.up_proj.weight": torch.randn(h, h, dtype=torch.float64),
        "mtp.layers.0.mlp.down_proj.weight": torch.randn(h, h, dtype=torch.float64),
        "mtp.norm.weight": torch.tensor([0.75, 1.25, 1.5, 2.0], dtype=torch.float64),
    }


def _hadamard4():
    return (
        torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0, -1.0],
                [1.0, 1.0, -1.0, -1.0],
                [1.0, -1.0, -1.0, 1.0],
            ],
            dtype=torch.float64,
        )
        / 2.0
    )


def test_rotate_qwen35_mtp_tensors_emits_exact_separate_lm_head():
    h = 4
    tensors = _tiny_mtp_tensors(h)
    lm_head = torch.arange(1, 1 + 3 * h, dtype=torch.float64).view(3, h)

    rotated = rotate_qwen35_mtp_tensors(
        tensors,
        transform=torch.eye(h, dtype=torch.float64),
        lm_head_weight=lm_head,
        final_norm_mode="separate_lm_head",
        device="cpu",
    )

    assert torch.allclose(rotated["mtp.norm.weight"], torch.zeros(h, dtype=torch.float64))
    assert torch.allclose(rotated["mtp.lm_head.weight"], lm_head * (1 + tensors["mtp.norm.weight"]).view(1, -1))


def test_rotated_mtp_fc_preserves_residual_basis_with_nontrivial_transform():
    h = 4
    tensors = _tiny_mtp_tensors(h)
    transform = _hadamard4()

    rotated = rotate_qwen35_mtp_tensors(
        tensors,
        transform=transform,
        final_norm_mode="ones",
        device="cpu",
    )

    embeds = torch.randn(6, h, dtype=torch.float64)
    hidden = torch.randn(6, h, dtype=torch.float64)
    original_in = torch.cat(
        [
            embeds * (1 + tensors["mtp.pre_fc_norm_embedding.weight"]).view(1, -1),
            hidden * (1 + tensors["mtp.pre_fc_norm_hidden.weight"]).view(1, -1),
        ],
        dim=-1,
    )
    original_out = original_in @ tensors["mtp.fc.weight"].T
    rotated_in = torch.cat([embeds @ transform, hidden @ transform], dim=-1)
    rotated_out = rotated_in @ rotated["mtp.fc.weight"].T

    assert torch.allclose(rotated_out, original_out @ transform, rtol=1e-8, atol=1e-8)


def test_rotated_mtp_separate_lm_head_preserves_logits():
    h = 4
    tensors = _tiny_mtp_tensors(h)
    transform = _hadamard4()
    lm_head = torch.randn(5, h, dtype=torch.float64)
    hidden = torch.randn(7, h, dtype=torch.float64)

    rotated = rotate_qwen35_mtp_tensors(
        tensors,
        transform=transform,
        lm_head_weight=lm_head,
        final_norm_mode="separate_lm_head",
        device="cpu",
    )

    original_logits = (hidden * (1 + tensors["mtp.norm.weight"]).view(1, -1)) @ lm_head.T
    rotated_logits = (hidden @ transform) @ rotated["mtp.lm_head.weight"].T

    assert torch.allclose(rotated_logits, original_logits, rtol=1e-8, atol=1e-8)


def test_inject_rotated_mtp_tensors_writes_separate_lm_head_to_index(tmp_path):
    h = 4
    src = tmp_path / "src"
    dst = tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (src / "config.json").write_text(json.dumps({"architectures": ["TinyForCausalLM"]}))
    (dst / "config.json").write_text(json.dumps({"architectures": ["TinyForCausalLM"]}))
    source_tensors = _tiny_mtp_tensors(h)
    source_tensors["lm_head.weight"] = torch.randn(8, h, dtype=torch.float64)
    save_file(source_tensors, src / "model.safetensors")
    save_file({"lm_head.weight": torch.randn(8, h, dtype=torch.float64)}, dst / "model.safetensors")

    injected = inject_rotated_mtp_tensors(
        dst,
        src,
        transform=torch.eye(h, dtype=torch.float64),
        final_norm_mode="separate_lm_head",
    )

    assert injected == len(_tiny_mtp_tensors(h)) + 1
    index = json.loads((dst / "model.safetensors.index.json").read_text())
    assert index["weight_map"]["mtp.lm_head.weight"] == "model-mtp.safetensors"
    with safe_open(dst / "model-mtp.safetensors", framework="pt", device="cpu") as f:
        assert "mtp.lm_head.weight" in f.keys()
        assert torch.allclose(f.get_tensor("mtp.norm.weight"), torch.zeros(h, dtype=torch.float64))


def test_qwen35_gemma_norm_prepare_and_zero_helpers_use_offset_identity():
    pytest.importorskip("llmcompressor")
    from scripts.quantize.rotation.rotate_bf16 import (
        prepare_qwen35_gemma_norms_for_fusion,
        zero_qwen35_fused_gemma_norms,
    )
    from torch import nn

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.language_model = nn.Module()
            self.model.language_model.norm = nn.Module()
            self.model.language_model.norm.weight = nn.Parameter(torch.tensor([-0.25, 0.0, 0.5, 1.0]))

    mapping = type("_Mapping", (), {"norm": "model.language_model.norm"})()
    model = TinyModel()

    norm_paths = prepare_qwen35_gemma_norms_for_fusion(model, [mapping])

    assert norm_paths == ["model.language_model.norm"]
    assert torch.allclose(model.model.language_model.norm.weight, torch.tensor([0.75, 1.0, 1.5, 2.0]))

    zero_qwen35_fused_gemma_norms(model, norm_paths)

    assert torch.allclose(model.model.language_model.norm.weight, torch.zeros(4))


def test_sglang_mtp_separate_lm_head_patch_keeps_draft_head():
    patch = Path("scripts/quantize/patches/sglang_qwen3_5_mtp_separate_lm_head.patch").read_text()

    assert 'name == "mtp.lm_head.weight"' in patch
    assert "self._has_separate_mtp_lm_head = True" in patch
    assert "Keeping checkpoint-provided separate MTP lm_head" in patch


def test_check_mtp_rotation_accepts_quantized_mtp_weights(tmp_path):
    pytest.importorskip("compressed_tensors")
    h = 4
    src = tmp_path / "src"
    rotated = tmp_path / "rotated"
    quantized = tmp_path / "quantized"
    src.mkdir()
    rotated.mkdir()
    (src / "config.json").write_text(json.dumps({"architectures": ["TinyForCausalLM"]}))
    (rotated / "config.json").write_text(json.dumps({"architectures": ["TinyForCausalLM"]}))
    source_tensors = _tiny_mtp_tensors(h)
    source_tensors["lm_head.weight"] = torch.randn(8, h, dtype=torch.float64)
    source_tensors["model.language_model.embed_tokens.weight"] = torch.randn(8, h, dtype=torch.float64)
    source_tensors["model.language_model.norm.weight"] = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    save_file(source_tensors, src / "model.safetensors")
    transform = _hadamard4()
    rotated_tensors = {
        "model.language_model.embed_tokens.weight": source_tensors["model.language_model.embed_tokens.weight"]
        @ transform,
        "lm_head.weight": (
            source_tensors["lm_head.weight"] * (1 + source_tensors["model.language_model.norm.weight"]).view(1, -1)
        )
        @ transform,
        "model.language_model.norm.weight": torch.zeros(h, dtype=torch.float64),
    }
    save_file(rotated_tensors, rotated / "model.safetensors")
    inject_rotated_mtp_tensors(
        rotated,
        src,
        transform=transform,
        final_norm_mode="separate_lm_head",
    )
    quantize_checkpoint(rotated, quantized, target="non_linear_attn_mtp")

    subprocess.run(
        [
            sys.executable,
            "scripts/quantize/rotation/check_mtp_rotation.py",
            "--checkpoint",
            str(quantized),
            "--source-model",
            str(src),
            "--hidden-size",
            str(h),
            "--transform-type",
            "hadamard",
            "--head-sample-rows",
            "2",
        ],
        check=True,
    )


def test_check_mtp_rotation_rejects_gemma_identity_left_as_one(tmp_path):
    pytest.importorskip("compressed_tensors")
    h = 4
    src = tmp_path / "src"
    bad = tmp_path / "bad"
    src.mkdir()
    bad.mkdir()
    (src / "config.json").write_text(json.dumps({"architectures": ["TinyForCausalLM"]}))
    (bad / "config.json").write_text(json.dumps({"architectures": ["TinyForCausalLM"]}))
    source_tensors = _tiny_mtp_tensors(h)
    source_tensors["lm_head.weight"] = torch.randn(8, h, dtype=torch.float64)
    source_tensors["model.language_model.embed_tokens.weight"] = torch.randn(8, h, dtype=torch.float64)
    source_tensors["model.language_model.norm.weight"] = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    save_file(source_tensors, src / "model.safetensors")
    transform = _hadamard4()

    save_file(
        {
            "model.language_model.embed_tokens.weight": source_tensors["model.language_model.embed_tokens.weight"]
            @ transform,
            "lm_head.weight": (
                source_tensors["lm_head.weight"] * (1 + source_tensors["model.language_model.norm.weight"]).view(1, -1)
            )
            @ transform,
            "model.language_model.norm.weight": torch.ones(h, dtype=torch.float64),
        },
        bad / "model.safetensors",
    )
    inject_rotated_mtp_tensors(
        bad,
        src,
        transform=transform,
        final_norm_mode="separate_lm_head",
    )
    with safe_open(bad / MTP_SHARD_NAME, framework="pt", device="cpu") as f:
        mtp_tensors = {name: f.get_tensor(name).contiguous() for name in f.keys()}
    mtp_tensors["mtp.norm.weight"] = torch.ones(h, dtype=torch.float64)
    save_file(mtp_tensors, bad / MTP_SHARD_NAME)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/quantize/rotation/check_mtp_rotation.py",
            "--checkpoint",
            str(bad),
            "--source-model",
            str(src),
            "--hidden-size",
            str(h),
            "--transform-type",
            "hadamard",
            "--head-sample-rows",
            "2",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "model.language_model.norm.weight" in output
    assert "mtp.norm.weight" in output
