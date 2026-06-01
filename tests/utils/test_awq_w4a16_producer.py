import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.quantize.patches.llmcompressor_qwen3_5_awq import awq_smooth_weight_for_module, is_qwen3_5_offset_rmsnorm
from scripts.quantize.producers.awq_w4a16 import (
    AWQ_DEFAULT_DUO_SCALING,
    AWQ_DEFAULT_N_GRID,
    MLP_TARGET_RE,
    awq_modifier_scope,
    awq_search_includes_identity,
    awq_targets_and_ignore,
    build_awq_mappings_for_target,
    build_recipe,
    parse_awq_duo_scaling,
    patch_transformers_broken_torchvision,
    should_restore_awq_preserved_tensor,
)


def test_awq_w4a16_mlp_targets_are_sglang_fused_safe():
    targets, ignore = awq_targets_and_ignore()

    assert targets == [MLP_TARGET_RE]
    assert "gate_up_proj" in targets[0]
    assert "down_proj" in targets[0]
    assert r"re:.*\.self_attn\..*" in ignore
    assert r"re:.*\.linear_attn\..*" in ignore
    assert r"re:.*mtp\..*" in ignore
    assert "lm_head" in ignore


def test_awq_w4a16_mappings_are_body_mlp_only():
    mappings = build_awq_mappings_for_target()

    assert mappings == [
        (
            r"re:model\.language_model\.layers\.\d+\.post_attention_layernorm$",
            [
                r"re:model\.language_model\.layers\.\d+\.mlp\.gate_proj$",
                r"re:model\.language_model\.layers\.\d+\.mlp\.up_proj$",
            ],
        ),
        (
            r"re:model\.language_model\.layers\.\d+\.mlp\.up_proj$",
            [r"re:model\.language_model\.layers\.\d+\.mlp\.down_proj$"],
        ),
    ]


def test_awq_modifier_keeps_default_quantization_fields():
    assert awq_modifier_scope() == (["Linear"], [])


def test_awq_recipe_scopes_only_final_quantization_modifier():
    patch_transformers_broken_torchvision()
    pytest.importorskip("llmcompressor")
    targets, ignore = awq_targets_and_ignore()

    awq_modifier, quantization_modifier = build_recipe("W4A16", duo_scaling="both", n_grid=40)
    AWQModifier = pytest.importorskip("llmcompressor.modifiers.awq.base").AWQModifier

    assert awq_modifier.targets == ["Linear"]
    assert awq_modifier.ignore == []
    assert quantization_modifier.targets == targets
    assert quantization_modifier.ignore == ignore
    assert getattr(AWQModifier, "_slime_qwen3_5_rmsnorm_patch", False)


def test_awq_default_search_includes_no_smoothing_baseline():
    assert parse_awq_duo_scaling(AWQ_DEFAULT_DUO_SCALING) == "both"
    assert awq_search_includes_identity(parse_awq_duo_scaling(AWQ_DEFAULT_DUO_SCALING))
    assert AWQ_DEFAULT_N_GRID >= 40


def test_awq_duo_scaling_true_does_not_include_identity():
    assert parse_awq_duo_scaling("true") is True
    assert parse_awq_duo_scaling("false") is False
    assert not awq_search_includes_identity(parse_awq_duo_scaling("true"))
    assert awq_search_includes_identity(parse_awq_duo_scaling("false"))


def test_awq_preserved_tensor_predicate_allows_only_mlp_smoothing_side_effects():
    assert not should_restore_awq_preserved_tensor("model.language_model.layers.0.post_attention_layernorm.weight")
    assert not should_restore_awq_preserved_tensor("model.language_model.layers.0.mlp.gate_proj.weight")
    assert not should_restore_awq_preserved_tensor("mtp.layers.0.input_layernorm.weight")

    assert should_restore_awq_preserved_tensor("model.language_model.layers.0.input_layernorm.weight")
    assert should_restore_awq_preserved_tensor("model.language_model.layers.0.self_attn.q_norm.weight")
    assert should_restore_awq_preserved_tensor("model.language_model.layers.0.linear_attn.out_proj.weight")
    assert should_restore_awq_preserved_tensor("lm_head.weight")


def test_awq_qwen_offset_rmsnorm_smoothing_uses_effective_one_plus_weight():
    torch = pytest.importorskip("torch")

    class Qwen3_5RMSNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([0.0, 0.5, -0.25]))

    module = Qwen3_5RMSNorm()
    scales = torch.tensor([2.0, 0.5, 4.0])

    smoothed = awq_smooth_weight_for_module(module, scales)

    assert is_qwen3_5_offset_rmsnorm(module)
    torch.testing.assert_close(smoothed, (module.weight + 1.0) / scales - 1.0)


def test_awq_standard_norm_smoothing_keeps_original_formula():
    torch = pytest.importorskip("torch")

    module = torch.nn.LayerNorm(3)
    module.weight = torch.nn.Parameter(torch.tensor([2.0, 4.0, 8.0]))
    scales = torch.tensor([2.0, 0.5, 4.0])

    smoothed = awq_smooth_weight_for_module(module, scales)

    torch.testing.assert_close(smoothed, torch.tensor([1.0, 8.0, 2.0]))


def test_sgl_smoke_can_stub_broken_torchvision_import():
    old_torchvision = sys.modules.pop("torchvision", None)
    old_torchvision_io = sys.modules.pop("torchvision.io", None)
    try:
        patch_broken_torchvision_import = pytest.importorskip(
            "scripts.quantize.sgl_engine_smoke"
        ).patch_broken_torchvision_import

        patch_broken_torchvision_import(force_stub=True)

        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as import_utils
        from torchvision.io import decode_jpeg
        from torchvision.transforms import InterpolationMode

        assert not import_utils.is_torchvision_available()
        assert not transformers_utils.is_torchvision_available()
        assert InterpolationMode.BICUBIC == "bicubic"
        with pytest.raises(RuntimeError, match="text-only smoke"):
            decode_jpeg(None)
    finally:
        sys.modules.pop("torchvision", None)
        sys.modules.pop("torchvision.io", None)
        if old_torchvision is not None:
            sys.modules["torchvision"] = old_torchvision
        if old_torchvision_io is not None:
            sys.modules["torchvision.io"] = old_torchvision_io


def _write_fake_awq_checkpoint(root: Path, tensors, *, symmetric: bool = True) -> None:
    save_file = pytest.importorskip("safetensors.torch").save_file

    targets, ignore = awq_targets_and_ignore()
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "format": "pack-quantized",
                    "config_groups": {
                        "group_0": {
                            "targets": targets,
                            "weights": {
                                "num_bits": 4,
                                "type": "int",
                                "strategy": "group",
                                "group_size": 128,
                                "symmetric": symmetric,
                            },
                        }
                    },
                    "ignore": ignore,
                }
            },
            indent=2,
        )
        + "\n"
    )
    save_file(tensors, root / "model.safetensors")


def _write_fake_reference_checkpoint(root: Path, tensors) -> None:
    save_file = pytest.importorskip("safetensors.torch").save_file

    root.mkdir()
    save_file(tensors, root / "model.safetensors")


def test_check_awq_w4a16_checkpoint_accepts_mlp_only(tmp_path):
    torch = pytest.importorskip("torch")
    check_awq_w4a16_checkpoint = pytest.importorskip(
        "scripts.quantize.utils.check_awq_w4a16"
    ).check_awq_w4a16_checkpoint
    ckpt = tmp_path / "ckpt"
    _write_fake_awq_checkpoint(
        ckpt,
        {
            "model.language_model.layers.0.mlp.gate_proj.weight_packed": torch.zeros(8, 4, dtype=torch.int32),
            "model.language_model.layers.0.mlp.gate_proj.weight_scale": torch.ones(1, 8, dtype=torch.float16),
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.zeros(8, 8, dtype=torch.bfloat16),
            "mtp.layers.0.mlp.gate_proj.weight": torch.zeros(8, 8, dtype=torch.bfloat16),
            "lm_head.weight": torch.zeros(8, 8, dtype=torch.bfloat16),
        },
    )

    report = check_awq_w4a16_checkpoint(ckpt, min_quantized_mlp=1, require_mtp=True)

    assert report.quantized_mlp == 1
    assert report.quantized_total == 1
    assert report.mtp_tensors == 1


def test_check_awq_w4a16_checkpoint_rejects_quantized_mtp(tmp_path):
    torch = pytest.importorskip("torch")
    check_awq_w4a16_checkpoint = pytest.importorskip(
        "scripts.quantize.utils.check_awq_w4a16"
    ).check_awq_w4a16_checkpoint
    ckpt = tmp_path / "ckpt"
    _write_fake_awq_checkpoint(
        ckpt,
        {
            "model.language_model.layers.0.mlp.gate_proj.weight_packed": torch.zeros(8, 4, dtype=torch.int32),
            "mtp.layers.0.mlp.gate_proj.weight_packed": torch.zeros(8, 4, dtype=torch.int32),
        },
    )

    with pytest.raises(AssertionError, match="forbidden W4A16 quantized weights"):
        check_awq_w4a16_checkpoint(ckpt)


def test_check_awq_w4a16_checkpoint_can_require_symmetric_weights(tmp_path):
    torch = pytest.importorskip("torch")
    check_awq_w4a16_checkpoint = pytest.importorskip(
        "scripts.quantize.utils.check_awq_w4a16"
    ).check_awq_w4a16_checkpoint
    ckpt = tmp_path / "ckpt"
    _write_fake_awq_checkpoint(
        ckpt,
        {
            "model.language_model.layers.0.mlp.gate_proj.weight_packed": torch.zeros(8, 4, dtype=torch.int32),
        },
        symmetric=False,
    )

    with pytest.raises(AssertionError, match="asymmetric W4A16"):
        check_awq_w4a16_checkpoint(ckpt, require_symmetric=True)


def _write_fake_sglang_wna16_source(root: Path, *, pass_symmetric: bool) -> Path:
    dispatch = root / "sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py"
    scheme = root / "sglang/srt/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py"
    dispatch.parent.mkdir(parents=True)
    scheme.parent.mkdir(parents=True)
    symmetric_arg = "symmetric=weight_quant.symmetric,\n" if pass_symmetric else ""
    dispatch.write_text(
        "def _is_wNa16_group_channel(weight_quant, input_quant):\n"
        "    return is_channel_group and input_quant_none and is_static\n\n"
        "def _get_scheme_from_parts(weight_quant, input_quant):\n"
        "    return CompressedTensorsWNA16(\n"
        "        num_bits=weight_quant.num_bits,\n"
        "        strategy=weight_quant.strategy,\n"
        f"        {symmetric_arg}"
        "        group_size=weight_quant.group_size,\n"
        "        actorder=weight_quant.actorder,\n"
        "    )\n"
    )
    scheme.write_text(
        "WNA16_ZP_SUPPORTED_TYPES_MAP = {4: object()}\n"
        "if not self.symmetric:\n"
        '    layer.register_parameter("weight_zero_point", qzeros)\n'
    )
    return root


def test_check_awq_w4a16_checkpoint_requires_sglang_asym_runtime_patch(tmp_path):
    torch = pytest.importorskip("torch")
    check_awq_w4a16_checkpoint = pytest.importorskip(
        "scripts.quantize.utils.check_awq_w4a16"
    ).check_awq_w4a16_checkpoint
    ckpt = tmp_path / "ckpt"
    _write_fake_awq_checkpoint(
        ckpt,
        {
            "model.language_model.layers.0.mlp.gate_proj.weight_packed": torch.zeros(8, 4, dtype=torch.int32),
            "model.language_model.layers.0.mlp.gate_proj.weight_zero_point": torch.zeros(1, 4, dtype=torch.int32),
        },
        symmetric=False,
    )

    bad_sglang = _write_fake_sglang_wna16_source(tmp_path / "bad_sglang", pass_symmetric=False)
    with pytest.raises(AssertionError, match="does not pass weight_quant.symmetric"):
        check_awq_w4a16_checkpoint(
            ckpt,
            require_sglang_asym_support=True,
            sglang_source_root=bad_sglang,
        )

    good_sglang = _write_fake_sglang_wna16_source(tmp_path / "good_sglang", pass_symmetric=True)
    report = check_awq_w4a16_checkpoint(
        ckpt,
        require_sglang_asym_support=True,
        sglang_source_root=good_sglang,
    )

    assert report.quantized_mlp == 1


def test_check_awq_w4a16_checkpoint_rejects_non_target_drift(tmp_path):
    torch = pytest.importorskip("torch")
    check_awq_w4a16_checkpoint = pytest.importorskip(
        "scripts.quantize.utils.check_awq_w4a16"
    ).check_awq_w4a16_checkpoint
    ref = tmp_path / "ref"
    ckpt = tmp_path / "ckpt"
    _write_fake_reference_checkpoint(
        ref,
        {
            "model.language_model.layers.0.input_layernorm.weight": torch.ones(8, dtype=torch.bfloat16),
            "model.language_model.layers.0.post_attention_layernorm.weight": torch.ones(8, dtype=torch.bfloat16),
        },
    )
    _write_fake_awq_checkpoint(
        ckpt,
        {
            "model.language_model.layers.0.mlp.gate_proj.weight_packed": torch.zeros(8, 4, dtype=torch.int32),
            "model.language_model.layers.0.input_layernorm.weight": torch.full((8,), 2.0, dtype=torch.bfloat16),
            "model.language_model.layers.0.post_attention_layernorm.weight": torch.full(
                (8,), 2.0, dtype=torch.bfloat16
            ),
        },
    )

    with pytest.raises(AssertionError, match="non-target BF16 tensors drifted"):
        check_awq_w4a16_checkpoint(ckpt, reference_checkpoint=ref)


def test_check_awq_w4a16_checkpoint_allows_post_attention_norm_drift(tmp_path):
    torch = pytest.importorskip("torch")
    check_awq_w4a16_checkpoint = pytest.importorskip(
        "scripts.quantize.utils.check_awq_w4a16"
    ).check_awq_w4a16_checkpoint
    ref = tmp_path / "ref"
    ckpt = tmp_path / "ckpt"
    _write_fake_reference_checkpoint(
        ref,
        {
            "model.language_model.layers.0.input_layernorm.weight": torch.ones(8, dtype=torch.bfloat16),
            "model.language_model.layers.0.post_attention_layernorm.weight": torch.ones(8, dtype=torch.bfloat16),
        },
    )
    _write_fake_awq_checkpoint(
        ckpt,
        {
            "model.language_model.layers.0.mlp.gate_proj.weight_packed": torch.zeros(8, 4, dtype=torch.int32),
            "model.language_model.layers.0.input_layernorm.weight": torch.ones(8, dtype=torch.bfloat16),
            "model.language_model.layers.0.post_attention_layernorm.weight": torch.full(
                (8,), 2.0, dtype=torch.bfloat16
            ),
        },
    )

    report = check_awq_w4a16_checkpoint(ckpt, reference_checkpoint=ref)

    assert report.preserved_checked == 1


def test_check_awq_w4a16_checkpoint_samples_large_preserved_tensors(tmp_path):
    torch = pytest.importorskip("torch")
    check_awq_w4a16_checkpoint = pytest.importorskip(
        "scripts.quantize.utils.check_awq_w4a16"
    ).check_awq_w4a16_checkpoint
    ref = tmp_path / "ref"
    ckpt = tmp_path / "ckpt"
    ref_lm_head = torch.zeros(4, 4, dtype=torch.bfloat16)
    drifted_lm_head = ref_lm_head.clone()
    drifted_lm_head[-1, :] = 1
    _write_fake_reference_checkpoint(ref, {"lm_head.weight": ref_lm_head})
    _write_fake_awq_checkpoint(
        ckpt,
        {
            "model.language_model.layers.0.mlp.gate_proj.weight_packed": torch.zeros(8, 4, dtype=torch.int32),
            "lm_head.weight": drifted_lm_head,
        },
    )

    with pytest.raises(AssertionError, match="sampled_rel_l2"):
        check_awq_w4a16_checkpoint(ckpt, reference_checkpoint=ref, max_preserved_elements=4)
