from argparse import ArgumentParser, Namespace
from types import SimpleNamespace

import pytest
import torch

from slime.backends.megatron_utils import model_provider as model_provider_module
from slime.backends.megatron_utils.hf_to_megatron.qwen3_5 import qwen3_5_hf_tensor
from slime.backends.megatron_utils.megatron_to_hf import qwen3_5 as qwen3_5_converter
from slime.backends.megatron_utils.megatron_to_hf.processors import quantizer_fp8
from slime.backends.megatron_utils.model_provider import _apply_qwen_gdn_pipeline_overrides
from slime.backends.megatron_utils.qwen_gdn_layout import (
    get_gdn_factory_group,
    group_gdn_factory_keys,
    interleave_gdn_tp_sections,
    merge_gdn_factory_tensors,
)
from slime.utils.arguments import add_qwen_gdn_arguments
from slime_plugins.models import distributed_gdn as distributed_gdn_module
from slime_plugins.models import qwen3_5 as qwen3_5_model
from slime_plugins.models.distributed_gdn import (
    _a2a_cp2hp_ragged,
    _a2a_hp2cp_ragged,
    _build_gdn_head_shards,
    _build_head_perm_for_split_sections,
    _build_thd_cp_a2a_perm,
    _get_parameter_local_cp,
    _reorder_nonpacked_zigzag,
    _resolve_cu_seqlens,
)
from slime_plugins.models.qwen3_5 import _validate_qwen_gdn_recompute_norm_out

NUM_GPUS = 0


def test_shared_qwen_gdn_arguments_support_distributed_flashqla():
    parser = add_qwen_gdn_arguments(ArgumentParser())
    args = parser.parse_args(
        [
            "--qwen-gdn-implementation",
            "distributed",
            "--qwen-gdn-backend",
            "flashqla",
            "--qwen-gdn-sp-disable-batch-p2p-comm",
            "--qwen-gdn-recompute-norm-out",
        ]
    )

    assert args.qwen_gdn_implementation == "distributed"
    assert args.qwen_gdn_backend == "flashqla"
    assert args.qwen_gdn_sp_disable_batch_p2p_comm is True
    assert args.qwen_gdn_recompute_norm_out is True


def test_qwen_gdn_recompute_norm_out_defaults_to_disabled():
    args = add_qwen_gdn_arguments(ArgumentParser()).parse_args([])

    assert args.qwen_gdn_recompute_norm_out is False


def test_qwen_gdn_recompute_norm_out_rejects_replicated_and_full_recompute():
    args = SimpleNamespace(qwen_gdn_recompute_norm_out=True)

    with pytest.raises(ValueError, match="requires --qwen-gdn-implementation distributed"):
        _validate_qwen_gdn_recompute_norm_out(args, SimpleNamespace(recompute_granularity=None), False)

    with pytest.raises(ValueError, match="cannot be combined with full"):
        _validate_qwen_gdn_recompute_norm_out(args, SimpleNamespace(recompute_granularity="full"), True)


@pytest.mark.parametrize("recompute_granularity", [None, "selective"])
def test_qwen_gdn_recompute_norm_out_accepts_distributed_non_full_recompute(recompute_granularity):
    args = SimpleNamespace(qwen_gdn_recompute_norm_out=True)

    _validate_qwen_gdn_recompute_norm_out(args, SimpleNamespace(recompute_granularity=recompute_granularity), True)


def test_qwen_gdn_pipeline_override_disables_batched_p2p_and_rejects_overlap():
    args = SimpleNamespace(
        qwen_gdn_implementation="distributed",
        qwen_gdn_sp_disable_batch_p2p_comm=True,
        sequence_parallel=True,
    )
    config = SimpleNamespace(
        pipeline_model_parallel_size=2,
        overlap_p2p_comm=False,
        batch_p2p_comm=True,
    )

    _apply_qwen_gdn_pipeline_overrides(args, config)
    assert config.batch_p2p_comm is False

    config = SimpleNamespace(
        pipeline_model_parallel_size=2,
        overlap_p2p_comm=True,
        batch_p2p_comm=False,
    )
    with pytest.raises(ValueError, match="cannot be combined with overlap"):
        _apply_qwen_gdn_pipeline_overrides(args, config)


@pytest.mark.parametrize(
    ("implementation", "sequence_parallel", "pipeline_size"),
    [
        ("distributed", True, 1),
        ("replicated", True, 2),
    ],
)
def test_qwen_gdn_pipeline_override_leaves_unaffected_paths_unchanged(
    implementation, sequence_parallel, pipeline_size
):
    args = SimpleNamespace(
        qwen_gdn_implementation=implementation,
        qwen_gdn_sp_disable_batch_p2p_comm=False,
        sequence_parallel=sequence_parallel,
    )
    config = SimpleNamespace(
        pipeline_model_parallel_size=pipeline_size,
        overlap_p2p_comm=False,
        batch_p2p_comm=True,
    )

    _apply_qwen_gdn_pipeline_overrides(args, config)

    assert config.batch_p2p_comm is True


def test_model_provider_applies_qwen_gdn_pipeline_override(monkeypatch):
    class ExpectedStop(Exception):
        pass

    config = SimpleNamespace(
        pipeline_model_parallel_size=2,
        overlap_p2p_comm=False,
        batch_p2p_comm=True,
    )
    monkeypatch.setattr(model_provider_module, "core_transformer_config_from_args", lambda _: config)
    monkeypatch.setattr(
        model_provider_module,
        "import_module",
        lambda _: (_ for _ in ()).throw(ExpectedStop),
    )
    args = SimpleNamespace(
        custom_model_provider_path=None,
        transformer_impl="transformer_engine",
        spec="unused",
        qwen_gdn_implementation="distributed",
        qwen_gdn_sp_disable_batch_p2p_comm=True,
        sequence_parallel=True,
    )

    provider = model_provider_module._get_model_provider_func(args)
    with pytest.raises(ExpectedStop):
        provider()

    assert config.batch_p2p_comm is False


def test_gdn_tp_section_layout_round_trip():
    sections = [
        torch.arange(0, 8).reshape(4, 2),
        torch.arange(100, 112).reshape(6, 2),
        torch.arange(200, 204).reshape(2, 2),
    ]

    interleaved = interleave_gdn_tp_sections(sections, tp_size=2)
    restored = qwen3_5_converter.deinterleave_gdn_tp_sections(
        interleaved,
        tuple(section.shape[0] for section in sections),
        tp_size=2,
    )

    assert all(torch.equal(actual, expected) for actual, expected in zip(restored, sections, strict=True))
    assert interleaved[:, 0].tolist() == [0, 2, 100, 102, 104, 200, 4, 6, 106, 108, 110, 202]


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_distributed_gdn_factory_tensors_restore_fused_projection(tp_size):
    fused_key = "decoder.layers.0.self_attention.in_proj.weight"
    section_names = ("query", "key", "value", "z", "beta", "alpha")
    sections = [torch.full((size, 3), float(index)) for index, size in enumerate((8, 8, 12, 12, 4, 4), 1)]
    state_dict = {f"{fused_key}.{name}": section for name, section in zip(section_names, sections, strict=True)}

    merged_keys = merge_gdn_factory_tensors(state_dict, tp_size=tp_size, implementation="distributed")

    assert merged_keys == [fused_key]
    assert list(state_dict) == [fused_key]
    assert torch.equal(state_dict[fused_key], interleave_gdn_tp_sections(sections, tp_size=tp_size))


def test_distributed_gdn_factory_tensors_support_layer_stacked_checkpoints():
    fused_key = "decoder.layers.self_attention.conv1d.weight"
    section_names = ("query", "key", "value")
    sections = [
        torch.stack([torch.full((size, 2), float(layer * 10 + index)) for layer in range(2)])
        for index, size in enumerate((4, 4, 6), 1)
    ]
    state_dict = {f"{fused_key}.{name}": section for name, section in zip(section_names, sections, strict=True)}

    merge_gdn_factory_tensors(state_dict, tp_size=2, implementation="auto")

    for layer in range(2):
        expected = interleave_gdn_tp_sections([section[layer] for section in sections], tp_size=2)
        assert torch.equal(state_dict[fused_key][layer], expected)


def test_distributed_gdn_factory_tensor_validation_is_actionable():
    fused_key = "decoder.layers.0.self_attention.in_proj.weight"
    state_dict = {f"{fused_key}.{name}": torch.ones(2, 2) for name in ("query", "key", "value", "z", "beta")}

    with pytest.raises(ValueError, match=r"Missing GDN factory sections.*alpha"):
        merge_gdn_factory_tensors(state_dict, tp_size=2, implementation="distributed")

    state_dict[f"{fused_key}.alpha"] = torch.ones(2, 2)
    with pytest.raises(ValueError, match=r"--qwen-gdn-implementation distributed or auto"):
        merge_gdn_factory_tensors(state_dict, tp_size=2, implementation="replicated")


def test_gdn_factory_group_matches_only_supported_factory_entries():
    assert get_gdn_factory_group("decoder.layers.0.self_attention.in_proj.weight.z") == (
        "decoder.layers.0.self_attention.in_proj.weight",
        ("query", "key", "value", "z", "beta", "alpha"),
    )
    assert get_gdn_factory_group("decoder.layers.0.self_attention.out_proj.weight") is None


def test_distributed_gdn_factory_keys_stay_in_one_conversion_worker():
    fused_key = "decoder.layers.0.self_attention.in_proj.weight"
    section_names = ("query", "key", "value", "z", "beta", "alpha")
    factory_keys = [f"{fused_key}.{name}" for name in section_names]

    grouped_keys = group_gdn_factory_keys(
        [factory_keys[3], "decoder.final_layernorm.weight", *factory_keys[:3], *factory_keys[4:]]
    )

    assert grouped_keys == [factory_keys]


def test_native_gdn_converter_restores_hf_projection_order(monkeypatch):
    monkeypatch.setattr(qwen3_5_converter, "_load_qwen_gdn_dimensions", lambda _: (4, 6, 2))
    logical_sections = [
        torch.full((4, 3), 1.0),
        torch.full((4, 3), 2.0),
        torch.full((6, 3), 3.0),
        torch.full((6, 3), 4.0),
        torch.full((2, 3), 5.0),
        torch.full((2, 3), 6.0),
    ]
    gathered_native = interleave_gdn_tp_sections(logical_sections, tp_size=2)
    args = Namespace(hf_checkpoint="unused", tensor_model_parallel_size=2, kv_channels=1)

    converted = qwen3_5_converter.convert_qwen3_5_to_hf(
        args,
        "module.module.decoder.layers.0.self_attention.in_proj.weight",
        gathered_native,
    )

    assert [name for name, _ in converted] == [
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        "model.language_model.layers.0.linear_attn.in_proj_z.weight",
        "model.language_model.layers.0.linear_attn.in_proj_b.weight",
        "model.language_model.layers.0.linear_attn.in_proj_a.weight",
    ]
    assert converted[0][1][:, 0].tolist() == [1.0] * 4 + [2.0] * 4 + [3.0] * 6
    assert converted[1][1][:, 0].tolist() == [4.0] * 6
    assert converted[2][1][:, 0].tolist() == [5.0] * 2
    assert converted[3][1][:, 0].tolist() == [6.0] * 2


def test_native_gdn_hf_loader_builds_tp_rank_major_projection_and_round_trips(monkeypatch):
    monkeypatch.setattr(
        "slime.backends.megatron_utils.hf_to_megatron.qwen3_5._get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(qwen3_5_converter, "_load_qwen_gdn_dimensions", lambda _: (4, 6, 2))
    prefix = "model.language_model.layers.0.linear_attn"
    logical_sections = [
        torch.full((4, 3), 1.0),
        torch.full((4, 3), 2.0),
        torch.full((6, 3), 3.0),
        torch.full((6, 3), 4.0),
        torch.full((2, 3), 5.0),
        torch.full((2, 3), 6.0),
    ]
    tensors = {
        f"{prefix}.in_proj_qkv.weight": torch.cat(logical_sections[:3], dim=0),
        f"{prefix}.in_proj_z.weight": logical_sections[3],
        f"{prefix}.in_proj_b.weight": logical_sections[4],
        f"{prefix}.in_proj_a.weight": logical_sections[5],
    }
    reader = SimpleNamespace(get_tensor=tensors.__getitem__)
    text_config = SimpleNamespace(
        linear_key_head_dim=2,
        linear_num_key_heads=2,
        linear_value_head_dim=3,
        linear_num_value_heads=2,
    )

    loaded = qwen3_5_hf_tensor(
        "module.module.decoder.layers.0.self_attention.in_proj.weight",
        reader,
        SimpleNamespace(text_config=text_config, tie_word_embeddings=False),
    )

    expected = interleave_gdn_tp_sections(logical_sections, tp_size=2)
    assert torch.equal(loaded, expected)
    rank_size = sum(section.shape[0] // 2 for section in logical_sections)
    assert loaded[:rank_size, 0].tolist() == [1.0] * 2 + [2.0] * 2 + [3.0] * 3 + [4.0] * 3 + [5.0, 6.0]
    converted = dict(
        qwen3_5_converter.convert_qwen3_5_to_hf(
            Namespace(hf_checkpoint="unused", tensor_model_parallel_size=2, kv_channels=1),
            "module.module.decoder.layers.0.self_attention.in_proj.weight",
            loaded,
        )
    )
    assert torch.equal(converted[f"{prefix}.in_proj_qkv.weight"], tensors[f"{prefix}.in_proj_qkv.weight"])
    assert torch.equal(converted[f"{prefix}.in_proj_z.weight"], tensors[f"{prefix}.in_proj_z.weight"])
    assert torch.equal(converted[f"{prefix}.in_proj_b.weight"], tensors[f"{prefix}.in_proj_b.weight"])
    assert torch.equal(converted[f"{prefix}.in_proj_a.weight"], tensors[f"{prefix}.in_proj_a.weight"])


def test_native_gdn_hf_loader_builds_tp_rank_major_convolution_and_round_trips(monkeypatch):
    monkeypatch.setattr(
        "slime.backends.megatron_utils.hf_to_megatron.qwen3_5._get_tensor_model_parallel_world_size",
        lambda: 2,
    )
    monkeypatch.setattr(qwen3_5_converter, "_load_qwen_gdn_dimensions", lambda _: (4, 6, 2))
    prefix = "model.language_model.layers.0.linear_attn"
    sections = [
        torch.full((4, 1, 2), 1.0),
        torch.full((4, 1, 2), 2.0),
        torch.full((6, 1, 2), 3.0),
    ]
    hf_conv = torch.cat(sections, dim=0)
    reader = SimpleNamespace(get_tensor={f"{prefix}.conv1d.weight": hf_conv}.__getitem__)
    text_config = SimpleNamespace(
        linear_key_head_dim=2,
        linear_num_key_heads=2,
        linear_value_head_dim=3,
        linear_num_value_heads=2,
    )

    loaded = qwen3_5_hf_tensor(
        "module.module.decoder.layers.0.self_attention.conv1d.weight",
        reader,
        SimpleNamespace(text_config=text_config, tie_word_embeddings=False),
    )

    expected = interleave_gdn_tp_sections(sections, tp_size=2)
    assert torch.equal(loaded, expected)
    converted = dict(
        qwen3_5_converter.convert_qwen3_5_to_hf(
            Namespace(hf_checkpoint="unused", tensor_model_parallel_size=2, kv_channels=1),
            "module.module.decoder.layers.0.self_attention.conv1d.weight",
            loaded,
        )
    )
    assert torch.equal(converted[f"{prefix}.conv1d.weight"], hf_conv)


@pytest.mark.parametrize(
    ("mcore_name", "hf_suffix"),
    [
        ("in_proj.layer_norm_weight", "input_layernorm.weight"),
        ("out_proj.weight", "linear_attn.out_proj.weight"),
        ("A_log", "linear_attn.A_log"),
        ("dt_bias", "linear_attn.dt_bias"),
    ],
)
def test_native_gdn_hf_loader_maps_direct_parameters(mcore_name, hf_suffix):
    prefix = "model.language_model.layers.0"
    parameter = torch.randn(4, 3)
    reader = SimpleNamespace(get_tensor={f"{prefix}.{hf_suffix}": parameter}.__getitem__)
    text_config = SimpleNamespace(
        linear_key_head_dim=2,
        linear_num_key_heads=2,
        linear_value_head_dim=3,
        linear_num_value_heads=2,
    )

    loaded = qwen3_5_hf_tensor(
        f"module.module.decoder.layers.0.self_attention.{mcore_name}",
        reader,
        SimpleNamespace(text_config=text_config, tie_word_embeddings=False),
    )

    assert loaded is parameter


def test_native_gdn_hf_loader_converts_out_norm_to_zero_centered_gamma():
    prefix = "model.language_model.layers.0.linear_attn"
    hf_gamma = torch.tensor([1.0, 1.25, 0.5])
    reader = SimpleNamespace(get_tensor={f"{prefix}.norm.weight": hf_gamma}.__getitem__)
    text_config = SimpleNamespace(
        linear_key_head_dim=2,
        linear_num_key_heads=2,
        linear_value_head_dim=3,
        linear_num_value_heads=2,
    )

    loaded = qwen3_5_hf_tensor(
        "module.module.decoder.layers.0.self_attention.out_norm.weight",
        reader,
        SimpleNamespace(text_config=text_config, tie_word_embeddings=False),
    )

    assert torch.equal(loaded, hf_gamma - 1)


def test_native_gdn_fp8_quantization_preserves_b_and_a(monkeypatch):
    quantized_names = []

    def fake_quantize(name, weight, weight_block_size, scale_suffix=None):
        del weight_block_size, scale_suffix
        quantized_names.append(name)
        return [(name, weight.to(torch.float16)), (name.replace(".weight", ".weight_scale_inv"), torch.ones(1))]

    monkeypatch.setattr(quantizer_fp8, "_quantize_param", fake_quantize)
    prefix = "model.language_model.layers.0.linear_attn"
    converted = [
        (f"{prefix}.in_proj_qkv.weight", torch.ones(4, 4, dtype=torch.bfloat16)),
        (f"{prefix}.in_proj_z.weight", torch.ones(4, 4, dtype=torch.bfloat16)),
        (f"{prefix}.in_proj_b.weight", torch.ones(2, 4, dtype=torch.bfloat16)),
        (f"{prefix}.in_proj_a.weight", torch.ones(2, 4, dtype=torch.bfloat16)),
    ]

    result = quantizer_fp8.quantize_params_fp8(
        Namespace(),
        "module.module.decoder.layers.0.self_attention.in_proj.weight",
        converted,
        {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
    )

    assert quantized_names == [f"{prefix}.in_proj_qkv.weight", f"{prefix}.in_proj_z.weight"]
    result_by_name = dict(result)
    assert result_by_name[f"{prefix}.in_proj_b.weight"].dtype == torch.bfloat16
    assert result_by_name[f"{prefix}.in_proj_a.weight"].dtype == torch.bfloat16
    assert f"{prefix}.in_proj_b.weight_scale_inv" not in result_by_name
    assert f"{prefix}.in_proj_a.weight_scale_inv" not in result_by_name


def test_native_gdn_input_norm_raw_export_preserves_qwen_zero_centered_gamma():
    args = Namespace(hf_checkpoint="unused", tensor_model_parallel_size=2, kv_channels=1)
    qwen_zero_centered_gamma = torch.tensor([0.0, 0.25, -0.5])

    converted = qwen3_5_converter.convert_qwen3_5_to_hf(
        args,
        "module.module.decoder.layers.0.self_attention.in_proj.layer_norm_weight",
        qwen_zero_centered_gamma,
    )

    assert converted[0][0] == "model.language_model.layers.0.input_layernorm.weight"
    assert torch.equal(converted[0][1], qwen_zero_centered_gamma)


def test_native_gdn_out_norm_converts_between_one_and_zero_centered_gamma():
    args = Namespace(hf_checkpoint="unused", tensor_model_parallel_size=2, kv_channels=1)
    mcore_zero_centered_gamma = torch.tensor([0.0, 0.25, -0.5])

    converted = qwen3_5_converter.convert_qwen3_5_to_hf(
        args,
        "module.module.decoder.layers.0.self_attention.out_norm.weight",
        mcore_zero_centered_gamma,
    )

    assert converted[0][0] == "model.language_model.layers.0.linear_attn.norm.weight"
    assert torch.equal(converted[0][1], mcore_zero_centered_gamma + 1)


@pytest.mark.parametrize("cp_size", [2, 3])
def test_distributed_gdn_python_spec_supports_sequence_parallel(monkeypatch, cp_size):
    block_spec = SimpleNamespace(layer_specs=[SimpleNamespace(submodules=SimpleNamespace(self_attention=None))])
    monkeypatch.setattr(qwen3_5_model, "get_gpt_decoder_block_spec", lambda *args, **kwargs: block_spec)
    monkeypatch.setattr(qwen3_5_model, "get_num_layers_to_build", lambda *args, **kwargs: 1)
    monkeypatch.setattr(qwen3_5_model, "get_transformer_layer_offset", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        qwen3_5_model,
        "_load_hf_config",
        lambda _: SimpleNamespace(
            num_hidden_layers=1,
            layer_types=["linear_attention"],
            linear_conv_kernel_dim=4,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_num_key_heads=16,
            linear_num_value_heads=32,
        ),
    )
    from megatron.core.models.gpt import experimental_attention_variant_module_specs

    monkeypatch.setattr(
        experimental_attention_variant_module_specs,
        "get_gated_delta_net_module_spec",
        lambda config: SimpleNamespace(module=None, params=None),
    )
    args = SimpleNamespace(
        num_experts=None,
        qwen_gdn_implementation="distributed",
        qwen_gdn_sp_disable_batch_p2p_comm=True,
        sequence_parallel=True,
        hf_checkpoint="unused",
    )
    config = SimpleNamespace(
        num_layers=1,
        pipeline_model_parallel_layout=None,
        tensor_model_parallel_size=4,
        context_parallel_size=cp_size,
        pipeline_model_parallel_size=2,
        overlap_p2p_comm=False,
        batch_p2p_comm=True,
    )

    result = qwen3_5_model.get_qwen3_5_spec(args, config, vp_stage=None)

    assert result.layer_specs[0].submodules.self_attention.params == {"args": args}
    assert config.batch_p2p_comm is False


def test_distributed_gdn_python_spec_rejects_sequence_parallel_with_batched_pp(monkeypatch):
    monkeypatch.setattr(qwen3_5_model, "get_gpt_decoder_block_spec", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(qwen3_5_model, "get_num_layers_to_build", lambda *args, **kwargs: 1)
    monkeypatch.setattr(qwen3_5_model, "get_transformer_layer_offset", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        qwen3_5_model,
        "_load_hf_config",
        lambda _: SimpleNamespace(num_hidden_layers=1, full_attention_interval=4),
    )
    args = SimpleNamespace(
        num_experts=None,
        qwen_gdn_implementation="distributed",
        qwen_gdn_sp_disable_batch_p2p_comm=False,
        sequence_parallel=True,
        hf_checkpoint="unused",
    )
    config = SimpleNamespace(
        num_layers=1,
        pipeline_model_parallel_layout=None,
        pipeline_model_parallel_size=2,
    )

    with pytest.raises(ValueError, match="requires --qwen-gdn-sp-disable-batch-p2p-comm"):
        qwen3_5_model.get_qwen3_5_spec(args, config, vp_stage=None)


def test_distributed_gdn_python_spec_rejects_contiguous_cp(monkeypatch):
    monkeypatch.setattr(qwen3_5_model, "get_gpt_decoder_block_spec", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(qwen3_5_model, "get_num_layers_to_build", lambda *args, **kwargs: 1)
    monkeypatch.setattr(qwen3_5_model, "get_transformer_layer_offset", lambda *args, **kwargs: 0)
    monkeypatch.setattr(
        qwen3_5_model,
        "_load_hf_config",
        lambda _: SimpleNamespace(num_hidden_layers=1, full_attention_interval=4),
    )
    args = SimpleNamespace(
        num_experts=None,
        qwen_gdn_implementation="distributed",
        sequence_parallel=False,
        cp_partition_mode="contiguous",
        hf_checkpoint="unused",
    )
    config = SimpleNamespace(
        num_layers=1,
        pipeline_model_parallel_layout=None,
        context_parallel_size=2,
    )

    with pytest.raises(ValueError, match="only zigzag"):
        qwen3_5_model.get_qwen3_5_spec(args, config, vp_stage=None)


def test_head_permutation_shards_each_fused_section():
    permutation = _build_head_perm_for_split_sections((8, 4), cp_size=2, device=torch.device("cpu"))
    assert permutation.tolist() == [0, 1, 2, 3, 8, 9, 4, 5, 6, 7, 10, 11]


def test_gdn_head_shards_balance_whole_gqa_groups_for_ragged_cp():
    key_head_counts, value_head_counts = _build_gdn_head_shards(4, 8, cp_size=3)

    assert key_head_counts == (2, 1, 1)
    assert value_head_counts == (4, 2, 2)

    with pytest.raises(ValueError, match="at least one TP-local key head"):
        _build_gdn_head_shards(4, 8, cp_size=5)


def test_distributed_gdn_builds_rank_local_shapes_for_ragged_cp(monkeypatch):
    class FakeGroup:
        def size(self):
            return 3

        def rank(self):
            return 1

    def fake_gdn_init(module, *args, **kwargs):
        del args, kwargs
        torch.nn.Module.__init__(module)
        module.config = SimpleNamespace(deterministic_mode=True)
        module.pg_collection = SimpleNamespace(cp=FakeGroup())
        module.cp_size = 3
        module.tp_size = 1
        module.num_key_heads = 4
        module.num_value_heads = 8
        module.key_head_dim = 2
        module.value_head_dim = 3
        module.qk_dim_local_tp = 8
        module.v_dim_local_tp = 24
        module.conv1d = SimpleNamespace(
            weight=torch.nn.Parameter(torch.zeros(40, 1, 2)),
            bias=None,
        )
        module.dt_bias = torch.nn.Parameter(torch.zeros(8))
        module.A_log = torch.nn.Parameter(torch.zeros(8))

    monkeypatch.setattr(distributed_gdn_module.GatedDeltaNet, "__init__", fake_gdn_init)

    module = distributed_gdn_module.DistributedQwenGatedDeltaNet(args=SimpleNamespace(qwen_gdn_backend="fla"))

    assert module.ragged_cp is True
    assert module.num_key_heads_local_cp == 1
    assert module.num_value_heads_local_cp == 2
    assert module.in_proj_rank_split_sections == (
        (4, 4, 12, 12, 4, 4),
        (2, 2, 6, 6, 2, 2),
        (2, 2, 6, 6, 2, 2),
    )
    assert module.feat_dim_split == (10, 6, 2, 2)
    assert module.output_rank_widths == (12, 6, 6)


def test_head_permutation_supports_ragged_cp_sections():
    rank_split_sections = ((2, 4), (1, 2), (1, 2))

    permutation = _build_head_perm_for_split_sections(
        (4, 8),
        cp_size=3,
        device=torch.device("cpu"),
        rank_split_sections=rank_split_sections,
    )

    assert permutation.tolist() == [0, 1, 4, 5, 6, 7, 2, 8, 9, 3, 10, 11]


def test_thd_cp_permutation_restores_natural_multi_sequence_order():
    cu_seqlens = torch.tensor([0, 8, 20], dtype=torch.int32)
    rank_major_zigzag = torch.tensor([0, 1, 6, 7, 8, 9, 10, 17, 18, 19, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16])

    permutation, inverse = _build_thd_cp_a2a_perm(cu_seqlens, cp_size=2, total_seq_len=20)

    assert rank_major_zigzag.index_select(0, permutation).tolist() == list(range(20))
    assert torch.equal(permutation.index_select(0, inverse), torch.arange(20))


def test_thd_cp_permutation_restores_natural_order_with_odd_cp():
    cu_seqlens = torch.tensor([0, 12], dtype=torch.int32)
    rank_major_zigzag = torch.tensor([0, 1, 10, 11, 2, 3, 8, 9, 4, 5, 6, 7])

    permutation, inverse = _build_thd_cp_a2a_perm(cu_seqlens, cp_size=3, total_seq_len=12)

    assert rank_major_zigzag.index_select(0, permutation).tolist() == list(range(12))
    assert torch.equal(permutation.index_select(0, inverse), torch.arange(12))


def test_packed_boundaries_must_match_total_and_cp():
    valid = torch.tensor([0, 8, 20], dtype=torch.int32)
    assert _resolve_cu_seqlens(None, valid, 20, "cu_seqlens_q", cp_size=2) is valid

    with pytest.raises(ValueError, match="total_sequence_length"):
        _resolve_cu_seqlens(None, valid, 24, "cu_seqlens_q", cp_size=2)

    with pytest.raises(ValueError, match="divisible"):
        _resolve_cu_seqlens(None, torch.tensor([0, 6, 20]), 20, "cu_seqlens_q", cp_size=2)


def test_packed_boundaries_allow_odd_lengths_without_cp():
    boundaries = torch.tensor([0, 4001, 4096], dtype=torch.int32)

    assert _resolve_cu_seqlens(None, boundaries, 4096, "cu_seqlens_q", cp_size=1) is boundaries


def test_parameter_cp_slice_preserves_fused_sections():
    class FakeGroup:
        def size(self):
            return 2

        def rank(self):
            return 1

    parameter = torch.arange(12).reshape(12, 1)
    actual = _get_parameter_local_cp(parameter, dim=0, cp_group=FakeGroup(), split_sections=(8, 4))

    assert actual[:, 0].tolist() == [4, 5, 6, 7, 10, 11]


def test_parameter_cp_slice_supports_ragged_sections():
    class FakeGroup:
        def size(self):
            return 3

        def rank(self):
            return 1

    parameter = torch.arange(12).reshape(12, 1)
    actual = _get_parameter_local_cp(
        parameter,
        dim=0,
        cp_group=FakeGroup(),
        split_sections=(4, 8),
        rank_split_sections=((2, 4), (1, 2), (1, 2)),
    )

    assert actual[:, 0].tolist() == [2, 8, 9]


def test_ragged_a2a_round_trip_layout(monkeypatch):
    class FakeGroup:
        def __init__(self, rank):
            self._rank = rank

        def size(self):
            return 3

        def rank(self):
            return self._rank

    rank_widths = (4, 2, 2)
    local_inputs = [torch.arange(16).reshape(2, 1, 8) + source_rank * 100 for source_rank in range(3)]
    head_shards = [
        torch.cat(
            [torch.split(source, rank_widths, dim=-1)[head_rank] for source in local_inputs],
            dim=0,
        )
        for head_rank in range(3)
    ]

    for rank in range(3):
        local_input = local_inputs[rank]
        expected_packed = torch.cat(
            [chunk.contiguous().view(-1) for chunk in torch.split(local_input, rank_widths, dim=-1)]
        )
        expected_exchanged = torch.cat(
            [torch.split(source, rank_widths, dim=-1)[rank].contiguous().view(-1) for source in local_inputs]
        )

        def fake_cp2hp(
            group,
            tensor,
            output_split_sizes,
            input_split_sizes,
            expected_rank=rank,
            expected_input=expected_packed,
            expected_output=expected_exchanged,
        ):
            assert group.rank() == expected_rank
            assert torch.equal(tensor, expected_input)
            assert input_split_sizes == [8, 4, 4]
            assert output_split_sizes == [2 * rank_widths[expected_rank]] * 3
            return expected_output

        monkeypatch.setattr(distributed_gdn_module, "all_to_all", fake_cp2hp)
        actual_head_shard = _a2a_cp2hp_ragged(local_input, rank_widths, FakeGroup(rank))
        assert torch.equal(actual_head_shard, head_shards[rank])

        expected_reverse_packed = torch.cat(
            [chunk.contiguous().view(-1) for chunk in torch.chunk(head_shards[rank], 3, dim=0)]
        )
        expected_reverse_exchanged = torch.cat(
            [torch.chunk(head_shard, 3, dim=0)[rank].contiguous().view(-1) for head_shard in head_shards]
        )

        def fake_hp2cp(
            group,
            tensor,
            output_split_sizes,
            input_split_sizes,
            expected_rank=rank,
            expected_input=expected_reverse_packed,
            expected_output=expected_reverse_exchanged,
        ):
            assert group.rank() == expected_rank
            assert torch.equal(tensor, expected_input)
            assert input_split_sizes == [2 * rank_widths[expected_rank]] * 3
            assert output_split_sizes == [8, 4, 4]
            return expected_output

        monkeypatch.setattr(distributed_gdn_module, "all_to_all", fake_hp2cp)
        restored = _a2a_hp2cp_ragged(head_shards[rank], rank_widths, FakeGroup(rank))
        assert torch.equal(restored, local_input)


def test_nonpacked_zigzag_reorder_round_trip_with_odd_cp():
    rank_major = torch.tensor([1, 6, 2, 5, 3, 4]).reshape(6, 1, 1)

    natural = _reorder_nonpacked_zigzag(rank_major, cp_size=3, undo=True)

    assert natural.flatten().tolist() == [1, 2, 3, 4, 5, 6]
    assert torch.equal(_reorder_nonpacked_zigzag(natural, cp_size=3, undo=False), rank_major)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
