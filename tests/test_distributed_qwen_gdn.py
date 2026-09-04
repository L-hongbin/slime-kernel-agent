from argparse import ArgumentParser, Namespace
from types import SimpleNamespace

import pytest
import torch

from slime.backends.megatron_utils import model_provider as model_provider_module
from slime.backends.megatron_utils.hf_to_megatron.qwen3_5 import qwen3_5_hf_tensor
from slime.backends.megatron_utils.megatron_to_hf import qwen3_5 as qwen3_5_converter
from slime.backends.megatron_utils.megatron_to_hf.processors import quantizer_fp8
from slime.backends.megatron_utils.model_provider import _apply_qwen_gdn_pipeline_overrides
from slime.utils.arguments import add_qwen_gdn_arguments
from slime_plugins.models import qwen3_5 as qwen3_5_model
from slime_plugins.models.distributed_gdn import (
    _build_head_perm_for_split_sections,
    _build_thd_cp_a2a_perm,
    _get_parameter_local_cp,
    _resolve_cu_seqlens,
)

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
        ]
    )

    assert args.qwen_gdn_implementation == "distributed"
    assert args.qwen_gdn_backend == "flashqla"
    assert args.qwen_gdn_sp_disable_batch_p2p_comm is True


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

    interleaved = qwen3_5_converter.interleave_gdn_tp_sections(sections, tp_size=2)
    restored = qwen3_5_converter.deinterleave_gdn_tp_sections(
        interleaved,
        tuple(section.shape[0] for section in sections),
        tp_size=2,
    )

    assert all(torch.equal(actual, expected) for actual, expected in zip(restored, sections, strict=True))
    assert interleaved[:, 0].tolist() == [0, 2, 100, 102, 104, 200, 4, 6, 106, 108, 110, 202]


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
    gathered_native = qwen3_5_converter.interleave_gdn_tp_sections(logical_sections, tp_size=2)
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

    expected = qwen3_5_converter.interleave_gdn_tp_sections(logical_sections, tp_size=2)
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

    expected = qwen3_5_converter.interleave_gdn_tp_sections(sections, tp_size=2)
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


def test_distributed_gdn_python_spec_supports_sequence_parallel(monkeypatch):
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
        context_parallel_size=2,
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


def test_thd_cp_permutation_restores_natural_multi_sequence_order():
    cu_seqlens = torch.tensor([0, 8, 20], dtype=torch.int32)
    rank_major_zigzag = torch.tensor([0, 1, 6, 7, 8, 9, 10, 17, 18, 19, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16])

    permutation, inverse = _build_thd_cp_a2a_perm(cu_seqlens, cp_size=2, total_seq_len=20)

    assert rank_major_zigzag.index_select(0, permutation).tolist() == list(range(20))
    assert torch.equal(permutation.index_select(0, inverse), torch.arange(20))


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
