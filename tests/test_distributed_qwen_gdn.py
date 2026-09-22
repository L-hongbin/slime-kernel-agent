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
from slime.utils.arguments import add_qwen_gdn_arguments, validate_qwen_gdn_distributed_options
from slime_plugins.models import distributed_gdn as distributed_gdn_module
from slime_plugins.models import gdn_a2a as gdn_a2a_module
from slime_plugins.models import qwen3_5 as qwen3_5_model
from slime_plugins.models.distributed_gdn import (
    _a2a_cp2hp_fused,
    _a2a_hp2cp_ragged,
    _build_gdn_head_shards,
    _build_head_perm_for_split_sections,
    _build_nonpacked_zigzag_perm,
    _build_thd_cp_a2a_perm,
    _get_packed_cu_seqlens,
    _get_parameter_local_cp,
    _get_thd_cp_a2a_perm,
    _reorder_nonpacked_zigzag,
    _resolve_cu_seqlens,
    a2a_cp_to_hp_packed,
    a2a_hp_to_cp_packed,
)
from slime_plugins.models.gdn_a2a import (
    _pack_rank_major,
    _pack_sequence,
    _unpack_equal,
    _unpack_rank_major,
    _unpack_sequence,
    clear_communication_workspaces,
)
from slime_plugins.models.qwen3_5 import _validate_qwen_gdn_recompute_norm_out

NUM_GPUS = 0


@pytest.mark.parametrize("shared", [False, True])
def test_packed_boundary_validation_cache(monkeypatch, shared):
    q = torch.tensor([0, 8, 16], dtype=torch.int32)
    kv = q if shared else q.clone()
    params = SimpleNamespace(cu_seqlens_q=q, cu_seqlens_kv=kv, cu_seqlens_q_padded=None, cu_seqlens_kv_padded=None)
    original = distributed_gdn_module._resolve_cu_seqlens
    calls = []

    def resolve(*args):
        calls.append(args[3])
        return original(*args)

    monkeypatch.setattr(distributed_gdn_module, "_resolve_cu_seqlens", resolve)
    assert _get_packed_cu_seqlens(params, 16, 2, cache_validation=True) is q
    assert len(calls) == (1 if shared else 2)
    calls.clear()
    assert _get_packed_cu_seqlens(params, 16, 2, cache_validation=True) is q
    assert calls == []
    # Uncached calls must validate even after a previous cached call.
    _get_packed_cu_seqlens(params, 16, 2)
    assert calls
    calls.clear()
    # CP topology is part of the key, even when the boundaries are unchanged.
    _get_packed_cu_seqlens(params, 16, 4, cache_validation=True)
    assert calls
    with pytest.raises(ValueError, match="total_sequence_length"):
        _get_packed_cu_seqlens(params, 32, 4, cache_validation=True)


@pytest.mark.parametrize("change", ["q_mutation", "kv_mutation", "replacement", "padded", "missing"])
def test_packed_boundary_cache_never_hides_invalid_input(change):
    q = torch.tensor([0, 8, 16], dtype=torch.int32)
    params = SimpleNamespace(
        cu_seqlens_q=q, cu_seqlens_kv=q.clone(), cu_seqlens_q_padded=None, cu_seqlens_kv_padded=None
    )
    _get_packed_cu_seqlens(params, 16, 2, cache_validation=True)
    if change == "q_mutation":
        params.cu_seqlens_q[1] = 4
    elif change == "kv_mutation":
        params.cu_seqlens_kv[1] = 4
    elif change == "replacement":
        params.cu_seqlens_kv = torch.tensor([0, 4, 16], dtype=torch.int32)
    elif change == "padded":
        params.cu_seqlens_q_padded = torch.tensor([0, 6, 16], dtype=torch.int32)
    else:
        params.cu_seqlens_kv = None
    with pytest.raises(ValueError):
        _get_packed_cu_seqlens(params, 16, 2, cache_validation=True)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"))],
)
@pytest.mark.parametrize("widths", [(4, 4), (4, 2, 2), (2, 2, 2, 2)])
@pytest.mark.parametrize("packed", [False, True])
def test_fused_packed_a2a_values_and_nonuniform_gradients(monkeypatch, device, widths, packed):
    """Simulate all ranks with independent layout references, including ragged heads.

    CUDA runs exercise the real Triton pack/unpack kernels, not NCCL.
    """
    cp = len(widths)
    local_s, batch, width = 6, 1, sum(widths)
    cu = torch.tensor([0, 2 * cp, 6 * cp], dtype=torch.int32, device=device)
    if packed:
        index, inverse = _build_thd_cp_a2a_perm(cu, cp, 6 * cp)
    else:
        inverse = _build_nonpacked_zigzag_perm(6 * cp, cp, torch.device(device))
        index = torch.argsort(inverse)
    permutation = torch.arange(width - 1, -1, -1, device=device)
    sources = [
        torch.arange(local_s * width, device=device, dtype=torch.float32).reshape(local_s, batch, width) + 100 * r
        for r in range(cp)
    ]
    reordered = [x.index_select(-1, permutation) for x in sources]
    heads = [torch.cat([x.split(widths, -1)[r] for x in reordered], 0) for r in range(cp)]
    # Non-contiguous, nonuniform gradients expose incorrect permutations hidden by sum().
    grads = [
        torch.arange(6 * cp * w, device=device, dtype=torch.float32).reshape(w, 1, 6 * cp).transpose(0, 2) + 10 * r
        for r, w in enumerate(widths)
    ]
    for rank in range(cp):
        group = SimpleNamespace(size=lambda: cp, rank=lambda: rank)
        source = sources[rank].clone().requires_grad_()
        expected_send = torch.cat([x.reshape(-1) for x in reordered[rank].split(widths, -1)])
        backward_chunks = [g.index_select(0, inverse).chunk(cp, 0)[rank] for g in grads]
        received_grad = torch.cat([x.reshape(-1) for x in backward_chunks])
        expected_grad = torch.empty_like(source)
        expected_grad.index_copy_(-1, permutation, torch.cat(backward_chunks, -1))
        calls = []

        def cp2hp(output, input_, output_split_sizes, input_split_sizes, actual_group):
            assert output.data_ptr() != input_.data_ptr()
            if not calls:
                torch.testing.assert_close(input_, expected_send)
                output.copy_(heads[rank].reshape(-1))
            else:
                torch.testing.assert_close(input_, grads[rank].index_select(0, inverse).reshape(-1))
                output.copy_(received_grad)
            calls.append(1)

        monkeypatch.setattr(gdn_a2a_module, "_all_to_all_single", cp2hp)
        actual = _a2a_cp2hp_fused(source, widths, group, permutation, index)
        torch.testing.assert_close(actual, heads[rank].index_select(0, index))
        actual.backward(grads[rank])
        torch.testing.assert_close(source.grad, expected_grad)
        assert len(calls) == 2

        head = heads[rank].index_select(0, index).detach().requires_grad_()
        expected_send = head.detach().index_select(0, inverse).reshape(-1)
        received = torch.cat([x.chunk(cp, 0)[rank].reshape(-1) for x in heads])
        expected_output = torch.cat([x.chunk(cp, 0)[rank] for x in heads], -1)
        out_grads = [x * 0.25 + 1 for x in sources]
        recv_grad = torch.cat([g.split(widths, -1)[rank] for g in out_grads], 0)
        expected_backward_send = torch.cat([x.reshape(-1) for x in out_grads[rank].split(widths, -1)])
        calls.clear()

        def hp2cp(output, input_, output_split_sizes, input_split_sizes, actual_group):
            assert output.data_ptr() != input_.data_ptr()
            if not calls:
                torch.testing.assert_close(input_, expected_send)
                output.copy_(received)
            else:
                torch.testing.assert_close(input_, expected_backward_send)
                output.copy_(recv_grad.reshape(-1))
            calls.append(1)

        monkeypatch.setattr(gdn_a2a_module, "_all_to_all_single", hp2cp)
        actual = a2a_hp_to_cp_packed(
            head,
            cp,
            group,
            SimpleNamespace(qkv_format="thd") if packed else None,
            inverse if packed else None,
            rank_widths=widths,
            a2a_implementation="fused",
        )
        torch.testing.assert_close(actual, expected_output)
        actual.backward(out_grads[rank])
        torch.testing.assert_close(head.grad, recv_grad.index_select(0, index))
        assert len(calls) == 2


def test_shared_qwen_gdn_arguments_support_distributed_flashqla():
    parser = add_qwen_gdn_arguments(ArgumentParser())
    args = parser.parse_args(
        [
            "--qwen-gdn-implementation",
            "distributed",
            "--qwen-gdn-backend",
            "flashqla",
            "--qwen-gdn-a2a-implementation",
            "native",
            "--qwen-gdn-cache-thd-permutation",
            "--qwen-gdn-sp-disable-batch-p2p-comm",
            "--qwen-gdn-recompute-norm-out",
        ]
    )

    assert args.qwen_gdn_implementation == "distributed"
    assert args.qwen_gdn_backend == "flashqla"
    assert args.qwen_gdn_a2a_implementation == "native"
    assert args.qwen_gdn_cache_thd_permutation is True
    assert args.qwen_gdn_sp_disable_batch_p2p_comm is True
    assert args.qwen_gdn_recompute_norm_out is True


def test_qwen_gdn_recompute_norm_out_defaults_to_disabled():
    args = add_qwen_gdn_arguments(ArgumentParser()).parse_args([])

    assert args.qwen_gdn_recompute_norm_out is False
    assert args.qwen_gdn_a2a_implementation is None
    assert args.qwen_gdn_cache_thd_permutation is False


@pytest.mark.parametrize(
    "args",
    [
        SimpleNamespace(
            qwen_gdn_implementation="replicated",
            qwen_gdn_a2a_implementation="native",
            qwen_gdn_cache_thd_permutation=False,
        ),
        SimpleNamespace(
            qwen_gdn_implementation="replicated",
            qwen_gdn_a2a_implementation=None,
            qwen_gdn_cache_thd_permutation=True,
        ),
    ],
)
def test_qwen_gdn_distributed_options_reject_replicated_gdn(args):
    with pytest.raises(ValueError, match="require --qwen-gdn-implementation distributed"):
        validate_qwen_gdn_distributed_options(args)


def test_qwen_gdn_distributed_options_accept_distributed_gdn():
    args = SimpleNamespace(
        qwen_gdn_implementation="distributed",
        qwen_gdn_a2a_implementation="fused",
        qwen_gdn_cache_thd_permutation=True,
    )
    validate_qwen_gdn_distributed_options(args)


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
    assert module.a2a_implementation == "native"
    assert module.cache_thd_permutation is False
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


def test_thd_cp_permutation_is_cached_per_packed_batch():
    packed_seq_params = SimpleNamespace()
    cu_seqlens = torch.tensor([0, 8, 20], dtype=torch.int32)

    first = _get_thd_cp_a2a_perm(packed_seq_params, cu_seqlens, cp_size=2, total_seq_len=20)
    cached = _get_thd_cp_a2a_perm(packed_seq_params, cu_seqlens, cp_size=2, total_seq_len=20)

    assert cached[0] is first[0]
    assert cached[1] is first[1]

    cu_seqlens[1] = 12
    after_mutation = _get_thd_cp_a2a_perm(packed_seq_params, cu_seqlens, cp_size=2, total_seq_len=20)
    assert after_mutation[0] is not first[0]

    replacement = cu_seqlens.clone()
    after_replacement = _get_thd_cp_a2a_perm(packed_seq_params, replacement, cp_size=2, total_seq_len=20)
    assert after_replacement[0] is not after_mutation[0]


def test_thd_cp_permutation_cache_is_opt_in(monkeypatch):
    class FakeGroup:
        def size(self):
            return 2

        def rank(self):
            return 0

    projected = torch.arange(16).reshape(2, 1, 8)
    exchanged = torch.arange(16).reshape(4, 1, 4)
    monkeypatch.setattr(distributed_gdn_module, "_a2a_cp2hp_fused", lambda *args, **kwargs: exchanged)
    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)

    cached_params = SimpleNamespace(qkv_format="thd")
    _, first_cached_inverse = a2a_cp_to_hp_packed(
        projected,
        (4, 4),
        2,
        FakeGroup(),
        cu_seqlens,
        4,
        cached_params,
        a2a_implementation="fused",
        cache_thd_permutation=True,
    )
    _, second_cached_inverse = a2a_cp_to_hp_packed(
        projected,
        (4, 4),
        2,
        FakeGroup(),
        cu_seqlens,
        4,
        cached_params,
        a2a_implementation="fused",
        cache_thd_permutation=True,
    )
    assert second_cached_inverse is first_cached_inverse

    uncached_params = SimpleNamespace(qkv_format="thd")
    _, first_uncached_inverse = a2a_cp_to_hp_packed(
        projected,
        (4, 4),
        2,
        FakeGroup(),
        cu_seqlens,
        4,
        uncached_params,
        a2a_implementation="fused",
        cache_thd_permutation=False,
    )
    _, second_uncached_inverse = a2a_cp_to_hp_packed(
        projected,
        (4, 4),
        2,
        FakeGroup(),
        cu_seqlens,
        4,
        uncached_params,
        a2a_implementation="fused",
        cache_thd_permutation=False,
    )
    assert second_uncached_inverse is not first_uncached_inverse
    assert not hasattr(uncached_params, "_slime_gdn_cp_permutation_cache")


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
            output,
            tensor,
            output_split_sizes,
            input_split_sizes,
            group,
            expected_rank=rank,
            expected_input=expected_packed,
            expected_output=expected_exchanged,
        ):
            assert group.rank() == expected_rank
            assert torch.equal(tensor, expected_input)
            assert input_split_sizes == [8, 4, 4]
            assert output_split_sizes == [2 * rank_widths[expected_rank]] * 3
            output.copy_(expected_output)

        monkeypatch.setattr(gdn_a2a_module, "_all_to_all_single", fake_cp2hp)
        actual_head_shard = _a2a_cp2hp_fused(local_input, rank_widths, FakeGroup(rank))
        assert torch.equal(actual_head_shard, head_shards[rank])

        expected_reverse_packed = torch.cat(
            [chunk.contiguous().view(-1) for chunk in torch.chunk(head_shards[rank], 3, dim=0)]
        )
        expected_reverse_exchanged = torch.cat(
            [torch.chunk(head_shard, 3, dim=0)[rank].contiguous().view(-1) for head_shard in head_shards]
        )

        def fake_hp2cp(
            output,
            tensor,
            output_split_sizes,
            input_split_sizes,
            group,
            expected_rank=rank,
            expected_input=expected_reverse_packed,
            expected_output=expected_reverse_exchanged,
        ):
            assert group.rank() == expected_rank
            assert torch.equal(tensor, expected_input)
            assert input_split_sizes == [2 * rank_widths[expected_rank]] * 3
            assert output_split_sizes == [8, 4, 4]
            output.copy_(expected_output)

        monkeypatch.setattr(gdn_a2a_module, "_all_to_all_single", fake_hp2cp)
        restored = _a2a_hp2cp_ragged(head_shards[rank], rank_widths, FakeGroup(rank))
        assert torch.equal(restored, local_input)


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("packed", [False, True])
def test_equal_cp_uses_fused_pack_and_matches_reference(monkeypatch, cp_size, packed):
    class FakeGroup:
        def size(self):
            return cp_size

        def rank(self):
            return cp_size - 1

    group = FakeGroup()
    split_sections = (2 * cp_size, 4 * cp_size)
    total_width = sum(split_sections)
    local_inputs = [
        torch.arange(2 * total_width).reshape(2, 1, total_width) + source_rank * 1000 for source_rank in range(cp_size)
    ]
    permutation = _build_head_perm_for_split_sections(split_sections, cp_size=cp_size, device=torch.device("cpu"))
    reordered_inputs = [source.index_select(-1, permutation) for source in local_inputs]
    rank_width = total_width // cp_size
    expected_send = torch.cat(
        [chunk.contiguous().view(-1) for chunk in torch.split(reordered_inputs[group.rank()], rank_width, dim=-1)]
    )
    expected_exchanged = torch.cat(
        [torch.split(source, rank_width, dim=-1)[group.rank()] for source in reordered_inputs], dim=0
    )

    def fake_all_to_all(output, input_, output_split_sizes, input_split_sizes, actual_group):
        assert actual_group is group
        assert torch.equal(input_, expected_send)
        assert input_split_sizes == [2 * rank_width] * cp_size
        assert output_split_sizes == [2 * rank_width] * cp_size
        output.copy_(expected_exchanged.view(-1))

    monkeypatch.setattr(gdn_a2a_module, "_all_to_all_single", fake_all_to_all)
    cu_seqlens = torch.tensor([0, 2 * cp_size], dtype=torch.int32)
    packed_seq_params = SimpleNamespace(qkv_format="thd") if packed else None

    actual, inverse = a2a_cp_to_hp_packed(
        local_inputs[group.rank()],
        split_sections,
        cp_size,
        group,
        cu_seqlens,
        total_seq_len=2 * cp_size,
        packed_seq_params=packed_seq_params,
        a2a_implementation="fused",
    )

    if packed:
        index, expected_inverse = _build_thd_cp_a2a_perm(cu_seqlens, cp_size, total_seq_len=2 * cp_size)
        assert torch.equal(actual, expected_exchanged.index_select(0, index))
        assert torch.equal(inverse, expected_inverse)
    else:
        assert torch.equal(actual, _reorder_nonpacked_zigzag(expected_exchanged, cp_size, undo=True))
        assert inverse is None


def test_native_equal_cp_preserves_megatron_paths(monkeypatch):
    class FakeGroup:
        def size(self):
            return 2

        def rank(self):
            return 0

    group = FakeGroup()
    projected = torch.arange(32).reshape(2, 1, 16)
    split_sections = (8, 8)
    head_permutation = _build_head_perm_for_split_sections(split_sections, 2, torch.device("cpu"))
    expected_projected = projected.index_select(-1, head_permutation)
    cp_to_hp_result = torch.full((4, 1, 8), 7)

    def fake_cp_to_hp(tensor, seq_dim, head_dim, cp_group, undo_attention_load_balancing=True):
        assert torch.equal(tensor, expected_projected)
        assert (seq_dim, head_dim, cp_group, undo_attention_load_balancing) == (0, -1, group, True)
        return cp_to_hp_result

    monkeypatch.setattr(distributed_gdn_module, "tensor_a2a_cp2hp", fake_cp_to_hp)
    monkeypatch.setattr(
        distributed_gdn_module,
        "_a2a_cp2hp_fused",
        lambda *args, **kwargs: pytest.fail("native path called fused CP-to-HP"),
    )
    actual, inverse = a2a_cp_to_hp_packed(
        projected,
        split_sections,
        cp_size=2,
        cp_group=group,
        cu_seqlens=None,
        total_seq_len=4,
        packed_seq_params=None,
        a2a_implementation="native",
    )
    assert actual is cp_to_hp_result
    assert inverse is None

    natural = torch.arange(16).reshape(4, 1, 4)
    sequence_inverse = torch.tensor([0, 3, 1, 2])
    expected_rank_major = natural.index_select(0, sequence_inverse)
    hp_to_cp_result = torch.full((2, 1, 8), 9)

    def fake_hp_to_cp(tensor, seq_dim, head_dim, cp_group, redo_attention_load_balancing):
        assert torch.equal(tensor, expected_rank_major)
        assert (seq_dim, head_dim, cp_group, redo_attention_load_balancing) == (0, -1, group, False)
        return hp_to_cp_result

    monkeypatch.setattr(distributed_gdn_module, "tensor_a2a_hp2cp", fake_hp_to_cp)
    monkeypatch.setattr(
        distributed_gdn_module,
        "fused_hp_to_cp",
        lambda *args, **kwargs: pytest.fail("native path called fused HP-to-CP"),
    )
    actual = a2a_hp_to_cp_packed(
        natural,
        cp_size=2,
        cp_group=group,
        packed_seq_params=SimpleNamespace(qkv_format="thd"),
        inverse=sequence_inverse,
        a2a_implementation="native",
    )
    assert actual is hp_to_cp_result


@pytest.mark.parametrize("cp_size", [2, 4])
@pytest.mark.parametrize("packed", [False, True])
def test_equal_hp_to_cp_fused_matches_reference_and_backward(monkeypatch, cp_size, packed):
    class FakeGroup:
        def size(self):
            return cp_size

        def rank(self):
            return cp_size - 1

    group = FakeGroup()
    total_seq_len = 2 * cp_size
    local_width = 3
    head_inputs = [
        torch.arange(total_seq_len * local_width, dtype=torch.float32).reshape(total_seq_len, 1, local_width)
        + head_rank * 100
        for head_rank in range(cp_size)
    ]
    if packed:
        cu_seqlens = torch.tensor([0, total_seq_len], dtype=torch.int32)
        _, sequence_permutation = _build_thd_cp_a2a_perm(cu_seqlens, cp_size, total_seq_len)
        packed_seq_params = SimpleNamespace(qkv_format="thd")
    else:
        sequence_permutation = _build_nonpacked_zigzag_perm(total_seq_len, cp_size, torch.device("cpu"))
        packed_seq_params = None
    rank_major_inputs = [head.index_select(0, sequence_permutation) for head in head_inputs]
    expected_send = rank_major_inputs[group.rank()].reshape(-1)
    expected_chunks = [torch.chunk(head, cp_size, dim=0)[group.rank()] for head in rank_major_inputs]
    expected_receive = torch.cat([chunk.reshape(-1) for chunk in expected_chunks])
    expected_output = torch.cat(expected_chunks, dim=-1)
    calls = 0
    workspace_pointers = []

    def fake_all_to_all(output, input_, output_split_sizes, input_split_sizes, actual_group):
        nonlocal calls
        assert actual_group is group
        workspace_pointers.append((input_.data_ptr(), output.data_ptr()))
        chunk_size = 2 * local_width
        assert input_split_sizes == [chunk_size] * cp_size
        assert output_split_sizes == [chunk_size] * cp_size
        if calls == 0:
            assert torch.equal(input_, expected_send)
            output.copy_(expected_receive)
        else:
            assert torch.equal(input_, torch.ones_like(input_))
            output.fill_(1)
        calls += 1

    monkeypatch.setattr(gdn_a2a_module, "_all_to_all_single", fake_all_to_all)
    head_input = head_inputs[group.rank()].requires_grad_()
    actual = a2a_hp_to_cp_packed(
        head_input,
        cp_size,
        group,
        packed_seq_params,
        sequence_permutation if packed else None,
        a2a_implementation="fused",
    )
    actual.sum().backward()

    assert calls == 2
    assert workspace_pointers[0] == workspace_pointers[1]
    assert workspace_pointers[0][0] != workspace_pointers[0][1]
    assert torch.equal(actual, expected_output)
    assert torch.equal(head_input.grad, torch.ones_like(head_input))


def test_ragged_pack_fuses_head_permutation_and_reuses_workspace():
    rank_widths = (6, 3, 3)
    permutation = torch.tensor([0, 1, 4, 5, 8, 9, 2, 6, 10, 3, 7, 11])
    source = torch.arange(48).reshape(2, 2, 12)
    expected_reordered = source.index_select(-1, permutation)
    expected_packed = torch.cat(
        [chunk.contiguous().view(-1) for chunk in torch.split(expected_reordered, rank_widths, dim=-1)]
    )

    clear_communication_workspaces()
    packed = _pack_rank_major(source, rank_widths, permutation)
    first_pointer = packed.data_ptr()
    actual_packed = packed.clone()
    second_packed = _pack_rank_major(source + 100, rank_widths, permutation)

    assert torch.equal(actual_packed, expected_packed)
    assert second_packed.data_ptr() == first_pointer

    restored = torch.empty_like(source)
    _unpack_rank_major(actual_packed, restored, rank_widths, permutation)
    assert torch.equal(restored, source)


def test_ragged_a2a_custom_backward_restores_input_layout(monkeypatch):
    class FakeGroup:
        def size(self):
            return 3

        def rank(self):
            return 0

    rank_widths = (4, 2, 2)
    source = torch.arange(16, dtype=torch.float32).reshape(2, 1, 8).requires_grad_()
    expected_packed = torch.cat(
        [chunk.contiguous().view(-1) for chunk in torch.split(source.detach(), rank_widths, dim=-1)]
    )
    calls = 0

    def fake_all_to_all(output, input_, output_split_sizes, input_split_sizes, group):
        nonlocal calls
        assert group.rank() == 0
        if calls == 0:
            assert torch.equal(input_, expected_packed)
            assert output_split_sizes == [8, 8, 8]
            assert input_split_sizes == [8, 4, 4]
            output.copy_(torch.arange(output.numel(), dtype=output.dtype))
        else:
            assert output_split_sizes == [8, 4, 4]
            assert input_split_sizes == [8, 8, 8]
            output.fill_(1)
        calls += 1

    monkeypatch.setattr(gdn_a2a_module, "_all_to_all_single", fake_all_to_all)
    _a2a_cp2hp_fused(source, rank_widths, FakeGroup()).sum().backward()

    assert calls == 2
    assert torch.equal(source.grad, torch.ones_like(source))


def test_ragged_reverse_a2a_custom_backward_restores_head_layout(monkeypatch):
    class FakeGroup:
        def size(self):
            return 3

        def rank(self):
            return 0

    rank_widths = (4, 2, 2)
    head_shard = torch.arange(24, dtype=torch.float32).reshape(6, 1, 4).requires_grad_()
    received = torch.arange(16, dtype=torch.float32)
    expected_output = torch.tensor([[[0, 1, 2, 3, 8, 9, 12, 13]], [[4, 5, 6, 7, 10, 11, 14, 15]]], dtype=torch.float32)
    calls = 0

    def fake_all_to_all(output, input_, output_split_sizes, input_split_sizes, group):
        nonlocal calls
        assert group.rank() == 0
        if calls == 0:
            assert torch.equal(input_, head_shard.detach().view(-1))
            assert output_split_sizes == [8, 4, 4]
            assert input_split_sizes == [8, 8, 8]
            output.copy_(received)
        else:
            assert torch.equal(input_, torch.ones(16))
            assert output_split_sizes == [8, 8, 8]
            assert input_split_sizes == [8, 4, 4]
            output.fill_(1)
        calls += 1

    monkeypatch.setattr(gdn_a2a_module, "_all_to_all_single", fake_all_to_all)
    output = _a2a_hp2cp_ragged(head_shard, rank_widths, FakeGroup())
    output.sum().backward()

    assert calls == 2
    assert torch.equal(output, expected_output)
    assert torch.equal(head_shard.grad, torch.ones_like(head_shard))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA for the fused Triton kernel")
def test_ragged_fused_pack_matches_reference_on_cuda():
    # Keep the number of blocks above CUDA's grid-Y limit to cover long-context launches.
    rank_widths = (8192, 4096, 4096)
    permutation = torch.randperm(sum(rank_widths), device="cuda")
    source = torch.randn(1, 2049, sum(rank_widths), device="cuda", dtype=torch.bfloat16).transpose(0, 1)
    expected_reordered = source.index_select(-1, permutation)
    expected_packed = torch.cat(
        [chunk.contiguous().view(-1) for chunk in torch.split(expected_reordered, rank_widths, dim=-1)]
    )

    clear_communication_workspaces()
    packed = _pack_rank_major(source, rank_widths, permutation)
    workspace_pointer = packed.data_ptr()
    actual_packed = packed.clone()
    reused_workspace = _pack_rank_major(source + 1, rank_widths, permutation)
    restored = torch.empty_like(source.contiguous())
    _unpack_rank_major(actual_packed, restored, rank_widths, permutation)
    torch.cuda.synchronize()

    assert reused_workspace.data_ptr() == workspace_pointer
    assert torch.equal(actual_packed, expected_packed)
    assert torch.equal(restored, source)

    sequence_permutation = torch.randperm(source.size(0), device="cuda")
    sequence_packed = _pack_sequence(source, sequence_permutation).clone()
    sequence_restored = torch.empty_like(restored)
    _unpack_sequence(sequence_packed, sequence_restored, sequence_permutation)
    assert torch.equal(sequence_packed.view_as(source), source.index_select(0, sequence_permutation))
    assert torch.equal(sequence_restored, source)

    equal_widths = (source.size(-1) // 4,) * 4
    equal_packed = _pack_rank_major(source, equal_widths, None).clone()
    equal_restored = torch.empty_like(restored)
    _unpack_equal(equal_packed, equal_restored, cp_size=4)
    assert torch.equal(equal_restored, source)


def test_nonpacked_zigzag_reorder_round_trip_with_odd_cp():
    rank_major = torch.tensor([1, 6, 2, 5, 3, 4]).reshape(6, 1, 1)

    natural = _reorder_nonpacked_zigzag(rank_major, cp_size=3, undo=True)

    assert natural.flatten().tolist() == [1, 2, 3, 4, 5, 6]
    assert torch.equal(_reorder_nonpacked_zigzag(natural, cp_size=3, undo=False), rank_major)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
