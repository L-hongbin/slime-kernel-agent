import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def load_model_provider_module():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    return importlib.import_module("custom_kernels.deepseek_v4.megatron.model_provider")


def load_mcore_model_module():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    return importlib.import_module("custom_kernels.deepseek_v4.megatron.mcore_model")


def load_megatron_model_module():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    megatron_lm = Path("/root/Megatron-LM")
    if megatron_lm.exists():
        sys.path.insert(0, str(megatron_lm))
    return importlib.import_module("slime.backends.megatron_utils.model")


def test_dsv4_pp_sequence_length_matches_bshd_padding():
    module = load_megatron_model_module()

    args = SimpleNamespace(
        seq_length=64,
        qkv_format="bshd",
        tensor_model_parallel_size=1,
        data_pad_size_multiplier=128,
    )
    assert module._v4_pp_sequence_length(args) == 128

    args.seq_length = 256
    assert module._v4_pp_sequence_length(args) == 256

    args.qkv_format = "thd"
    args.seq_length = 64
    assert module._v4_pp_sequence_length(args) == 64


def test_resolve_v4_layer_ids_handles_uneven_43_layer_pp4_split():
    module = load_model_provider_module()

    cfg = SimpleNamespace(
        pipeline_model_parallel_size=4,
        pipeline_model_parallel_layout=None,
        num_layers=43,
        num_layers_in_first_pipeline_stage=10,
        num_layers_in_last_pipeline_stage=11,
        account_for_embedding_in_pipeline_split=False,
        account_for_loss_in_pipeline_split=False,
        virtual_pipeline_model_parallel_size=None,
    )

    assert module.resolve_v4_layer_ids(cfg, pp_rank=0) == tuple(range(0, 10))
    assert module.resolve_v4_layer_ids(cfg, pp_rank=1) == tuple(range(10, 21))
    assert module.resolve_v4_layer_ids(cfg, pp_rank=2) == tuple(range(21, 32))
    assert module.resolve_v4_layer_ids(cfg, pp_rank=3) == tuple(range(32, 43))


def test_grouped_expert_ep_sharded_state_dict_uses_global_expert_axis():
    module = load_mcore_model_module()
    cfg = SimpleNamespace(
        num_local_experts=8,
        hidden_size=4,
        intermediate_size=3,
        hidden_act="silu",
        swiglu_limit=7.0,
    )
    experts = module.V4GroupedExperts(cfg, expert_model_parallel_size=4, expert_model_parallel_rank=2)

    assert tuple(experts.gate_up_proj.shape) == (2, 6, 4)
    assert tuple(experts.down_proj.shape) == (2, 4, 3)
    assert experts.gate_up_proj.allreduce is False
    assert experts.down_proj.allreduce is False

    state = experts.sharded_state_dict(prefix="layers.0.mlp.experts.")
    gate = state["layers.0.mlp.experts.gate_up_proj"]
    down = state["layers.0.mlp.experts.down_proj"]

    assert gate.key == "layers.0.mlp.experts.gate_up_proj"
    assert gate.local_shape == (2, 6, 4)
    assert gate.global_shape == (8, 6, 4)
    assert gate.global_offset == (4, 0, 0)
    assert gate.axis_fragmentations == (4, 1, 1)
    assert down.local_shape == (2, 4, 3)
    assert down.global_shape == (8, 4, 3)
    assert down.global_offset == (4, 0, 0)


def test_grouped_expert_ep1_params_remain_dense_allreduce():
    module = load_mcore_model_module()
    cfg = SimpleNamespace(
        num_local_experts=4,
        hidden_size=4,
        intermediate_size=3,
        hidden_act="silu",
        swiglu_limit=7.0,
    )
    experts = module.V4GroupedExperts(cfg)

    assert experts.gate_up_proj.allreduce is True
    assert experts.down_proj.allreduce is True


def test_grouped_expert_dispatched_compute_matches_legacy_permutation():
    module = load_mcore_model_module()
    from megatron.core.transformer.moe.moe_utils import permute, unpermute

    cfg = SimpleNamespace(
        num_local_experts=4,
        hidden_size=5,
        intermediate_size=3,
        hidden_act="silu",
        swiglu_limit=7.0,
    )
    torch.manual_seed(123)
    experts = module.V4GroupedExperts(cfg)
    with torch.no_grad():
        experts.gate_up_proj.normal_(mean=0.0, std=0.02)
        experts.down_proj.normal_(mean=0.0, std=0.02)
    hidden = torch.randn(6, cfg.hidden_size)
    indices = torch.tensor(
        [[0, 1], [2, 0], [3, 1], [1, 2], [0, 3], [2, 3]],
        dtype=torch.long,
    )
    weights = torch.rand(6, 2)

    legacy = experts(hidden, indices, weights)
    routing_map = torch.zeros(hidden.shape[0], cfg.num_local_experts, dtype=torch.bool)
    routing_map.scatter_(1, indices, True)
    probs = torch.zeros(hidden.shape[0], cfg.num_local_experts)
    probs = probs.scatter_add(1, indices, weights)
    tokens_per_expert = routing_map.sum(dim=0).long()
    permuted, permuted_probs, reverse = permute(hidden, routing_map, probs=probs)

    expert_output = experts.forward_dispatched(permuted, tokens_per_expert, permuted_probs)
    restored = unpermute(expert_output, reverse, restore_shape=hidden.shape)

    torch.testing.assert_close(restored, legacy, rtol=0, atol=1e-6)


def test_make_transformer_config_defaults_ep_to_deepep_flex():
    module = load_model_provider_module()
    cfg = SimpleNamespace(
        initializer_range=0.02,
        num_hidden_layers=2,
        hidden_size=8,
        num_attention_heads=2,
        head_dim=4,
        moe_intermediate_size=16,
        max_position_embeddings=32,
        num_local_experts=8,
        num_experts_per_tok=2,
    )

    mcore_cfg = module.make_transformer_config(cfg, expert_model_parallel_size=2)

    assert mcore_cfg.moe_token_dispatcher_type == "flex"
    assert mcore_cfg.moe_flex_dispatcher_backend == "deepep"
    assert mcore_cfg.moe_router_dtype == "fp32"


def test_make_transformer_config_rejects_v4_tensor_parallel():
    module = load_model_provider_module()
    cfg = SimpleNamespace(
        initializer_range=0.02,
        num_hidden_layers=2,
        hidden_size=8,
        num_attention_heads=2,
        head_dim=4,
        moe_intermediate_size=16,
        max_position_embeddings=32,
        num_local_experts=8,
        num_experts_per_tok=2,
    )

    with pytest.raises(ValueError, match="tensor_model_parallel_size=1"):
        module.make_transformer_config(cfg, tensor_model_parallel_size=2)
