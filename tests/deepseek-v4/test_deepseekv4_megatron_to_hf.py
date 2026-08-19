from types import SimpleNamespace

import torch

from slime.backends.megatron_utils.megatron_to_hf import convert_to_hf
from slime.backends.megatron_utils.megatron_to_hf.deepseekv4 import convert_deepseekv4_to_hf
from slime.backends.megatron_utils.megatron_to_hf.processors.quantizer_fp8 import quantize_params_fp8
from slime.backends.megatron_utils.update_weight.common import _maybe_v4_global_name
from slime.backends.megatron_utils.update_weight.update_weight_from_distributed import (
    _build_v4_lora_base_scales,
    _merge_lora_weight,
    _v4_incomplete_sglang_compressor_pairs,
    _v4_incomplete_sglang_wqkv_a_pairs,
)


def _args(**overrides):
    values = {
        "vocab_size": 8,
        "q_lora_rank": None,
        "num_experts": 256,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_deepseekv4_top_level_and_layer_names_convert_to_sglang_raw_names():
    tensor = torch.ones(2, 2)

    assert (
        convert_deepseekv4_to_hf(
            _args(),
            "module.module.embedding.word_embeddings.weight",
            torch.ones(10, 2),
        )[
            0
        ][0]
        == "embed.weight"
    )
    assert convert_deepseekv4_to_hf(_args(), "module.module.output_layer.weight", tensor)[0][0] == "head.weight"
    assert convert_deepseekv4_to_hf(_args(), "module.module.norm.weight", tensor)[0][0] == "norm.weight"
    assert convert_deepseekv4_to_hf(_args(), "module.module.hc_head.hc_fn", tensor)[0][0] == "hc_head_fn"

    cases = {
        "module.module.layers.3.input_layernorm.weight": "layers.3.attn_norm.weight",
        "module.module.layers.3.post_attention_layernorm.weight": "layers.3.ffn_norm.weight",
        "module.module.layers.3.attn_hc.fn": "layers.3.hc_attn_fn",
        "module.module.layers.3.ffn_hc.scale": "layers.3.hc_ffn_scale",
        "module.module.layers.3.self_attn.q_a_proj.weight": "layers.3.attn.wq_a.weight",
        "module.module.layers.3.self_attn.o_a_proj.weight": "layers.3.attn.wo_a.weight",
        "module.module.layers.3.self_attn.compressor.gate_proj.weight": ("layers.3.attn.compressor.wgate.weight"),
        "module.module.layers.3.self_attn.compressor.indexer.weights_proj.weight": (
            "layers.3.attn.indexer.weights_proj.weight"
        ),
        "module.module.layers.3.mlp.gate.e_score_correction_bias": "layers.3.ffn.gate.bias",
        "module.module.layers.3.mlp.shared_experts.up_proj.weight": "layers.3.ffn.shared_experts.w3.weight",
    }
    for megatron_name, expected in cases.items():
        assert convert_deepseekv4_to_hf(_args(), megatron_name, tensor)[0][0] == expected


def test_deepseekv4_lora_adapter_params_are_not_sent_to_sglang_base_model():
    got = convert_deepseekv4_to_hf(
        _args(),
        "module.module.layers.4.self_attn.q_a_proj.linear_in.weight",
        torch.ones(2, 2),
    )

    assert got == []


def test_deepseekv4_expert_range_splits_gate_up_and_down_to_raw_expert_names():
    gate_up = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    down = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)

    gate_outputs = convert_deepseekv4_to_hf(
        _args(),
        "module.module.layers.5.mlp.experts.gate_up_proj.expert_range32-34",
        gate_up,
    )
    down_outputs = convert_deepseekv4_to_hf(
        _args(),
        "module.module.layers.5.mlp.experts.down_proj.expert_range32-34",
        down,
    )

    assert [name for name, _ in gate_outputs] == [
        "layers.5.ffn.experts.32.w1.weight",
        "layers.5.ffn.experts.32.w3.weight",
        "layers.5.ffn.experts.33.w1.weight",
        "layers.5.ffn.experts.33.w3.weight",
    ]
    assert torch.equal(gate_outputs[0][1], gate_up[0, :2])
    assert torch.equal(gate_outputs[1][1], gate_up[0, 2:])
    assert [name for name, _ in down_outputs] == [
        "layers.5.ffn.experts.32.w2.weight",
        "layers.5.ffn.experts.33.w2.weight",
    ]
    assert torch.equal(down_outputs[1][1], down[1])


def test_deepseekv4_convert_to_hf_dispatches_model_name_and_removes_padding():
    param = torch.arange(20, dtype=torch.float32).reshape(10, 2)

    got = convert_to_hf(_args(vocab_size=6), "deepseekv4config", "module.module.output_layer.weight", param)

    assert len(got) == 1
    assert got[0][0] == "head.weight"
    assert torch.equal(got[0][1], param[:6])


def test_deepseekv4_raw_fp8_quantization_uses_wo_a_scale_name():
    quantization_config = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
    }
    converted = [
        ("layers.2.attn.wo_a.weight", torch.ones(2, 2)),
        ("layers.2.ffn.gate.weight", torch.ones(2, 2)),
        ("layers.2.attn.compressor.wkv.weight", torch.ones(2, 2, dtype=torch.bfloat16)),
        ("layers.2.attn.indexer.weights_proj.weight", torch.ones(2, 2, dtype=torch.bfloat16)),
        ("layers.2.attn.indexer.wq_b.weight", torch.ones(2, 2)),
    ]

    got = quantize_params_fp8(
        _args(), "module.module.layers.2.self_attn.o_a_proj.weight", converted, quantization_config
    )

    assert [name for name, _ in got] == [
        "layers.2.attn.wo_a.weight",
        "layers.2.attn.wo_a.scale",
        "layers.2.ffn.gate.weight",
        "layers.2.attn.compressor.wkv.weight",
        "layers.2.attn.indexer.weights_proj.weight",
        "layers.2.attn.indexer.wq_b.weight",
        "layers.2.attn.indexer.wq_b.weight_scale",
    ]
    assert got[0][1].dtype is torch.float8_e4m3fn
    assert got[1][1].dtype is torch.float32
    assert got[3][1].dtype is torch.bfloat16
    assert got[4][1].dtype is torch.bfloat16
    assert got[5][1].dtype is torch.float8_e4m3fn


def test_dsv4_global_name_maps_pp_local_layers_and_ep_expert_ranges():
    model_module = SimpleNamespace(layer_ids=(21, 22))
    layer_param = torch.ones(2, 2)
    expert_param = torch.ones(32, 4, 3)

    assert (
        _maybe_v4_global_name(
            _args(),
            model_module,
            "module.module.layers.1.self_attn.q_b_proj.weight",
            layer_param,
            expert_offset=None,
        )
        == "module.module.layers.22.self_attn.q_b_proj.weight"
    )
    assert (
        _maybe_v4_global_name(
            _args(),
            model_module,
            "module.module.layers.0.mlp.experts.gate_up_proj",
            expert_param,
            expert_offset=64,
        )
        == "module.module.layers.21.mlp.experts.gate_up_proj.expert_range64-96"
    )


def test_dsv4_global_name_unwraps_ddp_float16_module_chain():
    wrapped = SimpleNamespace(module=SimpleNamespace(module=SimpleNamespace(layer_ids=(21, 22))))

    assert (
        _maybe_v4_global_name(
            _args(),
            wrapped,
            "module.module.layers.1.self_attn.q_b_proj.weight",
            torch.ones(2, 2),
            expert_offset=None,
        )
        == "module.module.layers.22.self_attn.q_b_proj.weight"
    )


def test_dsv4_lora_merge_builds_effective_dense_weight():
    base = torch.zeros(2, 3, dtype=torch.bfloat16)
    lora_in = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.bfloat16)
    lora_out = torch.tensor([[1.0, 0.5], [2.0, 1.0]], dtype=torch.bfloat16)

    got = _merge_lora_weight(base, lora_in, lora_out, scale=0.25)
    expected = lora_out.float().matmul(lora_in.float()).mul(0.25).to(torch.bfloat16)

    assert got.dtype is torch.bfloat16
    torch.testing.assert_close(got, expected)


def test_dsv4_lora_base_scales_use_global_layer_names_through_wrappers():
    class FakeLora(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(2, 3))
            self.linear_in = torch.nn.Linear(3, 2, bias=False)
            self.linear_out = torch.nn.Linear(2, 2, bias=False)
            self.scale = 2.0

    layer0 = torch.nn.Module()
    layer0.self_attn = torch.nn.Module()
    layer0.self_attn.q_a_proj = FakeLora()
    layer1 = torch.nn.Module()
    layer1.self_attn = torch.nn.Module()
    layer1.self_attn.q_a_proj = FakeLora()

    v4 = torch.nn.Module()
    v4.layer_ids = (21, 22)
    v4.layers = torch.nn.ModuleList([layer0, layer1])
    float16_wrapper = torch.nn.Module()
    float16_wrapper.module = v4
    ddp_wrapper = torch.nn.Module()
    ddp_wrapper.module = float16_wrapper

    got = _build_v4_lora_base_scales(_args(), [ddp_wrapper])

    assert got == {
        "module.module.layers.21.self_attn.q_a_proj.weight": 2.0,
        "module.module.layers.22.self_attn.q_a_proj.weight": 2.0,
    }


def test_dsv4_lora_only_distributed_iterator_skips_frozen_base_and_experts(monkeypatch):
    from slime.backends.megatron_utils.update_weight import update_weight_from_distributed as upd

    base_name = "module.module.layers.21.self_attn.q_a_proj.weight"
    lora_in_name = "module.module.layers.21.self_attn.q_a_proj.linear_in.weight"
    lora_out_name = "module.module.layers.21.self_attn.q_a_proj.linear_out.weight"
    norm_name = "module.module.layers.21.input_layernorm.weight"
    expert_name = "module.module.layers.21.mlp.experts.gate_up_proj.expert_range0-1"
    named = [
        (base_name, torch.zeros(2, 3)),
        (lora_in_name, torch.ones(2, 3)),
        (lora_out_name, torch.ones(2, 2)),
        (norm_name, torch.ones(3)),
        (expert_name, torch.ones(1, 4, 3)),
    ]
    gathered = []

    monkeypatch.setattr(upd, "named_params_and_buffers", lambda _args, _model: iter(named))

    def fake_all_gather(name, param):
        gathered.append(name)
        return param

    monkeypatch.setattr(upd, "all_gather_param", fake_all_gather)
    monkeypatch.setattr(
        upd,
        "convert_to_hf",
        lambda _args, _model_name, name, param, _quantization_config: [(f"hf:{name}", param)],
    )

    updater = object.__new__(upd.UpdateWeightFromDistributed)
    updater.args = _args(update_weight_buffer_size=1 << 30)
    updater.model = []
    updater.model_name = "deepseekv4config"
    updater.quantization_config = None
    updater._is_pp_src_rank = True
    updater._v4_lora_base_scales = {base_name: 0.5}

    chunks = list(updater._iter_non_expert_chunks())

    assert [name for name, _ in chunks[0]] == [f"hf:{base_name}"]
    assert gathered == [base_name, lora_in_name, lora_out_name]
    assert list(updater._iter_expert_chunks()) == []


def test_dsv4_lora_only_iterator_keeps_sglang_compressor_pairs_in_same_chunk(monkeypatch):
    from slime.backends.megatron_utils.update_weight import update_weight_from_distributed as upd

    kv_base = "module.module.layers.21.self_attn.compressor.kv_proj.weight"
    kv_in = "module.module.layers.21.self_attn.compressor.kv_proj.linear_in.weight"
    kv_out = "module.module.layers.21.self_attn.compressor.kv_proj.linear_out.weight"
    gate_base = "module.module.layers.21.self_attn.compressor.gate_proj.weight"
    gate_in = "module.module.layers.21.self_attn.compressor.gate_proj.linear_in.weight"
    gate_out = "module.module.layers.21.self_attn.compressor.gate_proj.linear_out.weight"
    named = [
        (kv_base, torch.zeros(2, 3)),
        (kv_in, torch.ones(2, 3)),
        (kv_out, torch.ones(2, 2)),
        (gate_base, torch.zeros(2, 3)),
        (gate_in, torch.ones(2, 3)),
        (gate_out, torch.ones(2, 2)),
    ]

    monkeypatch.setattr(upd, "named_params_and_buffers", lambda _args, _model: iter(named))
    monkeypatch.setattr(upd, "all_gather_param", lambda _name, param: param)

    def fake_convert(_args, _model_name, name, param, _quantization_config):
        if name == kv_base:
            return [("layers.21.attn.compressor.wkv.weight", param)]
        if name == gate_base:
            return [("layers.21.attn.compressor.wgate.weight", param)]
        raise AssertionError(name)

    monkeypatch.setattr(upd, "convert_to_hf", fake_convert)

    updater = object.__new__(upd.UpdateWeightFromDistributed)
    updater.args = _args(update_weight_buffer_size=1)
    updater.model = []
    updater.model_name = "deepseekv4config"
    updater.quantization_config = None
    updater._is_pp_src_rank = True
    updater._v4_lora_base_scales = {kv_base: 0.5, gate_base: 0.5}

    chunks = list(updater._iter_non_expert_chunks())

    assert [[name for name, _ in chunk] for chunk in chunks] == [
        [
            "layers.21.attn.compressor.wkv.weight",
            "layers.21.attn.compressor.wgate.weight",
        ]
    ]


def test_dsv4_sglang_compressor_pair_detector_tracks_loader_constraint():
    assert _v4_incomplete_sglang_compressor_pairs(
        [
            ("layers.21.attn.compressor.wkv.weight", torch.ones(1)),
            ("layers.21.attn.compressor.wkv.weight_scale_inv", torch.ones(1)),
        ]
    ) == {"layers.21.attn.compressor": {"wkv"}}

    assert (
        _v4_incomplete_sglang_compressor_pairs(
            [
                ("layers.21.attn.compressor.wkv.weight", torch.ones(1)),
                ("layers.21.attn.compressor.wgate.weight", torch.ones(1)),
                ("layers.21.attn.indexer.compressor.wkv.weight", torch.ones(1)),
                ("layers.21.attn.indexer.compressor.wgate.weight", torch.ones(1)),
            ]
        )
        == {}
    )


def test_dsv4_sglang_wqkv_a_pair_detector_tracks_loader_constraint():
    assert _v4_incomplete_sglang_wqkv_a_pairs(
        [
            ("layers.27.attn.wq_a.weight", torch.ones(1)),
            ("layers.27.attn.wq_a.weight_scale_inv", torch.ones(1)),
        ]
    ) == {
        "layers.27.attn.wqkv_a.weight": {"q"},
        "layers.27.attn.wqkv_a.weight_scale_inv": {"q"},
    }

    assert (
        _v4_incomplete_sglang_wqkv_a_pairs(
            [
                ("layers.27.attn.wq_a.weight", torch.ones(1)),
                ("layers.27.attn.wkv.weight", torch.ones(1)),
                ("layers.27.attn.wq_a.weight_scale_inv", torch.ones(1)),
                ("layers.27.attn.wkv.weight_scale_inv", torch.ones(1)),
            ]
        )
        == {}
    )


def test_dsv4_chunks_keep_sglang_fused_wq_a_wkv_together(monkeypatch):
    from slime.backends.megatron_utils.update_weight import update_weight_from_distributed as upd

    q_name = "module.module.layers.27.self_attn.q_a_proj.weight"
    kv_name = "module.module.layers.27.self_attn.kv_proj.weight"
    norm_name = "module.module.layers.27.input_layernorm.weight"
    named = [
        (norm_name, torch.zeros(1)),
        (q_name, torch.zeros(1, 1)),
        (kv_name, torch.zeros(1, 1)),
    ]

    monkeypatch.setattr(upd, "named_params_and_buffers", lambda _args, _model: iter(named))
    monkeypatch.setattr(upd, "all_gather_param", lambda _name, param: param)

    def fake_convert(_args, _model_name, name, param, _quantization_config):
        if name == norm_name:
            return [("layers.27.attn_norm.weight", param)]
        if name == q_name:
            return [
                ("layers.27.attn.wq_a.weight", param),
                ("layers.27.attn.wq_a.weight_scale_inv", torch.ones(1)),
            ]
        if name == kv_name:
            return [
                ("layers.27.attn.wkv.weight", param),
                ("layers.27.attn.wkv.weight_scale_inv", torch.ones(1)),
            ]
        raise AssertionError(name)

    monkeypatch.setattr(upd, "convert_to_hf", fake_convert)

    updater = object.__new__(upd.UpdateWeightFromDistributed)
    updater.args = _args(update_weight_buffer_size=1)
    updater.model = []
    updater.model_name = "deepseekv4config"
    updater.quantization_config = None
    updater._is_pp_src_rank = True
    updater._v4_lora_base_scales = {}

    chunks = list(updater._iter_non_expert_chunks())

    assert [[name for name, _ in chunk] for chunk in chunks] == [
        ["layers.27.attn_norm.weight"],
        [
            "layers.27.attn.wq_a.weight",
            "layers.27.attn.wq_a.weight_scale_inv",
            "layers.27.attn.wkv.weight",
            "layers.27.attn.wkv.weight_scale_inv",
        ],
    ]


def test_dsv4_lora_only_chunks_keep_sglang_compressor_wkv_wgate_together(monkeypatch):
    from slime.backends.megatron_utils.update_weight import update_weight_from_distributed as upd

    q_name = "module.module.layers.21.self_attn.q_a_proj.weight"
    q_in_name = "module.module.layers.21.self_attn.q_a_proj.linear_in.weight"
    q_out_name = "module.module.layers.21.self_attn.q_a_proj.linear_out.weight"
    wkv_name = "module.module.layers.21.self_attn.kv_proj.weight"
    wkv_in_name = "module.module.layers.21.self_attn.kv_proj.linear_in.weight"
    wkv_out_name = "module.module.layers.21.self_attn.kv_proj.linear_out.weight"
    kv_name = "module.module.layers.21.self_attn.compressor.kv_proj.weight"
    kv_in_name = "module.module.layers.21.self_attn.compressor.kv_proj.linear_in.weight"
    kv_out_name = "module.module.layers.21.self_attn.compressor.kv_proj.linear_out.weight"
    gate_name = "module.module.layers.21.self_attn.compressor.gate_proj.weight"
    gate_in_name = "module.module.layers.21.self_attn.compressor.gate_proj.linear_in.weight"
    gate_out_name = "module.module.layers.21.self_attn.compressor.gate_proj.linear_out.weight"

    named = [
        (q_name, torch.zeros(1, 1)),
        (q_in_name, torch.ones(1, 1)),
        (q_out_name, torch.ones(1, 1)),
        (wkv_name, torch.zeros(1, 1)),
        (wkv_in_name, torch.ones(1, 1)),
        (wkv_out_name, torch.ones(1, 1)),
        (kv_name, torch.zeros(1, 1)),
        (kv_in_name, torch.ones(1, 1)),
        (kv_out_name, torch.ones(1, 1)),
        (gate_name, torch.zeros(1, 1)),
        (gate_in_name, torch.ones(1, 1)),
        (gate_out_name, torch.ones(1, 1)),
    ]

    monkeypatch.setattr(upd, "named_params_and_buffers", lambda _args, _model: iter(named))
    monkeypatch.setattr(upd, "all_gather_param", lambda _name, param: param)

    def fake_convert(_args, _model_name, name, param, _quantization_config):
        if name == q_name:
            return [("layers.21.attn.wq_a.weight", param)]
        if name == wkv_name:
            return [("layers.21.attn.wkv.weight", param)]
        if name == kv_name:
            return [
                ("layers.21.attn.compressor.wkv.weight", param),
                ("layers.21.attn.compressor.wkv.weight_scale_inv", torch.ones(1)),
            ]
        if name == gate_name:
            return [
                ("layers.21.attn.compressor.wgate.weight", param),
                ("layers.21.attn.compressor.wgate.weight_scale_inv", torch.ones(1)),
            ]
        raise AssertionError(name)

    monkeypatch.setattr(upd, "convert_to_hf", fake_convert)

    updater = object.__new__(upd.UpdateWeightFromDistributed)
    updater.args = _args(update_weight_buffer_size=1)
    updater.model = []
    updater.model_name = "deepseekv4config"
    updater.quantization_config = None
    updater._is_pp_src_rank = True
    updater._v4_lora_base_scales = {
        q_name: 0.5,
        wkv_name: 0.5,
        kv_name: 0.5,
        gate_name: 0.5,
    }

    chunks = list(updater._iter_non_expert_chunks())

    assert [[name for name, _ in chunk] for chunk in chunks] == [
        [
            "layers.21.attn.wq_a.weight",
            "layers.21.attn.wkv.weight",
        ],
        [
            "layers.21.attn.compressor.wkv.weight",
            "layers.21.attn.compressor.wkv.weight_scale_inv",
            "layers.21.attn.compressor.wgate.weight",
            "layers.21.attn.compressor.wgate.weight_scale_inv",
        ],
    ]
