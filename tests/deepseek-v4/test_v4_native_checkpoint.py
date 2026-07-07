import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


def load_r2_module():
    module_path = (
        Path(__file__).resolve().parents[2] / "custom_kernels/deepseek_v4" / "megatron" / "native_checkpoint.py"
    )
    spec = importlib.util.spec_from_file_location("test_v4_native_checkpoint_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_native_v4_keys_map_to_mcore_names():
    module = load_r2_module()
    cases = {
        "embed.weight": "embedding.word_embeddings.weight",
        "head.weight": "output_layer.weight",
        "hc_head_fn": "hc_head.hc_fn",
        "layers.2.attn.wq_a.weight": "layers.2.self_attn.q_a_proj.weight",
        "layers.2.attn.q_norm.weight": "layers.2.self_attn.q_a_norm.weight",
        "layers.2.attn.compressor.ape": "layers.2.self_attn.compressor.position_bias",
        "layers.2.attn.indexer.compressor.wkv.weight": "layers.2.self_attn.compressor.indexer.kv_proj.weight",
        "layers.3.ffn.gate.bias": "layers.3.mlp.gate.e_score_correction_bias",
        "layers.0.ffn.gate.tid2eid": "layers.0.mlp.gate.tid2eid",
        "layers.4.ffn.shared_experts.w3.weight": "layers.4.mlp.shared_experts.up_proj.weight",
    }
    for native, mcore in cases.items():
        assert module.native_key_to_mcore(native) == mcore

    assert module.native_key_to_mcore("layers.0.attn.wq_a.scale") is None
    assert module.native_key_to_mcore("mtp.0.attn.wq_a.weight") is None


def test_native_v4_keys_map_to_hf_names_for_real_weight_parity():
    module = load_r2_module()
    cases = {
        "embed.weight": "model.embed_tokens.weight",
        "head.weight": "lm_head.weight",
        "hc_head_scale": "model.hc_head.hc_scale",
        "layers.2.attn.wq_a.weight": "model.layers.2.self_attn.q_a_proj.weight",
        "layers.2.attn.wo_b.weight": "model.layers.2.self_attn.o_b_proj.weight",
        "layers.2.attn.compressor.wgate.weight": "model.layers.2.self_attn.compressor.gate_proj.weight",
        "layers.2.attn.indexer.weights_proj.weight": "model.layers.2.self_attn.compressor.indexer.weights_proj.weight",
        "layers.3.ffn.gate.bias": "model.layers.3.mlp.gate.e_score_correction_bias",
        "layers.0.ffn.gate.tid2eid": "model.layers.0.mlp.gate.tid2eid",
        "layers.4.ffn.shared_experts.w2.weight": "model.layers.4.mlp.shared_experts.down_proj.weight",
    }
    for native, hf in cases.items():
        assert module.native_key_to_hf(native) == hf

    assert module.native_key_to_hf("layers.0.ffn.experts.0.w1.weight") is None
    assert module.native_key_to_hf("layers.0.attn.wq_a.scale") is None
    assert module.native_key_to_hf("mtp.0.attn.wq_a.weight") is None


def test_layer_key_remap_can_build_single_layer_slices_from_nonzero_source_layer():
    module = load_r2_module()

    assert (
        module.remap_mcore_layer_key(
            "layers.2.self_attn.compressor.indexer.q_b_proj.weight",
            {2: 0},
        )
        == "layers.0.self_attn.compressor.indexer.q_b_proj.weight"
    )
    assert module.remap_mcore_layer_key("layers.3.mlp.gate.weight", {2: 0}) is None
    assert module.remap_mcore_layer_key("output_layer.weight", {2: 0}) == "output_layer.weight"


def test_fp8_block_dequant_infers_block_grid_from_scale_shape():
    module = load_r2_module()
    weight = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    scale = torch.tensor([[0.5, 1.0], [2.0, 4.0]], dtype=torch.float32)

    got = module.dequant_fp8_block(weight, scale, output_dtype=torch.float32)

    expected = torch.tensor(
        [
            [0.0, 0.5, 2.0, 3.0],
            [2.0, 2.5, 6.0, 7.0],
            [16.0, 18.0, 40.0, 44.0],
            [24.0, 26.0, 56.0, 60.0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(got, expected)


def test_size_estimate_dtype_helpers_cover_safetensors_and_torch_dtype_names():
    module = load_r2_module()

    assert module.numel((2, 3, 4)) == 24
    assert module.dtype_nbytes("F8_E4M3") == 1
    assert module.dtype_nbytes("BF16") == 2
    assert module.dtype_nbytes("I64") == 8
    assert module.dtype_nbytes(torch.bfloat16) == 2
    assert module.is_floating_dtype_name("F8_E4M3")
    assert module.is_floating_dtype_name("BF16")
    assert not module.is_floating_dtype_name("I64")


def test_contiguous_pp_layer_ids_match_43_layer_pp4_uneven_plan():
    module = load_r2_module()

    assert module.contiguous_pp_layer_ids(
        43,
        4,
        num_layers_in_first_pipeline_stage=10,
        num_layers_in_last_pipeline_stage=11,
    ) == (
        tuple(range(0, 10)),
        tuple(range(10, 21)),
        tuple(range(21, 32)),
        tuple(range(32, 43)),
    )


class FakeCheckpoint:
    def __init__(self):
        self.tensors = {
            "layers.1.ffn.experts.7.w1.weight": torch.full((2, 3), 1.0),
            "layers.1.ffn.experts.7.w3.weight": torch.full((2, 3), 3.0),
            "layers.1.ffn.experts.7.w2.weight": torch.full((3, 2), 2.0),
        }

    def get_tensor(self, key):
        return self.tensors[key]


def test_expert_assembly_concatenates_w1_then_w3_for_gate_up():
    module = load_r2_module()
    got = module.read_expert_tensors(FakeCheckpoint(), 1, 7, output_dtype=torch.float32)

    assert torch.equal(
        got["layers.1.mlp.experts.gate_up_proj[7]"],
        torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [3.0, 3.0, 3.0],
                [3.0, 3.0, 3.0],
            ]
        ),
    )
    assert torch.equal(got["layers.1.mlp.experts.down_proj[7]"], torch.full((3, 2), 2.0))


def test_local_expert_global_ids_use_ep_rank_offset():
    module = load_r2_module()
    expert_module = SimpleNamespace(local_expert_start=4, local_expert_end=6)

    assert module._local_expert_global_ids(expert_module, 2) == (4, 5)
