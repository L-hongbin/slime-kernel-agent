from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

NUM_GPUS = 0


def install_megatron_stubs() -> None:
    if "megatron" in sys.modules:
        return

    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    models_mod = types.ModuleType("megatron.core.models")
    gpt_mod = types.ModuleType("megatron.core.models.gpt")
    gpt_layer_specs_mod = types.ModuleType("megatron.core.models.gpt.gpt_layer_specs")
    inference_mod = types.ModuleType("megatron.core.inference")
    inference_contexts_mod = types.ModuleType("megatron.core.inference.contexts")
    packed_seq_mod = types.ModuleType("megatron.core.packed_seq_params")
    transformer_mod = types.ModuleType("megatron.core.transformer")
    transformer_module_mod = types.ModuleType("megatron.core.transformer.module")
    attention_mod = types.ModuleType("megatron.core.transformer.attention")
    spec_utils_mod = types.ModuleType("megatron.core.transformer.spec_utils")
    transformer_block_mod = types.ModuleType("megatron.core.transformer.transformer_block")
    transformer_layer_mod = types.ModuleType("megatron.core.transformer.transformer_layer")

    class PackedSeqParams:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    class MegatronModule(nn.Module):
        def __init__(self, config=None):
            super().__init__()
            self.config = config

    class ModuleSpec:
        def __init__(self, module=None, params=None):
            self.module = module
            self.params = params or {}

    mpu_stub = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_group=lambda: None,
        get_context_parallel_rank=lambda: 0,
        get_tensor_model_parallel_group=lambda: None,
    )
    tensor_parallel_stub = types.SimpleNamespace(
        gather_from_sequence_parallel_region=lambda x, group=None: x,
        scatter_to_sequence_parallel_region=lambda x, group=None: x,
    )

    gpt_layer_specs_mod.get_gpt_decoder_block_spec = lambda *args, **kwargs: None
    inference_contexts_mod.BaseInferenceContext = type("BaseInferenceContext", (), {})
    packed_seq_mod.PackedSeqParams = PackedSeqParams
    transformer_module_mod.MegatronModule = MegatronModule
    attention_mod.SelfAttention = type("SelfAttention", (nn.Module,), {})
    spec_utils_mod.ModuleSpec = ModuleSpec
    transformer_block_mod.get_num_layers_to_build = lambda *args, **kwargs: 0
    transformer_layer_mod.get_transformer_layer_offset = lambda *args, **kwargs: 0

    core_mod.mpu = mpu_stub
    core_mod.tensor_parallel = tensor_parallel_stub

    sys.modules["megatron"] = megatron_mod
    sys.modules["megatron.core"] = core_mod
    sys.modules["megatron.core.models"] = models_mod
    sys.modules["megatron.core.models.gpt"] = gpt_mod
    sys.modules["megatron.core.models.gpt.gpt_layer_specs"] = gpt_layer_specs_mod
    sys.modules["megatron.core.inference"] = inference_mod
    sys.modules["megatron.core.inference.contexts"] = inference_contexts_mod
    sys.modules["megatron.core.packed_seq_params"] = packed_seq_mod
    sys.modules["megatron.core.transformer"] = transformer_mod
    sys.modules["megatron.core.transformer.module"] = transformer_module_mod
    sys.modules["megatron.core.transformer.attention"] = attention_mod
    sys.modules["megatron.core.transformer.spec_utils"] = spec_utils_mod
    sys.modules["megatron.core.transformer.transformer_block"] = transformer_block_mod
    sys.modules["megatron.core.transformer.transformer_layer"] = transformer_layer_mod


class FakeShortConvolution(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, cu_seqlens=None, **kwargs):
        return x, None


class FakeFusedRMSNormGated(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, z):
        return x


def make_config() -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=32,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        dtype=torch.float32,
    )


def load_module(module_name: str):
    install_megatron_stubs()
    sys.modules.pop("slime_plugins.models.hf_attention", None)
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.mark.unit
@pytest.mark.parametrize("module_name", ["qwen3_5", "qwen3_next"])
@pytest.mark.parametrize("output_gate", [False, True])
def test_qwen_spec_installs_gate_compat_only_on_full_attention(monkeypatch, module_name, output_gate):
    module = load_module(f"slime_plugins.models.{module_name}")
    original_attention = SimpleNamespace(
        module=module.SelfAttention,
        params={"attn_mask_type": "causal"},
        submodules=SimpleNamespace(linear_qkv="original_qkv", core_attention="original_core"),
    )
    original_layer = SimpleNamespace(submodules=SimpleNamespace(self_attention=original_attention))
    # Upstream specs can share objects: changing full attention must not leak to GDN.
    block = SimpleNamespace(layer_specs=[original_layer, original_layer])
    monkeypatch.setattr(module, "get_gpt_decoder_block_spec", lambda *args, **kwargs: block)
    monkeypatch.setattr(module, "get_num_layers_to_build", lambda *args, **kwargs: 2)
    monkeypatch.setattr(module, "get_transformer_layer_offset", lambda *args, **kwargs: 1)
    monkeypatch.setattr(
        module,
        "_load_hf_config",
        lambda _: SimpleNamespace(
            layer_types=["linear_attention", "full_attention", "linear_attention", "full_attention"]
        ),
    )
    args = SimpleNamespace(num_experts=None, hf_checkpoint="unused", qwen_gdn_implementation="replicated")
    config = SimpleNamespace(num_layers=4, pipeline_model_parallel_layout=None, attention_output_gate=output_gate)
    builder = getattr(module, f"get_{module_name}_spec")

    result = builder(args, config, vp_stage=0)

    full = result.layer_specs[0].submodules.self_attention
    linear = result.layer_specs[1].submodules.self_attention
    assert full.module is (module.TPGatedSelfAttention if output_gate else module.SelfAttention)
    assert full.params == original_attention.params
    assert full.submodules.linear_qkv == "original_qkv"
    assert full.submodules.core_attention == "original_core"
    assert linear.module is module.Attention
    assert original_attention.module is module.SelfAttention
    # Rebuilding from an already-compatible full-attention spec is idempotent.
    assert builder(args, config, vp_stage=0).layer_specs[0].submodules.self_attention.module is full.module


@pytest.mark.unit
@pytest.mark.parametrize("module_name", ["qwen3_5", "qwen3_next"])
def test_gate_compat_does_not_silently_replace_custom_attention(monkeypatch, module_name):
    module = load_module(f"slime_plugins.models.{module_name}")
    custom_attention = SimpleNamespace(module=object)
    block = SimpleNamespace(layer_specs=[SimpleNamespace(submodules=SimpleNamespace(self_attention=custom_attention))])
    monkeypatch.setattr(module, "get_gpt_decoder_block_spec", lambda *args, **kwargs: block)
    monkeypatch.setattr(module, "get_num_layers_to_build", lambda *args, **kwargs: 1)
    monkeypatch.setattr(module, "get_transformer_layer_offset", lambda *args, **kwargs: 0)
    monkeypatch.setattr(module, "_load_hf_config", lambda _: SimpleNamespace(layer_types=["full_attention"]))
    args = SimpleNamespace(num_experts=None, hf_checkpoint="unused", qwen_gdn_implementation="replicated")
    config = SimpleNamespace(num_layers=1, pipeline_model_parallel_layout=None, attention_output_gate=True)
    with pytest.raises(ValueError, match="SelfAttention module spec"):
        getattr(module, f"get_{module_name}_spec")(args, config, vp_stage=None)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("module_name", "class_name", "args", "expected_backend"),
    [
        ("slime_plugins.models.qwen3_5", "Qwen3_5GatedDeltaNet", None, "fla"),
        (
            "slime_plugins.models.qwen3_5",
            "Qwen3_5GatedDeltaNet",
            SimpleNamespace(qwen_gdn_backend="flashqla"),
            "flashqla",
        ),
        ("slime_plugins.models.qwen3_next", "Qwen3NextGatedDeltaNet", None, "fla"),
        (
            "slime_plugins.models.qwen3_next",
            "Qwen3NextGatedDeltaNet",
            SimpleNamespace(qwen_gdn_backend="flashqla"),
            "flashqla",
        ),
    ],
)
def test_linear_attention_forwards_cu_seqlens_to_chunk_kernel(
    monkeypatch,
    module_name: str,
    class_name: str,
    args,
    expected_backend: str,
):
    module = load_module(module_name)

    monkeypatch.setattr(module.accelerator, "current_device", lambda: "cpu")
    monkeypatch.setattr(module, "ShortConvolution", FakeShortConvolution, raising=False)
    monkeypatch.setattr(module, "FusedRMSNormGated", FakeFusedRMSNormGated, raising=False)

    chunk_calls = []
    selected_backends = []

    def fake_chunk_gated_delta_rule(
        q,
        k,
        v,
        *,
        g,
        beta,
        initial_state,
        output_final_state,
        use_qk_l2norm_in_kernel,
        cu_seqlens=None,
        **kwargs,
    ):
        chunk_calls.append(cu_seqlens.clone() if cu_seqlens is not None else None)
        assert q.shape[0] == 1
        assert cu_seqlens is not None
        return torch.zeros_like(v), None

    def fake_get_chunk_gated_delta_rule(backend):
        selected_backends.append(backend)
        return fake_chunk_gated_delta_rule

    monkeypatch.setattr(module, "get_chunk_gated_delta_rule", fake_get_chunk_gated_delta_rule)

    layer = getattr(module, class_name)(make_config(), layer_idx=0, args=args)
    hidden_states = torch.randn(1, 7, 32)
    cu_seqlens = torch.tensor([0, 3, 7], dtype=torch.int32)

    output = layer(hidden_states, cu_seqlens=cu_seqlens)

    assert selected_backends == [expected_backend]
    assert layer.gdn_backend == expected_backend
    assert output.shape == hidden_states.shape
    assert len(chunk_calls) == 1
    assert torch.equal(chunk_calls[0], cu_seqlens)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
