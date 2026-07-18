import sys
import types

import torch


def _install_megatron_stubs():
    megatron = types.ModuleType("megatron")
    core = types.ModuleType("megatron.core")

    tensor_parallel = types.ModuleType("megatron.core.tensor_parallel")
    tensor_parallel.gather_from_sequence_parallel_region = lambda logits, tensor_parallel_output_grad=False: logits

    gpt = types.ModuleType("megatron.core.models.gpt")

    class GPTModel(torch.nn.Module):
        pass

    gpt.GPTModel = GPTModel

    gpt_layer_specs = types.ModuleType("megatron.core.models.gpt.gpt_layer_specs")
    gpt_layer_specs.get_gpt_decoder_block_spec = lambda *args, **kwargs: None
    gpt_layer_specs.get_gpt_layer_local_spec = lambda *args, **kwargs: None
    gpt_layer_specs.get_gpt_layer_with_transformer_engine_spec = lambda *args, **kwargs: None

    spec_utils = types.ModuleType("megatron.core.transformer.spec_utils")
    spec_utils.import_module = lambda path: None

    transformer_config = types.ModuleType("megatron.core.transformer.transformer_config")

    class TransformerConfig:
        sequence_parallel = False

    transformer_config.TransformerConfig = TransformerConfig

    training = types.ModuleType("megatron.training")
    training_arguments = types.ModuleType("megatron.training.arguments")
    training_arguments.core_transformer_config_from_args = lambda args: TransformerConfig()

    sys.modules.update(
        {
            "megatron": megatron,
            "megatron.core": core,
            "megatron.core.tensor_parallel": tensor_parallel,
            "megatron.core.models": types.ModuleType("megatron.core.models"),
            "megatron.core.models.gpt": gpt,
            "megatron.core.models.gpt.gpt_layer_specs": gpt_layer_specs,
            "megatron.core.transformer": types.ModuleType("megatron.core.transformer"),
            "megatron.core.transformer.spec_utils": spec_utils,
            "megatron.core.transformer.transformer_config": transformer_config,
            "megatron.training": training,
            "megatron.training.arguments": training_arguments,
        }
    )


_install_megatron_stubs()
from slime.backends.megatron_utils.model_provider import _enable_actor_fp32_lm_head


class DummyOutputLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(2, 3, dtype=torch.bfloat16))
        self.forward_calls = 0
        self.last_input_dtype = None
        self.last_weight_dtype = None

    def forward(self, input_, weight=None, runtime_gather_output=None):
        self.forward_calls += 1
        self.last_input_dtype = input_.dtype
        weight = self.weight if weight is None else weight
        self.last_weight_dtype = weight.dtype
        return input_.matmul(weight.t()), None


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.output_layer = DummyOutputLayer()


def test_enable_actor_fp32_lm_head_casts_weight_input_and_output():
    model = DummyModel()

    _enable_actor_fp32_lm_head(model)
    logits, bias = model.output_layer(torch.randn(4, 3, dtype=torch.bfloat16))

    assert bias is None
    assert model.output_layer.weight.dtype == torch.float32
    assert model.output_layer.last_input_dtype == torch.float32
    assert model.output_layer.last_weight_dtype == torch.float32
    assert logits.dtype == torch.float32


def test_enable_actor_fp32_lm_head_is_idempotent():
    model = DummyModel()

    _enable_actor_fp32_lm_head(model)
    first_forward = model.output_layer.forward
    _enable_actor_fp32_lm_head(model)

    assert model.output_layer.forward == first_forward
