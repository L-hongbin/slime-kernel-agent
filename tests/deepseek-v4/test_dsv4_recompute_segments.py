import copy
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


NUM_GPUS = 0


def _load_module(name):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    megatron_lm = Path("/root/Megatron-LM")
    if megatron_lm.exists():
        sys.path.insert(0, str(megatron_lm))
    return importlib.import_module(name)


class _TraceLayer(nn.Module):
    def __init__(self, layer_id, trace):
        super().__init__()
        self.layer_id = layer_id
        self.trace = trace

    def forward(self, hidden, position_embeddings, input_ids):
        self.trace.append(self.layer_id)
        return hidden + self.layer_id + position_embeddings * 0


class _CountingLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)
        self.calls = 0

    def forward(self, hidden, position_embeddings, input_ids):
        self.calls += 1
        return torch.tanh(self.proj(hidden) + position_embeddings)


def _run_layers(module, layers, hidden, positions, tokens, *, checkpoint, segment_size):
    return module._forward_v4_decoder_layers(
        layers,
        hidden,
        positions,
        tokens,
        use_act_ckpt=checkpoint,
        recompute_num_layers=segment_size,
        use_reentrant=True,
    )


def test_uniform_recompute_groups_layers_as_two_two_one(monkeypatch):
    module = _load_module("custom_kernels.deepseek_v4.megatron.mcore_model")
    trace = []
    checkpointed_segments = []
    layers = nn.ModuleList(_TraceLayer(i, trace) for i in range(5))

    def fake_checkpoint(function, *args, **kwargs):
        start = len(trace)
        output = function(*args)
        checkpointed_segments.append(tuple(trace[start:]))
        return output

    monkeypatch.setattr(module.torch.utils.checkpoint, "checkpoint", fake_checkpoint)
    output = _run_layers(
        module,
        layers,
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1, dtype=torch.long),
        checkpoint=True,
        segment_size=2,
    )

    assert checkpointed_segments == [(0, 1), (2, 3), (4,)]
    assert output.item() == sum(range(5))


@pytest.mark.parametrize("segment_size", [1, 2, 8])
def test_recompute_segment_output_and_gradients_match_eager(segment_size):
    module = _load_module("custom_kernels.deepseek_v4.megatron.mcore_model")
    torch.manual_seed(1234)
    eager_layers = nn.ModuleList(_CountingLayer(4) for _ in range(5))
    checkpoint_layers = copy.deepcopy(eager_layers)
    positions = torch.randn(3, 4)
    tokens = torch.zeros(3, dtype=torch.long)
    eager_input = torch.randn(3, 4, requires_grad=True)
    checkpoint_input = eager_input.detach().clone().requires_grad_(True)

    eager_output = _run_layers(
        module,
        eager_layers,
        eager_input,
        positions,
        tokens,
        checkpoint=False,
        segment_size=segment_size,
    )
    checkpoint_output = _run_layers(
        module,
        checkpoint_layers,
        checkpoint_input,
        positions,
        tokens,
        checkpoint=True,
        segment_size=segment_size,
    )
    eager_output.sum().backward()
    checkpoint_output.sum().backward()

    torch.testing.assert_close(checkpoint_output, eager_output)
    torch.testing.assert_close(checkpoint_input.grad, eager_input.grad)
    for checkpoint_layer, eager_layer in zip(checkpoint_layers, eager_layers, strict=True):
        torch.testing.assert_close(checkpoint_layer.proj.weight.grad, eager_layer.proj.weight.grad)
        torch.testing.assert_close(checkpoint_layer.proj.bias.grad, eager_layer.proj.bias.grad)


def test_recompute_backward_reexecutes_every_layer_once_with_tail_segment():
    module = _load_module("custom_kernels.deepseek_v4.megatron.mcore_model")
    layers = nn.ModuleList(_CountingLayer(4) for _ in range(5))
    hidden = torch.randn(3, 4, requires_grad=True)

    output = _run_layers(
        module,
        layers,
        hidden,
        torch.randn(3, 4),
        torch.zeros(3, dtype=torch.long),
        checkpoint=True,
        segment_size=2,
    )
    assert [layer.calls for layer in layers] == [1, 1, 1, 1, 1]

    output.sum().backward()
    assert [layer.calls for layer in layers] == [2, 2, 2, 2, 2]


@pytest.mark.parametrize("segment_size, error", [(0, ValueError), (True, TypeError), (1.5, TypeError)])
def test_recompute_segment_size_must_be_a_positive_integer(segment_size, error):
    module = _load_module("custom_kernels.deepseek_v4.megatron.mcore_model")

    with pytest.raises(error, match="recompute_num_layers"):
        _run_layers(
            module,
            nn.ModuleList([_CountingLayer(2)]),
            torch.randn(1, 2, requires_grad=True),
            torch.zeros(1, 2),
            torch.zeros(1, dtype=torch.long),
            checkpoint=True,
            segment_size=segment_size,
        )


def test_transformer_config_preserves_uniform_recompute_arguments():
    module = _load_module("custom_kernels.deepseek_v4.megatron.model_provider")
    hf_config = SimpleNamespace(
        initializer_range=0.02,
        num_hidden_layers=5,
        hidden_size=8,
        num_attention_heads=2,
        head_dim=4,
        moe_intermediate_size=16,
        max_position_embeddings=32,
    )

    config = module.make_transformer_config(
        hf_config,
        recompute_granularity="full",
        recompute_method="uniform",
        recompute_num_layers=2,
    )

    assert config.recompute_granularity == "full"
    assert config.recompute_method == "uniform"
    assert config.recompute_num_layers == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
