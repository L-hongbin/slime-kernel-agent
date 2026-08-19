"""Unit tests for block-method activation recompute in the V4 decoder loop.

Block semantics (Megatron): the FIRST ``recompute_num_layers`` layers are
checkpointed individually (recomputed in backward); the remaining layers store
their activations. Verifies:
  1. outputs and gradients are bitwise-identical across off/uniform/block;
  2. block actually recomputes exactly the first-K layers (forward-call counts);
  3. invalid method rejected.

CPU-only. Decoder numerics use an extracted helper plus dummy layers; provider
tests import the installed Megatron package but construct no model or process group.
"""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
MEGATRON_LM = Path("/root/Megatron-LM")
if MEGATRON_LM.exists():
    sys.path.insert(0, str(MEGATRON_LM))

NUM_GPUS = 0


def _load_forward_fn():
    """Import _forward_v4_decoder_layers without pulling in Megatron deps."""

    src = (REPO / "custom_kernels/deepseek_v4/megatron/mcore_model.py").read_text()
    # Extract just the function (it only needs torch); executing the whole
    # module requires megatron. The function is self-contained by design.
    start = src.index("def _forward_v4_decoder_layers(")
    end = src.index("\ndef ", start + 1)
    ns = {"torch": torch}
    exec(compile(src[start:end], "mcore_model_extract", "exec"), ns)
    return ns["_forward_v4_decoder_layers"]


_forward_v4_decoder_layers = _load_forward_fn()


def _load_provider_module():
    return importlib.import_module("custom_kernels.deepseek_v4.megatron.model_provider")


class DummyLayer(nn.Module):
    """Matches the V4 decoder layer call signature; counts forward calls."""

    calls = None  # set per-test to a shared list

    def __init__(self, dim, idx):
        super().__init__()
        self.idx = idx
        self.lin = nn.Linear(dim, dim)

    def forward(self, hidden, positions, tokens):
        if DummyLayer.calls is not None:
            DummyLayer.calls.append(self.idx)
        return torch.tanh(self.lin(hidden)) + 0.1 * positions


def _run(method, num_layers, use_act_ckpt=True, use_reentrant=True, seed=0):
    torch.manual_seed(seed)
    dim, n_layers = 8, 5
    layers = nn.ModuleList(DummyLayer(dim, i) for i in range(n_layers))
    hidden = torch.randn(2, 3, dim, requires_grad=True)
    positions = torch.randn(2, 3, dim)
    tokens = torch.zeros(2, 3, dtype=torch.long)

    out = _forward_v4_decoder_layers(
        layers,
        hidden,
        positions,
        tokens,
        use_act_ckpt=use_act_ckpt,
        recompute_num_layers=num_layers,
        recompute_method=method,
        use_reentrant=use_reentrant,
    )
    out.sum().backward()
    grads = [p.grad.clone() for p in layers.parameters()] + [hidden.grad.clone()]
    return out.detach(), grads


@pytest.mark.parametrize("use_reentrant", [True, False])
@pytest.mark.parametrize("method,num_layers", [("uniform", 1), ("uniform", 2), ("block", 2), ("block", 5)])
def test_outputs_and_grads_match_no_ckpt(method, num_layers, use_reentrant):
    ref_out, ref_grads = _run("uniform", 1, use_act_ckpt=False)
    out, grads = _run(method, num_layers, use_act_ckpt=True, use_reentrant=use_reentrant)
    assert torch.equal(out, ref_out)
    for g, rg in zip(grads, ref_grads, strict=True):
        assert torch.equal(g, rg)


def test_block_recomputes_exactly_first_k():
    DummyLayer.calls = []
    try:
        _run("block", 2)
        counts = {i: DummyLayer.calls.count(i) for i in range(5)}
        # forward once each (5 layers) + backward recompute of layers 0..1 only
        assert counts == {0: 2, 1: 2, 2: 1, 3: 1, 4: 1}
    finally:
        DummyLayer.calls = None


def test_uniform_recomputes_all_layers():
    DummyLayer.calls = []
    try:
        _run("uniform", 2)
        counts = {i: DummyLayer.calls.count(i) for i in range(5)}
        assert counts == {i: 2 for i in range(5)}
    finally:
        DummyLayer.calls = None


def test_invalid_method_rejected():
    with pytest.raises(ValueError, match="uniform.*block|block.*uniform"):
        _run("banana", 1)


def test_invalid_num_layers_rejected():
    with pytest.raises(ValueError):
        _run("block", 0)
    with pytest.raises(TypeError):
        _run("block", True)
    with pytest.raises(ValueError, match="cannot exceed.*local decoder"):
        _run("block", 6)


@pytest.mark.parametrize("method", ["uniform", "block"])
def test_provider_recompute_contract_accepts_full_layer_methods(method):
    module = _load_provider_module()
    assert module._v4_full_recompute_enabled("full", method, 2) is True


def test_provider_recompute_contract_accepts_explicit_off():
    module = _load_provider_module()
    assert module._v4_full_recompute_enabled(None, None, None) is False


@pytest.mark.parametrize(
    "granularity,method,num_layers,error,match",
    [
        ("selective", None, None, ValueError, "submodule-selective"),
        ("unknown", "uniform", 1, ValueError, "recompute_granularity"),
        ("full", None, 1, ValueError, "uniform or block"),
        ("full", "unknown", 1, ValueError, "uniform or block"),
        ("full", "block", None, TypeError, "integer"),
        ("full", "block", True, TypeError, "integer"),
        ("full", "block", 0, ValueError, "greater than zero"),
        (None, "block", 1, ValueError, "granularity full"),
    ],
)
def test_provider_recompute_contract_rejects_unsupported_or_ambiguous_configs(
    granularity, method, num_layers, error, match
):
    module = _load_provider_module()
    with pytest.raises(error, match=match):
        module._v4_full_recompute_enabled(granularity, method, num_layers)


def _call_provider(monkeypatch, *, granularity, method, num_layers):
    module = _load_provider_module()
    import megatron.training
    from transformers import AutoConfig

    args = SimpleNamespace(
        hf_checkpoint="unused-test-checkpoint",
        expert_model_parallel_size=1,
        mtp_num_layers=0,
        recompute_granularity=granularity,
        recompute_method=method,
        recompute_num_layers=num_layers,
        params_dtype=torch.bfloat16,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        mtp_loss_scaling_factor=0.2,
    )
    hf_config = object()
    built_model = object()
    captured = {}

    monkeypatch.setattr(megatron.training, "get_args", lambda: args)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *_args, **_kwargs: hf_config)

    def fake_build(config, **kwargs):
        captured["config"] = config
        captured.update(kwargs)
        return built_model

    monkeypatch.setattr(module, "build_v4_mcore_model", fake_build)
    monkeypatch.setattr(module, "_v4_lora_cfg", lambda _args: (0, 0, 0.0))
    result = module.v4_model_provider()
    return module, result, built_model, captured


def test_provider_connects_block_args_to_model_config(monkeypatch):
    _module, result, built_model, captured = _call_provider(
        monkeypatch,
        granularity="full",
        method="block",
        num_layers=3,
    )

    assert result is built_model
    assert captured["config"] is not None
    assert captured["recompute_granularity"] == "full"
    assert captured["recompute_method"] == "block"
    assert captured["recompute_num_layers"] == 3
    assert captured["context_parallel_size"] == 1


def test_provider_explicit_off_is_preserved_in_model_config(monkeypatch):
    _module, _result, _built_model, captured = _call_provider(
        monkeypatch,
        granularity=None,
        method=None,
        num_layers=None,
    )
    assert captured["recompute_granularity"] is None
    assert captured["recompute_method"] is None
    assert captured["recompute_num_layers"] is None


def test_transformer_config_preserves_block_recompute_arguments():
    module = _load_provider_module()
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
        recompute_method="block",
        recompute_num_layers=3,
    )

    assert config.recompute_granularity == "full"
    assert config.recompute_method == "block"
    assert config.recompute_num_layers == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
