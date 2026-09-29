"""CPU contracts for old/new Megatron gated-attention TP layouts."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

NUM_GPUS = 0
REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def attention_cls(monkeypatch):
    # Load only the compatibility module: CI does not need Megatron, TE or CUDA.
    module = ModuleType("megatron.core.transformer.attention")

    class SelfAttention(torch.nn.Module):
        def get_query_key_value_tensors(self, hidden_states, key_value_states=None, output_gate=False, split_qkv=True):
            self.call = (hidden_states, key_value_states, output_gate, split_qkv)
            self.upstream_output = self.upstream(hidden_states)
            return self.upstream_output

    module.SelfAttention = SelfAttention
    monkeypatch.setitem(sys.modules, module.__name__, module)
    spec = importlib.util.spec_from_file_location(
        "_test_gated_attention_compat", REPO_ROOT / "slime_plugins/models/gated_attention.py"
    )
    compat = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compat)
    return compat.TPGatedSelfAttention


def make_attention(cls, tp, rank, head_dim=8):
    attention = cls()
    attention.world_size = tp
    attention.config = SimpleNamespace(num_attention_heads=24, num_query_groups=4)
    attention.hidden_size_per_attention_head = head_dim
    attention.pg_collection = SimpleNamespace(tp=SimpleNamespace(rank=lambda: rank))
    return attention


def upstream_layout(mixed, tp, rank, *, fixed):
    """Model the gather/group selection, then old or fixed head slicing.

    `mixed` is the differentiable full gathered Q/G/K/V projection. No real
    collectives run here; every simulated rank selects its actual KV group.
    """
    head_dim = mixed.shape[-1] // 14  # 6 Q + 6 gate + 1 K + 1 V heads per group.
    group = mixed[:, :, rank // (tp // 4) : rank // (tp // 4) + 1]
    query, gate, key, value = group.split([6 * head_dim, 6 * head_dim, head_dim, head_dim], dim=-1)
    query = query.reshape(*mixed.shape[:2], 6, head_dim)
    gate = gate.reshape(*mixed.shape[:2], 6, head_dim)
    heads = 24 // tp
    start = (rank % (tp // 4)) * heads
    query = query[:, :, start : start + heads]
    if fixed:
        gate = gate[:, :, start : start + heads]
    return query, key, value, gate


@pytest.mark.unit
@pytest.mark.parametrize("fixed", [False, True], ids=["old_megatron", "fixed_megatron"])
@pytest.mark.parametrize("tp,rank", [(tp, rank) for tp in (4, 8) for rank in range(tp)])
def test_tp_gate_values_and_gradients(attention_cls, fixed, tp, rank):
    attention = make_attention(attention_cls, tp, rank)
    attention.upstream = lambda mixed: upstream_layout(mixed, tp, rank, fixed=fixed)
    generator = torch.Generator().manual_seed(123)
    # Include a real projection so gradients cover both hidden states and QKV weights.
    hidden = torch.randn(2, 1, 5, generator=generator, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(4 * 14 * 8, 5, generator=generator, dtype=torch.float64, requires_grad=True)
    mixed = torch.nn.functional.linear(hidden, weight).reshape(2, 1, 4, 14 * 8)
    mixed.retain_grad()
    query, key, value, gate = attention.get_query_key_value_tensors(mixed, output_gate=True)

    # Independent reference: flatten all groups and select global query-head indices.
    ref_hidden = hidden.detach().clone().requires_grad_()
    ref_weight = weight.detach().clone().requires_grad_()
    ref_mixed = torch.nn.functional.linear(ref_hidden, ref_weight).reshape(2, 1, 4, 14 * 8)
    ref_mixed.retain_grad()
    heads = 24 // tp
    q_ref = ref_mixed[..., :48].reshape(2, 1, 24, 8)[:, :, rank * heads : (rank + 1) * heads]
    g_ref = ref_mixed[..., 48:96].reshape(2, 1, 24, 8)[:, :, rank * heads : (rank + 1) * heads]
    group_index = rank // (tp // 4)
    k_ref = ref_mixed[:, :, group_index : group_index + 1, 96:104]
    v_ref = ref_mixed[:, :, group_index : group_index + 1, 104:112]
    assert query.shape == gate.shape == (2, 1, heads, 8)
    for actual, expected in zip((query, key, value, gate), (q_ref, k_ref, v_ref, g_ref), strict=True):
        torch.testing.assert_close(actual, expected)
    assert all(actual is original for actual, original in zip((query, key, value), attention.upstream_output[:3]))
    if fixed or tp == 4:
        assert gate is attention.upstream_output[3]
    else:
        assert gate.untyped_storage().data_ptr() == attention.upstream_output[3].untyped_storage().data_ptr()

    # Same flattened gate/view contract as Megatron's _apply_output_gate.
    out = query.reshape(2, 1, -1) * gate.contiguous().view(2, 1, -1).sigmoid()
    ref_out = q_ref.reshape(2, 1, -1) * g_ref.contiguous().view(2, 1, -1).sigmoid()
    (out.square().sum() + key.sin().sum() + value.cos().sum()).backward()
    (ref_out.square().sum() + k_ref.sin().sum() + v_ref.cos().sum()).backward()
    torch.testing.assert_close(mixed.grad, ref_mixed.grad)
    torch.testing.assert_close(hidden.grad, ref_hidden.grad)
    torch.testing.assert_close(weight.grad, ref_weight.grad)
    # Unselected gate heads must receive exactly zero gradient on this rank.
    global_gate_grad = mixed.grad[..., 48:96].reshape(2, 1, 24, 8)
    assert torch.count_nonzero(global_gate_grad[:, :, : rank * heads]) == 0
    assert torch.count_nonzero(global_gate_grad[:, :, (rank + 1) * heads :]) == 0


@pytest.mark.unit
@pytest.mark.parametrize("split_qkv", [False, True])
def test_ungated_api_is_unchanged(attention_cls, split_qkv):
    attention = make_attention(attention_cls, 8, 7)
    expected = (torch.ones(1), [1, 2, 3]) if not split_qkv else (torch.ones(1),) * 3
    attention.upstream = lambda _: expected
    hidden, key_value = torch.ones(1), torch.zeros(1)
    assert attention.get_query_key_value_tensors(hidden, key_value, False, split_qkv) is expected
    assert attention.call == (hidden, key_value, False, split_qkv)


@pytest.mark.unit
@pytest.mark.parametrize(
    "tp,query_shape,gate_shape",
    [
        (4, (2, 1, 6, 8), (2, 1, 12, 8)),
        (8, (2, 1, 3, 8), (2, 1, 9, 8)),
        (8, (2, 1, 3, 8), (2, 1, 6, 4)),
        (8, (2, 1, 3, 8), (1, 1, 6, 8)),
        (8, (2, 1, 3, 8), (2, 1, 48)),
        (8, (2, 1, 2, 8), (2, 1, 6, 8)),
    ],
)
def test_unknown_layout_is_rejected(attention_cls, tp, query_shape, gate_shape):
    attention = make_attention(attention_cls, tp, 0)
    attention.upstream = lambda _: (torch.zeros(query_shape), None, None, torch.zeros(gate_shape))
    with pytest.raises(RuntimeError, match="Unexpected gated attention layout"):
        attention.get_query_key_value_tensors(None, output_gate=True)


@pytest.mark.unit
def test_invalid_group_rank_is_rejected(attention_cls):
    attention = make_attention(attention_cls, 8, -1)
    attention.upstream = lambda _: (torch.zeros(2, 1, 3, 8), None, None, torch.zeros(2, 1, 6, 8))
    with pytest.raises(RuntimeError, match="Invalid gated attention TP group rank"):
        attention.get_query_key_value_tensors(None, output_gate=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
