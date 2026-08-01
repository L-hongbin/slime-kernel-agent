"""Tests for V4HashRouter — now a subclass of the OFFICIAL HF
``DeepseekV4HashRouter`` (structure/init/buffers inherited upstream), carrying
exactly one deliberate forward deviation: sglang-exact weight renorm (bare sum,
no ``+1e-20`` guard) so train-side gate weights match the rollout engine.

CPU-only; no GPU or sglang server needed.
Run: python -m pytest tests/deepseek-v4/test_dsv4_hash_router.py -q
"""

from __future__ import annotations

import pytest
import torch

transformers_v4 = pytest.importorskip("transformers.models.deepseek_v4.modeling_deepseek_v4")
mcore = pytest.importorskip("custom_kernels.deepseek_v4.megatron.mcore_model")

from custom_kernels.deepseek_v4.megatron.mcore_model import V4HashRouter  # noqa: E402
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HashRouter  # noqa: E402


class _Cfg:
    """Minimal config surface the router reads (mirrors DeepseekV4Config fields)."""

    num_experts_per_tok = 4
    num_local_experts = 16
    hidden_size = 32
    scoring_func = "sigmoid"
    routed_scaling_factor = 2.5
    vocab_size = 64


def _mk_pair(seed=0):
    """Ours + a pure-HF instance with IDENTICAL weights and tid2eid."""
    torch.manual_seed(seed)
    cfg = _Cfg()
    ours = V4HashRouter(cfg)
    ref = DeepseekV4HashRouter(cfg)
    with torch.no_grad():
        w = torch.randn(cfg.num_local_experts, cfg.hidden_size)
        ours.weight.copy_(w)
        ref.weight.copy_(w)
        t2e = torch.randint(0, cfg.num_local_experts, (cfg.vocab_size, cfg.num_experts_per_tok))
        ours.tid2eid.copy_(t2e)
        ref.tid2eid.copy_(t2e)
    return cfg, ours, ref


# ---------------------------------------------------------------- inheritance


def test_is_official_hf_subclass():
    """Structure/init/buffers come from the official HF class — the whole point
    of the refactor (community updates propagate; only forward is overridden)."""
    assert issubclass(V4HashRouter, DeepseekV4HashRouter)
    # __init__ is inherited verbatim (not overridden).
    assert "__init__" not in V4HashRouter.__dict__
    # forward IS overridden (the single sglang-renorm deviation).
    assert "forward" in V4HashRouter.__dict__


def test_state_dict_layout_matches_hf():
    """Checkpoint compatibility: identical parameter/buffer names, shapes, dtypes,
    and tid2eid persistence — HF-converted checkpoints keep loading."""
    cfg, ours, ref = _mk_pair()
    ours_sd, ref_sd = ours.state_dict(), ref.state_dict()
    assert set(ours_sd) == set(ref_sd) == {"weight", "tid2eid"}
    for k in ours_sd:
        assert ours_sd[k].shape == ref_sd[k].shape, k
        assert ours_sd[k].dtype == ref_sd[k].dtype, k
    assert ours.tid2eid.dtype == torch.long
    assert ours.tid2eid.shape == (cfg.vocab_size, cfg.num_experts_per_tok)


# ------------------------------------------------------------------- forward


def test_forward_parity_with_hf_reference():
    """With identical weights, outputs match HF within fp tolerance (the renorm
    guard difference is O(1e-20/denominator) — invisible for real scores)."""
    cfg, ours, ref = _mk_pair(seed=1)
    x = torch.randn(2, 5, cfg.hidden_size)
    ids = torch.randint(0, cfg.vocab_size, (2, 5))

    lo, wo, io = ours(x, ids)
    lr, wr, ir = ref(x, ids)

    assert torch.equal(io, ir), "expert selection must be identical (frozen table)"
    assert torch.equal(lo, lr), "logits are the same computation"
    assert torch.allclose(wo, wr, rtol=1e-6, atol=1e-7)


def test_renorm_is_sglang_exact_bare_sum():
    """The one deliberate deviation: weights renormed by a BARE sum (sglang
    moe/topk.py) — verify against the explicit formula, not HF's guarded one."""
    cfg, ours, _ = _mk_pair(seed=2)
    x = torch.randn(3, cfg.hidden_size)
    ids = torch.randint(0, cfg.vocab_size, (3,))

    logits, weights, indices = ours(x, ids)
    scores = torch.sigmoid(logits)
    gathered = scores.gather(1, indices)
    expected = gathered / gathered.sum(dim=-1, keepdim=True) * cfg.routed_scaling_factor
    assert torch.equal(weights, expected), "must be the bare-sum renorm, bit-exact"


def test_selection_is_frozen_table_only():
    """Expert indices depend ONLY on input_ids via tid2eid — never on hidden
    states or the learned gate (the defining property of hash routing)."""
    cfg, ours, _ = _mk_pair(seed=3)
    ids = torch.randint(0, cfg.vocab_size, (4,))
    _, _, i1 = ours(torch.randn(4, cfg.hidden_size), ids)
    _, _, i2 = ours(torch.randn(4, cfg.hidden_size) * 100.0, ids)
    assert torch.equal(i1, i2)
    assert torch.equal(i1, ours.tid2eid[ids])


def test_gate_weight_gets_grad_selection_does_not_block():
    """Gradients flow into the learned gate weight through the gathered scores;
    the frozen tid2eid buffer never requires grad."""
    cfg, ours, _ = _mk_pair(seed=4)
    x = torch.randn(4, cfg.hidden_size, requires_grad=True)
    ids = torch.randint(0, cfg.vocab_size, (4,))
    _, weights, _ = ours(x, ids)
    weights.sum().backward()
    assert ours.weight.grad is not None and ours.weight.grad.abs().sum() > 0
    assert x.grad is not None
    assert not ours.tid2eid.requires_grad


def test_requires_input_ids():
    cfg, ours, _ = _mk_pair()
    with pytest.raises(AssertionError, match="requires input_ids"):
        ours(torch.randn(2, cfg.hidden_size), None)


def test_routed_scaling_factor_applied():
    cfg, ours, _ = _mk_pair(seed=5)
    x = torch.randn(2, cfg.hidden_size)
    ids = torch.randint(0, cfg.vocab_size, (2,))
    _, weights, _ = ours(x, ids)
    # renormed weights sum to routed_scaling_factor per token (bare-sum renorm
    # makes the pre-scale sum exactly 1).
    assert torch.allclose(weights.sum(dim=-1), torch.full((2,), cfg.routed_scaling_factor))


def test_batchified_shapes():
    """[B,S,H] hidden + [B,S] ids flatten consistently (the MoE layer contract)."""
    cfg, ours, _ = _mk_pair(seed=6)
    B, S = 3, 7
    x = torch.randn(B, S, cfg.hidden_size)
    ids = torch.randint(0, cfg.vocab_size, (B, S))
    logits, weights, indices = ours(x, ids)
    assert logits.shape == (B * S, cfg.num_local_experts)
    assert weights.shape == (B * S, cfg.num_experts_per_tok)
    assert indices.shape == (B * S, cfg.num_experts_per_tok)


def test_no_routing_replay_hook():
    """Hash routing is deterministic in input_ids — it must NOT register with the
    rollout routing replay (only the learned V4TopKRouter records/replays)."""
    cfg, ours, _ = _mk_pair()
    assert not hasattr(ours, "routing_replay")
