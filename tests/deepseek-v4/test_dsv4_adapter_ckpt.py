"""Unit tests for adapter-only (LoRA) checkpoint filtering.

V4 uses megatron.bridge LinearAdapter (an nn.Linear subclass) whose LoRA params
are named linear_in.weight / linear_out.weight (base stays at .weight, frozen) —
NO .adapter. marker — so the save/load filter must be keyed on requires_grad, not
naming. A wrong filter would write an empty or full (corrupt) checkpoint, so the
save path also asserts a small non-empty subset.
"""

import pytest
import torch
import torch.nn as nn

from slime.backends.megatron_utils.adapter_ckpt import (
    _is_adapter_key,
    _norm,
    adapter_only_model_save,
    trainable_param_keys,
)


class _Chunk(nn.Module):
    """Mimics a V4 LinearAdapter-wrapped module: frozen base + trainable lin_in/out."""

    def __init__(self, adapter=True):
        super().__init__()
        self.base = nn.Parameter(torch.zeros(4, 4), requires_grad=False)
        if adapter:
            self.lin_in = nn.Linear(4, 2, bias=False)
            self.lin_out = nn.Linear(2, 4, bias=False)
        self._adapter = adapter

    def named_parameters(self, *a, **k):
        out = [("module.decoder.q_a_proj.weight", self.base)]
        if self._adapter:
            out += [
                ("module.decoder.q_a_proj.linear_in.weight", self.lin_in.weight),
                ("module.decoder.q_a_proj.linear_out.weight", self.lin_out.weight),
            ]
        return out

    def sharded_state_dict(self, *a, **k):
        sd = {
            "decoder.q_a_proj.weight": "BASE",
            "decoder.q_a_proj._extra_state": "EXTRA",
        }
        if self._adapter:
            sd["decoder.q_a_proj.linear_in.weight"] = "ADAPT_IN"
            sd["decoder.q_a_proj.linear_out.weight"] = "ADAPT_OUT"
        return sd


def test_norm_strips_wrapper_prefixes():
    assert _norm("module.x.y") == "x.y"
    assert _norm("module.module.x.y") == "x.y"
    assert _norm(("module.x.y", object())) == "x.y"  # tuple keys (dist ckpt)


def test_trainable_keys_are_adapter_only():
    tk = trainable_param_keys([_Chunk()])
    assert tk == {
        "decoder.q_a_proj.linear_in.weight",
        "decoder.q_a_proj.linear_out.weight",
    }
    assert "decoder.q_a_proj.weight" not in tk  # frozen base excluded


def test_is_adapter_key():
    tk = {"decoder.q_a_proj.linear_in.weight"}
    assert _is_adapter_key("decoder.q_a_proj.linear_in.weight", tk)
    assert not _is_adapter_key("decoder.q_a_proj.weight", tk)  # base
    assert not _is_adapter_key("decoder.q_a_proj._extra_state", tk)  # extra state
    assert _is_adapter_key("x.adapter.linear_in.weight", set())  # AdapterWrapper variant


def test_save_filter_keeps_adapters_drops_base_and_restores():
    m = [_Chunk()]
    with adapter_only_model_save(m):
        filtered = m[0].sharded_state_dict()
    assert set(filtered) == {
        "decoder.q_a_proj.linear_in.weight",
        "decoder.q_a_proj.linear_out.weight",
    }
    # original restored after the context
    assert "decoder.q_a_proj.weight" in m[0].sharded_state_dict()


def test_save_filter_refuses_when_nothing_trainable():
    # No adapters -> no trainable params -> must raise (would write empty ckpt).
    with pytest.raises(RuntimeError, match="no trainable params"):
        with adapter_only_model_save([_Chunk(adapter=False)]):
            pass


def test_save_filter_refuses_corrupt_all_or_nothing():
    # A key space where the filter keeps everything (no base to drop) must fail
    # the "strictly smaller" guard rather than silently save a full checkpoint.
    class AllAdapter(_Chunk):
        def sharded_state_dict(self, *a, **k):
            return {
                "decoder.q_a_proj.linear_in.weight": "IN",
                "decoder.q_a_proj.linear_out.weight": "OUT",
            }

    m = [AllAdapter()]
    with pytest.raises(AssertionError, match="kept"):
        with adapter_only_model_save(m):
            m[0].sharded_state_dict()


def test_chained_optimizer_synchronize_steps_tolerates_stub():
    """Muon+LoRA builds a ChainedOptimizer with a stub sub-optimizer
    (optimizer.optimizer is None). Upstream _synchronize_steps crashed on it
    ('NoneType' has no attribute 'param_groups') during any optimizer save/load;
    slime patches it stub-safe (checkpoint.py). Guards the patch stays applied."""
    from megatron.core.optimizer.optimizer import ChainedOptimizer

    import slime.backends.megatron_utils.checkpoint  # noqa: F401 (applies the patch on import)

    class _Stub:
        optimizer = None

    class _Real:
        class _Inner:
            param_groups = [{"params": [1], "step": 5}]

        optimizer = _Inner()

    co = ChainedOptimizer.__new__(ChainedOptimizer)
    co.chained_optimizers = [_Real(), _Stub()]
    # Must not raise on the stub, and must resolve the single real step.
    assert ChainedOptimizer._synchronize_steps(co) == 5
