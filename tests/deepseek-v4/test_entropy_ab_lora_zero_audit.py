"""CPU coverage for the entropy A/B fresh-LoRA functional-zero proof."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from slime.backends.megatron_utils import lora_zero_audit as audit


NUM_GPUS = 0


class _Adapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear_in = nn.Linear(2, 2, bias=False)
        self.linear_out = nn.Linear(2, 3, bias=False)
        self.base = nn.Linear(2, 3, bias=False)
        self.base.weight.requires_grad_(False)


def _model() -> list[nn.Module]:
    stage = nn.Module()
    stage.adapter = _Adapter()
    with torch.no_grad():
        stage.adapter.linear_in.weight.copy_(torch.tensor([[1.0, -2.0], [3.0, 4.0]]))
        stage.adapter.linear_out.weight.zero_()
    return [stage]


def test_collects_exact_zero_b_and_finite_a_moments_without_base_params():
    stats = audit.collect_local_lora_zero_stats(_model())

    assert stats.linear_in_tensors == 1
    assert stats.linear_in_numel == 4
    assert stats.linear_in_nonfinite == 0
    assert stats.linear_in_mean == pytest.approx(1.5)
    assert stats.linear_in_l2 == pytest.approx(math.sqrt(30.0))
    assert stats.linear_in_rms == pytest.approx(math.sqrt(7.5))
    assert stats.linear_in_max_abs == 4.0
    assert stats.linear_out_tensors == 1
    assert stats.linear_out_numel == 6
    assert stats.linear_out_nonzero == 0
    assert stats.linear_out_nonfinite == 0
    assert stats.linear_out_max_abs == 0.0
    assert stats.linear_out_l2 == 0.0


def test_clean_model_prints_one_stable_success_marker():
    messages: list[str] = []
    stats = audit.assert_zero_lora_out(_model(), should_print=True, print_fn=messages.append)

    assert stats.linear_out_nonzero == 0
    assert len(messages) == 1
    assert messages[0].startswith(f"[{audit.PASS_MARKER}] ")
    assert "linear_out_nonzero=0" in messages[0]
    assert "linear_in_nonfinite=0" in messages[0]


def test_any_local_nonzero_b_fails_closed():
    model = _model()
    with torch.no_grad():
        model[0].adapter.linear_out.weight[1, 0] = 0.25

    with pytest.raises(RuntimeError, match=audit.FAIL_MARKER) as exc_info:
        audit.assert_zero_lora_out(model)
    assert "linear_out_nonzero=1" in str(exc_info.value)
    assert "LoRA B is not exactly zero" in str(exc_info.value)


def test_nonfinite_a_fails_even_when_b_is_zero():
    model = _model()
    with torch.no_grad():
        model[0].adapter.linear_in.weight[0, 0] = float("nan")

    with pytest.raises(RuntimeError, match="LoRA A contains non-finite values"):
        audit.assert_zero_lora_out(model)


def test_remote_rank_nonzero_b_makes_this_clean_rank_fail(monkeypatch):
    """Simulate all-reduces where another rank owns one nonzero B element.

    Every rank receives the same globally reduced nonzero count before checking,
    so a clean local rank raises too instead of continuing into model collectives.
    """

    monkeypatch.setattr(audit.dist, "is_available", lambda: True)
    monkeypatch.setattr(audit.dist, "is_initialized", lambda: True)
    calls: list[tuple[object, object]] = []
    expected_group = object()

    def fake_all_reduce(tensor, *, op, group):
        calls.append((op, group))
        if len(calls) == 1:
            # A tensors/numel/nonfinite, B tensors/numel/nonzero/nonfinite.
            tensor += torch.tensor([1, 4, 0, 1, 6, 1, 0], dtype=tensor.dtype)
        elif len(calls) == 2:
            # A sum/sum_sq, B sum_sq from the bad remote rank.
            tensor += torch.tensor([6.0, 30.0, 0.0625], dtype=tensor.dtype)
        else:
            tensor.copy_(torch.maximum(tensor, torch.tensor([4.0, 0.25], dtype=tensor.dtype)))

    monkeypatch.setattr(audit.dist, "all_reduce", fake_all_reduce)

    with pytest.raises(RuntimeError, match="linear_out_nonzero=1"):
        audit.assert_zero_lora_out(_model(), process_group=expected_group)

    assert [op for op, _ in calls] == [audit.dist.ReduceOp.SUM, audit.dist.ReduceOp.SUM, audit.dist.ReduceOp.MAX]
    assert all(group is expected_group for _, group in calls)


def test_actor_hook_is_lazy_and_between_load_and_first_weight_owner():
    actor_path = REPO / "slime/backends/megatron_utils/actor.py"
    text = actor_path.read_text()
    gate = 'if role == "actor" and getattr(args, "assert_zero_lora_out", False):'
    gate_index = text.index(gate)

    assert text.index("initialize_model_and_optimizer(") < gate_index
    assert text.index("loaded_rollout_id = self.load_adapter_resume(") < gate_index
    assert gate_index < text.index("self.weights_backuper = TensorBackuper.create(")
    assert gate_index < text.index("self.weight_updater = update_weight_cls(")
    assert text.index("from .lora_zero_audit import assert_zero_lora_out") > gate_index


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
