"""Behavior tests for Direct Double-Sided Importance Sampling (DIS).

Reference: handoffs/ddpo/src/sao-arXiv-2607.07508v1/4.method.tex.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slime.utils.ppo_utils import compute_dis_policy_loss, compute_sequence_log_ratio

NUM_GPUS = 0


def _logs(values, *, requires_grad=False):
    return torch.log(torch.tensor(values, dtype=torch.float64)).requires_grad_(requires_grad)


def test_double_sided_gate_is_independent_of_advantage_sign():
    rollout = _logs([0.1] * 6)
    # Coding-task DIS uses eps_low=.8 and eps_high=3: keep interval (.2, 4.0).
    ratios = [0.19, 0.21, 1.0, 3.99, 4.01, 5.0]
    current = rollout + torch.log(torch.tensor(ratios, dtype=torch.float64))
    advantages = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=torch.float64)

    result = compute_dis_policy_loss(current, rollout, advantages, 0.8, 3.0)

    assert result["pg_lower_clipfrac"].tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert result["pg_upper_clipfrac"].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0]
    assert result["dis_valid_token_frac"].tolist() == [0.0, 1.0, 1.0, 1.0, 0.0, 0.0]
    assert result["pg_losses"][0].item() == 0.0
    assert result["pg_losses"][4].item() == 0.0
    assert result["pg_losses"][5].item() == 0.0
    assert result["pg_losses"][1].item() != 0.0  # negative A is kept inside the interval
    assert result["pg_losses"][3].item() != 0.0


def test_importance_weight_and_gate_are_detached():
    rollout = _logs([0.2], requires_grad=True)
    current = _logs([0.4], requires_grad=True)
    advantages = torch.tensor([1.5], dtype=torch.float64, requires_grad=True)

    result = compute_dis_policy_loss(current, rollout, advantages, 0.8, 3.0)
    result["pg_losses"].sum().backward()

    # d[-A * detach(r) * log pi]/d log pi = -A*r = -3.
    torch.testing.assert_close(current.grad, torch.tensor([-3.0], dtype=torch.float64))
    assert rollout.grad is None
    assert advantages.grad is None
    torch.testing.assert_close(result["dis_importance_weight"], torch.tensor([2.0], dtype=torch.float64))


def test_outside_tokens_stay_finite_even_for_extreme_log_ratios():
    rollout = torch.tensor([-1000.0, -1.0], dtype=torch.float32)
    current = torch.tensor([0.0, -1000.0], dtype=torch.float32, requires_grad=True)
    advantages = torch.ones(2, dtype=torch.float32)

    result = compute_dis_policy_loss(current, rollout, advantages, 0.8, 3.0)

    assert result["pg_clipfrac"].tolist() == [1.0, 1.0]
    assert torch.isfinite(result["pg_losses"]).all()
    assert result["pg_losses"].tolist() == [0.0, 0.0]


def test_sequence_ratio_uses_one_geometric_mean_gate_for_the_response():
    token_ratios = torch.tensor([0.9996, 1.0006], dtype=torch.float64)
    rollout = torch.zeros(2, dtype=torch.float64)
    current = token_ratios.log().requires_grad_()
    sequence_log_ratio = compute_sequence_log_ratio(
        full_log_probs=[current],
        full_rollout_log_probs=[rollout],
        local_log_probs=[current],
        loss_masks=[torch.ones(2, dtype=torch.float64)],
    )

    result = compute_dis_policy_loss(
        current,
        rollout,
        torch.ones(2, dtype=torch.float64),
        0.0003,
        0.0007,
        log_ratio=sequence_log_ratio,
    )

    expected_ratio = token_ratios.prod().sqrt().expand(2)
    torch.testing.assert_close(result["dis_importance_weight"], expected_ratio)
    assert result["dis_valid_token_frac"].tolist() == [1.0, 1.0]


@pytest.mark.parametrize(
    ("eps_low", "eps_high", "message"),
    [
        (0.0, 3.0, "eps_clip"),
        (1.0, 3.0, "eps_clip"),
        (0.8, 0.0, "eps_clip_high"),
        (0.8, math.inf, "eps_clip_high"),
    ],
)
def test_invalid_thresholds_fail(eps_low, eps_high, message):
    with pytest.raises(ValueError, match=message):
        compute_dis_policy_loss(_logs([0.2]), _logs([0.2]), torch.ones(1), eps_low, eps_high)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
