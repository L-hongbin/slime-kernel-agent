import torch

from slime.utils.ppo_utils import compute_ripo_policy_loss


def test_ripo_uses_old_probability_dependent_dynamic_bounds():
    old_prob = torch.tensor([0.01, 0.8])
    old_log_probs = old_prob.log()
    ratio = torch.tensor([2.0, 1.2])
    log_probs = old_log_probs + ratio.log()
    advantages = torch.tensor([1.0, 1.0])

    output = compute_ripo_policy_loss(
        log_probs,
        old_log_probs,
        advantages,
        ripo_delta=0.02,
        ripo_delta_high=0.02,
        ripo_ratio_min=0.0,
        ripo_ratio_max=None,
    )

    eps = torch.sqrt(torch.tensor(0.02) / old_prob)
    expected_upper = 1.0 + eps
    expected_clipped_ratio = torch.minimum(ratio, expected_upper)

    torch.testing.assert_close(output["ripo_eps_high"], eps)
    torch.testing.assert_close(output["ripo_clip_upper"], expected_upper)
    torch.testing.assert_close(output["pg_losses"], -expected_clipped_ratio * advantages)
    torch.testing.assert_close(output["pg_upper_clipfrac"], torch.tensor([0.0, 1.0]))


def test_ripo_applies_outer_ratio_bounds_from_reported_experiments():
    old_prob = torch.tensor([1e-6])
    old_log_probs = old_prob.log()
    ratio = torch.tensor([20.0])
    log_probs = old_log_probs + ratio.log()
    advantages = torch.tensor([1.0])

    output = compute_ripo_policy_loss(
        log_probs,
        old_log_probs,
        advantages,
        ripo_delta=0.05,
        ripo_delta_high=0.05,
        ripo_ratio_min=0.5,
        ripo_ratio_max=10.0,
    )

    torch.testing.assert_close(output["ripo_clip_upper"], torch.tensor([10.0]))
    torch.testing.assert_close(output["pg_losses"], torch.tensor([-10.0]))
    torch.testing.assert_close(output["pg_upper_clipfrac"], torch.tensor([1.0]))


def test_ripo_clips_negative_advantages_at_dynamic_lower_bound():
    old_prob = torch.tensor([0.8])
    old_log_probs = old_prob.log()
    ratio = torch.tensor([0.6])
    log_probs = old_log_probs + ratio.log()
    advantages = torch.tensor([-1.0])

    output = compute_ripo_policy_loss(
        log_probs,
        old_log_probs,
        advantages,
        ripo_delta=0.05,
        ripo_delta_high=0.05,
        ripo_ratio_min=0.5,
        ripo_ratio_max=10.0,
    )

    expected_lower = 1.0 - torch.sqrt(torch.tensor(0.05) / old_prob)
    torch.testing.assert_close(output["ripo_clip_lower"], expected_lower)
    torch.testing.assert_close(output["pg_losses"], expected_lower)
    torch.testing.assert_close(output["pg_lower_clipfrac"], torch.tensor([1.0]))
