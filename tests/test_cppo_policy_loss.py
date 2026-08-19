import pytest
import torch

from slime.utils.ppo_utils import compute_cppo_policy_loss

NUM_GPUS = 0


def test_cppo_position_weight_relaxes_late_token_threshold():
    old_probs = torch.full((3,), 0.2)
    probs = torch.full((3,), 0.36)
    advantages = torch.ones(3)

    output = compute_cppo_policy_loss(
        probs.log(),
        old_probs.log(),
        advantages,
        delta=0.15,
        prefix_delta=1.0,
        weight_floor=0.5,
    )

    torch.testing.assert_close(output["cppo_weighted_divergence"], torch.tensor([0.16, 0.12, 0.08]))
    torch.testing.assert_close(output["pg_clipfrac"], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(output["cppo_token_clipfrac"], torch.tensor([1.0, 0.0, 0.0]))


def test_cppo_prefix_budget_accounts_for_all_preceding_tokens():
    old_probs = torch.full((3,), 0.2)
    probs = torch.full((3,), 0.34)
    advantages = torch.ones(3)

    output = compute_cppo_policy_loss(
        probs.log(),
        old_probs.log(),
        advantages,
        delta=0.2,
        prefix_delta=0.05,
        weight_floor=1.0,
    )

    torch.testing.assert_close(output["cppo_effective_threshold"], torch.tensor([0.2, 0.16, 0.12]), atol=1e-6, rtol=0)
    torch.testing.assert_close(output["cppo_prefix_delta"], torch.full((3,), 0.1), atol=1e-6, rtol=0)
    torch.testing.assert_close(output["pg_clipfrac"], torch.tensor([0.0, 0.0, 1.0]))
    torch.testing.assert_close(output["cppo_prefix_clipfrac"], torch.tensor([0.0, 0.0, 1.0]))


def test_cppo_keeps_updates_toward_behavior_policy():
    old_probs = torch.tensor([0.5, 0.5])
    probs = torch.tensor([0.9, 0.1])
    advantages = torch.tensor([-1.0, 1.0])

    output = compute_cppo_policy_loss(
        probs.log(),
        old_probs.log(),
        advantages,
        delta=0.1,
        prefix_delta=0.01,
        weight_floor=0.8,
    )

    torch.testing.assert_close(output["pg_clipfrac"], torch.zeros(2))
    assert torch.all(output["pg_losses"] != 0.0)


def test_cppo_single_token_has_unit_position_weight_and_masks_gradient():
    old_log_probs = torch.tensor([0.2]).log()
    log_probs = torch.tensor([0.5]).log().requires_grad_()

    output = compute_cppo_policy_loss(
        log_probs,
        old_log_probs,
        torch.ones(1),
        delta=0.1,
        prefix_delta=0.02,
        weight_floor=0.8,
    )
    output["pg_losses"].sum().backward()

    torch.testing.assert_close(output["cppo_weighted_divergence"], torch.tensor([0.3]))
    torch.testing.assert_close(output["pg_clipfrac"], torch.ones(1))
    torch.testing.assert_close(log_probs.grad, torch.zeros_like(log_probs))


def test_cppo_eps_clip_c_caps_detached_score_function_weight():
    log_probs = torch.tensor([0.0], requires_grad=True)
    old_log_probs = torch.tensor([-2.0])

    output = compute_cppo_policy_loss(
        log_probs,
        old_log_probs,
        torch.ones(1),
        delta=10.0,
        prefix_delta=10.0,
        weight_floor=0.8,
        eps_clip_c=2.0,
    )

    output["pg_losses"].sum().backward()

    torch.testing.assert_close(log_probs.grad, torch.tensor([-2.0]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
