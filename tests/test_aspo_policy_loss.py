import pytest
import torch

from slime.utils.ppo_utils import compute_aspo_policy_loss, compute_policy_loss_output

NUM_GPUS = 0


def test_aspo_positive_advantages_use_reciprocal_ratio_weight():
    old_log_probs = torch.tensor([-1.0, -2.0])
    log_probs = torch.tensor([-1.4, -2.1])
    advantages = torch.tensor([2.0, 0.5])

    output = compute_aspo_policy_loss(
        log_probs,
        old_log_probs,
        advantages,
        eps_clip=0.2,
        eps_clip_high=0.28,
    )

    reciprocal_ratio = torch.exp(old_log_probs - log_probs)
    torch.testing.assert_close(output["pg_losses"], -advantages * reciprocal_ratio * log_probs)
    torch.testing.assert_close(output["pg_clipfrac"], torch.zeros_like(advantages))
    torch.testing.assert_close(output["pg_upper_clipfrac"], torch.zeros_like(advantages))
    torch.testing.assert_close(output["pg_lower_clipfrac"], torch.zeros_like(advantages))


def test_aspo_positive_advantages_hard_mask_high_ratio_updates():
    old_log_probs = torch.tensor([-2.0, -2.0])
    log_probs = torch.tensor([-1.0, -2.1])
    advantages = torch.tensor([1.0, 1.0])

    output = compute_aspo_policy_loss(
        log_probs,
        old_log_probs,
        advantages,
        eps_clip=0.2,
        eps_clip_high=0.28,
    )

    expected_loss = torch.tensor([0.0, -torch.exp(old_log_probs[1] - log_probs[1]) * log_probs[1]])
    torch.testing.assert_close(output["pg_losses"], expected_loss)
    torch.testing.assert_close(output["pg_clipfrac"], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(output["pg_upper_clipfrac"], torch.tensor([1.0, 0.0]))


def test_aspo_negative_advantages_match_ppo_clipped_branch():
    old_log_probs = torch.tensor([-2.0, -0.2, -1.0])
    log_probs = torch.tensor([-3.0, 0.4, -1.0])
    advantages = torch.tensor([-1.5, -0.25, 0.0])

    aspo_output = compute_aspo_policy_loss(
        log_probs,
        old_log_probs,
        advantages,
        eps_clip=0.2,
        eps_clip_high=0.28,
        eps_clip_c=2.0,
    )
    ppo_output = compute_policy_loss_output(
        old_log_probs - log_probs,
        advantages,
        eps_clip=0.2,
        eps_clip_high=0.28,
        eps_clip_c=None,
    )

    for key in ["pg_losses", "pg_clipfrac", "pg_upper_clipfrac", "pg_lower_clipfrac"]:
        torch.testing.assert_close(aspo_output[key], ppo_output[key])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
