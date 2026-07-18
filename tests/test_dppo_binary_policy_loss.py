import torch

from slime.utils.ppo_utils import compute_dppo_binary_policy_loss


def test_dppo_binary_tv_masks_advantage_direction_updates():
    rollout_probs = torch.tensor([0.10, 0.40, 0.50, 0.70])
    probs = torch.tensor([0.35, 0.10, 0.90, 0.20])
    rollout_log_probs = rollout_probs.log()
    log_probs = probs.log()
    advantages = torch.tensor([1.0, -1.0, 1.0, -1.0])

    output = compute_dppo_binary_policy_loss(
        log_probs,
        rollout_log_probs,
        advantages,
        eps_clip=0.2,
        eps_clip_high=0.28,
        loss_mode="dppo_binary_tv",
    )
    pg_loss = output["pg_losses"]
    clipfrac = output["pg_clipfrac"]

    expected_clipfrac = torch.tensor([0.0, 1.0, 1.0, 1.0])
    ratio = torch.exp(log_probs - rollout_log_probs).clamp(max=20.0)
    expected_loss = -advantages * ratio * (1.0 - expected_clipfrac) * log_probs

    torch.testing.assert_close(clipfrac, expected_clipfrac)
    torch.testing.assert_close(pg_loss, expected_loss)


def test_dppo_binary_kl_masks_only_matching_probability_direction():
    rollout_probs = torch.tensor([0.50, 0.50])
    probs = torch.tensor([0.90, 0.10])
    rollout_log_probs = rollout_probs.log()
    log_probs = probs.log()
    advantages = torch.tensor([-1.0, 1.0])

    output = compute_dppo_binary_policy_loss(
        log_probs,
        rollout_log_probs,
        advantages,
        eps_clip=0.05,
        eps_clip_high=0.05,
        loss_mode="dppo_binary_kl",
    )
    clipfrac = output["pg_clipfrac"]

    torch.testing.assert_close(clipfrac, torch.zeros_like(clipfrac))
