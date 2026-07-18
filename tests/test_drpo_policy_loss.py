import torch

from slime.utils.ppo_utils import compute_dppo_binary_policy_loss, compute_drpo_policy_loss


def test_drpo_matches_quadratic_binary_tv_objective():
    rollout_probs = torch.tensor([0.20, 0.60, 0.50, 0.50])
    probs = torch.tensor([0.35, 0.30, 0.55, 0.45])
    rollout_log_probs = rollout_probs.log()
    log_probs = probs.log()
    advantages = torch.tensor([1.0, -2.0, 0.5, -0.5])

    output = compute_drpo_policy_loss(
        log_probs,
        rollout_log_probs,
        advantages,
        eps_clip=0.1,
        eps_clip_high=0.12,
    )

    ratio = torch.exp(log_probs - rollout_log_probs)
    eps = torch.where(advantages > 0, torch.full_like(advantages, 0.12), torch.full_like(advantages, 0.1))
    expected_loss = -advantages * ratio + advantages.abs() * rollout_probs * (ratio - 1.0).pow(2) / (2.0 * eps)
    expected_clipfrac = torch.tensor([1.0, 1.0, 0.0, 0.0])

    torch.testing.assert_close(output["pg_losses"], expected_loss)
    torch.testing.assert_close(output["pg_clipfrac"], expected_clipfrac)


def test_drpo_keeps_smooth_loss_where_dppo_binary_masks_update():
    rollout_probs = torch.tensor([0.10, 0.70])
    probs = torch.tensor([0.40, 0.20])
    rollout_log_probs = rollout_probs.log()
    log_probs = probs.log()
    advantages = torch.tensor([1.0, -1.0])

    dppo_output = compute_dppo_binary_policy_loss(
        log_probs,
        rollout_log_probs,
        advantages,
        eps_clip=0.2,
        eps_clip_high=0.2,
        loss_mode="dppo_binary_tv",
    )
    drpo_output = compute_drpo_policy_loss(
        log_probs,
        rollout_log_probs,
        advantages,
        eps_clip=0.2,
        eps_clip_high=0.2,
    )

    torch.testing.assert_close(dppo_output["pg_clipfrac"], torch.ones_like(advantages))
    torch.testing.assert_close(dppo_output["pg_losses"], torch.zeros_like(advantages))
    assert torch.all(drpo_output["pg_clipfrac"] == 1.0)
    assert torch.all(drpo_output["pg_losses"] != 0.0)
