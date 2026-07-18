import torch

from slime.utils.ppo_utils import compute_policy_loss, compute_up_policy_loss


def test_up_uses_unclipped_logprob_loss_for_positive_advantages():
    old_log_probs = torch.tensor([0.2, -3.0]).log_softmax(dim=0)
    log_probs = old_log_probs + torch.tensor([1.0, -1.0])
    advantages = torch.tensor([2.0, 0.5])

    output = compute_up_policy_loss(
        log_probs,
        old_log_probs,
        advantages,
        eps_clip=0.2,
        eps_clip_high=0.28,
    )

    torch.testing.assert_close(output["pg_losses"], -advantages * log_probs)
    torch.testing.assert_close(output["pg_clipfrac"], torch.zeros_like(advantages))
    torch.testing.assert_close(output["pg_upper_clipfrac"], torch.zeros_like(advantages))
    torch.testing.assert_close(output["pg_lower_clipfrac"], torch.zeros_like(advantages))


def test_up_keeps_ppo_clipped_objective_for_non_positive_advantages():
    old_log_probs = torch.tensor([-2.0, -0.2, -1.0])
    log_probs = torch.tensor([-3.0, 0.4, -1.0])
    advantages = torch.tensor([-1.5, -0.25, 0.0])

    up_output = compute_up_policy_loss(
        log_probs,
        old_log_probs,
        advantages,
        eps_clip=0.2,
        eps_clip_high=0.28,
        eps_clip_c=2.0,
    )
    ppo_output = compute_policy_loss(
        old_log_probs - log_probs,
        advantages,
        eps_clip=0.2,
        eps_clip_high=0.28,
        eps_clip_c=2.0,
    )

    for key in ["pg_losses", "pg_clipfrac", "pg_upper_clipfrac", "pg_lower_clipfrac"]:
        torch.testing.assert_close(up_output[key], ppo_output[key])
