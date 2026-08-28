import pytest
import torch

from slime.utils.ppo_utils import compute_cispo_policy_loss

NUM_GPUS = 0


def test_cispo_uses_detached_clipped_importance_weights():
    old_log_probs = torch.tensor([-2.0, -1.0, -0.5])
    log_probs = torch.tensor([-0.5, -2.0, -0.5], requires_grad=True)
    advantages = torch.tensor([1.0, -2.0, 0.5])

    output = compute_cispo_policy_loss(
        log_probs,
        old_log_probs,
        advantages,
        eps_clip=0.2,
        eps_clip_high=0.28,
    )

    ratio = torch.exp(log_probs.detach() - old_log_probs)
    expected_weight = ratio.clamp(min=0.8, max=1.28)
    expected_loss = -expected_weight * advantages * log_probs.detach()

    torch.testing.assert_close(output["pg_losses"].detach(), expected_loss)
    torch.testing.assert_close(output["pg_upper_clipfrac"], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(output["pg_lower_clipfrac"], torch.tensor([0.0, 1.0, 0.0]))

    grad = torch.autograd.grad(output["pg_losses"].sum(), log_probs)[0]
    torch.testing.assert_close(grad, -expected_weight * advantages)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
