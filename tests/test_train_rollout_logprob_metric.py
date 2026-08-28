"""CPU regression tests for the train/rollout log-prob mismatch metric."""

import math
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slime.backends.megatron_utils import loss as loss_module  # noqa: E402
from slime.utils.ppo_utils import compute_policy_loss, compute_ppo_clip_diagnostics  # noqa: E402

NUM_GPUS = 0


def test_ordinary_ppo_clip_diagnostics_resolve_advantage_direction_and_dual_clip():
    ratios = torch.tensor([1.3, 0.7, 6.0, 0.9], dtype=torch.float64)
    ppo_kl = -ratios.log()
    advantages = torch.tensor([1.0, -1.0, -1.0, 0.0], dtype=torch.float64)

    diagnostics = compute_ppo_clip_diagnostics(
        ppo_kl,
        advantages,
        eps_clip=0.2,
        eps_clip_high=0.2,
        eps_clip_c=5.0,
    )
    assert diagnostics["pg_upper_clipfrac"].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert diagnostics["pg_lower_clipfrac"].tolist() == [0.0, 1.0, 0.0, 0.0]
    assert diagnostics["pg_dual_clipfrac"].tolist() == [0.0, 0.0, 1.0, 0.0]

    _, clipfrac = compute_policy_loss.__wrapped__(ppo_kl, advantages, 0.2, 0.2, 5.0)
    torch.testing.assert_close(
        diagnostics["pg_upper_clipfrac"] + diagnostics["pg_lower_clipfrac"],
        clipfrac,
    )


@pytest.mark.parametrize("use_rollout_logprobs", [False, True])
def test_metric_compares_current_forward_with_rollout_engine(monkeypatch, use_rollout_logprobs):
    current_log_probs = torch.tensor([-1.0, -2.0])
    rollout_log_probs = torch.tensor([-2.0, -4.0])
    stored_train_log_probs = torch.tensor([-8.0, -8.0])

    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: (
            torch.empty(0),
            {
                "log_probs": [current_log_probs],
                "entropy": [torch.zeros_like(current_log_probs)],
            },
        ),
    )
    observed_policy_args = {}

    def fake_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c):
        observed_policy_args.update(
            eps_clip=eps_clip,
            eps_clip_high=eps_clip_high,
            eps_clip_c=eps_clip_c,
        )
        zeros = torch.zeros_like(ppo_kl)
        return {
            "pg_losses": zeros,
            "pg_clipfrac": zeros,
            "pg_upper_clipfrac": zeros,
            "pg_lower_clipfrac": zeros,
        }

    monkeypatch.setattr(loss_module, "compute_policy_loss", fake_policy_loss)

    args = Namespace(
        use_rollout_logprobs=use_rollout_logprobs,
        use_opsm=False,
        advantage_estimator="grpo",
        policy_loss_mode="ppo",
        eps_clip=0.2,
        eps_clip_high=0.2,
        eps_clip_c=5.0,
        get_mismatch_metrics=False,
        use_tis=False,
        entropy_coef=0.0,
        use_kl_loss=False,
    )
    batch = {
        "advantages": [torch.ones(2)],
        "log_probs": [stored_train_log_probs],
        "rollout_log_probs": [rollout_log_probs],
        "response_lengths": [2],
        "total_lengths": [3],
        "unconcat_tokens": [torch.tensor([1, 2, 3])],
        "loss_masks": [torch.ones(2)],
    }

    _, metrics = loss_module.policy_loss_function(
        args,
        batch,
        logits=torch.zeros((1, 3, 4)),
        sum_of_sample_mean=torch.mean,
    )

    expected = (current_log_probs - rollout_log_probs).abs().mean()
    torch.testing.assert_close(metrics["train_rollout_logprob_abs_diff"], expected)
    assert metrics["train_rollout_logprob_abs_diff"].item() != 0.0
    assert observed_policy_args == {
        "eps_clip": 0.2,
        "eps_clip_high": 0.2,
        "eps_clip_c": 5.0,
    }


def test_dis_raw_importance_ratio_is_exposed_as_a_train_metric(monkeypatch):
    current_log_probs = torch.tensor([math.log(0.4), math.log(0.1)], dtype=torch.float64)
    rollout_log_probs = torch.tensor([math.log(0.2), math.log(0.2)], dtype=torch.float64)

    monkeypatch.setattr(
        loss_module,
        "get_log_probs_and_entropy",
        lambda *args, **kwargs: (
            torch.empty(0),
            {
                "log_probs": [current_log_probs],
                "entropy": [torch.zeros_like(current_log_probs)],
            },
        ),
    )
    args = Namespace(
        use_rollout_logprobs=True,
        use_opsm=False,
        advantage_estimator="grpo",
        policy_loss_mode="dis",
        dis_ratio_level="token",
        eps_clip=0.8,
        eps_clip_high=3.0,
        get_mismatch_metrics=False,
        use_tis=False,
        entropy_coef=0.0,
        use_kl_loss=False,
    )
    batch = {
        "advantages": [torch.ones(2, dtype=torch.float64)],
        "rollout_log_probs": [rollout_log_probs],
        "response_lengths": [2],
        "total_lengths": [3],
        "unconcat_tokens": [torch.tensor([1, 2, 3])],
        "loss_masks": [torch.ones(2)],
    }

    _, metrics = loss_module.policy_loss_function(
        args,
        batch,
        logits=torch.zeros((1, 3, 4)),
        sum_of_sample_mean=torch.mean,
    )

    # policy_loss_function's normal metric formatter/logging path maps this to
    # train/dis_importance_ratio in both the text log and W&B.
    torch.testing.assert_close(metrics["dis_importance_ratio"], torch.tensor(1.25, dtype=torch.float64))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
