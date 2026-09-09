from __future__ import annotations

import pytest
import torch

from slime.observability.exp_metrics import compute_binary_dppo_exp_metrics, finalize_exp_metrics, split_exp_metrics

NUM_GPUS = 0


@pytest.mark.unit
def test_binary_dppo_exp_metrics_keep_advantage_sides_and_zero_separate():
    behavior_prob = torch.tensor([0.10, 0.40, 0.60, 0.80])
    current_prob = torch.tensor([0.35, 0.10, 0.90, 0.55])
    advantages = torch.tensor([1.0, -2.0, 1.5, 0.0])

    metrics = compute_binary_dppo_exp_metrics(
        log_probs=current_prob.log(),
        old_log_probs=behavior_prob.log(),
        advantages=advantages,
        loss_mode="dppo_binary_tv",
        eps_clip=0.2,
        eps_clip_high=0.2,
        ratio_clip_c=20.0,
        metric_reducer=torch.mean,
        policy_lags=torch.tensor([1.0, 2.0, 4.0, float("nan")]),
        sample_ages_seconds=torch.tensor([10.0, 100.0, 400.0, 1000.0]),
        turn_indices=torch.tensor([0.0, 1.0, 2.0, 3.0]),
        engine_version_spans=torch.tensor([0.0, 0.0, 1.0, 0.0]),
        engine_version_mismatches=torch.tensor([0.0, 1.0, 0.0, 0.0]),
    )
    finalize_exp_metrics(metrics)

    base = "exp/train/dppo"
    assert not any(key.startswith("_exp/") for key in metrics)
    assert metrics[f"{base}/adv/positive_token_fraction"] == pytest.approx(0.5)
    assert metrics[f"{base}/adv/negative_token_fraction"] == pytest.approx(0.25)
    assert metrics[f"{base}/adv/zero_token_fraction"] == pytest.approx(0.25)
    assert metrics[f"{base}/clip/positive_rate"] == pytest.approx(1.0)
    assert metrics[f"{base}/clip/negative_rate"] == pytest.approx(1.0)
    assert f"{base}/clip/zero_lower_joint_fraction" not in metrics
    assert metrics[f"{base}/masked_update_mass/positive_fraction"] == pytest.approx(1.0)
    assert metrics[f"{base}/masked_update_mass/negative_fraction"] == pytest.approx(1.0)
    assert metrics[f"{base}/delta/0.20/positive_clip_rate"] == pytest.approx(1.0)
    assert metrics[f"{base}/delta/0.20/negative_clip_rate"] == pytest.approx(1.0)
    assert metrics[f"{base}/async/policy_lag_mean"] == pytest.approx(7.0 / 3.0)
    assert metrics[f"{base}/async/lag/1/positive_clip_rate"] == pytest.approx(1.0)
    assert metrics[f"{base}/turn/1/negative_clip_rate"] == pytest.approx(1.0)
    assert metrics[f"{base}/async/engine_version_span_fraction"] == pytest.approx(0.25)
    assert metrics[f"{base}/async/engine_version_mismatch_fraction"] == pytest.approx(0.25)
    assert not any("_joint_" in key for key in metrics)


@pytest.mark.unit
def test_split_exp_metrics_does_not_mutate_payload():
    payload = {"train/loss": 1.0, "exp/train/dppo/binary_tv/mean": 0.2, "train/step": 3}

    regular, experimental = split_exp_metrics(payload)

    assert regular == {"train/loss": 1.0, "train/step": 3}
    assert experimental == {"exp/train/dppo/binary_tv/mean": 0.2}
    assert len(payload) == 3


@pytest.mark.unit
def test_binary_kl_exp_clip_metrics_follow_advantage_direction():
    behavior_prob = torch.full((4,), 0.5)
    current_prob = torch.tensor([0.9, 0.1, 0.9, 0.1])
    advantages = torch.tensor([1.0, -1.0, -1.0, 1.0])

    metrics = compute_binary_dppo_exp_metrics(
        log_probs=current_prob.log(),
        old_log_probs=behavior_prob.log(),
        advantages=advantages,
        loss_mode="dppo_binary_kl",
        eps_clip=0.05,
        eps_clip_high=0.05,
        ratio_clip_c=None,
        metric_reducer=torch.mean,
    )
    finalize_exp_metrics(metrics)

    assert metrics["exp/train/dppo/clip/positive_rate"] == pytest.approx(0.5)
    assert metrics["exp/train/dppo/clip/negative_rate"] == pytest.approx(0.5)
    assert metrics["exp/train/dppo/delta/0.05/positive_clip_rate"] == pytest.approx(0.5)
    assert metrics["exp/train/dppo/delta/0.05/negative_clip_rate"] == pytest.approx(0.5)
