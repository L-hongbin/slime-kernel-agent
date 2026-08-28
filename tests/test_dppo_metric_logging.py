import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slime.observability import train_metric_utils, wandb_utils

NUM_GPUS = 0


def test_dppo_metrics_keep_a_top_level_tracking_namespace():
    assert train_metric_utils.format_train_metric_key("dppo/adv_negative_token_frac") == (
        "dppo/adv_negative_token_frac"
    )
    assert train_metric_utils.format_train_metric_key("dppo/adv_negative_token_frac", "critic-") == (
        "dppo/critic-adv_negative_token_frac"
    )
    assert train_metric_utils.format_train_metric_key("entropy/first_order_unmasked") == (
        "entropy/first_order_unmasked"
    )
    assert train_metric_utils.format_train_metric_key("entropy_loss") == "entropy/train"


def test_wandb_dppo_group_uses_the_train_step(monkeypatch):
    calls = []
    fake_wandb = SimpleNamespace(define_metric=lambda *args, **kwargs: calls.append((args, kwargs)))
    monkeypatch.setattr(wandb_utils, "wandb", fake_wandb)

    wandb_utils._init_wandb_common()

    assert (("dppo/*",), {"step_metric": "train/step"}) in calls


def test_dppo_ratios_are_derived_only_after_reduced_sufficient_statistics():
    metrics = {
        "dppo/adv_positive_token_frac": 0.2,
        "dppo/adv_negative_token_frac": 0.8,
        "dppo/upper_clip_joint_frac": 0.02,
        "dppo/lower_clip_joint_frac": 0.16,
        "dppo/positive_kept_joint_frac": 0.18,
        "dppo/negative_kept_joint_frac": 0.64,
        "dppo/update_mass_positive_mean": 0.4,
        "dppo/update_mass_negative_mean": 1.6,
        "dppo/masked_update_mass_positive_mean": 0.1,
        "dppo/masked_update_mass_negative_mean": 0.8,
        "dppo/kept_update_mass_mean": 1.1,
        "dppo/net_logprob_push_unmasked_numerator_mean": -1.2,
        "dppo/net_logprob_push_kept_numerator_mean": -0.3,
    }
    categories = {
        "positive_clipped": 0.02,
        "positive_kept": 0.18,
        "negative_clipped": 0.16,
        "negative_kept": 0.64,
    }
    for category, fraction in categories.items():
        metrics[f"dppo/train_sampled_prob_{category}_joint_mean"] = fraction * 0.3
        metrics[f"dppo/rollout_sampled_prob_{category}_joint_mean"] = fraction * 0.25

    result = train_metric_utils.add_derived_dppo_metrics(metrics)

    assert result["dppo/upper_clip_rate_given_positive"] == pytest.approx(0.1)
    assert result["dppo/lower_clip_rate_given_negative"] == pytest.approx(0.2)
    assert result["dppo/masked_update_mass_positive_frac"] == pytest.approx(0.25)
    assert result["dppo/masked_update_mass_negative_frac"] == pytest.approx(0.5)
    assert result["dppo/net_logprob_push_unmasked"] == pytest.approx(-0.6)
    assert result["dppo/net_logprob_push_kept"] == pytest.approx(-0.3 / 1.1)
    assert result["dppo/mask_induced_push_delta"] == pytest.approx(-0.3 / 1.1 + 0.6)
    for category in categories:
        assert result[f"dppo/train_sampled_prob_{category}"] == pytest.approx(0.3)
        assert result[f"dppo/rollout_sampled_prob_{category}"] == pytest.approx(0.25)


def test_dppo_derived_metrics_use_zero_for_an_empty_advantage_side():
    metrics = {
        "dppo/adv_positive_token_frac": 0.0,
        "dppo/adv_negative_token_frac": 0.0,
        "dppo/upper_clip_joint_frac": 0.0,
        "dppo/lower_clip_joint_frac": 0.0,
        "dppo/positive_kept_joint_frac": 0.0,
        "dppo/negative_kept_joint_frac": 0.0,
        "dppo/update_mass_positive_mean": 0.0,
        "dppo/update_mass_negative_mean": 0.0,
        "dppo/masked_update_mass_positive_mean": 0.0,
        "dppo/masked_update_mass_negative_mean": 0.0,
        "dppo/kept_update_mass_mean": 0.0,
        "dppo/net_logprob_push_unmasked_numerator_mean": 0.0,
        "dppo/net_logprob_push_kept_numerator_mean": 0.0,
    }
    for category in ("positive_clipped", "positive_kept", "negative_clipped", "negative_kept"):
        metrics[f"dppo/train_sampled_prob_{category}_joint_mean"] = 0.0
        metrics[f"dppo/rollout_sampled_prob_{category}_joint_mean"] = 0.0

    result = train_metric_utils.add_derived_dppo_metrics(metrics)

    assert result["dppo/upper_clip_rate_given_positive"] == 0.0
    assert result["dppo/lower_clip_rate_given_negative"] == 0.0
    assert result["dppo/mask_induced_push_delta"] == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
