"""Unit tests for the predictive Top-K-KL DPPO policy loss.

The hand calculations below follow arXiv:2607.10848: the retained support is
the rollout policy's Top-K union the sampled token, and the complement is
represented by either one aggregate bucket or a uniform tail.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slime.utils.ppo_utils import compute_dppo_predictive_topk_policy_loss  # noqa: E402

NUM_GPUS = 0


def _log_probs(probabilities, *, requires_grad=False):
    result = torch.log(torch.tensor(probabilities, dtype=torch.float64))
    return result.requires_grad_(requires_grad)


def _compute(
    *,
    sampled_current,
    sampled_behavior,
    behavior_support,
    current_support,
    advantages,
    delta=0.0,
    tail_estimator="aggregated",
    vocab_size=10,
    ratio_cap=5.0,
    valid_mask=None,
):
    behavior_support_logs = _log_probs(behavior_support)
    current_support_logs = _log_probs(current_support)
    if valid_mask is None:
        valid_mask = torch.ones_like(behavior_support_logs, dtype=torch.bool)
    return compute_dppo_predictive_topk_policy_loss(
        log_probs=_log_probs(sampled_current),
        old_log_probs=_log_probs(sampled_behavior),
        behavior_support_log_probs=behavior_support_logs,
        current_support_log_probs=current_support_logs,
        support_valid_mask=valid_mask,
        advantages=torch.tensor(advantages, dtype=torch.float64),
        delta=delta,
        tail_estimator=tail_estimator,
        vocab_size=vocab_size,
        eps_clip_c=ratio_cap,
    )


def test_aggregated_tail_matches_manual_kl_and_directional_derivative():
    # mu_S=(.4,.3), pi_S=(.5,.2), and both tail masses are .3.
    out = _compute(
        sampled_current=[0.5],
        sampled_behavior=[0.4],
        behavior_support=[[0.4, 0.3]],
        current_support=[[0.5, 0.2]],
        advantages=[1.0],
        delta=0.01,
        tail_estimator="aggregated",
        vocab_size=5,
    )

    expected_kl = 0.4 * math.log(0.4 / 0.5) + 0.3 * math.log(0.3 / 0.2) + 0.3 * math.log(0.3 / 0.3)
    expected_dot = (0.5 - 0.4) + 0.5 * (0.4 - 0.5) + 0.2 * (0.3 - 0.2) + 0.3 * (0.3 - 0.3)
    torch.testing.assert_close(out["dppo_topk_kl"], torch.tensor([expected_kl], dtype=torch.float64))
    torch.testing.assert_close(out["dppo_predictive_dot"], torch.tensor([expected_dot], dtype=torch.float64))
    torch.testing.assert_close(out["dppo_behavior_tail_mass"], torch.tensor([0.3], dtype=torch.float64))
    torch.testing.assert_close(out["dppo_current_tail_mass"], torch.tensor([0.3], dtype=torch.float64))
    assert out["dppo_outside"].item() == 1.0
    assert out["dppo_predictive_increasing"].item() == 1.0
    assert out["pg_clipfrac"].item() == 1.0
    assert out["pg_upper_clipfrac"].item() == 1.0
    assert out["pg_lower_clipfrac"].item() == 0.0
    assert out["pg_losses"].item() == 0.0


def test_predictive_dot_matches_full_vocab_finite_difference():
    mu = torch.tensor([0.31, 0.27, 0.19, 0.14, 0.09], dtype=torch.float64)
    pi = torch.tensor([0.22, 0.34, 0.17, 0.18, 0.09], dtype=torch.float64)
    sampled = 2
    out = compute_dppo_predictive_topk_policy_loss(
        log_probs=pi[sampled].log().reshape(1),
        old_log_probs=mu[sampled].log().reshape(1),
        behavior_support_log_probs=mu.log().reshape(1, -1),
        current_support_log_probs=pi.log().reshape(1, -1),
        support_valid_mask=torch.ones((1, mu.numel()), dtype=torch.bool),
        advantages=torch.ones(1, dtype=torch.float64),
        delta=10.0,
        tail_estimator="aggregated",
        vocab_size=mu.numel(),
    )

    direction = -pi.clone()
    direction[sampled] += 1.0

    def forward_kl(eta):
        perturbed = torch.softmax(pi.log() + eta * direction, dim=-1)
        return (mu * (mu.log() - perturbed.log())).sum()

    epsilon = 1e-6
    finite_difference = (forward_kl(epsilon) - forward_kl(-epsilon)) / (2 * epsilon)
    torch.testing.assert_close(out["dppo_predictive_dot"].squeeze(0), finite_difference, rtol=1e-8, atol=1e-10)


def test_uniform_tail_matches_manual_formula_and_can_change_mask():
    # mu_tail=.4, pi_tail=.2 and n-m=8.  Aggregating the tail yields a
    # positive dot_D, while spreading it uniformly yields a negative dot_D.
    kwargs = dict(
        sampled_current=[0.45],
        sampled_behavior=[0.4],
        behavior_support=[[0.4, 0.2]],
        current_support=[[0.45, 0.35]],
        advantages=[1.0],
        delta=0.01,
        vocab_size=10,
    )
    aggregated = _compute(**kwargs, tail_estimator="aggregated")
    uniform = _compute(**kwargs, tail_estimator="uniform")

    retained_term = 0.45 * (0.4 - 0.45) + 0.35 * (0.2 - 0.35)
    aggregated_tail = 0.2 * (0.4 - 0.2)
    expected_aggregated_dot = (0.45 - 0.4) + retained_term + aggregated_tail
    expected_uniform_dot = (0.45 - 0.4) + retained_term + aggregated_tail / 8
    expected_kl = 0.4 * math.log(0.4 / 0.45) + 0.2 * math.log(0.2 / 0.35) + 0.4 * math.log(0.4 / 0.2)

    torch.testing.assert_close(
        aggregated["dppo_predictive_dot"], torch.tensor([expected_aggregated_dot], dtype=torch.float64)
    )
    torch.testing.assert_close(
        uniform["dppo_predictive_dot"], torch.tensor([expected_uniform_dot], dtype=torch.float64)
    )
    torch.testing.assert_close(uniform["dppo_topk_kl"], torch.tensor([expected_kl], dtype=torch.float64))
    torch.testing.assert_close(
        uniform["dppo_predictive_tail_term"], torch.tensor([aggregated_tail / 8], dtype=torch.float64)
    )
    assert aggregated["pg_clipfrac"].item() == 1.0
    assert uniform["pg_clipfrac"].item() == 0.0


def test_predictive_direction_detects_sampled_ratio_disagreement():
    # The sampled probability increased (.40 -> .45), but the distribution-wide
    # correction is -.11, so dot_D=.05-.11=-.06.  For A>0 the ratio criterion
    # says "moving out" while the predictive criterion says "moving in".
    out = _compute(
        sampled_current=[0.45],
        sampled_behavior=[0.4],
        behavior_support=[[0.4, 0.3, 0.3]],
        current_support=[[0.45, 0.5, 0.05]],
        advantages=[1.0],
        delta=1e-3,
        tail_estimator="aggregated",
        vocab_size=4,
    )

    torch.testing.assert_close(out["dppo_predictive_dot"], torch.tensor([-0.06], dtype=torch.float64))
    assert out["dppo_outside"].item() == 1.0
    assert out["dppo_ratio_increasing"].item() == 1.0
    assert out["dppo_predictive_increasing"].item() == 0.0
    assert out["dppo_direction_disagreement"].item() == 1.0
    assert out["pg_clipfrac"].item() == 0.0
    assert out["pg_losses"].item() != 0.0


def test_negative_advantage_masks_negative_predictive_direction_as_lower_clip():
    out = _compute(
        sampled_current=[0.45],
        sampled_behavior=[0.4],
        behavior_support=[[0.4, 0.3, 0.3]],
        current_support=[[0.45, 0.5, 0.05]],
        advantages=[-1.0],
        delta=1e-3,
        tail_estimator="aggregated",
        vocab_size=4,
    )
    assert out["dppo_predictive_dot"].item() < 0
    assert out["pg_clipfrac"].item() == 1.0
    assert out["pg_upper_clipfrac"].item() == 0.0
    assert out["pg_lower_clipfrac"].item() == 1.0
    assert out["pg_losses"].item() == 0.0


def test_mask_requires_both_outside_region_and_increasing_direction():
    common = dict(
        sampled_current=[0.5],
        sampled_behavior=[0.4],
        behavior_support=[[0.4, 0.3]],
        current_support=[[0.5, 0.2]],
        advantages=[1.0],
        tail_estimator="aggregated",
        vocab_size=5,
    )
    inside = _compute(**common, delta=1.0)
    outside = _compute(**common, delta=0.0)
    assert inside["dppo_predictive_increasing"].item() == 1.0
    assert inside["dppo_outside"].item() == 0.0
    assert inside["pg_clipfrac"].item() == 0.0
    assert outside["pg_clipfrac"].item() == 1.0


def test_gradient_flows_only_through_sampled_current_log_prob_and_ratio_is_capped():
    sampled_current = _log_probs([0.9], requires_grad=True)
    sampled_behavior = torch.tensor([math.log(0.9) - 5.0], dtype=torch.float64, requires_grad=True)
    behavior_support = _log_probs([[0.5, 0.2]], requires_grad=True)
    current_support = _log_probs([[0.6, 0.1]], requires_grad=True)
    advantages = torch.tensor([2.0], dtype=torch.float64, requires_grad=True)
    out = compute_dppo_predictive_topk_policy_loss(
        log_probs=sampled_current,
        old_log_probs=sampled_behavior,
        behavior_support_log_probs=behavior_support,
        current_support_log_probs=current_support,
        support_valid_mask=torch.ones((1, 2), dtype=torch.bool),
        advantages=advantages,
        delta=10.0,
        tail_estimator="aggregated",
        vocab_size=10,
        eps_clip_c=1.25,
    )
    out["pg_losses"].sum().backward()

    torch.testing.assert_close(sampled_current.grad, torch.tensor([-2.0 * 1.25], dtype=torch.float64))
    torch.testing.assert_close(out["dppo_importance_ratio"], torch.tensor([math.exp(5.0)], dtype=torch.float64))
    torch.testing.assert_close(out["dppo_importance_weight"], torch.tensor([1.25], dtype=torch.float64))
    assert sampled_behavior.grad is None
    assert behavior_support.grad is None
    assert current_support.grad is None
    assert advantages.grad is None
    for key, value in out.items():
        if key != "pg_losses":
            assert not value.requires_grad, key


def test_diagnostic_sufficient_statistics_cover_sign_clip_mass_probability_and_centered_logits():
    behavior_support = torch.tensor(
        [
            [0.4, 0.3, 0.1],
            [0.4, 0.3, 0.1],
            [0.4, 0.3, 0.3],
            [0.4, 0.3, 0.3],
        ],
        dtype=torch.float64,
    )
    current_support = torch.tensor(
        [
            [0.5, 0.2, 0.1],  # positive, outside and moving farther out
            [0.4, 0.3, 0.1],  # positive, inside
            [0.45, 0.5, 0.05],  # negative, outside and moving farther out
            [0.4, 0.3, 0.3],  # negative, inside
        ],
        dtype=torch.float64,
    )
    sampled_behavior = torch.tensor([0.4, 0.4, 0.4, 0.4], dtype=torch.float64)
    sampled_current = torch.tensor([0.5, 0.4, 0.45, 0.4], dtype=torch.float64)
    advantages = torch.tensor([1.0, 1.0, -2.0, -2.0], dtype=torch.float64)

    out = compute_dppo_predictive_topk_policy_loss(
        log_probs=sampled_current.log(),
        old_log_probs=sampled_behavior.log(),
        behavior_support_log_probs=behavior_support.log(),
        current_support_log_probs=current_support.log(),
        support_valid_mask=torch.ones_like(behavior_support, dtype=torch.bool),
        advantages=advantages,
        delta=0.01,
        tail_estimator="aggregated",
        vocab_size=10,
        eps_clip_c=5.0,
    )

    def expected(values):
        return torch.tensor(values, dtype=torch.float64)

    expected_clip = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    torch.testing.assert_close(out["pg_clipfrac"], expected_clip)
    torch.testing.assert_close(out["dppo/adv_positive_token_frac"], expected([1.0, 1.0, 0.0, 0.0]))
    torch.testing.assert_close(out["dppo/adv_negative_token_frac"], expected([0.0, 0.0, 1.0, 1.0]))
    torch.testing.assert_close(out["dppo/adv_zero_token_frac"], torch.zeros(4, dtype=torch.float64))
    torch.testing.assert_close(out["dppo/upper_clip_joint_frac"], expected([1.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(out["dppo/lower_clip_joint_frac"], expected([0.0, 0.0, 1.0, 0.0]))
    torch.testing.assert_close(out["dppo/positive_kept_joint_frac"], expected([0.0, 1.0, 0.0, 0.0]))
    torch.testing.assert_close(out["dppo/negative_kept_joint_frac"], expected([0.0, 0.0, 0.0, 1.0]))

    importance_ratio = sampled_current / sampled_behavior
    update_mass = advantages.abs() * importance_ratio
    torch.testing.assert_close(out["dppo/update_mass_positive_mean"], update_mass * expected([1.0, 1.0, 0.0, 0.0]))
    torch.testing.assert_close(out["dppo/update_mass_negative_mean"], update_mass * expected([0.0, 0.0, 1.0, 1.0]))
    torch.testing.assert_close(out["dppo/masked_update_mass_positive_mean"], expected([1.25, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(out["dppo/masked_update_mass_negative_mean"], expected([0.0, 0.0, 2.25, 0.0]))
    torch.testing.assert_close(out["dppo/kept_update_mass_mean"], expected([0.0, 1.0, 0.0, 2.0]))
    torch.testing.assert_close(out["dppo/net_logprob_push_unmasked_numerator_mean"], advantages * importance_ratio)
    torch.testing.assert_close(out["dppo/net_logprob_push_kept_numerator_mean"], expected([0.0, 1.0, 0.0, -2.0]))
    signed_update = advantages * importance_ratio
    signed_update_kept = expected([0.0, 1.0, 0.0, -2.0])
    torch.testing.assert_close(out["dppo/signed_update_unmasked"], signed_update)
    torch.testing.assert_close(out["dppo/signed_update_kept"], signed_update_kept)
    torch.testing.assert_close(out["dppo/signed_update_mask_delta"], signed_update_kept - signed_update)
    torch.testing.assert_close(
        out["dppo/sampled_logit_first_order_unmasked"],
        signed_update * (1.0 - sampled_current),
    )
    torch.testing.assert_close(
        out["dppo/sampled_logit_first_order_kept"],
        signed_update_kept * (1.0 - sampled_current),
    )

    torch.testing.assert_close(
        out["dppo/train_sampled_prob_positive_clipped_joint_mean"],
        expected([0.5, 0.0, 0.0, 0.0]),
    )
    torch.testing.assert_close(
        out["dppo/rollout_sampled_prob_negative_clipped_joint_mean"],
        expected([0.0, 0.0, 0.4, 0.0]),
    )

    def centered(values):
        logs = values.log()
        return logs - logs.mean(dim=-1, keepdim=True)

    expected_centered_diff = (centered(current_support) - centered(behavior_support)).abs().mean(dim=-1)
    torch.testing.assert_close(
        out["dppo/train_rollout_predictive_support_centered_logit_abs_diff"], expected_centered_diff
    )
    torch.testing.assert_close(
        out["dppo/rollout_predictive_support_centered_logit_std"],
        centered(behavior_support).square().mean(dim=-1).sqrt(),
    )
    torch.testing.assert_close(
        out["dppo/train_predictive_support_centered_logit_std"],
        centered(current_support).square().mean(dim=-1).sqrt(),
    )
    for key, value in out.items():
        assert torch.isfinite(value).all(), key


def test_padding_is_excluded_from_support_math_even_when_nonfinite():
    behavior = torch.tensor([[math.log(0.4), math.log(0.3), float("nan")]], dtype=torch.float64)
    current = torch.tensor([[math.log(0.5), math.log(0.2), float("inf")]], dtype=torch.float64)
    out = compute_dppo_predictive_topk_policy_loss(
        log_probs=_log_probs([0.5]),
        old_log_probs=_log_probs([0.4]),
        behavior_support_log_probs=behavior,
        current_support_log_probs=current,
        support_valid_mask=torch.tensor([[True, True, False]]),
        advantages=torch.tensor([1.0], dtype=torch.float64),
        delta=1.0,
        tail_estimator="aggregated",
        vocab_size=5,
        eps_clip_c=5.0,
    )
    assert torch.isfinite(out["dppo_topk_kl"]).all()
    torch.testing.assert_close(out["dppo_behavior_tail_mass"], torch.tensor([0.3], dtype=torch.float64))


def test_all_padding_support_produces_neutral_finite_metrics():
    out = compute_dppo_predictive_topk_policy_loss(
        # Deliberately make sampled probabilities differ: an empty-support row
        # is padding/abort data and must still have neutral direction metrics.
        log_probs=_log_probs([0.8]),
        old_log_probs=_log_probs([0.2]),
        behavior_support_log_probs=torch.tensor([[float("nan"), float("-inf")]], dtype=torch.float64),
        current_support_log_probs=torch.tensor([[float("inf"), float("nan")]], dtype=torch.float64),
        support_valid_mask=torch.tensor([[False, False]]),
        advantages=torch.tensor([1.0], dtype=torch.float64),
        delta=0.0,
        tail_estimator="uniform",
        vocab_size=5,
        eps_clip_c=5.0,
    )
    for key in (
        "dppo_topk_kl",
        "dppo_predictive_dot",
        "dppo_outside",
        "dppo_predictive_increasing",
        "dppo_ratio_increasing",
        "dppo_direction_disagreement",
    ):
        assert out[key].item() == 0.0, key
    assert torch.isfinite(out["pg_losses"]).all()


def test_small_probability_sum_roundoff_is_tolerated_and_clamped():
    p = 0.500003
    out = _compute(
        sampled_current=[0.5],
        sampled_behavior=[0.5],
        behavior_support=[[p, p]],
        current_support=[[p, p]],
        advantages=[1.0],
        delta=0.0,
        tail_estimator="aggregated",
        vocab_size=3,
    )
    assert out["dppo_behavior_tail_mass"].item() == 0.0
    assert out["dppo_current_tail_mass"].item() == 0.0
    assert out["dppo_topk_kl"].item() == 0.0


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    [
        ("shape", ValueError, "identical shapes"),
        ("support_shape", ValueError, "support log-probs"),
        ("mask_shape", ValueError, "support_valid_mask"),
        ("mask_dtype", TypeError, "torch.bool"),
        ("nonfinite", ValueError, "finite"),
        ("behavior_mass", ValueError, "behavior retained-support"),
        ("current_mass", ValueError, "current retained-support"),
        ("sampled_probability", ValueError, "sampled current"),
        ("uniform_denominator", ValueError, "vocab_size - retained_support_size"),
        ("bad_estimator", ValueError, "tail_estimator"),
        ("bad_delta", ValueError, "delta"),
        ("bad_ratio_cap", ValueError, "eps_clip_c"),
    ],
)
def test_invalid_inputs_fail_loudly(mutation, error, message):
    kwargs = dict(
        log_probs=_log_probs([0.5]),
        old_log_probs=_log_probs([0.4]),
        behavior_support_log_probs=_log_probs([[0.4, 0.3]]),
        current_support_log_probs=_log_probs([[0.5, 0.2]]),
        support_valid_mask=torch.tensor([[True, True]]),
        advantages=torch.tensor([1.0], dtype=torch.float64),
        delta=0.1,
        tail_estimator="aggregated",
        vocab_size=5,
        eps_clip_c=5.0,
    )
    if mutation == "shape":
        kwargs["old_log_probs"] = _log_probs([0.4, 0.3])
    elif mutation == "support_shape":
        kwargs["current_support_log_probs"] = _log_probs([[0.5, 0.2, 0.1]])
    elif mutation == "mask_shape":
        kwargs["support_valid_mask"] = torch.tensor([[True]])
    elif mutation == "mask_dtype":
        kwargs["support_valid_mask"] = torch.ones((1, 2), dtype=torch.float64)
    elif mutation == "nonfinite":
        kwargs["behavior_support_log_probs"] = torch.tensor([[math.log(0.4), float("nan")]], dtype=torch.float64)
    elif mutation == "behavior_mass":
        kwargs["behavior_support_log_probs"] = _log_probs([[0.6, 0.5]])
    elif mutation == "current_mass":
        kwargs["current_support_log_probs"] = _log_probs([[0.6, 0.5]])
    elif mutation == "sampled_probability":
        kwargs["log_probs"] = torch.tensor([0.1], dtype=torch.float64)
    elif mutation == "uniform_denominator":
        kwargs["tail_estimator"] = "uniform"
        kwargs["vocab_size"] = 2
    elif mutation == "bad_estimator":
        kwargs["tail_estimator"] = "mystery"
    elif mutation == "bad_delta":
        kwargs["delta"] = float("nan")
    elif mutation == "bad_ratio_cap":
        kwargs["eps_clip_c"] = 0.0

    with pytest.raises(error, match=message):
        compute_dppo_predictive_topk_policy_loss(**kwargs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
