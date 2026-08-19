"""CPU mathematical audit of predictive-DPPO gradients and entropy effects.

The fixed batch uses full softmax distributions so the production loss can be
checked at the logits level, rather than treating sampled log-probabilities as
independent scalars.  It deliberately contains one outward and one corrective
update for each advantage sign.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slime.utils import ppo_utils  # noqa: E402
from slime.utils.ppo_utils import compute_dppo_predictive_topk_policy_loss, compute_policy_loss  # noqa: E402

NUM_GPUS = 0
DTYPE = torch.float64


def _fixed_batch():
    # Every row samples token 0.  Rows 0/2 are outward updates and should be
    # masked at delta=.2; rows 1/3 are corrective and must remain active.
    behavior = torch.tensor(
        [
            [0.15, 0.65, 0.10, 0.06, 0.04],
            [0.45, 0.10, 0.35, 0.06, 0.04],
            [0.45, 0.10, 0.35, 0.06, 0.04],
            [0.15, 0.65, 0.10, 0.06, 0.04],
        ],
        dtype=DTYPE,
    )
    current = torch.tensor(
        [
            [0.45, 0.10, 0.35, 0.06, 0.04],
            [0.15, 0.65, 0.10, 0.06, 0.04],
            [0.15, 0.65, 0.10, 0.06, 0.04],
            [0.45, 0.10, 0.35, 0.06, 0.04],
        ],
        dtype=DTYPE,
    )
    advantages = torch.tensor([1.0, 1.0, -2.0, -2.0], dtype=DTYPE)
    sampled_ids = torch.zeros(4, dtype=torch.long)
    return behavior, current, advantages, sampled_ids


def _predictive_analysis(delta: float, *, sign: str | None = None):
    behavior, current, advantages, sampled_ids = _fixed_batch()
    logits = current.log().clone().requires_grad_()
    log_probs = logits.log_softmax(dim=-1)
    sampled_log_probs = log_probs.gather(1, sampled_ids.unsqueeze(1)).squeeze(1)
    output = compute_dppo_predictive_topk_policy_loss(
        log_probs=sampled_log_probs,
        old_log_probs=behavior[:, 0].log(),
        behavior_support_log_probs=behavior.log(),
        current_support_log_probs=log_probs.detach(),
        support_valid_mask=torch.ones_like(behavior, dtype=torch.bool),
        advantages=advantages,
        delta=delta,
        tail_estimator="aggregated",
        vocab_size=current.size(1),
        eps_clip_c=5.0,
    )

    selected_losses = output["pg_losses"]
    if sign == "positive":
        selected_losses = selected_losses * (advantages > 0)
    elif sign == "negative":
        selected_losses = selected_losses * (advantages < 0)
    elif sign is not None:
        raise ValueError(sign)

    loss = selected_losses.mean()
    loss_gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    update_direction = -loss_gradient
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    entropy_gradient = torch.autograd.grad(entropy.sum(), logits)[0]
    entropy_first_order = (entropy_gradient * update_direction).sum(dim=-1)
    sampled_logit_first_order = update_direction.gather(1, sampled_ids.unsqueeze(1)).squeeze(1)
    return {
        "output": output,
        "base_logits": logits.detach(),
        "update_direction": update_direction.detach(),
        "sampled_logit_first_order": sampled_logit_first_order.detach(),
        "entropy_first_order": entropy_first_order.detach(),
    }


def _entropy(logits):
    log_probs = logits.log_softmax(dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def test_full_vocab_entropy_directional_moment_matches_softmax_and_backward(monkeypatch):
    # A size-one TP group makes every collective an identity.  This exercises
    # the exact production chunked entropy+moment path without requiring GPUs
    # or distributed initialization.
    monkeypatch.setattr(ppo_utils.dist, "all_reduce", lambda tensor, **kwargs: tensor)
    monkeypatch.setattr(
        ppo_utils,
        "compute_log_probs",
        lambda logits, tokens, _group: logits.log_softmax(dim=-1).gather(1, tokens.unsqueeze(1)),
    )

    logits = torch.tensor(
        [
            [0.7, -0.3, 0.2, -1.1, 0.4],
            [-0.2, 0.9, 0.1, 0.3, -0.5],
            [1.2, -0.4, 0.0, 0.2, -0.7],
        ],
        dtype=DTYPE,
        requires_grad=True,
    )
    tokens = torch.tensor([0, 3, 2], dtype=torch.long)
    actual_log_prob, actual_entropy, actual_moment = ppo_utils.calculate_log_probs_and_entropy(
        logits,
        tokens,
        None,
        with_entropy=True,
        chunk_size=2,
        with_dppo_directional_moment=True,
    )

    expected_log_probs = logits.detach().log_softmax(dim=-1)
    expected_probs = expected_log_probs.exp()
    expected_entropy = -(expected_probs * expected_log_probs).sum(dim=-1)
    expected_moment = (expected_probs.square() * (expected_log_probs + expected_entropy.unsqueeze(-1))).sum(dim=-1)

    torch.testing.assert_close(
        actual_log_prob.squeeze(-1),
        expected_log_probs.gather(1, tokens.unsqueeze(1)).squeeze(1),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(actual_entropy, expected_entropy, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_moment, expected_moment, rtol=1e-12, atol=1e-12)
    assert not actual_moment.requires_grad

    actual_entropy.sum().backward()
    expected_entropy_gradient = -expected_probs * (expected_log_probs + expected_entropy.unsqueeze(-1))
    torch.testing.assert_close(logits.grad, expected_entropy_gradient, rtol=1e-12, atol=1e-12)


def test_mask_off_predictive_gradient_matches_unclipped_ppo_on_the_same_logits():
    behavior, current, advantages, sampled_ids = _fixed_batch()

    ppo_logits = current.log().clone().requires_grad_()
    ppo_sampled = ppo_logits.log_softmax(dim=-1).gather(1, sampled_ids.unsqueeze(1)).squeeze(1)
    ppo_kl = behavior[:, 0].log() - ppo_sampled
    # Use the undecorated implementation to keep this CPU regression test from
    # invoking torch.compile.  Bounds of 10 leave all ratios unclipped.
    ppo_losses, ppo_clip = compute_policy_loss.__wrapped__(ppo_kl, advantages, 10.0, 10.0)
    ppo_gradient = torch.autograd.grad(ppo_losses.mean(), ppo_logits)[0]

    predictive_off = _predictive_analysis(10.0)
    predictive_gradient = -predictive_off["update_direction"]

    assert ppo_clip.count_nonzero().item() == 0
    assert predictive_off["output"]["pg_clipfrac"].count_nonzero().item() == 0
    torch.testing.assert_close(predictive_gradient, ppo_gradient, rtol=1e-12, atol=1e-12)


def test_delta_point_two_masks_the_outward_row_for_each_advantage_sign():
    mask_off = _predictive_analysis(10.0)
    masked = _predictive_analysis(0.2)
    positive = _predictive_analysis(0.2, sign="positive")
    negative = _predictive_analysis(0.2, sign="negative")

    torch.testing.assert_close(
        masked["output"]["pg_clipfrac"],
        torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=DTYPE),
    )
    torch.testing.assert_close(
        masked["output"]["pg_upper_clipfrac"],
        torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=DTYPE),
    )
    torch.testing.assert_close(
        masked["output"]["pg_lower_clipfrac"],
        torch.tensor([0.0, 0.0, 1.0, 0.0], dtype=DTYPE),
    )

    # Masking only zeroes outward rows; it does not flip either kept gradient.
    assert masked["update_direction"][[0, 2]].count_nonzero().item() == 0
    torch.testing.assert_close(
        masked["update_direction"][[1, 3]],
        mask_off["update_direction"][[1, 3]],
    )
    assert masked["sampled_logit_first_order"][1] > 0  # positive advantage
    assert masked["sampled_logit_first_order"][3] < 0  # negative advantage
    # Production emits the pre-reduction per-token coefficient; the audit's
    # autograd direction includes this test's mean-over-four loss reduction.
    token_count = masked["sampled_logit_first_order"].numel()
    torch.testing.assert_close(
        mask_off["output"]["dppo/sampled_logit_first_order_unmasked"] / token_count,
        mask_off["sampled_logit_first_order"],
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        masked["output"]["dppo/sampled_logit_first_order_kept"] / token_count,
        masked["sampled_logit_first_order"],
        rtol=1e-12,
        atol=1e-12,
    )

    # Positive/negative strata are additive contributions to the same batch
    # reduction, so they must reconstruct the full update exactly.
    torch.testing.assert_close(
        positive["update_direction"] + negative["update_direction"],
        masked["update_direction"],
    )


def test_mask_can_increase_entropy_trend_while_normalized_push_becomes_more_negative():
    mask_off = _predictive_analysis(10.0)
    masked = _predictive_analysis(0.2)
    output = masked["output"]

    unmasked_abs_mass = (output["dppo/update_mass_positive_mean"] + output["dppo/update_mass_negative_mean"]).sum()
    kept_abs_mass = output["dppo/kept_update_mass_mean"].sum()
    unmasked_push = output["dppo/net_logprob_push_unmasked_numerator_mean"].sum() / unmasked_abs_mass
    kept_push = output["dppo/net_logprob_push_kept_numerator_mean"].sum() / kept_abs_mass

    # This is an explicit counterexample to interpreting the existing
    # mask_induced_push metric as an entropy-direction metric.
    assert kept_push - unmasked_push < 0
    assert masked["entropy_first_order"].sum() > mask_off["entropy_first_order"].sum()
    assert mask_off["entropy_first_order"][[0, 2]].max() < 0
    assert masked["entropy_first_order"][[0, 2]].count_nonzero().item() == 0

    # Independently verify the autograd first-order entropy calculation with a
    # centered finite difference along each computed optimizer direction.
    epsilon = 1e-6
    for analysis in (mask_off, masked):
        base = analysis["base_logits"]
        direction = analysis["update_direction"]
        finite_difference = (_entropy(base + epsilon * direction) - _entropy(base - epsilon * direction)) / (
            2 * epsilon
        )
        torch.testing.assert_close(analysis["entropy_first_order"], finite_difference, rtol=2e-8, atol=2e-10)


def test_aggregated_and_uniform_tail_terms_match_their_finite_difference_models():
    behavior_support = torch.tensor([[0.25, 0.35]], dtype=DTYPE)
    current_support = torch.tensor([[0.45, 0.35]], dtype=DTYPE)
    valid = torch.ones_like(behavior_support, dtype=torch.bool)
    vocab_size = 10

    for estimator in ("aggregated", "uniform"):
        output = compute_dppo_predictive_topk_policy_loss(
            log_probs=current_support[:, 0].log(),
            old_log_probs=behavior_support[:, 0].log(),
            behavior_support_log_probs=behavior_support.log(),
            current_support_log_probs=current_support.log(),
            support_valid_mask=valid,
            advantages=torch.ones(1, dtype=DTYPE),
            delta=10.0,
            tail_estimator=estimator,
            vocab_size=vocab_size,
            eps_clip_c=5.0,
        )

        behavior_tail = 1.0 - behavior_support.sum()
        current_tail = 1.0 - current_support.sum()
        if estimator == "aggregated":
            behavior_model = torch.cat((behavior_support[0], behavior_tail.reshape(1)))
            current_model = torch.cat((current_support[0], current_tail.reshape(1)))
        else:
            tail_size = vocab_size - behavior_support.size(1)
            behavior_model = torch.cat((behavior_support[0], behavior_tail.repeat(tail_size) / tail_size))
            current_model = torch.cat((current_support[0], current_tail.repeat(tail_size) / tail_size))

        direction = -current_model.clone()
        direction[0] += 1.0

        def forward_kl(
            step,
            current=current_model,
            perturbation=direction,
            behavior=behavior_model,
        ):
            perturbed = torch.softmax(current.log() + step * perturbation, dim=-1)
            return (behavior * (behavior.log() - perturbed.log())).sum()

        epsilon = 1e-6
        finite_difference = (forward_kl(epsilon) - forward_kl(-epsilon)) / (2 * epsilon)
        torch.testing.assert_close(output["dppo_predictive_dot"].squeeze(0), finite_difference, rtol=1e-8, atol=1e-10)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
