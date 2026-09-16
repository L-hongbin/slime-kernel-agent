"""ArgMaxRL closed-form weights, grouping, and token-level advantage plumbing."""

import itertools
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import _cp_dist_helpers  # noqa: F401
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slime.backends.megatron_utils import loss as loss_module
from slime.utils.arguments import get_slime_extra_args_provider, slime_validate_args
from slime.utils.ppo_utils import get_argmaxrl_advantages, get_argmaxrl_weights

NUM_GPUS = 0


@pytest.mark.parametrize(
    "rewards,expected",
    [
        ([], []),
        ([0, 0, 0], [0, 0, 0]),
        ([2], [2]),
        ([1, 0, 1, 0], [0.5, 0, 0.5, 0]),
        ([2, 2, 2], [2 / 3] * 3),
        ([1, 2, 4], [1 / 3, 5 / 6, 17 / 6]),
        ([2, 0, 2, 1], [5 / 6, 0, 5 / 6, 1 / 3]),
    ],
)
def test_closed_form(rewards, expected):
    rewards = torch.tensor(rewards, dtype=torch.float64)
    weights = get_argmaxrl_weights(rewards)
    torch.testing.assert_close(weights, torch.tensor(expected, dtype=torch.float64))
    assert weights.sum().item() == pytest.approx(rewards.max().item() if rewards.numel() else 0)


@pytest.mark.parametrize("bad", [[-1.0, 1.0], [float("nan")], [float("inf")], [-float("inf")]])
def test_invalid_rewards_fail_loud(bad):
    with pytest.raises(ValueError, match="finite nonnegative"):
        get_argmaxrl_weights(torch.tensor(bad))


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_expected_estimator_matches_exact_harmonic_best_k_gradient(n):
    # Enumerate a small categorical policy, rather than testing against another
    # sort/cumsum implementation. This also catches missing N in A=N*w.
    logits = torch.tensor([0.3, -0.7, 0.1], dtype=torch.float64, requires_grad=True)
    probs = logits.softmax(0)
    outcomes = torch.tensor([0.0, 0.4, 1.8], dtype=torch.float64)
    objective = logits.sum() * 0
    for k in range(1, n + 1):
        for draw in itertools.product(range(3), repeat=k):
            indices = torch.tensor(draw)
            objective = objective + probs[indices].prod() * outcomes[indices].max() / k
    expected = torch.autograd.grad(objective, logits)[0]
    estimated = torch.zeros_like(logits)
    for draw in itertools.product(range(3), repeat=n):
        indices = torch.tensor(draw)
        rewards = outcomes[indices].tolist()
        advantages = get_argmaxrl_advantages(rewards, [0] * n, list(range(n)), [True] * n)
        scores = torch.nn.functional.one_hot(indices, 3) - probs.detach()
        estimated += probs[indices].prod().detach() * (torch.tensor(advantages)[:, None] * scores).mean(0)
    torch.testing.assert_close(estimated, expected, atol=1e-7, rtol=1e-7)


def test_grouping_padding_uneven_groups_and_fanout():
    # Interleaved groups; group 0 has a duplicated segment of candidate 10.
    rewards = [1.0, 2.0, 1.0, 0.0, 4.0, float("nan")]
    result = get_argmaxrl_advantages(rewards, [0, 1, 0, 0, 1, None], [10, 20, 10, 11, 21, 99], [True] * 5 + [False])
    assert result == pytest.approx([2, 2, 2, 0, 6, 0])


def test_fixed_offset_preserves_input_and_zero_is_not_shifted_by_group_min():
    rewards = [-1.0, 0.0, 2.0]
    result = get_argmaxrl_advantages(rewards, [0] * 3, list(range(3)), [True] * 3, reward_offset=1)
    assert result == pytest.approx([0, 1.5, 7.5])
    assert rewards == [-1.0, 0.0, 2.0]
    assert get_argmaxrl_advantages([2, 2], [0, 0], [0, 1], [True, True]) == [2, 2]


def test_reject_missing_group_conflicting_fanout_and_insufficient_offset():
    with pytest.raises(ValueError, match="group_index"):
        get_argmaxrl_advantages([1], [None], [0], [True])
    with pytest.raises(ValueError, match="same reward"):
        get_argmaxrl_advantages([1, 2], [0, 0], [7, 7], [True, True])
    with pytest.raises(ValueError, match="fixed offset"):
        get_argmaxrl_advantages([-1], [0], [0], [True])
    with pytest.raises(ValueError, match="finite"):
        get_argmaxrl_advantages([1], [0], [0], [True], reward_offset=float("nan"))


@pytest.mark.parametrize("estimator", ["argmaxrl", "tailrl"])
def test_cli(estimator):
    parser = get_slime_extra_args_provider()(ArgumentParser())
    args = parser.parse_args(["--rollout-batch-size", "1", "--advantage-estimator", estimator])
    assert args.advantage_estimator == estimator
    assert args.argmaxrl_reward_offset == 0
    assert not args.normalize_advantages
    assert parser.parse_args(["--rollout-batch-size", "1"]).advantage_estimator == "grpo"


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"normalize_advantages": True}, "normalize-advantages"),
        ({"argmaxrl_reward_offset": float("nan")}, "finite"),
        ({"custom_advantage_function_path": "custom.adv"}, "custom-advantage"),
        ({"kl_coef": 0.1}, "use-kl-loss"),
        ({"verify_rollout_ratio": 0.1, "verify_advantage_baseline": "history"}, "group"),
        ({"verify_rollout_ratio": 0.1, "verify_advantage_baseline": "greedy-anchor"}, "group"),
    ],
)
@pytest.mark.parametrize("estimator", ["argmaxrl", "tailrl"])
def test_validation(overrides, match, estimator):
    with pytest.raises(ValueError, match=match):
        slime_validate_args(Namespace(advantage_estimator=estimator, **overrides))


@pytest.mark.parametrize("estimator", ["grpo", "tailrl"])
def test_offset_cannot_silently_apply_to_other_estimators(estimator):
    with pytest.raises(ValueError, match="requires --advantage-estimator argmaxrl"):
        slime_validate_args(Namespace(advantage_estimator=estimator, argmaxrl_reward_offset=1.0))


@pytest.mark.parametrize("ctm", [False, True])
@pytest.mark.parametrize("estimator", ["argmaxrl", "tailrl"])
def test_precomputed_advantages_broadcast_to_local_tokens_after_dp_split(monkeypatch, ctm, estimator):
    monkeypatch.setattr(loss_module.mpu, "is_pipeline_last_stage", lambda: True, raising=False)
    args = Namespace(
        advantage_estimator=estimator,
        use_rollout_logprobs=False,
        kl_coef=0,
        custom_advantage_function_path=None,
        use_opd=False,
        normalize_advantages=False,
        use_conditional_truncation_mask=ctm,
    )
    advantages = get_argmaxrl_advantages([1, 2, 4], [0] * 3, [0, 1, 2], [True] * 3, center=estimator == "tailrl")
    # Only a subset of the prompt on this DP rank, including an empty CP slice.
    data = {
        "rewards": [advantages[2], advantages[0]],
        "log_probs": [torch.zeros(3), torch.zeros(0)],
        "response_lengths": [3, 2],
        "total_lengths": [7, 6],
        "loss_masks": [torch.ones(3), torch.ones(2)],
        "conditional_truncation_masked": [ctm, False],
    }
    loss_module.compute_advantages_and_returns(args, data)
    expected = 4.5 if estimator == "tailrl" else 8.5
    torch.testing.assert_close(data["returns"][0], torch.full((3,), expected))
    torch.testing.assert_close(data["advantages"][0], torch.full((3,), 0.0 if ctm else expected))
    assert data["advantages"][1].numel() == 0


@pytest.mark.parametrize("values", [[], [7.0], [-2.0], [0, 0], [2, 2, 2], [0, 1, 3], [-5, -2, -2, 1], [3, 1, 1, 0]])
def test_tailrl_matches_official_ascending_gap_centered_implementation(values):
    r = torch.tensor(values, dtype=torch.float64)
    n = len(values)
    actual = get_argmaxrl_advantages(values, [0] * n, list(range(n)), [True] * n, center=True)
    if n <= 1:
        expected = torch.zeros_like(r)
    else:
        rs, order = r.sort()
        gaps = rs - torch.cat([rs.new_zeros(1), rs[:-1]])
        w = (gaps / torch.arange(n, 0, -1, dtype=r.dtype)).cumsum(0) * n
        expected = torch.empty_like(w).scatter_(0, order, w - w.mean())
    torch.testing.assert_close(torch.tensor(actual, dtype=torch.float64), expected)
    assert sum(actual) == pytest.approx(0.0, abs=1e-12)


def test_tailrl_binary_recovery_translation_invariance_and_unique_candidate_center():
    assert get_argmaxrl_advantages([0, 1, 0, 1], [0] * 4, list(range(4)), [True] * 4, center=True) == [-1, 1, -1, 1]
    expected = [-3.0, -1.5, 4.5]
    for shift in [-100, 0, 1e10]:
        r = [value + shift for value in [0, 1, 3]]
        assert get_argmaxrl_advantages(r, [0] * 3, list(range(3)), [True] * 3, center=True) == expected
    # Count candidate 0 once when centering, despite its two fan-out segments.
    actual = get_argmaxrl_advantages(
        [0, 0, 1, 3, float("nan")], [0] * 5, [0, 0, 1, 2, 9], [True] * 4 + [False], center=True
    )
    assert actual == [-3, -3, -1.5, 4.5, 0]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_tailrl_rejects_nonfinite_rewards(value):
    with pytest.raises(ValueError, match="TailRL requires finite"):
        get_argmaxrl_advantages([value], [0], [0], [True], center=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
