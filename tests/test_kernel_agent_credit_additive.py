"""CPU contracts for speedup-qualified additive source credit on baseline TRLOO."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

NUM_GPUS = 0
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import examples

examples.__path__ = [str(ROOT / "examples"), *examples.__path__]

from examples.kernel_agent.component_reward import (
    ADDITIVE_METHOD,
    ADDITIVE_MODE,
    ComponentRewardContractError,
    compute_component_reward_metrics,
    filter_component_reward_group,
)
from examples.kernel_agent.kernel_filter import filter_cuda_kernel_group
from examples.kernel_agent.kernel_reward import reward_post_process_by_group
from examples.kernel_agent.utils import postprocess_turn_samples
from test_kernel_agent_component_reward import args as base_args
from test_kernel_agent_source_reward import response
from test_kernel_agent_source_reward import sample as source_sample

from slime.utils.arguments import _validate_component_reward_args
from slime.utils.types import Sample


def args(**changes):
    return base_args(
        **{
            "component_reward_backend": "source",
            "component_reward_mode": ADDITIVE_MODE,
            "component_reward_scale": 0.25,
            "component_reward_min_speedup": 1.0,
            "finalize_mode": "positive",
            "dynamic_sampling_filter_path": "examples.kernel_agent.kernel_filter.filter_cuda_kernel_group",
            **changes,
        }
    )


def sample(turn, code, reward, correct, speedup=1.2):
    s = source_sample(turn, code, reward, correct)
    s.metadata["env_extra_info"]["speedup"] = speedup
    return s


def finalize_pair(samples, **changes):
    baseline, additive = copy.deepcopy(samples), copy.deepcopy(samples)
    postprocess_turn_samples(args(component_reward=False, **changes), baseline, "max_turns")
    postprocess_turn_samples(args(**changes), additive, "max_turns")
    for old, new in zip(baseline, additive, strict=True):
        assert old.reward == new.reward
        assert old.remove_sample == new.remove_sample
        assert old.loss_mask == new.loss_mask
        assert old.tokens == new.tokens
        assert old.metadata == {k: v for k, v in new.metadata.items() if k != "component_reward"}
    reward_post_process_by_group(args(**changes), additive)
    return baseline, additive


def multiple_sources():
    return [sample(0, response(b=99), 0.5, True), sample(1, response(a=99), 1, True), sample(2, response(), 2, True)]


def test_adds_once_without_capping_or_changing_best_and_later_targets():
    _, rows = finalize_pair(
        [sample(0, response(), 1, True), sample(1, response(), 2, True), sample(2, response(), 1.5, True)]
    )
    r = rows[0].metadata["component_reward"]
    assert r["method"] == ADDITIVE_METHOD
    assert r["source_turns"] == [0]
    assert r["credits"] == [2, 0, 0]
    assert r["baseline_returns"] == [4.5, 3.5, 1.5]
    assert r["scaled_credits"] == [0.5, 0, 0]
    assert r["targets"] == [5, 3.5, 1.5]
    assert r["targets"][0] > r["quality_budget"]
    assert "cap_factor" not in r


def test_multiple_origins_share_budget_without_repeating_future_fold():
    _, rows = finalize_pair(multiple_sources())
    r = rows[0].metadata["component_reward"]
    assert r["credits"] == [1, 1, 0]
    assert r["scaled_credits"] == [0.25, 0.25, 0]
    assert r["targets"] == [3.75, 3.25, 2]
    assert sum(r["scaled_credits"]) == 0.25 * r["quality_budget"]


@pytest.mark.parametrize("scale", [0, 0.25, 0.5])
def test_coefficient_changes_addition_even_for_a_single_source(scale):
    _, rows = finalize_pair(
        [sample(0, response(), 0.5, True), sample(1, response(), 2, True)], component_reward_scale=scale
    )
    r = rows[0].metadata["component_reward"]
    assert r["targets"] == [2.5 + 2 * scale, 2]


def test_zero_scale_is_exact_baseline_including_multiple_sources():
    _, rows = finalize_pair(multiple_sources(), component_reward_scale=0)
    r = rows[0].metadata["component_reward"]
    assert r["targets"] == r["baseline_returns"]


@pytest.mark.parametrize("speedup", [None, False, "1.2", float("nan"), float("inf"), -0.1, 0, 0.999999])
def test_missing_invalid_or_slow_speedup_cannot_anchor(speedup):
    _, rows = finalize_pair([sample(0, response(), 0, False), sample(1, response(), 2, True, speedup)])
    r = rows[0].metadata["component_reward"]
    assert r["best_turn"] is None
    assert r["status"] == "no_qualifying_anchor"
    assert r["credits"] == [0, 0]
    assert r["targets"] == r["baseline_returns"]


def test_boundary_one_qualifies_and_failed_source_is_allowed():
    _, rows = finalize_pair([sample(0, response(), 0, False, 0.2), sample(1, response(), 2, True, 1.0)])
    r = rows[0].metadata["component_reward"]
    assert r["anchor_eligible_turns"] == [1]
    assert r["source_turns"] == [0]
    assert r["targets"] == [2.5, 2]


def test_correctness_is_required_even_with_fast_or_high_scoring_failure():
    _, rows = finalize_pair(
        [
            sample(0, response(), 0, False, 9),
            sample(1, response(), 0.25, False, 9),
            sample(2, response(), 0.25, False, 9),
        ]
    )
    r = rows[0].metadata["component_reward"]
    assert r["best_turn"] is None
    assert r["scaled_credits"] == [0, 0, 0]
    assert r["targets"] == [0.5, 0.5, 0.25]
    assert [s.reward for s in rows] == [0, 0.25, 0.25]


def test_choose_anchor_inside_qualified_set_not_global_highest_score():
    _, rows = finalize_pair(
        [
            sample(0, response(), 0, False),
            sample(1, response(), 1.2, True, 1.1),
            sample(2, response(a=99, b=99), 1.5, True, 0.9),
        ]
    )
    r = rows[0].metadata["component_reward"]
    assert r["best_turn"] == 1
    assert r["quality_budget"] == 1.2
    assert r["targets"] == pytest.approx([3, 2.7, 1.5])


def test_first_best_and_unresolved_components_get_no_extra_bonus():
    for code in [response(), "no usable code"]:
        _, rows = finalize_pair([sample(0, code, 2, True), sample(1, response(), 1, True)])
        r = rows[0].metadata["component_reward"]
        assert r["scaled_credits"] == [0, 0]
        assert r["targets"] == r["baseline_returns"]


def test_removed_source_is_not_resurrected_and_future_return_is_preserved():
    _, rows = finalize_pair(
        [sample(0, response(a=99, b=99), 0.5, True), sample(1, response(), 0, False), sample(2, response(), 2, True)]
    )
    r = rows[1].metadata["component_reward"]
    assert rows[1].remove_sample
    assert rows[1].metadata["remove_reason"] == "finalize_positive"
    assert r["source_turns"] == []
    assert r["targets"] == [2.5, 2, 2]


@pytest.mark.parametrize("exclusion", ["removed", "decoy", "pad", "aborted", "zero_mask", "different_task"])
def test_ineligible_origins_cannot_receive_extra_credit(exclusion):
    first, last = sample(0, response(), 0, False), sample(1, response(), 1.2, True)
    if exclusion == "removed":
        first.remove_sample = True
    elif exclusion == "decoy":
        first.metadata["env_extra_info"]["decoy_kernel"] = True
    elif exclusion == "pad":
        first.metadata["is_pad_turn"] = True
    elif exclusion == "aborted":
        first.status = Sample.Status.ABORTED
    elif exclusion == "zero_mask":
        first.loss_mask = [0] * first.response_length
    else:
        first.metadata["source_component_identity"]["task_sha256"] = "another task"
    _, rows = finalize_pair([first, last])
    r = rows[0].metadata["component_reward"]
    assert r["source_turns"] == []
    assert r["targets"] == r["baseline_returns"]


def test_filter_remains_baseline_and_component_filter_is_forbidden():
    reps = []
    for i in range(2):
        _, rows = finalize_pair([sample(0, response(), 0, False), sample(1, response(), 1, True)])
        for s in rows:
            s.index = i
        reps.append(rows[-1])
    assert not filter_cuda_kernel_group(args(), reps).keep
    with pytest.raises(ComponentRewardContractError, match="baseline dynamic filter"):
        filter_component_reward_group(args(), reps)


def test_loo_does_not_indirectly_raise_the_uncredited_peer():
    _, a = finalize_pair([sample(0, response(), 0.5, True), sample(1, response(), 2, True)])
    _, b = finalize_pair([sample(0, response(a=99, b=99), 0.5, True), sample(1, response(), 2, True)])
    for s in b:
        s.index = 1
    raw, adv = reward_post_process_by_group(args(), [a[0], b[0], a[1], b[1]])
    assert raw == [3, 2.5, 2, 2]
    assert adv == [0.5, -0.5, 0, 0]


@pytest.mark.parametrize(
    "mutation",
    [
        "method",
        "mode",
        "scale",
        "gate",
        "nan",
        "vector",
        "best_target",
        "baseline",
        "mask",
        "speedup",
        "correctness",
        "anchor",
    ],
)
def test_stale_or_inconsistent_records_cannot_reach_training(mutation):
    _, rows = finalize_pair(multiple_sources())
    r = rows[0].metadata["component_reward"]
    if mutation == "method":
        r["method"] = "trloo-component-credit-cap/v1"
    elif mutation == "mode":
        r["mode"] = "trloo-credit-cap"
    elif mutation == "scale":
        r["scale"] = 0.5
    elif mutation == "gate":
        r["anchor_min_speedup"] = 0.9
    elif mutation == "nan":
        r["turn_target"] = float("nan")
    elif mutation == "vector":
        r["targets"] = r["targets"][:-1]
    elif mutation == "best_target":
        r["targets"][2] += 1
    elif mutation == "baseline":
        rows[0].metadata["multi_turn_reward"] += 1
    elif mutation == "mask":
        rows[0].loss_mask = [0] * rows[0].response_length
    elif mutation == "speedup":
        rows[0].metadata["env_extra_info"]["speedup"] = 0.8
    elif mutation == "correctness":
        rows[0].metadata["env_extra_info"]["correctness"] = False
    else:
        r["best_turn"] = 1
    with pytest.raises(ComponentRewardContractError):
        reward_post_process_by_group(args(), rows)


def test_double_postprocess_is_rejected():
    _, rows = finalize_pair(multiple_sources())
    with pytest.raises(ComponentRewardContractError, match="already finalized"):
        postprocess_turn_samples(args(), rows, "max_turns")


def test_nonfinite_derived_bonus_fails():
    with pytest.raises(ComponentRewardContractError, match="finite"):
        finalize_pair([sample(0, response(), 0, False), sample(1, response(), 2, True)], component_reward_scale=1e308)


def test_metrics_distinguish_qualification_source_coverage_and_positive_bonus():
    _, rows = finalize_pair(multiple_sources())
    m = compute_component_reward_metrics(args(), rows + rows)
    assert m["component_reward/trajectories"] == 1
    assert m["component_reward/additive/qualified_trajectories"] == 1
    assert m["component_reward/additive/source_turns"] == 2
    assert m["component_reward/additive/applied_fraction"] == 1
    assert m["component_reward/additive/scaled_credit_sum"] == 0.5
    assert m["component_reward/additive/target_delta_sum"] == 0.5
    assert m["component_reward/additive/delta_by_turn/2/sum"] == 0
    assert not any("cap/" in key for key in m)


@pytest.mark.parametrize(
    "changes",
    [
        {"component_reward_backend": "runtime_graph"},
        {"component_reward_mode": "trloo-credit-cap"},
        {"dynamic_sampling_filter_path": "examples.kernel_agent.component_reward.filter_component_reward_group"},
        {"filter_by_last_turn": False},
        {"multi_turn_gamma": 0},
        {"component_reward_scale": -1},
        {"component_reward_scale": float("nan")},
        {"component_reward_min_speedup": float("inf")},
        {"component_reward_min_speedup": -0.1},
    ],
)
def test_argument_contract_rejects_invalid_additive_settings(changes):
    with pytest.raises(ValueError):
        _validate_component_reward_args(args(**changes))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
