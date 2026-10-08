"""The speed auxiliary ablation changes h only; measurement gates stay shared."""
import copy

import pytest

from examples.kernel_agent.kernel_coverage import _compute_coverage


def result(custom=600.0, total=1000.0, reference=1.0, candidate=2.0):
    return {
        "reference_runtime": reference,
        "kernel_runtime": candidate,
        "metadata": {
            "custom_kernel_cuda_time_in_profiling_us": custom,
            "total_kernel_run_time_in_profiling_us": total,
        },
    }


def metric(value, mode):
    return _compute_coverage(value, {"coverage_reward_type": mode})


@pytest.mark.parametrize(
    "custom,total,candidate,expected_h,expected_speed",
    [(600, 1000, 2, .6, .5), (600, 1000, .5, .6, 1),
     (1000, 1000, 2, 1, .5), (200, 1900, 2, 0, .5)],
)
def test_h_is_removed_only_from_reward(custom, total, candidate, expected_h, expected_speed):
    value = result(custom=custom, total=total, candidate=candidate)
    before = copy.deepcopy(value)
    eff = metric(value, "efficiency_reference_time_coverage")
    speed = metric(value, "capped_speed_auxiliary")
    ref = metric(value, "reference_time_coverage")
    assert ref["coverage"] == pytest.approx(expected_h)
    assert eff["coverage"] == pytest.approx(expected_h * expected_speed)
    assert speed["coverage"] == pytest.approx(expected_speed)
    assert {k: v for k, v in speed.items() if k != "coverage"} == {
        k: v for k, v in eff.items() if k != "coverage"
    }
    assert value == before


@pytest.mark.parametrize("value", [
    result(custom=0), result(custom=1001), result(custom=-1),
    result(reference=-1), result(reference=float("nan")),
    result(candidate=0), result(candidate=float("inf")), result(total=0),
    result(custom=True),
])
def test_invalid_or_no_custom_measurements_earn_no_auxiliary(value):
    assert metric(value, "capped_speed_auxiliary")["coverage"] == 0
    assert metric(value, "efficiency_reference_time_coverage")["coverage"] == 0


def test_explicit_measurement_invalid_flag_is_preserved():
    value = result()
    value["metadata"]["coverage_measurement_valid"] = False
    assert metric(value, "capped_speed_auxiliary")["coverage"] == 0


def test_zero_custom_with_positive_ref_coverage_keeps_eff_gate():
    value = result(custom=0, total=100, candidate=.5)
    assert metric(value, "reference_time_coverage")["coverage"] == pytest.approx(.9)
    assert metric(value, "capped_speed_auxiliary")["coverage"] == 0
