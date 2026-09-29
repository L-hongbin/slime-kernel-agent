"""CPU contracts for coverage auxiliary scores and their shared measurement gates."""

import copy
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.kernel_agent.kernel_coverage import _compute_coverage  # noqa: E402

NUM_GPUS = 0


def result(custom=600.0, total=1000.0, reference=1.0, candidate=2.0):
    return {
        "reference_runtime": reference,
        "kernel_runtime": candidate,
        "metadata": {
            "custom_kernel_cuda_time_in_profiling_us": custom,
            "total_kernel_run_time_in_profiling_us": total,
        },
    }


@pytest.mark.parametrize(
    "custom,total,candidate,expected_reference,expected_speed",
    [(600, 1000, 2, 0.6, 0.5), (600, 1000, 0.5, 0.6, 1), (1000, 1000, 2, 1, 0.5), (200, 1900, 2, 0, 0.5)],
)
def test_auxiliary_scores_share_diagnostics_without_mutating_input(
    custom, total, candidate, expected_reference, expected_speed
):
    value = result(custom=custom, total=total, candidate=candidate)
    before = copy.deepcopy(value)
    scores = {
        mode: _compute_coverage(value, {"coverage_reward_type": mode})
        for mode in (
            "reference_time_coverage",
            "time_coverage",
            "efficiency_reference_time_coverage",
            "capped_speed_auxiliary",
            "gated_time_coverage",
        )
    }
    assert scores["reference_time_coverage"]["coverage"] == pytest.approx(expected_reference)
    assert scores["efficiency_reference_time_coverage"]["coverage"] == pytest.approx(
        expected_reference * expected_speed
    )
    assert scores["capped_speed_auxiliary"]["coverage"] == pytest.approx(expected_speed)
    # A zero reference fraction must not suppress eligible time coverage.
    assert scores["gated_time_coverage"]["coverage"] == pytest.approx(custom / total)
    assert scores["gated_time_coverage"]["coverage"] == scores["time_coverage"]["coverage"]
    diagnostics = {k: v for k, v in scores["efficiency_reference_time_coverage"].items() if k != "coverage"}
    for mode in ("capped_speed_auxiliary", "gated_time_coverage"):
        assert {k: v for k, v in scores[mode].items() if k != "coverage"} == diagnostics
    assert value == before


@pytest.mark.parametrize(
    "value",
    [
        result(custom=0),
        result(custom=-1),
        result(custom=1001),
        result(custom=True),
        result(total=0),
        result(total=float("inf")),
        result(reference=-1),
        result(reference=float("nan")),
        result(reference=None),
        result(candidate=0),
        result(candidate=float("inf")),
        result(candidate=True),
    ],
)
def test_invalid_or_no_custom_measurements_share_zero_credit_and_reason(value):
    scores = [
        _compute_coverage(value, {"coverage_reward_type": mode})
        for mode in ("efficiency_reference_time_coverage", "capped_speed_auxiliary", "gated_time_coverage")
    ]
    assert all(score["coverage"] == 0 for score in scores)
    assert all(score["coverage_invalid_reason"] == scores[0]["coverage_invalid_reason"] for score in scores)


def test_explicit_measurement_invalid_flag_is_preserved():
    value = result()
    value["metadata"]["coverage_measurement_valid"] = False
    for mode in ("efficiency_reference_time_coverage", "capped_speed_auxiliary", "gated_time_coverage"):
        assert _compute_coverage(value, {"coverage_reward_type": mode})["coverage"] == 0


def test_zero_custom_with_positive_reference_coverage_earns_no_auxiliary():
    value = result(custom=0, total=100, candidate=0.5)
    assert _compute_coverage(value, {"coverage_reward_type": "reference_time_coverage"})["coverage"] == pytest.approx(
        0.9
    )
    for mode in ("efficiency_reference_time_coverage", "capped_speed_auxiliary", "gated_time_coverage"):
        assert _compute_coverage(value, {"coverage_reward_type": mode})["coverage"] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
