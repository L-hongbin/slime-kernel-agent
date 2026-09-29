"""CPU contracts for reference coverage reward and PRS."""

import sys
from pathlib import Path
from types import SimpleNamespace
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.kernel_agent.kernel_reward import _compute_coverage  # noqa: E402
from examples.kernel_agent.utils import _apply_coverage_rs, _extract_env_extra_info  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

NUM_GPUS = 0


def state(custom=10.0, total=30.0, reference=0.1, correct=True):
    return dict(
        correctness=correct,
        compiled=True,
        speedup=1.0,
        reference_runtime=reference,
        metadata={"custom_kernel_cuda_time_in_profiling_us": custom, "total_kernel_run_time_in_profiling_us": total},
    )


def sample_for(s):
    return Sample(metadata={"env_extra_info": _extract_env_extra_info(s)})


def apply(sample, key="reference_time_coverage"):
    args = SimpleNamespace(
        use_coverage_rs=True, coverage_rs_key=key, coverage_rs_threshold=0.3, coverage_rs_factor=0.1
    )
    _apply_coverage_rs(args, [sample])


@pytest.mark.parametrize(
    "custom,total,expected", [(10, 30, 0.8), (40, 60, 0.8), (10, 210, 0), (10, 10, 1), (20, 10, 1)]
)
def test_reward_and_sampling_metric_match(custom, total, expected):
    s = state(custom, total)
    extra = _extract_env_extra_info(s)
    assert extra["reference_time_coverage"] == pytest.approx(expected)
    assert (
        extra["reference_time_coverage"]
        == _compute_coverage(s, {"coverage_reward_type": "reference_time_coverage"})["coverage"]
    )


@pytest.mark.parametrize("custom,total", [(10, 30), (90, 190)])
def test_selected_metric_changes_sampling_decision(monkeypatch, custom, total):
    monkeypatch.setattr("examples.kernel_agent.utils.random.random", lambda: 0.5)
    ref = sample_for(state(custom, total))
    old = sample_for(state(custom, total))
    apply(ref)
    apply(old, "time_coverage")
    # First: ref=.8/time=.33; second: ref=0/time=.47.
    assert ref.remove_sample == (custom == 90)
    assert old.remove_sample == (custom == 10)


@pytest.mark.parametrize("draw,removed", [(0.25, False), (0.75, True)])
def test_linear_probability_uses_reference_coverage(monkeypatch, draw, removed):
    monkeypatch.setattr("examples.kernel_agent.utils.random.random", lambda: draw)
    sample = sample_for(state(10, 75))  # reference coverage .35 => keep probability .5
    apply(sample)
    assert sample.remove_sample == removed


@pytest.mark.parametrize("reference", [None, 0, -1, float("nan"), float("inf")])
@pytest.mark.parametrize("correct", [True, False])
def test_invalid_reference_does_not_fall_back(reference, correct):
    sample = sample_for(state(reference=reference, correct=correct))
    apply(sample)
    assert sample.remove_sample == correct
    if correct:
        assert sample.metadata["remove_reason"] == "invalid_reference_coverage"


def test_missing_profiling_is_excluded():
    sample = sample_for(state(total=0))
    apply(sample)
    assert sample.metadata["remove_reason"] == "invalid_reference_coverage"


def test_time_coverage_still_works_without_reference(monkeypatch):
    monkeypatch.setattr("examples.kernel_agent.utils.random.random", lambda: 0.5)
    sample = sample_for(state(80, 100, None))
    apply(sample, "time_coverage")
    assert not sample.remove_sample


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
