"""Unit tests for the DAPO-style soft overlong penalty + filter semantics.

With cap = rollout_max_response_len and buffer B, responses longer than cap-B
lose factor*min(1, exceed/B) training reward. The low-variance filter still
judges the pre-penalty task reward stored in metadata["task_reward"].
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

NUM_GPUS = 0


def _utils():
    import examples.kernel_agent.utils as utils

    return utils


def _args(penalty_on: bool, buffer_len: int = 2048, factor: float = 1.0):
    return SimpleNamespace(
        rollout_max_response_len=16384,
        overlong_penalty=penalty_on,
        overlong_buffer_len=buffer_len,
        overlong_penalty_factor=factor,
    )


def _sample(reward, response_length, removed=False):
    return SimpleNamespace(
        reward=reward,
        response_length=response_length,
        remove_sample=removed,
        metadata={},
    )


def test_disabled_by_default():
    sample = _sample(1.0, 16384)
    _utils()._apply_overlong_penalty(_args(False), [sample])
    assert sample.reward == 1.0
    assert "overlong_penalty" not in sample.metadata


def test_no_penalty_below_threshold():
    sample = _sample(1.0, 16384 - 2048)
    _utils()._apply_overlong_penalty(_args(True), [sample])
    assert sample.reward == 1.0


def test_linear_ramp_and_cap():
    half = _sample(1.0, 16384 - 1024)
    full = _sample(1.0, 16384)
    over = _sample(0.0, 16384)
    _utils()._apply_overlong_penalty(_args(True), [half, full, over])
    assert half.reward == pytest.approx(0.5)
    assert full.reward == pytest.approx(0.0)
    assert over.reward == pytest.approx(-1.0)
    assert half.metadata["overlong_penalty"] == pytest.approx(0.5)
    assert full.metadata["overlong_penalty"] == pytest.approx(1.0)


def test_removed_samples_untouched():
    sample = _sample(1.0, 16384, removed=True)
    _utils()._apply_overlong_penalty(_args(True), [sample])
    assert sample.reward == 1.0


def test_custom_factor_and_buffer():
    sample = _sample(2.0, 16384 - 500)
    _utils()._apply_overlong_penalty(_args(True, buffer_len=1000, factor=0.5), [sample])
    assert sample.reward == pytest.approx(1.75)


def test_task_reward_recorded_for_filter():
    fail_short = _sample(0.0, 5000)
    fail_long = _sample(0.0, 16384)
    _utils()._apply_overlong_penalty(_args(True), [fail_short, fail_long])
    assert "task_reward" not in fail_short.metadata
    assert fail_long.metadata["task_reward"] == 0.0
    assert fail_long.reward == -1.0
    filter_view = [sample.metadata.get("task_reward", sample.reward) for sample in (fail_short, fail_long)]
    assert filter_view == [0.0, 0.0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
