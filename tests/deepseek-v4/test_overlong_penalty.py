"""Unit tests for the DAPO-style soft overlong penalty calculation.

The per-sample cap is min(response cap, context cap - prompt length). Responses
longer than cap-B lose factor*min(1, exceed/B) training reward.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

NUM_GPUS = 0

from slime.utils.types import Sample


def _kernel_reward(args, sample):
    from examples.kernel_agent.kernel_reward import calculate_kernel_reward, post_process_rollout_rewards

    config = {
        "init_correct_weight": 0.5,
        "failed_score": 0.0,
        "apply_kernel_failed_score": False,
        "apply_failed_group_reward": False,
        "kernel_failed_score": {"output_mismatch": 0.0, "other": -1.0},
    }
    details = calculate_kernel_reward({}, config)
    sample.reward = details.pop("reward")
    sample.metadata.update(details)
    reward = post_process_rollout_rewards(args, [sample], stage="sample")[0]
    return {**sample.metadata, "reward": reward}


def _args(
    penalty_on: bool,
    buffer_len: int = 2048,
    factor: float = 1.0,
    response_cap: int = 16384,
    context_cap: int | None = None,
    effective_response_cap: bool = False,
):
    return SimpleNamespace(
        rollout_max_response_len=response_cap,
        rollout_max_context_len=context_cap if context_cap is not None else response_cap,
        overlong_penalty=penalty_on,
        overlong_buffer_len=buffer_len,
        overlong_penalty_factor=factor,
        overlong_use_effective_response_cap=effective_response_cap,
    )


def _sample(reward, response_length, removed=False, prompt_length=0):
    return Sample(
        reward=reward,
        response_length=response_length,
        tokens=[0] * (prompt_length + response_length),
        remove_sample=removed,
        metadata={},
    )


def test_disabled_by_default():
    sample = _sample(1.0, 16384)
    details = _kernel_reward(_args(False), sample)
    assert details["overlong_penalty"] == 0.0
    assert details["overlong_prompt_len"] == 0
    # Disabled processors do not evaluate length diagnostics.
    assert details["overlong_effective_response_cap"] == 0


def test_no_penalty_below_threshold():
    sample = _sample(1.0, 16384 - 2048)
    details = _kernel_reward(_args(True), sample)
    assert details["overlong_penalty"] == 0.0
    assert details["overlong_prompt_len"] == 0
    assert details["overlong_effective_response_cap"] == 16384


def test_linear_ramp_and_cap():
    half = _sample(1.0, 16384 - 1024)
    full = _sample(1.0, 16384)
    over = _sample(0.0, 16384)
    assert _kernel_reward(_args(True), half)["overlong_penalty"] == pytest.approx(0.5)
    assert _kernel_reward(_args(True), full)["overlong_penalty"] == pytest.approx(1.0)
    assert _kernel_reward(_args(True), over)["overlong_penalty"] == pytest.approx(1.0)


def test_removed_samples_untouched():
    sample = _sample(1.0, 16384, removed=True)
    assert _kernel_reward(_args(True), sample)["overlong_penalty"] == 0.0


def test_custom_factor_and_buffer():
    sample = _sample(2.0, 16384 - 500)
    details = _kernel_reward(_args(True, buffer_len=1000, factor=0.5), sample)
    assert details["overlong_penalty"] == pytest.approx(0.25)


def test_penalty_is_independent_of_task_reward():
    fail_short = _sample(0.0, 5000)
    fail_long = _sample(0.0, 16384)
    success_long = _sample(1.0, 16384)

    assert _kernel_reward(_args(True), fail_short)["overlong_penalty"] == 0.0
    assert _kernel_reward(_args(True), fail_long)["overlong_penalty"] == pytest.approx(1.0)
    assert _kernel_reward(_args(True), success_long)["overlong_penalty"] == pytest.approx(1.0)


def test_reviewed_24k_policy_uses_effective_response_cap_after_prompt():
    # With a 4K prompt inside a 24K context, the response can use at most 20K.
    partial = _sample(0.25, 20480, prompt_length=4096)
    correct = _sample(0.5, 20480, prompt_length=4096)

    partial_details = _kernel_reward(
        _args(
            True,
            buffer_len=4096,
            factor=0.2,
            response_cap=24576,
            context_cap=24576,
            effective_response_cap=True,
        ),
        partial,
    )
    correct_details = _kernel_reward(
        _args(
            True,
            buffer_len=4096,
            factor=0.2,
            response_cap=24576,
            context_cap=24576,
            effective_response_cap=True,
        ),
        correct,
    )

    for details in (partial_details, correct_details):
        assert details["overlong_penalty"] == pytest.approx(0.2)
        assert details["overlong_prompt_len"] == 4096
        assert details["overlong_effective_response_cap"] == 20480


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
