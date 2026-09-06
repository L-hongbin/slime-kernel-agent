"""CPU contracts for reviewed output-mismatch partial reward policy."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

NUM_GPUS = 0

REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_path = str(REPO_ROOT)
if repo_root_path in sys.path:
    sys.path.remove(repo_root_path)
sys.path.insert(0, repo_root_path)

from examples.kernel_agent import generate_with_cuda_agent
from examples.kernel_agent.config import CUDA_AGENT_CONFIGS
from examples.kernel_agent.kernel_filter import filter_cuda_kernel_group
from examples.kernel_agent.kernel_reward import calculate_reward_speedup
from examples.kernel_agent.utils import _apply_overlong_penalty, normalize_env_feedback
from slime.utils.types import Sample


def _reward_config(partial_reward: float = 0.25) -> dict:
    return {
        **CUDA_AGENT_CONFIGS["reward"],
        "output_mismatch_partial_reward": partial_reward,
        "performance_reward_requires_correctness": True,
    }


def _completed_env(
    *,
    compiled: bool = True,
    correctness: bool = False,
    decoy_kernel: bool = False,
    output_mismatch: bool = False,
    candidate_forward_completed: bool = False,
    runtime_error: str | None = None,
    speedup: float = 0.0,
) -> dict:
    metadata = {
        "correctness_candidate_forward_completed": candidate_forward_completed,
        "correctness_output_mismatch": output_mismatch,
    }
    env_state = {
        "status": "completed",
        "compiled": compiled,
        "correctness": correctness,
        "decoy_kernel": decoy_kernel,
        "speedup": speedup,
        "metadata": metadata,
    }
    if runtime_error is not None:
        env_state["error"] = "RUNTIME_ERROR"
        env_state["error_message"] = runtime_error
    return env_state


def test_output_mismatch_receives_reviewed_partial_reward():
    details = calculate_reward_speedup(
        _completed_env(output_mismatch=True, candidate_forward_completed=True),
        _reward_config(),
    )

    assert details["reward"] == pytest.approx(0.25)
    assert details["partial_credit_output_mismatch"] is True
    assert details["partial_credit_output_mismatch_reason"] == "applied"


@pytest.mark.parametrize(
    ("env_state", "reason"),
    [
        (_completed_env(compiled=False), "not_compiled"),
        (_completed_env(compiled=True), "candidate_forward_not_completed"),
        (
            _completed_env(compiled=True, candidate_forward_completed=True),
            "no_output_mismatch",
        ),
        (
            _completed_env(compiled=True, runtime_error="CUDA illegal memory access"),
            "runtime_error",
        ),
        (
            _completed_env(
                compiled=True,
                decoy_kernel=True,
                output_mismatch=True,
                candidate_forward_completed=True,
            ),
            "decoy",
        ),
    ],
)
def test_partial_reward_rejects_non_mismatch_runtime_and_decoy_failures(env_state, reason):
    details = calculate_reward_speedup(env_state, _reward_config())

    assert details["reward"] == 0.0
    assert details["partial_credit_output_mismatch"] is False
    assert details["partial_credit_output_mismatch_reason"] == reason


def test_correct_reward_remains_strictly_above_partial_reward():
    details = calculate_reward_speedup(
        _completed_env(compiled=True, correctness=True, candidate_forward_completed=True),
        _reward_config(),
    )

    assert details["reward"] == pytest.approx(0.5)
    assert details["partial_credit_output_mismatch"] is False
    assert details["partial_credit_output_mismatch_reason"] == "already_correct"


@pytest.mark.parametrize(
    ("env_state", "penalty_score"),
    [
        (
            {
                "status": "failed",
                "compiled": None,
                "correctness": None,
                "error": "PRECHECK_ERROR",
                "metadata": {},
            },
            -1.0,
        ),
        (_completed_env(compiled=False), -0.75),
        (_completed_env(runtime_error="CUDA illegal memory access"), -0.5),
        (
            {
                **_completed_env(compiled=True),
                "status": "timeout",
                "error": "KERNEL_EVAL_TIMEOUT",
            },
            -0.5,
        ),
        (_completed_env(compiled=True, correctness=False), -0.25),
        (_completed_env(compiled=True, decoy_kernel=True), -1.0),
        (_completed_env(compiled=True, correctness=True), 0.0),
    ],
)
def test_penalty_score_tracks_evaluation_progress(env_state, penalty_score):
    details = calculate_reward_speedup(env_state, _reward_config(partial_reward=0.0))

    assert details["penalty_score"] == pytest.approx(penalty_score)


def test_failed_group_reward_config_separates_flag_and_penalty_scores():
    reward_config = CUDA_AGENT_CONFIGS["reward"]

    assert reward_config["failed_score"] == 0.0
    assert reward_config["apply_failed_group_reward"] is True
    assert reward_config["penalty_score"] == {
        "precheck": -1.0,
        "compilation": -0.75,
        "runtime": -0.5,
        "correctness": -0.25,
        "decoy": -1.0,
        "other": -1.0,
    }
    legacy_keys = {
        "failure_stage_penalty_scores",
        "precheck_fail_penalty",
        "compilation_fail_penalty",
        "apply_precheck_fail_penalty",
        "apply_compilation_fail_penalty",
    }
    assert legacy_keys.isdisjoint(reward_config)


def test_failed_score_is_the_base_reward_for_failed_sample():
    config = {**_reward_config(partial_reward=0.0), "failed_score": -2.0}

    details = calculate_reward_speedup(_completed_env(compiled=True, correctness=False), config)

    assert details["reward"] == -2.0
    assert details["penalty_score"] == -0.25


def test_output_mismatch_requires_compilation_contract():
    with pytest.raises(AssertionError, match="correctness_output_mismatch=true requires compiled=true"):
        calculate_reward_speedup(
            _completed_env(
                compiled=False,
                output_mismatch=True,
                candidate_forward_completed=True,
            ),
            _reward_config(),
        )


def test_partial_reward_must_stay_below_correctness_floor():
    with pytest.raises(ValueError, match="lower than init_correct_weight"):
        calculate_reward_speedup(
            _completed_env(output_mismatch=True, candidate_forward_completed=True),
            _reward_config(partial_reward=0.5),
        )


def test_qwen_policy_gates_performance_reward_on_correctness_without_changing_global_default():
    incorrect = _completed_env(compiled=True, candidate_forward_completed=True, speedup=2.0)

    qwen_details = calculate_reward_speedup(incorrect, _reward_config())
    legacy_details = calculate_reward_speedup(
        incorrect,
        {
            **_reward_config(partial_reward=0.0),
            "performance_reward_requires_correctness": False,
        },
    )

    assert CUDA_AGENT_CONFIGS["reward"]["performance_reward_requires_correctness"] is False
    assert qwen_details["reward"] == 0.0
    assert legacy_details["reward"] == pytest.approx(1.0)


def test_normalization_uses_runtime_error_code_for_partial_reward():
    env_state, _ = normalize_env_feedback(
        {
            **_completed_env(output_mismatch=True, candidate_forward_completed=True),
            "error_code": "RUNTIME_ERROR",
            "error_message": "CUDA illegal memory access",
            "metadata": {
                "correctness_candidate_forward_completed": True,
                "correctness_output_mismatch": True,
                "runtime_error": "CUDA illegal memory access",
            },
        }
    )

    assert "runtime_error" not in env_state["metadata"]
    assert "correctness_runtime_error" not in env_state["metadata"]
    assert env_state["error"] == "RUNTIME_ERROR"
    details = calculate_reward_speedup(env_state, _reward_config())
    assert details["reward"] == 0.0
    assert details["partial_credit_output_mismatch_reason"] == "runtime_error"


def test_reward_func_records_partial_audit_metadata(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "output_mismatch_partial_reward", 0.25)
    env_state = _completed_env(output_mismatch=True, candidate_forward_completed=True)
    env_extra_info = {
        "correctness": False,
        "compilation": True,
        "speedup": 0.0,
        "decoy_kernel": False,
        "precheck": "passed",
    }
    sample = Sample(
        prompt="prompt",
        reward=None,
        metadata={
            "env_result": {"env_state": env_state, "env_extra_info": env_extra_info},
            "env_extra_info": env_extra_info,
        },
    )

    reward = asyncio.run(generate_with_cuda_agent.reward_func(SimpleNamespace(), sample))

    assert reward == pytest.approx(0.25)
    assert sample.metadata["partial_credit_output_mismatch"] is True
    assert sample.metadata["partial_credit_output_mismatch_reason"] == "applied"
    assert "reward_components" not in sample.metadata
    assert sample.metadata["env_extra_info"]["partial_credit_output_mismatch"] is True
    assert "failure_stage" not in sample.metadata
    assert sample.metadata["penalty_score"] == pytest.approx(-0.25)
    assert "penalty_score" not in sample.metadata["env_extra_info"]


def test_normalization_preserves_mismatch_and_summarizes_backend_probe():
    env_state, env_extra_info = normalize_env_feedback(
        {
            **_completed_env(output_mismatch=True, candidate_forward_completed=True),
            "metadata": {
                "correctness_candidate_forward_completed": True,
                "correctness_output_mismatch": True,
                "incorrect_backend_usage_probe": {
                    "backend": "tvm_ffi",
                    "attempted": True,
                    "num_forwards": 1,
                    "valid": True,
                    "custom_kernel_observed": True,
                    "decoy_detected": False,
                    "profiling": {"kernels": ["large raw payload"]},
                },
            },
        }
    )

    assert env_state["metadata"]["correctness_candidate_forward_completed"] is True
    assert env_extra_info["correctness_candidate_forward_completed"] is True
    assert env_extra_info["correctness_output_mismatch"] is True
    assert env_extra_info["incorrect_backend_probe_attempted"] is True
    assert env_extra_info["incorrect_backend_probe_valid"] is True
    assert env_extra_info["incorrect_backend_probe_custom_kernel_observed"] is True
    assert env_state["metadata"]["incorrect_backend_usage_probe"] == {
        "backend": "tvm_ffi",
        "attempted": True,
        "num_forwards": 1,
        "valid": True,
        "custom_kernel_observed": True,
        "decoy_detected": False,
    }


def test_partial_reward_metrics_keep_only_applied_rate_and_key_rejections():
    from slime.observability.rollout_metrics import _compute_kernel_agent_metrics

    sample = Sample(
        prompt="prompt",
        reward=0.25,
        metadata={
            "partial_credit_output_mismatch": True,
            "partial_credit_output_mismatch_reason": "applied",
            "overlong_penalty": 0.1,
            "env_extra_info": {
                "correctness": False,
                "compilation": True,
                "speedup": 0.0,
                "decoy_kernel": False,
                "correctness_candidate_forward_completed": True,
                "correctness_output_mismatch": True,
                "partial_credit_output_mismatch": True,
                "incorrect_backend_probe_attempted": True,
                "incorrect_backend_probe_valid": True,
                "incorrect_backend_probe_custom_kernel_observed": True,
                "incorrect_backend_probe_decoy_detected": False,
                "precheck": "passed",
            },
        },
    )
    rejected_samples = [
        Sample(
            prompt="prompt",
            reward=0.0,
            metadata={
                "partial_credit_output_mismatch": False,
                "partial_credit_output_mismatch_reason": reason,
            },
        )
        for reason in ("decoy", "runtime_error", "timeout")
    ]

    metrics = _compute_kernel_agent_metrics([sample, *rejected_samples])

    assert metrics["kernel/partial_credit/applied_rate"] == pytest.approx(0.25)
    assert metrics["kernel/partial_credit/rejected_decoy_count"] == 1
    assert metrics["kernel/partial_credit/rejected_runtime_error_count"] == 1
    assert metrics["kernel/partial_credit/rejected_timeout_count"] == 1
    assert metrics["kernel/overlong_penalty/mean"] == pytest.approx(0.1)
    assert "env_extra_info/partial_credit_output_mismatch/mean" not in metrics
    assert not any("reward_component" in key for key in metrics)
    assert metrics["kernel/incorrect_backend_probe/attempted_ratio"] == pytest.approx(0.25)
    assert metrics["kernel/incorrect_backend_probe/valid_ratio_of_attempted"] == 1.0
    assert metrics["kernel/incorrect_backend_probe/custom_kernel_observed_ratio_of_valid"] == 1.0


def test_qwen_reward_length_filter_chain_uses_task_reward_and_keeps_correct_coverage(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "output_mismatch_partial_reward", 0.25)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "performance_reward_requires_correctness", True)
    args = SimpleNamespace(
        overlong_penalty=True,
        overlong_use_effective_response_cap=True,
        overlong_buffer_len=4096,
        overlong_penalty_factor=0.2,
        rollout_max_response_len=24576,
        rollout_max_context_len=24576,
        n_samples_per_prompt=3,
        target_group_size=3,
        min_group_size=2,
        reward_std_threshold=0.001,
        reward_key=None,
    )

    async def make_sample(raw_env: dict, response_length: int, prompt_length: int) -> Sample:
        env_state, env_extra_info = normalize_env_feedback(raw_env)
        sample = Sample(
            group_index=0,
            prompt="prompt",
            tokens=[0] * (prompt_length + response_length),
            response_length=response_length,
            status=Sample.Status.COMPLETED,
            metadata={
                "env_result": {"env_state": env_state, "env_extra_info": env_extra_info},
                "env_extra_info": env_extra_info,
            },
        )
        sample.reward = await generate_with_cuda_agent.reward_func(args, sample)
        return sample

    async def build_group() -> list[Sample]:
        hard_failure = await make_sample(
            {
                "status": "failed",
                "compiled": False,
                "correctness": False,
                "decoy_kernel": False,
                "speedup": 0.0,
                "error": "compile failed",
                "metadata": {},
            },
            response_length=100,
            prompt_length=4096,
        )
        mismatch = await make_sample(
            _completed_env(output_mismatch=True, candidate_forward_completed=True),
            response_length=20480,
            prompt_length=4096,
        )
        correct_env = _completed_env(correctness=True)
        correct_env.update(
            {
                "num_custom_kernel": 1,
                "num_total_kernels": 2,
                "custom_kernel_cuda_time_in_profiling_us": 25.0,
                "total_kernel_run_time_in_profiling_us": 100.0,
            }
        )
        correct = await make_sample(correct_env, response_length=100, prompt_length=4096)
        return [hard_failure, mismatch, correct]

    samples = asyncio.run(build_group())
    _apply_overlong_penalty(args, samples)
    filter_result = filter_cuda_kernel_group(args, samples)

    assert [sample.metadata["task_reward"] for sample in samples] == pytest.approx([0.0, 0.25, 0.625])
    assert samples[1].reward == pytest.approx(0.05)
    assert samples[1].metadata["overlong_effective_response_cap"] == 20480
    assert samples[2].reward == pytest.approx(0.625)
    assert filter_result.keep is True

    # Different post-penalty lengths must not rescue a task-uniform 0.25 group.
    uniform_partial = [samples[1]]
    for response_length in (18000, 100):
        clone = asyncio.run(
            make_sample(
                _completed_env(output_mismatch=True, candidate_forward_completed=True),
                response_length=response_length,
                prompt_length=4096,
            )
        )
        uniform_partial.append(clone)
    _apply_overlong_penalty(args, uniform_partial[1:])
    uniform_result = filter_cuda_kernel_group(args, uniform_partial)
    assert uniform_result.keep is False
    assert uniform_result.reason == "reward_std_lt_0.001"


@pytest.mark.parametrize("failed_score", [0.0, -2.0])
def test_low_variance_filter_uses_penalties_for_all_failed_group(monkeypatch, failed_score):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "failed_score", failed_score)
    args = SimpleNamespace(
        n_samples_per_prompt=3,
        target_group_size=3,
        min_group_size=2,
        reward_std_threshold=0.001,
        reward_key=None,
    )
    samples = [
        Sample(index=index, group_index=0, reward=failed_score, metadata={"penalty_score": penalty_score})
        for index, penalty_score in enumerate((-1.0, -0.75, -0.25))
    ]

    result = filter_cuda_kernel_group(args, samples)

    assert result.keep is True


def test_low_variance_filter_ignores_penalties_when_failed_group_reward_is_disabled(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "apply_failed_group_reward", False)
    args = SimpleNamespace(
        n_samples_per_prompt=3,
        target_group_size=3,
        min_group_size=2,
        reward_std_threshold=0.001,
        reward_key=None,
    )
    samples = [
        Sample(index=index, group_index=0, reward=0.0, metadata={"penalty_score": penalty_score})
        for index, penalty_score in enumerate((-1.0, -0.75, -0.25))
    ]

    result = filter_cuda_kernel_group(args, samples)

    assert result.keep is False
    assert result.reason == "reward_std_lt_0.001"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
