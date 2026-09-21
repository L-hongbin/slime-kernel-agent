"""CPU contracts for reviewed output-mismatch partial reward policy."""

from __future__ import annotations

import asyncio
import importlib.util
import math
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
from examples.kernel_agent.kernel_reward import (
    _calculate_performance_score,
    _compute_speedup_log_standard_error,
    calculate_kernel_reward,
    post_process_rollout_rewards,
)
from examples.kernel_agent.utils import normalize_env_feedback, postprocess_turn_samples

from slime.utils.types import Sample


def _reward_config(output_mismatch_score: float = 0.25) -> dict:
    return {
        **CUDA_AGENT_CONFIGS["reward"],
        "kernel_failed_score": {
            **CUDA_AGENT_CONFIGS["reward"]["kernel_failed_score"],
            "output_mismatch": output_mismatch_score,
        },
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
    timing_metadata: dict | None = None,
) -> dict:
    metadata = {
        "correctness_candidate_forward_completed": candidate_forward_completed,
        "correctness_output_mismatch": output_mismatch,
        **(timing_metadata or {}),
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
    config = {**_reward_config(), "apply_kernel_failed_score": True}
    details = calculate_kernel_reward(
        _completed_env(output_mismatch=True, candidate_forward_completed=True),
        config,
    )

    assert details["kernel_failed_score"] == pytest.approx(0.25)
    assert details["reward"] == pytest.approx(0.125)
    assert details["kernel_failed_score_tag"] == "output_mismatch"


@pytest.mark.parametrize(
    ("env_state", "reason"),
    [
        (_completed_env(compiled=False), "compilation"),
        (_completed_env(compiled=True), "correctness"),
        (
            _completed_env(compiled=True, candidate_forward_completed=True),
            "correctness",
        ),
        (
            _completed_env(compiled=True, runtime_error="CUDA illegal memory access"),
            "runtime",
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
    config = {**_reward_config(), "apply_failed_group_reward": True}
    details = calculate_kernel_reward(env_state, config)

    assert details["reward"] == 0.0
    assert details["kernel_failed_score_tag"] == reason


def test_correct_reward_remains_strictly_above_partial_reward():
    details = calculate_kernel_reward(
        _completed_env(compiled=True, correctness=True, candidate_forward_completed=True),
        _reward_config(),
    )

    assert details["reward"] == pytest.approx(0.5)
    assert details["kernel_failed_score"] is None
    assert details["kernel_failed_score_tag"] is None


def test_failed_score_tag_is_disabled_when_no_fine_grained_mode_is_enabled():
    details = calculate_kernel_reward(_completed_env(compiled=False), _reward_config())

    assert details["reward"] == 0.0
    assert details["kernel_failed_score"] == 0.0
    assert details["kernel_failed_score_tag"] == "disabled"


@pytest.mark.parametrize(
    ("env_state", "expected_score", "expected_tag"),
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
            "precheck",
        ),
        (_completed_env(compiled=False), -0.75, "compilation"),
        (_completed_env(runtime_error="CUDA illegal memory access"), -0.5, "runtime"),
        (
            {
                **_completed_env(compiled=True),
                "status": "timeout",
                "error": "KERNEL_EVAL_TIMEOUT",
            },
            -0.5,
            "runtime",
        ),
        (_completed_env(compiled=True, correctness=False), -0.25, "correctness"),
        (_completed_env(compiled=True, decoy_kernel=True), -1.0, "decoy"),
    ],
)
def test_fail_score_tracks_evaluation_progress(env_state, expected_score, expected_tag):
    config = {**_reward_config(output_mismatch_score=0.0), "apply_failed_group_reward": True}
    details = calculate_kernel_reward(env_state, config)

    assert details["kernel_failed_score"] == pytest.approx(expected_score)
    assert details["kernel_failed_score_tag"] == expected_tag
    assert details["reward_component"]["failed"] == 0.0


def test_failed_group_reward_config_separates_flag_and_failure_scores():
    reward_config = CUDA_AGENT_CONFIGS["reward"]

    assert reward_config["failed_score"] == 0.0
    assert isinstance(reward_config["apply_kernel_failed_score"], bool)
    assert isinstance(reward_config["apply_failed_group_reward"], bool)
    assert reward_config["kernel_failed_score"] == {
        "precheck": -1.0,
        "compilation": -0.75,
        "runtime": -0.5,
        "correctness": -0.25,
        "output_mismatch": 0.0,
        "decoy": -1.0,
        "other": -1.0,
    }
    legacy_keys = {
        "failure_stage_penalty_scores",
        "precheck_fail_penalty",
        "compilation_fail_penalty",
        "apply_precheck_fail_penalty",
        "apply_compilation_fail_penalty",
        "apply_penalty_score",
        "penalty_score",
        "kernel_penalty",
        "apply_kernel_penalty",
        "output_mismatch_partial_reward",
    }
    assert legacy_keys.isdisjoint(reward_config)


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [(None, "legacy"), ("improvement", "improvement"), ("lcb_improvement", "lcb_improvement")],
)
def test_speedup_score_mode_reads_environment(monkeypatch, env_value, expected):
    env_name = "CUDA_AGENT_SPEEDUP_SCORE_MODE"
    monkeypatch.delenv("CUDA_AGENT_APPLY_KERNEL_FAILED_SCORE", raising=False)
    monkeypatch.delenv("CUDA_AGENT_APPLY_FAILED_GROUP_REWARD", raising=False)
    if env_value is None:
        monkeypatch.delenv(env_name, raising=False)
    else:
        monkeypatch.setenv(env_name, env_value)

    config_path = REPO_ROOT / "examples" / "kernel_agent" / "config.py"
    spec = importlib.util.spec_from_file_location("_kernel_agent_speedup_score_mode_env_test", config_path)
    assert spec is not None and spec.loader is not None
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    assert config_module.CUDA_AGENT_CONFIGS["reward"]["speedup_score_mode"] == expected


def test_speedup_score_mode_rejects_unknown_environment_value(monkeypatch):
    monkeypatch.setenv("CUDA_AGENT_SPEEDUP_SCORE_MODE", "unknown")
    monkeypatch.delenv("CUDA_AGENT_APPLY_KERNEL_FAILED_SCORE", raising=False)
    monkeypatch.delenv("CUDA_AGENT_APPLY_FAILED_GROUP_REWARD", raising=False)

    config_path = REPO_ROOT / "examples" / "kernel_agent" / "config.py"
    spec = importlib.util.spec_from_file_location("_kernel_agent_invalid_speedup_score_mode_env_test", config_path)
    assert spec is not None and spec.loader is not None
    config_module = importlib.util.module_from_spec(spec)

    with pytest.raises(ValueError, match="CUDA_AGENT_SPEEDUP_SCORE_MODE"):
        spec.loader.exec_module(config_module)


@pytest.mark.parametrize(("env_value", "expected"), [(None, False), ("0", False), ("1", True)])
def test_failed_group_reward_flag_reads_environment(monkeypatch, env_value, expected):
    env_name = "CUDA_AGENT_APPLY_FAILED_GROUP_REWARD"
    monkeypatch.delenv("CUDA_AGENT_APPLY_KERNEL_FAILED_SCORE", raising=False)
    if env_value is None:
        monkeypatch.delenv(env_name, raising=False)
    else:
        monkeypatch.setenv(env_name, env_value)

    config_path = REPO_ROOT / "examples" / "kernel_agent" / "config.py"
    spec = importlib.util.spec_from_file_location("_kernel_agent_config_env_test", config_path)
    assert spec is not None and spec.loader is not None
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    assert config_module.CUDA_AGENT_CONFIGS["reward"]["apply_failed_group_reward"] is expected


@pytest.mark.parametrize(("env_value", "expected"), [(None, False), ("0", False), ("1", True)])
def test_apply_kernel_failed_score_flag_reads_environment(monkeypatch, env_value, expected):
    env_name = "CUDA_AGENT_APPLY_KERNEL_FAILED_SCORE"
    monkeypatch.delenv("CUDA_AGENT_APPLY_FAILED_GROUP_REWARD", raising=False)
    if env_value is None:
        monkeypatch.delenv(env_name, raising=False)
    else:
        monkeypatch.setenv(env_name, env_value)

    config_path = REPO_ROOT / "examples" / "kernel_agent" / "config.py"
    spec = importlib.util.spec_from_file_location("_kernel_agent_apply_kernel_failed_score_env_test", config_path)
    assert spec is not None and spec.loader is not None
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    assert config_module.CUDA_AGENT_CONFIGS["reward"]["apply_kernel_failed_score"] is expected


def test_failed_score_reward_modes_are_mutually_exclusive(monkeypatch):
    monkeypatch.setenv("CUDA_AGENT_APPLY_KERNEL_FAILED_SCORE", "1")
    monkeypatch.setenv("CUDA_AGENT_APPLY_FAILED_GROUP_REWARD", "1")

    config_path = REPO_ROOT / "examples" / "kernel_agent" / "config.py"
    spec = importlib.util.spec_from_file_location("_kernel_agent_penalty_conflict_test", config_path)
    assert spec is not None and spec.loader is not None
    config_module = importlib.util.module_from_spec(spec)

    with pytest.raises(ValueError, match="cannot both be enabled"):
        spec.loader.exec_module(config_module)


@pytest.mark.parametrize(
    ("env_state", "expected_reward"),
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
        (_completed_env(compiled=True, correctness=False), -0.25),
        (_completed_env(compiled=True, decoy_kernel=True), -1.0),
    ],
)
def test_apply_kernel_failed_score_directly_rewards_each_failed_sample(env_state, expected_reward):
    config = {
        **_reward_config(),
        "apply_kernel_failed_score": True,
        "apply_failed_group_reward": False,
    }

    details = calculate_kernel_reward(env_state, config)

    assert details["kernel_failed_score"] == pytest.approx(expected_reward)
    assert details["reward"] == pytest.approx(expected_reward * 0.5)
    assert details["reward_component"]["failed"] == pytest.approx(expected_reward * 0.5)


def test_apply_kernel_failed_score_keeps_successful_reward():
    config = {
        **_reward_config(),
        "apply_kernel_failed_score": True,
        "apply_failed_group_reward": False,
    }

    details = calculate_kernel_reward(_completed_env(correctness=True), config)

    assert details["reward"] == pytest.approx(0.5)
    assert details["reward_component"]["failed"] is None


def test_kernel_and_overlong_penalties_are_recorded_separately():
    config = {
        **_reward_config(),
        "apply_kernel_failed_score": True,
        "apply_failed_group_reward": False,
    }
    args = SimpleNamespace(
        overlong_penalty="dapo",
        overlong_buffer_len=100,
        overlong_penalty_factor=1.0,
        rollout_max_response_len=100,
        rollout_max_context_len=100,
        overlong_use_effective_response_cap=False,
    )
    sample = Sample(response_length=100, tokens=[0] * 100)

    details = calculate_kernel_reward(_completed_env(compiled=False), config)
    assert details["reward"] == pytest.approx(-0.375)
    sample.reward = details.pop("reward")
    sample.metadata = details
    reward = post_process_rollout_rewards(args, [sample], stage="sample")[0]

    assert details["task_reward"] == pytest.approx(-0.375)
    assert reward == pytest.approx(-1.375)
    assert sample.reward == pytest.approx(reward)
    assert details["reward_component"]["failed"] == pytest.approx(-0.375)
    assert details["reward_component"]["length"] == pytest.approx(-1.0)


def test_calculate_kernel_reward_rejects_conflicting_penalty_modes():
    config = {
        **_reward_config(),
        "apply_kernel_failed_score": True,
        "apply_failed_group_reward": True,
    }

    with pytest.raises(ValueError, match="cannot both be enabled"):
        calculate_kernel_reward(_completed_env(compiled=False), config)


def test_failed_score_is_the_base_reward_for_failed_sample():
    config = {**_reward_config(output_mismatch_score=0.0), "failed_score": -2.0}

    details = calculate_kernel_reward(_completed_env(compiled=True, correctness=False), config)

    assert details["reward"] == -1.0
    assert details["kernel_failed_score"] == -2.0
    assert details["kernel_failed_score_tag"] == "disabled"
    assert details["reward_component"]["correctness"] == 0.0
    assert details["reward_component"]["failed"] == -1.0


@pytest.mark.parametrize(
    "env_state",
    [
        _completed_env(correctness=True, speedup=2.0),
        _completed_env(output_mismatch=True, candidate_forward_completed=True),
        _completed_env(compiled=False),
        {"status": "failed", "compiled": None, "correctness": None, "metadata": {}},
    ],
)
def test_reward_component_sums_to_reward(env_state):
    details = calculate_kernel_reward(env_state, _reward_config())

    component_sum = sum(value for value in details["reward_component"].values() if value is not None)
    assert component_sum == pytest.approx(details["reward"])
    assert details["task_reward"] == pytest.approx(details["reward"])
    assert details["raw_task_reward"] == pytest.approx(details["reward"])
    assert "raw_task_reward" not in details["reward_component"]
    assert details["length_score"] == 0.0
    assert details["overlong_prompt_len"] == 0
    assert details["overlong_effective_response_cap"] == 0
    assert "score" not in details


def test_output_mismatch_requires_compilation_contract():
    config = {**_reward_config(), "apply_failed_group_reward": True}
    with pytest.raises(AssertionError, match="correctness_output_mismatch=true requires compiled=true"):
        calculate_kernel_reward(
            _completed_env(
                compiled=False,
                output_mismatch=True,
                candidate_forward_completed=True,
            ),
            config,
        )


def test_output_mismatch_score_must_stay_below_correctness_score():
    config = {
        **_reward_config(output_mismatch_score=1.0),
        "apply_kernel_failed_score": True,
    }
    with pytest.raises(ValueError, match=r"lower than correctness_score \(1.0\)"):
        calculate_kernel_reward(
            _completed_env(output_mismatch=True, candidate_forward_completed=True),
            config,
        )


def test_failed_sample_does_not_receive_performance_reward_from_malformed_env_state():
    incorrect = _completed_env(compiled=True, candidate_forward_completed=True, speedup=2.0)

    gated_details = calculate_kernel_reward(incorrect, _reward_config())
    ungated_details = calculate_kernel_reward(
        incorrect,
        {
            **_reward_config(output_mismatch_score=0.0),
            "performance_reward_requires_correctness": False,
        },
    )

    assert CUDA_AGENT_CONFIGS["reward"]["performance_reward_requires_correctness"] is True
    assert gated_details["kernel_score"]["performance"] == 0.0
    assert gated_details["reward_component"]["performance"] == 0.0
    assert gated_details["reward_component"]["failed"] == 0.0
    assert gated_details["reward"] == 0.0
    assert ungated_details["reward"] == 0.0


@pytest.mark.parametrize(
    "mode,speedup,expected",
    [
        ("legacy", 2.0, 2.0),
        ("legacy", 3.0, 3.0),
        ("legacy", 5.0, 5.0),
        ("legacy", 6.0, 5.0),
        ("improvement", 2.0, 0.25),
        ("improvement", 3.0, 0.5),
        ("improvement", 5.0, 1.0),
        ("improvement", 6.0, 1.0),
    ],
)
def test_default_speedup_reward_upper_bound(mode, speedup, expected):
    config = _reward_config()
    assert config["speedup_reward_upper_bound"] == 5.0
    assert _calculate_performance_score(speedup, mode, None, config) == pytest.approx(expected)


def test_improvement_performance_score_only_rewards_gain_over_reference():
    config = {
        **_reward_config(),
        "coverage_reward_enable": False,
        "speedup_score_mode": "improvement",
        "speedup_reward_upper_bound": 2.0,
    }

    baseline = calculate_kernel_reward(_completed_env(correctness=True, speedup=1.0), config)
    improved = calculate_kernel_reward(_completed_env(correctness=True, speedup=1.5), config)

    assert baseline["speedup_log_standard_error"] is None
    assert baseline["reward"] == pytest.approx(0.5)
    assert improved["kernel_score"] == {
        "correctness": 1.0,
        "performance": pytest.approx(0.5),
        "coverage": 0.0,
    }
    assert improved["reward_component"]["performance"] == pytest.approx(0.25)
    assert improved["reward"] == pytest.approx(0.75)


def test_lcb_improvement_uses_timing_uncertainty_before_reward_mapping():
    config = {
        **_reward_config(),
        "coverage_reward_enable": False,
        "speedup_score_mode": "lcb_improvement",
        "speedup_reward_upper_bound": 2.0,
        "speedup_uncertainty_z_score": 2.0,
        "speedup_uncertainty_log_std_floor": 0.01,
    }
    timing_metadata = {
        "kg_kernel_perf_mean_ms": 10.0,
        "kg_kernel_perf_std_ms": 2.0,
        "kg_kernel_perf_num_trials": 100,
        "kg_reference_perf_mean_ms": 20.0,
        "kg_reference_perf_std_ms": 2.0,
        "kg_reference_perf_num_trials": 25,
    }

    expected_log_se = (0.2**2 / 100 + 0.1**2 / 25 + 0.01**2) ** 0.5
    actual_log_se = _compute_speedup_log_standard_error(timing_metadata, config)
    performance_score = _calculate_performance_score(2.0, "lcb_improvement", timing_metadata, config)
    expected_lcb = 2.0 * math.exp(-2.0 * expected_log_se)
    assert actual_log_se == pytest.approx(expected_log_se)
    assert performance_score == pytest.approx(expected_lcb - 1.0)


def test_lcb_improvement_rejects_missing_timing_metadata():
    config = {**_reward_config(), "speedup_score_mode": "lcb_improvement"}

    with pytest.raises(ValueError, match="timing metadata"):
        _compute_speedup_log_standard_error(None, config)


def test_lcb_improvement_supports_cached_reference_with_noise_floor():
    config = {
        **_reward_config(),
        "coverage_reward_enable": False,
        "speedup_score_mode": "lcb_improvement",
        "speedup_uncertainty_z_score": 1.0,
        "speedup_uncertainty_log_std_floor": 0.02,
    }
    timing_metadata = {
        "cached": True,
        "kg_kernel_perf_mean_ms": 10.0,
        "kg_kernel_perf_std_ms": 1.0,
        "kg_kernel_perf_num_trials": 100,
    }

    expected_log_se = (0.1**2 / 100 + 0.02**2) ** 0.5
    actual_log_se = _compute_speedup_log_standard_error(timing_metadata, config)
    performance_score = _calculate_performance_score(1.5, "lcb_improvement", timing_metadata, config)
    expected_lcb = 1.5 * math.exp(-expected_log_se)
    assert actual_log_se == pytest.approx(expected_log_se)
    assert performance_score == pytest.approx((expected_lcb - 1.0) / (config["speedup_reward_upper_bound"] - 1.0))


def test_normalization_uses_runtime_error_code_for_kernel_failed_score_tag():
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
    config = {**_reward_config(), "apply_failed_group_reward": True}
    details = calculate_kernel_reward(env_state, config)
    assert details["reward"] == 0.0
    assert details["kernel_failed_score_tag"] == "runtime"


def test_reward_func_records_fail_score_metadata(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"]["kernel_failed_score"], "output_mismatch", 0.25)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "apply_kernel_failed_score", True)
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

    assert reward == pytest.approx(0.125)
    assert sample.metadata["kernel_failed_score_tag"] == "output_mismatch"
    assert sample.metadata["kernel_score"] == {
        "correctness": 0.0,
        "performance": 0.0,
        "coverage": 0.0,
    }
    assert sample.metadata["reward_component"] == {
        "correctness": 0.0,
        "performance": 0.0,
        "coverage": 0.0,
        "failed": 0.125,
        "length": 0.0,
    }
    assert sample.metadata["env_extra_info"]["kernel_failed_score_tag"] == "output_mismatch"
    assert "failure_stage" not in sample.metadata
    assert sample.metadata["kernel_failed_score"] == pytest.approx(0.25)
    assert "kernel_failed_score" not in sample.metadata["env_extra_info"]


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


def test_kernel_failed_score_tag_metrics_count_each_failure_category():
    from slime.observability.rollout_metrics import _compute_kernel_agent_metrics

    sample = Sample(
        prompt="prompt",
        reward=0.25,
        metadata={
            "kernel_failed_score_tag": "output_mismatch",
            "length_score": -0.1,
            "env_extra_info": {
                "correctness": False,
                "compilation": True,
                "speedup": 0.0,
                "speedup_log_standard_error": 0.0125,
                "decoy_kernel": False,
                "correctness_candidate_forward_completed": True,
                "correctness_output_mismatch": True,
                "kernel_failed_score_tag": "output_mismatch",
                "incorrect_backend_probe_attempted": True,
                "incorrect_backend_probe_valid": True,
                "incorrect_backend_probe_custom_kernel_observed": True,
                "incorrect_backend_probe_decoy_detected": False,
                "precheck": "passed",
            },
        },
    )
    failed_samples = [
        Sample(
            prompt="prompt",
            reward=0.0,
            metadata={"kernel_failed_score_tag": reason},
        )
        for reason in ("decoy", "runtime", "compilation")
    ]

    metrics = _compute_kernel_agent_metrics([sample, *failed_samples])

    assert metrics["kernel/failed_score_tag/output_mismatch_count"] == 1
    assert metrics["kernel/failed_score_tag/decoy_count"] == 1
    assert metrics["kernel/failed_score_tag/runtime_count"] == 1
    assert metrics["kernel/failed_score_tag/compilation_count"] == 1
    assert metrics["kernel/length_score/mean"] == pytest.approx(-0.1)
    assert "env_extra_info/kernel_failed_score_tag/mean" not in metrics
    assert metrics["env_extra_info/speedup_log_standard_error/mean"] == pytest.approx(0.0125)
    assert not any("reward_component" in key for key in metrics)
    assert metrics["kernel/incorrect_backend_probe/attempted_ratio"] == pytest.approx(0.25)
    assert metrics["kernel/incorrect_backend_probe/valid_ratio_of_attempted"] == 1.0
    assert metrics["kernel/incorrect_backend_probe/custom_kernel_observed_ratio_of_valid"] == 1.0


def test_qwen_reward_length_filter_chain_uses_task_reward_and_keeps_correct_coverage(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"]["kernel_failed_score"], "output_mismatch", 0.25)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "apply_kernel_failed_score", True)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "performance_reward_requires_correctness", True)
    args = SimpleNamespace(
        overlong_penalty="dapo",
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
        use_coverage_rs=False,
        finalize_mode="none",
        advantage_estimator="grpo",
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
    rewards_before_postprocess = [sample.reward for sample in samples]
    postprocess_turn_samples(args, samples, finish_reason="env_done")
    filter_result = filter_cuda_kernel_group(args, samples)

    assert [sample.reward for sample in samples] == rewards_before_postprocess
    assert [sample.metadata["task_reward"] for sample in samples] == pytest.approx([-0.375, 0.125, 0.625])
    assert [sample.metadata["raw_task_reward"] for sample in samples] == pytest.approx([-0.375, 0.125, 0.625])
    assert samples[1].reward == pytest.approx(-0.075)
    assert samples[1].metadata["reward_component"]["failed"] == pytest.approx(0.125)
    assert samples[1].metadata["reward_component"]["length"] == pytest.approx(-0.2)
    assert sum(
        value for value in samples[1].metadata["reward_component"].values() if value is not None
    ) == pytest.approx(samples[1].reward)
    assert samples[1].metadata["overlong_effective_response_cap"] == 20480
    assert samples[2].reward == pytest.approx(0.625)
    assert samples[2].metadata["kernel_score"]["coverage"] == pytest.approx(0.25)
    assert samples[2].metadata["reward_component"]["coverage"] == pytest.approx(0.125)
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
    uniform_result = filter_cuda_kernel_group(args, uniform_partial)
    assert uniform_result.keep is False
    assert uniform_result.reason == "reward_std_lt_0.001"


@pytest.mark.parametrize("failed_score", [0.0, -2.0])
def test_low_variance_filter_uses_scores_for_all_failed_group(monkeypatch, failed_score):
    from examples.kernel_agent.kernel_reward import post_process_rollout_rewards

    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "apply_failed_group_reward", True)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "failed_score", failed_score)
    args = SimpleNamespace(
        n_samples_per_prompt=3,
        target_group_size=3,
        min_group_size=2,
        reward_std_threshold=0.001,
        reward_key=None,
    )
    samples = [
        Sample(
            index=index,
            group_index=0,
            reward=failed_score * CUDA_AGENT_CONFIGS["reward"]["init_correct_weight"],
            metadata={"kernel_failed_score": kernel_failed_score},
        )
        for index, kernel_failed_score in enumerate((-1.0, -0.75, -0.25))
    ]

    post_process_rollout_rewards(args, samples)
    result = filter_cuda_kernel_group(args, samples)

    assert result.keep is True
    assert [sample.reward for sample in samples] == [-0.5, -0.375, -0.125]


def test_low_variance_filter_ignores_scores_when_failed_group_reward_is_disabled(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "apply_failed_group_reward", False)
    args = SimpleNamespace(
        n_samples_per_prompt=3,
        target_group_size=3,
        min_group_size=2,
        reward_std_threshold=0.001,
        reward_key=None,
    )
    samples = [
        Sample(index=index, group_index=0, reward=0.0, metadata={"kernel_failed_score": kernel_failed_score})
        for index, kernel_failed_score in enumerate((-1.0, -0.75, -0.25))
    ]

    result = filter_cuda_kernel_group(args, samples)

    assert result.keep is False
    assert result.reason == "reward_std_lt_0.001"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
