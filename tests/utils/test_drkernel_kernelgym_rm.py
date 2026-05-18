import asyncio
import re
from types import SimpleNamespace

import pytest

from slime_plugins.drkernel.extract import MISSING_COMPLETE_SUBMISSION_ERROR
from slime_plugins.drkernel.kernelgym_rm import (
    KERNELGYM_CLIENT_TIMEOUT_S,
    KERNELGYM_DETECT_DECOY_KERNEL,
    KERNELGYM_ENABLE_PROFILING,
    KERNELGYM_NUM_CORRECT_TRIALS,
    KERNELGYM_NUM_PERF_TRIALS,
    KERNELGYM_NUM_WARMUP,
    KERNELGYM_PERF_TRIM_COUNT,
    KERNELGYM_REFERENCE_BACKEND,
    KERNELGYM_TASK_TIMEOUT_S,
    KERNELGYM_USE_REFERENCE_CACHE,
    KERNELGYM_VERBOSE_ERRORS,
    KERNELGYM_WORKFLOW,
    KernelGymClient,
    _get_rm_url,
    build_evaluation_request,
    evaluate_sample,
    kernelgym_result_to_reward,
)


class _Sample:
    def __init__(self):
        self.index = 7
        self.response = ""
        self.label = "import torch\nclass Model:\n    pass"
        self.metadata = {
            "raw_problem": "import torch\nclass Model:\n    pass",
            "uuid": "sample-uuid",
            "entry_point": "Model",
        }


@pytest.mark.unit
def test_build_evaluation_request_uses_sample_label_as_reference_code():
    request = build_evaluation_request(
        SimpleNamespace(),
        _Sample(),
        kernel_code="### CUDA_KERNELS\n```cpp\ncode\n```",
    )

    assert re.fullmatch(r"parallel_task_\d{6}_[0-9a-f]{8}", request["task_id"])
    assert request["reference_code"] == "import torch\nclass Model:\n    pass"
    assert request["backend"] == "cuda_agent"
    assert request["entry_point"] == "Model"
    assert request["workflow"] == KERNELGYM_WORKFLOW
    assert request["use_reference_cache"] is KERNELGYM_USE_REFERENCE_CACHE
    assert request["timeout"] == KERNELGYM_TASK_TIMEOUT_S
    assert request["verbose_errors"] is KERNELGYM_VERBOSE_ERRORS
    assert request["enable_profiling"] is KERNELGYM_ENABLE_PROFILING
    assert request["detect_decoy_kernel"] is KERNELGYM_DETECT_DECOY_KERNEL
    assert request["num_correct_trials"] == KERNELGYM_NUM_CORRECT_TRIALS
    assert request["num_perf_trials"] == KERNELGYM_NUM_PERF_TRIALS
    assert request["num_warmup"] == KERNELGYM_NUM_WARMUP
    assert request["perf_trim_count"] == KERNELGYM_PERF_TRIM_COUNT
    assert request["reference_backend"] == KERNELGYM_REFERENCE_BACKEND
    assert request["is_valid"] is False
    assert request["uuid"] == "sample-uuid"


@pytest.mark.unit
def test_build_evaluation_request_does_not_fallback_to_metadata_reference_code():
    sample = _Sample()
    sample.label = None

    with pytest.raises(KeyError, match="--label-key ground_truth"):
        build_evaluation_request(
            SimpleNamespace(),
            sample,
            kernel_code="### CUDA_KERNELS\n```cpp\ncode\n```",
        )


@pytest.mark.unit
def test_build_evaluation_request_requires_string_label():
    sample = _Sample()
    sample.label = {"code": "class Model: pass"}

    with pytest.raises(TypeError, match="reference code must be a string"):
        build_evaluation_request(
            SimpleNamespace(),
            sample,
            kernel_code="### CUDA_KERNELS\n```cpp\ncode\n```",
        )


@pytest.mark.unit
def test_build_evaluation_request_requires_uuid():
    sample = _Sample()
    del sample.metadata["uuid"]

    with pytest.raises(KeyError, match="missing uuid"):
        build_evaluation_request(
            SimpleNamespace(),
            sample,
            kernel_code="### CUDA_KERNELS\n```cpp\ncode\n```",
        )


@pytest.mark.unit
def test_build_evaluation_request_uses_kernelbench_eval_metadata_defaults():
    sample = _Sample()
    del sample.metadata["uuid"]
    del sample.metadata["entry_point"]
    sample.metadata["problem_id"] = 1

    request = build_evaluation_request(
        SimpleNamespace(),
        sample,
        kernel_code="### CUDA_KERNELS\n```cpp\ncode\n```",
    )

    assert request["entry_point"] == "Model"
    assert request["uuid"] == "1"


@pytest.mark.unit
def test_build_evaluation_request_ignores_metadata_task_id_override():
    sample = _Sample()
    sample.metadata["task_id"] = "custom-task-id"

    request = build_evaluation_request(
        SimpleNamespace(),
        sample,
        kernel_code="### CUDA_KERNELS\n```cpp\ncode\n```",
    )

    assert request["task_id"] != "custom-task-id"
    assert re.fullmatch(r"parallel_task_\d{6}_[0-9a-f]{8}", request["task_id"])


@pytest.mark.unit
def test_build_evaluation_request_ignores_metadata_overrides_and_unknown_args():
    sample = _Sample()
    for key in (
        "toolkit",
        "backend_adapter",
        "workflow",
        "use_reference_cache",
        "timeout",
        "verbose_errors",
        "enable_profiling",
        "detect_decoy_kernel",
        "num_correct_trials",
        "num_perf_trials",
        "num_warmup",
        "perf_trim_count",
        "priority",
        "device_preference",
        "force_refresh",
        "reference_backend",
        "measure_performance",
        "run_correctness",
        "run_performance",
        "resources",
    ):
        sample.metadata[key] = "metadata-override"

    args = SimpleNamespace(
        kernelgym_toolkit="arg-override",
        kernelgym_backend_adapter="arg-override",
        kernelgym_workflow="arg-override",
        kernelgym_use_reference_cache=False,
        kernelgym_timeout="arg-override",
        kernelgym_verbose_errors=False,
        kernelgym_enable_profiling=False,
        kernelgym_detect_decoy_kernel=False,
        kernelgym_priority="arg-override",
        kernelgym_device_preference="arg-override",
        kernelgym_force_refresh="arg-override",
        kernelgym_measure_performance="arg-override",
        kernelgym_run_correctness="arg-override",
        kernelgym_run_performance="arg-override",
        kernelgym_resources="arg-override",
    )

    request = build_evaluation_request(
        args,
        sample,
        kernel_code="### CUDA_KERNELS\n```cpp\ncode\n```",
    )

    assert request["workflow"] == KERNELGYM_WORKFLOW
    assert request["use_reference_cache"] is KERNELGYM_USE_REFERENCE_CACHE
    assert request["timeout"] == KERNELGYM_TASK_TIMEOUT_S
    assert request["verbose_errors"] is KERNELGYM_VERBOSE_ERRORS
    assert request["enable_profiling"] is KERNELGYM_ENABLE_PROFILING
    assert request["detect_decoy_kernel"] is KERNELGYM_DETECT_DECOY_KERNEL
    assert request["num_correct_trials"] == KERNELGYM_NUM_CORRECT_TRIALS
    assert request["num_perf_trials"] == KERNELGYM_NUM_PERF_TRIALS
    assert request["num_warmup"] == KERNELGYM_NUM_WARMUP
    assert request["perf_trim_count"] == KERNELGYM_PERF_TRIM_COUNT
    assert request["reference_backend"] == KERNELGYM_REFERENCE_BACKEND
    for key in (
        "toolkit",
        "backend_adapter",
        "priority",
        "device_preference",
        "force_refresh",
        "measure_performance",
        "run_correctness",
        "run_performance",
        "resources",
    ):
        assert key not in request


@pytest.mark.unit
def test_build_evaluation_request_accepts_explicit_tuning_args():
    request = build_evaluation_request(
        SimpleNamespace(
            kernelgym_num_correct_trials="7",
            kernelgym_num_perf_trials="80",
            kernelgym_num_warmup="11",
            kernelgym_perf_trim_count="2",
            kernelgym_reference_backend="pytorch",
        ),
        _Sample(),
        kernel_code="### CUDA_KERNELS\n```cpp\ncode\n```",
    )

    assert request["num_correct_trials"] == 7
    assert request["num_perf_trials"] == 80
    assert request["num_warmup"] == 11
    assert request["perf_trim_count"] == 2
    assert request["reference_backend"] == "pytorch"


@pytest.mark.unit
def test_build_evaluation_request_keeps_is_valid_configurable():
    sample = _Sample()
    sample.metadata["is_valid"] = True

    request = build_evaluation_request(
        SimpleNamespace(),
        sample,
        kernel_code="### CUDA_KERNELS\n```cpp\ncode\n```",
    )

    assert request["is_valid"] is True


@pytest.mark.unit
def test_rm_url_is_required():
    assert _get_rm_url(SimpleNamespace(rm_url="http://kernelgym")) == "http://kernelgym"
    with pytest.raises(ValueError):
        _get_rm_url(SimpleNamespace(rm_url=None))


@pytest.mark.unit
def test_kernelgym_client_default_timeout_is_hardcoded():
    assert KernelGymClient("http://kernelgym").timeout_s == KERNELGYM_CLIENT_TIMEOUT_S


@pytest.mark.unit
def test_kernelgym_result_to_reward_requires_compile_correct_and_not_decoy():
    assert kernelgym_result_to_reward({"compiled": True, "correctness": True, "decoy_kernel": False}) == 1.0
    assert kernelgym_result_to_reward({"compiled": False, "correctness": True, "decoy_kernel": False}) == 0.0
    assert kernelgym_result_to_reward({"compiled": True, "correctness": False, "decoy_kernel": False}) == 0.0
    assert kernelgym_result_to_reward({"compiled": True, "correctness": True, "decoy_kernel": True}) == 0.0


@pytest.mark.unit
def test_evaluate_sample_returns_zero_reward_for_incomplete_submission():
    sample = _Sample()
    sample.response = "### CUDA_KERNELS\n```cpp\nonly kernels\n```"

    output = asyncio.run(evaluate_sample(SimpleNamespace(), sample))

    assert output == {"reward": 0.0, "extract_error": MISSING_COMPLETE_SUBMISSION_ERROR}
    assert sample.metadata["kernelgym"] == {
        "extract_error": MISSING_COMPLETE_SUBMISSION_ERROR,
        "reward": 0.0,
    }
    assert "kernel_submission" not in sample.metadata
