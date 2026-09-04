import sys
from pathlib import Path

import pytest

NUM_GPUS = 0

# tests/conftest.py may add another checkout that also owns an ``examples``
# package. Keep this repository first so the request script import is stable.
REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_path = str(REPO_ROOT)
if repo_root_path in sys.path:
    sys.path.remove(repo_root_path)
sys.path.insert(0, repo_root_path)

from examples.kernel_agent.test.run_request_env import (
    COMPILE_ERROR_CASES,
    COMPILE_ERROR_EXPECTED_DETAILS,
    COMPILE_ERROR_KERNEL_CODES,
    SANITIZER_KERNEL_CODE,
    SANITIZER_NAME_ERROR_KERNEL_CODE,
    _build_payload,
    _validate_mode_result,
)


@pytest.fixture
def env_config() -> dict:
    return {
        "num_correct_trials": 2,
        "num_perf_trials": 4,
        "num_warmup": 1,
        "perf_trim_count": 0,
        "adaptive_perf_trials": False,
        "perf_min_trials": 2,
        "perf_cv_threshold": 0.05,
        "kernel_eval_task_timeout": 120,
        "kernel_eval_priority": "normal",
        "verbose_errors": True,
        "enable_profiling": True,
        "compute_sanitizer_mode": "error_based",
        "enable_correctness_input_perturbations": False,
        "memory_ratio_threshold": 1.8,
        "detect_decoy_kernel": True,
        "use_reference_cache": False,
        "split_compile_and_execute": True,
        "enable_compile_artifact_cache": True,
        "simplify_error": True,
    }


@pytest.mark.parametrize("mode", ["sanitizer", "ncu", "compile"])
def test_request_modes_enable_only_the_requested_diagnostic(mode: str, env_config: dict) -> None:
    payload = _build_payload(mode, f"test-{mode}", env_config)

    assert payload["enable_compute_sanitizer"] is (mode == "sanitizer")
    assert payload["enable_ncu"] is (mode == "ncu")
    assert payload["force_refresh"] is True
    assert payload["compute_sanitizer_mode"] == "error_based"
    assert payload["simplify_error"] is True

    if mode == "compile":
        assert payload["pure_compile_task"] is True
        assert payload["task_stage"] == "compile"
        assert payload["required_resource"] == "cpu"
        assert payload["split_compile_and_execute"] is False
        assert payload["enable_compile_artifact_cache"] is False
        assert "dtype().bytes" in payload["kernel_code"]
    else:
        assert "pure_compile_task" not in payload
        assert "dtype().bytes" not in payload["kernel_code"]


@pytest.mark.parametrize(
    ("mode", "raw_env"),
    [
        ("sanitizer", {"runtime_sanitizer": {"status": "issues_found"}}),
        ("ncu", {"metadata": {"ncu": {"status": "ok", "profiled_kernel_count": 1}}}),
        (
            "compile",
            {
                "compiled": False,
                "error_code": "COMPILATION_ERROR",
                "metadata": {
                    "compilation_error_detail": {
                        "tvm_ffi_api_dtype": ["generated_binding.cpp:8:70: error: DLDataType has no member bytes"]
                    },
                },
            },
        ),
    ],
)
def test_request_mode_result_validation_accepts_expected_response(mode: str, raw_env: dict) -> None:
    succeeded, _detail = _validate_mode_result(mode, raw_env)
    assert succeeded is True


def test_request_mode_result_validation_rejects_missing_diagnostic() -> None:
    succeeded, detail = _validate_mode_result("ncu", {"metadata": {}})
    assert succeeded is False
    assert "profiled_kernel_count=0" in detail


def test_python_name_error_sanitizer_case_builds_runtime_failure_payload(env_config: dict) -> None:
    payload = _build_payload(
        "sanitizer",
        "test-python-name-error",
        env_config,
        sanitizer_error_case="python_name_error",
    )

    assert payload["uuid"] == "request-env-sanitizer-python_name_error"
    assert payload["kernel_code"] == SANITIZER_NAME_ERROR_KERNEL_CODE
    assert "import tvm_ffi_extension" not in payload["kernel_code"]
    assert "tvm_ffi_extension.copy_forward" in payload["kernel_code"]


def test_cuda_memory_sanitizer_case_runs_unsafe_write_only_during_replay(env_config: dict) -> None:
    payload = _build_payload("sanitizer", "test-cuda-memory-error", env_config)

    assert payload["kernel_code"] == SANITIZER_KERNEL_CODE
    assert 'os.environ.get("KERNELGYM_COMPUTE_SANITIZER_TOOL") != "memcheck"' in payload["kernel_code"]
    assert "output[n] = 1.0f" in payload["kernel_code"]
    assert "reinterpret_cast<volatile float*>(0x1)" not in payload["kernel_code"]
    assert "an illegal memory access was encountered" in payload["kernel_code"]


def test_python_name_error_sanitizer_case_accepts_skipped_diagnostic() -> None:
    succeeded, detail = _validate_mode_result(
        "sanitizer",
        {
            "metadata": {
                "runtime_error": "NameError: name 'tvm_ffi_extension' is not defined",
            },
            "runtime_sanitizer": {"status": "skipped", "reason": "python_name_error"},
        },
        sanitizer_error_case="python_name_error",
    )

    assert succeeded is True, detail


@pytest.mark.parametrize("compile_error_case", COMPILE_ERROR_CASES)
def test_compile_error_cases_build_distinct_compile_only_payloads(
    compile_error_case: str,
    env_config: dict,
) -> None:
    payload = _build_payload(
        "compile",
        f"test-{compile_error_case}",
        env_config,
        compile_error_case=compile_error_case,
    )

    assert payload["uuid"] == f"request-env-compile-{compile_error_case}"
    assert payload["pure_compile_task"] is True
    assert payload["task_stage"] == "compile"
    assert payload["required_resource"] == "cpu"
    assert payload["kernel_code"] == COMPILE_ERROR_KERNEL_CODES[compile_error_case]


@pytest.mark.parametrize("compile_error_case", COMPILE_ERROR_CASES)
def test_compile_error_cases_validate_grouped_detail_excerpts(compile_error_case: str) -> None:
    expected_details = COMPILE_ERROR_EXPECTED_DETAILS[compile_error_case]
    raw_env = {
        "compiled": False,
        "error_code": "COMPILATION_ERROR",
        "metadata": {
            "compilation_error_detail": {
                error_type: [f"generated_binding.cpp:10:5: error: {excerpt_token}"]
                for error_type, excerpt_token in expected_details.items()
            },
        },
    }

    succeeded, _detail = _validate_mode_result(
        "compile",
        raw_env,
        compile_error_case=compile_error_case,
    )
    assert succeeded is True


def test_compile_error_case_requires_matching_excerpt() -> None:
    raw_env = {
        "compiled": False,
        "error_code": "COMPILATION_ERROR",
        "metadata": {
            "compilation_error_detail": {
                "invalid_type_conversion": ["generated_binding.cpp:10:5: error: unrelated failure"]
            },
        },
    }

    succeeded, detail = _validate_mode_result(
        "compile",
        raw_env,
        compile_error_case="invalid_type_conversion",
    )
    assert succeeded is False
    assert "unrelated failure" in detail


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
