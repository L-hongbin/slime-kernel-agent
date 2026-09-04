#!/usr/bin/env python3
"""Run focused KernelGym requests and print raw plus normalized ENV feedback.

The script provides three self-contained TVM-FFI request modes:

* ``sanitizer`` triggers Compute Sanitizer with an out-of-bounds global write.
* ``ncu`` profiles a correct identity kernel with Nsight Compute.
* ``compile`` triggers one or all supported compile-error classifications.

Run with the local KernelGym default::

    python examples/kernel_agent/test/run_request_env.py --mode sanitizer
    python examples/kernel_agent/test/run_request_env.py --mode sanitizer \
      --sanitizer-error-case python_name_error

Run another request mode::

    python examples/kernel_agent/test/run_request_env.py --mode ncu
    python examples/kernel_agent/test/run_request_env.py --mode compile
    python examples/kernel_agent/test/run_request_env.py --mode compile \
      --compile-error-case invalid_type_conversion
    python examples/kernel_agent/test/run_request_env.py --mode compile --compile-error-case all

Override the service when needed::

    KERNEL_ENV_URL=http://host:20111 \
      python examples/kernel_agent/test/run_request_env.py --mode compile
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.kernel_agent.config import CUDA_AGENT_CONFIGS
from examples.kernel_agent.utils import normalize_env_feedback

REFERENCE_CODE = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.clone()


def get_init_inputs():
    return []


def get_inputs():
    return [torch.randn(1000, dtype=torch.float32)]
"""

REQUEST_MODES = ("sanitizer", "ncu", "compile")
DEFAULT_SANITIZER_ERROR_CASE = "cuda_memory_error"
SANITIZER_ERROR_CASES = (DEFAULT_SANITIZER_ERROR_CASE, "python_name_error")
DEFAULT_COMPILE_ERROR_CASE = "tvm_ffi_api_dtype"
COMPILE_ERROR_CASES = (
    DEFAULT_COMPILE_ERROR_CASE,
    "undefined_identifier",
    "missing_header",
    "function_argument_mismatch",
    "invalid_type_conversion",
    "invalid_declaration",
    "syntax_error",
    "incomplete_type",
    "multiple_errors",
)

# Normal correctness raises a deterministic runtime error to trigger memcheck
# without executing unsafe CUDA work in the regular GPU worker. KernelGym sets
# KERNELGYM_COMPUTE_SANITIZER_TOOL only in its isolated sanitizer replay; during
# the memcheck replay, thread 0 performs a one-element write past an exact
# allocation so the tool can report invalid_global_write without using an
# absolute invalid address. Other sanitizer tools retain the safe trigger path.
SANITIZER_KERNEL_CODE = r"""
### CUDA_KERNELS
```cpp
#include <cuda_runtime.h>

__global__ void sanitizer_oob_kernel(const float* input, float* output, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        output[index] = input[index];
    }
    if (index == 0) {
        output[n] = 1.0f;
    }
}

extern "C" void sanitizer_oob_launcher(
    const float* input, float* output, int n, void* stream_handle) {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
    float* exact_allocation = nullptr;
    cudaMalloc(&exact_allocation, static_cast<size_t>(n) * sizeof(float));
    int grid = (n + 255) / 256;
    sanitizer_oob_kernel<<<grid, 256, 0, stream>>>(input, exact_allocation, n);
    cudaMemcpyAsync(
        output,
        exact_allocation,
        static_cast<size_t>(n) * sizeof(float),
        cudaMemcpyDeviceToDevice,
        stream);
    cudaFree(exact_allocation);
}
```

### APPLY_BINDINGS
```cpp
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

extern "C" void sanitizer_oob_launcher(
    const float* input, float* output, int n, void* stream_handle);

void sanitizer_forward(tvm::ffi::Tensor input, tvm::ffi::Tensor output) {
    void* stream =
        TVMFFIEnvGetStream(input.device().device_type, input.device().device_id);
    sanitizer_oob_launcher(
        static_cast<const float*>(input.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<int>(input.numel()),
        stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(sanitizer_forward, sanitizer_forward);
```

### MODEL_NEW
```python
import os

import torch
import torch.nn as nn
import tvm_ffi_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if os.environ.get("KERNELGYM_COMPUTE_SANITIZER_TOOL") != "memcheck":
            raise RuntimeError(
                "CUDA error: an illegal memory access was encountered "
                "(intentional sanitizer test trigger)"
            )
        output = torch.empty_like(x)
        tvm_ffi_extension.sanitizer_forward(x, output)
        return output
```
"""


NCU_KERNEL_CODE = r"""
### CUDA_KERNELS
```cpp
#include <cuda_runtime.h>

__global__ void copy_kernel(const float* input, float* output, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        output[index] = input[index];
    }
}

extern "C" void copy_launcher(
    const float* input, float* output, int n, void* stream_handle) {
    cudaStream_t stream = static_cast<cudaStream_t>(stream_handle);
    int grid = (n + 255) / 256;
    copy_kernel<<<grid, 256, 0, stream>>>(input, output, n);
}
```

### APPLY_BINDINGS
```cpp
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

extern "C" void copy_launcher(
    const float* input, float* output, int n, void* stream_handle);

void copy_forward(tvm::ffi::Tensor input, tvm::ffi::Tensor output) {
    DLDataType f32_dtype{kDLFloat, 32, 1};
    TVM_FFI_ICHECK(input.dtype() == f32_dtype) << "input must be float32";
    TVM_FFI_ICHECK(output.dtype() == f32_dtype) << "output must be float32";
    void* stream =
        TVMFFIEnvGetStream(input.device().device_type, input.device().device_id);
    copy_launcher(
        static_cast<const float*>(input.data_ptr()),
        static_cast<float*>(output.data_ptr()),
        static_cast<int>(input.numel()),
        stream);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(copy_forward, copy_forward);
```

### MODEL_NEW
```python
import torch
import torch.nn as nn
import tvm_ffi_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        output = torch.empty_like(x)
        tvm_ffi_extension.copy_forward(x, output)
        return output
```
"""


# Keep the extension and exported function valid, but omit the Python import.
# The candidate therefore compiles successfully and fails in forward() before
# launching its CUDA kernel with a deterministic Python NameError.
SANITIZER_NAME_ERROR_KERNEL_CODE = NCU_KERNEL_CODE.replace("import tvm_ffi_extension\n", "", 1)


_COMPILE_INJECTION_MARKER = "    DLDataType f32_dtype{kDLFloat, 32, 1};"


def _inject_compile_error(statement: str) -> str:
    assert _COMPILE_INJECTION_MARKER in NCU_KERNEL_CODE
    return NCU_KERNEL_CODE.replace(
        _COMPILE_INJECTION_MARKER,
        f"{statement}\n{_COMPILE_INJECTION_MARKER}",
        1,
    )


_CUDA_COMPILE_INJECTION_MARKER = "        output[index] = input[index];"


def _inject_cuda_compile_error(kernel_code: str, statement: str) -> str:
    assert _CUDA_COMPILE_INJECTION_MARKER in kernel_code
    return kernel_code.replace(
        _CUDA_COMPILE_INJECTION_MARKER,
        f"{statement}\n{_CUDA_COMPILE_INJECTION_MARKER}",
        1,
    )


# Keep the CUDA kernel valid and introduce exactly one binding compilation
# failure per case so the expected classifier is unambiguous.
COMPILE_ERROR_KERNEL_CODES = {
    "tvm_ffi_api_dtype": _inject_compile_error(
        """    TVM_FFI_ICHECK(output.dtype().code == kDLFloat && output.dtype().bytes == 4)
        << "output must be float32";"""
    ),
    "undefined_identifier": _inject_compile_error("    int parsed_value = undefined_identifier_for_test;"),
    "missing_header": NCU_KERNEL_CODE.replace(
        "#include <tvm/ffi/tvm_ffi.h>",
        "#include <kernelgym_missing_compile_case_header.h>\n#include <tvm/ffi/tvm_ffi.h>",
        1,
    ),
    "function_argument_mismatch": _inject_compile_error("    copy_launcher(nullptr, nullptr);"),
    "invalid_type_conversion": _inject_compile_error(
        """    void* raw_pointer = nullptr;
    float* typed_pointer = raw_pointer;"""
    ),
    "invalid_declaration": _inject_compile_error("    void invalid_declaration_for_test;"),
    "syntax_error": _inject_compile_error("    int syntax_error_for_test = ;"),
    "incomplete_type": _inject_compile_error(
        """    struct incomplete_type_for_test;
    incomplete_type_for_test incomplete_value_for_test;"""
    ),
    "multiple_errors": _inject_cuda_compile_error(
        _inject_compile_error("    int syntax_error_for_test = ;"),
        """        struct incomplete_type_for_test;
        incomplete_type_for_test incomplete_value_for_test;""",
    ),
}

COMPILE_ERROR_EXPECTED_DETAILS = {
    "tvm_ffi_api_dtype": {"tvm_ffi_api_dtype": "DLDataType"},
    "undefined_identifier": {"undefined_identifier": "undefined_identifier_for_test"},
    "missing_header": {"missing_header": "kernelgym_missing_compile_case_header.h"},
    "function_argument_mismatch": {"function_argument_mismatch": "copy_launcher"},
    "invalid_type_conversion": {"invalid_type_conversion": "void*"},
    "invalid_declaration": {"invalid_declaration": "invalid_declaration_for_test"},
    "syntax_error": {"syntax_error": "expected primary-expression"},
    "incomplete_type": {"incomplete_type": "incomplete type"},
    "multiple_errors": {
        "syntax_error": "expected primary-expression",
        "incomplete_type": "incomplete type",
    },
}

MODE_KERNEL_CODE = {
    "ncu": NCU_KERNEL_CODE,
}

SANITIZER_KERNEL_CODES = {
    "cuda_memory_error": SANITIZER_KERNEL_CODE,
    "python_name_error": SANITIZER_NAME_ERROR_KERNEL_CODE,
}


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _build_payload(
    mode: str,
    task_id: str,
    env_config: dict[str, Any],
    *,
    sanitizer_error_case: str = DEFAULT_SANITIZER_ERROR_CASE,
    compile_error_case: str = DEFAULT_COMPILE_ERROR_CASE,
) -> dict[str, Any]:
    if mode not in REQUEST_MODES:
        raise ValueError(f"Unsupported request mode: {mode!r}")
    if compile_error_case not in COMPILE_ERROR_CASES:
        raise ValueError(f"Unsupported compile error case: {compile_error_case!r}")
    if sanitizer_error_case not in SANITIZER_ERROR_CASES:
        raise ValueError(f"Unsupported sanitizer error case: {sanitizer_error_case!r}")

    if mode == "compile":
        kernel_code = COMPILE_ERROR_KERNEL_CODES[compile_error_case]
    elif mode == "sanitizer":
        kernel_code = SANITIZER_KERNEL_CODES[sanitizer_error_case]
    else:
        kernel_code = MODE_KERNEL_CODE[mode]

    payload = {
        "task_id": task_id,
        "reference_code": REFERENCE_CODE,
        "kernel_code": kernel_code,
        "backend": "tvm_ffi",
        "reference_backend": "torch",
        "entry_point": "Model",
        "uuid": (
            f"request-env-{mode}-{compile_error_case}"
            if mode == "compile"
            else f"request-env-{mode}-{sanitizer_error_case}" if mode == "sanitizer" else f"request-env-{mode}"
        ),
        "num_correct_trials": env_config.get("num_correct_trials"),
        "num_perf_trials": env_config.get("num_perf_trials"),
        "num_warmup": env_config.get("num_warmup"),
        "perf_trim_count": env_config.get("perf_trim_count"),
        "adaptive_perf_trials": env_config.get("adaptive_perf_trials"),
        "perf_min_trials": env_config.get("perf_min_trials"),
        "perf_cv_threshold": env_config.get("perf_cv_threshold"),
        "timeout": env_config.get("kernel_eval_task_timeout"),
        "priority": env_config.get("kernel_eval_priority", "normal"),
        "is_valid": False,
        "verbose_errors": env_config.get("verbose_errors", True),
        "enable_profiling": bool(env_config.get("enable_profiling", True)),
        "enable_ncu": mode == "ncu",
        "enable_compute_sanitizer": mode == "sanitizer",
        "compute_sanitizer_mode": env_config.get("compute_sanitizer_mode", "error_based"),
        "enable_correctness_input_perturbations": bool(
            env_config.get("enable_correctness_input_perturbations", False)
        ),
        "simplify_error": bool(env_config.get("simplify_error", True)),
        "memory_ratio_threshold": env_config.get("memory_ratio_threshold"),
        "detect_decoy_kernel": env_config.get("detect_decoy_kernel", True),
        "use_reference_cache": env_config.get("use_reference_cache", False),
        "split_compile_and_execute": env_config.get("split_compile_and_execute", True),
        "enable_compile_artifact_cache": env_config.get("enable_compile_artifact_cache", True),
        "force_refresh": True,
    }
    for name in ("refer_num_perf_trials", "correctness_timeout", "correctness_timeout_enabled"):
        value = env_config.get(name)
        if value is not None:
            payload[name] = value

    if mode == "compile":
        payload.update(
            {
                "pure_compile_task": True,
                "task_stage": "compile",
                "required_resource": "cpu",
                "split_compile_and_execute": False,
                "enable_compile_artifact_cache": False,
            }
        )
    return payload


def _submit_and_poll(payload: dict[str, Any], env_config: dict[str, Any]) -> dict[str, Any]:
    server_url = str(env_config["kernel_env_url"]).rstrip("/")
    task_id = payload["task_id"]
    task_timeout = float(env_config["kernel_eval_task_timeout"])
    client_timeout = float(env_config["kernel_eval_client_timeout"])
    poll_interval = float(env_config["kernel_eval_poll_interval"])
    heartbeat_interval = float(env_config["kernel_eval_heartbeat_interval"])
    max_retries = max(1, int(env_config["kernel_eval_max_retries"]))
    started_at = time.monotonic()
    next_heartbeat = started_at + heartbeat_interval

    timeout = httpx.Timeout(connect=10.0, read=min(task_timeout, 60.0), write=10.0, pool=5.0)
    with httpx.Client(timeout=timeout, headers={"Content-Type": "application/json"}) as client:
        for attempt in range(max_retries):
            try:
                response = client.post(f"{server_url}/evaluate", json=payload)
                if response.status_code in {200, 409}:
                    break
                response.raise_for_status()
            except (httpx.TimeoutException, httpx.ConnectError):
                if attempt + 1 == max_retries:
                    # The server may have accepted the client-provided task_id
                    # before the response timed out, so continue with polling.
                    break
                time.sleep(min(2**attempt, 10))

        while time.monotonic() - started_at < client_timeout:
            status_response = client.get(f"{server_url}/status/{task_id}")
            if status_response.status_code == 200:
                status_payload = status_response.json()
                status = status_payload.get("status", "unknown")
                if status in {"completed", "failed", "timeout", "cancelled"}:
                    result_response = client.get(f"{server_url}/results/{task_id}")
                    if result_response.status_code == 200:
                        result = result_response.json()
                        result["status"] = status
                        return result
                    return status_payload

            now = time.monotonic()
            if heartbeat_interval > 0 and now >= next_heartbeat:
                print(f"Waiting for task_id={task_id}: elapsed={now - started_at:.1f}s", flush=True)
                next_heartbeat = now + heartbeat_interval
            time.sleep(poll_interval)

    raise TimeoutError(f"KernelGym task {task_id} did not finish within {client_timeout:.0f}s")


def _validate_mode_result(
    mode: str,
    raw_env: dict[str, Any],
    *,
    sanitizer_error_case: str = DEFAULT_SANITIZER_ERROR_CASE,
    compile_error_case: str = DEFAULT_COMPILE_ERROR_CASE,
) -> tuple[bool, str]:
    if mode == "sanitizer":
        sanitizer = raw_env.get("runtime_sanitizer")
        status = sanitizer.get("status") if isinstance(sanitizer, dict) else None
        expected_status = "skipped" if sanitizer_error_case == "python_name_error" else "issues_found"
        runtime_error = str((raw_env.get("metadata") or {}).get("runtime_error") or "")
        success = status == expected_status
        if sanitizer_error_case == "python_name_error":
            success = (
                success
                and sanitizer.get("reason") == "python_name_error"
                and "NameError" in runtime_error
                and "tvm_ffi_extension" in runtime_error
            )
        return success, (
            f"runtime_sanitizer.status={status!r}, expected {expected_status!r}, " f"runtime_error={runtime_error!r}"
        )

    metadata = raw_env.get("metadata") if isinstance(raw_env.get("metadata"), dict) else {}
    if mode == "ncu":
        ncu = metadata.get("ncu") if isinstance(metadata.get("ncu"), dict) else {}
        status = ncu.get("status")
        kernel_count = ncu.get("profiled_kernel_count", 0)
        success = status == "ok" and isinstance(kernel_count, int) and kernel_count > 0
        return success, f"metadata.ncu status={status!r}, profiled_kernel_count={kernel_count!r}"

    detail = metadata.get("compilation_error_detail")
    expected_details = COMPILE_ERROR_EXPECTED_DETAILS[compile_error_case]
    details_match = isinstance(detail, dict) and all(
        isinstance(detail.get(error_type), list)
        and any(isinstance(excerpt, str) and expected_excerpt_token in excerpt for excerpt in detail[error_type])
        for error_type, expected_excerpt_token in expected_details.items()
    )
    success = raw_env.get("compiled") is False and raw_env.get("error_code") == "COMPILATION_ERROR" and details_match
    return success, (
        f"compiled={raw_env.get('compiled')!r}, error_code={raw_env.get('error_code')!r}, "
        f"compilation_error_detail={detail!r}"
    )


def _run(args: argparse.Namespace) -> int:
    env_config = dict(CUDA_AGENT_CONFIGS["env"])
    env_config["kernel_env_url"] = args.kernel_env_url

    if args.mode == "compile":
        compile_error_cases = (
            list(COMPILE_ERROR_CASES) if args.compile_error_case == "all" else [args.compile_error_case]
        )
    else:
        compile_error_cases = [DEFAULT_COMPILE_ERROR_CASE]
    records = []
    exit_code = 0
    for compile_error_case in compile_error_cases:
        task_label = (
            f"{args.mode}_{compile_error_case}"
            if args.mode == "compile"
            else f"{args.mode}_{args.sanitizer_error_case}" if args.mode == "sanitizer" else args.mode
        )
        task_id = f"request_env_{task_label}_{uuid4().hex[:12]}"
        payload = _build_payload(
            args.mode,
            task_id,
            env_config,
            sanitizer_error_case=args.sanitizer_error_case,
            compile_error_case=compile_error_case,
        )

        print(
            f"Submitting mode={args.mode} compile_error_case={compile_error_case} "
            f"task_id={task_id} to {args.kernel_env_url} "
            f"enable_compute_sanitizer={payload['enable_compute_sanitizer']} "
            f"enable_ncu={payload['enable_ncu']}",
            flush=True,
        )
        raw_env = _submit_and_poll(payload, env_config)
        if not isinstance(raw_env, dict):
            raise TypeError(f"KernelGym returned {type(raw_env).__name__}, expected dict")

        normalized_env, _env_extra_info = normalize_env_feedback(deepcopy(raw_env))
        record = {
            "mode": args.mode,
            "sanitizer_error_case": args.sanitizer_error_case if args.mode == "sanitizer" else None,
            "compile_error_case": compile_error_case if args.mode == "compile" else None,
            "raw_env": raw_env,
            "normalized_env": normalized_env,
        }
        records.append(record)

        print(f"\n========== RAW ENV ({task_label}) ==========")
        print(_json_dumps(raw_env))
        print(f"\n======= NORMALIZED ENV ({task_label}) =======")
        print(_json_dumps(normalized_env))

        succeeded, detail = _validate_mode_result(
            args.mode,
            raw_env,
            sanitizer_error_case=args.sanitizer_error_case,
            compile_error_case=compile_error_case,
        )
        if not succeeded:
            print(
                f"\nERROR: {task_label} request did not produce the expected result: {detail}",
                file=sys.stderr,
            )
            exit_code = 1
        else:
            print(f"\nValidated {task_label} result: {detail}")

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        saved_payload = records[0] if len(records) == 1 else {"mode": args.mode, "results": records}
        output.write_text(_json_dumps(saved_payload) + "\n", encoding="utf-8")
        print(f"\nSaved output to {output}")

    return exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=REQUEST_MODES, default="sanitizer")
    parser.add_argument(
        "--sanitizer-error-case",
        choices=SANITIZER_ERROR_CASES,
        default=DEFAULT_SANITIZER_ERROR_CASE,
        help="Runtime-error fixture to run with --mode sanitizer.",
    )
    parser.add_argument(
        "--compile-error-case",
        choices=(*COMPILE_ERROR_CASES, "all"),
        default=DEFAULT_COMPILE_ERROR_CASE,
        help="Compile-error fixture to run with --mode compile; 'all' runs every fixture sequentially.",
    )
    parser.add_argument(
        "--kernel-env-url",
        default=os.environ.get("KERNEL_ENV_URL", "http://127.0.0.1:20111"),
    )
    parser.add_argument("--output", default="")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return _run(args)


if __name__ == "__main__":
    raise SystemExit(main())
