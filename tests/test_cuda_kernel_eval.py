import asyncio
import inspect
import json
import logging
import os
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

NUM_GPUS = 0

# tests/conftest.py adds Megatron-LM to sys.path; that checkout also owns an
# ``examples`` package. Keep this repository first so the kernel-agent tests do
# not accidentally import Megatron-LM's unrelated package.
REPO_ROOT = Path(__file__).resolve().parents[1]
repo_root_path = str(REPO_ROOT)
if repo_root_path in sys.path:
    sys.path.remove(repo_root_path)
sys.path.insert(0, repo_root_path)

from examples.kernel_agent import generate_with_cuda_agent, kernel_response
from examples.kernel_agent import utils as kernel_agent_utils
from examples.kernel_agent.config import CUDA_AGENT_CONFIGS
from examples.kernel_agent.utils import (
    CORRECTNESS_ERROR,
    PRECHECK_ERROR,
    extract_cuda_agent_kernel_code,
    normalize_env_feedback,
    parse_cuda_agent_response,
    precheck_response,
    split_think_response,
)
from slime.utils.types import Sample

VALID_CUDA_AGENT_RESPONSE = """
### CUDA_KERNELS
```cpp
#include <cuda_runtime.h>

__global__ void copy_kernel(float* output, const float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx];
    }
}

extern "C" void copy_kernel_launcher(float* output, const float* input, int size, cudaStream_t stream) {
    copy_kernel<<<(size + 255) / 256, 256, 0, stream>>>(output, input, size);
}
```

### APPLY_BINDINGS
```cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void copy_kernel_launcher(float* output, const float* input, int size, cudaStream_t stream);

torch::Tensor copy_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    copy_kernel_launcher(output.data_ptr<float>(), input.data_ptr<float>(), input.numel(), stream);
    return output;
}

void register_copy(pybind11::module& m) {
    m.def("copy_forward", &copy_forward, "copy forward", py::arg("input"));
}

REGISTER_BINDING(copy, register_copy);
```

### MODEL_NEW
```python
import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return cuda_extension.copy_forward(x)
```
"""


VALID_TVM_FFI_RESPONSE = """
### CUDA_KERNELS
```cpp
#include <cuda_runtime.h>

__global__ void copy_kernel(float* output, const float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx];
    }
}

extern "C" void copy_kernel_launcher(float* output, const float* input, int size, void* stream_handle) {
    auto stream = static_cast<cudaStream_t>(stream_handle);
    copy_kernel<<<(size + 255) / 256, 256, 0, stream>>>(output, input, size);
}
```

### APPLY_BINDINGS
```cpp
#include <tvm/ffi/tvm_ffi.h>
#include <tvm/ffi/extra/c_env_api.h>

extern "C" void copy_kernel_launcher(float* output, const float* input, int size, void* stream_handle);

void copy_forward(tvm::ffi::Tensor input, tvm::ffi::Tensor output) {
    void* stream_handle = TVMFFIEnvGetStream(input.device().device_type, input.device().device_id);
    copy_kernel_launcher(
        static_cast<float*>(output.data_ptr()),
        static_cast<const float*>(input.data_ptr()),
        input.numel(),
        stream_handle);
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
        tvm_ffi_extension.copy_forward(x.contiguous(), output)
        return output
```
"""


INVALID_COMPILE_CUDA_AGENT_RESPONSE = """
### CUDA_KERNELS
```cpp
#include <cuda_runtime.h>

__global__ void broken_kernel(float* output, const float* input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = input[idx]
    }
}

extern "C" void broken_kernel_launcher(float* output, const float* input, int size, cudaStream_t stream) {
    broken_kernel<<<(size + 255) / 256, 256, 0, stream>>>(output, input, size);
}
```

### APPLY_BINDINGS
```cpp
#include <torch/types.h>
#include <torch/csrc/utils/pybind.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "../binding_registry.h"

extern "C" void broken_kernel_launcher(float* output, const float* input, int size, cudaStream_t stream);

torch::Tensor broken_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    broken_kernel_launcher(output.data_ptr<float>(), input.data_ptr<float>(), input.numel(), stream);
    return output;
}

void register_broken(pybind11::module& m) {
    m.def("broken_forward", &broken_forward, "broken forward", py::arg("input"));
}

REGISTER_BINDING(broken, register_broken);
```

### MODEL_NEW
```python
import torch
import torch.nn as nn
import cuda_extension


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return cuda_extension.broken_forward(x)
```
"""


REFERENCE_IDENTITY_CODE = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def get_inputs():
    return [torch.randn(8, 16, device="cuda")]


def get_init_inputs():
    return []
"""


KERNEL_EVAL_CASES = [
    pytest.param(
        {
            "source_row": 3136,
            "uuid": "5568",
            "feedback_compiled": True,
            "env_state": {
                "task_id": "parallel_task_000001_true",
                "status": "completed",
                "compiled": True,
                "correctness": True,
                "decoy_kernel": False,
                "reference_runtime": 0.0255,
                "kernel_runtime": 0.0171,
                "speedup": 1.4912280701754383,
                "metadata": {"backend": "cuda_agent"},
            },
        },
        id="feedback_compiled_true",
    ),
    pytest.param(
        {
            "source_row": 3960,
            "uuid": "6675",
            "feedback_compiled": False,
            "env_state": {
                "task_id": "parallel_task_000001_false",
                "status": "completed",
                "submitted_at": "2026-05-13T00:00:00Z",
                "completed_at": "2026-05-13T00:00:03Z",
                "compiled": False,
                "correctness": False,
                "decoy_kernel": False,
                "reference_runtime": 0.602,
                "kernel_runtime": 0,
                "speedup": 0,
                "metadata": {
                    "compile_only": True,
                    "entry_point": "Model",
                    "required_resource": "cpu",
                    "task_id": "parallel_task_000001_false",
                    "inline_gpu_execute_completed": True,
                    "inline_compile_worker_id": "worker_gpu_5",
                    "inline_compile_worker_device": "cuda:5",
                    "compile_artifact": {
                        "backend": "cuda_agent",
                        "compile_mode": "filesystem",
                        "compiled": False,
                        "entry_point": "ModelNew",
                        "error": "nvcc fatal: syntax error near token '}'",
                        "precheck": {"passed": True},
                        "source_mode": "files",
                    },
                },
                "error_message": "Compilation error: failed to build extension",
                "error_code": "compile_failed",
            },
        },
        id="feedback_compiled_false",
    ),
]


REAL_KERNEL_EVAL_CASES = [
    pytest.param(
        {
            "uuid": "real-kernel-eval-compiled-true",
            "response": VALID_CUDA_AGENT_RESPONSE,
            "expected_compiled": True,
        },
        id="real_feedback_compiled_true",
    ),
    pytest.param(
        {
            "uuid": "real-kernel-eval-compiled-false",
            "response": INVALID_COMPILE_CUDA_AGENT_RESPONSE,
            "expected_compiled": False,
        },
        id="real_feedback_compiled_false",
    ),
]


def _skip_unselected_compiled_case(request, case, compiled_key):
    selected = request.config.getoption("--cuda-kernel-compiled")
    if selected == "all":
        return
    expected_compiled = selected == "true"
    if case[compiled_key] is not expected_compiled:
        pytest.skip(f"filtered by --cuda-kernel-compiled={selected}")


def _format_feedback_for_test(env_result):
    template = generate_with_cuda_agent._get_tool_response_template(SimpleNamespace(multi_turn_template=None))
    return generate_with_cuda_agent._apply_feedback_template(env_result, template)


@pytest.mark.unit
@pytest.mark.parametrize("case", KERNEL_EVAL_CASES)
def test_precheck_accepts_cuda_agent_responses_from_compiled_feedback_cases(request, case):
    _skip_unselected_compiled_case(request, case, "feedback_compiled")
    precheck_passed, precheck_state = precheck_response(VALID_CUDA_AGENT_RESPONSE, "Model", "cuda_agent")
    assert precheck_passed is True
    assert precheck_state is None


@pytest.mark.unit
def test_precheck_accepts_tvm_ffi_responses():
    precheck_passed, precheck_state = precheck_response(VALID_TVM_FFI_RESPONSE, "Model", "tvm_ffi")
    assert precheck_passed is True
    assert precheck_state is None


@pytest.mark.unit
def test_precheck_rejects_tvm_ffi_missing_export():
    response = VALID_TVM_FFI_RESPONSE.replace(
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(copy_forward, copy_forward);",
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(copy_forward_exported, copy_forward);",
    )
    precheck_passed, result = precheck_response(response, "Model", "tvm_ffi")
    assert precheck_passed is False
    assert result is not None
    assert "TVM-FFI model calls are not exported: copy_forward" in result["error_message"]


@pytest.mark.unit
def test_parse_cuda_agent_response_uses_last_complete_section_group():
    response = """
### CUDA_KERNELS
```cpp
...
```

### APPLY_BINDINGS
```cpp
...
```

### MODEL_NEW
```python
...
```

### CUDA_KERNELS
```cpp
extern "C" void real_kernel_launcher(float* output, const float* input, int size, void* stream_handle) {}
```

### APPLY_BINDINGS
```cpp
#include <tvm/ffi/tvm_ffi.h>
void real_forward(tvm::ffi::Tensor input, tvm::ffi::Tensor output) {}
TVM_FFI_DLL_EXPORT_TYPED_FUNC(real_forward, real_forward);
```

### MODEL_NEW
```python
import torch.nn as nn
import tvm_ffi_extension

class ModelNew(nn.Module):
    def forward(self, x):
        tvm_ffi_extension.real_forward(x, x)
        return x
```
"""
    cuda_sources, model_new_code = parse_cuda_agent_response(response)
    assert "real_kernel_launcher" in cuda_sources["kernels/generated.cu"]
    assert "real_forward" in cuda_sources["kernels/generated_binding.cpp"]
    assert "class ModelNew" in model_new_code
    extracted = extract_cuda_agent_kernel_code(response)
    assert "..." not in extracted
    assert "real_kernel_launcher" in extracted


@pytest.mark.unit
@pytest.mark.parametrize("case", KERNEL_EVAL_CASES)
def test_cuda_kernel_env_uses_kernel_eval_result_and_multiturn_logs(request, monkeypatch, caplog, case):
    _skip_unselected_compiled_case(request, case, "feedback_compiled")
    monkeypatch.setitem(CUDA_AGENT_CONFIGS, "log_rollout_info_rate", 1.0)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS, "log_multi_turn_info", True)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS, "log_rollout_stats_only", False)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS, "log_first_rollout", True)
    monkeypatch.setattr(generate_with_cuda_agent, "_LOGGED_FIRST_ROLLOUT", False)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS, "max_feedback_chars", 8192)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["env"], "enable_ncu", False)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["env"], "enable_compute_sanitizer", True)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["env"], "compute_sanitizer_mode", "full")
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["env"], "enable_correctness_input_perturbations", True)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["env"], "memory_ratio_threshold", 2.25)

    captured_payload = {}

    async def fake_run_kernel_eval(args, sample, payload, config):
        captured_payload.update(payload)
        raw_env_state = dict(case["env_state"])
        raw_env_state["metadata"] = {
            **case["env_state"].get("metadata", {}),
            "kg_kernel_backend_compile_s": 1.0,
            "kg_kernel_perf_warmup_s": 0.5,
            "kg_kernel_perf_measure_wall_s": 1.5,
            "kg_kernel_perf_profile_s": 3.0,
            "kg_kernel_perf_mean_ms": 10.0,
            "kg_kernel_perf_std_ms": 2.5,
            "kg_reference_perf_warmup_s": 1.0,
            "kg_reference_perf_measure_wall_s": 3.0,
            "kg_reference_perf_mean_ms": 20.0,
            "kg_reference_perf_std_ms": 4.0,
        }
        return {"env_state": raw_env_state}

    monkeypatch.setattr(generate_with_cuda_agent, "run_kernel_eval", fake_run_kernel_eval)

    sample = Sample(
        prompt="Write a CUDA implementation.",
        label={"entry_point": "Model", "ground_truth": "class Model: pass"},
        metadata={
            "uuid": case["uuid"],
            "source_row": case["source_row"],
            "log_rollout_info": True,
        },
    )

    env_result = asyncio.run(
        generate_with_cuda_agent.cuda_kernel_env(
            SimpleNamespace(kernel_backend="cuda", reference_backend="torch"),
            sample,
            VALID_CUDA_AGENT_RESPONSE,
            turn_idx=0,
        )
    )

    env_state = env_result["env_state"]
    format_feedback = _format_feedback_for_test(env_result)
    print(f"\n[cuda_agent][test][format_feedback][{case['uuid']}]\n{format_feedback}")
    # uuid is the reference-cache key: a hash of (entry_point, ground_truth),
    # not the dataset's metadata uuid.
    assert captured_payload["uuid"] == generate_with_cuda_agent._reference_cache_uuid("class Model: pass", "Model")
    assert captured_payload["reference_code"] == "class Model: pass"
    assert captured_payload["kernel_code"] == extract_cuda_agent_kernel_code(VALID_CUDA_AGENT_RESPONSE)
    assert captured_payload["backend"] == "cuda"
    assert captured_payload["enable_ncu"] is False
    assert captured_payload["enable_compute_sanitizer"] is True
    assert captured_payload["compute_sanitizer_mode"] == "full"
    assert captured_payload["enable_correctness_input_perturbations"] is True
    assert captured_payload["memory_ratio_threshold"] == pytest.approx(2.25)
    assert "turn_idx" not in captured_payload
    assert "response" not in captured_payload
    assert "ground_truth" not in captured_payload
    assert "kernel_backend" not in captured_payload
    assert env_state["compiled"] is case["feedback_compiled"]
    assert env_state["status"] == "completed"
    assert "error_code" not in env_state
    assert "submitted_at" not in env_state
    assert "completed_at" not in env_state
    if not case["feedback_compiled"]:
        assert env_state["error"] == "COMPILATION_ERROR"
        assert env_state["correctness"] is None
        assert env_state["decoy_kernel"] is False
        assert env_state["reference_runtime"] == case["env_state"]["reference_runtime"]
        assert env_state["kernel_runtime"] == case["env_state"]["kernel_runtime"]
        assert env_state["speedup"] == case["env_state"]["speedup"]
        assert "Compilation failed. Compiler output:" in env_state["error_message"]
        assert "nvcc fatal: syntax error" in env_state["error_message"]
        assert "compile_only" not in env_state["metadata"]
        assert "device" not in env_state["metadata"]
        assert "entry_point" not in env_state["metadata"]
        assert "required_resource" not in env_state["metadata"]
        assert "task_id" not in env_state["metadata"]
        assert "inline_gpu_execute_completed" not in env_state["metadata"]
        assert "inline_compile_worker_id" not in env_state["metadata"]
        assert "inline_compile_worker_device" not in env_state["metadata"]
        assert "correctness_tf32_state_before" not in env_state["metadata"]
        assert "correctness_tf32_state_forced" not in env_state["metadata"]
        assert "correctness_atol" not in env_state["metadata"]
        assert "correctness_rtol" not in env_state["metadata"]
        assert "runtime_error" not in env_state["metadata"]
        assert env_state["metadata"]["refer_entry_point"] == "Model"
        assert env_state["metadata"]["kernel_entry_point"] == "ModelNew"
        assert "error" not in env_state["metadata"]["compile_artifact"]
        assert "precheck" not in env_state["metadata"]["compile_artifact"]
        assert "backend" not in env_state["metadata"]["compile_artifact"]
        assert "compile_mode" not in env_state["metadata"]["compile_artifact"]
        assert "compiled" not in env_state["metadata"]["compile_artifact"]
        assert "source_mode" not in env_state["metadata"]["compile_artifact"]
        assert "entry_point" not in env_state["metadata"]["compile_artifact"]
        assert "module_name" not in env_state["metadata"]["compile_artifact"]
        assert "profiling_hints" not in env_state["metadata"]["compile_artifact"]
        assert "artifact_node_id" not in env_state["metadata"]["compile_artifact"]
        assert "artifact_hostname" not in env_state["metadata"]["compile_artifact"]
        assert "target_gpu_worker_id" not in env_state["metadata"]["compile_artifact"]
        assert "target_gpu_selection_strategy" not in env_state["metadata"]["compile_artifact"]

    caplog.set_level(logging.INFO, logger=generate_with_cuda_agent.logger.name)
    response_with_think = f"<think>\ntry a simple copy kernel\n</think>\n{VALID_CUDA_AGENT_RESPONSE}"
    generate_with_cuda_agent._log_rollout_info(
        sample,
        messages=[
            {"role": "user", "content": sample.prompt},
            {"role": "assistant", "content": response_with_think},
            {"role": "user", "content": format_feedback},
        ],
        turn_logs=[
            {
                "turn_idx": 0,
                "task_id": case["env_state"]["task_id"],
                "model_time": 0.25,
                "env_time": 0.75,
                "prompt_tokens": 16,
                "response_tokens": 32,
                "finish_type": "stop",
                "prompt": sample.prompt,
                "response": response_with_think,
                "env_result": env_result,
                "format_feedback": format_feedback,
                "reward": 1.2456 if case["feedback_compiled"] else 0.0,
            }
        ],
        finish_reason="max_turns",
        should_log=True,
        is_slowest=True,
        total_request_time=1.0,
    )

    assert "[cuda_agent][first][rollout_info]" in caplog.text
    assert "[cuda_agent][first][slowest][rollout_info]" not in caplog.text
    assert "total_request_time=1.000s" in caplog.text
    assert f"compiled={case['feedback_compiled']}" in caplog.text
    assert "precheck=passed" in caplog.text
    assert f"task_id={case['env_state']['task_id']}" in caplog.text
    assert "reward=" in caplog.text
    assert "detail_env_time=" in caplog.text
    assert "compile_time" in caplog.text
    assert "perf_cv=" in caplog.text
    assert "kernel_perf_cv" in caplog.text
    assert "refer_perf_cv" in caplog.text
    expected_reward = 1.2456 if case["feedback_compiled"] else 0.0
    assert f"reward={expected_reward}" in caplog.text
    assert case["uuid"] in caplog.text
    assert "format_feedback" in caplog.text
    assert format_feedback in caplog.text
    assert "response_think" in caplog.text
    assert "try a simple copy kernel" in caplog.text
    assert "response_content" in caplog.text
    _, response_content = split_think_response(response_with_think)
    assert "### CUDA_KERNELS" in response_content
    if not case["feedback_compiled"]:
        assert "nvcc fatal: syntax error" in caplog.text


@pytest.mark.unit
def test_cuda_kernel_env_defaults_missing_entry_point_to_model(monkeypatch):
    captured_payload = {}

    async def fake_run_kernel_eval(args, sample, payload, config):
        captured_payload.update(payload)
        return {
            "env_state": {
                "compiled": True,
                "correctness": True,
                "speedup": 1.0,
                "metadata": {},
            }
        }

    monkeypatch.setattr(generate_with_cuda_agent, "run_kernel_eval", fake_run_kernel_eval)

    sample = Sample(
        prompt="Write a CUDA implementation.",
        label={"ground_truth": "class Model: pass"},
        metadata={"problem_id": 1},
    )

    asyncio.run(
        generate_with_cuda_agent.cuda_kernel_env(
            SimpleNamespace(kernel_backend="cuda", reference_backend="torch"),
            sample,
            VALID_CUDA_AGENT_RESPONSE,
            turn_idx=0,
        )
    )

    assert captured_payload["entry_point"] == "Model"
    assert captured_payload["uuid"] == generate_with_cuda_agent._reference_cache_uuid("class Model: pass", "Model")


@pytest.mark.unit
def test_split_think_response_handles_generation_prompt_prefilled_think():
    response = "reasoning from model\n</think>\n### CUDA_KERNELS\n```cpp\ncode\n```"
    response_think, response_content = split_think_response(response)
    assert response_think == "reasoning from model"
    assert response_content.startswith("### CUDA_KERNELS")


@pytest.mark.unit
def test_rollout_log_omits_turn_info_when_multi_turn_info_is_disabled(monkeypatch, caplog):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS, "log_multi_turn_info", False)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS, "log_rollout_stats_only", True)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS, "log_first_rollout", False)
    sample = Sample(
        prompt="Write a CUDA implementation.",
        metadata={"uuid": "log-trim"},
    )

    caplog.set_level(logging.INFO, logger=generate_with_cuda_agent.logger.name)
    generate_with_cuda_agent._log_rollout_info(
        sample,
        messages=[
            {"role": "user", "content": sample.prompt},
            {"role": "assistant", "content": VALID_CUDA_AGENT_RESPONSE},
        ],
        turn_logs=[
            {
                "turn_idx": 0,
                "model_time": 0.25,
                "env_time": 0.75,
                "prompt_tokens": 16,
                "response_tokens": 32,
                "finish_type": "stop",
                "prompt": sample.prompt,
                "response": VALID_CUDA_AGENT_RESPONSE,
                "env_result": {"env_state": {"status": "completed", "compiled": True}},
            }
        ],
        finish_reason="max_turns",
        is_slowest=True,
        total_request_time=1.0,
    )

    assert "[cuda_agent][slowest][rollout_info]" in caplog.text
    assert "sample=log-trim" in caplog.text
    assert "[turn 0]" not in caplog.text
    assert "### CUDA_KERNELS" not in caplog.text
    assert "response_content" not in caplog.text
    assert "[prompt]:" not in caplog.text


@pytest.mark.unit
def test_first_rollout_logs_turn_info_when_multi_turn_info_is_disabled(monkeypatch, caplog):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS, "log_multi_turn_info", False)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS, "log_rollout_stats_only", False)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS, "log_first_rollout", True)
    monkeypatch.setattr(generate_with_cuda_agent, "_LOGGED_FIRST_ROLLOUT", False)
    sample = Sample(prompt="first prompt", metadata={"uuid": "first-rollout"})
    messages = [{"role": "user", "content": sample.prompt}]

    caplog.set_level(logging.INFO, logger=generate_with_cuda_agent.logger.name)
    generate_with_cuda_agent._log_rollout_info(
        sample,
        messages=messages,
        turn_logs=[
            {
                "turn_idx": 0,
                "model_time": 0.25,
                "env_time": 0.75,
                "prompt": sample.prompt,
                "response": VALID_CUDA_AGENT_RESPONSE,
                "reward": 0.0,
                "env_result": {"env_state": {"status": "completed"}},
            }
        ],
        finish_reason="max_turns",
    )

    assert "[cuda_agent][first][rollout_info]" in caplog.text
    assert "[turn 0]" in caplog.text
    assert "[turn 0] prompt:" in caplog.text
    assert "first prompt" in caplog.text
    assert "[messages]:" in caplog.text
    assert '"role": "user"' in caplog.text


@pytest.mark.unit
def test_rollout_stats_only_omits_messages_and_turn_text(monkeypatch, caplog):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS, "log_multi_turn_info", True)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS, "log_rollout_stats_only", True)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS, "log_first_rollout", False)
    sample = Sample(prompt="hidden prompt", metadata={"uuid": "stats-only"})
    env_result = {
        "env_state": {"status": "completed", "error_message": "output mismatch"},
        "env_extra_info": {
            "precheck": "passed",
            "detail_env_time": {"compile_time": 0.1},
            "kernel_perf_cv": 0.02,
            "num_coverage": 0.8,
        },
    }
    sample.metadata["env_result"] = env_result
    original_sample = deepcopy(sample.to_dict())

    caplog.set_level(logging.INFO, logger=generate_with_cuda_agent.logger.name)
    generate_with_cuda_agent._log_rollout_info(
        sample,
        messages=[{"role": "user", "content": sample.prompt}],
        turn_logs=[
            {
                "turn_idx": 0,
                "task_id": "stats-only-task",
                "model_time": 0.25,
                "env_time": 0.75,
                "prompt": sample.prompt,
                "response": VALID_CUDA_AGENT_RESPONSE,
                "reward": 0.0,
                "env_result": env_result,
            }
        ],
        finish_reason="max_turns",
        should_log=True,
    )

    assert "[turn 0] task_id=stats-only-task" in caplog.text
    assert "[turn 0] env_feedback:" in caplog.text
    assert '"status": "completed"' in caplog.text
    feedback_records = [
        record.getMessage() for record in caplog.records if "[turn 0] env_feedback:" in record.getMessage()
    ]
    assert len(feedback_records) == 1
    assert json.loads(feedback_records[0].split("env_feedback:\n", 1)[1]) == env_result["env_state"]
    assert "env_extra_info" not in caplog.text
    assert "num_coverage" not in caplog.text
    assert "precheck=passed" in caplog.text
    assert "compile_time" in caplog.text
    assert "kernel_perf_cv" in caplog.text
    assert sample.to_dict() == original_sample
    assert "[prompt]:" not in caplog.text
    assert "[turn 0] user_content:" not in caplog.text
    assert "response_content" not in caplog.text


@pytest.mark.unit
def test_normalize_env_feedback_strips_compile_worker_routing_metadata():
    normalized, _ = normalize_env_feedback(
        {
            "compiled": False,
            "correctness": False,
            "speedup": 0.0,
            "error_code": "COMPILATION_ERROR",
            "error_message": "Kernel compilation failed: nvcc error",
            "metadata": {
                "cpu_worker_id": "node-a_cpu_0",
                "compile_node_id": "node-a",
                "compile_hostname": "host-a",
                "compilation_error_detail": "other",
            },
        }
    )

    assert normalized["metadata"] == {"compilation_error_detail": "other"}


@pytest.mark.unit
@pytest.mark.parametrize(
    "error_message,metadata_error,removed",
    [
        ("CUDA fault", "CUDA fault", True),
        ("Task processing failed: CUDA fault\nadditional detail", "CUDA fault", True),
        ("CUDA fault", "CUDA fault\nadditional detail", False),
        ("CUDA fault", "different error", False),
        (None, "CUDA fault", False),
        ("", "CUDA fault", False),
        ("CUDA fault", "", False),
        ("CUDA fault", None, False),
        ("CUDA fault", {"detail": "CUDA fault"}, False),
    ],
)
def test_strip_env_feedback_drops_only_contained_metadata_error(error_message, metadata_error, removed):
    raw = {"error_message": error_message, "metadata": {"error": metadata_error, "keep": "detail"}}

    cleaned = kernel_agent_utils._strip_env_feedback_fields(raw)

    assert cleaned["error_message"] == error_message
    assert cleaned["metadata"]["keep"] == "detail"
    assert ("error" not in cleaned["metadata"]) is removed
    if not removed:
        assert cleaned["metadata"]["error"] == metadata_error
    assert raw["metadata"] == {"error": metadata_error, "keep": "detail"}


@pytest.mark.unit
@pytest.mark.parametrize("already_in_message", [False, True])
def test_normalize_env_feedback_preserves_error_detail_once(already_in_message):
    detail = (
        "Task processing failed: CudaFinalSyncError: CUDA final synchronize failed: "
        "CUDA error: an illegal memory access was encountered"
    )
    raw = {
        "status": "failed",
        "compiled": True,
        "correctness": False,
        "error_code": "RUNTIME_ERROR",
        "error_message": detail if already_in_message else "Kernel execution failed",
        "speedup": 0.0,
        "metadata": {"error": detail},
    }

    normalized, _ = normalize_env_feedback(raw)

    assert normalized["error"] == "RUNTIME_ERROR"
    assert normalized["error_message"].count(detail) == 1
    if not already_in_message:
        assert normalized["error_message"].startswith("Kernel execution failed")
    assert "error" not in normalized["metadata"]
    assert raw["metadata"]["error"] == detail


@pytest.mark.unit
def test_log_multi_turn_info_uses_config_and_defaults_to_true(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS, "log_multi_turn_info", False)
    assert generate_with_cuda_agent._log_multi_turn_info() is False

    monkeypatch.setitem(CUDA_AGENT_CONFIGS, "log_multi_turn_info", True)
    assert generate_with_cuda_agent._log_multi_turn_info() is True

    monkeypatch.delitem(CUDA_AGENT_CONFIGS, "log_multi_turn_info")
    assert generate_with_cuda_agent._log_multi_turn_info() is True


@pytest.mark.unit
def test_log_first_rollout_uses_config_and_logs_once(monkeypatch):
    monkeypatch.setitem(CUDA_AGENT_CONFIGS, "log_first_rollout", True)
    monkeypatch.setattr(generate_with_cuda_agent, "_LOGGED_FIRST_ROLLOUT", False)
    assert generate_with_cuda_agent._log_first_rollout() is True
    assert generate_with_cuda_agent._log_first_rollout() is False

    monkeypatch.setitem(CUDA_AGENT_CONFIGS, "log_first_rollout", False)
    monkeypatch.setattr(generate_with_cuda_agent, "_LOGGED_FIRST_ROLLOUT", False)
    assert generate_with_cuda_agent._log_first_rollout() is False


@pytest.mark.unit
def test_cuda_agent_sampling_params_reserve_context_for_eagle():
    args = SimpleNamespace(
        rollout_max_context_len=16384,
        sglang_speculative_algorithm="EAGLE",
        sglang_speculative_num_draft_tokens=4,
    )
    sampling_params = {"max_new_tokens": 16384, "temperature": 1.0}

    adjusted = generate_with_cuda_agent._sampling_params_for_prompt_context(
        args,
        sampling_params,
        prompt_token_count=10000,
    )

    assert adjusted["max_new_tokens"] == 6380
    assert adjusted["temperature"] == 1.0
    assert sampling_params["max_new_tokens"] == 16384


@pytest.mark.integration
@pytest.mark.parametrize("case", REAL_KERNEL_EVAL_CASES)
@pytest.mark.skipif(
    os.environ.get("RUN_CUDA_KERNEL_EVAL_INTEGRATION") != "1",
    reason="Set RUN_CUDA_KERNEL_EVAL_INTEGRATION=1 to run the real KernelServer env test.",
)
def test_cuda_kernel_env_real_kernel_eval_server(request, monkeypatch, caplog, case):
    _skip_unselected_compiled_case(request, case, "expected_compiled")
    ray = pytest.importorskip("ray")

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=2)

    kernel_eval_url = os.environ.get("CUDA_KERNEL_EVAL_URL", "http://192.168.16.21:8003")
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["env"], "kernel_eval_url", kernel_eval_url)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["env"], "kernel_eval_max_retries", 1)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["env"], "kernel_eval_client_timeout", 300)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["env"], "kernel_eval_task_timeout", 120)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["env"], "kernel_eval_poll_interval", 1.0)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["env"], "kernel_eval_heartbeat_interval", 5.0)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["env"], "num_correct_trials", 1)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["env"], "num_perf_trials", 1)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS, "log_rollout_info_rate", 1.0)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS, "log_multi_turn_info", True)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS, "log_rollout_stats_only", False)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS, "max_feedback_chars", 8192)

    sample = Sample(
        prompt="Write a CUDA identity implementation.",
        label={"entry_point": "Model", "ground_truth": REFERENCE_IDENTITY_CODE},
        metadata={
            "uuid": case["uuid"],
            "source_row": "integration",
            "log_rollout_info": True,
        },
    )

    env_result = asyncio.run(
        generate_with_cuda_agent.cuda_kernel_env(
            SimpleNamespace(kernel_backend="cuda_agent", do_precheck=False),
            sample,
            case["response"],
            turn_idx=0,
        )
    )
    env_state = env_result["env_state"]
    format_feedback = _format_feedback_for_test(env_result)
    print(f"\n[cuda_agent][test][format_feedback][{case['uuid']}]\n{format_feedback}")

    caplog.set_level(logging.INFO, logger=generate_with_cuda_agent.logger.name)
    generate_with_cuda_agent._log_rollout_info(
        sample,
        prompt_text=f"{sample.prompt}\n{case['response']}\n{format_feedback}",
        turn_logs=[
            {
                "turn_idx": 0,
                "model_time": 0.0,
                "env_time": float(env_state.get("processing_time") or 0.0),
                "prompt_tokens": 16,
                "response_tokens": 32,
                "finish_type": "stop",
                "prompt": sample.prompt,
                "response": case["response"],
                "env_result": env_result,
                "format_feedback": format_feedback,
            }
        ],
        finish_reason="max_turns",
        should_log=True,
        is_slowest=True,
        total_request_time=float(env_state.get("processing_time") or 0.0),
    )

    assert env_state["status"] in {"completed", "failed", "timeout", "cancelled"}
    assert env_state.get("compiled") is case["expected_compiled"]
    assert "[cuda_agent][first][rollout_info]" in caplog.text
    assert case["uuid"] in caplog.text
    assert "format_feedback" in caplog.text
    assert "Server feedback (status/metrics/errors):" in caplog.text


def test_extract_detail_env_time_from_raw_env_state():
    raw_env_state = {
        "metadata": {
            "kg_kernel_backend_compile_s": 1.5,
            "kg_kernel_perf_warmup_s": 0.25,
            "kg_kernel_perf_measure_wall_s": 2.75,
            "kg_kernel_perf_profile_s": 0.5,
            "kg_reference_perf_warmup_s": 0.1,
            "kg_reference_perf_measure_wall_s": 0.9,
        }
    }

    detail_env_time = kernel_agent_utils._extract_detail_env_time(raw_env_state)

    assert detail_env_time == {
        "compile_time": 1.5,
        "kernel_runtime": 3.0,
        "profile_time": 0.5,
        "refer_runtime": 1.0,
    }


def test_cuda_kernel_env_returns_detail_env_time(monkeypatch):
    async def fake_run_kernel_eval(args, sample, payload, config):
        return {
            "env_state": {
                "status": "completed",
                "compiled": True,
                "correctness": True,
                "speedup": 1.0,
                "decoy_kernel": False,
                "metadata": {
                    "kg_kernel_backend_compile_s": 4.0,
                    "kg_kernel_perf_warmup_s": 0.5,
                    "kg_kernel_perf_measure_wall_s": 1.5,
                    "kg_kernel_perf_profile_s": 2.0,
                    "kg_kernel_perf_mean_ms": 10.0,
                    "kg_kernel_perf_std_ms": 2.5,
                    "kg_reference_perf_warmup_s": 0.25,
                    "kg_reference_perf_measure_wall_s": 0.75,
                    "kg_reference_perf_mean_ms": 20.0,
                    "kg_reference_perf_std_ms": 4.0,
                },
            }
        }

    monkeypatch.setattr(generate_with_cuda_agent, "run_kernel_eval", fake_run_kernel_eval)
    monkeypatch.setattr(generate_with_cuda_agent, "next_kernel_task_id", lambda: "parallel_task_detail_time")

    args = SimpleNamespace(
        kernel_backend="cuda",
        reference_backend="torch",
        enable_response_precheck=False,
    )
    sample = Sample(
        prompt="prompt", label={"ground_truth": REFERENCE_IDENTITY_CODE, "entry_point": "Model"}, metadata={}
    )

    env_result = asyncio.run(generate_with_cuda_agent.cuda_kernel_env(args, sample, VALID_CUDA_AGENT_RESPONSE, 0))

    env_extra_info = env_result["env_extra_info"]
    assert env_extra_info["detail_env_time"] == {
        "compile_time": 4.0,
        "kernel_runtime": 2.0,
        "profile_time": 2.0,
        "refer_runtime": 1.0,
    }
    assert env_extra_info["kernel_perf_cv"] == pytest.approx(0.25)
    assert env_extra_info["refer_perf_cv"] == pytest.approx(0.2)


def test_cuda_kernel_env_env_precheck_error_overrides_default_passed(monkeypatch):
    async def fake_run_kernel_eval(args, sample, payload, config):
        return {
            "env_state": {
                "status": "failed",
                "compiled": False,
                "correctness": False,
                "speedup": 0.0,
                "decoy_kernel": False,
                "error_message": "Precheck failed: static check failed: framework_compute",
                "metadata": {"compilation_error": "Precheck failed: static check failed: framework_compute"},
            }
        }

    monkeypatch.setattr(generate_with_cuda_agent, "run_kernel_eval", fake_run_kernel_eval)
    monkeypatch.setattr(generate_with_cuda_agent, "next_kernel_task_id", lambda: "parallel_task_env_precheck")
    monkeypatch.setattr(generate_with_cuda_agent, "precheck_response", lambda *args, **kwargs: (True, None))

    args = SimpleNamespace(
        kernel_backend="cuda",
        reference_backend="torch",
        do_precheck=True,
    )
    sample = Sample(
        prompt="prompt", label={"ground_truth": REFERENCE_IDENTITY_CODE, "entry_point": "Model"}, metadata={}
    )

    env_result = asyncio.run(generate_with_cuda_agent.cuda_kernel_env(args, sample, VALID_CUDA_AGENT_RESPONSE, 0))

    assert env_result["env_state"]["error"] == PRECHECK_ERROR
    assert env_result["env_state"]["precheck"] == "failed"
    assert env_result["env_extra_info"]["precheck"] == "failed"


def test_normalize_env_feedback_extra_info_defaults_missing_decoy_kernel():
    _env_state, env_extra_info = normalize_env_feedback(
        {
            "status": "failed",
            "compiled": False,
            "correctness": False,
            "speedup": 0.0,
            "metadata": {},
        }
    )

    assert env_extra_info["decoy_kernel"] is False


@pytest.mark.unit
def test_normalize_env_feedback_compacts_runtime_sanitizer_and_accounts_for_wall_time():
    runtime_error = "head:" + "H" * 320 + " middle " + "T" * 320 + ":tail"
    raw_output_tail = "output-head:" + "O" * 320 + " middle " + "Z" * 320 + ":output-tail"
    raw_env_state = {
        "status": "failed",
        "compiled": True,
        "correctness": False,
        "speedup": 0.0,
        "error_code": "RUNTIME_ERROR",
        "error_message": "Runtime Sanitizer detected an unsafe CUDA kernel",
        "metadata": {
            "runtime_error": runtime_error,
            "runtime_sanitizer_status": "issues_found",
            "runtime_sanitizer_issue_count": 1,
        },
        "runtime_sanitizer": {
            "status": "issues_found",
            "measurement_complete": True,
            "primary_check": "memcheck",
            "wall_time_s": 3.45678,
            "replayed_input_seed": 123456,
            "run_all_checks": False,
            "check_results": [
                {
                    "check": "memcheck",
                    "status": "issues_found",
                    "detected_issue_count": 1,
                    "raw_output_tail": raw_output_tail,
                    "issues": [
                        {
                            "hazard_type": "invalid_global_write",
                            "message": "Invalid __global__ write of size 4 bytes",
                            "raw_excerpt": "duplicated issue output",
                            "representative_occurrences": [{"thread": {"x": 232}}],
                        }
                    ],
                },
                {
                    "check": "synccheck",
                    "status": "clean",
                    "detected_issue_count": 0,
                    "raw_output_tail": "ERROR SUMMARY: 0 errors",
                    "issues": [],
                },
            ],
        },
    }

    normalized, env_extra_info = normalize_env_feedback(raw_env_state)

    sanitizer = normalized["runtime_sanitizer"]
    assert normalized["error"] == "RUNTIME_ERROR"
    assert "error_code" not in normalized
    assert set(sanitizer) == {"status", "measurement_complete", "primary_check", "check_results"}
    assert set(sanitizer["check_results"][0]) == {
        "check",
        "status",
        "detected_issue_count",
        "issues",
        "raw_output_tail",
    }
    assert len(sanitizer["check_results"]) == 1
    assert sanitizer["check_results"][0]["check"] == "memcheck"
    assert sanitizer["check_results"][0]["raw_output_tail"] == (
        f"{raw_output_tail[:250]}...(truncated)...{raw_output_tail[-250:]}"
    )
    assert "raw_excerpt" not in sanitizer["check_results"][0]["issues"][0]
    assert "representative_occurrences" not in sanitizer["check_results"][0]["issues"][0]
    assert normalized["error_message"] == "Runtime Sanitizer detected an unsafe CUDA kernel"
    assert normalized["metadata"]["runtime_error"] == f"{runtime_error[:250]}...(truncated)...{runtime_error[-250:]}"
    assert "runtime_sanitizer_status" not in normalized["metadata"]
    assert "runtime_sanitizer_issue_count" not in normalized["metadata"]
    assert env_extra_info["detail_env_time"]["runtime_sanitizer_time_s"] == pytest.approx(3.4568)
    assert raw_env_state["runtime_sanitizer"]["replayed_input_seed"] == 123456
    assert len(raw_env_state["runtime_sanitizer"]["check_results"]) == 2


@pytest.mark.unit
def test_normalize_env_feedback_removes_check_results_when_all_sanitizer_checks_are_clean():
    raw_env_state = {
        "status": "failed",
        "compiled": True,
        "correctness": False,
        "speedup": 0.0,
        "error_code": "RUNTIME_ERROR",
        "error_message": "Kernel execution failed",
        "metadata": {"runtime_error": "TypeError: unsupported operand type"},
        "runtime_sanitizer": {
            "status": "clean",
            "measurement_complete": True,
            "primary_check": "memcheck",
            "check_results": [
                {
                    "check": "memcheck",
                    "status": "clean",
                    "detected_issue_count": 0,
                    "raw_output_tail": "ERROR SUMMARY: 0 errors",
                    "issues": [],
                },
                {
                    "check": "synccheck",
                    "status": "clean",
                    "detected_issue_count": 0,
                    "raw_output_tail": "ERROR SUMMARY: 0 errors",
                    "issues": [],
                },
            ],
        },
    }

    normalized, _ = normalize_env_feedback(raw_env_state)

    assert normalized["runtime_sanitizer"] == {
        "status": "clean",
        "measurement_complete": True,
        "primary_check": "memcheck",
    }
    assert "runtime_error" not in normalized["metadata"]
    assert "TypeError: unsupported operand type" in normalized["error_message"]
    assert len(raw_env_state["runtime_sanitizer"]["check_results"]) == 2


@pytest.mark.unit
def test_normalize_env_feedback_keeps_runtime_error_without_structured_sanitizer_issue():
    runtime_error = "Traceback: original correctness failure"
    normalized, _ = normalize_env_feedback(
        {
            "status": "failed",
            "compiled": True,
            "correctness": False,
            "speedup": 0.0,
            "error_code": "RUNTIME_ERROR",
            "error_message": "Kernel execution failed",
            "metadata": {"runtime_error": runtime_error},
            "runtime_sanitizer": {"status": "skipped", "reason": "disabled", "check_results": []},
        }
    )

    assert normalized["error"] == "RUNTIME_ERROR"
    assert "error_code" not in normalized
    assert runtime_error in normalized["error_message"]
    assert "runtime_error" not in normalized["metadata"]


@pytest.mark.unit
def test_normalize_env_feedback_deduplicates_issue_and_strips_correctness_progress():
    correctness_issue = (
        "Numerical output mismatch under input perturbation scale_up: " "max_difference=11.4672, avg_difference=1.5962"
    )
    progress_fields = {
        "correctness_inputs_generated_on_gpu": True,
        "correctness_requested_trials": 5,
        "correctness_effective_trials": 5,
        "correctness_input_perturbation_trials": [{"trial": 1, "name": "scale_up"}],
        "correctness_reference_skipped_perturbations": [],
        "correctness_candidate_forward_completed": True,
        "correctness_candidate_forward_completed_trials": [0, 1, 2, 3, 4],
    }
    raw_error_message = f"Kernel produced incorrect results: {correctness_issue}"
    raw_env_state = {
        "status": "completed",
        "compiled": True,
        "correctness": False,
        "speedup": 0.0,
        "error_code": CORRECTNESS_ERROR,
        "error_message": raw_error_message,
        "metadata": {
            **progress_fields,
            "correctness_issue": correctness_issue,
            "correctness_trials": "(5 / 5)",
        },
    }

    normalized, _ = normalize_env_feedback(raw_env_state)

    assert normalized["error_message"] == raw_error_message
    assert normalized["metadata"]["correctness_candidate_forward_completed"] is True
    removed_progress_fields = progress_fields.keys() - {"correctness_candidate_forward_completed"}
    assert removed_progress_fields.isdisjoint(normalized["metadata"])
    assert "correctness_issue" not in normalized["metadata"]
    assert normalized["metadata"]["correctness_trials"] == "(5 / 5)"
    assert raw_env_state["metadata"]["correctness_issue"] == correctness_issue


def test_kernel_agent_metrics_reuse_kernel_time_for_detail_env_time():
    from slime.observability.rollout_metrics import _compute_kernel_agent_metrics

    samples = [
        Sample(
            prompt="prompt",
            metadata={
                "env_time": 1.0,
                "env_extra_info": {
                    "correctness": True,
                    "compilation": True,
                    "speedup": 1.0,
                    "decoy_kernel": False,
                    "precheck": "passed",
                    "detail_env_time": {
                        "compile_time": 1.0,
                        "kernel_runtime": 10.0,
                        "profile_time": 100.0,
                        "runtime_sanitizer_time_s": 5.0,
                        "refer_runtime": 1000.0,
                    },
                },
            },
        ),
        Sample(
            prompt="prompt",
            metadata={
                "env_time": 3.0,
                "env_extra_info": {
                    "correctness": True,
                    "compilation": True,
                    "speedup": 1.0,
                    "decoy_kernel": False,
                    "precheck": "passed",
                    "detail_env_time": {
                        "compile_time": 3.0,
                        "kernel_runtime": 30.0,
                        "profile_time": 300.0,
                        "runtime_sanitizer_time_s": 7.0,
                        "refer_runtime": 3000.0,
                    },
                },
            },
        ),
        Sample(
            prompt="prompt",
            metadata={
                "conditional_truncation_masked": True,
                "env_extra_info": {
                    "correctness": False,
                    "compilation": True,
                    "speedup": 0.0,
                    "decoy_kernel": False,
                    "precheck": "passed",
                },
            },
        ),
    ]

    metrics = _compute_kernel_agent_metrics(samples)

    assert metrics["kernel/time/detail_env_time/compile_time/count"] == 2
    assert metrics["kernel/time/detail_env_time/compile_time/p50"] == pytest.approx(2.0)
    assert metrics["kernel/time/detail_env_time/kernel_runtime/mean"] == pytest.approx(20.0)
    assert metrics["kernel/time/detail_env_time/profile_time/sum"] == pytest.approx(400.0)
    assert metrics["kernel/time/detail_env_time/runtime_sanitizer_time_s/mean"] == pytest.approx(6.0)
    assert metrics["kernel/time/detail_env_time/runtime_sanitizer_time_s/sum"] == pytest.approx(12.0)
    assert metrics["kernel/time/detail_env_time/refer_runtime/max"] == pytest.approx(3000.0)
    assert metrics["sample_mask/conditional_truncation_masked_fraction"] == pytest.approx(1 / 3)


class _LocalEvalRPC:
    """Schedule the real actor method locally without starting a Ray cluster."""

    def __init__(self, method):
        self.method = method

    def remote(self, *args, **kwargs):
        async def invoke():
            result = self.method(*args, **kwargs)
            return await result if inspect.isawaitable(result) else result

        return asyncio.create_task(invoke())


@pytest.fixture
def local_eval_worker(monkeypatch):
    limiter_cls = kernel_response._TokenBucketWorker.__ray_metadata__.modified_class
    limiter = limiter_cls(1)
    worker_cls = kernel_response._HybridHttpWorker.__ray_metadata__.modified_class
    worker = worker_cls.__new__(worker_cls)
    worker.server_url = "http://eval.test"
    worker.default_timeout = 1
    worker.acquire_timeout = 1
    worker._lock = threading.Lock()
    worker._running = {}
    worker._invalidated = {}
    worker._task_status = {}
    # HTTP behavior tests route both independent production clients through the
    # same mock transport. Pool separation is checked separately below.
    worker._control_client = SimpleNamespace(get=lambda *args, **kwargs: worker._client.get(*args, **kwargs))
    worker._rate_limit_worker = SimpleNamespace(
        **{name: _LocalEvalRPC(getattr(limiter, name)) for name in ("acquire", "release", "get_current_count")}
    )
    cancelled = []

    def cancel(ref, **kwargs):
        assert kwargs == {"force": False, "recursive": False}
        cancelled.append(ref)
        ref.cancel()

    monkeypatch.setattr(kernel_response.ray, "cancel", cancel)

    async def delete(*args):
        return True

    monkeypatch.setattr(kernel_response, "_delete_server_task", delete)
    return worker, limiter, cancelled


@pytest.mark.unit
def test_kernel_eval_leases_are_idempotent_expiring_and_fence_late_acquire(local_eval_worker, monkeypatch):
    _, limiter, _ = local_eval_worker
    now = [100.0]
    monkeypatch.setattr(kernel_response.time, "time", lambda: now[0])
    assert limiter.acquire("first", 110)
    assert limiter.acquire("first", 110)
    assert not limiter.acquire("second", 110)
    limiter.release("first", 110)
    limiter.release("first", 110)
    assert limiter.get_current_count() == 0
    assert not limiter.acquire("first", 110)
    # Cancellation may reach the limiter before the acquire RPC.
    limiter.release("late", 110)
    assert not limiter.acquire("late", 110)
    assert limiter.acquire("second", 110)
    assert not limiter.acquire("third", 110)
    now[0] = 111
    assert limiter.get_current_count() == 0
    assert not limiter.acquire("expired", 110)
    assert limiter.acquire("third", 120)


@pytest.mark.unit
@pytest.mark.parametrize("invalidated", [False, True])
def test_kernel_eval_queued_calls_do_not_post_after_deadline_or_invalidation(local_eval_worker, invalidated):
    worker, limiter, _ = local_eval_worker

    async def scenario():
        deadline = time.time() + (10 if invalidated else -1)
        if invalidated:
            assert await worker.invalidate("queued", deadline) is False
        # Deliberately no HTTP client: neither path may reach a POST.
        result = await worker.submit_and_poll({"task_id": "queued"}, 10, 1, 0.01, deadline=deadline)
        assert result["status"] == ("cancelled" if invalidated else "timeout")
        assert limiter.get_current_count() == 0
        assert not worker._running

    asyncio.run(scenario())


@pytest.mark.unit
def test_kernel_eval_acquire_timeout_does_not_post_or_leak_token(local_eval_worker):
    worker, limiter, _ = local_eval_worker

    async def scenario():
        assert limiter.acquire("occupied", time.time() + 10)
        worker.acquire_timeout = 0.02
        result = await worker.submit_and_poll({"task_id": "waiting"}, 1, 1, 0.01)
        assert result["status"] == "timeout"
        assert limiter.get_current_count() == 1
        limiter.release("occupied", time.time() + 10)
        assert limiter.get_current_count() == 0

    asyncio.run(scenario())


@pytest.mark.unit
@pytest.mark.parametrize("status", ["completed", "failed", "timeout", "cancelled"])
def test_kernel_eval_preserves_terminal_result_diagnostics(local_eval_worker, status):
    worker, limiter, _ = local_eval_worker
    urls = []

    async def handler(request):
        urls.append(request.url.path)
        if request.url.path == "/evaluate":
            return httpx.Response(409)
        if request.url.path.startswith("/status/"):
            return httpx.Response(200, json={"status": status})
        return httpx.Response(200, json={"compiled": True, "correctness": False, "metadata": {"compile_s": 2.0}})

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as worker._client:
            result = await worker.submit_and_poll({"task_id": "diagnostics"}, 1, 1, 0.01)
        assert result["status"] == status
        assert result["compiled"] is True
        assert result["metadata"]["compile_s"] == 2.0
        assert urls == ["/evaluate", "/status/diagnostics", "/results/diagnostics"]
        assert limiter.get_current_count() == 0

    asyncio.run(scenario())


@pytest.mark.unit
def test_kernel_eval_invalidation_after_token_grant_prevents_post(local_eval_worker):
    worker, limiter, _ = local_eval_worker

    def acquire(lease_id, expiry):
        granted = limiter.acquire(lease_id, expiry)
        with worker._lock:
            worker._invalidated["after-grant"] = expiry
        return granted

    worker._rate_limit_worker.acquire = _LocalEvalRPC(acquire)

    async def scenario():
        # No HTTP client: a cancellation observed after the grant must stop here.
        with pytest.raises(asyncio.CancelledError):
            await worker.submit_and_poll({"task_id": "after-grant"}, 1, 1, 0.01)
        assert limiter.get_current_count() == 0
        assert not worker._running

    asyncio.run(scenario())


@pytest.mark.unit
@pytest.mark.parametrize("heartbeat_interval", [0, 0.001])
def test_run_kernel_eval_deadline_includes_remote_queue(local_eval_worker, monkeypatch, heartbeat_interval):
    worker, _, cancelled = local_eval_worker
    submissions = []
    invalidations = []

    async def queued(payload, **kwargs):
        submissions.append(kwargs)
        await asyncio.Event().wait()

    async def invalidate(task_id, deadline):
        invalidations.append((task_id, deadline))
        return False  # never submitted HTTP

    async def blocked_query(*args):
        await asyncio.Event().wait()

    handle = SimpleNamespace(
        submit_and_poll=_LocalEvalRPC(queued),
        invalidate=_LocalEvalRPC(invalidate),
        get_task_status=_LocalEvalRPC(blocked_query),
    )
    monkeypatch.setattr(kernel_response, "_get_kernel_eval_worker", lambda args, config: handle)

    async def delete(*args):
        return True

    monkeypatch.setattr(kernel_response, "_delete_server_task", delete)
    config = {
        "kernel_env_url": worker.server_url,
        "kernel_eval_client_timeout": 0.03,
        "kernel_eval_cancel_timeout": 0.1,
        "kernel_eval_max_retries": 1,
        "kernel_eval_poll_interval": 0.01,
        "kernel_eval_heartbeat_interval": heartbeat_interval,
        "kernel_eval_rate_limit": 1,
    }

    async def scenario():
        before = time.time()
        result = await asyncio.wait_for(
            kernel_response.run_kernel_eval(SimpleNamespace(), Sample(), {"task_id": "queued"}, config), 0.5
        )
        assert result["env_state"]["status"] == "timeout"
        assert before <= submissions[0]["deadline"] <= before + 0.1
        assert invalidations == [("queued", submissions[0]["deadline"])]
        assert cancelled
        assert not kernel_response._ACTIVE_EVALS

    asyncio.run(scenario())


@pytest.mark.unit
@pytest.mark.parametrize("failure", ["http_error", "retry", "poll_timeout"])
def test_kernel_eval_errors_and_infinite_retries_respect_deadline_and_release(local_eval_worker, monkeypatch, failure):
    worker, limiter, _ = local_eval_worker
    deletes = []
    releases = []
    original_release = limiter.release

    def release(*args):
        releases.append(args[0])
        original_release(*args)

    worker._rate_limit_worker.release = _LocalEvalRPC(release)

    async def delete(url, task_id, timeout):
        deletes.append(task_id)
        return True

    monkeypatch.setattr(kernel_response, "_delete_server_task", delete)

    async def handler(request):
        if request.method == "POST":
            return httpx.Response({"http_error": 500, "retry": 503, "poll_timeout": 200}[failure])
        await asyncio.Event().wait()

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as worker._client:
            result = await asyncio.wait_for(worker.submit_and_poll({"task_id": "failed"}, 0.05, -1, 0.01), timeout=0.5)
        assert result["status"] == ("failed" if failure == "http_error" else "timeout")
        assert len(releases) == len(set(releases)) == 1
        assert limiter.get_current_count() == 0
        assert deletes == ["failed"]

    asyncio.run(scenario())


@pytest.mark.unit
def test_kernel_eval_ready_result_is_not_blocked_by_heartbeat(local_eval_worker, monkeypatch):
    worker, _, cancelled = local_eval_worker

    async def forbidden_thread(*args, **kwargs):
        pytest.fail("Kernel eval must not use the default thread pool")

    monkeypatch.setattr(asyncio, "to_thread", forbidden_thread)

    async def scenario():
        querying = asyncio.Event()

        async def blocked_query(*args):
            querying.set()
            await asyncio.Event().wait()

        handle = SimpleNamespace(get_task_status=_LocalEvalRPC(blocked_query))
        result = asyncio.get_running_loop().create_future()
        waiting = asyncio.create_task(
            kernel_response._wait_kernel_eval_result(result, handle, {"task_id": "heartbeat"}, 0.001, 1, 0.5)
        )
        await asyncio.wait_for(querying.wait(), 0.2)
        result.set_result({"status": "completed", "compiled": True})
        returned = await asyncio.wait_for(waiting, 0.2)
        assert returned["compiled"] is True
        assert cancelled  # outstanding heartbeat RPC was cancelled as well

    asyncio.run(scenario())


@pytest.mark.unit
@pytest.mark.parametrize("exit_mode", ["cancel", "timeout", "exception"])
def test_run_kernel_eval_cleans_ray_and_http_on_every_abnormal_exit(local_eval_worker, monkeypatch, exit_mode):
    worker, limiter, cancelled = local_eval_worker
    deletes = []
    handle = SimpleNamespace(
        submit_and_poll=_LocalEvalRPC(worker.submit_and_poll), invalidate=_LocalEvalRPC(worker.invalidate)
    )
    monkeypatch.setattr(kernel_response, "_get_kernel_eval_worker", lambda args, config: handle)

    async def delete(url, task_id, timeout):
        deletes.append(task_id)
        return True

    monkeypatch.setattr(kernel_response, "_delete_server_task", delete)
    config = {
        "kernel_env_url": worker.server_url,
        "kernel_eval_client_timeout": 0.05 if exit_mode == "timeout" else 10,
        "kernel_eval_cancel_timeout": 0.2,
        "kernel_eval_max_retries": 1,
        "kernel_eval_poll_interval": 0.01,
        "kernel_eval_heartbeat_interval": 0,
        "kernel_eval_rate_limit": 1,
    }

    async def scenario():
        submitted = asyncio.Event()

        async def handler(request):
            submitted.set()
            await asyncio.Event().wait()

        if exit_mode == "exception":

            async def failed_wait(*args, **kwargs):
                await submitted.wait()
                raise RuntimeError("failed result wait")

            monkeypatch.setattr(kernel_response, "_wait_kernel_eval_result", failed_wait)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as worker._client:
            task = asyncio.create_task(
                kernel_response.run_kernel_eval(SimpleNamespace(), Sample(), {"task_id": "running"}, config)
            )
            await asyncio.wait_for(submitted.wait(), 0.5)
            if exit_mode == "cancel":
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            elif exit_mode == "exception":
                with pytest.raises(RuntimeError, match="failed result wait"):
                    await task
            else:
                assert (await asyncio.wait_for(task, 0.5))["env_state"]["status"] == "timeout"
            # A second cancellation may leave shielded, bounded cleanup finishing.
            async with asyncio.timeout(0.5):
                while kernel_response._CLEANUP_TASKS or worker._running:
                    await asyncio.sleep(0.001)
        assert cancelled
        assert deletes and set(deletes) == {"running"}
        assert "running" in worker._invalidated
        assert not kernel_response._ACTIVE_EVALS
        assert limiter.get_current_count() == 0

    asyncio.run(scenario())


@pytest.mark.unit
def test_kernel_eval_delete_retries_404_and_has_total_timeout(monkeypatch):
    client_class = httpx.AsyncClient
    attempts = []

    async def handler(request):
        attempts.append(request.method)
        return httpx.Response(404 if len(attempts) == 1 else 200)

    monkeypatch.setattr(
        kernel_response.httpx,
        "AsyncClient",
        lambda **kwargs: client_class(transport=httpx.MockTransport(handler), **kwargs),
    )
    assert asyncio.run(kernel_response._delete_server_task("http://eval.test", "late", 0.5))
    assert attempts == ["DELETE", "DELETE"]

    async def missing(request):
        return httpx.Response(404)

    monkeypatch.setattr(
        kernel_response.httpx,
        "AsyncClient",
        lambda **kwargs: client_class(transport=httpx.MockTransport(missing), **kwargs),
    )
    assert asyncio.run(kernel_response._delete_server_task("http://eval.test", "missing", 0.02)) is False


@pytest.mark.unit
@pytest.mark.parametrize("ray_cancel_raises", [False, True])
def test_cancel_queued_call_always_installs_server_tombstone(
    local_eval_worker, monkeypatch, caplog, ray_cancel_raises
):
    worker, _, _ = local_eval_worker
    caplog.set_level(logging.INFO)
    deletes, invalidations = [], []

    async def invalidate(task_id, deadline):
        invalidations.append(task_id)
        return False  # HTTP has not started; this must not skip DELETE.

    async def delete(url, task_id, timeout):
        deletes.append(task_id)
        return True

    def cancel(*args, **kwargs):
        if ray_cancel_raises:
            raise RuntimeError("Ray cancellation unavailable")

    monkeypatch.setattr(kernel_response.ray, "cancel", cancel)
    monkeypatch.setattr(kernel_response, "_delete_server_task", delete)

    async def scenario():
        handle = SimpleNamespace(invalidate=_LocalEvalRPC(invalidate))
        assert await kernel_response._cancel_eval_call(
            handle, object(), worker.server_url, "queued", time.time() + 10, 0.1
        )
        assert invalidations == deletes == ["queued"]

    asyncio.run(scenario())
    assert "event=cancel_acknowledged scope=invalidation" in caplog.text


@pytest.mark.unit
def test_cancel_server_fence_does_not_wait_for_congested_ray_control(local_eval_worker, monkeypatch):
    worker, _, _ = local_eval_worker
    monkeypatch.setattr(kernel_response, "_CONTROL_TIMEOUT_S", 0.02)
    monkeypatch.setattr(kernel_response.ray, "cancel", lambda *args, **kwargs: None)

    async def scenario():
        release, deleted, invalidated = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def invalidate(*args):
            await release.wait()
            invalidated.set()
            return False

        async def delete(*args):
            deleted.set()
            return True

        monkeypatch.setattr(kernel_response, "_delete_server_task", delete)
        handle = SimpleNamespace(invalidate=_LocalEvalRPC(invalidate))
        cleanup = asyncio.create_task(
            kernel_response._cancel_eval_call(handle, object(), worker.server_url, "queued", time.time() + 1, 0.1)
        )
        await asyncio.wait_for(deleted.wait(), 0.1)
        assert not invalidated.is_set()
        assert await cleanup is False  # Server ACK alone is not a Ray invalidation ACK.
        release.set()
        await asyncio.wait_for(invalidated.wait(), 0.1)  # Timed-out invalidation was not discarded.

    asyncio.run(scenario())


@pytest.mark.unit
def test_expired_trajectory_never_creates_ray_eval(local_eval_worker, monkeypatch):
    worker, _, _ = local_eval_worker

    def lookup(*args):
        pytest.fail("Expired trajectories must not create Ray work")

    monkeypatch.setattr(kernel_response, "_get_kernel_eval_worker", lookup)

    async def scenario():
        token = kernel_response.KERNEL_EVAL_DEADLINE.set(time.time() - 1)
        try:
            result = await kernel_response.run_kernel_eval(
                SimpleNamespace(),
                Sample(),
                {"task_id": "expired"},
                {
                    "kernel_env_url": worker.server_url,
                    "kernel_eval_client_timeout": 100,
                },
            )
        finally:
            kernel_response.KERNEL_EVAL_DEADLINE.reset(token)
        assert result["env_state"]["status"] == "timeout"
        assert not kernel_response._ACTIVE_EVALS

    asyncio.run(scenario())


@pytest.mark.unit
@pytest.mark.parametrize("role", ["kernel", "verify"])
def test_generate_deadline_is_inherited_and_restored(monkeypatch, role):
    deadlines = []

    async def impl(*args):
        deadlines.append(kernel_response.KERNEL_EVAL_DEADLINE.get())
        return []

    monkeypatch.setattr(generate_with_cuda_agent, "_generate_kernel_impl", impl)
    monkeypatch.setattr(generate_with_cuda_agent, "_generate_with_verify_impl", impl)
    monkeypatch.setattr(generate_with_cuda_agent, "KERNEL_AGENT_GENERATE_GUARD_SEC", 100)

    async def scenario():
        deadline = time.time() + 1
        token = kernel_response.KERNEL_EVAL_DEADLINE.set(deadline)
        try:
            await generate_with_cuda_agent.generate(SimpleNamespace(), Sample(metadata={"role": role}), {})
            assert deadlines == [deadline]
            assert kernel_response.KERNEL_EVAL_DEADLINE.get() == deadline
        finally:
            kernel_response.KERNEL_EVAL_DEADLINE.reset(token)
        assert kernel_response.KERNEL_EVAL_DEADLINE.get() is None

    asyncio.run(scenario())


@pytest.mark.unit
def test_cancel_http_channel_is_bounded_and_logs_404_as_unconfirmed(monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    original = httpx.AsyncClient
    active = maximum = 0

    async def handler(request):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        try:
            await asyncio.sleep(0.01)
            return httpx.Response(404 if request.url.path.endswith("missing") else 200)
        finally:
            active -= 1

    monkeypatch.setattr(
        kernel_response.httpx,
        "AsyncClient",
        lambda **kwargs: original(transport=httpx.MockTransport(handler), **kwargs),
    )

    async def scenario():
        assert all(
            await asyncio.gather(
                *(kernel_response._delete_server_task("http://eval.test", f"t{i}", 1) for i in range(20))
            )
        )
        assert maximum <= 8
        assert not await kernel_response._delete_server_task("http://eval.test", "missing", 0.03)

    asyncio.run(scenario())
    assert "event=cancel_not_found scope=server task_id=missing" in caplog.text
    assert "event=cancel_acknowledged scope=server task_id=missing" not in caplog.text
    assert "event=cancel_failed scope=server task_id=missing" in caplog.text


@pytest.mark.unit
def test_http_poll_pool_and_ray_cancellation_slots_are_independent(monkeypatch):
    monkeypatch.setattr(
        kernel_response._TokenBucketWorker, "options", lambda **kwargs: SimpleNamespace(remote=lambda *args: None)
    )
    worker_cls = kernel_response._HybridHttpWorker.__ray_metadata__.modified_class

    async def scenario():
        worker = worker_cls("http://eval.test", 1, 10, 1)
        try:
            assert worker._control_client is not worker._client
            assert worker._control_client._transport._pool._max_connections == 8
            assert worker._client._transport._pool._max_connections == 128
            assert worker_cls.invalidate.__ray_concurrency_group__ == "cancellation"
            assert worker_cls.get_task_status.__ray_concurrency_group__ == "heartbeat"
        finally:
            await worker._control_client.aclose()
            await worker._client.aclose()

    asyncio.run(scenario())


@pytest.mark.unit
def test_cancelled_server_id_is_not_polled_until_client_deadline(local_eval_worker):
    worker, _, _ = local_eval_worker
    paths = []

    async def handler(request):
        paths.append(request.url.path)
        return httpx.Response(409, json={"detail": "Task was cancelled; use a new ID"})

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as worker._client:
            result = await worker.submit_and_poll({"task_id": "old"}, 100, -1, 1)
        assert result["status"] == "cancelled"
        assert paths == ["/evaluate"]

    asyncio.run(scenario())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
