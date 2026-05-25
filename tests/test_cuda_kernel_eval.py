import asyncio
import logging
import os
from types import SimpleNamespace

import pytest

from examples.kernel_agent import generate_with_cuda_agent
from examples.kernel_agent.config import CUDA_AGENT_CONFIGS
from examples.kernel_agent.utils import (
    extract_cuda_agent_kernel_code,
    normalize_env_feedback,
    parse_cuda_agent_response,
    precheck_cuda_agent_response,
    precheck_tvm_ffi_response,
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
    template = generate_with_cuda_agent._get_tool_response_template(SimpleNamespace(multi_turn_templates=None))
    return generate_with_cuda_agent._apply_feedback_template(env_result, template)


@pytest.mark.unit
@pytest.mark.parametrize("case", KERNEL_EVAL_CASES)
def test_precheck_accepts_cuda_agent_responses_from_compiled_feedback_cases(request, case):
    _skip_unselected_compiled_case(request, case, "feedback_compiled")
    assert precheck_cuda_agent_response(VALID_CUDA_AGENT_RESPONSE, "Model") is None


@pytest.mark.unit
def test_precheck_accepts_tvm_ffi_responses():
    assert precheck_tvm_ffi_response(VALID_TVM_FFI_RESPONSE, "Model") is None


@pytest.mark.unit
def test_precheck_rejects_tvm_ffi_missing_export():
    response = VALID_TVM_FFI_RESPONSE.replace(
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(copy_forward, copy_forward);",
        "TVM_FFI_DLL_EXPORT_TYPED_FUNC(copy_forward_exported, copy_forward);",
    )
    result = precheck_tvm_ffi_response(response, "Model")
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
    monkeypatch.setitem(CUDA_AGENT_CONFIGS, "log_multi_turn_sample_rate", 1.0)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS, "max_feedback_chars", 8192)

    captured_payload = {}

    async def fake_run_kernel_eval(args, sample, payload, config):
        captured_payload.update(payload)
        env_state = normalize_env_feedback(case["env_state"])
        return {"env_state": env_state, "reward_extra_info": env_state}

    monkeypatch.setattr(generate_with_cuda_agent, "run_kernel_eval", fake_run_kernel_eval)

    sample = Sample(
        prompt="Write a CUDA implementation.",
        label={"entry_point": "Model", "ground_truth": "class Model: pass"},
        metadata={
            "uuid": case["uuid"],
            "source_row": case["source_row"],
            "log_multi_turn": True,
        },
    )

    env_result = asyncio.run(
        generate_with_cuda_agent.cuda_kernel_env(
            SimpleNamespace(),
            sample,
            VALID_CUDA_AGENT_RESPONSE,
            turn_idx=0,
        )
    )

    env_state = env_result["env_state"]
    format_feedback = _format_feedback_for_test(env_result)
    print(f"\n[cuda_agent][test][format_feedback][{case['uuid']}]\n{format_feedback}")
    assert captured_payload["uuid"] == case["uuid"]
    assert captured_payload["turn_idx"] == 0
    assert captured_payload["response"] == VALID_CUDA_AGENT_RESPONSE
    assert env_state["compiled"] is case["feedback_compiled"]
    assert env_state["status"] == "completed"
    assert "error_code" not in env_state
    assert "submitted_at" not in env_state
    assert "completed_at" not in env_state
    if not case["feedback_compiled"]:
        assert env_state["error"] == "COMPILATION_ERROR"
        assert env_state["correctness"] is None
        assert env_state["decoy_kernel"] is None
        assert env_state["reference_runtime"] is None
        assert env_state["kernel_runtime"] is None
        assert env_state["speedup"] is None
        assert "Compilation failed. Compiler output:" in env_state["error_message"]
        assert "nvcc fatal: syntax error" in env_state["error_message"]
        assert "compile_only" not in env_state["metadata"]
        assert "entry_point" not in env_state["metadata"]
        assert "required_resource" not in env_state["metadata"]
        assert "task_id" not in env_state["metadata"]
        assert "inline_gpu_execute_completed" not in env_state["metadata"]
        assert "inline_compile_worker_id" not in env_state["metadata"]
        assert "inline_compile_worker_device" not in env_state["metadata"]
        assert env_state["metadata"]["refer_entry_point"] == "Model"
        assert env_state["metadata"]["kernel_entry_point"] == "ModelNew"
        assert "error" not in env_state["metadata"]["compile_artifact"]
        assert "precheck" not in env_state["metadata"]["compile_artifact"]
        assert "backend" not in env_state["metadata"]["compile_artifact"]
        assert "compile_mode" not in env_state["metadata"]["compile_artifact"]
        assert "compiled" not in env_state["metadata"]["compile_artifact"]
        assert "source_mode" not in env_state["metadata"]["compile_artifact"]
        assert "entry_point" not in env_state["metadata"]["compile_artifact"]

    caplog.set_level(logging.INFO, logger=generate_with_cuda_agent.logger.name)
    response_with_think = f"<think>\ntry a simple copy kernel\n</think>\n{VALID_CUDA_AGENT_RESPONSE}"
    generate_with_cuda_agent._log_multiturn_messages(
        sample,
        messages=[
            {"role": "user", "content": sample.prompt},
            {"role": "assistant", "content": response_with_think},
            {"role": "user", "content": format_feedback},
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
                "response": response_with_think,
                "env_state": env_state,
                "env_result": env_result,
                "format_feedback": format_feedback,
            }
        ],
        finish_reason="max_turns",
        is_slowest=True,
        total_request_time=1.0,
    )

    assert "[cuda_agent][multi_turn][slowest]" in caplog.text
    assert "total_request_time=1.000s" in caplog.text
    assert f"compiled={case['feedback_compiled']}" in caplog.text
    assert "reward=-0.2" in caplog.text
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
def test_split_think_response_handles_generation_prompt_prefilled_think():
    response = "reasoning from model\n</think>\n### CUDA_KERNELS\n```cpp\ncode\n```"
    response_think, response_content = split_think_response(response)
    assert response_think == "reasoning from model"
    assert response_content.startswith("### CUDA_KERNELS")


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
    monkeypatch.setitem(CUDA_AGENT_CONFIGS, "log_multi_turn_sample_rate", 1.0)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS, "max_feedback_chars", 8192)

    sample = Sample(
        prompt="Write a CUDA identity implementation.",
        label={"entry_point": "Model", "ground_truth": REFERENCE_IDENTITY_CODE},
        metadata={
            "uuid": case["uuid"],
            "source_row": "integration",
            "log_multi_turn": True,
        },
    )

    env_result = asyncio.run(
        generate_with_cuda_agent.cuda_kernel_env(
            SimpleNamespace(),
            sample,
            case["response"],
            turn_idx=0,
        )
    )
    env_state = env_result["env_state"]
    format_feedback = _format_feedback_for_test(env_result)
    print(f"\n[cuda_agent][test][format_feedback][{case['uuid']}]\n{format_feedback}")

    caplog.set_level(logging.INFO, logger=generate_with_cuda_agent.logger.name)
    generate_with_cuda_agent._log_multiturn_messages(
        sample,
        messages=[
            {"role": "user", "content": sample.prompt},
            {"role": "assistant", "content": case["response"]},
            {"role": "user", "content": format_feedback},
        ],
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
                "env_state": env_state,
                "format_feedback": format_feedback,
            }
        ],
        finish_reason="max_turns",
        is_slowest=True,
        total_request_time=float(env_state.get("processing_time") or 0.0),
    )

    assert env_state["status"] in {"completed", "failed", "timeout", "cancelled"}
    assert env_state.get("compiled") is case["expected_compiled"]
    assert "[cuda_agent][multi_turn][slowest]" in caplog.text
    assert case["uuid"] in caplog.text
    assert "format_feedback" in caplog.text
    assert "Server feedback (status/metrics/errors):" in caplog.text
