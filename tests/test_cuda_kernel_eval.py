import asyncio
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

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

from examples.kernel_agent import generate_with_cuda_agent
from examples.kernel_agent import utils as kernel_agent_utils
from examples.kernel_agent.config import CUDA_AGENT_CONFIGS
from examples.kernel_agent.utils import (
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
    diagnostic = result["metadata"]["precheck_diagnostic"]
    assert diagnostic["code"] == "TVM_FFI_UNRESOLVED_CALL"
    assert diagnostic["phase"] == "binding_contract"
    assert [(item["kind"], item["value"]) for item in diagnostic["evidence"]] == [
        ("extension_call", "copy_forward"),
        ("exported_symbol", "copy_forward_exported"),
    ]
    assert diagnostic["evidence"][0]["section"] == "MODEL_NEW"
    assert diagnostic["evidence"][0]["line"] > 0
    assert diagnostic["evidence"][1]["section"] == "APPLY_BINDINGS"
    assert diagnostic["evidence"][1]["line"] > 0
    assert not ({"nearest_export", "suggested_edit", "repair_scope"} & diagnostic.keys())


@pytest.mark.unit
def test_precheck_reports_factual_python_syntax_location():
    response = VALID_TVM_FFI_RESPONSE.replace("def forward(self, x):", "def forward(self, x)")

    precheck_passed, result = precheck_response(response, "Model", "tvm_ffi")

    assert precheck_passed is False
    assert result is not None
    diagnostic = result["metadata"]["precheck_diagnostic"]
    assert diagnostic["code"] == "MODEL_NEW_PYTHON_SYNTAX"
    assert diagnostic["phase"] == "python_syntax"
    assert diagnostic["evidence"][0]["section"] == "MODEL_NEW"
    assert diagnostic["evidence"][0]["line"] > 0
    assert "def forward(self, x)" in diagnostic["evidence"][0]["snippet"]


@pytest.mark.unit
def test_precheck_reports_factual_host_cuda_marker_location():
    response = VALID_TVM_FFI_RESPONSE.replace(
        "#include <tvm/ffi/tvm_ffi.h>",
        "#include <tvm/ffi/tvm_ffi.h>\n#include <cuda_runtime.h>",
    )

    precheck_passed, result = precheck_response(response, "Model", "tvm_ffi")

    assert precheck_passed is False
    assert result is not None
    diagnostic = result["metadata"]["precheck_diagnostic"]
    assert diagnostic["code"] == "TVM_FFI_HOST_CUDA_MARKER_FORBIDDEN"
    assert diagnostic["phase"] == "binding_contract"
    assert diagnostic["evidence"] == [
        {
            "kind": "host_cuda_marker",
            "value": "#include <cuda_runtime.h>",
            "section": "APPLY_BINDINGS",
            "line": 2,
            "column": 1,
        }
    ]


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
    monkeypatch.setitem(CUDA_AGENT_CONFIGS, "max_feedback_chars", 8192)

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
            SimpleNamespace(kernel_backend="cuda_agent", reference_backend="torch", do_precheck=False),
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
            }
        ],
        finish_reason="max_turns",
        should_log=True,
        is_slowest=True,
        total_request_time=1.0,
    )

    assert "[cuda_agent][slowest][rollout_info]" in caplog.text
    assert "total_request_time=1.000s" in caplog.text
    assert f"compiled={case['feedback_compiled']}" in caplog.text
    expected_reward = 1.2456140350877192 if case["feedback_compiled"] else 0.0
    assert f"reward={expected_reward}" in caplog.text
    assert "precheck=passed" in caplog.text
    assert f"task_id={case['env_state']['task_id']}" in caplog.text
    assert "reward=" in caplog.text
    assert "detail_env_time=" in caplog.text
    assert "compile_time" in caplog.text
    assert "perf_cv=" in caplog.text
    assert "kernel_perf_cv" in caplog.text
    assert "refer_perf_cv" in caplog.text
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
        env_state = {"compiled": True, "correctness": True, "speedup": 1.0, "metadata": {}}
        return {"env_state": env_state, "reward_extra_info": env_state}

    monkeypatch.setattr(generate_with_cuda_agent, "run_kernel_eval", fake_run_kernel_eval)

    sample = Sample(
        prompt="Write a CUDA implementation.",
        label={"ground_truth": "class Model: pass"},
        metadata={"problem_id": 1},
    )

    asyncio.run(
        generate_with_cuda_agent.cuda_kernel_env(
            SimpleNamespace(kernel_backend="cuda_agent", do_precheck=False),
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
def test_multiturn_log_can_omit_full_prompt_and_response(monkeypatch, caplog):
    monkeypatch.setenv("CUDA_AGENT_LOG_MULTI_TURN_TEXT", "0")
    sample = Sample(
        prompt="Write a CUDA implementation.",
        metadata={"uuid": "log-trim"},
    )

    caplog.set_level(logging.INFO, logger=generate_with_cuda_agent.logger.name)
    generate_with_cuda_agent._log_multiturn_messages(
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
                "env_state": {"status": "completed", "compiled": True},
                "env_result": {"env_state": {"status": "completed", "compiled": True}},
            }
        ],
        finish_reason="max_turns",
        is_slowest=True,
        total_request_time=1.0,
    )

    assert "[cuda_agent][multi_turn][slowest]" in caplog.text
    assert "sample=log-trim" in caplog.text
    assert "### CUDA_KERNELS" not in caplog.text
    assert "response_content" not in caplog.text
    assert "messages:" not in caplog.text


@pytest.mark.unit
def test_rollout_stats_only_omits_slowest_sample_body(monkeypatch, caplog):
    sentinel = "very-long-response-body-must-not-be-logged"
    sample = Sample(prompt="long prompt", metadata={"uuid": "slowest-stats-only"})
    monkeypatch.setitem(CUDA_AGENT_CONFIGS, "log_rollout_stats_only", True)

    caplog.set_level(logging.INFO, logger=generate_with_cuda_agent.logger.name)
    generate_with_cuda_agent._log_rollout_info(
        sample,
        messages=[
            {"role": "user", "content": sample.prompt},
            {"role": "assistant", "content": sentinel},
        ],
        turn_logs=[
            {
                "turn_idx": 0,
                "task_id": "slowest-task",
                "model_time": 12.5,
                "env_time": 3.5,
                "prompt_tokens": 128,
                "response_tokens": 16384,
                "finish_type": "length",
                "prompt": sample.prompt,
                "response": sentinel,
                "env_result": {"env_state": {"status": "completed"}},
                "format_feedback": sentinel,
            }
        ],
        finish_reason="max_turns",
        is_slowest=True,
        total_request_time=16.0,
    )

    assert "[cuda_agent][slowest][rollout_info]" in caplog.text
    assert "response_tokens=16384" in caplog.text
    assert sentinel not in caplog.text
    assert "response_content" not in caplog.text
    assert "messages:" not in caplog.text


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


@pytest.mark.parametrize("draft_tokens", [2, 3, 4])
def test_mtp_reserve_applies_only_at_serving_context_wall(draft_tokens):
    args = SimpleNamespace(
        rollout_max_context_len=40960,
        sglang_context_length=40960,
        turn_max_context_lens=[24576, 32768, 40960],
        sglang_speculative_algorithm="NEXTN",
        sglang_speculative_num_draft_tokens=draft_tokens,
    )
    actual = [
        generate_with_cuda_agent._sampling_params_for_prompt_context(
            args, {"max_new_tokens": 40960}, 20000, turn_idx=turn
        )["max_new_tokens"]
        for turn in range(3)
    ]
    assert actual == [4576, 12768, 20960 - draft_tokens]
    args.sglang_context_length = 32768
    assert (
        generate_with_cuda_agent._sampling_params_for_prompt_context(
            args, {"max_new_tokens": 40960}, 32767, turn_idx=2
        )["max_new_tokens"]
        == 0
    )


@pytest.mark.unit
def test_cuda_agent_sampling_params_use_first_turn_context_cap_only_for_turn_zero():
    args = SimpleNamespace(
        rollout_max_context_len=32768,
        first_turn_max_context_len=24576,
        sglang_speculative_algorithm=None,
    )
    sampling_params = {"max_new_tokens": 32768, "temperature": 1.0}

    first_turn = generate_with_cuda_agent._sampling_params_for_prompt_context(
        args,
        sampling_params,
        prompt_token_count=1536,
        turn_idx=0,
    )
    second_turn = generate_with_cuda_agent._sampling_params_for_prompt_context(
        args,
        sampling_params,
        prompt_token_count=20000,
        turn_idx=1,
    )

    assert first_turn["max_new_tokens"] == 24576 - 1536
    assert second_turn["max_new_tokens"] == 32768 - 20000
    assert sampling_params["max_new_tokens"] == 32768


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
    assert "[cuda_agent][slowest][rollout_info]" in caplog.text
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


def test_kernel_agent_metrics_keep_selected_detail_env_time_and_omit_compile_time():
    from slime.ray.rollout import _compute_kernel_agent_metrics

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

    assert not any(key.startswith("kernel/time/detail_env_time/compile_time/") for key in metrics)
    assert metrics["kernel/time/detail_env_time/kernel_runtime/mean"] == pytest.approx(20.0)
    assert metrics["kernel/time/detail_env_time/profile_time/sum"] == pytest.approx(400.0)
    assert metrics["kernel/time/detail_env_time/refer_runtime/max"] == pytest.approx(3000.0)
    assert metrics["sample_mask/conditional_truncation_masked_fraction"] == pytest.approx(1 / 3)


def test_single_turn_kernel_metrics_include_turn_zero_correctness():
    from slime.ray.rollout import compute_metrics_from_samples

    samples = [
        Sample(
            index=0,
            response="correct",
            response_length=1,
            status=Sample.Status.COMPLETED,
            metadata={
                "turn_idx": 0,
                "env_extra_info": {
                    "correctness": True,
                    "compilation": True,
                    "speedup": 1.5,
                    "decoy_kernel": False,
                },
            },
        ),
        Sample(
            index=1,
            response="decoy",
            response_length=1,
            status=Sample.Status.COMPLETED,
            metadata={
                "turn_idx": 0,
                "env_extra_info": {
                    "correctness": True,
                    "compilation": True,
                    "speedup": 1.0,
                    "decoy_kernel": True,
                },
            },
        ),
    ]
    args = SimpleNamespace(
        use_multi_turn=False,
        max_turns=1,
        advantage_estimator="ppo",
        sglang_speculative_algorithm=None,
        log_reward_category=None,
    )

    metrics = compute_metrics_from_samples(args, samples)

    assert metrics["kernel/turn0/correctness"] == pytest.approx(0.5)
    assert metrics["kernel/turn0/compilation"] == 1.0
    assert metrics["kernel/turn0/speedup"] == pytest.approx(1.25)
    assert metrics["kernel/turn0/fast@1"] == pytest.approx(0.5)
    assert not any(key.startswith("kernel/trajectory/") for key in metrics)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
