import argparse
import asyncio
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib import request

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.kernel_agent import generate_with_cuda_agent
from examples.kernel_agent.config import CUDA_AGENT_CONFIGS
from examples.kernel_agent.kernel_reward import calculate_reward
from slime.utils.data import read_file
from slime.utils.http_utils import init_http_client
from slime.utils.types import Sample


DEFAULT_SAMPLE_PATH = "/nfs/FM/lihongbin/datasets/CUDA_RL/SFT/prompt_v4/parallel_drkernel_minimax_results_sft.parquet"


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


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kwargs):
        rendered = []
        for message in messages:
            rendered.append(f"<|{message['role']}|>\n{message['content']}")
        if add_generation_prompt:
            rendered.append("<|assistant|>\n")
        text = "\n".join(rendered)
        if tokenize:
            return self(text, add_special_tokens=False)["input_ids"]
        return text

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(1, len(str(text).split()) + 1))}


class FakeGenerateState:
    def __init__(self, args):
        self.tokenizer = FakeTokenizer()
        self.multi_turn_templates = None
        self.apply_chat_template_kwargs = {}

    def _is_qwen3_5_model(self):
        return False


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _read_data_row(path: str, sample_index: int) -> dict[str, Any]:
    for idx, row in enumerate(read_file(path)):
        if idx == sample_index:
            return row
    raise IndexError(f"sample_index={sample_index} is out of range for {path}")


def _get_row_value(row: dict[str, Any], key: str) -> Any:
    value: Any = row
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _root_key(key: str) -> str:
    return key.split(".", 1)[0]


def _build_sample(args) -> Sample:
    if not args.sample_path:
        return Sample(
            prompt=args.prompt,
            label={"entry_point": "Model", "ground_truth": REFERENCE_IDENTITY_CODE},
            metadata={"uuid": args.uuid, "source_row": "generate_smoke", "log_multi_turn": True},
        )

    row = _read_data_row(args.sample_path, args.sample_index)
    prompt = _get_row_value(row, args.prompt_key)
    if prompt is None:
        raise KeyError(f"prompt_key={args.prompt_key!r} not found in sample. Available keys: {list(row.keys())}")
    if isinstance(prompt, list):
        if not prompt:
            raise ValueError(f"prompt_key={args.prompt_key!r} is an empty message list")
        if args.first_message_only:
            prompt = [prompt[0]]
        elif args.start_turn_index is not None:
            end_index = 2 * int(args.start_turn_index) + 1
            prompt = prompt[:end_index]

    entry_point = _get_row_value(row, args.entry_point_key) or "Model"
    ground_truth = _get_row_value(row, args.ground_truth_key) or REFERENCE_IDENTITY_CODE
    excluded_keys = {_root_key(args.prompt_key), _root_key(args.ground_truth_key)}
    metadata = {key: value for key, value in row.items() if key not in excluded_keys}
    metadata.update(
        {
            "uuid": str(_get_row_value(row, args.uuid_key) or args.uuid),
            "source_row": args.sample_index,
            "sample_path": args.sample_path,
            "log_multi_turn": True,
        }
    )
    print(
        "[cuda_agent][generate_smoke] loaded sample "
        f"path={args.sample_path} index={args.sample_index} uuid={metadata['uuid']} entry_point={entry_point}"
    )
    print(f"[cuda_agent][generate_smoke] prompt preview:\n{_json_dumps(prompt)[:2000]}")
    return Sample(
        prompt=prompt,
        label={"entry_point": entry_point, "ground_truth": ground_truth},
        metadata=metadata,
    )


def _mock_env_state(compiled: bool) -> dict[str, Any]:
    if compiled:
        return {
            "task_id": "mock_task_000001_true",
            "status": "completed",
            "compiled": True,
            "correctness": True,
            "decoy_kernel": False,
            "reference_runtime": 0.0255,
            "kernel_runtime": 0.0171,
            "speedup": 1.4912,
            "metadata": {"backend": "cuda_agent"},
            "success": True,
        }
    return {
        "task_id": "mock_task_000001_false",
        "status": "completed",
        "submitted_at": None,
        "completed_at": "2026-05-13T17:16:05.503366",
        "compiled": False,
        "correctness": False,
        "decoy_kernel": False,
        "reference_runtime": -1.0,
        "kernel_runtime": -1.0,
        "speedup": 0.0,
        "metadata": {
            "compile_only": True,
            "entry_point": "Model",
            "required_resource": "cpu",
            "task_id": "mock_task_000001_false",
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
    }


def _install_fake_generate_state() -> None:
    generate_with_cuda_agent.GenerateState = FakeGenerateState


def _install_fake_model(response: str) -> None:
    async def fake_post(url, payload):
        token_ids = list(range(1, len(response.split()) + 1))
        return {
            "text": response,
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "output_token_logprobs": [[0.0, token_id] for token_id in token_ids],
            },
        }

    generate_with_cuda_agent.post = fake_post


def _install_fake_env(compiled: bool) -> None:
    async def fake_run_kernel_eval(args, sample, payload, config):
        env_state = generate_with_cuda_agent.normalize_env_feedback(_mock_env_state(compiled))
        return {"env_state": env_state, "reward_extra_info": env_state}

    generate_with_cuda_agent.run_kernel_eval = fake_run_kernel_eval


def _init_ray_for_kernel_env(args) -> None:
    if args.mock_env:
        return
    try:
        import ray
    except ImportError as exc:
        raise ImportError("Ray is required for real kernel env evaluation.") from exc

    if ray.is_initialized():
        return
    ray_address = args.ray_address or os.environ.get("RAY_ADDRESS")
    if ray_address:
        ray.init(address=ray_address, ignore_reinit_error=True)
    else:
        try:
            ray.init(ignore_reinit_error=True, include_dashboard=False, num_cpus=args.ray_num_cpus)
        except ValueError as exc:
            if "When connecting to an existing cluster" not in str(exc):
                raise
            ray.init(address="auto", ignore_reinit_error=True)


def _wait_http_ok(url: str, timeout: float) -> None:
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            with request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return
        except Exception as exc:
            last_error = exc
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {url}: {last_error}")


@contextmanager
def _maybe_start_rollout(args):
    process = None
    if args.start_rollout:
        if args.rollout_command:
            command = shlex.split(args.rollout_command)
        else:
            if not args.model_path:
                raise ValueError("--model-path is required when --start-rollout is set and --rollout-command is empty")
            command = [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                args.model_path,
                "--host",
                args.rollout_host,
                "--port",
                str(args.rollout_port),
            ]
            command.extend(args.rollout_extra_args or [])

        print(f"[cuda_agent][generate_smoke] starting rollout: {' '.join(command)}")
        process = subprocess.Popen(command)
        _wait_http_ok(f"http://{args.rollout_host}:{args.rollout_port}/health_generate", args.wait_rollout_timeout)
        print(f"[cuda_agent][generate_smoke] rollout ready: http://{args.rollout_host}:{args.rollout_port}")

    try:
        yield
    finally:
        if process is not None and process.poll() is None:
            print("[cuda_agent][generate_smoke] stopping rollout")
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


async def _run(args) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    _init_ray_for_kernel_env(args)
    CUDA_AGENT_CONFIGS["max_feedback_chars"] = args.max_feedback_chars
    CUDA_AGENT_CONFIGS["log_multi_turn_sample_rate"] = 1.0
    CUDA_AGENT_CONFIGS["do_precheck"] = args.do_precheck
    CUDA_AGENT_CONFIGS["finalize_mode"] = None if args.finalize_mode == "none" else args.finalize_mode

    response = VALID_CUDA_AGENT_RESPONSE if args.compiled else INVALID_COMPILE_CUDA_AGENT_RESPONSE
    use_real_model = args.start_rollout or args.real_model
    if not use_real_model:
        _install_fake_generate_state()
        _install_fake_model(response)
    if args.mock_env:
        _install_fake_env(args.compiled)
    else:
        if not args.kernel_env_url:
            raise ValueError("--kernel-env-url is required when --real-env is set")
        CUDA_AGENT_CONFIGS["env"]["kernel_env_url"] = args.kernel_env_url
        CUDA_AGENT_CONFIGS["env"]["kernel_eval_max_retries"] = args.kernel_eval_max_retries
        CUDA_AGENT_CONFIGS["env"]["kernel_eval_client_timeout"] = args.kernel_eval_client_timeout
        CUDA_AGENT_CONFIGS["env"]["kernel_eval_task_timeout"] = args.kernel_eval_task_timeout
        CUDA_AGENT_CONFIGS["env"]["kernel_eval_heartbeat_interval"] = args.kernel_eval_heartbeat_interval
        CUDA_AGENT_CONFIGS["env"]["num_correct_trials"] = args.num_correct_trials
        CUDA_AGENT_CONFIGS["env"]["num_perf_trials"] = args.num_perf_trials

    with _maybe_start_rollout(args):
        sample = _build_sample(args)
        rollout_args = SimpleNamespace(
            hf_checkpoint=args.hf_checkpoint or args.model_path,
            sglang_router_ip=args.rollout_host if use_real_model else "mock-router",
            sglang_router_port=args.rollout_port if use_real_model else 0,
            rollout_max_context_len=args.rollout_max_context_len,
            global_step=0,
            sglang_server_concurrency=args.sglang_server_concurrency,
            rollout_num_gpus=args.rollout_num_gpus,
            rollout_num_gpus_per_engine=args.rollout_num_gpus_per_engine,
            rollout_temperature=0.0,
            rollout_top_p=1.0,
            rollout_top_k=-1,
            rollout_max_response_len=args.max_new_tokens,
            rollout_stop=None,
            rollout_stop_token_ids=None,
            rollout_skip_special_tokens=True,
            sglang_enable_deterministic_inference=False,
            sglang_dp_size=1,
            rollout_seed=42,
            n_samples_per_prompt=1,
            ci_test=False,
            use_rollout_routing_replay=False,
            use_distributed_post=False,
            multi_turn_prompt_config_path=args.multi_turn_prompt_config_path,
            preserve_history_thinking=args.preserve_history_thinking,
            keep_history_thinking=args.keep_history_thinking,
            kernel_backend=args.kernel_backend,
            num_gpus_per_node=args.num_gpus_per_node,
            max_turns=args.max_turns,
            use_multi_turn=args.use_multi_turn,
            padding_turns=args.padding_turns,
            advantage_estimator=args.advantage_estimator,
            multi_turn_gamma=args.multi_turn_gamma,
            use_coverage_rs=args.use_coverage_rs,
            coverage_rs_key=args.coverage_rs_key,
            coverage_rs_threshold=args.coverage_rs_threshold,
            coverage_rs_factor=args.coverage_rs_factor,
        )
        sampling_params = {"max_new_tokens": args.max_new_tokens, "temperature": 0.0}

        if use_real_model:
            init_http_client(rollout_args)
        output_samples = await generate_with_cuda_agent.generate(rollout_args, sample, sampling_params)
        state = generate_with_cuda_agent.GenerateState(rollout_args)
        tool_response_template = generate_with_cuda_agent._get_tool_response_template(state)
        print(f"\n[cuda_agent][generate_smoke] output_samples={len(output_samples)}")
        for idx, output_sample in enumerate(output_samples):
            env_result = (output_sample.metadata or {}).get("env_result", {})
            env_state = env_result.get("env_state", env_result)
            reward = calculate_reward(env_result, CUDA_AGENT_CONFIGS["reward"])
            format_feedback = generate_with_cuda_agent._apply_feedback_template(
                env_result,
                tool_response_template,
            )
            print(f"\n[cuda_agent][generate_smoke][sample {idx}] status={output_sample.status}")
            print(f"[cuda_agent][generate_smoke][sample {idx}] remove_sample={output_sample.remove_sample}")
            print(
                f"[cuda_agent][generate_smoke][sample {idx}] remove_reason="
                f"{(output_sample.metadata or {}).get('remove_reason')}"
            )
            print(
                f"[cuda_agent][generate_smoke][sample {idx}] turn_idx={(output_sample.metadata or {}).get('turn_idx')}"
            )
            print(
                f"[cuda_agent][generate_smoke][sample {idx}] is_pad_turn="
                f"{(output_sample.metadata or {}).get('is_pad_turn')}"
            )
            print(
                f"[cuda_agent][generate_smoke][sample {idx}] multi_turn_reward="
                f"{(output_sample.metadata or {}).get('multi_turn_reward')}"
            )
            print(
                f"[cuda_agent][generate_smoke][sample {idx}] env_extra_info:\n"
                f"{_json_dumps((output_sample.metadata or {}).get('env_extra_info'))}"
            )
            print(f"[cuda_agent][generate_smoke][sample {idx}] response_length={output_sample.response_length}")
            print(f"[cuda_agent][generate_smoke][sample {idx}] response:\n{output_sample.response}")
            print(f"[cuda_agent][generate_smoke][sample {idx}] reward={reward}")
            print(f"[cuda_agent][generate_smoke][sample {idx}] sample.reward={output_sample.reward}")
            print(f"[cuda_agent][generate_smoke][sample {idx}] env_state:\n{_json_dumps(env_state)}")
            print(f"[cuda_agent][generate_smoke][sample {idx}] format_feedback:\n{format_feedback}")


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test the full CUDA agent generate -> env -> feedback path.")
    parser.add_argument("--real-env", dest="mock_env", action="store_false", help="Call the real KernelServer env.")
    parser.set_defaults(mock_env=True)
    parser.add_argument(
        "--kernel-env-url", default="", help="KernelServer URL, for example http://192.168.16.21:8003."
    )
    parser.add_argument("--kernel-backend", choices=("cuda_agent", "tvm_ffi", "triton"), default="cuda_agent")
    parser.add_argument("--real-model", action="store_true", help="Use an already running rollout /generate endpoint.")
    parser.add_argument(
        "--start-rollout", action="store_true", help="Start a local SGLang rollout server before generate."
    )
    parser.add_argument("--rollout-command", default="", help="Custom command used by --start-rollout.")
    parser.add_argument("--model-path", default="", help="Model path for python -m sglang.launch_server.")
    parser.add_argument("--hf-checkpoint", default="", help="Tokenizer checkpoint. Defaults to --model-path.")
    parser.add_argument("--rollout-host", default="127.0.0.1")
    parser.add_argument("--rollout-port", type=int, default=30000)
    parser.add_argument("--wait-rollout-timeout", type=float, default=600.0)
    parser.add_argument("--rollout-extra-args", nargs=argparse.REMAINDER, default=[])
    parser.add_argument("--compiled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-turns", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-feedback-chars", type=int, default=0)
    parser.add_argument(
        "--multi-turn-prompt-config-path",
        default=None,
        help="YAML config path for multi-turn prompt templates. Defaults to the built-in CUDA agent template.",
    )
    parser.add_argument("--preserve-history-thinking", action="store_true", default=False)
    parser.add_argument("--keep-history-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-multi-turn", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--padding-turns", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--advantage-estimator", default="grpo")
    parser.add_argument("--multi-turn-gamma", type=float, default=1.0)
    parser.add_argument("--finalize-mode", choices=("none", "positive", "improve"), default="positive")
    parser.add_argument("--use-coverage-rs", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--coverage-rs-key", choices=("time_coverage", "num_coverage"), default="time_coverage")
    parser.add_argument("--coverage-rs-threshold", type=float, default=0.3)
    parser.add_argument("--coverage-rs-factor", type=float, default=0.1)
    parser.add_argument("--rollout-max-context-len", type=int, default=8192)
    parser.add_argument("--uuid", default="generate-smoke")
    parser.add_argument("--prompt", default="Write a CUDA identity implementation.")
    parser.add_argument("--sample-path", default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--prompt-key", default="messages")
    parser.add_argument("--first-message-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--start-turn-index",
        type=int,
        default=None,
        help="Use history through this assistant turn index: 1 keeps user-assistant-user. Requires --no-first-message-only.",
    )
    parser.add_argument("--ground-truth-key", default="original_python_code")
    parser.add_argument("--entry-point-key", default="entry_point")
    parser.add_argument("--uuid-key", default="uuid")
    parser.add_argument("--do-precheck", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--sglang-server-concurrency", type=int, default=1)
    parser.add_argument("--rollout-num-gpus", type=int, default=1)
    parser.add_argument("--rollout-num-gpus-per-engine", type=int, default=1)
    parser.add_argument("--num-gpus-per-node", type=int, default=1)
    parser.add_argument("--kernel-eval-max-retries", type=int, default=1)
    parser.add_argument("--kernel-eval-client-timeout", type=int, default=300)
    parser.add_argument("--kernel-eval-task-timeout", type=int, default=120)
    parser.add_argument("--kernel-eval-heartbeat-interval", type=float, default=5.0)
    parser.add_argument(
        "--ray-address", default="", help="Ray address for kernel env workers. Empty starts local Ray."
    )
    parser.add_argument("--ray-num-cpus", type=int, default=2)
    parser.add_argument("--num-correct-trials", type=int, default=1)
    parser.add_argument("--num-perf-trials", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(_run(parse_args()))
