import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.kernel_agent.config import CUDA_AGENT_CONFIGS
from examples.kernel_agent.kernel_response import run_kernel_eval
from examples.kernel_agent.test.run_generate_smoke import _build_sample, _init_ray_for_kernel_env
from examples.kernel_agent.utils import (
    extract_cuda_agent_kernel_code,
    normalize_env_feedback,
    parse_cuda_agent_response,
    precheck_response,
)


DEFAULT_LOG_PATH = "examples/kernel_agent/test/log/run_generate_smoke_real_sample_20260525_090634_sample0.log"
DEFAULT_SAMPLE_PATH = "/nfs/FM/lihongbin/datasets/CUDA_RL/RL_Data/prompt_tvm/drkernel_rl_thinking.parquet"


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _extract_response_from_log(log_path: str) -> str:
    text = Path(log_path).read_text()
    marker = "response_content:\n"
    start = text.rfind(marker)
    if start < 0:
        raise ValueError(f"Cannot find response_content marker in {log_path}")
    start += len(marker)

    end_markers = (
        "\nServer feedback (status/metrics/errors):",
        "\n[cuda_agent][generate_smoke]",
    )
    end = len(text)
    for end_marker in end_markers:
        candidate = text.find(end_marker, start)
        if candidate >= 0:
            end = min(end, candidate)

    response = text[start:end].strip()
    if not response:
        raise ValueError(f"Empty response extracted from {log_path}")
    return response


async def _run(args) -> None:
    logging.basicConfig(
        level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    response = args.response or _extract_response_from_log(args.log_path)
    sample = _build_sample(args)
    entry_point = sample.label["entry_point"]
    precheck_entry_point = f"{entry_point}New"

    cuda_sources, model_new_code = parse_cuda_agent_response(response)
    kernel_code = extract_cuda_agent_kernel_code(response)
    precheck_result = precheck_response(response, precheck_entry_point, args.kernel_backend)

    print("[cuda_agent][response_pipeline] parse summary:")
    print(
        _json_dumps(
            {
                "log_path": args.log_path,
                "response_chars": len(response),
                "kernel_code_chars": len(kernel_code),
                "source_files": {name: len(content) for name, content in cuda_sources.items()},
                "has_model_new": model_new_code is not None,
                "model_new_chars": len(model_new_code or ""),
                "kernel_backend": args.kernel_backend,
                "reference_backend": args.reference_backend,
                "entry_point": entry_point,
                "precheck_entry_point": precheck_entry_point,
            }
        )
    )

    print("[cuda_agent][response_pipeline] precheck result:")
    print(_json_dumps(precheck_result or {"precheck": "passed"}))

    if args.print_kernel_code:
        print("[cuda_agent][response_pipeline] extracted kernel_code:")
        print(kernel_code)

    if args.skip_env:
        return
    if precheck_result is not None and not args.run_env_on_precheck_fail:
        print("[cuda_agent][response_pipeline] skip Env because precheck failed.")
        return
    if not args.kernel_env_url:
        raise ValueError("--kernel-env-url is required unless --skip-env is set")

    _init_ray_for_kernel_env(args)
    CUDA_AGENT_CONFIGS["env"]["kernel_env_url"] = args.kernel_env_url

    payload = {
        "response": response,
        "ground_truth": sample.label["ground_truth"],
        "kernel_backend": args.kernel_backend,
        "reference_backend": args.reference_backend,
        "entry_point": entry_point,
        "uuid": (sample.metadata or {}).get("uuid"),
        "return_full_state": True,
        "metadata": sample.metadata,
        "turn_idx": args.turn_idx,
    }
    env_result = await run_kernel_eval(args, sample, payload, CUDA_AGENT_CONFIGS["env"])
    raw_env_state = env_result.get("env_state", env_result)
    normalized_env_state = normalize_env_feedback(raw_env_state)
    if args.do_precheck:
        normalized_env_state["precheck"] = "passed" if precheck_result is None else "failed"

    print("[cuda_agent][response_pipeline] raw Env result:")
    print(_json_dumps(raw_env_state))
    print("[cuda_agent][response_pipeline] normalized Env result:")
    print(_json_dumps(normalized_env_state))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay a logged CUDA-agent response through parse -> precheck -> Kernel Env -> normalize."
    )
    parser.add_argument("--log-path", default=DEFAULT_LOG_PATH)
    parser.add_argument(
        "--response", default="", help="Use this response string instead of extracting from --log-path."
    )
    parser.add_argument("--print-kernel-code", action="store_true")
    parser.add_argument("--skip-env", action="store_true")
    parser.add_argument("--run-env-on-precheck-fail", action="store_true")
    parser.add_argument("--kernel-env-url", default="http://192.168.16.21:20111")
    parser.add_argument("--kernel-backend", choices=("cuda_agent", "tvm_ffi", "triton"), default="tvm_ffi")
    parser.add_argument("--reference-backend", choices=("torch", "torch_compile"), default="torch")
    parser.add_argument("--sample-path", default=DEFAULT_SAMPLE_PATH)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--prompt-key", default="prompt")
    parser.add_argument("--first-message-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start-turn-index", type=int, default=None)
    parser.add_argument("--ground-truth-key", default="reward_model.ground_truth")
    parser.add_argument("--entry-point-key", default="extra_info.entry_point")
    parser.add_argument("--uuid-key", default="extra_info.uuid")
    parser.add_argument("--uuid", default="response-pipeline")
    parser.add_argument("--prompt", default="Replay a CUDA-agent response.")
    parser.add_argument("--turn-idx", type=int, default=0)
    parser.add_argument("--do-precheck", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mock-env", action="store_true", default=False)
    parser.add_argument("--ray-address", default="")
    parser.add_argument("--ray-num-cpus", type=int, default=2)
    parser.add_argument("--kernel-eval-max-retries", type=int, default=1)
    parser.add_argument("--kernel-eval-client-timeout", type=int, default=300)
    parser.add_argument("--kernel-eval-task-timeout", type=int, default=120)
    parser.add_argument("--kernel-eval-poll-interval", type=float, default=1.0)
    parser.add_argument("--kernel-eval-heartbeat-interval", type=float, default=5.0)
    parser.add_argument("--kernel-eval-worker-max-concurrency", type=int, default=32)
    parser.add_argument("--kernel-eval-rate-limit", type=int, default=32)
    parser.add_argument("--kernel-eval-acquire-timeout", type=int, default=300)
    parser.add_argument("--num-correct-trials", type=int, default=1)
    parser.add_argument("--num-perf-trials", type=int, default=1)
    parser.add_argument("--verbose-errors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-profiling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--detect-decoy-kernel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split-compile-and-execute", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-compile-artifact-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kernel-eval-function-path", default=None)
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(_run(parse_args()))
