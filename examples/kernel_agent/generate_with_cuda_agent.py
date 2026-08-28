import ast
import asyncio
import hashlib
import json
import logging
import os
import random
import time
from copy import deepcopy
from typing import Any

import numpy as np

try:
    import ray
except ImportError:
    ray = None

from slime.rollout.sglang_rollout import (
    GenerateState,
    PromptTemplate,
    _decode_routed_experts,
    _empty_predictive_support,
    _extract_predictive_support,
)
from slime.utils.http_utils import post
from slime.utils.lora_utils import rollout_lora_path as _rollout_lora_path
from slime.utils.types import Sample

try:
    from .config import CUDA_AGENT_CONFIGS
    from .kernel_response import cancel_kernel_eval, next_kernel_task_id, run_kernel_eval
    from .kernel_reward import calculate_reward, calculate_reward_speedup
    from .utils import (
        _extract_env_extra_info,
        extract_cuda_agent_kernel_code,
        normalize_env_feedback,
        postprocess_turn_samples,
        precheck_response,
        split_think_response,
    )
except ImportError:
    from config import CUDA_AGENT_CONFIGS
    from kernel_response import cancel_kernel_eval, next_kernel_task_id, run_kernel_eval
    from kernel_reward import calculate_reward, calculate_reward_speedup

    from utils import (
        _extract_env_extra_info,
        extract_cuda_agent_kernel_code,
        normalize_env_feedback,
        postprocess_turn_samples,
        precheck_response,
        split_think_response,
    )

logger = logging.getLogger(__name__)

KERNEL_AGENT_GENERATE_GUARD_SEC = int(os.environ.get("KERNEL_AGENT_GENERATE_GUARD_SEC", "0") or 0) or (
    int(CUDA_AGENT_CONFIGS["env"].get("kernel_eval_client_timeout", 2400))
    + int(CUDA_AGENT_CONFIGS["env"].get("kernel_eval_task_timeout", 300))
    + 900
)
KERNEL_AGENT_GENERATE_MAX_RETRIES = max(1, int(os.environ.get("KERNEL_AGENT_GENERATE_MAX_RETRIES", "60") or 60))
LOG_FIRST_ROLLOUT = bool(int(os.environ.get("CUDA_AGENT_LOG_FIRST_ROLLOUT", "0")))
_LOGGED_FIRST_ROLLOUT = False


def _log_multiturn_full_text_enabled() -> bool:
    value = os.environ.get("CUDA_AGENT_LOG_MULTI_TURN_TEXT")
    if value is not None:
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(CUDA_AGENT_CONFIGS.get("log_multi_turn_full_text", True))


if ray is not None:

    @ray.remote
    class SlowestRequestTracker:
        def __init__(self) -> None:
            self.slowest_request_time = 0.0
            self.current_step = -1

        def update_slowest_time(
            self,
            request_time: float,
            global_step: int,
            step_window: int,
            min_delta_seconds: float,
        ) -> bool:
            if step_window > 0 and global_step % step_window == 0 and global_step > self.current_step:
                self.slowest_request_time = 0.0
                self.current_step = global_step

            if request_time > self.slowest_request_time + min_delta_seconds:
                self.slowest_request_time = request_time
                return True
            return False


DEFAULT_TOOL_RESPONSE_TEMPLATE = """Now you have received the server feedback for your last implementation. Based on that and all your previous responses, improve the implementation.

Here is the server feedback. Please refer to this feedback to improve the implementation:
Server feedback (status/metrics/errors):
{feedback}

Modify any section as needed.

Return an improved CUDA implementation with the same output format.
Let's think step by step.
"""


def _as_messages(prompt: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(prompt, list):
        return deepcopy(prompt)
    return [{"role": "user", "content": str(prompt)}]


def _get_tool_response_template(state: GenerateState) -> PromptTemplate:
    response_template = getattr(state, "multi_turn_template", None)
    if response_template is None:
        logger.warning("multi-turn tool_response template is not set; using built-in CUDA agent prompt template.")
        return PromptTemplate(DEFAULT_TOOL_RESPONSE_TEMPLATE, "format", "built-in")
    return response_template


def _truncate_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep = max_chars // 2
    return text[:keep] + "...(truncated)..." + text[-keep:]


def _apply_feedback_template(env_result: dict[str, Any], response_template: PromptTemplate) -> str:
    feedback_dict = env_result.get("env_state") or env_result
    try:
        feedback = json.dumps(feedback_dict, ensure_ascii=False, indent=2)
    except TypeError:
        feedback = str(feedback_dict)

    max_chars = int(CUDA_AGENT_CONFIGS["max_feedback_chars"])
    # max_chars <= 0 means no truncation
    feedback = _truncate_middle(feedback, max_chars)
    return response_template.format(feedback=feedback, feedback_dict=feedback_dict)


def _format_log_value(value: Any, max_chars: int) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, indent=2)
        except TypeError:
            text = str(value)
    return _truncate_middle(text, max_chars)


def _as_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _should_log_rollout(sample: Sample) -> bool:
    if not logger.isEnabledFor(logging.INFO):
        return False

    metadata = sample.metadata or {}
    if metadata.get("log_rollout_info") or metadata.get("should_log"):
        return True

    log_rate = float(CUDA_AGENT_CONFIGS.get("log_rollout_info_rate", 0.0) or 0.0)
    if log_rate <= 0:
        return False
    if log_rate >= 1:
        return True
    return random.random() < log_rate


def _claim_first_rollout_log() -> bool:
    global _LOGGED_FIRST_ROLLOUT

    if not LOG_FIRST_ROLLOUT or _LOGGED_FIRST_ROLLOUT:
        return False

    _LOGGED_FIRST_ROLLOUT = True
    return True


def _update_sample_progress(
    sample: Sample,
    turn_logs: list[dict[str, Any]],
    finish_reason: str,
    *,
    abort_reason: str | None = None,
    elapsed_sec: float | None = None,
) -> dict[str, Any]:
    total_model_time = sum(float(item.get("model_time", 0.0)) for item in turn_logs)
    total_env_time = sum(float(item.get("env_time", 0.0)) for item in turn_logs)
    total_request_time = total_model_time + total_env_time
    if elapsed_sec is not None:
        total_request_time = max(total_request_time, float(elapsed_sec))

    metadata = dict(sample.metadata or {})
    metadata.update(
        {
            "finish_reason": finish_reason,
            "total_request_time": total_request_time,
            "num_turns_completed": len(turn_logs),
            "total_model_time": total_model_time,
            "total_env_time": total_env_time,
        }
    )
    if abort_reason is not None:
        metadata["abort_reason"] = abort_reason
    sample.metadata = metadata
    return metadata


async def _is_slowest_multiturn(args, sample: Sample, total_request_time: float) -> bool:
    if ray is None or not ray.is_initialized():
        return False

    try:
        try:
            tracker = ray.get_actor("CudaAgentSlowestRequestTracker")
        except ValueError:
            tracker = SlowestRequestTracker.options(name="CudaAgentSlowestRequestTracker", get_if_exists=True).remote()

        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        try:
            rollout_step = int(metadata.get("rollout_step", 0))
        except (TypeError, ValueError):
            rollout_step = 0

        object_ref = tracker.update_slowest_time.remote(
            float(total_request_time),
            rollout_step,
            int(CUDA_AGENT_CONFIGS.get("log_slowest_step_window", 10)),
            float(CUDA_AGENT_CONFIGS.get("log_slowest_min_delta_seconds", 5.0)),
        )
        timeout = float(CUDA_AGENT_CONFIGS.get("slowest_tracker_timeout", 2.0))
        done, _ = await asyncio.to_thread(ray.wait, [object_ref], num_returns=1, timeout=timeout)
        if not done:
            logger.debug("Slowest tracker update timed out after %.2fs; skipping slowest-request logging.", timeout)
            return False
        return bool(await asyncio.to_thread(ray.get, done[0]))
    except Exception as exc:
        logger.debug("Failed to update slowest tracker: %s", exc)
        return False


def _log_rollout_info(
    sample: Sample,
    messages: list[dict[str, Any]],
    turn_logs: list[dict[str, Any]],
    finish_reason: str,
    *,
    should_log: bool = False,
    is_slowest: bool = False,
    log_first_rollout: bool = False,
    total_request_time: float | None = None,
) -> None:
    if not logger.isEnabledFor(logging.INFO):
        return
    if not (should_log or is_slowest or log_first_rollout):
        return

    prefix = "[cuda_agent]"
    if log_first_rollout:
        prefix += "[first_rollout]"
    if is_slowest:
        prefix += "[slowest]"

    metadata = sample.metadata or {}
    sample_id = metadata.get("uuid") or metadata.get("uid") or metadata.get("index") or "unknown"
    total_model_time = sum(float(item.get("model_time", 0.0)) for item in turn_logs)
    total_env_time = sum(float(item.get("env_time", 0.0)) for item in turn_logs)
    if total_request_time is None:
        total_request_time = total_model_time + total_env_time

    total_detail_env_time: dict[str, float] = {}
    perf_cv_values: dict[str, list[float]] = {"kernel_perf_cv": [], "refer_perf_cv": []}
    for item in turn_logs:
        env_result = item.get("env_result") if isinstance(item.get("env_result"), dict) else {}
        env_extra_info = env_result.get("env_extra_info") if isinstance(env_result.get("env_extra_info"), dict) else {}
        detail_env_time = env_extra_info.get("detail_env_time")
        if isinstance(detail_env_time, dict):
            for key, value in detail_env_time.items():
                value = _as_float_or_none(value)
                if value is not None:
                    total_detail_env_time[key] = total_detail_env_time.get(key, 0.0) + value
        for key in perf_cv_values:
            value = _as_float_or_none(env_extra_info.get(key))
            if value is not None:
                perf_cv_values[key].append(value)
    mean_perf_cv = {key: sum(values) / len(values) for key, values in perf_cv_values.items() if values}

    log_max_chars = int(CUDA_AGENT_CONFIGS.get("max_feedback_chars", 0) or 0)
    stats_only = bool(CUDA_AGENT_CONFIGS.get("log_rollout_stats_only", False))

    logger.info(
        "%s[rollout_info] sample=%s turns=%s finish_reason=%s total_request_time=%.3fs "
        "total_model_time=%.3fs total_env_time=%.3fs detail_env_time=%s perf_cv=%s",
        prefix,
        sample_id,
        len(turn_logs),
        finish_reason,
        total_request_time,
        total_model_time,
        total_env_time,
        _format_log_value(total_detail_env_time, log_max_chars),
        _format_log_value(mean_perf_cv, log_max_chars),
    )

    for item in turn_logs:
        env_result = item.get("env_result") if isinstance(item.get("env_result"), dict) else {}
        env_state = env_result.get("env_state") if isinstance(env_result.get("env_state"), dict) else {}
        reward = item.get("reward")
        if reward is None:
            reward = calculate_reward(item.get("env_result", {}), CUDA_AGENT_CONFIGS["reward"])
        env_extra_info = env_result.get("env_extra_info") if isinstance(env_result.get("env_extra_info"), dict) else {}
        detail_env_time = (
            env_extra_info.get("detail_env_time") if isinstance(env_extra_info.get("detail_env_time"), dict) else {}
        )
        perf_cv = {
            key: value
            for key in ("kernel_perf_cv", "refer_perf_cv")
            if (value := _as_float_or_none(env_extra_info.get(key))) is not None
        }
        logger.info(
            "%s[turn %s] task_id=%s model_time=%.3fs env_time=%.3fs prompt_tokens=%s max_new_tokens=%s "
            "response_tokens=%s "
            "finish_type=%s status=%s error=%s precheck=%s speedup=%s correctness=%s compiled=%s "
            "partial_credit=%s partial_reason=%s reward=%s detail_env_time=%s perf_cv=%s",
            prefix,
            item.get("turn_idx"),
            item.get("task_id"),
            float(item.get("model_time", 0.0)),
            float(item.get("env_time", 0.0)),
            item.get("prompt_tokens"),
            item.get("max_new_tokens"),
            item.get("response_tokens"),
            item.get("finish_type"),
            env_state.get("status"),
            env_state.get("error"),
            env_extra_info.get("precheck"),
            env_state.get("speedup"),
            env_state.get("correctness"),
            env_state.get("compiled"),
            env_extra_info.get("partial_credit_output_mismatch"),
            env_extra_info.get("partial_credit_output_mismatch_reason"),
            reward,
            _format_log_value(detail_env_time, log_max_chars),
            _format_log_value(perf_cv, log_max_chars),
        )
        if stats_only:
            continue
        logger.info(
            "%s[turn %s] prompt:\n%s",
            prefix,
            item.get("turn_idx"),
            _format_log_value(item.get("prompt", ""), log_max_chars),
        )
        response_think, response_content = split_think_response(str(item.get("response", "")))
        if response_think is not None:
            logger.info(
                "%s[turn %s] response_think:\n%s",
                prefix,
                item.get("turn_idx"),
                _format_log_value(response_think, log_max_chars),
            )
        logger.info(
            "%s[turn %s] response_content:\n%s",
            prefix,
            item.get("turn_idx"),
            _format_log_value(response_content, log_max_chars),
        )
        if item.get("format_feedback") is not None:
            logger.info(
                "%s[turn %s] format_feedback:\n%s",
                prefix,
                item.get("turn_idx"),
                _format_log_value(item.get("format_feedback", ""), log_max_chars),
            )
    if not stats_only:
        logger.info(
            "%s[messages]:\n%s",
            prefix,
            _format_log_value(messages, log_max_chars),
        )


def _log_multiturn_messages(
    sample: Sample,
    messages: list[dict[str, Any]],
    turn_logs: list[dict[str, Any]],
    finish_reason: str,
    is_slowest: bool = False,
    total_request_time: float | None = None,
) -> None:
    """Compatibility entry point for callers predating rollout-info logging."""
    if not logger.isEnabledFor(logging.INFO):
        return

    metadata = sample.metadata or {}
    sample_id = metadata.get("uuid") or metadata.get("uid") or metadata.get("index") or "unknown"
    total_model_time = sum(float(item.get("model_time", 0.0)) for item in turn_logs)
    total_env_time = sum(float(item.get("env_time", 0.0)) for item in turn_logs)
    if total_request_time is None:
        total_request_time = total_model_time + total_env_time
    logger.info(
        "[cuda_agent][multi_turn][%s] sample=%s turns=%s finish_reason=%s total_request_time=%.3fs "
        "total_model_time=%.3fs total_env_time=%.3fs",
        "slowest" if is_slowest else "sampled",
        sample_id,
        len(turn_logs),
        finish_reason,
        total_request_time,
        total_model_time,
        total_env_time,
    )
    if not _log_multiturn_full_text_enabled():
        return

    normalized_turn_logs = []
    for item in turn_logs:
        normalized_item = dict(item)
        if not isinstance(normalized_item.get("env_result"), dict):
            env_state = normalized_item.get("env_state")
            normalized_item["env_result"] = {"env_state": env_state if isinstance(env_state, dict) else {}}
        normalized_turn_logs.append(normalized_item)
    _log_rollout_info(
        sample,
        messages,
        normalized_turn_logs,
        finish_reason,
        should_log=True,
        is_slowest=is_slowest,
        total_request_time=total_request_time,
    )


def _is_done(env_result: dict[str, Any], turn_idx: int, max_turns: int) -> bool:
    if turn_idx + 1 >= max_turns:
        return True
    for key in ("done", "env_done", "success"):
        if key in env_result:
            return bool(env_result[key])
    env_state = env_result.get("env_state") or {}
    if isinstance(env_state, dict):
        for key in ("done", "env_done"):
            if key in env_state:
                return bool(env_state[key])
    return False


def _sampling_params_for_prompt_context(
    args,
    sampling_params: dict[str, Any],
    prompt_token_count: int,
) -> dict[str, Any]:
    turn_sampling_params = sampling_params.copy()
    max_context_len = getattr(args, "rollout_max_context_len", None)
    if max_context_len is None:
        return turn_sampling_params

    draft_token_reserve = 0
    if getattr(args, "sglang_speculative_algorithm", None):
        draft_token_reserve = max(0, int(getattr(args, "sglang_speculative_num_draft_tokens", 0) or 0))
    remaining_context = int(max_context_len) - int(prompt_token_count) - draft_token_reserve
    configured_max_new_tokens = turn_sampling_params.get("max_new_tokens")
    if configured_max_new_tokens is None:
        max_new_tokens = remaining_context
    else:
        max_new_tokens = min(int(configured_max_new_tokens), remaining_context)
    turn_sampling_params["max_new_tokens"] = max(0, max_new_tokens)
    return turn_sampling_params


def _get_label_value(sample: Sample, key: str) -> Any:
    if isinstance(sample.label, dict):
        value = sample.label.get(key)
        if value is not None:
            return value
    if isinstance(sample.metadata, dict):
        return sample.metadata.get(key)
    return None


def _get_entry_point(sample: Sample) -> str:
    entry_point = _get_label_value(sample, "entry_point")
    if entry_point is not None:
        return str(entry_point)
    return "Model"


_PRECISION_ALIASES = {
    "fp32": "fp32",
    "float32": "fp32",
    "torch.float32": "fp32",
    "fp16": "fp16",
    "float16": "fp16",
    "half": "fp16",
    "torch.float16": "fp16",
    "torch.half": "fp16",
    "bf16": "bf16",
    "bfloat16": "bf16",
    "torch.bfloat16": "bf16",
}


def _canonical_task_precision(value: Any) -> str | None:
    if value is None:
        return None
    return _PRECISION_ALIASES.get(str(value).strip().lower())


def _reference_input_precision(reference_code: Any) -> str:
    """Infer the effective input precision from ``get_inputs`` only.

    Serial layout augmentation intentionally leaves ``augmentation.dtype_after``
    empty on the layout child even when its parent was a dtype intervention.  The
    rewritten reference remains authoritative, though: all floating factories in
    ``get_inputs`` carry the selected ``torch.float16``/``torch.bfloat16`` dtype.
    Restricting inference to that function avoids treating unrelated casts in the
    model implementation as the task's input precision.
    """

    if not isinstance(reference_code, str) or not reference_code.strip():
        return "fp32"
    try:
        tree = ast.parse(reference_code)
    except (SyntaxError, ValueError, TypeError):
        return "fp32"

    detected: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or node.name != "get_inputs":
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name) and child.value.id == "torch":
                precision = _canonical_task_precision(f"torch.{child.attr}")
                if precision in {"fp16", "bf16"}:
                    detected.add(precision)

    # The augmentation contract uses one low-precision dtype for every floating
    # input.  Ambiguous mixed-dtype references retain the historical fp32 policy
    # instead of silently weakening the static precision check.
    return next(iter(detected)) if len(detected) == 1 else "fp32"


def _resolve_task_precision(sample: Sample, reference_code: Any) -> str:
    """Resolve the precision KernelGym should enforce for this task."""

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    augmentation = metadata.get("augmentation")
    if isinstance(augmentation, dict):
        explicit = _canonical_task_precision(augmentation.get("dtype_after"))
        if explicit is not None:
            return explicit

    explicit = _canonical_task_precision(metadata.get("precision"))
    if explicit is not None:
        return explicit
    return _reference_input_precision(reference_code)


def _reference_cache_uuid(ground_truth: Any, entry_point: Any) -> str | None:
    """Collision-resistant key for KernelGym's reference-timing cache.

    Derived purely from the reference identity (reference code + entry point) and
    NOT from any dataset-supplied id: same reference -> same key, different
    reference -> different key (modulo the negligible 64-bit truncation collision),
    so it cannot false-share a cached baseline across datasets on a shared
    KernelGym the way a bare per-problem id like "1" would. Returns None when
    there is no reference to hash, which leaves the cache disabled.

    Safety note: the cached baseline is only valid because KernelBench references
    use fixed-shape get_inputs(); a dataset with randomized reference inputs must
    NOT enable use_reference_cache.
    """
    if not ground_truth:
        return None
    if not isinstance(ground_truth, str):
        ground_truth = str(ground_truth)
    # 64 bits (16 hex) — collision-safe for realistic problem counts (~1e3-1e4).
    digest = hashlib.sha256(f"{entry_point}\n{ground_truth}".encode()).hexdigest()[:16]
    return f"ref_{digest}"


def _kernel_eval_config_value(args, config: dict[str, Any], name: str, default: Any = None) -> Any:
    value = getattr(args, name, None)
    if value is not None:
        return value
    return config.get(name, default)
async def cuda_kernel_env(
    args,
    sample: Sample,
    response: str,
    turn_idx: int,
) -> dict[str, Any]:
    entry_point = _get_entry_point(sample)
    do_precheck = bool(getattr(args, "do_precheck", True))
    kernel_backend = args.kernel_backend
    reference_backend = getattr(args, "reference_backend", "torch")
    precheck_passed = True
    if do_precheck:
        precheck_entry_point = f"{entry_point}New"
        precheck_passed, precheck_state = precheck_response(response, precheck_entry_point, kernel_backend)
    if not precheck_passed:
        if precheck_state is None:
            raise ValueError("precheck_response must return precheck_state when precheck fails.")
        return {
            "env_state": precheck_state,
            "env_extra_info": _extract_env_extra_info(precheck_state),
        }
    else:
        task_id = next_kernel_task_id()
        metadata = dict(sample.metadata or {})
        metadata["task_id"] = task_id
        sample.metadata = metadata

        env_config = CUDA_AGENT_CONFIGS["env"]
        reference_code = _get_label_value(sample, "ground_truth")
        uuid = _reference_cache_uuid(reference_code, entry_point)
        payload = {
            "task_id": task_id,
            "reference_code": reference_code,
            "kernel_code": extract_cuda_agent_kernel_code(response),
            "backend": kernel_backend,
            "reference_backend": reference_backend,
            "entry_point": entry_point,
            "precision": _resolve_task_precision(sample, reference_code),
            "uuid": uuid,
            "num_correct_trials": env_config.get("num_correct_trials"),
            "num_perf_trials": env_config.get("num_perf_trials"),
            "num_warmup": env_config.get("num_warmup"),
            "perf_trim_count": env_config.get("perf_trim_count"),
            "adaptive_perf_trials": env_config.get("adaptive_perf_trials"),
            "perf_min_trials": env_config.get("perf_min_trials"),
            "perf_cv_threshold": env_config.get("perf_cv_threshold"),
            "timeout": _kernel_eval_config_value(args, env_config, "kernel_eval_task_timeout"),
            "priority": "normal",
            "is_valid": False,
            "verbose_errors": _kernel_eval_config_value(args, env_config, "verbose_errors", True),
            "enable_profiling": _kernel_eval_config_value(args, env_config, "enable_profiling", True),
            "enable_ncu": bool(env_config.get("enable_ncu", False)),
            "enable_compute_sanitizer": bool(env_config.get("enable_compute_sanitizer", False)),
            "compute_sanitizer_mode": env_config.get("compute_sanitizer_mode", "error_based"),
            "enable_correctness_input_perturbations": bool(
                env_config.get("enable_correctness_input_perturbations", False)
            ),
            "memory_ratio_threshold": env_config.get("memory_ratio_threshold", 1.8),
            "detect_decoy_kernel": _kernel_eval_config_value(args, env_config, "detect_decoy_kernel", True),
        }
        use_reference_cache = _kernel_eval_config_value(args, env_config, "use_reference_cache", False)
        if use_reference_cache and payload["uuid"] is not None:
            payload["use_reference_cache"] = True
        if _kernel_eval_config_value(args, env_config, "split_compile_and_execute", True):
            payload["split_compile_and_execute"] = True
        if _kernel_eval_config_value(args, env_config, "enable_compile_artifact_cache", True):
            payload["enable_compile_artifact_cache"] = True
        for name in ("refer_num_perf_trials", "correctness_timeout", "correctness_timeout_enabled"):
            value = env_config.get(name)
            if value is not None:
                payload[name] = value

        kernel_eval_result = await run_kernel_eval(args, sample, payload, env_config)
        raw_env_state = kernel_eval_result.get("env_state") if isinstance(kernel_eval_result, dict) else None
        if not isinstance(raw_env_state, dict):
            raw_env_state = kernel_eval_result
        if not isinstance(raw_env_state, dict):
            raise TypeError("Kernel eval result must be a dict or contain dict env_state.")
        normalized_env_state, env_extra_info = normalize_env_feedback(raw_env_state)
        return {
            "env_state": normalized_env_state,
            "env_extra_info": env_extra_info,
        }


def _sample_for_turn(
    base_sample: Sample,
    *,
    prompt_ids: list[int],
    response: str,
    response_ids: list[int],
    log_probs: list[float],
    reward: float | dict[str, Any],
    status: Sample.Status,
    turn_idx: int,
    env_result: dict[str, Any],
    args: Any = None,
    meta_info: dict[str, Any] | None = None,
    predictive_support: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> Sample:
    turn_sample = deepcopy(base_sample)
    turn_sample.tokens = prompt_ids + response_ids
    turn_sample.response = response
    turn_sample.response_length = len(response_ids)
    turn_sample.rollout_log_probs = log_probs
    if predictive_support is None:
        turn_sample.rollout_topk_token_ids = None
        turn_sample.rollout_topk_log_probs = None
        turn_sample.rollout_topk_valid_mask = None
    else:
        (
            turn_sample.rollout_topk_token_ids,
            turn_sample.rollout_topk_log_probs,
            turn_sample.rollout_topk_valid_mask,
        ) = predictive_support
    turn_sample.reward = reward
    turn_sample.status = status
    turn_sample.rollout_id = base_sample.rollout_id if base_sample.rollout_id is not None else base_sample.index
    turn_sample.loss_mask = [1] * len(response_ids)
    turn_sample.metadata = dict(turn_sample.metadata or {})
    env_extra_info = env_result.get("env_extra_info")
    if not isinstance(env_extra_info, dict):
        env_state = env_result.get("env_state", env_result)
        env_extra_info = _extract_env_extra_info(env_state)
    turn_sample.metadata.update(
        {
            "turn_idx": turn_idx,
            "env_result": env_result,
            "env_extra_info": env_extra_info,
        }
    )
    # Populate speculative-decoding / prefix-cache stats from the engine meta_info
    # so rollout/spec_accept_rate and rollout/prefix_cache_hit_rate are not silently 0.
    # Only the stat sub-updates are applied here (not the full update_from_meta_info)
    # so the turn's own status logic above is preserved. .add() accumulates across
    # turns, matching partial-rollout semantics.
    if meta_info is not None:
        if getattr(args, "sglang_speculative_algorithm", None):
            turn_sample.spec_info.add(meta_info=meta_info)
        turn_sample.prefix_cache_info.add(meta_info=meta_info)
        # Preserve SGLang's response-side version so the persistent fully-async
        # worker can stamp the policy that actually generated this turn. This
        # can differ from its submission snapshot when a request waited across
        # a pause -> weight update -> continue boundary.
        if "weight_version" in meta_info:
            turn_sample.weight_versions.append(meta_info["weight_version"])
        # V4 MoE routing replay: decode this call's routed-expert indices into the
        # turn sample (mirrors the default rollout, sglang_rollout.py; row count
        # must be len(tokens)-1 per the upstream shape contract, THUDM/slime
        # 1b73ddc1). fill_routing_replay requires this on EVERY sample when
        # --use-rollout-routing-replay is set.
        if getattr(args, "use_rollout_routing_replay", False) and "routed_experts" in meta_info:
            turn_sample.rollout_routed_experts = _decode_routed_experts(
                meta_info,
                token_count=len(turn_sample.tokens) - 1,
                num_layers=args.num_layers,
                expected_topk=getattr(args, "moe_router_topk", None),
            )
    return turn_sample


def _pad_turn_samples(
    output_samples: list[Sample],
    base_sample: Sample,
    *,
    max_turns: int,
    pad_token_id: int | None,
    pad_token: str | None,
    predictive_top_k: int = 0,
) -> list[Sample]:
    if pad_token_id is None or pad_token is None:
        raise ValueError("CUDA kernel agent turn padding requires tokenizer pad_token_id or eos_token_id.")

    samples_by_turn = {
        int(sample.metadata["turn_idx"]): sample
        for sample in output_samples
        if isinstance(sample.metadata, dict) and "turn_idx" in sample.metadata
    }
    padded_samples = list(output_samples)
    for turn_idx in range(max_turns):
        if turn_idx in samples_by_turn:
            continue

        fake_sample = deepcopy(base_sample)
        # Keep one prompt token before the dummy response token so Megatron can
        # produce a response log-prob. A one-token total sequence has no previous
        # token to score, which makes CP=1 return an empty train log-prob.
        fake_sample.tokens = [pad_token_id, pad_token_id]
        fake_sample.response = pad_token
        fake_sample.response_length = 1
        fake_sample.rollout_log_probs = [0.0]
        if predictive_top_k:
            (
                fake_sample.rollout_topk_token_ids,
                fake_sample.rollout_topk_log_probs,
                fake_sample.rollout_topk_valid_mask,
            ) = _empty_predictive_support(1, predictive_top_k)
        else:
            fake_sample.rollout_topk_token_ids = None
            fake_sample.rollout_topk_log_probs = None
            fake_sample.rollout_topk_valid_mask = None
        fake_sample.reward = 0.0
        fake_sample.status = Sample.Status.COMPLETED
        # V4 routing replay: a pad turn has len(tokens)-1 == 0 replayable tokens,
        # so it carries an EMPTY (0, num_layers, topk) routed array — shape taken
        # from any real turn — satisfying fill_routing_replay's per-sample
        # invariant without influencing training (loss_mask 0, remove_sample).
        for real in output_samples:
            routed = getattr(real, "rollout_routed_experts", None)
            if routed is not None:
                fake_sample.rollout_routed_experts = routed[:0]
                break
        fake_sample.rollout_id = base_sample.rollout_id if base_sample.rollout_id is not None else base_sample.index
        fake_sample.loss_mask = [0]
        fake_sample.remove_sample = True
        fake_sample.metadata = dict(fake_sample.metadata or {})
        fake_sample.metadata.update(
            {
                "turn_idx": turn_idx,
                "is_pad_turn": True,
                "remove_reason": "pad_turn",
            }
        )
        padded_samples.append(fake_sample)

    return sorted(padded_samples, key=lambda sample: int(sample.metadata["turn_idx"]))


def _get_abort_padding(args) -> tuple[int | None, int | None, str | None]:
    if not bool(getattr(args, "use_multi_turn", False) and getattr(args, "padding_turns", False)):
        return None, None, None

    max_turns = getattr(args, "max_turns", None)
    if max_turns is None:
        return None, None, None
    max_turns = int(max_turns)

    try:
        state = GenerateState(args)
        pad_token_id = state.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = state.tokenizer.eos_token_id
        pad_token = state.tokenizer.pad_token or state.tokenizer.eos_token
        if pad_token_id is None:
            pad_token_id = 0
        if pad_token is None:
            pad_token = state.tokenizer.decode([pad_token_id], skip_special_tokens=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("CUDA agent abort padding fell back to token 0: %s", exc)
        pad_token_id = 0
        pad_token = ""

    return max_turns, pad_token_id, pad_token


def _abort_result(args, sample: Sample, abort_reason: str, elapsed_sec: float) -> Sample | list[Sample]:
    metadata = dict(sample.metadata or {})
    num_turns_completed = int(metadata.get("num_turns_completed", 0) or 0)
    total_model_time = float(metadata.get("total_model_time", 0.0) or 0.0)
    total_env_time = float(metadata.get("total_env_time", 0.0) or 0.0)
    progress = {
        "finish_reason": "aborted",
        "abort_reason": abort_reason,
        "total_request_time": max(float(elapsed_sec), total_model_time + total_env_time),
        "num_turns_completed": num_turns_completed,
        "total_model_time": total_model_time,
        "total_env_time": total_env_time,
    }

    aborted = deepcopy(sample)
    aborted.tokens = [0, 0]
    aborted.response = ""
    aborted.response_length = 1
    aborted.rollout_log_probs = [0.0]
    predictive_top_k = int(getattr(args, "dppo_predictive_top_k", 0) or 0)
    if predictive_top_k:
        (
            aborted.rollout_topk_token_ids,
            aborted.rollout_topk_log_probs,
            aborted.rollout_topk_valid_mask,
        ) = _empty_predictive_support(1, predictive_top_k)
    else:
        aborted.rollout_topk_token_ids = None
        aborted.rollout_topk_log_probs = None
        aborted.rollout_topk_valid_mask = None
    aborted.reward = 0.0
    aborted.status = Sample.Status.ABORTED
    aborted.rollout_id = sample.rollout_id if sample.rollout_id is not None else sample.index
    aborted.loss_mask = [0]
    aborted.remove_sample = True
    max_turns_for_abort = getattr(args, "max_turns", None)
    if max_turns_for_abort is not None:
        turn_idx = min(num_turns_completed, max(0, int(max_turns_for_abort) - 1))
    else:
        turn_idx = max(num_turns_completed, 0)
    aborted.metadata = {**metadata, **progress, "turn_idx": turn_idx, "remove_reason": "aborted"}

    if not getattr(args, "use_multi_turn", False):
        return aborted

    output_samples = [aborted]
    max_turns, pad_token_id, pad_token = _get_abort_padding(args)
    if max_turns is not None:
        output_samples = _pad_turn_samples(
            output_samples,
            aborted,
            max_turns=max_turns,
            pad_token_id=pad_token_id,
            pad_token=pad_token,
            predictive_top_k=predictive_top_k,
        )
    return postprocess_turn_samples(args, output_samples, finish_reason="aborted")


async def generate(args, sample: Sample, sampling_params: dict[str, Any]) -> Sample | list[Sample]:
    started_at = time.monotonic()
    try:
        async with asyncio.timeout(KERNEL_AGENT_GENERATE_GUARD_SEC):
            return await _generate_impl(args, sample, sampling_params)
    except asyncio.TimeoutError:
        elapsed_sec = time.monotonic() - started_at
        task_id = (sample.metadata or {}).get("task_id")
        cancel_sent = False
        if task_id:
            try:
                cancel_sent = await cancel_kernel_eval(args, str(task_id), CUDA_AGENT_CONFIGS["env"])
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Failed to cancel KernelGYM task after generate timeout: task_id=%s error=%s", task_id, exc
                )
        logger.warning(
            "CUDA agent generate timed out after %.1fs (guard=%ss, task_id=%s, cancel_sent=%s)",
            elapsed_sec,
            KERNEL_AGENT_GENERATE_GUARD_SEC,
            task_id,
            cancel_sent,
        )
        return _abort_result(args, sample, "wall_clock_timeout", elapsed_sec)
    except Exception as exc:  # noqa: BLE001
        elapsed_sec = time.monotonic() - started_at
        logger.exception("CUDA agent generate failed after %.1fs: %s", elapsed_sec, exc)
        return _abort_result(args, sample, f"exception:{type(exc).__name__}", elapsed_sec)


async def _generate_impl(args, sample: Sample, sampling_params: dict[str, Any]) -> Sample | list[Sample]:
    """Generate CUDA-kernel multi-turn rollouts.

    This follows the drkernel-style structure: each assistant turn becomes one
    training Sample. Environment feedback is appended to the conversation
    messages and therefore becomes part of the next turn prompt, not part of the
    current turn response.
    """

    state = GenerateState(args)
    messages = _as_messages(sample.prompt)
    max_turns = getattr(args, "max_turns", None)
    if max_turns is None:
        raise ValueError("--max-turns must be set for CUDA kernel agent rollout")
    max_turns = int(max_turns)
    padding_turns = bool(getattr(args, "use_multi_turn", False) and getattr(args, "padding_turns", False))
    pad_token_id = None
    pad_token = None
    if padding_turns:
        pad_token_id = state.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = state.tokenizer.eos_token_id
        if pad_token_id is None:
            raise ValueError("CUDA kernel agent turn padding requires tokenizer pad_token_id or eos_token_id.")
        pad_token = state.tokenizer.pad_token or state.tokenizer.eos_token
        if pad_token is None:
            pad_token = state.tokenizer.decode([pad_token_id], skip_special_tokens=False)
    template = _get_tool_response_template(state)
    output_samples: list[Sample] = []
    log_rollout_info = bool(CUDA_AGENT_CONFIGS.get("log_rollout_info", True))
    should_log = _should_log_rollout(sample) if log_rollout_info else False
    log_first_rollout = _claim_first_rollout_log()
    turn_logs: list[dict[str, Any]] = []
    finish_reason = "max_turns"

    for turn_idx in range(max_turns):
        prompt_text = state.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **state.apply_chat_template_kwargs,
        )
        prompt_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        max_context_len = getattr(args, "rollout_max_context_len", None)
        if max_context_len is not None and len(prompt_ids) >= max_context_len:
            sample.status = Sample.Status.TRUNCATED
            finish_reason = "prompt_truncated"
            logger.warning("CUDA agent prompt exceeds context length at turn %s: %s", turn_idx, len(prompt_ids))
            break
        turn_sampling_params = _sampling_params_for_prompt_context(args, sampling_params, len(prompt_ids))
        if int(turn_sampling_params.get("max_new_tokens", 0) or 0) <= 0:
            sample.status = Sample.Status.TRUNCATED
            finish_reason = "response_budget_exhausted"
            logger.warning(
                "CUDA agent response budget exhausted at turn %s: prompt_tokens=%s max_context_len=%s",
                turn_idx,
                len(prompt_ids),
                max_context_len,
            )
            break

        payload = {
            "input_ids": prompt_ids,
            "sampling_params": turn_sampling_params,
            "return_logprob": True,
        }
        predictive_top_k = int(getattr(args, "dppo_predictive_top_k", 0) or 0)
        if predictive_top_k < 0:
            raise ValueError(f"dppo_predictive_top_k must be non-negative, got {predictive_top_k}")
        if predictive_top_k:
            payload["top_logprobs_num"] = predictive_top_k
        # V4 MoE routing replay: ask the engine for the per-token routed-expert
        # indices so the train side can replay rollout routing (same request the
        # default slime rollout makes, sglang_rollout.py). Each turn is a
        # standalone Sample (tokens = this call's prompt+response), so the
        # per-call payload aligns with the turn sample 1:1.
        if getattr(args, "use_rollout_routing_replay", False):
            payload["return_routed_experts"] = True
        # Route to the currently-served (alternating) LoRA adapter, mirroring the
        # default slime rollout (sglang_rollout.py generate). Without this the
        # USE_LORA_WEIGHT_SYNC path would serve the base model on this custom
        # rollout. The active name is refreshed onto the shared GenerateState by
        # the RolloutManager each step; None -> base-only (unset lora_path).
        lora_path = _rollout_lora_path(args, state.active_lora_name)
        if lora_path is not None:
            payload["lora_path"] = lora_path
        url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
        model_started_at = time.monotonic()
        output = await post(url, payload, max_retries=KERNEL_AGENT_GENERATE_MAX_RETRIES)
        model_time = time.monotonic() - model_started_at
        finish_type = output["meta_info"]["finish_reason"]["type"]
        if finish_type == "abort":
            sample.status = Sample.Status.ABORTED
            finish_reason = "model_abort"
            _update_sample_progress(sample, turn_logs, finish_reason, abort_reason=finish_reason)
            _log_rollout_info(
                sample,
                messages,
                turn_logs,
                finish_reason,
                should_log=should_log,
                is_slowest=False,
                log_first_rollout=log_first_rollout,
            )
            if padding_turns:
                output_samples = _pad_turn_samples(
                    output_samples,
                    sample,
                    max_turns=max_turns,
                    pad_token_id=pad_token_id,
                    pad_token=pad_token,
                    predictive_top_k=predictive_top_k,
                )
            output_samples = postprocess_turn_samples(
                args,
                output_samples,
                finish_reason=finish_reason,
            )
            if getattr(args, "use_multi_turn", False):
                return output_samples
            return output_samples[-1] if output_samples else sample

        predictive_support = None
        if predictive_top_k:
            (
                response_ids,
                log_probs,
                support_token_ids,
                support_log_probs,
                support_valid_mask,
            ) = _extract_predictive_support(output["meta_info"], predictive_top_k)
            predictive_support = (support_token_ids, support_log_probs, support_valid_mask)
        else:
            token_logprobs = output["meta_info"].get("output_token_logprobs", [])
            response_ids = [item[1] for item in token_logprobs]
            log_probs = [item[0] for item in token_logprobs]
        response = output["text"]
        if not response_ids:
            if predictive_top_k and response:
                raise ValueError(
                    "SGLang returned non-empty response text without output_token_logprobs "
                    "while predictive-mask support is enabled"
                )
            response_ids = state.tokenizer(response, add_special_tokens=False)["input_ids"]
            log_probs = [0.0] * len(response_ids)

        status = Sample.Status.TRUNCATED if finish_type == "length" else Sample.Status.COMPLETED
        env_started_at = time.monotonic()
        env_result = await cuda_kernel_env(
            args,
            sample,
            response,
            turn_idx,
        )
        env_time = time.monotonic() - env_started_at
        turn_sample = _sample_for_turn(
            sample,
            prompt_ids=prompt_ids,
            response=response,
            response_ids=response_ids,
            log_probs=log_probs,
            reward=None,
            status=status,
            turn_idx=turn_idx,
            env_result=env_result,
            args=args,
            meta_info=output["meta_info"],
            predictive_support=predictive_support,
        )
        turn_sample.metadata["model_time"] = model_time
        turn_sample.metadata["env_time"] = env_time
        turn_reward = await reward_func(args, turn_sample)
        turn_sample.reward = turn_reward
        turn_log = {
            "turn_idx": turn_idx,
            "task_id": turn_sample.metadata.get("task_id"),
            "model_time": model_time,
            "env_time": env_time,
            "prompt_tokens": len(prompt_ids),
            "max_new_tokens": turn_sampling_params.get("max_new_tokens"),
            "response_tokens": len(response_ids),
            "finish_type": finish_type,
            "prompt": prompt_text,
            "response": response,
            "env_result": env_result,
            "reward": turn_reward,
            "format_feedback": None,
        }
        turn_logs.append(turn_log)
        output_samples.append(turn_sample)
        _update_sample_progress(sample, turn_logs, "running")

        messages.append(
            {
                "role": "assistant",
                "content": response,
            }
        )

        format_feedback = _apply_feedback_template(env_result, template)
        turn_log["format_feedback"] = format_feedback

        if _is_done(env_result, turn_idx, max_turns):
            finish_reason = "env_done" if turn_idx + 1 < max_turns else "max_turns"
            break

        messages.append({"role": "user", "content": format_feedback})

    total_request_time = sum(
        float(item.get("model_time", 0.0)) + float(item.get("env_time", 0.0)) for item in turn_logs
    )
    _update_sample_progress(sample, turn_logs, finish_reason)
    is_slowest = (
        await _is_slowest_multiturn(args, sample, total_request_time) if log_rollout_info and turn_logs else False
    )
    _log_rollout_info(
        sample,
        messages,
        turn_logs,
        finish_reason,
        should_log=should_log,
        is_slowest=is_slowest,
        log_first_rollout=log_first_rollout,
        total_request_time=total_request_time,
    )
    if padding_turns:
        output_samples = _pad_turn_samples(
            output_samples,
            sample,
            max_turns=max_turns,
            pad_token_id=pad_token_id,
            pad_token=pad_token,
            predictive_top_k=int(getattr(args, "dppo_predictive_top_k", 0) or 0),
        )
    output_samples = postprocess_turn_samples(
        args,
        output_samples,
        finish_reason=finish_reason,
    )
    if getattr(args, "use_multi_turn", False):
        return output_samples
    return output_samples[-1] if output_samples else sample


async def reward_func(args, samples: Sample | list[Sample], **kwargs):
    """Compute reward from the CUDA env response collected during generation."""

    def get_reward(sample: Sample):
        if sample.reward is not None:
            return sample.reward
        metadata = dict(sample.metadata or {})
        env_result = metadata.get("env_result") if isinstance(metadata.get("env_result"), dict) else {}
        env_state = env_result.get("env_state") if isinstance(env_result.get("env_state"), dict) else {}
        reward_details = calculate_reward_speedup(env_state, CUDA_AGENT_CONFIGS["reward"])

        partial_applied = bool(reward_details["partial_credit_output_mismatch"])
        partial_reason = str(reward_details["partial_credit_output_mismatch_reason"])
        metadata.update(
            {
                "partial_credit_output_mismatch": partial_applied,
                "partial_credit_output_mismatch_reason": partial_reason,
            }
        )
        env_extra_info = metadata.get("env_extra_info")
        if isinstance(env_extra_info, dict):
            env_extra_info["partial_credit_output_mismatch"] = partial_applied
            env_extra_info["partial_credit_output_mismatch_reason"] = partial_reason
        sample.metadata = metadata
        return float(reward_details["reward"])

    if isinstance(samples, list):
        return [get_reward(sample) for sample in samples]
    return get_reward(samples)
