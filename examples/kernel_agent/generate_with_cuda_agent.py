import asyncio
import json
import logging
import random
import time
from copy import deepcopy
from typing import Any

try:
    import ray
except ImportError:
    ray = None

from slime.rollout.sglang_rollout import GenerateState
from slime.utils.http_utils import post
from slime.utils.types import Sample

try:
    from .config import CUDA_AGENT_CONFIGS
    from .kernel_response import run_kernel_eval
    from .kernel_reward import calculate_reward
    from .utils import normalize_env_feedback, postprocess_turn_samples, precheck_response, split_think_response
except ImportError:
    from config import CUDA_AGENT_CONFIGS
    from kernel_response import run_kernel_eval
    from kernel_reward import calculate_reward

    from utils import normalize_env_feedback, postprocess_turn_samples, precheck_response, split_think_response

logger = logging.getLogger(__name__)


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


def _get_tool_response_template(state: GenerateState) -> str:
    template = (state.multi_turn_templates or {}).get("tool_response")
    if template is None:
        logger.warning("multi-turn tool_response template is not set; using built-in CUDA agent prompt template.")
        return DEFAULT_TOOL_RESPONSE_TEMPLATE
    return template


def _truncate_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    keep = max_chars // 2
    return text[:keep] + "...(truncated)..." + text[-keep:]


def _apply_feedback_template(env_result: dict[str, Any], template: str) -> str:
    payload = env_result.get("env_state") or env_result.get("reward_extra_info") or env_result
    try:
        feedback = json.dumps(payload, ensure_ascii=False, indent=2)
    except TypeError:
        feedback = str(payload)

    max_chars = int(CUDA_AGENT_CONFIGS["max_feedback_chars"])
    # max_chars <= 0 means no truncation
    feedback = _truncate_middle(feedback, max_chars)
    return template.format(feedback=feedback)


def _json_dumps_for_log(payload: Any) -> str:
    try:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    except TypeError:
        return str(payload)


def _get_env_state(env_result: dict[str, Any]) -> dict[str, Any]:
    env_state = env_result.get("env_state") if isinstance(env_result, dict) else None
    return env_state if isinstance(env_state, dict) else {}


def _should_log_multiturn(sample: Sample) -> bool:
    if not logger.isEnabledFor(logging.INFO):
        return False

    metadata = sample.metadata or {}
    if metadata.get("log_multi_turn") or metadata.get("should_log"):
        return True

    sample_rate = float(CUDA_AGENT_CONFIGS.get("log_multi_turn_sample_rate", 0.0) or 0.0)
    if sample_rate <= 0:
        return False
    if sample_rate >= 1:
        return True
    return random.random() < sample_rate


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


def _log_multiturn_messages(
    sample: Sample,
    messages: list[dict[str, Any]],
    turn_logs: list[dict[str, Any]],
    finish_reason: str,
    is_slowest: bool = False,
    total_request_time: float | None = None,
) -> None:
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

    log_max_chars = int(CUDA_AGENT_CONFIGS.get("max_feedback_chars", 0) or 0)

    for item in turn_logs:
        env_state = item.get("env_state") if isinstance(item.get("env_state"), dict) else {}
        reward = item.get("reward")
        if reward is None:
            reward = calculate_reward(item.get("env_result", {}), CUDA_AGENT_CONFIGS["reward"])
        logger.info(
            "[cuda_agent][turn %s] model_time=%.3fs env_time=%.3fs prompt_tokens=%s response_tokens=%s "
            "finish_type=%s status=%s error=%s speedup=%s correctness=%s compiled=%s reward=%s",
            item.get("turn_idx"),
            float(item.get("model_time", 0.0)),
            float(item.get("env_time", 0.0)),
            item.get("prompt_tokens"),
            item.get("response_tokens"),
            item.get("finish_type"),
            env_state.get("status"),
            env_state.get("error"),
            env_state.get("speedup"),
            env_state.get("correctness"),
            env_state.get("compiled"),
            reward,
        )
        logger.info(
            "[cuda_agent][turn %s] prompt:\n%s",
            item.get("turn_idx"),
            _truncate_middle(str(item.get("prompt", "")), log_max_chars),
        )
        response_think, response_content = split_think_response(str(item.get("response", "")))
        if response_think is not None:
            logger.info(
                "[cuda_agent][turn %s] response_think:\n%s",
                item.get("turn_idx"),
                _truncate_middle(response_think, log_max_chars),
            )
        logger.info(
            "[cuda_agent][turn %s] response_content:\n%s",
            item.get("turn_idx"),
            _truncate_middle(response_content, log_max_chars),
        )
        # logger.info(
        #     "[cuda_agent][turn %s] env_feedback:\n%s",
        #     item.get("turn_idx"),
        #     _truncate_middle(_json_dumps_for_log(env_state), log_max_chars),
        # )
        if item.get("format_feedback") is not None:
            logger.info(
                "[cuda_agent][turn %s] format_feedback:\n%s",
                item.get("turn_idx"),
                _truncate_middle(str(item.get("format_feedback", "")), log_max_chars),
            )

    logger.info(
        "[cuda_agent][multi_turn] messages:\n%s",
        _truncate_middle(_json_dumps_for_log(messages), log_max_chars),
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


def _get_label_value(sample: Sample, key: str) -> Any:
    if isinstance(sample.label, dict):
        value = sample.label.get(key)
        if value is not None:
            return value
    if isinstance(sample.metadata, dict):
        return sample.metadata.get(key)
    return None


def _require_env_value(mapping: dict[str, Any], key: str, path: str) -> Any:
    if key not in mapping:
        raise KeyError(f"env_result missing required field for env_extra_info: {path}.{key}")
    return mapping[key]


def _extract_env_extra_info(env_result: dict[str, Any]) -> dict[str, Any]:
    env_state = _require_env_value(env_result, "env_state", "env_result")
    if not isinstance(env_state, dict):
        raise TypeError("env_result.env_state must be a dict for env_extra_info extraction.")

    metadata = _require_env_value(env_state, "metadata", "env_result.env_state")
    if not isinstance(metadata, dict):
        raise TypeError("env_result.env_state.metadata must be a dict for env_extra_info extraction.")

    num_custom_kernels = metadata.get("num_custom_kernels", 0)
    num_total_kernels = metadata.get("num_total_kernels", 0)
    custom_kernel_time = metadata.get(
        "custom_kernel_cuda_time_in_profiling_us",
        metadata.get("custom_kernel_time_in_profiling_us", 0),
    )
    total_kernel_time = metadata.get("total_kernel_run_time_in_profiling_us", 0)
    num_coverage = 0.0
    if float(num_total_kernels) > 0:
        num_coverage = float(num_custom_kernels) / float(num_total_kernels)
    time_coverage = 0.0
    if float(total_kernel_time) > 0:
        time_coverage = float(custom_kernel_time) / float(total_kernel_time)

    return {
        "time_coverage": float(f"{time_coverage:.2f}"),
        "num_coverage": float(f"{num_coverage:.2f}"),
        "correctness": _require_env_value(env_state, "correctness", "env_result.env_state"),
        "compilation": _require_env_value(env_state, "compiled", "env_result.env_state"),
        "speedup": _require_env_value(env_state, "speedup", "env_result.env_state"),
        "decoy_kernel": _require_env_value(env_state, "decoy_kernel", "env_result.env_state"),
        "precheck": env_state.get("precheck"),
    }


async def cuda_kernel_env(
    args,
    sample: Sample,
    response: str,
    turn_idx: int,
) -> dict[str, Any]:
    entry_point = _get_label_value(sample, "entry_point")
    if entry_point is None:
        raise ValueError("CUDA kernel env requires sample.label['entry_point'].")
    do_precheck = bool(getattr(args, "do_precheck", True))
    kernel_backend = args.kernel_backend
    reference_backend = getattr(args, "reference_backend", "torch")
    if do_precheck:
        precheck_entry_point = f"{entry_point}New"
        precheck_result = precheck_response(response, precheck_entry_point, kernel_backend)
        if precheck_result is not None:
            precheck_result = dict(precheck_result)
            precheck_result.setdefault("decoy_kernel", None)
            return {"env_state": precheck_result, "reward_extra_info": precheck_result}

    payload = {
        "response": response,
        "ground_truth": _get_label_value(sample, "ground_truth"),
        "kernel_backend": kernel_backend,
        "reference_backend": reference_backend,
        "entry_point": entry_point,
        "uuid": (sample.metadata or {}).get("uuid"),
        "return_full_state": True,
        "metadata": sample.metadata,
        "turn_idx": turn_idx,
    }
    env_result = await run_kernel_eval(args, sample, payload, CUDA_AGENT_CONFIGS["env"])
    raw_env_state = _get_env_state(env_result) or env_result
    normalized_env_state = normalize_env_feedback(raw_env_state)
    if do_precheck:
        normalized_env_state.setdefault("precheck", "passed")
    return {**env_result, "env_state": normalized_env_state, "reward_extra_info": normalized_env_state}


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
) -> Sample:
    turn_sample = deepcopy(base_sample)
    turn_sample.tokens = prompt_ids + response_ids
    turn_sample.response = response
    turn_sample.response_length = len(response_ids)
    turn_sample.rollout_log_probs = log_probs
    turn_sample.reward = reward
    turn_sample.status = status
    turn_sample.loss_mask = [1] * len(response_ids)
    turn_sample.metadata = dict(turn_sample.metadata or {})
    turn_sample.metadata.update(
        {
            "turn_idx": turn_idx,
            "env_result": env_result,
            "env_extra_info": _extract_env_extra_info(env_result),
        }
    )
    return turn_sample


def _pad_turn_samples(
    output_samples: list[Sample],
    base_sample: Sample,
    *,
    max_turns: int,
    pad_token_id: int | None,
    pad_token: str | None,
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
        fake_sample.tokens = [pad_token_id]
        fake_sample.response = pad_token
        fake_sample.response_length = 1
        fake_sample.rollout_log_probs = [0.0]
        fake_sample.reward = 0.0
        fake_sample.status = Sample.Status.COMPLETED
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


async def generate(args, sample: Sample, sampling_params: dict[str, Any]) -> Sample | list[Sample]:
    """Generate CUDA-kernel multi-turn rollouts.

    This follows the drkernel-style structure: each assistant turn becomes one
    training Sample. Environment feedback is appended to the conversation
    messages and therefore becomes part of the next turn prompt, not part of the
    current turn response.
    """

    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"
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
    should_log = _should_log_multiturn(sample)
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

        payload = {
            "input_ids": prompt_ids,
            "sampling_params": sampling_params,
            "return_logprob": True,
        }
        model_started_at = time.monotonic()
        output = await post(url, payload)
        model_time = time.monotonic() - model_started_at
        finish_type = output["meta_info"]["finish_reason"]["type"]
        if finish_type == "abort":
            sample.status = Sample.Status.ABORTED
            finish_reason = "model_abort"
            _log_multiturn_messages(sample, messages, turn_logs, finish_reason)
            if padding_turns:
                output_samples = _pad_turn_samples(
                    output_samples,
                    sample,
                    max_turns=max_turns,
                    pad_token_id=pad_token_id,
                    pad_token=pad_token,
                )
            output_samples = postprocess_turn_samples(
                args,
                output_samples,
                finish_reason=finish_reason,
            )
            if getattr(args, "use_multi_turn", False):
                return output_samples
            return output_samples[-1] if output_samples else sample

        token_logprobs = output["meta_info"].get("output_token_logprobs", [])
        response_ids = [item[1] for item in token_logprobs]
        log_probs = [item[0] for item in token_logprobs]
        response = output["text"]
        if not response_ids:
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
        env_state = _get_env_state(env_result)
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
        )
        turn_sample.metadata["model_time"] = model_time
        turn_sample.metadata["env_time"] = env_time
        turn_reward = await reward_func(args, turn_sample)
        turn_sample.reward = turn_reward
        turn_log = {
            "turn_idx": turn_idx,
            "model_time": model_time,
            "env_time": env_time,
            "prompt_tokens": len(prompt_ids),
            "response_tokens": len(response_ids),
            "finish_type": finish_type,
            "prompt": prompt_text,
            "response": response,
            "env_state": env_state,
            "env_result": env_result,
            "reward": turn_reward,
            "format_feedback": None,
        }
        turn_logs.append(turn_log)
        output_samples.append(turn_sample)

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
    is_slowest = await _is_slowest_multiturn(args, sample, total_request_time) if turn_logs else False
    if should_log or is_slowest:
        _log_multiturn_messages(
            sample,
            messages,
            turn_logs,
            finish_reason,
            is_slowest=is_slowest,
            total_request_time=total_request_time,
        )
    if padding_turns:
        output_samples = _pad_turn_samples(
            output_samples,
            sample,
            max_turns=max_turns,
            pad_token_id=pad_token_id,
            pad_token=pad_token,
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
        return calculate_reward(sample.metadata.get("env_result", {}), CUDA_AGENT_CONFIGS["reward"])

    if isinstance(samples, list):
        return [get_reward(sample) for sample in samples]
    return get_reward(samples)
