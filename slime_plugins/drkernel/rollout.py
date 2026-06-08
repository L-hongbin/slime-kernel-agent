from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
from argparse import Namespace
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment
from tqdm import tqdm
from transformers import AutoTokenizer

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.rollout.sglang_rollout import GenerateState, abort, generate_and_rm
from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.misc import load_function
from slime.utils.processing_utils import load_processor, load_tokenizer
from slime.utils.types import Sample

from .eval_throttle import get_positive_int_env, run_eval_coro

logger = logging.getLogger(__name__)

__all__ = [
    "DrKernelPromptRenderer",
    "PromptRenderResult",
    "format_kernelgym_feedback",
    "generate_multi_turn_eval_sample",
    "generate_rollout",
    "generate_rollout_async",
]


# Fixed composable prompt pack for this rollout plugin (not CLI-configurable).
_PROMPT_CONFIG_PATH = Path(__file__).resolve().parent / "prompt_templates" / "prompts_v1.yaml"


@dataclass(frozen=True)
class PromptRenderResult:
    prompt: str
    chosen: dict[str, Any]


def _get_arg(args: Namespace, name: str, default: Any = None) -> Any:
    return getattr(args, name, default)


def ensure_no_pre_chat_template(args: Namespace) -> None:
    if args.apply_chat_template:
        raise ValueError(
            "DrKernel rollout applies tokenizer chat template after dynamic prompt rendering; remove --apply-chat-template from the run args."
        )


class DrKernelPromptRenderer:
    """Render DrKernel first-turn prompts from the composable prompt YAML.

    The runtime YAML intentionally describes a composable search space:
    - role is selected independently;
    - backend selects the per-backend template set (first-turn body + tool-response body);
    - compiler/GPU info is injected from args/metadata as plain strings.

    Legacy equivalence is covered in tests by fixing specific role/backend pairs.
    """

    def __init__(self, hf_checkpoint: str) -> None:
        self.template_root = _PROMPT_CONFIG_PATH.parent
        self.config: dict = yaml.safe_load(_PROMPT_CONFIG_PATH.read_text(encoding="utf-8"))
        # self.profile_name = "drkernel_v1"
        self.profile_name = "drkernel_v1_tvm_ffi"

        profiles = self.config.get("profiles", {})
        if self.profile_name not in profiles:
            raise KeyError(f"Unknown DrKernel prompt profile: {self.profile_name}")
        self.profile: dict = profiles[self.profile_name]

        self.jinja_env = Environment(keep_trailing_newline=False)
        self._text_cache: dict[Path, str] = {}
        self.tokenizer: AutoTokenizer = load_tokenizer(hf_checkpoint, trust_remote_code=True)

    def render_sample(self, args: Namespace, sample: Any, rollout_id: int) -> PromptRenderResult:
        metadata = sample.metadata

        problem = sample.prompt
        if not isinstance(problem, str):
            raise TypeError(
                "DrKernel prompt renderer expects a raw string problem. Do not apply chat template before custom rollout."
            )

        role_candidate = self._select_candidate(
            rollout_id=rollout_id,
            sample=sample,
            slot_name="role",
        )
        backend_candidate = self._select_candidate(
            rollout_id=rollout_id,
            sample=sample,
            slot_name="backend",
        )

        # Render each fragment through jinja before string-injecting into the outer
        # layout. Without this pass, jinja constructs in the fragment ({# ... #}
        # comments, {{ ... }} placeholders) survive as literal text in the final
        # prompt — exactly the leak that produced the v2/v2.1/v2.2 ablation
        # confound (May 2026). Fragments currently have no placeholders, so an
        # empty render context is safe; if a fragment later needs vars, add them
        # here and in the test.
        role = self._render_template(self._load_fragment(role_candidate["text_path"]))
        backend = self._render_template(self._load_fragment(backend_candidate["first_turn_text_path"]))
        layout = self._load_fragment(self.profile["layout"])

        compiler_name = metadata.get("compiler_name") or _get_arg(args, "drkernel_compiler_name")
        gpu_name = metadata.get("gpu_name") or _get_arg(args, "drkernel_gpu_name")
        extra_environment = metadata.get("extra_environment") or _get_arg(args, "drkernel_extra_environment")

        prompt = self._render_template(
            layout,
            role=role,
            backend=backend,
            problem=problem,
            compiler_name=compiler_name,
            gpu_name=gpu_name,
            extra_environment=extra_environment,
        )

        chosen = {
            "profile": self.profile_name,
            "role": role_candidate["id"],
            "backend": backend_candidate["id"],
        }
        if compiler_name:
            chosen["compiler_name"] = compiler_name
        if gpu_name:
            chosen["gpu_name"] = gpu_name
        if extra_environment:
            chosen["extra_environment"] = extra_environment
        return PromptRenderResult(prompt=prompt, chosen=chosen)

    def render_first_turn_messages(self, args: Namespace, sample: Any, rollout_id: int) -> list[dict[str, str]]:
        """Render the first-turn user message and seed sample.metadata['messages'].

        The returned list is also stored on the sample so multi-turn callers can
        keep appending assistant responses and tool-response messages to it.
        """
        result = self.render_sample(args, sample, rollout_id)
        sample.metadata["raw_problem"] = sample.prompt
        sample.metadata["chosen_prompt_slots"] = result.chosen
        sample.metadata["drkernel_user_prompt"] = result.prompt

        messages = [{"role": "user", "content": result.prompt}]
        sample.metadata["messages"] = messages
        return messages

    def render_tool_response_message(self, args: Namespace, sample: Any, feedback: str) -> dict[str, str]:
        """Render the next user turn whose tool template is keyed off the chosen backend.

        Requires ``render_first_turn_messages`` to have populated
        ``sample.metadata['chosen_prompt_slots']['backend']`` first.
        """
        chosen = sample.metadata.get("chosen_prompt_slots") or {}
        backend_id = chosen.get("backend")
        if not backend_id:
            raise RuntimeError(
                "render_tool_response_message requires render_first_turn_messages to have run first; "
                "chosen_prompt_slots['backend'] is missing."
            )
        candidate = self._lookup_candidate("backend", backend_id)
        tool_path = candidate.get("tool_response_text_path")
        if not tool_path:
            raise KeyError(
                f"Profile {self.profile_name!r} backend candidate {backend_id!r} "
                "is missing 'tool_response_text_path'; multi-turn rendering is not configured for it."
            )
        body = self._render_template(self._load_fragment(tool_path), feedback=feedback)
        return {"role": "user", "content": body}

    def materialize_prompt(self, args: Namespace, sample: Any, messages: list[dict[str, str]]) -> Any:
        """Apply the tokenizer chat template over ``messages`` and write back ``sample.prompt``."""
        tools = sample.metadata.get("tools")
        sample.prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            **(_get_arg(args, "apply_chat_template_kwargs", {}) or {}),
        )
        if not isinstance(sample.prompt, str):
            raise TypeError("tokenizer.apply_chat_template(..., tokenize=False) must return a string.")
        if hasattr(sample, "tokens"):
            sample.tokens = []
        return sample

    def apply_to_sample(self, args: Namespace, sample: Any, rollout_id: int) -> Any:
        messages = self.render_first_turn_messages(args, sample, rollout_id)
        return self.materialize_prompt(args, sample, messages)

    def _lookup_candidate(self, slot_name: str, candidate_id: str) -> dict[str, Any]:
        slot_cfg: dict = self.profile[slot_name]
        for candidate in slot_cfg["candidates"]:
            if candidate["id"] == candidate_id:
                return candidate
        raise KeyError(f"Unknown candidate {candidate_id!r} for slot {slot_name!r} in profile {self.profile_name!r}")

    def _load_fragment(self, relative_path: str) -> str:
        path = (self.template_root / relative_path).resolve()
        if path not in self._text_cache:
            self._text_cache[path] = path.read_text(encoding="utf-8")
        return self._text_cache[path]

    def _render_template(self, text: str, **kwargs: Any) -> str:
        return self.jinja_env.from_string(text).render(**kwargs)

    def _select_candidate(
        self,
        *,
        rollout_id: int,
        sample: Any,
        slot_name: str,
    ) -> dict[str, Any]:
        slot_cfg: dict = self.profile[slot_name]
        candidates = list(slot_cfg["candidates"])

        allowed_ids = sample.metadata.get("template_allowed", {}).get(slot_name)
        if allowed_ids is not None:
            if not allowed_ids:
                raise ValueError(f"DrKernel prompt allowed list for {slot_name} is empty")
            allowed = set(allowed_ids)
            known = {candidate["id"] for candidate in candidates}
            unknown = sorted(allowed - known)
            if unknown:
                raise ValueError(
                    f"Unknown DrKernel prompt candidates for {slot_name}: {unknown}. Known: {sorted(known)}"
                )
            candidates = [candidate for candidate in candidates if candidate["id"] in allowed]

        select_mode = slot_cfg.get("select", "fixed")
        if select_mode == "fixed":
            return candidates[0]
        if select_mode == "cycle":
            basis = sample.index if sample.index is not None else rollout_id
            return candidates[int(basis) % len(candidates)]

        raise ValueError(f"Unsupported DrKernel prompt select mode for {slot_name}: {select_mode}")


@lru_cache(maxsize=1)
def _get_prompt_renderer(hf_checkpoint: str) -> DrKernelPromptRenderer:
    """Return a cached renderer with tokenizer loaded during construction.

    Importing this plugin remains side-effect-light, while rollout/eval paths
    pay tokenizer initialization once before samples are rendered.
    """
    return DrKernelPromptRenderer(hf_checkpoint)


def _train_sglang_context_len_for_sampling(args: Namespace) -> int:
    context_len = int(args.rollout_max_context_len)
    draft_tokens = 0
    if getattr(args, "sglang_speculative_algorithm", None):
        draft_tokens = int(getattr(args, "sglang_speculative_num_draft_tokens", 0) or 0)
    return max(1, context_len - draft_tokens)


async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Any]]]
) -> Any:
    """Minimal single-turn DrKernel rollout example.

    This shows how dynamic prompts plug into slime without rewriting SGLang
    generation. The generation/reward path is still the default
    `generate_and_rm_group`.
    """
    assert args.rollout_global_dataset
    ensure_no_pre_chat_template(args)

    renderer = _get_prompt_renderer(args.hf_checkpoint)
    state = GenerateState(args)
    state.sampling_params["_slime_max_context_len"] = _train_sglang_context_len_for_sampling(args)
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )
    metric_gatherer = MetricGatherer()

    # target_data_size is the total number of valid samples to get
    target_data_size = args.rollout_batch_size

    data = []
    all_data = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            for group in samples:
                for sample in group:
                    renderer.apply_to_sample(args, sample, rollout_id)

            state.submit_generate_tasks(samples)

        # wait for the generation to finish
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group: list = task.result()

            if do_print:
                sample: Sample = group[0][0] if isinstance(group[0], list) else group[0]
                logger.info(
                    "First rollout sample: index=%s prompt_len=%d response_len=%d reward=%s label_head=%r",
                    sample.index,
                    len(sample.prompt) if isinstance(sample.prompt, str) else -1,
                    len(sample.response) if isinstance(sample.response, str) else -1,
                    sample.reward,
                    str(sample.label)[:100],
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            all_data.append(group)
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                state.remaining_batch_size -= 1
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info(
        "Finish rollout: index=%s prompt_len=%d response_len=%d reward=%s label_head=%r",
        sample.index,
        len(sample.prompt) if isinstance(sample.prompt, str) else -1,
        len(sample.response) if isinstance(sample.response, str) else -1,
        sample.reward,
        str(sample.label)[:100],
    )

    # there are still some unfinished requests, abort them
    aborted_samples = await abort(args, rollout_id)

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)
    all_samples = sorted(
        all_data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index
    )

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()
    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, data)

    # There can be circumstances where users want to process all samples including filtered ones.
    if args.rollout_all_samples_process_path is not None:
        process_func = load_function(args.rollout_all_samples_process_path)
        process_func(args, all_samples, data_source)

    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples


# ---------------------------------------------------------------------------
# Feedback summarization. See design-docs/feedback_summarization.md for the
# whitelist / summarize-priority rationale. Whitelists are intentionally narrow:
# anything not listed gets dropped. The summary char limit defaults to 1600
# (about ~400 tokens) which fits 5-10 nvcc error lines or 3-4 stack frames.
# ---------------------------------------------------------------------------

# env_state top-level keys that carry signal for the next turn.
_ENV_STATE_SIGNAL_KEYS: tuple[str, ...] = (
    "status",
    "compiled",
    "correctness",
    "decoy_kernel",
    "reference_runtime",
    "kernel_runtime",
    "speedup",
    "error_code",
)
# Priority order for finding the primary error text to summarize. The raw stderr
# (``metadata.error``) is the last fallback; all of these are consumed by the
# summarizer and do NOT appear verbatim in the rendered feedback.
_ENV_STATE_PRIMARY_ERROR_SOURCES: tuple[tuple[str, str], ...] = (
    ("metadata", "compilation_error"),
    ("metadata", "runtime_error"),
    ("metadata", "correctness_issue"),
    # ``error_during_performance`` shows up when the kernel passed correctness but the
    # perf measurement phase failed (typically a worker-side CUDA OOM during the 30
    # warmup + 50 trial runs with torch profiler enabled). Surface it before the
    # generic ``metadata.error`` fallback so the model sees the specific signal.
    ("metadata", "error_during_performance"),
    ("metadata", "error"),
    ("state", "error_message"),
    ("state", "error"),
)
# Performance metrics, only emitted when compiled AND correctness are both True.
# Field names match KernelGym ``/evaluate`` response schema observed in production
# (custom_kernel_cuda_time_coverage / num_custom_kernels — note plural).
_ENV_STATE_METRICS_KEYS: tuple[str, ...] = (
    "custom_kernel_cuda_time_coverage",
    "num_custom_kernels",
    "num_total_kernels",
    "custom_kernel_cuda_time_in_profiling_us",
    "total_kernel_cuda_time_in_profiling_us",
)
# Lightweight metadata for the prompt. ``gpu_name`` / ``device`` / ``backend``
# are intentionally dropped (redundant with prompt context or noise).
# Correctness diagnostics (max_difference / avg_difference / correctness_issue_name)
# are included when present — they give the model a concrete signal for fixing
# numerical mismatches.
_ENV_STATE_METADATA_KEYS: tuple[str, ...] = (
    "hardware",
    "compilation_error_name",
    "runtime_error_name",
    "correctness_issue_name",
    "max_difference",
    "avg_difference",
)
# Signature keys used to recognize "this dict is itself an env_state, not a
# wrapper holding env_state under a sub-key". Production KernelGym returns this
# top-level shape; the nested {"env_state": {...}} shape is a defensive fallback.
_ENV_STATE_SIGNATURE_KEYS: tuple[str, ...] = ("compiled", "correctness", "status")
DEFAULT_KERNELGYM_ERROR_SUMMARY_CHARS = 1600


def _looks_like_env_state(payload: Any) -> bool:
    """Detect if ``payload`` is itself the KernelGym env_state dict (vs a wrapper)."""
    if not isinstance(payload, dict):
        return False
    return any(key in payload for key in _ENV_STATE_SIGNATURE_KEYS)


_CHAT_STOP_MARKERS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")


def _strip_chat_stop_markers(text: Any) -> str:
    """Remove chat-template stop / role markers leaked into raw SGLang output.

    With ``skip_special_tokens=False`` + ``no_stop_trim=True`` (current sampling
    defaults), SGLang returns the assistant response with the trailing ``<|im_end|>``
    still attached. If we feed that back into ``tokenizer.apply_chat_template`` on
    the next turn the template wraps the content with another ``<|im_end|>``, so
    every T2/T3 prompt ends up containing ``<|im_end|><|im_end|>`` and any empty
    assistant turn shows up as a stray marker. Strip the known markers before
    appending the response to ``messages`` history.
    """
    if not isinstance(text, str):
        return ""
    cleaned = text
    for marker in _CHAT_STOP_MARKERS:
        cleaned = cleaned.replace(marker, "")
    return cleaned.strip()


def _truncate_text(text: Any, limit: int = 400) -> Any:
    if not isinstance(text, str):
        return text
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


def _dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def summarize_diagnostic_text(text: Any, limit: int = DEFAULT_KERNELGYM_ERROR_SUMMARY_CHARS) -> str | None:
    """Compress arbitrary diagnostic text (nvcc stderr, python traceback, etc.) into a
    short, model-actionable summary at most ``limit`` characters long.

    Multi-step regex fallback; see design-docs/feedback_summarization.md for the
    priority chain. Returns ``None`` for ``None`` input and an empty string for blank
    input. Each branch's output is independently capped at ``limit`` chars.
    """
    if text is None:
        return None
    if not isinstance(text, str):
        text = str(text)
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    precheck_match = re.search(r"(Precheck failed:[^\n]+)", cleaned)
    if precheck_match:
        return precheck_match.group(1).strip()

    syntax_match = re.search(r"(Syntax error[^\n]+)", cleaned)
    if syntax_match:
        return syntax_match.group(1).strip()

    timeout_match = re.search(r"((?:Task|Operation)[^\n]*timeout[^\n]*)", cleaned, re.IGNORECASE)
    if timeout_match:
        return timeout_match.group(1).strip()

    attribute_error_match = re.search(r"('[^'\n]+' object has no attribute '[^'\n]+')", cleaned)
    if attribute_error_match:
        return attribute_error_match.group(1).strip()

    # CUDA OOM messages don't include `error:` / `XxxError:` / `failed` keywords,
    # so the generic chains below miss them and they fall through to the verbose
    # fallback. Real KernelGym OOM strings are typically one long line ending with
    # a noisy "See documentation for Memory Management (https://...)" pointer that
    # the model doesn't need. Catch the OOM line and trim that trailer.
    oom_match = re.search(r"CUDA out of memory[^\n]*", cleaned, re.IGNORECASE)
    if oom_match:
        oom_text = oom_match.group(0)
        sentinel = re.search(r"\s+See documentation\b", oom_text, re.IGNORECASE)
        if sentinel:
            oom_text = oom_text[: sentinel.start()]
        return oom_text.strip()

    compiler_error_lines = _dedupe_preserve_order(
        line.strip()
        for line in cleaned.splitlines()
        if line.strip() and (" error:" in line.lower() or line.strip().lower().startswith("error:"))
    )
    if compiler_error_lines:
        return _truncate_text("\n".join(compiler_error_lines[:4]), limit)

    exception_lines = _dedupe_preserve_order(
        match.strip()
        for match in re.findall(
            r"([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception):[^\n]+)",
            cleaned,
        )
        if match.strip()
    )
    if exception_lines:
        return _truncate_text("\n".join(exception_lines[-3:]), limit)

    signal_lines = _dedupe_preserve_order(
        line.strip()
        for line in cleaned.splitlines()
        if line.strip() and any(token in line.lower() for token in ("failed", "error", "exception", "timeout"))
    )
    if signal_lines:
        return _truncate_text("\n".join(signal_lines[-3:]), limit)

    return _truncate_text(cleaned, limit)


def build_prompt_feedback_payload(
    env_state: dict[str, Any] | None,
    *,
    error_summary_chars: int = DEFAULT_KERNELGYM_ERROR_SUMMARY_CHARS,
) -> dict[str, Any]:
    """Filter the KernelGym ``env_state`` down to the next-turn-actionable signals.

    Output dict contains at most ~10-12 keys: whitelisted top-level evaluation signals,
    a summarized ``error_message`` (when an error exists), optional ``metrics`` (only
    when the kernel actually ran correctly), and a lightweight ``metadata`` block.

    See design-docs/feedback_summarization.md for the field-by-field whitelist.
    Returns the original ``env_state`` unchanged as a last-resort fallback if no
    whitelisted field matched.
    """
    state = env_state or {}
    raw_metadata = state.get("metadata")
    metadata: dict[str, Any] = raw_metadata if isinstance(raw_metadata, dict) else {}
    payload: dict[str, Any] = {}

    for key in _ENV_STATE_SIGNAL_KEYS:
        value = state.get(key)
        if value is not None:
            payload[key] = value

    primary_error: Any = None
    for container, key in _ENV_STATE_PRIMARY_ERROR_SOURCES:
        src = metadata if container == "metadata" else state
        candidate = src.get(key) if isinstance(src, dict) else None
        if candidate:
            primary_error = candidate
            break
    summary = summarize_diagnostic_text(primary_error, limit=error_summary_chars)
    if summary:
        payload["error_message"] = summary

    # Performance metrics are only meaningful when the kernel actually ran correctly.
    if bool(state.get("compiled")) and bool(state.get("correctness")):
        metrics: dict[str, Any] = {}
        for key in _ENV_STATE_METRICS_KEYS:
            value = metadata.get(key, state.get(key))
            if value is not None:
                metrics[key] = value
        if metrics:
            payload["metrics"] = metrics

    prompt_metadata: dict[str, Any] = {}
    for key in _ENV_STATE_METADATA_KEYS:
        value = metadata.get(key)
        if value not in (None, ""):
            prompt_metadata[key] = value
    if prompt_metadata:
        payload["metadata"] = prompt_metadata

    return payload or state


def format_kernelgym_feedback(
    sample: Any,
    *,
    max_chars: int = 0,
    error_summary_chars: int = DEFAULT_KERNELGYM_ERROR_SUMMARY_CHARS,
) -> str:
    """Build the ``{{ feedback }}`` string for a tool_response template.

    The KernelGym ``env_state`` is passed through :func:`build_prompt_feedback_payload`
    to drop noise (raw nvcc stderr, internal kg_stage_*/tm_* profiling, redundant
    task_id/reward/success) and summarize the primary error to ``error_summary_chars``.
    Other response shapes (extract_error fallback, non-dict payloads) keep their
    legacy behavior. ``max_chars > 0`` enables a final middle-truncation safety
    net but is rarely needed once the env_state path runs.
    """
    kernelgym = sample.metadata.get("kernelgym") or {}
    payload: Any = kernelgym.get("response")

    if payload is None:
        # RM short-circuited before /evaluate ran (e.g. the response had no extractable
        # kernel submission). Synthesize a minimal feedback body so the next turn sees
        # the format-error signal instead of an opaque empty turn.
        body: Any = {
            "status": "extract_error",
            "extract_error": kernelgym.get("extract_error", "unknown"),
            "reward": kernelgym.get("reward"),
        }
    elif isinstance(payload, dict):
        # KernelGym /evaluate responses arrive in two shapes:
        #   1. Top-level env_state shape: keys like status/compiled/correctness sit
        #      directly on the response dict (observed in production, this is the
        #      dominant case).
        #   2. Wrapper shape: env_state nested under ``payload["env_state"]``.
        # Handle both by sniffing the dict's signature.
        env_state = payload.get("env_state")
        if isinstance(env_state, dict) and _looks_like_env_state(env_state):
            body = build_prompt_feedback_payload(env_state, error_summary_chars=error_summary_chars)
        elif _looks_like_env_state(payload):
            body = build_prompt_feedback_payload(payload, error_summary_chars=error_summary_chars)
        else:
            # Either reward_extra_info dict or a completely unfamiliar dict; pass through.
            body = payload.get("reward_extra_info") or payload
    else:
        body = payload

    try:
        text = json.dumps(body, ensure_ascii=False)
    except TypeError:
        text = str(body)

    if max_chars > 0 and len(text) > max_chars:
        keep = max_chars // 2
        text = text[:keep] + "...(truncated)..." + text[-keep:]
    return text


def _reset_sample_for_next_turn(sample: Sample) -> None:
    """Clear per-turn rollout fields so ``generate_and_rm`` runs fresh on the next turn.

    Leaves ``sample.metadata`` alone (in particular ``messages`` and ``chosen_prompt_slots``
    must persist across turns); the per-turn ``kernelgym`` block is naturally overwritten
    by ``kernelgym_rm.evaluate_sample`` when the next turn's RM call runs.
    """
    sample.status = Sample.Status.PENDING
    sample.response = ""
    sample.response_length = 0
    sample.tokens = []
    sample.reward = None
    sample.loss_mask = None
    sample.rollout_log_probs = None
    sample.weight_versions = []


async def generate_multi_turn_eval_sample(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    *,
    renderer: DrKernelPromptRenderer,
    feedback_max_chars: int = 0,
) -> Sample:
    """Run a multi-turn eval rollout for a single sample.

    The caller is expected to have already invoked
    ``renderer.apply_to_sample(args, sample, rollout_id)`` so turn-0's ``sample.prompt``
    and ``sample.metadata["messages"]`` are seeded.

    The loop runs exactly ``args.max_turns`` iterations (no early stop on max reward).
    Each turn after the first appends the assistant response and a backend-paired
    tool_response message to ``sample.metadata["messages"]``, re-renders the prompt,
    resets the per-turn rollout fields, and re-runs ``generate_and_rm``.

    Returns the same ``Sample`` instance carrying the final turn's response/reward.
    Per-turn audit snapshots are accumulated in ``sample.metadata["turns"]`` so
    ``--dump-details`` can show the full multi-turn trajectory.
    """
    max_turns = int(getattr(args, "max_turns", 1) or 1)
    assert max_turns >= 1, "--max-turns must be >= 1 when --use-multi-turn is set"

    # Resolve the per-turn feedback summary budget from args; falls back to the
    # library default if the plugin args are not wired up.
    error_summary_chars = int(
        getattr(args, "kernelgym_error_summary_chars", DEFAULT_KERNELGYM_ERROR_SUMMARY_CHARS)
        or DEFAULT_KERNELGYM_ERROR_SUMMARY_CHARS
    )

    messages = sample.metadata.get("messages")
    assert isinstance(messages, list) and len(messages) >= 1, (
        "generate_multi_turn_eval_sample requires renderer.render_first_turn_messages "
        "to have populated sample.metadata['messages']."
    )

    turns_log: list[dict[str, Any]] = sample.metadata.setdefault("turns", [])

    # Stash the slime-private context hint so we can re-inject it every turn:
    # slime's ``_cap_sampling_params_by_context`` pops the key on each call,
    # which would silently disable the cap from turn 1 onward if we let it
    # mutate the dict in place. Re-applying it here is a zero-cost workaround
    # that does NOT require any slime upstream change.
    cap_max_context_len = sampling_params.get("_slime_max_context_len")

    for turn_idx in range(max_turns):
        if cap_max_context_len is not None:
            sampling_params["_slime_max_context_len"] = cap_max_context_len
        try:
            sample = await generate_and_rm(args, sample, sampling_params=sampling_params, evaluation=True)
        except Exception as exc:  # noqa: BLE001 — keep eval moving when one sample blows up
            # Don't let a single sample's runtime error (e.g. SGLang 400 from a context-overflow
            # edge case, KernelGym 5xx, network blip) take down the whole eval. Mark this sample
            # FAILED, record the exception text for review, and exit the multi-turn loop.
            logger.warning(
                "generate_and_rm raised %s at turn %d for sample %s; aborting multi-turn loop for this sample",
                type(exc).__name__,
                turn_idx,
                getattr(sample, "index", "?"),
            )
            sample.status = Sample.Status.FAILED
            if sample.reward is None:
                sample.reward = 0.0
            sample.metadata.setdefault("multi_turn_errors", []).append(
                {"turn_idx": turn_idx, "exc_type": type(exc).__name__, "exc_msg": str(exc)[:1000]}
            )
            turns_log.append(
                {
                    "turn_idx": turn_idx,
                    "status": "failed",
                    "reward": sample.reward,
                    "response_length": sample.response_length,
                    "prompt_snapshot": sample.prompt,
                    "response": sample.response,
                    "kernelgym": None,
                    "exception": {"type": type(exc).__name__, "msg": str(exc)[:1000]},
                }
            )
            return sample
        # DrKernel multi-turn keeps a 1:1 turn-to-sample shape; fan-out via custom_generate
        # is not used here, so unwrap defensively if we ever see a list.
        if isinstance(sample, list):
            assert (
                len(sample) == 1
            ), "generate_multi_turn_eval_sample does not support generate_and_rm returning multi-sample lists."
            sample = sample[0]

        kg = (sample.metadata.get("kernelgym") or {}).get("response") or {}
        logger.info(
            "multi_turn sample=%s turn=%d/%d reward=%s resp_len=%d compiled=%s correct=%s",
            getattr(sample, "index", "?"),
            turn_idx + 1,
            max_turns,
            sample.reward,
            sample.response_length,
            kg.get("compiled"),
            kg.get("correctness"),
        )

        turns_log.append(
            {
                "turn_idx": turn_idx,
                "status": sample.status.value if hasattr(sample.status, "value") else str(sample.status),
                "reward": sample.reward,
                "response_length": sample.response_length,
                # Full rendered prompt that the model saw this turn (post chat_template).
                "prompt_snapshot": sample.prompt,
                # Raw decoded response from SGLang for this turn (no tokenizer post-processing).
                "response": sample.response,
                "kernelgym": kg,
            }
        )

        # Always record this turn's assistant message in the conversation log so
        # ``sample.metadata['messages']`` carries the complete multi-turn dialogue
        # (including the final turn, which never appears as history for any later turn).
        # SGLang returns the response with the stop token still attached when
        # skip_special_tokens=False; if we feed that text back into the chat template,
        # it wraps it again and produces `<|im_end|><|im_end|>` in the next-turn prompt.
        messages.append({"role": "assistant", "content": _strip_chat_stop_markers(sample.response)})

        # Last turn — nothing to render or reset.
        if turn_idx + 1 >= max_turns:
            break

        # If SGLang aborted (e.g. external killswitch), don't keep submitting; carry
        # the aborted snapshot forward and let the eval aggregator see the partial result.
        if sample.status == Sample.Status.ABORTED:
            break

        # Build the next-turn user message and re-render the prompt through chat_template.
        feedback = format_kernelgym_feedback(
            sample,
            max_chars=feedback_max_chars,
            error_summary_chars=error_summary_chars,
        )
        messages.append(renderer.render_tool_response_message(args, sample, feedback=feedback))
        renderer.materialize_prompt(args, sample, messages)
        _reset_sample_for_next_turn(sample)

    return sample


EVAL_PROMPT_DATASET = {}


async def eval_rollout(args: Namespace, rollout_id: int) -> tuple[dict[str, dict[str, list[Any]]], list[list[Sample]]]:
    assert not args.group_rm, "Group RM is not supported for eval rollout"
    ensure_no_pre_chat_template(args)

    coros = []
    for dataset_cfg in getattr(args, "eval_datasets", []) or []:
        coros.append(eval_rollout_single_dataset(args, rollout_id, dataset_cfg))
    results_list = await asyncio.gather(*coros)
    results = {}
    for r in results_list:
        results.update(r)
    return RolloutFnEvalOutput(data=results), []


async def eval_rollout_single_dataset(
    args: Namespace, rollout_id: int, dataset_cfg: EvalDatasetConfig
) -> dict[str, dict[str, list[Any]]]:
    """An example to implement the eval_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        dataset_cfg: configuration of the dataset
    """
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    global EVAL_PROMPT_DATASET

    renderer = _get_prompt_renderer(args.hf_checkpoint)
    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template)
    eval_max_prompt_len = dataset_cfg.max_prompt_len
    if eval_max_prompt_len is None:
        eval_max_prompt_len = getattr(args, "eval_max_prompt_len", None)
    if cache_key not in EVAL_PROMPT_DATASET:
        processor = (
            load_processor(args.hf_checkpoint, trust_remote_code=True) if args.multimodal_keys is not None else None
        )
        EVAL_PROMPT_DATASET[cache_key] = Dataset(
            path=dataset_cfg.path,
            tokenizer=renderer.tokenizer,
            processor=processor,
            max_length=eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        )
    dataset = EVAL_PROMPT_DATASET[cache_key]

    # Quietly shave a small reserve off ``_slime_max_context_len`` so slime's default
    # cap (``max_new = max_context_len - prompt - 1`` in
    # ``slime.rollout.sglang_rollout._cap_sampling_params_by_context``) leaves enough
    # headroom for SGLang's own internal reserve. Observed: SGLang rejects requests
    # whose ``prompt_len + max_new_tokens > sglang_context_length - 3`` even though
    # the configured context window is e.g. 32768. Without this offset, multi-turn
    # prompts that creep into the (context - 5, context - 1) range get 400'd. This
    # avoids touching slime upstream and adds no extra tokenize work.
    _SGLANG_INPUT_RESERVE = 32
    eval_context_len = dataset_cfg.max_context_len
    if eval_context_len is None:
        eval_context_len = getattr(args, "eval_max_context_len", None) or getattr(
            args, "rollout_max_context_len", None
        )
    if eval_context_len is None:
        raise ValueError("eval max context length is required for DrKernel eval rollout")
    eval_max_context_len = max(1, int(eval_context_len) - _SGLANG_INPUT_RESERVE)
    base_sampling_params = dict(
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_new_tokens=dataset_cfg.max_response_len,
        _slime_max_context_len=eval_max_context_len,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    tasks = []
    # do multiple samples for eval prompts
    sample_index = 0
    eval_samples = dataset.samples
    eval_max_concurrency = get_positive_int_env("DRKERNEL_EVAL_MAX_CONCURRENCY")
    concurrency_source = "DRKERNEL_EVAL_MAX_CONCURRENCY env"
    if eval_max_concurrency == 0:
        # Default: ~1.2x the total sglang in-flight request capacity, so engines
        # stay saturated while leaving some slack for multi-turn samples that
        # are between sglang turns (waiting on KernelGym reward).
        max_running = int(getattr(args, "sglang_max_running_requests", 0) or 0)
        gpus_per_engine = int(getattr(args, "rollout_num_gpus_per_engine", 0) or 0)
        gpus_per_node = int(getattr(args, "num_gpus_per_node", 0) or 0)
        num_nodes = int(getattr(args, "actor_num_nodes", 1) or 1)
        if max_running > 0 and gpus_per_engine > 0 and gpus_per_node > 0:
            engines_per_node = max(1, gpus_per_node // gpus_per_engine)
            num_engines = engines_per_node * num_nodes
            eval_max_concurrency = int(max_running * num_engines * 2)
            concurrency_source = f"auto (max_running={max_running} x engines={num_engines} x 2)"
    eval_semaphore = asyncio.Semaphore(eval_max_concurrency) if eval_max_concurrency > 0 else None
    if eval_max_concurrency > 0:
        logger.info(
            "DRKERNEL_EVAL_MAX_CONCURRENCY=%d (%s) limiting %s eval sample concurrency",
            eval_max_concurrency,
            concurrency_source,
            dataset_cfg.name,
        )
    for _i, prompt_sample in enumerate(eval_samples):
        for j in range(dataset_cfg.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
            sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
            sampling_params = base_sampling_params.copy()
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params["sampling_seed"] = args.rollout_seed + j
            renderer.apply_to_sample(args, sample, rollout_id)
            if getattr(args, "use_multi_turn", False) and int(getattr(args, "max_turns", 1) or 1) > 1:

                def coro_factory(sample=sample, sampling_params=sampling_params):
                    return generate_multi_turn_eval_sample(
                        args,
                        sample,
                        sampling_params=sampling_params,
                        renderer=renderer,
                    )

            else:

                def coro_factory(sample=sample, sampling_params=sampling_params):
                    return generate_and_rm(
                        args,
                        sample,
                        sampling_params=sampling_params,
                        evaluation=True,
                    )

            tasks.append(asyncio.create_task(run_eval_coro(coro_factory, eval_semaphore)))

    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}", disable=not do_print)
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if do_print:
            logged_sample = sample[0] if isinstance(sample, list) else sample
            logger.info(
                "eval_rollout_single_dataset first sample: index=%s prompt_len=%d response_len=%d reward=%s",
                logged_sample.index,
                len(logged_sample.prompt) if isinstance(logged_sample.prompt, str) else -1,
                len(logged_sample.response) if isinstance(logged_sample.response, str) else -1,
                logged_sample.reward,
            )
            do_print = False
        if isinstance(sample, list):
            data.extend(sample)
        else:
            data.append(sample)
        pbar.update(1)
    pbar.close()

    data.sort(key=lambda sample: sample.index)

    reward_key = args.eval_reward_key or args.reward_key
    return {
        dataset_cfg.name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
        }
    }


def generate_rollout(args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False) -> Any:
    assert args.rollout_global_dataset
    ensure_no_pre_chat_template(args)
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
    if aborted_samples:
        data_source.add_samples(aborted_samples)
    return output
