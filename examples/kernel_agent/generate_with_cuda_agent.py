import ast
import asyncio
import hashlib
import json
import logging
import os
import random
import re
import time
from copy import deepcopy
from string import Formatter
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
        _context_len_for_turn,
        _extract_env_extra_info,
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
        _context_len_for_turn,
        _extract_env_extra_info,
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


# gpt-oss (harmony) responses come back with literal channel control tokens
# (skip_special_tokens=False), e.g.
#   <|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>ANSWER<|end|>
# Feeding such text straight back as an assistant message crashes the harmony
# chat template ("message containing <|channel|> tags in the content field") when
# the NEXT turn's prompt is rendered, which aborts every multi-turn rollout. Reduce
# the assistant turn we replay to just its final-channel answer. This is a strict
# no-op for any response without <|channel|> (qwen / musacoder / plain markdown).
_HARMONY_CTRL_RE = re.compile(r"<\|[^|>]*\|>")
_HARMONY_FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?=<\|(?:end|return|channel|start)\|>|$)",
    re.DOTALL,
)


def _sanitize_assistant_history_content(text: str) -> str:
    """Make a generated assistant turn safe to replay through apply_chat_template.

    Returns ``text`` unchanged unless it carries harmony ``<|channel|>`` control
    tokens, in which case it returns the last final-channel message (falling back
    to the control-token-stripped text when no final channel is present, e.g. a
    response truncated mid-analysis).
    """
    if "<|channel|>" not in text:
        return text
    final_text = ""
    for match in _HARMONY_FINAL_RE.finditer(text):
        candidate = match.group(1).strip()
        if candidate:
            final_text = candidate  # keep the last non-empty final channel
    if final_text:
        return final_text
    return _HARMONY_CTRL_RE.sub("", text).strip()


def _tokenize_without_special_tokens(tokenizer: Any, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _contains_token_subsequence(token_ids: list[int], subsequence: list[int]) -> bool:
    if not subsequence or len(subsequence) > len(token_ids):
        return False
    width = len(subsequence)
    return any(token_ids[start : start + width] == subsequence for start in range(len(token_ids) - width + 1))


def _build_qwen_next_turn_prompt_ids(
    tokenizer: Any,
    prompt_ids: list[int],
    response_ids: list[int],
    feedback: str,
    *,
    enable_thinking: bool,
) -> tuple[list[int], int, bool]:
    """Append a Qwen user-feedback turn without re-encoding generated history.

    The previous prompt and generated response tokens stay byte-for-byte intact.
    A missing ``</think>`` is closed before ``<|im_end|>`` so length-truncated
    reasoning remains a structurally valid assistant message. If the generated
    response already ended in ``<|im_end|>``, only that terminal token may move
    after the inserted close marker; every earlier generated token is still an
    exact reusable prefix.
    """

    eos_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if eos_token_id is None or int(eos_token_id) < 0:
        raise ValueError("Qwen token-prefix continuation requires the <|im_end|> token.")
    eos_token_id = int(eos_token_id)

    response_ids = list(response_ids)
    had_terminal_eos = bool(response_ids and response_ids[-1] == eos_token_id)
    response_body_ids = response_ids[:-1] if had_terminal_eos else response_ids
    close_think_ids = _tokenize_without_special_tokens(tokenizer, "</think>")
    inserted_close_think = enable_thinking and not _contains_token_subsequence(response_body_ids, close_think_ids)

    next_prompt_ids = list(prompt_ids) + response_body_ids
    if inserted_close_think:
        next_prompt_ids.extend(_tokenize_without_special_tokens(tokenizer, "\n</think>\n\n"))
    next_prompt_ids.append(eos_token_id)

    thinking_prefix = "<think>\n" if enable_thinking else "<think>\n\n</think>\n\n"
    next_prompt_ids.extend(
        _tokenize_without_special_tokens(
            tokenizer,
            "\n<|im_start|>user\n" + feedback + "<|im_end|>\n<|im_start|>assistant\n" + thinking_prefix,
        )
    )

    if had_terminal_eos and inserted_close_think:
        exact_prefix_tokens = len(prompt_ids) + len(response_body_ids)
    else:
        exact_prefix_tokens = len(prompt_ids) + len(response_ids)
    return next_prompt_ids, exact_prefix_tokens, inserted_close_think


def _sglang_routing_headers(args: Any, sample: Sample) -> dict[str, str] | None:
    if sample.session_id and getattr(args, "router_policy", None) == "consistent_hashing":
        return {"X-SMG-Routing-Key": sample.session_id}
    return None


def _get_tool_response_template(state: GenerateState) -> PromptTemplate:
    response_template = getattr(state, "multi_turn_template", None)
    if response_template is None:
        logger.warning("multi-turn tool_response template is not set; using built-in CUDA agent prompt template.")
        response_template = PromptTemplate(DEFAULT_TOOL_RESPONSE_TEMPLATE, "format", "built-in")
    _validate_model_feedback_template(
        response_template,
        kernel_backend=getattr(getattr(state, "args", None), "kernel_backend", None),
    )
    return response_template


def _validate_model_feedback_template(
    response_template: PromptTemplate,
    *,
    kernel_backend: str | None = None,
) -> None:
    """Require exactly one plain feedback placeholder and no budget bypass."""

    template = response_template.template
    if kernel_backend == "tvm_ffi":
        forbidden_markers = ("PYBIND11_MODULE", "REGISTER_BINDING(", "binding_registry.h")
        found_markers = [marker for marker in forbidden_markers if marker in template]
        if found_markers:
            raise ValueError(
                "TVM-FFI multi-turn feedback template contains incompatible pybind binding instructions: "
                + ", ".join(found_markers)
            )
    if "feedback_dict" in template:
        raise ValueError(
            "Model feedback templates must use {feedback}; feedback_dict bypasses the strict text budget."
        )

    if response_template.render_mode == "format":
        fields = [item for item in Formatter().parse(template) if item[1] is not None]
        if len(fields) != 1 or fields[0][1:] != ("feedback", "", None):
            raise ValueError("Model feedback format templates require exactly one plain {feedback} placeholder.")
        return
    if response_template.render_mode == "jinja":
        output_expressions = re.findall(r"{{(.*?)}}", template, flags=re.DOTALL)
        block_expressions = re.findall(r"{%(.*?)%}", template, flags=re.DOTALL)
        if (
            len(output_expressions) != 1
            or output_expressions[0].strip() != "feedback"
            or any(re.search(r"\bfeedback\b", expression) for expression in block_expressions)
        ):
            raise ValueError("Model feedback Jinja templates require exactly one plain {{ feedback }} placeholder.")
        return
    raise ValueError(f"Unsupported model feedback template mode: {response_template.render_mode}")


def _truncate_middle(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    marker = "...(truncated)..."
    if max_chars <= len(marker):
        return text[:max_chars]
    remaining = max_chars - len(marker)
    keep_start = (remaining + 1) // 2
    keep_end = remaining - keep_start
    suffix = text[-keep_end:] if keep_end else ""
    return text[:keep_start] + marker + suffix


_MODEL_FEEDBACK_TOP_LEVEL_KEYS = (
    "status",
    "error",
    "precheck",
    "compiled",
    "correctness",
    "decoy_kernel",
)
_MODEL_FEEDBACK_ERROR_DETAIL_KEYS = (
    "model_load_error",
    "model_load_error_name",
    "compilation_error_name",
    "compilation_error_detail",
    "runtime_error_name",
    "correctness_runtime_error",
)
_MODEL_FEEDBACK_CORRECTNESS_KEYS = (
    "correctness_issue",
    "correctness_issue_name",
    "max_difference",
    "avg_difference",
    "correctness_failed_trial",
    "num_correct_trials",
    "correctness_trials_run",
    "correctness_output_mismatch",
    "correctness_candidate_forward_completed",
)
_MODEL_FEEDBACK_PROFILE_KEYS = (
    "num_custom_kernels",
    "num_total_kernels",
    "custom_kernel_names",
    "custom_kernel_not_in_profiling",
    "custom_kernel_coverage",
    "custom_kernel_cuda_time_coverage",
    "coverage_measurement_valid",
)
_MODEL_FEEDBACK_BACKEND_PROBE_KEYS = (
    "attempted",
    "valid",
    "custom_kernel_observed",
    "decoy_detected",
    "skip_reason",
    "error",
    "num_total_kernels",
    "num_matched_custom_kernels",
    "matched_kernel_names",
    "missing_kernel_names",
)
_MODEL_FEEDBACK_MAX_DETAIL_CHARS = 2048
_MODEL_FEEDBACK_MAX_LIST_ITEMS = 32
_MODEL_FEEDBACK_SANITIZER_SUMMARY_KEYS = (
    "status",
    "passed",
    "measurement_complete",
    "requested_checks",
    "executed_checks",
    "detected_issue_count",
    "primary_check",
    "primary_detected_issue_count",
    "issue_count_by_check",
    "issues_truncated",
    "kernel_filter_empty",
    "selection_mode",
    "mode",
    "error_classification",
    "run_all_checks",
    "reason",
    "replayed_input_seed",
    "wall_time_s",
    "error",
)
_MODEL_FEEDBACK_SANITIZER_CHECK_KEYS = (
    "check",
    "status",
    "passed",
    "process_completed",
    "target_application_failed",
    "sanitizer_issue_found",
    "input_generation",
    "input_values_exactly_replayed",
    "return_code",
    "summary_error_count",
    "parsed_issue_count",
    "detected_issue_count",
    "issues_truncated",
    "wall_time_s",
    "error",
)
_MODEL_FEEDBACK_SANITIZER_ISSUE_KEYS = (
    "hazard_type",
    "message",
    "occurrence_count",
    "access_type",
    "memory_space",
    "access_size_bytes",
)
_COMPILER_PRIMARY_DIAGNOSTIC_RE = re.compile(
    r"(?:fatal error:|\berror:|undefined reference|unresolved external symbol|nvcc fatal|collect2: error)",
    re.IGNORECASE,
)
_COMPILER_NOTE_RE = re.compile(r"\bnote:", re.IGNORECASE)
_COMPILER_MACRO_NOTE_RE = re.compile(
    r"\bnote:.*(?:in (?:definition|expansion) of macro|expanded from macro)",
    re.IGNORECASE,
)
_COMPILER_CANDIDATE_NOTE_RE = re.compile(r"\bnote:.*\bcandidate:", re.IGNORECASE)
_COMPILER_REJECTION_NOTE_RE = re.compile(
    r"\bnote:.*(?:no known conversion|candidate expects|deduced conflicting types|"
    r"template argument deduction/substitution failed|constraints not satisfied|could not convert|cannot convert)",
    re.IGNORECASE,
)
_COMPILER_HIGH_VALUE_NOTE_RE = re.compile(
    r"\bnote:.*(?:candidate:|no known conversion|candidate expects|deduced conflicting types|"
    r"template argument deduction/substitution failed|required from|constraints not satisfied|"
    r"could not convert|cannot convert)",
    re.IGNORECASE,
)
_COMPILER_TERMINAL_EXCEPTION_RE = re.compile(r"^\s*(?:(?:Error|Exception)|[A-Za-z_][\w.]*(?:Error|Exception)):\s*\S")
_COMPILER_BUILD_STEP_RE = re.compile(r"^\[(?P<step>\d+/\d+)\]\s+(?P<command>.*)$")
_ABSOLUTE_SOURCE_PATH_RE = re.compile(
    r"(?P<dir>/(?:[^/\s:'\"]+/)+)(?P<file>[^/\s:'\"]+\.(?:cc|cpp|cu|cuh|h|hpp|py))" r"(?P<location>:\d+(?::\d+)?)?"
)
_COMPILER_MAX_ACTIONABLE_NOTES = 4
_COMPILER_SUPPLEMENT_MAX_CHARS = 768
_COMPILER_CONTEXT_LINE_MAX_CHARS = 384


def _shorten_diagnostic_paths(text: str) -> str:
    def replace_path(match: re.Match[str]) -> str:
        directory = match.group("dir")
        parent = directory.rstrip("/").rsplit("/", 1)[-1]
        if "/dev/shm/kernelgym/compile_cache/" in directory and parent != "kernels":
            parent = "generated"
        return f".../{parent}/{match.group('file')}{match.group('location') or ''}"

    return _ABSOLUTE_SOURCE_PATH_RE.sub(replace_path, text)


def _summarize_compiler_build_step(line: str) -> str | None:
    match = _COMPILER_BUILD_STEP_RE.match(line.strip())
    if match is None:
        return None
    command = match.group("command")
    if not any(tool in command for tool in ("nvcc", "c++", "g++")):
        return None

    source_match = re.search(r"(?P<source>[^\s/]+\.(?:cpp|cu))(?=\s|$)", command)
    output_match = re.search(r"(?:^|\s)-o\s+(?P<output>[^\s]+)", command)
    source = source_match.group("source") if source_match else None
    output = output_match.group("output").rsplit("/", 1)[-1] if output_match else None
    operation = "link" if " -shared " in command else "compile"
    operands = " -> ".join(value for value in (source, output) if value)
    return f"[{match.group('step')}] {operation}{f' {operands}' if operands else ''}"


def _is_repeated_compiler_command(line: str) -> bool:
    stripped = line.lstrip()
    if not any(tool in stripped for tool in ("nvcc", "c++", "g++")):
        return False
    return stripped.startswith(("/usr/", "/opt/", ": &&", "nvcc ", "c++ ", "g++ "))


def _select_actionable_compiler_notes(notes: list[str], limit: int) -> list[str]:
    """Prefer complete overload-candidate/rejection pairs, then other useful notes."""

    selected: list[str] = []

    def add(line: str) -> None:
        if len(selected) < limit and line not in selected:
            selected.append(line)

    for index in range(len(notes) - 1):
        if len(selected) + 2 > limit:
            break
        if _COMPILER_CANDIDATE_NOTE_RE.search(notes[index]) and _COMPILER_REJECTION_NOTE_RE.search(notes[index + 1]):
            add(notes[index])
            add(notes[index + 1])

    for note in notes:
        if len(selected) >= limit:
            break
        if _COMPILER_HIGH_VALUE_NOTE_RE.search(note):
            add(note)
    for note in notes:
        if len(selected) >= limit:
            break
        add(note)
    return selected


def _fit_supplemental_diagnostics(
    terminal_exception: str | None,
    actionable_notes: list[str],
    max_chars: int,
) -> str:
    """Fit terminal failure and a bounded actionable-note summary."""

    if max_chars <= 0:
        return ""

    terminal = _truncate_middle(terminal_exception, min(512, max_chars)) if terminal_exception else None
    selected = [
        _truncate_middle(note, min(384, max_chars))
        for note in _select_actionable_compiler_notes(actionable_notes, _COMPILER_MAX_ACTIONABLE_NOTES)
    ]
    for shown in range(len(selected), -1, -1):
        omitted = len(actionable_notes) - shown
        parts = ([terminal] if terminal else []) + selected[:shown]
        if omitted > 0:
            parts.append(f"[omitted {omitted} additional unique actionable diagnostic notes]")
        result = "\n".join(parts)
        if len(result) <= max_chars:
            return result

    # A terminal exception has higher priority than the note omission marker.
    return _truncate_middle(terminal, max_chars) if terminal else ""


def _uniformly_sample_indices(indices: list[int], count: int) -> list[int]:
    if count <= 1:
        return [indices[-1]]
    return list(dict.fromkeys(indices[round(slot * (len(indices) - 1) / (count - 1))] for slot in range(count)))


def _diagnostic_excerpt(text: str, max_chars: int) -> str:
    """Fit an oversized diagnostic and make any omitted errors explicit.

    All recognized errors are retained when they fit. If even the primary error
    blocks exceed the fixed budget, uniformly sampled blocks are shown with an
    explicit omitted-count marker. Unknown formats use the conservative
    head+tail fallback instead of speculative parsing.
    """

    if max_chars <= 0 or len(text) <= max_chars:
        return text

    lines = text.splitlines()
    primary_indices = [index for index, line in enumerate(lines) if _COMPILER_PRIMARY_DIAGNOSTIC_RE.search(line)]
    if not primary_indices:
        return _truncate_middle(text, max_chars)

    selected: set[int] = set()
    for index in primary_indices:
        selected.update(range(max(0, index - 2), min(len(lines), index + 3)))
    for index, line in enumerate(lines):
        if _COMPILER_NOTE_RE.search(line):
            selected.update(range(max(0, index - 1), min(len(lines), index + 2)))
    terminal_indices = [index for index, line in enumerate(lines) if _COMPILER_TERMINAL_EXCEPTION_RE.search(line)]
    terminal_index = terminal_indices[-1] if terminal_indices else None
    if terminal_index is not None:
        selected.add(terminal_index)

    excerpt = "\n".join(lines[index] for index in sorted(selected))
    marker = f"[compiler context compacted from {len(text)} characters]\n"
    if len(marker) + len(excerpt) <= max_chars:
        return marker + excerpt

    if max_chars < 128:
        return _truncate_middle(text, max_chars)

    primary_context_indices: set[int] = set()
    for index in primary_indices:
        primary_context_indices.update(range(max(0, index - 2), min(len(lines), index + 3)))
    primary_context_indices = {
        index for index in primary_context_indices if not _COMPILER_MACRO_NOTE_RE.search(lines[index])
    }

    visible_notes = {lines[index] for index in primary_context_indices if _COMPILER_NOTE_RE.search(lines[index])}
    actionable_notes = list(
        dict.fromkeys(
            line
            for line in lines
            if _COMPILER_NOTE_RE.search(line)
            and not _COMPILER_MACRO_NOTE_RE.search(line)
            and line not in visible_notes
        )
    )
    terminal_exception = lines[terminal_index] if terminal_index is not None else None
    supplement = _fit_supplemental_diagnostics(
        terminal_exception if terminal_index not in primary_context_indices else None,
        actionable_notes,
        min(_COMPILER_SUPPLEMENT_MAX_CHARS, max_chars // 5),
    )

    primary_context = "\n".join(lines[index] for index in sorted(primary_context_indices))
    complete_sections = [marker.rstrip(), primary_context]
    if supplement:
        complete_sections.append(supplement)
    complete_result = "\n\n".join(complete_sections)
    if len(complete_result) <= max_chars:
        return complete_result

    def render_blocks(blocks: list[list[str]], extra: str, footer: str | None) -> str:
        sections = [marker.rstrip(), "\n\n".join("\n".join(block) for block in blocks)]
        if extra:
            sections.append(extra)
        if footer:
            sections.append(footer)
        return "\n\n".join(sections)

    total_blocks = len(primary_indices)
    terminal_only = _fit_supplemental_diagnostics(
        terminal_exception,
        [],
        min(512, max_chars // 5),
    )
    sampled_indices = primary_indices
    footer = None
    blocks = [[lines[index]] for index in sampled_indices]
    if len(render_blocks(blocks, terminal_only, footer)) > max_chars:
        for block_count in range(total_blocks - 1, 0, -1):
            candidate_indices = _uniformly_sample_indices(primary_indices, block_count)
            candidate_footer = (
                f"[omitted {total_blocks - len(candidate_indices)} of {total_blocks} primary diagnostic blocks]"
            )
            candidate_blocks = [[lines[index]] for index in candidate_indices]
            if len(render_blocks(candidate_blocks, terminal_only, candidate_footer)) <= max_chars:
                sampled_indices = candidate_indices
                footer = candidate_footer
                blocks = candidate_blocks
                break
        else:
            sampled_indices = [primary_indices[-1]]
            footer = f"[omitted {total_blocks - 1} of {total_blocks} primary diagnostic blocks]"
            fixed = render_blocks([[""]], terminal_only, footer)
            primary_budget = max(1, max_chars - len(fixed))
            blocks = [[_truncate_middle(lines[sampled_indices[0]], primary_budget)]]

    used_indices = set(sampled_indices)
    actionable_notes = list(
        dict.fromkeys(
            line
            for index, line in enumerate(lines)
            if _COMPILER_NOTE_RE.search(line)
            and not _COMPILER_MACRO_NOTE_RE.search(line)
            and index not in used_indices
        )
    )
    base_result = render_blocks(blocks, "", footer)
    supplement_budget = min(
        _COMPILER_SUPPLEMENT_MAX_CHARS,
        max_chars // 5,
        max(0, max_chars - len(base_result) - 2),
    )
    supplement = _fit_supplemental_diagnostics(terminal_exception, actionable_notes, supplement_budget)

    # Add source/caret and preceding context only after the terminal exception
    # and actionable-note summary are fixed. Oversized diagnostics may contain
    # dozens of repeated source snippets; those must not crowd out the reason an
    # overload candidate was rejected.
    for offset in (1, 2, -1, -2):
        for block_index, primary_index in enumerate(sampled_indices):
            context_index = primary_index + offset
            if (
                context_index < 0
                or context_index >= len(lines)
                or context_index in used_indices
                or context_index == terminal_index
                or _COMPILER_PRIMARY_DIAGNOSTIC_RE.search(lines[context_index])
                or _COMPILER_MACRO_NOTE_RE.search(lines[context_index])
                or _COMPILER_NOTE_RE.search(lines[context_index])
            ):
                continue
            blocks[block_index].append(_truncate_middle(lines[context_index], _COMPILER_CONTEXT_LINE_MAX_CHARS))
            if len(render_blocks(blocks, supplement, footer)) > max_chars:
                blocks[block_index].pop()
            else:
                used_indices.add(context_index)

    return render_blocks(blocks, supplement, footer)


def compact_compiler_diagnostics(text: str, max_chars: int = 6000) -> str:
    """Remove high-confidence build boilerplate while preserving diagnostics."""

    if not isinstance(text, str):
        text = str(text)

    compacted_lines: list[str] = []
    for raw_line in text.splitlines():
        line = _shorten_diagnostic_paths(raw_line.rstrip())
        stripped = line.strip()
        if stripped in {"stdout:", "stderr:"} or stripped.startswith("ninja exited with status"):
            continue

        build_summary = None
        if not _COMPILER_PRIMARY_DIAGNOSTIC_RE.search(line):
            build_summary = _summarize_compiler_build_step(line)
        if build_summary is not None:
            if not compacted_lines or compacted_lines[-1] != build_summary:
                compacted_lines.append(build_summary)
            continue
        if _is_repeated_compiler_command(line) and not _COMPILER_PRIMARY_DIAGNOSTIC_RE.search(line):
            continue
        if stripped.startswith("ninja: build stopped: subcommand failed"):
            continue
        if compacted_lines and compacted_lines[-1] == line:
            continue
        compacted_lines.append(line)

    compacted = "\n".join(compacted_lines).strip()
    if not compacted:
        compacted = text.strip()
    return _diagnostic_excerpt(compacted, max_chars)


def _bounded_feedback_value(
    value: Any,
    max_chars: int = _MODEL_FEEDBACK_MAX_DETAIL_CHARS,
    max_items: int = _MODEL_FEEDBACK_MAX_LIST_ITEMS,
    _seen: set[int] | None = None,
    _depth: int = 0,
) -> Any:
    if _depth >= 12:
        return "...(nested detail omitted)..."
    if _seen is None:
        _seen = set()
    if isinstance(value, str):
        return _truncate_middle(value, max_chars)
    if isinstance(value, (list, tuple)):
        if id(value) in _seen:
            return "...(cyclic reference omitted)..."
        _seen.add(id(value))
        try:
            items = [
                _bounded_feedback_value(item, max_chars, max_items, _seen, _depth + 1) for item in value[:max_items]
            ]
            if len(value) > max_items:
                items.append(f"...({len(value) - max_items} items omitted)...")
            return items
        finally:
            _seen.remove(id(value))
    if isinstance(value, dict):
        if id(value) in _seen:
            return "...(cyclic reference omitted)..."
        _seen.add(id(value))
        try:
            items = list(value.items())
            compacted = {
                str(key): _bounded_feedback_value(item_value, max_chars, max_items, _seen, _depth + 1)
                for key, item_value in items[:max_items]
            }
            if len(items) > max_items:
                compacted["_omitted_fields"] = len(items) - max_items
            return compacted
        finally:
            _seen.remove(id(value))
    return deepcopy(value)


def _serialize_feedback_dict(feedback: dict[str, Any]) -> str:
    return json.dumps(feedback, ensure_ascii=False, default=str)


def _fit_model_feedback_to_budget(feedback: dict[str, Any], max_chars: int) -> tuple[dict[str, Any], str, bool]:
    """Shrink feedback structurally so the injected text remains valid JSON."""

    serialized = _serialize_feedback_dict(feedback)
    if max_chars <= 0 or len(serialized) <= max_chars:
        return feedback, serialized, False
    if max_chars < 2:
        raise ValueError("max_feedback_chars must be 0 or at least 2 to preserve valid JSON feedback.")

    for detail_chars, list_items in ((1024, 16), (512, 8), (256, 4), (128, 2), (64, 1)):
        candidate = {key: _bounded_feedback_value(value, detail_chars, list_items) for key, value in feedback.items()}
        candidate["_feedback_budget"] = {"structured_reduction": True}
        candidate_text = _serialize_feedback_dict(candidate)
        if len(candidate_text) <= max_chars:
            return candidate, candidate_text, True

    # Extremely small budgets cannot carry every section. Preserve the outcome
    # and primary error in priority order, adding each field only when the whole
    # payload remains valid JSON inside the requested budget.
    minimal: dict[str, Any] = {}
    omitted_marker = {"_omitted": len(feedback)}
    if len(_serialize_feedback_dict(omitted_marker)) <= max_chars:
        minimal = omitted_marker
    for key in ("status", "error", "error_message", "precheck", "compiled", "correctness", "decoy_kernel"):
        if key not in feedback:
            continue
        value = _bounded_feedback_value(feedback[key], 64, 1)
        candidate = {**minimal, key: value}
        if "_omitted" in candidate:
            candidate["_omitted"] = max(0, len(feedback) - len(candidate) + 1)
        if len(_serialize_feedback_dict(candidate)) <= max_chars:
            minimal = candidate
            continue
        if not isinstance(value, str):
            continue
        low, high = 0, len(value)
        best: dict[str, Any] | None = None
        while low <= high:
            midpoint = (low + high) // 2
            candidate = {**minimal, key: _truncate_middle(value, midpoint)}
            if "_omitted" in candidate:
                candidate["_omitted"] = max(0, len(feedback) - len(candidate) + 1)
            if len(_serialize_feedback_dict(candidate)) <= max_chars:
                best = candidate
                low = midpoint + 1
            else:
                high = midpoint - 1
        if best is not None:
            minimal = best
    return minimal, _serialize_feedback_dict(minimal), True


def _copy_present(mapping: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: _bounded_feedback_value(mapping[key]) for key in keys if key in mapping and mapping[key] is not None}


def _compact_forbidden_aten_ops(metadata: dict[str, Any]) -> list[dict[str, Any]] | list[str] | None:
    forbidden_ops = metadata.get("forbidden_aten_ops")
    if isinstance(forbidden_ops, list):
        compacted: list[dict[str, Any]] = []
        for item in forbidden_ops[:_MODEL_FEEDBACK_MAX_LIST_ITEMS]:
            if not isinstance(item, dict):
                continue
            name = item.get("normalized_name") or item.get("name")
            if not isinstance(name, str) or not name:
                continue
            compact_item: dict[str, Any] = {"name": name}
            count = item.get("count")
            if isinstance(count, int) and not isinstance(count, bool):
                compact_item["count"] = count
            compacted.append(compact_item)
        if len(forbidden_ops) > _MODEL_FEEDBACK_MAX_LIST_ITEMS:
            compacted.append({"omitted_items": len(forbidden_ops) - _MODEL_FEEDBACK_MAX_LIST_ITEMS})
        if compacted:
            return compacted

    forbidden_names = metadata.get("forbidden_aten_op_names")
    if isinstance(forbidden_names, list):
        names = [
            _truncate_middle(name, _MODEL_FEEDBACK_MAX_DETAIL_CHARS)
            for name in forbidden_names[:_MODEL_FEEDBACK_MAX_LIST_ITEMS]
            if isinstance(name, str) and name
        ]
        if len(forbidden_names) > _MODEL_FEEDBACK_MAX_LIST_ITEMS:
            names.append(f"...({len(forbidden_names) - _MODEL_FEEDBACK_MAX_LIST_ITEMS} items omitted)...")
        if names:
            return names
    return None


def _compact_backend_probe(metadata: dict[str, Any]) -> dict[str, Any] | None:
    probe = metadata.get("incorrect_backend_usage_probe")
    if not isinstance(probe, dict):
        return None
    compacted = _copy_present(probe, _MODEL_FEEDBACK_BACKEND_PROBE_KEYS)
    return compacted or None


def _format_sanitizer_location(issue: dict[str, Any]) -> tuple[str | None, str | None]:
    kernel = issue.get("kernel")
    source = issue.get("source")
    kernel_text = str(kernel) if kernel is not None else None
    source_text = None
    if isinstance(source, dict):
        file_name = source.get("file")
        line = source.get("line")
        if file_name is not None:
            source_text = f"{file_name}:{line}" if line is not None else str(file_name)

    kernel_info = issue.get("kernel_info")
    if isinstance(kernel_info, list):
        locations = []
        for item in kernel_info:
            if not isinstance(item, dict):
                continue
            parts = [str(value) for value in (item.get("name"), item.get("source")) if value is not None]
            if parts:
                locations.append(" @ ".join(parts))
        if locations:
            kernel_text = "; ".join(locations)
    return kernel_text, source_text


def _format_sanitizer_range(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    parts = []
    for axis in ("x", "y", "z"):
        bounds = value.get(axis)
        if isinstance(bounds, list) and bounds:
            parts.append(f"{axis}={bounds[0]}..{bounds[-1]}")
    ranges = value.get("ranges")
    if isinstance(ranges, list) and ranges:
        parts.append("address=" + "..".join(str(item) for item in ranges))
    return ",".join(parts) or None


def _compact_runtime_sanitizer(value: dict[str, Any]) -> dict[str, Any]:
    """Keep bounded factual diagnostics without replay payloads or raw tool output."""

    compacted = _copy_present(value, _MODEL_FEEDBACK_SANITIZER_SUMMARY_KEYS)
    checks = []
    issues = []
    for check_result in value.get("check_results") or []:
        if not isinstance(check_result, dict):
            continue
        check_summary = _copy_present(check_result, _MODEL_FEEDBACK_SANITIZER_CHECK_KEYS)
        if check_summary:
            checks.append(check_summary)
        check_name = check_result.get("check")
        for issue in check_result.get("issues") or []:
            if not isinstance(issue, dict):
                continue
            compact_issue = _copy_present(issue, _MODEL_FEEDBACK_SANITIZER_ISSUE_KEYS)
            if check_name is not None:
                compact_issue["check"] = str(check_name)
            kernel, source = _format_sanitizer_location(issue)
            if kernel:
                compact_issue["kernel"] = _truncate_middle(kernel, _MODEL_FEEDBACK_MAX_DETAIL_CHARS)
            if source:
                compact_issue["source"] = _truncate_middle(source, _MODEL_FEEDBACK_MAX_DETAIL_CHARS)
            for source_key, target_key in (
                ("threads", "thread_range"),
                ("blocks", "block_range"),
                ("addresses", "address_range"),
            ):
                rendered = _format_sanitizer_range(issue.get(source_key))
                if rendered:
                    compact_issue[target_key] = rendered
            if compact_issue:
                issues.append(compact_issue)

    if checks:
        compacted["checks"] = checks[:_MODEL_FEEDBACK_MAX_LIST_ITEMS]
    if issues:
        compacted["issues"] = issues[:_MODEL_FEEDBACK_MAX_LIST_ITEMS]
        if len(issues) > _MODEL_FEEDBACK_MAX_LIST_ITEMS:
            compacted["omitted_issues"] = len(issues) - _MODEL_FEEDBACK_MAX_LIST_ITEMS
    return compacted


def build_model_feedback(env_result: dict[str, Any], *, compiler_max_chars: int | None = None) -> dict[str, Any]:
    """Build the actionable feedback shown to the next model turn.

    KernelGym's normalized ``env_state`` remains untouched and available to
    reward code, logs, and audits. Only this separately-built payload drops
    machine-facing metadata and summarizes repeated diagnostics.
    """

    if not isinstance(env_result, dict):
        env_result = {}
    raw_state = env_result.get("env_state")
    state = raw_state if isinstance(raw_state, dict) else env_result
    metadata = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}

    feedback = _copy_present(state, _MODEL_FEEDBACK_TOP_LEVEL_KEYS)
    precheck_failed = feedback.get("precheck") == "failed"
    if precheck_failed:
        # These checks did not run when static precheck rejected the candidate.
        feedback.pop("decoy_kernel", None)
    precheck_diagnostic = metadata.get("precheck_diagnostic")
    if isinstance(precheck_diagnostic, dict):
        feedback["precheck_diagnostic"] = _bounded_feedback_value(precheck_diagnostic)
    runtime_sanitizer = state.get("runtime_sanitizer")
    if isinstance(runtime_sanitizer, dict):
        feedback["runtime_sanitizer"] = _compact_runtime_sanitizer(runtime_sanitizer)
    error_message = state.get("error_message")
    diagnostic_has_error_message = isinstance(precheck_diagnostic, dict) and isinstance(
        precheck_diagnostic.get("error_message"), str
    )
    if error_message is not None and not (precheck_failed and diagnostic_has_error_message):
        error_message = str(error_message)
        if compiler_max_chars is None:
            feedback_cap = int(CUDA_AGENT_CONFIGS.get("max_feedback_chars", 0) or 0)
            compiler_max_chars = max(512, min(6000, feedback_cap - 1024)) if feedback_cap > 0 else 6000
        if state.get("error") == "COMPILATION_ERROR":
            error_message = compact_compiler_diagnostics(error_message, compiler_max_chars)
        else:
            error_message = _shorten_diagnostic_paths(error_message)
            error_message = _truncate_middle(error_message, compiler_max_chars)
        feedback["error_message"] = error_message

    performance = {} if precheck_failed else _copy_present(state, ("speedup", "kernel_runtime", "reference_runtime"))
    if performance:
        feedback["performance"] = performance

    error_details = {} if precheck_failed else _copy_present(metadata, _MODEL_FEEDBACK_ERROR_DETAIL_KEYS)
    metadata_error = metadata.get("error")
    if (
        not precheck_failed
        and metadata_error is not None
        and str(metadata_error) != str(state.get("error_message") or "")
    ):
        error_details["metadata_error"] = _bounded_feedback_value(metadata_error)
    if error_details:
        feedback["error_details"] = error_details

    correctness_details = _copy_present(metadata, _MODEL_FEEDBACK_CORRECTNESS_KEYS)
    if correctness_details:
        feedback["correctness_details"] = correctness_details

    policy_details: dict[str, Any] = {}
    if "aten_detection_valid" in metadata:
        policy_details["aten_detection_valid"] = bool(metadata["aten_detection_valid"])
    forbidden_ops = _compact_forbidden_aten_ops(metadata)
    if forbidden_ops:
        policy_details["forbidden_aten_ops"] = forbidden_ops
    decoy_reasons: list[str] = []
    for key in ("decoy_reason", "policy_violation_reason"):
        value = metadata.get(key)
        if isinstance(value, str) and value and value not in decoy_reasons:
            decoy_reasons.append(_truncate_middle(value, _MODEL_FEEDBACK_MAX_DETAIL_CHARS))
    policy_warnings: list[str] = []
    suspected_reason = metadata.get("suspected_decoy_reason")
    if isinstance(suspected_reason, str) and suspected_reason:
        policy_warnings.append(_truncate_middle(suspected_reason, _MODEL_FEEDBACK_MAX_DETAIL_CHARS))
    values = metadata.get("suspected_decoy_reasons")
    if isinstance(values, list):
        available_slots = max(0, _MODEL_FEEDBACK_MAX_LIST_ITEMS - len(policy_warnings))
        needs_omission = len(values) > available_slots
        value_slots = max(0, available_slots - int(needs_omission))
        for value in values[:value_slots]:
            if isinstance(value, str) and value and value not in policy_warnings:
                policy_warnings.append(_truncate_middle(value, _MODEL_FEEDBACK_MAX_DETAIL_CHARS))
        if needs_omission:
            policy_warnings.append(f"...({len(values) - value_slots} items omitted)...")
    if decoy_reasons:
        policy_details["decoy_reasons"] = decoy_reasons
    if policy_warnings:
        policy_details["policy_warnings"] = policy_warnings
    if "suspected_decoy_effect" in metadata:
        policy_details["suspected_decoy_effect"] = _bounded_feedback_value(metadata["suspected_decoy_effect"])
    backend_probe = _compact_backend_probe(metadata)
    if backend_probe:
        policy_details["incorrect_backend_usage_probe"] = backend_probe
    if policy_details:
        feedback["policy_details"] = policy_details

    profiling_summary = _copy_present(metadata, _MODEL_FEEDBACK_PROFILE_KEYS)
    if profiling_summary:
        feedback["profiling_summary"] = profiling_summary

    return feedback


def _apply_feedback_template(
    env_result: dict[str, Any],
    response_template: PromptTemplate,
    *,
    feedback_stats: dict[str, Any] | None = None,
) -> str:
    _validate_model_feedback_template(response_template)
    raw_feedback_dict = env_result.get("env_state") or env_result
    feedback_dict = build_model_feedback(env_result)
    compacted_feedback = _serialize_feedback_dict(feedback_dict)

    max_chars = int(CUDA_AGENT_CONFIGS["max_feedback_chars"])
    # A non-positive value disables only the final total budget. The
    # model-facing schema and per-field safety bounds still remove redundant
    # machine/debug data.
    feedback_dict, feedback, structured_reduced = _fit_model_feedback_to_budget(feedback_dict, max_chars)
    if feedback_stats is not None:
        try:
            original_chars = len(json.dumps(raw_feedback_dict, ensure_ascii=False, indent=2, default=str))
        except (TypeError, ValueError, RecursionError):
            original_chars = len(str(raw_feedback_dict))
        feedback_stats.update(
            {
                "original_chars": original_chars,
                "compacted_chars": len(compacted_feedback),
                "final_chars": len(feedback),
                "final_truncated": structured_reduced,
                "structured_reduced": structured_reduced,
            }
        )
    return response_template.format(feedback=feedback, feedback_dict={})


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
        model_feedback_stats = (
            item.get("model_feedback_stats") if isinstance(item.get("model_feedback_stats"), dict) else {}
        )
        logger.info(
            "%s[turn %s] task_id=%s model_time=%.3fs env_time=%.3fs prompt_tokens=%s max_new_tokens=%s "
            "response_tokens=%s cached_tokens=%s next_turn_prompt_tokens=%s next_turn_exact_prefix_tokens=%s "
            "next_turn_inserted_close_think=%s "
            "finish_type=%s status=%s error=%s precheck=%s speedup=%s correctness=%s compiled=%s "
            "partial_credit=%s partial_reason=%s reward=%s feedback_chars=%s->%s->%s "
            "feedback_truncated=%s detail_env_time=%s perf_cv=%s",
            prefix,
            item.get("turn_idx"),
            item.get("task_id"),
            float(item.get("model_time", 0.0)),
            float(item.get("env_time", 0.0)),
            item.get("prompt_tokens"),
            item.get("max_new_tokens"),
            item.get("response_tokens"),
            item.get("cached_tokens"),
            item.get("next_turn_prompt_tokens"),
            item.get("next_turn_exact_prefix_tokens"),
            item.get("next_turn_inserted_close_think"),
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
            model_feedback_stats.get("original_chars"),
            model_feedback_stats.get("compacted_chars"),
            model_feedback_stats.get("final_chars"),
            model_feedback_stats.get("final_truncated"),
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
    *,
    turn_idx: int | None = None,
) -> dict[str, Any]:
    turn_sampling_params = sampling_params.copy()
    max_context_len = _context_len_for_turn(args, turn_idx)
    if max_context_len is None:
        return turn_sampling_params

    draft_token_reserve = 0
    if getattr(args, "sglang_speculative_algorithm", None):
        draft_token_reserve = max(0, int(getattr(args, "sglang_speculative_num_draft_tokens", 0) or 0))
    serving_context = getattr(args, "sglang_context_length", None) or getattr(args, "rollout_max_context_len", None)
    if serving_context is not None:
        max_context_len = min(int(max_context_len), int(serving_context) - draft_token_reserve)
    elif draft_token_reserve:
        max_context_len = int(max_context_len) - draft_token_reserve
    remaining_context = int(max_context_len) - int(prompt_token_count)
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


async def cuda_kernel_env(
    args,
    sample: Sample,
    response: str,
    turn_idx: int,
) -> dict[str, Any]:
    entry_point = _get_entry_point(sample)
    ground_truth = _get_label_value(sample, "ground_truth")
    precision = _resolve_task_precision(sample, ground_truth)
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
        result = {
            "env_state": precheck_state,
            "env_extra_info": _extract_env_extra_info(precheck_state),
        }
        if getattr(args, "component_reward", False):
            result["runtime_graph"] = {
                "schema": "kernelgym-runtime-graph/v1",
                "status": "unavailable",
                "unknowns": ["client_precheck"],
            }
        return result
    else:
        task_id = next_kernel_task_id()
        metadata = dict(sample.metadata or {})
        metadata["task_id"] = task_id
        sample.metadata = metadata

    env_config = CUDA_AGENT_CONFIGS["env"]
    payload = {
        "task_id": task_id,
        "response": response,
        "ground_truth": ground_truth,
        "kernel_backend": kernel_backend,
        "reference_backend": reference_backend,
        "entry_point": entry_point,
        "precision": precision,
        # Reference-timing cache key: a hash of the reference identity, NOT any
        # dataset-supplied id — a bare problem_id/name (or a non-unique explicit
        # uuid) would false-share a wrong cached baseline across datasets on a
        # shared KernelGym. See _reference_cache_uuid.
        "uuid": _reference_cache_uuid(ground_truth, entry_point),
        "use_reference_cache": bool(getattr(args, "use_reference_cache", False)),
        "return_full_state": True,
        "metadata": sample.metadata,
        "turn_idx": turn_idx,
        "num_correct_trials": env_config.get("num_correct_trials"),
        "num_perf_trials": env_config.get("num_perf_trials"),
        "num_warmup": env_config.get("num_warmup"),
        "perf_trim_count": env_config.get("perf_trim_count"),
        "adaptive_perf_trials": env_config.get("adaptive_perf_trials"),
        "perf_min_trials": env_config.get("perf_min_trials"),
        "perf_cv_threshold": env_config.get("perf_cv_threshold"),
        "refer_num_perf_trials": env_config.get("refer_num_perf_trials"),
        "correctness_timeout": env_config.get("correctness_timeout"),
        "correctness_timeout_enabled": env_config.get("correctness_timeout_enabled"),
    }
    kernel_eval_result = await run_kernel_eval(args, sample, payload, CUDA_AGENT_CONFIGS["env"])
    raw_env_state = kernel_eval_result.get("env_state") if isinstance(kernel_eval_result, dict) else None
    if not isinstance(raw_env_state, dict):
        raw_env_state = kernel_eval_result
    if not isinstance(raw_env_state, dict):
        raise TypeError("Kernel eval result must be a dict or contain dict env_state.")
    # Runtime evidence belongs to reward attribution, not the model's repair
    # context. Separate it before feedback normalization and scalar scoring.
    raw_metadata = raw_env_state.get("metadata")
    runtime_graph = None
    if isinstance(raw_metadata, dict) and "runtime_graph" in raw_metadata:
        runtime_graph = raw_metadata["runtime_graph"]
        raw_env_state = {
            **raw_env_state,
            "metadata": {key: value for key, value in raw_metadata.items() if key != "runtime_graph"},
        }
        if not isinstance(runtime_graph, dict):
            runtime_graph = {
                "schema": "kernelgym-runtime-graph/v1",
                "status": "invalid",
                "unknowns": ["response_runtime_graph_not_object"],
            }
    elif getattr(args, "component_reward", False):
        runtime_graph = {
            "schema": "kernelgym-runtime-graph/v1",
            "status": "unavailable",
            "unknowns": ["missing_server_graph"],
        }
    normalized_env_state, env_extra_info = normalize_env_feedback(raw_env_state)
    result = {
        "env_state": normalized_env_state,
        "env_extra_info": env_extra_info,
        "reward_extra_info": normalized_env_state,
    }
    if runtime_graph is not None:
        result["runtime_graph"] = runtime_graph
    return result


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
    turn_sample.group_id = base_sample.group_id if base_sample.group_id is not None else base_sample.index
    turn_sample.loss_mask = [1] * len(response_ids)
    turn_sample.metadata = dict(turn_sample.metadata or {})
    env_extra_info = env_result.get("env_extra_info")
    if not isinstance(env_extra_info, dict):
        env_state = env_result.get("env_state") if isinstance(env_result.get("env_state"), dict) else {}
        env_extra_info = _extract_env_extra_info(env_state)
    turn_sample.metadata.update(
        {
            "turn_idx": turn_idx,
            "env_result": env_result,
            "env_extra_info": env_extra_info,
        }
    )
    turn_sample.metadata.pop("runtime_graph", None)
    if "runtime_graph" in env_result:
        turn_sample.metadata["runtime_graph"] = env_result["runtime_graph"]
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
        fake_sample.group_id = base_sample.group_id if base_sample.group_id is not None else base_sample.index
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
    aborted.group_id = sample.group_id if sample.group_id is not None else sample.index
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
    training Sample. Environment feedback becomes part of the next turn prompt,
    not the current response. Compatible Qwen runs can retain the previous
    prompt and generated token IDs directly instead of rendering history again.
    """

    state = GenerateState(args)
    messages = _as_messages(sample.prompt)
    preserve_history_thinking = bool(getattr(args, "preserve_history_thinking", False))
    if preserve_history_thinking and not state._is_qwen3_5_model():
        raise ValueError("--preserve-history-thinking is currently implemented only for Qwen3.5/Qwen3.8.")
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
    next_prompt_ids: list[int] | None = None
    next_exact_prefix_tokens: int | None = None
    next_inserted_close_think = False

    for turn_idx in range(max_turns):
        exact_prefix_tokens = next_exact_prefix_tokens
        inserted_close_think = next_inserted_close_think
        if next_prompt_ids is None:
            prompt_text = state.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **state.apply_chat_template_kwargs,
            )
            prompt_ids = state.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        else:
            prompt_ids = next_prompt_ids
            # Decode only for optional human-readable logging. The model request
            # consumes prompt_ids directly, so this cannot perturb the prefix.
            prompt_text = (
                ""
                if CUDA_AGENT_CONFIGS.get("log_rollout_stats_only", False)
                else state.tokenizer.decode(prompt_ids, skip_special_tokens=False)
            )
        max_context_len = _context_len_for_turn(args, turn_idx)
        if max_context_len is not None and len(prompt_ids) >= max_context_len:
            sample.status = Sample.Status.TRUNCATED
            finish_reason = "prompt_truncated"
            logger.warning("CUDA agent prompt exceeds context length at turn %s: %s", turn_idx, len(prompt_ids))
            break
        turn_sampling_params = _sampling_params_for_prompt_context(
            args,
            sampling_params,
            len(prompt_ids),
            turn_idx=turn_idx,
        )
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
        output = await post(
            url,
            payload,
            max_retries=KERNEL_AGENT_GENERATE_MAX_RETRIES,
            headers=_sglang_routing_headers(args, sample),
        )
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
        if exact_prefix_tokens is not None:
            turn_sample.metadata.update(
                {
                    "history_prefix_mode": "token_ids",
                    "history_exact_prefix_tokens": exact_prefix_tokens,
                    "history_prompt_tokens": len(prompt_ids),
                    "history_inserted_close_think": inserted_close_think,
                }
            )
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
            "cached_tokens": output["meta_info"].get("cached_tokens"),
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
                "content": _sanitize_assistant_history_content(response),
            }
        )

        model_feedback_stats: dict[str, Any] = {}
        format_feedback = _apply_feedback_template(env_result, template, feedback_stats=model_feedback_stats)
        turn_log["format_feedback"] = format_feedback
        turn_log["model_feedback_stats"] = model_feedback_stats

        if _is_done(env_result, turn_idx, max_turns):
            finish_reason = "env_done" if turn_idx + 1 < max_turns else "max_turns"
            break

        messages.append({"role": "user", "content": format_feedback})
        if preserve_history_thinking:
            enable_thinking = state.apply_chat_template_kwargs.get("enable_thinking", True) is not False
            next_prompt_ids, next_exact_prefix_tokens, next_inserted_close_think = _build_qwen_next_turn_prompt_ids(
                state.tokenizer,
                prompt_ids,
                response_ids,
                format_feedback,
                enable_thinking=enable_thinking,
            )
            turn_log.update(
                {
                    "next_turn_prompt_tokens": len(next_prompt_ids),
                    "next_turn_exact_prefix_tokens": next_exact_prefix_tokens,
                    "next_turn_inserted_close_think": next_inserted_close_think,
                }
            )

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
