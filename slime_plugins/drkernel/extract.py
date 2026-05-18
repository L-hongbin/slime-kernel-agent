"""Kernel submission extraction helpers for DrKernel rollouts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "CUDA_AGENT_SECTION_NAMES",
    "CUDA_AGENT_SECTION_ORDER",
    "KernelSubmission",
    "KernelSubmissionExtractError",
    "MISSING_COMPLETE_SUBMISSION_ERROR",
    "extract_assistant_response",
    "extract_kernel_submission",
    "strip_think_blocks",
]

CUDA_KERNEL_BACKEND = "cuda_agent"

CUDA_AGENT_SECTION_ORDER = (
    ("CUDA_KERNELS", "cpp"),
    ("APPLY_BINDINGS", "cpp"),
    ("MODEL_NEW", "python"),
)
CUDA_AGENT_SECTION_NAMES = tuple(section_name for section_name, _ in CUDA_AGENT_SECTION_ORDER)
MISSING_COMPLETE_SUBMISSION_ERROR = (
    f"missing complete CUDA-Agent submission sections: {', '.join(CUDA_AGENT_SECTION_NAMES)}"
)

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_END_RE = re.compile(r"</think\s*>", re.IGNORECASE)


class KernelSubmissionExtractError(ValueError):
    """Raised when an assistant response does not contain a usable submission."""


@dataclass(frozen=True)
class KernelSubmission:
    """Parsed kernel submission used by KernelGym.

    `code` is a single CUDA-Agent markdown submission containing CUDA sources,
    bindings, and `ModelNew`. KernelGym's CUDA-Agent backend parses those
    sections server-side.
    """

    code: str
    backend: str
    sections: dict[str, str]


def strip_think_blocks(text: str) -> str:
    """Return the final answer region after model reasoning."""

    think_end_matches = list(_THINK_END_RE.finditer(text))
    if think_end_matches:
        return text[think_end_matches[-1].end() :]
    return _THINK_BLOCK_RE.sub("", text)


def extract_assistant_response(messages: list[dict[str, Any]]) -> str:
    """Return the last assistant message content from a chat transcript."""

    for message in reversed(messages):
        if message.get("role") == "assistant":
            content = message.get("content", "")
            if not isinstance(content, str):
                raise TypeError("assistant message content must be a string")
            return content
    raise KernelSubmissionExtractError("missing assistant response")


def _section_pattern(section_name: str, language: str) -> re.Pattern[str]:
    if language == "python":
        language_pattern = r"(?:python|py)"
    else:
        language_pattern = r"(?:cpp|c\+\+|cxx|cuda|cu)?"
    return re.compile(
        rf"###\s*{section_name}\s*```{language_pattern}\s*\n(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )


def _find_last_complete_cuda_agent_group(text: str) -> dict[str, str]:
    """Return the last complete CUDA-Agent section group.

    A valid group must strictly follow the prompt contract order:
    `CUDA_KERNELS` -> `APPLY_BINDINGS` -> `MODEL_NEW`. If the response contains
    multiple complete submissions, the last one is treated as the final answer.
    """

    matches = {
        section_name: list(_section_pattern(section_name, language).finditer(text))
        for section_name, language in CUDA_AGENT_SECTION_ORDER
    }
    best_group: dict[str, str] = {}

    for cuda_match in matches["CUDA_KERNELS"]:
        binding_match = next(
            (match for match in matches["APPLY_BINDINGS"] if match.start() > cuda_match.end()),
            None,
        )
        if binding_match is None:
            continue

        model_match = next(
            (match for match in matches["MODEL_NEW"] if match.start() > binding_match.end()),
            None,
        )
        if model_match is None:
            continue

        best_group = {
            "CUDA_KERNELS": cuda_match.group(1).strip(),
            "APPLY_BINDINGS": binding_match.group(1).strip(),
            "MODEL_NEW": model_match.group(1).strip(),
        }

    return best_group


def extract_kernel_submission(text: str) -> KernelSubmission | None:
    """Extract a KernelGym-ready CUDA-Agent submission from an assistant response.

    Returns None when the response does not contain a complete three-section
    CUDA-Agent submission.
    """

    text = strip_think_blocks(text)
    sections = _find_last_complete_cuda_agent_group(text)
    if not sections:
        return None
    code = "\n\n".join(
        f"### {section_name}\n```{language}\n{sections[section_name]}\n```"
        for section_name, language in CUDA_AGENT_SECTION_ORDER
    )
    return KernelSubmission(
        code=code,
        backend=CUDA_KERNEL_BACKEND,
        sections=sections,
    )
