from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from slime.rollout.sglang_rollout import PromptTemplate

try:
    from .config import CUDA_AGENT_CONFIGS
    from .utils import _truncate_middle, split_think_response
except ImportError:
    from config import CUDA_AGENT_CONFIGS
    from utils import _truncate_middle, split_think_response


def as_messages(prompt: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(prompt, list):
        return deepcopy(prompt)
    return [{"role": "user", "content": str(prompt)}]


def extract_verify_response(response: str) -> str | None:
    """Extract one non-empty VERIFY block outside reasoning without changing the response."""
    _, content = split_think_response(response)
    if "<think>" in content or content.count("<VERIFY>") != 1 or content.count("</VERIFY>") != 1:
        return None
    start = content.index("<VERIFY>")
    end = content.index("</VERIFY>")
    if end <= start or not content[start + len("<VERIFY>") : end].strip():
        return None
    return content[start : end + len("</VERIFY>")]


def format_feedback(
    env_result: dict[str, Any], response_template: PromptTemplate, *, verification: str | None = None
) -> str:
    feedback_dict = env_result.get("env_state") or env_result
    try:
        feedback = json.dumps(feedback_dict, ensure_ascii=False, indent=2)
    except TypeError:
        feedback = str(feedback_dict)

    max_chars = int(CUDA_AGENT_CONFIGS["max_feedback_chars"])
    feedback = _truncate_middle(feedback, max_chars)
    if verification is not None:
        feedback += f"\n\nVerification analysis for the next revision:\n{verification}"
        # Also expose the analysis to templates using feedback_dict; keep the source feedback unchanged.
        feedback_dict = {**feedback_dict, "verification": verification}
    return response_template.format(feedback=feedback, feedback_dict=feedback_dict)
