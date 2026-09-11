from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from slime.rollout.sglang_rollout import PromptTemplate

try:
    from .config import CUDA_AGENT_CONFIGS
    from .utils import _truncate_middle
except ImportError:
    from config import CUDA_AGENT_CONFIGS
    from utils import _truncate_middle


def as_messages(prompt: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(prompt, list):
        return deepcopy(prompt)
    return [{"role": "user", "content": str(prompt)}]


def format_feedback(env_result: dict[str, Any], response_template: PromptTemplate) -> str:
    feedback_dict = env_result.get("env_state") or env_result
    try:
        feedback = json.dumps(feedback_dict, ensure_ascii=False, indent=2)
    except TypeError:
        feedback = str(feedback_dict)

    max_chars = int(CUDA_AGENT_CONFIGS["max_feedback_chars"])
    feedback = _truncate_middle(feedback, max_chars)
    return response_template.format(feedback=feedback, feedback_dict=feedback_dict)
