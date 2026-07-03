"""Regression tests for gpt-oss harmony multi-turn history sanitization.

gpt-oss returns assistant turns with literal `<|channel|>` control tokens
(skip_special_tokens=False). Replaying them verbatim through the harmony chat
template aborts every multi-turn rollout. `_sanitize_assistant_history_content`
reduces such a turn to its final-channel answer and is a no-op otherwise.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from examples.kernel_agent.generate_with_cuda_agent import _sanitize_assistant_history_content as sanitize


def test_noop_for_plain_text():
    # Non-harmony models (qwen / musacoder / plain markdown) must be untouched.
    text = "### MODEL_NEW\n```python\ny = 2\n```"
    assert sanitize(text) == text


def test_extracts_final_channel():
    text = (
        "<|channel|>analysis<|message|>let me think<|end|>"
        "<|start|>assistant<|channel|>final<|message|>### MODEL_NEW\n```python\nx = 1\n```<|end|>"
    )
    assert sanitize(text) == "### MODEL_NEW\n```python\nx = 1\n```"


def test_skips_commentary_takes_final():
    text = (
        "<|channel|>analysis<|message|>a<|end|>"
        "<|start|>assistant<|channel|>commentary<|message|>note<|end|>"
        "<|start|>assistant<|channel|>final<|message|>FINAL<|end|>"
    )
    assert sanitize(text) == "FINAL"


def test_truncated_analysis_strips_control_tokens():
    # No final channel (response cut off mid-analysis): drop control tokens so the
    # result no longer contains <|channel|> and the template accepts it.
    text = "<|channel|>analysis<|message|>still reasoning, no final yet"
    out = sanitize(text)
    assert "<|channel|>" not in out
    assert "still reasoning" in out


def test_result_never_contains_channel_token():
    samples = [
        "<|channel|>final<|message|>ok<|end|>",
        "<|channel|>analysis<|message|>x<|end|><|start|>assistant<|channel|>final<|message|>y<|end|>",
        "<|channel|>analysis<|message|>unterminated",
    ]
    for s in samples:
        assert "<|channel|>" not in sanitize(s)


@pytest.mark.parametrize("text", ["", "hello world", "no tags here at all"])
def test_passthrough_variants(text):
    assert sanitize(text) == text


def test_multiple_finals_last_empty_keeps_last_nonempty():
    # A trailing empty final must not blow away the real answer.
    text = "<|channel|>final<|message|>GOOD<|end|>" "<|start|>assistant<|channel|>final<|message|>   <|end|>"
    assert sanitize(text) == "GOOD"


def test_final_content_quoting_control_tokens_is_truncated_safely():
    # Final answer that quotes a harmony token: we bias toward avoiding a
    # template crash; the result must not retain <|channel|>.
    text = '<|channel|>final<|message|>print("<|channel|>")<|end|>'
    out = sanitize(text)
    assert "<|channel|>" not in out
