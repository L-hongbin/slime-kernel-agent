"""Regression tests for the multi-turn message accumulation pipeline.

These guard against the bug we shipped before commit (May 2026): SGLang returns
the assistant response with the trailing ``<|im_end|>`` still attached when
``skip_special_tokens=False`` + ``no_stop_trim=True``; if that raw text is
appended verbatim to ``messages`` and re-fed to ``tokenizer.apply_chat_template``
on the next turn, the chat template wraps it again and emits
``<|im_end|><|im_end|>``. Every T2/T3 prompt in our pre-fix evals contained
that double marker, and ~13% of assistant turns were only ``<|im_end|>``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("jinja2")
pytest.importorskip("torch")

from slime_plugins.drkernel.rollout import _CHAT_STOP_MARKERS, _strip_chat_stop_markers


# ---------------------------------------------------------------------------
# Pure-function tests for the sanitizer
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_strip_removes_trailing_im_end():
    raw = "<think>\nreasoning\n</think>\n\ncode\n```<|im_end|>"
    assert _strip_chat_stop_markers(raw) == "<think>\nreasoning\n</think>\n\ncode\n```"


@pytest.mark.unit
def test_strip_returns_empty_when_response_is_only_stop_marker():
    """Models occasionally emit only the stop token. Sanitizer collapses to ''."""
    assert _strip_chat_stop_markers("<|im_end|>") == ""
    assert _strip_chat_stop_markers("  <|im_end|>  \n") == ""


@pytest.mark.unit
def test_strip_removes_all_known_chat_markers():
    raw = "a<|im_end|>b<|im_start|>c<|endoftext|>d"
    assert _strip_chat_stop_markers(raw) == "abcd"


@pytest.mark.unit
def test_strip_is_idempotent():
    raw = "code\n```<|im_end|>"
    once = _strip_chat_stop_markers(raw)
    twice = _strip_chat_stop_markers(once)
    assert once == twice


@pytest.mark.unit
def test_strip_handles_non_string():
    assert _strip_chat_stop_markers(None) == ""
    assert _strip_chat_stop_markers(123) == ""


@pytest.mark.unit
def test_strip_marker_set_is_complete():
    """If this assertion fires the chat-template special-token surface changed
    and `_strip_chat_stop_markers` likely needs an extension."""
    assert set(_CHAT_STOP_MARKERS) == {"<|im_end|>", "<|endoftext|>", "<|im_start|>"}


# ---------------------------------------------------------------------------
# End-to-end round-trip through the Qwen3.6 chat template
# ---------------------------------------------------------------------------


_QWEN_PATH = Path("/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B")


def _qwen_tokenizer():
    if not _QWEN_PATH.exists():
        pytest.skip(f"Qwen3.6-27B tokenizer not present at {_QWEN_PATH}")
    transformers = pytest.importorskip("transformers")
    return transformers.AutoTokenizer.from_pretrained(str(_QWEN_PATH), trust_remote_code=True)


def _simulated_sglang_response(body: str) -> str:
    """Mimic SGLang output with skip_special_tokens=False + no_stop_trim=True."""
    return f"{body}<|im_end|>"


def _build_two_turn_history(assistant_body: str) -> list[dict[str, str]]:
    """[user, assistant(T1 response), user(feedback)]; chat_template would render this
    as the prompt the model sees at turn 2."""
    return [
        {"role": "user", "content": "solve problem X"},
        {"role": "assistant", "content": assistant_body},
        {"role": "user", "content": "feedback: kernel failed compilation"},
    ]


@pytest.mark.unit
def test_t2_prompt_has_no_double_im_end_after_strip():
    """Reproduces the bug + verifies the fix."""
    tok = _qwen_tokenizer()
    raw = _simulated_sglang_response("code body")

    # Bug path: append raw response, render → produces <|im_end|><|im_end|>
    buggy = _build_two_turn_history(assistant_body=raw)
    buggy_prompt = tok.apply_chat_template(buggy, tokenize=False, add_generation_prompt=True)
    assert "<|im_end|><|im_end|>" in buggy_prompt, (
        "Sanity: without the strip the bug must reproduce; if not, SGLang or the chat "
        "template changed and this test is no longer guarding what it was written for."
    )

    # Fixed path: sanitize before appending, render → exactly one closing marker per turn
    fixed = _build_two_turn_history(assistant_body=_strip_chat_stop_markers(raw))
    fixed_prompt = tok.apply_chat_template(fixed, tokenize=False, add_generation_prompt=True)
    assert "<|im_end|><|im_end|>" not in fixed_prompt
    # 3 messages → 3 closing markers (one per <|im_start|> block); generation prefix adds none.
    assert fixed_prompt.count("<|im_end|>") == 3


@pytest.mark.unit
def test_empty_assistant_response_does_not_corrupt_prompt():
    """If model emits only the stop token, sanitized content is "" — the resulting
    prompt must still be a well-formed assistant block, not a stray marker."""
    tok = _qwen_tokenizer()
    raw = _simulated_sglang_response("")  # SGLang returns just "<|im_end|>"

    sanitized = _strip_chat_stop_markers(raw)
    assert sanitized == ""

    hist = _build_two_turn_history(assistant_body=sanitized)
    prompt = tok.apply_chat_template(hist, tokenize=False, add_generation_prompt=True)
    # Exactly one assistant <|im_start|> block with empty body, plus its single closing marker.
    assert "<|im_start|>assistant\n<|im_end|>" in prompt
    assert "<|im_end|><|im_end|>" not in prompt
    assert prompt.count("<|im_end|>") == 3


@pytest.mark.unit
def test_strip_handles_double_marker_and_whitespace_input():
    """Adversarial inputs: raw response already had `<|im_end|><|im_end|>` or is whitespace-only."""
    assert _strip_chat_stop_markers("code<|im_end|><|im_end|>") == "code"
    assert _strip_chat_stop_markers("   \n\t  ") == ""
    assert _strip_chat_stop_markers("<|im_end|>\n<|im_end|>\n") == ""


@pytest.mark.unit
def test_generate_multi_turn_eval_sample_uses_sanitizer_on_response():
    """Source-level guardrail: the assistant-append call site in the multi-turn driver
    must call _strip_chat_stop_markers(sample.response). A revert to raw
    `sample.response` would reintroduce the `<|im_end|><|im_end|>` bug; this test
    catches that revert even though the helper itself would still pass its own tests.
    """
    import re as _re

    source = (Path(__file__).resolve().parents[2] / "slime_plugins" / "drkernel" / "rollout.py").read_text()
    # Match `messages.append({..."role": "assistant"..."content": _strip_chat_stop_markers(sample.response)...})`
    pattern = _re.compile(
        r'messages\.append\(\s*\{\s*"role"\s*:\s*"assistant"\s*,\s*'
        r'"content"\s*:\s*_strip_chat_stop_markers\(\s*sample\.response\s*\)',
        _re.DOTALL,
    )
    assert pattern.search(source), (
        "rollout.py must wrap sample.response in _strip_chat_stop_markers() when "
        "appending the assistant turn to messages history. Reverting to raw "
        "sample.response will reintroduce the <|im_end|><|im_end|> duplication bug."
    )


@pytest.mark.unit
def test_three_turn_round_trip_marker_count():
    """After 3 simulated assistant turns the final T3-input prompt has exactly one
    closing marker per im_start block. Catches accumulation drift across turns."""
    tok = _qwen_tokenizer()
    messages: list[dict[str, str]] = [{"role": "user", "content": "T1 user"}]

    for turn_idx in range(3):
        raw = _simulated_sglang_response(f"T{turn_idx + 1} body")
        messages.append({"role": "assistant", "content": _strip_chat_stop_markers(raw)})
        if turn_idx < 2:
            messages.append({"role": "user", "content": f"T{turn_idx + 2} feedback"})

    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # 6 messages total → 6 im_start blocks → 6 im_end markers, no doubles.
    assert "<|im_end|><|im_end|>" not in rendered
    assert rendered.count("<|im_start|>") == 6
    assert rendered.count("<|im_end|>") == 6
