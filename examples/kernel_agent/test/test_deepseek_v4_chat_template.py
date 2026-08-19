"""Unit tests for the hand-written DeepSeek-V4-Flash chat template.

V4-Flash ships no jinja ``chat_template`` (it expects formatting via the bundled
``encoding/encoding_dsv4.py``). We hand-wrote one at
``examples/kernel_agent/prompt_config/deepseek_v4_chat_template.jinja`` and install
it as ``<ckpt>/chat_template.jinja``. Because it is hand-written, these tests pin
its exact output.

Two layers:
  * Hermetic  -- render the template with the same jinja settings transformers uses
                 and assert byte-exact equality to hard-coded ground-truth strings.
                 Runs anywhere (only needs jinja2).
  * Reference -- when the real checkpoint is present, compare the template (via the
                 real tokenizer) against ``encoding_dsv4.encode_messages`` for many
                 message shapes, and assert the deployed copy matches the repo copy.
"""

import os
from pathlib import Path

import pytest
from jinja2.exceptions import TemplateError
from jinja2.sandbox import ImmutableSandboxedEnvironment

REPO_ROOT = Path(__file__).resolve().parents[3]
TEMPLATE_PATH = REPO_ROOT / "examples/kernel_agent/prompt_config/deepseek_v4_chat_template.jinja"

CKPT = Path(
    os.environ.get(
        "DSV4_CHAT_TEMPLATE_CKPT",
        "/nfs/FM/checkpoints/DeepSeek-V4-Flash",
    )
)
ENC_DIR = CKPT / "encoding"

# DeepSeek-V4 special-token literals (verified against the tokenizer).
BOS = "<｜begin▁of▁sentence｜>"
EOS = "<｜end▁of▁sentence｜>"
USER = "<｜User｜>"
ASSIST = "<｜Assistant｜>"
MAX_EFFORT = (
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n"
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, "
    "rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n"
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, "
    "and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n"
)


def _raise_exception(message):
    raise ValueError(message)


def _render(messages, add_generation_prompt=True, **kwargs):
    """Render the template the way transformers' apply_chat_template does."""
    src = TEMPLATE_PATH.read_text()
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["raise_exception"] = _raise_exception
    return env.from_string(src).render(messages=messages, add_generation_prompt=add_generation_prompt, **kwargs)


U = {"role": "user", "content": "Q1"}
A = {"role": "assistant", "content": "ANS1"}
F = {"role": "user", "content": "FB1"}
S = {"role": "system", "content": "SYS"}


# ----------------------------- hermetic layer -----------------------------


def test_thinking_single_user():
    assert _render([U], enable_thinking=True) == f"{BOS}{USER}Q1{ASSIST}<think>"


def test_chat_single_user():
    assert _render([U], enable_thinking=False) == f"{BOS}{USER}Q1{ASSIST}</think>"


def test_system_then_user():
    assert _render([S, U], enable_thinking=True) == f"{BOS}SYS{USER}Q1{ASSIST}<think>"


def test_max_reasoning_effort_prefix_is_byte_exact():
    assert _render([S, U], enable_thinking=True, reasoning_effort="max") == (
        f"{BOS}{MAX_EFFORT}SYS{USER}Q1{ASSIST}<think>"
    )


def test_max_reasoning_effort_is_not_applied_in_chat_mode():
    got = _render([U], enable_thinking=False, reasoning_effort="max")
    assert got == _render([U], enable_thinking=False)


def test_high_reasoning_effort_matches_old_model_default():
    assert _render([U], enable_thinking=True, reasoning_effort="high") == _render([U], enable_thinking=True)


def test_invalid_reasoning_effort_fails_closed():
    with pytest.raises(ValueError, match="Invalid reasoning effort: low"):
        _render([U], enable_thinking=True, reasoning_effort="low")


def test_multiturn_thinking_drops_prior_reasoning():
    expected = f"{BOS}{USER}Q1{ASSIST}</think>ANS1{EOS}{USER}FB1{ASSIST}<think>"
    assert _render([U, A, F], enable_thinking=True) == expected


def test_default_is_thinking_when_kwarg_absent():
    # enable_thinking undefined -> reasoning model defaults to thinking.
    assert _render([U]).endswith(f"{ASSIST}<think>")


def test_no_generation_prompt():
    # Without a generation prompt the trailing assistant marker is omitted.
    out = _render([U, A], add_generation_prompt=False, enable_thinking=True)
    assert out == f"{BOS}{USER}Q1{ASSIST}</think>ANS1{EOS}"


def test_raw_response_content_is_stripped_to_final_answer():
    # The harness replays the raw skip_special_tokens=False response verbatim;
    # the template must strip prior reasoning (up to last </think>) and trailing eos.
    raw = f"MY_HIDDEN_REASONING</think>ANS1{EOS}"
    got = _render([U, {"role": "assistant", "content": raw}, F], enable_thinking=True)
    clean = _render([U, A, F], enable_thinking=True)
    assert got == clean


def test_chat_mode_answer_without_think_marker_kept_verbatim():
    # In chat mode the model emits just the answer (no </think>): keep it as-is.
    raw = "PLAIN_ANSWER"
    got = _render([U, {"role": "assistant", "content": raw}, F], enable_thinking=False)
    assert f"{ASSIST}</think>PLAIN_ANSWER{EOS}" in got


def test_result_never_leaks_double_think_or_double_eos():
    raw = f"reason</think>ans{EOS}"
    got = _render([U, {"role": "assistant", "content": raw}, F], enable_thinking=True)
    assert "</think></think>" not in got
    assert f"{EOS}{EOS}" not in got


def test_splits_on_first_think_so_answer_keeps_embedded_marker():
    # The structural </think> is the FIRST one (closes reasoning). If the answer
    # itself contains "</think>", it must be preserved, not truncated.
    raw = "the hidden reasoning</think>real answer </think> still answer"
    got = _render([U, {"role": "assistant", "content": raw}, F], enable_thinking=True)
    assert f"{ASSIST}</think>real answer </think> still answer{EOS}" in got
    assert "the hidden reasoning" not in got


def test_strips_only_terminal_eos_not_embedded():
    raw = f"reason</think>ans{EOS}tail{EOS}"  # ends with eos; inner eos must survive
    got = _render([U, {"role": "assistant", "content": raw}, F], enable_thinking=True)
    assert f"{ASSIST}</think>ans{EOS}tail{EOS}{USER}" in got
    # exactly one terminal eos for this turn (no doubling)
    assert f"tail{EOS}{EOS}" not in got


def test_clean_answer_without_markers_unchanged():
    raw = "### MODEL_NEW\n```python\nx=1\n```"
    got = _render([U, {"role": "assistant", "content": raw}, F], enable_thinking=True)
    assert f"{ASSIST}</think>{raw}{EOS}{USER}FB1" in got


def test_three_turns_with_raw_sglang_style_responses():
    r1 = f"reasoning one</think>ANS1{EOS}"
    r2 = f"reasoning two</think>ANS2{EOS}"
    msgs = [
        U,
        {"role": "assistant", "content": r1},
        F,
        {"role": "assistant", "content": r2},
        {"role": "user", "content": "FB2"},
    ]
    got = _render(msgs, enable_thinking=True)
    expected = f"{BOS}{USER}Q1{ASSIST}</think>ANS1{EOS}{USER}FB1" f"{ASSIST}</think>ANS2{EOS}{USER}FB2{ASSIST}<think>"
    assert got == expected


# --------------------------- reference layer ------------------------------

_HAVE_REFERENCE = ENC_DIR.exists() and (CKPT / "tokenizer.json").exists()
_HAVE_DEPLOYED_TEMPLATE = (CKPT / "chat_template.jinja").exists()
skip_no_reference = pytest.mark.skipif(not _HAVE_REFERENCE, reason="DeepSeek-V4-Flash reference encoder not present")
skip_no_deployed_template = pytest.mark.skipif(
    not _HAVE_DEPLOYED_TEMPLATE, reason="deployed DeepSeek-V4-Flash chat template not present"
)


@skip_no_deployed_template
def test_deployed_template_matches_repo_copy():
    deployed = (CKPT / "chat_template.jinja").read_text()
    assert deployed == TEMPLATE_PATH.read_text(), "checkpoint chat_template.jinja drifted from repo copy"


@skip_no_reference
def test_byte_exact_against_official_encoder():
    import sys

    sys.path.insert(0, str(ENC_DIR))
    import encoding_dsv4 as E  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402

    tok = AutoTokenizer.from_pretrained(str(CKPT))
    tok.chat_template = TEMPLATE_PATH.read_text()

    U1 = {"role": "user", "content": "optimize this kernel"}
    A1 = {"role": "assistant", "content": "### MODEL_NEW\n```python\nx=1\n```"}
    F1 = {"role": "user", "content": "compile failed: error X"}
    A2 = {"role": "assistant", "content": "### MODEL_NEW\n```python\nx=2\n```"}
    F2 = {"role": "user", "content": "still wrong"}
    Sy = {"role": "system", "content": "You are a CUDA expert."}

    shapes = [
        [U1],
        [Sy, U1],
        [U1, A1, F1],
        [U1, A1, F1, A2, F2],
        [Sy, U1, A1, F1],
    ]
    for effort in (None, "high", "max"):
        for thinking, mode in ((True, "thinking"), (False, "chat")):
            for msgs in shapes:
                ref = E.encode_messages([dict(m) for m in msgs], thinking_mode=mode, reasoning_effort=effort)
                got = tok.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=thinking,
                    reasoning_effort=effort,
                )
                assert ref == got, f"effort={effort} mode={mode} shape_len={len(msgs)}\n" f"REF={ref!r}\nGOT={got!r}"

    with pytest.raises(TemplateError, match="Invalid reasoning effort: low"):
        tok.apply_chat_template(
            [U1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
            reasoning_effort="low",
        )
