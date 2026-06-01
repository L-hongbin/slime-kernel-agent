"""Unit / sanity tests for the cache-friendly drkernel chat template.

Template under test:
    slime_plugins/drkernel/prompt_templates/chat_template_no_think_norm.jinja

Contract: re-rendering a multi-turn conversation must reproduce, token-for-token,
the sequence that was generated at inference time, so the SGLang radix prefix
cache hits on the previous turn. Concretely, for a turn-2 prompt built from
``[user, assistant(turn1), user(feedback)]``, the rendered+tokenized turn-2 ids
must start with ``turn1_prompt_ids + generated_ids`` (longest-common-prefix ==
full stored tree).

The stock Qwen3.6 template re-normalizes the thinking block on re-render
(split('</think>') / rstrip / lstrip / |trim / forced '\\n</think>\\n\\n'), which
breaks that prefix on common outputs. These tests pin the new template's verbatim
behavior AND assert the stock template breaks the same cases (so we know the new
template is solving a real problem, not a phantom one).

Requires the real Qwen3.6 tokenizer; skipped if absent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("jinja2")
pytest.importorskip("transformers")

from slime_plugins.drkernel.rollout import _strip_chat_stop_markers

_QWEN_PATH = Path("/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B")
_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[2] / "slime_plugins/drkernel/prompt_templates/chat_template_no_think_norm.jinja"
)

# Assistant bodies = exactly what the model generates AFTER the injected '<think>\n'
# (i.e. NOT including a leading '<think>'). The new template must reproduce ALL of
# these verbatim.
_CASES = {
    "clean": "Reasoning here.\n</think>\n\nAnswer code block.",
    "answer_mentions_close_tag": "Reasoning.\n</think>\n\nNote: emit </think> to end thinking. Code: x = 1",
    "trailing_space_before_close": "Reasoning ends weird   \n</think>\n\nAnswer.",
    "triple_newline_in_reasoning": "Step1.\n\n\nStep2.\n</think>\n\nAnswer.",
    "single_newline_after_close": "Reasoning.\n</think>\nAnswer right after.",
    "truncated_no_close_tag": "Reasoning that never closes and keeps going and",
}

# Subset that the STOCK template's re-normalization provably breaks (verified with
# the real Qwen3.6 tokenizer). 'clean' and 'triple_newline_in_reasoning' happen to
# survive the stock split/rstrip/trim, so they are not discriminating cases.
_STOCK_BREAKS = [
    "answer_mentions_close_tag",
    "trailing_space_before_close",
    "single_newline_after_close",
    "truncated_no_close_tag",
]


def _tokenizer():
    if not _QWEN_PATH.exists():
        pytest.skip(f"Qwen3.6-27B tokenizer not present at {_QWEN_PATH}")
    if not _TEMPLATE_PATH.exists():
        pytest.skip(f"cache template not present at {_TEMPLATE_PATH}")
    import transformers

    return transformers.AutoTokenizer.from_pretrained(str(_QWEN_PATH), trust_remote_code=True)


def _template() -> str:
    return _TEMPLATE_PATH.read_text()


def _enc(tok, text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False)


def _lcp(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        n += 1
    return n


def _turn2_messages(content: str) -> list[dict[str, str]]:
    return [
        {"role": "user", "content": "Write a CUDA kernel for relu(x)*2."},
        {"role": "assistant", "content": content},
        {"role": "user", "content": "Use this feedback to revise: nvcc failed."},
    ]


# ---------------------------------------------------------------------------
# Core contract: re-render reproduces generated tokens (radix prefix == full tree)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("name", list(_CASES))
def test_verbatim_render_gives_full_radix_reuse(name):
    tok = _tokenizer()
    tmpl = _template()
    body = _CASES[name]

    # turn-1 prompt (same template renders the generation prompt with '<think>\n')
    turn1_prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Write a CUDA kernel for relu(x)*2."}],
        tokenize=False,
        add_generation_prompt=True,
        chat_template=tmpl,
        preserve_thinking=True,
    )
    turn1_ids = _enc(tok, turn1_prompt)

    # what SGLang generated + what the radix tree stores for turn-1
    generated_ids = _enc(tok, body + "<|im_end|>")
    tree = turn1_ids + generated_ids

    # producer path: sanitize the raw response, append as history, re-render turn-2.
    # The minimal-diff template keeps the stock preserve_thinking gating, so the
    # verbatim keep-branch is only taken when preserve_thinking=True is passed.
    content = _strip_chat_stop_markers(body + "<|im_end|>")
    turn2_prompt = tok.apply_chat_template(
        _turn2_messages(content),
        tokenize=False,
        add_generation_prompt=True,
        chat_template=tmpl,
        preserve_thinking=True,
    )
    turn2_ids = _enc(tok, turn2_prompt)

    hit = _lcp(turn2_ids, tree)
    assert hit == len(tree), (
        f"[{name}] radix prefix broke at {hit}/{len(tree)} "
        f"(reused {max(0, hit - len(turn1_ids))}/{len(generated_ids)} generated tokens)"
    )


@pytest.mark.unit
@pytest.mark.parametrize("name", _STOCK_BREAKS)
def test_stock_template_breaks_so_new_template_is_justified(name):
    """Sanity: the stock template (preserve_thinking=True) does NOT reproduce these
    cases verbatim — confirms the new template solves a real problem. If this ever
    starts passing, the stock template changed and the custom one may be redundant.
    """
    tok = _tokenizer()
    body = _CASES[name]
    turn1_ids = _enc(
        tok,
        tok.apply_chat_template(
            [{"role": "user", "content": "Write a CUDA kernel for relu(x)*2."}],
            tokenize=False,
            add_generation_prompt=True,
        ),
    )
    generated_ids = _enc(tok, body + "<|im_end|>")
    tree = turn1_ids + generated_ids
    content = _strip_chat_stop_markers(body + "<|im_end|>")
    turn2_ids = _enc(
        tok,
        tok.apply_chat_template(
            _turn2_messages(content), tokenize=False, add_generation_prompt=True, preserve_thinking=True
        ),
    )
    hit = _lcp(turn2_ids, tree)
    assert hit < len(tree), f"[{name}] stock template unexpectedly reproduced verbatim ({hit}/{len(tree)})"


# ---------------------------------------------------------------------------
# Structural sanity
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_generation_prompt_injects_think():
    tok = _tokenizer()
    p = tok.apply_chat_template(
        [{"role": "user", "content": "hi"}],
        tokenize=False,
        add_generation_prompt=True,
        chat_template=_template(),
    )
    assert p.endswith("<|im_start|>assistant\n<think>\n")


@pytest.mark.unit
def test_history_assistant_keeps_thinking_verbatim():
    tok = _tokenizer()
    body = "my reasoning\n</think>\n\nmy answer"
    rendered = tok.apply_chat_template(
        _turn2_messages(body),
        tokenize=False,
        add_generation_prompt=True,
        chat_template=_template(),
        preserve_thinking=True,
    )
    # the full reasoning text survives in the re-rendered history (no stripping)
    assert "my reasoning" in rendered
    assert "<|im_start|>assistant\n<think>\nmy reasoning\n</think>\n\nmy answer<|im_end|>\n" in rendered


@pytest.mark.unit
def test_without_preserve_thinking_strips_history():
    """Documents the gating dependency: this minimal-diff template only renders the
    history verbatim when preserve_thinking=True. Without it, the stock gating takes
    the strip branch (because the drkernel feedback is a plain user turn that moves
    last_query_index past the latest assistant), so the reasoning is dropped and the
    cache benefit is lost. If this ever changes, the launcher's preserve_thinking
    requirement may be revisitable.
    """
    tok = _tokenizer()
    body = "my reasoning\n</think>\n\nmy answer"
    rendered = tok.apply_chat_template(
        _turn2_messages(body), tokenize=False, add_generation_prompt=True, chat_template=_template()
    )
    assert "my reasoning" not in rendered  # thinking stripped without the flag


@pytest.mark.unit
def test_marker_counts_well_formed():
    tok = _tokenizer()
    body = "reasoning\n</think>\n\nanswer"
    # full conversation render (no generation prompt): 2 user + 1 assistant turns
    rendered = tok.apply_chat_template(
        _turn2_messages(body),
        tokenize=False,
        add_generation_prompt=False,
        chat_template=_template(),
        preserve_thinking=True,
    )
    assert "<|im_end|><|im_end|>" not in rendered
    assert rendered.count("<|im_start|>") == 3  # 2 user + 1 assistant
    assert rendered.count("<|im_end|>") == 3
    assert rendered.count("<think>") == 1  # only the single historical assistant turn


@pytest.mark.unit
def test_missing_user_query_raises():
    tok = _tokenizer()
    # The template raises inside jinja; transformers wraps it differently across
    # versions, so the concrete type is not stable — assert any failure.
    with pytest.raises(Exception):  # noqa: B017
        tok.apply_chat_template(
            [{"role": "assistant", "content": "x\n</think>\n\ny"}],
            tokenize=False,
            add_generation_prompt=True,
            chat_template=_template(),
        )
