"""Regression guard against jinja-comment leak in rendered prompts.

Before the May 2026 fix, `DrKernelPromptRenderer.render_sample` called
`_load_fragment` to read role / backend templates as raw text and then
string-injected them into the outer layout via Jinja. Because Jinja never
re-parsed the injected strings, any `{# ... #}` comment or unprocessed
`{{ var }}` placeholder in a role / backend template survived as literal
text in the final prompt sent to the model.

This silent leak confounded the v2 / v2.1 / v2.2 template ablations: the
"v2.2 = pure cleanup of v1 ..." documentation comment I had appended at the
end of `tvm_ffi_module_v2_2.jinja` ended up in every T1 prompt.

The fix renders role / backend fragments through Jinja before injecting
them into the layout. These tests pin that behavior so the leak can't
silently regress.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("jinja2")
pytest.importorskip("torch")

from slime_plugins.drkernel.rollout import DrKernelPromptRenderer


_REPO_ROOT = Path(__file__).resolve().parents[2]
_TEMPLATE_ROOT = _REPO_ROOT / "slime_plugins" / "drkernel" / "prompt_templates"
_QWEN_PATH = Path("/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B")


def _build_renderer():
    if not _QWEN_PATH.exists():
        pytest.skip(f"Qwen3.6-27B tokenizer not present at {_QWEN_PATH}")
    return DrKernelPromptRenderer(hf_checkpoint=str(_QWEN_PATH))


def _build_sample(problem: str = "import torch\n# placeholder problem\n"):
    """Minimal sample stub the renderer needs."""
    return SimpleNamespace(
        index=0,
        prompt=problem,
        metadata={},  # no overrides; renderer reads compiler/gpu/env from args fallback
    )


def _empty_args():
    return Namespace(
        drkernel_compiler_name=None,
        drkernel_gpu_name=None,
        drkernel_extra_environment=None,
    )


# ---------------------------------------------------------------------------
# Live-template render: every existing role + backend file must NOT leak
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_real_first_turn_prompt_has_no_jinja_leak():
    """Render the live `drkernel_v1_tvm_ffi` first-turn prompt and assert no
    jinja markers survive into the final text."""
    renderer = _build_renderer()
    sample = _build_sample()
    result = renderer.render_sample(_empty_args(), sample, rollout_id=0)

    rendered = result.prompt
    assert "{#" not in rendered, f"jinja comment-open leaked into prompt:\n{rendered[-500:]}"
    assert "#}" not in rendered, f"jinja comment-close leaked into prompt:\n{rendered[-500:]}"
    # Unprocessed `{{ ... }}` placeholders are equally bad (would mean a
    # fragment expects a variable the renderer didn't supply).
    assert "{{" not in rendered, f"unprocessed jinja placeholder leaked:\n{rendered[-500:]}"
    assert "}}" not in rendered, f"unprocessed jinja placeholder leaked:\n{rendered[-500:]}"


@pytest.mark.unit
def test_backend_with_jinja_comment_does_not_leak():
    """Synthetic case: a backend fragment containing a `{# ... #}` block must
    render to text that does NOT contain the comment in the final prompt.

    This pins the renderer's behavior of running each fragment through Jinja,
    even though current production fragments have no comments — the fix
    matters precisely for the case where a developer adds inline documentation
    as a Jinja comment (which is what triggered the bug).
    """
    renderer = _build_renderer()

    # Drop a fragment with a comment + a normal sentence into the template
    # tree, then rebuild a renderer that points at it.
    fragment_path = _TEMPLATE_ROOT / "backends" / "_test_jinja_leak.jinja"
    fragment_path.write_text(
        "Backend body sentinel line.\n" "\n" "{# this comment must be stripped before injection into the layout #}\n",
        encoding="utf-8",
    )
    try:
        # Point an existing backend candidate at our test fragment by
        # monkeypatching the profile.
        original_candidates = renderer.profile["backend"]["candidates"]
        renderer.profile["backend"]["candidates"] = [
            {
                "id": "_test_jinja_leak",
                "first_turn_text_path": "backends/_test_jinja_leak.jinja",
                "tool_response_text_path": "tool_response/tvm_ffi_short.jinja",
            }
        ]
        try:
            sample = _build_sample()
            result = renderer.render_sample(_empty_args(), sample, rollout_id=0)
            rendered = result.prompt
            assert "Backend body sentinel line." in rendered, "fragment text missing from render"
            assert "{#" not in rendered, "comment-open leaked"
            assert "#}" not in rendered, "comment-close leaked"
            assert "this comment must be stripped" not in rendered, "comment body leaked"
        finally:
            renderer.profile["backend"]["candidates"] = original_candidates
    finally:
        fragment_path.unlink(missing_ok=True)
