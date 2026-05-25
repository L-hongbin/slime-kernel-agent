#!/usr/bin/env python3
"""Pre-launch sanity check for the DrKernel first-turn prompt.

Renders one synthetic first-turn prompt with the current production renderer +
profile and asserts:
  - no jinja control tokens leak (`{#`, `#}`, `{{`, `}}`)
  - no obvious word-split bug in the "Target environment:" block (e.g. a
    multi-word GPU like "NVIDIA A800-SXM4-80GB" surviving as just "NVIDIA")

Designed to be the LAST guard before launching a 50-min eval — exits non-zero
if anything looks wrong so the run script can `set -e` out.

Usage (from debug.27b.sh, after building DRKERNEL_PLUGIN_ARGS):
    python3 scripts/debug/render_prompt_check.py \\
        --hf-checkpoint "${MODEL_DIR}" \\
        --drkernel-gpu-name "${DRKERNEL_GPU_NAME:-}" \\
        --drkernel-compiler-name "${DRKERNEL_COMPILER_NAME:-}" \\
        --drkernel-extra-environment "${DRKERNEL_EXTRA_ENVIRONMENT:-}" \\
        --expected-gpu-words "${DRKERNEL_GPU_NAME:-}"
"""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from types import SimpleNamespace


def _build_renderer(hf_checkpoint: str):
    # Defer heavy imports so --help works without transformers / torch.
    from slime_plugins.drkernel.rollout import DrKernelPromptRenderer

    return DrKernelPromptRenderer(hf_checkpoint=hf_checkpoint)


def _build_sample():
    return SimpleNamespace(
        index=0,
        prompt=(
            "import torch\n"
            "import torch.nn as nn\n\n"
            "class Model(nn.Module):\n"
            "    def __init__(self):\n"
            "        super().__init__()\n\n"
            "    def forward(self, x):\n"
            "        return x * 2\n"
        ),
        metadata={},
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-checkpoint", required=True)
    ap.add_argument("--drkernel-compiler-name", default=None)
    ap.add_argument("--drkernel-gpu-name", default=None)
    ap.add_argument("--drkernel-extra-environment", default=None)
    ap.add_argument(
        "--expected-gpu-words",
        default=None,
        help=(
            "If set, every space-separated word in this string must appear "
            "in the rendered prompt. Catches bash array-quoting bugs that "
            "truncate multi-word values like 'NVIDIA A800-SXM4-80GB' to "
            "just 'NVIDIA'."
        ),
    )
    ap.add_argument("--print-prompt", action="store_true", help="dump rendered prompt to stdout")
    args = ap.parse_args()

    renderer = _build_renderer(args.hf_checkpoint)
    sample = _build_sample()
    fake_args = Namespace(
        drkernel_compiler_name=args.drkernel_compiler_name or None,
        drkernel_gpu_name=args.drkernel_gpu_name or None,
        drkernel_extra_environment=args.drkernel_extra_environment or None,
    )
    result = renderer.render_sample(fake_args, sample, rollout_id=0)
    prompt = result.prompt

    failures: list[str] = []

    for marker in ("{#", "#}", "{{", "}}"):
        if marker in prompt:
            tail = prompt[max(0, prompt.find(marker) - 60) : prompt.find(marker) + 80]
            failures.append(f"jinja marker {marker!r} leaked into prompt around: ...{tail!r}...")

    if args.expected_gpu_words:
        words = args.expected_gpu_words.split()
        missing = [w for w in words if w not in prompt]
        if missing:
            failures.append(
                f"expected GPU words missing from prompt: {missing!r}. "
                f"Likely a bash array-quoting bug (unquoted ${{ARR[@]}}). "
                f"Got 'Target environment' block: "
                f"{_extract_env_block(prompt)!r}"
            )

    if args.print_prompt:
        print("=" * 80)
        print("RENDERED FIRST-TURN PROMPT:")
        print("=" * 80)
        print(prompt)
        print("=" * 80)

    if failures:
        print("RENDER CHECK FAILED:", file=sys.stderr)
        for line in failures:
            print(f"  - {line}", file=sys.stderr)
        sys.exit(1)

    # On success, print a one-liner so debug.sh logs show the check ran.
    has_env = "Target environment:" in prompt
    print(
        f"render_prompt_check: OK (prompt_len={len(prompt)}, has_target_env_block={has_env})",
        flush=True,
    )


def _extract_env_block(prompt: str) -> str:
    head = "Target environment:"
    if head not in prompt:
        return "<no env block>"
    start = prompt.find(head)
    # Take ~6 lines worth.
    end = start
    for _ in range(8):
        nl = prompt.find("\n", end + 1)
        if nl == -1:
            end = len(prompt)
            break
        end = nl
    return prompt[start:end]


if __name__ == "__main__":
    main()
