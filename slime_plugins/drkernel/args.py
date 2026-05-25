"""DrKernel-specific CLI arguments.

Plug into slime via ``parse_args(add_custom_arguments=add_custom_arguments)``
from the training entrypoint.
"""


def add_custom_arguments(parser):
    parser.add_argument("--debugpy", action="store_true")
    parser.add_argument(
        "--eval-unified-seqlen",
        type=int,
        default=None,
        help=(
            "When set (e.g. 32768), eval rollout uses one shared token budget: prompts are truncated "
            "to this length, max_new_tokens is capped by this length, and prompt_tokens + max_new_tokens "
            "never exceeds this value. Also sets --eval-max-prompt-len and --eval-max-context-len to the same "
            "value after defaults are applied."
        ),
    )
    parser.add_argument(
        "--use-multi-turn",
        action="store_true",
        default=False,
        help=(
            "Enable multi-turn rollout in the DrKernel plugin. The custom generate function is expected to "
            "produce one Sample per turn (with metadata['turn_idx']) instead of a single final Sample."
        ),
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Maximum number of turns for multi-turn rollout. Required when --use-multi-turn is set.",
    )
    parser.add_argument(
        "--padding-turns",
        action="store_true",
        default=False,
        help=(
            "Pad trajectories shorter than --max-turns with placeholder samples so every rollout produces a "
            "fixed number of turns. Padded samples must be loss-masked downstream."
        ),
    )
    parser.add_argument(
        "--preserve-history-thinking",
        action="store_true",
        default=False,
        help=(
            "Preserve previous assistant <think> blocks when rendering multi-turn prompts. Custom generate "
            "functions should pass this through to tokenizer.apply_chat_template when supported."
        ),
    )
    parser.add_argument(
        "--multi-turn-gamma",
        type=float,
        default=1.0,
        help="Discount factor applied across turns by multi-turn advantage / reward folding logic.",
    )
    parser.add_argument(
        "--filter-by-last-turn",
        action="store_true",
        default=False,
        help=(
            "When --use-multi-turn is enabled, apply the dynamic sampling filter only to the last turn group; "
            "the decision keeps or drops every turn group from the same rollout."
        ),
    )
    # Deferred import: rollout.py pulls in heavy deps (jinja2, transformers, slime
    # internals). We only need the default-value constant here, and parse_args
    # callers don't otherwise need rollout.py to be imported eagerly.
    from slime_plugins.drkernel.rollout import DEFAULT_KERNELGYM_ERROR_SUMMARY_CHARS

    parser.add_argument(
        "--kernelgym-error-summary-chars",
        type=int,
        default=DEFAULT_KERNELGYM_ERROR_SUMMARY_CHARS,
        help=(
            "Character limit for the summarized error_message field that gets injected into the next-turn "
            "tool_response. The summarizer keeps a few nvcc error / stack-frame lines and drops raw stderr / "
            "internal KernelGym profiling. ~400 tokens by default — covers most failure modes without flooding "
            "the prompt. See slime_plugins/drkernel/design-docs/feedback_summarization.md."
        ),
    )
    # First-turn target-environment hints that flow into the prompt layout's
    # "Target environment:" block (NVCC / GPU / extra notes). Per-sample
    # metadata can override these (e.g. mixed-GPU training); CLI args set the
    # eval-time default. Leave empty to suppress the block entirely.
    parser.add_argument(
        "--drkernel-compiler-name",
        type=str,
        default=None,
        help='Compiler version string injected into the prompt as "NVCC: <value>". E.g. "CUDA 12.9, V12.9.86".',
    )
    parser.add_argument(
        "--drkernel-gpu-name",
        type=str,
        default=None,
        help='GPU identification string injected into the prompt as "GPU: <value>". E.g. "NVIDIA A800-SXM4-80GB (SM 8.0)".',
    )
    parser.add_argument(
        "--drkernel-extra-environment",
        type=str,
        default=None,
        help="Free-form additional environment notes appended after NVCC/GPU lines (e.g. tvm_ffi version, sccache state).",
    )
    return parser
