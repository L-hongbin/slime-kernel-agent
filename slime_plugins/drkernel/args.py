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
    return parser
