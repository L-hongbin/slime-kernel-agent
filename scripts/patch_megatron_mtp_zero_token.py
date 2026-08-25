#!/usr/bin/env python3
"""Patch Megatron MTP loss normalization for empty local token slices.

Context-parallel and dynamically packed microbatches can leave a local rank
with a shifted MTP loss mask whose token count is zero.  The masked loss is
also zero, so clamping only the denominator preserves the mathematical result
while avoiding a 0/0 logging value (and protects the non-per-token loss path).
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_PATH = Path("/root/Megatron-LM/megatron/core/models/gpt/gpt_model.py")

_ROLL_BLOCK = """                loss_mask, num_tokens = roll_tensor(
                    loss_mask,
                    shifts=-1,
                    dims=-1,
                    cp_group=self.cp_group,
                    packed_seq_params=packed_seq_params,
                )

                # Compute mtp loss without storing logits to save memory.
"""

_PATCHED_ROLL_BLOCK = """                loss_mask, num_tokens = roll_tensor(
                    loss_mask,
                    shifts=-1,
                    dims=-1,
                    cp_group=self.cp_group,
                    packed_seq_params=packed_seq_params,
                )
                # A context-parallel/dynamic microbatch may own no valid tokens
                # after the one-token MTP shift.  Its masked loss is zero, so a
                # unit denominator preserves the result and avoids 0/0.
                safe_num_tokens = num_tokens.clamp_min(1.0)

                # Compute mtp loss without storing logits to save memory.
"""

_REPLACEMENTS = (
    ("torch.sum(mtp_loss) / num_tokens,", "torch.sum(mtp_loss) / safe_num_tokens,"),
    ("mtp_loss_scale * mtp_loss / num_tokens", "mtp_loss_scale * mtp_loss / safe_num_tokens"),
)


def _state(source: str) -> str:
    old_counts = [source.count(_ROLL_BLOCK), *(source.count(old) for old, _new in _REPLACEMENTS)]
    new_counts = [
        source.count(_PATCHED_ROLL_BLOCK),
        *(source.count(new) for _old, new in _REPLACEMENTS),
    ]
    if old_counts == [1, 1, 1] and new_counts == [0, 0, 0]:
        return "unpatched"
    if old_counts == [0, 0, 0] and new_counts == [1, 1, 1]:
        return "patched"
    raise RuntimeError(
        "Megatron GPTModel MTP source does not match the audited layout: "
        f"old_counts={old_counts}, new_counts={new_counts}"
    )


def patch_file(path: Path, *, check_only: bool = False) -> str:
    source = path.read_text(encoding="utf-8")
    state = _state(source)
    if check_only:
        if state != "patched":
            raise RuntimeError(f"MTP zero-token patch is not installed in {path}")
        return state

    if state == "unpatched":
        source = source.replace(_ROLL_BLOCK, _PATCHED_ROLL_BLOCK)
        for old, new in _REPLACEMENTS:
            source = source.replace(old, new)
        if _state(source) != "patched":
            raise RuntimeError("MTP zero-token patch failed post-write verification")
        path.write_text(source, encoding="utf-8")
        return "patched"

    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    state = patch_file(args.path, check_only=args.check)
    print(f"Megatron MTP zero-token normalization: {state} ({args.path})")


if __name__ == "__main__":
    main()
