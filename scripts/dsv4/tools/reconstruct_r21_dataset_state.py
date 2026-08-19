#!/usr/bin/env python3
"""Reconstruct the exact logical post-iter59 rollout dataset cursor.

The old async r21 process checkpointed model iter59 but was terminated before
the dataset state was written.  Its last durable state54 was serialized behind
generation55, so it is actually the post-rollout55 cursor.  Rollouts 56--59
each consumed two 16-prompt fetch batches (dynamic-filter drops 16/5/8/12),
which advances the cursor by 128 prompt groups / 2048 samples.

This tool refuses any source other than the audited state54 values and writes
atomically.  It exists so the binary ``.pt`` artifact is reproducible rather
than a hand-authored opaque file.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


EXPECTED_SOURCE = {
    "sample_offset": 2512,
    "epoch_id": 0,
    "sample_group_index": 2512,
    "sample_index": 40192,
    "metadata": {},
}

RECONSTRUCTED = {
    "sample_offset": 2640,
    "epoch_id": 0,
    "sample_group_index": 2640,
    "sample_index": 42240,
    "metadata": {},
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-state54", type=Path, required=True)
    parser.add_argument("--output-state59", type=Path, required=True)
    args = parser.parse_args()

    source = torch.load(args.source_state54, map_location="cpu", weights_only=False)
    if source != EXPECTED_SOURCE:
        raise RuntimeError(
            "refusing dataset reconstruction: state54 differs from audited "
            f"values; got={source!r} expected={EXPECTED_SOURCE!r}"
        )

    args.output_state59.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output_state59.with_name(f".{args.output_state59.name}.tmp.{os.getpid()}")
    torch.save(RECONSTRUCTED, tmp)
    os.replace(tmp, args.output_state59)
    print(f"wrote {args.output_state59}: {RECONSTRUCTED}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
