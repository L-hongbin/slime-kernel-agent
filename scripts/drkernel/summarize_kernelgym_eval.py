#!/usr/bin/env python3
"""Summarize compile/correctness/fast rates from KernelGym eval artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from slime_plugins.drkernel.eval_summary import DEFAULT_FAST_THRESHOLDS, load_records, summarize_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSON/JSONL/PT/PTH/PKL artifact containing KernelGym responses")
    parser.add_argument("--output", help="Optional path to write the JSON summary")
    parser.add_argument(
        "--fast-thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_FAST_THRESHOLDS),
        help="Speedup thresholds for fast@ metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.input)
    summary = summarize_records(records, fast_thresholds=args.fast_thresholds)
    text = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True)
    print(text)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
