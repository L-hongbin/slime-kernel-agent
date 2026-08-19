#!/usr/bin/env python3
"""Compare repeated non-spec, MTP depth, and optional DSpark frontier sweeps."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def load(paths: list[Path]) -> dict[int, list[dict]]:
    rows: dict[int, list[dict]] = defaultdict(list)
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    rows[int(row["batch_size"])].append(row)
    return dict(rows)


def metric(rows: list[dict], key: str) -> float:
    return mean(float(row[key]) for row in rows)


def assert_same_batches(arms: dict[str, dict[int, list[dict]]]) -> list[int]:
    expected: set[int] | None = None
    for name, rows in arms.items():
        if expected is None:
            expected = set(rows)
        elif set(rows) != expected:
            raise ValueError(f"batch-size mismatch for {name}: expected={sorted(expected)}, " f"actual={sorted(rows)}")
    return sorted(expected or ())


def repetition_counts(arms: dict[str, dict[int, list[dict]]]) -> dict[str, int]:
    counts = {}
    for name, rows in arms.items():
        arm_counts = {len(repetitions) for repetitions in rows.values()}
        if len(arm_counts) != 1:
            raise ValueError(f"inconsistent repetition counts for {name}: {arm_counts}")
        counts[name] = arm_counts.pop()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, nargs="+", required=True)
    parser.add_argument("--mtp-f1", type=Path, nargs="+", required=True)
    parser.add_argument("--mtp-f3", type=Path, nargs="+", required=True)
    parser.add_argument("--dspark", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    arms = {
        "baseline": load(args.baseline),
        "mtp_f1": load(args.mtp_f1),
        "mtp_f3": load(args.mtp_f3),
    }
    if args.dspark:
        arms["dspark"] = load(args.dspark)

    rows = []
    for batch_size in assert_same_batches(arms):
        baseline = metric(arms["baseline"][batch_size], "output_throughput")
        f1 = metric(arms["mtp_f1"][batch_size], "output_throughput")
        f3 = metric(arms["mtp_f3"][batch_size], "output_throughput")
        best_name, best = max((("mtp_f1", f1), ("mtp_f3", f3)), key=lambda x: x[1])
        row = {
            "batch_size": batch_size,
            "baseline_output_throughput": round(baseline, 2),
            "mtp_f1_output_throughput": round(f1, 2),
            "mtp_f1_ratio": round(f1 / baseline, 4),
            "mtp_f1_accept_length": round(metric(arms["mtp_f1"][batch_size], "acc_length"), 3),
            "mtp_f3_output_throughput": round(f3, 2),
            "mtp_f3_ratio": round(f3 / baseline, 4),
            "mtp_f3_accept_length": round(metric(arms["mtp_f3"][batch_size], "acc_length"), 3),
            "mtp_best_config": best_name,
            "mtp_best_output_throughput": round(best, 2),
            "mtp_best_ratio": round(best / baseline, 4),
        }
        if "dspark" in arms:
            dspark = metric(arms["dspark"][batch_size], "output_throughput")
            row.update(
                {
                    "dspark_output_throughput": round(dspark, 2),
                    "dspark_ratio_vs_base_baseline": round(dspark / baseline, 4),
                    "dspark_accept_length": round(metric(arms["dspark"][batch_size], "acc_length"), 3),
                }
            )
        rows.append(row)

    result = {"repetitions": repetition_counts(arms), "rows": rows}
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
