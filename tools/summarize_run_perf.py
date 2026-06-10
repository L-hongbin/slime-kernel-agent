#!/usr/bin/env python3
"""Summarize selected perf metrics from slime run logs."""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections.abc import Iterable
from pathlib import Path


METRICS = (
    "perf/step_time",
    "perf/actor_train_tflops",
    "perf/rollout_time",
)

METRIC_RE = re.compile(
    r"(?P<quote>['\"])(?P<metric>perf/(?:step_time|actor_train_tflops|rollout_time))(?P=quote)"
    r"\s*:\s*"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def extract_metrics(paths: Iterable[Path]) -> dict[str, list[float]]:
    values = {metric: [] for metric in METRICS}
    for path in paths:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                for match in METRIC_RE.finditer(line):
                    values[match.group("metric")].append(float(match.group("value")))
    return values


def summarize(values: dict[str, list[float]]) -> list[dict[str, object]]:
    rows = []
    for metric in METRICS:
        metric_values = values[metric]
        if not metric_values:
            rows.append(
                {
                    "metric": metric,
                    "count": 0,
                    "average": None,
                    "median": None,
                    "maximum": None,
                }
            )
            continue

        rows.append(
            {
                "metric": metric,
                "count": len(metric_values),
                "average": statistics.fmean(metric_values),
                "median": statistics.median(metric_values),
                "maximum": max(metric_values),
            }
        )
    return rows


def format_number(value: object, precision: int) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{precision}f}"


def format_table(rows: list[dict[str, object]], precision: int = 6) -> str:
    headers = ("metric", "count", "average", "median", "maximum")
    rendered_rows = [
        [str(row["metric"]), format_number(row["count"], precision)]
        + [format_number(row[column], precision) for column in headers[2:]]
        for row in rows
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rendered_rows))
        for i in range(len(headers))
    ]

    def border() -> str:
        return "+" + "+".join("-" * (width + 2) for width in widths) + "+"

    def render_row(cells: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells)) + " |"

    lines = [border(), render_row(list(headers)), border()]
    lines.extend(render_row(row) for row in rendered_rows)
    lines.append(border())
    return "\n".join(lines)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path, help="run.log path(s) to summarize")
    parser.add_argument("--precision", type=int, default=6, help="decimal places for numeric columns")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    missing = [str(path) for path in args.logs if not path.is_file()]
    if missing:
        print(f"error: log file not found: {', '.join(missing)}", file=sys.stderr)
        return 2

    values = extract_metrics(args.logs)
    print(format_table(summarize(values), precision=args.precision))
    if not any(values.values()):
        print("warning: no target perf metrics found", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
