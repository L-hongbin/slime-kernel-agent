#!/usr/bin/env python3
"""Summarize SGLang decode throughput lines from slime run logs.

The script is intentionally read-only: it parses ``Decode batch`` lines, groups
them by ``#running-req``, and reports same-batch throughput / accept stats. This
avoids comparing EAGLE round costs across different live batch sizes.
"""

from __future__ import annotations

import argparse
import ast
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2}) (\d{2}):(\d{2}):(\d{2})")
DECODE_RE = re.compile(
    r"Decode batch, #running-req:\s*(?P<req>\d+),"
    r".*?cuda graph:\s*(?P<cuda_graph>True|False),"
    r"\s*gen throughput \(token/s\):\s*(?P<tput>[\d.]+)"
)
ACCEPT_LEN_RE = re.compile(r"accept len:\s*([\d.]+)")
ACCEPT_RATE_RE = re.compile(r"accept rate:\s*([\d.]+)")
EVAL_RE = re.compile(r"eval 0:\s*(\{.*\})")
PBAR_RE = re.compile(r"Eval (?P<name>\S+):\s*(?P<pct>\d+)%\|.*?\|\s*(?P<done>\d+)/(?P<total>\d+)")


@dataclass(frozen=True)
class DecodeRow:
    minute: float
    req: int
    tput: float
    accept_len: float | None
    accept_rate: float | None
    cuda_graph: bool

    @property
    def round_ms(self) -> float | None:
        if self.accept_len is None:
            return None
        return self.req * self.accept_len / self.tput * 1000.0

    @property
    def step_ms(self) -> float:
        return self.req / self.tput * 1000.0


def strip_ansi(line: str) -> str:
    return ANSI_RE.sub("", line)


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def parse_log(path: Path) -> tuple[list[DecodeRow], dict[str, float] | None, tuple[int, int] | None]:
    rows: list[DecodeRow] = []
    first_sec: int | None = None
    final_eval: dict[str, float] | None = None
    last_progress: tuple[int, int] | None = None

    for raw in path.read_text(errors="ignore").splitlines():
        line = strip_ansi(raw)

        if "Eval " in line:
            mp = PBAR_RE.search(line)
            if mp:
                last_progress = (int(mp.group("done")), int(mp.group("total")))

        if "eval 0:" in line:
            me = EVAL_RE.search(line)
            if me:
                try:
                    final_eval = ast.literal_eval(me.group(1))
                except (SyntaxError, ValueError):
                    pass

        if "Decode batch" not in line:
            continue
        md = DECODE_RE.search(line)
        mt = TS_RE.search(line)
        if not (md and mt):
            continue

        _date, hh, mm, ss = mt.groups()
        sec = int(hh) * 3600 + int(mm) * 60 + int(ss)
        if first_sec is None:
            first_sec = sec
        elapsed = sec - first_sec
        if elapsed < 0:
            elapsed += 24 * 3600

        ma = ACCEPT_LEN_RE.search(line)
        mr = ACCEPT_RATE_RE.search(line)
        rows.append(
            DecodeRow(
                minute=elapsed / 60.0,
                req=int(md.group("req")),
                tput=float(md.group("tput")),
                accept_len=float(ma.group(1)) if ma else None,
                accept_rate=float(mr.group(1)) if mr else None,
                cuda_graph=md.group("cuda_graph") == "True",
            )
        )

    return rows, final_eval, last_progress


def summarize_one(label: str, path: Path, min_req: int, top_exact: int) -> str:
    rows, final_eval, last_progress = parse_log(path)
    if not rows:
        return f"\n## {label}\npath: {path}\nno decode rows parsed\n"

    selected = [r for r in rows if r.req >= min_req]
    exact: dict[int, list[DecodeRow]] = defaultdict(list)
    for row in selected:
        exact[row.req].append(row)

    lines: list[str] = []
    lines.append(f"\n## {label}")
    lines.append(f"path: {path}")
    lines.append(f"decode_rows: {len(rows)} total, {len(selected)} with req>={min_req}")
    lines.append(
        "overall_decode: "
        f"req median={median([r.req for r in rows]):.0f} mean={mean([r.req for r in rows]):.1f}; "
        f"tput median={median([r.tput for r in rows]):.0f} mean={mean([r.tput for r in rows]):.0f} tok/s"
    )
    if any(r.accept_len is not None for r in rows):
        accept_lens = [r.accept_len for r in rows if r.accept_len is not None]
        accept_rates = [r.accept_rate for r in rows if r.accept_rate is not None]
        lines.append(
            "overall_spec: "
            f"accept_len median={median(accept_lens):.2f} mean={mean(accept_lens):.2f}; "
            f"accept_rate median={median(accept_rates):.2f} mean={mean(accept_rates):.2f}"
        )
    if final_eval:
        key = next((k for k in final_eval if k.startswith("eval/") and "/" not in k.removeprefix("eval/")), None)
        score = final_eval.get(key) if key else None
        spec_len = next((v for k, v in final_eval.items() if k.endswith("/spec_accept_length")), None)
        truncated = next((v for k, v in final_eval.items() if k.endswith("-truncated_ratio")), None)
        lines.append("final_eval: " f"score={score}; spec_accept_length={spec_len}; truncated={truncated}")
    elif last_progress:
        lines.append(f"final_eval: missing; last_progress={last_progress[0]}/{last_progress[1]}")

    lines.append("same-batch exact req buckets:")
    lines.append("req  n  tput_med  tput_mean  accept_med  round_ms_med  step_ms_med")
    top_reqs = sorted(exact, key=lambda r: (len(exact[r]), r), reverse=True)[:top_exact]
    for req in top_reqs:
        bucket = exact[req]
        tputs = [r.tput for r in bucket]
        accepts = [r.accept_len for r in bucket if r.accept_len is not None]
        rounds = [r.round_ms for r in bucket if r.round_ms is not None]
        steps = [r.step_ms for r in bucket]
        lines.append(
            f"{req:3d} {len(bucket):2d} {median(tputs):9.0f} {mean(tputs):10.0f} "
            f"{median(accepts):10.2f} {median(rounds):12.1f} {median(steps):11.1f}"
        )

    high = [r for r in rows if r.req >= min_req]
    if high:
        lines.append(
            f"req>={min_req}: n={len(high)} "
            f"tput_med={median([r.tput for r in high]):.0f} "
            f"tput_mean={mean([r.tput for r in high]):.0f} "
            f"accept_med={median([r.accept_len for r in high if r.accept_len is not None]):.2f} "
            f"round_ms_med={median([r.round_ms for r in high if r.round_ms is not None]):.1f} "
            f"step_ms_med={median([r.step_ms for r in high]):.1f}"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--label", action="append", default=[])
    parser.add_argument("--min-req", type=int, default=80)
    parser.add_argument("--top-exact", type=int, default=12)
    args = parser.parse_args()

    if args.label and len(args.label) != len(args.logs):
        parser.error("--label count must match log count")

    chunks = []
    for i, path in enumerate(args.logs):
        label = args.label[i] if args.label else path.parent.name
        chunks.append(summarize_one(label, path, args.min_req, args.top_exact))
    print("\n".join(chunks).strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
