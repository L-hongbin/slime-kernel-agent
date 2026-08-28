"""Backfill the unified-denominator eval summary into existing summary logs.

Older `summary.*.txt` logs were produced by an earlier `summarize_eval.py` whose
per-turn (Tk) columns each used their OWN denominator (the count of trajectories
that actually produced that turn). The current `summarize_eval.py` uses ONE
denominator for every column — the trajectory total — so Tk and best are directly
comparable.

This script walks an input directory, finds each existing `summary*.txt` log,
locates the eval dump that sits next to it (same directory as the original run),
recomputes the table with the current (unified-denominator) `summarize_eval`
logic, and APPENDS the recomputed block to that same log file. The original
content is left untouched.

It is idempotent: a log that already carries the recomputed marker is skipped
unless --force is given.

Usage:
  python3 examples/kernel_agent/fix_old_summarize_eval.py <INPUT_DIR> \
      [--fast 1.0 1.2] [--max-turns N] [--force]
"""

import argparse
import glob
import io
import os
from contextlib import redirect_stdout
from types import SimpleNamespace

from eval import summarize_eval as se
import torch

MARKER = "=== RECOMPUTED: unified denominator (Tk & best share traj total) ==="


def _find_summary_logs(input_dir: str):
    """Return existing summary*.txt logs under input_dir (recursive)."""
    if os.path.isfile(input_dir):
        return [input_dir]
    return sorted(glob.glob(os.path.join(input_dir, "**", "summary*.txt"), recursive=True))


def _load_samples(dumps):
    samples = []
    for d in dumps:
        obj = torch.load(d, weights_only=False)
        samples.extend(obj.get("samples", []) if isinstance(obj, dict) else obj)
    return samples


def _recompute_block(dump_dir: str, fast_thresholds, max_turns) -> str | None:
    """Render the current-version debug header + table for the dump next to a log."""
    dumps = se._find_dumps(dump_dir)
    if not dumps:
        return None
    samples = _load_samples(dumps)
    if not samples:
        return None

    base = se.summarize(samples, fast_thresholds)
    group = None
    if not se._is_single_turn(samples, max_turns=max_turns):
        group = se._summarize_group_best(samples, fast_thresholds, max_turns=max_turns)

    args = SimpleNamespace(max_turns=max_turns)
    buf = io.StringIO()
    with redirect_stdout(buf):
        se._print_debug_header(dumps, samples, base, group, args)
        se._print_oneline_table(group, base, fast_thresholds)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", help="Directory to walk for existing summary*.txt logs.")
    ap.add_argument("--fast", type=float, nargs="+", default=list(se.FAST_DEFAULT))
    ap.add_argument("--max-turns", type=int, default=None, help="Maximum turns for the per-turn table columns.")
    ap.add_argument("--force", action="store_true", help="Re-append even if the recomputed marker is already present.")
    args = ap.parse_args()

    fast_thresholds = tuple(args.fast)
    logs = _find_summary_logs(args.input_dir)
    if not logs:
        raise SystemExit(f"no summary*.txt found under {args.input_dir}")

    updated = skipped = failed = 0
    for log_path in logs:
        dump_dir = os.path.dirname(log_path)
        try:
            with open(log_path) as f:
                existing = f.read()
        except OSError as exc:
            print(f"SKIP (unreadable): {log_path}  ({exc})")
            failed += 1
            continue

        if MARKER in existing and not args.force:
            print(f"SKIP (already recomputed): {log_path}")
            skipped += 1
            continue

        block = _recompute_block(dump_dir, fast_thresholds, args.max_turns)
        if block is None:
            print(f"SKIP (no eval dump next to log): {log_path}")
            failed += 1
            continue

        sep = "" if existing.endswith("\n") else "\n"
        with open(log_path, "a") as f:
            f.write(f"{sep}\n{MARKER}\n{block}")
        print(f"UPDATED: {log_path}")
        updated += 1

    print(f"\ndone: {updated} updated, {skipped} skipped, {failed} failed (of {len(logs)} logs)")


if __name__ == "__main__":
    main()
