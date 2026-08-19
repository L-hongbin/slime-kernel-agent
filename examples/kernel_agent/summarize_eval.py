"""Summarize Compile / Correct / Fast@1.0 / Fast@1.2 from a kernel-agent eval
dump.

The kernel-agent dump (written by --dump-details) stores per-sample results under
sample.metadata.env_result.env_state, NOT the metadata.kernelgym schema that the
sibling slime/scripts/eval_drkernel/summarize_kernelgym_eval.py expects — so that
summarizer reports 100% missing on these dumps. This reads the env_state schema
directly.

Metric denominators are ALL evaluated samples (in_all):
  Compile   = compiled / total
  Correct   = (correctness and not decoy_kernel) / total
  Fast@1.0  = (correct and speedup >= 1.0) / total
  Fast@1.2  = (correct and speedup >= 1.2) / total

For multi-turn dumps, every per-turn rate (T1..TN) and the cumulative Best* column
share ONE denominator: the trajectory total (number of group_ids). Tk counts the
trajectories whose turn k satisfied the metric; best counts trajectories where any
turn satisfied it. Both divide by the same trajectory total, so the columns are
directly comparable.

Best* metrics are read from dump-level metrics when present.
If the dump does not contain Best* metrics, this script warns and falls back to
group_id-based trajectory aggregation. If there is no group_id, Best* metrics are
not computed.

Usage: python3 examples/kernel_agent/summarize_eval.py <EVAL_DIR | dump.pt> [--fast 1.0 1.2] [--max-turns N]
"""

import argparse
import glob
import os
import sys
import warnings

import torch

FAST_DEFAULT = (1.0, 1.2)


def _as_bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v != 0
    if isinstance(v, str):
        return v.strip().lower() in {"true", "1", "yes"}
    return False


def _as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _env_state(sample: dict):
    """Return env_state dict for a dumped sample, or None if absent."""
    meta = sample.get("metadata") or {}
    env_result = meta.get("env_result")
    if not isinstance(env_result, dict):
        return None
    env_state = env_result.get("env_state")
    return env_state if isinstance(env_state, dict) else None


def _metadata(sample: dict) -> dict:
    meta = sample.get("metadata") or {}
    return meta if isinstance(meta, dict) else {}


def _partition_by_metadata(samples: list[dict], key: str) -> dict[str, list[dict]]:
    """Partition samples by one metadata value using stable string labels."""

    groups: dict[str, list[dict]] = {}
    for sample in samples:
        value = _metadata(sample).get(key)
        label = "<missing>" if value is None else str(value)
        groups.setdefault(label, []).append(sample)
    return groups


def _env_extra_info(sample: dict) -> dict | None:
    meta = _metadata(sample)
    env_extra_info = meta.get("env_extra_info")
    return env_extra_info if isinstance(env_extra_info, dict) else None


def _turn_idx(sample: dict) -> int | None:
    turn_idx = _metadata(sample).get("turn_idx")
    try:
        return int(turn_idx)
    except (TypeError, ValueError):
        return None


def _sample_metrics(sample: dict) -> dict | None:
    """Return normalized compile/correct/speedup metrics for one sample."""
    info = _env_extra_info(sample)
    if info is not None:
        compiled = _as_bool(info.get("compilation", info.get("compiled")))
        correct = _as_bool(info.get("correctness")) and not _as_bool(info.get("decoy_kernel"))
        speedup = _as_float(info.get("speedup"))
        return {"compiled": compiled, "correct": correct, "speedup": speedup}

    es = _env_state(sample)
    if es is None:
        return None
    compiled = _as_bool(es.get("compiled", es.get("compilation")))
    correct = _as_bool(es.get("correctness")) and not _as_bool(es.get("decoy_kernel"))
    speedup = _as_float(es.get("speedup"))
    return {"compiled": compiled, "correct": correct, "speedup": speedup}


def _find_dumps(path: str):
    if os.path.isfile(path):
        return [path]
    for sub in ("dumps/rollout_data", "rollout_data", "dumps", "."):
        hits = sorted(glob.glob(os.path.join(path, sub, "eval_*.pt")))
        if hits:
            return hits
    return sorted(glob.glob(os.path.join(path, "**", "eval_*.pt"), recursive=True))


def _flatten_best_metrics(obj, out: dict[str, float], prefix: str = ""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == "samples":
                continue
            next_prefix = f"{prefix}/{key}" if prefix else str(key)
            _flatten_best_metrics(value, out, next_prefix)
        return
    if isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            _flatten_best_metrics(value, out, f"{prefix}/{idx}" if prefix else str(idx))
        return
    if not prefix or "best" not in prefix.lower():
        return
    if isinstance(obj, bool):
        out[prefix] = float(obj)
    elif isinstance(obj, (int, float)):
        out[prefix] = float(obj)


def _extract_dump_best_metrics(obj) -> dict[str, float]:
    if not isinstance(obj, dict):
        return {}
    best_metrics: dict[str, float] = {}
    for key, value in obj.items():
        if key == "samples":
            continue
        if "best" in str(key).lower() or str(key) in {"metrics", "eval_metrics", "summary", "log_dict"}:
            _flatten_best_metrics(value, best_metrics, str(key) if "best" in str(key).lower() else "")
    return best_metrics


def _is_single_turn(samples, max_turns=None) -> bool:
    if max_turns is not None:
        return int(max_turns) <= 1
    turn_indices = [_turn_idx(sample) for sample in samples]
    turn_indices = [idx for idx in turn_indices if idx is not None]
    if not turn_indices:
        return True
    return max(turn_indices) <= 0


def _empty_counts(fast_thresholds):
    return {"compiled": 0, "correct": 0, "fast": {t: 0 for t in fast_thresholds}, "speedup": []}


def _summarize_group_best(samples, fast_thresholds, max_turns=None):
    trajectories = {}
    for sample in samples:
        group_id = sample.get("group_id")
        if group_id is None:
            continue
        trajectories.setdefault(group_id, []).append(sample)

    if not trajectories:
        return None

    if max_turns is None:
        turn_indices = [_turn_idx(sample) for sample in samples]
        turn_indices = [idx for idx in turn_indices if idx is not None]
        max_turns = max(turn_indices) + 1 if turn_indices else 1
    max_turns = max(1, int(max_turns))

    overall = _empty_counts(fast_thresholds)
    by_turn = {turn_count: _empty_counts(fast_thresholds) for turn_count in range(1, max_turns + 1)}
    by_turn_present = {turn_count: 0 for turn_count in range(1, max_turns + 1)}
    best_by_turn = {turn_count: _empty_counts(fast_thresholds) for turn_count in range(1, max_turns + 1)}
    missing = 0

    for trajectory_samples in trajectories.values():
        turn_metrics = []
        for sample in trajectory_samples:
            if _metadata(sample).get("is_pad_turn"):
                continue
            metrics = _sample_metrics(sample)
            if metrics is None:
                continue
            turn_idx = _turn_idx(sample)
            if turn_idx is None:
                turn_idx = len(turn_metrics)
            turn_metrics.append((turn_idx, metrics))

        if not turn_metrics:
            missing += 1

        turn_metrics.sort(key=lambda item: item[0])
        # Per-turn (single turn) metrics: turn_count k looks only at turn index k-1.
        # The denominator is the full trajectory total (same as Best*), NOT the count
        # of trajectories that produced that turn; `present` is tracked only for info.
        for turn_count, counts in by_turn.items():
            this_turn = [metrics for turn_idx, metrics in turn_metrics if turn_idx == turn_count - 1]
            if this_turn:
                by_turn_present[turn_count] += 1
                _add_best_counts(counts, this_turn, fast_thresholds)
            cumulative = [metrics for turn_idx, metrics in turn_metrics if turn_idx < turn_count]
            _add_best_counts(best_by_turn[turn_count], cumulative, fast_thresholds)
        # Overall "best" stays cumulative over the whole trajectory (all turns).
        _add_best_counts(overall, [metrics for _, metrics in turn_metrics], fast_thresholds)

    total = len(trajectories)

    def rate(n):
        return n / total if total else 0.0

    out = {
        "trajectory_total": total,
        "trajectory_missing_env_result": missing,
        "best_compile_count": overall["compiled"],
        "BestCompile": rate(overall["compiled"]),
        "best_correct_count": overall["correct"],
        "BestCorrect": rate(overall["correct"]),
    }
    if overall["speedup"]:
        out["BestSpeedupMean"] = sum(overall["speedup"]) / len(overall["speedup"])
    for t in fast_thresholds:
        out[f"BestFast@{t:g}"] = rate(overall["fast"][t])
        out[f"best_fast@{t:g}_count"] = overall["fast"][t]

    out["per_turn"] = {}
    for turn_count, counts in by_turn.items():
        present = by_turn_present[turn_count]

        # Unified denominator: every per-turn rate uses the full trajectory total,
        # so Tk and best are directly comparable on the same denominator.
        def prate(n, d=total):
            return n / d if d else 0.0

        turn_out = {
            "present": present,
            "compile_count": counts["compiled"],
            "Compile": prate(counts["compiled"]),
            "correct_count": counts["correct"],
            "Correct": prate(counts["correct"]),
        }
        if counts["speedup"]:
            turn_out["SpeedupMean"] = sum(counts["speedup"]) / len(counts["speedup"])
        for t in fast_thresholds:
            turn_out[f"Fast@{t:g}"] = prate(counts["fast"][t])
            turn_out[f"fast@{t:g}_count"] = counts["fast"][t]
        out["per_turn"][turn_count] = turn_out

    # Cumulative best after at most k turns. Keep this alongside the single-turn
    # table: the former answers "did any attempt up through k work?", while the
    # latter makes regressions or gains in the feedback turns visible.
    out["best_by_turn"] = {}
    for turn_count, counts in best_by_turn.items():
        turn_out = {
            "compile_count": counts["compiled"],
            "Compile": rate(counts["compiled"]),
            "correct_count": counts["correct"],
            "Correct": rate(counts["correct"]),
        }
        if counts["speedup"]:
            turn_out["SpeedupMean"] = sum(counts["speedup"]) / len(counts["speedup"])
        for t in fast_thresholds:
            turn_out[f"Fast@{t:g}"] = rate(counts["fast"][t])
            turn_out[f"fast@{t:g}_count"] = counts["fast"][t]
        out["best_by_turn"][turn_count] = turn_out
    return out


def _add_best_counts(counts, metrics_list, fast_thresholds):
    if not metrics_list:
        return
    if any(m["compiled"] for m in metrics_list):
        counts["compiled"] += 1
    if any(m["correct"] for m in metrics_list):
        counts["correct"] += 1
    valid_speedups = [m["speedup"] for m in metrics_list if m["speedup"] is not None]
    if valid_speedups:
        counts["speedup"].append(max(valid_speedups))
    for t in fast_thresholds:
        if any(m["correct"] and m["speedup"] is not None and m["speedup"] >= t for m in metrics_list):
            counts["fast"][t] += 1


def summarize(samples, fast_thresholds):
    total = len(samples)
    compiled = correct = missing = 0
    fast = {t: 0 for t in fast_thresholds}
    for s in samples:
        es = _env_state(s)
        if es is None:
            missing += 1
            continue
        if _as_bool(es.get("compiled")):
            compiled += 1
        is_correct = _as_bool(es.get("correctness")) and not _as_bool(es.get("decoy_kernel"))
        if is_correct:
            correct += 1
            sp = _as_float(es.get("speedup"))
            if sp is not None:
                for t in fast_thresholds:
                    if sp >= t:
                        fast[t] += 1

    def rate(n):
        return n / total if total else 0.0

    out = {
        "total": total,
        "missing_env_result": missing,
        "compile_count": compiled,
        "Compile": rate(compiled),
        "correct_count": correct,
        "Correct": rate(correct),
    }
    for t in fast_thresholds:
        out[f"Fast@{t:g}"] = rate(fast[t])
        out[f"fast@{t:g}_count"] = fast[t]
    return out


def summarize_with_best(samples, fast_thresholds, max_turns=None, dump_best_metrics=None):
    out = summarize(samples, fast_thresholds)
    if _is_single_turn(samples, max_turns=max_turns):
        out["best_source"] = "skipped_single_turn"
        return out

    dump_best_metrics = dump_best_metrics or {}
    if dump_best_metrics:
        out["best_source"] = "dump"
        out["dump_best_metrics"] = dump_best_metrics
        return out

    warnings.warn(
        "No Best* metrics found in eval dump; falling back to group_id-based trajectory best metrics.",
        RuntimeWarning,
        stacklevel=2,
    )
    group_best = _summarize_group_best(samples, fast_thresholds, max_turns=max_turns)
    if group_best is None:
        warnings.warn(
            "No group_id found in eval dump; Best* metrics will not be computed.",
            RuntimeWarning,
            stacklevel=2,
        )
        out["best_source"] = "skipped_no_group"
        return out

    out["best_source"] = "computed_group"
    out.update(group_best)
    return out


def _detected_turns(samples) -> int:
    turn_idxs = [t for t in (_turn_idx(s) for s in samples) if t is not None]
    return max(turn_idxs) + 1 if turn_idxs else 1


def _print_debug_header(dumps, samples, base, group, args):
    print(f"dumps: {len(dumps)} -> {dumps}")
    print(f"samples: {base['total']}  (missing env_result: {base['missing_env_result']})")
    configured = "auto" if args.max_turns is None else str(args.max_turns)
    print(f"turns: configured max-turns={configured}, detected={_detected_turns(samples)}")
    if group is not None:
        print(
            f"trajectories (group_id): {group['trajectory_total']}  "
            f"(missing env_result: {group['trajectory_missing_env_result']})"
        )
        per_turn = group.get("per_turn", {})
        if per_turn:
            counts = "  ".join(f"T{k}={per_turn[k]['present']}" for k in sorted(per_turn))
            print(f"per-turn records (Tk produced; all rates use traj total as denom): {counts}")
    else:
        print("trajectories (group_id): n/a (single-turn or no group_id; table shows overall rates only)")
    print()


def _print_oneline_table(group, base, fast_thresholds):
    """Single-row table: percentages laid out as compile, correct, then each fast
    threshold; within every metric block the columns are the per-turn accuracy at
    each individual turn (T1..TN = turn 1..N alone) followed by `best` (cumulative
    best over the whole trajectory / all turns)."""
    col_w = 6

    def fast_label(t):
        s = f"{t:g}"
        return f"fast@{s}" if "." in s else f"fast@{s}.0"

    if group is not None:
        turns = sorted(group["per_turn"].keys())
        labels = [f"T{k}" for k in turns] + ["best"]
        n = group["trajectory_total"]
        n_label = "traj"

        def turn_vals(metric_key, best_key):
            return [group["per_turn"][k][metric_key] for k in turns] + [group[best_key]]

        metric_groups = [
            ("compile", turn_vals("Compile", "BestCompile")),
            ("correct", turn_vals("Correct", "BestCorrect")),
        ]
        for t in fast_thresholds:
            metric_groups.append((fast_label(t), turn_vals(f"Fast@{t:g}", f"BestFast@{t:g}")))
    else:
        labels = ["all"]
        n = base["total"]
        n_label = "all"
        metric_groups = [
            ("compile", [base["Compile"]]),
            ("correct", [base["Correct"]]),
        ]
        for t in fast_thresholds:
            metric_groups.append((fast_label(t), [base[f"Fast@{t:g}"]]))

    def cell(s):
        return f"{s:>{col_w}}"

    blocks = [("n", cell(n_label), cell(str(n)))]
    for title, vals in metric_groups:
        sub = " ".join(cell(lab) for lab in labels)
        val = " ".join(cell(f"{v * 100:.2f}") for v in vals)
        blocks.append((title, sub, val))

    sep = " | "
    title_line, sub_line, val_line = [], [], []
    for title, sub, val in blocks:
        width = max(len(title), len(sub), len(val))
        title_line.append(f"{title:^{width}}")
        sub_line.append(f"{sub:>{width}}")
        val_line.append(f"{val:>{width}}")
    print(sep.join(title_line))
    print(sep.join(sub_line))
    print(sep.join(val_line))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="EVAL_DIR (…/dumps/rollout_data) or an eval_*.pt file")
    ap.add_argument("--fast", type=float, nargs="+", default=list(FAST_DEFAULT))
    ap.add_argument("--max-turns", type=int, default=None, help="Maximum turns for the per-turn table columns.")
    ap.add_argument(
        "--group-by-metadata",
        default=None,
        help="Print one independent summary per value of this sample.metadata key.",
    )
    args = ap.parse_args()

    dumps = _find_dumps(args.path)
    if not dumps:
        raise SystemExit(f"no eval_*.pt found under {args.path}")

    samples = []
    for d in dumps:
        obj = torch.load(d, weights_only=False)
        samples.extend(obj.get("samples", []) if isinstance(obj, dict) else obj)

    fast_thresholds = tuple(args.fast)
    partitions = _partition_by_metadata(samples, args.group_by_metadata) if args.group_by_metadata else {"": samples}
    for idx, (label, partition) in enumerate(sorted(partitions.items())):
        if idx:
            print()
        if args.group_by_metadata:
            print(f"=== {args.group_by_metadata}={label} ===")
        base = summarize(partition, fast_thresholds)
        group = None
        if not _is_single_turn(partition, max_turns=args.max_turns):
            group = _summarize_group_best(partition, fast_thresholds, max_turns=args.max_turns)
            if group is None:
                print(
                    "note: multi-turn dump but no group_id found; per-turn/best columns unavailable.",
                    file=sys.stderr,
                )

        _print_debug_header(dumps, partition, base, group, args)
        _print_oneline_table(group, base, fast_thresholds)


if __name__ == "__main__":
    main()
