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

For multi-turn dumps, Best* metrics are read from dump-level metrics when present.
If the dump does not contain Best* metrics, this script warns and falls back to
group_id-based trajectory aggregation. If there is no group_id, Best* metrics are
not computed.

Usage: python3 examples/kernel_agent/summarize_eval.py <EVAL_DIR | dump.pt> [--fast 1.0 1.2] [--max-turns N]
"""

import argparse
import glob
import os
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
        for turn_count, counts in by_turn.items():
            visible = [metrics for turn_idx, metrics in turn_metrics if turn_idx < turn_count]
            _add_best_counts(counts, visible, fast_thresholds)
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

    out["best_by_turn"] = {}
    for turn_count, counts in by_turn.items():
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="EVAL_DIR (…/dumps/rollout_data) or an eval_*.pt file")
    ap.add_argument("--fast", type=float, nargs="+", default=list(FAST_DEFAULT))
    ap.add_argument("--max-turns", type=int, default=None, help="Maximum turns for BestByTurn output.")
    args = ap.parse_args()

    dumps = _find_dumps(args.path)
    if not dumps:
        raise SystemExit(f"no eval_*.pt found under {args.path}")

    samples = []
    dump_best_metrics = {}
    for d in dumps:
        obj = torch.load(d, weights_only=False)
        dump_best_metrics.update(_extract_dump_best_metrics(obj))
        samples.extend(obj.get("samples", []) if isinstance(obj, dict) else obj)

    res = summarize_with_best(
        samples,
        tuple(args.fast),
        max_turns=args.max_turns,
        dump_best_metrics=dump_best_metrics,
    )
    print(f"dumps: {dumps}")
    print(f"total samples: {res['total']}  (missing env_result: {res['missing_env_result']})")
    print(f"Compile   : {res['Compile']:.4f}  ({res['compile_count']}/{res['total']})")
    print(f"Correct   : {res['Correct']:.4f}  ({res['correct_count']}/{res['total']})")
    for t in args.fast:
        print(f"Fast@{t:g}  : {res[f'Fast@{t:g}']:.4f}  ({res[f'fast@{t:g}_count']}/{res['total']})")

    if res.get("best_source") == "skipped_single_turn":
        print("Best metrics: skipped (single-turn eval)")
        return
    if res.get("best_source") == "skipped_no_group":
        print("Best metrics: skipped (no group_id in dump)")
        return
    if res.get("best_source") == "dump":
        print("Best metrics: read from dump")
        for key, value in sorted(res["dump_best_metrics"].items()):
            print(f"  {key}: {value:.6g}")
        return

    print(
        f"trajectory samples: {res['trajectory_total']}  "
        f"(missing env_result: {res['trajectory_missing_env_result']})"
    )
    print(f"BestCompile: {res['BestCompile']:.4f}  ({res['best_compile_count']}/{res['trajectory_total']})")
    print(f"BestCorrect: {res['BestCorrect']:.4f}  ({res['best_correct_count']}/{res['trajectory_total']})")
    for t in args.fast:
        print(
            f"BestFast@{t:g}: {res[f'BestFast@{t:g}']:.4f}  "
            f"({res[f'best_fast@{t:g}_count']}/{res['trajectory_total']})"
        )
    if "BestSpeedupMean" in res:
        print(f"BestSpeedupMean: {res['BestSpeedupMean']:.4f}")
    if len(res["best_by_turn"]) > 1:
        print("BestByTurn:")
        for turn_count, turn_res in sorted(res["best_by_turn"].items()):
            print(
                f"  turn<={turn_count}: "
                f"Compile={turn_res['Compile']:.4f} ({turn_res['compile_count']}/{res['trajectory_total']})  "
                f"Correct={turn_res['Correct']:.4f} ({turn_res['correct_count']}/{res['trajectory_total']})"
            )
            for t in args.fast:
                print(
                    f"    Fast@{t:g}={turn_res[f'Fast@{t:g}']:.4f} "
                    f"({turn_res[f'fast@{t:g}_count']}/{res['trajectory_total']})"
                )
            if "SpeedupMean" in turn_res:
                print(f"    SpeedupMean={turn_res['SpeedupMean']:.4f}")


if __name__ == "__main__":
    main()
