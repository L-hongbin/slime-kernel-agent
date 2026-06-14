"""Summarize Compile / Correct / Fast@1.0 / Fast@1.2 (in_all) from a kernel-agent
eval dump.

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

Usage: python3 examples/kernel_agent/summarize_eval.py <EVAL_DIR | dump.pt> [--fast 1.0 1.2]
"""

import argparse
import glob
import os

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


def _find_dumps(path: str):
    if os.path.isfile(path):
        return [path]
    for sub in ("dumps/rollout_data", "rollout_data", "dumps", "."):
        hits = sorted(glob.glob(os.path.join(path, sub, "eval_*.pt")))
        if hits:
            return hits
    return sorted(glob.glob(os.path.join(path, "**", "eval_*.pt"), recursive=True))


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="EVAL_DIR (…/dumps/rollout_data) or an eval_*.pt file")
    ap.add_argument("--fast", type=float, nargs="+", default=list(FAST_DEFAULT))
    args = ap.parse_args()

    dumps = _find_dumps(args.path)
    if not dumps:
        raise SystemExit(f"no eval_*.pt found under {args.path}")

    samples = []
    for d in dumps:
        obj = torch.load(d, weights_only=False)
        samples.extend(obj.get("samples", []) if isinstance(obj, dict) else obj)

    res = summarize(samples, tuple(args.fast))
    print(f"dumps: {dumps}")
    print(f"total samples: {res['total']}  (missing env_result: {res['missing_env_result']})")
    print(f"Compile   : {res['Compile']:.4f}  ({res['compile_count']}/{res['total']})")
    print(f"Correct   : {res['Correct']:.4f}  ({res['correct_count']}/{res['total']})")
    for t in args.fast:
        print(f"Fast@{t:g}  : {res[f'Fast@{t:g}']:.4f}  ({res[f'fast@{t:g}_count']}/{res['total']})")


if __name__ == "__main__":
    main()
