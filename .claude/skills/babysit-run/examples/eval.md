# Verifying a KernelBench eval

Handle: eval launcher or pipeline script. Evidence: `summarize_eval.py` plus real `eval_*.pt` dumps; rates alone can hide truncation, wrong grouping, or dropped samples.

## Pattern

1. Use a finalized converted artifact and dedicated resources.
2. Run eval on a small real slice or the intended full set.
3. Summarize metrics, then open `eval_*.pt` dumps with `torch.load(..., weights_only=False)` and read `obj["samples"]`.

## Worked example

**Diff/claim:** `summarize_eval.py` computes best-of-N per trajectory; per-turn records do not inflate the denominator.
**Expected:** single-turn numbers unchanged; multi-turn denominator equals trajectory count; picked turns are real samples.

```bash
python examples/kernel_agent/summarize_eval.py "$EVAL_DIR_T1"
# -> samples: 1600  compile 59.75%  correct 35.00%  fast@1.0 5.62%
#    matches recorded baseline

python examples/kernel_agent/summarize_eval.py "$EVAL_DIR_T3"
# -> trajectories: 1600  turns detected: 3
#    rates use trajectory total, not per-turn record count

python - <<'PY'
import glob, os, torch
root = os.environ["EVAL_DIR_T3"]
paths = sorted(glob.glob(os.path.join(root, "**", "eval_*.pt"), recursive=True))
assert paths, root
obj = torch.load(paths[0], weights_only=False)
samples = obj["samples"]
for s in samples[:2]:
    meta = s.get("metadata", {})
    print(meta.get("group_id"), meta.get("turn_idx"), meta.get("env_extra_info"))
PY
# -> kept turn is real, not truncated/empty/dropped
```

**PASS requires:** unchanged baseline, trajectory-based denominator, and dumps showing selected turns are real. 🔍 Probe: truncated/malformed generation follows the intended score/drop rule without denominator drift.

## Gotchas

- Health can be cached; a tiny canary eval is the real readiness check. Shared serving/compile resources can make a correct eval look stalled.
- Before `ray stop`, `pkill`, or cleanup on a shared node, confirm what else it would kill. Partial dumps after mid-run dependency failure may summarize cleanly; rerun instead of trusting the table.
