#!/usr/bin/env python3
"""Per-turn accuracy breakdown for a drkernel eval dump (eval_0.pt).

Reports, per turn (T1/T2/T3), over ALL samples (in_all, denominator = #samples):
  Comp  = compiled rate
  Corr  = correctness rate
  F1.0  = fast@1.0  (speedup >= 1.0)
  F1.2  = fast@1.2  (speedup >= 1.2)

fast@x is in_all: an incorrect/missing turn contributes speedup 0 (counts as a miss),
so F1.0 <= Corr.

Reads `metadata.turns[t].kernelgym.{compiled,correctness,speedup}` from the dump
written by `--dump-details` (dumps/rollout_data/eval_0.pt).

    python3 per_turn_acc.py <path/to/eval_0.pt>

See handoffs/in_progress/HANDOFF_CACHE_DRKERNEL_HYBRID.md.
"""
import sys

import torch

d = torch.load(sys.argv[1], weights_only=False, map_location="cpu")
S = d["samples"]
N = len(S)


def kg(s, t):
    turns = (s.get("metadata") or {}).get("turns") or []
    return (turns[t].get("kernelgym") or {}) if t < len(turns) else {}


comp = [0, 0, 0]
corr = [0, 0, 0]
f10 = [0, 0, 0]
f12 = [0, 0, 0]
for s in S:
    for t in range(3):
        k = kg(s, t)
        if k.get("compiled"):
            comp[t] += 1
        if k.get("correctness"):
            corr[t] += 1
        sp = k.get("speedup") or 0.0
        if sp >= 1.0:
            f10[t] += 1
        if sp >= 1.2:
            f12[t] += 1


def row(name, arr):
    v0, v1, v2 = (100.0 * arr[t] / N for t in range(3))
    return f"{name:<5s} {v0:5.1f}  {v1:5.1f}  {v2:5.1f}"


print(f"N={N}  (in_all 分母=全部样本)")
print("         T1     T2     T3")
print(row("Comp", comp))
print(row("Corr", corr))
print(row("F1.0", f10))
print(row("F1.2", f12))
