#!/usr/bin/env python3
"""Per-turn compile/correct/fast breakdown from a drkernel eval dump (eval_0.pt),
WITHOUT requiring torch.

Same metric definitions as per_turn_acc.py, but loads the torch.save zip via a
custom unpickler that stubs out tensor/storage rebuilds, so it runs on a CPU box
with no torch installed. Only plain-Python scalar fields are read:
    metadata.turns[t].kernelgym.{compiled, correctness, speedup}

Reports per turn (T1/T2/T3), denominator = #samples (in_all):
  Comp  = compiled rate
  Corr  = correctness rate
  F1.0  = fast@1.0  (speedup >= 1.0)   -> in_all, so F1.0 <= Corr
  F1.2  = fast@1.2  (speedup >= 1.2)

    python3 per_turn_acc_nopt.py <path/to/eval_0.pt> [label]
"""
import io
import pickle
import sys
import zipfile


def _stub(*a, **k):
    return None


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("torch"):
            return _stub  # tensor / storage rebuilds -> None
        return super().find_class(module, name)

    def persistent_load(self, pid):
        return None  # storage backends -> None


def load_nopt(path):
    zf = zipfile.ZipFile(path)
    pkl = next(n for n in zf.namelist() if n.endswith("data.pkl"))
    with zf.open(pkl) as f:
        return _Unpickler(io.BytesIO(f.read())).load()


def kg(s, t):
    turns = (s.get("metadata") or {}).get("turns") or []
    return (turns[t].get("kernelgym") or {}) if t < len(turns) else {}


def main():
    path = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else path
    d = load_nopt(path)
    S = d["samples"]
    N = len(S)
    comp = [0, 0, 0]
    corr = [0, 0, 0]
    f10 = [0, 0, 0]
    f12 = [0, 0, 0]
    reach = [0, 0, 0]
    for s in S:
        turns = (s.get("metadata") or {}).get("turns") or []
        for t in range(3):
            if t < len(turns):
                reach[t] += 1
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
        return f"{name:<7s} " + "  ".join(f"{100.0 * arr[t] / N:5.1f}" for t in range(3))

    print(f"# {label}  N={N}  (in_all 分母=全部样本)")
    print("          T1     T2     T3")
    print(f"{'reach#':<7s} " + "  ".join(f"{reach[t]:5d}" for t in range(3)))
    print(row("Comp", comp))
    print(row("Corr", corr))
    print(row("F1.0", f10))
    print(row("F1.2", f12))


if __name__ == "__main__":
    main()
