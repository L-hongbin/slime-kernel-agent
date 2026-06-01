#!/usr/bin/env python3
"""Plot decode concurrency (#running-req) over time for rm8 vs rm16 eval runs.

Parses SGLang "Decode batch" log lines and shows how full the decode pipeline
stayed during the 800-sample KernelBench eval. Higher sustained #running-req =
rollout less blocked on reward eval = higher decode throughput.
"""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNS = {
    "rm8 (8 workers)": "checkpoints/Qwen3.6-27B/20260529_075505_newSlimeKG.tp4.eagle_ctx65536_n8_summ1600/run.log",
    "rm16 (16 workers)": "checkpoints/Qwen3.6-27B/20260529_132454_newSlimeKG.tp4.eagle.rm16_ctx65536_n8_summ1600/run.log",
}
COLORS = {"rm8 (8 workers)": "#1f77b4", "rm16 (16 workers)": "#d62728"}

TS = re.compile(r"(\d{2}):(\d{2}):(\d{2})\b")
RUN = re.compile(r"#running-req:\s*(\d+)")
TPUT = re.compile(r"gen throughput \(token/s\):\s*([\d.]+)")


def parse(path):
    t, conc, tput = [], [], []
    t0 = None
    for line in Path(path).read_text(errors="ignore").splitlines():
        if "Decode batch" not in line:
            continue
        mt, mr = TS.search(line), RUN.search(line)
        if not (mt and mr):
            continue
        h, m, s = map(int, mt.groups())
        sec = h * 3600 + m * 60 + s
        if t0 is None:
            t0 = sec
        elapsed = (sec - t0) / 60.0
        if elapsed < 0:  # midnight wrap guard
            elapsed += 24 * 60
        t.append(elapsed)
        conc.append(int(mr.group(1)))
        mp = TPUT.search(line)
        tput.append(float(mp.group(1)) if mp else np.nan)
    return np.array(t), np.array(conc), np.array(tput)


def rolling(x, y, win=9):
    if len(y) < win:
        return x, y
    k = np.ones(win) / win
    ys = np.convolve(y, k, mode="valid")
    xs = x[win // 2 : win // 2 + len(ys)]
    return xs, ys


fig, ax = plt.subplots(1, 1, figsize=(11, 5))
stats = []
for label, path in RUNS.items():
    if not Path(path).exists():
        print(f"skip missing {path}")
        continue
    t, conc, tput = parse(path)
    c = COLORS[label]
    ax.scatter(t, conc, s=10, alpha=0.25, color=c)
    rx, ry = rolling(t, conc)
    ax.plot(rx, ry, color=c, lw=2.2, label=f"{label}  mean={conc.mean():.1f}  median={np.median(conc):.0f}")
    ax.axhline(conc.mean(), color=c, ls="--", lw=1, alpha=0.6)
    stats.append((label, len(conc), conc.mean(), np.median(conc), np.nanmean(tput), t.max()))

ax.set_xlabel("elapsed time (min)")
ax.set_ylabel("decode concurrency  (#running-req)")
ax.set_title(
    "Decode pipeline fullness during 800-sample eval — rm8 vs rm16\n" "(dashed = run mean; thick = 9-pt rolling mean)"
)
ax.legend(loc="lower center")
ax.grid(alpha=0.3)
ax.set_ylim(bottom=0)

fig.tight_layout()
out = Path("handoffs/images/decode_concurrency_rm8_vs_rm16.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=130)
print(f"saved {out}\n")
print(f"{'run':22s} {'n':>5s} {'mean':>7s} {'median':>7s} {'tput':>9s} {'span_min':>9s}")
for label, n, mean, med, tp, span in stats:
    print(f"{label:22s} {n:5d} {mean:7.1f} {med:7.1f} {tp:9.0f} {span:9.1f}")
