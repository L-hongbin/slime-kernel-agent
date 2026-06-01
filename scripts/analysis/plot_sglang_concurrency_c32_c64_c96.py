#!/usr/bin/env python3
"""Plot SGLang max-running-requests C32/C64/C96 decode metrics."""
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RUNS = {
    "rm8 C96 max-req=96": {
        "path": Path("checkpoints/Qwen3.6-27B/20260529_075505_newSlimeKG.tp4.eagle_ctx65536_n8_summ1600/run.log"),
        "cap": 96,
        "wall_s": 5769,
        "color": "#d62728",
    },
    "rm8 C64 max-req=64": {
        "path": Path(
            "checkpoints/Qwen3.6-27B/20260531_051603_newSlimeKG.tp4.eagle.rm16.C64_ctx65536_n8_summ1600/run.log"
        ),
        "cap": 64,
        "wall_s": 6608,
        "color": "#1f77b4",
    },
    "rm8 C32 max-req=32": {
        "path": Path(
            "checkpoints/Qwen3.6-27B/20260531_073240_newSlimeKG.tp4.eagle.rm16.C32_ctx65536_n8_summ1600/run.log"
        ),
        "cap": 32,
        "wall_s": 7646,
        "color": "#2ca02c",
    },
}

TS = re.compile(r"(\d{2}):(\d{2}):(\d{2})\b")
RUNNING = re.compile(r"#running-req:\s*(\d+)")
TPUT = re.compile(r"gen throughput \(token/s\):\s*([\d.]+)")


def parse(path):
    elapsed, conc, tput = [], [], []
    t0 = None
    for line in path.read_text(errors="ignore").splitlines():
        if "Decode batch" not in line:
            continue
        mt = TS.search(line)
        mr = RUNNING.search(line)
        if not (mt and mr):
            continue
        h, m, s = map(int, mt.groups())
        sec = h * 3600 + m * 60 + s
        if t0 is None:
            t0 = sec
        dt = (sec - t0) / 60.0
        if dt < 0:
            dt += 24 * 60
        elapsed.append(dt)
        conc.append(int(mr.group(1)))
        mp = TPUT.search(line)
        tput.append(float(mp.group(1)) if mp else np.nan)
    return np.array(elapsed), np.array(conc), np.array(tput)


def rolling(x, y, win=11):
    if len(y) < win:
        return x, y
    kernel = np.ones(win) / win
    ys = np.convolve(y, kernel, mode="valid")
    xs = x[win // 2 : win // 2 + len(ys)]
    return xs, ys


fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"width_ratios": [1.55, 1]})
ax, bar_ax = axes

stats = []
for label, cfg in RUNS.items():
    t, conc, tput = parse(cfg["path"])
    color = cfg["color"]
    ax.scatter(t, conc, s=9, alpha=0.18, color=color)
    rx, ry = rolling(t, conc)
    ax.plot(rx, ry, color=color, lw=2.3, label=f"{label}  med={np.median(conc):.0f}  mean={conc.mean():.1f}")
    ax.axhline(cfg["cap"], color=color, lw=1, ls=":", alpha=0.55)
    stats.append(
        {
            "label": label,
            "wall_s": cfg["wall_s"],
            "s_it": cfg["wall_s"] / 800.0,
            "conc_mean": conc.mean(),
            "conc_median": np.median(conc),
            "conc_p90": np.percentile(conc, 90),
            "tput_mean": np.nanmean(tput),
            "tput_p50": np.nanpercentile(tput, 50),
            "tput_p90": np.nanpercentile(tput, 90),
            "tput_p99": np.nanpercentile(tput, 99),
            "span_min": t.max(),
        }
    )

ax.set_title("Decode concurrency over eval")
ax.set_xlabel("elapsed time (min)")
ax.set_ylabel("SGLang #running-req")
ax.set_ylim(bottom=0, top=102)
ax.grid(alpha=0.25)
ax.legend(loc="upper right", framealpha=0.95)

metrics = [
    ("wall (s)\nlower better", "wall_s", False),
    ("s/it\nlower better", "s_it", False),
    ("decode mean\nhigher better", "conc_mean", True),
    ("gen tok/s\nhigher better", "tput_mean", True),
]
x = np.arange(len(metrics))
width = 0.24
n_runs = len(stats)
for i, st in enumerate(stats):
    vals = []
    for _, key, higher_better in metrics:
        baseline = stats[0][key]
        value = st[key]
        vals.append(value / baseline if higher_better else baseline / value)
    offset = (i - (n_runs - 1) / 2) * width
    bar_ax.bar(x + offset, vals, width, label=st["label"], color=RUNS[st["label"]]["color"], alpha=0.86)

bar_ax.axhline(1.0, color="#444444", lw=1)
bar_ax.set_xticks(x, [m[0] for m in metrics])
bar_ax.set_ylabel("relative to C96")
bar_ax.set_title("Efficiency summary")
bar_ax.grid(axis="y", alpha=0.25)
bar_ax.set_ylim(0, 1.15)
bar_ax.legend(loc="lower left")

fig.suptitle("SGLang max-running-requests: C96 vs C64 vs C32 (reward worker=8, 800-sample eval)", y=1.02)
fig.tight_layout()

out = Path("handoffs/images/sglang_concurrency_c32_c64_c96.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"saved {out}\n")
print(
    f"{'run':18s} {'wall_s':>7s} {'s/it':>6s} {'mean':>7s} {'median':>7s} "
    f"{'p90':>6s} {'tput':>8s} {'tp50':>8s} {'tp90':>8s} {'tp99':>8s} {'span':>7s}"
)
for st in stats:
    print(
        f"{st['label']:18s} {st['wall_s']:7.0f} {st['s_it']:6.2f} "
        f"{st['conc_mean']:7.1f} {st['conc_median']:7.1f} {st['conc_p90']:6.1f} "
        f"{st['tput_mean']:8.0f} {st['tput_p50']:8.0f} {st['tput_p90']:8.0f} "
        f"{st['tput_p99']:8.0f} {st['span_min']:7.1f}"
    )
