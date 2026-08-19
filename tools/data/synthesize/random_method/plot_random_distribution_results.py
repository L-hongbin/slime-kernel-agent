#!/usr/bin/env python3
"""Plot the authoritative random-distribution validation result."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FAMILY_ORDER = (
    "multinomial_categories",
    "poisson_counts",
    "signed_uniform",
    "uniform_01",
)
FAMILY_LABELS = ("Multinomial", "Poisson", "Signed uniform", "Uniform [0, 1)")


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _accepted_family_counts(path: Path) -> collections.Counter[str]:
    counts: collections.Counter[str] = collections.Counter()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            family = row.get("assigned_target")
            if family not in FAMILY_ORDER:
                raise ValueError(f"unexpected family at {path}:{line_number}: {family!r}")
            counts[family] += 1
    return counts


def plot(run_dir: Path, output: Path) -> None:
    reference = _read_json(run_dir / "analysis/reference_summary.json")
    liveness = _read_json(run_dir / "analysis/liveness_summary.json")
    semantic = _read_json(run_dir / "analysis/semantic_gate_summary.json")
    accepted = _accepted_family_counts(run_dir / "runtime/accepted.manifest.jsonl")

    candidate = int(reference["candidate_pairs"])
    both_pass = int(reference["both_pass_children"])
    live_pass = int(liveness["passed"])
    promoted = int(semantic["promoted_rows"])
    stages = np.asarray([candidate, both_pass, live_pass, promoted], dtype=float)

    family_reference = reference["family_classification_counts"]
    family_liveness = liveness["passed_family_counts"]
    family_candidates = np.asarray(
        [sum(int(count) for count in family_reference[name].values()) for name in FAMILY_ORDER],
        dtype=float,
    )
    reference_rates = (
        np.asarray([int(family_reference[name].get("both_pass", 0)) for name in FAMILY_ORDER], dtype=float)
        / family_candidates
    )
    liveness_rates = np.asarray([int(family_liveness[name]) for name in FAMILY_ORDER], dtype=float) / family_candidates
    promoted_rates = np.asarray([accepted[name] for name in FAMILY_ORDER], dtype=float) / family_candidates

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titleweight": "semibold",
            "axes.edgecolor": "#CBD5E1",
            "axes.labelcolor": "#334155",
            "xtick.color": "#475569",
            "ytick.color": "#475569",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, (left, right) = plt.subplots(1, 2, figsize=(14.2, 6.2), gridspec_kw={"width_ratios": (0.88, 1.45)})
    fig.suptitle("Random-distribution augmentation", fontsize=17, fontweight="bold", x=0.045, ha="left")

    stage_labels = ("Candidate", "Paired reference", "Liveness", "Semantic gate")
    stage_colors = ("#94A3B8", "#4C78A8", "#2A9D8F", "#F2A541")
    y = np.arange(len(stages))
    left.barh(y, stages / candidate * 100.0, height=0.58, color=stage_colors, edgecolor="none")
    left.set_yticks(y, stage_labels)
    left.invert_yaxis()
    left.set_xlim(0, 112)
    left.set_xlabel("Cumulative retention (%)")
    left.set_title("Overall validation funnel", loc="left", pad=13)
    left.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    left.set_axisbelow(True)
    for index, count in enumerate(stages.astype(int)):
        rate = count / candidate * 100.0
        left.text(rate + 1.2, index, f"{count:,}\n{rate:.1f}%", va="center", fontsize=9, color="#334155")

    x = np.arange(len(FAMILY_ORDER))
    width = 0.23
    series = (
        (reference_rates, "Paired reference", "#4C78A8"),
        (liveness_rates, "Liveness", "#2A9D8F"),
        (promoted_rates, "Semantic gate", "#F2A541"),
    )
    for offset, (rates, label, color) in zip((-width, 0.0, width), series, strict=True):
        bars = right.bar(x + offset, rates * 100.0, width=width, label=label, color=color, edgecolor="none")
        if label == "Semantic gate":
            for bar, rate, count in zip(bars, rates, (accepted[name] for name in FAMILY_ORDER), strict=True):
                high_bar = rate > 0.1
                right.annotate(
                    f"{rate * 100:.1f}%\n({count:,})",
                    (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                    xytext=(0, -5 if high_bar else 5),
                    textcoords="offset points",
                    ha="center",
                    va="top" if high_bar else "bottom",
                    fontsize=8.5,
                    fontweight="semibold",
                    color="#5C3A00" if high_bar else "#A52A2A",
                )

    right.set_title("Retention by assigned distribution", loc="left", pad=13)
    right.set_ylabel("Share of family candidates (%)")
    right.set_ylim(0, 112)
    right.set_xticks(x, FAMILY_LABELS)
    right.grid(axis="y", color="#E2E8F0", linewidth=0.8)
    right.set_axisbelow(True)
    right.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    right.text(
        0.01,
        -0.17,
        "Runtime gates retain nearly every family; the semantic gate removes almost all category-valued children.",
        transform=right.transAxes,
        color="#64748B",
        fontsize=9,
    )

    for axis in (left, right):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot(args.run_dir, args.output)


if __name__ == "__main__":
    main()
