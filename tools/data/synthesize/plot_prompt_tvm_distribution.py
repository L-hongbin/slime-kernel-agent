#!/usr/bin/env python3
"""Plot prompt_tvm_v4 and KernelBench distributions from a profiler JSON."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE = REPO_ROOT / "Data/prompt_tvm_v4/analysis/distribution_profile.json"
DEFAULT_OUTPUT = REPO_ROOT / "handoffs/data/synthesize/figures/distribution_overview.png"
PROFILE_SCHEMA_VERSION = "prompt-tvm-distribution-profile-v2"

FACTORIES = ("randn", "rand", "randint")
PERCENTILES = ("p50", "p75", "p90")
FAMILIES = (
    ("conv", "Conv"),
    ("normalization", "Norm"),
    ("matmul_linear", "Matmul / linear"),
    ("reduction", "Reduction"),
    ("shape_layout", "Shape / layout"),
    ("pooling", "Pooling"),
    ("attention_recurrent", "Attention / RNN"),
    ("activation", "Activation"),
    ("indexing_scatter", "Index / scatter"),
    ("loss_distance", "Loss / distance"),
)

TRAIN_COLOR = "#4C78A8"
KERNELBENCH_COLOR = "#F58518"
GRID_COLOR = "#D9DEE7"
TEXT_COLOR = "#253047"


def load_profile(path: Path) -> dict[str, Any]:
    profile = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if profile.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError(f"expected {PROFILE_SCHEMA_VERSION}, got {profile.get('schema_version')!r}")
    datasets = profile.get("datasets")
    if not isinstance(datasets, Mapping) or not {"active_train", "kernelbench"} <= datasets.keys():
        raise ValueError("profile must contain active_train and kernelbench datasets")
    return profile


def _share(count: int, rows: int) -> float:
    return 100.0 * count / rows


def _dataset(profile: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return profile["datasets"][name]


def _factory_shares(dataset: Mapping[str, Any]) -> list[float]:
    rows = dataset["rows"]
    factories = dataset["input_factory_row_presence"]["factories"]
    return [_share(factories[name]["count"], rows) for name in FACTORIES]


def _operator_bin_shares(dataset: Mapping[str, Any]) -> list[float]:
    rows = dataset["rows"]
    return [_share(item["count"], rows) for item in dataset["operator_count"]["exclusive_plot_bins"]]


def _family_shares(dataset: Mapping[str, Any]) -> list[float]:
    rows = dataset["rows"]
    families = dataset["operator_family_row_presence"]["families"]
    return [_share(families[name]["count"], rows) for name, _label in FAMILIES]


def _grouped_bars(
    ax: plt.Axes,
    labels: Sequence[str],
    train_values: Sequence[float],
    kernelbench_values: Sequence[float],
    train_label: str,
    kernelbench_label: str,
) -> None:
    positions = list(range(len(labels)))
    width = 0.36
    train = ax.bar(
        [position - width / 2 for position in positions],
        train_values,
        width,
        label=train_label,
        color=TRAIN_COLOR,
    )
    kernelbench = ax.bar(
        [position + width / 2 for position in positions],
        kernelbench_values,
        width,
        label=kernelbench_label,
        color=KERNELBENCH_COLOR,
    )
    ax.set_xticks(positions, labels)
    ax.bar_label(train, fmt="%.1f", padding=3, fontsize=9)
    ax.bar_label(kernelbench, fmt="%.1f", padding=3, fontsize=9)


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)


def render(profile: Mapping[str, Any], output: Path) -> None:
    train = _dataset(profile, "active_train")
    kernelbench = _dataset(profile, "kernelbench")
    train_label = train["label"]
    kernelbench_label = kernelbench["label"]
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)

    ax = axes[0, 0]
    _grouped_bars(
        ax,
        FACTORIES,
        _factory_shares(train),
        _factory_shares(kernelbench),
        train_label,
        kernelbench_label,
    )
    ax.set_title("A. Input-factory prevalence")
    ax.set_ylabel("Rows containing factory (%)")
    ax.legend()
    _style_axis(ax)

    ax = axes[0, 1]
    positions = [50, 75, 90]
    train_percentiles = train["resolved_tensor_numel"]["percentiles"]
    kernelbench_percentiles = kernelbench["resolved_tensor_numel"]["percentiles"]
    ax.plot(
        positions,
        [train_percentiles[name] for name in PERCENTILES],
        marker="o",
        linewidth=2.8,
        label=train_label,
        color=TRAIN_COLOR,
    )
    ax.plot(
        positions,
        [kernelbench_percentiles[name] for name in PERCENTILES],
        marker="o",
        linewidth=2.8,
        label=kernelbench_label,
        color=KERNELBENCH_COLOR,
    )
    ax.set_yscale("log")
    ax.set_xticks(positions, PERCENTILES)
    ax.set_title("B. Resolved tensor size")
    ax.set_ylabel("Elements (log scale)")
    ax.legend()
    _style_axis(ax)

    ax = axes[1, 0]
    train_bins = _operator_bin_shares(train)
    kernelbench_bins = _operator_bin_shares(kernelbench)
    positions = list(range(len(train_bins)))
    ax.plot(positions, train_bins, marker="o", linewidth=2.4, label=train_label, color=TRAIN_COLOR)
    ax.plot(
        positions,
        kernelbench_bins,
        marker="o",
        linewidth=2.4,
        label=kernelbench_label,
        color=KERNELBENCH_COLOR,
    )
    ticks = list(range(0, len(positions), 2))
    labels = [str(value) for value in ticks]
    labels[-1] = f"{positions[-1]}+"
    ax.set_xticks(ticks, labels)
    ax.set_title("C. Tasks by exact operator count")
    ax.set_xlabel("Operators per task")
    ax.set_ylabel("Tasks in exclusive bin (%)")
    ax.legend()
    _style_axis(ax)

    ax = axes[1, 1]
    labels = [label for _name, label in FAMILIES]
    train_families = _family_shares(train)
    kernelbench_families = _family_shares(kernelbench)
    positions = list(range(len(labels)))
    height = 0.36
    ax.barh(
        [position + height / 2 for position in positions],
        train_families,
        height,
        label=train_label,
        color=TRAIN_COLOR,
    )
    ax.barh(
        [position - height / 2 for position in positions],
        kernelbench_families,
        height,
        label=kernelbench_label,
        color=KERNELBENCH_COLOR,
    )
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_title("D. Operator-family prevalence")
    ax.set_xlabel("Rows containing family (%)")
    ax.legend()
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.8)
    ax.set_axisbelow(True)

    fig.suptitle(
        f"{train_label} (n={train['rows']:,}) vs {kernelbench_label} (n={kernelbench['rows']:,})",
        fontsize=20,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    destination = output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    render(load_profile(args.profile), args.output)
    print(args.output.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
