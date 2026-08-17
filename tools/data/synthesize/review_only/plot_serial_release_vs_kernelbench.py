#!/usr/bin/env python3
"""Profile the serial release and render bound KernelBench comparison figures."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from tools.data.synthesize.profile_prompt_tvm_distribution import profile_dataset


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_RELEASE = (
    REPO_ROOT
    / "local_artifacts/data/synthesize/prompt_tvm_v4/release_sources/serial_one_question_per_parent.v1/selected.parquet"
)
INTERMEDIATES = REPO_ROOT / "Data/prompt_tvm_v4/intermediate_artifacts"
DEFAULT_CANONICAL = INTERMEDIATES / "train.review.parquet"
DEFAULT_TERMINAL = (
    INTERMEDIATES / "serial_augmentation_v1/layout_fallback_base.v2.semantic_gate_double/selected.parquet"
)
DEFAULT_KERNELBENCH = tuple(
    REPO_ROOT / f"Data/kernelbench-level{level}-validation-tvm-v2/train.parquet" for level in (1, 2, 3)
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "handoffs/data/synthesize/artifacts"
DEFAULT_REPORT = (
    REPO_ROOT
    / "local_artifacts/data/synthesize/prompt_tvm_v4/release_sources/serial_one_question_per_parent.v1/kernelbench_comparison.json"
)
CONTRACT = "prompt_tvm_v4_serial_release_kernelbench_comparison_v1"
FAMILY_ORDER = (
    "activation",
    "attention_recurrent",
    "conv",
    "indexing_scatter",
    "loss_distance",
    "matmul_linear",
    "normalization",
    "pooling",
    "reduction",
    "shape_layout",
    "sort_select",
)
FAMILY_LABELS = (
    "Activation",
    "Attention / recurrent",
    "Convolution",
    "Indexing / scatter",
    "Loss / distance",
    "Matmul / linear",
    "Normalization",
    "Pooling",
    "Reduction",
    "Shape / layout",
    "Sort / select",
)
COLORS = {"Release": "#4C78A8", "KernelBench": "#F58518", "Canonical": "#72B7B2", "Terminal": "#B279A2"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "rows": pq.ParquetFile(path).metadata.num_rows, "sha256": _sha256(path)}


def _profile(label: str, paths: Sequence[Path]) -> dict[str, Any]:
    return profile_dataset(label, paths)


def _operator_groups(profile: Mapping[str, Any]) -> dict[str, int]:
    histogram = profile["operator_count"]["histogram"]
    values = {int(key): int(value) for key, value in histogram.items()}
    return {
        "<=1": sum(value for key, value in values.items() if key <= 1),
        "2-5": sum(value for key, value in values.items() if 2 <= key <= 5),
        "6-9": sum(value for key, value in values.items() if 6 <= key <= 9),
        "10-15": sum(value for key, value in values.items() if 10 <= key <= 15),
        ">=16": sum(value for key, value in values.items() if key >= 16),
    }


def _percent(count: int, denominator: int) -> float:
    return 100.0 * count / denominator


def _plot_structure(release: Mapping[str, Any], kernelbench: Mapping[str, Any], output: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold", "axes.edgecolor": "#B7C0CC"})
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.7), gridspec_kw={"width_ratios": [0.88, 1.3]})
    groups = ("<=1", "2-5", "6-9", "10-15", ">=16")
    x = np.arange(len(groups))
    width = 0.36
    for offset, (label, profile) in zip(
        (-width / 2, width / 2), (("Release", release), ("KernelBench", kernelbench)), strict=True
    ):
        counts = _operator_groups(profile)
        values = [_percent(counts[key], int(profile["rows"])) for key in groups]
        bars = axes[0].bar(x + offset, values, width, label=label, color=COLORS[label])
        axes[0].bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=2, fontsize=8)
    axes[0].set_xticks(x, groups)
    axes[0].set_ylabel("Rows (%)")
    axes[0].set_title("Operator-count distribution")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.22)

    release_families = release["operator_family_row_presence"]["families"]
    kb_families = kernelbench["operator_family_row_presence"]["families"]
    y = np.arange(len(FAMILY_ORDER))
    release_values = [float(release_families.get(key, {}).get("percent", 0.0)) for key in FAMILY_ORDER]
    kb_values = [float(kb_families.get(key, {}).get("percent", 0.0)) for key in FAMILY_ORDER]
    axes[1].barh(y + width / 2, release_values, width, label="Release", color=COLORS["Release"])
    axes[1].barh(y - width / 2, kb_values, width, label="KernelBench", color=COLORS["KernelBench"])
    axes[1].set_yticks(y, FAMILY_LABELS)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Rows with family (%) — multi-label")
    axes[1].set_title("Operator-family row presence")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="x", alpha=0.22)
    fig.suptitle("Original-training release candidate vs KernelBench", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_tensor_size(
    canonical: Mapping[str, Any],
    terminal: Mapping[str, Any],
    release: Mapping[str, Any],
    kernelbench: Mapping[str, Any],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.7))
    percentiles = ("p50", "p75", "p90", "p95", "p99")
    datasets = (("Canonical", canonical), ("Terminal", terminal), ("Release", release), ("KernelBench", kernelbench))
    for label, profile in datasets:
        values = [profile["resolved_tensor_numel"]["percentiles"][key] for key in percentiles]
        axes[0].plot(percentiles, values, marker="o", linewidth=2.2, label=label, color=COLORS[label])
    axes[0].set_yscale("log", base=2)
    axes[0].set_ylabel("Tensor elements (log2 scale)")
    axes[0].set_title("Resolved tensor-size percentiles")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False)

    tails = (("≥1M", "1000000"), ("≥10M", "10000000"), ("≥100M", "100000000"))
    x = np.arange(len(tails))
    width = 0.19
    for index, (label, profile) in enumerate(datasets):
        values = [float(profile["resolved_tensor_numel"]["tails"][key]["percent"]) for _, key in tails]
        axes[1].bar(x + (index - 1.5) * width, values, width, label=label, color=COLORS[label])
    axes[1].set_xticks(x, [label for label, _ in tails])
    axes[1].set_ylabel("Resolved occurrences (%)")
    axes[1].set_title("Large-tensor tail coverage")
    axes[1].grid(axis="y", alpha=0.22)
    axes[1].legend(frameon=False, ncol=2)
    fig.suptitle("Shape expansion moves the release toward KernelBench sizes", fontsize=15, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def build(
    *,
    release_path: Path,
    canonical_path: Path,
    terminal_path: Path,
    kernelbench_paths: Sequence[Path],
    output_dir: Path,
    report_path: Path,
) -> dict[str, Any]:
    release_path = release_path.resolve()
    canonical_path = canonical_path.resolve()
    terminal_path = terminal_path.resolve()
    kernelbench_paths = tuple(path.resolve() for path in kernelbench_paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    structure_path = output_dir / "training_release_vs_kernelbench_structure.png"
    tensor_path = output_dir / "training_release_vs_kernelbench_tensor_size.png"
    profiles = {
        "canonical": _profile("Canonical parents", (canonical_path,)),
        "terminal": _profile("Serial terminal", (terminal_path,)),
        "release": _profile("One-parent release", (release_path,)),
        "kernelbench": _profile("KernelBench L1/L2/L3", kernelbench_paths),
    }
    _plot_structure(profiles["release"], profiles["kernelbench"], structure_path)
    _plot_tensor_size(
        profiles["canonical"], profiles["terminal"], profiles["release"], profiles["kernelbench"], tensor_path
    )
    report = {
        "contract": CONTRACT,
        "profiles": profiles,
        "operator_count_groups": {name: _operator_groups(profile) for name, profile in profiles.items()},
        "source_binding": {
            "release": _source(release_path),
            "canonical": _source(canonical_path),
            "terminal": _source(terminal_path),
            **{f"kernelbench_level{index}": _source(path) for index, path in enumerate(kernelbench_paths, 1)},
            "analyzer": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
            "profile_helper": {
                "path": str((REPO_ROOT / "tools/data/synthesize/profile_prompt_tvm_distribution.py").resolve()),
                "sha256": _sha256(REPO_ROOT / "tools/data/synthesize/profile_prompt_tvm_distribution.py"),
            },
        },
        "artifacts": {
            "structure_figure": {"path": str(structure_path.resolve()), "sha256": _sha256(structure_path)},
            "tensor_size_figure": {"path": str(tensor_path.resolve()), "sha256": _sha256(tensor_path)},
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, default=DEFAULT_RELEASE)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--terminal", type=Path, default=DEFAULT_TERMINAL)
    parser.add_argument("--kernelbench", type=Path, nargs=3, default=DEFAULT_KERNELBENCH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    report = build(
        release_path=args.release,
        canonical_path=args.canonical,
        terminal_path=args.terminal,
        kernelbench_paths=args.kernelbench,
        output_dir=args.output_dir,
        report_path=args.report,
    )
    print(json.dumps(report["artifacts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
