#!/usr/bin/env python3
"""Plot input-tensor numel for every runtime-valid shape child vs KernelBench."""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.cleaning.complexity import extract_operator_signature
from tools.data.synthesize.augment_prompt_tasks import (
    _factory_records,
    _module_constant_environment,
    _top_level_function,
)
from tools.data.synthesize.model_shape.resample_shape_coverage import _load_runtime_lane
from tools.data.synthesize.profile_prompt_tvm_distribution import (
    CATEGORY_PATTERNS,
    FACTORY_BASES,
    _call_name,
    _classify_signature,
    _factory_shape,
    _last_top_level_get_inputs,
    _module_environment,
)

DEFAULT_CENSUS = REPO_ROOT / "Data/prompt_tvm_v4/shape_runtime_coverage_resample_v5/run.low128k_final/summary.json"
DEFAULT_KERNELBENCH = REPO_ROOT / "Data/external/converted/kernelbench_level1_2_3.reference.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "handoffs/data/synthesize/artifacts/shape_child_vs_kernelbench_numel.png"
DEFAULT_STATS = REPO_ROOT / "local_artifacts/data/synthesize/shape_child_vs_kernelbench_numel.json"
DEFAULT_POWER2_OUTPUT = REPO_ROOT / "handoffs/data/synthesize/artifacts/shape_child_vs_kernelbench_power2.png"
DEFAULT_OPERATOR_OUTPUT = (
    REPO_ROOT / "handoffs/data/synthesize/artifacts/shape_child_vs_kernelbench_operator_families.png"
)
DEFAULT_COMPLEXITY_OUTPUT = (
    REPO_ROOT / "handoffs/data/synthesize/artifacts/shape_child_vs_kernelbench_operator_complexity.png"
)

BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("<4K", 0, 2**12),
    ("4K–64K", 2**12, 2**16),
    ("64K–1M", 2**16, 2**20),
    ("1M–16M", 2**20, 2**24),
    ("16M–256M", 2**24, 2**28),
    ("256M–4B", 2**28, 2**32),
    ("≥4B", 2**32, None),
)

OPERATOR_COMPLEXITY_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("0", 0, 1),
    ("1", 1, 2),
    ("2", 2, 3),
    ("3", 3, 4),
    ("4–5", 4, 6),
    ("6–9", 6, 10),
    ("≥10", 10, None),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _child_shapes_from_source(code: str) -> list[tuple[int, ...]]:
    tree = ast.parse(code)
    get_inputs = _top_level_function(tree, "get_inputs")
    records, _ = _factory_records(tree, get_inputs, _module_constant_environment(tree))
    return [record.shape for record in records if math.prod(record.shape) > 0]


def _profile_from_decision(decision: Mapping[str, Any]) -> list[tuple[int, ...]] | None:
    profiles: list[Any] = []
    if isinstance(decision.get("child_input_profile"), Mapping):
        profiles.append(decision["child_input_profile"])
    for attempt in decision.get("attempts", []):
        if isinstance(attempt, Mapping) and isinstance(attempt.get("child_input_profile"), Mapping):
            profiles.append(attempt["child_input_profile"])
    for profile in profiles:
        tensors = profile.get("tensors")
        if profile.get("status") != "passed" or not isinstance(tensors, list):
            continue
        values: list[tuple[int, ...]] = []
        for tensor in tensors:
            if not isinstance(tensor, list) or not tensor or not isinstance(tensor[0], list):
                return None
            shape = tensor[0]
            if any(type(value) is not int or value <= 0 for value in shape):
                return None
            values.append(tuple(shape))
        if values:
            return values
    return None


def _fallback_profiles(manifest_path: Path, required: set[str]) -> dict[str, list[tuple[int, ...]]]:
    if not required:
        return {}
    manifest = json.loads(manifest_path.read_text())
    result: dict[str, list[tuple[int, ...]]] = {}
    for decision in manifest.get("decisions", []):
        child_uuid = decision.get("child_uuid")
        if child_uuid not in required:
            continue
        profile = _profile_from_decision(decision)
        if profile:
            result[str(child_uuid)] = profile
    return result


def _shape_children(
    census_path: Path,
) -> tuple[
    list[tuple[int, ...]],
    dict[str, list[int]],
    dict[str, list[int]],
    dict[str, int],
    dict[str, Any],
]:
    census = json.loads(census_path.read_text())
    all_shapes: list[tuple[int, ...]] = []
    family_numels: dict[str, list[int]] = collections.defaultdict(list)
    complexity_numels: dict[str, list[int]] = collections.defaultdict(list)
    complexity_tasks: collections.Counter[str] = collections.Counter()
    seen_children: set[str] = set()
    lane_records: list[dict[str, Any]] = []

    for source in census["source_runs"]:
        run_dir = Path(source["run_dir"])
        eligible, _, children_path, _, _, _, _ = _load_runtime_lane(run_dir)
        if len(eligible) != source["runtime_eligible_children"]:
            raise ValueError(f"runtime count mismatch for {run_dir}")

        parsed: dict[str, list[tuple[int, ...]]] = {}
        families_by_child: dict[str, set[str]] = {}
        complexity_by_child: dict[str, str] = {}
        parquet = pq.ParquetFile(children_path)
        for batch in parquet.iter_batches(columns=["reward_model", "extra_info"], batch_size=512):
            rewards = batch.column(0).to_pylist()
            extras = batch.column(1).to_pylist()
            for reward, extra in zip(rewards, extras, strict=True):
                child_uuid = extra.get("uuid") if isinstance(extra, Mapping) else None
                if child_uuid not in eligible:
                    continue
                if child_uuid in seen_children:
                    raise ValueError(f"duplicate runtime child UUID: {child_uuid}")
                seen_children.add(child_uuid)
                code = reward.get("ground_truth") if isinstance(reward, Mapping) else None
                if not isinstance(code, str):
                    raise ValueError(f"missing child reference: {child_uuid}")
                try:
                    signature = extract_operator_signature(code)
                    families_by_child[str(child_uuid)] = _classify_signature(signature)
                except (SyntaxError, ValueError):
                    signature = ()
                    families_by_child[str(child_uuid)] = {"other_only"}
                complexity_by_child[str(child_uuid)] = _operator_complexity_bucket(len(signature))
                try:
                    shapes = _child_shapes_from_source(code)
                except (SyntaxError, TypeError, ValueError):
                    continue
                if not shapes:
                    continue
                parsed[str(child_uuid)] = shapes

        missing = eligible - set(parsed)
        fallback = _fallback_profiles(Path(source["profile_evidence_path"]), set(missing))
        still_missing = sorted(missing - set(fallback))
        if still_missing:
            preview = ", ".join(still_missing[:5])
            raise ValueError(f"{run_dir}: {len(still_missing)} child profiles unresolved: {preview}")
        parsed.update(fallback)
        if set(parsed) != eligible:
            raise AssertionError(f"child profile accounting mismatch for {run_dir}")
        for child_uuid in sorted(parsed):
            shapes = parsed[child_uuid]
            all_shapes.extend(shapes)
            numels = [math.prod(shape) for shape in shapes]
            for family in families_by_child[child_uuid]:
                family_numels[family].extend(numels)
            complexity = complexity_by_child[child_uuid]
            complexity_numels[complexity].extend(numels)
            complexity_tasks[complexity] += 1

        lane_records.append(
            {
                "run_dir": str(run_dir),
                "runtime_children": len(eligible),
                "source_profiled_children": len(eligible) - len(fallback),
                "manifest_profiled_children": len(fallback),
                "tensor_occurrences": sum(len(values) for values in parsed.values()),
            }
        )

    expected = sum(int(source["runtime_eligible_children"]) for source in census["source_runs"])
    if len(seen_children) != expected:
        raise ValueError(f"runtime child census mismatch: {len(seen_children)} != {expected}")
    return (
        all_shapes,
        dict(family_numels),
        dict(complexity_numels),
        dict(complexity_tasks),
        {"runtime_children": expected, "lanes": lane_records},
    )


def _kernelbench_shapes(
    path: Path,
) -> tuple[
    list[tuple[int, ...]],
    dict[str, list[int]],
    dict[str, list[int]],
    dict[str, int],
    dict[str, int],
]:
    shapes: list[tuple[int, ...]] = []
    family_numels: dict[str, list[int]] = collections.defaultdict(list)
    complexity_numels: dict[str, list[int]] = collections.defaultdict(list)
    complexity_tasks: collections.Counter[str] = collections.Counter()
    total_factories = 0
    rows = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["reward_model"], batch_size=256):
        for reward in batch.column(0).to_pylist():
            rows += 1
            code = reward.get("ground_truth") if isinstance(reward, Mapping) else None
            if not isinstance(code, str):
                continue
            try:
                signature = extract_operator_signature(code)
                families = _classify_signature(signature)
            except (SyntaxError, ValueError):
                signature = ()
                families = {"other_only"}
            complexity = _operator_complexity_bucket(len(signature))
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            get_inputs = _last_top_level_get_inputs(tree)
            if get_inputs is None:
                continue
            environment = _module_environment(tree)
            calls = sorted(
                (node for node in ast.walk(get_inputs) if isinstance(node, ast.Call)),
                key=lambda node: (getattr(node, "lineno", -1), getattr(node, "col_offset", -1)),
            )
            positional = {"rand", "randn", "zeros", "ones", "empty", "full", "normal", "uniform"}
            row_numels: list[int] = []
            for call in calls:
                base = _call_name(call.func).split(".")[-1].lower()
                if base not in FACTORY_BASES or (base in positional and not call.args):
                    continue
                total_factories += 1
                shape = _factory_shape(call, base, environment)
                if not shape or any(type(value) not in (int, float) or value < 0 for value in shape):
                    continue
                numel = math.prod(shape)
                if numel > 0:
                    resolved_shape = tuple(int(value) for value in shape)
                    shapes.append(resolved_shape)
                    row_numels.append(int(numel))
                    for family in families:
                        family_numels[family].append(int(numel))
            if row_numels:
                complexity_numels[complexity].extend(row_numels)
                complexity_tasks[complexity] += 1
    return (
        shapes,
        dict(family_numels),
        dict(complexity_numels),
        dict(complexity_tasks),
        {"rows": rows, "total_factory_occurrences": total_factories},
    )


def _nearest(values: Sequence[int], percentile: int) -> int:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * percentile / 100)]


def _bucket_counts(values: Sequence[int]) -> list[int]:
    counts = []
    for _, lower, upper in BUCKETS:
        counts.append(sum(value >= lower and (upper is None or value < upper) for value in values))
    if sum(counts) != len(values):
        raise AssertionError("numel buckets do not cover every observation")
    return counts


def _operator_complexity_bucket(operator_count: int) -> str:
    for label, lower, upper in OPERATOR_COMPLEXITY_BUCKETS:
        if operator_count >= lower and (upper is None or operator_count < upper):
            return label
    raise AssertionError(f"operator count is outside complexity buckets: {operator_count}")


def _summary(values: Sequence[int]) -> dict[str, Any]:
    counts = _bucket_counts(values)
    return {
        "tensor_occurrences": len(values),
        "percentiles": {f"p{p}": _nearest(values, p) for p in (50, 75, 90, 95, 99)},
        "buckets": {
            label: {"count": count, "percent": count * 100 / len(values)}
            for (label, _, _), count in zip(BUCKETS, counts, strict=True)
        },
    }


def _plot(child_values: Sequence[int], kernelbench_values: Sequence[int], output: Path) -> None:
    colors = {"children": "#2563EB", "kernelbench": "#E11D48"}
    fig, (ax_cdf, ax_hist) = plt.subplots(1, 2, figsize=(13.5, 5.2), constrained_layout=True)

    for label, values, color in (
        ("All shape children", child_values, colors["children"]),
        ("KernelBench", kernelbench_values, colors["kernelbench"]),
    ):
        ordered = np.sort(np.asarray(values, dtype=np.float64))
        y = np.arange(1, len(ordered) + 1) / len(ordered)
        ax_cdf.step(ordered, y, where="post", label=f"{label} (n={len(values):,})", color=color, linewidth=2)
    ax_cdf.set_xscale("log", base=2)
    ax_cdf.set_xlabel("Input tensor numel (log₂ scale)")
    ax_cdf.set_ylabel("Cumulative fraction")
    ax_cdf.set_title("Empirical CDF")
    ax_cdf.grid(True, which="both", alpha=0.22)
    ax_cdf.legend(loc="lower right")

    x = np.arange(len(BUCKETS))
    width = 0.38
    child_counts = _bucket_counts(child_values)
    kb_counts = _bucket_counts(kernelbench_values)
    child_pct = np.asarray(child_counts) * 100 / len(child_values)
    kb_pct = np.asarray(kb_counts) * 100 / len(kernelbench_values)
    ax_hist.bar(x - width / 2, child_pct, width, label="All shape children", color=colors["children"])
    ax_hist.bar(x + width / 2, kb_pct, width, label="KernelBench", color=colors["kernelbench"])
    ax_hist.set_xticks(x, [label for label, _, _ in BUCKETS], rotation=30, ha="right")
    ax_hist.set_ylabel("Tensor occurrences (%)")
    ax_hist.set_title("Normalized size buckets")
    ax_hist.grid(True, axis="y", alpha=0.22)
    ax_hist.legend(loc="upper left")

    fig.suptitle("Input tensor size distribution: all H20-valid shape children vs KernelBench", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _power2_summary(shapes: Sequence[tuple[int, ...]]) -> dict[str, Any]:
    dimensions = [value for shape in shapes for value in shape]
    powers = [value for value in dimensions if value > 0 and value & (value - 1) == 0]
    exact_counts = collections.Counter(dimensions)
    power_counts = collections.Counter(value.bit_length() - 1 for value in powers)
    top = exact_counts.most_common(10)
    return {
        "axis_occurrences": len(dimensions),
        "power_of_two_occurrences": len(powers),
        "power_of_two_percent": len(powers) * 100 / len(dimensions),
        "top10_exact_values": [
            {"value": value, "count": count, "percent": count * 100 / len(dimensions)} for value, count in top
        ],
        "top10_exact_value_percent": sum(count for _, count in top) * 100 / len(dimensions),
        "power_exponents": {str(exponent): count for exponent, count in sorted(power_counts.items())},
    }


def _plot_power2(
    child_shapes: Sequence[tuple[int, ...]],
    kernelbench_shapes: Sequence[tuple[int, ...]],
    output: Path,
) -> None:
    child = _power2_summary(child_shapes)
    kernelbench = _power2_summary(kernelbench_shapes)
    colors = {"children": "#2563EB", "kernelbench": "#E11D48"}
    fig, (ax_share, ax_exp) = plt.subplots(1, 2, figsize=(13.5, 5.0), constrained_layout=True)

    labels = ["All shape children", "KernelBench"]
    shares = [child["power_of_two_percent"], kernelbench["power_of_two_percent"]]
    bars = ax_share.bar(labels, shares, color=[colors["children"], colors["kernelbench"]], width=0.58)
    ax_share.set_ylim(0, max(shares) * 1.22)
    ax_share.set_ylabel("Axis occurrences that are powers of two (%)")
    ax_share.set_title("Overall power-of-two concentration")
    ax_share.grid(True, axis="y", alpha=0.22)
    for bar, value in zip(bars, shares, strict=True):
        ax_share.text(bar.get_x() + bar.get_width() / 2, value + 0.8, f"{value:.2f}%", ha="center")

    exponents = sorted(
        {int(value) for value in child["power_exponents"]} | {int(value) for value in kernelbench["power_exponents"]}
    )
    for label, summary, color, marker in (
        ("All shape children", child, colors["children"], "o"),
        ("KernelBench", kernelbench, colors["kernelbench"], "s"),
    ):
        total = summary["axis_occurrences"]
        percentages = [summary["power_exponents"].get(str(exponent), 0) * 100 / total for exponent in exponents]
        ax_exp.plot(exponents, percentages, label=label, color=color, marker=marker, linewidth=1.8, markersize=4)
    ax_exp.set_xlabel("Exponent k in axis value 2ᵏ")
    ax_exp.set_ylabel("Share of all axis occurrences (%)")
    ax_exp.set_title("Where power-of-two axes concentrate")
    ax_exp.set_xticks(exponents[::2] if len(exponents) > 16 else exponents)
    ax_exp.grid(True, alpha=0.22)
    ax_exp.legend()

    fig.suptitle("Shape-axis power-of-two concentration: all H20-valid children vs KernelBench", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _family_bucket_matrix(family_numels: Mapping[str, Sequence[int]], families: Sequence[str]) -> np.ndarray:
    matrix = np.full((len(families), len(BUCKETS)), np.nan)
    for row, family in enumerate(families):
        values = family_numels.get(family, ())
        if not values:
            continue
        matrix[row, :] = np.asarray(_bucket_counts(values), dtype=np.float64) * 100 / len(values)
    return matrix


def _plot_operator_families(
    child_families: Mapping[str, Sequence[int]],
    kernelbench_families: Mapping[str, Sequence[int]],
    output: Path,
) -> None:
    ordered = list(CATEGORY_PATTERNS) + ["other_only"]
    families = [family for family in ordered if child_families.get(family) or kernelbench_families.get(family)]
    child_matrix = _family_bucket_matrix(child_families, families)
    kb_matrix = _family_bucket_matrix(kernelbench_families, families)
    finite = np.concatenate((child_matrix[np.isfinite(child_matrix)], kb_matrix[np.isfinite(kb_matrix)]))
    maximum = max(1.0, float(finite.max()))

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 8.0), constrained_layout=True, sharey=False)
    images = []
    for ax, title, matrix, source in (
        (axes[0], "All shape children", child_matrix, child_families),
        (axes[1], "KernelBench", kb_matrix, kernelbench_families),
    ):
        image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu", vmin=0, vmax=maximum)
        images.append(image)
        ax.set_title(title)
        ax.set_xticks(range(len(BUCKETS)), [label for label, _, _ in BUCKETS], rotation=35, ha="right")
        labels = [f"{family} (n={len(source.get(family, ())):,})" for family in families]
        ax.set_yticks(range(len(families)), labels)
        for row in range(len(families)):
            for column in range(len(BUCKETS)):
                value = matrix[row, column]
                if np.isfinite(value):
                    color = "white" if value > maximum * 0.55 else "black"
                    ax.text(column, row, f"{value:.0f}", ha="center", va="center", fontsize=6.5, color=color)
                else:
                    ax.text(column, row, "NA", ha="center", va="center", fontsize=6.5, color="#666666")
    colorbar = fig.colorbar(images[0], ax=axes, shrink=0.82)
    colorbar.set_label("Tensor occurrences in family (%)")
    fig.suptitle("Input tensor numel by operator family (multi-label tasks; cells are row percentages)", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_operator_complexity(
    child_complexity: Mapping[str, Sequence[int]],
    child_tasks: Mapping[str, int],
    kernelbench_complexity: Mapping[str, Sequence[int]],
    kernelbench_tasks: Mapping[str, int],
    output: Path,
) -> None:
    levels = [
        label
        for label, _, _ in OPERATOR_COMPLEXITY_BUCKETS
        if child_complexity.get(label) or kernelbench_complexity.get(label)
    ]
    child_matrix = _family_bucket_matrix(child_complexity, levels)
    kb_matrix = _family_bucket_matrix(kernelbench_complexity, levels)
    finite = np.concatenate((child_matrix[np.isfinite(child_matrix)], kb_matrix[np.isfinite(kb_matrix)]))
    maximum = max(1.0, float(finite.max()))

    fig, axes = plt.subplots(1, 2, figsize=(16.5, 6.2), constrained_layout=True, sharey=False)
    images = []
    for ax, title, matrix, source, task_counts in (
        (axes[0], "All shape children", child_matrix, child_complexity, child_tasks),
        (axes[1], "KernelBench", kb_matrix, kernelbench_complexity, kernelbench_tasks),
    ):
        image = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=maximum)
        images.append(image)
        ax.set_title(title)
        ax.set_xticks(range(len(BUCKETS)), [label for label, _, _ in BUCKETS], rotation=35, ha="right")
        labels = [
            f"{level} ops (tasks={task_counts.get(level, 0):,}, tensors={len(source.get(level, ())):,})"
            for level in levels
        ]
        ax.set_yticks(range(len(levels)), labels)
        for row in range(len(levels)):
            for column in range(len(BUCKETS)):
                value = matrix[row, column]
                if np.isfinite(value):
                    color = "white" if value > maximum * 0.55 else "black"
                    ax.text(column, row, f"{value:.0f}", ha="center", va="center", fontsize=7, color=color)
                else:
                    ax.text(column, row, "NA", ha="center", va="center", fontsize=7, color="#666666")
    colorbar = fig.colorbar(images[0], ax=axes, shrink=0.82)
    colorbar.set_label("Tensor occurrences in operator-count bucket (%)")
    fig.suptitle("Input tensor numel by task operator count (input-dependent compute multiset)", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _family_summary(family_numels: Mapping[str, Sequence[int]]) -> dict[str, Any]:
    return {family: _summary(values) for family, values in sorted(family_numels.items()) if values}


def _complexity_summary(
    complexity_numels: Mapping[str, Sequence[int]], complexity_tasks: Mapping[str, int]
) -> dict[str, Any]:
    return {
        level: {"task_count": complexity_tasks.get(level, 0), **_summary(complexity_numels[level])}
        for level, _, _ in OPERATOR_COMPLEXITY_BUCKETS
        if complexity_numels.get(level)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
    parser.add_argument("--kernelbench", type=Path, default=DEFAULT_KERNELBENCH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--power2-output", type=Path, default=DEFAULT_POWER2_OUTPUT)
    parser.add_argument("--operator-output", type=Path, default=DEFAULT_OPERATOR_OUTPUT)
    parser.add_argument("--complexity-output", type=Path, default=DEFAULT_COMPLEXITY_OUTPUT)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    args = parser.parse_args()

    child_shapes, child_families, child_complexity, child_complexity_tasks, child_accounting = _shape_children(
        args.census.resolve()
    )
    (
        kernelbench_shapes,
        kernelbench_families,
        kernelbench_complexity,
        kernelbench_complexity_tasks,
        kernelbench_accounting,
    ) = _kernelbench_shapes(args.kernelbench.resolve())
    child_values = [math.prod(shape) for shape in child_shapes]
    kernelbench_values = [math.prod(shape) for shape in kernelbench_shapes]
    _plot(child_values, kernelbench_values, args.output.resolve())
    _plot_power2(child_shapes, kernelbench_shapes, args.power2_output.resolve())
    _plot_operator_families(child_families, kernelbench_families, args.operator_output.resolve())
    _plot_operator_complexity(
        child_complexity,
        child_complexity_tasks,
        kernelbench_complexity,
        kernelbench_complexity_tasks,
        args.complexity_output.resolve(),
    )

    stats = {
        "contract": {
            "children": "every returned input tensor occurrence in every H20-runtime-valid shape child",
            "kernelbench": "every statically resolved positive input factory occurrence",
            "percentile": "nearest observation at round((n - 1) * p)",
            "uses_downstream_random_filter": False,
        },
        "sources": {
            "shape_census": {"path": str(args.census.resolve()), "sha256": _sha256(args.census.resolve())},
            "kernelbench": {
                "path": str(args.kernelbench.resolve()),
                "sha256": _sha256(args.kernelbench.resolve()),
            },
        },
        "children": {
            **child_accounting,
            **_summary(child_values),
            "power_of_two": _power2_summary(child_shapes),
            "operator_families": _family_summary(child_families),
            "operator_complexity": _complexity_summary(child_complexity, child_complexity_tasks),
        },
        "kernelbench": {
            **kernelbench_accounting,
            **_summary(kernelbench_values),
            "power_of_two": _power2_summary(kernelbench_shapes),
            "operator_families": _family_summary(kernelbench_families),
            "operator_complexity": _complexity_summary(kernelbench_complexity, kernelbench_complexity_tasks),
        },
        "artifacts": {
            "numel_plot": str(args.output.resolve()),
            "power2_plot": str(args.power2_output.resolve()),
            "operator_family_plot": str(args.operator_output.resolve()),
            "operator_complexity_plot": str(args.complexity_output.resolve()),
        },
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    args.stats.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
