#!/usr/bin/env python3
"""Count statically resolved ``get_inputs`` tensor-shape dimension values.

The official KernelBench validation sets and the separately screened v3 sets
are reported as different corpora.  Task source is parsed but never executed.
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.synthesize.augment_prompt_tasks import _SHAPE_FACTORIES, _call_name, _shape_nodes  # noqa: E402
from tools.data.synthesize.profile_prompt_tvm_distribution import _last_top_level_get_inputs, _safe_value  # noqa: E402

SCHEMA_VERSION = "kernelbench-shape-value-frequency-v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Data/kernelbench_shape_frequency"
DEFAULT_CORPORA = {
    "official_full": tuple(
        REPO_ROOT / f"Data/kernelbench-level{level}-validation-tvm-v2/train.parquet" for level in (1, 2, 3)
    ),
    "screened_v3": tuple(REPO_ROOT / f"Data/kernelbench-tvm-v3/level{level}/train.parquet" for level in (1, 2, 3)),
}


class ShapeProfileError(ValueError):
    """An input dataset does not satisfy the profiling contract."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _percent(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 6) if denominator else 0.0


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _bind_target(target: ast.AST, value: Any, environment: dict[str, Any]) -> bool:
    if isinstance(target, ast.Name):
        environment[target.id] = value
        return True
    if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (tuple, list)):
        if len(target.elts) != len(value):
            return False
        updates = dict(environment)
        if all(_bind_target(item, item_value, updates) for item, item_value in zip(target.elts, value, strict=True)):
            environment.update(updates)
            return True
    return False


def _apply_assignment(statement: ast.stmt, environment: dict[str, Any]) -> None:
    if isinstance(statement, ast.Assign):
        value_node = statement.value
        targets: Sequence[ast.AST] = statement.targets
    elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
        value_node = statement.value
        targets = (statement.target,)
    else:
        return
    try:
        value = _safe_value(value_node, environment)
    except (ArithmeticError, OverflowError, ValueError):
        return
    updates = dict(environment)
    if all(_bind_target(target, value, updates) for target in targets):
        environment.update(updates)


def _module_environment(tree: ast.Module) -> dict[str, Any]:
    environment: dict[str, Any] = {}
    for statement in tree.body:
        _apply_assignment(statement, environment)
    return environment


def _flatten_positive_integer_shape(nodes: Sequence[ast.AST], environment: Mapping[str, Any]) -> tuple[int, ...]:
    values: list[int] = []
    for node in nodes:
        value = _safe_value(node.value if isinstance(node, ast.Starred) else node, environment)
        expanded = value if isinstance(value, (tuple, list)) else (value,)
        for dimension in expanded:
            if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0:
                raise ValueError(f"not a positive integer dimension: {dimension!r}")
            values.append(dimension)
    return tuple(values)


def _resolved_call_shape(call: ast.Call, name: str, environment: Mapping[str, Any]) -> tuple[int, ...]:
    nodes = _shape_nodes(call, name)
    if nodes:
        return _flatten_positive_integer_shape(nodes, environment)
    size_node = next((keyword.value for keyword in call.keywords if keyword.arg in {"size", "shape"}), None)
    if size_node is None and call.args:
        size_node = call.args[2] if name == "torch.randint" and len(call.args) >= 3 else call.args[0]
    if isinstance(size_node, (ast.Tuple, ast.List)) and not size_node.elts:
        return ()  # A scalar tensor has a resolved rank-zero shape and no dimension values.
    raise ValueError("empty or dynamic shape")


def extract_task_dimensions(code: str) -> dict[str, Any]:
    """Extract independently resolvable factory shapes from one task."""

    tree = ast.parse(code)
    get_inputs = _last_top_level_get_inputs(tree)
    if get_inputs is None:
        return {"dimensions": [], "resolved_shapes": [], "factory_calls": 0, "unresolved": ["missing_get_inputs"]}

    environment = _module_environment(tree)
    dimensions: list[int] = []
    resolved_shapes: list[tuple[int, ...]] = []
    unresolved: list[str] = []
    factory_calls = 0
    for statement in get_inputs.body:
        calls = sorted(
            (
                node
                for node in ast.walk(statement)
                if isinstance(node, ast.Call) and _call_name(node.func) in _SHAPE_FACTORIES
            ),
            key=lambda node: (getattr(node, "lineno", -1), getattr(node, "col_offset", -1)),
        )
        for call in calls:
            factory_calls += 1
            name = _call_name(call.func)
            try:
                shape = _resolved_call_shape(call, name, environment)
            except (ArithmeticError, OverflowError, ValueError) as exc:
                unresolved.append(f"line {getattr(call, 'lineno', -1)} {name}: {exc}")
                continue
            resolved_shapes.append(shape)
            dimensions.extend(shape)
        _apply_assignment(statement, environment)
    return {
        "dimensions": dimensions,
        "resolved_shapes": resolved_shapes,
        "factory_calls": factory_calls,
        "unresolved": unresolved,
    }


def _level_name(path: Path) -> str:
    for part in path.parts:
        if part.startswith("level") and part[5:].isdigit():
            return part
        if part.startswith("kernelbench-level"):
            suffix = part.removeprefix("kernelbench-level").split("-", 1)[0]
            if suffix.isdigit():
                return f"level{suffix}"
    raise ShapeProfileError(f"cannot infer KernelBench level from {path}")


def profile_paths(paths: Sequence[Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    occurrence_counts: collections.Counter[int] = collections.Counter()
    task_sets: dict[int, set[str]] = collections.defaultdict(set)
    rows = 0
    tasks_with_dimensions = 0
    factory_calls = 0
    resolved_shapes = 0
    unresolved_shapes = 0
    parse_failures = 0
    missing_get_inputs = 0
    unresolved_examples: list[dict[str, Any]] = []
    sources = []

    for path in paths:
        path = path.resolve()
        parquet = pq.ParquetFile(path)
        sources.append({"path": _display_path(path), "rows": parquet.metadata.num_rows, "sha256": _sha256_file(path)})
        row_index = 0
        for batch in parquet.iter_batches(columns=["reward_model"], batch_size=512, use_threads=False):
            for reward_model in batch.column(0).to_pylist():
                task_id = f"{_level_name(path)}:{row_index}"
                rows += 1
                code = reward_model.get("ground_truth") if isinstance(reward_model, Mapping) else None
                if not isinstance(code, str):
                    raise ShapeProfileError(f"{path}: row {row_index} lacks reward_model.ground_truth")
                try:
                    extracted = extract_task_dimensions(code)
                except SyntaxError as exc:
                    parse_failures += 1
                    if len(unresolved_examples) < 20:
                        unresolved_examples.append({"task": task_id, "reason": f"syntax_error:{exc.msg}"})
                    row_index += 1
                    continue
                values = extracted["dimensions"]
                if values:
                    tasks_with_dimensions += 1
                occurrence_counts.update(values)
                for value in set(values):
                    task_sets[value].add(task_id)
                factory_calls += extracted["factory_calls"]
                resolved_shapes += len(extracted["resolved_shapes"])
                unresolved_shapes += len(extracted["unresolved"])
                if extracted["unresolved"] == ["missing_get_inputs"]:
                    missing_get_inputs += 1
                for reason in extracted["unresolved"]:
                    if len(unresolved_examples) < 20:
                        unresolved_examples.append({"task": task_id, "reason": reason})
                row_index += 1
        if row_index != parquet.metadata.num_rows:
            raise AssertionError(f"streamed row count differs for {path}")

    total_occurrences = sum(occurrence_counts.values())
    total_task_value_pairs = sum(len(tasks) for tasks in task_sets.values())
    power_occurrences = sum(count for value, count in occurrence_counts.items() if _is_power_of_two(value))
    power_task_pairs = sum(len(tasks) for value, tasks in task_sets.items() if _is_power_of_two(value))
    gt1_occurrences = sum(count for value, count in occurrence_counts.items() if value > 1)
    gt1_power_occurrences = sum(
        count for value, count in occurrence_counts.items() if value > 1 and _is_power_of_two(value)
    )
    top_values = [
        {
            "value": value,
            "dimension_occurrences": count,
            "occurrence_percent": _percent(count, total_occurrences),
            "task_count": len(task_sets[value]),
            "task_percent": _percent(len(task_sets[value]), rows),
            "is_power_of_two": _is_power_of_two(value),
        }
        for value, count in sorted(occurrence_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    summary = {
        "tasks": rows,
        "tasks_with_resolved_dimensions": tasks_with_dimensions,
        "factory_calls": factory_calls,
        "resolved_factory_shapes": resolved_shapes,
        "unresolved_factory_shapes": unresolved_shapes,
        "parse_failures": parse_failures,
        "missing_get_inputs": missing_get_inputs,
        "dimension_occurrences": total_occurrences,
        "task_value_pairs": total_task_value_pairs,
        "unique_dimension_values": len(occurrence_counts),
        "power_of_two": {
            "definition": "positive integer n with n & (n - 1) == 0; includes 1 = 2^0",
            "dimension_occurrences": power_occurrences,
            "dimension_occurrence_percent": _percent(power_occurrences, total_occurrences),
            "task_value_pairs": power_task_pairs,
            "task_value_pair_percent": _percent(power_task_pairs, total_task_value_pairs),
            "unique_values": sum(_is_power_of_two(value) for value in occurrence_counts),
            "unique_value_percent": _percent(
                sum(_is_power_of_two(value) for value in occurrence_counts), len(occurrence_counts)
            ),
            "excluding_one_dimension_occurrences": gt1_power_occurrences,
            "excluding_one_denominator": gt1_occurrences,
            "excluding_one_percent": _percent(gt1_power_occurrences, gt1_occurrences),
        },
        "top_values": top_values[:30],
        "unresolved_examples": unresolved_examples,
        "sources": sources,
    }
    frequencies = [
        {
            "value": value,
            "dimension_occurrences": occurrence_counts[value],
            "task_count": len(task_sets[value]),
            "is_power_of_two": _is_power_of_two(value),
        }
        for value in sorted(occurrence_counts)
    ]
    return summary, frequencies


def build_report(corpora: Mapping[str, Sequence[Path]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "extraction_contract": {
            "code_field": "reward_model.ground_truth",
            "execution": "none; Python AST only",
            "get_inputs": "last top-level get_inputs definition",
            "factories": sorted(_SHAPE_FACTORIES),
            "dimensions": "positive integer shape dimensions resolved from module/get_inputs assignments",
            "unresolved_policy": "count each unresolved factory but retain other independently resolved factories in the task",
            "frequency_units": {
                "dimension_occurrences": "each dimension slot in each resolved tensor factory shape",
                "task_count": "a task contributes at most once per dimension value",
            },
        },
        "corpora": {},
    }
    frequency_rows: list[dict[str, Any]] = []
    for corpus, paths in corpora.items():
        combined, combined_frequencies = profile_paths(paths)
        levels = {}
        for path in paths:
            level = _level_name(path)
            level_summary, level_frequencies = profile_paths((path,))
            levels[level] = level_summary
            frequency_rows.extend({"corpus": corpus, "level": level, **row} for row in level_frequencies)
        report["corpora"][corpus] = {"combined": combined, "levels": levels}
        frequency_rows.extend({"corpus": corpus, "level": "combined", **row} for row in combined_frequencies)
    return report, frequency_rows


def _markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# KernelBench `get_inputs` shape-value frequency",
        "",
        "This report counts statically resolved positive integer dimensions in tensor factories inside the last top-level `get_inputs()`. Task code is never executed. Repeated dimension slots count repeatedly; task coverage counts each value at most once per task.",
        "",
        "## Summary",
        "",
        "| Corpus | Level | Tasks | Resolved shapes | Unresolved | Dimension occurrences | Power-of-two | Power-of-two (>1) | Unique values |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for corpus, corpus_report in report["corpora"].items():
        for level, summary in [("combined", corpus_report["combined"]), *sorted(corpus_report["levels"].items())]:
            power = summary["power_of_two"]
            lines.append(
                f"| {corpus} | {level} | {summary['tasks']} | {summary['resolved_factory_shapes']} | "
                f"{summary['unresolved_factory_shapes']} | {summary['dimension_occurrences']} | "
                f"{power['dimension_occurrences']} ({power['dimension_occurrence_percent']:.2f}%) | "
                f"{power['excluding_one_dimension_occurrences']}/{power['excluding_one_denominator']} "
                f"({power['excluding_one_percent']:.2f}%) | {summary['unique_dimension_values']} |"
            )
    lines.extend(["", "## Most frequent values", ""])
    for corpus, corpus_report in report["corpora"].items():
        summary = corpus_report["combined"]
        lines.extend(
            [
                f"### {corpus}",
                "",
                "| Value | Dimension occurrences | Occurrence share | Task coverage | Power of two |",
                "|---:|---:|---:|---:|:---:|",
            ]
        )
        for row in summary["top_values"][:20]:
            lines.append(
                f"| {row['value']} | {row['dimension_occurrences']} | {row['occurrence_percent']:.2f}% | "
                f"{row['task_count']}/{summary['tasks']} ({row['task_percent']:.2f}%) | "
                f"{'yes' if row['is_power_of_two'] else 'no'} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Extraction boundary",
            "",
            "- Included factories are listed in `summary.json`; the current KernelBench inputs use `torch.rand` and `torch.randint`.",
            "- Module constants, chained assignments, tuple unpacking, simple arithmetic, and direct local assignments are resolved.",
            "- Dynamic expressions such as `x.shape` are not guessed. An unresolved factory is counted separately, while other resolvable factories in the same task remain included.",
            "- `1` is mathematically counted as `2^0`; the table also reports a power-of-two percentage after excluding dimension value 1.",
            "- `official_full` is the complete 100/100/50 validation corpus. `screened_v3` is a separate 100/93/37 post-screening comparison and must not replace the official denominator.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(output_dir: Path, report: Mapping[str, Any], frequency_rows: Sequence[Mapping[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (output_dir / "summary.md").write_text(_markdown(report))
    columns = ("corpus", "level", "value", "dimension_occurrences", "task_count", "is_power_of_two")
    lines = ["\t".join(columns)]
    lines.extend(
        "\t".join(
            str(row[column]).lower() if isinstance(row[column], bool) else str(row[column]) for column in columns
        )
        for row in frequency_rows
    )
    (output_dir / "value_frequency.tsv").write_text("\n".join(lines) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    report, frequency_rows = build_report(DEFAULT_CORPORA)
    write_outputs(args.output_dir, report, frequency_rows)
    print(json.dumps({"output_dir": str(args.output_dir), "corpora": list(report["corpora"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
