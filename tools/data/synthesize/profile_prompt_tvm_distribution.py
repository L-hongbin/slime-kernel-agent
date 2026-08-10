#!/usr/bin/env python3
"""Profile the exact parquet inputs used by the distribution figures.

The profiler never executes task source.  It streams reference text from the
parquets, parses it with Python's AST, and publishes exact counts plus the
input/report hashes needed to reproduce every plotted value.  KernelBench
step620 correctness is the one exception: raw evaluation dumps are not part of
the data handoff, so those integer counts remain pinned to two audited reports
whose byte hashes are checked before publication.
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.cleaning.complexity import extract_operator_signature

PROFILE_SCHEMA_VERSION = "prompt-tvm-distribution-profile-v1"
PROFILER_VERSION = "prompt-tvm-distribution-profiler-v1"
REPO_ROOT = _REPO_ROOT
DEFAULT_OUTPUT = REPO_ROOT / "Data/prompt_tvm_v4/analysis/distribution_profile.json"
DEFAULT_TRAIN = REPO_ROOT / "Data/prompt_tvm_v3/drkernel_rl_thinking.parquet"
DEFAULT_KERNELBENCH = tuple(
    REPO_ROOT / f"Data/kernelbench-level{level}-validation-tvm-v2/train.parquet" for level in (1, 2, 3)
)
SHA256_RE = re.compile(r"[0-9a-f]{64}")

FACTORY_BASES = frozenset(
    {
        "arange",
        "empty",
        "full",
        "linspace",
        "normal",
        "ones",
        "rand",
        "randint",
        "randn",
        "randperm",
        "uniform",
        "zeros",
    }
)
ROW_PRESENCE_FACTORIES = ("randn", "rand", "randint")
SHAPE_PERCENTILES = (50, 75, 90, 95, 99)
SHAPE_TAIL_THRESHOLDS = (1_000_000, 10_000_000, 100_000_000)
OP_TAIL_START = 16

CATEGORY_PATTERNS: dict[str, tuple[str, ...]] = {
    "conv": ("conv",),
    "matmul_linear": ("matmul", "bmm", ".mm", "einsum", "linear", "addmm", "dot"),
    "reduction": (
        "sum",
        "mean",
        "prod",
        "amax",
        "amin",
        "argmax",
        "argmin",
        "norm",
        "logsumexp",
        "all",
        "any",
    ),
    "normalization": ("batchnorm", "layernorm", "groupnorm", "instancenorm", "normalize"),
    "activation": (
        "relu",
        "gelu",
        "silu",
        "sigmoid",
        "tanh",
        "softplus",
        "leakyrelu",
        "elu",
        "softmax",
    ),
    "pooling": ("pool",),
    "indexing_scatter": ("gather", "scatter", "index", "embedding", "take", "masked", "where"),
    "sort_select": ("sort", "topk", "kthvalue", "median", "mode"),
    "shape_layout": (
        "reshape",
        "view",
        "permute",
        "transpose",
        "flatten",
        "squeeze",
        "unsqueeze",
        "cat",
        "stack",
        "chunk",
        "split",
        "repeat",
        "expand",
        "contiguous",
    ),
    "attention_recurrent": ("attention", "multihead", "lstm", "gru", "rnn"),
    "loss_distance": ("loss", "cross_entropy", "nll", "cosine_similarity", "pairwise_distance"),
    "fft_sparse": ("fft", "sparse"),
}

PINNED_CORRECTNESS = {
    "L1": {
        "generated": 800,
        "compiled": 782,
        "correct": 723,
        "report": REPO_ROOT / "handoffs/deepseek-v4/iter620_kernelbench_question_audit_20260729.md",
        "report_sha256": "d65a8f71c4c6f6d183473aa530f0d8e9d1eb50e96050a5501e8d9e3500d65af8",
    },
    "L2": {
        "generated": 800,
        "compiled": 666,
        "correct": 549,
        "report": REPO_ROOT / "handoffs/deepseek-v4/iter440_iter620_kernelbench_failure_attribution_20260729.md",
        "report_sha256": "27908959de98ba8df361e3f5c00e79fd5ffd0de3ec3c9dc806f7a4c29e29fcf4",
    },
    "L3": {
        "generated": 400,
        "compiled": 273,
        "correct": 40,
        "report": REPO_ROOT / "handoffs/deepseek-v4/iter440_iter620_kernelbench_failure_attribution_20260729.md",
        "report_sha256": "27908959de98ba8df361e3f5c00e79fd5ffd0de3ec3c9dc806f7a4c29e29fcf4",
    },
}


class ProfileError(ValueError):
    """An input or pinned provenance record is invalid."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _percent(count: int, denominator: int) -> float:
    return round(100.0 * count / denominator, 6) if denominator else 0.0


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _last_top_level_get_inputs(tree: ast.Module) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "get_inputs"
    ]
    return matches[-1] if matches else None


def _safe_value(node: ast.AST, environment: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, bool)):
        return node.value
    if isinstance(node, ast.Name) and node.id in environment:
        return environment[node.id]
    if isinstance(node, (ast.Tuple, ast.List)):
        return tuple(_safe_value(item, environment) for item in node.elts)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_value(node.operand, environment)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("unary operand is not numeric")
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _safe_value(node.left, environment)
        right = _safe_value(node.right, environment)
        try:
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.FloorDiv):
                return left // right
            if isinstance(node.op, ast.Pow):
                return left**right
        except (ArithmeticError, OverflowError, TypeError) as exc:
            raise ValueError("invalid static binary expression") from exc
    raise ValueError("unsupported static expression")


def _module_environment(tree: ast.Module) -> dict[str, Any]:
    environment: dict[str, Any] = {}
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        try:
            value = _safe_value(statement.value, environment)
        except (ArithmeticError, OverflowError, ValueError):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        for target in targets:
            if isinstance(target, ast.Name):
                environment[target.id] = value
    return environment


def _resolved_sequence(node: ast.AST, environment: Mapping[str, Any]) -> tuple[int | float, ...] | None:
    try:
        value = _safe_value(node, environment)
    except (ArithmeticError, OverflowError, ValueError):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = (value,)
    if not isinstance(value, tuple):
        return None
    if any(not isinstance(item, (int, float)) for item in value):
        return None
    return tuple(value)


def _factory_shape(call: ast.Call, base: str, environment: Mapping[str, Any]) -> tuple[int | float, ...] | None:
    if base in {"rand", "randn", "zeros", "ones", "empty", "full", "normal", "uniform"}:
        arguments = call.args[:1] if base == "full" else call.args
        if not arguments:
            return None
        if len(arguments) == 1:
            argument = arguments[0]
            return _resolved_sequence(argument.value if isinstance(argument, ast.Starred) else argument, environment)
        dimensions: list[Any] = []
        for argument in arguments:
            if isinstance(argument, ast.Starred):
                expanded = _resolved_sequence(argument.value, environment)
                if expanded is None:
                    return None
                dimensions.extend(expanded)
            else:
                try:
                    dimensions.append(_safe_value(argument, environment))
                except (ArithmeticError, OverflowError, ValueError):
                    return None
        if any(not isinstance(value, (int, float)) for value in dimensions):
            return None
        return tuple(dimensions)
    if base == "randint":
        node = call.args[2] if len(call.args) >= 3 else call.args[1] if len(call.args) >= 2 else None
        return _resolved_sequence(node, environment) if node is not None else None
    if base in {"randperm", "arange", "linspace"}:
        if not call.args:
            return None
        try:
            first_argument = _safe_value(call.args[0], environment)
        except (ArithmeticError, OverflowError, ValueError):
            return None
        return (first_argument,) if isinstance(first_argument, (int, float)) else None
    return None


def _classify_signature(signature: Sequence[str]) -> set[str]:
    categories: set[str] = set()
    for operator in signature:
        lowered = operator.lower()
        for category, patterns in CATEGORY_PATTERNS.items():
            if any(pattern in lowered for pattern in patterns):
                categories.add(category)
    if not categories:
        categories.add("other_only")
    return categories


def _nearest_observation(values: Sequence[int], percentile: int) -> int:
    if not values:
        raise ProfileError("cannot compute a percentile from no observations")
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percentile / 100.0)
    return ordered[index]


def _iter_reference_code(paths: Sequence[Path]) -> Iterable[tuple[Path, int, str]]:
    for path in paths:
        parquet = pq.ParquetFile(path)
        if "reward_model" not in parquet.schema_arrow.names:
            raise ProfileError(f"parquet lacks reward_model: {path}")
        row_index = 0
        for batch in parquet.iter_batches(columns=["reward_model"], batch_size=2048, use_threads=False):
            for reward_model in batch.column(0).to_pylist():
                code = reward_model.get("ground_truth") if isinstance(reward_model, Mapping) else None
                if not isinstance(code, str) or not code.strip():
                    raise ProfileError(f"{path}: row {row_index} lacks reward_model.ground_truth")
                yield path, row_index, code
                row_index += 1


def _source_records(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        resolved = path.expanduser().resolve()
        if not resolved.is_file() or resolved.suffix != ".parquet":
            raise FileNotFoundError(f"expected parquet input: {resolved}")
        parquet = pq.ParquetFile(resolved)
        records.append(
            {
                "path": _display_path(resolved),
                "sha256": _sha256_file(resolved),
                "rows": parquet.metadata.num_rows,
            }
        )
    return records


def profile_dataset(label: str, paths: Sequence[Path]) -> dict[str, Any]:
    resolved_paths = tuple(path.expanduser().resolve() for path in paths)
    sources = _source_records(resolved_paths)
    rows = 0
    ast_parse_failures = 0
    missing_get_inputs = 0
    signature_failures = 0
    factory_presence: collections.Counter[str] = collections.Counter()
    total_factory_occurrences = 0
    resolved_shape_occurrences = 0
    positive_numel: list[int] = []
    operator_histogram: collections.Counter[int] = collections.Counter()
    operator_families: collections.Counter[str] = collections.Counter()

    for _path, _row_index, code in _iter_reference_code(resolved_paths):
        rows += 1
        try:
            tree = ast.parse(code)
        except SyntaxError:
            ast_parse_failures += 1
            continue

        try:
            signature = extract_operator_signature(code)
        except (SyntaxError, ValueError):
            signature_failures += 1
        else:
            operator_histogram[len(signature)] += 1
            operator_families.update(_classify_signature(signature))

        get_inputs = _last_top_level_get_inputs(tree)
        if get_inputs is None:
            missing_get_inputs += 1
            continue
        calls = [node for node in ast.walk(get_inputs) if isinstance(node, ast.Call)]
        bases = {_call_name(call.func).split(".")[-1].lower() for call in calls}
        factory_presence.update(base for base in ROW_PRESENCE_FACTORIES if base in bases)

        environment = _module_environment(tree)
        factory_calls = sorted(
            (node for node in ast.walk(get_inputs) if isinstance(node, ast.Call)),
            key=lambda node: (getattr(node, "lineno", -1), getattr(node, "col_offset", -1)),
        )
        positional_shape_factories = {
            "rand",
            "randn",
            "zeros",
            "ones",
            "empty",
            "full",
            "normal",
            "uniform",
        }
        for call in factory_calls:
            base = _call_name(call.func).split(".")[-1].lower()
            if base not in FACTORY_BASES or (base in positional_shape_factories and not call.args):
                continue
            total_factory_occurrences += 1
            shape = _factory_shape(call, base, environment)
            if not shape or any(not isinstance(dimension, (int, float)) or dimension < 0 for dimension in shape):
                continue
            resolved_shape_occurrences += 1
            numel = math.prod(shape)
            if numel > 0:
                positive_numel.append(numel)

    if rows != sum(source["rows"] for source in sources):
        raise AssertionError(f"streamed {rows} rows but parquet metadata reports a different total")
    if sum(operator_histogram.values()) + signature_failures + ast_parse_failures != rows:
        raise AssertionError("operator accounting does not cover every row")

    numel_percentiles = {
        f"p{percentile}": _nearest_observation(positive_numel, percentile) for percentile in SHAPE_PERCENTILES
    }
    tails = {
        str(threshold): {
            "count": sum(value >= threshold for value in positive_numel),
            "percent": _percent(sum(value >= threshold for value in positive_numel), len(positive_numel)),
        }
        for threshold in SHAPE_TAIL_THRESHOLDS
    }
    op_bins = []
    for count in range(OP_TAIL_START):
        bin_count = operator_histogram.get(count, 0)
        op_bins.append(
            {
                "label": str(count),
                "minimum": count,
                "maximum": count,
                "count": bin_count,
                "percent": _percent(bin_count, rows),
            }
        )
    tail_count = sum(value for count, value in operator_histogram.items() if count >= OP_TAIL_START)
    op_bins.append(
        {
            "label": f"{OP_TAIL_START}+",
            "minimum": OP_TAIL_START,
            "maximum": None,
            "count": tail_count,
            "percent": _percent(tail_count, rows),
        }
    )

    return {
        "label": label,
        "sources": sources,
        "rows": rows,
        "parse_accounting": {
            "ast_parse_failures": ast_parse_failures,
            "operator_signature_failures": signature_failures,
            "missing_get_inputs": missing_get_inputs,
        },
        "input_factory_row_presence": {
            "denominator": "all dataset rows",
            "non_exclusive": True,
            "factories": {
                name: {
                    "count": factory_presence[name],
                    "percent": _percent(factory_presence[name], rows),
                }
                for name in ROW_PRESENCE_FACTORIES
            },
        },
        "resolved_tensor_numel": {
            "unit": "tensor-factory occurrence",
            "total_factory_occurrences": total_factory_occurrences,
            "resolved_shape_occurrences": resolved_shape_occurrences,
            "positive_numel_occurrences": len(positive_numel),
            "resolved_shape_percent": _percent(resolved_shape_occurrences, total_factory_occurrences),
            "percentile_method": "nearest observation at round((n - 1) * p)",
            "percentiles": numel_percentiles,
            "tails": tails,
        },
        "operator_count": {
            "unit": "task",
            "method": "len(tools.data.cleaning.complexity.extract_operator_signature(reference))",
            "histogram": {str(count): value for count, value in sorted(operator_histogram.items())},
            "exclusive_plot_bins": op_bins,
            "bins_sum_to_rows": sum(item["count"] for item in op_bins) == rows,
        },
        "operator_family_row_presence": {
            "denominator": "all dataset rows",
            "multi_label": True,
            "families": {
                name: {
                    "count": operator_families[name],
                    "percent": _percent(operator_families[name], rows),
                }
                for name in (*CATEGORY_PATTERNS, "other_only")
            },
        },
    }


def pinned_correctness_profile() -> dict[str, Any]:
    levels: dict[str, Any] = {}
    for level, values in PINNED_CORRECTNESS.items():
        report = Path(values["report"]).resolve()
        if not report.is_file():
            raise FileNotFoundError(f"pinned correctness report is missing: {report}")
        observed_hash = _sha256_file(report)
        expected_hash = values["report_sha256"]
        if observed_hash != expected_hash:
            raise ProfileError(
                f"pinned correctness report hash changed for {level}: expected {expected_hash}, got {observed_hash}"
            )
        generated = int(values["generated"])
        compiled = int(values["compiled"])
        correct = int(values["correct"])
        if not 0 <= correct <= compiled <= generated:
            raise AssertionError(f"invalid pinned counts for {level}")
        levels[level] = {
            "generated": generated,
            "compiled": compiled,
            "correct": correct,
            "compile_percent": _percent(compiled, generated),
            "correct_percent": _percent(correct, generated),
            "provenance": {
                "path": _display_path(report),
                "sha256": observed_hash,
            },
        }
    return {
        "source_kind": "pinned_audited_integer_counts",
        "percentages_recomputed_from_counts": True,
        "raw_evaluation_artifact_in_data_handoff": False,
        "levels": levels,
    }


def build_profile(train_paths: Sequence[Path], kernelbench_paths: Sequence[Path]) -> dict[str, Any]:
    train = profile_dataset("Active train", train_paths)
    kernelbench = profile_dataset("KernelBench", kernelbench_paths)
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profiler": {
            "version": PROFILER_VERSION,
            "source_path": _display_path(Path(__file__).resolve()),
            "source_sha256": _sha256_file(Path(__file__).resolve()),
            "executes_reference_code": False,
            "operator_family_patterns": {name: list(patterns) for name, patterns in CATEGORY_PATTERNS.items()},
        },
        "measurement_contract": {
            "reference_code_field": "reward_model.ground_truth",
            "effective_get_inputs": "last top-level get_inputs definition",
            "input_factory_row_presence": {
                "unit": "dataset row",
                "denominator": "all parquet rows",
                "non_exclusive": True,
                "criterion": "factory call appears anywhere in the effective get_inputs AST",
            },
            "resolved_tensor_numel": {
                "unit": "tensor-factory occurrence",
                "execution": "static AST evaluation only; task source is never executed",
                "constant_scope": "safe module-level numeric assignments evaluated in source order",
                "function_local_assignments_resolved": False,
                "shape_source": "positional factory arguments only",
                "negative_dimensions_included": False,
                "zero_numel_in_percentiles": False,
                "percentile_method": "nearest observation at round((n - 1) * p)",
                "factories": sorted(FACTORY_BASES),
            },
            "operator_count": {
                "unit": "task",
                "method": "len(tools.data.cleaning.complexity.extract_operator_signature(reference))",
                "exclusive_plot_bins": f"integer counts 0..{OP_TAIL_START - 1}, then {OP_TAIL_START}+",
            },
            "operator_family_row_presence": {
                "unit": "dataset row",
                "denominator": "all parquet rows",
                "multi_label": True,
                "matching": "case-insensitive substring match over extracted operator tokens",
            },
        },
        "panel_a_correctness": pinned_correctness_profile(),
        "datasets": {
            "active_train": train,
            "kernelbench": kernelbench,
        },
    }


def write_profile(profile: Mapping[str, Any], output: Path, *, overwrite: bool) -> None:
    destination = output.expanduser().resolve()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output already exists; pass --overwrite: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(profile, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", type=Path, action="append", dest="train_paths")
    parser.add_argument("--kernelbench-parquet", type=Path, action="append", dest="kernelbench_paths")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train_paths = tuple(args.train_paths) if args.train_paths else (DEFAULT_TRAIN,)
    kernelbench_paths = tuple(args.kernelbench_paths) if args.kernelbench_paths else DEFAULT_KERNELBENCH
    profile = build_profile(train_paths, kernelbench_paths)
    write_profile(profile, args.output, overwrite=args.overwrite)
    print(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
