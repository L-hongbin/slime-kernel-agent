#!/usr/bin/env python3
"""Profile static distributions from parquet reference code without executing it."""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.cleaning.complexity import extract_operator_signature

PROFILE_SCHEMA_VERSION = "prompt-tvm-distribution-profile-v2"
PROFILER_VERSION = "prompt-tvm-distribution-profiler-v3"
REPO_ROOT = _REPO_ROOT
DEFAULT_OUTPUT = REPO_ROOT / "Data/prompt_tvm_v4/analysis/distribution_profile.json"
DEFAULT_TRAIN = REPO_ROOT / "Data/prompt_tvm_v3/drkernel_rl_thinking.parquet"
DEFAULT_KERNELBENCH = tuple(
    REPO_ROOT / f"Data/kernelbench-level{level}-validation-tvm-v2/train.parquet" for level in (1, 2, 3)
)
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
        "zeros",
    }
)
FACTORY_CALLEES = frozenset(f"torch.{base}" for base in FACTORY_BASES)
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


class ProfileError(ValueError):
    """An input record cannot be profiled."""


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
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        try:
            value = _safe_value(statement.value, environment)
        except (ArithmeticError, OverflowError, ValueError):
            for name in names:
                environment.pop(name, None)
            continue
        for name in names:
            environment[name] = value
    return environment


def _factory_calls_with_environments(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    module_environment: Mapping[str, Any],
) -> list[tuple[ast.Call, dict[str, Any]]]:
    """Bind each call to the straight-line environment visible at that call."""
    environment = dict(module_environment)
    records: list[tuple[ast.Call, dict[str, Any]]] = []
    for statement in function.body:
        calls = sorted(
            (node for node in ast.walk(statement) if isinstance(node, ast.Call)),
            key=lambda node: (getattr(node, "lineno", -1), getattr(node, "col_offset", -1)),
        )
        records.extend((call, dict(environment)) for call in calls)

        if isinstance(statement, ast.Assign):
            targets = statement.targets
            value_node = statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
            value_node = statement.value
        elif isinstance(statement, ast.AugAssign):
            targets = [statement.target]
            value_node = None
        else:
            continue
        names = [target.id for target in targets if isinstance(target, ast.Name)]
        try:
            if value_node is None:
                raise ValueError("assignment has no statically safe value")
            value = _safe_value(value_node, environment)
        except (ArithmeticError, OverflowError, ValueError):
            for name in names:
                environment.pop(name, None)
            continue
        for name in names:
            environment[name] = value
    return records


def _resolved_sequence(
    node: ast.AST,
    environment: Mapping[str, Any],
    *,
    allow_scalar: bool,
) -> tuple[int, ...] | None:
    try:
        value = _safe_value(node, environment)
    except (ArithmeticError, OverflowError, ValueError):
        return None
    if isinstance(value, int) and not isinstance(value, bool) and allow_scalar:
        value = (value,)
    if not isinstance(value, tuple):
        return None
    dimensions = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            return None
        dimensions.append(item)
    return tuple(dimensions)


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    return next((keyword.value for keyword in call.keywords if keyword.arg == name), None)


def _resolved_shape(
    node: ast.AST | None,
    environment: Mapping[str, Any],
    *,
    allow_scalar: bool = False,
) -> tuple[int, ...] | None:
    if node is None:
        return None
    shape = _resolved_sequence(node, environment, allow_scalar=allow_scalar)
    return shape if shape is not None and all(dimension >= 0 for dimension in shape) else None


def _shape_arguments(
    arguments: Sequence[ast.AST],
    environment: Mapping[str, Any],
) -> tuple[int, ...] | None:
    if len(arguments) == 1:
        argument = arguments[0]
        return _resolved_shape(
            argument.value if isinstance(argument, ast.Starred) else argument,
            environment,
            allow_scalar=not isinstance(argument, ast.Starred),
        )
    dimensions: list[int] = []
    for argument in arguments:
        if isinstance(argument, ast.Starred):
            expanded = _resolved_shape(argument.value, environment)
            if expanded is None:
                return None
            dimensions.extend(expanded)
            continue
        try:
            value = _safe_value(argument, environment)
        except (ArithmeticError, OverflowError, ValueError):
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        dimensions.append(value)
    return tuple(dimensions) if dimensions else None


def _factory_shape(call: ast.Call, base: str, environment: Mapping[str, Any]) -> tuple[int, ...] | None:
    """Conservatively extract an explicit factory output shape."""

    size = _keyword(call, "size")
    if base in {"rand", "randn", "zeros", "ones", "empty"}:
        return _resolved_shape(size, environment) if size is not None else _shape_arguments(call.args, environment)
    if base == "full":
        node = size if size is not None else (call.args[0] if call.args else None)
        return _resolved_shape(node, environment)
    if base == "normal":
        node = size if size is not None else (call.args[2] if len(call.args) >= 3 else None)
        return _resolved_shape(node, environment)
    if base == "randint":
        node = (
            size
            if size is not None
            else (call.args[2] if len(call.args) >= 3 else call.args[1] if len(call.args) >= 2 else None)
        )
        return _resolved_shape(node, environment)
    if base in {"randperm", "arange"}:
        keyword_name = "n" if base == "randperm" else "end"
        node = _keyword(call, keyword_name) or (call.args[0] if len(call.args) == 1 else None)
        return _resolved_shape(node, environment, allow_scalar=True)
    if base == "linspace":
        node = _keyword(call, "steps") or (call.args[2] if len(call.args) >= 3 else None)
        return _resolved_shape(node, environment, allow_scalar=True)
    return None


def _exact_factory_observations(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    module_environment: Mapping[str, Any],
) -> list[tuple[str, tuple[int, ...] | None, str | None]]:
    observations = []
    for call, environment in _factory_calls_with_environments(function, module_environment):
        callee = _call_name(call.func)
        if callee not in FACTORY_CALLEES:
            continue
        base = callee.removeprefix("torch.")
        shape = _factory_shape(call, base, environment)
        observations.append((base, shape, None if shape is not None else "unresolved"))
    return observations


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


def _nearest_observation(values: Sequence[int | float], percentile: int) -> int | float:
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
    unresolved_by_factory: collections.Counter[str] = collections.Counter()
    positive_numel: list[int] = []
    positive_shapes: list[tuple[int, ...]] = []
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
        observations = _exact_factory_observations(get_inputs, _module_environment(tree))
        present_bases = {base for base, _shape, _reason in observations}
        factory_presence.update(base for base in ROW_PRESENCE_FACTORIES if base in present_bases)

        for base, shape, _reason in observations:
            total_factory_occurrences += 1
            if shape is None:
                unresolved_by_factory[base] += 1
                continue
            resolved_shape_occurrences += 1
            numel = math.prod(shape)
            if numel > 0:
                positive_numel.append(numel)
                positive_shapes.append(shape)

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
    rank_histogram = collections.Counter(len(shape) for shape in positive_shapes)
    axis_ratios = sorted(max(shape) / min(shape) for shape in positive_shapes if shape and min(shape) > 0)
    axis_ratio = {
        "count": len(axis_ratios),
        "p99": _nearest_observation(axis_ratios, 99) if axis_ratios else None,
        "gt_1e3_percent": _percent(sum(value > 1_000 for value in axis_ratios), len(axis_ratios)),
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
            "unresolved_shape_occurrences": total_factory_occurrences - resolved_shape_occurrences,
            "unresolved_by_factory": dict(sorted(unresolved_by_factory.items())),
            "resolved_shape_percent": _percent(resolved_shape_occurrences, total_factory_occurrences),
            "percentile_method": "nearest observation at round((n - 1) * p)",
            "percentiles": numel_percentiles,
            "tails": tails,
            "rank_histogram": {str(rank): count for rank, count in sorted(rank_histogram.items())},
            "axis_ratio": axis_ratio,
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
                "constant_scope": "safe module-level final bindings plus straight-line get_inputs numeric bindings captured at each call",
                "function_local_assignments_resolved": True,
                "shape_source": "factory-specific positional or explicit size/steps/n keyword grammar",
                "callee_policy": "exact torch factory names only; NumPy and leaf-name matches are excluded",
                "unresolved_calls": "counted as factory occurrences and excluded from shape statistics",
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
    destination.write_text(
        json.dumps(profile, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _self_check() -> None:
    def observe(expression: str, environment: Mapping[str, Any] | None = None) -> tuple[int, ...] | None:
        node = ast.parse(expression, mode="eval").body
        assert isinstance(node, ast.Call)
        callee = _call_name(node.func)
        if callee not in FACTORY_CALLEES:
            return None
        return _factory_shape(node, callee.removeprefix("torch."), environment or {})

    cases = {
        "torch.rand(2, 3)": (2, 3),
        "torch.rand(size=(2, 3))": (2, 3),
        "torch.rand(*(2, 3))": (2, 3),
        "torch.full((2, 3), 1.0)": (2, 3),
        "torch.randint(10, (2, 3))": (2, 3),
        "torch.randint(1, 10, (2, 3))": (2, 3),
        "torch.arange(5)": (5,),
        "np.random.rand(2, 3)": None,
        "torch.rand(dynamic_shape)": None,
    }
    assert all(observe(expression) == expected for expression, expected in cases.items())

    tree = ast.parse(
        """import torch
D = 4
def get_inputs():
    shape = (D, 2)
    first = torch.rand(shape)
    shape = (5, 3)
    second = torch.rand(shape)
    shape = dynamic_shape()
    third = torch.rand(shape)
    ignored = np.random.rand(7)
    return [first, second, third, ignored]
"""
    )
    function = _last_top_level_get_inputs(tree)
    assert function is not None
    observations = _exact_factory_observations(function, _module_environment(tree))
    assert observations == [
        ("rand", (4, 2), None),
        ("rand", (5, 3), None),
        ("rand", None, "unresolved"),
    ]
    print("prompt_tvm_distribution_profiler_self_check=passed")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-parquet", type=Path, action="append", dest="train_paths")
    parser.add_argument("--kernelbench-parquet", type=Path, action="append", dest="kernelbench_paths")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.self_check:
        _self_check()
        return
    train_paths = tuple(args.train_paths) if args.train_paths else (DEFAULT_TRAIN,)
    kernelbench_paths = tuple(args.kernelbench_paths) if args.kernelbench_paths else DEFAULT_KERNELBENCH
    profile = build_profile(train_paths, kernelbench_paths)
    write_profile(profile, args.output, overwrite=args.overwrite)
    print(args.output.expanduser().resolve())


if __name__ == "__main__":
    main()
