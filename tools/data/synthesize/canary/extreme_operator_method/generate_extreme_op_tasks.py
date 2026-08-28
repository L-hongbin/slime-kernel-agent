#!/usr/bin/env python3
"""Generate coverage-first atomic and long-DAG PyTorch task candidates.

This generator deliberately creates *new semantic tasks*.  It complements
``augment_prompt_tasks.py``, which derives shape/value children from existing
parents.  The design takes two ideas from 2026 work:

* DRTriton's CSP-DAG view: build a typed DAG, control its operator count, and
  solve tensor compatibility by construction.
* MusaCoder's staged coverage expansion: explicitly cover fundamental operator
  parameters before adding longer computation graphs.

The implementation is a small, auditable typed-DAG generator, not a re-release
of either system.  It emits candidates only; runtime validation is a separate
gate.  All shapes are statically bounded so generation cannot silently create
an unbounded tensor workload.
"""

from __future__ import annotations

import argparse
import ast
import dataclasses
import hashlib
import json
import os
import random
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from tools.data.cleaning.complexity import extract_operator_signature
from tools.data.cleaning.external import _drkernel_row, _template

GENERATOR_VERSION = "coverage_typed_dag_v3"
BUILD_SEED = 20260803
DEFAULT_MAX_INPUT_NUMEL = 40_000_000
METHOD_REFERENCES = (
    "https://arxiv.org/abs/2603.21465",  # DRTriton, CSP-DAG
    "https://arxiv.org/abs/2606.04847",  # MusaCoder, progressive task expansion
)


@dataclasses.dataclass(frozen=True)
class ShapeCell:
    name: str
    shape: tuple[int, ...]

    @property
    def numel(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result


@dataclasses.dataclass(frozen=True)
class AtomicOp:
    name: str
    expression: str
    ranks: frozenset[int]
    value_families: tuple[str, ...] = ("randn", "rand", "signed_uniform", "boundary_mix")


@dataclasses.dataclass(frozen=True)
class GeneratedTask:
    uuid: str
    code: str
    family: str
    config: dict[str, Any]
    operator_signature: tuple[str, ...]
    input_numel_upper_bound: int

    @property
    def reference_sha256(self) -> str:
        return hashlib.sha256(self.code.encode()).hexdigest()

    @property
    def normalized_ast_sha256(self) -> str:
        normalized = ast.dump(ast.parse(self.code), annotate_fields=True, include_attributes=False)
        return hashlib.sha256(normalized.encode()).hexdigest()


SHAPE_CELLS = (
    ShapeCell("small_vector_tail", (4093,)),
    ShapeCell("small_matrix_tail", (257, 509)),
    ShapeCell("small_rank3", (7, 33, 65)),
    ShapeCell("medium_matrix", (1024, 1025)),
    ShapeCell("medium_rank3", (16, 257, 513)),
    ShapeCell("large_matrix", (4096, 4097)),
    ShapeCell("large_rank3", (8, 1024, 4097)),
)


ATOMIC_OPS = (
    AtomicOp("relu", "torch.relu(x)", frozenset({1, 2, 3})),
    AtomicOp("sigmoid", "torch.sigmoid(x)", frozenset({1, 2, 3})),
    AtomicOp("tanh", "torch.tanh(x)", frozenset({1, 2, 3})),
    AtomicOp("silu", "torch.nn.functional.silu(x)", frozenset({1, 2, 3})),
    AtomicOp("gelu", "torch.nn.functional.gelu(x, approximate='tanh')", frozenset({1, 2, 3})),
    AtomicOp("softplus", "torch.nn.functional.softplus(x)", frozenset({1, 2, 3})),
    AtomicOp("sin", "torch.sin(x)", frozenset({1, 2, 3})),
    AtomicOp("cos", "torch.cos(x)", frozenset({1, 2, 3})),
    AtomicOp("erf", "torch.erf(x)", frozenset({1, 2, 3})),
    AtomicOp("abs", "torch.abs(x)", frozenset({1, 2, 3})),
    AtomicOp("square", "torch.square(x)", frozenset({1, 2, 3})),
    AtomicOp("neg", "torch.neg(x)", frozenset({1, 2, 3})),
    AtomicOp("exp", "torch.exp(x)", frozenset({1, 2, 3}), ("rand", "signed_uniform")),
    AtomicOp("log", "torch.log(x)", frozenset({1, 2, 3}), ("positive_uniform",)),
    AtomicOp("rsqrt", "torch.rsqrt(x)", frozenset({1, 2, 3}), ("positive_uniform",)),
    AtomicOp("clamp", "torch.clamp(x, min=-0.75, max=0.625)", frozenset({1, 2, 3})),
    AtomicOp("sum_last", "torch.sum(x, dim=-1)", frozenset({2, 3})),
    AtomicOp("mean_last", "torch.mean(x, dim=-1)", frozenset({2, 3})),
    AtomicOp("amax_last", "torch.amax(x, dim=-1)", frozenset({2, 3})),
    AtomicOp("var_last", "torch.var(x, dim=-1, correction=0)", frozenset({2, 3})),
    AtomicOp("softmax_last", "torch.softmax(x, dim=-1)", frozenset({2, 3})),
    AtomicOp("log_softmax_last", "torch.log_softmax(x, dim=-1)", frozenset({2, 3})),
    AtomicOp("cumsum_last", "torch.cumsum(x, dim=-1)", frozenset({2, 3})),
    AtomicOp("flip_last", "torch.flip(x, dims=(-1,))", frozenset({2, 3})),
    AtomicOp("transpose_tail", "torch.transpose(x, -1, -2)", frozenset({2, 3})),
    AtomicOp("triu", "torch.triu(x, diagonal=1)", frozenset({2})),
    AtomicOp("diagonal", "torch.diagonal(x, offset=1)", frozenset({2})),
    AtomicOp("layer_norm", "torch.nn.functional.layer_norm(x, (x.shape[-1],))", frozenset({2, 3})),
    AtomicOp("l2_normalize", "torch.nn.functional.normalize(x, p=2.0, dim=-1)", frozenset({2, 3})),
)


LONG_OPS = (
    "torch.sin({a})",
    "torch.cos({a})",
    "torch.tanh({a})",
    "torch.relu({a})",
    "torch.sigmoid({a})",
    "torch.neg({a})",
    "torch.abs({a})",
    "torch.square({a})",
    "torch.add({a}, {b})",
    "torch.mul({a}, {b})",
    "torch.maximum({a}, {b})",
    "torch.minimum({a}, {b})",
)

UNARY_ATOMIC_NAMES = frozenset(
    {
        "relu",
        "sigmoid",
        "tanh",
        "silu",
        "gelu",
        "softplus",
        "sin",
        "cos",
        "erf",
        "abs",
        "square",
        "neg",
        "exp",
        "log",
        "rsqrt",
        "clamp",
    }
)
REDUCTION_ATOMIC_NAMES = frozenset(
    {"sum_last", "mean_last", "amax_last", "var_last", "softmax_last", "log_softmax_last"}
)
LAYOUT_ATOMIC_NAMES = frozenset({"flip_last", "transpose_tail", "triu", "diagonal"})


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _factory_expression(shape: tuple[int, ...], family: str) -> str:
    shape_literal = repr(shape)
    if family == "randn":
        return f"torch.randn({shape_literal}, dtype=torch.float32)"
    if family == "rand":
        return f"torch.rand({shape_literal}, dtype=torch.float32)"
    if family == "signed_uniform":
        return f"2.0 * torch.rand({shape_literal}, dtype=torch.float32) - 1.0"
    if family == "positive_uniform":
        return f"0.01 + torch.rand({shape_literal}, dtype=torch.float32)"
    if family == "boundary_mix":
        numel = ShapeCell("temporary", shape).numel
        return f"((torch.arange({numel}, dtype=torch.float32) % 17) - 8).reshape({shape_literal}) / 8.0"
    raise ValueError(f"unknown value family: {family}")


def _task_uuid(config: dict[str, Any]) -> str:
    digest = hashlib.sha256(_stable_json(config).encode()).hexdigest()[:20]
    return f"synth26_{digest}"


def _atomic_family(name: str) -> str:
    if name in UNARY_ATOMIC_NAMES:
        return "pointwise_unary"
    if name in REDUCTION_ATOMIC_NAMES:
        return "reduction_or_softmax"
    if name in LAYOUT_ATOMIC_NAMES:
        return "layout_or_view"
    if name in {"layer_norm", "l2_normalize"}:
        return "normalization"
    if name == "cumsum_last":
        return "scan"
    raise ValueError(f"unclassified atomic operator: {name}")


def _checked_task(
    *,
    config: dict[str, Any],
    code: str,
    family: str,
    input_numel_upper_bound: int,
    expected_operator_count: int = 1,
) -> GeneratedTask:
    signature = extract_operator_signature(code)
    if len(signature) != expected_operator_count:
        raise AssertionError(
            f"template {config.get('operator', config.get('kind'))} produced {len(signature)} "
            f"compute tokens, expected {expected_operator_count}: {signature}"
        )
    return GeneratedTask(
        uuid=_task_uuid(config),
        code=code,
        family=family,
        config=config,
        operator_signature=signature,
        input_numel_upper_bound=input_numel_upper_bound,
    )


def _render_atomic(op: AtomicOp, shape_cell: ShapeCell, value_family: str) -> GeneratedTask:
    config = {
        "generator": GENERATOR_VERSION,
        "kind": "atomic",
        "operator": op.name,
        "operator_family": _atomic_family(op.name),
        "shape_cell": shape_cell.name,
        "shape": list(shape_cell.shape),
        "value_family": value_family,
    }
    code = f"""import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return {op.expression}


def get_inputs():
    return [{_factory_expression(shape_cell.shape, value_family)}]


def get_init_inputs():
    return []
"""
    return _checked_task(
        config=config,
        code=code,
        family="atomic_parameter_space",
        input_numel_upper_bound=shape_cell.numel,
    )


def _render_binary_atomic(
    *,
    operator: str,
    expression: str,
    shape_cell: ShapeCell,
    value_family: str,
) -> GeneratedTask:
    config = {
        "generator": GENERATOR_VERSION,
        "kind": "atomic",
        "operator": operator,
        "operator_family": "pointwise_binary",
        "shape_cell": shape_cell.name,
        "shape": list(shape_cell.shape),
        "value_family": value_family,
    }
    factory = _factory_expression(shape_cell.shape, value_family)
    code = f"""import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return {expression}


def get_inputs():
    return [{factory}, {factory}]


def get_init_inputs():
    return []
"""
    return _checked_task(
        config=config,
        code=code,
        family="atomic_parameter_space",
        input_numel_upper_bound=2 * shape_cell.numel,
    )


def _render_matmul_atomic(
    *,
    operator: str,
    left_shape: tuple[int, ...],
    right_shape: tuple[int, ...],
    shape_cell: str,
    value_family: str,
) -> GeneratedTask:
    expression = "torch.matmul(x, y)" if operator == "matmul" else "torch.bmm(x, y)"
    config = {
        "generator": GENERATOR_VERSION,
        "kind": "atomic",
        "operator": operator,
        "operator_family": "matmul",
        "shape_cell": shape_cell,
        "left_shape": list(left_shape),
        "right_shape": list(right_shape),
        "value_family": value_family,
    }
    code = f"""import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return {expression}


def get_inputs():
    return [{_factory_expression(left_shape, value_family)}, {_factory_expression(right_shape, value_family)}]


def get_init_inputs():
    return []
"""
    input_numel = ShapeCell("left", left_shape).numel + ShapeCell("right", right_shape).numel
    return _checked_task(
        config=config,
        code=code,
        family="atomic_parameter_space",
        input_numel_upper_bound=input_numel,
    )


def _render_module_atomic(
    *,
    operator: str,
    operator_family: str,
    constructor: str,
    input_shape: tuple[int, ...],
    shape_cell: str,
    value_family: str,
) -> GeneratedTask:
    config = {
        "generator": GENERATOR_VERSION,
        "kind": "atomic",
        "operator": operator,
        "operator_family": operator_family,
        "constructor": constructor,
        "shape_cell": shape_cell,
        "shape": list(input_shape),
        "value_family": value_family,
    }
    code = f"""import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.op = {constructor}

    def forward(self, x):
        return self.op(x)


def get_inputs():
    return [{_factory_expression(input_shape, value_family)}]


def get_init_inputs():
    return []
"""
    return _checked_task(
        config=config,
        code=code,
        family="atomic_parameter_space",
        input_numel_upper_bound=ShapeCell("input", input_shape).numel,
    )


def _render_index_atomic(*, operator: str, shape: tuple[int, int], value_family: str) -> GeneratedTask:
    rows, columns = shape
    if operator == "gather":
        expression = "torch.gather(x, 1, index)"
        inputs = (
            f"{_factory_expression(shape, value_family)}, "
            f"torch.randint(0, {columns}, {shape!r}, dtype=torch.int64)"
        )
        input_numel = rows * columns * 2
    elif operator == "index_select":
        expression = "torch.index_select(x, 1, index)"
        inputs = (
            f"{_factory_expression(shape, value_family)}, "
            f"torch.randint(0, {columns}, ({min(columns, 257)},), dtype=torch.int64)"
        )
        input_numel = rows * columns + min(columns, 257)
    else:
        raise ValueError(f"unknown index operator: {operator}")
    config = {
        "generator": GENERATOR_VERSION,
        "kind": "atomic",
        "operator": operator,
        "operator_family": "indexing",
        "shape_cell": "index_small" if rows * columns < 1_000_000 else "index_medium",
        "shape": list(shape),
        "value_family": value_family,
    }
    code = f"""import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, index):
        return {expression}


def get_inputs():
    return [{inputs}]


def get_init_inputs():
    return []
"""
    return _checked_task(
        config=config,
        code=code,
        family="atomic_parameter_space",
        input_numel_upper_bound=input_numel,
    )


def _special_atomic_pool(max_input_numel: int) -> list[GeneratedTask]:
    tasks: list[GeneratedTask] = []
    value_families = ("randn", "rand", "signed_uniform", "boundary_mix")
    binary_ops = {
        "add": "torch.add(x, y)",
        "sub": "torch.sub(x, y)",
        "mul": "torch.mul(x, y)",
        "maximum": "torch.maximum(x, y)",
        "minimum": "torch.minimum(x, y)",
    }
    for shape_cell in SHAPE_CELLS:
        if 2 * shape_cell.numel > max_input_numel:
            continue
        for value_family in value_families:
            for operator, expression in binary_ops.items():
                tasks.append(
                    _render_binary_atomic(
                        operator=operator,
                        expression=expression,
                        shape_cell=shape_cell,
                        value_family=value_family,
                    )
                )

    matrix_shapes = (
        ("matmul_small_tail", (257, 129), (129, 65)),
        ("matmul_medium", (1024, 1024), (1024, 1025)),
        ("matmul_large", (4096, 4096), (4096, 4097)),
    )
    batch_shapes = (
        ("bmm_small_tail", (7, 129, 65), (7, 65, 33)),
        ("bmm_medium", (16, 512, 257), (16, 257, 513)),
    )
    for operator, shapes in (("matmul", matrix_shapes), ("bmm", batch_shapes)):
        for shape_cell, left, right in shapes:
            input_numel = ShapeCell("left", left).numel + ShapeCell("right", right).numel
            if input_numel > max_input_numel:
                continue
            for value_family in value_families[:3]:
                tasks.append(
                    _render_matmul_atomic(
                        operator=operator,
                        left_shape=left,
                        right_shape=right,
                        shape_cell=shape_cell,
                        value_family=value_family,
                    )
                )

    module_specs = (
        ("conv1d", "convolution", "torch.nn.Conv1d(16, 24, 5, padding=4, dilation=2)", (8, 16, 4097), "conv1d_tail"),
        ("conv2d", "convolution", "torch.nn.Conv2d(16, 32, 3, padding=1)", (8, 16, 129, 131), "conv2d_small"),
        (
            "conv2d_grouped",
            "convolution",
            "torch.nn.Conv2d(32, 64, 3, padding=2, dilation=2, groups=8)",
            (16, 32, 128, 129),
            "conv2d_medium",
        ),
        (
            "conv2d_depthwise",
            "convolution",
            "torch.nn.Conv2d(64, 64, 5, padding=2, groups=64)",
            (8, 64, 256, 257),
            "conv2d_large",
        ),
        (
            "conv_transpose2d",
            "convolution",
            "torch.nn.ConvTranspose2d(16, 24, 4, stride=2, padding=1)",
            (8, 16, 65, 67),
            "conv_transpose_tail",
        ),
        ("conv3d", "convolution", "torch.nn.Conv3d(8, 12, 3, padding=1)", (4, 8, 33, 35, 37), "conv3d_tail"),
        ("batch_norm2d", "normalization", "torch.nn.BatchNorm2d(32)", (16, 32, 128, 129), "norm_medium"),
        ("group_norm", "normalization", "torch.nn.GroupNorm(8, 32)", (16, 32, 128, 129), "norm_medium"),
        (
            "instance_norm2d",
            "normalization",
            "torch.nn.InstanceNorm2d(32, affine=True)",
            (8, 32, 129, 131),
            "norm_tail",
        ),
        ("layer_norm_module", "normalization", "torch.nn.LayerNorm(1025)", (1024, 1025), "norm_matrix"),
        ("max_pool2d", "pooling", "torch.nn.MaxPool2d(3, stride=2, padding=1)", (16, 32, 257, 259), "pool_tail"),
        ("avg_pool2d", "pooling", "torch.nn.AvgPool2d(5, stride=3, padding=2)", (16, 32, 257, 259), "pool_tail"),
        ("adaptive_avg_pool2d", "pooling", "torch.nn.AdaptiveAvgPool2d((17, 19))", (8, 64, 129, 131), "pool_adaptive"),
    )
    for operator, family, constructor, shape, shape_cell in module_specs:
        if ShapeCell("input", shape).numel > max_input_numel:
            continue
        for value_family in value_families[:3]:
            tasks.append(
                _render_module_atomic(
                    operator=operator,
                    operator_family=family,
                    constructor=constructor,
                    input_shape=shape,
                    shape_cell=shape_cell,
                    value_family=value_family,
                )
            )

    for shape in ((257, 509), (2048, 2049)):
        if 2 * shape[0] * shape[1] > max_input_numel:
            continue
        for operator in ("gather", "index_select"):
            for value_family in value_families[:3]:
                tasks.append(_render_index_atomic(operator=operator, shape=shape, value_family=value_family))
    return tasks


def _atomic_pool(max_input_numel: int) -> list[GeneratedTask]:
    tasks = []
    for op in ATOMIC_OPS:
        for shape_cell in SHAPE_CELLS:
            if len(shape_cell.shape) not in op.ranks or shape_cell.numel > max_input_numel:
                continue
            for value_family in op.value_families:
                tasks.append(_render_atomic(op, shape_cell, value_family))
    tasks.extend(_special_atomic_pool(max_input_numel))
    return tasks


def _render_long_dag(
    *,
    operator_count: int,
    shape_cell: ShapeCell,
    value_family: str,
    topology: str,
    seed: int,
) -> GeneratedTask:
    if operator_count < 6:
        raise ValueError("long DAGs require at least six operators")
    rng = random.Random(seed)
    available = ["x"]
    branch_heads = ["x", "x"]
    statements = []
    chosen_ops = []
    for index in range(operator_count):
        if topology == "chain":
            template = LONG_OPS[rng.randrange(len(LONG_OPS))]
            a = available[-1]
            b = "x"
        elif topology == "residual":
            if index % 4 == 3:
                template = "torch.add({a}, {b})"
                a = available[-1]
                b = "x"
            else:
                template = LONG_OPS[rng.randrange(8)]
                a = available[-1]
                b = "x"
        elif topology == "branch_merge":
            if index == operator_count - 1:
                template = "torch.add({a}, {b})"
                a, b = branch_heads
            else:
                branch_index = index % 2
                template = LONG_OPS[rng.randrange(8)]
                a = branch_heads[branch_index]
                b = "x"
        elif topology == "random_dag":
            template = LONG_OPS[rng.randrange(len(LONG_OPS))]
            # Keep the newest node on the live path so every emitted operation
            # contributes to the returned value.  The second edge still
            # samples any earlier node, producing deterministic skip/merge
            # structure without dead branches.
            a = available[-1]
            b = available[rng.randrange(len(available))]
        else:
            raise ValueError(f"unknown topology: {topology}")
        expression = template.format(a=a, b=b)
        output = f"v{index}"
        statements.append(f"        {output} = {expression}")
        available.append(output)
        if topology == "branch_merge" and index < operator_count - 1:
            branch_heads[index % 2] = output
        chosen_ops.append(template.split("(", 1)[0].removeprefix("torch."))
    code = f"""import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
{os.linesep.join(statements)}
        return {available[-1]}


def get_inputs():
    return [{_factory_expression(shape_cell.shape, value_family)}]


def get_init_inputs():
    return []
"""
    config = {
        "generator": GENERATOR_VERSION,
        "kind": "typed_dag",
        "operator_count": operator_count,
        "operators": chosen_ops,
        "topology": topology,
        "shape_cell": shape_cell.name,
        "shape": list(shape_cell.shape),
        "value_family": value_family,
        "dag_seed": seed,
    }
    return _checked_task(
        config=config,
        code=code,
        family="typed_long_dag",
        input_numel_upper_bound=shape_cell.numel,
        expected_operator_count=operator_count,
    )


def _long_pool(max_input_numel: int, seed: int) -> list[GeneratedTask]:
    shape_cells = [cell for cell in SHAPE_CELLS if cell.numel <= min(max_input_numel, 4_500_000)]
    value_families = ("randn", "rand", "signed_uniform", "boundary_mix")
    topologies = ("chain", "residual", "branch_merge", "random_dag")
    tasks = []
    for operator_count in (6, 8, 10, 12, 16, 20):
        for shape_index, shape_cell in enumerate(shape_cells):
            for family_index, value_family in enumerate(value_families):
                for topology_index, topology in enumerate(topologies):
                    for replica in range(2):
                        dag_seed = (
                            seed
                            + operator_count * 1_000_000
                            + shape_index * 10_000
                            + family_index * 1_000
                            + topology_index * 100
                            + replica
                        )
                        tasks.append(
                            _render_long_dag(
                                operator_count=operator_count,
                                shape_cell=shape_cell,
                                value_family=value_family,
                                topology=topology,
                                seed=dag_seed,
                            )
                        )
    return tasks


def _coverage_tokens(task: GeneratedTask) -> frozenset[str]:
    config = task.config
    values = {
        f"kind={config.get('kind')}",
        f"family={config.get('operator_family', task.family)}",
        f"shape={config.get('shape_cell')}",
        f"value={config.get('value_family')}",
        f"op_count={len(task.operator_signature)}",
    }
    if config.get("operator") is not None:
        values.add(f"operator={config['operator']}")
    if config.get("topology") is not None:
        values.add(f"topology={config['topology']}")
    return frozenset(values)


def _select_coverage(tasks: Iterable[GeneratedTask], count: int, seed: int) -> list[GeneratedTask]:
    if count < 0:
        raise ValueError("count must be non-negative")
    ranked = sorted(tasks, key=lambda task: hashlib.sha256(f"{seed}:{task.uuid}".encode()).hexdigest())
    if count > len(ranked):
        raise ValueError(f"requested {count} tasks, but only {len(ranked)} unique candidates exist")
    selected: list[GeneratedTask] = []
    covered: set[str] = set()
    remaining = list(ranked)
    while remaining and len(selected) < count:
        best_index = max(
            range(len(remaining)),
            key=lambda index: (len(_coverage_tokens(remaining[index]) - covered), -index),
        )
        task = remaining.pop(best_index)
        selected.append(task)
        covered.update(_coverage_tokens(task))
    return selected


def generate_tasks(
    *,
    atomic_count: int,
    long_count: int,
    seed: int = BUILD_SEED,
    max_input_numel: int = DEFAULT_MAX_INPUT_NUMEL,
) -> list[GeneratedTask]:
    """Return a deterministic coverage sample with no code/AST duplicates."""

    if max_input_numel <= 0:
        raise ValueError("max_input_numel must be positive")
    atomic = _select_coverage(_atomic_pool(max_input_numel), atomic_count, seed)
    long = _select_coverage(_long_pool(max_input_numel, seed), long_count, seed + 1)
    tasks = atomic + long
    uuids = {task.uuid for task in tasks}
    references = {task.reference_sha256 for task in tasks}
    asts = {task.normalized_ast_sha256 for task in tasks}
    if len(uuids) != len(tasks) or len(references) != len(tasks) or len(asts) != len(tasks):
        raise AssertionError("generated tasks are not unique by UUID, source hash, and normalized AST")
    if any(task.input_numel_upper_bound > max_input_numel for task in tasks):
        raise AssertionError("generated task exceeded the static input-numel bound")
    return tasks


def _manifest_record(task: GeneratedTask, row_index: int, generator_source_sha256: str) -> dict[str, Any]:
    return {
        "row_index": row_index,
        "uuid": task.uuid,
        "task_kind": "semantic_synthetic",
        "semantic_family": task.family,
        "generator": GENERATOR_VERSION,
        "generator_source_sha256": generator_source_sha256,
        "generator_config": task.config,
        "method_references": list(METHOD_REFERENCES),
        "operator_count": len(task.operator_signature),
        "operator_signature": list(task.operator_signature),
        "input_numel_upper_bound": task.input_numel_upper_bound,
        "static_input_bytes_upper_bound_fp32": task.input_numel_upper_bound * 4,
        "reference_sha256": task.reference_sha256,
        "normalized_ast_sha256": task.normalized_ast_sha256,
        "runtime_validation_status": "pending_isolated_gpu_validation",
        "license": "project_generated_internal_review",
    }


def write_dataset(
    tasks: Sequence[GeneratedTask],
    *,
    template_parquet: Path,
    output_parquet: Path,
    manifest_jsonl: Path,
    summary_json: Path,
    overwrite: bool,
) -> dict[str, Any]:
    for path in (output_parquet, manifest_jsonl, summary_json):
        if path.exists() and not overwrite:
            raise FileExistsError(f"output exists (pass --overwrite): {path}")
    schema, prompt_prefix = _template(template_parquet)
    rows = [
        _drkernel_row(
            code=task.code,
            prompt_prefix=prompt_prefix,
            data_source="prompt_tvm_v4_synth26",
            level="coverage",
            repo_name=GENERATOR_VERSION,
            task_type="semantic_synthetic",
            uuid=task.uuid,
        )
        for task in tasks
    ]
    generator_source_sha256 = _sha256_file(Path(__file__).resolve())
    records = [_manifest_record(task, index, generator_source_sha256) for index, task in enumerate(tasks)]
    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    manifest_jsonl.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    parquet_tmp = output_parquet.with_name(f".{output_parquet.name}.tmp")
    manifest_tmp = manifest_jsonl.with_name(f".{manifest_jsonl.name}.tmp")
    summary_tmp = summary_json.with_name(f".{summary_json.name}.tmp")
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), parquet_tmp, compression="zstd", use_dictionary=True)
    with manifest_tmp.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    counts: dict[str, int] = {}
    for task in tasks:
        key = "atomic" if len(task.operator_signature) <= 1 else "long"
        counts[key] = counts.get(key, 0) + 1
        if len(task.operator_signature) >= 10:
            counts["op_count_ge_10"] = counts.get("op_count_ge_10", 0) + 1
    summary = {
        "generator": GENERATOR_VERSION,
        "generator_source_sha256": generator_source_sha256,
        "rows": len(tasks),
        "counts": counts,
        "max_input_numel_observed": max((task.input_numel_upper_bound for task in tasks), default=0),
        "method_references": list(METHOD_REFERENCES),
        "template_parquet": str(template_parquet.resolve()),
        "template_sha256": _sha256_file(template_parquet),
        "runtime_validation_status": "pending_isolated_gpu_validation",
    }
    summary_tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(parquet_tmp, output_parquet)
    os.replace(manifest_tmp, manifest_jsonl)
    os.replace(summary_tmp, summary_json)
    return summary


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--template-parquet",
        type=Path,
        default=Path("Data/prompt_tvm_v3/drkernel_rl_thinking.parquet"),
    )
    parser.add_argument(
        "--output-parquet",
        type=Path,
        default=Path("Data/prompt_tvm_v4/synthesis/candidates/extreme_ops.parquet"),
    )
    parser.add_argument(
        "--manifest-jsonl",
        type=Path,
        default=Path("Data/prompt_tvm_v4/synthesis/candidates/extreme_ops.manifest.jsonl"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("Data/prompt_tvm_v4/synthesis/candidates/extreme_ops.summary.json"),
    )
    parser.add_argument("--atomic-count", type=int, default=256)
    parser.add_argument("--long-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=BUILD_SEED)
    parser.add_argument("--max-input-numel", type=int, default=DEFAULT_MAX_INPUT_NUMEL)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    tasks = generate_tasks(
        atomic_count=args.atomic_count,
        long_count=args.long_count,
        seed=args.seed,
        max_input_numel=args.max_input_numel,
    )
    summary = write_dataset(
        tasks,
        template_parquet=args.template_parquet,
        output_parquet=args.output_parquet,
        manifest_jsonl=args.manifest_jsonl,
        summary_json=args.summary_json,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
