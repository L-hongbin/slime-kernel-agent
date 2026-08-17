#!/usr/bin/env python3
"""Generate review-only open CSP-DAG operator programs.

The generator has a finite catalog of low-level operator rules, but no graph
template registry.  Each row samples a family palette, a symbolic input shape,
an operator sequence, arbitrary earlier-value edges, concrete rule variants,
and a source-code lowering independently.  KernelBench contributes only a
low-level coverage checklist and coarse form buckets; no KernelBench graph,
source, or target proportions are used.
"""

from __future__ import annotations

import argparse
import ast
import collections
import dataclasses
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import networkx as nx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
import z3

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.cleaning.complexity import (
    extract_complexity_features,
    extract_operator_signature,
    feature_dict,
)
from tools.data.cleaning.external import _drkernel_row, _template
from tools.data.synthesize.select_low_level_kernelbench_canary import _token_cells


# These contract values are serialized in the released manifests.  Their
# historical names remain stable even though this module is the corpus
# generator rather than a canary-only entry point.
CONTRACT = "open_csp_dag_low_level_canary_v4"
RULE_CATALOG_VERSION = "low_level_operator_rules_v4"
# Preserve the approved v3 graph/shape/input sampling stream.  v4 changes only
# source-form policy by removing the identity multi-class wrapper.
SAMPLING_CONTRACT = "open_csp_dag_low_level_canary_v3"
SEED = 20260813
ROWS = 200
CANDIDATES = 3200
DATA_SOURCE = "project_generated_open_csp_dag_canary_v4"
_DEFAULT_TEMPLATE = _REPO_ROOT / "Data/prompt_tvm_v4/train.review.parquet"
_DEFAULT_OUTPUT = (
    _REPO_ROOT
    / "local_artifacts/data/synthesize/csp_dag_low_level_canary200/run.coverage_high_complexity.v3"
)

FAMILIES = (
    "activation",
    "conv",
    "indexing_scatter",
    "loss_distance",
    "matmul_linear",
    "normalization",
    "pooling",
    "reduction",
    "shape_layout",
    "scaled_dot_product_attention",
)

# KernelBench defines which low-level families and form buckets must be present,
# not their selected proportions.  These are coverage floors; the objective
# deliberately prefers long, many-node, many-operator candidates.
FAMILY_MINIMUMS = {
    "activation": 30,
    "conv": 30,
    "indexing_scatter": 5,
    "loss_distance": 5,
    "matmul_linear": 30,
    "normalization": 25,
    "pooling": 20,
    "reduction": 30,
    "shape_layout": 30,
    "scaled_dot_product_attention": 3,
}
LINE_BUCKET_MINIMUMS = {"20-34": 5, "35-49": 10, "50-74": 25, ">=75": 110}
OPCOUNT_BUCKET_MINIMUMS = {"1": 3, "2-4": 5, "5-9": 10, "10-15": 30, ">=16": 100}
COMPLEXITY_MINIMUMS = {
    "source_lines_ge_50": 140,
    "operator_count_ge_10": 170,
    "graph_nodes_ge_16": 100,
    "nonlocal_edge_rows": 120,
}
COMPARISON_ROOTS = (
    _REPO_ROOT / "Data/prompt_tvm_v3/drkernel_rl_thinking.parquet",
    _REPO_ROOT / "Data/prompt_tvm_v4/train.review.parquet",
    *tuple(
        _REPO_ROOT / f"Data/kernelbench-level{level}-validation-tvm-v2/train.parquet"
        for level in (1, 2, 3)
    ),
)


@dataclasses.dataclass(frozen=True)
class Rule:
    name: str
    family: str | None
    ranks: tuple[int, ...]
    variants: tuple[str, ...]
    module: bool = False


@dataclasses.dataclass(frozen=True)
class Node:
    index: int
    rule: str
    family: str | None
    variant: str
    predecessors: tuple[int, ...]
    module_name: str | None


@dataclasses.dataclass(frozen=True)
class Candidate:
    index: int
    uuid: str
    code: str
    manifest: dict[str, Any]
    row: dict[str, Any]
    families: frozenset[str]
    line_bucket: str
    op_bucket: str
    tie_key: str


RULES = (
    Rule("activation", "activation", (2, 3, 4, 5), ("relu", "gelu", "silu", "sigmoid", "tanh")),
    Rule("conv", "conv", (3, 4, 5), ("conv", "conv_transpose"), True),
    Rule("indexing", "indexing_scatter", (2, 3, 4, 5), ("gather", "scatter")),
    Rule("loss", "loss_distance", (2, 3, 4, 5), ("smooth_l1", "mse", "l1")),
    Rule("linear", "matmul_linear", (2, 3, 4, 5), ("linear",), True),
    Rule("normalization", "normalization", (2, 3, 4, 5), ("layer_norm", "group_norm"), True),
    Rule("pool", "pooling", (3, 4, 5), ("avg", "max")),
    Rule("reduction", "reduction", (2, 3, 4, 5), ("mean_center", "sum_scale", "amax_gate")),
    Rule("layout", "shape_layout", (2, 3, 4, 5), ("transpose", "chunk_cat", "binary_cat", "flip")),
    Rule("sdpa", "scaled_dot_product_attention", (4,), ("sdpa",)),
    Rule(
        "elementwise",
        None,
        (2, 3, 4, 5),
        ("sin", "cos", "square", "abs", "neg", "affine", "add", "mul", "mix"),
    ),
)
RULE_BY_NAME = {rule.name: rule for rule in RULES}
FAMILY_TO_RULE = {rule.family: rule.name for rule in RULES if rule.family is not None}
_NONPOINTWISE_RULES = frozenset(
    {"conv", "indexing", "linear", "normalization", "pool", "reduction", "layout", "sdpa"}
)
HIGH_COMPLEXITY_NODE_THRESHOLD = 16
HIGH_COMPLEXITY_MIN_NONPOINTWISE_FRACTION = 0.15
SINK_MERGE_NODE_RATIO_CAP = 0.40


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _normalized_ast_hash(code: str) -> str:
    return _sha256_text(ast.dump(ast.parse(code), annotate_fields=True, include_attributes=False))


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _line_bucket(lines: int) -> str:
    """Return the explicit code-only source-volume bucket.

    ``<20`` is intentionally separate.  A compact atomic operator is useful
    for low operator-count coverage, but it must not be presented as a 20--34
    line program merely because that was the smallest historical bucket.
    """

    if lines < 20:
        return "<20"
    if lines <= 34:
        return "20-34"
    if lines <= 49:
        return "35-49"
    if lines <= 74:
        return "50-74"
    return ">=75"


def _code_only_physical_line_count(code: str) -> int:
    """Count nonblank physical lines that contain Python code.

    The generated source deliberately contains no comment-only padding.  The
    definition still treats an inline comment as part of its code line, which
    is the usual physical-source convention and remains robust if a future
    lowering needs one for a real implementation detail.
    """

    return sum(
        bool(line.strip()) and not line.lstrip().startswith("#")
        for line in code.splitlines()
    )


def _op_bucket(count: int) -> str:
    if count == 1:
        return "1"
    if count <= 4:
        return "2-4"
    if count <= 9:
        return "5-9"
    if count <= 15:
        return "10-15"
    return ">=16"


def _rank_for_palette(rng: random.Random, palette: set[str]) -> int:
    if "scaled_dot_product_attention" in palette:
        return 4
    if palette & {"conv", "pooling"}:
        return rng.choice((3, 4, 5))
    return rng.choice((2, 3, 4))


def _symbolic_shape(rng: random.Random, rank: int) -> tuple[tuple[int, ...], dict[str, Any]]:
    solver = z3.Solver()
    dims = [z3.Int(f"d{i}") for i in range(rank)]
    solver.add(dims[0] >= 1, dims[0] <= 4)
    for dim in dims[1:]:
        solver.add(dim >= 4, dim <= 24, dim % 4 == 0)
    # Random cross-axis equalities are constraints, not selected concrete shapes.
    equalities: list[list[int]] = []
    if rank >= 3 and rng.random() < 0.45:
        left, right = rng.sample(range(1, rank), 2)
        solver.add(dims[left] == dims[right])
        equalities.append([left, right])
    solver.add(dims[0] == rng.choice((1, 2, 3, 4)))
    for dim in dims[1:]:
        solver.add(dim == rng.choice((4, 8, 12, 16, 20, 24)))
    if solver.check() != z3.sat:
        # Equalities can conflict with independently sampled witnesses.  The
        # retry is still a CSP solve and keeps the sampled relation.
        return _symbolic_shape(rng, rank)
    model = solver.model()
    concrete = tuple(model.eval(dim).as_long() for dim in dims)
    return concrete, {
        "variables": [str(dim) for dim in dims],
        "bounds": {str(dim): [1, 4] if i == 0 else [4, 24] for i, dim in enumerate(dims)},
        "divisibility": {str(dim): 4 for dim in dims[1:]},
        "equalities": equalities,
        "status": "sat",
        "witness": list(concrete),
    }


def _prove_graph_shapes(
    nodes: Sequence[Node],
    input_shape: tuple[int, ...],
    input_proof: dict[str, Any],
) -> dict[str, Any]:
    """Solve the joint edge-shape CSP for the sampled graph.

    The v1 rule catalog is deliberately shape preserving (the loss terminal is
    scalar).  This still matters at sampled fan-in nodes: every predecessor
    must unify on every axis before lowering is allowed.
    """

    solver = z3.Solver()
    rank = len(input_shape)
    input_dims = [z3.Int(f"input_d{axis}") for axis in range(rank)]
    for variable, value in zip(input_dims, input_shape, strict=True):
        solver.add(variable == value)
    outputs: dict[int, list[z3.IntNumRef | z3.ArithRef]] = {-1: input_dims}
    clauses: list[str] = []
    for node in nodes:
        if node.rule == "loss":
            for predecessor in node.predecessors:
                for axis in range(rank):
                    solver.add(outputs[predecessor][axis] == input_dims[axis])
                    clauses.append(f"loss operand {predecessor} axis {axis} matches target axis {axis}")
            outputs[node.index] = []
            continue
        current = [z3.Int(f"n{node.index}_d{axis}") for axis in range(rank)]
        main = outputs[node.predecessors[0]]
        for axis in range(rank):
            solver.add(current[axis] == main[axis])
            clauses.append(f"node {node.index} output axis {axis} equals main-input axis {axis}")
        for predecessor in node.predecessors[1:]:
            for axis in range(rank):
                solver.add(outputs[predecessor][axis] == main[axis])
                clauses.append(f"fanin node {node.index} predecessor {predecessor} unifies axis {axis}")
        outputs[node.index] = current
    status = solver.check()
    if status != z3.sat:
        raise AssertionError(f"sampled graph shape CSP is {status}")
    model = solver.model()
    node_witnesses = {
        str(index): [model.eval(variable).as_long() for variable in variables]
        for index, variables in outputs.items()
    }
    return {
        **input_proof,
        "joint_graph_status": "sat",
        "joint_graph_clause_count": len(clauses),
        "joint_graph_clauses": clauses,
        "node_output_witnesses": node_witnesses,
        "loss_terminal_rank": 0 if nodes[-1].rule == "loss" else None,
    }


def _sample_palette(rng: random.Random, index: int) -> set[str]:
    # Most rows request three to six distinct tracked families.  A separate
    # sparse reservation controls only compact node budgets; it does not
    # prescribe their families or graph topology.
    phase = index % 20
    if phase < 2:
        size = 1
    elif phase < 5:
        size = 2
    else:
        size = rng.choices((3, 4, 5, 6), weights=(2, 4, 3, 1), k=1)[0]
    palette = set(rng.sample(FAMILIES[:-1], k=size))
    # Oversupply rare cells in the reservoir; selection only imposes floors.
    if index % 19 == 0:
        palette.add("indexing_scatter")
    if index % 17 == 0:
        palette.add("loss_distance")
    if index % 41 == 0:
        palette.add("scaled_dot_product_attention")
    return palette


def _node_count_for_target(rng: random.Random, index: int) -> int:
    buckets = ("1", "2-4", "5-9", "10-15", ">=16")
    bucket = buckets[index % len(buckets)]
    ranges = {"1": (1, 1), "2-4": (2, 4), "5-9": (5, 9), "10-15": (10, 15), ">=16": (16, 24)}
    low, high = ranges[bucket]
    return rng.randint(low, high)


def _variant_arity(rule: str, variant: str) -> int:
    if rule == "sdpa" or (rule == "elementwise" and variant == "mix"):
        return 3
    if (rule == "layout" and variant == "binary_cat") or (
        rule == "elementwise" and variant in {"add", "mul"}
    ):
        return 2
    return 1


def _sample_predecessors(rng: random.Random, node_index: int, arity: int) -> tuple[int, ...]:
    available = [-1, *range(node_index)]
    if len(available) >= arity:
        return tuple(rng.sample(available, k=arity))
    # Only an early ternary rule can reach this branch.  Reusing x is legal for
    # SDPA and keeps the operator's true three-operand call explicit.
    return tuple(available[index % len(available)] for index in range(arity))


def _build_nodes(
    rng: random.Random,
    palette: set[str],
    rank: int,
    requested_nodes: int,
) -> tuple[list[Node], list[tuple[int, int]], int]:
    required = sorted(palette - {"loss_distance"})
    has_loss = "loss_distance" in palette
    target = max(requested_nodes - int(has_loss), len(required), 1)
    names = [FAMILY_TO_RULE[family] for family in required]
    family_fill_probability = 0.25 if palette <= {"reduction", "shape_layout"} else 0.76
    while len(names) < target:
        admissible = [
            FAMILY_TO_RULE[family]
            for family in sorted(palette - {"loss_distance"})
            if rank in RULE_BY_NAME[FAMILY_TO_RULE[family]].ranks
        ]
        names.append(
            rng.choice(admissible)
            if admissible and rng.random() < family_fill_probability
            else "elementwise"
        )
    rng.shuffle(names)

    nodes: list[Node] = []
    edges: list[tuple[int, int]] = []
    for name in names:
        index = len(nodes)
        rule = RULE_BY_NAME[name]
        if rank not in rule.ranks:
            name = "elementwise"
            rule = RULE_BY_NAME[name]
        variants = (
            ("layer_norm",)
            if name == "normalization" and rank < 3
            else rule.variants
        )
        variant = rng.choice(variants)
        predecessors = _sample_predecessors(rng, index, _variant_arity(name, variant))
        module_name = f"op_{index:03d}" if rule.module else None
        nodes.append(Node(index, name, rule.family, variant, predecessors, module_name))
        for predecessor in predecessors:
            edges.append((predecessor, index))

    # Arbitrary predecessor sampling can leave several output branches.  Merge
    # the sinks with genuine binary/ternary elementwise operators so every
    # sampled node reaches the single returned Tensor without forcing an i-1
    # backbone during graph growth.
    consumed = {predecessor for predecessor, _ in edges if predecessor >= 0}
    sinks = [node.index for node in nodes if node.index not in consumed]
    rng.shuffle(sinks)
    while len(sinks) > 1:
        arity = 3 if len(sinks) >= 3 and rng.random() < 0.35 else 2
        predecessors = tuple(sinks.pop() for _ in range(arity))
        index = len(nodes)
        variant = "mix" if arity == 3 else "add"
        nodes.append(Node(index, "elementwise", None, variant, predecessors, None))
        edges.extend((predecessor, index) for predecessor in predecessors)
        sinks.append(index)

    if has_loss:
        index = len(nodes)
        predecessors = (sinks[0],)
        variant = rng.choice(RULE_BY_NAME["loss"].variants)
        nodes.append(Node(index, "loss", "loss_distance", variant, predecessors, None))
        edges.append((sinks[0], index))
    return nodes, edges, len(nodes) - target - int(has_loss)


def _module_line(node: Node, shape: tuple[int, ...]) -> str | None:
    if node.rule == "conv":
        spatial_rank = len(shape) - 2
        cls = f"ConvTranspose{spatial_rank}d" if node.variant == "conv_transpose" else f"Conv{spatial_rank}d"
        channels = shape[1]
        return f"self.{node.module_name} = nn.{cls}({channels}, {channels}, 3, padding=1, bias=False)"
    if node.rule == "linear":
        width = shape[-1]
        return f"self.{node.module_name} = nn.Linear({width}, {width}, bias=False)"
    if node.rule == "normalization":
        if node.variant == "group_norm":
            if len(shape) < 3:
                raise AssertionError("group_norm requires ndim >= 3")
            return f"self.{node.module_name} = nn.GroupNorm(1, {shape[1]})"
        if node.variant == "layer_norm":
            return f"self.{node.module_name} = nn.LayerNorm(({shape[-1]},))"
        raise AssertionError(f"unsupported normalization variant: {node.variant}")
    return None


def _operation_expression(node: Node, sources: Sequence[str], rank: int) -> str:
    source = sources[0]
    if node.rule == "activation":
        return {
            "relu": f"torch.relu({source})",
            "gelu": f"F.gelu({source})",
            "silu": f"F.silu({source})",
            "sigmoid": f"torch.sigmoid({source})",
            "tanh": f"torch.tanh({source})",
        }[node.variant]
    if node.rule in {"conv", "linear", "normalization"}:
        return f"self.{node.module_name}({source})"
    if node.rule == "pool":
        spatial_rank = rank - 2
        function = "avg_pool" if node.variant == "avg" else "max_pool"
        return f"F.{function}{spatial_rank}d({source}, kernel_size=3, stride=1, padding=1)"
    if node.rule == "reduction":
        if node.variant == "mean_center":
            return f"{source} - {source}.mean(dim=-1, keepdim=True)"
        if node.variant == "sum_scale":
            return f"{source} + {source}.sum(dim=-1, keepdim=True) / {source}.shape[-1]"
        return f"{source} / ({source}.abs().amax(dim=-1, keepdim=True) + 1.0)"
    if node.rule == "layout":
        if node.variant == "transpose":
            return f"{source}.transpose(-1, -2).contiguous().transpose(-1, -2)"
        if node.variant == "chunk_cat":
            return f"torch.cat(torch.chunk({source}, 2, dim=-1), dim=-1)"
        if node.variant == "binary_cat":
            # Both operands affect the returned value.  The stack is a layout
            # operation and its reduction is legal extra low-level coverage;
            # the requested palette is a minimum contract, not an exact label.
            return f"torch.sum(torch.stack(({sources[0]}, {sources[1]}), dim=0), dim=0)"
        return f"torch.flip({source}, dims=(-1,))"
    if node.rule == "indexing":
        index = f"torch.zeros_like({source}, dtype=torch.long)"
        if node.variant == "gather":
            return f"torch.gather({source}, -1, {index})"
        # A full-size all-zero index writes several values to the same output
        # position.  CUDA documents duplicate-index scatter as nondeterministic
        # and its gradient as incorrect.  One last-axis index per outer row is
        # unique, preserves the complete input shape, and moves a live value
        # from the last position to the first instead of lowering scatter to an
        # identity operation.
        return (
            f"{source}.scatter(-1, "
            f"torch.zeros_like({source}[..., :1], dtype=torch.long), "
            f"{source}[..., -1:])"
        )
    if node.rule == "sdpa":
        return (
            f"F.scaled_dot_product_attention({sources[0]}, {sources[1]}, {sources[2]}, "
            "dropout_p=0.0)"
        )
    if node.rule == "loss":
        function = {"smooth_l1": "smooth_l1_loss", "mse": "mse_loss", "l1": "l1_loss"}[node.variant]
        return f"F.{function}({source}, target)"
    if node.variant == "add":
        return f"{sources[0]} + 0.125 * {sources[1]}"
    if node.variant == "mul":
        return f"{sources[0]} * (1.0 + 0.125 * {sources[1]})"
    if node.variant == "mix":
        return f"{sources[0]} + 0.125 * {sources[1]} - 0.0625 * {sources[2]}"
    return {
        "sin": f"torch.sin({source})",
        "cos": f"torch.cos({source})",
        "square": f"torch.square({source})",
        "abs": f"torch.abs({source})",
        "neg": f"torch.neg({source})",
        "affine": f"{source} * 1.03125 - 0.015625",
    }[node.variant]


def _input_expression(shape: tuple[int, ...], mode: str) -> list[str]:
    rendered = ", ".join(str(value) for value in shape)
    shape_expr = f"({rendered},)"
    if mode == "contiguous":
        return [f"x = torch.randn({shape_expr})"]
    if mode == "positive":
        return [f"x = torch.rand({shape_expr}) + 0.125"]
    if mode == "offset":
        prefix = ", ".join(str(value) for value in shape[:-1])
        return [f"base = torch.randn(({prefix}, {shape[-1] + 1}))", "x = base[..., 1:]"]
    if mode == "strided":
        prefix = ", ".join(str(value) for value in shape[:-1])
        return [f"base = torch.randn(({prefix}, {shape[-1] * 2}))", "x = base[..., ::2]"]
    return [f"base = torch.randn({shape_expr})", "x = base.movedim(-1, -2).movedim(-2, -1)"]


def _render_code_lines(
    *,
    nodes: Sequence[Node],
    shape: tuple[int, ...],
    input_mode: str,
) -> list[str]:
    """Build source lines using only the sampled graph's executable nodes."""

    lines = [
        "import torch",
        "import torch.nn as nn",
        "import torch.nn.functional as F",
        "",
    ]
    lines.extend(["class Model(nn.Module):", "    def __init__(self):", "        super().__init__()"])
    modules = [line for node in nodes if (line := _module_line(node, shape)) is not None]
    if modules:
        lines.extend(f"        {line}" for line in modules)
    else:
        lines.append("        self.register_buffer('_anchor', torch.tensor(0.0), persistent=False)")
    lines.append("")
    has_loss = any(node.rule == "loss" for node in nodes)
    parameters = "x, target" if has_loss else "x"
    lines.append(f"    def forward(self, {parameters}):")
    for node in nodes:
        sources = ["x" if predecessor == -1 else f"v{predecessor}" for predecessor in node.predecessors]
        expression = _operation_expression(node, sources, len(shape))
        lines.append(f"        v{node.index} = {expression}")
    lines.append(f"        return v{nodes[-1].index}")
    lines.extend(["", "", "def get_inputs():"])
    lines.extend(f"    {line}" for line in _input_expression(shape, input_mode))
    if has_loss:
        lines.append("    target = torch.randn_like(x)")
        lines.append("    return (x, target)")
    else:
        lines.append("    return (x,)")
    return lines


def _render_code(
    *,
    nodes: Sequence[Node],
    edges: Sequence[tuple[int, int]],
    shape: tuple[int, ...],
    shape_proof: dict[str, Any],
    input_mode: str,
    minimum_lines: int,
) -> str:
    # ``edges`` and ``shape_proof`` are part of the lowering contract and are
    # deliberately retained in this signature for the shape-expansion caller.
    # Their evidence lives in the manifest, not as source comments.
    del edges, shape_proof
    lines = _render_code_lines(
        nodes=nodes,
        shape=shape,
        input_mode=input_mode,
    )
    code = "\n".join(lines).rstrip() + "\n"
    actual_code_lines = _code_only_physical_line_count(code)
    if actual_code_lines < minimum_lines:
        raise AssertionError(
            f"reachable lowering has {actual_code_lines} code lines, below requested {minimum_lines}"
        )
    if any(line.lstrip().startswith("#") for line in code.splitlines()):
        raise AssertionError("comment-only provenance padding is forbidden")
    tree = ast.parse(code)
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Model")
    methods = [node.name for node in model.body if isinstance(node, ast.FunctionDef)]
    if set(methods) != {"__init__", "forward"}:
        raise AssertionError(f"unexpected executable Model helper(s): {methods}")
    forward = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    assignments = [
        statement
        for statement in forward.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id.startswith("v")
    ]
    if len(assignments) != len(nodes):
        raise AssertionError("forward assignment count no longer matches typed DAG node count")
    return code


def _graph_manifest(
    nodes: Sequence[Node],
    edges: Sequence[tuple[int, int]],
    rank: int,
    shape_proof: dict[str, Any],
) -> dict[str, Any]:
    graph = nx.DiGraph()
    graph.add_node(-1, rule="input")
    for node in nodes:
        graph.add_node(node.index, rule=node.rule)
    graph.add_edges_from(edges)
    if not nx.is_directed_acyclic_graph(graph):
        raise AssertionError("sampled graph is not acyclic")
    typed = {
        "input_rank": rank,
        "input_shape_witness": shape_proof["witness"],
        "nodes": [dataclasses.asdict(node) for node in nodes],
        "edges": [list(edge) for edge in edges],
    }
    return {
        "typed_graph": typed,
        "typed_graph_sha256": _sha256_text(_canonical_json(typed)),
        "wl_topology_hash": nx.weisfeiler_lehman_graph_hash(graph, node_attr="rule"),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "skip_edge_count": sum(target - source > 1 for source, target in edges if source >= 0),
        "nonlocal_edge_count": sum(source != target - 1 for source, target in edges if source >= 0),
        "maximum_indegree": max(dict(graph.in_degree()).values()),
        "maximum_outdegree": max(dict(graph.out_degree()).values()),
        "fanout_node_count": sum(degree > 1 for degree in dict(graph.out_degree()).values()),
        "operator_arity_histogram": dict(
            sorted(collections.Counter(len(node.predecessors) for node in nodes).items())
        ),
        "shape_csp": shape_proof,
    }


def _compact_reservation(index: int) -> int | None:
    """Reserve an atomic/compact node target independently of source length."""

    return {20: 1, 21: 2, 22: 4, 23: 7}.get(index % 160)


def _compact_palette(rng: random.Random, requested_nodes: int) -> set[str]:
    """Sample low-volume coverage parameters without prescribing a graph.

    The reservation controls only the approximate node budget.  Family,
    concrete variant, predecessor edges, and sink closure still go through the
    same open sampler as every other row.  A loss-only palette needs an extra
    value-producing node, so it is excluded from the one-node cell.
    """

    choices = list(FAMILIES)
    if requested_nodes == 1:
        choices.remove("loss_distance")
    palette_size = 1 if requested_nodes <= 2 else rng.choice((1, 2))
    return set(rng.sample(choices, k=palette_size))


def _minimum_high_complexity_nonpointwise_nodes(node_count: int) -> int:
    return max(3, int(np.ceil(HIGH_COMPLEXITY_MIN_NONPOINTWISE_FRACTION * node_count)))


def _enforce_high_complexity_heterogeneity(
    nodes: Sequence[Node], index: int
) -> tuple[list[Node], set[str], int, bool]:
    """Make long graphs heterogeneous without adding a prewritten subgraph."""

    required = (
        _minimum_high_complexity_nonpointwise_nodes(len(nodes))
        if len(nodes) >= HIGH_COMPLEXITY_NODE_THRESHOLD
        else 0
    )
    actual = sum(node.rule in _NONPOINTWISE_RULES for node in nodes)
    if actual >= required:
        return list(nodes), set(), required, False
    # Replacing a binary/ternary node by a unary rule would silently orphan an
    # edge in the typed graph.  Only replace unary nodes, so source lowering
    # retains every declared predecessor exactly as before.
    replacements = [
        node
        for node in nodes
        if node.rule not in _NONPOINTWISE_RULES
        and node.rule != "loss"
        and len(node.predecessors) == 1
    ]
    missing = required - actual
    if len(replacements) < missing:
        raise AssertionError(f"candidate {index} cannot satisfy high-complexity heterogeneity")
    lowered = list(nodes)
    injected_families: set[str] = set()
    for offset, node in enumerate(replacements[:missing]):
        if (index + node.index + offset) % 2 == 0:
            lowered[node.index] = dataclasses.replace(
                node,
                rule="reduction",
                family="reduction",
                variant=("mean_center", "sum_scale", "amax_gate")[(index + offset) % 3],
                module_name=None,
            )
            injected_families.add("reduction")
        else:
            lowered[node.index] = dataclasses.replace(
                node,
                rule="layout",
                family="shape_layout",
                variant=("transpose", "chunk_cat", "flip")[(index + offset) % 3],
                module_name=None,
            )
            injected_families.add("shape_layout")
    return lowered, injected_families, required, True


def _lowering_predecessors_bound(nodes: Sequence[Node], rank: int) -> bool:
    """Check that every lowered expression consumes precisely its graph inputs."""

    for node in nodes:
        sources = ["x" if predecessor == -1 else f"v{predecessor}" for predecessor in node.predecessors]
        expression = _operation_expression(node, sources, rank)
        names = {
            item.id
            for item in ast.walk(ast.parse(expression, mode="eval"))
            if isinstance(item, ast.Name) and (item.id == "x" or item.id.startswith("v"))
        }
        if names != set(sources):
            return False
    return True


def _make_candidate(index: int, prompt_prefix: str) -> Candidate:
    rng = random.Random(f"{SEED}:{index}:{SAMPLING_CONTRACT}")
    compact_nodes = _compact_reservation(index)
    compact_reservation = compact_nodes is not None
    palette = (
        _compact_palette(rng, compact_nodes)
        if compact_nodes is not None
        else _sample_palette(rng, index)
    )
    line_ranges = ((20, 22), (35, 45), (50, 70), (75, 100))
    line_style = index % len(line_ranges)
    # Compact reservations deliberately remain compact.  Elsewhere, source
    # volume is made only by real typed-DAG assignments (a conservative base
    # source has at least 12 code lines before those assignments).
    minimum_lines = 0 if compact_reservation else rng.randint(*line_ranges[line_style])
    minimum_nodes = max((1, 5, 9, 16)[line_style], max(1, minimum_lines - 12))
    requested_nodes = compact_nodes if compact_reservation else max(
        _node_count_for_target(rng, index), minimum_nodes
    )
    rank = _rank_for_palette(rng, palette)
    shape, shape_proof = _symbolic_shape(rng, rank)
    for _ in range(32):
        nodes, edges, sink_merge_node_count = _build_nodes(rng, palette, rank, requested_nodes)
        if sink_merge_node_count > int(np.floor(SINK_MERGE_NODE_RATIO_CAP * len(nodes))):
            continue
        try:
            candidate_nodes, injected_families, minimum_nonpointwise_rule_count, guard_applied = (
                _enforce_high_complexity_heterogeneity(nodes, index)
            )
        except AssertionError:
            continue
        if _lowering_predecessors_bound(candidate_nodes, rank):
            nodes = candidate_nodes
            high_complexity_heterogeneity_guard_applied = guard_applied
            break
    else:
        raise AssertionError(f"candidate {index} could not satisfy DAG quality constraints")
    if compact_reservation:
        injected_families = set()
        minimum_nonpointwise_rule_count = 0
        high_complexity_heterogeneity_guard_applied = False
    palette.update(injected_families)
    if not _lowering_predecessors_bound(nodes, rank):
        raise AssertionError(f"candidate {index} lowering does not consume its declared predecessors")
    shape_proof = _prove_graph_shapes(nodes, shape, shape_proof)
    nonpointwise_rule_count = sum(node.rule in _NONPOINTWISE_RULES for node in nodes)
    if nonpointwise_rule_count < minimum_nonpointwise_rule_count:
        raise AssertionError(f"candidate {index} violates high-complexity heterogeneity threshold")
    graph = _graph_manifest(nodes, edges, rank, shape_proof)
    input_mode = rng.choice(("contiguous", "positive", "offset", "strided", "movedim"))
    code = _render_code(
        nodes=nodes,
        edges=edges,
        shape=shape,
        shape_proof=shape_proof,
        input_mode=input_mode,
        minimum_lines=minimum_lines,
    )
    for node in nodes:
        if node.rule != "normalization":
            continue
        expected_class = "nn.GroupNorm" if node.variant == "group_norm" else "nn.LayerNorm"
        expected_binding = f"self.{node.module_name} = {expected_class}("
        if expected_binding not in code:
            raise AssertionError(
                f"normalization rule/lowering mismatch for candidate {index}, "
                f"node {node.index}: variant={node.variant}"
            )
    signature = extract_operator_signature(code)
    features = feature_dict(extract_complexity_features(code))
    code_only_lines = _code_only_physical_line_count(code)
    code_metrics = {
        "physical_line_count": int(features["source_line_count"]),
        "code_only_physical_line_count": code_only_lines,
        "comment_only_physical_line_count": sum(
            bool(line.strip()) and line.lstrip().startswith("#") for line in code.splitlines()
        ),
        "source_format_expansion_lines": 0,
    }
    actual_families = frozenset(
        cell
        for token in signature
        for cell in _token_cells(token)
        if cell in FAMILIES
    )
    if not actual_families.issuperset(palette):
        raise AssertionError(
            f"lowering lost required coverage for candidate {index}: palette={sorted(palette)}, "
            f"actual={sorted(actual_families)}"
        )
    code_sha = _sha256_text(code)
    uuid = f"cspdag_{graph['typed_graph_sha256'][:14]}_{code_sha[:10]}"
    row = _drkernel_row(
        code=code,
        prompt_prefix=prompt_prefix,
        data_source=DATA_SOURCE,
        level="open_csp_dag_canary",
        repo_name="local/open-csp-dag",
        task_type="synthetic_open_csp_dag",
        uuid=uuid,
    )
    manifest = {
        "candidate_index": index,
        "uuid": uuid,
        "contract": CONTRACT,
        "sampling_contract": SAMPLING_CONTRACT,
        "rule_catalog_version": RULE_CATALOG_VERSION,
        "parent_uuid": None,
        "kernelbench_usage": "low-level family and form-bucket coverage checklist only; no target proportions",
        "graph_seed_source": "fresh sampled CSP-DAG; no source graph template",
        "requested_family_palette": sorted(palette),
        "requested_family_palette_size": len(palette),
        "actual_low_level_families": sorted(actual_families),
        "input_mode": input_mode,
        "top_level_class_policy": "single_model_only",
        "semantic_multiclass_status": "deferred_until_real_subgraph_helper_lowering",
        # Retain the former field for the shape-expansion renderer, while
        # making its code-only meaning explicit in v3 manifests.
        "requested_minimum_source_lines": minimum_lines,
        "requested_minimum_code_only_physical_lines": minimum_lines,
        "requested_dag_node_count_before_sink_merges": requested_nodes,
        "high_complexity_heterogeneity_guard_applied": high_complexity_heterogeneity_guard_applied,
        "high_complexity_nonpointwise_rule_count": nonpointwise_rule_count,
        "high_complexity_minimum_nonpointwise_rule_count": minimum_nonpointwise_rule_count,
        "sink_merge_node_count": sink_merge_node_count,
        "sink_merge_node_ratio": round(sink_merge_node_count / len(nodes), 6),
        "sink_merge_node_ratio_cap": SINK_MERGE_NODE_RATIO_CAP,
        "graph": graph,
        "operator_signature": list(signature),
        "complexity": features,
        "code_metrics": code_metrics,
        "reference_sha256": code_sha,
        "normalized_ast_sha256": _normalized_ast_hash(code),
        "materialization_status": "review_only",
        "training_approved": False,
    }
    return Candidate(
        index=index,
        uuid=uuid,
        code=code,
        manifest=manifest,
        row=row,
        families=actual_families,
        line_bucket=_line_bucket(code_only_lines),
        op_bucket=_op_bucket(len(signature)),
        tie_key=_sha256_text(f"{SEED}:{uuid}"),
    )


def _constraint_row(candidates: Sequence[Candidate], predicate: Any) -> list[float]:
    return [1.0 if predicate(candidate) else 0.0 for candidate in candidates]


def _code_only_lines(candidate: Candidate) -> int:
    return int(candidate.manifest["code_metrics"]["code_only_physical_line_count"])


def _select(candidates: Sequence[Candidate], rows: int) -> tuple[list[Candidate], dict[str, Any]]:
    if rows != ROWS:
        raise ValueError(f"this canary contract requires exactly {ROWS} rows")
    matrix: list[list[float]] = []
    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    names: list[str] = []

    def add(name: str, lower: int, upper: int | float, predicate: Any) -> None:
        names.append(name)
        lower_bounds.append(float(lower))
        upper_bounds.append(float(upper))
        matrix.append(_constraint_row(candidates, predicate))

    add("rows", rows, rows, lambda candidate: True)
    for family, minimum in FAMILY_MINIMUMS.items():
        add(
            f"family:{family}",
            minimum,
            np.inf,
            lambda candidate, family=family: family in candidate.families,
        )
    for bucket, minimum in LINE_BUCKET_MINIMUMS.items():
        add(
            f"source_lines:{bucket}",
            minimum,
            np.inf,
            lambda candidate, bucket=bucket: candidate.line_bucket == bucket,
        )
    for bucket, minimum in OPCOUNT_BUCKET_MINIMUMS.items():
        add(
            f"operator_count:{bucket}",
            minimum,
            np.inf,
            lambda candidate, bucket=bucket: candidate.op_bucket == bucket,
        )
    add(
        "source_lines_ge_50",
        COMPLEXITY_MINIMUMS["source_lines_ge_50"],
        np.inf,
        lambda candidate: _code_only_lines(candidate) >= 50,
    )
    add(
        "operator_count_ge_10",
        COMPLEXITY_MINIMUMS["operator_count_ge_10"],
        np.inf,
        lambda candidate: len(candidate.manifest["operator_signature"]) >= 10,
    )
    add(
        "graph_nodes_ge_16",
        COMPLEXITY_MINIMUMS["graph_nodes_ge_16"],
        np.inf,
        lambda candidate: candidate.manifest["graph"]["node_count"] >= 16,
    )
    add(
        "nonlocal_edge_rows",
        COMPLEXITY_MINIMUMS["nonlocal_edge_rows"],
        np.inf,
        lambda candidate: candidate.manifest["graph"]["nonlocal_edge_count"] > 0,
    )
    objective = np.array(
        [
            -(
                5.0 * min(len(candidate.manifest["operator_signature"]), 64) / 64
                + 4.0 * min(_code_only_lines(candidate), 120) / 120
                + 4.0 * min(candidate.manifest["graph"]["node_count"], 40) / 40
                + 2.0 * min(candidate.manifest["graph"]["nonlocal_edge_count"], 20) / 20
                + len(candidate.families) / len(FAMILIES)
            )
            + 1e-6 * int(candidate.tie_key[:12], 16) / float(16**12)
            for candidate in candidates
        ],
        dtype=float,
    )
    constraints = LinearConstraint(
        np.asarray(matrix),
        np.asarray(lower_bounds),
        np.asarray(upper_bounds),
    )
    result = milp(
        c=objective,
        integrality=np.ones(len(candidates)),
        bounds=Bounds(np.zeros(len(candidates)), np.ones(len(candidates))),
        constraints=constraints,
        options={"time_limit": 120.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        availability = {
            name: int(sum(row)) for name, row in zip(names, matrix, strict=True)
        }
        raise RuntimeError(
            f"selection MILP failed: status={result.status}, message={result.message}, availability={availability}"
        )
    selected = [candidate for candidate, value in zip(candidates, result.x, strict=True) if value > 0.5]
    selected.sort(key=lambda candidate: candidate.tie_key)
    selected_uuids = {candidate.uuid for candidate in selected}
    achieved = {
        name: int(
            round(
                sum(
                    value
                    for candidate, value in zip(candidates, row, strict=True)
                    if candidate.uuid in selected_uuids
                )
            )
        )
        for name, row in zip(names, matrix, strict=True)
    }
    violations = {
        name: {"lower": lower, "upper": upper, "achieved": achieved[name]}
        for name, lower, upper in zip(names, lower_bounds, upper_bounds, strict=True)
        if achieved[name] < lower or achieved[name] > upper
    }
    if violations:
        raise AssertionError(f"post-selection contract mismatch: {violations}")
    return selected, {
        "solver": "scipy.optimize.milp",
        "scipy_version": scipy.__version__,
        "status": int(result.status),
        "message": result.message,
        "objective": float(result.fun),
        "constraints": {
            name: {
                "kind": "exact" if lower == upper else "minimum",
                "required": int(lower),
                "achieved": achieved[name],
            }
            for name, lower, upper in zip(names, lower_bounds, upper_bounds, strict=True)
        },
        "objective_policy": "maximize operator count, code-only physical lines, graph nodes, nonlocal edges, and family breadth; stable hash breaks residual ties",
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text("".join(_canonical_json(row) + "\n" for row in rows))


def _root_hashes(paths: Sequence[Path]) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    exact: set[str] = set()
    normalized: set[str] = set()
    sources: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            sources.append({"path": str(path), "status": "missing"})
            continue
        rows = 0
        parse_errors = 0
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=256, columns=["reward_model.ground_truth"]):
            for item in batch.to_pylist():
                rows += 1
                code = item.get("reward_model", {}).get("ground_truth")
                if not isinstance(code, str):
                    continue
                exact.add(_sha256_text(code))
                try:
                    normalized.add(_normalized_ast_hash(code))
                except SyntaxError:
                    parse_errors += 1
        sources.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256_file(path),
                "rows": rows,
                "normalized_ast_parse_errors": parse_errors,
                "status": "loaded",
            }
        )
    return exact, normalized, sources


def _dedup_audit(selected: Sequence[Candidate], paths: Sequence[Path]) -> dict[str, Any]:
    code_hashes = [candidate.manifest["reference_sha256"] for candidate in selected]
    ast_hashes = [candidate.manifest["normalized_ast_sha256"] for candidate in selected]
    graph_hashes = [candidate.manifest["graph"]["typed_graph_sha256"] for candidate in selected]
    topology_hashes = [candidate.manifest["graph"]["wl_topology_hash"] for candidate in selected]
    root_exact, root_ast, sources = _root_hashes(paths)
    return {
        "within_canary": {
            "rows": len(selected),
            "unique_reference_sha256": len(set(code_hashes)),
            "unique_normalized_ast_sha256": len(set(ast_hashes)),
            "unique_typed_graph_sha256": len(set(graph_hashes)),
            "unique_wl_topology_hash": len(set(topology_hashes)),
        },
        "against_roots": {
            "exact_reference_matches": sum(value in root_exact for value in code_hashes),
            "normalized_ast_matches": sum(value in root_ast for value in ast_hashes),
            "sources": sources,
        },
        "near_duplicate_status": "not computed by the generator; run audit_csp_dag_near_duplicates.py on the selected set and rebuilt candidate pool",
    }


def _histogram(values: Sequence[Any]) -> dict[str, int]:
    return dict(sorted(collections.Counter(str(value) for value in values).items()))


def _pool_profile(candidates: Sequence[Candidate]) -> dict[str, Any]:
    return {
        "rows": len(candidates),
        "family_row_presence": {
            family: sum(family in candidate.families for candidate in candidates)
            for family in FAMILIES
        },
        "source_line_buckets": _histogram([candidate.line_bucket for candidate in candidates]),
        "operator_count_buckets": _histogram([candidate.op_bucket for candidate in candidates]),
        "multiple_top_level_classes": sum(
            int(candidate.manifest["complexity"]["top_level_class_count"]) > 1
            for candidate in candidates
        ),
        "requested_palette_size": _histogram(
            [candidate.manifest["requested_family_palette_size"] for candidate in candidates]
        ),
        "actual_family_count": _histogram([len(candidate.families) for candidate in candidates]),
        "unique_reference_sha256": len(
            {candidate.manifest["reference_sha256"] for candidate in candidates}
        ),
        "unique_typed_graph_sha256": len(
            {candidate.manifest["graph"]["typed_graph_sha256"] for candidate in candidates}
        ),
    }


def _selected_profile(selected: Sequence[Candidate]) -> dict[str, Any]:
    bucket_nodes: dict[str, list[int]] = collections.defaultdict(list)
    for candidate in selected:
        bucket_nodes[candidate.line_bucket].append(candidate.manifest["graph"]["node_count"])
    return {
        "rows": len(selected),
        "family_row_presence": {
            family: sum(family in candidate.families for candidate in selected)
            for family in FAMILIES
        },
        "source_line_buckets": _histogram([candidate.line_bucket for candidate in selected]),
        "operator_count_buckets": _histogram([candidate.op_bucket for candidate in selected]),
        "multiple_top_level_classes": sum(
            int(candidate.manifest["complexity"]["top_level_class_count"]) > 1
            for candidate in selected
        ),
        "complexity_thresholds": {
            "source_lines_ge_50": sum(_code_only_lines(candidate) >= 50 for candidate in selected),
            "operator_count_ge_10": sum(
                len(candidate.manifest["operator_signature"]) >= 10
                for candidate in selected
            ),
            "graph_nodes_ge_16": sum(
                candidate.manifest["graph"]["node_count"] >= 16
                for candidate in selected
            ),
            "nonlocal_edge_rows": sum(
                candidate.manifest["graph"]["nonlocal_edge_count"] > 0
                for candidate in selected
            ),
        },
        "source_line_histogram": _histogram([_code_only_lines(candidate) for candidate in selected]),
        "physical_source_line_histogram": _histogram(
            [candidate.manifest["complexity"]["source_line_count"] for candidate in selected]
        ),
        "operator_count_histogram": _histogram(
            [len(candidate.manifest["operator_signature"]) for candidate in selected]
        ),
        "node_count_histogram": _histogram(
            [candidate.manifest["graph"]["node_count"] for candidate in selected]
        ),
        "source_line_bucket_node_counts": {
            bucket: {
                "minimum": min(values),
                "maximum": max(values),
                "mean": round(sum(values) / len(values), 6),
            }
            for bucket, values in sorted(bucket_nodes.items())
        },
        "skip_edge_rows": sum(
            candidate.manifest["graph"]["skip_edge_count"] > 0 for candidate in selected
        ),
        "fanin_ge_2_rows": sum(
            candidate.manifest["graph"]["maximum_indegree"] >= 2 for candidate in selected
        ),
        "nonlocal_edge_rows": sum(
            candidate.manifest["graph"]["nonlocal_edge_count"] > 0 for candidate in selected
        ),
        "fanout_rows": sum(
            candidate.manifest["graph"]["fanout_node_count"] > 0 for candidate in selected
        ),
        "requested_palette_size": _histogram(
            [candidate.manifest["requested_family_palette_size"] for candidate in selected]
        ),
        "actual_family_count": _histogram([len(candidate.families) for candidate in selected]),
        "operator_arities": _histogram(
            [
                len(node["predecessors"])
                for candidate in selected
                for node in candidate.manifest["graph"]["typed_graph"]["nodes"]
            ]
        ),
        "input_modes": _histogram(
            [candidate.manifest["input_mode"] for candidate in selected]
        ),
        "rule_variants": _histogram(
            [
                f"{node['rule']}:{node['variant']}"
                for candidate in selected
                for node in candidate.manifest["graph"]["typed_graph"]["nodes"]
            ]
        ),
    }


def _review_samples(selected: Sequence[Candidate]) -> list[Candidate]:
    samples: list[Candidate] = []
    seen: set[str] = set()
    predicates = [
        lambda candidate: candidate.op_bucket == "1",
        lambda candidate: candidate.op_bucket == "5-9",
        lambda candidate: candidate.op_bucket == ">=16",
        lambda candidate: "indexing_scatter" in candidate.families,
        lambda candidate: "loss_distance" in candidate.families,
        lambda candidate: "scaled_dot_product_attention" in candidate.families,
        lambda candidate: candidate.manifest["graph"]["skip_edge_count"] >= 3,
    ]
    for predicate in predicates:
        match = next(
            (candidate for candidate in selected if predicate(candidate) and candidate.uuid not in seen),
            None,
        )
        if match is not None:
            samples.append(match)
            seen.add(match.uuid)
    return samples


def _review_markdown(samples: Sequence[Candidate]) -> str:
    lines = [
        "# Open CSP-DAG canary manual review packet",
        "",
        "These are real selected references, not sketches. Each sample records its sampled typed graph.",
        "",
    ]
    for candidate in samples:
        graph = candidate.manifest["graph"]
        lines.extend(
            [
                f"## {candidate.uuid}",
                "",
                f"- families: {', '.join(sorted(candidate.families))}",
                f"- code-only source-line bucket: {candidate.line_bucket}",
                f"- operator-count bucket: {candidate.op_bucket}",
                f"- typed graph: {graph['node_count']} nodes, {graph['edge_count']} edges, {graph['skip_edge_count']} skip edges",
                f"- typed graph sha256: {graph['typed_graph_sha256']}",
                "",
                "```python",
                candidate.code.rstrip(),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def generate(
    output_dir: Path,
    template_path: Path = _DEFAULT_TEMPLATE,
    candidate_count: int = CANDIDATES,
    rows: int = ROWS,
    comparison_paths: Sequence[Path] = COMPARISON_ROOTS,
) -> dict[str, Any]:
    if candidate_count < rows:
        raise ValueError("candidate_count must be at least rows")
    output_dir.mkdir(parents=True, exist_ok=True)
    schema, prompt_prefix = _template(template_path)
    raw_candidates = [_make_candidate(index, prompt_prefix) for index in range(candidate_count)]
    candidates_by_identity: dict[str, Candidate] = {}
    for candidate in raw_candidates:
        identity = candidate.manifest["graph"]["typed_graph_sha256"]
        candidates_by_identity.setdefault(identity, candidate)
    candidates = list(candidates_by_identity.values())
    selected, solver = _select(candidates, rows)

    parquet_path = output_dir / "selected.parquet"
    manifest_path = output_dir / "selected.manifest.jsonl"
    review_path = output_dir / "manual_review.md"
    pool_profile_path = output_dir / "candidate_pool_profile.json"
    table = pa.Table.from_pylist([candidate.row for candidate in selected], schema=schema)
    table = table.replace_schema_metadata(
        {
            **(table.schema.metadata or {}),
            b"csp_dag.contract": CONTRACT.encode(),
            b"csp_dag.rule_catalog": RULE_CATALOG_VERSION.encode(),
            b"csp_dag.review_only": b"true",
            b"csp_dag.training_approved": b"false",
        }
    )
    pq.write_table(table, parquet_path, compression="zstd")
    _write_jsonl(manifest_path, [candidate.manifest for candidate in selected])
    _write_json(pool_profile_path, _pool_profile(candidates))
    review_path.write_text(_review_markdown(_review_samples(selected)))

    dedup = _dedup_audit(selected, comparison_paths)
    selected_profile = _selected_profile(selected)
    topology_counts = collections.Counter(
        candidate.manifest["graph"]["wl_topology_hash"] for candidate in selected
    )
    generator_path = Path(__file__).resolve()
    imported_sources = (
        _REPO_ROOT / "tools/data/cleaning/complexity.py",
        _REPO_ROOT / "tools/data/cleaning/external.py",
        _REPO_ROOT / "tools/data/synthesize/select_low_level_kernelbench_canary.py",
    )
    summary = {
        "contract": CONTRACT,
        "method": "open_constraint_solved_heterogeneous_dag",
        "rows": len(selected),
        "raw_candidate_rows": len(raw_candidates),
        "candidate_rows": len(candidates),
        "candidate_graph_duplicates_removed": len(raw_candidates) - len(candidates),
        "seed": SEED,
        "design": {
            "method_positioning": "custom rule-based open CSP-DAG; shares handwritten operator specifications and solver validation with NNSmith and the multi-input DAG/whole-graph CSP goal with DRTriton, but reproduces neither algorithm",
            "drtriton_gap": "uses a preselected simple shape witness and mostly shape-preserving rules instead of generating the full operator DAG first and using CP-SAT to search all tensor shapes",
            "nnsmith_gap": "does not implement forward/backward incremental insertion, placeholder replacement, attribute binning, or per-insertion-point solving",
            "closed_graph_templates": False,
            "finite_component": "low-level operator rule catalog only",
            "graph_generation": "fresh DAG with sampled length, rule sequence, variants, true operator arity, and arbitrary earlier-value predecessors; live sinks are joined by sampled binary/ternary nodes",
            "reservoir_dedup": "typed-graph exact dedup before optimization; the selector never sees duplicate typed graphs",
            "shape_generation": "Z3 integer CSP with bounds, divisibility, optional axis equalities, and stored witness",
            "source_lowering": "independently sampled input mode, one Model class, and one executable assignment per sampled DAG node",
            "source_line_policy": "source-volume buckets count code-only physical lines; long forms contain more reachable typed-DAG assignments or module definitions, and no comment-only or formatting-only padding is emitted",
            "kernelbench_usage": "low-level family and coarse source/operator bucket coverage only; selected proportions are not aligned to KernelBench",
            "kernelbench_high_level_compositions": "out of scope and not used as seeds",
        },
        "selection_policy": {
            "family_minimums": FAMILY_MINIMUMS,
            "code_only_source_line_bucket_minimums": LINE_BUCKET_MINIMUMS,
            "operator_count_bucket_minimums": OPCOUNT_BUCKET_MINIMUMS,
            "complexity_minimums": COMPLEXITY_MINIMUMS,
            "distribution_policy": "coverage floors only; no KernelBench proportion matching",
            "objective": "after satisfying coverage floors, prefer high operator count, long source, large graph, nonlocal edges, and broad actual family coverage",
        },
        "solver": solver,
        "candidate_pool": _pool_profile(candidates),
        "selected": selected_profile,
        "open_diversity_evidence": {
            "unique_typed_graph_sha256": dedup["within_canary"]["unique_typed_graph_sha256"],
            "unique_wl_topology_hash": dedup["within_canary"]["unique_wl_topology_hash"],
            "largest_wl_topology_cluster": max(topology_counts.values()),
            "rows_with_skip_edges": selected_profile["skip_edge_rows"],
            "rows_with_nonlocal_edges": selected_profile["nonlocal_edge_rows"],
            "rows_with_fanin_ge_2": selected_profile["fanin_ge_2_rows"],
            "rows_with_fanout": selected_profile["fanout_rows"],
            "template_identifier_field_present": False,
        },
        "dedup_audit": dedup,
        "source_binding": {
            "git_commit": _git_commit(),
            "generator": {
                "path": str(generator_path),
                "sha256": _sha256_file(generator_path),
            },
            "imported_sources": [
                {"path": str(path.resolve()), "sha256": _sha256_file(path)}
                for path in imported_sources
            ],
            "template": {
                "path": str(template_path.resolve()),
                "sha256": _sha256_file(template_path),
                "usage": "schema and prompt prefix only",
            },
            "versions": {
                "networkx": nx.__version__,
                "numpy": np.__version__,
                "pyarrow": pa.__version__,
                "scipy": scipy.__version__,
                "z3": z3.get_version_string(),
            },
        },
        "artifacts": {
            "selected": {"path": str(parquet_path.resolve()), "sha256": _sha256_file(parquet_path)},
            "manifest": {"path": str(manifest_path.resolve()), "sha256": _sha256_file(manifest_path)},
            "candidate_pool_profile": {
                "path": str(pool_profile_path.resolve()),
                "sha256": _sha256_file(pool_profile_path),
            },
            "manual_review": {"path": str(review_path.resolve()), "sha256": _sha256_file(review_path)},
        },
        "review_only": True,
        "training_approved": False,
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--template-parquet", type=Path, default=_DEFAULT_TEMPLATE)
    parser.add_argument("--candidate-count", type=int, default=CANDIDATES)
    parser.add_argument("--rows", type=int, default=ROWS)
    args = parser.parse_args()
    summary = generate(
        output_dir=args.output_dir,
        template_path=args.template_parquet,
        candidate_count=args.candidate_count,
        rows=args.rows,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
