"""AST-structure similarity for held-out benchmark decontamination.

The scoring contract follows ``PythonASTSimilarity`` as used by CUDA-Agent:
compare significant Python subtrees, allow child reordering through maximum
weight assignment, and normalize the matched nodes by both programs' weighted
subtree sizes.  Only the declared entry-point class is compared so helpers and
the shared ``get_inputs`` scaffold do not dominate the score.

The public metric is implemented in Relari's Apache-2.0 ``continuous-eval``
project and credits Pedro Salazar Paredes' MIT ``python-ast-comparison``.  This
module independently implements the published scoring contract; v3 review
checked its scores against the public implementation on representative pairs.
No runtime dependency on either package is required.

The implementation is dependency-free.  Its assignment solver and optimistic
size bound make a many-training-to-small-held-out scan practical while keeping
the score deterministic and auditable.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


@dataclass(frozen=True)
class _SignificantTree:
    node: ast.AST
    node_count: int
    child_count: int
    weight: float


@dataclass(frozen=True)
class ASTSimilarityBaseline:
    dataset_index: int
    row_index: int
    entry_point: str
    subtrees: tuple[_SignificantTree, ...]
    total_weight: float


@dataclass(frozen=True)
class ASTSimilarityMatch:
    dataset_index: int
    row_index: int
    similarity: float


def _nested(row: dict[str, Any], path: str) -> Any:
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _root_weight(node: ast.AST) -> float:
    if isinstance(node, ast.Import):
        return 0.3
    if isinstance(node, ast.FunctionDef):
        return 1.2
    if isinstance(node, ast.If):
        return 0.5
    return 1.0


def _is_significant(node: ast.AST) -> bool:
    return isinstance(
        node,
        (
            ast.Import,
            ast.FunctionDef,
            ast.If,
            ast.ClassDef,
            ast.While,
            ast.For,
            ast.comprehension,
            ast.Return,
        ),
    )


def _entry_point_tree(code: str, entry_point: str) -> ast.ClassDef:
    tree = ast.parse(code)
    matches = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == entry_point]
    if not matches:
        raise ValueError(f"code does not define entry-point class {entry_point!r}")
    # Python uses the last top-level definition when names are shadowed.  The
    # cleanup canonicalizer applies the same rule before this function runs.
    return matches[-1]


def _significant_trees(code: str, entry_point: str) -> tuple[_SignificantTree, ...]:
    root = _entry_point_tree(code, entry_point)
    result = []
    for node in ast.walk(root):
        if not _is_significant(node):
            continue
        node_count = sum(1 for _ in ast.walk(node))
        result.append(
            _SignificantTree(
                node,
                node_count,
                sum(1 for _ in ast.iter_child_nodes(node)),
                _root_weight(node),
            )
        )
    return tuple(result)


def _maximum_assignment_sum(weights: list[list[float]]) -> float:
    """Return the maximum one-to-one assignment weight for a rectangle.

    This is the O(n^3) Hungarian algorithm.  Padding with zero-weight dummy
    rows/columns gives the same rectangular matching semantics as Munkres.
    """

    if not weights or not weights[0]:
        return 0.0
    size = max(len(weights), len(weights[0]))
    maximum = max(max(row, default=0.0) for row in weights)
    costs = [
        [maximum - (weights[i][j] if i < len(weights) and j < len(weights[0]) else 0.0) for j in range(size)]
        for i in range(size)
    ]
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    assigned_row = [0] * (size + 1)
    previous_column = [0] * (size + 1)
    for row in range(1, size + 1):
        assigned_row[0] = row
        column = 0
        minimum = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column] = True
            current_row = assigned_row[column]
            delta = float("inf")
            next_column = 0
            for candidate in range(1, size + 1):
                if used[candidate]:
                    continue
                reduced = costs[current_row - 1][candidate - 1] - u[current_row] - v[candidate]
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    previous_column[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(size + 1):
                if used[candidate]:
                    u[assigned_row[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if assigned_row[column] == 0:
                break
        while True:
            prior = previous_column[column]
            assigned_row[column] = assigned_row[prior]
            column = prior
            if column == 0:
                break
    total = 0.0
    for column in range(1, size + 1):
        row = assigned_row[column]
        if 1 <= row <= len(weights) and column <= len(weights[0]):
            total += weights[row - 1][column - 1]
    return total


def _matching_nodes(left: ast.AST, right: ast.AST) -> int:
    left_children = list(ast.iter_child_nodes(left))
    right_children = list(ast.iter_child_nodes(right))
    if type(left) is not type(right) or len(left_children) != len(right_children):
        return 0
    if not left_children:
        return 1
    matrix = [
        [float(_matching_nodes(left_child, right_child)) for right_child in right_children]
        for left_child in left_children
    ]
    return 1 + int(round(_maximum_assignment_sum(matrix)))


def _optimistic_similarity_bound(
    left: tuple[_SignificantTree, ...],
    right: tuple[_SignificantTree, ...],
    denominator: float,
) -> float:
    matrix = [
        [
            (
                min(left_tree.node_count, right_tree.node_count) * (left_tree.weight + right_tree.weight) / 2
                if type(left_tree.node) is type(right_tree.node) and left_tree.child_count == right_tree.child_count
                else 0.0
            )
            for right_tree in right
        ]
        for left_tree in left
    ]
    return 2 * _maximum_assignment_sum(matrix) / denominator if denominator else 0.0


def ast_structure_similarity(
    left: tuple[_SignificantTree, ...],
    right: tuple[_SignificantTree, ...],
    *,
    left_total_weight: float,
    right_total_weight: float,
) -> float:
    denominator = left_total_weight + right_total_weight
    if not left or not right or not denominator:
        return 0.0
    matrix = [
        [
            _matching_nodes(left_tree.node, right_tree.node) * (left_tree.weight + right_tree.weight) / 2
            for right_tree in right
        ]
        for left_tree in left
    ]
    return round(2 * _maximum_assignment_sum(matrix) / denominator, 4)


def read_ast_similarity_baselines(
    paths: Iterable[Path],
    *,
    code_key: str,
    entry_point_key: str,
) -> list[ASTSimilarityBaseline]:
    baselines: list[ASTSimilarityBaseline] = []
    columns = sorted({code_key.split(".", 1)[0], entry_point_key.split(".", 1)[0]})
    for dataset_index, path in enumerate(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        row_index = 0
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=4096,
            columns=columns,
            use_threads=False,
        ):
            for row in batch.to_pylist():
                code = _nested(row, code_key)
                entry_point = _nested(row, entry_point_key)
                if not isinstance(entry_point, str) or not entry_point:
                    entry_point = "Model"
                if isinstance(code, str):
                    try:
                        subtrees = _significant_trees(code, entry_point)
                    except (SyntaxError, ValueError, RecursionError):
                        subtrees = ()
                    if subtrees:
                        baselines.append(
                            ASTSimilarityBaseline(
                                dataset_index,
                                row_index,
                                entry_point,
                                subtrees,
                                sum(tree.node_count * tree.weight for tree in subtrees),
                            )
                        )
                row_index += 1
    return baselines


def best_ast_similarity_match(
    code: str,
    entry_point: str,
    baselines: Iterable[ASTSimilarityBaseline],
    *,
    threshold: float,
) -> ASTSimilarityMatch | None:
    """Return the strongest entry-point AST match strictly above threshold."""

    subtrees = _significant_trees(code, entry_point)
    total_weight = sum(tree.node_count * tree.weight for tree in subtrees)
    best: ASTSimilarityMatch | None = None
    for baseline in baselines:
        if baseline.entry_point != entry_point:
            continue
        denominator = total_weight + baseline.total_weight
        if _optimistic_similarity_bound(subtrees, baseline.subtrees, denominator) <= threshold:
            continue
        similarity = ast_structure_similarity(
            subtrees,
            baseline.subtrees,
            left_total_weight=total_weight,
            right_total_weight=baseline.total_weight,
        )
        if similarity <= threshold:
            continue
        if best is None or similarity > best.similarity:
            best = ASTSimilarityMatch(
                baseline.dataset_index,
                baseline.row_index,
                similarity,
            )
    return best
