"""KernelBench-calibrated structural complexity labels and deterministic caps."""

from __future__ import annotations

import ast
import collections
import hashlib
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

KERNELBENCH_LEVELS = ("level1", "level2", "level3")
LEVEL1_NOOP_TOKENS = frozenset({"nn.Identity", "torch.nn.Identity"})
FEATURE_NAMES = (
    "forward_call_count",
    "forward_unique_call_count",
    "forward_self_call_count",
    "init_nn_constructor_count",
    "forward_statement_count",
    "forward_control_flow_count",
    "forward_operator_count",
    "top_level_class_count",
    "forward_ast_depth",
    "source_line_count",
)
_NON_COMPUTE_CALLS = frozenset(
    {
        "bool",
        "dict",
        "enumerate",
        "float",
        "getattr",
        "hasattr",
        "int",
        "len",
        "list",
        "max",
        "min",
        "range",
        "set",
        "str",
        "sum",
        "super",
        "tuple",
        "type",
        "zip",
    }
)
_NON_COMPUTE_METHODS = frozenset(
    {
        "append",
        "clear",
        "copy",
        "dim",
        "extend",
        "get",
        "index",
        "insert",
        "is_contiguous",
        "items",
        "keys",
        "numel",
        "pop",
        "remove",
        "size",
        "stride",
        "update",
        "values",
    }
)


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _ast_depth(node: ast.AST) -> int:
    children = list(ast.iter_child_nodes(node))
    return 1 + (max(_ast_depth(child) for child in children) if children else 0)


def extract_complexity_features(code: str, entry_point: str = "Model") -> tuple[float, ...]:
    """Extract transparent structure-only features from the effective model.

    The last top-level entry-point class and its last ``forward``/``__init__``
    definitions are effective under Python's shadowing rules. Calls in
    ``get_inputs`` are deliberately excluded: workload construction is not
    model complexity.
    """

    tree = ast.parse(code)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    models = [node for node in classes if node.name == entry_point]
    if not models:
        raise ValueError(f"missing top-level entry-point class {entry_point!r}")
    model = models[-1]
    forwards = [
        node
        for node in model.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "forward"
    ]
    if not forwards:
        raise ValueError(f"entry-point class {entry_point!r} has no forward method")
    forward = forwards[-1]
    initializers = [
        node
        for node in model.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__"
    ]
    initializer = initializers[-1] if initializers else None

    forward_calls = [_call_name(node.func) for node in ast.walk(forward) if isinstance(node, ast.Call)]
    nn_constructors = []
    if initializer is not None:
        nn_constructors = [
            name
            for node in ast.walk(initializer)
            if isinstance(node, ast.Call)
            for name in [_call_name(node.func)]
            if name.startswith(("nn.", "torch.nn."))
        ]

    features = (
        len(forward_calls),
        len(set(forward_calls)),
        sum(name.startswith("self.") for name in forward_calls),
        len(nn_constructors),
        sum(isinstance(node, ast.stmt) for node in ast.walk(forward)) - 1,
        sum(
            isinstance(node, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.Match))
            for node in ast.walk(forward)
        ),
        sum(isinstance(node, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare)) for node in ast.walk(forward)),
        len(classes),
        _ast_depth(forward),
        len(code.splitlines()),
    )
    return tuple(float(value) for value in features)


def feature_dict(features: Sequence[float]) -> dict[str, int | float]:
    if len(features) != len(FEATURE_NAMES):
        raise ValueError(f"expected {len(FEATURE_NAMES)} features, got {len(features)}")
    return {
        name: int(value) if float(value).is_integer() else float(value)
        for name, value in zip(FEATURE_NAMES, features, strict=True)
    }


def _forward_input_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names = {
        argument.arg
        for argument in [*function.args.posonlyargs, *function.args.args, *function.args.kwonlyargs]
        if argument.arg not in {"self", "cls"}
    }
    if function.args.vararg is not None:
        names.add(function.args.vararg.arg)
    if function.args.kwarg is not None:
        names.add(function.args.kwarg.arg)
    return names


def _depends_on_names(node: ast.AST | None, names: set[str]) -> bool:
    return node is not None and any(isinstance(child, ast.Name) and child.id in names for child in ast.walk(node))


def _assign_taint(target: ast.AST, dependency: bool, tainted: set[str]) -> None:
    if isinstance(target, ast.Name):
        if dependency:
            tainted.add(target.id)
        else:
            tainted.discard(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for item in target.elts:
            _assign_taint(item, dependency, tainted)


def _module_bindings(initializer: ast.FunctionDef | ast.AsyncFunctionDef | None) -> dict[str, str]:
    bindings: dict[str, str] = {}
    if initializer is None:
        return bindings
    for node in ast.walk(initializer):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or not isinstance(node.value, ast.Call):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        constructor = _call_name(node.value.func)
        for target in targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                bindings[target.attr] = constructor
    return bindings


def _normalized_compute_call(call: ast.Call, bindings: dict[str, str]) -> str | None:
    if isinstance(call.func, ast.Call):
        constructor = _call_name(call.func.func)
        if constructor.startswith(("nn.", "torch.nn.")):
            return constructor
    name = _call_name(call.func)
    if not name:
        return None
    if name in _NON_COMPUTE_CALLS:
        return None
    if name.startswith("F."):
        return "torch.nn.functional." + name.removeprefix("F.")
    if name.startswith("nn.functional."):
        return "torch.nn.functional." + name.removeprefix("nn.functional.")
    if name.startswith("torch.nn.functional.") or name.startswith("torch."):
        return name
    if name.startswith("self."):
        parts = name.split(".")
        if len(parts) == 2 and parts[1] in bindings:
            return bindings[parts[1]]
        if parts[-1] in _NON_COMPUTE_METHODS:
            return None
        return "self." + ".".join(parts[1:])
    if isinstance(call.func, ast.Attribute):
        if call.func.attr in _NON_COMPUTE_METHODS:
            return None
        return f"tensor.{call.func.attr}"
    return f"call.{name}"


def _expression_compute_tokens(expression: ast.AST, tainted: set[str], bindings: dict[str, str]) -> list[str]:
    local_tainted = set(tainted)
    # Targets in a comprehension inherit dependency from its iterable. Repeat
    # because nested comprehensions may consume an outer target.
    changed = True
    while changed:
        changed = False
        for generator in (node for node in ast.walk(expression) if isinstance(node, ast.comprehension)):
            if not _depends_on_names(generator.iter, local_tainted):
                continue
            targets = {node.id for node in ast.walk(generator.target) if isinstance(node, ast.Name)}
            if not targets.issubset(local_tainted):
                local_tainted.update(targets)
                changed = True
    tokens: list[str] = []
    for node in ast.walk(expression):
        if isinstance(node, ast.Call) and _depends_on_names(node, local_tainted):
            name = _normalized_compute_call(node, bindings)
            if name is not None:
                tokens.append(name)
        elif isinstance(node, ast.BinOp) and _depends_on_names(node, local_tainted):
            tokens.append(f"operator.{type(node.op).__name__.lower()}")
        elif isinstance(node, ast.UnaryOp) and _depends_on_names(node, local_tainted):
            tokens.append(f"operator.{type(node.op).__name__.lower()}")
        elif isinstance(node, ast.Compare) and _depends_on_names(node, local_tainted):
            tokens.extend(f"operator.{type(operator).__name__.lower()}" for operator in node.ops)
        elif isinstance(node, ast.BoolOp) and _depends_on_names(node, local_tainted):
            tokens.extend(f"operator.{type(node.op).__name__.lower()}" for _ in range(len(node.values) - 1))
    return tokens


def extract_operator_signature(code: str, entry_point: str = "Model") -> tuple[str, ...]:
    """Return the effective forward's input-dependent compute multiset.

    Calls used only for shape/control metadata are excluded. Module invocations
    are resolved to their constructor when ``self.name = Constructor(...)`` is
    visible in the effective initializer. The result is sorted because Level
    membership uses an operator multiset, not source spelling or local names.
    """

    tree = ast.parse(code)
    models = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == entry_point]
    if not models:
        raise ValueError(f"missing top-level entry-point class {entry_point!r}")
    model = models[-1]
    forwards = [
        node
        for node in model.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "forward"
    ]
    if not forwards:
        raise ValueError(f"entry-point class {entry_point!r} has no forward method")
    initializers = [
        node
        for node in model.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__"
    ]
    forward = forwards[-1]
    initializer = initializers[-1] if initializers else None
    bindings = _module_bindings(initializer)
    tainted = _forward_input_names(forward)
    tokens: list[str] = []

    for statement in forward.body:
        if isinstance(statement, ast.Assign):
            tokens.extend(_expression_compute_tokens(statement.value, tainted, bindings))
            dependency = _depends_on_names(statement.value, tainted)
            for target in statement.targets:
                _assign_taint(target, dependency, tainted)
        elif isinstance(statement, ast.AnnAssign):
            if statement.value is not None:
                tokens.extend(_expression_compute_tokens(statement.value, tainted, bindings))
            _assign_taint(statement.target, _depends_on_names(statement.value, tainted), tainted)
        elif isinstance(statement, ast.AugAssign):
            tokens.extend(_expression_compute_tokens(statement, tainted, bindings))
            _assign_taint(statement.target, _depends_on_names(statement, tainted), tainted)
        elif isinstance(statement, ast.Return):
            if statement.value is not None:
                tokens.extend(_expression_compute_tokens(statement.value, tainted, bindings))
        elif isinstance(statement, ast.Expr):
            tokens.extend(_expression_compute_tokens(statement.value, tainted, bindings))
        else:
            # Control flow is never admitted by the simple Level1 envelope, but
            # its compute still belongs in the reviewable signature.
            tokens.extend(_expression_compute_tokens(statement, tainted, bindings))
    return tuple(sorted(tokens))


@dataclass(frozen=True)
class Neighbor:
    distance: float
    level: str
    reference_id: str


@dataclass(frozen=True)
class Prediction:
    level: str
    vote_count: int
    neighbors: tuple[Neighbor, ...]
    basis: str = "structural_knn"
    structural_level: str | None = None


def level1_cap_candidate_bases(
    features: Sequence[float],
    signature: Sequence[str],
    prediction: Prediction,
) -> tuple[str, ...]:
    """Return conservative evidence for including a row in the Level1 cap.

    The taxonomy label deliberately refuses to call a one-call program Level1
    when structural neighbors look architecture-like.  That precision guard is
    unsuitable for a hard mixture cap: domain-shifted k-NN labels can call an
    explicit ``Identity -> single op`` program Level3.  For cap accounting we
    therefore also include every simple-envelope program with at most one
    compute token after removing explicit identity modules.
    """

    bases: list[str] = []
    if prediction.level == "level1":
        bases.append("taxonomy_level1")
    if prediction.structural_level == "level1":
        bases.append("structural_knn_level1")
    effective_signature = tuple(token for token in signature if token not in LEVEL1_NOOP_TOKENS)
    if KernelBenchTaxonomyClassifier._simple_level1_envelope(features) and len(effective_signature) <= 1:
        bases.append("single_compute_after_explicit_noop_removal")
    return tuple(bases)


class KernelBenchKNNClassifier:
    """Dependency-free k-NN over standardized, interpretable AST features."""

    def __init__(
        self,
        features: Sequence[Sequence[float]],
        labels: Sequence[str],
        reference_ids: Sequence[str],
        *,
        neighbors: int = 3,
    ) -> None:
        if not features or len(features) != len(labels) or len(features) != len(reference_ids):
            raise ValueError("reference features, labels, and IDs must have the same non-zero length")
        if neighbors < 1 or neighbors % 2 == 0 or neighbors > len(features):
            raise ValueError("neighbors must be an odd positive integer no larger than the reference set")
        width = len(FEATURE_NAMES)
        if any(len(row) != width for row in features):
            raise ValueError(f"every reference must have {width} features")
        if any(label not in KERNELBENCH_LEVELS for label in labels):
            raise ValueError(f"labels must be one of {KERNELBENCH_LEVELS}")
        if len(set(reference_ids)) != len(reference_ids):
            raise ValueError("reference IDs must be unique")

        self.features = tuple(tuple(float(value) for value in row) for row in features)
        self.labels = tuple(labels)
        self.reference_ids = tuple(reference_ids)
        self.neighbors = neighbors
        self.means = tuple(sum(row[i] for row in self.features) / len(self.features) for i in range(width))
        self.scales = tuple(
            math.sqrt(sum((row[i] - self.means[i]) ** 2 for row in self.features) / len(self.features)) or 1.0
            for i in range(width)
        )

    def _distance(self, left: Sequence[float], right: Sequence[float]) -> float:
        return sum(((float(a) - float(b)) / scale) ** 2 for a, b, scale in zip(left, right, self.scales, strict=True))

    def predict(self, features: Sequence[float], *, exclude_reference_id: str | None = None) -> Prediction:
        if len(features) != len(FEATURE_NAMES):
            raise ValueError(f"expected {len(FEATURE_NAMES)} features, got {len(features)}")
        ranked = sorted(
            (
                Neighbor(self._distance(features, reference), level, reference_id)
                for reference, level, reference_id in zip(self.features, self.labels, self.reference_ids, strict=True)
                if reference_id != exclude_reference_id
            ),
            key=lambda item: (item.distance, item.level, item.reference_id),
        )
        if len(ranked) < self.neighbors:
            raise ValueError("not enough references remain for the requested neighbor count")
        nearest = tuple(ranked[: self.neighbors])
        votes = collections.Counter(item.level for item in nearest)
        distance_by_level = collections.defaultdict(float)
        for item in nearest:
            distance_by_level[item.level] += item.distance
        level = min(
            votes,
            key=lambda candidate: (
                -votes[candidate],
                distance_by_level[candidate],
                KERNELBENCH_LEVELS.index(candidate),
            ),
        )
        return Prediction(
            level=level,
            vote_count=votes[level],
            neighbors=nearest,
            structural_level=level,
        )

    def leave_one_out_report(self) -> dict[str, Any]:
        confusion = {actual: {predicted: 0 for predicted in KERNELBENCH_LEVELS} for actual in KERNELBENCH_LEVELS}
        for features, actual, reference_id in zip(self.features, self.labels, self.reference_ids, strict=True):
            predicted = self.predict(features, exclude_reference_id=reference_id).level
            confusion[actual][predicted] += 1
        correct = sum(confusion[level][level] for level in KERNELBENCH_LEVELS)
        total = len(self.features)
        per_level: dict[str, dict[str, float | int]] = {}
        for level in KERNELBENCH_LEVELS:
            true_positive = confusion[level][level]
            actual_count = sum(confusion[level].values())
            predicted_count = sum(confusion[actual][level] for actual in KERNELBENCH_LEVELS)
            per_level[level] = {
                "support": actual_count,
                "predicted": predicted_count,
                "precision": true_positive / predicted_count if predicted_count else 0.0,
                "recall": true_positive / actual_count if actual_count else 0.0,
            }
        return {
            "rows": total,
            "accuracy": correct / total,
            "confusion": confusion,
            "per_level": per_level,
        }


class KernelBenchTaxonomyClassifier:
    """High-precision Level1 rule plus structural Level2/Level3 calibration.

    General structural similarity is insufficient for synthetic fusions: a
    two-op fusion can have the same short ``forward`` shape as an official
    composite loss. Level1 therefore requires either exactly one effective
    input-dependent compute token or an exact operator signature observed in
    an official Level1 reference. The k-NN result is used only after that gate.
    """

    def __init__(
        self,
        features: Sequence[Sequence[float]],
        signatures: Sequence[Sequence[str]],
        labels: Sequence[str],
        reference_ids: Sequence[str],
        *,
        neighbors: int = 3,
    ) -> None:
        if len(signatures) != len(features):
            raise ValueError("reference signatures and features must have the same length")
        self.knn = KernelBenchKNNClassifier(features, labels, reference_ids, neighbors=neighbors)
        self.signatures = tuple(tuple(signature) for signature in signatures)
        self.labels = tuple(labels)
        self.reference_ids = tuple(reference_ids)
        self.neighbors = neighbors

    @staticmethod
    def _simple_level1_envelope(features: Sequence[float]) -> bool:
        values = dict(zip(FEATURE_NAMES, features, strict=True))
        return (
            values["init_nn_constructor_count"] <= 2
            and values["forward_statement_count"] <= 4
            and values["forward_control_flow_count"] == 0
            and values["top_level_class_count"] == 1
        )

    def _known_level1_signature(self, signature: Sequence[str], exclude_reference_id: str | None) -> bool:
        target = tuple(signature)
        return any(
            level == "level1" and reference_signature == target and reference_id != exclude_reference_id
            for reference_signature, level, reference_id in zip(
                self.signatures, self.labels, self.reference_ids, strict=True
            )
        )

    def predict(
        self,
        features: Sequence[float],
        signature: Sequence[str],
        *,
        exclude_reference_id: str | None = None,
    ) -> Prediction:
        structural = self.knn.predict(features, exclude_reference_id=exclude_reference_id)
        envelope = self._simple_level1_envelope(features)
        # A one-call architecture such as a multi-layer GRU is Level3 in the
        # official taxonomy even though its Python forward has one call. Keep
        # the single-op shortcut only when structural neighbors do not identify
        # an architecture-level task.
        if envelope and len(signature) == 1 and structural.level != "level3":
            return Prediction(
                "level1",
                structural.vote_count,
                structural.neighbors,
                "single_compute_signature",
                structural.level,
            )
        if envelope and self._known_level1_signature(signature, exclude_reference_id):
            return Prediction(
                "level1",
                structural.vote_count,
                structural.neighbors,
                "known_level1_signature",
                structural.level,
            )
        level = "level2" if structural.level == "level1" else structural.level
        return Prediction(
            level,
            structural.vote_count,
            structural.neighbors,
            "structural_knn_non_level1",
            structural.level,
        )

    def leave_one_out_report(self) -> dict[str, Any]:
        confusion = {actual: {predicted: 0 for predicted in KERNELBENCH_LEVELS} for actual in KERNELBENCH_LEVELS}
        basis_counts: collections.Counter[str] = collections.Counter()
        for features, signature, actual, reference_id in zip(
            self.knn.features,
            self.signatures,
            self.labels,
            self.reference_ids,
            strict=True,
        ):
            prediction = self.predict(features, signature, exclude_reference_id=reference_id)
            confusion[actual][prediction.level] += 1
            basis_counts[prediction.basis] += 1
        correct = sum(confusion[level][level] for level in KERNELBENCH_LEVELS)
        total = len(self.labels)
        per_level: dict[str, dict[str, float | int]] = {}
        for level in KERNELBENCH_LEVELS:
            true_positive = confusion[level][level]
            actual_count = sum(confusion[level].values())
            predicted_count = sum(confusion[actual][level] for actual in KERNELBENCH_LEVELS)
            per_level[level] = {
                "support": actual_count,
                "predicted": predicted_count,
                "precision": true_positive / predicted_count if predicted_count else 0.0,
                "recall": true_positive / actual_count if actual_count else 0.0,
            }
        return {
            "rows": total,
            "accuracy": correct / total,
            "confusion": confusion,
            "per_level": per_level,
            "prediction_basis_counts": dict(sorted(basis_counts.items())),
        }


def deterministic_level1_selection(
    records: Iterable[tuple[int, str, str]],
    *,
    max_fraction: float,
    seed: int,
) -> tuple[set[int], dict[str, Any]]:
    """Keep all non-Level1 rows and hash-rank Level1 rows to satisfy a cap.

    ``records`` contains ``(row_index, uuid, predicted_level)``. Source order is
    not used for ranking, so concatenating or sharding an unchanged corpus does
    not change the selected UUIDs.
    """

    if not 0.0 <= max_fraction < 1.0:
        raise ValueError("max_fraction must be in [0, 1)")
    materialized = list(records)
    if len({row_index for row_index, _, _ in materialized}) != len(materialized):
        raise ValueError("row indices must be unique")
    if len({uuid for _, uuid, _ in materialized}) != len(materialized):
        raise ValueError("UUIDs must be unique")
    level1 = [(row_index, uuid) for row_index, uuid, level in materialized if level == "level1"]
    non_level1 = [(row_index, uuid) for row_index, uuid, level in materialized if level != "level1"]
    fraction = Fraction(str(max_fraction))
    max_level1 = fraction.numerator * len(non_level1) // (fraction.denominator - fraction.numerator)
    keep_level1 = min(len(level1), max_level1)
    ranked_level1 = sorted(
        level1,
        key=lambda item: (hashlib.sha256(f"{seed}:{item[1]}".encode()).hexdigest(), item[1], item[0]),
    )
    selected = {row_index for row_index, _ in non_level1}
    selected.update(row_index for row_index, _ in ranked_level1[:keep_level1])
    output_level1 = keep_level1
    output_rows = len(non_level1) + output_level1
    report = {
        "seed": seed,
        "max_level1_fraction": max_fraction,
        "input_rows": len(materialized),
        "input_level1_rows": len(level1),
        "input_level1_fraction": len(level1) / len(materialized) if materialized else 0.0,
        "non_level1_rows": len(non_level1),
        "maximum_level1_rows_given_non_level1": max_level1,
        "output_rows": output_rows,
        "output_level1_rows": output_level1,
        "output_level1_fraction": output_level1 / output_rows if output_rows else 0.0,
        "dropped_level1_rows": len(level1) - output_level1,
        "cap_applied": output_level1 < len(level1),
    }
    if report["output_level1_fraction"] > max_fraction + 1e-15:
        raise AssertionError("Level1 cap invariant was not satisfied")
    return selected, report
