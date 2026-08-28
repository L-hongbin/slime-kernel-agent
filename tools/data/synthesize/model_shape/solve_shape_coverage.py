#!/usr/bin/env python3
"""Generate Medium/Large input-shape siblings with a fail-closed static solver.

This is deliberately not a general operator-shape solver.  It changes one
whole input-factory dimension slot at a time, proves that returned-input
storage is affine in that slot, solves the resulting integer equation near a
deterministic target, and retains only children that pass both the existing
reference identity gate and a FakeTensor forward.  Candidate shapes are also
required to keep their largest dimension within 1,000 times their
second-largest dimension.  Directly mapped axes whose every ``forward`` use
selects a fixed item are excluded because enlarging them would create an unused
tail.

The default input and targets are the same 1,000-parent pilot used by the
DeepSeek-V4-Flash random-target run.  This makes the output an apples-to-apples
diagnostic while avoiding model generation entirely.  All source edits are
token-span replacements of existing positive integer literals; comments,
docstrings, formatting, ``Model``, and ``get_init_inputs`` remain byte-for-byte
unchanged.  Runtime acceptance on the target GPU remains a separate gate.
"""

from __future__ import annotations

import argparse
import ast
import collections
import contextlib
import copy
import dataclasses
import difflib
import hashlib
import json
import logging
import math
import os
import shutil
import signal
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.cleaning.pipeline import inspect_row_schema
from tools.data.synthesize.augment_prompt_tasks import (
    _SHAPE_FACTORIES,
    _call_name,
    _factory_records,
    _module_constant_environment,
    _replace_reference,
    _section_hashes,
    _shape_nodes,
    _top_level_function,
    analyze_code,
)
from tools.data.synthesize.model_shape.shape_contract import (
    LARGE_INPUT_MAX_BYTES,
    MEDIUM_INPUT_MAX_BYTES,
    MEDIUM_INPUT_MIN_BYTES,
    MIN_INPUT_SCALE,
    TARGET_SALT,
    _target_input_bytes,
    _validate_target_proximity,
    _validate_variant_storage,
    static_gate,
)

CONTRACT_VERSION = "shape_affine_solver_v3"
GENERATOR_VERSION = "whole_dimension_fixed_record_balance_solver_guard_fakecopy_timeout_v6"
DEFAULT_SELECTED = _REPO_ROOT / "Data/prompt_tvm_v4/shape_ai_random_targets_low_tp8_v7/run.1000/selected.parquet"
DEFAULT_RUN_DIR = _REPO_ROOT / "Data/prompt_tvm_v4/shape_solver_random_targets_v3/run.1000"
DEFAULT_TARGET_MAP_SHA256 = "b60b37a5f0f43e752575ae7cec760ba82e864ef2b854deccf4dc0673c721be0f"
VARIANTS = ("medium", "large")
MAX_REVIEW_EXAMPLES = 16
MIB = 1024**2
TARGET_ERROR_EQUIVALENCE_DENOMINATOR = 1_000
DEFAULT_FAKE_GATE_TIMEOUT_SECONDS = 30.0
MAX_DIMENSION_IMBALANCE_RATIO = 1_000

TARGET_SCHEMA = pa.schema(
    [
        pa.field("selected_index", pa.int64()),
        pa.field("parent_uuid", pa.string()),
        pa.field("parent_reference_sha256", pa.string()),
        pa.field("parent_input_bytes", pa.int64()),
        pa.field("variant", pa.string()),
        pa.field("target_lower_mib", pa.int32()),
        pa.field("target_upper_mib", pa.int32()),
        pa.field("target_input_bytes", pa.int64()),
    ]
)


@dataclasses.dataclass(frozen=True, order=True)
class SourceSpan:
    """UTF-8 byte coordinates reported by the Python AST."""

    lineno: int
    col_offset: int
    end_lineno: int
    end_col_offset: int

    @classmethod
    def from_node(cls, node: ast.AST) -> SourceSpan:
        values = (
            getattr(node, "lineno", None),
            getattr(node, "col_offset", None),
            getattr(node, "end_lineno", None),
            getattr(node, "end_col_offset", None),
        )
        if not all(isinstance(value, int) for value in values):
            raise ValueError("shape_slot_missing_source_span")
        span = cls(*values)  # type: ignore[arg-type]
        if span.lineno != span.end_lineno:
            raise ValueError("shape_integer_literal_must_be_single_line")
        if span.col_offset >= span.end_col_offset:
            raise ValueError("shape_integer_literal_has_empty_source_span")
        return span

    def as_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ShapeOccurrence:
    factory_index: int
    factory_name: str
    axis: int
    rank: int
    axis_from_right: int
    source_span: SourceSpan

    def as_dict(self) -> dict[str, Any]:
        return {
            "factory_index": self.factory_index,
            "factory_name": self.factory_name,
            "axis": self.axis,
            "rank": self.rank,
            "axis_from_right": self.axis_from_right,
            "source_span": self.source_span.as_dict(),
        }


@dataclasses.dataclass(frozen=True)
class ShapeSlot:
    slot_id: str
    kind: str
    old_value: int
    patch_spans: tuple[SourceSpan, ...]
    occurrences: tuple[ShapeOccurrence, ...]
    symbol_name: str | None = None
    symbol_scope: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "kind": self.kind,
            "old_value": self.old_value,
            "patch_spans": [span.as_dict() for span in self.patch_spans],
            "occurrences": [occurrence.as_dict() for occurrence in self.occurrences],
            "symbol_name": self.symbol_name,
            "symbol_scope": self.symbol_scope,
        }


@dataclasses.dataclass(frozen=True)
class AffineProfile:
    slot: ShapeSlot
    fixed_bytes: int
    bytes_per_slot_unit: int

    def input_bytes(self, value: int) -> int:
        return self.fixed_bytes + self.bytes_per_slot_unit * value


@dataclasses.dataclass(frozen=True)
class SolvedCandidate:
    variant: str
    target_input_bytes: int
    slot_value: int
    input_bytes_after: int
    target_delta_bytes: int
    target_relative_error: float
    profile: AffineProfile
    dimension_balance_value_bounds: Mapping[str, Any]


@dataclasses.dataclass(frozen=True)
class FakeGateResult:
    status: str
    reason: str | None = None
    exception_type: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class FakeInputProfileResult:
    status: str
    input_bytes: int | None = None
    tensor_count: int = 0
    tensors: tuple[tuple[tuple[int, ...], str, int], ...] = ()
    reason: str | None = None
    exception_type: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == "passed"

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class _FakeGateRuntime:
    torch: Any
    fake_copy_mode_type: Any
    fake_tensor_mode_type: Any
    data_dependent_types: tuple[Any, ...]
    unsupported_types: tuple[Any, ...]


_FAKE_GATE_RUNTIME: _FakeGateRuntime | FakeGateResult | None = None


class _FakeGateWallClockTimeout(BaseException):
    """Bypass ordinary ``except Exception`` blocks inside model/framework code."""


@contextlib.contextmanager
def _fake_gate_wall_clock_deadline(timeout_seconds: float) -> Any:
    """Apply a per-call Unix wall-clock deadline and restore process signal state."""

    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("fake_gate_timeout_seconds_must_be_finite_and_positive")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def raise_timeout(_signum: int, _frame: Any) -> None:
        raise _FakeGateWallClockTimeout

    armed = False
    try:
        signal.signal(signal.SIGALRM, raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, timeout_seconds)
        armed = True
        yield
    finally:
        try:
            if armed:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
        finally:
            signal.signal(signal.SIGALRM, previous_handler)
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


@dataclasses.dataclass(frozen=True)
class RunPaths:
    selected: Path
    selection: Path
    targets: Path
    children: Path
    paired: Path
    manifest: Path
    review: Path


def _run_paths(run_dir: Path) -> RunPaths:
    return RunPaths(
        selected=run_dir / "selected.parquet",
        selection=run_dir / "selection.json",
        targets=run_dir / "targets.parquet",
        children=run_dir / "static" / "children.parquet",
        paired=run_dir / "static" / "paired.parquet",
        manifest=run_dir / "static" / "manifest.json",
        review=run_dir / "analysis" / "review_samples.md",
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _logical_target_map_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    canonical = "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records).encode(
        "utf-8"
    )
    return _sha256_bytes(canonical)


def _nested(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _positive_integer_constant(node: ast.AST) -> int | None:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
        and node.value > 0
    ):
        return node.value
    return None


def _direct_name_assignment(
    tree: ast.Module,
    get_inputs: ast.FunctionDef,
    name: str,
    *,
    allow_destructuring: bool = False,
) -> tuple[str, ast.Constant] | None:
    """Find one literal binding without changing expression topology.

    The opt-in model lane also recognizes tuple unpacking (for example
    ``height, width = 32, 32``).  Historical static-solver callers retain the
    narrower simple-assignment behavior by default.
    """

    def bound_literal(target: ast.AST, value: ast.AST) -> ast.Constant | None:
        if isinstance(target, ast.Name):
            if target.id == name and _positive_integer_constant(value) is not None:
                assert isinstance(value, ast.Constant)
                return value
            return None
        if (
            allow_destructuring
            and isinstance(target, (ast.Tuple, ast.List))
            and isinstance(value, (ast.Tuple, ast.List))
        ):
            if len(target.elts) != len(value.elts):
                return None
            matches = [
                match
                for target_item, value_item in zip(target.elts, value.elts, strict=True)
                if (match := bound_literal(target_item, value_item)) is not None
            ]
            return matches[0] if len(matches) == 1 else None
        return None

    def direct_matches(statements: Sequence[ast.stmt]) -> list[ast.Constant]:
        matches: list[ast.Constant] = []
        for statement in statements:
            target: ast.AST | None = None
            value: ast.AST | None = None
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target, value = statement.targets[0], statement.value
            elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
                target, value = statement.target, statement.value
            if target is not None and value is not None:
                match = bound_literal(target, value)
                if match is not None:
                    matches.append(match)
        return matches

    local_stores = [
        node
        for node in ast.walk(get_inputs)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == name
    ]
    local_matches = direct_matches(get_inputs.body)
    if local_stores:
        if len(local_stores) == 1 and len(local_matches) == 1:
            return "get_inputs", local_matches[0]
        return None

    module_matches = direct_matches(tree.body)
    if len(module_matches) == 1:
        return "module", module_matches[0]
    return None


def _slot_id(kind: str, old_value: int, spans: Sequence[SourceSpan]) -> str:
    payload = {
        "kind": kind,
        "old_value": old_value,
        "spans": [span.as_dict() for span in sorted(spans)],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return f"slot_{_sha256_bytes(encoded)[:16]}"


def _raw_shape_slots(
    code: str,
    *,
    allow_coupled_consumers: bool = False,
) -> list[ShapeSlot]:
    """Return fail-closed whole-dimension literals and linked name slots."""

    tree = ast.parse(code)
    get_inputs_node = _top_level_function(tree, "get_inputs")
    if not isinstance(get_inputs_node, ast.FunctionDef):
        raise ValueError("get_inputs_must_not_be_async")
    get_inputs = get_inputs_node

    calls = sorted(
        (
            node
            for node in ast.walk(get_inputs)
            if isinstance(node, ast.Call) and _call_name(node.func) in _SHAPE_FACTORIES
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    direct: list[tuple[int, ShapeOccurrence]] = []
    names: dict[str, list[tuple[ast.Name, ShapeOccurrence]]] = collections.defaultdict(list)
    for factory_index, call in enumerate(calls):
        factory_name = _call_name(call.func)
        dimensions = _shape_nodes(call, factory_name)
        rank = len(dimensions)
        for axis, dimension in enumerate(dimensions):
            occurrence = ShapeOccurrence(
                factory_index=factory_index,
                factory_name=factory_name,
                axis=axis,
                rank=rank,
                axis_from_right=rank - axis - 1,
                source_span=SourceSpan.from_node(dimension),
            )
            literal = _positive_integer_constant(dimension)
            if literal is not None:
                direct.append((literal, occurrence))
            elif isinstance(dimension, ast.Name) and isinstance(dimension.ctx, ast.Load):
                names[dimension.id].append((dimension, occurrence))

    slots: list[ShapeSlot] = []
    for old_value, occurrence in direct:
        spans = (occurrence.source_span,)
        slots.append(
            ShapeSlot(
                slot_id=_slot_id("literal", old_value, spans),
                kind="literal",
                old_value=old_value,
                patch_spans=spans,
                occurrences=(occurrence,),
            )
        )

    grouped_direct: dict[tuple[int, int], list[ShapeOccurrence]] = collections.defaultdict(list)
    for old_value, occurrence in direct:
        grouped_direct[(occurrence.axis_from_right, old_value)].append(occurrence)
    for (_axis_from_right, old_value), occurrences in grouped_direct.items():
        if len(occurrences) < 2:
            continue
        factory_indices = [occurrence.factory_index for occurrence in occurrences]
        if len(factory_indices) != len(set(factory_indices)):
            continue
        spans = tuple(sorted(occurrence.source_span for occurrence in occurrences))
        slots.append(
            ShapeSlot(
                slot_id=_slot_id("shared_trailing_axis_literals", old_value, spans),
                kind="shared_trailing_axis_literals",
                old_value=old_value,
                patch_spans=spans,
                occurrences=tuple(occurrences),
            )
        )

    for name, named_occurrences in names.items():
        assignment = _direct_name_assignment(
            tree,
            get_inputs,
            name,
            allow_destructuring=allow_coupled_consumers,
        )
        if assignment is None:
            continue
        scope, value_node = assignment
        old_value = _positive_integer_constant(value_node)
        assert old_value is not None
        allowed_load_ids = {id(node) for node, _ in named_occurrences}
        if not allow_coupled_consumers:
            get_inputs_loads = [
                node
                for node in ast.walk(get_inputs)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == name
            ]
            if any(id(node) not in allowed_load_ids for node in get_inputs_loads):
                continue
            if scope == "module":
                tree_loads = [
                    node
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == name
                ]
                if any(id(node) not in allowed_load_ids for node in tree_loads):
                    continue
        patch_span = SourceSpan.from_node(value_node)
        slots.append(
            ShapeSlot(
                slot_id=_slot_id("linked_name", old_value, (patch_span,)),
                kind="linked_name",
                old_value=old_value,
                patch_spans=(patch_span,),
                occurrences=tuple(occurrence for _, occurrence in named_occurrences),
                symbol_name=name,
                symbol_scope=scope,
            )
        )

    by_spans: dict[tuple[SourceSpan, ...], ShapeSlot] = {}
    for slot in sorted(slots, key=lambda value: value.slot_id):
        key = tuple(sorted(slot.patch_spans))
        by_spans.setdefault(key, slot)
    return list(by_spans.values())


def _integer_constant(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return int(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and type(node.operand.value) is int
    ):
        value = int(node.operand.value)
        return value if isinstance(node.op, ast.UAdd) else -value
    return None


def _subscript_axis_component(
    slice_node: ast.AST,
    *,
    rank: int,
    axis: int,
) -> ast.AST | None:
    """Map a direct subscript component to a tensor axis, if unambiguous."""

    components = list(slice_node.elts) if isinstance(slice_node, ast.Tuple) else [slice_node]
    ellipses = [
        index
        for index, component in enumerate(components)
        if isinstance(component, ast.Constant) and component.value is Ellipsis
    ]
    if len(ellipses) > 1:
        return None

    def consumes_axis(component: ast.AST) -> bool:
        return not (isinstance(component, ast.Constant) and (component.value is None or component.value is Ellipsis))

    explicit_axes = sum(consumes_axis(component) for component in components)
    if explicit_axes > rank:
        return None
    if ellipses:
        expansion = rank - explicit_axes
        expanded = components[: ellipses[0]] + [ast.Slice()] * expansion + components[ellipses[0] + 1 :]
    else:
        expanded = components + [ast.Slice()] * (rank - explicit_axes)

    tensor_axis = 0
    for component in expanded:
        if not consumes_axis(component):
            continue
        if tensor_axis == axis:
            return component
        tensor_axis += 1
    return None


def _returned_factory_parameters(
    tree: ast.Module,
    get_inputs: ast.FunctionDef,
    entry_point: str,
    calls: Sequence[ast.Call],
) -> dict[int, str]:
    """Map simple returned factory origins to positional ``forward`` args."""

    model_classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == entry_point]
    if len(model_classes) != 1:
        return {}
    forwards = [
        node
        for node in model_classes[0].body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "forward"
    ]
    if len(forwards) != 1:
        return {}
    positional = list(forwards[0].args.posonlyargs) + list(forwards[0].args.args)
    if not positional:
        return {}
    parameters = [argument.arg for argument in positional[1:]]

    returns = [statement for statement in get_inputs.body if isinstance(statement, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0].value, (ast.List, ast.Tuple)):
        return {}
    returned = list(returns[0].value.elts)

    assignments: dict[str, list[ast.AST]] = collections.defaultdict(list)
    for statement in get_inputs.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target, value = statement.targets[0], statement.value
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            target, value = statement.target, statement.value
        else:
            continue
        if isinstance(target, ast.Name):
            assignments[target.id].append(value)

    call_indices = {id(call): index for index, call in enumerate(calls)}

    def origins(node: ast.AST, resolving: frozenset[str] = frozenset()) -> set[int]:
        if isinstance(node, ast.Call) and id(node) in call_indices:
            return {call_indices[id(node)]}
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            values = assignments.get(node.id, [])
            if len(values) == 1 and node.id not in resolving:
                return origins(values[0], resolving | {node.id})
            return set()
        found: set[int] = set()
        for child in ast.iter_child_nodes(node):
            found.update(origins(child, resolving))
        return found

    result: dict[int, str] = {}
    ambiguous: set[int] = set()
    for return_index, expression in enumerate(returned):
        if return_index >= len(parameters):
            break
        factory_origins = origins(expression)
        if len(factory_origins) != 1:
            continue
        factory_index = next(iter(factory_origins))
        if factory_index in result:
            ambiguous.add(factory_index)
        else:
            result[factory_index] = parameters[return_index]
    for factory_index in ambiguous:
        result.pop(factory_index, None)
    return result


def _fixed_record_axes(
    code: str,
    entry_point: str,
    slots: Sequence[ShapeSlot],
) -> dict[tuple[int, int], dict[str, Any]]:
    """Find directly mapped axes whose every forward use selects a fixed item."""

    tree = ast.parse(code)
    get_inputs_node = _top_level_function(tree, "get_inputs")
    if not isinstance(get_inputs_node, ast.FunctionDef):
        return {}
    calls = sorted(
        (
            node
            for node in ast.walk(get_inputs_node)
            if isinstance(node, ast.Call) and _call_name(node.func) in _SHAPE_FACTORIES
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    factory_parameters = _returned_factory_parameters(tree, get_inputs_node, entry_point, calls)
    if not factory_parameters:
        return {}

    model = next(
        (node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == entry_point),
        None,
    )
    if model is None:
        return {}
    forward = next(
        (
            node
            for node in model.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "forward"
        ),
        None,
    )
    if forward is None:
        return {}
    parents = {id(child): node for node in ast.walk(forward) for child in ast.iter_child_nodes(node)}

    axis_shapes: dict[tuple[int, int], tuple[int, int, str]] = {}
    for slot in slots:
        for occurrence in slot.occurrences:
            parameter = factory_parameters.get(occurrence.factory_index)
            if parameter is not None:
                axis_shapes[(occurrence.factory_index, occurrence.axis)] = (
                    occurrence.rank,
                    slot.old_value,
                    parameter,
                )

    fixed_axes: dict[tuple[int, int], dict[str, Any]] = {}
    for key, (rank, old_value, parameter) in axis_shapes.items():
        if any(
            isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id == parameter
            for node in ast.walk(forward)
        ):
            continue
        loads = [
            node
            for node in ast.walk(forward)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == parameter
        ]
        if not loads:
            continue
        indices: set[int] = set()
        all_uses_fixed = True
        for load in loads:
            parent = parents.get(id(load))
            if not isinstance(parent, ast.Subscript) or parent.value is not load:
                all_uses_fixed = False
                break
            component = _subscript_axis_component(parent.slice, rank=rank, axis=key[1])
            index = _integer_constant(component) if component is not None else None
            if index is None or not -old_value <= index < old_value:
                all_uses_fixed = False
                break
            indices.add(index % old_value)
        if all_uses_fixed:
            fixed_axes[key] = {
                "factory_index": key[0],
                "axis": key[1],
                "rank": rank,
                "parameter": parameter,
                "old_value": old_value,
                "fixed_indices": sorted(indices),
            }
    return fixed_axes


def _shape_slots_with_rejections(
    code: str,
    entry_point: str,
    *,
    allow_coupled_consumers: bool = False,
) -> tuple[list[ShapeSlot], list[dict[str, Any]]]:
    slots = _raw_shape_slots(
        code,
        allow_coupled_consumers=allow_coupled_consumers,
    )
    fixed_axes = _fixed_record_axes(code, entry_point, slots)
    accepted: list[ShapeSlot] = []
    rejected: list[dict[str, Any]] = []
    for slot in slots:
        guarded = [
            fixed_axes[(occurrence.factory_index, occurrence.axis)]
            for occurrence in slot.occurrences
            if (occurrence.factory_index, occurrence.axis) in fixed_axes
        ]
        if guarded:
            rejected.append(
                {
                    "slot_id": slot.slot_id,
                    "reason": "fixed_record_dead_tail_guard",
                    "guarded_axes": guarded,
                }
            )
        else:
            accepted.append(slot)
    return accepted, rejected


def _shape_slots(code: str) -> list[ShapeSlot]:
    """Compatibility accessor for analyzers that need every raw source slot."""

    return _raw_shape_slots(code)


def _affected_shape_balance_guard(
    code: str,
    slot: ShapeSlot,
    expected_slot_value: int,
) -> dict[str, Any]:
    """Fail closed unless every slot-affected final shape is reasonably balanced."""

    affected_axes: dict[int, set[int]] = collections.defaultdict(set)
    expected_ranks: dict[int, set[int]] = collections.defaultdict(set)
    for occurrence in slot.occurrences:
        affected_axes[occurrence.factory_index].add(occurrence.axis)
        expected_ranks[occurrence.factory_index].add(occurrence.rank)

    evidence: dict[str, Any] = {
        "passed": False,
        "resolved": False,
        "reason": None,
        "ratio_limit": MAX_DIMENSION_IMBALANCE_RATIO,
        "comparison": ("largest_dimension <= 1000 * second_largest_dimension"),
        "expected_slot_value": expected_slot_value,
        "affected_occurrence_count": len(slot.occurrences),
        "affected_factory_count": len(affected_axes),
        "affected_factories": [],
        "violations": [],
    }
    if expected_slot_value <= 0 or not affected_axes:
        evidence["reason"] = "invalid_or_empty_affected_slot"
        return evidence

    try:
        tree = ast.parse(code)
        get_inputs_node = _top_level_function(tree, "get_inputs")
        if not isinstance(get_inputs_node, ast.FunctionDef):
            raise ValueError("get_inputs_must_be_a_synchronous_function")
        calls = sorted(
            (
                node
                for node in ast.walk(get_inputs_node)
                if isinstance(node, ast.Call) and _call_name(node.func) in _SHAPE_FACTORIES
            ),
            key=lambda node: (node.lineno, node.col_offset),
        )
        records, _ = _factory_records(
            tree,
            get_inputs_node,
            _module_constant_environment(tree),
        )
        records_by_location: dict[tuple[int, int], Any] = {}
        for record in records:
            location = (record.line_number, record.column_offset)
            if location in records_by_location:
                raise ValueError(f"duplicate_direct_factory_location:{location[0]}:{location[1]}")
            records_by_location[location] = record

        for factory_index in sorted(affected_axes):
            if not 0 <= factory_index < len(calls):
                raise ValueError(f"affected_factory_index_out_of_range:{factory_index}:{len(calls)}")
            call = calls[factory_index]
            location = (call.lineno, call.col_offset)
            record = records_by_location.get(location)
            if record is None:
                raise ValueError(
                    f"affected_direct_factory_shape_unresolved:{factory_index}:"
                    f"line{call.lineno}:col{call.col_offset}"
                )
            if record.name != _call_name(call.func):
                raise ValueError(f"affected_direct_factory_identity_mismatch:{factory_index}")
            ranks = expected_ranks[factory_index]
            if ranks != {len(record.shape)}:
                raise ValueError(
                    f"affected_direct_factory_rank_mismatch:{factory_index}:" f"{sorted(ranks)}:{len(record.shape)}"
                )
            axes = sorted(affected_axes[factory_index])
            if any(not 0 <= axis < len(record.shape) for axis in axes):
                raise ValueError(
                    f"affected_direct_factory_axis_out_of_range:{factory_index}:" f"{axes}:{len(record.shape)}"
                )
            mismatched_axes = [axis for axis in axes if record.shape[axis] != expected_slot_value]
            if mismatched_axes:
                raise ValueError(
                    f"affected_direct_factory_slot_value_mismatch:{factory_index}:"
                    f"{mismatched_axes}:{expected_slot_value}"
                )

            shape = list(record.shape)
            factory_evidence: dict[str, Any] = {
                "factory_index": factory_index,
                "factory_name": record.name,
                "line_number": record.line_number,
                "column_offset": record.column_offset,
                "affected_axes": axes,
                "shape": shape,
                "rank": len(shape),
                "guard_applies": len(shape) >= 2,
            }
            if len(shape) >= 2:
                largest, second_largest = sorted(shape, reverse=True)[:2]
                balanced = largest <= MAX_DIMENSION_IMBALANCE_RATIO * second_largest
                factory_evidence.update(
                    {
                        "largest_dimension": largest,
                        "second_largest_dimension": second_largest,
                        "balanced": balanced,
                    }
                )
                if not balanced:
                    evidence["violations"].append(copy.deepcopy(factory_evidence))
            evidence["affected_factories"].append(factory_evidence)
    except (SyntaxError, TypeError, ValueError) as exc:
        evidence["reason"] = "affected_direct_factory_shape_resolution_failed:" f"{type(exc).__name__}:{exc}"
        return evidence

    evidence["resolved"] = True
    if evidence["violations"]:
        first = evidence["violations"][0]
        evidence["reason"] = (
            "dimension_balance_ratio_exceeded:"
            f"factory_index={first['factory_index']}:"
            f"largest={first['largest_dimension']}:"
            f"second_largest={first['second_largest_dimension']}:"
            f"limit={MAX_DIMENSION_IMBALANCE_RATIO}"
        )
        return evidence
    evidence["passed"] = True
    return evidence


def _dimension_balance_slot_value_bounds(
    code: str,
    slot: ShapeSlot,
) -> dict[str, Any]:
    """Derive the exact positive-integer interval accepted by the balance guard."""

    resolution = _affected_shape_balance_guard(code, slot, slot.old_value)
    result: dict[str, Any] = {
        "resolved": bool(resolution["resolved"]),
        "feasible": False,
        "reason": None,
        "ratio_limit": MAX_DIMENSION_IMBALANCE_RATIO,
        "minimum_slot_value": None,
        "maximum_slot_value": None,
        "factory_bounds": [],
    }
    if not resolution["resolved"]:
        result["reason"] = resolution["reason"]
        return result

    minimum = 1
    maximum: int | None = None
    for factory in resolution["affected_factories"]:
        shape = list(factory["shape"])
        affected_axes = set(factory["affected_axes"])
        factory_bound: dict[str, Any] = {
            "factory_index": factory["factory_index"],
            "affected_axes": sorted(affected_axes),
            "source_shape": shape,
            "guard_applies": len(shape) >= 2,
            "minimum_slot_value": 1,
            "maximum_slot_value": None,
        }
        if len(shape) >= 2:
            fixed_dimensions = sorted(
                (dimension for axis, dimension in enumerate(shape) if axis not in affected_axes),
                reverse=True,
            )
            if not affected_axes:
                result["reason"] = "affected_factory_has_no_affected_axes:" f"{factory['factory_index']}"
                return result

            factory_minimum = 1
            if fixed_dimensions:
                fixed_largest = fixed_dimensions[0]
                fixed_second_largest = fixed_dimensions[1] if len(fixed_dimensions) >= 2 else None
                fixed_pair_is_balanced = (
                    fixed_second_largest is not None
                    and fixed_largest <= MAX_DIMENSION_IMBALANCE_RATIO * fixed_second_largest
                )
                if not fixed_pair_is_balanced:
                    factory_minimum = _ceil_div(
                        fixed_largest,
                        MAX_DIMENSION_IMBALANCE_RATIO,
                    )
                factory_bound["fixed_largest_dimension"] = fixed_largest
                factory_bound["fixed_second_largest_dimension"] = fixed_second_largest
                if len(affected_axes) == 1:
                    factory_bound["maximum_slot_value"] = MAX_DIMENSION_IMBALANCE_RATIO * fixed_largest
            factory_bound["minimum_slot_value"] = factory_minimum
            minimum = max(minimum, factory_minimum)
            factory_maximum = factory_bound["maximum_slot_value"]
            if factory_maximum is not None:
                maximum = factory_maximum if maximum is None else min(maximum, factory_maximum)
        result["factory_bounds"].append(factory_bound)

    result["minimum_slot_value"] = minimum
    result["maximum_slot_value"] = maximum
    if maximum is not None and minimum > maximum:
        result["reason"] = "empty_dimension_balance_value_interval"
        return result
    result["feasible"] = True
    return result


def _patch_integer_spans(code: str, slot: ShapeSlot, new_value: int) -> str:
    if new_value <= 0:
        raise ValueError("shape_slot_value_must_be_positive")
    raw = code.encode("utf-8")
    lines = raw.splitlines(keepends=True)
    line_offsets: list[int] = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line)
    if raw and (not lines or offset != len(raw)):
        raise ValueError("source_line_offset_construction_failed")

    replacements: list[tuple[int, int, bytes]] = []
    for span in slot.patch_spans:
        if not 1 <= span.lineno <= len(lines):
            raise ValueError("shape_slot_source_line_out_of_range")
        start = line_offsets[span.lineno - 1] + span.col_offset
        end = line_offsets[span.end_lineno - 1] + span.end_col_offset
        token = raw[start:end]
        try:
            observed = ast.literal_eval(token.decode("utf-8"))
        except (SyntaxError, UnicodeDecodeError, ValueError) as exc:
            raise ValueError("shape_slot_source_span_is_not_a_literal") from exc
        if type(observed) is not int or observed != slot.old_value:
            raise ValueError(f"shape_slot_source_value_mismatch:{slot.old_value}:{observed!r}")
        replacements.append((start, end, str(new_value).encode("ascii")))

    replacements.sort(reverse=True)
    previous_start = len(raw) + 1
    patched = raw
    for start, end, replacement in replacements:
        if end > previous_start:
            raise ValueError("overlapping_shape_slot_source_spans")
        patched = patched[:start] + replacement + patched[end:]
        previous_start = start
    child = patched.decode("utf-8")
    ast.parse(child)
    return child


def _affine_profiles(
    code: str,
    entry_point: str,
    parent_input_bytes: int,
    slots: Sequence[ShapeSlot],
) -> tuple[list[AffineProfile], list[dict[str, Any]]]:
    profiles: list[AffineProfile] = []
    rejected: list[dict[str, Any]] = []
    for slot in slots:
        try:
            plus_one = analyze_code(_patch_integer_spans(code, slot, slot.old_value + 1), entry_point).input_bytes
            plus_two = analyze_code(_patch_integer_spans(code, slot, slot.old_value + 2), entry_point).input_bytes
            slope = plus_one - parent_input_bytes
            if slope <= 0:
                raise ValueError("shape_slot_does_not_increase_returned_input_storage")
            if plus_two - plus_one != slope:
                raise ValueError("shape_slot_storage_is_not_affine")
            fixed = parent_input_bytes - slope * slot.old_value
            if fixed < 0:
                raise ValueError("shape_slot_affine_fixed_storage_is_negative")
            if fixed + slope * slot.old_value != parent_input_bytes:
                raise ValueError("shape_slot_affine_reconstruction_failed")
            profiles.append(AffineProfile(slot, fixed, slope))
        except (SyntaxError, TypeError, ValueError) as exc:
            rejected.append(
                {
                    "slot_id": slot.slot_id,
                    "reason": f"{type(exc).__name__}:{exc}",
                }
            )
    return profiles, rejected


def _ceil_div(numerator: int, denominator: int) -> int:
    return -((-numerator) // denominator)


def _solve_profile(
    profile: AffineProfile,
    *,
    variant: str,
    target_input_bytes: int,
    parent_input_bytes: int,
    tie_salt: str,
    dimension_balance_value_bounds: Mapping[str, Any],
) -> SolvedCandidate | None:
    bounds_resolved = dimension_balance_value_bounds.get("resolved")
    bounds_feasible = dimension_balance_value_bounds.get("feasible")
    if not bounds_resolved or not bounds_feasible:
        return None
    if variant == "medium":
        band_lower, band_upper = MEDIUM_INPUT_MIN_BYTES, MEDIUM_INPUT_MAX_BYTES
    elif variant == "large":
        band_lower, band_upper = MEDIUM_INPUT_MAX_BYTES + 1, LARGE_INPUT_MAX_BYTES
    else:
        raise ValueError(f"unknown_shape_variant:{variant}")
    growth_lower = math.ceil(MIN_INPUT_SCALE * parent_input_bytes)
    lower = max(band_lower, growth_lower)
    upper = band_upper
    slope = profile.bytes_per_slot_unit
    fixed = profile.fixed_bytes
    minimum_value = max(1, _ceil_div(lower - fixed, slope))
    maximum_value = (upper - fixed) // slope
    balance_minimum = dimension_balance_value_bounds.get("minimum_slot_value")
    balance_maximum = dimension_balance_value_bounds.get("maximum_slot_value")
    if type(balance_minimum) is not int or balance_minimum <= 0:
        raise ValueError("invalid_dimension_balance_minimum_slot_value")
    if balance_maximum is not None and (type(balance_maximum) is not int or balance_maximum <= 0):
        raise ValueError("invalid_dimension_balance_maximum_slot_value")
    minimum_value = max(minimum_value, balance_minimum)
    if balance_maximum is not None:
        maximum_value = min(maximum_value, balance_maximum)
    if minimum_value > maximum_value:
        return None

    quotient = (target_input_bytes - fixed) // slope
    values = {
        minimum_value,
        maximum_value,
        max(minimum_value, min(maximum_value, quotient)),
        max(minimum_value, min(maximum_value, quotient + 1)),
    }

    def key(value: int) -> tuple[int, str]:
        after = profile.input_bytes(value)
        tie = _sha256_bytes(f"{tie_salt}:{variant}:{profile.slot.slot_id}:{value}".encode())
        return abs(after - target_input_bytes), tie

    value = min(values, key=key)
    after = profile.input_bytes(value)
    delta = after - target_input_bytes
    return SolvedCandidate(
        variant=variant,
        target_input_bytes=target_input_bytes,
        slot_value=value,
        input_bytes_after=after,
        target_delta_bytes=delta,
        target_relative_error=abs(delta) / target_input_bytes,
        profile=profile,
        dimension_balance_value_bounds=copy.deepcopy(dict(dimension_balance_value_bounds)),
    )


class _FakeDeviceNormalizer(ast.NodeTransformer):
    """Keep the FakeTensor compatibility probe device-agnostic."""

    def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
        if isinstance(node.value, str) and (node.value == "cuda" or node.value.startswith("cuda:")):
            return ast.copy_location(ast.Constant(value="cpu"), node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "cuda":
            node.func.attr = "to"
            node.args = [ast.Constant(value="cpu")]
            node.keywords = []
        return node


def _prepare_init_inputs(raw: Any, torch: Any) -> tuple[str, Any]:
    if isinstance(raw, Mapping):
        return "kwargs", dict(raw)
    if not isinstance(raw, (list, tuple)):
        raise TypeError("get_init_inputs() must return list, tuple, or mapping")
    values = list(raw)
    if len(values) > 1 and isinstance(values[0], (list, tuple)) and len(values[0]) == 0:
        if not isinstance(values[1], Mapping):
            raise TypeError("[[], kwargs] get_init_inputs convention requires a mapping")
        return "kwargs", dict(values[1])
    if any(isinstance(value, torch.Tensor) and value.device.type != "cpu" for value in values):
        raise ValueError("fake_init_input_device_normalization_failed")
    return "args", values


def _invoke_model(model: Any, raw_inputs: Any) -> Any:
    if isinstance(raw_inputs, Mapping):
        return model(**dict(raw_inputs))
    if not isinstance(raw_inputs, (list, tuple)):
        raise TypeError("get_inputs() must return list, tuple, or mapping")
    values = list(raw_inputs)
    if len(values) == 2 and isinstance(values[0], (list, tuple)) and isinstance(values[1], Mapping):
        return model(*list(values[0]), **dict(values[1]))
    return model(*values)


def _exception_chain(exc: BaseException) -> list[BaseException]:
    result: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        result.append(current)
        current = current.__cause__ or current.__context__
    return result


def _fake_tensor_gate(
    code: str,
    entry_point: str,
    *,
    timeout_seconds: float = DEFAULT_FAKE_GATE_TIMEOUT_SECONDS,
) -> FakeGateResult:
    """Initialize FakeTensor once, then time only this reference's work."""

    runtime = _load_fake_gate_runtime()
    if isinstance(runtime, FakeGateResult):
        return runtime

    try:
        with _fake_gate_wall_clock_deadline(timeout_seconds):
            return _fake_tensor_gate_task(code, entry_point, runtime)
    except _FakeGateWallClockTimeout:
        return FakeGateResult(
            "timeout",
            f"wall_clock_timeout_exceeded:{timeout_seconds:g}s",
            "_FakeGateWallClockTimeout",
        )


def _fake_input_profile(
    code: str,
    *,
    timeout_seconds: float = DEFAULT_FAKE_GATE_TIMEOUT_SECONDS,
) -> FakeInputProfileResult:
    """Measure tensors actually returned by ``get_inputs`` without allocation."""

    runtime = _load_fake_gate_runtime()
    if isinstance(runtime, FakeGateResult):
        return FakeInputProfileResult(
            "failed",
            reason=runtime.reason,
            exception_type=runtime.exception_type,
        )
    try:
        with _fake_gate_wall_clock_deadline(timeout_seconds):
            return _fake_input_profile_task(code, runtime)
    except _FakeGateWallClockTimeout:
        return FakeInputProfileResult(
            "timeout",
            reason=f"wall_clock_timeout_exceeded:{timeout_seconds:g}s",
            exception_type="_FakeGateWallClockTimeout",
        )


def _load_fake_gate_runtime() -> _FakeGateRuntime | FakeGateResult:
    """Load and minimally warm process-global FakeTensor state outside deadlines."""

    global _FAKE_GATE_RUNTIME
    if _FAKE_GATE_RUNTIME is not None:
        return _FAKE_GATE_RUNTIME

    try:
        import torch
        from torch._subclasses.fake_tensor import (
            DataDependentOutputException,
            DynamicOutputShapeException,
            FakeCopyMode,
            FakeTensorMode,
            UnsupportedFakeTensorException,
            UnsupportedOperatorException,
        )
        from torch.fx.experimental.symbolic_shapes import GuardOnDataDependentSymNode

        logging.getLogger("torch._subclasses.fake_tensor").setLevel(logging.CRITICAL)
        fake_mode = FakeTensorMode(
            allow_fallback_kernels=False,
            allow_non_fake_inputs=False,
            static_shapes=True,
        )
        # Trigger process-global dispatcher/decomposition initialization before
        # any sample deadline.  The warm-up remains allocation-free.
        with fake_mode, FakeCopyMode(fake_mode), torch.no_grad():
            warmup_model = torch.nn.Linear(1, 1)
            warmup_copy = copy.deepcopy(warmup_model)
            warmup_copy(torch.empty((1, 1)))
        _FAKE_GATE_RUNTIME = _FakeGateRuntime(
            torch=torch,
            fake_copy_mode_type=FakeCopyMode,
            fake_tensor_mode_type=FakeTensorMode,
            data_dependent_types=(
                DataDependentOutputException,
                DynamicOutputShapeException,
                GuardOnDataDependentSymNode,
            ),
            unsupported_types=(
                UnsupportedFakeTensorException,
                UnsupportedOperatorException,
                NotImplementedError,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - cached infrastructure failure
        message = str(exc).replace("\n", " ")[:1000]
        _FAKE_GATE_RUNTIME = FakeGateResult(
            "failed",
            f"fake_gate_runtime_initialization_failed:{message}",
            type(exc).__name__,
        )
    return _FAKE_GATE_RUNTIME


def _fake_tensor_gate_task(
    code: str,
    entry_point: str,
    runtime: _FakeGateRuntime,
) -> FakeGateResult:
    """Run task-specific compile/model/input/forward work under the deadline."""

    torch = runtime.torch
    try:
        # Shape mismatches are expected while trying alternate slots. PyTorch's
        # FakeTensor dispatcher logs a full traceback before re-raising each
        # such mismatch; the structured attempt record below is the evidence we
        # want, so keep CLI/shard logs bounded.
        parsed = _FakeDeviceNormalizer().visit(ast.parse(code, filename="<shape_solver>"))
        ast.fix_missing_locations(parsed)
        namespace: dict[str, Any] = {"__name__": "__shape_solver_fake__"}
        fake_mode = runtime.fake_tensor_mode_type(
            # A fallback materializes real tensors on the normalized CPU
            # device.  Candidate inputs may be as large as 4 GiB, so keep
            # this gate genuinely allocation-free and fail closed instead.
            allow_fallback_kernels=False,
            allow_non_fake_inputs=False,
            static_shapes=True,
        )
        # FakeCopyMode must cover construction as well as our explicit copy:
        # Transformer-style modules deepcopy sublayers inside __init__.
        with fake_mode, runtime.fake_copy_mode_type(fake_mode), torch.no_grad():
            exec(compile(parsed, "<shape_solver>", "exec"), namespace)  # noqa: S102
            model_type = namespace.get(entry_point)
            get_inputs = namespace.get("get_inputs")
            get_init_inputs = namespace.get("get_init_inputs")
            if not isinstance(model_type, type) or not issubclass(model_type, torch.nn.Module):
                raise TypeError("entry point must be a torch.nn.Module class")
            if not callable(get_inputs) or not callable(get_init_inputs):
                raise TypeError("reference must define get_inputs() and get_init_inputs()")
            init_kind, init_inputs = _prepare_init_inputs(get_init_inputs(), torch)
            model = model_type(*init_inputs) if init_kind == "args" else model_type(**init_inputs)
            raw_inputs = get_inputs()
            copied_model = copy.deepcopy(model)
            copied_model.train(True)
            _invoke_model(copied_model, raw_inputs)
        return FakeGateResult("passed")
    except Exception as exc:  # noqa: BLE001 - classification is fail-closed
        chain = _exception_chain(exc)
        exception_type = type(exc).__name__
        message = str(exc).replace("\n", " ")[:1000]
        if any(isinstance(item, runtime.data_dependent_types) for item in chain):
            return FakeGateResult(
                "unsupported",
                f"data_dependent_shape:{message}",
                exception_type,
            )
        if any(isinstance(item, runtime.unsupported_types) for item in chain):
            return FakeGateResult(
                "unsupported",
                f"fake_tensor_operator_unsupported:{message}",
                exception_type,
            )
        return FakeGateResult(
            "failed",
            f"fake_forward_incompatible:{message}",
            exception_type,
        )


def _fake_input_profile_task(
    code: str,
    runtime: _FakeGateRuntime,
) -> FakeInputProfileResult:
    """Execute only ``get_inputs`` and profile its concrete FakeTensor leaves."""

    torch = runtime.torch
    try:
        parsed = _FakeDeviceNormalizer().visit(ast.parse(code, filename="<shape_input_profile>"))
        ast.fix_missing_locations(parsed)
        namespace: dict[str, Any] = {"__name__": "__shape_input_profile__"}
        fake_mode = runtime.fake_tensor_mode_type(
            allow_fallback_kernels=False,
            allow_non_fake_inputs=False,
            static_shapes=True,
        )
        with fake_mode, runtime.fake_copy_mode_type(fake_mode), torch.no_grad():
            exec(compile(parsed, "<shape_input_profile>", "exec"), namespace)  # noqa: S102
            get_inputs = namespace.get("get_inputs")
            if not callable(get_inputs):
                raise TypeError("reference must define get_inputs()")
            raw_inputs = get_inputs()
            tensors: list[Any] = []
            tensor_ids: set[int] = set()
            container_ids: set[int] = set()

            def collect(value: Any) -> None:
                if isinstance(value, torch.Tensor):
                    if id(value) in tensor_ids:
                        raise ValueError("returned_input_tensor_alias_or_duplicate")
                    tensor_ids.add(id(value))
                    tensors.append(value)
                    return
                if isinstance(value, Mapping):
                    if id(value) in container_ids:
                        raise ValueError("cyclic_returned_input_container")
                    container_ids.add(id(value))
                    for item in value.values():
                        collect(item)
                    container_ids.remove(id(value))
                    return
                if isinstance(value, (list, tuple)):
                    if id(value) in container_ids:
                        raise ValueError("cyclic_returned_input_container")
                    container_ids.add(id(value))
                    for item in value:
                        collect(item)
                    container_ids.remove(id(value))
                    return
                if isinstance(value, (str, bytes, int, float, complex, bool, type(None))):
                    return
                raise TypeError(f"unsupported returned input leaf: {type(value).__name__}")

            collect(raw_inputs)
            if not tensors:
                raise ValueError("get_inputs_returned_no_tensor")
            profiles: list[tuple[tuple[int, ...], str, int]] = []
            for tensor in tensors:
                shape = tuple(int(value) for value in tensor.shape)
                tensor_bytes = int(tensor.numel()) * int(tensor.element_size())
                profiles.append((shape, str(tensor.dtype), tensor_bytes))
        return FakeInputProfileResult(
            "passed",
            input_bytes=sum(item[2] for item in profiles),
            tensor_count=len(profiles),
            tensors=tuple(profiles),
        )
    except Exception as exc:  # noqa: BLE001 - classification is fail-closed
        chain = _exception_chain(exc)
        exception_type = type(exc).__name__
        message = str(exc).replace("\n", " ")[:1000]
        if any(isinstance(item, runtime.data_dependent_types) for item in chain):
            return FakeInputProfileResult(
                "unsupported",
                reason=f"data_dependent_shape:{message}",
                exception_type=exception_type,
            )
        if any(isinstance(item, runtime.unsupported_types) for item in chain):
            return FakeInputProfileResult(
                "unsupported",
                reason=f"fake_tensor_operator_unsupported:{message}",
                exception_type=exception_type,
            )
        return FakeInputProfileResult(
            "failed",
            reason=f"fake_input_profile_failed:{message}",
            exception_type=exception_type,
        )


def _make_child(
    parent: Mapping[str, Any],
    child_code: str,
    static: Mapping[str, Any],
) -> dict[str, Any]:
    child = copy.deepcopy(dict(parent))
    parent_code = _nested(parent, "reward_model.ground_truth")
    parent_uuid = _nested(parent, "extra_info.uuid")
    entry_point = str(_nested(parent, "extra_info.entry_point", "Model"))
    if not isinstance(parent_code, str) or not isinstance(parent_uuid, str):
        raise ValueError("parent_requires_reference_and_uuid")
    child_hash = str(static["child_reference_sha256"])
    child_uuid = f"shapesolver_{_sha256_bytes(f'{parent_uuid}:{child_hash}'.encode())[:24]}"
    child["reward_model"]["ground_truth"] = child_code
    child["prompt"] = _replace_reference(child.get("prompt"), parent_code, child_code, required=True)
    extra = child["extra_info"]
    if extra.get("original_prompt") is not None:
        extra["original_prompt"] = _replace_reference(
            extra.get("original_prompt"), parent_code, child_code, required=False
        )
    extra["uuid"] = child_uuid
    v4 = dict(extra.get("v4") or {})
    v4["parent_uuid"] = parent_uuid
    v4["reference_sha256"] = child_hash
    v4["normalized_ast_sha256"] = str(static["child_normalized_ast_sha256"])
    v4["included_in_review_train"] = False
    v4["runtime_validation_status"] = "shape_solver_static_fake_pass_runtime_validation_required"
    v4["governance_status"] = "shape_solver_review_only"
    extra["v4"] = v4
    fatal, _ = inspect_row_schema(child, child_code)
    if fatal:
        raise ValueError(f"child_row_schema_validation_failed:{fatal}")
    if _top_level_function(ast.parse(child_code), "get_init_inputs").name != "get_init_inputs":
        raise ValueError("child_get_init_inputs_missing")
    if entry_point not in {node.name for node in ast.parse(child_code).body if isinstance(node, ast.ClassDef)}:
        raise ValueError("child_entry_point_missing")
    return child


def _candidate_key(
    candidate: SolvedCandidate,
    parent_uuid: str,
) -> tuple[int, float, int, int, str]:
    error_bucket_bytes = max(1, candidate.target_input_bytes // TARGET_ERROR_EQUIVALENCE_DENOMINATOR)
    relative_slot_growth = candidate.slot_value / candidate.profile.slot.old_value
    propagated_factories = len({occurrence.factory_index for occurrence in candidate.profile.slot.occurrences})
    tie = _sha256_bytes(f"{parent_uuid}:{candidate.variant}:{candidate.profile.slot.slot_id}".encode())
    return (
        abs(candidate.target_delta_bytes) // error_bucket_bytes,
        relative_slot_growth,
        -propagated_factories,
        abs(candidate.target_delta_bytes),
        tie,
    )


def _candidate_manifest(candidate: SolvedCandidate) -> dict[str, Any]:
    profile = candidate.profile
    return {
        "variant": candidate.variant,
        "target_input_bytes": candidate.target_input_bytes,
        "slot_value": candidate.slot_value,
        "input_bytes_after": candidate.input_bytes_after,
        "target_delta_bytes": candidate.target_delta_bytes,
        "target_relative_error": candidate.target_relative_error,
        "fixed_bytes": profile.fixed_bytes,
        "bytes_per_slot_unit": profile.bytes_per_slot_unit,
        "slot_growth_factor": candidate.slot_value / profile.slot.old_value,
        "propagated_factory_count": len({occurrence.factory_index for occurrence in profile.slot.occurrences}),
        "target_error_equivalence_bytes": max(
            1,
            candidate.target_input_bytes // TARGET_ERROR_EQUIVALENCE_DENOMINATOR,
        ),
        "dimension_balance_value_bounds": copy.deepcopy(dict(candidate.dimension_balance_value_bounds)),
        "slot": profile.slot.as_dict(),
    }


def _review_markdown(records: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Shape solver accepted diffs",
        "",
        "Representative source-span-only edits. These are static/FakeTensor candidates, not H20 acceptance.",
        "",
    ]
    for record in records[:MAX_REVIEW_EXAMPLES]:
        lines.extend(
            [
                f"## {record['child_uuid']} ({record['variant']})",
                "",
                (
                    f"Parent `{record['parent_uuid']}`; slot `{record['slot_id']}`; "
                    f"{record['input_bytes_before']} -> {record['input_bytes_after']} bytes; "
                    f"target error {record['target_relative_error']:.6%}."
                ),
                "",
                "```diff",
            ]
        )
        diff = difflib.unified_diff(
            str(record["parent_code"]).splitlines(),
            str(record["child_code"]).splitlines(),
            fromfile="parent.py",
            tofile="child.py",
            lineterm="",
        )
        lines.extend(diff)
        lines.extend(["```", ""])
    return "\n".join(lines).rstrip() + "\n"


def solve_shape_coverage(
    selected_path: Path,
    run_dir: Path,
    *,
    max_parents: int | None = None,
    fake_gate_timeout_seconds: float = DEFAULT_FAKE_GATE_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    if not selected_path.is_file():
        raise FileNotFoundError(selected_path)
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {run_dir}")
    if max_parents is not None and max_parents <= 0:
        raise ValueError("max_parents must be positive")
    if not math.isfinite(fake_gate_timeout_seconds) or fake_gate_timeout_seconds <= 0:
        raise ValueError("fake_gate_timeout_seconds_must_be_finite_and_positive")

    source = pq.read_table(selected_path)
    selected = source if max_parents is None else source.slice(0, max_parents)
    rows = selected.to_pylist()
    final_paths = _run_paths(run_dir.resolve())
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{run_dir.name}.tmp-", dir=run_dir.parent))
    temporary_paths = _run_paths(temporary_dir)
    temporary_paths.children.parent.mkdir(parents=True, exist_ok=True)
    temporary_paths.review.parent.mkdir(parents=True, exist_ok=True)

    children: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    target_map_records: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    review_records: list[dict[str, Any]] = []
    counters: collections.Counter[str] = collections.Counter()
    skip_reasons: collections.Counter[str] = collections.Counter()
    parents_with_children: set[str] = set()

    try:
        if selected.num_rows == source.num_rows:
            shutil.copy2(selected_path, temporary_paths.selected)
        else:
            pq.write_table(selected, temporary_paths.selected, compression="zstd")

        source_selection_path = selected_path.with_name("selection.json")
        source_selection: dict[str, Any] = {}
        if source_selection_path.is_file():
            loaded_selection = json.loads(source_selection_path.read_text(encoding="utf-8"))
            if not isinstance(loaded_selection, dict):
                raise ValueError("source_selection_manifest_must_be_an_object")
            source_selection = loaded_selection
        selection = {
            **source_selection,
            "derivation_contract_version": CONTRACT_VERSION,
            "source_selection_manifest": (
                str(source_selection_path.resolve()) if source_selection_path.is_file() else None
            ),
            "source_selection_manifest_sha256": (
                _sha256_file(source_selection_path) if source_selection_path.is_file() else None
            ),
            "selected_path": str(final_paths.selected),
            "selected_sha256": _sha256_file(temporary_paths.selected),
            "selected_count": len(rows),
            "rows": list(source_selection.get("rows", []))[: len(rows)],
        }
        temporary_paths.selection.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for source_row_index, parent in enumerate(rows):
            counters["parents_scanned"] += 1
            parent_uuid = str(_nested(parent, "extra_info.uuid", ""))
            parent_code = _nested(parent, "reward_model.ground_truth")
            entry_point = str(_nested(parent, "extra_info.entry_point", "Model"))
            if not parent_uuid or not isinstance(parent_code, str):
                skip_reasons["parent_missing_uuid_or_reference"] += 1
                continue
            parent_reference_hash = _sha256_bytes(parent_code.encode("utf-8"))
            try:
                parent_analysis = analyze_code(parent_code, entry_point)
                target_map = _target_input_bytes(parent)
            except (SyntaxError, TypeError, ValueError) as exc:
                reason = f"parent_static_analysis_failed:{type(exc).__name__}:{exc}"
                skip_reasons[reason] += 1
                continue
            target_map_records.append(
                {
                    "selected_index": source_row_index,
                    "parent_uuid": parent_uuid,
                    "parent_reference_sha256": parent_reference_hash,
                    "target_input_bytes": target_map,
                }
            )
            minimum_growth_mib = math.ceil(MIN_INPUT_SCALE * parent_analysis.input_bytes / MIB)
            for variant in VARIANTS:
                target_rows.append(
                    {
                        "selected_index": source_row_index,
                        "parent_uuid": parent_uuid,
                        "parent_reference_sha256": parent_reference_hash,
                        "parent_input_bytes": parent_analysis.input_bytes,
                        "variant": variant,
                        "target_lower_mib": max(
                            (
                                MEDIUM_INPUT_MIN_BYTES // MIB
                                if variant == "medium"
                                else MEDIUM_INPUT_MAX_BYTES // MIB + 1
                            ),
                            minimum_growth_mib,
                        ),
                        "target_upper_mib": (
                            MEDIUM_INPUT_MAX_BYTES // MIB if variant == "medium" else LARGE_INPUT_MAX_BYTES // MIB
                        ),
                        "target_input_bytes": int(target_map[variant]),
                    }
                )

            parent_fake = _fake_tensor_gate(
                parent_code,
                entry_point,
                timeout_seconds=fake_gate_timeout_seconds,
            )
            counters[f"parent_fake_{parent_fake.status}"] += 1
            try:
                slots, guard_rejections = _shape_slots_with_rejections(parent_code, entry_point)
                profiles, rejected_profiles = _affine_profiles(
                    parent_code,
                    entry_point,
                    parent_analysis.input_bytes,
                    slots,
                )
                rejected_profiles = guard_rejections + rejected_profiles
            except (SyntaxError, TypeError, ValueError) as exc:
                slots = []
                profiles = []
                rejected_profiles = [{"slot_id": None, "reason": f"{type(exc).__name__}:{exc}"}]
            counters["slots_found"] += len(slots)
            counters["affine_slots"] += len(profiles)
            counters["non_affine_or_unpatchable_slots"] += len(rejected_profiles)
            counters["fixed_record_guard_rejected_slots"] += sum(
                rejection.get("reason") == "fixed_record_dead_tail_guard" for rejection in rejected_profiles
            )
            dimension_balance_bounds: dict[str, dict[str, Any]] = {}
            dimension_balance_bound_rejections: list[dict[str, Any]] = []
            if parent_fake.passed:
                for profile in profiles:
                    slot_id = profile.slot.slot_id
                    if slot_id in dimension_balance_bounds:
                        continue
                    bounds = _dimension_balance_slot_value_bounds(
                        parent_code,
                        profile.slot,
                    )
                    dimension_balance_bounds[slot_id] = bounds
                    if bounds["resolved"] and bounds["feasible"]:
                        counters["dimension_balance_feasible_affine_slots"] += 1
                    else:
                        counters["dimension_balance_infeasible_or_unresolved_affine_slots"] += 1
                        dimension_balance_bound_rejections.append(
                            {
                                "slot_id": slot_id,
                                "reason": bounds["reason"],
                                "bounds": bounds,
                            }
                        )
            parent_children: list[dict[str, Any]] = []

            for variant in VARIANTS:
                decision: dict[str, Any] = {
                    "source_row_index": source_row_index,
                    "parent_uuid": parent_uuid,
                    "parent_reference_sha256": parent_reference_hash,
                    "variant": variant,
                    "target_input_bytes": int(target_map[variant]),
                    "input_bytes_before": parent_analysis.input_bytes,
                    "accepted": False,
                    "parent_fake_gate": parent_fake.as_dict(),
                    "slot_count": len(slots),
                    "affine_slot_count": len(profiles),
                    "rejected_profiles": rejected_profiles,
                    "dimension_balance_bound_rejections": (dimension_balance_bound_rejections),
                    "attempts": [],
                }
                if not parent_fake.passed:
                    decision["reason"] = f"parent_fake_gate_{parent_fake.status}:{parent_fake.reason}"
                    skip_reasons[str(decision["reason"])] += 1
                    decisions.append(decision)
                    continue

                candidates = [
                    candidate
                    for profile in profiles
                    if (
                        candidate := _solve_profile(
                            profile,
                            variant=variant,
                            target_input_bytes=int(target_map[variant]),
                            parent_input_bytes=parent_analysis.input_bytes,
                            tie_salt=parent_uuid,
                            dimension_balance_value_bounds=(dimension_balance_bounds[profile.slot.slot_id]),
                        )
                    )
                    is not None
                ]
                candidates.sort(key=lambda candidate: _candidate_key(candidate, parent_uuid))
                if not candidates:
                    decision["reason"] = "no_balance_feasible_affine_slot_solution_in_variant_band"
                    skip_reasons[str(decision["reason"])] += 1
                    decisions.append(decision)
                    continue

                for candidate in candidates:
                    attempt = _candidate_manifest(candidate)
                    try:
                        child_code = _patch_integer_spans(
                            parent_code,
                            candidate.profile.slot,
                            candidate.slot_value,
                        )
                        shape_balance = _affected_shape_balance_guard(
                            child_code,
                            candidate.profile.slot,
                            candidate.slot_value,
                        )
                        attempt["dimension_balance_guard"] = shape_balance
                        if not shape_balance["passed"]:
                            counters["dimension_balance_guard_rejected_candidates"] += 1
                            raise ValueError(f"dimension_balance_guard_rejected:{shape_balance['reason']}")
                        counters["dimension_balance_guard_passed_candidates"] += 1
                        parent_sections = _section_hashes(ast.parse(parent_code), entry_point)
                        child_sections = _section_hashes(ast.parse(child_code), entry_point)
                        if parent_sections != child_sections:
                            raise ValueError("model_or_get_init_inputs_changed")
                        static = static_gate(parent_code, child_code, entry_point)
                        _validate_variant_storage(variant, int(static["input_bytes_after"]))
                        relative_error = _validate_target_proximity(
                            int(static["input_bytes_after"]), int(target_map[variant])
                        )
                        if int(static["input_bytes_after"]) != candidate.input_bytes_after:
                            raise ValueError("affine_solution_storage_mismatch")
                        fake = _fake_tensor_gate(
                            child_code,
                            entry_point,
                            timeout_seconds=fake_gate_timeout_seconds,
                        )
                        attempt["static_gate"] = {
                            "passed": True,
                            "input_scale": static["input_scale"],
                        }
                        attempt["fake_gate"] = fake.as_dict()
                        if not fake.passed:
                            raise ValueError(f"child_fake_gate_{fake.status}:{fake.reason}")
                        child = _make_child(parent, child_code, static)
                        child_uuid = str(_nested(child, "extra_info.uuid"))
                        attempt["accepted"] = True
                        solver_evidence = _candidate_manifest(candidate)
                        solver_evidence["dimension_balance_guard"] = shape_balance
                        parent_children.append(child)
                        decision.update(
                            {
                                "accepted": True,
                                "child_uuid": child_uuid,
                                "child_reference_sha256": static["child_reference_sha256"],
                                "child_normalized_ast_sha256": static["child_normalized_ast_sha256"],
                                "input_bytes_after": static["input_bytes_after"],
                                "input_scale": static["input_scale"],
                                "target_delta_bytes": int(static["input_bytes_after"]) - int(target_map[variant]),
                                "target_relative_error": relative_error,
                                "solver": solver_evidence,
                                "fake_gate": fake.as_dict(),
                            }
                        )
                        review_records.append(
                            {
                                "parent_uuid": parent_uuid,
                                "child_uuid": child_uuid,
                                "variant": variant,
                                "slot_id": candidate.profile.slot.slot_id,
                                "input_bytes_before": parent_analysis.input_bytes,
                                "input_bytes_after": static["input_bytes_after"],
                                "target_relative_error": relative_error,
                                "parent_code": parent_code,
                                "child_code": child_code,
                            }
                        )
                        counters[f"accepted_{variant}"] += 1
                        break
                    except (SyntaxError, TypeError, ValueError) as exc:
                        attempt["accepted"] = False
                        attempt["reason"] = f"{type(exc).__name__}:{exc}"
                    finally:
                        decision["attempts"].append(attempt)

                if not decision["accepted"]:
                    reasons = [str(item.get("reason")) for item in decision["attempts"]]
                    decision["reason"] = reasons[-1] if reasons else "all_candidates_rejected"
                    skip_reasons[str(decision["reason"])] += 1
                decisions.append(decision)

            if parent_children:
                parents_with_children.add(parent_uuid)
                paired.append(copy.deepcopy(parent))
                paired.extend(parent_children)
                children.extend(parent_children)

        target_map_sha256 = _logical_target_map_sha256(target_map_records)
        if (
            selected_path.resolve() == DEFAULT_SELECTED.resolve()
            and len(rows) == source.num_rows == 1000
            and target_map_sha256 != DEFAULT_TARGET_MAP_SHA256
        ):
            raise ValueError("default_target_map_sha256_mismatch:" f"{DEFAULT_TARGET_MAP_SHA256}:{target_map_sha256}")
        pq.write_table(
            pa.Table.from_pylist(target_rows, schema=TARGET_SCHEMA),
            temporary_paths.targets,
            compression="zstd",
        )
        schema = selected.schema
        pq.write_table(
            pa.Table.from_pylist(children, schema=schema),
            temporary_paths.children,
            compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pylist(paired, schema=schema),
            temporary_paths.paired,
            compression="zstd",
        )
        temporary_paths.review.write_text(_review_markdown(review_records), encoding="utf-8")

        counters["parents_with_children"] = len(parents_with_children)
        counters["children_written"] = len(children)
        counters["paired_rows_written"] = len(paired)
        manifest = {
            "contract_version": CONTRACT_VERSION,
            "generator_version": GENERATOR_VERSION,
            "solver_source_path": str(Path(__file__).resolve()),
            "solver_source_sha256": _sha256_file(Path(__file__)),
            "method_boundary": (
                "single whole-dimension affine input-storage solver with a conservative "
                "direct fixed-record-axis guard, an exact balance-feasible slot interval, "
                "and a hard 10^3 largest-to-second-largest dimension guard; FakeTensor "
                "is a compatibility filter, not target-GPU correctness or performance "
                "validation"
            ),
            "source_edit_contract": (
                "replace existing positive integer token spans only; preserve Model and " "get_init_inputs exactly"
            ),
            "target_contract": {
                "source": "shape_contract._target_input_bytes",
                "target_salt": TARGET_SALT,
                "logical_target_map_sha256": target_map_sha256,
                "variants": list(VARIANTS),
                "medium_bytes": [MEDIUM_INPUT_MIN_BYTES, MEDIUM_INPUT_MAX_BYTES],
                "large_bytes": [MEDIUM_INPUT_MAX_BYTES + 1, LARGE_INPUT_MAX_BYTES],
                "minimum_input_scale": MIN_INPUT_SCALE,
            },
            "candidate_ranking_contract": {
                "candidate_domain": (
                    "intersection of the variant/storage-growth interval and the exact "
                    "dimension-balance slot-value interval"
                ),
                "per_slot_choice": (
                    "integer slot value with affine input storage nearest the target within "
                    "that feasible intersection, including a clipped balance boundary"
                ),
                "target_error_equivalence_fraction": (f"1/{TARGET_ERROR_EQUIVALENCE_DENOMINATOR}"),
                "within_equivalent_error": [
                    "smaller_slot_growth_factor",
                    "more_propagated_input_factories",
                    "smaller_absolute_target_error",
                    "deterministic_hash",
                ],
            },
            "fixed_record_guard_contract": {
                "scope": (
                    "single-origin get_inputs return mapped to a positional forward argument; "
                    "every direct argument load must index the candidate axis by a fixed integer"
                ),
                "ambiguous_dataflow": "fail_open_to_fake_and_runtime_gates",
            },
            "dimension_balance_guard_contract": {
                "scope": ("every final rank>=2 direct input-factory tensor shape affected by " "the candidate slot"),
                "comparison": ("largest_dimension <= 1000 * second_largest_dimension"),
                "ratio_limit": MAX_DIMENSION_IMBALANCE_RATIO,
                "limit_expression": "10^3",
                "slot_reuse": (
                    "check every affected factory and verify every affected axis has the " "candidate slot value"
                ),
                "unresolved_affected_direct_factory": "fail_closed",
                "balance_aware_solver": {
                    "domain": "positive integer slot values",
                    "shape_model": (
                        "all affected axes equal the slot value; every unaffected axis is "
                        "fixed by the directly resolved parent factory shape"
                    ),
                    "result": ("exact intersection of per-factory minimum and maximum slot values"),
                    "post_patch_check": (
                        "independently resolve every final affected shape and reapply the "
                        "hard comparison before static/FakeTensor gates"
                    ),
                },
                "evidence": (
                    "per-attempt dimension_balance_guard and dimension_balance_value_bounds "
                    "plus the accepted solver record"
                ),
            },
            "fake_gate_contract": {
                "mode": "FakeTensorMode plus FakeCopyMode",
                "allow_fallback_kernels": False,
                "allow_non_fake_inputs": False,
                "static_shapes": True,
                "device_semantics": "CUDA literals and .cuda() calls normalized to fake CPU",
                "model_training_mode": True,
                "forward_trials": 1,
                "data_dependent_shape": "unsupported and excluded",
                "process_initialization": {
                    "timed": False,
                    "once_per_process": True,
                    "scope": (
                        "torch/FakeTensor imports, exception-type loading, logger setup, "
                        "and allocation-free Linear/FakeCopyMode warm-up"
                    ),
                    "failure_status": "failed",
                    "failure_reason": ("fake_gate_runtime_initialization_failed:<message>"),
                },
                "wall_clock_timeout": {
                    "seconds_per_invocation": fake_gate_timeout_seconds,
                    "timer": "Unix SIGALRM with ITIMER_REAL",
                    "scope": "independent parent and child gate invocation",
                    "timed_scope": (
                        "task-specific AST normalization/compile, exec, model and input "
                        "construction, deepcopy, and one forward"
                    ),
                    "excluded_scope": "process-level FakeTensor initialization and warm-up",
                    "status": "timeout",
                    "reason": ("wall_clock_timeout_exceeded:<seconds>s"),
                    "exception_semantics": (
                        "BaseException subclass bypasses ordinary internal " "except Exception handlers"
                    ),
                    "cleanup": ("previous SIGALRM handler and ITIMER_REAL timer restored in finally"),
                },
            },
            "selected_source": str(selected_path.resolve()),
            "selected_source_sha256": _sha256_file(selected_path),
            "selected_rows": len(rows),
            "artifacts": {
                "selected": str(final_paths.selected),
                "selection": str(final_paths.selection),
                "targets": str(final_paths.targets),
                "children": str(final_paths.children),
                "paired": str(final_paths.paired),
                "review": str(final_paths.review),
            },
            "artifact_sha256": {
                "selected": _sha256_file(temporary_paths.selected),
                "selection": _sha256_file(temporary_paths.selection),
                "targets": _sha256_file(temporary_paths.targets),
                "children": _sha256_file(temporary_paths.children),
                "paired": _sha256_file(temporary_paths.paired),
                "review": _sha256_file(temporary_paths.review),
            },
            "counts": dict(sorted(counters.items())),
            "skip_reason_counts": dict(sorted(skip_reasons.items())),
            "runtime_validation_required": True,
            "training_approved": False,
            "decisions": decisions,
        }
        temporary_paths.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary_dir, run_dir)
        return manifest
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=f"new output directory (default: {DEFAULT_RUN_DIR})",
    )
    parser.add_argument(
        "--selected",
        type=Path,
        default=DEFAULT_SELECTED,
        help=f"selected parent parquet (default: {DEFAULT_SELECTED})",
    )
    parser.add_argument(
        "--max-parents",
        type=int,
        help="process only the first N selected parents (for smoke runs)",
    )
    parser.add_argument(
        "--fake-gate-timeout-seconds",
        type=float,
        default=DEFAULT_FAKE_GATE_TIMEOUT_SECONDS,
        help=(
            "wall-clock timeout for each parent/child FakeTensor gate "
            f"(default: {DEFAULT_FAKE_GATE_TIMEOUT_SECONDS:g})"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    result = solve_shape_coverage(
        args.selected,
        args.run_dir,
        max_parents=args.max_parents,
        fake_gate_timeout_seconds=args.fake_gate_timeout_seconds,
    )
    print(
        json.dumps(
            {
                "run_dir": str(args.run_dir.resolve()),
                "counts": result["counts"],
                "training_approved": result["training_approved"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
