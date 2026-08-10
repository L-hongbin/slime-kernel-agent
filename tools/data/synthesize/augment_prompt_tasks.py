#!/usr/bin/env python3
"""Generate deterministic, fail-closed input-coverage children for prompt tasks.

The input parquet is never sampled, rewritten, or downsampled.  This tool emits
only child rows; callers retain every canonical parent and concatenate approved
children later.  Each child belongs to one explicit coverage cell:

* a larger shared leading (batch) dimension;
* a value family (``randn``, ``rand``, signed uniform, or finite +/-1
  boundary values); or
* the Cartesian product of one shape and one value intervention;
* a floating input dtype (``bfloat16`` or ``float16``); or
* a non-contiguous input layout with unchanged shape and element values.

Shape changes are deliberately narrow.  A child is emitted only when one
integer dimension symbol is the leading dimension of every statically resolved
input tensor factory and every load of that symbol is confined to those shape
slots in ``get_inputs``.  Unknown shapes, unsupported allocation factories,
ambiguous dimension coupling, and memory-risk estimates above the configured
budget are skipped with a reviewable reason.

Dtype and layout mutations are separate cells, never implicit parts of shape or
value mutations.  They are execution-gated because static input-factory proofs
cannot establish operator dtype or stride compatibility.  This is static
candidate generation, not runtime acceptance.  Generated rows record
``runtime_validation_required`` and must pass the normal target-GPU
cleanup/runtime contract before training use.
"""

from __future__ import annotations

import argparse
import ast
import collections
import copy
import dataclasses
import hashlib
import json
import math
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from tools.data.cleaning.pipeline import inspect_row_schema

CONTRACT_VERSION = "prompt_input_coverage_augmentation_v3"
GENERATOR_VERSION = "ast_input_coverage_v3"
MEMORY_ESTIMATOR_VERSION = "proven_returned_input_storage_bytes_times_working_set_v2"
VALUE_FAMILIES = ("randn", "rand", "signed_uniform", "boundary_pm1")
DTYPE_TARGETS = ("bfloat16", "float16")
LAYOUT_TARGETS = ("non_contiguous",)
DEFAULT_SHAPE_SCALES = (2, 4, 8)
DEFAULT_AVAILABLE_GPU_BYTES = 70 * 1024**3
DEFAULT_MEMORY_SAFETY_FRACTION = 0.70
DEFAULT_WORKING_SET_MULTIPLIER = 24.0

_BATCH_NAME_RE = re.compile(r"^(?:batch_size|batch|bs|n_batch)$", re.IGNORECASE)
_CONTINUOUS_FACTORIES = frozenset({"torch.rand", "torch.randn"})
_FLOAT_FACTORIES = frozenset({"torch.empty", "torch.full", "torch.ones", "torch.rand", "torch.randn", "torch.zeros"})
_SHAPE_FACTORIES = frozenset(
    {
        "torch.empty",
        "torch.full",
        "torch.ones",
        "torch.rand",
        "torch.randint",
        "torch.randn",
        "torch.zeros",
    }
)
_UNSUPPORTED_ALLOCATION_CALLS = frozenset(
    {
        "torch.arange",
        "torch.asarray",
        "torch.as_tensor",
        "torch.cat",
        "torch.empty_like",
        "torch.eye",
        "torch.from_numpy",
        "torch.full_like",
        "torch.linspace",
        "torch.logspace",
        "torch.normal",
        "torch.ones_like",
        "torch.rand_like",
        "torch.randint_like",
        "torch.randn_like",
        "torch.randperm",
        "torch.stack",
        "torch.tensor",
        "torch.zeros_like",
    }
)
_DTYPE_BYTES = {
    "torch.bool": 1,
    "torch.uint8": 1,
    "torch.int8": 1,
    "torch.float8_e4m3fn": 1,
    "torch.float8_e5m2": 1,
    "torch.int16": 2,
    "torch.short": 2,
    "torch.float16": 2,
    "torch.half": 2,
    "torch.bfloat16": 2,
    "torch.int32": 4,
    "torch.int": 4,
    "torch.float32": 4,
    "torch.float": 4,
    "torch.complex64": 8,
    "torch.cfloat": 8,
    "torch.int64": 8,
    "torch.long": 8,
    "torch.float64": 8,
    "torch.double": 8,
    "torch.complex128": 16,
    "torch.cdouble": 16,
}

_FLOAT_DTYPE_NAMES = frozenset(
    {
        "torch.bfloat16",
        "torch.float16",
        "torch.half",
        "torch.float32",
        "torch.float",
        "torch.float64",
        "torch.double",
    }
)


AUGMENTATION_METADATA_TYPE = pa.struct(
    [
        pa.field("contract_version", pa.string()),
        pa.field("generator_version", pa.string()),
        pa.field("parent_uuid", pa.string()),
        pa.field("child_uuid", pa.string()),
        pa.field("source_artifact_sha256", pa.string()),
        pa.field("source_row_index", pa.int64()),
        pa.field("parent_reference_sha256", pa.string()),
        pa.field("parent_normalized_ast_sha256", pa.string()),
        pa.field("child_reference_sha256", pa.string()),
        pa.field("child_normalized_ast_sha256", pa.string()),
        pa.field("intervention_id", pa.string()),
        pa.field("intervention_sha256", pa.string()),
        pa.field("intervention_kind", pa.string()),
        pa.field("coverage_cell", pa.string()),
        pa.field("shape_scale", pa.int32()),
        pa.field("value_family_before", pa.string()),
        pa.field("value_family_after", pa.string()),
        pa.field("dtype_before", pa.string()),
        pa.field("dtype_after", pa.string()),
        pa.field("layout_before", pa.string()),
        pa.field("layout_after", pa.string()),
        pa.field("shape_dimension_name", pa.string()),
        pa.field("shape_dimension_before", pa.int64()),
        pa.field("shape_dimension_after", pa.int64()),
        pa.field("factory_count", pa.int32()),
        pa.field("input_bytes_before", pa.int64()),
        pa.field("input_bytes_after", pa.int64()),
        pa.field("estimated_peak_bytes", pa.int64()),
        pa.field("memory_budget_bytes", pa.int64()),
        pa.field("working_set_multiplier", pa.float64()),
        pa.field("memory_estimator_version", pa.string()),
        pa.field("shape_proof", pa.string()),
        pa.field("compatibility_proof", pa.string()),
        pa.field("validation_status", pa.string()),
    ]
)


@dataclasses.dataclass(frozen=True)
class AugmentationPolicy:
    """Bounded coverage cells and the conservative static memory gate."""

    shape_scales: tuple[int, ...] = DEFAULT_SHAPE_SCALES
    value_families: tuple[str, ...] = VALUE_FAMILIES
    dtype_targets: tuple[str, ...] = DTYPE_TARGETS
    layout_targets: tuple[str, ...] = LAYOUT_TARGETS
    include_joint_cells: bool = True
    available_gpu_bytes: int = DEFAULT_AVAILABLE_GPU_BYTES
    memory_safety_fraction: float = DEFAULT_MEMORY_SAFETY_FRACTION
    working_set_multiplier: float = DEFAULT_WORKING_SET_MULTIPLIER
    max_scaled_dimension: int = 8192

    def __post_init__(self) -> None:
        if any(scale <= 1 for scale in self.shape_scales):
            raise ValueError("shape_scales must contain integers greater than one")
        if len(set(self.shape_scales)) != len(self.shape_scales):
            raise ValueError("shape_scales must be unique")
        invalid_families = set(self.value_families) - set(VALUE_FAMILIES)
        if invalid_families:
            raise ValueError(f"unsupported value families: {sorted(invalid_families)}")
        if len(set(self.value_families)) != len(self.value_families):
            raise ValueError("value_families must be unique")
        invalid_dtypes = set(self.dtype_targets) - set(DTYPE_TARGETS)
        if invalid_dtypes:
            raise ValueError(f"unsupported dtype targets: {sorted(invalid_dtypes)}")
        if len(set(self.dtype_targets)) != len(self.dtype_targets):
            raise ValueError("dtype_targets must be unique")
        invalid_layouts = set(self.layout_targets) - set(LAYOUT_TARGETS)
        if invalid_layouts:
            raise ValueError(f"unsupported layout targets: {sorted(invalid_layouts)}")
        if len(set(self.layout_targets)) != len(self.layout_targets):
            raise ValueError("layout_targets must be unique")
        if self.available_gpu_bytes <= 0:
            raise ValueError("available_gpu_bytes must be positive")
        if not 0.0 < self.memory_safety_fraction <= 1.0:
            raise ValueError("memory_safety_fraction must be in (0, 1]")
        if self.working_set_multiplier < 1.0:
            raise ValueError("working_set_multiplier must be at least one")
        if self.max_scaled_dimension <= 0:
            raise ValueError("max_scaled_dimension must be positive")

    @property
    def memory_budget_bytes(self) -> int:
        return math.floor(self.available_gpu_bytes * self.memory_safety_fraction)

    def as_dict(self) -> dict[str, Any]:
        return {
            "shape_scales": list(self.shape_scales),
            "value_families": list(self.value_families),
            "dtype_targets": list(self.dtype_targets),
            "layout_targets": list(self.layout_targets),
            "include_joint_cells": self.include_joint_cells,
            "available_gpu_bytes": self.available_gpu_bytes,
            "memory_safety_fraction": self.memory_safety_fraction,
            "memory_budget_bytes": self.memory_budget_bytes,
            "working_set_multiplier": self.working_set_multiplier,
            "max_scaled_dimension": self.max_scaled_dimension,
        }


@dataclasses.dataclass(frozen=True)
class FactoryRecord:
    name: str
    shape: tuple[int, ...]
    dtype_name: str
    dtype_bytes: int
    call_node_id: int
    first_dimension_node_id: int
    line_number: int
    column_offset: int

    @property
    def bytes(self) -> int:
        return math.prod(self.shape) * self.dtype_bytes


@dataclasses.dataclass(frozen=True)
class ShapeProof:
    scope: str
    name: str
    before: int
    factory_count: int
    proof: str = "shared_leading_dimension_symbol_only_used_in_get_inputs_factory_shapes"


@dataclasses.dataclass(frozen=True)
class CodeAnalysis:
    tree: ast.Module
    source_value_family: str
    continuous_factory_count: int
    value_transform_eligible: bool
    value_skip_reason: str | None
    dtype_factory_count: int
    dtype_factory_locations: tuple[tuple[int, int], ...]
    floating_input_dtypes: tuple[str, ...]
    dtype_transform_eligible: bool
    dtype_skip_reason: str | None
    layout_transform_eligible: bool
    layout_skip_reason: str | None
    input_bytes: int
    factory_count: int
    shape_proof: ShapeProof | None
    shape_skip_reason: str | None


@dataclasses.dataclass(frozen=True)
class Intervention:
    shape_scale: int | None
    value_family: str | None
    dtype_target: str | None = None
    layout_target: str | None = None

    def __post_init__(self) -> None:
        if self.dtype_target is not None and any(
            value is not None for value in (self.shape_scale, self.value_family, self.layout_target)
        ):
            raise ValueError("dtype interventions must be separate coverage cells")
        if self.layout_target is not None and any(
            value is not None for value in (self.shape_scale, self.value_family, self.dtype_target)
        ):
            raise ValueError("layout interventions must be separate coverage cells")

    def payload(self, analysis: CodeAnalysis) -> dict[str, Any]:
        return {
            "shape": (
                {
                    "axis": "shared_leading_input_dimension",
                    "dimension_name": analysis.shape_proof.name if analysis.shape_proof else None,
                    "scale": self.shape_scale,
                }
                if self.shape_scale is not None
                else None
            ),
            "value": (
                {"from": analysis.source_value_family, "to": self.value_family}
                if self.value_family is not None
                else None
            ),
            "dtype": self.dtype_target,
            "layout": self.layout_target,
        }

    @property
    def kind(self) -> str:
        if self.dtype_target is not None:
            return "dtype"
        if self.layout_target is not None:
            return "layout"
        if self.shape_scale is not None and self.value_family is not None:
            return "shape_value"
        if self.shape_scale is not None:
            return "shape"
        if self.value_family is not None:
            return "value"
        raise ValueError("empty intervention")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized_ast_sha256(code: str) -> str:
    tree = ast.parse(code)
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return _sha256_bytes(normalized.encode("utf-8"))


def _call_name(node: ast.AST) -> str:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    else:
        return ""
    return ".".join(reversed(parts))


def _top_level_function(tree: ast.Module, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    matches = [
        node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one top-level {name}(), found {len(matches)}")
    return matches[0]


def _top_level_class(tree: ast.Module, name: str) -> ast.ClassDef:
    matches = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one top-level class {name!r}, found {len(matches)}")
    return matches[0]


def _safe_scalar(node: ast.AST, environment: Mapping[str, Any]) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.Name) and node.id in environment:
        value = environment[node.id]
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _safe_scalar(node.operand, environment)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp):
        left = _safe_scalar(node.left, environment)
        right = _safe_scalar(node.right, environment)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv):
            if right == 0:
                raise ValueError("division by zero in static integer expression")
            return left // right
    raise ValueError(f"unsupported static scalar expression: {ast.dump(node, include_attributes=False)}")


def _assignment_values(target: ast.AST, value: ast.AST, environment: Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(target, ast.Name):
        return {target.id: _safe_scalar(value, environment)}
    if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (ast.Tuple, ast.List)):
        if len(target.elts) != len(value.elts):
            return {}
        updates: dict[str, Any] = {}
        local = dict(environment)
        for left, right in zip(target.elts, value.elts, strict=True):
            if not isinstance(left, ast.Name):
                return {}
            resolved = _safe_scalar(right, local)
            updates[left.id] = resolved
            local[left.id] = resolved
        return updates
    return {}


def _target_names(target: ast.AST) -> set[str]:
    return {node.id for node in ast.walk(target) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}


def _statement_assignment(statement: ast.stmt) -> tuple[ast.AST, ast.AST] | None:
    if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
        return statement.targets[0], statement.value
    if isinstance(statement, ast.AnnAssign) and statement.value is not None:
        return statement.target, statement.value
    return None


def _module_constant_environment(tree: ast.Module) -> dict[str, Any]:
    """Resolve deterministic module constants in execution order.

    Module assignments have all executed before ``get_inputs`` is called, so a
    later direct module assignment legitimately replaces an earlier one.  A
    control-flow assignment is not statically deterministic and is rejected;
    silently retaining a pre-branch value would make the byte proof unsound.
    """

    module_environment: dict[str, Any] = {}
    for statement in tree.body:
        assignment = _statement_assignment(statement)
        if assignment is not None:
            target, value = assignment
            try:
                module_environment.update(_assignment_values(target, value, module_environment))
            except ValueError:
                continue
            continue
        if isinstance(
            statement, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.Match)
        ):
            stored = {
                node.id
                for node in ast.walk(statement)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
            }
            if stored:
                raise ValueError(f"ambiguous_module_control_flow_assignment:{','.join(sorted(stored))}")
    return module_environment


def _shape_nodes(call: ast.Call, name: str) -> list[ast.AST]:
    size_keyword = next((keyword.value for keyword in call.keywords if keyword.arg in {"size", "shape"}), None)
    shape: ast.AST | Sequence[ast.AST] | None = size_keyword
    if shape is None:
        if name in {"torch.rand", "torch.randn", "torch.zeros", "torch.ones", "torch.empty"}:
            if len(call.args) == 1 and isinstance(call.args[0], (ast.Tuple, ast.List)):
                shape = call.args[0]
            else:
                shape = call.args
        elif name == "torch.full":
            shape = call.args[0] if call.args else None
        elif name == "torch.randint":
            if len(call.args) >= 3:
                shape = call.args[2]
            elif len(call.args) >= 2:
                shape = call.args[1]
    if isinstance(shape, (ast.Tuple, ast.List)):
        return list(shape.elts)
    if isinstance(shape, Sequence):
        return list(shape)
    if isinstance(shape, ast.AST):
        # randint commonly receives a tuple-valued name (for example
        # ``torch.randint(0, 2, target_shape)``).  Preserve that expression as
        # one shape node so the dependency walker can trace the module-level
        # tuple and its numeric dimensions.
        return [shape]
    return []


def _full_default_dtype(
    call: ast.Call,
    environment: Mapping[str, Any],
) -> tuple[str, int]:
    fill_node = next((item.value for item in call.keywords if item.arg == "fill_value"), None)
    if fill_node is None and len(call.args) >= 2:
        fill_node = call.args[1]
    if fill_node is None:
        raise ValueError("torch.full input factory has no statically resolved fill_value")

    if isinstance(fill_node, ast.Constant):
        value = fill_node.value
    else:
        try:
            value = _safe_scalar(fill_node, environment)
        except ValueError:
            if (
                isinstance(fill_node, ast.Call)
                and isinstance(fill_node.func, ast.Name)
                and fill_node.func.id == "float"
                and len(fill_node.args) == 1
                and not fill_node.keywords
                and isinstance(fill_node.args[0], ast.Constant)
                and isinstance(fill_node.args[0].value, (str, int, float))
            ):
                value = float(fill_node.args[0].value)
            else:
                raise ValueError("unsupported torch.full fill_value without explicit dtype") from None

    if isinstance(value, bool):
        return "torch.bool", _DTYPE_BYTES["torch.bool"]
    if isinstance(value, int):
        return "torch.int64", _DTYPE_BYTES["torch.int64"]
    if isinstance(value, float):
        return "torch.float32", _DTYPE_BYTES["torch.float32"]
    if isinstance(value, complex):
        return "torch.complex64", _DTYPE_BYTES["torch.complex64"]
    raise ValueError(f"unsupported torch.full fill_value type without explicit dtype: {type(value).__name__}")


def _factory_dtype(
    call: ast.Call,
    factory_name: str,
    environment: Mapping[str, Any],
) -> tuple[str, int]:
    keyword = next((item.value for item in call.keywords if item.arg == "dtype"), None)
    if keyword is None:
        if factory_name == "torch.randint":
            return "torch.int64", _DTYPE_BYTES["torch.int64"]
        if factory_name == "torch.full":
            return _full_default_dtype(call, environment)
        return "torch.float32", _DTYPE_BYTES["torch.float32"]
    dtype_name = _call_name(keyword)
    if dtype_name not in _DTYPE_BYTES:
        raise ValueError(f"unsupported dtype in input factory: {dtype_name or ast.dump(keyword)}")
    return dtype_name, _DTYPE_BYTES[dtype_name]


def _factory_records(
    tree: ast.Module,
    get_inputs: ast.FunctionDef | ast.AsyncFunctionDef,
    module_environment: Mapping[str, Any],
) -> tuple[list[FactoryRecord], dict[str, Any]]:
    if any(
        isinstance(node, ast.Call)
        and _call_name(node.func) in {"torch.set_default_dtype", "torch.set_default_tensor_type"}
        for node in ast.walk(tree)
    ):
        raise ValueError("dynamic_default_dtype")
    supported_calls: list[tuple[ast.Call, str]] = []
    for node in ast.walk(get_inputs):
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if name in _UNSUPPORTED_ALLOCATION_CALLS:
            raise ValueError(f"unsupported_input_allocation:{name}")
        if name in _SHAPE_FACTORIES:
            if any(keyword.arg in {"out", "layout", "memory_format"} for keyword in node.keywords):
                raise ValueError(f"unsupported_input_factory_storage_modifier:{name}")
            supported_calls.append((node, name))
    if not supported_calls:
        raise ValueError("no_supported_input_factories")

    shape_names: set[str] = set()
    for call, name in supported_calls:
        dimensions = _shape_nodes(call, name)
        if not dimensions:
            raise ValueError(f"unresolved_input_shape:{name}:line{getattr(call, 'lineno', -1)}")
        for dimension in dimensions:
            shape_names.update(
                node.id
                for node in ast.walk(dimension)
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
            )

    local_assignment_counts: collections.Counter[str] = collections.Counter()
    direct_assignment_names: set[str] = set()
    for statement in get_inputs.body:
        assignment = _statement_assignment(statement)
        if assignment is not None:
            names = _target_names(assignment[0])
            local_assignment_counts.update(names)
            direct_assignment_names.update(names)
            continue
        stored = {
            node.id for node in ast.walk(statement) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        }
        ambiguous = stored & shape_names
        if ambiguous:
            raise ValueError(f"ambiguous_control_flow_shape_assignment:{','.join(sorted(ambiguous))}")
    repeated = sorted(name for name in shape_names if local_assignment_counts[name] > 1)
    if repeated:
        raise ValueError(f"ambiguous_shape_symbol_reassignment:{','.join(repeated)}")

    locally_shadowed_shape_names = shape_names & direct_assignment_names
    local_environment = {
        name: value for name, value in module_environment.items() if name not in locally_shadowed_shape_names
    }
    records: list[FactoryRecord] = []
    seen_call_ids: set[int] = set()
    ambiguous_execution_nodes = (
        ast.If,
        ast.IfExp,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.Try,
        ast.With,
        ast.AsyncWith,
        ast.Match,
        ast.Lambda,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.BoolOp,
    )
    for statement in get_inputs.body:
        statement_parents = {
            id(child): parent for parent in ast.walk(statement) for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node.func)
            if name not in _SHAPE_FACTORIES:
                continue
            current: ast.AST = node
            while current is not statement:
                current = statement_parents.get(id(current), statement)
                if isinstance(current, ambiguous_execution_nodes):
                    raise ValueError(f"input_factory_in_ambiguous_control_flow:{name}")
            dimensions = _shape_nodes(node, name)
            resolved: list[int] = []
            for dimension in dimensions:
                value = _safe_scalar(dimension, local_environment)
                if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                    raise ValueError(f"non_positive_or_non_integer_input_dimension:{value!r}")
                resolved.append(value)
            dtype_name, dtype_bytes = _factory_dtype(node, name, local_environment)
            records.append(
                FactoryRecord(
                    name=name,
                    shape=tuple(resolved),
                    dtype_name=dtype_name,
                    dtype_bytes=dtype_bytes,
                    call_node_id=id(node),
                    first_dimension_node_id=id(dimensions[0]),
                    line_number=getattr(node, "lineno", -1),
                    column_offset=getattr(node, "col_offset", -1),
                )
            )
            seen_call_ids.add(id(node))

        assignment = _statement_assignment(statement)
        if assignment is not None:
            target, value = assignment
            target_names = _target_names(target)
            try:
                updates = _assignment_values(target, value, local_environment)
            except ValueError:
                updates = {}
            unresolved_shape_names = (target_names & shape_names) - set(updates)
            if unresolved_shape_names:
                raise ValueError(f"unresolved_shape_symbol_assignment:{','.join(sorted(unresolved_shape_names))}")
            local_environment.update(updates)

    expected_call_ids = {id(call) for call, _ in supported_calls}
    if seen_call_ids != expected_call_ids:
        raise ValueError("not_all_input_factories_resolved_in_statement_order")
    return records, local_environment


def _numeric_constant(node: ast.AST, expected: int | float) -> bool:
    try:
        value = _safe_scalar(node, {})
    except ValueError:
        return False
    return value == expected


def _generated_wrapper_base(node: ast.AST) -> ast.AST | None:
    """Recognize only wrappers emitted by this generator with exact storage.

    Each accepted wrapper has the same final shape, element count, and dtype as
    its base factory.  Arbitrary user-authored method chains remain unsupported.
    """

    if (
        isinstance(node, ast.Call)
        and _call_name(node.func) == "torch.sign"
        and len(node.args) == 1
        and not node.keywords
    ):
        return node.args[0]

    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Sub)
        and _numeric_constant(node.right, 1.0)
        and isinstance(node.left, ast.BinOp)
        and isinstance(node.left.op, ast.Mult)
        and _numeric_constant(node.left.right, 2.0)
    ):
        return node.left.left

    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "transpose"
        and len(node.args) == 2
        and _numeric_constant(node.args[0], -1)
        and _numeric_constant(node.args[1], -2)
        and not node.keywords
    ):
        return None
    contiguous = node.func.value
    if not (
        isinstance(contiguous, ast.Call)
        and isinstance(contiguous.func, ast.Attribute)
        and contiguous.func.attr == "contiguous"
        and not contiguous.args
        and not contiguous.keywords
    ):
        return None
    first_transpose = contiguous.func.value
    if not (
        isinstance(first_transpose, ast.Call)
        and isinstance(first_transpose.func, ast.Attribute)
        and first_transpose.func.attr == "transpose"
        and len(first_transpose.args) == 2
        and _numeric_constant(first_transpose.args[0], -1)
        and _numeric_constant(first_transpose.args[1], -2)
        and not first_transpose.keywords
    ):
        return None
    return first_transpose.func.value


def _prove_returned_input_storage(
    get_inputs: ast.FunctionDef | ast.AsyncFunctionDef,
    records: Sequence[FactoryRecord],
    scalar_environment: Mapping[str, Any],
) -> bool:
    """Require exact provenance for every tensor returned by ``get_inputs``.

    The proof accepts a supported factory directly, a simple assignment/alias,
    scalar arguments, literal containers, and the exact wrappers generated by
    this file.  Any other call, method chain, mutation, duplicate alias, or
    conditional return is rejected rather than guessed.
    """

    returns = [node for node in ast.walk(get_inputs) if isinstance(node, ast.Return)]
    if len(returns) != 1 or returns[0] not in get_inputs.body:
        raise ValueError("get_inputs_return_control_flow_not_proven")
    returned = returns[0]
    if returned.value is None:
        raise ValueError("get_inputs_return_value_missing")

    bindings: dict[str, list[tuple[ast.AST, ast.Name]]] = collections.defaultdict(list)
    for statement in get_inputs.body:
        assignment = _statement_assignment(statement)
        if assignment is None:
            continue
        target, value = assignment
        if isinstance(target, ast.Name):
            bindings[target.id].append((value, target))

    record_by_call_id = {record.call_node_id: record for record in records}
    seen_factory_ids: list[int] = []
    followed_names: set[str] = set()
    allowed_name_node_ids: set[int] = set()
    generated_wrapper_count = 0

    def allow_expression_names(expression: ast.AST) -> None:
        allowed_name_node_ids.update(id(node) for node in ast.walk(expression) if isinstance(node, ast.Name))

    def visit(expression: ast.AST, stack: tuple[str, ...] = ()) -> None:
        nonlocal generated_wrapper_count
        if isinstance(expression, ast.Constant):
            if isinstance(expression.value, (str, bytes, int, float, complex, bool, type(None))):
                return
            raise ValueError("unsupported_returned_input_constant")
        if isinstance(expression, (ast.List, ast.Tuple, ast.Set)):
            allow_expression_names(expression)
            for element in expression.elts:
                if isinstance(element, ast.Starred):
                    raise ValueError("starred_returned_inputs_not_proven")
                visit(element, stack)
            return
        if isinstance(expression, ast.Dict):
            allow_expression_names(expression)
            if any(key is None for key in expression.keys):
                raise ValueError("expanded_returned_input_dict_not_proven")
            for value in expression.values:
                visit(value, stack)
            return
        if isinstance(expression, ast.Name):
            matches = bindings.get(expression.id, [])
            if matches:
                if len(matches) != 1:
                    raise ValueError(f"returned_input_name_reassigned:{expression.id}")
                if expression.id in stack:
                    raise ValueError(f"returned_input_alias_cycle:{expression.id}")
                followed_names.add(expression.id)
                allowed_name_node_ids.add(id(expression))
                value, target = matches[0]
                allowed_name_node_ids.add(id(target))
                allow_expression_names(value)
                visit(value, (*stack, expression.id))
                return
            if expression.id in scalar_environment:
                allowed_name_node_ids.add(id(expression))
                return
            raise ValueError(f"returned_input_name_unresolved:{expression.id}")
        if isinstance(expression, ast.UnaryOp):
            try:
                _safe_scalar(expression, scalar_environment)
            except ValueError:
                raise ValueError("unsupported_returned_input_unary_expression") from None
            allow_expression_names(expression)
            return
        if isinstance(expression, ast.BinOp):
            wrapper_base = _generated_wrapper_base(expression)
            if wrapper_base is not None:
                generated_wrapper_count += 1
                allow_expression_names(expression)
                visit(wrapper_base, stack)
                return
            try:
                _safe_scalar(expression, scalar_environment)
            except ValueError:
                raise ValueError("unsupported_returned_input_binary_expression") from None
            allow_expression_names(expression)
            return
        if isinstance(expression, ast.Call):
            if id(expression) in record_by_call_id:
                seen_factory_ids.append(id(expression))
                allow_expression_names(expression)
                return
            wrapper_base = _generated_wrapper_base(expression)
            if wrapper_base is not None:
                generated_wrapper_count += 1
                allow_expression_names(expression)
                visit(wrapper_base, stack)
                return
            raise ValueError(f"unsupported_returned_input_call:{_call_name(expression.func) or 'dynamic'}")
        raise ValueError(f"unsupported_returned_input_expression:{type(expression).__name__}")

    allow_expression_names(returned.value)
    visit(returned.value)
    expected_factory_ids = set(record_by_call_id)
    observed_factory_ids = set(seen_factory_ids)
    if len(seen_factory_ids) != len(observed_factory_ids):
        raise ValueError("returned_input_storage_alias_or_duplicate")
    if observed_factory_ids != expected_factory_ids:
        missing = len(expected_factory_ids - observed_factory_ids)
        unexpected = len(observed_factory_ids - expected_factory_ids)
        raise ValueError(
            f"not_all_input_factories_are_unique_returned_inputs:missing={missing}:unexpected={unexpected}"
        )

    for node in ast.walk(get_inputs):
        if isinstance(node, ast.Name) and node.id in followed_names and id(node) not in allowed_name_node_ids:
            raise ValueError(f"returned_input_name_used_outside_provenance:{node.id}")
    return generated_wrapper_count == 0


def _bare_continuous_calls(
    get_inputs: ast.FunctionDef | ast.AsyncFunctionDef,
) -> tuple[list[ast.Call], bool]:
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(get_inputs):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    calls = [
        node
        for node in ast.walk(get_inputs)
        if isinstance(node, ast.Call) and _call_name(node.func) in _CONTINUOUS_FACTORIES
    ]
    for call in calls:
        current: ast.AST = call
        while True:
            parent = parents.get(id(current))
            if parent is None:
                return calls, False
            if isinstance(parent, (ast.List, ast.Tuple, ast.Set)):
                current = parent
                continue
            if isinstance(parent, ast.Dict) and current in parent.values:
                current = parent
                continue
            if isinstance(parent, ast.Assign) and parent.value is current:
                break
            if isinstance(parent, ast.AnnAssign) and parent.value is current:
                break
            if isinstance(parent, ast.Return) and parent.value is current:
                break
            return calls, False
    return calls, bool(calls)


def _source_value_family(calls: Sequence[ast.Call]) -> str:
    names = {_call_name(call.func).removeprefix("torch.") for call in calls}
    if len(names) == 1:
        return next(iter(names))
    return "mixed" if names else "none"


def _dtype_eligibility(records: Sequence[FactoryRecord]) -> tuple[int, bool, str | None]:
    candidates = [
        record for record in records if record.name in _FLOAT_FACTORIES and record.dtype_name in _FLOAT_DTYPE_NAMES
    ]
    if not candidates:
        return 0, False, "no_supported_floating_input_factory"
    return len(candidates), True, None


def _layout_eligibility(records: Sequence[FactoryRecord], continuous_calls_are_bare: bool) -> tuple[bool, str | None]:
    continuous = [record for record in records if record.name in _CONTINUOUS_FACTORIES]
    if not continuous:
        return False, "no_continuous_factory"
    if not continuous_calls_are_bare:
        return False, "continuous_factory_not_bare"
    if any(len(record.shape) < 2 for record in continuous):
        return False, "non_contiguous_requires_rank_at_least_two"
    if any(record.shape[-1] <= 1 or record.shape[-2] <= 1 for record in continuous):
        return False, "non_contiguous_requires_two_nontrivial_trailing_dimensions"
    return True, None


def _assignment_scope(tree: ast.Module, get_inputs: ast.AST, name: str) -> tuple[str, int] | None:
    module_matches: list[int] = []
    local_matches: list[int] = []

    def record(target: ast.AST, value: ast.AST, destination: list[int]) -> None:
        if isinstance(target, ast.Name) and target.id == name:
            try:
                resolved = _safe_scalar(value, {})
            except ValueError:
                return
            if isinstance(resolved, int) and not isinstance(resolved, bool):
                destination.append(resolved)

    for statement in tree.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            record(statement.targets[0], statement.value, module_matches)
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            record(statement.target, statement.value, module_matches)
    assert isinstance(get_inputs, (ast.FunctionDef, ast.AsyncFunctionDef))
    for statement in get_inputs.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            record(statement.targets[0], statement.value, local_matches)
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            record(statement.target, statement.value, local_matches)
    if len(local_matches) == 1:
        return "get_inputs", local_matches[0]
    if not local_matches and len(module_matches) == 1:
        return "module", module_matches[0]
    return None


def _shape_proof(
    tree: ast.Module,
    get_inputs: ast.FunctionDef | ast.AsyncFunctionDef,
    records: Sequence[FactoryRecord],
) -> tuple[ShapeProof | None, str | None]:
    candidate_nodes: dict[str, list[ast.Name]] = collections.defaultdict(list)
    allowed_node_ids = {record.first_dimension_node_id for record in records}
    for node in ast.walk(get_inputs):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and id(node) in allowed_node_ids:
            if _BATCH_NAME_RE.match(node.id):
                candidate_nodes[node.id].append(node)
    viable = [name for name, nodes in candidate_nodes.items() if len(nodes) == len(records)]
    if len(viable) != 1:
        return None, "no_unique_shared_leading_batch_symbol"
    name = viable[0]
    assignment = _assignment_scope(tree, get_inputs, name)
    if assignment is None:
        return None, "batch_symbol_assignment_not_unique_literal"
    scope, before = assignment

    for node in ast.walk(get_inputs):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == name:
            if id(node) not in allowed_node_ids:
                return None, "batch_symbol_used_outside_factory_leading_dimensions"
    if scope == "module":
        for top_level in tree.body:
            if top_level is get_inputs:
                continue
            if any(
                isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == name
                for node in ast.walk(top_level)
            ):
                return None, "module_batch_symbol_used_outside_get_inputs"
    return ShapeProof(scope, name, before, len(records)), None


def analyze_code(code: str, entry_point: str = "Model") -> CodeAnalysis:
    """Return eligibility and a proven returned-input storage-byte estimate."""

    tree = ast.parse(code)
    _top_level_class(tree, entry_point)
    get_inputs = _top_level_function(tree, "get_inputs")
    module_environment = _module_constant_environment(tree)
    records, local_environment = _factory_records(tree, get_inputs, module_environment)
    returned_factories_are_unwrapped = _prove_returned_input_storage(
        get_inputs,
        records,
        local_environment,
    )
    continuous, bare = _bare_continuous_calls(get_inputs)
    source_family = _source_value_family(continuous)
    shape_proof, shape_skip = _shape_proof(tree, get_inputs, records)
    dtype_count, dtype_eligible, dtype_skip = _dtype_eligibility(records)
    layout_eligible, layout_skip = _layout_eligibility(
        records,
        bare and returned_factories_are_unwrapped,
    )
    return CodeAnalysis(
        tree=tree,
        source_value_family=source_family,
        continuous_factory_count=len(continuous),
        value_transform_eligible=bool(continuous) and bare and returned_factories_are_unwrapped,
        value_skip_reason=(
            None
            if continuous and bare and returned_factories_are_unwrapped
            else (
                "no_continuous_factory"
                if not continuous
                else ("continuous_factory_not_bare" if not bare else "returned_input_has_value_or_layout_wrapper")
            )
        ),
        dtype_factory_count=dtype_count,
        dtype_factory_locations=tuple(
            sorted(
                (record.line_number, record.column_offset)
                for record in records
                if record.name in _FLOAT_FACTORIES and record.dtype_name in _FLOAT_DTYPE_NAMES
            )
        ),
        floating_input_dtypes=tuple(
            sorted(
                {
                    record.dtype_name
                    for record in records
                    if record.name in _FLOAT_FACTORIES and record.dtype_name in _FLOAT_DTYPE_NAMES
                }
            )
        ),
        dtype_transform_eligible=dtype_eligible,
        dtype_skip_reason=dtype_skip,
        layout_transform_eligible=layout_eligible,
        layout_skip_reason=layout_skip,
        input_bytes=sum(record.bytes for record in records),
        factory_count=len(records),
        shape_proof=shape_proof,
        shape_skip_reason=shape_skip,
    )


def _replace_dimension_assignment(tree: ast.Module, proof: ShapeProof, value: int) -> None:
    get_inputs = _top_level_function(tree, "get_inputs")
    statements: Iterable[ast.stmt] = get_inputs.body if proof.scope == "get_inputs" else tree.body
    replaced = 0
    for statement in statements:
        target: ast.AST | None = None
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
        elif isinstance(statement, ast.AnnAssign):
            target = statement.target
        if isinstance(target, ast.Name) and target.id == proof.name:
            statement.value = ast.copy_location(ast.Constant(value=value), statement.value)  # type: ignore[union-attr]
            replaced += 1
    if replaced != 1:
        raise ValueError(f"expected one {proof.scope} assignment for {proof.name!r}, replaced {replaced}")


class _ValueTransformer(ast.NodeTransformer):
    def __init__(self, target: str) -> None:
        self.target = target
        self.in_get_inputs = False
        self.replacements = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        previous = self.in_get_inputs
        self.in_get_inputs = node.name == "get_inputs"
        result = self.generic_visit(node)
        self.in_get_inputs = previous
        return result

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        previous = self.in_get_inputs
        self.in_get_inputs = node.name == "get_inputs"
        result = self.generic_visit(node)
        self.in_get_inputs = previous
        return result

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        assert isinstance(node, ast.Call)
        if not self.in_get_inputs or _call_name(node.func) not in _CONTINUOUS_FACTORIES:
            return node
        base = copy.deepcopy(node)
        assert isinstance(base.func, ast.Attribute)
        if self.target == "randn":
            base.func.attr = "randn"
            replacement: ast.AST = base
        elif self.target == "rand":
            base.func.attr = "rand"
            replacement = base
        elif self.target == "signed_uniform":
            base.func.attr = "rand"
            replacement = ast.BinOp(
                left=ast.BinOp(left=base, op=ast.Mult(), right=ast.Constant(value=2.0)),
                op=ast.Sub(),
                right=ast.Constant(value=1.0),
            )
        elif self.target == "boundary_pm1":
            base.func.attr = "randn"
            replacement = ast.Call(
                func=ast.Attribute(value=ast.Name(id="torch", ctx=ast.Load()), attr="sign", ctx=ast.Load()),
                args=[base],
                keywords=[],
            )
        else:  # pragma: no cover - policy validates this
            raise AssertionError(self.target)
        self.replacements += 1
        return ast.copy_location(replacement, node)


class _DtypeTransformer(ast.NodeTransformer):
    def __init__(self, target: str, eligible_locations: Sequence[tuple[int, int]]) -> None:
        self.target = target
        self.eligible_locations = frozenset(eligible_locations)
        self.in_get_inputs = False
        self.eligible_seen = 0
        self.replacements = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        previous = self.in_get_inputs
        self.in_get_inputs = node.name == "get_inputs"
        result = self.generic_visit(node)
        self.in_get_inputs = previous
        return result

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        previous = self.in_get_inputs
        self.in_get_inputs = node.name == "get_inputs"
        result = self.generic_visit(node)
        self.in_get_inputs = previous
        return result

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        assert isinstance(node, ast.Call)
        name = _call_name(node.func)
        if not self.in_get_inputs or name not in _FLOAT_FACTORIES:
            return node
        location = (getattr(node, "lineno", -1), getattr(node, "col_offset", -1))
        if location not in self.eligible_locations:
            return node
        keyword = next((item for item in node.keywords if item.arg == "dtype"), None)
        self.eligible_seen += 1
        current = _call_name(keyword.value) if keyword is not None else "torch.float32"
        target_aliases = {f"torch.{self.target}"}
        if self.target == "float16":
            target_aliases.add("torch.half")
        if current in target_aliases:
            return node
        dtype = ast.Attribute(
            value=ast.Name(id="torch", ctx=ast.Load()),
            attr=self.target,
            ctx=ast.Load(),
        )
        if keyword is None:
            node.keywords.append(ast.keyword(arg="dtype", value=dtype))
        else:
            keyword.value = dtype
        self.replacements += 1
        return node


class _LayoutTransformer(ast.NodeTransformer):
    def __init__(self) -> None:
        self.in_get_inputs = False
        self.replacements = 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        previous = self.in_get_inputs
        self.in_get_inputs = node.name == "get_inputs"
        result = self.generic_visit(node)
        self.in_get_inputs = previous
        return result

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        previous = self.in_get_inputs
        self.in_get_inputs = node.name == "get_inputs"
        result = self.generic_visit(node)
        self.in_get_inputs = previous
        return result

    def visit_Call(self, node: ast.Call) -> ast.AST:
        node = self.generic_visit(node)
        assert isinstance(node, ast.Call)
        if not self.in_get_inputs or _call_name(node.func) not in _CONTINUOUS_FACTORIES:
            return node
        base = copy.deepcopy(node)
        first = ast.Call(
            func=ast.Attribute(value=base, attr="transpose", ctx=ast.Load()),
            args=[ast.Constant(value=-1), ast.Constant(value=-2)],
            keywords=[],
        )
        contiguous = ast.Call(
            func=ast.Attribute(value=first, attr="contiguous", ctx=ast.Load()),
            args=[],
            keywords=[],
        )
        replacement = ast.Call(
            func=ast.Attribute(value=contiguous, attr="transpose", ctx=ast.Load()),
            args=[ast.Constant(value=-1), ast.Constant(value=-2)],
            keywords=[],
        )
        self.replacements += 1
        return ast.copy_location(replacement, node)


def _section_hashes(tree: ast.Module, entry_point: str) -> tuple[str, str]:
    model = _top_level_class(tree, entry_point)
    init = _top_level_function(tree, "get_init_inputs")
    return (
        _sha256_bytes(ast.dump(model, include_attributes=False).encode("utf-8")),
        _sha256_bytes(ast.dump(init, include_attributes=False).encode("utf-8")),
    )


def transform_code(
    code: str,
    analysis: CodeAnalysis,
    intervention: Intervention,
    policy: AugmentationPolicy,
    *,
    entry_point: str = "Model",
) -> tuple[str, CodeAnalysis, dict[str, Any]]:
    """Apply one coverage cell and verify the reference-only AST boundary."""

    tree = copy.deepcopy(analysis.tree)
    before_sections = _section_hashes(analysis.tree, entry_point)
    shape_before: int | None = None
    shape_after: int | None = None
    if intervention.shape_scale is not None:
        proof = analysis.shape_proof
        if proof is None:
            raise ValueError(analysis.shape_skip_reason or "shape_not_proven")
        shape_before = proof.before
        shape_after = proof.before * intervention.shape_scale
        if shape_after > policy.max_scaled_dimension:
            raise ValueError("scaled_dimension_above_policy_maximum")
        _replace_dimension_assignment(tree, proof, shape_after)

    value_replacements = 0
    if intervention.value_family is not None:
        if not analysis.value_transform_eligible:
            raise ValueError(analysis.value_skip_reason or "value_transform_not_proven")
        if intervention.value_family == analysis.source_value_family:
            raise ValueError("identity_value_family")
        # A rand-only parent establishes only non-negative input support.  Do
        # not introduce signed values without a separate semantic proof.
        if analysis.source_value_family == "rand" and intervention.value_family != "rand":
            raise ValueError("rand_parent_signed_domain_not_proven")
        transformer = _ValueTransformer(intervention.value_family)
        tree = transformer.visit(tree)  # type: ignore[assignment]
        ast.fix_missing_locations(tree)
        value_replacements = transformer.replacements
        if value_replacements != analysis.continuous_factory_count:
            raise ValueError("not_all_continuous_factories_transformed")

    dtype_replacements = 0
    layout_replacements = 0
    compatibility_proof: str | None = None
    if intervention.dtype_target is not None:
        if intervention.dtype_target not in DTYPE_TARGETS:
            raise ValueError("unsupported_dtype_target")
        if not analysis.dtype_transform_eligible:
            raise ValueError(analysis.dtype_skip_reason or "dtype_transform_not_proven")
        transformer = _DtypeTransformer(intervention.dtype_target, analysis.dtype_factory_locations)
        tree = transformer.visit(tree)  # type: ignore[assignment]
        ast.fix_missing_locations(tree)
        if transformer.eligible_seen != analysis.dtype_factory_count:
            raise ValueError("not_all_eligible_float_factories_inspected")
        dtype_replacements = transformer.replacements
        if dtype_replacements == 0:
            raise ValueError("identity_dtype_target")
        compatibility_proof = (
            "supported_real_floating_input_factory_dtype_mutation_only;" "operator_dtype_compatibility_execution_gated"
        )

    if intervention.layout_target is not None:
        if intervention.layout_target not in LAYOUT_TARGETS:
            raise ValueError("unsupported_layout_target")
        if not analysis.layout_transform_eligible:
            raise ValueError(analysis.layout_skip_reason or "layout_transform_not_proven")
        transformer = _LayoutTransformer()
        tree = transformer.visit(tree)  # type: ignore[assignment]
        ast.fix_missing_locations(tree)
        layout_replacements = transformer.replacements
        if layout_replacements != analysis.continuous_factory_count:
            raise ValueError("not_all_continuous_factories_made_non_contiguous")
        compatibility_proof = (
            "transpose_contiguous_transpose_preserves_shape_and_values;"
            "operator_stride_compatibility_execution_gated"
        )

    ast.fix_missing_locations(tree)
    child_code = ast.unparse(tree).rstrip() + "\n"
    child_analysis = analyze_code(child_code, entry_point)
    after_sections = _section_hashes(child_analysis.tree, entry_point)
    if before_sections != after_sections:
        raise ValueError("model_or_get_init_inputs_changed")
    if intervention.shape_scale is not None:
        if child_analysis.input_bytes <= analysis.input_bytes:
            raise ValueError("shape_intervention_did_not_increase_input_bytes")
        expected = analysis.input_bytes * intervention.shape_scale
        if child_analysis.input_bytes != expected:
            raise ValueError("shared_dimension_memory_scaling_invariant_failed")
    elif intervention.dtype_target is not None:
        if child_analysis.input_bytes > analysis.input_bytes:
            raise ValueError("dtype_intervention_increased_input_bytes")
    elif child_analysis.input_bytes != analysis.input_bytes:
        raise ValueError("non_dtype_intervention_changed_input_bytes")

    estimated_peak = math.ceil(child_analysis.input_bytes * policy.working_set_multiplier)
    if estimated_peak > policy.memory_budget_bytes:
        raise ValueError("memory_budget_exceeded")
    return (
        child_code,
        child_analysis,
        {
            "shape_before": shape_before,
            "shape_after": shape_after,
            "value_replacements": value_replacements,
            "dtype_replacements": dtype_replacements,
            "layout_replacements": layout_replacements,
            "compatibility_proof": compatibility_proof,
            "estimated_peak_bytes": estimated_peak,
        },
    )


def _replace_reference(messages: Any, old_code: str, new_code: str, *, required: bool) -> Any:
    if messages is None and not required:
        return None
    if not isinstance(messages, list) or not messages:
        raise ValueError("reference message collection is not a non-empty list")
    copied = [dict(message) for message in messages]
    occurrences = 0
    for message in copied:
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("reference prompt contains non-text content")
        count = content.count(old_code)
        occurrences += count
        if count:
            message["content"] = content.replace(old_code, new_code)
    if occurrences != 1:
        raise ValueError(f"expected exactly one reference occurrence in messages, found {occurrences}")
    return copied


def _coverage_cell(intervention: Intervention, analysis: CodeAnalysis) -> str:
    if intervention.dtype_target is not None:
        return f"dtype_{intervention.dtype_target}"
    if intervention.layout_target is not None:
        return f"layout_{intervention.layout_target}"
    shape = f"batch_x{intervention.shape_scale}" if intervention.shape_scale is not None else "shape_parent"
    value = intervention.value_family or analysis.source_value_family
    return f"{shape}|value_{value}"


def planned_interventions(analysis: CodeAnalysis, policy: AugmentationPolicy) -> list[Intervention]:
    """Return every bounded cell; ineligible cells are rejected later with evidence."""

    interventions: list[Intervention] = []
    for family in policy.value_families:
        if family != analysis.source_value_family:
            interventions.append(Intervention(None, family))
    for scale in policy.shape_scales:
        interventions.append(Intervention(scale, None))
        if policy.include_joint_cells:
            for family in policy.value_families:
                if family != analysis.source_value_family:
                    interventions.append(Intervention(scale, family))
    interventions.extend(Intervention(None, None, dtype_target=target) for target in policy.dtype_targets)
    interventions.extend(Intervention(None, None, layout_target=target) for target in policy.layout_targets)
    return interventions


def _extend_schema(schema: pa.Schema) -> pa.Schema:
    index = schema.get_field_index("extra_info")
    if index < 0:
        raise ValueError("input schema has no extra_info field")
    extra_field = schema.field(index)
    if not pa.types.is_struct(extra_field.type):
        raise ValueError("extra_info must be a struct")
    if extra_field.type.get_field_index("augmentation") >= 0:
        raise ValueError("input extra_info already contains augmentation metadata")
    extended = pa.struct([*extra_field.type, pa.field("augmentation", AUGMENTATION_METADATA_TYPE)])
    fields = list(schema)
    fields[index] = pa.field("extra_info", extended, nullable=extra_field.nullable, metadata=extra_field.metadata)
    return pa.schema(fields, metadata=schema.metadata)


def make_child_row(
    parent: Mapping[str, Any],
    *,
    parent_analysis: CodeAnalysis,
    child_code: str,
    child_analysis: CodeAnalysis,
    intervention: Intervention,
    transform_metadata: Mapping[str, Any],
    source_artifact_sha256: str,
    source_row_index: int,
    policy: AugmentationPolicy,
) -> dict[str, Any]:
    child = copy.deepcopy(dict(parent))
    reward_model = child.get("reward_model")
    extra_info = child.get("extra_info")
    if not isinstance(reward_model, dict) or not isinstance(extra_info, dict):
        raise ValueError("row must contain reward_model and extra_info structs")
    parent_code = reward_model.get("ground_truth")
    parent_uuid = extra_info.get("uuid")
    entry_point = extra_info.get("entry_point") or "Model"
    if not isinstance(parent_code, str) or not parent_code:
        raise ValueError("parent has no reward_model.ground_truth")
    if not isinstance(parent_uuid, str) or not parent_uuid:
        raise ValueError("parent has no extra_info.uuid")

    parent_reference_hash = _sha256_bytes(parent_code.encode("utf-8"))
    parent_ast_hash = _normalized_ast_sha256(parent_code)
    child_reference_hash = _sha256_bytes(child_code.encode("utf-8"))
    child_ast_hash = _normalized_ast_sha256(child_code)
    payload = intervention.payload(parent_analysis)
    canonical_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    intervention_hash = _sha256_bytes(canonical_payload.encode("utf-8"))
    intervention_id = f"inputcov_{intervention_hash[:20]}"
    child_uuid = f"aug_{_sha256_bytes(f'{parent_uuid}:{parent_reference_hash}:{intervention_hash}'.encode())[:24]}"
    coverage_cell = _coverage_cell(intervention, parent_analysis)

    reward_model["ground_truth"] = child_code
    child["prompt"] = _replace_reference(child.get("prompt"), parent_code, child_code, required=True)
    if extra_info.get("original_prompt") is not None:
        extra_info["original_prompt"] = _replace_reference(
            extra_info.get("original_prompt"), parent_code, child_code, required=False
        )
    extra_info["uuid"] = child_uuid

    if isinstance(extra_info.get("v4"), dict):
        v4 = dict(extra_info["v4"])
        if "reference_sha256" in v4:
            v4["reference_sha256"] = child_reference_hash
        if "normalized_ast_sha256" in v4:
            v4["normalized_ast_sha256"] = child_ast_hash
        extra_info["v4"] = v4

    proof = parent_analysis.shape_proof
    augmentation = {
        "contract_version": CONTRACT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "parent_uuid": parent_uuid,
        "child_uuid": child_uuid,
        "source_artifact_sha256": source_artifact_sha256,
        "source_row_index": source_row_index,
        "parent_reference_sha256": parent_reference_hash,
        "parent_normalized_ast_sha256": parent_ast_hash,
        "child_reference_sha256": child_reference_hash,
        "child_normalized_ast_sha256": child_ast_hash,
        "intervention_id": intervention_id,
        "intervention_sha256": intervention_hash,
        "intervention_kind": intervention.kind,
        "coverage_cell": coverage_cell,
        "shape_scale": intervention.shape_scale,
        "value_family_before": parent_analysis.source_value_family,
        "value_family_after": intervention.value_family or parent_analysis.source_value_family,
        "dtype_before": (
            ",".join(name.removeprefix("torch.") for name in parent_analysis.floating_input_dtypes)
            if intervention.dtype_target is not None
            else None
        ),
        "dtype_after": intervention.dtype_target,
        "layout_before": ("contiguous_factory_output" if intervention.layout_target is not None else None),
        "layout_after": intervention.layout_target,
        "shape_dimension_name": proof.name if intervention.shape_scale is not None and proof else None,
        "shape_dimension_before": transform_metadata.get("shape_before"),
        "shape_dimension_after": transform_metadata.get("shape_after"),
        "factory_count": child_analysis.factory_count,
        "input_bytes_before": parent_analysis.input_bytes,
        "input_bytes_after": child_analysis.input_bytes,
        "estimated_peak_bytes": transform_metadata["estimated_peak_bytes"],
        "memory_budget_bytes": policy.memory_budget_bytes,
        "working_set_multiplier": policy.working_set_multiplier,
        "memory_estimator_version": MEMORY_ESTIMATOR_VERSION,
        "shape_proof": proof.proof if intervention.shape_scale is not None and proof else None,
        "compatibility_proof": transform_metadata.get("compatibility_proof"),
        "validation_status": "static_fail_closed_pass_runtime_validation_required",
    }
    extra_info["augmentation"] = augmentation
    fatal, _ = inspect_row_schema(child, child_code)
    if fatal:
        raise ValueError(f"child row schema validation failed: {fatal}")
    if _top_level_class(ast.parse(child_code), str(entry_point)).name != entry_point:
        raise ValueError("child entry point changed")
    return child


def augment_row(
    parent: Mapping[str, Any],
    *,
    source_artifact_sha256: str,
    source_row_index: int,
    policy: AugmentationPolicy,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate all safe cells for one parent plus a decision for every cell."""

    extra_info = parent.get("extra_info")
    reward_model = parent.get("reward_model")
    parent_uuid = extra_info.get("uuid") if isinstance(extra_info, Mapping) else None
    code = reward_model.get("ground_truth") if isinstance(reward_model, Mapping) else None
    entry_point = extra_info.get("entry_point", "Model") if isinstance(extra_info, Mapping) else "Model"
    if not isinstance(parent_uuid, str) or not parent_uuid:
        raise ValueError(f"row {source_row_index} has no UUID")
    if not isinstance(code, str) or not code:
        raise ValueError(f"row {source_row_index} has no reference code")
    if not isinstance(entry_point, str) or not entry_point.isidentifier():
        raise ValueError(f"row {source_row_index} has invalid entry point")
    try:
        analysis = analyze_code(code, entry_point)
    except (SyntaxError, ValueError) as exc:
        return [], [
            {
                "source_row_index": source_row_index,
                "parent_uuid": parent_uuid,
                "coverage_cell": "all",
                "keep": False,
                "reason": f"parent_static_analysis_failed:{type(exc).__name__}:{exc}",
            }
        ]

    decisions: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    seen_child_hashes: set[str] = set()
    for intervention in planned_interventions(analysis, policy):
        cell = _coverage_cell(intervention, analysis)
        try:
            child_code, child_analysis, transform_metadata = transform_code(
                code,
                analysis,
                intervention,
                policy,
                entry_point=entry_point,
            )
            child_hash = _sha256_bytes(child_code.encode("utf-8"))
            if child_hash in seen_child_hashes:
                raise ValueError("duplicate_child_reference")
            child = make_child_row(
                parent,
                parent_analysis=analysis,
                child_code=child_code,
                child_analysis=child_analysis,
                intervention=intervention,
                transform_metadata=transform_metadata,
                source_artifact_sha256=source_artifact_sha256,
                source_row_index=source_row_index,
                policy=policy,
            )
            seen_child_hashes.add(child_hash)
            children.append(child)
            decisions.append(
                {
                    "source_row_index": source_row_index,
                    "parent_uuid": parent_uuid,
                    "child_uuid": child["extra_info"]["uuid"],
                    "coverage_cell": cell,
                    "keep": True,
                    "reason": "static_fail_closed_pass_runtime_validation_required",
                    "estimated_peak_bytes": transform_metadata["estimated_peak_bytes"],
                }
            )
        except (SyntaxError, ValueError) as exc:
            decisions.append(
                {
                    "source_row_index": source_row_index,
                    "parent_uuid": parent_uuid,
                    "coverage_cell": cell,
                    "keep": False,
                    "reason": str(exc),
                }
            )
    return children, decisions


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def build_augmentations(
    input_path: Path,
    output_path: Path,
    *,
    policy: AugmentationPolicy,
    manifest_path: Path | None = None,
    profile_path: Path | None = None,
    dry_run: bool = False,
    overwrite: bool = False,
    max_rows: int | None = None,
    batch_size: int = 256,
) -> dict[str, Any]:
    """Profile or materialize deterministic child rows from one parquet."""

    if max_rows is not None and max_rows < 0:
        raise ValueError("max_rows must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    manifest_path = manifest_path or output_path.with_name(f"{output_path.stem}.manifest.jsonl")
    profile_path = profile_path or output_path.with_name(f"{output_path.stem}.profile.json")
    materialized_paths = (output_path, manifest_path, profile_path)
    paths_to_check = (profile_path,) if dry_run else materialized_paths
    for path in paths_to_check:
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")

    source_sha256 = _sha256_file(input_path)
    parquet = pq.ParquetFile(input_path)
    output_schema = _extend_schema(parquet.schema_arrow)
    source_rows = parquet.metadata.num_rows
    row_limit = source_rows if max_rows is None else min(source_rows, max_rows)
    output_tmp = output_path.with_name(f".{output_path.name}.tmp")
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.tmp")
    writer: pq.ParquetWriter | None = None
    manifest_handle: Any = None
    if not dry_run:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(output_tmp, output_schema, compression="zstd", use_dictionary=True)
        manifest_handle = manifest_tmp.open("w", encoding="utf-8")

    scanned = 0
    children_written = 0
    parents_with_children: set[str] = set()
    coverage_counts: collections.Counter[str] = collections.Counter()
    skip_reasons: collections.Counter[str] = collections.Counter()
    decision_counts: collections.Counter[str] = collections.Counter()
    try:
        for batch in parquet.iter_batches(batch_size=batch_size, use_threads=False):
            child_batch: list[dict[str, Any]] = []
            for row in batch.to_pylist():
                if scanned >= row_limit:
                    break
                children, decisions = augment_row(
                    row,
                    source_artifact_sha256=source_sha256,
                    source_row_index=scanned,
                    policy=policy,
                )
                child_batch.extend(children)
                if children:
                    parents_with_children.add(str(row["extra_info"]["uuid"]))
                for decision in decisions:
                    if decision["keep"]:
                        coverage_counts[str(decision["coverage_cell"])] += 1
                        decision_counts["kept"] += 1
                    else:
                        skip_reasons[str(decision["reason"])] += 1
                        decision_counts["skipped"] += 1
                    if manifest_handle is not None:
                        manifest_handle.write(json.dumps(decision, sort_keys=True) + "\n")
                scanned += 1
            if writer is not None and child_batch:
                writer.write_table(pa.Table.from_pylist(child_batch, schema=output_schema))
            children_written += len(child_batch)
            if scanned >= row_limit:
                break
    except BaseException:
        if writer is not None:
            writer.close()
        if manifest_handle is not None:
            manifest_handle.close()
        output_tmp.unlink(missing_ok=True)
        manifest_tmp.unlink(missing_ok=True)
        raise
    else:
        if writer is not None:
            writer.close()
        if manifest_handle is not None:
            manifest_handle.close()

    profile: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "parent_policy": "retain every canonical input parent; this artifact contains children only",
        "mixture_policy": "coverage cells only; no distribution matching, weighting, sampling, or downsampling",
        "runtime_boundary": "every child requires target-GPU cleanup/runtime validation before training",
        "input": str(input_path.resolve()),
        "input_sha256": source_sha256,
        "input_rows": source_rows,
        "rows_scanned": scanned,
        "dry_run": dry_run,
        "policy": policy.as_dict(),
        "parents_with_at_least_one_child": len(parents_with_children),
        "child_rows": children_written,
        "decision_counts": dict(sorted(decision_counts.items())),
        "coverage_cell_counts": dict(sorted(coverage_counts.items())),
        "skip_reason_counts": dict(sorted(skip_reasons.items())),
        "memory_estimator": {
            "version": MEMORY_ESTIMATOR_VERSION,
            "quantity": (
                "sum of exactly proven unique returned get_inputs tensor-storage bytes times " "working_set_multiplier"
            ),
            "static_proof": (
                "statement-order shape and dtype resolution; unique factory-to-return provenance; only exact "
                "generator value/layout wrappers are recognized"
            ),
            "boundary": "does not prove compiler workspace or model-parameter memory; runtime validation remains required",
        },
    }
    if not dry_run:
        os.replace(output_tmp, output_path)
        os.replace(manifest_tmp, manifest_path)
        profile.update(
            {
                "output": str(output_path.resolve()),
                "output_sha256": _sha256_file(output_path),
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": _sha256_file(manifest_path),
            }
        )
    _atomic_write_text(profile_path, json.dumps(profile, indent=2, sort_keys=True) + "\n")
    return profile


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--shape-scale", action="append", type=int, dest="shape_scales")
    parser.add_argument("--value-family", action="append", choices=VALUE_FAMILIES, dest="value_families")
    parser.add_argument("--dtype-target", action="append", choices=DTYPE_TARGETS, dest="dtype_targets")
    parser.add_argument("--layout-target", action="append", choices=LAYOUT_TARGETS, dest="layout_targets")
    parser.add_argument("--no-dtype-cells", action="store_true")
    parser.add_argument("--no-layout-cells", action="store_true")
    parser.add_argument("--no-joint-cells", action="store_true")
    parser.add_argument("--available-gpu-gib", type=float, default=70.0)
    parser.add_argument("--memory-safety-fraction", type=float, default=DEFAULT_MEMORY_SAFETY_FRACTION)
    parser.add_argument("--working-set-multiplier", type=float, default=DEFAULT_WORKING_SET_MULTIPLIER)
    parser.add_argument("--max-scaled-dimension", type=int, default=8192)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    available_bytes = math.floor(args.available_gpu_gib * 1024**3)
    policy = AugmentationPolicy(
        shape_scales=tuple(args.shape_scales or DEFAULT_SHAPE_SCALES),
        value_families=tuple(args.value_families or VALUE_FAMILIES),
        dtype_targets=() if args.no_dtype_cells else tuple(args.dtype_targets or DTYPE_TARGETS),
        layout_targets=() if args.no_layout_cells else tuple(args.layout_targets or LAYOUT_TARGETS),
        include_joint_cells=not args.no_joint_cells,
        available_gpu_bytes=available_bytes,
        memory_safety_fraction=args.memory_safety_fraction,
        working_set_multiplier=args.working_set_multiplier,
        max_scaled_dimension=args.max_scaled_dimension,
    )
    profile = build_augmentations(
        args.input,
        args.output,
        policy=policy,
        manifest_path=args.manifest,
        profile_path=args.profile,
        dry_run=args.dry_run,
        overwrite=args.overwrite,
        max_rows=args.max_rows,
        batch_size=args.batch_size,
    )
    print(json.dumps(profile, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
