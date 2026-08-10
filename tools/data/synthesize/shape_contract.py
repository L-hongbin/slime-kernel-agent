#!/usr/bin/env python3
"""Shared, model-independent contract for input-shape augmentation.

Keep this module free of generation, deployment, and artifact orchestration.
The historical static solvers and the current model-assisted lane import the
same fail-closed identity and storage checks from here.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from tools.data.synthesize.augment_prompt_tasks import (
    _SHAPE_FACTORIES,
    _call_name,
    _factory_records,
    _module_constant_environment,
    _normalized_ast_sha256,
    _shape_nodes,
    analyze_code,
)

MIN_INPUT_SCALE = 2.0
MEDIUM_INPUT_MIN_BYTES = 64 * 1024**2
MEDIUM_INPUT_MAX_BYTES = 256 * 1024**2
LARGE_INPUT_MAX_BYTES = 4 * 1024**3
TARGET_TOLERANCE_FRACTION = 0.25

# Retained verbatim for reproducibility of the historical static-solver runs.
TARGET_SALT = "prompt_tvm_v4_ai_shape_random_targets_v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _nested(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _operator_family(ops: Any) -> str:
    if isinstance(ops, str):
        try:
            ops = json.loads(ops)
        except json.JSONDecodeError:
            ops = [ops]
    text = " ".join(str(item) for item in (ops or [])).lower()
    if "conv" in text:
        return "convolution"
    if any(token in text for token in ("matmul", "linear", "bmm", "einsum", "mm")):
        return "matrix"
    if any(token in text for token in ("norm", "softmax", "mean", "sum", "max", "min")):
        return "reduction_normalization"
    if any(token in text for token in ("gather", "scatter", "index", "embedding", "sort", "topk")):
        return "indexing"
    if any(token in text for token in ("reshape", "view", "permute", "transpose", "cat", "stack")):
        return "shape_layout"
    return "pointwise_other"


def _sample_target_mib(uuid: str, variant: str, lower_mib: int, upper_mib: int) -> int:
    if lower_mib > upper_mib:
        raise ValueError(f"no_{variant}_target_available:{lower_mib}:{upper_mib}")
    digest = _sha256_bytes(f"{TARGET_SALT}:{uuid}:{variant}".encode())
    return lower_mib + int(digest[:16], 16) % (upper_mib - lower_mib + 1)


def _target_input_bytes(row: Mapping[str, Any]) -> dict[str, int]:
    """Historical integer-MiB targets used by the static solver artifacts."""

    uuid = _nested(row, "extra_info.uuid")
    code = _nested(row, "reward_model.ground_truth")
    entry_point = str(_nested(row, "extra_info.entry_point", "Model"))
    if not isinstance(uuid, str) or not isinstance(code, str):
        raise ValueError("target_sampling_requires_parent_uuid_and_reference")
    parent_bytes = int(analyze_code(code, entry_point).input_bytes)
    mib = 1024**2
    minimum_growth_mib = (int(MIN_INPUT_SCALE * parent_bytes) + mib - 1) // mib
    medium_lower_mib = max(MEDIUM_INPUT_MIN_BYTES // mib, minimum_growth_mib)
    large_lower_mib = max(MEDIUM_INPUT_MAX_BYTES // mib + 1, minimum_growth_mib)
    return {
        "medium": _sample_target_mib(uuid, "medium", medium_lower_mib, MEDIUM_INPUT_MAX_BYTES // mib) * mib,
        "large": _sample_target_mib(uuid, "large", large_lower_mib, LARGE_INPUT_MAX_BYTES // mib) * mib,
    }


class _StructureNormalizer(ast.NodeTransformer):
    def __init__(self, mutable_numeric_node_ids: set[int]) -> None:
        self.mutable_numeric_node_ids = mutable_numeric_node_ids

    def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802
        if id(node) in self.mutable_numeric_node_ids:
            return ast.copy_location(ast.Constant(value=0), node)
        return node


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            if isinstance(node, ast.AsyncFunctionDef):
                raise ValueError(f"{name}_must_not_be_async")
            return node
    raise ValueError(f"missing_{name}")


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise ValueError(f"missing_entry_point_{name}")


def _assigned_names(target: ast.AST) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return {name for item in target.elts for name in _assigned_names(item)}
    return set()


def _numeric_constants(node: ast.AST) -> list[ast.Constant]:
    return [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Constant) and isinstance(item.value, (int, float)) and not isinstance(item.value, bool)
    ]


def _shape_numeric_node_ids(tree: ast.Module) -> set[int]:
    """Return numeric literals proved to feed input-factory shapes directly."""

    get_inputs = _function(tree, "get_inputs")
    mutable: set[int] = set()
    linked_names: set[str] = set()
    for node in ast.walk(get_inputs):
        if not isinstance(node, ast.Call) or _call_name(node.func) not in _SHAPE_FACTORIES:
            continue
        for shape_node in _shape_nodes(node, _call_name(node.func)):
            mutable.update(id(item) for item in _numeric_constants(shape_node))
            linked_names.update(item.id for item in ast.walk(shape_node) if isinstance(item, ast.Name))

    assignments: list[tuple[set[str], ast.AST]] = []
    for statement in [*tree.body, *ast.walk(get_inputs)]:
        if isinstance(statement, ast.Assign):
            names = {name for target in statement.targets for name in _assigned_names(target)}
            assignments.append((names, statement.value))
        elif isinstance(statement, ast.AnnAssign) and statement.value is not None:
            assignments.append((_assigned_names(statement.target), statement.value))

    changed = True
    while changed:
        changed = False
        for names, value in assignments:
            if not names.intersection(linked_names):
                continue
            before = len(linked_names)
            mutable.update(id(item) for item in _numeric_constants(value))
            linked_names.update(item.id for item in ast.walk(value) if isinstance(item, ast.Name))
            changed |= len(linked_names) != before
    return mutable


def _shape_normalized_dumps(parent_tree: ast.Module, child_tree: ast.Module) -> tuple[str, str]:
    """Normalize proven shape literals and consistently linked init literals."""

    parent = copy.deepcopy(parent_tree)
    child = copy.deepcopy(child_tree)
    parent_mutable = _shape_numeric_node_ids(parent)
    child_mutable = _shape_numeric_node_ids(child)
    parent_shape_constants = [
        item for item in ast.walk(parent) if isinstance(item, ast.Constant) and id(item) in parent_mutable
    ]
    child_shape_constants = [
        item for item in ast.walk(child) if isinstance(item, ast.Constant) and id(item) in child_mutable
    ]
    change_pairs: set[tuple[int | float, int | float]] = set()
    if len(parent_shape_constants) == len(child_shape_constants):
        change_pairs = {
            (before.value, after.value)
            for before, after in zip(parent_shape_constants, child_shape_constants, strict=True)
            if before.value != after.value
        }

    parent_init = _numeric_constants(_function(parent, "get_init_inputs"))
    child_init = _numeric_constants(_function(child, "get_init_inputs"))
    if len(parent_init) == len(child_init):
        for before, after in zip(parent_init, child_init, strict=True):
            if before.value != after.value and (before.value, after.value) in change_pairs:
                parent_mutable.add(id(before))
                child_mutable.add(id(after))

    parent = _StructureNormalizer(parent_mutable).visit(parent)
    child = _StructureNormalizer(child_mutable).visit(child)
    ast.fix_missing_locations(parent)
    ast.fix_missing_locations(child)
    return (
        ast.dump(parent, annotate_fields=True, include_attributes=False),
        ast.dump(child, annotate_fields=True, include_attributes=False),
    )


def _factory_names(function: ast.FunctionDef) -> list[str]:
    return [
        name
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and (name := _call_name(node.func)).startswith("torch.")
    ]


def relaxed_factory_storage(code: str, entry_point: str = "Model") -> tuple[int, int]:
    """Measure resolved factories without requiring bare returned-input provenance.

    This intentionally relaxes only ``analyze_code``'s return-expression proof.
    Factory shapes, dtypes, and storage still have to resolve exactly.
    """

    tree = ast.parse(code)
    _class(tree, entry_point)
    get_inputs = _function(tree, "get_inputs")
    records, _ = _factory_records(tree, get_inputs, _module_constant_environment(tree))
    if not records:
        raise ValueError("no_resolved_input_factory")
    return sum(record.bytes for record in records), len(records)


def _shape_structure_gate(parent_code: str, child_code: str, entry_point: str) -> dict[str, Any]:
    """Prove that the two references differ only in input-shape numbers."""

    parent_tree, child_tree = ast.parse(parent_code), ast.parse(child_code)
    parent_class, child_class = _class(parent_tree, entry_point), _class(child_tree, entry_point)
    if ast.dump(parent_class, annotate_fields=True, include_attributes=False) != ast.dump(
        child_class, annotate_fields=True, include_attributes=False
    ):
        raise ValueError("model_changed")
    parent_normalized, child_normalized = _shape_normalized_dumps(parent_tree, child_tree)
    if parent_normalized != child_normalized:
        raise ValueError("reference_changed_beyond_proven_input_shape_numbers")
    parent_inputs = _function(parent_tree, "get_inputs")
    child_inputs = _function(child_tree, "get_inputs")
    _function(parent_tree, "get_init_inputs")
    _function(child_tree, "get_init_inputs")
    if _factory_names(parent_inputs) != _factory_names(child_inputs):
        raise ValueError("torch_factory_or_random_method_sequence_changed")
    return {
        "parent_reference_sha256": _sha256_bytes(parent_code.encode()),
        "child_reference_sha256": _sha256_bytes(child_code.encode()),
        "child_normalized_ast_sha256": _normalized_ast_sha256(child_code),
    }


def relaxed_structure_gate(parent_code: str, child_code: str, entry_point: str = "Model") -> dict[str, Any]:
    """Apply shape-only identity checks without asserting returned storage."""

    result = _shape_structure_gate(parent_code, child_code, entry_point)
    _, parent_factory_count = relaxed_factory_storage(parent_code, entry_point)
    _, child_factory_count = relaxed_factory_storage(child_code, entry_point)
    if parent_factory_count != child_factory_count:
        raise ValueError("input_factory_count_changed")
    return {
        **result,
        "factory_count": parent_factory_count,
        "input_storage_contract": "fake_returned_tensor_storage_v1",
    }


def _static_gate(
    parent_code: str,
    child_code: str,
    entry_point: str,
    *,
    relaxed_return_provenance: bool,
    parent_input_bytes: int | None = None,
    child_input_bytes: int | None = None,
) -> dict[str, Any]:
    """Fail closed unless shape-only identity and storage growth are proven."""

    structure = _shape_structure_gate(parent_code, child_code, entry_point)
    if relaxed_return_provenance:
        _, parent_factory_count = relaxed_factory_storage(parent_code, entry_point)
        _, child_factory_count = relaxed_factory_storage(child_code, entry_point)
        if parent_input_bytes is None or child_input_bytes is None:
            raise ValueError("relaxed_gate_requires_fake_returned_tensor_storage")
        parent_bytes, child_bytes = parent_input_bytes, child_input_bytes
        storage_contract = "fake_returned_tensor_storage_v1"
    else:
        parent_analysis = analyze_code(parent_code, entry_point)
        child_analysis = analyze_code(child_code, entry_point)
        parent_bytes, parent_factory_count = parent_analysis.input_bytes, parent_analysis.factory_count
        child_bytes, child_factory_count = child_analysis.input_bytes, child_analysis.factory_count
        storage_contract = "proven_returned_input_storage_v1"
    if parent_factory_count != child_factory_count:
        raise ValueError("input_factory_count_changed")
    if parent_bytes <= 0 or child_bytes <= 0:
        raise ValueError("returned_input_storage_must_be_positive")
    if child_bytes <= parent_bytes:
        raise ValueError("returned_input_storage_did_not_increase")
    scale = child_bytes / parent_bytes
    if scale < MIN_INPUT_SCALE:
        raise ValueError(f"input_storage_scale_below_minimum:{scale:.4f}")
    return {
        "input_bytes_before": parent_bytes,
        "input_bytes_after": child_bytes,
        "input_scale": scale,
        "input_storage_contract": storage_contract,
        **structure,
    }


def static_gate(parent_code: str, child_code: str, entry_point: str = "Model") -> dict[str, Any]:
    """Use the strict returned-input storage proof for established lanes."""

    return _static_gate(parent_code, child_code, entry_point, relaxed_return_provenance=False)


def relaxed_static_gate(
    parent_code: str,
    child_code: str,
    entry_point: str = "Model",
    *,
    parent_input_bytes: int,
    child_input_bytes: int,
) -> dict[str, Any]:
    """Use exact FakeTensor-returned storage for wrapped-return references."""

    return _static_gate(
        parent_code,
        child_code,
        entry_point,
        relaxed_return_provenance=True,
        parent_input_bytes=parent_input_bytes,
        child_input_bytes=child_input_bytes,
    )


def _validate_variant_storage(variant: str, input_bytes: int) -> None:
    if variant == "medium" and not MEDIUM_INPUT_MIN_BYTES <= input_bytes <= MEDIUM_INPUT_MAX_BYTES:
        raise ValueError(f"medium_input_storage_out_of_range:{input_bytes}")
    if variant == "large" and not MEDIUM_INPUT_MAX_BYTES < input_bytes <= LARGE_INPUT_MAX_BYTES:
        raise ValueError(f"large_input_storage_out_of_range:{input_bytes}")


def _validate_target_proximity(input_bytes: int, target_input_bytes: int) -> float:
    if target_input_bytes <= 0:
        raise ValueError("target_input_storage_must_be_positive")
    relative_error = abs(input_bytes - target_input_bytes) / target_input_bytes
    if relative_error > TARGET_TOLERANCE_FRACTION:
        raise ValueError(
            "target_input_storage_outside_tolerance:" f"{input_bytes}:{target_input_bytes}:{relative_error:.6f}"
        )
    return relative_error
