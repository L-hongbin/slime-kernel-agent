#!/usr/bin/env python3
"""Build deterministic review-only layout siblings for canonical prompt tasks.

The proof domain is deliberately small: every returned tensor must be a direct
supported factory proven by ``augment_prompt_tasks``.  A stable parent hash
assigns exactly one family; there is no fallback family selection.  Runtime
parent/child equivalence, layout realization, and output liveness remain
mandatory gates before any training use.
"""

from __future__ import annotations

import argparse
import ast
import collections
import copy
import dataclasses
import difflib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from tools.data.cleaning.pipeline import inspect_row_schema
from tools.data.synthesize.augment_prompt_tasks import (
    AUGMENTATION_METADATA_TYPE,
    _call_name,
    _factory_records,
    _module_constant_environment,
    _normalized_ast_sha256,
    _prove_returned_input_storage,
    _replace_reference,
    _safe_scalar,
    _section_hashes,
    _sha256_bytes,
    _sha256_file,
    _top_level_class,
    _top_level_function,
)
from tools.data.synthesize.serial_random_wrappers import logical_factory_view, root_base_call
from tools.data.synthesize.serial_source_contract import canonical_sha256 as _serial_canonical_sha256
from tools.data.synthesize.serial_source_contract import verify_serial_source

CONTRACT_VERSION = "layout_direct_factory_solver_v1"
SERIAL_CONTRACT_VERSION = "layout_serial_random_factory_solver_v2"
GENERATOR_VERSION = "exact_layout_wrapper_ast_v1"
SERIAL_GENERATOR_VERSION = "source_span_layout_wrapper_over_serial_value_dtype_v2"
ASSIGNMENT_VERSION = "stable_parent_hash_over_eligible_layout_families_v1"
SELECTION_VERSION = "rare_expand_reserve_then_source_operator_family_proportional_v1"
EXPECTED_CANONICAL_PARENT_SHA256 = "b07205fcadc543964cfc7ee5fd9c1e4d011f0f3481447f656e5297e40b4b99f4"
EXPECTED_CANONICAL_PARENT_ROWS = 64_315
MAX_AUTHORIZED_CANDIDATES = 5_000
LAYOUT_FAMILIES = ("transpose_noncontiguous", "slice_storage_offset", "expand_zero_stride")
_REPO_ROOT = Path(__file__).resolve().parents[4]
_CANONICAL_PARENT = _REPO_ROOT / "Data/prompt_tvm_v4/train.review.parquet"
_REJECTED_KEYWORDS = frozenset({"requires_grad", "pin_memory", "pinned", "out", "layout", "memory_format"})
_EXPAND_FACTORIES = frozenset({"torch.zeros", "torch.ones", "torch.full"})
_MODEL_ALLOWED_IMPORT_ROOTS = frozenset({"math", "torch", "typing"})
_MODEL_ALLOWED_TORCH_IMPORT_PREFIXES = ("torch.nn",)
_MODEL_ALLOWED_TORCH_EXPORTS = frozenset({"Tensor", "nn"})
_MODEL_FORBIDDEN_TORCH_PREFIXES = (
    "torch._c",
    "torch._dynamo",
    "torch.autograd",
    "torch.classes",
    "torch.compile",
    "torch.compiler",
    "torch.distributed",
    "torch.export",
    "torch.fx",
    "torch.from_dlpack",
    "torch.hub",
    "torch.jit",
    "torch.library",
    "torch.onnx",
    "torch.ops",
    "torch.overrides",
    "torch.utils",
)
_MODEL_LAYOUT_OBSERVERS = frozenset(
    {
        "__getattr__",
        "__getattribute__",
        "_is_zerotensor",
        "_version",
        "data",
        "stride",
        "storage_offset",
        "is_contiguous",
        "is_conj",
        "is_neg",
        "data_ptr",
        "storage",
        "untyped_storage",
        "as_strided",
    }
)
_MODEL_MATERIALIZERS = frozenset({"contiguous", "clone", "_to_copy", "copy_", "flatten", "ravel", "reshape", "view"})
_MODEL_DYNAMIC_CALLS = frozenset(
    {"__import__", "eval", "exec", "getattr", "globals", "hasattr", "locals", "setattr", "vars"}
)
_MODEL_MUTATION_DUNDERS = frozenset(
    {
        "__delattr__",
        "__delitem__",
        "__iadd__",
        "__iand__",
        "__ifloordiv__",
        "__ilshift__",
        "__imatmul__",
        "__imod__",
        "__imul__",
        "__ior__",
        "__ipow__",
        "__irshift__",
        "__isub__",
        "__itruediv__",
        "__ixor__",
        "__setattr__",
        "__setitem__",
    }
)


@dataclasses.dataclass(frozen=True)
class EligibleParent:
    source_row_index: int
    parent_uuid: str
    parent_reference_sha256: str
    parent_normalized_ast_sha256: str
    eligible_families: tuple[str, ...]
    assigned_family: str
    assignment_sha256: str
    selection_sha256: str
    source_family: str
    operator_bucket: str
    operator_count: int | None
    factory_specs: tuple[tuple[str, tuple[int, ...], str], ...]
    transformed_factory_indices: tuple[int, ...]
    expected_layout: tuple[dict[str, Any], ...]
    serial_random_wrapper_count: int

    @property
    def stratum(self) -> tuple[str, str, str]:
        return (self.source_family, self.operator_bucket, self.assigned_family)

    @property
    def layout_application_scope(self) -> str:
        if self.assigned_family == "expand_zero_stride":
            return "eligible_constant_factory_leaves_only"
        return "all_direct_factory_leaves"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _nested(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _git_commit() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot bind layout generation to the current Git commit") from exc
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError(f"git rev-parse returned an invalid commit:{commit!r}")
    return commit


def _git_blob_sha256(commit: str, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(_REPO_ROOT)
        content = subprocess.check_output(
            ["git", "show", f"{commit}:{relative.as_posix()}"], cwd=_REPO_ROOT, stderr=subprocess.DEVNULL
        )
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot read source from Git commit:{path}") from exc
    return _sha256_bytes(content)


def _reserved_layout_name_conflict(tree: ast.Module) -> bool:
    identifiers = [node.id for node in ast.walk(tree) if isinstance(node, ast.Name)]
    identifiers.extend(node.arg for node in ast.walk(tree) if isinstance(node, ast.arg))
    return any(identifier.startswith("__layout_base_") for identifier in identifiers)


def _torch_aliases(tree: ast.Module) -> frozenset[str]:
    aliases = {"torch"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update(alias.asname or "torch" for alias in node.names if alias.name == "torch")
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            value = node.value
            if not isinstance(value, ast.Name) or value.id not in aliases:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id not in aliases:
                    aliases.add(target.id)
                    changed = True
    return frozenset(aliases)


def _normalize_torch_name(name: str, torch_aliases: frozenset[str]) -> str:
    root, separator, remainder = name.partition(".")
    if root not in torch_aliases:
        return name
    return "torch" + (separator + remainder if separator else "")


def _allowed_init_attribute_targets(tree: ast.Module) -> frozenset[int]:
    result: set[int] = set()
    for function in ast.walk(tree):
        if not isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef)) or function.name != "__init__":
            continue
        for node in ast.walk(function):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    result.add(id(target))
    return frozenset(result)


def _model_layout_compatibility_proof(tree: ast.Module, entry_point: str) -> tuple[bool, str | None]:
    """Reject statically visible module paths whose behavior can depend on layout.

    This intentionally scans helpers as well as ``Model``.  A helper invoked
    from ``forward`` is otherwise an easy path around a class-only check.
    """

    try:
        _top_level_class(tree, entry_point)
    except ValueError as exc:
        return False, f"model_lookup_failed:{exc}"
    torch_aliases = _torch_aliases(tree)
    allowed_init_targets = _allowed_init_attribute_targets(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.AugAssign):
            return False, "model_augmented_assignment"
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.Delete)):
            targets: list[ast.expr]
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            else:
                targets = node.targets
            walked_targets = list(ast.walk(ast.Tuple(elts=targets)))
            if any(isinstance(target, ast.Subscript) for target in walked_targets):
                return False, "model_subscript_mutation"
            attribute_targets = [target for target in walked_targets if isinstance(target, ast.Attribute)]
            if attribute_targets and (
                isinstance(node, ast.Delete)
                or any(id(target) not in allowed_init_targets for target in attribute_targets)
            ):
                return False, "model_attribute_mutation_outside_init"
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root not in _MODEL_ALLOWED_IMPORT_ROOTS:
                    return False, f"model_opaque_import:{root}"
                module = alias.name.lower()
                if root == "torch" and not (
                    module == "torch"
                    or module.startswith(tuple(f"{prefix}." for prefix in _MODEL_ALLOWED_TORCH_IMPORT_PREFIXES))
                    or module in _MODEL_ALLOWED_TORCH_IMPORT_PREFIXES
                ):
                    return False, f"model_opaque_torch_import:{alias.name}"
        if isinstance(node, ast.ImportFrom):
            if not node.module:
                return False, "model_relative_import"
            root = node.module.split(".", 1)[0]
            if root not in _MODEL_ALLOWED_IMPORT_ROOTS:
                return False, f"model_opaque_import:{root}"
            module = node.module.lower()
            if any(alias.name == "*" for alias in node.names):
                return False, f"model_star_import:{node.module}"
            if root == "torch":
                if module == "torch":
                    if any(alias.name not in _MODEL_ALLOWED_TORCH_EXPORTS for alias in node.names):
                        return False, f"model_opaque_torch_import:{node.module}"
                elif not (
                    module in _MODEL_ALLOWED_TORCH_IMPORT_PREFIXES
                    or module.startswith(tuple(f"{prefix}." for prefix in _MODEL_ALLOWED_TORCH_IMPORT_PREFIXES))
                ):
                    return False, f"model_opaque_torch_import:{node.module}"
        if isinstance(node, ast.Attribute):
            if node.attr in _MODEL_LAYOUT_OBSERVERS:
                return False, f"model_layout_storage_observer:{node.attr}"
            attribute_name = _normalize_torch_name(_call_name(node), torch_aliases).lower()
            if attribute_name.startswith(_MODEL_FORBIDDEN_TORCH_PREFIXES):
                return False, f"model_opaque_torch_path:{attribute_name}"
        if not isinstance(node, ast.Call):
            continue
        if any(keyword.arg == "out" for keyword in node.keywords):
            return False, "model_out_keyword"
        if any(keyword.arg is None for keyword in node.keywords):
            return False, "model_expanded_keyword_arguments"
        if any(isinstance(argument, ast.Starred) for argument in node.args):
            return False, "model_expanded_positional_arguments"
        name = _call_name(node.func)
        if name.rsplit(".", 1)[-1] in _MODEL_DYNAMIC_CALLS:
            return False, f"model_dynamic_dispatch:{name}"
        if isinstance(node.func, ast.Attribute):
            method = node.func.attr
            if method in _MODEL_MUTATION_DUNDERS:
                return False, f"model_mutation_dunder:{method}"
            if method in _MODEL_MATERIALIZERS:
                return False, f"model_early_materialization:{method}"
            if method.endswith("_") and not method.startswith("__"):
                return False, f"model_inplace_or_alias_sensitive_call:{method}"
    return True, None


def _assignment(parent_uuid: str, reference_sha256: str, eligible_families: Sequence[str]) -> tuple[str, str]:
    eligible_set = set(eligible_families)
    normalized = tuple(family for family in LAYOUT_FAMILIES if family in eligible_set)
    if len(normalized) != len(eligible_families) or not normalized:
        raise ValueError("eligible_families_must_be_nonempty_unique_known_families")
    choice_payload = {
        "assignment_version": ASSIGNMENT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "parent_uuid": parent_uuid,
        "parent_reference_sha256": reference_sha256,
        "eligible_families": list(normalized),
    }
    choice_sha256 = _canonical_sha256(choice_payload)
    assigned_family = normalized[int(choice_sha256[:16], 16) % len(normalized)]
    assignment_sha256 = _canonical_sha256(
        {
            **choice_payload,
            "choice_sha256": choice_sha256,
            "assigned_family": assigned_family,
        }
    )
    return assigned_family, assignment_sha256


def _selection_hash(parent_uuid: str, reference_sha256: str, family: str) -> str:
    return _canonical_sha256(
        {
            "selection_version": SELECTION_VERSION,
            "parent_uuid": parent_uuid,
            "parent_reference_sha256": reference_sha256,
            "assigned_family": family,
        }
    )


def _factory_calls(tree: ast.Module, get_inputs: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[int, ast.Call]:
    return {
        id(node): node
        for node in ast.walk(get_inputs)
        if isinstance(node, ast.Call) and _call_name(node.func).startswith("torch.")
    }


def _static_full_value(call: ast.Call, environment: Mapping[str, Any]) -> bool:
    value_node = next((item.value for item in call.keywords if item.arg == "fill_value"), None)
    if value_node is None and len(call.args) >= 2:
        value_node = call.args[1]
    if value_node is None:
        return False
    try:
        value = _safe_scalar(value_node, environment)
    except ValueError:
        return False
    return isinstance(value, (bool, int, float, complex))


def _direct_factory_proof(
    tree: ast.Module,
) -> tuple[tuple[Any, ...] | None, Mapping[str, Any] | None, str | None]:
    """Accept only unwrapped returned leaves from the shared exact proof."""

    try:
        get_inputs = _top_level_function(tree, "get_inputs")
        environment = _module_constant_environment(tree)
        records, local_environment = _factory_records(tree, get_inputs, environment)
        if not records:
            return None, None, "no_supported_input_factory"
        if not _prove_returned_input_storage(get_inputs, records, local_environment):
            return None, None, "returned_factory_leaf_has_generated_wrapper"
    except (SyntaxError, ValueError) as exc:
        return None, None, f"returned_factory_proof_failed:{type(exc).__name__}:{exc}"
    calls = _factory_calls(tree, get_inputs)
    if set(calls) != {record.call_node_id for record in records}:
        return None, None, "unsupported_or_non_direct_factory_call"
    for record in records:
        call = calls[record.call_node_id]
        if record.name == "torch.empty":
            return None, None, "empty_factory_rejected"
        keywords = {item.arg for item in call.keywords if item.arg is not None}
        bad = sorted(keywords & _REJECTED_KEYWORDS)
        if bad:
            return None, None, f"factory_keyword_rejected:{bad[0]}"
    return tuple(records), local_environment, None


def _logical_factory_proof(
    tree: ast.Module,
) -> tuple[
    tuple[Any, ...] | None,
    Mapping[str, Any] | None,
    Mapping[int, ast.Call] | None,
    tuple[ast.Call, ...] | None,
    int,
    str | None,
]:
    """Expose frozen random/value wrappers as their one logical input leaf."""

    try:
        view = logical_factory_view(tree)
    except ValueError as exc:
        return None, None, None, None, 0, f"serial_logical_factory_view_failed:{exc}"
    records, environment, reason = _direct_factory_proof(view.tree)
    if records is None:
        return None, None, None, None, view.random_wrapper_count, reason
    virtual_get_inputs = _top_level_function(view.tree, "get_inputs")
    virtual_calls = _factory_calls(view.tree, virtual_get_inputs)
    if len(view.original_roots) != len(records):
        return (
            None,
            None,
            None,
            None,
            view.random_wrapper_count,
            f"serial_logical_factory_count_mismatch:{len(view.original_roots)}:{len(records)}",
        )
    for record, root in zip(records, view.original_roots, strict=True):
        virtual_call = virtual_calls.get(record.call_node_id)
        if virtual_call is None or ast.dump(root_base_call(root), include_attributes=False) != ast.dump(
            virtual_call, include_attributes=False
        ):
            return None, None, None, None, view.random_wrapper_count, "serial_logical_factory_order_mismatch"
    return (
        records,
        environment,
        virtual_calls,
        view.original_roots,
        view.random_wrapper_count,
        None,
    )


def _contiguous_strides(shape: Sequence[int]) -> tuple[int, ...]:
    stride = 1
    result: list[int] = []
    for size in reversed(shape):
        result.append(stride)
        stride *= size
    return tuple(reversed(result))


def _is_contiguous(shape: Sequence[int], strides: Sequence[int]) -> bool:
    required = 1
    for size, stride in zip(reversed(shape), reversed(strides), strict=True):
        if size != 1:
            if stride != required:
                return False
            required *= size
    return True


def _leaf_metadata(
    index: int,
    record: Any,
    strides: Sequence[int],
    storage_offset: int,
    zero_stride_dimensions: Sequence[int],
    **extra: Any,
) -> dict[str, Any]:
    shape = record.shape
    contiguous = _is_contiguous(shape, strides)
    return {
        "factory_index": index,
        "logical_shape": list(shape),
        "dtype": record.dtype_name,
        "expected_strides": list(strides),
        "expected_storage_offset": storage_offset,
        "expected_is_contiguous": contiguous,
        # Retained for compatibility with the first analyzer draft.
        "expected_contiguous": contiguous,
        "expected_storage_offset_positive": storage_offset > 0,
        "expected_zero_stride_dimensions": list(zero_stride_dimensions),
        **extra,
    }


def _family_metadata(
    family: str, records: Sequence[Any], calls: Mapping[int, ast.Call], environment: Mapping[str, Any], digest: str
) -> tuple[tuple[dict[str, Any], ...] | None, str | None]:
    shapes = [record.shape for record in records]
    if family == "transpose_noncontiguous":
        if any(len(shape) < 2 or shape[-2] <= 1 or shape[-1] <= 1 for shape in shapes):
            return None, "transpose_requires_rank_two_nontrivial_trailing_dims"
        return (
            tuple(
                _leaf_metadata(
                    index,
                    record,
                    (*_contiguous_strides(shape)[:-2], 1, shape[-2]),
                    0,
                    (),
                )
                for index, (record, shape) in enumerate(zip(records, shapes, strict=True))
            ),
            None,
        )
    if family == "slice_storage_offset":
        if any(not shape or shape[-1] <= 0 for shape in shapes):
            return None, "slice_requires_rank_at_least_one"
        return (
            tuple(
                _leaf_metadata(
                    index,
                    record,
                    (*_contiguous_strides((*shape[:-1], shape[-1] + 1))[:-1], 1),
                    1,
                    (),
                )
                for index, (record, shape) in enumerate(zip(records, shapes, strict=True))
            ),
            None,
        )
    if family != "expand_zero_stride":
        return None, f"unsupported_family:{family}"
    metadata: list[dict[str, Any]] = []
    for index, (record, shape) in enumerate(zip(records, shapes, strict=True)):
        if record.name not in _EXPAND_FACTORIES:
            continue
        if record.name == "torch.full" and not _static_full_value(calls[record.call_node_id], environment):
            continue
        dimensions = [dimension for dimension, size in enumerate(shape) if size > 1]
        if not dimensions:
            continue
        leaf_digest = _canonical_sha256(
            {
                "assignment_sha256": digest,
                "family": family,
                "factory_index": index,
                "factory": record.name,
                "shape": list(shape),
                "dtype": record.dtype_name,
            }
        )
        dimension = dimensions[int(leaf_digest[:16], 16) % len(dimensions)]
        base_strides = _contiguous_strides(shape)
        metadata.append(
            _leaf_metadata(
                index,
                record,
                (*base_strides[:dimension], 0, *base_strides[dimension + 1 :]),
                0,
                (dimension,),
                expand_dimension=dimension,
            )
        )
    if not metadata:
        return None, "expand_requires_constant_leaf_with_nontrivial_dimension"
    return tuple(metadata), None


def _eligible_families(
    records: Sequence[Any], calls: Mapping[int, ast.Call], environment: Mapping[str, Any]
) -> tuple[str, ...]:
    """Return the proven family set before selecting exactly one family."""

    eligibility_digest = "0" * 64
    result: list[str] = []
    for family in LAYOUT_FAMILIES:
        metadata, reason = _family_metadata(family, records, calls, environment, eligibility_digest)
        if metadata is not None and reason is None:
            result.append(family)
    return tuple(result)


def _analyze_parent(row: Mapping[str, Any], source_row_index: int) -> tuple[EligibleParent | None, str]:
    parent_uuid = _nested(row, "extra_info.uuid")
    code = _nested(row, "reward_model.ground_truth")
    entry_point = _nested(row, "extra_info.entry_point", "Model")
    if not isinstance(parent_uuid, str) or not parent_uuid:
        return None, "missing_parent_uuid"
    if not isinstance(code, str) or not code:
        return None, "missing_parent_reference"
    if not isinstance(entry_point, str) or not entry_point.isidentifier():
        return None, "invalid_entry_point"
    try:
        tree = ast.parse(code)
        _top_level_function(tree, "get_inputs")
    except (SyntaxError, ValueError) as exc:
        return None, f"parent_ast_failed:{type(exc).__name__}:{exc}"
    parent_reference_sha256 = _sha256_bytes(code.encode("utf-8"))
    if _reserved_layout_name_conflict(tree):
        return None, "reserved_layout_base_identifier_conflict"
    model_compatible, reason = _model_layout_compatibility_proof(tree, entry_point)
    if not model_compatible:
        return None, reason or "model_layout_compatibility_not_proven"
    records, environment, calls, _, wrapper_count, reason = _logical_factory_proof(tree)
    if records is None:
        return None, reason or "direct_factory_proof_failed"
    assert calls is not None
    eligible_families = _eligible_families(records, calls, environment or {})
    if not eligible_families:
        return None, "no_layout_family_eligible"
    family, assignment_sha256 = _assignment(parent_uuid, parent_reference_sha256, eligible_families)
    expected_layout, reason = _family_metadata(family, records, calls, environment or {}, assignment_sha256)
    if expected_layout is None:
        return None, reason or "assigned_layout_family_not_eligible"
    source_family = _nested(row, "extra_info.v4.source_family", "unknown")
    operator_bucket = _nested(row, "extra_info.v4.operator_bucket", "unknown")
    operator_count = _nested(row, "extra_info.v4.operator_count")
    return (
        EligibleParent(
            source_row_index=source_row_index,
            parent_uuid=parent_uuid,
            parent_reference_sha256=parent_reference_sha256,
            parent_normalized_ast_sha256=_normalized_ast_sha256(code),
            eligible_families=eligible_families,
            assigned_family=family,
            assignment_sha256=assignment_sha256,
            selection_sha256=_selection_hash(parent_uuid, parent_reference_sha256, family),
            source_family=source_family if isinstance(source_family, str) and source_family else "unknown",
            operator_bucket=operator_bucket if isinstance(operator_bucket, str) and operator_bucket else "unknown",
            operator_count=operator_count if type(operator_count) is int else None,
            factory_specs=tuple((record.name, record.shape, record.dtype_name) for record in records),
            transformed_factory_indices=tuple(int(item["factory_index"]) for item in expected_layout),
            expected_layout=expected_layout,
            serial_random_wrapper_count=wrapper_count,
        ),
        "eligible",
    )


def _select_proportional_stratified(eligible: Sequence[EligibleParent], limit: int) -> list[EligibleParent]:
    if limit >= len(eligible):
        return sorted(eligible, key=lambda item: item.source_row_index)
    groups: dict[tuple[str, str, str], list[EligibleParent]] = collections.defaultdict(list)
    for item in eligible:
        groups[item.stratum].append(item)
    for values in groups.values():
        values.sort(key=lambda item: (item.selection_sha256, item.source_row_index))
    keys = sorted(groups, key=_canonical_sha256)
    quotas = {key: 0 for key in keys}
    remaining = limit
    if limit >= len(keys):
        for key in keys:
            quotas[key] = 1
        remaining -= len(keys)
    capacities = {key: len(groups[key]) - quotas[key] for key in keys}
    total = sum(capacities.values())
    exact = {key: remaining * capacities[key] / total for key in keys} if total else {}
    for key in keys:
        add = min(capacities[key], int(exact.get(key, 0)))
        quotas[key] += add
        remaining -= add
    order = sorted(keys, key=lambda key: (-(exact.get(key, 0) % 1), _canonical_sha256(key)))
    while remaining:
        progressed = False
        for key in order:
            if quotas[key] == len(groups[key]):
                continue
            quotas[key] += 1
            remaining -= 1
            progressed = True
            if not remaining:
                break
        if not progressed:
            raise ValueError("stratified_layout_allocation_exhausted")
    return sorted(
        [item for key in keys for item in groups[key][: quotas[key]]], key=lambda item: item.source_row_index
    )


def _select_stratified(eligible: Sequence[EligibleParent], limit: int) -> list[EligibleParent]:
    """Reserve rare assigned expand rows, then stratify the remaining quota."""

    if limit >= len(eligible):
        return sorted(eligible, key=lambda item: item.source_row_index)
    expand = sorted(
        (item for item in eligible if item.assigned_family == "expand_zero_stride"),
        key=lambda item: (item.selection_sha256, item.source_row_index),
    )
    other_families = {item.assigned_family for item in eligible if item.assigned_family != "expand_zero_stride"}
    expand_quota = min(len(expand), max(0, limit - len(other_families)))
    reserved = expand[:expand_quota]
    reserved_rows = {item.source_row_index for item in reserved}
    remaining_pool = [item for item in eligible if item.source_row_index not in reserved_rows]
    selected = [
        *reserved,
        *_select_proportional_stratified(remaining_pool, limit - len(reserved)),
    ]
    if len(selected) != limit:
        raise ValueError(f"selected_layout_count_mismatch:{len(selected)}:{limit}")
    return sorted(selected, key=lambda item: item.source_row_index)


def _attr(value: ast.expr, name: str) -> ast.Attribute:
    return ast.Attribute(value=value, attr=name, ctx=ast.Load())


def _call(value: ast.expr, name: str, *args: ast.expr, **keywords: ast.expr) -> ast.Call:
    return ast.Call(
        func=_attr(value, name),
        args=list(args),
        keywords=[ast.keyword(arg=key, value=item) for key, item in keywords.items()],
    )


def _torch_call(name: str, *args: ast.expr, **keywords: ast.expr) -> ast.Call:
    return ast.Call(
        func=ast.Attribute(value=ast.Name(id="torch", ctx=ast.Load()), attr=name, ctx=ast.Load()),
        args=list(args),
        keywords=[ast.keyword(arg=key, value=item) for key, item in keywords.items()],
    )


def _wrapper(base_name: str, family: str, spec: Mapping[str, Any]) -> ast.expr:
    base = ast.Name(id=base_name, ctx=ast.Load())
    if family == "transpose_noncontiguous":
        return _call(
            _call(_call(base, "transpose", ast.Constant(-1), ast.Constant(-2)), "contiguous"),
            "transpose",
            ast.Constant(-1),
            ast.Constant(-2),
        )
    if family == "slice_storage_offset":
        first = _call(base, "narrow", ast.Constant(-1), ast.Constant(0), ast.Constant(1))
        zeros = _torch_call("zeros_like", first)
        padded = _torch_call("cat", ast.Tuple(elts=[zeros, base], ctx=ast.Load()), dim=ast.Constant(-1))
        return _call(padded, "narrow", ast.Constant(-1), ast.Constant(1), ast.Constant(spec["logical_shape"][-1]))
    if family == "expand_zero_stride":
        dimension = int(spec["expand_dimension"])
        narrowed = _call(base, "narrow", ast.Constant(dimension), ast.Constant(0), ast.Constant(1))
        return _call(
            narrowed,
            "expand",
            ast.Tuple(elts=[ast.Constant(value=value) for value in spec["logical_shape"]], ctx=ast.Load()),
        )
    raise ValueError(f"unsupported_family:{family}")


def _byte_span(code: str, node: ast.AST) -> tuple[int, int]:
    if None in (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset):
        raise ValueError("source_span_is_unavailable")
    lines = code.encode("utf-8").splitlines(keepends=True)
    start = sum(len(line) for line in lines[: node.lineno - 1]) + node.col_offset
    end = sum(len(line) for line in lines[: node.end_lineno - 1]) + node.end_col_offset
    return start, end


def _replace_byte_spans(code: str, replacements: Sequence[tuple[int, int, bytes]]) -> str:
    raw = code.encode("utf-8")
    ordered = sorted(replacements, reverse=True)
    for index, (start, end, replacement) in enumerate(ordered):
        if not 0 <= start <= end <= len(raw):
            raise ValueError("source_replacement_span_is_invalid")
        if index and end > ordered[index - 1][0]:
            raise ValueError("source_replacement_spans_overlap")
        raw = raw[:start] + replacement + raw[end:]
    return raw.decode("utf-8")


def _transform_layout(code: str, entry_point: str, parent: EligibleParent) -> str:
    """Wrap each target factory inline at its original evaluation position.

    The single-argument lambda evaluates the original factory exactly once
    without moving it across sibling expressions that may affect RNG or state.
    """

    tree = ast.parse(code)
    before = _section_hashes(tree, entry_point)
    if _reserved_layout_name_conflict(tree):
        raise ValueError("reserved_layout_base_identifier_conflict")
    records, environment, calls, original_roots, wrapper_count, reason = _logical_factory_proof(tree)
    if records is None:
        raise ValueError(reason or "replay_direct_factory_proof_failed")
    if wrapper_count != parent.serial_random_wrapper_count:
        raise ValueError("serial_random_wrapper_count_changed_during_replay")
    if _sha256_bytes(code.encode("utf-8")) != parent.parent_reference_sha256:
        raise ValueError("parent_reference_sha_changed_during_replay")
    assert calls is not None and original_roots is not None
    replay_eligible = _eligible_families(records, calls, environment or {})
    replay_family, replay_assignment_sha256 = _assignment(
        parent.parent_uuid, parent.parent_reference_sha256, replay_eligible
    )
    if (
        replay_eligible != parent.eligible_families
        or replay_family != parent.assigned_family
        or replay_assignment_sha256 != parent.assignment_sha256
    ):
        raise ValueError("eligible_family_assignment_changed_during_replay")
    expected, reason = _family_metadata(
        parent.assigned_family, records, calls, environment or {}, parent.assignment_sha256
    )
    if expected != parent.expected_layout or reason is not None:
        raise ValueError("layout_expected_metadata_changed_during_replay")
    replay_indices = tuple(int(item["factory_index"]) for item in expected)
    if replay_indices != parent.transformed_factory_indices:
        raise ValueError("layout_target_factory_indices_changed_during_replay")
    spec_by_index = {int(item["factory_index"]): item for item in parent.expected_layout}
    source_replacements: list[tuple[int, int, bytes]] = []
    raw = code.encode("utf-8")
    for index, call in enumerate(original_roots):
        if index not in spec_by_index:
            continue
        start, end = _byte_span(code, call)
        base_name = f"__layout_base_{index}"
        wrapper_body = ast.unparse(_wrapper(base_name, parent.assigned_family, spec_by_index[index]))
        replacement = f"(lambda {base_name}: {wrapper_body})(".encode() + raw[start:end] + b")"
        source_replacements.append((start, end, replacement))
    if len(source_replacements) != len(parent.transformed_factory_indices):
        raise ValueError("layout_factory_rewrite_count_mismatch")
    child_code = _replace_byte_spans(code, source_replacements)
    if _section_hashes(ast.parse(child_code), entry_point) != before:
        raise ValueError("model_or_get_init_inputs_changed")
    return child_code


def _extend_schema(schema: pa.Schema) -> pa.Schema:
    index = schema.get_field_index("extra_info")
    if index < 0 or not pa.types.is_struct(schema.field(index).type):
        raise ValueError("canonical_schema_has_no_extra_info_struct")
    extra = schema.field(index)
    augmentation_index = extra.type.get_field_index("augmentation")
    if augmentation_index >= 0:
        if extra.type.field(augmentation_index).type != AUGMENTATION_METADATA_TYPE:
            raise ValueError("input_augmentation_metadata_has_incompatible_type")
        return schema
    fields = list(schema)
    fields[index] = pa.field(
        "extra_info",
        pa.struct([*extra.type, pa.field("augmentation", AUGMENTATION_METADATA_TYPE)]),
        nullable=extra.nullable,
        metadata=extra.metadata,
    )
    return pa.schema(fields, metadata=schema.metadata)


def _make_child(
    parent: Mapping[str, Any],
    eligible: EligibleParent,
    source_path: Path,
    source_sha256: str,
    generator_sha256: str,
    dependency_sha256: str,
    git_commit: str,
    source_binding: Mapping[str, Any] | None = None,
    source_row_binding: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    child = copy.deepcopy(dict(parent))
    serial_mode = source_binding is not None
    solver_contract = SERIAL_CONTRACT_VERSION if serial_mode else CONTRACT_VERSION
    solver_generator = SERIAL_GENERATOR_VERSION if serial_mode else GENERATOR_VERSION
    if serial_mode:
        if source_row_binding is None or source_row_binding.get("row_index") != eligible.source_row_index:
            raise ValueError("serial layout child lacks its exact upstream manifest row")
        upstream_manifest_sha256 = _serial_canonical_sha256(source_row_binding)
        lineage = f"{source_binding.get('stage')}_layout_child"
    else:
        if source_row_binding is not None:
            raise ValueError("canonical layout child unexpectedly has an upstream manifest row")
        upstream_manifest_sha256 = None
        lineage = "canonical_parent_layout_child"
    parent_code = _nested(parent, "reward_model.ground_truth")
    entry_point = _nested(parent, "extra_info.entry_point", "Model")
    if not isinstance(parent_code, str) or not isinstance(entry_point, str):
        raise ValueError("invalid_parent_code_or_entry_point")
    child_code = _transform_layout(parent_code, entry_point, eligible)
    child_reference_sha256 = _sha256_bytes(child_code.encode("utf-8"))
    child_ast_sha256 = _normalized_ast_sha256(child_code)
    intervention = {
        "primary_intervention": "layout",
        "eligible_families": list(eligible.eligible_families),
        "family": eligible.assigned_family,
        "factory_count": len(eligible.factory_specs),
        "transformed_factory_count": len(eligible.transformed_factory_indices),
        "target_factory_indices": list(eligible.transformed_factory_indices),
        "layout_application_scope": eligible.layout_application_scope,
        "logical_shape_preserved": True,
        "value_preserved": True,
        "dtype_preserved": True,
        "rng_consumption_preserved": True,
        "expected_metadata": eligible.expected_layout,
    }
    intervention_sha256 = _canonical_sha256(intervention)
    child_uuid = (
        "layout_"
        + _sha256_bytes(
            f"{solver_contract}:{eligible.parent_uuid}:{eligible.parent_reference_sha256}:{intervention_sha256}".encode()
        )[:24]
    )
    child["reward_model"] = dict(child["reward_model"])
    child["reward_model"]["ground_truth"] = child_code
    child["prompt"] = _replace_reference(child.get("prompt"), parent_code, child_code, required=True)
    extra = dict(child["extra_info"])
    if extra.get("original_prompt") is not None:
        extra["original_prompt"] = _replace_reference(
            extra["original_prompt"], parent_code, child_code, required=False
        )
    extra["uuid"] = child_uuid
    v4 = dict(extra.get("v4") or {})
    v4.update(
        {
            "parent_uuid": eligible.parent_uuid,
            "reference_sha256": child_reference_sha256,
            "normalized_ast_sha256": child_ast_sha256,
            "included_in_review_train": False,
            "runtime_validation_status": "layout_paired_runtime_pending",
            "governance_status": "layout_intervention_review_only",
        }
    )
    extra["v4"] = v4
    extra["augmentation"] = {
        "contract_version": solver_contract,
        "generator_version": solver_generator,
        "parent_uuid": eligible.parent_uuid,
        "child_uuid": child_uuid,
        "source_artifact_sha256": source_sha256,
        "source_row_index": eligible.source_row_index,
        "parent_reference_sha256": eligible.parent_reference_sha256,
        "parent_normalized_ast_sha256": eligible.parent_normalized_ast_sha256,
        "child_reference_sha256": child_reference_sha256,
        "child_normalized_ast_sha256": child_ast_sha256,
        "intervention_id": f"layout_{intervention_sha256[:20]}",
        "intervention_sha256": intervention_sha256,
        "intervention_kind": "layout",
        "coverage_cell": eligible.assigned_family,
        "shape_scale": None,
        "value_family_before": None,
        "value_family_after": None,
        "dtype_before": None,
        "dtype_after": None,
        "layout_before": "contiguous_direct_factory",
        "layout_after": eligible.assigned_family,
        "shape_dimension_name": None,
        "shape_dimension_before": None,
        "shape_dimension_after": None,
        "factory_count": len(eligible.factory_specs),
        "input_bytes_before": None,
        "input_bytes_after": None,
        "estimated_peak_bytes": None,
        "memory_budget_bytes": None,
        "working_set_multiplier": None,
        "memory_estimator_version": "layout_metadata_only_v1",
        "shape_proof": "logical_shapes_exactly_preserved",
        "compatibility_proof": "direct_returned_factory_proof;exact_family_wrapper_only;Model_and_get_init_inputs_frozen",
        "validation_status": "static_layout_pass_paired_runtime_layout_realization_and_liveness_required",
    }
    child["extra_info"] = extra
    fatal, _ = inspect_row_schema(child, child_code)
    if fatal:
        raise ValueError(f"child_schema_failed:{','.join(fatal)}")
    manifest = {
        "manifest_contract_version": "layout_lane_manifest_v1",
        "candidate_row_index": None,
        "source_artifact_path": str(source_path.resolve()),
        "source_artifact_sha256": source_sha256,
        "source_row_index": eligible.source_row_index,
        "parent_uuid": eligible.parent_uuid,
        "child_uuid": child_uuid,
        "parent_reference_sha256": eligible.parent_reference_sha256,
        "parent_normalized_ast_sha256": eligible.parent_normalized_ast_sha256,
        "child_reference_sha256": child_reference_sha256,
        "child_normalized_ast_sha256": child_ast_sha256,
        "primary_intervention": "layout",
        "eligible_families": list(eligible.eligible_families),
        "assigned_family": eligible.assigned_family,
        "factory_count": len(eligible.factory_specs),
        "transformed_factory_count": len(eligible.transformed_factory_indices),
        "target_factory_indices": list(eligible.transformed_factory_indices),
        "layout_application_scope": eligible.layout_application_scope,
        "realized_intervention": intervention,
        "expected_layout_metadata": eligible.expected_layout,
        "assignment_version": ASSIGNMENT_VERSION,
        "assignment_sha256": eligible.assignment_sha256,
        "selection_version": SELECTION_VERSION,
        "selection_sha256": eligible.selection_sha256,
        "source_family": eligible.source_family,
        "operator_bucket": eligible.operator_bucket,
        "operator_count": eligible.operator_count,
        "mode_class": _nested(parent, "extra_info.v4.mode_class"),
        "source_hashes": {
            "canonical_parent_artifact": source_sha256,
            "parent_reference": eligible.parent_reference_sha256,
            "parent_normalized_ast": eligible.parent_normalized_ast_sha256,
            "child_reference": child_reference_sha256,
            "child_normalized_ast": child_ast_sha256,
            "generator_source": generator_sha256,
            "augment_prompt_tasks_dependency": dependency_sha256,
        },
        "ast_hashes": {"parent": eligible.parent_normalized_ast_sha256, "child": child_ast_sha256},
        "generator_contract_version": solver_contract,
        "generator_version": solver_generator,
        "generator_source_sha256": generator_sha256,
        "dependency_source_sha256": dependency_sha256,
        "git_commit": git_commit,
        "shape_changed": False,
        "value_changed": False,
        "dtype_changed": False,
        "layout_changed": True,
        "model_changed": False,
        "get_init_inputs_changed": False,
        "rng_consumption_status": "static_exactly_once_runtime_pending",
        "static_status": "passed",
        "parent_runtime_status": "pending",
        "child_runtime_status": "pending",
        "layout_realization_status": "pending",
        "liveness_status": "pending",
        "materialization_status": "review_only",
        "training_approved": False,
    }
    if source_binding is not None:
        manifest["source_binding"] = dict(source_binding)
        manifest["upstream_serial_manifest_row_sha256"] = upstream_manifest_sha256
        manifest["lineage"] = lineage
    return child, manifest


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _review_markdown(samples: Sequence[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]]) -> str:
    lines = ["# Layout solver review samples", ""]
    for parent, child, manifest in samples:
        lines.extend(
            [
                f"## {manifest['child_uuid']}",
                "",
                f"- source: `{manifest['source_family']}`; operator: `{manifest['operator_bucket']}`; family: `{manifest['assigned_family']}`",
                "",
                "```diff",
                *difflib.unified_diff(
                    str(_nested(parent, "reward_model.ground_truth", "")).splitlines(),
                    str(_nested(child, "reward_model.ground_truth", "")).splitlines(),
                    fromfile="parent.py",
                    tofile="child.py",
                    lineterm="",
                ),
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def build_lane(
    input_path: Path,
    output_dir: Path,
    *,
    limit: int,
    overwrite: bool,
    serial_source_manifest: Path | None = None,
    excluded_source_rows: frozenset[int] = frozenset(),
) -> dict[str, Any]:
    if type(limit) is not int or not 1 <= limit <= MAX_AUTHORIZED_CANDIDATES:
        raise ValueError(f"limit must be an integer in [1, {MAX_AUTHORIZED_CANDIDATES}]")
    serial_mode = serial_source_manifest is not None
    if not serial_mode and input_path.resolve() != _CANONICAL_PARENT.resolve():
        raise ValueError(f"input must be canonical parent unless a serial manifest is supplied:{_CANONICAL_PARENT}")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    source_sha256 = _sha256_file(input_path)
    parquet = pq.ParquetFile(input_path)
    if serial_mode:
        assert serial_source_manifest is not None
        source_binding, source_row_bindings = verify_serial_source(
            input_path, serial_source_manifest, expected_stage="dtype_fallback_base"
        )
    else:
        source_binding = None
        source_row_bindings = None
        if source_sha256 != EXPECTED_CANONICAL_PARENT_SHA256:
            raise ValueError(f"canonical_parent_sha_mismatch:{source_sha256}")
        if parquet.metadata.num_rows != EXPECTED_CANONICAL_PARENT_ROWS:
            raise ValueError(f"canonical_parent_row_mismatch:{parquet.metadata.num_rows}")
    if any(type(index) is not int or not 0 <= index < parquet.metadata.num_rows for index in excluded_source_rows):
        raise ValueError("excluded_source_row_index_out_of_bounds")
    paths = {
        key: output_dir / name
        for key, name in {
            "parents": "parents.parquet",
            "candidates": "candidates.parquet",
            "paired": "paired.parquet",
            "manifest": "manifest.jsonl",
            "decisions": "decisions.jsonl",
            "summary": "summary.json",
            "review": "review_samples.md",
        }.items()
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing_existing_layout_artifacts:{existing[:3]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    generator_path = Path(__file__).resolve()
    dependency_path = generator_path.parent.parent / "augment_prompt_tasks.py"
    generator_sha256 = _sha256_file(generator_path)
    dependency_sha256 = _sha256_file(dependency_path)
    git_commit = _git_commit()
    if not serial_mode:
        for path, digest in ((generator_path, generator_sha256), (dependency_path, dependency_sha256)):
            if _git_blob_sha256(git_commit, path) != digest:
                raise RuntimeError(f"source differs from Git commit {git_commit}:{path}")
    eligible: list[EligibleParent] = []
    decisions: list[dict[str, Any]] = []
    skip_counts: collections.Counter[str] = collections.Counter()
    row_index = 0
    for batch in parquet.iter_batches(batch_size=256, use_threads=False):
        for row in batch.to_pylist():
            item, reason = _analyze_parent(row, row_index)
            if item is None:
                skip_counts[reason] += 1
                decisions.append(
                    {
                        "source_row_index": row_index,
                        "parent_uuid": _nested(row, "extra_info.uuid"),
                        "eligible": False,
                        "selected": False,
                        "eligible_families": [],
                        "assigned_family": None,
                        "reason": reason,
                    }
                )
            elif row_index in excluded_source_rows:
                decisions.append(
                    {
                        "source_row_index": row_index,
                        "parent_uuid": item.parent_uuid,
                        "eligible": True,
                        "selected": False,
                        "eligible_families": list(item.eligible_families),
                        "assigned_family": item.assigned_family,
                        "reason": "excluded_by_prior_serial_layout_batch",
                    }
                )
            else:
                eligible.append(item)
                decisions.append(
                    {
                        "source_row_index": row_index,
                        "parent_uuid": item.parent_uuid,
                        "eligible": True,
                        "selected": False,
                        "eligible_families": list(item.eligible_families),
                        "assigned_family": item.assigned_family,
                        "factory_count": len(item.factory_specs),
                        "transformed_factory_count": len(item.transformed_factory_indices),
                        "target_factory_indices": list(item.transformed_factory_indices),
                        "layout_application_scope": item.layout_application_scope,
                        "assignment_sha256": item.assignment_sha256,
                        "reason": "static_layout_eligible",
                    }
                )
            row_index += 1
    if len(eligible) < limit:
        raise ValueError(f"insufficient_eligible_layout_parents:{len(eligible)}:{limit}")
    selected = _select_stratified(eligible, limit)
    eligible_pool_families = {item.assigned_family for item in eligible}
    selected_families = {item.assigned_family for item in selected}
    if selected_families != eligible_pool_families:
        missing = sorted(eligible_pool_families - selected_families)
        raise ValueError(f"selection_does_not_cover_eligible_layout_families:{missing}")
    selected_by_row = {item.source_row_index: item for item in selected}
    for decision in decisions:
        if decision["eligible"]:
            if decision["reason"] == "excluded_by_prior_serial_layout_batch":
                continue
            decision["selected"] = decision["source_row_index"] in selected_by_row
            if not decision["selected"]:
                decision["reason"] = "eligible_not_selected_by_limit"
    schema = _extend_schema(parquet.schema_arrow)
    parents: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    review_pool: dict[tuple[str, str, str], tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]] = {}
    row_index = 0
    for batch in parquet.iter_batches(batch_size=256, use_threads=False):
        for row in batch.to_pylist():
            item = selected_by_row.get(row_index)
            if item is not None:
                child, manifest = _make_child(
                    row,
                    item,
                    input_path,
                    source_sha256,
                    generator_sha256,
                    dependency_sha256,
                    git_commit,
                    source_binding=source_binding,
                    source_row_binding=(source_row_bindings[row_index] if source_row_bindings is not None else None),
                )
                parent = copy.deepcopy(row)
                if not serial_mode:
                    parent["extra_info"] = dict(parent["extra_info"])
                    parent["extra_info"]["augmentation"] = None
                manifest["candidate_row_index"] = len(children)
                parents.append(parent)
                children.append(child)
                manifests.append(manifest)
                review_pool.setdefault(item.stratum, (row, child, manifest))
            row_index += 1
    if len(children) != limit:
        raise ValueError(f"materialized_layout_children_mismatch:{len(children)}:{limit}")
    for field in ("child_uuid", "child_reference_sha256", "child_normalized_ast_sha256"):
        values = [str(manifest[field]) for manifest in manifests]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate_manifest_field:{field}")
    paired = [item for pair in zip(parents, children, strict=True) for item in pair]
    suffix = f".tmp.{os.getpid()}"
    for key, rows in (("parents", parents), ("candidates", children), ("paired", paired)):
        temporary = paths[key].with_name(paths[key].name + suffix)
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd")
        os.replace(temporary, paths[key])
    _atomic_text(paths["manifest"], "".join(_canonical_json(item) + "\n" for item in manifests))
    _atomic_text(paths["decisions"], "".join(_canonical_json(item) + "\n" for item in decisions))
    _atomic_text(paths["review"], _review_markdown([review_pool[key] for key in sorted(review_pool)[:24]]))

    def count(items: Sequence[EligibleParent], attribute: str) -> dict[str, int]:
        return dict(sorted(collections.Counter(str(getattr(item, attribute)) for item in items).items()))

    summary = {
        "contract_version": SERIAL_CONTRACT_VERSION if serial_mode else CONTRACT_VERSION,
        "generator_version": SERIAL_GENERATOR_VERSION if serial_mode else GENERATOR_VERSION,
        "assignment_version": ASSIGNMENT_VERSION,
        "selection_version": SELECTION_VERSION,
        "input": str(input_path.resolve()),
        "input_sha256": source_sha256,
        "input_rows": parquet.metadata.num_rows,
        "source_binding": source_binding,
        "excluded_source_rows": len(excluded_source_rows),
        "generator_source_sha256": generator_sha256,
        "dependency_source_sha256": dependency_sha256,
        "git_commit": git_commit,
        "eligible_parents": len(eligible),
        "selected_parents": len(selected),
        "candidate_rows": len(children),
        "paired_rows": len(paired),
        "selection_limit": limit,
        "eligible_family_availability_counts": {
            family: sum(family in item.eligible_families for item in eligible) for family in LAYOUT_FAMILIES
        },
        "assigned_family_counts": count(eligible, "assigned_family"),
        "selected_family_counts": count(selected, "assigned_family"),
        "eligible_family_counts": count(eligible, "assigned_family"),
        "family_counts": count(selected, "assigned_family"),
        "source_counts": count(selected, "source_family"),
        "operator_bucket_counts": count(selected, "operator_bucket"),
        "skip_reason_counts": dict(sorted(skip_counts.items())),
        "proof": {
            "one_family_per_parent": True,
            "transpose_and_slice_application_scope": "all_direct_factory_leaves",
            "expand_application_scope": "all_and_only_eligible_constant_factory_leaves",
            "expand_non_target_leaves": "unchanged_without_added_binding",
            "rare_expand_selection": "reserve_all_assigned_expand_rows_when_limit_permits_family_coverage",
            "stratified_by": ["source_family", "operator_bucket", "family"],
            "model_frozen": True,
            "get_init_inputs_frozen": True,
            "runtime_boundary": "paired parent/child reference, logical value/dtype/shape equivalence, layout realization, and output liveness required",
        },
        "artifacts": {},
        "training_approved": False,
    }
    for key, path in paths.items():
        if key != "summary":
            summary["artifacts"][key] = {"path": str(path.resolve()), "sha256": _sha256_file(path)}
    _atomic_text(paths["summary"], json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--limit", type=int, required=True, help="Canary size in [1, 5000]")
    parser.add_argument("--serial-source-manifest", type=Path)
    parser.add_argument("--exclude-source-row-file", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    excluded_source_rows = frozenset()
    if args.exclude_source_row_file is not None:
        excluded_source_rows = frozenset(
            int(line) for line in args.exclude_source_row_file.read_text(encoding="utf-8").splitlines() if line.strip()
        )
    print(
        json.dumps(
            build_lane(
                args.input,
                args.output_dir,
                limit=args.limit,
                overwrite=args.overwrite,
                serial_source_manifest=args.serial_source_manifest,
                excluded_source_rows=excluded_source_rows,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
