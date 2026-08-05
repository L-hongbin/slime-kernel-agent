#!/usr/bin/env python3
"""Build deterministic low-precision siblings for statically parameter-free tasks.

The canary deliberately implements only the proof-friendly half of the dtype
plan: every floating direct input factory is FP32, every returned input factory
has exact provenance, and the reference ``Model`` has no statically visible
parameter, buffer, tensor state, tensor constant, or dtype cast.  One target
(``float16`` or ``bfloat16``) is assigned per parent.  ``Model`` and
``get_init_inputs`` remain byte-for-byte equivalent at the AST level.

Runtime validation is still mandatory.  In particular, this static solver does
not claim that an operator supports the target dtype or that its implementation
avoids an implicit FP32 fallback.
"""

from __future__ import annotations

import argparse
import ast
import collections
import copy
import dataclasses
import difflib
import json
import math
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
    AugmentationPolicy,
    Intervention,
    _call_name,
    _factory_records,
    _module_constant_environment,
    _normalized_ast_sha256,
    _prove_returned_input_storage,
    _replace_reference,
    _section_hashes,
    _sha256_bytes,
    _sha256_file,
    _top_level_class,
    _top_level_function,
    analyze_code,
    transform_code,
)

CONTRACT_VERSION = "dtype_parameter_free_solver_v1"
GENERATOR_VERSION = "all_direct_float_inputs_low_precision_v1"
ASSIGNMENT_VERSION = "stable_parent_hash_single_dtype_v1"
SELECTION_VERSION = "source_operator_dtype_proportional_largest_remainder_v1"
EXPECTED_CANONICAL_PARENT_SHA256 = "b07205fcadc543964cfc7ee5fd9c1e4d011f0f3481447f656e5297e40b4b99f4"
EXPECTED_CANONICAL_PARENT_ROWS = 64_315
DTYPE_TARGETS = ("float16", "bfloat16")
MAX_AUTHORIZED_CANDIDATES = 5_000
RUNTIME_MEMORY_BUDGET_BYTES = 64 * 1024**3
_REPO_ROOT = Path(__file__).resolve().parents[4]

_FP32_NAMES = frozenset({"torch.float32", "torch.float"})
_LOW_PRECISION_NAMES = {
    "float16": frozenset({"torch.float16", "torch.half"}),
    "bfloat16": frozenset({"torch.bfloat16"}),
}
_DEFAULT_DTYPE_MUTATORS = frozenset({"set_default_dtype", "set_default_tensor_type"})
_MODEL_DTYPE_NAMES = frozenset(
    {
        "torch.float",
        "torch.float16",
        "torch.half",
        "torch.bfloat16",
        "torch.float32",
        "torch.float64",
        "torch.double",
    }
)
_MODEL_TENSOR_FACTORIES = frozenset(
    {
        "torch.arange",
        "torch.asarray",
        "torch.as_tensor",
        "torch.empty",
        "torch.empty_like",
        "torch.empty_strided",
        "torch.eye",
        "torch.from_numpy",
        "torch.full",
        "torch.full_like",
        "torch.linspace",
        "torch.logspace",
        "torch.normal",
        "torch.ones",
        "torch.ones_like",
        "torch.poisson",
        "torch.rand",
        "torch.rand_like",
        "torch.randint",
        "torch.randint_like",
        "torch.randn",
        "torch.randn_like",
        "torch.randperm",
        "torch.scalar_tensor",
        "torch.tensor",
        "torch.zeros",
        "torch.zeros_like",
    }
)
_CAST_METHODS = frozenset({"bfloat16", "double", "float", "half", "to", "type", "type_as"})
_SAFE_INIT_CALLS = frozenset({"bool", "float", "int", "len", "list", "str", "super", "tuple"})
DTYPE_POLICY = AugmentationPolicy(
    shape_scales=(),
    value_families=(),
    dtype_targets=DTYPE_TARGETS,
    layout_targets=(),
    include_joint_cells=False,
)


@dataclasses.dataclass(frozen=True)
class EligibleParent:
    source_row_index: int
    parent_uuid: str
    parent_reference_sha256: str
    parent_normalized_ast_sha256: str
    assigned_dtype: str
    assignment_sha256: str
    selection_sha256: str
    source_family: str
    operator_bucket: str
    operator_count: int | None
    factory_count: int
    transformed_factory_count: int
    input_bytes_before: int
    input_bytes_after: int
    factory_specs: tuple[tuple[str, tuple[int, ...], str], ...]

    @property
    def stratum(self) -> tuple[str, str, str]:
        return (self.source_family, self.operator_bucket, self.assigned_dtype)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode())


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
        raise RuntimeError("cannot bind dtype generation to the current Git commit") from exc
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError(f"git rev-parse returned an invalid commit: {commit!r}")
    return commit


def _git_blob_sha256(commit: str, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(_REPO_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"source path is outside the repository: {path}") from exc
    try:
        content = subprocess.check_output(
            ["git", "show", f"{commit}:{relative.as_posix()}"], cwd=_REPO_ROOT, stderr=subprocess.DEVNULL
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot read {relative} from Git commit {commit}") from exc
    return _sha256_bytes(content)


def _has_default_dtype_mutation(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "torch":
            if any(alias.name in _DEFAULT_DTYPE_MUTATORS for alias in node.names):
                return True
        if isinstance(node, ast.Attribute) and node.attr in _DEFAULT_DTYPE_MUTATORS:
            return True
        if isinstance(node, ast.Name) and node.id in _DEFAULT_DTYPE_MUTATORS:
            return True
        if isinstance(node, ast.Constant) and node.value in _DEFAULT_DTYPE_MUTATORS:
            return True
    return False


def _is_super_init(call: ast.Call) -> bool:
    if not isinstance(call.func, ast.Attribute) or call.func.attr != "__init__":
        return False
    value = call.func.value
    return isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "super"


def _parameter_free_model_proof(tree: ast.Module, entry_point: str) -> tuple[bool, str | None]:
    """Reject every statically visible source of model tensor state or FP32 fallback."""

    if _has_default_dtype_mutation(tree):
        return False, "dynamic_default_dtype"
    model = _top_level_class(tree, entry_point)
    init_methods = [
        node
        for node in model.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__"
    ]
    if len(init_methods) > 1:
        return False, "multiple_model_init_methods"
    if init_methods:
        for node in ast.walk(init_methods[0]):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node.func)
            if _is_super_init(node) or name in {"nn.Module.__init__", "torch.nn.Module.__init__"}:
                continue
            if name in _SAFE_INIT_CALLS:
                continue
            return False, f"model_init_call_not_parameter_free:{name or 'dynamic'}"

    get_inputs = _top_level_function(tree, "get_inputs")
    get_init_inputs = _top_level_function(tree, "get_init_inputs")
    excluded_ids = {id(node) for root in (model, get_inputs, get_init_inputs) for node in ast.walk(root)}
    for node in ast.walk(get_init_inputs):
        if isinstance(node, ast.Call):
            return False, f"get_init_inputs_call_not_scalar_proven:{_call_name(node.func) or 'dynamic'}"
        if isinstance(node, ast.Attribute) and _call_name(node) in _MODEL_DTYPE_NAMES:
            return False, "get_init_inputs_contains_dtype_literal"

    for node in ast.walk(model):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in {"nn.Parameter", "torch.nn.Parameter"}:
                return False, "model_parameter_constructor"
            if name.endswith(".register_buffer") or name.endswith(".register_parameter"):
                return False, "model_registered_tensor_state"
            if (name.startswith("nn.") or name.startswith("torch.nn.")) and name not in {
                "nn.Module.__init__",
                "torch.nn.Module.__init__",
            }:
                return False, f"model_nn_call_not_parameter_free:{name}"
            if name in _MODEL_TENSOR_FACTORIES:
                return False, f"model_tensor_factory:{name}"
            if any(keyword.arg == "dtype" for keyword in node.keywords):
                return False, "model_explicit_dtype_keyword"
            if isinstance(node.func, ast.Attribute) and node.func.attr in _CAST_METHODS:
                return False, f"model_dtype_cast_or_to:{node.func.attr}"
        if isinstance(node, ast.Attribute) and _call_name(node) in _MODEL_DTYPE_NAMES:
            return False, f"model_dtype_literal:{_call_name(node)}"

    for node in ast.walk(tree):
        if id(node) in excluded_ids:
            continue
        if isinstance(node, ast.Call) and _call_name(node.func) in _MODEL_TENSOR_FACTORIES:
            return False, f"module_tensor_constant:{_call_name(node.func)}"
        if isinstance(node, ast.Attribute) and _call_name(node) in _MODEL_DTYPE_NAMES:
            return False, f"module_dtype_literal:{_call_name(node)}"
    return True, None


def _resolved_factory_specs(tree: ast.Module) -> tuple[tuple[str, tuple[int, ...], str], ...]:
    get_inputs = _top_level_function(tree, "get_inputs")
    records, environment = _factory_records(tree, get_inputs, _module_constant_environment(tree))
    if not _prove_returned_input_storage(get_inputs, records, environment):
        raise ValueError("returned_input_factory_is_wrapped")
    return tuple((record.name, record.shape, record.dtype_name) for record in records)


def _assignment(parent_uuid: str, reference_sha256: str) -> tuple[str, str]:
    payload = {
        "assignment_version": ASSIGNMENT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "parent_uuid": parent_uuid,
        "parent_reference_sha256": reference_sha256,
    }
    digest = _canonical_sha256(payload)
    return DTYPE_TARGETS[int(digest[:16], 16) % len(DTYPE_TARGETS)], digest


def _selection_hash(parent_uuid: str, reference_sha256: str, target: str) -> str:
    return _canonical_sha256(
        {
            "selection_version": SELECTION_VERSION,
            "parent_uuid": parent_uuid,
            "parent_reference_sha256": reference_sha256,
            "assigned_dtype": target,
        }
    )


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
    reference_sha256 = _sha256_bytes(code.encode())
    try:
        analysis = analyze_code(code, entry_point)
    except (SyntaxError, ValueError) as exc:
        return None, f"parent_static_analysis_failed:{type(exc).__name__}:{exc}"
    if not analysis.dtype_transform_eligible:
        return None, analysis.dtype_skip_reason or "dtype_transform_not_proven"
    if not analysis.floating_input_dtypes or any(name not in _FP32_NAMES for name in analysis.floating_input_dtypes):
        return None, "floating_direct_inputs_not_all_fp32"
    compatible, reason = _parameter_free_model_proof(analysis.tree, entry_point)
    if not compatible:
        return None, reason or "parameter_free_model_not_proven"
    try:
        specs = _resolved_factory_specs(analysis.tree)
    except ValueError as exc:
        return None, f"returned_input_proof_failed:{exc}"
    transformed = tuple(spec for spec in specs if spec[2] in _FP32_NAMES)
    if len(transformed) != analysis.dtype_factory_count:
        return None, "transformed_factory_count_mismatch"
    if not transformed:
        return None, "no_fp32_direct_input_factory"
    target, assignment_sha256 = _assignment(parent_uuid, reference_sha256)
    input_bytes_after = analysis.input_bytes - sum(2 * _numel(shape) for _, shape, _ in transformed)
    if math.ceil(input_bytes_after * DTYPE_POLICY.working_set_multiplier) > DTYPE_POLICY.memory_budget_bytes:
        return None, "memory_budget_exceeded"
    source_family = _nested(row, "extra_info.v4.source_family", "unknown")
    operator_bucket = _nested(row, "extra_info.v4.operator_bucket", "unknown")
    operator_count = _nested(row, "extra_info.v4.operator_count")
    if not isinstance(source_family, str) or not source_family:
        source_family = "unknown"
    if not isinstance(operator_bucket, str) or not operator_bucket:
        operator_bucket = "unknown"
    if isinstance(operator_count, bool) or not isinstance(operator_count, int):
        operator_count = None
    return (
        EligibleParent(
            source_row_index=source_row_index,
            parent_uuid=parent_uuid,
            parent_reference_sha256=reference_sha256,
            parent_normalized_ast_sha256=_normalized_ast_sha256(code),
            assigned_dtype=target,
            assignment_sha256=assignment_sha256,
            selection_sha256=_selection_hash(parent_uuid, reference_sha256, target),
            source_family=source_family,
            operator_bucket=operator_bucket,
            operator_count=operator_count,
            factory_count=analysis.factory_count,
            transformed_factory_count=len(transformed),
            input_bytes_before=analysis.input_bytes,
            input_bytes_after=input_bytes_after,
            factory_specs=specs,
        ),
        "eligible",
    )


def _numel(shape: Sequence[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _select_stratified(eligible: Sequence[EligibleParent], limit: int) -> list[EligibleParent]:
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
    total_capacity = sum(capacities.values())
    exact = {key: remaining * capacities[key] / total_capacity for key in keys} if total_capacity else {}
    for key in keys:
        addition = min(capacities[key], int(exact.get(key, 0)))
        quotas[key] += addition
        remaining -= addition
    remainder_order = sorted(
        keys, key=lambda key: (-(exact.get(key, 0) - int(exact.get(key, 0))), _canonical_sha256(key))
    )
    while remaining:
        progressed = False
        for key in remainder_order:
            if quotas[key] >= len(groups[key]):
                continue
            quotas[key] += 1
            remaining -= 1
            progressed = True
            if not remaining:
                break
        if not progressed:
            raise ValueError("stratified allocation exhausted all dtype strata")
    selected = [item for key in keys for item in groups[key][: quotas[key]]]
    if len(selected) != limit:
        raise ValueError(f"selected {len(selected)} rows, expected {limit}")
    return sorted(selected, key=lambda item: item.source_row_index)


def _extend_schema(schema: pa.Schema) -> pa.Schema:
    index = schema.get_field_index("extra_info")
    if index < 0 or not pa.types.is_struct(schema.field(index).type):
        raise ValueError("canonical schema has no extra_info struct")
    extra = schema.field(index)
    if extra.type.get_field_index("augmentation") >= 0:
        raise ValueError("canonical input already has augmentation metadata")
    fields = list(schema)
    fields[index] = pa.field(
        "extra_info",
        pa.struct([*extra.type, pa.field("augmentation", AUGMENTATION_METADATA_TYPE)]),
        nullable=extra.nullable,
        metadata=extra.metadata,
    )
    return pa.schema(fields, metadata=schema.metadata)


def _transform_dtype(code: str, entry_point: str, target: str, expected: int) -> tuple[str, int, int]:
    analysis = analyze_code(code, entry_point)
    parent_sections = _section_hashes(analysis.tree, entry_point)
    child_code, child_analysis, metadata = transform_code(
        code,
        analysis,
        Intervention(None, None, dtype_target=target),
        DTYPE_POLICY,
        entry_point=entry_point,
    )
    if metadata.get("dtype_replacements") != expected:
        raise ValueError(f"dtype_replacement_count_mismatch:{metadata.get('dtype_replacements')}:{expected}")
    if _section_hashes(ast.parse(child_code), entry_point) != parent_sections:
        raise ValueError("model_or_get_init_inputs_changed")
    if tuple(child_analysis.floating_input_dtypes) not in tuple((name,) for name in _LOW_PRECISION_NAMES[target]):
        raise ValueError(f"child_input_dtype_not_exact_target:{child_analysis.floating_input_dtypes}")
    return child_code, analysis.input_bytes, child_analysis.input_bytes


def _make_child(
    parent: Mapping[str, Any],
    eligible: EligibleParent,
    *,
    source_path: Path,
    source_sha256: str,
    generator_sha256: str,
    dependency_sha256: str,
    git_commit: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    child = copy.deepcopy(dict(parent))
    parent_code = _nested(parent, "reward_model.ground_truth")
    entry_point = _nested(parent, "extra_info.entry_point", "Model")
    if not isinstance(parent_code, str) or not isinstance(entry_point, str):
        raise ValueError("invalid_parent_code_or_entry_point")
    child_code, input_bytes_before, input_bytes_after = _transform_dtype(
        parent_code, entry_point, eligible.assigned_dtype, eligible.transformed_factory_count
    )
    if input_bytes_before != eligible.input_bytes_before or input_bytes_after != eligible.input_bytes_after:
        raise ValueError("input_byte_accounting_changed_during_replay")
    child_reference_sha256 = _sha256_bytes(child_code.encode())
    child_ast_sha256 = _normalized_ast_sha256(child_code)
    intervention = {
        "primary_intervention": "dtype",
        "coherence_class": "parameter_free_all_direct_float_inputs",
        "dtype_before": "float32",
        "dtype_after": eligible.assigned_dtype,
        "factory_count": eligible.transformed_factory_count,
        "model_parameter_policy": "runtime_must_be_empty",
        "model_buffer_policy": "runtime_must_be_empty",
        "explicit_cast_policy": "none_in_model",
        "fp32_fallback_policy": "runtime_trace_reject",
    }
    intervention_sha256 = _canonical_sha256(intervention)
    child_uuid = (
        "dtype_"
        + _sha256_bytes(
            f"{CONTRACT_VERSION}:{eligible.parent_uuid}:{eligible.parent_reference_sha256}:{intervention_sha256}".encode()
        )[:24]
    )
    child["reward_model"] = dict(child["reward_model"])
    child["reward_model"]["ground_truth"] = child_code
    child["prompt"] = _replace_reference(child.get("prompt"), parent_code, child_code, required=True)
    extra = dict(child["extra_info"])
    if extra.get("original_prompt") is not None:
        extra["original_prompt"] = _replace_reference(
            extra.get("original_prompt"), parent_code, child_code, required=False
        )
    extra["uuid"] = child_uuid
    v4 = dict(extra.get("v4") or {})
    v4.update(
        {
            "parent_uuid": eligible.parent_uuid,
            "reference_sha256": child_reference_sha256,
            "normalized_ast_sha256": child_ast_sha256,
            "included_in_review_train": False,
            "runtime_validation_status": "dtype_parameter_free_paired_runtime_pending",
            "governance_status": "dtype_intervention_review_only",
        }
    )
    extra["v4"] = v4
    extra["augmentation"] = {
        "contract_version": CONTRACT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "parent_uuid": eligible.parent_uuid,
        "child_uuid": child_uuid,
        "source_artifact_sha256": source_sha256,
        "source_row_index": eligible.source_row_index,
        "parent_reference_sha256": eligible.parent_reference_sha256,
        "parent_normalized_ast_sha256": eligible.parent_normalized_ast_sha256,
        "child_reference_sha256": child_reference_sha256,
        "child_normalized_ast_sha256": child_ast_sha256,
        "intervention_id": f"dtype_{intervention_sha256[:20]}",
        "intervention_sha256": intervention_sha256,
        "intervention_kind": "dtype",
        "coverage_cell": f"dtype_parameter_free_{eligible.assigned_dtype}",
        "shape_scale": None,
        "value_family_before": None,
        "value_family_after": None,
        "dtype_before": "float32",
        "dtype_after": eligible.assigned_dtype,
        "layout_before": None,
        "layout_after": None,
        "shape_dimension_name": None,
        "shape_dimension_before": None,
        "shape_dimension_after": None,
        "factory_count": eligible.factory_count,
        "input_bytes_before": input_bytes_before,
        "input_bytes_after": input_bytes_after,
        "estimated_peak_bytes": math.ceil(input_bytes_after * DTYPE_POLICY.working_set_multiplier),
        "memory_budget_bytes": DTYPE_POLICY.memory_budget_bytes,
        "working_set_multiplier": DTYPE_POLICY.working_set_multiplier,
        "memory_estimator_version": "dtype_input_storage_exact_runtime_peak_guard_v1",
        "shape_proof": None,
        "compatibility_proof": (
            "all_unique_returned_float_factories_are_fp32;model_and_get_init_inputs_unchanged;"
            "no_static_parameter_buffer_tensor_state_or_model_dtype_cast"
        ),
        "validation_status": "static_parameter_free_dtype_pass_paired_runtime_required",
    }
    child["extra_info"] = extra
    fatal, _ = inspect_row_schema(child, child_code)
    if fatal:
        raise ValueError(f"child_schema_failed:{','.join(fatal)}")
    manifest = {
        "manifest_contract_version": "dtype_lane_manifest_v1",
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
        "primary_intervention": "dtype",
        "assigned_target": eligible.assigned_dtype,
        "realized_intervention": intervention,
        "reject_reason": None,
        "assignment_version": ASSIGNMENT_VERSION,
        "assignment_sha256": eligible.assignment_sha256,
        "selection_version": SELECTION_VERSION,
        "selection_sha256": eligible.selection_sha256,
        "generator_contract_version": CONTRACT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "generator_source_sha256": generator_sha256,
        "dependency_source_sha256": dependency_sha256,
        "git_commit": git_commit,
        "source_family": eligible.source_family,
        "operator_bucket": eligible.operator_bucket,
        "operator_count": eligible.operator_count,
        "mode_class": _nested(parent, "extra_info.v4.mode_class"),
        "shape_changed": False,
        "value_changed": False,
        "dtype_changed": True,
        "layout_changed": False,
        "model_changed": False,
        "get_init_inputs_changed": False,
        "rng_consumption_status": "runtime_pending",
        "factory_count": eligible.factory_count,
        "transformed_factory_count": eligible.transformed_factory_count,
        "input_bytes_before": input_bytes_before,
        "input_bytes_after": input_bytes_after,
        "estimated_peak_bytes": math.ceil(input_bytes_after * DTYPE_POLICY.working_set_multiplier),
        "static_memory_budget_bytes": DTYPE_POLICY.memory_budget_bytes,
        "runtime_memory_budget_bytes": RUNTIME_MEMORY_BUDGET_BYTES,
        "factory_specs_before": [
            {"factory": name, "shape": list(shape), "dtype": dtype} for name, shape, dtype in eligible.factory_specs
        ],
        "model_parameter_count": 0,
        "model_buffer_count": 0,
        "explicit_model_casts": [],
        "runtime_promotion_status": "pending",
        "static_status": "passed",
        "parent_runtime_status": "pending",
        "child_runtime_status": "pending",
        "liveness_status": "pending",
        "materialization_status": "review_only",
        "runtime_policy_fingerprint": None,
        "provenance_status": _nested(parent, "extra_info.v4.provenance_status"),
        "licenses": _nested(parent, "extra_info.v4.licenses", []),
        "lineage": "canonical_parent_parameter_free_dtype_child",
        "row_sha256": _canonical_sha256(child),
        "training_approved": False,
    }
    return child, manifest


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _review_markdown(samples: Sequence[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]]) -> str:
    lines = ["# Parameter-free dtype solver review samples", ""]
    for parent, child, manifest in samples:
        before = str(_nested(parent, "reward_model.ground_truth", ""))
        after = str(_nested(child, "reward_model.ground_truth", ""))
        lines.extend(
            [
                f"## {manifest['child_uuid']}",
                "",
                (
                    f"- source: `{manifest['source_family']}`; operator: `{manifest['operator_bucket']}`; "
                    f"target: `{manifest['assigned_target']}`; parent row: `{manifest['source_row_index']}`"
                ),
                "",
                "```diff",
                *difflib.unified_diff(
                    before.splitlines(), after.splitlines(), fromfile="parent", tofile="child", lineterm=""
                ),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def build_lane(input_path: Path, output_dir: Path, *, limit: int, overwrite: bool) -> dict[str, Any]:
    if type(limit) is not int or not 1 <= limit <= MAX_AUTHORIZED_CANDIDATES:
        raise ValueError(f"limit must be an integer in [1, {MAX_AUTHORIZED_CANDIDATES}]")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    source_sha256 = _sha256_file(input_path)
    parquet = pq.ParquetFile(input_path)
    if source_sha256 != EXPECTED_CANONICAL_PARENT_SHA256:
        raise ValueError(f"canonical parent SHA mismatch:{source_sha256}")
    if parquet.metadata.num_rows != EXPECTED_CANONICAL_PARENT_ROWS:
        raise ValueError(f"canonical parent row mismatch:{parquet.metadata.num_rows}")
    paths = {
        "parents": output_dir / "parents.parquet",
        "candidates": output_dir / "candidates.parquet",
        "paired": output_dir / "paired.parquet",
        "manifest": output_dir / "manifest.jsonl",
        "decisions": output_dir / "decisions.jsonl",
        "summary": output_dir / "summary.json",
        "review": output_dir / "review_samples.md",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing existing dtype artifacts:{existing[:3]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    generator_path = Path(__file__).resolve()
    dependency_path = generator_path.parent.parent / "augment_prompt_tasks.py"
    generator_sha256 = _sha256_file(generator_path)
    dependency_sha256 = _sha256_file(dependency_path)
    git_commit = _git_commit()
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
                        "assigned_target": None,
                        "reason": reason,
                    }
                )
            else:
                eligible.append(item)
                decisions.append(
                    {
                        "source_row_index": row_index,
                        "parent_uuid": item.parent_uuid,
                        "eligible": True,
                        "assigned_target": item.assigned_dtype,
                        "assignment_sha256": item.assignment_sha256,
                        "reason": "static_parameter_free_dtype_eligible",
                    }
                )
            row_index += 1
    selected = _select_stratified(eligible, limit)
    selected_by_row = {item.source_row_index: item for item in selected}
    for decision in decisions:
        if decision["eligible"]:
            decision["selected"] = decision["source_row_index"] in selected_by_row
            if not decision["selected"]:
                decision["reason"] = "eligible_not_selected_by_limit"

    output_schema = _extend_schema(parquet.schema_arrow)
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
                    source_path=input_path,
                    source_sha256=source_sha256,
                    generator_sha256=generator_sha256,
                    dependency_sha256=dependency_sha256,
                    git_commit=git_commit,
                )
                parent = copy.deepcopy(row)
                parent["extra_info"] = dict(parent["extra_info"])
                parent["extra_info"]["augmentation"] = None
                manifest["candidate_row_index"] = len(children)
                parents.append(parent)
                children.append(child)
                manifests.append(manifest)
                review_pool.setdefault(item.stratum, (row, child, manifest))
            row_index += 1
    if len(children) != limit:
        raise ValueError(f"materialized {len(children)} dtype children, expected {limit}")
    for field in ("child_uuid", "child_reference_sha256", "child_normalized_ast_sha256"):
        values = [str(item[field]) for item in manifests]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate manifest field:{field}")

    paired = [row for pair in zip(parents, children, strict=True) for row in pair]
    suffix = f".tmp.{os.getpid()}"
    for key, rows in (("parents", parents), ("candidates", children), ("paired", paired)):
        temporary = paths[key].with_name(paths[key].name + suffix)
        pq.write_table(pa.Table.from_pylist(rows, schema=output_schema), temporary, compression="zstd")
        os.replace(temporary, paths[key])
    _atomic_text(paths["manifest"], "".join(_canonical_json(item) + "\n" for item in manifests))
    _atomic_text(paths["decisions"], "".join(_canonical_json(item) + "\n" for item in decisions))
    _atomic_text(paths["review"], _review_markdown([review_pool[key] for key in sorted(review_pool)[:24]]))

    def counts(items: Sequence[EligibleParent], attribute: str) -> dict[str, int]:
        return dict(sorted(collections.Counter(str(getattr(item, attribute)) for item in items).items()))

    summary: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "generator_version": GENERATOR_VERSION,
        "assignment_version": ASSIGNMENT_VERSION,
        "selection_version": SELECTION_VERSION,
        "input": str(input_path.resolve()),
        "input_sha256": source_sha256,
        "input_rows": parquet.metadata.num_rows,
        "generator_source_sha256": generator_sha256,
        "dependency_source_sha256": dependency_sha256,
        "git_commit": git_commit,
        "eligible_parents": len(eligible),
        "eligible_dtype_counts": counts(eligible, "assigned_dtype"),
        "eligible_source_counts": counts(eligible, "source_family"),
        "eligible_operator_bucket_counts": counts(eligible, "operator_bucket"),
        "selected_parents": len(selected),
        "candidate_rows": len(children),
        "paired_rows": len(paired),
        "selection_limit": limit,
        "dtype_counts": counts(selected, "assigned_dtype"),
        "source_counts": counts(selected, "source_family"),
        "operator_bucket_counts": counts(selected, "operator_bucket"),
        "skip_reason_counts": dict(sorted(skip_counts.items())),
        "proof": {
            "primary_intervention": "dtype",
            "coherence_class": "parameter_free_all_direct_float_inputs",
            "one_dtype_per_parent": True,
            "shape_changed": False,
            "value_family_changed": False,
            "layout_changed": False,
            "model_changed": False,
            "get_init_inputs_changed": False,
            "runtime_boundary": (
                "paired parent/child reference, exact target realization, no-FP32 dispatch trace, "
                "and cast-equivalent output comparison required"
            ),
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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    print(json.dumps(build_lane(args.input, args.output_dir, limit=args.limit, overwrite=args.overwrite), indent=2))


if __name__ == "__main__":
    main()
