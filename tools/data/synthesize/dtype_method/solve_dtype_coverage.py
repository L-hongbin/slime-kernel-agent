#!/usr/bin/env python3
"""Build deterministic low-precision siblings for proof-friendly dtype tasks.

Every floating direct input factory is FP32 and has exact returned-input
provenance.  The ``parameter_free`` class leaves ``Model`` unchanged.  The
``module_state`` class admits only canonical built-in ``torch.nn`` registered
state and appends one explicit base-class conversion to ``Model.__init__``.
One target (``float16`` or ``bfloat16``) is assigned per parent.

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
from tools.data.synthesize import intervention_semantic_gates as semantic_gates
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
from tools.data.synthesize.serial_random_wrappers import logical_factory_view, recognize_random_wrapper
from tools.data.synthesize.serial_source_contract import canonical_sha256 as _serial_canonical_sha256
from tools.data.synthesize.serial_source_contract import verify_serial_source

PARAMETER_FREE = "parameter_free"
MODULE_STATE = "module_state"
COHERENCE_CLASSES = (PARAMETER_FREE, MODULE_STATE)
CONTRACT_VERSIONS = {
    PARAMETER_FREE: "dtype_parameter_free_solver_v2",
    MODULE_STATE: "dtype_module_state_solver_v1",
}
SERIAL_CONTRACT_VERSIONS = {
    PARAMETER_FREE: "dtype_parameter_free_serial_random_solver_v2",
    MODULE_STATE: "dtype_module_state_serial_random_solver_v2",
}
GENERATOR_VERSIONS = {
    PARAMETER_FREE: "all_direct_float_inputs_low_precision_v2",
    MODULE_STATE: "registered_module_state_low_precision_v1",
}
SERIAL_GENERATOR_VERSIONS = {
    PARAMETER_FREE: "frozen_random_wrapper_source_span_low_precision_v2",
    MODULE_STATE: "frozen_random_wrapper_and_registered_state_source_span_low_precision_v2",
}
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
_STATEFUL_NN_CONSTRUCTORS = frozenset(
    {
        "BatchNorm1d",
        "BatchNorm2d",
        "BatchNorm3d",
        "Bilinear",
        "Conv1d",
        "Conv2d",
        "Conv3d",
        "ConvTranspose1d",
        "ConvTranspose2d",
        "ConvTranspose3d",
        "Embedding",
        "EmbeddingBag",
        "GRU",
        "GRUCell",
        "GroupNorm",
        "LayerNorm",
        "Linear",
        "LSTM",
        "LSTMCell",
        "MultiheadAttention",
        "PReLU",
        "RMSNorm",
        "RNN",
        "RNNCell",
        "Transformer",
        "TransformerDecoder",
        "TransformerDecoderLayer",
        "TransformerEncoder",
        "TransformerEncoderLayer",
    }
)
DTYPE_POLICY = AugmentationPolicy(
    shape_scales=(),
    value_families=(),
    dtype_targets=DTYPE_TARGETS,
    layout_targets=(),
    include_joint_cells=False,
)


@dataclasses.dataclass(frozen=True)
class EligibleParent:
    coherence_class: str
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
    serial_random_wrapper_count: int

    @property
    def stratum(self) -> tuple[str, str, str]:
        return (self.source_family, self.operator_bucket, self.assigned_dtype)


def contract_version(coherence_class: str, serial_random: bool = False) -> str:
    try:
        versions = SERIAL_CONTRACT_VERSIONS if serial_random else CONTRACT_VERSIONS
        return versions[coherence_class]
    except KeyError as exc:
        raise ValueError(f"unsupported coherence class:{coherence_class}") from exc


def generator_version(coherence_class: str, serial_random: bool = False) -> str:
    try:
        versions = SERIAL_GENERATOR_VERSIONS if serial_random else GENERATOR_VERSIONS
        return versions[coherence_class]
    except KeyError as exc:
        raise ValueError(f"unsupported coherence class:{coherence_class}") from exc


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


def _canonical_torch_imports(tree: ast.Module) -> bool:
    has_torch = False
    has_nn = False
    for statement in tree.body:
        if isinstance(statement, ast.Import):
            for alias in statement.names:
                bound = alias.asname or alias.name.split(".", 1)[0]
                canonical_torch = alias.name == "torch" and alias.asname in {None, "torch"}
                canonical_nn = alias.name == "torch.nn" and alias.asname == "nn"
                if bound in {"torch", "nn"} and not (canonical_torch or canonical_nn):
                    return False
                has_torch |= canonical_torch
                has_nn |= canonical_nn
        elif isinstance(statement, ast.ImportFrom):
            for alias in statement.names:
                if (alias.asname or alias.name) in {"torch", "nn"}:
                    return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store) and node.id in {"torch", "nn"}:
            return False
        if isinstance(node, ast.arg) and node.arg in {"torch", "nn"}:
            return False
        if isinstance(node, ast.Attribute) and isinstance(node.ctx, (ast.Store, ast.Del)):
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in {"torch", "nn"}:
                return False
    return has_torch and has_nn


def _is_registration_wrapper(call: ast.Call) -> bool:
    name = _call_name(call.func)
    return name in {"nn.Parameter", "torch.nn.Parameter"} or name.endswith(".register_buffer")


def _module_state_model_proof(tree: ast.Module, entry_point: str) -> tuple[bool, str | None]:
    """Admit only state whose dtype conversion has a closed runtime proof."""

    if _has_default_dtype_mutation(tree):
        return False, "dynamic_default_dtype"
    if not _canonical_torch_imports(tree):
        return False, "noncanonical_torch_imports"
    models = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == entry_point]
    if len(models) != 1:
        return False, "model_class_count_not_one"
    model = models[0]
    if model.decorator_list or model.keywords:
        return False, "decorated_or_dynamic_model_class"
    if len(model.bases) != 1 or _call_name(model.bases[0]) not in {"nn.Module", "torch.nn.Module"}:
        return False, "model_base_not_canonical_nn_module"
    forbidden_methods = {"to", "_apply", "__getattr__", "__getattribute__", "__setattr__"}
    defined_methods = {node.name for node in model.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    conflict = sorted(forbidden_methods & defined_methods)
    if conflict:
        return False, f"model_overrides_conversion_dispatch:{conflict[0]}"
    init_methods = [
        node
        for node in model.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__"
    ]
    if len(init_methods) != 1 or not isinstance(init_methods[0], ast.FunctionDef):
        return False, "model_init_not_one_synchronous_function"
    init = init_methods[0]
    if init.decorator_list:
        return False, "decorated_model_init"
    if any(isinstance(node, (ast.Return, ast.Yield, ast.YieldFrom, ast.Await)) for node in ast.walk(init)):
        return False, "model_init_has_early_exit_or_async_control"
    super_calls = [node for node in ast.walk(init) if isinstance(node, ast.Call) and _is_super_init(node)]
    if len(super_calls) != 1 or super_calls[0].args or super_calls[0].keywords:
        return False, "model_init_not_one_canonical_super_init"
    for node in ast.walk(init):
        if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr in forbidden_methods
            ):
                return False, f"model_assigns_conversion_dispatch:{target.attr}"

    get_inputs = _top_level_function(tree, "get_inputs")
    get_init_inputs = _top_level_function(tree, "get_init_inputs")
    for node in ast.walk(get_init_inputs):
        if isinstance(node, ast.Call):
            return False, f"get_init_inputs_call_not_scalar_proven:{_call_name(node.func) or 'dynamic'}"
        if isinstance(node, ast.Attribute) and _call_name(node) in _MODEL_DTYPE_NAMES:
            return False, "get_init_inputs_contains_dtype_literal"

    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(model):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent

    def inside_registration(node: ast.AST) -> bool:
        current = node
        while id(current) in parents:
            current = parents[id(current)]
            if isinstance(current, ast.Call) and _is_registration_wrapper(current):
                return True
            if current is init:
                break
        return False

    state_evidence = 0
    init_ids = {id(node) for node in ast.walk(init)}
    for node in ast.walk(model):
        if isinstance(node, ast.Attribute) and _call_name(node) in _MODEL_DTYPE_NAMES:
            return False, f"model_dtype_literal:{_call_name(node)}"
        if not isinstance(node, ast.Call):
            continue
        name = _call_name(node.func)
        if any(keyword.arg == "dtype" for keyword in node.keywords):
            return False, "model_explicit_dtype_keyword"
        if isinstance(node.func, ast.Attribute) and node.func.attr in _CAST_METHODS:
            return False, f"model_dtype_cast_or_to:{node.func.attr}"
        if name in _MODEL_TENSOR_FACTORIES:
            if id(node) not in init_ids or not inside_registration(node):
                return False, f"unregistered_or_runtime_tensor_factory:{name}"
            continue
        if id(node) not in init_ids:
            continue
        if _is_super_init(node) or name in _SAFE_INIT_CALLS:
            continue
        if name in {"nn.Parameter", "torch.nn.Parameter"}:
            state_evidence += 1
            continue
        if name.endswith(".register_buffer"):
            state_evidence += 1
            continue
        if name.startswith("nn.") or name.startswith("torch.nn."):
            constructor = name.rsplit(".", 1)[-1]
            if constructor and constructor[0].isupper() and constructor != "Module":
                state_evidence += constructor in _STATEFUL_NN_CONSTRUCTORS
                continue
        return False, f"model_init_call_not_builtin_registered_state:{name or 'dynamic'}"
    if not state_evidence:
        return False, "no_static_registered_state_evidence"

    excluded_ids = {id(node) for root in (model, get_inputs, get_init_inputs) for node in ast.walk(root)}
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


def _assignment(parent_uuid: str, reference_sha256: str, coherence_class: str) -> tuple[str, str]:
    payload = {
        "assignment_version": ASSIGNMENT_VERSION,
        "contract_version": contract_version(coherence_class),
        "coherence_class": coherence_class,
        "parent_uuid": parent_uuid,
        "parent_reference_sha256": reference_sha256,
    }
    digest = _canonical_sha256(payload)
    return DTYPE_TARGETS[int(digest[:16], 16) % len(DTYPE_TARGETS)], digest


def _selection_hash(parent_uuid: str, reference_sha256: str, target: str, coherence_class: str) -> str:
    return _canonical_sha256(
        {
            "selection_version": SELECTION_VERSION,
            "parent_uuid": parent_uuid,
            "parent_reference_sha256": reference_sha256,
            "assigned_dtype": target,
            "coherence_class": coherence_class,
        }
    )


def _analyze_parent(
    row: Mapping[str, Any], source_row_index: int, coherence_class: str = PARAMETER_FREE
) -> tuple[EligibleParent | None, str]:
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
        logical_view = logical_factory_view(ast.parse(code))
        if logical_view.random_wrapper_count and len(logical_view.original_roots) != logical_view.random_wrapper_count:
            return None, "mixed_random_wrappers_and_direct_factories"
        analysis_code = ast.unparse(logical_view.tree) + "\n" if logical_view.random_wrapper_count else code
        analysis = analyze_code(analysis_code, entry_point)
    except (SyntaxError, ValueError) as exc:
        return None, f"parent_static_analysis_failed:{type(exc).__name__}:{exc}"
    if not analysis.dtype_transform_eligible:
        return None, analysis.dtype_skip_reason or "dtype_transform_not_proven"
    if not analysis.floating_input_dtypes or any(name not in _FP32_NAMES for name in analysis.floating_input_dtypes):
        return None, "floating_direct_inputs_not_all_fp32"
    proof = _parameter_free_model_proof if coherence_class == PARAMETER_FREE else _module_state_model_proof
    compatible, reason = proof(analysis.tree, entry_point)
    if not compatible:
        return None, reason or f"{coherence_class}_model_not_proven"
    try:
        specs = _resolved_factory_specs(analysis.tree)
    except ValueError as exc:
        return None, f"returned_input_proof_failed:{exc}"
    transformed = tuple(spec for spec in specs if spec[2] in _FP32_NAMES)
    if len(transformed) != analysis.dtype_factory_count:
        return None, "transformed_factory_count_mismatch"
    if not transformed:
        return None, "no_fp32_direct_input_factory"
    target, assignment_sha256 = _assignment(parent_uuid, reference_sha256, coherence_class)
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
            coherence_class=coherence_class,
            source_row_index=source_row_index,
            parent_uuid=parent_uuid,
            parent_reference_sha256=reference_sha256,
            parent_normalized_ast_sha256=_normalized_ast_sha256(code),
            assigned_dtype=target,
            assignment_sha256=assignment_sha256,
            selection_sha256=_selection_hash(parent_uuid, reference_sha256, target, coherence_class),
            source_family=source_family,
            operator_bucket=operator_bucket,
            operator_count=operator_count,
            factory_count=analysis.factory_count,
            transformed_factory_count=len(transformed),
            input_bytes_before=analysis.input_bytes,
            input_bytes_after=input_bytes_after,
            factory_specs=specs,
            serial_random_wrapper_count=logical_view.random_wrapper_count,
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
    augmentation_index = extra.type.get_field_index("augmentation")
    if augmentation_index >= 0:
        if extra.type.field(augmentation_index).type != AUGMENTATION_METADATA_TYPE:
            raise ValueError("input augmentation metadata has an incompatible type")
        return schema
    fields = list(schema)
    fields[index] = pa.field(
        "extra_info",
        pa.struct([*extra.type, pa.field("augmentation", AUGMENTATION_METADATA_TYPE)]),
        nullable=extra.nullable,
        metadata=extra.metadata,
    )
    return pa.schema(fields, metadata=schema.metadata)


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


def _inject_module_conversion(code: str, entry_point: str, target: str) -> str:
    shared_tree = ast.parse(code)
    model = _top_level_class(shared_tree, entry_point)
    init = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    last = init.body[-1]
    lines = code.encode("utf-8").splitlines(keepends=True)
    insertion = sum(len(line) for line in lines[: last.end_lineno])
    body_line = lines[init.body[0].lineno - 1]
    indentation = body_line[: len(body_line) - len(body_line.lstrip())]
    statement_text = indentation + f"torch.nn.Module.to(self, dtype=torch.{target})\n".encode()
    raw = code.encode("utf-8")
    child_code = (raw[:insertion] + statement_text + raw[insertion:]).decode("utf-8")
    child_tree = ast.parse(child_code)
    statement = ast.parse(f"torch.nn.Module.to(self, dtype=torch.{target})").body[0]
    replay_tree = copy.deepcopy(child_tree)
    replay_model = _top_level_class(replay_tree, entry_point)
    replay_init = next(
        node for node in replay_model.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    if ast.dump(replay_init.body[-1], include_attributes=False) != ast.dump(statement, include_attributes=False):
        raise ValueError("module_conversion_statement_not_exact")
    replay_init.body.pop()
    if ast.dump(replay_tree, include_attributes=False) != ast.dump(shared_tree, include_attributes=False):
        raise ValueError("module_conversion_changed_unapproved_ast")
    return child_code


def _transform_serial_random_dtype(
    code: str, entry_point: str, target: str, expected: int, coherence_class: str
) -> tuple[str, int, int]:
    tree = ast.parse(code)
    view = logical_factory_view(tree)
    if view.random_wrapper_count != expected or expected <= 0:
        raise ValueError(f"serial_random_wrapper_count_mismatch:{view.random_wrapper_count}:{expected}")
    parent_analysis = analyze_code(ast.unparse(view.tree) + "\n", entry_point)
    parent_sections = _section_hashes(tree, entry_point)
    get_inputs = _top_level_function(tree, "get_inputs")
    source_replacements: list[tuple[int, int, bytes]] = []
    for node in ast.walk(get_inputs):
        if not isinstance(node, ast.Call):
            continue
        wrapper = recognize_random_wrapper(node)
        if wrapper is None:
            continue
        base = wrapper.base
        dtype_keywords = [keyword for keyword in base.keywords if keyword.arg == "dtype"]
        if len(dtype_keywords) > 1:
            raise ValueError("serial_random_base_has_duplicate_dtype_keyword")
        if dtype_keywords:
            if _call_name(dtype_keywords[0].value) not in _FP32_NAMES:
                raise ValueError("serial_random_base_dtype_is_not_fp32")
            start, end = _byte_span(code, dtype_keywords[0].value)
            source_replacements.append((start, end, f"torch.{target}".encode()))
        else:
            start, end = _byte_span(code, base)
            source = code.encode("utf-8")[start:end]
            if not source.endswith(b")"):
                raise ValueError("serial_random_base_source_does_not_end_in_parenthesis")
            prefix = source[:-1]
            separator = b" " if prefix.rstrip().endswith(b",") else b", "
            source_replacements.append((start, end, prefix + separator + f"dtype=torch.{target})".encode()))
    if len(source_replacements) != expected:
        raise ValueError(f"serial_random_dtype_replacement_count_mismatch:{len(source_replacements)}:{expected}")
    child_code = _replace_byte_spans(code, source_replacements)
    child_view = logical_factory_view(ast.parse(child_code))
    if child_view.random_wrapper_count != expected:
        raise ValueError("serial_random_wrapper_contract_changed_after_dtype_transform")
    child_analysis = analyze_code(ast.unparse(child_view.tree) + "\n", entry_point)
    if _section_hashes(ast.parse(child_code), entry_point) != parent_sections:
        raise ValueError("model_or_get_init_inputs_changed")
    if tuple(child_analysis.floating_input_dtypes) not in tuple((name,) for name in _LOW_PRECISION_NAMES[target]):
        raise ValueError(f"child_input_dtype_not_exact_target:{child_analysis.floating_input_dtypes}")
    if coherence_class == MODULE_STATE:
        child_code = _inject_module_conversion(child_code, entry_point, target)
    elif coherence_class != PARAMETER_FREE:
        raise ValueError(f"unsupported coherence class:{coherence_class}")
    return child_code, parent_analysis.input_bytes, child_analysis.input_bytes


def _transform_dtype(
    code: str, entry_point: str, target: str, expected: int, coherence_class: str
) -> tuple[str, int, int]:
    view = logical_factory_view(ast.parse(code))
    if view.random_wrapper_count:
        return _transform_serial_random_dtype(
            code,
            entry_point,
            target,
            view.random_wrapper_count,
            coherence_class,
        )
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
    if coherence_class == MODULE_STATE:
        child_code = _inject_module_conversion(child_code, entry_point, target)
    elif coherence_class != PARAMETER_FREE:
        raise ValueError(f"unsupported coherence class:{coherence_class}")
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
    source_binding: Mapping[str, Any] | None = None,
    source_row_binding: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    child = copy.deepcopy(dict(parent))
    coherence_class = eligible.coherence_class
    serial_mode = source_binding is not None
    solver_contract = contract_version(coherence_class, serial_random=serial_mode)
    solver_generator = generator_version(coherence_class, serial_random=serial_mode)
    if serial_mode:
        if source_row_binding is None:
            raise ValueError("serial dtype child lacks its upstream manifest row")
        if source_row_binding.get("row_index") != eligible.source_row_index:
            raise ValueError("serial dtype upstream row index mismatch")
        upstream_manifest_sha256 = _serial_canonical_sha256(source_row_binding)
        lineage = f"{source_binding.get('stage')}_dtype_child"
    else:
        if source_row_binding is not None:
            raise ValueError("canonical dtype child unexpectedly has an upstream manifest row")
        upstream_manifest_sha256 = None
        lineage = f"canonical_parent_{coherence_class}_dtype_child"
    parent_code = _nested(parent, "reward_model.ground_truth")
    entry_point = _nested(parent, "extra_info.entry_point", "Model")
    if not isinstance(parent_code, str) or not isinstance(entry_point, str):
        raise ValueError("invalid_parent_code_or_entry_point")
    child_code, input_bytes_before, input_bytes_after = _transform_dtype(
        parent_code,
        entry_point,
        eligible.assigned_dtype,
        eligible.transformed_factory_count,
        coherence_class,
    )
    if input_bytes_before != eligible.input_bytes_before or input_bytes_after != eligible.input_bytes_after:
        raise ValueError("input_byte_accounting_changed_during_replay")
    child_reference_sha256 = _sha256_bytes(child_code.encode())
    child_ast_sha256 = _normalized_ast_sha256(child_code)
    intervention = {
        "primary_intervention": "dtype",
        "coherence_class": coherence_class,
        "dtype_before": "float32",
        "dtype_after": eligible.assigned_dtype,
        "factory_count": eligible.transformed_factory_count,
        "model_parameter_policy": (
            "runtime_must_be_empty" if coherence_class == PARAMETER_FREE else "registered_fp32_parent_exact_cast"
        ),
        "model_buffer_policy": (
            "runtime_must_be_empty" if coherence_class == PARAMETER_FREE else "registered_state_exact_cast_or_equal"
        ),
        "explicit_cast_policy": "none_in_model",
        "fp32_fallback_policy": "runtime_trace_reject",
    }
    intervention_sha256 = _canonical_sha256(intervention)
    child_uuid = (
        "dtype_"
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
            "runtime_validation_status": f"dtype_{coherence_class}_paired_runtime_pending",
            "governance_status": "dtype_intervention_review_only",
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
        "intervention_id": f"dtype_{intervention_sha256[:20]}",
        "intervention_sha256": intervention_sha256,
        "intervention_kind": "dtype",
        "coverage_cell": f"dtype_{coherence_class}_{eligible.assigned_dtype}",
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
            if coherence_class == PARAMETER_FREE
            else "all_unique_returned_float_factories_are_fp32;get_init_inputs_unchanged;"
            "canonical_builtin_registered_state;single_explicit_base_module_dtype_conversion"
        ),
        "validation_status": f"static_{coherence_class}_dtype_pass_paired_runtime_required",
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
        "coherence_class": coherence_class,
        "reject_reason": None,
        "assignment_version": ASSIGNMENT_VERSION,
        "assignment_sha256": eligible.assignment_sha256,
        "selection_version": SELECTION_VERSION,
        "selection_sha256": eligible.selection_sha256,
        "generator_contract_version": solver_contract,
        "generator_version": solver_generator,
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
        "model_changed": coherence_class == MODULE_STATE,
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
        "model_parameter_count": 0 if coherence_class == PARAMETER_FREE else None,
        "model_buffer_count": 0 if coherence_class == PARAMETER_FREE else None,
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
        "lineage": lineage,
        "row_sha256": _canonical_sha256(child),
        "training_approved": False,
    }
    if source_binding is not None:
        manifest["source_binding"] = dict(source_binding)
        manifest["upstream_serial_manifest_row_sha256"] = upstream_manifest_sha256
    return child, manifest


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _review_markdown(samples: Sequence[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]]) -> str:
    coherence_class = samples[0][2]["coherence_class"] if samples else "unknown"
    lines = [f"# {coherence_class} dtype solver review samples", ""]
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


def build_lane(
    input_path: Path,
    output_dir: Path,
    *,
    limit: int,
    overwrite: bool,
    coherence_class: str = PARAMETER_FREE,
    serial_source_manifest: Path | None = None,
    excluded_source_rows: frozenset[int] = frozenset(),
) -> dict[str, Any]:
    if type(limit) is not int or not 1 <= limit <= MAX_AUTHORIZED_CANDIDATES:
        raise ValueError(f"limit must be an integer in [1, {MAX_AUTHORIZED_CANDIDATES}]")
    serial_mode = serial_source_manifest is not None
    solver_contract = contract_version(coherence_class, serial_random=serial_mode)
    solver_generator = generator_version(coherence_class, serial_random=serial_mode)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    source_sha256 = _sha256_file(input_path)
    parquet = pq.ParquetFile(input_path)
    if serial_mode:
        assert serial_source_manifest is not None
        source_binding, source_row_bindings = verify_serial_source(
            input_path, serial_source_manifest, expected_stage="random_fallback_base"
        )
    else:
        source_binding = None
        source_row_bindings = None
        if source_sha256 != EXPECTED_CANONICAL_PARENT_SHA256:
            raise ValueError(f"canonical parent SHA mismatch:{source_sha256}")
        if parquet.metadata.num_rows != EXPECTED_CANONICAL_PARENT_ROWS:
            raise ValueError(f"canonical parent row mismatch:{parquet.metadata.num_rows}")
    if any(type(index) is not int or not 0 <= index < parquet.metadata.num_rows for index in excluded_source_rows):
        raise ValueError("excluded source row index is out of bounds")
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
            item, reason = _analyze_parent(row, row_index, coherence_class)
            if item is not None:
                parent_code = _nested(row, "reward_model.ground_truth")
                if not isinstance(parent_code, str):
                    raise ValueError(f"eligible dtype parent lacks reference:{row_index}")
                semantic_verdict = semantic_gates.evaluate_semantic_gate(
                    "dtype",
                    parent_code=parent_code,
                    child_code=parent_code,
                    manifest={
                        "assigned_target": item.assigned_dtype,
                        "factory_specs_before": [
                            {"factory": name, "shape": list(shape), "dtype": dtype}
                            for name, shape, dtype in item.factory_specs
                        ],
                    },
                )
                if semantic_verdict["status"] == "rejected":
                    reason = "semantic_gate:" + ",".join(semantic_verdict["reasons"])
                    item = None
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
            elif row_index in excluded_source_rows:
                decisions.append(
                    {
                        "source_row_index": row_index,
                        "parent_uuid": item.parent_uuid,
                        "eligible": True,
                        "selected": False,
                        "assigned_target": item.assigned_dtype,
                        "assignment_sha256": item.assignment_sha256,
                        "reason": "excluded_by_prior_serial_dtype_batch",
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
                        "reason": f"static_{coherence_class}_dtype_eligible",
                    }
                )
            row_index += 1
    selected = _select_stratified(eligible, limit)
    selected_by_row = {item.source_row_index: item for item in selected}
    for decision in decisions:
        if decision["eligible"]:
            if decision["reason"] == "excluded_by_prior_serial_dtype_batch":
                continue
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
        "contract_version": solver_contract,
        "generator_version": solver_generator,
        "coherence_class": coherence_class,
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
            "coherence_class": coherence_class,
            "one_dtype_per_parent": True,
            "shape_changed": False,
            "value_family_changed": False,
            "layout_changed": False,
            "model_changed": coherence_class == MODULE_STATE,
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
    parser.add_argument("--coherence-class", choices=COHERENCE_CLASSES, default=PARAMETER_FREE)
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
                coherence_class=args.coherence_class,
                serial_source_manifest=args.serial_source_manifest,
                excluded_source_rows=excluded_source_rows,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
