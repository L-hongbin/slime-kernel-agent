#!/usr/bin/env python3
"""Build a deterministic, value-only random-family intervention lane.

This solver intentionally handles a narrow proof domain.  A parent is eligible
only when the legacy input analysis proves that every input allocation is
uniquely returned, every continuous random factory is a bare ``torch.randn``,
and the factory dtype is a supported real floating dtype.  One target family is
assigned per parent by a stable hash:

* ``uniform_01``: Normal-CDF map into ``[0, 1)``;
* ``signed_uniform``: Normal-CDF map into ``[-1, 1)``;
* ``poisson_counts``: bounded positive rates sampled with a local generator; or
* ``multinomial_categories``: last-axis categorical samples represented in the
  original floating dtype and sampled with a local generator.

All mappings consume the original ``randn`` draw exactly once.  Stochastic
mappings use a per-factory local CUDA generator, preserving the global RNG
position so later unrelated random inputs cannot change as an accidental joint
intervention.  Shape, dtype, layout, ``Model``, and ``get_init_inputs`` are
checked structurally. Runtime parent/child correctness and value-liveness
remain separate gates.
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
    _SHAPE_FACTORIES,
    AUGMENTATION_METADATA_TYPE,
    _call_name,
    _factory_records,
    _module_constant_environment,
    _normalized_ast_sha256,
    _replace_reference,
    _section_hashes,
    _sha256_bytes,
    _sha256_file,
    analyze_code,
)

CONTRACT_VERSION = "random_value_only_solver_v2"
GENERATOR_VERSION = "isolated_distribution_mapping_v2"
ASSIGNMENT_VERSION = "stable_parent_hash_single_family_v2"
SELECTION_VERSION = "source_operator_family_proportional_largest_remainder_v1"
EXPECTED_CANONICAL_PARENT_SHA256 = "b07205fcadc543964cfc7ee5fd9c1e4d011f0f3481447f656e5297e40b4b99f4"
EXPECTED_CANONICAL_PARENT_ROWS = 64_315
VALUE_FAMILIES = ("uniform_01", "signed_uniform", "poisson_counts", "multinomial_categories")
LOCAL_GENERATOR_MAX_SEED = 2**63 - 1
MAX_AUTHORIZED_CANDIDATES = 5_000
REAL_FLOAT_DTYPES = frozenset(
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
MAX_CONSECUTIVE_INTEGER_BY_DTYPE = {
    "torch.bfloat16": 2**8,
    "torch.float16": 2**11,
    "torch.half": 2**11,
    "torch.float32": 2**24,
    "torch.float": 2**24,
    "torch.float64": 2**53,
    "torch.double": 2**53,
}
DEFAULT_MEMORY_BUDGET_BYTES = 64 * 1024**3
_DEFAULT_DTYPE_MUTATORS = frozenset({"set_default_dtype", "set_default_tensor_type"})
_REPO_ROOT = Path(__file__).resolve().parents[4]


@dataclasses.dataclass(frozen=True)
class EligibleParent:
    source_row_index: int
    parent_uuid: str
    parent_reference_sha256: str
    parent_normalized_ast_sha256: str
    assigned_family: str
    assignment_sha256: str
    selection_sha256: str
    source_family: str
    operator_bucket: str
    operator_count: int | None
    factory_count: int
    continuous_factory_count: int
    continuous_shapes: tuple[tuple[int, ...], ...]
    input_bytes: int

    @property
    def stratum(self) -> tuple[str, str, str]:
        return (self.source_family, self.operator_bucket, self.assigned_family)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_json_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _git_commit() -> str:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError("cannot bind random/value generation to the current Git commit") from exc
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError(f"git rev-parse returned an invalid commit: {commit!r}")
    return commit


def _git_blob_sha256(commit: str, path: Path) -> str:
    """Hash one repository file exactly as stored in ``commit``."""

    try:
        relative_path = path.resolve().relative_to(_REPO_ROOT)
    except ValueError as exc:
        raise RuntimeError(f"source path is outside the repository: {path}") from exc
    try:
        content = subprocess.check_output(
            ["git", "show", f"{commit}:{relative_path.as_posix()}"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot read {relative_path} from Git commit {commit}") from exc
    return _sha256_bytes(content)


def _nested(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _has_default_dtype_mutation(tree: ast.Module) -> bool:
    """Conservatively detect common direct, imported, and aliased mutators.

    The shared storage analyzer recognizes only ``torch.set_default_*`` calls.
    Imported aliases such as ``from torch import set_default_dtype as sdd``
    would otherwise make a dtype-less ``randn`` look like float32.  Runtime
    validation remains the final guard for dynamically hidden aliases.
    """

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


def _real_randn_factory_proof(tree: ast.Module) -> tuple[bool, str | None]:
    if _has_default_dtype_mutation(tree):
        return False, "dynamic_default_dtype"
    get_inputs = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "get_inputs"
        ),
        None,
    )
    if get_inputs is None:
        return False, "missing_top_level_get_inputs"
    calls = [
        node for node in ast.walk(get_inputs) if isinstance(node, ast.Call) and _call_name(node.func) == "torch.randn"
    ]
    if not calls:
        return False, "no_randn_factory"
    for call in calls:
        dtype_keyword = next((item.value for item in call.keywords if item.arg == "dtype"), None)
        dtype_name = _call_name(dtype_keyword) if dtype_keyword is not None else "torch.float32"
        if dtype_name not in REAL_FLOAT_DTYPES:
            return False, f"unsupported_randn_dtype:{dtype_name or 'dynamic'}"
        pin_memory = next((item.value for item in call.keywords if item.arg == "pin_memory"), None)
        if pin_memory is not None and not (isinstance(pin_memory, ast.Constant) and pin_memory.value is False):
            return False, "randn_pin_memory_not_false"
        requires_grad = next((item.value for item in call.keywords if item.arg == "requires_grad"), None)
        if requires_grad is not None and not (
            isinstance(requires_grad, ast.Constant) and requires_grad.value is False
        ):
            return False, "randn_requires_grad_not_false"
    return True, None


def _resolved_randn_specs(tree: ast.Module) -> tuple[tuple[tuple[int, ...], str], ...]:
    get_inputs = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "get_inputs"
    )
    records, _ = _factory_records(tree, get_inputs, _module_constant_environment(tree))
    return tuple((record.shape, record.dtype_name) for record in records if record.name == "torch.randn")


def _assignment(parent_uuid: str, reference_sha256: str) -> tuple[str, str]:
    payload = {
        "assignment_version": ASSIGNMENT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "parent_reference_sha256": reference_sha256,
        "parent_uuid": parent_uuid,
    }
    digest = _canonical_json_sha256(payload)
    family = VALUE_FAMILIES[int(digest[:16], 16) % len(VALUE_FAMILIES)]
    return family, digest


def _selection_hash(parent_uuid: str, reference_sha256: str, family: str) -> str:
    return _canonical_json_sha256(
        {
            "selection_version": SELECTION_VERSION,
            "parent_uuid": parent_uuid,
            "parent_reference_sha256": reference_sha256,
            "assigned_family": family,
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
    reference_sha256 = _sha256_bytes(code.encode("utf-8"))
    try:
        analysis = analyze_code(code, entry_point)
    except (SyntaxError, ValueError) as exc:
        return None, f"parent_static_analysis_failed:{type(exc).__name__}:{exc}"
    if not analysis.value_transform_eligible:
        return None, analysis.value_skip_reason or "value_transform_not_proven"
    if analysis.source_value_family != "randn":
        return None, f"source_family_not_all_randn:{analysis.source_value_family}"
    dtype_ok, dtype_reason = _real_randn_factory_proof(analysis.tree)
    if not dtype_ok:
        return None, dtype_reason or "randn_dtype_not_proven"
    family, assignment_sha256 = _assignment(parent_uuid, reference_sha256)
    try:
        continuous_specs = _resolved_randn_specs(analysis.tree)
    except ValueError as exc:
        return None, f"randn_shape_resolution_failed:{type(exc).__name__}:{exc}"
    continuous_shapes = tuple(shape for shape, _ in continuous_specs)
    if len(continuous_shapes) != analysis.continuous_factory_count:
        return None, "resolved_randn_count_mismatch"
    if family == "multinomial_categories":
        if any(not shape or shape[-1] < 2 for shape in continuous_shapes):
            return None, "multinomial_requires_last_dimension_at_least_two"
        if any(shape[-1] - 1 > MAX_CONSECUTIVE_INTEGER_BY_DTYPE[dtype_name] for shape, dtype_name in continuous_specs):
            return None, "multinomial_category_id_not_exactly_representable_in_dtype"
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
            assigned_family=family,
            assignment_sha256=assignment_sha256,
            selection_sha256=_selection_hash(parent_uuid, reference_sha256, family),
            source_family=source_family,
            operator_bucket=operator_bucket,
            operator_count=operator_count,
            factory_count=analysis.factory_count,
            continuous_factory_count=analysis.continuous_factory_count,
            continuous_shapes=continuous_shapes,
            input_bytes=analysis.input_bytes,
        ),
        "eligible",
    )


def _select_stratified(eligible: Sequence[EligibleParent], limit: int | None) -> list[EligibleParent]:
    if limit is None or limit >= len(eligible):
        return sorted(eligible, key=lambda item: item.source_row_index)
    if limit < 0:
        raise ValueError("limit must be non-negative")
    groups: dict[tuple[str, str, str], list[EligibleParent]] = collections.defaultdict(list)
    for item in eligible:
        groups[item.stratum].append(item)
    for values in groups.values():
        values.sort(key=lambda item: (item.selection_sha256, item.source_row_index))
    keys = sorted(groups, key=lambda key: _canonical_json_sha256(key))
    quotas = {key: 0 for key in keys}
    remaining = limit
    if limit >= len(keys):
        for key in keys:
            quotas[key] = 1
        remaining -= len(keys)
    capacities = {key: len(groups[key]) - quotas[key] for key in keys}
    capacity_total = sum(capacities.values())
    if remaining > capacity_total:
        raise ValueError("selection limit exceeds eligible capacity")
    if remaining and capacity_total:
        exact = {key: remaining * capacities[key] / capacity_total for key in keys}
        for key in keys:
            addition = min(capacities[key], int(exact[key]))
            quotas[key] += addition
            remaining -= addition
        remainder_order = sorted(
            keys,
            key=lambda key: (-(exact[key] - int(exact[key])), _canonical_json_sha256(key)),
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
                raise ValueError("largest-remainder allocation exhausted all strata")
    selected = [item for key in keys for item in groups[key][: quotas[key]]]
    if len(selected) != limit:
        raise ValueError(f"stratified selection produced {len(selected)} rows, expected {limit}")
    return sorted(selected, key=lambda item: item.source_row_index)


def _torch_call(name: str, *args: ast.AST) -> ast.Call:
    return ast.Call(
        func=ast.Attribute(value=ast.Name(id="torch", ctx=ast.Load()), attr=name, ctx=ast.Load()),
        args=list(args),
        keywords=[],
    )


def _attribute(value: ast.AST, name: str) -> ast.Attribute:
    return ast.Attribute(value=value, attr=name, ctx=ast.Load())


def _local_generator(value: ast.AST, seed: int) -> ast.Call:
    constructor = ast.Call(
        func=_attribute(ast.Name(id="torch", ctx=ast.Load()), "Generator"),
        args=[],
        keywords=[ast.keyword(arg="device", value=_attribute(copy.deepcopy(value), "device"))],
    )
    return ast.Call(func=_attribute(constructor, "manual_seed"), args=[ast.Constant(value=seed)], keywords=[])


def _factory_seed(seed_base: int, factory_index: int) -> int:
    seed = (seed_base + 1_000_003 * (factory_index + 1)) % LOCAL_GENERATOR_MAX_SEED
    return seed or 1


def _value_wrapper(base: ast.Call, family: str, local_seed: int) -> ast.Call:
    value = ast.Name(id="_value_draw", ctx=ast.Load())
    ones = _torch_call("ones_like", copy.deepcopy(value))
    zeros = _torch_call("zeros_like", copy.deepcopy(value))
    upper = _torch_call("nextafter", ones, zeros)
    gaussian_cdf_signed = _torch_call(
        "erf",
        ast.BinOp(
            left=copy.deepcopy(value),
            op=ast.Div(),
            right=ast.Constant(value=1.4142135623730951),
        ),
    )
    if family == "uniform_01":
        cdf = ast.BinOp(
            left=ast.BinOp(left=gaussian_cdf_signed, op=ast.Add(), right=ast.Constant(value=1.0)),
            op=ast.Mult(),
            right=ast.Constant(value=0.5),
        )
        body: ast.AST = _torch_call("minimum", cdf, upper)
    elif family == "signed_uniform":
        body = _torch_call("minimum", gaussian_cdf_signed, upper)
    elif family == "poisson_counts":
        softplus = ast.Call(
            func=_attribute(
                _attribute(_attribute(ast.Name(id="torch", ctx=ast.Load()), "nn"), "functional"),
                "softplus",
            ),
            args=[copy.deepcopy(value)],
            keywords=[],
        )
        rate = ast.Call(
            func=_attribute(ast.Name(id="torch", ctx=ast.Load()), "clamp"),
            args=[softplus],
            keywords=[
                ast.keyword(arg="min", value=ast.Constant(value=0.125)),
                ast.keyword(arg="max", value=ast.Constant(value=8.0)),
            ],
        )
        body = ast.Call(
            func=_attribute(ast.Name(id="torch", ctx=ast.Load()), "poisson"),
            args=[rate],
            keywords=[ast.keyword(arg="generator", value=_local_generator(value, local_seed))],
        )
    elif family == "multinomial_categories":
        last_dimension = ast.Subscript(
            value=_attribute(copy.deepcopy(value), "shape"),
            slice=ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1)),
            ctx=ast.Load(),
        )
        flattened = ast.Call(
            func=_attribute(copy.deepcopy(value), "reshape"),
            args=[
                ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1)),
                copy.deepcopy(last_dimension),
            ],
            keywords=[],
        )
        probabilities = ast.Call(
            func=_attribute(ast.Name(id="torch", ctx=ast.Load()), "softmax"),
            args=[flattened],
            keywords=[
                ast.keyword(
                    arg="dim",
                    value=ast.UnaryOp(op=ast.USub(), operand=ast.Constant(value=1)),
                )
            ],
        )
        samples = ast.Call(
            func=_attribute(ast.Name(id="torch", ctx=ast.Load()), "multinomial"),
            args=[probabilities],
            keywords=[
                ast.keyword(arg="num_samples", value=copy.deepcopy(last_dimension)),
                ast.keyword(arg="replacement", value=ast.Constant(value=True)),
                ast.keyword(arg="generator", value=_local_generator(value, local_seed)),
            ],
        )
        restored_shape = ast.Call(
            func=_attribute(samples, "reshape_as"),
            args=[copy.deepcopy(value)],
            keywords=[],
        )
        body = ast.Call(
            func=_attribute(restored_shape, "to"),
            args=[],
            keywords=[ast.keyword(arg="dtype", value=_attribute(copy.deepcopy(value), "dtype"))],
        )
    else:  # pragma: no cover - CLI/assignment constants constrain this
        raise AssertionError(family)
    function = ast.Lambda(
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="_value_draw")],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=body,
    )
    return ast.Call(func=function, args=[copy.deepcopy(base)], keywords=[])


class _ValueTransformer(ast.NodeTransformer):
    def __init__(self, family: str, seed_base: int) -> None:
        self.family = family
        self.seed_base = seed_base
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
        if not self.in_get_inputs or _call_name(node.func) != "torch.randn":
            return node
        self.replacements += 1
        local_seed = _factory_seed(self.seed_base, self.replacements - 1)
        return ast.copy_location(_value_wrapper(node, self.family, local_seed), node)


def _factory_signatures(tree: ast.Module) -> list[str]:
    get_inputs = next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "get_inputs"
    )
    return [
        ast.dump(node, annotate_fields=True, include_attributes=False)
        for node in ast.walk(get_inputs)
        if isinstance(node, ast.Call) and _call_name(node.func) in _SHAPE_FACTORIES
    ]


def _transform_code(
    code: str,
    entry_point: str,
    family: str,
    expected_replacements: int,
    seed_base: int,
) -> str:
    parent_tree = ast.parse(code)
    tree = copy.deepcopy(parent_tree)
    before_sections = _section_hashes(parent_tree, entry_point)
    before_factories = collections.Counter(_factory_signatures(parent_tree))
    transformer = _ValueTransformer(family, seed_base)
    transformed = transformer.visit(tree)
    assert isinstance(transformed, ast.Module)
    ast.fix_missing_locations(transformed)
    if transformer.replacements != expected_replacements:
        raise ValueError(f"randn_replacement_count_mismatch:{transformer.replacements}:{expected_replacements}")
    if _section_hashes(transformed, entry_point) != before_sections:
        raise ValueError("model_or_get_init_inputs_changed")
    if collections.Counter(_factory_signatures(transformed)) != before_factories:
        raise ValueError("input_factory_shape_dtype_layout_or_rng_call_changed")
    child_code = ast.unparse(transformed).rstrip() + "\n"
    reparsed = ast.parse(child_code)
    if ast.dump(reparsed, include_attributes=False) != ast.dump(transformed, include_attributes=False):
        raise ValueError("unparse_reparse_ast_mismatch")
    if _section_hashes(reparsed, entry_point) != before_sections:
        raise ValueError("model_or_get_init_inputs_changed_after_reparse")
    if collections.Counter(_factory_signatures(reparsed)) != before_factories:
        raise ValueError("factory_signature_changed_after_reparse")
    return child_code


def _extend_schema(schema: pa.Schema) -> pa.Schema:
    index = schema.get_field_index("extra_info")
    if index < 0 or not pa.types.is_struct(schema.field(index).type):
        raise ValueError("input schema has no extra_info struct")
    extra = schema.field(index)
    if extra.type.get_field_index("augmentation") >= 0:
        raise ValueError("canonical input already contains augmentation metadata")
    fields = list(schema)
    fields[index] = pa.field(
        "extra_info",
        pa.struct([*extra.type, pa.field("augmentation", AUGMENTATION_METADATA_TYPE)]),
        nullable=extra.nullable,
        metadata=extra.metadata,
    )
    return pa.schema(fields, metadata=schema.metadata)


def _family_labels(family: str, shapes: Sequence[Sequence[int]]) -> dict[str, Any]:
    if family == "uniform_01":
        return {
            "support": "[0,1)",
            "sign": "nonnegative",
            "zero": "measure_zero_except_finite_rounding",
            "sparsity": "dense",
            "magnitude": "bounded_unit",
            "cardinality": "continuous",
        }
    if family == "signed_uniform":
        return {
            "support": "[-1,1)",
            "sign": "signed",
            "zero": "measure_zero_except_finite_rounding",
            "sparsity": "dense",
            "magnitude": "bounded_unit",
            "cardinality": "continuous",
        }
    if family == "poisson_counts":
        return {
            "support": "nonnegative_integer_valued_float",
            "sign": "nonnegative",
            "zero": "allowed",
            "sparsity": "data_dependent",
            "magnitude": "poisson_tail_from_rate_[0.125,8]",
            "cardinality": "discrete_count",
            "rate_range": [0.125, 8.0],
        }
    if family == "multinomial_categories":
        return {
            "support": "integer_category_[0,last_dim)",
            "sign": "nonnegative",
            "zero": "allowed",
            "sparsity": "dense_category_ids",
            "magnitude": "bounded_by_last_dimension",
            "cardinality": "categorical_with_replacement",
            "category_sizes": [int(shape[-1]) for shape in shapes],
        }
    raise AssertionError(family)


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
    local_seed_base = int(eligible.assignment_sha256[16:32], 16) % LOCAL_GENERATOR_MAX_SEED
    local_generator_seeds = [
        _factory_seed(local_seed_base, index) for index in range(eligible.continuous_factory_count)
    ]
    child_code = _transform_code(
        parent_code,
        entry_point,
        eligible.assigned_family,
        eligible.continuous_factory_count,
        local_seed_base,
    )
    child_reference_sha256 = _sha256_bytes(child_code.encode("utf-8"))
    child_ast_sha256 = _normalized_ast_sha256(child_code)
    intervention = {
        "primary_intervention": "random_value",
        "family_before": "randn",
        "family_after": eligible.assigned_family,
        "mapping": GENERATOR_VERSION,
        "factory_count": eligible.continuous_factory_count,
        "local_generator": (
            {
                "scheme": "per_factory_torch_generator_v1",
                "seed_base": local_seed_base,
                "seeds": local_generator_seeds,
            }
            if eligible.assigned_family in {"poisson_counts", "multinomial_categories"}
            else None
        ),
    }
    intervention_sha256 = _canonical_json_sha256(intervention)
    child_uuid = (
        "value_"
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
            "runtime_validation_status": "value_only_paired_runtime_pending",
            "governance_status": "value_intervention_review_only",
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
        "intervention_id": f"value_{intervention_sha256[:20]}",
        "intervention_sha256": intervention_sha256,
        "intervention_kind": "value",
        "coverage_cell": f"value_{eligible.assigned_family}",
        "shape_scale": None,
        "value_family_before": "randn",
        "value_family_after": eligible.assigned_family,
        "dtype_before": None,
        "dtype_after": None,
        "layout_before": None,
        "layout_after": None,
        "shape_dimension_name": None,
        "shape_dimension_before": None,
        "shape_dimension_after": None,
        "factory_count": eligible.factory_count,
        "input_bytes_before": eligible.input_bytes,
        "input_bytes_after": eligible.input_bytes,
        "estimated_peak_bytes": None,
        "memory_budget_bytes": DEFAULT_MEMORY_BUDGET_BYTES,
        "working_set_multiplier": None,
        "memory_estimator_version": "value_only_same_final_storage_runtime_transient_guard_v2",
        "shape_proof": None,
        "compatibility_proof": (
            "all_continuous_inputs_are_bare_unique_returned_randn;"
            "same_randn_draw;global_rng_consumption_preserved;"
            "stochastic_families_use_per_factory_local_generators"
        ),
        "validation_status": "static_value_only_pass_paired_runtime_required",
    }
    child["extra_info"] = extra
    fatal, _ = inspect_row_schema(child, child_code)
    if fatal:
        raise ValueError(f"child_schema_failed:{','.join(fatal)}")
    row_sha256 = _canonical_json_sha256(child)
    manifest = {
        "manifest_contract_version": "random_value_lane_manifest_v2",
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
        "primary_intervention": "random_value",
        "assigned_target": eligible.assigned_family,
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
        "dtype_changed": False,
        "layout_changed": False,
        "model_changed": False,
        "get_init_inputs_changed": False,
        "rng_consumption_preserved": True,
        "factory_count": eligible.factory_count,
        "transformed_factory_count": eligible.continuous_factory_count,
        "input_bytes_before": eligible.input_bytes,
        "input_bytes_after": eligible.input_bytes,
        "value_labels": _family_labels(eligible.assigned_family, eligible.continuous_shapes),
        "static_status": "passed",
        "parent_runtime_status": "pending",
        "child_runtime_status": "pending",
        "liveness_status": "pending",
        "materialization_status": "review_only",
        "runtime_policy_fingerprint": None,
        "provenance_status": _nested(parent, "extra_info.v4.provenance_status"),
        "licenses": _nested(parent, "extra_info.v4.licenses", []),
        "lineage": "canonical_parent_value_only_child",
        "row_sha256": row_sha256,
        "training_approved": False,
    }
    return child, manifest


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _review_markdown(samples: Sequence[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]]) -> str:
    lines = ["# Random/value solver review samples", ""]
    for parent, child, manifest in samples:
        parent_code = str(_nested(parent, "reward_model.ground_truth", ""))
        child_code = str(_nested(child, "reward_model.ground_truth", ""))
        lines.extend(
            [
                f"## {manifest['child_uuid']}",
                "",
                (
                    f"- source: `{manifest['source_family']}`; operator: `{manifest['operator_bucket']}`; "
                    f"family: `{manifest['assigned_target']}`; parent row: `{manifest['source_row_index']}`"
                ),
                "",
                "```diff",
                *list(
                    difflib.unified_diff(
                        parent_code.splitlines(),
                        child_code.splitlines(),
                        fromfile="parent.py",
                        tofile="child.py",
                        lineterm="",
                    )
                )[:160],
                "```",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def build_lane(input_path: Path, output_dir: Path, *, limit: int | None, overwrite: bool) -> dict[str, Any]:
    if type(limit) is not int or not 1 <= limit <= MAX_AUTHORIZED_CANDIDATES:
        raise ValueError(f"limit must be an integer in [1, {MAX_AUTHORIZED_CANDIDATES}]")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    source_sha256 = _sha256_file(input_path)
    parquet = pq.ParquetFile(input_path)
    if source_sha256 != EXPECTED_CANONICAL_PARENT_SHA256:
        raise ValueError(
            f"canonical parent SHA mismatch: expected {EXPECTED_CANONICAL_PARENT_SHA256}, found {source_sha256}"
        )
    if parquet.metadata.num_rows != EXPECTED_CANONICAL_PARENT_ROWS:
        raise ValueError(
            f"canonical parent row mismatch: expected {EXPECTED_CANONICAL_PARENT_ROWS}, "
            f"found {parquet.metadata.num_rows}"
        )
    output_paths = {
        "parents": output_dir / "parents.parquet",
        "candidates": output_dir / "candidates.parquet",
        "paired": output_dir / "paired.parquet",
        "manifest": output_dir / "manifest.jsonl",
        "decisions": output_dir / "decisions.jsonl",
        "summary": output_dir / "summary.json",
        "review": output_dir / "review_samples.md",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing existing lane artifacts: {existing[:3]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    generator_path = Path(__file__).resolve()
    dependency_path = Path(__file__).resolve().parent.parent / "augment_prompt_tasks.py"
    generator_sha256 = _sha256_file(generator_path)
    dependency_sha256 = _sha256_file(dependency_path)
    git_commit = _git_commit()
    for source_path, source_sha256 in (
        (generator_path, generator_sha256),
        (dependency_path, dependency_sha256),
    ):
        if _git_blob_sha256(git_commit, source_path) != source_sha256:
            raise RuntimeError(f"source differs from Git commit {git_commit}: {source_path}")

    eligible: list[EligibleParent] = []
    skip_counts: collections.Counter[str] = collections.Counter()
    decisions: list[dict[str, Any]] = []
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
                        "assigned_target": item.assigned_family,
                        "assignment_sha256": item.assignment_sha256,
                        "reason": "static_solver_eligible",
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
        else:
            decision["selected"] = False

    output_schema = _extend_schema(parquet.schema_arrow)
    parents: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    review_pool: dict[tuple[str, str, str], tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]] = {}
    selected_row_indices = set(selected_by_row)
    row_index = 0
    for batch in parquet.iter_batches(batch_size=256, use_threads=False):
        for row in batch.to_pylist():
            if row_index in selected_row_indices:
                eligible_item = selected_by_row[row_index]
                child, manifest = _make_child(
                    row,
                    eligible_item,
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
                key = (
                    str(manifest["source_family"]),
                    str(manifest["operator_bucket"]),
                    str(manifest["assigned_target"]),
                )
                review_pool.setdefault(key, (row, child, manifest))
            row_index += 1
    if len(children) != len(selected):
        raise ValueError(f"materialized {len(children)} children for {len(selected)} selected parents")

    child_uuids = [str(item["child_uuid"]) for item in manifests]
    child_refs = [str(item["child_reference_sha256"]) for item in manifests]
    child_asts = [str(item["child_normalized_ast_sha256"]) for item in manifests]
    if len(set(child_uuids)) != len(child_uuids):
        raise ValueError("duplicate child UUID")
    if len(set(child_refs)) != len(child_refs):
        raise ValueError("duplicate child reference")
    if len(set(child_asts)) != len(child_asts):
        raise ValueError("duplicate child normalized AST")
    parent_refs = {item.parent_reference_sha256 for item in selected}
    if parent_refs & set(child_refs):
        raise ValueError("child reference collides with a selected parent")

    paired: list[dict[str, Any]] = []
    for parent, child in zip(parents, children, strict=True):
        paired.extend((parent, child))
    tmp_suffix = f".tmp.{os.getpid()}"
    for key, rows in (("parents", parents), ("candidates", children), ("paired", paired)):
        path = output_paths[key]
        tmp = path.with_name(path.name + tmp_suffix)
        pq.write_table(pa.Table.from_pylist(rows, schema=output_schema), tmp, compression="zstd")
        os.replace(tmp, path)
    _atomic_text(output_paths["manifest"], "".join(_canonical_json(item) + "\n" for item in manifests))
    _atomic_text(output_paths["decisions"], "".join(_canonical_json(item) + "\n" for item in decisions))
    review_samples = [review_pool[key] for key in sorted(review_pool)[:24]]
    _atomic_text(output_paths["review"], _review_markdown(review_samples))

    eligible_family_counts = collections.Counter(item.assigned_family for item in eligible)
    eligible_source_counts = collections.Counter(item.source_family for item in eligible)
    eligible_operator_counts = collections.Counter(item.operator_bucket for item in eligible)
    eligible_joint_counts = collections.Counter("|".join(item.stratum) for item in eligible)
    family_counts = collections.Counter(item.assigned_family for item in selected)
    source_counts = collections.Counter(item.source_family for item in selected)
    operator_counts = collections.Counter(item.operator_bucket for item in selected)
    joint_counts = collections.Counter("|".join(item.stratum) for item in selected)
    summary = {
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
        "eligible_family_counts": dict(sorted(eligible_family_counts.items())),
        "eligible_source_counts": dict(sorted(eligible_source_counts.items())),
        "eligible_operator_bucket_counts": dict(sorted(eligible_operator_counts.items())),
        "eligible_source_operator_family_counts": dict(sorted(eligible_joint_counts.items())),
        "selected_parents": len(selected),
        "candidate_rows": len(children),
        "paired_rows": len(paired),
        "selection_limit": limit,
        "family_counts": dict(sorted(family_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "operator_bucket_counts": dict(sorted(operator_counts.items())),
        "source_operator_family_counts": dict(sorted(joint_counts.items())),
        "skip_reason_counts": dict(sorted(skip_counts.items())),
        "proof": {
            "primary_intervention": "random_value",
            "one_family_per_parent": True,
            "shape_changed": False,
            "dtype_changed": False,
            "layout_changed": False,
            "model_changed": False,
            "get_init_inputs_changed": False,
            "global_rng_consumption_preserved": True,
            "static_domain": "bare unique returned real-floating all-randn factories",
            "runtime_boundary": "paired parent/child reference plus family/value/output liveness required",
        },
        "artifacts": {},
        "training_approved": False,
    }
    for key, path in output_paths.items():
        if key == "summary":
            continue
        summary["artifacts"][key] = {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
        }
    _atomic_text(output_paths["summary"], json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--limit",
        type=int,
        required=True,
        help=f"Deterministic stratified canary size in [1, {MAX_AUTHORIZED_CANDIDATES}]",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = build_lane(args.input, args.output_dir, limit=args.limit, overwrite=args.overwrite)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
