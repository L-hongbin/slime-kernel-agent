#!/usr/bin/env python3
"""Analyze and independently verify a static shape-solver pilot run.

The solver run is intentionally kept separate from training data.  This tool
checks every accepted child against its unchanged parent, summarizes coverage
and failure modes, audits the generated shape distribution, and writes small
reviewable artifacts under ``RUN_DIR/analysis``.
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import difflib
import hashlib
import io
import json
import math
import os
import sys
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.synthesize.ai_shape_coverage import _operator_family, static_gate
from tools.data.synthesize.augment_prompt_tasks import (
    _SHAPE_FACTORIES,
    _call_name,
    _factory_records,
    _module_constant_environment,
    _normalized_ast_sha256,
    _shape_nodes,
)
from tools.data.synthesize.solve_shape_coverage import _shape_slots


DEFAULT_RUN_DIR = REPO_ROOT / "Data/prompt_tvm_v4/shape_solver_random_targets_v3/run.1000"
DEFAULT_AI_RUN_DIR = REPO_ROOT / "Data/prompt_tvm_v4/shape_ai_random_targets_low_tp8_v7/run.1000"
SCHEMA_VERSION = "shape-solver-analysis-v5"
REFERENCE_CONTRACT_VERSION = "kernelgym-reference-self-train-mode-v3"
REGION_CONTRACT_VERSION = "shape_changed_region_liveness_v3"
PERTURBATION_CONTRACT_VERSION = "seeded_bounded_non_affine_mix_v1"
REGION_RUN_BINDING_CONTRACT_VERSION = "shape_changed_region_liveness_run_binding_v2"
PREVIOUS_REGION_RUN_BINDING_CONTRACT_VERSION = (
    "shape_changed_region_liveness_run_binding_v1"
)
PREVIOUS_REGION_CONTRACT_VERSION = "shape_changed_region_liveness_v2"
LEGACY_REGION_CONTRACT_VERSION = "shape_changed_region_liveness_v1"
MIB = 1024**2
DIMENSION_ANCHORS = frozenset({64, 128, 256, 512, 1024, 2048, 4096})
CAPACITY_ANCHORS = frozenset(value * MIB for value in DIMENSION_ANCHORS)
MAX_REVIEW_ACCEPTED = 20
MAX_REVIEW_FAILURES = 12
POWER_OF_TWO_MIN_NUMERATOR = 3
POWER_OF_TWO_MIN_DENOMINATOR = 10
POWER_OF_TWO_MAX_NUMERATOR = 1
POWER_OF_TWO_MAX_DENOMINATOR = 2
SINGLE_CHANGED_MAX_NUMERATOR = 1
SINGLE_CHANGED_MAX_DENOMINATOR = 10

STATIC_MANIFEST_ARTIFACTS = {
    "selected": Path("selected.parquet"),
    "selection": Path("selection.json"),
    "targets": Path("targets.parquet"),
    "children": Path("static/children.parquet"),
    "paired": Path("static/paired.parquet"),
    "review": Path("analysis/review_samples.md"),
}


def _nested(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_static_manifest_artifacts(
    run_dir: Path,
    manifest: Mapping[str, Any],
) -> None:
    expected = manifest.get("artifact_sha256")
    if not isinstance(expected, Mapping):
        raise ValueError("solver manifest is missing artifact_sha256")
    for name, relative_path in STATIC_MANIFEST_ARTIFACTS.items():
        path = run_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        declared = expected.get(name)
        actual = _sha256_file(path)
        if declared != actual:
            raise ValueError(
                f"solver manifest artifact hash mismatch for {name}:"
                f" expected {declared}, found {actual}"
            )


def _canonical_mapping_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _percentile(values: Sequence[int | float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def _counter(counter: Mapping[Any, int]) -> dict[str, int]:
    return {
        str(key): int(counter[key])
        for key in sorted(counter, key=lambda item: (type(item).__name__, repr(item)))
    }


def _distribution(values: Sequence[int | float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "min": min(values) if values else None,
        "p10": _percentile(values, 0.10),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "max": max(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
    }


def _value_profile(values: Sequence[int], sample_count: int) -> dict[str, Any]:
    counts = collections.Counter(values)
    total = len(values)
    entropy = -sum(
        (count / total) * math.log2(count / total) for count in counts.values()
    ) if total else 0.0
    top = counts.most_common(20)
    hhi = sum((count / total) ** 2 for count in counts.values()) if total else 0.0
    return {
        **_distribution(values),
        "samples": sample_count,
        "unique_values": len(counts),
        "singleton_values": sum(count == 1 for count in counts.values()),
        "shannon_entropy_bits": entropy,
        "normalized_entropy": entropy / math.log2(len(counts)) if len(counts) > 1 else 0.0,
        "effective_values": 2**entropy,
        "hhi": hhi,
        "top1_fraction": _fraction(sum(count for _, count in top[:1]), total),
        "top5_fraction": _fraction(sum(count for _, count in top[:5]), total),
        "top10_fraction": _fraction(sum(count for _, count in top[:10]), total),
        "power_of_two_count": sum(
            count for value, count in counts.items() if value > 0 and value & (value - 1) == 0
        ),
        "power_of_two_fraction": _fraction(
            sum(count for value, count in counts.items() if value > 0 and value & (value - 1) == 0),
            total,
        ),
        "multiple_fractions": {
            str(multiple): _fraction(sum(value % multiple == 0 for value in values), total)
            for multiple in (8, 32)
        },
        "residue_histograms": {
            str(modulus): {
                "expected_uniform_fraction": 1 / modulus,
                "counts": {
                    str(residue): sum(value % modulus == residue for value in values)
                    for residue in range(modulus)
                },
                "fractions": {
                    str(residue): _fraction(
                        sum(value % modulus == residue for value in values),
                        total,
                    )
                    for residue in range(modulus)
                },
                "minus_one_fraction": _fraction(
                    sum(value % modulus == modulus - 1 for value in values),
                    total,
                ),
            }
            for modulus in (8, 16, 32)
        },
        "anchor_count": sum(value in DIMENSION_ANCHORS for value in values),
        "anchor_fraction": _fraction(sum(value in DIMENSION_ANCHORS for value in values), total),
        "top_values": [
            {"value": value, "count": count, "fraction": _fraction(count, total)}
            for value, count in top
        ],
    }


def _partitioned_value_profiles(
    values: Sequence[int],
    sample_count: int,
) -> dict[str, dict[str, Any]]:
    power_of_two = [value for value in values if value > 0 and value & (value - 1) == 0]
    non_power_of_two = [
        value for value in values if not (value > 0 and value & (value - 1) == 0)
    ]
    return {
        "power_of_two": _value_profile(power_of_two, sample_count),
        "non_power_of_two": _value_profile(non_power_of_two, sample_count),
    }


def _capacity_profile(values: Sequence[int]) -> dict[str, Any]:
    counts = collections.Counter(values)
    total = len(values)
    return {
        **_distribution(values),
        "unique_byte_sizes": len(counts),
        "hhi": sum((count / total) ** 2 for count in counts.values()) if total else 0.0,
        "integer_mib_fraction": _fraction(sum(value % MIB == 0 for value in values), total),
        "multiple_8_mib_fraction": _fraction(sum(value % (8 * MIB) == 0 for value in values), total),
        "multiple_32_mib_fraction": _fraction(sum(value % (32 * MIB) == 0 for value in values), total),
        "anchor_fraction": _fraction(sum(value in CAPACITY_ANCHORS for value in values), total),
        "top_sizes": [
            {
                "bytes": value,
                "mib": value / MIB,
                "count": count,
                "fraction": _fraction(count, total),
            }
            for value, count in counts.most_common(15)
        ],
    }


def _top_level_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected_one_top_level_{name}:{len(matches)}")
    return matches[0]


def _top_level_class(tree: ast.Module, name: str) -> ast.ClassDef:
    matches = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected_one_top_level_class_{name}:{len(matches)}")
    return matches[0]


def _factory_signature(tree: ast.Module) -> list[dict[str, Any]]:
    get_inputs = _top_level_function(tree, "get_inputs")
    calls = sorted(
        (
            node
            for node in ast.walk(get_inputs)
            if isinstance(node, ast.Call) and _call_name(node.func) in _SHAPE_FACTORIES
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    return [
        {
            "name": _call_name(call.func),
            "rank": len(_shape_nodes(call, _call_name(call.func))),
        }
        for call in calls
    ]


def _resolved_direct_factories(code: str) -> list[dict[str, Any]]:
    """Resolve direct ``get_inputs`` factories in stable source order."""

    tree = ast.parse(code)
    get_inputs = _top_level_function(tree, "get_inputs")
    records, _ = _factory_records(
        tree,
        get_inputs,
        _module_constant_environment(tree),
    )
    ordered = sorted(records, key=lambda record: (record.line_number, record.column_offset))
    locations = [(record.line_number, record.column_offset) for record in ordered]
    if len(locations) != len(set(locations)):
        raise ValueError("duplicate_direct_factory_source_location")
    return [
        {
            "factory_index": factory_index,
            "factory_name": record.name,
            "shape": tuple(int(value) for value in record.shape),
            "dtype_name": record.dtype_name,
            "dtype_bytes": int(record.dtype_bytes),
            "line_number": int(record.line_number),
            "column_offset": int(record.column_offset),
        }
        for factory_index, record in enumerate(ordered)
    ]


def _independent_shape_diff(parent_code: str, child_code: str) -> list[dict[str, Any]]:
    """Derive changed input axes from independently resolved parent/child source."""

    parent_factories = _resolved_direct_factories(parent_code)
    child_factories = _resolved_direct_factories(child_code)
    if len(parent_factories) != len(child_factories):
        raise ValueError(
            f"direct_factory_count_changed:{len(parent_factories)}:{len(child_factories)}"
        )
    changed: list[dict[str, Any]] = []
    for parent, child in zip(parent_factories, child_factories, strict=True):
        factory_index = int(parent["factory_index"])
        if child["factory_index"] != factory_index:
            raise ValueError(f"direct_factory_index_changed:{factory_index}")
        if child["factory_name"] != parent["factory_name"]:
            raise ValueError(
                f"direct_factory_name_changed:{factory_index}:"
                f"{parent['factory_name']}:{child['factory_name']}"
            )
        if child["dtype_name"] != parent["dtype_name"] or child["dtype_bytes"] != parent["dtype_bytes"]:
            raise ValueError(f"direct_factory_dtype_changed:{factory_index}")
        parent_shape = tuple(parent["shape"])
        child_shape = tuple(child["shape"])
        if len(parent_shape) != len(child_shape):
            raise ValueError(
                f"direct_factory_rank_changed:{factory_index}:"
                f"{len(parent_shape)}:{len(child_shape)}"
            )
        rank = len(parent_shape)
        for axis, (old_value, new_value) in enumerate(
            zip(parent_shape, child_shape, strict=True)
        ):
            if old_value == new_value:
                continue
            changed.append(
                {
                    "factory_index": factory_index,
                    "factory_name": parent["factory_name"],
                    "axis": axis,
                    "rank": rank,
                    "axis_from_right": rank - axis - 1,
                    "old_value": int(old_value),
                    "new_value": int(new_value),
                    "parent_shape": list(parent_shape),
                    "child_shape": list(child_shape),
                }
            )
    if not changed:
        raise ValueError("accepted_child_has_no_changed_direct_factory_dimension")
    keys = [(item["factory_index"], item["axis"]) for item in changed]
    if len(keys) != len(set(keys)):
        raise ValueError("independent_shape_diff_contains_duplicate_occurrence")
    return changed


def _solver_slot_edits(decision: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize legacy singular and non-empty multi-slot solver evidence."""

    solver = decision.get("solver")
    if not isinstance(solver, Mapping):
        raise ValueError("accepted_decision_missing_solver")
    raw_multi = solver.get("slots")
    if raw_multi is not None:
        if not isinstance(raw_multi, list) or not raw_multi:
            raise ValueError("accepted_solver_slots_must_be_a_nonempty_list")
        if solver.get("slot") is not None or solver.get("slot_value") is not None:
            raise ValueError("accepted_solver_mixes_singular_and_multi_slot_contracts")
        edits: list[dict[str, Any]] = []
        for edit_index, raw_edit in enumerate(raw_multi):
            if not isinstance(raw_edit, Mapping):
                raise ValueError(f"solver_slot_edit_not_an_object:{edit_index}")
            slot = raw_edit.get("slot")
            new_value = raw_edit.get("new_value")
            if not isinstance(slot, Mapping):
                raise ValueError(f"solver_slot_edit_missing_slot:{edit_index}")
            if type(new_value) is not int or new_value <= 0:
                raise ValueError(
                    f"solver_slot_edit_invalid_new_value:{edit_index}:{new_value!r}"
                )
            edits.append(
                {"slot_index": edit_index, "slot": slot, "new_value": new_value}
            )
    else:
        slot = solver.get("slot")
        new_value = solver.get("slot_value")
        if not isinstance(slot, Mapping):
            raise ValueError("accepted_decision_missing_solver_slot")
        if type(new_value) is not int or new_value <= 0:
            raise ValueError(f"invalid_slot_value:{new_value!r}")
        edits = [{"slot_index": 0, "slot": slot, "new_value": new_value}]
    slot_ids = [edit["slot"].get("slot_id") for edit in edits]
    if any(not isinstance(slot_id, str) or not slot_id for slot_id in slot_ids):
        raise ValueError("accepted_solver_slot_has_invalid_slot_id")
    if len(slot_ids) != len(set(slot_ids)):
        raise ValueError("accepted_solver_slot_ids_are_not_unique")
    return edits


def _declared_occurrences(
    edits: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, int], dict[str, Any]]:
    """Index declared occurrences while rejecting duplicate ownership."""

    result: dict[tuple[int, int], dict[str, Any]] = {}
    for edit in edits:
        slot = edit["slot"]
        occurrences = slot.get("occurrences")
        if not isinstance(occurrences, list) or not occurrences:
            raise ValueError(
                f"accepted_slot_has_no_occurrences:{slot.get('slot_id')}"
            )
        for slot_occurrence_index, occurrence in enumerate(occurrences):
            if not isinstance(occurrence, Mapping):
                raise ValueError(
                    f"slot_occurrence_not_an_object:{slot.get('slot_id')}:"
                    f"{slot_occurrence_index}"
                )
            factory_index = occurrence.get("factory_index")
            axis = occurrence.get("axis")
            if type(factory_index) is not int or type(axis) is not int:
                raise ValueError(
                    f"slot_occurrence_has_invalid_key:{slot.get('slot_id')}:"
                    f"{factory_index!r}:{axis!r}"
                )
            key = (factory_index, axis)
            if key in result:
                raise ValueError(
                    f"changed_occurrence_declared_by_multiple_slots:{factory_index}:{axis}"
                )
            result[key] = {
                "slot_index": int(edit["slot_index"]),
                "slot_occurrence_index": slot_occurrence_index,
                "slot": slot,
                "new_value": int(edit["new_value"]),
                "occurrence": occurrence,
            }
    return result


def _dimension_balance_ratio_limit(manifest: Mapping[str, Any]) -> int | float | None:
    """Return the declared hard limit, or None for pre-contract solver runs."""

    if "dimension_balance_guard_contract" not in manifest:
        return None
    contract = manifest["dimension_balance_guard_contract"]
    if not isinstance(contract, Mapping):
        raise ValueError("dimension_balance_guard_contract_must_be_an_object")
    ratio_limit = contract.get("ratio_limit")
    if (
        type(ratio_limit) not in (int, float)
        or not math.isfinite(float(ratio_limit))
        or ratio_limit <= 0
    ):
        raise ValueError(f"invalid_dimension_balance_ratio_limit:{ratio_limit!r}")
    return ratio_limit


def _required_logical_slot_count(manifest: Mapping[str, Any]) -> int | None:
    """Return the exact cardinality only for the legacy exact-two contract."""

    contract_version = manifest.get("contract_version")
    if contract_version != "shape_multidim_solver_v4":
        return None
    pair_contract = manifest.get("pair_contract")
    if not isinstance(pair_contract, Mapping):
        raise ValueError("shape_multidim_solver_v4 requires pair_contract")
    return 2


def _logical_slot_count_range(
    manifest: Mapping[str, Any],
) -> tuple[int, int] | None:
    """Return a declared variable-cardinality range, if present."""

    contract = manifest.get("group_contract")
    if contract is None:
        if manifest.get("contract_version") == "shape_variable_multislot_solver_v5":
            raise ValueError(
                "shape_variable_multislot_solver_v5 requires group_contract"
            )
        return None
    if not isinstance(contract, Mapping):
        raise ValueError("group_contract_must_be_an_object")
    raw = contract.get("logical_slot_count_range")
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or any(type(value) is not int for value in raw)
        or raw[0] <= 0
        or raw[0] > raw[1]
    ):
        raise ValueError(f"invalid_logical_slot_count_range:{raw!r}")
    return int(raw[0]), int(raw[1])


def _dimension_balance_child_audit(
    child_code: str,
    decision: Mapping[str, Any],
    ratio_limit: int | float | None,
    changed_occurrences: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve every final direct factory without trusting solver shape evidence."""

    try:
        declared_occurrence_count = len(
            _declared_occurrences(_solver_slot_edits(decision))
        )
    except (TypeError, ValueError):
        declared_occurrence_count = 0
    result: dict[str, Any] = {
        "child_uuid": decision.get("child_uuid"),
        "resolved": False,
        "reason": None,
        "declared_occurrence_count": declared_occurrence_count,
        "occurrences": [],
        "factories": [],
        "child_max_ratio": None,
        "boundary": False if ratio_limit is not None else None,
        "violation": False if ratio_limit is not None else None,
    }
    try:
        if not changed_occurrences:
            raise ValueError("accepted_child_has_no_independently_changed_occurrence")
        child_factories = _resolved_direct_factories(child_code)
        seen_occurrences: set[tuple[int, int]] = set()
        affected_factories: dict[int, dict[str, Any]] = {}
        for occurrence_index, occurrence in enumerate(changed_occurrences):
            if not isinstance(occurrence, Mapping):
                raise ValueError(f"occurrence_not_an_object:{occurrence_index}")
            integer_fields: dict[str, int] = {}
            for field in ("factory_index", "axis", "rank", "axis_from_right"):
                value = occurrence.get(field)
                if type(value) is not int:
                    raise ValueError(
                        f"occurrence_{field}_not_an_integer:{occurrence_index}:{value!r}"
                    )
                integer_fields[field] = value
            factory_index = integer_fields["factory_index"]
            axis = integer_fields["axis"]
            expected_rank = integer_fields["rank"]
            expected_axis_from_right = integer_fields["axis_from_right"]
            factory_name = occurrence.get("factory_name")
            if not isinstance(factory_name, str) or not factory_name:
                raise ValueError(f"invalid_occurrence_factory_name:{occurrence_index}")
            if not 0 <= factory_index < len(child_factories):
                raise ValueError(
                    f"occurrence_factory_index_out_of_range:{factory_index}:"
                    f"{len(child_factories)}"
                )
            if (factory_index, axis) in seen_occurrences:
                raise ValueError(
                    f"duplicate_independent_occurrence:{factory_index}:{axis}"
                )
            seen_occurrences.add((factory_index, axis))

            factory = child_factories[factory_index]
            actual_factory_name = str(factory["factory_name"])
            if actual_factory_name != factory_name:
                raise ValueError(
                    f"occurrence_factory_name_mismatch:{factory_index}:"
                    f"{factory_name}:{actual_factory_name}"
                )
            shape = tuple(int(value) for value in factory["shape"])
            if len(shape) != expected_rank:
                raise ValueError(
                    f"occurrence_rank_mismatch:{factory_index}:{expected_rank}:{len(shape)}"
                )
            if not 0 <= axis < expected_rank:
                raise ValueError(
                    f"occurrence_axis_out_of_range:{factory_index}:{axis}:{expected_rank}"
                )
            if expected_axis_from_right != expected_rank - axis - 1:
                raise ValueError(
                    f"occurrence_axis_from_right_mismatch:{factory_index}:{axis}:"
                    f"{expected_axis_from_right}:{expected_rank - axis - 1}"
                )

            actual_new_value = occurrence.get("new_value")
            if type(actual_new_value) is not int or actual_new_value <= 0:
                raise ValueError(
                    f"independent_occurrence_new_value_invalid:{factory_index}:{axis}:"
                    f"{actual_new_value!r}"
                )
            if shape[axis] != actual_new_value:
                raise ValueError(
                    f"independent_occurrence_child_shape_mismatch:{factory_index}:{axis}:"
                    f"{shape[axis]}:{actual_new_value}"
                )

            ratio: float | None = None
            boundary: bool | None = None
            violation: bool | None = None
            largest: int | None = None
            second_largest: int | None = None
            if expected_rank >= 2:
                largest, second_largest = sorted(shape, reverse=True)[:2]
                ratio = largest / second_largest
                if ratio_limit is not None:
                    boundary = largest == ratio_limit * second_largest
                    violation = largest > ratio_limit * second_largest
            occurrence_evidence = {
                "occurrence_index": occurrence_index,
                "factory_index": factory_index,
                "factory_name": factory_name,
                "axis": axis,
                "rank": expected_rank,
                "shape": list(shape),
                "largest_dimension": largest,
                "second_largest_dimension": second_largest,
                "ratio": ratio,
                "boundary": boundary,
                "violation": violation,
            }
            result["occurrences"].append(occurrence_evidence)

            existing = affected_factories.get(factory_index)
            if existing is None:
                affected_factories[factory_index] = {
                    "factory_index": factory_index,
                    "factory_name": factory_name,
                    "rank": expected_rank,
                    "shape": list(shape),
                    "affected_axes": [axis],
                    "largest_dimension": largest,
                    "second_largest_dimension": second_largest,
                    "ratio": ratio,
                    "boundary": boundary,
                    "violation": violation,
                }
            else:
                if (
                    existing["factory_name"] != factory_name
                    or existing["rank"] != expected_rank
                    or existing["shape"] != list(shape)
                ):
                    raise ValueError(f"inconsistent_affected_factory:{factory_index}")
                existing["affected_axes"].append(axis)

        final_factories: list[dict[str, Any]] = []
        for factory_index, factory in enumerate(child_factories):
            factory_name = str(factory["factory_name"])
            shape = tuple(int(value) for value in factory["shape"])
            largest: int | None = None
            second_largest: int | None = None
            ratio: float | None = None
            boundary: bool | None = None
            violation: bool | None = None
            if len(shape) >= 2:
                largest, second_largest = sorted(shape, reverse=True)[:2]
                ratio = largest / second_largest
                if ratio_limit is not None:
                    boundary = largest == ratio_limit * second_largest
                    violation = largest > ratio_limit * second_largest
            affected = affected_factories.get(factory_index)
            if affected is not None and (
                affected["factory_name"] != factory_name
                or affected["rank"] != len(shape)
                or affected["shape"] != list(shape)
            ):
                raise ValueError(f"inconsistent_final_factory:{factory_index}")
            final_factories.append(
                {
                    "factory_index": factory_index,
                    "factory_name": factory_name,
                    "rank": len(shape),
                    "shape": list(shape),
                    "affected_axes": (
                        sorted(affected["affected_axes"])
                        if affected is not None
                        else []
                    ),
                    "largest_dimension": largest,
                    "second_largest_dimension": second_largest,
                    "ratio": ratio,
                    "boundary": boundary,
                    "violation": violation,
                }
            )
        result["factories"] = final_factories
        factory_ratios = [
            float(factory["ratio"])
            for factory in result["factories"]
            if factory["ratio"] is not None
        ]
        result["child_max_ratio"] = max(factory_ratios) if factory_ratios else None
        if ratio_limit is not None:
            result["boundary"] = any(factory["boundary"] for factory in result["factories"])
            result["violation"] = any(factory["violation"] for factory in result["factories"])
        result["resolved"] = True
    except (SyntaxError, TypeError, ValueError) as exc:
        result["reason"] = f"{type(exc).__name__}:{exc}"
    return result


def _ratio_distribution(values: Sequence[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "p99": _percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def _dimension_balance_summary(
    audits: Sequence[Mapping[str, Any]],
    ratio_limit: int | float | None,
) -> dict[str, Any]:
    resolved = [audit for audit in audits if audit.get("resolved") is True]
    occurrence_evidence = [
        occurrence
        for audit in resolved
        for occurrence in audit.get("occurrences", [])
        if isinstance(occurrence, Mapping)
    ]
    factory_evidence = [
        factory
        for audit in resolved
        for factory in audit.get("factories", [])
        if isinstance(factory, Mapping)
    ]
    occurrence_ratios = [
        float(occurrence["ratio"])
        for occurrence in occurrence_evidence
        if occurrence.get("ratio") is not None
    ]
    factory_ratios = [
        float(factory["ratio"])
        for factory in factory_evidence
        if factory.get("ratio") is not None
    ]
    affected_factory_evidence = [
        factory for factory in factory_evidence if factory.get("affected_axes")
    ]
    affected_factory_ratios = [
        float(factory["ratio"])
        for factory in affected_factory_evidence
        if factory.get("ratio") is not None
    ]
    child_ratios = [
        float(audit["child_max_ratio"])
        for audit in resolved
        if audit.get("child_max_ratio") is not None
    ]
    resolution_failures = [
        {
            "child_uuid": audit.get("child_uuid"),
            "reason": audit.get("reason"),
        }
        for audit in audits
        if audit.get("resolved") is not True
    ]

    if ratio_limit is None:
        boundary = {"children": None, "occurrences": None, "factories": None}
        violations = {
            "children": None,
            "occurrences": None,
            "factories": None,
            "details": [],
        }
        passed: bool | None = None
    else:
        boundary = {
            "children": sum(audit.get("boundary") is True for audit in resolved),
            "occurrences": sum(
                occurrence.get("boundary") is True for occurrence in occurrence_evidence
            ),
            "factories": sum(factory.get("boundary") is True for factory in factory_evidence),
        }
        violating_factories = [
            {
                "child_uuid": audit.get("child_uuid"),
                **factory,
            }
            for audit in resolved
            for factory in audit.get("factories", [])
            if isinstance(factory, Mapping) and factory.get("violation") is True
        ]
        violations = {
            "children": sum(audit.get("violation") is True for audit in resolved),
            "occurrences": sum(
                occurrence.get("violation") is True for occurrence in occurrence_evidence
            ),
            "factories": len(violating_factories),
            "details": violating_factories[:100],
        }
        passed = not resolution_failures and not violating_factories

    return {
        "measurement_contract": {
            "source": "independent parent/child direct-factory shape diff and child-source reparse",
            "child_ratio": "maximum largest/second-largest ratio across all final rank>=2 direct factories",
            "occurrence_ratio": "final affected factory ratio repeated once per actual changed (factory, axis)",
            "factory_ratio": "final ratio once per direct factory within each child",
        },
        "manifest_contract_present": ratio_limit is not None,
        "enforced": ratio_limit is not None,
        "ratio_limit": ratio_limit,
        "comparison": (
            f"largest_dimension <= {ratio_limit} * second_largest_dimension"
            if ratio_limit is not None
            else None
        ),
        "accepted_children": len(audits),
        "resolved_children": len(resolved),
        "resolution_failure_children": len(resolution_failures),
        "resolution_failures": resolution_failures[:100],
        "declared_occurrences": sum(
            int(audit.get("declared_occurrence_count", 0)) for audit in audits
        ),
        "resolved_occurrences": len(occurrence_evidence),
        "rank_ge_2_occurrences": len(occurrence_ratios),
        "final_direct_factories": len(factory_evidence),
        "affected_factories": len(affected_factory_evidence),
        "rank_ge_2_factories": len(factory_ratios),
        "ratios": {
            "per_child_max": _ratio_distribution(child_ratios),
            "per_occurrence": _ratio_distribution(occurrence_ratios),
            "per_final_direct_factory": _ratio_distribution(factory_ratios),
            "per_affected_factory": _ratio_distribution(affected_factory_ratios),
        },
        "boundary_counts": boundary,
        "violations": violations,
        "passed": passed,
    }


def _apply_declared_patches(
    parent_code: str,
    edits: Sequence[Mapping[str, Any]],
) -> str:
    """Apply every declared slot span simultaneously against the parent bytes."""

    raw = parent_code.encode("utf-8")
    lines = raw.splitlines(keepends=True)
    line_offsets: list[int] = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line)
    replacements: list[tuple[int, int, bytes]] = []
    for edit in edits:
        slot = edit["slot"]
        new_value = edit["new_value"]
        old_value = slot.get("old_value")
        if type(old_value) is not int or old_value <= 0:
            raise ValueError(
                f"declared_patch_invalid_old_value:{slot.get('slot_id')}:{old_value!r}"
            )
        raw_spans = slot.get("patch_spans")
        if not isinstance(raw_spans, list) or not raw_spans:
            raise ValueError(f"declared_patch_has_no_spans:{slot.get('slot_id')}")
        for raw_span in raw_spans:
            if not isinstance(raw_span, Mapping):
                raise ValueError(f"declared_patch_span_not_an_object:{slot.get('slot_id')}")
            span = {
                key: int(raw_span[key])
                for key in ("lineno", "col_offset", "end_lineno", "end_col_offset")
            }
            if not 1 <= span["lineno"] <= len(lines) or not 1 <= span["end_lineno"] <= len(lines):
                raise ValueError("declared_patch_line_out_of_range")
            start = line_offsets[span["lineno"] - 1] + span["col_offset"]
            end = line_offsets[span["end_lineno"] - 1] + span["end_col_offset"]
            token = raw[start:end].decode("utf-8")
            parsed = ast.literal_eval(token)
            if type(parsed) is not int or parsed != old_value:
                raise ValueError(
                    f"declared_patch_old_value_mismatch:{token!r}:{old_value}"
                )
            replacements.append((start, end, str(new_value).encode("ascii")))
    if not replacements:
        raise ValueError("declared_patch_has_no_spans")
    patched = raw
    previous_start = len(raw) + 1
    for start, end, replacement in sorted(replacements, reverse=True):
        if end > previous_start:
            raise ValueError("declared_patch_spans_overlap")
        patched = patched[:start] + replacement + patched[end:]
        previous_start = start
    return patched.decode("utf-8")


def _canonical_slot(slot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "slot_id": slot.get("slot_id"),
        "kind": slot.get("kind"),
        "old_value": slot.get("old_value"),
        "patch_spans": slot.get("patch_spans", []),
        "occurrences": slot.get("occurrences", []),
        "symbol_name": slot.get("symbol_name"),
        "symbol_scope": slot.get("symbol_scope"),
    }


def _annotate_independent_changes(
    edits: Sequence[Mapping[str, Any]],
    changed_occurrences: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Cross-check declared occurrences and attach slot provenance to real diffs."""

    declared = _declared_occurrences(edits)
    actual_by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    for occurrence in changed_occurrences:
        key = (int(occurrence["factory_index"]), int(occurrence["axis"]))
        if key in actual_by_key:
            raise ValueError(
                f"independent_shape_diff_duplicate:{key[0]}:{key[1]}"
            )
        actual_by_key[key] = occurrence
    if set(declared) != set(actual_by_key):
        missing = sorted(set(actual_by_key) - set(declared))
        extra = sorted(set(declared) - set(actual_by_key))
        raise ValueError(
            f"declared_vs_actual_changed_occurrence_mismatch:missing={missing[:20]}:"
            f"extra={extra[:20]}"
        )

    annotated: list[dict[str, Any]] = []
    for key in sorted(actual_by_key):
        actual = actual_by_key[key]
        evidence = declared[key]
        slot = evidence["slot"]
        occurrence = evidence["occurrence"]
        expected = {
            "factory_name": actual["factory_name"],
            "rank": actual["rank"],
            "axis_from_right": actual["axis_from_right"],
        }
        for field, expected_value in expected.items():
            if occurrence.get(field) != expected_value:
                raise ValueError(
                    f"declared_occurrence_{field}_mismatch:{key[0]}:{key[1]}:"
                    f"{occurrence.get(field)!r}:{expected_value!r}"
                )
        if slot.get("old_value") != actual["old_value"]:
            raise ValueError(
                f"declared_occurrence_old_value_mismatch:{key[0]}:{key[1]}:"
                f"{slot.get('old_value')!r}:{actual['old_value']!r}"
            )
        if evidence["new_value"] != actual["new_value"]:
            raise ValueError(
                f"declared_occurrence_new_value_mismatch:{key[0]}:{key[1]}:"
                f"{evidence['new_value']!r}:{actual['new_value']!r}"
            )
        annotated.append(
            {
                **actual,
                "slot_index": evidence["slot_index"],
                "slot_occurrence_index": evidence["slot_occurrence_index"],
                "slot": slot,
                "declared_occurrence": occurrence,
            }
        )
    return annotated


def _verify_child(
    parent: Mapping[str, Any],
    child: Mapping[str, Any],
    decision: Mapping[str, Any],
    extracted_slots: Mapping[str, Mapping[str, Any]],
    target: Mapping[str, Any] | None,
    changed_occurrences: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, bool], list[str]]:
    errors: list[str] = []
    checks: dict[str, bool] = {}
    parent_code = _nested(parent, "reward_model.ground_truth")
    child_code = _nested(child, "reward_model.ground_truth")
    entry_point = str(_nested(parent, "extra_info.entry_point", "Model"))
    if not isinstance(parent_code, str) or not isinstance(child_code, str):
        return {}, ["missing_parent_or_child_code"]

    try:
        parent_tree, child_tree = ast.parse(parent_code), ast.parse(child_code)
    except SyntaxError as exc:
        return {}, [f"syntax_error:{exc}"]

    def record(name: str, operation: Any) -> None:
        try:
            value = operation() if callable(operation) else operation
            checks[name] = bool(value)
            if not checks[name]:
                errors.append(name)
        except Exception as exc:  # noqa: BLE001 - report every invariant independently
            checks[name] = False
            errors.append(f"{name}:{type(exc).__name__}:{exc}")

    record(
        "model_ast_equal",
        lambda: ast.dump(_top_level_class(parent_tree, entry_point), include_attributes=False)
        == ast.dump(_top_level_class(child_tree, entry_point), include_attributes=False),
    )
    record(
        "get_init_inputs_ast_equal",
        lambda: ast.dump(_top_level_function(parent_tree, "get_init_inputs"), include_attributes=False)
        == ast.dump(_top_level_function(child_tree, "get_init_inputs"), include_attributes=False),
    )
    record(
        "factory_sequence_equal",
        lambda: [item["name"] for item in _factory_signature(parent_tree)]
        == [item["name"] for item in _factory_signature(child_tree)],
    )
    record(
        "factory_rank_equal",
        lambda: [item["rank"] for item in _factory_signature(parent_tree)]
        == [item["rank"] for item in _factory_signature(child_tree)],
    )

    try:
        edits = _solver_slot_edits(decision)
    except (TypeError, ValueError) as exc:
        errors.append(f"accepted_solver_slots_invalid:{type(exc).__name__}:{exc}")
        return checks, errors
    slot_ids = [str(edit["slot"].get("slot_id")) for edit in edits]
    record(
        "declared_slots_reextracted",
        lambda: all(slot_id in extracted_slots for slot_id in slot_ids),
    )
    if all(slot_id in extracted_slots for slot_id in slot_ids):
        record(
            "declared_slots_exact_match",
            lambda: all(
                _canonical_slot(edit["slot"])
                == _canonical_slot(extracted_slots[str(edit["slot"].get("slot_id"))])
                for edit in edits
            ),
        )
    record(
        "exact_simultaneous_source_span_patch",
        lambda: _apply_declared_patches(parent_code, edits) == child_code,
    )
    record(
        "independent_shape_diff_matches_declared_slots",
        lambda: bool(_annotate_independent_changes(edits, changed_occurrences)),
    )
    record("static_identity_gate", lambda: bool(static_gate(parent_code, child_code, entry_point)))
    record(
        "child_reference_hash",
        lambda: _sha256_bytes(child_code.encode("utf-8")) == decision.get("child_reference_sha256"),
    )
    record(
        "child_normalized_ast_hash",
        lambda: _normalized_ast_sha256(child_code) == decision.get("child_normalized_ast_sha256"),
    )
    record(
        "child_uuid",
        lambda: _nested(child, "extra_info.uuid") == decision.get("child_uuid"),
    )
    record(
        "direct_parent_uuid",
        lambda: _nested(child, "extra_info.v4.parent_uuid") == decision.get("parent_uuid"),
    )
    if target is None:
        checks["target_record_match"] = False
        errors.append("target_record_missing")
    else:
        record(
            "target_record_match",
            lambda: int(target["target_input_bytes"]) == int(decision["target_input_bytes"])
            and str(target["parent_reference_sha256"]) == str(decision["parent_reference_sha256"]),
        )
    record(
        "parent_fake_passed",
        lambda: _nested(decision, "parent_fake_gate.status") == "passed",
    )
    record("child_fake_passed", lambda: _nested(decision, "fake_gate.status") == "passed")
    return checks, errors


def _variant_cardinality_contract(
    manifest: Mapping[str, Any],
    selected_uuids: Sequence[str],
    target_rows: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify legacy-v3 two-variant or v4 one-variant artifact cardinality."""

    contract_version = str(manifest.get("contract_version", "missing"))
    legacy_v3 = contract_version == "shape_affine_solver_v3"
    expected_per_parent = 2 if legacy_v3 else 1
    selected = set(selected_uuids)
    errors: list[str] = []

    targets_by_parent: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in target_rows:
        parent_uuid = row.get("parent_uuid")
        if not isinstance(parent_uuid, str) or not parent_uuid:
            errors.append(f"target_missing_parent_uuid:{parent_uuid!r}")
            continue
        targets_by_parent[parent_uuid].append(row)
    decisions_by_parent: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for decision in decisions:
        parent_uuid = decision.get("parent_uuid")
        if not isinstance(parent_uuid, str) or not parent_uuid:
            errors.append(f"decision_missing_parent_uuid:{parent_uuid!r}")
            continue
        decisions_by_parent[parent_uuid].append(decision)

    for label, grouped in (("target", targets_by_parent), ("decision", decisions_by_parent)):
        unknown = sorted(set(grouped) - selected)
        missing = sorted(selected - set(grouped))
        if unknown:
            errors.append(f"{label}_unknown_parents:{unknown[:20]}")
        if missing:
            errors.append(f"{label}_missing_parents:{missing[:20]}")
        for parent_uuid in sorted(selected & set(grouped)):
            rows = grouped[parent_uuid]
            if len(rows) != expected_per_parent:
                errors.append(
                    f"{label}_cardinality:{parent_uuid}:{len(rows)}:"
                    f"expected={expected_per_parent}"
                )
            variants = [row.get("variant") for row in rows]
            if any(variant not in {"medium", "large"} for variant in variants):
                errors.append(f"{label}_invalid_variant:{parent_uuid}:{variants}")
            if len(variants) != len(set(variants)):
                errors.append(f"{label}_duplicate_variant:{parent_uuid}:{variants}")

    for parent_uuid in sorted(selected & set(targets_by_parent) & set(decisions_by_parent)):
        target_variants = sorted(str(row.get("variant")) for row in targets_by_parent[parent_uuid])
        decision_variants = sorted(
            str(row.get("variant")) for row in decisions_by_parent[parent_uuid]
        )
        if target_variants != decision_variants:
            errors.append(
                f"target_decision_variant_mismatch:{parent_uuid}:"
                f"{target_variants}:{decision_variants}"
            )

    accepted_by_parent: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for decision in decisions:
        if decision.get("accepted") is True and isinstance(decision.get("parent_uuid"), str):
            accepted_by_parent[str(decision["parent_uuid"])].append(decision)
    accepted_limit = 2 if legacy_v3 else 1
    for parent_uuid, rows in sorted(accepted_by_parent.items()):
        if len(rows) > accepted_limit:
            errors.append(
                f"accepted_child_cardinality:{parent_uuid}:{len(rows)}:"
                f"limit={accepted_limit}"
            )
        child_uuids = [row.get("child_uuid") for row in rows]
        if any(not isinstance(value, str) or not value for value in child_uuids):
            errors.append(f"accepted_child_missing_uuid:{parent_uuid}:{child_uuids}")
        if len(child_uuids) != len(set(child_uuids)):
            errors.append(f"accepted_child_duplicate_uuid:{parent_uuid}:{child_uuids}")

    return {
        "contract_version": contract_version,
        "mode": "legacy_two_variants_per_parent" if legacy_v3 else "one_variant_per_parent",
        "expected_targets_per_parent": expected_per_parent,
        "expected_decisions_per_parent": expected_per_parent,
        "accepted_children_per_parent_limit": accepted_limit,
        "selected_parents": len(selected_uuids),
        "target_rows": len(target_rows),
        "decision_rows": len(decisions),
        "accepted_parents": len(accepted_by_parent),
        "target_variant_counts": _counter(
            collections.Counter(str(row.get("variant")) for row in target_rows)
        ),
        "decision_variant_counts": _counter(
            collections.Counter(str(row.get("variant")) for row in decisions)
        ),
        "errors": errors[:200],
        "error_count": len(errors),
        "passed": not errors,
    }


def _shape_quality_gate(
    *,
    scope: str,
    child_uuids: Sequence[str],
    changed_rows: Sequence[Mapping[str, Any]],
    available: bool,
    evidence_complete: bool,
    evidence_errors: Sequence[str] = (),
    logical_slot_counts: Mapping[str, int],
    required_logical_slots: int | None,
    logical_slot_range: tuple[int, int] | None,
) -> dict[str, Any]:
    """Apply exact occurrence-weighted distribution gates without float boundaries."""

    child_ids = list(child_uuids)
    unique_children = set(child_ids)
    errors = list(evidence_errors)
    if len(child_ids) != len(unique_children):
        errors.append("quality_scope_contains_duplicate_child_uuid")
    rows_by_child: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    seen_keys: set[tuple[str, int, int]] = set()
    for row in changed_rows:
        child_uuid = row.get("child_uuid")
        if not isinstance(child_uuid, str) or child_uuid not in unique_children:
            continue
        factory_index = row.get("factory_index")
        axis = row.get("axis")
        new_value = row.get("new_value")
        if type(factory_index) is not int or type(axis) is not int:
            errors.append(
                f"quality_invalid_occurrence_key:{child_uuid}:"
                f"{factory_index!r}:{axis!r}"
            )
            continue
        if type(new_value) is not int or new_value <= 0:
            errors.append(
                f"quality_invalid_new_dimension:{child_uuid}:{factory_index}:{axis}:"
                f"{new_value!r}"
            )
            continue
        key = (child_uuid, factory_index, axis)
        if key in seen_keys:
            errors.append(
                f"quality_duplicate_changed_occurrence:{child_uuid}:{factory_index}:{axis}"
            )
            continue
        seen_keys.add(key)
        rows_by_child[child_uuid].append(row)
    missing_children = sorted(unique_children - set(rows_by_child))
    if missing_children:
        errors.append(f"quality_children_without_changed_occurrences:{missing_children[:20]}")

    missing_slot_counts = sorted(unique_children - set(logical_slot_counts))
    if missing_slot_counts:
        errors.append(f"quality_children_without_logical_slot_count:{missing_slot_counts[:20]}")
    invalid_slot_counts = {
        child_uuid: logical_slot_counts.get(child_uuid)
        for child_uuid in sorted(unique_children)
        if child_uuid in logical_slot_counts
        and (
            type(logical_slot_counts[child_uuid]) is not int
            or logical_slot_counts[child_uuid] <= 0
        )
    }
    if invalid_slot_counts:
        errors.append(f"quality_invalid_logical_slot_counts:{invalid_slot_counts}")
    valid_slot_counts = [
        int(logical_slot_counts[child_uuid])
        for child_uuid in unique_children
        if type(logical_slot_counts.get(child_uuid)) is int
        and logical_slot_counts[child_uuid] > 0
    ]
    exact_required_slot_children = (
        sum(value == required_logical_slots for value in valid_slot_counts)
        if required_logical_slots is not None
        else None
    )
    within_declared_range_children = (
        sum(
            logical_slot_range[0] <= value <= logical_slot_range[1]
            for value in valid_slot_counts
        )
        if logical_slot_range is not None
        else None
    )
    logical_slot_cardinality_passed = bool(
        not missing_slot_counts
        and not invalid_slot_counts
        and (
            required_logical_slots is None
            or exact_required_slot_children == len(unique_children)
        )
        and (
            logical_slot_range is None
            or within_declared_range_children == len(unique_children)
        )
    )

    occurrence_count = sum(len(rows_by_child[child_uuid]) for child_uuid in unique_children)
    power_of_two_count = sum(
        1
        for child_uuid in unique_children
        for row in rows_by_child[child_uuid]
        if int(row["new_value"]) & (int(row["new_value"]) - 1) == 0
    )
    single_changed_children = sum(
        len(rows_by_child[child_uuid]) == 1 for child_uuid in unique_children
    )
    child_count = len(unique_children)
    power_lower_passed = (
        occurrence_count > 0
        and POWER_OF_TWO_MIN_DENOMINATOR * power_of_two_count
        >= POWER_OF_TWO_MIN_NUMERATOR * occurrence_count
    )
    power_upper_passed = (
        occurrence_count > 0
        and POWER_OF_TWO_MAX_DENOMINATOR * power_of_two_count
        <= POWER_OF_TWO_MAX_NUMERATOR * occurrence_count
    )
    single_upper_passed = (
        child_count > 0
        and SINGLE_CHANGED_MAX_DENOMINATOR * single_changed_children
        <= SINGLE_CHANGED_MAX_NUMERATOR * child_count
    )
    complete = bool(available and evidence_complete and not errors)
    passed = bool(
        complete
        and logical_slot_cardinality_passed
        and power_lower_passed
        and power_upper_passed
        and single_upper_passed
    )
    return {
        "scope": scope,
        "available": bool(available),
        "evidence_complete": bool(evidence_complete),
        "measurement_contract": {
            "child": "one child UUID in this scope",
            "changed_occurrence": (
                "one unique (child_uuid, direct_factory_index, axis) whose independently "
                "resolved parent and child values differ"
            ),
            "power_of_two": "positive integer n with n & (n - 1) == 0; includes 1 = 2^0",
            "boundaries": "inclusive and evaluated with exact integer cross-products",
        },
        "children": child_count,
        "logical_slots": {
            "measurement": "one solver-declared logical shape slot; shared direct-factory occurrences count once",
            "required_per_child": required_logical_slots,
            "declared_range": list(logical_slot_range) if logical_slot_range else None,
            "observed_children": len(valid_slot_counts),
            "count_distribution": _counter(collections.Counter(valid_slot_counts)),
            "exact_required_children": exact_required_slot_children,
            "within_declared_range_children": within_declared_range_children,
            "passed": logical_slot_cardinality_passed,
        },
        "changed_occurrences": occurrence_count,
        "power_of_two": {
            "count": power_of_two_count,
            "denominator": occurrence_count,
            "fraction": _fraction(power_of_two_count, occurrence_count),
            "minimum_inclusive": 0.30,
            "maximum_inclusive": 0.50,
            "minimum_passed": power_lower_passed,
            "maximum_passed": power_upper_passed,
            "passed": power_lower_passed and power_upper_passed,
        },
        "single_changed_dimension": {
            "children": single_changed_children,
            "denominator": child_count,
            "fraction": _fraction(single_changed_children, child_count),
            "maximum_inclusive": 0.10,
            "passed": single_upper_passed,
        },
        "missing_changed_occurrence_children": missing_children[:100],
        "errors": errors[:200],
        "error_count": len(errors),
        "passed": passed,
    }


def _decision_failure_category(decision: Mapping[str, Any]) -> str:
    parent_status = str(_nested(decision, "parent_fake_gate.status", "missing"))
    if parent_status == "unsupported":
        return "parent_fake_unsupported"
    if parent_status == "failed":
        return "parent_fake_failed"
    if parent_status == "timeout":
        return "parent_fake_timeout"
    reason = str(decision.get("reason", "missing_reason"))
    if reason == "no_balance_feasible_affine_slot_solution_in_variant_band":
        affine_slot_count = int(decision.get("affine_slot_count", 0))
        if affine_slot_count == 0:
            return "no_affine_safe_slot"
        if affine_slot_count == 1:
            return "only_one_affine_slot_no_balanced_solution"
        return "multi_slot_required_for_balanced_target"
    if reason == "no_affine_slot_solution_in_variant_band":
        return "no_solution_in_variant_band"
    if reason == "no_strict_two_slot_exactly_one_power_solution":
        structural_pair_count = decision.get("structural_bilinear_pair_count")
        if type(structural_pair_count) is int:
            if structural_pair_count == 0:
                return "no_structural_pair"
            return "no_power_of_two_feasible_solution"
    if reason == "no_variable_multislot_product_solution":
        profile_counts = decision.get(
            "product_profile_count_by_logical_slot_count", {}
        )
        profile_count = (
            sum(int(value) for value in profile_counts.values())
            if isinstance(profile_counts, Mapping)
            else 0
        )
        return (
            "no_exact_product_profile"
            if profile_count == 0
            else "no_numeric_product_solution"
        )
    if int(decision.get("slot_count", 0)) == 0:
        return "no_patchable_shape_slot"
    if int(decision.get("affine_slot_count", 0)) == 0:
        return "no_affine_shape_slot"
    attempts = decision.get("attempts", [])
    if isinstance(attempts, list):
        child_statuses = {
            str(_nested(attempt, "fake_gate.status"))
            for attempt in attempts
            if isinstance(attempt, Mapping) and _nested(attempt, "fake_gate.status") is not None
        }
        if "unsupported" in child_statuses:
            return "child_candidate_fake_unsupported"
        if "failed" in child_statuses:
            return "child_candidate_fake_failed"
        if "timeout" in child_statuses:
            return "child_candidate_fake_timeout"
        if attempts:
            return "child_candidate_static_or_target_failed"
    return "other_solver_failure"


def _group_coverage(
    selected_uuids: Sequence[str],
    parent_meta: Mapping[str, Mapping[str, str]],
    accepted: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, Any]:
    selected_by_group: dict[str, set[str]] = collections.defaultdict(set)
    for uuid in selected_uuids:
        selected_by_group[str(parent_meta[uuid][key])].add(uuid)
    accepted_children: collections.Counter[str] = collections.Counter()
    accepted_parents: dict[str, set[str]] = collections.defaultdict(set)
    for decision in accepted:
        uuid = str(decision["parent_uuid"])
        group = str(parent_meta[uuid][key])
        accepted_children[group] += 1
        accepted_parents[group].add(uuid)
    return {
        group: {
            "selected_parents": len(parents),
            "accepted_parents": len(accepted_parents[group]),
            "accepted_children": accepted_children[group],
            "parent_coverage": _fraction(len(accepted_parents[group]), len(parents)),
        }
        for group, parents in sorted(selected_by_group.items())
    }


def _load_ai_baseline(
    ai_run_dir: Path,
    selected_uuids: Sequence[str],
) -> dict[str, Any] | None:
    summary_path = ai_run_dir / "analysis" / "summary.json"
    bias_path = ai_run_dir / "analysis" / "shape_bias_audit.json"
    manifest_path = ai_run_dir / "static" / "manifest.json"
    if not summary_path.is_file() or not bias_path.is_file() or not manifest_path.is_file():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    bias = json.loads(bias_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    changed = _nested(bias, "accepted.changed_values", {})
    changed_per = _nested(bias, "accepted.changed_dimensions_per_proposal.distribution", {})
    full_accepted_count = int(_nested(summary, "static.accepted_children", 0))
    decisions = [
        decision
        for decision in manifest.get("decisions", [])
        if isinstance(decision, Mapping) and decision.get("parent_uuid") is not None
    ]
    attempted_parents = {str(decision["parent_uuid"]) for decision in decisions}
    accepted = [decision for decision in decisions if bool(decision.get("accepted"))]
    accepted_parents = {str(decision["parent_uuid"]) for decision in accepted}
    target_errors = [float(decision["target_relative_error"]) for decision in accepted]
    input_scales = [float(decision["input_scale"]) for decision in accepted]
    selected = set(selected_uuids)
    return {
        "run_dir": str(ai_run_dir.resolve()),
        "coverage_scope": "within the AI run's own attempted-parent denominator",
        "distribution_scope": "full AI run accepted distribution",
        "attempted_parents": len(attempted_parents),
        "attempted_variant_decisions": len(decisions),
        "attempted_parents_overlapping_solver_selection": len(
            attempted_parents & selected
        ),
        "accepted_children": len(accepted),
        "accepted_parents": len(accepted_parents),
        "parent_coverage": _fraction(len(accepted_parents), len(attempted_parents)),
        "target_relative_error_p50": _percentile(target_errors, 0.50),
        "target_relative_error_p90": _percentile(target_errors, 0.90),
        "input_scale_p50": _percentile(input_scales, 0.50),
        "power_of_two_fraction": changed.get("power_of_two_fraction"),
        "multiple_8_fraction": _nested(changed, "multiple_fractions.8"),
        "multiple_32_fraction": _nested(changed, "multiple_fractions.32"),
        "effective_values": changed.get("effective_values"),
        "hhi": changed.get("hhi"),
        "top5_fraction": changed.get("top5_fraction"),
        "single_changed_dimension_fraction": _fraction(
            int(changed_per.get("1", 0)), full_accepted_count
        ),
    }


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _load_jsonl_files(paths: Sequence[Path]) -> list[tuple[Path, int, dict[str, Any]]]:
    records: list[tuple[Path, int, dict[str, Any]]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: runtime record must be an object")
                records.append((path, line_number, value))
    return records


def _runtime_passed(record: Mapping[str, Any]) -> bool:
    passed = record.get("passed")
    status = record.get("status")
    if type(passed) is not bool:
        raise ValueError(f"runtime record has non-boolean passed field: {passed!r}")
    if (status == "passed") != passed:
        raise ValueError(f"runtime status/passed mismatch: {status!r}:{passed!r}")
    return passed


def _failure_reasons(record: Mapping[str, Any]) -> list[str]:
    reasons = record.get("failure_reasons")
    if isinstance(reasons, list):
        values = [str(value).replace("\n", " ")[:500] for value in reasons if str(value)]
        if values:
            return values
    reason = record.get("reason")
    if reason not in (None, ""):
        return [str(reason).replace("\n", " ")[:500]]
    error_type = record.get("error_type")
    error = record.get("error")
    if error_type not in (None, "") or error not in (None, ""):
        return [f"{error_type or 'error'}:{error or ''}".replace("\n", " ")[:500]]
    return [f"status:{record.get('status', 'missing')}"]


def _kernelgym_runtime_error_name(record: Mapping[str, Any]) -> str | None:
    metadata = record.get("kernelgym_metadata")
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("runtime_error_name")
    if not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _kernelgym_runtime_error_text(record: Mapping[str, Any]) -> str:
    metadata = record.get("kernelgym_metadata")
    if not isinstance(metadata, Mapping):
        return ""
    value = metadata.get("runtime_error")
    return value.lower() if isinstance(value, str) else ""


def _is_oom(record: Mapping[str, Any]) -> bool:
    status = str(record.get("status", "")).lower()
    if status in {"cuda_out_of_memory", "device_memory_limit_exceeded"}:
        return True
    runtime_error_name = (_kernelgym_runtime_error_name(record) or "").lower()
    runtime_error_class = "".join(
        character for character in runtime_error_name.rsplit(".", 1)[-1]
        if character.isalnum()
    )
    if runtime_error_class in {"outofmemoryerror", "cudaoutofmemoryerror"}:
        return True
    text = " ".join(
        [
            status,
            _kernelgym_runtime_error_text(record),
            *[value.lower() for value in _failure_reasons(record)],
        ]
    )
    return "out of memory" in text or "cuda_out_of_memory" in text


def _runtime_child_groups(
    accepted: Sequence[Mapping[str, Any]],
    records: Mapping[str, Mapping[str, Any]],
    parent_meta: Mapping[str, Mapping[str, str]],
    key: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for decision in accepted:
        parent_uuid = str(decision["parent_uuid"])
        group = (
            str(decision.get("variant", "unknown"))
            if key == "variant"
            else str(parent_meta[parent_uuid][key])
        )
        grouped[group].append(decision)
    result: dict[str, dict[str, Any]] = {}
    for group, decisions in sorted(grouped.items()):
        child_uuids = [str(decision["child_uuid"]) for decision in decisions]
        observed = [records[uuid] for uuid in child_uuids if uuid in records]
        passed = sum(_runtime_passed(record) for record in observed)
        result[group] = {
            "expected_children": len(child_uuids),
            "observed_children": len(observed),
            "passed_children": passed,
            "failed_children": len(observed) - passed,
            "missing_children": len(child_uuids) - len(observed),
            "pass_fraction_of_expected": _fraction(passed, len(child_uuids)),
            "pass_fraction_of_observed": _fraction(passed, len(observed)),
        }
    return result


def _accepted_by_child(
    accepted: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for decision in accepted:
        child_uuid = decision.get("child_uuid")
        if not isinstance(child_uuid, str) or not child_uuid:
            raise ValueError(f"accepted decision has invalid child UUID: {child_uuid!r}")
        if child_uuid in result:
            raise ValueError(f"duplicate accepted child decision: {child_uuid}")
        result[child_uuid] = decision
    return result


def _reference_runtime_summary(
    run_dir: Path,
    accepted: Sequence[Mapping[str, Any]],
    selected_by_uuid: Mapping[str, Mapping[str, Any]],
    child_by_uuid: Mapping[str, Mapping[str, Any]],
    parent_meta: Mapping[str, Mapping[str, str]],
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    paths = sorted((run_dir / "h20" / "reference").glob("shard-*.jsonl"))
    accepted_by_child = _accepted_by_child(accepted)
    expected_parents = {str(decision["parent_uuid"]) for decision in accepted}
    expected_children = set(accepted_by_child)
    if not paths:
        return {
            "available": False,
            "expected_parents": len(expected_parents),
            "expected_children": len(expected_children),
            "expected_rows": len(expected_parents) + len(expected_children),
        }, {}

    paired_path = run_dir / "static" / "paired.parquet"
    if not paired_path.is_file():
        raise FileNotFoundError(paired_path)
    logical_rows: dict[str, Mapping[str, Any]] = {}
    for uuid in expected_parents:
        if uuid not in selected_by_uuid:
            raise ValueError(f"accepted parent missing from selected parquet: {uuid}")
        logical_rows[uuid] = selected_by_uuid[uuid]
    for uuid in expected_children:
        if uuid not in child_by_uuid:
            raise ValueError(f"accepted child missing from children parquet: {uuid}")
        logical_rows[uuid] = child_by_uuid[uuid]
    paired_rows = pq.read_table(paired_path).to_pylist()
    expected_rows: dict[str, dict[str, Any]] = {}
    for row_index, row in enumerate(paired_rows):
        uuid = _nested(row, "extra_info.uuid")
        code = _nested(row, "reward_model.ground_truth")
        if not isinstance(uuid, str) or not uuid or not isinstance(code, str):
            raise ValueError(f"invalid paired row identity at index {row_index}")
        if uuid in expected_rows:
            raise ValueError(f"duplicate paired UUID: {uuid}")
        logical_code = _nested(logical_rows.get(uuid), "reward_model.ground_truth")
        if not isinstance(logical_code, str) or code != logical_code:
            raise ValueError(f"paired reference differs from source artifact: {uuid}")
        expected_rows[uuid] = {
            "row_index": row_index,
            "reference_sha256": _sha256_bytes(code.encode("utf-8")),
        }
    expected_uuids = expected_parents | expected_children
    if set(expected_rows) != expected_uuids:
        missing = sorted(expected_uuids - set(expected_rows))
        extra = sorted(set(expected_rows) - expected_uuids)
        raise ValueError(f"paired UUID coverage mismatch: missing={missing[:20]}; extra={extra[:20]}")

    paired_sha256 = _sha256_file(paired_path)
    records: dict[str, Mapping[str, Any]] = {}
    contract_fingerprints: set[str] = set()
    validator_hashes: set[str] = set()
    launcher_hashes: set[str] = set()
    for path, line_number, record in _load_jsonl_files(paths):
        context = f"{path}:{line_number}"
        uuid = record.get("uuid")
        if not isinstance(uuid, str) or not uuid:
            raise ValueError(f"{context}: missing reference UUID")
        if uuid in records:
            raise ValueError(f"{context}: duplicate reference UUID: {uuid}")
        if uuid not in expected_rows:
            raise ValueError(f"{context}: unexpected reference UUID: {uuid}")
        expected = expected_rows[uuid]
        if record.get("contract_version") != REFERENCE_CONTRACT_VERSION:
            raise ValueError(f"{context}: unexpected reference contract version")
        if record.get("source_sha256") != paired_sha256:
            raise ValueError(f"{context}: reference source hash mismatch")
        if _nested(record, "contract_payload.source_sha256") != paired_sha256:
            raise ValueError(f"{context}: reference contract source hash mismatch")
        if record.get("reference_sha256") != expected["reference_sha256"]:
            raise ValueError(f"{context}: reference code hash mismatch for {uuid}")
        if record.get("row_index") != expected["row_index"]:
            raise ValueError(f"{context}: reference row index mismatch for {uuid}")
        expected_key = f"{expected['row_index']}:{expected['reference_sha256']}"
        if record.get("row_key") != expected_key:
            raise ValueError(f"{context}: reference row key mismatch for {uuid}")
        _runtime_passed(record)
        records[uuid] = record
        for field, destination in (
            ("contract_fingerprint", contract_fingerprints),
            ("validator_source_sha256", validator_hashes),
            ("launcher_source_sha256", launcher_hashes),
        ):
            value = record.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{context}: missing {field}")
            destination.add(value)
    if len(contract_fingerprints) > 1:
        raise ValueError(f"reference contract fingerprints differ: {sorted(contract_fingerprints)}")
    if len(validator_hashes) > 1:
        raise ValueError(f"reference validator hashes differ: {sorted(validator_hashes)}")
    if len(launcher_hashes) > 1:
        raise ValueError(f"reference launcher hashes differ: {sorted(launcher_hashes)}")
    launcher_source_sha256 = next(iter(launcher_hashes), None)
    launcher_archive_path = run_dir / "h20" / "reference" / "launcher_source.sh"
    launcher_archive: dict[str, Any] = {
        "available": launcher_archive_path.is_file(),
        "path": str(launcher_archive_path.resolve()),
        "sha256": None,
        "matches_records": None,
    }
    if launcher_archive_path.is_file():
        launcher_archive_sha256 = _sha256_file(launcher_archive_path)
        launcher_archive["sha256"] = launcher_archive_sha256
        launcher_archive["matches_records"] = (
            launcher_archive_sha256 == launcher_source_sha256
        )
        if launcher_archive_sha256 != launcher_source_sha256:
            raise ValueError(
                "reference launcher source archive hash differs from runtime records:"
                f" {launcher_archive_sha256}:{launcher_source_sha256}"
            )

    observed_parent_uuids = expected_parents & set(records)
    observed_child_uuids = expected_children & set(records)
    passed_parents = sum(_runtime_passed(records[uuid]) for uuid in observed_parent_uuids)
    passed_children = sum(_runtime_passed(records[uuid]) for uuid in observed_child_uuids)
    causal = collections.Counter()
    incomplete = collections.Counter()
    for decision in accepted:
        parent_uuid = str(decision["parent_uuid"])
        child_uuid = str(decision["child_uuid"])
        parent_record = records.get(parent_uuid)
        child_record = records.get(child_uuid)
        if parent_record is None or child_record is None:
            if parent_record is None and child_record is None:
                incomplete["missing_both"] += 1
            elif parent_record is None:
                incomplete["missing_parent"] += 1
            else:
                incomplete["missing_child"] += 1
            continue
        parent_passed = _runtime_passed(parent_record)
        child_passed = _runtime_passed(child_record)
        causal[
            "both_pass"
            if parent_passed and child_passed
            else "parent_only"
            if parent_passed
            else "child_only"
            if child_passed
            else "neither"
        ] += 1

    status_counts = collections.Counter(str(record.get("status", "missing")) for record in records.values())
    kernelgym_runtime_errors = collections.Counter(
        error_name
        for record in records.values()
        if (error_name := _kernelgym_runtime_error_name(record)) is not None
    )
    failure_reasons: collections.Counter[str] = collections.Counter()
    for record in records.values():
        if not _runtime_passed(record):
            failure_reasons.update(_failure_reasons(record))
    missing_uuids = sorted(expected_uuids - set(records))
    return {
        "available": True,
        "files": {str(path.relative_to(run_dir)): _sha256_file(path) for path in paths},
        "paired_sha256": paired_sha256,
        "contract_fingerprint": next(iter(contract_fingerprints), None),
        "validator_source_sha256": next(iter(validator_hashes), None),
        "launcher_source_sha256": launcher_source_sha256,
        "launcher_source_archive": launcher_archive,
        "expected_rows": len(expected_uuids),
        "observed_rows": len(records),
        "coverage_complete": not missing_uuids,
        "missing_uuids": missing_uuids,
        "parents": {
            "expected": len(expected_parents),
            "observed": len(observed_parent_uuids),
            "passed": passed_parents,
            "failed": len(observed_parent_uuids) - passed_parents,
            "missing": len(expected_parents - set(records)),
        },
        "children": {
            "expected": len(expected_children),
            "observed": len(observed_child_uuids),
            "passed": passed_children,
            "failed": len(observed_child_uuids) - passed_children,
            "missing": len(expected_children - set(records)),
        },
        "parent_child_causal": {
            "total_pairs": len(accepted),
            "complete_pairs": sum(causal.values()),
            "incomplete_pairs": sum(incomplete.values()),
            "buckets": {key: causal[key] for key in ("both_pass", "parent_only", "child_only", "neither")},
            "incomplete": _counter(incomplete),
        },
        "child_pass_by_variant": _runtime_child_groups(accepted, records, parent_meta, "variant"),
        "child_pass_by_source_family": _runtime_child_groups(accepted, records, parent_meta, "source_family"),
        "child_pass_by_operator_family": _runtime_child_groups(accepted, records, parent_meta, "operator_family"),
        "status_counts": _counter(status_counts),
        "kernelgym_runtime_error_counts": _counter(kernelgym_runtime_errors),
        "failure_reason_counts": _counter(failure_reasons),
        "timeout_count": status_counts["timeout"],
        "oom_count": sum(_is_oom(record) for record in records.values()),
        "duration_seconds": _distribution(
            [float(record["duration_seconds"]) for record in records.values() if isinstance(record.get("duration_seconds"), (int, float))]
        ),
    }, records


def _decision_slot_ids(decision: Mapping[str, Any]) -> list[str]:
    try:
        return [str(edit["slot"]["slot_id"]) for edit in _solver_slot_edits(decision)]
    except (KeyError, TypeError, ValueError):
        pass
    attempts = decision.get("attempts")
    if isinstance(attempts, list):
        accepted_attempts = [
            attempt for attempt in attempts
            if isinstance(attempt, Mapping) and attempt.get("accepted") is True
        ]
        if len(accepted_attempts) == 1:
            value = _nested(accepted_attempts[0], "slot.slot_id")
            return [value] if isinstance(value, str) else []
    return []


def _region_slot_binding_matches(
    record: Mapping[str, Any],
    decision: Mapping[str, Any],
) -> bool:
    expected = _decision_slot_ids(decision)
    if not expected:
        return False
    if len(expected) == 1:
        return record.get("slot_id") == expected[0]
    return record.get("slot_id") is None and record.get("slot_ids") == expected


def _reference_both_pass_children(
    accepted: Sequence[Mapping[str, Any]],
    reference_records: Mapping[str, Mapping[str, Any]],
) -> tuple[set[str], dict[str, list[str]]]:
    expected: set[str] = set()
    incomplete: dict[str, list[str]] = collections.defaultdict(list)
    for decision in accepted:
        parent_uuid = str(decision["parent_uuid"])
        child_uuid = str(decision["child_uuid"])
        parent_record = reference_records.get(parent_uuid)
        child_record = reference_records.get(child_uuid)
        if parent_record is None or child_record is None:
            reason = (
                "missing_both"
                if parent_record is None and child_record is None
                else "missing_parent"
                if parent_record is None
                else "missing_child"
            )
            incomplete[reason].append(child_uuid)
            continue
        if _runtime_passed(parent_record) and _runtime_passed(child_record):
            expected.add(child_uuid)
    return expected, dict(incomplete)


def _write_region_reference_allowlist(
    run_dir: Path,
    accepted: Sequence[Mapping[str, Any]],
    reference_summary: Mapping[str, Any],
    reference_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if reference_summary.get("available") is not True:
        return {"available": False, "count": 0}
    expected, incomplete = _reference_both_pass_children(accepted, reference_records)
    ordered = [
        str(decision["child_uuid"])
        for decision in accepted
        if str(decision["child_uuid"]) in expected
    ]
    content = "".join(f"{child_uuid}\n" for child_uuid in ordered)
    path = run_dir / "analysis" / "region_reference_both_pass_uuids.txt"
    _atomic_write(path, content)
    return {
        "available": True,
        "derivation": "accepted decision order filtered by parent and child reference pass",
        "derivation_complete": bool(reference_summary.get("coverage_complete"))
        and not incomplete,
        "path": str(path.resolve()),
        "sha256": _sha256_bytes(content.encode("utf-8")),
        "count": len(ordered),
    }


def _region_bucket(status: Any) -> str:
    value = str(status)
    return value if value in {"passed", "rejected", "unsupported", "timeout"} else "failed"


def _region_runtime_summary(
    run_dir: Path,
    accepted: Sequence[Mapping[str, Any]],
    artifact_hashes: Mapping[str, str],
    reference_summary: Mapping[str, Any],
    reference_records: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Mapping[str, Any]]]:
    paths = sorted((run_dir / "h20" / "region").glob("shard-*.jsonl"))
    accepted_by_child = _accepted_by_child(accepted)
    reference_expected, reference_incomplete = _reference_both_pass_children(
        accepted,
        reference_records,
    )
    reference_incomplete_count = sum(len(values) for values in reference_incomplete.values())
    allowlist_derivation_complete = bool(reference_summary.get("available")) and bool(
        reference_summary.get("coverage_complete")
    ) and not reference_incomplete
    if not paths:
        return {
            "available": False,
            "evidence_present": False,
            "contract_compatible": None,
            "expected_contract_version": REGION_CONTRACT_VERSION,
            "expected_perturbation_contract_version": PERTURBATION_CONTRACT_VERSION,
            "expected_scope": "reference_parent_and_child_both_pass",
            "allowlist_derivation_complete": allowlist_derivation_complete,
            "expected_children": len(reference_expected),
            "excluded_static_children": (
                len(accepted_by_child)
                - len(reference_expected)
                - reference_incomplete_count
            ),
            "unclassified_static_children": reference_incomplete_count,
        }, {}

    loaded_records = _load_jsonl_files(paths)
    observed_contracts = collections.Counter(
        str(record.get("contract_version")) for _, _, record in loaded_records
    )
    observed_perturbation_contracts = collections.Counter(
        str(record.get("perturbation_contract_version"))
        for _, _, record in loaded_records
    )
    observed_contract_set = set(observed_contracts)
    if observed_contracts and observed_contract_set == {LEGACY_REGION_CONTRACT_VERSION}:
        return {
            "available": False,
            "evidence_present": True,
            "contract_compatible": False,
            "incompatibility_reason": (
                f"legacy_region_contract:{LEGACY_REGION_CONTRACT_VERSION}; "
                f"expected:{REGION_CONTRACT_VERSION}+{PERTURBATION_CONTRACT_VERSION}"
            ),
            "expected_contract_version": REGION_CONTRACT_VERSION,
            "expected_perturbation_contract_version": PERTURBATION_CONTRACT_VERSION,
            "observed_contract_versions": _counter(observed_contracts),
            "observed_perturbation_contract_versions": _counter(
                observed_perturbation_contracts
            ),
            "files": {str(path.relative_to(run_dir)): _sha256_file(path) for path in paths},
            "expected_children": len(accepted_by_child),
            "observed_children": len(loaded_records),
        }, {}
    supported_contracts = {REGION_CONTRACT_VERSION, PREVIOUS_REGION_CONTRACT_VERSION}
    if len(observed_contract_set) != 1 or not observed_contract_set <= supported_contracts:
        raise ValueError(
            f"region contract versions must contain exactly one supported version: "
            f"{sorted(observed_contract_set)}"
        )
    contract_version = next(iter(observed_contract_set))
    current_contract = contract_version == REGION_CONTRACT_VERSION
    if current_contract:
        expected_child_uuids = reference_expected
        expected_scope = "reference_parent_and_child_both_pass"
        derivation_complete = allowlist_derivation_complete
    else:
        expected_child_uuids = set(accepted_by_child)
        expected_scope = "legacy_all_static_accepted_children"
        derivation_complete = True

    records: dict[str, Mapping[str, Any]] = {}
    validator_hashes: set[str] = set()
    launcher_hashes: set[str] = set()
    binding_contract_versions: set[str] = set()
    binding_hashes: set[str] = set()
    for path, line_number, record in loaded_records:
        context = f"{path}:{line_number}"
        child_uuid = record.get("child_uuid")
        if not isinstance(child_uuid, str) or not child_uuid:
            raise ValueError(f"{context}: missing region child UUID")
        if child_uuid in records:
            raise ValueError(f"{context}: duplicate region child UUID: {child_uuid}")
        decision = accepted_by_child.get(child_uuid)
        if decision is None:
            raise ValueError(f"{context}: unexpected region child UUID: {child_uuid}")
        if child_uuid not in expected_child_uuids:
            raise ValueError(
                f"{context}: region child UUID is outside {expected_scope}: {child_uuid}"
            )
        if record.get("contract_version") != contract_version:
            raise ValueError(f"{context}: unexpected region contract version")
        if record.get("perturbation_contract_version") != PERTURBATION_CONTRACT_VERSION:
            raise ValueError(f"{context}: unexpected region perturbation contract version")
        if record.get("parent_uuid") != decision.get("parent_uuid"):
            raise ValueError(f"{context}: region parent UUID mismatch for {child_uuid}")
        if record.get("variant") != decision.get("variant"):
            raise ValueError(f"{context}: region variant mismatch for {child_uuid}")
        if not _region_slot_binding_matches(record, decision):
            raise ValueError(f"{context}: region slot binding mismatch for {child_uuid}")
        for field, expected in (
            ("selected_sha256", artifact_hashes["selected"]),
            ("children_sha256", artifact_hashes["children"]),
            ("manifest_sha256", artifact_hashes["manifest"]),
        ):
            if record.get(field) != expected:
                raise ValueError(f"{context}: region {field} mismatch")
        _runtime_passed(record)
        validator_hash = record.get("validator_source_sha256")
        if not isinstance(validator_hash, str) or not validator_hash:
            raise ValueError(f"{context}: missing validator_source_sha256")
        validator_hashes.add(validator_hash)
        if current_contract:
            binding_contract_version = record.get(
                "validation_binding_contract_version"
            )
            supported_binding_contracts = {
                REGION_RUN_BINDING_CONTRACT_VERSION,
                PREVIOUS_REGION_RUN_BINDING_CONTRACT_VERSION,
            }
            if binding_contract_version not in supported_binding_contracts:
                raise ValueError(f"{context}: unexpected validation binding contract")
            binding_contract_versions.add(str(binding_contract_version))
            validation_config = record.get("validation_config")
            if not isinstance(validation_config, Mapping):
                raise ValueError(f"{context}: missing validation_config")
            binding = {
                "validation_binding_contract_version": binding_contract_version,
                "contract_version": contract_version,
                "perturbation_contract_version": PERTURBATION_CONTRACT_VERSION,
                "validator_source_sha256": validator_hash,
                "selected_sha256": artifact_hashes["selected"],
                "children_sha256": artifact_hashes["children"],
                "manifest_sha256": artifact_hashes["manifest"],
                "validation_config": dict(validation_config),
            }
            if binding_contract_version == REGION_RUN_BINDING_CONTRACT_VERSION:
                launcher_hash = record.get("launcher_source_sha256")
                if not isinstance(launcher_hash, str) or not launcher_hash:
                    raise ValueError(f"{context}: missing launcher_source_sha256")
                launcher_hashes.add(launcher_hash)
                binding["launcher_source_sha256"] = launcher_hash
            binding_hash = record.get("validation_binding_sha256")
            if not isinstance(binding_hash, str) or not binding_hash:
                raise ValueError(f"{context}: missing validation_binding_sha256")
            if binding_hash != _canonical_mapping_sha256(binding):
                raise ValueError(f"{context}: validation binding hash mismatch")
            binding_hashes.add(binding_hash)
        records[child_uuid] = record
    if len(validator_hashes) > 1:
        raise ValueError(f"region validator hashes differ: {sorted(validator_hashes)}")
    if len(launcher_hashes) > 1:
        raise ValueError(f"region launcher hashes differ: {sorted(launcher_hashes)}")
    if len(binding_contract_versions) > 1:
        raise ValueError(
            "region validation binding contracts differ:"
            f" {sorted(binding_contract_versions)}"
        )
    if len(binding_hashes) > 1:
        raise ValueError(f"region validation bindings differ: {sorted(binding_hashes)}")

    launcher_source_sha256 = next(iter(launcher_hashes), None)
    launcher_archive_path = run_dir / "h20" / "region" / "launcher_source.sh"
    launcher_archive: dict[str, Any] = {
        "available": launcher_archive_path.is_file(),
        "path": str(launcher_archive_path.resolve()),
        "sha256": None,
        "matches_records": None,
    }
    if launcher_archive_path.is_file():
        launcher_archive_sha256 = _sha256_file(launcher_archive_path)
        launcher_archive["sha256"] = launcher_archive_sha256
        launcher_archive["matches_records"] = (
            launcher_source_sha256 is not None
            and launcher_archive_sha256 == launcher_source_sha256
        )
        if (
            launcher_source_sha256 is not None
            and launcher_archive_sha256 != launcher_source_sha256
        ):
            raise ValueError(
                "region launcher source archive hash differs from runtime records:"
                f" {launcher_archive_sha256}:{launcher_source_sha256}"
            )

    raw_status = collections.Counter(str(record.get("status", "missing")) for record in records.values())
    kernelgym_runtime_errors = collections.Counter(
        error_name
        for record in records.values()
        if (error_name := _kernelgym_runtime_error_name(record)) is not None
    )
    buckets = collections.Counter(_region_bucket(record.get("status")) for record in records.values())
    reasons: collections.Counter[str] = collections.Counter()
    for record in records.values():
        if not _runtime_passed(record):
            reasons.update(_failure_reasons(record))
    by_variant: dict[str, dict[str, Any]] = {}
    expected_decisions = [
        decision
        for decision in accepted
        if str(decision["child_uuid"]) in expected_child_uuids
    ]
    variants = sorted(
        {str(decision.get("variant", "unknown")) for decision in expected_decisions}
    )
    for variant in variants:
        expected = [
            str(decision["child_uuid"])
            for decision in expected_decisions
            if str(decision.get("variant", "unknown")) == variant
        ]
        observed = [records[uuid] for uuid in expected if uuid in records]
        variant_buckets = collections.Counter(_region_bucket(record.get("status")) for record in observed)
        by_variant[variant] = {
            "expected_children": len(expected),
            "observed_children": len(observed),
            "missing_children": len(expected) - len(observed),
            **{key: variant_buckets[key] for key in ("passed", "rejected", "unsupported", "failed", "timeout")},
        }
    missing = sorted(expected_child_uuids - set(records))
    return {
        "available": True,
        "evidence_present": True,
        "contract_compatible": True,
        "contract_version": contract_version,
        "perturbation_contract_version": PERTURBATION_CONTRACT_VERSION,
        "validation_binding_contract_version": (
            next(iter(binding_contract_versions), None) if current_contract else None
        ),
        "validation_binding_sha256": next(iter(binding_hashes), None),
        "files": {str(path.relative_to(run_dir)): _sha256_file(path) for path in paths},
        "validator_source_sha256": next(iter(validator_hashes), None),
        "launcher_source_sha256": launcher_source_sha256,
        "launcher_source_archive": launcher_archive,
        "expected_scope": expected_scope,
        "allowlist_derivation_complete": derivation_complete,
        "allowlist_derivation_incomplete": {
            key: sorted(values)[:100] for key, values in sorted(reference_incomplete.items())
        }
        if current_contract
        else {},
        "static_accepted_children": len(accepted_by_child),
        "expected_children": len(expected_child_uuids),
        "excluded_static_children": (
            len(accepted_by_child)
            - len(expected_child_uuids)
            - (reference_incomplete_count if current_contract else 0)
        ),
        "unclassified_static_children": (
            reference_incomplete_count if current_contract else 0
        ),
        "observed_children": len(records),
        "coverage_complete": derivation_complete and not missing,
        "missing_child_uuids": missing,
        "status_buckets": {
            key: buckets[key] for key in ("passed", "rejected", "unsupported", "failed", "timeout")
        },
        "raw_status_counts": _counter(raw_status),
        "kernelgym_runtime_error_counts": _counter(kernelgym_runtime_errors),
        "reason_counts": _counter(reasons),
        "oom_count": sum(_is_oom(record) for record in records.values()),
        "by_variant": by_variant,
        "duration_seconds": _distribution(
            [float(record["duration_seconds"]) for record in records.values() if isinstance(record.get("duration_seconds"), (int, float))]
        ),
    }, records


def _eligible_runtime_summary(
    accepted: Sequence[Mapping[str, Any]],
    selected_parent_count: int,
    parent_meta: Mapping[str, Mapping[str, str]],
    reference_summary: Mapping[str, Any],
    reference_records: Mapping[str, Mapping[str, Any]],
    region_summary: Mapping[str, Any],
    region_records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if not reference_summary.get("available") or not region_summary.get("available"):
        return {
            "available": False,
            "contract": "child reference passed AND changed-region liveness passed",
            "expected_children": len(accepted),
        }
    eligible_decisions = [
        decision
        for decision in accepted
        if (reference := reference_records.get(str(decision["child_uuid"]))) is not None
        and _runtime_passed(reference)
        and (region := region_records.get(str(decision["child_uuid"]))) is not None
        and _runtime_passed(region)
    ]
    eligible_children = {str(decision["child_uuid"]) for decision in eligible_decisions}
    covered_parents = {str(decision["parent_uuid"]) for decision in eligible_decisions}
    variants_by_parent: dict[str, set[str]] = collections.defaultdict(set)
    for decision in eligible_decisions:
        variants_by_parent[str(decision["parent_uuid"])].add(str(decision.get("variant")))
    both_variants = {
        parent_uuid
        for parent_uuid, variants in variants_by_parent.items()
        if {"medium", "large"}.issubset(variants)
    }

    def eligible_groups(key: str) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
        for decision in accepted:
            parent_uuid = str(decision["parent_uuid"])
            group = (
                str(decision.get("variant", "unknown"))
                if key == "variant"
                else str(parent_meta[parent_uuid][key])
            )
            groups[group].append(decision)
        return {
            group: {
                "expected_children": len(items),
                "eligible_children": sum(str(item["child_uuid"]) in eligible_children for item in items),
                "eligible_fraction": _fraction(
                    sum(str(item["child_uuid"]) in eligible_children for item in items), len(items)
                ),
            }
            for group, items in sorted(groups.items())
        }

    return {
        "available": True,
        "contract": "child reference passed AND changed-region liveness passed",
        "coverage_complete": bool(reference_summary.get("coverage_complete"))
        and bool(region_summary.get("coverage_complete")),
        "expected_children": len(accepted),
        "eligible_children": len(eligible_children),
        "eligible_child_fraction": _fraction(len(eligible_children), len(accepted)),
        "eligible_child_uuids": sorted(eligible_children),
        "covered_parents": len(covered_parents),
        "selected_parent_denominator": selected_parent_count,
        "parent_coverage": _fraction(len(covered_parents), selected_parent_count),
        "covered_parent_uuids": sorted(covered_parents),
        "both_variants_parents": len(both_variants),
        "both_variants_parent_fraction": _fraction(len(both_variants), selected_parent_count),
        "both_variants_parent_uuids": sorted(both_variants),
        "by_variant": eligible_groups("variant"),
        "by_source_family": eligible_groups("source_family"),
        "by_operator_family": eligible_groups("operator_family"),
    }


def _runtime_eligible_bias_profile(
    runtime_eligible: Mapping[str, Any],
    accepted: Sequence[Mapping[str, Any]],
    changed_rows: Sequence[Mapping[str, Any]],
    dimension_balance_audits: Sequence[Mapping[str, Any]],
    ratio_limit: int | float | None,
) -> dict[str, Any]:
    """Profile the runtime-surviving subset without changing eligibility."""

    contract = str(
        runtime_eligible.get(
            "contract",
            "child reference passed AND changed-region liveness passed",
        )
    )
    if runtime_eligible.get("available") is not True:
        return {
            "available": False,
            "contract": contract,
            "coverage_complete": runtime_eligible.get("coverage_complete"),
            "static_children": len(accepted),
            "eligible_children": None,
            "profiles_computed": False,
            "reason": "runtime eligibility is unavailable; no empty-set bias profile was emitted",
        }

    raw_eligible_uuids = runtime_eligible.get("eligible_child_uuids")
    if not isinstance(raw_eligible_uuids, list) or any(
        not isinstance(value, str) or not value for value in raw_eligible_uuids
    ):
        raise ValueError("runtime eligible child UUIDs are missing or invalid")
    eligible_uuids = set(raw_eligible_uuids)
    if len(eligible_uuids) != len(raw_eligible_uuids):
        raise ValueError("runtime eligible child UUIDs contain duplicates")
    declared_eligible_children = runtime_eligible.get("eligible_children")
    if type(declared_eligible_children) is not int or declared_eligible_children != len(
        eligible_uuids
    ):
        raise ValueError("runtime eligible child count does not match UUID coverage")

    accepted_by_child = _accepted_by_child(accepted)
    unknown_eligible = sorted(eligible_uuids - set(accepted_by_child))
    if unknown_eligible:
        raise ValueError(f"runtime eligible UUIDs are not accepted children: {unknown_eligible[:20]}")
    eligible_decisions = [
        decision
        for decision in accepted
        if str(decision["child_uuid"]) in eligible_uuids
    ]

    rows_by_child: dict[str, list[Mapping[str, Any]]] = collections.defaultdict(list)
    for row in changed_rows:
        child_uuid = row.get("child_uuid")
        if isinstance(child_uuid, str) and child_uuid in eligible_uuids:
            rows_by_child[child_uuid].append(row)
    missing_changed_rows = sorted(eligible_uuids - set(rows_by_child))
    if missing_changed_rows:
        raise ValueError(
            f"runtime eligible children have no changed-dimension rows: {missing_changed_rows[:20]}"
        )
    for decision in eligible_decisions:
        child_uuid = str(decision["child_uuid"])
        expected_occurrences = len(
            _declared_occurrences(_solver_slot_edits(decision))
        )
        if expected_occurrences != len(rows_by_child[child_uuid]):
            raise ValueError(f"runtime eligible occurrence coverage mismatch: {child_uuid}")

    eligible_rows = [row for row in changed_rows if row.get("child_uuid") in eligible_uuids]
    new_values = [int(row["new_value"]) for row in eligible_rows]
    occurrence_counts = [len(rows_by_child[str(decision["child_uuid"])]) for decision in eligible_decisions]
    unique_axes_per_child = [
        len(
            {
                int(row["axis_from_right"])
                for row in rows_by_child[str(decision["child_uuid"])]
            }
        )
        for decision in eligible_decisions
    ]
    children_with_leading = sum(
        any(bool(row["leading"]) for row in rows_by_child[str(decision["child_uuid"])])
        for decision in eligible_decisions
    )
    leading_occurrences = sum(bool(row["leading"]) for row in eligible_rows)
    axis_counts = collections.Counter(int(row["axis"]) for row in eligible_rows)
    axis_from_right_counts = collections.Counter(
        int(row["axis_from_right"]) for row in eligible_rows
    )
    dominant_axis_count = max(axis_from_right_counts.values(), default=0)
    slot_kind_counts: collections.Counter[str] = collections.Counter()
    seen_slots: set[tuple[str, int]] = set()
    for row in eligible_rows:
        key = (str(row["child_uuid"]), int(row["slot_index"]))
        if key in seen_slots:
            continue
        seen_slots.add(key)
        slot_kind_counts[str(row["slot_kind"])] += 1

    audits_by_child: dict[str, Mapping[str, Any]] = {}
    for audit in dimension_balance_audits:
        child_uuid = audit.get("child_uuid")
        if not isinstance(child_uuid, str):
            raise ValueError("dimension-balance audit has an invalid child UUID")
        if child_uuid in audits_by_child:
            raise ValueError(f"duplicate dimension-balance child audit: {child_uuid}")
        audits_by_child[child_uuid] = audit
    missing_audits = sorted(eligible_uuids - set(audits_by_child))
    if missing_audits:
        raise ValueError(
            f"runtime eligible children have no dimension-balance audit: {missing_audits[:20]}"
        )
    eligible_audits = [
        audits_by_child[str(decision["child_uuid"])] for decision in eligible_decisions
    ]

    return {
        "available": True,
        "contract": contract,
        "coverage_complete": bool(runtime_eligible.get("coverage_complete")),
        "profiles_computed": True,
        "measurement_contract": {
            "sample": "one runtime-eligible child",
            "dimension_occurrence": (
                "one independently resolved parent/child direct-factory (factory_index, axis) diff; "
                "shared slots contribute once per actual factory occurrence"
            ),
            "capacity": "aggregate input bytes after the shape edit, once per eligible child",
            "ratio": "independent child-source reparse; rank-1 factories have no balance ratio",
            "retention": "eligible children divided by statically accepted children in each group",
        },
        "static_children": len(accepted),
        "eligible_children": len(eligible_decisions),
        "retention_fraction": _fraction(len(eligible_decisions), len(accepted)),
        "changed_dimensions": _value_profile(new_values, len(eligible_decisions)),
        "changed_dimension_value_partitions": _partitioned_value_profiles(
            new_values, len(eligible_decisions)
        ),
        "slot_kind_counts": _counter(slot_kind_counts),
        "position": {
            "eligible_children": len(eligible_decisions),
            "changed_occurrences": len(eligible_rows),
            "changed_occurrences_per_child": {
                **_distribution(occurrence_counts),
                "counts": _counter(collections.Counter(occurrence_counts)),
            },
            "single_occurrence_child_fraction": _fraction(
                sum(value == 1 for value in occurrence_counts), len(occurrence_counts)
            ),
            "unique_axis_from_right_per_child": {
                **_distribution(unique_axes_per_child),
                "counts": _counter(collections.Counter(unique_axes_per_child)),
            },
            "single_axis_from_right_child_fraction": _fraction(
                sum(value == 1 for value in unique_axes_per_child),
                len(unique_axes_per_child),
            ),
            "children_with_leading_fraction": _fraction(
                children_with_leading, len(eligible_decisions)
            ),
            "leading_occurrence_fraction": _fraction(
                leading_occurrences, len(eligible_rows)
            ),
            "axis_counts": _counter(axis_counts),
            "axis_from_right_counts": _counter(axis_from_right_counts),
            "dominant_axis_from_right_fraction": _fraction(
                dominant_axis_count, len(eligible_rows)
            ),
        },
        "capacity": _capacity_profile(
            [int(decision["input_bytes_after"]) for decision in eligible_decisions]
        ),
        "dimension_balance": _dimension_balance_summary(eligible_audits, ratio_limit),
        "retention": {
            "by_variant": runtime_eligible.get("by_variant", {}),
            "by_source_family": runtime_eligible.get("by_source_family", {}),
            "by_operator_family": runtime_eligible.get("by_operator_family", {}),
        },
    }


def _format_float(value: Any, digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def _format_percent(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.2%}"


def _format_mib(value: Any) -> str:
    return "n/a" if value is None else _format_float(float(value) / MIB, 2)


def _dimension_balance_markdown(balance: Mapping[str, Any], heading: str) -> list[str]:
    ratios = balance["ratios"]
    boundary = balance["boundary_counts"]
    violations = balance["violations"]
    lines = [
        "",
        heading,
        "",
        (
            "This independently reparses every final direct-input factory in each accepted "
            "child, plus every actual changed axis; stored solver balance evidence is not used."
        ),
        "",
    ]
    if balance["manifest_contract_present"]:
        lines.extend(
            [
                (
                    f"The manifest limit is **{balance['ratio_limit']}:1** and is enforced "
                    "fail-closed by this verifier."
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                (
                    "This pre-contract manifest declares no ratio limit. Ratios are reported "
                    "for observation only and do not change verification status."
                ),
                "",
            ]
        )
    lines.extend(
        [
            (
                f"Resolved children: **{balance['resolved_children']}/{balance['accepted_children']}**; "
                f"resolved occurrences: **{balance['resolved_occurrences']}/"
                f"{balance['declared_occurrences']}**."
            ),
            "",
            "| Ratio scope | Samples | p50 | p90 | p99 | max |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for label, key in (
        ("Child maximum", "per_child_max"),
        ("Rank>=2 occurrence", "per_occurrence"),
        ("Rank>=2 final direct factory", "per_final_direct_factory"),
        ("Rank>=2 affected factory", "per_affected_factory"),
    ):
        row = ratios[key]
        lines.append(
            f"| {label} | {row['count']} | {_format_float(row['p50'], 4)} | "
            f"{_format_float(row['p90'], 4)} | {_format_float(row['p99'], 4)} | "
            f"{_format_float(row['max'], 4)} |"
        )
    lines.extend(
        [
            "",
            "| Contract outcome | Children | Occurrences | Final direct factories |",
            "| --- | ---: | ---: | ---: |",
            (
                f"| Exactly at boundary | {boundary['children'] if boundary['children'] is not None else 'n/a'} | "
                f"{boundary['occurrences'] if boundary['occurrences'] is not None else 'n/a'} | "
                f"{boundary['factories'] if boundary['factories'] is not None else 'n/a'} |"
            ),
            (
                f"| Violations | {violations['children'] if violations['children'] is not None else 'n/a'} | "
                f"{violations['occurrences'] if violations['occurrences'] is not None else 'n/a'} | "
                f"{violations['factories'] if violations['factories'] is not None else 'n/a'} |"
            ),
            "",
        ]
    )
    return lines


def _quality_markdown(quality: Mapping[str, Any]) -> list[str]:
    variant = quality["variant_cardinality"]
    lines = [
        "",
        "## Shape distribution quality gates",
        "",
        (
            f"Variant cardinality mode: `{variant['mode']}`; "
            f"passed: **{variant['passed']}**."
        ),
        "",
        "| Scope | Available | Complete | Children | Logical slots | Changed occurrences | Power-of-two | Single changed dimension | Passed |",
        "| --- | :---: | :---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for key in ("static", "runtime"):
        gate = quality[key]
        logical = gate["logical_slots"]
        required = logical["required_per_child"]
        logical_label = (
            f"{logical['exact_required_children']}/{gate['children']} @ {required}"
            if required is not None
            else json.dumps(logical["count_distribution"], sort_keys=True)
        )
        lines.append(
            f"| {key} | {gate['available']} | {gate['evidence_complete']} | "
            f"{gate['children']} | {logical_label} | {gate['changed_occurrences']} | "
            f"{_format_percent(gate['power_of_two']['fraction'])} | "
            f"{_format_percent(gate['single_changed_dimension']['fraction'])} | "
            f"{gate['passed']} |"
        )
    exact_slot_requirements = {
        gate["logical_slots"]["required_per_child"]
        for gate in (quality["static"], quality["runtime"])
        if gate["logical_slots"]["required_per_child"] is not None
    }
    range_requirements = {
        tuple(gate["logical_slots"]["declared_range"])
        for gate in (quality["static"], quality["runtime"])
        if gate["logical_slots"]["declared_range"] is not None
    }
    slot_contract_sentence = (
        "This manifest requires exactly "
        + ", ".join(str(value) for value in sorted(exact_slot_requirements))
        + " logical slots per child."
        if exact_slot_requirements
        else (
            "This manifest requires logical-slot counts in the inclusive range "
            + ", ".join(
                f"[{minimum}, {maximum}]"
                for minimum, maximum in sorted(range_requirements)
            )
            + "."
            if range_requirements
            else "This manifest accepts any positive logical-slot count per child."
        )
    )
    lines.extend(
        [
            "",
            "Logical slots count one solver-declared shape variable once even when it must be "
            f"changed consistently in several input tensors. {slot_contract_sentence}",
            "",
            "Power-of-two is inclusive 30%–50% over independently changed dimension "
            "occurrences. Single changed dimension is inclusive at most 10% over children. "
            "Missing or incomplete evidence fails closed.",
            "",
        ]
    )
    if variant.get("errors"):
        lines.append(
            "Variant cardinality errors: `"
            + json.dumps(variant["errors"][:20], sort_keys=True)
            + "`"
        )
        lines.append("")
    for key in ("static", "runtime"):
        gate = quality[key]
        if gate.get("errors"):
            lines.append(
                f"{key.capitalize()} quality evidence errors: `"
                + json.dumps(gate["errors"][:20], sort_keys=True)
                + "`"
            )
            lines.append("")
    return lines


def _runtime_markdown(runtime: Mapping[str, Any]) -> list[str]:
    reference = runtime["reference"]
    region = runtime["region"]
    eligible = runtime["eligible"]
    if (
        not reference.get("available")
        and not region.get("available")
        and not region.get("evidence_present")
    ):
        return []
    lines = ["", "## H20 runtime validation", ""]
    if eligible.get("available"):
        lines.extend(
            [
                (
                    f"The intersection contains **{eligible['eligible_children']}/"
                    f"{eligible['expected_children']} eligible children** and covers "
                    f"**{eligible['covered_parents']}/{eligible['selected_parent_denominator']} parents** "
                    f"({_format_percent(eligible['parent_coverage'])}); "
                    f"{eligible['both_variants_parents']} parents retain both variants."
                ),
                "",
                (
                    "Eligibility is exactly child-reference pass AND changed-region-liveness pass. "
                    "The unchanged-parent reference result is reported separately for causal diagnosis."
                ),
                "",
                "| Variant | Static children | Eligible children | Eligible fraction |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for variant, row in eligible["by_variant"].items():
            lines.append(
                f"| {variant} | {row['expected_children']} | {row['eligible_children']} | "
                f"{_format_percent(row['eligible_fraction'])} |"
            )
        lines.append("")
        if not eligible["coverage_complete"]:
            lines.extend(
                [
                    "Runtime evidence is incomplete, so these are partial eligible counts.",
                    "",
                ]
            )
    elif region.get("contract_compatible") is False:
        lines.extend(
            [
                (
                    "Changed-region evidence is present but contract-incompatible with the current "
                    "validator, so it is excluded from runtime eligibility."
                ),
                "",
            ]
        )
    else:
        lines.extend(
            [
                "Only one runtime evidence family is present; final eligibility cannot yet be computed.",
                "",
            ]
        )

    if reference.get("available"):
        launcher_archive = reference.get("launcher_source_archive", {})
        lines.extend(
            [
                "### Paired reference gate",
                "",
                (
                    "Launcher source archive: "
                    f"`{launcher_archive.get('path', 'unavailable')}`; "
                    f"SHA-256 `{launcher_archive.get('sha256')}`; "
                    f"matches all runtime records: "
                    f"`{str(launcher_archive.get('matches_records')).lower()}`."
                ),
                "",
                "| Row type | Expected | Observed | Passed | Failed | Missing |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for label, key in (("Unchanged parent", "parents"), ("Solver child", "children")):
            row = reference[key]
            lines.append(
                f"| {label} | {row['expected']} | {row['observed']} | {row['passed']} | "
                f"{row['failed']} | {row['missing']} |"
            )
        causal = reference["parent_child_causal"]
        lines.extend(
            [
                "",
                "Parent-child outcomes over pairs with both records present:",
                "",
                "| Both pass | Parent only | Child only | Neither | Incomplete pairs |",
                "| ---: | ---: | ---: | ---: | ---: |",
                (
                    f"| {causal['buckets']['both_pass']} | {causal['buckets']['parent_only']} | "
                    f"{causal['buckets']['child_only']} | {causal['buckets']['neither']} | "
                    f"{causal['incomplete_pairs']} |"
                ),
                "",
                "Reference status and resource failures:",
                "",
                "| Status | Rows |",
                "| --- | ---: |",
            ]
        )
        for status, count in reference["status_counts"].items():
            lines.append(f"| {status} | {count} |")
        lines.extend(
            [
                "",
                f"Timeout rows: **{reference['timeout_count']}**; OOM rows: **{reference['oom_count']}**.",
                "",
            ]
        )
        if reference["failure_reason_counts"]:
            lines.extend(["Top reference failure reasons:", "", "| Reason | Occurrences |", "| --- | ---: |"])
            for reason, count in sorted(
                reference["failure_reason_counts"].items(), key=lambda item: (-item[1], item[0])
            )[:20]:
                lines.append(f"| `{reason.replace('|', '&#124;')}` | {count} |")
            lines.append("")

        for title, key, label in (
            ("Reference child pass by variant", "child_pass_by_variant", "Variant"),
            ("Reference child pass by source", "child_pass_by_source_family", "Source"),
            ("Reference child pass by operator", "child_pass_by_operator_family", "Operator"),
        ):
            lines.extend(
                [
                    title + ":",
                    "",
                    f"| {label} | Expected | Observed | Passed | Failed | Missing | Pass / expected |",
                    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for group, row in reference[key].items():
                lines.append(
                    f"| {group} | {row['expected_children']} | {row['observed_children']} | "
                    f"{row['passed_children']} | {row['failed_children']} | {row['missing_children']} | "
                    f"{_format_percent(row['pass_fraction_of_expected'])} |"
                )
            lines.append("")

    allowlist = reference.get("region_allowlist", {})
    if allowlist.get("available"):
        lines.extend(
            [
                "Region launch allowlist derived from paired reference evidence:",
                "",
                (
                    f"**{allowlist['count']} child UUIDs**; derivation complete: "
                    f"`{str(bool(allowlist['derivation_complete'])).lower()}`; "
                    f"SHA-256 `{allowlist['sha256']}`."
                ),
                "",
                f"Artifact: `{allowlist['path']}`",
                "",
            ]
        )

    if region.get("available"):
        buckets = region["status_buckets"]
        lines.extend(["### Changed-region liveness gate", ""])
        region_launcher_archive = region.get("launcher_source_archive", {})
        if region.get("launcher_source_sha256") is not None:
            lines.extend(
                [
                    (
                        "Launcher source archive: "
                        f"`{region_launcher_archive.get('path', 'unavailable')}`; "
                        f"SHA-256 `{region_launcher_archive.get('sha256')}`; "
                        f"matches all runtime records: "
                        f"`{str(region_launcher_archive.get('matches_records')).lower()}`."
                    ),
                    "",
                ]
            )
        lines.extend(
            [
                (
                    f"Observed {region['observed_children']}/{region['expected_children']} children; "
                    f"missing {len(region['missing_child_uuids'])}. "
                    f"Expected scope: `{region['expected_scope']}`; "
                    f"excluded static children: {region['excluded_static_children']}; "
                    f"unclassified static children: "
                    f"{region['unclassified_static_children']}."
                ),
                "",
                "| Scope | Passed | Rejected | Unsupported | Failed | Timeout | Missing |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                (
                    f"| All | {buckets['passed']} | {buckets['rejected']} | {buckets['unsupported']} | "
                    f"{buckets['failed']} | {buckets['timeout']} | {len(region['missing_child_uuids'])} |"
                ),
            ]
        )
        for variant, row in region["by_variant"].items():
            lines.append(
                f"| {variant} | {row['passed']} | {row['rejected']} | {row['unsupported']} | "
                f"{row['failed']} | {row['timeout']} | {row['missing_children']} |"
            )
        if region["reason_counts"]:
            lines.extend(["", "Top region failure reasons:", "", "| Reason | Occurrences |", "| --- | ---: |"])
            for reason, count in sorted(
                region["reason_counts"].items(), key=lambda item: (-item[1], item[0])
            )[:20]:
                lines.append(f"| `{reason.replace('|', '&#124;')}` | {count} |")
    elif region.get("contract_compatible") is False:
        lines.extend(
            [
                "### Changed-region liveness gate",
                "",
                f"Contract incompatible: `{region['incompatibility_reason']}`",
                "",
                (
                    f"Observed {region.get('observed_children', 0)} legacy records; expected "
                    f"`{region['expected_contract_version']}` plus perturbation contract "
                    f"`{region['expected_perturbation_contract_version']}`."
                ),
            ]
        )
    return lines


def _summary_markdown(summary: Mapping[str, Any]) -> str:
    coverage = summary["coverage"]
    target = summary["target"]
    verification = summary["verification"]
    lines = [
        f"# Static shape solver: {coverage['selected_parents']}-parent analysis",
        "",
        (
            f"The solver statically/FakeTensor accepted **{coverage['accepted_children']} children** from "
            f"**{coverage['accepted_parents']}/{coverage['selected_parents']} parents** "
            f"({coverage['parent_coverage']:.2%}). Runtime eligibility, when available, is reported below."
        ),
        "",
        "## Coverage",
        "",
        "| Variant | Accepted children | Parent denominator | Coverage |",
        "| --- | ---: | ---: | ---: |",
    ]
    for variant, row in summary["by_variant"].items():
        lines.append(
            f"| {variant} | {row['accepted_children']} | {row['selected_parents']} | "
            f"{row['coverage']:.2%} |"
        )
    lines.extend(_quality_markdown(summary["quality"]))
    lines.extend(_runtime_markdown(summary["runtime"]))
    lines.extend(
        [
            "",
            "## Target fit and scale",
            "",
            "| Metric | p50 | p90 | max |",
            "| --- | ---: | ---: | ---: |",
            (
                "| Absolute relative target error | "
                f"{_format_float(target['relative_error']['p50'], 6)} | "
                f"{_format_float(target['relative_error']['p90'], 6)} | "
                f"{_format_float(target['relative_error']['max'], 6)} |"
            ),
            (
                "| Input scale | "
                f"{_format_float(summary['input_scale']['p50'], 3)}× | "
                f"{_format_float(summary['input_scale']['p90'], 3)}× | "
                f"{_format_float(summary['input_scale']['max'], 3)}× |"
            ),
            "",
            "## Coverage by source",
            "",
            "| Source | Selected | Accepted parents | Accepted children | Parent coverage |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for group, row in summary["by_source_family"].items():
        lines.append(
            f"| {group} | {row['selected_parents']} | {row['accepted_parents']} | "
            f"{row['accepted_children']} | {row['parent_coverage']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Coverage by operator family",
            "",
            "| Operator family | Selected | Accepted parents | Accepted children | Parent coverage |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for group, row in summary["by_operator_family"].items():
        lines.append(
            f"| {group} | {row['selected_parents']} | {row['accepted_parents']} | "
            f"{row['accepted_children']} | {row['parent_coverage']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Failures",
            "",
            "| Category | Variant decisions | Parents |",
            "| --- | ---: | ---: |",
        ]
    )
    for category, row in summary["failures"]["decision_categories"].items():
        lines.append(f"| {category} | {row['decisions']} | {row['parents']} |")
    lines.extend(
        [
            "",
            "## Independent identity verification",
            "",
            (
                f"Verified {verification['children_checked']} accepted children. "
                f"Invalid children: **{verification['invalid_children']}**."
            ),
            "",
            "The verifier independently re-extracts every declared shape slot, replays all exact "
            "UTF-8 source spans simultaneously, and requires the resulting source to equal the child "
            "byte-for-byte. It independently resolves parent and child direct-factory shapes and "
            "requires the actual changed `(factory, axis)` set to equal the declared occurrence set. "
            "It also requires exact `Model` and `get_init_inputs` AST equality, unchanged input-factory "
            "sequence and ranks, the existing static identity gate, hashes, direct lineage, and parent/child "
            "FakeTensor passes.",
        ]
    )
    lines.extend(_dimension_balance_markdown(summary["dimension_balance"], "## Dimension balance re-audit"))
    comparison = summary.get("ai_comparison")
    if comparison:
        lines.extend(
            [
                "",
                "## Descriptive cross-run context",
                "",
                "The runs use different attempted-parent denominators and generation contracts; "
                "this table is descriptive and is not an acceptance-efficiency comparison.",
                "",
                "| Run | Attempted parents | Variant decisions | Accepted children | Accepted parents | Within-run parent coverage | Target error p50 | Target error p90 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                (
                    f"| Static solver | {coverage['selected_parents']} | {coverage['variant_decisions']} | "
                    f"{coverage['accepted_children']} | {coverage['accepted_parents']} | "
                    f"{coverage['parent_coverage']:.2%} | {_format_float(target['relative_error']['p50'], 6)} | "
                    f"{_format_float(target['relative_error']['p90'], 6)} |"
                ),
                (
                    f"| DeepSeek-V4-Flash | {comparison['attempted_parents']} | "
                    f"{comparison['attempted_variant_decisions']} | {comparison['accepted_children']} | "
                    f"{comparison['accepted_parents']} | {comparison['parent_coverage']:.2%} | "
                    f"{_format_float(comparison['target_relative_error_p50'], 6)} | "
                    f"{_format_float(comparison['target_relative_error_p90'], 6)} |"
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## Evidence",
            "",
            "- `shape_bias_audit.md`: dimension, slot, axis, and capacity bias.",
            "- `changed_slots.tsv`: one row per changed input-dimension occurrence.",
            "- `review_samples.md`: solver/merge-owned accepted diffs bound by the static manifest.",
            "- `stratified_review_samples.md`: analyzer-owned stratified diffs and representative failures.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _bias_markdown(bias: Mapping[str, Any]) -> str:
    values = bias["changed_dimensions"]
    value_partitions = bias["changed_dimension_value_partitions"]
    position = bias["position"]
    capacity = bias["capacity"]
    lines = [
        "# Static shape solver bias audit",
        "",
        "The unit below is one changed input-factory dimension occurrence. A linked slot used by two "
        "input factories therefore contributes two occurrences; accepted children remain the sample unit.",
        "",
        "## New dimension values",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Occurrences | {values['count']} |",
        f"| Unique values | {values['unique_values']} |",
        f"| Power-of-two fraction | {values['power_of_two_fraction']:.2%} |",
        f"| Multiple-of-8 fraction | {values['multiple_fractions']['8']:.2%} |",
        f"| Multiple-of-32 fraction | {values['multiple_fractions']['32']:.2%} |",
        f"| Dimension-anchor fraction | {values['anchor_fraction']:.2%} |",
        f"| Effective values | {values['effective_values']:.2f} |",
        f"| HHI | {values['hhi']:.6f} |",
        f"| Top-5 fraction | {values['top5_fraction']:.2%} |",
        "",
        "Top values:",
        "",
        "| Value | Occurrences | Fraction |",
        "| ---: | ---: | ---: |",
    ]
    for row in values["top_values"][:15]:
        lines.append(f"| {row['value']} | {row['count']} | {row['fraction']:.2%} |")
    lines.extend(
        [
            "",
            "Power-of-two versus non-power-of-two concentration:",
            "",
            "| Partition | Occurrences | Unique | Effective values | Top-1 | Top-5 | Multiple-of-8 | Multiple-of-32 | Anchors |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, label in (
        ("power_of_two", "Power-of-two"),
        ("non_power_of_two", "Non-power-of-two"),
    ):
        profile = value_partitions[key]
        lines.append(
            f"| {label} | {profile['count']} | {profile['unique_values']} | "
            f"{_format_float(profile['effective_values'], 2)} | "
            f"{_format_percent(profile['top1_fraction'])} | "
            f"{_format_percent(profile['top5_fraction'])} | "
            f"{_format_percent(profile['multiple_fractions']['8'])} | "
            f"{_format_percent(profile['multiple_fractions']['32'])} | "
            f"{_format_percent(profile['anchor_fraction'])} |"
        )
    non_power_residues = value_partitions["non_power_of_two"][
        "residue_histograms"
    ]
    lines.extend(
        [
            "",
            "Non-power-of-two minus-one residue diagnostic:",
            "",
            "| Modulus | Observed fraction | Uniform fraction |",
            "| ---: | ---: | ---: |",
        ]
    )
    for modulus in ("8", "16", "32"):
        profile = non_power_residues[modulus]
        lines.append(
            f"| {modulus} | {_format_percent(profile['minus_one_fraction'])} | "
            f"{_format_percent(profile['expected_uniform_fraction'])} |"
        )
    lines.extend(
        [
            "",
            "## Slot and axis concentration",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Accepted children | {position['accepted_children']} |",
            f"| Single changed occurrence | {position['single_occurrence_child_fraction']:.2%} |",
            f"| One axis-from-right per child | {position['single_axis_from_right_child_fraction']:.2%} |",
            f"| Children touching a leading axis | {position['children_with_leading_fraction']:.2%} |",
            f"| Leading occurrence fraction | {position['leading_occurrence_fraction']:.2%} |",
            f"| Dominant axis-from-right fraction | {position['dominant_axis_from_right_fraction']:.2%} |",
            "",
            f"Slot kinds: `{json.dumps(bias['slot_kind_counts'], sort_keys=True)}`",
            "",
            f"Axis-from-right occurrences: `{json.dumps(position['axis_from_right_counts'], sort_keys=True)}`",
        ]
    )
    lines.extend(_dimension_balance_markdown(bias["dimension_balance"], "## Dimension balance re-audit"))
    lines.extend(
        [
            "## Resulting input capacity",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Unique byte sizes | {capacity['unique_byte_sizes']} |",
            f"| Integer-MiB fraction | {capacity['integer_mib_fraction']:.2%} |",
            f"| Multiple-of-32-MiB fraction | {capacity['multiple_32_mib_fraction']:.2%} |",
            f"| Capacity-anchor fraction | {capacity['anchor_fraction']:.2%} |",
            f"| Capacity HHI | {capacity['hhi']:.6f} |",
        ]
    )
    runtime_eligible = bias.get("runtime_eligible", {})
    lines.extend(["", "## Runtime-eligible subset bias", ""])
    if runtime_eligible.get("available") is not True:
        lines.extend(
            [
                (
                    "Runtime eligibility is unavailable, so no empty-set subset profile was emitted. "
                    f"Contract: `{runtime_eligible.get('contract', 'unknown')}`."
                ),
                "",
            ]
        )
    else:
        eligible_values = runtime_eligible["changed_dimensions"]
        eligible_value_partitions = runtime_eligible[
            "changed_dimension_value_partitions"
        ]
        eligible_position = runtime_eligible["position"]
        eligible_capacity = runtime_eligible["capacity"]
        lines.extend(
            [
                (
                    f"The runtime intersection retains **{runtime_eligible['eligible_children']}/"
                    f"{runtime_eligible['static_children']} children** "
                    f"({_format_percent(runtime_eligible['retention_fraction'])})."
                ),
                "",
            ]
        )
        if not runtime_eligible["coverage_complete"]:
            lines.extend(
                [
                    "Runtime evidence is incomplete; the subset statistics below are partial.",
                    "",
                ]
            )
        lines.extend(
            [
                "Dimension-value and axis comparison:",
                "",
                "| Metric | Static accepted | Runtime eligible |",
                "| --- | ---: | ---: |",
                f"| Children | {position['accepted_children']} | {eligible_position['eligible_children']} |",
                f"| Changed occurrences | {values['count']} | {eligible_values['count']} |",
                f"| Unique values | {values['unique_values']} | {eligible_values['unique_values']} |",
                f"| Effective values | {_format_float(values['effective_values'], 2)} | {_format_float(eligible_values['effective_values'], 2)} |",
                f"| Power-of-two fraction | {_format_percent(values['power_of_two_fraction'])} | {_format_percent(eligible_values['power_of_two_fraction'])} |",
                f"| Multiple-of-8 fraction | {_format_percent(values['multiple_fractions']['8'])} | {_format_percent(eligible_values['multiple_fractions']['8'])} |",
                f"| Multiple-of-32 fraction | {_format_percent(values['multiple_fractions']['32'])} | {_format_percent(eligible_values['multiple_fractions']['32'])} |",
                f"| Leading occurrence fraction | {_format_percent(position['leading_occurrence_fraction'])} | {_format_percent(eligible_position['leading_occurrence_fraction'])} |",
                f"| Dominant axis-from-right fraction | {_format_percent(position['dominant_axis_from_right_fraction'])} | {_format_percent(eligible_position['dominant_axis_from_right_fraction'])} |",
                "",
                "Eligible axis-from-right distribution:",
                "",
                "| Axis from right | Occurrences | Fraction |",
                "| ---: | ---: | ---: |",
            ]
        )
        for axis, count in eligible_position["axis_from_right_counts"].items():
            lines.append(
                f"| {axis} | {count} | {_format_percent(_fraction(count, eligible_values['count']))} |"
            )
        lines.extend(
            [
                "",
                "Top eligible dimension values:",
                "",
                "| Value | Occurrences | Fraction |",
                "| ---: | ---: | ---: |",
            ]
        )
        for row in eligible_values["top_values"][:10]:
            lines.append(
                f"| {row['value']} | {row['count']} | {_format_percent(row['fraction'])} |"
            )
        lines.extend(
            [
                "",
                "Runtime-eligible concentration by value partition:",
                "",
                "| Partition | Occurrences | Unique | Effective values | Top-1 | Top-5 | Multiple-of-8 | Multiple-of-32 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for key, label in (
            ("power_of_two", "Power-of-two"),
            ("non_power_of_two", "Non-power-of-two"),
        ):
            profile = eligible_value_partitions[key]
            lines.append(
                f"| {label} | {profile['count']} | {profile['unique_values']} | "
                f"{_format_float(profile['effective_values'], 2)} | "
                f"{_format_percent(profile['top1_fraction'])} | "
                f"{_format_percent(profile['top5_fraction'])} | "
                f"{_format_percent(profile['multiple_fractions']['8'])} | "
                f"{_format_percent(profile['multiple_fractions']['32'])} |"
            )
        static_residues = value_partitions["non_power_of_two"][
            "residue_histograms"
        ]
        eligible_residues = eligible_value_partitions["non_power_of_two"][
            "residue_histograms"
        ]
        lines.extend(
            [
                "",
                "Non-power-of-two minus-one residue comparison:",
                "",
                "| Modulus | Static accepted | Runtime eligible | Uniform |",
                "| ---: | ---: | ---: | ---: |",
            ]
        )
        for modulus in ("8", "16", "32"):
            static_profile = static_residues[modulus]
            eligible_profile = eligible_residues[modulus]
            lines.append(
                f"| {modulus} | "
                f"{_format_percent(static_profile['minus_one_fraction'])} | "
                f"{_format_percent(eligible_profile['minus_one_fraction'])} | "
                f"{_format_percent(static_profile['expected_uniform_fraction'])} |"
            )
        lines.extend(
            [
                "",
                "Capacity comparison (aggregate input bytes per child):",
                "",
                "| Scope | p50 MiB | p90 MiB | max MiB | Integer-MiB fraction | HHI |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                (
                    f"| Static accepted | {_format_mib(capacity['p50'])} | "
                    f"{_format_mib(capacity['p90'])} | "
                    f"{_format_mib(capacity['max'])} | "
                    f"{_format_percent(capacity['integer_mib_fraction'])} | "
                    f"{_format_float(capacity['hhi'], 6)} |"
                ),
                (
                    f"| Runtime eligible | {_format_mib(eligible_capacity['p50'])} | "
                    f"{_format_mib(eligible_capacity['p90'])} | "
                    f"{_format_mib(eligible_capacity['max'])} | "
                    f"{_format_percent(eligible_capacity['integer_mib_fraction'])} | "
                    f"{_format_float(eligible_capacity['hhi'], 6)} |"
                ),
                "",
            ]
        )
        lines.extend(
            _dimension_balance_markdown(
                runtime_eligible["dimension_balance"],
                "### Runtime-eligible dimension balance",
            )
        )
        for title, key, label in (
            ("Variant retention", "by_variant", "Variant"),
            ("Source-family retention", "by_source_family", "Source family"),
            ("Operator-family retention", "by_operator_family", "Operator family"),
        ):
            lines.extend(
                [
                    title + ":",
                    "",
                    f"| {label} | Static children | Eligible children | Retention |",
                    "| --- | ---: | ---: | ---: |",
                ]
            )
            for group, row in runtime_eligible["retention"][key].items():
                lines.append(
                    f"| {group} | {row['expected_children']} | {row['eligible_children']} | "
                    f"{_format_percent(row['eligible_fraction'])} |"
                )
            lines.append("")
    lines.extend(_quality_markdown(bias["quality_gates"]))
    comparison = bias.get("ai_comparison")
    if comparison:
        lines.extend(
            [
                "",
                "## Comparison with full AI accepted distribution",
                "",
                "The solver and AI runs use different generation contracts and accepted-sample "
                "denominators; the table compares observed value distributions only.",
                "",
                "| Metric | Solver | AI |",
                "| --- | ---: | ---: |",
                f"| Power-of-two dimensions | {values['power_of_two_fraction']:.2%} | {comparison['power_of_two_fraction']:.2%} |",
                f"| Multiple-of-8 dimensions | {values['multiple_fractions']['8']:.2%} | {comparison['multiple_8_fraction']:.2%} |",
                f"| Multiple-of-32 dimensions | {values['multiple_fractions']['32']:.2%} | {comparison['multiple_32_fraction']:.2%} |",
                f"| Effective values | {values['effective_values']:.2f} | {comparison['effective_values']:.2f} |",
                f"| HHI | {values['hhi']:.6f} | {comparison['hhi']:.6f} |",
                f"| Top-5 fraction | {values['top5_fraction']:.2%} | {comparison['top5_fraction']:.2%} |",
                (
                    f"| One changed occurrence per child | "
                    f"{position['single_occurrence_child_fraction']:.2%} | "
                    f"{comparison['single_changed_dimension_fraction']:.2%} |"
                ),
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def _review_markdown(
    accepted_samples: Sequence[Mapping[str, Any]],
    failed: Sequence[Mapping[str, Any]],
) -> str:
    selected: dict[str, dict[str, Any]] = {}

    def add(reason: str, candidates: Iterable[Mapping[str, Any]]) -> None:
        for candidate in candidates:
            child_uuid = str(candidate["decision"]["child_uuid"])
            if child_uuid not in selected and len(selected) >= MAX_REVIEW_ACCEPTED:
                return
            item = selected.setdefault(child_uuid, {"sample": candidate, "reasons": []})
            if reason not in item["reasons"]:
                item["reasons"].append(reason)
            return

    for variant in ("medium", "large"):
        items = [item for item in accepted_samples if item["decision"]["variant"] == variant]
        add(f"{variant}:farthest_target", sorted(items, key=lambda item: -item["decision"]["target_relative_error"]))
        add(f"{variant}:closest_target", sorted(items, key=lambda item: item["decision"]["target_relative_error"]))
    add("largest_child", sorted(accepted_samples, key=lambda item: -item["decision"]["input_bytes_after"]))
    add("largest_scale", sorted(accepted_samples, key=lambda item: -item["decision"]["input_scale"]))
    slot_kinds = sorted(
        {
            str(edit["slot"].get("kind"))
            for item in accepted_samples
            for edit in item.get("slot_edits", [])
        }
    )
    for slot_kind in slot_kinds:
        add(
            f"slot_kind:{slot_kind}",
            [
                item
                for item in accepted_samples
                if any(
                    str(edit["slot"].get("kind")) == slot_kind
                    for edit in item.get("slot_edits", [])
                )
            ],
        )
    for source in sorted({str(item["source_family"]) for item in accepted_samples}):
        add(f"source:{source}", [item for item in accepted_samples if item["source_family"] == source])
    add(
        "nonleading_axis",
        [
            item
            for item in accepted_samples
            if any(
                int(occurrence["axis"]) > 0
                for occurrence in item.get("changed_occurrences", [])
            )
        ],
    )
    add(
        "multi_changed_dimension",
        [
            item
            for item in accepted_samples
            if len(item.get("changed_occurrences", [])) > 1
        ],
    )

    lines = [
        "# Static shape solver review samples",
        "",
        "These are stratified static/FakeTensor candidates, not H20-accepted training rows.",
        "",
    ]
    for item in selected.values():
        sample = item["sample"]
        decision = sample["decision"]
        slot_edits = sample.get("slot_edits", [])
        slot_summaries = [
            {
                "slot_id": edit["slot"].get("slot_id"),
                "kind": edit["slot"].get("kind"),
                "old_value": edit["slot"].get("old_value"),
                "new_value": edit["new_value"],
            }
            for edit in slot_edits
        ]
        changed_summaries = [
            {
                "factory_index": occurrence["factory_index"],
                "factory_name": occurrence["factory_name"],
                "axis": occurrence["axis"],
                "rank": occurrence["rank"],
                "old_value": occurrence["old_value"],
                "new_value": occurrence["new_value"],
                "slot_id": occurrence["slot"].get("slot_id"),
            }
            for occurrence in sample.get("changed_occurrences", [])
        ]
        lines.extend(
            [
                f"## {decision['child_uuid']} ({decision['variant']})",
                "",
                f"Selection: `{', '.join(item['reasons'])}`",
                "",
                (
                    f"Parent `{decision['parent_uuid']}`; source `{sample['source_family']}`; "
                    f"operator `{sample['operator_family']}`; "
                    f"slots `{json.dumps(slot_summaries, sort_keys=True)}`."
                ),
                "",
                (
                    f"Input bytes: {decision['input_bytes_before']} → {decision['input_bytes_after']}; "
                    f"target {decision['target_input_bytes']}; relative error "
                    f"{decision['target_relative_error']:.6%}."
                ),
                "",
                (
                    "Independent changed occurrences: `"
                    f"{json.dumps(changed_summaries, sort_keys=True)}`"
                ),
                "",
                "```diff",
            ]
        )
        lines.extend(
            difflib.unified_diff(
                str(sample["parent_code"]).splitlines(),
                str(sample["child_code"]).splitlines(),
                fromfile="parent.py",
                tofile="child.py",
                lineterm="",
                n=3,
            )
        )
        lines.extend(["```", ""])

    lines.extend(["# Representative failures", ""])
    seen_categories: set[str] = set()
    for item in failed:
        category = str(item["category"])
        if category in seen_categories or len(seen_categories) >= MAX_REVIEW_FAILURES:
            continue
        seen_categories.add(category)
        decision = item["decision"]
        lines.extend(
            [
                f"## {category}",
                "",
                f"Parent `{decision.get('parent_uuid')}`, variant `{decision.get('variant')}`.",
                "",
                f"Reason: `{decision.get('reason')}`",
                "",
                f"Parent FakeTensor gate: `{json.dumps(decision.get('parent_fake_gate'), sort_keys=True)}`",
                "",
                f"Slots: `{decision.get('slot_count')}` total, `{decision.get('affine_slot_count')}` affine.",
                "",
            ]
        )
        attempts = decision.get("attempts")
        if attempts:
            lines.extend(["Last candidate attempt:", "", "```json", json.dumps(attempts[-1], indent=2, sort_keys=True)[:8000], "```", ""])
    return "\n".join(lines).rstrip() + "\n"


def analyze(run_dir: Path, ai_run_dir: Path = DEFAULT_AI_RUN_DIR) -> dict[str, Any]:
    selected_path = run_dir / "selected.parquet"
    targets_path = run_dir / "targets.parquet"
    children_path = run_dir / "static" / "children.parquet"
    manifest_path = run_dir / "static" / "manifest.json"
    for path in (selected_path, targets_path, children_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    selected_rows = pq.read_table(selected_path).to_pylist()
    target_rows = pq.read_table(targets_path).to_pylist()
    child_rows = pq.read_table(children_path).to_pylist()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    _verify_static_manifest_artifacts(run_dir, manifest)
    decisions = manifest.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("solver manifest decisions must be a list")
    dimension_balance_ratio_limit = _dimension_balance_ratio_limit(manifest)
    required_logical_slots = _required_logical_slot_count(manifest)
    logical_slot_range = _logical_slot_count_range(manifest)

    selected_by_uuid: dict[str, Mapping[str, Any]] = {}
    selected_uuids: list[str] = []
    parent_meta: dict[str, dict[str, str]] = {}
    for row in selected_rows:
        uuid = str(_nested(row, "extra_info.uuid"))
        if uuid in selected_by_uuid:
            raise ValueError(f"duplicate selected UUID: {uuid}")
        selected_by_uuid[uuid] = row
        selected_uuids.append(uuid)
        parent_meta[uuid] = {
            "source_family": str(_nested(row, "extra_info.v4.source_family", "unknown")),
            "operator_family": _operator_family(_nested(row, "extra_info.ops", "")),
            "operator_bucket": str(_nested(row, "extra_info.v4.operator_bucket", "unknown")),
        }
    target_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in target_rows:
        key = (str(row["parent_uuid"]), str(row["variant"]))
        if key in target_by_key:
            raise ValueError(f"duplicate target key: {key}")
        target_by_key[key] = row
    child_by_uuid: dict[str, Mapping[str, Any]] = {}
    for row in child_rows:
        uuid = str(_nested(row, "extra_info.uuid"))
        if uuid in child_by_uuid:
            raise ValueError(f"duplicate child UUID: {uuid}")
        child_by_uuid[uuid] = row

    accepted = [decision for decision in decisions if bool(decision.get("accepted"))]
    rejected = [decision for decision in decisions if not bool(decision.get("accepted"))]
    logical_slot_counts: dict[str, int] = {}
    for decision in accepted:
        child_uuid = str(decision.get("child_uuid"))
        try:
            logical_slot_counts[child_uuid] = len(_solver_slot_edits(decision))
        except (TypeError, ValueError):
            continue
    accepted_parent_uuids = {str(decision["parent_uuid"]) for decision in accepted}
    variant_cardinality = _variant_cardinality_contract(
        manifest,
        selected_uuids,
        target_rows,
        decisions,
    )
    by_variant = {
        variant: {
            "selected_parents": assigned_parents,
            "accepted_children": sum(str(item.get("variant")) == variant for item in accepted),
            "coverage": _fraction(
                sum(str(item.get("variant")) == variant for item in accepted),
                assigned_parents,
            ) or 0.0,
        }
        for variant in ("medium", "large")
        for assigned_parents in (
            sum(str(row.get("variant")) == variant for row in target_rows),
        )
    }

    verification_errors: dict[str, list[str]] = {}
    check_counts: collections.Counter[str] = collections.Counter()
    slot_cache: dict[str, dict[str, Mapping[str, Any]]] = {}
    accepted_samples: list[dict[str, Any]] = []
    changed_rows: list[dict[str, Any]] = []
    new_values: list[int] = []
    occurrence_counts: list[int] = []
    unique_axes_per_child: list[int] = []
    children_with_leading = 0
    leading_occurrences = 0
    axis_from_right_counts: collections.Counter[int] = collections.Counter()
    slot_kind_counts: collections.Counter[str] = collections.Counter()
    dimension_balance_audits: list[dict[str, Any]] = []
    shape_diff_errors: dict[str, str] = {}

    for decision in accepted:
        parent_uuid = str(decision["parent_uuid"])
        child_uuid = str(decision.get("child_uuid"))
        parent = selected_by_uuid.get(parent_uuid)
        child = child_by_uuid.get(child_uuid)
        if parent is None or child is None:
            try:
                declared_occurrence_count = len(
                    _declared_occurrences(_solver_slot_edits(decision))
                )
            except (TypeError, ValueError):
                declared_occurrence_count = 0
            missing_reason = (
                "accepted_parent_missing" if parent is None else "accepted_child_missing"
            )
            verification_errors[child_uuid] = [
                missing_reason
            ]
            shape_diff_errors[child_uuid] = missing_reason
            dimension_balance_audits.append(
                {
                    "child_uuid": child_uuid,
                    "resolved": False,
                    "reason": "accepted_child_missing" if child is None else "accepted_parent_missing",
                    "declared_occurrence_count": declared_occurrence_count,
                    "occurrences": [],
                    "factories": [],
                    "child_max_ratio": None,
                    "boundary": None,
                    "violation": None,
                }
            )
            continue
        parent_code = str(_nested(parent, "reward_model.ground_truth"))
        child_code = str(_nested(child, "reward_model.ground_truth"))
        if parent_uuid not in slot_cache:
            try:
                slot_cache[parent_uuid] = {
                    slot.slot_id: slot.as_dict() for slot in _shape_slots(parent_code)
                }
            except (SyntaxError, TypeError, ValueError) as exc:
                slot_cache[parent_uuid] = {}
                verification_errors.setdefault(child_uuid, []).append(
                    f"slot_reextraction_failed:{type(exc).__name__}:{exc}"
                )

        independent_changes: list[dict[str, Any]] = []
        annotated_changes: list[dict[str, Any]] = []
        edits: list[dict[str, Any]] = []
        try:
            independent_changes = _independent_shape_diff(parent_code, child_code)
            edits = _solver_slot_edits(decision)
            annotated_changes = _annotate_independent_changes(
                edits,
                independent_changes,
            )
        except (SyntaxError, TypeError, ValueError) as exc:
            reason = f"{type(exc).__name__}:{exc}"
            shape_diff_errors[child_uuid] = reason
            verification_errors.setdefault(child_uuid, []).append(
                f"independent_shape_diff_failed:{reason}"
            )
        checks, errors = _verify_child(
            parent,
            child,
            decision,
            slot_cache[parent_uuid],
            target_by_key.get((parent_uuid, str(decision["variant"]))),
            independent_changes,
        )
        check_counts.update(name for name, passed in checks.items() if passed)
        if errors:
            verification_errors.setdefault(child_uuid, []).extend(errors)

        dimension_balance_audit = _dimension_balance_child_audit(
            child_code,
            decision,
            dimension_balance_ratio_limit,
            independent_changes,
        )
        dimension_balance_audits.append(dimension_balance_audit)
        if dimension_balance_audit["resolved"]:
            check_counts["dimension_balance_reparsed"] += 1
        if dimension_balance_ratio_limit is not None:
            if (
                dimension_balance_audit["resolved"]
                and dimension_balance_audit["violation"] is False
            ):
                check_counts["dimension_balance_contract_passed"] += 1
            else:
                reason = dimension_balance_audit.get("reason")
                if reason is None:
                    violating_factories = [
                        factory
                        for factory in dimension_balance_audit.get("factories", [])
                        if factory.get("violation") is True
                    ]
                    reason = f"ratio_exceeded:{violating_factories[:3]}"
                verification_errors.setdefault(child_uuid, []).append(
                    f"dimension_balance_fail_closed:{reason}"
                )
        occurrence_counts.append(len(annotated_changes))
        axis_set = {
            int(occurrence["axis_from_right"])
            for occurrence in annotated_changes
        }
        unique_axes_per_child.append(len(axis_set))
        if any(int(occurrence["axis"]) == 0 for occurrence in annotated_changes):
            children_with_leading += 1
        for edit in edits:
            slot_kind_counts[str(edit["slot"].get("kind"))] += 1
        for occurrence_index, occurrence in enumerate(annotated_changes):
            slot = occurrence["slot"]
            declared_occurrence = occurrence["declared_occurrence"]
            leading = int(occurrence["axis"]) == 0
            leading_occurrences += int(leading)
            axis_from_right_counts[int(occurrence["axis_from_right"])] += 1
            new_values.append(int(occurrence["new_value"]))
            changed_rows.append(
                {
                    "parent_uuid": parent_uuid,
                    "child_uuid": child_uuid,
                    "source_family": parent_meta[parent_uuid]["source_family"],
                    "operator_family": parent_meta[parent_uuid]["operator_family"],
                    "variant": decision["variant"],
                    "slot_index": occurrence["slot_index"],
                    "slot_occurrence_index": occurrence["slot_occurrence_index"],
                    "slot_id": slot.get("slot_id"),
                    "slot_kind": slot.get("kind"),
                    "symbol_name": slot.get("symbol_name"),
                    "symbol_scope": slot.get("symbol_scope"),
                    "old_value": occurrence["old_value"],
                    "new_value": occurrence["new_value"],
                    "occurrence_index": occurrence_index,
                    "factory_index": occurrence["factory_index"],
                    "factory_name": occurrence["factory_name"],
                    "axis": occurrence["axis"],
                    "rank": occurrence["rank"],
                    "axis_from_right": occurrence["axis_from_right"],
                    "leading": leading,
                    "source_span": json.dumps(
                        declared_occurrence.get("source_span"), sort_keys=True
                    ),
                    "input_bytes_before": decision.get("input_bytes_before"),
                    "input_bytes_after": decision.get("input_bytes_after"),
                    "input_scale": decision.get("input_scale"),
                    "target_input_bytes": decision.get("target_input_bytes"),
                    "target_delta_bytes": decision.get("target_delta_bytes"),
                    "target_relative_error": decision.get("target_relative_error"),
                }
            )
        accepted_samples.append(
            {
                "decision": decision,
                "parent_code": parent_code,
                "child_code": child_code,
                "slot_edits": edits,
                "changed_occurrences": annotated_changes,
                **parent_meta[parent_uuid],
            }
        )

    accepted_decision_children = {str(item.get("child_uuid")) for item in accepted}
    orphan_children = sorted(set(child_by_uuid) - accepted_decision_children)
    missing_children = sorted(accepted_decision_children - set(child_by_uuid))
    if orphan_children:
        verification_errors["__orphan_children__"] = orphan_children[:50]
    if missing_children:
        verification_errors["__missing_children__"] = missing_children[:50]
    if not variant_cardinality["passed"]:
        verification_errors["__variant_cardinality_contract__"] = list(
            variant_cardinality["errors"]
        )
    if (
        variant_cardinality["mode"] == "one_variant_per_parent"
        and dimension_balance_ratio_limit is None
    ):
        verification_errors["__dimension_balance_contract__"] = [
            "v4_one_variant_manifest_missing_dimension_balance_ratio_limit"
        ]

    failure_counts: collections.Counter[str] = collections.Counter()
    failure_parents: dict[str, set[str]] = collections.defaultdict(set)
    failed_samples: list[dict[str, Any]] = []
    child_attempt_failures: collections.Counter[str] = collections.Counter()
    for decision in rejected:
        category = _decision_failure_category(decision)
        failure_counts[category] += 1
        failure_parents[category].add(str(decision.get("parent_uuid")))
        failed_samples.append({"category": category, "decision": decision})
    for decision in decisions:
        attempts = decision.get("attempts", [])
        if isinstance(attempts, list):
            for attempt in attempts:
                if not isinstance(attempt, Mapping) or attempt.get("accepted") is not False:
                    continue
                fake_status = _nested(attempt, "fake_gate.status")
                if fake_status is not None:
                    child_attempt_failures[f"fake_{fake_status}"] += 1
                else:
                    reason = str(attempt.get("reason", "missing_reason"))
                    child_attempt_failures[reason.split(":", 2)[-1][:160]] += 1

    target_errors = [float(item["target_relative_error"]) for item in accepted]
    target_abs_bytes = [abs(int(item["target_delta_bytes"])) for item in accepted]
    input_scales = [float(item["input_scale"]) for item in accepted]
    input_before = [int(item["input_bytes_before"]) for item in accepted]
    input_after = [int(item["input_bytes_after"]) for item in accepted]
    ai_baseline = _load_ai_baseline(ai_run_dir, selected_uuids)
    artifact_hashes = {
        "selected": _sha256_file(selected_path),
        "targets": _sha256_file(targets_path),
        "children": _sha256_file(children_path),
        "manifest": _sha256_file(manifest_path),
    }
    reference_runtime, reference_records = _reference_runtime_summary(
        run_dir,
        accepted,
        selected_by_uuid,
        child_by_uuid,
        parent_meta,
    )
    reference_runtime["region_allowlist"] = _write_region_reference_allowlist(
        run_dir,
        accepted,
        reference_runtime,
        reference_records,
    )
    region_runtime, region_records = _region_runtime_summary(
        run_dir,
        accepted,
        artifact_hashes,
        reference_runtime,
        reference_records,
    )
    if region_runtime.get("expected_scope") == "reference_parent_and_child_both_pass":
        region_runtime["expected_allowlist_artifact"] = reference_runtime[
            "region_allowlist"
        ]
    runtime = {
        "reference": reference_runtime,
        "region": region_runtime,
        "eligible": _eligible_runtime_summary(
            accepted,
            len(selected_rows),
            parent_meta,
            reference_runtime,
            reference_records,
            region_runtime,
            region_records,
        ),
    }
    dimension_balance = _dimension_balance_summary(
        dimension_balance_audits,
        dimension_balance_ratio_limit,
    )
    runtime_eligible_bias = _runtime_eligible_bias_profile(
        runtime["eligible"],
        accepted,
        changed_rows,
        dimension_balance_audits,
        dimension_balance_ratio_limit,
    )
    accepted_child_uuids = [str(decision["child_uuid"]) for decision in accepted]
    static_quality = _shape_quality_gate(
        scope="static_accepted",
        child_uuids=accepted_child_uuids,
        changed_rows=changed_rows,
        available=True,
        evidence_complete=not shape_diff_errors,
        evidence_errors=[
            f"{child_uuid}:{reason}"
            for child_uuid, reason in sorted(shape_diff_errors.items())
        ],
        logical_slot_counts=logical_slot_counts,
        required_logical_slots=required_logical_slots,
        logical_slot_range=logical_slot_range,
    )
    runtime_eligible = runtime["eligible"]
    raw_runtime_eligible_uuids = runtime_eligible.get("eligible_child_uuids", [])
    runtime_eligible_uuids = (
        list(raw_runtime_eligible_uuids)
        if isinstance(raw_runtime_eligible_uuids, list)
        else []
    )
    runtime_shape_errors = {
        child_uuid: reason
        for child_uuid, reason in shape_diff_errors.items()
        if child_uuid in set(runtime_eligible_uuids)
    }
    runtime_quality = _shape_quality_gate(
        scope="runtime_eligible",
        child_uuids=runtime_eligible_uuids,
        changed_rows=changed_rows,
        available=runtime_eligible.get("available") is True,
        evidence_complete=(
            runtime_eligible.get("available") is True
            and runtime_eligible.get("coverage_complete") is True
            and not runtime_shape_errors
        ),
        evidence_errors=[
            f"{child_uuid}:{reason}"
            for child_uuid, reason in sorted(runtime_shape_errors.items())
        ],
        logical_slot_counts=logical_slot_counts,
        required_logical_slots=required_logical_slots,
        logical_slot_range=logical_slot_range,
    )
    quality = {
        "contract_version": "shape_distribution_quality_v2",
        "variant_cardinality": variant_cardinality,
        "static": static_quality,
        "runtime": runtime_quality,
    }

    check_name_set = {
        "model_ast_equal",
        "get_init_inputs_ast_equal",
        "factory_sequence_equal",
        "factory_rank_equal",
        "declared_slots_reextracted",
        "declared_slots_exact_match",
        "exact_simultaneous_source_span_patch",
        "independent_shape_diff_matches_declared_slots",
        "static_identity_gate",
        "child_reference_hash",
        "child_normalized_ast_hash",
        "child_uuid",
        "direct_parent_uuid",
        "target_record_match",
        "parent_fake_passed",
        "child_fake_passed",
        "dimension_balance_reparsed",
    }
    if dimension_balance_ratio_limit is not None:
        check_name_set.add("dimension_balance_contract_passed")
    check_names = sorted(check_name_set)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir.resolve()),
        "artifact_sha256": artifact_hashes,
        "coverage": {
            "selected_parents": len(selected_rows),
            "target_rows": len(target_rows),
            "variant_decisions": len(decisions),
            "accepted_children": len(accepted),
            "accepted_parents": len(accepted_parent_uuids),
            "parent_coverage": _fraction(len(accepted_parent_uuids), len(selected_rows)) or 0.0,
            "children_parquet_rows": len(child_rows),
        },
        "by_variant": by_variant,
        "by_source_family": _group_coverage(
            selected_uuids, parent_meta, accepted, "source_family"
        ),
        "by_operator_family": _group_coverage(
            selected_uuids, parent_meta, accepted, "operator_family"
        ),
        "by_operator_bucket": _group_coverage(
            selected_uuids, parent_meta, accepted, "operator_bucket"
        ),
        "target": {
            "absolute_error_bytes": _distribution(target_abs_bytes),
            "relative_error": _distribution(target_errors),
            "exact_count": sum(value == 0 for value in target_abs_bytes),
            "exact_fraction": _fraction(sum(value == 0 for value in target_abs_bytes), len(target_abs_bytes)),
        },
        "input_bytes_before": _distribution(input_before),
        "input_bytes_after": _distribution(input_after),
        "input_scale": _distribution(input_scales),
        "dimension_balance": dimension_balance,
        "quality": quality,
        "variant_cardinality": variant_cardinality,
        "failures": {
            "decision_categories": {
                category: {
                    "decisions": count,
                    "parents": len(failure_parents[category]),
                }
                for category, count in sorted(failure_counts.items())
            },
            "child_candidate_attempt_failures": _counter(child_attempt_failures),
            "manifest_skip_reason_counts": manifest.get("skip_reason_counts", {}),
            "missing_variant_decisions": max(
                0,
                len(selected_rows)
                * int(variant_cardinality["expected_decisions_per_parent"])
                - len(decisions),
            ),
        },
        "verification": {
            "children_checked": len(accepted),
            "invalid_children": len(verification_errors),
            "all_passed": not verification_errors,
            "check_pass_counts": {name: check_counts[name] for name in check_names},
            "errors": verification_errors,
            "orphan_child_count": len(orphan_children),
            "missing_child_count": len(missing_children),
        },
        "runtime": runtime,
        "ai_comparison": ai_baseline,
    }

    dominant_axis_count = max(axis_from_right_counts.values(), default=0)
    bias = {
        "schema_version": SCHEMA_VERSION,
        "run_dir": str(run_dir.resolve()),
        "measurement_contract": {
            "sample": "one statically accepted child",
            "dimension_occurrence": (
                "one independently resolved parent/child direct-factory (factory_index, axis) diff; "
                "shared slots contribute once per actual factory occurrence"
            ),
            "leading": "axis index zero within its input factory",
            "one_axis": "one unique axis-from-right value among a child's changed occurrences",
        },
        "changed_dimensions": _value_profile(new_values, len(accepted)),
        "changed_dimension_value_partitions": _partitioned_value_profiles(
            new_values, len(accepted)
        ),
        "slot_kind_counts": _counter(slot_kind_counts),
        "position": {
            "accepted_children": len(accepted),
            "changed_occurrences": len(new_values),
            "changed_occurrences_per_child": {
                **_distribution(occurrence_counts),
                "counts": _counter(collections.Counter(occurrence_counts)),
            },
            "single_occurrence_child_fraction": _fraction(
                sum(value == 1 for value in occurrence_counts), len(occurrence_counts)
            ) or 0.0,
            "unique_axis_from_right_per_child": {
                **_distribution(unique_axes_per_child),
                "counts": _counter(collections.Counter(unique_axes_per_child)),
            },
            "single_axis_from_right_child_fraction": _fraction(
                sum(value == 1 for value in unique_axes_per_child), len(unique_axes_per_child)
            ) or 0.0,
            "children_with_leading_fraction": _fraction(children_with_leading, len(accepted)) or 0.0,
            "leading_occurrence_fraction": _fraction(leading_occurrences, len(new_values)) or 0.0,
            "axis_from_right_counts": _counter(axis_from_right_counts),
            "dominant_axis_from_right_fraction": _fraction(dominant_axis_count, len(new_values)) or 0.0,
        },
        "capacity": _capacity_profile(input_after),
        "dimension_balance": dimension_balance,
        "runtime_eligible": runtime_eligible_bias,
        "quality_gates": quality,
        "by_variant": {
            variant: {
                "accepted_children": len(items := [item for item in accepted if item["variant"] == variant]),
                "target_relative_error": _distribution(
                    [float(item["target_relative_error"]) for item in items]
                ),
                "input_bytes_after": _capacity_profile(
                    [int(item["input_bytes_after"]) for item in items]
                ),
            }
            for variant in ("medium", "large")
        },
        "ai_comparison": ai_baseline,
    }

    output_dir = run_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    bias_json = output_dir / "shape_bias_audit.json"
    bias_md = output_dir / "shape_bias_audit.md"
    changed_tsv = output_dir / "changed_slots.tsv"
    review_md = output_dir / "stratified_review_samples.md"
    _atomic_write(summary_json, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    _atomic_write(summary_md, _summary_markdown(summary))
    _atomic_write(bias_json, json.dumps(bias, indent=2, sort_keys=True) + "\n")
    _atomic_write(bias_md, _bias_markdown(bias))
    columns = [
        "parent_uuid", "child_uuid", "source_family", "operator_family", "variant",
        "slot_index", "slot_occurrence_index", "slot_id", "slot_kind", "symbol_name",
        "symbol_scope", "old_value", "new_value", "occurrence_index", "factory_index",
        "factory_name", "axis", "rank", "axis_from_right",
        "leading", "source_span", "input_bytes_before", "input_bytes_after", "input_scale",
        "target_input_bytes", "target_delta_bytes", "target_relative_error",
    ]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, dialect="excel-tab", lineterminator="\n")
    writer.writeheader()
    writer.writerows(changed_rows)
    _atomic_write(changed_tsv, buffer.getvalue())
    _atomic_write(review_md, _review_markdown(accepted_samples, failed_samples))
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", nargs="?", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--ai-run-dir", type=Path, default=DEFAULT_AI_RUN_DIR)
    parser.add_argument(
        "--require-runtime-quality",
        action="store_true",
        help=(
            "fail unless complete reference+region evidence exists and the runtime-eligible "
            "subset passes the shape-distribution quality contract"
        ),
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    summary = analyze(args.run_dir, args.ai_run_dir)
    eligible = summary["runtime"]["eligible"]
    static_quality_passed = summary["quality"]["static"]["passed"] is True
    runtime_quality_passed = summary["quality"]["runtime"]["passed"] is True
    variant_cardinality_passed = summary["variant_cardinality"]["passed"] is True
    print(json.dumps({
        "coverage": summary["coverage"],
        "runtime": {
            "reference_available": summary["runtime"]["reference"]["available"],
            "region_available": summary["runtime"]["region"]["available"],
            "region_contract_compatible": summary["runtime"]["region"].get(
                "contract_compatible"
            ),
            "eligible_children": eligible.get("eligible_children"),
            "covered_parents": eligible.get("covered_parents"),
            "both_variants_parents": eligible.get("both_variants_parents"),
            "coverage_complete": eligible.get("coverage_complete"),
        },
        "verification": {
            "children_checked": summary["verification"]["children_checked"],
            "invalid_children": summary["verification"]["invalid_children"],
            "all_passed": summary["verification"]["all_passed"],
        },
        "quality": {
            "variant_cardinality_passed": variant_cardinality_passed,
            "static_passed": static_quality_passed,
            "runtime_passed": runtime_quality_passed,
            "runtime_required": bool(args.require_runtime_quality),
        },
        "analysis_dir": str((args.run_dir / "analysis").resolve()),
    }, indent=2, sort_keys=True))
    passed = (
        summary["verification"]["all_passed"]
        and variant_cardinality_passed
        and static_quality_passed
        and (runtime_quality_passed or not args.require_runtime_quality)
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
