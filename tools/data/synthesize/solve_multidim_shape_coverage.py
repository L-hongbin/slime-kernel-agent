#!/usr/bin/env python3
"""Generate one deterministic two-dimension shape sibling per parent.

The solver is intentionally narrower than ``solve_shape_coverage.py``.  It
reuses that solver's conservative slot discovery, fixed-record dead-tail guard,
single-slot affine proof, FakeTensor gate, and child-row construction, but only
admits strict structural pairs of shape slots inside ``get_inputs``.  The two
slots must affect the same non-empty set of input factories, exactly once per
factory on distinct axes.

Both slot values strictly increase.  Exactly one new value is a power of two
and the other is not.  Therefore a pair affecting ``k`` factories changes
``2*k`` dimension occurrences, exactly ``k`` of which are powers of two.  This
locally proves a 50% power-of-two occurrence share and zero single-occurrence
children, independently of sharding and later runtime filtering.

Each parent is assigned exactly one byte-granularity Medium or Large target by
a stable hash.
Candidates remain subject to the existing 2x input-growth, target +/-25%,
Medium/Large storage, 4 GiB Large ceiling, 1000:1 dimension-balance, static
identity, and parent/child FakeTensor gates.  Target-GPU reference and changed-
region validation remain separate gates.
"""

from __future__ import annotations

import argparse
import ast
import collections
import copy
import dataclasses
import difflib
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.synthesize.augment_prompt_tasks import (  # noqa: E402
    _SHAPE_FACTORIES,
    _call_name,
    _factory_records,
    _module_constant_environment,
    _section_hashes,
    _top_level_function,
    analyze_code,
)
from tools.data.synthesize.shape_contract import (  # noqa: E402
    LARGE_INPUT_MAX_BYTES,
    MEDIUM_INPUT_MAX_BYTES,
    MEDIUM_INPUT_MIN_BYTES,
    MIN_INPUT_SCALE,
    _validate_target_proximity,
    _validate_variant_storage,
    static_gate,
)
from tools.data.synthesize.solve_shape_coverage import (  # noqa: E402
    DEFAULT_FAKE_GATE_TIMEOUT_SECONDS,
    MAX_DIMENSION_IMBALANCE_RATIO,
    AffineProfile,
    ShapeSlot,
    SourceSpan,
    _affine_profiles,
    _fake_tensor_gate,
    _make_child,
    _shape_slots_with_rejections,
)

CONTRACT_VERSION = "shape_multidim_solver_v4"
GENERATOR_VERSION = "strict_structural_two_slot_exactly_one_power_v3"
VARIANT_ASSIGNMENT_SALT = "shape_multidim_solver_v4_single_variant_v1"
TARGET_BYTE_SALT = "shape_multidim_solver_v4_byte_target_v1"
DEFAULT_SELECTED = _REPO_ROOT / "Data/prompt_tvm_v4/shape_ai_random_targets_low_tp8_v7/run.1000/selected.parquet"
DEFAULT_RUN_DIR = _REPO_ROOT / "Data/prompt_tvm_v4/shape_solver_multidim_v4/run.500"
VARIANTS = ("medium", "large")
MIB = 1024**2
TARGET_ERROR_EQUIVALENCE_DENOMINATOR = 1_000
MAX_REVIEW_EXAMPLES = 16
MAX_CHILD_FAKE_ATTEMPTS = 4
MAX_CANDIDATES_PER_PAIR = 4


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


@dataclasses.dataclass(frozen=True)
class ResolvedFactory:
    factory_index: int
    factory_name: str
    shape: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class BilinearProfile:
    slot_a: ShapeSlot
    slot_b: ShapeSlot
    constant_bytes: int
    x_bytes: int
    y_bytes: int
    xy_bytes: int

    def input_bytes(self, x: int, y: int) -> int:
        return self.constant_bytes + self.x_bytes * x + self.y_bytes * y + self.xy_bytes * x * y


@dataclasses.dataclass(frozen=True)
class SolvedCandidate:
    profile: BilinearProfile
    variant: str
    target_input_bytes: int
    value_a: int
    value_b: int
    input_bytes_after: int
    target_delta_bytes: int
    target_relative_error: float
    changed_occurrences: int
    power_of_two_occurrences: int
    balance_evidence: Mapping[str, Any]
    relative_growth_ratio: Fraction
    maximum_dimension_ratio: Fraction
    child_code: str


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


def _nested(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _logical_target_map_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    canonical = "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records).encode(
        "utf-8"
    )
    return _sha256_bytes(canonical)


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("ceil_div_requires_positive_denominator")
    return -((-numerator) // denominator)


def _span_inside(span: SourceSpan, node: ast.AST) -> bool:
    start = (getattr(node, "lineno", -1), getattr(node, "col_offset", -1))
    end = (getattr(node, "end_lineno", -1), getattr(node, "end_col_offset", -1))
    return (
        start <= (span.lineno, span.col_offset)
        and (
            span.end_lineno,
            span.end_col_offset,
        )
        <= end
    )


def _spans_overlap(left: SourceSpan, right: SourceSpan) -> bool:
    left_start = (left.lineno, left.col_offset)
    left_end = (left.end_lineno, left.end_col_offset)
    right_start = (right.lineno, right.col_offset)
    right_end = (right.end_lineno, right.end_col_offset)
    return left_start < right_end and right_start < left_end


def _patch_integer_spans_many(
    code: str,
    assignments: Sequence[tuple[ShapeSlot, int]],
) -> str:
    """Patch multiple slots against the same original UTF-8 source coordinates."""

    if not assignments:
        raise ValueError("at_least_one_shape_slot_assignment_is_required")
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
    seen_slot_ids: set[str] = set()
    for slot, new_value in assignments:
        if slot.slot_id in seen_slot_ids:
            raise ValueError(f"duplicate_shape_slot_assignment:{slot.slot_id}")
        seen_slot_ids.add(slot.slot_id)
        if type(new_value) is not int or new_value <= slot.old_value:
            raise ValueError(f"shape_slot_must_strictly_increase:{slot.slot_id}:" f"{slot.old_value}:{new_value}")
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


def _resolved_factories(
    code: str,
    required_factory_indices: set[int],
) -> tuple[ResolvedFactory, ...]:
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

    resolved: list[ResolvedFactory] = []
    for factory_index, call in enumerate(calls):
        if factory_index not in required_factory_indices:
            continue
        location = (call.lineno, call.col_offset)
        record = records_by_location.get(location)
        if record is None:
            raise ValueError(
                f"direct_factory_shape_unresolved:{factory_index}:" f"line{call.lineno}:col{call.col_offset}"
            )
        name = _call_name(call.func)
        if record.name != name:
            raise ValueError(f"direct_factory_identity_mismatch:{factory_index}")
        resolved.append(
            ResolvedFactory(
                factory_index=factory_index,
                factory_name=name,
                shape=tuple(int(value) for value in record.shape),
            )
        )
    return tuple(resolved)


def _structural_pair(
    slot_a: ShapeSlot,
    slot_b: ShapeSlot,
    get_inputs: ast.FunctionDef,
) -> bool:
    if slot_a.slot_id == slot_b.slot_id:
        return False
    if any(_spans_overlap(left, right) for left in slot_a.patch_spans for right in slot_b.patch_spans):
        return False
    if not all(_span_inside(span, get_inputs) for slot in (slot_a, slot_b) for span in slot.patch_spans):
        return False

    by_factory_a: dict[int, list[int]] = collections.defaultdict(list)
    by_factory_b: dict[int, list[int]] = collections.defaultdict(list)
    for occurrence in slot_a.occurrences:
        by_factory_a[occurrence.factory_index].append(occurrence.axis)
    for occurrence in slot_b.occurrences:
        by_factory_b[occurrence.factory_index].append(occurrence.axis)
    if not by_factory_a or set(by_factory_a) != set(by_factory_b):
        return False
    for factory_index in by_factory_a:
        axes_a = by_factory_a[factory_index]
        axes_b = by_factory_b[factory_index]
        if len(axes_a) != 1 or len(axes_b) != 1 or axes_a[0] == axes_b[0]:
            return False
    return len(slot_a.occurrences) == len(slot_b.occurrences)


def _bilinear_profile(
    code: str,
    entry_point: str,
    parent_input_bytes: int,
    profile_a: AffineProfile,
    profile_b: AffineProfile,
) -> BilinearProfile:
    slot_a = profile_a.slot
    slot_b = profile_b.slot
    x0, y0 = slot_a.old_value, slot_b.old_value
    f00 = parent_input_bytes
    f10 = profile_a.input_bytes(x0 + 1)
    f01 = profile_b.input_bytes(y0 + 1)
    plus_one_code = _patch_integer_spans_many(
        code,
        ((slot_a, x0 + 1), (slot_b, y0 + 1)),
    )
    f11 = analyze_code(plus_one_code, entry_point).input_bytes
    xy_bytes = f11 - f10 - f01 + f00
    x_bytes = profile_a.bytes_per_slot_unit - xy_bytes * y0
    y_bytes = profile_b.bytes_per_slot_unit - xy_bytes * x0
    constant_bytes = f00 - x_bytes * x0 - y_bytes * y0 - xy_bytes * x0 * y0
    if min(constant_bytes, x_bytes, y_bytes) < 0 or xy_bytes <= 0:
        raise ValueError("two_slot_storage_polynomial_is_not_positive_bilinear")
    result = BilinearProfile(
        slot_a=slot_a,
        slot_b=slot_b,
        constant_bytes=constant_bytes,
        x_bytes=x_bytes,
        y_bytes=y_bytes,
        xy_bytes=xy_bytes,
    )
    if result.input_bytes(x0, y0) != f00:
        raise ValueError("two_slot_storage_polynomial_parent_mismatch")
    plus_two_code = _patch_integer_spans_many(
        code,
        ((slot_a, x0 + 2), (slot_b, y0 + 2)),
    )
    f22 = analyze_code(plus_two_code, entry_point).input_bytes
    if result.input_bytes(x0 + 2, y0 + 2) != f22:
        raise ValueError("two_slot_storage_is_not_bilinear")
    return result


def _bilinear_profiles(
    code: str,
    entry_point: str,
    parent_input_bytes: int,
    affine_profiles: Sequence[AffineProfile],
) -> tuple[list[BilinearProfile], list[dict[str, Any]]]:
    tree = ast.parse(code)
    get_inputs_node = _top_level_function(tree, "get_inputs")
    if not isinstance(get_inputs_node, ast.FunctionDef):
        raise ValueError("get_inputs_must_be_a_synchronous_function")
    profiles: list[BilinearProfile] = []
    rejected: list[dict[str, Any]] = []
    for index, profile_a in enumerate(affine_profiles):
        for profile_b in affine_profiles[index + 1 :]:
            if not _structural_pair(profile_a.slot, profile_b.slot, get_inputs_node):
                continue
            try:
                profiles.append(
                    _bilinear_profile(
                        code,
                        entry_point,
                        parent_input_bytes,
                        profile_a,
                        profile_b,
                    )
                )
            except (SyntaxError, TypeError, ValueError) as exc:
                rejected.append(
                    {
                        "slot_ids": [
                            profile_a.slot.slot_id,
                            profile_b.slot.slot_id,
                        ],
                        "reason": f"{type(exc).__name__}:{exc}",
                    }
                )
    return profiles, rejected


def _affected_shape_balance_guard(
    child_code: str,
    assignments: Sequence[tuple[ShapeSlot, int]],
) -> dict[str, Any]:
    """Independently reparse every final direct-input shape and enforce 1000:1."""

    result: dict[str, Any] = {
        "passed": False,
        "resolved": False,
        "reason": None,
        "ratio_limit": MAX_DIMENSION_IMBALANCE_RATIO,
        "comparison": "largest_dimension <= 1000 * second_largest_dimension",
        "scope": "all_final_direct_input_factories",
        "changed_occurrences": 0,
        "affected_factories": [],
        "violations": [],
    }
    expected: dict[tuple[int, int], tuple[int, int, str]] = {}
    for slot, value in assignments:
        if type(value) is not int or value <= slot.old_value:
            result["reason"] = f"slot_is_not_a_strict_expansion:{slot.slot_id}"
            return result
        for occurrence in slot.occurrences:
            key = (occurrence.factory_index, occurrence.axis)
            if key in expected:
                result["reason"] = f"overlapping_affected_factory_axis:{key}"
                return result
            expected[key] = (value, occurrence.rank, slot.slot_id)
    result["changed_occurrences"] = len(expected)
    if not expected:
        result["reason"] = "empty_affected_occurrences"
        return result

    try:
        required_factory_indices = {factory for factory, _ in expected}
        tree = ast.parse(child_code)
        get_inputs_node = _top_level_function(tree, "get_inputs")
        if not isinstance(get_inputs_node, ast.FunctionDef):
            raise ValueError("get_inputs_must_be_a_synchronous_function")
        direct_calls = sorted(
            (
                node
                for node in ast.walk(get_inputs_node)
                if isinstance(node, ast.Call) and _call_name(node.func) in _SHAPE_FACTORIES
            ),
            key=lambda node: (node.lineno, node.col_offset),
        )
        all_factory_indices = set(range(len(direct_calls)))
        if not required_factory_indices <= all_factory_indices:
            raise ValueError("affected_factory_index_out_of_range")
        factories = _resolved_factories(child_code, all_factory_indices)
        by_index = {factory.factory_index: factory for factory in factories}
        for factory_index in sorted(all_factory_indices):
            factory = by_index.get(factory_index)
            if factory is None:
                raise ValueError(f"final_direct_factory_missing:{factory_index}")
            axes = sorted(axis for current, axis in expected if current == factory_index)
            if any(not 0 <= axis < len(factory.shape) for axis in axes):
                raise ValueError(f"affected_axis_out_of_range:{factory_index}:{axes}")
            for axis in axes:
                value, rank, slot_id = expected[(factory_index, axis)]
                if rank != len(factory.shape):
                    raise ValueError(f"affected_rank_mismatch:{factory_index}:{axis}:{rank}:" f"{len(factory.shape)}")
                if factory.shape[axis] != value:
                    raise ValueError(
                        f"affected_value_mismatch:{slot_id}:{factory_index}:{axis}:" f"{value}:{factory.shape[axis]}"
                    )
            evidence: dict[str, Any] = {
                "factory_index": factory_index,
                "factory_name": factory.factory_name,
                "affected_axes": axes,
                "shape": list(factory.shape),
                "rank": len(factory.shape),
                "guard_applies": len(factory.shape) >= 2,
            }
            if len(factory.shape) >= 2:
                largest, second = sorted(factory.shape, reverse=True)[:2]
                balanced = largest <= MAX_DIMENSION_IMBALANCE_RATIO * second
                evidence.update(
                    {
                        "largest_dimension": largest,
                        "second_largest_dimension": second,
                        "largest_to_second_ratio": largest / second,
                        "balanced": balanced,
                    }
                )
                if not balanced:
                    result["violations"].append(copy.deepcopy(evidence))
            result["affected_factories"].append(evidence)
    except (SyntaxError, TypeError, ValueError) as exc:
        result["reason"] = f"affected_shape_resolution_failed:{type(exc).__name__}:{exc}"
        return result

    result["resolved"] = True
    if result["violations"]:
        first = result["violations"][0]
        result["reason"] = (
            "dimension_balance_ratio_exceeded:"
            f"factory_index={first['factory_index']}:"
            f"largest={first['largest_dimension']}:"
            f"second_largest={first['second_largest_dimension']}:"
            f"limit={MAX_DIMENSION_IMBALANCE_RATIO}"
        )
        return result
    result["passed"] = True
    return result


def _variant_and_target(
    parent_uuid: str,
    parent_input_bytes: int,
) -> tuple[str, int, int, int]:
    """Choose one stable variant and sample its legal interval at byte granularity."""

    digest = _sha256_bytes(f"{VARIANT_ASSIGNMENT_SALT}:{parent_uuid}".encode())
    variant = VARIANTS[int(digest[:16], 16) % len(VARIANTS)]
    minimum_growth_mib = math.ceil(MIN_INPUT_SCALE * parent_input_bytes / MIB)
    if variant == "medium":
        lower_mib = max(MEDIUM_INPUT_MIN_BYTES // MIB, minimum_growth_mib)
        upper_mib = MEDIUM_INPUT_MAX_BYTES // MIB
    else:
        lower_mib = max(MEDIUM_INPUT_MAX_BYTES // MIB + 1, minimum_growth_mib)
        upper_mib = LARGE_INPUT_MAX_BYTES // MIB
    if lower_mib > upper_mib:
        raise ValueError(f"assigned_variant_has_no_2x_target_band:{variant}:{lower_mib}:{upper_mib}")
    lower_bytes = lower_mib * MIB
    upper_bytes = upper_mib * MIB
    target_digest = _sha256_bytes(f"{TARGET_BYTE_SALT}:{parent_uuid}:{variant}".encode())
    target = lower_bytes + int(target_digest[:16], 16) % (upper_bytes - lower_bytes + 1)
    return variant, target, lower_mib, upper_mib


def _storage_interval(
    variant: str,
    target_input_bytes: int,
    parent_input_bytes: int,
) -> tuple[int, int]:
    if variant == "medium":
        band_lower, band_upper = MEDIUM_INPUT_MIN_BYTES, MEDIUM_INPUT_MAX_BYTES
    elif variant == "large":
        band_lower, band_upper = MEDIUM_INPUT_MAX_BYTES + 1, LARGE_INPUT_MAX_BYTES
    else:
        raise ValueError(f"unknown_shape_variant:{variant}")
    target_lower = _ceil_div(3 * target_input_bytes, 4)
    target_upper = (5 * target_input_bytes) // 4
    growth_lower = math.ceil(MIN_INPUT_SCALE * parent_input_bytes)
    lower = max(band_lower, target_lower, growth_lower)
    upper = min(band_upper, target_upper)
    if lower > upper:
        raise ValueError("empty_variant_target_growth_storage_interval")
    return lower, upper


def _powers_between(lower_exclusive: int, upper_inclusive: int) -> list[int]:
    if upper_inclusive <= lower_exclusive:
        return []
    value = 1 << max(0, lower_exclusive.bit_length())
    result: list[int] = []
    while value <= upper_inclusive:
        result.append(value)
        value <<= 1
    return result


def _candidate_from_values(
    parent_code: str,
    entry_point: str,
    profile: BilinearProfile,
    *,
    variant: str,
    target_input_bytes: int,
    parent_input_bytes: int,
    value_a: int,
    value_b: int,
) -> SolvedCandidate | None:
    slot_a, slot_b = profile.slot_a, profile.slot_b
    if value_a <= slot_a.old_value or value_b <= slot_b.old_value:
        return None
    if _is_power_of_two(value_a) == _is_power_of_two(value_b):
        return None
    input_bytes_after = profile.input_bytes(value_a, value_b)
    try:
        _validate_variant_storage(variant, input_bytes_after)
        target_relative_error = _validate_target_proximity(
            input_bytes_after,
            target_input_bytes,
        )
        if input_bytes_after < math.ceil(MIN_INPUT_SCALE * parent_input_bytes):
            raise ValueError("minimum_input_growth_not_met")
        child_code = _patch_integer_spans_many(
            parent_code,
            ((slot_a, value_a), (slot_b, value_b)),
        )
        balance = _affected_shape_balance_guard(
            child_code,
            ((slot_a, value_a), (slot_b, value_b)),
        )
        if not balance["passed"]:
            raise ValueError(f"dimension_balance_guard_rejected:{balance['reason']}")
        if analyze_code(child_code, entry_point).input_bytes != input_bytes_after:
            raise ValueError("bilinear_solution_storage_mismatch")
    except (SyntaxError, TypeError, ValueError):
        return None

    count_a = len(slot_a.occurrences)
    count_b = len(slot_b.occurrences)
    if count_a <= 0 or count_a != count_b:
        return None
    power_occurrences = count_a if _is_power_of_two(value_a) else count_b
    changed_occurrences = count_a + count_b
    if 2 * power_occurrences != changed_occurrences:
        return None
    growth_a = Fraction(value_a, slot_a.old_value)
    growth_b = Fraction(value_b, slot_b.old_value)
    relative_growth_ratio = max(growth_a, growth_b) / min(growth_a, growth_b)
    ratios = [
        Fraction(
            int(factory["largest_dimension"]),
            int(factory["second_largest_dimension"]),
        )
        for factory in balance["affected_factories"]
        if factory["guard_applies"]
    ]
    return SolvedCandidate(
        profile=profile,
        variant=variant,
        target_input_bytes=target_input_bytes,
        value_a=value_a,
        value_b=value_b,
        input_bytes_after=input_bytes_after,
        target_delta_bytes=input_bytes_after - target_input_bytes,
        target_relative_error=target_relative_error,
        changed_occurrences=changed_occurrences,
        power_of_two_occurrences=power_occurrences,
        balance_evidence=balance,
        relative_growth_ratio=relative_growth_ratio,
        maximum_dimension_ratio=max(ratios, default=Fraction(0, 1)),
        child_code=child_code,
    )


def _solve_profile(
    parent_code: str,
    entry_point: str,
    profile: BilinearProfile,
    *,
    parent_uuid: str,
    variant: str,
    target_input_bytes: int,
    parent_input_bytes: int,
) -> list[SolvedCandidate]:
    """Fix one slot to a power of two, then solve the other affine variable."""

    storage_lower, storage_upper = _storage_interval(
        variant,
        target_input_bytes,
        parent_input_bytes,
    )
    candidates: dict[tuple[int, int], SolvedCandidate] = {}
    orientations = (
        (True, profile.slot_a, profile.slot_b, profile.x_bytes, profile.y_bytes),
        (False, profile.slot_b, profile.slot_a, profile.y_bytes, profile.x_bytes),
    )
    for power_is_a, fixed_slot, solved_slot, fixed_linear, solved_linear in orientations:
        solved_minimum = solved_slot.old_value + 1
        fixed_denominator = fixed_linear + profile.xy_bytes * solved_minimum
        if fixed_denominator <= 0:
            continue
        fixed_maximum = (storage_upper - profile.constant_bytes - solved_linear * solved_minimum) // fixed_denominator
        for power_value in _powers_between(fixed_slot.old_value, fixed_maximum):
            solved_slope = solved_linear + profile.xy_bytes * power_value
            solved_fixed = profile.constant_bytes + fixed_linear * power_value
            if solved_slope <= 0:
                continue
            solved_lower = max(
                solved_minimum,
                _ceil_div(storage_lower - solved_fixed, solved_slope),
            )
            solved_upper = (storage_upper - solved_fixed) // solved_slope
            if solved_lower > solved_upper:
                continue
            quotient = (target_input_bytes - solved_fixed) // solved_slope
            raw_values = {
                solved_lower,
                solved_upper,
                max(solved_lower, min(solved_upper, quotient)),
                max(solved_lower, min(solved_upper, quotient + 1)),
            }
            solved_values: set[int] = set()
            for value in raw_values:
                for nearby in (value - 1, value, value + 1):
                    if solved_lower <= nearby <= solved_upper and not _is_power_of_two(nearby):
                        solved_values.add(nearby)
            for solved_value in solved_values:
                value_a, value_b = (power_value, solved_value) if power_is_a else (solved_value, power_value)
                candidate = _candidate_from_values(
                    parent_code,
                    entry_point,
                    profile,
                    variant=variant,
                    target_input_bytes=target_input_bytes,
                    parent_input_bytes=parent_input_bytes,
                    value_a=value_a,
                    value_b=value_b,
                )
                if candidate is not None:
                    candidates[(value_a, value_b)] = candidate
    ranked = sorted(
        candidates.values(),
        key=lambda candidate: _candidate_key(candidate, parent_uuid),
    )
    return ranked[:MAX_CANDIDATES_PER_PAIR]


def _candidate_key(candidate: SolvedCandidate, parent_uuid: str) -> tuple[Any, ...]:
    error_bucket = max(
        1,
        candidate.target_input_bytes // TARGET_ERROR_EQUIVALENCE_DENOMINATOR,
    )
    pair_id = ":".join(
        sorted(
            (
                candidate.profile.slot_a.slot_id,
                candidate.profile.slot_b.slot_id,
            )
        )
    )
    tie = _sha256_bytes(
        f"{parent_uuid}:{candidate.variant}:{pair_id}:" f"{candidate.value_a}:{candidate.value_b}".encode()
    )
    return (
        abs(candidate.target_delta_bytes) // error_bucket,
        candidate.relative_growth_ratio,
        candidate.maximum_dimension_ratio,
        abs(candidate.target_delta_bytes),
        tie,
    )


def _diverse_candidate_attempts(
    candidates: Sequence[SolvedCandidate],
) -> list[SolvedCandidate]:
    """Try different slot pairs before alternate values for the same pair."""

    first_by_pair: list[SolvedCandidate] = []
    remaining: list[SolvedCandidate] = []
    seen_pairs: set[tuple[str, str]] = set()
    for candidate in candidates:
        pair = tuple(
            sorted(
                (
                    candidate.profile.slot_a.slot_id,
                    candidate.profile.slot_b.slot_id,
                )
            )
        )
        if pair in seen_pairs:
            remaining.append(candidate)
        else:
            seen_pairs.add(pair)
            first_by_pair.append(candidate)
    return first_by_pair + remaining


def _slot_assignment_manifest(slot: ShapeSlot, new_value: int) -> dict[str, Any]:
    return {
        "slot": slot.as_dict(),
        "new_value": new_value,
        "power_of_two": _is_power_of_two(new_value),
    }


def _candidate_manifest(candidate: SolvedCandidate) -> dict[str, Any]:
    profile = candidate.profile
    return {
        "kind": "strict_get_inputs_structural_two_slot",
        "variant": candidate.variant,
        "target_input_bytes": candidate.target_input_bytes,
        "input_bytes_after": candidate.input_bytes_after,
        "target_delta_bytes": candidate.target_delta_bytes,
        "target_relative_error": candidate.target_relative_error,
        "slots": [
            _slot_assignment_manifest(profile.slot_a, candidate.value_a),
            _slot_assignment_manifest(profile.slot_b, candidate.value_b),
        ],
        "changed_occurrences": candidate.changed_occurrences,
        "power_of_two_occurrences": candidate.power_of_two_occurrences,
        "power_of_two_occurrence_fraction": (candidate.power_of_two_occurrences / candidate.changed_occurrences),
        "relative_growth_ratio": float(candidate.relative_growth_ratio),
        "maximum_dimension_ratio": float(candidate.maximum_dimension_ratio),
        "storage_polynomial": {
            "a": profile.constant_bytes,
            "b": profile.x_bytes,
            "c": profile.y_bytes,
            "d": profile.xy_bytes,
            "expression": "a + b*x + c*y + d*x*y",
        },
        "dimension_balance_guard": copy.deepcopy(dict(candidate.balance_evidence)),
    }


def _review_markdown(records: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Multidimensional shape solver accepted diffs",
        "",
        "Representative two-slot source-span-only edits. These are static/FakeTensor candidates, not H20 acceptance.",
        "",
    ]
    for record in records[:MAX_REVIEW_EXAMPLES]:
        lines.extend(
            [
                f"## {record['child_uuid']} ({record['variant']})",
                "",
                (
                    f"Parent `{record['parent_uuid']}`; slots "
                    f"`{', '.join(record['slot_ids'])}`; "
                    f"{record['input_bytes_before']} -> {record['input_bytes_after']} bytes; "
                    f"target error {record['target_relative_error']:.6%}."
                ),
                "",
                "```diff",
            ]
        )
        lines.extend(
            difflib.unified_diff(
                str(record["parent_code"]).splitlines(),
                str(record["child_code"]).splitlines(),
                fromfile="parent.py",
                tofile="child.py",
                lineterm="",
            )
        )
        lines.extend(["```", ""])
    return "\n".join(lines).rstrip() + "\n"


def solve_multidim_shape_coverage(
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
    targets: list[dict[str, Any]] = []
    target_map_records: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    review_records: list[dict[str, Any]] = []
    counters: collections.Counter[str] = collections.Counter()
    skip_reasons: collections.Counter[str] = collections.Counter()

    try:
        if selected.num_rows == source.num_rows:
            shutil.copy2(selected_path, temporary_paths.selected)
        else:
            pq.write_table(selected, temporary_paths.selected, compression="zstd")

        source_selection_path = selected_path.with_name("selection.json")
        source_selection: dict[str, Any] = {}
        if source_selection_path.is_file():
            loaded = json.loads(source_selection_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("source_selection_manifest_must_be_an_object")
            source_selection = loaded
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
        temporary_paths.selection.write_text(
            json.dumps(selection, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        for source_row_index, parent in enumerate(rows):
            counters["parents_scanned"] += 1
            parent_uuid = str(_nested(parent, "extra_info.uuid", ""))
            parent_code = _nested(parent, "reward_model.ground_truth")
            entry_point = str(_nested(parent, "extra_info.entry_point", "Model"))
            if not parent_uuid or not isinstance(parent_code, str):
                raise ValueError(f"parent_requires_uuid_and_reference:{source_row_index}")
            parent_reference_hash = _sha256_bytes(parent_code.encode("utf-8"))
            parent_analysis = analyze_code(parent_code, entry_point)
            variant, target, lower_mib, upper_mib = _variant_and_target(
                parent_uuid,
                parent_analysis.input_bytes,
            )
            target_record = {
                "selected_index": source_row_index,
                "parent_uuid": parent_uuid,
                "parent_reference_sha256": parent_reference_hash,
                "parent_input_bytes": parent_analysis.input_bytes,
                "variant": variant,
                "target_lower_mib": lower_mib,
                "target_upper_mib": upper_mib,
                "target_input_bytes": target,
            }
            targets.append(target_record)
            target_map_records.append(
                {
                    "selected_index": source_row_index,
                    "parent_uuid": parent_uuid,
                    "parent_reference_sha256": parent_reference_hash,
                    "variant": variant,
                    "target_input_bytes": target,
                }
            )
            counters[f"assigned_{variant}"] += 1

            parent_fake = _fake_tensor_gate(
                parent_code,
                entry_point,
                timeout_seconds=fake_gate_timeout_seconds,
            )
            counters[f"parent_fake_{parent_fake.status}"] += 1
            decision: dict[str, Any] = {
                "source_row_index": source_row_index,
                "parent_uuid": parent_uuid,
                "parent_reference_sha256": parent_reference_hash,
                "variant": variant,
                "target_input_bytes": target,
                "input_bytes_before": parent_analysis.input_bytes,
                "accepted": False,
                "parent_fake_gate": parent_fake.as_dict(),
                "attempts": [],
            }
            if not parent_fake.passed:
                decision["reason"] = f"parent_fake_gate_{parent_fake.status}:{parent_fake.reason}"
                skip_reasons[str(decision["reason"])] += 1
                decisions.append(decision)
                continue

            try:
                slots, guard_rejections = _shape_slots_with_rejections(
                    parent_code,
                    entry_point,
                )
                affine_profiles, affine_rejections = _affine_profiles(
                    parent_code,
                    entry_point,
                    parent_analysis.input_bytes,
                    slots,
                )
                bilinear_profiles, pair_rejections = _bilinear_profiles(
                    parent_code,
                    entry_point,
                    parent_analysis.input_bytes,
                    affine_profiles,
                )
            except (SyntaxError, TypeError, ValueError) as exc:
                slots = []
                affine_profiles = []
                bilinear_profiles = []
                guard_rejections = []
                affine_rejections = []
                pair_rejections = [{"slot_ids": [], "reason": f"{type(exc).__name__}:{exc}"}]
            decision.update(
                {
                    "slot_count": len(slots),
                    "affine_slot_count": len(affine_profiles),
                    "structural_bilinear_pair_count": len(bilinear_profiles),
                    "slot_rejections": guard_rejections + affine_rejections,
                    "pair_rejections": pair_rejections,
                }
            )
            counters["slots_found"] += len(slots)
            counters["affine_slots"] += len(affine_profiles)
            counters["strict_structural_bilinear_pairs"] += len(bilinear_profiles)

            candidates: list[SolvedCandidate] = []
            for profile in bilinear_profiles:
                candidates.extend(
                    _solve_profile(
                        parent_code,
                        entry_point,
                        profile,
                        parent_uuid=parent_uuid,
                        variant=variant,
                        target_input_bytes=target,
                        parent_input_bytes=parent_analysis.input_bytes,
                    )
                )
            candidates.sort(key=lambda item: _candidate_key(item, parent_uuid))
            candidates = _diverse_candidate_attempts(candidates)
            counters["static_solved_candidates"] += len(candidates)
            if not candidates:
                decision["reason"] = "no_strict_two_slot_exactly_one_power_solution"
                skip_reasons[str(decision["reason"])] += 1
                decisions.append(decision)
                continue

            parent_sections = _section_hashes(ast.parse(parent_code), entry_point)
            for candidate in candidates[:MAX_CHILD_FAKE_ATTEMPTS]:
                attempt = _candidate_manifest(candidate)
                try:
                    child_sections = _section_hashes(
                        ast.parse(candidate.child_code),
                        entry_point,
                    )
                    if parent_sections != child_sections:
                        raise ValueError("model_or_get_init_inputs_changed")
                    static = static_gate(
                        parent_code,
                        candidate.child_code,
                        entry_point,
                    )
                    if int(static["input_bytes_after"]) != candidate.input_bytes_after:
                        raise ValueError("static_gate_storage_mismatch")
                    _validate_variant_storage(variant, candidate.input_bytes_after)
                    relative_error = _validate_target_proximity(
                        candidate.input_bytes_after,
                        target,
                    )
                    fake = _fake_tensor_gate(
                        candidate.child_code,
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
                    child = _make_child(parent, candidate.child_code, static)
                    child_uuid = str(_nested(child, "extra_info.uuid"))
                    attempt["accepted"] = True
                    solver_evidence = _candidate_manifest(candidate)
                    children.append(child)
                    paired.extend((copy.deepcopy(parent), child))
                    decision.update(
                        {
                            "accepted": True,
                            "child_uuid": child_uuid,
                            "child_reference_sha256": static["child_reference_sha256"],
                            "child_normalized_ast_sha256": static["child_normalized_ast_sha256"],
                            "input_bytes_after": static["input_bytes_after"],
                            "input_scale": static["input_scale"],
                            "target_delta_bytes": candidate.target_delta_bytes,
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
                            "slot_ids": [
                                candidate.profile.slot_a.slot_id,
                                candidate.profile.slot_b.slot_id,
                            ],
                            "input_bytes_before": parent_analysis.input_bytes,
                            "input_bytes_after": candidate.input_bytes_after,
                            "target_relative_error": relative_error,
                            "parent_code": parent_code,
                            "child_code": candidate.child_code,
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
                reasons = [str(attempt.get("reason")) for attempt in decision["attempts"]]
                decision["reason"] = reasons[-1] if reasons else "all_candidate_attempts_rejected"
                skip_reasons[str(decision["reason"])] += 1
            decisions.append(decision)

        if len(decisions) != len(rows) or len(targets) != len(rows):
            raise ValueError("one_decision_and_target_per_parent_contract_failed")
        accepted = [decision for decision in decisions if decision["accepted"]]
        if len({decision["parent_uuid"] for decision in accepted}) != len(accepted):
            raise ValueError("more_than_one_accepted_child_per_parent")
        changed_occurrences = sum(int(_nested(decision, "solver.changed_occurrences", 0)) for decision in accepted)
        power_occurrences = sum(int(_nested(decision, "solver.power_of_two_occurrences", 0)) for decision in accepted)
        if accepted and 2 * power_occurrences != changed_occurrences:
            raise ValueError("aggregate_power_of_two_occurrence_contract_failed")
        if any(int(_nested(decision, "solver.changed_occurrences", 0)) < 2 for decision in accepted):
            raise ValueError("single_changed_occurrence_contract_failed")

        pq.write_table(
            pa.Table.from_pylist(targets, schema=TARGET_SCHEMA),
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
        temporary_paths.review.write_text(
            _review_markdown(review_records),
            encoding="utf-8",
        )

        counters["parents_with_children"] = len(accepted)
        counters["children_written"] = len(children)
        counters["paired_rows_written"] = len(paired)
        target_map_sha256 = _logical_target_map_sha256(target_map_records)
        manifest = {
            "contract_version": CONTRACT_VERSION,
            "generator_version": GENERATOR_VERSION,
            "solver_source_path": str(Path(__file__).resolve()),
            "solver_source_sha256": _sha256_file(Path(__file__)),
            "v3_helper_source_path": str((_REPO_ROOT / "tools/data/synthesize/solve_shape_coverage.py").resolve()),
            "v3_helper_source_sha256": _sha256_file(_REPO_ROOT / "tools/data/synthesize/solve_shape_coverage.py"),
            "method_boundary": (
                "one stable-hash Medium/Large target per parent; strict get_inputs "
                "structural two-slot positive bilinear solver; exactly one new slot "
                "value is a power of two; every final direct-input factory satisfies "
                "the dimension-balance guard; static/FakeTensor review only"
            ),
            "source_edit_contract": (
                "simultaneously replace two disjoint existing positive integer token "
                "spans inside get_inputs; preserve Model and get_init_inputs exactly"
            ),
            "target_contract": {
                "target_salt": TARGET_BYTE_SALT,
                "sampling_granularity_bytes": 1,
                "sampling_interval": "inclusive byte range",
                "variant_assignment_salt": VARIANT_ASSIGNMENT_SALT,
                "variant_assignment": "sha256 modulo 2",
                "one_variant_per_parent": True,
                "variants_per_parent": 1,
                "logical_target_map_sha256": target_map_sha256,
                "logical_target_map_order": "selected_index ascending",
                "logical_target_record_fields": [
                    "selected_index",
                    "parent_uuid",
                    "parent_reference_sha256",
                    "variant",
                    "target_input_bytes",
                ],
                "medium_bytes": [MEDIUM_INPUT_MIN_BYTES, MEDIUM_INPUT_MAX_BYTES],
                "large_bytes": [MEDIUM_INPUT_MAX_BYTES + 1, LARGE_INPUT_MAX_BYTES],
                "minimum_input_scale": MIN_INPUT_SCALE,
                "target_relative_error_maximum": 0.25,
            },
            "pair_contract": {
                "scope": "strict get_inputs fields structural",
                "factory_set": "same non-empty set for both slots",
                "per_factory": "each slot occurs exactly once on distinct axes",
                "patch_spans": "disjoint and physically inside get_inputs",
                "storage_model": "exact positive a + b*x + c*y + d*x*y",
                "slot_values": "both strictly greater than their parent values",
            },
            "distribution_contract": {
                "dimension_occurrence": "one changed direct input-factory axis",
                "single_changed_occurrence_child_maximum_fraction": 0.10,
                "constructed_single_changed_occurrence_child_fraction": 0.0,
                "power_of_two_occurrence_fraction_range": [0.30, 0.50],
                "constructed_power_of_two_occurrence_fraction": 0.50,
                "local_proof": (
                    "for k shared affected factories the two structural slots contribute "
                    "k occurrences each; exactly one new slot value is a power of two"
                ),
                "runtime_subset_closed": True,
            },
            "candidate_ranking_contract": [
                "smaller_target_error_0.1_percent_bucket",
                "smaller_relative_slot_growth_ratio",
                "smaller_maximum_final_dimension_ratio",
                "smaller_absolute_target_error",
                "deterministic_hash",
            ],
            "dimension_balance_guard_contract": {
                "scope": "every final rank>=2 direct input factory in each child",
                "comparison": "largest_dimension <= 1000 * second_largest_dimension",
                "ratio_limit": MAX_DIMENSION_IMBALANCE_RATIO,
                "unresolved": "fail_closed",
                "post_patch_independent_reparse": True,
            },
            "fake_gate_contract": {
                "implementation": "solve_shape_coverage._fake_tensor_gate",
                "parent_and_child_required": True,
                "timeout_seconds_per_invocation": fake_gate_timeout_seconds,
                "maximum_child_attempts_per_parent": MAX_CHILD_FAKE_ATTEMPTS,
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
        temporary_paths.manifest.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
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
        help="process only the first N selected parents for smoke runs",
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
    manifest = solve_multidim_shape_coverage(
        args.selected,
        args.run_dir,
        max_parents=args.max_parents,
        fake_gate_timeout_seconds=args.fake_gate_timeout_seconds,
    )
    print(
        json.dumps(
            {
                "run_dir": str(args.run_dir.resolve()),
                "selected_rows": manifest["selected_rows"],
                "counts": manifest["counts"],
                "training_approved": manifest["training_approved"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
