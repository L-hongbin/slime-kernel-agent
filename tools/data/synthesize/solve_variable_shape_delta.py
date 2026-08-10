#!/usr/bin/env python3
"""Generate variable-cardinality shape children for the strict-pair delta lane.

This solver consumes only the compact parent parquet produced by
``prepare_variable_shape_delta.py``.  It deliberately leaves the completed
``shape_multidim_solver_v4`` run untouched.

Each candidate changes two through five disjoint logical shape slots.  Slots
may be positive integer tokens inside ``get_inputs`` or module-scope linked
names whose discovery proof shows that every load is a direct input-shape
dimension.  All slots in a group affect the same non-empty factory set once
per factory on distinct axes.  That structure gives the exact storage model
``constant + coefficient * product(slot values)``.

The generic mode prefers the largest supported group for each parent.  The
opt-in nonleading scope first tries groups that touch neither a leading axis
nor an explicit batch symbol, then uses a bounded generic fallback.  Power-of-
two values are independent per-slot soft preferences, not a per-child pattern
or acceptance condition.  Static output is not training-approved: the
aggregate 30%-50% power-of-two occurrence range must be re-audited after
target-GPU filtering.
"""

from __future__ import annotations

import argparse
import ast
import collections
import copy
import dataclasses
import difflib
import itertools
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

from tools.data.synthesize.augment_prompt_tasks import _section_hashes, analyze_code  # noqa: E402
from tools.data.synthesize.shape_contract import (  # noqa: E402
    LARGE_INPUT_MAX_BYTES,
    MEDIUM_INPUT_MAX_BYTES,
    MEDIUM_INPUT_MIN_BYTES,
    MIN_INPUT_SCALE,
    _validate_target_proximity,
    _validate_variant_storage,
    static_gate,
)
from tools.data.synthesize.solve_multidim_shape_coverage import (  # noqa: E402
    TARGET_SCHEMA,
    _affected_shape_balance_guard,
    _logical_target_map_sha256,
    _nested,
    _patch_integer_spans_many,
    _run_paths,
    _sha256_bytes,
    _sha256_file,
    _spans_overlap,
    _storage_interval,
    _variant_and_target,
)
from tools.data.synthesize.solve_shape_coverage import (  # noqa: E402
    DEFAULT_FAKE_GATE_TIMEOUT_SECONDS,
    MAX_DIMENSION_IMBALANCE_RATIO,
    AffineProfile,
    ShapeSlot,
    _affine_profiles,
    _fake_tensor_gate,
    _make_child,
    _shape_slots_with_rejections,
)

CONTRACT_VERSION = "shape_variable_multislot_solver_v5"
NONBATCH_CONTRACT_VERSION = "shape_variable_multislot_solver_v6"
BALANCED_NONBATCH_CONTRACT_VERSION = "shape_variable_multislot_solver_v7"
GENERATOR_VERSION = "same_factory_product_variable_2_to_5_soft_p2_50_v3"
NONBATCH_GENERATOR_VERSION = "same_factory_product_variable_2_to_5_nonleading_no_explicit_batch_soft_p2_50_v1"
BALANCED_NONBATCH_GENERATOR_VERSION = "same_factory_product_variable_2_to_5_balanced_nonleading_soft_p2_50_v1"
DELTA_SELECTION_CONTRACT_VERSION = "shape_variable_multislot_delta_selection_v1"
DEFAULT_SELECTED = (
    _REPO_ROOT / "Data/prompt_tvm_v4/shape_solver_variable_multislot_v5/input.recoverable12318/selected.parquet"
)
DEFAULT_RUN_DIR = _REPO_ROOT / "Data/prompt_tvm_v4/shape_solver_variable_multislot_v5/run.recoverable12318"
MIN_GROUP_SLOTS = 2
MAX_GROUP_SLOTS = 5
MAX_GROUPS_PER_SIZE = 12
MIN_GENERAL_GROUPS_PER_SIZE = 4
MAX_CANDIDATES_PER_GROUP = 4
MAX_CHILD_FAKE_ATTEMPTS = 6
MAX_SCOPE_CHILD_FAKE_ATTEMPTS = 4
MAX_SCOPE_FALLBACK_CHILD_FAKE_ATTEMPTS = 2
MAX_PREFERRED_CARDINALITY_ATTEMPTS = 3
MAX_REVIEW_EXAMPLES = 20
POWER_OF_TWO_PREFERENCE_NUMERATOR = 1
POWER_OF_TWO_PREFERENCE_DENOMINATOR = 2
TARGET_ERROR_EQUIVALENCE_DENOMINATOR = 1_000
GENERIC_GROUP_SCOPE = "generic"
NONLEADING_NO_EXPLICIT_BATCH_GROUP_SCOPE = "nonleading_no_explicit_batch"
BALANCED_NONLEADING_NO_EXPLICIT_BATCH_GROUP_SCOPE = "balanced_nonleading_no_explicit_batch"
GROUP_SCOPE_CHOICES = (
    GENERIC_GROUP_SCOPE,
    NONLEADING_NO_EXPLICIT_BATCH_GROUP_SCOPE,
    BALANCED_NONLEADING_NO_EXPLICIT_BATCH_GROUP_SCOPE,
)
EXPLICIT_BATCH_SYMBOLS = frozenset({"batch_size", "batchsize", "batch", "bs", "n_batch"})


@dataclasses.dataclass(frozen=True)
class ProductProfile:
    slots: tuple[ShapeSlot, ...]
    factory_indices: tuple[int, ...]
    constant_bytes: int
    product_coefficient: int

    def input_bytes(self, values: Sequence[int]) -> int:
        if len(values) != len(self.slots):
            raise ValueError("product_profile_value_count_mismatch")
        return self.constant_bytes + self.product_coefficient * math.prod(values)


@dataclasses.dataclass(frozen=True)
class VariableCandidate:
    profile: ProductProfile
    variant: str
    target_input_bytes: int
    values: tuple[int, ...]
    input_bytes_after: int
    target_delta_bytes: int
    target_relative_error: float
    changed_occurrences: int
    power_of_two_occurrences: int
    preference_mismatch_occurrences: int
    relative_growth_spread: Fraction
    maximum_dimension_ratio: Fraction
    balance_evidence: Mapping[str, Any]
    child_code: str


def _is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError("ceil_div_requires_positive_denominator")
    return -((-numerator) // denominator)


def _slot_factory_axes(slot: ShapeSlot) -> dict[int, list[int]]:
    result: dict[int, list[int]] = collections.defaultdict(list)
    for occurrence in slot.occurrences:
        result[occurrence.factory_index].append(occurrence.axis)
    return dict(result)


def _group_scope_evidence(slots: Sequence[ShapeSlot]) -> dict[str, Any]:
    leading_occurrences = sum(occurrence.axis == 0 for slot in slots for occurrence in slot.occurrences)
    explicit_batch_symbols = sorted(
        {
            slot.symbol_name.strip().lower()
            for slot in slots
            if isinstance(slot.symbol_name, str) and slot.symbol_name.strip().lower() in EXPLICIT_BATCH_SYMBOLS
        }
    )
    return {
        "touches_leading_axis": leading_occurrences > 0,
        "leading_axis_occurrences": leading_occurrences,
        "touches_explicit_batch_symbol": bool(explicit_batch_symbols),
        "explicit_batch_symbols": explicit_batch_symbols,
        "matches_nonleading_no_explicit_batch": (leading_occurrences == 0 and not explicit_batch_symbols),
    }


def _slots_match_scope(
    slots: Sequence[ShapeSlot],
    group_scope: str,
) -> bool:
    if group_scope == GENERIC_GROUP_SCOPE:
        return True
    if group_scope in {
        NONLEADING_NO_EXPLICIT_BATCH_GROUP_SCOPE,
        BALANCED_NONLEADING_NO_EXPLICIT_BATCH_GROUP_SCOPE,
    }:
        evidence = _group_scope_evidence(slots)
        return bool(evidence["matches_nonleading_no_explicit_batch"])
    raise ValueError(f"unknown_group_scope:{group_scope}")


def _group_matches_scope(
    profiles: Sequence[AffineProfile],
    group_scope: str,
) -> bool:
    return _slots_match_scope(
        tuple(profile.slot for profile in profiles),
        group_scope,
    )


def relaxed_structural_pair(slot_a: ShapeSlot, slot_b: ShapeSlot) -> bool:
    """The old strict pair contract without its lexical inside-function rule."""

    if slot_a.slot_id == slot_b.slot_id:
        return False
    if any(_spans_overlap(left, right) for left in slot_a.patch_spans for right in slot_b.patch_spans):
        return False
    by_factory_a = _slot_factory_axes(slot_a)
    by_factory_b = _slot_factory_axes(slot_b)
    if not by_factory_a or set(by_factory_a) != set(by_factory_b):
        return False
    for factory_index in by_factory_a:
        axes_a = by_factory_a[factory_index]
        axes_b = by_factory_b[factory_index]
        if len(axes_a) != 1 or len(axes_b) != 1 or axes_a[0] == axes_b[0]:
            return False
    return len(slot_a.occurrences) == len(slot_b.occurrences)


def _structural_group(slots: Sequence[ShapeSlot]) -> bool:
    if not MIN_GROUP_SLOTS <= len(slots) <= MAX_GROUP_SLOTS:
        return False
    if len({slot.slot_id for slot in slots}) != len(slots):
        return False
    if any(
        _spans_overlap(left, right)
        for index, slot in enumerate(slots)
        for other in slots[index + 1 :]
        for left in slot.patch_spans
        for right in other.patch_spans
    ):
        return False
    factory_axes = [_slot_factory_axes(slot) for slot in slots]
    common_factories = set(factory_axes[0]) if factory_axes else set()
    if not common_factories or any(set(value) != common_factories for value in factory_axes):
        return False
    for value in factory_axes:
        if any(len(value[factory]) != 1 for factory in common_factories):
            return False
    for factory in common_factories:
        axes = [value[factory][0] for value in factory_axes]
        if len(axes) != len(set(axes)):
            return False
    return True


def variable_slot_group_inventory(
    affine_profiles: Sequence[AffineProfile],
    *,
    parent_uuid: str,
    group_scope: str = GENERIC_GROUP_SCOPE,
) -> tuple[list[tuple[AffineProfile, ...]], dict[str, dict[str, int]]]:
    """Enumerate bounded groups and retain auditable scope counts around the cap."""

    if group_scope not in GROUP_SCOPE_CHOICES:
        raise ValueError(f"unknown_group_scope:{group_scope}")

    buckets: dict[tuple[int, ...], list[AffineProfile]] = collections.defaultdict(list)
    for profile in affine_profiles:
        axes = _slot_factory_axes(profile.slot)
        if axes and all(len(value) == 1 for value in axes.values()):
            buckets[tuple(sorted(axes))].append(profile)

    by_size: dict[int, list[tuple[AffineProfile, ...]]] = collections.defaultdict(list)
    seen: set[tuple[str, ...]] = set()
    for factory_indices, bucket in sorted(buckets.items()):
        del factory_indices
        ordered = sorted(bucket, key=lambda value: value.slot.slot_id)
        maximum = min(MAX_GROUP_SLOTS, len(ordered))
        for size in range(MIN_GROUP_SLOTS, maximum + 1):
            for group in itertools.combinations(ordered, size):
                slots = tuple(profile.slot for profile in group)
                if not _structural_group(slots):
                    continue
                identity = tuple(sorted(slot.slot_id for slot in slots))
                if identity in seen:
                    continue
                seen.add(identity)
                by_size[size].append(group)

    result: list[tuple[AffineProfile, ...]] = []
    scope_before_cap: dict[str, int] = {}
    scope_after_cap: dict[str, int] = {}
    scope_discarded_by_cap: dict[str, int] = {}
    general_before_cap: dict[str, int] = {}
    general_after_cap: dict[str, int] = {}
    general_discarded_by_cap: dict[str, int] = {}
    for size in range(MAX_GROUP_SLOTS, MIN_GROUP_SLOTS - 1, -1):
        ranked = sorted(
            by_size.get(size, []),
            key=lambda group: _sha256_bytes(
                (f"{parent_uuid}:{size}:" + ":".join(sorted(profile.slot.slot_id for profile in group))).encode(
                    "utf-8"
                )
            ),
        )
        if group_scope == GENERIC_GROUP_SCOPE:
            retained = ranked[:MAX_GROUPS_PER_SIZE]
        else:
            matching = [group for group in ranked if _group_matches_scope(group, group_scope)]
            fallback = [group for group in ranked if not _group_matches_scope(group, group_scope)]
            reserved_general = min(MIN_GENERAL_GROUPS_PER_SIZE, len(fallback))
            maximum_scope = MAX_GROUPS_PER_SIZE - reserved_general
            retained_matching = matching[:maximum_scope]
            retained_fallback = fallback[: MAX_GROUPS_PER_SIZE - len(retained_matching)]
            retained = retained_matching + retained_fallback
            key = str(size)
            scope_before_cap[key] = len(matching)
            scope_after_cap[key] = len(retained_matching)
            scope_discarded_by_cap[key] = len(matching) - len(retained_matching)
            general_before_cap[key] = len(fallback)
            general_after_cap[key] = len(retained_fallback)
            general_discarded_by_cap[key] = len(fallback) - len(retained_fallback)
        result.extend(retained)
    return result, {
        "scope_group_count_before_cap_by_logical_slot_count": scope_before_cap,
        "scope_group_count_after_cap_by_logical_slot_count": scope_after_cap,
        "scope_group_count_discarded_by_cap_by_logical_slot_count": (scope_discarded_by_cap),
        "general_group_count_before_cap_by_logical_slot_count": general_before_cap,
        "general_group_count_after_cap_by_logical_slot_count": general_after_cap,
        "general_group_count_discarded_by_cap_by_logical_slot_count": (general_discarded_by_cap),
    }


def variable_slot_groups(
    affine_profiles: Sequence[AffineProfile],
    *,
    parent_uuid: str,
    group_scope: str = GENERIC_GROUP_SCOPE,
) -> list[tuple[AffineProfile, ...]]:
    groups, _inventory = variable_slot_group_inventory(
        affine_profiles,
        parent_uuid=parent_uuid,
        group_scope=group_scope,
    )
    return groups


def _product_profile(
    code: str,
    entry_point: str,
    parent_input_bytes: int,
    profiles: Sequence[AffineProfile],
) -> ProductProfile:
    slots = tuple(profile.slot for profile in profiles)
    if not _structural_group(slots):
        raise ValueError("slots_do_not_form_a_structural_product_group")
    old_values = tuple(slot.old_value for slot in slots)
    old_product = math.prod(old_values)
    coefficients: set[Fraction] = set()
    for index, profile in enumerate(profiles):
        other_product = old_product // old_values[index]
        coefficients.add(Fraction(profile.bytes_per_slot_unit, other_product))
    if len(coefficients) != 1:
        raise ValueError("slot_affine_slopes_disagree_on_product_coefficient")
    coefficient_fraction = next(iter(coefficients))
    if coefficient_fraction.denominator != 1 or coefficient_fraction <= 0:
        raise ValueError("product_coefficient_must_be_a_positive_integer")
    coefficient = coefficient_fraction.numerator
    constant = parent_input_bytes - coefficient * old_product
    if constant < 0:
        raise ValueError("product_profile_constant_is_negative")
    profile = ProductProfile(
        slots=slots,
        factory_indices=tuple(sorted(_slot_factory_axes(slots[0]))),
        constant_bytes=constant,
        product_coefficient=coefficient,
    )
    if profile.input_bytes(old_values) != parent_input_bytes:
        raise ValueError("product_profile_parent_reconstruction_failed")

    probes = [
        tuple(value + 1 for value in old_values),
        tuple(value + 1 + (index % 2) for index, value in enumerate(old_values)),
    ]
    for values in probes:
        child = _patch_integer_spans_many(code, tuple(zip(slots, values, strict=True)))
        observed = analyze_code(child, entry_point).input_bytes
        if profile.input_bytes(values) != observed:
            raise ValueError("shape_group_storage_is_not_exact_product_affine")
    return profile


def _power_preference(parent_uuid: str, slot_id: str) -> bool:
    digest = _sha256_bytes(f"shape-variable-p2-v1:{parent_uuid}:{slot_id}".encode())
    return int(digest[:16], 16) % POWER_OF_TWO_PREFERENCE_DENOMINATOR < POWER_OF_TWO_PREFERENCE_NUMERATOR


def _nearest_powers(value: int, minimum: int, maximum: int) -> set[int]:
    if minimum > maximum or maximum <= 0:
        return set()
    anchor = max(1, value)
    lower = 1 << max(0, anchor.bit_length() - 1)
    result: set[int] = set()
    for candidate in (lower >> 1, lower, lower << 1, lower << 2):
        if minimum <= candidate <= maximum:
            result.add(candidate)
    return result


def _non_power_near(value: int, minimum: int, maximum: int) -> set[int]:
    result: set[int] = set()
    for candidate in range(value - 2, value + 3):
        if minimum <= candidate <= maximum and not _is_power_of_two(candidate):
            result.add(candidate)
    return result


def _rounded_nonpivot_value(
    raw: float,
    slot: ShapeSlot,
    *,
    prefer_power: bool,
    salt: str,
) -> int:
    minimum = slot.old_value + 1
    rounded = max(minimum, int(round(raw)))
    if prefer_power:
        choices = _nearest_powers(rounded, minimum, max(minimum, rounded * 2))
        if choices:
            return min(
                choices,
                key=lambda value: (
                    abs(value - rounded),
                    _sha256_bytes(f"{salt}:{value}".encode()),
                ),
            )
    if _is_power_of_two(rounded):
        alternatives = _non_power_near(rounded, minimum, rounded + 2)
        if alternatives:
            return min(
                alternatives,
                key=lambda value: (
                    abs(value - rounded),
                    _sha256_bytes(f"{salt}:{value}".encode()),
                ),
            )
    return rounded


def _candidate_from_values(
    parent_code: str,
    entry_point: str,
    profile: ProductProfile,
    *,
    parent_uuid: str,
    variant: str,
    target_input_bytes: int,
    parent_input_bytes: int,
    values: Sequence[int],
    rejection_counts: collections.Counter[str],
) -> VariableCandidate | None:
    values_tuple = tuple(int(value) for value in values)
    if len(values_tuple) != len(profile.slots) or any(
        value <= slot.old_value for slot, value in zip(profile.slots, values_tuple, strict=True)
    ):
        rejection_counts["not_a_strict_expansion"] += 1
        return None
    input_bytes_after = profile.input_bytes(values_tuple)
    assignments = tuple(zip(profile.slots, values_tuple, strict=True))
    try:
        _validate_variant_storage(variant, input_bytes_after)
        relative_error = _validate_target_proximity(input_bytes_after, target_input_bytes)
        if input_bytes_after < math.ceil(MIN_INPUT_SCALE * parent_input_bytes):
            raise ValueError("minimum_input_growth_not_met")
        child_code = _patch_integer_spans_many(parent_code, assignments)
        balance = _affected_shape_balance_guard(child_code, assignments)
        if not balance["passed"]:
            raise ValueError(f"dimension_balance_guard_rejected:{balance['reason']}")
        if analyze_code(child_code, entry_point).input_bytes != input_bytes_after:
            raise ValueError("product_solution_storage_mismatch")
    except (SyntaxError, TypeError, ValueError) as exc:
        reason = str(exc).split(":", 1)[0] or type(exc).__name__
        rejection_counts[f"{type(exc).__name__}:{reason}"] += 1
        return None

    changed_occurrences = sum(len(slot.occurrences) for slot in profile.slots)
    power_occurrences = sum(
        len(slot.occurrences)
        for slot, value in zip(profile.slots, values_tuple, strict=True)
        if _is_power_of_two(value)
    )
    mismatch_occurrences = sum(
        len(slot.occurrences)
        for slot, value in zip(profile.slots, values_tuple, strict=True)
        if _is_power_of_two(value) != _power_preference(parent_uuid, slot.slot_id)
    )
    growth = [Fraction(value, slot.old_value) for slot, value in zip(profile.slots, values_tuple, strict=True)]
    ratios = [
        Fraction(
            int(factory["largest_dimension"]),
            int(factory["second_largest_dimension"]),
        )
        for factory in balance["affected_factories"]
        if factory["guard_applies"]
    ]
    return VariableCandidate(
        profile=profile,
        variant=variant,
        target_input_bytes=target_input_bytes,
        values=values_tuple,
        input_bytes_after=input_bytes_after,
        target_delta_bytes=input_bytes_after - target_input_bytes,
        target_relative_error=relative_error,
        changed_occurrences=changed_occurrences,
        power_of_two_occurrences=power_occurrences,
        preference_mismatch_occurrences=mismatch_occurrences,
        relative_growth_spread=max(growth) / min(growth),
        maximum_dimension_ratio=max(ratios, default=Fraction(0, 1)),
        balance_evidence=balance,
        child_code=child_code,
    )


def _solve_product_profile(
    parent_code: str,
    entry_point: str,
    profile: ProductProfile,
    *,
    parent_uuid: str,
    variant: str,
    target_input_bytes: int,
    parent_input_bytes: int,
    rejection_counts: collections.Counter[str],
) -> list[VariableCandidate]:
    storage_lower, storage_upper = _storage_interval(variant, target_input_bytes, parent_input_bytes)
    coefficient = profile.product_coefficient
    desired_product = max(
        1,
        (target_input_bytes - profile.constant_bytes) // coefficient,
    )
    old_product = math.prod(slot.old_value for slot in profile.slots)
    product_ratio = max(1.0, desired_product / old_product)
    base_scale = product_ratio ** (1.0 / len(profile.slots))
    candidates: dict[tuple[int, ...], VariableCandidate] = {}

    for allocation_index, scale_exponent in enumerate((0.82, 1.0, 1.18)):
        raw_weights: list[float] = []
        for slot in profile.slots:
            digest = _sha256_bytes(
                f"shape-variable-weight-v1:{parent_uuid}:{allocation_index}:" f"{slot.slot_id}".encode()
            )
            raw_weights.append(0.85 + (int(digest[:8], 16) % 301) / 1000.0)
        mean_weight = sum(raw_weights) / len(raw_weights)
        weights = [value / mean_weight for value in raw_weights]

        pivot_order = sorted(
            range(len(profile.slots)),
            key=lambda index: _sha256_bytes(
                f"shape-variable-pivot-v1:{parent_uuid}:{allocation_index}:" f"{profile.slots[index].slot_id}".encode()
            ),
        )
        for pivot_index in pivot_order:
            values: list[int | None] = [None] * len(profile.slots)
            for index, slot in enumerate(profile.slots):
                if index == pivot_index:
                    continue
                scale = base_scale ** (scale_exponent * weights[index])
                values[index] = _rounded_nonpivot_value(
                    slot.old_value * scale,
                    slot,
                    prefer_power=_power_preference(parent_uuid, slot.slot_id),
                    salt=f"{parent_uuid}:{allocation_index}:{slot.slot_id}",
                )
            nonpivot_product = math.prod(int(value) for value in values if value is not None)
            slope = coefficient * nonpivot_product
            pivot = profile.slots[pivot_index]
            minimum = max(
                pivot.old_value + 1,
                _ceil_div(storage_lower - profile.constant_bytes, slope),
            )
            maximum = (storage_upper - profile.constant_bytes) // slope
            if minimum > maximum:
                rejection_counts["empty_pivot_storage_interval"] += 1
                continue
            quotient = (target_input_bytes - profile.constant_bytes) // slope
            raw_pivots = {
                minimum,
                maximum,
                max(minimum, min(maximum, quotient)),
                max(minimum, min(maximum, quotient + 1)),
            }
            prefer_power = _power_preference(parent_uuid, pivot.slot_id)
            if prefer_power:
                raw_pivots.update(_nearest_powers(quotient, minimum, maximum))
            else:
                raw_pivots.update(_non_power_near(quotient, minimum, maximum))
            for pivot_value in raw_pivots:
                complete = tuple(
                    pivot_value if index == pivot_index else int(value) for index, value in enumerate(values)
                )
                candidate = _candidate_from_values(
                    parent_code,
                    entry_point,
                    profile,
                    parent_uuid=parent_uuid,
                    variant=variant,
                    target_input_bytes=target_input_bytes,
                    parent_input_bytes=parent_input_bytes,
                    values=complete,
                    rejection_counts=rejection_counts,
                )
                if candidate is not None:
                    candidates[complete] = candidate

    ranked = sorted(
        candidates.values(),
        key=lambda candidate: _candidate_key(candidate, parent_uuid),
    )
    return ranked[:MAX_CANDIDATES_PER_GROUP]


def _candidate_key(candidate: VariableCandidate, parent_uuid: str) -> tuple[Any, ...]:
    error_bucket = max(
        1,
        candidate.target_input_bytes // TARGET_ERROR_EQUIVALENCE_DENOMINATOR,
    )
    identity = ":".join(slot.slot_id for slot in candidate.profile.slots)
    tie = _sha256_bytes(f"{parent_uuid}:{identity}:{candidate.values}".encode())
    return (
        -len(candidate.profile.slots),
        abs(candidate.target_delta_bytes) // error_bucket,
        candidate.preference_mismatch_occurrences,
        candidate.relative_growth_spread,
        candidate.maximum_dimension_ratio,
        abs(candidate.target_delta_bytes),
        tie,
    )


def _batch_like_growth_dominates(candidate: VariableCandidate) -> bool:
    growth = [
        math.log(value / slot.old_value) for slot, value in zip(candidate.profile.slots, candidate.values, strict=True)
    ]
    batch_like = [
        any(occurrence.axis == 0 for occurrence in slot.occurrences)
        or (isinstance(slot.symbol_name, str) and slot.symbol_name.strip().lower() in EXPLICIT_BATCH_SYMBOLS)
        for slot in candidate.profile.slots
    ]
    return bool(growth) and any(
        is_batch_like and value >= max(growth) for value, is_batch_like in zip(growth, batch_like, strict=True)
    )


def _diverse_candidate_attempts(
    candidates: Sequence[VariableCandidate],
) -> list[VariableCandidate]:
    """Try several maximum-cardinality candidates before recorded fallback."""

    if not candidates:
        return []
    maximum_size = len(candidates[0].profile.slots)
    maximum_candidates = [candidate for candidate in candidates if len(candidate.profile.slots) == maximum_size]
    preferred: list[VariableCandidate] = []
    seen_groups: set[tuple[str, ...]] = set()
    for candidate in maximum_candidates:
        group = tuple(slot.slot_id for slot in candidate.profile.slots)
        if group not in seen_groups:
            seen_groups.add(group)
            preferred.append(candidate)
    preferred_ids = {id(candidate) for candidate in preferred}
    preferred.extend(candidate for candidate in maximum_candidates if id(candidate) not in preferred_ids)
    preferred = preferred[:MAX_PREFERRED_CARDINALITY_ATTEMPTS]

    fallback: list[VariableCandidate] = []
    seen_fallback_sizes: set[int] = set()
    preferred_ids = {id(candidate) for candidate in preferred}
    for candidate in candidates:
        size = len(candidate.profile.slots)
        if size == maximum_size or size in seen_fallback_sizes:
            continue
        seen_fallback_sizes.add(size)
        fallback.append(candidate)
    selected_ids = preferred_ids | {id(candidate) for candidate in fallback}
    remaining = [candidate for candidate in candidates if id(candidate) not in selected_ids]
    return preferred + fallback + remaining


def _ordered_candidate_attempts(
    candidates: Sequence[VariableCandidate],
    *,
    parent_uuid: str,
    group_scope: str,
) -> tuple[list[VariableCandidate], str, dict[str, Any]]:
    """Return the bounded attempt order and its preferred lane."""

    if group_scope == GENERIC_GROUP_SCOPE:
        ranked = sorted(
            candidates,
            key=lambda candidate: _candidate_key(candidate, parent_uuid),
        )
        selected = _diverse_candidate_attempts(ranked)[:MAX_CHILD_FAKE_ATTEMPTS]
        return (
            selected,
            "generic",
            {
                "preference_reason": "generic_ranking",
                "scope_candidate_count_before_bound": None,
                "general_candidate_count_before_bound": None,
                "maximum_scope_logical_slot_count": None,
                "best_general_batch_like_growth_dominates": None,
            },
        )
    if group_scope not in {
        NONLEADING_NO_EXPLICIT_BATCH_GROUP_SCOPE,
        BALANCED_NONLEADING_NO_EXPLICIT_BATCH_GROUP_SCOPE,
    }:
        raise ValueError(f"unknown_group_scope:{group_scope}")

    matching = [candidate for candidate in candidates if _slots_match_scope(candidate.profile.slots, group_scope)]
    fallback = [candidate for candidate in candidates if not _slots_match_scope(candidate.profile.slots, group_scope)]

    def ranked_diverse(values: Sequence[VariableCandidate]) -> list[VariableCandidate]:
        ranked = sorted(
            values,
            key=lambda candidate: _candidate_key(candidate, parent_uuid),
        )
        return _diverse_candidate_attempts(ranked)

    ordered_matching = ranked_diverse(matching)
    ordered_fallback = ranked_diverse(fallback)
    if not ordered_matching:
        return (
            ordered_fallback[:MAX_CHILD_FAKE_ATTEMPTS],
            "general",
            {
                "preference_reason": "no_scope_static_candidate",
                "scope_candidate_count_before_bound": 0,
                "general_candidate_count_before_bound": len(ordered_fallback),
                "maximum_scope_logical_slot_count": None,
                "best_general_batch_like_growth_dominates": (
                    _batch_like_growth_dominates(ordered_fallback[0]) if ordered_fallback else None
                ),
            },
        )
    if not ordered_fallback:
        return (
            ordered_matching[:MAX_CHILD_FAKE_ATTEMPTS],
            "scope",
            {
                "preference_reason": "no_general_static_candidate",
                "scope_candidate_count_before_bound": len(ordered_matching),
                "general_candidate_count_before_bound": 0,
                "maximum_scope_logical_slot_count": max(
                    len(candidate.profile.slots) for candidate in ordered_matching
                ),
                "best_general_batch_like_growth_dominates": None,
            },
        )

    prefer_scope = group_scope == NONLEADING_NO_EXPLICIT_BATCH_GROUP_SCOPE
    maximum_scope_size = max(len(candidate.profile.slots) for candidate in ordered_matching)
    general_batch_like_growth_dominates = _batch_like_growth_dominates(ordered_fallback[0])
    preference_reason = "strict_scope_preference"
    if group_scope == BALANCED_NONLEADING_NO_EXPLICIT_BATCH_GROUP_SCOPE:
        prefer_scope = maximum_scope_size >= 3 or general_batch_like_growth_dominates
        if maximum_scope_size >= 3:
            preference_reason = "scope_cardinality_at_least_three"
        elif general_batch_like_growth_dominates:
            preference_reason = "best_general_batch_like_growth_dominates"
        else:
            preference_reason = "balanced_general_preference"
    preferred = ordered_matching if prefer_scope else ordered_fallback
    secondary = ordered_fallback if prefer_scope else ordered_matching
    selected_preferred = preferred[:MAX_SCOPE_CHILD_FAKE_ATTEMPTS]
    selected_secondary = secondary[:MAX_SCOPE_FALLBACK_CHILD_FAKE_ATTEMPTS]
    remaining = MAX_CHILD_FAKE_ATTEMPTS - len(selected_preferred) - len(selected_secondary)
    if remaining > 0:
        selected_secondary.extend(secondary[len(selected_secondary) : len(selected_secondary) + remaining])
    return (
        selected_preferred + selected_secondary,
        ("scope" if prefer_scope else "general"),
        {
            "preference_reason": preference_reason,
            "scope_candidate_count_before_bound": len(ordered_matching),
            "general_candidate_count_before_bound": len(ordered_fallback),
            "maximum_scope_logical_slot_count": maximum_scope_size,
            "best_general_batch_like_growth_dominates": (general_batch_like_growth_dominates),
        },
    )


def _slot_assignment_manifest(
    parent_uuid: str,
    slot: ShapeSlot,
    new_value: int,
) -> dict[str, Any]:
    return {
        "slot": slot.as_dict(),
        "new_value": new_value,
        "power_of_two": _is_power_of_two(new_value),
        "power_of_two_soft_preference": _power_preference(parent_uuid, slot.slot_id),
    }


def _candidate_manifest(
    parent_uuid: str,
    candidate: VariableCandidate,
) -> dict[str, Any]:
    profile = candidate.profile
    group_scope = _group_scope_evidence(profile.slots)
    return {
        "kind": "same_factory_product_variable_multislot",
        "logical_slot_count": len(profile.slots),
        "variant": candidate.variant,
        "target_input_bytes": candidate.target_input_bytes,
        "input_bytes_after": candidate.input_bytes_after,
        "target_delta_bytes": candidate.target_delta_bytes,
        "target_relative_error": candidate.target_relative_error,
        "slots": [
            _slot_assignment_manifest(parent_uuid, slot, value)
            for slot, value in zip(profile.slots, candidate.values, strict=True)
        ],
        "changed_occurrences": candidate.changed_occurrences,
        "power_of_two_occurrences": candidate.power_of_two_occurrences,
        "power_of_two_occurrence_fraction": (candidate.power_of_two_occurrences / candidate.changed_occurrences),
        "power_of_two_preference_mismatch_occurrences": (candidate.preference_mismatch_occurrences),
        "relative_growth_spread": float(candidate.relative_growth_spread),
        "maximum_dimension_ratio": float(candidate.maximum_dimension_ratio),
        "storage_polynomial": {
            "constant_bytes": profile.constant_bytes,
            "product_coefficient": profile.product_coefficient,
            "expression": "constant_bytes + product_coefficient * product(values)",
        },
        "affected_factory_indices": list(profile.factory_indices),
        "group_scope": group_scope,
        "dimension_balance_guard": copy.deepcopy(dict(candidate.balance_evidence)),
    }


def _review_markdown(records: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Variable-multislot shape solver accepted diffs",
        "",
        "Static/FakeTensor candidates from the independent recovery lane; not H20 acceptance.",
        "",
    ]
    for record in records[:MAX_REVIEW_EXAMPLES]:
        lines.extend(
            [
                (f"## {record['child_uuid']} ({record['variant']}, " f"{record['logical_slot_count']} logical slots)"),
                "",
                (
                    f"Parent `{record['parent_uuid']}`; "
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


def _delta_selection_rows(selected_path: Path, expected_count: int) -> list[Mapping[str, Any]]:
    selection_path = selected_path.with_name("selection.json")
    if not selection_path.is_file():
        raise FileNotFoundError(f"variable solver requires the delta selection manifest: {selection_path}")
    loaded = json.loads(selection_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("delta_selection_manifest_must_be_an_object")
    if loaded.get("delta_contract_version") != DELTA_SELECTION_CONTRACT_VERSION:
        raise ValueError("selected_input_is_not_the_variable_multislot_delta_lane")
    rows = loaded.get("rows")
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError("delta_selection_rows_do_not_match_selected_parquet")
    return rows


def solve_variable_shape_delta(
    selected_path: Path,
    run_dir: Path,
    *,
    max_parents: int | None = None,
    fake_gate_timeout_seconds: float = DEFAULT_FAKE_GATE_TIMEOUT_SECONDS,
    group_scope: str = GENERIC_GROUP_SCOPE,
) -> dict[str, Any]:
    if not selected_path.is_file():
        raise FileNotFoundError(selected_path)
    if run_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing run directory: {run_dir}")
    if max_parents is not None and max_parents <= 0:
        raise ValueError("max_parents must be positive")
    if not math.isfinite(fake_gate_timeout_seconds) or fake_gate_timeout_seconds <= 0:
        raise ValueError("fake_gate_timeout_seconds_must_be_finite_and_positive")
    if group_scope not in GROUP_SCOPE_CHOICES:
        raise ValueError(f"unknown_group_scope:{group_scope}")
    solver_contract_version = {
        GENERIC_GROUP_SCOPE: CONTRACT_VERSION,
        NONLEADING_NO_EXPLICIT_BATCH_GROUP_SCOPE: NONBATCH_CONTRACT_VERSION,
        BALANCED_NONLEADING_NO_EXPLICIT_BATCH_GROUP_SCOPE: (BALANCED_NONBATCH_CONTRACT_VERSION),
    }[group_scope]
    generator_version = {
        GENERIC_GROUP_SCOPE: GENERATOR_VERSION,
        NONLEADING_NO_EXPLICIT_BATCH_GROUP_SCOPE: NONBATCH_GENERATOR_VERSION,
        BALANCED_NONLEADING_NO_EXPLICIT_BATCH_GROUP_SCOPE: (BALANCED_NONBATCH_GENERATOR_VERSION),
    }[group_scope]
    scoped_mode = group_scope != GENERIC_GROUP_SCOPE

    source = pq.read_table(selected_path)
    source_selection_rows = _delta_selection_rows(selected_path, source.num_rows)
    selected = source if max_parents is None else source.slice(0, max_parents)
    rows = selected.to_pylist()
    selection_rows = source_selection_rows[: len(rows)]
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
        source_selection = json.loads(source_selection_path.read_text(encoding="utf-8"))
        selection = {
            **source_selection,
            "derivation_contract_version": solver_contract_version,
            "source_selection_manifest": str(source_selection_path.resolve()),
            "source_selection_manifest_sha256": _sha256_file(source_selection_path),
            "selected_path": str(final_paths.selected),
            "selected_sha256": _sha256_file(temporary_paths.selected),
            "selected_count": len(rows),
            "rows": selection_rows,
        }
        temporary_paths.selection.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        for source_row_index, (parent, selection_row) in enumerate(zip(rows, selection_rows, strict=True)):
            counters["parents_scanned"] += 1
            parent_uuid = str(_nested(parent, "extra_info.uuid", ""))
            parent_code = _nested(parent, "reward_model.ground_truth")
            entry_point = str(_nested(parent, "extra_info.entry_point", "Model"))
            if not parent_uuid or not isinstance(parent_code, str):
                raise ValueError(f"parent_requires_uuid_and_reference:{source_row_index}")
            delta = selection_row.get("variable_shape_delta")
            if not isinstance(delta, Mapping) or delta.get("parent_uuid") != parent_uuid:
                raise ValueError(f"delta_selection_identity_mismatch:{source_row_index}")
            parent_reference_hash = _sha256_bytes(parent_code.encode("utf-8"))
            if delta.get("parent_reference_sha256") != parent_reference_hash:
                raise ValueError(f"delta_selection_reference_mismatch:{source_row_index}")
            parent_analysis = analyze_code(parent_code, entry_point)
            variant, target, lower_mib, upper_mib = _variant_and_target(parent_uuid, parent_analysis.input_bytes)
            if variant != delta.get("variant") or target != int(delta["target_input_bytes"]):
                raise ValueError(f"delta_selection_target_mismatch:{source_row_index}")
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
                "prior_full_source_row_index": int(delta["prior_full_source_row_index"]),
                "parent_uuid": parent_uuid,
                "parent_reference_sha256": parent_reference_hash,
                "variant": variant,
                "target_input_bytes": target,
                "input_bytes_before": parent_analysis.input_bytes,
                "accepted": False,
                "parent_fake_gate": parent_fake.as_dict(),
                "group_scope_mode": group_scope,
                "group_scope_status": ("not_evaluated_parent_fake_failed" if scoped_mode else "not_applicable"),
                "attempts": [],
            }
            if not parent_fake.passed:
                decision["reason"] = f"parent_fake_gate_{parent_fake.status}:{parent_fake.reason}"
                skip_reasons[str(decision["reason"])] += 1
                decisions.append(decision)
                continue

            try:
                slots, guard_rejections = _shape_slots_with_rejections(parent_code, entry_point)
                affine_profiles, affine_rejections = _affine_profiles(
                    parent_code,
                    entry_point,
                    parent_analysis.input_bytes,
                    slots,
                )
                groups, group_scope_inventory = variable_slot_group_inventory(
                    affine_profiles,
                    parent_uuid=parent_uuid,
                    group_scope=group_scope,
                )
                product_profiles: list[ProductProfile] = []
                product_rejections: list[dict[str, Any]] = []
                for group in groups:
                    try:
                        product_profiles.append(
                            _product_profile(
                                parent_code,
                                entry_point,
                                parent_analysis.input_bytes,
                                group,
                            )
                        )
                    except (SyntaxError, TypeError, ValueError) as exc:
                        product_rejections.append(
                            {
                                "slot_ids": [profile.slot.slot_id for profile in group],
                                "reason": f"{type(exc).__name__}:{exc}",
                            }
                        )
            except (SyntaxError, TypeError, ValueError) as exc:
                slots = []
                affine_profiles = []
                groups = []
                group_scope_inventory = {
                    "scope_group_count_before_cap_by_logical_slot_count": {},
                    "scope_group_count_after_cap_by_logical_slot_count": {},
                    "scope_group_count_discarded_by_cap_by_logical_slot_count": {},
                    "general_group_count_before_cap_by_logical_slot_count": {},
                    "general_group_count_after_cap_by_logical_slot_count": {},
                    "general_group_count_discarded_by_cap_by_logical_slot_count": {},
                }
                product_profiles = []
                guard_rejections = []
                affine_rejections = []
                product_rejections = [{"slot_ids": [], "reason": f"{type(exc).__name__}:{exc}"}]
            group_counts = collections.Counter(len(group) for group in groups)
            profile_counts = collections.Counter(len(profile.slots) for profile in product_profiles)
            scope_group_counts = (
                collections.Counter(len(group) for group in groups if _group_matches_scope(group, group_scope))
                if scoped_mode
                else collections.Counter()
            )
            scope_profile_counts = (
                collections.Counter(
                    len(profile.slots)
                    for profile in product_profiles
                    if _slots_match_scope(profile.slots, group_scope)
                )
                if scoped_mode
                else collections.Counter()
            )
            decision.update(
                {
                    "group_scope_status": ("evaluated" if scoped_mode else "not_applicable"),
                    "slot_count": len(slots),
                    "affine_slot_count": len(affine_profiles),
                    "maximum_compatible_logical_slot_count": max(group_counts, default=0),
                    "maximum_product_profile_logical_slot_count": max(profile_counts, default=0),
                    "variable_group_count_by_logical_slot_count": {
                        str(key): value for key, value in sorted(group_counts.items())
                    },
                    "product_profile_count_by_logical_slot_count": {
                        str(key): value for key, value in sorted(profile_counts.items())
                    },
                    "scope_matching_group_count_by_logical_slot_count": {
                        str(key): value for key, value in sorted(scope_group_counts.items())
                    },
                    "scope_matching_product_profile_count_by_logical_slot_count": {
                        str(key): value for key, value in sorted(scope_profile_counts.items())
                    },
                    **group_scope_inventory,
                    "slot_rejections": guard_rejections + affine_rejections,
                    "group_rejections": product_rejections,
                }
            )
            counters["slots_found"] += len(slots)
            counters["affine_slots"] += len(affine_profiles)
            for size, count in group_counts.items():
                counters[f"structural_groups_k{size}"] += count
            for size, count in profile_counts.items():
                counters[f"product_profiles_k{size}"] += count

            candidates: list[VariableCandidate] = []
            candidate_rejections: collections.Counter[str] = collections.Counter()
            for profile in product_profiles:
                candidates.extend(
                    _solve_product_profile(
                        parent_code,
                        entry_point,
                        profile,
                        parent_uuid=parent_uuid,
                        variant=variant,
                        target_input_bytes=target,
                        parent_input_bytes=parent_analysis.input_bytes,
                        rejection_counts=candidate_rejections,
                    )
                )
            counters["static_solved_candidates"] += len(candidates)
            decision["static_solved_candidate_count"] = len(candidates)
            scope_candidate_count = (
                sum(_slots_match_scope(candidate.profile.slots, group_scope) for candidate in candidates)
                if scoped_mode
                else 0
            )
            decision["scope_matching_static_solved_candidate_count"] = scope_candidate_count
            (
                candidates,
                preferred_attempt_lane,
                attempt_plan_evidence,
            ) = _ordered_candidate_attempts(
                candidates,
                parent_uuid=parent_uuid,
                group_scope=group_scope,
            )
            decision["bounded_candidate_attempt_count"] = len(candidates)
            decision["preferred_attempt_lane"] = preferred_attempt_lane
            decision["attempt_plan_evidence"] = attempt_plan_evidence
            decision["scope_candidate_attempt_count"] = sum(
                scoped_mode and _slots_match_scope(candidate.profile.slots, group_scope) for candidate in candidates
            )
            decision["general_fallback_candidate_attempt_count"] = sum(
                scoped_mode and not _slots_match_scope(candidate.profile.slots, group_scope)
                for candidate in candidates
            )
            decision["candidate_rejection_counts"] = dict(sorted(candidate_rejections.items()))
            for reason, count in candidate_rejections.items():
                counters[f"candidate_rejection:{reason}"] += count
            if not candidates:
                decision["reason"] = "no_variable_multislot_product_solution"
                skip_reasons[str(decision["reason"])] += 1
                decisions.append(decision)
                continue

            parent_sections = _section_hashes(ast.parse(parent_code), entry_point)
            for attempt_index, candidate in enumerate(candidates):
                attempt = _candidate_manifest(parent_uuid, candidate)
                attempt_scope_match = _slots_match_scope(candidate.profile.slots, group_scope) if scoped_mode else None
                attempt["attempt_index"] = attempt_index
                attempt_lane = "scope" if attempt_scope_match else ("general" if scoped_mode else "generic")
                attempt["attempt_lane"] = attempt_lane
                try:
                    child_sections = _section_hashes(ast.parse(candidate.child_code), entry_point)
                    if parent_sections != child_sections:
                        raise ValueError("model_or_get_init_inputs_changed")
                    static = static_gate(parent_code, candidate.child_code, entry_point)
                    if int(static["input_bytes_after"]) != candidate.input_bytes_after:
                        raise ValueError("static_gate_storage_mismatch")
                    _validate_variant_storage(variant, candidate.input_bytes_after)
                    relative_error = _validate_target_proximity(candidate.input_bytes_after, target)
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
                    solver_evidence = _candidate_manifest(parent_uuid, candidate)
                    selected_scope_match = attempt_scope_match
                    used_group_scope_fallback = bool(
                        scoped_mode
                        and not selected_scope_match
                        and (preferred_attempt_lane == "scope" or scope_candidate_count == 0)
                    )
                    group_scope_fallback_reason = None
                    if used_group_scope_fallback:
                        if not scope_group_counts:
                            group_scope_fallback_reason = "no_compatible_scope_group"
                        elif not scope_profile_counts:
                            group_scope_fallback_reason = "no_exact_scope_product_profile"
                        elif scope_candidate_count == 0:
                            group_scope_fallback_reason = "no_scope_static_solution"
                        else:
                            group_scope_fallback_reason = "scope_candidate_attempts_exhausted"
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
                            "selected_logical_slot_count": len(candidate.profile.slots),
                            "selected_group_scope_match": selected_scope_match,
                            "selected_candidate_attempt_index": attempt_index,
                            "used_group_scope_fallback": used_group_scope_fallback,
                            "used_preferred_attempt_lane_fallback": (
                                preferred_attempt_lane not in {"generic", attempt_lane}
                            ),
                            "group_scope_fallback_reason": (group_scope_fallback_reason),
                            "preferred_attempt_lane_fallback_reason": (
                                None
                                if preferred_attempt_lane in {"generic", attempt_lane}
                                else (
                                    "scope_candidate_attempts_exhausted"
                                    if preferred_attempt_lane == "scope"
                                    else "general_candidate_attempts_exhausted"
                                )
                            ),
                            "used_smaller_compatible_group_fallback": (
                                len(candidate.profile.slots) < int(decision["maximum_compatible_logical_slot_count"])
                            ),
                            "used_smaller_product_profile_fallback": (
                                len(candidate.profile.slots)
                                < int(decision["maximum_product_profile_logical_slot_count"])
                            ),
                        }
                    )
                    review_records.append(
                        {
                            "parent_uuid": parent_uuid,
                            "child_uuid": child_uuid,
                            "variant": variant,
                            "logical_slot_count": len(candidate.profile.slots),
                            "input_bytes_before": parent_analysis.input_bytes,
                            "input_bytes_after": candidate.input_bytes_after,
                            "target_relative_error": relative_error,
                            "parent_code": parent_code,
                            "child_code": candidate.child_code,
                        }
                    )
                    counters[f"accepted_{variant}"] += 1
                    accepted_lane = (
                        "generic" if not scoped_mode else ("scope_match" if selected_scope_match else "general")
                    )
                    counters[f"accepted_{accepted_lane}"] += 1
                    counters[f"accepted_logical_slot_count_{len(candidate.profile.slots)}"] += 1
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
        if any(int(_nested(decision, "solver.changed_occurrences", 0)) < 2 for decision in accepted):
            raise ValueError("single_changed_occurrence_contract_failed")
        changed_occurrences = sum(int(_nested(decision, "solver.changed_occurrences", 0)) for decision in accepted)
        power_occurrences = sum(int(_nested(decision, "solver.power_of_two_occurrences", 0)) for decision in accepted)
        observed_power_fraction = power_occurrences / changed_occurrences if changed_occurrences else None
        static_power_range_passed = observed_power_fraction is not None and 0.30 <= observed_power_fraction <= 0.50

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
        temporary_paths.review.write_text(_review_markdown(review_records), encoding="utf-8")

        counters["parents_with_children"] = len(accepted)
        counters["children_written"] = len(children)
        counters["paired_rows_written"] = len(paired)
        target_map_sha256 = _logical_target_map_sha256(target_map_records)
        manifest = {
            "contract_version": solver_contract_version,
            "generator_version": generator_version,
            "solver_source_path": str(Path(__file__).resolve()),
            "solver_source_sha256": _sha256_file(Path(__file__)),
            "v3_helper_source_path": str((_REPO_ROOT / "tools/data/synthesize/solve_shape_coverage.py").resolve()),
            "v3_helper_source_sha256": _sha256_file(_REPO_ROOT / "tools/data/synthesize/solve_shape_coverage.py"),
            "dependency_source_contract": {
                "solve_multidim_shape_coverage_path": str(
                    (_REPO_ROOT / "tools/data/synthesize/solve_multidim_shape_coverage.py").resolve()
                ),
                "solve_multidim_shape_coverage_sha256": _sha256_file(
                    _REPO_ROOT / "tools/data/synthesize/solve_multidim_shape_coverage.py"
                ),
                "solve_shape_coverage_path": str(
                    (_REPO_ROOT / "tools/data/synthesize/solve_shape_coverage.py").resolve()
                ),
                "solve_shape_coverage_sha256": _sha256_file(
                    _REPO_ROOT / "tools/data/synthesize/solve_shape_coverage.py"
                ),
            },
            "method_boundary": (
                "one stable Medium/Large target per delta parent; variable two-to-five "
                "same-factory structural slots; exact constant-plus-product storage "
                "proof; static identity and parent/child FakeTensor gates"
            ),
            "source_edit_contract": (
                "replace two through five disjoint proven input-shape integer spans; "
                "module-scope linked names are allowed only when slot discovery proves "
                "all loads are direct input-shape dimensions; preserve Model and "
                "get_init_inputs exactly"
            ),
            "target_contract": {
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
                "targets_reused_from_prior_full_lane": True,
            },
            "group_contract": {
                "logical_slot_count_range": [MIN_GROUP_SLOTS, MAX_GROUP_SLOTS],
                "preference": (
                    "scope match first, then largest supported structural group"
                    if group_scope == NONLEADING_NO_EXPLICIT_BATCH_GROUP_SCOPE
                    else (
                        "preserve scope groups with at least three slots; prefer a "
                        "two-slot scope group only when general batch-like growth dominates"
                        if group_scope == BALANCED_NONLEADING_NO_EXPLICIT_BATCH_GROUP_SCOPE
                        else "largest supported structural group first"
                    )
                ),
                "factory_set": "same non-empty set for every slot",
                "per_factory": "every slot occurs once on a distinct axis",
                "patch_spans": "pairwise disjoint",
                "module_scope_linked_names": "allowed after shape-only load proof",
                "storage_model": "exact constant + coefficient * product(values)",
                "slot_values": "all strictly greater than parent values",
            },
            "scope_selection_contract": {
                "mode": group_scope,
                "scope_definition": (
                    "all occurrences of every selected slot have axis > 0 and no "
                    "selected linked-name slot is an explicit batch symbol"
                    if group_scope != GENERIC_GROUP_SCOPE
                    else None
                ),
                "balanced_two_slot_scope_preference": (
                    "prefer scope when the best general candidate has a leading-axis "
                    "or explicit-batch slot tied for maximum logical log growth"
                    if group_scope == BALANCED_NONLEADING_NO_EXPLICIT_BATCH_GROUP_SCOPE
                    else None
                ),
                "explicit_batch_symbols": sorted(EXPLICIT_BATCH_SYMBOLS),
                "group_retention": (
                    "scope groups precede a reserved general lane before the "
                    "per-cardinality cap; unused scope capacity is filled by general "
                    "groups"
                    if group_scope != GENERIC_GROUP_SCOPE
                    else "deterministic hash order before the per-cardinality cap"
                ),
                "maximum_groups_per_cardinality": MAX_GROUPS_PER_SIZE,
                "minimum_general_groups_per_cardinality_when_available": (
                    MIN_GENERAL_GROUPS_PER_SIZE if scoped_mode else None
                ),
                "maximum_child_attempts_per_parent": MAX_CHILD_FAKE_ATTEMPTS,
                "maximum_preferred_lane_attempts_when_both_lanes_exist": (
                    MAX_SCOPE_CHILD_FAKE_ATTEMPTS if scoped_mode else None
                ),
                "reserved_secondary_lane_attempts_when_both_lanes_exist": (
                    MAX_SCOPE_FALLBACK_CHILD_FAKE_ATTEMPTS if scoped_mode else None
                ),
                "one_child_per_parent": True,
                "attempt_plan_evidence_fields": [
                    "preference_reason",
                    "scope_candidate_count_before_bound",
                    "general_candidate_count_before_bound",
                    "maximum_scope_logical_slot_count",
                    "best_general_batch_like_growth_dominates",
                ],
                "fallback_reason_values": (
                    [
                        "no_compatible_scope_group",
                        "no_exact_scope_product_profile",
                        "no_scope_static_solution",
                        "scope_candidate_attempts_exhausted",
                    ]
                    if scoped_mode
                    else []
                ),
            },
            "distribution_contract": {
                "dimension_occurrence": "one changed direct input-factory axis",
                "single_changed_occurrence_child_maximum_fraction": 0.10,
                "constructed_single_changed_occurrence_child_fraction": 0.0,
                "power_of_two_occurrence_fraction_range": [0.30, 0.50],
                "power_of_two_soft_preference_probability": (
                    POWER_OF_TWO_PREFERENCE_NUMERATOR / POWER_OF_TWO_PREFERENCE_DENOMINATOR
                ),
                "preference_independence_key": "parent_uuid + logical_slot_id",
                "per_child_power_of_two_constraint": None,
                "runtime_subset_closed": False,
                "required_final_audit": "recompute after H20 filtering",
            },
            "distribution_observation": {
                "static_changed_occurrences": changed_occurrences,
                "static_power_of_two_occurrences": power_occurrences,
                "static_power_of_two_occurrence_fraction": observed_power_fraction,
                "static_power_of_two_range_passed": static_power_range_passed,
                "scope": "this unmerged solver invocation only",
            },
            "candidate_ranking_contract": [
                *(
                    ["contract-selected preferred lane before bounded secondary lane"]
                    if group_scope != GENERIC_GROUP_SCOPE
                    else []
                ),
                "larger_logical_slot_count",
                "smaller_target_error_0.1_percent_bucket",
                "fewer_occurrence_weighted_soft_preference_mismatches",
                "smaller_relative_growth_spread",
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
                "preferred_lane_attempt_limit_when_both_lanes_exist": (
                    MAX_SCOPE_CHILD_FAKE_ATTEMPTS if scoped_mode else None
                ),
                "secondary_lane_attempt_reservation_when_both_lanes_exist": (
                    MAX_SCOPE_FALLBACK_CHILD_FAKE_ATTEMPTS if scoped_mode else None
                ),
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
        help=f"prepared delta parent parquet (default: {DEFAULT_SELECTED})",
    )
    parser.add_argument(
        "--max-parents",
        type=int,
        help="process only the first N delta parents for a smoke run",
    )
    parser.add_argument(
        "--fake-gate-timeout-seconds",
        type=float,
        default=DEFAULT_FAKE_GATE_TIMEOUT_SECONDS,
    )
    parser.add_argument(
        "--group-scope",
        choices=GROUP_SCOPE_CHOICES,
        default=GENERIC_GROUP_SCOPE,
        help=(
            "candidate group preference; scoped modes retain non-leading, "
            "non-explicit-batch groups before truncation, and the balanced mode "
            "preserves multi-slot coverage unless batch-like growth dominates"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    manifest = solve_variable_shape_delta(
        args.selected,
        args.run_dir,
        max_parents=args.max_parents,
        fake_gate_timeout_seconds=args.fake_gate_timeout_seconds,
        group_scope=args.group_scope,
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
