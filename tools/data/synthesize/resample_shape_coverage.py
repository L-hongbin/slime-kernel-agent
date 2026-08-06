#!/usr/bin/env python3
"""Select a deterministic, coverage-balanced subset of runtime-safe shape children.

Each source row can expose several direct input tensors.  This tool assigns one
stable sampled tensor occurrence to the row, then balances rows over aggregate
input size, sampled-tensor size, rank, aspect ratio, coarse ordered geometry,
and the random/value family that the downstream solver would assign.  It never
rewrites a row.  Runtime-ineligible shape children fail closed before sampling.

The repository defaults intentionally name the two production shape runs.  A
normal invocation therefore needs only an output directory; the downstream
random/value lane can consume ``selected.parquet`` directly.
"""

from __future__ import annotations

import argparse
import ast
import collections
import csv
import dataclasses
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tools.data.synthesize.augment_prompt_tasks import (
    _factory_records,
    _module_constant_environment,
    _sha256_file,
    _top_level_function,
)
from tools.data.synthesize.random_method.solve_value_coverage import _analyze_parent, _git_blob_sha256, _git_commit

CONTRACT_VERSION = "shape_runtime_single_changed_tensor_coverage_resample_v3"
SELECTION_VERSION = "shape_cells_proportional_quota_with_secondary_marginals_v4"
DEFAULT_LIMIT = 5_000
MAX_MARGINAL_TOTAL_VARIATION = 0.05
MARGINAL_REPAIR_TARGET = 0.045
MIB = 1024**2

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SHAPE_RUNS = (
    _REPO_ROOT / "Data/prompt_tvm_v4/shape_solver_multidim_v4_byte_targets_v1/run.full53896",
    _REPO_ROOT / "Data/prompt_tvm_v4/shape_solver_variable_multislot_v5/run.recoverable12318.balanced_v7",
)

_SAMPLED_NUMEL_BOUNDS = (2**12, 2**16, 2**20, 2**24, 2**28, 2**32)
_SAMPLED_NUMEL_LABELS = (
    "lt_4k",
    "4k_to_64k",
    "64k_to_1m",
    "1m_to_16m",
    "16m_to_256m",
    "256m_to_4b",
    "ge_4b",
)
_EXPLICIT_BATCH_SYMBOLS = frozenset({"batch_size", "batchsize", "batch", "bs", "n_batch"})


@dataclasses.dataclass(frozen=True)
class Candidate:
    row: Mapping[str, Any] = dataclasses.field(compare=False, repr=False)
    lane_index: int
    lane_id: str
    source_artifact_path: str
    source_artifact_sha256: str
    source_row_index: int
    child_uuid: str
    canonical_parent_uuid: str
    reference_sha256: str
    source_family: str
    operator_bucket: str
    assigned_random_family: str
    variant: str
    input_scale: float
    scale_bucket: str
    logical_slots: int
    logical_slot_bucket: str
    axis_position_bucket: str
    explicit_batch_bucket: str
    changed_factory_indices: tuple[int, ...]
    direct_factory_count: int
    sampled_factory_index: int
    sampled_factory_name: str
    sampled_factory_dtype: str
    sampled_shape: tuple[int, ...]
    sampled_numel: int
    sampled_tensor_bytes: int
    aggregate_input_bytes: int
    aggregate_size_bucket: str
    sampled_size_bucket: str
    rank_bucket: str
    aspect_bucket: str
    geometry_signature: str
    selection_sha256: str

    @property
    def main_cell(self) -> tuple[str, str, str, str]:
        return (
            self.aggregate_size_bucket,
            self.sampled_size_bucket,
            self.rank_bucket,
            self.aspect_bucket,
        )


@dataclasses.dataclass(frozen=True)
class ChildShapeProfile:
    child_uuid: str
    variant: str
    input_bytes_before: int
    input_bytes_after: int
    input_scale: float
    logical_slots: int
    changed_factories: tuple[ChangedFactoryEvidence, ...]
    touches_leading: bool
    touches_nonleading: bool
    touches_explicit_batch: bool

    @property
    def changed_factory_indices(self) -> tuple[int, ...]:
        return tuple(item.factory_index for item in self.changed_factories)


@dataclasses.dataclass(frozen=True)
class ChangedFactoryEvidence:
    factory_index: int
    factory_name: str
    rank: int
    changed_axes: tuple[int, ...]


_SECONDARY_MARGINALS = (
    "lane_id",
    "variant",
    "source_family",
    "operator_bucket",
    "assigned_random_family",
    "scale_bucket",
    "logical_slot_bucket",
    "axis_position_bucket",
    "explicit_batch_bucket",
)

_SHAPE_MARGINALS = (
    "aggregate_size_bucket",
    "sampled_size_bucket",
    "rank_bucket",
    "aspect_bucket",
)


def _nested(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _stable_sha256(*parts: object) -> str:
    payload = ":".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sampled_size_bucket(numel: int) -> str:
    for bound, label in zip(_SAMPLED_NUMEL_BOUNDS, _SAMPLED_NUMEL_LABELS, strict=False):
        if numel < bound:
            return label
    return _SAMPLED_NUMEL_LABELS[-1]


def _aggregate_size_bucket(input_bytes: int) -> str:
    if input_bytes < 64 * MIB:
        return "below_medium"
    if input_bytes <= 256 * MIB:
        return "medium_64m_to_256m"
    if input_bytes <= 4 * 1024**3:
        return "large_256m_to_4g"
    return "above_4g"


def _rank_bucket(rank: int) -> str:
    return f"rank_{rank}" if rank <= 5 else "rank_6_plus"


def _scale_bucket(scale: float) -> str:
    if not math.isfinite(scale) or scale < 2:
        raise ValueError(f"shape input scale must be finite and at least 2x, found {scale!r}")
    if scale <= 4:
        return "scale_2_to_4"
    if scale <= 32:
        return "scale_4_to_32"
    if scale <= 256:
        return "scale_32_to_256"
    if scale <= 4_096:
        return "scale_256_to_4096"
    return "scale_above_4096"


def _parse_tsv_bool(value: str, *, field: str, child_uuid: str) -> bool:
    if value == "True":
        return True
    if value in {"", "False"}:
        return False
    raise ValueError(f"invalid {field} for {child_uuid}: {value!r}")


def _aspect_bucket(shape: Sequence[int]) -> str:
    """Bucket the largest/second-largest ratio used by the shape contract."""

    if len(shape) == 1:
        return "rank1"
    largest, second_largest = sorted(shape, reverse=True)[:2]
    if largest == second_largest:
        return "equal"
    ratio = largest / second_largest
    for upper, label in (
        (2, "gt1_to_2"),
        (8, "gt2_to_8"),
        (32, "gt8_to_32"),
        (128, "gt32_to_128"),
        (1_000, "gt128_to_1000"),
    ):
        if ratio <= upper:
            return label
    return "gt1000"


def _geometry_signature(shape: Sequence[int]) -> str:
    """Coarsen ordered axis proportions without collapsing axis order."""

    rank = len(shape)
    mean_log2 = sum(math.log2(value) for value in shape) / rank
    offsets = tuple(max(-12, min(12, round(math.log2(value) - mean_log2))) for value in shape)
    return f"r{rank}:" + ",".join(str(value) for value in offsets)


def _load_child_profiles(path: Path) -> dict[str, ChildShapeProfile]:
    """Load independently reconstructed shape changes emitted by the analyzer."""

    if not path.is_file():
        raise FileNotFoundError(f"shape analysis is missing changed-slot evidence: {path}")
    grouped: dict[str, list[dict[str, str]]] = collections.defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {
            "child_uuid",
            "variant",
            "slot_index",
            "factory_index",
            "factory_name",
            "rank",
            "axis",
            "leading",
            "input_bytes_before",
            "input_bytes_after",
            "input_scale",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = sorted(required - set(reader.fieldnames or ()))
            raise ValueError(f"changed-slot TSV lacks required columns {missing}: {path}")
        for row in reader:
            child_uuid = row.get("child_uuid", "")
            if not child_uuid:
                raise ValueError(f"changed-slot TSV has an empty child UUID: {path}")
            grouped[child_uuid].append(row)

    profiles: dict[str, ChildShapeProfile] = {}
    for child_uuid, rows in grouped.items():
        variants = {row["variant"] for row in rows}
        before_values = {int(row["input_bytes_before"]) for row in rows}
        after_values = {int(row["input_bytes_after"]) for row in rows}
        scales = {float(row["input_scale"]) for row in rows}
        if len(variants) != 1 or len(before_values) != 1 or len(after_values) != 1 or len(scales) != 1:
            raise ValueError(f"inconsistent changed-slot child metadata: {child_uuid}")
        slot_indices = {int(row["slot_index"]) for row in rows}
        factory_indices = tuple(sorted({int(row["factory_index"]) for row in rows}))
        if not slot_indices or not factory_indices:
            raise ValueError(f"changed-slot child has no logical slot or factory: {child_uuid}")
        changed_factories: list[ChangedFactoryEvidence] = []
        for factory_index in factory_indices:
            factory_rows = [row for row in rows if int(row["factory_index"]) == factory_index]
            names = {row["factory_name"] for row in factory_rows}
            ranks = {int(row["rank"]) for row in factory_rows}
            axes = tuple(sorted({int(row["axis"]) for row in factory_rows}))
            if len(names) != 1 or len(ranks) != 1 or not axes:
                raise ValueError(f"inconsistent changed-factory evidence: {child_uuid}:{factory_index}")
            rank = next(iter(ranks))
            if rank <= 0 or any(axis < 0 or axis >= rank for axis in axes):
                raise ValueError(
                    f"changed-factory axis/rank evidence is invalid: " f"{child_uuid}:{factory_index}:{rank}:{axes}"
                )
            changed_factories.append(
                ChangedFactoryEvidence(
                    factory_index=factory_index,
                    factory_name=next(iter(names)),
                    rank=rank,
                    changed_axes=axes,
                )
            )
        leading_values = [_parse_tsv_bool(row["leading"], field="leading", child_uuid=child_uuid) for row in rows]
        explicit_batch_values = [
            (
                _parse_tsv_bool(
                    row.get("explicit_batch_symbol", ""),
                    field="explicit_batch_symbol",
                    child_uuid=child_uuid,
                )
                or row.get("symbol_name", "").strip().lower() in _EXPLICIT_BATCH_SYMBOLS
            )
            for row in rows
        ]
        profiles[child_uuid] = ChildShapeProfile(
            child_uuid=child_uuid,
            variant=next(iter(variants)),
            input_bytes_before=next(iter(before_values)),
            input_bytes_after=next(iter(after_values)),
            input_scale=next(iter(scales)),
            logical_slots=len(slot_indices),
            changed_factories=tuple(changed_factories),
            touches_leading=any(leading_values),
            touches_nonleading=not all(leading_values),
            touches_explicit_batch=any(explicit_batch_values),
        )
    return profiles


def _load_runtime_eligible(run_dir: Path) -> tuple[set[str], Mapping[str, Any], Path]:
    summary_path = run_dir / "analysis/summary.json"
    children_path = run_dir / "static/children.parquet"
    if not summary_path.is_file() or not children_path.is_file():
        raise FileNotFoundError(f"shape run is missing summary or children: {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    eligible = _nested(summary, "runtime.eligible")
    if not isinstance(eligible, Mapping) or eligible.get("available") is not True:
        raise ValueError(f"shape run has no runtime-eligible partition: {run_dir}")
    if eligible.get("coverage_complete") is not True:
        raise ValueError(f"shape runtime coverage is incomplete: {run_dir}")
    raw_uuids = eligible.get("eligible_child_uuids")
    if not isinstance(raw_uuids, list) or any(not isinstance(value, str) for value in raw_uuids):
        raise ValueError(f"shape runtime eligible UUIDs are invalid: {run_dir}")
    eligible_uuids = set(raw_uuids)
    if len(eligible_uuids) != len(raw_uuids):
        raise ValueError(f"shape runtime eligible UUIDs contain duplicates: {run_dir}")
    if eligible.get("eligible_children") != len(eligible_uuids):
        raise ValueError(f"shape runtime eligible count disagrees with UUIDs: {run_dir}")
    expected_children_sha = _nested(summary, "artifact_sha256.children")
    actual_children_sha = _sha256_file(children_path)
    if expected_children_sha != actual_children_sha:
        raise ValueError(
            f"shape children SHA mismatch for {run_dir}: expected {expected_children_sha}, "
            f"found {actual_children_sha}"
        )
    return eligible_uuids, summary, children_path


def _candidate_from_row(
    row: Mapping[str, Any],
    *,
    profile: ChildShapeProfile,
    lane_index: int,
    lane_id: str,
    source_artifact_path: Path,
    source_artifact_sha256: str,
    source_row_index: int,
) -> tuple[Candidate | None, str]:
    eligible, reason = _analyze_parent(row, source_row_index)
    if eligible is None:
        return None, reason
    code = _nested(row, "reward_model.ground_truth")
    child_uuid = _nested(row, "extra_info.uuid")
    canonical_parent_uuid = _nested(row, "extra_info.v4.parent_uuid")
    if not isinstance(code, str) or not isinstance(child_uuid, str):
        return None, "invalid_shape_child_identity"
    if not isinstance(canonical_parent_uuid, str) or not canonical_parent_uuid:
        return None, "missing_canonical_parent_uuid"
    if profile.child_uuid != child_uuid:
        return None, "changed_slot_profile_uuid_mismatch"
    if profile.input_bytes_after != eligible.input_bytes:
        return None, "changed_slot_profile_input_bytes_mismatch"
    try:
        tree = ast.parse(code)
        get_inputs = _top_level_function(tree, "get_inputs")
        records, _ = _factory_records(tree, get_inputs, _module_constant_environment(tree))
    except (SyntaxError, ValueError) as exc:
        return None, f"shape_extraction_failed:{type(exc).__name__}:{exc}"
    if not records:
        return None, "no_direct_input_factory"
    records = sorted(records, key=lambda record: (record.line_number, record.column_offset))
    locations = [(record.line_number, record.column_offset) for record in records]
    if len(locations) != len(set(locations)):
        return None, "duplicate_direct_factory_source_location"
    for evidence in profile.changed_factories:
        if evidence.factory_index < 0 or evidence.factory_index >= len(records):
            return None, "changed_factory_index_out_of_range"
        record = records[evidence.factory_index]
        if record.name != evidence.factory_name:
            return None, "changed_factory_name_mismatch"
        if len(record.shape) != evidence.rank:
            return None, "changed_factory_rank_mismatch"
        if any(axis < 0 or axis >= len(record.shape) for axis in evidence.changed_axes):
            return None, "changed_factory_axis_out_of_range"
    sample_sha256 = _stable_sha256(
        CONTRACT_VERSION,
        source_artifact_sha256,
        child_uuid,
        eligible.parent_reference_sha256,
    )
    sampled_factory_index = profile.changed_factory_indices[
        int(sample_sha256[:16], 16) % len(profile.changed_factory_indices)
    ]
    sampled = records[sampled_factory_index]
    sampled_numel = math.prod(sampled.shape)
    selection_sha256 = _stable_sha256(
        SELECTION_VERSION,
        source_artifact_sha256,
        child_uuid,
        eligible.parent_reference_sha256,
        sampled_factory_index,
        sampled.shape,
    )
    return (
        Candidate(
            row=row,
            lane_index=lane_index,
            lane_id=lane_id,
            source_artifact_path=str(source_artifact_path.resolve()),
            source_artifact_sha256=source_artifact_sha256,
            source_row_index=source_row_index,
            child_uuid=child_uuid,
            canonical_parent_uuid=canonical_parent_uuid,
            reference_sha256=eligible.parent_reference_sha256,
            source_family=eligible.source_family,
            operator_bucket=eligible.operator_bucket,
            assigned_random_family=eligible.assigned_family,
            variant=profile.variant,
            input_scale=profile.input_scale,
            scale_bucket=_scale_bucket(profile.input_scale),
            logical_slots=profile.logical_slots,
            logical_slot_bucket=f"slots_{profile.logical_slots}",
            axis_position_bucket=(
                "mixed_leading_nonleading"
                if profile.touches_leading and profile.touches_nonleading
                else "touches_leading" if profile.touches_leading else "nonleading_only"
            ),
            explicit_batch_bucket=(
                "touches_explicit_batch" if profile.touches_explicit_batch else "no_explicit_batch"
            ),
            changed_factory_indices=profile.changed_factory_indices,
            direct_factory_count=len(records),
            sampled_factory_index=sampled_factory_index,
            sampled_factory_name=sampled.name,
            sampled_factory_dtype=sampled.dtype_name,
            sampled_shape=sampled.shape,
            sampled_numel=sampled_numel,
            sampled_tensor_bytes=sampled.bytes,
            aggregate_input_bytes=eligible.input_bytes,
            aggregate_size_bucket=_aggregate_size_bucket(eligible.input_bytes),
            sampled_size_bucket=_sampled_size_bucket(sampled_numel),
            rank_bucket=_rank_bucket(len(sampled.shape)),
            aspect_bucket=_aspect_bucket(sampled.shape),
            geometry_signature=_geometry_signature(sampled.shape),
            selection_sha256=selection_sha256,
        ),
        "eligible",
    )


def _target_quotas(groups: Mapping[Any, Sequence[Any]], limit: int) -> dict[Any, int]:
    """Allocate exact, capacity-bounded quotas with one row per occupied cell.

    The divisor score is proportional to population size.  The previous square-
    root score deliberately flattened the population and substantially
    overrepresented rare ranks, sources, and geometries.
    """

    if limit < len(groups):
        raise ValueError(f"limit {limit} cannot cover {len(groups)} occupied joint cells")
    quotas = {key: 1 for key in groups}
    for _ in range(limit - len(groups)):
        active = [key for key, values in groups.items() if quotas[key] < len(values)]
        if not active:
            raise ValueError("coverage quota allocator exhausted candidate capacity")
        key = max(
            active,
            key=lambda item: (
                len(groups[item]) / (quotas[item] + 1),
                _stable_sha256("quota", item),
            ),
        )
        quotas[key] += 1
    return quotas


def _select(candidates: Sequence[Candidate], limit: int) -> list[Candidate]:
    if not 1 <= limit <= len(candidates):
        raise ValueError(f"limit must be in [1, {len(candidates)}], found {limit}")
    groups: dict[tuple[str, ...], list[Candidate]] = collections.defaultdict(list)
    for candidate in candidates:
        groups[candidate.main_cell].append(candidate)
    cell_targets = _target_quotas(groups, limit)
    marginal_targets: dict[str, dict[str, int]] = {}
    for attribute in _SECONDARY_MARGINALS:
        marginal_groups: dict[str, list[Candidate]] = collections.defaultdict(list)
        for candidate in candidates:
            marginal_groups[str(getattr(candidate, attribute))].append(candidate)
        marginal_targets[attribute] = _target_quotas(marginal_groups, limit)

    chosen: dict[str, Candidate] = {}
    selected_by_cell: collections.Counter[tuple[str, ...]] = collections.Counter()
    selected_marginals: dict[str, collections.Counter[str]] = {
        attribute: collections.Counter() for attribute in _SECONDARY_MARGINALS
    }
    selected_geometries: collections.Counter[str] = collections.Counter()
    selected_exact_shapes: collections.Counter[tuple[int, ...]] = collections.Counter()

    def add(candidate: Candidate) -> None:
        if candidate.child_uuid in chosen or len(chosen) >= limit:
            raise RuntimeError(f"invalid duplicate or over-limit selection: {candidate.child_uuid}")
        chosen[candidate.child_uuid] = candidate
        selected_by_cell[candidate.main_cell] += 1
        for attribute in _SECONDARY_MARGINALS:
            selected_marginals[attribute][str(getattr(candidate, attribute))] += 1
        selected_geometries[candidate.geometry_signature] += 1
        selected_exact_shapes[candidate.sampled_shape] += 1

    while len(chosen) < limit:
        cells_with_deficit = [key for key in groups if selected_by_cell[key] < cell_targets[key]]
        if not cells_with_deficit:
            raise RuntimeError("shape-cell quotas ended before the requested limit")
        key = max(
            cells_with_deficit,
            key=lambda item: (
                (cell_targets[item] - selected_by_cell[item]) / cell_targets[item],
                cell_targets[item] - selected_by_cell[item],
                _stable_sha256("cell_fill", item),
            ),
        )
        options = [candidate for candidate in groups[key] if candidate.child_uuid not in chosen]
        if not options:
            raise RuntimeError(f"shape-cell capacity exhausted before quota: {key}")

        def candidate_score(candidate: Candidate) -> tuple[int, float, int, int, str]:
            uncovered = 0
            marginal_gain = 0.0
            for attribute in _SECONDARY_MARGINALS:
                value = str(getattr(candidate, attribute))
                count = selected_marginals[attribute][value]
                target = marginal_targets[attribute][value]
                uncovered += int(count == 0)
                marginal_gain += (abs(count - target) - abs(count + 1 - target)) / max(target, 1)
            return (
                uncovered,
                marginal_gain,
                int(selected_geometries[candidate.geometry_signature] == 0),
                int(selected_exact_shapes[candidate.sampled_shape] == 0),
                candidate.selection_sha256,
            )

        add(max(options, key=candidate_score))

    # Main-cell quotas exactly preserve the four primary shape marginals, but
    # correlated secondary attributes can still drift after the greedy fill.
    # Repair that drift with deterministic swaps inside the same main cell.
    # A same-cell swap cannot change aggregate/sample size, rank, aspect, or
    # occupied-cell coverage.
    population_counts = {
        attribute: collections.Counter(str(getattr(candidate, attribute)) for candidate in candidates)
        for attribute in _SECONDARY_MARGINALS
    }
    expected_counts = {
        attribute: {value: count * limit / len(candidates) for value, count in population_counts[attribute].items()}
        for attribute in _SECONDARY_MARGINALS
    }

    def marginal_tvds() -> dict[str, float]:
        return {
            attribute: 0.5
            * sum(
                abs(selected_marginals[attribute][value] - expected) / limit
                for value, expected in expected_counts[attribute].items()
            )
            for attribute in _SECONDARY_MARGINALS
        }

    def objective(tvds: Mapping[str, float]) -> tuple[int, float, float]:
        return (
            sum(value > MARGINAL_REPAIR_TARGET for value in tvds.values()),
            max(tvds.values()),
            sum(tvds.values()),
        )

    def signature(candidate: Candidate) -> tuple[str, ...]:
        return tuple(str(getattr(candidate, attribute)) for attribute in _SECONDARY_MARGINALS)

    def distinct_delta(counter: Mapping[Any, int], old: Any, new: Any) -> int:
        if old == new:
            return 0
        return int(counter.get(new, 0) == 0) - int(counter.get(old, 0) == 1)

    for _ in range(limit):
        current_tvds = marginal_tvds()
        current_objective = objective(current_tvds)
        if current_objective[0] == 0:
            break
        target_attribute = max(
            _SECONDARY_MARGINALS,
            key=lambda attribute: (current_tvds[attribute], attribute),
        )
        target_index = _SECONDARY_MARGINALS.index(target_attribute)
        over_values = {
            value
            for value, expected in expected_counts[target_attribute].items()
            if selected_marginals[target_attribute][value] > expected
        }
        under_values = {
            value
            for value, expected in expected_counts[target_attribute].items()
            if selected_marginals[target_attribute][value] < expected
        }

        selected_signatures: dict[tuple[str, ...], dict[tuple[str, ...], list[Candidate]]] = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        remaining_signatures: dict[tuple[str, ...], dict[tuple[str, ...], list[Candidate]]] = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )
        for candidate in candidates:
            destination = selected_signatures if candidate.child_uuid in chosen else remaining_signatures
            destination[candidate.main_cell][signature(candidate)].append(candidate)

        best: (
            tuple[
                tuple[tuple[int, float, float], int, int, str, str],
                Candidate,
                Candidate,
            ]
            | None
        ) = None
        for cell in sorted(selected_signatures):
            if cell not in remaining_signatures:
                continue
            for old_signature, old_candidates in selected_signatures[cell].items():
                if old_signature[target_index] not in over_values:
                    continue
                for new_signature, new_candidates in remaining_signatures[cell].items():
                    if new_signature[target_index] not in under_values:
                        continue
                    if any(
                        old_value != new_value and selected_marginals[attribute][old_value] <= 1
                        for attribute, old_value, new_value in zip(
                            _SECONDARY_MARGINALS,
                            old_signature,
                            new_signature,
                            strict=True,
                        )
                    ):
                        continue
                    swapped_tvds = dict(current_tvds)
                    for attribute, old_value, new_value in zip(
                        _SECONDARY_MARGINALS,
                        old_signature,
                        new_signature,
                        strict=True,
                    ):
                        if old_value == new_value:
                            continue
                        counts = selected_marginals[attribute]
                        expected = expected_counts[attribute]
                        absolute_error = 2 * limit * current_tvds[attribute]
                        absolute_error -= abs(counts[old_value] - expected[old_value])
                        absolute_error -= abs(counts[new_value] - expected[new_value])
                        absolute_error += abs(counts[old_value] - 1 - expected[old_value])
                        absolute_error += abs(counts[new_value] + 1 - expected[new_value])
                        swapped_tvds[attribute] = absolute_error / (2 * limit)
                    swapped_objective = objective(swapped_tvds)
                    if swapped_objective >= current_objective:
                        continue
                    old_candidate = max(
                        old_candidates,
                        key=lambda candidate: (
                            selected_geometries[candidate.geometry_signature] > 1,
                            selected_exact_shapes[candidate.sampled_shape] > 1,
                            candidate.selection_sha256,
                        ),
                    )
                    new_candidate = max(
                        new_candidates,
                        key=lambda candidate: (
                            selected_geometries[candidate.geometry_signature] == 0,
                            selected_exact_shapes[candidate.sampled_shape] == 0,
                            candidate.selection_sha256,
                        ),
                    )
                    geometry_delta = distinct_delta(
                        selected_geometries,
                        old_candidate.geometry_signature,
                        new_candidate.geometry_signature,
                    )
                    exact_shape_delta = distinct_delta(
                        selected_exact_shapes,
                        old_candidate.sampled_shape,
                        new_candidate.sampled_shape,
                    )
                    key = (
                        swapped_objective,
                        -geometry_delta,
                        -exact_shape_delta,
                        old_candidate.selection_sha256,
                        new_candidate.selection_sha256,
                    )
                    if best is None or key < best[0]:
                        best = (key, old_candidate, new_candidate)
        if best is None:
            raise RuntimeError(
                "same-cell secondary marginal repair has no improving swap: " f"{dict(sorted(current_tvds.items()))}"
            )
        _, old_candidate, new_candidate = best
        del chosen[old_candidate.child_uuid]
        chosen[new_candidate.child_uuid] = new_candidate
        for attribute in _SECONDARY_MARGINALS:
            selected_marginals[attribute][str(getattr(old_candidate, attribute))] -= 1
            selected_marginals[attribute][str(getattr(new_candidate, attribute))] += 1
        selected_geometries[old_candidate.geometry_signature] -= 1
        selected_geometries[new_candidate.geometry_signature] += 1
        selected_exact_shapes[old_candidate.sampled_shape] -= 1
        selected_exact_shapes[new_candidate.sampled_shape] += 1
    else:
        raise RuntimeError("same-cell secondary marginal repair exceeded its deterministic bound")

    return sorted(chosen.values(), key=lambda item: (item.lane_index, item.source_row_index))


def _counter(items: Sequence[Candidate], key: str) -> dict[str, int]:
    counts = collections.Counter(str(getattr(item, key)) for item in items)
    return dict(sorted(counts.items()))


def _joint_counter(items: Sequence[Candidate]) -> dict[str, int]:
    counts = collections.Counter("|".join(item.main_cell) for item in items)
    return dict(sorted(counts.items()))


def _quantiles(values: Sequence[int | float]) -> dict[str, int | float]:
    ordered = sorted(values)
    result: dict[str, int] = {}
    for label, fraction in (("min", 0), ("p50", 0.5), ("p90", 0.9), ("p99", 0.99), ("max", 1)):
        index = round((len(ordered) - 1) * fraction)
        result[label] = ordered[index]
    return result


def _marginal_audit(
    population: Sequence[Candidate], selected: Sequence[Candidate], attributes: Sequence[str]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for attribute in attributes:
        population_counts = collections.Counter(str(getattr(item, attribute)) for item in population)
        selected_counts = collections.Counter(str(getattr(item, attribute)) for item in selected)
        population_groups: dict[str, list[Candidate]] = collections.defaultdict(list)
        for item in population:
            population_groups[str(getattr(item, attribute))].append(item)
        target_counts = _target_quotas(population_groups, len(selected))
        values = sorted(population_counts)
        total_variation = 0.5 * sum(
            abs(selected_counts[value] / len(selected) - population_counts[value] / len(population))
            for value in values
        )
        max_fraction_delta = max(
            abs(selected_counts[value] / len(selected) - population_counts[value] / len(population))
            for value in values
        )
        result[attribute] = {
            "population": dict(sorted(population_counts.items())),
            "proportional_target": dict(sorted(target_counts.items())),
            "selected": dict(sorted(selected_counts.items())),
            "all_values_covered": all(selected_counts[value] > 0 for value in values),
            "total_variation": total_variation,
            "max_fraction_delta": max_fraction_delta,
            "max_target_count_delta": max(abs(selected_counts[value] - target_counts[value]) for value in values),
            "passed": (
                all(selected_counts[value] > 0 for value in values) and total_variation <= MAX_MARGINAL_TOTAL_VARIATION
            ),
        }
    return result


def _manifest_row(candidate: Candidate, selected_index: int) -> dict[str, Any]:
    return {
        "contract_version": CONTRACT_VERSION,
        "selection_version": SELECTION_VERSION,
        "selected_index": selected_index,
        "shape_lane_id": candidate.lane_id,
        "source_artifact_path": candidate.source_artifact_path,
        "source_artifact_sha256": candidate.source_artifact_sha256,
        "source_row_index": candidate.source_row_index,
        "shape_child_uuid": candidate.child_uuid,
        "canonical_parent_uuid": candidate.canonical_parent_uuid,
        "reference_sha256": candidate.reference_sha256,
        "source_family": candidate.source_family,
        "operator_bucket": candidate.operator_bucket,
        "assigned_random_family": candidate.assigned_random_family,
        "variant": candidate.variant,
        "input_scale": candidate.input_scale,
        "scale_bucket": candidate.scale_bucket,
        "logical_slots": candidate.logical_slots,
        "logical_slot_bucket": candidate.logical_slot_bucket,
        "axis_position_bucket": candidate.axis_position_bucket,
        "explicit_batch_bucket": candidate.explicit_batch_bucket,
        "changed_factory_indices": list(candidate.changed_factory_indices),
        "direct_factory_count": candidate.direct_factory_count,
        "sampled_factory_index": candidate.sampled_factory_index,
        "sampled_factory_name": candidate.sampled_factory_name,
        "sampled_factory_dtype": candidate.sampled_factory_dtype,
        "sampled_shape": list(candidate.sampled_shape),
        "sampled_numel": candidate.sampled_numel,
        "sampled_tensor_bytes": candidate.sampled_tensor_bytes,
        "aggregate_input_bytes": candidate.aggregate_input_bytes,
        "aggregate_size_bucket": candidate.aggregate_size_bucket,
        "sampled_size_bucket": candidate.sampled_size_bucket,
        "rank_bucket": candidate.rank_bucket,
        "aspect_bucket": candidate.aspect_bucket,
        "geometry_signature": candidate.geometry_signature,
        "joint_coverage_cell": "|".join(candidate.main_cell),
        "selection_sha256": candidate.selection_sha256,
        "training_approved": False,
    }


def _review_markdown(selected: Sequence[Candidate]) -> str:
    lines = [
        "# Single-shape coverage resampling review",
        "",
        "每行只从 shape solver 实际改动过的 direct-input factory 中稳定抽取一个 tensor shape。"
        "下面按 coarse shape coverage cell 给出可人工复核的真实样本。",
        "",
    ]
    seen: set[tuple[str, ...]] = set()
    for candidate in selected:
        if candidate.main_cell in seen:
            continue
        seen.add(candidate.main_cell)
        lines.extend(
            [
                f"## `{candidate.child_uuid}`",
                "",
                f"- lane: `{candidate.lane_id}`; canonical parent: `{candidate.canonical_parent_uuid}`",
                f"- sampled factory: `{candidate.sampled_factory_index}/{candidate.direct_factory_count}` "
                f"`{candidate.sampled_factory_name}` `{candidate.sampled_factory_dtype}`",
                f"- sampled shape: `{list(candidate.sampled_shape)}`; numel: `{candidate.sampled_numel}`",
                f"- variant/scale/slots: `{candidate.variant}` / `{candidate.input_scale:.6g}x` / "
                f"`{candidate.logical_slots}`; axis: `{candidate.axis_position_bucket}`; "
                f"batch: `{candidate.explicit_batch_bucket}`",
                f"- cell: `{'|'.join(candidate.main_cell)}`; geometry: `{candidate.geometry_signature}`",
                "",
            ]
        )
        if len(seen) >= 48:
            break
    return "\n".join(lines).rstrip() + "\n"


def _atomic_text(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def build_resample(output_dir: Path, *, shape_runs: Sequence[Path], limit: int, overwrite: bool) -> dict[str, Any]:
    output_paths = {
        "selected": output_dir / "selected.parquet",
        "manifest": output_dir / "manifest.jsonl",
        "eligibility": output_dir / "eligibility.jsonl",
        "summary": output_dir / "summary.json",
        "review": output_dir / "review_samples.md",
    }
    existing = [path for path in output_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"refusing existing resample artifacts: {existing[:3]}")
    resampler_path = Path(__file__).resolve()
    augment_path = _REPO_ROOT / "tools/data/synthesize/augment_prompt_tasks.py"
    random_solver_path = _REPO_ROOT / "tools/data/synthesize/random_method/solve_value_coverage.py"
    git_commit = _git_commit()
    source_sha256 = {
        "resampler_source_sha256": _sha256_file(resampler_path),
        "augment_prompt_tasks.py": _sha256_file(augment_path),
        "solve_value_coverage.py": _sha256_file(random_solver_path),
    }
    for label, path in (
        ("resampler_source_sha256", resampler_path),
        ("augment_prompt_tasks.py", augment_path),
        ("solve_value_coverage.py", random_solver_path),
    ):
        if _git_blob_sha256(git_commit, path) != source_sha256[label]:
            raise RuntimeError(f"resample source differs from Git commit {git_commit}: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[Candidate] = []
    eligibility_rows: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    output_schema: pa.Schema | None = None
    seen_child_uuids: set[str] = set()
    seen_parent_uuids: set[str] = set()
    for lane_index, run_dir in enumerate(shape_runs):
        eligible_uuids, summary, children_path = _load_runtime_eligible(run_dir)
        children_sha256 = _sha256_file(children_path)
        lane_id = f"shape_lane_{lane_index}_{_stable_sha256(run_dir.resolve(), children_sha256)[:12]}"
        summary_path = run_dir / "analysis/summary.json"
        changed_slots_path = run_dir / "analysis/changed_slots.tsv"
        child_profiles = _load_child_profiles(changed_slots_path)
        parquet = pq.ParquetFile(children_path)
        if output_schema is None:
            output_schema = parquet.schema_arrow
        elif parquet.schema_arrow != output_schema:
            raise ValueError(f"shape child schema differs across lanes: {run_dir}")
        found_runtime_eligible: set[str] = set()
        lane_candidate_count = 0
        row_index = 0
        for batch in parquet.iter_batches(batch_size=256, use_threads=False):
            for row in batch.to_pylist():
                child_uuid = _nested(row, "extra_info.uuid")
                if child_uuid not in eligible_uuids:
                    row_index += 1
                    continue
                found_runtime_eligible.add(child_uuid)
                profile = child_profiles.get(str(child_uuid))
                if profile is None:
                    raise ValueError(f"runtime-eligible child lacks changed-slot evidence: {child_uuid}")
                candidate, reason = _candidate_from_row(
                    row,
                    profile=profile,
                    lane_index=lane_index,
                    lane_id=lane_id,
                    source_artifact_path=children_path,
                    source_artifact_sha256=children_sha256,
                    source_row_index=row_index,
                )
                eligibility_rows.append(
                    {
                        "shape_lane_id": lane_id,
                        "source_row_index": row_index,
                        "shape_child_uuid": child_uuid,
                        "eligibility_universe": "shape_runtime_eligible_children",
                        "random_static_eligible": candidate is not None,
                        "reason": reason,
                    }
                )
                if candidate is not None:
                    if candidate.child_uuid in seen_child_uuids:
                        raise ValueError(f"duplicate shape child UUID across lanes: {candidate.child_uuid}")
                    if candidate.canonical_parent_uuid in seen_parent_uuids:
                        raise ValueError(
                            "multiple runtime shape children share canonical parent: "
                            f"{candidate.canonical_parent_uuid}"
                        )
                    seen_child_uuids.add(candidate.child_uuid)
                    seen_parent_uuids.add(candidate.canonical_parent_uuid)
                    candidates.append(candidate)
                    lane_candidate_count += 1
                row_index += 1
        if found_runtime_eligible != eligible_uuids:
            missing = sorted(eligible_uuids - found_runtime_eligible)
            raise ValueError(f"runtime eligible UUIDs absent from children for {run_dir}: {missing[:3]}")
        source_records.append(
            {
                "lane_id": lane_id,
                "run_dir": str(run_dir.resolve()),
                "run_summary_path": str(summary_path.resolve()),
                "run_summary_sha256": _sha256_file(summary_path),
                "changed_slots_path": str(changed_slots_path.resolve()),
                "changed_slots_sha256": _sha256_file(changed_slots_path),
                "children_path": str(children_path.resolve()),
                "children_sha256": children_sha256,
                "runtime_eligible_children": len(eligible_uuids),
                "random_static_eligible_children": lane_candidate_count,
                "shape_contract": summary.get("schema_version"),
            }
        )
    if output_schema is None:
        raise ValueError("no shape run was provided")
    if limit > len(candidates):
        raise ValueError(f"requested {limit} rows but only {len(candidates)} are random-static-eligible")

    selected = _select(candidates, limit)
    selected_rows = [dict(item.row) for item in selected]
    manifest_rows = [_manifest_row(item, index) for index, item in enumerate(selected)]
    selected_joint = set(item.main_cell for item in selected)
    population_joint = set(item.main_cell for item in candidates)
    if selected_joint != population_joint:
        raise RuntimeError("selected rows do not cover every occupied joint cell")
    for attribute in _SHAPE_MARGINALS:
        if {getattr(item, attribute) for item in selected} != {getattr(item, attribute) for item in candidates}:
            raise RuntimeError(f"selected rows do not cover the {attribute} marginal")
    marginal_audit = _marginal_audit(candidates, selected, _SHAPE_MARGINALS + _SECONDARY_MARGINALS)
    failed_marginals = [attribute for attribute, record in marginal_audit.items() if not record["passed"]]
    if failed_marginals:
        details = {attribute: marginal_audit[attribute]["total_variation"] for attribute in failed_marginals}
        raise RuntimeError(f"coverage selection violates marginal drift gates: {details}")

    tmp = output_paths["selected"].with_name(f".{output_paths['selected'].name}.tmp.{os.getpid()}")
    pq.write_table(pa.Table.from_pylist(selected_rows, schema=output_schema), tmp, compression="zstd")
    os.replace(tmp, output_paths["selected"])
    _atomic_text(output_paths["manifest"], "".join(_canonical_json(row) + "\n" for row in manifest_rows))
    _atomic_text(output_paths["eligibility"], "".join(_canonical_json(row) + "\n" for row in eligibility_rows))
    _atomic_text(output_paths["review"], _review_markdown(selected))

    population_geometries = {item.geometry_signature for item in candidates}
    selected_geometries = {item.geometry_signature for item in selected}
    proof = {
        "runtime_eligible_shape_children_only": len(eligibility_rows)
        == sum(int(record["runtime_eligible_children"]) for record in source_records),
        "random_static_eligibility_checked_before_sampling": all(
            row["eligibility_universe"] == "shape_runtime_eligible_children"
            and type(row["random_static_eligible"]) is bool
            for row in eligibility_rows
        ),
        "unique_shape_child_uuid": len({item.child_uuid for item in candidates}) == len(candidates),
        "unique_canonical_parent_uuid": len({item.canonical_parent_uuid for item in candidates}) == len(candidates),
        "all_occupied_joint_cells_covered": selected_joint == population_joint,
        "all_shape_marginals_covered": all(
            {getattr(item, attribute) for item in selected} == {getattr(item, attribute) for item in candidates}
            for attribute in _SHAPE_MARGINALS
        ),
        "sampled_factory_is_shape_changed": all(
            item.sampled_factory_index in item.changed_factory_indices for item in selected
        ),
        "secondary_marginal_drift_gate_passed": not failed_marginals,
        "rows_preserved_exactly": all(
            selected_row == item.row for selected_row, item in zip(selected_rows, selected, strict=True)
        ),
        "training_approved": False,
    }
    if not all(value for key, value in proof.items() if key != "training_approved"):
        raise RuntimeError(f"resample proof failed: {proof}")
    summary = {
        "contract_version": CONTRACT_VERSION,
        "selection_version": SELECTION_VERSION,
        "provenance": {
            "git_commit": git_commit,
            "resampler_source_sha256": source_sha256["resampler_source_sha256"],
            "dependency_source_sha256": {
                "augment_prompt_tasks.py": source_sha256["augment_prompt_tasks.py"],
                "solve_value_coverage.py": source_sha256["solve_value_coverage.py"],
            },
        },
        "limit": limit,
        "source_runs": source_records,
        "runtime_eligible_shape_children": len(eligibility_rows),
        "random_static_eligible_shape_children": len(candidates),
        "selected_rows": len(selected),
        "one_sampled_shape_per_row": True,
        "sampled_shape_selection": "stable_sha256_mod_changed_direct_factory_count",
        "maximum_marginal_total_variation": MAX_MARGINAL_TOTAL_VARIATION,
        "marginal_repair_target": MARGINAL_REPAIR_TARGET,
        "marginal_audit": marginal_audit,
        "population": {
            "aggregate_size_buckets": _counter(candidates, "aggregate_size_bucket"),
            "sampled_size_buckets": _counter(candidates, "sampled_size_bucket"),
            "rank_buckets": _counter(candidates, "rank_bucket"),
            "aspect_buckets": _counter(candidates, "aspect_bucket"),
            "random_families": _counter(candidates, "assigned_random_family"),
            "variants": _counter(candidates, "variant"),
            "scale_buckets": _counter(candidates, "scale_bucket"),
            "logical_slot_buckets": _counter(candidates, "logical_slot_bucket"),
            "axis_position_buckets": _counter(candidates, "axis_position_bucket"),
            "explicit_batch_buckets": _counter(candidates, "explicit_batch_bucket"),
            "source_families": _counter(candidates, "source_family"),
            "operator_buckets": _counter(candidates, "operator_bucket"),
            "joint_cells": _joint_counter(candidates),
            "occupied_joint_cells": len(population_joint),
            "unique_exact_sampled_shapes": len({item.sampled_shape for item in candidates}),
            "unique_geometry_signatures": len(population_geometries),
            "sampled_numel_quantiles": _quantiles([item.sampled_numel for item in candidates]),
            "sampled_tensor_bytes_quantiles": _quantiles([item.sampled_tensor_bytes for item in candidates]),
            "aggregate_input_bytes_quantiles": _quantiles([item.aggregate_input_bytes for item in candidates]),
            "input_scale_quantiles": _quantiles([item.input_scale for item in candidates]),
        },
        "selected": {
            "aggregate_size_buckets": _counter(selected, "aggregate_size_bucket"),
            "sampled_size_buckets": _counter(selected, "sampled_size_bucket"),
            "rank_buckets": _counter(selected, "rank_bucket"),
            "aspect_buckets": _counter(selected, "aspect_bucket"),
            "random_families": _counter(selected, "assigned_random_family"),
            "variants": _counter(selected, "variant"),
            "scale_buckets": _counter(selected, "scale_bucket"),
            "logical_slot_buckets": _counter(selected, "logical_slot_bucket"),
            "axis_position_buckets": _counter(selected, "axis_position_bucket"),
            "explicit_batch_buckets": _counter(selected, "explicit_batch_bucket"),
            "source_families": _counter(selected, "source_family"),
            "operator_buckets": _counter(selected, "operator_bucket"),
            "joint_cells": _joint_counter(selected),
            "covered_joint_cells": len(selected_joint),
            "unique_exact_sampled_shapes": len({item.sampled_shape for item in selected}),
            "unique_geometry_signatures": len(selected_geometries),
            "geometry_coverage_fraction": len(selected_geometries) / len(population_geometries),
            "sampled_numel_quantiles": _quantiles([item.sampled_numel for item in selected]),
            "sampled_tensor_bytes_quantiles": _quantiles([item.sampled_tensor_bytes for item in selected]),
            "aggregate_input_bytes_quantiles": _quantiles([item.aggregate_input_bytes for item in selected]),
            "input_scale_quantiles": _quantiles([item.input_scale for item in selected]),
        },
        "proof": proof,
        "artifacts": {},
        "training_approved": False,
    }
    for key, path in output_paths.items():
        if key == "summary":
            continue
        summary["artifacts"][key] = {"path": str(path.resolve()), "sha256": _sha256_file(path)}
    _atomic_text(output_paths["summary"], json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = build_resample(
        args.output_dir,
        shape_runs=DEFAULT_SHAPE_RUNS,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
