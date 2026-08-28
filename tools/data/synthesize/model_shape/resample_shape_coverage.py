#!/usr/bin/env python3
"""Build the full one-child-per-parent runtime-safe shape population.

Each source row can expose several direct input tensors.  This tool assigns one
stable sampled tensor occurrence to the row, then audits aggregate input size,
sampled-tensor size, rank, aspect ratio, coarse ordered geometry, and the
random/value family that the downstream solver would assign.  It never rewrites
a row.  Runtime-ineligible shape children fail closed before sampling, and at
most one runtime-safe child is retained for each canonical parent.

The repository defaults intentionally name the three static production lanes,
the original model production lanes, and the runtime-validated low/128K
residual lanes.  A normal invocation therefore needs only an output directory;
the downstream random/value lane can consume ``selected.parquet`` directly.
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

CONTRACT_VERSION = "shape_runtime_single_changed_tensor_coverage_resample_v5"
SELECTION_VERSION = "shape_full_population_one_child_per_parent_v5"
MAX_MARGINAL_TOTAL_VARIATION = 0.05
SMALL_NUMEL_THRESHOLD = 1_000_000
MAX_SMALL_SHAPE_SHARE = 0.10
MIB = 1024**2
MAX_AGGREGATE_INPUT_BYTES = 4 * 1024**3
_PROFILED_TENSOR_FACTORY_NAME = "__profiled_tensor_index__"

_REPO_ROOT = Path(__file__).resolve().parents[4]
KERNELBENCH_BASELINE = _REPO_ROOT / "Data/external/converted/kernelbench_level1_2_3.reference.parquet"
KERNELBENCH_BASELINE_SHA256 = "b4de490253a0f5a1f5e971f0f723cffd1de2cc2506ceabe97a369804e3a3d7f2"
DEFAULT_SHAPE_RUNS = (
    _REPO_ROOT / "Data/prompt_tvm_v4/shape_solver_multidim_v4_byte_targets_v1/run.full53896",
    _REPO_ROOT / "Data/prompt_tvm_v4/shape_solver_variable_multislot_v5/run.recoverable12318.balanced_v7",
    _REPO_ROOT / "Data/prompt_tvm_v4/shape_solver_variable_multislot_v8/run.remaining22566.balanced_v7",
    _REPO_ROOT / "Data/prompt_tvm_v4/shape_model_hardtail_v2/run.full17864",
    _REPO_ROOT / "Data/prompt_tvm_v4/shape_model_full_residual_v3/run.measurable12674",
    _REPO_ROOT / "Data/prompt_tvm_v4/shape_model_relaxed_residual_v2/run.full2321",
    _REPO_ROOT / "Data/prompt_tvm_v4/shape_model_retry_single_v3/run.full6460",
    _REPO_ROOT / "Data/prompt_tvm_v4/shape_model_retry_single_v3/run.residual_low128k",
    _REPO_ROOT / "Data/prompt_tvm_v4/shape_model_retry_single_v3/run.final_low128k",
    _REPO_ROOT / "Data/prompt_tvm_v4/shape_model_unprofiled_v3/run.residual_low128k",
    _REPO_ROOT / "Data/prompt_tvm_v4/shape_model_unprofiled_v3/run.final_low128k",
    _REPO_ROOT / "Data/prompt_tvm_v4/shape_model_unprofiled_v3/run.canary128.hash",
    _REPO_ROOT / "Data/prompt_tvm_v4/shape_model_unprofiled_v3/run.canary32.low128k",
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
    touches_explicit_batch: bool | None

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


def _load_model_child_profiles(path: Path) -> dict[str, ChildShapeProfile]:
    """Reconstruct the same profile from the model lane's exact slot manifest."""

    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid model-shape manifest: {path}") from exc
    raw_decisions = manifest.get("decisions")
    if not isinstance(raw_decisions, list):
        raise ValueError(f"model-shape manifest has no decisions: {path}")
    profiles: dict[str, ChildShapeProfile] = {}
    for decision in raw_decisions:
        if not isinstance(decision, Mapping) or decision.get("accepted") is not True:
            continue
        child_uuid = decision.get("child_uuid")
        slots = _nested(decision, "solver.slots")
        if not isinstance(child_uuid, str) or not child_uuid or not isinstance(slots, list) or not slots:
            raise ValueError(f"invalid accepted model-shape decision: {child_uuid!r}")
        if child_uuid in profiles:
            raise ValueError(f"duplicate model-shape child decision: {child_uuid}")
        factory_rows: dict[tuple[str, int], list[Mapping[str, Any]]] = collections.defaultdict(list)
        leading: list[bool] = []
        for slot in slots:
            occurrences = slot.get("occurrences") if isinstance(slot, Mapping) else None
            if not isinstance(occurrences, list) or not occurrences:
                raise ValueError(f"model-shape slot has no occurrences: {child_uuid}")
            for occurrence in occurrences:
                if not isinstance(occurrence, Mapping):
                    raise ValueError(f"invalid model-shape occurrence: {child_uuid}")
                try:
                    axis = int(occurrence["axis"])
                    rank = int(occurrence["rank"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(f"invalid model-shape occurrence fields: {child_uuid}") from exc
                if rank <= 0 or axis < 0 or axis >= rank:
                    raise ValueError(f"invalid model-shape occurrence geometry: {child_uuid}")
                factory_index = occurrence.get("factory_index")
                tensor_index = occurrence.get("tensor_index")
                if type(factory_index) is int and tensor_index is None:
                    index_kind = "factory"
                    index = factory_index
                    factory_name = occurrence.get("factory_name")
                    if not isinstance(factory_name, str) or not factory_name:
                        raise ValueError(f"invalid model-shape factory name: {child_uuid}")
                elif type(tensor_index) is int and factory_index is None:
                    index_kind = "profiled_tensor"
                    index = tensor_index
                    factory_name = _PROFILED_TENSOR_FACTORY_NAME
                    child_profile = decision.get("child_input_profile")
                    tensors = child_profile.get("tensors") if isinstance(child_profile, Mapping) else None
                    if not isinstance(tensors, list) or index < 0 or index >= len(tensors):
                        raise ValueError(f"invalid model-shape profiled tensor index: {child_uuid}")
                    tensor = tensors[index]
                    if (
                        not isinstance(tensor, list)
                        or len(tensor) != 3
                        or not isinstance(tensor[0], list)
                        or len(tensor[0]) != rank
                        or tensor[0][axis] != slot.get("new_value")
                    ):
                        raise ValueError(f"inconsistent model-shape profiled tensor: {child_uuid}:{index}")
                else:
                    raise ValueError(f"ambiguous model-shape occurrence index: {child_uuid}")
                if index < 0:
                    raise ValueError(f"invalid model-shape occurrence geometry: {child_uuid}")
                factory_rows[(index_kind, index)].append(occurrence)
                leading.append(axis == 0)
        changed_factories: list[ChangedFactoryEvidence] = []
        for (index_kind, index), occurrences in sorted(factory_rows.items()):
            names = (
                {str(item["factory_name"]) for item in occurrences}
                if index_kind == "factory"
                else {_PROFILED_TENSOR_FACTORY_NAME}
            )
            ranks = {int(item["rank"]) for item in occurrences}
            axes = tuple(sorted({int(item["axis"]) for item in occurrences}))
            if len(names) != 1 or len(ranks) != 1 or not axes:
                raise ValueError(f"inconsistent model-shape factory evidence: {child_uuid}:{index}")
            changed_factories.append(
                ChangedFactoryEvidence(
                    factory_index=index,
                    factory_name=next(iter(names)),
                    rank=next(iter(ranks)),
                    changed_axes=axes,
                )
            )
        logical_slot_count = decision.get("logical_slot_count")
        if logical_slot_count != len(slots):
            raise ValueError(f"model-shape logical slot count mismatch: {child_uuid}")
        variant = str(decision.get("variant", "")).split(":")[-1]
        if variant not in {"medium", "large"}:
            raise ValueError(f"invalid model-shape variant: {child_uuid}:{variant}")
        try:
            before = int(decision["input_bytes_before"])
            after = int(decision["input_bytes_after"])
            scale = float(decision["input_scale"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid model-shape storage evidence: {child_uuid}") from exc
        profiles[child_uuid] = ChildShapeProfile(
            child_uuid=child_uuid,
            variant=variant,
            input_bytes_before=before,
            input_bytes_after=after,
            input_scale=scale,
            logical_slots=len(slots),
            changed_factories=tuple(changed_factories),
            touches_leading=any(leading),
            touches_nonleading=not all(leading),
            # The model manifest proves exact factory/axis occurrences but does
            # not claim that a linked module constant was named like a batch.
            touches_explicit_batch=None,
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


def _load_runtime_lane(
    run_dir: Path,
) -> tuple[
    set[str],
    Mapping[str, Any],
    Path,
    Path,
    Path,
    dict[str, ChildShapeProfile],
    str,
]:
    """Load either a static-solver lane or the model lane without weakening evidence."""

    static_summary_path = run_dir / "analysis/summary.json"
    if static_summary_path.is_file():
        eligible, summary, children_path = _load_runtime_eligible(run_dir)
        profile_path = run_dir / "analysis/changed_slots.tsv"
        return (
            eligible,
            summary,
            children_path,
            static_summary_path,
            profile_path,
            _load_child_profiles(profile_path),
            "static_solver",
        )

    runtime_summary_path = run_dir / "analysis/runtime_final.json"
    accepted_path = run_dir / "runtime/accepted.parquet"
    accepted_manifest_path = run_dir / "runtime/accepted_manifest.json"
    static_manifest_path = run_dir / "static/manifest.json"
    static_children_path = run_dir / "static/children.parquet"
    for path in (
        runtime_summary_path,
        accepted_path,
        accepted_manifest_path,
        static_manifest_path,
        static_children_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(f"shape run lacks a complete runtime lane: {run_dir} ({path.name})")
    try:
        runtime_summary = json.loads(runtime_summary_path.read_text(encoding="utf-8"))
        accepted_manifest = json.loads(accepted_manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid model runtime evidence: {run_dir}") from exc
    if runtime_summary.get("contract_version") != "model_shape_runtime_final_analysis_v1":
        raise ValueError(f"unexpected model runtime summary contract: {run_dir}")
    for verifier_name in ("reference_verifier", "region_verifier"):
        verifier = runtime_summary.get(verifier_name)
        if not isinstance(verifier, Mapping) or verifier.get("ok") is not True:
            raise ValueError(f"model runtime {verifier_name} is incomplete: {run_dir}")
    if accepted_manifest.get("contract_version") != "model_shape_runtime_accepted_v1":
        raise ValueError(f"unexpected model accepted manifest contract: {run_dir}")
    if runtime_summary.get("accepted_sha256") != _sha256_file(accepted_path):
        raise ValueError(f"model runtime accepted parquet SHA mismatch: {run_dir}")
    if runtime_summary.get("accepted_manifest_sha256") != _sha256_file(accepted_manifest_path):
        raise ValueError(f"model runtime accepted manifest SHA mismatch: {run_dir}")
    if accepted_manifest.get("source_manifest_sha256") != _sha256_file(static_manifest_path):
        raise ValueError(f"model runtime source manifest SHA mismatch: {run_dir}")
    if accepted_manifest.get("source_children_sha256") != _sha256_file(static_children_path):
        raise ValueError(f"model runtime source children SHA mismatch: {run_dir}")
    if accepted_manifest.get("accepted_sha256") != _sha256_file(accepted_path):
        raise ValueError(f"model accepted manifest parquet SHA mismatch: {run_dir}")
    raw_rows = accepted_manifest.get("rows")
    if not isinstance(raw_rows, list):
        raise ValueError(f"model accepted manifest has no rows: {run_dir}")
    eligible = [row.get("child_uuid") for row in raw_rows if isinstance(row, Mapping)]
    if any(not isinstance(value, str) or not value for value in eligible) or len(set(eligible)) != len(eligible):
        raise ValueError(f"model accepted UUIDs are invalid or duplicated: {run_dir}")
    if accepted_manifest.get("accepted_children") != len(eligible):
        raise ValueError(f"model accepted child count mismatch: {run_dir}")
    if pq.ParquetFile(accepted_path).metadata.num_rows != len(eligible):
        raise ValueError(f"model accepted parquet row count mismatch: {run_dir}")
    profiles = _load_model_child_profiles(static_manifest_path)
    missing_profiles = sorted(set(eligible) - set(profiles))
    if missing_profiles:
        raise ValueError(f"model accepted children lack slot profiles: {missing_profiles[:3]}")
    return (
        set(eligible),
        runtime_summary,
        accepted_path,
        runtime_summary_path,
        static_manifest_path,
        profiles,
        "model_shape",
    )


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
        if evidence.factory_name == _PROFILED_TENSOR_FACTORY_NAME:
            return None, "profiled_tensor_index_not_statically_mapped"
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
                "unknown_explicit_batch"
                if profile.touches_explicit_batch is None
                else "touches_explicit_batch" if profile.touches_explicit_batch else "no_explicit_batch"
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
    if limit == len(candidates):
        return sorted(candidates, key=lambda item: (item.lane_index, item.source_row_index))
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
    return sorted(chosen.values(), key=lambda item: (item.lane_index, item.source_row_index))


def _choose_one_per_parent(candidates: Sequence[Candidate]) -> tuple[list[Candidate], int]:
    """Resolve the rare multi-variant parent while favoring underused shape cells."""

    groups: dict[str, list[Candidate]] = collections.defaultdict(list)
    for candidate in candidates:
        groups[candidate.canonical_parent_uuid].append(candidate)
    selected: list[Candidate] = []
    multi_groups: list[tuple[str, list[Candidate]]] = []
    cell_counts: collections.Counter[tuple[str, ...]] = collections.Counter()
    size_counts: collections.Counter[str] = collections.Counter()
    variant_counts: collections.Counter[str] = collections.Counter()
    family_counts: collections.Counter[str] = collections.Counter()
    exact_counts: collections.Counter[tuple[int, ...]] = collections.Counter()

    def add(candidate: Candidate) -> None:
        selected.append(candidate)
        cell_counts[candidate.main_cell] += 1
        size_counts[candidate.sampled_size_bucket] += 1
        variant_counts[candidate.variant] += 1
        family_counts[candidate.assigned_random_family] += 1
        exact_counts[candidate.sampled_shape] += 1

    for parent_uuid, options in groups.items():
        if len(options) == 1:
            add(options[0])
        else:
            multi_groups.append((parent_uuid, options))
    multi_groups.sort(key=lambda item: _stable_sha256(SELECTION_VERSION, item[0]))
    for _, options in multi_groups:
        choice = min(
            options,
            key=lambda item: (
                cell_counts[item.main_cell],
                size_counts[item.sampled_size_bucket],
                variant_counts[item.variant],
                family_counts[item.assigned_random_family],
                exact_counts[item.sampled_shape],
                item.selection_sha256,
            ),
        )
        add(choice)
    if len({item.canonical_parent_uuid for item in selected}) != len(selected):
        raise AssertionError("one-child-per-parent selection retained a duplicate parent")
    return selected, len(multi_groups)


def _counter(items: Sequence[Candidate], key: str) -> dict[str, int]:
    counts = collections.Counter(str(getattr(item, key)) for item in items)
    return dict(sorted(counts.items()))


def _distribution_tvd(left: Mapping[str, int], right: Mapping[str, int]) -> float:
    left_total, right_total = sum(left.values()), sum(right.values())
    if left_total <= 0 or right_total <= 0:
        raise ValueError("distribution TVD requires two non-empty count maps")
    values = set(left) | set(right)
    return 0.5 * sum(abs(left.get(value, 0) / left_total - right.get(value, 0) / right_total) for value in values)


def _one_shape_per_question_cover(
    questions: Sequence[Mapping[str, Any]],
    support: set[tuple[str, str]],
) -> tuple[list[int], int]:
    """Choose one compatible factory per question while retaining marginal support."""

    assignments: dict[int, int] = {}
    covered: set[tuple[str, str]] = set()

    def search() -> bool:
        missing = support - covered
        if not missing:
            return True
        candidate_counts: dict[tuple[str, str], int] = {}
        for label in missing:
            candidate_counts[label] = sum(
                label in choice["labels"]
                for question_index, question in enumerate(questions)
                if question_index not in assignments
                for choice in question["choices"]
            )
        target = min(missing, key=lambda label: (candidate_counts[label], label))
        options: list[tuple[int, int, Mapping[str, Any]]] = []
        for question_index, question in enumerate(questions):
            if question_index in assignments:
                continue
            for choice_index, choice in enumerate(question["choices"]):
                if target in choice["labels"]:
                    options.append((question_index, choice_index, choice))
        options.sort(
            key=lambda item: (
                -len(item[2]["labels"] & missing),
                _stable_sha256(
                    "kernelbench_question_support_choice_v1",
                    questions[item[0]]["sample_sha256"],
                    item[2]["factory_index"],
                ),
            )
        )
        for question_index, choice_index, choice in options:
            assignments[question_index] = choice_index
            newly_covered = choice["labels"] - covered
            covered.update(newly_covered)
            if search():
                return True
            covered.difference_update(newly_covered)
            del assignments[question_index]
        return False

    if not search():
        raise ValueError("one-shape-per-question selection cannot retain KernelBench marginal support")
    coverage_forced_questions = len(assignments)
    for question_index, question in enumerate(questions):
        if question_index in assignments:
            continue
        assignments[question_index] = int(question["sample_sha256"][:16], 16) % len(question["choices"])
    return [assignments[index] for index in range(len(questions))], coverage_forced_questions


def _kernelbench_profile(path: Path) -> dict[str, Any]:
    if _sha256_file(path) != KERNELBENCH_BASELINE_SHA256:
        raise ValueError(f"KernelBench baseline SHA mismatch: {path}")
    parquet = pq.ParquetFile(path)
    all_counts = {
        "sampled_size_bucket": collections.Counter(),
        "rank_bucket": collections.Counter(),
        "aspect_bucket": collections.Counter(),
    }
    compatible_occurrence_counts = {key: collections.Counter() for key in all_counts}
    compatible_question_counts = {key: collections.Counter() for key in all_counts}
    resolved_occurrences = 0
    compatible_occurrences = 0
    resolved_rows = 0
    unresolved_rows = 0
    rows_without_compatible_shape = 0
    compatible_questions: list[dict[str, Any]] = []
    row_index = 0
    for batch in parquet.iter_batches(batch_size=64, use_threads=False):
        for row in batch.to_pylist():
            code = _nested(row, "reward_model.ground_truth")
            if not isinstance(code, str):
                unresolved_rows += 1
                row_index += 1
                continue
            try:
                tree = ast.parse(code)
                get_inputs = _top_level_function(tree, "get_inputs")
                records, _ = _factory_records(tree, get_inputs, _module_constant_environment(tree))
            except (SyntaxError, ValueError):
                unresolved_rows += 1
                row_index += 1
                continue
            if not records:
                unresolved_rows += 1
                row_index += 1
                continue
            resolved_rows += 1
            compatible_records: list[dict[str, Any]] = []
            for factory_index, record in enumerate(records):
                numel = math.prod(record.shape)
                labels = {
                    "sampled_size_bucket": _sampled_size_bucket(numel),
                    "rank_bucket": _rank_bucket(len(record.shape)),
                    "aspect_bucket": _aspect_bucket(record.shape),
                }
                for key, label in labels.items():
                    all_counts[key][label] += 1
                resolved_occurrences += 1
                compatible = record.bytes <= MAX_AGGREGATE_INPUT_BYTES and labels["aspect_bucket"] != "gt1000"
                if compatible:
                    for key, label in labels.items():
                        compatible_occurrence_counts[key][label] += 1
                    compatible_records.append(
                        {
                            "factory_index": factory_index,
                            "label_map": labels,
                            "labels": frozenset(labels.items()),
                            "numel": numel,
                        }
                    )
                    compatible_occurrences += 1
            if compatible_records:
                row_uuid = _nested(row, "extra_info.uuid", "")
                code_sha256 = hashlib.sha256(code.encode("utf-8")).hexdigest()
                compatible_questions.append(
                    {
                        "sample_sha256": _stable_sha256(
                            "kernelbench_contract_compatible_one_shape_per_question_v2",
                            KERNELBENCH_BASELINE_SHA256,
                            row_index,
                            row_uuid,
                            code_sha256,
                        ),
                        "choices": compatible_records,
                    }
                )
            else:
                rows_without_compatible_shape += 1
            row_index += 1
    if not compatible_occurrences:
        raise ValueError("KernelBench baseline has no contract-compatible resolved shapes")
    if not compatible_questions:
        raise ValueError("KernelBench baseline has no contract-compatible question-level shape sample")
    occurrence_support = {(key, label) for key, counts in compatible_occurrence_counts.items() for label in counts}
    question_choices, coverage_forced_questions = _one_shape_per_question_cover(
        compatible_questions, occurrence_support
    )
    compatible_question_numel: list[int] = []
    for question, choice_index in zip(compatible_questions, question_choices, strict=True):
        choice = question["choices"][choice_index]
        for key, label in choice["label_map"].items():
            compatible_question_counts[key][label] += 1
        compatible_question_numel.append(choice["numel"])
    question_support = {(key, label) for key, counts in compatible_question_counts.items() for label in counts}
    if question_support != occurrence_support:
        raise AssertionError("KernelBench one-shape-per-question sample lost compatible marginal support")
    return {
        "path": str(path.resolve()),
        "sha256": KERNELBENCH_BASELINE_SHA256,
        "rows": parquet.metadata.num_rows,
        "resolved_rows": resolved_rows,
        "resolved_factory_occurrences": resolved_occurrences,
        "unresolved_rows": unresolved_rows,
        "compatible_contract": ("factory storage <=4 GiB and, for rank>=2, largest dimension <=1000x second-largest"),
        "compatible_factory_occurrences": compatible_occurrences,
        "compatible_questions": len(compatible_question_numel),
        "resolved_rows_without_compatible_shape": rows_without_compatible_shape,
        "all_counts": {key: dict(sorted(value.items())) for key, value in all_counts.items()},
        "compatible_factory_occurrence_counts": {
            key: dict(sorted(value.items())) for key, value in compatible_occurrence_counts.items()
        },
        "compatible_one_shape_per_question_sampling": (
            "marginal_support_backtracking_then_stable_sha256_mod_compatible_factory_count_v2"
        ),
        "compatible_one_shape_per_question_support_preserved": True,
        "compatible_one_shape_per_question_coverage_forced_questions": coverage_forced_questions,
        "compatible_one_shape_per_question_counts": {
            key: dict(sorted(value.items())) for key, value in compatible_question_counts.items()
        },
        "compatible_one_shape_per_question_small_share": (
            sum(value < SMALL_NUMEL_THRESHOLD for value in compatible_question_numel) / len(compatible_question_numel)
        ),
    }


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


def build_resample(output_dir: Path, *, shape_runs: Sequence[Path], overwrite: bool) -> dict[str, Any]:
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
    for lane_index, run_dir in enumerate(shape_runs):
        (
            eligible_uuids,
            summary,
            children_path,
            summary_path,
            profile_path,
            child_profiles,
            lane_kind,
        ) = _load_runtime_lane(run_dir)
        children_sha256 = _sha256_file(children_path)
        lane_id = f"shape_lane_{lane_index}_{_stable_sha256(run_dir.resolve(), children_sha256)[:12]}"
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
                        "canonical_parent_uuid": (
                            candidate.canonical_parent_uuid
                            if candidate is not None
                            else _nested(row, "extra_info.v4.parent_uuid")
                        ),
                        "eligibility_universe": "shape_runtime_eligible_children",
                        "random_static_eligible": candidate is not None,
                        "reason": reason,
                    }
                )
                if candidate is not None:
                    if candidate.child_uuid in seen_child_uuids:
                        raise ValueError(f"duplicate shape child UUID across lanes: {candidate.child_uuid}")
                    seen_child_uuids.add(candidate.child_uuid)
                    candidates.append(candidate)
                    lane_candidate_count += 1
                row_index += 1
        if found_runtime_eligible != eligible_uuids:
            missing = sorted(eligible_uuids - found_runtime_eligible)
            raise ValueError(f"runtime eligible UUIDs absent from children for {run_dir}: {missing[:3]}")
        source_records.append(
            {
                "lane_id": lane_id,
                "lane_kind": lane_kind,
                "run_dir": str(run_dir.resolve()),
                "run_summary_path": str(summary_path.resolve()),
                "run_summary_sha256": _sha256_file(summary_path),
                "profile_evidence_path": str(profile_path.resolve()),
                "profile_evidence_sha256": _sha256_file(profile_path),
                "children_path": str(children_path.resolve()),
                "children_sha256": children_sha256,
                "runtime_eligible_children": len(eligible_uuids),
                "random_static_eligible_children": lane_candidate_count,
                "shape_contract": summary.get("schema_version", summary.get("contract_version")),
            }
        )
    if output_schema is None:
        raise ValueError("no shape run was provided")
    raw_candidate_count = len(candidates)
    candidates, multi_variant_parents = _choose_one_per_parent(candidates)
    selected = _select(candidates, len(candidates))
    selected_child_uuids = {item.child_uuid for item in selected}
    for row in eligibility_rows:
        row["selected_for_canonical_parent"] = (
            row["shape_child_uuid"] in selected_child_uuids if row["random_static_eligible"] else None
        )
    limit = len(selected)
    small_shape_count = sum(item.sampled_numel < SMALL_NUMEL_THRESHOLD for item in selected)
    small_shape_share = small_shape_count / len(selected)
    if small_shape_share >= MAX_SMALL_SHAPE_SHARE:
        raise RuntimeError(
            f"small sampled-shape share must be below {MAX_SMALL_SHAPE_SHARE:.0%}: "
            f"{small_shape_count}/{len(selected)}={small_shape_share:.4%}"
        )
    kernelbench = _kernelbench_profile(KERNELBENCH_BASELINE)
    selected_kernelbench_counts = {
        "sampled_size_bucket": _counter(selected, "sampled_size_bucket"),
        "rank_bucket": _counter(selected, "rank_bucket"),
        "aspect_bucket": _counter(selected, "aspect_bucket"),
    }
    missing_kernelbench_support = {
        key: sorted(
            set(kernelbench["compatible_one_shape_per_question_counts"][key]) - set(selected_kernelbench_counts[key])
        )
        for key in selected_kernelbench_counts
    }
    kernelbench_support_passed = not any(missing_kernelbench_support.values())
    if not kernelbench_support_passed:
        raise RuntimeError(
            f"selected shape rows miss contract-compatible KernelBench support: {missing_kernelbench_support}"
        )
    kernelbench_tvd = {
        key: _distribution_tvd(
            selected_kernelbench_counts[key], kernelbench["compatible_one_shape_per_question_counts"][key]
        )
        for key in selected_kernelbench_counts
    }
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
        "one_runtime_child_per_canonical_parent": len(selected) == len(candidates),
        "kernelbench_contract_compatible_support_covered": kernelbench_support_passed,
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
        "random_static_eligible_shape_children_before_parent_dedup": raw_candidate_count,
        "random_static_eligible_shape_children": len(candidates),
        "multi_variant_parents_resolved": multi_variant_parents,
        "selected_rows": len(selected),
        "one_sampled_shape_per_row": True,
        "sampled_shape_selection": "stable_sha256_mod_changed_direct_factory_count",
        "parent_variant_selection": ("underused_joint_cell_then_size_variant_random_family_exact_shape_then_sha256"),
        "small_shape_contract": {
            "measurement": "one stable sampled shape-changed direct-input factory per selected row",
            "threshold_numel_exclusive": SMALL_NUMEL_THRESHOLD,
            "required_share_strictly_below": MAX_SMALL_SHAPE_SHARE,
            "count": small_shape_count,
            "share": small_shape_share,
            "passed": small_shape_share < MAX_SMALL_SHAPE_SHARE,
        },
        "kernelbench_comparison": {
            "comparison_unit": "one_contract_compatible_shape_per_question_v1",
            "baseline": kernelbench,
            "selected_counts": selected_kernelbench_counts,
            "missing_contract_compatible_support": missing_kernelbench_support,
            "support_coverage_passed": kernelbench_support_passed,
            "distribution_tvd": kernelbench_tvd,
            "interpretation": (
                "Both sides use one stable shape per question. TVD is reported, not minimized by "
                "dropping otherwise eligible parent questions; support coverage and the <10% "
                "small-shape gate are hard requirements."
            ),
        },
        "maximum_marginal_total_variation": MAX_MARGINAL_TOTAL_VARIATION,
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
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    summary = build_resample(
        args.output_dir,
        shape_runs=DEFAULT_SHAPE_RUNS,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
