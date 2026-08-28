#!/usr/bin/env python3
"""Shared checks for finalizing the CSP-DAG shape and dtype lanes."""

from __future__ import annotations

import collections
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tools.data.synthesize.augment_prompt_tasks import AUGMENTATION_METADATA_TYPE, _normalized_ast_sha256
from tools.data.synthesize.csp_dag_method import build_csp_dag_input_expansions as expansion
from tools.data.synthesize.csp_dag_method import generate_csp_dag as generator
from tools.data.synthesize.csp_dag_method import validate_csp_dag as row_validator
from tools.data.synthesize.csp_dag_method import validate_csp_dag_dataset as dataset_validator
from tools.data.synthesize.csp_dag_method import validate_csp_dag_repeatability as repeatability

_DATASET_RUNTIME_CONTRACT = "open_csp_dag_dataset_validation_v1"
_POST_SELECTION_MODES = {
    "shape": "shape_full_row",
    "dtype": "dtype_standalone_selected_child",
}
_COMMON_ADAPTER_CHECKS = frozenset(
    {
        "uuid_bound",
        "reference_hash_bound",
        "ast_hash_bound",
        "governance",
        "manifest_kind",
        "primary_intervention",
        "get_inputs_present",
        "model_present",
        "runtime_tensor_output",
        "runtime_finite",
        "runtime_dispatch_nonempty",
    }
)
_DTYPE_ADAPTER_CHECKS = _COMMON_ADAPTER_CHECKS | {
    "dtype_manifest_intervention_bound",
    "augmentation_manifest_bound",
    "floating_inputs_assigned_target",
    "device_inputs_assigned_target",
    "registered_floating_state_assigned_target",
    "no_unexpected_float32_output",
}
_REQUIRED_H20_SHARDS = expansion.REQUIRED_H20_SHARDS


def _sha256_file(path: Path) -> str:
    return expansion._sha256_file(path)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    return expansion._canonical_json(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return expansion._read_jsonl(path)


def _file(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _exact_file(binding: Mapping[str, Any], name: str, path: Path, context: str) -> None:
    declared = binding.get(name)
    if not isinstance(declared, Mapping):
        raise ValueError(f"{context} lacks {name} binding")
    expected = _file(path)
    if dict(declared) != expected:
        raise ValueError(f"{context} {name} binding mismatch")


def _identity(row: Mapping[str, Any]) -> tuple[str, str]:
    return expansion._identity(row)


def _status(record: Mapping[str, Any]) -> str:
    return str(record.get("status", "passed" if record.get("passed") else "failed"))


def _required_postselection_checks(lane: str, row: Mapping[str, Any], manifest: Mapping[str, Any]) -> frozenset[str]:
    """Derive checks from the production validator, not a record's claims."""
    if lane == "dtype":
        return frozenset(_DTYPE_ADAPTER_CHECKS)
    if lane != "shape":
        raise ValueError(f"unknown post-selection lane:{lane}")
    _, code = _identity(row)
    graph_checks, nodes, rank, shape = expansion.static_binding.typed_graph_context(manifest)
    lowering_checks = expansion.static_binding.source_lowering_checks(code, manifest, nodes, rank, shape)
    graph_runtime_checks = row_validator._graph_check(manifest)
    return frozenset(
        {
            "uuid_bound",
            "reference_hash_bound",
            "ast_hash_bound",
            "governance",
            "no_template_identifier",
            "no_repeated_provenance_padding",
            "signature_bound",
            "families_bound",
            "complexity_bound",
            "multiclass_bound",
            "multiclass_helper_reachable",
            "no_forbidden_composites",
            "normalization_variant_lowering_bound",
            "runtime_tensor_output",
            "runtime_finite",
            "runtime_dispatch_nonempty",
            *graph_checks.keys(),
            *lowering_checks.keys(),
            *(f"graph_{name}" for name in graph_runtime_checks),
        }
    )


def _exact_rows(expected: Sequence[Mapping[str, Any]], actual: Sequence[Mapping[str, Any]], context: str) -> None:
    if len(expected) != len(actual):
        raise ValueError(f"{context} row count mismatch")
    for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
        if _canonical(left) != _canonical(right):
            raise ValueError(f"{context} row mismatch:{index}")


def _extended_schema(schema: pa.Schema) -> pa.Schema:
    index = schema.get_field_index("extra_info")
    if index < 0 or not pa.types.is_struct(schema.field(index).type):
        raise ValueError("input schema has no extra_info struct")
    extra = schema.field(index)
    if extra.type.get_field_index("augmentation") >= 0:
        return schema
    fields = list(schema)
    fields[index] = pa.field(
        "extra_info",
        pa.struct([*extra.type, pa.field("augmentation", AUGMENTATION_METADATA_TYPE)]),
        nullable=extra.nullable,
        metadata=extra.metadata,
    )
    return pa.schema(fields, metadata=schema.metadata)


def _selection_summary(
    lane_dir: Path,
    *,
    selection_stage: str,
    candidates_path: Path,
    manifest_path: Path,
    selected_path: Path,
    selected_manifest_path: Path,
    runtime_paths: Sequence[Path],
    candidate_rows: Sequence[Mapping[str, Any]],
    candidate_manifests: Sequence[Mapping[str, Any]],
    runtime_records: Mapping[str, Mapping[str, Any]],
    parents_path: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]] | None, dict[str, Any]]:
    """Reconstruct the selected subsequence and require byte-bound selection metadata."""
    summary_path = lane_dir / "selection_summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("contract") != expansion.CONTRACT or summary.get("stage") != selection_stage:
        raise ValueError(f"{lane_dir.name} selection summary contract mismatch")
    if summary.get("review_only") is not True or summary.get("training_approved") is not False:
        raise ValueError(f"{lane_dir.name} selection summary governance mismatch")
    source_binding = summary.get("source_binding")
    if not isinstance(source_binding, Mapping):
        raise ValueError(f"{lane_dir.name} selection summary lacks source binding")
    _exact_file(source_binding, "candidates", candidates_path, f"{lane_dir.name} selection summary")
    _exact_file(source_binding, "manifest", manifest_path, f"{lane_dir.name} selection summary")
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"{lane_dir.name} selection summary lacks artifacts")
    _exact_file(artifacts, "selected", selected_path, f"{lane_dir.name} selection summary")
    _exact_file(artifacts, "selected_manifest", selected_manifest_path, f"{lane_dir.name} selection summary")
    declared_runtime = summary.get("runtime_sources")
    expected_runtime = [_file(path) for path in runtime_paths]
    if declared_runtime != expected_runtime:
        raise ValueError(f"{lane_dir.name} selection summary runtime source mismatch")
    if summary.get("candidate_rows") != len(candidate_rows) or summary.get("runtime_record_rows") != len(
        runtime_records
    ):
        raise ValueError(f"{lane_dir.name} selection summary row counts mismatch")

    selected_indices: list[int] = []
    statuses: collections.Counter[str] = collections.Counter()
    for index, row in enumerate(candidate_rows):
        uuid, _ = _identity(row)
        record = runtime_records.get(uuid)
        status = "missing" if record is None else _status(record)
        statuses[status] += 1
        if record is not None and record.get("passed") is True:
            selected_indices.append(index)
    if summary.get("status_counts") != dict(sorted(statuses.items())):
        raise ValueError(f"{lane_dir.name} selection summary status histogram mismatch")
    expected_rows = [dict(candidate_rows[index]) for index in selected_indices]
    expected_manifests = [
        {**candidate_manifests[index], "runtime_evidence": dict(runtime_records[_identity(candidate_rows[index])[0]])}
        for index in selected_indices
    ]
    selected_rows = pq.read_table(selected_path).to_pylist()
    selected_manifests = _read_jsonl(selected_manifest_path)
    _exact_rows(expected_rows, selected_rows, f"{lane_dir.name} selected parquet")
    _exact_rows(expected_manifests, selected_manifests, f"{lane_dir.name} selected manifest")
    if summary.get("selected_rows") != len(selected_rows):
        raise ValueError(f"{lane_dir.name} selection summary selected count mismatch")

    selected_parents: list[dict[str, Any]] | None = None
    if parents_path is not None:
        candidate_parents = pq.read_table(lane_dir / "parents.parquet").to_pylist()
        if len(candidate_parents) != len(candidate_rows):
            raise ValueError(f"{lane_dir.name} candidate parents are not aligned")
        selected_parents = pq.read_table(parents_path).to_pylist()
        _exact_rows(
            [dict(candidate_parents[index]) for index in selected_indices],
            selected_parents,
            f"{lane_dir.name} selected parents",
        )
    return selected_rows, selected_manifests, selected_parents, summary


def _post_selection_records(
    lane_dir: Path,
    runtime_dir: Path,
    selected_path: Path,
    selected_manifest_path: Path,
    selection_summary_path: Path,
    rows: Sequence[Mapping[str, Any]],
    manifests: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[Path], list[Path]]:
    """Validate the post-selection H20 dataset-validator evidence exactly."""
    if not runtime_dir.is_dir():
        raise FileNotFoundError(f"{lane_dir.name} post-selection H20 runtime directory:{runtime_dir}")
    shards = expansion._runtime_shards(runtime_dir, f"{lane_dir.name} post-selection")
    if len(shards) != _REQUIRED_H20_SHARDS:
        raise ValueError(f"{lane_dir.name} post-selection must contain {_REQUIRED_H20_SHARDS} H20 shards")
    expected_summaries = {
        path.with_name(path.name.replace(".records.jsonl", ".summary.json")) for _, _, path in shards
    }
    actual_summaries = set(runtime_dir.glob("*.summary.json"))
    if actual_summaries != expected_summaries:
        raise ValueError(f"{lane_dir.name} post-selection runtime summaries are missing or unknown")
    records: dict[str, dict[str, Any]] = {}
    summary_paths: list[Path] = []
    for shard_index, shard_count, records_path in shards:
        summary_path = records_path.with_name(records_path.name.replace(".records.jsonl", ".summary.json"))
        summary = json.loads(summary_path.read_text())
        summary_paths.append(summary_path)
        if summary.get("contract") != _DATASET_RUNTIME_CONTRACT:
            raise ValueError(f"{lane_dir.name} post-selection runtime contract mismatch:{records_path.name}")
        if summary.get("validation_mode") != _POST_SELECTION_MODES[lane_dir.name]:
            raise ValueError(f"{lane_dir.name} post-selection validation mode mismatch:{records_path.name}")
        if summary.get("device") != f"cuda:{shard_index}":
            raise ValueError(f"{lane_dir.name} post-selection runtime is not CUDA:{records_path.name}")
        identity = summary.get("gpu_identity")
        if (
            not isinstance(identity, Mapping)
            or identity.get("device") != f"cuda:{shard_index}"
            or "h20" not in str(identity.get("name", "")).lower()
            or not isinstance(identity.get("compute_capability"), list)
            or not isinstance(identity.get("total_memory_bytes"), int)
        ):
            raise ValueError(f"{lane_dir.name} post-selection H20 identity mismatch:{records_path.name}")
        if summary.get("shard_index") != shard_index or summary.get("shard_count") != shard_count:
            raise ValueError(f"{lane_dir.name} post-selection shard metadata mismatch:{records_path.name}")
        if summary.get("review_only") is not True or summary.get("training_approved") is not False:
            raise ValueError(f"{lane_dir.name} post-selection governance mismatch:{records_path.name}")
        context = summary.get("runtime_execution")
        if (
            not isinstance(context, Mapping)
            or context.get("device_type") != "cuda"
            or context.get("cudnn_benchmark") is not False
            or context.get("cudnn_deterministic") is not True
        ):
            raise ValueError(f"{lane_dir.name} post-selection deterministic CUDA context mismatch:{records_path.name}")
        bindings = summary.get("source_binding")
        if not isinstance(bindings, Mapping):
            raise ValueError(f"{lane_dir.name} post-selection source binding missing:{records_path.name}")
        _exact_file(bindings, "selected", selected_path, f"{lane_dir.name} post-selection:{records_path.name}")
        _exact_file(
            bindings, "manifest", selected_manifest_path, f"{lane_dir.name} post-selection:{records_path.name}"
        )
        _exact_file(bindings, "summary", selection_summary_path, f"{lane_dir.name} post-selection:{records_path.name}")
        expected_sources = {
            "validator": Path(dataset_validator.__file__).resolve(),
            "row_validator": Path(row_validator.__file__).resolve(),
            "execution_context": Path(repeatability.__file__).resolve(),
            "generator": Path(generator.__file__).resolve(),
        }
        for source_name, source_path in expected_sources.items():
            if bindings.get(source_name) != _file(source_path):
                raise ValueError(
                    f"{lane_dir.name} post-selection production source binding mismatch:"
                    f"{source_name}:{records_path.name}"
                )
        expected_adapter = {
            **_file(Path(expansion.__file__).resolve()),
            "contract": expansion.SOURCE_BINDING_CONTRACT,
        }
        if bindings.get("expansion_adapter") != expected_adapter:
            raise ValueError(f"{lane_dir.name} post-selection expansion adapter binding mismatch:{records_path.name}")
        expected_expansion_binding = expansion._artifact_binding(
            selected_path,
            selected_manifest_path,
            rows,
            _read_jsonl(selected_manifest_path),
            stage=lane_dir.name,
        )
        if bindings.get("expansion_selected_binding") != expected_expansion_binding:
            raise ValueError(f"{lane_dir.name} post-selection selected adapter binding mismatch:{records_path.name}")
        _exact_file(summary, "records", records_path, f"{lane_dir.name} post-selection:{records_path.name}")
        shard_records = _read_jsonl(records_path)
        expected_positions = list(range(shard_index, len(rows), shard_count))
        if len(shard_records) != len(expected_positions) or summary.get("rows") != len(shard_records):
            raise ValueError(f"{lane_dir.name} post-selection row count mismatch:{records_path.name}")
        if summary.get("passed") != len(shard_records) or summary.get("failed") != 0:
            raise ValueError(f"{lane_dir.name} post-selection summary has a failure:{records_path.name}")
        positions = {
            "first": expected_positions[0] if expected_positions else None,
            "last": expected_positions[-1] if expected_positions else None,
        }
        if summary.get("positions") != positions:
            raise ValueError(f"{lane_dir.name} post-selection positions mismatch:{records_path.name}")
        for position, record in zip(expected_positions, shard_records, strict=True):
            uuid, _ = _identity(rows[position])
            if record.get("position") != position or record.get("uuid") != uuid or record.get("passed") is not True:
                raise ValueError(
                    f"{lane_dir.name} post-selection record identity/status mismatch:{records_path.name}:{position}"
                )
            if record.get("errors") not in ([], None) or record.get("failed_checks") not in ([], None):
                raise ValueError(f"{lane_dir.name} post-selection record errors:{records_path.name}:{position}")
            checks = record.get("checks")
            runtime = record.get("runtime")
            if not isinstance(checks, Mapping) or not checks or not all(value is True for value in checks.values()):
                raise ValueError(
                    f"{lane_dir.name} post-selection record checks mismatch:{records_path.name}:{position}"
                )
            required_checks = _required_postselection_checks(lane_dir.name, rows[position], manifests[position])
            missing_checks = sorted(name for name in required_checks if checks.get(name) is not True)
            if missing_checks:
                raise ValueError(
                    f"{lane_dir.name} post-selection record lacks production checks:"
                    f"{records_path.name}:{position}:{','.join(missing_checks)}"
                )
            if (
                not isinstance(runtime, Mapping)
                or not isinstance(runtime.get("dispatch_count"), int)
                or runtime["dispatch_count"] <= 0
            ):
                raise ValueError(
                    f"{lane_dir.name} post-selection dispatch evidence missing:{records_path.name}:{position}"
                )
            if uuid in records:
                raise ValueError(f"{lane_dir.name} post-selection duplicate UUID:{uuid}")
            records[uuid] = dict(record)
    if len(records) != len(rows):
        raise ValueError(f"{lane_dir.name} post-selection runtime coverage is incomplete")
    return records, [path for _, _, path in shards], summary_paths


def _assert_global_identity(lanes: Sequence[tuple[str, Sequence[Mapping[str, Any]], Mapping[str, str]]]) -> None:
    """Require global UUID/reference uniqueness and scoped AST uniqueness.

    Shape and dtype can retain the same operator DAG by design.  Therefore
    an AST collision is only admissible within one base lineage and only across
    different intervention lanes; a collision across two independent base
    programs remains a dedup failure.
    """
    uuids: set[str] = set()
    references: set[str] = set()
    ast_occurrences: dict[str, list[tuple[str, str]]] = {}
    for lane, rows, roots in lanes:
        for row in rows:
            uuid, code = _identity(row)
            reference = _sha256_text(code)
            ast_hash = _normalized_ast_sha256(code)
            if uuid in uuids:
                raise ValueError(f"global UUID collision:{uuid}")
            if reference in references:
                raise ValueError(f"global reference collision:{uuid}")
            uuids.add(uuid)
            references.add(reference)
            ast_occurrences.setdefault(ast_hash, []).append((lane, roots[uuid]))
    for ast_hash, entries in ast_occurrences.items():
        if len(entries) <= 1:
            continue
        roots = {root for _, root in entries}
        stages = [lane for lane, _ in entries]
        if len(roots) != 1 or len(set(stages)) != len(stages):
            raise ValueError(f"global normalized AST collision:{ast_hash}")
