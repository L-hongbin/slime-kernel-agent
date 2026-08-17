"""Runtime authority and selected-subsequence checks for the final review."""

from __future__ import annotations

import collections
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from tools.data.synthesize.review_only.csp_dag_review_common import (
    EXPANSION_CONTRACT,
    POST_CONTRACT,
    POST_MODES,
    _bound_source_errors,
    _canonical,
    _exact_file_binding,
    _load_jsonl,
    _rows_and_manifests,
    _sha256_file,
    _shard_paths,
    _uuid,
)


def _dtype_audit_errors(
    lane_dir: Path,
    runtime_paths: Sequence[Path],
    candidate_rows: int,
    status_counts: Mapping[str, int],
) -> list[str]:
    """Bind the final review to the dedicated full dtype runtime audit."""
    path = lane_dir / "review_runtime_independent/audit.json"
    sidecar = path.with_suffix(".json.sha256")
    if not path.is_file() or not sidecar.is_file():
        return ["dtype: independent runtime audit or sidecar missing"]
    audit = json.loads(path.read_text())
    errors: list[str] = []
    if sidecar.read_text().split()[0] != _sha256_file(path):
        errors.append("dtype: independent runtime audit sidecar mismatch")
    if (
        audit.get("contract") != "independent_dtype_v6_v5_runtime_audit_v3"
        or audit.get("passed") is not True
        or audit.get("findings") != []
        or audit.get("require_complete") is not True
        or audit.get("review_only") is not True
        or audit.get("training_approved") is not False
        or audit.get("candidate_rows") != candidate_rows
        or audit.get("observed_runtime_rows") != candidate_rows
        or audit.get("status_counts") != dict(status_counts)
    ):
        errors.append("dtype: independent runtime audit verdict/count mismatch")
    inputs = audit.get("input_bindings", {})
    required_inputs = [
        lane_dir / "parents.parquet",
        lane_dir / "candidates.parquet",
        lane_dir / "manifest.jsonl",
        lane_dir / "summary.json",
        *runtime_paths,
    ]
    for source in required_inputs:
        error = _exact_file_binding(inputs.get(str(source.resolve())), source, f"dtype audit:{source.name}")
        if error:
            errors.append(error)
    return errors


def _preselection(
    run_root: Path,
    lane: str,
    rows: Sequence[Mapping[str, Any]],
    errors: list[str],
) -> tuple[list[int], list[Path]]:
    """Read the exact four shards and rebuild their passed row indices."""
    runtime_dir = run_root / lane / "runtime_h20"
    passed: list[int] = []
    runtime_paths: list[Path] = []
    statuses: collections.Counter[str] = collections.Counter()
    observed = 0
    for shard, records_path, summary_path in _shard_paths(
        runtime_dir, lane, errors, require_summaries=lane == "shape"
    ):
        if not records_path.is_file():
            errors.append(f"{lane}: missing preselection shard:{shard}")
            continue
        runtime_paths.append(records_path)
        records = _load_jsonl(records_path)
        expected_indices = list(range(shard, len(rows), 4))
        if len(records) != len(expected_indices):
            errors.append(f"{lane}: preselection shard count mismatch:{shard}")
        if lane == "shape" and summary_path.is_file():
            summary = json.loads(summary_path.read_text())
            if (
                summary.get("rows") != len(expected_indices)
                or summary.get("passed") != len(expected_indices)
                or summary.get("failed") != 0
                or summary.get("shard_index") != shard
                or summary.get("shard_count") != 4
            ):
                errors.append(f"shape: preselection summary mismatch:{shard}")
            error = _exact_file_binding(summary.get("records"), records_path, f"shape:preselection:{shard}")
            if error:
                errors.append(error)
        for expected_index, record in zip(expected_indices, records, strict=False):
            index = record.get("position") if lane == "shape" else record.get("candidate_row_index")
            uuid = record.get("uuid") if lane == "shape" else record.get("child_uuid")
            if index != expected_index or uuid != _uuid(rows[expected_index]):
                errors.append(f"{lane}: preselection row identity mismatch:{shard}:{expected_index}")
                continue
            status = "passed" if record.get("passed") is True else record.get("status")
            statuses[str(status)] += 1
            if status == "passed":
                passed.append(expected_index)
            elif (
                lane != "dtype"
                or status != "unsupported"
                or not str(record.get("reason", "")).startswith("UnsupportedCase:")
            ):
                errors.append(f"{lane}: forbidden preselection status:{expected_index}:{status}")
        observed += len(records)
    if observed != len(rows):
        errors.append(f"{lane}: preselection coverage mismatch:{observed}/{len(rows)}")
    passed.sort()
    if lane == "dtype":
        errors.extend(_dtype_audit_errors(run_root / lane, runtime_paths, len(rows), statuses))
    return passed, runtime_paths


def _unsupported_reason_histogram(runtime_paths: Sequence[Path]) -> dict[str, int]:
    counts: collections.Counter[str] = collections.Counter()
    for path in runtime_paths:
        for record in _load_jsonl(path):
            if record.get("status") == "unsupported":
                detail = str(record.get("reason", "")).removeprefix("UnsupportedCase:")
                counts[detail.split(":", 1)[0]] += 1
    return dict(sorted(counts.items()))


def _selection_reconstruction(
    run_root: Path,
    lane: str,
    candidates: list[dict[str, Any]],
    candidate_manifests: list[dict[str, Any]],
    passed: list[int],
    runtime_paths: list[Path],
    errors: list[str],
    parents: list[dict[str, Any]] | None,
) -> None:
    lane_dir = run_root / lane
    selection_path = lane_dir / "selection_summary.json"
    selection = json.loads(selection_path.read_text())
    expected_stage = "shape_runtime_selection" if lane == "shape" else "runtime_selection"
    if (
        selection.get("contract") != EXPANSION_CONTRACT
        or selection.get("stage") != expected_stage
        or selection.get("review_only") is not True
        or selection.get("training_approved") is not False
    ):
        errors.append(f"{lane}: selection contract/governance mismatch")
    for key, path in (("candidates", lane_dir / "candidates.parquet"), ("manifest", lane_dir / "manifest.jsonl")):
        error = _exact_file_binding(selection.get("source_binding", {}).get(key), path, f"{lane}:selection:{key}")
        if error:
            errors.append(error)
    for key, path in (
        ("selected", lane_dir / "selected.parquet"),
        ("selected_manifest", lane_dir / "selected.manifest.jsonl"),
    ):
        error = _exact_file_binding(selection.get("artifacts", {}).get(key), path, f"{lane}:selection:{key}")
        if error:
            errors.append(error)
    expected_sources = [{"path": str(path.resolve()), "sha256": _sha256_file(path)} for path in runtime_paths]
    if selection.get("runtime_sources") != expected_sources:
        errors.append(f"{lane}: selection runtime source binding mismatch")
    records = [record for path in runtime_paths for record in _load_jsonl(path)]
    status_counts = collections.Counter(
        "passed" if record.get("passed") is True else str(record.get("status")) for record in records
    )
    if (
        selection.get("candidate_rows") != len(candidates)
        or selection.get("runtime_record_rows") != len(candidates)
        or selection.get("status_counts") != dict(sorted(status_counts.items()))
    ):
        errors.append(f"{lane}: selection count/status mismatch")
    selected_rows, selected_manifests = _rows_and_manifests(
        lane_dir / "selected.parquet", lane_dir / "selected.manifest.jsonl"
    )
    expected_rows = [candidates[index] for index in passed]
    if [_canonical(row) for row in selected_rows] != [_canonical(row) for row in expected_rows]:
        errors.append(f"{lane}: selected parquet is not the passed subsequence")
    by_uuid = {record.get("uuid", record.get("child_uuid")): record for record in records}
    expected_manifests = [
        {**candidate_manifests[index], "runtime_evidence": by_uuid.get(_uuid(candidates[index]))} for index in passed
    ]
    if [_canonical(row) for row in selected_manifests] != [_canonical(row) for row in expected_manifests]:
        errors.append(f"{lane}: selected manifest is not the passed subsequence")
    if selection.get("selected_rows") != len(selected_rows):
        errors.append(f"{lane}: selection selected count mismatch")
    if parents is not None:
        selected_parents = pq.read_table(lane_dir / "selected.parents.parquet").to_pylist()
        if [_canonical(row) for row in selected_parents] != [_canonical(parents[index]) for index in passed]:
            errors.append(f"{lane}: selected parents is not the passed subsequence")


def _check_postselection(
    run_root: Path, lane: str, selected: Path, manifest: Path, selection: Path
) -> tuple[list[str], dict[str, int]]:
    errors: list[str] = []
    rows, _ = _rows_and_manifests(selected, manifest)
    runtime_dir = run_root / lane / "runtime_h20_post_selection"
    observed = 0
    for shard, records_path, summary_path in _shard_paths(runtime_dir, lane, errors, require_summaries=True):
        if not records_path.is_file() or not summary_path.is_file():
            errors.append(f"{lane}: missing post-selection shard:{shard}")
            continue
        records = _load_jsonl(records_path)
        summary = json.loads(summary_path.read_text())
        positions = list(range(shard, len(rows), 4))
        if (
            summary.get("contract") != POST_CONTRACT
            or summary.get("validation_mode") != POST_MODES[lane]
            or summary.get("device") != f"cuda:{shard}"
            or "h20" not in str(summary.get("gpu_identity", {}).get("name", "")).lower()
            or summary.get("rows") != len(positions)
            or summary.get("passed") != len(positions)
            or summary.get("failed") != 0
            or summary.get("review_only") is not True
            or summary.get("training_approved") is not False
        ):
            errors.append(f"{lane}: post-selection summary mismatch:{shard}")
        error = _exact_file_binding(summary.get("records"), records_path, f"{lane}:post-selection:{shard}")
        if error:
            errors.append(error)
        for key, source in (("selected", selected), ("manifest", manifest), ("summary", selection)):
            errors.extend(
                _bound_source_errors(summary.get("source_binding", {}).get(key), source, f"{lane}:{shard}:{key}")
            )
        if len(records) != len(positions):
            errors.append(f"{lane}: post-selection record count mismatch:{shard}")
        for position, record in zip(positions, records, strict=False):
            if (
                record.get("position") != position
                or record.get("uuid") != _uuid(rows[position])
                or record.get("passed") is not True
                or record.get("errors") not in (None, [])
                or record.get("failed_checks") not in (None, [])
            ):
                errors.append(f"{lane}: post-selection row mismatch:{shard}:{position}")
        observed += len(records)
    if observed != len(rows):
        errors.append(f"{lane}: post-selection coverage mismatch:{observed}/{len(rows)}")
    return errors, {"rows": len(rows), "records": observed, "shards": 4}
