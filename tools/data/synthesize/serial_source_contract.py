"""Fail-closed binding for one-row-per-parent serial augmentation bases."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

CONTRACT_VERSION = "serial_augmentation_source_binding_v1"
COMPOSER_CONTRACT = "serial_augmentation_fallback_v1"


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"serial manifest contains a non-object row: {path}")
    return rows


def _nested(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def verify_serial_source(
    input_path: Path,
    manifest_path: Path,
    *,
    expected_stage: str | None = None,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    if input_path.resolve().parent != manifest_path.resolve().parent:
        raise ValueError("serial selected parquet and manifest must share one artifact directory")
    summary_path = input_path.resolve().parent / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping) or summary.get("contract_version") != COMPOSER_CONTRACT:
        raise ValueError("serial source summary contract mismatch")
    if summary.get("training_approved") is not False:
        raise ValueError("serial source summary lacks review-only governance")
    if expected_stage is not None and summary.get("stage") != expected_stage:
        raise ValueError(f"serial source stage mismatch: {summary.get('stage')} != {expected_stage}")
    input_sha256 = sha256_file(input_path)
    manifest_sha256 = sha256_file(manifest_path)
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("serial source summary lacks artifacts")
    selected_artifact = artifacts.get("selected_parquet")
    manifest_artifact = artifacts.get("manifest")
    if not isinstance(selected_artifact, Mapping) or not isinstance(manifest_artifact, Mapping):
        raise ValueError("serial source artifact bindings are malformed")
    if (
        selected_artifact.get("path") != str(input_path.resolve())
        or selected_artifact.get("sha256") != input_sha256
        or manifest_artifact.get("path") != str(manifest_path.resolve())
        or manifest_artifact.get("sha256") != manifest_sha256
    ):
        raise ValueError("serial source artifact path/SHA binding mismatch")
    parquet = pq.ParquetFile(input_path)
    manifests = _read_jsonl(manifest_path)
    if not manifests or parquet.metadata.num_rows != len(manifests) or summary.get("rows") != len(manifests):
        raise ValueError("serial source row counts differ")
    replacement_gate = summary.get("replacement_gate")
    if not isinstance(replacement_gate, Mapping) or replacement_gate.get("passed") is not True:
        raise ValueError("serial source summary lacks a passing replacement gate")
    metadata = parquet.schema_arrow.metadata or {}
    if metadata.get(b"serial.contract_version") != COMPOSER_CONTRACT.encode():
        raise ValueError("serial source parquet metadata contract mismatch")
    if metadata.get(b"serial.training_approved") != b"false":
        raise ValueError("serial source parquet metadata lacks review-only governance")
    output_uuids: set[str] = set()
    row_index = 0
    for batch in parquet.iter_batches(batch_size=256, use_threads=False):
        for row in batch.to_pylist():
            manifest = manifests[row_index]
            uuid = _nested(row, "extra_info.uuid")
            reference = _nested(row, "reward_model.ground_truth")
            if (
                manifest.get("contract_version") != COMPOSER_CONTRACT
                or manifest.get("row_index") != row_index
                or manifest.get("selected_uuid") != uuid
                or not isinstance(reference, str)
                or manifest.get("selected_reference_sha256") != hashlib.sha256(reference.encode("utf-8")).hexdigest()
                or manifest.get("selected_row_sha256") != canonical_sha256(row)
                or manifest.get("training_approved") is not False
            ):
                raise ValueError(f"serial source row binding mismatch: {row_index}")
            if not isinstance(uuid, str) or not uuid or uuid in output_uuids:
                raise ValueError(f"serial source UUID is missing or duplicated: {row_index}")
            output_uuids.add(uuid)
            row_index += 1
    if row_index != len(manifests):
        raise ValueError("serial source parquet iteration count differs")
    binding = {
        "contract_version": CONTRACT_VERSION,
        "composer_contract_version": COMPOSER_CONTRACT,
        "stage": summary.get("stage"),
        "source_artifact_path": str(input_path.resolve()),
        "source_artifact_sha256": input_sha256,
        "source_manifest_path": str(manifest_path.resolve()),
        "source_manifest_sha256": manifest_sha256,
        "source_summary_path": str(summary_path.resolve()),
        "source_summary_sha256": sha256_file(summary_path),
        "source_rows": len(manifests),
    }
    return binding, tuple(manifests)


def resolve_lane_serial_source(
    manifests: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[int, dict[str, Any]], dict[int, dict[str, Any]]] | None:
    """Reload and exactly bind the serial source rows referenced by one lane."""

    if not manifests:
        raise ValueError("serial lane manifest is empty")
    bindings = [item.get("source_binding") for item in manifests]
    if all(binding is None for binding in bindings):
        return None
    if any(not isinstance(binding, Mapping) for binding in bindings):
        raise ValueError("serial lane mixes bound and unbound source rows")
    first = dict(bindings[0])
    if any(canonical_sha256(dict(binding)) != canonical_sha256(first) for binding in bindings[1:]):
        raise ValueError("serial lane source bindings differ")
    input_path = Path(str(first.get("source_artifact_path", "")))
    manifest_path = Path(str(first.get("source_manifest_path", "")))
    if not input_path.is_absolute() or not manifest_path.is_absolute():
        raise ValueError("serial lane source paths must be absolute")
    verified, upstream = verify_serial_source(input_path, manifest_path)
    if canonical_sha256(verified) != canonical_sha256(first):
        raise ValueError("serial lane source binding differs from current artifacts")
    requested: set[int] = set()
    upstream_by_index: dict[int, dict[str, Any]] = {}
    for lane_index, item in enumerate(manifests):
        source_index = item.get("source_row_index")
        if type(source_index) is not int or not 0 <= source_index < len(upstream):
            raise ValueError(f"serial lane source row index is invalid: {lane_index}")
        if (
            item.get("source_artifact_path") != verified["source_artifact_path"]
            or item.get("source_artifact_sha256") != verified["source_artifact_sha256"]
            or item.get("upstream_serial_manifest_row_sha256") != canonical_sha256(upstream[source_index])
        ):
            raise ValueError(f"serial lane upstream row binding mismatch: {lane_index}")
        requested.add(source_index)
        upstream_by_index[source_index] = upstream[source_index]
    rows: dict[int, dict[str, Any]] = {}
    offset = 0
    parquet = pq.ParquetFile(input_path)
    for batch in parquet.iter_batches(batch_size=256, use_threads=False):
        for local_index, row in enumerate(batch.to_pylist()):
            source_index = offset + local_index
            if source_index in requested:
                rows[source_index] = row
        offset += batch.num_rows
        if len(rows) == len(requested):
            break
    if set(rows) != requested:
        raise ValueError("serial lane failed to reload all referenced source rows")
    return verified, rows, upstream_by_index
