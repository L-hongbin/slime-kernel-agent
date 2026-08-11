#!/usr/bin/env python3
"""Compose one-row-per-parent serial augmentation bases with exact fallback lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.synthesize import intervention_semantic_gates as semantic_gates

CONTRACT_VERSION = "serial_augmentation_fallback_v1"
MIN_REPLACEMENT_RATE = 0.10


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"non-object JSONL row: {path}")
    return rows


def _uuid(row: Mapping[str, Any]) -> str:
    extra = row.get("extra_info")
    value = extra.get("uuid") if isinstance(extra, Mapping) else None
    if not isinstance(value, str) or not value:
        raise ValueError("row lacks extra_info.uuid")
    return value


def _reference_sha256(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(_reference_code(row).encode("utf-8")).hexdigest()


def _reference_code(row: Mapping[str, Any]) -> str:
    reward = row.get("reward_model")
    code = reward.get("ground_truth") if isinstance(reward, Mapping) else None
    if not isinstance(code, str):
        raise ValueError(f"row lacks reference code: {_uuid(row)}")
    return code


def _atomic_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text("".join(_canonical_bytes(row).decode("utf-8") + "\n" for row in rows), encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_parquet(path: Path, rows: Sequence[Mapping[str, Any]], schema: pa.Schema) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    pq.write_table(pa.Table.from_pylist(list(rows), schema=schema), temporary, compression="zstd")
    os.replace(temporary, path)


def _accepted_rows(
    parquet_path: Path,
    manifest_path: Path,
    *,
    stage: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    table = pq.read_table(parquet_path)
    manifests = _read_jsonl(manifest_path)
    rows = table.to_pylist()
    if len(rows) != len(manifests):
        raise ValueError(f"accepted parquet/manifest row mismatch: {parquet_path}")
    by_uuid: dict[str, dict[str, Any]] = {}
    manifest_by_uuid: dict[str, dict[str, Any]] = {}
    for index, (row, manifest) in enumerate(zip(rows, manifests, strict=True)):
        uuid = _uuid(row)
        if manifest.get("child_uuid") != uuid or uuid in by_uuid:
            raise ValueError(f"accepted UUID mismatch or duplicate at {parquet_path}:{index}")
        required_statuses = ["parent_runtime_status", "child_runtime_status", "liveness_status"]
        if stage == "layout":
            required_statuses.append("layout_realization_status")
        if (
            manifest.get("training_approved") is not False
            or any(manifest.get(field) != "passed" for field in required_statuses)
            or manifest.get("semantic_promotion_status") != "passed"
            or manifest.get("row_sha256") != _canonical_sha256(row)
        ):
            raise ValueError(f"accepted manifest lacks review-only governance: {uuid}")
        by_uuid[uuid] = row
        manifest_by_uuid[uuid] = manifest
    return by_uuid, manifest_by_uuid


def _output_schema(source: pa.Schema, *, stage: str, source_sha256: str) -> pa.Schema:
    metadata = dict(source.metadata or {})
    metadata.update(
        {
            b"serial.contract_version": CONTRACT_VERSION.encode(),
            b"serial.stage": stage.encode(),
            b"serial.source_sha256": source_sha256.encode(),
            b"serial.training_approved": b"false",
        }
    )
    return source.with_metadata(metadata)


def _write_result(
    *,
    output_dir: Path,
    stage: str,
    rows: Sequence[Mapping[str, Any]],
    lineage: Sequence[Mapping[str, Any]],
    schema: pa.Schema,
    inputs: Mapping[str, Any],
    selected_counts: Mapping[str, int],
    minimum_replacement_rows: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_path = output_dir / "selected.parquet"
    manifest_path = output_dir / "manifest.jsonl"
    summary_path = output_dir / "summary.json"
    for path in (selected_path, manifest_path, summary_path):
        if path.exists():
            raise FileExistsError(path)
    if len(rows) != len(lineage) or not rows:
        raise ValueError("serial output rows and lineage must be non-empty and aligned")
    replacement_layer = stage.removesuffix("_fallback_base")
    replacement_rows = selected_counts.get(replacement_layer)
    rate_minimum = math.ceil(len(rows) * MIN_REPLACEMENT_RATE)
    if minimum_replacement_rows is not None and not 0 < minimum_replacement_rows <= len(rows):
        raise ValueError("minimum replacement rows must be in [1, total rows]")
    required_replacements = max(rate_minimum, minimum_replacement_rows or 0)
    if type(replacement_rows) is not int or replacement_rows < required_replacements:
        raise ValueError(
            f"{stage} replacement gate failed: {replacement_rows}/{len(rows)} < "
            f"{required_replacements}/{len(rows)}"
        )
    normalized_schema = _output_schema(schema, stage=stage, source_sha256=str(inputs["primary_sha256"]))
    normalized_rows = pa.Table.from_pylist(list(rows), schema=normalized_schema).to_pylist()
    normalized_lineage: list[dict[str, Any]] = []
    for index, (row, manifest) in enumerate(zip(normalized_rows, lineage, strict=True)):
        item = dict(manifest)
        item["row_index"] = index
        item["selected_uuid"] = _uuid(row)
        item["selected_reference_sha256"] = _reference_sha256(row)
        item["selected_row_sha256"] = _canonical_sha256(row)
        normalized_lineage.append(item)
    uuids = [_uuid(row) for row in normalized_rows]
    if len(uuids) != len(set(uuids)):
        raise ValueError("serial output UUIDs are not unique")
    _atomic_parquet(selected_path, normalized_rows, normalized_schema)
    _atomic_jsonl(manifest_path, normalized_lineage)
    summary: dict[str, Any] = {
        "contract_version": CONTRACT_VERSION,
        "stage": stage,
        "rows": len(rows),
        "unique_output_uuids": len(set(uuids)),
        "inputs": dict(inputs),
        "selected_counts": dict(sorted(selected_counts.items())),
        "replacement_gate": {
            "replacement_layer": replacement_layer,
            "replacement_rows": replacement_rows,
            "total_rows": len(rows),
            "minimum_rate": MIN_REPLACEMENT_RATE,
            "rate_minimum_rows": rate_minimum,
            "requested_minimum_rows": minimum_replacement_rows,
            "minimum_rows": required_replacements,
            "passed": True,
        },
        "artifacts": {
            "selected_parquet": {
                "path": str(selected_path.resolve()),
                "sha256": _sha256_file(selected_path),
            },
            "manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": _sha256_file(manifest_path),
            },
        },
        "training_approved": False,
    }
    _atomic_json(summary_path, summary)
    return summary


def compose_random_base(
    *,
    shape_path: Path,
    random_candidates_path: Path,
    random_manifest_path: Path,
    random_accepted_path: Path,
    random_accepted_manifest_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    shape = pq.read_table(shape_path)
    candidates = pq.read_table(random_candidates_path)
    manifests = _read_jsonl(random_manifest_path)
    shape_rows = shape.to_pylist()
    candidate_rows = candidates.to_pylist()
    if not (len(shape_rows) == len(candidate_rows) == len(manifests)):
        raise ValueError("shape/random candidate/manifest row counts differ")
    accepted, accepted_manifests = _accepted_rows(random_accepted_path, random_accepted_manifest_path, stage="random")

    rows: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    counts = {"random": 0, "shape_fallback": 0}
    revalidated_accepted: set[str] = set()
    for index, (shape_row, candidate, manifest) in enumerate(zip(shape_rows, candidate_rows, manifests, strict=True)):
        shape_uuid = _uuid(shape_row)
        candidate_uuid = _uuid(candidate)
        if manifest.get("candidate_row_index") != index or manifest.get("source_row_index") != index:
            raise ValueError(f"random manifest index binding mismatch: {index}")
        if manifest.get("parent_uuid") != shape_uuid or manifest.get("child_uuid") != candidate_uuid:
            raise ValueError(f"random parent/child UUID binding mismatch: {index}")
        if manifest.get("parent_reference_sha256") != _reference_sha256(shape_row):
            raise ValueError(f"random parent reference binding mismatch: {index}")
        if manifest.get("child_reference_sha256") != _reference_sha256(candidate):
            raise ValueError(f"random child reference binding mismatch: {index}")
        chosen = accepted.get(candidate_uuid)
        semantic_verdict = None
        if chosen is not None:
            semantic_verdict = semantic_gates.evaluate_semantic_gate(
                "random",
                parent_code=_reference_code(shape_row),
                child_code=_reference_code(chosen),
                manifest=manifest,
            )
            if semantic_verdict["status"] != "passed":
                chosen = None
        layer = "random" if chosen is not None else "shape_fallback"
        if chosen is None:
            chosen = shape_row
        else:
            revalidated_accepted.add(candidate_uuid)
            accepted_manifest = accepted_manifests[candidate_uuid]
            if not all(
                accepted_manifest.get(field) == "passed"
                for field in (
                    "parent_runtime_status",
                    "child_runtime_status",
                    "liveness_status",
                )
            ):
                raise ValueError(f"random accepted child lacks passed runtime chain: {candidate_uuid}")
        counts[layer] += 1
        row = dict(chosen)
        rows.append(row)
        lineage.append(
            {
                "contract_version": CONTRACT_VERSION,
                "stage": "random_fallback_base",
                "row_index": index,
                "canonical_parent_uuid": manifest.get("canonical_parent_uuid"),
                "shape_uuid": shape_uuid,
                "random_candidate_uuid": candidate_uuid,
                "selected_layer": layer,
                "selected_uuid": _uuid(row),
                "selected_reference_sha256": _reference_sha256(row),
                "selected_row_sha256": _canonical_sha256(row),
                "upstream_random_manifest_row_sha256": _canonical_sha256(manifest),
                "random_runtime_policy_fingerprint": (
                    accepted_manifests[candidate_uuid].get("runtime_policy_fingerprint") if layer == "random" else None
                ),
                "semantic_gate": semantic_verdict if layer == "random" else None,
                "training_approved": False,
            }
        )
    if revalidated_accepted != {
        manifest["random_candidate_uuid"] for manifest in lineage if manifest["selected_layer"] == "random"
    }:
        raise ValueError("current-policy random UUID set was not consumed exactly")
    inputs = {
        "primary_path": str(shape_path.resolve()),
        "primary_sha256": _sha256_file(shape_path),
        "random_candidates_path": str(random_candidates_path.resolve()),
        "random_candidates_sha256": _sha256_file(random_candidates_path),
        "random_manifest_path": str(random_manifest_path.resolve()),
        "random_manifest_sha256": _sha256_file(random_manifest_path),
        "random_accepted_path": str(random_accepted_path.resolve()),
        "random_accepted_sha256": _sha256_file(random_accepted_path),
        "random_accepted_manifest_path": str(random_accepted_manifest_path.resolve()),
        "random_accepted_manifest_sha256": _sha256_file(random_accepted_manifest_path),
        "runtime_accepted_rows": len(accepted),
        "current_semantic_promoted_rows": len(revalidated_accepted),
        "semantic_gate_policy_version": semantic_gates.POLICY_VERSION,
        "semantic_gate_source_sha256": _sha256_file(Path(semantic_gates.__file__).resolve()),
    }
    return _write_result(
        output_dir=output_dir,
        stage="random_fallback_base",
        rows=rows,
        lineage=lineage,
        schema=candidates.schema,
        inputs=inputs,
        selected_counts=counts,
    )


def compose_stage_base(
    *,
    stage: str,
    prior_path: Path,
    prior_manifest_path: Path,
    lane_dirs: Sequence[Path],
    output_dir: Path,
    reuse_exact_parent_rows: bool = False,
    minimum_replacement_rows: int | None = None,
) -> dict[str, Any]:
    if stage not in {"dtype", "layout"}:
        raise ValueError(f"unsupported serial stage: {stage}")
    prior = pq.read_table(prior_path)
    prior_rows = prior.to_pylist()
    prior_lineage = _read_jsonl(prior_manifest_path)
    if len(prior_rows) != len(prior_lineage):
        raise ValueError("prior base parquet/manifest row counts differ")
    expected_prior_stage = "random_fallback_base" if stage == "dtype" else "dtype_fallback_base"
    for index, prior_manifest in enumerate(prior_lineage):
        if (
            prior_manifest.get("contract_version") != CONTRACT_VERSION
            or prior_manifest.get("stage") != expected_prior_stage
            or prior_manifest.get("row_index") != index
            or prior_manifest.get("selected_row_sha256") != _canonical_sha256(prior_rows[index])
        ):
            raise ValueError(f"{stage} prior lineage contract/stage/row mismatch: {index}")

    accepted_by_source: dict[
        int,
        tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
    ] = {}
    lane_inputs: list[dict[str, Any]] = []
    candidate_schema: pa.Schema | None = None
    for lane_dir in lane_dirs:
        candidate_path = lane_dir / "candidates.parquet"
        manifest_path = lane_dir / "manifest.jsonl"
        accepted_path = lane_dir / "runtime/accepted.parquet"
        accepted_manifest_path = lane_dir / "runtime/accepted.manifest.jsonl"
        candidates = pq.read_table(candidate_path)
        manifests = _read_jsonl(manifest_path)
        if candidates.num_rows != len(manifests):
            raise ValueError(f"candidate/manifest row mismatch: {lane_dir}")
        if candidate_schema is None:
            candidate_schema = candidates.schema
        elif not candidate_schema.equals(candidates.schema, check_metadata=False):
            raise ValueError("serial stage candidate schemas differ across lanes")
        accepted, accepted_manifests = _accepted_rows(accepted_path, accepted_manifest_path, stage=stage)
        manifest_by_uuid = {str(item.get("child_uuid")): item for item in manifests}
        candidate_by_uuid = {_uuid(row): row for row in candidates.to_pylist()}
        if len(candidate_by_uuid) != candidates.num_rows:
            raise ValueError(f"candidate UUIDs are not unique: {lane_dir}")
        if set(accepted) - set(manifest_by_uuid):
            raise ValueError(f"accepted UUID outside static lane: {lane_dir}")
        compatible_rows = 0
        incompatible_rows = 0
        semantic_rejected_rows = 0
        for uuid, row in accepted.items():
            manifest = manifest_by_uuid[uuid]
            runtime_manifest = accepted_manifests[uuid]
            if (
                uuid not in candidate_by_uuid
                or runtime_manifest.get("candidate_row_sha256") != _canonical_sha256(candidate_by_uuid[uuid])
                or runtime_manifest.get("source_row_index") != manifest.get("source_row_index")
                or runtime_manifest.get("parent_uuid") != manifest.get("parent_uuid")
                or runtime_manifest.get("parent_reference_sha256") != manifest.get("parent_reference_sha256")
                or runtime_manifest.get("child_reference_sha256") != manifest.get("child_reference_sha256")
            ):
                raise ValueError(f"accepted/static candidate binding mismatch: {uuid}")
            source_index = manifest.get("source_row_index")
            if type(source_index) is not int or not 0 <= source_index < len(prior_rows):
                raise ValueError(f"invalid source row index for {uuid}")
            parent_matches = manifest.get("parent_uuid") == _uuid(prior_rows[source_index]) and manifest.get(
                "parent_reference_sha256"
            ) == _reference_sha256(prior_rows[source_index])
            if not parent_matches:
                incompatible_rows += 1
                if reuse_exact_parent_rows:
                    continue
                raise ValueError(f"{stage} parent differs from prior base: {uuid}")
            semantic_verdict = semantic_gates.evaluate_semantic_gate(
                stage,
                parent_code=_reference_code(prior_rows[source_index]),
                child_code=_reference_code(row),
                manifest=manifest,
            )
            if semantic_verdict["status"] != "passed":
                semantic_rejected_rows += 1
                continue
            compatible_rows += 1
            if source_index in accepted_by_source:
                raise ValueError(f"multiple accepted {stage} children for source row {source_index}")
            accepted_by_source[source_index] = (row, manifest, runtime_manifest, semantic_verdict)
        lane_inputs.append(
            {
                "lane_dir": str(lane_dir.resolve()),
                "candidates_sha256": _sha256_file(candidate_path),
                "manifest_sha256": _sha256_file(manifest_path),
                "accepted_sha256": _sha256_file(accepted_path),
                "accepted_manifest_sha256": _sha256_file(accepted_manifest_path),
                "candidate_rows": candidates.num_rows,
                "accepted_rows": len(accepted),
                "compatible_accepted_rows": compatible_rows,
                "incompatible_accepted_rows_skipped": incompatible_rows,
                "current_semantic_rejected_rows": semantic_rejected_rows,
                "exact_parent_row_reuse": reuse_exact_parent_rows,
            }
        )

    rows: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    for index, (prior_row, prior_manifest) in enumerate(zip(prior_rows, prior_lineage, strict=True)):
        accepted = accepted_by_source.get(index)
        if accepted is None:
            row = dict(prior_row)
            selected_layer = "prior_fallback"
            stage_child_uuid = None
            runtime_policy_fingerprint = None
            semantic_verdict = None
        else:
            row, static_manifest, runtime_manifest, semantic_verdict = accepted
            row = dict(row)
            selected_layer = stage
            stage_child_uuid = static_manifest["child_uuid"]
            runtime_policy_fingerprint = runtime_manifest.get("runtime_policy_fingerprint")
        rows.append(row)
        lineage.append(
            {
                "contract_version": CONTRACT_VERSION,
                "stage": f"{stage}_fallback_base",
                "row_index": index,
                "prior_uuid": _uuid(prior_row),
                "prior_manifest_row_sha256": _canonical_sha256(prior_manifest),
                "selected_layer": selected_layer,
                "stage_child_uuid": stage_child_uuid,
                "selected_uuid": _uuid(row),
                "selected_reference_sha256": _reference_sha256(row),
                "selected_row_sha256": _canonical_sha256(row),
                "runtime_policy_fingerprint": runtime_policy_fingerprint,
                "semantic_gate": semantic_verdict,
                "training_approved": False,
            }
        )
    inputs = {
        "primary_path": str(prior_path.resolve()),
        "primary_sha256": _sha256_file(prior_path),
        "prior_manifest_path": str(prior_manifest_path.resolve()),
        "prior_manifest_sha256": _sha256_file(prior_manifest_path),
        "lanes": lane_inputs,
        "semantic_gate_policy_version": semantic_gates.POLICY_VERSION,
        "semantic_gate_source_sha256": _sha256_file(Path(semantic_gates.__file__).resolve()),
    }
    return _write_result(
        output_dir=output_dir,
        stage=f"{stage}_fallback_base",
        rows=rows,
        lineage=lineage,
        schema=candidate_schema or prior.schema,
        inputs=inputs,
        selected_counts={stage: len(accepted_by_source), "prior_fallback": len(rows) - len(accepted_by_source)},
        minimum_replacement_rows=minimum_replacement_rows,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    random_parser = subparsers.add_parser("random-base")
    random_parser.add_argument("--shape", type=Path, required=True)
    random_parser.add_argument("--random-candidates", type=Path, required=True)
    random_parser.add_argument("--random-manifest", type=Path, required=True)
    random_parser.add_argument("--random-accepted", type=Path, required=True)
    random_parser.add_argument("--random-accepted-manifest", type=Path, required=True)
    random_parser.add_argument("--output-dir", type=Path, required=True)
    stage_parser = subparsers.add_parser("stage-base")
    stage_parser.add_argument("--stage", choices=("dtype", "layout"), required=True)
    stage_parser.add_argument("--prior", type=Path, required=True)
    stage_parser.add_argument("--prior-manifest", type=Path, required=True)
    stage_parser.add_argument("--lane-dir", type=Path, action="append", required=True)
    stage_parser.add_argument("--output-dir", type=Path, required=True)
    stage_parser.add_argument(
        "--reuse-exact-parent-rows",
        action="store_true",
        help="Reuse accepted evidence from an older base only when UUID and reference are exact.",
    )
    stage_parser.add_argument(
        "--minimum-replacement-rows",
        type=int,
        help="Fail unless the composed base contains at least this many rows from the requested stage.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "random-base":
        summary = compose_random_base(
            shape_path=args.shape,
            random_candidates_path=args.random_candidates,
            random_manifest_path=args.random_manifest,
            random_accepted_path=args.random_accepted,
            random_accepted_manifest_path=args.random_accepted_manifest,
            output_dir=args.output_dir,
        )
    else:
        summary = compose_stage_base(
            stage=args.stage,
            prior_path=args.prior,
            prior_manifest_path=args.prior_manifest,
            lane_dirs=args.lane_dir,
            output_dir=args.output_dir,
            reuse_exact_parent_rows=args.reuse_exact_parent_rows,
            minimum_replacement_rows=args.minimum_replacement_rows,
        )
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
