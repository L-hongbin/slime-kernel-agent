#!/usr/bin/env python3
"""Materialize only synthesized references backed by complete GPU evidence.

The generator manifest binds each candidate row to its UUID, source hash, and
normalized AST.  One or more JSONL audit shards then bind the candidate
parquet's byte hash to KernelGym's five-trial, persistent train-mode contract.
This program verifies those bindings before it writes anything.

Runtime failures (including timeouts and OOMs) are normal rejected rows.  A
missing row, duplicate audit, identity disagreement, or a claimed pass without
all contract markers is an integrity error and aborts the whole operation.

Decontamination here is deliberately exact: reference SHA-256 and normalized
AST SHA-256 are compared with the supplied baseline parquets.  Near-duplicate
or embedding-based checks belong to the upstream static provenance pipeline
and are not approximated by this acceptance step.
"""

from __future__ import annotations

import argparse
import ast
import collections
import dataclasses
import hashlib
import json
import math
import os
import re
import sys
import uuid as uuidlib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

MATERIALIZER_VERSION = "validated-candidate-materializer-v2"
EXPECTED_TRIALS = 5
EXPECTED_SEED = 42
EXPECTED_DEVICE = "cuda:0"
EXPECTED_MAX_DEVICE_MEMORY_GIB = 64.0
REQUIRED_CONTRACT_VERSION = "kernelgym-reference-self-train-mode-v3"
SHA256_RE = re.compile(r"[0-9a-f]{64}")
GIB_BYTES = 1024**3
REQUIRED_MEMORY_LIMIT_BYTES = 64 * GIB_BYTES
KERNELGYM_EVALUATOR_FILES = {
    "correctness": "kernelgym/toolkit/kernelbench/correctness.py",
    "loading": "kernelgym/toolkit/kernelbench/loading.py",
    "exec_types": "kernelgym/toolkit/kernelbench/exec_types.py",
    "profiling": "kernelgym/toolkit/kernelbench/profiling.py",
    "config_init": "kernelgym/config/__init__.py",
    "config_settings": "kernelgym/config/settings.py",
}
KERNELGYM_HASH_FIELDS = tuple(f"{name}_sha256" for name in KERNELGYM_EVALUATOR_FILES) + ("evaluator_bundle_sha256",)
EXPECTED_CONTRACT_PAYLOAD_KEYS = {
    "contract_version",
    "source_sha256",
    "validator_source_sha256",
    "launcher_source_sha256",
    *(f"kernelgym_{field}" for field in KERNELGYM_HASH_FIELDS),
    "code_column",
    "entry_point_column",
    "mode_class_column",
    "expected_mode_class",
    "device",
    "max_device_memory_gib",
    "trials",
    "seed",
    "training",
    "paired_rng_seed_reset_required",
    "persistent_model_instances_required",
}


class IntegrityError(ValueError):
    """The supplied provenance or runtime evidence is incomplete/contradictory."""


@dataclasses.dataclass(frozen=True)
class RowIdentity:
    row_index: int
    uuid: str
    reference_sha256: str
    normalized_ast_sha256: str


@dataclasses.dataclass(frozen=True)
class AuditEvidence:
    record: Mapping[str, Any]
    source_path: Path
    source_sha256: str
    line_number: int


@dataclasses.dataclass(frozen=True)
class MaterializeConfig:
    input_parquet: Path
    generator_manifest: Path
    audit_paths: tuple[Path, ...]
    output_parquet: Path
    accepted_manifest: Path
    rejected_manifest: Path
    summary_json: Path
    dedup_against: tuple[Path, ...] = ()
    code_column: str = "reward_model.ground_truth"
    uuid_column: str = "extra_info.uuid"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return _sha256_bytes(payload.encode("utf-8"))


def _normalized_ast_sha256(code: str, *, context: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise IntegrityError(f"{context}: reference is not valid Python: {exc}") from exc
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return _sha256_bytes(normalized.encode("utf-8"))


def _nested(value: Any, dotted_path: str) -> Any:
    current = value
    for part in dotted_path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _require_sha256(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise IntegrityError(f"{context}: expected a lowercase SHA-256, got {value!r}")
    return value


def _require_nonempty_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IntegrityError(f"{context}: expected non-empty text, got {value!r}")
    return value


def _require_integer(value: Any, *, context: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise IntegrityError(f"{context}: expected an integer, got {value!r}")
    if minimum is not None and value < minimum:
        raise IntegrityError(f"{context}: expected >= {minimum}, got {value}")
    return value


def _resolve_file(path: Path, *, label: str, suffix: str | None = None) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    if suffix is not None and resolved.suffix != suffix:
        raise ValueError(f"{label} must end in {suffix}: {resolved}")
    return resolved


def _expand_paths(paths: Sequence[Path], *, suffix: str, label: str) -> tuple[Path, ...]:
    """Expand repeatable files/directories into a deterministic unique file list."""

    expanded: list[Path] = []
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if path.is_file():
            if path.suffix != suffix:
                raise ValueError(f"{label} file must end in {suffix}: {path}")
            expanded.append(path)
            continue
        if path.is_dir():
            matches = sorted(candidate.resolve() for candidate in path.rglob(f"*{suffix}") if candidate.is_file())
            if not matches:
                raise FileNotFoundError(f"{label} directory has no *{suffix} files: {path}")
            expanded.extend(matches)
            continue
        raise FileNotFoundError(f"{label} path does not exist: {path}")

    unique = sorted(set(expanded), key=lambda item: str(item))
    if not unique:
        raise ValueError(f"at least one {label} file is required")
    return tuple(unique)


def _read_jsonl(path: Path, *, label: str) -> list[tuple[int, dict[str, Any]]]:
    records: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise IntegrityError(f"{path}:{line_number}: invalid {label} JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise IntegrityError(f"{path}:{line_number}: {label} record must be an object")
            records.append((line_number, record))
    if not records:
        raise IntegrityError(f"{path}: {label} JSONL is empty")
    return records


def _load_generator_manifest(path: Path, *, expected_rows: int) -> list[dict[str, Any]]:
    records = [record for _, record in _read_jsonl(path, label="generator manifest")]
    if len(records) != expected_rows:
        raise IntegrityError(
            f"generator manifest has {len(records)} records but candidate parquet has {expected_rows} rows"
        )

    seen_uuids: set[str] = set()
    seen_references: set[str] = set()
    seen_asts: set[str] = set()
    for expected_index, record in enumerate(records):
        row_index = _require_integer(
            record.get("row_index"), context=f"generator manifest row {expected_index}.row_index", minimum=0
        )
        if row_index != expected_index:
            raise IntegrityError(
                f"generator manifest record {expected_index} declares row_index={row_index}; "
                "records must cover the parquet exactly once in row order"
            )
        row_uuid = _require_nonempty_string(
            record.get("uuid"), context=f"generator manifest row {expected_index}.uuid"
        )
        reference_hash = _require_sha256(
            record.get("reference_sha256"),
            context=f"generator manifest row {expected_index}.reference_sha256",
        )
        ast_hash = _require_sha256(
            record.get("normalized_ast_sha256"),
            context=f"generator manifest row {expected_index}.normalized_ast_sha256",
        )
        if row_uuid in seen_uuids:
            raise IntegrityError(f"generator manifest has duplicate UUID {row_uuid!r}")
        if reference_hash in seen_references:
            raise IntegrityError(f"generator manifest has duplicate reference SHA-256 {reference_hash}")
        if ast_hash in seen_asts:
            raise IntegrityError(f"generator manifest has duplicate normalized AST SHA-256 {ast_hash}")
        seen_uuids.add(row_uuid)
        seen_references.add(reference_hash)
        seen_asts.add(ast_hash)
    return records


def _parquet_identities(
    path: Path,
    manifest: Sequence[Mapping[str, Any]],
    *,
    code_column: str,
    uuid_column: str,
) -> list[RowIdentity]:
    parquet = pq.ParquetFile(path)
    top_level = set(parquet.schema_arrow.names)
    selected_columns = sorted({code_column.split(".", 1)[0], uuid_column.split(".", 1)[0]})
    missing = set(selected_columns) - top_level
    if missing:
        raise IntegrityError(f"candidate parquet is missing columns: {sorted(missing)}")

    identities: list[RowIdentity] = []
    row_index = 0
    for batch in parquet.iter_batches(batch_size=512, columns=selected_columns):
        for row in batch.to_pylist():
            code = _nested(row, code_column)
            if not isinstance(code, str) or not code.strip():
                raise IntegrityError(f"candidate row {row_index}: {code_column} is not non-empty text")
            row_uuid = _require_nonempty_string(
                _nested(row, uuid_column), context=f"candidate row {row_index}.{uuid_column}"
            )
            reference_hash = _sha256_bytes(code.encode("utf-8"))
            ast_hash = _normalized_ast_sha256(code, context=f"candidate row {row_index}")
            manifest_record = manifest[row_index]
            expected = (
                manifest_record["uuid"],
                manifest_record["reference_sha256"],
                manifest_record["normalized_ast_sha256"],
            )
            observed = (row_uuid, reference_hash, ast_hash)
            if observed != expected:
                labels = ("uuid", "reference_sha256", "normalized_ast_sha256")
                differences = [
                    f"{label}: parquet={left!r}, manifest={right!r}"
                    for label, left, right in zip(labels, observed, expected, strict=True)
                    if left != right
                ]
                raise IntegrityError(
                    f"candidate row {row_index} disagrees with generator manifest ({'; '.join(differences)})"
                )
            identities.append(RowIdentity(row_index, row_uuid, reference_hash, ast_hash))
            row_index += 1
    if row_index != len(manifest):
        raise IntegrityError(f"candidate parquet yielded {row_index} rows but generator manifest has {len(manifest)}")
    return identities


def _load_audits(
    audit_files: Sequence[Path],
    identities: Sequence[RowIdentity],
    *,
    candidate_sha256: str,
) -> tuple[dict[int, AuditEvidence], list[dict[str, Any]], str, str, dict[str, Any]]:
    expected_by_index = {identity.row_index: identity for identity in identities}
    evidence_by_index: dict[int, AuditEvidence] = {}
    file_summaries: list[dict[str, Any]] = []
    contract_versions: set[str] = set()
    contract_fingerprints: set[str] = set()
    toolchain_contracts: set[tuple[str, str]] = set()
    kernelgym_contracts: set[tuple[str, ...]] = set()
    passing_environments: set[tuple[str, str, tuple[int, int], str, str]] = set()
    source_independent_contracts: set[str] = set()
    expected_validator_hash = _sha256_file(Path(__file__).with_name("validate_train_mode_contract.py"))
    expected_launcher_hash = _sha256_file(Path(__file__).with_name("launch_reference_validation_shards.sh"))

    for path in audit_files:
        file_hash = _sha256_file(path)
        records = _read_jsonl(path, label="runtime audit")
        file_summaries.append({"path": str(path), "sha256": file_hash, "records": len(records)})
        for line_number, record in records:
            context = f"{path}:{line_number}"
            source_hash = _require_sha256(record.get("source_sha256"), context=f"{context}.source_sha256")
            if source_hash != candidate_sha256:
                raise IntegrityError(
                    f"{context}: audit source SHA-256 {source_hash} does not match candidate {candidate_sha256}"
                )
            row_index = _require_integer(record.get("row_index"), context=f"{context}.row_index", minimum=0)
            identity = expected_by_index.get(row_index)
            if identity is None:
                raise IntegrityError(f"{context}: audit refers to out-of-range row {row_index}")
            if row_index in evidence_by_index:
                previous = evidence_by_index[row_index]
                raise IntegrityError(
                    f"row {row_index} has more than one audit: "
                    f"{previous.source_path}:{previous.line_number} and {context}"
                )

            audit_uuid = _require_nonempty_string(record.get("uuid"), context=f"{context}.uuid")
            reference_hash = _require_sha256(record.get("reference_sha256"), context=f"{context}.reference_sha256")
            row_key = _require_nonempty_string(record.get("row_key"), context=f"{context}.row_key")
            expected_key = f"{row_index}:{identity.reference_sha256}"
            if audit_uuid != identity.uuid or reference_hash != identity.reference_sha256 or row_key != expected_key:
                raise IntegrityError(
                    f"{context}: audit identity disagrees with candidate/manifest for row {row_index}"
                )
            audit_ast = record.get("normalized_ast_sha256")
            if audit_ast is not None:
                audit_ast = _require_sha256(audit_ast, context=f"{context}.normalized_ast_sha256")
                if audit_ast != identity.normalized_ast_sha256:
                    raise IntegrityError(f"{context}: normalized AST hash disagrees for row {row_index}")

            trials = _require_integer(record.get("trials"), context=f"{context}.trials", minimum=1)
            if trials != EXPECTED_TRIALS:
                raise IntegrityError(f"{context}: expected trials={EXPECTED_TRIALS}, got {trials}")
            if record.get("training") is not True:
                raise IntegrityError(f"{context}: audit was not configured for training=True")
            seed = _require_integer(record.get("seed"), context=f"{context}.seed")
            if seed != EXPECTED_SEED:
                raise IntegrityError(f"{context}: expected seed={EXPECTED_SEED}, got {seed}")
            device = _require_nonempty_string(record.get("device"), context=f"{context}.device")
            if device != EXPECTED_DEVICE:
                raise IntegrityError(f"{context}: expected device={EXPECTED_DEVICE!r}, got {device!r}")
            max_device_memory_gib = record.get("max_device_memory_gib")
            if (
                isinstance(max_device_memory_gib, bool)
                or not isinstance(max_device_memory_gib, (int, float))
                or not math.isfinite(float(max_device_memory_gib))
                or float(max_device_memory_gib) != EXPECTED_MAX_DEVICE_MEMORY_GIB
            ):
                raise IntegrityError(
                    f"{context}: expected exact max_device_memory_gib={EXPECTED_MAX_DEVICE_MEMORY_GIB:g}, "
                    f"got {max_device_memory_gib!r}"
                )
            if not isinstance(record.get("passed"), bool):
                raise IntegrityError(f"{context}: passed must be a Boolean")
            status = _require_nonempty_string(record.get("status"), context=f"{context}.status")
            if record["passed"] is True and status != "passed":
                raise IntegrityError(f"{context}: passed=True contradicts status={status!r}")
            if record["passed"] is False and status == "passed":
                raise IntegrityError(f"{context}: passed=False contradicts status='passed'")

            version = _require_nonempty_string(record.get("contract_version"), context=f"{context}.contract_version")
            fingerprint = _require_sha256(
                record.get("contract_fingerprint"), context=f"{context}.contract_fingerprint"
            )
            contract_payload = record.get("contract_payload")
            if not isinstance(contract_payload, Mapping):
                raise IntegrityError(f"{context}: missing source-bound contract_payload")
            payload_keys = set(contract_payload)
            if payload_keys != EXPECTED_CONTRACT_PAYLOAD_KEYS:
                missing_keys = sorted(EXPECTED_CONTRACT_PAYLOAD_KEYS - payload_keys)
                extra_keys = sorted(payload_keys - EXPECTED_CONTRACT_PAYLOAD_KEYS)
                raise IntegrityError(
                    f"{context}: contract_payload schema mismatch; missing={missing_keys}, extra={extra_keys}"
                )
            if _canonical_json_sha256(contract_payload) != fingerprint:
                raise IntegrityError(f"{context}: contract_payload does not reproduce contract_fingerprint")
            source_independent_contract = dict(contract_payload)
            source_independent_contract.pop("source_sha256", None)
            source_independent_contracts.add(
                json.dumps(
                    source_independent_contract,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
            )
            contract_versions.add(version)
            contract_fingerprints.add(fingerprint)

            validator_hash = _require_sha256(
                record.get("validator_source_sha256"), context=f"{context}.validator_source_sha256"
            )
            launcher_hash = _require_sha256(
                record.get("launcher_source_sha256"), context=f"{context}.launcher_source_sha256"
            )
            if validator_hash != expected_validator_hash:
                raise IntegrityError(
                    f"{context}: validator source hash {validator_hash} does not match current tool {expected_validator_hash}"
                )
            if launcher_hash != expected_launcher_hash:
                raise IntegrityError(
                    f"{context}: launcher source hash {launcher_hash} does not match current tool {expected_launcher_hash}"
                )
            toolchain_contracts.add((validator_hash, launcher_hash))

            kernelgym = record.get("kernelgym")
            if not isinstance(kernelgym, Mapping):
                raise IntegrityError(f"{context}: missing KernelGym provenance")
            kernelgym_hashes = tuple(
                _require_sha256(kernelgym.get(field), context=f"{context}.kernelgym.{field}")
                for field in KERNELGYM_HASH_FIELDS
            )
            kernelgym_by_field = dict(zip(KERNELGYM_HASH_FIELDS, kernelgym_hashes, strict=True))
            bundle_payload = {
                KERNELGYM_EVALUATOR_FILES[name]: kernelgym_by_field[f"{name}_sha256"]
                for name in sorted(KERNELGYM_EVALUATOR_FILES)
            }
            if _canonical_json_sha256(bundle_payload) != kernelgym_by_field["evaluator_bundle_sha256"]:
                raise IntegrityError(f"{context}: KernelGym evaluator bundle hash is inconsistent")
            kernelgym_contracts.add(kernelgym_hashes)

            payload_expectations: dict[str, Any] = {
                "contract_version": version,
                "source_sha256": source_hash,
                "validator_source_sha256": validator_hash,
                "launcher_source_sha256": launcher_hash,
                "trials": trials,
                "seed": seed,
                "training": True,
                "device": device,
                "max_device_memory_gib": float(max_device_memory_gib),
                "paired_rng_seed_reset_required": True,
                "persistent_model_instances_required": True,
                "code_column": "reward_model.ground_truth",
                "entry_point_column": "extra_info.entry_point",
                "mode_class_column": "extra_info.v4.mode_class",
                "expected_mode_class": None,
                **{f"kernelgym_{field}": value for field, value in kernelgym_by_field.items()},
            }
            payload_disagreements = [
                f"{field}={contract_payload.get(field)!r} (expected {expected!r})"
                for field, expected in payload_expectations.items()
                if contract_payload.get(field) != expected
            ]
            if payload_disagreements:
                raise IntegrityError(
                    f"{context}: contract_payload disagrees with audit fields: " + ", ".join(payload_disagreements)
                )
            if record["passed"] is True:
                gpu = record.get("gpu")
                if not isinstance(gpu, Mapping):
                    raise IntegrityError(f"{context}: passing audit lacks GPU/runtime environment")
                gpu_device = _require_nonempty_string(gpu.get("device"), context=f"{context}.gpu.device")
                if gpu_device != device:
                    raise IntegrityError(
                        f"{context}: gpu.device={gpu_device!r} disagrees with contract device={device!r}"
                    )
                gpu_name = _require_nonempty_string(gpu.get("name"), context=f"{context}.gpu.name")
                compute_capability = gpu.get("compute_capability")
                if (
                    not isinstance(compute_capability, list)
                    or len(compute_capability) != 2
                    or any(isinstance(item, bool) or not isinstance(item, int) for item in compute_capability)
                ):
                    raise IntegrityError(f"{context}.gpu.compute_capability: expected two integers")
                torch_version = _require_nonempty_string(
                    gpu.get("torch_version"), context=f"{context}.gpu.torch_version"
                )
                torch_cuda_version = _require_nonempty_string(
                    gpu.get("torch_cuda_version"), context=f"{context}.gpu.torch_cuda_version"
                )
                passing_environments.add(
                    (
                        gpu_device,
                        gpu_name,
                        (compute_capability[0], compute_capability[1]),
                        torch_version,
                        torch_cuda_version,
                    )
                )
            evidence_by_index[row_index] = AuditEvidence(record, path, file_hash, line_number)

    missing = sorted(set(expected_by_index) - set(evidence_by_index))
    if missing:
        preview = ", ".join(map(str, missing[:20]))
        suffix = "..." if len(missing) > 20 else ""
        raise IntegrityError(f"missing runtime audits for {len(missing)} candidate rows: {preview}{suffix}")
    if len(contract_versions) != 1:
        raise IntegrityError(f"audit shards mix contract versions: {sorted(contract_versions)}")
    contract_version = next(iter(contract_versions))
    if contract_version != REQUIRED_CONTRACT_VERSION:
        raise IntegrityError(
            f"runtime audits require contract {REQUIRED_CONTRACT_VERSION!r}, got {contract_version!r}"
        )
    if len(contract_fingerprints) != 1:
        raise IntegrityError("audit shards mix contract fingerprints")
    if len(toolchain_contracts) != 1:
        raise IntegrityError("audit shards mix validator/launcher implementations")
    if len(kernelgym_contracts) != 1:
        raise IntegrityError("audit shards mix KernelGym evaluator implementations")
    if len(source_independent_contracts) != 1:
        raise IntegrityError("audit shards mix source-independent runtime policies")
    if len(passing_environments) > 1:
        raise IntegrityError("passing runtime audits mix GPU/PyTorch/CUDA environments")
    validator_hash, launcher_hash = next(iter(toolchain_contracts))
    kernelgym_hashes = dict(zip(KERNELGYM_HASH_FIELDS, next(iter(kernelgym_contracts)), strict=True))
    runtime_environment: dict[str, Any] | None = None
    if passing_environments:
        gpu_device, gpu_name, compute_capability, torch_version, torch_cuda_version = next(iter(passing_environments))
        runtime_environment = {
            "device": gpu_device,
            "gpu_name": gpu_name,
            "compute_capability": list(compute_capability),
            "torch_version": torch_version,
            "torch_cuda_version": torch_cuda_version,
        }
    source_independent_contract = json.loads(next(iter(source_independent_contracts)))
    return (
        evidence_by_index,
        file_summaries,
        contract_version,
        next(iter(contract_fingerprints)),
        {
            "validator_source_sha256": validator_hash,
            "launcher_source_sha256": launcher_hash,
            **{f"kernelgym_{field}": value for field, value in kernelgym_hashes.items()},
            "runtime_environment": runtime_environment,
            "source_independent_contract": source_independent_contract,
            "runtime_policy_fingerprint": _canonical_json_sha256(source_independent_contract),
        },
    )


def _validate_memory_guard(
    guard: Any,
    *,
    context: str,
    required_limit_bytes: int,
) -> dict[str, Any]:
    if not isinstance(guard, Mapping):
        raise IntegrityError(f"{context}: missing memory_guard object")
    if guard.get("enabled") is not True:
        raise IntegrityError(f"{context}: memory_guard.enabled is not true")
    if guard.get("configured") is not True:
        raise IntegrityError(f"{context}: memory_guard.configured is not true")
    if guard.get("synchronized") is not True:
        raise IntegrityError(f"{context}: memory_guard.synchronized is not true")
    if guard.get("over_limit") is not False:
        raise IntegrityError(f"{context}: memory_guard.over_limit is not false")
    limit = _require_integer(guard.get("limit_bytes"), context=f"{context}.limit_bytes", minimum=1)
    if limit != required_limit_bytes:
        raise IntegrityError(f"{context}.limit_bytes: expected exact policy limit {required_limit_bytes}, got {limit}")
    peak_allocated = _require_integer(
        guard.get("peak_allocated_bytes"), context=f"{context}.peak_allocated_bytes", minimum=0
    )
    peak_reserved = _require_integer(
        guard.get("peak_reserved_bytes"), context=f"{context}.peak_reserved_bytes", minimum=0
    )
    fraction = guard.get("allocator_fraction")
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)):
        raise IntegrityError(f"{context}.allocator_fraction: expected a finite number")
    fraction = float(fraction)
    if not math.isfinite(fraction) or not 0.0 < fraction <= 1.0:
        raise IntegrityError(f"{context}.allocator_fraction: expected 0 < value <= 1, got {fraction}")
    if guard.get("within_limit") is not True:
        raise IntegrityError(f"{context}: memory_guard.within_limit is not true for a claimed pass")
    if peak_allocated > limit or peak_reserved > limit:
        raise IntegrityError(
            f"{context}: claimed within_limit but peaks ({peak_allocated}, {peak_reserved}) exceed {limit}"
        )
    return {
        "enabled": True,
        "configured": True,
        "synchronized": True,
        "over_limit": False,
        "limit_bytes": limit,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "allocator_fraction": fraction,
        "within_limit": True,
    }


def _runtime_rejection_reasons(
    evidence: AuditEvidence,
) -> tuple[list[str], dict[str, Any] | None]:
    """Return runtime reasons, or fail if a claimed pass lacks complete proof."""

    record = evidence.record
    context = f"{evidence.source_path}:{evidence.line_number}"
    if record["passed"] is False:
        reasons = [f"runtime_status:{record['status']}"]
        validator_reasons = record.get("failure_reasons")
        if isinstance(validator_reasons, list):
            for reason in validator_reasons:
                if isinstance(reason, str) and reason:
                    reasons.append(f"validator:{reason}")
        error_type = record.get("error_type")
        if isinstance(error_type, str) and error_type:
            reasons.append(f"error_type:{error_type}")
        return list(dict.fromkeys(reasons)), None

    required_true = {
        "kernelgym_compiled": record.get("kernelgym_compiled"),
        "kernelgym_correctness": record.get("kernelgym_correctness"),
        "persistent_model_instances": record.get("persistent_model_instances"),
        "reference_training": record.get("reference_training"),
        "identical_training": record.get("identical_training"),
    }
    bad_true = sorted(name for name, value in required_true.items() if value is not True)
    if bad_true:
        raise IntegrityError(f"{context}: claimed pass lacks true markers: {bad_true}")
    for field in ("reference_forward_calls", "identical_forward_calls"):
        if record.get(field) != EXPECTED_TRIALS:
            raise IntegrityError(
                f"{context}: claimed pass has {field}={record.get(field)!r}, expected {EXPECTED_TRIALS}"
            )

    failure_reasons = record.get("failure_reasons")
    if failure_reasons != []:
        raise IntegrityError(f"{context}: claimed pass must have failure_reasons=[]")
    metadata = record.get("kernelgym_metadata")
    if not isinstance(metadata, Mapping):
        raise IntegrityError(f"{context}: claimed pass lacks kernelgym_metadata")
    metadata_expectations = {
        "correctness_forward_seed_reset_enabled": True,
        "correctness_trials_run": EXPECTED_TRIALS,
        "correctness_trials": f"({EXPECTED_TRIALS} / {EXPECTED_TRIALS})",
        "correctness_budget_min_pass_trials": EXPECTED_TRIALS,
        "train_mode_validator_persistent_instances": True,
        "train_mode_validator_training": True,
        "train_mode_validator_contract": REQUIRED_CONTRACT_VERSION,
    }
    disagreements = [
        f"{name}={metadata.get(name)!r}"
        for name, expected in metadata_expectations.items()
        if metadata.get(name) != expected
    ]
    if disagreements:
        raise IntegrityError(
            f"{context}: claimed pass has incomplete KernelGym contract metadata: {', '.join(disagreements)}"
        )

    guard = _validate_memory_guard(
        record.get("memory_guard"),
        context=f"{context}.memory_guard",
        required_limit_bytes=REQUIRED_MEMORY_LIMIT_BYTES,
    )
    configured_limit_bytes = int(float(record["max_device_memory_gib"]) * GIB_BYTES)
    if guard["limit_bytes"] != configured_limit_bytes:
        raise IntegrityError(
            f"{context}: memory guard limit {guard['limit_bytes']} disagrees with "
            f"configured max_device_memory_gib={record['max_device_memory_gib']!r}"
        )
    return [], guard


def _scan_baselines(
    baseline_files: Sequence[Path],
    *,
    code_column: str,
) -> tuple[dict[str, str], dict[str, str], list[dict[str, Any]]]:
    reference_origins: dict[str, str] = {}
    ast_origins: dict[str, str] = {}
    summaries: list[dict[str, Any]] = []
    for path in baseline_files:
        parquet = pq.ParquetFile(path)
        top_column = code_column.split(".", 1)[0]
        if top_column not in parquet.schema_arrow.names:
            raise IntegrityError(f"decontamination baseline is missing {top_column!r}: {path}")
        row_index = 0
        for batch in parquet.iter_batches(batch_size=512, columns=[top_column]):
            for row in batch.to_pylist():
                code = _nested(row, code_column)
                if not isinstance(code, str) or not code.strip():
                    raise IntegrityError(f"decontamination baseline {path} row {row_index} lacks reference code")
                reference_hash = _sha256_bytes(code.encode("utf-8"))
                ast_hash = _normalized_ast_sha256(code, context=f"baseline {path} row {row_index}")
                origin = f"{path}:{row_index}"
                reference_origins.setdefault(reference_hash, origin)
                ast_origins.setdefault(ast_hash, origin)
                row_index += 1
        summaries.append({"path": str(path), "sha256": _sha256_file(path), "rows": row_index})
    return reference_origins, ast_origins, summaries


def _runtime_evidence_summary(evidence: AuditEvidence, guard: Mapping[str, Any] | None) -> dict[str, Any]:
    record = evidence.record
    summary: dict[str, Any] = {
        "audit_path": str(evidence.source_path),
        "audit_sha256": evidence.source_sha256,
        "audit_line_number": evidence.line_number,
        "contract_version": record["contract_version"],
        "contract_fingerprint": record["contract_fingerprint"],
        "contract_payload": record.get("contract_payload"),
        "source_sha256": record.get("source_sha256"),
        "validator_source_sha256": record.get("validator_source_sha256"),
        "launcher_source_sha256": record.get("launcher_source_sha256"),
        "status": record["status"],
        "passed": record["passed"],
        "trials": record["trials"],
        "training": record["training"],
        "seed": record.get("seed"),
        "device": record.get("device"),
        "max_device_memory_gib": record.get("max_device_memory_gib"),
        "gpu": record.get("gpu"),
        "kernelgym": record.get("kernelgym"),
    }
    if guard is not None:
        summary["memory_guard"] = dict(guard)
    return summary


def _output_record(
    manifest_record: Mapping[str, Any],
    *,
    decision: str,
    reasons: Sequence[str],
    evidence: AuditEvidence,
    guard: Mapping[str, Any] | None,
    decontamination_matches: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    record = dict(manifest_record)
    record["runtime_validation_status"] = decision
    record["runtime_acceptance"] = {
        "materializer": MATERIALIZER_VERSION,
        "decision": decision,
        "reasons": list(reasons),
        "evidence": _runtime_evidence_summary(evidence, guard),
        "exact_decontamination_matches": [dict(match) for match in decontamination_matches],
    }
    return record


def _temp_path(destination: Path) -> Path:
    token = uuidlib.uuid4().hex
    return destination.with_name(f".{destination.name}.{os.getpid()}.{token}.tmp")


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_selected_parquet(input_path: Path, output_path: Path, accepted_indices: Sequence[int]) -> None:
    parquet = pq.ParquetFile(input_path)
    accepted = set(accepted_indices)
    writer = pq.ParquetWriter(output_path, parquet.schema_arrow, compression="zstd", use_dictionary=True)
    try:
        global_index = 0
        for batch in parquet.iter_batches(batch_size=512):
            mask = pa.array([global_index + offset in accepted for offset in range(batch.num_rows)])
            if any(mask.to_pylist()):
                writer.write_batch(batch.filter(mask))
            global_index += batch.num_rows
    finally:
        writer.close()
    _fsync_file(output_path)
    if pq.read_schema(output_path) != parquet.schema_arrow:
        raise RuntimeError("accepted parquet schema changed during materialization")


def _validate_destinations(config: MaterializeConfig, input_files: Sequence[Path]) -> tuple[Path, ...]:
    destinations = tuple(
        path.expanduser().resolve()
        for path in (
            config.output_parquet,
            config.accepted_manifest,
            config.rejected_manifest,
            config.summary_json,
        )
    )
    if len(set(destinations)) != len(destinations):
        raise ValueError("all four output paths must be distinct")
    reads = {path.expanduser().resolve() for path in input_files}
    collisions = sorted(str(path) for path in destinations if path in reads)
    if collisions:
        raise ValueError(f"output paths collide with inputs: {collisions}")
    existing = [str(path) for path in destinations if path.exists()]
    if existing:
        raise FileExistsError(f"outputs already exist: {existing}")
    return destinations


def materialize(config: MaterializeConfig) -> dict[str, Any]:
    """Verify all evidence, stage all outputs, then publish the summary last."""

    input_path = _resolve_file(config.input_parquet, label="candidate parquet", suffix=".parquet")
    manifest_path = _resolve_file(config.generator_manifest, label="generator manifest", suffix=".jsonl")
    audit_files = _expand_paths(config.audit_paths, suffix=".jsonl", label="runtime audit")
    baseline_files = (
        _expand_paths(config.dedup_against, suffix=".parquet", label="decontamination baseline")
        if config.dedup_against
        else ()
    )
    if input_path in baseline_files:
        raise ValueError("candidate parquet cannot also be a decontamination baseline")
    destinations = _validate_destinations(config, [input_path, manifest_path, *audit_files, *baseline_files])

    parquet = pq.ParquetFile(input_path)
    row_count = parquet.metadata.num_rows
    candidate_hash = _sha256_file(input_path)
    manifest_records = _load_generator_manifest(manifest_path, expected_rows=row_count)
    identities = _parquet_identities(
        input_path,
        manifest_records,
        code_column=config.code_column,
        uuid_column=config.uuid_column,
    )
    (
        evidence_by_index,
        audit_summaries,
        contract_version,
        contract_fingerprint,
        toolchain_provenance,
    ) = _load_audits(audit_files, identities, candidate_sha256=candidate_hash)
    reference_origins, ast_origins, baseline_summaries = _scan_baselines(
        baseline_files, code_column=config.code_column
    )

    accepted_indices: list[int] = []
    accepted_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []
    reason_counts: collections.Counter[str] = collections.Counter()
    rejection_class_counts: collections.Counter[str] = collections.Counter()
    for identity, manifest_record in zip(identities, manifest_records, strict=True):
        evidence = evidence_by_index[identity.row_index]
        runtime_reasons, guard = _runtime_rejection_reasons(evidence)
        matches: list[dict[str, str]] = []
        if identity.reference_sha256 in reference_origins:
            matches.append(
                {
                    "kind": "reference_sha256",
                    "sha256": identity.reference_sha256,
                    "baseline_origin": reference_origins[identity.reference_sha256],
                }
            )
        if identity.normalized_ast_sha256 in ast_origins:
            matches.append(
                {
                    "kind": "normalized_ast_sha256",
                    "sha256": identity.normalized_ast_sha256,
                    "baseline_origin": ast_origins[identity.normalized_ast_sha256],
                }
            )
        decontamination_reasons = [f"exact_{match['kind']}_match_baseline" for match in matches]
        reasons = [*runtime_reasons, *decontamination_reasons]
        decision = "accepted" if not reasons else "rejected"
        output_record = _output_record(
            manifest_record,
            decision=decision,
            reasons=reasons,
            evidence=evidence,
            guard=guard,
            decontamination_matches=matches,
        )
        if decision == "accepted":
            accepted_indices.append(identity.row_index)
            accepted_records.append(output_record)
        else:
            rejected_records.append(output_record)
            reason_counts.update(reasons)
            if runtime_reasons:
                rejection_class_counts["runtime"] += 1
            if decontamination_reasons:
                rejection_class_counts["exact_decontamination"] += 1

    output_parquet, accepted_manifest, rejected_manifest, summary_json = destinations
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
    parquet_tmp = _temp_path(output_parquet)
    accepted_tmp = _temp_path(accepted_manifest)
    rejected_tmp = _temp_path(rejected_manifest)
    summary_tmp = _temp_path(summary_json)
    staged = (parquet_tmp, accepted_tmp, rejected_tmp, summary_tmp)
    tool_path = Path(__file__).resolve()
    try:
        _write_selected_parquet(input_path, parquet_tmp, accepted_indices)
        _write_jsonl(accepted_tmp, accepted_records)
        _write_jsonl(rejected_tmp, rejected_records)
        summary: dict[str, Any] = {
            "materializer": MATERIALIZER_VERSION,
            "materializer_source_sha256": _sha256_file(tool_path),
            "candidate": {
                "path": str(input_path),
                "sha256": candidate_hash,
                "rows": row_count,
                "arrow_schema_sha256": _sha256_bytes(parquet.schema_arrow.serialize().to_pybytes()),
            },
            "generator_manifest": {
                "path": str(manifest_path),
                "sha256": _sha256_file(manifest_path),
                "records": len(manifest_records),
            },
            "runtime_audits": audit_summaries,
            "runtime_contract": {
                "contract_version": contract_version,
                "trials_required": EXPECTED_TRIALS,
                "training_required": True,
                "paired_rng_required": True,
                "persistent_instances_required": True,
                "memory_guard_required": True,
                "memory_guard_limit_bytes": REQUIRED_MEMORY_LIMIT_BYTES,
                "contract_fingerprint": contract_fingerprint,
                **toolchain_provenance,
            },
            "exact_decontamination": {
                "baselines": baseline_summaries,
                "reference_sha256_values": len(reference_origins),
                "normalized_ast_sha256_values": len(ast_origins),
                "method": "exact reference SHA-256 and normalized AST SHA-256 only",
                "near_duplicate_handling": "not performed here; use the upstream static provenance pipeline",
            },
            "decisions": {
                "accepted": len(accepted_records),
                "rejected": len(rejected_records),
                "rejection_class_counts": dict(sorted(rejection_class_counts.items())),
                "rejection_reason_counts": dict(sorted(reason_counts.items())),
                "accepted_row_order": "original candidate row order",
            },
            "outputs": {
                "accepted_parquet": {
                    "path": str(output_parquet),
                    "sha256": _sha256_file(parquet_tmp),
                    "rows": len(accepted_records),
                },
                "accepted_manifest": {
                    "path": str(accepted_manifest),
                    "sha256": _sha256_file(accepted_tmp),
                    "records": len(accepted_records),
                },
                "rejected_manifest": {
                    "path": str(rejected_manifest),
                    "sha256": _sha256_file(rejected_tmp),
                    "records": len(rejected_records),
                },
                "publication": "all outputs staged first; per-file atomic replace; summary published last",
            },
            "evidence_complete": True,
        }
        _write_json(summary_tmp, summary)

        for staged_path, destination in zip(staged, destinations, strict=True):
            os.replace(staged_path, destination)
        for parent in {destination.parent for destination in destinations}:
            directory_fd = os.open(parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        return summary
    except BaseException:
        for path in staged:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Synthesized candidate parquet")
    parser.add_argument("--generator-manifest", type=Path, required=True, help="Generator JSONL manifest")
    parser.add_argument(
        "--audit",
        type=Path,
        action="append",
        required=True,
        dest="audit_paths",
        help="Runtime audit JSONL or directory; repeat for multiple shards/hosts",
    )
    parser.add_argument(
        "--dedup-against",
        type=Path,
        action="append",
        default=[],
        help="Baseline parquet or directory for exact decontamination; repeat as needed",
    )
    parser.add_argument("--output-parquet", type=Path, required=True)
    parser.add_argument("--accepted-manifest", type=Path, required=True)
    parser.add_argument("--rejected-manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True, dest="summary_json")
    parser.add_argument("--code-column", default="reward_model.ground_truth")
    parser.add_argument("--uuid-column", default="extra_info.uuid")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    summary = materialize(
        MaterializeConfig(
            input_parquet=args.input,
            generator_manifest=args.generator_manifest,
            audit_paths=tuple(args.audit_paths),
            output_parquet=args.output_parquet,
            accepted_manifest=args.accepted_manifest,
            rejected_manifest=args.rejected_manifest,
            summary_json=args.summary_json,
            dedup_against=tuple(args.dedup_against),
            code_column=args.code_column,
            uuid_column=args.uuid_column,
        )
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
