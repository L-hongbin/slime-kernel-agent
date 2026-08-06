#!/usr/bin/env python3
"""Exactly verify paired value-lane evidence and materialize live children."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from tools.data.synthesize import validate_train_mode_contract as reference_validator
from tools.data.synthesize.random_method import solve_value_coverage as value_solver

REFERENCE_CONTRACT = "kernelgym-reference-self-train-mode-v3"
LIVENESS_CONTRACT = "random_value_runtime_liveness_v2"
LIVENESS_BINDING_VERSION = "random_value_runtime_liveness_binding_v2"
ANALYSIS_CONTRACT = "random_value_lane_exact_analysis_v3"
SHARD_RE = re.compile(r"^shard-(\d+)-of-(\d+)\.jsonl$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
EXPECTED_VALUE_FAMILIES = frozenset({"uniform_01", "signed_uniform", "poisson_counts", "multinomial_categories"})
MAX_AUTHORIZED_CANDIDATES = 5_000
MIN_LIVENESS_TRIALS = 3
MAX_COUNT_HISTOGRAM_BINS = 64
EXPECTED_KERNELGYM_COMMIT = "26255057463a77b23abac0f3e5eafeeebf2ebbb5"
EXPECTED_KERNELGYM_HASHES = {
    "correctness_sha256": "a77f758a1ffb600cf8ada2290c54fe91b3cb691e81aef03f4fdb5a7e42a764b5",
    "loading_sha256": "8d0f6bb9f17802281997d764b349903799b3d9235627bc1c73c474f187862d06",
    "exec_types_sha256": "8c209627ec288679520dec4c1f8232512cd927051a17c11ec15d5768a56d9907",
    "profiling_sha256": "952e9d1618c1ef1803d7fae53171bcd7fa982e522b70d6355bb00f932ae6c29e",
    "config_init_sha256": "e027bcaa35e4cb55062f4ac0a5250393fa85976625b3c31c6051df84efb97bc7",
    "config_settings_sha256": "6bf33dde4269fcd54452de04191d8257069d3da396d5f2f31256a95596c7cf7e",
    "evaluator_bundle_sha256": "135f1758dcfd29ce8e8311a61adacc4a78e7c12433194b21749b37a1882a14ee",
}
REFERENCE_GPU_FIELDS = frozenset({"device", "name", "compute_capability", "torch_version", "torch_cuda_version"})
LIVENESS_TENSOR_FIELDS = frozenset(
    {
        "path",
        "shape",
        "dtype",
        "stride",
        "storage_offset",
        "is_pinned",
        "numel",
        "parent_finite_elements",
        "minimum",
        "maximum",
        "unique_count",
        "count_histogram",
        "rate_summary",
    }
)
LIVENESS_TRIAL_FIELDS = frozenset(
    {
        "trial",
        "seed",
        "changed_tensors",
        "tensors",
        "output_changed",
        "changed_output_elements",
        "output_elements",
        "max_changed_output_fraction",
    }
)
POISSON_RATE_FIELDS = frozenset(
    {
        "mapping",
        "minimum",
        "maximum",
        "mean",
        "histogram_schema_version",
        "histogram",
        "histogram_total_frequency",
    }
)
POISSON_RATE_HISTOGRAM_FIELDS = frozenset({"lower", "upper", "lower_inclusive", "upper_inclusive", "frequency"})
POISSON_RATE_MAPPING = "clamp(softplus(parent),0.125,8.0)"
POISSON_RATE_HISTOGRAM_VERSION = "poisson_rate_histogram_v1"
POISSON_RATE_BINS = (
    (0.125, 0.25, False),
    (0.25, 0.5, False),
    (0.5, 1.0, False),
    (1.0, 2.0, False),
    (2.0, 4.0, False),
    (4.0, 8.0, True),
)
REFERENCE_PAYLOAD_FIELDS = frozenset(
    {
        "contract_version",
        "source_sha256",
        "validator_source_sha256",
        "launcher_source_sha256",
        "kernelgym_correctness_sha256",
        "kernelgym_loading_sha256",
        "kernelgym_exec_types_sha256",
        "kernelgym_profiling_sha256",
        "kernelgym_config_init_sha256",
        "kernelgym_config_settings_sha256",
        "kernelgym_evaluator_bundle_sha256",
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
)
REFERENCE_KERNELGYM_HASH_FIELDS = frozenset(
    {
        "correctness_sha256",
        "loading_sha256",
        "exec_types_sha256",
        "profiling_sha256",
        "config_init_sha256",
        "config_settings_sha256",
        "evaluator_bundle_sha256",
    }
)
REFERENCE_FIXED_POLICY = {
    "contract_version": REFERENCE_CONTRACT,
    "code_column": "reward_model.ground_truth",
    "entry_point_column": "extra_info.entry_point",
    "mode_class_column": "extra_info.v4.mode_class",
    "expected_mode_class": None,
    "device": "cuda:0",
    "max_device_memory_gib": 64.0,
    "trials": 5,
    "seed": 42,
    "training": True,
    "paired_rng_seed_reset_required": True,
    "persistent_model_instances_required": True,
}
LIVENESS_FIXED_POLICY = {
    "device": "cuda:0",
    "seed": 17,
    "max_device_memory_gib": 64.0,
}
LIVENESS_CONFIG_FIELDS = frozenset({"device", "trials", "seed", "timeout_seconds", "max_device_memory_gib"})
ACCEPTED_RUNTIME_STATUS = "value_only_paired_reference_and_liveness_passed"
ACCEPTED_GOVERNANCE_STATUS = "runtime_accepted_review_only_training_not_approved"
LIVENESS_EVIDENCE_FIELDS = (
    "contract_version",
    "binding_version",
    "validator_source_sha256",
    "launcher_source_sha256",
    "parents_sha256",
    "children_sha256",
    "manifest_sha256",
    "allowlist_sha256",
    "validation_config",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be 64 lowercase hexadecimal characters")
    return value


def _require_exact_mapping(value: Any, expected_keys: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    actual_keys = set(value)
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(f"{label} keys differ: missing={missing}, unexpected={unexpected}")
    return value


def _verify_gpu_evidence(
    value: Any,
    *,
    label: str,
    expected_device: str,
    require_cudnn: bool,
) -> dict[str, Any]:
    expected_fields = REFERENCE_GPU_FIELDS | ({"cudnn_version"} if require_cudnn else set())
    gpu = _require_exact_mapping(value, frozenset(expected_fields), label=label)
    if gpu.get("device") != expected_device:
        raise ValueError(f"{label}.device differs from the runtime policy")
    if not isinstance(gpu.get("name"), str) or not gpu["name"]:
        raise ValueError(f"{label}.name must be a non-empty string")
    capability = gpu.get("compute_capability")
    if (
        not isinstance(capability, list)
        or len(capability) != 2
        or any(type(part) is not int or part < 0 for part in capability)
    ):
        raise ValueError(f"{label}.compute_capability must contain two non-negative integers")
    for field in ("torch_version", "torch_cuda_version"):
        if not isinstance(gpu.get(field), str) or not gpu[field]:
            raise ValueError(f"{label}.{field} must be a non-empty string")
    if require_cudnn and (type(gpu.get("cudnn_version")) is not int or gpu["cudnn_version"] <= 0):
        raise ValueError(f"{label}.cudnn_version must be a positive integer")
    return {field: gpu[field] for field in sorted(REFERENCE_GPU_FIELDS)}


def _verify_kernelgym_installation(root_value: Any) -> dict[str, Any]:
    if not isinstance(root_value, str) or not root_value:
        raise ValueError("reference evidence does not identify a KernelGYM installation")
    observed = reference_validator.kernelgym_contract_metadata(Path(root_value))
    if observed.get("git_commit") != EXPECTED_KERNELGYM_COMMIT:
        raise ValueError("KernelGYM installation commit differs from the frozen authority")
    for field, expected in EXPECTED_KERNELGYM_HASHES.items():
        if observed.get(field) != expected:
            raise ValueError(f"KernelGYM installation source hash differs from the frozen authority: {field}")
    return observed


def _require_candidate_count(count: int) -> None:
    if not 1 <= count <= MAX_AUTHORIZED_CANDIDATES:
        raise ValueError(f"candidate count must be in [1, {MAX_AUTHORIZED_CANDIDATES}], found {count}")


def _nested_counts(counter: Mapping[tuple[str, str], int]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for (outer, inner), count in sorted(counter.items()):
        result.setdefault(outer, {})[inner] = count
    return result


def _finite_float(value: Any) -> float | None:
    if type(value) not in (int, float):
        return None
    try:
        converted = float(value)
    except (OverflowError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _nested(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(record, dict) for record in records):
        raise ValueError(f"manifest contains a non-object record: {path}")
    return records


def _shard_records(run_dir: Path) -> tuple[int, list[tuple[int, int, dict[str, Any]]]]:
    matched: list[tuple[int, int, Path]] = []
    for path in run_dir.glob("shard-*-of-*.jsonl"):
        match = SHARD_RE.fullmatch(path.name)
        if match:
            matched.append((int(match.group(1)), int(match.group(2)), path))
    if not matched:
        raise FileNotFoundError(f"no shard JSONL files under {run_dir}")
    counts = {count for _, count, _ in matched}
    if len(counts) != 1:
        raise ValueError(f"mixed shard counts under {run_dir}: {sorted(counts)}")
    shard_count = counts.pop()
    by_index = {index: path for index, count, path in matched if count == shard_count}
    if len(by_index) != len(matched):
        raise ValueError(f"duplicate shard index under {run_dir}")
    if set(by_index) != set(range(shard_count)):
        raise ValueError(f"incomplete shard family under {run_dir}: {sorted(by_index)}/{shard_count}")
    records: list[tuple[int, int, dict[str, Any]]] = []
    for shard_index in range(shard_count):
        path = by_index[shard_index]
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"non-object record at {path}:{line_number}")
            records.append((shard_index, line_number, record))
    return shard_count, records


def _normalize_source_parent(row: Mapping[str, Any], *, source_row_index: int) -> dict[str, Any]:
    """Mirror the nullable augmentation field added when source parents enter the lane."""

    normalized = dict(row)
    extra_info = normalized.get("extra_info")
    if not isinstance(extra_info, Mapping):
        raise ValueError(f"source row lacks extra_info: {source_row_index}")
    normalized_extra_info = dict(extra_info)
    if "augmentation" in normalized_extra_info:
        raise ValueError(f"source unexpectedly contains augmentation metadata: {source_row_index}")
    normalized_extra_info["augmentation"] = None
    normalized["extra_info"] = normalized_extra_info
    return normalized


def _verify_source_binding(
    manifests: Sequence[Mapping[str, Any]],
) -> tuple[dict[int, dict[str, Any]], value_solver.SourceContext]:
    """Reopen and revalidate the exact canonical or shape-resampled source artifact."""

    if not manifests:
        raise ValueError("cannot verify source binding for an empty manifest")
    source_paths: set[str] = set()
    source_hashes: set[str] = set()
    source_bindings: set[bytes] = set()
    source_indices: set[int] = set()
    for candidate_index, manifest in enumerate(manifests):
        source_path = manifest.get("source_artifact_path")
        if not isinstance(source_path, str) or not source_path:
            raise ValueError(f"invalid source artifact path at row {candidate_index}")
        source_paths.add(source_path)
        source_hashes.add(
            _require_sha256(
                manifest.get("source_artifact_sha256"),
                label=f"manifest[{candidate_index}].source_artifact_sha256",
            )
        )
        source_binding = manifest.get("source_binding")
        if not isinstance(source_binding, Mapping):
            raise ValueError(f"invalid source binding at row {candidate_index}")
        source_bindings.add(_canonical_json_bytes(source_binding))
        source_row_index = manifest.get("source_row_index")
        if type(source_row_index) is not int or source_row_index < 0:
            raise ValueError(f"invalid source row index at row {candidate_index}: {source_row_index}")
        source_indices.add(source_row_index)
    if len(source_paths) != 1 or len(source_hashes) != 1 or len(source_bindings) != 1:
        raise ValueError("lane manifests do not bind exactly one source artifact and contract")

    source_path_text = next(iter(source_paths))
    source_path = Path(source_path_text)
    if not source_path.is_absolute() or not source_path.is_file():
        raise ValueError(f"source artifact is unavailable: {source_path_text}")
    resolved_source_path = source_path.resolve()
    if str(resolved_source_path) != source_path_text:
        raise ValueError(f"source path is not resolved: {source_path_text}")
    recorded_source_sha256 = next(iter(source_hashes))
    actual_source_sha256 = _sha256_file(resolved_source_path)
    if actual_source_sha256 != recorded_source_sha256:
        raise ValueError("source artifact hash differs from the manifest binding")

    parquet = pq.ParquetFile(resolved_source_path)
    source_context = value_solver._source_context(resolved_source_path, actual_source_sha256, parquet)
    recorded_binding = json.loads(next(iter(source_bindings)))
    if _canonical_json_bytes(source_context.binding) != _canonical_json_bytes(recorded_binding):
        raise ValueError("reconstructed source binding differs from the lane manifest")
    if source_indices and max(source_indices) >= parquet.metadata.num_rows:
        raise ValueError(f"source row index is out of bounds: {max(source_indices)}")

    selected_rows: dict[int, dict[str, Any]] = {}
    row_offset = 0
    for batch in parquet.iter_batches(batch_size=256, use_threads=False):
        batch_rows = batch.to_pylist()
        for local_index, row in enumerate(batch_rows):
            source_row_index = row_offset + local_index
            if source_row_index in source_indices:
                if not isinstance(row, dict):
                    raise ValueError(f"source row is not an object: {source_row_index}")
                selected_rows[source_row_index] = _normalize_source_parent(row, source_row_index=source_row_index)
        row_offset += len(batch_rows)
        if len(selected_rows) == len(source_indices):
            break
    if set(selected_rows) != source_indices:
        raise ValueError("failed to load every source row bound by the lane manifest")
    return selected_rows, source_context


def _failure_kind(record: Mapping[str, Any]) -> str:
    if record.get("passed") is True:
        return "passed"
    text = " ".join(
        [
            str(record.get("status", "")),
            str(record.get("reason", "")),
            str(record.get("error", "")),
            str(record.get("error_type", "")),
            str(record.get("worker_stderr_tail", "")),
        ]
        + [str(item) for item in record.get("failure_reasons", [])]
    ).lower()
    if "out of memory" in text or "oom" in text or "memory_guard" in text or "memory" in text:
        return "oom_or_memory_guard"
    if "timeout" in text:
        return "timeout"
    if any(
        token in text
        for token in (
            "protocol",
            "worker",
            "compile",
            "evaluator",
            "infrastructure",
            "cuda is unavailable",
            "driver initialization failed",
        )
    ):
        return "evaluator_or_environment"
    if "non_finite" in text or "numer" in text or "correctness" in text:
        return "numerical_or_correctness"
    return "other_reference_failure"


def _verify_aligned_static(
    parents: list[dict[str, Any]],
    children: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
) -> value_solver.SourceContext:
    _require_candidate_count(len(children))
    if not (len(parents) == len(children) == len(manifests)):
        raise ValueError(f"static artifact count mismatch: {len(parents)}:{len(children)}:{len(manifests)}")

    generator_path = Path(value_solver.__file__).resolve()
    dependency_path = generator_path.parent.parent / "augment_prompt_tasks.py"
    current_generator_sha256 = _sha256_file(generator_path)
    current_dependency_sha256 = _sha256_file(dependency_path)
    git_commit_values = [manifest.get("git_commit") for manifest in manifests]
    if any(not isinstance(value, str) for value in git_commit_values):
        raise ValueError("lane manifest source git commit is invalid")
    git_commits = set(git_commit_values)
    if len(git_commits) != 1:
        raise ValueError("lane manifests do not bind exactly one source git commit")
    source_git_commit = next(iter(git_commits))
    if not isinstance(source_git_commit, str) or re.fullmatch(r"[0-9a-f]{40}", source_git_commit) is None:
        raise ValueError("lane manifest source git commit is invalid")
    committed_generator_sha256 = value_solver._git_blob_sha256(source_git_commit, generator_path)
    committed_dependency_sha256 = value_solver._git_blob_sha256(source_git_commit, dependency_path)
    if committed_generator_sha256 != current_generator_sha256:
        raise ValueError("current generator source differs from the manifest commit blob")
    if committed_dependency_sha256 != current_dependency_sha256:
        raise ValueError("current dependency source differs from the manifest commit blob")
    source_rows, source_context = _verify_source_binding(manifests)
    parent_uuids: set[str] = set()
    child_uuids: set[str] = set()
    child_references: set[str] = set()
    child_asts: set[str] = set()
    for index, (parent, child, manifest) in enumerate(zip(parents, children, manifests, strict=True)):
        parent_uuid = _nested(parent, "extra_info.uuid")
        child_uuid = _nested(child, "extra_info.uuid")
        parent_code = _nested(parent, "reward_model.ground_truth")
        child_code = _nested(child, "reward_model.ground_truth")
        if type(manifest.get("candidate_row_index")) is not int or manifest["candidate_row_index"] != index:
            raise ValueError(f"manifest index mismatch: {index}")
        if manifest.get("manifest_contract_version") != "random_value_lane_manifest_v3":
            raise ValueError(f"manifest contract mismatch at row {index}")
        if manifest.get("generator_source_sha256") != current_generator_sha256:
            raise ValueError(f"current generator source hash mismatch at row {index}")
        if manifest.get("dependency_source_sha256") != current_dependency_sha256:
            raise ValueError(f"current dependency source hash mismatch at row {index}")
        if manifest.get("parent_uuid") != parent_uuid or manifest.get("child_uuid") != child_uuid:
            raise ValueError(f"manifest UUID mismatch: {index}")
        if not isinstance(parent_code, str) or not isinstance(child_code, str):
            raise ValueError(f"missing reference code: {index}")
        if _sha256_bytes(parent_code.encode()) != manifest.get("parent_reference_sha256"):
            raise ValueError(f"parent reference hash mismatch: {index}")
        if _sha256_bytes(child_code.encode()) != manifest.get("child_reference_sha256"):
            raise ValueError(f"child reference hash mismatch: {index}")
        if manifest.get("primary_intervention") != "random_value":
            raise ValueError(f"non-value intervention at row {index}")
        if manifest.get("assigned_target") not in EXPECTED_VALUE_FAMILIES:
            raise ValueError(f"unexpected value family at row {index}: {manifest.get('assigned_target')}")
        if manifest.get("generator_contract_version") != value_solver.CONTRACT_VERSION:
            raise ValueError(f"unexpected generator contract at row {index}")
        for field in ("shape_changed", "dtype_changed", "layout_changed", "model_changed"):
            if manifest.get(field) is not False:
                raise ValueError(f"{field} is not false at row {index}")
        if (
            manifest.get("get_init_inputs_changed") is not False
            or manifest.get("rng_consumption_preserved") is not True
        ):
            raise ValueError(f"value-only static contract failed at row {index}")
        if not all(isinstance(value, str) and value for value in (parent_uuid, child_uuid)):
            raise ValueError(f"invalid UUID at row {index}")

        source_row_index = manifest.get("source_row_index")
        if type(source_row_index) is not int or source_row_index < 0:
            raise ValueError(f"invalid source row index at row {index}: {source_row_index}")
        source_artifact_path = manifest.get("source_artifact_path")
        source_artifact_sha256 = manifest.get("source_artifact_sha256")
        git_commit = manifest.get("git_commit")
        if not all(
            isinstance(value, str) and value for value in (source_artifact_path, source_artifact_sha256, git_commit)
        ):
            raise ValueError(f"invalid replay provenance at row {index}")
        if re.fullmatch(r"[0-9a-f]{40}", git_commit) is None:
            raise ValueError(f"invalid source git commit at row {index}")
        if git_commit != source_git_commit:
            raise ValueError(f"source git commit differs across lane manifests at row {index}")
        if manifest["generator_source_sha256"] != committed_generator_sha256:
            raise ValueError(f"generator source does not match the manifest commit blob at row {index}")
        if manifest["dependency_source_sha256"] != committed_dependency_sha256:
            raise ValueError(f"dependency source does not match the manifest commit blob at row {index}")
        source_parent = source_rows[source_row_index]
        if _canonical_json_bytes(source_parent) != _canonical_json_bytes(parent):
            raise ValueError(f"parent differs from source row at candidate {index}")
        if manifest.get("source_kind") != source_context.kind:
            raise ValueError(f"source kind mismatch at row {index}")
        if _canonical_json_bytes(manifest.get("source_binding")) != _canonical_json_bytes(source_context.binding):
            raise ValueError(f"source binding mismatch at row {index}")
        if source_context.resample_manifests is None:
            source_row_binding = None
            expected_canonical_parent_uuid = parent_uuid
            expected_lineage = "canonical_parent_value_only_child"
            expected_inherited_shape = False
            expected_upstream_index = None
            expected_upstream_sha256 = None
        else:
            source_row_binding = source_context.resample_manifests[source_row_index]
            expected_canonical_parent_uuid = source_row_binding.get("canonical_parent_uuid")
            expected_lineage = "canonical_parent_shape_child_random_value_grandchild"
            expected_inherited_shape = True
            expected_upstream_index = source_row_binding.get("selected_index")
            expected_upstream_sha256 = _canonical_sha256(source_row_binding)
        if (
            manifest.get("canonical_parent_uuid") != expected_canonical_parent_uuid
            or manifest.get("lineage") != expected_lineage
            or manifest.get("inherited_shape_intervention") is not expected_inherited_shape
            or manifest.get("upstream_resample_selected_index") != expected_upstream_index
            or manifest.get("upstream_resample_manifest_row_sha256") != expected_upstream_sha256
        ):
            raise ValueError(f"source lineage mismatch at row {index}")
        if _nested(child, "extra_info.v4.parent_uuid") != parent_uuid:
            raise ValueError(f"child does not point to its immediate parent at row {index}")
        replay_eligible, replay_reason = value_solver._analyze_parent(parent, source_row_index)
        if replay_eligible is None:
            raise ValueError(f"current solver rejects parent at row {index}: {replay_reason}")
        replay_child, replay_manifest = value_solver._make_child(
            parent,
            replay_eligible,
            source_path=Path(source_artifact_path),
            source_sha256=source_artifact_sha256,
            generator_sha256=manifest["generator_source_sha256"],
            dependency_sha256=manifest["dependency_source_sha256"],
            git_commit=git_commit,
            source_binding=source_context.binding,
            source_row_binding=source_row_binding,
        )
        replay_manifest["candidate_row_index"] = index
        if replay_child != child or _canonical_json_bytes(replay_child) != _canonical_json_bytes(child):
            raise ValueError(f"deterministic child replay mismatch at row {index}")
        if replay_manifest != manifest or _canonical_json_bytes(replay_manifest) != _canonical_json_bytes(manifest):
            raise ValueError(f"deterministic manifest replay mismatch at row {index}")
        parent_uuids.add(parent_uuid)
        child_uuids.add(child_uuid)
        child_references.add(manifest["child_reference_sha256"])
        child_asts.add(manifest["child_normalized_ast_sha256"])
    expected = len(manifests)
    if len(parent_uuids) != expected:
        raise ValueError("candidate parents are not unique")
    if len(child_uuids) != expected or len(child_references) != expected or len(child_asts) != expected:
        raise ValueError("child UUID/reference/AST uniqueness contract failed")
    return source_context


def _verify_reference_passed_record(record: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    if record.get("passed") is not True or record.get("status") != "passed":
        raise ValueError(f"{label} is not a passed reference record")
    if record.get("failure_reasons") != []:
        raise ValueError(f"{label} passed with non-empty failure reasons")
    gpu = _verify_gpu_evidence(
        record.get("gpu"),
        label=f"{label}.gpu",
        expected_device=REFERENCE_FIXED_POLICY["device"],
        require_cudnn=False,
    )
    memory_guard = record.get("memory_guard")
    if not isinstance(memory_guard, Mapping) or memory_guard.get("within_limit") is not True:
        raise ValueError(f"{label} lacks a passing memory guard")
    for field in (
        "persistent_model_instances",
        "reference_training",
        "identical_training",
        "kernelgym_correctness",
        "kernelgym_compiled",
    ):
        if record.get(field) is not True:
            raise ValueError(f"{label}.{field} is not true")
    expected_calls = REFERENCE_FIXED_POLICY["trials"]
    for field in ("reference_forward_calls", "identical_forward_calls"):
        if type(record.get(field)) is not int or record[field] != expected_calls:
            raise ValueError(f"{label}.{field} does not match the frozen trial count")
    return gpu


def _verify_reference_contract_record(
    record: Mapping[str, Any],
    *,
    row_index: int,
    source_path: str,
    source_sha256: str,
    launcher_source_sha256: str,
    validator_source_sha256: str,
    expected_kernelgym: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Verify one reference record against the frozen paired-runtime policy."""

    label = f"reference row {row_index}"
    if record.get("contract_version") != REFERENCE_CONTRACT:
        raise ValueError(f"{label} has an unexpected contract")
    if record.get("source_path") != source_path or record.get("source_sha256") != source_sha256:
        raise ValueError(f"{label} is not bound to the paired parquet")
    if record.get("launcher_source_sha256") != launcher_source_sha256:
        raise ValueError(f"{label} launcher source hash mismatch")
    if record.get("validator_source_sha256") != validator_source_sha256:
        raise ValueError(f"{label} validator source hash mismatch")
    for field in ("device", "max_device_memory_gib", "trials", "seed", "training"):
        if record.get(field) != REFERENCE_FIXED_POLICY[field]:
            raise ValueError(f"{label} top-level policy mismatch: {field}")

    payload_mapping = _require_exact_mapping(
        record.get("contract_payload"),
        REFERENCE_PAYLOAD_FIELDS,
        label=f"{label}.contract_payload",
    )
    payload = dict(payload_mapping)
    expected_links = {
        **REFERENCE_FIXED_POLICY,
        "source_sha256": source_sha256,
        "launcher_source_sha256": launcher_source_sha256,
        "validator_source_sha256": validator_source_sha256,
    }
    for field, expected in expected_links.items():
        if payload.get(field) != expected:
            raise ValueError(f"{label} contract payload mismatch: {field}")
    for field in (
        "source_sha256",
        "launcher_source_sha256",
        "validator_source_sha256",
        *(f"kernelgym_{name}" for name in REFERENCE_KERNELGYM_HASH_FIELDS),
    ):
        _require_sha256(payload.get(field), label=f"{label}.contract_payload.{field}")

    fingerprint = _require_sha256(record.get("contract_fingerprint"), label=f"{label}.contract_fingerprint")
    if fingerprint != _canonical_sha256(payload):
        raise ValueError(f"{label} contract fingerprint does not match its payload")

    kernelgym = record.get("kernelgym")
    if not isinstance(kernelgym, Mapping):
        raise ValueError(f"{label} lacks KernelGYM source evidence")
    for field, expected in expected_kernelgym.items():
        if kernelgym.get(field) != expected:
            raise ValueError(f"{label} KernelGYM installation mismatch: {field}")
    for field in REFERENCE_KERNELGYM_HASH_FIELDS:
        if kernelgym.get(field) != payload[f"kernelgym_{field}"]:
            raise ValueError(f"{label} KernelGYM source hash mismatch: {field}")
    return fingerprint, payload


def _verify_reference(
    *,
    reference_dir: Path,
    paired_path: Path,
    parents: list[dict[str, Any]],
    children: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
) -> tuple[dict[str, Any], set[str], dict[str, dict[str, Any]]]:
    shard_count, raw = _shard_records(reference_dir)
    expected_rows = 2 * len(children)
    if len(raw) != expected_rows:
        raise ValueError(f"reference row count mismatch: expected {expected_rows}, found {len(raw)}")
    paired_sha256 = _sha256_file(paired_path)
    paired_resolved = str(paired_path.resolve())
    archived_launcher_sha256 = _sha256_file(reference_dir / "launcher_source.sh")
    current_launcher_sha256 = _sha256_file(
        Path(__file__).resolve().parent.parent / "launch_reference_validation_shards.sh"
    )
    if archived_launcher_sha256 != current_launcher_sha256:
        raise ValueError("archived reference launcher differs from the current frozen source")
    current_validator_sha256 = _sha256_file(Path(reference_validator.__file__).resolve())
    by_row: dict[int, dict[str, Any]] = {}
    fingerprints: set[str] = set()
    payloads: set[bytes] = set()
    kernelgym_by_root: dict[str, dict[str, Any]] = {}
    environment_records: list[dict[str, Any]] = []
    for shard_index, line_number, record in raw:
        row_index = record.get("row_index")
        if type(row_index) is not int or not 0 <= row_index < expected_rows:
            raise ValueError(f"invalid reference row index at shard {shard_index}:{line_number}: {row_index}")
        if row_index % shard_count != shard_index:
            raise ValueError(f"reference row assigned to wrong shard: row={row_index}, shard={shard_index}")
        if row_index in by_row:
            raise ValueError(f"duplicate reference row: {row_index}")
        if type(record.get("passed")) is not bool:
            raise ValueError(f"invalid reference verdict at row {row_index}")
        if (record["passed"] and record.get("status") != "passed") or (
            not record["passed"] and record.get("status") == "passed"
        ):
            raise ValueError(f"reference status/verdict mismatch at row {row_index}")
        kernelgym_record = record.get("kernelgym")
        kernelgym_root = kernelgym_record.get("root") if isinstance(kernelgym_record, Mapping) else None
        if not isinstance(kernelgym_root, str):
            raise ValueError(f"reference row {row_index} lacks a KernelGYM root")
        expected_kernelgym = kernelgym_by_root.get(kernelgym_root)
        if expected_kernelgym is None:
            expected_kernelgym = _verify_kernelgym_installation(kernelgym_root)
            kernelgym_by_root[kernelgym_root] = expected_kernelgym
        fingerprint, payload = _verify_reference_contract_record(
            record,
            row_index=row_index,
            source_path=paired_resolved,
            source_sha256=paired_sha256,
            launcher_source_sha256=archived_launcher_sha256,
            validator_source_sha256=current_validator_sha256,
            expected_kernelgym=expected_kernelgym,
        )
        fingerprints.add(fingerprint)
        payloads.add(_canonical_json_bytes(payload))
        if record["passed"]:
            common_gpu = _verify_reference_passed_record(record, label=f"reference row {row_index}")
        elif isinstance(record.get("gpu"), Mapping):
            common_gpu = _verify_gpu_evidence(
                record["gpu"],
                label=f"reference row {row_index}.gpu",
                expected_device=REFERENCE_FIXED_POLICY["device"],
                require_cudnn=False,
            )
        else:
            common_gpu = None
        if common_gpu is not None:
            environment_records.append(
                {
                    "gpu": common_gpu,
                    "kernelgym": {"git_commit": EXPECTED_KERNELGYM_COMMIT, **EXPECTED_KERNELGYM_HASHES},
                }
            )
        by_row[row_index] = record
    if set(by_row) != set(range(expected_rows)):
        raise ValueError("reference rows are not exactly contiguous")

    launchers = {record["launcher_source_sha256"] for record in by_row.values()}
    validators = {record["validator_source_sha256"] for record in by_row.values()}
    environments = {_canonical_sha256(record) for record in environment_records}
    if any(len(values) != 1 for values in (fingerprints, payloads, launchers, validators, environments)):
        raise ValueError("reference evidence mixes policy or runtime fingerprints")

    classifications: collections.Counter[str] = collections.Counter()
    family_classifications: collections.Counter[tuple[str, str]] = collections.Counter()
    failure_attribution: collections.Counter[str] = collections.Counter()
    failure_details: collections.Counter[str] = collections.Counter()
    both_pass: set[str] = set()
    child_records: dict[str, dict[str, Any]] = {}
    for index, (parent, child, manifest) in enumerate(zip(parents, children, manifests, strict=True)):
        parent_record = by_row[2 * index]
        child_record = by_row[2 * index + 1]
        expected_parent = (
            _nested(parent, "extra_info.uuid"),
            manifest["parent_reference_sha256"],
        )
        expected_child = (
            _nested(child, "extra_info.uuid"),
            manifest["child_reference_sha256"],
        )
        if (parent_record.get("uuid"), parent_record.get("reference_sha256")) != expected_parent:
            raise ValueError(f"parent reference identity mismatch at candidate {index}")
        if (child_record.get("uuid"), child_record.get("reference_sha256")) != expected_child:
            raise ValueError(f"child reference identity mismatch at candidate {index}")
        parent_pass = bool(parent_record["passed"])
        child_pass = bool(child_record["passed"])
        if parent_pass and child_pass:
            classification = "both_pass"
            attribution = "no_reference_failure"
            both_pass.add(manifest["child_uuid"])
        elif parent_pass:
            classification = "parent_pass_child_fail"
            child_failure = _failure_kind(child_record)
            if child_failure == "evaluator_or_environment":
                attribution = "evaluator_or_environment_transient"
            else:
                attribution = f"possible_intervention_induced:{child_failure}"
        elif child_pass:
            classification = "parent_fail_child_pass"
            attribution = f"parent_baseline_failure:{_failure_kind(parent_record)}"
        else:
            classification = "both_fail"
            attribution = "baseline_or_evaluator_failure"
        classifications[classification] += 1
        family_classifications[(str(manifest["assigned_target"]), classification)] += 1
        failure_attribution[attribution] += 1
        if not parent_pass:
            failure_details[f"parent:{_failure_kind(parent_record)}"] += 1
        if not child_pass:
            failure_details[f"child:{_failure_kind(child_record)}"] += 1
        child_records[manifest["child_uuid"]] = child_record

    first = by_row[0]
    runtime_environment = environment_records[0]
    summary = {
        "contract_version": ANALYSIS_CONTRACT,
        "reference_contract_version": REFERENCE_CONTRACT,
        "paired_sha256": paired_sha256,
        "shard_count": shard_count,
        "paired_rows": expected_rows,
        "candidate_pairs": len(children),
        "classifications": dict(sorted(classifications.items())),
        "family_classification_counts": _nested_counts(family_classifications),
        "failure_attribution": dict(sorted(failure_attribution.items())),
        "failure_details": dict(sorted(failure_details.items())),
        "both_pass_children": len(both_pass),
        "contract_fingerprint": next(iter(fingerprints)),
        "launcher_source_sha256": next(iter(launchers)),
        "validator_source_sha256": next(iter(validators)),
        "runtime_environment_fingerprint": next(iter(environments)),
        "rows_with_runtime_environment": len(environment_records),
        "rows_failed_before_runtime_environment": expected_rows - len(environment_records),
        "gpu": runtime_environment["gpu"],
        "kernelgym": runtime_environment["kernelgym"],
        "kernelgym_roots": sorted(kernelgym_by_root),
        "contract_payload": first.get("contract_payload"),
    }
    return summary, both_pass, child_records


def _verify_liveness_policy(record: Mapping[str, Any], *, label: str) -> tuple[dict[str, Any], str | None]:
    """Verify one liveness record's policy and return its runtime partition fingerprint."""

    if record.get("contract_version") != LIVENESS_CONTRACT or type(record.get("passed")) is not bool:
        raise ValueError(f"{label} has an invalid contract or verdict")
    if record.get("binding_version") != LIVENESS_BINDING_VERSION:
        raise ValueError(f"{label} has an invalid binding version")
    config_mapping = _require_exact_mapping(
        record.get("validation_config"), LIVENESS_CONFIG_FIELDS, label=f"{label}.validation_config"
    )
    validation_config = dict(config_mapping)
    for field, expected in LIVENESS_FIXED_POLICY.items():
        if validation_config.get(field) != expected:
            raise ValueError(f"{label} liveness policy mismatch: {field}")
    trials = validation_config.get("trials")
    if type(trials) is not int or trials < MIN_LIVENESS_TRIALS:
        raise ValueError(f"{label} must configure at least {MIN_LIVENESS_TRIALS} liveness trials")
    timeout_seconds = _finite_float(validation_config.get("timeout_seconds"))
    if timeout_seconds is None or timeout_seconds <= 0:
        raise ValueError(f"{label} has an invalid liveness timeout")

    evidence = {field: record.get(field) for field in LIVENESS_EVIDENCE_FIELDS}
    for field in (
        "validator_source_sha256",
        "launcher_source_sha256",
        "parents_sha256",
        "children_sha256",
        "manifest_sha256",
        "allowlist_sha256",
    ):
        _require_sha256(evidence.get(field), label=f"{label}.{field}")
    validation_binding_sha256 = _require_sha256(
        record.get("validation_binding_sha256"), label=f"{label}.validation_binding_sha256"
    )
    if validation_binding_sha256 != _canonical_sha256(evidence):
        raise ValueError(f"{label} binding does not match its evidence")

    gpu = record.get("gpu")
    if gpu is None:
        if record["passed"]:
            raise ValueError(f"{label} passed without GPU evidence")
        return validation_config, None
    _verify_gpu_evidence(
        gpu,
        label=f"{label}.gpu",
        expected_device=validation_config["device"],
        require_cudnn=True,
    )
    runtime_partition = {"gpu": dict(gpu), "validation_config": validation_config}
    return validation_config, _canonical_sha256(runtime_partition)


def _validated_poisson_histogram(tensor: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Validate one bounded Poisson histogram emitted by the liveness validator."""

    histogram = _require_exact_mapping(
        tensor.get("count_histogram"),
        frozenset(
            {
                "value_frequencies",
                "truncated",
                "recorded_bins",
                "omitted_bins",
                "omitted_frequency",
                "omitted_value_minimum",
                "omitted_value_maximum",
            }
        ),
        label=f"{label}.count_histogram",
    )
    value_frequencies = histogram["value_frequencies"]
    if not isinstance(value_frequencies, Mapping) or not value_frequencies:
        raise ValueError(f"{label} lacks non-empty Poisson value_frequencies")
    recorded: collections.Counter[int] = collections.Counter()
    for raw_count, frequency in value_frequencies.items():
        if not isinstance(raw_count, str) or re.fullmatch(r"(?:0|[1-9][0-9]*)", raw_count) is None:
            raise ValueError(f"{label} has a non-canonical Poisson histogram key: {raw_count!r}")
        if type(frequency) is not int or frequency <= 0:
            raise ValueError(f"{label} has an invalid Poisson histogram frequency: {raw_count!r}:{frequency!r}")
        recorded[int(raw_count)] += frequency
    numel = tensor.get("numel")
    if type(numel) is not int or numel <= 0:
        raise ValueError(f"{label} has invalid Poisson tensor numel")
    unique_count = tensor.get("unique_count")
    if type(unique_count) is not int or not 1 <= unique_count <= numel:
        raise ValueError(f"{label} has invalid Poisson unique_count")
    recorded_bins = histogram["recorded_bins"]
    omitted_bins = histogram["omitted_bins"]
    omitted_frequency = histogram["omitted_frequency"]
    truncated = histogram["truncated"]
    if (
        type(recorded_bins) is not int
        or recorded_bins != len(recorded)
        or recorded_bins != min(unique_count, MAX_COUNT_HISTOGRAM_BINS)
        or type(omitted_bins) is not int
        or omitted_bins != unique_count - recorded_bins
        or type(omitted_frequency) is not int
        or omitted_frequency < omitted_bins
        or type(truncated) is not bool
        or truncated != (omitted_bins > 0)
        or sum(recorded.values()) + omitted_frequency != numel
    ):
        raise ValueError(f"{label} has inconsistent bounded Poisson histogram metadata")

    minimum = _finite_float(tensor.get("minimum"))
    maximum = _finite_float(tensor.get("maximum"))
    omitted_minimum = histogram["omitted_value_minimum"]
    omitted_maximum = histogram["omitted_value_maximum"]
    if minimum is None or maximum is None or minimum != min(recorded):
        raise ValueError(f"{label} Poisson histogram minimum does not match tensor evidence")
    if truncated:
        if (
            type(omitted_minimum) is not int
            or type(omitted_maximum) is not int
            or omitted_minimum <= max(recorded)
            or omitted_maximum < omitted_minimum
            or omitted_maximum - omitted_minimum + 1 < omitted_bins
            or maximum != omitted_maximum
        ):
            raise ValueError(f"{label} has inconsistent omitted Poisson histogram range")
    elif omitted_minimum is not None or omitted_maximum is not None or maximum != max(recorded):
        raise ValueError(f"{label} has inconsistent complete Poisson histogram extrema")
    return {
        "recorded": recorded,
        "recorded_frequency": sum(recorded.values()),
        "omitted_bins": omitted_bins,
        "omitted_frequency": omitted_frequency,
        "omitted_value_minimum": omitted_minimum,
        "omitted_value_maximum": omitted_maximum,
        "truncated": truncated,
        "unique_count": unique_count,
    }


def _validated_poisson_rate_summary(tensor: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    """Validate the fixed softplus-rate summary for one Poisson tensor."""

    summary = _require_exact_mapping(
        tensor.get("rate_summary"),
        POISSON_RATE_FIELDS,
        label=f"{label}.rate_summary",
    )
    if summary["mapping"] != POISSON_RATE_MAPPING:
        raise ValueError(f"{label} has an unexpected Poisson rate mapping")
    if summary["histogram_schema_version"] != POISSON_RATE_HISTOGRAM_VERSION:
        raise ValueError(f"{label} has an unexpected Poisson rate histogram schema")
    minimum = _finite_float(summary["minimum"])
    maximum = _finite_float(summary["maximum"])
    mean = _finite_float(summary["mean"])
    if (
        minimum is None
        or maximum is None
        or mean is None
        or minimum < POISSON_RATE_BINS[0][0]
        or maximum > POISSON_RATE_BINS[-1][1]
        or not minimum <= mean <= maximum
    ):
        raise ValueError(f"{label} has invalid Poisson rate extrema or mean")

    histogram = summary["histogram"]
    if not isinstance(histogram, list) or len(histogram) != len(POISSON_RATE_BINS):
        raise ValueError(f"{label} has an invalid Poisson rate histogram length")
    frequencies: list[int] = []
    for bin_index, (item, expected_bin) in enumerate(zip(histogram, POISSON_RATE_BINS, strict=True)):
        item = _require_exact_mapping(
            item,
            POISSON_RATE_HISTOGRAM_FIELDS,
            label=f"{label}.rate_summary.histogram[{bin_index}]",
        )
        lower, upper, upper_inclusive = expected_bin
        if (
            _finite_float(item["lower"]) != lower
            or _finite_float(item["upper"]) != upper
            or item["lower_inclusive"] is not True
            or item["upper_inclusive"] is not upper_inclusive
            or type(item["frequency"]) is not int
            or item["frequency"] < 0
        ):
            raise ValueError(f"{label} has invalid Poisson rate histogram bin {bin_index}")
        frequencies.append(item["frequency"])

    numel = tensor.get("numel")
    total_frequency = summary["histogram_total_frequency"]
    if type(total_frequency) is not int or total_frequency != numel or sum(frequencies) != total_frequency:
        raise ValueError(f"{label} has inconsistent Poisson rate histogram frequency")

    def matching_bin(value: float) -> int | None:
        for bin_index, (lower, upper, upper_inclusive) in enumerate(POISSON_RATE_BINS):
            if value >= lower and (value <= upper if upper_inclusive else value < upper):
                return bin_index
        return None

    occupied = [index for index, frequency in enumerate(frequencies) if frequency]
    if not occupied or matching_bin(minimum) != occupied[0] or matching_bin(maximum) != occupied[-1]:
        raise ValueError(f"{label} Poisson rate histogram does not match its extrema")
    return {
        "minimum": minimum,
        "maximum": maximum,
        "mean": mean,
        "frequencies": frequencies,
        "total_frequency": total_frequency,
    }


def _verify_liveness_tensor_record(
    value: Any,
    *,
    family: str,
    label: str,
) -> dict[str, Any]:
    """Verify the common tensor schema and the assigned family's support."""

    tensor = _require_exact_mapping(value, LIVENESS_TENSOR_FIELDS, label=label)
    path = tensor["path"]
    if not isinstance(path, str) or not path:
        raise ValueError(f"{label}.path must be a non-empty string")
    shape = tensor["shape"]
    if not isinstance(shape, list) or any(type(dimension) is not int or dimension < 0 for dimension in shape):
        raise ValueError(f"{label}.shape must contain non-negative integers")
    numel = tensor["numel"]
    if type(numel) is not int or numel <= 0 or math.prod(shape) != numel:
        raise ValueError(f"{label}.numel does not match its positive shape product")
    dtype = tensor["dtype"]
    if dtype not in value_solver.MAX_CONSECUTIVE_INTEGER_BY_DTYPE:
        raise ValueError(f"{label}.dtype is not an authorized real floating dtype")
    stride = tensor["stride"]
    if (
        not isinstance(stride, list)
        or len(stride) != len(shape)
        or any(type(element) is not int or element < 0 for element in stride)
    ):
        raise ValueError(f"{label}.stride is invalid for its shape")
    storage_offset = tensor["storage_offset"]
    if type(storage_offset) is not int or storage_offset < 0:
        raise ValueError(f"{label}.storage_offset must be a non-negative integer")
    if type(tensor["is_pinned"]) is not bool:
        raise ValueError(f"{label}.is_pinned must be boolean")
    if type(tensor["parent_finite_elements"]) is not int or tensor["parent_finite_elements"] != numel:
        raise ValueError(f"{label} parent tensor was not completely finite")
    minimum = _finite_float(tensor["minimum"])
    maximum = _finite_float(tensor["maximum"])
    if minimum is None or maximum is None or minimum > maximum:
        raise ValueError(f"{label} has invalid finite extrema")

    histogram: dict[str, Any] | None = None
    rate_summary: dict[str, Any] | None = None
    if family == "uniform_01":
        if minimum < 0.0 or maximum >= 1.0:
            raise ValueError(f"{label} violates uniform_01 support")
    elif family == "signed_uniform":
        if minimum < -1.0 or maximum >= 1.0:
            raise ValueError(f"{label} violates signed_uniform support")
    elif family == "poisson_counts":
        if minimum < 0.0 or not minimum.is_integer() or not maximum.is_integer():
            raise ValueError(f"{label} violates poisson_counts support")
        histogram = _validated_poisson_histogram(tensor, label=label)
        rate_summary = _validated_poisson_rate_summary(tensor, label=label)
    elif family == "multinomial_categories":
        if not shape or shape[-1] < 2:
            raise ValueError(f"{label} lacks a valid multinomial category axis")
        maximum_exact = value_solver.MAX_CONSECUTIVE_INTEGER_BY_DTYPE[dtype]
        if shape[-1] - 1 > maximum_exact:
            raise ValueError(f"{label} multinomial categories are not exactly representable in {dtype}")
        if minimum < 0.0 or maximum >= shape[-1] or not minimum.is_integer() or not maximum.is_integer():
            raise ValueError(f"{label} violates multinomial_categories support")
    else:
        raise ValueError(f"{label} has an unknown assigned family: {family}")

    if family != "poisson_counts":
        if tensor["unique_count"] is not None or tensor["count_histogram"] is not None:
            raise ValueError(f"{label} has Poisson count evidence for a non-Poisson family")
        if tensor["rate_summary"] is not None:
            raise ValueError(f"{label} has Poisson rate evidence for a non-Poisson family")
    return {
        "schema": {
            "path": path,
            "shape": list(shape),
            "dtype": dtype,
            "stride": list(stride),
            "storage_offset": storage_offset,
            "is_pinned": tensor["is_pinned"],
            "numel": numel,
            "parent_finite_elements": tensor["parent_finite_elements"],
        },
        "minimum": minimum,
        "maximum": maximum,
        "histogram": histogram,
        "rate_summary": rate_summary,
    }


def _verify_liveness_trial(
    value: Any,
    *,
    trial_index: int,
    validation_config: Mapping[str, Any],
    expected_changed: int,
    family: str,
    label: str,
) -> list[dict[str, Any]]:
    """Verify trial identity, output activity, and every changed tensor."""

    trial = _require_exact_mapping(value, LIVENESS_TRIAL_FIELDS, label=label)
    if trial["trial"] != trial_index or type(trial["trial"]) is not int:
        raise ValueError(f"{label} has an invalid trial index")
    expected_seed = validation_config["seed"] + 10_007 * trial_index
    if trial["seed"] != expected_seed or type(trial["seed"]) is not int:
        raise ValueError(f"{label} has an invalid trial seed")
    if type(trial["changed_tensors"]) is not int or trial["changed_tensors"] != expected_changed:
        raise ValueError(f"{label} has an invalid changed tensor count")
    if trial["output_changed"] is not True:
        raise ValueError(f"{label} did not change the model output")
    changed_elements = trial["changed_output_elements"]
    output_elements = trial["output_elements"]
    changed_fraction = _finite_float(trial["max_changed_output_fraction"])
    if (
        type(changed_elements) is not int
        or type(output_elements) is not int
        or not 0 < changed_elements <= output_elements
        or changed_fraction is None
        or not 0.0 < changed_fraction <= 1.0
        or changed_fraction + 1e-15 < changed_elements / output_elements
    ):
        raise ValueError(f"{label} has inconsistent non-empty output activity")
    tensors = trial["tensors"]
    if not isinstance(tensors, list) or len(tensors) != expected_changed:
        raise ValueError(f"{label} has an invalid changed tensor evidence count")
    verified = [
        _verify_liveness_tensor_record(
            tensor,
            family=family,
            label=f"{label}.tensors[{tensor_index}]",
        )
        for tensor_index, tensor in enumerate(tensors)
    ]
    paths = [item["schema"]["path"] for item in verified]
    if len(paths) != len(set(paths)):
        raise ValueError(f"{label} repeats a changed tensor path")
    return verified


def _verify_liveness(
    liveness_dir: Path,
    both_pass: set[str],
    manifests: list[dict[str, Any]],
    *,
    parents_path: Path,
    children_path: Path,
    manifest_path: Path,
    allowlist_path: Path,
) -> tuple[dict[str, Any], set[str], dict[str, dict[str, Any]]]:
    shard_count, raw = _shard_records(liveness_dir)
    if len(raw) != len(both_pass):
        raise ValueError(f"liveness row count mismatch: expected {len(both_pass)}, found {len(raw)}")
    archived_validator_sha256 = _sha256_file(liveness_dir / "validator_source.py")
    archived_launcher_sha256 = _sha256_file(liveness_dir / "launcher_source.sh")
    current_validator_sha256 = _sha256_file(Path(__file__).with_name("validate_value_liveness.py"))
    current_launcher_sha256 = _sha256_file(Path(__file__).with_name("launch_value_liveness_shards.sh"))
    if archived_validator_sha256 != current_validator_sha256 or archived_launcher_sha256 != current_launcher_sha256:
        raise ValueError("archived liveness sources differ from the current frozen sources")
    manifest_index = {manifest["child_uuid"]: manifest for manifest in manifests}
    selected_order = [manifest["child_uuid"] for manifest in manifests if manifest["child_uuid"] in both_pass]
    selected_position = {uuid: index for index, uuid in enumerate(selected_order)}
    expected_artifacts = {
        "parents_sha256": _sha256_file(parents_path),
        "children_sha256": _sha256_file(children_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "allowlist_sha256": _sha256_file(allowlist_path),
    }
    by_uuid: dict[str, dict[str, Any]] = {}
    bindings: set[str] = set()
    launchers: set[str] = set()
    validators: set[str] = set()
    runtime_partitions: set[str] = set()
    passed_gpu_mappings: dict[str, dict[str, Any]] = {}
    counts: collections.Counter[str] = collections.Counter()
    family_statuses: collections.Counter[tuple[str, str]] = collections.Counter()
    passed_families: collections.Counter[str] = collections.Counter()
    passed_trial_counts: collections.Counter[str] = collections.Counter()
    passed_tensor_observation_counts: collections.Counter[str] = collections.Counter()
    poisson_count_histogram: collections.Counter[int] = collections.Counter()
    poisson_recorded_frequency = 0
    poisson_omitted_bins = 0
    poisson_omitted_frequency = 0
    poisson_omitted_value_minimum: int | None = None
    poisson_omitted_value_maximum: int | None = None
    poisson_truncated_tensor_observations = 0
    poisson_unique_count_sum = 0
    poisson_unique_count_minimum: int | None = None
    poisson_unique_count_maximum: int | None = None
    poisson_observed_min: float | None = None
    poisson_observed_max: float | None = None
    poisson_rate_histogram = [0] * len(POISSON_RATE_BINS)
    poisson_rate_total_frequency = 0
    poisson_rate_weighted_sum = 0.0
    poisson_rate_observed_min: float | None = None
    poisson_rate_observed_max: float | None = None
    multinomial_cardinalities: collections.Counter[int] = collections.Counter()
    reasons: collections.Counter[str] = collections.Counter()

    for shard_index, line_number, record in raw:
        uuid = record.get("child_uuid")
        if uuid not in both_pass or uuid in by_uuid:
            raise ValueError(f"invalid liveness UUID at shard {shard_index}:{line_number}: {uuid}")
        if selected_position[uuid] % shard_count != shard_index:
            raise ValueError(f"liveness UUID assigned to wrong shard: {uuid}:{shard_index}")
        validation_config, runtime_partition = _verify_liveness_policy(record, label=f"liveness record {uuid}")
        if runtime_partition is not None:
            runtime_partitions.add(runtime_partition)
        if record.get("validator_source_sha256") != archived_validator_sha256:
            raise ValueError(f"archived validator source hash mismatch for {uuid}")
        if record.get("launcher_source_sha256") != archived_launcher_sha256:
            raise ValueError(f"archived launcher source hash mismatch for {uuid}")
        for field, expected in expected_artifacts.items():
            if record.get(field) != expected:
                raise ValueError(f"liveness artifact binding mismatch for {uuid}:{field}")
        manifest = manifest_index[uuid]
        family = manifest["assigned_target"]
        expected_changed = manifest["transformed_factory_count"]
        if family not in EXPECTED_VALUE_FAMILIES:
            raise ValueError(f"liveness manifest has an unknown family for {uuid}: {family}")
        if type(expected_changed) is not int or expected_changed <= 0:
            raise ValueError(f"liveness manifest has an invalid transformed factory count for {uuid}")
        if record.get("parent_uuid") != manifest["parent_uuid"]:
            raise ValueError(f"liveness parent mismatch for {uuid}")
        if record.get("assigned_family") != family:
            raise ValueError(f"liveness family mismatch for {uuid}")
        if record.get("transformed_factory_count") != expected_changed:
            raise ValueError(f"liveness factory count mismatch for {uuid}")

        if record["passed"]:
            if record.get("status") != "passed" or record.get("reason") is not None:
                raise ValueError(f"passed liveness status/reason mismatch for {uuid}")
            memory_guard = record.get("memory_guard")
            if not isinstance(memory_guard, Mapping) or memory_guard.get("within_limit") is not True:
                raise ValueError(f"passed liveness memory guard failed for {uuid}")
            common_gpu = _verify_gpu_evidence(
                record.get("gpu"),
                label=f"liveness record {uuid}.gpu",
                expected_device=validation_config["device"],
                require_cudnn=True,
            )
            passed_gpu_mappings[_canonical_sha256(common_gpu)] = common_gpu
            configured_trials = validation_config["trials"]
            trial_records = record.get("trials")
            if not isinstance(trial_records, list) or len(trial_records) != configured_trials:
                raise ValueError(f"passed liveness trial count mismatch for {uuid}")
            output_effects = record.get("output_effects")
            if (
                not isinstance(output_effects, list)
                or len(output_effects) != configured_trials
                or any(effect is not True for effect in output_effects)
            ):
                raise ValueError(f"passed liveness output effects mismatch for {uuid}")

            verified_trials: list[list[dict[str, Any]]] = []
            frozen_tensor_schema: list[dict[str, Any]] | None = None
            for trial_index, trial in enumerate(trial_records):
                verified_tensors = _verify_liveness_trial(
                    trial,
                    trial_index=trial_index,
                    validation_config=validation_config,
                    expected_changed=expected_changed,
                    family=family,
                    label=f"liveness record {uuid}.trials[{trial_index}]",
                )
                tensor_schema = [item["schema"] for item in verified_tensors]
                if frozen_tensor_schema is None:
                    frozen_tensor_schema = tensor_schema
                elif tensor_schema != frozen_tensor_schema:
                    raise ValueError(f"liveness tensor schema changed across trials for {uuid}")
                verified_trials.append(verified_tensors)

            passed_families[family] += 1
            passed_trial_counts[family] += len(verified_trials)
            for verified_tensors in verified_trials:
                for tensor in verified_tensors:
                    passed_tensor_observation_counts[family] += 1
                    if family == "poisson_counts":
                        histogram = tensor["histogram"]
                        rate_summary = tensor["rate_summary"]
                        assert histogram is not None and rate_summary is not None
                        poisson_count_histogram.update(histogram["recorded"])
                        poisson_recorded_frequency += histogram["recorded_frequency"]
                        poisson_omitted_bins += histogram["omitted_bins"]
                        poisson_omitted_frequency += histogram["omitted_frequency"]
                        poisson_unique_count_sum += histogram["unique_count"]
                        poisson_unique_count_minimum = (
                            histogram["unique_count"]
                            if poisson_unique_count_minimum is None
                            else min(poisson_unique_count_minimum, histogram["unique_count"])
                        )
                        poisson_unique_count_maximum = (
                            histogram["unique_count"]
                            if poisson_unique_count_maximum is None
                            else max(poisson_unique_count_maximum, histogram["unique_count"])
                        )
                        if histogram["truncated"]:
                            poisson_truncated_tensor_observations += 1
                            poisson_omitted_value_minimum = (
                                histogram["omitted_value_minimum"]
                                if poisson_omitted_value_minimum is None
                                else min(poisson_omitted_value_minimum, histogram["omitted_value_minimum"])
                            )
                            poisson_omitted_value_maximum = (
                                histogram["omitted_value_maximum"]
                                if poisson_omitted_value_maximum is None
                                else max(poisson_omitted_value_maximum, histogram["omitted_value_maximum"])
                            )
                        poisson_observed_min = (
                            tensor["minimum"]
                            if poisson_observed_min is None
                            else min(poisson_observed_min, tensor["minimum"])
                        )
                        poisson_observed_max = (
                            tensor["maximum"]
                            if poisson_observed_max is None
                            else max(poisson_observed_max, tensor["maximum"])
                        )
                        poisson_rate_observed_min = (
                            rate_summary["minimum"]
                            if poisson_rate_observed_min is None
                            else min(poisson_rate_observed_min, rate_summary["minimum"])
                        )
                        poisson_rate_observed_max = (
                            rate_summary["maximum"]
                            if poisson_rate_observed_max is None
                            else max(poisson_rate_observed_max, rate_summary["maximum"])
                        )
                        poisson_rate_total_frequency += rate_summary["total_frequency"]
                        poisson_rate_weighted_sum += rate_summary["mean"] * rate_summary["total_frequency"]
                        for bin_index, frequency in enumerate(rate_summary["frequencies"]):
                            poisson_rate_histogram[bin_index] += frequency
                    elif family == "multinomial_categories":
                        multinomial_cardinalities[tensor["schema"]["shape"][-1]] += 1
        elif record.get("status") == "passed":
            raise ValueError(f"failed liveness record has passed status for {uuid}")

        by_uuid[uuid] = record
        bindings.add(record["validation_binding_sha256"])
        launchers.add(record["launcher_source_sha256"])
        validators.add(record["validator_source_sha256"])
        status = str(record.get("status"))
        counts[status] += 1
        family_statuses[(family, status)] += 1
        if record.get("passed") is not True:
            reason = str(record.get("reason"))
            error = record.get("error")
            if isinstance(error, str) and error:
                reason = f"{reason}:{error}"
            reasons[reason] += 1

    if set(by_uuid) != both_pass:
        raise ValueError("liveness UUID set differs from reference both-pass allowlist")
    if any(len(values) != 1 for values in (bindings, launchers, validators)):
        raise ValueError("liveness evidence mixes bindings or source hashes")
    passed = {uuid for uuid, record in by_uuid.items() if record["passed"]}
    if len(runtime_partitions) > 1 or (passed and len(runtime_partitions) != 1):
        raise ValueError("liveness evidence mixes GPU/runtime fingerprint partitions")
    if len(passed_gpu_mappings) > 1 or (passed and len(passed_gpu_mappings) != 1):
        raise ValueError("passed liveness evidence mixes common GPU fields")
    liveness_gpu = next(iter(passed_gpu_mappings.values()), None)
    first = next(iter(by_uuid.values()), {})
    rate_histogram_summary = [
        {
            "lower": lower,
            "upper": upper,
            "lower_inclusive": True,
            "upper_inclusive": upper_inclusive,
            "frequency": poisson_rate_histogram[index],
        }
        for index, (lower, upper, upper_inclusive) in enumerate(POISSON_RATE_BINS)
    ]
    summary = {
        "contract_version": LIVENESS_CONTRACT,
        "shard_count": shard_count,
        "selected": len(by_uuid),
        "passed": len(passed),
        "failed": len(by_uuid) - len(passed),
        "status_counts": dict(sorted(counts.items())),
        "family_status_counts": _nested_counts(family_statuses),
        "passed_family_counts": {family: passed_families.get(family, 0) for family in sorted(EXPECTED_VALUE_FAMILIES)},
        "passed_trial_observations": {
            "poisson_counts": {
                "passed_records": passed_families.get("poisson_counts", 0),
                "raw_trials": passed_trial_counts.get("poisson_counts", 0),
                "tensor_observations": passed_tensor_observation_counts.get("poisson_counts", 0),
                "observed_min": poisson_observed_min,
                "observed_max": poisson_observed_max,
                "recorded_value_frequencies": {
                    str(count): frequency for count, frequency in sorted(poisson_count_histogram.items())
                },
                "recorded_frequency": poisson_recorded_frequency,
                "omitted_bins": poisson_omitted_bins,
                "omitted_frequency": poisson_omitted_frequency,
                "omitted_value_minimum": poisson_omitted_value_minimum,
                "omitted_value_maximum": poisson_omitted_value_maximum,
                "truncated_tensor_observations": poisson_truncated_tensor_observations,
                "unique_count_sum": poisson_unique_count_sum,
                "unique_count_minimum": poisson_unique_count_minimum,
                "unique_count_maximum": poisson_unique_count_maximum,
                "rate_summary": {
                    "mapping": POISSON_RATE_MAPPING,
                    "histogram_schema_version": POISSON_RATE_HISTOGRAM_VERSION,
                    "observed_minimum": poisson_rate_observed_min,
                    "observed_maximum": poisson_rate_observed_max,
                    "weighted_mean": (
                        poisson_rate_weighted_sum / poisson_rate_total_frequency
                        if poisson_rate_total_frequency
                        else None
                    ),
                    "histogram": rate_histogram_summary,
                    "histogram_total_frequency": poisson_rate_total_frequency,
                },
            },
            "multinomial_categories": {
                "passed_records": passed_families.get("multinomial_categories", 0),
                "raw_trials": passed_trial_counts.get("multinomial_categories", 0),
                "tensor_observations": passed_tensor_observation_counts.get("multinomial_categories", 0),
                "last_dimension_cardinality_counts": {
                    str(cardinality): count for cardinality, count in sorted(multinomial_cardinalities.items())
                },
            },
        },
        "failure_reasons": dict(sorted(reasons.items())),
        "validation_binding_sha256": next(iter(bindings)),
        "launcher_source_sha256": next(iter(launchers)),
        "validator_source_sha256": next(iter(validators)),
        "validation_config": first.get("validation_config"),
        "runtime_partition_fingerprint": next(iter(runtime_partitions), None),
        "gpu": liveness_gpu,
        "gpu_fingerprints": sorted(
            {_canonical_sha256(record.get("gpu")) for record in by_uuid.values() if record.get("gpu")}
        ),
    }
    return summary, passed, by_uuid


def _materialize_accepted_table(
    children_table: pa.Table,
    accepted_indices: Sequence[int],
    *,
    runtime_policy_fingerprint: str,
) -> pa.Table:
    """Update existing status fields and schema metadata without widening the row schema."""

    _require_sha256(runtime_policy_fingerprint, label="runtime_policy_fingerprint")
    extra_info_index = children_table.schema.get_field_index("extra_info")
    if extra_info_index < 0 or not pa.types.is_struct(children_table.schema.field(extra_info_index).type):
        raise ValueError("candidate schema lacks the extra_info struct")
    extra_info_type = children_table.schema.field(extra_info_index).type
    for struct_name, required_fields in (
        ("v4", {"runtime_validation_status", "governance_status", "included_in_review_train"}),
        ("augmentation", {"validation_status"}),
    ):
        struct_index = extra_info_type.get_field_index(struct_name)
        if struct_index < 0 or not pa.types.is_struct(extra_info_type.field(struct_index).type):
            raise ValueError(f"candidate schema lacks extra_info.{struct_name}")
        struct_type = extra_info_type.field(struct_index).type
        missing = sorted(field for field in required_fields if struct_type.get_field_index(field) < 0)
        if missing:
            raise ValueError(f"candidate schema lacks extra_info.{struct_name} fields: {missing}")

    accepted_table = children_table.take(pa.array(list(accepted_indices), type=pa.int64()))
    accepted_rows = accepted_table.to_pylist()
    for row_index, row in enumerate(accepted_rows):
        extra_info = row.get("extra_info")
        if not isinstance(extra_info, dict):
            raise ValueError(f"accepted row lacks extra_info: {row_index}")
        v4 = extra_info.get("v4")
        augmentation = extra_info.get("augmentation")
        if not isinstance(v4, dict) or not isinstance(augmentation, dict):
            raise ValueError(f"accepted row lacks runtime status structs: {row_index}")
        v4["runtime_validation_status"] = ACCEPTED_RUNTIME_STATUS
        v4["governance_status"] = ACCEPTED_GOVERNANCE_STATUS
        v4["included_in_review_train"] = False
        augmentation["validation_status"] = ACCEPTED_GOVERNANCE_STATUS

    metadata = dict(children_table.schema.metadata or {})
    metadata.update(
        {
            b"random_value.analysis_contract": ANALYSIS_CONTRACT.encode(),
            b"random_value.runtime_policy_fingerprint": runtime_policy_fingerprint.encode(),
            b"random_value.runtime_validation_status": ACCEPTED_RUNTIME_STATUS.encode(),
            b"random_value.governance_status": ACCEPTED_GOVERNANCE_STATUS.encode(),
            b"random_value.parent_runtime_status": b"passed",
            b"random_value.child_runtime_status": b"passed",
            b"random_value.liveness_status": b"passed",
            b"random_value.materialization_status": ACCEPTED_GOVERNANCE_STATUS.encode(),
            b"random_value.training_approved": b"false",
        }
    )
    accepted_schema = children_table.schema.with_metadata(metadata)
    return pa.Table.from_pylist(accepted_rows, schema=accepted_schema)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_raw_artifact_manifest(lane_dir: Path, roots: Sequence[Path]) -> Path:
    resolved_lane = lane_dir.resolve()
    files: dict[str, str] = {}
    root_names: list[str] = []
    for root in roots:
        resolved_root = root.resolve()
        try:
            relative_root = resolved_root.relative_to(resolved_lane)
        except ValueError as exc:
            raise ValueError(f"raw evidence root is outside lane directory: {resolved_root}") from exc
        root_names.append(str(relative_root))
        for path in sorted(resolved_root.rglob("*")):
            if not path.is_file() or path.name.startswith("."):
                continue
            relative = str(path.relative_to(resolved_lane))
            if relative in files:
                raise ValueError(f"duplicate raw evidence path: {relative}")
            files[relative] = _sha256_file(path)
    payload = {
        "contract_version": "random_value_raw_artifact_manifest_v2",
        "roots": root_names,
        "file_count": len(files),
        "files": files,
    }
    output = lane_dir / "runtime/raw_artifact_sha256.json"
    _write_json(output, payload)
    return output


def _write_report(
    path: Path,
    reference: Mapping[str, Any],
    liveness: Mapping[str, Any] | None,
    manifests: list[dict[str, Any]],
    accepted: set[str] | None,
) -> None:
    lines = [
        "# Random/value runtime audit",
        "",
        f"- Candidate pairs: {reference['candidate_pairs']}",
        f"- Reference both-pass: {reference['both_pass_children']}",
        f"- Reference fingerprint: `{reference['contract_fingerprint']}`",
        f"- GPU: `{reference.get('gpu')}`",
        "",
        "## Paired reference classification",
        "",
        "| Classification | Count |",
        "| --- | ---: |",
    ]
    lines.extend(f"| {key} | {value} |" for key, value in reference["classifications"].items())
    lines.extend(
        [
            "",
            "## Paired reference by method",
            "",
            "| Method | Classification | Count |",
            "| --- | --- | ---: |",
        ]
    )
    for family, family_counts in reference["family_classification_counts"].items():
        lines.extend(f"| {family} | {classification} | {count} |" for classification, count in family_counts.items())
    lines.extend(["", "## Failure attribution", "", "| Attribution | Count |", "| --- | ---: |"])
    lines.extend(f"| {key} | {value} |" for key, value in reference["failure_attribution"].items())
    if liveness is not None and accepted is not None:
        lines.extend(
            [
                "",
                "## Value liveness",
                "",
                f"- Selected after paired reference: {liveness['selected']}",
                f"- Passed: {liveness['passed']}",
                f"- Failed: {liveness['failed']}",
                "",
                "### Liveness by method and status",
                "",
                "| Method | Status | Count |",
                "| --- | --- | ---: |",
            ]
        )
        for family, family_counts in liveness["family_status_counts"].items():
            lines.extend(f"| {family} | {status} | {count} |" for status, count in family_counts.items())
        lines.extend(
            [
                "",
                "### Passed methods",
                "",
                "| Method | Passed |",
                "| --- | ---: |",
            ]
        )
        lines.extend(f"| {family} | {count} |" for family, count in liveness["passed_family_counts"].items())

        observations = liveness["passed_trial_observations"]
        poisson = observations["poisson_counts"]
        lines.extend(
            [
                "",
                "### Passed Poisson raw-trial observations",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                f"| Passed records | {poisson['passed_records']} |",
                f"| Raw trials | {poisson['raw_trials']} |",
                f"| Tensor observations | {poisson['tensor_observations']} |",
                f"| Observed minimum | {poisson['observed_min']} |",
                f"| Observed maximum | {poisson['observed_max']} |",
                f"| Unique count range per tensor | {poisson['unique_count_minimum']}–{poisson['unique_count_maximum']} |",
                f"| Recorded element frequency | {poisson['recorded_frequency']} |",
                f"| Omitted bins across truncated tensors | {poisson['omitted_bins']} |",
                f"| Omitted element frequency | {poisson['omitted_frequency']} |",
                f"| Omitted value range | {poisson['omitted_value_minimum']}–{poisson['omitted_value_maximum']} |",
                f"| Truncated tensor observations | {poisson['truncated_tensor_observations']} |",
                "",
                "| Recorded Poisson count | Observed elements |",
                "| ---: | ---: |",
            ]
        )
        lines.extend(
            f"| {count} | {frequency} |" for count, frequency in poisson["recorded_value_frequencies"].items()
        )
        multinomial = observations["multinomial_categories"]
        lines.extend(
            [
                "",
                "### Passed Multinomial raw-trial observations",
                "",
                f"- Passed records: {multinomial['passed_records']}",
                f"- Raw trials: {multinomial['raw_trials']}",
                f"- Tensor observations: {multinomial['tensor_observations']}",
                "",
                "| Last-dimension cardinality | Tensor observations |",
                "| ---: | ---: |",
            ]
        )
        lines.extend(
            f"| {cardinality} | {count} |"
            for cardinality, count in multinomial["last_dimension_cardinality_counts"].items()
        )
        lines.extend(
            [
                "",
                "## Accepted bias slices",
                "",
            ]
        )
        manifest_by_uuid = {manifest["child_uuid"]: manifest for manifest in manifests}
        for label, field in (
            ("family", "assigned_target"),
            ("source", "source_family"),
            ("operator bucket", "operator_bucket"),
        ):
            counter = collections.Counter(str(manifest_by_uuid[uuid].get(field)) for uuid in accepted)
            lines.append(f"- {label}: `{dict(sorted(counter.items()))}`")
        for label in ("support", "sign", "zero", "sparsity", "magnitude", "cardinality"):
            counter = collections.Counter(
                str(manifest_by_uuid[uuid].get("value_labels", {}).get(label)) for uuid in accepted
            )
            lines.append(f"- value {label}: `{dict(sorted(counter.items()))}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lane_dir", type=Path)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--liveness-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    parents_path = args.lane_dir / "parents.parquet"
    children_path = args.lane_dir / "candidates.parquet"
    paired_path = args.lane_dir / "paired.parquet"
    manifest_path = args.lane_dir / "manifest.jsonl"
    _require_candidate_count(pq.ParquetFile(children_path).metadata.num_rows)
    parents = pq.read_table(parents_path).to_pylist()
    children_table = pq.read_table(children_path)
    children = children_table.to_pylist()
    manifests = _read_manifest(manifest_path)
    source_context = _verify_aligned_static(parents, children, manifests)
    reference, both_pass, child_reference_records = _verify_reference(
        reference_dir=args.reference_dir,
        paired_path=paired_path,
        parents=parents,
        children=children,
        manifests=manifests,
    )
    analysis_dir = args.lane_dir / "analysis"
    _write_json(analysis_dir / "reference_summary.json", reference)
    allowlist_path = analysis_dir / "reference_both_pass_child_uuids.txt"
    allowlist_path.write_text("".join(f"{uuid}\n" for uuid in sorted(both_pass)), encoding="utf-8")

    liveness: dict[str, Any] | None = None
    accepted: set[str] | None = None
    if args.liveness_dir is not None:
        liveness, accepted, liveness_records = _verify_liveness(
            args.liveness_dir,
            both_pass,
            manifests,
            parents_path=parents_path,
            children_path=children_path,
            manifest_path=manifest_path,
            allowlist_path=allowlist_path,
        )
        if accepted and liveness["gpu"] != reference["gpu"]:
            raise ValueError("reference and passed liveness evidence use different GPUs")
        _write_json(analysis_dir / "liveness_summary.json", liveness)
        accepted_indices = [index for index, manifest in enumerate(manifests) if manifest["child_uuid"] in accepted]
        analyzer_source_sha256 = _sha256_file(Path(__file__).resolve())
        policy = {
            "analyzer_source_sha256": analyzer_source_sha256,
            "reference_contract_fingerprint": reference["contract_fingerprint"],
            "reference_launcher_source_sha256": reference["launcher_source_sha256"],
            "reference_validator_source_sha256": reference["validator_source_sha256"],
            "reference_runtime_environment_fingerprint": reference["runtime_environment_fingerprint"],
            "liveness_binding_sha256": liveness["validation_binding_sha256"],
            "liveness_launcher_source_sha256": liveness["launcher_source_sha256"],
            "liveness_validator_source_sha256": liveness["validator_source_sha256"],
            "liveness_runtime_partition_fingerprint": liveness["runtime_partition_fingerprint"],
            "liveness_gpu_fingerprints": liveness["gpu_fingerprints"],
        }
        runtime_policy_fingerprint = _canonical_sha256(policy)
        accepted_table = _materialize_accepted_table(
            children_table,
            accepted_indices,
            runtime_policy_fingerprint=runtime_policy_fingerprint,
        )
        accepted_path = args.lane_dir / "runtime/accepted.parquet"
        accepted_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(accepted_table, accepted_path)
        accepted_rows_by_uuid = {_nested(row, "extra_info.uuid"): row for row in accepted_table.to_pylist()}
        if set(accepted_rows_by_uuid) != accepted:
            raise ValueError("accepted parquet UUIDs differ from the liveness-passed set")
        accepted_manifest_path = args.lane_dir / "runtime/accepted.manifest.jsonl"
        with accepted_manifest_path.open("w", encoding="utf-8") as handle:
            for manifest in manifests:
                uuid = manifest["child_uuid"]
                if uuid not in accepted:
                    continue
                enriched = {
                    **manifest,
                    "candidate_row_sha256": manifest["row_sha256"],
                    "row_sha256": _sha256_bytes(_canonical_json_bytes(accepted_rows_by_uuid[uuid])),
                    "parent_runtime_status": "passed",
                    "child_runtime_status": "passed",
                    "liveness_status": "passed",
                    "materialization_status": ACCEPTED_GOVERNANCE_STATUS,
                    "runtime_policy_fingerprint": runtime_policy_fingerprint,
                    "runtime_evidence": {
                        "policy": policy,
                        "parent_reference_row": 2 * manifest["candidate_row_index"],
                        "child_reference_row": 2 * manifest["candidate_row_index"] + 1,
                        "child_reference_status": child_reference_records[uuid]["status"],
                        "liveness_status": liveness_records[uuid]["status"],
                    },
                    "training_approved": False,
                }
                handle.write(json.dumps(enriched, sort_keys=True, ensure_ascii=False) + "\n")
        raw_artifact_manifest = _write_raw_artifact_manifest(args.lane_dir, [args.reference_dir, args.liveness_dir])
        final_summary = {
            "contract_version": ANALYSIS_CONTRACT,
            "scope": (
                "shape_coverage_resample_5000_review_lane"
                if source_context.kind == "shape_coverage_resample"
                else "canonical_parent_canary"
            ),
            "maximum_authorized_candidates": 5_000,
            "candidate_rows": len(children),
            "source_binding": dict(source_context.binding),
            "reference": reference,
            "liveness": liveness,
            "accepted_rows": len(accepted_indices),
            "rejected_rows": len(children) - len(accepted_indices),
            "materialization_status": ACCEPTED_GOVERNANCE_STATUS,
            "training_approved": False,
            "runtime_policy_fingerprint": runtime_policy_fingerprint,
            "artifacts": {
                "accepted_parquet_sha256": _sha256_file(accepted_path),
                "accepted_manifest_sha256": _sha256_file(accepted_manifest_path),
                "reference_allowlist_sha256": _sha256_file(allowlist_path),
                "raw_artifact_manifest_sha256": _sha256_file(raw_artifact_manifest),
            },
            "analyzer_source_sha256": analyzer_source_sha256,
        }
        _write_json(args.lane_dir / "runtime/final_summary.json", final_summary)
    _write_report(analysis_dir / "failure_bias_report.md", reference, liveness, manifests, accepted)
    print(
        json.dumps(
            {
                "candidate_pairs": len(children),
                "reference_both_pass": len(both_pass),
                "liveness_passed": len(accepted) if accepted is not None else None,
                "analysis_dir": str(analysis_dir.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
