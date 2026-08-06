#!/usr/bin/env python3
"""Audit source-bound layout canary evidence and materialize review-only rows."""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import math
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from tools.data.synthesize.layout_method import solve_layout_coverage as solver
from tools.data.synthesize.layout_method import validate_layout_liveness as validator
from tools.data.synthesize.random_method import analyze_value_run as evidence_common

MAX_AUTHORIZED_CANDIDATES = 5_000
FAMILIES = frozenset(solver.LAYOUT_FAMILIES)
ANALYSIS_CONTRACT = "layout_lane_exact_analysis_v1"
ACCEPTED_RUNTIME_STATUS = "layout_paired_reference_and_raw_liveness_passed"
ACCEPTED_GOVERNANCE_STATUS = "layout_intervention_review_only"
METADATA_FIELDS = frozenset(
    {
        "shape",
        "dtype",
        "device",
        "layout",
        "stride",
        "storage_offset",
        "requires_grad",
        "is_contiguous",
        "is_pinned",
        "numel",
    }
)
CONFIG_FIELDS = frozenset(
    {"device", "trials", "seed", "timeout_seconds", "max_device_memory_gib", "raw_direct_invocation"}
)
TRACE_FIELDS = frozenset(
    {
        "target_storage_count",
        "target_inputs_immutable",
        "per_target_storage",
        "recorded_calls",
        "calls_truncated",
    }
)
PER_STORAGE_FIELDS = frozenset(
    {
        "storage",
        "semantic_consumer_count",
        "first_semantic_consumer",
        "materialization_before_consumer",
        "erase_before_semantic_calls",
        "erase_after_semantic_calls",
    }
)
ALIAS_FIELDS = frozenset({"alias_groups", "target_storage_count", "target_storages_independent"})
TRIAL_FIELDS = frozenset(
    {
        "ordinal",
        "seed",
        "input_layouts",
        "input_alias_contract",
        "factory_count",
        "transformed_factory_count",
        "raw_parent_child_output_exact",
        "raw_child_control_trace_output_exact",
        "semantic_dispatch",
    }
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _factory_coverage(manifests: Sequence[Mapping[str, Any]], child_uuids: set[str] | None = None) -> dict[str, Any]:
    selected = [row for row in manifests if child_uuids is None or str(row.get("child_uuid")) in child_uuids]

    def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        total = sum(int(row["factory_count"]) for row in rows)
        transformed = sum(int(row["transformed_factory_count"]) for row in rows)
        partial = sum(int(row["transformed_factory_count"]) < int(row["factory_count"]) for row in rows)
        return {
            "records": len(rows),
            "total_factory_leaves": total,
            "transformed_factory_leaves": transformed,
            "partial_records": partial,
            "full_records": len(rows) - partial,
        }

    return {
        **summarize(selected),
        "by_family": {
            family: summarize([row for row in selected if row.get("assigned_family") == family])
            for family in sorted(FAMILIES)
        },
    }


def _nested(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _require_mapping(value: Any, fields: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(f"{label} fields differ:{actual}:{sorted(fields)}")
    return dict(value)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not 1 <= len(rows) <= MAX_AUTHORIZED_CANDIDATES:
        raise ValueError(f"row count must be in [1,{MAX_AUTHORIZED_CANDIDATES}]:{len(rows)}")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("JSONL rows must be objects")
    return rows


def _normalize_source_parent(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(row))
    extra = normalized.get("extra_info")
    if not isinstance(extra, Mapping):
        raise ValueError(f"canonical parent lacks extra_info:{index}")
    extra = dict(extra)
    if extra.get("augmentation") not in (None,):
        raise ValueError(f"canonical parent unexpectedly augmented:{index}")
    extra["augmentation"] = None
    normalized["extra_info"] = extra
    return normalized


def _canonical_source(manifests: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    paths = {row.get("source_artifact_path") for row in manifests}
    hashes = {row.get("source_artifact_sha256") for row in manifests}
    if len(paths) != 1 or len(hashes) != 1:
        raise ValueError("manifests must bind exactly one canonical source")
    path_text, digest = next(iter(paths)), next(iter(hashes))
    if not isinstance(path_text, str) or not Path(path_text).is_absolute() or not isinstance(digest, str):
        raise ValueError("source binding is invalid")
    source = Path(path_text)
    if not source.is_file() or str(source.resolve()) != path_text:
        raise ValueError("source artifact unavailable or unresolved")
    if digest != solver.EXPECTED_CANONICAL_PARENT_SHA256 or _sha256_file(source) != digest:
        raise ValueError("canonical source SHA mismatch")
    parquet = pq.ParquetFile(source)
    if parquet.metadata.num_rows != solver.EXPECTED_CANONICAL_PARENT_ROWS:
        raise ValueError("canonical source row count mismatch")
    indices = {row.get("source_row_index") for row in manifests}
    if any(type(index) is not int or not 0 <= index < parquet.metadata.num_rows for index in indices):
        raise ValueError("invalid source row index")
    found: dict[int, dict[str, Any]] = {}
    offset = 0
    for batch in parquet.iter_batches(batch_size=256, use_threads=False):
        for local, row in enumerate(batch.to_pylist()):
            index = offset + local
            if index in indices:
                found[index] = _normalize_source_parent(row, index)
        offset += batch.num_rows
        if len(found) == len(indices):
            break
    if set(found) != indices:
        raise ValueError("failed to reload every bound canonical parent")
    return found


def _verify_static(
    parents: list[dict[str, Any]],
    children: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
    paired: list[dict[str, Any]],
) -> dict[str, Any]:
    if not (len(parents) == len(children) == len(manifests)):
        raise ValueError("static parent/child/manifest counts differ")
    if _canonical_bytes([row for pair in zip(parents, children, strict=True) for row in pair]) != _canonical_bytes(
        paired
    ):
        raise ValueError("paired parquet is not exact parent/child interleave")
    canonical = _canonical_source(manifests)
    generator_path = Path(solver.__file__).resolve()
    dependency_path = generator_path.parent.parent / "augment_prompt_tasks.py"
    source_paths = {
        "solver": generator_path,
        "dependency": dependency_path,
        "validator": Path(validator.__file__).resolve(),
        "launcher": Path(__file__).with_name("launch_layout_liveness_shards.sh").resolve(),
        "analyzer": Path(__file__).resolve(),
        "reference_analyzer": Path(evidence_common.__file__).resolve(),
    }
    source_sha256 = {label: _sha256_file(path) for label, path in source_paths.items()}
    generator_sha = source_sha256["solver"]
    dependency_sha = source_sha256["dependency"]
    commits = {row.get("git_commit") for row in manifests}
    if (
        len(commits) != 1
        or not isinstance(next(iter(commits)), str)
        or re.fullmatch(r"[0-9a-f]{40}", next(iter(commits))) is None
    ):
        raise ValueError("manifests do not bind one valid Git commit")
    commit = str(next(iter(commits)))
    for label, path in source_paths.items():
        if solver._git_blob_sha256(commit, path) != source_sha256[label]:
            raise ValueError(f"current {label} source differs from manifest Git commit")
    seen: dict[str, set[str]] = {
        key: set() for key in ("parent_uuid", "child_uuid", "child_reference_sha256", "child_normalized_ast_sha256")
    }
    for index, (parent, child, manifest) in enumerate(zip(parents, children, manifests, strict=True)):
        if (
            manifest.get("manifest_contract_version") != "layout_lane_manifest_v1"
            or manifest.get("candidate_row_index") != index
        ):
            raise ValueError(f"manifest contract/index mismatch:{index}")
        family = manifest.get("assigned_family")
        if manifest.get("primary_intervention") != "layout" or family not in FAMILIES:
            raise ValueError(f"layout family mismatch:{index}")
        realized = manifest.get("realized_intervention")
        expected_layout = manifest.get("expected_layout_metadata")
        total_count = manifest.get("factory_count")
        transformed_count = manifest.get("transformed_factory_count")
        target_indices = manifest.get("target_factory_indices")
        scope = manifest.get("layout_application_scope")
        if (
            not isinstance(realized, Mapping)
            or realized.get("family") != family
            or not isinstance(expected_layout, list)
            or not expected_layout
            or type(total_count) is not int
            or type(transformed_count) is not int
            or not 0 < transformed_count <= total_count
            or len(expected_layout) != transformed_count
            or not isinstance(target_indices, list)
            or any(type(item) is not int for item in target_indices)
            or target_indices != sorted(set(target_indices))
            or any(not 0 <= item < total_count for item in target_indices)
            or [item.get("factory_index") for item in expected_layout] != target_indices
            or realized.get("factory_count") != total_count
            or realized.get("transformed_factory_count") != transformed_count
            or realized.get("target_factory_indices") != target_indices
            or realized.get("layout_application_scope") != scope
            or _canonical_bytes(realized.get("expected_metadata")) != _canonical_bytes(expected_layout)
        ):
            raise ValueError(f"realized exact-layout metadata mismatch:{index}")
        if family == "expand_zero_stride":
            if scope != "eligible_constant_factory_leaves_only":
                raise ValueError(f"partial expand scope is not declared:{index}")
        elif scope != "all_direct_factory_leaves" or transformed_count != total_count:
            raise ValueError(f"non-expand family is not an all-leaf transform:{index}")
        if any(
            manifest.get(field) is not expected
            for field, expected in (
                ("shape_changed", False),
                ("value_changed", False),
                ("dtype_changed", False),
                ("layout_changed", True),
            )
        ):
            raise ValueError(f"intervention-axis declaration mismatch:{index}")
        if any(manifest.get(field) is not False for field in ("model_changed", "get_init_inputs_changed")):
            raise ValueError(f"frozen-section declaration mismatch:{index}")
        if manifest.get("static_status") != "passed" or manifest.get("training_approved") is not False:
            raise ValueError(f"static/training status mismatch:{index}")
        if (
            manifest.get("generator_contract_version") != solver.CONTRACT_VERSION
            or manifest.get("generator_source_sha256") != generator_sha
            or manifest.get("dependency_source_sha256") != dependency_sha
            or manifest.get("git_commit") != commit
        ):
            raise ValueError(f"generator source binding mismatch:{index}")
        expected_pending = {
            "parent_runtime_status": "pending",
            "child_runtime_status": "pending",
            "layout_realization_status": "pending",
            "liveness_status": "pending",
            "materialization_status": "review_only",
        }
        if any(manifest.get(key) != value for key, value in expected_pending.items()):
            raise ValueError(f"runtime pending status mismatch:{index}")
        parent_uuid, child_uuid = _nested(parent, "extra_info.uuid"), _nested(child, "extra_info.uuid")
        parent_code, child_code = _nested(parent, "reward_model.ground_truth"), _nested(
            child, "reward_model.ground_truth"
        )
        if not all(isinstance(value, str) and value for value in (parent_uuid, child_uuid, parent_code, child_code)):
            raise ValueError(f"invalid static identity:{index}")
        if manifest.get("parent_uuid") != parent_uuid or manifest.get("child_uuid") != child_uuid:
            raise ValueError(f"manifest UUID mismatch:{index}")
        if _sha256_bytes(parent_code.encode()) != manifest.get("parent_reference_sha256") or _sha256_bytes(
            child_code.encode()
        ) != manifest.get("child_reference_sha256"):
            raise ValueError(f"manifest reference SHA mismatch:{index}")
        source_index = manifest["source_row_index"]
        if _canonical_bytes(canonical[source_index]) != _canonical_bytes(parent):
            raise ValueError(f"parent differs from canonical source:{index}")
        eligible, reason = solver._analyze_parent(parent, source_index)
        if eligible is None or eligible.assigned_family != family:
            raise ValueError(f"current solver rejects/reassigns parent:{index}:{reason}")
        if (
            manifest.get("eligible_families") != list(eligible.eligible_families)
            or manifest.get("assignment_version") != solver.ASSIGNMENT_VERSION
            or manifest.get("assignment_sha256") != eligible.assignment_sha256
        ):
            raise ValueError(f"eligible-family assignment replay mismatch:{index}")
        replay_child, replay_manifest = solver._make_child(
            parent,
            eligible,
            Path(manifest["source_artifact_path"]),
            manifest["source_artifact_sha256"],
            generator_sha,
            dependency_sha,
            commit,
        )
        replay_manifest["candidate_row_index"] = index
        if _canonical_bytes(replay_child) != _canonical_bytes(child) or _canonical_bytes(
            replay_manifest
        ) != _canonical_bytes(manifest):
            raise ValueError(f"deterministic solver replay mismatch:{index}")
        for field, values in seen.items():
            value = manifest.get(field)
            if value in values:
                raise ValueError(f"duplicate static identifier:{field}:{value}")
            values.add(value)
    if any(len(values) != len(manifests) for values in seen.values()):
        raise ValueError("static uniqueness failure")
    return {"source_git_commit": commit, "source_sha256": source_sha256}


def _layout_metadata(record: Any, expected: Mapping[str, Any], family: str, label: str) -> None:
    mapping = _require_mapping(record, METADATA_FIELDS, label)
    for key in ("shape", "stride"):
        if not isinstance(mapping[key], list) or any(type(value) is not int for value in mapping[key]):
            raise ValueError(f"{label}.{key} invalid")
    required_expected = {
        "factory_index",
        "logical_shape",
        "dtype",
        "expected_strides",
        "expected_storage_offset",
        "expected_is_contiguous",
        "expected_zero_stride_dimensions",
    }
    if not required_expected.issubset(expected):
        raise ValueError(f"{label} lacks exact expected layout metadata")
    expected_dtype = str(expected["dtype"])
    if not expected_dtype.startswith("torch."):
        expected_dtype = f"torch.{expected_dtype}"
    if (
        mapping["shape"] != expected["logical_shape"]
        or mapping["dtype"] != expected_dtype
        or mapping["layout"] != "torch.strided"
        or mapping["device"] != "cuda:0"
        or mapping["requires_grad"] is not False
        or mapping["is_pinned"] is not False
        or mapping["numel"] != math.prod(mapping["shape"])
        or mapping["stride"] != expected["expected_strides"]
        or mapping["storage_offset"] != expected["expected_storage_offset"]
        or mapping["is_contiguous"] != expected["expected_is_contiguous"]
    ):
        raise ValueError(f"{label} semantic metadata mismatch")
    if not isinstance(mapping["storage_offset"], int) or not isinstance(mapping["is_contiguous"], bool):
        raise ValueError(f"{label} layout scalar invalid")
    zero_dims = [
        index
        for index, (size, stride) in enumerate(zip(mapping["shape"], mapping["stride"], strict=True))
        if size > 1 and stride == 0
    ]
    if family == "transpose_noncontiguous":
        if mapping["is_contiguous"] or mapping["storage_offset"] != 0 or zero_dims:
            raise ValueError(f"{label} transpose predicate failed")
    elif family == "slice_storage_offset":
        # Rank-1 and leading-singleton slices can remain PyTorch-contiguous
        # despite a nonzero storage offset.  Contiguity is already checked
        # against the solver's exact per-leaf metadata above.
        if mapping["storage_offset"] <= 0 or zero_dims:
            raise ValueError(f"{label} slice predicate failed")
    elif family == "expand_zero_stride":
        if (
            mapping["is_contiguous"]
            or mapping["storage_offset"] != 0
            or zero_dims != expected["expected_zero_stride_dimensions"]
        ):
            raise ValueError(f"{label} expand predicate failed")
    else:
        raise ValueError(f"unknown family:{family}")


def _verify_trial(value: Any, manifest: Mapping[str, Any], ordinal: int, config: Mapping[str, Any], label: str) -> int:
    trial = _require_mapping(value, TRIAL_FIELDS, label)
    if (
        trial["ordinal"] != ordinal
        or trial["seed"] != config["seed"] + 10_007 * ordinal
        or trial["raw_parent_child_output_exact"] is not True
        or trial["raw_child_control_trace_output_exact"] is not True
    ):
        raise ValueError(f"{label} control/output proof mismatch")
    expected = manifest["expected_layout_metadata"]
    if (
        trial["factory_count"] != manifest["factory_count"]
        or trial["transformed_factory_count"] != manifest["transformed_factory_count"]
        or trial["transformed_factory_count"] != len(expected)
        or not isinstance(trial["input_layouts"], list)
        or len(trial["input_layouts"]) != len(expected)
    ):
        raise ValueError(f"{label} transformed input count mismatch")
    expected_by_index = {int(item["factory_index"]): item for item in expected}
    target_indices = manifest["target_factory_indices"]
    if sorted(expected_by_index) != target_indices:
        raise ValueError(f"{label} expected metadata indices differ from manifest targets")
    seen_paths: set[str] = set()
    seen_factory_indices: set[int] = set()
    for index, item in enumerate(trial["input_layouts"]):
        item = _require_mapping(
            item,
            frozenset({"factory_index", "path", "parent", "child"}),
            f"{label}.input[{index}]",
        )
        factory_index = item["factory_index"]
        if (
            type(factory_index) is not int
            or factory_index not in expected_by_index
            or factory_index in seen_factory_indices
            or not isinstance(item["path"], str)
            or item["path"] in seen_paths
        ):
            raise ValueError(f"{label} input path/factory index invalid")
        seen_paths.add(item["path"])
        seen_factory_indices.add(factory_index)
        expected_item = expected_by_index[factory_index]
        parent = _require_mapping(item["parent"], METADATA_FIELDS, f"{label}.parent[{index}]")
        expected_dtype = str(expected_item["dtype"])
        if not expected_dtype.startswith("torch."):
            expected_dtype = f"torch.{expected_dtype}"
        if (
            not parent["is_contiguous"]
            or parent["storage_offset"] != 0
            or parent["shape"] != expected_item["logical_shape"]
            or parent["dtype"] != expected_dtype
        ):
            raise ValueError(f"{label} parent layout control mismatch")
        _layout_metadata(
            item["child"],
            expected_item,
            str(manifest["assigned_family"]),
            f"{label}.child[{index}]",
        )
    if sorted(seen_factory_indices) != target_indices:
        raise ValueError(f"{label} runtime factory indices differ from manifest targets")
    alias = _require_mapping(trial["input_alias_contract"], ALIAS_FIELDS, f"{label}.input_alias_contract")
    if (
        not isinstance(alias["alias_groups"], list)
        or alias["target_storage_count"] != len(expected)
        or alias["target_storages_independent"] is not True
    ):
        raise ValueError(f"{label} alias proof mismatch")
    trace = _require_mapping(trial["semantic_dispatch"], TRACE_FIELDS, f"{label}.semantic_dispatch")
    storages = trace["per_target_storage"]
    if (
        type(trace["target_storage_count"]) is not int
        or trace["target_storage_count"] != len(expected)
        or trace["target_inputs_immutable"] is not True
        or not isinstance(storages, list)
        or len(storages) != len(expected)
        or not isinstance(trace["recorded_calls"], list)
        or type(trace["calls_truncated"]) is not bool
    ):
        raise ValueError(f"{label} dispatch/liveness proof mismatch")
    seen_storages: set[str] = set()
    for index, storage in enumerate(storages):
        storage = _require_mapping(storage, PER_STORAGE_FIELDS, f"{label}.storage[{index}]")
        if (
            not isinstance(storage["storage"], str)
            or storage["storage"] in seen_storages
            or type(storage["semantic_consumer_count"]) is not int
            or storage["semantic_consumer_count"] <= 0
            or not isinstance(storage["first_semantic_consumer"], str)
            or not storage["first_semantic_consumer"]
            or storage["materialization_before_consumer"] is not False
            or storage["erase_before_semantic_calls"] != []
            or not isinstance(storage["erase_after_semantic_calls"], list)
        ):
            raise ValueError(f"{label} per-storage liveness proof mismatch")
        seen_storages.add(storage["storage"])
    return len(expected)


def _verify_liveness(
    liveness_dir: Path,
    both_pass: set[str],
    manifests: Sequence[Mapping[str, Any]],
    parents_path: Path,
    children_path: Path,
    manifest_path: Path,
    allowlist_path: Path,
) -> tuple[dict[str, Any], set[str], dict[str, dict[str, Any]]]:
    shard_count, raw = evidence_common._shard_records(liveness_dir)
    if len(raw) != len(both_pass):
        raise ValueError(f"liveness count differs from both-pass allowlist:{len(raw)}:{len(both_pass)}")
    validator_archive = liveness_dir / "validator_source.py"
    launcher_archive = liveness_dir / "launcher_source.sh"
    if _sha256_file(validator_archive) != _sha256_file(Path(validator.__file__).resolve()) or _sha256_file(
        launcher_archive
    ) != _sha256_file(Path(__file__).with_name("launch_layout_liveness_shards.sh")):
        raise ValueError("archived liveness source differs from current source")
    expected_artifacts = {
        "parents_sha256": _sha256_file(parents_path),
        "children_sha256": _sha256_file(children_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "allowlist_sha256": _sha256_file(allowlist_path),
    }
    by_manifest = {str(row["child_uuid"]): row for row in manifests}
    selected = [str(row["child_uuid"]) for row in manifests if str(row["child_uuid"]) in both_pass]
    position = {uuid: index for index, uuid in enumerate(selected)}
    records: dict[str, dict[str, Any]] = {}
    passed: set[str] = set()
    statuses: collections.Counter[str] = collections.Counter()
    reasons: collections.Counter[str] = collections.Counter()
    family_status: collections.Counter[tuple[str, str]] = collections.Counter()
    bindings: set[str] = set()
    runtime_fingerprints: set[str] = set()
    gpus: dict[str, dict[str, Any]] = {}
    observations: collections.Counter[str] = collections.Counter()
    for shard, line, record in raw:
        uuid = record.get("child_uuid")
        if uuid not in both_pass or uuid in records or position[uuid] % shard_count != shard:
            raise ValueError(f"invalid liveness shard identity:{shard}:{line}:{uuid}")
        if (
            record.get("contract_version") != validator.CONTRACT_VERSION
            or record.get("binding_version") != validator.RUN_BINDING_VERSION
            or type(record.get("passed")) is not bool
        ):
            raise ValueError(f"liveness contract/verdict mismatch:{uuid}")
        config = _require_mapping(record.get("validation_config"), CONFIG_FIELDS, f"{uuid}.config")
        if (
            config
            != {
                "device": "cuda:0",
                "trials": config["trials"],
                "seed": 17,
                "timeout_seconds": config["timeout_seconds"],
                "max_device_memory_gib": 64.0,
                "raw_direct_invocation": True,
            }
            or type(config["trials"]) is not int
            or config["trials"] < 3
            or not isinstance(config["timeout_seconds"], (int, float))
            or config["timeout_seconds"] <= 0
        ):
            raise ValueError(f"liveness policy mismatch:{uuid}")
        if record.get("validator_source_sha256") != _sha256_file(validator_archive) or record.get(
            "launcher_source_sha256"
        ) != _sha256_file(launcher_archive):
            raise ValueError(f"liveness source mismatch:{uuid}")
        if any(record.get(field) != expected for field, expected in expected_artifacts.items()):
            raise ValueError(f"liveness artifact binding mismatch:{uuid}")
        evidence = {
            "contract_version": record["contract_version"],
            "binding_version": record["binding_version"],
            "validator_source_sha256": record["validator_source_sha256"],
            "launcher_source_sha256": record["launcher_source_sha256"],
            **expected_artifacts,
            "validation_config": config,
        }
        if record.get("validation_binding_sha256") != _canonical_sha256(evidence):
            raise ValueError(f"liveness binding fingerprint mismatch:{uuid}")
        bindings.add(record["validation_binding_sha256"])
        manifest = by_manifest[uuid]
        family = str(manifest["assigned_family"])
        expected_count = int(manifest["transformed_factory_count"])
        if (
            record.get("parent_uuid") != manifest["parent_uuid"]
            or record.get("assigned_family") != family
            or record.get("factory_count") != manifest["factory_count"]
            or record.get("transformed_factory_count") != expected_count
        ):
            raise ValueError(f"liveness manifest identity mismatch:{uuid}")
        status = str(record.get("status"))
        statuses[status] += 1
        family_status[(family, status)] += 1
        if record["passed"]:
            if (
                status != "passed"
                or record.get("reason") is not None
                or not isinstance(record.get("memory_guard"), Mapping)
                or record["memory_guard"].get("within_limit") is not True
            ):
                raise ValueError(f"passed liveness record malformed:{uuid}")
            gpu = evidence_common._verify_gpu_evidence(
                record.get("gpu"), label=f"{uuid}.gpu", expected_device="cuda:0", require_cudnn=True
            )
            gpus[_canonical_sha256(gpu)] = gpu
            runtime_fingerprints.add(_canonical_sha256({"gpu": gpu, "config": config}))
            trials = record.get("trials")
            if not isinstance(trials, list) or len(trials) != config["trials"]:
                raise ValueError(f"liveness trial count mismatch:{uuid}")
            for ordinal, trial in enumerate(trials):
                observations[family] += _verify_trial(trial, manifest, ordinal, config, f"{uuid}.trials[{ordinal}]")
            passed.add(uuid)
        elif status == "passed":
            raise ValueError(f"failed row has passed status:{uuid}")
        else:
            reasons[str(record.get("reason"))] += 1
        records[uuid] = record
    if set(records) != both_pass or len(bindings) != 1 or len(gpus) > 1 or len(runtime_fingerprints) > 1:
        raise ValueError("liveness evidence is incomplete or mixes partitions")
    return (
        {
            "contract_version": validator.CONTRACT_VERSION,
            "shard_count": shard_count,
            "selected": len(records),
            "passed": len(passed),
            "failed": len(records) - len(passed),
            "status_counts": dict(sorted(statuses.items())),
            "family_status_counts": {
                family: {status: count for (name, status), count in sorted(family_status.items()) if name == family}
                for family in sorted(FAMILIES)
            },
            "passed_family_counts": dict(
                sorted(collections.Counter(by_manifest[uuid]["assigned_family"] for uuid in passed).items())
            ),
            "failure_reasons": dict(sorted(reasons.items())),
            "input_layout_observations": dict(sorted(observations.items())),
            "selected_factory_coverage": _factory_coverage(manifests, both_pass),
            "passed_factory_coverage": _factory_coverage(manifests, passed),
            "validation_binding_sha256": next(iter(bindings)),
            "launcher_source_sha256": _sha256_file(launcher_archive),
            "validator_source_sha256": _sha256_file(validator_archive),
            "runtime_partition_fingerprint": next(iter(runtime_fingerprints), None),
            "gpu": next(iter(gpus.values()), None),
        },
        passed,
        records,
    )


def _materialize(children: pa.Table, indices: Sequence[int], fingerprint: str) -> pa.Table:
    rows = children.take(pa.array(list(indices), type=pa.int64())).to_pylist()
    for index, row in enumerate(rows):
        extra = row.get("extra_info")
        if (
            not isinstance(extra, dict)
            or not isinstance(extra.get("v4"), dict)
            or not isinstance(extra.get("augmentation"), dict)
        ):
            raise ValueError(f"accepted child lacks status fields:{index}")
        extra["v4"].update(
            {
                "runtime_validation_status": ACCEPTED_RUNTIME_STATUS,
                "governance_status": ACCEPTED_GOVERNANCE_STATUS,
                "included_in_review_train": False,
            }
        )
        extra["augmentation"]["validation_status"] = ACCEPTED_GOVERNANCE_STATUS
    metadata = dict(children.schema.metadata or {})
    metadata.update(
        {
            b"layout.analysis_contract": ANALYSIS_CONTRACT.encode(),
            b"layout.runtime_policy_fingerprint": fingerprint.encode(),
            b"layout.governance_status": ACCEPTED_GOVERNANCE_STATUS.encode(),
            b"layout.training_approved": b"false",
        }
    )
    return pa.Table.from_pylist(rows, schema=children.schema.with_metadata(metadata))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _raw_manifest(lane: Path, roots: Sequence[Path]) -> Path:
    files: dict[str, str] = {}
    for root in roots:
        resolved = root.resolve()
        try:
            resolved.relative_to(lane.resolve())
        except ValueError as exc:
            raise ValueError(f"evidence root outside lane:{resolved}") from exc
        for path in sorted(resolved.rglob("*")):
            if path.is_file() and not path.name.startswith("."):
                files[str(path.relative_to(lane))] = _sha256_file(path)
    output = lane / "runtime/raw_artifact_sha256.json"
    _write_json(
        output, {"contract_version": "layout_raw_artifact_manifest_v1", "file_count": len(files), "files": files}
    )
    return output


def _report(
    path: Path,
    reference: Mapping[str, Any],
    liveness: Mapping[str, Any] | None,
    manifests: Sequence[Mapping[str, Any]],
    accepted: set[str] | None,
) -> None:
    candidate_coverage = reference["candidate_factory_coverage"]
    both_pass_coverage = reference["both_pass_factory_coverage"]
    lines = [
        "# Layout canary audit",
        "",
        f"- Candidate pairs: {reference['candidate_pairs']}",
        f"- Paired-reference both-pass: {reference['both_pass_children']}",
        f"- Candidate factory leaves total/transformed: {candidate_coverage['total_factory_leaves']}/{candidate_coverage['transformed_factory_leaves']}",
        f"- Candidate partial/full records: {candidate_coverage['partial_records']}/{candidate_coverage['full_records']}",
        f"- Both-pass factory leaves total/transformed: {both_pass_coverage['total_factory_leaves']}/{both_pass_coverage['transformed_factory_leaves']}",
        "",
        "## Reference classification",
        "",
        "| Classification | Count |",
        "| --- | ---: |",
        *(f"| {key} | {value} |" for key, value in reference["classifications"].items()),
    ]
    if liveness is not None and accepted is not None:
        lines += [
            "",
            "## Raw layout liveness",
            "",
            f"- Selected: {liveness['selected']}",
            f"- Passed: {liveness['passed']}",
            f"- Failed: {liveness['failed']}",
            f"- Failure buckets: `{liveness['failure_reasons']}`",
            "",
            "## Accepted bias slices",
            "",
        ]
        indexed = {str(row["child_uuid"]): row for row in manifests}
        accepted_coverage = _factory_coverage(manifests, accepted)
        lines.extend(
            [
                f"- factory leaves total/transformed: {accepted_coverage['total_factory_leaves']}/{accepted_coverage['transformed_factory_leaves']}",
                f"- partial/full records: {accepted_coverage['partial_records']}/{accepted_coverage['full_records']}",
                f"- factory coverage by family: `{accepted_coverage['by_family']}`",
            ]
        )
        for title, field in (
            ("family", "assigned_family"),
            ("source", "source_family"),
            ("operator", "operator_bucket"),
        ):
            lines.append(
                f"- {title}: `{dict(sorted(collections.Counter(str(indexed[uuid].get(field)) for uuid in accepted).items()))}`"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lane_dir", type=Path)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--liveness-dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError(
            f"layout exact AST replay requires Python 3.12, found {sys.version_info.major}.{sys.version_info.minor}"
        )
    args = _parser().parse_args(argv)
    lane = args.lane_dir
    parents_path, children_path, paired_path, manifest_path = (
        lane / "parents.parquet",
        lane / "candidates.parquet",
        lane / "paired.parquet",
        lane / "manifest.jsonl",
    )
    parent_rows = pq.ParquetFile(parents_path).metadata.num_rows
    child_rows = pq.ParquetFile(children_path).metadata.num_rows
    paired_rows = pq.ParquetFile(paired_path).metadata.num_rows
    if not 1 <= child_rows <= MAX_AUTHORIZED_CANDIDATES:
        raise ValueError(f"candidate count must be in [1,{MAX_AUTHORIZED_CANDIDATES}]:{child_rows}")
    if parent_rows != child_rows or paired_rows != 2 * child_rows:
        raise ValueError(f"parquet row counts must be N/N/2N:{parent_rows}:{child_rows}:{paired_rows}")
    parents = pq.read_table(parents_path).to_pylist()
    children_table = pq.read_table(children_path)
    children = children_table.to_pylist()
    paired = pq.read_table(paired_path).to_pylist()
    manifests = _read_jsonl(manifest_path)
    source_binding = _verify_static(parents, children, manifests, paired)
    # The shared verifier's reporting key is named assigned_target.  Supplying
    # this in-memory alias keeps its strict paired-runtime contract unchanged.
    reference_manifests = [{**row, "assigned_target": row["assigned_family"]} for row in manifests]
    reference, both_pass, reference_records = evidence_common._verify_reference(
        reference_dir=args.reference_dir,
        paired_path=paired_path,
        parents=parents,
        children=children,
        manifests=reference_manifests,
    )
    reference = {
        **reference,
        "contract_version": ANALYSIS_CONTRACT,
        "static_source_binding": source_binding,
        "candidate_factory_coverage": _factory_coverage(manifests),
        "both_pass_factory_coverage": _factory_coverage(manifests, both_pass),
    }
    analysis = lane / "analysis"
    _write_json(analysis / "reference_summary.json", reference)
    allowlist = analysis / "reference_both_pass_child_uuids.txt"
    allowlist.write_text("".join(f"{uuid}\n" for uuid in sorted(both_pass)), encoding="utf-8")
    liveness = None
    accepted = None
    if args.liveness_dir is not None:
        liveness, accepted, live_records = _verify_liveness(
            args.liveness_dir, both_pass, manifests, parents_path, children_path, manifest_path, allowlist
        )
        if accepted and liveness["gpu"] != reference["gpu"]:
            raise ValueError("reference and liveness GPU differ")
        _write_json(analysis / "liveness_summary.json", liveness)
        indices = [index for index, row in enumerate(manifests) if row["child_uuid"] in accepted]
        policy = {
            **source_binding,
            "reference_contract_fingerprint": reference["contract_fingerprint"],
            "liveness_binding_sha256": liveness["validation_binding_sha256"],
            "liveness_runtime_partition_fingerprint": liveness["runtime_partition_fingerprint"],
        }
        fingerprint = _canonical_sha256(policy)
        accepted_table = _materialize(children_table, indices, fingerprint)
        accepted_path = lane / "runtime/accepted.parquet"
        accepted_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(accepted_table, accepted_path)
        rows = {_nested(row, "extra_info.uuid"): row for row in accepted_table.to_pylist()}
        accepted_manifest = lane / "runtime/accepted.manifest.jsonl"
        with accepted_manifest.open("w", encoding="utf-8") as handle:
            for row in manifests:
                if row["child_uuid"] in accepted:
                    handle.write(
                        json.dumps(
                            {
                                **row,
                                "candidate_row_sha256": _canonical_sha256(
                                    next(
                                        child
                                        for child in children
                                        if _nested(child, "extra_info.uuid") == row["child_uuid"]
                                    )
                                ),
                                "row_sha256": _canonical_sha256(rows[row["child_uuid"]]),
                                "parent_runtime_status": "passed",
                                "child_runtime_status": "passed",
                                "layout_realization_status": "passed",
                                "liveness_status": "passed",
                                "materialization_status": ACCEPTED_GOVERNANCE_STATUS,
                                "runtime_policy_fingerprint": fingerprint,
                                "training_approved": False,
                                "runtime_evidence": {
                                    "policy": policy,
                                    "child_reference_status": reference_records[row["child_uuid"]]["status"],
                                    "liveness_status": live_records[row["child_uuid"]]["status"],
                                },
                            },
                            sort_keys=True,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
        raw = _raw_manifest(lane, [args.reference_dir, args.liveness_dir])
        _write_json(
            lane / "runtime/final_summary.json",
            {
                "contract_version": ANALYSIS_CONTRACT,
                "scope": "small_batch_canary_only",
                "maximum_authorized_candidates": MAX_AUTHORIZED_CANDIDATES,
                "candidate_rows": len(children),
                "reference": reference,
                "liveness": liveness,
                "accepted_rows": len(indices),
                "rejected_rows": len(children) - len(indices),
                "materialization_status": ACCEPTED_GOVERNANCE_STATUS,
                "training_approved": False,
                "runtime_policy_fingerprint": fingerprint,
                "runtime_policy": policy,
                "source_git_commit": source_binding["source_git_commit"],
                "source_sha256": source_binding["source_sha256"],
                "factory_coverage": {
                    "candidate": _factory_coverage(manifests),
                    "accepted": _factory_coverage(manifests, accepted),
                },
                "artifacts": {
                    "accepted_parquet_sha256": _sha256_file(accepted_path),
                    "accepted_manifest_sha256": _sha256_file(accepted_manifest),
                    "reference_allowlist_sha256": _sha256_file(allowlist),
                    "raw_artifact_manifest_sha256": _sha256_file(raw),
                },
            },
        )
    _report(analysis / "failure_bias_report.md", reference, liveness, manifests, accepted)
    print(
        json.dumps(
            {
                "candidate_pairs": len(children),
                "reference_both_pass": len(both_pass),
                "liveness_passed": len(accepted) if accepted is not None else None,
                "analysis_dir": str(analysis.resolve()),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
