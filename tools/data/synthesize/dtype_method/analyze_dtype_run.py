#!/usr/bin/env python3
"""Exactly verify dtype canary evidence and materialize review-only children."""

from __future__ import annotations

import argparse
import collections
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from tools.data.synthesize.dtype_method import solve_dtype_coverage as dtype_solver
from tools.data.synthesize.dtype_method import validate_dtype_liveness as dtype_validator
from tools.data.synthesize.random_method import analyze_value_run as evidence_common

ANALYSIS_CONTRACT = "dtype_parameter_free_exact_analysis_v1"
EXPECTED_DTYPES = frozenset({"float16", "bfloat16"})
MAX_AUTHORIZED_CANDIDATES = 5_000
MIN_LIVENESS_TRIALS = 3
ACCEPTED_RUNTIME_STATUS = "dtype_parameter_free_reference_and_liveness_passed"
ACCEPTED_GOVERNANCE_STATUS = "dtype_intervention_review_only"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
LIVENESS_CONFIG_FIELDS = frozenset(
    {
        "device",
        "trials",
        "seed",
        "timeout_seconds",
        "max_device_memory_gib",
        "float16_rtol",
        "float16_atol",
        "bfloat16_rtol",
        "bfloat16_atol",
        "dispatch_fp32_fallback_allowed",
    }
)
DISPATCH_FIELDS = frozenset(
    {
        "total_dispatch_calls",
        "target_consuming_calls",
        "fp32_or_complex_fallback_calls",
        "transition_count",
        "transitions_truncated",
        "transitions",
    }
)
INPUT_RECORD_FIELDS = frozenset(
    {
        "path",
        "shape",
        "parent_dtype",
        "child_dtype",
        "stride",
        "storage_offset",
        "is_pinned",
        "numel",
        "factory_values_equal_to_parent_cast",
        "maximum_factory_cast_difference",
        "minimum",
        "maximum",
    }
)
OUTPUT_RECORD_FIELDS = frozenset(
    {
        "path",
        "shape",
        "parent_dtype",
        "child_dtype",
        "elements",
        "maximum_absolute_difference",
        "mean_absolute_difference",
        "rtol",
        "atol",
    }
)


def _sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _nested(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _require_exact_mapping(value: Any, fields: frozenset[str], *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ValueError(f"{label} fields differ:{actual}:{sorted(fields)}")
    return value


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"manifest line is not an object:{line_number}")
        records.append(value)
    if not 1 <= len(records) <= MAX_AUTHORIZED_CANDIDATES:
        raise ValueError(f"candidate count must be in [1, {MAX_AUTHORIZED_CANDIDATES}]:{len(records)}")
    return records


def _normalize_canonical_parent(row: Mapping[str, Any], *, source_row_index: int) -> dict[str, Any]:
    normalized = dict(row)
    extra = normalized.get("extra_info")
    if not isinstance(extra, Mapping):
        raise ValueError(f"canonical source row lacks extra_info:{source_row_index}")
    normalized_extra = dict(extra)
    if "augmentation" in normalized_extra and normalized_extra["augmentation"] is not None:
        raise ValueError(f"canonical source unexpectedly has augmentation:{source_row_index}")
    normalized_extra["augmentation"] = None
    normalized["extra_info"] = normalized_extra
    return normalized


def _canonical_rows(manifests: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    paths = {manifest.get("source_artifact_path") for manifest in manifests}
    hashes = {manifest.get("source_artifact_sha256") for manifest in manifests}
    if len(paths) != 1 or len(hashes) != 1:
        raise ValueError("dtype manifests do not bind one canonical source")
    path_text = next(iter(paths))
    digest = next(iter(hashes))
    if not isinstance(path_text, str) or not Path(path_text).is_absolute():
        raise ValueError("canonical source path is not absolute")
    path = Path(path_text)
    if not path.is_file() or str(path.resolve()) != path_text:
        raise ValueError("canonical source path is unavailable or unresolved")
    if digest != dtype_solver.EXPECTED_CANONICAL_PARENT_SHA256 or _sha256_file(path) != digest:
        raise ValueError("canonical source SHA mismatch")
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != dtype_solver.EXPECTED_CANONICAL_PARENT_ROWS:
        raise ValueError("canonical source row count mismatch")
    indices = {manifest.get("source_row_index") for manifest in manifests}
    if any(type(index) is not int or not 0 <= index < parquet.metadata.num_rows for index in indices):
        raise ValueError("invalid canonical source row index")
    selected: dict[int, dict[str, Any]] = {}
    offset = 0
    for batch in parquet.iter_batches(batch_size=256, use_threads=False):
        rows = batch.to_pylist()
        for local_index, row in enumerate(rows):
            index = offset + local_index
            if index in indices:
                selected[index] = _normalize_canonical_parent(row, source_row_index=index)
        offset += len(rows)
        if len(selected) == len(indices):
            break
    if set(selected) != indices:
        raise ValueError("failed to reload every canonical dtype parent")
    return selected


def _verify_static(
    parents: list[dict[str, Any]],
    children: list[dict[str, Any]],
    manifests: list[dict[str, Any]],
    paired: list[dict[str, Any]],
) -> None:
    if not (len(parents) == len(children) == len(manifests)):
        raise ValueError("aligned dtype artifact count mismatch")
    expected_paired = [row for pair in zip(parents, children, strict=True) for row in pair]
    if _canonical_bytes(expected_paired) != _canonical_bytes(paired):
        raise ValueError("paired parquet is not the exact parent/child interleave")
    generator_path = Path(dtype_solver.__file__).resolve()
    dependency_path = generator_path.parent.parent / "augment_prompt_tasks.py"
    current_generator_sha = _sha256_file(generator_path)
    current_dependency_sha = _sha256_file(dependency_path)
    commits = {manifest.get("git_commit") for manifest in manifests}
    if len(commits) != 1:
        raise ValueError("dtype manifests mix Git commits")
    commit = next(iter(commits))
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError("dtype manifest Git commit is invalid")
    if dtype_solver._git_blob_sha256(commit, generator_path) != current_generator_sha:
        raise ValueError("dtype solver differs from its manifest commit")
    if dtype_solver._git_blob_sha256(commit, dependency_path) != current_dependency_sha:
        raise ValueError("dtype dependency differs from its manifest commit")
    canonical = _canonical_rows(manifests)
    uniqueness: dict[str, set[str]] = {
        "parent_uuid": set(),
        "child_uuid": set(),
        "child_reference_sha256": set(),
        "child_normalized_ast_sha256": set(),
    }
    for index, (parent, child, manifest) in enumerate(zip(parents, children, manifests, strict=True)):
        if manifest.get("candidate_row_index") != index:
            raise ValueError(f"manifest index mismatch:{index}")
        if manifest.get("primary_intervention") != "dtype" or manifest.get("assigned_target") not in EXPECTED_DTYPES:
            raise ValueError(f"manifest dtype intervention mismatch:{index}")
        if manifest.get("generator_contract_version") != dtype_solver.CONTRACT_VERSION:
            raise ValueError(f"manifest solver contract mismatch:{index}")
        if manifest.get("generator_source_sha256") != current_generator_sha:
            raise ValueError(f"manifest solver SHA mismatch:{index}")
        if manifest.get("dependency_source_sha256") != current_dependency_sha:
            raise ValueError(f"manifest dependency SHA mismatch:{index}")
        for field in ("shape_changed", "value_changed", "layout_changed", "model_changed", "get_init_inputs_changed"):
            if manifest.get(field) is not False:
                raise ValueError(f"non-dtype axis changed:{index}:{field}")
        if manifest.get("dtype_changed") is not True:
            raise ValueError(f"dtype axis not declared changed:{index}")
        if manifest.get("model_parameter_count") != 0 or manifest.get("model_buffer_count") != 0:
            raise ValueError(f"parameter-free manifest proof mismatch:{index}")
        if manifest.get("explicit_model_casts") != [] or manifest.get("runtime_promotion_status") != "pending":
            raise ValueError(f"static cast/promotion status mismatch:{index}")
        parent_uuid = _nested(parent, "extra_info.uuid")
        child_uuid = _nested(child, "extra_info.uuid")
        parent_code = _nested(parent, "reward_model.ground_truth")
        child_code = _nested(child, "reward_model.ground_truth")
        if manifest.get("parent_uuid") != parent_uuid or manifest.get("child_uuid") != child_uuid:
            raise ValueError(f"manifest UUID mismatch:{index}")
        if not isinstance(parent_code, str) or _sha256_bytes(parent_code.encode()) != manifest.get(
            "parent_reference_sha256"
        ):
            raise ValueError(f"parent reference mismatch:{index}")
        if not isinstance(child_code, str) or _sha256_bytes(child_code.encode()) != manifest.get(
            "child_reference_sha256"
        ):
            raise ValueError(f"child reference mismatch:{index}")
        source_index = manifest.get("source_row_index")
        if _canonical_bytes(canonical[source_index]) != _canonical_bytes(parent):
            raise ValueError(f"parent differs from canonical source:{index}")
        replay_item, reason = dtype_solver._analyze_parent(parent, source_index)
        if replay_item is None:
            raise ValueError(f"current dtype solver rejects parent:{index}:{reason}")
        replay_child, replay_manifest = dtype_solver._make_child(
            parent,
            replay_item,
            source_path=Path(manifest["source_artifact_path"]),
            source_sha256=manifest["source_artifact_sha256"],
            generator_sha256=current_generator_sha,
            dependency_sha256=current_dependency_sha,
            git_commit=commit,
        )
        replay_manifest["candidate_row_index"] = index
        if _canonical_bytes(replay_child) != _canonical_bytes(child):
            raise ValueError(f"deterministic dtype child replay mismatch:{index}")
        if _canonical_bytes(replay_manifest) != _canonical_bytes(manifest):
            raise ValueError(f"deterministic dtype manifest replay mismatch:{index}")
        for field in uniqueness:
            value = manifest[field]
            if value in uniqueness[field]:
                raise ValueError(f"duplicate dtype manifest field:{field}:{value}")
            uniqueness[field].add(value)


def _verify_dispatch(value: Any, *, target: str, label: str) -> dict[str, Any]:
    dispatch = _require_exact_mapping(value, DISPATCH_FIELDS, label=label)
    total = dispatch["total_dispatch_calls"]
    consuming = dispatch["target_consuming_calls"]
    fallback = dispatch["fp32_or_complex_fallback_calls"]
    transition_count = dispatch["transition_count"]
    truncated = dispatch["transitions_truncated"]
    transitions = dispatch["transitions"]
    if (
        type(total) is not int
        or type(consuming) is not int
        or type(fallback) is not int
        or type(transition_count) is not int
        or type(truncated) is not bool
        or not isinstance(transitions, list)
        or not 0 < consuming <= total
        or fallback != 0
        or transition_count < len(transitions)
        or truncated != (transition_count > dtype_validator.MAX_RECORDED_DISPATCH_TRANSITIONS)
    ):
        raise ValueError(f"{label} has inconsistent dispatch totals")
    target_name = f"torch.{target}"
    saw_target_input = False
    for index, transition in enumerate(transitions):
        mapping = _require_exact_mapping(
            transition, frozenset({"operator", "input_dtypes", "output_dtypes", "calls"}), label=f"{label}[{index}]"
        )
        if not isinstance(mapping["operator"], str) or not mapping["operator"]:
            raise ValueError(f"{label}[{index}] has invalid operator")
        if type(mapping["calls"]) is not int or mapping["calls"] <= 0:
            raise ValueError(f"{label}[{index}] has invalid call count")
        for dtype_field in ("input_dtypes", "output_dtypes"):
            if not isinstance(mapping[dtype_field], list) or any(
                not isinstance(item, str) for item in mapping[dtype_field]
            ):
                raise ValueError(f"{label}[{index}] has invalid {dtype_field}")
        saw_target_input |= target_name in mapping["input_dtypes"]
        forbidden = {
            dtype
            for dtype in mapping["output_dtypes"]
            if dtype.startswith("torch.float") and dtype != target_name or dtype.startswith("torch.complex")
        }
        if forbidden:
            raise ValueError(f"{label}[{index}] records fallback dtypes:{sorted(forbidden)}")
    if not saw_target_input and not truncated:
        raise ValueError(f"{label} does not record a target-dtype consumer")
    return dict(dispatch)


def _verify_input_record(value: Any, *, target: str, label: str) -> dict[str, Any]:
    record = _require_exact_mapping(value, INPUT_RECORD_FIELDS, label=label)
    shape = record["shape"]
    stride = record["stride"]
    if (
        not isinstance(record["path"], str)
        or not isinstance(shape, list)
        or any(type(item) is not int or item <= 0 for item in shape)
        or not isinstance(stride, list)
        or len(stride) != len(shape)
        or any(type(item) is not int for item in stride)
        or record["parent_dtype"] != "torch.float32"
        or record["child_dtype"] != f"torch.{target}"
        or type(record["storage_offset"]) is not int
        or type(record["is_pinned"]) is not bool
        or type(record["numel"]) is not int
        or record["numel"] <= 0
        or type(record["factory_values_equal_to_parent_cast"]) is not bool
        or _finite(record["maximum_factory_cast_difference"]) is None
        or _finite(record["maximum_factory_cast_difference"]) < 0
        or _finite(record["minimum"]) is None
        or _finite(record["maximum"]) is None
    ):
        raise ValueError(f"{label} has invalid dtype input evidence")
    numel = math.prod(shape)
    if record["numel"] != numel:
        raise ValueError(f"{label} numel differs from shape")
    return dict(record)


def _verify_output_record(value: Any, *, target: str, label: str) -> dict[str, Any]:
    record = _require_exact_mapping(value, OUTPUT_RECORD_FIELDS, label=label)
    shape = record["shape"]
    if (
        not isinstance(record["path"], str)
        or not isinstance(shape, list)
        or any(type(item) is not int or item < 0 for item in shape)
        or type(record["elements"]) is not int
        or record["elements"] < 0
        or record["elements"] != math.prod(shape)
    ):
        raise ValueError(f"{label} has invalid output shape evidence")
    parent_dtype = record["parent_dtype"]
    child_dtype = record["child_dtype"]
    if parent_dtype == "torch.float32":
        tolerance = dtype_validator.LOW_PRECISION_TOLERANCES[target]
        if child_dtype != f"torch.{target}":
            raise ValueError(f"{label} does not preserve target output dtype")
        if record["rtol"] != tolerance or record["atol"] != tolerance:
            raise ValueError(f"{label} has an invalid low-precision tolerance")
    elif parent_dtype != child_dtype or record["rtol"] != 0.0 or record["atol"] != 0.0:
        raise ValueError(f"{label} has an invalid non-floating output comparison")
    maximum = _finite(record["maximum_absolute_difference"])
    mean = _finite(record["mean_absolute_difference"])
    if maximum is None or mean is None or maximum < 0 or mean < 0 or mean > maximum + 1e-15:
        raise ValueError(f"{label} has invalid output difference evidence")
    return dict(record)


def _verify_trial(
    value: Any,
    *,
    trial_index: int,
    target: str,
    expected_changed: int,
    label: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    fields = frozenset(
        {
            "trial",
            "seed",
            "changed_dtype_tensors",
            "inputs",
            "cast_equivalent_outputs",
            "semantic_dispatch",
            "realized_dispatch",
        }
    )
    trial = _require_exact_mapping(value, fields, label=label)
    if trial["trial"] != trial_index or trial["seed"] != 17 + 10_007 * trial_index:
        raise ValueError(f"{label} has an invalid trial index or seed")
    if trial["changed_dtype_tensors"] != expected_changed:
        raise ValueError(f"{label} has an invalid changed dtype count")
    inputs = trial["inputs"]
    outputs = trial["cast_equivalent_outputs"]
    if not isinstance(inputs, list) or len(inputs) != expected_changed:
        raise ValueError(f"{label} has an invalid input evidence count")
    if not isinstance(outputs, list) or not outputs:
        raise ValueError(f"{label} has no cast-equivalent output evidence")
    verified_inputs = [
        _verify_input_record(item, target=target, label=f"{label}.inputs[{index}]")
        for index, item in enumerate(inputs)
    ]
    verified_outputs = [
        _verify_output_record(item, target=target, label=f"{label}.outputs[{index}]")
        for index, item in enumerate(outputs)
    ]
    if len({item["path"] for item in verified_inputs}) != len(verified_inputs):
        raise ValueError(f"{label} repeats an input path")
    semantic = _verify_dispatch(trial["semantic_dispatch"], target=target, label=f"{label}.semantic_dispatch")
    realized = _verify_dispatch(trial["realized_dispatch"], target=target, label=f"{label}.realized_dispatch")
    return verified_inputs, verified_outputs, semantic["total_dispatch_calls"] + realized["total_dispatch_calls"]


def _verify_liveness(
    *,
    liveness_dir: Path,
    both_pass: set[str],
    manifests: list[dict[str, Any]],
    parents_path: Path,
    children_path: Path,
    manifest_path: Path,
    allowlist_path: Path,
) -> tuple[dict[str, Any], set[str], dict[str, dict[str, Any]]]:
    shard_count, raw = evidence_common._shard_records(liveness_dir)
    if len(raw) != len(both_pass):
        raise ValueError(f"liveness row count mismatch:{len(raw)}:{len(both_pass)}")
    archived_validator_sha = _sha256_file(liveness_dir / "validator_source.py")
    archived_launcher_sha = _sha256_file(liveness_dir / "launcher_source.sh")
    if archived_validator_sha != _sha256_file(Path(dtype_validator.__file__).resolve()):
        raise ValueError("archived dtype validator differs from current source")
    if archived_launcher_sha != _sha256_file(Path(__file__).with_name("launch_dtype_liveness_shards.sh")):
        raise ValueError("archived dtype launcher differs from current source")
    expected_artifacts = {
        "parents_sha256": _sha256_file(parents_path),
        "children_sha256": _sha256_file(children_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "allowlist_sha256": _sha256_file(allowlist_path),
    }
    manifest_by_uuid = {manifest["child_uuid"]: manifest for manifest in manifests}
    selected_order = [manifest["child_uuid"] for manifest in manifests if manifest["child_uuid"] in both_pass]
    selected_position = {uuid: index for index, uuid in enumerate(selected_order)}
    by_uuid: dict[str, dict[str, Any]] = {}
    passed: set[str] = set()
    status_counts: collections.Counter[str] = collections.Counter()
    target_status: collections.Counter[tuple[str, str]] = collections.Counter()
    failure_reasons: collections.Counter[str] = collections.Counter()
    bindings: set[str] = set()
    runtime_partitions: set[str] = set()
    gpu_mappings: dict[str, dict[str, Any]] = {}
    trial_counts: collections.Counter[str] = collections.Counter()
    input_observations: collections.Counter[str] = collections.Counter()
    output_observations: collections.Counter[str] = collections.Counter()
    dispatch_calls: collections.Counter[str] = collections.Counter()
    cast_equal_observations: collections.Counter[str] = collections.Counter()
    max_factory_difference: collections.defaultdict[str, float] = collections.defaultdict(float)
    max_output_difference: collections.defaultdict[str, float] = collections.defaultdict(float)
    for shard_index, line_number, record in raw:
        uuid = record.get("child_uuid")
        if uuid not in both_pass or uuid in by_uuid:
            raise ValueError(f"invalid liveness UUID:{shard_index}:{line_number}:{uuid}")
        if selected_position[uuid] % shard_count != shard_index:
            raise ValueError(f"liveness UUID assigned to wrong shard:{uuid}")
        if record.get("contract_version") != dtype_validator.CONTRACT_VERSION:
            raise ValueError(f"liveness contract mismatch:{uuid}")
        if record.get("binding_version") != dtype_validator.RUN_BINDING_VERSION:
            raise ValueError(f"liveness binding version mismatch:{uuid}")
        if type(record.get("passed")) is not bool:
            raise ValueError(f"liveness verdict is not boolean:{uuid}")
        config = _require_exact_mapping(
            record.get("validation_config"), LIVENESS_CONFIG_FIELDS, label=f"{uuid}.config"
        )
        if (
            config["device"] != "cuda:0"
            or type(config["trials"]) is not int
            or config["trials"] < MIN_LIVENESS_TRIALS
            or config["seed"] != 17
            or _finite(config["timeout_seconds"]) is None
            or _finite(config["timeout_seconds"]) <= 0
            or config["max_device_memory_gib"] != 64.0
            or config["float16_rtol"] != dtype_validator.LOW_PRECISION_TOLERANCES["float16"]
            or config["float16_atol"] != dtype_validator.LOW_PRECISION_TOLERANCES["float16"]
            or config["bfloat16_rtol"] != dtype_validator.LOW_PRECISION_TOLERANCES["bfloat16"]
            or config["bfloat16_atol"] != dtype_validator.LOW_PRECISION_TOLERANCES["bfloat16"]
            or config["dispatch_fp32_fallback_allowed"] is not False
        ):
            raise ValueError(f"liveness policy mismatch:{uuid}")
        if record.get("validator_source_sha256") != archived_validator_sha:
            raise ValueError(f"validator source mismatch:{uuid}")
        if record.get("launcher_source_sha256") != archived_launcher_sha:
            raise ValueError(f"launcher source mismatch:{uuid}")
        for field, expected in expected_artifacts.items():
            if record.get(field) != expected:
                raise ValueError(f"artifact binding mismatch:{uuid}:{field}")
        evidence = {
            "contract_version": record["contract_version"],
            "binding_version": record["binding_version"],
            "validator_source_sha256": record["validator_source_sha256"],
            "launcher_source_sha256": record["launcher_source_sha256"],
            **expected_artifacts,
            "validation_config": dict(config),
        }
        binding = _canonical_sha256(evidence)
        if record.get("validation_binding_sha256") != binding:
            raise ValueError(f"validation binding mismatch:{uuid}")
        bindings.add(binding)
        manifest = manifest_by_uuid[uuid]
        target = manifest["assigned_target"]
        expected_changed = manifest["transformed_factory_count"]
        if (
            record.get("parent_uuid") != manifest["parent_uuid"]
            or record.get("assigned_dtype") != target
            or record.get("transformed_factory_count") != expected_changed
        ):
            raise ValueError(f"liveness identity mismatch:{uuid}")
        status = str(record.get("status"))
        status_counts[status] += 1
        target_status[(target, status)] += 1
        if record["passed"]:
            if status != "passed" or record.get("reason") is not None or record.get("target_dtype") != target:
                raise ValueError(f"passed liveness status mismatch:{uuid}")
            tolerance = dtype_validator.LOW_PRECISION_TOLERANCES[target]
            if record.get("tolerance") != {"rtol": tolerance, "atol": tolerance}:
                raise ValueError(f"passed liveness tolerance mismatch:{uuid}")
            memory = record.get("memory_guard")
            if not isinstance(memory, Mapping) or memory.get("within_limit") is not True:
                raise ValueError(f"passed liveness memory guard mismatch:{uuid}")
            gpu = evidence_common._verify_gpu_evidence(
                record.get("gpu"), label=f"liveness {uuid}.gpu", expected_device="cuda:0", require_cudnn=True
            )
            gpu_mappings[_canonical_sha256(gpu)] = gpu
            runtime_partitions.add(_canonical_sha256({"gpu": gpu, "validation_config": dict(config)}))
            trials = record.get("trials")
            if not isinstance(trials, list) or len(trials) != config["trials"]:
                raise ValueError(f"passed liveness trial count mismatch:{uuid}")
            frozen_input_schema: list[dict[str, Any]] | None = None
            frozen_output_schema: list[dict[str, Any]] | None = None
            for trial_index, trial in enumerate(trials):
                inputs, outputs, calls = _verify_trial(
                    trial,
                    trial_index=trial_index,
                    target=target,
                    expected_changed=expected_changed,
                    label=f"{uuid}.trials[{trial_index}]",
                )
                input_schema = [
                    {
                        key: item[key]
                        for key in (
                            "path",
                            "shape",
                            "parent_dtype",
                            "child_dtype",
                            "stride",
                            "storage_offset",
                            "is_pinned",
                            "numel",
                        )
                    }
                    for item in inputs
                ]
                output_schema = [
                    {key: item[key] for key in ("path", "shape", "parent_dtype", "child_dtype", "elements")}
                    for item in outputs
                ]
                if frozen_input_schema is None:
                    frozen_input_schema = input_schema
                    frozen_output_schema = output_schema
                elif frozen_input_schema != input_schema or frozen_output_schema != output_schema:
                    raise ValueError(f"dtype tensor schema changes across trials:{uuid}")
                trial_counts[target] += 1
                input_observations[target] += len(inputs)
                output_observations[target] += len(outputs)
                dispatch_calls[target] += calls
                cast_equal_observations[target] += sum(item["factory_values_equal_to_parent_cast"] for item in inputs)
                max_factory_difference[target] = max(
                    max_factory_difference[target], *(item["maximum_factory_cast_difference"] for item in inputs)
                )
                max_output_difference[target] = max(
                    max_output_difference[target], *(item["maximum_absolute_difference"] for item in outputs)
                )
            passed.add(uuid)
        elif status == "passed":
            raise ValueError(f"failed liveness row has passed status:{uuid}")
        else:
            failure_reasons[str(record.get("reason"))] += 1
        by_uuid[uuid] = record
    if set(by_uuid) != both_pass:
        raise ValueError("liveness UUID set differs from reference allowlist")
    if len(bindings) != 1:
        raise ValueError("liveness evidence mixes validation bindings")
    if len(runtime_partitions) > 1 or len(gpu_mappings) > 1:
        raise ValueError("passed liveness evidence mixes runtime partitions")

    def nested_counts(counter: collections.Counter[tuple[str, str]]) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = collections.defaultdict(dict)
        for (target, status), count in sorted(counter.items()):
            result[target][status] = count
        return dict(result)

    summary = {
        "contract_version": dtype_validator.CONTRACT_VERSION,
        "shard_count": shard_count,
        "selected": len(by_uuid),
        "passed": len(passed),
        "failed": len(by_uuid) - len(passed),
        "status_counts": dict(sorted(status_counts.items())),
        "dtype_status_counts": nested_counts(target_status),
        "passed_dtype_counts": dict(
            sorted(collections.Counter(manifest_by_uuid[uuid]["assigned_target"] for uuid in passed).items())
        ),
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "passed_trial_observations": {
            target: {
                "trials": trial_counts[target],
                "input_tensor_observations": input_observations[target],
                "factory_equal_to_parent_cast_observations": cast_equal_observations[target],
                "output_tensor_observations": output_observations[target],
                "dispatch_calls": dispatch_calls[target],
                "maximum_factory_cast_difference": max_factory_difference[target],
                "maximum_cast_equivalent_output_difference": max_output_difference[target],
                "fp32_or_complex_fallback_calls": 0,
            }
            for target in sorted(EXPECTED_DTYPES)
        },
        "validation_binding_sha256": next(iter(bindings)),
        "launcher_source_sha256": archived_launcher_sha,
        "validator_source_sha256": archived_validator_sha,
        "runtime_partition_fingerprint": next(iter(runtime_partitions), None),
        "gpu": next(iter(gpu_mappings.values()), None),
    }
    return summary, passed, by_uuid


def _materialize_accepted(
    children_table: pa.Table, indices: Sequence[int], *, runtime_policy_fingerprint: str
) -> pa.Table:
    accepted = children_table.take(pa.array(list(indices), type=pa.int64()))
    rows = accepted.to_pylist()
    for index, row in enumerate(rows):
        extra = row.get("extra_info")
        if (
            not isinstance(extra, dict)
            or not isinstance(extra.get("v4"), dict)
            or not isinstance(extra.get("augmentation"), dict)
        ):
            raise ValueError(f"accepted dtype row lacks status structs:{index}")
        extra["v4"]["runtime_validation_status"] = ACCEPTED_RUNTIME_STATUS
        extra["v4"]["governance_status"] = ACCEPTED_GOVERNANCE_STATUS
        extra["v4"]["included_in_review_train"] = False
        extra["augmentation"]["validation_status"] = ACCEPTED_GOVERNANCE_STATUS
    metadata = dict(children_table.schema.metadata or {})
    metadata.update(
        {
            b"dtype.analysis_contract": ANALYSIS_CONTRACT.encode(),
            b"dtype.runtime_policy_fingerprint": runtime_policy_fingerprint.encode(),
            b"dtype.runtime_validation_status": ACCEPTED_RUNTIME_STATUS.encode(),
            b"dtype.governance_status": ACCEPTED_GOVERNANCE_STATUS.encode(),
            b"dtype.training_approved": b"false",
        }
    )
    return pa.Table.from_pylist(rows, schema=children_table.schema.with_metadata(metadata))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_raw_manifest(lane_dir: Path, roots: Sequence[Path]) -> Path:
    lane = lane_dir.resolve()
    files: dict[str, str] = {}
    root_names: list[str] = []
    for root in roots:
        resolved = root.resolve()
        try:
            relative_root = resolved.relative_to(lane)
        except ValueError as exc:
            raise ValueError(f"raw evidence root is outside lane:{resolved}") from exc
        root_names.append(str(relative_root))
        for path in sorted(resolved.rglob("*")):
            if path.is_file() and not path.name.startswith("."):
                relative = str(path.relative_to(lane))
                if relative in files:
                    raise ValueError(f"duplicate raw artifact:{relative}")
                files[relative] = _sha256_file(path)
    output = lane_dir / "runtime/raw_artifact_sha256.json"
    _write_json(
        output,
        {
            "contract_version": "dtype_raw_artifact_manifest_v1",
            "roots": root_names,
            "file_count": len(files),
            "files": files,
        },
    )
    return output


def _write_report(
    path: Path,
    reference: Mapping[str, Any],
    liveness: Mapping[str, Any] | None,
    manifests: list[dict[str, Any]],
    accepted: set[str] | None,
) -> None:
    lines = [
        "# Parameter-free dtype canary audit",
        "",
        f"- Candidate pairs: {reference['candidate_pairs']}",
        f"- Reference both-pass: {reference['both_pass_children']}",
        f"- GPU: `{reference.get('gpu')}`",
        "",
        "## Reference classification",
        "",
        "| Classification | Count |",
        "| --- | ---: |",
        *(f"| {key} | {value} |" for key, value in reference["classifications"].items()),
    ]
    if liveness is not None and accepted is not None:
        lines.extend(
            [
                "",
                "## Dtype liveness",
                "",
                f"- Selected: {liveness['selected']}",
                f"- Passed: {liveness['passed']}",
                f"- Failed: {liveness['failed']}",
                f"- Passed by dtype: `{liveness['passed_dtype_counts']}`",
                f"- Failure reasons: `{liveness['failure_reasons']}`",
                "",
                "## Accepted bias slices",
                "",
            ]
        )
        by_uuid = {manifest["child_uuid"]: manifest for manifest in manifests}
        for label, field in (
            ("dtype", "assigned_target"),
            ("source", "source_family"),
            ("operator", "operator_bucket"),
        ):
            counter = collections.Counter(str(by_uuid[uuid].get(field)) for uuid in accepted)
            lines.append(f"- {label}: `{dict(sorted(counter.items()))}`")
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
    parents = pq.read_table(parents_path).to_pylist()
    children_table = pq.read_table(children_path)
    children = children_table.to_pylist()
    paired = pq.read_table(paired_path).to_pylist()
    manifests = _read_manifest(manifest_path)
    _verify_static(parents, children, manifests, paired)
    reference, both_pass, child_reference_records = evidence_common._verify_reference(
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
            liveness_dir=args.liveness_dir,
            both_pass=both_pass,
            manifests=manifests,
            parents_path=parents_path,
            children_path=children_path,
            manifest_path=manifest_path,
            allowlist_path=allowlist_path,
        )
        if accepted and liveness["gpu"] != reference["gpu"]:
            raise ValueError("reference and dtype liveness use different GPUs")
        _write_json(analysis_dir / "liveness_summary.json", liveness)
        accepted_indices = [index for index, manifest in enumerate(manifests) if manifest["child_uuid"] in accepted]
        analyzer_sha = _sha256_file(Path(__file__).resolve())
        common_reference_analyzer_sha = _sha256_file(Path(evidence_common.__file__).resolve())
        policy = {
            "analyzer_source_sha256": analyzer_sha,
            "common_reference_analyzer_sha256": common_reference_analyzer_sha,
            "reference_contract_fingerprint": reference["contract_fingerprint"],
            "reference_runtime_environment_fingerprint": reference["runtime_environment_fingerprint"],
            "liveness_binding_sha256": liveness["validation_binding_sha256"],
            "liveness_runtime_partition_fingerprint": liveness["runtime_partition_fingerprint"],
        }
        runtime_policy_fingerprint = _canonical_sha256(policy)
        accepted_table = _materialize_accepted(
            children_table, accepted_indices, runtime_policy_fingerprint=runtime_policy_fingerprint
        )
        accepted_path = args.lane_dir / "runtime/accepted.parquet"
        accepted_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(accepted_table, accepted_path)
        accepted_rows = {_nested(row, "extra_info.uuid"): row for row in accepted_table.to_pylist()}
        if set(accepted_rows) != accepted:
            raise ValueError("accepted dtype parquet UUID set mismatch")
        accepted_manifest_path = args.lane_dir / "runtime/accepted.manifest.jsonl"
        with accepted_manifest_path.open("w", encoding="utf-8") as handle:
            for manifest in manifests:
                uuid = manifest["child_uuid"]
                if uuid not in accepted:
                    continue
                enriched = {
                    **manifest,
                    "candidate_row_sha256": manifest["row_sha256"],
                    "row_sha256": _canonical_sha256(accepted_rows[uuid]),
                    "parent_runtime_status": "passed",
                    "child_runtime_status": "passed",
                    "liveness_status": "passed",
                    "runtime_promotion_status": "no_fp32_or_complex_dispatch_fallback",
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
        raw_manifest = _write_raw_manifest(args.lane_dir, [args.reference_dir, args.liveness_dir])
        _write_json(
            args.lane_dir / "runtime/final_summary.json",
            {
                "contract_version": ANALYSIS_CONTRACT,
                "scope": "small_batch_canary_only",
                "maximum_authorized_candidates": MAX_AUTHORIZED_CANDIDATES,
                "candidate_rows": len(children),
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
                    "raw_artifact_manifest_sha256": _sha256_file(raw_manifest),
                },
                "analyzer_source_sha256": analyzer_sha,
                "common_reference_analyzer_sha256": common_reference_analyzer_sha,
            },
        )
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
