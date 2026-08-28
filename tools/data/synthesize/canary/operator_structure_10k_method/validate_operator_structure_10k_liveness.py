#!/usr/bin/env python3
"""Fail-closed CUDA liveness adapter for the operator-structure 10k lane.

The generator owns the registry and its row count.  This adapter deliberately
does not duplicate that registry: before any CUDA work it replays every
manifest through the generator, then delegates isolated paired execution and
returned-output ATen provenance to the reviewed semantic runtime core.
"""

from __future__ import annotations

import argparse
import collections
import fcntl
import hashlib
import importlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.cleaning import runtime_validation as runtime_validation_module  # noqa: E402
from tools.data.synthesize import validate_train_mode_contract as train_mode_contract_module  # noqa: E402
from tools.data.synthesize.semantic_operator_method import (  # noqa: E402
    generate_semantic_operator as semantic_operator_generator,
)
from tools.data.synthesize.semantic_operator_method import validate_semantic_liveness as runtime_core  # noqa: E402

CONTRACT_VERSION = "operator_structure_10k_runtime_liveness_v1"
RUN_BINDING_VERSION = "operator_structure_10k_runtime_binding_v1"
MIN_LIVENESS_TRIALS = 3
REQUIRED_LIVENESS_SEED = 17
DEFAULT_GENERATOR_MODULE = "tools.data.synthesize.canary.operator_structure_10k_method.generate_operator_structure_10k"
EXECUTION_CONTROLS = dict(runtime_core.EXECUTION_CONTROLS)
_RUNTIME_CORE_PATH = Path(runtime_core.__file__).resolve()
FAILURE_STATUSES = frozenset(
    {
        "reference_failed",
        "unsupported",
        "failed",
        "timeout",
        "cuda_out_of_memory",
        "device_memory_limit_exceeded",
        "gpu_memory_telemetry_error",
        "memory_guard_configuration_error",
        "error",
        "protocol_error",
    }
)
RAW_FAILURE_STATUS = {
    "reference_failed": "reference_failed",
    "unsupported": "unsupported",
    "failed": "failed",
    "timeout": "timeout",
    "cuda_out_of_memory": "cuda_out_of_memory",
    "device_memory_limit_exceeded": "device_memory_limit_exceeded",
    "gpu_memory_telemetry_error": "gpu_memory_telemetry_error",
    "memory_guard_configuration_error": "memory_guard_configuration_error",
    "error": "error",
    "worker_protocol_error": "protocol_error",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def _nested(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _load_generator(module_name: str) -> Any:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # Import failures must not accidentally run a partial lane.
        raise ValueError(f"cannot import configured operator-structure generator:{module_name}") from exc
    path = Path(getattr(module, "__file__", "")).resolve()
    if not path.is_file() or not callable(getattr(module, "replay_manifest", None)):
        raise ValueError("generator must be a file-backed module exporting replay_manifest")
    return module


def _generator_rows(generator: Any) -> int:
    value = getattr(generator, "EXACT_CANDIDATE_ROWS", None)
    if type(value) is not int or value <= 0:
        raise ValueError("generator must expose a positive EXACT_CANDIDATE_ROWS")
    return value


def _generator_templates(generator: Any) -> set[str]:
    values = getattr(generator, "TEMPLATES", None)
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError("generator TEMPLATES must be a non-empty sequence")
    result = {
        item.get("template_id") if isinstance(item, Mapping) else getattr(item, "template_id", None) for item in values
    }
    if not result or None in result or not all(isinstance(value, str) and value for value in result):
        raise ValueError("generator TEMPLATES has invalid template_id")
    return result


def _generator_string_contract(generator: Any, names: Sequence[str], label: str) -> str:
    values = [getattr(generator, name) for name in names if hasattr(generator, name)]
    if not values or any(not isinstance(value, str) or not value for value in values) or len(set(values)) != 1:
        raise ValueError(f"generator must expose one unambiguous {label} contract")
    return str(values[0])


def _validate_declared_ops(value: Any, index: int) -> list[dict[str, Any]]:
    required = {"op_id", "source_calls", "runtime_identities", "min_calls_per_trial", "must_reach_returned_output"}
    if not isinstance(value, list) or not value:
        raise ValueError(f"declared ops missing:{index}")
    seen, result = set(), []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != required:
            raise ValueError(f"declared op schema mismatch:{index}")
        op_id, calls, identities = item.get("op_id"), item.get("source_calls"), item.get("runtime_identities")
        if not isinstance(op_id, str) or not op_id or op_id in seen:
            raise ValueError(f"declared op id invalid:{index}:{op_id!r}")
        if not isinstance(calls, list) or not calls or not all(isinstance(x, str) and x for x in calls):
            raise ValueError(f"declared source calls invalid:{index}:{op_id}")
        if type(item.get("min_calls_per_trial")) is not int or item["min_calls_per_trial"] <= 0:
            raise ValueError(f"declared minimum invalid:{index}:{op_id}")
        if item.get("must_reach_returned_output") is not True or not isinstance(identities, list) or not identities:
            raise ValueError(f"declared output provenance invalid:{index}:{op_id}")
        for identity in identities:
            if not isinstance(identity, Mapping) or set(identity) != {"schema", "overload"}:
                raise ValueError(f"declared ATen identity schema invalid:{index}:{op_id}")
            if (
                not isinstance(identity["schema"], str)
                or not identity["schema"].startswith("aten::")
                or not isinstance(identity["overload"], str)
            ):
                raise ValueError(f"declared ATen identity invalid:{index}:{op_id}")
        seen.add(op_id)
        result.append(dict(item))
    return result


def _check_manifest_contract(manifest: Mapping[str, Any], generator: Any, index: int) -> None:
    for field, names, label in (
        ("manifest_contract_version", ("MANIFEST_CONTRACT_VERSION", "MANIFEST_VERSION"), "manifest"),
        ("generator_contract_version", ("GENERATOR_CONTRACT_VERSION", "CONTRACT_VERSION"), "generator"),
        ("static_contract_version", ("STATIC_CONTRACT_VERSION",), "static"),
        ("runtime_contract_version", ("RUNTIME_CONTRACT_VERSION",), "runtime"),
    ):
        if manifest.get(field) != _generator_string_contract(generator, names, label):
            raise ValueError(f"generator manifest contract mismatch:{index}:{field}")
    if manifest.get("parent_uuid") is not None or manifest.get("training_approved") is not False:
        raise ValueError(f"operator-structure task is not parentless/review-only:{index}")
    if manifest.get("structured_output_deferred") is not True:
        raise ValueError(f"operator-structure structured-output policy mismatch:{index}")
    if manifest.get("final_output_contract") != {"kind": "single_tensor", "finite_required": True}:
        raise ValueError(f"operator-structure final output contract mismatch:{index}")
    if manifest.get("static_status") != "passed" or not isinstance(manifest.get("static_proof"), Mapping):
        raise ValueError(f"operator-structure static proof missing/failed:{index}")


def tasks(
    candidates_path: Path, manifest_path: Path, *, generator_module: str = DEFAULT_GENERATOR_MODULE
) -> list[dict[str, Any]]:
    """Static replay gate shared by the launcher and evidence coordinator."""

    generator = _load_generator(generator_module)
    expected_rows = _generator_rows(generator)
    quotas = getattr(generator, "FAMILY_CANDIDATE_QUOTAS", None)
    templates = _generator_templates(generator)
    if (
        not isinstance(quotas, Mapping)
        or not quotas
        or any(type(v) is not int or v <= 0 for v in quotas.values())
        or sum(quotas.values()) != expected_rows
    ):
        raise ValueError("generator FAMILY_CANDIDATE_QUOTAS must be positive and sum to its exact row count")
    if pq.ParquetFile(candidates_path).metadata.num_rows != expected_rows:
        raise ValueError(f"candidate count differs from generator exact count:{expected_rows}")
    rows = pq.read_table(candidates_path).to_pylist()
    manifests = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != expected_rows or len(manifests) != expected_rows:
        raise ValueError("candidate/manifest exact count mismatch")
    generator_sha = _sha256_file(Path(generator.__file__).resolve())
    families: collections.Counter[str] = collections.Counter()
    template_counts: collections.Counter[str] = collections.Counter()
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, (row, manifest) in enumerate(zip(rows, manifests, strict=True)):
        if not isinstance(manifest, Mapping):
            raise ValueError(f"manifest is not an object:{index}")
        uuid, code, entry_point = (
            _nested(row, "extra_info.uuid"),
            _nested(row, "reward_model.ground_truth"),
            _nested(row, "extra_info.entry_point"),
        )
        if manifest.get("candidate_row_index") != index or manifest.get("uuid") != uuid:
            raise ValueError(f"candidate/manifest identity mismatch:{index}")
        _check_manifest_contract(manifest, generator, index)
        if manifest.get("generator_source_sha256") != generator_sha:
            raise ValueError(f"generator source binding mismatch:{index}")
        if not all(isinstance(value, str) and value for value in (uuid, code, entry_point)) or uuid in seen:
            raise ValueError(f"invalid/duplicate candidate identity:{index}")
        if entry_point != "Model" or _sha256_bytes(code.encode()) != manifest.get("reference_sha256"):
            raise ValueError(f"candidate reference binding mismatch:{index}")
        replay = generator.replay_manifest(manifest)
        if not isinstance(replay, Mapping):
            raise ValueError(f"generator replay is not an object:{index}")
        if any(
            manifest.get(key) != replay.get(key) for key in ("declared_ops", "coverage_labels", "static_proof")
        ) or code != replay.get("code"):
            raise ValueError(f"generator static replay mismatch:{uuid}")
        family, template, mode = (
            manifest.get("primary_family"),
            manifest.get("template_id"),
            manifest.get("mode_behavior"),
        )
        if (
            family not in quotas
            or template not in templates
            or mode not in {"stateless", "train_stateful", "recurrent_state"}
        ):
            raise ValueError(f"unregistered family/template/mode:{index}")
        seen.add(uuid)
        families[str(family)] += 1
        template_counts[str(template)] += 1
        result.append(
            {
                "candidate_row_index": index,
                "uuid": uuid,
                "reference_code": code,
                "entry_point": entry_point,
                "template_id": template,
                "primary_family": family,
                "mode_behavior": mode,
                "declared_ops": _validate_declared_ops(manifest.get("declared_ops"), index),
            }
        )
    if (
        dict(families) != dict(quotas)
        or set(template_counts) != templates
        or any(v <= 0 for v in template_counts.values())
    ):
        raise ValueError("generator family/template quota replay mismatch")
    return result


def _load_allowlist(path: Path) -> set[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("UUID allowlist must be non-empty and unique")
    return set(values)


def _source_binding(generator: Any, launcher_path: Path, launcher_sha256: str) -> dict[str, Any]:
    launcher_path, generator_path, adapter_path = (
        launcher_path.resolve(),
        Path(generator.__file__).resolve(),
        Path(__file__).resolve(),
    )
    if not launcher_path.is_file() or _sha256_file(launcher_path) != launcher_sha256:
        raise ValueError("launcher source binding mismatch")
    return {
        "adapter_source": {"path": str(adapter_path), "sha256": _sha256_file(adapter_path)},
        "shared_runtime_core_source": {
            "path": str(_RUNTIME_CORE_PATH),
            "sha256": _sha256_file(_RUNTIME_CORE_PATH),
            "contract_version": runtime_core.CONTRACT_VERSION,
        },
        "runtime_validation_source": {
            "path": str(Path(runtime_validation_module.__file__).resolve()),
            "sha256": _sha256_file(Path(runtime_validation_module.__file__).resolve()),
        },
        "train_mode_contract_source": {
            "path": str(Path(train_mode_contract_module.__file__).resolve()),
            "sha256": _sha256_file(Path(train_mode_contract_module.__file__).resolve()),
        },
        "semantic_operator_generator_source": {
            "path": str(Path(semantic_operator_generator.__file__).resolve()),
            "sha256": _sha256_file(Path(semantic_operator_generator.__file__).resolve()),
        },
        "generator_source": {"path": str(generator_path), "sha256": _sha256_file(generator_path)},
        "launcher_source": {"path": str(launcher_path), "sha256": launcher_sha256},
    }


def _append(handle: Any, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _failure_signature(record: Mapping[str, Any]) -> str | None:
    reason = record.get("reason")
    return (
        None
        if reason is None
        else _canonical_sha256(
            {
                "status": record.get("status"),
                "reason": reason,
                "error_type": record.get("error_type"),
                "error": record.get("error"),
            }
        )
    )


def _driver_h20_gpu_evidence_ok(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("device") == "cuda:0"
        and "H20" in str(value.get("name"))
        and isinstance(value.get("cuda_visible_devices"), str)
        and bool(value["cuda_visible_devices"])
    )


def _worker_h20_gpu_evidence_ok(value: Any) -> bool:
    capability = value.get("compute_capability") if isinstance(value, Mapping) else None
    cuda_version = value.get("torch_cuda_version") if isinstance(value, Mapping) else None
    return (
        isinstance(value, Mapping)
        and value.get("device") == "cuda:0"
        and "H20" in str(value.get("name"))
        and capability == [9, 0]
        and isinstance(cuda_version, str)
        and bool(cuda_version)
    )


def _memory_guard_structure_ok(value: Any) -> bool:
    """Validate the shared core's guard structure for either outcome."""

    telemetry_fields = (
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "current_allocated_bytes",
        "current_reserved_bytes",
    )
    if not (
        isinstance(value, Mapping)
        and value.get("max_device_memory_gib") == runtime_core.MAX_DEVICE_MEMORY_GIB
        and isinstance(value.get("enabled"), bool)
        and isinstance(value.get("configured"), bool)
        and isinstance(value.get("synchronized"), bool)
        and isinstance(value.get("over_limit"), bool)
        and isinstance(value.get("within_limit"), bool)
        and isinstance(value.get("limit_bytes"), int)
        and value["limit_bytes"] == int(runtime_core.MAX_DEVICE_MEMORY_GIB * 1024**3)
        and isinstance(value.get("total_device_bytes"), int)
        and value["total_device_bytes"] > 0
        and all(
            isinstance(value.get(field), int) and not isinstance(value.get(field), bool) and value[field] >= 0
            for field in telemetry_fields
        )
    ):
        return False
    synchronize_error = value.get("synchronize_error")
    telemetry_errors = value.get("telemetry_errors")
    if synchronize_error is not None and not isinstance(synchronize_error, Mapping):
        return False
    if telemetry_errors is not None and (
        not isinstance(telemetry_errors, Mapping)
        or not all(isinstance(item, Mapping) for item in telemetry_errors.values())
    ):
        return False
    if value["configured"] is True:
        fraction = value.get("allocator_fraction")
        if (
            not isinstance(fraction, (int, float))
            or isinstance(fraction, bool)
            or value["total_device_bytes"] < value["limit_bytes"]
            or not 0.0 < float(fraction) <= 1.0
            or not math.isclose(
                float(fraction), value["limit_bytes"] / value["total_device_bytes"], rel_tol=1e-12, abs_tol=0.0
            )
        ):
            return False
        if (value["synchronized"] is True) != (synchronize_error is None):
            return False
    elif value.get("allocator_fraction") is not None:
        fraction = value["allocator_fraction"]
        if (
            not isinstance(fraction, (int, float))
            or isinstance(fraction, bool)
            or not math.isclose(
                float(fraction), value["limit_bytes"] / value["total_device_bytes"], rel_tol=1e-12, abs_tol=0.0
            )
        ):
            return False
    observed_over = any(
        value[field] > value["limit_bytes"] for field in ("peak_allocated_bytes", "peak_reserved_bytes")
    )
    if value["over_limit"] is not observed_over:
        return False
    telemetry_error = synchronize_error is not None or bool(telemetry_errors)
    expected_within = (
        value["enabled"] is True
        and value["configured"] is True
        and value["synchronized"] is True
        and not telemetry_error
        and not observed_over
    )
    return value["within_limit"] is expected_within


def _passing_memory_guard_ok(value: Any) -> bool:
    if not _memory_guard_structure_ok(value):
        return False
    limit = value["limit_bytes"]
    fraction = value.get("allocator_fraction")
    telemetry = [
        value.get(field)
        for field in (
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "current_allocated_bytes",
            "current_reserved_bytes",
        )
    ]
    return (
        value["enabled"] is True
        and value["configured"] is True
        and value["synchronized"] is True
        and value["within_limit"] is True
        and value["over_limit"] is False
        and limit == int(runtime_core.MAX_DEVICE_MEMORY_GIB * 1024**3)
        and value["total_device_bytes"] >= limit
        and isinstance(fraction, (int, float))
        and not isinstance(fraction, bool)
        and math.isclose(float(fraction), limit / value["total_device_bytes"], rel_tol=1e-12, abs_tol=0.0)
        and all(isinstance(item, int) and not isinstance(item, bool) and 0 <= item <= limit for item in telemetry)
        and value.get("synchronize_error") is None
        and value.get("telemetry_errors") in (None, {})
    )


def _authorization_evidence_missing(result: Mapping[str, Any], task: Mapping[str, Any] | None = None) -> list[str]:
    expected_mutation = (task if task is not None else result).get("mode_behavior") == "train_stateful"
    aliases, trials = result.get("registered_object_alias_evidence"), result.get("trials")
    checks = {
        "worker_gpu": _worker_h20_gpu_evidence_ok(result.get("gpu")),
        "memory_guard": _passing_memory_guard_ok(result.get("memory_guard")),
        "persistent_model_instances": result.get("persistent_model_instances") is True,
        "training_mode_preserved": result.get("training_mode_preserved") is True,
        "single_tensor_final_output": result.get("final_output_kind") in {"single_tensor", "single_dense_tensor"},
        "execution_controls": result.get("execution_controls") == EXECUTION_CONTROLS,
        "state_mutation_contract": result.get("state_mutation_expected") is expected_mutation
        and result.get("state_mutation_observed") is expected_mutation,
        "registered_object_alias_evidence": isinstance(aliases, Mapping)
        and set(aliases) == {"module_attribute_paths", "tensor_attribute_paths"}
        and all(
            isinstance(value, list) and all(isinstance(path, str) and path for path in value)
            for value in aliases.values()
        ),
        "three_trials": isinstance(trials, list)
        and len(trials) == MIN_LIVENESS_TRIALS
        and all(
            isinstance(item, Mapping)
            and item.get("control_trace_output_exact") is True
            and item.get("control_trace_state_exact") is True
            and item.get("inputs_immutable") is True
            and item.get("single_tensor_output") is True
            and item.get("output_finite") is True
            for item in trials
        ),
    }
    return sorted(name for name, ok in checks.items() if not ok)


def _guard_failure_metadata_ok(record: Mapping[str, Any]) -> bool:
    """Match the exact failure envelope emitted by ``_CudaMemoryGuardFailure``."""

    reason = record.get("reason")
    return (
        isinstance(reason, str)
        and bool(reason)
        and record.get("failure_reasons") == [reason]
        and all(
            isinstance(record.get(field), str) and bool(record[field].strip())
            for field in ("error_type", "error", "traceback")
        )
    )


def _exception_evidence_reports_oom(value: Any) -> bool:
    return isinstance(value, Mapping) and (
        "OutOfMemory" in str(value.get("error_type")) or "out of memory" in str(value.get("error")).lower()
    )


def _capture_driver_h20_gpu_evidence(device: str, visible_devices: str) -> dict[str, str]:
    """Bind every row, including pre-worker failures, to the launcher H20."""

    if not visible_devices or os.environ.get("CUDA_VISIBLE_DEVICES") != visible_devices:
        raise ValueError("driver CUDA_VISIBLE_DEVICES does not match launcher binding")
    import torch

    if device != "cuda:0" or not torch.cuda.is_available():
        raise ValueError("driver H20 CUDA device is unavailable")
    torch.cuda.set_device(device)
    evidence = {
        "device": device,
        "name": str(torch.cuda.get_device_name(device)),
        "cuda_visible_devices": visible_devices,
    }
    if not _driver_h20_gpu_evidence_ok(evidence):
        raise ValueError("driver GPU is not an H20")
    return evidence


def _failure_record_is_attributable(record: Mapping[str, Any]) -> bool:
    """Reject invented failure records while allowing pre-worker failures."""

    reason, status, raw_status = record.get("reason"), record.get("status"), record.get("raw_status")
    base = (
        record.get("passed") is False
        and status in FAILURE_STATUSES
        and isinstance(raw_status, str)
        and bool(raw_status)
        and record.get("failure_stage") == "liveness"
        and isinstance(reason, str)
        and bool(reason)
        and record.get("failure_signature") == _failure_signature(record)
    )
    if not base:
        return False
    guard_statuses = {
        "cuda_out_of_memory",
        "device_memory_limit_exceeded",
        "gpu_memory_telemetry_error",
        "memory_guard_configuration_error",
        "error",
    }
    pseudo_pass = raw_status == "passed"
    if pseudo_pass:
        missing = record.get("authorization_evidence_missing")
        return (
            status == "protocol_error"
            and reason == "worker_reported_pass_without_authorized_runtime_evidence"
            and isinstance(missing, list)
            and bool(missing)
            and missing == _authorization_evidence_missing(record)
        )
    if raw_status in RAW_FAILURE_STATUS:
        if status != RAW_FAILURE_STATUS[raw_status]:
            return False
    elif status != "protocol_error" or reason != f"unexpected_worker_status:{raw_status}":
        return False
    if "gpu" in record and not _worker_h20_gpu_evidence_ok(record.get("gpu")):
        return False
    guard = record.get("memory_guard")
    if status in guard_statuses:
        if not _memory_guard_structure_ok(guard) or not _guard_failure_metadata_ok(record):
            return False
        nested_guard_errors = [guard.get("synchronize_error"), *(guard.get("telemetry_errors") or {}).values()]
        reports_oom = _exception_evidence_reports_oom(record) or any(
            _exception_evidence_reports_oom(value) for value in nested_guard_errors
        )
        if status in {"device_memory_limit_exceeded", "gpu_memory_telemetry_error", "error"} and reports_oom:
            return False
    if status == "memory_guard_configuration_error":
        telemetry_fields = (
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "current_allocated_bytes",
            "current_reserved_bytes",
        )
        if not (
            guard.get("enabled") is False
            and guard.get("configured") is False
            and guard.get("synchronized") is False
            and guard.get("within_limit") is False
            and guard.get("over_limit") is False
            and all(guard.get(field) == 0 for field in telemetry_fields)
            and guard.get("synchronize_error") is None
            and guard.get("telemetry_errors") in (None, {})
        ):
            return False
        if reason == "device_memory_limit_exceeds_device_capacity":
            return guard["total_device_bytes"] < guard["limit_bytes"] and guard.get("allocator_fraction") is None
        if reason == "device_memory_allocator_cap_not_configured":
            fraction = guard.get("allocator_fraction")
            return (
                guard["total_device_bytes"] >= guard["limit_bytes"]
                and isinstance(fraction, (int, float))
                and not isinstance(fraction, bool)
                and 0.0 < float(fraction) <= 1.0
                and math.isclose(
                    float(fraction), guard["limit_bytes"] / guard["total_device_bytes"], rel_tol=1e-12, abs_tol=0.0
                )
            )
        return False
    if status == "cuda_out_of_memory":
        return (
            guard.get("enabled") is True
            and guard.get("configured") is True
            and reason == "cuda_out_of_memory"
            and _exception_evidence_reports_oom(record)
        )
    if status == "device_memory_limit_exceeded":
        return (
            guard.get("enabled") is True
            and guard.get("configured") is True
            and guard.get("over_limit") is True
            and guard.get("within_limit") is False
            and reason == "device_memory_limit_exceeded"
        )
    if status == "gpu_memory_telemetry_error":
        if not (
            guard.get("enabled") is True
            and guard.get("configured") is True
            and guard.get("over_limit") is False
            and guard.get("within_limit") is False
        ):
            return False
        if reason == "cuda_synchronize_failed":
            return guard.get("synchronized") is False and isinstance(guard.get("synchronize_error"), Mapping)
        if reason == "gpu_memory_telemetry_failed":
            return (
                guard.get("synchronized") is True
                and guard.get("synchronize_error") is None
                and isinstance(guard.get("telemetry_errors"), Mapping)
                and bool(guard["telemetry_errors"])
            )
        return False
    if status == "error":
        return (
            guard.get("enabled") is True
            and guard.get("configured") is True
            and guard.get("synchronized") is True
            and guard.get("within_limit") is True
            and guard.get("over_limit") is False
            and guard.get("synchronize_error") is None
            and guard.get("telemetry_errors") in (None, {})
            and reason == "worker_exception"
        )
    if status in guard_statuses:
        return False
    return "memory_guard" not in record and "gpu" not in record


def _passed_result_is_authorized(result: Mapping[str, Any], task: Mapping[str, Any]) -> bool:
    return not _authorization_evidence_missing(result, task)


def _normalize_runtime_result(
    result: Mapping[str, Any], task: Mapping[str, Any]
) -> tuple[str, bool, str, dict[str, Any]]:
    """Map the shared worker result to this lane's explicit status contract."""

    normalized = dict(result)
    raw = str(normalized.get("status", "failed"))
    missing = _authorization_evidence_missing(normalized, task) if normalized.get("passed") is True else []
    if normalized.get("passed") is True and raw == "passed":
        if not missing:
            return "passed", True, raw, normalized
        return (
            "protocol_error",
            False,
            raw,
            {
                **normalized,
                "reason": "worker_reported_pass_without_authorized_runtime_evidence",
                "authorization_evidence_missing": missing,
            },
        )
    if normalized.get("passed") is True:
        return "protocol_error", False, raw, {**normalized, "reason": f"unexpected_worker_status:{raw}"}
    if raw in RAW_FAILURE_STATUS:
        return RAW_FAILURE_STATUS[raw], False, raw, normalized
    return "protocol_error", False, raw, {**normalized, "reason": f"unexpected_worker_status:{raw}"}


def _record_schema_ok(record: Mapping[str, Any], task: Mapping[str, Any]) -> bool:
    if any(
        record.get(field) != task.get(field)
        for field in ("uuid", "candidate_row_index", "template_id", "primary_family", "mode_behavior")
    ):
        return False
    if type(record.get("passed")) is not bool or not _driver_h20_gpu_evidence_ok(record.get("driver_gpu")):
        return False
    if record["passed"] is True:
        return (
            record.get("status") == "passed"
            and record.get("raw_status") == "passed"
            and record.get("failure_stage") is None
            and _passed_result_is_authorized(record, task)
        )
    return _failure_record_is_attributable(record)


def _record_binding_ok(
    record: Mapping[str, Any],
    evidence: Mapping[str, Any],
    source_binding: Mapping[str, Mapping[str, Any]],
    binding: str,
) -> bool:
    return (
        record.get("contract_version") == CONTRACT_VERSION
        and record.get("raw_runtime_core_contract_version") == runtime_core.CONTRACT_VERSION
        and record.get("validation_binding_sha256") == binding
        and record.get("binding_evidence") == evidence
        and record.get("candidates_sha256") == evidence.get("candidates_sha256")
        and record.get("manifest_sha256") == evidence.get("manifest_sha256")
        and record.get("allowlist_sha256") == evidence.get("allowlist_sha256")
        and record.get("validation_config") == evidence.get("validation_config")
        and all(record.get(f"{name}_sha256") == source.get("sha256") for name, source in source_binding.items())
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--uuid-file", type=Path, required=True)
    parser.add_argument("--reference-passed-binding", type=Path, required=True)
    parser.add_argument("--generator-module", default=DEFAULT_GENERATOR_MODULE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trials", type=int, default=MIN_LIVENESS_TRIALS)
    parser.add_argument("--seed", type=int, default=REQUIRED_LIVENESS_SEED)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--launcher-sha256", required=True)
    parser.add_argument("--launcher-source-path", type=Path, required=True)
    parser.add_argument("--driver-cuda-visible-devices", required=True)
    parser.add_argument("--execution-host", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    runtime_core._install_driver_signal_handlers()
    if (
        args.device != "cuda:0"
        or args.trials != MIN_LIVENESS_TRIALS
        or args.seed != REQUIRED_LIVENESS_SEED
        or args.timeout_seconds <= 0
    ):
        raise ValueError("require cuda:0, exactly 3 trials, seed 17, and a positive timeout")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count or not args.execution_host.strip():
        raise ValueError("invalid global shard identity or execution host")
    generator = _load_generator(args.generator_module)
    source_binding = _source_binding(
        generator, args.launcher_source_path, _require_sha256(args.launcher_sha256, label="launcher-sha256")
    )
    all_tasks = tasks(args.candidates, args.manifest, generator_module=args.generator_module)
    allowlist = _load_allowlist(args.uuid_file)
    known = {str(task["uuid"]) for task in all_tasks}
    if allowlist - known:
        raise ValueError("allowlist contains unknown UUID")
    reference_binding = args.reference_passed_binding.resolve()
    if not reference_binding.is_file():
        raise ValueError("reference-passed binding artifact is missing")
    reference_binding_pair = {"path": str(reference_binding), "sha256": _sha256_file(reference_binding)}
    selected = [
        task
        for task in all_tasks
        if str(task["uuid"]) in allowlist and int(task["candidate_row_index"]) % args.shard_count == args.shard_index
    ]
    if not selected:
        raise ValueError("global shard selection is empty")
    driver_gpu = _capture_driver_h20_gpu_evidence(args.device, args.driver_cuda_visible_devices)
    evidence = {
        "contract_version": CONTRACT_VERSION,
        "binding_version": RUN_BINDING_VERSION,
        **source_binding,
        "generator_module": args.generator_module,
        "total_rows": len(all_tasks),
        "execution_host": args.execution_host,
        "global_shard_index": args.shard_index,
        "global_shard_count": args.shard_count,
        "candidates_path": str(args.candidates.resolve()),
        "candidates_sha256": _sha256_file(args.candidates),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": _sha256_file(args.manifest),
        "allowlist_path": str(args.uuid_file.resolve()),
        "allowlist_sha256": _sha256_file(args.uuid_file),
        "reference_passed_binding": reference_binding_pair,
        "validation_config": {
            "device": args.device,
            "trials": args.trials,
            "seed": args.seed,
            "timeout_seconds": args.timeout_seconds,
            "max_device_memory_gib": runtime_core.MAX_DEVICE_MEMORY_GIB,
            "persistent_train_mode_models": True,
            "single_tensor_final_output": True,
            "execution_controls": EXECUTION_CONTROLS,
        },
    }
    binding = _canonical_sha256(evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: collections.Counter[str] = collections.Counter()
    executed = resumed = 0
    with args.output.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another liveness worker owns output:{args.output}") from exc
        handle.seek(0)
        prior: dict[str, Mapping[str, Any]] = {}
        selected_by_uuid = {str(t["uuid"]): t for t in selected}
        for number, line in enumerate(handle, 1):
            record = json.loads(line)
            uuid = str(record.get("uuid"))
            if (
                uuid not in selected_by_uuid
                or uuid in prior
                or not _record_binding_ok(record, evidence, source_binding, binding)
                or not _record_schema_ok(record, selected_by_uuid[uuid])
            ):
                raise ValueError(f"invalid resume record:{args.output}:{number}")
            prior[uuid] = record
        handle.seek(0, 2)
        for task in selected:
            uuid = str(task["uuid"])
            if uuid in prior:
                resumed += 1
                counts["passed" if prior[uuid]["passed"] else "failed"] += 1
                continue
            result = runtime_core._run_subprocess(
                {
                    **task,
                    "device": args.device,
                    "trials": args.trials,
                    "seed": args.seed,
                    "max_device_memory_gib": runtime_core.MAX_DEVICE_MEMORY_GIB,
                },
                args.timeout_seconds,
            )
            status, passed, raw, result = _normalize_runtime_result(result, task)
            record = {
                **result,
                "contract_version": CONTRACT_VERSION,
                "raw_runtime_core_contract_version": result.get("contract_version"),
                "status": status,
                "passed": passed,
                "raw_status": raw,
                "failure_stage": None if passed else "liveness",
                "driver_gpu": driver_gpu,
                "validation_binding_sha256": binding,
                "binding_evidence": evidence,
                **{f"{name}_sha256": source["sha256"] for name, source in source_binding.items()},
                "candidates_sha256": evidence["candidates_sha256"],
                "manifest_sha256": evidence["manifest_sha256"],
                "allowlist_sha256": evidence["allowlist_sha256"],
                "validation_config": evidence["validation_config"],
            }
            record["failure_signature"] = _failure_signature(record)
            if not _record_binding_ok(record, evidence, source_binding, binding) or not _record_schema_ok(
                record, task
            ):
                raise ValueError(f"runtime producer emitted an invalid liveness record:{uuid}:{raw}")
            _append(handle, record)
            executed += 1
            counts["passed" if passed else "failed"] += 1
    print(
        json.dumps(
            {
                "contract_version": CONTRACT_VERSION,
                "validation_binding_sha256": binding,
                "source_binding": source_binding,
                "execution_host": args.execution_host,
                "global_shard_index": args.shard_index,
                "global_shard_count": args.shard_count,
                "selected": len(selected),
                "executed": executed,
                "resumed": resumed,
                "passed": counts["passed"],
                "failed": counts["failed"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
