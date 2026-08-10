#!/usr/bin/env python3
"""Fail-closed acceptance check for gathered shape-solver runtime shards.

``output_dir`` is only the directory from which evidence is read.  Shard log
summaries must still name the canonical production path under
``run_dir/h20/<phase>``; this lets the same check validate a staging copy
without weakening the run binding.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

SHA256_RE = re.compile(r"[0-9a-f]{64}")
CONTRACT_FILE_RE = re.compile(r"scheduler-contract-rank-(\d+)\.json")

REFERENCE_SCHEDULER_FIELDS = {
    "contract_version",
    "launcher_source_sha256",
    "validator_source_sha256",
    "machine_rank",
    "machine_count",
    "gpus_per_machine",
    "virtual_shards_per_gpu",
    "shard_count",
    "input_path",
    "input_sha256",
    "timeout_seconds",
    "expected_mode_class",
    "max_device_memory_gib",
    "idle_memory_mib",
}
REGION_SCHEDULER_FIELDS = {
    "contract_version",
    "launcher_source_sha256",
    "validator_source_sha256",
    "runtime_validation_source_sha256",
    "augment_prompt_tasks_source_sha256",
    "solve_shape_coverage_source_sha256",
    "validate_train_mode_contract_source_sha256",
    "machine_rank",
    "machine_count",
    "gpus_per_machine",
    "virtual_shards_per_gpu",
    "shard_count",
    "selected_path",
    "selected_sha256",
    "children_path",
    "children_sha256",
    "manifest_path",
    "manifest_sha256",
    "allowlist_path",
    "allowlist_sha256",
    "timeout_seconds",
    "trials",
    "seed",
    "idle_memory_mib",
}

REFERENCE_CONTRACT_VERSION = "reference_scheduler_contract_v1"
REFERENCE_RECORD_CONTRACT = "kernelgym-reference-self-train-mode-v3"
REGION_CONTRACT_VERSION = "shape_region_scheduler_contract_v1"
REGION_RECORD_CONTRACT = "shape_changed_region_liveness_v3"
REGION_PERTURBATION_CONTRACT = "seeded_bounded_non_affine_mix_v1"
REGION_BINDING_CONTRACT = "shape_changed_region_liveness_run_binding_v2"

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.synthesize.model_shape.validate_shape_region_liveness import _accepted_tasks  # noqa: E402
from tools.data.synthesize.validate_train_mode_contract import (  # noqa: E402
    validate_harness_result,
    validate_memory_guard_evidence,
)

REFERENCE_SOURCES = {
    "validator_source_sha256": REPO_ROOT / "tools/data/synthesize/validate_train_mode_contract.py",
}
LAUNCHER_SOURCES = {
    "reference": REPO_ROOT / "tools/data/synthesize/launch_reference_validation_shards.sh",
    "region": REPO_ROOT / "tools/data/synthesize/model_shape/launch_shape_region_validation.sh",
}

DEFAULT_PROFILE = {
    "machine_count": 4,
    "gpus_per_machine": 8,
    "virtual_shards_per_gpu": 8,
}
REFERENCE_PROFILE = {
    "timeout_seconds": 180.0,
    "expected_mode_class": "any",
    "max_device_memory_gib": 64.0,
    "idle_memory_mib": 64,
}
REGION_PROFILE = {
    "timeout_seconds": 600.0,
    "trials": 3,
    "seed": 17,
    "idle_memory_mib": 64,
}
REFERENCE_STATUSES = {
    "passed",
    "failed",
    "memory_guard_failed",
    "timeout",
    "worker_protocol_error",
    "input_error",
    "error",
    "memory_guard_configuration_error",
    "cuda_out_of_memory",
    "device_memory_limit_exceeded",
    "gpu_memory_telemetry_error",
}
REGION_STATUSES = {
    "passed",
    "rejected",
    "unsupported",
    "failed",
    "timeout",
    "worker_protocol_error",
    "error",
    "memory_guard_configuration_error",
    "cuda_out_of_memory",
    "device_memory_limit_exceeded",
    "gpu_memory_telemetry_error",
}
REFERENCE_PAYLOAD_FIELDS = {
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
REFERENCE_KERNELGYM_HASH_FIELDS = {
    "correctness_sha256",
    "loading_sha256",
    "exec_types_sha256",
    "profiling_sha256",
    "config_init_sha256",
    "config_settings_sha256",
    "evaluator_bundle_sha256",
}
GIB_BYTES = 1024**3
REGION_SOURCES = {
    "validator_source_sha256": REPO_ROOT / "tools/data/synthesize/model_shape/validate_shape_region_liveness.py",
    "runtime_validation_source_sha256": REPO_ROOT / "tools/data/cleaning/runtime_validation.py",
    "augment_prompt_tasks_source_sha256": REPO_ROOT / "tools/data/synthesize/augment_prompt_tasks.py",
    "solve_shape_coverage_source_sha256": REPO_ROOT / "tools/data/synthesize/model_shape/solve_shape_coverage.py",
    "validate_train_mode_contract_source_sha256": REPO_ROOT / "tools/data/synthesize/validate_train_mode_contract.py",
}


class VerificationError(RuntimeError):
    """The gathered evidence does not satisfy its frozen runtime contract."""


def _fail(message: str) -> None:
    raise VerificationError(message)


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _decode_json(text: str, *, context: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_object_without_duplicate_keys)
    except (json.JSONDecodeError, UnicodeError) as exc:
        _fail(f"{context}: invalid JSON: {exc}")


def _load_json(path: Path) -> Any:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        _fail(f"cannot read {path}: {exc}")
    return _decode_json(text, context=str(path))


def _jsonl(path: Path) -> Iterator[tuple[int, Mapping[str, Any]]]:
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as exc:
        _fail(f"cannot open {path}: {exc}")
    with handle:
        try:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    _fail(f"{path}:{line_number}: blank JSONL line")
                value = _decode_json(line, context=f"{path}:{line_number}")
                if not isinstance(value, Mapping):
                    _fail(f"{path}:{line_number}: record is not an object")
                yield line_number, value
        except UnicodeError as exc:
            _fail(f"{path}: invalid UTF-8: {exc}")


def _last_log_summary(path: Path) -> Mapping[str, Any]:
    last: tuple[int, str] | None = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    last = (line_number, line)
    except (OSError, UnicodeError) as exc:
        _fail(f"cannot read {path}: {exc}")
    if last is None:
        _fail(f"{path}: missing final JSON summary")
    line_number, line = last
    value = _decode_json(line, context=f"{path}:{line_number}")
    if not isinstance(value, Mapping):
        _fail(f"{path}:{line_number}: final summary is not an object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        _fail(f"cannot hash {path}: {exc}")
    return digest.hexdigest()


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_mapping(value: Any, *, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{context}: expected an object")
    return value


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], *, context: str) -> None:
    observed = set(value)
    if observed != expected:
        _fail(
            f"{context}: contract fields differ; "
            f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}"
        )


def _require_int(value: Any, *, context: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _fail(f"{context}: expected an integer >= {minimum}, found {value!r}")
    return value


def _require_number(value: Any, *, context: str, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(f"{context}: expected a number, found {value!r}")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0):
        _fail(f"{context}: invalid numeric value {value!r}")
    return result


def _require_sha256(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        _fail(f"{context}: expected a lowercase SHA-256 digest, found {value!r}")
    return value


def _require_canonical_file(
    contract: Mapping[str, Any],
    path_field: str,
    sha_field: str,
    *,
    expected_path: Path | None = None,
) -> Path:
    raw_path = contract.get(path_field)
    if not isinstance(raw_path, str) or not raw_path:
        _fail(f"scheduler contract has invalid {path_field}")
    path = Path(raw_path)
    if not path.is_absolute() or str(path.resolve()) != raw_path:
        _fail(f"scheduler contract {path_field} is not canonical: {raw_path!r}")
    if expected_path is not None and path != expected_path.resolve():
        _fail(
            f"scheduler contract {path_field} mismatch; "
            f"expected {str(expected_path.resolve())!r}, found {raw_path!r}"
        )
    if not path.is_file():
        _fail(f"scheduler contract input is missing: {path}")
    expected = _require_sha256(contract.get(sha_field), context=sha_field)
    observed = _sha256_file(path)
    if observed != expected:
        _fail(f"{path}: SHA-256 mismatch; expected {expected}, found {observed}")
    return path


def _verify_profile(
    contract: Mapping[str, Any],
    *,
    phase: str,
    expected_topology: Mapping[str, int],
) -> None:
    expected_shard_count = (
        expected_topology["machine_count"]
        * expected_topology["gpus_per_machine"]
        * expected_topology["virtual_shards_per_gpu"]
    )
    expected = {
        **expected_topology,
        "shard_count": expected_shard_count,
        **(REFERENCE_PROFILE if phase == "reference" else REGION_PROFILE),
    }
    mismatches = {
        field: {"expected": value, "found": contract.get(field)}
        for field, value in expected.items()
        if contract.get(field) != value
    }
    if mismatches:
        _fail(f"{phase} scheduler does not match the expected profile: {mismatches}")


def _load_scheduler_contracts(output_dir: Path, *, phase: str) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    expected_fields = REFERENCE_SCHEDULER_FIELDS if phase == "reference" else REGION_SCHEDULER_FIELDS
    expected_version = REFERENCE_CONTRACT_VERSION if phase == "reference" else REGION_CONTRACT_VERSION
    paths = sorted(output_dir.glob("scheduler-contract-rank-*.json"))
    if not paths:
        _fail(f"{output_dir}: no scheduler contracts")

    by_rank: dict[int, Mapping[str, Any]] = {}
    for path in paths:
        match = CONTRACT_FILE_RE.fullmatch(path.name)
        if match is None:
            _fail(f"unexpected scheduler contract filename: {path.name}")
        filename_rank = int(match.group(1))
        contract = _require_mapping(_load_json(path), context=str(path))
        _require_exact_keys(contract, expected_fields, context=str(path))
        if contract.get("contract_version") != expected_version:
            _fail(f"{path}: wrong contract_version")
        rank = _require_int(contract.get("machine_rank"), context=f"{path}:machine_rank")
        if rank != filename_rank or rank in by_rank:
            _fail(f"{path}: duplicate or mismatched machine rank")
        by_rank[rank] = contract

    first = next(iter(by_rank.values()))
    machine_count = _require_int(first.get("machine_count"), context="machine_count", minimum=1)
    if set(by_rank) != set(range(machine_count)):
        _fail(
            "scheduler contract ranks are incomplete: "
            f"expected={list(range(machine_count))}, observed={sorted(by_rank)}"
        )
    baseline = {key: value for key, value in first.items() if key != "machine_rank"}
    for rank, contract in by_rank.items():
        observed = {key: value for key, value in contract.items() if key != "machine_rank"}
        if observed != baseline:
            _fail(f"scheduler contract rank {rank} differs beyond machine_rank")

    gpus = _require_int(first.get("gpus_per_machine"), context="gpus_per_machine", minimum=1)
    virtual = _require_int(
        first.get("virtual_shards_per_gpu"),
        context="virtual_shards_per_gpu",
        minimum=1,
    )
    shard_count = _require_int(first.get("shard_count"), context="shard_count", minimum=1)
    if shard_count != machine_count * gpus * virtual:
        _fail("scheduler shard_count mismatch: " f"{shard_count} != {machine_count} * {gpus} * {virtual}")
    _require_int(first.get("idle_memory_mib"), context="idle_memory_mib")
    _require_number(first.get("timeout_seconds"), context="timeout_seconds")
    for hash_field in (key for key in first if key.endswith("_sha256")):
        _require_sha256(first.get(hash_field), context=hash_field)
    if phase == "reference":
        expected_mode = first.get("expected_mode_class")
        if not isinstance(expected_mode, str) or not expected_mode:
            _fail("reference expected_mode_class must be non-empty text")
        _require_number(first.get("max_device_memory_gib"), context="max_device_memory_gib")
    else:
        _require_int(first.get("trials"), context="trials", minimum=1)
        _require_int(first.get("seed"), context="seed")
    return first, [by_rank[rank] for rank in range(machine_count)]


def _verify_sources(
    output_dir: Path,
    contract: Mapping[str, Any],
    *,
    phase: str,
) -> None:
    launcher_hash = _require_sha256(contract.get("launcher_source_sha256"), context="launcher_source_sha256")
    launcher_archive = output_dir / "launcher_source.sh"
    if not launcher_archive.is_file():
        _fail(f"missing launcher archive: {launcher_archive}")
    if _sha256_file(launcher_archive) != launcher_hash:
        _fail(f"launcher archive hash differs from scheduler contract: {launcher_archive}")
    current_launcher = LAUNCHER_SOURCES[phase]
    if not current_launcher.is_file() or _sha256_file(current_launcher) != launcher_hash:
        _fail(f"launcher archive/contract does not equal current repository launcher: " f"{current_launcher}")
    sources = REFERENCE_SOURCES if phase == "reference" else REGION_SOURCES
    for field, path in sources.items():
        if not path.is_file():
            _fail(f"validation source is missing: {path}")
        observed = _sha256_file(path)
        expected = contract.get(field)
        if observed != expected:
            _fail(f"validation source hash mismatch for {field}: " f"expected {expected}, found {observed}")


def _shard_name(phase: str, shard_index: int, shard_count: int, suffix: str) -> str:
    # Bash printf widths are minimum widths.  Reference therefore uses 00..99
    # followed by 100..255, while region consistently uses a three-digit floor.
    width = 2 if phase == "reference" else 3
    return f"shard-{shard_index:0{width}d}-of-{shard_count:0{width}d}.{suffix}"


def _verify_shard_files(output_dir: Path, *, phase: str, shard_count: int) -> None:
    for suffix in ("jsonl", "log"):
        expected = {_shard_name(phase, shard_index, shard_count, suffix) for shard_index in range(shard_count)}
        observed = {path.name for path in output_dir.glob(f"shard-*.{suffix}")}
        if observed != expected:
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            _fail(f"{output_dir}: {suffix} shard family differs; " f"missing={missing[:20]}, extra={extra[:20]}")


def _nested(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _verify_h20(record: Mapping[str, Any], *, context: str, reference: bool) -> None:
    gpu = _require_mapping(record.get("gpu"), context=f"{context}:gpu")
    if gpu.get("name") != "NVIDIA H20" or gpu.get("device") != "cuda:0":
        _fail(f"{context}: passed evidence was not produced on NVIDIA H20 cuda:0")
    if reference and gpu.get("compute_capability") != [9, 0]:
        _fail(f"{context}: passed reference has wrong H20 compute capability")
    for field in ("torch_version", "torch_cuda_version") if reference else ("torch", "cuda"):
        if not isinstance(gpu.get(field), str) or not gpu[field]:
            _fail(f"{context}: passed H20 evidence is missing gpu.{field}")


def _verify_64gib_memory(record: Mapping[str, Any], *, context: str) -> None:
    memory_guard = record.get("memory_guard")
    reasons = validate_memory_guard_evidence(memory_guard)
    if reasons:
        _fail(f"{context}: invalid passed memory guard: {reasons}")
    assert isinstance(memory_guard, Mapping)
    expected = {
        "enabled": True,
        "configured": True,
        "synchronized": True,
        "max_device_memory_gib": 64.0,
        "limit_bytes": 64 * GIB_BYTES,
        "over_limit": False,
        "within_limit": True,
    }
    for field, value in expected.items():
        if memory_guard.get(field) != value:
            _fail(
                f"{context}: passed memory_guard.{field} mismatch; "
                f"expected {value!r}, found {memory_guard.get(field)!r}"
            )


def _verify_reference_passed_evidence(record: Mapping[str, Any], *, context: str) -> None:
    harness_passed, reasons = validate_harness_result(record, trials=5, training=True)
    if not harness_passed:
        _fail(f"{context}: passed reference fails harness validation: {reasons}")
    if record.get("kernelgym_compiled") is not True:
        _fail(f"{context}: passed reference lacks compiled KernelGYM evidence")
    if record.get("failure_reasons") != []:
        _fail(f"{context}: passed reference has failure_reasons")
    _verify_h20(record, context=context, reference=True)
    _verify_64gib_memory(record, context=context)


def _verify_region_passed_evidence(
    record: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    context: str,
) -> None:
    _verify_h20(record, context=context, reference=False)
    _verify_64gib_memory(record, context=context)
    if record.get("trials_completed") != 3:
        _fail(f"{context}: passed region evidence did not complete 3 trials")
    if record.get("reason") is not None:
        _fail(f"{context}: passed region evidence has a failure reason")
    for field in ("dead_slot_ids", "inconsistent_slot_ids"):
        if record.get(field, []) not in (None, []):
            _fail(f"{context}: passed region evidence has non-empty {field}")
    slot_ids = task.get("slot_ids")
    if not isinstance(slot_ids, list) or not slot_ids:
        _fail(f"{context}: accepted task has invalid slot_ids")
    declared_slots = task.get("slots")
    if (
        not isinstance(declared_slots, list)
        or [slot.get("slot_id") if isinstance(slot, Mapping) else None for slot in declared_slots] != slot_ids
    ):
        _fail(f"{context}: accepted task slots differ from declared slot_ids")
    effects = _require_mapping(record.get("slot_effects"), context=f"{context}:slot_effects")
    if set(effects) != set(slot_ids):
        _fail(f"{context}: passed region slot_effects keys differ from declared slots")
    for slot_id in slot_ids:
        if effects.get(slot_id) != [True, True, True]:
            _fail(f"{context}: slot {slot_id} lacks 3/3 positive effects")
    trials = record.get("trials")
    if not isinstance(trials, list) or len(trials) != 3:
        _fail(f"{context}: passed region evidence must contain exactly 3 trials")
    for trial_index, trial in enumerate(trials):
        trial = _require_mapping(trial, context=f"{context}:trial[{trial_index}]")
        expected_seed = 17 + trial_index * 10_007
        if trial.get("trial") != trial_index or trial.get("seed") != expected_seed:
            _fail(f"{context}: trial {trial_index} identity/seed mismatch")
        trial_slots = trial.get("slots")
        if (
            not isinstance(trial_slots, list)
            or [slot.get("slot_id") if isinstance(slot, Mapping) else None for slot in trial_slots] != slot_ids
        ):
            _fail(f"{context}: trial {trial_index} slot IDs differ from declared slots")
        for slot_index, (trial_slot, declared_slot) in enumerate(zip(trial_slots, declared_slots, strict=True)):
            assert isinstance(trial_slot, Mapping)
            assert isinstance(declared_slot, Mapping)
            slot_id = slot_ids[slot_index]
            for field in ("old_value", "new_value"):
                if trial_slot.get(field) != declared_slot.get(field):
                    _fail(f"{context}: trial {trial_index} slot {slot_id} " f"{field} differs from declared task")
            output_changed = trial_slot.get("output_changed")
            if (
                type(output_changed) is not bool
                or output_changed is not effects[slot_id][trial_index]
                or output_changed is not True
            ):
                _fail(f"{context}: trial {trial_index} slot {slot_id} " "output_changed contradicts slot_effects")
            for field in ("changed_region_elements", "expanded_region_elements"):
                value = trial_slot.get(field)
                if type(value) is not int or value <= 0:
                    _fail(f"{context}: trial {trial_index} slot {slot_id} " f"has invalid {field}")
            if trial_slot["changed_region_elements"] > trial_slot["expanded_region_elements"]:
                _fail(f"{context}: trial {trial_index} slot {slot_id} " "changed region exceeds expanded region")
            arguments = trial_slot.get("arguments")
            if not isinstance(arguments, list) or not arguments:
                _fail(f"{context}: trial {trial_index} slot {slot_id} " "has no argument evidence")
            for argument_index, argument in enumerate(arguments):
                argument = _require_mapping(
                    argument,
                    context=(f"{context}:trial[{trial_index}].slot[{slot_index}]" f".arguments[{argument_index}]"),
                )
                for field in ("changed_region_elements", "expanded_region_elements"):
                    value = argument.get(field)
                    if type(value) is not int or value <= 0:
                        _fail(
                            f"{context}: trial {trial_index} slot {slot_id} "
                            f"argument {argument_index} has invalid {field}"
                        )
                if argument["changed_region_elements"] > argument["expanded_region_elements"]:
                    _fail(
                        f"{context}: trial {trial_index} slot {slot_id} "
                        f"argument {argument_index} changed region exceeds expanded region"
                    )
                expanded_axes = argument.get("expanded_axes")
                if (
                    not isinstance(expanded_axes, list)
                    or not expanded_axes
                    or any(type(axis) is not int or axis < 0 for axis in expanded_axes)
                ):
                    _fail(
                        f"{context}: trial {trial_index} slot {slot_id} "
                        f"argument {argument_index} has invalid expanded_axes"
                    )


def _reference_identities(input_path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        _fail(f"pyarrow is required to inspect reference coverage: {exc}")
    try:
        parquet = pq.ParquetFile(input_path)
        top_level = set(parquet.schema_arrow.names)
    except Exception as exc:
        _fail(f"cannot inspect reference parquet {input_path}: {exc}")
    required = {"reward_model", "extra_info"}
    if not required.issubset(top_level):
        _fail(f"{input_path}: missing reference identity columns {sorted(required - top_level)}")

    identities: list[dict[str, Any]] = []
    try:
        for batch in parquet.iter_batches(batch_size=256, columns=["reward_model", "extra_info"]):
            for row in batch.to_pylist():
                row_index = len(identities)
                code = _nested(row, "reward_model.ground_truth")
                uuid = _nested(row, "extra_info.uuid")
                if isinstance(code, str) and code.strip():
                    reference_sha256 = hashlib.sha256(code.encode("utf-8")).hexdigest()
                    row_key = f"{row_index}:{reference_sha256}"
                else:
                    reference_sha256 = None
                    row_key = f"{row_index}:invalid-reference"
                identities.append(
                    {
                        "row_index": row_index,
                        "row_key": row_key,
                        "uuid": uuid if isinstance(uuid, str) and uuid else None,
                        "reference_sha256": reference_sha256,
                    }
                )
    except Exception as exc:
        _fail(f"cannot read reference identities from {input_path}: {exc}")
    if len(identities) != parquet.metadata.num_rows:
        _fail(f"{input_path}: parquet row count changed while reading")
    if not identities:
        _fail("reference input has no rows")
    return identities


def _check_summary_counts(
    summary: Mapping[str, Any],
    *,
    context: str,
    selected: int,
    passed: int,
) -> None:
    expected = {
        "selected": selected,
        "passed": passed,
        "failed": selected - passed,
    }
    for field, expected_value in expected.items():
        value = _require_int(summary.get(field), context=f"{context}:{field}")
        if value != expected_value:
            _fail(f"{context}: {field} mismatch; expected {expected_value}, found {value}")
    executed = _require_int(summary.get("executed"), context=f"{context}:executed")
    resumed = _require_int(summary.get("resumed"), context=f"{context}:resumed")
    if executed + resumed != selected:
        _fail(f"{context}: selected != executed + resumed")


def _verify_reference(
    run_dir: Path,
    output_dir: Path,
    canonical_dir: Path,
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, bool]]:
    input_path = _require_canonical_file(
        contract,
        "input_path",
        "input_sha256",
        expected_path=run_dir / "static" / "paired.parquet",
    )
    identities = _reference_identities(input_path)
    shard_count = int(contract["shard_count"])
    launcher_hash = str(contract["launcher_source_sha256"])
    validator_hash = str(contract["validator_source_sha256"])
    source_hash = str(contract["input_sha256"])
    expected_mode = contract["expected_mode_class"]
    expected_payload_mode = None if expected_mode == "any" else expected_mode

    common_payload: Mapping[str, Any] | None = None
    common_fingerprint: str | None = None
    pass_by_uuid: dict[str, bool] = {}
    total_passed = 0
    for shard_index in range(shard_count):
        name = _shard_name("reference", shard_index, shard_count, "jsonl")
        path = output_dir / name
        expected_rows = identities[shard_index::shard_count]
        observed_count = 0
        shard_passed = 0
        records = _jsonl(path)
        for expected in expected_rows:
            try:
                _, record = next(records)
            except StopIteration:
                _fail(f"{path}: row count mismatch; expected {len(expected_rows)}, " f"found {observed_count}")
            context = f"{path}:{observed_count + 1}"
            observed_count += 1
            for field in ("row_index", "row_key", "uuid", "reference_sha256"):
                if record.get(field) != expected[field]:
                    _fail(
                        f"{context}: {field} mismatch; " f"expected {expected[field]!r}, found {record.get(field)!r}"
                    )
            if record.get("contract_version") != REFERENCE_RECORD_CONTRACT:
                _fail(f"{context}: wrong reference record contract")
            if record.get("source_path") != str(input_path):
                _fail(f"{context}: source_path mismatch")
            if record.get("source_sha256") != source_hash:
                _fail(f"{context}: source_sha256 mismatch")
            if record.get("launcher_source_sha256") != launcher_hash:
                _fail(f"{context}: launcher hash mismatch")
            if record.get("validator_source_sha256") != validator_hash:
                _fail(f"{context}: validator hash mismatch")
            top_level_policy = {
                "trials": 5,
                "seed": 42,
                "training": True,
                "device": "cuda:0",
                "max_device_memory_gib": 64.0,
            }
            for field, expected_value in top_level_policy.items():
                if record.get(field) != expected_value:
                    _fail(f"{context}: top-level {field} policy mismatch")

            payload = _require_mapping(record.get("contract_payload"), context=f"{context}:contract_payload")
            _require_exact_keys(payload, REFERENCE_PAYLOAD_FIELDS, context=f"{context}:contract_payload")
            fingerprint = _require_sha256(
                record.get("contract_fingerprint"),
                context=f"{context}:contract_fingerprint",
            )
            if _canonical_json_sha256(payload) != fingerprint:
                _fail(f"{context}: contract fingerprint does not match payload")
            payload_links = {
                "contract_version": REFERENCE_RECORD_CONTRACT,
                "source_sha256": source_hash,
                "launcher_source_sha256": launcher_hash,
                "validator_source_sha256": validator_hash,
                "expected_mode_class": expected_payload_mode,
                "max_device_memory_gib": contract["max_device_memory_gib"],
                "code_column": "reward_model.ground_truth",
                "entry_point_column": "extra_info.entry_point",
                "mode_class_column": "extra_info.v4.mode_class",
                "device": "cuda:0",
                "trials": 5,
                "seed": 42,
                "training": True,
                "paired_rng_seed_reset_required": True,
                "persistent_model_instances_required": True,
            }
            for field, expected_value in payload_links.items():
                if payload.get(field) != expected_value:
                    _fail(f"{context}: contract_payload.{field} mismatch")
            if common_payload is None:
                common_payload = payload
                common_fingerprint = fingerprint
            elif payload != common_payload or fingerprint != common_fingerprint:
                _fail(f"{context}: reference runtime contracts differ across records")
            kernelgym = _require_mapping(record.get("kernelgym"), context=f"{context}:kernelgym")
            for field in REFERENCE_KERNELGYM_HASH_FIELDS:
                expected_hash = payload.get(f"kernelgym_{field}")
                _require_sha256(expected_hash, context=f"{context}:kernelgym_{field}")
                if kernelgym.get(field) != expected_hash:
                    _fail(f"{context}: KernelGYM {field} differs from contract payload")
            passed = record.get("passed")
            status = record.get("status")
            if type(passed) is not bool or not isinstance(status, str) or not status:
                _fail(f"{context}: invalid passed/status fields")
            if status not in REFERENCE_STATUSES:
                _fail(f"{context}: unknown reference status {status!r}")
            if (status == "passed") != passed:
                _fail(f"{context}: passed/status fields conflict")
            uuid = record.get("uuid")
            if not isinstance(uuid, str) or not uuid:
                _fail(f"{context}: full reference record has no UUID")
            if uuid in pass_by_uuid:
                _fail(f"{context}: duplicate reference UUID {uuid}")
            pass_by_uuid[uuid] = passed
            if passed:
                _verify_reference_passed_evidence(record, context=context)
            shard_passed += int(passed)
        try:
            extra_line, _ = next(records)
        except StopIteration:
            pass
        else:
            _fail(f"{path}:{extra_line}: unexpected extra JSONL record")
        log_name = _shard_name("reference", shard_index, shard_count, "log")
        log_path = output_dir / log_name
        summary = _last_log_summary(log_path)
        context = f"{log_path}:final"
        if summary.get("shard_index") != shard_index or summary.get("shard_count") != shard_count:
            _fail(f"{context}: shard identity mismatch")
        canonical_output = str((canonical_dir / name).resolve())
        if summary.get("output_path") != canonical_output:
            _fail(
                f"{context}: output_path mismatch; "
                f"expected {canonical_output!r}, found {summary.get('output_path')!r}"
            )
        if summary.get("input_path") != str(input_path):
            _fail(f"{context}: input_path mismatch")
        if summary.get("contract_fingerprint") != common_fingerprint:
            _fail(f"{context}: contract_fingerprint mismatch")
        if summary.get("all_passed") is not (shard_passed == len(expected_rows)):
            _fail(f"{context}: all_passed mismatch")
        _check_summary_counts(
            summary,
            context=context,
            selected=len(expected_rows),
            passed=shard_passed,
        )
        total_passed += shard_passed

    if common_payload is None or common_fingerprint is None:
        _fail("reference output has no contract-bound records")
    return (
        {
            "records": len(identities),
            "passed": total_passed,
            "failed": len(identities) - total_passed,
            "input_sha256": source_hash,
            "contract_fingerprint": common_fingerprint,
        },
        pass_by_uuid,
    )


def _read_allowlist(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        _fail(f"cannot read region allowlist {path}: {exc}")
    values: list[str] = []
    for line_number, line in enumerate(lines, start=1):
        value = line.strip()
        if not value:
            continue
        if any(character.isspace() for character in value):
            _fail(f"{path}:{line_number}: expected one UUID")
        values.append(value)
    duplicates = sorted(uuid for uuid, count in collections.Counter(values).items() if count > 1)
    if duplicates:
        _fail(f"{path}: duplicate UUIDs: {duplicates[:20]}")
    return values


def _region_tasks(
    selected_path: Path,
    children_path: Path,
    manifest_path: Path,
) -> list[Mapping[str, Any]]:
    try:
        tasks = _accepted_tasks(selected_path, children_path, manifest_path)
    except Exception as exc:
        _fail(f"cannot reconstruct exact region tasks: {type(exc).__name__}: {exc}")
    if not tasks:
        _fail("region manifest has no accepted tasks")
    return tasks


def _verified_canonical_reference_passes(
    run_dir: Path,
    *,
    expected_topology: Mapping[str, int],
) -> dict[str, bool]:
    output_dir = (run_dir / "h20" / "reference").resolve()
    if not output_dir.is_dir():
        _fail(f"canonical reference evidence is missing: {output_dir}")
    contract, _ = _load_scheduler_contracts(output_dir, phase="reference")
    _verify_profile(
        contract,
        phase="reference",
        expected_topology=expected_topology,
    )
    _verify_sources(output_dir, contract, phase="reference")
    _verify_shard_files(output_dir, phase="reference", shard_count=int(contract["shard_count"]))
    _, pass_by_uuid = _verify_reference(
        run_dir,
        output_dir,
        output_dir,
        contract,
    )
    return pass_by_uuid


def _region_binding(contract: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    binding = {
        "validation_binding_contract_version": REGION_BINDING_CONTRACT,
        "contract_version": REGION_RECORD_CONTRACT,
        "perturbation_contract_version": REGION_PERTURBATION_CONTRACT,
        "validator_source_sha256": contract["validator_source_sha256"],
        "launcher_source_sha256": contract["launcher_source_sha256"],
        "selected_sha256": contract["selected_sha256"],
        "children_sha256": contract["children_sha256"],
        "manifest_sha256": contract["manifest_sha256"],
        "validation_config": {
            "device": "cuda:0",
            "trials": contract["trials"],
            "seed": contract["seed"],
            "timeout_seconds": contract["timeout_seconds"],
            "max_device_memory_gib": 64.0,
        },
    }
    return binding, _canonical_json_sha256(binding)


def _verify_region(
    run_dir: Path,
    output_dir: Path,
    canonical_dir: Path,
    contract: Mapping[str, Any],
    *,
    expected_reference_topology: Mapping[str, int],
) -> dict[str, Any]:
    selected_path = _require_canonical_file(
        contract,
        "selected_path",
        "selected_sha256",
        expected_path=run_dir / "selected.parquet",
    )
    children_path = _require_canonical_file(
        contract,
        "children_path",
        "children_sha256",
        expected_path=run_dir / "static" / "children.parquet",
    )
    manifest_path = _require_canonical_file(
        contract,
        "manifest_path",
        "manifest_sha256",
        expected_path=run_dir / "static" / "manifest.json",
    )
    allowlist_path = _require_canonical_file(
        contract,
        "allowlist_path",
        "allowlist_sha256",
        expected_path=run_dir / "analysis" / "region_reference_both_pass_uuids.txt",
    )
    all_tasks = _region_tasks(selected_path, children_path, manifest_path)
    pass_by_uuid = _verified_canonical_reference_passes(
        run_dir,
        expected_topology=expected_reference_topology,
    )
    missing_reference_uuids = sorted(
        {
            str(task[field])
            for task in all_tasks
            for field in ("parent_uuid", "child_uuid")
            if task.get(field) not in pass_by_uuid
        }
    )
    if missing_reference_uuids:
        _fail("canonical reference evidence lacks region task UUIDs: " f"{missing_reference_uuids[:20]}")
    expected_allowlist = [
        str(task["child_uuid"])
        for task in all_tasks
        if pass_by_uuid[str(task["parent_uuid"])] and pass_by_uuid[str(task["child_uuid"])]
    ]
    observed_allowlist = _read_allowlist(allowlist_path)
    if observed_allowlist != expected_allowlist:
        first_difference = next(
            (
                index
                for index, (observed, expected) in enumerate(zip(observed_allowlist, expected_allowlist, strict=False))
                if observed != expected
            ),
            min(len(observed_allowlist), len(expected_allowlist)),
        )
        _fail(
            "region allowlist is not exactly parent+child both-pass in manifest order; "
            f"observed={len(observed_allowlist)}, expected={len(expected_allowlist)}, "
            f"first_difference={first_difference}"
        )
    allowlist_set = set(observed_allowlist)
    tasks = [task for task in all_tasks if task["child_uuid"] in allowlist_set]
    if not tasks:
        _fail("region allowlist selects no accepted tasks")
    shard_count = int(contract["shard_count"])
    binding, binding_sha256 = _region_binding(contract)
    launcher_hash = str(contract["launcher_source_sha256"])
    validator_hash = str(contract["validator_source_sha256"])

    total_passed = 0
    status_totals: collections.Counter[str] = collections.Counter()
    for shard_index in range(shard_count):
        name = _shard_name("region", shard_index, shard_count, "jsonl")
        path = output_dir / name
        expected_tasks = tasks[shard_index::shard_count]
        observed_count = 0
        shard_passed = 0
        shard_statuses: collections.Counter[str] = collections.Counter()
        records = _jsonl(path)
        for decision in expected_tasks:
            try:
                _, record = next(records)
            except StopIteration:
                _fail(f"{path}: row count mismatch; expected {len(expected_tasks)}, " f"found {observed_count}")
            context = f"{path}:{observed_count + 1}"
            observed_count += 1
            expected_fields = {
                "child_uuid": decision.get("child_uuid"),
                "parent_uuid": decision.get("parent_uuid"),
                "variant": decision.get("variant"),
                "slot_id": decision.get("slot_id"),
                "slot_ids": decision.get("slot_ids"),
                "slots": decision.get("slots"),
                "contract_version": REGION_RECORD_CONTRACT,
                "perturbation_contract_version": REGION_PERTURBATION_CONTRACT,
                "validation_binding_contract_version": REGION_BINDING_CONTRACT,
                "validation_binding_sha256": binding_sha256,
                "validator_source_sha256": validator_hash,
                "launcher_source_sha256": launcher_hash,
                "selected_sha256": contract["selected_sha256"],
                "children_sha256": contract["children_sha256"],
                "manifest_sha256": contract["manifest_sha256"],
                "validation_config": binding["validation_config"],
            }
            for field, expected_value in expected_fields.items():
                if record.get(field) != expected_value:
                    _fail(f"{context}: {field} mismatch; " f"expected {expected_value!r}, found {record.get(field)!r}")
            passed = record.get("passed")
            status = record.get("status")
            if type(passed) is not bool or not isinstance(status, str) or not status:
                _fail(f"{context}: invalid passed/status fields")
            if status not in REGION_STATUSES:
                _fail(f"{context}: unknown region status {status!r}")
            if (status == "passed") != passed:
                _fail(f"{context}: passed/status fields conflict")
            if passed:
                _verify_region_passed_evidence(record, decision, context=context)
            shard_passed += int(passed)
            shard_statuses[status] += 1
        try:
            extra_line, _ = next(records)
        except StopIteration:
            pass
        else:
            _fail(f"{path}:{extra_line}: unexpected extra JSONL record")
        log_name = _shard_name("region", shard_index, shard_count, "log")
        log_path = output_dir / log_name
        summary = _last_log_summary(log_path)
        context = f"{log_path}:final"
        expected_log_fields = {
            "contract_version": REGION_RECORD_CONTRACT,
            "perturbation_contract_version": REGION_PERTURBATION_CONTRACT,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "rows": len(expected_tasks),
            "launcher_source_sha256": launcher_hash,
            "validator_source_sha256": validator_hash,
            "validation_binding_sha256": binding_sha256,
        }
        for field, expected_value in expected_log_fields.items():
            if summary.get(field) != expected_value:
                _fail(f"{context}: {field} mismatch")
        canonical_output = str((canonical_dir / name).resolve())
        if summary.get("output_path") != canonical_output:
            _fail(
                f"{context}: output_path mismatch; "
                f"expected {canonical_output!r}, found {summary.get('output_path')!r}"
            )
        raw_counts = summary.get("counts")
        if not isinstance(raw_counts, Mapping) or dict(raw_counts) != dict(shard_statuses):
            _fail(f"{context}: raw status counts differ from JSONL records")
        _check_summary_counts(
            summary,
            context=context,
            selected=len(expected_tasks),
            passed=shard_passed,
        )
        total_passed += shard_passed
        status_totals.update(shard_statuses)

    return {
        "records": len(tasks),
        "passed": total_passed,
        "failed": len(tasks) - total_passed,
        "status_counts": dict(sorted(status_totals.items())),
        "allowlist_sha256": contract["allowlist_sha256"],
        "validation_binding_sha256": binding_sha256,
    }


def verify(
    run_dir: Path,
    phase: str,
    output_dir: Path | None,
    *,
    expected_topology: Mapping[str, int] = DEFAULT_PROFILE,
    expected_reference_topology: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    resolved_run = run_dir.expanduser().resolve()
    if not resolved_run.is_dir():
        _fail(f"run_dir is not a directory: {resolved_run}")
    canonical_dir = (resolved_run / "h20" / phase).resolve()
    resolved_output = output_dir.expanduser().resolve() if output_dir is not None else canonical_dir
    if not resolved_output.is_dir():
        _fail(f"output_dir is not a directory: {resolved_output}")
    reference_topology = expected_topology if expected_reference_topology is None else expected_reference_topology

    contract, contracts = _load_scheduler_contracts(resolved_output, phase=phase)
    _verify_profile(
        contract,
        phase=phase,
        expected_topology=expected_topology,
    )
    _verify_sources(resolved_output, contract, phase=phase)
    shard_count = int(contract["shard_count"])
    _verify_shard_files(resolved_output, phase=phase, shard_count=shard_count)
    if phase == "reference":
        details, _ = _verify_reference(resolved_run, resolved_output, canonical_dir, contract)
    else:
        details = _verify_region(
            resolved_run,
            resolved_output,
            canonical_dir,
            contract,
            expected_reference_topology=reference_topology,
        )
    return {
        "ok": True,
        "phase": phase,
        "run_dir": str(resolved_run),
        "output_dir": str(resolved_output),
        "canonical_output_dir": str(canonical_dir),
        "machine_count": contract["machine_count"],
        "gpus_per_machine": contract["gpus_per_machine"],
        "virtual_shards_per_gpu": contract["virtual_shards_per_gpu"],
        "scheduler_contracts": len(contracts),
        "shards": shard_count,
        "launcher_source_sha256": contract["launcher_source_sha256"],
        "validator_source_sha256": contract["validator_source_sha256"],
        "expected_topology": dict(expected_topology),
        "expected_reference_topology": (dict(reference_topology) if phase == "region" else None),
        **details,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("phase", choices=("reference", "region"))
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        help="staged/gathered evidence directory (default: RUN_DIR/h20/PHASE)",
    )
    parser.add_argument(
        "--expected-machine-count",
        type=int,
        default=DEFAULT_PROFILE["machine_count"],
    )
    parser.add_argument(
        "--expected-gpus-per-machine",
        type=int,
        default=DEFAULT_PROFILE["gpus_per_machine"],
    )
    parser.add_argument(
        "--expected-virtual-shards-per-gpu",
        type=int,
        default=DEFAULT_PROFILE["virtual_shards_per_gpu"],
    )
    parser.add_argument(
        "--expected-reference-machine-count",
        type=int,
        help=("region only: expected canonical reference machine count " "(default: --expected-machine-count)"),
    )
    parser.add_argument(
        "--expected-reference-gpus-per-machine",
        type=int,
        help=("region only: expected canonical reference GPUs per machine " "(default: --expected-gpus-per-machine)"),
    )
    parser.add_argument(
        "--expected-reference-virtual-shards-per-gpu",
        type=int,
        help=(
            "region only: expected canonical reference virtual shards per GPU "
            "(default: --expected-virtual-shards-per-gpu)"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    expected_topology = {
        "machine_count": args.expected_machine_count,
        "gpus_per_machine": args.expected_gpus_per_machine,
        "virtual_shards_per_gpu": args.expected_virtual_shards_per_gpu,
    }
    reference_overrides = {
        "machine_count": args.expected_reference_machine_count,
        "gpus_per_machine": args.expected_reference_gpus_per_machine,
        "virtual_shards_per_gpu": args.expected_reference_virtual_shards_per_gpu,
    }
    expected_reference_topology = {
        field: expected_topology[field] if override is None else override
        for field, override in reference_overrides.items()
    }
    invalid_topology = any(
        type(value) is not int or value <= 0
        for topology in (expected_topology, expected_reference_topology)
        for value in topology.values()
    )
    if invalid_topology:
        print(
            json.dumps(
                {
                    "ok": False,
                    "phase": args.phase,
                    "error": "expected topology values must be positive integers",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    try:
        summary = verify(
            args.run_dir,
            args.phase,
            args.output_dir,
            expected_topology=expected_topology,
            expected_reference_topology=expected_reference_topology,
        )
    except Exception as exc:  # Fail closed with one machine-readable line.
        summary = {
            "ok": False,
            "phase": args.phase,
            "run_dir": str(args.run_dir.expanduser().resolve()),
            "output_dir": str(
                args.output_dir.expanduser().resolve()
                if args.output_dir is not None
                else (args.run_dir.expanduser().resolve() / "h20" / args.phase)
            ),
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 1
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
