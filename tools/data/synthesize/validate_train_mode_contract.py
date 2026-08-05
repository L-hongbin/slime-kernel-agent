#!/usr/bin/env python3
"""GPU-smoke mode-variant references under the production KernelGYM contract.

Each selected parquet row is evaluated in its own subprocess.  The worker
constructs two identically seeded copies of the PyTorch reference and passes
them to KernelGYM's production ``run_and_check_correctness`` implementation.
Models stay in the requested mode and persist across all correctness trials;
the KernelGYM implementation must report that it reset the paired Torch RNG
before every reference and candidate forward.

This is a reference-contract smoke, not a CUDA-kernel evaluation: ATen fallback
detection and performance measurement are intentionally out of scope.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import inspect
import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import traceback
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

CONTRACT_VERSION = "kernelgym-reference-self-train-mode-v3"
RESULT_MARKER = "__KERNELGYM_TRAIN_MODE_RESULT__="
DEFAULT_TRIALS = 5
DEFAULT_SEED = 42
DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_MAX_DEVICE_MEMORY_GIB = 64.0
MAX_ALLOWED_DEVICE_MEMORY_GIB = 64.0
GIB_BYTES = 1024**3
SHA256_RE = re.compile(r"[0-9a-f]{64}")
KERNELGYM_EVALUATOR_FILES = {
    "correctness": "kernelgym/toolkit/kernelbench/correctness.py",
    "loading": "kernelgym/toolkit/kernelbench/loading.py",
    "exec_types": "kernelgym/toolkit/kernelbench/exec_types.py",
    "profiling": "kernelgym/toolkit/kernelbench/profiling.py",
    "config_init": "kernelgym/config/__init__.py",
    "config_settings": "kernelgym/config/settings.py",
}
KERNELGYM_HASH_FIELDS = tuple(f"{name}_sha256" for name in KERNELGYM_EVALUATOR_FILES) + ("evaluator_bundle_sha256",)
_DRIVER_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
_ACTIVE_WORKER_PROCESS: subprocess.Popen[str] | None = None


def _raise_driver_interrupt(signum: int, _frame: Any) -> None:
    """Turn inherited shell signals into cleanup-capable exceptions."""

    for handled_signal in _DRIVER_SIGNALS:
        signal.signal(handled_signal, signal.SIG_IGN)
    process = _ACTIVE_WORKER_PROCESS
    if process is not None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    raise KeyboardInterrupt(f"validation driver received signal {signum}")


def _install_driver_signal_handlers() -> None:
    for handled_signal in _DRIVER_SIGNALS:
        signal.signal(handled_signal, _raise_driver_interrupt)


@dataclasses.dataclass(frozen=True)
class DriverConfig:
    input_path: Path
    output_path: Path
    kernelgym_root: Path
    code_column: str = "reward_model.ground_truth"
    entry_point_column: str = "extra_info.entry_point"
    uuid_column: str = "extra_info.uuid"
    mode_class_column: str = "extra_info.v4.mode_class"
    expected_mode_class: str | None = "mode_variant"
    start_row: int = 0
    end_row: int | None = None
    limit: int | None = None
    shard_index: int = 0
    shard_count: int = 1
    device: str = "cuda:0"
    max_device_memory_gib: float = DEFAULT_MAX_DEVICE_MEMORY_GIB
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    trials: int = DEFAULT_TRIALS
    seed: int = DEFAULT_SEED
    training: bool = True
    resume: bool = True
    rerun_failures: bool = False
    launcher_sha256: str | None = None


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _nested(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _json_safe(value: Any, *, depth: int = 0) -> Any:
    """Convert evaluator metadata to bounded JSON without hiding failures."""

    if depth > 8:
        return "<maximum JSON depth reached>"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value if len(value) <= 20_000 else value[:20_000] + "<truncated>"
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item, depth=depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, depth=depth + 1) for item in value[:1_000]]
    return _json_safe(repr(value), depth=depth + 1)


def _exception_record(exc: BaseException) -> dict[str, str]:
    detail = str(exc)
    if len(detail) > 20_000:
        detail = detail[:20_000] + "<truncated>"
    trace = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    if len(trace) > 40_000:
        trace = trace[-40_000:]
    return {
        "error_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
        "error": detail,
        "traceback": trace,
    }


class _CudaMemoryGuardFailure(RuntimeError):
    """Carry fail-closed allocator evidence across the worker boundary."""

    def __init__(
        self,
        *,
        status: str,
        failure_reason: str,
        evidence: Mapping[str, Any],
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(failure_reason)
        self.status = status
        self.failure_reason = failure_reason
        self.evidence = dict(evidence)
        self.cause = cause

    def result_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "status": self.status,
            "passed": False,
            "failure_reasons": [self.failure_reason],
            "memory_guard": self.evidence,
        }
        if self.cause is not None:
            record.update(_exception_record(self.cause))
        else:
            record.update(_exception_record(self))
        return record


def _is_cuda_out_of_memory(exc: BaseException, *, torch: Any) -> bool:
    oom_types = tuple(
        oom_type
        for oom_type in (
            getattr(torch, "OutOfMemoryError", None),
            getattr(getattr(torch, "cuda", None), "OutOfMemoryError", None),
        )
        if isinstance(oom_type, type)
    )
    if oom_types and isinstance(exc, oom_types):
        return True
    return "out of memory" in str(exc).lower()


def _configure_cuda_memory_guard(*, torch: Any, device: Any, max_device_memory_gib: float) -> dict[str, Any]:
    """Set the PyTorch allocator cap before executing any untrusted task code."""

    requested_limit_bytes = int(max_device_memory_gib * GIB_BYTES)
    total_device_bytes = int(torch.cuda.get_device_properties(device).total_memory)
    evidence: dict[str, Any] = {
        "enabled": False,
        "configured": False,
        "synchronized": False,
        "max_device_memory_gib": max_device_memory_gib,
        "limit_bytes": requested_limit_bytes,
        "total_device_bytes": total_device_bytes,
        "allocator_fraction": None,
        "peak_allocated_bytes": 0,
        "peak_reserved_bytes": 0,
        "current_allocated_bytes": 0,
        "current_reserved_bytes": 0,
        "over_limit": False,
        "within_limit": False,
    }
    if requested_limit_bytes <= 0:
        raise _CudaMemoryGuardFailure(
            status="memory_guard_configuration_error",
            failure_reason="device_memory_limit_not_positive",
            evidence=evidence,
        )
    if requested_limit_bytes > total_device_bytes:
        raise _CudaMemoryGuardFailure(
            status="memory_guard_configuration_error",
            failure_reason="device_memory_limit_exceeds_device_capacity",
            evidence=evidence,
        )
    fraction = requested_limit_bytes / total_device_bytes
    evidence["allocator_fraction"] = fraction
    try:
        torch.cuda.set_per_process_memory_fraction(fraction, device=device)
        torch.cuda.reset_peak_memory_stats(device=device)
    except BaseException as exc:  # noqa: BLE001 - a missing cap must fail closed
        raise _CudaMemoryGuardFailure(
            status="memory_guard_configuration_error",
            failure_reason="device_memory_allocator_cap_not_configured",
            evidence=evidence,
            cause=exc,
        ) from exc
    evidence["configured"] = True
    evidence["enabled"] = True
    return evidence


def _finalize_cuda_memory_guard(*, torch: Any, device: Any, evidence: dict[str, Any]) -> BaseException | None:
    """Synchronize and populate allocator telemetry, returning any telemetry error."""

    first_error: BaseException | None = None
    try:
        torch.cuda.synchronize(device=device)
        evidence["synchronized"] = True
    except BaseException as exc:  # noqa: BLE001 - asynchronous CUDA errors are row failures
        evidence["synchronize_error"] = _exception_record(exc)
        first_error = exc

    measurements = {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved,
        "current_allocated_bytes": torch.cuda.memory_allocated,
        "current_reserved_bytes": torch.cuda.memory_reserved,
    }
    telemetry_errors: dict[str, Any] = {}
    for field, reader in measurements.items():
        try:
            evidence[field] = int(reader(device=device))
        except BaseException as exc:  # noqa: BLE001 - missing telemetry must fail closed
            telemetry_errors[field] = _exception_record(exc)
            if first_error is None:
                first_error = exc
    if telemetry_errors:
        evidence["telemetry_errors"] = telemetry_errors

    limit_bytes = int(evidence["limit_bytes"])
    observed_peaks = (
        evidence.get("peak_allocated_bytes"),
        evidence.get("peak_reserved_bytes"),
    )
    evidence["over_limit"] = any(isinstance(value, int) and value > limit_bytes for value in observed_peaks)
    evidence["within_limit"] = bool(
        evidence.get("enabled") is True
        and evidence.get("synchronized") is True
        and not telemetry_errors
        and evidence["over_limit"] is False
    )
    return first_error


@contextlib.contextmanager
def _cuda_memory_guard(*, torch: Any, device: Any, max_device_memory_gib: float) -> Iterator[dict[str, Any]]:
    """Cap, measure, and fail closed around all task-controlled CUDA work."""

    evidence = _configure_cuda_memory_guard(
        torch=torch,
        device=device,
        max_device_memory_gib=max_device_memory_gib,
    )
    caught: BaseException | None = None
    try:
        yield evidence
    except BaseException as exc:  # noqa: BLE001 - attach telemetry to every worker failure
        caught = exc

    telemetry_error = _finalize_cuda_memory_guard(torch=torch, device=device, evidence=evidence)
    oom_error = next(
        (exc for exc in (caught, telemetry_error) if exc is not None and _is_cuda_out_of_memory(exc, torch=torch)),
        None,
    )
    if oom_error is not None:
        raise _CudaMemoryGuardFailure(
            status="cuda_out_of_memory",
            failure_reason="cuda_out_of_memory",
            evidence=evidence,
            cause=oom_error,
        ) from oom_error
    if evidence.get("over_limit") is True:
        raise _CudaMemoryGuardFailure(
            status="device_memory_limit_exceeded",
            failure_reason="device_memory_limit_exceeded",
            evidence=evidence,
            cause=caught,
        ) from caught
    if telemetry_error is not None:
        raise _CudaMemoryGuardFailure(
            status="gpu_memory_telemetry_error",
            failure_reason=(
                "cuda_synchronize_failed"
                if evidence.get("synchronize_error") is not None
                else "gpu_memory_telemetry_failed"
            ),
            evidence=evidence,
            cause=telemetry_error,
        ) from telemetry_error
    if caught is not None:
        raise _CudaMemoryGuardFailure(
            status="error",
            failure_reason="worker_exception",
            evidence=evidence,
            cause=caught,
        ) from caught


def _git_commit(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    commit = result.stdout.strip()
    return commit if result.returncode == 0 and len(commit) == 40 else None


def kernelgym_contract_metadata(kernelgym_root: Path) -> dict[str, Any]:
    root = kernelgym_root.expanduser().resolve()
    paths = {name: root / relative for name, relative in KERNELGYM_EVALUATOR_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"KernelGYM contract files are missing: {missing}")
    file_hashes = {f"{name}_sha256": _sha256_file(path) for name, path in paths.items()}
    bundle_payload = {
        KERNELGYM_EVALUATOR_FILES[name]: file_hashes[f"{name}_sha256"] for name in sorted(KERNELGYM_EVALUATOR_FILES)
    }
    return {
        "root": str(root),
        "git_commit": _git_commit(root),
        **file_hashes,
        "evaluator_bundle_sha256": _sha256_bytes(
            json.dumps(bundle_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
    }


def contract_payload(
    config: DriverConfig,
    *,
    source_sha256: str,
    kernelgym: Mapping[str, Any],
) -> dict[str, Any]:
    launcher_sha256 = _require_sha256(config.launcher_sha256, label="launcher_sha256")
    validator_source_sha256 = _sha256_file(Path(__file__).resolve())
    return {
        "contract_version": CONTRACT_VERSION,
        "source_sha256": source_sha256,
        "validator_source_sha256": validator_source_sha256,
        "launcher_source_sha256": launcher_sha256,
        **{f"kernelgym_{name}": kernelgym[name] for name in KERNELGYM_HASH_FIELDS},
        "code_column": config.code_column,
        "entry_point_column": config.entry_point_column,
        "mode_class_column": config.mode_class_column,
        "expected_mode_class": config.expected_mode_class,
        "device": config.device,
        "max_device_memory_gib": config.max_device_memory_gib,
        "trials": config.trials,
        "seed": config.seed,
        "training": config.training,
        "paired_rng_seed_reset_required": True,
        "persistent_model_instances_required": True,
    }


def contract_fingerprint(config: DriverConfig, *, source_sha256: str, kernelgym: Mapping[str, Any]) -> str:
    contract = contract_payload(config, source_sha256=source_sha256, kernelgym=kernelgym)
    return _sha256_bytes(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def selected_row_indices(
    total_rows: int,
    *,
    start_row: int,
    end_row: int | None,
    shard_index: int,
    shard_count: int,
    limit: int | None,
) -> list[int]:
    if total_rows < 0:
        raise ValueError("total_rows must be non-negative")
    if start_row < 0:
        raise ValueError("start_row must be non-negative")
    resolved_end = total_rows if end_row is None else end_row
    if resolved_end < start_row or resolved_end > total_rows:
        raise ValueError(f"end_row must be in [{start_row}, {total_rows}], got {resolved_end}")
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must satisfy 0 <= shard_index < shard_count")
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    indices = [index for index in range(start_row, resolved_end) if index % shard_count == shard_index]
    return indices if limit is None else indices[:limit]


def _iter_rows(path: Path, indices: Sequence[int], *, columns: Sequence[str]) -> Iterator[tuple[int, dict[str, Any]]]:
    import pyarrow.parquet as pq

    if not indices:
        return
    wanted = iter(indices)
    next_index = next(wanted, None)
    global_index = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=64, columns=list(columns)):
        rows = batch.to_pylist()
        batch_end = global_index + len(rows)
        while next_index is not None and next_index < batch_end:
            yield next_index, rows[next_index - global_index]
            next_index = next(wanted, None)
        if next_index is None:
            return
        global_index = batch_end
    if next_index is not None:
        raise RuntimeError(f"parquet ended before selected row {next_index}")


def _row_identity(row_index: int, row: Mapping[str, Any], config: DriverConfig) -> dict[str, Any]:
    code = _nested(row, config.code_column)
    if not isinstance(code, str) or not code.strip():
        raise ValueError(f"row {row_index}: {config.code_column} is not non-empty text")
    entry_point = _nested(row, config.entry_point_column, "Model")
    if not isinstance(entry_point, str) or not entry_point.isidentifier():
        raise ValueError(f"row {row_index}: invalid entry point {entry_point!r}")
    uuid = _nested(row, config.uuid_column)
    mode_class = _nested(row, config.mode_class_column)
    if mode_class is not None and not isinstance(mode_class, str):
        raise ValueError(f"row {row_index}: invalid mode class {mode_class!r}")
    if config.expected_mode_class is not None and mode_class is None:
        raise ValueError(
            f"row {row_index}: missing required {config.mode_class_column}; "
            "use --expected-mode-class any only for an intentional non-v4 or cross-mode smoke"
        )
    if config.expected_mode_class is not None and mode_class != config.expected_mode_class:
        raise ValueError(
            f"row {row_index}: expected {config.mode_class_column}={config.expected_mode_class!r}, "
            f"got {mode_class!r}; use --expected-mode-class to override explicitly"
        )
    reference_sha256 = _sha256_bytes(code.encode("utf-8"))
    return {
        "row_index": row_index,
        "row_key": f"{row_index}:{reference_sha256}",
        "uuid": uuid if isinstance(uuid, str) and uuid else None,
        "reference_sha256": reference_sha256,
        "reference_code": code,
        "entry_point": entry_point,
        "mode_class": mode_class,
        "data_source": row.get("data_source"),
    }


def load_resume_results(path: Path, expected_fingerprint: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    results: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSONL: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: result must be a JSON object")
            if record.get("contract_fingerprint") != expected_fingerprint:
                raise ValueError(f"{path}:{line_number}: contract fingerprint differs; use a new output file")
            row_key = record.get("row_key")
            if not isinstance(row_key, str) or not row_key:
                raise ValueError(f"{path}:{line_number}: missing row_key")
            results[row_key] = record
    return results


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(_json_safe(record), sort_keys=True, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as handle:
        try:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - Linux is the supported runtime
            pass
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())


def validate_harness_result(result: Mapping[str, Any], *, trials: int, training: bool) -> tuple[bool, list[str]]:
    """Apply fail-closed checks to the production correctness result."""

    metadata = result.get("kernelgym_metadata")
    reasons: list[str] = []
    if result.get("kernelgym_correctness") is not True:
        reasons.append("kernelgym_correctness_not_true")
    if not isinstance(metadata, Mapping):
        reasons.append("missing_kernelgym_metadata")
        metadata = {}
    if metadata.get("correctness_forward_seed_reset_enabled") is not True:
        reasons.append("paired_forward_rng_not_confirmed")
    if metadata.get("correctness_trials_run") != trials:
        reasons.append("not_all_trials_ran")
    if metadata.get("correctness_trials") != f"({trials} / {trials})":
        reasons.append("not_all_trials_passed")
    if result.get("reference_forward_calls") != trials:
        reasons.append("reference_model_not_observed_on_every_trial")
    if result.get("identical_forward_calls") != trials:
        reasons.append("identical_model_not_observed_on_every_trial")
    if result.get("reference_training") is not training or result.get("identical_training") is not training:
        reasons.append("requested_model_mode_not_preserved")
    return not reasons, reasons


def validate_memory_guard_evidence(value: Any) -> list[str]:
    """Validate the materializer-facing memory audit required for a passing row."""

    if not isinstance(value, Mapping):
        return ["missing_memory_guard"]
    reasons: list[str] = []
    if value.get("enabled") is not True:
        reasons.append("memory_guard_not_enabled")
    limit_bytes = value.get("limit_bytes")
    if isinstance(limit_bytes, bool) or not isinstance(limit_bytes, int) or limit_bytes <= 0:
        reasons.append("invalid_memory_guard_limit")
        limit_bytes = None
    fraction = value.get("allocator_fraction")
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(float(fraction))
        or not 0.0 < float(fraction) <= 1.0
    ):
        reasons.append("invalid_memory_guard_allocator_fraction")
    for field in ("peak_allocated_bytes", "peak_reserved_bytes"):
        peak = value.get(field)
        if isinstance(peak, bool) or not isinstance(peak, int) or peak < 0:
            reasons.append(f"invalid_memory_guard_{field}")
        elif limit_bytes is not None and peak > limit_bytes:
            reasons.append(f"memory_guard_{field}_over_limit")
    if value.get("within_limit") is not True:
        reasons.append("memory_guard_not_within_limit")
    return reasons


def _run_worker_subprocess(payload: Mapping[str, Any], *, timeout_seconds: float) -> dict[str, Any]:
    global _ACTIVE_WORKER_PROCESS

    command = [sys.executable, str(Path(__file__).resolve()), "--_worker"]
    started = time.monotonic()
    serialized = json.dumps(payload, ensure_ascii=False)
    process: subprocess.Popen[str] | None = None
    try:
        prior_signal_mask = signal.pthread_sigmask(signal.SIG_BLOCK, _DRIVER_SIGNALS)
        try:
            process = subprocess.Popen(  # noqa: S603 - fixed interpreter/script command
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=True,
            )
            _ACTIVE_WORKER_PROCESS = process
        finally:
            # Deliver a pending stop only after the new process group is
            # registered and while this cleanup try is active.
            signal.pthread_sigmask(signal.SIG_SETMASK, prior_signal_mask)
        stdout, stderr = process.communicate(serialized, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        assert process is not None
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        return {
            "status": "timeout",
            "passed": False,
            "duration_seconds": time.monotonic() - started,
            "timeout_seconds": timeout_seconds,
            "worker_stdout_tail": stdout[-4_000:],
            "worker_stderr_tail": stderr[-12_000:],
        }
    except BaseException:  # Keep an interrupted shard driver from orphaning its row worker.
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
        raise
    finally:
        _ACTIVE_WORKER_PROCESS = None

    assert process is not None
    marked = [line[len(RESULT_MARKER) :] for line in stdout.splitlines() if line.startswith(RESULT_MARKER)]
    if process.returncode != 0 or len(marked) != 1:
        return {
            "status": "worker_protocol_error",
            "passed": False,
            "duration_seconds": time.monotonic() - started,
            "worker_returncode": process.returncode,
            "worker_result_markers": len(marked),
            "worker_stdout_tail": stdout[-4_000:],
            "worker_stderr_tail": stderr[-12_000:],
        }
    try:
        result = json.loads(marked[0])
    except json.JSONDecodeError as exc:
        return {
            "status": "worker_protocol_error",
            "passed": False,
            "duration_seconds": time.monotonic() - started,
            "error": f"invalid worker JSON: {exc}",
            "worker_stdout_tail": stdout[-4_000:],
            "worker_stderr_tail": stderr[-12_000:],
        }
    if not isinstance(result, dict):
        return {
            "status": "worker_protocol_error",
            "passed": False,
            "duration_seconds": time.monotonic() - started,
            "error": "worker result was not an object",
        }
    result["duration_seconds"] = time.monotonic() - started
    if stderr.strip():
        result["worker_stderr_tail"] = stderr[-12_000:]
    return result


def _prepare_init_inputs(raw: Any, *, torch: Any) -> tuple[str, Any]:
    if not isinstance(raw, (list, tuple, Mapping)):
        raise TypeError("get_init_inputs() must return list, tuple, or mapping")
    if isinstance(raw, Mapping):
        return "kwargs", dict(raw)
    init_inputs = list(raw)
    if (
        len(init_inputs) > 1
        and hasattr(init_inputs[0], "__len__")
        and not isinstance(init_inputs[0], (str, torch.Tensor))
        and len(init_inputs[0]) == 0
    ):
        kwargs = init_inputs[1]
        if not isinstance(kwargs, Mapping):
            raise TypeError("[[], kwargs] get_init_inputs convention requires a mapping")
        return "kwargs", kwargs
    return "args", init_inputs


def _evaluate_worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    kernelgym_root = Path(str(payload["kernelgym_root"])).resolve()
    expected_contract = payload["kernelgym_contract"]
    observed_contract = kernelgym_contract_metadata(kernelgym_root)
    for key in KERNELGYM_HASH_FIELDS:
        if observed_contract[key] != expected_contract[key]:
            raise RuntimeError(f"KernelGYM {key} changed between driver and worker")

    sys.path.insert(0, str(kernelgym_root))
    import torch
    from kernelgym.toolkit.kernelbench.correctness import run_and_check_correctness
    from kernelgym.toolkit.kernelbench.exec_types import set_seed
    from kernelgym.toolkit.kernelbench.loading import load_original_model_and_inputs

    correctness_module = Path(inspect.getfile(run_and_check_correctness)).resolve()
    if kernelgym_root not in correctness_module.parents:
        raise RuntimeError(f"imported KernelGYM from unexpected path: {correctness_module}")
    required_parameters = {
        "stop_on_first_failure",
        "max_wall_time_s",
        "pass_on_time_budget",
        "budget_min_pass_trials",
        "detect_aten_fallback",
    }
    missing_parameters = required_parameters - set(inspect.signature(run_and_check_correctness).parameters)
    if missing_parameters:
        raise RuntimeError(
            "KernelGYM correctness API is older than the required contract; "
            f"missing parameters: {sorted(missing_parameters)}"
        )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the row subprocess")
    device = torch.device(str(payload["device"]))
    if device.type != "cuda":
        raise ValueError(f"device must be CUDA, got {device}")
    torch.cuda.set_device(device)
    trials = int(payload["trials"])
    seed = int(payload["seed"])
    training = bool(payload["training"])
    max_device_memory_gib = float(payload["max_device_memory_gib"])
    code = payload["reference_code"]
    entry_point = payload["entry_point"]

    with _cuda_memory_guard(
        torch=torch,
        device=device,
        max_device_memory_gib=max_device_memory_gib,
    ) as memory_guard:
        # Pin environment-controlled branches of the production correctness code.
        os.environ["KERNELGYM_CORRECTNESS_GPU_INPUTS"] = "1"
        os.environ["KERNELGYM_CORRECTNESS_DISABLE_TF32"] = "1"
        os.environ.pop("KERNELGYM_CORRECTNESS_MAX_WALL_S", None)
        os.environ["KERNELGYM_CORRECTNESS_PASS_ON_BUDGET"] = "0"

        context: dict[str, Any] = {}
        Model, get_init_inputs, get_inputs = load_original_model_and_inputs(code, context, entry_point)
        set_seed(seed)
        init_kind, init_inputs = _prepare_init_inputs(get_init_inputs(), torch=torch)

        def build_model() -> Any:
            set_seed(seed)
            if init_kind == "args":
                return Model(*init_inputs)
            return Model(**init_inputs)

        with torch.no_grad():
            reference_model = build_model()
            identical_model = build_model()
        if not isinstance(reference_model, torch.nn.Module) or not isinstance(identical_model, torch.nn.Module):
            raise TypeError("entry point must construct torch.nn.Module instances")
        set_seed(seed)
        reference_model.train(training)
        set_seed(seed)
        identical_model.train(training)

        call_counts = {"reference": 0, "identical": 0}

        def count_reference(_module: Any, _inputs: Any) -> None:
            call_counts["reference"] += 1

        def count_identical(_module: Any, _inputs: Any) -> None:
            call_counts["identical"] += 1

        reference_hook = reference_model.register_forward_pre_hook(count_reference)
        identical_hook = identical_model.register_forward_pre_hook(count_identical)
        try:
            result = run_and_check_correctness(
                reference_model,
                identical_model,
                get_inputs,
                metadata={
                    "train_mode_validator_contract": CONTRACT_VERSION,
                    "train_mode_validator_training": training,
                    "train_mode_validator_persistent_instances": True,
                },
                num_correct_trials=trials,
                verbose=False,
                seed=seed,
                device=device,
                stop_on_first_failure=True,
                max_wall_time_s=None,
                pass_on_time_budget=False,
                budget_min_pass_trials=trials,
                detect_aten_fallback=False,
            )
        finally:
            reference_hook.remove()
            identical_hook.remove()

        raw_result = {
            "kernelgym_correctness": bool(result.correctness),
            "kernelgym_compiled": bool(result.compiled),
            "kernelgym_metadata": _json_safe(result.metadata),
            "reference_forward_calls": call_counts["reference"],
            "identical_forward_calls": call_counts["identical"],
            "reference_training": bool(reference_model.training),
            "identical_training": bool(identical_model.training),
            "persistent_model_instances": True,
            "gpu": {
                "device": str(device),
                "name": torch.cuda.get_device_name(device),
                "compute_capability": list(torch.cuda.get_device_capability(device)),
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
            },
        }
        passed, failure_reasons = validate_harness_result(raw_result, trials=trials, training=training)
        raw_result.update(
            {
                "status": "passed" if passed else "failed",
                "passed": passed,
                "failure_reasons": failure_reasons,
            }
        )

    raw_result["memory_guard"] = memory_guard
    memory_failure_reasons = validate_memory_guard_evidence(memory_guard)
    if memory_failure_reasons:
        raw_result["passed"] = False
        raw_result["status"] = "memory_guard_failed"
        raw_result.setdefault("failure_reasons", []).extend(memory_failure_reasons)
    return raw_result


def _worker_main() -> int:
    try:
        payload = json.load(sys.stdin)
        if not isinstance(payload, dict):
            raise TypeError("worker payload must be a JSON object")
        # Reference snippets sometimes print. Keep stdout reserved for exactly
        # one machine-readable result marker.
        with contextlib.redirect_stdout(sys.stderr):
            result = _evaluate_worker(payload)
    except _CudaMemoryGuardFailure as exc:
        result = exc.result_record()
    except BaseException as exc:  # noqa: BLE001 - row failures must become records
        result = {"status": "error", "passed": False, **_exception_record(exc)}
    print(RESULT_MARKER + json.dumps(_json_safe(result), sort_keys=True, ensure_ascii=False))
    return 0


def run(config: DriverConfig) -> dict[str, Any]:
    import pyarrow.parquet as pq

    input_path = config.input_path.expanduser().resolve()
    output_path = config.output_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if config.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if config.trials <= 0:
        raise ValueError("trials must be positive")
    if not math.isfinite(config.max_device_memory_gib) or config.max_device_memory_gib <= 0:
        raise ValueError("max_device_memory_gib must be finite and positive")
    if config.max_device_memory_gib > MAX_ALLOWED_DEVICE_MEMORY_GIB:
        raise ValueError(f"max_device_memory_gib must be <= the {MAX_ALLOWED_DEVICE_MEMORY_GIB:g}-GiB policy cap")
    launcher_sha256 = _require_sha256(config.launcher_sha256, label="launcher_sha256")
    validator_source_sha256 = _sha256_file(Path(__file__).resolve())

    parquet = pq.ParquetFile(input_path)
    top_level_columns = set(parquet.schema_arrow.names)
    selected_columns = sorted(
        {
            config.code_column.split(".", 1)[0],
            config.entry_point_column.split(".", 1)[0],
            config.uuid_column.split(".", 1)[0],
            config.mode_class_column.split(".", 1)[0],
            "data_source",
        }
    )
    missing = set(selected_columns) - top_level_columns
    if missing:
        raise ValueError(f"input parquet is missing top-level columns: {sorted(missing)}")
    indices = selected_row_indices(
        parquet.metadata.num_rows,
        start_row=config.start_row,
        end_row=config.end_row,
        shard_index=config.shard_index,
        shard_count=config.shard_count,
        limit=config.limit,
    )

    source_sha256 = _sha256_file(input_path)
    kernelgym = kernelgym_contract_metadata(config.kernelgym_root)
    bound_contract = contract_payload(config, source_sha256=source_sha256, kernelgym=kernelgym)
    fingerprint = _sha256_bytes(json.dumps(bound_contract, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    if output_path.exists() and not config.resume:
        raise FileExistsError(f"output exists and --no-resume was requested: {output_path}")
    prior = load_resume_results(output_path, fingerprint) if config.resume else {}

    counts = {"selected": len(indices), "executed": 0, "resumed": 0, "passed": 0, "failed": 0}
    for row_index, row in _iter_rows(input_path, indices, columns=selected_columns):
        try:
            identity = _row_identity(row_index, row, config)
        except Exception as exc:  # Schema errors are row failures; control-flow signals must propagate.
            # Missing reference text cannot derive a content hash, so retain a
            # stable index-scoped key and keep the run fail-closed.
            identity = {
                "row_index": row_index,
                "row_key": f"{row_index}:invalid-reference",
                "uuid": _nested(row, config.uuid_column),
                "reference_sha256": None,
                "data_source": row.get("data_source"),
            }
            previous = prior.get(identity["row_key"])
            if previous is not None and not (config.rerun_failures and previous.get("passed") is not True):
                counts["resumed"] += 1
                counts["failed"] += 1
                continue
            worker_result = {"status": "input_error", "passed": False, **_exception_record(exc)}
        else:
            previous = prior.get(identity["row_key"])
            if previous is not None and not (config.rerun_failures and previous.get("passed") is not True):
                counts["resumed"] += 1
                if previous.get("passed") is True:
                    counts["passed"] += 1
                else:
                    counts["failed"] += 1
                continue
            payload = {
                "kernelgym_root": str(Path(kernelgym["root"])),
                "kernelgym_contract": kernelgym,
                "reference_code": identity.pop("reference_code"),
                "entry_point": identity["entry_point"],
                "device": config.device,
                "max_device_memory_gib": config.max_device_memory_gib,
                "trials": config.trials,
                "seed": config.seed,
                "training": config.training,
            }
            worker_result = _run_worker_subprocess(payload, timeout_seconds=config.timeout_seconds)
            if worker_result.get("passed") is True:
                memory_failure_reasons = validate_memory_guard_evidence(worker_result.get("memory_guard"))
                if memory_failure_reasons:
                    worker_result["passed"] = False
                    worker_result["status"] = "memory_guard_failed"
                    worker_result.setdefault("failure_reasons", []).extend(memory_failure_reasons)

        record = {
            "contract_version": CONTRACT_VERSION,
            "contract_fingerprint": fingerprint,
            "contract_payload": bound_contract,
            "validator_source_sha256": validator_source_sha256,
            "launcher_source_sha256": launcher_sha256,
            "source_path": str(input_path),
            "source_sha256": source_sha256,
            "kernelgym": kernelgym,
            "trials": config.trials,
            "seed": config.seed,
            "training": config.training,
            "device": config.device,
            "max_device_memory_gib": config.max_device_memory_gib,
            **identity,
            **worker_result,
        }
        _append_jsonl(output_path, record)
        counts["executed"] += 1
        if record.get("passed") is True:
            counts["passed"] += 1
        else:
            counts["failed"] += 1

    counts.update(
        {
            "all_passed": counts["failed"] == 0,
            "input_path": str(input_path),
            "output_path": str(output_path),
            "contract_fingerprint": fingerprint,
            "shard_index": config.shard_index,
            "shard_count": config.shard_count,
        }
    )
    return counts


def _default_kernelgym_root() -> Path:
    configured = os.environ.get("KERNELGYM_ROOT")
    if configured:
        return Path(configured)
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root.parent / "KernelGYM-reward-only"


def _default_output(input_path: Path, shard_index: int, shard_count: int) -> Path:
    suffix = f".train_mode_contract.shard-{shard_index:02d}-of-{shard_count:02d}.jsonl"
    return input_path.with_suffix(suffix)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, help="V4 mode-variant parquet")
    parser.add_argument("--output", type=Path, help="Append-only JSONL result (default: beside input, per shard)")
    parser.add_argument("--kernelgym-root", type=Path, default=_default_kernelgym_root())
    parser.add_argument("--code-column", default="reward_model.ground_truth")
    parser.add_argument("--entry-point-column", default="extra_info.entry_point")
    parser.add_argument("--uuid-column", default="extra_info.uuid")
    parser.add_argument("--mode-class-column", default="extra_info.v4.mode_class")
    parser.add_argument(
        "--expected-mode-class",
        choices=("mode_variant", "mode_same_observed", "mode_inconclusive", "any"),
        default="mode_variant",
        help=(
            "Require this v4 mode class, including presence of the metadata. Use 'any' only for "
            "an intentional non-v4 or cross-mode smoke (default: mode_variant)."
        ),
    )
    parser.add_argument("--start-row", type=int, default=0, help="Inclusive global parquet row")
    parser.add_argument("--end-row", type=int, help="Exclusive global parquet row")
    parser.add_argument("--limit", type=int, help="Maximum selected rows after range and shard selection")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-device-memory-gib",
        type=float,
        default=DEFAULT_MAX_DEVICE_MEMORY_GIB,
        help=(
            "Hard per-worker PyTorch CUDA allocator cap in GiB; peak allocated/reserved "
            "memory is synchronized and audited (default: 64)."
        ),
    )
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--training",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep both reference instances in train mode (default: true)",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip rows already present under the exact same contract (default: true)",
    )
    parser.add_argument("--rerun-failures", action="store_true", help="On resume, retry prior non-passing rows")
    parser.add_argument(
        "--launcher-sha256",
        help="SHA-256 of the exact distributed launcher that invoked this validator (required by the driver)",
    )
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args._worker:
        return _worker_main()
    _install_driver_signal_handlers()
    if args.input is None:
        raise SystemExit("input parquet is required")
    output = args.output or _default_output(args.input, args.shard_index, args.shard_count)
    config = DriverConfig(
        input_path=args.input,
        output_path=output,
        kernelgym_root=args.kernelgym_root,
        code_column=args.code_column,
        entry_point_column=args.entry_point_column,
        uuid_column=args.uuid_column,
        mode_class_column=args.mode_class_column,
        expected_mode_class=None if args.expected_mode_class == "any" else args.expected_mode_class,
        start_row=args.start_row,
        end_row=args.end_row,
        limit=args.limit,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        device=args.device,
        max_device_memory_gib=args.max_device_memory_gib,
        timeout_seconds=args.timeout_seconds,
        trials=args.trials,
        seed=args.seed,
        training=args.training,
        resume=args.resume,
        rerun_failures=args.rerun_failures,
        launcher_sha256=args.launcher_sha256,
    )
    summary = run(config)
    print(json.dumps(summary, sort_keys=True, ensure_ascii=False))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
