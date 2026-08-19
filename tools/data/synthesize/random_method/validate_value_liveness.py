#!/usr/bin/env python3
"""Validate runtime realization and output liveness for value-only children.

For every aligned parent/child pair, the worker generates inputs from the same
seed on one CUDA device and proves:

* input tree structure, tensor shape, dtype, device, stride, offset, layout,
  gradient flag, and pinned-storage state match;
* the post-``get_inputs`` CPU/CUDA RNG states are identical;
* exactly the declared ``randn`` factory outputs changed;
* changed tensors realize the assigned value-family support; and
* the changed values affect the returned output in every deterministic trial.

Each row runs in an isolated subprocess under the same 64-GiB allocator guard
as reference validation.  Output JSONL is append-resumable only under an exact
source/artifact/config binding.
"""

from __future__ import annotations

import argparse
import collections
import copy
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.cleaning.runtime_validation import (
    _all_finite,
    _build_model,
    _exec_ops_code,
    _invoke_model,
    _normalize_forward_inputs,
    _output_change_activity,
    _outputs_allclose,
    _restore_rng_states,
    _seed_torch,
    _snapshot_output,
    _snapshot_rng_states,
    _to_device,
)
from tools.data.synthesize.validate_train_mode_contract import _cuda_memory_guard, _CudaMemoryGuardFailure

CONTRACT_VERSION = "random_value_runtime_liveness_v2"
RUN_BINDING_VERSION = "random_value_runtime_liveness_binding_v2"
RESULT_MARKER = "__VALUE_LIVENESS_RESULT__="
MAX_DEVICE_MEMORY_GIB = 64.0
MAX_CANONICAL_CANARY_CANDIDATES = 5_000
MAX_COUNT_HISTOGRAM_BINS = 64
MIN_LIVENESS_TRIALS = 3
REQUIRED_LIVENESS_SEED = 17
POISSON_RATE_MINIMUM = 0.125
POISSON_RATE_MAXIMUM = 8.0
POISSON_RATE_HISTOGRAM_SCHEMA_VERSION = "poisson_rate_histogram_v1"
POISSON_RATE_HISTOGRAM_BINS = (
    (0.125, 0.25, False),
    (0.25, 0.5, False),
    (0.5, 1.0, False),
    (1.0, 2.0, False),
    (2.0, 4.0, False),
    (4.0, 8.0, True),
)
_DRIVER_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
_ACTIVE_WORKER_PROCESS: subprocess.Popen[str] | None = None


class UnsupportedCase(RuntimeError):
    """The experiment cannot establish a reliable value-liveness verdict."""


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


def _nested(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _exception_detail(exc: BaseException) -> str:
    return f"{type(exc).__name__}:{str(exc).replace(chr(10), ' ')[:2_000]}"


def _install_driver_signal_handlers() -> None:
    def handler(signum: int, _frame: Any) -> None:
        global _ACTIVE_WORKER_PROCESS
        process = _ACTIVE_WORKER_PROCESS
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        raise KeyboardInterrupt(f"value-liveness driver received signal {signum}")

    for handled in _DRIVER_SIGNALS:
        signal.signal(handled, handler)


def _generate_inputs_on_device(namespace: Mapping[str, Any], device: Any, torch: Any) -> Any:
    previous = torch.get_default_device()
    torch.set_default_device(device)
    try:
        return namespace["get_inputs"]()
    finally:
        torch.set_default_device(previous)


def _rng_states_equal(left: Any, right: Any, torch: Any) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
    numpy = sys.modules.get("numpy")
    if numpy is not None and isinstance(left, numpy.ndarray) and isinstance(right, numpy.ndarray):
        return bool(numpy.array_equal(left, right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(_rng_states_equal(left[key], right[key], torch) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _rng_states_equal(a, b, torch) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _maximum_consecutive_integer(dtype: Any, torch: Any) -> int:
    """Return the largest integer up to which every integer is representable."""

    return int(2.0 / torch.finfo(dtype).eps)


def _count_histogram(tensor: Any, torch: Any) -> tuple[int, dict[str, Any]]:
    """Build bounded, exact-bin evidence for a validated count tensor."""

    values, frequencies = torch.unique(tensor, sorted=True, return_counts=True)
    unique_count = int(values.numel())
    recorded_bins = min(unique_count, MAX_COUNT_HISTOGRAM_BINS)
    value_frequencies = {
        str(int(value.item())): int(frequency.item())
        for value, frequency in zip(values[:recorded_bins], frequencies[:recorded_bins], strict=True)
    }
    recorded_frequency = sum(value_frequencies.values())
    omitted_frequency = int(frequencies[recorded_bins:].sum().item()) if recorded_bins < unique_count else 0
    if recorded_frequency + omitted_frequency != tensor.numel():
        raise UnsupportedCase("poisson_count_histogram_frequency_mismatch")
    return unique_count, {
        "value_frequencies": value_frequencies,
        "truncated": recorded_bins < unique_count,
        "recorded_bins": recorded_bins,
        "omitted_bins": unique_count - recorded_bins,
        "omitted_frequency": omitted_frequency,
        "omitted_value_minimum": int(values[recorded_bins].item()) if recorded_bins < unique_count else None,
        "omitted_value_maximum": int(values[-1].item()) if recorded_bins < unique_count else None,
    }


def _poisson_rate_summary(parent: Any, torch: Any, path: str) -> dict[str, Any]:
    """Recompute the solver's Poisson rates and summarize fixed bins."""

    rates = torch.clamp(
        torch.nn.functional.softplus(parent),
        min=POISSON_RATE_MINIMUM,
        max=POISSON_RATE_MAXIMUM,
    )
    if not bool(torch.isfinite(rates).all().item()):
        raise UnsupportedCase(f"poisson_rate_non_finite:{path}")
    minimum = float(rates.min().item())
    maximum = float(rates.max().item())
    if minimum < POISSON_RATE_MINIMUM or maximum > POISSON_RATE_MAXIMUM:
        raise UnsupportedCase(f"poisson_rate_support_violation:{path}:{minimum}:{maximum}")
    histogram: list[dict[str, Any]] = []
    for lower, upper, upper_inclusive in POISSON_RATE_HISTOGRAM_BINS:
        in_bin = torch.logical_and(rates >= lower, rates <= upper if upper_inclusive else rates < upper)
        histogram.append(
            {
                "lower": lower,
                "upper": upper,
                "lower_inclusive": True,
                "upper_inclusive": upper_inclusive,
                "frequency": int(in_bin.sum().item()),
            }
        )
    total_frequency = sum(item["frequency"] for item in histogram)
    if total_frequency != rates.numel():
        raise UnsupportedCase(f"poisson_rate_histogram_frequency_mismatch:{path}:{total_frequency}:{rates.numel()}")
    return {
        "mapping": "clamp(softplus(parent),0.125,8.0)",
        "minimum": minimum,
        "maximum": maximum,
        "mean": float(rates.to(dtype=torch.float64).mean().item()),
        "histogram_schema_version": POISSON_RATE_HISTOGRAM_SCHEMA_VERSION,
        "histogram": histogram,
        "histogram_total_frequency": total_frequency,
    }


def _compare_input_tree(
    parent: Any,
    child: Any,
    *,
    path: str,
    family: str,
    torch: Any,
) -> tuple[int, list[dict[str, Any]]]:
    if isinstance(parent, torch.Tensor) or isinstance(child, torch.Tensor):
        if not isinstance(parent, torch.Tensor) or not isinstance(child, torch.Tensor):
            raise UnsupportedCase(f"tensor_structure_mismatch:{path}")
        structural = (
            tuple(parent.shape),
            parent.dtype,
            parent.device,
            parent.layout,
            tuple(parent.stride()),
            parent.storage_offset(),
            parent.requires_grad,
            parent.is_pinned(),
        )
        child_structural = (
            tuple(child.shape),
            child.dtype,
            child.device,
            child.layout,
            tuple(child.stride()),
            child.storage_offset(),
            child.requires_grad,
            child.is_pinned(),
        )
        if structural != child_structural:
            raise UnsupportedCase(f"tensor_metadata_changed:{path}:{structural!r}:{child_structural!r}")
        changed = not torch.equal(parent, child)
        if not changed:
            return 0, []
        if not child.dtype.is_floating_point or child.dtype.is_complex:
            raise UnsupportedCase(f"changed_tensor_not_real_floating:{path}:{child.dtype}")
        if not bool(torch.isfinite(child).all().item()):
            raise UnsupportedCase(f"changed_tensor_non_finite:{path}")
        minimum = float(child.min().item())
        maximum = float(child.max().item())
        unique_count = None
        count_histogram = None
        rate_summary = None
        if family == "uniform_01":
            if minimum < 0.0 or maximum >= 1.0:
                raise UnsupportedCase(f"uniform_01_support_violation:{path}:{minimum}:{maximum}")
        elif family == "signed_uniform":
            if minimum < -1.0 or maximum >= 1.0:
                raise UnsupportedCase(f"signed_uniform_support_violation:{path}:{minimum}:{maximum}")
        elif family == "poisson_counts":
            if minimum < 0.0 or not bool((child == torch.floor(child)).all().item()):
                raise UnsupportedCase(f"poisson_counts_support_violation:{path}:{minimum}:{maximum}")
            unique_count, count_histogram = _count_histogram(child, torch)
            rate_summary = _poisson_rate_summary(parent, torch, path)
        elif family == "multinomial_categories":
            if child.ndim < 1 or child.shape[-1] < 2:
                raise UnsupportedCase(f"multinomial_invalid_category_axis:{path}:{tuple(child.shape)}")
            maximum_exact = _maximum_consecutive_integer(child.dtype, torch)
            if child.shape[-1] - 1 > maximum_exact:
                raise UnsupportedCase(
                    f"multinomial_category_id_not_exactly_representable_in_dtype:"
                    f"{path}:{child.dtype}:{child.shape[-1] - 1}:{maximum_exact}"
                )
            if minimum < 0.0 or maximum >= child.shape[-1] or not bool((child == torch.floor(child)).all().item()):
                raise UnsupportedCase(f"multinomial_support_violation:{path}:{minimum}:{maximum}:{child.shape[-1]}")
        else:
            raise UnsupportedCase(f"unknown_assigned_family:{family}")
        parent_finite = int(torch.isfinite(parent).sum().item())
        return 1, [
            {
                "path": path,
                "shape": list(child.shape),
                "dtype": str(child.dtype),
                "stride": list(child.stride()),
                "storage_offset": child.storage_offset(),
                "is_pinned": child.is_pinned(),
                "numel": child.numel(),
                "parent_finite_elements": parent_finite,
                "minimum": minimum,
                "maximum": maximum,
                "unique_count": unique_count,
                "count_histogram": count_histogram,
                "rate_summary": rate_summary,
            }
        ]
    if isinstance(parent, Mapping) or isinstance(child, Mapping):
        if not isinstance(parent, Mapping) or not isinstance(child, Mapping) or list(parent) != list(child):
            raise UnsupportedCase(f"mapping_structure_mismatch:{path}")
        changed = 0
        records: list[dict[str, Any]] = []
        for key in parent:
            count, nested_records = _compare_input_tree(
                parent[key], child[key], path=f"{path}.{key}", family=family, torch=torch
            )
            changed += count
            records.extend(nested_records)
        return changed, records
    if isinstance(parent, (list, tuple)) or isinstance(child, (list, tuple)):
        if type(parent) is not type(child) or len(parent) != len(child):
            raise UnsupportedCase(f"sequence_structure_mismatch:{path}")
        changed = 0
        records = []
        for index, (left, right) in enumerate(zip(parent, child, strict=True)):
            count, nested_records = _compare_input_tree(
                left, right, path=f"{path}[{index}]", family=family, torch=torch
            )
            changed += count
            records.extend(nested_records)
        return changed, records
    if parent != child:
        raise UnsupportedCase(f"non_tensor_input_changed:{path}:{parent!r}:{child!r}")
    return 0, []


def _run_arm(model: Any, inputs: Sequence[Any], rng_state: Mapping[str, Any]) -> Any:
    import torch

    trial_model = copy.deepcopy(model)
    try:
        _restore_rng_states(rng_state)
        with torch.no_grad():
            output = _snapshot_output(_invoke_model(trial_model, inputs))
        if not _all_finite(output):
            raise UnsupportedCase("non_finite_output")
        return output
    finally:
        del trial_model


def _evaluate_without_guard(payload: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    device = torch.device(str(payload["device"]))
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError(f"CUDA device required, got {device}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parent_namespace = _exec_ops_code(
        str(payload["parent_code"]), entry_point=str(payload["entry_point"]), device=str(device)
    )
    child_namespace = _exec_ops_code(
        str(payload["child_code"]), entry_point=str(payload["entry_point"]), device=str(device)
    )
    trials = int(payload["trials"])
    seed = int(payload["seed"])
    family = str(payload["assigned_family"])
    expected_changed = int(payload["transformed_factory_count"])
    original_rng = _snapshot_rng_states(str(device))
    trial_records: list[dict[str, Any]] = []
    try:
        for trial in range(trials):
            trial_seed = seed + 10_007 * trial
            _seed_torch(trial_seed, str(device))
            parent_raw = _generate_inputs_on_device(parent_namespace, device, torch)
            parent_after_rng = _snapshot_rng_states(str(device))
            _seed_torch(trial_seed, str(device))
            child_raw = _generate_inputs_on_device(child_namespace, device, torch)
            child_after_rng = _snapshot_rng_states(str(device))
            if not _rng_states_equal(parent_after_rng, child_after_rng, torch):
                raise UnsupportedCase("post_get_inputs_rng_state_changed")
            changed_tensors, tensor_records = _compare_input_tree(
                parent_raw, child_raw, path="inputs", family=family, torch=torch
            )
            if changed_tensors != expected_changed:
                raise UnsupportedCase(f"changed_tensor_count_mismatch:{changed_tensors}:{expected_changed}")
            parent_inputs = _normalize_forward_inputs(parent_raw, str(device))
            child_inputs = _normalize_forward_inputs(child_raw, str(device))
            del parent_raw, child_raw

            _seed_torch(trial_seed + 1, str(device))
            init_inputs = child_namespace["get_init_inputs"]()
            init_inputs = [] if init_inputs is None else init_inputs
            if not isinstance(init_inputs, (list, tuple)):
                raise UnsupportedCase("get_init_inputs_is_not_list_or_tuple")
            init_inputs = _to_device(list(init_inputs), str(device))
            base_model = _build_model(child_namespace["__entry_point__"], init_inputs)
            if not isinstance(base_model, torch.nn.Module):
                raise UnsupportedCase("entry_point_did_not_build_torch_module")
            base_model = base_model.to(device)
            base_model.train(True)
            del init_inputs
            forward_rng = _snapshot_rng_states(str(device))
            parent_output = _run_arm(base_model, parent_inputs, forward_rng)
            parent_control = _run_arm(base_model, parent_inputs, forward_rng)
            if not _outputs_allclose(parent_output, parent_control, rtol=0.0, atol=0.0):
                raise UnsupportedCase("unchanged_parent_control_is_not_exact")
            child_output = _run_arm(base_model, child_inputs, forward_rng)
            output_changed = not _outputs_allclose(parent_output, child_output, rtol=0.0, atol=0.0)
            activity = _output_change_activity(parent_output, child_output, rtol=0.0, atol=0.0)
            trial_records.append(
                {
                    "trial": trial,
                    "seed": trial_seed,
                    "changed_tensors": changed_tensors,
                    "tensors": tensor_records,
                    "output_changed": output_changed,
                    "changed_output_elements": activity[0],
                    "output_elements": activity[1],
                    "max_changed_output_fraction": activity[2],
                }
            )
            del parent_output, parent_control, child_output, parent_inputs, child_inputs, base_model
            torch.cuda.empty_cache()
    finally:
        _restore_rng_states(original_rng)
    effects = [bool(item["output_changed"]) for item in trial_records]
    if not all(effects):
        return {
            "status": "rejected",
            "passed": False,
            "reason": "value_intervention_output_effect_not_consistent",
            "output_effects": effects,
            "trials": trial_records,
        }
    return {
        "status": "passed",
        "passed": True,
        "reason": None,
        "output_effects": effects,
        "trials": trial_records,
        "gpu": {
            "device": str(device),
            "name": torch.cuda.get_device_name(device),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
        },
    }


def _evaluate(payload: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    device = torch.device(str(payload["device"]))
    torch.cuda.set_device(device)
    with _cuda_memory_guard(torch=torch, device=device, max_device_memory_gib=MAX_DEVICE_MEMORY_GIB) as guard:
        try:
            result = _evaluate_without_guard(payload)
        except UnsupportedCase as exc:
            result = {"status": "unsupported", "passed": False, "reason": _exception_detail(exc)}
    result["memory_guard"] = guard
    return result


def _identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: payload.get(key)
        for key in ("candidate_row_index", "child_uuid", "parent_uuid", "assigned_family", "transformed_factory_count")
    }


def _worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = _evaluate(payload)
    except _CudaMemoryGuardFailure as exc:
        result = exc.result_record()
        result["reason"] = exc.failure_reason
    except UnsupportedCase as exc:
        result = {"status": "unsupported", "passed": False, "reason": _exception_detail(exc)}
    except BaseException as exc:  # noqa: BLE001 - row failures are evidence
        result = {"status": "failed", "passed": False, "reason": _exception_detail(exc)}
    return {
        "contract_version": CONTRACT_VERSION,
        **_identity(payload),
        **result,
        "duration_seconds": time.monotonic() - started,
    }


def _run_subprocess(payload: Mapping[str, Any], timeout_seconds: float) -> dict[str, Any]:
    global _ACTIVE_WORKER_PROCESS

    process: subprocess.Popen[str] | None = None
    started = time.monotonic()
    try:
        process = subprocess.Popen(  # noqa: S603 - fixed interpreter/script
            [sys.executable, str(Path(__file__).resolve()), "--_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        _ACTIVE_WORKER_PROCESS = process
        stdout, stderr = process.communicate(json.dumps(payload), timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        assert process is not None
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        return {
            "contract_version": CONTRACT_VERSION,
            **_identity(payload),
            "status": "timeout",
            "passed": False,
            "reason": f"row_timeout_after_{timeout_seconds:g}_seconds",
            "duration_seconds": time.monotonic() - started,
            "worker_stdout_tail": stdout[-2_000:],
            "worker_stderr_tail": stderr[-4_000:],
        }
    finally:
        _ACTIVE_WORKER_PROCESS = None
    assert process is not None
    markers = [line[len(RESULT_MARKER) :] for line in stdout.splitlines() if line.startswith(RESULT_MARKER)]
    if process.returncode != 0 or len(markers) != 1:
        return {
            "contract_version": CONTRACT_VERSION,
            **_identity(payload),
            "status": "worker_protocol_error",
            "passed": False,
            "reason": f"returncode={process.returncode};markers={len(markers)}",
            "duration_seconds": time.monotonic() - started,
            "worker_stdout_tail": stdout[-2_000:],
            "worker_stderr_tail": stderr[-4_000:],
        }
    result = json.loads(markers[0])
    if stderr.strip():
        result["worker_stderr_tail"] = stderr[-4_000:]
    return result


def _rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist()


def _tasks(parents_path: Path, children_path: Path, manifest_path: Path) -> list[dict[str, Any]]:
    parents = _rows(parents_path)
    children = _rows(children_path)
    manifests = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not (len(parents) == len(children) == len(manifests)):
        raise ValueError(f"aligned artifact count mismatch:{len(parents)}:{len(children)}:{len(manifests)}")
    tasks: list[dict[str, Any]] = []
    for index, (parent, child, manifest) in enumerate(zip(parents, children, manifests, strict=True)):
        parent_uuid = _nested(parent, "extra_info.uuid")
        child_uuid = _nested(child, "extra_info.uuid")
        parent_code = _nested(parent, "reward_model.ground_truth")
        child_code = _nested(child, "reward_model.ground_truth")
        entry_point = _nested(child, "extra_info.entry_point", "Model")
        if manifest.get("candidate_row_index") != index:
            raise ValueError(f"manifest row index mismatch:{index}")
        if manifest.get("parent_uuid") != parent_uuid or manifest.get("child_uuid") != child_uuid:
            raise ValueError(f"manifest UUID mismatch:{index}")
        if not all(isinstance(value, str) and value for value in (parent_code, child_code, entry_point)):
            raise ValueError(f"invalid code fields:{index}")
        if _sha256_bytes(parent_code.encode()) != manifest.get("parent_reference_sha256"):
            raise ValueError(f"parent reference hash mismatch:{index}")
        if _sha256_bytes(child_code.encode()) != manifest.get("child_reference_sha256"):
            raise ValueError(f"child reference hash mismatch:{index}")
        tasks.append(
            {
                "candidate_row_index": index,
                "parent_uuid": parent_uuid,
                "child_uuid": child_uuid,
                "parent_code": parent_code,
                "child_code": child_code,
                "entry_point": entry_point,
                "assigned_family": manifest["assigned_target"],
                "transformed_factory_count": manifest["transformed_factory_count"],
                "source_kind": _nested(manifest, "source_binding.source_kind"),
            }
        )
    return tasks


def _load_allowlist(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    values = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not values:
        raise ValueError("child allowlist is empty")
    return values


def _append(handle: Any, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parents", type=Path)
    parser.add_argument("children", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--child-uuid-file", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trials", type=int, default=MIN_LIVENESS_TRIALS)
    parser.add_argument("--seed", type=int, default=REQUIRED_LIVENESS_SEED)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--launcher-sha256", required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv == ["--_worker"]:
        payload = json.loads(sys.stdin.read())
        print(RESULT_MARKER + json.dumps(_worker(payload), ensure_ascii=False))
        return
    args = _parser().parse_args(raw_argv)
    _install_driver_signal_handlers()
    if args.trials < MIN_LIVENESS_TRIALS:
        raise ValueError(f"trials must be at least {MIN_LIVENESS_TRIALS}")
    if args.seed != REQUIRED_LIVENESS_SEED:
        raise ValueError(f"seed must be exactly {REQUIRED_LIVENESS_SEED}")
    if args.timeout_seconds <= 0:
        raise ValueError("timeout must be positive")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must satisfy 0 <= index < shard-count")
    if len(args.launcher_sha256) != 64 or any(char not in "0123456789abcdef" for char in args.launcher_sha256):
        raise ValueError("launcher-sha256 must be 64 lowercase hexadecimal characters")
    tasks = _tasks(args.parents, args.children, args.manifest)
    allowlist = _load_allowlist(args.child_uuid_file)
    if allowlist is not None:
        known = {str(task["child_uuid"]) for task in tasks}
        missing = allowlist - known
        if missing:
            raise ValueError(f"allowlisted child UUIDs not found:{sorted(missing)[:10]}")
        tasks = [task for task in tasks if task["child_uuid"] in allowlist]
    if not tasks:
        raise ValueError("selected candidate count must be positive")
    source_kinds = {task.get("source_kind") for task in tasks}
    if len(tasks) > MAX_CANONICAL_CANARY_CANDIDATES and source_kinds != {"shape_coverage_resample"}:
        raise ValueError(
            f"more than {MAX_CANONICAL_CANARY_CANDIDATES} liveness candidates require one bound "
            f"shape-resample source, found {len(tasks)}:{sorted(str(value) for value in source_kinds)}"
        )
    if args.limit is not None and source_kinds == {"shape_coverage_resample"}:
        raise ValueError("--limit is forbidden for a bound shape-resample source")
    tasks = [task for index, task in enumerate(tasks) if index % args.shard_count == args.shard_index]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        tasks = tasks[: args.limit]
    evidence = {
        "contract_version": CONTRACT_VERSION,
        "binding_version": RUN_BINDING_VERSION,
        "validator_source_sha256": _sha256_file(Path(__file__).resolve()),
        "launcher_source_sha256": args.launcher_sha256,
        "parents_sha256": _sha256_file(args.parents),
        "children_sha256": _sha256_file(args.children),
        "manifest_sha256": _sha256_file(args.manifest),
        "allowlist_sha256": _sha256_file(args.child_uuid_file) if args.child_uuid_file else None,
        "validation_config": {
            "device": args.device,
            "trials": args.trials,
            "seed": args.seed,
            "timeout_seconds": args.timeout_seconds,
            "max_device_memory_gib": MAX_DEVICE_MEMORY_GIB,
        },
    }
    binding_sha256 = _canonical_sha256(evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: collections.Counter[str] = collections.Counter()
    executed = 0
    resumed = 0
    with args.output.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process owns output:{args.output}") from exc
        handle.seek(0)
        prior: dict[str, dict[str, Any]] = {}
        task_by_uuid = {str(task["child_uuid"]): task for task in tasks}
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            uuid = record.get("child_uuid")
            if uuid not in task_by_uuid or uuid in prior:
                raise ValueError(f"invalid resume child at {args.output}:{line_number}:{uuid}")
            if record.get("validation_binding_sha256") != binding_sha256:
                raise ValueError(f"resume binding mismatch at {args.output}:{line_number}")
            if type(record.get("passed")) is not bool:
                raise ValueError(f"resume passed field invalid at {args.output}:{line_number}")
            prior[str(uuid)] = record
        handle.seek(0, os.SEEK_END)
        for task in tasks:
            child_uuid = str(task["child_uuid"])
            if child_uuid in prior:
                resumed += 1
                counts[str(prior[child_uuid]["status"])] += 1
                continue
            result = _run_subprocess(
                {**task, "device": args.device, "trials": args.trials, "seed": args.seed},
                args.timeout_seconds,
            )
            result.update(evidence)
            result["validation_binding_sha256"] = binding_sha256
            _append(handle, result)
            counts[str(result["status"])] += 1
            executed += 1
            print(
                json.dumps(
                    {"child_uuid": child_uuid, "status": result.get("status"), "reason": result.get("reason")},
                    ensure_ascii=False,
                ),
                flush=True,
            )
    passed = int(counts.get("passed", 0))
    print(
        json.dumps(
            {
                "contract_version": CONTRACT_VERSION,
                "selected": len(tasks),
                "executed": executed,
                "resumed": resumed,
                "passed": passed,
                "failed": len(tasks) - passed,
                "counts": dict(sorted(counts.items())),
                "output_path": str(args.output.resolve()),
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
                "launcher_source_sha256": args.launcher_sha256,
                "validator_source_sha256": evidence["validator_source_sha256"],
                "validation_binding_sha256": binding_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
