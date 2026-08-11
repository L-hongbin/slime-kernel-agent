#!/usr/bin/env python3
"""Validate dtype realization, coherent model state, and no-FP32 fallback.

Each row is isolated in a subprocess under the 64-GiB allocator guard.  Three
deterministic trials prove direct-input realization, construction RNG equality,
the selected model-state coherence contract, cast-equivalent outputs, and a
child dispatcher trace with no floating output outside the assigned dtype.
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
    _outputs_allclose,
    _restore_rng_states,
    _seed_torch,
    _snapshot_output,
    _snapshot_rng_states,
    _to_device,
)
from tools.data.synthesize.serial_source_contract import resolve_lane_serial_source
from tools.data.synthesize.validate_train_mode_contract import _cuda_memory_guard, _CudaMemoryGuardFailure

PARAMETER_FREE = "parameter_free"
MODULE_STATE = "module_state"
COHERENCE_CLASSES = frozenset({PARAMETER_FREE, MODULE_STATE})
CONTRACT_VERSIONS = {
    PARAMETER_FREE: "dtype_parameter_free_runtime_liveness_v2",
    MODULE_STATE: "dtype_module_state_runtime_liveness_v1",
}
RUN_BINDING_VERSIONS = {
    PARAMETER_FREE: "dtype_parameter_free_runtime_binding_v2",
    MODULE_STATE: "dtype_module_state_runtime_binding_v1",
}
RESULT_MARKER = "__DTYPE_LIVENESS_RESULT__="
MAX_DEVICE_MEMORY_GIB = 64.0
MAX_AUTHORIZED_CANDIDATES = 5_000
MIN_LIVENESS_TRIALS = 3
REQUIRED_LIVENESS_SEED = 17
LOW_PRECISION_TOLERANCES = {"float16": 1e-2, "bfloat16": 2e-2}
TARGET_DTYPES = frozenset({"float16", "bfloat16"})
MAX_RECORDED_DISPATCH_TRANSITIONS = 256
_DRIVER_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
_ACTIVE_WORKER_PROCESS: subprocess.Popen[str] | None = None


def contract_version(coherence_class: str) -> str:
    try:
        return CONTRACT_VERSIONS[coherence_class]
    except KeyError as exc:
        raise ValueError(f"unsupported coherence class:{coherence_class}") from exc


def binding_version(coherence_class: str) -> str:
    try:
        return RUN_BINDING_VERSIONS[coherence_class]
    except KeyError as exc:
        raise ValueError(f"unsupported coherence class:{coherence_class}") from exc


class UnsupportedCase(RuntimeError):
    """The experiment cannot establish the dtype contract for one row."""


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
        process = _ACTIVE_WORKER_PROCESS
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        raise KeyboardInterrupt(f"dtype-liveness driver received signal {signum}")

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


def _target_dtype(target: str, torch: Any) -> Any:
    if target not in TARGET_DTYPES:
        raise UnsupportedCase(f"unsupported_target_dtype:{target}")
    return getattr(torch, target)


def _compare_input_tree(
    parent: Any,
    child: Any,
    *,
    path: str,
    target_dtype: Any,
    torch: Any,
) -> tuple[int, list[dict[str, Any]]]:
    if isinstance(parent, torch.Tensor) or isinstance(child, torch.Tensor):
        if not isinstance(parent, torch.Tensor) or not isinstance(child, torch.Tensor):
            raise UnsupportedCase(f"tensor_structure_mismatch:{path}")
        parent_metadata = (
            tuple(parent.shape),
            parent.device,
            parent.layout,
            tuple(parent.stride()),
            parent.storage_offset(),
            parent.requires_grad,
            parent.is_pinned(),
        )
        child_metadata = (
            tuple(child.shape),
            child.device,
            child.layout,
            tuple(child.stride()),
            child.storage_offset(),
            child.requires_grad,
            child.is_pinned(),
        )
        if parent_metadata != child_metadata:
            raise UnsupportedCase(f"tensor_metadata_changed:{path}:{parent_metadata!r}:{child_metadata!r}")
        if parent.dtype.is_complex or child.dtype.is_complex:
            raise UnsupportedCase(f"complex_input_not_supported:{path}")
        if parent.dtype.is_floating_point:
            if parent.dtype != torch.float32:
                raise UnsupportedCase(f"parent_float_input_not_fp32:{path}:{parent.dtype}")
            if child.dtype != target_dtype:
                raise UnsupportedCase(f"child_float_input_not_target:{path}:{child.dtype}:{target_dtype}")
            if not bool(torch.isfinite(parent).all().item()) or not bool(torch.isfinite(child).all().item()):
                raise UnsupportedCase(f"non_finite_input:{path}")
            cast_parent = parent.to(dtype=target_dtype)
            maximum_factory_cast_difference = (
                float((cast_parent.to(torch.float32) - child.to(torch.float32)).abs().max().item())
                if child.numel()
                else 0.0
            )
            return 1, [
                {
                    "path": path,
                    "shape": list(child.shape),
                    "parent_dtype": str(parent.dtype),
                    "child_dtype": str(child.dtype),
                    "stride": list(child.stride()),
                    "storage_offset": child.storage_offset(),
                    "is_pinned": child.is_pinned(),
                    "numel": child.numel(),
                    "factory_values_equal_to_parent_cast": bool(torch.equal(cast_parent, child)),
                    "maximum_factory_cast_difference": maximum_factory_cast_difference,
                    "minimum": float(child.min().item()) if child.numel() else None,
                    "maximum": float(child.max().item()) if child.numel() else None,
                }
            ]
        if child.dtype != parent.dtype or not torch.equal(parent, child):
            raise UnsupportedCase(f"non_float_tensor_changed:{path}:{parent.dtype}:{child.dtype}")
        return 0, []
    if isinstance(parent, Mapping) or isinstance(child, Mapping):
        if not isinstance(parent, Mapping) or not isinstance(child, Mapping) or list(parent) != list(child):
            raise UnsupportedCase(f"mapping_structure_mismatch:{path}")
        changed = 0
        records: list[dict[str, Any]] = []
        for key in parent:
            count, nested = _compare_input_tree(
                parent[key], child[key], path=f"{path}.{key}", target_dtype=target_dtype, torch=torch
            )
            changed += count
            records.extend(nested)
        return changed, records
    if isinstance(parent, (list, tuple)) or isinstance(child, (list, tuple)):
        if type(parent) is not type(child) or len(parent) != len(child):
            raise UnsupportedCase(f"sequence_structure_mismatch:{path}")
        changed = 0
        records = []
        for index, (left, right) in enumerate(zip(parent, child, strict=True)):
            count, nested = _compare_input_tree(
                left, right, path=f"{path}[{index}]", target_dtype=target_dtype, torch=torch
            )
            changed += count
            records.extend(nested)
        return changed, records
    if parent != child:
        raise UnsupportedCase(f"non_tensor_input_changed:{path}:{parent!r}:{child!r}")
    return 0, []


def _cast_float_tree(value: Any, *, target_dtype: Any, torch: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.dtype.is_complex:
            raise UnsupportedCase("complex_cast_equivalence_not_supported")
        return value.to(dtype=target_dtype) if value.dtype.is_floating_point else value
    if isinstance(value, Mapping):
        return type(value)(
            (key, _cast_float_tree(item, target_dtype=target_dtype, torch=torch)) for key, item in value.items()
        )
    if isinstance(value, list):
        return [_cast_float_tree(item, target_dtype=target_dtype, torch=torch) for item in value]
    if isinstance(value, tuple):
        return tuple(_cast_float_tree(item, target_dtype=target_dtype, torch=torch) for item in value)
    return value


def _flatten_tensors(value: Any, torch: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, Mapping):
        return [tensor for item in value.values() for tensor in _flatten_tensors(item, torch)]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in _flatten_tensors(item, torch)]
    return []


def _assert_parameter_free(model: Any, torch: Any) -> dict[str, int]:
    parameters = list(model.named_parameters(recurse=True))
    buffers = list(model.named_buffers(recurse=True))
    state = model.state_dict()
    modules = list(model.named_modules())
    if parameters:
        raise UnsupportedCase(f"runtime_parameter_count_not_zero:{len(parameters)}")
    if buffers:
        raise UnsupportedCase(f"runtime_buffer_count_not_zero:{len(buffers)}")
    if state:
        raise UnsupportedCase(f"runtime_state_dict_not_empty:{len(state)}")
    if len(modules) != 1:
        raise UnsupportedCase(f"runtime_submodule_count_not_zero:{len(modules) - 1}")
    _assert_no_unregistered_tensors(model, torch)
    return {"parameter_count": 0, "buffer_count": 0, "state_dict_entries": 0, "submodule_count": 0}


def _assert_no_unregistered_tensors(model: Any, torch: Any) -> None:
    seen_modules: set[int] = set()
    seen_containers: set[int] = set()

    def inspect(value: Any, path: str) -> None:
        if isinstance(value, torch.Tensor):
            raise UnsupportedCase(f"runtime_unregistered_tensor_state:{path}")
        if isinstance(value, torch.nn.Module):
            raise UnsupportedCase(f"runtime_unregistered_module_state:{path}")
        if isinstance(value, (str, bytes, int, float, bool, type(None))):
            return
        if isinstance(value, Mapping):
            if id(value) in seen_containers:
                return
            seen_containers.add(id(value))
            for key, item in value.items():
                inspect(item, f"{path}.{key}")
        elif isinstance(value, (list, tuple, set, frozenset)):
            if id(value) in seen_containers:
                return
            seen_containers.add(id(value))
            for index, item in enumerate(value):
                inspect(item, f"{path}[{index}]")

    def visit(module: Any, path: str) -> None:
        if id(module) in seen_modules:
            return
        seen_modules.add(id(module))
        ignored = {"_parameters", "_buffers", "_modules"}
        for name, value in vars(module).items():
            if name not in ignored:
                inspect(value, f"{path}.{name}")
        for name, child in module._modules.items():
            if child is not None:
                visit(child, f"{path}.{name}")

    visit(model, "model")


def _logical_registered_state(model: Any, torch: Any) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}

    def visit(module: Any, path: str, ancestry: frozenset[int]) -> None:
        if id(module) in ancestry:
            raise UnsupportedCase(f"cyclic_module_graph:{path or '<root>'}")
        next_ancestry = ancestry | {id(module)}
        for kind, values in (("parameter", module._parameters), ("buffer", module._buffers)):
            for name, tensor in values.items():
                logical_name = f"{path}.{name}" if path else name
                key = f"{kind}:{logical_name}"
                if key in entries:
                    raise UnsupportedCase(f"duplicate_logical_state_name:{key}")
                entries[key] = {
                    "kind": kind,
                    "name": logical_name,
                    "tensor": tensor,
                    "persistent": kind == "parameter" or name not in module._non_persistent_buffers_set,
                }
        for name, child in module._modules.items():
            if child is not None:
                visit(child, f"{path}.{name}" if path else name, next_ancestry)

    visit(model, "", frozenset())
    if len(entries) > 4096:
        raise UnsupportedCase(f"registered_state_entry_limit_exceeded:{len(entries)}")
    return entries


def _validate_state_tensor(tensor: Any, path: str, torch: Any) -> None:
    uninitialized_types = tuple(
        value
        for value in (
            getattr(torch.nn.parameter, "UninitializedParameter", None),
            getattr(torch.nn.parameter, "UninitializedBuffer", None),
        )
        if isinstance(value, type)
    )
    if uninitialized_types and isinstance(tensor, uninitialized_types):
        raise UnsupportedCase(f"lazy_or_uninitialized_state:{path}")
    if tensor.layout != torch.strided or tensor.is_sparse or tensor.is_quantized:
        raise UnsupportedCase(f"unsupported_state_layout:{path}:{tensor.layout}")
    if tensor.dtype.is_complex:
        raise UnsupportedCase(f"complex_state_not_supported:{path}:{tensor.dtype}")


def _alias_groups(entries: Mapping[str, Mapping[str, Any]]) -> tuple[list[list[str]], list[list[str]]]:
    object_groups: dict[int, list[str]] = collections.defaultdict(list)
    storage_groups: dict[int, list[str]] = collections.defaultdict(list)
    for key, entry in entries.items():
        tensor = entry["tensor"]
        if tensor is None:
            continue
        object_groups[id(tensor)].append(key)
        storage_groups[int(tensor.untyped_storage()._cdata)].append(key)
    objects = sorted(sorted(group) for group in object_groups.values() if len(group) > 1)
    storages = sorted(sorted(group) for group in storage_groups.values() if len(group) > 1)
    return objects, storages


def _unique_state_bytes(entries: Mapping[str, Mapping[str, Any]]) -> int:
    storages: dict[int, int] = {}
    for entry in entries.values():
        tensor = entry["tensor"]
        if tensor is not None:
            storage = tensor.untyped_storage()
            storages[int(storage._cdata)] = int(storage.nbytes())
    return sum(storages.values())


def _assert_module_state_pair(parent: Any, child: Any, target_dtype: Any, torch: Any) -> dict[str, Any]:
    _assert_no_unregistered_tensors(parent, torch)
    _assert_no_unregistered_tensors(child, torch)
    parent_entries = _logical_registered_state(parent, torch)
    child_entries = _logical_registered_state(child, torch)
    if set(parent_entries) != set(child_entries):
        raise UnsupportedCase("registered_state_names_changed")
    floating_entries = 0
    nonfloating_entries = 0
    parameter_entries = 0
    buffer_entries = 0
    schema: list[dict[str, Any]] = []
    for key in sorted(parent_entries):
        parent_entry = parent_entries[key]
        child_entry = child_entries[key]
        if parent_entry["kind"] != child_entry["kind"] or parent_entry["persistent"] != child_entry["persistent"]:
            raise UnsupportedCase(f"registered_state_kind_or_persistence_changed:{key}")
        parameter_entries += parent_entry["kind"] == "parameter"
        buffer_entries += parent_entry["kind"] == "buffer"
        left = parent_entry["tensor"]
        right = child_entry["tensor"]
        if (left is None) != (right is None):
            raise UnsupportedCase(f"registered_none_state_changed:{key}")
        if left is None:
            schema.append({"key": key, "persistent": parent_entry["persistent"], "none": True})
            continue
        _validate_state_tensor(left, key, torch)
        _validate_state_tensor(right, key, torch)
        left_metadata = (
            tuple(left.shape),
            left.device,
            left.layout,
            tuple(left.stride()),
            left.storage_offset(),
            left.requires_grad,
        )
        right_metadata = (
            tuple(right.shape),
            right.device,
            right.layout,
            tuple(right.stride()),
            right.storage_offset(),
            right.requires_grad,
        )
        if left_metadata != right_metadata:
            raise UnsupportedCase(f"registered_state_metadata_changed:{key}")
        if left.dtype.is_floating_point:
            floating_entries += 1
            if left.dtype != torch.float32 or right.dtype != target_dtype:
                raise UnsupportedCase(f"registered_float_state_dtype_mismatch:{key}:{left.dtype}:{right.dtype}")
            if not torch.equal(left.to(dtype=target_dtype), right):
                raise UnsupportedCase(f"registered_float_state_not_exact_parent_cast:{key}")
            dtype_class = "floating"
        else:
            nonfloating_entries += 1
            if left.dtype != right.dtype or not torch.equal(left, right):
                raise UnsupportedCase(f"registered_nonfloat_state_changed:{key}")
            dtype_class = str(left.dtype)
        schema.append(
            {
                "key": key,
                "persistent": parent_entry["persistent"],
                "none": False,
                "shape": list(left.shape),
                "stride": list(left.stride()),
                "storage_offset": left.storage_offset(),
                "requires_grad": left.requires_grad,
                "dtype_class": dtype_class,
            }
        )
    if floating_entries <= 0:
        raise UnsupportedCase("registered_floating_state_count_not_positive")
    parent_objects, parent_storages = _alias_groups(parent_entries)
    child_objects, child_storages = _alias_groups(child_entries)
    if parent_objects != child_objects or parent_storages != child_storages:
        raise UnsupportedCase("registered_state_alias_graph_changed")
    parent_bytes = _unique_state_bytes(parent_entries)
    child_bytes = _unique_state_bytes(child_entries)
    if parent_bytes + child_bytes > int(MAX_DEVICE_MEMORY_GIB * 1024**3):
        raise UnsupportedCase(f"registered_state_memory_budget_exceeded:{parent_bytes}:{child_bytes}")
    return {
        "parameter_entries": parameter_entries,
        "buffer_entries": buffer_entries,
        "floating_entries": floating_entries,
        "nonfloating_entries": nonfloating_entries,
        "state_bytes_parent": parent_bytes,
        "state_bytes_child": child_bytes,
        "object_alias_groups": len(parent_objects),
        "storage_alias_groups": len(parent_storages),
        "state_schema_sha256": _canonical_sha256(schema),
        "values_equal_to_parent_cast": True,
        "nonfloat_values_equal": True,
        "unregistered_tensor_count": 0,
    }


def _assert_model_state_dtype(model: Any, target_dtype: Any, torch: Any) -> None:
    _assert_no_unregistered_tensors(model, torch)
    for key, entry in _logical_registered_state(model, torch).items():
        tensor = entry["tensor"]
        if tensor is None:
            continue
        _validate_state_tensor(tensor, key, torch)
        if tensor.dtype.is_floating_point and tensor.dtype != target_dtype:
            raise UnsupportedCase(f"post_forward_registered_state_dtype_mismatch:{key}:{tensor.dtype}")


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


def _run_traced_arm(
    model: Any,
    inputs: Sequence[Any],
    rng_state: Mapping[str, Any],
    *,
    target_dtype: Any,
    coherence_class: str,
    torch: Any,
) -> tuple[Any, dict[str, Any]]:
    from torch.utils._python_dispatch import TorchDispatchMode

    class DtypeTraceMode(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.transitions: collections.Counter[tuple[str, tuple[str, ...], tuple[str, ...]]] = collections.Counter()
            self.fallbacks: collections.Counter[tuple[str, str]] = collections.Counter()
            self.target_consuming_calls = 0
            self.total_calls = 0

        def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
            del types
            kwargs = kwargs or {}
            input_tensors = _flatten_tensors((args, kwargs), torch)
            output = func(*args, **kwargs)
            output_tensors = _flatten_tensors(output, torch)
            input_dtypes = tuple(sorted({str(tensor.dtype) for tensor in input_tensors}))
            output_dtypes = tuple(sorted({str(tensor.dtype) for tensor in output_tensors}))
            name = str(func)
            self.transitions[(name, input_dtypes, output_dtypes)] += 1
            self.total_calls += 1
            if any(tensor.dtype == target_dtype for tensor in input_tensors):
                self.target_consuming_calls += 1
            for tensor in output_tensors:
                if tensor.dtype.is_complex:
                    self.fallbacks[(name, str(tensor.dtype))] += 1
                elif tensor.dtype.is_floating_point and tensor.dtype != target_dtype:
                    self.fallbacks[(name, str(tensor.dtype))] += 1
            return output

    trial_model = copy.deepcopy(model)
    mode = DtypeTraceMode()
    try:
        _restore_rng_states(rng_state)
        with torch.no_grad(), mode:
            raw_output = _invoke_model(trial_model, inputs)
        if coherence_class == MODULE_STATE:
            _assert_model_state_dtype(trial_model, target_dtype, torch)
        else:
            _assert_parameter_free(trial_model, torch)
        output = _snapshot_output(raw_output)
        if not _all_finite(output):
            raise UnsupportedCase("non_finite_traced_output")
    finally:
        del trial_model
    if mode.target_consuming_calls <= 0:
        raise UnsupportedCase("dispatch_trace_did_not_consume_target_dtype")
    if mode.fallbacks:
        first = next(iter(sorted(mode.fallbacks.items())))
        raise UnsupportedCase(f"dispatch_fp32_or_complex_fallback:{first[0][0]}:{first[0][1]}:{first[1]}")
    ordered = sorted(mode.transitions.items(), key=lambda item: (-item[1], item[0]))
    return output, {
        "total_dispatch_calls": mode.total_calls,
        "target_consuming_calls": mode.target_consuming_calls,
        "fp32_or_complex_fallback_calls": 0,
        "transition_count": len(ordered),
        "transitions_truncated": len(ordered) > MAX_RECORDED_DISPATCH_TRANSITIONS,
        "transitions": [
            {"operator": key[0], "input_dtypes": list(key[1]), "output_dtypes": list(key[2]), "calls": count}
            for key, count in ordered[:MAX_RECORDED_DISPATCH_TRANSITIONS]
        ],
    }


def _compare_outputs(
    parent: Any,
    child: Any,
    *,
    path: str,
    target_dtype: Any,
    tolerance: float,
    torch: Any,
) -> list[dict[str, Any]]:
    if isinstance(parent, torch.Tensor) or isinstance(child, torch.Tensor):
        if not isinstance(parent, torch.Tensor) or not isinstance(child, torch.Tensor):
            raise UnsupportedCase(f"output_tensor_structure_mismatch:{path}")
        if tuple(parent.shape) != tuple(child.shape):
            raise UnsupportedCase(f"output_shape_mismatch:{path}:{tuple(parent.shape)}:{tuple(child.shape)}")
        if parent.dtype.is_complex or child.dtype.is_complex:
            raise UnsupportedCase(f"complex_output_not_supported:{path}")
        if parent.dtype.is_floating_point:
            if parent.dtype != torch.float32:
                raise UnsupportedCase(f"parent_float_output_not_fp32:{path}:{parent.dtype}")
            if child.dtype != target_dtype:
                raise UnsupportedCase(f"child_float_output_not_target:{path}:{child.dtype}:{target_dtype}")
            parent_float = parent.to(torch.float32)
            child_float = child.to(torch.float32)
            difference = (parent_float - child_float).abs()
            if not bool(
                torch.allclose(
                    parent_float,
                    child_float,
                    rtol=tolerance,
                    atol=tolerance,
                    equal_nan=False,
                )
            ):
                raise UnsupportedCase(
                    f"cast_equivalent_output_mismatch:{path}:max_abs={float(difference.max().item())}"
                )
            return [
                {
                    "path": path,
                    "shape": list(child.shape),
                    "parent_dtype": str(parent.dtype),
                    "child_dtype": str(child.dtype),
                    "elements": child.numel(),
                    "maximum_absolute_difference": float(difference.max().item()) if child.numel() else 0.0,
                    "mean_absolute_difference": float(difference.mean().item()) if child.numel() else 0.0,
                    "rtol": tolerance,
                    "atol": tolerance,
                }
            ]
        if child.dtype != parent.dtype or not torch.equal(parent, child):
            raise UnsupportedCase(f"non_float_output_mismatch:{path}:{parent.dtype}:{child.dtype}")
        return [
            {
                "path": path,
                "shape": list(child.shape),
                "parent_dtype": str(parent.dtype),
                "child_dtype": str(child.dtype),
                "elements": child.numel(),
                "maximum_absolute_difference": 0.0,
                "mean_absolute_difference": 0.0,
                "rtol": 0.0,
                "atol": 0.0,
            }
        ]
    if isinstance(parent, Mapping) or isinstance(child, Mapping):
        if not isinstance(parent, Mapping) or not isinstance(child, Mapping) or list(parent) != list(child):
            raise UnsupportedCase(f"output_mapping_structure_mismatch:{path}")
        return [
            record
            for key in parent
            for record in _compare_outputs(
                parent[key],
                child[key],
                path=f"{path}.{key}",
                target_dtype=target_dtype,
                tolerance=tolerance,
                torch=torch,
            )
        ]
    if isinstance(parent, (list, tuple)) or isinstance(child, (list, tuple)):
        if type(parent) is not type(child) or len(parent) != len(child):
            raise UnsupportedCase(f"output_sequence_structure_mismatch:{path}")
        return [
            record
            for index, (left, right) in enumerate(zip(parent, child, strict=True))
            for record in _compare_outputs(
                left,
                right,
                path=f"{path}[{index}]",
                target_dtype=target_dtype,
                tolerance=tolerance,
                torch=torch,
            )
        ]
    if parent != child:
        raise UnsupportedCase(f"non_tensor_output_mismatch:{path}:{parent!r}:{child!r}")
    return []


def _build_model_instance(
    namespace: Mapping[str, Any], device: Any, seed: int, coherence_class: str, torch: Any
) -> tuple[Any, Any, Mapping[str, Any]]:
    _seed_torch(seed, str(device))
    init_inputs = namespace["get_init_inputs"]()
    init_inputs = [] if init_inputs is None else init_inputs
    if not isinstance(init_inputs, (list, tuple)):
        raise UnsupportedCase("get_init_inputs_is_not_list_or_tuple")
    model = _build_model(namespace["__entry_point__"], _to_device(list(init_inputs), str(device)))
    if not isinstance(model, torch.nn.Module):
        raise UnsupportedCase("entry_point_did_not_build_torch_module")
    model = model.to(device)
    model.train(True)
    if coherence_class == PARAMETER_FREE:
        _assert_parameter_free(model, torch)
    elif coherence_class != MODULE_STATE:
        raise UnsupportedCase(f"unsupported_coherence_class:{coherence_class}")
    return model, init_inputs, _snapshot_rng_states(str(device))


def _evaluate_without_guard(payload: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    device = torch.device(str(payload["device"]))
    if not torch.cuda.is_available() or device.type != "cuda":
        raise RuntimeError(f"CUDA device required, got {device}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    target = str(payload["assigned_dtype"])
    coherence_class = str(payload["coherence_class"])
    if coherence_class not in COHERENCE_CLASSES:
        raise UnsupportedCase(f"unsupported_coherence_class:{coherence_class}")
    target_dtype = _target_dtype(target, torch)
    tolerance = LOW_PRECISION_TOLERANCES[target]
    parent_namespace = _exec_ops_code(
        str(payload["parent_code"]), entry_point=str(payload["entry_point"]), device=str(device)
    )
    child_namespace = _exec_ops_code(
        str(payload["child_code"]), entry_point=str(payload["entry_point"]), device=str(device)
    )
    trials = int(payload["trials"])
    seed = int(payload["seed"])
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
            changed_tensors, input_records = _compare_input_tree(
                parent_raw, child_raw, path="inputs", target_dtype=target_dtype, torch=torch
            )
            if changed_tensors != expected_changed:
                raise UnsupportedCase(f"changed_dtype_tensor_count_mismatch:{changed_tensors}:{expected_changed}")
            cast_raw = _cast_float_tree(parent_raw, target_dtype=target_dtype, torch=torch)
            parent_inputs = _normalize_forward_inputs(parent_raw, str(device))
            child_inputs = _normalize_forward_inputs(child_raw, str(device))
            cast_inputs = _normalize_forward_inputs(cast_raw, str(device))
            del parent_raw, child_raw, cast_raw

            parent_model, parent_init_inputs, parent_construct_rng = _build_model_instance(
                parent_namespace, device, trial_seed + 1, coherence_class, torch
            )
            child_model, child_init_inputs, child_construct_rng = _build_model_instance(
                child_namespace, device, trial_seed + 1, coherence_class, torch
            )
            if not _rng_states_equal(parent_init_inputs, child_init_inputs, torch):
                raise UnsupportedCase("get_init_inputs_values_changed")
            if not _rng_states_equal(parent_construct_rng, child_construct_rng, torch):
                raise UnsupportedCase("post_model_construction_rng_state_changed")
            if coherence_class == MODULE_STATE:
                model_state = _assert_module_state_pair(parent_model, child_model, target_dtype, torch)
            else:
                model_state = {
                    "parameter_entries": 0,
                    "buffer_entries": 0,
                    "floating_entries": 0,
                    "nonfloating_entries": 0,
                    "state_bytes_parent": 0,
                    "state_bytes_child": 0,
                    "object_alias_groups": 0,
                    "storage_alias_groups": 0,
                    "state_schema_sha256": _canonical_sha256([]),
                    "values_equal_to_parent_cast": True,
                    "nonfloat_values_equal": True,
                    "unregistered_tensor_count": 0,
                }
            model_state["coherence_class"] = coherence_class
            model_state["construction_rng_equal"] = True
            forward_rng = _snapshot_rng_states(str(device))
            parent_output = _run_arm(parent_model, parent_inputs, forward_rng)
            parent_control = _run_arm(parent_model, parent_inputs, forward_rng)
            if not _outputs_allclose(parent_output, parent_control, rtol=0.0, atol=0.0):
                raise UnsupportedCase("unchanged_parent_control_is_not_exact")
            semantic_output, semantic_trace = _run_traced_arm(
                child_model,
                cast_inputs,
                forward_rng,
                target_dtype=target_dtype,
                coherence_class=coherence_class,
                torch=torch,
            )
            output_records = _compare_outputs(
                parent_output,
                semantic_output,
                path="output",
                target_dtype=target_dtype,
                tolerance=tolerance,
                torch=torch,
            )
            realized_output, realized_trace = _run_traced_arm(
                child_model,
                child_inputs,
                forward_rng,
                target_dtype=target_dtype,
                coherence_class=coherence_class,
                torch=torch,
            )
            realized_control = _run_arm(child_model, child_inputs, forward_rng)
            if not _outputs_allclose(realized_output, realized_control, rtol=0.0, atol=0.0):
                raise UnsupportedCase("unchanged_child_control_is_not_exact")
            trial_records.append(
                {
                    "trial": trial,
                    "seed": trial_seed,
                    "changed_dtype_tensors": changed_tensors,
                    "inputs": input_records,
                    "model_state": model_state,
                    "cast_equivalent_outputs": output_records,
                    "semantic_dispatch": semantic_trace,
                    "realized_dispatch": realized_trace,
                }
            )
            del (
                parent_inputs,
                child_inputs,
                cast_inputs,
                parent_model,
                child_model,
                parent_init_inputs,
                child_init_inputs,
                parent_output,
                parent_control,
                semantic_output,
                realized_output,
                realized_control,
            )
            torch.cuda.empty_cache()
    finally:
        _restore_rng_states(original_rng)
    return {
        "status": "passed",
        "passed": True,
        "reason": None,
        "target_dtype": target,
        "tolerance": {"rtol": tolerance, "atol": tolerance},
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
        for key in (
            "candidate_row_index",
            "child_uuid",
            "parent_uuid",
            "assigned_dtype",
            "transformed_factory_count",
            "coherence_class",
        )
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
    except BaseException as exc:  # noqa: BLE001 - each row failure is evidence
        result = {"status": "failed", "passed": False, "reason": _exception_detail(exc)}
    return {
        "contract_version": contract_version(str(payload["coherence_class"])),
        **_identity(payload),
        **result,
        "duration_seconds": time.monotonic() - started,
    }


def _run_subprocess(payload: Mapping[str, Any], timeout_seconds: float) -> dict[str, Any]:
    global _ACTIVE_WORKER_PROCESS

    started = time.monotonic()
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(  # noqa: S603 - fixed interpreter and script path
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
            "contract_version": contract_version(str(payload["coherence_class"])),
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
            "contract_version": contract_version(str(payload["coherence_class"])),
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
    serial_source = resolve_lane_serial_source(manifests)
    if serial_source is not None:
        _, source_rows, _ = serial_source
        for index, (parent, manifest) in enumerate(zip(parents, manifests, strict=True)):
            source_index = manifest.get("source_row_index")
            if _canonical_sha256(parent) != _canonical_sha256(source_rows[source_index]):
                raise ValueError(f"serial dtype parent differs from its source row:{index}")
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
        if manifest.get("primary_intervention") != "dtype":
            raise ValueError(f"manifest intervention mismatch:{index}")
        realized = manifest.get("realized_intervention")
        coherence_class = manifest.get("coherence_class")
        if coherence_class not in COHERENCE_CLASSES:
            raise ValueError(f"manifest coherence class unsupported:{index}:{coherence_class}")
        if not isinstance(realized, Mapping) or realized.get("coherence_class") != coherence_class:
            raise ValueError(f"manifest coherence class mismatch:{index}")
        if coherence_class == PARAMETER_FREE and (
            manifest.get("model_parameter_count") != 0 or manifest.get("model_buffer_count") != 0
        ):
            raise ValueError(f"manifest parameter-free proof mismatch:{index}")
        if coherence_class == MODULE_STATE and (
            manifest.get("model_parameter_count") is not None or manifest.get("model_buffer_count") is not None
        ):
            raise ValueError(f"manifest module-state runtime-pending proof mismatch:{index}")
        if not all(isinstance(value, str) and value for value in (parent_code, child_code, entry_point)):
            raise ValueError(f"invalid code fields:{index}")
        if _sha256_bytes(parent_code.encode()) != manifest.get("parent_reference_sha256"):
            raise ValueError(f"parent reference hash mismatch:{index}")
        if _sha256_bytes(child_code.encode()) != manifest.get("child_reference_sha256"):
            raise ValueError(f"child reference hash mismatch:{index}")
        target = manifest.get("assigned_target")
        if target not in TARGET_DTYPES:
            raise ValueError(f"manifest dtype target mismatch:{index}:{target}")
        tasks.append(
            {
                "candidate_row_index": index,
                "parent_uuid": parent_uuid,
                "child_uuid": child_uuid,
                "parent_code": parent_code,
                "child_code": child_code,
                "entry_point": entry_point,
                "assigned_dtype": target,
                "transformed_factory_count": manifest["transformed_factory_count"],
                "coherence_class": coherence_class,
            }
        )
    coherence_classes = {task["coherence_class"] for task in tasks}
    if len(coherence_classes) != 1:
        raise ValueError(f"mixed coherence classes in one validation lane:{sorted(coherence_classes)}")
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
    if not 1 <= len(tasks) <= MAX_AUTHORIZED_CANDIDATES:
        raise ValueError(f"selected candidate count must be in [1, {MAX_AUTHORIZED_CANDIDATES}]:{len(tasks)}")
    tasks = [task for index, task in enumerate(tasks) if index % args.shard_count == args.shard_index]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        tasks = tasks[: args.limit]
    if not tasks:
        raise ValueError("shard selection produced no dtype validation tasks")
    coherence_class = str(tasks[0]["coherence_class"])
    runtime_contract = contract_version(coherence_class)
    evidence = {
        "contract_version": runtime_contract,
        "binding_version": binding_version(coherence_class),
        "coherence_class": coherence_class,
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
            "float16_rtol": LOW_PRECISION_TOLERANCES["float16"],
            "float16_atol": LOW_PRECISION_TOLERANCES["float16"],
            "bfloat16_rtol": LOW_PRECISION_TOLERANCES["bfloat16"],
            "bfloat16_atol": LOW_PRECISION_TOLERANCES["bfloat16"],
            "dispatch_fp32_fallback_allowed": False,
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
                {**task, "device": args.device, "trials": args.trials, "seed": args.seed}, args.timeout_seconds
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
                "contract_version": runtime_contract,
                "coherence_class": coherence_class,
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
