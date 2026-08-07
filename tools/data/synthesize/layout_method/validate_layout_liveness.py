#!/usr/bin/env python3
"""Fail-closed raw-input liveness validation for layout canary children.

Unlike the generic runtime helper this validator intentionally never clones a
forward input.  Cloning makes a contiguous, offset-zero tensor and would turn
the two layout interventions this lane is intended to measure into a false
negative.  Every forward arm therefore regenerates its own input tree and
calls ``model(*args, **kwargs)`` directly.
"""

from __future__ import annotations

import argparse
import collections
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

from tools.data.cleaning.runtime_validation import (  # noqa: E402
    _all_finite,
    _build_model,
    _exec_ops_code,
    _restore_rng_states,
    _seed_torch,
    _snapshot_output,
    _snapshot_rng_states,
    _to_device,
)
from tools.data.synthesize.validate_train_mode_contract import (  # noqa: E402
    _CudaMemoryGuardFailure,
    _cuda_memory_guard,
)

CONTRACT_VERSION = "layout_raw_runtime_liveness_v1"
RUN_BINDING_VERSION = "layout_raw_runtime_liveness_binding_v1"
RESULT_MARKER = "__LAYOUT_LIVENESS_RESULT__="
MAX_DEVICE_MEMORY_GIB = 64.0
MAX_AUTHORIZED_CANDIDATES = 5_000
MIN_LIVENESS_TRIALS = 3
REQUIRED_LIVENESS_SEED = 17
LAYOUT_FAMILIES = frozenset({"transpose_noncontiguous", "slice_storage_offset", "expand_zero_stride"})
EXPECTED_CANONICAL_PARENT_SHA256 = "b07205fcadc543964cfc7ee5fd9c1e4d011f0f3481447f656e5297e40b4b99f4"
EXPECTED_CANONICAL_PARENT_ROWS = 64_315
MAX_RECORDED_DISPATCH_CALLS = 128
_DRIVER_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
_ACTIVE_WORKER_PROCESS: subprocess.Popen[str] | None = None

# These classifications are intentionally fail-closed.  View operations may
# propagate a target storage to a later real consumer, while metadata queries
# and allocation-like operations do not establish that input values reached a
# computation.  Materializers are tracked separately so an erase before the
# first semantic consumer rejects the row.
_VIEW_DISPATCH_OPS = frozenset(
    {
        "aten._unsafe_view",
        "aten._nested_view_from_buffer",
        "aten._reshape_alias",
        "aten._conj",
        "aten._neg_view",
        "aten.adjoint",
        "aten.alias",
        "aten.as_strided",
        "aten.broadcast_to",
        "aten.chunk",
        "aten.conj",
        "aten.detach",
        "aten.diagonal",
        "aten.expand",
        "aten.flatten",
        "aten.imag",
        "aten.lift_fresh",
        "aten.movedim",
        "aten.narrow",
        "aten.permute",
        "aten.ravel",
        "aten.real",
        "aten.reshape",
        "aten.select",
        "aten.slice",
        "aten.split",
        "aten.split_with_sizes",
        "aten.squeeze",
        "aten.swapaxes",
        "aten.swapdims",
        "aten.t",
        "aten.tensor_split",
        "aten.transpose",
        "aten.unbind",
        "aten.unfold",
        "aten.unsqueeze",
        "aten.view",
        "aten.view_as_complex",
        "aten.view_as_real",
    }
)
_ERASE_DISPATCH_OPS = frozenset(
    {
        "aten._autocast_to_full_precision",
        "aten._autocast_to_reduced_precision",
        "aten._copy_from",
        "aten._copy_from_and_resize",
        "aten._nested_view_from_buffer_copy",
        "aten._reshape_copy",
        "aten._to_copy",
        "aten.alias_copy",
        "aten.as_strided_copy",
        "aten.clone",
        "aten.contiguous",
        "aten.copy",
        "aten.copy_",
        "aten.diagonal_copy",
        "aten.expand_copy",
        "aten.lift_fresh_copy",
        "aten.narrow_copy",
        "aten.permute_copy",
        "aten.pin_memory",
        "aten.split_with_sizes_copy",
        "aten.squeeze_copy",
        "aten.t_copy",
        "aten.to",
        "aten.transpose_copy",
        "aten.type_as",
        "aten.unbind_copy",
        "aten.unfold_copy",
        "aten.unsqueeze_copy",
        "aten.view_copy",
    }
)
_NONSEMANTIC_DISPATCH_OPS = frozenset(
    {
        # Tensor metadata only.
        "aten._is_zerotensor",
        "aten._shape_as_tensor",
        "aten._version",
        "aten.device",
        "aten.dim",
        "aten.dtype",
        "aten.is_contiguous",
        "aten.is_conj",
        "aten.is_complex",
        "aten.is_floating_point",
        "aten.is_inference",
        "aten.is_neg",
        "aten.is_pinned",
        "aten.is_same_size",
        "aten.is_set_to",
        "aten.is_signed",
        "aten.layout",
        "aten.ndimension",
        "aten.numel",
        "aten.size",
        "aten.storage_offset",
        "aten.stride",
        "aten.sym_numel",
        "aten.sym_size",
        "aten.sym_storage_offset",
        "aten.sym_stride",
        # Allocation/template operations inspect metadata but not input values.
        "aten.empty_like",
        "aten.full_like",
        "aten.new_empty",
        "aten.new_empty_strided",
        "aten.new_full",
        "aten.new_ones",
        "aten.new_zeros",
        "aten.ones_like",
        "aten.rand_like",
        "aten.randn_like",
        "aten.zeros_like",
    }
)
_SEMANTIC_DISPATCH_OPS = frozenset(
    {
        "aten.block_diag",
        "aten.cartesian_prod",
        "aten.dropout",
        "aten.feature_alpha_dropout",
        "aten.feature_dropout",
        "aten.meshgrid",
        "aten.scaled_dot_product_attention",
        "aten.stack",
    }
)
_SEMANTIC_DISPATCH_PREFIXES = (
    "aten._fft_",
    "aten._linalg_",
    "aten._scaled_dot_product_",
    "aten.adaptive_",
    "aten.fft_",
    "aten.grid_sampler_",
    "aten.linalg_",
    "aten.upsample_",
)
_SEMANTIC_DISPATCH_TAGS = frozenset({"Tag.core", "Tag.pointwise", "Tag.reduction"})
_DATA_DEPENDENT_DISPATCH_TAG = "Tag.data_dependent_output"
_VIEW_COPY_DISPATCH_TAG = "Tag.view_copy"


class UnsupportedCase(RuntimeError):
    """The row cannot establish the strict layout contract."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(payload.encode("utf-8"))


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
        raise KeyboardInterrupt(f"layout-liveness driver received signal {signum}")

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
        return list(left) == list(right) and all(_rng_states_equal(left[key], right[key], torch) for key in left)
    if isinstance(left, (list, tuple)) and type(left) is type(right):
        return len(left) == len(right) and all(
            _rng_states_equal(a, b, torch) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _raw_forward_arguments(value: Any) -> tuple[list[Any], dict[str, Any]]:
    """Normalize only the container; never clone or ``.to`` a tensor."""

    if isinstance(value, Mapping):
        return [], dict(value)
    if not isinstance(value, (list, tuple)):
        raise UnsupportedCase("get_inputs_is_not_list_tuple_or_mapping")
    if len(value) == 2 and isinstance(value[0], (list, tuple)) and isinstance(value[1], Mapping):
        return list(value[0]), dict(value[1])
    return list(value), {}


def _tensor_metadata(tensor: Any) -> dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "layout": str(tensor.layout),
        "stride": list(tensor.stride()),
        "storage_offset": int(tensor.storage_offset()),
        "requires_grad": bool(tensor.requires_grad),
        "is_contiguous": bool(tensor.is_contiguous()),
        "is_pinned": bool(tensor.is_pinned()),
        "numel": int(tensor.numel()),
    }


def _target_predicate(parent: Any, child: Any, family: str, path: str) -> None:
    if parent.layout != child.layout or str(parent.layout) != "torch.strided":
        raise UnsupportedCase(f"non_strided_or_layout_changed:{path}")
    if not parent.is_contiguous() or parent.storage_offset() != 0:
        raise UnsupportedCase(f"parent_not_contiguous_offset_zero:{path}")
    if family == "transpose_noncontiguous":
        if child.is_contiguous() or child.storage_offset() != 0 or tuple(child.stride()) == tuple(parent.stride()):
            raise UnsupportedCase(f"transpose_layout_not_realized:{path}")
    elif family == "slice_storage_offset":
        if child.storage_offset() <= 0:
            raise UnsupportedCase(f"slice_storage_offset_not_realized:{path}")
    elif family == "expand_zero_stride":
        if not any(size > 1 and stride == 0 for size, stride in zip(child.shape, child.stride(), strict=True)):
            raise UnsupportedCase(f"expand_zero_stride_not_realized:{path}")
    else:
        raise UnsupportedCase(f"unsupported_layout_family:{family}")


def _contiguous_stride(shape: Sequence[int]) -> tuple[int, ...]:
    stride = 1
    result: list[int] = []
    for size in reversed(shape):
        result.append(stride)
        stride *= int(size)
    return tuple(reversed(result))


def _exact_expected_stride(shape: Sequence[int], family: str, expected: Mapping[str, Any]) -> tuple[int, ...]:
    if family == "transpose_noncontiguous":
        base = list(_contiguous_stride([*shape[:-2], shape[-1], shape[-2]]))
        base[-2], base[-1] = base[-1], base[-2]
        return tuple(base)
    if family == "slice_storage_offset":
        return _contiguous_stride([*shape[:-1], int(shape[-1]) + 1])
    if family == "expand_zero_stride":
        dimension = expected.get("expand_dimension")
        if not isinstance(dimension, int) or not 0 <= dimension < len(shape):
            raise UnsupportedCase("manifest_expand_dimension_invalid")
        base = list(_contiguous_stride(shape))
        base[dimension] = 0
        return tuple(base)
    raise UnsupportedCase(f"unsupported_layout_family:{family}")


def _stride_is_contiguous(shape: Sequence[int], strides: Sequence[int]) -> bool:
    required = 1
    for size, stride in zip(reversed(shape), reversed(strides), strict=True):
        if size != 1:
            if stride != required:
                return False
            required *= size
    return True


def _validate_expected_layout(
    input_records: Sequence[Mapping[str, Any]], expected: Any, family: str, total_factory_count: int
) -> list[dict[str, Any]]:
    """Bind source-order factory metadata to observed return-tree leaves.

    Factory indices are AST/source ordered, while a valid ``get_inputs`` may
    return its bindings in another order.  Matching is therefore an exact
    bipartite match over realized metadata, with an explicit returned path
    taking precedence when a future manifest supplies one.
    """

    if not isinstance(expected, list) or len(expected) != len(input_records):
        raise UnsupportedCase("manifest_expected_layout_count_mismatch")
    if not all(isinstance(item, Mapping) for item in expected):
        raise UnsupportedCase("manifest_expected_layout_entry_invalid")
    factory_indices = [item.get("factory_index") for item in expected]
    if (
        not all(type(index) is int for index in factory_indices)
        or factory_indices != sorted(factory_indices)
        or len(set(factory_indices)) != len(factory_indices)
        or any(not 0 <= index < total_factory_count for index in factory_indices)
    ):
        raise UnsupportedCase("manifest_expected_layout_factory_indices_invalid")
    expected_items = list(expected)
    exact_metadata: list[dict[str, Any]] = []
    for item in expected_items:
        shape = item.get("logical_shape")
        if not isinstance(shape, list) or not shape or not all(type(size) is int and size >= 0 for size in shape):
            raise UnsupportedCase("manifest_expected_logical_shape_invalid")
        if not isinstance(item.get("dtype"), str) or not item["dtype"]:
            raise UnsupportedCase("manifest_expected_dtype_invalid")
        expected_stride = _exact_expected_stride(shape, family, item)
        expected_offset = 1 if family == "slice_storage_offset" else 0
        expected_contiguous = _stride_is_contiguous(shape, expected_stride)
        expected_zero_dimensions = [
            index
            for index, (size, stride) in enumerate(zip(shape, expected_stride, strict=True))
            if size > 1 and stride == 0
        ]
        if item.get("expected_strides") != list(expected_stride):
            raise UnsupportedCase(f"manifest_exact_expected_stride_mismatch:factory={item['factory_index']}")
        if item.get("expected_storage_offset") != expected_offset:
            raise UnsupportedCase(f"manifest_exact_expected_storage_offset_mismatch:factory={item['factory_index']}")
        if item.get("expected_is_contiguous") is not expected_contiguous:
            raise UnsupportedCase(f"manifest_exact_contiguity_mismatch:factory={item['factory_index']}")
        if item.get("expected_contiguous") is not expected_contiguous:
            raise UnsupportedCase(f"manifest_compatible_contiguity_mismatch:factory={item['factory_index']}")
        if item.get("expected_storage_offset_positive") != (expected_offset > 0):
            raise UnsupportedCase(f"manifest_compatible_storage_offset_mismatch:factory={item['factory_index']}")
        if item.get("expected_zero_stride_dimensions") != expected_zero_dimensions:
            raise UnsupportedCase(f"manifest_expected_zero_stride_mismatch:factory={item['factory_index']}")
        returned_path = item.get("returned_input_path")
        if returned_path is not None and (not isinstance(returned_path, str) or not returned_path):
            raise UnsupportedCase(f"manifest_returned_input_path_invalid:factory={item['factory_index']}")
        exact_metadata.append(
            {
                "shape": shape,
                "dtype": item["dtype"],
                "stride": list(expected_stride),
                "storage_offset": expected_offset,
                "is_contiguous": expected_contiguous,
                "zero_dimensions": expected_zero_dimensions,
                "returned_input_path": returned_path,
            }
        )

    candidates: dict[int, list[int]] = {}
    for expected_index, metadata in enumerate(exact_metadata):
        candidates[expected_index] = []
        for record_index, record in enumerate(input_records):
            child = record["child"]
            zero_dimensions = [
                index
                for index, (size, stride) in enumerate(zip(child["shape"], child["stride"], strict=True))
                if size > 1 and stride == 0
            ]
            if metadata["returned_input_path"] is not None and record["path"] != metadata["returned_input_path"]:
                continue
            if (
                child["shape"] == metadata["shape"]
                and child["dtype"] == metadata["dtype"]
                and child["stride"] == metadata["stride"]
                and child["storage_offset"] == metadata["storage_offset"]
                and child["is_contiguous"] is metadata["is_contiguous"]
                and zero_dimensions == metadata["zero_dimensions"]
            ):
                candidates[expected_index].append(record_index)
        candidates[expected_index].sort(key=lambda index: str(input_records[index]["path"]))
        if not candidates[expected_index]:
            raise UnsupportedCase(
                f"observed_layout_has_no_exact_manifest_match:factory={expected_items[expected_index]['factory_index']}"
            )

    record_owner: dict[int, int] = {}

    def assign(expected_index: int, visited: set[int]) -> bool:
        for record_index in candidates[expected_index]:
            if record_index in visited:
                continue
            visited.add(record_index)
            owner = record_owner.get(record_index)
            if owner is None or assign(owner, visited):
                record_owner[record_index] = expected_index
                return True
        return False

    assignment_order = sorted(
        candidates, key=lambda index: (len(candidates[index]), expected_items[index]["factory_index"])
    )
    if not all(assign(index, set()) for index in assignment_order):
        raise UnsupportedCase("observed_layout_manifest_bipartite_match_failed")
    expected_to_record = {expected_index: record_index for record_index, expected_index in record_owner.items()}
    return [
        {**dict(input_records[expected_to_record[index]]), "factory_index": expected_items[index]["factory_index"]}
        for index in range(len(expected_items))
    ]


def _compare_input_tree(
    parent: Any, child: Any, *, family: str, path: str, torch: Any
) -> tuple[int, list[dict[str, Any]]]:
    """Prove values and all semantic metadata, admitting only the chosen layout change."""

    if isinstance(parent, torch.Tensor) or isinstance(child, torch.Tensor):
        if not isinstance(parent, torch.Tensor) or not isinstance(child, torch.Tensor):
            raise UnsupportedCase(f"input_tensor_structure_mismatch:{path}")
        same_semantic = (
            tuple(parent.shape) == tuple(child.shape)
            and parent.dtype == child.dtype
            and parent.device == child.device
            and parent.layout == child.layout
            and parent.requires_grad == child.requires_grad
            and parent.is_pinned() == child.is_pinned()
        )
        if not same_semantic:
            raise UnsupportedCase(f"input_semantic_metadata_changed:{path}")
        if not torch.equal(parent, child):
            raise UnsupportedCase(f"input_values_changed:{path}")
        layout_changed = (
            tuple(parent.stride()) != tuple(child.stride()) or parent.storage_offset() != child.storage_offset()
        )
        if layout_changed:
            _target_predicate(parent, child, family, path)
            return 1, [{"path": path, "parent": _tensor_metadata(parent), "child": _tensor_metadata(child)}]
        return 0, []
    if isinstance(parent, Mapping) or isinstance(child, Mapping):
        if not isinstance(parent, Mapping) or not isinstance(child, Mapping) or list(parent) != list(child):
            raise UnsupportedCase(f"input_mapping_structure_mismatch:{path}")
        count, records = 0, []
        for key in parent:
            nested_count, nested_records = _compare_input_tree(
                parent[key], child[key], family=family, path=f"{path}.{key}", torch=torch
            )
            count += nested_count
            records.extend(nested_records)
        return count, records
    if isinstance(parent, (list, tuple)) or isinstance(child, (list, tuple)):
        if type(parent) is not type(child) or len(parent) != len(child):
            raise UnsupportedCase(f"input_sequence_structure_mismatch:{path}")
        count, records = 0, []
        for index, (left, right) in enumerate(zip(parent, child, strict=True)):
            nested_count, nested_records = _compare_input_tree(
                left, right, family=family, path=f"{path}[{index}]", torch=torch
            )
            count += nested_count
            records.extend(nested_records)
        return count, records
    if type(parent) is not type(child) or parent != child:
        raise UnsupportedCase(f"input_non_tensor_changed:{path}")
    return 0, []


def _input_alias_graph(value: Any, *, torch: Any) -> list[list[str]]:
    groups: dict[int, list[str]] = collections.defaultdict(list)
    for tensor, path in _tensor_paths(value, "inputs", torch):
        groups[_storage_key(tensor)].append(path)
    return sorted(sorted(paths) for paths in groups.values())


def _assert_alias_contract(parent: Any, child: Any, target_paths: set[str], torch: Any) -> dict[str, Any]:
    parent_graph = _input_alias_graph(parent, torch=torch)
    child_graph = _input_alias_graph(child, torch=torch)
    if parent_graph != child_graph:
        raise UnsupportedCase("input_alias_graph_changed")
    targets = [tensor for tensor, path in _tensor_paths(child, "inputs", torch) if path in target_paths]
    target_storages = [_storage_key(tensor) for tensor in targets]
    if len(target_storages) != len(set(target_storages)):
        raise UnsupportedCase("target_layout_storages_not_independent")
    return {
        "alias_groups": parent_graph,
        "target_storage_count": len(target_storages),
        "target_storages_independent": True,
    }


def _tree_exact(left: Any, right: Any, *, path: str, torch: Any) -> None:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise UnsupportedCase(f"output_tensor_structure_mismatch:{path}")
        if tuple(left.shape) != tuple(right.shape) or left.dtype != right.dtype or left.device != right.device:
            raise UnsupportedCase(f"output_tensor_metadata_mismatch:{path}")
        if not torch.equal(left, right):
            raise UnsupportedCase(f"output_values_not_exact:{path}")
        return
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping) or list(left) != list(right):
            raise UnsupportedCase(f"output_mapping_structure_mismatch:{path}")
        for key in left:
            _tree_exact(left[key], right[key], path=f"{path}.{key}", torch=torch)
        return
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        if type(left) is not type(right) or len(left) != len(right):
            raise UnsupportedCase(f"output_sequence_structure_mismatch:{path}")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _tree_exact(a, b, path=f"{path}[{index}]", torch=torch)
        return
    if type(left) is not type(right) or left != right:
        raise UnsupportedCase(f"output_non_tensor_mismatch:{path}")


def _flatten_tensors(value: Any, torch: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, Mapping):
        return [tensor for item in value.values() for tensor in _flatten_tensors(item, torch)]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in _flatten_tensors(item, torch)]
    return []


def _storage_key(tensor: Any) -> int:
    return int(tensor.untyped_storage()._cdata)


def _dispatch_op_matches(name: str, candidates: frozenset[str]) -> bool:
    """Match an aten overload without substring false positives.

    ``str(func)`` is normally ``aten.<op>.<overload>``.  Prefix matching at
    the overload separator keeps, for example, ``aten.copy_`` distinct from
    the value-consuming ``aten.copysign``.
    """

    normalized = name.lower()
    return any(normalized == candidate or normalized.startswith(candidate + ".") for candidate in candidates)


def _schema_write_storages(func: Any, args: Any, kwargs: Mapping[str, Any], torch: Any) -> set[int]:
    """Return storages passed to write-role arguments in an operator schema."""

    schema = getattr(func, "_schema", None)
    if schema is None:
        raise UnsupportedCase(f"dispatch_schema_unavailable:{func}")
    positional_index = 0
    storages: set[int] = set()
    for argument in schema.arguments:
        if argument.kwarg_only:
            if argument.name not in kwargs:
                continue
            value = kwargs[argument.name]
        elif positional_index < len(args):
            value = args[positional_index]
            positional_index += 1
        elif argument.name in kwargs:
            value = kwargs[argument.name]
        else:
            continue
        alias_info = argument.alias_info
        if alias_info is None or not alias_info.is_write:
            continue
        storages.update(_storage_key(tensor) for tensor in _flatten_tensors(value, torch))
    return storages


def _schema_has_write_argument(func: Any) -> bool:
    return any(argument.alias_info is not None and argument.alias_info.is_write for argument in func._schema.arguments)


def _schema_has_aliased_tensor_return(func: Any) -> bool:
    return any("Tensor" in str(result.type) and result.alias_info is not None for result in func._schema.returns)


def _schema_has_fresh_tensor_return(func: Any) -> bool:
    return any("Tensor" in str(result.type) and result.alias_info is None for result in func._schema.returns)


def _semantic_dispatch_proven(func: Any, name: str) -> bool:
    normalized = name.lower()
    if _dispatch_op_matches(normalized, _SEMANTIC_DISPATCH_OPS) or normalized.startswith(_SEMANTIC_DISPATCH_PREFIXES):
        return True
    tags = {str(tag) for tag in func.tags}
    if _DATA_DEPENDENT_DISPATCH_TAG in tags:
        return True
    return _schema_has_fresh_tensor_return(func) and bool(tags & _SEMANTIC_DISPATCH_TAGS)


def _layout_trace(
    model: Any, args: list[Any], kwargs: dict[str, Any], target_inputs: list[Any], torch: Any
) -> tuple[Any, dict[str, Any]]:
    """Trace only the direct raw forward and reject erase-before-consume paths."""

    from torch.utils._python_dispatch import TorchDispatchMode

    target_storages = {_storage_key(tensor) for tensor in target_inputs}
    if not target_storages:
        raise UnsupportedCase("no_target_tensors_for_dispatch_trace")
    target_snapshots = [
        {
            "tensor": tensor,
            "storage": _storage_key(tensor),
            "shape": tuple(tensor.shape),
            "stride": tuple(tensor.stride()),
            "storage_offset": int(tensor.storage_offset()),
            "dtype": tensor.dtype,
            "device": tensor.device,
            "layout": tensor.layout,
            "requires_grad": bool(tensor.requires_grad),
            "is_conj": bool(tensor.is_conj()),
            "is_neg": bool(tensor.is_neg()),
            "version": int(tensor._version),
            "values": tensor.detach().clone(),
        }
        for tensor in target_inputs
    ]

    class LayoutTraceMode(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.by_storage = {
                storage: {"semantic_consumers": [], "erase_before_semantic": [], "erase_after_semantic": []}
                for storage in target_storages
            }
            self.calls: list[dict[str, Any]] = []

        def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
            del types
            kwargs = kwargs or {}
            tensors = _flatten_tensors((args, kwargs), torch)
            tracked = sorted({_storage_key(tensor) for tensor in tensors if _storage_key(tensor) in target_storages})
            name = str(func)
            if tracked:
                write_tracked = set(tracked) & _schema_write_storages(func, args, kwargs, torch)
                erase_tracked = set(tracked) if _dispatch_op_matches(name, _ERASE_DISPATCH_OPS) else write_tracked
                if erase_tracked:
                    for storage in sorted(erase_tracked):
                        key = (
                            "erase_after_semantic"
                            if self.by_storage[storage]["semantic_consumers"]
                            else "erase_before_semantic"
                        )
                        self.by_storage[storage][key].append(name)
                semantic_tracked = set(tracked) - erase_tracked
                if semantic_tracked:
                    if _dispatch_op_matches(name, _NONSEMANTIC_DISPATCH_OPS):
                        pass
                    elif _dispatch_op_matches(name, _VIEW_DISPATCH_OPS) or (
                        not _schema_has_write_argument(func) and _schema_has_aliased_tensor_return(func)
                    ):
                        pass
                    elif _semantic_dispatch_proven(func, name):
                        for storage in sorted(semantic_tracked):
                            self.by_storage[storage]["semantic_consumers"].append(name)
                    elif _VIEW_COPY_DISPATCH_TAG in {str(tag) for tag in func.tags}:
                        for storage in sorted(semantic_tracked):
                            key = (
                                "erase_after_semantic"
                                if self.by_storage[storage]["semantic_consumers"]
                                else "erase_before_semantic"
                            )
                            self.by_storage[storage][key].append(name)
                    else:
                        raise UnsupportedCase(f"unclassified_target_dispatch:{name}")
                if len(self.calls) < MAX_RECORDED_DISPATCH_CALLS:
                    self.calls.append(
                        {"operator": name, "target_storage_inputs": [str(storage) for storage in tracked]}
                    )
            return func(*args, **kwargs)

    mode = LayoutTraceMode()
    with torch.no_grad(), mode:
        raw_output = model(*args, **kwargs)
    for snapshot in target_snapshots:
        tensor = snapshot["tensor"]
        if (
            _storage_key(tensor) != snapshot["storage"]
            or tuple(tensor.shape) != snapshot["shape"]
            or tuple(tensor.stride()) != snapshot["stride"]
            or int(tensor.storage_offset()) != snapshot["storage_offset"]
            or tensor.dtype != snapshot["dtype"]
            or tensor.device != snapshot["device"]
            or tensor.layout != snapshot["layout"]
            or bool(tensor.requires_grad) != snapshot["requires_grad"]
            or bool(tensor.is_conj()) != snapshot["is_conj"]
            or bool(tensor.is_neg()) != snapshot["is_neg"]
            or int(tensor._version) != snapshot["version"]
            or not torch.equal(tensor, snapshot["values"])
        ):
            raise UnsupportedCase(f"target_input_mutated:storage={snapshot['storage']}")
    for storage, state in mode.by_storage.items():
        if state["erase_before_semantic"]:
            raise UnsupportedCase(
                f"layout_materialized_before_consumer:storage={storage}:{state['erase_before_semantic'][0]}"
            )
        if not state["semantic_consumers"]:
            raise UnsupportedCase(f"target_layout_not_consumed_by_semantic_operator:storage={storage}")
    per_storage = [
        {
            "storage": str(storage),
            "semantic_consumer_count": len(state["semantic_consumers"]),
            "first_semantic_consumer": state["semantic_consumers"][0],
            "materialization_before_consumer": False,
            "erase_before_semantic_calls": [],
            "erase_after_semantic_calls": state["erase_after_semantic"],
        }
        for storage, state in sorted(mode.by_storage.items())
    ]
    return _snapshot_output(raw_output), {
        "target_storage_count": len(target_storages),
        "target_inputs_immutable": True,
        "per_target_storage": per_storage,
        "recorded_calls": mode.calls,
        "calls_truncated": len(mode.calls) >= MAX_RECORDED_DISPATCH_CALLS,
    }


def _build_model_instance(
    namespace: Mapping[str, Any], device: Any, seed: int, torch: Any
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
    return model, init_inputs, _snapshot_rng_states(str(device))


def _prepare_raw_arm(namespace: Mapping[str, Any], *, device: Any, trial_seed: int, torch: Any) -> dict[str, Any]:
    """Construct an arm, leaving its pristine raw inputs unconsumed."""

    _seed_torch(trial_seed, str(device))
    raw_inputs = _generate_inputs_on_device(namespace, device, torch)
    post_inputs_rng = _snapshot_rng_states(str(device))
    args, kwargs = _raw_forward_arguments(raw_inputs)
    model, init_inputs, construction_rng = _build_model_instance(namespace, device, trial_seed + 1, torch)
    return {
        "raw_inputs": raw_inputs,
        "post_inputs_rng": post_inputs_rng,
        "args": args,
        "kwargs": kwargs,
        "model": model,
        "init_inputs": init_inputs,
        "construction_rng": construction_rng,
    }


def _run_raw_arm(arm: Mapping[str, Any], *, torch: Any) -> Any:
    _restore_rng_states(arm["construction_rng"])
    with torch.no_grad():
        output = _snapshot_output(arm["model"](*arm["args"], **arm["kwargs"]))
    if not _all_finite(output):
        raise UnsupportedCase("non_finite_output")
    return output


def _tensor_paths(value: Any, path: str, torch: Any) -> list[tuple[Any, str]]:
    if isinstance(value, torch.Tensor):
        return [(value, path)]
    if isinstance(value, Mapping):
        return [pair for key, item in value.items() for pair in _tensor_paths(item, f"{path}.{key}", torch)]
    if isinstance(value, (list, tuple)):
        return [pair for index, item in enumerate(value) for pair in _tensor_paths(item, f"{path}[{index}]", torch)]
    return []


def _evaluate_without_guard(payload: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    device = torch.device(str(payload["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device required, got {device}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    family = str(payload["assigned_family"])
    if family not in LAYOUT_FAMILIES:
        raise UnsupportedCase(f"unsupported_layout_family:{family}")
    parent_namespace = _exec_ops_code(
        str(payload["parent_code"]), entry_point=str(payload["entry_point"]), device=str(device)
    )
    child_namespace = _exec_ops_code(
        str(payload["child_code"]), entry_point=str(payload["entry_point"]), device=str(device)
    )
    original_rng = _snapshot_rng_states(str(device))
    trials: list[dict[str, Any]] = []
    try:
        for ordinal in range(int(payload["trials"])):
            trial_seed = int(payload["seed"]) + 10_007 * ordinal
            # Construct every arm first so the layout/value proof observes the
            # pristine tensors, before a model with an in-place forward can see
            # them.  No arm ever shares an input tree with another arm.
            parent_arm = _prepare_raw_arm(parent_namespace, device=device, trial_seed=trial_seed, torch=torch)
            child_control_arm = _prepare_raw_arm(child_namespace, device=device, trial_seed=trial_seed, torch=torch)
            child_trace_arm = _prepare_raw_arm(child_namespace, device=device, trial_seed=trial_seed, torch=torch)
            if not _rng_states_equal(parent_arm["post_inputs_rng"], child_control_arm["post_inputs_rng"], torch):
                raise UnsupportedCase("post_get_inputs_rng_state_changed")
            if not _rng_states_equal(parent_arm["init_inputs"], child_control_arm["init_inputs"], torch):
                raise UnsupportedCase("get_init_inputs_values_changed")
            if not _rng_states_equal(parent_arm["construction_rng"], child_control_arm["construction_rng"], torch):
                raise UnsupportedCase("post_model_construction_rng_state_changed")
            total_factory_count = int(payload["factory_count"])
            tensor_leaf_counts = {
                "parent": len(_tensor_paths(parent_arm["raw_inputs"], "inputs", torch)),
                "child_control": len(_tensor_paths(child_control_arm["raw_inputs"], "inputs", torch)),
                "child_trace": len(_tensor_paths(child_trace_arm["raw_inputs"], "inputs", torch)),
            }
            if set(tensor_leaf_counts.values()) != {total_factory_count}:
                raise UnsupportedCase(
                    f"runtime_tensor_factory_leaf_count_mismatch:{tensor_leaf_counts}:{total_factory_count}"
                )
            transformed, input_records = _compare_input_tree(
                parent_arm["raw_inputs"], child_control_arm["raw_inputs"], family=family, path="inputs", torch=torch
            )
            if transformed != int(payload["transformed_factory_count"]):
                raise UnsupportedCase(
                    f"transformed_factory_count_mismatch:{transformed}:{payload['transformed_factory_count']}"
                )
            input_records = _validate_expected_layout(
                input_records, payload["expected_layout_metadata"], family, total_factory_count
            )
            target_paths = {record["path"] for record in input_records}
            alias_contract = _assert_alias_contract(
                parent_arm["raw_inputs"], child_control_arm["raw_inputs"], target_paths, torch
            )
            traced_transformed, traced_records = _compare_input_tree(
                parent_arm["raw_inputs"], child_trace_arm["raw_inputs"], family=family, path="inputs", torch=torch
            )
            if traced_transformed != transformed:
                raise UnsupportedCase("traced_child_layout_realization_changed")
            traced_records = _validate_expected_layout(
                traced_records, payload["expected_layout_metadata"], family, total_factory_count
            )
            if [(item["factory_index"], item["path"]) for item in traced_records] != [
                (item["factory_index"], item["path"]) for item in input_records
            ]:
                raise UnsupportedCase("traced_child_factory_path_binding_changed")
            _assert_alias_contract(parent_arm["raw_inputs"], child_trace_arm["raw_inputs"], target_paths, torch)
            if not _rng_states_equal(
                child_control_arm["post_inputs_rng"], child_trace_arm["post_inputs_rng"], torch
            ) or not _rng_states_equal(child_control_arm["init_inputs"], child_trace_arm["init_inputs"], torch):
                raise UnsupportedCase("traced_child_reconstruction_changed_rng_or_init_inputs")
            parent_output = _run_raw_arm(parent_arm, torch=torch)
            child_control_output = _run_raw_arm(child_control_arm, torch=torch)
            targets = [
                tensor
                for tensor, path in _tensor_paths(child_trace_arm["raw_inputs"], "inputs", torch)
                if path in target_paths
            ]
            _restore_rng_states(child_trace_arm["construction_rng"])
            child_trace_output, trace_record = _layout_trace(
                child_trace_arm["model"], child_trace_arm["args"], child_trace_arm["kwargs"], targets, torch
            )
            if not _all_finite(child_trace_output):
                raise UnsupportedCase("non_finite_traced_output")
            _tree_exact(parent_output, child_control_output, path="output", torch=torch)
            _tree_exact(child_control_output, child_trace_output, path="traced_output", torch=torch)
            trials.append(
                {
                    "ordinal": ordinal,
                    "seed": trial_seed,
                    "input_layouts": input_records,
                    "input_alias_contract": alias_contract,
                    "factory_count": total_factory_count,
                    "transformed_factory_count": transformed,
                    "raw_parent_child_output_exact": True,
                    "raw_child_control_trace_output_exact": True,
                    "semantic_dispatch": trace_record,
                }
            )
            del parent_output, child_control_output, child_trace_output, parent_arm, child_control_arm, child_trace_arm
            torch.cuda.empty_cache()
    finally:
        _restore_rng_states(original_rng)
    return {
        "status": "passed",
        "passed": True,
        "reason": None,
        "trials": trials,
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
            "assigned_family",
            "factory_count",
            "transformed_factory_count",
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
    except BaseException as exc:  # noqa: BLE001 - row failure is evidence
        result = {"status": "failed", "passed": False, "reason": _exception_detail(exc)}
    return {
        "contract_version": CONTRACT_VERSION,
        **_identity(payload),
        **result,
        "duration_seconds": time.monotonic() - started,
    }


def _run_subprocess(payload: Mapping[str, Any], timeout_seconds: float) -> dict[str, Any]:
    global _ACTIVE_WORKER_PROCESS
    started = time.monotonic()
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--_worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )  # noqa: S603
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


def _validate_canonical_source_binding(manifests: Sequence[Mapping[str, Any]]) -> None:
    paths = {item.get("source_artifact_path") for item in manifests}
    if len(paths) != 1 or not isinstance(next(iter(paths)), str):
        raise ValueError("manifest canonical source path is not unique")
    source = Path(next(iter(paths)))
    if not source.is_file():
        raise ValueError(f"canonical source artifact unavailable:{source}")
    if pq.ParquetFile(source).metadata.num_rows != EXPECTED_CANONICAL_PARENT_ROWS:
        raise ValueError("canonical source row count mismatch")
    if _sha256_file(source) != EXPECTED_CANONICAL_PARENT_SHA256:
        raise ValueError("canonical source SHA-256 mismatch")


def _tasks(parents_path: Path, children_path: Path, manifest_path: Path) -> list[dict[str, Any]]:
    parent_count = pq.ParquetFile(parents_path).metadata.num_rows
    child_count = pq.ParquetFile(children_path).metadata.num_rows
    if not 1 <= parent_count <= MAX_AUTHORIZED_CANDIDATES or parent_count != child_count:
        raise ValueError(f"bounded parent/child artifact count invalid:{parent_count}:{child_count}")
    parents, children = _rows(parents_path), _rows(children_path)
    manifests = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not (len(parents) == len(children) == len(manifests)):
        raise ValueError(f"aligned artifact count mismatch:{len(parents)}:{len(children)}:{len(manifests)}")
    _validate_canonical_source_binding(manifests)
    tasks: list[dict[str, Any]] = []
    for index, (parent, child, manifest) in enumerate(zip(parents, children, manifests, strict=True)):
        parent_uuid, child_uuid = _nested(parent, "extra_info.uuid"), _nested(child, "extra_info.uuid")
        parent_code, child_code = _nested(parent, "reward_model.ground_truth"), _nested(
            child, "reward_model.ground_truth"
        )
        entry_point = _nested(child, "extra_info.entry_point", "Model")
        family = manifest.get("assigned_family", manifest.get("assigned_target"))
        realized = manifest.get("realized_intervention")
        expected_layout = manifest.get("expected_layout_metadata")
        source_hashes = manifest.get("source_hashes")
        if (
            manifest.get("candidate_row_index") != index
            or manifest.get("parent_uuid") != parent_uuid
            or manifest.get("child_uuid") != child_uuid
        ):
            raise ValueError(f"manifest identity mismatch:{index}")
        if manifest.get("manifest_contract_version") != "layout_lane_manifest_v1":
            raise ValueError(f"manifest contract mismatch:{index}")
        if manifest.get("source_artifact_sha256") != EXPECTED_CANONICAL_PARENT_SHA256 or not isinstance(
            source_hashes, Mapping
        ):
            raise ValueError(f"manifest canonical source SHA mismatch:{index}")
        if (
            source_hashes.get("canonical_parent_artifact") != EXPECTED_CANONICAL_PARENT_SHA256
            or source_hashes.get("parent_reference") != manifest.get("parent_reference_sha256")
            or source_hashes.get("child_reference") != manifest.get("child_reference_sha256")
        ):
            raise ValueError(f"manifest source-hash identity mismatch:{index}")
        if (
            not isinstance(manifest.get("source_row_index"), int)
            or not 0 <= manifest["source_row_index"] < EXPECTED_CANONICAL_PARENT_ROWS
        ):
            raise ValueError(f"manifest canonical row identity mismatch:{index}")
        if manifest.get("primary_intervention") != "layout" or family not in LAYOUT_FAMILIES:
            raise ValueError(f"manifest layout family mismatch:{index}:{family}")
        if (
            not isinstance(realized, Mapping)
            or realized.get("family", realized.get("assigned_family", family)) != family
        ):
            raise ValueError(f"manifest realized family mismatch:{index}")
        if not all(isinstance(value, str) and value for value in (parent_code, child_code, entry_point)):
            raise ValueError(f"invalid code fields:{index}")
        if _sha256_bytes(parent_code.encode()) != manifest.get("parent_reference_sha256") or _sha256_bytes(
            child_code.encode()
        ) != manifest.get("child_reference_sha256"):
            raise ValueError(f"manifest reference hash mismatch:{index}")
        total_count = realized.get("factory_count") if isinstance(realized, Mapping) else None
        transformed_count = manifest.get("transformed_factory_count")
        if not isinstance(total_count, int) or total_count <= 0:
            raise ValueError(f"invalid total factory_count:{index}")
        if not isinstance(transformed_count, int) or not 0 < transformed_count <= total_count:
            raise ValueError(f"invalid transformed_factory_count:{index}")
        if not isinstance(expected_layout, list) or len(expected_layout) != transformed_count:
            raise ValueError(f"manifest expected layout metadata invalid:{index}")
        if realized.get("expected_metadata") != expected_layout:
            raise ValueError(f"manifest intervention expected metadata mismatch:{index}")
        if "transformed_factory_count" in realized and realized.get("transformed_factory_count") != transformed_count:
            raise ValueError(f"manifest intervention transformed count mismatch:{index}")
        child_augmentation = _nested(child, "extra_info.augmentation")
        if (
            not isinstance(child_augmentation, Mapping)
            or child_augmentation.get("contract_version") != manifest.get("generator_contract_version")
            or child_augmentation.get("parent_uuid") != parent_uuid
            or child_augmentation.get("child_uuid") != child_uuid
            or child_augmentation.get("parent_reference_sha256") != manifest.get("parent_reference_sha256")
            or child_augmentation.get("child_reference_sha256") != manifest.get("child_reference_sha256")
            or child_augmentation.get("factory_count") != total_count
        ):
            raise ValueError(f"child augmentation identity mismatch:{index}")
        child_v4 = _nested(child, "extra_info.v4")
        if (
            not isinstance(child_v4, Mapping)
            or child_v4.get("parent_uuid") != parent_uuid
            or child_v4.get("reference_sha256") != manifest.get("child_reference_sha256")
            or child_v4.get("included_in_review_train") is not False
        ):
            raise ValueError(f"child v4 identity mismatch:{index}")
        tasks.append(
            {
                "candidate_row_index": index,
                "parent_uuid": parent_uuid,
                "child_uuid": child_uuid,
                "parent_code": parent_code,
                "child_code": child_code,
                "entry_point": entry_point,
                "assigned_family": family,
                "factory_count": total_count,
                "transformed_factory_count": transformed_count,
                "expected_layout_metadata": expected_layout,
                "source_artifact_sha256": manifest["source_artifact_sha256"],
                "source_row_index": manifest["source_row_index"],
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
        print(RESULT_MARKER + json.dumps(_worker(json.loads(sys.stdin.read())), ensure_ascii=False))
        return
    args = _parser().parse_args(raw_argv)
    _install_driver_signal_handlers()
    if args.trials < MIN_LIVENESS_TRIALS or args.seed != REQUIRED_LIVENESS_SEED or args.timeout_seconds <= 0:
        raise ValueError("require trials >= 3, seed exactly 17, and positive timeout")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must satisfy 0 <= index < shard-count")
    if len(args.launcher_sha256) != 64 or any(char not in "0123456789abcdef" for char in args.launcher_sha256):
        raise ValueError("launcher-sha256 must be 64 lowercase hexadecimal characters")
    tasks = _tasks(args.parents, args.children, args.manifest)
    allowlist = _load_allowlist(args.child_uuid_file)
    if allowlist is not None:
        known = {str(task["child_uuid"]) for task in tasks}
        if missing := allowlist - known:
            raise ValueError(f"allowlisted child UUIDs not found:{sorted(missing)[:10]}")
        tasks = [task for task in tasks if str(task["child_uuid"]) in allowlist]
    if not 1 <= len(tasks) <= MAX_AUTHORIZED_CANDIDATES:
        raise ValueError(f"selected candidate count must be in [1, {MAX_AUTHORIZED_CANDIDATES}]:{len(tasks)}")
    tasks = [task for index, task in enumerate(tasks) if index % args.shard_count == args.shard_index]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        tasks = tasks[: args.limit]
    if not tasks:
        raise ValueError("shard selection produced no layout validation tasks")
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
            "raw_direct_invocation": True,
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
            raise RuntimeError(f"another process owns output:{args.output}") from exc
        handle.seek(0)
        prior: dict[str, dict[str, Any]] = {}
        by_uuid = {str(task["child_uuid"]): task for task in tasks}
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            uuid = str(record.get("child_uuid"))
            if (
                uuid not in by_uuid
                or uuid in prior
                or record.get("validation_binding_sha256") != binding
                or type(record.get("passed")) is not bool
            ):
                raise ValueError(f"invalid resume record:{args.output}:{line_number}")
            prior[uuid] = record
        handle.seek(0, os.SEEK_END)
        for task in tasks:
            uuid = str(task["child_uuid"])
            if uuid in prior:
                resumed += 1
                counts[str(prior[uuid]["status"])] += 1
                continue
            result = _run_subprocess(
                {**task, "device": args.device, "trials": args.trials, "seed": args.seed}, args.timeout_seconds
            )
            result.update(evidence)
            result["validation_binding_sha256"] = binding
            _append(handle, result)
            counts[str(result["status"])] += 1
            executed += 1
            print(
                json.dumps(
                    {"child_uuid": uuid, "status": result.get("status"), "reason": result.get("reason")},
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
                "validation_binding_sha256": binding,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
