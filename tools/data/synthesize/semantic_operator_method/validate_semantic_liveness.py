#!/usr/bin/env python3
"""Prove declared ATen execution and returned-output provenance on CUDA.

Each candidate runs in a bounded subprocess.  Two identically initialized,
persistent train-mode models execute the same three-trial sequence: one is a
control and one is wrapped in ``TorchDispatchMode``.  A row passes only when
every declared op executes on every trial and its produced Tensor provenance
reaches the single returned Tensor without changing output or registered
state.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
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
    _clone_value,
    _exec_ops_code,
    _normalize_forward_inputs,
    _restore_rng_states,
    _seed_torch,
    _snapshot_output,
    _snapshot_rng_states,
    _to_device,
)
from tools.data.synthesize.semantic_operator_method import generate_semantic_operator as generator  # noqa: E402
from tools.data.synthesize.validate_train_mode_contract import (  # noqa: E402
    _cuda_memory_guard,
    _CudaMemoryGuardFailure,
)

CONTRACT_VERSION = "semantic_operator_runtime_liveness_v1"
RUN_BINDING_VERSION = "semantic_operator_runtime_binding_v1"
RESULT_MARKER = "__SEMANTIC_OPERATOR_RESULT__="
MAX_DEVICE_MEMORY_GIB = 64.0
MAX_AUTHORIZED_CANDIDATES = 5_000
MIN_LIVENESS_TRIALS = 3
REQUIRED_LIVENESS_SEED = 17
MAX_RECORDED_DISPATCH_CALLS = 192
EXECUTION_CONTROLS = {
    "control_trace_comparison": "exact",
    "cudnn_benchmark": False,
    "cudnn_deterministic": True,
}
_DRIVER_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
_ACTIVE_WORKER_PROCESS: subprocess.Popen[str] | None = None

_ALLOCATION_SCHEMAS = frozenset(
    {
        "aten::empty",
        "aten::empty_like",
        "aten::empty_strided",
        "aten::full",
        "aten::full_like",
        "aten::new_empty",
        "aten::new_empty_strided",
        "aten::new_full",
        "aten::new_ones",
        "aten::new_zeros",
        "aten::ones",
        "aten::ones_like",
        "aten::rand",
        "aten::rand_like",
        "aten::randn",
        "aten::randn_like",
        "aten::randint",
        "aten::zeros",
        "aten::zeros_like",
    }
)
_METADATA_SCHEMAS = frozenset(
    {
        "aten::_shape_as_tensor",
        "aten::dim",
        "aten::is_contiguous",
        "aten::numel",
        "aten::size",
        "aten::storage_offset",
        "aten::stride",
        "aten::sym_numel",
        "aten::sym_size",
        "aten::sym_storage_offset",
        "aten::sym_stride",
    }
)
_EXPLICIT_VALUE_SCHEMAS = frozenset(
    {
        "aten::_adaptive_avg_pool2d",
        "aten::_log_softmax",
        "aten::_native_batch_norm_legit",
        "aten::_native_batch_norm_legit_functional",
        "aten::_native_multi_head_attention",
        "aten::_softmax",
        "aten::_cudnn_rnn",
        "aten::adaptive_avg_pool2d",
        "aten::add",
        "aten::addmm",
        "aten::alias",
        "aten::amax",
        "aten::avg_pool1d",
        "aten::avg_pool2d",
        "aten::bmm",
        "aten::cat",
        "aten::clamp",
        "aten::constant_pad_nd",
        "aten::convolution",
        "aten::cudnn_batch_norm",
        "aten::embedding",
        "aten::expand",
        "aten::gather",
        "aten::gelu",
        "aten::gru",
        "aten::index_select",
        "aten::linear",
        "aten::logsumexp",
        "aten::lstm",
        "aten::matmul",
        "aten::max_pool1d_with_indices",
        "aten::max_pool2d_with_indices",
        "aten::mean",
        "aten::mm",
        "aten::mul",
        "aten::native_batch_norm",
        "aten::native_group_norm",
        "aten::native_layer_norm",
        "aten::nll_loss_forward",
        "aten::permute",
        "aten::relu",
        "aten::reshape",
        "aten::scatter",
        "aten::select",
        "aten::silu",
        "aten::slice",
        "aten::split",
        "aten::split_with_sizes",
        "aten::squeeze",
        "aten::sum",
        "aten::tanh",
        "aten::take_along_dim",
        "aten::transpose",
        "aten::unfold",
        "aten::unsqueeze",
        "aten::upsample_bilinear2d",
        "aten::upsample_nearest2d",
        "aten::view",
    }
)
_EXPLICIT_VALUE_PREFIXES = ("aten::_scaled_dot_product_", "aten::scaled_dot_product_")
_VALUE_TAGS = frozenset({"Tag.core", "Tag.pointwise", "Tag.reduction", "Tag.data_dependent_output"})


class UnsupportedCase(RuntimeError):
    """A candidate failed the strict semantic/operator proof contract."""


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
        raise KeyboardInterrupt(f"semantic liveness driver received signal {signum}")

    for handled in _DRIVER_SIGNALS:
        signal.signal(handled, handler)


def _flatten_tensors(value: Any, torch: Any) -> list[Any]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, Mapping):
        return [tensor for item in value.values() for tensor in _flatten_tensors(item, torch)]
    if isinstance(value, (list, tuple)):
        return [tensor for item in value for tensor in _flatten_tensors(item, torch)]
    return []


def _schema_identity(func: Any) -> tuple[str, str]:
    schema = func._schema
    return str(schema.name), str(schema.overload_name or "")


def _schema_has_write(func: Any) -> bool:
    return any(argument.alias_info is not None and argument.alias_info.is_write for argument in func._schema.arguments)


def _schema_has_alias_return(func: Any) -> bool:
    return any("Tensor" in str(result.type) and result.alias_info is not None for result in func._schema.returns)


def _value_propagating(func: Any, schema: str) -> bool:
    if schema in _EXPLICIT_VALUE_SCHEMAS or schema.startswith(_EXPLICIT_VALUE_PREFIXES):
        return True
    if _schema_has_alias_return(func):
        return True
    return bool({str(tag) for tag in getattr(func, "tags", ())} & _VALUE_TAGS)


def _tensor_exact(left: Any, right: Any, torch: Any) -> bool:
    return (
        isinstance(left, torch.Tensor)
        and isinstance(right, torch.Tensor)
        and left.shape == right.shape
        and left.dtype == right.dtype
        and left.device == right.device
        and torch.equal(left, right)
    )


def _tree_exact(left: Any, right: Any, torch: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return _tensor_exact(left, right, torch)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and list(left) == list(right)
            and all(_tree_exact(left[key], right[key], torch) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_tree_exact(a, b, torch) for a, b in zip(left, right, strict=True))
        )
    return left == right


def _assert_input_unchanged(before: Any, after: Any, torch: Any) -> None:
    if not _tree_exact(before, after, torch):
        raise UnsupportedCase("forward_input_mutated")


def _assert_no_unregistered_tensors(model: Any, torch: Any) -> dict[str, list[str]]:
    registered_tensor_ids = {id(tensor) for _, tensor in model.named_parameters(recurse=True, remove_duplicate=False)}
    registered_tensor_ids.update(id(tensor) for _, tensor in model.named_buffers(recurse=True, remove_duplicate=False))
    registered_module_ids = {id(module) for module in model.modules()}
    registered_tensor_attribute_paths: list[str] = []
    registered_module_attribute_paths: list[str] = []
    seen_modules: set[int] = set()
    seen_containers: set[int] = set()

    def inspect(value: Any, path: str) -> None:
        if isinstance(value, torch.Tensor):
            if id(value) in registered_tensor_ids:
                registered_tensor_attribute_paths.append(path)
                return
            raise UnsupportedCase(f"unregistered_tensor_state:{path}")
        if isinstance(value, torch.nn.Module):
            if id(value) in registered_module_ids:
                registered_module_attribute_paths.append(path)
                return
            raise UnsupportedCase(f"unregistered_module_state:{path}")
        if isinstance(value, (str, bytes, int, float, complex, bool, type(None), torch.device, torch.dtype)):
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
        for name, value in vars(module).items():
            if name not in {"_parameters", "_buffers", "_modules"}:
                inspect(value, f"{path}.{name}")
        for name, child in module._modules.items():
            if child is not None:
                visit(child, f"{path}.{name}")

    visit(model, "model")
    return {
        "registered_module_attribute_paths": sorted(registered_module_attribute_paths),
        "registered_tensor_attribute_paths": sorted(registered_tensor_attribute_paths),
    }


def _state_snapshot(model: Any, torch: Any) -> dict[str, Any]:
    registered_attribute_aliases = _assert_no_unregistered_tensors(model, torch)
    parameters = dict(model.named_parameters(recurse=True, remove_duplicate=False))
    buffers = dict(model.named_buffers(recurse=True, remove_duplicate=False))
    overlap = set(parameters) & set(buffers)
    if overlap:
        raise UnsupportedCase(f"state_name_kind_overlap:{sorted(overlap)}")
    entries: dict[str, dict[str, Any]] = {}
    object_groups: dict[int, list[str]] = collections.defaultdict(list)
    storage_groups: dict[int, list[str]] = collections.defaultdict(list)
    for kind, values in (("parameter", parameters), ("buffer", buffers)):
        for name, tensor in values.items():
            key = f"{kind}:{name}"
            if tensor.layout != torch.strided or tensor.is_sparse or tensor.is_quantized:
                raise UnsupportedCase(f"unsupported_registered_state:{key}:{tensor.layout}")
            if tensor.dtype.is_complex:
                raise UnsupportedCase(f"complex_registered_state:{key}")
            entries[key] = {
                "metadata": {
                    "kind": kind,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "stride": list(tensor.stride()),
                    "storage_offset": int(tensor.storage_offset()),
                    "requires_grad": bool(tensor.requires_grad),
                },
                "value": tensor.detach().clone(),
            }
            object_groups[id(tensor)].append(key)
            storage_groups[int(tensor.untyped_storage()._cdata)].append(key)
    return {
        "entries": entries,
        "object_alias_groups": sorted(sorted(group) for group in object_groups.values() if len(group) > 1),
        "storage_alias_groups": sorted(sorted(group) for group in storage_groups.values() if len(group) > 1),
        **registered_attribute_aliases,
    }


def _state_schema(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "entries": {key: value["metadata"] for key, value in snapshot["entries"].items()},
        "object_alias_groups": snapshot["object_alias_groups"],
        "storage_alias_groups": snapshot["storage_alias_groups"],
        "registered_module_attribute_paths": snapshot["registered_module_attribute_paths"],
        "registered_tensor_attribute_paths": snapshot["registered_tensor_attribute_paths"],
    }


def _state_values_equal(left: Mapping[str, Any], right: Mapping[str, Any], torch: Any) -> bool:
    if _state_schema(left) != _state_schema(right):
        return False
    return all(torch.equal(left["entries"][key]["value"], right["entries"][key]["value"]) for key in left["entries"])


def _state_changed(before: Mapping[str, Any], after: Mapping[str, Any], torch: Any) -> bool:
    if _state_schema(before) != _state_schema(after):
        raise UnsupportedCase("registered_state_schema_or_alias_changed")
    return any(
        not torch.equal(before["entries"][key]["value"], after["entries"][key]["value"]) for key in before["entries"]
    )


def _build_pair(namespace: Mapping[str, Any], device: Any, seed: int, torch: Any) -> tuple[Any, Any, dict[str, Any]]:
    _seed_torch(seed, str(device))
    raw_init = namespace["get_init_inputs"]()
    raw_init = [] if raw_init is None else raw_init
    if not isinstance(raw_init, (list, tuple)):
        raise UnsupportedCase("get_init_inputs_is_not_list_or_tuple")
    init_inputs = _to_device(list(raw_init), str(device))
    _seed_torch(seed + 1, str(device))
    control = _build_model(namespace["__entry_point__"], _clone_value(init_inputs))
    _seed_torch(seed + 1, str(device))
    traced = _build_model(namespace["__entry_point__"], _clone_value(init_inputs))
    if not isinstance(control, torch.nn.Module) or not isinstance(traced, torch.nn.Module):
        raise UnsupportedCase("entry_point_did_not_build_module")
    control = control.to(device).train(True)
    traced = traced.to(device).train(True)
    control_state = _state_snapshot(control, torch)
    traced_state = _state_snapshot(traced, torch)
    if not _state_values_equal(control_state, traced_state, torch):
        raise UnsupportedCase("identically_seeded_initial_state_differs")
    return control, traced, control_state


@contextlib.contextmanager
def _deterministic_cudnn_controls(torch: Any) -> Any:
    previous_deterministic = bool(torch.backends.cudnn.deterministic)
    previous_benchmark = bool(torch.backends.cudnn.benchmark)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        observed = {
            "control_trace_comparison": "exact",
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        }
        if observed != EXECUTION_CONTROLS:
            raise UnsupportedCase(f"cudnn_execution_controls_not_applied:{observed}")
        yield observed
    finally:
        torch.backends.cudnn.deterministic = previous_deterministic
        torch.backends.cudnn.benchmark = previous_benchmark


def _invoke(model: Any, normalized: Sequence[Any]) -> Any:
    if len(normalized) == 2 and isinstance(normalized[0], (list, tuple)) and isinstance(normalized[1], Mapping):
        return model(*normalized[0], **normalized[1])
    return model(*normalized)


def _trace_forward(
    model: Any, inputs: Sequence[Any], declared_ops: Sequence[Mapping[str, Any]], torch: Any
) -> tuple[Any, dict[str, Any]]:
    from torch.utils._python_dispatch import TorchDispatchMode

    expected_ids = {str(item["op_id"]) for item in declared_ops}
    identity_to_ids: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    minimums: dict[str, int] = {}
    for item in declared_ops:
        op_id = str(item["op_id"])
        minimums[op_id] = int(item["min_calls_per_trial"])
        for identity in item["runtime_identities"]:
            identity_to_ids[(str(identity["schema"]), str(identity["overload"]))].add(op_id)

    class ProvenanceMode(TorchDispatchMode):
        def __init__(self) -> None:
            super().__init__()
            self.tags: dict[int, frozenset[str]] = {}
            self.refs: list[Any] = []
            self.calls: collections.Counter[str] = collections.Counter()
            self.matches: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
            self.sequence: list[dict[str, Any]] = []

        def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
            del types
            kwargs = kwargs or {}
            schema, overload = _schema_identity(func)
            identity_text = f"{schema}.{overload or '<default>'}"
            input_tensors = _flatten_tensors((args, kwargs), torch)
            input_tags = frozenset().union(*(self.tags.get(id(tensor), frozenset()) for tensor in input_tensors))
            matched = frozenset(identity_to_ids.get((schema, overload), set()))
            if input_tags and _schema_has_write(func):
                raise UnsupportedCase(f"tainted_inplace_or_write_dispatch:{identity_text}")
            output = func(*args, **kwargs)
            output_tensors = _flatten_tensors(output, torch)
            if input_tags:
                if schema in _ALLOCATION_SCHEMAS or schema in _METADATA_SCHEMAS:
                    propagated = frozenset()
                elif _value_propagating(func, schema):
                    propagated = input_tags
                else:
                    raise UnsupportedCase(f"unclassified_tainted_dispatch:{identity_text}")
                if propagated and not output_tensors:
                    raise UnsupportedCase(f"tainted_dispatch_has_no_tensor_output:{identity_text}")
            else:
                propagated = frozenset()
            output_tags = propagated | matched
            for tensor in output_tensors:
                self.tags[id(tensor)] = self.tags.get(id(tensor), frozenset()) | output_tags
                self.refs.append(tensor)
            self.calls[identity_text] += 1
            for op_id in matched:
                self.matches[op_id][identity_text] += 1
            if len(self.sequence) < MAX_RECORDED_DISPATCH_CALLS:
                self.sequence.append(
                    {
                        "identity": identity_text,
                        "input_declared_op_ids": sorted(input_tags),
                        "matched_declared_op_ids": sorted(matched),
                        "output_declared_op_ids": sorted(output_tags),
                    }
                )
            return output

    mode = ProvenanceMode()
    with torch.no_grad(), mode:
        raw_output = _invoke(model, inputs)
    if not isinstance(raw_output, torch.Tensor):
        raise UnsupportedCase(f"final_output_not_single_tensor:{type(raw_output).__name__}")
    final_tags = mode.tags.get(id(raw_output), frozenset())
    if missing := expected_ids - set(final_tags):
        raise UnsupportedCase(f"declared_ops_not_in_returned_output:{sorted(missing)}")
    per_op: list[dict[str, Any]] = []
    for op_id in sorted(expected_ids):
        matches = mode.matches.get(op_id, collections.Counter())
        calls = sum(matches.values())
        if calls < minimums[op_id]:
            raise UnsupportedCase(f"declared_op_call_count_below_minimum:{op_id}:{calls}:{minimums[op_id]}")
        per_op.append(
            {
                "op_id": op_id,
                "calls": calls,
                "minimum_calls": minimums[op_id],
                "matched_identities": dict(sorted(matches.items())),
                "returned_output_witness": op_id in final_tags,
            }
        )
    return raw_output, {
        "final_output_declared_op_ids": sorted(final_tags),
        "per_declared_op": per_op,
        "total_dispatch_calls": sum(mode.calls.values()),
        "dispatch_identity_histogram": dict(sorted(mode.calls.items())),
        "dispatch_sequence_sha256": _canonical_sha256(mode.sequence),
        "recorded_dispatch_calls": mode.sequence,
        "dispatch_calls_truncated": sum(mode.calls.values()) > MAX_RECORDED_DISPATCH_CALLS,
    }


def _evaluate_without_guard(payload: Mapping[str, Any], torch: Any, device: Any) -> dict[str, Any]:
    namespace = _exec_ops_code(
        str(payload["reference_code"]), entry_point=str(payload["entry_point"]), device=str(device)
    )
    control, traced, initial_state = _build_pair(namespace, device, int(payload["seed"]) + 100_000, torch)
    initial_schema_sha256 = _canonical_sha256(_state_schema(initial_state))
    trials: list[dict[str, Any]] = []
    any_state_mutation = False
    for ordinal in range(int(payload["trials"])):
        trial_seed = int(payload["seed"]) + ordinal * 10_007
        _seed_torch(trial_seed, str(device))
        raw_inputs = namespace["get_inputs"]()
        raw_inputs = [] if raw_inputs is None else raw_inputs
        normalized = _normalize_forward_inputs(raw_inputs, str(device))
        control_inputs = _clone_value(normalized)
        traced_inputs = _clone_value(normalized)
        control_before = _clone_value(control_inputs)
        traced_before = _clone_value(traced_inputs)
        paired_rng = _snapshot_rng_states(str(device))

        _restore_rng_states(paired_rng)
        with torch.no_grad():
            control_raw = _invoke(control, control_inputs)
        if not isinstance(control_raw, torch.Tensor):
            raise UnsupportedCase(f"control_final_output_not_single_tensor:{type(control_raw).__name__}")
        control_output = _snapshot_output(control_raw)
        control_state = _state_snapshot(control, torch)
        _assert_input_unchanged(control_before, control_inputs, torch)

        _restore_rng_states(paired_rng)
        traced_raw, trace = _trace_forward(traced, traced_inputs, payload["declared_ops"], torch)
        traced_output = _snapshot_output(traced_raw)
        traced_state = _state_snapshot(traced, torch)
        _assert_input_unchanged(traced_before, traced_inputs, torch)

        if not _all_finite(control_output) or not _all_finite(traced_output):
            raise UnsupportedCase("non_finite_output")
        if not _tensor_exact(control_output, traced_output, torch):
            raise UnsupportedCase("control_trace_output_not_exact")
        if not _state_values_equal(control_state, traced_state, torch):
            raise UnsupportedCase("control_trace_registered_state_not_exact")
        state_mutated = _state_changed(initial_state, control_state, torch)
        any_state_mutation = any_state_mutation or state_mutated
        trials.append(
            {
                "ordinal": ordinal,
                "seed": trial_seed,
                "single_tensor_output": True,
                "output_shape": list(control_output.shape),
                "output_dtype": str(control_output.dtype),
                "output_finite": True,
                "control_trace_output_exact": True,
                "control_trace_state_exact": True,
                "inputs_immutable": True,
                "registered_state_mutated_from_initial": state_mutated,
                "trace": trace,
            }
        )
    expected_mutation = str(payload["mode_behavior"]) == "train_stateful"
    if expected_mutation and not any_state_mutation:
        raise UnsupportedCase("declared_train_stateful_but_registered_state_did_not_mutate")
    if not control.training or not traced.training:
        raise UnsupportedCase("persistent_model_left_train_mode")
    return {
        "status": "passed",
        "passed": True,
        "persistent_model_instances": True,
        "training_mode_preserved": True,
        "final_output_kind": "single_tensor",
        "registered_state_schema_sha256": initial_schema_sha256,
        "registered_state_entry_count": len(initial_state["entries"]),
        "registered_object_alias_evidence": {
            "module_attribute_paths": initial_state["registered_module_attribute_paths"],
            "tensor_attribute_paths": initial_state["registered_tensor_attribute_paths"],
        },
        "state_mutation_expected": expected_mutation,
        "state_mutation_observed": any_state_mutation,
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
    if device.type != "cuda" or not torch.cuda.is_available():
        raise UnsupportedCase(f"CUDA device required:{device}")
    torch.cuda.set_device(device)
    with _deterministic_cudnn_controls(torch) as execution_controls:
        with _cuda_memory_guard(torch=torch, device=device, max_device_memory_gib=MAX_DEVICE_MEMORY_GIB) as guard:
            result = _evaluate_without_guard(payload, torch, device)
    result["execution_controls"] = execution_controls
    result["memory_guard"] = guard
    return result


def _identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "candidate_row_index": payload.get("candidate_row_index"),
        "uuid": payload.get("uuid"),
        "template_id": payload.get("template_id"),
        "primary_family": payload.get("primary_family"),
        "mode_behavior": payload.get("mode_behavior"),
    }


def _worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    try:
        result = _evaluate(payload)
    except _CudaMemoryGuardFailure as exc:
        result = {**exc.result_record(), "reason": exc.failure_reason}
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

    started = time.monotonic()
    process: subprocess.Popen[str] | None = None
    try:
        process = subprocess.Popen(  # noqa: S603 - fixed interpreter and script
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


def _tasks(candidates_path: Path, manifest_path: Path) -> list[dict[str, Any]]:
    parquet = pq.ParquetFile(candidates_path)
    if not 1 <= parquet.metadata.num_rows <= MAX_AUTHORIZED_CANDIDATES:
        raise ValueError(f"bounded candidate count invalid:{parquet.metadata.num_rows}")
    candidates = pq.read_table(candidates_path).to_pylist()
    manifests = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(candidates) != len(manifests):
        raise ValueError(f"candidate/manifest count mismatch:{len(candidates)}:{len(manifests)}")
    current_generator_sha = _sha256_file(Path(generator.__file__).resolve())
    tasks: list[dict[str, Any]] = []
    for index, (row, manifest) in enumerate(zip(candidates, manifests, strict=True)):
        uuid = _nested(row, "extra_info.uuid")
        code = _nested(row, "reward_model.ground_truth")
        entry_point = _nested(row, "extra_info.entry_point")
        declared_ops = manifest.get("declared_ops")
        if manifest.get("candidate_row_index") != index or manifest.get("uuid") != uuid:
            raise ValueError(f"manifest identity mismatch:{index}")
        if (
            manifest.get("manifest_contract_version") != generator.MANIFEST_VERSION
            or manifest.get("generator_contract_version") != generator.CONTRACT_VERSION
            or manifest.get("runtime_contract_version") != generator.RUNTIME_CONTRACT_VERSION
            or manifest.get("primary_intervention") != "semantic_operator"
        ):
            raise ValueError(f"manifest contract mismatch:{index}")
        if manifest.get("generator_source_sha256") != current_generator_sha:
            raise ValueError(f"generator source binding mismatch:{index}")
        if manifest.get("parent_uuid") is not None or manifest.get("lineage_kind") != "standalone_semantic_synthetic":
            raise ValueError(f"parentless lineage mismatch:{index}")
        if manifest.get("training_approved") is not False or manifest.get("structured_output_deferred") is not True:
            raise ValueError(f"governance/structured decision mismatch:{index}")
        if manifest.get("final_output_contract") != {"kind": "single_tensor", "finite_required": True}:
            raise ValueError(f"final output contract mismatch:{index}")
        if not all(isinstance(value, str) and value for value in (uuid, code, entry_point)):
            raise ValueError(f"invalid candidate code identity:{index}")
        if _sha256_bytes(code.encode()) != manifest.get("reference_sha256"):
            raise ValueError(f"reference hash mismatch:{index}")
        if not isinstance(declared_ops, list) or not declared_ops:
            raise ValueError(f"declared ops missing:{index}")
        seen_ids: set[str] = set()
        for op in declared_ops:
            if not isinstance(op, Mapping) or set(op) != {
                "op_id",
                "source_calls",
                "runtime_identities",
                "min_calls_per_trial",
                "must_reach_returned_output",
            }:
                raise ValueError(f"declared op schema mismatch:{index}")
            op_id = op.get("op_id")
            if not isinstance(op_id, str) or not op_id or op_id in seen_ids:
                raise ValueError(f"declared op id invalid:{index}:{op_id!r}")
            seen_ids.add(op_id)
            if type(op.get("min_calls_per_trial")) is not int or op["min_calls_per_trial"] <= 0:
                raise ValueError(f"declared op minimum invalid:{index}:{op_id}")
            if op.get("must_reach_returned_output") is not True:
                raise ValueError(f"declared op output contract invalid:{index}:{op_id}")
            identities = op.get("runtime_identities")
            if not isinstance(identities, list) or not identities:
                raise ValueError(f"declared op identities absent:{index}:{op_id}")
            for identity in identities:
                if (
                    not isinstance(identity, Mapping)
                    or set(identity) != {"schema", "overload"}
                    or not isinstance(identity.get("schema"), str)
                    or not identity["schema"].startswith("aten::")
                    or not isinstance(identity.get("overload"), str)
                ):
                    raise ValueError(f"declared ATen identity invalid:{index}:{op_id}:{identity!r}")
        tasks.append(
            {
                "candidate_row_index": index,
                "uuid": uuid,
                "reference_code": code,
                "entry_point": entry_point,
                "template_id": manifest["template_id"],
                "primary_family": manifest["primary_family"],
                "mode_behavior": manifest["mode_behavior"],
                "declared_ops": declared_ops,
            }
        )
    return tasks


def _load_allowlist(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    values = {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if not values:
        raise ValueError("UUID allowlist is empty")
    return values


def _append(handle: Any, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--uuid-file", type=Path)
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


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv == ["--_worker"]:
        with contextlib.redirect_stdout(sys.stderr):
            result = _worker(json.loads(sys.stdin.read()))
        print(RESULT_MARKER + json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    args = _parser().parse_args(raw_argv)
    _install_driver_signal_handlers()
    if args.trials < MIN_LIVENESS_TRIALS or args.seed != REQUIRED_LIVENESS_SEED or args.timeout_seconds <= 0:
        raise ValueError("require trials >= 3, seed exactly 17, and positive timeout")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must satisfy 0 <= index < shard-count")
    if len(args.launcher_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in args.launcher_sha256
    ):
        raise ValueError("launcher-sha256 must be 64 lowercase hexadecimal characters")
    tasks = _tasks(args.candidates, args.manifest)
    allowlist = _load_allowlist(args.uuid_file)
    if allowlist is not None:
        known = {str(task["uuid"]) for task in tasks}
        if missing := allowlist - known:
            raise ValueError(f"allowlisted UUIDs not found:{sorted(missing)[:10]}")
        tasks = [task for task in tasks if str(task["uuid"]) in allowlist]
    if not 1 <= len(tasks) <= MAX_AUTHORIZED_CANDIDATES:
        raise ValueError(f"selected candidate count must be in [1,{MAX_AUTHORIZED_CANDIDATES}]:{len(tasks)}")
    tasks = [task for task in tasks if int(task["candidate_row_index"]) % args.shard_count == args.shard_index]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        tasks = tasks[: args.limit]
    if not tasks:
        raise ValueError("shard selection produced no semantic validation tasks")
    evidence = {
        "contract_version": CONTRACT_VERSION,
        "binding_version": RUN_BINDING_VERSION,
        "validator_source_sha256": _sha256_file(Path(__file__).resolve()),
        "generator_source_sha256": _sha256_file(Path(generator.__file__).resolve()),
        "launcher_source_sha256": args.launcher_sha256,
        "candidates_sha256": _sha256_file(args.candidates),
        "manifest_sha256": _sha256_file(args.manifest),
        "allowlist_sha256": _sha256_file(args.uuid_file) if args.uuid_file else None,
        "validation_config": {
            "device": args.device,
            "trials": args.trials,
            "seed": args.seed,
            "timeout_seconds": args.timeout_seconds,
            "max_device_memory_gib": MAX_DEVICE_MEMORY_GIB,
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
            raise RuntimeError(f"another process owns output:{args.output}") from exc
        handle.seek(0)
        prior: dict[str, dict[str, Any]] = {}
        selected_by_uuid = {str(task["uuid"]): task for task in tasks}
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            uuid = str(record.get("uuid"))
            if (
                uuid not in selected_by_uuid
                or uuid in prior
                or record.get("validation_binding_sha256") != binding
                or type(record.get("passed")) is not bool
            ):
                raise ValueError(f"invalid resume record:{args.output}:{line_number}")
            prior[uuid] = record
        handle.seek(0, os.SEEK_END)
        for task in tasks:
            uuid = str(task["uuid"])
            if uuid in prior:
                resumed += 1
                counts["passed" if prior[uuid]["passed"] else "failed"] += 1
                continue
            payload = {
                **task,
                "device": args.device,
                "trials": args.trials,
                "seed": args.seed,
                "max_device_memory_gib": MAX_DEVICE_MEMORY_GIB,
            }
            result = _run_subprocess(payload, args.timeout_seconds)
            result.update(
                {
                    "validation_binding_sha256": binding,
                    "binding_evidence": evidence,
                    "candidates_sha256": evidence["candidates_sha256"],
                    "manifest_sha256": evidence["manifest_sha256"],
                    "validator_source_sha256": evidence["validator_source_sha256"],
                    "generator_source_sha256": evidence["generator_source_sha256"],
                    "launcher_source_sha256": evidence["launcher_source_sha256"],
                    "validation_config": evidence["validation_config"],
                }
            )
            _append(handle, result)
            executed += 1
            counts["passed" if result.get("passed") is True else "failed"] += 1
    summary = {
        "contract_version": CONTRACT_VERSION,
        "validation_binding_sha256": binding,
        "launcher_source_sha256": args.launcher_sha256,
        "validator_source_sha256": evidence["validator_source_sha256"],
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "selected": len(tasks),
        "executed": executed,
        "resumed": resumed,
        "passed": counts["passed"],
        "failed": counts["failed"],
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
