#!/usr/bin/env python3
"""Reject solver children whose newly added input region cannot affect output.

Each child runs in an isolated CUDA subprocess.  For every trial, one baseline
and one unchanged control are shared by all slot probes.  Every slot then gets
its own changed-region arm starting from the same inputs, initialized model,
and RNG state; a live slot can therefore never hide a dead sibling through a
union perturbation.  Only tensor elements added beyond that slot's parent value
are perturbed.  Floating and complex values use a seeded, per-element, bounded
non-affine mixture; discrete values use seeded biased resampling inside their
observed range.  A child passes only when the unchanged control is exactly
stable and every slot affects output in every trial.  Ambiguous mappings and
unsupported values fail closed.

The reader remains compatible with the v3 solver's singular ``slot`` record,
preserves the v4 solver's exact-two ``slots`` contract, and accepts any
non-empty ``slots`` record from newer solver contracts.  Output JSONL is
append-resumable only under an exact validator/config/artifact binding.
"""

from __future__ import annotations

import argparse
import ast
import collections
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.cleaning.runtime_validation import (
    _all_finite,
    _build_model,
    _clone_inputs,
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
from tools.data.synthesize.augment_prompt_tasks import (
    _SHAPE_FACTORIES,
    _call_name,
    _top_level_function,
)
from tools.data.synthesize.solve_shape_coverage import _returned_factory_parameters
from tools.data.synthesize.validate_train_mode_contract import (
    _CudaMemoryGuardFailure,
    _cuda_memory_guard,
)

CONTRACT_VERSION = "shape_changed_region_liveness_v3"
PERTURBATION_CONTRACT_VERSION = "seeded_bounded_non_affine_mix_v1"
RUN_BINDING_CONTRACT_VERSION = "shape_changed_region_liveness_run_binding_v2"
PERTURBATION_CHUNK_ELEMENTS = 1 << 20
RESULT_MARKER = "__SHAPE_REGION_LIVENESS_RESULT__="
MAX_DEVICE_MEMORY_GIB = 64.0
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


class UnsupportedCase(RuntimeError):
    """The changed-region experiment cannot establish a reliable verdict."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nested(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _exception_detail(exc: BaseException) -> str:
    return f"{type(exc).__name__}:{str(exc).replace(chr(10), ' ')[:2_000]}"


def _canonical_json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _exact_integer(
    values: Sequence[Any],
    *,
    field: str,
    child_uuid: str,
) -> int:
    invalid = [
        value for value in values if value is not None and type(value) is not int
    ]
    if invalid:
        raise ValueError(
            f"child {child_uuid} has non-integer {field} values: {invalid}"
        )
    integers = [value for value in values if type(value) is int]
    if not integers:
        raise ValueError(f"child {child_uuid} has no integer {field}")
    if len(set(integers)) != 1:
        raise ValueError(
            f"child {child_uuid} has conflicting {field} values: {integers}"
        )
    return integers[0]


def _normalize_solver_slot(
    raw: Mapping[str, Any],
    *,
    child_uuid: str,
) -> dict[str, Any]:
    """Normalize a legacy singular slot or one multi-slot wrapper."""

    nested_slot = raw.get("slot")
    slot = nested_slot if isinstance(nested_slot, Mapping) else raw
    slot_id = slot.get("slot_id")
    occurrences = slot.get("occurrences")
    if not isinstance(slot_id, str) or not slot_id:
        raise ValueError(f"child {child_uuid} has an invalid slot_id")
    if not isinstance(occurrences, list) or not occurrences:
        raise ValueError(f"child {child_uuid} slot {slot_id} has no occurrences")
    old_value = _exact_integer(
        (slot.get("old_value"), raw.get("old_value")),
        field=f"old_value for slot {slot_id}",
        child_uuid=child_uuid,
    )
    new_value = _exact_integer(
        (
            raw.get("new_value"),
            raw.get("slot_value"),
            slot.get("new_value"),
            slot.get("slot_value"),
        ),
        field=f"new_value for slot {slot_id}",
        child_uuid=child_uuid,
    )
    if not 0 < old_value < new_value:
        raise ValueError(
            f"child {child_uuid} slot {slot_id} is not a strict expansion:"
            f"{old_value}:{new_value}"
        )
    result = {
        "slot_id": slot_id,
        "old_value": old_value,
        "new_value": new_value,
        "occurrences": occurrences,
    }
    if "power_of_two" in raw:
        if type(raw["power_of_two"]) is not bool:
            raise ValueError(
                f"child {child_uuid} slot {slot_id} has invalid power_of_two"
            )
        result["power_of_two"] = raw["power_of_two"]
    return result


def _solver_slots_from_container(
    container: Mapping[str, Any],
    *,
    child_uuid: str,
) -> list[dict[str, Any]] | None:
    raw_slots = container.get("slots")
    if raw_slots is not None:
        if not isinstance(raw_slots, list):
            raise ValueError(f"child {child_uuid} solver slots must be a list")
        result: list[dict[str, Any]] = []
        for raw in raw_slots:
            if not isinstance(raw, Mapping):
                raise ValueError(f"child {child_uuid} has a non-object solver slot")
            result.append(_normalize_solver_slot(raw, child_uuid=child_uuid))
        return result
    raw_slot = container.get("slot")
    if raw_slot is None:
        return None
    if not isinstance(raw_slot, Mapping):
        raise ValueError(f"child {child_uuid} singular solver slot must be an object")
    return [
        _normalize_solver_slot(
            {
                "slot": raw_slot,
                "new_value": container.get("new_value"),
                "slot_value": container.get("slot_value"),
            },
            child_uuid=child_uuid,
        )
    ]


def _decision_solver_slots(
    decision: Mapping[str, Any],
    accepted_attempt: Mapping[str, Any],
    *,
    child_uuid: str,
    solver_contract_version: Any,
) -> list[dict[str, Any]]:
    sources: list[tuple[str, list[dict[str, Any]]]] = []
    for label, container in (
        ("decision.solver", decision.get("solver")),
        ("accepted attempt", accepted_attempt),
    ):
        if not isinstance(container, Mapping):
            if container is not None:
                raise ValueError(f"child {child_uuid} {label} must be an object")
            continue
        slots = _solver_slots_from_container(container, child_uuid=child_uuid)
        if slots is not None:
            sources.append((label, slots))
    if not sources:
        raise ValueError(f"child {child_uuid} has no solver slot evidence")
    expected = sources[0][1]
    for label, slots in sources[1:]:
        if slots != expected:
            raise ValueError(
                f"child {child_uuid} solver slots conflict between "
                f"{sources[0][0]} and {label}"
            )
    if not expected:
        raise ValueError(
            f"child {child_uuid} solver slots must be non-empty"
        )
    if solver_contract_version == "shape_multidim_solver_v4" and len(expected) != 2:
        raise ValueError(
            f"child {child_uuid} shape_multidim_solver_v4 must have exactly "
            f"two slots; found {len(expected)}"
        )
    slot_ids = [str(slot["slot_id"]) for slot in expected]
    if len(slot_ids) != len(set(slot_ids)):
        raise ValueError(f"child {child_uuid} has duplicate solver slot IDs")
    return expected


def _forward_parameter_positions(code: str, entry_point: str) -> dict[str, int]:
    tree = ast.parse(code)
    model = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == entry_point
        ),
        None,
    )
    if model is None:
        return {}
    forward = next(
        (
            node
            for node in model.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "forward"
        ),
        None,
    )
    if forward is None:
        return {}
    positional = list(forward.args.posonlyargs) + list(forward.args.args)
    if not positional:
        return {}
    return {
        argument.arg: index for index, argument in enumerate(positional[1:])
    }


def _factory_argument_positions(code: str, entry_point: str) -> dict[int, int]:
    tree = ast.parse(code)
    get_inputs_node = _top_level_function(tree, "get_inputs")
    if not isinstance(get_inputs_node, ast.FunctionDef):
        return {}
    calls = sorted(
        (
            node
            for node in ast.walk(get_inputs_node)
            if isinstance(node, ast.Call) and _call_name(node.func) in _SHAPE_FACTORIES
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    factory_parameters = _returned_factory_parameters(
        tree, get_inputs_node, entry_point, calls
    )
    parameter_positions = _forward_parameter_positions(code, entry_point)
    return {
        factory_index: parameter_positions[parameter]
        for factory_index, parameter in factory_parameters.items()
        if parameter in parameter_positions
    }


def _logical_positional_argument(inputs: Sequence[Any], index: int) -> Any:
    if (
        len(inputs) == 2
        and isinstance(inputs[0], (list, tuple))
        and isinstance(inputs[1], Mapping)
    ):
        if index >= len(inputs[0]):
            raise UnsupportedCase("mapped_forward_argument_is_not_positional")
        return inputs[0][index]
    if index >= len(inputs):
        raise UnsupportedCase("mapped_forward_argument_out_of_range")
    return inputs[index]


def _expanded_arguments(
    parent_code: str,
    entry_point: str,
    occurrences: Sequence[Mapping[str, Any]],
    old_value: int,
    new_value: int,
) -> dict[int, list[tuple[int, int, int]]]:
    if not 0 < old_value < new_value:
        raise UnsupportedCase("slot_is_not_a_strict_expansion")
    factory_positions = _factory_argument_positions(parent_code, entry_point)
    result: dict[int, set[tuple[int, int, int]]] = collections.defaultdict(set)
    for occurrence in occurrences:
        factory_index = occurrence.get("factory_index")
        axis = occurrence.get("axis")
        if type(factory_index) is not int or type(axis) is not int:
            raise UnsupportedCase("invalid_slot_occurrence")
        if factory_index not in factory_positions:
            raise UnsupportedCase(
                f"factory_{factory_index}_cannot_map_to_positional_forward_argument"
            )
        result[factory_positions[factory_index]].add((axis, old_value, new_value))
    if not result:
        raise UnsupportedCase("slot_has_no_mapped_occurrences")
    return {index: sorted(specs) for index, specs in result.items()}


def _expanded_arguments_by_slot(
    parent_code: str,
    entry_point: str,
    slots: Sequence[Mapping[str, Any]],
) -> list[tuple[Mapping[str, Any], dict[int, list[tuple[int, int, int]]]]]:
    expanded_slots: list[
        tuple[Mapping[str, Any], dict[int, list[tuple[int, int, int]]]]
    ] = []
    occupied_axes: dict[tuple[int, int], str] = {}
    for slot in slots:
        slot_id = str(slot["slot_id"])
        expanded = _expanded_arguments(
            parent_code,
            entry_point,
            slot["occurrences"],
            int(slot["old_value"]),
            int(slot["new_value"]),
        )
        for argument_index, specs in expanded.items():
            for axis, _, _ in specs:
                key = (argument_index, axis)
                prior_slot_id = occupied_axes.get(key)
                if prior_slot_id is not None:
                    raise UnsupportedCase(
                        "multiple_slots_expand_one_forward_argument_axis:"
                        f"{prior_slot_id}:{slot_id}:{argument_index}:{axis}"
                    )
                occupied_axes[key] = slot_id
        expanded_slots.append((slot, expanded))
    return expanded_slots


def _chunked_views(tensor: Any) -> Iterator[Any]:
    """Yield bounded-size views without materializing a flattened copy."""

    pending = [tensor]
    while pending:
        view = pending.pop()
        if view.numel() <= PERTURBATION_CHUNK_ELEMENTS:
            yield view
            continue
        axis = max(range(view.ndim), key=lambda index: int(view.shape[index]))
        axis_size = int(view.shape[axis])
        if axis_size <= 1:
            raise UnsupportedCase("cannot_chunk_large_expanded_region")
        split = axis_size // 2
        pending.append(view.narrow(axis, split, axis_size - split))
        pending.append(view.narrow(axis, 0, split))


def _perturbation_seed(
    tensor: Any,
    *,
    axis: int,
    old_value: int,
    new_value: int,
    trial: int,
) -> int:
    material = "|".join(
        (
            PERTURBATION_CONTRACT_VERSION,
            str(trial),
            str(axis),
            str(old_value),
            str(new_value),
            str(tensor.dtype),
            ",".join(str(int(value)) for value in tensor.shape),
        )
    )
    return int.from_bytes(hashlib.sha256(material.encode("utf-8")).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def _finite_component_statistics(component: Any, torch: Any) -> dict[str, float]:
    minimum, maximum = torch.aminmax(component)
    standard_deviation, mean = torch.std_mean(component, correction=0)
    result = {
        "minimum": float(minimum.item()),
        "maximum": float(maximum.item()),
        "mean": float(mean.item()),
        "standard_deviation": float(standard_deviation.item()),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise UnsupportedCase("non_finite_floating_expanded_region")
    return result


def _constant_float_alternative(value: float) -> float:
    if value == 0.0:
        return 0.25
    return value * 0.5


def _floating_target_bounds(
    statistics: Mapping[str, float], *, trial: int
) -> tuple[float, float]:
    minimum = statistics["minimum"]
    maximum = statistics["maximum"]
    mean = statistics["mean"]
    standard_deviation = statistics["standard_deviation"]
    if minimum == maximum:
        alternative = _constant_float_alternative(minimum)
        if not math.isfinite(alternative) or alternative == minimum:
            raise UnsupportedCase("constant_float_has_no_safe_alternative")
        return min(minimum, alternative), max(minimum, alternative)

    # Each trial shifts the target mean and changes its variance.  The target
    # interval stays inside the region's observed range; the final convex mix
    # therefore stays inside it too.  Deliberately non-zero mean shifts prevent
    # very large reductions from rounding an otherwise centered probe away.
    mean_shift = (-0.375, 0.25, 0.5)[trial % 3]
    half_width = (0.625, 0.75, 0.5)[trial % 3]
    lower = max(minimum, mean + (mean_shift - half_width) * standard_deviation)
    upper = min(maximum, mean + (mean_shift + half_width) * standard_deviation)
    if not lower < upper:
        lower, upper = minimum, maximum
    if not lower < upper:
        raise UnsupportedCase("floating_region_has_no_representable_target_range")
    return lower, upper


def _floating_region_evidence(view: Any, *, trial: int, torch: Any) -> dict[str, Any]:
    if view.dtype.is_complex:
        real_statistics = _finite_component_statistics(view.real, torch)
        imaginary_statistics = _finite_component_statistics(view.imag, torch)
        return {
            "method": "bounded_complex_component_random_mix",
            "real": {
                **real_statistics,
                "target_bounds": list(
                    _floating_target_bounds(real_statistics, trial=trial)
                ),
            },
            "imaginary": {
                **imaginary_statistics,
                "target_bounds": list(
                    _floating_target_bounds(imaginary_statistics, trial=trial)
                ),
            },
        }
    statistics = _finite_component_statistics(view, torch)
    return {
        "method": "bounded_float_random_mix",
        **statistics,
        "target_bounds": list(_floating_target_bounds(statistics, trial=trial)),
    }


def _random_target_like(
    chunk: Any,
    evidence: Mapping[str, Any],
    *,
    generator: Any,
    trial: int,
    torch: Any,
) -> Any:
    target = torch.empty(chunk.shape, dtype=chunk.dtype, device=chunk.device)
    if chunk.dtype == torch.bool:
        probability = (0.125, 0.625, 0.875)[trial % 3]
        target.bernoulli_(probability, generator=generator)
        return target
    if chunk.dtype.is_complex:
        real_lower, real_upper = evidence["real"]["target_bounds"]
        imaginary_lower, imaginary_upper = evidence["imaginary"]["target_bounds"]
        target.real.uniform_(real_lower, real_upper, generator=generator)
        target.imag.uniform_(imaginary_lower, imaginary_upper, generator=generator)
        return target
    if chunk.dtype.is_floating_point:
        lower, upper = evidence["target_bounds"]
        target.uniform_(lower, upper, generator=generator)
        return target

    minimum = int(evidence["minimum"])
    maximum = int(evidence["maximum"])
    if minimum == maximum:
        info = torch.iinfo(chunk.dtype)
        alternative = minimum - 1 if minimum > 0 else minimum + 1
        if not info.min <= alternative <= info.max:
            alternative = minimum + 1 if minimum < info.max else minimum - 1
        if not info.min <= alternative <= info.max or alternative == minimum:
            raise UnsupportedCase("constant_integer_has_no_representable_alternative")
        target.fill_(minimum)
        mask = torch.empty(chunk.shape, dtype=torch.bool, device=chunk.device)
        mask.bernoulli_((0.25, 0.625, 0.875)[trial % 3], generator=generator)
        target.masked_fill_(mask, alternative)
        return target

    # Endpoint resampling is range-preserving even for an int64 range whose
    # width cannot itself be represented.  Trial-specific probabilities also
    # avoid preserving a pre-existing integer histogram by construction.
    target.fill_(minimum)
    mask = torch.empty(chunk.shape, dtype=torch.bool, device=chunk.device)
    mask.bernoulli_((0.125, 0.625, 0.875)[trial % 3], generator=generator)
    target.masked_fill_(mask, maximum)
    return target


def _perturb_tensor_region(
    tensor: Any,
    specs: Sequence[tuple[int, int, int]],
    *,
    trial: int,
    torch: Any,
) -> tuple[int, int, list[dict[str, Any]]]:
    if not isinstance(tensor, torch.Tensor):
        raise UnsupportedCase("mapped_forward_argument_is_not_a_tensor")
    if tensor.layout != torch.strided:
        raise UnsupportedCase(f"unsupported_input_layout:{tensor.layout}")
    normalized: list[tuple[int, int, int]] = []
    for axis, old_value, new_value in specs:
        resolved_axis = axis if axis >= 0 else tensor.ndim + axis
        if not 0 <= resolved_axis < tensor.ndim:
            raise UnsupportedCase(f"expanded_axis_out_of_range:{axis}:{tensor.ndim}")
        if tensor.shape[resolved_axis] != new_value:
            raise UnsupportedCase(
                f"expanded_axis_size_mismatch:{resolved_axis}:"
                f"{tensor.shape[resolved_axis]}:{new_value}"
            )
        normalized.append((resolved_axis, old_value, new_value))
    if len({axis for axis, _, _ in normalized}) != len(normalized):
        raise UnsupportedCase("conflicting_expansions_on_one_tensor_axis")

    changed_elements = 0
    region_elements = 0
    region_records: list[dict[str, Any]] = []
    previous_axes: list[tuple[int, int]] = []
    for axis, old_value, new_value in sorted(normalized):
        slices = [slice(None)] * tensor.ndim
        for previous_axis, previous_old_value in previous_axes:
            slices[previous_axis] = slice(0, previous_old_value)
        slices[axis] = slice(old_value, new_value)
        view = tensor[tuple(slices)]
        if view.numel() == 0:
            raise UnsupportedCase("expanded_region_is_empty")
        region_elements += view.numel()
        seed = _perturbation_seed(
            tensor,
            axis=axis,
            old_value=old_value,
            new_value=new_value,
            trial=trial,
        )
        generator = torch.Generator(device=tensor.device)
        generator.manual_seed(seed)
        if view.dtype == torch.bool:
            evidence: dict[str, Any] = {
                "method": "biased_boolean_resample",
                "true_probability": (0.125, 0.625, 0.875)[trial % 3],
            }
        elif view.dtype.is_complex or view.dtype.is_floating_point:
            evidence = _floating_region_evidence(view, trial=trial, torch=torch)
            evidence["mix_weight"] = (0.5, 0.625, 0.75)[trial % 3]
        else:
            minimum = int(view.min().item())
            maximum = int(view.max().item())
            evidence = {
                "method": (
                    "biased_integer_endpoint_resample"
                    if minimum < maximum
                    else "constant_integer_neighbor_resample"
                ),
                "minimum": minimum,
                "maximum": maximum,
                "upper_probability": (0.125, 0.625, 0.875)[trial % 3],
            }

        region_changed = 0
        chunks = 0
        for chunk in _chunked_views(view):
            target = _random_target_like(
                chunk,
                evidence,
                generator=generator,
                trial=trial,
                torch=torch,
            )
            if chunk.dtype.is_complex or chunk.dtype.is_floating_point:
                target.lerp_(chunk, 1.0 - float(evidence["mix_weight"]))
            region_changed += int(torch.count_nonzero(chunk != target).item())
            chunk.copy_(target)
            chunks += 1
            del target
        if region_changed == 0:
            # Tiny or low-precision regions can coincidentally resample their
            # original values.  Force one bounded difference so the seeded
            # probe never becomes a no-op merely by chance.
            first = view[(0,) * view.ndim]
            original_first = first.item()
            if view.dtype == torch.bool:
                first.logical_not_()
            elif view.dtype.is_complex:
                candidate = complex(
                    evidence["real"]["target_bounds"][0],
                    evidence["imaginary"]["target_bounds"][0],
                )
                if first.item() == candidate:
                    candidate = complex(
                        evidence["real"]["target_bounds"][1],
                        evidence["imaginary"]["target_bounds"][1],
                    )
                first.fill_(candidate)
            elif view.dtype.is_floating_point:
                candidate = evidence["target_bounds"][0]
                if first.item() == candidate:
                    candidate = evidence["target_bounds"][1]
                first.fill_(candidate)
            else:
                current = int(first.item())
                minimum = int(evidence["minimum"])
                maximum = int(evidence["maximum"])
                if minimum < maximum:
                    first.fill_(maximum if current != maximum else minimum)
                else:
                    info = torch.iinfo(view.dtype)
                    alternative = current - 1 if current > 0 else current + 1
                    if not info.min <= alternative <= info.max:
                        alternative = current + 1 if current < info.max else current - 1
                    first.fill_(alternative)
            if first.item() == original_first:
                raise UnsupportedCase("region_perturbation_has_no_representable_difference")
            region_changed = 1
            evidence["forced_single_difference"] = True
        changed_elements += region_changed
        region_records.append(
            {
                "axis": axis,
                "old_value": old_value,
                "new_value": new_value,
                "dtype": str(view.dtype),
                "seed": seed,
                "chunks": chunks,
                "changed_elements": region_changed,
                "region_elements": int(view.numel()),
                **evidence,
            }
        )
        previous_axes.append((axis, old_value))
    return changed_elements, region_elements, region_records


def _validate_output_tree(value: Any, torch: Any) -> None:
    if isinstance(value, torch.Tensor):
        if value.layout != torch.strided:
            raise UnsupportedCase(f"unsupported_output_layout:{value.layout}")
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _validate_output_tree(item, torch)
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _validate_output_tree(item, torch)
        return
    if value is None or isinstance(value, (bool, int, float, complex, str, bytes)):
        return
    raise UnsupportedCase(f"unsupported_output_type:{type(value).__qualname__}")


def _run_arm(model: Any, inputs: Sequence[Any], rng_state: Mapping[str, Any]) -> Any:
    import copy
    import torch

    trial_model = None
    try:
        trial_model = copy.deepcopy(model)
        _restore_rng_states(rng_state)
        with torch.no_grad():
            output = _snapshot_output(_invoke_model(trial_model, inputs))
        _validate_output_tree(output, torch)
        if not _all_finite(output):
            raise UnsupportedCase("non_finite_output")
        return output
    finally:
        del trial_model


def _generate_inputs_on_device(namespace: Mapping[str, Any], device: Any, torch: Any) -> Any:
    previous_device = torch.get_default_device()
    torch.set_default_device(device)
    try:
        return namespace["get_inputs"]()
    finally:
        torch.set_default_device(previous_device)


def _evaluate_without_memory_guard(payload: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = str(payload["device"])
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise ValueError(f"device must be CUDA, got {resolved_device}")
    torch.cuda.set_device(resolved_device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    parent_code = str(payload["parent_code"])
    child_code = str(payload["child_code"])
    entry_point = str(payload["entry_point"])
    slots = payload.get("slots")
    if not isinstance(slots, list) or not slots:
        raise UnsupportedCase("worker_requires_nonempty_solver_slots")
    if not all(isinstance(slot, Mapping) for slot in slots):
        raise UnsupportedCase("worker_slot_is_not_an_object")
    expanded_slots = _expanded_arguments_by_slot(
        parent_code,
        entry_point,
        slots,
    )
    namespace = _exec_ops_code(child_code, entry_point=entry_point, device=device)
    trials = int(payload["trials"])
    seed = int(payload["seed"])
    trial_records: list[dict[str, Any]] = []
    slot_effects: dict[str, list[bool]] = {
        str(slot["slot_id"]): [] for slot, _ in expanded_slots
    }
    original_rng = _snapshot_rng_states(device)
    try:
        for trial in range(trials):
            trial_seed = seed + 10_007 * trial
            _seed_torch(trial_seed, device)
            init_inputs = namespace["get_init_inputs"]()
            init_inputs = [] if init_inputs is None else init_inputs
            if not isinstance(init_inputs, (list, tuple)):
                raise UnsupportedCase("get_init_inputs_is_not_list_or_tuple")
            init_inputs = _to_device(list(init_inputs), device)
            base_model = _build_model(namespace["__entry_point__"], init_inputs)
            if not isinstance(base_model, torch.nn.Module):
                raise UnsupportedCase("entry_point_did_not_build_torch_module")
            base_model = base_model.to(device)
            base_model.train(True)
            del init_inputs

            raw_inputs = _generate_inputs_on_device(
                namespace, resolved_device, torch
            )
            inputs = _normalize_forward_inputs(raw_inputs, device)
            del raw_inputs
            forward_rng = _snapshot_rng_states(device)
            baseline = _run_arm(base_model, inputs, forward_rng)
            control = _run_arm(base_model, inputs, forward_rng)
            if not _outputs_allclose(baseline, control, rtol=0.0, atol=0.0):
                raise UnsupportedCase("paired_unchanged_control_is_not_exact")
            del control
            slot_records: list[dict[str, Any]] = []
            for slot, expanded in expanded_slots:
                slot_id = str(slot["slot_id"])
                perturbed = _clone_inputs(inputs)
                argument_records: list[dict[str, Any]] = []
                changed_region_elements = 0
                expanded_region_elements = 0
                for argument_index, specs in sorted(expanded.items()):
                    target = _logical_positional_argument(perturbed, argument_index)
                    (
                        changed_elements,
                        region_elements,
                        perturbation_evidence,
                    ) = _perturb_tensor_region(
                        target,
                        specs,
                        trial=trial,
                        torch=torch,
                    )
                    changed_region_elements += changed_elements
                    expanded_region_elements += region_elements
                    argument_records.append(
                        {
                            "forward_argument_index": argument_index,
                            "expanded_axes": [axis for axis, _, _ in specs],
                            "changed_region_elements": changed_elements,
                            "expanded_region_elements": region_elements,
                            "perturbation": perturbation_evidence,
                        }
                    )
                    del target
                changed_output = _run_arm(base_model, perturbed, forward_rng)
                output_changed = not _outputs_allclose(
                    baseline, changed_output, rtol=0.0, atol=0.0
                )
                activity = _output_change_activity(
                    baseline, changed_output, rtol=0.0, atol=0.0
                )
                slot_records.append(
                    {
                        "slot_id": slot_id,
                        "old_value": int(slot["old_value"]),
                        "new_value": int(slot["new_value"]),
                        "arguments": argument_records,
                        "changed_region_elements": changed_region_elements,
                        "expanded_region_elements": expanded_region_elements,
                        "output_changed": output_changed,
                        "changed_output_elements": activity[0],
                        "output_elements": activity[1],
                        "max_changed_output_fraction": activity[2],
                    }
                )
                slot_effects[slot_id].append(output_changed)
                del changed_output, perturbed
            trial_record: dict[str, Any] = {
                "trial": trial,
                "seed": trial_seed,
                "slots": slot_records,
            }
            if len(slot_records) == 1:
                # Preserve the v2 evidence convenience field for singular v3
                # tasks while making the slot boundary explicit in v3.
                trial_record["arguments"] = slot_records[0]["arguments"]
            trial_records.append(trial_record)
            del baseline, inputs, base_model
            torch.cuda.empty_cache()
    finally:
        _restore_rng_states(original_rng)
    dead_slots = [
        slot_id for slot_id, effects in slot_effects.items() if not any(effects)
    ]
    inconsistent_slots = [
        slot_id
        for slot_id, effects in slot_effects.items()
        if any(effects) and not all(effects)
    ]
    if dead_slots:
        return {
            "status": "rejected",
            "passed": False,
            "reason": "expanded_region_no_output_effect",
            "dead_slot_ids": dead_slots,
            "inconsistent_slot_ids": inconsistent_slots,
            "slot_effects": slot_effects,
            "trials_completed": trials,
            "trials": trial_records,
        }
    if inconsistent_slots:
        return {
            "status": "unsupported",
            "passed": False,
            "reason": "expanded_region_effect_not_consistent",
            "inconsistent_slot_ids": inconsistent_slots,
            "slot_effects": slot_effects,
            "trials_completed": trials,
            "trials": trial_records,
        }
    return {
        "status": "passed",
        "passed": True,
        "reason": None,
        "slot_effects": slot_effects,
        "trials_completed": trials,
        "trials": trial_records,
        "gpu": {
            "device": device,
            "name": torch.cuda.get_device_name(resolved_device),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
    }


def _evaluate(payload: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device(str(payload["device"]))
    if device.type != "cuda":
        raise ValueError(f"device must be CUDA, got {device}")
    torch.cuda.set_device(device)
    with _cuda_memory_guard(
        torch=torch,
        device=device,
        max_device_memory_gib=MAX_DEVICE_MEMORY_GIB,
    ) as memory_guard:
        result = _evaluate_without_memory_guard(payload)
    result["memory_guard"] = memory_guard
    return result


def _payload_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: payload.get(key)
        for key in (
            "child_uuid",
            "parent_uuid",
            "variant",
            "slot_id",
            "slot_ids",
            "slots",
        )
    }


def _worker(payload: Mapping[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    identity = _payload_identity(payload)
    try:
        result = _evaluate(payload)
    except _CudaMemoryGuardFailure as exc:
        result = exc.result_record()
        result["reason"] = exc.failure_reason
        if isinstance(exc.cause, UnsupportedCase):
            result["status"] = "unsupported"
            result["reason"] = _exception_detail(exc.cause)
    except UnsupportedCase as exc:
        result = {
            "status": "unsupported",
            "passed": False,
            "reason": _exception_detail(exc),
        }
    except BaseException as exc:  # noqa: BLE001 - one row must fail closed
        result = {
            "status": "failed",
            "passed": False,
            "reason": _exception_detail(exc),
        }
    return {
        "contract_version": CONTRACT_VERSION,
        "perturbation_contract_version": PERTURBATION_CONTRACT_VERSION,
        **identity,
        **result,
        "duration_seconds": time.monotonic() - started,
    }


def _run_subprocess(payload: Mapping[str, Any], timeout_seconds: float) -> dict[str, Any]:
    global _ACTIVE_WORKER_PROCESS

    started = time.monotonic()
    serialized = json.dumps(payload, ensure_ascii=False)
    process: subprocess.Popen[str] | None = None
    try:
        prior_signal_mask = signal.pthread_sigmask(signal.SIG_BLOCK, _DRIVER_SIGNALS)
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
        finally:
            # Do not expose a child-spawn/register race to asynchronous stops.
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
            "contract_version": CONTRACT_VERSION,
            "perturbation_contract_version": PERTURBATION_CONTRACT_VERSION,
            **_payload_identity(payload),
            "status": "timeout",
            "passed": False,
            "reason": f"row_timeout_after_{timeout_seconds:g}_seconds",
            "duration_seconds": time.monotonic() - started,
            "worker_stdout_tail": stdout[-2_000:],
            "worker_stderr_tail": stderr[-4_000:],
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
    markers = [
        line[len(RESULT_MARKER) :]
        for line in stdout.splitlines()
        if line.startswith(RESULT_MARKER)
    ]
    if process.returncode != 0 or len(markers) != 1:
        return {
            "contract_version": CONTRACT_VERSION,
            "perturbation_contract_version": PERTURBATION_CONTRACT_VERSION,
            **_payload_identity(payload),
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


def _rows_by_uuid(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in pq.read_table(path).to_pylist():
        uuid = _nested(row, "extra_info.uuid")
        if not isinstance(uuid, str) or not uuid:
            raise ValueError(f"row in {path} has no UUID")
        if uuid in result:
            raise ValueError(f"duplicate UUID in {path}: {uuid}")
        result[uuid] = row
    return result


def _declared_logical_slot_range(
    manifest: Mapping[str, Any],
) -> tuple[int, int] | None:
    contract = manifest.get("group_contract")
    if contract is None:
        if manifest.get("contract_version") == "shape_variable_multislot_solver_v5":
            raise ValueError(
                "shape_variable_multislot_solver_v5 requires group_contract"
            )
        return None
    if not isinstance(contract, Mapping):
        raise ValueError("group_contract must be an object")
    raw = contract.get("logical_slot_count_range")
    if (
        not isinstance(raw, list)
        or len(raw) != 2
        or any(type(value) is not int for value in raw)
        or raw[0] <= 0
        or raw[0] > raw[1]
    ):
        raise ValueError(f"invalid logical_slot_count_range: {raw!r}")
    return int(raw[0]), int(raw[1])


def _accepted_tasks(
    selected_path: Path,
    children_path: Path,
    manifest_path: Path,
) -> list[dict[str, Any]]:
    parents = _rows_by_uuid(selected_path)
    children = _rows_by_uuid(children_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    solver_contract_version = manifest.get("contract_version")
    if not isinstance(solver_contract_version, str) or not solver_contract_version:
        raise ValueError("manifest contract_version must be a non-empty string")
    logical_slot_range = _declared_logical_slot_range(manifest)
    decisions = manifest.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("manifest decisions must be a list")
    tasks: list[dict[str, Any]] = []
    for decision in decisions:
        if not isinstance(decision, Mapping) or decision.get("accepted") is not True:
            continue
        child_uuid = decision.get("child_uuid")
        parent_uuid = decision.get("parent_uuid")
        if not isinstance(child_uuid, str) or child_uuid not in children:
            raise ValueError(f"accepted child missing from children parquet: {child_uuid}")
        if not isinstance(parent_uuid, str) or parent_uuid not in parents:
            raise ValueError(f"accepted parent missing from selected parquet: {parent_uuid}")
        attempts = decision.get("attempts")
        accepted_attempts = (
            [
                attempt
                for attempt in attempts
                if isinstance(attempt, Mapping) and attempt.get("accepted") is True
            ]
            if isinstance(attempts, list)
            else []
        )
        if len(accepted_attempts) != 1:
            raise ValueError(f"child {child_uuid} must have one accepted attempt")
        attempt = accepted_attempts[0]
        slots = _decision_solver_slots(
            decision,
            attempt,
            child_uuid=child_uuid,
            solver_contract_version=solver_contract_version,
        )
        if logical_slot_range is not None and not (
            logical_slot_range[0] <= len(slots) <= logical_slot_range[1]
        ):
            raise ValueError(
                f"child {child_uuid} logical slot count {len(slots)} is outside "
                f"declared range {logical_slot_range}"
            )
        child_code = _nested(children[child_uuid], "reward_model.ground_truth")
        parent_code = _nested(parents[parent_uuid], "reward_model.ground_truth")
        entry_point = _nested(children[child_uuid], "extra_info.entry_point", "Model")
        if not all(isinstance(value, str) for value in (child_code, parent_code, entry_point)):
            raise ValueError(f"child {child_uuid} has invalid reference fields")
        expected_hash = decision.get("child_reference_sha256")
        observed_hash = _sha256_bytes(child_code.encode("utf-8"))
        if expected_hash != observed_hash:
            raise ValueError(f"child reference hash mismatch: {child_uuid}")
        expected_parent_hash = decision.get("parent_reference_sha256")
        observed_parent_hash = _sha256_bytes(parent_code.encode("utf-8"))
        if expected_parent_hash != observed_parent_hash:
            raise ValueError(f"parent reference hash mismatch: {parent_uuid}")
        child_parent_uuid = _nested(children[child_uuid], "extra_info.v4.parent_uuid")
        if child_parent_uuid != parent_uuid:
            raise ValueError(f"child parent lineage mismatch: {child_uuid}")
        slot_ids = [str(slot["slot_id"]) for slot in slots]
        task = {
            "child_uuid": child_uuid,
            "parent_uuid": parent_uuid,
            "variant": decision.get("variant"),
            "slot_id": slot_ids[0] if len(slot_ids) == 1 else None,
            "slot_ids": slot_ids,
            "slots": slots,
            "solver_contract_version": solver_contract_version,
            "entry_point": entry_point,
            "parent_code": parent_code,
            "child_code": child_code,
        }
        if len(slots) == 1:
            # Keep the singular fields in the in-memory v3 task contract so
            # old diagnostics that inspect tasks remain usable.
            task.update(
                {
                    "old_value": slots[0]["old_value"],
                    "new_value": slots[0]["new_value"],
                    "occurrences": slots[0]["occurrences"],
                }
            )
        tasks.append(task)
    if len(tasks) != len(children):
        raise ValueError(
            f"accepted manifest/children count mismatch:{len(tasks)}:{len(children)}"
        )
    return tasks


def _requested_child_uuids(args: argparse.Namespace) -> tuple[bool, set[str]]:
    requested: list[str] = []
    for value in args.child_uuid:
        if not isinstance(value, str) or not value or any(char.isspace() for char in value):
            raise ValueError(f"invalid --child-uuid value: {value!r}")
        requested.append(value)
    if args.child_uuid_file is not None:
        if not args.child_uuid_file.is_file():
            raise FileNotFoundError(args.child_uuid_file)
        for line_number, line in enumerate(
            args.child_uuid_file.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            value = line.strip()
            if not value:
                continue
            if any(char.isspace() for char in value):
                raise ValueError(
                    f"{args.child_uuid_file}:{line_number}: expected one UUID"
                )
            requested.append(value)
    duplicates = [
        uuid
        for uuid, count in collections.Counter(requested).items()
        if count > 1
    ]
    if duplicates:
        raise ValueError(f"duplicate requested child UUIDs: {sorted(duplicates)}")
    return bool(args.child_uuid) or args.child_uuid_file is not None, set(requested)


def _validation_binding(
    *,
    evidence_hashes: Mapping[str, str],
    device: str,
    trials: int,
    seed: int,
    timeout_seconds: float,
) -> tuple[dict[str, Any], str]:
    binding = {
        "validation_binding_contract_version": RUN_BINDING_CONTRACT_VERSION,
        "contract_version": CONTRACT_VERSION,
        "perturbation_contract_version": PERTURBATION_CONTRACT_VERSION,
        **dict(evidence_hashes),
        "validation_config": {
            "device": device,
            "trials": trials,
            "seed": seed,
            "timeout_seconds": timeout_seconds,
            "max_device_memory_gib": MAX_DEVICE_MEMORY_GIB,
        },
    }
    return binding, _canonical_json_sha256(binding)


def _validate_resume_record(
    record: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    binding: Mapping[str, Any],
    binding_sha256: str,
    context: str,
) -> None:
    for field in (
        "validation_binding_contract_version",
        "contract_version",
        "perturbation_contract_version",
        "validator_source_sha256",
        "launcher_source_sha256",
        "selected_sha256",
        "children_sha256",
        "manifest_sha256",
    ):
        if record.get(field) != binding.get(field):
            raise ValueError(f"{context}: resume {field} mismatch")
    if record.get("validation_config") != binding.get("validation_config"):
        raise ValueError(f"{context}: resume validation_config mismatch")
    if record.get("validation_binding_sha256") != binding_sha256:
        raise ValueError(f"{context}: resume validation binding mismatch")
    for field in (
        "child_uuid",
        "parent_uuid",
        "variant",
        "slot_id",
        "slot_ids",
        "slots",
    ):
        if record.get(field) != task.get(field):
            raise ValueError(f"{context}: resume task identity mismatch for {field}")
    if not isinstance(record.get("status"), str):
        raise ValueError(f"{context}: resume record has no status")
    if type(record.get("passed")) is not bool:
        raise ValueError(f"{context}: resume record has no boolean passed field")
    if (record.get("status") == "passed") != record.get("passed"):
        raise ValueError(f"{context}: resume status/passed fields conflict")


def _load_resume_records(
    handle: Any,
    *,
    tasks_by_uuid: Mapping[str, Mapping[str, Any]],
    binding: Mapping[str, Any],
    binding_sha256: str,
    output_path: Path,
) -> dict[str, dict[str, Any]]:
    handle.seek(0)
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(handle, start=1):
        context = f"{output_path}:{line_number}"
        if not line.strip():
            raise ValueError(f"{context}: blank resume JSONL line")
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{context}: invalid resume JSON: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ValueError(f"{context}: resume record must be an object")
        child_uuid = loaded.get("child_uuid")
        if not isinstance(child_uuid, str) or not child_uuid:
            raise ValueError(f"{context}: resume record has no child UUID")
        if child_uuid in records:
            raise ValueError(f"{context}: duplicate resume child UUID: {child_uuid}")
        task = tasks_by_uuid.get(child_uuid)
        if task is None:
            raise ValueError(f"{context}: resume child is absent from current artifacts")
        _validate_resume_record(
            loaded,
            task=task,
            binding=binding,
            binding_sha256=binding_sha256,
            context=context,
        )
        records[child_uuid] = loaded
    handle.seek(0, os.SEEK_END)
    return records


def _append_jsonl(handle: Any, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("selected", type=Path)
    parser.add_argument("children", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument(
        "--launcher-sha256",
        required=True,
        help="SHA-256 of the exact orchestration launcher source",
    )
    parser.add_argument("--child-uuid", action="append", default=[])
    parser.add_argument(
        "--child-uuid-file",
        type=Path,
        help="UTF-8 allowlist with one child UUID per line",
    )
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    _install_driver_signal_handlers()
    args = _parser().parse_args(argv)
    if args.trials <= 0 or args.timeout_seconds <= 0:
        raise ValueError("trials and timeout must be positive")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must satisfy 0 <= index < shard-count")
    if (
        len(args.launcher_sha256) != 64
        or any(character not in "0123456789abcdef" for character in args.launcher_sha256)
    ):
        raise ValueError("launcher-sha256 must be 64 lowercase hexadecimal characters")
    all_tasks = _accepted_tasks(args.selected, args.children, args.manifest)
    tasks_by_uuid = {str(task["child_uuid"]): task for task in all_tasks}
    if len(tasks_by_uuid) != len(all_tasks):
        raise ValueError("accepted tasks have duplicate child UUIDs")
    filter_requested, wanted = _requested_child_uuids(args)
    tasks = all_tasks
    if filter_requested:
        tasks = [task for task in tasks if task["child_uuid"] in wanted]
        missing = wanted - set(tasks_by_uuid)
        if missing:
            raise ValueError(f"requested child UUIDs not found: {sorted(missing)}")
    tasks = [
        task
        for index, task in enumerate(tasks)
        if index % args.shard_count == args.shard_index
    ]
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("limit must be positive")
        tasks = tasks[: args.limit]
    shard_tasks_by_uuid = {str(task["child_uuid"]): task for task in tasks}
    counts: collections.Counter[str] = collections.Counter()
    evidence_hashes = {
        "validator_source_sha256": _sha256_file(Path(__file__).resolve()),
        "launcher_source_sha256": args.launcher_sha256,
        "selected_sha256": _sha256_file(args.selected),
        "children_sha256": _sha256_file(args.children),
        "manifest_sha256": _sha256_file(args.manifest),
    }
    binding, binding_sha256 = _validation_binding(
        evidence_hashes=evidence_hashes,
        device=args.device,
        trials=args.trials,
        seed=args.seed,
        timeout_seconds=args.timeout_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    executed = 0
    resumed = 0
    with args.output.open("a+", encoding="utf-8") as output_handle:
        try:
            fcntl.flock(output_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another process owns output: {args.output}") from exc
        prior = _load_resume_records(
            output_handle,
            tasks_by_uuid=shard_tasks_by_uuid,
            binding=binding,
            binding_sha256=binding_sha256,
            output_path=args.output,
        )
        for task in tasks:
            child_uuid = str(task["child_uuid"])
            if child_uuid in prior:
                resumed += 1
                counts[str(prior[child_uuid]["status"])] += 1
                continue
            payload = {
                **task,
                "device": args.device,
                "trials": args.trials,
                "seed": args.seed,
            }
            result = _run_subprocess(payload, args.timeout_seconds)
            result.update(evidence_hashes)
            result["validation_binding_contract_version"] = (
                RUN_BINDING_CONTRACT_VERSION
            )
            result["validation_config"] = binding["validation_config"]
            result["validation_binding_sha256"] = binding_sha256
            _append_jsonl(output_handle, result)
            executed += 1
            counts[str(result["status"])] += 1
            print(
                json.dumps(
                    {
                        "child_uuid": result.get("child_uuid"),
                        "status": result.get("status"),
                        "reason": result.get("reason"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    passed = int(counts.get("passed", 0))
    print(
        json.dumps(
            {
                "contract_version": CONTRACT_VERSION,
                "perturbation_contract_version": PERTURBATION_CONTRACT_VERSION,
                "rows": len(tasks),
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
                "validator_source_sha256": evidence_hashes[
                    "validator_source_sha256"
                ],
                "validation_binding_sha256": binding_sha256,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    if sys.argv[1:] == ["--_worker"]:
        worker_payload = json.loads(sys.stdin.read())
        print(RESULT_MARKER + json.dumps(_worker(worker_payload), ensure_ascii=False))
    else:
        main()
