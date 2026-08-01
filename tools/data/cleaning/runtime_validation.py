#!/usr/bin/env python3
"""Validate a KernelBench-style PyTorch reference implementation.

The compatibility entry point is :func:`validate_ops_text`, which returns the
same compact verdict strings as the original standalone script supplied for the
cleanup.  :func:`validate_ops_text_detailed` additionally reports the stage and
exception needed by the dataset audit.

This module executes input text. Dataset callers must run it in a disposable
subprocess with a timeout; :mod:`tools.data.cleaning.pipeline` does that by
default.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import random
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_VERDICT_MARKER = "__OPS_VERDICT__:"
_VALID_VERDICTS = frozenset(
    {
        "passed",
        "fixed_input_values",
        "natural_output_low_activity",
        "forward_argument_sensitivity_inconclusive",
        "synthetic_sensitivity_only",
        "non_finite_input",
        "non_finite_output",
        "no_same_output",
        "unstable_input_structure",
        "different_input_not_changed",
        "Failed",
    }
)


def _import_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on the runtime
        raise RuntimeError("validate_ops_text requires torch") from exc
    return torch


def extract_python_code(text: str, *, entry_point: str = "Model") -> str:
    """Extract the likely reference snippet from raw or fenced text."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("ops text must be a non-empty string")
    class_marker = f"class {entry_point}"
    fenced = re.findall(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    for block in fenced:
        if class_marker in block and "def get_inputs" in block:
            return block.strip()
    stripped = text.strip()
    if class_marker in stripped and "def get_inputs" in stripped:
        return stripped
    raise ValueError(f"ops text must contain `{class_marker}` and `def get_inputs`")


class _CpuDeviceNormalizer(ast.NodeTransformer):
    """Map explicit CUDA placement to CPU for CPU semantic validation only."""

    def visit_Constant(self, node: ast.Constant) -> ast.AST:  # noqa: N802 - ast API
        if isinstance(node.value, str) and (node.value == "cuda" or node.value.startswith("cuda:")):
            return ast.copy_location(ast.Constant(value="cpu"), node)
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802 - ast API
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "cuda":
            node.func.attr = "to"
            node.args = [ast.Constant(value="cpu")]
            node.keywords = []
        return node


def _exec_ops_code(code: str, *, entry_point: str, device: str) -> dict[str, Any]:
    torch = _import_torch()
    namespace: dict[str, Any] = {"torch": torch}
    parsed = ast.parse(code, filename="<ops_text>", mode="exec")
    if device == "cpu":
        parsed = _CpuDeviceNormalizer().visit(parsed)
        ast.fix_missing_locations(parsed)
    exec(compile(parsed, "<ops_text>", "exec"), namespace)  # noqa: S102 - isolated by the dataset runner
    if entry_point not in namespace:
        raise ValueError(f"parsed code does not define `{entry_point}`")
    if "get_inputs" not in namespace:
        raise ValueError("parsed code does not define `get_inputs`")
    if "get_init_inputs" not in namespace:
        namespace["get_init_inputs"] = lambda: []
    namespace["__entry_point__"] = namespace[entry_point]
    return namespace


def _to_device(value: Any, device: str) -> Any:
    torch = _import_torch()
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


def _clone_value(value: Any) -> Any:
    torch = _import_torch()
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {key: _clone_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    return copy.deepcopy(value)


def _clone_inputs(inputs: Sequence[Any]) -> list[Any]:
    return [_clone_value(item) for item in inputs]


def _forward_argument_locations(inputs: Sequence[Any]) -> list[tuple[str, int | str]]:
    """Enumerate logical positional/keyword arguments in normalized inputs."""

    if len(inputs) == 2 and isinstance(inputs[0], (list, tuple)) and isinstance(inputs[1], Mapping):
        return [
            *(("positional", index) for index in range(len(inputs[0]))),
            *(("keyword", key) for key in inputs[1]),
        ]
    return [("positional", index) for index in range(len(inputs))]


def _replace_forward_argument(
    inputs: Sequence[Any],
    source: Sequence[Any],
    location: tuple[str, int | str],
    transform: Any | None = None,
) -> list[Any]:
    """Clone normalized inputs and replace one logical forward argument."""

    probe = _clone_inputs(inputs)
    kind, key = location
    if len(inputs) == 2 and isinstance(inputs[0], (list, tuple)) and isinstance(inputs[1], Mapping):
        if kind == "positional":
            assert isinstance(key, int)
            value = source[0][key]
            positional = list(probe[0])
            positional[key] = transform(value) if transform is not None else _clone_value(value)
            probe[0] = positional
        else:
            value = source[1][key]
            probe[1][key] = transform(value) if transform is not None else _clone_value(value)
    else:
        assert isinstance(key, int)
        value = source[key]
        probe[key] = transform(value) if transform is not None else _clone_value(value)
    return probe


def _normalize_forward_inputs(value: Any, device: str) -> list[Any]:
    """Normalize positional or keyword-style KernelBench inputs.

    A mapping means keyword arguments.  ``[positional_args, keyword_args]`` is
    also accepted, mirroring the constructor convention already used by the
    corpus.  Ordinary lists/tuples retain their positional meaning.
    """

    if isinstance(value, Mapping):
        return [[], _to_device(dict(value), device)]
    if not isinstance(value, (list, tuple)):
        raise TypeError("get_inputs() must return list/tuple/mapping")
    return _to_device(list(value), device)


def _invoke_model(model: Any, inputs: Sequence[Any]) -> Any:
    if len(inputs) == 2 and isinstance(inputs[0], (list, tuple)) and isinstance(inputs[1], Mapping):
        return model(*_clone_inputs(inputs[0]), **_clone_value(inputs[1]))
    return model(*_clone_inputs(inputs))


def _snapshot_output(value: Any) -> Any:
    """Detach outputs from model state before a later forward can mutate them."""

    torch = _import_torch()
    uninitialized_types = tuple(
        candidate
        for candidate in (
            getattr(torch.nn.parameter, "UninitializedParameter", None),
            getattr(torch.nn.parameter, "UninitializedBuffer", None),
        )
        if candidate is not None
    )
    if uninitialized_types and isinstance(value, uninitialized_types):
        # Lazy modules legitimately expose placeholder state before their first
        # forward.  Preserve that state as a comparable marker; calling
        # detach/clone on the placeholder raises ValueError and used to turn a
        # valid mode audit into an inconclusive infrastructure failure.
        return ("__uninitialized_state__", type(value).__qualname__)
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    # Parameter containers are nn.Module subclasses rather than registered
    # Mapping/Sequence instances.  Deep-copying the container and later using
    # object equality makes an unchanged output look unstable.  Snapshot their
    # public ordered contents just like ordinary nested tensor outputs.
    if isinstance(value, torch.nn.ParameterDict):
        return {key: _snapshot_output(item) for key, item in value.items()}
    if isinstance(value, torch.nn.ParameterList):
        return [_snapshot_output(item) for item in value]
    if isinstance(value, Mapping):
        return {key: _snapshot_output(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot_output(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_snapshot_output(item) for item in value)
    return copy.deepcopy(value)


def _perturb_value(value: Any) -> Any:
    torch = _import_torch()
    if isinstance(value, torch.Tensor):
        out = value.detach().clone()
        if out.numel() == 0:
            return out
        if out.dtype.is_floating_point or out.dtype.is_complex:
            return out + torch.randn_like(out) * 0.1 + 0.123
        if out.dtype == torch.bool:
            return ~out
        lo = int(out.min())
        hi = int(out.max())
        if hi > lo:
            return lo + (out - lo + 1) % (hi - lo + 1)
        # There is no nontrivial in-range perturbation for a constant integer
        # tensor.  This nudge may violate an index domain; the caller retries a
        # range-preserving perturbation if the model rejects it.
        return out - 1 if lo > 0 else out + 1
    if isinstance(value, list):
        return [_perturb_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_perturb_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _perturb_value(item) for key, item in value.items()}
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float, complex)):
        return value + 1
    return copy.deepcopy(value)


def _perturb_inputs(inputs: Sequence[Any]) -> list[Any]:
    return [_perturb_value(item) for item in inputs]


def _strongly_perturb_value(value: Any) -> Any:
    """Make a larger, domain-conscious value change for sensitivity checks."""

    torch = _import_torch()
    if isinstance(value, torch.Tensor):
        out = value.detach().clone()
        if out.numel() == 0:
            return out
        if out.dtype == torch.bool:
            return ~out
        if out.dtype.is_floating_point or out.dtype.is_complex:
            return -1.7 * out + 0.731
        lo = int(out.min())
        hi = int(out.max())
        if hi > lo:
            return lo + (out - lo + max(1, (hi - lo + 1) // 2)) % (hi - lo + 1)
        return out - 1 if lo > 0 else out + 1
    if isinstance(value, list):
        return [_strongly_perturb_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_strongly_perturb_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _strongly_perturb_value(item) for key, item in value.items()}
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float, complex)):
        return -1.7 * value + 1
    return copy.deepcopy(value)


def _strongly_perturb_inputs(inputs: Sequence[Any]) -> list[Any]:
    return [_strongly_perturb_value(item) for item in inputs]


def _boundary_perturb_value(value: Any, *, nonfinite: bool) -> Any:
    """Probe exact zero predicates and finite/non-finite classification ops."""

    torch = _import_torch()
    if isinstance(value, torch.Tensor):
        out = value.detach().clone()
        if out.numel() == 0:
            return out
        if not nonfinite or not (out.dtype.is_floating_point or out.dtype.is_complex):
            return torch.zeros_like(out)
        flat = out.reshape(-1)
        if out.dtype.is_complex:
            flat.fill_(complex(float("-inf"), 0.0))
            flat[0] = complex(float("nan"), 0.0)
            if flat.numel() > 1:
                flat[1] = complex(float("inf"), 0.0)
        else:
            flat.fill_(float("-inf"))
            flat[0] = float("nan")
            if flat.numel() > 1:
                flat[1] = float("inf")
        return out
    if isinstance(value, list):
        return [_boundary_perturb_value(item, nonfinite=nonfinite) for item in value]
    if isinstance(value, tuple):
        return tuple(_boundary_perturb_value(item, nonfinite=nonfinite) for item in value)
    if isinstance(value, dict):
        return {key: _boundary_perturb_value(item, nonfinite=nonfinite) for key, item in value.items()}
    if isinstance(value, bool):
        return False
    if isinstance(value, (float, complex)) and nonfinite:
        return float("nan")
    if isinstance(value, (int, float, complex)):
        return type(value)(0)
    return copy.deepcopy(value)


def _boundary_perturb_inputs(inputs: Sequence[Any], *, nonfinite: bool) -> list[Any]:
    return [_boundary_perturb_value(item, nonfinite=nonfinite) for item in inputs]


def _scale_floating_value(value: Any, factor: float) -> Any:
    """Scale floating tensors while preserving discrete/domain arguments.

    This mirrors KernelBench-Verified's hidden-distribution construction:
    floating tensor values change, while integer and boolean tensors (often
    indices, dimensions, or masks) retain their valid domains.  Python scalar
    configuration arguments are also left unchanged.
    """

    torch = _import_torch()
    if isinstance(value, torch.Tensor):
        out = value.detach().clone()
        if out.dtype.is_floating_point or out.dtype.is_complex:
            return out * factor
        return out
    if isinstance(value, list):
        return [_scale_floating_value(item, factor) for item in value]
    if isinstance(value, tuple):
        return tuple(_scale_floating_value(item, factor) for item in value)
    if isinstance(value, dict):
        return {key: _scale_floating_value(item, factor) for key, item in value.items()}
    return copy.deepcopy(value)


def _scale_floating_inputs(inputs: Sequence[Any], factor: float) -> list[Any]:
    return [_scale_floating_value(item, factor) for item in inputs]


def _perturb_value_in_range(value: Any) -> Any:
    """Perturb without leaving a tensor's observed value range."""

    torch = _import_torch()
    if isinstance(value, torch.Tensor):
        out = value.detach().clone()
        if out.numel() == 0:
            return out
        if out.dtype == torch.bool:
            return ~out
        if out.dtype.is_floating_point:
            lo = out.min()
            hi = out.max()
            return (lo + hi) - out if bool((hi > lo).item()) else out
        if out.dtype.is_complex:
            return out
        lo = int(out.min())
        hi = int(out.max())
        return lo + (out - lo + 1) % (hi - lo + 1) if hi > lo else out
    if isinstance(value, list):
        return [_perturb_value_in_range(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_perturb_value_in_range(item) for item in value)
    if isinstance(value, dict):
        return {key: _perturb_value_in_range(item) for key, item in value.items()}
    return copy.deepcopy(value)


def _outputs_allclose(a: Any, b: Any, *, rtol: float, atol: float) -> bool:
    torch = _import_torch()
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if a.dtype == torch.bool or not (a.dtype.is_floating_point or a.dtype.is_complex):
            return bool(torch.equal(a, b))
        return bool(torch.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if a.keys() != b.keys():
            return False
        return all(_outputs_allclose(a[key], b[key], rtol=rtol, atol=atol) for key in a)
    if isinstance(a, (tuple, list)) and isinstance(b, type(a)):
        if len(a) != len(b):
            return False
        return all(_outputs_allclose(x, y, rtol=rtol, atol=atol) for x, y in zip(a, b, strict=True))
    if isinstance(a, (float, complex)) and isinstance(b, type(a)):
        return bool(
            torch.allclose(
                torch.as_tensor(a),
                torch.as_tensor(b),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )
        )
    try:
        return bool(a == b)
    except Exception:
        return False


def _output_change_activity(a: Any, b: Any, *, rtol: float, atol: float) -> tuple[int, int, float]:
    """Return changed elements, total elements, and the largest leaf fraction.

    A single changed bit in a million-element mask is enough to make two
    outputs unequal, but it is weak evidence that the natural input
    distribution exercises a useful computation.  The largest per-leaf
    fraction prevents a large unchanged auxiliary output from hiding a
    genuinely dynamic scalar or small tensor in a tuple.
    """

    torch = _import_torch()
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        total = max(a.numel(), b.numel())
        if a.shape != b.shape or a.dtype != b.dtype:
            return total or 1, total or 1, 1.0
        if total == 0:
            return 0, 0, 0.0
        if a.dtype == torch.bool or not (a.dtype.is_floating_point or a.dtype.is_complex):
            changed = int(torch.count_nonzero(a != b).item())
        else:
            changed = int(torch.count_nonzero(~torch.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)).item())
        return changed, total, changed / total
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if a.keys() != b.keys():
            return 1, 1, 1.0
        parts = [_output_change_activity(a[key], b[key], rtol=rtol, atol=atol) for key in a]
    elif isinstance(a, (tuple, list)) and isinstance(b, type(a)):
        if len(a) != len(b):
            return 1, 1, 1.0
        parts = [_output_change_activity(x, y, rtol=rtol, atol=atol) for x, y in zip(a, b, strict=True)]
    else:
        changed = 0 if _outputs_allclose(a, b, rtol=rtol, atol=atol) else 1
        return changed, 1, float(changed)
    if not parts:
        return 0, 0, 0.0
    return sum(part[0] for part in parts), sum(part[1] for part in parts), max(part[2] for part in parts)


def _snapshot_state(model: Any) -> dict[str, Any]:
    """Clone parameters and buffers so mode-dependent mutation is observable."""

    if not hasattr(model, "state_dict"):
        return {}
    return {name: _snapshot_output(value) for name, value in model.state_dict().items()}


def _seed_torch(seed: int, device: str) -> None:
    """Seed Torch without touching the CUDA driver during CPU validation."""

    torch = _import_torch()
    torch.manual_seed(seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _snapshot_rng_states(device: str) -> dict[str, Any]:
    """Capture RNGs a diagnostic forward could consume or reseed."""

    torch = _import_torch()
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch_cpu": torch.random.get_rng_state().clone(),
    }
    numpy = sys.modules.get("numpy")
    if numpy is not None and hasattr(numpy, "random") and hasattr(numpy.random, "get_state"):
        state["numpy"] = numpy.random.get_state()
    if device.startswith("cuda") and torch.cuda.is_available():
        state["torch_cuda"] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng_states(state: Mapping[str, Any]) -> None:
    """Restore states captured by :func:`_snapshot_rng_states`."""

    torch = _import_torch()
    random.setstate(state["python"])
    torch.random.set_rng_state(state["torch_cpu"])
    numpy = sys.modules.get("numpy")
    if "numpy" in state and numpy is not None:
        numpy.random.set_state(state["numpy"])
    if "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _audit_train_eval_modes(
    model: Any,
    inputs: Sequence[Any],
    *,
    device: str,
    seed: int,
    rtol: float,
    atol: float,
    model_factory: Any | None = None,
) -> dict[str, Any]:
    """Compare train/eval output, mutable state, failures, and train RNG.

    Each arm starts from a deep copy of the same initialized model.  Only one
    copy is live at a time, which bounds per-worker memory while preserving an
    identical parameter/buffer baseline.
    """

    torch = _import_torch()

    uninitialized_types = tuple(
        candidate
        for candidate in (
            getattr(torch.nn.parameter, "UninitializedParameter", None),
            getattr(torch.nn.parameter, "UninitializedBuffer", None),
        )
        if candidate is not None
    )
    if uninitialized_types and any(isinstance(value, uninitialized_types) for value in model.state_dict().values()):
        # A LazyModule must materialize once before the two arms are copied.
        # Otherwise each arm initializes different random parameters and the
        # train-repeat probe falsely reports forward stochasticity.
        original_training = getattr(model, "training", None)
        if hasattr(model, "eval"):
            model.eval()
        _seed_torch(seed + 7919, device)
        with torch.no_grad():
            _invoke_model(model, inputs)
        if original_training is not None and hasattr(model, "train"):
            model.train(original_training)

    def run(mode: str, forward_seed: int) -> tuple[Any | None, dict[str, Any] | None, str | None]:
        trial = None
        try:
            try:
                trial = copy.deepcopy(model)
            except RuntimeError:
                if model_factory is None:
                    raise
                trial = model_factory()
                if hasattr(trial, "load_state_dict"):
                    trial.load_state_dict(baseline_state)
            getattr(trial, mode)()
            _seed_torch(forward_seed, device)
            with torch.no_grad():
                output = _snapshot_output(_invoke_model(trial, inputs))
            return output, _snapshot_state(trial), None
        except BaseException as exc:  # noqa: BLE001 - this is diagnostic evidence
            return None, None, _exception_detail(exc)
        finally:
            del trial

    baseline_state = _snapshot_state(model)
    eval_output, eval_state, eval_error = run("eval", seed + 1009)
    train_output, train_state, train_error = run("train", seed + 1009)
    train_repeat, _, train_repeat_error = run("train", seed + 2027)

    flags: list[str] = []
    details: list[str] = []
    if eval_error and not train_error:
        flags.append("eval_only_failure")
        details.append(f"eval={eval_error}")
    elif train_error and not eval_error:
        flags.append("train_only_failure")
        details.append(f"train={train_error}")
    elif eval_error and train_error:
        flags.append("train_eval_inconclusive")
        details.append(f"eval={eval_error}; train={train_error}")
    else:
        assert eval_output is not None and train_output is not None
        if not _outputs_allclose(eval_output, train_output, rtol=rtol, atol=atol):
            flags.append("train_eval_output_diff")
        if eval_state is not None and train_state is not None:
            eval_changed = not _outputs_allclose(baseline_state, eval_state, rtol=0.0, atol=0.0)
            train_changed = not _outputs_allclose(baseline_state, train_state, rtol=0.0, atol=0.0)
            if eval_changed != train_changed or not _outputs_allclose(eval_state, train_state, rtol=0.0, atol=0.0):
                flags.append("train_eval_state_diff")

    if train_error is None:
        if train_repeat_error is not None:
            flags.append("train_eval_inconclusive")
            details.append(f"train_repeat={train_repeat_error}")
        elif (
            train_output is not None
            and train_repeat is not None
            and not _outputs_allclose(train_output, train_repeat, rtol=rtol, atol=atol)
        ):
            flags.append("train_only_stochastic")
    if not flags:
        flags.append("train_eval_same")
    return {"mode_flags": sorted(set(flags)), "mode_detail": "; ".join(details)}


def audit_train_eval_modes_detailed(
    ops_text: str,
    *,
    entry_point: str = "Model",
    device: str = "cpu",
    seed: int = 0,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> dict[str, Any]:
    """Run only the non-filtering train/eval diagnostic for one reference.

    Dataset orchestration executes this in a process separate from acceptance
    validation.  A slow or failed diagnostic can therefore be marked
    inconclusive without changing the row's acceptance verdict.
    """

    code = extract_python_code(ops_text, entry_point=entry_point)
    namespace = _exec_ops_code(code, entry_point=entry_point, device=device)
    _seed_torch(seed, device)
    init_inputs = namespace["get_init_inputs"]()
    inputs = namespace["get_inputs"]()
    init_inputs = [] if init_inputs is None else init_inputs
    inputs = [] if inputs is None else inputs
    if not isinstance(init_inputs, (list, tuple)):
        raise TypeError("get_init_inputs() must return list/tuple")
    init_inputs = _to_device(list(init_inputs), device)
    inputs = _normalize_forward_inputs(inputs, device)

    def build_model() -> Any:
        built = _build_model(namespace["__entry_point__"], init_inputs)
        return built.to(device) if hasattr(built, "to") else built

    model = build_model()
    rng_states = _snapshot_rng_states(device)
    try:
        result = _audit_train_eval_modes(
            model,
            inputs,
            device=device,
            seed=seed,
            rtol=rtol,
            atol=atol,
            model_factory=build_model,
        )
    finally:
        _restore_rng_states(rng_states)
    return {"verdict": "mode_audit_complete", "detail": "", **result}


def _all_finite(value: Any) -> bool:
    torch = _import_torch()
    if isinstance(value, torch.Tensor):
        if value.dtype.is_floating_point or value.dtype.is_complex:
            return bool(torch.isfinite(value).all())
        return True
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    if isinstance(value, (float, complex)):
        return bool(torch.isfinite(torch.as_tensor(value)))
    return True


def _input_structure_signature(value: Any) -> Any:
    """Describe shapes/types without requiring randomized values to match."""

    torch = _import_torch()
    if isinstance(value, torch.Tensor):
        return ("tensor", tuple(value.shape), str(value.dtype), str(value.layout), str(value.device))
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple((repr(key), _input_structure_signature(item)) for key, item in value.items()),
        )
    if isinstance(value, tuple):
        return ("tuple", tuple(_input_structure_signature(item) for item in value))
    if isinstance(value, list):
        return ("list", tuple(_input_structure_signature(item) for item in value))
    return (type(value).__qualname__,)


def _build_model(model_cls: Any, init_inputs: Any) -> Any:
    """Instantiate either positional or ``[[], kwargs]`` references."""

    torch = _import_torch()
    init = list(init_inputs)
    if (
        len(init) > 1
        and hasattr(init[0], "__len__")
        and not isinstance(init[0], (str, bytes, torch.Tensor))
        and len(init[0]) == 0
    ):
        init = init[1]
    if isinstance(init, Mapping):
        return model_cls(**init)
    return model_cls(*init)


def _validate_on_device(
    ops_text: str,
    *,
    entry_point: str,
    device: str,
    seed: int,
    rtol: float,
    atol: float,
    same_input_repeats: int,
    fresh_input_trials: int,
    fixed_input_repeats: int,
    min_natural_output_change_fraction: float,
    require_each_forward_input_sensitive: bool,
    audit_train_eval: bool,
) -> dict[str, Any]:
    torch = _import_torch()
    code = extract_python_code(ops_text, entry_point=entry_point)
    namespace = _exec_ops_code(code, entry_point=entry_point, device=device)
    _seed_torch(seed, device)

    init_inputs = namespace["get_init_inputs"]()
    inputs = namespace["get_inputs"]()
    init_inputs = [] if init_inputs is None else init_inputs
    inputs = [] if inputs is None else inputs
    if not isinstance(init_inputs, (list, tuple)):
        raise TypeError("get_init_inputs() must return list/tuple")
    init_inputs = _to_device(list(init_inputs), device)
    inputs = _normalize_forward_inputs(inputs, device)
    model = _build_model(namespace["__entry_point__"], init_inputs)
    model = model.to(device) if hasattr(model, "to") else model
    if audit_train_eval:
        rng_states = _snapshot_rng_states(device)
        try:
            mode_result = _audit_train_eval_modes(
                model,
                inputs,
                device=device,
                seed=seed,
                rtol=rtol,
                atol=atol,
            )
        finally:
            # The mode comparison is diagnostic only. Its seeded forwards must
            # not alter the natural get_inputs sequence used for acceptance.
            _restore_rng_states(rng_states)
    else:
        mode_result = {"mode_flags": [], "mode_detail": ""}
    if hasattr(model, "eval"):
        model.eval()

    def result(verdict: str, detail: str) -> dict[str, Any]:
        return {"verdict": verdict, "detail": detail, **mode_result}

    with torch.no_grad():
        # Three repeats catch stochastic operations whose first two draws happen
        # to agree.  Snapshot immediately because some references return a
        # mutable ParameterDict or other state that a later call changes.
        same_a = _snapshot_output(_invoke_model(model, inputs))
        if not _all_finite(same_a):
            return result("non_finite_output", "baseline output is non-finite")
        for _ in range(same_input_repeats - 1):
            repeated = _snapshot_output(_invoke_model(model, inputs))
            if not _all_finite(repeated):
                return result("non_finite_output", "repeated output is non-finite")
            if not _outputs_allclose(same_a, repeated, rtol=rtol, atol=atol):
                return result("no_same_output", f"same input changed within {same_input_repeats} forwards")
            del repeated

        def probe_changes_output(probe: list[Any]) -> bool | None:
            """Return None when a probe leaves the valid input domain."""

            try:
                diff_out = _snapshot_output(_invoke_model(model, probe))
            except Exception:  # noqa: BLE001 - domain-changing probes may be invalid
                return None
            if not _all_finite(diff_out):
                return None
            return not _outputs_allclose(same_a, diff_out, rtol=rtol, atol=atol)

        def argument_sensitivity_issue(natural_fresh: Sequence[Any]) -> str | None:
            if not require_each_forward_input_sensitive:
                return None
            transforms = (
                ("natural", None),
                ("strong_affine", _strongly_perturb_value),
                ("range_reflection", _perturb_value_in_range),
                ("all_zero", lambda value: _boundary_perturb_value(value, nonfinite=False)),
            )
            for location in _forward_argument_locations(inputs):
                valid_probes = 0
                changed_input_probes = 0
                for probe_name, transform in transforms:
                    source = natural_fresh if probe_name == "natural" else inputs
                    probe = _replace_forward_argument(inputs, source, location, transform)
                    if _outputs_allclose(inputs, probe, rtol=0.0, atol=0.0):
                        del probe
                        continue
                    changed_input_probes += 1
                    changed = probe_changes_output(probe)
                    del probe
                    if changed is None:
                        continue
                    valid_probes += 1
                    if changed:
                        break
                else:
                    label = f"{location[0]}:{location[1]}"
                    return (
                        f"forward argument {label} did not affect output across "
                        f"{valid_probes}/{changed_input_probes} valid/changed probes"
                    )
            return None

        # Multiple independently generated input sets are the safest probes for
        # indices, masks, and other constrained domains.  Five trials match the
        # v3 cleaning contract and reduce the two-sample false-pass/false-fail
        # risk in the earlier validator.  Only structure, not values, must match.
        # KernelGym's reference timing cache assumes that structure is stable;
        # randomized values are expected and valid.
        inputs_remained_fixed = True
        natural_output_changed_trials = 0
        maximum_natural_leaf_change_fraction = 0.0
        maximum_natural_changed_elements = 0
        for fresh_index in range(fresh_input_trials):
            fresh = namespace["get_inputs"]()
            if not isinstance(fresh, (list, tuple, Mapping)):
                return result(
                    "unstable_input_structure",
                    f"fresh get_inputs call {fresh_index + 1} returned an invalid container",
                )
            fresh = _normalize_forward_inputs(fresh, device)
            if _input_structure_signature(inputs) != _input_structure_signature(fresh):
                return result(
                    "unstable_input_structure",
                    f"fresh get_inputs call {fresh_index + 1} changed structure",
                )
            if not _outputs_allclose(inputs, fresh, rtol=0.0, atol=0.0):
                inputs_remained_fixed = False
            diff_out = _snapshot_output(_invoke_model(model, fresh))
            if not _all_finite(diff_out):
                return result(
                    "non_finite_output",
                    f"fresh get_inputs call {fresh_index + 1} produced non-finite output",
                )
            if not _outputs_allclose(same_a, diff_out, rtol=rtol, atol=atol):
                natural_output_changed_trials += 1
                changed, _, leaf_fraction = _output_change_activity(
                    same_a,
                    diff_out,
                    rtol=rtol,
                    atol=atol,
                )
                maximum_natural_changed_elements = max(maximum_natural_changed_elements, changed)
                maximum_natural_leaf_change_fraction = max(
                    maximum_natural_leaf_change_fraction,
                    leaf_fraction,
                )
                if leaf_fraction >= min_natural_output_change_fraction:
                    argument_issue = argument_sensitivity_issue(fresh)
                    if argument_issue is not None:
                        return result("forward_argument_sensitivity_inconclusive", argument_issue)
                    return result(
                        "passed",
                        "natural get_inputs output changed on fresh trial "
                        f"{fresh_index + 1}; max_leaf_change_fraction={leaf_fraction:.9g}; "
                        f"changed_elements={changed}",
                    )

        # Six equal draws can occur by chance for a tiny Boolean or categorical
        # input.  Confirm value equality without more model forwards until 20
        # total get_inputs() calls have been observed.  If a later draw changes,
        # execute that one natural input before falling through to diagnostics.
        if inputs_remained_fixed:
            observed_calls = fresh_input_trials + 1
            while observed_calls < fixed_input_repeats:
                confirmation = namespace["get_inputs"]()
                observed_calls += 1
                if not isinstance(confirmation, (list, tuple, Mapping)):
                    return result(
                        "unstable_input_structure",
                        f"fixed-input confirmation call {observed_calls} returned an invalid container",
                    )
                confirmation = _normalize_forward_inputs(confirmation, device)
                if _input_structure_signature(inputs) != _input_structure_signature(confirmation):
                    return result(
                        "unstable_input_structure",
                        f"fixed-input confirmation call {observed_calls} changed structure",
                    )
                if _outputs_allclose(inputs, confirmation, rtol=0.0, atol=0.0):
                    continue
                inputs_remained_fixed = False
                confirmation_output = _snapshot_output(_invoke_model(model, confirmation))
                if not _all_finite(confirmation_output):
                    return result(
                        "non_finite_output",
                        f"fixed-input confirmation call {observed_calls} produced non-finite output",
                    )
                if not _outputs_allclose(same_a, confirmation_output, rtol=rtol, atol=atol):
                    changed, _, leaf_fraction = _output_change_activity(
                        same_a,
                        confirmation_output,
                        rtol=rtol,
                        atol=atol,
                    )
                    if leaf_fraction >= min_natural_output_change_fraction:
                        argument_issue = argument_sensitivity_issue(confirmation)
                        if argument_issue is not None:
                            return result("forward_argument_sensitivity_inconclusive", argument_issue)
                        return result(
                            "passed",
                            "natural get_inputs output changed on fixed-input confirmation call "
                            f"{observed_calls}; max_leaf_change_fraction={leaf_fraction:.9g}; "
                            f"changed_elements={changed}",
                        )
                    natural_output_changed_trials += 1
                    maximum_natural_changed_elements = max(maximum_natural_changed_elements, changed)
                    maximum_natural_leaf_change_fraction = max(
                        maximum_natural_leaf_change_fraction,
                        leaf_fraction,
                    )
                break
            if inputs_remained_fixed:
                return result(
                    "fixed_input_values",
                    f"get_inputs returned exactly equal values on {observed_calls} consecutive calls",
                )

        if natural_output_changed_trials:
            return result(
                "natural_output_low_activity",
                f"{natural_output_changed_trials}/{fresh_input_trials} natural fresh outputs changed, "
                f"but max_leaf_change_fraction={maximum_natural_leaf_change_fraction:.9g} "
                f"was below {min_natural_output_change_fraction:.9g}; "
                f"max_changed_elements={maximum_natural_changed_elements}",
            )

        # Deterministic probes cover locally constant discrete outputs and
        # cancellation when all arguments change together. Exactly fixed
        # get_inputs() values have already been rejected above.
        def synthetic_changes_output(probe: list[Any]) -> bool:
            return probe_changes_output(probe) is True

        # Generate and release one probe at a time.  Some references use very
        # large tensors, so retaining every variant concurrently multiplies
        # per-worker memory by the number of forward arguments.
        probes = (
            ("scale_x3", lambda: _scale_floating_inputs(inputs, 3.0)),
            ("scale_x0.01", lambda: _scale_floating_inputs(inputs, 0.01)),
            ("scale_x-1", lambda: _scale_floating_inputs(inputs, -1.0)),
            ("strong_affine", lambda: _strongly_perturb_inputs(inputs)),
            ("range_reflection", lambda: [_perturb_value_in_range(item) for item in inputs]),
            ("all_zero", lambda: _boundary_perturb_inputs(inputs, nonfinite=False)),
            ("nan_inf", lambda: _boundary_perturb_inputs(inputs, nonfinite=True)),
        )
        for probe_name, make_probe in probes:
            probe = make_probe()
            if synthetic_changes_output(probe):
                return result(
                    "synthetic_sensitivity_only",
                    f"all {fresh_input_trials} natural outputs matched; synthetic probe {probe_name} changed output",
                )
            del probe
        for index in range(len(inputs)):
            probe = _clone_inputs(inputs)
            probe[index] = _strongly_perturb_value(probe[index])
            if synthetic_changes_output(probe):
                return result(
                    "synthetic_sensitivity_only",
                    f"all {fresh_input_trials} natural outputs matched; per-argument probe {index} changed output",
                )
            del probe
        return result(
            "different_input_not_changed",
            f"all {fresh_input_trials} natural outputs and every synthetic probe matched",
        )
    return result("passed", "validation completed")


def _exception_detail(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    return f"{type(exc).__name__}: {text}"[:1000]


def validate_ops_text_detailed(
    ops_text: str,
    *,
    entry_point: str = "Model",
    device: str = "cpu",
    seed: int = 0,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    same_input_repeats: int = 3,
    fresh_input_trials: int = 5,
    fixed_input_repeats: int = 20,
    validation_seeds: int = 1,
    min_natural_output_change_fraction: float = 0.0,
    require_each_forward_input_sensitive: bool = False,
    audit_train_eval: bool = False,
) -> dict[str, Any]:
    """Return a verdict plus an actionable runtime detail."""

    if same_input_repeats < 1:
        return {"verdict": "Failed", "detail": "ValueError: same_input_repeats must be positive"}
    if fresh_input_trials < 1:
        return {"verdict": "Failed", "detail": "ValueError: fresh_input_trials must be positive"}
    if fixed_input_repeats < fresh_input_trials + 1:
        return {
            "verdict": "Failed",
            "detail": "ValueError: fixed_input_repeats must cover the initial and fresh input calls",
        }
    if validation_seeds < 1:
        return {"verdict": "Failed", "detail": "ValueError: validation_seeds must be positive"}
    if not 0.0 <= min_natural_output_change_fraction <= 1.0:
        return {
            "verdict": "Failed",
            "detail": "ValueError: min_natural_output_change_fraction must be in [0, 1]",
        }
    first_result: dict[str, Any] | None = None
    details: list[str] = []
    for seed_index in range(validation_seeds):
        validation_seed = seed + seed_index * 1_000_003
        try:
            current = _validate_on_device(
                ops_text,
                entry_point=entry_point,
                device=device,
                seed=validation_seed,
                rtol=rtol,
                atol=atol,
                same_input_repeats=same_input_repeats,
                fresh_input_trials=fresh_input_trials,
                fixed_input_repeats=fixed_input_repeats,
                min_natural_output_change_fraction=min_natural_output_change_fraction,
                require_each_forward_input_sensitive=require_each_forward_input_sensitive,
                audit_train_eval=audit_train_eval and seed_index == 0,
            )
        except BaseException as exc:  # noqa: BLE001 - isolated caller records every failure
            current = {"verdict": "Failed", "detail": _exception_detail(exc)}
        if first_result is None:
            first_result = current
        details.append(f"seed[{seed_index}]={current['detail']}")
        if current["verdict"] != "passed":
            return {
                **current,
                "detail": f"validation seed {seed_index} failed: {current['detail']}",
                "mode_flags": list((first_result or current).get("mode_flags", ())),
                "mode_detail": str((first_result or current).get("mode_detail", "")),
            }
    assert first_result is not None
    return {
        **first_result,
        "detail": f"passed {validation_seeds} validation seed(s); " + "; ".join(details),
    }


def _validate_on_gpu_subprocess(
    ops_text: str,
    *,
    entry_point: str,
    seed: int,
    rtol: float,
    atol: float,
    timeout: float,
    same_input_repeats: int,
    fresh_input_trials: int,
    fixed_input_repeats: int,
    validation_seeds: int,
    min_natural_output_change_fraction: float,
    require_each_forward_input_sensitive: bool,
    audit_train_eval: bool,
) -> str:
    payload = json.dumps(
        {
            "ops_text": ops_text,
            "entry_point": entry_point,
            "seed": seed,
            "rtol": rtol,
            "atol": atol,
            "same_input_repeats": same_input_repeats,
            "fresh_input_trials": fresh_input_trials,
            "fixed_input_repeats": fixed_input_repeats,
            "validation_seeds": validation_seeds,
            "min_natural_output_change_fraction": min_natural_output_change_fraction,
            "require_each_forward_input_sensitive": require_each_forward_input_sensitive,
            "audit_train_eval": audit_train_eval,
        }
    )
    try:
        # The supplied script used ``-m syn_kernel.validate_ops_text``, but that
        # package does not exist in this repository.  Invoking this file makes
        # fallback work from both a checkout and an installed namespace package.
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--gpu-worker"],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return "Failed"
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(_VERDICT_MARKER):
            encoded = line[len(_VERDICT_MARKER) :].strip()
            try:
                verdict = json.loads(encoded)["verdict"]
            except (json.JSONDecodeError, KeyError, TypeError):
                verdict = encoded
            return verdict if verdict in _VALID_VERDICTS else "Failed"
    return "Failed"


def validate_ops_text(
    ops_text: str,
    *,
    entry_point: str = "Model",
    device: str = "cpu",
    seed: int = 0,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    gpu_fallback: bool = True,
    timeout: float = 180.0,
    same_input_repeats: int = 3,
    fresh_input_trials: int = 5,
    fixed_input_repeats: int = 20,
    validation_seeds: int = 1,
    min_natural_output_change_fraction: float = 0.0,
    require_each_forward_input_sensitive: bool = False,
    audit_train_eval: bool = False,
) -> str:
    """Return the compatibility verdict for one reference implementation."""

    result = validate_ops_text_detailed(
        ops_text,
        entry_point=entry_point,
        device=device,
        seed=seed,
        rtol=rtol,
        atol=atol,
        same_input_repeats=same_input_repeats,
        fresh_input_trials=fresh_input_trials,
        fixed_input_repeats=fixed_input_repeats,
        validation_seeds=validation_seeds,
        min_natural_output_change_fraction=min_natural_output_change_fraction,
        require_each_forward_input_sensitive=require_each_forward_input_sensitive,
        audit_train_eval=audit_train_eval,
    )
    if result["verdict"] != "Failed":
        return result["verdict"]
    if gpu_fallback and device == "cpu" and _import_torch().cuda.is_available():
        return _validate_on_gpu_subprocess(
            ops_text,
            entry_point=entry_point,
            seed=seed,
            rtol=rtol,
            atol=atol,
            timeout=timeout,
            same_input_repeats=same_input_repeats,
            fresh_input_trials=fresh_input_trials,
            fixed_input_repeats=fixed_input_repeats,
            validation_seeds=validation_seeds,
            min_natural_output_change_fraction=min_natural_output_change_fraction,
            require_each_forward_input_sensitive=require_each_forward_input_sensitive,
            audit_train_eval=audit_train_eval,
        )
    return "Failed"


def _run_gpu_worker() -> None:
    try:
        payload = json.loads(sys.stdin.read())
        result = validate_ops_text_detailed(
            payload["ops_text"],
            entry_point=payload.get("entry_point", "Model"),
            device="cuda",
            seed=payload.get("seed", 0),
            rtol=payload.get("rtol", 1e-4),
            atol=payload.get("atol", 1e-5),
            same_input_repeats=payload.get("same_input_repeats", 3),
            fresh_input_trials=payload.get("fresh_input_trials", 5),
            fixed_input_repeats=payload.get("fixed_input_repeats", 20),
            validation_seeds=payload.get("validation_seeds", 1),
            min_natural_output_change_fraction=payload.get(
                "min_natural_output_change_fraction",
                0.0,
            ),
            require_each_forward_input_sensitive=payload.get(
                "require_each_forward_input_sensitive",
                False,
            ),
            audit_train_eval=payload.get("audit_train_eval", False),
        )
    except BaseException:  # noqa: BLE001 - worker failures map to the compatibility verdict
        result = {"verdict": "Failed", "detail": "GPU worker failed"}
    print(f"{_VERDICT_MARKER}{json.dumps(result, ensure_ascii=False)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", help="Reference text path; stdin when omitted.")
    parser.add_argument("--entry-point", default="Model")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--same-input-repeats", type=int, default=3)
    parser.add_argument("--fresh-input-trials", type=int, default=5)
    parser.add_argument("--validation-seeds", type=int, default=1)
    parser.add_argument("--min-natural-output-change-fraction", type=float, default=0.0)
    parser.add_argument("--require-each-forward-input-sensitive", action="store_true")
    parser.add_argument("--gpu-fallback", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gpu-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--detailed", action="store_true")
    args = parser.parse_args()
    if args.gpu_worker:
        _run_gpu_worker()
        return
    ops_text = Path(args.path).read_text(encoding="utf-8") if args.path else sys.stdin.read()
    if args.detailed:
        result: Any = validate_ops_text_detailed(
            ops_text,
            entry_point=args.entry_point,
            device=args.device,
            seed=args.seed,
            rtol=args.rtol,
            atol=args.atol,
            same_input_repeats=args.same_input_repeats,
            fresh_input_trials=args.fresh_input_trials,
            validation_seeds=args.validation_seeds,
            min_natural_output_change_fraction=args.min_natural_output_change_fraction,
            require_each_forward_input_sensitive=args.require_each_forward_input_sensitive,
        )
    else:
        result = validate_ops_text(
            ops_text,
            entry_point=args.entry_point,
            device=args.device,
            seed=args.seed,
            rtol=args.rtol,
            atol=args.atol,
            gpu_fallback=args.gpu_fallback,
            timeout=args.timeout,
            same_input_repeats=args.same_input_repeats,
            fresh_input_trials=args.fresh_input_trials,
            validation_seeds=args.validation_seeds,
            min_natural_output_change_fraction=args.min_natural_output_change_fraction,
            require_each_forward_input_sensitive=args.require_each_forward_input_sensitive,
        )
    print(json.dumps(result, ensure_ascii=False) if isinstance(result, dict) else result)
    verdict = result["verdict"] if isinstance(result, dict) else result
    raise SystemExit(0 if verdict == "passed" else 1)


if __name__ == "__main__":
    main()
