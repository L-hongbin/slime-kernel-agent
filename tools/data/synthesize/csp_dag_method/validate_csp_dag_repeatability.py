#!/usr/bin/env python3
"""Fail-closed repeatability validation for CSP-DAG reference programs.

The validator deliberately creates one model and one fixed input tuple per row, then
calls that same object repeatedly.  It is consequently a liveness test for output
stability under the execution mode selected on the command line; it is not a
replacement for the source-bound static or semantic validators.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import time
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch

CONTRACT = "csp_dag_repeatability_validation_v1"


@contextmanager
def _repeatability_execution_context(device: torch.device) -> Iterable[dict[str, Any]]:
    """Use deterministic cuDNN selection only while validating CUDA rows.

    The repeatability check is specifically about byte-stable executions.  CUDA
    convolution and transposed convolution can otherwise select fast
    nondeterministic cuDNN algorithms.  This narrow context neither changes
    ``torch.use_deterministic_algorithms`` nor leaks backend settings into a
    caller that runs other work in the same Python process.
    """
    if device.type != "cuda":
        yield {
            "device_type": device.type,
            "cudnn_benchmark": None,
            "cudnn_deterministic": None,
            "cudnn_enabled": None,
        }
        return
    previous_benchmark = torch.backends.cudnn.benchmark
    previous_deterministic = torch.backends.cudnn.deterministic
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    context = {
        "device_type": "cuda",
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
    }
    try:
        yield context
    finally:
        torch.backends.cudnn.benchmark = previous_benchmark
        torch.backends.cudnn.deterministic = previous_deterministic


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode())


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _is_minus_one(value: ast.AST) -> bool:
    return (
        isinstance(value, ast.UnaryOp)
        and isinstance(value.op, ast.USub)
        and isinstance(value.operand, ast.Constant)
        and value.operand.value == 1
    )


def _is_zero_index_for_source(index: ast.AST, source: ast.AST) -> bool:
    """Recognize the generator's duplicate-write all-zero-index scatter form."""
    return (
        isinstance(index, ast.Call)
        and isinstance(index.func, ast.Attribute)
        and isinstance(index.func.value, ast.Name)
        and index.func.value.id == "torch"
        and index.func.attr == "zeros_like"
        and bool(index.args)
        and ast.dump(index.args[0], include_attributes=False) == ast.dump(source, include_attributes=False)
    )


def _subscript_of(value: ast.AST, source: ast.AST, *, lower: int | None, upper: int | None) -> bool:
    """Match exactly ``source[..., lower:upper]`` for the singleton proof."""
    if not isinstance(value, ast.Subscript) or ast.dump(value.value, include_attributes=False) != ast.dump(
        source, include_attributes=False
    ):
        return False
    slice_node = value.slice
    if not (
        isinstance(slice_node, ast.Tuple)
        and len(slice_node.elts) == 2
        and isinstance(slice_node.elts[0], ast.Constant)
        and slice_node.elts[0].value is Ellipsis
        and isinstance(slice_node.elts[1], ast.Slice)
    ):
        return False
    actual = slice_node.elts[1]

    def _integer_or_none(node: ast.AST | None) -> int | None | str:
        if node is None:
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and node.operand.value == 1
        ):
            return -1
        return "not_a_supported_slice_bound"

    return _integer_or_none(actual.lower) == lower and _integer_or_none(actual.upper) == upper


def _is_proven_unique_singleton(index: ast.AST, source_value: ast.AST) -> bool:
    """Recognize the v3 repair's exact, one-write-per-outer-row scatter form."""
    if not (
        isinstance(index, ast.Call)
        and isinstance(index.func, ast.Attribute)
        and isinstance(index.func.value, ast.Name)
        and index.func.value.id == "torch"
        and index.func.attr == "zeros_like"
        and bool(index.args)
        and isinstance(source_value, ast.Subscript)
    ):
        return False
    base = source_value.value
    return _subscript_of(index.args[0], base, lower=None, upper=1) and _subscript_of(
        source_value, base, lower=-1, upper=None
    )


def classify_scatter(code: str) -> dict[str, int | str]:
    """Classify scatter calls without claiming unproved calls are safe.

    ``unsafe_zero_index`` is syntactically known to have duplicate destinations
    whenever the final dimension has more than one element.  The exact singleton
    form used by the v3 repair is proven unique. Other scatter calls are reported
    as ``unproven`` rather than called safe: index uniqueness is a runtime property
    in the general case. ``--require-safe-scatter`` rejects unsafe and unproven
    categories, but allows proven singleton scatters.
    """
    tree = ast.parse(code)
    all_scatter = 0
    unsafe_zero_index = 0
    proven_unique_singleton = 0
    unproven = 0
    for node in ast.walk(tree):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"scatter", "scatter_"}
        ):
            continue
        all_scatter += 1
        if (
            len(node.args) >= 3
            and _is_minus_one(node.args[0])
            and _is_proven_unique_singleton(node.args[1], node.args[2])
        ):
            proven_unique_singleton += 1
        elif (
            len(node.args) >= 3
            and _is_minus_one(node.args[0])
            and _is_zero_index_for_source(node.args[1], node.args[2])
        ):
            unsafe_zero_index += 1
        else:
            unproven += 1
    return {
        "scatter_calls": all_scatter,
        "unsafe_zero_index_scatter_calls": unsafe_zero_index,
        "proven_unique_singleton_scatter_calls": proven_unique_singleton,
        "unproven_scatter_calls": unproven,
        "scatter_status": (
            "none"
            if not all_scatter
            else "unsafe_zero_index" if unsafe_zero_index else "unproven" if unproven else "proven_unique_singleton"
        ),
    }


def _move(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, tuple):
        return tuple(_move(item, device) for item in value)
    if isinstance(value, list):
        return [_move(item, device) for item in value]
    if isinstance(value, dict):
        return {key: _move(item, device) for key, item in value.items()}
    return value


def _tensor_paths(value: Any, path: str = "output") -> Iterable[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            yield from _tensor_paths(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _tensor_paths(value[key], f"{path}[{key!r}]")


def _non_tensor_leaf_count(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return 0
    if isinstance(value, (tuple, list)):
        return sum(_non_tensor_leaf_count(item) for item in value)
    if isinstance(value, dict):
        return sum(_non_tensor_leaf_count(item) for item in value.values())
    return 1


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    # uint8 view preserves exact NaN payloads and distinguishes signed zero.
    value = tensor.detach().contiguous()
    if value.device.type != "cpu":
        value = value.cpu()
    # ``view`` cannot change item size on a scalar tensor, while flattening is
    # byte-identical and works for scalars as well as higher-rank tensors.
    return value.reshape(-1).view(torch.uint8).numpy().tobytes()


def tensor_fingerprint(value: Any, path: str = "output") -> dict[str, Any]:
    tensors = list(_tensor_paths(value, path))
    leaves = []
    digest = hashlib.sha256()
    for path, tensor in tensors:
        metadata = {"path": path, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        raw = _tensor_bytes(tensor)
        digest.update(_canonical_json(metadata).encode())
        digest.update(raw)
        leaves.append({**metadata, "bytes": len(raw), "sha256": _sha256_bytes(raw)})
    return {
        "tensor_count": len(tensors),
        "non_tensor_leaf_count": _non_tensor_leaf_count(value),
        "leaves": leaves,
        "sha256": digest.hexdigest(),
    }


def _model_state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        metadata = {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        digest.update(_canonical_json(metadata).encode())
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def _load_manifest(path: Path) -> tuple[dict[str, dict[str, Any]], list[str]]:
    by_uuid: dict[str, dict[str, Any]] = {}
    duplicate_uuids: list[str] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        uuid = item.get("uuid")
        if not isinstance(uuid, str) or not uuid:
            raise ValueError(f"manifest line {line_number} has no nonempty uuid")
        if uuid in by_uuid:
            duplicate_uuids.append(uuid)
        by_uuid[uuid] = item
    return by_uuid, duplicate_uuids


def _row_record(
    row: dict[str, Any],
    manifest: dict[str, Any] | None,
    device: torch.device,
    trials: int,
    eval_mode: bool,
    require_safe_scatter: bool,
    execution_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    extra = row.get("extra_info", {})
    uuid = extra.get("uuid") if isinstance(extra, dict) else None
    reward = row.get("reward_model", {})
    code = reward.get("ground_truth") if isinstance(reward, dict) else None
    checks: dict[str, bool] = {
        "uuid_present": isinstance(uuid, str) and bool(uuid),
        "code_present": isinstance(code, str),
        "manifest_present": manifest is not None,
    }
    errors: list[str] = []
    scatter: dict[str, int | str] = {
        "scatter_calls": 0,
        "unsafe_zero_index_scatter_calls": 0,
        "proven_unique_singleton_scatter_calls": 0,
        "unproven_scatter_calls": 0,
        "scatter_status": "unavailable",
    }
    runtime: dict[str, Any] = {}
    if checks["uuid_present"] and checks["code_present"] and manifest is not None:
        checks["uuid_bound"] = manifest.get("uuid") == uuid
        checks["reference_hash_bound"] = manifest.get("reference_sha256") == _sha256_text(code)
        try:
            scatter = classify_scatter(code)
            checks["safe_scatter_required"] = not require_safe_scatter or (
                scatter["unsafe_zero_index_scatter_calls"] == 0 and scatter["unproven_scatter_calls"] == 0
            )
            if require_safe_scatter and not checks["safe_scatter_required"]:
                failed_checks = sorted(name for name, passed in checks.items() if not passed)
                return {
                    "uuid": uuid,
                    "repeatability_execution": execution_context,
                    "passed": False,
                    "failed_checks": failed_checks,
                    "errors": [],
                    "checks": checks,
                    "scatter": scatter,
                    "runtime": {},
                    "elapsed_seconds": round(time.perf_counter() - started, 6),
                }
            namespace: dict[str, Any] = {}
            exec(compile(code, f"<{uuid}>", "exec"), namespace)
            # A single deterministic input/model construction is reused for all trials.
            torch.manual_seed(0)
            source_args = namespace["get_inputs"]()
            args = _move(source_args, device)
            torch.manual_seed(1)
            model = namespace["Model"]().to(device)
            if eval_mode:
                model.eval()
            _sync(device)
            source_input = tensor_fingerprint(source_args, "source_inputs")
            input_fingerprint = tensor_fingerprint(args, "inputs")
            initial_state_hash = _model_state_hash(model)
            trial_outputs: list[dict[str, Any]] = []
            with torch.no_grad():
                for _trial in range(trials):
                    output = model(*args)
                    _sync(device)
                    trial_outputs.append(tensor_fingerprint(output))
            first = trial_outputs[0]
            checks["runtime_tensor_output"] = first["tensor_count"] > 0
            checks["runtime_tensor_only_output"] = first["non_tensor_leaf_count"] == 0
            checks["output_shape_dtype_exact"] = all(
                [{key: leaf[key] for key in ("path", "shape", "dtype")} for leaf in item["leaves"]]
                == [{key: leaf[key] for key in ("path", "shape", "dtype")} for leaf in first["leaves"]]
                for item in trial_outputs
            )
            checks["output_exact_bytes"] = all(item["sha256"] == first["sha256"] for item in trial_outputs)
            runtime = {
                "source_input": source_input,
                "input": input_fingerprint,
                "initial_model_state_sha256": initial_state_hash,
                "final_model_state_sha256": _model_state_hash(model),
                "trial_outputs": trial_outputs,
            }
        except Exception as exc:  # retain a row-level, fail-closed error ledger
            errors.append(f"{type(exc).__name__}: {exc}")
    else:
        checks["safe_scatter_required"] = not require_safe_scatter
    failed_checks = sorted(name for name, passed in checks.items() if not passed)
    return {
        "uuid": uuid,
        "repeatability_execution": execution_context,
        "passed": not errors and not failed_checks,
        "failed_checks": failed_checks,
        "errors": errors,
        "checks": checks,
        "scatter": scatter,
        "runtime": runtime,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    parquet_path = args.parquet.resolve()
    manifest_path = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("require 0 <= --shard-index < --num-shards")
    if args.trials < 2:
        raise ValueError("--trials must be at least 2")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    manifests, duplicate_manifest_uuids = _load_manifest(manifest_path)
    rows = pq.read_table(parquet_path).to_pylist()
    selected = [row for position, row in enumerate(rows) if position % args.num_shards == args.shard_index]
    output_dir.mkdir(parents=True, exist_ok=True)
    with _repeatability_execution_context(device) as execution_context:
        records = [
            _row_record(
                row,
                (
                    manifests.get(row.get("extra_info", {}).get("uuid"))
                    if isinstance(row.get("extra_info"), dict)
                    else None
                ),
                device,
                args.trials,
                args.eval_mode,
                args.require_safe_scatter,
                execution_context,
            )
            for row in selected
        ]
    records_path = output_dir / f"shard_{args.shard_index:03d}_of_{args.num_shards:03d}.records.jsonl"
    records_path.write_text("".join(_canonical_json(record) + "\n" for record in records))
    summary = {
        "contract": CONTRACT,
        "device": str(device),
        "torch_version": torch.__version__,
        "eval_mode": args.eval_mode,
        "trials": args.trials,
        "require_safe_scatter": args.require_safe_scatter,
        "repeatability_execution": execution_context,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "source_binding": {
            "parquet": {"path": str(parquet_path), "sha256": _sha256_file(parquet_path), "rows": len(rows)},
            "manifest": {"path": str(manifest_path), "sha256": _sha256_file(manifest_path), "rows": len(manifests)},
            "validator": {"path": str(Path(__file__).resolve()), "sha256": _sha256_file(Path(__file__).resolve())},
        },
        "selected_rows": len(records),
        "passed": sum(record["passed"] for record in records),
        "failed": sum(not record["passed"] for record in records),
        "unsafe_zero_index_rows": sum(record["scatter"]["unsafe_zero_index_scatter_calls"] > 0 for record in records),
        "unproven_scatter_rows": sum(record["scatter"]["unproven_scatter_calls"] > 0 for record in records),
        "duplicate_manifest_uuids": sorted(set(duplicate_manifest_uuids)),
        "records": {"path": str(records_path), "sha256": _sha256_file(records_path)},
    }
    summary["all_passed"] = summary["failed"] == 0 and not summary["duplicate_manifest_uuids"]
    summary_path = output_dir / f"shard_{args.shard_index:03d}_of_{args.num_shards:03d}.summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--eval-mode", action="store_true", help="Run Model.eval(); default preserves training mode.")
    parser.add_argument(
        "--require-safe-scatter",
        action="store_true",
        help="Fail rows containing known-unsafe or unproven scatter calls; exact proven-singleton scatters are allowed.",
    )
    return parser.parse_args()


def main() -> int:
    summary = validate(parse_args())
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
