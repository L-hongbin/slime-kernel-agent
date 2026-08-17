#!/usr/bin/env python3
"""Validate a row shard of a materialized CSP-DAG dataset."""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import torch

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.synthesize.csp_dag_method import build_csp_dag_input_expansions as expansions
from tools.data.synthesize.csp_dag_method import generate_csp_dag as generator
from tools.data.synthesize.csp_dag_method import validate_csp_dag as validator
from tools.data.synthesize.csp_dag_method import validate_csp_dag_repeatability as repeatability

CONTRACT = "open_csp_dag_dataset_validation_v1"

_SHAPE_MANIFEST_CONTRACT = expansions.CONTRACT
_DTYPE_MANIFEST_CONTRACT = "dtype_lane_manifest_v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _expansion_kind(manifests: list[dict[str, Any]]) -> str | None:
    """Recognize one homogeneous selected-expansion manifest batch.

    A selected lane is only meaningful together with its selection summary.  Do
    not infer a lane from a row field: a mixed or unknown batch is rejected
    before any generated source is executed.
    """

    if not manifests:
        raise ValueError("manifest is empty")
    signatures = {
        (
            manifest.get("contract"),
            manifest.get("manifest_contract_version"),
        )
        for manifest in manifests
    }
    if len(signatures) != 1:
        raise ValueError("mixed manifest contracts are not valid:" f"{sorted(map(repr, signatures))!r}")
    contract, manifest_contract = next(iter(signatures))
    if contract == _SHAPE_MANIFEST_CONTRACT and manifest_contract is None:
        return "shape"
    if manifest_contract == _DTYPE_MANIFEST_CONTRACT:
        return "dtype"
    return None


def _load_bound_rows(
    parquet_path: Path,
    manifest_path: Path,
    summary_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str, dict[str, Any] | None]:
    """Load a direct base batch or a source-bound selected expansion batch."""

    preliminary = validator._load_manifest(manifest_path)
    kind = _expansion_kind(preliminary)
    if kind is None:
        rows = pq.read_table(parquet_path).to_pylist()
        if len(rows) != len(preliminary):
            raise ValueError("parquet and manifest row counts differ")
        return rows, preliminary, "base_full_row", None

    if kind == "shape" and summary_path.name == "summary.json":
        # Shape pre-selection executes the builder's candidates before a
        # selection summary exists.  It is still byte-bound to the candidate
        # summary and production builder, and is deliberately distinct from
        # selected shape post-selection below.
        if summary_path.name != "summary.json" or summary_path.resolve().parent != parquet_path.resolve().parent:
            raise ValueError("shape candidates require their exact builder summary.json")
        summary = json.loads(summary_path.read_text())
        if summary.get("contract") != expansions.CONTRACT or summary.get("stage") != "shape":
            raise ValueError("shape candidate summary contract mismatch")
        if summary.get("review_only") is not True or summary.get("training_approved") is not False:
            raise ValueError("shape candidate summary governance mismatch")
        artifacts = summary.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError("shape candidate summary artifacts missing")
        for name, path in (("candidates", parquet_path), ("manifest", manifest_path)):
            binding = artifacts.get(name)
            if (
                not isinstance(binding, dict)
                or binding.get("path") != str(path.resolve())
                or binding.get("sha256") != _sha256_file(path)
            ):
                raise ValueError(f"shape candidate summary {name} binding mismatch")
        expansions._validate_lane_builder_binding(summary, parquet_path.resolve().parent, "shape")
        rows = pq.read_table(parquet_path).to_pylist()
        manifests = preliminary
        if len(rows) != len(manifests) or not rows:
            raise ValueError("shape candidates and manifests differ or are empty")
        for index, (row, manifest) in enumerate(zip(rows, manifests, strict=True)):
            uuid, code = expansions._identity(row)
            if (
                manifest.get("uuid") != uuid
                or manifest.get("reference_sha256") != expansions._sha256_text(code)
                or manifest.get("normalized_ast_sha256") != expansions._normalized_ast_sha256(code)
            ):
                raise ValueError(f"shape candidate identity binding mismatch:{index}")
            graph_checks, nodes, rank, shape = expansions.static_binding.typed_graph_context(manifest)
            lowering_checks = expansions.static_binding.source_lowering_checks(code, manifest, nodes, rank, shape)
            if not all(value is True for value in {**graph_checks, **lowering_checks}.values()):
                raise ValueError(f"shape candidate static binding mismatch:{index}")
        return (
            rows,
            manifests,
            "shape_candidate_full_row",
            {
                "contract": expansions.SOURCE_BINDING_CONTRACT,
                "stage": "shape_candidate",
                "source_summary": {"path": str(summary_path.resolve()), "sha256": _sha256_file(summary_path)},
                "source_artifact": {"path": str(parquet_path.resolve()), "sha256": _sha256_file(parquet_path)},
                "source_manifest": {"path": str(manifest_path.resolve()), "sha256": _sha256_file(manifest_path)},
                "source_rows": len(rows),
            },
        )
    expected = (
        ("contract", _SHAPE_MANIFEST_CONTRACT)
        if kind == "shape"
        else ("manifest_contract_version", _DTYPE_MANIFEST_CONTRACT)
    )
    rows, manifests, expansion_binding = expansions._verify_source_rows(
        parquet_path,
        manifest_path,
        stage=kind,
        expected_manifest=expected,
    )
    bound_summary = Path(expansion_binding["source_summary"]["path"])
    if summary_path.resolve() != bound_summary:
        raise ValueError("summary must be the exact selection summary bound to the supplied selected artifacts")
    mode = "shape_full_row" if kind == "shape" else f"{kind}_standalone_selected_child"
    return rows, manifests, mode, expansion_binding


def _child_identity_checks(row: dict[str, Any], manifest: dict[str, Any]) -> tuple[str, str, dict[str, bool]]:
    """Check the selected child identity independently of paired liveness."""

    code = row["reward_model"]["ground_truth"]
    uuid = row["extra_info"]["uuid"]
    expected_uuid = manifest.get("uuid", manifest.get("child_uuid"))
    expected_hash = manifest.get("reference_sha256", manifest.get("child_reference_sha256"))
    expected_ast = manifest.get("normalized_ast_sha256", manifest.get("child_normalized_ast_sha256"))
    parent_uuid = manifest.get("parent_uuid")
    v4 = row.get("extra_info", {}).get("v4", {})
    checks = {
        "uuid_bound": isinstance(expected_uuid, str) and uuid == expected_uuid,
        "reference_hash_bound": validator._sha256_text(code) == expected_hash,
        "ast_hash_bound": generator._normalized_ast_hash(code) == expected_ast,
        "governance": (
            manifest.get("training_approved") is False
            and manifest.get("materialization_status") == "review_only"
            and isinstance(parent_uuid, str)
            and bool(parent_uuid)
            and isinstance(v4, dict)
            and v4.get("parent_uuid") == parent_uuid
        ),
    }
    return uuid, code, checks


def _floating_dtype_matches(tensor: torch.Tensor, target: torch.dtype) -> bool:
    return not (tensor.is_floating_point() or tensor.is_complex()) or tensor.dtype == target


def _tensor_metadata_snapshot(tensors: list[torch.Tensor]) -> list[dict[str, Any]]:
    return [
        {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "strides": list(tensor.stride()),
            "storage_offset": tensor.storage_offset(),
            "is_contiguous": tensor.is_contiguous(),
            "zero_stride_dimensions": [axis for axis, value in enumerate(tensor.stride()) if value == 0],
        }
        for tensor in tensors
    ]


def _get_inputs_for_device(factory: Any, device: torch.device) -> Any:
    """Run a generated input factory directly on the selected CUDA device."""
    if device.type != "cuda":
        return factory()
    previous = torch.get_default_device()
    torch.set_default_device(device)
    try:
        return factory()
    finally:
        torch.set_default_device(previous)


def _validate_dtype_row(
    row: dict[str, Any],
    manifest: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Execute one selected dtype child without reusing base lowering checks.

    The paired CUDA liveness record remains the semantic authority; this pass
    separately checks source-bound child integrity and execution.
    """

    started = time.perf_counter()
    checks: dict[str, Any] = {}
    errors: list[str] = []
    runtime: dict[str, Any] = {}
    uuid = str(row.get("extra_info", {}).get("uuid", "<missing>"))
    try:
        uuid, code, checks = _child_identity_checks(row, manifest)
        checks["manifest_kind"] = manifest.get("manifest_contract_version") == _DTYPE_MANIFEST_CONTRACT
        checks["primary_intervention"] = manifest.get("primary_intervention") == "dtype"
        realized = manifest.get("realized_intervention")
        augmentation = row.get("extra_info", {}).get("augmentation")
        checks["dtype_manifest_intervention_bound"] = (
            isinstance(realized, dict)
            and realized.get("dtype_after") == manifest.get("assigned_target")
            and realized.get("factory_count") == manifest.get("factory_count")
            and realized.get("logical_forward_dtype_policy")
            == "all_floating_vN_and_final_outputs_match_assigned_dtype_v1"
            and realized.get("internal_aten_non_target_floating_outputs") == "diagnostic_only"
            and realized.get("internal_aten_complex_outputs") == "reject"
            and "fp32_fallback_policy" not in realized
        )
        checks["augmentation_manifest_bound"] = (
            isinstance(augmentation, dict)
            and augmentation.get("parent_uuid") == manifest.get("parent_uuid")
            and augmentation.get("child_uuid") == manifest.get("child_uuid")
            and augmentation.get("intervention_kind") == "dtype"
            and augmentation.get("factory_count") == manifest.get("factory_count")
            and augmentation.get("dtype_after") == manifest.get("assigned_target")
            and augmentation.get("layout_after") is None
        )
        namespace: dict[str, Any] = {}
        exec(compile(code, f"<{uuid}>", "exec"), namespace)
        checks["get_inputs_present"] = callable(namespace.get("get_inputs"))
        checks["model_present"] = isinstance(namespace.get("Model"), type)
        if not checks["get_inputs_present"] or not checks["model_present"]:
            raise ValueError("reference lacks callable get_inputs or Model")
        torch.manual_seed(0)
        source_args = _get_inputs_for_device(namespace["get_inputs"], device)
        source_input_tensors = list(validator._tensors(source_args))
        args = validator._move(source_args, device)
        input_tensors = list(validator._tensors(args))
        model = namespace["Model"]().eval().to(device)

        target_name = manifest.get("assigned_target")
        target = getattr(torch, str(target_name), None)
        if target_name not in {"float16", "bfloat16"} or not isinstance(target, torch.dtype):
            raise ValueError(f"unsupported dtype target:{target_name!r}")
        checks["floating_inputs_assigned_target"] = bool(source_input_tensors) and all(
            _floating_dtype_matches(tensor, target) for tensor in source_input_tensors
        )
        checks["device_inputs_assigned_target"] = bool(input_tensors) and all(
            _floating_dtype_matches(tensor, target) for tensor in input_tensors
        )
        registered = [*model.named_parameters(recurse=True), *model.named_buffers(recurse=True)]
        checks["registered_floating_state_assigned_target"] = all(
            _floating_dtype_matches(tensor, target) for _, tensor in registered
        )
        runtime["assigned_target"] = f"torch.{target_name}"
        runtime["registered_parameter_dtypes"] = {
            name: str(tensor.dtype) for name, tensor in model.named_parameters(recurse=True)
        }
        runtime["registered_buffer_dtypes"] = {
            name: str(tensor.dtype) for name, tensor in model.named_buffers(recurse=True)
        }

        trace = validator.DispatchTrace()
        with torch.no_grad(), trace:
            output = model(*args)
        tensors = list(validator._tensors(output))
        checks["runtime_tensor_output"] = bool(tensors)
        checks["runtime_finite"] = bool(tensors) and all(torch.isfinite(tensor).all().item() for tensor in tensors)
        checks["runtime_dispatch_nonempty"] = bool(trace.events)
        target = getattr(torch, str(manifest["assigned_target"]))
        checks["no_unexpected_float32_output"] = bool(tensors) and all(
            _floating_dtype_matches(tensor, target) for tensor in tensors
        )
        runtime.update(
            {
                "dispatch_count": len(trace.events),
                "unique_dispatch_count": len(set(trace.events)),
                "source_input_shapes": [list(tensor.shape) for tensor in source_input_tensors],
                "source_input_dtypes": [str(tensor.dtype) for tensor in source_input_tensors],
                "source_input_strides": [list(tensor.stride()) for tensor in source_input_tensors],
                "source_input_storage_offsets": [tensor.storage_offset() for tensor in source_input_tensors],
                "source_input_is_contiguous": [tensor.is_contiguous() for tensor in source_input_tensors],
                "device_input_layout_metadata": _tensor_metadata_snapshot(input_tensors),
                "input_dtypes": [str(tensor.dtype) for tensor in input_tensors],
                "output_dtypes": [str(tensor.dtype) for tensor in tensors],
            }
        )
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    failed_checks = sorted(name for name, passed in checks.items() if passed is not True)
    return {
        "uuid": uuid,
        "passed": not errors and not failed_checks,
        "failed_checks": failed_checks,
        "errors": errors,
        "checks": checks,
        "runtime": runtime,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
    }


def validate(
    parquet_path: Path,
    manifest_path: Path,
    summary_path: Path,
    device_name: str,
    shard_index: int,
    shard_count: int,
    output_dir: Path,
    workers: int,
) -> dict[str, Any]:
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")
    rows, manifests, validation_mode, expansion_binding = _load_bound_rows(parquet_path, manifest_path, summary_path)
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.set_num_threads(1)
    device = torch.device(device_name)
    gpu_identity = None
    if device.type == "cuda":
        torch.cuda.set_device(device)
        properties = torch.cuda.get_device_properties(device)
        gpu_identity = {
            "device": str(device),
            "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
        }
    positions = list(range(shard_index, len(rows), shard_count))

    def validate_position(position: int) -> dict[str, Any]:
        if validation_mode in {"base_full_row", "shape_full_row", "shape_candidate_full_row"}:
            record = validator._validate_row(rows[position], manifests[position], device)
        else:
            record = _validate_dtype_row(rows[position], manifests[position], device)
        return {"position": position, **record}

    execution_context = (
        repeatability._repeatability_execution_context(device)
        if device.type == "cuda"
        else contextlib.nullcontext(
            {
                "device_type": device.type,
                "cudnn_benchmark": None,
                "cudnn_deterministic": None,
                "cudnn_enabled": None,
            }
        )
    )
    with execution_context as runtime_execution:
        if workers == 1:
            records = [validate_position(position) for position in positions]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                records = list(pool.map(validate_position, positions))

    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / f"shard_{shard_index:03d}_of_{shard_count:03d}.records.jsonl"
    records_path.write_text("".join(generator._canonical_json(record) + "\n" for record in records))
    script_path = Path(__file__).resolve()
    result = {
        "contract": CONTRACT,
        "device": str(device),
        "gpu_identity": gpu_identity,
        "host": __import__("socket").gethostname(),
        "torch_version": torch.__version__,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "workers": workers,
        "validation_mode": validation_mode,
        "runtime_execution": runtime_execution,
        "rows": len(records),
        "passed": sum(record["passed"] for record in records),
        "failed": sum(not record["passed"] for record in records),
        "positions": {"first": positions[0] if positions else None, "last": positions[-1] if positions else None},
        "dispatch_count": sum(record.get("runtime", {}).get("dispatch_count", 0) for record in records),
        "source_binding": {
            "validator": {"path": str(script_path), "sha256": _sha256_file(script_path)},
            "row_validator": {
                "path": str(Path(validator.__file__).resolve()),
                "sha256": _sha256_file(Path(validator.__file__)),
            },
            "execution_context": {
                "path": str(Path(repeatability.__file__).resolve()),
                "sha256": _sha256_file(Path(repeatability.__file__)),
            },
            "generator": {
                "path": str(Path(generator.__file__).resolve()),
                "sha256": _sha256_file(Path(generator.__file__)),
            },
            "expansion_adapter": {
                "path": str(Path(expansions.__file__).resolve()),
                "sha256": _sha256_file(Path(expansions.__file__)),
                "contract": expansions.SOURCE_BINDING_CONTRACT,
            },
            "selected": {"path": str(parquet_path.resolve()), "sha256": _sha256_file(parquet_path)},
            "manifest": {"path": str(manifest_path.resolve()), "sha256": _sha256_file(manifest_path)},
            "summary": {"path": str(summary_path.resolve()), "sha256": _sha256_file(summary_path)},
            **(
                {"expansion_candidate_binding": expansion_binding}
                if validation_mode == "shape_candidate_full_row"
                else {"expansion_selected_binding": expansion_binding}
            ),
        },
        "records": {"path": str(records_path.resolve()), "sha256": _sha256_file(records_path)},
        "review_only": True,
        "training_approved": False,
    }
    summary_out = output_dir / f"shard_{shard_index:03d}_of_{shard_count:03d}.summary.json"
    _write_json(summary_out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    if args.workers <= 0:
        raise ValueError("workers must be positive")
    print(
        json.dumps(
            validate(
                args.parquet,
                args.manifest,
                args.summary,
                args.device,
                args.shard_index,
                args.shard_count,
                args.output_dir,
                args.workers,
            ),
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
