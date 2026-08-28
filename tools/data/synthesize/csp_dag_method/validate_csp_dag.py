#!/usr/bin/env python3
"""Validate static contracts and execute every open CSP-DAG reference."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import networkx as nx
import pyarrow.parquet as pq
import torch
from torch.utils._python_dispatch import TorchDispatchMode

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.cleaning.complexity import extract_complexity_features, extract_operator_signature
from tools.data.synthesize.csp_dag_method import csp_dag_static_binding as static_binding
from tools.data.synthesize.csp_dag_method import generate_csp_dag as generator
from tools.data.synthesize.select_low_level_kernelbench_canary import FORBIDDEN_COMPOSITE_APIS, _token_cells

CONTRACT = "open_csp_dag_reference_validation_v3"
_DEFAULT_RUN = (
    _REPO_ROOT / "local_artifacts/data/synthesize/csp_dag_low_level_5k/run.generated5000.strict_near_dedup.v4"
)


class DispatchTrace(TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []

    def __torch_dispatch__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        self.events.append(str(func))
        return func(*args, **(kwargs or {}))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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


def _contains_template_identifier(value: Any) -> bool:
    """Find a real generation-time template identifier without self-inspecting evidence."""

    if isinstance(value, dict):
        return any(
            key == "template_id" or _contains_template_identifier(item)
            for key, item in value.items()
            if key != "runtime_evidence"
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_template_identifier(item) for item in value)
    return False


def _tensors(value: Any) -> Iterable[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _tensors(item)


def _graph_check(manifest: dict[str, Any]) -> dict[str, Any]:
    try:
        typed = manifest["graph"]["typed_graph"]
        nodes = typed["nodes"]
        edges = typed["edges"]
        if not isinstance(nodes, list) or not nodes:
            raise ValueError("typed graph has no nodes")
        graph = nx.DiGraph()
        graph.add_node(-1)
        graph.add_nodes_from(node["index"] for node in nodes)
        graph.add_edges_from(tuple(edge) for edge in edges)
        final = nodes[-1]["index"]
        return {
            "acyclic": nx.is_directed_acyclic_graph(graph),
            "connected_from_input": len(nx.descendants(graph, -1)) == len(graph) - 1,
            "all_nodes_reach_return": len(nx.ancestors(graph, final)) == len(graph) - 1,
        }
    except (KeyError, TypeError, ValueError, nx.NetworkXError):
        return {
            "acyclic": False,
            "connected_from_input": False,
            "all_nodes_reach_return": False,
        }


def _validate_row(
    row: dict[str, Any],
    manifest: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    started = time.perf_counter()
    code = row["reward_model"]["ground_truth"]
    uuid = row["extra_info"]["uuid"]
    checks: dict[str, Any] = {}
    errors: list[str] = []
    try:
        checks["uuid_bound"] = uuid == manifest["uuid"]
        checks["reference_hash_bound"] = _sha256_text(code) == manifest["reference_sha256"]
        checks["ast_hash_bound"] = generator._normalized_ast_hash(code) == manifest["normalized_ast_sha256"]
        # Use the static-only implementation so runtime and quality gates share
        # an exact fail-closed graph/source contract.
        graph_binding_checks, nodes, rank, shape = static_binding.typed_graph_context(manifest)
        checks.update(graph_binding_checks)
        checks.update(static_binding.source_lowering_checks(code, manifest, nodes, rank, shape))
        parent_uuid = manifest.get("parent_uuid")
        original_governance = parent_uuid is None
        expansion_governance = (
            manifest.get("contract") == "csp_dag_shape_dtype_layout_expansion_v1"
            and isinstance(parent_uuid, str)
            and bool(parent_uuid)
            and row.get("extra_info", {}).get("v4", {}).get("parent_uuid") == parent_uuid
        )
        checks["governance"] = (
            (original_governance or expansion_governance)
            and manifest.get("training_approved") is False
            and manifest.get("materialization_status") == "review_only"
        )
        checks["no_template_identifier"] = not _contains_template_identifier(manifest)
        checks["no_repeated_provenance_padding"] = "lowering provenance" not in code
        signature = extract_operator_signature(code)
        features = generator.feature_dict(extract_complexity_features(code))
        actual_families = sorted(
            {cell for token in signature for cell in _token_cells(token) if cell in generator.FAMILIES}
        )
        checks["signature_bound"] = list(signature) == manifest["operator_signature"]
        checks["families_bound"] = actual_families == manifest["actual_low_level_families"]
        checks["complexity_bound"] = features == manifest["complexity"]
        current_policy = static_binding.uses_single_model_policy(manifest)
        checks["multiclass_bound"] = (
            current_policy
            and int(features["top_level_class_count"]) == 1
            and "requested_multiclass" not in manifest
            and manifest.get("top_level_class_policy") == "single_model_only"
        )
        checks["multiclass_helper_reachable"] = (
            current_policy
            and manifest.get("semantic_multiclass_status") == "deferred_until_real_subgraph_helper_lowering"
        )
        checks["no_forbidden_composites"] = not (set(signature) & FORBIDDEN_COMPOSITE_APIS)
        normalization_bindings = []
        for node in nodes or []:
            if node.rule != "normalization":
                continue
            expected_class = "nn.GroupNorm" if node.variant == "group_norm" else "nn.LayerNorm"
            normalization_bindings.append(f"self.{node.module_name} = {expected_class}(" in code)
        checks["normalization_variant_lowering_bound"] = bool(nodes) and all(normalization_bindings)
        checks.update({f"graph_{key}": value for key, value in _graph_check(manifest).items()})

        namespace: dict[str, Any] = {}
        exec(compile(code, f"<{uuid}>", "exec"), namespace)
        torch.manual_seed(0)
        source_args = namespace["get_inputs"]()
        source_input_tensors = list(_tensors(source_args))
        args = _move(source_args, device)
        input_tensors = list(_tensors(args))
        model = namespace["Model"]().eval().to(device)
        trace = DispatchTrace()
        with torch.no_grad(), trace:
            output = model(*args)
        tensors = list(_tensors(output))
        checks["runtime_tensor_output"] = bool(tensors)
        checks["runtime_finite"] = bool(tensors) and all(torch.isfinite(tensor).all().item() for tensor in tensors)
        checks["runtime_dispatch_nonempty"] = bool(trace.events)
        dispatch = collections.Counter(event.split(".")[0] + "." + event.split(".")[1] for event in trace.events)
        runtime = {
            "dispatch_count": len(trace.events),
            "unique_dispatch_count": len(set(trace.events)),
            "dispatch_prefixes": dict(dispatch.most_common(20)),
            "source_input_shapes": [list(tensor.shape) for tensor in source_input_tensors],
            "source_input_dtypes": [str(tensor.dtype) for tensor in source_input_tensors],
            "source_input_strides": [list(tensor.stride()) for tensor in source_input_tensors],
            "source_input_storage_offsets": [tensor.storage_offset() for tensor in source_input_tensors],
            "source_input_is_contiguous": [tensor.is_contiguous() for tensor in source_input_tensors],
            "input_shapes": [list(tensor.shape) for tensor in input_tensors],
            "input_dtypes": [str(tensor.dtype) for tensor in input_tensors],
            "input_strides": [list(tensor.stride()) for tensor in input_tensors],
            "input_storage_offsets": [tensor.storage_offset() for tensor in input_tensors],
            "input_is_contiguous": [tensor.is_contiguous() for tensor in input_tensors],
            "output_shapes": [list(tensor.shape) for tensor in tensors],
            "output_dtypes": [str(tensor.dtype) for tensor in tensors],
        }
    except Exception as exc:  # keep a complete row failure ledger
        errors.append(f"{type(exc).__name__}: {exc}")
        runtime = {}
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


def validate(run_dir: Path, device_name: str) -> dict[str, Any]:
    selected_path = run_dir / "selected.parquet"
    manifest_path = run_dir / "selected.manifest.jsonl"
    summary_path = run_dir / "summary.json"
    rows = pq.read_table(selected_path).to_pylist()
    manifests = _load_manifest(manifest_path)
    if len(rows) != len(manifests):
        raise ValueError("selected parquet and manifest row counts differ")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(device_name)
    torch.set_num_threads(1)
    records = [_validate_row(row, manifest, device) for row, manifest in zip(rows, manifests, strict=True)]

    output_dir = run_dir / "runtime" / device_name.replace(":", "_")
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    records_path.write_text("".join(generator._canonical_json(record) + "\n" for record in records))
    validator_path = Path(__file__).resolve()
    result = {
        "contract": CONTRACT,
        "device": str(device),
        "torch_version": torch.__version__,
        "rows": len(records),
        "passed": sum(record["passed"] for record in records),
        "failed": sum(not record["passed"] for record in records),
        "dispatch_count": sum(record.get("runtime", {}).get("dispatch_count", 0) for record in records),
        "source_binding": {
            "validator": {"path": str(validator_path), "sha256": _sha256_file(validator_path)},
            "generator": {
                "path": str(Path(generator.__file__).resolve()),
                "sha256": _sha256_file(Path(generator.__file__).resolve()),
            },
            "imported_sources": [
                {
                    "path": str(path.resolve()),
                    "sha256": _sha256_file(path),
                }
                for path in (
                    _REPO_ROOT / "tools/data/cleaning/complexity.py",
                    _REPO_ROOT / "tools/data/synthesize/csp_dag_method/csp_dag_static_binding.py",
                    _REPO_ROOT / "tools/data/synthesize/select_low_level_kernelbench_canary.py",
                )
            ],
            "selected": {"path": str(selected_path.resolve()), "sha256": _sha256_file(selected_path)},
            "manifest": {"path": str(manifest_path.resolve()), "sha256": _sha256_file(manifest_path)},
            "summary": {"path": str(summary_path.resolve()), "sha256": _sha256_file(summary_path)},
        },
        "records": {"path": str(records_path.resolve()), "sha256": _sha256_file(records_path)},
        "review_only": True,
        "training_approved": False,
    }
    result_path = output_dir / "summary.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=_DEFAULT_RUN)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(validate(args.run_dir, args.device), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
