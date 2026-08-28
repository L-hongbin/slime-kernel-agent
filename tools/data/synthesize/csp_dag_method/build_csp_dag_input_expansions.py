#!/usr/bin/env python3
"""Build shape and dtype children for the retained CSP-DAG dataset.

The CSP-DAG manifest, rather than a source-literal heuristic, is the shape
authority.  Dtype children construct the same random values in FP32 and cast
the input storage before any view is made.
"""

from __future__ import annotations

import argparse
import copy
import difflib
import hashlib
import json
import math
import re
import shutil
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tools.data.cleaning.complexity import extract_complexity_features, extract_operator_signature, feature_dict
from tools.data.synthesize import intervention_semantic_gates
from tools.data.synthesize.augment_prompt_tasks import (
    AUGMENTATION_METADATA_TYPE,
    _normalized_ast_sha256,
    _replace_reference,
)
from tools.data.synthesize.csp_dag_method import csp_dag_static_binding as static_binding
from tools.data.synthesize.csp_dag_method import generate_csp_dag as generator
from tools.data.synthesize.dtype_method.solve_dtype_coverage import _inject_module_conversion

CONTRACT = "csp_dag_shape_dtype_layout_expansion_v1"
SOURCE_BINDING_CONTRACT = "csp_dag_expansion_exact_source_v1"
SHAPE_TARGETS = (32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 393216)
WORKLOAD_RESCUE_TARGETS = (16, 8, 4)
MAX_LOGICAL_INPUT_ELEMENTS = 4 * 1024 * 1024
# The shape lane changes input cardinality while the base generator deliberately
# biases toward long DAGs.  Bound the product of logical input elements and
# typed-DAG nodes, so a 120-node sample cannot turn the 4M input ceiling into
# hundreds of millions of tensor-element operations during H20 validation.
MAX_LOGICAL_NODE_ELEMENTS = 64 * 1024 * 1024
_RUNTIME_SHARD_RE = re.compile(r"^shard_(?P<index>\d{3})_of_(?P<count>\d{3})\.records\.jsonl$")
_SHAPE_RUNTIME_CONTRACT = "open_csp_dag_dataset_validation_v1"
_DTYPE_RUNTIME_CONTRACTS = {
    "dtype_parameter_free_runtime_liveness_v6",
    "dtype_module_state_runtime_liveness_v5",
}
_DTYPE_RESULT_BINDING_VERSION = "dtype_runtime_result_payload_binding_v1"
_DTYPE_FLOAT_COMPARATOR_CONTRACT = "dtype_cast_equivalent_torch_allclose_v1"
_DTYPE_EXACT_COMPARATOR_CONTRACT = "dtype_nonfloating_torch_equal_v1"
_NONFLOATING_DTYPES = frozenset(
    {
        "torch.bool",
        "torch.uint8",
        "torch.uint16",
        "torch.uint32",
        "torch.uint64",
        "torch.int8",
        "torch.int16",
        "torch.int32",
        "torch.int64",
    }
)
_UNSUPPORTED_RUNTIME_OPTIONAL_FIELDS = frozenset({"worker_stderr_tail"})
_DTYPE_UNSUPPORTED_RUNTIME_FIELDS = frozenset(
    {
        "allowlist_sha256",
        "assigned_dtype",
        "binding_version",
        "candidate_row_index",
        "child_uuid",
        "children_sha256",
        "coherence_class",
        "contract_version",
        "duration_seconds",
        "execution_context",
        "launcher_source_sha256",
        "manifest_sha256",
        "memory_guard",
        "parent_uuid",
        "parents_sha256",
        "passed",
        "reason",
        "result_binding_version",
        "result_payload_sha256",
        "runtime_dependencies",
        "status",
        "transformed_factory_count",
        "validation_binding_sha256",
        "validation_config",
        "validator_source_sha256",
    }
)
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DTYPE_LIVENESS_VALIDATOR = _REPO_ROOT / "tools/data/synthesize/dtype_method/validate_dtype_liveness.py"
_LIVENESS_LAUNCHER = _REPO_ROOT / "tools/data/synthesize/csp_dag_method/run_input_expansion_liveness.sh"
_LIVENESS_RUNTIME_DEPENDENCIES = {
    "runtime_validation": _REPO_ROOT / "tools/data/cleaning/runtime_validation.py",
    "serial_source_contract": _REPO_ROOT / "tools/data/synthesize/serial_source_contract.py",
    "repeatability_execution_context": _REPO_ROOT
    / "tools/data/synthesize/csp_dag_method/validate_csp_dag_repeatability.py",
    "train_mode_memory_guard": _REPO_ROOT / "tools/data/synthesize/validate_train_mode_contract.py",
}
_SHAPE_RUNTIME_SOURCES = {
    "validator": _REPO_ROOT / "tools/data/synthesize/csp_dag_method/validate_csp_dag_dataset.py",
    "row_validator": _REPO_ROOT / "tools/data/synthesize/csp_dag_method/validate_csp_dag.py",
    "execution_context": _REPO_ROOT / "tools/data/synthesize/csp_dag_method/validate_csp_dag_repeatability.py",
    "generator": _REPO_ROOT / "tools/data/synthesize/csp_dag_method/generate_csp_dag.py",
}
REQUIRED_H20_SHARDS = 4


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_sha256(value: Any) -> str:
    return _sha256_text(_canonical_json(value))


def _payload_sha256(value: Mapping[str, Any], digest_field: str) -> str:
    return _canonical_sha256({key: item for key, item in value.items() if key != digest_field})


def _comparison_payload_sha256(value: Mapping[str, Any]) -> str:
    return _payload_sha256(value, "comparison_payload_sha256")


def _trial_payload_sha256(value: Mapping[str, Any]) -> str:
    return _payload_sha256(value, "trial_payload_sha256")


def _result_payload_sha256(value: Mapping[str, Any]) -> str:
    return _payload_sha256(value, "result_payload_sha256")


def _validate_unsupported_runtime_schema(record: Mapping[str, Any], child_uuid: str) -> None:
    """Reject unsupported rows that carry success-only or undeclared evidence."""

    expected = _DTYPE_UNSUPPORTED_RUNTIME_FIELDS
    actual = set(record)
    missing = expected - actual
    unexpected = actual - expected - _UNSUPPORTED_RUNTIME_OPTIONAL_FIELDS
    reason = record.get("reason")
    detail = reason.removeprefix("UnsupportedCase:").strip() if isinstance(reason, str) else ""
    stderr = record.get("worker_stderr_tail")
    if (
        record.get("status") != "unsupported"
        or record.get("passed") is not False
        or not isinstance(reason, str)
        or not reason.startswith("UnsupportedCase:")
        or not detail
        or missing
        or unexpected
        or ("worker_stderr_tail" in record and (not isinstance(stderr, str) or not stderr.strip()))
    ):
        raise ValueError(
            f"dtype runtime unsupported status/schema mismatch:{child_uuid}:"
            f"missing={sorted(missing)}:unexpected={sorted(unexpected)}"
        )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _write_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(_canonical_json(value) + "\n" for value in values))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _row_sha256(row: Mapping[str, Any]) -> str:
    return _canonical_sha256(row)


def _artifact_binding(
    parquet_path: Path,
    manifest_path: Path,
    rows: Sequence[Mapping[str, Any]],
    manifests: Sequence[Mapping[str, Any]],
    *,
    stage: str,
) -> dict[str, Any]:
    """Bind one serial source exactly, including its selection summary."""
    if parquet_path.resolve().parent != manifest_path.resolve().parent:
        raise ValueError("source parquet and manifest must share one directory")
    source_root = parquet_path.resolve().parent
    summary_path = source_root / "selection_summary.json"
    if not summary_path.is_file():
        summary_path = source_root / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"source summary is required:{summary_path}")
    summary = json.loads(summary_path.read_text())
    artifacts = summary.get("artifacts") if isinstance(summary, Mapping) else None
    selected = artifacts.get("selected") if isinstance(artifacts, Mapping) else None
    selected_manifest = artifacts.get("selected_manifest") if isinstance(artifacts, Mapping) else None
    if not isinstance(selected, Mapping) or not isinstance(selected_manifest, Mapping):
        raise ValueError("source summary lacks selected artifact bindings")
    if (
        selected.get("path") != str(parquet_path.resolve())
        or selected.get("sha256") != _sha256_file(parquet_path)
        or selected_manifest.get("path") != str(manifest_path.resolve())
        or selected_manifest.get("sha256") != _sha256_file(manifest_path)
    ):
        raise ValueError("source summary does not bind the supplied selected artifacts")
    if summary.get("review_only") is not True or summary.get("training_approved") is not False:
        raise ValueError("source summary lacks review-only governance")
    declared_rows = summary.get("selected_rows", summary.get("rows"))
    if len(rows) != len(manifests) or (declared_rows is not None and declared_rows != len(rows)):
        raise ValueError("source selected row count differs from its summary")
    return {
        "contract": SOURCE_BINDING_CONTRACT,
        "stage": stage,
        "source_summary": {"path": str(summary_path.resolve()), "sha256": _sha256_file(summary_path)},
        "source_artifact": {"path": str(parquet_path.resolve()), "sha256": _sha256_file(parquet_path)},
        "source_manifest": {"path": str(manifest_path.resolve()), "sha256": _sha256_file(manifest_path)},
        "source_rows": len(rows),
    }


def _verify_source_rows(
    parquet_path: Path,
    manifest_path: Path,
    *,
    stage: str,
    expected_manifest: tuple[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = pq.read_table(parquet_path).to_pylist()
    manifests = _read_jsonl(manifest_path)
    if not rows or len(rows) != len(manifests):
        raise ValueError("source rows and manifests differ or are empty")
    binding = _artifact_binding(parquet_path, manifest_path, rows, manifests, stage=stage)
    for index, (row, manifest) in enumerate(zip(rows, manifests, strict=True)):
        uuid, code = _identity(row)
        if manifest.get("uuid", manifest.get("child_uuid")) != uuid:
            raise ValueError(f"source UUID mismatch:{stage}:{index}")
        expected_hash = manifest.get("reference_sha256", manifest.get("child_reference_sha256"))
        expected_ast = manifest.get("normalized_ast_sha256", manifest.get("child_normalized_ast_sha256"))
        if expected_hash != _sha256_text(code) or expected_ast != _normalized_ast_sha256(code):
            raise ValueError(f"source code hash mismatch:{stage}:{index}")
        if manifest.get(expected_manifest[0]) != expected_manifest[1]:
            raise ValueError(f"source manifest contract mismatch:{stage}:{index}")
        if manifest.get("training_approved") is not False:
            raise ValueError(f"source manifest governance mismatch:{stage}:{index}")
        if stage in {"base", "shape"}:
            graph_checks, nodes, rank, shape = static_binding.typed_graph_context(manifest)
            lowering_checks = static_binding.source_lowering_checks(code, manifest, nodes, rank, shape)
            failed = sorted(name for name, passed in {**graph_checks, **lowering_checks}.items() if passed is not True)
            if failed:
                raise ValueError(f"source static graph/lowering binding mismatch:{stage}:{index}:{','.join(failed)}")
    return rows, manifests, binding


def _nested(value: Mapping[str, Any], path: str, default: Any = None) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _extend_schema(schema: pa.Schema) -> pa.Schema:
    index = schema.get_field_index("extra_info")
    if index < 0 or not pa.types.is_struct(schema.field(index).type):
        raise ValueError("input schema has no extra_info struct")
    extra = schema.field(index)
    augmentation_index = extra.type.get_field_index("augmentation")
    if augmentation_index >= 0:
        if extra.type.field(augmentation_index).type != AUGMENTATION_METADATA_TYPE:
            raise ValueError("input augmentation metadata has an incompatible type")
        return schema
    fields = list(schema)
    fields[index] = pa.field(
        "extra_info",
        pa.struct([*extra.type, pa.field("augmentation", AUGMENTATION_METADATA_TYPE)]),
        nullable=extra.nullable,
        metadata=extra.metadata,
    )
    return pa.schema(fields, metadata=schema.metadata)


def _with_schema_metadata(schema: pa.Schema, stage: str) -> pa.Schema:
    return schema.with_metadata(
        {
            **(schema.metadata or {}),
            b"csp_dag.expansion_contract": CONTRACT.encode(),
            b"csp_dag.expansion_stage": stage.encode(),
            b"csp_dag.review_only": b"true",
            b"csp_dag.training_approved": b"false",
        }
    )


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]], schema: pa.Schema, stage: str) -> None:
    table = pa.Table.from_pylist(list(rows), schema=_with_schema_metadata(schema, stage))
    pq.write_table(table, path, compression="zstd")


def _bundle_builder(output_dir: Path) -> dict[str, Any]:
    bundle = output_dir / "source_bundle"
    bundle.mkdir(exist_ok=True)
    target = bundle / Path(__file__).name
    shutil.copy2(Path(__file__).resolve(), target)
    dependencies = {
        "generator": generator,
        "static_binding": static_binding,
        "intervention_semantic_gates": intervention_semantic_gates,
        "solve_dtype_coverage": __import__("tools.data.synthesize.dtype_method.solve_dtype_coverage", fromlist=["*"]),
        "augment_prompt_tasks": __import__("tools.data.synthesize.augment_prompt_tasks", fromlist=["*"]),
        "complexity": __import__("tools.data.cleaning.complexity", fromlist=["*"]),
    }
    copied: dict[str, dict[str, str]] = {}
    for name, module in dependencies.items():
        source = Path(module.__file__).resolve()
        destination = bundle / f"{name}_{source.name}"
        shutil.copy2(source, destination)
        copied[name] = {"path": str(destination.resolve()), "sha256": _sha256_file(destination)}
    return {"builder": {"path": str(target.resolve()), "sha256": _sha256_file(target)}, "dependencies": copied}


def _identity(row: Mapping[str, Any]) -> tuple[str, str]:
    uuid = _nested(row, "extra_info.uuid")
    code = _nested(row, "reward_model.ground_truth")
    if not isinstance(uuid, str) or not isinstance(code, str):
        raise ValueError("row lacks UUID or reference")
    return uuid, code


def _mutate_row(
    parent: Mapping[str, Any],
    *,
    child_uuid: str,
    child_code: str,
    parent_uuid: str,
    runtime_status: str,
    governance_status: str,
    augmentation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    child = copy.deepcopy(dict(parent))
    _, parent_code = _identity(parent)
    child["reward_model"] = dict(child["reward_model"])
    child["reward_model"]["ground_truth"] = child_code
    child["prompt"] = _replace_reference(child.get("prompt"), parent_code, child_code, required=True)
    extra = dict(child["extra_info"])
    if extra.get("original_prompt") is not None:
        extra["original_prompt"] = _replace_reference(
            extra["original_prompt"], parent_code, child_code, required=False
        )
    extra["uuid"] = child_uuid
    v4 = dict(extra.get("v4") or {})
    v4.update(
        {
            "parent_uuid": parent_uuid,
            "reference_sha256": _sha256_text(child_code),
            "normalized_ast_sha256": _normalized_ast_sha256(child_code),
            "included_in_review_train": False,
            "runtime_validation_status": runtime_status,
            "governance_status": governance_status,
        }
    )
    extra["v4"] = v4
    if augmentation is not None:
        extra["augmentation"] = dict(augmentation)
    child["extra_info"] = extra
    return child


def _review(path: Path, triples: Sequence[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]]) -> None:
    lines = ["# CSP-DAG input expansion review", ""]
    for parent, child, manifest in triples[:12]:
        parent_uuid, parent_code = _identity(parent)
        child_uuid, child_code = _identity(child)
        intervention = manifest.get("primary_intervention")
        if intervention is None and isinstance(manifest.get("shape_intervention"), Mapping):
            intervention = "shape"
        lines.extend(
            [
                f"## {intervention}: {parent_uuid} -> {child_uuid}",
                "",
                "```diff",
                *difflib.unified_diff(
                    parent_code.splitlines(),
                    child_code.splitlines(),
                    fromfile="parent.py",
                    tofile="child.py",
                    lineterm="",
                ),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n")


def _shape_groups(ndim: int, equalities: Sequence[Sequence[int]]) -> list[tuple[int, ...]]:
    parent = list(range(ndim))

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    for equality in equalities:
        if len(equality) != 2:
            raise ValueError(f"invalid shape equality:{equality}")
        union(int(equality[0]), int(equality[1]))
    groups: dict[int, list[int]] = {}
    for axis in range(1, ndim):
        groups.setdefault(find(axis), []).append(axis)
    return [tuple(axes) for axes in groups.values()]


def _choose_shape(manifest: Mapping[str, Any]) -> tuple[tuple[int, ...], dict[str, Any]]:
    graph = manifest["graph"]
    proof = graph["shape_csp"]
    old = tuple(int(value) for value in proof["witness"])
    families = set(manifest["actual_low_level_families"])
    groups = _shape_groups(len(old), proof.get("equalities", []))
    if not groups:
        raise ValueError("shape expansion needs one non-batch axis")

    risky_last = bool(families & {"matmul_linear", "normalization"})
    risky_channel = bool(families & {"conv", "pooling"})

    def group_risk(group: tuple[int, ...]) -> tuple[int, int, tuple[int, ...]]:
        return (
            int(risky_channel and 1 in group) + int(risky_last and len(old) - 1 in group),
            len(group),
            group,
        )

    ordered_groups = sorted(groups, key=group_risk)
    digest = _sha256_text(f"shape:{manifest['uuid']}:{manifest['reference_sha256']}")

    family_cap = 393216
    if "scaled_dot_product_attention" in families:
        family_cap = 256
    elif "matmul_linear" in families:
        family_cap = 2048
    elif families & {"conv", "pooling"}:
        family_cap = 4096
    # Cover the full KernelBench formal-value range without imitating its
    # histogram.  Most children stay in a practical high-complexity runtime
    # range; the tail values are sparse coverage witnesses.
    percentile = int(digest[:16], 16) % 100
    if percentile < 60:
        requested_index = int(digest[16:32], 16) % 5
    elif percentile < 90:
        requested_index = 5 + int(digest[16:32], 16) % 3
    elif percentile < 98:
        requested_index = 8 + int(digest[16:32], 16) % 2
    else:
        requested_index = 10 + int(digest[16:32], 16) % 2
    target_order = [
        *SHAPE_TARGETS[requested_index:],
        *SHAPE_TARGETS[:requested_index],
    ]
    loss_multiplier = 2 if "loss_distance" in families else 1
    node_count = int(graph["node_count"])
    base_logical_elements = math.prod(old) * loss_multiplier
    node_element_cap = MAX_LOGICAL_NODE_ELEMENTS // max(1, node_count)
    workload_cap = max(base_logical_elements, node_element_cap)
    logical_element_cap = max(
        base_logical_elements,
        min(MAX_LOGICAL_INPUT_ELEMENTS, workload_cap),
    )
    choices: list[tuple[int, tuple[int, ...], tuple[int, ...], str]] = []
    for group in ordered_groups:
        for target in target_order:
            if target <= max(old[axis] for axis in group) or target > family_cap:
                continue
            shape = list(old)
            for axis in group:
                shape[axis] = target
            if math.prod(shape) * loss_multiplier > logical_element_cap:
                continue
            if math.prod(shape) * loss_multiplier > node_element_cap:
                continue
            choices.append((target, group, tuple(shape), "catalog_growth"))
    if not choices:
        # A few already-large, long graphs have no growth target under the
        # node-work budget.  Keep a shape intervention by changing one legal
        # non-batch equality group to a smaller formal value instead of
        # silently restoring a 4M-by-120 workload or dropping the row.
        for group in ordered_groups:
            for target in WORKLOAD_RESCUE_TARGETS:
                if target > family_cap or all(target == old[axis] for axis in group):
                    continue
                shape = list(old)
                for axis in group:
                    shape[axis] = target
                if (
                    math.prod(shape) * loss_multiplier > logical_element_cap
                    or math.prod(shape) * loss_multiplier > node_element_cap
                ):
                    continue
                choices.append((target, group, tuple(shape), "workload_rescue_downscale"))
    if not choices:
        raise ValueError(f"no bounded shape target for {manifest['uuid']}")
    target, axes, shape, selection_mode = choices[0]
    return shape, {
        "assignment_sha256": digest,
        "changed_axes": list(axes),
        "old_shape": list(old),
        "new_shape": list(shape),
        "assigned_target_value": target,
        "target_catalog": list(SHAPE_TARGETS),
        "workload_rescue_catalog": list(WORKLOAD_RESCUE_TARGETS),
        "selection_mode": selection_mode,
        "logical_input_elements": math.prod(shape) * loss_multiplier,
        "logical_input_element_cap": logical_element_cap,
        "node_count": node_count,
        "logical_node_elements": math.prod(shape) * loss_multiplier * node_count,
        "logical_node_element_cap": MAX_LOGICAL_NODE_ELEMENTS,
    }


def build_shape(parent_parquet: Path, parent_manifest: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    rows, manifests, parent_binding = _verify_source_rows(
        parent_parquet,
        parent_manifest,
        stage="base",
        expected_manifest=("contract", generator.CONTRACT),
    )
    children: list[dict[str, Any]] = []
    child_manifests: list[dict[str, Any]] = []
    review: list[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]] = []
    targets: dict[str, int] = {}
    for position, (row, manifest) in enumerate(zip(rows, manifests, strict=True)):
        parent_uuid, parent_code = _identity(row)
        if manifest["uuid"] != parent_uuid or manifest["reference_sha256"] != _sha256_text(parent_code):
            raise ValueError(f"shape parent binding mismatch:{position}")
        shape, intervention = _choose_shape(manifest)
        nodes = [generator.Node(**node) for node in manifest["graph"]["typed_graph"]["nodes"]]
        edges = [tuple(edge) for edge in manifest["graph"]["typed_graph"]["edges"]]
        old_proof = manifest["graph"]["shape_csp"]
        input_proof = {
            "variables": [f"d{axis}" for axis in range(len(shape))],
            "bounds": {f"d{axis}": [value, value] for axis, value in enumerate(shape)},
            "divisibility": {f"d{axis}": 4 for axis in range(1, len(shape))},
            "equalities": copy.deepcopy(old_proof.get("equalities", [])),
            "status": "sat",
            "witness": list(shape),
        }
        shape_proof = generator._prove_graph_shapes(nodes, shape, input_proof)
        graph = generator._graph_manifest(nodes, edges, len(shape), shape_proof)
        child_code = generator._render_code(
            nodes=nodes,
            edges=edges,
            shape=shape,
            shape_proof=shape_proof,
            input_mode=manifest["input_mode"],
            minimum_lines=int(manifest["requested_minimum_source_lines"]),
        )
        signature = list(extract_operator_signature(child_code))
        if signature != manifest["operator_signature"]:
            raise ValueError(f"shape changed operator signature:{parent_uuid}")
        child_hash = _sha256_text(child_code)
        child_uuid = (
            "shape_" + _sha256_text(f"{CONTRACT}:{parent_uuid}:{child_hash}:{_canonical_sha256(intervention)}")[:24]
        )
        child = _mutate_row(
            row,
            child_uuid=child_uuid,
            child_code=child_code,
            parent_uuid=parent_uuid,
            runtime_status="shape_h20_pending",
            governance_status="shape_intervention_review_only",
            augmentation=None,
        )
        child_manifest = copy.deepcopy(manifest)
        child_complexity = feature_dict(extract_complexity_features(child_code))
        child_code_metrics = {
            "physical_line_count": int(child_complexity["source_line_count"]),
            "code_only_physical_line_count": generator._code_only_physical_line_count(child_code),
            "comment_only_physical_line_count": sum(
                bool(line.strip()) and line.lstrip().startswith("#") for line in child_code.splitlines()
            ),
            "source_format_expansion_lines": 0,
        }
        child_manifest.update(
            {
                "candidate_index": position,
                "uuid": child_uuid,
                "parent_uuid": parent_uuid,
                "contract": CONTRACT,
                "graph": graph,
                "reference_sha256": child_hash,
                "normalized_ast_sha256": generator._normalized_ast_hash(child_code),
                "operator_signature": signature,
                "complexity": child_complexity,
                "code_metrics": child_code_metrics,
                "materialization_status": "review_only",
                "training_approved": False,
                "shape_intervention": intervention,
                "csp_dag_source_binding": {
                    **parent_binding,
                    "source_row_index": position,
                    "source_row_sha256": _row_sha256(row),
                    "source_manifest_row_sha256": _row_sha256(manifest),
                },
            }
        )
        children.append(child)
        child_manifests.append(child_manifest)
        targets[str(intervention["assigned_target_value"])] = (
            targets.get(str(intervention["assigned_target_value"]), 0) + 1
        )
        if len(review) < 12:
            review.append((row, child, child_manifest))
    schema = pq.ParquetFile(parent_parquet).schema_arrow
    selected_path = output_dir / "candidates.parquet"
    manifest_path = output_dir / "manifest.jsonl"
    _write_rows(selected_path, children, schema, "shape_candidates")
    _write_jsonl(manifest_path, child_manifests)
    _review(output_dir / "review_samples.md", review)
    bundled_builder = _bundle_builder(output_dir)
    summary = {
        "contract": CONTRACT,
        "stage": "shape",
        "parent_rows": len(rows),
        "candidate_rows": len(children),
        "target_value_counts": dict(sorted(targets.items(), key=lambda item: int(item[0]))),
        "source_binding": {
            "parent": parent_binding,
            **bundled_builder,
        },
        "artifacts": {
            "candidates": {"path": str(selected_path.resolve()), "sha256": _sha256_file(selected_path)},
            "manifest": {"path": str(manifest_path.resolve()), "sha256": _sha256_file(manifest_path)},
        },
        "review_only": True,
        "training_approved": False,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def _augmentation(
    *,
    kind: str,
    parent_uuid: str,
    child_uuid: str,
    source_sha256: str,
    source_row_index: int,
    parent_code: str,
    child_code: str,
    intervention_sha256: str,
    coverage_cell: str,
    factory_count: int,
    dtype_before: str | None,
    dtype_after: str | None,
    input_bytes_before: int | None,
    input_bytes_after: int | None,
) -> dict[str, Any]:
    return {
        "contract_version": CONTRACT,
        "generator_version": CONTRACT,
        "parent_uuid": parent_uuid,
        "child_uuid": child_uuid,
        "source_artifact_sha256": source_sha256,
        "source_row_index": source_row_index,
        "parent_reference_sha256": _sha256_text(parent_code),
        "parent_normalized_ast_sha256": _normalized_ast_sha256(parent_code),
        "child_reference_sha256": _sha256_text(child_code),
        "child_normalized_ast_sha256": _normalized_ast_sha256(child_code),
        "intervention_id": f"{kind}_{intervention_sha256[:20]}",
        "intervention_sha256": intervention_sha256,
        "intervention_kind": kind,
        "coverage_cell": coverage_cell,
        "shape_scale": None,
        "value_family_before": None,
        "value_family_after": None,
        "dtype_before": dtype_before,
        "dtype_after": dtype_after,
        "layout_before": None,
        "layout_after": None,
        "shape_dimension_name": None,
        "shape_dimension_before": None,
        "shape_dimension_after": None,
        "factory_count": factory_count,
        "input_bytes_before": input_bytes_before,
        "input_bytes_after": input_bytes_after,
        "estimated_peak_bytes": input_bytes_after,
        "memory_budget_bytes": 64 * 1024**3,
        "working_set_multiplier": 1.0,
        "memory_estimator_version": "csp_dag_exact_input_bytes_v1",
        "shape_proof": "shape inherited from bound CSP-DAG shape child",
        "compatibility_proof": "CSP-DAG controlled lowering plus paired target-GPU validation",
        "validation_status": f"{kind}_paired_h20_pending",
    }


def _replace_get_inputs_for_dtype(code: str, target: str) -> str:
    lines = code.splitlines()
    in_inputs = False
    replacements = 0
    for index, line in enumerate(lines):
        if line == "def get_inputs():":
            in_inputs = True
            continue
        if not in_inputs:
            continue
        stripped = line.strip()
        if stripped.startswith("base = torch.randn("):
            lines[index] = line + f".to(dtype=torch.{target})"
            replacements += 1
        elif stripped.startswith("x = torch.randn("):
            lines[index] = line + f".to(dtype=torch.{target})"
            replacements += 1
        elif stripped.startswith("x = torch.rand(") and stripped.endswith(" + 0.125"):
            prefix = line[: len(line) - len(line.lstrip())]
            expression = stripped[len("x = ") :]
            lines[index] = prefix + f"x = ({expression}).to(dtype=torch.{target})"
            replacements += 1
        elif stripped == "target = torch.randn_like(x)":
            prefix = line[: len(line) - len(line.lstrip())]
            lines[index] = prefix + f"target = torch.randn_like(x, dtype=torch.float32).to(dtype=torch.{target})"
            replacements += 1
    if replacements not in {1, 2}:
        raise ValueError(f"unexpected CSP-DAG dtype replacement count:{replacements}")
    child = "\n".join(lines).rstrip() + "\n"
    return _inject_module_conversion(child, "Model", target)


def build_dtype(shape_parquet: Path, shape_manifest: Path, output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    rows, shape_manifests, shape_binding = _verify_source_rows(
        shape_parquet,
        shape_manifest,
        stage="shape",
        expected_manifest=("contract", CONTRACT),
    )
    source_sha = _sha256_file(shape_parquet)
    parents: list[dict[str, Any]] = []
    children: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    review: list[tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]] = []
    target_counts: dict[str, int] = {}
    rejected: dict[str, int] = {}
    for source_index, (row, shape_manifest_row) in enumerate(zip(rows, shape_manifests, strict=True)):
        parent_uuid, parent_code = _identity(row)
        shape = tuple(shape_manifest_row["graph"]["shape_csp"]["witness"])
        target = ("float16", "bfloat16")[int(_sha256_text(f"dtype:{parent_uuid}")[:16], 16) % 2]
        tensor_count = 2 if "loss_distance" in shape_manifest_row["actual_low_level_families"] else 1
        verdict = intervention_semantic_gates.evaluate_semantic_gate(
            "dtype",
            parent_code=parent_code,
            child_code=parent_code,
            manifest={
                "assigned_target": target,
                "factory_specs_before": [
                    {"factory": "logical_csp_input", "shape": list(shape), "dtype": "torch.float32"}
                    for _ in range(tensor_count)
                ],
            },
        )
        if verdict["status"] == "rejected":
            key = ",".join(verdict["reasons"])
            rejected[key] = rejected.get(key, 0) + 1
            decisions.append(
                {
                    "source_row_index": source_index,
                    "parent_uuid": parent_uuid,
                    "eligible": False,
                    "semantic_gate": verdict,
                }
            )
            continue
        child_code = _replace_get_inputs_for_dtype(parent_code, target)
        intervention = {
            "primary_intervention": "dtype",
            "coherence_class": "module_state",
            "dtype_before": "float32",
            "dtype_after": target,
            "factory_count": tensor_count,
            "model_parameter_policy": "registered_fp32_parent_exact_cast",
            "model_buffer_policy": "registered_state_exact_cast_or_equal",
            "explicit_cast_policy": "input storage cast before view; one base Module conversion",
            "logical_forward_dtype_policy": "all_floating_vN_and_final_outputs_match_assigned_dtype_v1",
            "internal_aten_non_target_floating_outputs": "diagnostic_only",
            "internal_aten_complex_outputs": "reject",
        }
        intervention_sha = _canonical_sha256(intervention)
        child_uuid = (
            "dtype_" + _sha256_text(f"{CONTRACT}:{parent_uuid}:{_sha256_text(parent_code)}:{intervention_sha}")[:24]
        )
        input_before = math.prod(shape) * 4 * tensor_count
        input_after = math.prod(shape) * 2 * tensor_count
        augmentation = _augmentation(
            kind="dtype",
            parent_uuid=parent_uuid,
            child_uuid=child_uuid,
            source_sha256=source_sha,
            source_row_index=source_index,
            parent_code=parent_code,
            child_code=child_code,
            intervention_sha256=intervention_sha,
            coverage_cell=f"dtype_module_state_{target}",
            factory_count=tensor_count,
            dtype_before="float32",
            dtype_after=target,
            input_bytes_before=input_before,
            input_bytes_after=input_after,
        )
        child = _mutate_row(
            row,
            child_uuid=child_uuid,
            child_code=child_code,
            parent_uuid=parent_uuid,
            runtime_status="dtype_module_state_paired_runtime_pending",
            governance_status="dtype_intervention_review_only",
            augmentation=augmentation,
        )
        manifest = {
            "manifest_contract_version": "dtype_lane_manifest_v1",
            "candidate_row_index": len(children),
            "source_artifact_path": str(shape_parquet.resolve()),
            "source_artifact_sha256": source_sha,
            "source_row_index": source_index,
            "csp_dag_source_binding": {
                **shape_binding,
                "source_row_index": source_index,
                "source_row_sha256": _row_sha256(row),
                "source_manifest_row_sha256": _row_sha256(shape_manifest_row),
            },
            "parent_uuid": parent_uuid,
            "child_uuid": child_uuid,
            "parent_reference_sha256": _sha256_text(parent_code),
            "parent_normalized_ast_sha256": _normalized_ast_sha256(parent_code),
            "child_reference_sha256": _sha256_text(child_code),
            "child_normalized_ast_sha256": _normalized_ast_sha256(child_code),
            "primary_intervention": "dtype",
            "assigned_target": target,
            "realized_intervention": intervention,
            "coherence_class": "module_state",
            "reject_reason": None,
            "generator_contract_version": CONTRACT,
            "generator_version": CONTRACT,
            "generator_source_sha256": _sha256_file(Path(__file__).resolve()),
            "dependency_source_sha256": _sha256_file(Path(intervention_semantic_gates.__file__).resolve()),
            "source_family": "csp_dag",
            "operator_bucket": str(_nested(row, "extra_info.v4.operator_bucket", "unknown")),
            "operator_count": len(shape_manifest_row["operator_signature"]),
            "source_graph_node_count": len(shape_manifest_row["graph"]["typed_graph"]["nodes"]),
            "source_graph_operator_count": len(shape_manifest_row["operator_signature"]),
            "shape_changed": False,
            "value_changed": False,
            "dtype_changed": True,
            "layout_changed": False,
            "model_changed": True,
            "get_init_inputs_changed": False,
            "rng_consumption_status": "runtime_pending",
            "factory_count": tensor_count,
            "transformed_factory_count": tensor_count,
            "input_bytes_before": input_before,
            "input_bytes_after": input_after,
            "estimated_peak_bytes": input_after,
            "static_memory_budget_bytes": 64 * 1024**3,
            "runtime_memory_budget_bytes": 64 * 1024**3,
            "factory_specs_before": [
                {"factory": "logical_csp_input", "shape": list(shape), "dtype": "torch.float32"}
                for _ in range(tensor_count)
            ],
            "model_parameter_count": None,
            "model_buffer_count": None,
            "explicit_model_casts": [f"torch.nn.Module.to(self, dtype=torch.{target})"],
            "runtime_promotion_status": "pending",
            "static_status": "passed",
            "parent_runtime_status": "pending",
            "child_runtime_status": "pending",
            "liveness_status": "pending",
            "materialization_status": "review_only",
            "lineage": "csp_dag_shape_dtype_child",
            "semantic_gate": verdict,
            "logical_shape": list(shape),
            "parent_input_mode": shape_manifest_row["input_mode"],
            "training_approved": False,
        }
        parent = copy.deepcopy(row)
        parent["extra_info"] = dict(parent["extra_info"])
        parent["extra_info"]["augmentation"] = None
        parents.append(parent)
        children.append(child)
        manifests.append(manifest)
        target_counts[target] = target_counts.get(target, 0) + 1
        decisions.append(
            {
                "source_row_index": source_index,
                "parent_uuid": parent_uuid,
                "eligible": True,
                "candidate_row_index": len(children) - 1,
                "semantic_gate": verdict,
            }
        )
        if len(review) < 12:
            review.append((parent, child, manifest))
    schema = _extend_schema(pq.ParquetFile(shape_parquet).schema_arrow)
    _write_rows(output_dir / "parents.parquet", parents, schema, "dtype_parents")
    _write_rows(output_dir / "candidates.parquet", children, schema, "dtype_candidates")
    _write_jsonl(output_dir / "manifest.jsonl", manifests)
    _write_jsonl(output_dir / "decisions.jsonl", decisions)
    _review(output_dir / "review_samples.md", review)
    bundled_builder = _bundle_builder(output_dir)
    summary = {
        "contract": CONTRACT,
        "stage": "dtype",
        "source_rows": len(rows),
        "eligible_rows": len(children),
        "candidate_rows": len(children),
        "target_counts": dict(sorted(target_counts.items())),
        "semantic_rejection_counts": dict(sorted(rejected.items())),
        "source_binding": {
            "source": shape_binding,
            **bundled_builder,
        },
        "artifacts": {
            name: {"path": str((output_dir / filename).resolve()), "sha256": _sha256_file(output_dir / filename)}
            for name, filename in (
                ("parents", "parents.parquet"),
                ("candidates", "candidates.parquet"),
                ("manifest", "manifest.jsonl"),
            )
        },
        "review_only": True,
        "training_approved": False,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def _exact_artifact(binding: Mapping[str, Any], name: str, path: Path, context: str) -> None:
    declared = binding.get(name)
    if not isinstance(declared, Mapping):
        raise ValueError(f"{context} lacks {name} binding")
    if declared.get("path") != str(path.resolve()) or declared.get("sha256") != _sha256_file(path):
        raise ValueError(f"{context} {name} binding mismatch")


def _runtime_shards(runtime_dir: Path, context: str) -> list[tuple[int, int, Path]]:
    paths = sorted(runtime_dir.glob("*.records.jsonl"))
    all_jsonl = sorted(runtime_dir.rglob("*.jsonl"))
    if not paths:
        raise ValueError(f"no {context} runtime JSONL files in {runtime_dir}")
    if set(paths) != set(all_jsonl):
        raise ValueError(f"{context} runtime directory contains an unknown JSONL artifact")
    shards: list[tuple[int, int, Path]] = []
    for path in paths:
        match = _RUNTIME_SHARD_RE.fullmatch(path.name)
        if match is None:
            raise ValueError(f"invalid {context} runtime shard name:{path.name}")
        shards.append((int(match["index"]), int(match["count"]), path))
    counts = {count for _, count, _ in shards}
    if len(counts) != 1 or next(iter(counts)) <= 0:
        raise ValueError(f"{context} runtime shard count is inconsistent")
    count = next(iter(counts))
    if sorted(index for index, _, _ in shards) != list(range(count)):
        raise ValueError(f"{context} runtime shards are missing or duplicated")
    return shards


def _expected_liveness_source_hashes() -> dict[str, Any]:
    """Return the production sources invoked by the checked-in launcher.

    Runtime records only contain hashes, so accepting a self-consistent record
    without comparing it to these files would allow a different validator and
    launcher to assert the same private binding digest.
    """
    validator_path = _DTYPE_LIVENESS_VALIDATOR
    return {
        "validator_source_sha256": _sha256_file(validator_path),
        "launcher_source_sha256": _sha256_file(_LIVENESS_LAUNCHER),
        "runtime_dependencies": {
            name: {
                "path": str(path.relative_to(_REPO_ROOT)),
                "sha256": _sha256_file(path),
            }
            for name, path in _LIVENESS_RUNTIME_DEPENDENCIES.items()
        },
    }


def _validate_lane_builder_binding(lane_summary: Mapping[str, Any], lane_dir: Path, stage: str) -> None:
    """Bind a lane's copied builder to the current production builder.

    The copied source bundle is retained for review, but its hash must still
    equal this exact builder.  A source bundle from another implementation is
    therefore not accepted merely because the lane summary names it.
    """
    source_binding = lane_summary.get("source_binding")
    if not isinstance(source_binding, Mapping):
        raise ValueError(f"{stage} summary has no source binding")
    bundle = source_binding.get("builder")
    expected_path = lane_dir / "source_bundle" / Path(__file__).name
    if not isinstance(bundle, Mapping):
        raise ValueError(f"{stage} summary has no builder source bundle")
    if bundle.get("path") != str(expected_path.resolve()):
        raise ValueError(f"{stage} summary builder source bundle path mismatch")
    if bundle.get("sha256") != _sha256_file(Path(__file__).resolve()):
        raise ValueError(f"{stage} summary builder source bundle hash mismatch")
    if not expected_path.is_file() or _sha256_file(expected_path) != bundle["sha256"]:
        raise ValueError(f"{stage} summary builder source bundle file mismatch")
    declared_dependencies = source_binding.get("dependencies")
    expected_modules = {
        "generator": generator,
        "static_binding": static_binding,
        "intervention_semantic_gates": intervention_semantic_gates,
        "solve_dtype_coverage": __import__("tools.data.synthesize.dtype_method.solve_dtype_coverage", fromlist=["*"]),
        "augment_prompt_tasks": __import__("tools.data.synthesize.augment_prompt_tasks", fromlist=["*"]),
        "complexity": __import__("tools.data.cleaning.complexity", fromlist=["*"]),
    }
    if not isinstance(declared_dependencies, Mapping) or set(declared_dependencies) != set(expected_modules):
        raise ValueError(f"{stage} summary dependency source bundle mismatch")
    for name, module in expected_modules.items():
        source = Path(module.__file__).resolve()
        copied = expected_path.parent / f"{name}_{source.name}"
        expected = {"path": str(copied.resolve()), "sha256": _sha256_file(source)}
        if (
            declared_dependencies.get(name) != expected
            or not copied.is_file()
            or _sha256_file(copied) != expected["sha256"]
        ):
            raise ValueError(f"{stage} summary dependency source bundle mismatch:{name}")


def _shape_required_checks(row: Mapping[str, Any], manifest: Mapping[str, Any]) -> frozenset[str]:
    """Compute the exact check names emitted by the current shape row validator."""
    # Delayed import avoids the dataset validator's import of this builder at
    # module initialization while still pinning the check schema to production.
    from tools.data.synthesize.csp_dag_method import validate_csp_dag as row_validator

    _, code = _identity(row)
    graph_checks, nodes, rank, shape = static_binding.typed_graph_context(manifest)
    lowering_checks = static_binding.source_lowering_checks(code, manifest, nodes, rank, shape)
    runtime_graph_checks = row_validator._graph_check(manifest)
    return frozenset(
        {
            "uuid_bound",
            "reference_hash_bound",
            "ast_hash_bound",
            "governance",
            "no_template_identifier",
            "no_repeated_provenance_padding",
            "signature_bound",
            "families_bound",
            "complexity_bound",
            "multiclass_bound",
            "multiclass_helper_reachable",
            "no_forbidden_composites",
            "normalization_variant_lowering_bound",
            "runtime_tensor_output",
            "runtime_finite",
            "runtime_dispatch_nonempty",
            *graph_checks.keys(),
            *lowering_checks.keys(),
            *(f"graph_{name}" for name in runtime_graph_checks),
        }
    )


def _validate_dtype_output_comparison(
    output: Mapping[str, Any], *, target: str, tolerance: float, child_uuid: str
) -> None:
    """Recompute every serialized invariant available without runtime tensors."""

    if output.get("comparison_payload_sha256") != _comparison_payload_sha256(output):
        raise ValueError(f"dtype runtime comparison payload binding mismatch:{child_uuid}")
    shape = output.get("shape")
    elements = output.get("elements")
    if (
        not isinstance(output.get("path"), str)
        or not output["path"]
        or not isinstance(shape, list)
        or any(type(value) is not int or value < 0 for value in shape)
        or type(elements) is not int
        or elements < 0
        or elements != math.prod(shape)
    ):
        raise ValueError(f"dtype runtime comparison shape/path mismatch:{child_uuid}")
    numeric_names = (
        "maximum_absolute_difference",
        "mean_absolute_difference",
        "maximum_tolerance_ratio",
        "maximum_tolerance_excess",
        "maximum_reference_absolute_value",
        "maximum_allowed_absolute_difference",
    )
    if any(
        not isinstance(output.get(name), (int, float))
        or isinstance(output.get(name), bool)
        or not math.isfinite(float(output[name]))
        for name in numeric_names
    ):
        raise ValueError(f"dtype runtime comparison has non-finite metric:{child_uuid}")
    maximum = float(output["maximum_absolute_difference"])
    mean = float(output["mean_absolute_difference"])
    ratio = float(output["maximum_tolerance_ratio"])
    excess = float(output["maximum_tolerance_excess"])
    reference_max = float(output["maximum_reference_absolute_value"])
    allowed_max = float(output["maximum_allowed_absolute_difference"])
    if (
        not 0.0 <= mean <= maximum
        or not 0.0 <= ratio <= 1.0
        or reference_max < 0.0
        or excess > 0.0
        or output.get("violating_elements") != 0
        or output.get("within_tolerance") is not True
        or output.get("allclose_passed") is not True
    ):
        raise ValueError(f"dtype runtime comparison verdict/metric mismatch:{child_uuid}")
    comparator = output.get("comparator")
    floating = output.get("parent_dtype") == "torch.float32" and output.get("child_dtype") == f"torch.{target}"
    nonfloating = (
        output.get("parent_dtype") == output.get("child_dtype") and output.get("parent_dtype") in _NONFLOATING_DTYPES
    )
    if floating:
        expected_comparator = {
            "contract": _DTYPE_FLOAT_COMPARATOR_CONTRACT,
            "implementation": "torch.allclose",
            "rtol": tolerance,
            "atol": tolerance,
            "equal_nan": False,
            "relative_reference_operand": "child",
            "elementwise_bound": "abs(parent-child) <= atol + rtol * abs(child)",
        }
        expected_allowed = tolerance + tolerance * reference_max if elements else 0.0
        if (
            output.get("rtol") != tolerance
            or output.get("atol") != tolerance
            or comparator != expected_comparator
            or not math.isclose(allowed_max, expected_allowed, rel_tol=0.0, abs_tol=1e-12)
            or maximum > allowed_max
        ):
            raise ValueError(f"dtype runtime floating comparator mismatch:{child_uuid}")
    elif nonfloating:
        expected_comparator = {
            "contract": _DTYPE_EXACT_COMPARATOR_CONTRACT,
            "implementation": "torch.equal",
            "rtol": 0.0,
            "atol": 0.0,
            "equal_nan": False,
            "relative_reference_operand": None,
            "elementwise_bound": "exact equality",
        }
        if (
            output.get("rtol") != 0.0
            or output.get("atol") != 0.0
            or comparator != expected_comparator
            or any(value != 0.0 for value in (maximum, mean, ratio, excess, reference_max, allowed_max))
        ):
            raise ValueError(f"dtype runtime exact comparator mismatch:{child_uuid}")
    else:
        raise ValueError(f"dtype runtime comparison dtype mismatch:{child_uuid}")


def _validate_liveness_semantic_evidence(
    record: Mapping[str, Any], child_uuid: str, manifest: Mapping[str, Any]
) -> None:
    """Fail closed unless bound semantic evidence has the production schema."""
    if record.get("status") != "passed" or record.get("reason") is not None:
        raise ValueError(f"dtype runtime passed status/reason mismatch:{child_uuid}")
    if (
        not isinstance(record.get("gpu"), Mapping)
        or not str(record["gpu"].get("device", "")).startswith("cuda:")
        or "h20" not in str(record["gpu"].get("name", "")).lower()
    ):
        raise ValueError(f"dtype runtime GPU evidence missing:{child_uuid}")
    trials = record.get("trials")
    configured_trials = (
        record.get("validation_config", {}).get("trials")
        if isinstance(record.get("validation_config"), Mapping)
        else None
    )
    if not isinstance(trials, list) or not isinstance(configured_trials, int) or len(trials) != configured_trials:
        raise ValueError(f"dtype runtime trial evidence mismatch:{child_uuid}")
    if record.get("assigned_dtype") not in {"float16", "bfloat16"}:
        raise ValueError(f"dtype runtime assigned target evidence missing:{child_uuid}")
    tolerance = record.get("tolerance")
    if not isinstance(tolerance, Mapping) or not all(
        isinstance(tolerance.get(key), (int, float)) and tolerance[key] >= 0 for key in ("rtol", "atol")
    ):
        raise ValueError(f"dtype runtime tolerance evidence missing:{child_uuid}")
    config = record["validation_config"]
    if (
        config.get("logical_forward_dtype_contract") != "all_floating_vN_and_final_outputs_match_assigned_dtype_v1"
        or config.get("internal_aten_non_target_floating_outputs") != "diagnostic_only"
        or config.get("internal_aten_complex_outputs") != "reject"
        or config.get("cast_equivalent_comparator_contract") != _DTYPE_FLOAT_COMPARATOR_CONTRACT
        or config.get("nonfloating_comparator_contract") != _DTYPE_EXACT_COMPARATOR_CONTRACT
    ):
        raise ValueError(f"dtype runtime semantic policy mismatch:{child_uuid}")
    if any(
        config.get(name) != value
        for name, value in (
            ("float16_rtol", 1e-2),
            ("float16_atol", 1e-2),
            ("bfloat16_rtol", 2e-2),
            ("bfloat16_atol", 2e-2),
        )
    ):
        raise ValueError(f"dtype runtime comparator config mismatch:{child_uuid}")
    target_name = f"torch.{record['assigned_dtype']}"
    expected_tolerance = 1e-2 if record["assigned_dtype"] == "float16" else 2e-2
    if tolerance != {"rtol": expected_tolerance, "atol": expected_tolerance}:
        raise ValueError(f"dtype runtime fixed tolerance mismatch:{child_uuid}")
    node_count = manifest.get("source_graph_node_count")
    if (
        not isinstance(node_count, int)
        or node_count <= 0
        or manifest.get("source_graph_operator_count") != manifest.get("operator_count")
    ):
        raise ValueError(f"dtype runtime source graph binding missing:{child_uuid}")
    expected_indices = _canonical_sha256(list(range(node_count)))
    expected_names = _canonical_sha256([f"v{index}" for index in range(node_count)])
    for ordinal, trial in enumerate(trials):
        if (
            not isinstance(trial, Mapping)
            or not isinstance(trial.get("inputs"), list)
            or not trial["inputs"]
            or not isinstance(trial.get("model_state"), Mapping)
            or not trial["model_state"]
            or not isinstance(trial.get("cast_equivalent_outputs"), list)
            or not trial["cast_equivalent_outputs"]
        ):
            raise ValueError(f"dtype runtime semantic payload missing:{child_uuid}")
        if trial.get("trial") != ordinal or trial.get("seed") != 17 + 10_007 * ordinal:
            raise ValueError(f"dtype runtime trial ordinal/seed mismatch:{child_uuid}")
        if trial.get("trial_payload_sha256") != _trial_payload_sha256(trial):
            raise ValueError(f"dtype runtime trial payload binding mismatch:{child_uuid}")
        for output_record in trial["cast_equivalent_outputs"]:
            if not isinstance(output_record, Mapping):
                raise ValueError(f"dtype runtime comparison payload missing:{child_uuid}")
            _validate_dtype_output_comparison(
                output_record,
                target=record["assigned_dtype"],
                tolerance=expected_tolerance,
                child_uuid=child_uuid,
            )
        for name in ("semantic_dispatch", "realized_dispatch"):
            trace = trial.get(name)
            logical = trace.get("logical_forward") if isinstance(trace, Mapping) else None
            output = trace.get("final_output") if isinstance(trace, Mapping) else None
            if (
                not isinstance(trace, Mapping)
                or not isinstance(logical, Mapping)
                or not isinstance(output, Mapping)
                or logical.get("target_dtype") != target_name
                or output.get("target_dtype") != target_name
                or logical.get("forward_return_count") != 1
                or logical.get("logical_value_count") != node_count
                or logical.get("logical_value_indices_sha256") != expected_indices
                or logical.get("logical_value_names_sha256") != expected_names
                or not isinstance(logical.get("floating_logical_value_count"), int)
                or not isinstance(logical.get("nonfloating_logical_value_count"), int)
                or logical["floating_logical_value_count"] + logical["nonfloating_logical_value_count"] != node_count
                or logical["floating_logical_value_count"] <= 0
                or not isinstance(output.get("floating_final_output_count"), int)
                or output["floating_final_output_count"] <= 0
                or logical.get("floating_logical_dtypes") != {target_name: logical["floating_logical_value_count"]}
                or output.get("floating_final_output_dtypes") != {target_name: output["floating_final_output_count"]}
                or trace.get("complex_output_calls") != 0
                or not isinstance(trace.get("internal_non_target_floating_output_calls"), int)
                or trace["internal_non_target_floating_output_calls"] < 0
            ):
                raise ValueError(f"dtype runtime dispatch evidence missing:{child_uuid}")


def _validate_liveness_record(
    record: Mapping[str, Any],
    *,
    parents_path: Path,
    children_path: Path,
    manifest_path: Path,
    candidate_index: int,
    parent_uuid: str,
    child_uuid: str,
    expected_device: str,
    manifest: Mapping[str, Any],
    child: Mapping[str, Any],
) -> None:
    if record.get("candidate_row_index") != candidate_index:
        raise ValueError(f"dtype runtime candidate index mismatch:{child_uuid}")
    if record.get("parent_uuid") != parent_uuid or record.get("child_uuid") != child_uuid:
        raise ValueError(f"dtype runtime UUID binding mismatch:{child_uuid}")
    for name, path in (
        ("parents_sha256", parents_path),
        ("children_sha256", children_path),
        ("manifest_sha256", manifest_path),
    ):
        if record.get(name) != _sha256_file(path):
            raise ValueError(f"dtype runtime {name} mismatch:{child_uuid}")
    config = record.get("validation_config")
    if not isinstance(config, Mapping) or config.get("device") != expected_device:
        raise ValueError(f"dtype runtime lacks a CUDA validation config:{child_uuid}")
    if config.get("seed") != 17 or not isinstance(config.get("trials"), int) or config["trials"] < 3:
        raise ValueError(f"dtype runtime liveness config mismatch:{child_uuid}")
    if not isinstance(config.get("timeout_seconds"), (int, float)) or config["timeout_seconds"] <= 0:
        raise ValueError(f"dtype runtime timeout config mismatch:{child_uuid}")
    execution_context = record.get("execution_context")
    if not isinstance(execution_context, Mapping) or execution_context != config.get("execution_context"):
        raise ValueError(f"dtype runtime execution context binding mismatch:{child_uuid}")
    if (
        execution_context.get("device_type") != "cuda"
        or execution_context.get("cudnn_benchmark") is not False
        or execution_context.get("cudnn_deterministic") is not True
        or type(execution_context.get("cudnn_enabled")) is not bool
    ):
        raise ValueError(f"dtype runtime deterministic CUDA context mismatch:{child_uuid}")
    if type(record.get("passed")) is not bool or not isinstance(record.get("status"), str):
        raise ValueError(f"dtype runtime status contract mismatch:{child_uuid}")
    coherence = record.get("coherence_class")
    if record.get("contract_version") not in _DTYPE_RUNTIME_CONTRACTS:
        raise ValueError(f"unsupported dtype runtime contract:{child_uuid}")
    if coherence not in {"parameter_free", "module_state"}:
        raise ValueError(f"dtype runtime coherence class mismatch:{child_uuid}")
    expected_contract = (
        "dtype_parameter_free_runtime_liveness_v6"
        if coherence == "parameter_free"
        else "dtype_module_state_runtime_liveness_v5"
    )
    expected_binding = (
        "dtype_parameter_free_runtime_binding_v6"
        if coherence == "parameter_free"
        else "dtype_module_state_runtime_binding_v5"
    )
    if record.get("contract_version") != expected_contract or record.get("binding_version") != expected_binding:
        raise ValueError(f"dtype runtime version binding mismatch:{child_uuid}")
    if record.get("result_binding_version") != _DTYPE_RESULT_BINDING_VERSION:
        raise ValueError(f"dtype runtime result binding version mismatch:{child_uuid}")
    if record.get("result_payload_sha256") != _result_payload_sha256(record):
        raise ValueError(f"dtype runtime result payload binding mismatch:{child_uuid}")
    fields = (
        "contract_version",
        "binding_version",
        "coherence_class",
        "validator_source_sha256",
        "launcher_source_sha256",
        "runtime_dependencies",
        "parents_sha256",
        "children_sha256",
        "manifest_sha256",
        "allowlist_sha256",
        "validation_config",
        "execution_context",
    )
    expected_sources = _expected_liveness_source_hashes()
    for field, expected_hash in expected_sources.items():
        if record.get(field) != expected_hash:
            raise ValueError(f"dtype runtime {field} does not bind the production source:{child_uuid}")
    # The production launcher invokes all children in the serial lane and does
    # not pass --child-uuid-file.  Reject a self-declared allowlist hash that
    # could otherwise hide a partial validation run.
    if record.get("allowlist_sha256") is not None:
        raise ValueError(f"dtype runtime unexpected allowlist binding:{child_uuid}")
    if (
        record.get("assigned_dtype") != manifest.get("assigned_target")
        or record.get("transformed_factory_count") != manifest.get("transformed_factory_count")
        or record.get("coherence_class") != manifest.get("coherence_class")
    ):
        raise ValueError(f"dtype runtime assignment/manifest mismatch:{child_uuid}")
    realized = manifest.get("realized_intervention")
    if (
        not isinstance(realized, Mapping)
        or realized.get("dtype_after") != manifest.get("assigned_target")
        or realized.get("factory_count") != manifest.get("factory_count")
        or realized.get("logical_forward_dtype_policy") != "all_floating_vN_and_final_outputs_match_assigned_dtype_v1"
        or realized.get("internal_aten_non_target_floating_outputs") != "diagnostic_only"
        or realized.get("internal_aten_complex_outputs") != "reject"
        or "fp32_fallback_policy" in realized
    ):
        raise ValueError(f"dtype manifest realized intervention mismatch:{child_uuid}")
    status = record.get("status")
    if status == "passed":
        if record.get("passed") is not True or record.get("reason") is not None:
            raise ValueError(f"dtype runtime passed status mismatch:{child_uuid}")
    elif status == "unsupported":
        _validate_unsupported_runtime_schema(record, child_uuid)
    else:
        raise ValueError(f"dtype runtime forbidden status:{child_uuid}:{status}")
    augmentation = _nested(child, "extra_info.augmentation")
    expected_augmented = {
        "parent_uuid": manifest.get("parent_uuid"),
        "child_uuid": child_uuid,
        "intervention_kind": "dtype",
        "dtype_after": manifest.get("assigned_target"),
        "layout_after": None,
        "factory_count": manifest.get("factory_count"),
    }
    if not isinstance(augmentation, Mapping) or any(
        augmentation.get(name) != value for name, value in expected_augmented.items()
    ):
        raise ValueError(f"dtype augmentation/manifest mismatch:{child_uuid}")
    if record.get("passed") is True:
        _validate_liveness_semantic_evidence(record, child_uuid, manifest)
    evidence = {name: record.get(name) for name in fields}
    if record.get("validation_binding_sha256") != _canonical_sha256(evidence):
        raise ValueError(f"dtype runtime validation binding mismatch:{child_uuid}")


def _strict_liveness_records(
    lane_dir: Path,
    runtime_dir: Path,
    parents: Sequence[Mapping[str, Any]],
    children: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[Path]]:
    parents_path = lane_dir / "parents.parquet"
    children_path = lane_dir / "candidates.parquet"
    manifest_path = lane_dir / "manifest.jsonl"
    manifests = _read_jsonl(manifest_path)
    if len(manifests) != len(children):
        raise ValueError(f"{lane_dir.name} runtime manifest row count mismatch")
    lane_summary = json.loads((lane_dir / "summary.json").read_text())
    stage = lane_summary.get("stage")
    if stage != "dtype":
        raise ValueError(f"unsupported runtime selection lane:{stage!r}")
    _validate_lane_builder_binding(lane_summary, lane_dir, stage)
    artifacts = lane_summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError(f"{stage} summary has no artifacts")
    _exact_artifact(artifacts, "parents", parents_path, f"{stage} summary")
    _exact_artifact(artifacts, "candidates", children_path, f"{stage} summary")
    _exact_artifact(artifacts, "manifest", manifest_path, f"{stage} summary")
    shards = _runtime_shards(runtime_dir, stage)
    if len(shards) != REQUIRED_H20_SHARDS:
        raise ValueError(f"{stage} runtime must contain {REQUIRED_H20_SHARDS} H20 shards")
    records: dict[str, dict[str, Any]] = {}
    for shard_index, shard_count, path in shards:
        shard_records = _read_jsonl(path)
        expected_positions = list(range(shard_index, len(children), shard_count))
        if len(shard_records) != len(expected_positions):
            raise ValueError(f"{stage} runtime shard row count mismatch:{path.name}")
        if any(not isinstance(record.get("validation_binding_sha256"), str) for record in shard_records):
            raise ValueError(f"{stage} runtime shard lacks validation binding:{path.name}")
        for expected_index, record in zip(expected_positions, shard_records, strict=True):
            parent_uuid, _ = _identity(parents[expected_index])
            child_uuid, _ = _identity(children[expected_index])
            _validate_liveness_record(
                record,
                parents_path=parents_path,
                children_path=children_path,
                manifest_path=manifest_path,
                candidate_index=expected_index,
                parent_uuid=parent_uuid,
                child_uuid=child_uuid,
                expected_device=f"cuda:{shard_index}",
                manifest=manifests[expected_index],
                child=children[expected_index],
            )
            if child_uuid in records:
                raise ValueError(f"invalid or duplicate runtime identity:{child_uuid}")
            records[child_uuid] = dict(record)
    if len(records) != len(children):
        raise ValueError(f"{stage} runtime evidence is incomplete")
    return records, [path for _, _, path in shards]


def _strict_shape_records(
    shape_dir: Path,
    runtime_dir: Path,
    children: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[Path]]:
    children_path = shape_dir / "candidates.parquet"
    manifest_path = shape_dir / "manifest.jsonl"
    summary_path = shape_dir / "summary.json"
    shape_summary = json.loads(summary_path.read_text())
    if shape_summary.get("contract") != CONTRACT or shape_summary.get("stage") != "shape":
        raise ValueError("shape summary contract mismatch")
    _validate_lane_builder_binding(shape_summary, shape_dir, "shape")
    artifacts = shape_summary.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("shape summary has no artifacts")
    _exact_artifact(artifacts, "candidates", children_path, "shape summary")
    _exact_artifact(artifacts, "manifest", manifest_path, "shape summary")
    manifests = _read_jsonl(manifest_path)
    if len(manifests) != len(children):
        raise ValueError("shape runtime manifest row count mismatch")
    shards = _runtime_shards(runtime_dir, "shape")
    if len(shards) != REQUIRED_H20_SHARDS:
        raise ValueError(f"shape runtime must contain {REQUIRED_H20_SHARDS} H20 shards")
    expected_summaries = {
        path.with_name(path.name.replace(".records.jsonl", ".summary.json")) for _, _, path in shards
    }
    if set(runtime_dir.glob("*.summary.json")) != expected_summaries:
        raise ValueError("shape runtime directory contains an unknown or missing shard summary")
    records: dict[str, dict[str, Any]] = {}
    for shard_index, shard_count, path in shards:
        summary_file = path.with_name(path.name.replace(".records.jsonl", ".summary.json"))
        if not summary_file.is_file():
            raise ValueError(f"shape runtime shard lacks its summary:{path.name}")
        runtime_summary = json.loads(summary_file.read_text())
        if runtime_summary.get("contract") != _SHAPE_RUNTIME_CONTRACT:
            raise ValueError(f"shape runtime contract mismatch:{path.name}")
        if runtime_summary.get("validation_mode") != "shape_candidate_full_row":
            raise ValueError(f"shape runtime validation mode mismatch:{path.name}")
        if runtime_summary.get("device") != f"cuda:{shard_index}":
            raise ValueError(f"shape runtime is not CUDA:{path.name}")
        identity = runtime_summary.get("gpu_identity")
        if (
            not isinstance(identity, Mapping)
            or identity.get("device") != f"cuda:{shard_index}"
            or "h20" not in str(identity.get("name", "")).lower()
            or not isinstance(identity.get("compute_capability"), list)
            or not isinstance(identity.get("total_memory_bytes"), int)
        ):
            raise ValueError(f"shape runtime H20 identity mismatch:{path.name}")
        context = runtime_summary.get("runtime_execution")
        if (
            not isinstance(context, Mapping)
            or context.get("device_type") != "cuda"
            or context.get("cudnn_benchmark") is not False
            or context.get("cudnn_deterministic") is not True
        ):
            raise ValueError(f"shape runtime deterministic CUDA context mismatch:{path.name}")
        if runtime_summary.get("shard_index") != shard_index or runtime_summary.get("shard_count") != shard_count:
            raise ValueError(f"shape runtime shard metadata mismatch:{path.name}")
        binding = runtime_summary.get("source_binding")
        if not isinstance(binding, Mapping):
            raise ValueError(f"shape runtime source binding missing:{path.name}")
        _exact_artifact(binding, "selected", children_path, f"shape runtime:{path.name}")
        _exact_artifact(binding, "manifest", manifest_path, f"shape runtime:{path.name}")
        _exact_artifact(binding, "summary", summary_path, f"shape runtime:{path.name}")
        for source_name, source_path in _SHAPE_RUNTIME_SOURCES.items():
            _exact_artifact(binding, source_name, source_path, f"shape runtime:{path.name}")
        expected_adapter = {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
            "contract": SOURCE_BINDING_CONTRACT,
        }
        if binding.get("expansion_adapter") != expected_adapter:
            raise ValueError(f"shape runtime expansion adapter binding mismatch:{path.name}")
        _exact_artifact(runtime_summary, "records", path, f"shape runtime:{path.name}")
        shard_records = _read_jsonl(path)
        expected_positions = list(range(shard_index, len(children), shard_count))
        if len(shard_records) != len(expected_positions) or runtime_summary.get("rows") != len(shard_records):
            raise ValueError(f"shape runtime shard row count mismatch:{path.name}")
        if runtime_summary.get("passed") != sum(record.get("passed") is True for record in shard_records):
            raise ValueError(f"shape runtime summary passed count mismatch:{path.name}")
        if runtime_summary.get("failed") != sum(record.get("passed") is not True for record in shard_records):
            raise ValueError(f"shape runtime summary failed count mismatch:{path.name}")
        positions = {
            "first": expected_positions[0] if expected_positions else None,
            "last": expected_positions[-1] if expected_positions else None,
        }
        if runtime_summary.get("positions") != positions:
            raise ValueError(f"shape runtime position summary mismatch:{path.name}")
        for expected_position, record in zip(expected_positions, shard_records, strict=True):
            uuid, _ = _identity(children[expected_position])
            if record.get("position") != expected_position or record.get("uuid") != uuid:
                raise ValueError(f"shape runtime row identity mismatch:{path.name}:{expected_position}")
            if (
                record.get("passed") is not True
                or record.get("errors") not in ([], None)
                or record.get("failed_checks") not in ([], None)
            ):
                raise ValueError(f"shape runtime row status mismatch:{path.name}:{expected_position}")
            checks = record.get("checks")
            if not isinstance(checks, Mapping) or not checks:
                raise ValueError(f"shape runtime checks missing:{path.name}:{expected_position}")
            required_checks = _shape_required_checks(children[expected_position], manifests[expected_position])
            missing_checks = sorted(name for name in required_checks if checks.get(name) is not True)
            if missing_checks:
                raise ValueError(
                    f"shape runtime required checks missing:{path.name}:{expected_position}:{','.join(missing_checks)}"
                )
            runtime = record.get("runtime")
            if (
                not isinstance(runtime, Mapping)
                or not isinstance(runtime.get("dispatch_count"), int)
                or runtime["dispatch_count"] <= 0
            ):
                raise ValueError(f"shape runtime dispatch evidence missing:{path.name}:{expected_position}")
            if uuid in records:
                raise ValueError(f"invalid or duplicate shape runtime identity:{uuid}")
            records[uuid] = dict(record)
    if len(records) != len(children):
        raise ValueError("shape runtime evidence is incomplete")
    if not all(record.get("passed") is True for record in records.values()):
        raise ValueError("shape runtime evidence contains a failed row")
    return records, [path for _, _, path in shards]


_SELECTION_OUTPUT_NAMES = (
    "selected.parquet",
    "selected.parents.parquet",
    "selected.manifest.jsonl",
    "selection_summary.json",
)


def _assert_selection_outputs_absent(lane_dir: Path) -> None:
    existing = [name for name in _SELECTION_OUTPUT_NAMES if (lane_dir / name).exists()]
    if existing:
        raise FileExistsError(f"selection outputs already exist:{','.join(existing)}")


def select_passed(lane_dir: Path, runtime_dir: Path) -> dict[str, Any]:
    parents = pq.read_table(lane_dir / "parents.parquet").to_pylist()
    children = pq.read_table(lane_dir / "candidates.parquet").to_pylist()
    manifests = _read_jsonl(lane_dir / "manifest.jsonl")
    if not (len(parents) == len(children) == len(manifests)):
        raise ValueError("lane artifacts are not aligned")
    records, runtime_paths = _strict_liveness_records(lane_dir, runtime_dir, parents, children)
    _assert_selection_outputs_absent(lane_dir)
    selected_parents: list[dict[str, Any]] = []
    selected_children: list[dict[str, Any]] = []
    selected_manifests: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for parent, child, manifest in zip(parents, children, manifests, strict=True):
        uuid, _ = _identity(child)
        record = records.get(uuid)
        status = (
            "missing" if record is None else str(record.get("status", "passed" if record.get("passed") else "failed"))
        )
        counts[status] = counts.get(status, 0) + 1
        if record is None or record.get("passed") is not True:
            continue
        selected_parents.append(parent)
        selected_children.append(child)
        selected_manifests.append({**manifest, "runtime_evidence": record})
    schema = pq.ParquetFile(lane_dir / "candidates.parquet").schema_arrow
    _write_rows(lane_dir / "selected.parquet", selected_children, schema, "runtime_selected")
    _write_rows(lane_dir / "selected.parents.parquet", selected_parents, schema, "runtime_selected_parents")
    _write_jsonl(lane_dir / "selected.manifest.jsonl", selected_manifests)
    summary = {
        "contract": CONTRACT,
        "stage": "runtime_selection",
        "candidate_rows": len(children),
        "runtime_record_rows": len(records),
        "selected_rows": len(selected_children),
        "status_counts": dict(sorted(counts.items())),
        "runtime_sources": [{"path": str(path.resolve()), "sha256": _sha256_file(path)} for path in runtime_paths],
        "source_binding": {
            "candidates": {
                "path": str((lane_dir / "candidates.parquet").resolve()),
                "sha256": _sha256_file(lane_dir / "candidates.parquet"),
            },
            "manifest": {
                "path": str((lane_dir / "manifest.jsonl").resolve()),
                "sha256": _sha256_file(lane_dir / "manifest.jsonl"),
            },
        },
        "artifacts": {
            "selected": {
                "path": str((lane_dir / "selected.parquet").resolve()),
                "sha256": _sha256_file(lane_dir / "selected.parquet"),
            },
            "selected_manifest": {
                "path": str((lane_dir / "selected.manifest.jsonl").resolve()),
                "sha256": _sha256_file(lane_dir / "selected.manifest.jsonl"),
            },
        },
        "review_only": True,
        "training_approved": False,
    }
    _write_json(lane_dir / "selection_summary.json", summary)
    return summary


def select_shape(shape_dir: Path, runtime_dir: Path) -> dict[str, Any]:
    children = pq.read_table(shape_dir / "candidates.parquet").to_pylist()
    manifests = _read_jsonl(shape_dir / "manifest.jsonl")
    if len(children) != len(manifests):
        raise ValueError("shape artifacts are not aligned")
    records, runtime_paths = _strict_shape_records(shape_dir, runtime_dir, children)
    _assert_selection_outputs_absent(shape_dir)
    selected_children: list[dict[str, Any]] = []
    selected_manifests: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for child, manifest in zip(children, manifests, strict=True):
        uuid, _ = _identity(child)
        record = records.get(uuid)
        status = "missing" if record is None else ("passed" if record.get("passed") is True else "failed")
        counts[status] = counts.get(status, 0) + 1
        if record is None or record.get("passed") is not True:
            continue
        selected_children.append(child)
        selected_manifests.append({**manifest, "runtime_evidence": record})
    schema = pq.ParquetFile(shape_dir / "candidates.parquet").schema_arrow
    _write_rows(shape_dir / "selected.parquet", selected_children, schema, "shape_runtime_selected")
    _write_jsonl(shape_dir / "selected.manifest.jsonl", selected_manifests)
    summary = {
        "contract": CONTRACT,
        "stage": "shape_runtime_selection",
        "candidate_rows": len(children),
        "runtime_record_rows": len(records),
        "selected_rows": len(selected_children),
        "status_counts": dict(sorted(counts.items())),
        "runtime_sources": [{"path": str(path.resolve()), "sha256": _sha256_file(path)} for path in runtime_paths],
        "source_binding": {
            "candidates": {
                "path": str((shape_dir / "candidates.parquet").resolve()),
                "sha256": _sha256_file(shape_dir / "candidates.parquet"),
            },
            "manifest": {
                "path": str((shape_dir / "manifest.jsonl").resolve()),
                "sha256": _sha256_file(shape_dir / "manifest.jsonl"),
            },
        },
        "artifacts": {
            "selected": {
                "path": str((shape_dir / "selected.parquet").resolve()),
                "sha256": _sha256_file(shape_dir / "selected.parquet"),
            },
            "selected_manifest": {
                "path": str((shape_dir / "selected.manifest.jsonl").resolve()),
                "sha256": _sha256_file(shape_dir / "selected.manifest.jsonl"),
            },
        },
        "review_only": True,
        "training_approved": False,
    }
    _write_json(shape_dir / "selection_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    shape = subparsers.add_parser("shape")
    shape.add_argument("parent_parquet", type=Path)
    shape.add_argument("parent_manifest", type=Path)
    shape.add_argument("output_dir", type=Path)
    dtype = subparsers.add_parser("dtype")
    dtype.add_argument("shape_parquet", type=Path)
    dtype.add_argument("shape_manifest", type=Path)
    dtype.add_argument("output_dir", type=Path)
    select = subparsers.add_parser("select-passed")
    select.add_argument("lane_dir", type=Path)
    select.add_argument("runtime_dir", type=Path)
    shape_select = subparsers.add_parser("select-shape")
    shape_select.add_argument("shape_dir", type=Path)
    shape_select.add_argument("runtime_dir", type=Path)
    args = parser.parse_args()
    if args.command == "shape":
        result = build_shape(args.parent_parquet, args.parent_manifest, args.output_dir)
    elif args.command == "dtype":
        result = build_dtype(args.shape_parquet, args.shape_manifest, args.output_dir)
    elif args.command == "select-passed":
        result = select_passed(args.lane_dir, args.runtime_dir)
    else:
        result = select_shape(args.shape_dir, args.runtime_dir)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
