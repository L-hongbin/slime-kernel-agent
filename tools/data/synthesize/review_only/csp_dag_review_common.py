#!/usr/bin/env python3
"""Shared, production-independent helpers for the CSP-DAG release review."""

from __future__ import annotations

import ast
import collections
import functools
import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

CONTRACT = "csp_dag_shape_dtype_final_independent_review_v1"
FINAL_CONTRACT = "csp_dag_shape_dtype_expansion_final_v1"
EXPANSION_CONTRACT = "csp_dag_shape_dtype_layout_expansion_v1"
SEED = "csp_dag_shape_dtype_final_stratified_review_v1"
EXPANSION_LANES = ("shape", "dtype")
LANES = ("base", *EXPANSION_LANES)
POST_MODES = {
    "shape": "shape_full_row",
    "dtype": "dtype_standalone_selected_child",
}
POST_CONTRACT = "open_csp_dag_dataset_validation_v1"
SOURCE_BINDING_CONTRACT = "csp_dag_expansion_exact_source_v1"
REQUIRED_FAMILIES = frozenset(
    {
        "activation",
        "conv",
        "normalization",
        "pooling",
        "reduction",
        "indexing_scatter",
        "matmul_linear",
        "loss_distance",
        "scaled_dot_product_attention",
        "shape_layout",
    }
)
BASE_MANIFEST_CONTRACT = "open_csp_dag_low_level_canary_v4"
SOURCE_FREEZE_PATHS = (
    "tools/data/cleaning/__init__.py",
    "tools/data/cleaning/ast_similarity.py",
    "tools/data/cleaning/complexity.py",
    "tools/data/cleaning/external.py",
    "tools/data/cleaning/pipeline.py",
    "tools/data/cleaning/runtime_validation.py",
    "tools/data/cleaning/similarity.py",
    "tools/data/cleaning/static_analysis.py",
    "tools/__init__.py",
    "tools/data/__init__.py",
    "tools/data/synthesize/__init__.py",
    "tools/data/synthesize/augment_prompt_tasks.py",
    "tools/data/synthesize/intervention_semantic_gates.py",
    "tools/data/synthesize/profile_prompt_tvm_distribution.py",
    "tools/data/synthesize/random_method/__init__.py",
    "tools/data/synthesize/random_method/solve_value_coverage.py",
    "tools/data/synthesize/select_four_lane_kernelbench_canary.py",
    "tools/data/synthesize/select_low_level_kernelbench_canary.py",
    "tools/data/synthesize/serial_random_wrappers.py",
    "tools/data/synthesize/serial_source_contract.py",
    "tools/data/synthesize/validate_train_mode_contract.py",
    "tools/data/synthesize/csp_dag_method/__init__.py",
    "tools/data/synthesize/csp_dag_method/build_csp_dag_input_expansions.py",
    "tools/data/synthesize/csp_dag_method/csp_dag_static_binding.py",
    "tools/data/synthesize/csp_dag_method/csp_dag_finalization.py",
    "tools/data/synthesize/csp_dag_method/generate_csp_dag.py",
    "tools/data/synthesize/csp_dag_method/run_input_expansion_liveness.sh",
    "tools/data/synthesize/csp_dag_method/validate_csp_dag.py",
    "tools/data/synthesize/csp_dag_method/validate_csp_dag_dataset.py",
    "tools/data/synthesize/csp_dag_method/validate_csp_dag_repeatability.py",
    "tools/data/synthesize/dtype_method/__init__.py",
    "tools/data/synthesize/dtype_method/solve_dtype_coverage.py",
    "tools/data/synthesize/dtype_method/validate_dtype_liveness.py",
)
SOURCE_FREEZE_INVENTORY_ROWS = len(SOURCE_FREEZE_PATHS)
FINALIZATION_SOURCE_FREEZE_PATHS = (
    *SOURCE_FREEZE_PATHS,
    "tools/data/synthesize/csp_dag_method/finalize_shape_dtype_expansion.py",
)
FINALIZATION_SOURCE_FREEZE_INVENTORY_ROWS = len(FINALIZATION_SOURCE_FREEZE_PATHS)
ADDITIVE_LINEAGE_FIELDS = frozenset(
    {
        "additive_row_index",
        "lane",
        "lane_row_index",
        "uuid",
        "root_base_uuid",
        "parent_uuid",
        "reference_sha256",
        "normalized_ast_sha256",
        "source_parquet",
        "source_manifest",
        "source_manifest_row_sha256",
    }
)


@functools.cache
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _row_hash(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _exact_file_binding(binding: Any, expected: Path, label: str) -> str | None:
    if not isinstance(binding, Mapping):
        return f"missing file binding:{label}"
    if binding.get("path") != str(expected.resolve()) or binding.get("sha256") != _sha256_file(expected):
        return f"file binding mismatch:{label}"
    return None


def _bound_source_errors(binding: Any, expected: Path, label: str) -> list[str]:
    error = _exact_file_binding(binding, expected, label)
    return [error] if error else []


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _code(row: Mapping[str, Any]) -> str:
    return str(row["reward_model"]["ground_truth"])


def _uuid(row: Mapping[str, Any]) -> str:
    return str(row["extra_info"]["uuid"])


def _ast_hash(code: str) -> str:
    tree = ast.parse(code)
    normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _ref_hash(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _code_bucket(code: str) -> str:
    count = sum(bool(line.strip()) and not line.lstrip().startswith("#") for line in code.splitlines())
    if count < 20:
        return "<20"
    if count <= 34:
        return "20-34"
    if count <= 49:
        return "35-49"
    if count <= 74:
        return "50-74"
    return ">=75"


def _operator_bucket(manifest: Mapping[str, Any]) -> str:
    signature = manifest.get("operator_signature")
    count = len(signature) if isinstance(signature, list) else int(manifest.get("operator_count", 0))
    if count == 1:
        return "1"
    if count <= 4:
        return "2-4"
    if count <= 9:
        return "5-9"
    if count <= 15:
        return "10-15"
    return ">=16"


def _class_count(manifest: Mapping[str, Any]) -> int:
    return int(manifest.get("complexity", {}).get("top_level_class_count", 0))


def _manifest_uuid(manifest: Mapping[str, Any]) -> str | None:
    value = manifest.get("uuid", manifest.get("child_uuid"))
    return str(value) if isinstance(value, str) else None


def _resolved_metadata(manifest: Mapping[str, Any], parent: Mapping[str, Any] | None) -> dict[str, Any]:
    """Carry graph/form metadata through dtype child manifests.

    Shape manifests own the base graph metadata.  Dtype manifests only
    retain scalar summaries such as `operator_count`; their parent lineage is
    the authoritative carrier for the family palette and class form.
    """
    inherited = dict(parent or {})
    for key in ("actual_low_level_families", "operator_signature", "complexity", "operator_count"):
        if manifest.get(key) is not None:
            inherited[key] = manifest[key]
    return inherited


def _check_source_freeze(
    freeze: Path,
    inventory: Path,
    repo_root: Path,
    *,
    expected_paths: Sequence[str] = SOURCE_FREEZE_PATHS,
    label: str = "runtime",
) -> list[str]:
    errors: list[str] = []
    if not freeze.is_file() or not inventory.is_file():
        return [f"missing source freeze/inventory:{freeze}:{inventory}"]
    inventory_rows = inventory.read_text().splitlines()
    if tuple(inventory_rows) != tuple(expected_paths):
        errors.append(f"{label} source freeze inventory differs from the trusted production path set")
    frozen_rows: list[str] = []
    frozen_hashes: dict[str, str] = {}
    for line_no, line in enumerate(freeze.read_text().splitlines(), 1):
        if not line.strip():
            errors.append(f"empty source freeze row:{line_no}")
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            errors.append(f"invalid source freeze row:{line_no}")
            continue
        if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
            errors.append(f"invalid source freeze SHA-256:{line_no}")
        if relative in frozen_hashes:
            errors.append(f"duplicate source freeze row:{relative}")
        frozen_rows.append(relative)
        frozen_hashes[relative] = expected
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts or relative_path.as_posix() != relative:
            errors.append(f"source freeze path is not normalized repository-relative:{relative}")
            continue
        source = repo_root / relative_path
        try:
            source.resolve().relative_to(repo_root.resolve())
        except ValueError:
            errors.append(f"source freeze path leaves repository:{relative}")
            continue
        if source.is_symlink():
            errors.append(f"source freeze path is a symlink:{relative}")
        elif not source.is_file() or _sha256_file(source) != expected:
            errors.append(f"source freeze mismatch:{relative}")
    if frozen_rows != inventory_rows or set(frozen_rows) != set(inventory_rows):
        errors.append(f"{label} source freeze does not exactly match its inventory/order")
    return errors


def _rows_and_manifests(path: Path, manifest: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = pq.read_table(path).to_pylist()
    manifests = _load_jsonl(manifest)
    if len(rows) != len(manifests):
        raise ValueError(f"row/manifest length mismatch:{path}")
    return rows, manifests


def _normalized_for_schema(row: Mapping[str, Any], schema: Any) -> dict[str, Any]:
    """Mirror Arrow's nullable-field normalization without importing synthesis code."""
    import pyarrow as pa

    return pa.Table.from_pylist([dict(row)], schema=schema).to_pylist()[0]


def _histogram(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(collections.Counter(str(value) for value in values).items()))


def _shard_paths(
    runtime_dir: Path, lane: str, errors: list[str], *, require_summaries: bool
) -> list[tuple[int, Path, Path]]:
    expected = {runtime_dir / f"shard_{index:03d}_of_004.records.jsonl" for index in range(4)}
    actual = set(runtime_dir.glob("*.records.jsonl")) if runtime_dir.is_dir() else set()
    if actual != expected:
        errors.append(f"{lane}: pre/post runtime records have missing or extra shards")
    if runtime_dir.is_dir() and any(path.parent != runtime_dir for path in runtime_dir.rglob("*.jsonl")):
        errors.append(f"{lane}: runtime directory contains nested JSONL evidence")
    expected_summaries = {path.with_name(path.name.replace(".records.jsonl", ".summary.json")) for path in expected}
    actual_summaries = set(runtime_dir.glob("*.summary.json")) if runtime_dir.is_dir() else set()
    if require_summaries and actual_summaries != expected_summaries:
        errors.append(f"{lane}: pre/post runtime summaries have missing or extra shards")
    if not require_summaries and actual_summaries:
        errors.append(f"{lane}: liveness runtime has unexpected shard summaries")
    return [
        (
            index,
            runtime_dir / f"shard_{index:03d}_of_004.records.jsonl",
            runtime_dir / f"shard_{index:03d}_of_004.summary.json",
        )
        for index in range(4)
    ]


def _manifest_identity_errors(lane: str, row: Mapping[str, Any], manifest: Mapping[str, Any], index: int) -> list[str]:
    code, uuid = _code(row), _uuid(row)
    errors: list[str] = []
    if _manifest_uuid(manifest) != uuid:
        errors.append(f"{lane}: manifest UUID mismatch:{index}")
    if manifest.get("reference_sha256", manifest.get("child_reference_sha256")) != _ref_hash(code):
        errors.append(f"{lane}: manifest reference SHA-256 mismatch:{index}")
    if manifest.get("normalized_ast_sha256", manifest.get("child_normalized_ast_sha256")) != _ast_hash(code):
        errors.append(f"{lane}: manifest AST SHA-256 mismatch:{index}")
    if manifest.get("training_approved") is not False:
        errors.append(f"{lane}: manifest governance mismatch:{index}")
    return errors


def _base_authority_errors(base: Path, summary: Mapping[str, Any]) -> list[str]:
    """Check the bound base release summaries without rerunning their validators."""
    errors: list[str] = []
    final_path = base / "final_summary.json"
    if not final_path.is_file():
        return ["base final summary missing"]
    final = json.loads(final_path.read_text())
    rows = pq.ParquetFile(base / "selected.parquet").metadata.num_rows
    if (
        final.get("contract") != "open_csp_dag_low_level_5k_final_v3"
        or final.get("generation_summary_contract") != summary.get("contract")
        or final.get("rows") != rows
        or final.get("runtime_complete_cpu_and_h20") is not True
        or final.get("review_only") is not True
        or final.get("training_approved") is not False
    ):
        errors.append("base final summary contract/count mismatch")
    for key, path in (
        ("selected", base / "selected.parquet"),
        ("manifest", base / "selected.manifest.jsonl"),
        ("generation_summary", base / "summary.json"),
    ):
        error = _exact_file_binding(final.get("source_binding", {}).get(key), path, f"base final:{key}")
        if error:
            errors.append(error)

    authority_paths = {
        "retained_summary": base / "near_dedup/retained.summary.json",
        "raw_generation_summary": base / "near_dedup/raw5000.summary.json",
        "retained_strict_retained_uuids": base / "near_dedup/retained.strict_retained_uuids.jsonl",
        "raw_generation_strict_retained_uuids": base / "near_dedup/raw5000.strict_retained_uuids.jsonl",
    }
    declared_authority = final.get("source_binding", {}).get("near_dedup_authority", {})
    for key, path in authority_paths.items():
        error = _exact_file_binding(declared_authority.get(key), path, f"base final:{key}")
        if error:
            errors.append(error)
        if key.endswith("uuids") and declared_authority.get(key, {}).get("rows") != rows:
            errors.append(f"base final near-dedup row count mismatch:{key}")
    if all(path.is_file() for path in authority_paths.values()):
        retained = json.loads(authority_paths["retained_summary"].read_text())
        raw = json.loads(authority_paths["raw_generation_summary"].read_text())
        if final.get("dedup_audit") != retained or final.get("diversity") != retained:
            errors.append("base final retained near-dedup summary mismatch")
        if final.get("raw_generation_dedup") != raw:
            errors.append("base final raw near-dedup summary mismatch")

    runtime = final.get("runtime_validations", {})
    if set(runtime) != {"cpu", "cuda:0"}:
        errors.append("base final runtime catalog mismatch")
    for device, directory in (("cpu", "cpu"), ("cuda:0", "cuda_0")):
        declared = runtime.get(device, {})
        if (
            declared.get("rows") != rows
            or declared.get("passed") != rows
            or declared.get("failed") != 0
            or declared.get("all_coverage_checks_passed") is not True
        ):
            errors.append(f"base final runtime count mismatch:{device}")
        for key, path in (
            ("summary", base / "runtime" / directory / "summary.json"),
            ("records", base / "runtime" / directory / "records.jsonl"),
        ):
            error = _exact_file_binding(declared.get(key), path, f"base final:{device}:{key}")
            if error:
                errors.append(error)
    return errors
