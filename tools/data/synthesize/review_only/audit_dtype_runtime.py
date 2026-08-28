#!/usr/bin/env python3
"""Read-only evidence audit for the CSP-DAG dtype v6/v5 H20 lane.

This program imports no synthesis or runtime-validator code.  It reconstructs
the dtype lane's artifact, source-lineage, four-shard, and per-trial contracts
from serialized artifacts, then writes an audit which binds every input path
and SHA and its own final SHA.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from tools.data.synthesize.review_only.dtype_runtime_evidence import (
    BINDING_FIELDS,
    CONTRACTS,
    EXACT_COMPARATOR_CONTRACT,
    EXPANSION_CONTRACT,
    FLOAT_COMPARATOR_CONTRACT,
    RESULT_BINDING_VERSION,
    add,
    canonical_sha256,
    manifest_errors,
    memory_guard_errors,
    payload_sha256,
    row_identity,
    trial_errors,
    unsupported_record_errors,
)

CONTRACT = "independent_dtype_v6_v5_runtime_audit_v3"
SOURCE_CONTRACT = "csp_dag_expansion_exact_source_v1"
REQUIRED_SHARDS = 4
REPO_ROOT = Path(__file__).resolve().parents[4]
VALIDATOR = REPO_ROOT / "tools/data/synthesize/dtype_method/validate_dtype_liveness.py"
LAUNCHER = REPO_ROOT / "tools/data/synthesize/csp_dag_method/run_input_expansion_liveness.sh"
DEPENDENCIES = {
    "runtime_validation": REPO_ROOT / "tools/data/cleaning/runtime_validation.py",
    "serial_source_contract": REPO_ROOT / "tools/data/synthesize/serial_source_contract.py",
    "repeatability_execution_context": REPO_ROOT
    / "tools/data/synthesize/csp_dag_method/validate_csp_dag_repeatability.py",
    "train_mode_memory_guard": REPO_ROOT / "tools/data/synthesize/validate_train_mode_contract.py",
}
BUILDER = REPO_ROOT / "tools/data/synthesize/csp_dag_method/build_csp_dag_input_expansions.py"
BUILDER_DEPENDENCIES = {
    "generator": REPO_ROOT / "tools/data/synthesize/csp_dag_method/generate_csp_dag.py",
    "static_binding": REPO_ROOT / "tools/data/synthesize/csp_dag_method/csp_dag_static_binding.py",
    "intervention_semantic_gates": REPO_ROOT / "tools/data/synthesize/intervention_semantic_gates.py",
    "solve_dtype_coverage": REPO_ROOT / "tools/data/synthesize/dtype_method/solve_dtype_coverage.py",
    "augment_prompt_tasks": REPO_ROOT / "tools/data/synthesize/augment_prompt_tasks.py",
    "complexity": REPO_ROOT / "tools/data/cleaning/complexity.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL:{path.name}:{number}")
            rows.append(value)
    return rows


def exact_file(value: Any, path: Path) -> bool:
    return (
        isinstance(value, Mapping)
        and path.is_file()
        and value.get("path") == str(path.resolve())
        and value.get("sha256") == sha256_file(path)
    )


def dtype_parent_matches_source(parent: Mapping[str, Any], source: Mapping[str, Any]) -> bool:
    """Allow only the frozen dtype builder's exact nullable parent projection."""
    expected = json.loads(json.dumps(source))
    extra = expected.get("extra_info")
    if not isinstance(extra, dict):
        return False
    extra["augmentation"] = None
    return canonical_sha256(parent) == canonical_sha256(expected)


def source_binding_errors(
    lane_dir: Path,
    summary: Mapping[str, Any],
    parents: Sequence[Mapping[str, Any]],
    manifests: Sequence[Mapping[str, Any]],
    findings: list[dict[str, str]],
) -> tuple[dict[str, str], list[dict[str, Any]]]:
    source = summary.get("source_binding")
    source = source.get("source") if isinstance(source, Mapping) else None
    shape = lane_dir.parent / "shape"
    bindings = {
        "source_artifact": shape / "selected.parquet",
        "source_manifest": shape / "selected.manifest.jsonl",
        "source_summary": shape / "selection_summary.json",
    }
    if not isinstance(source, Mapping) or source.get("contract") != SOURCE_CONTRACT or source.get("stage") != "shape":
        add(findings, "lineage", "dtype summary source lineage contract/stage mismatch")
        return {}, []
    exact_bindings = {
        key: {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for key, path in bindings.items()
        if path.is_file()
    }
    for key in bindings:
        if source.get(key) != exact_bindings.get(key):
            add(findings, "lineage", f"dtype summary shape source binding mismatch:{key}")
    bundle = summary.get("source_binding")
    if not isinstance(bundle, Mapping):
        add(findings, "source_bundle", "dtype summary source binding missing")
    else:
        bundle_dir = lane_dir / "source_bundle"
        builder_copy = bundle_dir / BUILDER.name
        expected_builder = {"path": str(builder_copy.resolve()), "sha256": sha256_file(BUILDER)}
        if (
            bundle.get("builder") != expected_builder
            or not builder_copy.is_file()
            or sha256_file(builder_copy) != expected_builder["sha256"]
        ):
            add(findings, "source_bundle", "dtype builder source bundle mismatch")
        declared = bundle.get("dependencies")
        if not isinstance(declared, Mapping) or set(declared) != set(BUILDER_DEPENDENCIES):
            add(findings, "source_bundle", "dtype dependency source bundle catalog mismatch")
        else:
            for name, production in BUILDER_DEPENDENCIES.items():
                copy = bundle_dir / f"{name}_{production.name}"
                expected = {"path": str(copy.resolve()), "sha256": sha256_file(production)}
                if declared.get(name) != expected or not copy.is_file() or sha256_file(copy) != expected["sha256"]:
                    add(findings, "source_bundle", f"dtype dependency source bundle mismatch:{name}")
    shape_rows = (
        pq.read_table(bindings["source_artifact"]).to_pylist() if bindings["source_artifact"].is_file() else []
    )
    shape_manifest = jsonl(bindings["source_manifest"]) if bindings["source_manifest"].is_file() else []
    if source.get("source_rows") != len(shape_rows) or len(shape_rows) != len(shape_manifest):
        add(findings, "lineage", "shape selected rows/manifest/source_rows mismatch")
        return {}, shape_rows
    if (
        summary.get("source_rows") != len(shape_rows)
        or summary.get("eligible_rows") != len(manifests)
        or summary.get("candidate_rows") != len(manifests)
    ):
        add(findings, "lineage", "dtype summary row counts do not bind source and candidates")
    for index, manifest in enumerate(manifests):
        binding = manifest.get("csp_dag_source_binding")
        source_index = manifest.get("source_row_index")
        if (
            not isinstance(binding, Mapping)
            or type(source_index) is not int
            or not 0 <= source_index < len(shape_rows)
        ):
            add(findings, f"candidate:{index}", "manifest csp_dag source binding/index invalid")
            continue
        shape_row, shape_manifest_row = shape_rows[source_index], shape_manifest[source_index]
        try:
            source_uuid, _ = row_identity(shape_row)
        except ValueError:
            add(findings, f"candidate:{index}", "shape parent row identity invalid")
            continue
        if (binding.get("contract"), binding.get("stage")) != (SOURCE_CONTRACT, "shape") or any(
            binding.get(key) != exact_bindings.get(key) for key in bindings
        ):
            add(findings, f"candidate:{index}", "manifest source artifact lineage mismatch")
        if (
            binding.get("source_row_index") != source_index
            or binding.get("source_row_sha256") != canonical_sha256(shape_row)
            or binding.get("source_manifest_row_sha256") != canonical_sha256(shape_manifest_row)
        ):
            add(findings, f"candidate:{index}", "manifest source row/hash lineage mismatch")
        if manifest.get("parent_uuid") != source_uuid:
            add(findings, f"candidate:{index}", "manifest parent UUID differs from shape source row")
        if index >= len(parents) or not dtype_parent_matches_source(parents[index], shape_row):
            add(findings, f"candidate:{index}", "dtype parent row differs from bound shape source row")
    return {key: str(value) for key, value in bindings.items()}, shape_rows


def audit(lane_dir: Path, repo_root: Path, *, require_complete: bool = False) -> dict[str, Any]:
    lane_dir, repo_root = lane_dir.resolve(), repo_root.resolve()
    findings: list[dict[str, str]] = []
    required = {
        name: lane_dir / name
        for name in ("parents.parquet", "candidates.parquet", "manifest.jsonl", "summary.json", "decisions.jsonl")
    }
    for name, path in required.items():
        if not path.is_file():
            add(findings, "lane", f"missing artifact:{name}")
    if findings:
        return report(lane_dir, repo_root, 0, 0, {}, {}, findings, require_complete)
    parents, children, manifests = (
        pq.read_table(required["parents.parquet"]).to_pylist(),
        pq.read_table(required["candidates.parquet"]).to_pylist(),
        jsonl(required["manifest.jsonl"]),
    )
    if not parents or not (len(parents) == len(children) == len(manifests)):
        add(findings, "lane", "parents/candidates/manifest row alignment invalid")
    total = min(len(parents), len(children), len(manifests))
    summary = json.loads(required["summary.json"].read_text(encoding="utf-8"))
    artifacts = summary.get("artifacts") if isinstance(summary, Mapping) else None
    if (
        not isinstance(artifacts, Mapping)
        or summary.get("contract") != EXPANSION_CONTRACT
        or summary.get("stage") != "dtype"
        or summary.get("review_only") is not True
        or summary.get("training_approved") is not False
    ):
        add(findings, "lane", "dtype summary contract/governance invalid")
    for name, path in (
        ("parents", required["parents.parquet"]),
        ("candidates", required["candidates.parquet"]),
        ("manifest", required["manifest.jsonl"]),
    ):
        if not exact_file(artifacts.get(name) if isinstance(artifacts, Mapping) else None, path):
            add(findings, "lane", f"summary artifact binding mismatch:{name}")
    lineage, shape_rows = source_binding_errors(lane_dir, summary, parents, manifests, findings)
    decisions = jsonl(required["decisions.jsonl"])
    source_rows = summary.get("source_rows")
    rejected = collections.Counter()
    eligible_by_candidate: dict[int, Mapping[str, Any]] = {}
    decisions_by_source: dict[int, Mapping[str, Any]] = {}
    if not isinstance(source_rows, int) or source_rows < total or len(decisions) != source_rows:
        add(findings, "lineage", "dtype decisions do not exactly cover source rows")
    for decision in decisions:
        source_index = decision.get("source_row_index")
        if type(source_index) is not int or source_index in decisions_by_source:
            add(findings, "lineage", "dtype decision source row index is invalid or duplicated")
            continue
        decisions_by_source[source_index] = decision
        gate = decision.get("semantic_gate")
        if decision.get("eligible") is True:
            candidate = decision.get("candidate_row_index")
            if type(candidate) is not int or candidate in eligible_by_candidate:
                add(findings, "lineage", "dtype eligible decision candidate index invalid")
            else:
                eligible_by_candidate[candidate] = decision
            if not isinstance(gate, Mapping) or gate.get("status") != "passed" or gate.get("reasons") != []:
                add(findings, "lineage", "dtype eligible decision semantic gate is not passed")
        elif decision.get("eligible") is False:
            reasons = gate.get("reasons") if isinstance(gate, Mapping) else None
            if not isinstance(gate, Mapping) or gate.get("status") != "rejected":
                add(findings, "lineage", "dtype rejected decision semantic gate status invalid")
            if (
                not isinstance(reasons, list)
                or not reasons
                or not all(isinstance(item, str) and item for item in reasons)
            ):
                add(findings, "lineage", "dtype rejected decision semantic gate invalid")
            else:
                rejected[",".join(reasons)] += 1
        else:
            add(findings, "lineage", "dtype decision eligibility invalid")
    if isinstance(source_rows, int) and set(decisions_by_source) != set(range(source_rows)):
        add(findings, "lineage", "dtype decisions source-row set is not exact")
    for source_index, decision in decisions_by_source.items():
        if not 0 <= source_index < len(shape_rows):
            add(findings, "lineage", f"dtype decision source row is outside shape source:{source_index}")
            continue
        try:
            expected_parent_uuid, _ = row_identity(shape_rows[source_index])
        except ValueError:
            add(findings, "lineage", f"dtype decision shape source identity malformed:{source_index}")
            continue
        if decision.get("parent_uuid") != expected_parent_uuid:
            add(findings, "lineage", f"dtype decision parent UUID differs from shape source:{source_index}")
    if set(eligible_by_candidate) != set(range(total)) or any(
        eligible_by_candidate[index].get("parent_uuid") != manifests[index].get("parent_uuid")
        or eligible_by_candidate[index].get("source_row_index") != manifests[index].get("source_row_index")
        or eligible_by_candidate[index].get("semantic_gate") != manifests[index].get("semantic_gate")
        for index in range(total)
    ):
        add(findings, "lineage", "dtype eligible decisions do not exactly bind candidates")
    if (
        summary.get("eligible_rows") != total
        or summary.get("candidate_rows") != total
        or summary.get("semantic_rejection_counts") != dict(sorted(rejected.items()))
    ):
        add(findings, "lineage", "dtype summary eligibility/rejection histogram mismatch")
    meta = [manifest_errors(parents[i], children[i], manifests[i], i, findings) for i in range(total)]
    runtime = lane_dir / "runtime_h20"
    expected_names = {f"shard_{i:03d}_of_004.records.jsonl" for i in range(REQUIRED_SHARDS)}
    flat = set(runtime.glob("*.records.jsonl")) if runtime.is_dir() else set()
    nested = set(runtime.rglob("*.jsonl")) if runtime.is_dir() else set()
    if nested != flat or {path.name for path in flat} - expected_names:
        add(findings, "runtime", "runtime JSONL collection has extra or nested evidence")
    observed, status_counts, unsupported_reason_counts, shards = (
        set(),
        collections.Counter(),
        collections.Counter(),
        {},
    )
    artifact_hashes = {
        "parents_sha256": sha256_file(required["parents.parquet"]),
        "children_sha256": sha256_file(required["candidates.parquet"]),
        "manifest_sha256": sha256_file(required["manifest.jsonl"]),
    }
    source_hashes = {
        "validator_source_sha256": sha256_file(repo_root / VALIDATOR.relative_to(REPO_ROOT)),
        "launcher_source_sha256": sha256_file(repo_root / LAUNCHER.relative_to(REPO_ROOT)),
        "runtime_dependencies": {
            name: {
                "path": str(path.relative_to(REPO_ROOT)),
                "sha256": sha256_file(repo_root / path.relative_to(REPO_ROOT)),
            }
            for name, path in DEPENDENCIES.items()
        },
    }
    for shard in range(REQUIRED_SHARDS):
        path = runtime / f"shard_{shard:03d}_of_004.records.jsonl"
        if not path.is_file():
            add(findings, "runtime", f"missing shard:{path.name}", "P1" if require_complete else "P2")
            continue
        rows = jsonl(path)
        shards[path.name] = len(rows)
        positions = list(range(shard, total, REQUIRED_SHARDS))
        if [row.get("candidate_row_index") for row in rows] != positions[: len(rows)] or (
            require_complete and len(rows) != len(positions)
        ):
            add(findings, f"runtime:{path.name}", "shard positions/order/completeness mismatch")
        for row in rows:
            index = row.get("candidate_row_index")
            scope = f"shard:{shard}:candidate:{index}"
            if (
                type(index) is not int
                or not 0 <= index < total
                or index % REQUIRED_SHARDS != shard
                or index in observed
            ):
                add(findings, scope, "candidate position/duplicate mismatch")
                continue
            observed.add(index)
            status = row.get("status")
            status_counts[str(status)] += 1
            if status == "unsupported" and isinstance(row.get("reason"), str):
                unsupported_reason_counts[
                    row["reason"].split(":", 2)[1] if ":" in row["reason"] else row["reason"]
                ] += 1
            parent_uuid, child_uuid, coherence, factories = meta[index]
            if (
                status not in {"passed", "unsupported"}
                or row.get("passed") is not (status == "passed")
                or (status == "passed" and row.get("reason") is not None)
            ):
                add(findings, scope, "forbidden or incoherent dtype runtime status")
            if row.get("result_binding_version") != RESULT_BINDING_VERSION or row.get(
                "result_payload_sha256"
            ) != payload_sha256(row, "result_payload_sha256"):
                add(findings, scope, "dtype runtime result payload binding mismatch")
            if (
                row.get("parent_uuid") != parent_uuid
                or row.get("child_uuid") != child_uuid
                or row.get("assigned_dtype") != manifests[index].get("assigned_target")
                or row.get("transformed_factory_count") != factories
                or row.get("coherence_class") != coherence
                or (row.get("contract_version"), row.get("binding_version")) != CONTRACTS.get(coherence)
            ):
                add(findings, scope, "record identity/assignment/coherence contract mismatch")
            if (
                any(row.get(key) != value for key, value in {**artifact_hashes, **source_hashes}.items())
                or row.get("allowlist_sha256") is not None
            ):
                add(findings, scope, "runtime source/helper/input SHA binding mismatch")
            if row.get("validation_binding_sha256") != canonical_sha256({key: row.get(key) for key in BINDING_FIELDS}):
                add(findings, scope, "runtime self-binding SHA mismatch")
            if status == "unsupported":
                for message in unsupported_record_errors(row):
                    add(findings, scope, message)
            context = row.get("execution_context")
            expected_context = {
                "device_type": "cuda",
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "cudnn_enabled": True,
            }
            expected_config = {
                "device": f"cuda:{shard}",
                "trials": 3,
                "seed": 17,
                "timeout_seconds": 600.0,
                "max_device_memory_gib": 64.0,
                "float16_rtol": 0.01,
                "float16_atol": 0.01,
                "bfloat16_rtol": 0.02,
                "bfloat16_atol": 0.02,
                "logical_forward_dtype_contract": "all_floating_vN_and_final_outputs_match_assigned_dtype_v1",
                "internal_aten_non_target_floating_outputs": "diagnostic_only",
                "internal_aten_complex_outputs": "reject",
                "cast_equivalent_comparator_contract": FLOAT_COMPARATOR_CONTRACT,
                "nonfloating_comparator_contract": EXACT_COMPARATOR_CONTRACT,
                "execution_context": expected_context,
            }
            if context != expected_context or row.get("validation_config") != expected_config:
                add(findings, scope, "formal dtype validation configuration/context mismatch")
            duration = row.get("duration_seconds")
            if (
                not isinstance(duration, (int, float))
                or isinstance(duration, bool)
                or not math.isfinite(float(duration))
                or duration < 0.0
            ):
                add(findings, scope, "dtype runtime duration evidence invalid")
            memory_guard_errors(row.get("memory_guard"), scope, findings)
            if status == "passed":
                if row.get("target_dtype") != manifests[index].get("assigned_target"):
                    add(findings, scope, "passed record target dtype mismatch")
                gpu = row.get("gpu")
                if (
                    not isinstance(gpu, Mapping)
                    or gpu.get("device") != f"cuda:{shard}"
                    or "h20" not in str(gpu.get("name", "")).lower()
                    or gpu.get("compute_capability") != [9, 0]
                    or not isinstance(gpu.get("torch_version"), str)
                    or not isinstance(gpu.get("torch_cuda_version"), str)
                    or not isinstance(gpu.get("cudnn_version"), int)
                ):
                    add(findings, scope, "CUDA deterministic/H20/configuration evidence mismatch")
                trial_errors(row, manifests[index], coherence, factories, findings, scope)
    if require_complete and (len(observed) != total or set(shards) != expected_names):
        add(findings, "coverage", f"runtime coverage incomplete:{total-len(observed)}")
    elif not require_complete and len(observed) != total:
        add(findings, "coverage", f"runtime records incomplete:{total-len(observed)}", "P2")
    return report(
        lane_dir,
        repo_root,
        total,
        len(observed),
        shards,
        dict(status_counts),
        findings,
        require_complete,
        lineage,
        unsupported_reason_counts,
    )


def report(
    lane: Path,
    repo: Path,
    total: int,
    observed: int,
    shards: Mapping[str, int],
    statuses: Mapping[str, int],
    findings: list[dict[str, str]],
    complete: bool,
    lineage: Mapping[str, str] | None = None,
    unsupported_reasons: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    severity = collections.Counter(item["severity"] for item in findings)
    return {
        "contract": CONTRACT,
        "lane_dir": str(lane),
        "repo_root": str(repo),
        "require_complete": complete,
        "candidate_rows": total,
        "observed_runtime_rows": observed,
        "runtime_shards": dict(sorted(shards.items())),
        "status_counts": dict(sorted(statuses.items())),
        "unsupported_reason_counts": dict(sorted((unsupported_reasons or {}).items())),
        "lineage_inputs": dict(lineage or {}),
        "findings": findings,
        "counts_by_severity": dict(sorted(severity.items())),
        "passed": not any(item["severity"] == "P1" for item in findings),
        "review_only": True,
        "training_approved": False,
    }


def input_bindings(lane: Path, repo_root: Path) -> dict[str, dict[str, str]]:
    paths = [
        lane / name
        for name in ("parents.parquet", "candidates.parquet", "manifest.jsonl", "summary.json", "decisions.jsonl")
    ]
    paths.extend(
        lane.parent / "shape" / name
        for name in ("selected.parquet", "selected.manifest.jsonl", "selection_summary.json")
    )
    paths.extend(
        lane / "source_bundle" / name
        for name in [BUILDER.name, *(f"{key}_{path.name}" for key, path in BUILDER_DEPENDENCIES.items())]
    )
    paths.extend(
        (
            Path(__file__).resolve(),
            Path(__file__).with_name("dtype_runtime_evidence.py").resolve(),
        )
    )
    paths.extend(
        repo_root / path.relative_to(REPO_ROOT)
        for path in (VALIDATOR, LAUNCHER, BUILDER, *DEPENDENCIES.values(), *BUILDER_DEPENDENCIES.values())
    )
    runtime = lane / "runtime_h20"
    if runtime.is_dir():
        paths.extend(sorted(runtime.rglob("*.jsonl")))
    return {
        str(path.resolve()): {"path": str(path.resolve()), "sha256": sha256_file(path)}
        for path in paths
        if path.is_file()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("lane_dir", type=Path)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    result = audit(args.lane_dir, args.repo_root, require_complete=args.require_complete)
    target = args.output or args.lane_dir / "review_runtime_independent" / "audit.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    result["input_bindings"] = input_bindings(args.lane_dir.resolve(), args.repo_root.resolve())
    target.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    sidecar = target.with_suffix(target.suffix + ".sha256")
    sidecar.write_text(f"{sha256_file(target)}  {target.name}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(target.resolve()),
                "sha256": sha256_file(target),
                "sha256_sidecar": str(sidecar.resolve()),
                "passed": result["passed"],
                "counts_by_severity": result["counts_by_severity"],
            },
            sort_keys=True,
        )
    )
    if not result["passed"] or (args.require_complete and result["observed_runtime_rows"] != result["candidate_rows"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
