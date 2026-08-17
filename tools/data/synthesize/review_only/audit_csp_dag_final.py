#!/usr/bin/env python3
"""Independently review the final base -> shape -> dtype CSP-DAG artifact."""

from __future__ import annotations

import argparse
import collections
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from tools.data.synthesize.review_only import csp_dag_review_common, csp_dag_review_runtime, csp_dag_review_sampling
from tools.data.synthesize.review_only.csp_dag_review_common import (
    BASE_MANIFEST_CONTRACT,
    CONTRACT,
    EXPANSION_LANES,
    FINAL_CONTRACT,
    FINALIZATION_SOURCE_FREEZE_INVENTORY_ROWS,
    FINALIZATION_SOURCE_FREEZE_PATHS,
    LANES,
    POST_MODES,
    SOURCE_BINDING_CONTRACT,
    SOURCE_FREEZE_INVENTORY_ROWS,
    SOURCE_FREEZE_PATHS,
    _base_authority_errors,
    _check_source_freeze,
    _exact_file_binding,
    _load_jsonl,
    _manifest_identity_errors,
    _manifest_uuid,
    _resolved_metadata,
    _row_hash,
    _rows_and_manifests,
    _sha256_file,
    _uuid,
)
from tools.data.synthesize.review_only.csp_dag_review_runtime import (
    _check_postselection,
    _preselection,
    _selection_reconstruction,
    _unsupported_reason_histogram,
)
from tools.data.synthesize.review_only.csp_dag_review_sampling import (
    _additive_row_errors,
    _feature,
    _kimi_exact_sample,
    _selected_histogram,
    _stratified_sample,
    _write_kimi_packet,
    _write_samples,
)


def _audit_uncached(
    run_root: Path, base: Path, output: Path, *, sample_count: int, prepare_kimi: bool
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"independent review output already exists:{output}")
    output.mkdir(parents=True)
    errors: list[str] = []
    final_path = run_root / "final_summary.json"
    if not final_path.is_file():
        raise FileNotFoundError(f"finalizer artifact unavailable:{final_path}")
    final = json.loads(final_path.read_text())
    repo_root = Path(__file__).resolve().parents[4]
    runtime_freeze = run_root.parent / "source.freeze.v8.sha256"
    runtime_inventory = run_root.parent / "source_sync.v8.files"
    errors.extend(
        _check_source_freeze(
            runtime_freeze,
            runtime_inventory,
            repo_root,
            expected_paths=SOURCE_FREEZE_PATHS,
            label="runtime",
        )
    )
    finalization_freeze = run_root.parent / "source.freeze.shape_dtype_final.v3.sha256"
    finalization_inventory = run_root.parent / "source_sync.shape_dtype_final.v3.files"
    errors.extend(
        _check_source_freeze(
            finalization_freeze,
            finalization_inventory,
            repo_root,
            expected_paths=FINALIZATION_SOURCE_FREEZE_PATHS,
            label="finalization",
        )
    )
    if final.get("contract") != FINAL_CONTRACT:
        errors.append("final summary contract mismatch")
    if final.get("serial_order") != list(EXPANSION_LANES):
        errors.append("serial lane order mismatch")
    if final.get("review_only") is not True or final.get("training_approved") is not False:
        errors.append("governance mismatch")
    required_final_fields = {
        "contract",
        "serial_order",
        "lanes",
        "excluded_lanes",
        "preselection_runtime_sources",
        "postselection_h20_evidence",
        "source_binding",
        "additive_artifact",
        "accepted_review",
        "review_only",
        "training_approved",
    }
    if set(final) != required_final_fields:
        errors.append("final summary field set mismatch")
    production_finalizer = repo_root / "tools/data/synthesize/csp_dag_method/finalize_shape_dtype_expansion.py"
    shared_finalizer_helpers = repo_root / "tools/data/synthesize/csp_dag_method/csp_dag_finalization.py"
    production_builder = repo_root / "tools/data/synthesize/csp_dag_method/build_csp_dag_input_expansions.py"
    for key, path in (
        ("finalizer", production_finalizer),
        ("shared_finalizer_helpers", shared_finalizer_helpers),
        ("expansion_builder", production_builder),
        ("base_summary", base / "summary.json"),
        ("shape_summary", run_root / "shape/summary.json"),
        ("dtype_summary", run_root / "dtype/summary.json"),
    ):
        error = _exact_file_binding(final.get("source_binding", {}).get(key), path, f"final:{key}")
        if error:
            errors.append(error)
    if set(final.get("source_binding", {})) != {
        "finalizer",
        "shared_finalizer_helpers",
        "expansion_builder",
        "base_summary",
        "shape_summary",
        "dtype_summary",
    }:
        errors.append("final source binding field set mismatch")
    deferred_path = run_root.parent / "layout_deferred.scope.v1.json"
    deferred = final.get("excluded_lanes")
    if not isinstance(deferred, Mapping) or set(deferred) != {"layout"}:
        errors.append("final excluded lane field set mismatch")
    else:
        layout_deferred = deferred.get("layout")
        if (
            not isinstance(layout_deferred, Mapping)
            or set(layout_deferred)
            != {"status", "reason", "evidence", "candidate_rows", "gpu_runtime_rows", "included_in_additive"}
            or layout_deferred.get("status") != "deferred"
            or layout_deferred.get("reason") != "excluded_by_user_scope"
            or layout_deferred.get("candidate_rows") != 0
            or layout_deferred.get("gpu_runtime_rows") != 0
            or layout_deferred.get("included_in_additive") is not False
        ):
            errors.append("final layout deferral contract mismatch")
        else:
            error = _exact_file_binding(layout_deferred.get("evidence"), deferred_path, "final:layout_deferred")
            if error:
                errors.append(error)
    if (run_root / "layout").exists():
        errors.append("layout directory exists in the shape+dtype-only final run")
    for lane in LANES:
        if lane not in final.get("lanes", {}):
            errors.append(f"final lane omitted:{lane}")
    if set(final.get("lanes", {})) != set(LANES):
        errors.append("final lane field set mismatch")
    for key, path in (
        ("parquet", run_root / "selected.additive.parquet"),
        ("manifest", run_root / "selected.additive.manifest.jsonl"),
    ):
        error = _exact_file_binding(final.get("additive_artifact", {}).get(key), path, f"final:additive:{key}")
        if error:
            errors.append(error)
    error = _exact_file_binding(final.get("accepted_review"), run_root / "accepted_review.md", "final:accepted_review")
    if error:
        errors.append(error)

    rows_by_lane: dict[str, list[dict[str, Any]]] = {}
    manifests_by_lane: dict[str, list[dict[str, Any]]] = {}
    selected_paths: dict[str, Path] = {"base": base / "selected.parquet"}
    manifest_paths: dict[str, Path] = {"base": base / "selected.manifest.jsonl"}
    base_summary = json.loads((base / "summary.json").read_text()) if (base / "summary.json").is_file() else {}
    if (
        base_summary.get("contract") != "open_csp_dag_low_level_5k_v4"
        or base_summary.get("review_only") is not True
        or base_summary.get("training_approved") is not False
    ):
        errors.append("base summary contract/governance mismatch")
    for key, path in (("selected", selected_paths["base"]), ("selected_manifest", manifest_paths["base"])):
        error = _exact_file_binding(base_summary.get("artifacts", {}).get(key), path, f"base summary:{key}")
        if error:
            errors.append(error)
    errors.extend(_base_authority_errors(base, base_summary))
    quality_path = base / "quality_audit.json"
    if not quality_path.is_file():
        errors.append("base quality audit missing")
    else:
        quality = json.loads(quality_path.read_text())
        base_rows = pq.ParquetFile(selected_paths["base"]).metadata.num_rows
        if (
            quality.get("contract") != "csp_dag_static_quality_audit_v1"
            or quality.get("review_only") is not True
            or quality.get("training_approved") is not False
            or quality.get("training_gate_passed") is not True
            or quality.get("gate_failures") != []
            or quality.get("rows") != {"parquet": base_rows, "manifest": base_rows}
            or quality.get("row_binding_errors") != []
            or quality.get("graph_source_binding", {}).get("failed_rows") != 0
            or quality.get("graph_source_binding_failures") != []
        ):
            errors.append("base quality audit contract/governance/static result mismatch")
        quality_source = quality.get("source_binding", {})
        if (
            quality_source.get("binding_errors") != []
            or quality_source.get("summary_artifacts_bound") is not True
            or quality_source.get("run_dir") != str(base.resolve())
        ):
            errors.append("base quality audit source binding errors")
        for key, path in (
            ("parquet", selected_paths["base"]),
            ("manifest", manifest_paths["base"]),
            ("summary", base / "summary.json"),
        ):
            error = _exact_file_binding(quality_source.get(key), path, f"base quality:{key}")
            if error:
                errors.append(error)
    preselection: dict[str, tuple[list[int], list[Path]]] = {}
    for lane in EXPANSION_LANES:
        selected_paths[lane] = run_root / lane / "selected.parquet"
        manifest_paths[lane] = run_root / lane / "selected.manifest.jsonl"
    candidate_data: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]] | None]] = {}
    for lane in EXPANSION_LANES:
        candidates, candidate_manifests = _rows_and_manifests(
            run_root / lane / "candidates.parquet", run_root / lane / "manifest.jsonl"
        )
        parents = pq.read_table(run_root / lane / "parents.parquet").to_pylist() if lane != "shape" else None
        candidate_data[lane] = (candidates, candidate_manifests, parents)
        passed, paths = _preselection(run_root, lane, candidates, errors)
        preselection[lane] = (passed, paths)
        _selection_reconstruction(run_root, lane, candidates, candidate_manifests, passed, paths, errors, parents)
        lane_errors, _ = _check_postselection(
            run_root, lane, selected_paths[lane], manifest_paths[lane], run_root / lane / "selection_summary.json"
        )
        errors.extend(lane_errors)
    expected_pre = {
        lane: [{"path": str(path.resolve()), "sha256": _sha256_file(path)} for path in preselection[lane][1]]
        for lane in EXPANSION_LANES
    }
    if final.get("preselection_runtime_sources") != expected_pre:
        errors.append("final preselection runtime evidence anchor mismatch")
    post_evidence = final.get("postselection_h20_evidence")
    if not isinstance(post_evidence, Mapping) or set(post_evidence) != set(EXPANSION_LANES):
        errors.append("final postselection evidence lanes mismatch")
    else:
        for lane in EXPANSION_LANES:
            runtime_dir = run_root / lane / "runtime_h20_post_selection"
            expected_records = (
                [
                    {
                        "path": str((runtime_dir / f"shard_{index:03d}_of_004.records.jsonl").resolve()),
                        "sha256": _sha256_file(runtime_dir / f"shard_{index:03d}_of_004.records.jsonl"),
                    }
                    for index in range(4)
                ]
                if runtime_dir.is_dir()
                else []
            )
            expected_summaries = (
                [
                    {
                        "path": str((runtime_dir / f"shard_{index:03d}_of_004.summary.json").resolve()),
                        "sha256": _sha256_file(runtime_dir / f"shard_{index:03d}_of_004.summary.json"),
                    }
                    for index in range(4)
                ]
                if runtime_dir.is_dir()
                else []
            )
            evidence = post_evidence.get(lane, {})
            if (
                evidence.get("records") != expected_records
                or evidence.get("summaries") != expected_summaries
                or evidence.get("rows") != final.get("lanes", {}).get(lane, {}).get("selected")
                or evidence.get("status_counts") != {"passed": evidence.get("rows")}
            ):
                errors.append(f"final postselection evidence anchor mismatch:{lane}")
    for lane in LANES:
        rows_by_lane[lane], manifests_by_lane[lane] = _rows_and_manifests(selected_paths[lane], manifest_paths[lane])
        declared = final.get("lanes", {}).get(lane, {}).get("selected")
        if declared != len(rows_by_lane[lane]):
            errors.append(f"final summary selected count mismatch:{lane}")
        lane_final = final.get("lanes", {}).get(lane, {})
        expected_lane_fields = (
            {"selected", "artifact"}
            if lane == "base"
            else {"candidates", "selected", "selected_histogram", "selection_summary", "selected_artifact"}
        ) | ({"selected_parents"} if lane == "dtype" else set())
        if set(lane_final) != expected_lane_fields:
            errors.append(f"final lane field set mismatch:{lane}")
        if lane != "base":
            histogram = _selected_histogram(lane, manifests_by_lane[lane])
            if (
                lane_final.get("candidates") != len(candidate_data[lane][0])
                or lane_final.get("selected_histogram") != histogram
            ):
                errors.append(f"final lane candidate/histogram mismatch:{lane}")
            selection_binding = _exact_file_binding(
                lane_final.get("selection_summary"),
                run_root / lane / "selection_summary.json",
                f"final:{lane}:selection_summary",
            )
            if selection_binding:
                errors.append(selection_binding)
        error = _exact_file_binding(
            lane_final.get("artifact", lane_final.get("selected_artifact")),
            selected_paths[lane],
            f"final:{lane}:selected",
        )
        if error:
            errors.append(error)
        if lane == "dtype":
            error = _exact_file_binding(
                lane_final.get("selected_parents"),
                run_root / lane / "selected.parents.parquet",
                f"final:{lane}:selected_parents",
            )
            if error:
                errors.append(error)

    root_by_uuid: dict[str, str] = {_uuid(row): _uuid(row) for row in rows_by_lane["base"]}
    items: list[dict[str, Any]] = []
    lineage_errors: list[str] = []
    metadata_by_uuid: dict[str, dict[str, Any]] = {}
    for lane in LANES:
        for lane_index, (row, manifest) in enumerate(zip(rows_by_lane[lane], manifests_by_lane[lane], strict=True)):
            uuid = _uuid(row)
            manifest_uuid = _manifest_uuid(manifest)
            if uuid != manifest_uuid:
                lineage_errors.append(f"{lane}: row/manifest UUID mismatch:{uuid}")
            lineage_errors.extend(_manifest_identity_errors(lane, row, manifest, lane_index))
            if lane == "base":
                root = uuid
                resolved = _resolved_metadata(manifest, None)
                if manifest.get("contract") != BASE_MANIFEST_CONTRACT:
                    lineage_errors.append(f"base: manifest contract mismatch:{lane_index}")
            else:
                parent = manifest.get("parent_uuid")
                if parent not in root_by_uuid:
                    lineage_errors.append(f"{lane}: absent serial parent:{uuid}:{parent}")
                    root = uuid
                    resolved = _resolved_metadata(manifest, None)
                else:
                    root = root_by_uuid[parent]
                    resolved = _resolved_metadata(manifest, metadata_by_uuid.get(parent))
                binding = manifest.get("csp_dag_source_binding", {})
                expected_stage = {"shape": "base", "dtype": "shape"}[lane]
                if binding.get("contract") != SOURCE_BINDING_CONTRACT or binding.get("stage") != expected_stage:
                    lineage_errors.append(f"{lane}: lineage stage mismatch:{uuid}")
                parent_lane = {"shape": "base", "dtype": "shape"}[lane]
                expected_summary = (
                    base / "summary.json"
                    if parent_lane == "base"
                    else run_root / parent_lane / "selection_summary.json"
                )
                for key, expected in (
                    ("source_artifact", selected_paths[parent_lane]),
                    ("source_manifest", manifest_paths[parent_lane]),
                    ("source_summary", expected_summary),
                ):
                    declared = binding.get(key, {})
                    if declared.get("path") != str(expected.resolve()) or declared.get("sha256") != _sha256_file(
                        expected
                    ):
                        lineage_errors.append(f"{lane}: lineage artifact binding mismatch:{uuid}:{key}")
                source_index = binding.get("source_row_index")
                parent_rows = rows_by_lane[parent_lane]
                parent_manifests = manifests_by_lane[parent_lane]
                if not isinstance(source_index, int) or not 0 <= source_index < len(parent_rows):
                    lineage_errors.append(f"{lane}: lineage source row index invalid:{uuid}")
                else:
                    source_parent = parent_rows[source_index]
                    source_parent_manifest = parent_manifests[source_index]
                    if _uuid(source_parent) != parent:
                        lineage_errors.append(f"{lane}: lineage direct parent/index mismatch:{uuid}")
                    if binding.get("source_rows") != len(parent_rows):
                        lineage_errors.append(f"{lane}: lineage source row count mismatch:{uuid}")
                    if binding.get("source_row_sha256") != _row_hash(source_parent):
                        lineage_errors.append(f"{lane}: lineage source row SHA-256 mismatch:{uuid}")
                    if binding.get("source_manifest_row_sha256") != _row_hash(source_parent_manifest):
                        lineage_errors.append(f"{lane}: lineage source manifest row SHA-256 mismatch:{uuid}")
            root_by_uuid[uuid] = root
            metadata_by_uuid[uuid] = resolved
            item = _feature(lane, row, resolved, root)
            item["lane_row_index"] = lane_index
            item["source_manifest_row_sha256"] = _row_hash(manifest)
            items.append(item)
    errors.extend(lineage_errors)

    uuids = collections.Counter(item["uuid"] for item in items)
    refs = collections.Counter(item["reference_sha256"] for item in items)
    errors.extend(f"global UUID collision:{key}" for key, value in uuids.items() if value > 1)
    errors.extend(f"global exact reference collision:{key}" for key, value in refs.items() if value > 1)
    ast_groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for item in items:
        ast_groups[item["normalized_ast_sha256"]].append(item)
    intentional: list[dict[str, Any]] = []
    cross_root_ast: list[dict[str, Any]] = []
    for digest, group in ast_groups.items():
        if len(group) <= 1:
            continue
        roots = {item["root_base_uuid"] for item in group}
        lanes = [item["lane"] for item in group]
        record = {
            "normalized_ast_sha256": digest,
            "rows": len(group),
            "roots": sorted(roots),
            "lanes": sorted(lanes),
            "uuids": sorted(item["uuid"] for item in group),
        }
        if len(roots) == 1 and len(set(lanes)) == len(lanes):
            intentional.append(record)
        else:
            cross_root_ast.append(record)
            errors.append(f"cross-root or repeated-lane normalized-AST collision:{digest}")

    additive_manifest = run_root / "selected.additive.manifest.jsonl"
    additive = run_root / "selected.additive.parquet"
    lineage = _load_jsonl(additive_manifest)
    additive_rows = pq.read_table(additive).to_pylist()
    if len(lineage) != len(additive_rows) or len(additive_rows) != len(items):
        errors.append("additive row conservation mismatch")
    if final.get("additive_artifact", {}).get("rows") != len(additive_rows):
        errors.append("final summary additive count mismatch")
    expected_lane_counts = {lane: len(rows_by_lane[lane]) for lane in LANES}
    if (
        set(final.get("additive_artifact", {})) != {"rows", "lane_counts", "parquet", "manifest"}
        or final.get("additive_artifact", {}).get("lane_counts") != expected_lane_counts
    ):
        errors.append("final summary additive fields/lane counts mismatch")
    metadata = pq.ParquetFile(additive).schema_arrow.metadata or {}
    if (
        metadata.get(b"csp_dag.additive_contract") != FINAL_CONTRACT.encode()
        or metadata.get(b"csp_dag.review_only") != b"true"
        or metadata.get(b"csp_dag.training_approved") != b"false"
    ):
        errors.append("additive parquet governance schema metadata mismatch")
    if len(lineage) == len(additive_rows) == len(items):
        additive_schema = pq.ParquetFile(additive).schema_arrow
        for index, (entry, additive_row, item) in enumerate(zip(lineage, additive_rows, items, strict=True)):
            lane = item["lane"]
            source_row = rows_by_lane[lane][item["lane_row_index"]]
            source_manifest = manifests_by_lane[lane][item["lane_row_index"]]
            row_errors = _additive_row_errors(
                entry,
                additive_row,
                source_row,
                source_manifest,
                index=index,
                lane=lane,
                lane_index=item["lane_row_index"],
                root=item["root_base_uuid"],
                selected=selected_paths[lane],
                selected_manifest=manifest_paths[lane],
                additive_schema=additive_schema,
            )
            errors.extend(row_errors)
            if row_errors:
                break

    sample, sample_gaps = _stratified_sample(items, sample_count)
    if sample_gaps:
        errors.extend(f"sample coverage gap:{gap}" for gap in sample_gaps)
    samples = _write_samples(output, sample, name="stratified_sample")
    near_dedup_path = base / "near_dedup" / "retained.summary.json"
    near_dedup: dict[str, Any]
    if not near_dedup_path.is_file():
        errors.append(f"missing base retained near-dedup summary:{near_dedup_path}")
        near_dedup = {"path": str(near_dedup_path.resolve()), "missing": True}
    else:
        try:
            near_payload = json.loads(near_dedup_path.read_text())
            strict = near_payload.get("strict_recommended_union", {})
            expected_rows = len(rows_by_lane["base"])
            if near_payload.get("contract") != "open_csp_dag_near_duplicate_audit_v2":
                errors.append("base retained near-dedup contract mismatch")
            near_artifacts = near_payload.get("artifacts", {})
            near_paths = {
                "pairs": base / "near_dedup/retained.pairs.jsonl",
                "strict_clusters": base / "near_dedup/retained.strict_clusters.jsonl",
                "strict_retained_uuids": base / "near_dedup/retained.strict_retained_uuids.jsonl",
            }
            if not isinstance(near_artifacts, Mapping) or set(near_artifacts) != set(near_paths):
                errors.append("base retained near-dedup artifact field set mismatch")
            for name, path in near_paths.items():
                error = _exact_file_binding(near_artifacts.get(name), path, f"near-dedup:{name}")
                if error:
                    errors.append(error)
            retained_uuid_rows = _load_jsonl(near_paths["strict_retained_uuids"])
            expected_retained = [
                {"candidate_index": manifest.get("candidate_index"), "position": position, "uuid": _uuid(row)}
                for position, (row, manifest) in enumerate(
                    zip(rows_by_lane["base"], manifests_by_lane["base"], strict=True)
                )
            ]
            if (
                retained_uuid_rows != expected_retained
                or near_artifacts.get("strict_retained_uuids", {}).get("rows") != expected_rows
            ):
                errors.append("base retained near-dedup UUID authority differs from selected base rows")
            strict_cluster_rows = _load_jsonl(near_paths["strict_clusters"])
            if strict_cluster_rows:
                errors.append("base retained near-dedup strict cluster artifact is nonempty")
            pair_rows = _load_jsonl(near_paths["pairs"])
            top_pairs = near_payload.get("approximate_graph_sensitivity", {}).get("top_pairs")
            normalized_pairs = [{key: value for key, value in pair.items() if key != "kind"} for pair in pair_rows]
            if not isinstance(top_pairs, list) or normalized_pairs != top_pairs:
                errors.append("base retained near-dedup pairs artifact/top-pairs mismatch")
            base_uuid_by_position = [_uuid(row) for row in rows_by_lane["base"]]
            for pair_index, pair in enumerate(pair_rows):
                left, right = pair.get("left_position"), pair.get("right_position")
                if (
                    pair.get("kind") != "approximate_graph"
                    or type(left) is not int
                    or type(right) is not int
                    or not 0 <= left < expected_rows
                    or not 0 <= right < expected_rows
                    or pair.get("left_uuid") != base_uuid_by_position[left]
                    or pair.get("right_uuid") != base_uuid_by_position[right]
                ):
                    errors.append(f"base retained near-dedup pair position/UUID mismatch:{pair_index}")
                    break
            if (
                strict.get("pair_count") != 0
                or strict.get("cluster_count") != 0
                or strict.get("clustered_rows") != 0
                or strict.get("effective_rows") != expected_rows
            ):
                errors.append("base retained near-dedup strict result is not zero-collision")
            near_dedup = {
                "path": str(near_dedup_path.resolve()),
                "sha256": _sha256_file(near_dedup_path),
                "strict_recommended_union": {
                    key: strict.get(key) for key in ("pair_count", "cluster_count", "clustered_rows", "effective_rows")
                },
            }
        except (json.JSONDecodeError, OSError) as error:
            errors.append(f"unreadable base retained near-dedup summary:{error}")
            near_dedup = {"path": str(near_dedup_path.resolve()), "unreadable": True}
    result: dict[str, Any] = {
        "contract": CONTRACT,
        "review_source": {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in {
                "audit": Path(__file__).resolve(),
                "common": Path(csp_dag_review_common.__file__).resolve(),
                "runtime": Path(csp_dag_review_runtime.__file__).resolve(),
                "sampling": Path(csp_dag_review_sampling.__file__).resolve(),
            }.items()
        },
        "final_summary": {"path": str(final_path.resolve()), "sha256": _sha256_file(final_path)},
        "source_freeze": {
            "runtime": {
                "path": str(runtime_freeze.resolve()),
                "sha256": _sha256_file(runtime_freeze) if runtime_freeze.is_file() else None,
                "inventory": {
                    "path": str(runtime_inventory.resolve()),
                    "sha256": _sha256_file(runtime_inventory) if runtime_inventory.is_file() else None,
                    "rows": SOURCE_FREEZE_INVENTORY_ROWS,
                },
            },
            "finalization": {
                "path": str(finalization_freeze.resolve()),
                "sha256": _sha256_file(finalization_freeze) if finalization_freeze.is_file() else None,
                "inventory": {
                    "path": str(finalization_inventory.resolve()),
                    "sha256": _sha256_file(finalization_inventory) if finalization_inventory.is_file() else None,
                    "rows": FINALIZATION_SOURCE_FREEZE_INVENTORY_ROWS,
                },
            },
        },
        "lane_rows": {lane: len(rows_by_lane[lane]) for lane in LANES},
        "preselection_unsupported_reason_histograms": {
            "dtype": _unsupported_reason_histogram(preselection["dtype"][1])
        },
        "postselection": {lane: {"required_shards": 4, "checked": lane in POST_MODES} for lane in LANES},
        "identity": {
            "unique_uuids": len(uuids),
            "unique_references": len(refs),
            "normalized_ast_groups": len(ast_groups),
        },
        "lineage": {
            "serial_order": list(LANES),
            "additive_rows": len(additive_rows),
            "intentional_same_root_cross_lane_ast_groups": intentional,
            "cross_root_or_repeated_lane_ast_groups": cross_root_ast,
        },
        "near_dedup_reporting": {
            "base_audit": near_dedup,
            "interpretation": "base strict-near-dedup measures independently synthesized roots; child sibling collision groups are reported separately and are never collapsed into a generative diversity rate",
            "cross_root_exact_ast_is_p1": True,
        },
        "stratified_sample": samples,
        "p1_errors": errors,
        "passed": not errors,
        "review_only": True,
        "training_approved": False,
    }
    if prepare_kimi:
        kimi, kimi_gaps = _kimi_exact_sample(items, near_payload if "near_payload" in locals() else {})
        if kimi_gaps or len(kimi) != 28:
            result["p1_errors"].extend(f"Kimi 28-row coverage gap:{gap}" for gap in kimi_gaps)
            if len(kimi) != 28:
                result["p1_errors"].append(f"Kimi exact 28-row packet unavailable:{len(kimi)}")
            result["passed"] = False
        else:
            result["kimi_k3_packet"] = _write_kimi_packet(output, kimi)
    summary = output / "summary.json"
    summary.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return {**result, "artifacts": {"summary": {"path": str(summary.resolve()), "sha256": _sha256_file(summary)}}}


def audit(
    run_root: Path,
    base: Path,
    output: Path,
    *,
    sample_count: int,
    prepare_kimi: bool,
) -> dict[str, Any]:
    return _audit_uncached(
        run_root,
        base,
        output,
        sample_count=sample_count,
        prepare_kimi=prepare_kimi,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--base", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--prepare-kimi", action="store_true")
    args = parser.parse_args()
    if args.sample_count < 1:
        raise ValueError("sample count must be positive")
    result = audit(
        args.run_root, args.base, args.output_dir, sample_count=args.sample_count, prepare_kimi=args.prepare_kimi
    )
    print(
        json.dumps(
            {"passed": result["passed"], "p1_errors": result["p1_errors"], "summary": result["artifacts"]["summary"]},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
