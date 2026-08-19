#!/usr/bin/env python3
"""Select one deterministic review question for each CSP-DAG root parent."""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


CONTRACT = "csp_dag_one_question_per_parent_sample_v1"
SELECTION_POLICY = "sha256_mod_available_lanes_v1"
FINAL_CONTRACT = "csp_dag_shape_dtype_expansion_final_v1"
LANE_ORDER = ("base", "shape", "dtype")
NODE_BUCKETS = (
    (15, "01_to_15"),
    (31, "16_to_31"),
    (63, "32_to_63"),
    (95, "64_to_95"),
    (None, "96_plus"),
)
OPERATOR_BUCKETS = (
    (15, "01_to_15"),
    (31, "16_to_31"),
    (63, "32_to_63"),
    (127, "64_to_127"),
    (None, "128_plus"),
)
CODE_LINE_BUCKETS = (
    (39, "01_to_39"),
    (79, "40_to_79"),
    (119, "80_to_119"),
    (159, "120_to_159"),
    (None, "160_plus"),
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"non-object JSONL row: {path}")
    return rows


def _row_uuid(row: Mapping[str, Any]) -> str:
    extra = row.get("extra_info")
    uuid = extra.get("uuid") if isinstance(extra, Mapping) else None
    if not isinstance(uuid, str) or not uuid:
        raise ValueError("row lacks extra_info.uuid")
    return uuid


def _reference_code(row: Mapping[str, Any]) -> str:
    reward = row.get("reward_model")
    code = reward.get("ground_truth") if isinstance(reward, Mapping) else None
    if not isinstance(code, str) or not code:
        raise ValueError(f"row lacks reward_model.ground_truth: {_row_uuid(row)}")
    return code


def _reference_sha256(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(_reference_code(row).encode("utf-8")).hexdigest()


def _normalized_ast_sha256(row: Mapping[str, Any]) -> str:
    normalized = ast.dump(ast.parse(_reference_code(row)), annotate_fields=True, include_attributes=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _bucket(value: int, bounds: Sequence[tuple[int | None, str]]) -> str:
    for upper, label in bounds:
        if upper is None or value <= upper:
            return label
    raise AssertionError("unreachable")


def _artifact(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path.resolve()), "sha256": _file_sha256(path)}
    if rows is not None:
        result["rows"] = rows
    return result


def _staged_artifact(path: Path, final_path: Path, *, rows: int) -> dict[str, Any]:
    result = _artifact(path, rows=rows)
    result["path"] = str(final_path.resolve())
    return result


def _declared_artifact(summary: Mapping[str, Any], key: str, actual: Path) -> None:
    additive = summary.get("additive_artifact")
    declared = additive.get(key) if isinstance(additive, Mapping) else None
    if not isinstance(declared, Mapping):
        raise ValueError(f"final summary lacks additive_artifact.{key}")
    declared_path = declared.get("path")
    if not isinstance(declared_path, str) or Path(declared_path).resolve() != actual.resolve():
        raise ValueError(f"final summary {key} path mismatch")
    if declared.get("sha256") != _file_sha256(actual):
        raise ValueError(f"final summary {key} SHA mismatch")


def _source_manifests(
    additive_manifests: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[Path, list[dict[str, Any]]]]:
    result: dict[str, tuple[Path, list[dict[str, Any]]]] = {}
    for lane in LANE_ORDER:
        lane_rows = [row for row in additive_manifests if row.get("lane") == lane]
        if not lane_rows:
            raise ValueError(f"additive manifest lacks lane: {lane}")
        declarations = {
            (row.get("source_manifest") or {}).get("path"): (row.get("source_manifest") or {}).get("sha256")
            for row in lane_rows
            if isinstance(row.get("source_manifest"), Mapping)
        }
        if len(declarations) != 1:
            raise ValueError(f"{lane} source manifest declaration is not exact")
        raw_path, declared_sha = next(iter(declarations.items()))
        if not isinstance(raw_path, str) or not isinstance(declared_sha, str):
            raise ValueError(f"{lane} source manifest declaration is malformed")
        path = Path(raw_path).resolve()
        if not path.is_file() or _file_sha256(path) != declared_sha:
            raise ValueError(f"{lane} source manifest binding mismatch")
        rows = _read_jsonl(path)
        if len(rows) != len(lane_rows):
            raise ValueError(f"{lane} source/additive manifest row mismatch")
        for index, (additive, source) in enumerate(zip(lane_rows, rows, strict=True)):
            if additive.get("lane_row_index") != index:
                raise ValueError(f"{lane} lane_row_index mismatch: {index}")
            if additive.get("source_manifest_row_sha256") != _canonical_sha256(source):
                raise ValueError(f"{lane} source manifest row binding mismatch: {index}")
        result[lane] = (path, rows)
    return result


def _validate_and_group(
    rows: Sequence[Mapping[str, Any]],
    manifests: Sequence[Mapping[str, Any]],
) -> tuple[list[str], dict[str, dict[str, tuple[int, Mapping[str, Any], Mapping[str, Any]]]]]:
    if len(rows) != len(manifests) or not rows:
        raise ValueError("additive parquet/manifest row counts differ or are empty")
    groups: dict[str, dict[str, tuple[int, Mapping[str, Any], Mapping[str, Any]]]] = {}
    root_order: list[str] = []
    uuids: list[str] = []
    references: list[str] = []
    ast_hashes: list[str] = []
    for index, (row, manifest) in enumerate(zip(rows, manifests, strict=True)):
        if manifest.get("additive_row_index") != index:
            raise ValueError(f"additive row index mismatch: {index}")
        lane = manifest.get("lane")
        root = manifest.get("root_base_uuid")
        if lane not in LANE_ORDER or not isinstance(root, str) or not root:
            raise ValueError(f"invalid additive lane/root: {index}")
        uuid = _row_uuid(row)
        reference_sha = _reference_sha256(row)
        ast_sha = _normalized_ast_sha256(row)
        if (
            manifest.get("uuid") != uuid
            or manifest.get("reference_sha256") != reference_sha
            or manifest.get("normalized_ast_sha256") != ast_sha
        ):
            raise ValueError(f"additive row identity mismatch: {index}")
        if root not in groups:
            groups[root] = {}
            root_order.append(root)
        if lane in groups[root]:
            raise ValueError(f"duplicate lane for root {root}: {lane}")
        groups[root][lane] = (index, row, manifest)
        uuids.append(uuid)
        references.append(reference_sha)
        ast_hashes.append(ast_sha)
    if not (len(uuids) == len(set(uuids)) == len(set(references)) == len(set(ast_hashes))):
        raise ValueError("additive UUID/reference/normalized-AST identity is not globally unique")
    for root, lanes in groups.items():
        lane_set = set(lanes)
        if lane_set not in ({"base", "shape"}, {"base", "shape", "dtype"}):
            raise ValueError(f"unexpected available lanes for root {root}: {sorted(lane_set)}")
        base = lanes["base"][2]
        shape = lanes["shape"][2]
        if base.get("uuid") != root or base.get("parent_uuid") is not None:
            raise ValueError(f"invalid base lineage: {root}")
        if shape.get("parent_uuid") != root:
            raise ValueError(f"invalid shape lineage: {root}")
        if "dtype" in lanes and lanes["dtype"][2].get("parent_uuid") != shape.get("uuid"):
            raise ValueError(f"invalid dtype lineage: {root}")
    return root_order, groups


def _selection(root: str, available: Sequence[str]) -> tuple[str, str, int]:
    digest = hashlib.sha256(f"{CONTRACT}\0{root}".encode()).hexdigest()
    slot = int(digest, 16) % len(available)
    return available[slot], digest, slot


def _topology_analysis(base_manifests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    families: collections.Counter[str] = collections.Counter()
    family_cardinality: collections.Counter[str] = collections.Counter()
    nodes: collections.Counter[str] = collections.Counter()
    operators: collections.Counter[str] = collections.Counter()
    code_lines: collections.Counter[str] = collections.Counter()
    input_modes: collections.Counter[str] = collections.Counter()
    multiclass: collections.Counter[str] = collections.Counter()
    for manifest in base_manifests:
        actual = manifest.get("actual_low_level_families")
        graph = manifest.get("graph")
        code = manifest.get("code_metrics")
        signature = manifest.get("operator_signature")
        if (
            not isinstance(actual, list)
            or not all(isinstance(item, str) for item in actual)
            or not isinstance(graph, Mapping)
            or type(graph.get("node_count")) is not int
            or not isinstance(code, Mapping)
            or type(code.get("physical_line_count")) is not int
            or not isinstance(signature, list)
        ):
            raise ValueError("base topology manifest is malformed")
        families.update(actual)
        family_cardinality[str(len(actual))] += 1
        nodes[_bucket(graph["node_count"], NODE_BUCKETS)] += 1
        operators[_bucket(len(signature), OPERATOR_BUCKETS)] += 1
        code_lines[_bucket(code["physical_line_count"], CODE_LINE_BUCKETS)] += 1
        input_modes[str(manifest.get("input_mode"))] += 1
        multiclass[str(manifest.get("semantic_multiclass_status"))] += 1
    return {
        "measurement": "one row per root, projected from the immutable base topology manifest",
        "family_presence_multilabel": dict(sorted(families.items())),
        "family_cardinality": dict(sorted(family_cardinality.items())),
        "graph_node_count_buckets": dict(sorted(nodes.items())),
        "operator_signature_count_buckets": dict(sorted(operators.items())),
        "physical_code_line_buckets": dict(sorted(code_lines.items())),
        "input_modes": dict(sorted(input_modes.items())),
        "semantic_multiclass_status": dict(sorted(multiclass.items())),
    }


def _review_entries(
    selected: Sequence[Mapping[str, Any]],
    *,
    sample_count: int,
) -> list[Mapping[str, Any]]:
    by_cell: dict[tuple[str, str], list[Mapping[str, Any]]] = collections.defaultdict(list)
    for item in selected:
        by_cell[(str(item["selected_lane"]), str(item["node_bucket"]))].append(item)
    chosen: list[Mapping[str, Any]] = []
    seen: set[str] = set()
    for lane in LANE_ORDER:
        for _, bucket in NODE_BUCKETS:
            candidates = by_cell.get((lane, bucket), [])
            if not candidates:
                continue
            item = min(candidates, key=lambda value: str(value["review_sha256"]))
            uuid = str(item["uuid"])
            if uuid not in seen:
                chosen.append(item)
                seen.add(uuid)
    remaining = sorted(selected, key=lambda value: str(value["review_sha256"]))
    for item in remaining:
        if len(chosen) >= sample_count:
            break
        uuid = str(item["uuid"])
        if uuid not in seen:
            chosen.append(item)
            seen.add(uuid)
    return chosen[:sample_count]


def _write_review(path: Path, entries: Sequence[Mapping[str, Any]]) -> None:
    chunks = [
        "# One-question-per-parent review samples\n",
        "These examples are selected deterministically across lane and graph-node buckets.\n",
    ]
    for ordinal, item in enumerate(entries, 1):
        chunks.extend(
            [
                f"\n## {ordinal:02d}. {item['uuid']}\n",
                f"- root: `{item['root_base_uuid']}`\n",
                f"- lane: `{item['selected_lane']}`\n",
                f"- available lanes: `{','.join(item['available_lanes'])}`\n",
                f"- graph nodes: `{item['node_count']}`; operator signature entries: `{item['operator_count']}`\n",
                f"- low-level families: `{','.join(item['families'])}`\n",
                "\n### Prompt\n\n```json\n",
                json.dumps(item["row"]["prompt"], indent=2, ensure_ascii=False),
                "\n```\n\n### Reference program\n\n```python\n",
                _reference_code(item["row"]),
                "\n```\n",
            ]
        )
    path.write_text("".join(chunks), encoding="utf-8")


def sample_one_question_per_parent(
    *,
    source_parquet: Path,
    source_manifest: Path,
    final_summary: Path,
    output_dir: Path,
    review_sample_count: int = 15,
) -> dict[str, Any]:
    source_parquet = source_parquet.resolve()
    source_manifest = source_manifest.resolve()
    final_summary = final_summary.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if review_sample_count <= 0:
        raise ValueError("review_sample_count must be positive")
    final = json.loads(final_summary.read_text(encoding="utf-8"))
    if (
        final.get("contract") != FINAL_CONTRACT
        or final.get("review_only") is not True
        or final.get("training_approved") is not False
    ):
        raise ValueError("final summary contract/governance mismatch")
    _declared_artifact(final, "parquet", source_parquet)
    _declared_artifact(final, "manifest", source_manifest)

    table = pq.read_table(source_parquet)
    rows = table.to_pylist()
    manifests = _read_jsonl(source_manifest)
    additive = final.get("additive_artifact")
    if not isinstance(additive, Mapping) or additive.get("rows") != len(rows):
        raise ValueError("final summary additive row count mismatch")
    root_order, groups = _validate_and_group(rows, manifests)
    sources = _source_manifests(manifests)
    base_manifests = sources["base"][1]
    base_by_uuid = {str(row.get("uuid")): row for row in base_manifests}
    shape_by_uuid = {str(row.get("uuid")): row for row in sources["shape"][1]}
    dtype_by_uuid = {str(row.get("child_uuid")): row for row in sources["dtype"][1]}
    if set(base_by_uuid) != set(root_order):
        raise ValueError("base source manifest roots do not match additive roots")

    selected_rows: list[Mapping[str, Any]] = []
    selected_manifests: list[dict[str, Any]] = []
    review_metadata: list[dict[str, Any]] = []
    lane_counts: collections.Counter[str] = collections.Counter()
    availability_counts: collections.Counter[str] = collections.Counter()
    shape_targets: collections.Counter[str] = collections.Counter()
    dtype_targets: collections.Counter[str] = collections.Counter()
    for output_index, root in enumerate(root_order):
        lanes = groups[root]
        available = tuple(lane for lane in LANE_ORDER if lane in lanes)
        chosen_lane, digest, slot = _selection(root, available)
        additive_index, row, source = lanes[chosen_lane]
        base_manifest = base_by_uuid[root]
        graph = base_manifest["graph"]
        families = base_manifest["actual_low_level_families"]
        operator_signature = base_manifest["operator_signature"]
        node_count = int(graph["node_count"])
        lane_counts[chosen_lane] += 1
        availability_counts["+".join(available)] += 1
        if chosen_lane == "shape":
            shape_manifest = shape_by_uuid.get(str(source.get("uuid")))
            intervention = shape_manifest.get("shape_intervention") if isinstance(shape_manifest, Mapping) else None
            target = intervention.get("assigned_target_value") if isinstance(intervention, Mapping) else None
            if type(target) is not int:
                raise ValueError(f"selected shape target is unavailable: {source.get('uuid')}")
            shape_targets[str(target)] += 1
        if chosen_lane == "dtype":
            dtype_manifest = dtype_by_uuid.get(str(source.get("uuid")))
            target = dtype_manifest.get("assigned_target") if isinstance(dtype_manifest, Mapping) else None
            if target not in {"float16", "bfloat16"}:
                raise ValueError(f"selected dtype target is unavailable: {source.get('uuid')}")
            dtype_targets[str(target)] += 1
        output_manifest = {
            "contract": CONTRACT,
            "sample_row_index": output_index,
            "root_base_uuid": root,
            "available_lanes": list(available),
            "selection_policy": SELECTION_POLICY,
            "selection_sha256": digest,
            "selection_slot": slot,
            "selected_lane": chosen_lane,
            "source_additive_row_index": additive_index,
            "source_additive_manifest_row_sha256": _canonical_sha256(source),
            "source_lane_row_index": source.get("lane_row_index"),
            "uuid": source.get("uuid"),
            "parent_uuid": source.get("parent_uuid"),
            "reference_sha256": source.get("reference_sha256"),
            "normalized_ast_sha256": source.get("normalized_ast_sha256"),
            "training_approved": False,
        }
        selected_rows.append(row)
        selected_manifests.append(output_manifest)
        review_metadata.append(
            {
                **output_manifest,
                "node_count": node_count,
                "node_bucket": _bucket(node_count, NODE_BUCKETS),
                "operator_count": len(operator_signature),
                "families": families,
                "review_sha256": hashlib.sha256(f"review\0{root}\0{chosen_lane}".encode()).hexdigest(),
                "row": row,
            }
        )

    selected_uuids = [_row_uuid(row) for row in selected_rows]
    selected_references = [_reference_sha256(row) for row in selected_rows]
    selected_asts = [_normalized_ast_sha256(row) for row in selected_rows]
    if not (
        len(selected_rows)
        == len(groups)
        == len(set(selected_uuids))
        == len(set(selected_references))
        == len(set(selected_asts))
    ):
        raise ValueError("one-question output identity or root cardinality mismatch")

    source_metadata = dict(table.schema.metadata or {})
    source_metadata.update(
        {
            b"csp_dag.one_question_contract": CONTRACT.encode(),
            b"csp_dag.one_question_selection_policy": SELECTION_POLICY.encode(),
            b"csp_dag.one_question_source_sha256": _file_sha256(source_parquet).encode(),
            b"csp_dag.review_only": b"true",
            b"csp_dag.training_approved": b"false",
        }
    )
    output_schema = table.schema.with_metadata(source_metadata)
    normalized_rows = pa.Table.from_pylist(list(selected_rows), schema=output_schema).to_pylist()
    if [_row_uuid(row) for row in normalized_rows] != selected_uuids:
        raise ValueError("Arrow normalization changed output row identity")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp.", dir=output_dir.parent))
    try:
        parquet_path = temporary / "selected.parquet"
        manifest_path = temporary / "selected.manifest.jsonl"
        review_path = temporary / "review_samples.md"
        summary_path = temporary / "summary.json"
        pq.write_table(pa.Table.from_pylist(normalized_rows, schema=output_schema), parquet_path, compression="zstd")
        manifest_path.write_text(
            "".join(_canonical_bytes(row).decode("utf-8") + "\n" for row in selected_manifests),
            encoding="utf-8",
        )
        review = _review_entries(review_metadata, sample_count=review_sample_count)
        _write_review(review_path, review)
        topology = _topology_analysis(base_manifests)
        summary: dict[str, Any] = {
            "contract": CONTRACT,
            "selection_policy": {
                "name": SELECTION_POLICY,
                "lane_order": list(LANE_ORDER),
                "definition": "sha256(contract + NUL + root_base_uuid) modulo the ordered lanes available for that root",
            },
            "source_rows": len(rows),
            "source_roots": len(groups),
            "selected_rows": len(normalized_rows),
            "rows_removed_as_same_root_siblings": len(rows) - len(normalized_rows),
            "source_rows_per_root": round(len(rows) / len(groups), 6),
            "availability_counts": dict(sorted(availability_counts.items())),
            "selected_lane_counts": dict(sorted(lane_counts.items())),
            "selected_shape_target_counts": dict(sorted(shape_targets.items(), key=lambda item: int(item[0]))),
            "selected_dtype_target_counts": dict(sorted(dtype_targets.items())),
            "identity": {
                "unique_roots": len(groups),
                "unique_uuids": len(set(selected_uuids)),
                "unique_reference_sha256": len(set(selected_references)),
                "unique_normalized_ast_sha256": len(set(selected_asts)),
            },
            "topology": topology,
            "source_binding": {
                "final_summary": _artifact(final_summary),
                "additive_parquet": _artifact(source_parquet, rows=len(rows)),
                "additive_manifest": _artifact(source_manifest, rows=len(manifests)),
                "lane_source_manifests": {
                    lane: _artifact(path, rows=len(source_rows)) for lane, (path, source_rows) in sources.items()
                },
                "sampler": _artifact(Path(__file__).resolve()),
            },
            "artifacts": {
                "selected_parquet": _staged_artifact(
                    parquet_path, output_dir / "selected.parquet", rows=len(normalized_rows)
                ),
                "selected_manifest": _staged_artifact(
                    manifest_path, output_dir / "selected.manifest.jsonl", rows=len(selected_manifests)
                ),
                "review_samples": _staged_artifact(review_path, output_dir / "review_samples.md", rows=len(review)),
            },
            "review_only": True,
            "training_approved": False,
        }
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output_dir)
        return summary
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--review-sample-count", type=int, default=15)
    return parser


def main() -> int:
    args = _parser().parse_args()
    run = args.run_root.resolve()
    summary = sample_one_question_per_parent(
        source_parquet=run / "selected.additive.parquet",
        source_manifest=run / "selected.additive.manifest.jsonl",
        final_summary=run / "final_summary.json",
        output_dir=args.output_dir,
        review_sample_count=args.review_sample_count,
    )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
