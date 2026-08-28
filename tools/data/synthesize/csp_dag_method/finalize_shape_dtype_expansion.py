#!/usr/bin/env python3
"""Finalize the review-only base -> shape -> dtype CSP-DAG chain.

Layout is intentionally excluded from this contract.  The caller must bind a
review artifact that explains why layout was deferred; no layout candidate or
runtime directory is read by this finalizer.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import json
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tools.data.synthesize.augment_prompt_tasks import _normalized_ast_sha256
from tools.data.synthesize.csp_dag_method import build_csp_dag_input_expansions as expansion
from tools.data.synthesize.csp_dag_method import csp_dag_finalization as common
from tools.data.synthesize.csp_dag_method import generate_csp_dag as generator

FINAL_CONTRACT = "csp_dag_shape_dtype_expansion_final_v1"
SERIAL_ORDER = ("shape", "dtype")
OUTPUT_NAMES = (
    "selected.additive.parquet",
    "selected.additive.manifest.jsonl",
    "accepted_review.md",
    "final_summary.json",
)
DEFERRED_LAYOUT_CONTRACT = "csp_dag_layout_deferred_scope_v1"


def _sha256_text(value: str) -> str:
    return common._sha256_text(value)


def _identity(row: Mapping[str, Any]) -> tuple[str, str]:
    return common._identity(row)


def _file(path: Path) -> dict[str, str]:
    return common._file(path)


def _histogram(values: Iterable[Any]) -> dict[str, int]:
    counts: collections.Counter[str] = collections.Counter(str(value) for value in values)
    return dict(sorted(counts.items()))


@contextmanager
def _cached_expansion_file_hashes() -> Iterable[None]:
    """Cache immutable artifact hashes during one strict record scan.

    The shared selector validates the same parents, candidates, and manifest
    bindings for every runtime record.  That is useful fail-closed behavior,
    but recomputing those file hashes per row turns a final review into an
    O(rows * artifact_size) scan.  This finalizer is single-threaded and the
    artifacts are frozen, so a call-scoped cache preserves the exact checks.
    """
    original = expansion._sha256_file
    cache: dict[Path, str] = {}

    def cached(path: Path) -> str:
        resolved = Path(path).resolve()
        if resolved not in cache:
            cache[resolved] = original(Path(path))
        return cache[resolved]

    expansion._sha256_file = cached
    try:
        yield
    finally:
        expansion._sha256_file = original


def _assert_outputs_absent(run_root: Path) -> None:
    existing = [name for name in OUTPUT_NAMES if (run_root / name).exists()]
    if existing:
        raise FileExistsError(f"shape+dtype final outputs already exist:{','.join(existing)}")


def _deferred_layout(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"layout deferred evidence missing:{path}")
    payload = json.loads(path.read_text())
    decision = payload.get("decision")
    if (
        payload.get("artifact_contract") != DEFERRED_LAYOUT_CONTRACT
        or payload.get("reason") != "excluded_by_user_scope"
        or not isinstance(decision, Mapping)
        or decision.get("requested_by_user") is not True
        or decision.get("included_in_current_chain") is not False
        or decision.get("layout_build_started") is not False
        or decision.get("layout_gpu_started") is not False
        or decision.get("layout_candidate_rows") != 0
        or decision.get("layout_runtime_rows") != 0
    ):
        raise ValueError("layout deferred evidence contract/result mismatch")
    return {
        "status": "deferred",
        "reason": "excluded_by_user_scope",
        "evidence": _file(path),
        "candidate_rows": 0,
        "gpu_runtime_rows": 0,
        "included_in_additive": False,
    }


def _assert_direct_lineage_cached(
    *,
    stage: str,
    parent_rows: Sequence[Mapping[str, Any]],
    parent_manifests: Sequence[Mapping[str, Any]],
    parent_summary: Path,
    children: Sequence[Mapping[str, Any]],
    child_manifests: Sequence[Mapping[str, Any]],
    selected_parents: Sequence[Mapping[str, Any]] | None,
) -> dict[str, str]:
    """Check direct lineage while hashing each shared parent artifact once."""
    if len(parent_rows) != len(parent_manifests):
        raise ValueError(f"lineage parent rows/manifests differ:{stage}")
    if len(children) != len(child_manifests):
        raise ValueError(f"lineage child rows/manifests differ:{stage}")
    if selected_parents is not None and len(selected_parents) != len(children):
        raise ValueError(f"lineage selected parent rows differ:{stage}")

    parent_by_uuid = {
        _identity(row)[0]: (row, manifest) for row, manifest in zip(parent_rows, parent_manifests, strict=True)
    }
    if len(parent_by_uuid) != len(parent_rows):
        raise ValueError(f"lineage parent UUID collision:{stage}")

    parent_dir = parent_summary.parent
    expected_files = {
        "source_summary": _file(parent_summary),
        "source_artifact": _file(parent_dir / "selected.parquet"),
        "source_manifest": _file(parent_dir / "selected.manifest.jsonl"),
    }
    expected_stage = {"shape": "base", "dtype": "shape"}[stage]
    root_by_child: dict[str, str] = {}
    for index, (child, manifest) in enumerate(zip(children, child_manifests, strict=True)):
        child_uuid, child_code = _identity(child)
        parent_uuid = manifest.get("parent_uuid")
        if not isinstance(parent_uuid, str) or parent_uuid not in parent_by_uuid:
            raise ValueError(f"lineage parent UUID missing:{stage}:{index}")
        parent, parent_manifest = parent_by_uuid[parent_uuid]
        parent_uuid_actual, parent_code = _identity(parent)
        if selected_parents is not None and _identity(selected_parents[index]) != (
            parent_uuid_actual,
            parent_code,
        ):
            raise ValueError(f"lineage selected parent mismatch:{stage}:{index}")

        binding = manifest.get("csp_dag_source_binding")
        if not isinstance(binding, Mapping):
            raise ValueError(f"lineage source binding missing:{stage}:{index}")
        if binding.get("contract") != expansion.SOURCE_BINDING_CONTRACT or binding.get("stage") != expected_stage:
            raise ValueError(f"lineage source binding stage mismatch:{stage}:{index}")
        for name, expected in expected_files.items():
            declared = binding.get(name)
            if not isinstance(declared, Mapping):
                raise ValueError(f"lineage:{stage}:{index} lacks {name} binding")
            if dict(declared) != expected:
                raise ValueError(f"lineage:{stage}:{index} {name} binding mismatch")

        source_index = binding.get("source_row_index")
        if not isinstance(source_index, int) or not 0 <= source_index < len(parent_rows):
            raise ValueError(f"lineage source row index mismatch:{stage}:{index}")
        if common._canonical(parent_rows[source_index]) != common._canonical(parent):
            raise ValueError(f"lineage parent row mismatch:{stage}:{index}")
        if common._canonical(parent_manifests[source_index]) != common._canonical(parent_manifest):
            raise ValueError(f"lineage parent manifest mismatch:{stage}:{index}")
        if (
            binding.get("source_rows") != len(parent_rows)
            or binding.get("source_row_sha256") != expansion._row_sha256(parent)
            or binding.get("source_manifest_row_sha256") != expansion._row_sha256(parent_manifest)
        ):
            raise ValueError(f"lineage source row hash mismatch:{stage}:{index}")

        if child_uuid == parent_uuid_actual or _sha256_text(child_code) == _sha256_text(parent_code):
            raise ValueError(f"lineage intervention did not change reference:{stage}:{index}")
        declared_reference = manifest.get("reference_sha256", manifest.get("child_reference_sha256"))
        declared_ast = manifest.get("normalized_ast_sha256", manifest.get("child_normalized_ast_sha256"))
        if declared_reference != _sha256_text(child_code) or declared_ast != _normalized_ast_sha256(child_code):
            raise ValueError(f"lineage child code binding mismatch:{stage}:{index}")
        root_by_child[child_uuid] = root_by_child.get(parent_uuid_actual, parent_uuid_actual)
    return root_by_child


def _write_additive(
    run_root: Path,
    lanes: Sequence[
        tuple[
            str,
            Sequence[Mapping[str, Any]],
            Sequence[Mapping[str, Any]],
            Path,
            Path,
            Mapping[str, str],
        ]
    ],
    schema: pa.Schema,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    lane_counts: dict[str, int] = {}
    for lane, lane_rows, manifests, selected, selected_manifest, roots in lanes:
        if len(lane_rows) != len(manifests):
            raise ValueError(f"additive lane rows/manifests differ:{lane}")
        lane_counts[lane] = len(lane_rows)
        selected_binding = _file(selected)
        manifest_binding = _file(selected_manifest)
        for lane_index, (row, manifest) in enumerate(zip(lane_rows, manifests, strict=True)):
            uuid, code = _identity(row)
            rows.append(dict(row))
            lineage.append(
                {
                    "additive_row_index": len(rows) - 1,
                    "lane": lane,
                    "lane_row_index": lane_index,
                    "uuid": uuid,
                    "root_base_uuid": roots[uuid],
                    "parent_uuid": manifest.get("parent_uuid"),
                    "reference_sha256": _sha256_text(code),
                    "normalized_ast_sha256": _normalized_ast_sha256(code),
                    "source_parquet": selected_binding,
                    "source_manifest": manifest_binding,
                    "source_manifest_row_sha256": expansion._row_sha256(manifest),
                }
            )
    output_schema = common._extended_schema(schema).with_metadata(
        {
            **(schema.metadata or {}),
            b"csp_dag.additive_contract": FINAL_CONTRACT.encode(),
            b"csp_dag.review_only": b"true",
            b"csp_dag.training_approved": b"false",
        }
    )
    parquet_path = run_root / "selected.additive.parquet"
    manifest_path = run_root / "selected.additive.manifest.jsonl"
    pq.write_table(pa.Table.from_pylist(rows, schema=output_schema), parquet_path, compression="zstd")
    manifest_path.write_text("".join(expansion._canonical_json(item) + "\n" for item in lineage))
    written = pq.read_table(parquet_path).to_pylist()
    if len(written) != len(rows) or len(lineage) != len(rows):
        raise ValueError("additive artifact row conservation failure")
    for index, (expected, actual) in enumerate(zip(rows, written, strict=True)):
        if _identity(expected) != _identity(actual):
            raise ValueError(f"additive parquet identity mismatch:{index}")
    if sum(lane_counts.values()) != len(rows):
        raise ValueError("additive lane count mismatch")
    return {
        "rows": len(rows),
        "lane_counts": lane_counts,
        "parquet": _file(parquet_path),
        "manifest": _file(manifest_path),
    }


def _write_review(
    run_root: Path,
    lanes: Sequence[tuple[str, Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]],
) -> dict[str, str]:
    lines = ["# H20-accepted shape/dtype examples", ""]
    for lane, parents, children in lanes:
        if len(parents) != len(children):
            raise ValueError(f"accepted review parent/child count mismatch:{lane}")
        for parent, child in list(zip(parents, children, strict=True))[:4]:
            parent_uuid, parent_code = _identity(parent)
            child_uuid, child_code = _identity(child)
            lines.extend(
                [
                    f"## {lane}: {parent_uuid} -> {child_uuid}",
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
    path = run_root / "accepted_review.md"
    path.write_text("\n".join(lines).rstrip() + "\n")
    return _file(path)


def finalize(
    run_root: Path,
    base: Path,
    *,
    layout_deferred_evidence: Path,
    shape_post_selection_runtime: Path | None = None,
    dtype_post_selection_runtime: Path | None = None,
) -> dict[str, Any]:
    _assert_outputs_absent(run_root)
    if (run_root / "layout").exists():
        raise ValueError("layout directory exists in the shape+dtype-only run")
    deferred_layout = _deferred_layout(layout_deferred_evidence)
    shape = run_root / "shape"
    dtype = run_root / "dtype"

    base_rows, base_manifests, _ = expansion._verify_source_rows(
        base / "selected.parquet",
        base / "selected.manifest.jsonl",
        stage="base",
        expected_manifest=("contract", generator.CONTRACT),
    )

    shape_candidates = pq.read_table(shape / "candidates.parquet").to_pylist()
    shape_candidate_manifests = expansion._read_jsonl(shape / "manifest.jsonl")
    if len(shape_candidates) != len(shape_candidate_manifests):
        raise ValueError("shape candidates/manifests differ")
    with _cached_expansion_file_hashes():
        shape_records, shape_runtime_paths = expansion._strict_shape_records(
            shape, shape / "runtime_h20", shape_candidates
        )
    shape_rows, shape_manifests, _, _ = common._selection_summary(
        shape,
        selection_stage="shape_runtime_selection",
        candidates_path=shape / "candidates.parquet",
        manifest_path=shape / "manifest.jsonl",
        selected_path=shape / "selected.parquet",
        selected_manifest_path=shape / "selected.manifest.jsonl",
        runtime_paths=shape_runtime_paths,
        candidate_rows=shape_candidates,
        candidate_manifests=shape_candidate_manifests,
        runtime_records=shape_records,
    )

    dtype_candidates = pq.read_table(dtype / "candidates.parquet").to_pylist()
    dtype_candidate_manifests = expansion._read_jsonl(dtype / "manifest.jsonl")
    dtype_parents = pq.read_table(dtype / "parents.parquet").to_pylist()
    if not (len(dtype_candidates) == len(dtype_candidate_manifests) == len(dtype_parents)):
        raise ValueError("dtype candidates/parents/manifests differ")
    with _cached_expansion_file_hashes():
        dtype_records, dtype_runtime_paths = expansion._strict_liveness_records(
            dtype, dtype / "runtime_h20", dtype_parents, dtype_candidates
        )
    dtype_rows, dtype_manifests, dtype_selected_parents, _ = common._selection_summary(
        dtype,
        selection_stage="runtime_selection",
        candidates_path=dtype / "candidates.parquet",
        manifest_path=dtype / "manifest.jsonl",
        selected_path=dtype / "selected.parquet",
        selected_manifest_path=dtype / "selected.manifest.jsonl",
        runtime_paths=dtype_runtime_paths,
        candidate_rows=dtype_candidates,
        candidate_manifests=dtype_candidate_manifests,
        runtime_records=dtype_records,
        parents_path=dtype / "selected.parents.parquet",
    )

    roots_base = {_identity(row)[0]: _identity(row)[0] for row in base_rows}
    roots_shape = _assert_direct_lineage_cached(
        stage="shape",
        parent_rows=base_rows,
        parent_manifests=base_manifests,
        parent_summary=base / "summary.json",
        children=shape_rows,
        child_manifests=shape_manifests,
        selected_parents=None,
    )
    roots_dtype = _assert_direct_lineage_cached(
        stage="dtype",
        parent_rows=shape_rows,
        parent_manifests=shape_manifests,
        parent_summary=shape / "selection_summary.json",
        children=dtype_rows,
        child_manifests=dtype_manifests,
        selected_parents=dtype_selected_parents,
    )
    for child, manifest in zip(shape_rows, shape_manifests, strict=True):
        roots_shape[_identity(child)[0]] = roots_base[manifest["parent_uuid"]]
    for child, manifest in zip(dtype_rows, dtype_manifests, strict=True):
        roots_dtype[_identity(child)[0]] = roots_shape[manifest["parent_uuid"]]
    common._assert_global_identity(
        (("base", base_rows, roots_base), ("shape", shape_rows, roots_shape), ("dtype", dtype_rows, roots_dtype))
    )

    selected_lane_data = {
        "shape": (
            shape,
            shape_rows,
            shape_manifests,
            shape / "selected.parquet",
            shape / "selected.manifest.jsonl",
            shape / "selection_summary.json",
            shape_post_selection_runtime,
        ),
        "dtype": (
            dtype,
            dtype_rows,
            dtype_manifests,
            dtype / "selected.parquet",
            dtype / "selected.manifest.jsonl",
            dtype / "selection_summary.json",
            dtype_post_selection_runtime,
        ),
    }
    post_evidence: dict[str, dict[str, Any]] = {}
    for lane in SERIAL_ORDER:
        lane_dir, rows, manifests, selected, selected_manifest, selection, override = selected_lane_data[lane]
        runtime_dir = override if override is not None else lane_dir / "runtime_h20_post_selection"
        records, record_paths, summary_paths = common._post_selection_records(
            lane_dir, runtime_dir, selected, selected_manifest, selection, rows, manifests
        )
        post_evidence[lane] = {
            "records": [_file(path) for path in record_paths],
            "summaries": [_file(path) for path in summary_paths],
            "rows": len(records),
            "status_counts": {"passed": len(records)},
        }

    shape_summary = json.loads((shape / "summary.json").read_text())
    dtype_summary = json.loads((dtype / "summary.json").read_text())
    shape_histogram = _histogram(
        manifest["shape_intervention"]["assigned_target_value"] for manifest in shape_manifests
    )
    dtype_histogram = _histogram(manifest["assigned_target"] for manifest in dtype_manifests)
    if shape_summary.get("target_value_counts") != shape_histogram:
        raise ValueError("shape selected target histogram mismatch")
    candidate_dtype_histogram = dtype_summary.get("target_counts")
    if not isinstance(candidate_dtype_histogram, Mapping) or any(
        count > candidate_dtype_histogram.get(name, 0) for name, count in dtype_histogram.items()
    ):
        raise ValueError("dtype selected histogram is not conserved by candidates")

    additive = _write_additive(
        run_root,
        (
            (
                "base",
                base_rows,
                base_manifests,
                base / "selected.parquet",
                base / "selected.manifest.jsonl",
                roots_base,
            ),
            (
                "shape",
                shape_rows,
                shape_manifests,
                shape / "selected.parquet",
                shape / "selected.manifest.jsonl",
                roots_shape,
            ),
            (
                "dtype",
                dtype_rows,
                dtype_manifests,
                dtype / "selected.parquet",
                dtype / "selected.manifest.jsonl",
                roots_dtype,
            ),
        ),
        pq.ParquetFile(dtype / "selected.parquet").schema_arrow,
    )
    accepted_review = _write_review(
        run_root,
        (("shape", base_rows, shape_rows), ("dtype", dtype_selected_parents or [], dtype_rows)),
    )
    result = {
        "contract": FINAL_CONTRACT,
        "serial_order": list(SERIAL_ORDER),
        "lanes": {
            "base": {"selected": len(base_rows), "artifact": _file(base / "selected.parquet")},
            "shape": {
                "candidates": len(shape_candidates),
                "selected": len(shape_rows),
                "selected_histogram": shape_histogram,
                "selection_summary": _file(shape / "selection_summary.json"),
                "selected_artifact": _file(shape / "selected.parquet"),
            },
            "dtype": {
                "candidates": len(dtype_candidates),
                "selected": len(dtype_rows),
                "selected_histogram": dtype_histogram,
                "selection_summary": _file(dtype / "selection_summary.json"),
                "selected_artifact": _file(dtype / "selected.parquet"),
                "selected_parents": _file(dtype / "selected.parents.parquet"),
            },
        },
        "excluded_lanes": {"layout": deferred_layout},
        "preselection_runtime_sources": {
            "shape": [_file(path) for path in shape_runtime_paths],
            "dtype": [_file(path) for path in dtype_runtime_paths],
        },
        "postselection_h20_evidence": post_evidence,
        "source_binding": {
            "finalizer": _file(Path(__file__).resolve()),
            "shared_finalizer_helpers": _file(Path(common.__file__).resolve()),
            "expansion_builder": _file(Path(expansion.__file__).resolve()),
            "base_summary": _file(base / "summary.json"),
            "shape_summary": _file(shape / "summary.json"),
            "dtype_summary": _file(dtype / "summary.json"),
        },
        "additive_artifact": additive,
        "accepted_review": accepted_review,
        "review_only": True,
        "training_approved": False,
    }
    output = run_root / "final_summary.json"
    output.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--layout-deferred-evidence", type=Path, required=True)
    parser.add_argument("--shape-post-selection-runtime", type=Path)
    parser.add_argument("--dtype-post-selection-runtime", type=Path)
    args = parser.parse_args()
    result = finalize(
        args.run_root,
        args.base,
        layout_deferred_evidence=args.layout_deferred_evidence,
        shape_post_selection_runtime=args.shape_post_selection_runtime,
        dtype_post_selection_runtime=args.dtype_post_selection_runtime,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
