#!/usr/bin/env python3
"""Build, strict-near-deduplicate, and materialize 5k CSP-DAG candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.synthesize.csp_dag_method import audit_csp_dag_near_duplicates as dedup
from tools.data.synthesize.csp_dag_method import generate_csp_dag as generator


CONTRACT = "open_csp_dag_low_level_5k_v4"
RAW_ROWS = 5000
_DEFAULT_OUTPUT = (
    _REPO_ROOT
    / "local_artifacts/data/synthesize/csp_dag_low_level_5k/run.generated5000.strict_near_dedup.v4"
)
_PROMPT_PREFIX: str | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _init_worker(prompt_prefix: str) -> None:
    global _PROMPT_PREFIX
    _PROMPT_PREFIX = prompt_prefix


def _make_candidate(index: int) -> Any:
    if _PROMPT_PREFIX is None:
        raise RuntimeError("worker prompt prefix was not initialized")
    return generator._make_candidate(index, _PROMPT_PREFIX)


def _write_dataset(
    path: Path,
    manifest_path: Path,
    candidates: list[Any],
    schema: pa.Schema,
    layer: str,
) -> None:
    table = pa.Table.from_pylist([candidate.row for candidate in candidates], schema=schema)
    table = table.replace_schema_metadata(
        {
            **(table.schema.metadata or {}),
            b"csp_dag.contract": CONTRACT.encode(),
            b"csp_dag.layer": layer.encode(),
            b"csp_dag.review_only": b"true",
            b"csp_dag.training_approved": b"false",
        }
    )
    pq.write_table(table, path, compression="zstd")
    manifest_path.write_text(
        "".join(generator._canonical_json(candidate.manifest) + "\n" for candidate in candidates)
    )


def build(output_dir: Path, workers: int, chunksize: int) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    schema, prompt_prefix = generator._template(generator._DEFAULT_TEMPLATE)
    with multiprocessing.get_context("fork").Pool(
        workers, initializer=_init_worker, initargs=(prompt_prefix,)
    ) as pool:
        candidates = list(pool.imap(_make_candidate, range(RAW_ROWS), chunksize=chunksize))

    raw_path = output_dir / "raw.parquet"
    raw_manifest_path = output_dir / "raw.manifest.jsonl"
    _write_dataset(raw_path, raw_manifest_path, candidates, schema, "raw_generated")

    items = dedup._items_from_candidates(candidates)
    near_dir = output_dir / "near_dedup"
    near_dir.mkdir(exist_ok=True)
    near_summary = dedup._audit("raw5000", items, near_dir)
    retained_positions = {
        json.loads(line)["position"]
        for line in Path(
            near_summary["artifacts"]["strict_retained_uuids"]["path"]
        ).read_text().splitlines()
        if line.strip()
    }
    retained = [
        candidate for position, candidate in enumerate(candidates) if position in retained_positions
    ]
    retained_path = output_dir / "selected.parquet"
    retained_manifest_path = output_dir / "selected.manifest.jsonl"
    _write_dataset(
        retained_path,
        retained_manifest_path,
        retained,
        schema,
        "strict_near_dedup_retained",
    )

    retained_audit = dedup._audit(
        "retained", dedup._items_from_candidates(retained), near_dir
    )
    if retained_audit["strict_recommended_union"]["effective_rows"] != len(retained):
        raise AssertionError("retained set still contains strict near-duplicate components")

    review_path = dedup._manual_review(
        near_dir, items, dedup._items_from_candidates(retained), (near_summary,)
    )
    generator_path = Path(generator.__file__).resolve()
    dedup_path = Path(dedup.__file__).resolve()
    builder_path = Path(__file__).resolve()
    imported_sources = (
        _REPO_ROOT / "tools/data/cleaning/ast_similarity.py",
        _REPO_ROOT / "tools/data/cleaning/similarity.py",
        _REPO_ROOT / "tools/data/cleaning/complexity.py",
        _REPO_ROOT / "tools/data/cleaning/external.py",
        _REPO_ROOT / "tools/data/synthesize/select_low_level_kernelbench_canary.py",
    )
    summary = {
        "contract": CONTRACT,
        "generation": {
            "candidate_indices": [0, RAW_ROWS - 1],
            "raw_rows": len(candidates),
            "seed": generator.SEED,
            "generator_contract": generator.CONTRACT,
            "policy": "exactly 5,000 direct generator calls; no oversampling or selection MILP",
            "source_length_metric": "code-only physical lines; blank and comment-only lines excluded",
        },
        "near_dedup": {
            "strict_definition": near_summary["strict_recommended_union"]["definition"],
            "source_gate": near_summary["source_gate"],
            "exact_semantic_graph": near_summary["exact_semantic_graph"],
            "identity_normalized_semantic_graph": near_summary[
                "identity_normalized_semantic_graph"
            ],
            "strict_recommended_union": near_summary["strict_recommended_union"],
            "approximate_graph_sensitivity": near_summary["approximate_graph_sensitivity"],
            "retained_recheck": retained_audit["strict_recommended_union"],
        },
        "profiles": {
            "raw": generator._selected_profile(candidates),
            "retained": generator._selected_profile(retained),
        },
        "source_binding": {
            "builder": {"path": str(builder_path), "sha256": _sha256_file(builder_path)},
            "generator": {"path": str(generator_path), "sha256": _sha256_file(generator_path)},
            "near_dedup": {"path": str(dedup_path), "sha256": _sha256_file(dedup_path)},
            "imported_sources": [
                {"path": str(path.resolve()), "sha256": _sha256_file(path)}
                for path in imported_sources
            ],
            "template": {
                "path": str(generator._DEFAULT_TEMPLATE.resolve()),
                "sha256": _sha256_file(generator._DEFAULT_TEMPLATE),
                "usage": "schema and prompt prefix only",
            },
        },
        "artifacts": {
            "raw": {"path": str(raw_path.resolve()), "sha256": _sha256_file(raw_path)},
            "raw_manifest": {
                "path": str(raw_manifest_path.resolve()),
                "sha256": _sha256_file(raw_manifest_path),
            },
            "selected": {
                "path": str(retained_path.resolve()),
                "sha256": _sha256_file(retained_path),
            },
            "selected_manifest": {
                "path": str(retained_manifest_path.resolve()),
                "sha256": _sha256_file(retained_manifest_path),
            },
            "manual_review": {
                "path": str(review_path.resolve()),
                "sha256": _sha256_file(review_path),
            },
        },
        "review_only": True,
        "training_approved": False,
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=_DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=min(16, multiprocessing.cpu_count()))
    parser.add_argument("--chunksize", type=int, default=32)
    args = parser.parse_args()
    if args.workers <= 0 or args.chunksize <= 0:
        raise ValueError("workers and chunksize must be positive")
    print(json.dumps(build(args.output_dir, args.workers, args.chunksize), ensure_ascii=False))


if __name__ == "__main__":
    main()
