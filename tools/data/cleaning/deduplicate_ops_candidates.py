#!/usr/bin/env python3
"""Near-deduplicate an ops candidate pool against priority baselines and itself.

The token/AST metrics are intentionally gated by an exact normalized operator
set.  In same-domain synthetic corpora, generic ``nn.Module`` control-flow
shapes otherwise make unrelated operator programs look structurally similar.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.data.cleaning.ast_similarity import _significant_trees, ast_structure_similarity  # noqa: E402
from tools.data.cleaning.similarity import python_token_set, token_jaccard  # noqa: E402


@dataclass
class Reference:
    dataset_index: int
    row_index: int
    source_path: str
    source_row_index: int
    uuid: str
    entry_point: str
    code: str
    tokens: frozenset[tuple[int, str]] | None = None
    subtrees: tuple[Any, ...] | None = None
    total_weight: float | None = None


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _nested(row: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _ops_signature(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"invalid normalized ops metadata: {value!r}")
    return tuple(sorted(set(value)))


def _prepare(reference: Reference) -> None:
    if reference.tokens is None:
        reference.tokens = python_token_set(reference.code)
    if reference.subtrees is None:
        reference.subtrees = _significant_trees(reference.code, reference.entry_point)
        reference.total_weight = sum(tree.node_count * tree.weight for tree in reference.subtrees)


def _pair_scores(candidate: Reference, baseline: Reference) -> tuple[float, float]:
    _prepare(candidate)
    _prepare(baseline)
    assert candidate.tokens is not None and baseline.tokens is not None
    assert candidate.subtrees is not None and baseline.subtrees is not None
    assert candidate.total_weight is not None and baseline.total_weight is not None
    token_score = token_jaccard(candidate.tokens, baseline.tokens)
    ast_score = ast_structure_similarity(
        candidate.subtrees,
        baseline.subtrees,
        left_total_weight=candidate.total_weight,
        right_total_weight=baseline.total_weight,
    )
    return token_score, ast_score


def _best_match(
    candidate: Reference,
    references: list[Reference],
    *,
    token_threshold: float,
    ast_threshold: float,
) -> tuple[Reference, float, float] | None:
    matches: list[tuple[Reference, float, float]] = []
    for reference in references:
        if reference.entry_point != candidate.entry_point:
            continue
        token_score, ast_score = _pair_scores(candidate, reference)
        if token_score > token_threshold or ast_score > ast_threshold:
            matches.append((reference, token_score, ast_score))
    if not matches:
        return None
    return max(matches, key=lambda item: (max(item[1], item[2]), item[1], item[2], -item[0].row_index))


def _load_baseline_buckets(
    paths: list[Path],
    *,
    code_key: str,
    entry_point_key: str,
    ops_key: str,
) -> dict[tuple[str, ...], list[Reference]]:
    buckets: dict[tuple[str, ...], list[Reference]] = defaultdict(list)
    for dataset_index, path in enumerate(paths):
        for row_index, row in enumerate(pq.read_table(path).to_pylist()):
            code = _nested(row, code_key)
            entry_point = _nested(row, entry_point_key, "Model")
            uuid = _nested(row, "extra_info.uuid")
            if not all(isinstance(value, str) and value for value in (code, entry_point, uuid)):
                raise ValueError(f"invalid baseline identity in {path} row {row_index}")
            buckets[_ops_signature(_nested(row, ops_key))].append(
                Reference(dataset_index, row_index, str(path.resolve()), row_index, uuid, entry_point, code)
            )
    return buckets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--baseline", action="append", default=[], type=Path)
    parser.add_argument("--dedup-within", action="store_true")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--code-key", default="reward_model.ground_truth")
    parser.add_argument("--entry-point-key", default="extra_info.entry_point")
    parser.add_argument("--ops-key", default="extra_info.ops")
    parser.add_argument("--token-jaccard-threshold", type=float, default=0.8)
    parser.add_argument("--ast-similarity-threshold", type=float, default=0.9)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for path in (args.output, args.audit_jsonl, args.summary_json):
        if path.exists() and not args.overwrite:
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)

    baseline_buckets = _load_baseline_buckets(
        args.baseline,
        code_key=args.code_key,
        entry_point_key=args.entry_point_key,
        ops_key=args.ops_key,
    )
    input_tables = [pq.read_table(path) for path in args.input]
    if any(table.schema != input_tables[0].schema for table in input_tables[1:]):
        raise ValueError("input parquet schemas differ")
    table = pa.concat_tables(input_tables)
    source_locations = [
        (str(path.resolve()), source_row_index, input_index)
        for input_index, (path, input_table) in enumerate(zip(args.input, input_tables, strict=True))
        for source_row_index in range(len(input_table))
    ]
    rows = table.to_pylist()
    within_buckets: dict[tuple[str, ...], list[Reference]] = defaultdict(list)
    retained_indices: list[int] = []
    decisions: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    for row_index, row in enumerate(rows):
        code = _nested(row, args.code_key)
        entry_point = _nested(row, args.entry_point_key, "Model")
        uuid = _nested(row, "extra_info.uuid")
        if not all(isinstance(value, str) and value for value in (code, entry_point, uuid)):
            raise ValueError(f"invalid candidate identity at row {row_index}")
        signature = _ops_signature(_nested(row, args.ops_key))
        source_path, source_row_index, input_index = source_locations[row_index]
        candidate = Reference(
            len(args.baseline) + input_index,
            row_index,
            source_path,
            source_row_index,
            uuid,
            entry_point,
            code,
        )

        match = _best_match(
            candidate,
            baseline_buckets.get(signature, []),
            token_threshold=args.token_jaccard_threshold,
            ast_threshold=args.ast_similarity_threshold,
        )
        stage = "cross_pool"
        if match is None and args.dedup_within:
            match = _best_match(
                candidate,
                within_buckets.get(signature, []),
                token_threshold=args.token_jaccard_threshold,
                ast_threshold=args.ast_similarity_threshold,
            )
            stage = "within_candidate_pool"

        if match is None:
            retained_indices.append(row_index)
            decisions.append(
                {
                    "row_index": row_index,
                    "uuid": uuid,
                    "keep": True,
                    "stage": "retained",
                    "reason": "kept",
                    "ops_signature": list(signature),
                    "match": None,
                }
            )
            counts["kept"] += 1
            if args.dedup_within:
                within_buckets[signature].append(candidate)
            continue

        reference, token_score, ast_score = match
        reason = "near_duplicate_against" if stage == "cross_pool" else "near_duplicate_within"
        decisions.append(
            {
                "row_index": row_index,
                "uuid": uuid,
                "keep": False,
                "stage": stage,
                "reason": reason,
                "ops_signature": list(signature),
                "match": {
                    "dataset_index": reference.dataset_index,
                    "dataset_path": reference.source_path,
                    "row_index": reference.source_row_index,
                    "candidate_pool_row_index": reference.row_index,
                    "uuid": reference.uuid,
                    "token_jaccard": token_score,
                    "ast_similarity": ast_score,
                },
            }
        )
        counts[reason] += 1

    output = table.take(pa.array(retained_indices, type=pa.int64()))
    pq.write_table(output, args.output, compression=args.compression)
    with args.audit_jsonl.open("w", encoding="utf-8") as handle:
        for decision in decisions:
            handle.write(json.dumps(decision, sort_keys=True) + "\n")

    summary = {
        "contract": (
            "Preserve priority baseline rows and earlier candidate rows. Compare only rows with "
            "identical normalized operator sets; reject when token Jaccard is strictly above its "
            "threshold or entry-point AST structural similarity is strictly above its threshold."
        ),
        "inputs": [
            {
                "dataset_index": len(args.baseline) + index,
                "path": str(path.resolve()),
                "rows": len(input_table),
                "sha256": _sha256_file(path),
            }
            for index, (path, input_table) in enumerate(zip(args.input, input_tables, strict=True))
        ],
        "input_rows": len(rows),
        "baselines": [
            {
                "dataset_index": index,
                "path": str(path.resolve()),
                "rows": pq.ParquetFile(path).metadata.num_rows,
                "sha256": _sha256_file(path),
            }
            for index, path in enumerate(args.baseline)
        ],
        "dedup_within": args.dedup_within,
        "operator_gate": "exact normalized extra_info.ops set",
        "token_jaccard_threshold_strictly_greater_than": args.token_jaccard_threshold,
        "ast_similarity_threshold_strictly_greater_than": args.ast_similarity_threshold,
        "reason_counts": dict(sorted(counts.items())),
        "output": str(args.output.resolve()),
        "output_rows": len(output),
        "output_sha256": _sha256_file(args.output),
        "audit_jsonl": str(args.audit_jsonl.resolve()),
        "audit_jsonl_sha256": _sha256_file(args.audit_jsonl),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
