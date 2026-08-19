#!/usr/bin/env python3
"""Combine the two one-parent releases and apply a source-bound internal dedup."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.cleaning.complexity import extract_operator_signature  # noqa: E402
from tools.data.cleaning.deduplicate_ops_candidates import Reference, _best_match, _pair_scores  # noqa: E402
from tools.data.synthesize.augment_prompt_tasks import _normalized_ast_sha256  # noqa: E402

CONTRACT = "combined_training_release_internal_dedup_v1"
TOKEN_THRESHOLD = 0.8
AST_THRESHOLD = 0.9
SOURCE_ORDER = ("original_training", "high_complexity")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _nested(row: Mapping[str, Any], path: str) -> Any:
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def _display(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(_REPO_ROOT))
    except ValueError:
        return str(resolved)


def _source(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"path": _display(path), "sha256": _sha256_file(path)}
    if rows is not None:
        result["rows"] = rows
    return result


def _read_manifest(path: Path, expected_rows: int) -> tuple[list[dict[str, Any]], list[str]]:
    raw_lines = [line for line in path.read_text().splitlines() if line.strip()]
    if len(raw_lines) != expected_rows:
        raise ValueError(f"manifest rows mismatch for {path}: {len(raw_lines)} != {expected_rows}")
    return [json.loads(line) for line in raw_lines], raw_lines


def _identity_from_manifest(source_name: str, manifest: Mapping[str, Any]) -> tuple[str, str, str]:
    if source_name == "original_training":
        return (
            str(manifest.get("terminal_uuid") or manifest.get("canonical_parent_uuid")),
            str(manifest.get("selected_reference_sha256")),
            str(manifest.get("selected_normalized_ast_sha256")),
        )
    return (
        str(manifest.get("uuid")),
        str(manifest.get("reference_sha256")),
        str(manifest.get("normalized_ast_sha256")),
    )


def _identity(
    source_name: str,
    source_row_index: int,
    row: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[str, str, str, str, tuple[str, ...]]:
    code = _nested(row, "reward_model.ground_truth")
    uuid = _nested(row, "extra_info.uuid")
    entry_point = _nested(row, "extra_info.entry_point") or "Model"
    if not all(isinstance(value, str) and value for value in (code, uuid, entry_point)):
        raise ValueError(f"invalid row identity: {source_name}:{source_row_index}")
    reference_sha256 = _sha256_text(code)
    normalized_ast_sha256 = _normalized_ast_sha256(code)
    manifest_uuid, manifest_reference, manifest_ast = _identity_from_manifest(source_name, manifest)
    if (manifest_uuid, manifest_reference, manifest_ast) != (
        uuid,
        reference_sha256,
        normalized_ast_sha256,
    ):
        raise ValueError(f"manifest identity mismatch: {source_name}:{source_row_index}")
    v4 = _nested(row, "extra_info.v4")
    if isinstance(v4, Mapping):
        if v4.get("reference_sha256") != reference_sha256:
            raise ValueError(f"v4 reference binding mismatch: {source_name}:{source_row_index}")
        if v4.get("normalized_ast_sha256") != normalized_ast_sha256:
            raise ValueError(f"v4 AST binding mismatch: {source_name}:{source_row_index}")
    signature = tuple(sorted(set(extract_operator_signature(code, entry_point))))
    return uuid, reference_sha256, normalized_ast_sha256, entry_point, signature


def _score_summary(values: Sequence[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p50": statistics.median(ordered),
        "p90": ordered[round((len(ordered) - 1) * 0.9)],
        "max": ordered[-1],
    }


def _review_selection(decisions: Sequence[Mapping[str, Any]], limit: int) -> list[Mapping[str, Any]]:
    rejected = [row for row in decisions if not row["keep"] and row.get("match")]
    selected: list[Mapping[str, Any]] = []

    def add(rows: Sequence[Mapping[str, Any]]) -> None:
        for row in rows:
            if row not in selected and len(selected) < limit:
                selected.append(row)

    for source_name in SOURCE_ORDER:
        source_rows = [row for row in rejected if row["source_dataset"] == source_name]
        both = [
            row
            for row in source_rows
            if row["match"]["token_jaccard"] > TOKEN_THRESHOLD and row["match"]["ast_similarity"] > AST_THRESHOLD
        ]
        token_only = [
            row
            for row in source_rows
            if row["match"]["token_jaccard"] > TOKEN_THRESHOLD and row["match"]["ast_similarity"] <= AST_THRESHOLD
        ]
        ast_only = [
            row
            for row in source_rows
            if row["match"]["token_jaccard"] <= TOKEN_THRESHOLD and row["match"]["ast_similarity"] > AST_THRESHOLD
        ]
        for group in (both, token_only, ast_only):
            add(sorted(group, key=lambda row: -max(row["match"]["token_jaccard"], row["match"]["ast_similarity"]))[:1])
            add(sorted(group, key=lambda row: max(row["match"]["token_jaccard"], row["match"]["ast_similarity"]))[:1])
    add(rejected)
    return selected[:limit]


def _write_review(
    path: Path,
    decisions: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    limit: int,
) -> int:
    selected = _review_selection(decisions, limit)
    lines = [
        "# Combined training release near-dedup review pairs",
        "",
        "The original-training release has priority. Each rejected row is shown beside the earlier retained row that triggered the fixed source gate.",
        "",
    ]
    for review_index, decision in enumerate(selected):
        match = decision["match"]
        candidate = rows[int(decision["global_row_index"])]
        retained = rows[int(match["global_row_index"])]
        lines.extend(
            [
                f"## Pair {review_index:02d}: {decision['uuid']} vs {match['uuid']}",
                "",
                f"- candidate: `{decision['source_dataset']}:{decision['source_row_index']}`",
                f"- retained: `{match['source_dataset']}:{match['source_row_index']}`",
                f"- token Jaccard: `{match['token_jaccard']:.6f}`",
                f"- Model AST similarity: `{match['ast_similarity']:.6f}`",
                f"- unified forward operator set: `{_canonical_json(decision['operator_signature'])}`",
                "",
                "### Rejected candidate",
                "",
                "```python",
                str(_nested(candidate, "reward_model.ground_truth")),
                "```",
                "",
                "### Retained row",
                "",
                "```python",
                str(_nested(retained, "reward_model.ground_truth")),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines))
    return len(selected)


def _write_cross_review(
    path: Path,
    cross_pairs: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    limit: int = 8,
) -> int:
    selected = sorted(
        cross_pairs,
        key=lambda row: (-max(row["token_jaccard"], row["ast_similarity"]), row["original_global_row_index"]),
    )[:limit]
    lines = [
        "# Cross-source near-dedup review pairs",
        "",
        "These are the strongest pairs that passed the unified operator-set prefilter. None crossed the fixed rejection threshold.",
        "",
    ]
    for review_index, pair in enumerate(selected):
        original = rows[int(pair["original_global_row_index"])]
        high = rows[int(pair["high_complexity_global_row_index"])]
        lines.extend(
            [
                f"## Pair {review_index:02d}: {pair['original_uuid']} vs {pair['high_complexity_uuid']}",
                "",
                f"- token Jaccard: `{pair['token_jaccard']:.6f}`",
                f"- Model AST similarity: `{pair['ast_similarity']:.6f}`",
                f"- unified forward operator set: `{_canonical_json(pair['operator_signature'])}`",
                "",
                "### Original-training row",
                "",
                "```python",
                str(_nested(original, "reward_model.ground_truth")),
                "```",
                "",
                "### High-complexity row",
                "",
                "```python",
                str(_nested(high, "reward_model.ground_truth")),
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines))
    return len(selected)


def build(
    sources: Sequence[tuple[str, Path, Path]],
    output_dir: Path,
    review_pair_count: int,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)

    tables: list[pa.Table] = []
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    manifest_lines: list[str] = []
    source_locations: list[tuple[str, int, Path, Path]] = []
    input_bindings: dict[str, Any] = {}
    for source_name, parquet_path, manifest_path in sources:
        if source_name not in SOURCE_ORDER:
            raise ValueError(f"unexpected source name: {source_name}")
        table = pq.read_table(parquet_path)
        if tables and table.schema != tables[0].schema:
            raise ValueError("source parquet schemas differ")
        source_manifests, raw_manifest_lines = _read_manifest(manifest_path, table.num_rows)
        tables.append(table)
        rows.extend(table.to_pylist())
        manifests.extend(source_manifests)
        manifest_lines.extend(raw_manifest_lines)
        source_locations.extend(
            (source_name, row_index, parquet_path, manifest_path) for row_index in range(table.num_rows)
        )
        input_bindings[source_name] = {
            "parquet": _source(parquet_path, rows=table.num_rows),
            "manifest": _source(manifest_path, rows=table.num_rows),
        }

    identities: list[tuple[str, str, str, str, tuple[str, ...]]] = []
    signature_counts: dict[str, collections.Counter[tuple[str, ...]]] = {
        source_name: collections.Counter() for source_name, _, _ in sources
    }
    for row, manifest, location in zip(rows, manifests, source_locations, strict=True):
        source_name, source_row_index, _, _ = location
        identity = _identity(source_name, source_row_index, row, manifest)
        identities.append(identity)
        signature_counts[source_name][identity[-1]] += 1

    all_references: list[Reference] = []
    indices_by_source_signature: dict[tuple[str, tuple[str, ...]], list[int]] = collections.defaultdict(list)
    for global_index, (identity, location) in enumerate(zip(identities, source_locations, strict=True)):
        uuid, _, _, entry_point, signature = identity
        source_name, source_row_index, parquet_path, _ = location
        all_references.append(
            Reference(
                SOURCE_ORDER.index(source_name),
                global_index,
                str(parquet_path.resolve()),
                source_row_index,
                uuid,
                entry_point,
                str(_nested(rows[global_index], "reward_model.ground_truth")),
            )
        )
        indices_by_source_signature[(source_name, signature)].append(global_index)

    cross_pairs: list[dict[str, Any]] = []
    shared_signatures = sorted(set(signature_counts["original_training"]) & set(signature_counts["high_complexity"]))
    for signature in shared_signatures:
        for original_index in indices_by_source_signature[("original_training", signature)]:
            for high_index in indices_by_source_signature[("high_complexity", signature)]:
                token_score, ast_score = _pair_scores(all_references[original_index], all_references[high_index])
                cross_pairs.append(
                    {
                        "contract": CONTRACT,
                        "original_global_row_index": original_index,
                        "original_source_row_index": source_locations[original_index][1],
                        "original_uuid": identities[original_index][0],
                        "high_complexity_global_row_index": high_index,
                        "high_complexity_source_row_index": source_locations[high_index][1],
                        "high_complexity_uuid": identities[high_index][0],
                        "operator_signature": list(signature),
                        "token_jaccard": token_score,
                        "ast_similarity": ast_score,
                        "reject": token_score > TOKEN_THRESHOLD or ast_score > AST_THRESHOLD,
                    }
                )

    retained_indices: list[int] = []
    decisions: list[dict[str, Any]] = []
    buckets: dict[tuple[str, ...], list[Reference]] = collections.defaultdict(list)
    reference_locations: dict[int, dict[str, Any]] = {}
    seen: dict[str, dict[str, int]] = {"uuid": {}, "reference_sha256": {}, "normalized_ast_sha256": {}}
    counts: collections.Counter[str] = collections.Counter()
    source_retained: collections.Counter[str] = collections.Counter()
    source_removed: collections.Counter[str] = collections.Counter()

    for global_index, (identity, location) in enumerate(zip(identities, source_locations, strict=True)):
        uuid, reference_sha256, ast_sha256, entry_point, signature = identity
        source_name, source_row_index, parquet_path, _ = location
        exact_reason = None
        exact_match_index = None
        for key, value in (
            ("uuid", uuid),
            ("reference_sha256", reference_sha256),
            ("normalized_ast_sha256", ast_sha256),
        ):
            if value in seen[key]:
                exact_reason = f"exact_{key}"
                exact_match_index = seen[key][value]
                break

        match_payload: dict[str, Any] | None = None
        reason = "kept"
        if exact_reason is not None:
            reason = exact_reason
            assert exact_match_index is not None
            match_payload = {**reference_locations[exact_match_index], "global_row_index": exact_match_index}
        else:
            candidate = Reference(
                SOURCE_ORDER.index(source_name),
                global_index,
                str(parquet_path.resolve()),
                source_row_index,
                uuid,
                entry_point,
                str(_nested(rows[global_index], "reward_model.ground_truth")),
            )
            near = _best_match(
                candidate,
                buckets[signature],
                token_threshold=TOKEN_THRESHOLD,
                ast_threshold=AST_THRESHOLD,
            )
            if near is not None:
                reference, token_score, ast_score = near
                reference_location = reference_locations[reference.row_index]
                reason = (
                    "near_duplicate_cross_source"
                    if reference_location["source_dataset"] != source_name
                    else "near_duplicate_within_source"
                )
                match_payload = {
                    **reference_location,
                    "global_row_index": reference.row_index,
                    "token_jaccard": token_score,
                    "ast_similarity": ast_score,
                }

        keep = reason == "kept"
        decision = {
            "contract": CONTRACT,
            "global_row_index": global_index,
            "source_dataset": source_name,
            "source_row_index": source_row_index,
            "uuid": uuid,
            "reference_sha256": reference_sha256,
            "normalized_ast_sha256": ast_sha256,
            "operator_signature": list(signature),
            "keep": keep,
            "reason": reason,
            "match": match_payload,
        }
        decisions.append(decision)
        counts[reason] += 1
        if not keep:
            source_removed[source_name] += 1
            continue

        retained_indices.append(global_index)
        source_retained[source_name] += 1
        reference = Reference(
            SOURCE_ORDER.index(source_name),
            global_index,
            str(parquet_path.resolve()),
            source_row_index,
            uuid,
            entry_point,
            str(_nested(rows[global_index], "reward_model.ground_truth")),
        )
        buckets[signature].append(reference)
        reference_locations[global_index] = {
            "source_dataset": source_name,
            "source_row_index": source_row_index,
            "uuid": uuid,
            "reference_sha256": reference_sha256,
            "normalized_ast_sha256": ast_sha256,
        }
        for key, value in (
            ("uuid", uuid),
            ("reference_sha256", reference_sha256),
            ("normalized_ast_sha256", ast_sha256),
        ):
            seen[key][value] = global_index

    combined = pa.concat_tables(tables)
    metadata = dict(combined.schema.metadata or {})
    metadata.update(
        {
            b"release.contract": CONTRACT.encode(),
            b"release.selection_policy": b"original_training_priority_then_high_complexity_fixed_source_gate_v1",
            b"release.status": b"formal_prompt_tvm_v4",
            b"release.review_only": b"false",
            b"release.training_approved": b"false",
        }
    )
    output = combined.take(pa.array(retained_indices, type=pa.int64())).replace_schema_metadata(metadata)
    selected_path = output_dir / "train.parquet"
    pq.write_table(output, selected_path, compression="zstd")

    selected_manifest_path = output_dir / "manifest.jsonl"
    with selected_manifest_path.open("w") as handle:
        for output_position, global_index in enumerate(retained_indices):
            source_name, source_row_index, parquet_path, manifest_path = source_locations[global_index]
            uuid, reference_sha256, ast_sha256, _, signature = identities[global_index]
            record = {
                "contract": CONTRACT,
                "output_position": output_position,
                "source_dataset": source_name,
                "source_row_index": source_row_index,
                "source_parquet_path": _display(parquet_path),
                "source_parquet_sha256": input_bindings[source_name]["parquet"]["sha256"],
                "source_manifest_path": _display(manifest_path),
                "source_manifest_sha256": input_bindings[source_name]["manifest"]["sha256"],
                "source_manifest_row_sha256": _sha256_text(manifest_lines[global_index]),
                "uuid": uuid,
                "reference_sha256": reference_sha256,
                "normalized_ast_sha256": ast_sha256,
                "unified_forward_operator_set": list(signature),
                "selection_policy": "original_training_priority_then_high_complexity_fixed_source_gate_v1",
                "release_status": "formal_prompt_tvm_v4",
                "training_approved": False,
            }
            handle.write(_canonical_json(record) + "\n")

    decisions_path = output_dir / "dedup.decisions.jsonl"
    decisions_path.write_text("".join(_canonical_json(row) + "\n" for row in decisions))
    cross_audit_path = output_dir / "cross_source.audit.jsonl"
    cross_audit_path.write_text("".join(_canonical_json(row) + "\n" for row in cross_pairs))
    review_path = output_dir / "review_pairs.md"
    review_rows = _write_review(review_path, decisions, rows, review_pair_count)
    cross_review_path = output_dir / "cross_source_review.md"
    cross_review_rows = _write_cross_review(cross_review_path, cross_pairs, rows)

    final_rows = output.to_pylist()
    final_uuids = [_nested(row, "extra_info.uuid") for row in final_rows]
    final_references = [_sha256_text(str(_nested(row, "reward_model.ground_truth"))) for row in final_rows]
    final_asts = [_normalized_ast_sha256(str(_nested(row, "reward_model.ground_truth"))) for row in final_rows]
    if not (len(set(final_uuids)) == len(set(final_references)) == len(set(final_asts)) == len(final_rows)):
        raise ValueError("final exact identity is not unique")
    if output.num_rows != len(retained_indices):
        raise ValueError("output row count mismatch")

    rejected_with_scores = [
        row for row in decisions if not row["keep"] and row.get("match", {}).get("token_jaccard") is not None
    ]
    cross_candidate_pairs = sum(
        signature_counts["original_training"][signature] * signature_counts["high_complexity"][signature]
        for signature in set(signature_counts["original_training"]) | set(signature_counts["high_complexity"])
    )
    script = Path(__file__).resolve()
    complexity_path = _REPO_ROOT / "tools/data/cleaning/complexity.py"
    dedup_path = _REPO_ROOT / "tools/data/cleaning/deduplicate_ops_candidates.py"
    similarity_path = _REPO_ROOT / "tools/data/cleaning/similarity.py"
    ast_path = _REPO_ROOT / "tools/data/cleaning/ast_similarity.py"
    summary = {
        "contract": CONTRACT,
        "release_status": "formal_prompt_tvm_v4",
        "review_only": False,
        "training_approved": False,
        "selection_policy": {
            "source_priority": list(SOURCE_ORDER),
            "exact_gate": ["uuid", "reference_sha256", "normalized_ast_sha256"],
            "operator_gate": "exact set of tokens re-extracted from the input-dependent Model.forward compute signature",
            "token_jaccard_threshold_strictly_greater_than": TOKEN_THRESHOLD,
            "ast_similarity_threshold_strictly_greater_than": AST_THRESHOLD,
            "near_rule": "operator gate AND (token Jaccard > 0.8 OR Model AST similarity > 0.9)",
            "keeper": "earliest retained row in source-priority order",
        },
        "input_rows": len(rows),
        "output_rows": output.num_rows,
        "removed_rows": len(rows) - output.num_rows,
        "source_rows": {name: sum(counts.values()) for name, counts in signature_counts.items()},
        "retained_rows_by_source": dict(sorted(source_retained.items())),
        "removed_rows_by_source": dict(sorted(source_removed.items())),
        "reason_counts": dict(sorted(counts.items())),
        "unified_operator_gate": {
            "unique_signatures_by_source": {name: len(counts) for name, counts in signature_counts.items()},
            "shared_signature_count": sum(
                bool(signature_counts["original_training"][signature])
                and bool(signature_counts["high_complexity"][signature])
                for signature in set(signature_counts["original_training"]) | set(signature_counts["high_complexity"])
            ),
            "cross_source_candidate_pairs": cross_candidate_pairs,
            "cross_source_rejected_pairs": sum(bool(row["reject"]) for row in cross_pairs),
            "cross_source_score_max": {
                "token_jaccard": max((row["token_jaccard"] for row in cross_pairs), default=None),
                "ast_similarity": max((row["ast_similarity"] for row in cross_pairs), default=None),
            },
        },
        "near_score_distribution": {
            "token_jaccard": _score_summary([row["match"]["token_jaccard"] for row in rejected_with_scores]),
            "ast_similarity": _score_summary([row["match"]["ast_similarity"] for row in rejected_with_scores]),
        },
        "identity": {
            "unique_uuids": len(set(final_uuids)),
            "unique_reference_sha256": len(set(final_references)),
            "unique_normalized_ast_sha256": len(set(final_asts)),
        },
        "source_binding": {
            **input_bindings,
            "builder": _source(script),
            "complexity_extractor": _source(complexity_path),
            "dedup_policy_implementation": _source(dedup_path),
            "token_similarity": _source(similarity_path),
            "ast_similarity": _source(ast_path),
        },
        "artifacts": {
            "train_parquet": _source(selected_path, rows=output.num_rows),
            "manifest": _source(selected_manifest_path, rows=output.num_rows),
            "dedup_decisions": _source(decisions_path, rows=len(decisions)),
            "cross_source_audit": _source(cross_audit_path, rows=len(cross_pairs)),
            "review_pairs": _source(review_path, rows=review_rows),
            "cross_source_review": _source(cross_review_path, rows=cross_review_rows),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--original-parquet",
        type=Path,
        default=_REPO_ROOT
        / "local_artifacts/data/synthesize/prompt_tvm_v4/release_sources/serial_one_question_per_parent.v1/selected.parquet",
    )
    parser.add_argument(
        "--original-manifest",
        type=Path,
        default=_REPO_ROOT
        / "local_artifacts/data/synthesize/prompt_tvm_v4/release_sources/serial_one_question_per_parent.v1/selected.manifest.jsonl",
    )
    parser.add_argument(
        "--high-complexity-parquet",
        type=Path,
        default=_REPO_ROOT
        / "local_artifacts/data/synthesize/csp_dag_low_level_5k/expansion_v3/run.serial_sd.v8/one_question_per_parent.v1/selected.parquet",
    )
    parser.add_argument(
        "--high-complexity-manifest",
        type=Path,
        default=_REPO_ROOT
        / "local_artifacts/data/synthesize/csp_dag_low_level_5k/expansion_v3/run.serial_sd.v8/one_question_per_parent.v1/selected.manifest.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "Data/prompt_tvm_v4/release",
    )
    parser.add_argument("--review-pair-count", type=int, default=12)
    args = parser.parse_args()
    summary = build(
        (
            ("original_training", args.original_parquet, args.original_manifest),
            ("high_complexity", args.high_complexity_parquet, args.high_complexity_manifest),
        ),
        args.output_dir,
        args.review_pair_count,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
