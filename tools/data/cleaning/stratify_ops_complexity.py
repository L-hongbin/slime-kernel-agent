#!/usr/bin/env python3
"""Label an ops parquet by KernelBench-like complexity and cap Level1 rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.data.cleaning.complexity import (  # noqa: E402
    FEATURE_NAMES,
    KERNELBENCH_LEVELS,
    KernelBenchTaxonomyClassifier,
    deterministic_level1_selection,
    extract_complexity_features,
    extract_operator_signature,
    feature_dict,
    level1_cap_candidate_bases,
)
from tools.data.cleaning.pipeline import _sha256_file, _write_filtered_parquet  # noqa: E402


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _reference_classifier(path: Path, neighbors: int) -> tuple[KernelBenchTaxonomyClassifier, Counter[str]]:
    features = []
    signatures = []
    labels = []
    reference_ids = []
    source_counts: Counter[str] = Counter()
    for batch in pq.ParquetFile(path).iter_batches(columns=["reward_model", "extra_info"], batch_size=1024):
        reward_models = batch.column(batch.schema.get_field_index("reward_model")).to_pylist()
        extra_infos = batch.column(batch.schema.get_field_index("extra_info")).to_pylist()
        for reward_model, extra_info in zip(reward_models, extra_infos, strict=True):
            code = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
            level = extra_info.get("level") if isinstance(extra_info, dict) else None
            reference_id = extra_info.get("uuid") if isinstance(extra_info, dict) else None
            if not isinstance(code, str) or not code:
                raise ValueError(f"reference row has no ground_truth code: {path}")
            if level not in KERNELBENCH_LEVELS:
                raise ValueError(f"reference row has invalid KernelBench level {level!r}: {path}")
            if not isinstance(reference_id, str) or not reference_id:
                reference_id = f"reference_{len(reference_ids)}"
            features.append(extract_complexity_features(code))
            signatures.append(extract_operator_signature(code))
            labels.append(level)
            reference_ids.append(reference_id)
            source_counts[level] += 1
    return (
        KernelBenchTaxonomyClassifier(features, signatures, labels, reference_ids, neighbors=neighbors),
        source_counts,
    )


def _rank_hash(seed: int, uuid: str) -> str:
    return hashlib.sha256(f"{seed}:{uuid}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--assignments-jsonl", required=True, type=Path)
    parser.add_argument("--summary-json", required=True, type=Path)
    parser.add_argument("--neighbors", type=int, default=3)
    parser.add_argument("--max-level1-fraction", type=float, default=0.30)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    for path in (args.output, args.assignments_jsonl, args.summary_json):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")

    classifier, reference_counts = _reference_classifier(args.reference, args.neighbors)
    loo_report = classifier.leave_one_out_report()
    assignments: list[dict[str, Any]] = []
    predicted_counts: Counter[str] = Counter()
    prediction_basis_counts: Counter[str] = Counter()
    row_index = 0
    for batch in pq.ParquetFile(args.input).iter_batches(
        columns=["reward_model", "extra_info"], batch_size=args.batch_size
    ):
        reward_models = batch.column(batch.schema.get_field_index("reward_model")).to_pylist()
        extra_infos = batch.column(batch.schema.get_field_index("extra_info")).to_pylist()
        for reward_model, extra_info in zip(reward_models, extra_infos, strict=True):
            code = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
            uuid = extra_info.get("uuid") if isinstance(extra_info, dict) else None
            if not isinstance(code, str) or not code:
                raise ValueError(f"input row {row_index} has no ground_truth code")
            if not isinstance(uuid, str) or not uuid:
                raise ValueError(f"input row {row_index} has no extra_info.uuid")
            features = extract_complexity_features(code)
            signature = extract_operator_signature(code)
            prediction = classifier.predict(features, signature)
            predicted_counts[prediction.level] += 1
            prediction_basis_counts[prediction.basis] += 1
            cap_candidate_bases = level1_cap_candidate_bases(features, signature, prediction)
            level1_cap_candidate = bool(cap_candidate_bases)
            assignments.append(
                {
                    "row_index": row_index,
                    "uuid": uuid,
                    "kernelbench_like_level": prediction.level,
                    "vote_count": prediction.vote_count,
                    "prediction_basis": prediction.basis,
                    "structural_knn_level": prediction.structural_level,
                    "level1_cap_candidate": level1_cap_candidate,
                    "level1_cap_candidate_bases": list(cap_candidate_bases),
                    "operator_signature": list(signature),
                    "features": feature_dict(features),
                    "neighbors": [
                        {
                            "reference_id": neighbor.reference_id,
                            "level": neighbor.level,
                            "distance": neighbor.distance,
                        }
                        for neighbor in prediction.neighbors
                    ],
                }
            )
            row_index += 1

    selected_indices, cap_report = deterministic_level1_selection(
        (
            (
                record["row_index"],
                record["uuid"],
                "level1" if record["level1_cap_candidate"] else record["kernelbench_like_level"],
            )
            for record in assignments
        ),
        max_fraction=args.max_level1_fraction,
        seed=args.seed,
    )
    for record in assignments:
        record["selected"] = record["row_index"] in selected_indices
        if record["kernelbench_like_level"] == "level1":
            record["selection_rank_hash"] = _rank_hash(args.seed, record["uuid"])
        elif record["level1_cap_candidate"]:
            record["selection_rank_hash"] = _rank_hash(args.seed, record["uuid"])

    selected_level_counts = Counter(record["kernelbench_like_level"] for record in assignments if record["selected"])
    dropped_level_counts = Counter(
        record["kernelbench_like_level"] for record in assignments if not record["selected"]
    )
    strict_level1_rows = predicted_counts["level1"]
    cap_report.update(
        {
            "level1_cap_candidate_definition": (
                "taxonomy Level1 OR structural k-NN Level1 OR a simple-envelope program with at "
                "most one input-dependent compute token after removing explicit nn.Identity; "
                "ambiguous rows are conservatively capped because domain-shifted structural labels "
                "cannot prove that every short program is non-Level1"
            ),
            "level1_cap_candidate_rows": cap_report["input_level1_rows"],
            "strict_predicted_level1_rows": strict_level1_rows,
        }
    )

    decisions = [record["selected"] for record in assignments]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_filtered_parquet(
        args.input,
        args.output,
        decisions,
        {},
        batch_size=args.batch_size,
        compression=args.compression,
    )
    assignments_text = "".join(json.dumps(record, sort_keys=True) + "\n" for record in assignments)
    _atomic_write_text(args.assignments_jsonl, assignments_text)
    summary = {
        "contract": "kernelbench_operator_signature_plus_structural_knn_v3",
        "input": str(args.input.resolve()),
        "input_sha256": _sha256_file(args.input),
        "reference": str(args.reference.resolve()),
        "reference_sha256": _sha256_file(args.reference),
        "output": str(args.output.resolve()),
        "output_sha256": _sha256_file(args.output),
        "assignments_jsonl": str(args.assignments_jsonl.resolve()),
        "assignments_jsonl_sha256": _sha256_file(args.assignments_jsonl),
        "features": list(FEATURE_NAMES),
        "neighbors": args.neighbors,
        "reference_level_counts": dict(sorted(reference_counts.items())),
        "leave_one_out_validation": loo_report,
        "predicted_level_counts": dict(sorted(predicted_counts.items())),
        "selected_predicted_level_counts": {level: selected_level_counts[level] for level in KERNELBENCH_LEVELS},
        "dropped_predicted_level_counts": {level: dropped_level_counts[level] for level in KERNELBENCH_LEVELS},
        "prediction_basis_counts": dict(sorted(prediction_basis_counts.items())),
        "predicted_level_fractions": {
            level: predicted_counts[level] / len(assignments) if assignments else 0.0 for level in KERNELBENCH_LEVELS
        },
        "selection": cap_report,
        "notes": [
            "This is a KernelBench-taxonomy calibration, not a validation-contamination check.",
            "Only effective Model AST structure and source length are used; get_inputs calls are excluded.",
            "Level1 requires one input-dependent compute token or an exact official Level1 operator signature.",
            "Structural k-NN only separates rows that do not pass the Level1 signature gate.",
            f"The {100 * args.max_level1_fraction:g}% cap conservatively includes structural-k-NN Level1 ambiguities as Level1 candidates.",
            "Cap accounting additionally treats explicit Identity wrappers around at most one compute token as Level1-like.",
            "Rows retain source order; only hash-ranked Level1-like rows may be removed.",
        ],
    }
    _atomic_write_text(args.summary_json, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
