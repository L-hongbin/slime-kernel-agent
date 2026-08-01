"""Aggregate a complete row-level cleanup audit without re-running references."""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .pipeline import _runtime_failure_category, _sha256_file


def summarize_cleanup_audit(
    audit_path: Path,
    *,
    input_path: Path | None = None,
    clean_path: Path | None = None,
    contract_version: int = 5,
    title: str = "Ops training-data cleanup",
) -> dict[str, Any]:
    reason_counts: collections.Counter[str] = collections.Counter()
    flag_counts: collections.Counter[str] = collections.Counter()
    repair_counts: collections.Counter[str] = collections.Counter()
    attempted_repair_counts: collections.Counter[str] = collections.Counter()
    runtime_counts: collections.Counter[str] = collections.Counter()
    runtime_failure_categories: collections.Counter[str] = collections.Counter()
    runtime_sensitivity_probe_counts: collections.Counter[str] = collections.Counter()
    train_eval_counts: collections.Counter[str] = collections.Counter()
    runtime_source_counts: collections.Counter[str] = collections.Counter()
    primary_counts: collections.Counter[str] = collections.Counter()
    rows = 0
    kept = 0
    quarantined = 0

    with audit_path.open(encoding="utf-8") as handle:
        for expected_index, line in enumerate(handle):
            record = json.loads(line)
            if record.get("row_index") != expected_index:
                raise ValueError(
                    f"audit row_index mismatch at line {expected_index + 1}: " f"found {record.get('row_index')!r}"
                )
            rows += 1
            is_kept = bool(record.get("keep"))
            kept += is_kept
            quarantined += bool(record.get("quarantined"))
            reasons = list(record.get("reasons") or ())
            flags = list(record.get("flags") or ())
            mode_flags = list(record.get("mode_flags") or ())
            repairs = dict(record.get("repairs") or {})
            reason_counts.update(reasons)
            flag_counts.update(flag.split(":", 1)[0] for flag in flags)
            # ``repairs`` maps field names to their replacement values.  Count
            # repaired fields, not the (often string-valued) replacements.
            attempted_repair_counts.update(repairs.keys())
            if is_kept:
                repair_counts.update(repairs.keys())
            primary_counts[reasons[0] if reasons else "kept"] += 1
            verdict = str(record.get("runtime_verdict", "missing"))
            runtime_counts[verdict] += 1
            runtime_source_counts[str(record.get("runtime_source", "missing"))] += 1
            train_eval_counts.update(mode_flags)
            if verdict == "Failed":
                runtime_failure_categories[_runtime_failure_category(str(record.get("runtime_detail", "")))] += 1
            elif verdict == "synthetic_sensitivity_only":
                match = re.search(r"synthetic probe ([^ ]+)", str(record.get("runtime_detail", "")))
                runtime_sensitivity_probe_counts[match.group(1) if match else "per_argument"] += 1

    if input_path is not None:
        input_rows = pq.ParquetFile(input_path).metadata.num_rows
        if input_rows != rows:
            raise ValueError(f"input parquet has {input_rows} rows but audit has {rows}")
    if clean_path is not None:
        clean_rows = pq.ParquetFile(clean_path).metadata.num_rows
        if clean_rows != kept:
            raise ValueError(f"clean parquet has {clean_rows} rows but audit keeps {kept}")

    summary: dict[str, Any] = {
        "title": title,
        "contract_version": contract_version,
        "audit_jsonl": str(audit_path.resolve()),
        "audit_jsonl_sha256": _sha256_file(audit_path),
        "input_rows_considered": rows,
        "output_rows": kept,
        "rejected_rows": rows - kept,
        "quarantined_rows": quarantined,
        "retention_rate": kept / rows if rows else 0.0,
        "reason_counts_nonexclusive": dict(sorted(reason_counts.items())),
        "primary_reason_counts": dict(sorted(primary_counts.items())),
        "flag_counts_nonexclusive": dict(sorted(flag_counts.items())),
        "repair_counts": dict(sorted(repair_counts.items())),
        "attempted_repair_counts": dict(sorted(attempted_repair_counts.items())),
        "runtime_verdict_counts": dict(sorted(runtime_counts.items())),
        "runtime_failure_categories": dict(sorted(runtime_failure_categories.items())),
        "runtime_sensitivity_probe_counts": dict(sorted(runtime_sensitivity_probe_counts.items())),
        "train_eval_counts_nonexclusive": dict(sorted(train_eval_counts.items())),
        "runtime_source_counts": dict(sorted(runtime_source_counts.items())),
        "configuration": {"aggregated_from_complete_audit": True},
    }
    if input_path is not None:
        summary.update(input=str(input_path.resolve()), input_sha256=_sha256_file(input_path))
    if clean_path is not None:
        summary.update(output=str(clean_path.resolve()), output_sha256=_sha256_file(clean_path))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-jsonl", required=True, type=Path)
    parser.add_argument("--output-summary", required=True, type=Path)
    parser.add_argument("--input-parquet", type=Path)
    parser.add_argument("--clean-parquet", type=Path)
    parser.add_argument("--contract-version", type=int, default=5)
    parser.add_argument("--title", default="Ops training-data cleanup")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.output_summary.exists() and not args.overwrite:
        raise FileExistsError(f"refusing to overwrite {args.output_summary}; pass --overwrite")
    summary = summarize_cleanup_audit(
        args.audit_jsonl,
        input_path=args.input_parquet,
        clean_path=args.clean_parquet,
        contract_version=args.contract_version,
        title=args.title,
    )
    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_summary.with_name(f".{args.output_summary.name}.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output_summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
