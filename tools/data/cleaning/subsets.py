#!/usr/bin/env python3
"""Materialize reproducible row subsets from a complete cleanup audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .pipeline import _runtime_failure_category, _sha256_file, _write_filtered_parquet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--audit-jsonl", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument(
        "--keep-status",
        choices=("kept", "rejected", "any"),
        default="any",
        help="Select the audit's final keep decision before applying other selectors.",
    )
    parser.add_argument("--runtime-verdict", action="append", default=[])
    parser.add_argument("--failure-category", action="append", default=[])
    parser.add_argument("--require-any-mode-flag", action="append", default=[])
    parser.add_argument("--exclude-mode-flag", action="append", default=[])
    parser.add_argument(
        "--align-by-effective-uuid",
        action="store_true",
        help=(
            "Align an already-cleaned/filtered input to the complete source audit "
            "using repairs.uuid or the source UUID instead of row_index. Repairs are "
            "not applied again in this mode."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--compression", default="zstd")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def select_record(record: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.keep_status == "kept" and not record.get("keep", False):
        return False
    if args.keep_status == "rejected" and record.get("keep", False):
        return False
    if args.runtime_verdict and record.get("runtime_verdict") not in args.runtime_verdict:
        return False
    if args.failure_category:
        category = _runtime_failure_category(str(record.get("runtime_detail", "")))
        if category not in args.failure_category:
            return False
    mode_flags = set(record.get("mode_flags", ()))
    if args.require_any_mode_flag and not mode_flags.intersection(args.require_any_mode_flag):
        return False
    if mode_flags.intersection(args.exclude_mode_flag):
        return False
    return True


def align_records_by_effective_uuid(records: list[dict[str, Any]], input_uuids: list[str]) -> list[dict[str, Any]]:
    """Align cleaned output rows to their complete-audit decisions."""

    by_uuid: dict[str, dict[str, Any]] = {}
    for record in records:
        repairs = record.get("repairs") or {}
        effective_uuid = repairs.get("uuid", record.get("uuid"))
        if not isinstance(effective_uuid, str) or not effective_uuid:
            raise ValueError(f"audit row {record.get('row_index')} has no effective UUID")
        if effective_uuid in by_uuid:
            raise ValueError(f"duplicate effective UUID in audit: {effective_uuid!r}")
        by_uuid[effective_uuid] = record

    if len(set(input_uuids)) != len(input_uuids):
        raise ValueError("input parquet contains duplicate UUIDs")
    missing = [uuid for uuid in input_uuids if uuid not in by_uuid]
    if missing:
        raise ValueError(f"input UUIDs absent from audit: {missing[:5]}")
    return [by_uuid[uuid] for uuid in input_uuids]


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    summary_path = args.summary_json or args.output.with_name(f"{args.output.stem}.summary.json")
    for path in (args.output, summary_path):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"refusing to overwrite {path}; pass --overwrite")

    records: list[dict[str, Any]] = []
    with args.audit_jsonl.open(encoding="utf-8") as handle:
        for expected_index, line in enumerate(handle):
            record = json.loads(line)
            if record.get("row_index") != expected_index:
                raise ValueError(
                    f"audit row_index mismatch at line {expected_index + 1}: " f"found {record.get('row_index')!r}"
                )
            records.append(record)

    input_rows = pq.ParquetFile(args.input).metadata.num_rows
    if args.align_by_effective_uuid:
        input_uuids = pq.read_table(args.input, columns=["extra_info.uuid"])["uuid"].to_pylist()
        if not all(isinstance(uuid, str) and uuid for uuid in input_uuids):
            raise ValueError("every input row must have a non-empty extra_info.uuid")
        aligned_records = align_records_by_effective_uuid(records, input_uuids)
        repairs: dict[int, dict[str, Any]] = {}
    else:
        if len(records) != input_rows:
            raise ValueError(f"audit has {len(records)} rows but parquet has {input_rows}")
        aligned_records = records
        repairs = {index: dict(record["repairs"]) for index, record in enumerate(records) if record.get("repairs")}
    decisions = [select_record(record, args) for record in aligned_records]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_filtered_parquet(
        args.input,
        args.output,
        decisions,
        repairs,
        batch_size=args.batch_size,
        compression=args.compression,
    )
    summary = {
        "input": str(args.input.resolve()),
        "input_sha256": _sha256_file(args.input),
        "audit_jsonl": str(args.audit_jsonl.resolve()),
        "audit_jsonl_sha256": _sha256_file(args.audit_jsonl),
        "output": str(args.output.resolve()),
        "output_sha256": _sha256_file(args.output),
        "input_rows": input_rows,
        "output_rows": sum(decisions),
        "selector": {
            "keep_status": args.keep_status,
            "runtime_verdicts": sorted(set(args.runtime_verdict)),
            "failure_categories": sorted(set(args.failure_category)),
            "require_any_mode_flags": sorted(set(args.require_any_mode_flag)),
            "exclude_mode_flags": sorted(set(args.exclude_mode_flag)),
            "align_by_effective_uuid": args.align_by_effective_uuid,
        },
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = summary_path.with_name(f".{summary_path.name}.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
