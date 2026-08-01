"""Split, merge, and overlay sharded cleanup artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _uuid_values(table: pa.Table) -> list[str]:
    uuids = table.column("extra_info").combine_chunks().field("uuid").to_pylist()
    if not all(isinstance(value, str) and value for value in uuids):
        raise ValueError("every input row must have a non-empty extra_info.uuid")
    return uuids


def _read_audits(paths: Sequence[Path]) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                uuid = record.get("uuid")
                if not isinstance(uuid, str) or not uuid:
                    raise ValueError(f"missing UUID in {path}:{line_number}")
                if uuid in records:
                    raise ValueError(f"duplicate audit UUID {uuid!r} in {path}:{line_number}")
                records[uuid] = record
    return records


def split_parquet_contiguous(
    *,
    input_path: Path,
    output_dir: Path,
    shards: int,
    prefix: str | None = None,
    manifest_path: Path | None = None,
    compression: str = "zstd",
) -> dict[str, Any]:
    """Split a parquet into balanced contiguous shards without changing row order."""

    if shards <= 0:
        raise ValueError("shards must be positive")
    table = pq.read_table(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = prefix or input_path.stem
    base, remainder = divmod(len(table), shards)
    outputs: list[dict[str, Any]] = []
    offset = 0
    for shard_index in range(shards):
        row_count = base + (1 if shard_index < remainder else 0)
        output_path = output_dir / f"{stem}.shard{shard_index:02d}.parquet"
        pq.write_table(table.slice(offset, row_count), output_path, compression=compression)
        outputs.append(
            {
                "index": shard_index,
                "row_offset": offset,
                "rows": row_count,
                "path": str(output_path.resolve()),
                "sha256": _sha256_file(output_path),
            }
        )
        offset += row_count
    manifest = {
        "input": str(input_path.resolve()),
        "input_sha256": _sha256_file(input_path),
        "input_rows": len(table),
        "shards": outputs,
        "order": "contiguous source row order",
    }
    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def overlay_runtime_audits(
    *,
    base_audit_path: Path,
    runtime_audit_paths: Sequence[Path],
    output_audit_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Overlay shard runtime evidence onto a complete source-row static audit."""

    if not runtime_audit_paths:
        raise ValueError("at least one runtime audit JSONL is required")
    base_records: list[dict[str, Any]] = []
    effective_to_index: dict[str, int] = {}
    with base_audit_path.open(encoding="utf-8") as handle:
        for expected_index, line in enumerate(handle):
            record = json.loads(line)
            if record.get("row_index") != expected_index:
                raise ValueError(
                    f"base audit row_index mismatch at line {expected_index + 1}: "
                    f"found {record.get('row_index')!r}"
                )
            repairs = record.get("repairs") or {}
            effective_uuid = repairs.get("uuid", record.get("uuid"))
            if not isinstance(effective_uuid, str) or not effective_uuid:
                raise ValueError(f"base audit has no effective UUID at row {expected_index}")
            if effective_uuid in effective_to_index:
                raise ValueError(f"duplicate effective UUID in base audit: {effective_uuid!r}")
            effective_to_index[effective_uuid] = expected_index
            base_records.append(record)

    runtime_records = _read_audits(runtime_audit_paths)
    unmatched = sorted(set(runtime_records) - set(effective_to_index))
    if unmatched:
        raise ValueError(f"runtime audit UUIDs absent from base audit: {unmatched[:5]}")

    overlay_fields = (
        "flags",
        "keep",
        "mode_detail",
        "mode_flags",
        "quarantined",
        "reasons",
        "runtime_detail",
        "runtime_source",
        "runtime_verdict",
    )
    for effective_uuid, runtime_record in runtime_records.items():
        base_record = base_records[effective_to_index[effective_uuid]]
        if base_record.get("entry_point") != runtime_record.get("entry_point") or base_record.get(
            "semantic_hash"
        ) != runtime_record.get("semantic_hash"):
            raise ValueError(f"runtime audit identity mismatch for effective UUID {effective_uuid!r}")
        for field in overlay_fields:
            base_record[field] = runtime_record.get(field)

    uncovered_static_survivors = [
        record.get("row_index")
        for record in base_records
        if record.get("keep") and record.get("runtime_verdict") == "skipped_by_option"
    ]
    if uncovered_static_survivors:
        raise ValueError("base static survivors lack runtime evidence: " f"{uncovered_static_survivors[:5]}")

    output_audit_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with output_audit_path.open("w", encoding="utf-8") as handle:
        for row_index, record in enumerate(base_records):
            record["row_index"] = row_index
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    manifest = {
        "base_audit": str(base_audit_path.resolve()),
        "base_audit_sha256": _sha256_file(base_audit_path),
        "runtime_audits": [
            {"path": str(path.resolve()), "sha256": _sha256_file(path)} for path in runtime_audit_paths
        ],
        "base_rows": len(base_records),
        "runtime_rows_overlaid": len(runtime_records),
        "output_audit": str(output_audit_path.resolve()),
        "output_audit_sha256": _sha256_file(output_audit_path),
        "alignment": "runtime UUID to base effective UUID (repairs.uuid or source uuid)",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def merge_cleanup_shards(
    *,
    input_paths: Sequence[Path],
    audit_paths: Sequence[Path],
    source_order_path: Path,
    output_input_path: Path,
    output_audit_path: Path,
    manifest_path: Path,
    compression: str = "zstd",
) -> dict[str, Any]:
    if not input_paths:
        raise ValueError("at least one input parquet is required")
    if not audit_paths:
        raise ValueError("at least one audit JSONL is required")

    tables = [pq.read_table(path) for path in input_paths]
    schema = tables[0].schema
    for path, table in zip(input_paths[1:], tables[1:], strict=True):
        if table.schema != schema:
            raise ValueError(f"parquet schema mismatch: {path}")
    combined = pa.concat_tables(tables)
    uuids = _uuid_values(combined)
    if len(set(uuids)) != len(uuids):
        raise ValueError("input parquets contain duplicate UUIDs")

    source_uuids = pq.read_table(source_order_path, columns=["extra_info.uuid"])["uuid"].to_pylist()
    source_positions = {uuid: index for index, uuid in enumerate(source_uuids)}
    missing_from_source = sorted(set(uuids) - set(source_positions))
    if missing_from_source:
        raise ValueError(f"input UUIDs absent from source-order parquet: {missing_from_source[:5]}")
    order = sorted(range(len(uuids)), key=lambda index: source_positions[uuids[index]])
    ordered = combined.take(pa.array(order, type=pa.int64()))
    ordered_uuids = [uuids[index] for index in order]

    audits = _read_audits(audit_paths)
    missing_audits = sorted(set(ordered_uuids) - set(audits))
    extra_audits = sorted(set(audits) - set(ordered_uuids))
    if missing_audits or extra_audits:
        raise ValueError(
            "audit/input UUID coverage mismatch: " f"missing={missing_audits[:5]} extra={extra_audits[:5]}"
        )

    output_input_path.parent.mkdir(parents=True, exist_ok=True)
    output_audit_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(ordered, output_input_path, compression=compression)
    with output_audit_path.open("w", encoding="utf-8") as handle:
        for row_index, uuid in enumerate(ordered_uuids):
            record = dict(audits[uuid])
            record["row_index"] = row_index
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    manifest = {
        "source_order": str(source_order_path.resolve()),
        "source_order_sha256": _sha256_file(source_order_path),
        "input_parquets": [{"path": str(path.resolve()), "sha256": _sha256_file(path)} for path in input_paths],
        "audit_jsonl": [{"path": str(path.resolve()), "sha256": _sha256_file(path)} for path in audit_paths],
        "output_rows": len(ordered_uuids),
        "output_input": str(output_input_path.resolve()),
        "output_input_sha256": _sha256_file(output_input_path),
        "output_audit": str(output_audit_path.resolve()),
        "output_audit_sha256": _sha256_file(output_audit_path),
        "order": "source parquet row order by unique extra_info.uuid",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    split_parser = commands.add_parser(
        "split",
        help="Split a parquet into balanced, source-ordered shards.",
    )
    split_parser.add_argument("--input", required=True, type=Path)
    split_parser.add_argument("--output-dir", required=True, type=Path)
    split_parser.add_argument("--shards", required=True, type=int)
    split_parser.add_argument("--prefix")
    split_parser.add_argument("--manifest", type=Path)
    split_parser.add_argument("--compression", default="zstd")

    merge_parser = commands.add_parser(
        "merge",
        help="Merge audited shard inputs in original source order.",
    )
    merge_parser.add_argument("--input", action="append", required=True, type=Path)
    merge_parser.add_argument("--audit-jsonl", action="append", required=True, type=Path)
    merge_parser.add_argument("--source-order", required=True, type=Path)
    merge_parser.add_argument("--output-input", required=True, type=Path)
    merge_parser.add_argument("--output-audit", required=True, type=Path)
    merge_parser.add_argument("--manifest", required=True, type=Path)
    merge_parser.add_argument("--compression", default="zstd")

    overlay_parser = commands.add_parser(
        "overlay",
        help="Overlay shard runtime evidence onto a complete static audit.",
    )
    overlay_parser.add_argument("--base-audit", required=True, type=Path)
    overlay_parser.add_argument("--runtime-audit", action="append", required=True, type=Path)
    overlay_parser.add_argument("--output-audit", required=True, type=Path)
    overlay_parser.add_argument("--manifest", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "split":
        manifest = split_parquet_contiguous(
            input_path=args.input,
            output_dir=args.output_dir,
            shards=args.shards,
            prefix=args.prefix,
            manifest_path=args.manifest,
            compression=args.compression,
        )
    elif args.command == "merge":
        manifest = merge_cleanup_shards(
            input_paths=args.input,
            audit_paths=args.audit_jsonl,
            source_order_path=args.source_order,
            output_input_path=args.output_input,
            output_audit_path=args.output_audit,
            manifest_path=args.manifest,
            compression=args.compression,
        )
    elif args.command == "overlay":
        manifest = overlay_runtime_audits(
            base_audit_path=args.base_audit,
            runtime_audit_paths=args.runtime_audit,
            output_audit_path=args.output_audit,
            manifest_path=args.manifest,
        )
    else:  # pragma: no cover - argparse restricts this value
        raise AssertionError(args.command)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
