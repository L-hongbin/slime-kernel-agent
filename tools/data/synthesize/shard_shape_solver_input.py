#!/usr/bin/env python3
"""Split a selected parent parquet into deterministic modulo shards.

Each shard carries an explicit local-to-global index map in ``selection.json``.
The existing shape solver can consume the shard without modification.
"""

from __future__ import annotations

import argparse
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


CONTRACT_VERSION = "shape_solver_modulo_shards_v1"
MANIFEST_NAME = "shards.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _nested(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def shard_name(shard_index: int, shard_count: int) -> str:
    width = max(3, len(str(shard_count - 1)))
    return f"shard-{shard_index:0{width}d}-of-{shard_count:0{width}d}"


def shard_selected(
    selected_path: Path,
    output_root: Path,
    *,
    shard_count: int,
) -> dict[str, Any]:
    selected_path = selected_path.resolve()
    output_root = output_root.resolve()
    if not selected_path.is_file():
        raise FileNotFoundError(selected_path)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {output_root}")
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")

    selected = pq.read_table(selected_path)
    if selected.num_rows == 0:
        raise ValueError("selected parquet is empty")
    if shard_count > selected.num_rows:
        raise ValueError("shard_count cannot exceed selected row count")

    source_selection_path = selected_path.with_name("selection.json")
    source_selection: dict[str, Any] = {}
    if source_selection_path.is_file():
        loaded = json.loads(source_selection_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError("source selection manifest must be an object")
        source_selection = loaded
    source_rows = source_selection.get("rows", [])
    if not isinstance(source_rows, list):
        raise ValueError("source selection rows must be a list")
    if source_rows and len(source_rows) != selected.num_rows:
        raise ValueError(
            "source selection row count does not match selected parquet:"
            f"{len(source_rows)}:{selected.num_rows}"
        )
    source_selected_sha256 = _sha256_file(selected_path)
    source_selection_manifest = (
        str(source_selection_path.resolve()) if source_selection_path.is_file() else None
    )
    source_selection_manifest_sha256 = (
        _sha256_file(source_selection_path) if source_selection_path.is_file() else None
    )

    selected_rows = selected.to_pylist()
    identities: list[dict[str, Any]] = []
    for global_index, row in enumerate(selected_rows):
        parent_uuid = _nested(row, "extra_info.uuid")
        reference = _nested(row, "reward_model.ground_truth")
        if not isinstance(parent_uuid, str) or not parent_uuid:
            raise ValueError(f"row {global_index} is missing extra_info.uuid")
        if not isinstance(reference, str):
            raise ValueError(f"row {global_index} is missing reward_model.ground_truth")
        identities.append(
            {
                "global_index": global_index,
                "parent_uuid": parent_uuid,
                "parent_reference_sha256": _sha256_text(reference),
            }
        )

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent)
    )
    try:
        shard_records: list[dict[str, Any]] = []
        for shard_index in range(shard_count):
            name = shard_name(shard_index, shard_count)
            shard_dir = temporary_root / name
            shard_dir.mkdir()
            global_indices = list(range(shard_index, selected.num_rows, shard_count))
            shard_table = selected.take(pa.array(global_indices, type=pa.int64()))
            shard_selected_path = shard_dir / "selected.parquet"
            pq.write_table(shard_table, shard_selected_path, compression="zstd")
            shard_selection = {
                **source_selection,
                "sharding_contract_version": CONTRACT_VERSION,
                "source_selected_path": str(selected_path),
                "source_selected_sha256": source_selected_sha256,
                "source_selection_manifest": source_selection_manifest,
                "source_selection_manifest_sha256": source_selection_manifest_sha256,
                "global_selected_count": selected.num_rows,
                "shard_scheme": "global_index_modulo_shard_count",
                "shard_index": shard_index,
                "shard_count": shard_count,
                "global_indices": global_indices,
                "global_parent_identities": [identities[index] for index in global_indices],
                "selected_path": str((output_root / name / "selected.parquet").resolve()),
                "selected_sha256": _sha256_file(shard_selected_path),
                "selected_count": len(global_indices),
                "rows": [source_rows[index] for index in global_indices]
                if source_rows
                else [],
            }
            shard_selection_path = shard_dir / "selection.json"
            shard_selection_path.write_text(
                json.dumps(shard_selection, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            shard_records.append(
                {
                    "shard_index": shard_index,
                    "name": name,
                    "row_count": len(global_indices),
                    "global_indices": global_indices,
                    "selected_sha256": _sha256_file(shard_selected_path),
                    "selection_sha256": _sha256_file(shard_selection_path),
                }
            )

        manifest = {
            "contract_version": CONTRACT_VERSION,
            "source_selected": str(selected_path),
            "source_selected_sha256": source_selected_sha256,
            "source_selection_manifest": source_selection_manifest,
            "source_selection_manifest_sha256": source_selection_manifest_sha256,
            "selected_rows": selected.num_rows,
            "shard_count": shard_count,
            "shard_scheme": "global_index_modulo_shard_count",
            "shards": shard_records,
            "solver_command_template": (
                "python -m tools.data.synthesize.solve_multidim_shape_coverage "
                "<run-root>/{name} --selected <shard-root>/{name}/selected.parquet"
            ),
        }
        manifest_path = temporary_root / MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary_root, output_root)
        return manifest
    except BaseException:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path, help="new directory for shard inputs")
    parser.add_argument("--selected", type=Path, required=True, help="source selected parquet")
    parser.add_argument("--shard-count", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    manifest = shard_selected(
        args.selected,
        args.output_root,
        shard_count=args.shard_count,
    )
    print(
        json.dumps(
            {
                "output_root": str(args.output_root.resolve()),
                "selected_rows": manifest["selected_rows"],
                "shard_count": manifest["shard_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
