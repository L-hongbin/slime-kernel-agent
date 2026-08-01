#!/usr/bin/env python3
"""Assemble node-local PP checkpoint shards into a PP1 resume checkpoint.

Adapter-only torch_dist checkpoints live on per-node ``/nfs``. Under PP2 each
node retains only its pipeline stage's ``.distcp`` files and a node-local
component marker. A PP1 resume owns both stages, so every train node must see
the union of those files and a marker whose LoRA/Muon counts describe the
combined model. This tool creates that union in a new directory; it never
modifies either source checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
import tempfile
from pathlib import Path


SHARED_FILES = (".metadata", "common.pt", "metadata.json", "v4_lora_scaling.json")
MARKER_FILE = "v4_adapter_checkpoint.json"
SUM_FIELDS = (
    "lora_param_tensors",
    "lora_param_numel",
    "optimizer_param_tensors",
    "optimizer_param_numel",
    "fp32_master_tensors",
    "fp32_master_numel",
    "fp32_master_bytes",
    "optimizer_state_tensors",
    "optimizer_state_numel",
    "optimizer_state_bytes",
)
IDENTICAL_MARKER_FIELDS = ("schema_version", "saved_optimizer", "saved_rng")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_marker(source: Path) -> dict:
    marker_path = source / MARKER_FILE
    try:
        marker = json.loads(marker_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid adapter marker: {marker_path}: {exc}") from exc
    expected = set(SUM_FIELDS) | set(IDENTICAL_MARKER_FIELDS)
    if set(marker) != expected:
        raise RuntimeError(
            f"unexpected adapter marker schema in {marker_path}: "
            f"missing={sorted(expected - set(marker))} extra={sorted(set(marker) - expected)}"
        )
    return marker


def _load_metadata(metadata_path: Path):
    try:
        with metadata_path.open("rb") as stream:
            metadata = pickle.load(stream)
    except (OSError, AttributeError, pickle.UnpicklingError) as exc:
        raise RuntimeError(f"cannot read torch_dist metadata {metadata_path}: {exc}") from exc
    return metadata


def _metadata_storage_files(metadata_path: Path) -> set[str]:
    metadata = _load_metadata(metadata_path)
    try:
        paths = {entry.relative_path for entry in metadata.storage_data.values()}
    except AttributeError as exc:
        raise RuntimeError(f"invalid torch_dist storage metadata {metadata_path}: {exc}") from exc
    if not paths or any(not isinstance(path, str) or "/" in path for path in paths):
        raise RuntimeError(f"invalid storage paths in torch_dist metadata {metadata_path}: {paths}")
    return paths


def _metadata_component_stats(metadata_path: Path) -> dict[str, int]:
    metadata = _load_metadata(metadata_path)
    stats = {
        "model_tensors": 0,
        "model_numel": 0,
        "fp32_master_tensors": 0,
        "fp32_master_numel": 0,
        "optimizer_state_tensors": 0,
        "optimizer_state_numel": 0,
        "rng_objects": 0,
    }
    try:
        items = metadata.state_dict_metadata.items()
    except AttributeError as exc:
        raise RuntimeError(f"invalid torch_dist state metadata {metadata_path}: {exc}") from exc
    for key, value in items:
        if not hasattr(value, "size"):
            if str(key).startswith("rng_state/"):
                stats["rng_objects"] += 1
            continue
        numel = math.prod(value.size)
        key = str(key)
        if key.startswith("optimizer.state.fp32_param."):
            prefix = "fp32_master"
        elif key.startswith("optimizer.state.momentum_buffer."):
            prefix = "optimizer_state"
        elif key.startswith("optimizer.state."):
            continue
        else:
            prefix = "model"
        stats[f"{prefix}_tensors"] += 1
        stats[f"{prefix}_numel"] += numel
    return stats


def _validate_sources(sources: list[Path]) -> tuple[dict, set[str]]:
    if len(sources) < 2:
        raise RuntimeError("at least two node-local source iteration directories are required")
    for source in sources:
        if not source.is_dir():
            raise RuntimeError(f"source iteration directory does not exist: {source}")
        for filename in (*SHARED_FILES, MARKER_FILE):
            if not (source / filename).is_file():
                raise RuntimeError(f"source is missing {filename}: {source}")

    for filename in SHARED_FILES:
        hashes = {_sha256(source / filename) for source in sources}
        if len(hashes) != 1:
            raise RuntimeError(f"node-local sources disagree on shared file {filename}")

    markers = [_load_marker(source) for source in sources]
    for field in IDENTICAL_MARKER_FIELDS:
        values = {marker[field] for marker in markers}
        if len(values) != 1:
            raise RuntimeError(f"node-local adapter markers disagree on {field}: {sorted(values)}")

    merged = {field: markers[0][field] for field in IDENTICAL_MARKER_FIELDS}
    for field in SUM_FIELDS:
        values = [marker[field] for marker in markers]
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
            raise RuntimeError(f"invalid {field} values in adapter markers: {values}")
        merged[field] = sum(values)
    if merged["lora_param_numel"] <= 0:
        raise RuntimeError("assembled adapter marker would contain no LoRA parameters")
    for field in ("optimizer_param_numel", "fp32_master_numel"):
        if merged[field] != merged["lora_param_numel"]:
            raise RuntimeError(
                f"assembled marker mismatch: {field}={merged[field]} " f"lora_param_numel={merged['lora_param_numel']}"
            )

    union: set[str] = set()
    for source in sources:
        names = {path.name for path in source.glob("*.distcp")}
        if not names:
            raise RuntimeError(f"source contains no .distcp files: {source}")
        overlap = union & names
        if overlap:
            raise RuntimeError(f"node-local source shard names overlap: {sorted(overlap)}")
        union.update(names)

    referenced = _metadata_storage_files(sources[0] / ".metadata")
    if union != referenced:
        raise RuntimeError(
            "assembled shard set does not match torch_dist metadata: "
            f"missing={sorted(referenced - union)} extra={sorted(union - referenced)}"
        )
    metadata_stats = _metadata_component_stats(sources[0] / ".metadata")
    marker_pairs = {
        "model_tensors": "lora_param_tensors",
        "model_numel": "lora_param_numel",
        "fp32_master_tensors": "fp32_master_tensors",
        "fp32_master_numel": "fp32_master_numel",
        "optimizer_state_tensors": "optimizer_state_tensors",
        "optimizer_state_numel": "optimizer_state_numel",
    }
    mismatches = {
        metadata_field: (metadata_stats[metadata_field], merged[marker_field])
        for metadata_field, marker_field in marker_pairs.items()
        if metadata_stats[metadata_field] != merged[marker_field]
    }
    if mismatches:
        raise RuntimeError(f"assembled marker disagrees with global torch_dist metadata: {mismatches}")
    if merged["saved_rng"] and metadata_stats["rng_objects"] == 0:
        raise RuntimeError("adapter marker says RNG was saved, but metadata contains no RNG objects")
    return merged, union


def assemble_pp1_resume(sources: list[Path], output_root: Path, iteration: int) -> Path:
    """Create ``output_root/iter_N`` from disjoint node-local PP sources."""
    sources = [source.resolve() for source in sources]
    output_root = output_root.resolve()
    target = output_root / f"iter_{iteration:07d}"
    expected_source_name = target.name
    wrong_names = [str(source) for source in sources if source.name != expected_source_name]
    if wrong_names:
        raise RuntimeError(f"source iteration directories must be named {expected_source_name}: {wrong_names}")
    if target.exists():
        raise RuntimeError(f"refusing to overwrite existing assembled checkpoint: {target}")

    merged_marker, shard_names = _validate_sources(sources)
    output_root.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{target.name}.assemble-", dir=output_root))
    try:
        for filename in SHARED_FILES:
            shutil.copy2(sources[0] / filename, tmp / filename)
        for source in sources:
            for shard in sorted(source.glob("*.distcp")):
                shutil.copy2(shard, tmp / shard.name)
                if _sha256(shard) != _sha256(tmp / shard.name):
                    raise RuntimeError(f"post-copy checksum mismatch for {shard}")
        (tmp / MARKER_FILE).write_text(json.dumps(merged_marker, indent=2, sort_keys=True) + "\n")

        copied = {path.name for path in tmp.glob("*.distcp")}
        if copied != shard_names:
            raise RuntimeError(
                f"post-copy shard mismatch: missing={sorted(shard_names - copied)} "
                f"extra={sorted(copied - shard_names)}"
            )
        os.rename(tmp, target)
        latest_tmp = output_root / ".latest_checkpointed_iteration.txt.tmp"
        latest_tmp.write_text(f"{iteration}\n")
        os.replace(latest_tmp, output_root / "latest_checkpointed_iteration.txt")
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return target


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-iter",
        action="append",
        required=True,
        type=Path,
        help="Node-local source iter_N directory; pass once per PP stage/node",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--iteration", required=True, type=int)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    target = assemble_pp1_resume(args.source_iter, args.output_root, args.iteration)
    marker = json.loads((target / MARKER_FILE).read_text())
    print(
        "PP1 adapter resume assembled: "
        f"path={target} shards={len(list(target.glob('*.distcp')))} "
        f"lora_tensors={marker['lora_param_tensors']} "
        f"lora_numel={marker['lora_param_numel']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
