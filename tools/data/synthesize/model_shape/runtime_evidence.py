#!/usr/bin/env python3
"""Verify model-shape H20 evidence and materialize runtime-accepted children."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pyarrow.parquet as pq
from tools.data.synthesize.model_shape.pipeline import (
    _atomic_parquet,
    _atomic_text,
    _nested,
    _paths,
    _rows,
    _sha256_bytes,
    _sha256_file,
)
from tools.data.synthesize.model_shape.validate_shape_region_liveness import _accepted_tasks
from tools.data.synthesize.model_shape.verify_shape_solver_runtime_shards import (
    _verified_canonical_reference_passes,
    verify,
)


def _topology(machine_count: int, virtual_shards_per_gpu: int) -> dict[str, int]:
    if machine_count <= 0 or virtual_shards_per_gpu <= 0:
        raise ValueError("runtime topology values must be positive")
    return {
        "machine_count": machine_count,
        "gpus_per_machine": 8,
        "virtual_shards_per_gpu": virtual_shards_per_gpu,
    }


def prepare_region(
    run_dir: Path,
    *,
    reference_topology: Mapping[str, int],
) -> dict[str, Any]:
    reference = verify(
        run_dir,
        "reference",
        None,
        expected_topology=reference_topology,
    )
    output = _paths(run_dir)
    tasks = _accepted_tasks(output.selected, output.children, output.manifest)
    pass_by_uuid = _verified_canonical_reference_passes(run_dir.resolve(), expected_topology=reference_topology)
    missing = sorted(
        {
            str(task[field])
            for task in tasks
            for field in ("parent_uuid", "child_uuid")
            if str(task[field]) not in pass_by_uuid
        }
    )
    if missing:
        raise ValueError(f"reference evidence lacks task UUIDs: {missing[:20]}")
    allowlist = [
        str(task["child_uuid"])
        for task in tasks
        if pass_by_uuid[str(task["parent_uuid"])] and pass_by_uuid[str(task["child_uuid"])]
    ]
    if not allowlist:
        raise ValueError("reference produced an empty region allowlist")
    allowlist_path = run_dir / "analysis/region_reference_both_pass_uuids.txt"
    _atomic_text(allowlist_path, "".join(f"{uuid}\n" for uuid in allowlist))
    summary = {
        "contract_version": "model_shape_runtime_reference_analysis_v1",
        "run_dir": str(run_dir.resolve()),
        "reference_verifier": reference,
        "static_children": len(tasks),
        "parent_child_both_pass": len(allowlist),
        "allowlist_path": str(allowlist_path.resolve()),
        "allowlist_sha256": _sha256_file(allowlist_path),
        "reference_topology": dict(reference_topology),
    }
    _atomic_text(
        run_dir / "analysis/runtime_reference.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    return summary


def _region_records(run_dir: Path) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    for path in sorted((run_dir / "h20/region").glob("shard-*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                uuid = record.get("child_uuid")
                if not isinstance(uuid, str) or not uuid:
                    raise ValueError(f"{path}:{line_number}: missing child UUID")
                if uuid in records:
                    raise ValueError(f"duplicate region child UUID: {uuid}")
                records[uuid] = record
    return records


def finalize(
    run_dir: Path,
    *,
    reference_topology: Mapping[str, int],
    region_topology: Mapping[str, int],
) -> dict[str, Any]:
    reference = verify(
        run_dir,
        "reference",
        None,
        expected_topology=reference_topology,
    )
    region = verify(
        run_dir,
        "region",
        None,
        expected_topology=region_topology,
        expected_reference_topology=reference_topology,
    )
    output = _paths(run_dir)
    tasks = _accepted_tasks(output.selected, output.children, output.manifest)
    task_by_uuid = {str(task["child_uuid"]): task for task in tasks}
    records = _region_records(run_dir)
    allowlist = [
        line.strip()
        for line in (run_dir / "analysis/region_reference_both_pass_uuids.txt")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    if set(records) != set(allowlist):
        raise ValueError(f"region UUID set differs from allowlist: {len(records)}:{len(allowlist)}")
    accepted_uuids = [uuid for uuid in allowlist if records[uuid].get("passed") is True]
    children = {str(_nested(row, "extra_info.uuid")): row for row in _rows(output.children)}
    accepted_rows = [children[uuid] for uuid in accepted_uuids]
    accepted_path = run_dir / "runtime/accepted.parquet"
    _atomic_parquet(
        accepted_path,
        accepted_rows,
        pq.ParquetFile(output.children).schema_arrow,
    )
    accepted_manifest = []
    for uuid in accepted_uuids:
        task = task_by_uuid[uuid]
        child = children[uuid]
        child_code = _nested(child, "reward_model.ground_truth")
        if not isinstance(child_code, str):
            raise ValueError(f"accepted child has no reference: {uuid}")
        reference_sha256 = _sha256_bytes(child_code.encode())
        if _nested(child, "extra_info.v4.reference_sha256") != reference_sha256:
            raise ValueError(f"accepted child reference hash mismatch: {uuid}")
        accepted_manifest.append(
            {
                "child_uuid": uuid,
                "parent_uuid": task["parent_uuid"],
                "variant": task["variant"],
                "source_family": str(_nested(child, "extra_info.v4.source_family", "unknown")),
                "operator_family": str(_nested(child, "extra_info.ops", "")),
                "reference_status": "parent_child_both_pass",
                "region_status": "passed",
                "child_reference_sha256": reference_sha256,
            }
        )
    manifest_path = run_dir / "runtime/accepted_manifest.json"
    accepted_artifact = {
        "contract_version": "model_shape_runtime_accepted_v1",
        "run_dir": str(run_dir.resolve()),
        "source_children_sha256": _sha256_file(output.children),
        "source_manifest_sha256": _sha256_file(output.manifest),
        "accepted_path": str(accepted_path.resolve()),
        "accepted_sha256": _sha256_file(accepted_path),
        "accepted_children": len(accepted_rows),
        "accepted_parents": len({str(task_by_uuid[uuid]["parent_uuid"]) for uuid in accepted_uuids}),
        "training_approved": False,
        "rows": accepted_manifest,
    }
    _atomic_text(
        manifest_path,
        json.dumps(accepted_artifact, indent=2, sort_keys=True) + "\n",
    )
    status_counts = collections.Counter(str(record.get("status")) for record in records.values())
    summary = {
        "contract_version": "model_shape_runtime_final_analysis_v1",
        "run_dir": str(run_dir.resolve()),
        "reference_verifier": reference,
        "region_verifier": region,
        "static_children": len(tasks),
        "reference_both_pass": len(allowlist),
        "region_status_counts": dict(sorted(status_counts.items())),
        "runtime_accepted_children": len(accepted_rows),
        "runtime_accepted_parents": accepted_artifact["accepted_parents"],
        "accepted_path": str(accepted_path.resolve()),
        "accepted_sha256": accepted_artifact["accepted_sha256"],
        "accepted_manifest_sha256": _sha256_file(manifest_path),
        "reference_topology": dict(reference_topology),
        "region_topology": dict(region_topology),
        "training_approved": False,
    }
    _atomic_text(
        run_dir / "analysis/runtime_final.json",
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
    )
    markdown = [
        "# Model shape H20 runtime result",
        "",
        f"- Static children: {len(tasks)}.",
        f"- Parent+child reference both pass: {len(allowlist)}.",
        f"- Region statuses: {dict(sorted(status_counts.items()))}.",
        f"- Runtime accepted: {len(accepted_rows)} children over " f"{accepted_artifact['accepted_parents']} parents.",
        "- Governance: review-only; training_approved=false.",
    ]
    _atomic_text(run_dir / "analysis/runtime_final.md", "\n".join(markdown) + "\n")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("command", choices=("prepare-region", "finalize"))
    parser.add_argument("--reference-machine-count", type=int, required=True)
    parser.add_argument("--reference-virtual-shards-per-gpu", type=int, required=True)
    parser.add_argument("--region-machine-count", type=int)
    parser.add_argument("--region-virtual-shards-per-gpu", type=int)
    return parser


def main() -> None:
    args = _parser().parse_args()
    reference_topology = _topology(args.reference_machine_count, args.reference_virtual_shards_per_gpu)
    if args.command == "prepare-region":
        result = prepare_region(args.run_dir, reference_topology=reference_topology)
    else:
        if args.region_machine_count is None or args.region_virtual_shards_per_gpu is None:
            raise ValueError("finalize requires the region topology")
        result = finalize(
            args.run_dir,
            reference_topology=reference_topology,
            region_topology=_topology(args.region_machine_count, args.region_virtual_shards_per_gpu),
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
