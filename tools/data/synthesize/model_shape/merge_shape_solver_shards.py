#!/usr/bin/env python3
"""Validate and merge modulo-sharded shape-solver runs in global parent order."""

from __future__ import annotations

import argparse
import collections
import copy
import difflib
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
from tools.data.synthesize.model_shape.shard_shape_solver_input import CONTRACT_VERSION as SHARD_CONTRACT_VERSION
from tools.data.synthesize.model_shape.shard_shape_solver_input import MANIFEST_NAME, shard_name

CONTRACT_VERSION = "shape_solver_shard_merge_v2"
VARIANT_ORDER = {"medium": 0, "large": 1}
MAX_REVIEW_EXAMPLES = 16


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"JSON object required: {path}")
    return loaded


def _nested(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _row_uuid(row: Mapping[str, Any]) -> str:
    value = _nested(row, "extra_info.uuid")
    if not isinstance(value, str) or not value:
        raise ValueError("row is missing extra_info.uuid")
    return value


def _row_reference(row: Mapping[str, Any]) -> str:
    value = _nested(row, "reward_model.ground_truth")
    if not isinstance(value, str):
        raise ValueError(f"row {_row_uuid(row)} is missing reward_model.ground_truth")
    return value


def _logical_target_map_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    canonical = "".join(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records).encode(
        "utf-8"
    )
    return _sha256_bytes(canonical)


def _known_run_artifacts(run_dir: Path) -> dict[str, Path]:
    return {
        "selected": run_dir / "selected.parquet",
        "selection": run_dir / "selection.json",
        "targets": run_dir / "targets.parquet",
        "children": run_dir / "static" / "children.parquet",
        "paired": run_dir / "static" / "paired.parquet",
        "review": run_dir / "analysis" / "review_samples.md",
    }


def _verify_artifact_hashes(run_dir: Path, manifest: Mapping[str, Any]) -> None:
    expected = manifest.get("artifact_sha256")
    if not isinstance(expected, Mapping):
        raise ValueError(f"missing artifact_sha256: {run_dir}")
    for key, path in _known_run_artifacts(run_dir).items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if expected.get(key) != _sha256_file(path):
            raise ValueError(f"artifact hash mismatch: {run_dir}:{key}")


def _review_markdown(
    source_rows: Sequence[Mapping[str, Any]],
    child_by_uuid: Mapping[str, Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Shape solver accepted diffs",
        "",
        "Representative source-span-only edits from the globally ordered merged run.",
        "",
    ]
    accepted = [decision for decision in decisions if decision.get("accepted")]
    for decision in accepted[:MAX_REVIEW_EXAMPLES]:
        global_index = int(decision["source_row_index"])
        parent = source_rows[global_index]
        child_uuid = str(decision["child_uuid"])
        child = child_by_uuid[child_uuid]
        solver = decision.get("solver", {})
        slot_ids: list[str] = []
        if isinstance(solver, Mapping):
            raw_slots = solver.get("slots")
            if isinstance(raw_slots, list):
                for raw_edit in raw_slots:
                    raw_slot = raw_edit.get("slot") if isinstance(raw_edit, Mapping) else None
                    if isinstance(raw_slot, Mapping):
                        slot_ids.append(str(raw_slot.get("slot_id", "unknown")))
            elif isinstance(solver.get("slot"), Mapping):
                slot_ids.append(str(solver["slot"].get("slot_id", "unknown")))
        slot_label = ", ".join(slot_ids) if slot_ids else "unknown"
        lines.extend(
            [
                f"## {child_uuid} ({decision['variant']})",
                "",
                (
                    f"Parent `{decision['parent_uuid']}`; slots `{slot_label}`; "
                    f"{decision['input_bytes_before']} -> {decision['input_bytes_after']} bytes; "
                    f"target error {float(decision['target_relative_error']):.6%}."
                ),
                "",
                "```diff",
            ]
        )
        lines.extend(
            difflib.unified_diff(
                _row_reference(parent).splitlines(),
                _row_reference(child).splitlines(),
                fromfile="parent.py",
                tofile="child.py",
                lineterm="",
            )
        )
        lines.extend(["```", ""])
    return "\n".join(lines).rstrip() + "\n"


def _manifest_invariants(manifest: Mapping[str, Any]) -> dict[str, Any]:
    target_contract = copy.deepcopy(manifest.get("target_contract"))
    if not isinstance(target_contract, dict):
        raise ValueError("target_contract must be an object")
    target_contract.pop("logical_target_map_sha256", None)
    invariants = {
        key: manifest.get(key)
        for key in (
            "contract_version",
            "generator_version",
            "solver_source_sha256",
            "v3_helper_source_sha256",
            "method_boundary",
            "runtime_validation_required",
            "training_approved",
        )
    }
    invariants["solver_contracts"] = {
        key: copy.deepcopy(value)
        for key, value in manifest.items()
        if key.endswith("_contract") and key != "target_contract"
    }
    invariants["target_contract_without_logical_hash"] = target_contract
    return invariants


def merge_shards(
    shard_input_root: Path,
    shard_run_root: Path,
    output_dir: Path,
    *,
    source_selected: Path | None = None,
) -> dict[str, Any]:
    shard_input_root = shard_input_root.resolve()
    shard_run_root = shard_run_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing directory: {output_dir}")

    shard_manifest_path = shard_input_root / MANIFEST_NAME
    shard_manifest = _load_json(shard_manifest_path)
    if shard_manifest.get("contract_version") != SHARD_CONTRACT_VERSION:
        raise ValueError("unexpected shard input contract")
    shard_count = int(shard_manifest["shard_count"])
    if shard_count <= 0:
        raise ValueError("invalid shard count")
    manifest_shards = shard_manifest.get("shards")
    if not isinstance(manifest_shards, list) or len(manifest_shards) != shard_count:
        raise ValueError("shard input manifest is incomplete")

    selected_path = (
        source_selected.resolve()
        if source_selected is not None
        else Path(str(shard_manifest["source_selected"])).resolve()
    )
    if not selected_path.is_file():
        raise FileNotFoundError(selected_path)
    if _sha256_file(selected_path) != shard_manifest.get("source_selected_sha256"):
        raise ValueError("source selected hash does not match shard input manifest")
    source = pq.read_table(selected_path)
    source_rows = source.to_pylist()
    if source.num_rows != int(shard_manifest["selected_rows"]):
        raise ValueError("source selected row count does not match shard manifest")

    all_targets: list[dict[str, Any]] = []
    all_decisions: list[dict[str, Any]] = []
    child_by_uuid: dict[str, dict[str, Any]] = {}
    counters: collections.Counter[str] = collections.Counter()
    skip_reasons: collections.Counter[str] = collections.Counter()
    run_manifest_records: list[dict[str, Any]] = []
    target_schema: pa.Schema | None = None
    first_manifest: dict[str, Any] | None = None
    expected_invariants: dict[str, Any] | None = None
    seen_global_indices: set[int] = set()

    for shard_index in range(shard_count):
        expected_name = shard_name(shard_index, shard_count)
        shard_record = manifest_shards[shard_index]
        if not isinstance(shard_record, Mapping):
            raise ValueError(f"invalid shard record {shard_index}")
        if shard_record.get("name") != expected_name or int(shard_record["shard_index"]) != shard_index:
            raise ValueError(f"shard manifest order mismatch at {shard_index}")
        global_indices = [int(value) for value in shard_record["global_indices"]]
        expected_indices = list(range(shard_index, source.num_rows, shard_count))
        if global_indices != expected_indices:
            raise ValueError(f"global index map mismatch for {expected_name}")
        if seen_global_indices.intersection(global_indices):
            raise ValueError(f"duplicate global index in {expected_name}")
        seen_global_indices.update(global_indices)

        input_dir = shard_input_root / expected_name
        input_selected = input_dir / "selected.parquet"
        input_selection = input_dir / "selection.json"
        if _sha256_file(input_selected) != shard_record.get("selected_sha256"):
            raise ValueError(f"input selected hash mismatch for {expected_name}")
        if _sha256_file(input_selection) != shard_record.get("selection_sha256"):
            raise ValueError(f"input selection hash mismatch for {expected_name}")
        expected_table = source.take(pa.array(global_indices, type=pa.int64()))
        if not pq.read_table(input_selected).equals(expected_table):
            raise ValueError(f"input selected rows differ from canonical source: {expected_name}")

        run_dir = shard_run_root / expected_name
        run_manifest_path = run_dir / "static" / "manifest.json"
        run_manifest = _load_json(run_manifest_path)
        _verify_artifact_hashes(run_dir, run_manifest)
        if _sha256_file(run_dir / "selected.parquet") != _sha256_file(input_selected):
            raise ValueError(f"solver selected copy differs for {expected_name}")
        run_selection = _load_json(run_dir / "selection.json")
        if [int(value) for value in run_selection.get("global_indices", [])] != global_indices:
            raise ValueError(f"solver output lost global index map: {expected_name}")
        if int(run_manifest.get("selected_rows", -1)) != len(global_indices):
            raise ValueError(f"solver selected row count mismatch: {expected_name}")

        invariants = _manifest_invariants(run_manifest)
        if expected_invariants is None:
            expected_invariants = invariants
            first_manifest = run_manifest
        elif invariants != expected_invariants:
            raise ValueError(f"solver contract or source hash differs: {expected_name}")

        for key, value in run_manifest.get("counts", {}).items():
            counters[str(key)] += int(value)
        for key, value in run_manifest.get("skip_reason_counts", {}).items():
            skip_reasons[str(key)] += int(value)

        local_decisions: list[dict[str, Any]] = []
        for raw_decision in run_manifest.get("decisions", []):
            if not isinstance(raw_decision, Mapping):
                raise ValueError(f"invalid decision in {expected_name}")
            decision = copy.deepcopy(dict(raw_decision))
            local_index = int(decision["source_row_index"])
            if not 0 <= local_index < len(global_indices):
                raise ValueError(f"decision index out of range in {expected_name}")
            global_index = global_indices[local_index]
            if decision.get("parent_uuid") != _row_uuid(source_rows[global_index]):
                raise ValueError(f"decision parent lineage mismatch in {expected_name}")
            decision["shard_index"] = shard_index
            decision["shard_source_row_index"] = local_index
            decision["source_row_index"] = global_index
            local_decisions.append(decision)
            all_decisions.append(decision)

        target_table = pq.read_table(run_dir / "targets.parquet")
        if target_schema is None:
            target_schema = target_table.schema
        elif target_table.schema != target_schema:
            raise ValueError(f"target schema mismatch in {expected_name}")
        for target in target_table.to_pylist():
            local_index = int(target["selected_index"])
            if not 0 <= local_index < len(global_indices):
                raise ValueError(f"target index out of range in {expected_name}")
            global_index = global_indices[local_index]
            if target.get("parent_uuid") != _row_uuid(source_rows[global_index]):
                raise ValueError(f"target parent lineage mismatch in {expected_name}")
            target["selected_index"] = global_index
            all_targets.append(target)

        shard_children = pq.read_table(run_dir / "static" / "children.parquet")
        shard_paired = pq.read_table(run_dir / "static" / "paired.parquet")
        if shard_children.schema != source.schema or shard_paired.schema != source.schema:
            raise ValueError(f"row schema mismatch in {expected_name}")
        local_child_rows = shard_children.to_pylist()
        local_child_uuids = [_row_uuid(row) for row in local_child_rows]
        accepted_local = [str(decision["child_uuid"]) for decision in local_decisions if decision.get("accepted")]
        if local_child_uuids != accepted_local:
            raise ValueError(f"children are not in accepted-decision order: {expected_name}")
        for child in local_child_rows:
            child_uuid = _row_uuid(child)
            if child_uuid in child_by_uuid:
                raise ValueError(f"duplicate child UUID: {child_uuid}")
            child_by_uuid[child_uuid] = child

        accepted_by_local: dict[int, list[str]] = collections.defaultdict(list)
        for decision in local_decisions:
            if decision.get("accepted"):
                accepted_by_local[int(decision["shard_source_row_index"])].append(str(decision["child_uuid"]))
        expected_paired_uuids: list[str] = []
        for local_index, global_index in enumerate(global_indices):
            child_uuids = accepted_by_local.get(local_index, [])
            if child_uuids:
                expected_paired_uuids.append(_row_uuid(source_rows[global_index]))
                expected_paired_uuids.extend(child_uuids)
        if [_row_uuid(row) for row in shard_paired.to_pylist()] != expected_paired_uuids:
            raise ValueError(f"paired rows are not in local parent order: {expected_name}")

        run_manifest_records.append(
            {
                "shard_index": shard_index,
                "name": expected_name,
                "manifest_sha256": _sha256_file(run_manifest_path),
                "selected_rows": len(global_indices),
                "children_written": len(local_child_rows),
            }
        )

    if seen_global_indices != set(range(source.num_rows)):
        raise ValueError("shards do not cover every global selected index exactly once")
    if first_manifest is None or target_schema is None:
        raise ValueError("no solver shards were loaded")

    all_decisions.sort(
        key=lambda item: (
            int(item["source_row_index"]),
            VARIANT_ORDER.get(str(item.get("variant")), 99),
        )
    )
    decision_keys = [(int(item["source_row_index"]), str(item["variant"])) for item in all_decisions]
    if len(decision_keys) != len(set(decision_keys)):
        raise ValueError("duplicate global decision key")
    all_targets.sort(
        key=lambda item: (
            int(item["selected_index"]),
            VARIANT_ORDER.get(str(item.get("variant")), 99),
        )
    )
    target_keys = [(int(item["selected_index"]), str(item["variant"])) for item in all_targets]
    if target_keys != decision_keys:
        raise ValueError("targets and decisions do not have the same global keys")

    children: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    accepted_by_global: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for decision in all_decisions:
        if not decision.get("accepted"):
            continue
        global_index = int(decision["source_row_index"])
        child_uuid = str(decision["child_uuid"])
        child = child_by_uuid.get(child_uuid)
        if child is None:
            raise ValueError(f"accepted decision has no child row: {child_uuid}")
        parent_uuid = _row_uuid(source_rows[global_index])
        if _nested(child, "extra_info.v4.parent_uuid") != parent_uuid:
            raise ValueError(f"child parent lineage mismatch: {child_uuid}")
        child_hash = _sha256_bytes(_row_reference(child).encode("utf-8"))
        if child_hash != decision.get("child_reference_sha256"):
            raise ValueError(f"child reference hash mismatch: {child_uuid}")
        accepted_by_global[global_index].append(child)
        children.append(child)
    if len(children) != len(child_by_uuid):
        raise ValueError("one or more child rows are not referenced by accepted decisions")
    for global_index, parent in enumerate(source_rows):
        parent_children = accepted_by_global.get(global_index, [])
        if parent_children:
            paired.append(copy.deepcopy(parent))
            paired.extend(parent_children)

    if counters["parents_scanned"] != source.num_rows:
        raise ValueError("merged parents_scanned does not equal selected row count")
    if counters["children_written"] != len(children):
        raise ValueError("merged children_written count mismatch")
    if counters["paired_rows_written"] != len(paired):
        raise ValueError("merged paired_rows_written count mismatch")
    if counters["parents_with_children"] != len(accepted_by_global):
        raise ValueError("merged parents_with_children count mismatch")

    variants_per_parent = int(_nested(first_manifest, "target_contract.variants_per_parent", 2))
    if variants_per_parent not in (1, 2):
        raise ValueError(f"unsupported variants_per_parent: {variants_per_parent}")
    target_indices = sorted({int(row["selected_index"]) for row in all_targets})
    if variants_per_parent == 1:
        expected_indices = list(range(source.num_rows))
        if target_indices != expected_indices:
            raise ValueError("single-variant targets do not cover every selected parent")
        decision_indices = [int(item["source_row_index"]) for item in all_decisions]
        if decision_indices != expected_indices:
            raise ValueError("single-variant decisions are not one-per-selected-parent")
    target_maps: list[dict[str, Any]] = []
    for global_index in target_indices:
        parent_targets = [row for row in all_targets if int(row["selected_index"]) == global_index]
        if len(parent_targets) != variants_per_parent:
            raise ValueError(
                f"unexpected target count at global index {global_index}: "
                f"{len(parent_targets)} != {variants_per_parent}"
            )
        first = parent_targets[0]
        if variants_per_parent == 1:
            variant = str(first["variant"])
            if variant not in VARIANT_ORDER:
                raise ValueError(f"unknown target variant at global index {global_index}: {variant}")
            target_maps.append(
                {
                    "selected_index": global_index,
                    "parent_uuid": first["parent_uuid"],
                    "parent_reference_sha256": first["parent_reference_sha256"],
                    "variant": variant,
                    "target_input_bytes": int(first["target_input_bytes"]),
                }
            )
        else:
            variants = {str(row["variant"]): int(row["target_input_bytes"]) for row in parent_targets}
            if set(variants) != set(VARIANT_ORDER):
                raise ValueError(f"incomplete target variants at global index {global_index}")
            target_maps.append(
                {
                    "selected_index": global_index,
                    "parent_uuid": first["parent_uuid"],
                    "parent_reference_sha256": first["parent_reference_sha256"],
                    "target_input_bytes": variants,
                }
            )
    logical_target_hash = _logical_target_map_sha256(target_maps)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        (temporary_dir / "static").mkdir(parents=True)
        (temporary_dir / "analysis").mkdir(parents=True)
        shutil.copy2(selected_path, temporary_dir / "selected.parquet")
        pq.write_table(
            pa.Table.from_pylist(all_targets, schema=target_schema),
            temporary_dir / "targets.parquet",
            compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pylist(children, schema=source.schema),
            temporary_dir / "static" / "children.parquet",
            compression="zstd",
        )
        pq.write_table(
            pa.Table.from_pylist(paired, schema=source.schema),
            temporary_dir / "static" / "paired.parquet",
            compression="zstd",
        )
        review_path = temporary_dir / "analysis" / "review_samples.md"
        review_path.write_text(_review_markdown(source_rows, child_by_uuid, all_decisions), encoding="utf-8")

        source_selection_path = selected_path.with_name("selection.json")
        source_selection = _load_json(source_selection_path) if source_selection_path.is_file() else {}
        source_selection_rows = source_selection.get("rows", [])
        if source_selection_rows and len(source_selection_rows) != source.num_rows:
            raise ValueError("source selection rows do not match source selected parquet")
        selection = {
            **source_selection,
            "derivation_contract_version": first_manifest["contract_version"],
            "merge_contract_version": CONTRACT_VERSION,
            "source_selection_manifest": (
                str(source_selection_path.resolve()) if source_selection_path.is_file() else None
            ),
            "source_selection_manifest_sha256": (
                _sha256_file(source_selection_path) if source_selection_path.is_file() else None
            ),
            "shard_input_manifest": str(shard_manifest_path),
            "shard_input_manifest_sha256": _sha256_file(shard_manifest_path),
            "shard_count": shard_count,
            "shard_scheme": "global_index_modulo_shard_count",
            "selected_path": str(output_dir / "selected.parquet"),
            "selected_sha256": _sha256_file(temporary_dir / "selected.parquet"),
            "selected_count": source.num_rows,
            "rows": source_selection_rows,
        }
        selection_path = temporary_dir / "selection.json"
        selection_path.write_text(json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        target_contract = copy.deepcopy(first_manifest["target_contract"])
        target_contract["logical_target_map_sha256"] = logical_target_hash
        final_artifacts = {
            "selected": str(output_dir / "selected.parquet"),
            "selection": str(output_dir / "selection.json"),
            "targets": str(output_dir / "targets.parquet"),
            "children": str(output_dir / "static" / "children.parquet"),
            "paired": str(output_dir / "static" / "paired.parquet"),
            "review": str(output_dir / "analysis" / "review_samples.md"),
        }
        temporary_artifacts = _known_run_artifacts(temporary_dir)
        final_manifest = {
            key: copy.deepcopy(first_manifest[key])
            for key in (
                "contract_version",
                "generator_version",
                "solver_source_path",
                "solver_source_sha256",
                "v3_helper_source_path",
                "v3_helper_source_sha256",
                "method_boundary",
                "runtime_validation_required",
                "training_approved",
            )
        }
        final_manifest.update(
            {key: copy.deepcopy(value) for key, value in first_manifest.items() if key.endswith("_contract")}
        )
        final_manifest.update(
            {
                "target_contract": target_contract,
                "selected_source": str(selected_path),
                "selected_source_sha256": _sha256_file(selected_path),
                "selected_rows": source.num_rows,
                "artifacts": final_artifacts,
                "artifact_sha256": {key: _sha256_file(path) for key, path in temporary_artifacts.items()},
                "counts": dict(sorted(counters.items())),
                "skip_reason_counts": dict(sorted(skip_reasons.items())),
                "decisions": all_decisions,
                "merge": {
                    "contract_version": CONTRACT_VERSION,
                    "merge_tool": str(Path(__file__).resolve()),
                    "merge_tool_sha256": _sha256_file(Path(__file__)),
                    "shard_input_manifest": str(shard_manifest_path),
                    "shard_input_manifest_sha256": _sha256_file(shard_manifest_path),
                    "shard_run_root": str(shard_run_root),
                    "shard_count": shard_count,
                    "shard_scheme": "global_index_modulo_shard_count",
                    "shards": run_manifest_records,
                },
            }
        )
        accepted_decisions = [decision for decision in all_decisions if decision.get("accepted") is True]
        if accepted_decisions and all(
            type(_nested(decision, "solver.changed_occurrences")) is int
            and type(_nested(decision, "solver.power_of_two_occurrences")) is int
            for decision in accepted_decisions
        ):
            changed_occurrences = sum(
                int(_nested(decision, "solver.changed_occurrences")) for decision in accepted_decisions
            )
            power_occurrences = sum(
                int(_nested(decision, "solver.power_of_two_occurrences")) for decision in accepted_decisions
            )
            power_fraction = power_occurrences / changed_occurrences if changed_occurrences else None
            declared_range = _nested(
                final_manifest,
                "distribution_contract.power_of_two_occurrence_fraction_range",
            )
            range_passed = (
                isinstance(declared_range, list)
                and len(declared_range) == 2
                and all(type(value) in (int, float) for value in declared_range)
                and power_fraction is not None
                and float(declared_range[0]) <= power_fraction <= float(declared_range[1])
            )
            final_manifest["distribution_observation"] = {
                "static_changed_occurrences": changed_occurrences,
                "static_power_of_two_occurrences": power_occurrences,
                "static_power_of_two_occurrence_fraction": power_fraction,
                "static_power_of_two_range_passed": range_passed,
                "scope": "globally_merged_static_accepted_children",
            }
        manifest_path = temporary_dir / "static" / "manifest.json"
        manifest_path.write_text(
            json.dumps(final_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_dir, output_dir)
        return final_manifest
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="new merged solver run directory")
    parser.add_argument("--shard-input-root", type=Path, required=True)
    parser.add_argument("--shard-run-root", type=Path, required=True)
    parser.add_argument(
        "--source-selected",
        type=Path,
        help="canonical selected parquet; defaults to the path recorded by shards.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    manifest = merge_shards(
        args.shard_input_root,
        args.shard_run_root,
        args.output_dir,
        source_selected=args.source_selected,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "selected_rows": manifest["selected_rows"],
                "counts": manifest["counts"],
                "logical_target_map_sha256": manifest["target_contract"]["logical_target_map_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
