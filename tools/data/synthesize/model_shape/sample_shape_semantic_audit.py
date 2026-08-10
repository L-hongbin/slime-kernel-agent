#!/usr/bin/env python3
"""Materialize a deterministic semantic-audit sample of runtime-valid shape children."""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.cleaning.complexity import extract_operator_signature
from tools.data.synthesize.augment_prompt_tasks import (
    _factory_records,
    _module_constant_environment,
    _top_level_function,
)
from tools.data.synthesize.model_shape.resample_shape_coverage import ChildShapeProfile, _load_runtime_lane
from tools.data.synthesize.profile_prompt_tvm_distribution import _classify_signature

DEFAULT_CENSUS = REPO_ROOT / "Data/prompt_tvm_v4/shape_runtime_coverage_resample_v5/run.low128k_final/summary.json"
DEFAULT_PARENTS = REPO_ROOT / "Data/prompt_tvm_v4/train.review.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "local_artifacts/data/synthesize/shape_semantic_audit"
AUDIT_VERSION = "shape_semantic_audit_sample_v1"


@dataclass(frozen=True)
class AuditItem:
    child_uuid: str
    parent_uuid: str
    lane_id: str
    lane_kind: str
    source_family: str
    operator_families: tuple[str, ...]
    variant: str
    logical_slots: int
    changed_factories: int
    input_scale: float
    input_bytes_after: int
    maximum_axis: int
    maximum_axis_ratio: float
    stable_hash: str


def _nested(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _sha(*parts: object) -> str:
    return hashlib.sha256(":".join(str(part) for part in parts).encode()).hexdigest()


def _records(code: str) -> list[Any]:
    tree = ast.parse(code)
    function = _top_level_function(tree, "get_inputs")
    records, _ = _factory_records(tree, function, _module_constant_environment(tree))
    return sorted(records, key=lambda item: (item.line_number, item.column_offset))


def _positive_input_constants(code: str) -> list[int]:
    tree = ast.parse(code)
    function = _top_level_function(tree, "get_inputs")
    return [
        int(node.value)
        for node in ast.walk(function)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
        and node.value > 0
    ]


def _parent_uuids(path: Path) -> set[str]:
    result: set[str] = set()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["extra_info"], batch_size=1024):
        for extra in batch.column(0).to_pylist():
            uuid = extra.get("uuid") if isinstance(extra, Mapping) else None
            if isinstance(uuid, str):
                if uuid in result:
                    raise ValueError(f"duplicate parent UUID: {uuid}")
                result.add(uuid)
    return result


def _make_item(
    row: Mapping[str, Any],
    profile: ChildShapeProfile,
    lane_id: str,
    lane_kind: str,
) -> AuditItem:
    child_code = _nested(row, "reward_model.ground_truth")
    child_uuid = _nested(row, "extra_info.uuid")
    parent_uuid = _nested(row, "extra_info.v4.parent_uuid")
    if not all(isinstance(value, str) for value in (child_code, child_uuid, parent_uuid)):
        raise ValueError("invalid child row identity")
    try:
        records = _records(child_code)
        changed_records = [
            records[item.factory_index] for item in profile.changed_factories if 0 <= item.factory_index < len(records)
        ]
        changed_shapes = [record.shape for record in changed_records]
    except (SyntaxError, ValueError):
        changed_shapes = []
    if changed_shapes:
        maximum_axis = max(value for shape in changed_shapes for value in shape)
        maximum_axis_ratio = max(
            max(shape) / max(1, min(shape)) if len(shape) > 1 else 1.0 for shape in changed_shapes
        )
    else:
        constants = _positive_input_constants(child_code)
        maximum_axis = max(constants, default=1)
        maximum_axis_ratio = maximum_axis / max(1, min(constants, default=1))
    source_family = str(_nested(row, "extra_info.v4.source_family", "unknown"))
    try:
        operator_families = tuple(sorted(_classify_signature(extract_operator_signature(child_code))))
    except (SyntaxError, ValueError):
        operator_families = ("other_only",)
    return AuditItem(
        child_uuid=child_uuid,
        parent_uuid=parent_uuid,
        lane_id=lane_id,
        lane_kind=lane_kind,
        source_family=source_family,
        operator_families=operator_families,
        variant=profile.variant,
        logical_slots=profile.logical_slots,
        changed_factories=len(profile.changed_factories),
        input_scale=profile.input_scale,
        input_bytes_after=profile.input_bytes_after,
        maximum_axis=maximum_axis,
        maximum_axis_ratio=maximum_axis_ratio,
        stable_hash=_sha(AUDIT_VERSION, child_uuid),
    )


def _load_items(census_path: Path, parent_path: Path) -> list[AuditItem]:
    census = json.loads(census_path.read_text())
    parents = _parent_uuids(parent_path)
    result: list[AuditItem] = []
    seen: set[str] = set()
    for source in census["source_runs"]:
        run_dir = Path(source["run_dir"])
        eligible, _, children_path, _, _, profiles, lane_kind = _load_runtime_lane(run_dir)
        parquet = pq.ParquetFile(children_path)
        found: set[str] = set()
        for batch in parquet.iter_batches(columns=["reward_model", "extra_info"], batch_size=512):
            rewards = batch.column(0).to_pylist()
            extras = batch.column(1).to_pylist()
            for reward, extra in zip(rewards, extras, strict=True):
                child_uuid = extra.get("uuid") if isinstance(extra, Mapping) else None
                if child_uuid not in eligible:
                    continue
                parent_uuid = _nested(extra, "v4.parent_uuid")
                if not isinstance(parent_uuid, str) or parent_uuid not in parents:
                    raise ValueError(f"child lacks canonical parent reference: {child_uuid}:{parent_uuid}")
                row = {"reward_model": reward, "extra_info": extra}
                result.append(
                    _make_item(
                        row,
                        profiles[str(child_uuid)],
                        str(source["lane_id"]),
                        lane_kind,
                    )
                )
                found.add(str(child_uuid))
                if child_uuid in seen:
                    raise ValueError(f"duplicate child UUID: {child_uuid}")
                seen.add(str(child_uuid))
        if found != eligible:
            raise ValueError(f"runtime lane accounting mismatch: {run_dir}")
    expected = sum(int(source["runtime_eligible_children"]) for source in census["source_runs"])
    if len(result) != expected:
        raise ValueError(f"child census mismatch: {len(result)} != {expected}")
    return result


def _select(items: Sequence[AuditItem], count: int) -> list[AuditItem]:
    if count > len(items):
        raise ValueError("sample count exceeds population")
    selected: dict[str, AuditItem] = {}

    def add(item: AuditItem) -> None:
        if len(selected) < count:
            selected.setdefault(item.child_uuid, item)

    risk_orders = (
        sorted(items, key=lambda item: (-item.maximum_axis_ratio, item.stable_hash)),
        sorted(items, key=lambda item: (-item.maximum_axis, item.stable_hash)),
        sorted(items, key=lambda item: (-item.input_scale, item.stable_hash)),
        sorted(items, key=lambda item: (-item.input_bytes_after, item.stable_hash)),
        sorted(items, key=lambda item: (-item.changed_factories, -item.logical_slots, item.stable_hash)),
    )
    for ordered in risk_orders:
        for item in ordered[: max(12, count // 12)]:
            add(item)

    strata: dict[tuple[str, str], list[AuditItem]] = collections.defaultdict(list)
    for item in items:
        values = {
            "lane": item.lane_id,
            "kind": item.lane_kind,
            "source": item.source_family,
            "variant": item.variant,
            "slot_count": str(item.logical_slots),
            "factory_count": str(item.changed_factories),
        }
        for family in item.operator_families:
            strata[("operator_family", family)].append(item)
        for name, value in values.items():
            strata[(name, value)].append(item)
    ordered_strata = sorted(strata, key=lambda key: (_sha("stratum", *key), key))
    cursor = 0
    while len(selected) < count:
        progress = False
        for key in ordered_strata:
            options = sorted(strata[key], key=lambda item: item.stable_hash)
            if cursor < len(options):
                before = len(selected)
                add(options[cursor])
                progress |= len(selected) > before
                if len(selected) == count:
                    break
        if not progress and cursor >= max(len(values) for values in strata.values()):
            break
        cursor += 1
    for item in sorted(items, key=lambda item: item.stable_hash):
        add(item)
    return sorted(selected.values(), key=lambda item: item.stable_hash)


def _write_manifest(path: Path, items: Sequence[AuditItem]) -> None:
    header = (
        "child_uuid\tparent_uuid\tlane_kind\tlane_id\tsource_family\toperator_families\t"
        "variant\tlogical_slots\tchanged_factories\tinput_scale\tinput_bytes_after\tmaximum_axis\tmaximum_axis_ratio"
    )
    lines = [header]
    for item in items:
        values = (
            item.child_uuid,
            item.parent_uuid,
            item.lane_kind,
            item.lane_id,
            item.source_family,
            ",".join(item.operator_families),
            item.variant,
            str(item.logical_slots),
            str(item.changed_factories),
            f"{item.input_scale:.6f}",
            str(item.input_bytes_after),
            str(item.maximum_axis),
            f"{item.maximum_axis_ratio:.6f}",
        )
        lines.append("\t".join(values))
    path.write_text("\n".join(lines) + "\n")


def run(census: Path, parents: Path, output: Path, sample_count: int) -> None:
    items = _load_items(census, parents)
    sample = _select(items, sample_count)
    output.mkdir(parents=True, exist_ok=True)
    _write_manifest(output / "manifest.tsv", sample)
    lane_counts = collections.Counter(item.lane_kind for item in sample)
    family_counts = collections.Counter(family for item in sample for family in item.operator_families)
    (output / "README.md").write_text(
        "\n".join(
            [
                "# Shape semantic audit sample",
                "",
                "结论和人工复核见 `SUMMARY.md`。",
                "",
                f"- Population: {len(items):,} runtime-valid children",
                f"- Sample: {len(sample):,} unique children",
                f"- Lane kinds: {dict(sorted(lane_counts.items()))}",
                f"- Operator-family occurrences: {dict(sorted(family_counts.items()))}",
                "- Selection: deterministic heuristic-extreme anchors plus round-robin coverage of lane, source, variant, slot/factory count and operator family; this is not an equal-probability sample.",
                "- This is a semantic audit, not a new runtime validation or training approval.",
                "",
            ]
        )
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census", type=Path, default=DEFAULT_CENSUS)
    parser.add_argument("--parents", type=Path, default=DEFAULT_PARENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-count", type=int, default=264)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run(args.census, args.parents, args.output, args.sample_count)
