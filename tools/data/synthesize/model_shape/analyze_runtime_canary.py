#!/usr/bin/env python3
"""Summarize reference/region attrition and bias for a model-shape run."""

from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from tools.data.synthesize.model_shape.pipeline import (
    _atomic_text,
    _distance_to_grid,
    _distance_to_power_of_two,
    _nested,
    _paths,
    _rows,
    _sha256_file,
)


def _runtime_records(path: Path, *, uuid_field: str) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    for shard in sorted(path.glob("shard-*.jsonl")):
        with shard.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                uuid = record.get(uuid_field)
                if not isinstance(uuid, str) or not uuid:
                    raise ValueError(f"{shard}:{line_number}: invalid {uuid_field}")
                if uuid in records:
                    raise ValueError(f"duplicate {uuid_field}: {uuid}")
                records[uuid] = record
    return records


def _shard_family_sha256(paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _rate_table(
    decisions: Sequence[Mapping[str, Any]],
    passed: set[str],
    key: Callable[[Mapping[str, Any]], str],
) -> dict[str, dict[str, Any]]:
    totals: collections.Counter[str] = collections.Counter()
    successes: collections.Counter[str] = collections.Counter()
    for decision in decisions:
        label = key(decision)
        totals[label] += 1
        successes[label] += str(decision["child_uuid"]) in passed
    return {
        label: {
            "passed": successes[label],
            "total": totals[label],
            "pass_rate": successes[label] / totals[label],
        }
        for label in sorted(totals)
    }


def _bytes_bucket(value: int) -> str:
    if value < 256 * 2**20:
        return "lt_256_mib"
    if value < 2**30:
        return "256_mib_to_1_gib"
    if value < 2 * 2**30:
        return "1_to_2_gib"
    return "2_to_4_gib"


def _scale_bucket(value: float) -> str:
    if value < 4:
        return "lt_4x"
    if value < 8:
        return "4_to_8x"
    if value < 16:
        return "8_to_16x"
    return "ge_16x"


def _failure_signature(record: Mapping[str, Any]) -> str:
    status = str(record.get("status"))
    tail = str(record.get("worker_stderr_tail") or "").strip().splitlines()
    if tail:
        return f"{status}: {tail[-1][:240]}"
    error = record.get("error")
    if isinstance(error, str) and error:
        return f"{status}: {error[:240]}"
    reasons = record.get("failure_reasons")
    if isinstance(reasons, list) and reasons:
        return f"{status}: {','.join(str(value) for value in reasons)}"
    return status


def _changed_values(decisions: Sequence[Mapping[str, Any]]) -> list[int]:
    return [
        int(slot["new_value"])
        for decision in decisions
        for slot in _nested(decision, "solver.slots", [])
        for _ in slot.get("occurrences", [])
    ]


def _value_bias(decisions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = _changed_values(decisions)
    frequency = collections.Counter(values)
    top = frequency.most_common(10)
    total = len(values)
    hhi = sum((count / total) ** 2 for count in frequency.values()) if total else None
    return {
        "changed_occurrences": total,
        "unique_changed_values": len(frequency),
        "power_of_two_share": (sum(value & (value - 1) == 0 for value in values) / total if total else None),
        "near_decimal_100_grid_share": (
            sum(value >= 100 and _distance_to_grid(value, 100) <= 3 for value in values) / total if total else None
        ),
        "near_power_of_two_within_3_share": (
            sum(value >= 8 and _distance_to_power_of_two(value) <= 3 for value in values) / total if total else None
        ),
        "top10_share": sum(count for _, count in top) / total if total else None,
        "inverse_hhi_effective_values": 1 / hhi if hhi else None,
        "top10": [{"value": value, "occurrences": count} for value, count in top],
    }


def analyze(run_dir: Path, *, allow_partial_reference: bool = False) -> dict[str, Any]:
    output = _paths(run_dir)
    manifest = json.loads(output.manifest.read_text(encoding="utf-8"))
    decisions = [item for item in manifest["decisions"] if isinstance(item, Mapping) and item.get("accepted") is True]
    if len(decisions) != int(manifest["children_written"]):
        raise ValueError("accepted decision count differs from children_written")
    decision_by_child = {str(item["child_uuid"]): item for item in decisions}
    if len(decision_by_child) != len(decisions):
        raise ValueError("duplicate child UUID in manifest decisions")

    reference = _runtime_records(run_dir / "h20/reference", uuid_field="uuid")
    expected_reference = {str(item["parent_uuid"]) for item in decisions} | set(decision_by_child)
    unexpected_reference = set(reference) - expected_reference
    missing_reference = expected_reference - set(reference)
    if unexpected_reference:
        raise ValueError(f"reference contains unexpected UUIDs: {sorted(unexpected_reference)[:20]}")
    if missing_reference and not allow_partial_reference:
        raise ValueError(f"reference UUID set mismatch: {len(reference)} != {len(expected_reference)}")
    missing_path = run_dir / "analysis/reference_missing_uuids.txt"
    if missing_reference:
        _atomic_text(missing_path, "".join(f"{uuid}\n" for uuid in sorted(missing_reference)))
    reference_passed = {uuid for uuid, record in reference.items() if record.get("passed") is True}
    parent_uuids = {str(item["parent_uuid"]) for item in decisions}
    child_uuids = set(decision_by_child)
    parent_passed = parent_uuids & reference_passed
    child_passed = child_uuids & reference_passed

    def reference_outcome(uuid: str) -> str:
        if uuid in missing_reference:
            return "missing"
        return "pass" if uuid in reference_passed else "fail"

    pair_outcomes = collections.Counter(
        (
            f"parent_{reference_outcome(str(item['parent_uuid']))}",
            f"child_{reference_outcome(str(item['child_uuid']))}",
        )
        for item in decisions
    )
    partial_pairs = [
        item
        for item in decisions
        if str(item["parent_uuid"]) in missing_reference or str(item["child_uuid"]) in missing_reference
    ]
    conditional_child_failures = [
        item
        for item in decisions
        if str(item["parent_uuid"]) in reference_passed
        and str(item["child_uuid"]) not in reference_passed
        and str(item["child_uuid"]) not in missing_reference
    ]
    conditional_status = collections.Counter(
        str(reference[str(item["child_uuid"])].get("status")) for item in conditional_child_failures
    )
    reference_failure_signatures = collections.Counter(
        _failure_signature(record) for uuid, record in reference.items() if uuid not in reference_passed
    )

    reference_eligible_decisions = [
        item
        for item in decisions
        if str(item["parent_uuid"]) in reference_passed and str(item["child_uuid"]) in reference
    ]
    both_pass_decisions = [
        item for item in reference_eligible_decisions if str(item["child_uuid"]) in reference_passed
    ]
    both_reference_passed = {str(item["child_uuid"]) for item in both_pass_decisions}
    both_pass_path = run_dir / "analysis/reference_both_pass_child_uuids.txt"
    _atomic_text(
        both_pass_path,
        "".join(f"{item['child_uuid']}\n" for item in both_pass_decisions),
    )
    region = _runtime_records(run_dir / "h20/region", uuid_field="child_uuid")
    unexpected_region = set(region) - both_reference_passed
    if unexpected_region:
        raise ValueError(f"region contains unexpected UUIDs: {sorted(unexpected_region)[:10]}")
    region_passed = {uuid for uuid, record in region.items() if record.get("passed") is True}
    region_status = collections.Counter(str(record.get("status")) for record in region.values())
    region_failure_signatures = collections.Counter(
        _failure_signature(record) for uuid, record in region.items() if uuid not in region_passed
    )

    def variant(decision: Mapping[str, Any]) -> str:
        return str(decision["variant"]).split(":")[-1]

    def arm(decision: Mapping[str, Any]) -> str:
        return str(decision["variant"]).split(":")[0]

    child_rows = {str(_nested(row, "extra_info.uuid")): row for row in _rows(output.children)}

    def source_family(decision: Mapping[str, Any]) -> str:
        return str(
            _nested(
                child_rows[str(decision["child_uuid"])],
                "extra_info.v4.source_family",
                "unknown",
            )
        )

    def operator_bucket(decision: Mapping[str, Any]) -> str:
        return str(
            _nested(
                child_rows[str(decision["child_uuid"])],
                "extra_info.v4.operator_bucket",
                "unknown",
            )
        )

    conditional_status_by_variant: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    conditional_status_by_source: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    for item in conditional_child_failures:
        status = str(reference[str(item["child_uuid"])].get("status"))
        conditional_status_by_variant[variant(item)][status] += 1
        conditional_status_by_source[source_family(item)][status] += 1

    report = {
        "contract_version": "model_shape_runtime_diagnostics_v1",
        "run_dir": str(run_dir.resolve()),
        "artifacts": {
            "selected_sha256": _sha256_file(output.selected),
            "children_sha256": _sha256_file(output.children),
            "manifest_sha256": _sha256_file(output.manifest),
            "reference_shard_count": len(list((run_dir / "h20/reference").glob("shard-*.jsonl"))),
            "reference_shards_sha256": _shard_family_sha256(list((run_dir / "h20/reference").glob("shard-*.jsonl"))),
        },
        "static": {
            "selected_parents": pq.ParquetFile(output.selected).metadata.num_rows,
            "parents_with_child": len(parent_uuids),
            "children": len(decisions),
        },
        "reference": {
            "expected_records": len(expected_reference),
            "records": len(reference),
            "complete": not missing_reference,
            "missing": len(missing_reference),
            "missing_parents": len(missing_reference & parent_uuids),
            "missing_children": len(missing_reference & child_uuids),
            "partial_pairs": len(partial_pairs),
            "missing_uuids_path": str(missing_path.resolve()) if missing_reference else None,
            "missing_uuids_sha256": _sha256_file(missing_path) if missing_reference else None,
            "passed": len(reference_passed),
            "parents_passed": len(parent_passed),
            "parents_total": len(parent_uuids),
            "parents_observed": len(parent_uuids - missing_reference),
            "children_passed": len(child_passed),
            "children_total": len(child_uuids),
            "children_observed": len(child_uuids - missing_reference),
            "parent_observed_pass_rate": (
                len(parent_passed) / len(parent_uuids - missing_reference)
                if parent_uuids - missing_reference
                else None
            ),
            "child_observed_pass_rate": (
                len(child_passed) / len(child_uuids - missing_reference) if child_uuids - missing_reference else None
            ),
            "both_pass_rate_over_complete_pairs": (
                len(both_reference_passed) / (len(decisions) - len(partial_pairs))
                if len(decisions) > len(partial_pairs)
                else None
            ),
            "both_pass_child_uuids_path": str(both_pass_path.resolve()),
            "both_pass_child_uuids_sha256": _sha256_file(both_pass_path),
            "pair_outcomes": {f"{left}|{right}": count for (left, right), count in sorted(pair_outcomes.items())},
            "child_failures_given_parent_pass": dict(sorted(conditional_status.items())),
            "top_failure_signatures": [
                {"signature": signature, "count": count}
                for signature, count in reference_failure_signatures.most_common(20)
            ],
            "child_pass_rate_by_arm": _rate_table(decisions, child_passed, arm),
            "child_pass_rate_by_variant": _rate_table(decisions, child_passed, variant),
            "child_pass_rate_by_slots": _rate_table(
                decisions, child_passed, lambda item: str(item["logical_slot_count"])
            ),
            "child_pass_rate_by_input_bytes": _rate_table(
                decisions,
                child_passed,
                lambda item: _bytes_bucket(int(item["input_bytes_after"])),
            ),
            "child_pass_rate_by_input_scale": _rate_table(
                decisions,
                child_passed,
                lambda item: _scale_bucket(float(item["input_scale"])),
            ),
            "conditional_on_parent_pass": {
                "children_observed": len(reference_eligible_decisions),
                "children_passed": len(both_pass_decisions),
                "pass_rate": (
                    len(both_pass_decisions) / len(reference_eligible_decisions)
                    if reference_eligible_decisions
                    else None
                ),
                "pass_rate_by_arm": _rate_table(reference_eligible_decisions, child_passed, arm),
                "pass_rate_by_variant": _rate_table(reference_eligible_decisions, child_passed, variant),
                "pass_rate_by_slots": _rate_table(
                    reference_eligible_decisions,
                    child_passed,
                    lambda item: str(item["logical_slot_count"]),
                ),
                "pass_rate_by_input_bytes": _rate_table(
                    reference_eligible_decisions,
                    child_passed,
                    lambda item: _bytes_bucket(int(item["input_bytes_after"])),
                ),
                "pass_rate_by_input_scale": _rate_table(
                    reference_eligible_decisions,
                    child_passed,
                    lambda item: _scale_bucket(float(item["input_scale"])),
                ),
                "pass_rate_by_source": _rate_table(reference_eligible_decisions, child_passed, source_family),
                "pass_rate_by_operator_bucket": _rate_table(
                    reference_eligible_decisions, child_passed, operator_bucket
                ),
                "failure_status_by_variant": {
                    label: dict(sorted(counts.items()))
                    for label, counts in sorted(conditional_status_by_variant.items())
                },
                "failure_status_by_source": {
                    label: dict(sorted(counts.items()))
                    for label, counts in sorted(conditional_status_by_source.items())
                },
            },
        },
        "region": {
            "expected": len(both_reference_passed),
            "records": len(region),
            "complete": set(region) == both_reference_passed,
            "status_counts": dict(sorted(region_status.items())),
            "passed": len(region_passed),
            "top_failure_signatures": [
                {"signature": signature, "count": count}
                for signature, count in region_failure_signatures.most_common(20)
            ],
            "pass_rate_by_arm": _rate_table(decisions, region_passed, arm),
            "pass_rate_by_variant": _rate_table(decisions, region_passed, variant),
            "pass_rate_by_slots": _rate_table(decisions, region_passed, lambda item: str(item["logical_slot_count"])),
            "pass_rate_by_input_bytes": _rate_table(
                decisions,
                region_passed,
                lambda item: _bytes_bucket(int(item["input_bytes_after"])),
            ),
        },
        "value_bias": {
            "static": _value_bias(decisions),
            "reference_both_pass": _value_bias(both_pass_decisions),
            "region_pass": _value_bias([item for item in decisions if str(item["child_uuid"]) in region_passed]),
        },
    }
    diagnostics_path = run_dir / "analysis/runtime_diagnostics.json"
    _atomic_text(diagnostics_path, json.dumps(report, indent=2, sort_keys=True) + "\n")

    parent_rows = {str(_nested(row, "extra_info.uuid")): row for row in _rows(output.selected)}
    lines = [
        "# Model shape runtime review samples",
        "",
        f"Reference completeness: {len(reference)}/{len(expected_reference)}; "
        f"missing={len(missing_reference)}; affected pairs={len(partial_pairs)}.",
        f"Reference pair outcomes: `{dict(sorted(pair_outcomes.items()))}`.",
        f"Region progress: {len(region)}/{len(both_reference_passed)}; passed={len(region_passed)}.",
        "",
        "## Parent-pass / child-fail examples",
        "",
    ]

    def add_diff(item: Mapping[str, Any]) -> None:
        parent_uuid = str(item["parent_uuid"])
        child_uuid = str(item["child_uuid"])
        parent_code = str(_nested(parent_rows[parent_uuid], "reward_model.ground_truth"))
        child_code = str(_nested(child_rows[child_uuid], "reward_model.ground_truth"))
        diff = "".join(
            difflib.unified_diff(
                parent_code.splitlines(keepends=True),
                child_code.splitlines(keepends=True),
                fromfile=f"parent/{parent_uuid}",
                tofile=f"child/{child_uuid}",
                n=2,
            )
        ).rstrip()
        lines.extend(["", f"### `{child_uuid}`", "", "```diff", diff, "```"])

    for item in conditional_child_failures[:8]:
        uuid = str(item["child_uuid"])
        slots = [f"{slot['old_value']}->{slot['new_value']}" for slot in _nested(item, "solver.slots", [])]
        lines.append(
            f"- `{uuid}` parent=`{item['parent_uuid']}` variant=`{item['variant']}` "
            f"slots={slots} input={int(item['input_bytes_after']) / 2**30:.3f} GiB; "
            f"{_failure_signature(reference[uuid])}"
        )
    for item in conditional_child_failures[:4]:
        add_diff(item)
    lines.extend(["", "## Region-fail examples", ""])
    for uuid in sorted(set(region) - region_passed)[:8]:
        item = decision_by_child[uuid]
        slots = [f"{slot['old_value']}->{slot['new_value']}" for slot in _nested(item, "solver.slots", [])]
        lines.append(
            f"- `{uuid}` parent=`{item['parent_uuid']}` variant=`{item['variant']}` "
            f"slots={slots}; {_failure_signature(region[uuid])}"
        )
    for uuid in sorted(set(region) - region_passed)[:4]:
        add_diff(decision_by_child[uuid])
    lines.extend(["", "## Region-pass examples", ""])
    for uuid in sorted(region_passed)[:8]:
        item = decision_by_child[uuid]
        slots = [f"{slot['old_value']}->{slot['new_value']}" for slot in _nested(item, "solver.slots", [])]
        code = str(_nested(child_rows[uuid], "reward_model.ground_truth"))
        lines.append(
            f"- `{uuid}` parent=`{item['parent_uuid']}` variant=`{item['variant']}` "
            f"slots={slots}; reference_sha256=`"
            f"{_nested(child_rows[uuid], 'extra_info.v4.reference_sha256')}`; "
            f"code_chars={len(code)}"
        )
    for uuid in sorted(region_passed)[:4]:
        add_diff(decision_by_child[uuid])
    _atomic_text(
        run_dir / "analysis/runtime_review_samples.md",
        "\n".join(lines).rstrip() + "\n",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--allow-partial-reference", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(args.run_dir, allow_partial_reference=args.allow_partial_reference),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
