#!/usr/bin/env python3
"""Compare the paired v2 DSPARK A/B and build its canonical runtime input."""

from __future__ import annotations

import collections
import copy
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow.parquet as pq
from tools.data.synthesize.model_shape.pipeline import (
    REPO_ROOT,
    _atomic_parquet,
    _atomic_text,
    _identity,
    _load_generation,
    _nested,
    _paths,
    _rows,
    _sha256_file,
)

AB_ROOT = REPO_ROOT / "Data/prompt_tvm_v4/shape_model_hardtail_v2/ab260"
ARM_RUNS = {
    "dspark": AB_ROOT / "run.dspark",
    "no_dspark": AB_ROOT / "run.no_dspark",
}
RUNTIME_RUN = AB_ROOT / "runtime_canary"


def _finish(record: Mapping[str, Any]) -> str:
    if record.get("http_status") != 200 or record.get("error") is not None:
        return "transport_error"
    choices = _nested(record, "response.choices", [])
    choice = choices[0] if isinstance(choices, list) and choices else {}
    reason = choice.get("finish_reason") if isinstance(choice, Mapping) else None
    content = _nested(choice, "message.content") if isinstance(choice, Mapping) else None
    completion = _nested(record, "response.usage.completion_tokens")
    if reason == "length" and completion == 32768 and (not isinstance(content, str) or not content.strip()):
        return "length_empty"
    if reason == "stop" and isinstance(content, str) and content.strip():
        return "stop_content"
    return f"other:{reason}"


def _quantiles(values: Sequence[int | float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    result: dict[str, float] = {}
    for label, q in (("p50", 0.5), ("p90", 0.9), ("p99", 0.99)):
        position = q * (len(ordered) - 1)
        lower, upper = math.floor(position), math.ceil(position)
        result[label] = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
    return result


def _arm_metrics(
    arm: str,
    records: Mapping[str, Mapping[str, Any]],
    roles: Mapping[str, str],
) -> dict[str, Any]:
    finish = collections.Counter(_finish(record) for record in records.values())
    by_role: dict[str, Any] = {}
    for role in ("prior_length", "matched_stop"):
        subset = [record for uuid, record in records.items() if roles[uuid] == role]
        counts = collections.Counter(_finish(record) for record in subset)
        by_role[role] = {
            "records": len(subset),
            "finish": dict(sorted(counts.items())),
            "length_rate": counts["length_empty"] / len(subset),
        }
    elapsed = [float(record["elapsed_seconds"]) for record in records.values()]
    completion = [
        int(value)
        for record in records.values()
        if type(value := _nested(record, "response.usage.completion_tokens")) is int
    ]
    return {
        "arm": arm,
        "records": len(records),
        "finish": dict(sorted(finish.items())),
        "length_rate": finish["length_empty"] / len(records),
        "by_prior_role": by_role,
        "elapsed_seconds": _quantiles(elapsed),
        "completion_tokens": _quantiles(completion),
    }


def analyze() -> dict[str, Any]:
    selection_summary = json.loads((AB_ROOT / "selection_summary.json").read_text(encoding="utf-8"))
    records = {arm: _load_generation(_paths(run_dir).generation) for arm, run_dir in ARM_RUNS.items()}
    expected = int(selection_summary["parents_per_arm"])
    if any(len(value) != expected for value in records.values()):
        raise ValueError(f"A/B generation incomplete: { {arm: len(value) for arm, value in records.items()} }")
    if set(records["dspark"]) != set(records["no_dspark"]):
        raise ValueError("A/B parent UUID sets differ")

    dspark_selection = json.loads(_paths(ARM_RUNS["dspark"]).selection.read_text(encoding="utf-8"))
    roles = {str(item["parent_uuid"]): str(item["ab_role"]) for item in dspark_selection["rows"]}
    prompt_mismatches = [
        uuid
        for uuid in records["dspark"]
        if records["dspark"][uuid].get("prompt_sha256") != records["no_dspark"][uuid].get("prompt_sha256")
    ]
    if prompt_mismatches:
        raise ValueError(f"A/B prompt hashes differ: {prompt_mismatches[:10]}")

    paired_finish = collections.Counter(
        (_finish(records["dspark"][uuid]), _finish(records["no_dspark"][uuid])) for uuid in records["dspark"]
    )
    dspark_only_length = paired_finish[("length_empty", "stop_content")]
    no_dspark_only_length = paired_finish[("stop_content", "length_empty")]
    discordant = dspark_only_length + no_dspark_only_length
    exact_mcnemar_p = min(
        1.0,
        2
        * sum(math.comb(discordant, value) for value in range(min(dspark_only_length, no_dspark_only_length) + 1))
        / (2**discordant),
    )
    role_pairing: dict[str, dict[str, int]] = {}
    for role in ("prior_length", "matched_stop"):
        role_pairing[role] = {
            f"{left}|{right}": count
            for (left, right), count in sorted(
                collections.Counter(
                    (
                        _finish(records["dspark"][uuid]),
                        _finish(records["no_dspark"][uuid]),
                    )
                    for uuid in records["dspark"]
                    if roles[uuid] == role
                ).items()
            )
        }

    materialization: dict[str, Any] = {}
    bias: dict[str, Any] = {}
    for arm, run_dir in ARM_RUNS.items():
        output = _paths(run_dir)
        manifest = json.loads(output.manifest.read_text(encoding="utf-8"))
        bias_report = json.loads(output.bias_json.read_text(encoding="utf-8"))
        canary = bias_report["canary"]
        materialization[arm] = {
            "children": manifest["children_written"],
            "parents_with_child": manifest["parents_with_child"],
            "variant_decisions": manifest["variant_decision_count"],
            "rejection_reasons": manifest["rejection_reasons"],
        }
        bias[arm] = {
            "power_of_two_share": canary["power_of_two_occurrence_share"],
            "near_decimal_100_grid_share": canary["near_decimal_100_grid_share"],
            "near_power_of_two_share": canary["near_power_of_two_within_3_share"],
            "top10_share": canary["top10_changed_value_share"],
            "logical_slot_count_histogram": canary["logical_slot_count_histogram"],
            "leading_axis_share": canary["leading_axis_occurrence_share"],
            "trailing_axis_share": canary["trailing_axis_occurrence_share"],
            "warnings": bias_report["bias_warnings"],
        }

    report = {
        "contract_version": "dsv4_shape_hardtail_ab_analysis_v1",
        "selected_sha256": _sha256_file(_paths(ARM_RUNS["dspark"]).selected),
        "prompt_hash_mismatches": 0,
        "parents_per_arm": expected,
        "arms": {arm: _arm_metrics(arm, arm_records, roles) for arm, arm_records in records.items()},
        "paired_finish": {f"{left}|{right}": count for (left, right), count in sorted(paired_finish.items())},
        "paired_length_mcnemar": {
            "dspark_only_length": dspark_only_length,
            "no_dspark_only_length": no_dspark_only_length,
            "both_length": paired_finish[("length_empty", "length_empty")],
            "exact_two_sided_p": exact_mcnemar_p,
            "interpretation": "no statistically established DSPARK length-limit effect at alpha=0.05",
        },
        "paired_finish_by_prior_role": role_pairing,
        "materialization": materialization,
        "bias": bias,
        "artifact_sha256": {
            arm: {
                "generation": _sha256_file(_paths(run_dir).generation),
                "manifest": _sha256_file(_paths(run_dir).manifest),
                "children": _sha256_file(_paths(run_dir).children),
                "bias": _sha256_file(_paths(run_dir).bias_json),
            }
            for arm, run_dir in ARM_RUNS.items()
        },
    }
    _atomic_text(AB_ROOT / "analysis/ab.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = [
        "# v2 prompt × DSPARK paired A/B",
        "",
        f"Both arms contain the same {expected} parents and identical prompt hashes.",
        "",
        "| Arm | Length-limit | Prior-length subset | Matched-stop subset | Static children | Covered parents |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for arm in ("dspark", "no_dspark"):
        item = report["arms"][arm]
        materialized = materialization[arm]
        lines.append(
            f"| {arm} | {item['finish'].get('length_empty', 0)}/{expected} "
            f"| {item['by_prior_role']['prior_length']['finish'].get('length_empty', 0)}/130 "
            f"| {item['by_prior_role']['matched_stop']['finish'].get('length_empty', 0)}/130 "
            f"| {materialized['children']} | {materialized['parents_with_child']} |"
        )
    lines.extend(["", "## Paired finish outcomes", ""])
    lines.extend(f"- `{outcome}`: {count}" for outcome, count in report["paired_finish"].items())
    lines.append(f"- Exact paired McNemar p={exact_mcnemar_p:.4f}; no established DSPARK effect at alpha=0.05.")
    lines.extend(["", "## Bias", ""])
    for arm in ("dspark", "no_dspark"):
        item = bias[arm]
        lines.append(
            f"- {arm}: P2 {item['power_of_two_share']:.2%}; near decimal-100 "
            f"{item['near_decimal_100_grid_share']:.2%}; near P2 "
            f"{item['near_power_of_two_share']:.2%}; top-10 {item['top10_share']:.2%}."
        )
    _atomic_text(AB_ROOT / "analysis/ab.md", "\n".join(lines).rstrip() + "\n")
    return report


def combine_runtime() -> dict[str, Any]:
    selected_paths = [_paths(run_dir).selected for run_dir in ARM_RUNS.values()]
    selected_hashes = {_sha256_file(path) for path in selected_paths}
    if len(selected_hashes) != 1:
        raise ValueError("A/B selected parquet hashes differ")
    parents = _rows(selected_paths[0])

    child_rows: dict[str, dict[str, Any]] = {}
    decisions: dict[str, dict[str, Any]] = {}
    child_order: dict[str, list[str]] = collections.defaultdict(list)
    for arm, run_dir in ARM_RUNS.items():
        output = _paths(run_dir)
        children = {str(_nested(row, "extra_info.uuid")): row for row in _rows(output.children)}
        manifest = json.loads(output.manifest.read_text(encoding="utf-8"))
        for raw in manifest["decisions"]:
            if raw.get("accepted") is not True:
                continue
            decision = copy.deepcopy(raw)
            child_uuid = str(decision["child_uuid"])
            parent_uuid = str(decision["parent_uuid"])
            if child_uuid not in children:
                raise ValueError(f"arm {arm} manifest child missing: {child_uuid}")
            if child_uuid in child_rows:
                old_code = str(_nested(child_rows[child_uuid], "reward_model.ground_truth"))
                new_code = str(_nested(children[child_uuid], "reward_model.ground_truth"))
                if old_code != new_code or decisions[child_uuid]["parent_uuid"] != parent_uuid:
                    raise ValueError(f"conflicting duplicate A/B child: {child_uuid}")
                conditions = decisions[child_uuid].setdefault("ab_engine_conditions", [])
                conditions.append(arm)
                decisions[child_uuid]["variant"] = f"both:{raw['variant']}"
                continue
            decision["variant"] = f"{arm}:{decision['variant']}"
            decision["ab_engine_conditions"] = [arm]
            child_rows[child_uuid] = children[child_uuid]
            decisions[child_uuid] = decision
            child_order[parent_uuid].append(child_uuid)

    children_ordered: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    for parent in parents:
        parent_uuid = _identity(parent)[0]
        uuids = child_order.get(parent_uuid, [])
        if not uuids:
            continue
        paired.append(parent)
        for child_uuid in uuids:
            child = child_rows[child_uuid]
            children_ordered.append(child)
            paired.append(child)
    if len(children_ordered) != len(child_rows):
        raise AssertionError("combined A/B child order lost rows")

    runtime_paths = _paths(RUNTIME_RUN)
    schema = pq.ParquetFile(selected_paths[0]).schema_arrow
    _atomic_parquet(runtime_paths.selected, parents, schema)
    _atomic_parquet(runtime_paths.children, children_ordered, schema)
    _atomic_parquet(runtime_paths.paired, paired, schema)
    selection = {
        "contract_version": "dsv4_shape_hardtail_ab_runtime_selection_v1",
        "source_runs": {arm: str(path.resolve()) for arm, path in ARM_RUNS.items()},
        "selected_parent_count": len(parents),
        "rows": json.loads(_paths(ARM_RUNS["dspark"]).selection.read_text(encoding="utf-8"))["rows"],
    }
    _atomic_text(runtime_paths.selection, json.dumps(selection, indent=2, sort_keys=True) + "\n")
    manifest = {
        "contract_version": "dsv4_shape_hardtail_ab_runtime_v1",
        "group_contract": {
            "logical_slot_count_range": [1, 8],
            "fixed_cardinality": False,
            "power_of_two_quota": None,
            "maximum_dimension_imbalance_ratio": 1000,
        },
        "selected_sha256": _sha256_file(runtime_paths.selected),
        "selection_sha256": _sha256_file(runtime_paths.selection),
        "children_sha256": _sha256_file(runtime_paths.children),
        "paired_sha256": _sha256_file(runtime_paths.paired),
        "parent_count": len(parents),
        "children_written": len(children_ordered),
        "parents_with_child": len(child_order),
        "paired_layout": "parent_once_followed_by_all_unique_ab_children",
        "paired_rows": len(paired),
        "decisions": [decisions[str(_nested(row, "extra_info.uuid"))] for row in children_ordered],
        "training_approved": False,
    }
    _atomic_text(runtime_paths.manifest, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "run_dir": str(RUNTIME_RUN.resolve()),
        "parents": len(parents),
        "children": len(children_ordered),
        "parents_with_child": len(child_order),
        "paired_rows": len(paired),
        "children_sha256": manifest["children_sha256"],
        "paired_sha256": manifest["paired_sha256"],
        "manifest_sha256": _sha256_file(runtime_paths.manifest),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("analyze", "combine-runtime"))
    args = parser.parse_args()
    result = analyze() if args.command == "analyze" else combine_runtime()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
