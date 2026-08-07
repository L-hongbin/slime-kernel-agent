#!/usr/bin/env python3
"""Prepare identical v2 DSPARK on/off selections from the completed v1 canary."""

from __future__ import annotations

import collections
import copy
import json
import math
from collections.abc import Mapping
from typing import Any

import pyarrow.parquet as pq
from tools.data.synthesize.model_shape.pipeline import (
    REPO_ROOT,
    _atomic_parquet,
    _atomic_text,
    _identity,
    _load_generation,
    _paths,
    _rows,
    _sha256_file,
)
from tools.data.synthesize.model_shape.prompt import PROMPT_VERSION

SOURCE_RUN = REPO_ROOT / "Data/prompt_tvm_v4/shape_model_hardtail_v1/run.canary1000"
AB_ROOT = REPO_ROOT / "Data/prompt_tvm_v4/shape_model_hardtail_v2/ab260"
ARM_RUNS = {
    "dspark": AB_ROOT / "run.dspark",
    "no_dspark": AB_ROOT / "run.no_dspark",
}
EXPECTED_LENGTH_FAILURES = 130


def _nested(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _finish(record: Mapping[str, Any]) -> tuple[str | None, int | None, str | None]:
    choice = _nested(record, "response.choices", [])
    choice = choice[0] if isinstance(choice, list) and choice else {}
    finish_reason = choice.get("finish_reason") if isinstance(choice, Mapping) else None
    content = _nested(choice, "message.content") if isinstance(choice, Mapping) else None
    completion_tokens = _nested(record, "response.usage.completion_tokens")
    return finish_reason, completion_tokens, content


def _prompt_tokens(record: Mapping[str, Any]) -> int:
    value = _nested(record, "response.usage.prompt_tokens", 0)
    return int(value) if type(value) is int else 0


def _target_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    left_targets = left["target_input_bytes"]
    right_targets = right["target_input_bytes"]
    return sum(
        abs(math.log2(int(left_targets[variant])) - math.log2(int(right_targets[variant])))
        for variant in ("medium", "large")
    )


def prepare() -> dict[str, Any]:
    source = _paths(SOURCE_RUN)
    source_selection = json.loads(source.selection.read_text(encoding="utf-8"))
    source_meta = {str(item["parent_uuid"]): item for item in source_selection["rows"]}
    source_rows = {_identity(row)[0]: row for row in _rows(source.selected)}
    generation = _load_generation(source.generation)
    if set(source_rows) != set(generation) or set(source_rows) != set(source_meta):
        raise ValueError("v1 canary selected/selection/generation UUID sets differ")

    failures: list[str] = []
    controls: list[str] = []
    for uuid, record in generation.items():
        finish_reason, completion_tokens, content = _finish(record)
        if (
            finish_reason == "length"
            and completion_tokens == 32768
            and (not isinstance(content, str) or not content.strip())
        ):
            failures.append(uuid)
        elif finish_reason == "stop" and isinstance(content, str) and content.strip():
            controls.append(uuid)
    if len(failures) != EXPECTED_LENGTH_FAILURES:
        raise ValueError(f"expected {EXPECTED_LENGTH_FAILURES} v1 length failures, found {len(failures)}")

    controls_by_group = collections.Counter(
        (source_meta[uuid]["source_family"], source_meta[uuid]["operator_family"]) for uuid in controls
    )
    failures.sort(
        key=lambda uuid: (
            controls_by_group[(source_meta[uuid]["source_family"], source_meta[uuid]["operator_family"])],
            uuid,
        )
    )
    unused = set(controls)
    matches: list[tuple[str, str]] = []
    for failure_uuid in failures:
        failure_meta = source_meta[failure_uuid]
        failure_record = generation[failure_uuid]

        def score(
            control_uuid: str,
            failure_meta: Mapping[str, Any] = failure_meta,
            failure_record: Mapping[str, Any] = failure_record,
        ) -> tuple[float, float, int, str]:
            control_meta = source_meta[control_uuid]
            if (
                control_meta["source_family"] == failure_meta["source_family"]
                and control_meta["operator_family"] == failure_meta["operator_family"]
            ):
                group_penalty = 0.0
            elif control_meta["source_family"] == failure_meta["source_family"]:
                group_penalty = 4.0
            else:
                group_penalty = 8.0
            return (
                group_penalty,
                _target_distance(failure_meta, control_meta),
                abs(_prompt_tokens(failure_record) - _prompt_tokens(generation[control_uuid])),
                control_uuid,
            )

        control_uuid = min(unused, key=score)
        unused.remove(control_uuid)
        matches.append((failure_uuid, control_uuid))

    ordered_uuids = [uuid for pair in matches for uuid in pair]
    if len(ordered_uuids) != 260 or len(set(ordered_uuids)) != 260:
        raise AssertionError("A/B selection must contain 260 unique parents")
    role_by_uuid = {
        uuid: role
        for failure_uuid, control_uuid in matches
        for uuid, role in ((failure_uuid, "prior_length"), (control_uuid, "matched_stop"))
    }
    match_by_uuid = {
        uuid: peer
        for failure_uuid, control_uuid in matches
        for uuid, peer in ((failure_uuid, control_uuid), (control_uuid, failure_uuid))
    }

    summaries: dict[str, Any] = {}
    schema = pq.ParquetFile(source.selected).schema_arrow
    for arm, run_dir in ARM_RUNS.items():
        output = _paths(run_dir)
        if output.generation.exists() or output.manifest.exists():
            raise FileExistsError(f"refusing to replace existing A/B arm: {run_dir}")
        rows = [source_rows[uuid] for uuid in ordered_uuids]
        _atomic_parquet(output.selected, rows, schema)
        selected_rows = []
        for selected_index, uuid in enumerate(ordered_uuids):
            item = copy.deepcopy(source_meta[uuid])
            item["selected_index"] = selected_index
            item["ab_role"] = role_by_uuid[uuid]
            item["matched_parent_uuid"] = match_by_uuid[uuid]
            selected_rows.append(item)
        selection = {
            "contract_version": "dsv4_shape_hardtail_ab_selection_v1",
            "prompt_version": PROMPT_VERSION,
            "engine_condition": arm,
            "source_run": str(SOURCE_RUN.resolve()),
            "source_selected_sha256": _sha256_file(source.selected),
            "source_generation_sha256": _sha256_file(source.generation),
            "selected_parent_count": len(rows),
            "representative_for_projection": False,
            "prior_length_parent_count": EXPECTED_LENGTH_FAILURES,
            "matched_stop_parent_count": EXPECTED_LENGTH_FAILURES,
            "matching_contract": (
                "greedy without replacement; exact source/operator first, then source; "
                "minimize Medium/Large log2 target distance and prompt-token distance"
            ),
            "rows": selected_rows,
        }
        _atomic_text(output.selection, json.dumps(selection, indent=2, sort_keys=True) + "\n")
        summaries[arm] = {
            "run_dir": str(run_dir.resolve()),
            "selected_sha256": _sha256_file(output.selected),
            "selection_sha256": _sha256_file(output.selection),
        }

    summary = {
        "contract_version": "dsv4_shape_hardtail_ab_selection_v1",
        "prompt_version": PROMPT_VERSION,
        "source_run": str(SOURCE_RUN.resolve()),
        "pairs": len(matches),
        "parents_per_arm": len(ordered_uuids),
        "requests_total": len(ordered_uuids) * len(ARM_RUNS),
        "exact_group_matches": sum(
            source_meta[left]["source_family"] == source_meta[right]["source_family"]
            and source_meta[left]["operator_family"] == source_meta[right]["operator_family"]
            for left, right in matches
        ),
        "same_source_matches": sum(
            source_meta[left]["source_family"] == source_meta[right]["source_family"] for left, right in matches
        ),
        "arms": summaries,
        "matches": [{"prior_length_parent_uuid": left, "matched_stop_parent_uuid": right} for left, right in matches],
    }
    _atomic_text(AB_ROOT / "selection_summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(prepare(), indent=2, sort_keys=True))
