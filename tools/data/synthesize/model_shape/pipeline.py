#!/usr/bin/env python3
"""Select, generate, canonicalize, gate, and audit DSV4 shape hard tails.

The CLI intentionally exposes only a run directory plus the few controls that
change experiment scale. Stable source, endpoint, model, prompt, and artifact
layout contracts live in code and are recorded in every output manifest.
"""

from __future__ import annotations

import argparse
import ast
import collections
import concurrent.futures
import copy
import difflib
import hashlib
import json
import math
import multiprocessing
import os
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.data.cleaning.pipeline import inspect_row_schema
from tools.data.synthesize.augment_prompt_tasks import _replace_reference
from tools.data.synthesize.model_shape.prompt import (
    PROMPT_VERSION,
    USER_TEMPLATES,
    VARIANTS,
    render_user_prompt,
    target_input_bytes,
    target_input_bytes_from_size,
)
from tools.data.synthesize.shape_contract import (
    _function,
    _numeric_constants,
    _operator_family,
    _shape_numeric_node_ids,
    _validate_target_proximity,
    _validate_variant_storage,
    relaxed_factory_storage,
    relaxed_static_gate,
    relaxed_structure_gate,
    static_gate,
)
from tools.data.synthesize.solve_shape_coverage import (
    MAX_DIMENSION_IMBALANCE_RATIO,
    ShapeSlot,
    SourceSpan,
    _affected_shape_balance_guard,
    _fake_input_profile,
    _fake_tensor_gate,
    _shape_slots_with_rejections,
)

CONTRACT_VERSION = "dsv4_shape_hardtail_v1"
GENERATION_CONTRACT_VERSION_V1 = "dsv4_shape_hardtail_generation_v1"
GENERATION_CONTRACT_VERSION = "dsv4_shape_hardtail_generation_v2"
GENERATION_CONTRACT_VERSIONS = frozenset({GENERATION_CONTRACT_VERSION_V1, GENERATION_CONTRACT_VERSION})
SELECTION_REASON = "no_variable_multislot_product_solution"
SELECTION_SALT = "dsv4_shape_hardtail_canary_v1"
FULL_RESIDUAL_SELECTION_CONTRACT_VERSION = "dsv4_shape_full_measurable_residual_selection_v1"
FULL_RESIDUAL_SELECTION_SALT = "dsv4_shape_full_measurable_residual_v1"
RELAXED_RESIDUAL_SELECTION_CONTRACT_VERSION = "dsv4_shape_relaxed_provenance_residual_selection_v2"
RELAXED_STORAGE_CONTRACT = "fake_returned_tensor_storage_v1"
MODEL_NAME = "deepseek-v4-flash-0731"
DEFAULT_SOURCE_RUN = (
    REPO_ROOT / "Data/prompt_tvm_v4/shape_solver_variable_multislot_v8/" "run.remaining22566.balanced_v7"
)
DEFAULT_MEASURABLE_SOURCE_RUN = REPO_ROOT / "Data/prompt_tvm_v4/shape_solver_multidim_v4_byte_targets_v1/run.full53896"
DEFAULT_RUNTIME_SHAPE_RUNS = (
    DEFAULT_MEASURABLE_SOURCE_RUN,
    REPO_ROOT / "Data/prompt_tvm_v4/shape_solver_variable_multislot_v5/run.recoverable12318.balanced_v7",
    DEFAULT_SOURCE_RUN,
)
DEFAULT_PRIOR_MODEL_RUN = REPO_ROOT / "Data/prompt_tvm_v4/shape_model_hardtail_v2/run.full17864"
DEFAULT_CANONICAL_PARENTS = REPO_ROOT / "Data/prompt_tvm_v4/train.review.parquet"
DEFAULT_RUN_DIR = REPO_ROOT / "Data/prompt_tvm_v4/shape_model_hardtail_v1/run.canary1000"
DEFAULT_ENDPOINTS = (
    "http://10.11.2.153:31053",
    "http://10.11.2.164:31064",
    "http://10.11.2.169:31069",
    "http://10.11.2.170:31070",
)
DEFAULT_COUNT = 1000
DEFAULT_CONCURRENCY_PER_ENDPOINT = 64
DEFAULT_MAX_TOKENS = 32 * 1024
DEFAULT_TIMEOUT_SECONDS = 2700.0
DEFAULT_FAKE_TIMEOUT_SECONDS = 30.0
DEFAULT_WORKERS = min(32, max(1, (os.cpu_count() or 4) // 4))
VARIANT_ORDER = {variant: index for index, variant in enumerate(VARIANTS)}
SECTION_RE = re.compile(
    r"^\s*##\s*Medium\s*\n```(?:python)?\s*\n(?P<medium>.*?)\n```"
    r"\s*##\s*Large\s*\n```(?:python)?\s*\n(?P<large>.*?)\n```\s*$",
    re.IGNORECASE | re.DOTALL,
)
BASELINE_FULL_ROWS = 64_315
BASELINE_CHANGED_PARENTS = 23_358
BASELINE_UNCHANGED_PARENTS = 40_957
HARDTAIL_ELIGIBLE_PARENTS = 17_864
INVERSE_POWER_OF_TWO_WARNING_SHARE = 0.05
REFERENCE_POWER_OF_TWO_SHARES = {
    "current_changed_occurrences": 0.4445,
    "prior_dsv4_random_target_proposals": 0.6213,
    "official_kernelbench_all_dimensions": 0.7951,
}


@dataclass(frozen=True)
class RunPaths:
    selected: Path
    selection: Path
    prompts: Path
    candidate_prompts: Path
    generation: Path
    generation_review: Path
    children: Path
    paired: Path
    manifest: Path
    materialized_review: Path
    failure_review: Path
    bias_json: Path
    bias_markdown: Path


def _paths(run_dir: Path) -> RunPaths:
    return RunPaths(
        selected=run_dir / "selected.parquet",
        selection=run_dir / "selection.json",
        prompts=run_dir / "review/prompts.md",
        candidate_prompts=run_dir / "review/prompt_candidate.md",
        generation=run_dir / "generation/responses.jsonl",
        generation_review=run_dir / "review/generations.md",
        children=run_dir / "static/children.parquet",
        paired=run_dir / "static/paired.parquet",
        manifest=run_dir / "static/manifest.json",
        materialized_review=run_dir / "review/materialized.md",
        failure_review=run_dir / "review/failures.md",
        bias_json=run_dir / "analysis/bias.json",
        bias_markdown=run_dir / "analysis/bias.md",
    )


def _nested(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(text)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_parquet(path: Path, rows: Sequence[Mapping[str, Any]], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".parquet", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        pq.write_table(pa.Table.from_pylist(list(rows), schema=schema), temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _iter_rows(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    index = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
        for row in batch.to_pylist():
            yield index, row
            index += 1


def _rows(path: Path) -> list[dict[str, Any]]:
    return [row for _, row in _iter_rows(path)]


def _identity(row: Mapping[str, Any]) -> tuple[str, str, str]:
    uuid = _nested(row, "extra_info.uuid")
    code = _nested(row, "reward_model.ground_truth")
    entry_point = str(_nested(row, "extra_info.entry_point", "Model"))
    if not isinstance(uuid, str) or not isinstance(code, str):
        raise ValueError("row_requires_uuid_and_reference")
    return uuid, code, entry_point


def _group(row: Mapping[str, Any]) -> tuple[str, str]:
    source = str(_nested(row, "extra_info.v4.source_family", "unknown"))
    operator = _operator_family(_nested(row, "extra_info.ops", ""))
    return source, operator


def select(source_run: Path, run_dir: Path, count: int) -> dict[str, Any]:
    if count <= 0:
        raise ValueError("count_must_be_positive")
    source_selected = source_run / "selected.parquet"
    source_manifest = source_run / "static/manifest.json"
    if not source_selected.is_file() or not source_manifest.is_file():
        raise FileNotFoundError(f"incomplete source run: {source_run}")
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if manifest.get("contract_version") != "shape_variable_multislot_solver_v7":
        raise ValueError("hardtail_source_must_be_balanced_variable_solver_v7")
    decisions = manifest.get("decisions")
    if not isinstance(decisions, list):
        raise ValueError("source_manifest_decisions_missing")
    eligible_decisions = {
        str(decision.get("parent_uuid")): decision
        for decision in decisions
        if isinstance(decision, Mapping)
        and decision.get("accepted") is False
        and decision.get("reason") == SELECTION_REASON
    }
    candidates: list[tuple[int, dict[str, Any]]] = []
    for source_index, row in _iter_rows(source_selected):
        uuid, code, _ = _identity(row)
        decision = eligible_decisions.get(uuid)
        if decision is None:
            continue
        if decision.get("parent_reference_sha256") != _sha256_bytes(code.encode()):
            raise ValueError(f"upstream_parent_hash_mismatch:{uuid}")
        parent_fake = decision.get("parent_fake_gate")
        if not isinstance(parent_fake, Mapping) or parent_fake.get("status") != "passed":
            raise ValueError(f"upstream_parent_fake_gate_not_passed:{uuid}")
        candidates.append((source_index, row))
    if len(candidates) != len(eligible_decisions):
        raise ValueError(f"eligible_source_join_mismatch:{len(candidates)}:{len(eligible_decisions)}")
    if count > len(candidates):
        raise ValueError(f"requested_{count}_from_{len(candidates)}_eligible_parents")

    grouped: dict[tuple[str, str], list[tuple[int, dict[str, Any]]]] = collections.defaultdict(list)
    for item in candidates:
        grouped[_group(item[1])].append(item)
    for items in grouped.values():
        items.sort(key=lambda item: _sha256_bytes(f"{SELECTION_SALT}:{_identity(item[1])[0]}".encode()))
    chosen: list[tuple[int, dict[str, Any]]] = []
    chosen_by_group: collections.Counter[tuple[str, str]] = collections.Counter()
    # Seed every observed source/operator cross when the canary is large enough.
    for group in sorted(grouped):
        if len(chosen) >= count:
            break
        chosen.append(grouped[group].pop(0))
        chosen_by_group[group] += 1
    original_sizes = {group: len(items) + chosen_by_group[group] for group, items in grouped.items()}
    while len(chosen) < count:
        available = [group for group, items in grouped.items() if items]
        group = min(
            available,
            key=lambda value: (
                (chosen_by_group[value] + 1) / original_sizes[value],
                value,
            ),
        )
        chosen.append(grouped[group].pop(0))
        chosen_by_group[group] += 1

    output = _paths(run_dir)
    output.selected.parent.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(
        output.selected,
        [row for _, row in chosen],
        pq.ParquetFile(source_selected).schema_arrow,
    )
    rows_manifest = []
    for selected_index, (source_index, row) in enumerate(chosen):
        uuid, code, _ = _identity(row)
        source, operator = _group(row)
        rows_manifest.append(
            {
                "selected_index": selected_index,
                "source_row_index": source_index,
                "parent_uuid": uuid,
                "parent_reference_sha256": _sha256_bytes(code.encode()),
                "source_family": source,
                "operator_family": operator,
                "target_input_bytes": target_input_bytes(row),
            }
        )
    eligible_by_group = collections.Counter(_group(row) for _, row in candidates)
    selection = {
        "contract_version": CONTRACT_VERSION,
        "selection_reason": SELECTION_REASON,
        "selection_salt": SELECTION_SALT,
        "source_run": str(source_run.resolve()),
        "source_selected_sha256": _sha256_file(source_selected),
        "source_manifest_sha256": _sha256_file(source_manifest),
        "eligible_parent_count": len(candidates),
        "selected_parent_count": len(chosen),
        "eligible_by_source_operator": {"|".join(group): value for group, value in sorted(eligible_by_group.items())},
        "selected_by_source_operator": {"|".join(group): value for group, value in sorted(chosen_by_group.items())},
        "rows": rows_manifest,
    }
    _atomic_text(output.selection, json.dumps(selection, indent=2, sort_keys=True) + "\n")
    return selection


def _runtime_parent_uuids(run_dir: Path) -> tuple[set[str], dict[str, Any]]:
    summary_path = run_dir / "analysis/summary.json"
    children_path = run_dir / "static/children.parquet"
    if not summary_path.is_file() or not children_path.is_file():
        raise FileNotFoundError(f"runtime shape lane is incomplete: {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    eligible = _nested(summary, "runtime.eligible")
    if not isinstance(eligible, Mapping) or eligible.get("available") is not True:
        raise ValueError(f"runtime shape lane has no eligible partition: {run_dir}")
    if eligible.get("coverage_complete") is not True:
        raise ValueError(f"runtime shape lane coverage is incomplete: {run_dir}")
    raw_child_uuids = eligible.get("eligible_child_uuids")
    if not isinstance(raw_child_uuids, list) or any(
        not isinstance(value, str) or not value for value in raw_child_uuids
    ):
        raise ValueError(f"runtime shape lane eligible UUIDs are invalid: {run_dir}")
    child_uuids = set(raw_child_uuids)
    if len(child_uuids) != len(raw_child_uuids):
        raise ValueError(f"runtime shape lane eligible UUIDs are duplicated: {run_dir}")
    parents: set[str] = set()
    found: set[str] = set()
    for row in _rows(children_path):
        child_uuid = _nested(row, "extra_info.uuid")
        if child_uuid not in child_uuids:
            continue
        found.add(str(child_uuid))
        parent_uuid = _nested(row, "extra_info.v4.parent_uuid")
        if not isinstance(parent_uuid, str) or not parent_uuid:
            raise ValueError(f"runtime child lacks canonical parent UUID: {child_uuid}")
        if parent_uuid in parents:
            raise ValueError(f"runtime lane has more than one eligible child for parent: {parent_uuid}")
        parents.add(parent_uuid)
    if found != child_uuids:
        raise ValueError(f"runtime eligible children are absent from parquet: {run_dir}")
    if len(parents) != int(eligible.get("eligible_children", -1)):
        raise ValueError(f"runtime eligible parent/child cardinality differs: {run_dir}")
    return parents, {
        "run_dir": str(run_dir.resolve()),
        "summary_path": str(summary_path.resolve()),
        "summary_sha256": _sha256_file(summary_path),
        "children_path": str(children_path.resolve()),
        "children_sha256": _sha256_file(children_path),
        "runtime_eligible_parents": len(parents),
    }


def select_full_measurable_residual(
    measurable_source_run: Path,
    runtime_shape_runs: Sequence[Path],
    prior_model_run: Path,
    run_dir: Path,
) -> dict[str, Any]:
    """Select every measurable parent not covered by prior runtime-safe or model lanes."""

    source_selected = measurable_source_run / "selected.parquet"
    source_selection = measurable_source_run / "selection.json"
    prior_selected = prior_model_run / "selected.parquet"
    prior_selection = prior_model_run / "selection.json"
    for path in (source_selected, source_selection, prior_selected, prior_selection):
        if not path.is_file():
            raise FileNotFoundError(path)
    source_manifest = json.loads(source_selection.read_text(encoding="utf-8"))
    if source_manifest.get("selected_count") != 53_896:
        raise ValueError("measurable source must be the complete 53,896-parent lane")
    if source_manifest.get("selected_sha256") != _sha256_file(source_selected):
        raise ValueError("measurable source selected SHA mismatch")

    runtime_parents: set[str] = set()
    runtime_sources: list[dict[str, Any]] = []
    for source_index, source_run in enumerate(runtime_shape_runs):
        parents, record = _runtime_parent_uuids(source_run)
        overlap = runtime_parents & parents
        if overlap:
            raise ValueError(f"runtime shape lanes overlap on parent: {sorted(overlap)[:3]}")
        runtime_parents.update(parents)
        runtime_sources.append({"source_index": source_index, **record})

    prior_manifest = json.loads(prior_selection.read_text(encoding="utf-8"))
    prior_rows = prior_manifest.get("rows")
    if not isinstance(prior_rows, list):
        raise ValueError("prior model selection rows are missing")
    prior_model_parents = {
        str(row.get("parent_uuid"))
        for row in prior_rows
        if isinstance(row, Mapping) and isinstance(row.get("parent_uuid"), str)
    }
    if len(prior_model_parents) != prior_manifest.get("selected_parent_count"):
        raise ValueError("prior model selection parent count differs")
    if prior_manifest.get("selected_parent_count") != pq.ParquetFile(prior_selected).metadata.num_rows:
        raise ValueError("prior model selected parquet count differs")
    overlap = runtime_parents & prior_model_parents
    if overlap:
        raise ValueError(f"prior model selection overlaps runtime-safe parent: {sorted(overlap)[:3]}")

    selected_rows: list[dict[str, Any]] = []
    rows_manifest: list[dict[str, Any]] = []
    source_parent_uuids: set[str] = set()
    for source_index, row in _iter_rows(source_selected):
        uuid, code, _ = _identity(row)
        if uuid in source_parent_uuids:
            raise ValueError(f"measurable source contains duplicate parent UUID: {uuid}")
        source_parent_uuids.add(uuid)
        if uuid in runtime_parents or uuid in prior_model_parents:
            continue
        source, operator = _group(row)
        selected_index = len(selected_rows)
        selected_rows.append(row)
        rows_manifest.append(
            {
                "selected_index": selected_index,
                "source_row_index": source_index,
                "parent_uuid": uuid,
                "parent_reference_sha256": _sha256_bytes(code.encode()),
                "source_family": source,
                "operator_family": operator,
                "target_input_bytes": target_input_bytes(row),
            }
        )
    missing_runtime = runtime_parents - source_parent_uuids
    missing_prior = prior_model_parents - source_parent_uuids
    if missing_runtime or missing_prior:
        raise ValueError(
            f"excluded parents are outside measurable source: runtime={len(missing_runtime)} prior={len(missing_prior)}"
        )
    expected = len(source_parent_uuids) - len(runtime_parents) - len(prior_model_parents)
    if len(selected_rows) != expected:
        raise AssertionError(f"full measurable residual cardinality mismatch: {len(selected_rows)}:{expected}")

    output = _paths(run_dir)
    output.selected.parent.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(output.selected, selected_rows, pq.ParquetFile(source_selected).schema_arrow)
    selection = {
        "contract_version": CONTRACT_VERSION,
        "selection_contract_version": FULL_RESIDUAL_SELECTION_CONTRACT_VERSION,
        "selection_reason": "full_measurable_residual_after_runtime_and_prior_model_exclusion",
        "selection_salt": FULL_RESIDUAL_SELECTION_SALT,
        "representative_for_projection": False,
        "source_run": str(measurable_source_run.resolve()),
        "source_selected_path": str(source_selected.resolve()),
        "source_selected_sha256": _sha256_file(source_selected),
        "source_selection_path": str(source_selection.resolve()),
        "source_selection_sha256": _sha256_file(source_selection),
        "measurable_parent_count": len(source_parent_uuids),
        "excluded_runtime_safe_parent_count": len(runtime_parents),
        "excluded_prior_model_parent_count": len(prior_model_parents),
        "eligible_parent_count": len(selected_rows),
        "selected_parent_count": len(selected_rows),
        "parent_gate_contract": "statically_measurable_below_128MiB; upstream FakeTensor pass not required",
        "scope_boundary": {
            "canonical_parent_count": BASELINE_FULL_ROWS,
            "already_large_parent_count": int(source_manifest.get("already_large_parent_count", -1)),
            "statically_unmeasurable_parent_count": BASELINE_FULL_ROWS
            - len(source_parent_uuids)
            - int(source_manifest.get("already_large_parent_count", -1)),
        },
        "runtime_shape_sources": runtime_sources,
        "prior_model_source": {
            "run_dir": str(prior_model_run.resolve()),
            "selected_path": str(prior_selected.resolve()),
            "selected_sha256": _sha256_file(prior_selected),
            "selection_path": str(prior_selection.resolve()),
            "selection_sha256": _sha256_file(prior_selection),
            "selected_parents": len(prior_model_parents),
        },
        "rows": rows_manifest,
    }
    _atomic_text(output.selection, json.dumps(selection, indent=2, sort_keys=True) + "\n")
    return {key: value for key, value in selection.items() if key != "rows"}


def select_relaxed_provenance_residual(canonical_parents: Path, run_dir: Path) -> dict[str, Any]:
    """Select strict-analysis failures whose direct factory storage still resolves."""

    from tools.data.synthesize.augment_prompt_tasks import analyze_code

    selected_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    strict_failure_reasons: collections.Counter[str] = collections.Counter()
    factory_resolution_failure_reasons: collections.Counter[str] = collections.Counter()
    input_profile_failure_reasons: collections.Counter[str] = collections.Counter()
    strict_measurable = 0
    already_large = 0
    for source_row_index, row in _iter_rows(canonical_parents):
        uuid, code, entry_point = _identity(row)
        try:
            analyze_code(code, entry_point)
            strict_measurable += 1
            continue
        except (SyntaxError, ValueError) as exc:
            strict_reason = f"{type(exc).__name__}:{exc}"
            strict_failure_reasons[strict_reason] += 1
        try:
            _, factory_count = relaxed_factory_storage(code, entry_point)
        except (SyntaxError, ValueError) as exc:
            factory_resolution_failure_reasons[f"{type(exc).__name__}:{exc}"] += 1
            continue
        input_profile = _fake_input_profile(code, timeout_seconds=DEFAULT_FAKE_TIMEOUT_SECONDS)
        if not input_profile.passed or input_profile.input_bytes is None:
            input_profile_failure_reasons[
                f"{input_profile.status}:{input_profile.exception_type}:{input_profile.reason}"
            ] += 1
            continue
        input_bytes = int(input_profile.input_bytes)
        if input_bytes >= 128 * 1024**2:
            already_large += 1
            continue
        source, operator = _group(row)
        selected_index = len(selected_rows)
        selected_rows.append(row)
        manifest_rows.append(
            {
                "selected_index": selected_index,
                "source_row_index": source_row_index,
                "parent_uuid": uuid,
                "parent_reference_sha256": _sha256_bytes(code.encode()),
                "source_family": source,
                "operator_family": operator,
                "input_bytes_before": input_bytes,
                "factory_count": factory_count,
                "strict_analysis_failure": strict_reason,
                "input_measurement_contract": RELAXED_STORAGE_CONTRACT,
                "input_profile": input_profile.as_dict(),
                "target_input_bytes": target_input_bytes_from_size(uuid, input_bytes),
            }
        )
    canonical_count = pq.ParquetFile(canonical_parents).metadata.num_rows
    factory_unresolved = sum(factory_resolution_failure_reasons.values())
    input_profile_unresolved = sum(input_profile_failure_reasons.values())
    relaxed_unresolved = factory_unresolved + input_profile_unresolved
    if strict_measurable + len(selected_rows) + already_large + relaxed_unresolved != canonical_count:
        raise RuntimeError("relaxed residual census does not close over canonical parents")
    output = _paths(run_dir)
    output.selected.parent.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(output.selected, selected_rows, pq.ParquetFile(canonical_parents).schema_arrow)
    selection = {
        "contract_version": CONTRACT_VERSION,
        "selection_contract_version": RELAXED_RESIDUAL_SELECTION_CONTRACT_VERSION,
        "selection_reason": "strict_return_provenance_failed_but_fake_returned_tensor_storage_resolved_below_128MiB",
        "canonical_parent_count": canonical_count,
        "strict_measurable_parent_count": strict_measurable,
        "relaxed_selected_parent_count": len(selected_rows),
        "relaxed_already_large_parent_count": already_large,
        "relaxed_unresolved_parent_count": relaxed_unresolved,
        "factory_shape_unresolved_parent_count": factory_unresolved,
        "fake_input_profile_unresolved_parent_count": input_profile_unresolved,
        "selected_parent_count": len(selected_rows),
        "eligible_parent_count": len(selected_rows),
        "parent_gate_contract": RELAXED_STORAGE_CONTRACT,
        "source_path": str(canonical_parents.resolve()),
        "source_sha256": _sha256_file(canonical_parents),
        "strict_failure_reasons": dict(strict_failure_reasons.most_common()),
        "factory_resolution_failure_reasons": dict(factory_resolution_failure_reasons.most_common()),
        "input_profile_failure_reasons": dict(input_profile_failure_reasons.most_common()),
        "rows": manifest_rows,
    }
    _atomic_text(output.selection, json.dumps(selection, indent=2, sort_keys=True) + "\n")
    return {key: value for key, value in selection.items() if key != "rows"}


def preview(run_dir: Path, count: int, *, candidate: bool = False) -> dict[str, Any]:
    output = _paths(run_dir)
    selection = json.loads(output.selection.read_text(encoding="utf-8"))
    by_uuid = {str(item["parent_uuid"]): item for item in selection["rows"]}
    existing = _load_generation(output.generation)
    prompt_version = PROMPT_VERSION if candidate else _generation_prompt_version(existing)
    destination = output.candidate_prompts if candidate else output.prompts
    blocks = [
        "# DSV4 shape hard-tail prompt review",
        "",
        f"Prompt version: `{prompt_version}`",
        "",
    ]
    rendered = 0
    for row in _rows(output.selected):
        if rendered >= count:
            break
        uuid, _, _ = _identity(row)
        targets = by_uuid[uuid]["target_input_bytes"]
        prompt = render_user_prompt(row, targets, prompt_version)
        blocks.extend(
            [
                f"## Parent {rendered + 1}: `{uuid}`",
                "",
                "````text",
                prompt,
                "````",
                "",
            ]
        )
        rendered += 1
    _atomic_text(destination, "\n".join(blocks).rstrip() + "\n")
    return {
        "prompt_version": prompt_version,
        "prompts_written": rendered,
        "path": str(destination.resolve()),
    }


def _request_payload(prompt: str) -> dict[str, Any]:
    return {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "chat_template_kwargs": {"thinking": True},
        "reasoning_effort": "low",
    }


def _post(endpoint: str, payload: Mapping[str, Any], timeout: float) -> tuple[int, Any]:
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": exc.read().decode(errors="replace")}


def _require_healthy_endpoints(endpoints: Sequence[str]) -> None:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for endpoint in endpoints:
        request = urllib.request.Request(endpoint.rstrip("/") + "/health", method="GET")
        try:
            with opener.open(request, timeout=20) as response:
                if response.status != 200:
                    raise ValueError(f"endpoint_health_status:{endpoint}:{response.status}")
        except Exception as exc:
            raise ValueError(f"endpoint_health_failed:{endpoint}:{type(exc).__name__}:{exc}") from exc


def _endpoint_for(uuid: str, endpoints: Sequence[str]) -> str:
    return endpoints[int(_sha256_bytes(f"endpoint:{uuid}".encode())[:16], 16) % len(endpoints)]


def _load_generation(path: Path) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    if not path.is_file():
        return records
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid_generation_line:{line_number}:{exc}") from exc
            uuid = str(record.get("parent_uuid"))
            prior = records.get(uuid)
            attempt = record.get("attempt_index", 0)
            if type(attempt) is not int or attempt < 0:
                raise ValueError(f"invalid_generation_attempt:{line_number}:{uuid}")
            if prior is not None and attempt <= int(prior.get("attempt_index", 0)):
                raise ValueError(f"nonmonotonic_generation_attempt:{line_number}:{uuid}")
            records[uuid] = record
    return records


def _generation_prompt_version(records: Mapping[str, Mapping[str, Any]]) -> str:
    if not records:
        return PROMPT_VERSION
    versions = {str(record.get("prompt_version")) for record in records.values()}
    if len(versions) != 1:
        raise ValueError(f"generation_contains_multiple_prompt_versions:{sorted(versions)}")
    version = versions.pop()
    if version not in USER_TEMPLATES:
        raise ValueError(f"unsupported_generation_prompt_version:{version}")
    return version


def _generation_contract_version(records: Mapping[str, Mapping[str, Any]]) -> str:
    if not records:
        return GENERATION_CONTRACT_VERSION
    versions = {str(record.get("generation_contract_version")) for record in records.values()}
    if len(versions) != 1:
        raise ValueError(f"generation_contains_multiple_contract_versions:{sorted(versions)}")
    version = versions.pop()
    if version not in GENERATION_CONTRACT_VERSIONS:
        raise ValueError(f"unsupported_generation_contract_version:{version}")
    return version


def _write_generation_review(
    path: Path, rows: Sequence[Mapping[str, Any]], records: Mapping[str, Mapping[str, Any]]
) -> None:
    blocks = ["# DSV4 shape generation samples", ""]
    written = 0
    for row in rows:
        if written >= 12:
            break
        uuid, _, _ = _identity(row)
        record = records.get(uuid)
        if record is None:
            continue
        try:
            content = _response_content(record)
        except ValueError as exc:
            content = f"GENERATION ERROR: {exc}"
        blocks.extend(
            [
                f"## `{uuid}`",
                "",
                f"Endpoint: `{record.get('endpoint')}`",
                "",
                "````text",
                content,
                "````",
                "",
            ]
        )
        written += 1
    _atomic_text(path, "\n".join(blocks).rstrip() + "\n")


def generate(
    run_dir: Path,
    endpoints: Sequence[str],
    concurrency_per_endpoint: int,
    timeout: float,
) -> dict[str, Any]:
    if not 1 <= len(endpoints) <= 4 or len(set(endpoints)) != len(endpoints):
        raise ValueError("generation_requires_one_to_four_distinct_endpoints")
    if not 1 <= concurrency_per_endpoint <= 64:
        raise ValueError("concurrency_per_endpoint_must_be_between_1_and_64")
    _require_healthy_endpoints(endpoints)
    output = _paths(run_dir)
    selection = json.loads(output.selection.read_text(encoding="utf-8"))
    metadata = {str(item["parent_uuid"]): item for item in selection["rows"]}
    rows = _rows(output.selected)
    completed = _load_generation(output.generation)
    prompt_version = _generation_prompt_version(completed)
    generation_contract_version = _generation_contract_version(completed)
    active_endpoints = list(endpoints)
    endpoint_pool = list(active_endpoints)
    if completed and generation_contract_version == GENERATION_CONTRACT_VERSION:
        recorded_pools = {
            (
                tuple(record.get("endpoint_pool", [])),
                record.get("endpoint_pool_sha256"),
            )
            for record in completed.values()
        }
        if len(recorded_pools) != 1:
            raise ValueError("generation_resume_contains_multiple_endpoint_pools")
        recorded_pool, recorded_pool_sha256 = recorded_pools.pop()
        if (
            not 1 <= len(recorded_pool) <= 4
            or len(set(recorded_pool)) != len(recorded_pool)
            or any(not isinstance(endpoint, str) or not endpoint for endpoint in recorded_pool)
        ):
            raise ValueError("generation_resume_endpoint_pool_is_invalid")
        endpoint_pool = list(recorded_pool)
        expected_pool_sha256 = _sha256_bytes(json.dumps(endpoint_pool, separators=(",", ":")).encode())
        if recorded_pool_sha256 != expected_pool_sha256:
            raise ValueError("generation_resume_endpoint_pool_sha256_mismatch")
        if not set(active_endpoints).issubset(endpoint_pool):
            raise ValueError("generation_resume_active_endpoint_is_outside_frozen_pool")
    endpoint_pool_sha256 = _sha256_bytes(json.dumps(endpoint_pool, separators=(",", ":")).encode())
    pending: list[tuple[int, dict[str, Any], str, str, dict[str, Any]]] = []
    for index, row in enumerate(rows):
        uuid, code, _ = _identity(row)
        targets = metadata[uuid]["target_input_bytes"]
        prompt = render_user_prompt(row, targets, prompt_version)
        prompt_hash = _sha256_bytes(prompt.encode())
        prior = completed.get(uuid)
        if prior is not None:
            expected = (
                prior.get("generation_contract_version") == generation_contract_version
                and prior.get("prompt_version") == prompt_version
                and prior.get("prompt_sha256") == prompt_hash
                and prior.get("parent_reference_sha256") == _sha256_bytes(code.encode())
                and prior.get("target_input_bytes") == targets
                and (
                    generation_contract_version == GENERATION_CONTRACT_VERSION_V1
                    or (
                        prior.get("endpoint_pool") == endpoint_pool
                        and prior.get("endpoint_pool_sha256") == endpoint_pool_sha256
                    )
                )
            )
            if not expected:
                raise ValueError(f"incompatible_generation_resume_record:{uuid}")
            if prior.get("http_status") == 200 and prior.get("error") is None:
                continue
        pending.append((index, row, prompt, prompt_hash, targets))

    semaphores = {endpoint: threading.BoundedSemaphore(concurrency_per_endpoint) for endpoint in active_endpoints}

    def one(item: tuple[int, dict[str, Any], str, str, dict[str, Any]]) -> dict[str, Any]:
        index, row, prompt, prompt_hash, targets = item
        uuid, code, _ = _identity(row)
        endpoint = _endpoint_for(uuid, active_endpoints)
        payload = _request_payload(prompt)
        attempt_index = int(completed.get(uuid, {}).get("attempt_index", -1)) + 1
        started = time.time()
        error = None
        try:
            with semaphores[endpoint]:
                status, response = _post(endpoint, payload, timeout)
        except Exception as exc:  # network evidence is retained for resume/review
            status, response, error = 0, {}, f"{type(exc).__name__}:{exc}"
        return {
            "contract_version": CONTRACT_VERSION,
            "generation_contract_version": generation_contract_version,
            "selected_index": index,
            "attempt_index": attempt_index,
            "parent_uuid": uuid,
            "parent_reference_sha256": _sha256_bytes(code.encode()),
            "prompt_version": prompt_version,
            "prompt_sha256": prompt_hash,
            "target_input_bytes": targets,
            "endpoint": endpoint,
            "endpoint_pool": endpoint_pool,
            "endpoint_pool_sha256": endpoint_pool_sha256,
            "request_contract": {
                "model": MODEL_NAME,
                "message_roles": ["user"],
                "reasoning_effort": "low",
                "thinking": True,
                "temperature": 0.6,
                "top_p": 0.95,
                "max_tokens": DEFAULT_MAX_TOKENS,
            },
            "http_status": status,
            "response": response,
            "error": error,
            "elapsed_seconds": round(time.time() - started, 3),
        }

    output.generation.parent.mkdir(parents=True, exist_ok=True)
    with output.generation.open("a", encoding="utf-8") as handle:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(endpoints) * concurrency_per_endpoint) as executor:
            futures = [executor.submit(one, item) for item in pending]
            for completed_count, future in enumerate(concurrent.futures.as_completed(futures), start=1):
                record = future.result()
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                if completed_count % 8 == 0:
                    handle.flush()
                completed[str(record["parent_uuid"])] = record
        handle.flush()
        os.fsync(handle.fileno())
    _write_generation_review(output.generation_review, rows, completed)
    return {
        "parents": len(rows),
        "resumed": len(rows) - len(pending),
        "executed": len(pending),
        "http_200": sum(record.get("http_status") == 200 for record in completed.values()),
        "generation_contract_version": generation_contract_version,
        "prompt_version": prompt_version,
        "active_endpoints": active_endpoints,
        "endpoint_pool": endpoint_pool,
        "by_endpoint": dict(collections.Counter(str(record.get("endpoint")) for record in completed.values())),
    }


def _response_content(record: Mapping[str, Any]) -> str:
    if record.get("http_status") != 200:
        raise ValueError(f"generation_http_status_{record.get('http_status')}")
    try:
        content = record["response"]["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ValueError("generation_response_missing_content") from None
    if not isinstance(content, str) or not content.strip():
        raise ValueError("generation_response_empty_content")
    return content


def _variants(content: str) -> dict[str, str]:
    match = SECTION_RE.fullmatch(content)
    if match is None:
        raise ValueError("response_must_contain_exact_medium_large_markdown_sections")
    return {variant: match.group(variant).rstrip() + "\n" for variant in VARIANTS}


def _span_key(span: SourceSpan) -> tuple[int, int, int, int]:
    return span.lineno, span.col_offset, span.end_lineno, span.end_col_offset


def _aligned_changes(
    parent_code: str, proposal_code: str
) -> tuple[dict[tuple[int, int, int, int], int], dict[tuple[int, int, int, int], int], list[int]]:
    parent_tree, proposal_tree = ast.parse(parent_code), ast.parse(proposal_code)
    parent_ids = _shape_numeric_node_ids(parent_tree)
    proposal_ids = _shape_numeric_node_ids(proposal_tree)
    parent_shape = [
        node for node in ast.walk(parent_tree) if isinstance(node, ast.Constant) and id(node) in parent_ids
    ]
    proposal_shape = [
        node for node in ast.walk(proposal_tree) if isinstance(node, ast.Constant) and id(node) in proposal_ids
    ]
    if len(parent_shape) != len(proposal_shape):
        raise ValueError("shape_literal_count_changed")
    shape_changes: dict[tuple[int, int, int, int], int] = {}
    changed_pairs: set[tuple[int | float, int | float]] = set()
    proposed_values: list[int] = []
    for before, after in zip(parent_shape, proposal_shape, strict=True):
        if before.value == after.value:
            continue
        if type(before.value) is not int or type(after.value) is not int or after.value <= 0:
            raise ValueError("changed_shape_literal_must_be_a_positive_integer")
        shape_changes[_span_key(SourceSpan.from_node(before))] = after.value
        changed_pairs.add((before.value, after.value))
        proposed_values.append(after.value)

    parent_init = _numeric_constants(_function(parent_tree, "get_init_inputs"))
    proposal_init = _numeric_constants(_function(proposal_tree, "get_init_inputs"))
    if len(parent_init) != len(proposal_init):
        raise ValueError("get_init_inputs_numeric_literal_count_changed")
    init_changes: dict[tuple[int, int, int, int], int] = {}
    for before, after in zip(parent_init, proposal_init, strict=True):
        if before.value == after.value:
            continue
        if (
            type(before.value) is not int
            or type(after.value) is not int
            or after.value <= 0
            or (before.value, after.value) not in changed_pairs
        ):
            raise ValueError("get_init_inputs_change_is_not_linked_to_an_input_shape_change")
        init_changes[_span_key(SourceSpan.from_node(before))] = after.value
    return shape_changes, init_changes, proposed_values


def _patch_spans(
    code: str,
    changes: Mapping[tuple[int, int, int, int], int],
) -> str:
    raw = code.encode("utf-8")
    lines = raw.splitlines(keepends=True)
    line_offsets: list[int] = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line)
    replacements: list[tuple[int, int, bytes]] = []
    for (line, column, end_line, end_column), value in changes.items():
        if line != end_line or not 1 <= line <= len(lines):
            raise ValueError("invalid_shape_change_source_span")
        start = line_offsets[line - 1] + column
        end = line_offsets[end_line - 1] + end_column
        observed = ast.literal_eval(raw[start:end].decode("utf-8"))
        if type(observed) is not int:
            raise ValueError("shape_change_source_span_is_not_an_integer")
        replacements.append((start, end, str(value).encode("ascii")))
    result = raw
    previous_start = len(raw) + 1
    for start, end, replacement in sorted(replacements, reverse=True):
        if end > previous_start:
            raise ValueError("overlapping_shape_change_source_spans")
        result = result[:start] + replacement + result[end:]
        previous_start = start
    child = result.decode("utf-8")
    ast.parse(child)
    return child


def _canonicalize(
    parent_code: str,
    proposal_code: str,
    entry_point: str,
    *,
    relaxed_return_provenance: bool = False,
) -> tuple[str, list[dict[str, Any]], list[int]]:
    # This first comparison rejects changed docstrings, equivalent rewrites,
    # rank changes, Model changes, and all non-shape edits before reconstruction.
    gate = relaxed_structure_gate if relaxed_return_provenance else static_gate
    gate(parent_code, proposal_code, entry_point)
    shape_changes, init_changes, proposed_values = _aligned_changes(parent_code, proposal_code)
    if not shape_changes:
        raise ValueError("proposal_changes_no_input_shape_literal")
    slots, rejected_slots = _shape_slots_with_rejections(
        parent_code,
        entry_point,
        allow_coupled_consumers=True,
    )
    # Shared-axis groups are search conveniences for the historical solver.
    # A model proposal already names every changed source occurrence, so its
    # canonical manifest uses the non-overlapping atomic literal/name slots.
    slots = [slot for slot in slots if slot.kind in {"literal", "linked_name"}]
    selected: list[tuple[ShapeSlot, int]] = []
    covered: set[tuple[int, int, int, int]] = set()
    for slot in slots:
        keys = {_span_key(span) for span in slot.patch_spans}
        present = keys.intersection(shape_changes)
        if not present:
            continue
        if present != keys:
            raise ValueError(f"logical_shape_slot_only_partially_changed:{slot.slot_id}")
        values = {shape_changes[key] for key in keys}
        if len(values) != 1:
            raise ValueError(f"logical_shape_slot_has_conflicting_values:{slot.slot_id}")
        new_value = values.pop()
        if new_value <= slot.old_value:
            raise ValueError(
                f"logical_shape_slot_did_not_strictly_increase:{slot.slot_id}:" f"{slot.old_value}:{new_value}"
            )
        selected.append((slot, new_value))
        covered.update(keys)
    unknown = set(shape_changes) - covered
    if unknown:
        rejected_ids = [item.get("slot_id") for item in rejected_slots if isinstance(item, Mapping)]
        raise ValueError(
            f"changed_shape_literal_is_not_an_accepted_logical_slot:"
            f"{len(unknown)}:fixed_record_candidates={rejected_ids[:8]}"
        )
    if len(selected) > 8:
        raise ValueError(f"logical_shape_slot_count_above_eight:{len(selected)}")
    canonical = _patch_spans(parent_code, {**shape_changes, **init_changes})
    canonical_static = gate(parent_code, canonical, entry_point)
    slots_manifest: list[dict[str, Any]] = []
    for slot, new_value in selected:
        balance = _affected_shape_balance_guard(canonical, slot, new_value)
        if balance.get("passed") is not True:
            raise ValueError(f"dimension_balance_guard_rejected:{slot.slot_id}:{balance.get('reason')}")
        slots_manifest.append(
            {
                "slot_id": slot.slot_id,
                "kind": slot.kind,
                "old_value": slot.old_value,
                "new_value": new_value,
                "power_of_two": new_value & (new_value - 1) == 0,
                "patch_spans": [span.as_dict() for span in slot.patch_spans],
                "occurrences": [item.as_dict() for item in slot.occurrences],
                "dimension_balance_guard": balance,
            }
        )
    if canonical_static["child_reference_sha256"] != _sha256_bytes(canonical.encode()):
        raise AssertionError("canonical_child_hash_mismatch")
    return canonical, slots_manifest, proposed_values


def _make_child(parent: Mapping[str, Any], child_code: str, static: Mapping[str, Any]) -> dict[str, Any]:
    child = copy.deepcopy(dict(parent))
    parent_uuid, parent_code, entry_point = _identity(parent)
    child_hash = str(static["child_reference_sha256"])
    child_uuid = f"shapeai_{_sha256_bytes(f'{parent_uuid}:{child_hash}'.encode())[:24]}"
    child["reward_model"]["ground_truth"] = child_code
    child["prompt"] = _replace_reference(child.get("prompt"), parent_code, child_code, required=True)
    extra = child["extra_info"]
    if extra.get("original_prompt") is not None:
        extra["original_prompt"] = _replace_reference(
            extra.get("original_prompt"), parent_code, child_code, required=False
        )
    extra["uuid"] = child_uuid
    v4 = dict(extra.get("v4") or {})
    v4.update(
        {
            "parent_uuid": parent_uuid,
            "reference_sha256": child_hash,
            "normalized_ast_sha256": str(static["child_normalized_ast_sha256"]),
            "included_in_review_train": False,
            "runtime_validation_status": "model_shape_static_fake_pass_runtime_required",
            "governance_status": "model_shape_review_only",
        }
    )
    extra["v4"] = v4
    fatal, _ = inspect_row_schema(child, child_code)
    if fatal:
        raise ValueError(f"child_row_schema_validation_failed:{fatal}")
    if entry_point not in {node.name for node in ast.parse(child_code).body if isinstance(node, ast.ClassDef)}:
        raise ValueError("child_entry_point_missing")
    return child


def _decision_base(parent: Mapping[str, Any], record: Mapping[str, Any] | None, variant: str) -> dict[str, Any]:
    uuid, code, _ = _identity(parent)
    return {
        "parent_uuid": uuid,
        "parent_reference_sha256": _sha256_bytes(code.encode()),
        "variant": variant,
        "accepted": False,
        "endpoint": record.get("endpoint") if record else None,
        "attempts": [],
    }


def _materialize_parent(
    task: tuple[int, dict[str, Any], Mapping[str, Any] | None, Mapping[str, Any]],
) -> dict[str, Any]:
    selected_index, parent, record, selection_row = task
    uuid, parent_code, entry_point = _identity(parent)
    decisions = {variant: _decision_base(parent, record, variant) for variant in VARIANTS}
    result: dict[str, Any] = {
        "selected_index": selected_index,
        "parent": parent,
        "children": [],
        "decisions": decisions,
    }
    if record is None:
        for decision in decisions.values():
            decision["reason"] = "missing_generation_record"
        return result
    if record.get("generation_contract_version") not in GENERATION_CONTRACT_VERSIONS:
        for decision in decisions.values():
            decision["reason"] = "generation_contract_mismatch"
        return result
    if record.get("parent_reference_sha256") != _sha256_bytes(parent_code.encode()):
        for decision in decisions.values():
            decision["reason"] = "generation_parent_hash_mismatch"
        return result
    targets = selection_row.get("target_input_bytes")
    if record.get("target_input_bytes") != targets:
        for decision in decisions.values():
            decision["reason"] = "generation_target_mismatch"
        return result
    record_prompt_version = str(record.get("prompt_version"))
    try:
        expected_prompt = render_user_prompt(parent, targets, record_prompt_version)
    except ValueError:
        for decision in decisions.values():
            decision["reason"] = "generation_prompt_or_effort_contract_mismatch"
        return result
    request_contract = record.get("request_contract")
    if (
        record.get("prompt_sha256") != _sha256_bytes(expected_prompt.encode())
        or not isinstance(request_contract, Mapping)
        or request_contract.get("message_roles") != ["user"]
        or request_contract.get("reasoning_effort") != "low"
        or request_contract.get("thinking") is not True
    ):
        for decision in decisions.values():
            decision["reason"] = "generation_prompt_or_effort_contract_mismatch"
        return result
    try:
        proposals = _variants(_response_content(record))
    except (SyntaxError, TypeError, ValueError) as exc:
        for decision in decisions.values():
            decision["reason"] = str(exc)
        return result

    measurement_contract = selection_row.get("input_measurement_contract")
    if measurement_contract == RELAXED_STORAGE_CONTRACT:
        relaxed_return_provenance = True
    elif measurement_contract in (None, "proven_returned_input_storage_v1"):
        relaxed_return_provenance = False
    else:
        for decision in decisions.values():
            decision["reason"] = f"unsupported_input_measurement_contract:{measurement_contract}"
        return result
    parent_input_profile = None
    if relaxed_return_provenance:
        parent_input_profile = _fake_input_profile(
            parent_code,
            timeout_seconds=DEFAULT_FAKE_TIMEOUT_SECONDS,
        )
        if (
            not parent_input_profile.passed
            or parent_input_profile.input_bytes is None
            or int(parent_input_profile.input_bytes) != int(selection_row.get("input_bytes_before", -1))
            or _canonical_json(parent_input_profile.as_dict()) != _canonical_json(selection_row.get("input_profile"))
        ):
            for decision in decisions.values():
                decision["reason"] = "parent_fake_input_profile_differs_from_selection"
            return result

    seen_hashes: set[str] = set()
    for variant in VARIANTS:
        decision = decisions[variant]
        proposal = proposals[variant]
        target = int(targets[variant])
        decision["target_input_bytes"] = target
        try:
            try:
                _, _, raw_proposed_values = _aligned_changes(parent_code, proposal)
                decision["proposed_changed_values"] = raw_proposed_values
            except (SyntaxError, TypeError, ValueError):
                pass
            canonical, slots, _ = _canonicalize(
                parent_code,
                proposal,
                entry_point,
                relaxed_return_provenance=relaxed_return_provenance,
            )
            child_input_profile = None
            if relaxed_return_provenance:
                assert parent_input_profile is not None and parent_input_profile.input_bytes is not None
                child_input_profile = _fake_input_profile(
                    canonical,
                    timeout_seconds=DEFAULT_FAKE_TIMEOUT_SECONDS,
                )
                if not child_input_profile.passed or child_input_profile.input_bytes is None:
                    raise ValueError(
                        f"child_fake_input_profile_{child_input_profile.status}:" f"{child_input_profile.reason}"
                    )
                if child_input_profile.tensor_count != parent_input_profile.tensor_count:
                    raise ValueError("returned_input_tensor_count_changed")
                for parent_tensor, child_tensor in zip(
                    parent_input_profile.tensors,
                    child_input_profile.tensors,
                    strict=True,
                ):
                    if len(parent_tensor[0]) != len(child_tensor[0]):
                        raise ValueError("returned_input_tensor_rank_changed")
                    if parent_tensor[1] != child_tensor[1]:
                        raise ValueError("returned_input_tensor_dtype_changed")
                static = relaxed_static_gate(
                    parent_code,
                    canonical,
                    entry_point,
                    parent_input_bytes=int(parent_input_profile.input_bytes),
                    child_input_bytes=int(child_input_profile.input_bytes),
                )
            else:
                static = static_gate(parent_code, canonical, entry_point)
            after = int(static["input_bytes_after"])
            _validate_variant_storage(variant, after)
            target_error = _validate_target_proximity(after, target)
            fake = _fake_tensor_gate(canonical, entry_point, timeout_seconds=DEFAULT_FAKE_TIMEOUT_SECONDS)
            attempt = {
                "attempt_index": 0,
                "accepted": False,
                "slots": slots,
                "fake_gate": fake.as_dict(),
                "input_bytes_after": after,
                "target_input_bytes": target,
                "target_relative_error": target_error,
                "input_storage_contract": static["input_storage_contract"],
                "child_input_profile": child_input_profile.as_dict() if child_input_profile else None,
            }
            decision["attempts"] = [attempt]
            if not fake.passed:
                raise ValueError(f"child_fake_gate_{fake.status}:{fake.reason}")
            digest = str(static["child_reference_sha256"])
            if digest in seen_hashes:
                raise ValueError("duplicate_child_reference_across_variants")
            seen_hashes.add(digest)
            child = _make_child(parent, canonical, static)
            child_uuid = str(_nested(child, "extra_info.uuid"))
            decision.update(
                {
                    "accepted": True,
                    "reason": None,
                    "child_uuid": child_uuid,
                    "child_reference_sha256": digest,
                    "input_bytes_before": int(static["input_bytes_before"]),
                    "input_bytes_after": after,
                    "input_scale": float(static["input_scale"]),
                    "input_storage_contract": static["input_storage_contract"],
                    "parent_input_profile": parent_input_profile.as_dict() if parent_input_profile else None,
                    "child_input_profile": child_input_profile.as_dict() if child_input_profile else None,
                    "target_relative_error": target_error,
                    "solver": {"slots": slots},
                    "logical_slot_count": len(slots),
                }
            )
            attempt["accepted"] = True
            result["children"].append((variant, child))
        except (SyntaxError, TypeError, ValueError) as exc:
            decision["reason"] = str(exc)
    return result


def materialize(run_dir: Path, workers: int) -> dict[str, Any]:
    if workers <= 0:
        raise ValueError("workers_must_be_positive")
    output = _paths(run_dir)
    parents = _rows(output.selected)
    selection = json.loads(output.selection.read_text(encoding="utf-8"))
    selection_rows = {str(item["parent_uuid"]): item for item in selection.get("rows", [])}
    generations = _load_generation(output.generation)
    prompt_version = _generation_prompt_version(generations)
    generation_contract_version = _generation_contract_version(generations)
    tasks = []
    for index, parent in enumerate(parents):
        uuid, _, _ = _identity(parent)
        tasks.append((index, parent, generations.get(uuid), selection_rows[uuid]))

    results: list[dict[str, Any]] = []
    context = multiprocessing.get_context("spawn")
    # FakeTensor keeps dispatcher-mode state process-wide.  A malformed parent
    # can leave that state poisoned even after its structured rejection path;
    # never reuse the process for a different canonical parent.
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        max_tasks_per_child=1,
    ) as executor:
        for result in executor.map(_materialize_parent, tasks, chunksize=1):
            results.append(result)
    results.sort(key=lambda item: int(item["selected_index"]))
    children: list[dict[str, Any]] = []
    paired: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for result in results:
        parent = result["parent"]
        for variant in VARIANTS:
            decision = result["decisions"][variant]
            decision["selected_index"] = result["selected_index"]
            decisions.append(decision)
        parent_children = sorted(result["children"], key=lambda item: VARIANT_ORDER[item[0]])
        if parent_children:
            paired.append(parent)
        for _, child in parent_children:
            children.append(child)
            paired.append(child)

    schema = pq.ParquetFile(output.selected).schema_arrow
    _atomic_parquet(output.children, children, schema)
    _atomic_parquet(output.paired, paired, schema)
    reason_counts = collections.Counter(
        str(decision.get("reason")) for decision in decisions if decision.get("accepted") is not True
    )
    review = ["# Model shape materialization review", ""]
    accepted_reviewed = 0
    rejected_reviewed = 0
    parent_by_uuid = {_identity(parent)[0]: parent for parent in parents}
    child_by_uuid = {str(_nested(child, "extra_info.uuid")): child for child in children}
    for decision in decisions:
        uuid = str(decision["parent_uuid"])
        variant = str(decision["variant"])
        if decision.get("accepted") is True and accepted_reviewed < 12:
            child = child_by_uuid[str(decision["child_uuid"])]
            parent_code = str(_nested(parent_by_uuid[uuid], "reward_model.ground_truth"))
            child_code = str(_nested(child, "reward_model.ground_truth"))
            diff = "".join(
                difflib.unified_diff(
                    parent_code.splitlines(keepends=True),
                    child_code.splitlines(keepends=True),
                    fromfile=f"{uuid}:parent",
                    tofile=f"{uuid}:{variant}",
                )
            )
            review.extend(
                [
                    f"## Accepted `{uuid}` / `{variant}`",
                    "",
                    f"Slots: `{decision.get('logical_slot_count')}`; input bytes: "
                    f"`{decision.get('input_bytes_before')}` → `{decision.get('input_bytes_after')}`.",
                    "",
                    "```diff",
                    diff.rstrip(),
                    "```",
                    "",
                ]
            )
            accepted_reviewed += 1
        elif decision.get("accepted") is not True and rejected_reviewed < 20:
            review.extend(
                [
                    f"## Rejected `{uuid}` / `{variant}`",
                    "",
                    f"Reason: `{decision.get('reason')}`",
                    "",
                ]
            )
            rejected_reviewed += 1
    _atomic_text(output.materialized_review, "\n".join(review).rstrip() + "\n")
    manifest = {
        "contract_version": CONTRACT_VERSION,
        "generation_contract_version": generation_contract_version,
        "prompt_version": prompt_version,
        "selected_sha256": _sha256_file(output.selected),
        "selection_sha256": _sha256_file(output.selection),
        "generation_sha256": _sha256_file(output.generation),
        "children_sha256": _sha256_file(output.children),
        "paired_sha256": _sha256_file(output.paired),
        "group_contract": {
            "logical_slot_count_range": [1, 8],
            "fixed_cardinality": False,
            "power_of_two_quota": None,
            "maximum_dimension_imbalance_ratio": MAX_DIMENSION_IMBALANCE_RATIO,
        },
        "upstream_parent_fake_gate": selection.get("parent_gate_contract", "passed_in_source_manifest"),
        "child_fake_gate_timeout_seconds": DEFAULT_FAKE_TIMEOUT_SECONDS,
        "parent_count": len(parents),
        "variant_decision_count": len(decisions),
        "children_written": len(children),
        "parents_with_child": len({str(decision["parent_uuid"]) for decision in decisions if decision["accepted"]}),
        "paired_layout": "parent_once_followed_by_all_children",
        "paired_rows": len(paired),
        "rejection_reasons": dict(reason_counts.most_common()),
        "decisions": decisions,
    }
    _atomic_text(output.manifest, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return {
        "parents": len(parents),
        "children": len(children),
        "parents_with_child": manifest["parents_with_child"],
        "top_rejections": reason_counts.most_common(12),
    }


def _fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _rejection_category(reason: str) -> str:
    if reason.startswith(("generation_http_status_", "missing_generation_record")):
        return "transport_or_missing_generation"
    if "response_" in reason or "markdown_sections" in reason:
        return "response_format"
    if reason.startswith("model_changed"):
        return "model_changed"
    if "reference_changed_beyond" in reason:
        return "nonshape_or_structure_changed"
    if "literal_count_changed" in reason or "factory" in reason or "entry_point" in reason:
        return "rank_factory_or_structure_changed"
    if "logical_shape_slot" in reason or "accepted_logical_slot" in reason:
        return "invalid_or_dead_shape_slot"
    if "dimension_balance" in reason:
        return "dimension_balance"
    if "target_" in reason or "storage_" in reason or "input_scale" in reason:
        return "storage_or_target"
    if "fake_gate" in reason:
        return "fake_tensor_forward"
    if "prompt_or_effort_contract" in reason or "generation_contract" in reason:
        return "artifact_contract"
    return "other"


def _quantiles(values: Sequence[int | float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    result: dict[str, float] = {}
    for label, quantile in (("p50", 0.50), ("p75", 0.75), ("p90", 0.90), ("p95", 0.95), ("p99", 0.99)):
        position = (len(ordered) - 1) * quantile
        lower = math.floor(position)
        upper = math.ceil(position)
        value = ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)
        result[label] = value
    return result


def _log2_bucket(value: int) -> str:
    exponent = value.bit_length() - 1
    return f"2^{exponent}..2^{exponent + 1}-1"


def _distance_to_grid(value: int, grid: int) -> int:
    residue = value % grid
    return min(residue, grid - residue)


def _distance_to_power_of_two(value: int) -> int:
    lower = 1 << (value.bit_length() - 1)
    upper = lower << 1
    return min(value - lower, upper - value)


def _bucket_tvd(left: Sequence[int], right: Sequence[int]) -> float | None:
    """Total-variation distance between two log2-scale histograms."""

    if not left or not right:
        return None
    left_counts = collections.Counter(_log2_bucket(value) for value in left)
    right_counts = collections.Counter(_log2_bucket(value) for value in right)
    buckets = set(left_counts) | set(right_counts)
    return 0.5 * sum(abs(left_counts[bucket] / len(left) - right_counts[bucket] / len(right)) for bucket in buckets)


def _write_failure_review(
    path: Path,
    parents: Mapping[str, Mapping[str, Any]],
    generations: Mapping[str, Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
) -> None:
    """Dump bounded real examples for the failure modes used in the report."""

    blocks = ["# Model shape canary failure samples", ""]
    length_written = 0
    for uuid, generation in generations.items():
        try:
            choice = generation["response"]["choices"][0]
            if choice.get("finish_reason") != "length":
                continue
            reasoning = str(choice.get("message", {}).get("reasoning_content") or "")
        except (IndexError, KeyError, TypeError):
            continue
        parent_meta = metadata[uuid]
        blocks.extend(
            [
                f"## Length limit: `{uuid}`",
                "",
                f"Source/operator: `{parent_meta['source_family']}` / `{parent_meta['operator_family']}`. ",
                f"Reasoning characters: `{len(reasoning)}`.",
                "",
                "Reasoning head:",
                "",
                "```text",
                reasoning[:900].strip(),
                "```",
                "",
                "Reasoning tail:",
                "",
                "```text",
                reasoning[-900:].strip(),
                "```",
                "",
            ]
        )
        length_written += 1
        if length_written >= 4:
            break

    category_counts: collections.Counter[str] = collections.Counter()
    for decision in decisions:
        if decision.get("accepted") is True:
            continue
        reason = str(decision.get("reason"))
        category = _rejection_category(reason)
        if category in {"response_format", "transport_or_missing_generation"}:
            continue
        if category_counts[category] >= 4:
            continue
        uuid = str(decision["parent_uuid"])
        variant = str(decision["variant"])
        _, parent_code, _ = _identity(parents[uuid])
        proposal: str | None = None
        generation = generations.get(uuid)
        if generation is not None:
            try:
                proposal = _variants(_response_content(generation))[variant]
            except (SyntaxError, TypeError, ValueError):
                pass
        diff = ""
        if proposal is not None:
            diff = "\n".join(
                difflib.unified_diff(
                    parent_code.splitlines(),
                    proposal.splitlines(),
                    fromfile=f"{uuid}:parent",
                    tofile=f"{uuid}:{variant}",
                    lineterm="",
                )
            )
        parent_meta = metadata[uuid]
        blocks.extend(
            [
                f"## {category}: `{uuid}` / `{variant}`",
                "",
                f"Source/operator: `{parent_meta['source_family']}` / `{parent_meta['operator_family']}`.",
                "",
                f"Reason: `{reason}`",
                "",
            ]
        )
        if diff:
            blocks.extend(["```diff", diff, "```", ""])
        category_counts[category] += 1
    _atomic_text(path, "\n".join(blocks).rstrip() + "\n")


def analyze_bias(run_dir: Path) -> dict[str, Any]:
    output = _paths(run_dir)
    selection = json.loads(output.selection.read_text(encoding="utf-8"))
    manifest = json.loads(output.manifest.read_text(encoding="utf-8"))
    metadata = {str(item["parent_uuid"]): item for item in selection["rows"]}
    generations = _load_generation(output.generation)
    decisions = [item for item in manifest["decisions"] if isinstance(item, Mapping)]
    parents = {_identity(parent)[0]: parent for parent in _rows(output.selected)}
    _write_failure_review(output.failure_review, parents, generations, decisions, metadata)
    accepted = [item for item in decisions if item.get("accepted") is True]
    parents_with_child = {str(item["parent_uuid"]) for item in accepted}
    values: list[int] = []
    occurrence_values: list[int] = []
    proposed_values = [
        int(value)
        for decision in decisions
        for value in decision.get("proposed_changed_values", [])
        if type(value) is int and value > 0
    ]
    slot_counts: collections.Counter[int] = collections.Counter()
    axis_counts: collections.Counter[int] = collections.Counter()
    rank_counts: collections.Counter[int] = collections.Counter()
    output_bytes: list[int] = []
    input_scales: list[float] = []
    target_errors: list[float] = []
    for decision in accepted:
        slots = _nested(decision, "solver.slots", [])
        slot_counts[len(slots)] += 1
        output_bytes.append(int(decision["input_bytes_after"]))
        input_scales.append(float(decision["input_scale"]))
        target_errors.append(float(decision["target_relative_error"]))
        for slot in slots:
            value = int(slot["new_value"])
            values.append(value)
            occurrences = slot.get("occurrences", [])
            occurrence_values.extend([value] * len(occurrences))
            for occurrence in occurrences:
                axis_counts[int(occurrence["axis"])] += 1
                rank_counts[int(occurrence["rank"])] += 1
    frequency = collections.Counter(occurrence_values)
    proposed_frequency = collections.Counter(proposed_values)
    total_occurrences = len(occurrence_values)
    p2 = sum(value & (value - 1) == 0 for value in occurrence_values)
    multiple_10 = sum(value % 10 == 0 for value in occurrence_values)
    multiple_100 = sum(value % 100 == 0 for value in occurrence_values)
    multiple_1000 = sum(value % 1000 == 0 for value in occurrence_values)
    multiple_8 = sum(value % 8 == 0 for value in occurrence_values)
    multiple_32 = sum(value % 32 == 0 for value in occurrence_values)
    near_decimal_anchor = sum(value >= 100 and _distance_to_grid(value, 100) <= 3 for value in occurrence_values)
    near_binary_anchor = sum(value >= 8 and _distance_to_power_of_two(value) <= 3 for value in occurrence_values)
    top = frequency.most_common(10)
    top5_share = _fraction(sum(count for _, count in top[:5]), total_occurrences)
    top10_share = _fraction(sum(count for _, count in top), total_occurrences)
    hhi = sum((count / total_occurrences) ** 2 for count in frequency.values()) if total_occurrences else None
    proposed_top = proposed_frequency.most_common(10)
    proposed_p2 = sum(value & (value - 1) == 0 for value in proposed_values)
    proposed_near_decimal_anchor = sum(
        value >= 100 and _distance_to_grid(value, 100) <= 3 for value in proposed_values
    )
    proposed_near_binary_anchor = sum(
        value >= 8 and _distance_to_power_of_two(value) <= 3 for value in proposed_values
    )
    proposed_top10_share = _fraction(sum(count for _, count in proposed_top), len(proposed_values))

    def acceptance_table(field: str) -> dict[str, dict[str, Any]]:
        totals: collections.Counter[str] = collections.Counter()
        successes: collections.Counter[str] = collections.Counter()
        for decision in decisions:
            parent = metadata[str(decision["parent_uuid"])]
            value = str(parent[field])
            totals[value] += 1
            successes[value] += decision.get("accepted") is True
        return {
            value: {
                "decisions": totals[value],
                "accepted": successes[value],
                "acceptance_rate": successes[value] / totals[value],
            }
            for value in sorted(totals)
        }

    endpoint_totals: collections.Counter[str] = collections.Counter()
    endpoint_success: collections.Counter[str] = collections.Counter()
    for decision in decisions:
        endpoint = str(decision.get("endpoint"))
        endpoint_totals[endpoint] += 1
        endpoint_success[endpoint] += decision.get("accepted") is True
    endpoint_table = {
        endpoint: {
            "decisions": endpoint_totals[endpoint],
            "accepted": endpoint_success[endpoint],
            "acceptance_rate": endpoint_success[endpoint] / endpoint_totals[endpoint],
        }
        for endpoint in sorted(endpoint_totals)
    }
    variant_totals = collections.Counter(str(item["variant"]) for item in decisions)
    variant_success = collections.Counter(str(item["variant"]) for item in decisions if item.get("accepted") is True)
    variant_table = {
        variant: {
            "decisions": variant_totals[variant],
            "accepted": variant_success[variant],
            "acceptance_rate": variant_success[variant] / variant_totals[variant],
        }
        for variant in sorted(variant_totals)
    }
    accepted_parent_rate = _fraction(len(parents_with_child), int(selection["selected_parent_count"]))
    representative_for_projection = bool(
        selection.get(
            "representative_for_projection",
            selection.get("contract_version") == CONTRACT_VERSION,
        )
    )
    if representative_for_projection and accepted_parent_rate is not None:
        eligible_for_projection = int(selection["eligible_parent_count"])
        projected_added = round(eligible_for_projection * accepted_parent_rate)
        projected_changed = min(BASELINE_FULL_ROWS, BASELINE_CHANGED_PARENTS + projected_added)
        projected_unchanged = BASELINE_FULL_ROWS - projected_changed
        projected_coverage: dict[str, Any] = {
            "available": True,
            "method": "representative canary parent-acceptance rate applied to upstream hard-tail eligible count",
            "projected_additional_changed_parents": projected_added,
            "projected_changed_parents": projected_changed,
            "projected_unchanged_parents": projected_unchanged,
            "projected_unchanged_share": projected_unchanged / BASELINE_FULL_ROWS,
            "warning": "projection is diagnostic only; final distribution must be recomputed from runtime-accepted children",
        }
    else:
        projected_unchanged = None
        projected_coverage = {
            "available": False,
            "reason": "selection_is_not_representative_of_the_full_hardtail",
        }
    reasons = collections.Counter(str(item.get("reason")) for item in decisions if item.get("accepted") is not True)
    rejection_categories = collections.Counter(
        _rejection_category(str(item.get("reason"))) for item in decisions if item.get("accepted") is not True
    )
    finish_reasons: collections.Counter[str] = collections.Counter()
    completion_tokens: list[int] = []
    generation_elapsed: list[float] = []
    empty_generation_count = 0
    length_by_source: collections.Counter[str] = collections.Counter()
    generation_by_source: collections.Counter[str] = collections.Counter()
    length_by_operator: collections.Counter[str] = collections.Counter()
    generation_by_operator: collections.Counter[str] = collections.Counter()
    length_by_endpoint: collections.Counter[str] = collections.Counter()
    generation_by_endpoint: collections.Counter[str] = collections.Counter()
    prompt_tokens_by_finish: dict[str, list[int]] = collections.defaultdict(list)
    for uuid, generation in generations.items():
        source = str(metadata[uuid]["source_family"])
        operator = str(metadata[uuid]["operator_family"])
        endpoint = str(generation.get("endpoint"))
        generation_by_source[source] += 1
        generation_by_operator[operator] += 1
        generation_by_endpoint[endpoint] += 1
        try:
            choice = generation["response"]["choices"][0]
            finish_reason = str(choice.get("finish_reason"))
            finish_reasons[finish_reason] += 1
            if not str(choice.get("message", {}).get("content") or "").strip():
                empty_generation_count += 1
            if finish_reason == "length":
                length_by_source[source] += 1
                length_by_operator[operator] += 1
                length_by_endpoint[endpoint] += 1
            usage = generation["response"].get("usage", {})
            tokens = usage.get("completion_tokens")
            if type(tokens) is int:
                completion_tokens.append(tokens)
            prompt_tokens = usage.get("prompt_tokens")
            if type(prompt_tokens) is int:
                prompt_tokens_by_finish[finish_reason].append(prompt_tokens)
        except (IndexError, KeyError, TypeError):
            finish_reasons["malformed_response"] += 1
        elapsed = generation.get("elapsed_seconds")
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
            generation_elapsed.append(float(elapsed))
    warnings: list[str] = []
    power_share = _fraction(p2, total_occurrences)
    proposal_power_share = _fraction(proposed_p2, len(proposed_values))
    near_decimal_share = _fraction(near_decimal_anchor, total_occurrences)
    if power_share is not None and power_share < INVERSE_POWER_OF_TWO_WARNING_SHARE:
        warnings.append("power_of_two_active_avoidance_share_below_5_percent")
    if proposal_power_share is not None and proposal_power_share < INVERSE_POWER_OF_TWO_WARNING_SHARE:
        warnings.append("proposal_power_of_two_active_avoidance_share_below_5_percent")
    if power_share is not None and power_share > 0.70:
        warnings.append("power_of_two_absorption_share_above_70_percent")
    if near_decimal_share is not None and near_decimal_share > 0.05:
        warnings.append("near_decimal_100_grid_share_above_5_percent")
    if top10_share is not None and top10_share > 0.35:
        warnings.append("top10_changed_value_share_above_35_percent")
    if proposed_top10_share is not None and proposed_top10_share > 0.35:
        warnings.append("proposal_top10_changed_value_share_above_35_percent")
    if accepted and slot_counts[1] / len(accepted) > 0.25:
        warnings.append("single_logical_slot_child_share_above_25_percent")
    if generations and finish_reasons["length"] / len(generations) > 0.02:
        warnings.append("generation_length_limit_rate_above_2_percent")
    if output_bytes and sum(value % (1024**2) == 0 for value in output_bytes) / len(output_bytes) > 0.20:
        warnings.append("output_storage_integer_mib_share_above_20_percent")
    for label, table in (
        ("source", acceptance_table("source_family")),
        ("operator", acceptance_table("operator_family")),
        ("endpoint", endpoint_table),
    ):
        rates = [float(item["acceptance_rate"]) for item in table.values() if int(item["decisions"]) >= 20]
        if rates and max(rates) - min(rates) > 0.20:
            warnings.append(f"{label}_acceptance_rate_spread_above_20_points")
    report = {
        "contract_version": CONTRACT_VERSION,
        "prompt_version": str(manifest["prompt_version"]),
        "known_full_distribution_gap": {
            "rows": BASELINE_FULL_ROWS,
            "changed_parents": BASELINE_CHANGED_PARENTS,
            "unchanged_parents": BASELINE_UNCHANGED_PARENTS,
            "unchanged_share": BASELINE_UNCHANGED_PARENTS / BASELINE_FULL_ROWS,
            "hardtail_eligible_parents": HARDTAIL_ELIGIBLE_PARENTS,
            "per_tensor_numel_quantiles_current": {
                "p50": 786_432,
                "p75": 53_865_588,
                "p90": 399_507_456,
                "p95": 689_831_936,
                "p99": 1_017_249_792,
            },
            "per_tensor_numel_quantiles_kernelbench": {
                "p50": 33_554_432,
                "p75": 268_435_456,
                "p90": 1_610_612_736,
                "p95": 2_146_959_360,
                "p99": 2_147_483_648,
            },
            "known_limits": [
                "63.68% of parents remain unchanged after three nested static recovery rounds",
                "the 4 GiB aggregate-input cap truncates the extreme right tail, especially fp32",
                "changed values remain concentrated on familiar power-of-two anchors",
                "replacement coverage is weaker for CUDA-Agent and operator-dense parents",
            ],
        },
        "canary": {
            "selected_parents": int(selection["selected_parent_count"]),
            "variant_decisions": len(decisions),
            "accepted_variants": len(accepted),
            "parents_with_accepted_child": len(parents_with_child),
            "parent_acceptance_rate": accepted_parent_rate,
            "rejection_reasons": dict(reasons.most_common()),
            "rejection_categories": dict(rejection_categories.most_common()),
            "logical_slot_count_histogram": {str(key): value for key, value in sorted(slot_counts.items())},
            "changed_occurrence_axis_histogram": {str(key): value for key, value in sorted(axis_counts.items())},
            "changed_occurrence_rank_histogram": {str(key): value for key, value in sorted(rank_counts.items())},
            "leading_axis_occurrence_share": _fraction(axis_counts[0], total_occurrences),
            "trailing_axis_occurrence_share": _fraction(
                sum(
                    1
                    for decision in accepted
                    for slot in _nested(decision, "solver.slots", [])
                    for occurrence in slot.get("occurrences", [])
                    if int(occurrence["axis"]) == int(occurrence["rank"]) - 1
                ),
                total_occurrences,
            ),
            "changed_logical_values": len(values),
            "changed_occurrences": total_occurrences,
            "proposal_changed_values_observed": len(proposed_values),
            "proposal_unique_changed_values": len(proposed_frequency),
            "proposal_power_of_two_value_share": proposal_power_share,
            "proposal_near_decimal_100_grid_share": _fraction(proposed_near_decimal_anchor, len(proposed_values)),
            "proposal_near_power_of_two_within_3_share": _fraction(proposed_near_binary_anchor, len(proposed_values)),
            "proposal_top10_changed_value_share": proposed_top10_share,
            "proposal_top10_changed_values": [{"value": value, "count": count} for value, count in proposed_top],
            "unique_changed_values": len(frequency),
            "power_of_two_occurrence_share": power_share,
            "reference_power_of_two_shares": REFERENCE_POWER_OF_TWO_SHARES,
            "multiple_of_10_occurrence_share": _fraction(multiple_10, total_occurrences),
            "multiple_of_100_occurrence_share": _fraction(multiple_100, total_occurrences),
            "multiple_of_1000_occurrence_share": _fraction(multiple_1000, total_occurrences),
            "near_decimal_100_grid_share": near_decimal_share,
            "near_power_of_two_within_3_share": _fraction(near_binary_anchor, total_occurrences),
            "multiple_of_8_occurrence_share": _fraction(multiple_8, total_occurrences),
            "multiple_of_32_occurrence_share": _fraction(multiple_32, total_occurrences),
            "top5_changed_value_share": top5_share,
            "top10_changed_value_share": top10_share,
            "top10_changed_values": [{"value": value, "occurrences": count} for value, count in top],
            "inverse_hhi_effective_values": (1 / hhi if hhi else None),
            "accepted_logical_value_log2_buckets": dict(
                sorted(collections.Counter(_log2_bucket(value) for value in values).items())
            ),
            "proposal_value_log2_buckets": dict(
                sorted(collections.Counter(_log2_bucket(value) for value in proposed_values).items())
            ),
            "proposal_to_accepted_log2_bucket_tvd": _bucket_tvd(proposed_values, values),
            "input_bytes_quantiles": _quantiles(output_bytes),
            "input_scale_quantiles": _quantiles(input_scales),
            "target_relative_error_quantiles": _quantiles(target_errors),
            "integer_mib_output_share": _fraction(
                sum(value % (1024**2) == 0 for value in output_bytes), len(output_bytes)
            ),
            "source_acceptance": acceptance_table("source_family"),
            "operator_acceptance": acceptance_table("operator_family"),
            "variant_acceptance": variant_table,
            "endpoint_acceptance": endpoint_table,
        },
        "generation": {
            "records": len(generations),
            "finish_reasons": dict(finish_reasons.most_common()),
            "empty_content_count": empty_generation_count,
            "completion_token_quantiles": _quantiles(completion_tokens),
            "prompt_token_quantiles_by_finish_reason": {
                reason: _quantiles(tokens) for reason, tokens in sorted(prompt_tokens_by_finish.items())
            },
            "elapsed_second_quantiles": _quantiles(generation_elapsed),
            "length_limit_by_source": {
                source: {
                    "records": generation_by_source[source],
                    "length_limited": length_by_source[source],
                    "length_limit_rate": length_by_source[source] / generation_by_source[source],
                }
                for source in sorted(generation_by_source)
            },
            "length_limit_by_operator": {
                operator: {
                    "records": generation_by_operator[operator],
                    "length_limited": length_by_operator[operator],
                    "length_limit_rate": length_by_operator[operator] / generation_by_operator[operator],
                }
                for operator in sorted(generation_by_operator)
            },
            "length_limit_by_endpoint": {
                endpoint: {
                    "records": generation_by_endpoint[endpoint],
                    "length_limited": length_by_endpoint[endpoint],
                    "length_limit_rate": length_by_endpoint[endpoint] / generation_by_endpoint[endpoint],
                }
                for endpoint in sorted(generation_by_endpoint)
            },
        },
        "projected_full_parent_coverage": projected_coverage,
        "bias_warnings": warnings,
    }
    _atomic_text(output.bias_json, json.dumps(report, indent=2, sort_keys=True) + "\n")
    canary = report["canary"]
    lines = [
        "# Model shape canary bias report",
        "",
        "## Outcome",
        "",
        f"- Parents with a static+FakeTensor child: {len(parents_with_child)}/{selection['selected_parent_count']} ({(accepted_parent_rate or 0):.2%}).",
        f"- Accepted variants: {len(accepted)}/{len(decisions)}.",
        f"- Changed occurrences: {total_occurrences}; unique values: {len(frequency)}.",
        f"- Power-of-two share: {(canary['power_of_two_occurrence_share'] or 0):.2%}; top-10 share: {(top10_share or 0):.2%}.",
        f"- Near-decimal-100-grid share (within 3): {(canary['near_decimal_100_grid_share'] or 0):.2%}; "
        f"near-power-of-two share (within 3): {(canary['near_power_of_two_within_3_share'] or 0):.2%}.",
        (
            f"- Projected unchanged parents: {projected_unchanged}/{BASELINE_FULL_ROWS} "
            f"({projected_unchanged / BASELINE_FULL_ROWS:.2%}); diagnostic only."
            if projected_unchanged is not None
            else "- Full-distribution projection: unavailable for this non-representative selection."
        ),
        "",
        "## Bias warnings",
        "",
    ]
    lines.extend(f"- `{warning}`" for warning in warnings or ["none"])
    lines.extend(["", "## Rejection categories", ""])
    lines.extend(f"- {count}: `{category}`" for category, count in rejection_categories.most_common())
    lines.extend(["", "## Rejection reasons", ""])
    lines.extend(f"- {count}: `{reason}`" for reason, count in reasons.most_common(20))
    lines.extend(["", "## Top changed values", ""])
    lines.extend(f"- {value}: {count}" for value, count in top)
    _atomic_text(output.bias_markdown, "\n".join(lines).rstrip() + "\n")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    subparsers = parser.add_subparsers(dest="command", required=True)

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--count", type=int, default=DEFAULT_COUNT)
    subparsers.add_parser(
        "select-full-residual",
        help="Select every measurable parent not covered by runtime-safe static or prior model lanes.",
    )
    subparsers.add_parser(
        "select-relaxed-residual",
        help="Select strict-return-proof failures whose factory storage still resolves below 128 MiB.",
    )

    preview_parser = subparsers.add_parser("preview")
    preview_parser.add_argument("--count", type=int, default=8)

    preview_candidate_parser = subparsers.add_parser("preview-candidate")
    preview_candidate_parser.add_argument("--count", type=int, default=8)

    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument(
        "--endpoint",
        action="append",
        dest="endpoints",
        help=(
            "Use one active endpoint; repeat at most four times. Defaults to the four production endpoints. "
            "A v2 resume may use a healthy subset of its frozen original endpoint pool."
        ),
    )

    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)

    subparsers.add_parser("analyze-bias")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "select":
        result = select(DEFAULT_SOURCE_RUN, args.run_dir, args.count)
    elif args.command == "select-full-residual":
        result = select_full_measurable_residual(
            DEFAULT_MEASURABLE_SOURCE_RUN,
            DEFAULT_RUNTIME_SHAPE_RUNS,
            DEFAULT_PRIOR_MODEL_RUN,
            args.run_dir,
        )
    elif args.command == "select-relaxed-residual":
        result = select_relaxed_provenance_residual(DEFAULT_CANONICAL_PARENTS, args.run_dir)
    elif args.command == "preview":
        result = preview(args.run_dir, args.count)
    elif args.command == "preview-candidate":
        result = preview(args.run_dir, args.count, candidate=True)
    elif args.command == "generate":
        result = generate(
            args.run_dir,
            tuple(args.endpoints) if args.endpoints else DEFAULT_ENDPOINTS,
            DEFAULT_CONCURRENCY_PER_ENDPOINT,
            DEFAULT_TIMEOUT_SECONDS,
        )
    elif args.command == "materialize":
        result = materialize(args.run_dir, args.workers)
    elif args.command == "analyze-bias":
        result = analyze_bias(args.run_dir)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
