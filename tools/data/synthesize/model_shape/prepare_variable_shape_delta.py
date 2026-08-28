#!/usr/bin/env python3
"""Prepare the old full-run parents recovered by relaxing one lexical rule.

This is a selection-only bridge from the completed strict two-slot run to a
new variable-slot solver.  It does not generate children.  A parent is kept
only when its old decision:

* was ``no_strict_two_slot_exactly_one_power_solution``;
* had at least two affine slots and zero accepted strict bilinear profiles;
* has a positive bilinear witness after removing only the requirement that
  both slots' patch tokens be physically inside ``get_inputs``.

All other structural checks from the old pair predicate remain in force.  The
output order is the source ``selected.parquet`` row order.  The source UUIDs,
reference hashes, artifact hashes, helper-source hashes, and expected output
count are verified before the output directory is atomically installed.
"""

from __future__ import annotations

import argparse
import ast
import collections
import concurrent.futures
import csv
import hashlib
import itertools
import json
import multiprocessing
import os
import shutil
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.synthesize.augment_prompt_tasks import _top_level_function, analyze_code  # noqa: E402
from tools.data.synthesize.model_shape.solve_multidim_shape_coverage import (  # noqa: E402
    _bilinear_profile,
    _span_inside,
    _spans_overlap,
)
from tools.data.synthesize.model_shape.solve_shape_coverage import (  # noqa: E402
    AffineProfile,
    ShapeSlot,
    _affine_profiles,
    _shape_slots_with_rejections,
)

CONTRACT_VERSION = "variable_shape_delta_selection_v1"
DELTA_CONTRACT_VERSION = "shape_variable_multislot_delta_selection_v1"
OLD_FAILURE_REASON = "no_strict_two_slot_exactly_one_power_solution"
DEFAULT_EXPECTED_COUNT = 12_318
MAX_SELECTION_WORKERS = 32
DEFAULT_SOURCE_RUN = _REPO_ROOT / "Data/prompt_tvm_v4/shape_solver_multidim_v4_byte_targets_v1/run.full53896"
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "Data/prompt_tvm_v4/shape_solver_variable_multislot_v5/input.recoverable12318"

LEDGER_FIELDS = (
    "output_index",
    "source_row_index",
    "original_source_row_index",
    "parent_uuid",
    "parent_reference_sha256",
    "source_family",
    "variant",
    "target_input_bytes",
    "old_reason",
    "old_slot_count",
    "old_affine_slot_count",
    "old_structural_bilinear_pair_count",
    "recomputed_affine_slot_count",
    "relaxed_structural_pair_count",
    "relaxed_positive_profile_count",
    "positive_profile_slot_count",
    "first_slot_a_id",
    "first_slot_a_kind",
    "first_slot_a_symbol_scope",
    "first_slot_b_id",
    "first_slot_b_kind",
    "first_slot_b_symbol_scope",
    "first_profile_factory_indices",
    "first_profile_changed_occurrences",
    "first_profile_outside_get_inputs_patch_spans",
    "first_profile_constant_bytes",
    "first_profile_x_bytes",
    "first_profile_y_bytes",
    "first_profile_xy_bytes",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nested(value: Mapping[str, Any], path: str, default: Any = None) -> Any:
    current: Any = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _compatible_without_lexical_inside(slot_a: ShapeSlot, slot_b: ShapeSlot) -> bool:
    """The old structural-pair predicate minus only ``_span_inside``."""

    if slot_a.slot_id == slot_b.slot_id:
        return False
    if any(_spans_overlap(left, right) for left in slot_a.patch_spans for right in slot_b.patch_spans):
        return False

    by_factory_a: dict[int, list[int]] = collections.defaultdict(list)
    by_factory_b: dict[int, list[int]] = collections.defaultdict(list)
    for occurrence in slot_a.occurrences:
        by_factory_a[occurrence.factory_index].append(occurrence.axis)
    for occurrence in slot_b.occurrences:
        by_factory_b[occurrence.factory_index].append(occurrence.axis)
    if not by_factory_a or set(by_factory_a) != set(by_factory_b):
        return False
    for factory_index in by_factory_a:
        axes_a = by_factory_a[factory_index]
        axes_b = by_factory_b[factory_index]
        if len(axes_a) != 1 or len(axes_b) != 1 or axes_a[0] == axes_b[0]:
            return False
    return len(slot_a.occurrences) == len(slot_b.occurrences)


def _relaxed_positive_profiles(
    code: str,
    entry_point: str,
    parent_input_bytes: int,
    affine_profiles: Sequence[AffineProfile],
) -> tuple[int, list[Any]]:
    """Return the structural candidate count and every positive witness."""

    structural_count = 0
    positive: list[Any] = []
    for profile_a, profile_b in itertools.combinations(affine_profiles, 2):
        if not _compatible_without_lexical_inside(profile_a.slot, profile_b.slot):
            continue
        structural_count += 1
        try:
            positive.append(
                _bilinear_profile(
                    code,
                    entry_point,
                    parent_input_bytes,
                    profile_a,
                    profile_b,
                )
            )
        except (SyntaxError, TypeError, ValueError):
            continue
    return structural_count, positive


def _evaluate_candidate(
    payload: tuple[int, str, str, int, int],
) -> dict[str, Any] | None:
    """Re-prove one manifest candidate in an isolated CPU worker."""

    (
        source_row_index,
        parent_code,
        entry_point,
        expected_input_bytes,
        expected_affine_slot_count,
    ) = payload
    parent_analysis = analyze_code(parent_code, entry_point)
    if parent_analysis.input_bytes != expected_input_bytes:
        raise ValueError(f"parent_input_bytes_mismatch:{source_row_index}")
    slots, _ = _shape_slots_with_rejections(parent_code, entry_point)
    affine_profiles, _ = _affine_profiles(
        parent_code,
        entry_point,
        parent_analysis.input_bytes,
        slots,
    )
    if len(affine_profiles) != expected_affine_slot_count:
        raise ValueError(f"recomputed_affine_slot_count_mismatch:{source_row_index}")

    structural_count, positive_profiles = _relaxed_positive_profiles(
        parent_code,
        entry_point,
        parent_analysis.input_bytes,
        affine_profiles,
    )
    if not positive_profiles:
        return None

    tree = ast.parse(parent_code)
    get_inputs = _top_level_function(tree, "get_inputs")
    if not isinstance(get_inputs, ast.FunctionDef):
        raise ValueError(f"get_inputs_is_not_sync_function:{source_row_index}")
    first = positive_profiles[0]
    first_slots = (first.slot_a, first.slot_b)
    outside_count = sum(not _span_inside(span, get_inputs) for slot in first_slots for span in slot.patch_spans)
    if outside_count <= 0:
        raise ValueError(f"relaxation_witness_did_not_relax_lexical_inside:{source_row_index}")
    participating_slots = {slot.slot_id for profile in positive_profiles for slot in (profile.slot_a, profile.slot_b)}
    factory_indices = sorted({occurrence.factory_index for occurrence in first.slot_a.occurrences})
    return {
        "source_row_index": source_row_index,
        "recomputed_affine_slot_count": len(affine_profiles),
        "relaxed_structural_pair_count": structural_count,
        "relaxed_positive_profile_count": len(positive_profiles),
        "positive_profile_slot_count": len(participating_slots),
        "first_slot_a_id": first.slot_a.slot_id,
        "first_slot_a_kind": first.slot_a.kind,
        "first_slot_a_symbol_scope": first.slot_a.symbol_scope or "",
        "first_slot_b_id": first.slot_b.slot_id,
        "first_slot_b_kind": first.slot_b.kind,
        "first_slot_b_symbol_scope": first.slot_b.symbol_scope or "",
        "first_profile_factory_indices": ",".join(map(str, factory_indices)),
        "first_profile_changed_occurrences": (len(first.slot_a.occurrences) + len(first.slot_b.occurrences)),
        "first_profile_outside_get_inputs_patch_spans": outside_count,
        "first_profile_constant_bytes": first.constant_bytes,
        "first_profile_x_bytes": first.x_bytes,
        "first_profile_y_bytes": first.y_bytes,
        "first_profile_xy_bytes": first.xy_bytes,
    }


def _manifest_artifact_hash(manifest: Mapping[str, Any], artifact_name: str) -> str:
    hashes = manifest.get("artifact_sha256")
    if not isinstance(hashes, Mapping):
        raise ValueError("source_manifest_missing_artifact_sha256")
    value = hashes.get(artifact_name)
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"source_manifest_missing_{artifact_name}_sha256")
    return value


def _validate_source_provenance(
    source_run: Path,
    source: pa.Table,
    source_manifest: Mapping[str, Any],
    source_selection: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    selected_path = source_run / "selected.parquet"
    selection_path = source_run / "selection.json"
    manifest_path = source_run / "static" / "manifest.json"

    selected_hash = _sha256_file(selected_path)
    if selected_hash != _manifest_artifact_hash(source_manifest, "selected"):
        raise ValueError("source_selected_hash_does_not_match_manifest")
    recorded_source_hash = source_manifest.get("selected_source_sha256")
    if recorded_source_hash != selected_hash:
        raise ValueError("source_selected_hash_does_not_match_selected_source_hash")
    if _sha256_file(selection_path) != _manifest_artifact_hash(source_manifest, "selection"):
        raise ValueError("source_selection_hash_does_not_match_manifest")

    selected_rows = source_manifest.get("selected_rows")
    if selected_rows != source.num_rows:
        raise ValueError(f"source_selected_row_count_mismatch:{selected_rows}:{source.num_rows}")
    if source_selection.get("selected_count") != source.num_rows:
        raise ValueError("source_selection_row_count_mismatch")

    decisions = source_manifest.get("decisions")
    if not isinstance(decisions, list) or len(decisions) != source.num_rows:
        raise ValueError("source_manifest_decisions_are_not_one_per_selected_row")
    selection_rows = source_selection.get("rows")
    if not isinstance(selection_rows, list) or len(selection_rows) != source.num_rows:
        raise ValueError("source_selection_rows_are_not_one_per_selected_row")

    solver_path = Path(__file__).with_name("solve_multidim_shape_coverage.py")
    helper_path = Path(__file__).with_name("solve_shape_coverage.py")
    if _sha256_file(solver_path) != source_manifest.get("solver_source_sha256"):
        raise ValueError("old_solver_source_hash_drifted_from_source_manifest")
    if _sha256_file(helper_path) != source_manifest.get("v3_helper_source_sha256"):
        raise ValueError("old_helper_source_hash_drifted_from_source_manifest")
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    return decisions


def prepare_variable_shape_delta(
    source_run: Path,
    output_dir: Path,
    *,
    expected_count: int = DEFAULT_EXPECTED_COUNT,
) -> dict[str, Any]:
    source_run = source_run.resolve()
    output_dir = output_dir.resolve()
    selected_path = source_run / "selected.parquet"
    selection_path = source_run / "selection.json"
    manifest_path = source_run / "static" / "manifest.json"
    for required in (selected_path, selection_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    if expected_count <= 0:
        raise ValueError("expected_count_must_be_positive")

    source = pq.read_table(selected_path)
    with selection_path.open(encoding="utf-8") as handle:
        source_selection = json.load(handle)
    with manifest_path.open(encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    if not isinstance(source_selection, dict) or not isinstance(source_manifest, dict):
        raise ValueError("source_manifests_must_be_objects")
    decisions = _validate_source_provenance(source_run, source, source_manifest, source_selection)

    extra = source.column("extra_info").combine_chunks()
    reward_model = source.column("reward_model").combine_chunks()
    uuids = extra.field("uuid")
    entry_points = extra.field("entry_point")
    references = reward_model.field("ground_truth")
    source_selection_rows = source_selection["rows"]

    candidate_count = 0
    candidate_payloads: list[tuple[int, str, str, int, int]] = []
    selected_indices: list[int] = []
    ledger: list[dict[str, Any]] = []
    relaxed_rejection_counts: collections.Counter[str] = collections.Counter()

    for source_row_index, decision in enumerate(decisions):
        if not isinstance(decision, Mapping):
            raise ValueError(f"decision_is_not_an_object:{source_row_index}")
        if decision.get("source_row_index") != source_row_index:
            raise ValueError(f"decision_source_index_mismatch:{source_row_index}")

        parent_uuid = uuids[source_row_index].as_py()
        parent_code = references[source_row_index].as_py()
        entry_point = entry_points[source_row_index].as_py()
        if not isinstance(parent_uuid, str) or not parent_uuid:
            raise ValueError(f"source_uuid_missing:{source_row_index}")
        if not isinstance(parent_code, str) or not parent_code:
            raise ValueError(f"source_reference_missing:{source_row_index}")
        if not isinstance(entry_point, str) or not entry_point:
            raise ValueError(f"source_entry_point_missing:{source_row_index}")
        reference_hash = _sha256_bytes(parent_code.encode("utf-8"))
        if decision.get("parent_uuid") != parent_uuid:
            raise ValueError(f"decision_uuid_mismatch:{source_row_index}")
        if decision.get("parent_reference_sha256") != reference_hash:
            raise ValueError(f"decision_reference_hash_mismatch:{source_row_index}")

        source_selection_row = source_selection_rows[source_row_index]
        if not isinstance(source_selection_row, Mapping):
            raise ValueError(f"selection_row_is_not_an_object:{source_row_index}")
        if source_selection_row.get("uuid") != parent_uuid:
            raise ValueError(f"selection_uuid_mismatch:{source_row_index}")
        if source_selection_row.get("reference_sha256") != reference_hash:
            raise ValueError(f"selection_reference_hash_mismatch:{source_row_index}")

        reason = str(decision.get("reason", ""))
        if not (
            reason == OLD_FAILURE_REASON
            and type(decision.get("affine_slot_count")) is int
            and decision["affine_slot_count"] >= 2
            and decision.get("structural_bilinear_pair_count") == 0
        ):
            continue
        candidate_count += 1
        input_bytes_before = decision.get("input_bytes_before")
        if type(input_bytes_before) is not int or input_bytes_before <= 0:
            raise ValueError(f"decision_input_bytes_missing:{source_row_index}")
        candidate_payloads.append(
            (
                source_row_index,
                parent_code,
                entry_point,
                input_bytes_before,
                decision["affine_slot_count"],
            )
        )

    worker_count = min(
        MAX_SELECTION_WORKERS,
        len(candidate_payloads),
        len(os.sched_getaffinity(0)),
    )
    if worker_count <= 0:
        raise ValueError("no_old_decision_candidates")
    fork_context = multiprocessing.get_context("fork")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=fork_context,
    ) as executor:
        results = executor.map(
            _evaluate_candidate,
            candidate_payloads,
            chunksize=16,
        )
        for payload, evidence in zip(candidate_payloads, results, strict=True):
            source_row_index = payload[0]
            if evidence is None:
                relaxed_rejection_counts["no_positive_profile"] += 1
                continue
            if evidence["source_row_index"] != source_row_index:
                raise AssertionError("worker_source_row_index_mismatch")
            decision = decisions[source_row_index]
            selection_record = source_selection_rows[source_row_index]
            parent_uuid = uuids[source_row_index].as_py()
            reference_hash = decision["parent_reference_sha256"]
            selected_indices.append(source_row_index)
            ledger.append(
                {
                    "output_index": len(selected_indices) - 1,
                    "source_row_index": source_row_index,
                    "original_source_row_index": selection_record.get("row_index", ""),
                    "parent_uuid": parent_uuid,
                    "parent_reference_sha256": reference_hash,
                    "source_family": selection_record.get("source_family", ""),
                    "variant": decision["variant"],
                    "target_input_bytes": decision["target_input_bytes"],
                    "old_reason": decision["reason"],
                    "old_slot_count": decision.get("slot_count", ""),
                    "old_affine_slot_count": decision["affine_slot_count"],
                    "old_structural_bilinear_pair_count": decision["structural_bilinear_pair_count"],
                    **evidence,
                }
            )

    if selected_indices != sorted(selected_indices):
        raise AssertionError("selected_source_indices_are_not_sorted")
    if len(selected_indices) != expected_count:
        raise ValueError(f"selected_count_mismatch:expected={expected_count}:" f"observed={len(selected_indices)}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        output_selected = temporary_dir / "selected.parquet"
        output_ledger = temporary_dir / "selection_ledger.tsv"
        output_selection = temporary_dir / "selection.json"

        selected = source.take(pa.array(selected_indices, type=pa.int64()))
        pq.write_table(selected, output_selected, compression="zstd")
        with output_ledger.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=LEDGER_FIELDS,
                dialect="excel-tab",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(ledger)

        selected_hash = _sha256_file(output_selected)
        ledger_hash = _sha256_file(output_ledger)
        filtered_selection_rows = []
        for index in selected_indices:
            decision = decisions[index]
            source_row = dict(source_selection_rows[index])
            source_row["variable_shape_delta"] = {
                "parent_uuid": decision["parent_uuid"],
                "parent_reference_sha256": decision["parent_reference_sha256"],
                "prior_full_source_row_index": index,
                "variant": decision["variant"],
                "target_input_bytes": decision["target_input_bytes"],
            }
            filtered_selection_rows.append(source_row)
        selection = {
            "contract_version": CONTRACT_VERSION,
            "delta_contract_version": DELTA_CONTRACT_VERSION,
            "selector_source_path": str(Path(__file__).resolve()),
            "selector_source_sha256": _sha256_file(Path(__file__)),
            "selection_boundary": "selection_only_no_children_generated",
            "selection_criteria": {
                "old_reason": OLD_FAILURE_REASON,
                "old_affine_slot_count_minimum": 2,
                "old_structural_bilinear_pair_count": 0,
                "relaxation": "remove_only_patch_span_physically_inside_get_inputs",
                "retained_pair_rules": [
                    "disjoint_slot_patch_spans",
                    "same_nonempty_direct_input_factory_set",
                    "one_occurrence_per_slot_per_factory",
                    "distinct_axes_per_factory",
                    "equal_occurrence_counts",
                    "positive_exact_bilinear_storage_profile",
                ],
                "power_of_two_value_constraint": None,
                "exact_slot_count_constraint": None,
            },
            "source_run": str(source_run),
            "source_selected": str(selected_path),
            "source_selected_sha256": _sha256_file(selected_path),
            "source_selection": str(selection_path),
            "source_selection_sha256": _sha256_file(selection_path),
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": _sha256_file(manifest_path),
            "source_selected_count": source.num_rows,
            "old_decision_candidate_count": candidate_count,
            "selection_worker_count": worker_count,
            "relaxed_rejection_counts": dict(sorted(relaxed_rejection_counts.items())),
            "expected_count": expected_count,
            "selected_count": len(selected_indices),
            "order": "source_row_index_ascending",
            "selected_path": str(output_dir / "selected.parquet"),
            "selected_sha256": selected_hash,
            "ledger_path": str(output_dir / "selection_ledger.tsv"),
            "ledger_sha256": ledger_hash,
            "rows": filtered_selection_rows,
        }
        output_selection.write_text(
            json.dumps(selection, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_dir, output_dir)
        return selection
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output_dir",
        nargs="?",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"new output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--source-run",
        type=Path,
        default=DEFAULT_SOURCE_RUN,
        help=f"completed old full-run directory (default: {DEFAULT_SOURCE_RUN})",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=DEFAULT_EXPECTED_COUNT,
        help=f"fail unless exactly this many parents are selected (default: {DEFAULT_EXPECTED_COUNT})",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    selection = prepare_variable_shape_delta(
        args.source_run,
        args.output_dir,
        expected_count=args.expected_count,
    )
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir.resolve()),
                "selected_count": selection["selected_count"],
                "selected_sha256": selection["selected_sha256"],
                "ledger_sha256": selection["ledger_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
