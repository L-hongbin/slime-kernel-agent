#!/usr/bin/env python3
"""Select every unprocessed measurable failure from the exact-two shape lane.

The earlier variable-multislot bridge selected only failures recoverable after
relaxing one lexical pair rule.  This selector deliberately performs no shape
group prefilter.  It retains every old decision whose unchanged parent passed
FakeTensor, whose exact-two solver produced no accepted child, and whose parent
UUID was not already sent through the first variable-multislot lane.  The
downstream solver therefore performs the general K2--K5 enumeration over the
complete remaining measurable failure set.

This tool only selects parents and preserves their original deterministic
Medium/Large targets.  It never generates a child.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from tools.data.synthesize.prepare_variable_shape_delta import _sha256_bytes, _sha256_file, _validate_source_provenance

CONTRACT_VERSION = "variable_shape_all_measurable_remaining_selection_v1"
DELTA_CONTRACT_VERSION = "shape_variable_multislot_delta_selection_v1"
DEFAULT_EXPECTED_COUNT = 22_566

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SELECTOR_DEPENDENCY = Path(__file__).with_name("prepare_variable_shape_delta.py")
DEFAULT_SOURCE_RUN = _REPO_ROOT / "Data/prompt_tvm_v4/shape_solver_multidim_v4_byte_targets_v1/run.full53896"
DEFAULT_PRIOR_SELECTION = (
    _REPO_ROOT / "Data/prompt_tvm_v4/shape_solver_variable_multislot_v5/input.recoverable12318/selection.json"
)
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "Data/prompt_tvm_v4/shape_solver_variable_multislot_v8/input.remaining22566"

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
    "old_attempt_count",
)


def _git_commit(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _load_prior_parent_uuids(path: Path) -> tuple[set[str], Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError("prior_selection_manifest_must_be_an_object")
    if loaded.get("delta_contract_version") != DELTA_CONTRACT_VERSION:
        raise ValueError("prior_selection_has_wrong_delta_contract")
    rows = loaded.get("rows")
    if not isinstance(rows, list):
        raise ValueError("prior_selection_rows_must_be_a_list")
    parent_uuids: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"prior_selection_row_is_not_an_object:{index}")
        delta = row.get("variable_shape_delta")
        if not isinstance(delta, Mapping):
            raise ValueError(f"prior_selection_row_lacks_delta:{index}")
        parent_uuid = delta.get("parent_uuid")
        if not isinstance(parent_uuid, str) or not parent_uuid:
            raise ValueError(f"prior_selection_row_lacks_parent_uuid:{index}")
        parent_uuids.append(parent_uuid)
    if len(set(parent_uuids)) != len(parent_uuids):
        raise ValueError("prior_selection_contains_duplicate_parent_uuid")
    if loaded.get("selected_count") != len(parent_uuids):
        raise ValueError("prior_selection_count_mismatch")
    return set(parent_uuids), loaded


def prepare_remaining_selection(
    source_run: Path,
    prior_selection_path: Path,
    output_dir: Path,
    *,
    expected_count: int = DEFAULT_EXPECTED_COUNT,
) -> dict[str, Any]:
    source_run = source_run.resolve()
    prior_selection_path = prior_selection_path.resolve()
    output_dir = output_dir.resolve()
    selected_path = source_run / "selected.parquet"
    selection_path = source_run / "selection.json"
    manifest_path = source_run / "static/manifest.json"
    for required in (selected_path, selection_path, manifest_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output_dir}")
    if expected_count <= 0:
        raise ValueError("expected_count_must_be_positive")

    source = pq.read_table(selected_path)
    source_selection = json.loads(selection_path.read_text(encoding="utf-8"))
    source_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(source_selection, Mapping) or not isinstance(source_manifest, Mapping):
        raise ValueError("source_manifests_must_be_objects")
    decisions = _validate_source_provenance(source_run, source, source_manifest, source_selection)
    source_selection_rows = source_selection["rows"]
    prior_parent_uuids, prior_selection = _load_prior_parent_uuids(prior_selection_path)

    extra = source.column("extra_info").combine_chunks()
    reward_model = source.column("reward_model").combine_chunks()
    uuids = extra.field("uuid")
    references = reward_model.field("ground_truth")

    selected_indices: list[int] = []
    ledger: list[dict[str, Any]] = []
    old_reason_counts: collections.Counter[str] = collections.Counter()
    accounting: collections.Counter[str] = collections.Counter()

    for source_row_index, decision in enumerate(decisions):
        if not isinstance(decision, Mapping):
            raise ValueError(f"decision_is_not_an_object:{source_row_index}")
        if decision.get("source_row_index") != source_row_index:
            raise ValueError(f"decision_source_index_mismatch:{source_row_index}")
        selection_row = source_selection_rows[source_row_index]
        if not isinstance(selection_row, Mapping):
            raise ValueError(f"selection_row_is_not_an_object:{source_row_index}")

        parent_uuid = uuids[source_row_index].as_py()
        parent_code = references[source_row_index].as_py()
        if not isinstance(parent_uuid, str) or not parent_uuid:
            raise ValueError(f"source_uuid_missing:{source_row_index}")
        if not isinstance(parent_code, str) or not parent_code:
            raise ValueError(f"source_reference_missing:{source_row_index}")
        reference_hash = _sha256_bytes(parent_code.encode("utf-8"))
        if decision.get("parent_uuid") != parent_uuid:
            raise ValueError(f"decision_uuid_mismatch:{source_row_index}")
        if decision.get("parent_reference_sha256") != reference_hash:
            raise ValueError(f"decision_reference_hash_mismatch:{source_row_index}")
        if selection_row.get("uuid") != parent_uuid:
            raise ValueError(f"selection_uuid_mismatch:{source_row_index}")
        if selection_row.get("reference_sha256") != reference_hash:
            raise ValueError(f"selection_reference_mismatch:{source_row_index}")

        accounting["source_decisions"] += 1
        if decision.get("accepted") is True:
            accounting["old_static_accepted"] += 1
            continue
        parent_fake = decision.get("parent_fake_gate")
        if not isinstance(parent_fake, Mapping) or parent_fake.get("status") != "passed":
            accounting["old_parent_fake_not_passed"] += 1
            continue
        accounting["old_parent_fake_passed_unaccepted"] += 1
        if parent_uuid in prior_parent_uuids:
            accounting["excluded_prior_variable_parent"] += 1
            continue

        reason = decision.get("reason")
        if not isinstance(reason, str) or not reason:
            raise ValueError(f"unaccepted_decision_lacks_reason:{source_row_index}")
        old_reason_counts[reason] += 1
        selected_indices.append(source_row_index)
        ledger.append(
            {
                "output_index": len(selected_indices) - 1,
                "source_row_index": source_row_index,
                "original_source_row_index": selection_row.get("row_index", ""),
                "parent_uuid": parent_uuid,
                "parent_reference_sha256": reference_hash,
                "source_family": selection_row.get("source_family", ""),
                "variant": decision["variant"],
                "target_input_bytes": decision["target_input_bytes"],
                "old_reason": reason,
                "old_slot_count": decision.get("slot_count", ""),
                "old_affine_slot_count": decision.get("affine_slot_count", ""),
                "old_structural_bilinear_pair_count": decision.get("structural_bilinear_pair_count", ""),
                "old_attempt_count": len(decision.get("attempts", [])),
            }
        )

    accounting["selected_remaining"] = len(selected_indices)
    if selected_indices != sorted(selected_indices):
        raise AssertionError("selected_source_indices_are_not_sorted")
    if len(selected_indices) != expected_count:
        raise ValueError(f"selected_count_mismatch:expected={expected_count}:" f"observed={len(selected_indices)}")
    if set(uuids[index].as_py() for index in selected_indices) & prior_parent_uuids:
        raise AssertionError("remaining_selection_overlaps_prior_variable_selection")

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
            source_row["broad_failure_recovery"] = {
                "old_reason": decision["reason"],
                "old_slot_count": decision.get("slot_count"),
                "old_affine_slot_count": decision.get("affine_slot_count"),
                "old_structural_bilinear_pair_count": decision.get("structural_bilinear_pair_count"),
                "old_attempt_count": len(decision.get("attempts", [])),
            }
            filtered_selection_rows.append(source_row)

        selection = {
            "contract_version": CONTRACT_VERSION,
            "delta_contract_version": DELTA_CONTRACT_VERSION,
            "git_commit": _git_commit(_REPO_ROOT),
            "selector_source_path": str(Path(__file__).resolve()),
            "selector_source_sha256": _sha256_file(Path(__file__)),
            "selector_dependency_source_path": str(_SELECTOR_DEPENDENCY.resolve()),
            "selector_dependency_source_sha256": _sha256_file(_SELECTOR_DEPENDENCY),
            "selection_boundary": "selection_only_no_children_generated",
            "selection_criteria": {
                "old_exact_two_accepted": False,
                "old_parent_fake_gate_status": "passed",
                "exclude_prior_variable_selection": True,
                "shape_group_prefilter": None,
                "downstream_search": "general_variable_K2_to_K5",
                "targets": "reuse_prior_exact_two_variant_and_target",
            },
            "source_run": str(source_run),
            "source_selected": str(selected_path),
            "source_selected_sha256": _sha256_file(selected_path),
            "source_selection": str(selection_path),
            "source_selection_sha256": _sha256_file(selection_path),
            "source_manifest": str(manifest_path),
            "source_manifest_sha256": _sha256_file(manifest_path),
            "source_selected_count": source.num_rows,
            "prior_variable_selection": str(prior_selection_path),
            "prior_variable_selection_sha256": _sha256_file(prior_selection_path),
            "prior_variable_selection_contract": prior_selection.get("contract_version"),
            "prior_variable_selected_count": len(prior_parent_uuids),
            "accounting": dict(sorted(accounting.items())),
            "old_reason_counts": dict(sorted(old_reason_counts.items())),
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
    )
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--prior-selection", type=Path, default=DEFAULT_PRIOR_SELECTION)
    parser.add_argument("--expected-count", type=int, default=DEFAULT_EXPECTED_COUNT)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    selection = prepare_remaining_selection(
        args.source_run,
        args.prior_selection,
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
                "accounting": selection["accounting"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
