#!/usr/bin/env python3
"""Materialize the fixed 240-row H20 canary from a frozen 13k registry.

This program is selection-only: it never re-renders candidates or starts CUDA.
Five rows are selected per each
of the 48 registered templates.  Three are first/middle/last variant anchors;
the other two are a deterministic global marginal-coverage fill.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

ROWS = 240
PER_TEMPLATE = 5
SEED = "operator_structure_10k_h20_canary_v1"
CONTRACT = "operator_structure_10k_h20_canary_selection_v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != 13_000 or not all(isinstance(record, dict) for record in records):
        raise ValueError("canary requires exactly 13k manifest records")
    return records


def _uuid(row: Mapping[str, Any]) -> str:
    extra = row.get("extra_info")
    value = extra.get("uuid") if isinstance(extra, Mapping) else None
    if not isinstance(value, str) or not value:
        raise ValueError("candidate lacks extra_info.uuid")
    return value


def _code(row: Mapping[str, Any]) -> str:
    reward = row.get("reward_model")
    value = reward.get("ground_truth") if isinstance(reward, Mapping) else None
    if not isinstance(value, str) or not value:
        raise ValueError("candidate lacks reward_model.ground_truth")
    return value


def _stable_key(record: Mapping[str, Any]) -> str:
    return _sha256_bytes(f"{SEED}|{record['uuid']}|{record['reference_sha256']}".encode())


def _categories(record: Mapping[str, Any]) -> set[str]:
    labels = record["coverage_labels"]
    categories = {
        f"family:{record['primary_family']}",
        f"rank:{labels['input_rank']}",
        f"input_mode:{labels['input_mode']}",
        f"complexity:{record['complexity_bucket']}",
        f"mode_behavior:{record['mode_behavior']}",
    }
    if labels["input_arity"] >= 2:
        categories.add("special:multi_input")
    if record["top_level_class_count"] >= 3:
        categories.add("special:multiclass")
    if record["mode_behavior"] == "train_stateful":
        categories.add("special:stateful")
    return categories


def _load_registry(
    candidates_path: Path, manifest_path: Path, generator: Any
) -> tuple[pa.Table, list[dict[str, Any]], list[dict[str, Any]]]:
    table = pq.read_table(candidates_path)
    rows, manifests = table.to_pylist(), _read_manifest(manifest_path)
    if len(rows) != 13_000:
        raise ValueError("canary requires exactly 13k candidate rows")
    generator_sha = _sha256_file(Path(generator.__file__).resolve())
    templates = {item.template_id: item for item in generator.TEMPLATES}
    records = []
    for index, (row, manifest) in enumerate(zip(rows, manifests, strict=True)):
        uuid, code = _uuid(row), _code(row)
        labels = manifest.get("coverage_labels")
        if (
            manifest.get("uuid") != uuid
            or manifest.get("candidate_row_index") != index
            or manifest.get("generator_source_sha256") != generator_sha
            or manifest.get("reference_sha256") != _sha256_bytes(code.encode())
            or manifest.get("static_status") != "passed"
            or not isinstance(labels, Mapping)
        ):
            raise ValueError(f"static candidate/manifest binding mismatch:{index}")
        template = manifest.get("template_id")
        if template not in templates or manifest.get("primary_family") != templates[template].family:
            raise ValueError(f"unknown template/family:{index}")
        required_labels = {"input_rank", "input_mode", "input_arity"}
        if not required_labels.issubset(labels) or manifest.get("mode_behavior") not in {
            "stateless",
            "train_stateful",
            "recurrent_state",
        }:
            raise ValueError(f"canary label contract mismatch:{index}")
        complexity = (
            manifest.get("static_proof", {}).get("complexity")
            if isinstance(manifest.get("static_proof"), Mapping)
            else None
        )
        calls = complexity.get("forward_call_count") if isinstance(complexity, Mapping) else None
        bucket = (
            "10-14"
            if isinstance(calls, int) and 10 <= calls <= 14
            else (
                "15-24"
                if isinstance(calls, int) and 15 <= calls <= 24
                else "25+" if isinstance(calls, int) and calls >= 25 else None
            )
        )
        if bucket is None:
            raise ValueError(f"invalid forward-call complexity:{index}")
        records.append(
            {
                "row_index": index,
                "uuid": uuid,
                "reference_sha256": manifest["reference_sha256"],
                "template_id": template,
                "template_variant": manifest.get("template_variant"),
                "primary_family": manifest["primary_family"],
                "coverage_labels": dict(labels),
                "mode_behavior": manifest["mode_behavior"],
                "top_level_class_count": manifest.get("static_proof", {}).get("top_level_class_count"),
                "complexity_bucket": bucket,
                "forward_calls": calls,
                "manifest": manifest,
            }
        )
    if {record["template_id"] for record in records} != set(templates):
        raise ValueError("frozen static registry has incomplete template coverage")
    return table, manifests, records


def _anchors(group: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(group, key=lambda record: (record["template_variant"], _stable_key(record)))
    positions = (0, len(ordered) // 2, len(ordered) - 1)
    return [ordered[position] for position in positions]


def _op_ids(record: Mapping[str, Any]) -> set[str]:
    declared = record["manifest"].get("declared_ops")
    if not isinstance(declared, list):
        return set()
    return {item.get("op_id") for item in declared if isinstance(item, Mapping) and isinstance(item.get("op_id"), str)}


def _intervention_keys(record: Mapping[str, Any]) -> set[str]:
    proof = record["manifest"].get("static_proof")
    semantic = proof.get("semantic_intervention_proof") if isinstance(proof, Mapping) else None
    return set(semantic) if isinstance(semantic, Mapping) else set()


def _required_cell_coverage(records: Sequence[dict[str, Any]], selected: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Confirm the fixed 240 contains every changed v2 execution cell."""

    by_template: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for record in selected:
        by_template[str(record["template_id"])].append(record)
    all_templates = {str(record["template_id"]) for record in records}
    missing: dict[str, list[str]] = {}

    def require_templates(name: str, templates: set[str], predicate: Any) -> None:
        absent = sorted(
            template for template in templates if not any(predicate(record) for record in by_template[template])
        )
        if absent:
            missing[name] = absent

    lf_templates = {
        str(record["template_id"]) for record in records if record["primary_family"] == "layout_fusion_graph"
    }
    require_templates(
        "layout", lf_templates, lambda record: "layout" in _op_ids(record) and "layout" in _intervention_keys(record)
    )
    rp_templates = {
        str(record["template_id"]) for record in records if record["primary_family"] == "reduction_pool_graph"
    }
    require_templates(
        "pooling_rank4",
        rp_templates,
        lambda record: record["coverage_labels"]["input_rank"] == 4
        and "pooling" in _op_ids(record)
        and "pooling" in _intervention_keys(record),
    )
    require_templates(
        "reduction_rank2",
        rp_templates,
        lambda record: record["coverage_labels"]["input_rank"] == 2
        and "reduction" in _op_ids(record)
        and "reduction" in _intervention_keys(record),
    )
    require_templates(
        "reduction_rank3",
        rp_templates,
        lambda record: record["coverage_labels"]["input_rank"] == 3
        and "reduction" in _op_ids(record)
        and "reduction" in _intervention_keys(record),
    )
    matrix_templates = {
        str(record["template_id"])
        for record in records
        if record["primary_family"] in {"matmul_linear_graph", "modular_multiclass_graph"}
    }
    require_templates(
        "matrix_rank2",
        matrix_templates,
        lambda record: record["coverage_labels"]["input_rank"] == 2 and "matrix_product" in _op_ids(record),
    )
    require_templates(
        "matrix_rank3",
        matrix_templates,
        lambda record: record["coverage_labels"]["input_rank"] == 3 and "matrix_product" in _op_ids(record),
    )
    expected_signed = {
        (str(record["primary_family"]), int(record["coverage_labels"]["input_rank"]))
        for record in records
        if record["coverage_labels"]["input_mode"] == "signed"
    }
    observed_signed = {
        (str(record["primary_family"]), int(record["coverage_labels"]["input_rank"]))
        for record in selected
        if record["coverage_labels"]["input_mode"] == "signed"
    }
    signed_missing = sorted(f"{family}:rank{rank}" for family, rank in expected_signed - observed_signed)
    if signed_missing:
        missing["signed_family_rank"] = signed_missing
    if missing:
        raise ValueError(f"canary misses required v2 execution cells:{missing}")
    return {
        "layout_templates": len(lf_templates),
        "pooling_rank4_templates": len(rp_templates),
        "reduction_rank2_templates": len(rp_templates),
        "reduction_rank3_templates": len(rp_templates),
        "matrix_rank2_templates": len(matrix_templates),
        "matrix_rank3_templates": len(matrix_templates),
        "signed_family_rank_cells": sorted(f"{family}:rank{rank}" for family, rank in expected_signed),
        "all_registry_templates": len(all_templates),
    }


def select(records: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_template: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    all_categories: set[str] = set()
    for record in records:
        by_template[record["template_id"]].append(record)
        all_categories.update(_categories(record))
    selected: dict[str, dict[str, Any]] = {}
    reasons: dict[str, list[str]] = collections.defaultdict(list)
    for _template, group in sorted(by_template.items()):
        for name, record in zip(("anchor:first", "anchor:middle", "anchor:last"), _anchors(group), strict=True):
            selected[record["uuid"]] = record
            reasons[record["uuid"]].append(name)
    # Each template gets precisely two diversity fills.  Scoring unseen
    # marginal categories first avoids a literal-only selection while SHA is
    # the sole tie-breaker.
    covered = set().union(*(_categories(record) for record in selected.values()))
    for template, group in sorted(by_template.items()):
        for fill in range(2):
            choices = [record for record in group if record["uuid"] not in selected]
            if not choices:
                raise ValueError(f"template lacks five distinct variants:{template}")
            record = min(choices, key=lambda item: (-len(_categories(item) - covered), _stable_key(item)))
            selected[record["uuid"]] = record
            reasons[record["uuid"]].append(f"marginal_fill:{fill + 1}")
            covered.update(_categories(record))
    values = list(selected.values())
    per_template = collections.Counter(record["template_id"] for record in values)
    if len(values) != ROWS or set(per_template.values()) != {PER_TEMPLATE}:
        raise ValueError("canary selection is not exactly five rows per template")
    selected_categories = set().union(*(_categories(record) for record in values))
    missing = sorted(all_categories - selected_categories)
    if missing:
        raise ValueError(f"canary misses supported marginal categories:{missing}")
    materialized = [
        {**record, "selection_reasons": reasons[record["uuid"]]}
        for record in sorted(values, key=lambda record: record["row_index"])
    ]
    required_cells = _required_cell_coverage(records, materialized)
    return materialized, {
        "per_template": dict(sorted(per_template.items())),
        "supported_categories": sorted(all_categories),
        "selected_categories": sorted(selected_categories),
        "required_v2_execution_cells": required_cells,
        "selection_order": "first/middle/last variant anchors; two template-local unseen-marginal fills; SHA tie-break",
    }


def _review_contract() -> dict[str, Any]:
    return {
        "manual_review_rows": 60,
        "strata": [
            {"name": "family", "coverage": "5 families, 12 assignments each"},
            {"name": "input_mode", "coverage": "all 8 modes, at least 4 assignments each"},
            {"name": "complexity", "coverage": "10-14, 15-24, 25+ each at least 12 assignments"},
            {"name": "special", "coverage": "train_stateful, multiclass, multi_input each at least 8 assignments"},
        ],
        "rating_schema": {
            "A": "语义、输入可达性和 runtime 证据闭合",
            "B": "可用但有明确代表性边界",
            "C": "candidate 缺陷，如死输入/helper、伪复杂度、虚假标签或无意义输入",
            "D": "reference、liveness 或审查证据未闭合",
        },
    }


def _failure_attribution_schema() -> dict[str, Any]:
    return {
        "columns": [
            "uuid",
            "template_id",
            "primary_family",
            "phase",
            "status",
            "cause",
            "evidence_path",
            "disposition",
        ],
        "phases": {
            "reference": "fresh H20 five-trial paired reference",
            "liveness": "three-trial returned-output operator provenance",
            "manual": "fixed stratified semantic review",
        },
        "causes": {
            "candidate_semantic": "C",
            "input_or_shape": "C",
            "operator_provenance": "C",
            "reference_harness_or_environment": "D",
            "runtime_infrastructure": "D",
            "insufficient_evidence": "D",
        },
        "rule": "A failing row is never materialized as passed; preserve the raw record and attribute one primary cause with evidence path.",
    }


def materialize(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output must be empty:{output}")
    generator = importlib.import_module(args.generator_module)
    table, _, records = _load_registry(args.candidates.resolve(), args.manifest.resolve(), generator)
    selected, coverage = select(records)
    output.mkdir(parents=True, exist_ok=True)
    indices = pa.array([record["row_index"] for record in selected], type=pa.int64())
    pq.write_table(table.take(indices), output / "canary.parquet", compression="zstd")
    uuid_file = output / "canary_uuids.txt"
    uuid_file.write_text("".join(f"{record['uuid']}\n" for record in selected), encoding="utf-8")
    with (output / "canary.manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in selected:
            projected = dict(record["manifest"])
            projected.update(
                {
                    "canary_contract": CONTRACT,
                    "canary_selection_reasons": record["selection_reasons"],
                    "training_approved": False,
                }
            )
            handle.write(_canonical(projected) + "\n")
    summary = {
        "contract": CONTRACT,
        "seed": SEED,
        "selected_rows": len(selected),
        "templates": len(coverage["per_template"]),
        "source": {
            "candidates": {"path": str(args.candidates.resolve()), "sha256": _sha256_file(args.candidates.resolve())},
            "manifest": {"path": str(args.manifest.resolve()), "sha256": _sha256_file(args.manifest.resolve())},
            "generator": {
                "module": generator.__name__,
                "path": str(Path(generator.__file__).resolve()),
                "sha256": _sha256_file(Path(generator.__file__).resolve()),
            },
        },
        "artifacts": {"uuid_allowlist": {"path": str(uuid_file), "sha256": _sha256_file(uuid_file)}},
        "coverage": coverage,
        "review": _review_contract(),
        "failure_attribution": _failure_attribution_schema(),
        "review_only": True,
        "training_approved": False,
    }
    (output / "canary_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "failure_attribution_schema.json").write_text(
        json.dumps(_failure_attribution_schema(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--generator-module",
        default="tools.data.synthesize.canary.operator_structure_10k_method.generate_operator_structure_10k",
    )
    args = parser.parse_args(argv)
    print(json.dumps(materialize(args), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
