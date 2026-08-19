#!/usr/bin/env python3
"""Select terminal serial children plus a coverage-weighted canonical fallback sample."""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import pyarrow as pa
import pyarrow.parquet as pq

from tools.data.cleaning.complexity import extract_operator_signature
from tools.data.synthesize.low_level_operator_families import token_family_cells

REPO_ROOT = Path(__file__).resolve().parents[4]
SERIAL_CONTRACT = "serial_augmentation_fallback_v1"
SHAPE_CONTRACT = "shape_runtime_single_changed_tensor_coverage_resample_v5"
INTERMEDIATES = REPO_ROOT / "Data/prompt_tvm_v4/intermediate_artifacts"
DEFAULT_CANONICAL = INTERMEDIATES / "train.review.parquet"
DEFAULT_BUILD_MANIFEST = INTERMEDIATES / "build_manifest.json"
DEFAULT_SHAPE_DIR = INTERMEDIATES / "shape_runtime_coverage_resample_v5/run.low128k_final"
DEFAULT_SERIAL_ROOT = INTERMEDIATES / "serial_augmentation_v1"


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reference_code(row: Mapping[str, Any]) -> str:
    reward = row.get("reward_model")
    code = reward.get("ground_truth") if isinstance(reward, Mapping) else None
    if not isinstance(code, str) or not code:
        raise ValueError("row lacks reward_model.ground_truth")
    return code


def _uuid(row: Mapping[str, Any]) -> str:
    extra = row.get("extra_info")
    value = extra.get("uuid") if isinstance(extra, Mapping) else None
    if not isinstance(value, str) or not value:
        raise ValueError("row lacks extra_info.uuid")
    return value


def _v4(row: Mapping[str, Any]) -> Mapping[str, Any]:
    extra = row.get("extra_info")
    value = extra.get("v4") if isinstance(extra, Mapping) else None
    if not isinstance(value, Mapping):
        raise ValueError(f"row lacks extra_info.v4: {_uuid(row)}")
    return value


def _reference_sha256(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(_reference_code(row).encode("utf-8")).hexdigest()


def _normalized_ast_sha256(row: Mapping[str, Any]) -> str:
    normalized = ast.dump(ast.parse(_reference_code(row)), annotate_fields=True, include_attributes=False)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _artifact(path: Path, *, rows: int | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"path": str(path.resolve()), "sha256": _file_sha256(path)}
    if rows is not None:
        item["rows"] = rows
    return item


def _staged_artifact(path: Path, final: Path, *, rows: int | None = None) -> dict[str, Any]:
    item = _artifact(path, rows=rows)
    item["path"] = str(final.resolve())
    return item


def _jsonl_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"non-object JSONL row: {path}:{line_number}")
            yield value


def _parquet_rows(path: Path, *, columns: Sequence[str] | None = None) -> Iterator[dict[str, Any]]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=16, columns=columns, use_threads=False):
        yield from pa.Table.from_batches([batch]).to_pylist()


def _declared_artifact(summary: Mapping[str, Any], key: str, path: Path) -> None:
    artifacts = summary.get("artifacts")
    declared = artifacts.get(key) if isinstance(artifacts, Mapping) else None
    if not isinstance(declared, Mapping):
        raise ValueError(f"summary lacks artifacts.{key}: {path.parent}")
    if declared.get("sha256") != _file_sha256(path):
        raise ValueError(f"artifact SHA mismatch: {key}")


def _validate_authorities(
    *,
    canonical: Path,
    build_manifest: Path,
    shape_dir: Path,
    serial_root: Path,
) -> dict[str, Any]:
    build = json.loads(build_manifest.read_text(encoding="utf-8"))
    declared = (build.get("outputs") or {}).get("train.review.parquet")
    if (
        build.get("contract_version") != "prompt_tvm_v4_union_review_v1"
        or not isinstance(declared, Mapping)
        or declared.get("rows") != 64_315
        or declared.get("sha256") != _file_sha256(canonical)
    ):
        raise ValueError("canonical parent build authority mismatch")

    shape_summary_path = shape_dir / "summary.json"
    shape_summary = json.loads(shape_summary_path.read_text(encoding="utf-8"))
    shape_selected = shape_dir / "selected.parquet"
    shape_manifest = shape_dir / "manifest.jsonl"
    if (
        shape_summary.get("contract_version") != SHAPE_CONTRACT
        or shape_summary.get("training_approved") is not False
        or shape_summary.get("selected_rows") != 31_648
    ):
        raise ValueError("shape resample authority mismatch")
    _declared_artifact(shape_summary, "selected", shape_selected)
    _declared_artifact(shape_summary, "manifest", shape_manifest)

    stages = {
        "random": serial_root / "random_fallback_base.v3.semantic_gate",
        "dtype": serial_root / "dtype_fallback_base.v2.semantic_gate_double",
        "layout": serial_root / "layout_fallback_base.v2.semantic_gate_double",
    }
    summaries: dict[str, Mapping[str, Any]] = {}
    for stage, directory in stages.items():
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        if (
            summary.get("contract_version") != SERIAL_CONTRACT
            or summary.get("training_approved") is not False
            or summary.get("rows") != 31_648
        ):
            raise ValueError(f"{stage} serial authority mismatch")
        _declared_artifact(summary, "selected_parquet", directory / "selected.parquet")
        _declared_artifact(summary, "manifest", directory / "manifest.jsonl")
        summaries[stage] = summary
    if (summaries["random"].get("inputs") or {}).get("primary_sha256") != _file_sha256(shape_selected):
        raise ValueError("random stage does not bind the shape resample")
    if (summaries["dtype"].get("inputs") or {}).get("primary_sha256") != _file_sha256(
        stages["random"] / "selected.parquet"
    ):
        raise ValueError("dtype stage does not bind the random fallback base")
    if (summaries["layout"].get("inputs") or {}).get("primary_sha256") != _file_sha256(
        stages["dtype"] / "selected.parquet"
    ):
        raise ValueError("layout stage does not bind the dtype fallback base")
    return {
        "shape_summary": shape_summary_path,
        "shape_selected": shape_selected,
        "shape_manifest": shape_manifest,
        "stages": stages,
    }


def _canonical_parent_indices(canonical: Path) -> dict[str, int]:
    result: dict[str, int] = {}
    for index, row in enumerate(_parquet_rows(canonical, columns=("reward_model", "extra_info"))):
        uuid = _uuid(row)
        if uuid in result:
            raise ValueError(f"duplicate canonical parent UUID: {uuid}")
        v4 = _v4(row)
        if v4.get("row_index") != index or v4.get("reference_sha256") != _reference_sha256(row):
            raise ValueError(f"canonical parent row binding mismatch: {index}")
        result[uuid] = index
    if len(result) != 64_315:
        raise ValueError(f"canonical parent count mismatch: {len(result)}")
    return result


def _interventions(
    random_manifest: Mapping[str, Any],
    dtype_manifest: Mapping[str, Any],
    layout_manifest: Mapping[str, Any],
) -> tuple[str, ...]:
    values = ["shape"]
    if random_manifest.get("selected_layer") == "random":
        values.append("random")
    if dtype_manifest.get("selected_layer") == "dtype":
        values.append("dtype")
    if layout_manifest.get("selected_layer") == "layout":
        values.append("layout")
    return tuple(values)


def _review_hash(root: str, interventions: Sequence[str]) -> str:
    return hashlib.sha256(f"review\0{root}\0{'+'.join(interventions)}".encode()).hexdigest()


def _consider_review(
    review_cells: dict[tuple[str, str, str], tuple[str, dict[str, Any]]],
    *,
    row: Mapping[str, Any],
    root: str,
    interventions: Sequence[str],
) -> None:
    v4 = _v4(row)
    cell = (
        "+".join(interventions) if interventions else "canonical_parent",
        str(v4.get("source_family")),
        str(v4.get("operator_bucket")),
    )
    digest = _review_hash(root, interventions)
    previous = review_cells.get(cell)
    if previous is None or digest < previous[0]:
        review_cells[cell] = (digest, dict(row))


def _write_review(
    path: Path, review_cells: Mapping[tuple[str, str, str], tuple[str, Mapping[str, Any]]], count: int
) -> int:
    required_combos: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for (combo, _source, _operator), value in review_cells.items():
        if combo not in required_combos or value[0] < required_combos[combo][0]:
            required_combos[combo] = value
    chosen: list[tuple[str, Mapping[str, Any]]] = sorted(required_combos.values(), key=lambda item: item[0])
    seen = {_uuid(row) for _, row in chosen}
    for item in sorted(review_cells.values(), key=lambda value: value[0]):
        if len(chosen) >= count:
            break
        if _uuid(item[1]) not in seen:
            chosen.append(item)
            seen.add(_uuid(item[1]))
    chunks = [
        "# Serial-expanded one-question-per-parent review samples\n",
        "Samples cover every intervention combination first, then source and operator buckets.\n",
    ]
    for ordinal, (digest, row) in enumerate(chosen[:count], 1):
        v4 = _v4(row)
        extra = row["extra_info"]
        augmentation = extra.get("augmentation") if isinstance(extra, Mapping) else None
        chunks.extend(
            [
                f"\n## {ordinal:02d}. {_uuid(row)}\n",
                f"- canonical parent: `{v4.get('row_index')}:{v4.get('parent_uuid') or _uuid(row)}`\n",
                f"- source: `{v4.get('source_family')}`; operator bucket: `{v4.get('operator_bucket')}`\n",
                f"- terminal intervention: `{(augmentation or {}).get('intervention_kind') or 'canonical/shape'}`\n",
                f"- review SHA: `{digest}`\n",
                "\n### Prompt\n\n```json\n",
                json.dumps(row.get("prompt"), indent=2, ensure_ascii=False),
                "\n```\n\n### Reference program\n\n```python\n",
                _reference_code(row),
                "\n```\n",
            ]
        )
    path.write_text("".join(chunks), encoding="utf-8")
    return min(len(chosen), count)


def _normalize_and_record(
    rows: Sequence[Mapping[str, Any]],
    *,
    schema: pa.Schema,
    writer: pq.ParquetWriter,
) -> list[dict[str, Any]]:
    table = pa.Table.from_pylist(list(rows), schema=schema)
    normalized = table.to_pylist()
    writer.write_table(table)
    return normalized


CONTRACT = "prompt_tvm_v4_serial_one_question_per_parent_release_sample_v2"
SELECTION_POLICY = "terminal_serial_child_plus_coverage_weighted_20pct_canonical_fallback_v1"
DEFAULT_OUTPUT = DEFAULT_SERIAL_ROOT / "one_question_per_parent.v1"
DEFAULT_FALLBACK_FRACTION = 0.20


class FallbackCandidate(NamedTuple):
    row_index: int
    root: str
    stratum: tuple[str, str, str]
    families: tuple[str, ...]
    tie_break: str


def _feature_labels(row: Mapping[str, Any]) -> tuple[tuple[str, str, str], tuple[str, ...]]:
    v4 = _v4(row)
    stratum = (
        str(v4.get("source_family")),
        str(v4.get("operator_bucket")),
        str(v4.get("mode_class")),
    )
    families: set[str] = set()
    for token in extract_operator_signature(_reference_code(row)):
        families.update(token_family_cells(token))
    return stratum, tuple(sorted(families))


def _selection_hash(root: str) -> str:
    return hashlib.sha256(f"{SELECTION_POLICY}\0{root}".encode()).hexdigest()


def _proportional_targets(counts: Mapping[tuple[str, str, str], int], total: int) -> dict[tuple[str, str, str], int]:
    denominator = sum(counts.values())
    raw = {key: total * value / denominator for key, value in counts.items()}
    result = {key: math.floor(value) for key, value in raw.items()}
    remainder = total - sum(result.values())
    order = sorted(raw, key=lambda key: (-(raw[key] - result[key]), key))
    for key in order[:remainder]:
        result[key] += 1
    return result


def _bounded_proportional_quotas(
    weights: Mapping[tuple[str, str, str], int],
    capacities: Mapping[tuple[str, str, str], int],
    total: int,
) -> dict[tuple[str, str, str], int]:
    keys = sorted(capacities)
    result = {key: 0 for key in keys}
    positive_capacity = sum(capacities[key] for key in keys if weights.get(key, 0) > 0)
    if positive_capacity < total:
        raise ValueError("positive-deficit strata cannot satisfy fallback quota")
    for _ in range(total):
        active = [key for key in keys if result[key] < capacities[key] and weights.get(key, 0) > 0]
        if not active:
            raise ValueError("fallback stratum allocation exhausted")
        key = min(
            active,
            key=lambda item: (
                -weights[item] / (result[item] + 1),
                item,
            ),
        )
        result[key] += 1
    return result


def _choose_fallbacks(
    candidates: Sequence[FallbackCandidate],
    *,
    quota: int,
    canonical_strata: Mapping[tuple[str, str, str], int],
    terminal_strata: Mapping[tuple[str, str, str], int],
    canonical_families: Mapping[str, int],
    terminal_families: Mapping[str, int],
    final_rows: int,
    canonical_rows: int,
) -> tuple[set[int], dict[str, Any]]:
    candidate_strata: collections.Counter[tuple[str, str, str]] = collections.Counter(
        item.stratum for item in candidates
    )
    target_strata = _proportional_targets(canonical_strata, final_rows)
    deficits = {key: max(target_strata.get(key, 0) - terminal_strata.get(key, 0), 0) for key in canonical_strata}
    quotas = _bounded_proportional_quotas(deficits, candidate_strata, quota)

    family_targets = {
        family: round(count * final_rows / canonical_rows) for family, count in canonical_families.items()
    }
    family_deficits = {
        family: max(family_targets[family] - terminal_families.get(family, 0), 0) for family in family_targets
    }
    family_priority = {
        family: family_deficits[family] / max(canonical_families[family], 1) for family in family_deficits
    }

    by_stratum: dict[tuple[str, str, str], list[FallbackCandidate]] = collections.defaultdict(list)
    for item in candidates:
        by_stratum[item.stratum].append(item)
    selected: set[int] = set()
    selected_strata: collections.Counter[tuple[str, str, str]] = collections.Counter()
    selected_families: collections.Counter[str] = collections.Counter()
    for stratum in sorted(by_stratum):
        ranked = sorted(
            by_stratum[stratum],
            key=lambda item: (
                -sum(family_priority.get(family, 0.0) for family in item.families),
                item.tie_break,
                item.row_index,
            ),
        )
        for item in ranked[: quotas[stratum]]:
            selected.add(item.row_index)
            selected_strata[item.stratum] += 1
            selected_families.update(item.families)
    if len(selected) != quota or selected_strata != collections.Counter(quotas):
        raise ValueError("fallback selection did not satisfy exact stratum quotas")

    def display_stratum(value: tuple[str, str, str]) -> str:
        return "|".join(value)

    return selected, {
        "stratum_definition": ["source_family", "operator_bucket", "mode_class"],
        "canonical_counts": {display_stratum(key): canonical_strata[key] for key in sorted(canonical_strata)},
        "terminal_counts": {display_stratum(key): terminal_strata.get(key, 0) for key in sorted(canonical_strata)},
        "fallback_population_counts": {
            display_stratum(key): candidate_strata.get(key, 0) for key in sorted(canonical_strata)
        },
        "target_final_counts": {display_stratum(key): target_strata.get(key, 0) for key in sorted(canonical_strata)},
        "fallback_selected_counts": {
            display_stratum(key): selected_strata.get(key, 0) for key in sorted(canonical_strata)
        },
        "family_target_counts": dict(sorted(family_targets.items())),
        "family_terminal_counts": dict(sorted(terminal_families.items())),
        "family_fallback_selected_counts": dict(sorted(selected_families.items())),
    }


def _schema(serial_path: Path) -> pa.Schema:
    schema = pq.read_schema(serial_path)
    metadata = dict(schema.metadata or {})
    metadata.update(
        {
            b"release.contract": CONTRACT.encode(),
            b"release.selection_policy": SELECTION_POLICY.encode(),
            b"release.review_only": b"true",
            b"release.training_approved": b"false",
        }
    )
    return schema.with_metadata(metadata)


def _record_batch(
    rows: list[Mapping[str, Any]],
    manifests: list[dict[str, Any]],
    *,
    schema: pa.Schema,
    writer: pq.ParquetWriter,
    manifest_handle: Any,
    selected_uuids: set[str],
    selected_references: set[str],
    selected_asts: set[str],
) -> int:
    if not rows:
        return 0
    normalized = _normalize_and_record(rows, schema=schema, writer=writer)
    for row, item in zip(normalized, manifests, strict=True):
        item["selected_row_sha256"] = _canonical_sha256(row)
        manifest_handle.write(_canonical_bytes(item).decode("utf-8") + "\n")
        selected_uuids.add(_uuid(row))
        selected_references.add(str(item["selected_reference_sha256"]))
        selected_asts.add(str(item["selected_normalized_ast_sha256"]))
    rows.clear()
    manifests.clear()
    return len(normalized)


def sample_release(
    *,
    canonical: Path,
    build_manifest: Path,
    shape_dir: Path,
    serial_root: Path,
    output_dir: Path,
    review_sample_count: int,
    fallback_fraction: float,
) -> dict[str, Any]:
    canonical = canonical.resolve()
    build_manifest = build_manifest.resolve()
    shape_dir = shape_dir.resolve()
    serial_root = serial_root.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if review_sample_count <= 0:
        raise ValueError("review_sample_count must be positive")
    if not 0.0 < fallback_fraction < 1.0:
        raise ValueError("fallback_fraction must be between zero and one")
    authority = _validate_authorities(
        canonical=canonical,
        build_manifest=build_manifest,
        shape_dir=shape_dir,
        serial_root=serial_root,
    )
    canonical_indices = _canonical_parent_indices(canonical)
    stages = authority["stages"]
    schema = _schema(stages["layout"] / "selected.parquet")
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp.", dir=output_dir.parent))
    parquet_path = temporary / "selected.parquet"
    manifest_path = temporary / "selected.manifest.jsonl"
    summary_path = temporary / "summary.json"
    review_path = temporary / "review_samples.md"

    covered_roots: set[str] = set()
    terminal_features: dict[str, tuple[tuple[str, str, str], tuple[str, ...]]] = {}
    terminal_strata: collections.Counter[tuple[str, str, str]] = collections.Counter()
    terminal_families: collections.Counter[str] = collections.Counter()
    combination_counts: collections.Counter[str] = collections.Counter()
    random_family_counts: collections.Counter[str] = collections.Counter()
    dtype_counts: collections.Counter[str] = collections.Counter()
    layout_family_counts: collections.Counter[str] = collections.Counter()
    shape_rank_counts: collections.Counter[str] = collections.Counter()
    shape_size_counts: collections.Counter[str] = collections.Counter()
    review_cells: dict[tuple[str, str, str], tuple[str, dict[str, Any]]] = {}
    selected_uuids: set[str] = set()
    selected_references: set[str] = set()
    selected_asts: set[str] = set()
    selected_roots: set[str] = set()
    output_rows = 0

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pq.ParquetWriter(parquet_path, schema, compression="zstd") as writer, manifest_path.open(
            "w", encoding="utf-8"
        ) as manifest_handle:
            rows_buffer: list[Mapping[str, Any]] = []
            manifest_buffer: list[dict[str, Any]] = []
            sources = zip(
                _parquet_rows(authority["shape_selected"]),
                _parquet_rows(stages["random"] / "selected.parquet"),
                _parquet_rows(stages["dtype"] / "selected.parquet"),
                _parquet_rows(stages["layout"] / "selected.parquet"),
                _jsonl_rows(authority["shape_manifest"]),
                _jsonl_rows(stages["random"] / "manifest.jsonl"),
                _jsonl_rows(stages["dtype"] / "manifest.jsonl"),
                _jsonl_rows(stages["layout"] / "manifest.jsonl"),
                strict=True,
            )
            for index, values in enumerate(sources):
                (
                    shape_row,
                    random_row,
                    dtype_row,
                    terminal_row,
                    shape_manifest,
                    random_manifest,
                    dtype_manifest,
                    layout_manifest,
                ) = values
                root = shape_manifest.get("canonical_parent_uuid")
                if not isinstance(root, str) or root not in canonical_indices or root in covered_roots:
                    raise ValueError(f"invalid or duplicate shape root: {index}")
                shape_uuid = _uuid(shape_row)
                if (
                    shape_manifest.get("selected_index") != index
                    or shape_manifest.get("shape_child_uuid") != shape_uuid
                    or _v4(shape_row).get("parent_uuid") != root
                    or _v4(shape_row).get("row_index") != canonical_indices[root]
                    or shape_manifest.get("reference_sha256") != _reference_sha256(shape_row)
                ):
                    raise ValueError(f"shape row lineage mismatch: {index}")
                if (
                    random_manifest.get("row_index") != index
                    or random_manifest.get("canonical_parent_uuid") != root
                    or random_manifest.get("shape_uuid") != shape_uuid
                    or random_manifest.get("selected_uuid") != _uuid(random_row)
                    or random_manifest.get("selected_row_sha256") != _canonical_sha256(random_row)
                ):
                    raise ValueError(f"random row lineage mismatch: {index}")
                if (
                    dtype_manifest.get("row_index") != index
                    or dtype_manifest.get("prior_uuid") != _uuid(random_row)
                    or dtype_manifest.get("prior_manifest_row_sha256") != _canonical_sha256(random_manifest)
                    or dtype_manifest.get("selected_uuid") != _uuid(dtype_row)
                    or dtype_manifest.get("selected_row_sha256") != _canonical_sha256(dtype_row)
                ):
                    raise ValueError(f"dtype row lineage mismatch: {index}")
                if (
                    layout_manifest.get("row_index") != index
                    or layout_manifest.get("prior_uuid") != _uuid(dtype_row)
                    or layout_manifest.get("prior_manifest_row_sha256") != _canonical_sha256(dtype_manifest)
                    or layout_manifest.get("selected_uuid") != _uuid(terminal_row)
                    or layout_manifest.get("selected_row_sha256") != _canonical_sha256(terminal_row)
                    or _v4(terminal_row).get("row_index") != canonical_indices[root]
                ):
                    raise ValueError(f"layout row lineage mismatch: {index}")
                reference_sha = _reference_sha256(terminal_row)
                ast_sha = _normalized_ast_sha256(terminal_row)
                v4 = _v4(terminal_row)
                if v4.get("reference_sha256") != reference_sha or v4.get("normalized_ast_sha256") != ast_sha:
                    raise ValueError(f"terminal identity mismatch: {index}")
                features = _feature_labels(terminal_row)
                terminal_features[root] = features
                terminal_strata[features[0]] += 1
                terminal_families.update(features[1])
                interventions = _interventions(random_manifest, dtype_manifest, layout_manifest)
                combination_counts["+".join(interventions)] += 1
                if "random" in interventions:
                    random_family_counts[str(shape_manifest.get("assigned_random_family"))] += 1
                if "dtype" in interventions:
                    augmentation = dtype_row["extra_info"].get("augmentation") or {}
                    dtype_counts[str(augmentation.get("dtype_after"))] += 1
                if "layout" in interventions:
                    augmentation = terminal_row["extra_info"].get("augmentation") or {}
                    layout_family_counts[str(augmentation.get("layout_after"))] += 1
                shape_rank_counts[str(shape_manifest.get("rank_bucket"))] += 1
                shape_size_counts[str(shape_manifest.get("sampled_size_bucket"))] += 1
                rows_buffer.append(terminal_row)
                manifest_buffer.append(
                    {
                        "contract": CONTRACT,
                        "sample_row_index": output_rows + len(rows_buffer) - 1,
                        "canonical_parent_uuid": root,
                        "canonical_parent_row_index": canonical_indices[root],
                        "selection_policy": SELECTION_POLICY,
                        "selected_origin": "serial_terminal",
                        "serial_row_index": index,
                        "shape_uuid": shape_uuid,
                        "random_uuid": _uuid(random_row),
                        "dtype_uuid": _uuid(dtype_row),
                        "terminal_uuid": _uuid(terminal_row),
                        "interventions": list(interventions),
                        "shape_manifest_row_sha256": _canonical_sha256(shape_manifest),
                        "random_manifest_row_sha256": _canonical_sha256(random_manifest),
                        "dtype_manifest_row_sha256": _canonical_sha256(dtype_manifest),
                        "layout_manifest_row_sha256": _canonical_sha256(layout_manifest),
                        "selected_reference_sha256": reference_sha,
                        "selected_normalized_ast_sha256": ast_sha,
                        "training_approved": False,
                    }
                )
                covered_roots.add(root)
                selected_roots.add(root)
                _consider_review(review_cells, row=terminal_row, root=root, interventions=interventions)
                if len(rows_buffer) == 16:
                    output_rows += _record_batch(
                        rows_buffer,
                        manifest_buffer,
                        schema=schema,
                        writer=writer,
                        manifest_handle=manifest_handle,
                        selected_uuids=selected_uuids,
                        selected_references=selected_references,
                        selected_asts=selected_asts,
                    )
            output_rows += _record_batch(
                rows_buffer,
                manifest_buffer,
                schema=schema,
                writer=writer,
                manifest_handle=manifest_handle,
                selected_uuids=selected_uuids,
                selected_references=selected_references,
                selected_asts=selected_asts,
            )

            canonical_strata: collections.Counter[tuple[str, str, str]] = collections.Counter()
            canonical_families: collections.Counter[str] = collections.Counter()
            fallback_candidates: list[FallbackCandidate] = []
            for base_index, row in enumerate(_parquet_rows(canonical, columns=("reward_model", "extra_info"))):
                root = _uuid(row)
                if canonical_indices.get(root) != base_index:
                    raise ValueError(f"canonical order mismatch: {base_index}")
                v4 = _v4(row)
                if v4.get("reference_sha256") != _reference_sha256(row) or v4.get(
                    "normalized_ast_sha256"
                ) != _normalized_ast_sha256(row):
                    raise ValueError(f"canonical identity mismatch: {base_index}")
                features = _feature_labels(row)
                canonical_strata[features[0]] += 1
                canonical_families.update(features[1])
                if root in covered_roots:
                    if terminal_features[root] != features:
                        raise ValueError(f"terminal changed operator coverage: {root}")
                    continue
                fallback_candidates.append(
                    FallbackCandidate(base_index, root, features[0], features[1], _selection_hash(root))
                )
            fallback_population = len(fallback_candidates)
            fallback_quota = int(math.floor(fallback_population * fallback_fraction + 0.5))
            selected_fallback_indices, coverage = _choose_fallbacks(
                fallback_candidates,
                quota=fallback_quota,
                canonical_strata=canonical_strata,
                terminal_strata=terminal_strata,
                canonical_families=canonical_families,
                terminal_families=terminal_families,
                final_rows=len(covered_roots) + fallback_quota,
                canonical_rows=len(canonical_indices),
            )

            fallback_lookup = {item.row_index: item for item in fallback_candidates}
            for base_index, row in enumerate(_parquet_rows(canonical)):
                if base_index not in selected_fallback_indices:
                    continue
                item = fallback_lookup[base_index]
                root = _uuid(row)
                if root != item.root or root in selected_roots:
                    raise ValueError(f"selected fallback root mismatch: {base_index}")
                reference_sha = _reference_sha256(row)
                ast_sha = _normalized_ast_sha256(row)
                rows_buffer.append(row)
                manifest_buffer.append(
                    {
                        "contract": CONTRACT,
                        "sample_row_index": output_rows + len(rows_buffer) - 1,
                        "canonical_parent_uuid": root,
                        "canonical_parent_row_index": base_index,
                        "selection_policy": SELECTION_POLICY,
                        "selected_origin": "coverage_weighted_canonical_fallback",
                        "serial_row_index": None,
                        "shape_uuid": None,
                        "random_uuid": None,
                        "dtype_uuid": None,
                        "terminal_uuid": root,
                        "interventions": [],
                        "fallback_selection": {
                            "stratum": list(item.stratum),
                            "operator_families": list(item.families),
                            "tie_break_sha256": item.tie_break,
                        },
                        "canonical_row_sha256": _canonical_sha256(row),
                        "selected_reference_sha256": reference_sha,
                        "selected_normalized_ast_sha256": ast_sha,
                        "training_approved": False,
                    }
                )
                combination_counts["canonical_parent"] += 1
                selected_roots.add(root)
                _consider_review(review_cells, row=row, root=root, interventions=())
                if len(rows_buffer) == 16:
                    output_rows += _record_batch(
                        rows_buffer,
                        manifest_buffer,
                        schema=schema,
                        writer=writer,
                        manifest_handle=manifest_handle,
                        selected_uuids=selected_uuids,
                        selected_references=selected_references,
                        selected_asts=selected_asts,
                    )
            output_rows += _record_batch(
                rows_buffer,
                manifest_buffer,
                schema=schema,
                writer=writer,
                manifest_handle=manifest_handle,
                selected_uuids=selected_uuids,
                selected_references=selected_references,
                selected_asts=selected_asts,
            )

        expected_rows = len(covered_roots) + fallback_quota
        if len(covered_roots) != 31_648 or output_rows != expected_rows or len(selected_roots) != expected_rows:
            raise ValueError("release row/root count mismatch")
        if not (len(selected_uuids) == len(selected_references) == len(selected_asts) == expected_rows):
            raise ValueError("release contains duplicate UUID, reference, or normalized AST")

        source_counts: collections.Counter[str] = collections.Counter()
        operator_counts: collections.Counter[str] = collections.Counter()
        mode_counts: collections.Counter[str] = collections.Counter()
        final_families: collections.Counter[str] = collections.Counter()
        for row in _parquet_rows(parquet_path, columns=("reward_model", "extra_info")):
            v4 = _v4(row)
            source_counts[str(v4.get("source_family"))] += 1
            operator_counts[str(v4.get("operator_bucket"))] += 1
            mode_counts[str(v4.get("mode_class"))] += 1
            final_families.update(_feature_labels(row)[1])
        review_rows = _write_review(review_path, review_cells, review_sample_count)
        script = Path(__file__).resolve()
        complexity = REPO_ROOT / "tools/data/cleaning/complexity.py"
        classifier = REPO_ROOT / "tools/data/synthesize/low_level_operator_families.py"
        summary: dict[str, Any] = {
            "contract": CONTRACT,
            "selection_policy": {
                "name": SELECTION_POLICY,
                "definition": (
                    "retain every validated terminal serial child; sample exactly 20% of uncovered canonical "
                    "parents with source/operator/mode deficit quotas and operator-family deficit priority"
                ),
                "fallback_fraction_requested": fallback_fraction,
                "fallback_fraction_actual": fallback_quota / fallback_population,
            },
            "canonical_parent_rows": len(canonical_indices),
            "serial_terminal_rows": len(covered_roots),
            "canonical_fallback_population_rows": fallback_population,
            "canonical_fallback_selected_rows": fallback_quota,
            "selected_rows": output_rows,
            "combination_counts": dict(sorted(combination_counts.items())),
            "intervention_counts": {
                "shape": len(covered_roots),
                "random": sum(count for combo, count in combination_counts.items() if "random" in combo.split("+")),
                "dtype": sum(count for combo, count in combination_counts.items() if "dtype" in combo.split("+")),
                "layout": sum(count for combo, count in combination_counts.items() if "layout" in combo.split("+")),
            },
            "random_family_counts": dict(sorted(random_family_counts.items())),
            "dtype_target_counts": dict(sorted(dtype_counts.items())),
            "layout_family_counts": dict(sorted(layout_family_counts.items())),
            "shape_rank_counts": dict(sorted(shape_rank_counts.items())),
            "shape_size_counts": dict(sorted(shape_size_counts.items())),
            "source_family_counts": dict(sorted(source_counts.items())),
            "operator_bucket_counts": dict(sorted(operator_counts.items())),
            "mode_class_counts": dict(sorted(mode_counts.items())),
            "operator_family_row_presence": dict(sorted(final_families.items())),
            "coverage_weighting": coverage,
            "identity": {
                "unique_selected_canonical_parents": len(selected_roots),
                "unique_selected_uuids": len(selected_uuids),
                "unique_reference_sha256": len(selected_references),
                "unique_normalized_ast_sha256": len(selected_asts),
            },
            "source_binding": {
                "canonical_parquet": _artifact(canonical, rows=len(canonical_indices)),
                "canonical_build_manifest": _artifact(build_manifest),
                "shape_summary": _artifact(authority["shape_summary"]),
                "shape_selected": _artifact(authority["shape_selected"], rows=len(covered_roots)),
                "shape_manifest": _artifact(authority["shape_manifest"], rows=len(covered_roots)),
                **{
                    f"{stage}_{kind}": _artifact(directory / filename, rows=31_648 if kind != "summary" else None)
                    for stage, directory in stages.items()
                    for kind, filename in (
                        ("selected", "selected.parquet"),
                        ("manifest", "manifest.jsonl"),
                        ("summary", "summary.json"),
                    )
                },
                "sampler": _artifact(script),
                "complexity_extractor": _artifact(complexity),
                "operator_family_classifier": _artifact(classifier),
            },
            "artifacts": {
                "selected_parquet": _staged_artifact(parquet_path, output_dir / "selected.parquet", rows=output_rows),
                "selected_manifest": _staged_artifact(
                    manifest_path, output_dir / "selected.manifest.jsonl", rows=output_rows
                ),
                "review_samples": _staged_artifact(review_path, output_dir / "review_samples.md", rows=review_rows),
            },
            "review_only": True,
            "training_approved": False,
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
        os.replace(temporary, output_dir)
        return summary
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--build-manifest", type=Path, default=DEFAULT_BUILD_MANIFEST)
    parser.add_argument("--shape-dir", type=Path, default=DEFAULT_SHAPE_DIR)
    parser.add_argument("--serial-root", type=Path, default=DEFAULT_SERIAL_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--review-sample-count", type=int, default=24)
    parser.add_argument("--fallback-fraction", type=float, default=DEFAULT_FALLBACK_FRACTION)
    return parser


def main() -> None:
    args = _parser().parse_args()
    result = sample_release(
        canonical=args.canonical,
        build_manifest=args.build_manifest,
        shape_dir=args.shape_dir,
        serial_root=args.serial_root,
        output_dir=args.output_dir,
        review_sample_count=args.review_sample_count,
        fallback_fraction=args.fallback_fraction,
    )
    print(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
