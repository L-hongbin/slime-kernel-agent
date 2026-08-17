#!/usr/bin/env python3
"""Select a deterministic 200-row four-lane KernelBench-support canary."""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.cleaning.complexity import extract_complexity_features, extract_operator_signature, feature_dict
from tools.data.synthesize import profile_prompt_tvm_distribution as profiler


CONTRACT_VERSION = "four_lane_kernelbench_high_complexity_canary_v3"
SELECTION_SEED_VERSION = "four_lane_kernelbench_support_canary_v1"
ROWS = 200
LANES = ("extreme", "semantic", "frontier", "gap")
MIN_LANE_ROWS = {lane: 10 for lane in LANES}
COMPLEXITY_MINIMUMS = {
    "operator_count_ge_10": 160,
    "operator_count_ge_16": 100,
    "forward_calls_ge_10": 160,
    "forward_calls_ge_15": 100,
    "source_lines_ge_50": 80,
    "registered_modules_ge_5": 80,
    "multiple_top_level_classes": 40,
}
FAMILY_MINIMUMS = {
    "activation": 40,
    "attention_recurrent": 8,
    "conv": 40,
    "indexing_scatter": 5,
    "matmul_linear": 30,
    "normalization": 30,
    "pooling": 20,
    "reduction": 30,
    "shape_layout": 30,
}
HIGH_COMPLEXITY_LANE_MINIMUMS = {"extreme": 20, "frontier": 40, "gap": 80}
SOURCE_STRATUM_MINIMUMS = {
    ("semantic", "attention_recurrent_stateful"): 5,
    ("semantic", "pool_index"): 4,
    ("frontier", "attention_cache"): 2,
    ("frontier", "moe_routing"): 1,
    ("frontier", "selective_state_space"): 1,
    ("gap", "long_single_class"): 2,
    ("gap", "modular_multiclass"): 2,
}
FAMILIES = (
    "activation",
    "attention_recurrent",
    "conv",
    "fft_sparse",
    "indexing_scatter",
    "loss_distance",
    "matmul_linear",
    "normalization",
    "pooling",
    "reduction",
    "shape_layout",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _percent(count: int, denominator: int) -> float:
    return round(100.0 * count / denominator, 6) if denominator else 0.0


def _largest_shape(code: str) -> tuple[int | None, tuple[int, ...]]:
    tree = ast.parse(code)
    get_inputs = profiler._last_top_level_get_inputs(tree)
    if get_inputs is None:
        return None, ()
    observations = profiler._exact_factory_observations(get_inputs, profiler._module_environment(tree))
    shapes = [shape for _base, shape, _reason in observations if shape is not None and math.prod(shape) > 0]
    if not shapes:
        return None, ()
    largest = max(math.prod(shape) for shape in shapes)
    ranks = tuple(sorted({len(shape) for shape in shapes}))
    return largest, ranks


def _op_bucket(operator_count: int) -> str:
    if operator_count == 1:
        return "1"
    if operator_count <= 3:
        return "2_3"
    if operator_count <= 5:
        return "4_5"
    if operator_count <= 9:
        return "6_9"
    return "ge10"


def _shape_bucket(max_numel: int | None) -> str:
    if max_numel is None:
        return "unresolved"
    if max_numel < 1_000:
        return "lt1k"
    if max_numel < 1_000_000:
        return "1k_1m"
    if max_numel < 100_000_000:
        return "1m_100m"
    return "ge100m"


@dataclass(frozen=True)
class Candidate:
    lane: str
    lane_row_index: int
    uuid: str
    code: str
    reference_sha256: str
    normalized_ast_sha256: str
    operator_count: int
    operator_bucket: str
    families: frozenset[str]
    factories: frozenset[str]
    max_input_numel: int | None
    shape_bucket: str
    ranks: tuple[int, ...]
    forward_call_count: int
    source_line_count: int
    init_nn_constructor_count: int
    top_level_class_count: int
    source_row: dict[str, Any]
    source_manifest: dict[str, Any]

    def tie_key(self) -> str:
        return _sha256_text(f"{SELECTION_SEED_VERSION}|{self.lane}|{self.uuid}")

    def source_stratum(self) -> str:
        return str(
            self.source_manifest.get("primary_family")
            or self.source_manifest.get("semantic_family")
            or "unknown"
        )


def _load_lane(lane: str, parquet_path: Path, manifest_path: Path) -> list[Candidate]:
    table = pq.read_table(parquet_path)
    manifests = _read_jsonl(manifest_path)
    if table.num_rows != len(manifests):
        raise ValueError(f"{lane}: parquet/manifest row mismatch")
    rows = table.to_pylist()
    output = []
    for index, (source_row, manifest) in enumerate(zip(rows, manifests, strict=True)):
        code = source_row["reward_model"]["ground_truth"]
        uuid = source_row["extra_info"]["uuid"]
        if manifest.get("uuid") != uuid:
            raise ValueError(f"{lane}: UUID mismatch at row {index}")
        reference_sha = _sha256_text(code)
        if manifest.get("reference_sha256") != reference_sha:
            raise ValueError(f"{lane}: reference hash mismatch at row {index}")
        normalized = _sha256_text(ast.dump(ast.parse(code), include_attributes=False))
        if manifest.get("normalized_ast_sha256") != normalized:
            raise ValueError(f"{lane}: normalized AST mismatch at row {index}")
        signature = extract_operator_signature(code)
        complexity = feature_dict(extract_complexity_features(code))
        largest, ranks = _largest_shape(code)
        tree = ast.parse(code)
        get_inputs = profiler._last_top_level_get_inputs(tree)
        observations = (
            profiler._exact_factory_observations(get_inputs, profiler._module_environment(tree))
            if get_inputs is not None
            else []
        )
        output.append(
            Candidate(
                lane=lane,
                lane_row_index=index,
                uuid=uuid,
                code=code,
                reference_sha256=reference_sha,
                normalized_ast_sha256=normalized,
                operator_count=len(signature),
                operator_bucket=_op_bucket(len(signature)),
                families=frozenset(profiler._classify_signature(signature)),
                factories=frozenset(base for base, _shape, _reason in observations),
                max_input_numel=largest,
                shape_bucket=_shape_bucket(largest),
                ranks=ranks,
                forward_call_count=int(complexity["forward_call_count"]),
                source_line_count=int(complexity["source_line_count"]),
                init_nn_constructor_count=int(complexity["init_nn_constructor_count"]),
                top_level_class_count=int(complexity["top_level_class_count"]),
                source_row=source_row,
                source_manifest=manifest,
            )
        )
    return output


def _kernelbench_targets(kernelbench_path: Path) -> dict[str, dict[str, float]]:
    profile = profiler.profile_dataset("KernelBench", (kernelbench_path,))
    rows = int(profile["rows"])
    op_hist = {int(key): int(value) for key, value in profile["operator_count"]["histogram"].items()}
    op_buckets = collections.Counter()
    for count, value in op_hist.items():
        op_buckets[_op_bucket(count)] += value
    families = {
        key: float(profile["operator_family_row_presence"]["families"][key]["percent"]) / 100.0
        for key in FAMILIES
    }
    factories = {
        key: float(value["percent"]) / 100.0
        for key, value in profile["input_factory_row_presence"]["factories"].items()
    }
    return {
        "operator_bucket": {key: value / rows for key, value in op_buckets.items()},
        "family": families,
        "factory": factories,
    }


def _complexity_predicate(candidate: Candidate, name: str) -> bool:
    predicates = {
        "operator_count_ge_10": candidate.operator_count >= 10,
        "operator_count_ge_16": candidate.operator_count >= 16,
        "forward_calls_ge_10": candidate.forward_call_count >= 10,
        "forward_calls_ge_15": candidate.forward_call_count >= 15,
        "source_lines_ge_50": candidate.source_line_count >= 50,
        "registered_modules_ge_5": candidate.init_nn_constructor_count >= 5,
        "multiple_top_level_classes": candidate.top_level_class_count > 1,
    }
    return predicates[name]


def _choose(candidates: Sequence[Candidate], targets: Mapping[str, Mapping[str, float]]) -> list[Candidate]:
    # KernelBench is a support checklist, not a sampling target. Family and
    # structural coverage use hard floors; the remaining budget maximizes
    # realized operator/call depth and program structure.
    axes: list[tuple[str, str, int, float]] = []
    candidate_count = len(candidates)
    variable_count = candidate_count + 2 * len(axes)
    objective = np.zeros(variable_count)
    for axis_index, (_kind, _key, _target, weight) in enumerate(axes):
        objective[candidate_count + 2 * axis_index : candidate_count + 2 * axis_index + 2] = weight
    for index, candidate in enumerate(candidates):
        objective[index] = int(candidate.tie_key()[:8], 16) * 1e-15

    matrix_rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []
    row = np.zeros(variable_count)
    row[:candidate_count] = 1
    matrix_rows.append(row)
    lower.append(ROWS)
    upper.append(ROWS)
    for lane, minimum in MIN_LANE_ROWS.items():
        row = np.zeros(variable_count)
        row[:candidate_count] = [candidate.lane == lane for candidate in candidates]
        matrix_rows.append(row)
        lower.append(minimum)
        upper.append(np.inf)
    for lane, minimum in HIGH_COMPLEXITY_LANE_MINIMUMS.items():
        row = np.zeros(variable_count)
        row[:candidate_count] = [candidate.lane == lane for candidate in candidates]
        matrix_rows.append(row)
        lower.append(minimum)
        upper.append(np.inf)
    for (lane, stratum), minimum in SOURCE_STRATUM_MINIMUMS.items():
        row = np.zeros(variable_count)
        row[:candidate_count] = [
            candidate.lane == lane and candidate.source_stratum() == stratum
            for candidate in candidates
        ]
        matrix_rows.append(row)
        lower.append(minimum)
        upper.append(np.inf)
    for name, minimum in COMPLEXITY_MINIMUMS.items():
        row = np.zeros(variable_count)
        row[:candidate_count] = [_complexity_predicate(candidate, name) for candidate in candidates]
        matrix_rows.append(row)
        lower.append(minimum)
        upper.append(np.inf)
    for name, minimum in FAMILY_MINIMUMS.items():
        row = np.zeros(variable_count)
        row[:candidate_count] = [name in candidate.families for candidate in candidates]
        matrix_rows.append(row)
        lower.append(minimum)
        upper.append(np.inf)
    for axis_index, (kind, key, target, _weight) in enumerate(axes):
        row = np.zeros(variable_count)
        if kind == "family":
            row[:candidate_count] = [key in candidate.families for candidate in candidates]
        else:
            raise AssertionError(kind)
        row[candidate_count + 2 * axis_index] = -1
        row[candidate_count + 2 * axis_index + 1] = 1
        matrix_rows.append(row)
        lower.append(target)
        upper.append(target)
    result = milp(
        objective,
        integrality=np.r_[np.ones(candidate_count), np.zeros(2 * len(axes))],
        bounds=Bounds(
            np.zeros(variable_count),
            np.r_[np.ones(candidate_count), np.full(2 * len(axes), np.inf)],
        ),
        constraints=LinearConstraint(np.vstack(matrix_rows), lower, upper),
    )
    if not result.success or result.x is None:
        raise ValueError(f"MILP selection failed: {result.message}")
    selected = [candidate for candidate, value in zip(candidates, result.x[:candidate_count], strict=True) if value > 0.5]
    selected.sort(key=lambda candidate: (candidate.lane, candidate.lane_row_index))
    if len(selected) != ROWS or len({candidate.uuid for candidate in selected}) != ROWS:
        raise AssertionError("selection cardinality mismatch")
    lane_counts = collections.Counter(candidate.lane for candidate in selected)
    if any(lane_counts[lane] < minimum for lane, minimum in MIN_LANE_ROWS.items()):
        raise AssertionError(f"lane minima not satisfied: {lane_counts}")
    if any(lane_counts[lane] < minimum for lane, minimum in HIGH_COMPLEXITY_LANE_MINIMUMS.items()):
        raise AssertionError(f"high-complexity lane minima not satisfied: {lane_counts}")
    for (lane, stratum), minimum in SOURCE_STRATUM_MINIMUMS.items():
        actual = sum(
            candidate.lane == lane and candidate.source_stratum() == stratum
            for candidate in selected
        )
        if actual < minimum:
            raise AssertionError(f"source stratum minimum not satisfied: {lane}/{stratum}={actual}")
    for name, minimum in COMPLEXITY_MINIMUMS.items():
        actual = sum(_complexity_predicate(candidate, name) for candidate in selected)
        if actual < minimum:
            raise AssertionError(f"complexity minimum not satisfied: {name}={actual}")
    for name, minimum in FAMILY_MINIMUMS.items():
        actual = sum(name in candidate.families for candidate in selected)
        if actual < minimum:
            raise AssertionError(f"family minimum not satisfied: {name}={actual}")
    return selected


def _metrics(rows: Sequence[Candidate]) -> dict[str, Any]:
    count = len(rows)
    op = collections.Counter(row.operator_bucket for row in rows)
    families = collections.Counter(family for row in rows for family in row.families)
    factories = collections.Counter(factory for row in rows for factory in row.factories)
    lanes = collections.Counter(row.lane for row in rows)
    shapes = sorted(row.max_input_numel for row in rows if row.max_input_numel is not None)
    complexity = {
        name: {
            "count": sum(_complexity_predicate(row, name) for row in rows),
            "percent": _percent(sum(_complexity_predicate(row, name) for row in rows), count),
        }
        for name in COMPLEXITY_MINIMUMS
    }
    return {
        "rows": count,
        "lanes": dict(sorted(lanes.items())),
        "source_strata": dict(
            sorted(collections.Counter(f"{row.lane}:{row.source_stratum()}" for row in rows).items())
        ),
        "operator_buckets": {key: {"count": op[key], "percent": _percent(op[key], count)} for key in sorted(op)},
        "families": {
            key: {"count": families[key], "percent": _percent(families[key], count)} for key in FAMILIES
        },
        "factories": {
            key: {"count": factories[key], "percent": _percent(factories[key], count)}
            for key in sorted(set(factories) | {"rand", "randn", "randint"})
        },
        "shape": {
            "resolved_rows": len(shapes),
            "p50_max_input_numel": shapes[round((len(shapes) - 1) * 0.5)] if shapes else None,
            "p90_max_input_numel": shapes[round((len(shapes) - 1) * 0.9)] if shapes else None,
            "ge_1m_rows": sum(value >= 1_000_000 for value in shapes),
            "ge_100m_rows": sum(value >= 100_000_000 for value in shapes),
        },
        "structural_complexity": complexity,
        "unique_reference_sha256": len({row.reference_sha256 for row in rows}),
        "unique_normalized_ast_sha256": len({row.normalized_ast_sha256 for row in rows}),
    }


def _distribution_distance(
    metrics: Mapping[str, Any], targets: Mapping[str, Mapping[str, float]]
) -> dict[str, float]:
    def tvd(section: str, keys: Sequence[str]) -> float:
        return round(
            0.5
            * sum(
                abs(
                    float(metrics[section].get(key, {}).get("percent", 0.0)) / 100.0
                    - float(targets["operator_bucket" if section == "operator_buckets" else "family"][key])
                )
                for key in keys
            ),
            6,
        )

    operator_keys = ("1", "2_3", "4_5", "6_9", "ge10")
    family_keys = tuple(key for key in FAMILIES if key not in {"loss_distance", "fft_sparse"})
    return {
        "operator_bucket_tvd": tvd("operator_buckets", operator_keys),
        "operator_family_mean_absolute_percentage_point_error": round(
            sum(
                abs(
                    float(metrics["families"][key]["percent"])
                    - 100.0 * float(targets["family"][key])
                )
                for key in family_keys
            )
            / len(family_keys),
            6,
        ),
    }


def select(
    lane_inputs: Mapping[str, tuple[Path, Path]],
    kernelbench_path: Path,
    output_dir: Path,
    excluded_uuid_path: Path | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    if set(lane_inputs) != set(LANES):
        raise ValueError(f"expected exact lane set {LANES}")
    source_candidates = [
        candidate
        for lane in LANES
        for candidate in _load_lane(lane, lane_inputs[lane][0], lane_inputs[lane][1])
    ]
    if len(source_candidates) != 3_508:
        raise ValueError(f"expected frozen 3,508-row source pool, got {len(source_candidates)}")
    if len({candidate.uuid for candidate in source_candidates}) != len(source_candidates):
        raise ValueError("cross-lane UUID collision")
    if len({candidate.reference_sha256 for candidate in source_candidates}) != len(source_candidates):
        raise ValueError("cross-lane exact-reference collision")
    if len({candidate.normalized_ast_sha256 for candidate in source_candidates}) != len(source_candidates):
        raise ValueError("cross-lane normalized-AST collision")
    excluded = (
        [line.strip() for line in excluded_uuid_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if excluded_uuid_path is not None
        else []
    )
    if len(excluded) != len(set(excluded)) or any(uuid not in {row.uuid for row in source_candidates} for uuid in excluded):
        raise ValueError("excluded UUID file must be unique and a subset of the frozen source pool")
    candidates = [candidate for candidate in source_candidates if candidate.uuid not in set(excluded)]

    targets = _kernelbench_targets(kernelbench_path)
    selected = _choose(candidates, targets)
    output_dir.mkdir(parents=True)
    selected_table = pa.Table.from_pylist([row.source_row for row in selected])
    selected_path = output_dir / "selected.parquet"
    manifest_path = output_dir / "selected.manifest.jsonl"
    ledger_path = output_dir / "selection.jsonl"
    uuid_path = output_dir / "selected.uuids.txt"
    report_path = output_dir / "selection_report.md"
    summary_path = output_dir / "summary.json"
    pq.write_table(selected_table, selected_path, compression="zstd")
    manifest_path.write_text(
        "".join(
            _canonical_json(
                {
                    "canary_contract_version": CONTRACT_VERSION,
                    "canary_row_index": index,
                    "source_lane": row.lane,
                    "source_lane_row_index": row.lane_row_index,
                    "source_manifest": row.source_manifest,
                    "selection_features": {
                        "operator_count": row.operator_count,
                        "operator_bucket": row.operator_bucket,
                        "operator_families": sorted(row.families),
                        "input_factories": sorted(row.factories),
                        "max_input_numel": row.max_input_numel,
                        "shape_bucket": row.shape_bucket,
                        "input_ranks": list(row.ranks),
                        "structural_complexity": {
                            "forward_call_count": row.forward_call_count,
                            "source_line_count": row.source_line_count,
                            "init_nn_constructor_count": row.init_nn_constructor_count,
                            "top_level_class_count": row.top_level_class_count,
                        },
                    },
                    "uuid": row.uuid,
                    "reference_sha256": row.reference_sha256,
                    "normalized_ast_sha256": row.normalized_ast_sha256,
                    "review_only": True,
                    "training_approved": False,
                }
            )
            + "\n"
            for index, row in enumerate(selected)
        ),
        encoding="utf-8",
    )
    ledger_path.write_text(
        "".join(
            _canonical_json(
                {
                    "canary_row_index": index,
                    "uuid": row.uuid,
                    "source_lane": row.lane,
                    "source_lane_row_index": row.lane_row_index,
                    "operator_bucket": row.operator_bucket,
                    "operator_families": sorted(row.families),
                    "max_input_numel": row.max_input_numel,
                    "forward_call_count": row.forward_call_count,
                    "source_line_count": row.source_line_count,
                    "init_nn_constructor_count": row.init_nn_constructor_count,
                    "top_level_class_count": row.top_level_class_count,
                }
            )
            + "\n"
            for index, row in enumerate(selected)
        ),
        encoding="utf-8",
    )
    uuid_path.write_text("".join(row.uuid + "\n" for row in selected), encoding="utf-8")
    metrics = _metrics(selected)
    distance = _distribution_distance(metrics, targets)
    kb_profile = profiler.profile_dataset("KernelBench", (kernelbench_path,))
    report = [
        "# 四 lane 联合 KernelBench coverage canary",
        "",
        "本 canary 从四条冻结的 review-only lane 中确定性选择 200 条。KernelBench 仅用于 operator-family support checklist，不作为采样比例目标；选择器提高 operator count、forward call、源码行数、registered module 和多顶层 class 的高复杂度占比，同时要求每条 lane 至少保留 10 条。shape 留给后续 shape 扩增 pipeline，本轮不参与优化",
        "",
        "| Lane | Rows |",
        "| --- | ---: |",
    ]
    report.extend(f"| `{lane}` | {metrics['lanes'].get(lane, 0)} |" for lane in LANES)
    report.extend(["", "| Operator bucket | Canary | KernelBench reference |", "| --- | ---: | ---: |"])
    for key in ("1", "2_3", "4_5", "6_9", "ge10"):
        report.append(
            f"| `{key}` | {metrics['operator_buckets'].get(key, {}).get('percent', 0):.2f}% | "
            f"{100 * targets['operator_bucket'].get(key, 0):.2f}% |"
        )
    report.extend(["", "| Operator family | Canary | KernelBench |", "| --- | ---: | ---: |"])
    for family in FAMILIES:
        report.append(
            f"| `{family}` | {metrics['families'][family]['percent']:.2f}% | "
            f"{100 * targets['family'][family]:.2f}% |"
        )
    report.extend(["", "| Structural metric | Canary | Minimum |", "| --- | ---: | ---: |"])
    for name, minimum in COMPLEXITY_MINIMUMS.items():
        report.append(f"| `{name}` | {metrics['structural_complexity'][name]['percent']:.2f}% | {minimum} rows |")
    report.extend(
        [
            "",
            f"已解析 row-level 最大输入的 p50/p90 为 {metrics['shape']['p50_max_input_numel']:,}/{metrics['shape']['p90_max_input_numel']:,}，其中 {metrics['shape']['ge_1m_rows']} 条达到 1M。该统计只用于后续 shape pipeline 的输入，不参与本轮选择或通过判定",
            "",
            "本产物没有统一重跑四条 lane 的 GPU evidence，也没有把不同 runtime policy 的历史记录投影成一份新 authority。它是静态选择与人工/GPU复核输入，保持 `review_only=true`、`training_approved=false`",
            "",
        ]
    )
    report_path.write_text("\n".join(report), encoding="utf-8")
    summary = {
        "contract_version": CONTRACT_VERSION,
        "rows": ROWS,
        "selection_policy": {
            "target": "maximize high-complexity structure while covering KernelBench operator families within frozen source support",
            "lane_minimum_rows": MIN_LANE_ROWS,
            "high_complexity_lane_minimum_rows": HIGH_COMPLEXITY_LANE_MINIMUMS,
            "complexity_minimum_rows": COMPLEXITY_MINIMUMS,
            "operator_family_minimum_rows": FAMILY_MINIMUMS,
            "source_stratum_minimum_rows": {
                f"{lane}:{stratum}": minimum
                for (lane, stratum), minimum in SOURCE_STRATUM_MINIMUMS.items()
            },
            "shape_policy": "record only; deferred to the existing shape-expansion pipeline",
            "non_claim": "does not align KernelBench sampling ratios or shape distribution",
            "excluded_candidate_uuids": excluded,
            "selection_seed_version": SELECTION_SEED_VERSION,
        },
        "metrics": metrics,
        "distribution_distance": distance,
        "kernelbench_target": targets,
        "source": {
            "lanes": {
                lane: {
                    "parquet": {"path": str(paths[0].resolve()), "sha256": _sha256_file(paths[0])},
                    "manifest": {"path": str(paths[1].resolve()), "sha256": _sha256_file(paths[1])},
                }
                for lane, paths in lane_inputs.items()
            },
            "kernelbench": {
                "path": str(kernelbench_path.resolve()),
                "sha256": _sha256_file(kernelbench_path),
            },
            "selector": {"path": str(Path(__file__).resolve()), "sha256": _sha256_file(Path(__file__))},
            "exclusion_audit": (
                {"path": str(excluded_uuid_path.resolve()), "sha256": _sha256_file(excluded_uuid_path)}
                if excluded_uuid_path is not None
                else None
            ),
        },
        "artifacts": {
            path.name: {"path": str(path.resolve()), "sha256": _sha256_file(path)}
            for path in (selected_path, manifest_path, ledger_path, uuid_path, report_path)
        },
        "runtime_evidence_status": "requires_unified_canary_reference_and_liveness",
        "review_only": True,
        "training_approved": False,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for lane in LANES:
        parser.add_argument(f"--{lane}-parquet", type=Path, required=True)
        parser.add_argument(f"--{lane}-manifest", type=Path, required=True)
    parser.add_argument("--kernelbench", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-uuid-file", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    lane_inputs = {
        lane: (getattr(args, f"{lane}_parquet"), getattr(args, f"{lane}_manifest")) for lane in LANES
    }
    summary = select(lane_inputs, args.kernelbench, args.output_dir, args.exclude_uuid_file)
    print(_canonical_json({"rows": summary["rows"], "metrics": summary["metrics"]}))


if __name__ == "__main__":
    main()
