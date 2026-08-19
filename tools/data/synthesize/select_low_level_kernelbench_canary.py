#!/usr/bin/env python3
"""Select a dedup-audited 200-row low-level KernelBench coverage canary.

KernelBench contributes only a low-level operator support checklist.  Its
architecture-specific helper calls and composite modules are explicitly not
selection targets.  Program form (long source, call depth, registered modules,
and multiple top-level classes) is selected independently from operator
support.
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.cleaning.complexity import extract_operator_signature
from tools.data.synthesize import profile_prompt_tvm_distribution as profiler
from tools.data.synthesize import select_four_lane_kernelbench_canary as base


CONTRACT_VERSION = "low_level_kernelbench_high_complexity_canary_v1"
# Candidate.tie_key in the shared loader historically used this frozen seed.
# Keep the value explicit here so the emitted contract and actual tie-break agree.
SELECTION_SEED_VERSION = "four_lane_kernelbench_support_canary_v1"
ROWS = 200
LANES = ("structure", "gap", "semantic", "extreme")
EXACT_LANE_ROWS = {"structure": 120, "gap": 40, "semantic": 20, "extreme": 20}
COMPLEXITY_MINIMUMS = {
    "operator_count_ge_10": 180,
    "operator_count_ge_16": 100,
    "forward_calls_ge_10": 180,
    "forward_calls_ge_15": 100,
    "source_lines_ge_50": 160,
    "registered_modules_ge_5": 160,
    "multiple_top_level_classes": 50,
}
LOW_LEVEL_FAMILY_MINIMUMS = {
    "activation": 40,
    "conv": 40,
    "indexing_scatter": 5,
    "loss_distance": 6,
    "matmul_linear": 30,
    "normalization": 30,
    "pooling": 20,
    "reduction": 30,
    "shape_layout": 30,
}
DIRECT_PRIMITIVE_MINIMUMS = {"scaled_dot_product_attention": 2}
SOURCE_STRATUM_MINIMUMS = {
    ("structure", "matmul_linear_graph"): 13,
    ("structure", "conv_norm_graph"): 13,
    ("structure", "reduction_pool_graph"): 40,
    ("structure", "modular_multiclass_graph"): 40,
    ("structure", "layout_fusion_graph"): 14,
    ("gap", "long_single_class"): 10,
    ("gap", "modular_multiclass"): 10,
    ("gap", "loss_cross_entropy"): 2,
    ("gap", "loss_smooth_l1"): 2,
    ("gap", "loss_triplet_margin"): 2,
    ("semantic", "attention_recurrent_stateful"): 2,
    ("semantic", "heterogeneous_natural_dag"): 5,
    ("semantic", "matmul_reduction"): 4,
    ("semantic", "shape_branch"): 2,
    ("semantic", "conv_norm"): 2,
    ("semantic", "pool_index"): 5,
}

# These APIs or receiver-specific calls express an architecture/container or
# recurrent block rather than a low-level operator.  They are visible in the
# ledger but never become coverage constraints.
HIGH_LEVEL_EXACT = {
    "CausalSelfAttention",
    "ChannelShuffle",
    "DoubleConv",
    "InceptionModule",
    "nn.GRU",
    "nn.LSTM",
    "nn.MultiheadAttention",
    "nn.Sequential",
    "nn.TransformerEncoder",
}
HIGH_LEVEL_PREFIXES = (
    "call.",
    "self._make_",
    "self.forward_features",
    "self.head",
    "self.mlpf",
    "self.segsum",
)
NON_TARGET_EXACT = {
    "operator.eq",
    "operator.is",
    "operator.isnot",
    "operator.noteq",
    "self.hidden.copy_",
    "self.hidden.to",
    "torch.randn",
    "torch.tensor",
}
FORBIDDEN_COMPOSITE_APIS = {
    "nn.GRU",
    "nn.LSTM",
    "nn.MultiheadAttention",
    "nn.Sequential",
    "nn.TransformerEncoder",
}

_ACTIVATION_BASENAMES = {
    "elu",
    "gelu",
    "hardsigmoid",
    "hardswish",
    "hardtanh",
    "leaky_relu",
    "leakyrelu",
    "mish",
    "relu",
    "selu",
    "sigmoid",
    "silu",
    "softmax",
    "softplus",
    "tanh",
}
_MATMUL_BASENAMES = {"addmm", "bmm", "dot", "einsum", "linear", "matmul", "mm"}
_REDUCTION_BASENAMES = {
    "all",
    "amax",
    "amin",
    "any",
    "argmax",
    "argmin",
    "cumprod",
    "cumsum",
    "logsumexp",
    "max",
    "mean",
    "min",
    "norm",
    "prod",
    "sum",
}
_NORMALIZATION_BASENAMES = {
    "batch_norm",
    "batchnorm1d",
    "batchnorm2d",
    "batchnorm3d",
    "group_norm",
    "groupnorm",
    "instance_norm",
    "instancenorm2d",
    "instancenorm3d",
    "layer_norm",
    "layernorm",
    "normalize",
    "rms_norm",
}
_INDEXING_BASENAMES = {
    "embedding",
    "embedding_bag",
    "gather",
    "index_add",
    "index_select",
    "masked_fill",
    "narrow",
    "scatter",
    "select",
    "take",
    "take_along_dim",
    "topk",
    "tril",
    "triu",
    "where",
}
_SHAPE_LAYOUT_BASENAMES = {
    "cat",
    "chunk",
    "contiguous",
    "expand",
    "expand_as",
    "flatten",
    "flip",
    "movedim",
    "permute",
    "repeat",
    "repeat_interleave",
    "reshape",
    "reshape_as",
    "split",
    "squeeze",
    "stack",
    "transpose",
    "unfold",
    "unsqueeze",
    "view",
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _classify_kernelbench_token(token: str) -> str:
    if token in HIGH_LEVEL_EXACT or token.startswith(HIGH_LEVEL_PREFIXES):
        return "high_level_composition"
    if token in NON_TARGET_EXACT or token.startswith(("self.cls_token.",)):
        return "non_target_plumbing_or_control"
    return "low_level_operator"


def _token_cells(token: str) -> list[str]:
    lowered = token.lower()
    basename = lowered.rsplit(".", 1)[-1]
    cells: set[str] = set()
    if "conv" in basename:
        cells.add("conv")
    if basename in _MATMUL_BASENAMES:
        cells.add("matmul_linear")
    if basename in _REDUCTION_BASENAMES:
        cells.add("reduction")
    if basename in _NORMALIZATION_BASENAMES:
        cells.add("normalization")
    if basename in _ACTIVATION_BASENAMES:
        cells.add("activation")
    if "pool" in basename:
        cells.add("pooling")
    if (
        basename in _INDEXING_BASENAMES
        or "gather" in basename
        or "index" in basename
        or "scatter" in basename
    ):
        cells.add("indexing_scatter")
    if basename in _SHAPE_LAYOUT_BASENAMES:
        cells.add("shape_layout")
    if "loss" in basename or basename in {"cross_entropy", "nll"}:
        cells.add("loss_distance")
    if token == "torch.nn.functional.scaled_dot_product_attention":
        cells.add("scaled_dot_product_attention")
    return sorted(cells)


def _kernelbench_ledger(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    counts: collections.Counter[str] = collections.Counter()
    rows = 0
    for path in paths:
        for row in pq.read_table(path, columns=["reward_model"]).column(0).to_pylist():
            code = row["ground_truth"]
            counts.update(set(extract_operator_signature(code)))
            rows += 1
    ledger = [
        {
            "token": token,
            "kernelbench_row_count": count,
            "classification": _classify_kernelbench_token(token),
            "coverage_cells": _token_cells(token)
            if _classify_kernelbench_token(token) == "low_level_operator"
            else [],
        }
        for token, count in sorted(counts.items())
    ]
    classes = collections.Counter(row["classification"] for row in ledger)
    return ledger, {"rows": rows, "unique_tokens": len(ledger), **dict(sorted(classes.items()))}


def _candidate_cells(candidate: base.Candidate) -> set[str]:
    return {
        cell
        for token in extract_operator_signature(candidate.code)
        for cell in _token_cells(token)
    }


def _complexity(candidate: base.Candidate, name: str) -> bool:
    return {
        "operator_count_ge_10": candidate.operator_count >= 10,
        "operator_count_ge_16": candidate.operator_count >= 16,
        "forward_calls_ge_10": candidate.forward_call_count >= 10,
        "forward_calls_ge_15": candidate.forward_call_count >= 15,
        "source_lines_ge_50": candidate.source_line_count >= 50,
        "registered_modules_ge_5": candidate.init_nn_constructor_count >= 5,
        "multiple_top_level_classes": candidate.top_level_class_count > 1,
    }[name]


def _load_sources(
    lane_inputs: Mapping[str, tuple[Path, Path]], excluded: set[str]
) -> tuple[list[base.Candidate], dict[str, Any]]:
    all_rows = [
        candidate
        for lane in LANES
        for candidate in base._load_lane(lane, lane_inputs[lane][0], lane_inputs[lane][1])
    ]
    expected = {"structure": 200, "gap": 996, "semantic": 1_000, "extreme": 512}
    actual = collections.Counter(row.lane for row in all_rows)
    if dict(actual) != expected:
        raise ValueError(f"unexpected frozen source split: {dict(actual)}")
    for attribute in ("uuid", "reference_sha256", "normalized_ast_sha256"):
        values = [getattr(row, attribute) for row in all_rows]
        if len(values) != len(set(values)):
            raise ValueError(f"cross-lane collision in {attribute}")
    known = {row.uuid for row in all_rows}
    if not excluded <= known:
        raise ValueError(f"excluded UUIDs outside frozen sources: {sorted(excluded - known)}")
    filtered = []
    rejected_composites: collections.Counter[str] = collections.Counter()
    for row in all_rows:
        signature = set(extract_operator_signature(row.code))
        composites = signature & FORBIDDEN_COMPOSITE_APIS
        if row.uuid in excluded:
            continue
        if composites:
            rejected_composites.update(composites)
            continue
        filtered.append(row)
    return filtered, {
        "source_rows": dict(sorted(actual.items())),
        "excluded_uuid_rows": len(excluded),
        "excluded_composite_rows": sum(rejected_composites.values()),
        "excluded_composite_api_occurrences": dict(sorted(rejected_composites.items())),
        "eligible_rows": len(filtered),
    }


def _choose(candidates: Sequence[base.Candidate]) -> list[base.Candidate]:
    n = len(candidates)
    objective = np.zeros(n)
    for index, row in enumerate(candidates):
        # MILP minimizes.  Prefer deeper rows, then use a tiny stable hash tie-break.
        structural_score = min(row.operator_count, 24) + min(row.forward_call_count, 30)
        tie_key = _sha256_text(f"{SELECTION_SEED_VERSION}|{row.lane}|{row.uuid}")
        objective[index] = -structural_score * 1e-4 + int(tie_key[:8], 16) * 1e-15
    matrix: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    def constraint(mask: Iterable[bool], lo: float, hi: float = np.inf) -> None:
        matrix.append(np.fromiter(mask, dtype=float, count=n))
        lower.append(lo)
        upper.append(hi)

    constraint((True for _ in candidates), ROWS, ROWS)
    for lane, count in EXACT_LANE_ROWS.items():
        constraint((row.lane == lane for row in candidates), count, count)
    for (lane, stratum), minimum in SOURCE_STRATUM_MINIMUMS.items():
        constraint(
            (row.lane == lane and row.source_stratum() == stratum for row in candidates),
            minimum,
        )
    for name, minimum in COMPLEXITY_MINIMUMS.items():
        constraint((_complexity(row, name) for row in candidates), minimum)
    cells_by_row = [_candidate_cells(row) for row in candidates]
    for family, minimum in LOW_LEVEL_FAMILY_MINIMUMS.items():
        constraint((family in cells for cells in cells_by_row), minimum)
    for primitive, minimum in DIRECT_PRIMITIVE_MINIMUMS.items():
        constraint((primitive in cells for cells in cells_by_row), minimum)

    result = milp(
        objective,
        integrality=np.ones(n),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(np.vstack(matrix), lower, upper),
    )
    if not result.success or result.x is None:
        raise ValueError(f"MILP selection failed: {result.message}")
    selected = [row for row, value in zip(candidates, result.x, strict=True) if value > 0.5]
    selected.sort(key=lambda row: (LANES.index(row.lane), row.lane_row_index))
    if len(selected) != ROWS or len({row.uuid for row in selected}) != ROWS:
        raise AssertionError("selection cardinality mismatch")
    return selected


def _metrics(rows: Sequence[base.Candidate]) -> dict[str, Any]:
    cells = [_candidate_cells(row) for row in rows]
    return {
        "rows": len(rows),
        "lanes": dict(sorted(collections.Counter(row.lane for row in rows).items())),
        "source_strata": dict(
            sorted(collections.Counter(f"{row.lane}:{row.source_stratum()}" for row in rows).items())
        ),
        "low_level_family_row_presence": {
            family: sum(family in row_cells for row_cells in cells)
            for family in (*LOW_LEVEL_FAMILY_MINIMUMS, *DIRECT_PRIMITIVE_MINIMUMS)
        },
        "structural_complexity": {
            name: sum(_complexity(row, name) for row in rows) for name in COMPLEXITY_MINIMUMS
        },
        "unique_reference_sha256": len({row.reference_sha256 for row in rows}),
        "unique_normalized_ast_sha256": len({row.normalized_ast_sha256 for row in rows}),
        "manifest_semantic_skeletons": len(
            {
                row.source_manifest.get("model_semantic_skeleton_sha256")
                for row in rows
                if row.source_manifest.get("model_semantic_skeleton_sha256")
            }
        ),
        "template_skeletons": len(
            {
                row.source_manifest.get("template_skeleton")
                for row in rows
                if row.source_manifest.get("template_skeleton")
            }
        ),
    }


def _reference_hashes(paths: Sequence[Path]) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    exact: set[str] = set()
    normalized: set[str] = set()
    sources = []
    for path in paths:
        rows = 0
        for record in pq.read_table(path, columns=["reward_model"]).column(0).to_pylist():
            code = record["ground_truth"]
            exact.add(_sha256_text(code))
            normalized.add(_sha256_text(ast.dump(ast.parse(code), include_attributes=False)))
            rows += 1
        sources.append({"path": str(path.resolve()), "sha256": _sha256_file(path), "rows": rows})
    return exact, normalized, sources


def _dedup_audit(
    rows: Sequence[base.Candidate], decontamination_roots: Sequence[Path]
) -> dict[str, Any]:
    exact, normalized, sources = _reference_hashes(decontamination_roots)
    exact_hits = sorted(row.uuid for row in rows if row.reference_sha256 in exact)
    ast_hits = sorted(row.uuid for row in rows if row.normalized_ast_sha256 in normalized)
    if exact_hits or ast_hits:
        raise ValueError(f"decontamination collision: exact={exact_hits}, normalized_ast={ast_hits}")
    return {
        "within_selection": {
            "rows": len(rows),
            "unique_reference_sha256": len({row.reference_sha256 for row in rows}),
            "unique_normalized_ast_sha256": len({row.normalized_ast_sha256 for row in rows}),
        },
        "against_roots": {
            "exact_reference_matches": 0,
            "normalized_ast_matches": 0,
            "sources": sources,
        },
        "remaining_gate": (
            "canonical-graph/near-duplicate dedup is required before training approval; "
            "template variants are intentionally retained in this review-only canary"
        ),
    }


def _write_report(
    path: Path,
    metrics: Mapping[str, Any],
    ledger_summary: Mapping[str, int],
    dedup: Mapping[str, Any],
) -> None:
    lines = [
        "# KernelBench low-level operator / high-complexity canary",
        "",
        "本集合只把 KernelBench 的 low-level operator family 和直接 `scaled_dot_product_attention` primitive 当作覆盖目标。KernelBench 中的 architecture helper、container、LSTM/GRU/MHA/Transformer 等 high-level 组合结构不参与选择，避免把测试集的组合结构当成生成模板。",
        "",
        "程序形式是独立硬约束：源码行数、`forward` 调用数、registered module 与多个顶层 class 均保留。多 class 是形式覆盖，不表示复刻 KernelBench 的 high-level architecture。",
        "",
        "| Source lane | Rows |",
        "| --- | ---: |",
    ]
    lines.extend(f"| `{lane}` | {metrics['lanes'].get(lane, 0)} |" for lane in LANES)
    lines.extend(["", "| Low-level coverage cell | Rows | Minimum |", "| --- | ---: | ---: |"])
    for name, minimum in {**LOW_LEVEL_FAMILY_MINIMUMS, **DIRECT_PRIMITIVE_MINIMUMS}.items():
        lines.append(f"| `{name}` | {metrics['low_level_family_row_presence'][name]} | {minimum} |")
    lines.extend(["", "| Formal complexity | Rows | Minimum |", "| --- | ---: | ---: |"])
    for name, minimum in COMPLEXITY_MINIMUMS.items():
        lines.append(f"| `{name}` | {metrics['structural_complexity'][name]} | {minimum} |")
    lines.extend(
        [
            "",
            f"KernelBench 静态账本含 {ledger_summary['unique_tokens']} 个唯一调用 token；其中 {ledger_summary.get('low_level_operator', 0)} 个归为 low-level、{ledger_summary.get('high_level_composition', 0)} 个归为 high-level composition、{ledger_summary.get('non_target_plumbing_or_control', 0)} 个归为 plumbing/control。token 明细见 `kernelbench_operator_ledger.json`。主验收以 family/primitive support 为单位，不把同一底层算子的 module/functional/维度别名机械当作不同生成图。",
            "",
            f"去重审计：集合内 exact reference 与 normalized AST 均为 {dedup['within_selection']['rows']}/{dedup['within_selection']['rows']} 唯一；对 KernelBench、活动训练集和 review union 的 exact/normalized-AST 命中均为 0。canonical graph / near-duplicate 去重仍是进入训练前的独立 gate。",
            "",
            "## 明确遗留问题",
            "",
            "- 本批没有覆盖 KernelBench 的 high-level 组合结构；这是按本轮边界主动排除的遗留问题，不计作 low-level operator coverage 失败。若以后要覆盖，必须重新做污染边界和独立模板设计。",
            "- shape 分布仍交给既有 shape-expansion pipeline，本批不以 KernelBench shape 比例为目标。",
            "- 当前产物保持 `review_only=true`、`training_approved=false`；训练前仍需 canonical-graph/near-duplicate dedup 和混合权重实验。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def select(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(f"output directory already exists: {args.output_dir}")
    excluded = {
        line.strip()
        for line in args.exclude_uuid_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    lane_inputs = {
        lane: (getattr(args, f"{lane}_parquet"), getattr(args, f"{lane}_manifest"))
        for lane in LANES
    }
    candidates, filter_audit = _load_sources(lane_inputs, excluded)
    selected = _choose(candidates)
    ledger, ledger_summary = _kernelbench_ledger(args.kernelbench)
    dedup = _dedup_audit(selected, args.decontamination_root)
    metrics = _metrics(selected)
    for lane, expected in EXACT_LANE_ROWS.items():
        if metrics["lanes"].get(lane) != expected:
            raise AssertionError(f"lane count mismatch: {lane}")
    for name, minimum in COMPLEXITY_MINIMUMS.items():
        if metrics["structural_complexity"][name] < minimum:
            raise AssertionError(f"complexity minimum mismatch: {name}")
    for name, minimum in {**LOW_LEVEL_FAMILY_MINIMUMS, **DIRECT_PRIMITIVE_MINIMUMS}.items():
        if metrics["low_level_family_row_presence"][name] < minimum:
            raise AssertionError(f"coverage minimum mismatch: {name}")

    args.output_dir.mkdir(parents=True)
    selected_path = args.output_dir / "selected.parquet"
    manifest_path = args.output_dir / "selected.manifest.jsonl"
    selection_path = args.output_dir / "selection.jsonl"
    uuid_path = args.output_dir / "selected.uuids.txt"
    ledger_path = args.output_dir / "kernelbench_operator_ledger.json"
    dedup_path = args.output_dir / "dedup_audit.json"
    report_path = args.output_dir / "selection_report.md"
    pq.write_table(pa.Table.from_pylist([row.source_row for row in selected]), selected_path, compression="zstd")
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
                        "operator_families": sorted(_candidate_cells(row)),
                        "forward_call_count": row.forward_call_count,
                        "source_line_count": row.source_line_count,
                        "init_nn_constructor_count": row.init_nn_constructor_count,
                        "top_level_class_count": row.top_level_class_count,
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
    selection_path.write_text(
        "".join(
            _canonical_json(
                {
                    "canary_row_index": index,
                    "uuid": row.uuid,
                    "source_lane": row.lane,
                    "source_stratum": row.source_stratum(),
                    "low_level_cells": sorted(_candidate_cells(row)),
                    "operator_count": row.operator_count,
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
    ledger_path.write_text(
        json.dumps({"summary": ledger_summary, "tokens": ledger}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    dedup_path.write_text(json.dumps(dedup, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_report(report_path, metrics, ledger_summary, dedup)
    summary = {
        "contract_version": CONTRACT_VERSION,
        "rows": ROWS,
        "selection_policy": {
            "target": "KernelBench low-level operator support plus independent program-form complexity",
            "exact_lane_rows": EXACT_LANE_ROWS,
            "complexity_minimum_rows": COMPLEXITY_MINIMUMS,
            "low_level_family_minimum_rows": LOW_LEVEL_FAMILY_MINIMUMS,
            "direct_primitive_minimum_rows": DIRECT_PRIMITIVE_MINIMUMS,
            "source_stratum_minimum_rows": {
                f"{lane}:{stratum}": value
                for (lane, stratum), value in SOURCE_STRATUM_MINIMUMS.items()
            },
            "explicit_non_target": "KernelBench high-level composition structures",
            "shape_policy": "deferred to shape-expansion pipeline",
            "selection_seed_version": SELECTION_SEED_VERSION,
        },
        "metrics": metrics,
        "kernelbench_operator_ledger_summary": ledger_summary,
        "source_filter_audit": filter_audit,
        "dedup_audit": dedup,
        "source": {
            "lanes": {
                lane: {
                    "parquet": {"path": str(paths[0].resolve()), "sha256": _sha256_file(paths[0])},
                    "manifest": {"path": str(paths[1].resolve()), "sha256": _sha256_file(paths[1])},
                }
                for lane, paths in lane_inputs.items()
            },
            "kernelbench": [
                {"path": str(path.resolve()), "sha256": _sha256_file(path)} for path in args.kernelbench
            ],
            "selector": {
                "path": str(Path(__file__).resolve()),
                "sha256": _sha256_file(Path(__file__).resolve()),
            },
            "selector_dependencies": [
                {"path": str(path.resolve()), "sha256": _sha256_file(path)}
                for path in (
                    _REPO_ROOT / "tools/data/cleaning/complexity.py",
                    _REPO_ROOT / "tools/data/synthesize/profile_prompt_tvm_distribution.py",
                    _REPO_ROOT / "tools/data/synthesize/select_four_lane_kernelbench_canary.py",
                )
            ],
            "exclusion_audit": {
                "path": str(args.exclude_uuid_file.resolve()),
                "sha256": _sha256_file(args.exclude_uuid_file),
            },
        },
        "selection_environment": {
            "git_commit": _git_commit(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pyarrow": pa.__version__,
            "milp_backend": "scipy.optimize.milp (HiGHS)",
        },
        "runtime_evidence_status": "requires unified H20 reference, paired sensitivity, and declared-op liveness",
        "remaining_issues": [
            "KernelBench high-level composition structures intentionally not covered",
            "canonical-graph/near-duplicate dedup required before training approval",
            "shape expansion deferred",
        ],
        "review_only": True,
        "training_approved": False,
    }
    summary_path = args.output_dir / "summary.json"
    artifacts = (selected_path, manifest_path, selection_path, uuid_path, ledger_path, dedup_path, report_path)
    summary["artifacts"] = {
        path.name: {"path": str(path.resolve()), "sha256": _sha256_file(path)} for path in artifacts
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    for lane in LANES:
        parser.add_argument(f"--{lane}-parquet", type=Path, required=True)
        parser.add_argument(f"--{lane}-manifest", type=Path, required=True)
    parser.add_argument("--kernelbench", type=Path, action="append", required=True)
    parser.add_argument("--decontamination-root", type=Path, action="append", required=True)
    parser.add_argument("--exclude-uuid-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    summary = select(_parse_args())
    print(_canonical_json({"rows": summary["rows"], "metrics": summary["metrics"]}))


if __name__ == "__main__":
    main()
