#!/usr/bin/env python3
"""Fail-closed static quality audit for materialized CSP-DAG datasets.

The audit deliberately does not execute reference programs.  It binds the
selected parquet, manifest, graph lowering, and producing summary before
reporting line-accounting and graph-quality diagnostics.  ``training_gate`` is
therefore a static gate, not a training approval.
"""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import math
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.synthesize.csp_dag_method import csp_dag_static_binding
from tools.data.synthesize.csp_dag_method import generate_csp_dag as generator

CONTRACT = "csp_dag_static_quality_audit_v1"
ORIGINAL_LINE_BUCKETS = ("20-34", "35-49", "50-74", ">=75")
ALL_LINE_BUCKETS = ("<20", *ORIGINAL_LINE_BUCKETS)
REQUIRED_CODE_ONLY_BUCKETS = ORIGINAL_LINE_BUCKETS
PROVENANCE_PREFIXES = (
    "# graph node ",
    "# graph edge: ",
    "# shape constraint: ",
    "# input domain: ",
    "# symbolic equality: ",
    "# graph CSP: ",
)
POINTWISE_RULES = frozenset({"activation", "elementwise"})
POINTWISE_ALLOWED_RULES = POINTWISE_RULES | {"loss"}
NON_POINTWISE_RULES = frozenset({"conv", "linear", "normalization", "pool", "reduction", "layout", "indexing", "sdpa"})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    result = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not all(isinstance(item, dict) for item in result):
        raise ValueError(f"manifest contains a non-object row: {path}")
    return result


def _line_bucket(lines: int) -> str:
    if lines < 20:
        return "<20"
    if lines <= 34:
        return "20-34"
    if lines <= 49:
        return "35-49"
    if lines <= 74:
        return "50-74"
    return ">=75"


def _line_metrics(code: str) -> dict[str, Any]:
    """Separate physical executable lines from blank/provenance-only lines.

    A line containing executable code and an inline comment is intentionally a
    code-only line.  Only whole-line comments participate in provenance checks.
    """

    counts: collections.Counter[str] = collections.Counter()
    provenance: collections.Counter[str] = collections.Counter()
    unexplained: list[dict[str, Any]] = []
    for line_number, line in enumerate(code.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            counts["blank"] += 1
        elif stripped.startswith("#"):
            counts["comment_only"] += 1
            prefix = next((item for item in PROVENANCE_PREFIXES if stripped.startswith(item)), None)
            if prefix is None:
                unexplained.append({"line": line_number, "text": stripped})
            else:
                provenance[prefix.removeprefix("# ").rstrip(": ")] += 1
        else:
            counts["code_only"] += 1
    return {
        "physical": len(code.splitlines()),
        "code_only": counts["code_only"],
        "blank": counts["blank"],
        "comment_only": counts["comment_only"],
        "provenance_comment_types": dict(sorted(provenance.items())),
        "unexplained_comments": unexplained,
    }


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _is_zero_index(expression: ast.AST, assignments: dict[str, ast.AST]) -> bool:
    """Recognize statically all-zero tensor index constructions."""

    seen: set[str] = set()
    while isinstance(expression, ast.Name) and expression.id in assignments:
        if expression.id in seen:
            return False
        seen.add(expression.id)
        expression = assignments[expression.id]
    if not isinstance(expression, ast.Call):
        return False
    name = _call_name(expression.func)
    if name.endswith(("zeros", "zeros_like")):
        return True
    if name.endswith(("full", "full_like")):
        value = (
            expression.args[1]
            if len(expression.args) > 1
            else next((keyword.value for keyword in expression.keywords if keyword.arg == "fill_value"), None)
        )
        return isinstance(value, ast.Constant) and value.value == 0
    return False


def _is_singleton_last_axis(expression: ast.AST) -> bool:
    """Return whether an expression is exactly ``base[..., :1]``."""

    if not isinstance(expression, ast.Subscript):
        return False
    slice_node = expression.slice
    if not (
        isinstance(slice_node, ast.Tuple)
        and len(slice_node.elts) == 2
        and isinstance(slice_node.elts[0], ast.Constant)
        and slice_node.elts[0].value is Ellipsis
        and isinstance(slice_node.elts[1], ast.Slice)
    ):
        return False
    last = slice_node.elts[1]
    return last.lower is None and isinstance(last.upper, ast.Constant) and last.upper.value == 1 and last.step is None


def _is_proven_singleton_zero_index(expression: ast.AST, assignments: dict[str, ast.AST]) -> bool:
    """Recognize ``zeros_like(base[..., :1])``, one last-axis index per row."""

    seen: set[str] = set()
    while isinstance(expression, ast.Name) and expression.id in assignments:
        if expression.id in seen:
            return False
        seen.add(expression.id)
        expression = assignments[expression.id]
    return (
        isinstance(expression, ast.Call)
        and _call_name(expression.func).endswith("zeros_like")
        and bool(expression.args)
        and _is_singleton_last_axis(expression.args[0])
    )


class _UnsafeScatterVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.assignments: dict[str, ast.AST] = {}
        self.findings: list[dict[str, Any]] = []

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            self.assignments[node.targets[0].id] = node.value
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:  # noqa: N802
        if isinstance(node.target, ast.Name) and node.value is not None:
            self.assignments[node.target.id] = node.value
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        name = _call_name(node.func)
        is_method = isinstance(node.func, ast.Attribute) and node.func.attr in {"scatter", "scatter_"}
        is_torch = name in {"torch.scatter", "torch.scatter_"}
        if is_method or is_torch:
            positional_index = 1 if is_method else 2
            index = (
                node.args[positional_index]
                if len(node.args) > positional_index
                else next((keyword.value for keyword in node.keywords if keyword.arg == "index"), None)
            )
            if (
                index is not None
                and _is_zero_index(index, self.assignments)
                and not _is_proven_singleton_zero_index(index, self.assignments)
            ):
                self.findings.append(
                    {
                        "line": node.lineno,
                        "column": node.col_offset,
                        "call": name,
                        "reason": "statically all-zero scatter index can contain duplicate destinations",
                    }
                )
        self.generic_visit(node)


def _unsafe_duplicate_index_scatters(code: str) -> list[dict[str, Any]]:
    tree = ast.parse(code)
    visitor = _UnsafeScatterVisitor()
    visitor.visit(tree)
    return visitor.findings


def _scatter_call_count(code: str) -> int:
    return sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"scatter", "scatter_"}
        for node in ast.walk(ast.parse(code))
    )


def _pointwise_heavy(manifest: dict[str, Any]) -> bool:
    """Return whether a high-node terminal-loss graph is pointwise heavy."""

    nodes = manifest.get("graph", {}).get("typed_graph", {}).get("nodes", [])
    if not isinstance(nodes, list) or len(nodes) < 16:
        return False
    rules = [node.get("rule") for node in nodes if isinstance(node, dict)]
    return (
        len(rules) == len(nodes)
        and rules[-1] == "loss"
        and set(rules).issubset(POINTWISE_ALLOWED_RULES)
        and not (set(rules) & NON_POINTWISE_RULES)
        and sum(rule in POINTWISE_RULES for rule in rules) / len(rules) >= 0.75
    )


def _graph_source_binding(code: str, manifest: dict[str, Any]) -> dict[str, Any]:
    """Bind source to graph through the shared pure-static AST contract.

    ``typed_graph_context`` recomputes the canonical typed-graph SHA and
    compares manifest edges in ordered/multiset form to node predecessors.
    ``source_lowering_checks`` reconstructs each expression with
    ``generator._operation_expression`` and compares ASTs, making formatting
    differences irrelevant while binding rules, variants, modules, inputs,
    predecessor uses, assignments, and terminal return.
    """

    graph_checks, nodes, rank, shape = csp_dag_static_binding.typed_graph_context(manifest)
    lowering_checks = csp_dag_static_binding.source_lowering_checks(code, manifest, nodes, rank, shape)
    checks = {**graph_checks, **lowering_checks}
    failed_checks = sorted(name for name, passed in checks.items() if passed is not True)
    return {
        "bound": not failed_checks,
        "checks": checks,
        "failed_checks": failed_checks,
        "non_required_checks": [],
        "typed_node_count": len(nodes) if nodes is not None else None,
        "input_rank": rank,
        "semantic_multiclass": {
            "top_level_class_policy": manifest.get("top_level_class_policy"),
            "status": manifest.get("semantic_multiclass_status"),
            "identity_wrapper_forbidden": csp_dag_static_binding.uses_single_model_policy(manifest),
            "bound": (
                checks.get("multiclass_wrapper_bound", False) and checks.get("semantic_multiclass_policy_bound", False)
            ),
        },
    }


def _high_complexity_heterogeneity(manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify the current generator's non-pointwise and merge-added contract."""

    if manifest.get("contract") != generator.CONTRACT:
        return {"applicable": False, "reason": "not the current generator contract", "failed_checks": []}
    raw_nodes = manifest.get("graph", {}).get("typed_graph", {}).get("nodes", [])
    if not isinstance(raw_nodes, list):
        return {
            "applicable": True,
            "checks": {"typed_nodes_available": False},
            "failed_checks": ["typed_nodes_available"],
        }
    node_count = len(raw_nodes)
    actual_nonpointwise = sum(
        isinstance(node, dict) and node.get("rule") in generator._NONPOINTWISE_RULES for node in raw_nodes
    )
    required_nonpointwise = (
        max(3, math.ceil(generator.HIGH_COMPLEXITY_MIN_NONPOINTWISE_FRACTION * node_count))
        if node_count >= generator.HIGH_COMPLEXITY_NODE_THRESHOLD
        else 0
    )
    merge_count = manifest.get("sink_merge_node_count")
    if type(merge_count) is not int or merge_count < 0 or merge_count > node_count:
        normalized_merge_count = None
    else:
        normalized_merge_count = merge_count
    expected_ratio = (
        round(normalized_merge_count / node_count, 6) if node_count and normalized_merge_count is not None else None
    )
    maximum_merge_count = math.floor(generator.SINK_MERGE_NODE_RATIO_CAP * node_count)
    suffix_end = node_count - int(
        bool(raw_nodes) and isinstance(raw_nodes[-1], dict) and raw_nodes[-1].get("rule") == "loss"
    )
    suffix_is_merge_lowering = (
        normalized_merge_count is not None
        and normalized_merge_count <= suffix_end
        and all(
            isinstance(node, dict) and node.get("rule") == "elementwise" and node.get("variant") in {"add", "mix"}
            for node in raw_nodes[suffix_end - normalized_merge_count : suffix_end]
        )
    )
    checks = {
        "declared_nonpointwise_count_bound": (
            type(manifest.get("high_complexity_nonpointwise_rule_count")) is int
            and manifest["high_complexity_nonpointwise_rule_count"] == actual_nonpointwise
        ),
        "declared_nonpointwise_minimum_bound": (
            type(manifest.get("high_complexity_minimum_nonpointwise_rule_count")) is int
            and manifest["high_complexity_minimum_nonpointwise_rule_count"] == required_nonpointwise
        ),
        "high_complexity_nonpointwise_minimum_met": actual_nonpointwise >= required_nonpointwise,
        "sink_merge_node_count_valid": normalized_merge_count is not None,
        "sink_merge_node_ratio_bound": (
            expected_ratio is not None
            and isinstance(manifest.get("sink_merge_node_ratio"), (int, float))
            and not isinstance(manifest.get("sink_merge_node_ratio"), bool)
            and math.isclose(float(manifest["sink_merge_node_ratio"]), expected_ratio, abs_tol=1e-6)
        ),
        "sink_merge_node_ratio_cap_bound": (
            isinstance(manifest.get("sink_merge_node_ratio_cap"), (int, float))
            and not isinstance(manifest.get("sink_merge_node_ratio_cap"), bool)
            and math.isclose(
                float(manifest["sink_merge_node_ratio_cap"]),
                generator.SINK_MERGE_NODE_RATIO_CAP,
                abs_tol=1e-12,
            )
        ),
        "sink_merge_node_ratio_within_cap": (
            normalized_merge_count is not None and normalized_merge_count <= maximum_merge_count
        ),
        "sink_merge_suffix_lowering_bound": suffix_is_merge_lowering,
    }
    return {
        "applicable": True,
        "node_count": node_count,
        "nonpointwise_rule_count": actual_nonpointwise,
        "minimum_nonpointwise_rule_count": required_nonpointwise,
        "sink_merge_node_count": normalized_merge_count,
        "sink_merge_node_ratio": expected_ratio,
        "sink_merge_node_ratio_cap": generator.SINK_MERGE_NODE_RATIO_CAP,
        "checks": checks,
        "failed_checks": sorted(name for name, passed in checks.items() if passed is not True),
    }


def _artifact_binding(summary: dict[str, Any], parquet_path: Path, manifest_path: Path) -> tuple[bool, list[str]]:
    errors: list[str] = []
    artifacts = summary.get("artifacts")
    if not isinstance(artifacts, dict):
        return False, ["summary.artifacts missing"]
    expected = {"selected": parquet_path, "selected_manifest": manifest_path}
    for name, actual_path in expected.items():
        declared = artifacts.get(name)
        if not isinstance(declared, dict):
            errors.append(f"summary.artifacts.{name} missing")
            continue
        declared_path = declared.get("path")
        declared_hash = declared.get("sha256")
        if not isinstance(declared_path, str) or Path(declared_path).resolve() != actual_path.resolve():
            errors.append(f"summary.artifacts.{name}.path does not bind input")
        if not isinstance(declared_hash, str) or declared_hash != _sha256_file(actual_path):
            errors.append(f"summary.artifacts.{name}.sha256 does not bind input")
    return not errors, errors


def _near_dedup_report(path: Path) -> tuple[dict[str, Any], list[str]]:
    if not path.is_file():
        return {}, [f"near-dedup retained summary missing: {path}"]
    summary = _read_json(path)
    strict = summary.get("strict_recommended_union")
    approximate = summary.get("approximate_graph_sensitivity", {}).get("thresholds")
    if not isinstance(strict, dict) or not isinstance(approximate, dict):
        return {}, ["near-dedup retained summary lacks strict/approximate results"]
    strict_pairs = strict.get("pair_count")
    if not isinstance(strict_pairs, int):
        return {}, ["near-dedup strict pair_count missing"]
    return (
        {
            "path": str(path.resolve()),
            "sha256": _sha256_file(path),
            "strict": {
                "pair_count": strict_pairs,
                "cluster_count": strict.get("cluster_count"),
                "clustered_rows": strict.get("clustered_rows"),
                "effective_rows": strict.get("effective_rows"),
            },
            "approximate_thresholds": {
                str(threshold): {
                    "approximate_pair_count": value.get("approximate_pair_count"),
                    "union_cluster_count": value.get("union_cluster_count"),
                    "union_clustered_rows": value.get("union_clustered_rows"),
                    "effective_rows": value.get("effective_rows"),
                }
                for threshold, value in sorted(approximate.items())
                if isinstance(value, dict)
            },
        },
        [],
    )


def audit_quality(
    run_dir: Path,
    *,
    parquet_path: Path | None = None,
    manifest_path: Path | None = None,
    summary_path: Path | None = None,
    retained_near_dedup_summary_path: Path | None = None,
) -> dict[str, Any]:
    """Perform a fail-closed audit without writing dataset artifacts."""

    run_dir = run_dir.resolve()
    parquet_path = (parquet_path or run_dir / "selected.parquet").resolve()
    manifest_path = (manifest_path or run_dir / "selected.manifest.jsonl").resolve()
    summary_path = (summary_path or run_dir / "summary.json").resolve()
    if retained_near_dedup_summary_path is None:
        retained_candidate = run_dir / "near_dedup" / "retained.summary.json"
        retained_near_dedup_summary_path = (
            retained_candidate if retained_candidate.is_file() else run_dir / "near_dedup" / "summary.json"
        )
    retained_near_dedup_summary_path = retained_near_dedup_summary_path.resolve()
    gate_failures: list[str] = []
    for path in (parquet_path, manifest_path, summary_path):
        if not path.is_file():
            gate_failures.append(f"required input missing: {path}")
    if gate_failures:
        return {
            "contract": CONTRACT,
            "training_gate_passed": False,
            "gate_failures": gate_failures,
            "source_binding": {"run_dir": str(run_dir)},
        }

    summary = _read_json(summary_path)
    binding_ok, binding_errors = _artifact_binding(summary, parquet_path, manifest_path)
    if not binding_ok:
        gate_failures.extend(binding_errors)
    rows = pq.read_table(parquet_path).to_pylist()
    manifests = _load_manifest(manifest_path)
    if len(rows) != len(manifests):
        gate_failures.append("parquet and manifest row counts differ")

    physical = collections.Counter[str]()
    provenance = collections.Counter[str]()
    code_only_bucket_counts = collections.Counter[str]()
    nominal_to_code = {nominal: collections.Counter[str]() for nominal in ALL_LINE_BUCKETS}
    unsafe_scatters: list[dict[str, Any]] = []
    scatter_rows = 0
    scatter_calls = 0
    pointwise_heavy_uuids: list[str] = []
    unexplained_comments: list[dict[str, Any]] = []
    row_binding_errors: list[dict[str, Any]] = []
    graph_source_binding_failures: list[dict[str, Any]] = []
    heterogeneity_failures: list[dict[str, Any]] = []
    graph_source_bound_rows = 0
    for position, (row, manifest) in enumerate(zip(rows, manifests, strict=False)):
        try:
            code = row["reward_model"]["ground_truth"]
            row_uuid = row["extra_info"]["uuid"]
            manifest_uuid = manifest["uuid"]
            declared_hash = manifest["reference_sha256"]
            nominal_lines = manifest["complexity"]["source_line_count"]
        except (KeyError, TypeError) as error:
            row_binding_errors.append({"position": position, "error": f"missing required field: {error}"})
            continue
        if not all(isinstance(value, str) for value in (code, row_uuid, manifest_uuid, declared_hash)):
            row_binding_errors.append({"position": position, "error": "invalid code/uuid/reference hash type"})
            continue
        if row_uuid != manifest_uuid or _sha256_text(code) != declared_hash:
            row_binding_errors.append(
                {
                    "position": position,
                    "uuid": row_uuid,
                    "error": "row UUID or reference SHA is not bound",
                    "uuid_bound": row_uuid == manifest_uuid,
                    "reference_hash_bound": _sha256_text(code) == declared_hash,
                }
            )
        metrics = _line_metrics(code)
        physical.update(
            {
                "physical": metrics["physical"],
                "code_only": metrics["code_only"],
                "blank": metrics["blank"],
                "comment_only": metrics["comment_only"],
            }
        )
        provenance.update(metrics["provenance_comment_types"])
        code_bucket = _line_bucket(metrics["code_only"])
        code_only_bucket_counts[code_bucket] += 1
        if type(nominal_lines) is not int or nominal_lines < 0:
            row_binding_errors.append(
                {"position": position, "uuid": row_uuid, "error": "source_line_count missing or invalid"}
            )
        else:
            nominal_bucket = _line_bucket(nominal_lines)
            nominal_to_code[nominal_bucket][code_bucket] += 1
            if metrics["physical"] != nominal_lines:
                row_binding_errors.append(
                    {
                        "position": position,
                        "uuid": row_uuid,
                        "error": "source_line_count is not physical line count",
                        "declared": nominal_lines,
                        "actual": metrics["physical"],
                    }
                )
        try:
            row_scatter_calls = _scatter_call_count(code)
            scatter_calls += row_scatter_calls
            scatter_rows += row_scatter_calls > 0
            scatter_findings = _unsafe_duplicate_index_scatters(code)
        except SyntaxError as error:
            row_binding_errors.append(
                {"position": position, "uuid": row_uuid, "error": f"source AST parse failed: {error.msg}"}
            )
            scatter_findings = []
        for finding in scatter_findings:
            unsafe_scatters.append({"position": position, "uuid": row_uuid, **finding})
        graph_binding = _graph_source_binding(code, manifest)
        if graph_binding["bound"]:
            graph_source_bound_rows += 1
        else:
            detail = {"position": position, "uuid": row_uuid, **graph_binding}
            graph_source_binding_failures.append(detail)
            row_binding_errors.append(
                {
                    "position": position,
                    "uuid": row_uuid,
                    "error": "typed graph/source lowering binding failed",
                    "failed_checks": graph_binding["failed_checks"],
                }
            )
        heterogeneity = _high_complexity_heterogeneity(manifest)
        if heterogeneity.get("applicable") and heterogeneity["failed_checks"]:
            heterogeneity_failures.append({"position": position, "uuid": row_uuid, **heterogeneity})
        if _pointwise_heavy(manifest):
            pointwise_heavy_uuids.append(row_uuid)
        unexplained_comments.extend(
            {"position": position, "uuid": row_uuid, **comment} for comment in metrics["unexplained_comments"]
        )

    if row_binding_errors:
        gate_failures.append("row-level parquet/manifest/source binding failure")
    if graph_source_binding_failures:
        gate_failures.append("typed graph/source lowering binding mismatch")
    if heterogeneity_failures:
        gate_failures.append("high-complexity heterogeneity or sink-merge contract mismatch")
    missing_code_only_buckets = [
        bucket for bucket in REQUIRED_CODE_ONLY_BUCKETS if rows and code_only_bucket_counts[bucket] == 0
    ]
    if missing_code_only_buckets:
        gate_failures.append("code-only line bucket coverage incomplete")
    if unsafe_scatters:
        gate_failures.append("unsafe duplicate-index scatter detected")
    if pointwise_heavy_uuids:
        gate_failures.append("pointwise-heavy terminal-loss graph detected")
    if unexplained_comments:
        gate_failures.append("unexplained comment-only source lines detected")
    near_dedup, near_errors = _near_dedup_report(retained_near_dedup_summary_path)
    if near_errors:
        gate_failures.extend(near_errors)
    elif near_dedup["strict"]["pair_count"] != 0:
        gate_failures.append("strict retained near-duplicates detected")

    return {
        "contract": CONTRACT,
        "training_gate_passed": not gate_failures,
        "gate_failures": gate_failures,
        "source_binding": {
            "run_dir": str(run_dir),
            "parquet": {"path": str(parquet_path), "sha256": _sha256_file(parquet_path)},
            "manifest": {"path": str(manifest_path), "sha256": _sha256_file(manifest_path)},
            "summary": {"path": str(summary_path), "sha256": _sha256_file(summary_path)},
            "summary_artifacts_bound": binding_ok,
            "binding_errors": binding_errors,
        },
        "rows": {"parquet": len(rows), "manifest": len(manifests)},
        "physical_line_metrics": {
            **dict(physical),
            "provenance_comment_types": dict(sorted(provenance.items())),
            "comment_only_equals_provenance": physical["comment_only"] == sum(provenance.values()),
        },
        "code_only_line_buckets": {
            "counts": {bucket: code_only_bucket_counts[bucket] for bucket in ALL_LINE_BUCKETS},
            "required_nonempty": list(REQUIRED_CODE_ONLY_BUCKETS),
            "missing_required": missing_code_only_buckets,
            "below_20_allowed_but_not_required": True,
        },
        "nominal_to_code_only_buckets": {
            nominal: {bucket: counts[bucket] for bucket in ALL_LINE_BUCKETS}
            for nominal, counts in nominal_to_code.items()
        },
        "graph_source_binding": {
            "rows": len(rows),
            "bound_rows": graph_source_bound_rows,
            "failed_rows": len(graph_source_binding_failures),
            "check_contract": "csp_dag_static_binding typed graph + AST lowering checks",
            "multiclass_contract": "single Model class; semantic helper subgraph deferred; identity wrappers forbidden",
        },
        "graph_source_binding_failures": graph_source_binding_failures,
        "high_complexity_heterogeneity_failures": heterogeneity_failures,
        "unsafe_duplicate_index_scatters": unsafe_scatters,
        "scatter": {
            "rows": scatter_rows,
            "calls": scatter_calls,
            "unsafe_rows": len({item["position"] for item in unsafe_scatters}),
            "unsafe_calls": len(unsafe_scatters),
        },
        "pointwise_heavy": {
            "definition": (
                "node_count>=16; terminal rule=loss; every rule is activation, elementwise, or loss; "
                "activation+elementwise nodes / all nodes >= 0.75"
            ),
            "count": len(pointwise_heavy_uuids),
            "uuids": pointwise_heavy_uuids,
        },
        "unexplained_comments": unexplained_comments,
        "row_binding_errors": row_binding_errors,
        "near_dedup": near_dedup,
        "review_only": True,
        "training_approved": False,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--parquet", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--retained-near-dedup-summary", type=Path)
    parser.add_argument("--output", type=Path, help="optional audit JSON output path")
    args = parser.parse_args(argv)
    result = audit_quality(
        args.run_dir,
        parquet_path=args.parquet,
        manifest_path=args.manifest,
        summary_path=args.summary,
        retained_near_dedup_summary_path=args.retained_near_dedup_summary,
    )
    rendered = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")
    return 0 if result["training_gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
