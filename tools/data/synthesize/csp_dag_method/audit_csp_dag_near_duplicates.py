#!/usr/bin/env python3
"""Audit source and graph near-duplicates in the open CSP-DAG canary.

This is an audit, not a destructive filter.  It reports a conservative union
of source near-duplicates, exact semantic graphs, and exact graphs after
contracting generator variants that are proven value identities.  It also
reports threshold sensitivity for approximate graph similarity.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import networkx as nx
import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.cleaning.ast_similarity import _significant_trees, ast_structure_similarity
from tools.data.cleaning.similarity import python_token_set, token_jaccard
from tools.data.synthesize.csp_dag_method import generate_csp_dag as generator
from tools.data.synthesize.csp_dag_method import validate_csp_dag_canary as runtime_validator


CONTRACT = "open_csp_dag_near_duplicate_audit_v2"
SOURCE_TOKEN_THRESHOLD = 0.8
SOURCE_AST_THRESHOLD = 0.9
STRUCTURAL_THRESHOLDS = (0.75, 0.8, 0.85, 0.9)
_VERIFIED_IDENTITY_VARIANTS = frozenset(
    {
        ("layout", "chunk_cat"),
        ("layout", "transpose"),
    }
)
_RANK_SPECIALIZED_RULES = frozenset({"conv", "pool"})
_DEFAULT_RUN = (
    _REPO_ROOT
    / "local_artifacts/data/synthesize/csp_dag_low_level_canary200/run.coverage_high_complexity.v2"
)


@dataclasses.dataclass(frozen=True)
class Item:
    position: int
    uuid: str
    code: str
    manifest: dict[str, Any]
    normalized_ops: tuple[str, ...]


class DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root

    def clusters(self) -> list[list[int]]:
        grouped: dict[int, list[int]] = collections.defaultdict(list)
        for value in range(len(self.parent)):
            grouped[self.find(value)].append(value)
        return sorted(
            (values for values in grouped.values() if len(values) > 1),
            key=lambda values: (-len(values), values),
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _load_selected(run_dir: Path) -> list[Item]:
    rows = pq.read_table(run_dir / "selected.parquet").to_pylist()
    manifests = [
        json.loads(line)
        for line in (run_dir / "selected.manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != len(manifests):
        raise ValueError("selected parquet and manifest lengths differ")
    result = []
    for position, (row, manifest) in enumerate(zip(rows, manifests, strict=True)):
        uuid = row["extra_info"]["uuid"]
        if uuid != manifest["uuid"]:
            raise ValueError(f"selected identity mismatch at row {position}")
        ops = row["extra_info"]["ops"]
        if isinstance(ops, str):
            ops = json.loads(ops)
        if not isinstance(ops, list) or not all(isinstance(token, str) for token in ops):
            raise ValueError(f"invalid normalized ops at selected row {position}")
        result.append(
            Item(
                position,
                uuid,
                row["reward_model"]["ground_truth"],
                manifest,
                tuple(sorted(set(ops))),
            )
        )
    return result


def _load_pool_candidates() -> list[Any]:
    _, prompt_prefix = generator._template(generator._DEFAULT_TEMPLATE)
    raw = [generator._make_candidate(index, prompt_prefix) for index in range(generator.CANDIDATES)]
    unique: dict[str, Any] = {}
    for candidate in raw:
        unique.setdefault(candidate.manifest["graph"]["typed_graph_sha256"], candidate)
    return list(unique.values())


def _items_from_candidates(candidates: Sequence[Any]) -> list[Item]:
    return [
        Item(
            position,
            candidate.uuid,
            candidate.code,
            candidate.manifest,
            tuple(sorted(set(json.loads(candidate.row["extra_info"]["ops"])))),
        )
        for position, candidate in enumerate(candidates)
    ]


def _load_pool() -> list[Item]:
    return _items_from_candidates(_load_pool_candidates())


def _counter_jaccard(left: Iterable[Any], right: Iterable[Any]) -> float:
    left_count, right_count = collections.Counter(left), collections.Counter(right)
    keys = left_count.keys() | right_count.keys()
    denominator = sum(max(left_count[key], right_count[key]) for key in keys)
    return (
        sum(min(left_count[key], right_count[key]) for key in keys) / denominator
        if denominator
        else 1.0
    )


def _graph(item: Item, *, variant: bool) -> nx.DiGraph:
    graph = nx.DiGraph()
    typed = item.manifest["graph"]["typed_graph"]
    graph.add_node(-1, label=f"input:ndim={typed['input_rank']}")
    for node in typed["nodes"]:
        label = f"{node['rule']}:{node['variant']}" if variant else node["rule"]
        graph.add_node(node["index"], label=label)
    graph.add_edges_from(tuple(edge) for edge in typed["edges"])
    return graph


def _identity_normalized_graph(item: Item) -> nx.DiGraph:
    """Return an exact functional-structure graph for verified identities.

    ``layout:transpose`` is lowered as a transpose-contiguous-inverse-
    transpose chain and ``layout:chunk_cat`` concatenates the original chunks
    in order, so both preserve values exactly.  Explicit argument nodes retain
    fan-in multiplicity that a simple ``DiGraph`` edge list would otherwise
    collapse.  ``layout:binary_cat`` is commutative because its lowering is a
    stack followed by a sum.

    Input rank is ignored for dimension-agnostic operators.  It remains part
    of the label when a surviving Conv/Pool node selects a rank-specific
    callable such as Conv1d versus Conv2d.
    """

    typed = item.manifest["graph"]["typed_graph"]
    nodes = typed["nodes"]
    preserve_rank = any(node["rule"] in _RANK_SPECIALIZED_RULES for node in nodes)
    graph = nx.DiGraph()
    graph.add_node(
        "input",
        label=(f"input:ndim={typed['input_rank']}" if preserve_rank else "input:rank_agnostic"),
    )
    representative: dict[int, str] = {}
    created = 0
    for node in nodes:
        original_index = int(node["index"])
        predecessors = tuple(
            "input" if int(value) < 0 else representative[int(value)]
            for value in node["predecessors"]
        )
        key = (str(node["rule"]), str(node["variant"]))
        if key in _VERIFIED_IDENTITY_VARIANTS:
            if len(predecessors) != 1:
                raise ValueError(f"identity node {original_index} does not have one predecessor")
            representative[original_index] = predecessors[0]
            continue
        operation = f"operation:{created}"
        created += 1
        graph.add_node(operation, label=f"{key[0]}:{key[1]}")
        commutative = key == ("layout", "binary_cat")
        for slot, predecessor in enumerate(predecessors):
            argument = f"argument:{original_index}:{slot}"
            graph.add_node(argument, label="argument:any" if commutative else f"argument:{slot}")
            graph.add_edge(predecessor, argument)
            graph.add_edge(argument, operation)
        representative[original_index] = operation

    output_source = representative[int(nodes[-1]["index"])] if nodes else "input"
    graph.add_node("output", label="output")
    graph.add_edge(output_source, "output")
    return graph


def _isomorphic_pairs_for_graphs(graphs: Sequence[nx.DiGraph]) -> list[tuple[int, int]]:
    buckets: dict[str, list[tuple[int, nx.DiGraph]]] = collections.defaultdict(list)
    for index, graph in enumerate(graphs):
        digest = nx.weisfeiler_lehman_graph_hash(graph, node_attr="label", iterations=6)
        buckets[digest].append((index, graph))
    matches: list[tuple[int, int]] = []
    node_match = nx.algorithms.isomorphism.categorical_node_match("label", None)
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        for offset, (left_index, left) in enumerate(bucket):
            for right_index, right in bucket[:offset]:
                if nx.is_isomorphic(left, right, node_match=node_match):
                    matches.append((left_index, right_index))
    return sorted(matches)


def _isomorphic_pairs(items: Sequence[Item], *, variant: bool) -> list[tuple[int, int]]:
    return _isomorphic_pairs_for_graphs([_graph(item, variant=variant) for item in items])


def _identity_normalized_isomorphic_pairs(items: Sequence[Item]) -> list[tuple[int, int]]:
    return _isomorphic_pairs_for_graphs([_identity_normalized_graph(item) for item in items])


def _source_pairs(items: Sequence[Item]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[int]] = collections.defaultdict(list)
    tokens = []
    trees = []
    weights = []
    for index, item in enumerate(items):
        buckets[item.normalized_ops].append(index)
        tokens.append(python_token_set(item.code))
        significant = _significant_trees(item.code, "Model")
        trees.append(significant)
        weights.append(sum(tree.node_count * tree.weight for tree in significant))
    matches: list[dict[str, Any]] = []
    for signature, bucket in buckets.items():
        for offset, left in enumerate(bucket):
            for right in bucket[:offset]:
                token_score = token_jaccard(tokens[left], tokens[right])
                ast_score = ast_structure_similarity(
                    trees[left],
                    trees[right],
                    left_total_weight=weights[left],
                    right_total_weight=weights[right],
                )
                if token_score > SOURCE_TOKEN_THRESHOLD or ast_score > SOURCE_AST_THRESHOLD:
                    matches.append(
                        {
                            "left": left,
                            "right": right,
                            "operator_signature": list(signature),
                            "token_jaccard": token_score,
                            "ast_similarity": ast_score,
                        }
                    )
    return matches


def _features(item: Item) -> dict[str, Any]:
    typed = item.manifest["graph"]["typed_graph"]
    nodes = typed["nodes"]
    rule = {-1: "input", **{node["index"]: node["rule"] for node in nodes}}
    fine = {
        -1: "input",
        **{node["index"]: f"{node['rule']}:{node['variant']}" for node in nodes},
    }
    sequence = [fine[node["index"]] for node in nodes]
    edge_labels = [(fine[source], fine[target]) for source, target in typed["edges"]]
    edge_spans = []
    for source, target in typed["edges"]:
        distance = target - source if source >= 0 else target + 1
        span = "1" if distance == 1 else "2-3" if distance <= 3 else "4-7" if distance <= 7 else "8+"
        edge_spans.append((rule[source], rule[target], span))
    shape_csp = item.manifest["graph"]["shape_csp"]
    shape_contract = [f"ndim={typed['input_rank']}"]
    shape_contract.extend(
        f"divisible:{axis}:{divisor}"
        for axis, divisor in sorted(shape_csp.get("divisibility", {}).items())
    )
    shape_contract.extend(
        f"equality:{equality}" for equality in sorted(shape_csp.get("equalities", []))
    )
    return {
        "families": frozenset(item.manifest["actual_low_level_families"]),
        "node_count": len(nodes),
        "fine_nodes": sequence,
        "sequence_bigrams": list(zip(sequence, sequence[1:])),
        "operator_tokens": item.manifest["operator_signature"],
        "edge_labels": edge_labels,
        "edge_spans": edge_spans,
        "arities": [len(node["predecessors"]) for node in nodes],
        "shape_contract": shape_contract,
    }


def _structural_pairs(items: Sequence[Item]) -> list[dict[str, Any]]:
    features = [_features(item) for item in items]
    pairs: list[dict[str, Any]] = []
    for left in range(len(items)):
        for right in range(left):
            left_features, right_features = features[left], features[right]
            if left_features["families"] != right_features["families"]:
                continue
            count_ratio = min(left_features["node_count"], right_features["node_count"]) / max(
                left_features["node_count"], right_features["node_count"]
            )
            if count_ratio < 0.75:
                continue
            scores = {
                key: _counter_jaccard(left_features[key], right_features[key])
                for key in (
                    "fine_nodes",
                    "sequence_bigrams",
                    "operator_tokens",
                    "edge_labels",
                    "edge_spans",
                    "arities",
                    "shape_contract",
                )
            }
            composite = (
                0.18 * scores["fine_nodes"]
                + 0.14 * scores["sequence_bigrams"]
                + 0.18 * scores["operator_tokens"]
                + 0.18 * scores["edge_labels"]
                + 0.09 * scores["edge_spans"]
                + 0.04 * scores["arities"]
                + 0.09 * scores["shape_contract"]
                + 0.10 * count_ratio
            )
            if composite >= min(STRUCTURAL_THRESHOLDS):
                pairs.append(
                    {
                        "left": left,
                        "right": right,
                        "score": round(composite, 6),
                        "node_count_ratio": round(count_ratio, 6),
                        "components": {key: round(value, 6) for key, value in scores.items()},
                    }
                )
    return sorted(pairs, key=lambda pair: (-pair["score"], pair["left"], pair["right"]))


def _cluster_summary(size: int, pairs: Iterable[tuple[int, int]]) -> tuple[list[list[int]], int]:
    disjoint = DisjointSet(size)
    for left, right in pairs:
        disjoint.union(left, right)
    clusters = disjoint.clusters()
    retained = size - sum(len(cluster) - 1 for cluster in clusters)
    return clusters, retained


def _representative_key(item: Item) -> tuple[int, int, int, int, int, int]:
    """Prefer semantic breadth without retaining redundant identity padding."""
    nodes = item.manifest["graph"]["typed_graph"]["nodes"]
    semantic_nodes = [
        node
        for node in nodes
        if (str(node["rule"]), str(node["variant"])) not in _VERIFIED_IDENTITY_VARIANTS
    ]
    identity_count = len(nodes) - len(semantic_nodes)
    return (
        len(semantic_nodes),
        len({(node["rule"], node["variant"]) for node in semantic_nodes}),
        len({node["family"] for node in semantic_nodes}),
        -identity_count,
        -int(item.manifest["complexity"]["source_line_count"]),
        -int(item.manifest["candidate_index"]),
    )


def _retained_positions(items: Sequence[Item], clusters: Sequence[Sequence[int]]) -> set[int]:
    removed: set[int] = set()
    for cluster in clusters:
        keeper = max(cluster, key=lambda position: _representative_key(items[position]))
        removed.update(position for position in cluster if position != keeper)
    return set(range(len(items))) - removed


def _identity_pair(items: Sequence[Item], pair: dict[str, Any]) -> dict[str, Any]:
    left_typed = items[pair["left"]].manifest["graph"]["typed_graph"]
    right_typed = items[pair["right"]].manifest["graph"]["typed_graph"]
    return {
        **{key: value for key, value in pair.items() if key not in {"left", "right"}},
        "left_position": pair["left"],
        "left_uuid": items[pair["left"]].uuid,
        "left_ndim": left_typed["input_rank"],
        "left_node_count": len(left_typed["nodes"]),
        "right_position": pair["right"],
        "right_uuid": items[pair["right"]].uuid,
        "right_ndim": right_typed["input_rank"],
        "right_node_count": len(right_typed["nodes"]),
    }


def _audit(scope: str, items: Sequence[Item], output_dir: Path) -> dict[str, Any]:
    source = _source_pairs(items)
    exact_semantic = _isomorphic_pairs(items, variant=True)
    identity_normalized = _identity_normalized_isomorphic_pairs(items)
    exact_coarse = _isomorphic_pairs(items, variant=False)
    structural = _structural_pairs(items)

    source_edges = [(pair["left"], pair["right"]) for pair in source]
    strict_edges = sorted(set(source_edges) | set(exact_semantic) | set(identity_normalized))
    strict_clusters, strict_retained = _cluster_summary(len(items), strict_edges)
    semantic_clusters, semantic_retained = _cluster_summary(len(items), exact_semantic)
    normalized_clusters, normalized_retained = _cluster_summary(
        len(items), identity_normalized
    )
    coarse_clusters, coarse_retained = _cluster_summary(len(items), exact_coarse)

    sensitivity = {}
    for threshold in STRUCTURAL_THRESHOLDS:
        threshold_edges = [
            (pair["left"], pair["right"])
            for pair in structural
            if pair["score"] >= threshold
        ]
        clusters, retained = _cluster_summary(len(items), strict_edges + threshold_edges)
        sensitivity[str(threshold)] = {
            "approximate_pair_count": len(threshold_edges),
            "union_cluster_count": len(clusters),
            "union_clustered_rows": sum(len(cluster) for cluster in clusters),
            "effective_rows": retained,
        }

    pair_records = [
        *({"kind": "source_gate", **_identity_pair(items, pair)} for pair in source),
        *(
            {
                "kind": "exact_semantic_graph",
                **_identity_pair(items, {"left": left, "right": right}),
            }
            for left, right in exact_semantic
        ),
        *(
            {
                "kind": "identity_normalized_semantic_graph",
                **_identity_pair(items, {"left": left, "right": right}),
            }
            for left, right in identity_normalized
        ),
        *({"kind": "approximate_graph", **_identity_pair(items, pair)} for pair in structural[:100]),
    ]
    pairs_path = output_dir / f"{scope}.pairs.jsonl"
    pairs_path.write_text("".join(_canonical_json(row) + "\n" for row in pair_records))

    strict_retained_positions = _retained_positions(items, strict_clusters)
    clusters_path = output_dir / f"{scope}.strict_clusters.jsonl"
    clusters_path.write_text(
        "".join(
            _canonical_json(
                {
                    "cluster_index": index,
                    "positions": cluster,
                    "uuids": [items[position].uuid for position in cluster],
                    "keeper_position": max(
                        cluster, key=lambda position: _representative_key(items[position])
                    ),
                    "keeper_uuid": items[
                        max(cluster, key=lambda position: _representative_key(items[position]))
                    ].uuid,
                }
            )
            + "\n"
            for index, cluster in enumerate(strict_clusters)
        )
    )
    retained_path = output_dir / f"{scope}.strict_retained_uuids.jsonl"
    retained_path.write_text(
        "".join(
            _canonical_json(
                {
                    "position": position,
                    "candidate_index": items[position].manifest["candidate_index"],
                    "uuid": items[position].uuid,
                }
            )
            + "\n"
            for position in sorted(strict_retained_positions)
        )
    )

    summary = {
        "contract": CONTRACT,
        "scope": scope,
        "rows": len(items),
        "source_gate": {
            "operator_gate": "exact normalized extra_info.ops set (same gate as deduplicate_ops_candidates.py)",
            "token_jaccard_strictly_greater_than": SOURCE_TOKEN_THRESHOLD,
            "ast_similarity_strictly_greater_than": SOURCE_AST_THRESHOLD,
            "pair_count": len(source),
        },
        "exact_semantic_graph": {
            "definition": "directed graph isomorphism after dropping concrete dimension sizes, input mode, source form, and node indices while retaining ndim and rule:variant labels",
            "pair_count": len(exact_semantic),
            "cluster_count": len(semantic_clusters),
            "clustered_rows": sum(len(cluster) for cluster in semantic_clusters),
            "effective_rows": semantic_retained,
        },
        "identity_normalized_semantic_graph": {
            "definition": "exact directed functional-structure isomorphism after contracting verified value identities layout:transpose and layout:chunk_cat, preserving repeated argument slots, treating layout:binary_cat inputs as commutative, ignoring rank for dimension-agnostic operators, and retaining rank when Conv/Pool selects a rank-specific callable",
            "verified_identity_variants": sorted(
                f"{rule}:{variant}" for rule, variant in _VERIFIED_IDENTITY_VARIANTS
            ),
            "rank_specialized_rules": sorted(_RANK_SPECIALIZED_RULES),
            "pair_count": len(identity_normalized),
            "cluster_count": len(normalized_clusters),
            "clustered_rows": sum(len(cluster) for cluster in normalized_clusters),
            "effective_rows": normalized_retained,
        },
        "coarse_rule_graph_diagnostic": {
            "definition": "same isomorphism check retaining ndim and rule labels but dropping concrete variants; diagnostic only",
            "pair_count": len(exact_coarse),
            "cluster_count": len(coarse_clusters),
            "clustered_rows": sum(len(cluster) for cluster in coarse_clusters),
            "effective_rows_if_collapsed": coarse_retained,
        },
        "strict_recommended_union": {
            "definition": "source gate OR exact semantic graph OR identity-normalized exact semantic graph; approximate scores do not automatically delete rows",
            "keeper_policy": "within each connected component prefer non-identity semantic node count, semantic variant breadth, semantic family breadth, fewer verified identity nodes, fewer source lines, then earliest candidate index",
            "pair_count": len(strict_edges),
            "cluster_count": len(strict_clusters),
            "clustered_rows": sum(len(cluster) for cluster in strict_clusters),
            "effective_rows": strict_retained,
        },
        "approximate_graph_sensitivity": {
            "definition": "same exact family set and node-count ratio >=0.75; weighted multiset Jaccard over rule:variant nodes, sequence bigrams, operator tokens, labeled edges, edge spans, arities, and symbolic shape-contract features",
            "thresholds": sensitivity,
            "top_pairs": [_identity_pair(items, pair) for pair in structural[:20]],
        },
        "artifacts": {
            "pairs": {"path": str(pairs_path.resolve()), "sha256": _sha256_file(pairs_path)},
            "strict_clusters": {
                "path": str(clusters_path.resolve()),
                "sha256": _sha256_file(clusters_path),
            },
            "strict_retained_uuids": {
                "path": str(retained_path.resolve()),
                "sha256": _sha256_file(retained_path),
                "rows": len(strict_retained_positions),
            },
        },
    }
    summary_path = output_dir / f"{scope}.summary.json"
    _write_json(summary_path, summary)
    return summary


def _manual_review(
    output_dir: Path,
    selected: Sequence[Item],
    pool: Sequence[Item],
    summaries: Sequence[dict[str, Any]],
) -> Path:
    lookup = {item.uuid: item for item in [*selected, *pool]}
    lines = [
        "# CSP-DAG near-duplicate manual review",
        "",
        "This packet contains real pairs surfaced by the audit. The original artifacts were not mutated; retained and reselected outputs are separate derived artifacts.",
        "",
    ]
    seen: set[tuple[str, str]] = set()
    for summary in summaries:
        candidates = summary["approximate_graph_sensitivity"]["top_pairs"][:3]
        pair_rows = [
            json.loads(line)
            for line in Path(summary["artifacts"]["pairs"]["path"]).read_text().splitlines()
        ]
        candidates = [row for row in pair_rows if row["kind"] != "approximate_graph"] + candidates
        for pair in candidates:
            left_uuid, right_uuid = pair["left_uuid"], pair["right_uuid"]
            key = tuple(sorted((left_uuid, right_uuid)))
            if key in seen:
                continue
            seen.add(key)
            lines.extend(
                [
                    f"## {summary['scope']}: {left_uuid} vs {right_uuid}",
                    "",
                    f"- evidence: `{_canonical_json({k: v for k, v in pair.items() if k not in {'left_position', 'right_position'}})}`",
                    "",
                ]
            )
            for label, uuid in (("left", left_uuid), ("right", right_uuid)):
                item = lookup[uuid]
                lines.extend([f"### {label}: {uuid}", "", "```python", item.code.rstrip(), "```", ""])
            if len(seen) >= 8:
                path = output_dir / "manual_review.md"
                path.write_text("\n".join(lines) + "\n")
                return path
    path = output_dir / "manual_review.md"
    path.write_text("\n".join(lines) + "\n")
    return path


def _materialize_reselected(
    run_dir: Path,
    output_dir: Path,
    pool_candidates: Sequence[Any],
    retained_uuids: set[str],
) -> dict[str, Any]:
    eligible = [candidate for candidate in pool_candidates if candidate.uuid in retained_uuids]
    selected, solver = generator._select(eligible, generator.ROWS)
    parquet_path = output_dir / "reselected200.parquet"
    manifest_path = output_dir / "reselected200.manifest.jsonl"
    schema = pq.read_schema(run_dir / "selected.parquet")
    table = pa.Table.from_pylist([candidate.row for candidate in selected], schema=schema)
    pq.write_table(table, parquet_path, compression="zstd")
    manifest_path.write_text(
        "".join(_canonical_json(candidate.manifest) + "\n" for candidate in selected)
    )
    runtime_records = [
        runtime_validator._validate_row(
            candidate.row,
            candidate.manifest,
            runtime_validator.torch.device("cpu"),
        )
        for candidate in selected
    ]
    runtime_path = output_dir / "reselected200.cpu_validation.jsonl"
    runtime_path.write_text(
        "".join(_canonical_json(record) + "\n" for record in runtime_records)
    )
    runtime_passed = sum(record["passed"] for record in runtime_records)
    if runtime_passed != generator.ROWS:
        raise AssertionError(
            f"reselected CPU reference validation failed: {runtime_passed}/{generator.ROWS}"
        )
    items = _items_from_candidates(selected)
    audit = _audit("reselected200", items, output_dir)
    if audit["strict_recommended_union"]["effective_rows"] != generator.ROWS:
        raise AssertionError("reselected canary still contains strict near-duplicate components")
    return {
        "rows": len(selected),
        "source_pool_rows": len(pool_candidates),
        "strict_retained_pool_rows": len(eligible),
        "selection_solver": solver,
        "selected_profile": generator._selected_profile(selected),
        "near_duplicate_audit": audit,
        "artifacts": {
            "parquet": {"path": str(parquet_path.resolve()), "sha256": _sha256_file(parquet_path)},
            "manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": _sha256_file(manifest_path),
            },
        },
        "runtime_validation": {
            "device": "cpu",
            "passed": runtime_passed,
            "failed": len(runtime_records) - runtime_passed,
            "records": {
                "path": str(runtime_path.resolve()),
                "sha256": _sha256_file(runtime_path),
            },
        },
        "review_only": True,
        "training_approved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=_DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or args.run_dir / "near_dedup"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = _load_selected(args.run_dir)
    pool_candidates = _load_pool_candidates()
    pool = _items_from_candidates(pool_candidates)
    selected_summary = _audit("selected", selected, output_dir)
    pool_summary = _audit("candidate_pool", pool, output_dir)
    retained_uuids = {
        json.loads(line)["uuid"]
        for line in Path(
            pool_summary["artifacts"]["strict_retained_uuids"]["path"]
        ).read_text().splitlines()
        if line.strip()
    }
    reselected = _materialize_reselected(
        args.run_dir, output_dir, pool_candidates, retained_uuids
    )
    review_path = _manual_review(
        output_dir,
        selected,
        pool,
        (selected_summary, pool_summary, reselected["near_duplicate_audit"]),
    )
    final = {
        "contract": CONTRACT,
        "selected": selected_summary,
        "candidate_pool": pool_summary,
        "reselected200": reselected,
        "manual_review": {"path": str(review_path.resolve()), "sha256": _sha256_file(review_path)},
        "source_binding": {
            "auditor": {"path": str(Path(__file__).resolve()), "sha256": _sha256_file(Path(__file__))},
            "generator": {
                "path": str(Path(generator.__file__).resolve()),
                "sha256": _sha256_file(Path(generator.__file__)),
            },
            "imported_sources": [
                {
                    "path": str(path.resolve()),
                    "sha256": _sha256_file(path),
                }
                for path in (
                    _REPO_ROOT / "tools/data/cleaning/ast_similarity.py",
                    _REPO_ROOT / "tools/data/cleaning/similarity.py",
                    _REPO_ROOT / "tools/data/cleaning/complexity.py",
                    _REPO_ROOT / "tools/data/cleaning/external.py",
                    _REPO_ROOT / "tools/data/synthesize/select_low_level_kernelbench_canary.py",
                    Path(runtime_validator.__file__),
                )
            ],
            "template": {
                "path": str(generator._DEFAULT_TEMPLATE.resolve()),
                "sha256": _sha256_file(generator._DEFAULT_TEMPLATE),
                "usage": "schema and prompt prefix for deterministic candidate-pool reconstruction",
            },
            "selected": {
                "path": str((args.run_dir / "selected.parquet").resolve()),
                "sha256": _sha256_file(args.run_dir / "selected.parquet"),
            },
            "manifest": {
                "path": str((args.run_dir / "selected.manifest.jsonl").resolve()),
                "sha256": _sha256_file(args.run_dir / "selected.manifest.jsonl"),
            },
        },
        "destructive_filter_applied": False,
    }
    final_path = output_dir / "summary.json"
    _write_json(final_path, final)
    print(json.dumps(final, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
