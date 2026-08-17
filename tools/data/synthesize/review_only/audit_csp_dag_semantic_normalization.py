#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any

import networkx as nx

from tools.data.synthesize.csp_dag_method import audit_csp_dag_near_duplicates as audit

IDENTITY = {("layout", "transpose"), ("layout", "chunk_cat")}
INVOLUTIONS = {("elementwise", "neg"), ("layout", "flip")}
IDEMPOTENT = {
    ("elementwise", "abs"),
    ("activation", "relu"),
    ("indexing", "gather"),
    ("indexing", "scatter"),
    ("reduction", "mean_center"),
}
MODULE_RULES = {"conv", "linear", "normalization"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reduced_graph(item: audit.Item, *, cse: bool, rank: bool, strip_loss: bool) -> nx.DiGraph:
    typed = item.manifest["graph"]["typed_graph"]
    graph = nx.DiGraph()
    input_id = "input"
    graph.add_node(input_id, label=f"input:ndim={typed['input_rank']}" if rank else "input")
    representative: dict[int, str] = {}
    labels: dict[str, tuple[str, str]] = {}
    predecessors: dict[str, tuple[str, ...]] = {}
    cse_keys: dict[tuple[Any, ...], str] = {}
    created = 0

    def child_label(node_id: str) -> tuple[str, str] | None:
        return labels.get(node_id)

    def child_predecessor(node_id: str) -> str | None:
        values = predecessors.get(node_id, ())
        return values[0] if len(values) == 1 else None

    nodes = list(typed["nodes"])
    if strip_loss and nodes and nodes[-1]["rule"] == "loss":
        nodes = nodes[:-1]
    for node in nodes:
        original_index = int(node["index"])
        key = (str(node["rule"]), str(node["variant"]))
        pred_ids = tuple(input_id if int(value) < 0 else representative[int(value)] for value in node["predecessors"])

        if key in IDENTITY:
            representative[original_index] = pred_ids[0]
            continue
        if len(pred_ids) == 1 and key in INVOLUTIONS and child_label(pred_ids[0]) == key:
            grandparent = child_predecessor(pred_ids[0])
            if grandparent is not None:
                representative[original_index] = grandparent
                continue
        if len(pred_ids) == 1 and key in IDEMPOTENT and child_label(pred_ids[0]) == key:
            representative[original_index] = pred_ids[0]
            continue
        if (
            key == ("indexing", "scatter")
            and len(pred_ids) == 1
            and child_label(pred_ids[0]) == ("indexing", "gather")
        ):
            representative[original_index] = pred_ids[0]
            continue
        if key in {("elementwise", "abs"), ("elementwise", "square")} and len(pred_ids) == 1:
            child = child_label(pred_ids[0])
            if child in {("elementwise", "neg"), ("elementwise", "abs")}:
                grandparent = child_predecessor(pred_ids[0])
                if grandparent is not None:
                    pred_ids = (grandparent,)

        label = f"{key[0]}:{key[1]}"
        module_backed = key[0] in MODULE_RULES or node.get("module_name") is not None
        cse_key = (label, pred_ids)
        if cse and not module_backed and cse_key in cse_keys:
            representative[original_index] = cse_keys[cse_key]
            continue

        node_id = f"op:{created}"
        created += 1
        graph.add_node(node_id, label=label)
        labels[node_id] = key
        predecessors[node_id] = pred_ids
        commutative = key == ("layout", "binary_cat")
        for slot, pred_id in enumerate(pred_ids):
            arg_id = f"arg:{created}:{slot}:{original_index}"
            graph.add_node(arg_id, label="arg:any" if commutative else f"arg:{slot}")
            graph.add_edge(pred_id, arg_id)
            graph.add_edge(arg_id, node_id)
        representative[original_index] = node_id
        if cse and not module_backed:
            cse_keys[cse_key] = node_id

    if nodes:
        output_id = representative[int(nodes[-1]["index"])]
    else:
        output_id = input_id
    graph.add_node("output", label="output")
    graph.add_edge(output_id, "output")
    return graph


def exact_pairs(items: list[audit.Item], **kwargs: Any) -> list[tuple[int, int]]:
    buckets: dict[str, list[tuple[int, nx.DiGraph]]] = collections.defaultdict(list)
    for index, item in enumerate(items):
        graph = reduced_graph(item, **kwargs)
        digest = nx.weisfeiler_lehman_graph_hash(graph, node_attr="label", iterations=6)
        buckets[digest].append((index, graph))
    match = nx.algorithms.isomorphism.categorical_node_match("label", None)
    pairs: list[tuple[int, int]] = []
    for bucket in buckets.values():
        for offset, (left_index, left_graph) in enumerate(bucket):
            for right_index, right_graph in bucket[:offset]:
                if nx.is_isomorphic(left_graph, right_graph, node_match=match):
                    pairs.append((left_index, right_index))
    return sorted(pairs)


def pair_rows(items: list[audit.Item], pairs: list[tuple[int, int]]) -> list[dict[str, Any]]:
    return [
        {
            "left_position": left,
            "left_uuid": items[left].uuid,
            "left_nodes": items[left].manifest["graph"]["node_count"],
            "left_input_rank": items[left].manifest["graph"]["typed_graph"]["input_rank"],
            "right_position": right,
            "right_uuid": items[right].uuid,
            "right_nodes": items[right].manifest["graph"]["node_count"],
            "right_input_rank": items[right].manifest["graph"]["typed_graph"]["input_rank"],
        }
        for left, right in pairs
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    items = audit._load_selected(args.run_dir)
    profiles = {
        "identity_reduced_variant_rank": dict(cse=False, rank=True, strip_loss=False),
        "identity_cse_variant_rank": dict(cse=True, rank=True, strip_loss=False),
        "identity_cse_variant_no_rank": dict(cse=True, rank=False, strip_loss=False),
        "identity_cse_body_without_loss_rank": dict(cse=True, rank=True, strip_loss=True),
    }
    results = {name: exact_pairs(items, **options) for name, options in profiles.items()}
    report = {
        "contract": "csp_dag_semantic_normalization_duplicate_audit_v1",
        "rows": len(items),
        "normalizations": {
            "identity": sorted(f"{rule}:{variant}" for rule, variant in IDENTITY),
            "involutions": sorted(f"{rule}:{variant}" for rule, variant in INVOLUTIONS),
            "idempotent": sorted(f"{rule}:{variant}" for rule, variant in IDEMPOTENT),
            "additional": [
                "scatter(gather(x)) -> gather(x)",
                "abs(neg(x)) -> abs(x)",
                "square(neg(x)) -> square(x)",
                "square(abs(x)) -> square(x)",
                "safe pure-op common-subexpression elimination",
                "binary_cat argument order treated as commutative",
            ],
        },
        "profile_pair_counts": {name: len(pairs) for name, pairs in results.items()},
        "profile_pairs": {name: pair_rows(items, pairs[:200]) for name, pairs in results.items()},
        "profile_pairs_truncated": {name: len(pairs) > 200 for name, pairs in results.items()},
        "source_binding": {
            "selected": {
                "path": str((args.run_dir / "selected.parquet").resolve()),
                "sha256": sha256(args.run_dir / "selected.parquet"),
            },
            "manifest": {
                "path": str((args.run_dir / "selected.manifest.jsonl").resolve()),
                "sha256": sha256(args.run_dir / "selected.manifest.jsonl"),
            },
            "auditor": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
        },
        "interpretation": {
            "identity_reduced_variant_rank": "deletion-sensitive exact semantic graph after verified identity/involution reductions",
            "identity_cse_variant_rank": "more aggressive exact functional-structure screen; candidates require manual review before deletion",
            "identity_cse_variant_no_rank": "rank-insensitive diagnostic only; never an automatic deletion rule",
            "identity_cse_body_without_loss_rank": "loss-head-peeled diagnostic only; distinct losses remain valid output semantics",
        },
        "review_only": True,
        "training_approved": False,
    }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
