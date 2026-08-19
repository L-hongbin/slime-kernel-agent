#!/usr/bin/env python3
"""Offline generation and dedup check for the CSP-DAG canary."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.synthesize.csp_dag_method import audit_csp_dag_near_duplicates as dedup  # noqa: E402
from tools.data.synthesize.csp_dag_method import generate_csp_dag as generator  # noqa: E402


def _node(index: int, rule: str, variant: str, predecessors: tuple[int, ...]) -> dict:
    return {
        "family": "conv" if rule == "conv" else "shape_layout",
        "index": index,
        "module_name": f"op_{index:03d}" if rule == "conv" else None,
        "predecessors": predecessors,
        "rule": rule,
        "variant": variant,
    }


def _dedup_item(position: int, rank: int, nodes: list[dict]) -> dedup.Item:
    return dedup.Item(
        position=position,
        uuid=f"offline-{position}",
        code="\n".join(f"operation_{index}" for index in range(len(nodes))),
        manifest={
            "candidate_index": position,
            "operator_signature": sorted({f"{node['rule']}:{node['variant']}" for node in nodes}),
            "actual_low_level_families": sorted({node["family"] for node in nodes}),
            "complexity": {"source_line_count": len(nodes)},
            "graph": {
                "node_count": len(nodes),
                "typed_graph": {
                    "input_rank": rank,
                    "nodes": nodes,
                    "edges": [[predecessor, node["index"]] for node in nodes for predecessor in node["predecessors"]],
                },
            },
        },
        normalized_ops=(),
    )


def main() -> None:
    candidates = [generator._make_candidate(index, "") for index in range(100)]
    assert len({candidate.uuid for candidate in candidates}) == 100
    assert len({candidate.code for candidate in candidates}) == 100
    for candidate in candidates:
        assert "TensorBridge" not in candidate.code
        assert "form_bridge" not in candidate.code
        assert candidate.manifest["top_level_class_policy"] == "single_model_only"
        assert candidate.manifest["semantic_multiclass_status"] == "deferred_until_real_subgraph_helper_lowering"
        repeated = generator._make_candidate(candidate.index, "")
        assert (candidate.uuid, candidate.code, candidate.manifest) == (
            repeated.uuid,
            repeated.code,
            repeated.manifest,
        )

    safe_scatter = next(candidate.code for candidate in candidates if ".scatter(-1," in candidate.code)
    assert ".scatter(-1, torch.zeros_like(" in safe_scatter and "[..., :1], dtype=torch.long)" in safe_scatter
    high = generator._make_candidate(3, "").manifest
    assert high["graph"]["node_count"] >= generator.HIGH_COMPLEXITY_NODE_THRESHOLD
    assert high["sink_merge_node_ratio"] <= high["sink_merge_node_ratio_cap"]

    items = [
        _dedup_item(0, 3, [_node(0, "layout", "transpose", (-1,)), _node(1, "layout", "chunk_cat", (0,))]),
        _dedup_item(1, 3, [_node(0, "layout", "transpose", (-1,))]),
        _dedup_item(2, 2, [_node(0, "layout", "binary_cat", (-1, -1))]),
        _dedup_item(3, 3, [_node(0, "layout", "transpose", (-1,)), _node(1, "layout", "binary_cat", (-1, 0))]),
        _dedup_item(4, 4, [_node(0, "conv", "conv_transpose", (-1,))]),
        _dedup_item(5, 3, [_node(0, "conv", "conv_transpose", (-1,))]),
    ]
    pairs = {frozenset(pair) for pair in dedup._identity_normalized_isomorphic_pairs(items)}
    assert frozenset((0, 1)) in pairs
    assert frozenset((2, 3)) in pairs
    assert frozenset((4, 5)) not in pairs

    print(json.dumps({"candidates": len(candidates), "identity_normalized_pairs": len(pairs), "status": "passed"}))


if __name__ == "__main__":
    main()
