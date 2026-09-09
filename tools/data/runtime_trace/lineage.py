"""Backtrack best-answer runtime neighborhoods with its independently implemented graph matcher.

Input: JSON list of snapshots {turn, correct, score, graph}. Scores are supplied
by the caller; this workflow does not choose a replacement training objective.
Evidence is observed correspondence, never invention or causal reward credit.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def load_matcher():
    from . import match

    return match


def components(graph, node_ids):
    adj = defaultdict(set)
    ids = set(node_ids)
    for edge in graph["edges"]:
        a, b = edge["source"], edge["target"]
        if a in ids and b in ids:
            adj[a].add(b)
            adj[b].add(a)
    result = []
    while ids:
        seen = {min(ids)}
        todo = list(seen)
        while todo:
            for nxt in adj[todo.pop()] - seen:
                seen.add(nxt)
                todo.append(nxt)
        result.append(sorted(seen))
        ids -= seen
    return result


def observed_output_ancestors(graph):
    reverse = defaultdict(set)
    for edge in graph["edges"]:
        if edge["kind"] in ("register", "predicate", "memory", "output_memory"):
            reverse[edge["target"]].add(edge["source"])
    seen = {n["id"] for n in graph["nodes"] if n["kind"] == "output"}
    todo = list(seen)
    while todo:
        for parent in reverse[todo.pop()] - seen:
            seen.add(parent)
            todo.append(parent)
    return seen


def trace_best(snapshots, matcher, *, radius=1):
    turns = [s["turn"] for s in snapshots]
    if len(set(turns)) != len(turns):
        raise ValueError("duplicate turn index")
    snapshots = sorted(snapshots, key=lambda s: s["turn"])
    valid = []
    for s in snapshots:
        if s.get("correct") is True:
            score = s.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("correct snapshot requires finite externally supplied score")
            valid.append(s)
    base = {
        "schema_version": "runtime_component_candidates/v1",
        "rule": "earliest maximum of caller-supplied correct scores",
        "unit": "candidate rooted runtime neighborhood; no certified retention",
        "radius": radius,
        "semantics": {
            "source_similarity_used": False,
            "creative_origin_proven": False,
            "causal_contribution_proven": False,
            "training_reward_modified": False,
            "certified_component_retention": False,
            "unmatched_means_absent": False,
            "node_counts_are_credit_weights": False,
        },
    }
    if not valid:
        return {**base, "status": "no_correct_answer", "best_turn": None, "candidates": [], "turns": []}
    best = max(valid, key=lambda s: (s["score"], -s["turn"]))
    base["best_turn"] = best["turn"]
    base["best_score"] = best["score"]
    if not best.get("graph"):
        return {**base, "status": "best_runtime_unavailable", "candidates": [], "turns": []}
    bg = best["graph"]
    best_nodes = {n["id"]: n for n in bg["nodes"] if n["kind"] == "instruction"}
    origins = defaultdict(list)
    output_ancestors = observed_output_ancestors(bg)
    relations = []
    gaps = []
    for previous in snapshots:
        if previous["turn"] >= best["turn"]:
            continue
        if not previous.get("graph"):
            gaps.append({"turn": previous["turn"], "reason": "runtime_unavailable"})
            relations.append({"turn": previous["turn"], "status": "unknown", "matched_best_instruction_nodes": []})
            continue
        result = matcher.align_graphs(previous["graph"], bg, radius=radius)
        pairmap = {p["left"]: p["right"] for p in result["local_candidates"] if p["right"] in best_nodes}
        for left, right in pairmap.items():
            origins[right].append({"turn": previous["turn"], "node": left})
        groups = components(bg, pairmap.values())
        relations.append(
            {
                "turn": previous["turn"],
                "status": "candidate_correspondence" if pairmap else "unknown",
                "matched_best_instruction_nodes": sorted(pairmap.values()),
                "pairs": pairmap,
                "connected_observed_fragments": groups,
                "ambiguity": result["ambiguous"],
                "unknowns": result["unknowns"],
            }
        )
    retained = []
    for key, n in best_nodes.items():
        matches = origins[key]
        retained.append(
            {
                "best_node": key,
                "opcode": n["attrs"].get("opcode"),
                "earliest_observed_correspondence_turn": matches[0]["turn"] if matches else None,
                "observed_correspondences": matches,
                "status": "candidate_observed_neighborhood" if matches else "origin_unknown",
                "best_evidence": n.get("evidence", {}),
                "on_observed_output_dependency_path": key in output_ancestors,
                "eligible_for_credit": False,
                "unknowns": n.get("unknowns", []),
            }
        )
    return {
        **base,
        "status": "candidate_lineage_only",
        "candidates": retained,
        "turns": relations,
        "coverage": bg["coverage"],
        "missing_snapshots": gaps,
        "unresolved": [
            "correspondence outside captured inputs/CTAs",
            "whole implementation preservation beyond neighborhood radius",
            "semantic correspondence across fusion/splitting/recompilation",
            "historical inheritance versus independent recreation",
        ],
    }


def markdown(result):
    lines = [
        "# 最佳答案的 runtime 组件对应",
        "",
        f"最佳轮次：{result.get('best_turn')}；状态：`{result['status']}`",
        "",
        "每行是一条实际执行指令及其局部依赖邻域；相同表示观测结构对应，不能解释成创作来源、性能贡献或整份实现等价。未对齐项保留未知",
        "",
        "| 最佳版本节点 | 指令 | 最早候选轮次 | 连接到已观测输出 | 状态 |",
        "|---|---|---|---|---|",
    ]
    for r in result["candidates"]:
        lines.append(
            f"| {r['best_node']} | {r['opcode']} | {r['earliest_observed_correspondence_turn']} | {r['on_observed_output_dependency_path']} | {r['status']} |"
        )
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshots", type=Path, required=True)
    p.add_argument("--radius", type=int, default=1)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    snapshots = json.loads(a.snapshots.read_text())
    for s in snapshots:
        if isinstance(s.get("graph"), str):
            s["graph"] = json.loads((a.snapshots.parent / s["graph"]).read_text())
    result = trace_best(snapshots, load_matcher(), radius=a.radius)
    a.output.mkdir(parents=True, exist_ok=False)
    (a.output / "lineage.json").write_text(json.dumps(result, indent=2))
    (a.output / "review.md").write_text(markdown(result))
    print(
        json.dumps(
            {
                "status": result["status"],
                "best_turn": result.get("best_turn"),
                "instruction_neighborhoods": len(result["candidates"]),
                "matched": sum(bool(x["observed_correspondences"]) for x in result["candidates"]),
            }
        )
    )


if __name__ == "__main__":
    main()
