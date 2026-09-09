"""Backtrack best-answer coarse components with an explicit retention heuristic."""

import argparse
import json
import math
from pathlib import Path
from .match import compare


def trace_best(snapshots):
    if len({s["turn"] for s in snapshots}) != len(snapshots):
        raise ValueError("duplicate turn")
    ordered = sorted(snapshots, key=lambda s: s["turn"])
    correct = [s for s in ordered if s.get("correct") is True]
    for s in correct:
        if (
            isinstance(s.get("score"), bool)
            or not isinstance(s.get("score"), (int, float))
            or not math.isfinite(s["score"])
        ):
            raise ValueError("finite caller-supplied score required")
    result = {
        "schema": "coarse-component-lineage/v1",
        "status": "no_correct_answer",
        "best_turn": None,
        "components": [],
        "no_observed_data_effect_calls": [],
        "comparisons": [],
        "creative_origin_proven": False,
        "training_reward_modified": False,
        "node_count_is_reward": False,
    }
    if not correct:
        return result
    best = max(correct, key=lambda s: (s["score"], -s["turn"]))
    result["best_turn"] = best["turn"]
    result["best_score"] = best["score"]
    if not best.get("graph"):
        result["status"] = "best_graph_missing"
        return result
    origins = {}
    for previous in ordered:
        if previous["turn"] >= best["turn"]:
            continue
        if not previous.get("graph"):
            result["comparisons"].append({"turn": previous["turn"], "status": "graph_missing"})
            continue
        comparison = compare(previous["graph"], best["graph"])
        result["comparisons"].append({"turn": previous["turn"], **comparison})
        for match in comparison["matches"]:
            if match["relation"] == "retained_observed_component":
                origins.setdefault(match["after"], {"turn": previous["turn"], "node": match["before"]})
    for node in best["graph"]["nodes"]:
        if node["footprint_complete"] and not node["reads"] and not node["writes"]:
            result["no_observed_data_effect_calls"].append(node["id"])
            continue
        result["components"].append(
            {
                "node": node["id"],
                "kind": node["kind"],
                "evidence": node["evidence"],
                "earliest_observed_retained_match": origins.get(node["id"]),
                "status": "observed_retention" if node["id"] in origins else "new_changed_or_unresolved",
                "unknowns": node["unknowns"],
            }
        )
    result["status"] = "observed_component_heuristic"
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshots", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = json.loads(args.snapshots.read_text())
    for row in rows:
        if isinstance(row.get("graph"), str):
            row["graph"] = json.loads((args.snapshots.parent / row["graph"]).read_text())
    result = trace_best(rows)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
    print(
        json.dumps(
            {"best_turn": result["best_turn"], "components": len(result["components"]), "status": result["status"]}
        )
    )


if __name__ == "__main__":
    main()
