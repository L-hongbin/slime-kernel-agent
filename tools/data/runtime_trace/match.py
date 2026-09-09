"""Independent component correspondence over observed dependency neighborhoods.

Uses NetworkX VF2 as a third-party graph primitive. No state-matching code is
imported or copied. Cheap root labels only shortlist; exact rooted multigraph
isomorphism and two-way uniqueness are required for correspondence.
"""

import json
import time
from collections import Counter, defaultdict

import networkx as nx


def label(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def graph(record):
    if record.get("schema_version") != "runtime_program_graph/v1":
        raise ValueError("unsupported graph schema")
    g = nx.MultiDiGraph()
    for n in record["nodes"]:
        if n["id"] in g:
            raise ValueError("duplicate node")
        g.add_node(n["id"], label=label([n["kind"], n["attrs"]]), unknowns=n.get("unknowns", []))
    for e in record["edges"]:
        if e["source"] not in g or e["target"] not in g:
            raise ValueError("dangling edge")
        g.add_edge(e["source"], e["target"], label=label([e["kind"], e["attrs"]]), unknowns=e.get("unknowns", []))
    return g


def neighborhood(g, root, radius):
    ids = {root}
    frontier = {root}
    for _ in range(radius):
        nxt = set()
        for n in frontier:
            nxt.update(g.predecessors(n))
            nxt.update(g.successors(n))
        frontier = nxt - ids
        ids.update(nxt)
    h = g.subgraph(ids).copy()
    for n in h:
        h.nodes[n]["root"] = n == root
    return h


class Exhausted(Exception):
    pass


class BoundedVF2(nx.algorithms.isomorphism.MultiDiGraphMatcher):
    def __init__(self, a, b, deadline, steps):
        self.deadline = deadline
        self.steps = steps
        self.visits = 0
        super().__init__(
            a,
            b,
            node_match=lambda x, y: x["root"] == y["root"] and x["label"] == y["label"],
            edge_match=lambda x, y: Counter(e["label"] for e in x.values()) == Counter(e["label"] for e in y.values()),
        )

    def semantic_feasibility(self, a, b):
        self.visits += 1
        if self.visits > self.steps or time.monotonic() > self.deadline:
            raise Exhausted()
        return super().semantic_feasibility(a, b)


def align_graphs(
    left, right, *, radius=1, max_pairs=10000, max_neighborhood_nodes=32, timeout_seconds=20, max_vf2_steps=20000
):
    if radius < 1:
        raise ValueError("dependency neighborhood radius must be positive")
    context = ["task", "model", "input_signature", "environment_signature"]
    missing = [k for k in context if left.get("context", {}).get(k) is None or right.get("context", {}).get(k) is None]
    different = [k for k in context if k not in missing and left["context"][k] != right["context"][k]]
    base = {
        "method": "independent rooted VF2 multigraph correspondence",
        "radius": radius,
        "complete_state_match": False,
        "certifies_retention": False,
    }
    if missing or different:
        return {
            **base,
            "local_candidates": [],
            "ambiguous": [],
            "unknowns": {"missing_context": missing, "different_context": different},
        }
    a, b = graph(left), graph(right)
    deadline = time.monotonic() + timeout_seconds
    candidates = defaultdict(list)
    for n, data in b.nodes(data=True):
        candidates[data["label"]].append(n)
    neighborhoods = {}
    eligible = {}
    omitted = []
    for side, g in [("left", a), ("right", b)]:
        for root in g:
            h = neighborhood(g, root, radius)
            key = side, root
            neighborhoods[key] = h
            reasons = []
            if len(h) > max_neighborhood_nodes:
                reasons.append("neighborhood_size_budget")
            if any(d["unknowns"] for _, d in h.nodes(data=True)) or any(d["unknowns"] for *_, d in h.edges(data=True)):
                reasons.append("unknown_local_facts")
            eligible[key] = not reasons
            if reasons:
                omitted.append({"side": side, "node": root, "reasons": reasons})
    forward = defaultdict(list)
    backward = defaultdict(list)
    uncertain_a = set()
    uncertain_b = set()
    checked = 0
    for x, data in a.nodes(data=True):
        for y in candidates[data["label"]]:
            if not eligible["left", x] or not eligible["right", y]:
                # An unexamined alternative can defeat apparent uniqueness.
                uncertain_a.add(x)
                uncertain_b.add(y)
                continue
            ah, bh = neighborhoods["left", x], neighborhoods["right", y]
            if (len(ah), ah.number_of_edges()) != (len(bh), bh.number_of_edges()):
                continue
            if checked >= max_pairs or time.monotonic() > deadline:
                uncertain_a.add(x)
                uncertain_b.add(y)
                continue
            checked += 1
            try:
                equal = BoundedVF2(ah, bh, deadline, max_vf2_steps).is_isomorphic()
            except Exhausted:
                uncertain_a.add(x)
                uncertain_b.add(y)
                continue
            if equal:
                forward[x].append(y)
                backward[y].append(x)
    pairs = [
        {"left": x, "right": ys[0], "relation": "observed_dependency_neighborhood", "radius": radius}
        for x, ys in forward.items()
        if len(ys) == 1 and len(backward[ys[0]]) == 1 and x not in uncertain_a and ys[0] not in uncertain_b
    ]

    # Verify that individually matched neighborhoods can coexist as one partial
    # mapping. Conflicts remove both pairs, never pick an arbitrary winner.
    def edges(g, u, v):
        return Counter(d["label"] for d in (g.get_edge_data(u, v) or {}).values())

    conflicts = set()
    for p in pairs:
        for q in pairs:
            if edges(a, p["left"], q["left"]) != edges(b, p["right"], q["right"]):
                conflicts.update([p["left"], q["left"]])
    return {
        **base,
        "local_candidates": [p for p in pairs if p["left"] not in conflicts],
        "ambiguous": [
            {"left": x, "right_candidates": ys}
            for x, ys in forward.items()
            if len(ys) > 1 or any(len(backward[y]) > 1 for y in ys)
        ],
        "unknowns": {
            "omitted": omitted,
            "unexamined_left": sorted(uncertain_a),
            "unexamined_right": sorted(uncertain_b),
            "inconsistent_pairs": sorted(conflicts),
            "left_coverage": left.get("coverage", {}),
            "right_coverage": right.get("coverage", {}),
        },
        "checked_pairs": checked,
    }
