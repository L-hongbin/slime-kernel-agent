"""CPU equivalence checks for exact coarse-trace extractor fast paths."""

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.data.runtime_trace import extract


def _reference_entry_read_regions(reads, writes):
    index = extract.writer_index(writes)
    entry, unknown = [], []
    for read in reads:
        overlaps = [write for write in extract.related(index, read[0], read[1]) if extract.overlap(read, write)]
        if len(overlaps) > 512:
            unknown.append(read[:2])
            continue
        cuts = sorted({read[0], read[1], *[value for write in overlaps for value in extract.overlap(read, write)]})
        for lo, hi in zip(cuts, cuts[1:], strict=False):
            relevant = [write for write in overlaps if write[0] < hi and write[1] > lo]
            safe = all(extract.same_owner(read, write, lo, hi) and read[4] < write[4] for write in relevant)
            (entry if safe else unknown).append((lo, hi))
    return extract.union(entry), extract.union(unknown)


def _reference_conflicting_writes(writes):
    index = extract.writer_index(writes)
    conflicts = []
    for write in writes:
        candidates = extract.related(index, write[0], write[1])
        if len(candidates) > 512:
            conflicts.append(write[:2])
            continue
        for other in candidates:
            common = extract.overlap(write, other)
            if common and not extract.same_owner(write, other, *common):
                conflicts.append(common)
    return extract.union(conflicts)


def _reference_subtract_regions(left, right):
    """Pre-optimization interval subtraction oracle (intentionally quadratic)."""
    right = extract.union(map(tuple, right))
    result = []
    for lo, hi in extract.union(map(tuple, left)):
        cursor = lo
        for start, end in right:
            if end <= cursor:
                continue
            if start >= hi:
                break
            if cursor < start:
                result.append([cursor, min(start, hi)])
            cursor = max(cursor, end)
            if cursor >= hi:
                break
        if cursor < hi:
            result.append([cursor, hi])
    return result


def _reference_version_dependencies(nodes, allocations, memory_unknown=(), orders=()):
    """Frozen linear-scan oracle for the indexed version-dependency path."""
    ancestors, tails, events, pending = {}, {}, {}, extract.defaultdict(set)
    global_barrier = set()
    unknown_memory_nodes = [
        {
            "id": f"unknown_memory:{event['seq']}",
            "seq": event["seq"],
            "stream": event.get("stream", -1000 - event["seq"]),
            "footprint_complete": False,
        }
        for event in memory_unknown
    ]
    timeline = sorted(
        [("node", node) for node in nodes + unknown_memory_nodes] + [("order", order) for order in orders],
        key=lambda item: item[1]["seq"],
    )
    for kind, item in timeline:
        if kind == "node":
            parents = set(global_barrier) | pending[item["stream"]]
            if item["stream"] in tails:
                parents.add(tails[item["stream"]])
            ancestors[item["id"]] = parents | {parent for value in parents for parent in ancestors.get(value, ())}
            tails[item["stream"]] = item["id"]
        elif item["action"] == "record":
            events[item["event"]] = tails.get(item["stream"])
        elif item["action"] == "wait" and events.get(item["event"]) is not None:
            pending[item["stream"]].add(events[item["event"]])
        elif item["action"] == "destroy":
            events.pop(item["event"], None)
        elif item["action"] == "device_sync":
            global_barrier.update(tails.values())

    writers = extract.defaultdict(list)
    versions = []
    for node in nodes:
        for access in node["writes"]:
            version = {
                "id": f"{access['buffer']}@{node['id']}",
                "buffer": access["buffer"],
                "producer": node["id"],
                "regions": access["regions"],
                "seq": node["seq"],
                "stream": node["stream"],
                "complete": node["footprint_complete"],
                "conflicts": access["conflicting_regions"],
                "atomic_unknown": access["atomic_unknown"],
            }
            versions.append(version)
            writers[access["buffer"]].append(version)
    initial = {allocation["id"]: allocation for allocation in allocations}
    unknown_calls = [node for node in nodes if not node["footprint_complete"]] + unknown_memory_nodes
    edges = []

    def intersecting(version, lo, hi, field="regions"):
        return [region for region in version[field] if extract.overlap((lo, hi), region)]

    for node in nodes:
        for read in node["reads"]:
            candidates = [writer for writer in writers[read["buffer"]] if writer["producer"] != node["id"]]
            grouped = extract.defaultdict(list)
            for lo, hi in read["entry_regions"]:
                relevant = [writer for writer in candidates if intersecting(writer, lo, hi)]
                cuts = sorted(
                    {
                        lo,
                        hi,
                        *[
                            value
                            for writer in relevant
                            for region in intersecting(writer, lo, hi)
                            if extract.overlap((lo, hi), region)
                            for value in extract.overlap((lo, hi), region)
                        ],
                    }
                )
                for start, end in zip(cuts, cuts[1:], strict=False):
                    active = [writer for writer in relevant if intersecting(writer, start, end)]
                    before = [writer for writer in active if writer["producer"] in ancestors[node["id"]]]
                    unordered = [
                        writer
                        for writer in active
                        if writer["producer"] not in ancestors[node["id"]]
                        and node["id"] not in ancestors[writer["producer"]]
                    ]
                    latest_versions = [
                        writer
                        for writer in before
                        if not any(
                            writer["producer"] in ancestors[other["producer"]]
                            for other in before
                            if other is not writer
                        )
                    ]
                    latest = latest_versions[0] if len(latest_versions) == 1 else None
                    unknown_between = any(
                        unknown["id"] != node["id"]
                        and node["id"] not in ancestors[unknown["id"]]
                        and not (latest and unknown["id"] in ancestors[latest["producer"]])
                        for unknown in unknown_calls
                    )
                    if unordered or len(latest_versions) > 1:
                        source, version, certainty = None, None, "unordered_stream_writers"
                    elif latest:
                        conflict = bool(intersecting(latest, start, end, "conflicts"))
                        source, version = latest["producer"], latest["id"]
                        certainty = (
                            "proven_region_dependency"
                            if latest["complete"]
                            and node["footprint_complete"]
                            and not conflict
                            and not latest["atomic_unknown"]
                            and not unknown_between
                            else "partial_or_ambiguous_dependency"
                        )
                    elif (
                        initial[read["buffer"]]["roles"]
                        and any(
                            role.startswith(("input:", "parameter:", "state:"))
                            for role in initial[read["buffer"]]["roles"]
                        )
                        and not unknown_between
                    ):
                        source, version, certainty = (
                            "initial:" + read["buffer"],
                            read["buffer"] + "@entry",
                            "initial_value_read" if node["footprint_complete"] else "possible_initial_value_read",
                        )
                    else:
                        source, version, certainty = None, None, "entry_value_unobserved"
                    grouped[source, version, certainty].append((start, end))
            for (source, version, certainty), regions in grouped.items():
                edges.append(
                    {
                        "source": source,
                        "target": node["id"],
                        "buffer": read["buffer"],
                        "version": version,
                        "regions": extract.union(regions),
                        "port": read["port"],
                        "certainty": certainty,
                        "evidence": read["evidence"],
                    }
                )
    return edges, versions


def _random_intervals(generator, *, maximum=100, count=8):
    return [
        (start := generator.randrange(maximum), start + generator.randrange(1, 16))
        for _ in range(generator.randrange(count + 1))
    ]


def _random_version_dependency_equivalence():
    generator = random.Random(20260911)
    allocations = [{"id": "storage:0", "roles": ["input:0"]}]
    for _ in range(300):
        nodes = []
        for sequence in range(generator.randrange(1, 8)):
            reads = []
            writes = []
            if generator.randrange(2):
                regions = _random_intervals(generator)
                reads.append(
                    {
                        "buffer": "storage:0",
                        "regions": regions,
                        "entry_regions": regions,
                        "port": f"read:{sequence}",
                        "evidence": "random_oracle",
                    }
                )
            if generator.randrange(2):
                writes.append(
                    {
                        "buffer": "storage:0",
                        "regions": _random_intervals(generator),
                        "conflicting_regions": _random_intervals(generator, count=3),
                        "atomic_unknown": bool(generator.randrange(5) == 0),
                    }
                )
            nodes.append(
                {
                    "id": f"node:{sequence}",
                    "seq": sequence * 10,
                    "stream": generator.randrange(2),
                    "footprint_complete": bool(generator.randrange(4)),
                    "reads": reads,
                    "writes": writes,
                }
            )
        # Include order edges as part of the whole-graph oracle, without
        # relying on a recorded CUDA trace.
        orders = [
            {"seq": 5, "action": "record", "event": "e", "stream": nodes[0]["stream"]},
            {"seq": nodes[-1]["seq"] + 1, "action": "wait", "event": "e", "stream": nodes[-1]["stream"]},
        ]
        actual = extract.version_dependencies(nodes, allocations, orders=orders)
        reference = _reference_version_dependencies(nodes, allocations, orders=orders)
        assert actual == reference


def _randomized_unit_equivalence():
    generator = random.Random(20260910)
    for _ in range(500):
        reads = []
        writes = []
        for seq in range(generator.randrange(35)):
            lo = generator.randrange(100)
            width = generator.randrange(1, 9)
            item = (lo, lo + generator.randrange(1, 17), generator.randrange(32), width, seq)
            (reads if generator.randrange(2) else writes).append(item)
        assert extract.entry_read_regions(reads, writes) == _reference_entry_read_regions(reads, writes)
        assert extract.conflicting_writes(writes) == _reference_conflicting_writes(writes)
    for _ in range(1000):
        left = _random_intervals(generator, maximum=500, count=20)
        right = _random_intervals(generator, maximum=500, count=20)
        assert extract.subtract_regions(left, right) == _reference_subtract_regions(left, right)
    _random_version_dependency_equivalence()


def _stable_graph(graph):
    graph = json.loads(json.dumps(graph))
    graph["provenance"].pop("parse_seconds", None)
    graph["provenance"].pop("extractor_sha256", None)
    return graph


def _real_graph_equivalence(root):
    allocations = json.loads((root / "trace_allocations.json").read_text())["allocations"]
    started = time.monotonic()
    optimized = extract.build_graph(root / "trace", allocations, {})
    optimized_seconds = time.monotonic() - started
    original_entry, original_conflict = extract.entry_read_regions, extract.conflicting_writes
    try:
        extract.entry_read_regions = _reference_entry_read_regions
        extract.conflicting_writes = _reference_conflicting_writes
        started = time.monotonic()
        reference = extract.build_graph(root / "trace", allocations, {})
        reference_seconds = time.monotonic() - started
    finally:
        extract.entry_read_regions, extract.conflicting_writes = original_entry, original_conflict
    stable = _stable_graph(optimized)
    assert stable == _stable_graph(reference)
    print(
        json.dumps(
            {
                "optimized_wall_seconds": optimized_seconds,
                "reference_wall_seconds": reference_seconds,
                "stable_graph_sha256": hashlib.sha256(extract.canonical(stable).encode()).hexdigest(),
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-capture", type=Path)
    args = parser.parse_args()
    _randomized_unit_equivalence()
    if args.real_capture:
        _real_graph_equivalence(args.real_capture)
    else:
        print("extract fast paths: 500 run and 1000 subtraction/random-graph exact-equivalence cases passed")


if __name__ == "__main__":
    main()
