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
        print("extract fast paths: 500 randomized exact-equivalence cases passed")


if __name__ == "__main__":
    main()
