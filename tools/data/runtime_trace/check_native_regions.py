"""GPU calibration of byte-region aggregation against ordered capture.

Run inside a frozen node-local bundle. Results are diagnostic artifacts only.
"""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from .extract import build_graph


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--tracer", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    for name, expected in json.loads(a.manifest.read_text()).items():
        assert hashlib.sha256((a.manifest.parent / name).read_bytes()).hexdigest() == expected, name
    a.output.mkdir(exist_ok=False)
    summaries = []
    for mode in [0, 4, 6, 7, 10, 11, 12, 13, 14, 15]:
        graphs, outputs = {}, {}
        for format in ["control", "runs", "regions"]:
            env = dict(os.environ)
            env.pop("LD_PRELOAD", None)
            prefix = a.output / f"mode{mode}_{format}"
            if format != "control":
                env.update(
                    LD_PRELOAD=str(a.tracer.resolve()),
                    RUNTIME_TRACE_OUTPUT=str(prefix.resolve()),
                    RUNTIME_TRACE_START_ENABLED="0",
                    RUNTIME_TRACE_RECORD_FORMAT=format,
                    RUNTIME_TRACE_CAPACITY="262144" if format == "runs" else "1024",
                    RUNTIME_TRACE_CTAS="-1",
                )
                env.pop("RUNTIME_TRACE_TOTAL_RECORDS", None)
            completed = subprocess.run(
                [str(a.fixture.resolve()), str(mode)], env=env, text=True, capture_output=True, timeout=60
            )
            prefix.with_suffix(".stdout").write_text(completed.stdout)
            prefix.with_suffix(".stderr").write_text(completed.stderr)
            completed.check_returncode()
            lines = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
            outputs[format] = lines[-1]["output"]
            if format != "control":
                graph = build_graph(
                    prefix,
                    lines[0]["allocations"],
                    {"task": str(mode), "input_signature": "fixture", "environment_signature": "same_gpu"},
                )
                prefix.with_suffix(".graph.json").write_text(json.dumps(graph, indent=2))
                graphs[format] = graph
        if mode not in {6, 14}:  # Deliberately racing writers have no deterministic output.
            assert outputs["control"] == outputs["runs"] == outputs["regions"], mode
        for raw, region in zip(graphs["runs"]["nodes"], graphs["regions"]["nodes"], strict=True):
            assert raw["implementation"] == region["implementation"], mode
            for side in ["reads", "writes"]:
                assert sorted((x["buffer"], x["regions"]) for x in raw[side]) == sorted(
                    (x["buffer"], x["regions"]) for x in region[side]
                ), (mode, side)
        if mode == 6:
            assert "write_version_ambiguous" in graphs["regions"]["nodes"][0]["unknowns"]
        if mode == 11:
            assert "inplace_order_requires_detailed_capture" in graphs["regions"]["nodes"][0]["unknowns"]
        if mode == 13:
            assert len(graphs["regions"]["nodes"][-1]["reads"][0]["regions"]) == 8
        if mode == 14:
            write = graphs["regions"]["nodes"][0]["writes"][0]
            assert write["conflicting_regions"] == [[0, 4], [16, 20]]
            assert [2048, 2052] in write["regions"]
        row = {
            "mode": mode,
            "same_interfaces": True,
            "records": {
                key: sum(n.get("evidence", {}).get("raw_runs", 0) for n in graph["nodes"])
                for key, graph in graphs.items()
            },
        }
        summaries.append(row)
        if mode in {11, 15}:
            # Run the entire program again, selecting only the in-place call.
            # Mode 15 launches one CUfunction in region -> ordered -> region modes.
            prefix = a.output / f"mode{mode}_detail"
            env.update(
                RUNTIME_TRACE_OUTPUT=str(prefix.resolve()),
                RUNTIME_TRACE_RECORD_FORMAT="regions",
                RUNTIME_TRACE_ORDERED_LAUNCHES="0" if mode == 11 else "1",
                RUNTIME_TRACE_CAPACITY="1024",
            )
            completed = subprocess.run(
                [str(a.fixture.resolve()), str(mode)], env=env, text=True, capture_output=True, timeout=60, check=True
            )
            lines = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
            detailed = build_graph(prefix, lines[0]["allocations"])
            assert lines[-1]["output"] == outputs["control"]
            assert all(not n["unknowns"] for n in detailed["nodes"]), detailed["nodes"]
            for raw, detail in zip(graphs["runs"]["nodes"], detailed["nodes"], strict=True):
                assert raw["reads"] == detail["reads"]
                assert raw["writes"] == detail["writes"]
            row["ordered_retry_resolved"] = True
    # A depleted raw-event budget must not erase the later launch's identity.
    prefix = a.output / "raw_exhausted"
    env = dict(
        os.environ,
        LD_PRELOAD=str(a.tracer.resolve()),
        RUNTIME_TRACE_OUTPUT=str(prefix.resolve()),
        RUNTIME_TRACE_START_ENABLED="0",
        RUNTIME_TRACE_RECORD_FORMAT="runs",
        RUNTIME_TRACE_CAPACITY="1",
        RUNTIME_TRACE_TOTAL_RECORDS="1",
        RUNTIME_TRACE_CTAS="-1",
    )
    completed = subprocess.run(
        [str(a.fixture.resolve()), "0"], env=env, text=True, capture_output=True, timeout=60, check=True
    )
    allocations = next(
        json.loads(line)["allocations"]
        for line in completed.stdout.splitlines()
        if line.startswith("{") and "allocations" in line
    )
    exhausted = build_graph(prefix, allocations)
    assert len(exhausted["nodes"]) == 2
    later = exhausted["nodes"][1]
    assert later["implementation"] not in {"0", "uninspected"}
    assert "opaque_implementation_configuration" not in later["unknowns"]
    assert later["configuration"]["arguments"]
    # Deliberately tiny spatial table cannot silently produce a full footprint.
    prefix = a.output / "region_overflow"
    env.update(
        RUNTIME_TRACE_OUTPUT=str(prefix.resolve()),
        RUNTIME_TRACE_RECORD_FORMAT="regions",
        RUNTIME_TRACE_CAPACITY="1",
        RUNTIME_TRACE_TOTAL_RECORDS="100",
    )
    completed = subprocess.run(
        [str(a.fixture.resolve()), "14"], env=env, text=True, capture_output=True, timeout=60, check=True
    )
    allocations = next(
        json.loads(line)["allocations"]
        for line in completed.stdout.splitlines()
        if line.startswith("{") and "allocations" in line
    )
    overflow = build_graph(prefix, allocations)
    assert not overflow["nodes"][0]["footprint_complete"]
    assert overflow["nodes"][0]["evidence"]["dropped_runs"] > 0
    (a.output / "summary.json").write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
