"""Run control/trace calibration for cuBLAS strided-batched contracts on a GPU."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

from .extract import build_graph

EXPECTED_OUTPUT = [19.0, 43.0, 22.0, 50.0, 5.0, 6.0, 10.0, 4.0]


def _run(binary, environment):
    completed = subprocess.run([str(binary)], env=environment, check=True, capture_output=True, text=True, timeout=60)
    messages = [json.loads(line) for line in completed.stdout.splitlines() if line.startswith("{")]
    allocations = next(message for message in messages if "allocations" in message)
    output = next(message for message in messages if "output" in message)
    return allocations["allocations"], output["output"]


def _assert_bounds(graph):
    buffers = {item["id"]: item["bytes"] for item in graph["buffers"]}
    for node in graph["nodes"]:
        for access in node["reads"] + node["writes"]:
            assert all(0 <= lo < hi <= buffers[access["buffer"]] for lo, hi in access["regions"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--tracer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    for name, expected in json.loads(args.manifest.read_text()).items():
        if hashlib.sha256((args.manifest.parent / name).read_bytes()).hexdigest() != expected:
            raise ValueError("fixture bundle hash mismatch: " + name)
    args.output.mkdir(parents=True, exist_ok=False)
    control_env = dict(os.environ)
    control_env.pop("LD_PRELOAD", None)
    control_allocations, control_output = _run(args.binary, control_env)
    trace_prefix = args.output / "trace"
    trace_env = dict(
        control_env,
        LD_PRELOAD=str(args.tracer),
        RUNTIME_TRACE_OUTPUT=str(trace_prefix),
        RUNTIME_TRACE_START_ENABLED="0",
        RUNTIME_TRACE_CAPACITY="1048576",
        RUNTIME_TRACE_TOTAL_RECORDS="1048576",
        RUNTIME_TRACE_MAX_LAUNCHES="32",
        RUNTIME_TRACE_MEMORY_POLICY="interfaces",
        RUNTIME_TRACE_SKIP_VENDOR_MEMORY="1",
    )
    trace_allocations, trace_output = _run(args.binary, trace_env)
    assert control_output == trace_output == EXPECTED_OUTPUT
    assert [{k: v for k, v in item.items() if k != "base"} for item in control_allocations] == [
        {k: v for k, v in item.items() if k != "base"} for item in trace_allocations
    ]
    graph = build_graph(trace_prefix, trace_allocations)
    records = [json.loads(line) for line in trace_prefix.with_suffix(".jsonl").read_text().splitlines()]
    children = [record for record in records if record["type"] == "launch" and record["parent"] >= 0]
    assert children
    for child in children:
        assert child["memory_traced"] is False
        assert child["memory_skip_reason"] == "library_api_contract"
        assert child["implementation_inspected"] is True
        assert child["library_binary"]["complete"] is True
        binary = Path(str(trace_prefix) + f".module{child['library_binary']['module_id']}.cubin")
        assert binary.stat().st_size == child["library_binary"]["module_bytes"]
        assert not Path(str(trace_prefix) + f".launch{child['id']}.bin").exists()
    libraries = [node for node in graph["nodes"] if node["kind"] == "library"]
    assert len(libraries) == 2
    assert all(node["evidence"]["api"] == "cublasSgemmStridedBatched" for node in libraries)
    first, second = libraries
    assert first["footprint_complete"] and second["footprint_complete"]
    assert "opaque_library_implementation_configuration" not in libraries[0]["unknowns"]
    first_child = children[0]["library_binary"]
    binary = Path(str(trace_prefix) + f".module{first_child['module_id']}.cubin")
    assert (
        libraries[0]["configuration"]["child_launch_configurations"][0]["library_binary"]["cubin_sha256"]
        == hashlib.sha256(binary.read_bytes()).hexdigest()
    )
    # alpha=0,beta=1 may return without launching a GPU kernel. There is then
    # no child implementation witness to invent for that no-op API call.
    for library in libraries:
        if not library["evidence"]["kernel_launches"]:
            assert "opaque_library_implementation_configuration" in library["unknowns"]
    assert [first["configuration"][key] for key in ("batch_count", "stride_a", "stride_b", "stride_c")] == [2, 4, 4, 4]
    assert [second["configuration"][key] for key in ("batch_count", "stride_a", "stride_b", "stride_c")] == [
        2,
        0,
        0,
        4,
    ]
    assert first["configuration"]["alpha"] == 1 and first["configuration"]["beta"] == 0
    assert second["configuration"]["alpha"] == 0 and second["configuration"]["beta"] == 1
    assert [item["port"] for item in first["reads"]] == ["a", "b"]
    assert [item["port"] for item in first["writes"]] == ["c"]
    assert [item["port"] for item in second["reads"]] == ["c"]
    assert [item["port"] for item in second["writes"]] == ["c"]
    assert "unmapped_contract_operand" not in first["unknowns"] + second["unknowns"]
    _assert_bounds(graph)
    (args.output / "graph.json").write_text(json.dumps(graph, indent=2))
    print(json.dumps({"status": "ok", "library_nodes": [node["id"] for node in libraries], "output": trace_output}))


if __name__ == "__main__":
    main()
