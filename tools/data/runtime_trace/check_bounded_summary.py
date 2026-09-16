"""CPU contract for bounded runtime-graph summaries.

The HTTP form may erase only incomplete interfaces.  Complete call footprints
remain byte-for-byte visible, while omitted incomplete calls stay explicit
residual units so reward cannot silently renormalize their mass away.
"""

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The installed Megatron package can own a top-level ``examples`` package.
import examples

examples.__path__ = [str(ROOT / "examples"), *examples.__path__]

from examples.kernel_agent.component_reward import _units
from tools.data.runtime_trace.service_replay import _full_graph_summary, canonical


def _ranges(count, *, offset=0):
    return [[offset + 16 * index, offset + 16 * index + 8] for index in range(count)]


def _access(buffer, regions, *, port):
    return {
        "buffer": buffer,
        "regions": regions,
        "entry_regions": regions,
        "internal_or_unordered_regions": [],
        "port": port,
    }


def _node(index, *, complete):
    node_id = f"kernel:{index}"
    regions = _ranges(1 if complete else 600, offset=index * 100000)
    return {
        "id": node_id,
        "kind": "kernel",
        "implementation": f"impl:{index}",
        "configuration": {"block": [256, 1, 1]},
        "footprint_complete": complete,
        "unknowns": [] if complete else ["memory_runs_dropped"],
        "reads": [_access("input", regions, port="input")],
        "writes": [
            {
                **_access(f"output:{index}", regions, port="output"),
                "conflicting_regions": [],
                "atomic_unknown": False,
            }
        ],
        # This deliberately duplicates large unresolved dependency detail.  A
        # bounded summary may retain its digest/count but must not retain the
        # interval list after an incomplete node is omitted.
        "evidence": (
            {}
            if complete
            else {
                "unknown_input_dependencies": [
                    {
                        "source": None,
                        "target": node_id,
                        "buffer": "input",
                        "regions": _ranges(300, offset=index * 200000),
                        "certainty": "partial_or_ambiguous_dependency",
                    }
                ]
            }
        ),
    }


def _graph():
    nodes = [_node(index, complete=index not in {1, 3}) for index in range(5)]
    buffers = [{"id": "input", "roles": ["input:0"], "bytes": 1 << 30, "views": []}]
    buffers.extend(
        {"id": f"output:{index}", "roles": [f"output:{index}"], "bytes": 1 << 30, "views": []} for index in range(5)
    )
    return {
        "schema": "coarse-component-graph/v1",
        "nodes": nodes,
        "edges": [],
        "buffers": buffers,
        "versions": [
            {
                "producer": node["id"],
                "buffer": f"output:{index}",
                "regions": _ranges(600 if not node["footprint_complete"] else 1, offset=index * 100000),
            }
            for index, node in enumerate(nodes)
        ],
        "outputs": [
            {
                "source": node["id"],
                "target": "forward_output",
                "buffer": f"output:{index}",
                "regions": _ranges(1, offset=index * 100000),
                "certainty": "proven_region_dependency",
            }
            for index, node in enumerate(nodes)
        ],
        "output_unknowns": [],
        "coverage": {
            "trace_process_complete": True,
            "total_launches_reported": 5,
            "kernel_launches": 5,
            "completed_kernel_launches": 5,
            "memory_unknown_events": [],
        },
    }


def main():
    full = _graph()
    full_bytes = len(canonical(full).encode("utf-8"))
    complete_footprints = {
        node["id"]: (copy.deepcopy(node["reads"]), copy.deepcopy(node["writes"]))
        for node in full["nodes"]
        if node["footprint_complete"]
    }
    inline, reason = _full_graph_summary(full, max_nodes=5, max_bytes=16 * 1024)
    assert reason is None
    assert inline is not None
    assert full_bytes > 16 * 1024
    assert len(canonical(inline).encode("utf-8")) <= 16 * 1024
    assert inline["coverage"]["summary_complete"] is True
    assert len(inline["nodes"]) == 5

    nodes = {node["id"]: node for node in inline["nodes"]}
    for node_id, (reads, writes) in complete_footprints.items():
        assert nodes[node_id]["footprint_complete"] is True
        assert nodes[node_id]["reads"] == reads
        assert nodes[node_id]["writes"] == writes
    for node_id in ("kernel:1", "kernel:3"):
        assert nodes[node_id]["footprint_complete"] is False
        assert nodes[node_id]["reads"] == [] and nodes[node_id]["writes"] == []
        assert "inline_footprint_omitted_byte_budget" in nodes[node_id]["unknowns"]
        summary = nodes[node_id]["evidence"]["unknown_input_dependency_summary"]
        assert summary["count"] == 1 and summary["detail"] == "full_graph_artifact"
    assert {version["producer"] for version in inline["versions"]} == {"kernel:0", "kernel:2", "kernel:4"}
    assert set(inline["output_unknowns"]) == {
        "inline_footprint_omitted_output_closure:kernel:1",
        "inline_footprint_omitted_output_closure:kernel:3",
    }

    units, gaps = _units({"graph": inline})
    assert gaps == [] and len(units) == 5
    assert {unit["id"] for unit in units if unit["signature"] is None} == {"kernel:1", "kernel:3"}
    assert len([unit for unit in units if unit["signature"] is not None]) == 3
    print("bounded summary: five units retained, two explicit residual units, complete footprints preserved")


if __name__ == "__main__":
    main()
