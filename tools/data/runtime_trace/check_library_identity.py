"""CPU contracts for launch attributes and cuBLAS child implementation evidence."""

import hashlib
import json
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.data.runtime_trace.extract import build_graph, launch_configuration


def cubin_fixture():
    names = b"\0.shstrtab\0.text._Z5entryv\0.text._Z6otherv\0"
    table_end = 64 + 4 * 64
    code = b"0123456789abcdef" * 2
    header = struct.pack(
        "<16sHHIQQQIHHHHHH", b"\x7fELF\x02\x01\x01" + bytes(9), 2, 190, 1, 0, 0, 64, 0, 64, 0, 0, 64, 4, 1
    )
    sections = [bytes(64), struct.pack("<IIQQQQIIQQ", 1, 3, 0, 0, table_end, len(names), 0, 0, 1, 0)]
    for index, name in enumerate([b".text._Z5entryv", b".text._Z6otherv"]):
        sections.append(
            struct.pack(
                "<IIQQQQIIQQ", names.index(name), 1, 6, 0, table_end + len(names) + 16 * index, 16, 0, 0, 16, 0
            )
        )
    return header + b"".join(sections) + names + code


BINARY = cubin_fixture()


def _supported_attributes():
    return {
        "complete": True,
        "values": [
            {"id": "cluster_dimension", "value": [2, 1, 1]},
        ],
    }


def _records(child_name, *, attributes=None, inspected=True, binary=BINARY, entry="_Z5entryv"):
    return [
        {"type": "config", "schema": "coarse-memory-runs/v1", "cta_limit": -1},
        {"type": "scope", "seq": 0, "enabled": True},
        {
            "type": "library_begin",
            "seq": 1,
            "api": "cublasSgemm_v2",
            "stream": 0,
            "attrs": {
                "ta": 0,
                "tb": 0,
                "m": 2,
                "n": 2,
                "k": 2,
                "a": 100,
                "b": 200,
                "c": 300,
                "at": 0,
                "bt": 0,
                "ct": 0,
                "lda": 2,
                "ldb": 2,
                "ldc": 2,
                "alpha": 1,
                "beta": 0,
                "state_query_ok": True,
                "workspace_mode": "default_pool",
                "library_version": 120901,
            },
        },
        {
            "type": "launch",
            "id": 0,
            "seq": 2,
            "parent": 1,
            "name": child_name,
            "stream": 0,
            "grid": [1, 1, 1],
            "block": [128, 1, 1],
            "shared": 0,
            "memory_traced": False,
            "implementation": "legacy-name-dependent-value",
            "implementation_version": "sass-fingerprint-v1",
            "implementation_inspected": inspected,
            "unsupported_memory_instructions": 0,
            "arguments": [],
            "arguments_present": False,
            "launch_api": "cuLaunchKernelEx",
            "launch_num_attrs": 1,
            "launch_attributes": _supported_attributes() if attributes is None else attributes,
            "library_binary": (
                {
                    "complete": True,
                    "module_id": 0,
                    "module_bytes": len(binary),
                    "entry_selector": entry,
                }
                if binary is not None
                else {"complete": False, "reason": "module_dump_budget_exceeded"}
            ),
            "memory_skip_reason": "library_api_contract",
        },
        {"type": "complete", "seq": 3, "id": 0, "runs": 0, "dropped": 0, "drop_count_exact": True},
        {"type": "library_end", "seq": 4, "parent": 1, "status": 0},
        {"type": "end", "seq": 5, "launches": 1},
    ]


def _library(child_name, *, attributes=None, inspected=True, binary=BINARY, entry="_Z5entryv"):
    allocations = [
        {"id": "a", "base": 100, "bytes": 16, "roles": ["input:0"]},
        {"id": "b", "base": 200, "bytes": 16, "roles": ["parameter:0"]},
        {"id": "c", "base": 300, "bytes": 16, "roles": ["output:0"]},
    ]
    with tempfile.TemporaryDirectory(prefix="library-identity-") as temporary:
        prefix = Path(temporary) / "trace"
        Path(str(prefix) + ".jsonl").write_text(
            "\n".join(
                json.dumps(record)
                for record in _records(
                    child_name, attributes=attributes, inspected=inspected, binary=binary, entry=entry
                )
            )
            + "\n"
        )
        if binary is not None:
            Path(str(prefix) + ".module0.cubin").write_bytes(binary)
        graph = build_graph(prefix, allocations)
    libraries = [node for node in graph["nodes"] if node["kind"] == "library"]
    assert len(libraries) == 1
    return libraries[0]


def main():
    configuration, unknowns = launch_configuration(
        {"launch_num_attrs": 1, "launch_attributes": _supported_attributes()}
    )
    assert unknowns == []
    assert configuration["launch_attributes"] == _supported_attributes()["values"]

    for summary in (
        {"complete": False, "unsupported_ids": [1], "values": []},
        {"complete": True, "values": [{"id": "programmatic_event", "value": 0}]},
        {"complete": True, "values": [{"id": "cluster_dimension", "value": [1, 1]}]},
    ):
        assert launch_configuration({"launch_num_attrs": 1, "launch_attributes": summary})[1] == [
            "opaque_launch_configuration"
        ]

    first = _library("vendor_symbol_alpha")
    second = _library("vendor_symbol_beta")
    assert first["implementation"] == second["implementation"]
    assert "opaque_launch_configuration" not in first["unknowns"]
    assert "opaque_library_implementation_configuration" not in first["unknowns"]
    assert first["configuration"]["child_launch_configurations"] == [
        {
            "launch_num_attrs": 1,
            "launch_attributes": _supported_attributes()["values"],
            "library_binary": {
                "cubin_sha256": hashlib.sha256(BINARY).hexdigest(),
                "entry_selector": "_Z5entryv",
            },
        }
    ]
    assert first["evidence"]["kernel_launches"] == [0]

    opaque = _library("vendor_symbol_alpha", attributes={"complete": False, "unsupported_ids": [7], "values": []})
    assert "opaque_launch_configuration" in opaque["unknowns"]
    inspected_false = _library("vendor_symbol_alpha", inspected=False)
    assert "opaque_library_implementation_configuration" in inspected_false["unknowns"]
    missing_binary = _library("vendor_symbol_alpha", binary=None)
    assert "opaque_library_implementation_configuration" in missing_binary["unknowns"]
    assert _library("vendor_symbol_alpha", entry="_Z6otherv")["implementation"] != first["implementation"]
    for invalid in [BINARY[:63], BINARY[:-1], b"not a cubin", BINARY[:18] + b"\x3e\0" + BINARY[20:]]:
        assert "opaque_library_implementation_configuration" in _library("vendor", binary=invalid)["unknowns"]
    assert "opaque_library_implementation_configuration" in _library("vendor", entry="absent_entry")["unknowns"]
    print("library identity: supported attrs and cubin+entry child evidence are canonical; opaque paths fail closed")


if __name__ == "__main__":
    main()
