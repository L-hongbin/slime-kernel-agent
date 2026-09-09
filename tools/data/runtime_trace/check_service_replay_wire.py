"""Offline producer-wire check using an archived real coarse-trace fixture.

This deliberately invokes only ``report``: no CUDA, KernelGym service, or
native collector is loaded.  The archived capture supplies a realistic JSONL
trace/lease pair; binding evidence is added only to the temporary copy because
that old capture predates the v1 HTTP envelope.
"""

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:  # Supports both ``python -m`` and direct use.
    sys.path.insert(0, str(ROOT))
from examples.kernel_agent.component_reward import _units
from tools.data.runtime_trace.service_replay import (
    _complete_call_enumeration,
    _prepare_precompiled_artifact,
    canonical,
    report,
)

DEFAULT_FIXTURE = (
    ROOT.parent
    / "KernelGYM-component-runtime"
    / "local_artifacts/runtime_graph_integration/delivery_results/local_artifacts/capture-egykfq4m"
)
WIRE = ROOT.parent / "slime-trace-state-matching/local_artifacts/component_reward_integration/wire/schema.json"
H20_FIXTURE = ROOT / "local_artifacts/component_reward_training/h20_first_capture"


def _hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _prepare(root, *, max_summary_nodes, include_collector=True, binding_case="ok", enumeration_case="ok"):
    request_path = root / "request.json"
    request = json.loads(request_path.read_text())
    if include_collector:
        request["collector_sha256"] = "b" * 64
    request["compile_artifact_provenance"] = "a" * 64
    request["options"]["max_summary_nodes"] = max_summary_nodes
    request_path.write_text(json.dumps(request))

    capsule_path = root / "capsule.json"
    capsule = json.loads(capsule_path.read_text())
    capsule["request_identity"] = {
        "task_sha256": _hash(request["reference_code"]),
        "candidate_source_sha256": _hash(request["kernel_code"]),
    }
    capsule["candidate_provenance"] = "a" * 64
    capsule["configuration_snapshot"] = json.loads((root / "control_result.json").read_text())["configuration"]
    capsule["rng"] = {"cpu": "archived-fixture", "cuda": "archived-fixture"}
    if binding_case == "missing_captured_identity":
        capsule.pop("request_identity")
    capsule_path.write_text(json.dumps(capsule))

    for phase in ("control", "trace"):
        path = root / f"{phase}_result.json"
        value = json.loads(path.read_text())
        value["artifact_provenance"] = "a" * 64
        value["rng"] = {"cpu": "archived-fixture", "cuda": "archived-fixture"}
        value.update({key: request[key] for key in ("backend", "precision", "entry_point")})
        if binding_case in {"both_mismatched", f"{phase}_mismatched"}:
            value["source_sha256"] = "0" * 64
        path.write_text(json.dumps(value))
    if enumeration_case != "ok":
        trace_path = root / "trace.jsonl"
        records = [json.loads(line) for line in trace_path.read_text().splitlines()]
        end = next(record for record in records if record.get("type") == "end")
        if enumeration_case == "end_count_mismatch":
            end["launches"] += 1
        elif enumeration_case == "unsupported_launch":
            records.insert(
                records.index(end),
                {
                    "type": "unsupported_launch",
                    "name": "launch_budget_exhausted",
                    "seq": end["seq"] - 1,
                },
            )
        else:
            raise ValueError("unknown enumeration test case: " + enumeration_case)
        trace_path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


def _run(fixture, *, max_summary_nodes, include_collector=True, binding_case="ok", enumeration_case="ok"):
    with tempfile.TemporaryDirectory(prefix="service-replay-wire-") as temporary:
        root = Path(temporary) / "capture"
        shutil.copytree(fixture, root)
        _prepare(
            root,
            max_summary_nodes=max_summary_nodes,
            include_collector=include_collector,
            binding_case=binding_case,
            enumeration_case=enumeration_case,
        )
        report(root)
        summary = root / "summary.json"
        result = json.loads(summary.read_text())
        assert summary.stat().st_size <= 128 * 1024
        assert summary.read_text() == canonical(result)
        return result


def _assert_h20_compression(fixture, schema):
    compressed = _run(fixture, max_summary_nodes=64)
    jsonschema.Draft202012Validator(json.loads(schema.read_text())).validate(compressed)
    assert compressed["status"] == "ok", compressed
    graph = compressed["graph"]
    assert graph["coverage"]["summary_complete"] is True
    assert graph["coverage"]["inline_footprint_omitted_nodes"] == ["kernel:1"]
    assert set(graph["output_unknowns"]) == {
        "inline_footprint_omitted_output_closure:kernel:1",
        "output_producer_unresolved:storage:1",
    }
    assert graph["nodes"][0]["footprint_complete"] is False
    assert graph["nodes"][0]["reads"] == [] and graph["nodes"][0]["writes"] == []
    assert "inline_footprint_omitted_byte_budget" in graph["nodes"][0]["unknowns"]
    assert compressed["graph_artifact"]["content"] == "full_graph_before_inline_footprint_compression"
    assert "inline_graph_sha256" in compressed
    units, closure_gaps = _units({"graph": graph})
    assert closure_gaps == [] and len(units) == 1
    assert units[0]["signature"] is None and "output_membership_unresolved" in units[0]["unknowns"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--schema", type=Path, default=WIRE)
    parser.add_argument("--h20-fixture", type=Path, default=H20_FIXTURE)
    parser.add_argument("--h20-only", action="store_true")
    parser.add_argument(
        "--extended", action="store_true", help="run slower legacy binding/identity envelope negatives"
    )
    args = parser.parse_args()
    if not args.fixture.is_dir():
        raise SystemExit(f"archived fixture unavailable: {args.fixture}")

    supplied = {"so_path": "/tmp/existing.so", "entry_point": "CustomNew", "backend": "tvm_ffi"}
    prepared = _prepare_precompiled_artifact(
        {"compile_artifact": supplied, "kernel_code": "candidate", "entry_point": "Model", "backend": "cuda"},
        "cuda:3",
    )
    assert prepared["compiled"] is True and prepared["code"] == "candidate"
    assert prepared["entry_point"] == "CustomNew" and prepared["backend"] == "tvm_ffi"
    assert prepared["device"] == "cuda:3" and supplied == {
        "so_path": "/tmp/existing.so",
        "entry_point": "CustomNew",
        "backend": "tvm_ffi",
    }
    assert _prepare_precompiled_artifact({"kernel_code": "candidate", "backend": "cuda"}, "cuda:0") is None

    if args.h20_only:
        if not args.h20_fixture.is_dir():
            raise SystemExit(f"H20 fixture unavailable: {args.h20_fixture}")
        _assert_h20_compression(args.h20_fixture, args.schema)
        print("service replay wire: real H20 incomplete-footprint compression passed")
        return

    complete = _run(args.fixture, max_summary_nodes=64)
    jsonschema.Draft202012Validator(json.loads(args.schema.read_text())).validate(complete)
    assert complete["status"] == "ok", complete
    assert complete["alignment"]["scored_control"] is True
    assert complete["alignment"]["control_trace"] is True
    assert complete["graph"]["coverage"]["summary_complete"] is True
    assert complete["identity"]["collector_sha256"] == "b" * 64

    truncated = _run(args.fixture, max_summary_nodes=1)
    jsonschema.Draft202012Validator(json.loads(args.schema.read_text())).validate(truncated)
    assert truncated["status"] == "unavailable", truncated
    assert "graph" not in truncated
    assert any("graph_summary_incomplete" in value for value in truncated["unknowns"])

    for enumeration_case in ("end_count_mismatch", "unsupported_launch"):
        enumeration_clipped = _run(args.fixture, max_summary_nodes=64, enumeration_case=enumeration_case)
        jsonschema.Draft202012Validator(json.loads(args.schema.read_text())).validate(enumeration_clipped)
        assert enumeration_clipped["status"] == "unavailable", enumeration_clipped
        assert "graph" not in enumeration_clipped
        assert "missing_call_units" in enumeration_clipped["unknowns"]
    assert _complete_call_enumeration(
        {
            "total_launches_reported": 2,
            "kernel_launches": 2,
            "completed_kernel_launches": 2,
            "memory_unknown_events": [],
        }
    )
    assert not _complete_call_enumeration(
        {
            "total_launches_reported": 3,
            "kernel_launches": 2,
            "completed_kernel_launches": 2,
            "memory_unknown_events": [],
        }
    )
    assert not _complete_call_enumeration(
        {
            "total_launches_reported": 2,
            "kernel_launches": 2,
            "completed_kernel_launches": 2,
            "memory_unknown_events": [{"type": "unsupported_launch", "name": "launch_budget_exhausted"}],
        }
    )
    assert not _complete_call_enumeration(
        {
            "total_launches_reported": 2,
            "kernel_launches": 2,
            "completed_kernel_launches": 2,
            "memory_unknown_events": [{"type": "memory_api_unknown", "name": "cuMemcpy"}],
        }
    )
    if args.extended:
        missing_collector = _run(args.fixture, max_summary_nodes=64, include_collector=False)
        jsonschema.Draft202012Validator(json.loads(args.schema.read_text())).validate(missing_collector)
        assert missing_collector["status"] == "unavailable", missing_collector
        assert "graph" not in missing_collector
        assert "runtime_graph_identity_incomplete" in missing_collector["unknowns"]

        both_mismatched = _run(args.fixture, max_summary_nodes=64, binding_case="both_mismatched")
        jsonschema.Draft202012Validator(json.loads(args.schema.read_text())).validate(both_mismatched)
        assert both_mismatched["status"] == "partial", both_mismatched
        assert both_mismatched["alignment"]["checks"]["scored_control"]["request_binding"] is False
        assert both_mismatched["alignment"]["checks"]["control_trace"]["request_binding"] is False

        trace_mismatched = _run(args.fixture, max_summary_nodes=64, binding_case="trace_mismatched")
        jsonschema.Draft202012Validator(json.loads(args.schema.read_text())).validate(trace_mismatched)
        assert trace_mismatched["status"] == "partial", trace_mismatched
        assert trace_mismatched["alignment"]["checks"]["scored_control"]["request_binding"] is True
        assert trace_mismatched["alignment"]["checks"]["control_trace"]["request_binding"] is False

        missing_captured_identity = _run(args.fixture, max_summary_nodes=64, binding_case="missing_captured_identity")
        jsonschema.Draft202012Validator(json.loads(args.schema.read_text())).validate(missing_captured_identity)
        assert missing_captured_identity["status"] == "partial", missing_captured_identity
        assert missing_captured_identity["alignment"]["checks"]["scored_control"]["request_binding"] is None
    print("service replay wire: complete, truncation, and call-enumeration negatives passed")


if __name__ == "__main__":
    main()
