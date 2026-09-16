"""Interpret a bounded set of successful host-API state transitions.

Rules come from the CUDA 12.9 cuBLAS/cuDNN API contracts, not numerical
counterfactuals. Unknown APIs/initial state are not reconstructed. Allocation
events describe requests, not peak resident memory or GPU completion times.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def interpret(events, *, coverage=(), dropped_events=0, write_errors=0):
    events = sorted(events, key=lambda e: e["sequence"])
    generations = collections.Counter()
    current = {}
    lifetimes = []
    handle_state = {}
    descriptor_state = {}
    gemms, convolutions, resets = [], [], []
    coverage = set(coverage)
    workspace_observable = (
        {"cublasSetWorkspace_v2", "cublasSetStream_v2"} <= coverage and not dropped_events and not write_errors
    )
    unknown_prior = set()
    creations = {
        "cublasCreate_v2": "cublas_handle",
        "cublasLtMatmulDescCreate": "lt_descriptor",
        "cudnnCreateConvolutionDescriptor": "convolution_descriptor",
    }
    destroys = {
        "cublasDestroy_v2": "cublas_handle",
        "cublasLtMatmulDescDestroy": "lt_descriptor",
        "cudnnDestroyConvolutionDescriptor": "convolution_descriptor",
    }

    def identity(kind, pointer):
        key = (kind, pointer)
        if key not in current:
            unknown_prior.add(key)
            return f"{kind}:{pointer}:unknown_generation"
        return current[key]["id"]

    for event in events:
        if event["status"] != 0:
            continue
        api, pointer = event["api"], event["object"]
        if api in creations:
            kind = creations[api]
            key = (kind, pointer)
            generations[key] += 1
            item = {
                "id": f"{kind}:{pointer}:{generations[key]}",
                "kind": kind,
                "pointer": pointer,
                "created_call": event["call_id"],
                "created_sequence": event["sequence"],
                "destroyed_sequence": None,
            }
            current[key] = item
            lifetimes.append(item)
            if kind == "cublas_handle":
                handle_state[item["id"]] = {
                    "workspace": {"kind": "default_on_create"},
                    "math_mode": "default_on_create",
                }
            if kind == "lt_descriptor":
                descriptor_state[item["id"]] = {"compute_type": event["value_name"], "attributes": {}}
        elif api in destroys:
            key = (destroys[api], pointer)
            if key in current:
                current[key]["destroyed_sequence"] = event["sequence"]
                del current[key]
            else:
                unknown_prior.add(key)
        elif api in {"cublasSetMathMode", "cublasSetWorkspace_v2", "cublasSetStream_v2", "cublasGemmEx"}:
            ident = identity("cublas_handle", pointer)
            state = handle_state.setdefault(ident, {"workspace": {"kind": "unknown"}, "math_mode": "unknown"})
            if api == "cublasSetMathMode":
                state["math_mode"] = event["value_name"]
            elif api == "cublasSetWorkspace_v2":
                state["workspace"] = {
                    "kind": "user_bound",
                    "pointer": event["resource"],
                    "bytes": event["bytes"],
                    "set_sequence": event["sequence"],
                }
            elif api == "cublasSetStream_v2":
                if state["workspace"]["kind"] == "user_bound":
                    resets.append(
                        {
                            "handle": ident,
                            "call_id": event["call_id"],
                            "sequence": event["sequence"],
                            "cleared_binding": dict(state["workspace"]),
                            "evidence": "successful_SetStream_resets_user_workspace_by_cuBLAS_contract",
                        }
                    )
                state["workspace"] = {"kind": "default_after_SetStream", "reset_sequence": event["sequence"]}
            else:
                observed_state = json.loads(json.dumps(state))
                if not workspace_observable:
                    observed_state["workspace"] = {
                        "kind": "unknown_due_to_probe_coverage",
                        "last_observed": observed_state["workspace"],
                    }
                gemms.append(
                    {
                        "api": api,
                        "call_id": event["call_id"],
                        "sequence": event["sequence"],
                        "handle": ident,
                        "compute_type_argument": event["value_name"],
                        "observed_handle_state": observed_state,
                    }
                )
        elif api in {"cublasLtMatmulDescSetAttribute", "cublasLtMatmul"}:
            ident = identity("lt_descriptor", pointer)
            state = descriptor_state.setdefault(ident, {"compute_type": "unknown", "attributes": {}})
            if api == "cublasLtMatmulDescSetAttribute":
                state["attributes"][event["value_name"]] = {"value": event["value"], "pointer": event["resource"]}
                if event["value_name"] == "other_matmul_descriptor_attribute":
                    state["compute_type"] = "unknown_after_undecoded_attribute_update"
            else:
                gemms.append(
                    {
                        "api": api,
                        "call_id": event["call_id"],
                        "sequence": event["sequence"],
                        "descriptor": ident,
                        "observed_descriptor_state": json.loads(json.dumps(state)),
                        "workspace_argument": {"pointer": event["resource"], "bytes": event["bytes"]},
                    }
                )
        elif api == "cudnnConvolutionForward":
            convolutions.append(
                {
                    "call_id": event["call_id"],
                    "sequence": event["sequence"],
                    "descriptor": identity("convolution_descriptor", pointer),
                    "algorithm": event["value"],
                    "workspace_argument": {"pointer": event["resource"], "bytes": event["bytes"]},
                }
            )
    return {
        "lifetimes": lifetimes,
        "gemm_state": gemms,
        "convolution_calls": convolutions,
        "workspace_resets": resets,
        "probe_coverage": sorted(coverage),
        "dropped_events": dropped_events,
        "write_errors": write_errors,
        "unknown_prior_objects": [list(k) for k in sorted(unknown_prior)],
        "api_counts": dict(collections.Counter(e["api"] for e in events)),
        "failed_api_counts": dict(collections.Counter(e["api"] for e in events if e["status"] != 0)),
        "semantics": {
            "observed_successful_api_state_only": True,
            "compute_type_is_requested_policy_not_proof_of_tensor_core_execution": True,
            "missing_destroy_implies_leak": False,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    for path in sorted(args.runtime_root.glob("*/native_state.jsonl")):
        observation = json.loads(path.with_name("observation.json").read_text())
        if "observer_id" not in observation:
            continue
        events = [json.loads(line) for line in path.read_text().splitlines()]
        events = [e for e in events if e["observer"] == observation["observer_id"]]
        run_result = json.loads(path.with_name("result.json").read_text())
        result = {
            "source_identity": observation.get("source_identity"),
            **interpret(
                events,
                coverage=observation.get("native_probe_coverage", []),
                dropped_events=max(
                    observation.get("native_probe_dropped_events", 0), run_result.get("native_probe_dropped_events", 0)
                ),
                write_errors=run_result.get("native_probe_write_errors", 0),
            ),
        }
        (args.output / (path.parent.name + ".json")).write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(
            json.dumps(
                {
                    "case": path.parent.name,
                    "events": len(events),
                    "workspace_resets": len(result["workspace_resets"]),
                    "lifetime_instances": len(result["lifetimes"]),
                }
            )
        )


if __name__ == "__main__":
    main()
