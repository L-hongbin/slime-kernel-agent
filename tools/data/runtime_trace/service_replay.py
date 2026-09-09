"""KernelGym's fresh-process diagnostic adapter around the canonical tracer.

The capsule comes from a normal correctness trial; scoring is already finished.
Control, instrumented execution, and CPU reporting each use a fresh process.
"""

import argparse
import ctypes
import hashlib
import json
import os
import time
from pathlib import Path

from .extract import build_graph, digest


def write(path, value):
    path.write_text(json.dumps(value, indent=2))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(root, phase, device):
    os.environ.pop("LD_PRELOAD", None)  # Never instrument compiler subprocesses.
    import torch

    torch.cuda.set_device(torch.device(device))
    from kernelgym.backend.kernelbench.dispatcher import KernelBenchBackend
    from kernelgym.toolkit.kernelbench.execution_policy import EXECUTION_POLICY_VERSION, tf32_execution_context
    from kernelgym.toolkit.kernelbench.runtime_graph_capture import fingerprints, model_tensor_state

    from .observer import StorageObserver

    request = json.loads((root / "request.json").read_text())
    original = json.loads((root / "capsule.json").read_text())
    result = {
        "status": "partial",
        "phase": phase,
        "unknowns": [],
        "source_sha256": hashlib.sha256(request["kernel_code"].encode()).hexdigest(),
        "reference_sha256": hashlib.sha256(request["reference_code"].encode()).hexdigest(),
        "capsule_sha256": sha(root / "capsule.pt"),
        "runner_sha256": sha(Path(__file__)),
    }
    enable = None
    observer = None
    started = time.monotonic()
    try:
        namespace = {}
        exec(compile(request["reference_code"], "<diagnostic-reference>", "exec"), namespace)
        torch.manual_seed(original["model_seed"])
        torch.cuda.manual_seed(original["model_seed"])
        init = namespace["get_init_inputs"]()

        def move(value):
            if isinstance(value, torch.Tensor):
                return value.to(device)
            if isinstance(value, dict):
                return {k: move(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return type(value)(move(v) for v in value)
            return value

        init = move(init)
        backend = KernelBenchBackend()
        artifact = request.get("compile_artifact") or backend.compile(
            request["kernel_code"],
            backend=request["backend"],
            precision=request["precision"],
            entry_point=(request.get("entry_point") or "Model") + "New",
            device=device,
            enable_compile_artifact_cache=True,
        )
        if not artifact.get("compiled"):
            result["unknowns"].append("diagnostic_compile_failed")
            return result
        handle = backend.load(artifact, device=device)
        torch.manual_seed(original["model_seed"])
        torch.cuda.manual_seed(original["model_seed"])
        model = backend.create_model(handle, init, device=device)
        model.train(original["training"])
        # Loading the entire archive on the target device preserves shared
        # storage/views between input tensors, unlike individual .to() calls.
        capsule = torch.load(root / "capsule.pt", map_location=device, weights_only=True)
        if original["state_mode"] == "restore":
            model.load_state_dict(capsule["state_dict"], strict=True)
            with torch.no_grad():
                for name, value in capsule["nonpersistent_buffers"].items():
                    model.get_buffer(name).copy_(value)
        inputs = capsule["inputs"]
        result["input"] = fingerprints(inputs)
        result["state"] = fingerprints(model_tensor_state(model))
        result["initial_state_matches_normal"] = (
            result["input"] == original["input"] and result["state"] == original["state"]
        )
        if not result["initial_state_matches_normal"]:
            result["unknowns"].append("initial_state_or_input_mismatch")
            return result
        observer = StorageObserver(
            traced=phase == "trace",
            checkpoint=root / f"{phase}_allocations.json",
            max_leases=request["options"]["max_launches"] * 8 + 512,
        )
        observer.observe(inputs, "input")
        for i, value in enumerate(model.parameters()):
            observer.observe(value, f"parameter:{i}")
        for i, value in enumerate(model.buffers()):
            observer.observe(value, f"state:{i}")
        write(root / f"{phase}_allocations.json", observer.as_dict())
        if phase == "trace":
            enable = ctypes.CDLL(None).runtime_trace_set_enabled
            enable.argtypes = [ctypes.c_int]
        torch.set_rng_state(capsule["cpu_rng"].cpu())
        torch.cuda.set_rng_state(capsule["cuda_rng"].cpu(), device)
        policy = {}
        with tf32_execution_context(policy, stage="diagnostic"), torch.no_grad():
            result["configuration"] = {
                "execution_policy": EXECUTION_POLICY_VERSION,
                "tf32_forced": policy["diagnostic_tf32_state_forced"],
                "training": model.training,
                "grad_enabled": torch.is_grad_enabled(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "trial_seed": original["trial_seed"],
            }
            result["configuration_matches_normal"] = (
                original["execution_policy"] == EXECUTION_POLICY_VERSION
                and original["tf32_forced"] == policy["diagnostic_tf32_state_forced"]
                and original["training"] == model.training
            )
            if not result["configuration_matches_normal"]:
                result["unknowns"].append("execution_policy_mismatch")
                return result
            torch.cuda.synchronize(device)
            forward_start = time.monotonic()
            if enable:
                enable(1)
                with observer:
                    output = model(**inputs) if isinstance(inputs, dict) else model(*inputs)
            else:
                output = model(**inputs) if isinstance(inputs, dict) else model(*inputs)
            torch.cuda.synchronize(device)
            result["forward_wall_seconds"] = time.monotonic() - forward_start
            if enable:
                enable(0)
            observer.observe(output, "output:0")
            result["output"] = fingerprints(output)
        result["status"] = "completed"
    except Exception as exc:
        result["unknowns"].append(f"{type(exc).__name__}: {str(exc)[:2000]}")
    finally:
        if enable:
            enable(0)
        if observer:
            write(root / f"{phase}_allocations.json", observer.as_dict())
        result["process_work_seconds"] = time.monotonic() - started
        write(root / f"{phase}_result.json", result)
    return result


def report(root):
    request = json.loads((root / "request.json").read_text())
    result = {
        "schema": "kernelgym-runtime-graph/v1",
        "status": "partial",
        "unknowns": [],
        "training_state_equivalence": False,
        "node_count_is_reward": False,
    }
    result["unknowns"].extend(json.loads((root / "capsule.json").read_text()).get("unknowns", []))
    phases = {}
    for name in ["control", "trace"]:
        path = root / f"{name}_result.json"
        phases[name] = json.loads(path.read_text()) if path.exists() else {"unknowns": [f"{name}_process_incomplete"]}
        result["unknowns"].extend(phases[name].get("unknowns", []))
    control, traced = phases["control"], phases["trace"]
    normal_path = root / "normal_output.json"
    result["alignment"] = {
        "control_trace_outputs_equal": "output" in control and control.get("output") == traced.get("output"),
        "normal_control_outputs_equal": normal_path.exists()
        and control.get("output") == json.loads(normal_path.read_text()),
        "input_state_equal": all(p.get("initial_state_matches_normal") for p in phases.values()),
        "configuration_equal": all(p.get("configuration_matches_normal") for p in phases.values()),
    }
    result["identity"] = {
        k: control.get(k) for k in ["source_sha256", "reference_sha256", "capsule_sha256", "configuration"]
    }
    result["phase_costs"] = {k: v.get("forward_wall_seconds") for k, v in phases.items()}
    metadata = root / "trace.jsonl"
    allocation_path = root / "trace_allocations.json"
    try:
        if not metadata.exists():
            raise ValueError("trace_metadata_missing")
        if allocation_path.exists():
            registry = json.loads(allocation_path.read_text())
            allocations = registry["allocations"]
            result["unknowns"].extend(registry.get("unknowns", []))
        else:
            allocations = []
            result["unknowns"].append("storage_registry_missing_after_process_failure")
        context = {
            "task": control.get("reference_sha256"),
            "input_signature": digest([control.get("input"), control.get("state")]),
            "environment_signature": digest(control.get("configuration")),
        }
        graph = build_graph(root / "trace", allocations, context)
        write(root / "graph.json", graph)
        result["graph_artifact"] = {
            "path": str(root / "graph.json"),
            "sha256": sha(root / "graph.json"),
            "bytes": (root / "graph.json").stat().st_size,
        }
        result["coverage"] = graph["coverage"]
        limit = request["options"]["max_summary_nodes"]
        result["components"] = [
            {
                "id": n["id"],
                "kind": n["kind"],
                "implementation": n["implementation"],
                "read_buffers": len(n["reads"]),
                "write_buffers": len(n["writes"]),
                "footprint_complete": n["footprint_complete"],
                "unknowns": n["unknowns"],
            }
            for n in graph["nodes"][:limit]
        ]
        result["summary_truncated"] = len(graph["nodes"]) > limit
        if not result["summary_truncated"] and len(json.dumps(graph)) <= 48 * 1024:
            result["graph"] = graph
        if all(result["alignment"].values()) and graph["coverage"]["trace_process_complete"]:
            result["status"] = "observed"
    except Exception as exc:
        result["unknowns"].append(f"{type(exc).__name__}: {str(exc)[:1000]}")
    result["unknowns"] = sorted(set(result["unknowns"]))
    if len(json.dumps(result, indent=2)) > 120 * 1024:
        result.pop("graph", None)
        result["unknowns"].append("inline_graph_byte_budget_exceeded")
    write(root / "summary.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--phase", choices=["control", "trace", "report"], required=True)
    args = parser.parse_args()
    if args.phase == "report":
        report(args.directory)
    else:
        execute(args.directory, args.phase, args.device)


if __name__ == "__main__":
    main()
