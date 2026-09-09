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

from .extract import build_graph


def write(path, value):
    path.write_text(json.dumps(value, indent=2))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    """Stable digest material for the HTTP graph envelope, not Python reprs."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str)


def value_digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


_ALIGNMENT_CHECKS = (
    "request_binding",
    "candidate_binding",
    "inputs",
    "initial_state",
    "execution_configuration",
    "outputs",
)


def _aggregate_checks(checks):
    """True means every required observation is positively established.

    A missing fact is deliberately null rather than an optimistic true.  This
    keeps a replay graph reviewable without accidentally making it eligible
    for component credit.
    """
    values = [checks[name] for name in _ALIGNMENT_CHECKS]
    if any(value is False for value in values):
        return False
    return True if all(value is True for value in values) else None


def _same(left, right):
    return left == right if left is not None and right is not None else None


def _and_facts(*values):
    """Three-valued conjunction for observations, never ``False == False``."""
    if any(value is False for value in values):
        return False
    return True if all(value is True for value in values) else None


def _phase_binding(phase, request):
    """Whether a replay phase identifies the exact requested candidate.

    ``artifact_provenance`` is intentionally only a locator/fingerprint.  It
    does not claim that independently loaded binaries have equal bits.
    """
    return all(
        (
            phase.get("source_sha256") == hashlib.sha256(request["kernel_code"].encode("utf-8")).hexdigest(),
            phase.get("reference_sha256") == hashlib.sha256(request["reference_code"].encode("utf-8")).hexdigest(),
            phase.get("backend") == request.get("backend"),
            phase.get("precision") == request.get("precision"),
            phase.get("entry_point") == (request.get("entry_point") or "Model"),
            bool(phase.get("artifact_provenance")),
        )
    )


def _environment_material(configuration):
    """Keep cross-turn-compatible execution facts, excluding GPU identity.

    Device ordinal/UUID, CUDA pointer values and temporary paths are process
    provenance only.  GPU model, software versions and policy remain part of
    the compatibility key whenever the replay reports them.
    """
    allowed = {
        "execution_policy",
        "tf32_forced",
        "training",
        "grad_enabled",
        "torch",
        "cuda",
        "gpu_name",
        "gpu_compute_capability",
        "backend",
        "precision",
        "entry_point",
        "default_dtype",
    }
    return {key: configuration[key] for key in sorted(allowed & set(configuration))}


def _full_graph_summary(graph, *, max_nodes, max_bytes):
    """Return an inline graph only when the complete call summary fits.

    There is no prefix mode: a truncated graph is unavailable to the reward
    consumer so a short header can never absorb omitted component mass.
    """
    graph = json.loads(json.dumps(graph))
    coverage = graph.setdefault("coverage", {})
    coverage["summary_complete"] = bool(
        coverage.get("trace_process_complete")
        and coverage.get("kernel_launches") == coverage.get("completed_kernel_launches")
        and len(graph.get("nodes", ())) <= max_nodes
    )
    encoded = canonical(graph).encode("utf-8")
    if len(encoded) > max_bytes:
        coverage["summary_complete"] = False
        return None, "inline_graph_byte_budget_exceeded"
    if not coverage["summary_complete"]:
        return None, "graph_summary_incomplete"
    return graph, None


def _prepare_precompiled_artifact(request, device):
    """Mirror the scored precompiled-artifact defaults before replay loading.

    The service request carries the usable full artifact, while metadata keeps
    a sanitized audit copy.  Never replace the former with the latter: paths
    and code deliberately disappear from the audit representation.
    """
    supplied = request.get("compile_artifact")
    if not isinstance(supplied, dict):
        return None
    artifact = dict(supplied)
    artifact.setdefault("compiled", True)
    artifact.setdefault("code", request["kernel_code"])
    artifact.setdefault("entry_point", (request.get("entry_point") or "Model") + "New")
    artifact.setdefault("backend", request["backend"])
    artifact.setdefault("device", str(device))
    return artifact


def execute(root, phase, device):
    os.environ.pop("LD_PRELOAD", None)  # Never instrument compiler subprocesses.
    import torch

    torch.cuda.set_device(torch.device(device))
    from kernelgym.backend.kernelbench.dispatcher import KernelBenchBackend
    from kernelgym.toolkit.kernelbench.execution_policy import EXECUTION_POLICY_VERSION, tf32_execution_context
    from kernelgym.toolkit.kernelbench.pipeline import _sanitize_compile_artifact
    from kernelgym.toolkit.kernelbench.runtime_graph_capture import (
        configuration_snapshot,
        fingerprints,
        model_tensor_state,
        stable_digest,
    )

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
        "backend": request.get("backend"),
        "precision": request.get("precision"),
        "entry_point": request.get("entry_point") or "Model",
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
        artifact = _prepare_precompiled_artifact(request, device) or backend.compile(
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
        # This is a provenance locator for the loaded artifact, not a claim
        # that the control and traced binary are byte-identical.
        result["artifact_provenance"] = stable_digest(_sanitize_compile_artifact(artifact))
        handle = backend.load(artifact, device=device)
        torch.manual_seed(original["model_seed"])
        torch.cuda.manual_seed(original["model_seed"])
        model = backend.create_model(handle, init, device=device)
        model.train(original["training"])
        # Loading the entire archive on the target device preserves shared
        # storage/views between input tensors, unlike individual .to() calls.
        capsule = torch.load(root / "capsule.pt", map_location=device, weights_only=True)
        if original["state_mode"] == "restore":
            saved = capsule["tensor_state"]
            target = model_tensor_state(model)
            if set(saved) != set(target):
                raise ValueError("registered_tensor_state_keys_mismatch")
            with torch.no_grad():
                for name, value in saved.items():
                    if target[name].shape != value.shape or target[name].dtype != value.dtype:
                        raise ValueError("registered_tensor_state_shape_or_dtype_mismatch: " + name)
                    target[name].copy_(value)
        inputs = capsule["inputs"]
        torch.set_rng_state(capsule["cpu_rng"].cpu())
        torch.cuda.set_rng_state(capsule["cuda_rng"].cpu(), device)
        result["input"] = fingerprints(inputs)
        result["state"] = fingerprints(model_tensor_state(model))
        result["rng"] = fingerprints({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state(device)})
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
        for name, value in model_tensor_state(model).items():
            observer.observe(value, name)
        write(root / f"{phase}_allocations.json", observer.as_dict())
        if phase == "trace":
            enable = ctypes.CDLL(None).runtime_trace_set_enabled
            enable.argtypes = [ctypes.c_int]
        policy = {}
        with tf32_execution_context(policy, stage="diagnostic"), torch.no_grad():
            replay_metadata = {
                "execution_policy": EXECUTION_POLICY_VERSION,
                "correctness_tf32_state_forced": policy["diagnostic_tf32_state_forced"],
                "precision": request.get("precision"),
            }
            result["configuration"] = configuration_snapshot(model, device, request, replay_metadata)
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
        "status": "unavailable",
        "unknowns": [],
        "training_state_equivalence": "registered_tensor_state_only",
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
    normal = json.loads(normal_path.read_text()) if normal_path.exists() else None
    capsule = json.loads((root / "capsule.json").read_text())
    captured_request = capsule.get("request_identity") or {
        "task_sha256": capsule.get("task_sha256"),
        "candidate_source_sha256": capsule.get("candidate_source_sha256"),
    }
    captured_configuration = capsule.get("configuration_snapshot") or capsule.get("execution_configuration")
    expected = {
        "task_sha256": hashlib.sha256(request["reference_code"].encode("utf-8")).hexdigest(),
        "candidate_source_sha256": hashlib.sha256(request["kernel_code"].encode("utf-8")).hexdigest(),
    }
    captured_request_binding = (
        (
            captured_request.get("task_sha256") == expected["task_sha256"]
            and captured_request.get("candidate_source_sha256") == expected["candidate_source_sha256"]
        )
        if captured_request.get("task_sha256") is not None
        and captured_request.get("candidate_source_sha256") is not None
        else None
    )
    # A phase can only claim request binding after it independently carries
    # the request digest and an actual candidate artifact provenance.
    control_request_binding = _phase_binding(control, request) if control.get("artifact_provenance") else None
    trace_request_binding = _phase_binding(traced, request) if traced.get("artifact_provenance") else None
    control["request_binding"] = control_request_binding
    traced["request_binding"] = trace_request_binding
    scored = {
        "request_binding": _and_facts(captured_request_binding, control_request_binding),
        "candidate_binding": _same(capsule.get("candidate_provenance"), control.get("artifact_provenance")),
        "inputs": _same(capsule.get("input"), control.get("input")),
        # This is equality over the explicitly declared registered-tensor
        # scope, not a claim about arbitrary Python attributes or external
        # state.  A known required omission/budget failure must be emitted by
        # capture as a missing state fingerprint, which becomes null here.
        "initial_state": (
            (capsule.get("state") == control.get("state") and capsule.get("rng") == control.get("rng"))
            if capsule.get("state") is not None
            and control.get("state") is not None
            and capsule.get("rng") is not None
            and control.get("rng") is not None
            else None
        ),
        "execution_configuration": _same(captured_configuration, control.get("configuration")),
        "outputs": _same(normal, control.get("output")),
    }
    trace = {
        "request_binding": _and_facts(control_request_binding, trace_request_binding),
        "candidate_binding": _same(control.get("artifact_provenance"), traced.get("artifact_provenance")),
        "inputs": _same(control.get("input"), traced.get("input")),
        "initial_state": (
            (control.get("state") == traced.get("state") and control.get("rng") == traced.get("rng"))
            if control.get("state") is not None
            and traced.get("state") is not None
            and control.get("rng") is not None
            and traced.get("rng") is not None
            else None
        ),
        "execution_configuration": _same(control.get("configuration"), traced.get("configuration")),
        "outputs": _same(control.get("output"), traced.get("output")),
    }
    result["alignment"] = {
        "scored_control": _aggregate_checks(scored),
        "control_trace": _aggregate_checks(trace),
        "checks": {"scored_control": scored, "control_trace": trace},
        "evidence": {
            "state_scope": capsule.get("state_scope")
            or "registered_parameters_and_buffers; arbitrary_python_attributes_not_snapshotted",
            "scored_output_fingerprint_present": normal is not None,
            "candidate_artifact_provenance_is_not_binary_equivalence": True,
        },
    }
    result["identity"] = {
        **expected,
        "input_signature": value_digest(control.get("input")),
        "state_signature": value_digest({"tensors": control.get("state"), "rng": control.get("rng")}),
        "environment_signature": value_digest(_environment_material(control.get("configuration") or {})),
    }
    if request.get("collector_sha256") is not None:
        result["identity"]["collector_sha256"] = request["collector_sha256"]
    result["cost"] = {
        "control_forward_seconds": control.get("forward_wall_seconds"),
        "trace_forward_seconds": traced.get("forward_wall_seconds"),
        "report_seconds": None,
    }
    metadata = root / "trace.jsonl"
    allocation_path = root / "trace_allocations.json"
    try:
        if control.get("status") != "completed" or traced.get("status") != "completed":
            raise ValueError("control_or_trace_execution_incomplete")
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
            "task": result["identity"]["task_sha256"],
            "input_signature": result["identity"]["input_signature"],
            "environment_signature": result["identity"]["environment_signature"],
        }
        graph = build_graph(root / "trace", allocations, context)
        graph, graph_reason = _full_graph_summary(
            graph,
            max_nodes=request["options"]["max_summary_nodes"],
            max_bytes=120 * 1024,
        )
        if graph_reason:
            result["unknowns"].append(graph_reason)
            graph = None
        if graph is None:
            raise ValueError("runtime_graph_not_self_contained")
        write(root / "graph.json", graph)
        result["graph_artifact"] = {
            "path": str(root / "graph.json"),
            "sha256": sha(root / "graph.json"),
            "bytes": (root / "graph.json").stat().st_size,
        }
        result["graph"] = graph
        # Valid complete graphs may still be partial attribution evidence when
        # any alignment check is false/null or a node carries an unknown.
        if result["alignment"]["scored_control"] and result["alignment"]["control_trace"]:
            result["status"] = "ok"
        else:
            result["status"] = "partial"
    except Exception as exc:
        result["unknowns"].append(f"{type(exc).__name__}: {str(exc)[:1000]}")
    identity_required = {
        "task_sha256",
        "candidate_source_sha256",
        "input_signature",
        "state_signature",
        "environment_signature",
        "collector_sha256",
    }
    identity_complete = (
        identity_required <= set(result["identity"])
        and control.get("input") is not None
        and control.get("state") is not None
        and control.get("rng") is not None
        and control.get("configuration") is not None
        and all(
            isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
            for value in result["identity"].values()
        )
    )
    if result["status"] in {"ok", "partial"} and not identity_complete:
        result.pop("graph", None)
        result.pop("graph_artifact", None)
        result["status"] = "unavailable"
        result["unknowns"].append("runtime_graph_identity_incomplete")
    result["unknowns"] = sorted(set(result["unknowns"]))
    if len(canonical(result).encode("utf-8")) > 120 * 1024:
        # The envelope has lost the only self-contained graph and must never
        # masquerade as partial/ok.  Keep identity for binding diagnostics.
        result.pop("graph", None)
        result.pop("graph_artifact", None)
        result["status"] = "unavailable"
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
