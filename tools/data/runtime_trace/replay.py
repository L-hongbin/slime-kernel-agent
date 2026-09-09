"""Replay one trusted saved TVM-FFI submission under an optional NVBit scope.

Run in a disposable GPU process with a matching KernelGym source manifest. No
services, feedback, precheck overrides, or scored timing are involved.
"""

import argparse
import ctypes
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernelgym-root", type=Path, required=True)
    p.add_argument("--submission", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--trace", action="store_true")
    p.add_argument("--tracer", type=Path)
    a = p.parse_args()
    if a.trace and os.environ.get("RUNTIME_TRACE_START_ENABLED") != "0":
        raise ValueError("TVM-FFI replay requires RUNTIME_TRACE_START_ENABLED=0 before process start")
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = a.kernelgym_root / "source_manifest.json"
    m = json.loads(manifest.read_text())
    for f, h in m.items():
        if sha(a.kernelgym_root / f) != h:
            raise RuntimeError("KernelGym source synchronization mismatch: " + f)
    preload = os.environ.pop("LD_PRELOAD", "")
    if a.trace and (a.tracer is None or str(a.tracer.resolve()) not in Path("/proc/self/maps").read_text()):
        raise RuntimeError("expected tracer is not mapped into this process")
    sys.path.insert(0, str(a.kernelgym_root))
    import torch
    from kernelgym.backend.kernelbench.tvm_ffi_backend import KernelBenchTvmFfiBackend
    from kernelgym.toolkit.kernelbench.component_trace import ComponentObserver
    from kernelgym.toolkit.kernelbench.exec_types import set_seed
    from kernelgym.toolkit.kernelbench.execution_policy import tf32_execution_context
    from kernelgym.toolkit.kernelbench.loading import load_original_model_and_inputs

    payload = json.loads(a.submission.read_text())
    backend = KernelBenchTvmFfiBackend()
    result = {
        "identity": payload["identity"],
        "payload_sha256": sha(a.submission),
        "runner_sha256": sha(__file__),
        "kernelgym_manifest_sha256": sha(manifest),
        "verified_source_files": len(m),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "trace": a.trace,
        "preload_hashes": {f: sha(f) for f in preload.split(":") if f and Path(f).is_file()},
    }
    artifact = backend.compile(
        payload["custom_code"],
        device="cuda:0",
        precision="fp32",
        entry_point="ModelNew",
        enable_compile_artifact_cache=True,
    )
    if not artifact.get("compiled"):
        (a.output / "result.json").write_text(
            json.dumps({**result, "status": "compile_rejected", "artifact": artifact}, indent=2, default=str)
        )
        return 2
    result["tracer_sha256"] = sha(a.tracer) if a.trace else None
    result["so_sha256"] = sha(artifact["so_path"])
    handle = backend.load(artifact, device="cuda:0")
    _, get_init, get_inputs = load_original_model_and_inputs(payload["reference_code"], {})
    set_seed(42)
    init = get_init()
    model = backend.create_model(handle, init, device="cuda:0").eval()
    set_seed(17)
    inputs = backend._move_to_device(get_inputs(), torch.device("cuda:0"))
    allocations = []
    objects = []

    def capture(value, role):
        if isinstance(value, torch.Tensor) and value.is_cuda:
            storage = value.untyped_storage()
            base = storage.data_ptr()
            size = storage.nbytes()
            existing = next((x for x in allocations if x["base"] == base), None)
            if existing and role.startswith("output:"):
                existing.setdefault("aliases", []).append(role)
                if not existing["role"].startswith("input:"):
                    existing["role"] = role
            if not existing:
                allocations.append(
                    {
                        "role": role,
                        "base": base,
                        "bytes": size,
                        "shape": list(value.shape),
                        "stride": list(value.stride()),
                        "dtype": str(value.dtype),
                    }
                )
            objects.append(value)
        elif isinstance(value, (tuple, list)):
            for i, x in enumerate(value):
                capture(x, f"{role}:{i}")
        elif isinstance(value, dict):
            for i, x in enumerate(value.values()):
                capture(x, f"{role}:{i}")

    capture(inputs, "input")
    for i, x in enumerate(model.parameters()):
        capture(x, f"parameter:{i}")
    for i, x in enumerate(model.buffers()):
        capture(x, f"state:{i}")
    observer = ComponentObserver()
    # The existing observer records actual FFI arguments, including scratch
    # tensor storage. Presence supplies allocation identity, never access mode.
    original = observer.invoke

    def invoke(name, func, args, kwargs):
        capture(args, "ffi_argument")
        capture(kwargs, "ffi_keyword")
        return original(name, func, args, kwargs)

    observer.invoke = invoke
    torch.cuda.synchronize()
    start = time.monotonic()
    enable = ctypes.CDLL(None).runtime_trace_set_enabled if a.trace else None
    with torch.no_grad(), tf32_execution_context(result, stage="diagnostic"), observer.activate():
        if enable:
            enable(1)
        output = model(**inputs) if isinstance(inputs, dict) else model(*inputs)
        torch.cuda.synchronize()
        if enable:
            enable(0)
    result["forward_wall_seconds"] = time.monotonic() - start
    capture(output, "output:0")

    def fingerprint(v):
        if isinstance(v, torch.Tensor):
            value = v.detach().contiguous().cpu()
            return {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest(),
                "first_values": value.flatten()[:8].tolist(),
            }
        if isinstance(v, (tuple, list)):
            return [fingerprint(x) for x in v]
        if isinstance(v, dict):
            return {k: fingerprint(x) for k, x in v.items()}
        return v

    result["output"] = fingerprint(output)
    result["status"] = "completed"
    (a.output / "allocations.json").write_text(json.dumps({"allocations": allocations}, indent=2))
    (a.output / "observation.json").write_text(json.dumps(observer.as_dict(), indent=2))
    (a.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
