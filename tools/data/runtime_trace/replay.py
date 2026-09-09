"""Whole-forward replay of trusted reference or TVM-FFI candidate payloads."""

import argparse
import ctypes
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--payload", type=Path, required=True)
    p.add_argument("--kernelgym-root", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tracer", type=Path)
    p.add_argument("--manifest", type=Path, required=True)
    args = p.parse_args()
    for filename, expected in json.loads(args.manifest.read_text()).items():
        if sha(args.manifest.parent / filename) != expected:
            raise RuntimeError("node-local synchronization mismatch: " + filename)
    if args.tracer and os.environ.get("RUNTIME_TRACE_START_ENABLED") != "0":
        raise ValueError("forward replay requires RUNTIME_TRACE_START_ENABLED=0")
    os.environ.pop("LD_PRELOAD", None)
    args.output.mkdir(parents=True, exist_ok=False)
    import numpy as np
    import torch
    from tools.data.runtime_trace.observer import StorageObserver

    def seed(value):
        random.seed(value)
        np.random.seed(value)
        torch.manual_seed(value)
        torch.cuda.manual_seed_all(value)

    def move(value):
        if isinstance(value, torch.Tensor):
            return value.cuda()
        if isinstance(value, (tuple, list)):
            return type(value)(move(x) for x in value)
        if isinstance(value, dict):
            return {k: move(v) for k, v in value.items()}
        return value

    def fingerprint(value):
        if isinstance(value, torch.Tensor):
            cpu = value.detach().contiguous().cpu()
            return {
                "shape": list(value.shape),
                "stride": list(value.stride()),
                "dtype": str(value.dtype),
                "sha256": hashlib.sha256(cpu.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
                "first_values": cpu.flatten()[:8].tolist(),
            }
        if isinstance(value, (list, tuple)):
            return [fingerprint(x) for x in value]
        if isinstance(value, dict):
            return {k: fingerprint(v) for k, v in value.items()}
        return value

    payload = json.loads(args.payload.read_text())
    reference_sha = hashlib.sha256(payload["reference_code"].encode()).hexdigest()
    if payload.get("source_sha256", reference_sha) != reference_sha:
        raise ValueError("reference source hash mismatch")
    namespace = {}
    exec(compile(payload["reference_code"], "<frozen-reference>", "exec"), namespace)
    seed(42)
    init = namespace["get_init_inputs"]() if "get_init_inputs" in namespace else []
    provenance = {
        "payload_sha256": sha(args.payload),
        "runner_sha256": sha(__file__),
        "manifest_sha256": sha(args.manifest),
        "source_sha256": hashlib.sha256(payload["reference_code"].encode()).hexdigest(),
    }
    if "custom_code" in payload:
        if args.kernelgym_root is None:
            raise ValueError("candidate requires isolated KernelGym backend")
        manifest = args.kernelgym_root / "source_manifest.json"
        for filename, expected in json.loads(manifest.read_text()).items():
            if sha(args.kernelgym_root / filename) != expected:
                raise RuntimeError("KernelGym source mismatch: " + filename)
        provenance["kernelgym_manifest_sha256"] = sha(manifest)
        sys.path.insert(0, str(args.kernelgym_root))
        from kernelgym.backend.kernelbench.tvm_ffi_backend import KernelBenchTvmFfiBackend

        backend = KernelBenchTvmFfiBackend()
        artifact = backend.compile(
            payload["custom_code"],
            device="cuda:0",
            precision="fp32",
            entry_point="ModelNew",
            enable_compile_artifact_cache=True,
        )
        if not artifact.get("compiled"):
            (args.output / "result.json").write_text(
                json.dumps(
                    {"status": "compile_or_precheck_failed", "artifact": artifact, "provenance": provenance},
                    default=str,
                    indent=2,
                )
            )
            return 2
        handle = backend.load(artifact, device="cuda:0")
        seed(42)
        model = backend.create_model(handle, init, device="cuda:0").eval()
        provenance["candidate_so_sha256"] = sha(artifact["so_path"])
    else:
        model = (namespace["Model"](**init) if isinstance(init, dict) else namespace["Model"](*init)).cuda()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    seed(17)
    inputs = move(namespace["get_inputs"]())
    input_fingerprint = fingerprint(inputs)
    reference = "custom_code" not in payload
    initial_state = (
        {
            "parameters": {name: fingerprint(value) for name, value in model.named_parameters()},
            "buffers": {name: fingerprint(value) for name, value in model.named_buffers()},
        }
        if reference
        else None
    )
    recorder = StorageObserver(traced=bool(args.tracer))
    recorder.observe(inputs, "input")
    for i, value in enumerate(model.parameters()):
        recorder.observe(value, f"parameter:{i}")
    for i, value in enumerate(model.buffers()):
        recorder.observe(value, f"state:{i}")
    enable = None
    if args.tracer:
        if str(args.tracer.resolve()) not in Path("/proc/self/maps").read_text():
            raise RuntimeError("requested native library is not mapped")
        provenance["tracer_sha256"] = sha(args.tracer)
        enable = ctypes.CDLL(None).runtime_trace_set_enabled
        enable.argtypes = [ctypes.c_int]
        enable.restype = None
    if reference:
        seed(23)
    execution_config = {
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "init_seed": 42,
        "input_seed": 17,
        "forward_seed": 23 if reference else None,
        "training": model.training,
        "grad_enabled_during_forward": False,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    provenance["execution_config_sha256"] = hashlib.sha256(
        json.dumps(execution_config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    torch.cuda.synchronize()
    started = time.monotonic()
    with torch.no_grad():
        if enable:
            enable(1)
            with recorder:
                output = model(**inputs) if isinstance(inputs, dict) else model(*inputs)
        else:
            output = model(**inputs) if isinstance(inputs, dict) else model(*inputs)
        torch.cuda.synchronize()
        if enable:
            enable(0)
    wall = time.monotonic() - started
    recorder.observe(output, "output:0")
    result = {
        "schema": "coarse-replay/v1",
        "status": "completed",
        "identity": payload.get("identity", payload.get("id")),
        "input": input_fingerprint,
        "initial_state": initial_state,
        "execution_config": execution_config,
        "output": fingerprint(output),
        "forward_wall_seconds": wall,
        "training": model.training,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "seeds": [42, 17, 23] if reference else [42, 17],
        "provenance": provenance,
    }
    (args.output / "allocations.json").write_text(json.dumps(recorder.as_dict(), indent=2))
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
