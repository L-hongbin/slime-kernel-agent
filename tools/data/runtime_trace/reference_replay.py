"""Independent generic replay of a trusted exported training reference.

The JSON supplies dataset source, not task-specific extraction instructions.
Run in an isolated process with the native tracer preloaded and initially off.
"""

import argparse
import ctypes
import hashlib
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tracer", type=Path)
    args = parser.parse_args()
    if args.tracer and os.environ.get("RUNTIME_TRACE_START_ENABLED") != "0":
        raise ValueError("reference replay requires RUNTIME_TRACE_START_ENABLED=0 before process start")
    args.output.mkdir(parents=True, exist_ok=False)
    payload = json.loads(args.input.read_text())
    source = payload["reference_code"]
    assert hashlib.sha256(source.encode()).hexdigest() == payload["source_sha256"]
    os.environ.pop("LD_PRELOAD", None)
    import random

    import numpy as np
    import torch

    def seed(n):
        random.seed(n)
        np.random.seed(n)
        torch.manual_seed(n)
        torch.cuda.manual_seed_all(n)

    scope = {}
    exec(compile(source, "<frozen-training-reference>", "exec"), scope)
    seed(42)
    init = scope["get_init_inputs"]() if "get_init_inputs" in scope else []
    model = (scope["Model"](**init) if isinstance(init, dict) else scope["Model"](*init)).cuda()
    seed(17)
    inputs = scope["get_inputs"]()

    def to_device(v):
        if isinstance(v, torch.Tensor):
            return v.cuda()
        if isinstance(v, (list, tuple)):
            return type(v)(to_device(x) for x in v)
        if isinstance(v, dict):
            return {k: to_device(x) for k, x in v.items()}
        return v

    inputs = to_device(inputs)
    allocations = []

    def record(v, role):
        if isinstance(v, torch.Tensor):
            storage = v.untyped_storage()
            base = storage.data_ptr()
            if not any(x["base"] == base for x in allocations):
                allocations.append(
                    {
                        "role": role,
                        "base": base,
                        "bytes": storage.nbytes(),
                        "shape": list(v.shape),
                        "stride": list(v.stride()),
                        "dtype": str(v.dtype),
                    }
                )
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                record(x, f"{role}:{i}")
        elif isinstance(v, dict):
            for i, x in enumerate(v.values()):
                record(x, f"{role}:{i}")

    def fingerprint(v):
        if isinstance(v, torch.Tensor):
            t = v.detach().cpu().contiguous()
            return {
                "shape": list(t.shape),
                "dtype": str(t.dtype),
                "sha256": hashlib.sha256(t.view(torch.uint8).numpy().tobytes()).hexdigest(),
                "first_values": t.flatten()[:8].tolist(),
            }
        if isinstance(v, (list, tuple)):
            return [fingerprint(x) for x in v]
        if isinstance(v, dict):
            return {k: fingerprint(x) for k, x in v.items()}
        return v

    record(inputs, "input")
    for i, p in enumerate(model.parameters()):
        record(p, f"parameter:{i}")
    for i, p in enumerate(model.buffers()):
        record(p, f"state:{i}")
    input_fingerprint = fingerprint(inputs)
    enable = None
    if args.tracer:
        if str(args.tracer.resolve()) not in Path("/proc/self/maps").read_text():
            raise RuntimeError("expected tracer not loaded")
        enable = ctypes.CDLL(None).runtime_trace_set_enabled
        enable.argtypes = [ctypes.c_int]
        enable.restype = None
    torch.cuda.synchronize()
    started = time.monotonic()
    with torch.no_grad():
        if enable:
            enable(1)
        output = model(**inputs) if isinstance(inputs, dict) else model(*inputs)
        torch.cuda.synchronize()
        if enable:
            enable(0)
    wall = time.monotonic() - started
    record(output, "output:0")
    result = {
        "id": payload["id"],
        "source_sha256": payload["source_sha256"],
        "input": input_fingerprint,
        "output": fingerprint(output),
        "forward_wall_seconds": wall,
        "training": model.training,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "tracer_sha256": hashlib.sha256(args.tracer.read_bytes()).hexdigest() if args.tracer else None,
        "seeds": [42, 17],
    }
    (args.output / "result.json").write_text(json.dumps(result, indent=2))
    (args.output / "allocations.json").write_text(json.dumps({"allocations": allocations}, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
