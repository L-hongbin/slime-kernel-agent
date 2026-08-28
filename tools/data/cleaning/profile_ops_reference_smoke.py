#!/usr/bin/env python3
"""Profile one reference construction and forward per ops-dataset row.

This is a diagnostic companion to :mod:`tools.data.cleaning.pipeline`. It deliberately
does not assign an acceptance verdict: each row runs in a disposable process
and records where a basic reference smoke test spends time or fails.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from tools.data.cleaning.runtime_validation import (
    _all_finite,
    _build_model,
    _exec_ops_code,
    _invoke_model,
    _normalize_forward_inputs,
    _seed_torch,
    _snapshot_output,
    _to_device,
    extract_python_code,
)


def _nested(row: dict[str, Any], path: str, default: Any = None) -> Any:
    value: Any = row
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value


def _exception_detail(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {' '.join(str(exc).split())}"[:2000]


def _profile_child(sender: Any, code: str, entry_point: str, device: str, seed: int) -> None:
    started = time.perf_counter()
    previous = started
    stages: dict[str, float] = {}
    try:
        # Reference snippets can print or warn heavily.  The structured record
        # is the reviewable evidence, so keep shard logs bounded.
        devnull = open(os.devnull, "w", encoding="utf-8")
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        import torch

        torch.set_num_threads(1)

        def mark(name: str, *, synchronize: bool = False) -> None:
            nonlocal previous
            if synchronize and device.startswith("cuda"):
                torch.cuda.synchronize()
            now = time.perf_counter()
            stages[name] = now - previous
            previous = now

        mark("torch_import")
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
            mark("cuda_initialize", synchronize=True)
        extracted = extract_python_code(code, entry_point=entry_point)
        namespace = _exec_ops_code(extracted, entry_point=entry_point, device=device)
        mark("compile_exec")
        _seed_torch(seed, device)
        init_inputs = namespace["get_init_inputs"]()
        mark("get_init_inputs", synchronize=True)
        inputs = namespace["get_inputs"]()
        mark("get_inputs", synchronize=True)
        init_inputs = [] if init_inputs is None else init_inputs
        inputs = [] if inputs is None else inputs
        if not isinstance(init_inputs, (list, tuple)):
            raise TypeError("get_init_inputs() must return list/tuple")
        init_inputs = _to_device(list(init_inputs), device)
        inputs = _normalize_forward_inputs(inputs, device)
        mark("inputs_to_device", synchronize=True)
        model = _build_model(namespace["__entry_point__"], init_inputs)
        mark("instantiate")
        model = model.to(device) if hasattr(model, "to") else model
        if hasattr(model, "eval"):
            model.eval()
        mark("model_to_device", synchronize=True)
        with torch.no_grad():
            output = _snapshot_output(_invoke_model(model, inputs))
        mark("one_forward", synchronize=True)
        peak = int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else None
        sender.send(
            {
                "status": "passed",
                "detail": "",
                "stage_seconds": stages,
                "total_seconds": time.perf_counter() - started,
                "peak_memory_bytes": peak,
                "output_all_finite": _all_finite(output),
            }
        )
    except BaseException as exc:  # noqa: BLE001 - process isolation is the safety boundary
        try:
            sender.send(
                {
                    "status": "failed",
                    "detail": _exception_detail(exc),
                    "stage_seconds": stages,
                    "total_seconds": time.perf_counter() - started,
                    "peak_memory_bytes": None,
                }
            )
        except BaseException:
            pass
    finally:
        sender.close()


def _profile_one(code: str, entry_point: str, *, device: str, seed: int, timeout: float) -> dict[str, Any]:
    context = mp.get_context("fork")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(
        target=_profile_child,
        args=(sender, code, entry_point, device, seed),
        daemon=True,
    )
    started = time.monotonic()
    process.start()
    sender.close()
    result: dict[str, Any] | None = None
    while result is None:
        if receiver.poll(0.05):
            try:
                result = receiver.recv()
            except (EOFError, OSError):
                result = {"status": "failed", "detail": "worker_pipe_closed"}
        elif not process.is_alive():
            result = {"status": "failed", "detail": f"worker_exit:{process.exitcode}"}
        elif time.monotonic() - started > timeout:
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
            result = {
                "status": "timeout",
                "detail": f"timeout_after_{timeout:g}s",
                "stage_seconds": {},
                "total_seconds": timeout,
                "peak_memory_bytes": None,
            }
    process.join(timeout=2)
    receiver.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")
    if args.output_jsonl.exists() and not args.overwrite:
        raise FileExistsError(f"output exists (pass --overwrite): {args.output_jsonl}")

    rows = pq.read_table(
        args.input,
        columns=["reward_model", "extra_info"],
    ).to_pylist()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_jsonl.with_name(f".{args.output_jsonl.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row_index, row in enumerate(rows):
            code = _nested(row, "reward_model.ground_truth")
            entry_point = _nested(row, "extra_info.entry_point", "Model")
            uuid = _nested(row, "extra_info.uuid")
            result = _profile_one(
                code,
                entry_point,
                device=args.device,
                seed=args.seed,
                timeout=args.timeout,
            )
            record = {"row_index": row_index, "uuid": uuid, "entry_point": entry_point, **result}
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            handle.flush()
            print(
                f"profiled {row_index + 1}/{len(rows)} rows; "
                f"status={result['status']} total={result.get('total_seconds', 0):.3f}s",
                flush=True,
            )
    os.replace(temporary, args.output_jsonl)


if __name__ == "__main__":
    main()
