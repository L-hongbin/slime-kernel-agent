#!/usr/bin/env python3
"""Reference selected 10k candidates without renumbering the 13k registry.

The generic reference validator intentionally numbers a materialized input
parquet from zero.  This adapter keeps the canonical 13k parquet as its only
input, selects UUIDs from it, and delegates each row's real five-trial
KernelGym evaluation to that validator's isolated worker implementation.
"""

from __future__ import annotations

import argparse
import collections
import fcntl
import hashlib
import importlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.synthesize import validate_train_mode_contract as reference_core  # noqa: E402

CONTRACT_VERSION = reference_core.CONTRACT_VERSION
RUN_BINDING_VERSION = "operator_structure_10k_reference_selection_binding_v1"
SELECTION_CONTRACT_VERSION = "operator_structure_10k_reference_original_index_v1"
DEFAULT_GENERATOR_MODULE = "tools.data.synthesize.canary.operator_structure_10k_method.generate_operator_structure_10k"
REFERENCE_TRIALS = 5
REFERENCE_SEED = 42
REFERENCE_MAX_DEVICE_MEMORY_GIB = 64.0
_CORE_PATH = Path(reference_core.__file__).resolve()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode())


def _require_sha256(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _nested(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _load_generator(module_name: str) -> Any:
    module = importlib.import_module(module_name)
    path = Path(getattr(module, "__file__", "")).resolve()
    if not path.is_file() or not callable(getattr(module, "replay_manifest", None)):
        raise ValueError("generator must be a file-backed module exporting replay_manifest")
    rows = getattr(module, "EXACT_CANDIDATE_ROWS", None)
    if type(rows) is not int or rows <= 0:
        raise ValueError("generator must expose EXACT_CANDIDATE_ROWS")
    return module


def _read_allowlist(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("UUID allowlist must be nonempty and unique")
    return values


def _manifest_rows(path: Path, expected_rows: int) -> list[dict[str, Any]]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(values) != expected_rows or not all(isinstance(value, dict) for value in values):
        raise ValueError("manifest must be an exact registry-sized JSONL object sequence")
    return values


def _check_pair(index: int, row: Mapping[str, Any], manifest: Mapping[str, Any], generator: Any) -> dict[str, Any]:
    uuid = _nested(row, "extra_info.uuid")
    code = _nested(row, "reward_model.ground_truth")
    if not isinstance(uuid, str) or not uuid or not isinstance(code, str) or not code.strip():
        raise ValueError(f"candidate identity/code missing:{index}")
    if manifest.get("uuid") != uuid or manifest.get("candidate_row_index") != index:
        raise ValueError(f"candidate/manifest original-index mismatch:{index}")
    if manifest.get("generator_source_sha256") != _sha256_file(Path(generator.__file__).resolve()):
        raise ValueError(f"generator source binding mismatch:{index}")
    if manifest.get("reference_sha256") != _sha256_bytes(code.encode()):
        raise ValueError(f"candidate reference hash mismatch:{index}")
    if (
        manifest.get("static_status") != "passed"
        or manifest.get("parent_uuid") is not None
        or manifest.get("training_approved") is not False
    ):
        raise ValueError(f"candidate static/review-only contract mismatch:{index}")
    replay = generator.replay_manifest(manifest)
    if not isinstance(replay, Mapping) or replay.get("code") != code:
        raise ValueError(f"candidate generator replay mismatch:{index}")
    return {
        "candidate_row_index": index,
        "uuid": uuid,
        "reference_sha256": manifest["reference_sha256"],
        "row": dict(row),
        "manifest": dict(manifest),
    }


def tasks(
    candidates_path: Path,
    manifest_path: Path,
    uuid_file: Path,
    *,
    generator_module: str = DEFAULT_GENERATOR_MODULE,
    full_replay: bool = False,
    shard_index: int | None = None,
    shard_count: int | None = None,
) -> list[dict[str, Any]]:
    """Return selected canonical tasks, optionally replaying all 13k rows."""

    if (shard_index is None) != (shard_count is None) or (
        shard_count is not None and (shard_count <= 0 or not 0 <= shard_index < shard_count)
    ):
        raise ValueError("shard selector must be absent or satisfy 0 <= index < count")
    generator = _load_generator(generator_module)
    expected_rows = int(generator.EXACT_CANDIDATE_ROWS)
    parquet = pq.ParquetFile(candidates_path)
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(f"candidate count differs from generator exact count:{expected_rows}")
    rows = pq.read_table(candidates_path).to_pylist()
    manifests = _manifest_rows(manifest_path, expected_rows)
    allowed = _read_allowlist(uuid_file)
    allowed_set = set(allowed)
    by_uuid: dict[str, dict[str, Any]] = {}
    for index, (row, manifest) in enumerate(zip(rows, manifests, strict=True)):
        uuid = _nested(row, "extra_info.uuid")
        if not isinstance(uuid, str) or not uuid or uuid in by_uuid:
            raise ValueError(f"candidate UUID invalid/duplicate:{index}")
        selected_here = uuid in allowed_set and (shard_count is None or index % shard_count == shard_index)
        if full_replay or selected_here:
            checked = _check_pair(index, row, manifest, generator)
        else:
            if manifest.get("uuid") != uuid or manifest.get("candidate_row_index") != index:
                raise ValueError(f"candidate/manifest original-index mismatch:{index}")
            checked = {
                "candidate_row_index": index,
                "uuid": uuid,
                "reference_sha256": manifest.get("reference_sha256"),
                "row": dict(row),
                "manifest": dict(manifest),
            }
        by_uuid[uuid] = checked
    if set(allowed) - set(by_uuid):
        raise ValueError("allowlist contains UUID absent from canonical candidates")
    return [
        by_uuid[uuid]
        for uuid in allowed
        if shard_count is None or by_uuid[uuid]["candidate_row_index"] % shard_count == shard_index
    ]


def _source_binding(generator: Any, launcher_path: Path, launcher_sha256: str) -> dict[str, dict[str, str]]:
    launcher = launcher_path.resolve()
    if not launcher.is_file() or _sha256_file(launcher) != _require_sha256(launcher_sha256, label="launcher SHA"):
        raise ValueError("launcher source binding mismatch")
    return {
        "reference_adapter_source": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256_file(Path(__file__).resolve()),
        },
        "generic_reference_core_source": {"path": str(_CORE_PATH), "sha256": _sha256_file(_CORE_PATH)},
        "generator_source": {
            "path": str(Path(generator.__file__).resolve()),
            "sha256": _sha256_file(Path(generator.__file__).resolve()),
        },
        "launcher_source": {"path": str(launcher), "sha256": launcher_sha256},
    }


def _config(candidates: Path, output: Path, kernelgym_root: Path, launcher_sha256: str, timeout: float) -> Any:
    return reference_core.DriverConfig(
        input_path=candidates,
        output_path=output,
        kernelgym_root=kernelgym_root,
        expected_mode_class=None,
        device="cuda:0",
        max_device_memory_gib=REFERENCE_MAX_DEVICE_MEMORY_GIB,
        timeout_seconds=timeout,
        trials=REFERENCE_TRIALS,
        seed=REFERENCE_SEED,
        training=True,
        launcher_sha256=launcher_sha256,
    )


def _append(handle: Any, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _driver_gpu_evidence() -> dict[str, Any]:
    """Bind a real H20 to failures that occur before the worker reports GPU."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the reference driver")
    torch.cuda.set_device(0)
    name = torch.cuda.get_device_name(0)
    if "H20" not in name:
        raise RuntimeError(f"reference driver requires H20, got:{name}")
    return {
        "device": "cuda:0",
        "name": name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


def _prior_records(
    path: Path, selected: Mapping[str, Mapping[str, Any]], binding: str, evidence: Mapping[str, Any]
) -> dict[str, Mapping[str, Any]]:
    if not path.exists():
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        uuid = record.get("uuid")
        task = selected.get(uuid) if isinstance(uuid, str) else None
        if (
            task is None
            or uuid in result
            or record.get("reference_binding_sha256") != binding
            or record.get("reference_binding_evidence") != evidence
        ):
            raise ValueError(f"invalid resume record:{path}:{number}")
        generic = evidence["generic_reference_contract"]
        sources = ("reference_adapter_source", "generic_reference_core_source", "generator_source", "launcher_source")
        if (
            record.get("contract_version") != CONTRACT_VERSION
            or record.get("contract_fingerprint") != _canonical_sha256(generic)
            or record.get("contract_payload") != generic
            or record.get("validator_source_sha256") != generic["validator_source_sha256"]
            or record.get("launcher_source_sha256") != generic["launcher_source_sha256"]
            or record.get("source_sha256") != generic["source_sha256"]
            or record.get("candidates_sha256") != generic["source_sha256"]
            or record.get("manifest_sha256") != evidence["manifest"]["sha256"]
            or record.get("allowlist_sha256") != evidence["allowlist"]["sha256"]
            or record.get("trials") != REFERENCE_TRIALS
            or record.get("seed") != REFERENCE_SEED
            or record.get("training") is not True
            or record.get("device") != "cuda:0"
            or record.get("max_device_memory_gib") != REFERENCE_MAX_DEVICE_MEMORY_GIB
            or any(
                record.get(f"{name}_sha256") != evidence[name]["sha256"]
                for name in sources
                if name != "launcher_source"
            )
            or record.get("candidate_row_index") != task["candidate_row_index"]
            or record.get("row_index") != task["candidate_row_index"]
            or record.get("reference_sha256") != task["reference_sha256"]
            or type(record.get("passed")) is not bool
            or not isinstance(record.get("status"), str)
            or (record.get("passed") is True and record.get("status") != "passed")
        ):
            raise ValueError(f"resume candidate identity mismatch:{path}:{number}")
        driver_gpu = record.get("driver_gpu")
        if (
            not isinstance(driver_gpu, Mapping)
            or driver_gpu.get("device") != "cuda:0"
            or "H20" not in str(driver_gpu.get("name"))
            or not isinstance(driver_gpu.get("cuda_visible_devices"), str)
            or not driver_gpu["cuda_visible_devices"]
        ):
            raise ValueError(f"resume driver GPU evidence mismatch:{path}:{number}")
        result[uuid] = record
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.device != "cuda:0"
        or args.trials != REFERENCE_TRIALS
        or args.seed != REFERENCE_SEED
        or args.timeout_seconds <= 0
    ):
        raise ValueError("require cuda:0, exactly 5 trials, seed 42, and positive timeout")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count or not args.execution_host.strip():
        raise ValueError("invalid global shard identity or execution host")
    candidates, manifest, allowlist, output = (
        args.candidates.resolve(),
        args.manifest.resolve(),
        args.uuid_file.resolve(),
        args.output.resolve(),
    )
    generator = _load_generator(args.generator_module)
    selected = tasks(
        candidates,
        manifest,
        allowlist,
        generator_module=args.generator_module,
        full_replay=False,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    if not selected:
        raise ValueError("global shard selection is empty")
    source_sha, manifest_sha, allowlist_sha = _sha256_file(candidates), _sha256_file(manifest), _sha256_file(allowlist)
    binding_sources = _source_binding(
        generator, args.launcher_source_path, _require_sha256(args.launcher_sha256, label="launcher SHA")
    )
    config = _config(candidates, output, args.kernelgym_root.resolve(), args.launcher_sha256, args.timeout_seconds)
    kernelgym = reference_core.kernelgym_contract_metadata(config.kernelgym_root)
    generic_payload = reference_core.contract_payload(config, source_sha256=source_sha, kernelgym=kernelgym)
    generic_fingerprint = _canonical_sha256(generic_payload)
    driver_gpu = _driver_gpu_evidence()
    evidence = {
        "binding_version": RUN_BINDING_VERSION,
        "selection_contract_version": SELECTION_CONTRACT_VERSION,
        **binding_sources,
        "generator_module": args.generator_module,
        "execution_host": args.execution_host,
        "global_shard_index": args.shard_index,
        "global_shard_count": args.shard_count,
        "candidates": {"path": str(candidates), "sha256": source_sha},
        "manifest": {"path": str(manifest), "sha256": manifest_sha},
        "allowlist": {"path": str(allowlist), "sha256": allowlist_sha},
        "generic_reference_contract": generic_payload,
    }
    binding = _canonical_sha256(evidence)
    output.parent.mkdir(parents=True, exist_ok=True)
    selected_by_uuid = {task["uuid"]: task for task in selected}
    counts: collections.Counter[str] = collections.Counter()
    executed = resumed = 0
    with output.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another reference worker owns output:{output}") from exc
        handle.seek(0)
        prior = _prior_records(output, selected_by_uuid, binding, evidence)
        handle.seek(0, 2)
        for task in selected:
            uuid = task["uuid"]
            if uuid in prior:
                resumed += 1
                counts["passed" if prior[uuid]["passed"] else "failed"] += 1
                continue
            identity = reference_core._row_identity(task["candidate_row_index"], task["row"], config)
            if identity.get("uuid") != uuid or identity.get("reference_sha256") != task["reference_sha256"]:
                raise ValueError(f"generic identity drift:{uuid}")
            payload = {
                "kernelgym_root": str(Path(kernelgym["root"])),
                "kernelgym_contract": kernelgym,
                "reference_code": identity.pop("reference_code"),
                "entry_point": identity["entry_point"],
                "device": config.device,
                "max_device_memory_gib": config.max_device_memory_gib,
                "trials": config.trials,
                "seed": config.seed,
                "training": config.training,
            }
            worker = reference_core._run_worker_subprocess(payload, timeout_seconds=config.timeout_seconds)
            if worker.get("passed") is True:
                reasons = reference_core.validate_memory_guard_evidence(worker.get("memory_guard"))
                if reasons:
                    worker["passed"] = False
                    worker["status"] = "memory_guard_failed"
                    worker.setdefault("failure_reasons", []).extend(reasons)
            record = {
                "contract_version": CONTRACT_VERSION,
                "contract_fingerprint": generic_fingerprint,
                "contract_payload": generic_payload,
                "validator_source_sha256": _sha256_file(_CORE_PATH),
                "launcher_source_sha256": args.launcher_sha256,
                "source_path": str(candidates),
                "source_sha256": source_sha,
                "candidates_sha256": source_sha,
                "manifest_sha256": manifest_sha,
                "allowlist_sha256": allowlist_sha,
                "candidate_row_index": task["candidate_row_index"],
                "reference_binding_sha256": binding,
                "reference_binding_evidence": evidence,
                "reference_adapter_source_sha256": binding_sources["reference_adapter_source"]["sha256"],
                "generic_reference_core_source_sha256": binding_sources["generic_reference_core_source"]["sha256"],
                "generator_source_sha256": binding_sources["generator_source"]["sha256"],
                "driver_gpu": driver_gpu,
                "kernelgym": kernelgym,
                "trials": config.trials,
                "seed": config.seed,
                "training": config.training,
                "device": config.device,
                "max_device_memory_gib": config.max_device_memory_gib,
                **identity,
                **worker,
            }
            _append(handle, record)
            executed += 1
            counts["passed" if record.get("passed") is True else "failed"] += 1
    result = {
        "contract_version": CONTRACT_VERSION,
        "selection_contract_version": SELECTION_CONTRACT_VERSION,
        "reference_binding_sha256": binding,
        "execution_host": args.execution_host,
        "global_shard_index": args.shard_index,
        "global_shard_count": args.shard_count,
        "source_binding": binding_sources,
        "kernelgym": kernelgym,
        "selected": len(selected),
        "executed": executed,
        "resumed": resumed,
        "passed": counts["passed"],
        "failed": counts["failed"],
        "all_passed": counts["failed"] == 0,
        "output": str(output),
    }
    if result["selected"] != result["executed"] + result["resumed"]:
        raise AssertionError("reference selected/executed/resumed accounting mismatch")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--uuid-file", type=Path, required=True)
    parser.add_argument("--generator-module", default=DEFAULT_GENERATOR_MODULE)
    parser.add_argument("--kernelgym-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trials", type=int, default=REFERENCE_TRIALS)
    parser.add_argument("--seed", type=int, default=REFERENCE_SEED)
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    parser.add_argument("--launcher-sha256", required=True)
    parser.add_argument("--launcher-source-path", type=Path, required=True)
    parser.add_argument("--execution-host", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(values)
    reference_core._install_driver_signal_handlers()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
