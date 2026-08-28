#!/usr/bin/env python3
"""CUDA liveness proof for the KernelBench-gap operator canary.

The gap registry has its own static contracts, but deliberately reuses the
reviewed semantic-operator runtime core for subprocess isolation, persistent
train-mode paired execution, exact control/trace comparison, registered-state
checks, and declared-ATen provenance to the returned Tensor.  This adapter
does *not* modify that core: both source files, the gap generator, and the
launcher are bound into every result before resume is allowed.
"""

from __future__ import annotations

import argparse
import collections
import fcntl
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.synthesize.kernelbench_gap_method import generate_kernelbench_gap as generator  # noqa: E402
from tools.data.synthesize.semantic_operator_method import validate_semantic_liveness as runtime_core  # noqa: E402

CONTRACT_VERSION = "kernelbench_gap_runtime_liveness_v1"
RUN_BINDING_VERSION = "kernelbench_gap_runtime_binding_v1"
MAX_AUTHORIZED_CANDIDATES = 1_000
MIN_LIVENESS_TRIALS = 3
REQUIRED_LIVENESS_SEED = 17
EXECUTION_CONTROLS = dict(runtime_core.EXECUTION_CONTROLS)
_RUNTIME_CORE_PATH = Path(runtime_core.__file__).resolve()


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


def _nested(value: Any, path: str) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _require_sha256(value: str, *, label: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _generator_constant(name: str, expected: Any | None = None) -> Any:
    if not hasattr(generator, name):
        raise ValueError(f"kernelbench-gap generator lacks required constant:{name}")
    value = getattr(generator, name)
    if expected is not None and value != expected:
        raise ValueError(f"kernelbench-gap generator constant mismatch:{name}:{value!r}")
    return value


def _validate_declared_ops(value: Any, index: int) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"declared ops missing:{index}")
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    expected_fields = {
        "op_id",
        "source_calls",
        "runtime_identities",
        "min_calls_per_trial",
        "must_reach_returned_output",
    }
    for declared in value:
        if not isinstance(declared, Mapping) or set(declared) != expected_fields:
            raise ValueError(f"declared op schema mismatch:{index}")
        op_id = declared.get("op_id")
        source_calls = declared.get("source_calls")
        identities = declared.get("runtime_identities")
        if not isinstance(op_id, str) or not op_id or op_id in seen:
            raise ValueError(f"declared op id invalid:{index}:{op_id!r}")
        if (
            not isinstance(source_calls, list)
            or not source_calls
            or not all(isinstance(item, str) and item for item in source_calls)
        ):
            raise ValueError(f"declared source calls invalid:{index}:{op_id}")
        if type(declared.get("min_calls_per_trial")) is not int or declared["min_calls_per_trial"] <= 0:
            raise ValueError(f"declared op minimum invalid:{index}:{op_id}")
        if declared.get("must_reach_returned_output") is not True:
            raise ValueError(f"declared op output contract invalid:{index}:{op_id}")
        if not isinstance(identities, list) or not identities:
            raise ValueError(f"declared ATen identities absent:{index}:{op_id}")
        for identity in identities:
            if (
                not isinstance(identity, Mapping)
                or set(identity) != {"schema", "overload"}
                or not isinstance(identity.get("schema"), str)
                or not identity["schema"].startswith("aten::")
                or not isinstance(identity.get("overload"), str)
            ):
                raise ValueError(f"declared ATen identity invalid:{index}:{op_id}:{identity!r}")
        seen.add(op_id)
        validated.append(dict(declared))
    return validated


def _tasks(candidates_path: Path, manifest_path: Path) -> list[dict[str, Any]]:
    expected_rows = _generator_constant("EXACT_CANARY_ROWS", MAX_AUTHORIZED_CANDIDATES)
    family_quotas = _generator_constant("FAMILY_QUOTAS")
    if not isinstance(family_quotas, Mapping) or sum(family_quotas.values()) != expected_rows:
        raise ValueError("kernelbench-gap family quotas do not form the exact canary")
    templates = _generator_constant("TEMPLATES")
    registered_templates = {getattr(item, "template_id", None) for item in templates}
    if not registered_templates or None in registered_templates:
        raise ValueError("kernelbench-gap template registry is invalid")

    parquet = pq.ParquetFile(candidates_path)
    if parquet.metadata.num_rows != expected_rows:
        raise ValueError(
            f"kernelbench-gap canary must contain exactly {expected_rows} rows:{parquet.metadata.num_rows}"
        )
    candidates = pq.read_table(candidates_path).to_pylist()
    manifests = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(candidates) != expected_rows or len(manifests) != expected_rows:
        raise ValueError(f"candidate/manifest exact count mismatch:{len(candidates)}:{len(manifests)}")

    generator_sha = _sha256_file(Path(generator.__file__).resolve())
    families: collections.Counter[str] = collections.Counter()
    template_counts: collections.Counter[str] = collections.Counter()
    tasks: list[dict[str, Any]] = []
    for index, (row, manifest) in enumerate(zip(candidates, manifests, strict=True)):
        if not isinstance(manifest, Mapping):
            raise ValueError(f"manifest is not an object:{index}")
        uuid = _nested(row, "extra_info.uuid")
        code = _nested(row, "reward_model.ground_truth")
        entry_point = _nested(row, "extra_info.entry_point")
        if manifest.get("candidate_row_index") != index or manifest.get("uuid") != uuid:
            raise ValueError(f"candidate/manifest identity mismatch:{index}")
        if (
            manifest.get("manifest_contract_version") != _generator_constant("MANIFEST_VERSION")
            or manifest.get("generator_contract_version") != _generator_constant("CONTRACT_VERSION")
            or manifest.get("runtime_contract_version") != _generator_constant("RUNTIME_CONTRACT_VERSION")
            or manifest.get("primary_intervention") != "semantic_operator"
            or manifest.get("lineage_kind") != "standalone_semantic_synthetic"
            or manifest.get("method") != "kernelbench_gap_canary"
        ):
            raise ValueError(f"kernelbench-gap manifest contract mismatch:{index}")
        if manifest.get("generator_source_sha256") != generator_sha:
            raise ValueError(f"generator source binding mismatch:{index}")
        if manifest.get("parent_uuid") is not None:
            raise ValueError(f"kernelbench-gap task is not parentless:{index}")
        if manifest.get("training_approved") is not False or manifest.get("structured_output_deferred") is not True:
            raise ValueError(f"review-only/structured-output decision mismatch:{index}")
        if manifest.get("final_output_contract") != {"kind": "single_tensor", "finite_required": True}:
            raise ValueError(f"final output contract mismatch:{index}")
        if manifest.get("static_status") != "passed" or not isinstance(manifest.get("static_proof"), Mapping):
            raise ValueError(f"static proof is missing or failed:{index}")
        if not all(isinstance(value, str) and value for value in (uuid, code, entry_point)):
            raise ValueError(f"invalid candidate code identity:{index}")
        if entry_point != "Model" or _sha256_bytes(code.encode()) != manifest.get("reference_sha256"):
            raise ValueError(f"candidate reference binding mismatch:{index}")

        family, template_id = manifest.get("primary_family"), manifest.get("template_id")
        if family not in family_quotas or template_id not in registered_templates:
            raise ValueError(f"unregistered family/template:{index}:{family}:{template_id}")
        mode_behavior = manifest.get("mode_behavior")
        if mode_behavior not in {"stateless", "train_stateful", "recurrent_state"}:
            raise ValueError(f"invalid mode behavior:{index}:{mode_behavior!r}")
        declared_ops = _validate_declared_ops(manifest.get("declared_ops"), index)
        families[str(family)] += 1
        template_counts[str(template_id)] += 1
        tasks.append(
            {
                "candidate_row_index": index,
                "uuid": uuid,
                "reference_code": code,
                "entry_point": entry_point,
                "template_id": template_id,
                "primary_family": family,
                "mode_behavior": mode_behavior,
                "declared_ops": declared_ops,
            }
        )
    if dict(families) != dict(family_quotas):
        raise ValueError(f"family quota mismatch:{dict(families)}")
    if any(count <= 0 for count in template_counts.values()) or set(template_counts) != registered_templates:
        raise ValueError(f"template registry coverage mismatch:{dict(template_counts)}")
    return tasks


def _load_allowlist(path: Path) -> set[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("UUID allowlist must be non-empty and unique")
    return set(values)


def _append(handle: Any, record: Mapping[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def _failure_signature(record: Mapping[str, Any]) -> str | None:
    reason = record.get("reason")
    if reason is None:
        return None
    return _canonical_sha256(
        {
            "status": record.get("status"),
            "reason": reason,
            "error_type": record.get("error_type"),
            "error": record.get("error"),
        }
    )


def _source_binding(launcher_path: Path, launcher_sha256: str) -> dict[str, Any]:
    adapter_path = Path(__file__).resolve()
    generator_path = Path(generator.__file__).resolve()
    launcher_path = launcher_path.resolve()
    if not launcher_path.is_file():
        raise ValueError(f"launcher source is missing:{launcher_path}")
    observed_launcher_sha = _sha256_file(launcher_path)
    if observed_launcher_sha != launcher_sha256:
        raise ValueError(
            "launcher source binding mismatch:" f"expected={launcher_sha256}:observed={observed_launcher_sha}"
        )
    if not _RUNTIME_CORE_PATH.is_file() or not generator_path.is_file():
        raise ValueError("shared runtime core or generator source is missing")
    return {
        "adapter_source": {"path": str(adapter_path), "sha256": _sha256_file(adapter_path)},
        "shared_runtime_core_source": {
            "path": str(_RUNTIME_CORE_PATH),
            "sha256": _sha256_file(_RUNTIME_CORE_PATH),
            "contract_version": runtime_core.CONTRACT_VERSION,
        },
        "generator_source": {"path": str(generator_path), "sha256": _sha256_file(generator_path)},
        "launcher_source": {"path": str(launcher_path), "sha256": observed_launcher_sha},
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--uuid-file", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trials", type=int, default=MIN_LIVENESS_TRIALS)
    parser.add_argument("--seed", type=int, default=REQUIRED_LIVENESS_SEED)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    parser.add_argument("--launcher-sha256", required=True)
    parser.add_argument("--launcher-source-path", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    # ``_run_subprocess`` owns a new process group per row.  Reuse the shared
    # core's signal handler so a launcher interrupt kills the active worker
    # group instead of leaving a detached CUDA child behind.
    runtime_core._install_driver_signal_handlers()
    if args.trials != MIN_LIVENESS_TRIALS or args.seed != REQUIRED_LIVENESS_SEED or args.timeout_seconds <= 0:
        raise ValueError("require trials exactly 3, seed exactly 17, and positive timeout")
    if args.shard_count <= 0 or not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must satisfy 0 <= index < shard-count")
    launcher_sha = _require_sha256(args.launcher_sha256, label="launcher-sha256")
    source_binding = _source_binding(args.launcher_source_path, launcher_sha)
    tasks = _tasks(args.candidates, args.manifest)
    allowlist = _load_allowlist(args.uuid_file)
    known = {str(task["uuid"]) for task in tasks}
    if missing := allowlist - known:
        raise ValueError(f"allowlisted UUIDs not found:{sorted(missing)[:10]}")
    selected = [task for task in tasks if str(task["uuid"]) in allowlist]
    if not 1 <= len(selected) <= MAX_AUTHORIZED_CANDIDATES:
        raise ValueError(f"selected candidate count must be in [1,{MAX_AUTHORIZED_CANDIDATES}]:{len(selected)}")
    selected = [task for task in selected if int(task["candidate_row_index"]) % args.shard_count == args.shard_index]
    if not selected:
        raise ValueError("shard selection produced no kernelbench-gap validation tasks")

    evidence = {
        "contract_version": CONTRACT_VERSION,
        "binding_version": RUN_BINDING_VERSION,
        **source_binding,
        "candidates_path": str(args.candidates.resolve()),
        "candidates_sha256": _sha256_file(args.candidates),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": _sha256_file(args.manifest),
        "allowlist_path": str(args.uuid_file.resolve()),
        "allowlist_sha256": _sha256_file(args.uuid_file),
        "validation_config": {
            "device": args.device,
            "trials": args.trials,
            "seed": args.seed,
            "timeout_seconds": args.timeout_seconds,
            "max_device_memory_gib": runtime_core.MAX_DEVICE_MEMORY_GIB,
            "persistent_train_mode_models": True,
            "single_tensor_final_output": True,
            "execution_controls": EXECUTION_CONTROLS,
        },
    }
    binding = _canonical_sha256(evidence)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: collections.Counter[str] = collections.Counter()
    executed = resumed = 0
    with args.output.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another liveness worker owns output:{args.output}") from exc
        handle.seek(0)
        prior: dict[str, Mapping[str, Any]] = {}
        selected_by_uuid = {str(task["uuid"]): task for task in selected}
        for line_number, line in enumerate(handle, start=1):
            record = json.loads(line)
            uuid = str(record.get("uuid"))
            if (
                uuid not in selected_by_uuid
                or uuid in prior
                or record.get("contract_version") != CONTRACT_VERSION
                or record.get("validation_binding_sha256") != binding
                or record.get("binding_evidence") != evidence
                or record.get("adapter_source_sha256") != source_binding["adapter_source"]["sha256"]
                or record.get("shared_runtime_core_source_sha256")
                != source_binding["shared_runtime_core_source"]["sha256"]
                or record.get("generator_source_sha256") != source_binding["generator_source"]["sha256"]
                or record.get("launcher_source_sha256") != source_binding["launcher_source"]["sha256"]
                or type(record.get("passed")) is not bool
            ):
                raise ValueError(f"invalid resume record:{args.output}:{line_number}")
            prior[uuid] = record
        handle.seek(0, 2)
        for task in selected:
            uuid = str(task["uuid"])
            if uuid in prior:
                resumed += 1
                counts["passed" if prior[uuid]["passed"] else "failed"] += 1
                continue
            result = runtime_core._run_subprocess(
                {
                    **task,
                    "device": args.device,
                    "trials": args.trials,
                    "seed": args.seed,
                    "max_device_memory_gib": runtime_core.MAX_DEVICE_MEMORY_GIB,
                },
                args.timeout_seconds,
            )
            raw_status = str(result.get("status", "failed"))
            status = "protocol_error" if raw_status == "worker_protocol_error" else raw_status
            if status not in {"passed", "reference_failed", "unsupported", "failed", "timeout", "protocol_error"}:
                status = "failed"
            record = {
                **result,
                "contract_version": CONTRACT_VERSION,
                "raw_runtime_core_contract_version": result.get("contract_version"),
                "status": status,
                "passed": result.get("passed") is True,
                "raw_status": raw_status,
                "failure_stage": None if result.get("passed") is True else "liveness",
                "validation_binding_sha256": binding,
                "binding_evidence": evidence,
                "adapter_source_sha256": source_binding["adapter_source"]["sha256"],
                "shared_runtime_core_source_sha256": source_binding["shared_runtime_core_source"]["sha256"],
                "generator_source_sha256": source_binding["generator_source"]["sha256"],
                "launcher_source_sha256": source_binding["launcher_source"]["sha256"],
                "candidates_sha256": evidence["candidates_sha256"],
                "manifest_sha256": evidence["manifest_sha256"],
                "allowlist_sha256": evidence["allowlist_sha256"],
                "validation_config": evidence["validation_config"],
            }
            record["failure_signature"] = _failure_signature(record)
            _append(handle, record)
            executed += 1
            counts["passed" if record["passed"] else "failed"] += 1
    summary = {
        "contract_version": CONTRACT_VERSION,
        "validation_binding_sha256": binding,
        "source_binding": source_binding,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "selected": len(selected),
        "executed": executed,
        "resumed": resumed,
        "passed": counts["passed"],
        "failed": counts["failed"],
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
