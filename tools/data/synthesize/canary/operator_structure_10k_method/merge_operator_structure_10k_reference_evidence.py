#!/usr/bin/env python3
"""Gather rank-local original-index reference evidence into one authority view."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.synthesize.canary.operator_structure_10k_method import (  # noqa: E402
    validate_operator_structure_10k_reference as adapter,
)

CONTRACT_VERSION = "operator_structure_10k_global_reference_evidence_v1"
SUMMARY_VERSION = "operator_structure_10k_global_reference_summary_v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}:{path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object:{path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid shard JSONL:{path}") from exc
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"shard contains a non-object:{path}")
    return rows


def _pair(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"} or not isinstance(value.get("path"), str):
        raise ValueError(f"invalid source pair:{label}")
    adapter._require_sha256(value.get("sha256"), label=label)
    return {"path": str(value["path"]), "sha256": str(value["sha256"])}


def _current_sources(generator: Any) -> dict[str, dict[str, str]]:
    paths = {
        "reference_adapter_source": Path(adapter.__file__).resolve(),
        "generic_reference_core_source": Path(
            importlib.import_module("tools.data.synthesize.validate_train_mode_contract").__file__
        ).resolve(),
        "generator_source": Path(generator.__file__).resolve(),
        "launcher_source": _REPO_ROOT
        / "tools/data/synthesize/canary/operator_structure_10k_method/launch_operator_structure_10k_reference_rank.sh",
    }
    return {name: {"path": str(path), "sha256": _sha256(path)} for name, path in paths.items()}


def _archive_names() -> dict[str, str]:
    return {
        "reference_adapter_source": "reference_adapter_source.py",
        "generic_reference_core_source": "generic_reference_core_source.py",
        "generator_source": "generator_source.py",
        "launcher_source": "launcher_source.sh",
    }


def _install_artifact(source: Path, destination: Path) -> dict[str, str]:
    digest = _sha256(source)
    if destination.exists():
        if not destination.is_file() or _sha256(destination) != digest:
            raise ValueError(f"authority shard collision:{destination}")
    else:
        try:
            os.link(source, destination)
        except OSError:
            temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
            shutil.copy2(source, temporary)
            if _sha256(temporary) != digest:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"authority shard copy mismatch:{source}") from None
            os.replace(temporary, destination)
    return {"path": str(destination), "sha256": digest}


def _verify_authority_files(contract: Mapping[str, Any]) -> None:
    """Ensure a copied authority remains readable after rank staging is removed."""

    for shard in contract.get("shards", []):
        if not isinstance(shard, Mapping):
            raise ValueError("invalid authority shard")
        path, digest = shard.get("path"), shard.get("sha256")
        if (
            not isinstance(path, str)
            or not isinstance(digest, str)
            or not Path(path).is_file()
            or _sha256(Path(path)) != digest
        ):
            raise ValueError("authority shard is missing or changed")
    for rank in contract.get("rank_evidence", []):
        if not isinstance(rank, Mapping):
            raise ValueError("invalid authority rank evidence")
        for field in ("scheduler_contract", "launcher_summary"):
            value = rank.get(field)
            if (
                not isinstance(value, Mapping)
                or not isinstance(value.get("path"), str)
                or not isinstance(value.get("sha256"), str)
                or not Path(value["path"]).is_file()
                or _sha256(Path(value["path"])) != value["sha256"]
            ):
                raise ValueError(f"authority {field} is missing or changed")
        archives = rank.get("source_archives")
        if not isinstance(archives, Mapping):
            raise ValueError("authority source archives missing")
        for value in archives.values():
            if (
                not isinstance(value, Mapping)
                or not isinstance(value.get("path"), str)
                or not isinstance(value.get("sha256"), str)
                or not Path(value["path"]).is_file()
                or _sha256(Path(value["path"])) != value["sha256"]
            ):
                raise ValueError("authority source archive is missing or changed")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"authority artifact collision:{path}")
    else:
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
    return _sha256(path)


def merge(
    candidates: Path,
    manifest: Path,
    allowlist: Path,
    output: Path,
    rank_dirs: Sequence[Path],
    *,
    generator_module: str,
) -> dict[str, Any]:
    machine_count, gpus_per_machine, scope = 1, 8, "partial"
    if len(rank_dirs) != machine_count:
        raise ValueError("canary reference merge requires one rank directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"authority output must be empty:{output}")
    output.mkdir(parents=True, exist_ok=True)
    candidates, manifest, allowlist = candidates.resolve(), manifest.resolve(), allowlist.resolve()
    generator = adapter._load_generator(generator_module)
    tasks = adapter.tasks(candidates, manifest, allowlist, generator_module=generator_module, full_replay=False)
    expected_rows = int(generator.EXACT_CANDIDATE_ROWS)
    ordered = [task["uuid"] for task in tasks]
    if len(ordered) >= expected_rows:
        raise ValueError("canary authority must be a strict subset")
    task_by_uuid = {task["uuid"]: task for task in tasks}
    total_shards = machine_count * gpus_per_machine
    expected_sources = _current_sources(generator)
    candidate_pair = {"path": str(candidates), "sha256": _sha256(candidates)}
    manifest_pair = {"path": str(manifest), "sha256": _sha256(manifest)}
    allowlist_pair = {"path": str(allowlist), "sha256": _sha256(allowlist)}
    empty = [
        shard
        for shard in range(total_shards)
        if not any(task["candidate_row_index"] % total_shards == shard for task in tasks)
    ]
    if empty:
        raise ValueError(f"allowlist leaves reference shards empty:{empty}")
    all_records: dict[str, Mapping[str, Any]] = {}
    rank_evidence = []
    shards = []
    totals = {key: 0 for key in ("selected", "executed", "resumed", "passed", "failed")}
    common_kernelgym: Mapping[str, Any] | None = None
    for rank, directory in enumerate(rank_dirs):
        directory = directory.resolve()
        scheduler_path = directory / f"scheduler-contract-rank-{rank}.json"
        summary_path = directory / "launcher-summary.json"
        scheduler, summary = _read_json(scheduler_path, "rank scheduler"), _read_json(summary_path, "rank summary")
        owned = set(range(rank * gpus_per_machine, (rank + 1) * gpus_per_machine))
        if (
            scheduler.get("contract_version") != "operator_structure_10k_reference_rank_scheduler_v1"
            or scheduler.get("machine_rank") != rank
            or scheduler.get("machine_count") != machine_count
            or scheduler.get("gpus_per_machine") != gpus_per_machine
            or scheduler.get("global_shard_count") != total_shards
            or set(scheduler.get("global_shard_indices", [])) != owned
            or scheduler.get("generator_module") != generator_module
            or scheduler.get("candidates") != candidate_pair
            or scheduler.get("manifest") != manifest_pair
            or scheduler.get("allowlist") != allowlist_pair
            or scheduler.get("trials") != 5
            or scheduler.get("seed") != 42
            or scheduler.get("training") is not True
            or scheduler.get("expected_mode_class") is not None
            or scheduler.get("max_device_memory_gib") != 64.0
        ):
            raise ValueError(f"rank scheduler policy/binding mismatch:{rank}")
        sources = scheduler.get("source_binding")
        if not isinstance(sources, Mapping) or set(sources) != set(expected_sources):
            raise ValueError(f"rank scheduler source set mismatch:{rank}")
        for name, expected in expected_sources.items():
            if _pair(sources[name], f"scheduler:{rank}:{name}") != expected:
                raise ValueError(f"rank scheduler source hash mismatch:{rank}:{name}")
            archive = directory / _archive_names()[name]
            if not archive.is_file() or _sha256(archive) != expected["sha256"]:
                raise ValueError(f"rank source archive mismatch:{rank}:{name}")
        if (
            summary.get("contract_version") != "operator_structure_10k_reference_rank_summary_v1"
            or summary.get("machine_rank") != rank
            or summary.get("machine_count") != machine_count
            or summary.get("gpus_per_machine") != gpus_per_machine
            or summary.get("global_shard_count") != total_shards
            or summary.get("source_binding") != sources
            or summary.get("kernelgym") != scheduler.get("kernelgym")
        ):
            raise ValueError(f"rank summary identity mismatch:{rank}")
        if common_kernelgym is None:
            common_kernelgym = scheduler.get("kernelgym")
        elif scheduler.get("kernelgym") != common_kernelgym:
            raise ValueError(f"mixed KernelGym bundle:{rank}")
        shard_summaries = summary.get("shards")
        if (
            not isinstance(shard_summaries, list)
            or {item.get("global_shard_index") for item in shard_summaries if isinstance(item, Mapping)} != owned
        ):
            raise ValueError(f"rank summary ownership mismatch:{rank}")
        rank_totals = {key: 0 for key in totals}
        for shard in sorted(owned):
            path = directory / f"shard-{shard:02d}-of-{total_shards:02d}.jsonl"
            records = _read_jsonl(path)
            expected_uuids = {
                uuid for uuid, task in task_by_uuid.items() if task["candidate_row_index"] % total_shards == shard
            }
            observed = {str(record.get("uuid")) for record in records}
            if not expected_uuids or observed != expected_uuids or len(observed) != len(records):
                raise ValueError(f"rank shard UUID partition mismatch:{rank}:{shard}")
            for record in records:
                uuid = str(record["uuid"])
                task = task_by_uuid[uuid]
                evidence = record.get("reference_binding_evidence")
                if uuid in all_records or not isinstance(evidence, Mapping):
                    raise ValueError(f"duplicate/missing reference binding:{uuid}")
                if (
                    record.get("contract_version") != adapter.CONTRACT_VERSION
                    or record.get("row_index") != task["candidate_row_index"]
                    or record.get("candidate_row_index") != task["candidate_row_index"]
                    or record.get("reference_sha256") != task["reference_sha256"]
                    or record.get("source_sha256") != candidate_pair["sha256"]
                    or record.get("candidates_sha256") != candidate_pair["sha256"]
                    or record.get("manifest_sha256") != manifest_pair["sha256"]
                    or record.get("allowlist_sha256") != allowlist_pair["sha256"]
                    or type(record.get("passed")) is not bool
                ):
                    raise ValueError(f"reference record identity mismatch:{uuid}")
                driver_gpu = record.get("driver_gpu")
                if (
                    not isinstance(driver_gpu, Mapping)
                    or driver_gpu.get("device") != "cuda:0"
                    or "H20" not in str(driver_gpu.get("name"))
                    or driver_gpu.get("cuda_visible_devices") != str(shard % gpus_per_machine)
                ):
                    raise ValueError(f"reference driver GPU evidence missing:{uuid}")
                if record.get("kernelgym") != common_kernelgym:
                    raise ValueError(f"reference KernelGym evidence mismatch:{uuid}")
                if record.get("reference_binding_sha256") != _sha256_bytes(_canonical(evidence).encode()):
                    raise ValueError(f"reference binding digest mismatch:{uuid}")
                if (
                    evidence.get("binding_version") != adapter.RUN_BINDING_VERSION
                    or evidence.get("selection_contract_version") != adapter.SELECTION_CONTRACT_VERSION
                    or evidence.get("generator_module") != generator_module
                    or evidence.get("global_shard_index") != shard
                    or evidence.get("global_shard_count") != total_shards
                    or evidence.get("candidates") != candidate_pair
                    or evidence.get("manifest") != manifest_pair
                    or evidence.get("allowlist") != allowlist_pair
                ):
                    raise ValueError(f"reference binding policy mismatch:{uuid}")
                if evidence.get("generic_reference_contract") != record.get("contract_payload"):
                    raise ValueError(f"reference generic contract mismatch:{uuid}")
                evidence_sources = (
                    evidence.get("reference_adapter_source"),
                    evidence.get("generic_reference_core_source"),
                    evidence.get("generator_source"),
                    evidence.get("launcher_source"),
                )
                if any(
                    value != expected_sources[name]
                    for value, name in zip(evidence_sources, expected_sources, strict=True)
                ):
                    raise ValueError(f"reference record source binding mismatch:{uuid}")
                if (
                    record.get("reference_adapter_source_sha256")
                    != expected_sources["reference_adapter_source"]["sha256"]
                    or record.get("generic_reference_core_source_sha256")
                    != expected_sources["generic_reference_core_source"]["sha256"]
                    or record.get("generator_source_sha256") != expected_sources["generator_source"]["sha256"]
                    or record.get("launcher_source_sha256") != expected_sources["launcher_source"]["sha256"]
                ):
                    raise ValueError(f"reference direct source hash mismatch:{uuid}")
                all_records[uuid] = record
            logged = next((item for item in shard_summaries if item.get("global_shard_index") == shard), None)
            if not isinstance(logged, Mapping):
                raise ValueError(f"missing shard summary:{rank}:{shard}")
            for key in rank_totals:
                value = logged.get(key)
                if type(value) is not int or value < 0:
                    raise ValueError(f"invalid shard total:{rank}:{shard}:{key}")
                rank_totals[key] += value
            if (
                logged.get("selected") != len(records)
                or logged.get("selected") != logged.get("executed", -1) + logged.get("resumed", -1)
                or logged.get("selected") != logged.get("passed", -1) + logged.get("failed", -1)
            ):
                raise ValueError(f"shard accounting mismatch:{rank}:{shard}")
            shards.append(
                {
                    "rank": rank,
                    "global_shard_index": shard,
                    "rows": len(records),
                    **_install_artifact(path, output / path.name),
                }
            )
            binding = logged.get("reference_binding_sha256")
            by_shard = summary.get("reference_binding_by_shard")
            if (
                not isinstance(binding, str)
                or not isinstance(by_shard, Mapping)
                or by_shard.get(str(shard)) != binding
            ):
                raise ValueError(f"rank shard binding mismatch:{rank}:{shard}")
        if any(summary.get(key) != value for key, value in rank_totals.items()):
            raise ValueError(f"rank aggregate mismatch:{rank}")
        for key, value in rank_totals.items():
            totals[key] += value
        authority_rank = output / "rank-evidence" / f"rank-{rank:02d}"
        authority_rank.mkdir(parents=True, exist_ok=True)
        scheduler_copy = _install_artifact(scheduler_path, authority_rank / scheduler_path.name)
        summary_copy = _install_artifact(summary_path, authority_rank / summary_path.name)
        archive_copies = {
            name: _install_artifact(directory / archive, authority_rank / archive)
            for name, archive in _archive_names().items()
        }
        rank_evidence.append(
            {
                "machine_rank": rank,
                "execution_host": scheduler.get("execution_host"),
                "gpu_inventory": scheduler.get("gpu_inventory"),
                "scheduler_contract": scheduler_copy,
                "launcher_summary": summary_copy,
                "source_archives": archive_copies,
            }
        )
    if (
        set(all_records) != set(ordered)
        or totals["selected"] != len(ordered)
        or totals["selected"] != totals["executed"] + totals["resumed"]
        or totals["selected"] != totals["passed"] + totals["failed"]
    ):
        raise ValueError("global reference UUID/accounting mismatch")
    if not isinstance(common_kernelgym, Mapping):
        raise ValueError("missing common KernelGym binding")
    merge_source = {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())}
    contract = {
        "contract_version": CONTRACT_VERSION,
        "selection_scope": scope,
        "candidate_rows": expected_rows,
        "machine_count": machine_count,
        "gpus_per_machine": gpus_per_machine,
        "global_shard_count": total_shards,
        "generator_module": generator_module,
        "merge_source": merge_source,
        "candidates": candidate_pair,
        "manifest": manifest_pair,
        "allowlist": allowlist_pair,
        "source_binding": expected_sources,
        "kernelgym": dict(common_kernelgym),
        "rank_evidence": rank_evidence,
        "shards": shards,
    }
    contract_path = output / "global-reference-contract.json"
    contract_sha = _write_immutable(contract_path, contract)
    _verify_authority_files(contract)
    summary = {
        "contract_version": SUMMARY_VERSION,
        "merge_source": merge_source,
        "global_contract": {"path": str(contract_path), "sha256": contract_sha},
        **totals,
        "rank_count": machine_count,
        "shard_count": total_shards,
    }
    summary_path = output / "global-reference-summary.json"
    summary_sha = _write_immutable(summary_path, summary)
    return {
        "contract": {"path": str(contract_path), "sha256": contract_sha},
        "summary": {"path": str(summary_path), "sha256": summary_sha},
        **summary,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("allowlist", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("rank_dirs", nargs="+", type=Path)
    parser.add_argument("--generator-module", default=adapter.DEFAULT_GENERATOR_MODULE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    args = _parser().parse_args(values)
    print(
        json.dumps(
            merge(
                args.candidates,
                args.manifest,
                args.allowlist,
                args.output_dir,
                args.rank_dirs,
                generator_module=args.generator_module,
            ),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
