#!/usr/bin/env python3
"""Merge copied rank-local 10k liveness evidence into an immutable authority view.

Every H20 host has a local `/nfs/FM`; this program therefore never assumes a
shared run directory.  It accepts only complete, explicitly copied rank
directories and rechecks all row ownership and source bindings before writing
the two authority artifacts.
"""

from __future__ import annotations

import argparse
import fcntl
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

from tools.data.cleaning import runtime_validation as runtime_validation_module  # noqa: E402
from tools.data.synthesize import validate_train_mode_contract as train_mode_contract_module  # noqa: E402
from tools.data.synthesize.canary.operator_structure_10k_method import (  # noqa: E402
    validate_operator_structure_10k_liveness as adapter,
)
from tools.data.synthesize.semantic_operator_method import (  # noqa: E402
    generate_semantic_operator as semantic_operator_generator,
)
from tools.data.synthesize.semantic_operator_method import validate_semantic_liveness as runtime_core  # noqa: E402

CONTRACT_VERSION = "operator_structure_10k_global_liveness_evidence_v1"
SUMMARY_VERSION = "operator_structure_10k_global_liveness_summary_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}:{path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object:{path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid shard JSONL:{path}") from exc
    if not all(isinstance(value, dict) for value in values):
        raise ValueError(f"shard JSONL contains non-objects:{path}")
    return values


def _source_pair(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"} or not isinstance(value.get("path"), str):
        raise ValueError(f"invalid source pair:{label}")
    adapter._require_sha256(value.get("sha256"), label=label)
    return {"path": str(value["path"]), "sha256": str(value["sha256"])}


def _scheduler_sources(document: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    source = document.get("source_binding")
    names = {
        "adapter_source",
        "shared_runtime_core_source",
        "runtime_validation_source",
        "train_mode_contract_source",
        "semantic_operator_generator_source",
        "generator_source",
        "launcher_source",
    }
    if not isinstance(source, Mapping) or set(source) != names:
        raise ValueError("rank scheduler source-binding keys mismatch")
    return {name: _source_pair(source[name], name) for name in names}


def _adapter_sources(value: Any, expected: Mapping[str, Mapping[str, str]]) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError("adapter source-binding keys mismatch")
    for name, pair in expected.items():
        actual = value.get(name)
        fields = {"path", "sha256", "contract_version"} if name == "shared_runtime_core_source" else {"path", "sha256"}
        if (
            not isinstance(actual, Mapping)
            or set(actual) != fields
            or actual.get("path") != pair["path"]
            or actual.get("sha256") != pair["sha256"]
        ):
            raise ValueError(f"adapter source-binding mismatch:{name}")
        if name == "shared_runtime_core_source" and actual.get("contract_version") != runtime_core.CONTRACT_VERSION:
            raise ValueError("adapter runtime-core contract mismatch")


def _install_immutable(path: Path, value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if not path.is_file() or path.read_text(encoding="utf-8") != encoded:
            raise ValueError(f"authority artifact already exists with different content:{path}")
    else:
        temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, path)
    return _sha256(path)


def _install_authority_file(source: Path, destination: Path) -> dict[str, str]:
    """Install a staged artifact under the authority with a strict file pair."""

    source_sha = _sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or _sha256(destination) != source_sha:
            raise ValueError(f"authority artifact collision:{destination}")
    else:
        try:
            os.link(source, destination)
        except OSError:
            temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
            shutil.copy2(source, temporary)
            if _sha256(temporary) != source_sha:
                temporary.unlink(missing_ok=True)
                raise ValueError(f"authority artifact copy hash mismatch:{source}") from None
            os.replace(temporary, destination)
    return {"path": str(destination), "sha256": source_sha}


def _ordered_allowlist(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError("UUID allowlist must be non-empty and unique")
    return values


def _verify_authority_files(contract: Mapping[str, Any]) -> None:
    for shard in contract.get("shards", []):
        if not isinstance(shard, Mapping):
            raise ValueError("invalid authority shard")
        pair = {"path": shard.get("path"), "sha256": shard.get("sha256")}
        if (
            set(pair.values()) == {None}
            or not isinstance(pair["path"], str)
            or not isinstance(pair["sha256"], str)
            or not Path(pair["path"]).is_file()
            or _sha256(Path(pair["path"])) != pair["sha256"]
        ):
            raise ValueError("authority shard is missing or changed")
    for rank in contract.get("rank_evidence", []):
        if not isinstance(rank, Mapping):
            raise ValueError("invalid authority rank evidence")
        for field in ("scheduler_contract", "launcher_summary"):
            pair = rank.get(field)
            if (
                not isinstance(pair, Mapping)
                or set(pair) != {"path", "sha256"}
                or not isinstance(pair.get("path"), str)
                or not Path(pair["path"]).is_file()
                or _sha256(Path(pair["path"])) != pair.get("sha256")
            ):
                raise ValueError(f"authority {field} is missing or changed")
        archives = rank.get("source_archives")
        if not isinstance(archives, Mapping):
            raise ValueError("authority source archives missing")
        for pair in archives.values():
            if (
                not isinstance(pair, Mapping)
                or set(pair) != {"path", "sha256"}
                or not isinstance(pair.get("path"), str)
                or not Path(pair["path"]).is_file()
                or _sha256(Path(pair["path"])) != pair.get("sha256")
            ):
                raise ValueError("authority source archive is missing or changed")


def merge(
    candidates: Path,
    manifest: Path,
    allowlist: Path,
    output_dir: Path,
    rank_dirs: Sequence[Path],
    *,
    generator_module: str,
) -> dict[str, Any]:
    machine_count, gpus_per_machine, scope = 1, 8, "partial"
    if len(rank_dirs) != machine_count:
        raise ValueError("canary liveness merge requires one rank directory")
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / ".merge.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another authority merge owns:{output_dir}") from exc
        tasks = adapter.tasks(candidates, manifest, generator_module=generator_module)
        task_by_uuid = {str(task["uuid"]): task for task in tasks}
        generator = importlib.import_module(generator_module)
        expected_rows = int(getattr(generator, "EXACT_CANDIDATE_ROWS", -1))
        if (
            expected_rows != 13_000
            or len(tasks) != expected_rows
            or [task["candidate_row_index"] for task in tasks] != list(range(expected_rows))
        ):
            raise ValueError("liveness must replay the complete 13,000-row original-index registry")
        ordered_allowlist = _ordered_allowlist(allowlist)
        allowed = set(ordered_allowlist)
        if not allowed.issubset(task_by_uuid):
            raise ValueError("allowlist contains UUID absent from static replay")
        allowlist_indices = [int(task_by_uuid[uuid]["candidate_row_index"]) for uuid in ordered_allowlist]
        if allowlist_indices != sorted(allowlist_indices):
            raise ValueError("reference-passed UUID allowlist must be in original candidate-row-index order")
        if len(ordered_allowlist) >= expected_rows:
            raise ValueError("canary liveness allowlist must be a strict original-index subset")
        total_shards = machine_count * gpus_per_machine
        candidates_pair = {"path": str(candidates.resolve()), "sha256": _sha256(candidates)}
        manifest_pair = {"path": str(manifest.resolve()), "sha256": _sha256(manifest)}
        allowlist_pair = {"path": str(allowlist.resolve()), "sha256": _sha256(allowlist)}
        expected_current = {
            "adapter_source": {
                "path": str(Path(adapter.__file__).resolve()),
                "sha256": _sha256(Path(adapter.__file__).resolve()),
            },
            "shared_runtime_core_source": {
                "path": str(Path(runtime_core.__file__).resolve()),
                "sha256": _sha256(Path(runtime_core.__file__).resolve()),
            },
            "runtime_validation_source": {
                "path": str(Path(runtime_validation_module.__file__).resolve()),
                "sha256": _sha256(Path(runtime_validation_module.__file__).resolve()),
            },
            "train_mode_contract_source": {
                "path": str(Path(train_mode_contract_module.__file__).resolve()),
                "sha256": _sha256(Path(train_mode_contract_module.__file__).resolve()),
            },
            "semantic_operator_generator_source": {
                "path": str(Path(semantic_operator_generator.__file__).resolve()),
                "sha256": _sha256(Path(semantic_operator_generator.__file__).resolve()),
            },
            "generator_source": {
                "path": str(Path(generator.__file__).resolve()),
                "sha256": _sha256(Path(generator.__file__).resolve()),
            },
            "launcher_source": {
                "path": str(
                    (
                        _REPO_ROOT
                        / "tools/data/synthesize/canary/operator_structure_10k_method/launch_operator_structure_10k_liveness_rank.sh"
                    ).resolve()
                ),
                "sha256": _sha256(
                    _REPO_ROOT
                    / "tools/data/synthesize/canary/operator_structure_10k_method/launch_operator_structure_10k_liveness_rank.sh"
                ),
            },
        }
        common_sources: dict[str, dict[str, str]] | None = None
        common_reference_binding: dict[str, str] | None = None
        all_records: dict[str, Mapping[str, Any]] = {}
        rank_evidence, shard_evidence = [], []
        aggregate = {"selected": 0, "executed": 0, "resumed": 0, "passed": 0, "failed": 0}
        for rank, raw_dir in enumerate(rank_dirs):
            directory = raw_dir.resolve()
            scheduler_path = directory / f"scheduler-contract-rank-{rank}.json"
            summary_path = directory / "launcher-summary.json"
            scheduler = _read_object(scheduler_path, "rank scheduler")
            summary = _read_object(summary_path, "rank summary")
            if (
                scheduler.get("contract_version") != "operator_structure_10k_liveness_rank_scheduler_v1"
                or scheduler.get("machine_rank") != rank
                or scheduler.get("machine_count") != machine_count
                or scheduler.get("gpus_per_machine") != gpus_per_machine
                or scheduler.get("global_shard_count") != total_shards
                or scheduler.get("global_shard_indices")
                != list(range(rank * gpus_per_machine, (rank + 1) * gpus_per_machine))
                or scheduler.get("candidates") != candidates_pair
                or scheduler.get("manifest") != manifest_pair
                or scheduler.get("allowlist") != allowlist_pair
                or scheduler.get("generator_module") != generator_module
                or scheduler.get("trials") != 3
                or scheduler.get("seed") != 17
                or scheduler.get("max_device_memory_gib") != 64.0
                or scheduler.get("persistent_train_mode_models") is not True
                or scheduler.get("single_tensor_final_output") is not True
                or scheduler.get("execution_controls") != adapter.EXECUTION_CONTROLS
            ):
                raise ValueError(f"rank scheduler policy/identity mismatch:{rank}")
            host = scheduler.get("execution_host")
            inventory = scheduler.get("gpu_inventory")
            if (
                not isinstance(host, str)
                or not host
                or not isinstance(inventory, list)
                or len(inventory) != gpus_per_machine
                or not all("H20" in str(x) for x in inventory)
            ):
                raise ValueError(f"rank execution-host/GPU evidence invalid:{rank}")
            sources = _scheduler_sources(scheduler)
            for name, expected in expected_current.items():
                if sources[name] != expected:
                    raise ValueError(f"authority source drift:{rank}:{name}")
            if common_sources is None:
                common_sources = sources
            elif sources != common_sources:
                raise ValueError(f"rank source bindings differ:{rank}")
            reference_binding = _source_pair(
                scheduler.get("reference_passed_binding"), f"reference-passed-binding:{rank}"
            )
            if (
                not Path(reference_binding["path"]).is_file()
                or _sha256(Path(reference_binding["path"])) != reference_binding["sha256"]
            ):
                raise ValueError(f"rank reference-passed binding artifact mismatch:{rank}")
            if common_reference_binding is None:
                common_reference_binding = reference_binding
            elif reference_binding != common_reference_binding:
                raise ValueError(f"rank reference-passed binding differs:{rank}")
            archives = {
                "launcher_source": directory / "launcher_source.sh",
                "adapter_source": directory / "adapter_source.py",
                "generator_source": directory / "generator_source.py",
                "shared_runtime_core_source": directory / "shared_runtime_core_source.py",
                "runtime_validation_source": directory / "runtime_validation_source.py",
                "train_mode_contract_source": directory / "train_mode_contract_source.py",
                "semantic_operator_generator_source": directory / "semantic_operator_generator_source.py",
            }
            for name, path in archives.items():
                if not path.is_file() or _sha256(path) != sources[name]["sha256"]:
                    raise ValueError(f"rank source archive mismatch:{rank}:{name}")
            if summary.get("contract_version") != "operator_structure_10k_liveness_rank_summary_v1" or any(
                summary.get(k) != scheduler.get(k)
                for k in ("machine_rank", "machine_count", "gpus_per_machine", "global_shard_count", "execution_host")
            ):
                raise ValueError(f"rank launcher summary identity mismatch:{rank}")
            _adapter_sources(summary.get("source_binding"), sources)
            binding_by_shard = summary.get("validation_binding_by_shard")
            shard_summaries = summary.get("shards")
            if not isinstance(shard_summaries, list) or len(shard_summaries) != gpus_per_machine:
                raise ValueError(f"rank shard summary count mismatch:{rank}")
            by_shard_summary = {
                item.get("global_shard_index"): item for item in shard_summaries if isinstance(item, Mapping)
            }
            expected_shards = set(range(rank * gpus_per_machine, (rank + 1) * gpus_per_machine))
            if set(by_shard_summary) != expected_shards:
                raise ValueError(f"rank shard summary ownership mismatch:{rank}")
            if (
                not isinstance(binding_by_shard, Mapping)
                or set(binding_by_shard) != {str(shard) for shard in expected_shards}
                or any(not isinstance(value, str) or len(value) != 64 for value in binding_by_shard.values())
            ):
                raise ValueError(f"rank per-shard validation-binding schema mismatch:{rank}")
            rank_totals = {key: 0 for key in aggregate}
            for shard in sorted(expected_shards):
                path = directory / f"shard-{shard:02d}-of-{total_shards:02d}.jsonl"
                records = _read_jsonl(path)
                expected_uuids = {
                    uuid for uuid in allowed if int(task_by_uuid[uuid]["candidate_row_index"]) % total_shards == shard
                }
                observed_uuids = {str(record.get("uuid")) for record in records}
                if not expected_uuids or len(observed_uuids) != len(records) or observed_uuids != expected_uuids:
                    raise ValueError(f"global shard UUID partition mismatch:{rank}:{shard}")
                for record in records:
                    uuid = str(record["uuid"])
                    evidence = record.get("binding_evidence")
                    if uuid in all_records or not isinstance(evidence, Mapping):
                        raise ValueError(f"duplicate/missing liveness evidence:{uuid}")
                    if (
                        record.get("contract_version") != adapter.CONTRACT_VERSION
                        or record.get("candidate_row_index") != task_by_uuid[uuid]["candidate_row_index"]
                    ):
                        raise ValueError(f"record identity mismatch:{uuid}")
                    # Binding digest uses compact canonical JSON, unlike stored JSON formatting.
                    if (
                        record.get("validation_binding_sha256")
                        != hashlib.sha256(_canonical(evidence).encode()).hexdigest()
                    ):
                        raise ValueError(f"record binding digest mismatch:{uuid}")
                    if record.get("validation_binding_sha256") != binding_by_shard[str(shard)]:
                        raise ValueError(f"rank/shard validation binding mismatch:{uuid}")
                    if (
                        evidence.get("execution_host") != host
                        or evidence.get("global_shard_index") != shard
                        or evidence.get("global_shard_count") != total_shards
                        or evidence.get("total_rows") != len(tasks)
                        or evidence.get("generator_module") != generator_module
                        or evidence.get("candidates_path") != candidates_pair["path"]
                        or evidence.get("candidates_sha256") != candidates_pair["sha256"]
                        or evidence.get("manifest_path") != manifest_pair["path"]
                        or evidence.get("manifest_sha256") != manifest_pair["sha256"]
                        or evidence.get("allowlist_path") != allowlist_pair["path"]
                        or evidence.get("allowlist_sha256") != allowlist_pair["sha256"]
                        or evidence.get("reference_passed_binding") != reference_binding
                        or evidence.get("validation_config")
                        != {
                            "device": "cuda:0",
                            "trials": 3,
                            "seed": 17,
                            "timeout_seconds": scheduler.get("timeout_seconds"),
                            "max_device_memory_gib": 64.0,
                            "persistent_train_mode_models": True,
                            "single_tensor_final_output": True,
                            "execution_controls": adapter.EXECUTION_CONTROLS,
                        }
                    ):
                        raise ValueError(f"record binding policy mismatch:{uuid}")
                    _adapter_sources({name: evidence.get(name) for name in sources}, sources)
                    task = task_by_uuid[uuid]
                    if not adapter._record_binding_ok(
                        record, evidence, sources, binding_by_shard[str(shard)]
                    ) or not adapter._record_schema_ok(record, task):
                        raise ValueError(f"record adapter binding/schema mismatch:{uuid}")
                    driver_gpu = record.get("driver_gpu")
                    if not adapter._driver_h20_gpu_evidence_ok(driver_gpu) or driver_gpu.get(
                        "cuda_visible_devices"
                    ) != str(shard % gpus_per_machine):
                        raise ValueError(f"record driver H20 evidence mismatch:{uuid}")
                    all_records[uuid] = record
                logged = by_shard_summary[shard]
                counts = {key: logged.get(key) for key in rank_totals}
                if (
                    any(type(value) is not int or value < 0 for value in counts.values())
                    or counts["selected"] != len(records)
                    or counts["selected"] != counts["executed"] + counts["resumed"]
                    or counts["passed"] != sum(record.get("passed") is True for record in records)
                    or counts["failed"] != sum(record.get("passed") is not True for record in records)
                    or counts["selected"] != counts["passed"] + counts["failed"]
                    or logged.get("global_shard_count") != total_shards
                    or logged.get("validation_binding_sha256") != binding_by_shard[str(shard)]
                ):
                    raise ValueError(f"rank shard summary accounting mismatch:{rank}:{shard}")
                for key, value in counts.items():
                    rank_totals[key] += value
                authority_path = output_dir / path.name
                materialized = _install_authority_file(path, authority_path)
                shard_evidence.append(
                    {"rank": rank, "global_shard_index": shard, **materialized, "rows": len(records)}
                )
            if any(summary.get(key) != value for key, value in rank_totals.items()):
                raise ValueError(f"rank aggregate accounting mismatch:{rank}")
            for key, value in rank_totals.items():
                aggregate[key] += value
            authority_rank = output_dir / "rank-evidence" / f"rank-{rank:02d}"
            rank_evidence.append(
                {
                    "machine_rank": rank,
                    "execution_host": host,
                    "gpu_inventory": inventory,
                    "scheduler_contract": _install_authority_file(
                        scheduler_path, authority_rank / scheduler_path.name
                    ),
                    "launcher_summary": _install_authority_file(summary_path, authority_rank / summary_path.name),
                    "source_archives": {
                        name: _install_authority_file(path, authority_rank / path.name)
                        for name, path in archives.items()
                    },
                }
            )
        if (
            set(all_records) != allowed
            or aggregate["selected"] != len(allowed)
            or aggregate["selected"] != aggregate["executed"] + aggregate["resumed"]
            or aggregate["selected"] != aggregate["passed"] + aggregate["failed"]
        ):
            raise ValueError("global aggregate accounting/UUID coverage mismatch")
        if common_sources is None or common_reference_binding is None:
            raise ValueError("no rank source/reference-passed binding")
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
            "candidates": candidates_pair,
            "manifest": manifest_pair,
            "allowlist": allowlist_pair,
            "reference_passed_uuids": allowlist_pair,
            "reference_passed_binding": common_reference_binding,
            "source_binding": common_sources,
            "rank_evidence": rank_evidence,
            "shards": shard_evidence,
        }
        contract_path = output_dir / "global-liveness-contract.json"
        contract_sha = _install_immutable(contract_path, contract)
        _verify_authority_files(contract)
        summary = {
            "contract_version": SUMMARY_VERSION,
            "selection_scope": scope,
            "merge_source": merge_source,
            "global_contract": {"path": str(contract_path), "sha256": contract_sha},
            "selected": aggregate["selected"],
            "executed": aggregate["executed"],
            "resumed": aggregate["resumed"],
            "passed": aggregate["passed"],
            "failed": aggregate["failed"],
            "rank_count": machine_count,
            "shard_count": total_shards,
        }
        summary_path = output_dir / "global-liveness-summary.json"
        summary_sha = _install_immutable(summary_path, summary)
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
