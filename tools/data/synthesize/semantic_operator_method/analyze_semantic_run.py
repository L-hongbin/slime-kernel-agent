#!/usr/bin/env python3
"""Exactly join static, KernelGym reference, and operator-liveness evidence."""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.synthesize import validate_train_mode_contract as reference_validator  # noqa: E402
from tools.data.synthesize.semantic_operator_method import generate_semantic_operator as generator  # noqa: E402
from tools.data.synthesize.semantic_operator_method import (  # noqa: E402
    validate_semantic_liveness as liveness_validator,
)

ANALYSIS_CONTRACT = "semantic_operator_exact_analysis_v1"
ACCEPTED_RUNTIME_STATUS = "kernelgym_reference_and_declared_operator_liveness_passed"
ACCEPTED_GOVERNANCE_STATUS = "semantic_operator_review_only_training_not_approved"
REFERENCE_CONTRACT = "kernelgym-reference-self-train-mode-v3"
EXPECTED_KERNELGYM_COMMIT = "26255057463a77b23abac0f3e5eafeeebf2ebbb5"
EXPECTED_KERNELGYM_HASHES = {
    "correctness_sha256": "a77f758a1ffb600cf8ada2290c54fe91b3cb691e81aef03f4fdb5a7e42a764b5",
    "loading_sha256": "8d0f6bb9f17802281997d764b349903799b3d9235627bc1c73c474f187862d06",
    "exec_types_sha256": "8c209627ec288679520dec4c1f8232512cd927051a17c11ec15d5768a56d9907",
    "profiling_sha256": "952e9d1618c1ef1803d7fae53171bcd7fa982e522b70d6355bb00f932ae6c29e",
    "config_init_sha256": "e027bcaa35e4cb55062f4ac0a5250393fa85976625b3c31c6051df84efb97bc7",
    "config_settings_sha256": "6bf33dde4269fcd54452de04191d8257069d3da396d5f2f31256a95596c7cf7e",
    "evaluator_bundle_sha256": "135f1758dcfd29ce8e8311a61adacc4a78e7c12433194b21749b37a1882a14ee",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(_canonical_json(value).encode())


def _nested(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"JSONL must contain objects:{path}")
    return rows


def _collect_shards(directory: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = sorted(directory.glob("shard-*-of-*.jsonl"))
    if not paths:
        raise ValueError(f"no shard JSONL found:{directory}")
    rows: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    for path in paths:
        shard_rows = _read_jsonl(path)
        rows.extend(shard_rows)
        bindings.append({"path": str(path.resolve()), "sha256": _sha256_file(path), "rows": len(shard_rows)})
    return rows, bindings


def _verify_static(
    candidates_path: Path, manifest_path: Path
) -> tuple[pa.Table, list[dict[str, Any]], dict[str, dict[str, Any]]]:
    table = pq.read_table(candidates_path)
    rows = table.to_pylist()
    manifests = _read_jsonl(manifest_path)
    if len(rows) != generator.EXACT_CANARY_ROWS or len(rows) != len(manifests):
        raise ValueError(f"static canary count must be exactly 1000:{len(rows)}:{len(manifests)}")
    generator_sha = _sha256_file(Path(generator.__file__).resolve())
    seen: dict[str, set[str]] = {
        key: set() for key in ("uuid", "reference_sha256", "normalized_ast_sha256", "row_payload_sha256")
    }
    family_counts: collections.Counter[str] = collections.Counter()
    template_counts: collections.Counter[str] = collections.Counter()
    by_uuid: dict[str, dict[str, Any]] = {}
    prefix_hashes: set[str] = set()
    template_bindings: set[str] = set()
    roots_bindings: set[str] = set()
    for index, (row, manifest) in enumerate(zip(rows, manifests, strict=True)):
        uuid = _nested(row, "extra_info.uuid")
        code = _nested(row, "reward_model.ground_truth")
        prompt = _nested(row, "prompt")
        if manifest.get("candidate_row_index") != index or manifest.get("uuid") != uuid:
            raise ValueError(f"static identity mismatch:{index}")
        if (
            manifest.get("manifest_contract_version") != generator.MANIFEST_VERSION
            or manifest.get("generator_contract_version") != generator.CONTRACT_VERSION
            or manifest.get("registry_version") != generator.REGISTRY_VERSION
            or manifest.get("static_contract_version") != generator.STATIC_CONTRACT_VERSION
            or manifest.get("runtime_contract_version") != generator.RUNTIME_CONTRACT_VERSION
        ):
            raise ValueError(f"static contract mismatch:{index}")
        if manifest.get("generator_source_sha256") != generator_sha:
            raise ValueError(f"generator source differs from static artifact:{index}")
        if (
            manifest.get("lineage_kind") != "standalone_semantic_synthetic"
            or manifest.get("parent_uuid") is not None
            or manifest.get("training_approved") is not False
            or manifest.get("structured_output_deferred") is not True
            or manifest.get("final_output_contract") != {"kind": "single_tensor", "finite_required": True}
        ):
            raise ValueError(f"governance/output/lineage mismatch:{index}")
        if not isinstance(uuid, str) or not isinstance(code, str) or not isinstance(prompt, list) or len(prompt) != 1:
            raise ValueError(f"candidate schema mismatch:{index}")
        if not prompt[0].get("content", "").endswith(code):
            raise ValueError(f"prompt does not end in exact reference:{index}")
        if _sha256_bytes(code.encode()) != manifest.get("reference_sha256"):
            raise ValueError(f"reference hash mismatch:{index}")
        if generator._normalized_ast_sha256(code) != manifest.get("normalized_ast_sha256"):
            raise ValueError(f"normalized AST hash mismatch:{index}")
        content = prompt[0].get("content", "")
        prefix = content[: -len(code)]
        if _sha256_bytes(content.encode()) != manifest.get("prompt_sha256") or _sha256_bytes(
            prefix.encode()
        ) != manifest.get("prompt_prefix_sha256"):
            raise ValueError(f"prompt/prefix hash mismatch:{index}")
        if _canonical_sha256(row) != manifest.get("row_payload_sha256"):
            raise ValueError(f"row payload hash mismatch:{index}")
        expected_code, expected_ops, expected_labels = generator._render(
            str(manifest.get("template_id")), int(manifest.get("template_variant"))
        )
        if (
            code != expected_code
            or manifest.get("declared_ops") != expected_ops
            or manifest.get("coverage_labels") != expected_labels
        ):
            raise ValueError(f"closed registry replay mismatch:{index}")
        if manifest.get("static_proof") != generator._static_contract(code, expected_ops):
            raise ValueError(f"static dependency replay mismatch:{index}")
        for field in seen:
            value = manifest.get(field)
            if not isinstance(value, str) or value in seen[field]:
                raise ValueError(f"static {field} not unique/valid:{index}")
            seen[field].add(value)
        family_counts[str(manifest["primary_family"])] += 1
        template_counts[str(manifest["template_id"])] += 1
        prefix_hashes.add(str(manifest.get("prompt_prefix_sha256")))
        template_bindings.add(_canonical_json(manifest.get("prompt_template")))
        roots_bindings.add(_canonical_json(manifest.get("decontamination_roots")))
        by_uuid[uuid] = {"row_index": index, "row": row, "manifest": manifest}
    if dict(family_counts) != generator.FAMILY_QUOTAS:
        raise ValueError(f"exclusive family quotas differ:{dict(family_counts)}")
    if dict(template_counts) != {item.template_id: item.variants for item in generator.TEMPLATES}:
        raise ValueError(f"template quotas differ:{dict(template_counts)}")
    if len(prefix_hashes) != 1 or len(template_bindings) != 1 or len(roots_bindings) != 1:
        raise ValueError("static provenance/prompt bindings are not lane-constant")
    template_binding = json.loads(next(iter(template_bindings)))
    if _sha256_file(Path(template_binding["path"])) != template_binding["sha256"]:
        raise ValueError("prompt template artifact changed")
    for root in json.loads(next(iter(roots_bindings))):
        if (
            _sha256_file(Path(root["path"])) != root["sha256"]
            or pq.ParquetFile(root["path"]).metadata.num_rows != root["rows"]
        ):
            raise ValueError(f"decontamination root changed:{root['path']}")
    return table, manifests, by_uuid


def _memory_guard_valid(value: Any) -> bool:
    return not reference_validator.validate_memory_guard_evidence(value)


def _verify_reference(
    records: Sequence[Mapping[str, Any]],
    by_uuid: Mapping[str, Mapping[str, Any]],
    candidates_path: Path,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, str]]:
    if len(records) != len(by_uuid):
        raise ValueError(f"reference result count must equal static rows:{len(records)}:{len(by_uuid)}")
    expected_source_sha = _sha256_file(candidates_path)
    expected_validator_sha = _sha256_file(Path(reference_validator.__file__).resolve())
    indexed: dict[str, Mapping[str, Any]] = {}
    failures: dict[str, str] = {}
    fingerprints: set[str] = set()
    launcher_hashes: set[str] = set()
    for record in records:
        uuid = record.get("uuid")
        if not isinstance(uuid, str) or uuid not in by_uuid or uuid in indexed:
            raise ValueError(f"reference UUID missing/duplicate/unknown:{uuid!r}")
        static = by_uuid[uuid]
        manifest = static["manifest"]
        if (
            record.get("row_index") != static["row_index"]
            or record.get("reference_sha256") != manifest["reference_sha256"]
        ):
            raise ValueError(f"reference row identity mismatch:{uuid}")
        if (
            record.get("contract_version") != REFERENCE_CONTRACT
            or record.get("source_sha256") != expected_source_sha
            or record.get("validator_source_sha256") != expected_validator_sha
            or record.get("trials") != 5
            or record.get("seed") != 42
            or record.get("training") is not True
            or record.get("max_device_memory_gib") != 64.0
        ):
            raise ValueError(f"reference policy/source binding mismatch:{uuid}")
        contract = record.get("contract_payload")
        if not isinstance(contract, Mapping) or contract.get("expected_mode_class") is not None:
            raise ValueError(f"reference contract payload mismatch:{uuid}")
        kernelgym = record.get("kernelgym")
        if not isinstance(kernelgym, Mapping) or kernelgym.get("git_commit") != EXPECTED_KERNELGYM_COMMIT:
            raise ValueError(f"KernelGym authority commit mismatch:{uuid}")
        if any(kernelgym.get(key) != value for key, value in EXPECTED_KERNELGYM_HASHES.items()):
            raise ValueError(f"KernelGym evaluator bundle mismatch:{uuid}")
        gpu = record.get("gpu")
        if not isinstance(gpu, Mapping) or not any(name in str(gpu.get("name")) for name in ("A800", "H20")):
            raise ValueError(f"reference GPU is not an authorized A800/H20:{uuid}")
        if not _memory_guard_valid(record.get("memory_guard")):
            raise ValueError(f"reference memory guard invalid:{uuid}")
        fingerprints.add(str(record.get("contract_fingerprint")))
        launcher_hashes.add(str(record.get("launcher_source_sha256")))
        indexed[uuid] = record
        if record.get("passed") is True:
            if (
                record.get("status") != "passed"
                or record.get("kernelgym_correctness") is not True
                or record.get("reference_forward_calls") != 5
                or record.get("identical_forward_calls") != 5
                or record.get("reference_training") is not True
                or record.get("identical_training") is not True
                or record.get("persistent_model_instances") is not True
            ):
                raise ValueError(f"reference passing evidence incomplete:{uuid}")
        else:
            failures[uuid] = str(record.get("status", "unknown_reference_failure"))
    if (
        len(fingerprints) != 1
        or len(launcher_hashes) != 1
        or not all(SHA256_RE.fullmatch(value) for value in fingerprints | launcher_hashes)
    ):
        raise ValueError("reference run mixes/omits contract or launcher bindings")
    return indexed, failures


def _verify_liveness(
    records: Sequence[Mapping[str, Any]],
    by_uuid: Mapping[str, Mapping[str, Any]],
    reference_passed: set[str],
    candidates_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, str]]:
    if len(records) != len(reference_passed):
        raise ValueError(f"liveness count must equal reference allowlist:{len(records)}:{len(reference_passed)}")
    expected_candidate_sha = _sha256_file(candidates_path)
    expected_manifest_sha = _sha256_file(manifest_path)
    expected_validator_sha = _sha256_file(Path(liveness_validator.__file__).resolve())
    expected_generator_sha = _sha256_file(Path(generator.__file__).resolve())
    indexed: dict[str, Mapping[str, Any]] = {}
    failures: dict[str, str] = {}
    bindings: set[str] = set()
    launcher_hashes: set[str] = set()
    for record in records:
        uuid = record.get("uuid")
        if not isinstance(uuid, str) or uuid not in reference_passed or uuid in indexed:
            raise ValueError(f"liveness UUID missing/duplicate/not-reference-passed:{uuid!r}")
        manifest = by_uuid[uuid]["manifest"]
        config = record.get("validation_config")
        if (
            record.get("contract_version") != liveness_validator.CONTRACT_VERSION
            or record.get("candidate_row_index") != manifest["candidate_row_index"]
            or record.get("template_id") != manifest["template_id"]
            or record.get("primary_family") != manifest["primary_family"]
            or record.get("mode_behavior") != manifest["mode_behavior"]
            or record.get("candidates_sha256") != expected_candidate_sha
            or record.get("manifest_sha256") != expected_manifest_sha
            or record.get("validator_source_sha256") != expected_validator_sha
            or record.get("generator_source_sha256") != expected_generator_sha
        ):
            raise ValueError(f"liveness source/identity mismatch:{uuid}")
        if (
            not isinstance(config, Mapping)
            or config.get("device") != "cuda:0"
            or config.get("trials") != 3
            or config.get("seed") != 17
            or config.get("max_device_memory_gib") != 64.0
            or config.get("persistent_train_mode_models") is not True
            or config.get("single_tensor_final_output") is not True
            or config.get("execution_controls") != liveness_validator.EXECUTION_CONTROLS
        ):
            raise ValueError(f"liveness policy mismatch:{uuid}")
        bindings.add(str(record.get("validation_binding_sha256")))
        launcher_hashes.add(str(record.get("launcher_source_sha256")))
        indexed[uuid] = record
        if record.get("passed") is True:
            if (
                record.get("status") != "passed"
                or record.get("persistent_model_instances") is not True
                or record.get("training_mode_preserved") is not True
                or record.get("final_output_kind") != "single_tensor"
                or record.get("execution_controls") != liveness_validator.EXECUTION_CONTROLS
                or not _memory_guard_valid(record.get("memory_guard"))
            ):
                raise ValueError(f"liveness passing summary incomplete:{uuid}")
            alias_evidence = record.get("registered_object_alias_evidence")
            if not isinstance(alias_evidence, Mapping) or set(alias_evidence) != {
                "module_attribute_paths",
                "tensor_attribute_paths",
            }:
                raise ValueError(f"registered object-alias evidence absent:{uuid}")
            for paths in alias_evidence.values():
                if (
                    not isinstance(paths, list)
                    or not all(isinstance(path, str) and path for path in paths)
                    or len(paths) != len(set(paths))
                ):
                    raise ValueError(f"registered object-alias evidence invalid:{uuid}")
            gpu = record.get("gpu")
            if not isinstance(gpu, Mapping) or not any(name in str(gpu.get("name")) for name in ("A800", "H20")):
                raise ValueError(f"liveness GPU is not authorized A800/H20:{uuid}")
            trials = record.get("trials")
            if not isinstance(trials, list) or len(trials) != 3:
                raise ValueError(f"liveness trial count mismatch:{uuid}")
            expected_ops = {
                item["op_id"]: {
                    "minimum_calls": item["min_calls_per_trial"],
                    "identities": {
                        f"{identity['schema']}.{identity['overload'] or '<default>'}"
                        for identity in item["runtime_identities"]
                    },
                }
                for item in manifest["declared_ops"]
            }
            for ordinal, trial in enumerate(trials):
                trace = trial.get("trace") if isinstance(trial, Mapping) else None
                if (
                    trial.get("ordinal") != ordinal
                    or trial.get("seed") != 17 + ordinal * 10_007
                    or trial.get("single_tensor_output") is not True
                    or trial.get("output_finite") is not True
                    or trial.get("control_trace_output_exact") is not True
                    or trial.get("control_trace_state_exact") is not True
                    or trial.get("inputs_immutable") is not True
                    or not isinstance(trace, Mapping)
                ):
                    raise ValueError(f"liveness trial contract incomplete:{uuid}:{ordinal}")
                per_op = trace.get("per_declared_op")
                if (
                    trace.get("final_output_declared_op_ids") != sorted(expected_ops)
                    or not SHA256_RE.fullmatch(str(trace.get("dispatch_sequence_sha256")))
                    or not isinstance(per_op, list)
                    or len(per_op) != len(expected_ops)
                    or {item.get("op_id") for item in per_op} != set(expected_ops)
                ):
                    raise ValueError(f"liveness declared-op set mismatch:{uuid}:{ordinal}")
                for item in per_op:
                    expected = expected_ops[item["op_id"]]
                    matches = item.get("matched_identities")
                    if (
                        item.get("minimum_calls") != expected["minimum_calls"]
                        or type(item.get("calls")) is not int
                        or item["calls"] < item["minimum_calls"]
                        or item.get("returned_output_witness") is not True
                        or not isinstance(matches, Mapping)
                        or not matches
                        or not set(matches).issubset(expected["identities"])
                        or not all(type(count) is int and count > 0 for count in matches.values())
                        or sum(matches.values()) != item["calls"]
                    ):
                        raise ValueError(f"liveness declared-op witness incomplete:{uuid}:{ordinal}:{item}")
        else:
            failures[uuid] = str(
                record.get("error") or record.get("reason", record.get("status", "unknown_liveness_failure"))
            )
    if (
        len(bindings) != 1
        or len(launcher_hashes) != 1
        or not all(SHA256_RE.fullmatch(value) for value in bindings | launcher_hashes)
    ):
        raise ValueError("liveness run mixes/omits binding or launcher hashes")
    return indexed, failures


def _failure_report(
    manifests: Sequence[Mapping[str, Any]],
    reference_passed: set[str],
    liveness_passed: set[str],
    reference_failures: Mapping[str, str],
    liveness_failures: Mapping[str, str],
) -> str:
    rows = []
    for item in manifests:
        uuid = str(item["uuid"])
        rows.append(
            {
                "family": str(item["primary_family"]),
                "template": str(item["template_id"]),
                "reference": uuid in reference_passed,
                "liveness": uuid in liveness_passed,
            }
        )
    lines = [
        "# Semantic/operator canary failure and bias audit",
        "",
        "Counts are row counts. Families are exclusive primary labels; declared ops are multi-label and are not counted as rows here.",
        "",
        "| family | generated | reference pass | liveness pass |",
        "| --- | ---: | ---: | ---: |",
    ]
    for family in sorted(generator.FAMILY_QUOTAS):
        selected = [row for row in rows if row["family"] == family]
        lines.append(
            f"| {family} | {len(selected)} | {sum(row['reference'] for row in selected)} | {sum(row['liveness'] for row in selected)} |"
        )
    lines.extend(["", "| template | generated | reference pass | liveness pass |", "| --- | ---: | ---: | ---: |"])
    for template in sorted(generator.TEMPLATE_BY_ID):
        selected = [row for row in rows if row["template"] == template]
        lines.append(
            f"| {template} | {len(selected)} | {sum(row['reference'] for row in selected)} | {sum(row['liveness'] for row in selected)} |"
        )
    lines.extend(["", "## Failure attribution", ""])
    reference_reasons = collections.Counter(reference_failures.values())
    liveness_reasons = collections.Counter(liveness_failures.values())
    lines.append(
        "Reference failures: "
        + (", ".join(f"`{key}`={value}" for key, value in reference_reasons.most_common()) or "none")
    )
    lines.append("")
    lines.append(
        "Operator-liveness failures: "
        + (", ".join(f"`{key}`={value}" for key, value in liveness_reasons.most_common()) or "none")
    )
    lines.append("")
    return "\n".join(lines)


def analyze(
    candidates_path: Path,
    manifest_path: Path,
    reference_dir: Path,
    liveness_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    candidates_path = candidates_path.resolve()
    manifest_path = manifest_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"analysis output directory is not empty:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    table, manifests, by_uuid = _verify_static(candidates_path, manifest_path)
    reference_records, reference_bindings = _collect_shards(reference_dir)
    references, reference_failures = _verify_reference(reference_records, by_uuid, candidates_path)
    reference_passed = {uuid for uuid, record in references.items() if record.get("passed") is True}
    liveness_records, liveness_bindings = _collect_shards(liveness_dir)
    liveness, liveness_failures = _verify_liveness(
        liveness_records, by_uuid, reference_passed, candidates_path, manifest_path
    )
    accepted = {uuid for uuid, record in liveness.items() if record.get("passed") is True}
    if not accepted.issubset(reference_passed):
        raise ValueError("liveness accepted set is not a reference-pass subset")

    accepted_indices = [index for index, item in enumerate(manifests) if item["uuid"] in accepted]
    accepted_table = table.take(pa.array(accepted_indices, type=pa.int64()))
    accepted_path = output_dir / "accepted.parquet"
    accepted_manifest_path = output_dir / "accepted.manifest.jsonl"
    failure_report_path = output_dir / "failure_bias_report.md"
    raw_hash_path = output_dir / "raw_artifact_sha256.json"
    pq.write_table(accepted_table, accepted_path, compression="zstd")
    accepted_manifests: list[dict[str, Any]] = []
    for item in manifests:
        uuid = str(item["uuid"])
        if uuid not in accepted:
            continue
        updated = copy.deepcopy(item)
        updated.update(
            {
                "reference_runtime_status": "passed",
                "operator_liveness_status": "passed",
                "runtime_status": ACCEPTED_RUNTIME_STATUS,
                "governance_status": ACCEPTED_GOVERNANCE_STATUS,
                "training_approved": False,
                "runtime_evidence": {
                    "reference_contract_fingerprint": references[uuid]["contract_fingerprint"],
                    "reference_record_sha256": _canonical_sha256(references[uuid]),
                    "liveness_binding_sha256": liveness[uuid]["validation_binding_sha256"],
                    "liveness_record_sha256": _canonical_sha256(liveness[uuid]),
                    "execution_controls": liveness[uuid]["execution_controls"],
                },
            }
        )
        accepted_manifests.append(updated)
    with accepted_manifest_path.open("w", encoding="utf-8") as handle:
        for item in accepted_manifests:
            handle.write(_canonical_json(item) + "\n")
    failure_report_path.write_text(
        _failure_report(manifests, reference_passed, accepted, reference_failures, liveness_failures),
        encoding="utf-8",
    )
    raw_hashes = {
        "candidates": {"path": str(candidates_path), "sha256": _sha256_file(candidates_path)},
        "manifest": {"path": str(manifest_path), "sha256": _sha256_file(manifest_path)},
        "reference_shards": reference_bindings,
        "liveness_shards": liveness_bindings,
    }
    raw_hash_path.write_text(json.dumps(raw_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    family_generated = collections.Counter(str(item["primary_family"]) for item in manifests)
    family_reference = collections.Counter(
        str(by_uuid[uuid]["manifest"]["primary_family"]) for uuid in reference_passed
    )
    family_accepted = collections.Counter(str(by_uuid[uuid]["manifest"]["primary_family"]) for uuid in accepted)
    summary = {
        "analysis_contract": ANALYSIS_CONTRACT,
        "generated": len(manifests),
        "reference_passed": len(reference_passed),
        "reference_failed": len(manifests) - len(reference_passed),
        "liveness_executed": len(liveness),
        "liveness_passed": len(accepted),
        "liveness_failed": len(liveness) - len(accepted),
        "accepted": len(accepted),
        "family_counts": {
            family: {
                "generated": family_generated[family],
                "reference_passed": family_reference[family],
                "liveness_passed": family_accepted[family],
            }
            for family in sorted(generator.FAMILY_QUOTAS)
        },
        "final_output_kind": "single_tensor",
        "structured_output_status": "explicitly_deferred",
        "execution_controls": liveness_validator.EXECUTION_CONTROLS,
        "runtime_status": ACCEPTED_RUNTIME_STATUS,
        "governance_status": ACCEPTED_GOVERNANCE_STATUS,
        "training_approved": False,
        "artifacts": {
            "accepted": {"path": str(accepted_path), "sha256": _sha256_file(accepted_path)},
            "accepted_manifest": {"path": str(accepted_manifest_path), "sha256": _sha256_file(accepted_manifest_path)},
            "failure_bias_report": {"path": str(failure_report_path), "sha256": _sha256_file(failure_report_path)},
            "raw_artifact_hashes": {"path": str(raw_hash_path), "sha256": _sha256_file(raw_hash_path)},
        },
    }
    summary_path = output_dir / "final_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidates", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("reference_dir", type=Path)
    parser.add_argument("liveness_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    analyze(args.candidates, args.manifest, args.reference_dir, args.liveness_dir, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
