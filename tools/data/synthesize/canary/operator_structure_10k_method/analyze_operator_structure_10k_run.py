#!/usr/bin/env python3
"""Audit the fixed operator/structure canary and its H20 evidence."""

from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import importlib
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.cleaning import runtime_validation as runtime_validation_module  # noqa: E402
from tools.data.synthesize import profile_prompt_tvm_distribution as distribution_profiler  # noqa: E402
from tools.data.synthesize import validate_train_mode_contract as train_mode_contract_module  # noqa: E402
from tools.data.synthesize.canary.operator_structure_10k_method import (  # noqa: E402
    validate_operator_structure_10k_liveness as liveness_adapter,
)
from tools.data.synthesize.canary.operator_structure_10k_method import (  # noqa: E402
    validate_operator_structure_10k_reference as reference_adapter,
)
from tools.data.synthesize.semantic_operator_method import (  # noqa: E402
    generate_semantic_operator as semantic_operator_generator,
)
from tools.data.synthesize.semantic_operator_method import validate_semantic_liveness as runtime_core  # noqa: E402

STATIC_ROWS = 13_000
FINAL_ROWS = 10_000
FAMILY_COUNT = 5
LIVENESS_SHARDS = 8
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
FINAL_OUTPUTS = {"single_tensor", "single_dense_tensor"}
REFERENCE_CONTRACT = "kernelgym-reference-self-train-mode-v3"
REFERENCE_AUTHORITY_CONTRACT = "operator_structure_10k_global_reference_evidence_v1"
REFERENCE_AUTHORITY_SUMMARY = "operator_structure_10k_global_reference_summary_v1"
REFERENCE_SOURCES = (
    "reference_adapter_source",
    "generic_reference_core_source",
    "generator_source",
    "launcher_source",
)
REFERENCE_PASSED_BINDING_CONTRACT = "operator_structure_10k_reference_passed_binding_v1"
KERNELGYM_HASH_FIELDS = (
    "config_init_sha256",
    "config_settings_sha256",
    "correctness_sha256",
    "evaluator_bundle_sha256",
    "exec_types_sha256",
    "loading_sha256",
    "profiling_sha256",
)

REWORK_THRESHOLDS = {
    "input_semantic_skeleton_unique_min": 1_000,
    "model_semantic_skeleton_max_reuse": 96,
    "model_input_semantic_skeleton_max_reuse": 32,
    "axis_ratio_max_without_annotation": 1_000.0,
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_ast_sha256(code: str) -> str:
    normalized = ast.dump(ast.parse(code), include_attributes=False)
    return _sha256_bytes(normalized.encode())


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_new_atomically(path: Path, value: str) -> None:
    """Create an explicit derived artifact without overwriting prior evidence."""

    if path.exists():
        raise FileExistsError(f"derived evidence output already exists:{path}")
    if not path.parent.is_dir():
        raise ValueError(f"derived evidence parent must already exist:{path.parent}")
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values or not all(isinstance(value, dict) for value in values):
        raise ValueError(f"JSONL must contain nonempty objects:{path}")
    return values


def _collect_shards(
    directory: Path, shard_count: int | None = None
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = sorted(directory.glob("shard-*-of-*.jsonl"))
    if not paths:
        raise ValueError(f"no shard JSONL found:{directory}")
    if shard_count is not None:
        expected = {f"shard-{index:02d}-of-{shard_count:02d}.jsonl" for index in range(shard_count)}
        actual = {path.name for path in paths}
        if actual != expected:
            raise ValueError(f"unexpected shard set:{sorted(actual)}")
    records: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    for path in paths:
        rows = _read_jsonl(path)
        records.extend(rows)
        artifacts.append({"path": str(path.resolve()), "sha256": _sha256_file(path), "rows": len(rows)})
    return records, artifacts


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return ""


def _row_uuid(row: Mapping[str, Any]) -> str:
    value = row.get("extra_info")
    uuid = value.get("uuid") if isinstance(value, Mapping) else None
    if not isinstance(uuid, str) or not uuid:
        raise ValueError("candidate row lacks extra_info.uuid")
    return uuid


def _row_code(row: Mapping[str, Any]) -> str:
    value = row.get("reward_model")
    code = value.get("ground_truth") if isinstance(value, Mapping) else None
    if not isinstance(code, str) or not code.strip():
        raise ValueError("candidate row lacks reward_model.ground_truth")
    return code


def _load_generator(name: str) -> Any:
    module = importlib.import_module(name)
    required = (
        "TEMPLATES",
        "FAMILY_CANDIDATE_QUOTAS",
        "FAMILY_FINAL_QUOTAS",
        "EXACT_CANDIDATE_ROWS",
        "replay_manifest",
    )
    missing = [field for field in required if not hasattr(module, field)]
    if missing:
        raise ValueError(f"generator does not expose required 10k interface:{missing}")
    if not callable(module.replay_manifest):
        raise ValueError("generator replay_manifest is not callable")
    return module


def _template_recipe_map(generator: Any) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in generator.TEMPLATES:
        template = getattr(item, "template_id", None)
        recipe = getattr(item, "kind", None)
        if not isinstance(template, str) or not template or not isinstance(recipe, str) or not recipe:
            raise ValueError("generator templates require template_id and kind")
        if template in mapping:
            raise ValueError(f"duplicate template:{template}")
        mapping[template] = recipe
    return mapping


def _generator_contract(generator: Any) -> dict[str, Any]:
    generated = dict(generator.FAMILY_CANDIDATE_QUOTAS)
    selected = dict(generator.FAMILY_FINAL_QUOTAS)
    if len(generated) != FAMILY_COUNT or len(selected) != FAMILY_COUNT or set(generated) != set(selected):
        raise ValueError("generator must expose the same five generated/final families")
    if sum(generated.values()) != STATIC_ROWS or sum(selected.values()) != FINAL_ROWS:
        raise ValueError(f"generator quota totals must be {STATIC_ROWS}/{FINAL_ROWS}")
    if any(type(value) is not int or value <= 0 for value in [*generated.values(), *selected.values()]):
        raise ValueError("family quotas must be positive integers")
    templates = _template_recipe_map(generator)
    # The registry deliberately owns only its candidate/final family quotas.
    # Finalizer minima are derived, so the same frozen rules are visible here
    # instead of being optional, stale generator constants.
    template_min = {template: 80 for template in templates}
    recipe_min: dict[str, int] = collections.Counter()
    for recipe in templates.values():
        recipe_min[recipe] += 80
    exact_rows = generator.EXACT_CANDIDATE_ROWS
    if exact_rows != STATIC_ROWS or getattr(generator, "MAX_AUTHORIZED_CANDIDATES", exact_rows) != STATIC_ROWS:
        raise ValueError("generator exact candidate count must equal 13k")
    return {
        "generated_family_quotas": generated,
        "final_family_quotas": selected,
        "template_recipe": templates,
        "template_minimums": template_min,
        "recipe_minimums": recipe_min,
    }


def _replay(
    generator: Any, manifest: Mapping[str, Any]
) -> tuple[str, Any, Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    result = generator.replay_manifest(manifest)
    if not isinstance(result, Mapping):
        raise ValueError(f"invalid replay result:{manifest.get('uuid')}")
    code, ops, labels, proof, spec, input_contract = (
        result.get("code"),
        result.get("declared_ops"),
        result.get("coverage_labels"),
        result.get("static_proof"),
        result.get("spec"),
        result.get("input_semantic_contract"),
    )
    if (
        not isinstance(code, str)
        or not isinstance(ops, list)
        or not all(isinstance(value, Mapping) for value in (labels, proof, spec, input_contract))
    ):
        raise ValueError(f"invalid replay schema:{manifest.get('uuid')}")
    return code, ops, labels, proof, spec, input_contract


def _verify_decontamination_roots(bindings: Any, generator: Any) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    if not isinstance(bindings, list) or not bindings:
        raise ValueError("decontamination roots missing")
    expected_historical = Path(getattr(generator, "_HISTORICAL_1K_ROOT", "")).resolve()
    observed_paths: set[Path] = set()
    references, normalized, verified = set(), set(), []
    for item in bindings:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"path", "sha256", "rows"}
            or not isinstance(item.get("path"), str)
            or type(item.get("rows")) is not int
            or item["rows"] <= 0
        ):
            raise ValueError("decontamination root schema is invalid")
        path = Path(item["path"]).resolve()
        if (
            not path.is_file()
            or _sha256_file(path) != item["sha256"]
            or pq.ParquetFile(path).metadata.num_rows != item["rows"]
        ):
            raise ValueError(f"decontamination root drift:{path}")
        observed_paths.add(path)
        for batch in pq.ParquetFile(path).iter_batches(columns=["reward_model"], batch_size=512):
            for reward in batch.column(0).to_pylist():
                code = reward.get("ground_truth") if isinstance(reward, Mapping) else None
                if not isinstance(code, str) or not code:
                    raise ValueError(f"decontamination root has invalid reference:{path}")
                references.add(_sha256_bytes(code.encode()))
                normalized.add(_normalized_ast_sha256(code))
        verified.append({"path": str(path), "sha256": item["sha256"], "rows": item["rows"]})
    if expected_historical not in observed_paths:
        raise ValueError("decontamination roots omit the historical 1k accepted set")
    return references, normalized, verified


def _verify_candidate_hashes(row: Mapping[str, Any], manifest: Mapping[str, Any], code: str, uuid: str) -> None:
    prompt = row.get("prompt")
    if (
        not isinstance(prompt, list)
        or len(prompt) != 1
        or not isinstance(prompt[0], Mapping)
        or not isinstance(prompt[0].get("content"), str)
        or not prompt[0]["content"].endswith(code)
    ):
        raise ValueError(f"candidate prompt does not bind exact reference:{uuid}")
    if manifest.get("prompt_sha256") != _sha256_bytes(prompt[0]["content"].encode()) or manifest.get(
        "row_payload_sha256"
    ) != _sha256_bytes(_canonical_json(row).encode()):
        raise ValueError(f"candidate prompt/payload hash mismatch:{uuid}")
    if manifest.get("reference_sha256") != _sha256_bytes(code.encode()) or manifest.get(
        "normalized_ast_sha256"
    ) != _normalized_ast_sha256(code):
        raise ValueError(f"candidate reference/normalized AST hash mismatch:{uuid}")


def _static_index(
    candidates: Path, manifest_path: Path, generator: Any
) -> tuple[pa.Table, list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    contract, generator_sha = _generator_contract(generator), _sha256_file(Path(generator.__file__).resolve())
    table = pq.read_table(candidates)
    rows, manifests = table.to_pylist(), _read_jsonl(manifest_path)
    if len(rows) != STATIC_ROWS or len(manifests) != STATIC_ROWS:
        raise ValueError(f"static lane must contain exactly {STATIC_ROWS} candidates")
    seen: dict[str, set[str]] = {
        field: set() for field in ("uuid", "reference_sha256", "normalized_ast_sha256", "row_payload_sha256")
    }
    by_uuid: dict[str, dict[str, Any]] = {}
    family_counts: collections.Counter[str] = collections.Counter()
    template_counts: collections.Counter[str] = collections.Counter()
    input_skeletons: collections.Counter[str] = collections.Counter()
    model_skeletons: collections.Counter[str] = collections.Counter()
    pair_skeletons: collections.Counter[str] = collections.Counter()
    roots, commits = set(), set()
    candidate_references, candidate_normalized = set(), set()
    for index, (row, manifest) in enumerate(zip(rows, manifests, strict=True)):
        uuid, code = _row_uuid(row), _row_code(row)
        if manifest.get("uuid") != uuid or manifest.get("candidate_row_index") != index:
            raise ValueError(f"row/manifest identity mismatch:{index}")
        if manifest.get("generator_source_sha256") != generator_sha:
            raise ValueError(f"generator source drift:{uuid}")
        _verify_candidate_hashes(row, manifest, code, uuid)
        replay_code, replay_ops, replay_labels, replay_proof, replay_spec, replay_input_contract = _replay(
            generator, manifest
        )
        if (
            replay_code != code
            or manifest.get("declared_ops") != replay_ops
            or manifest.get("static_proof") != replay_proof
            or manifest.get("static_status") != "passed"
        ):
            raise ValueError(f"static exact replay mismatch:{uuid}")
        if (
            manifest.get("coverage_labels") != replay_labels
            or manifest.get("input_semantic_contract") != replay_input_contract
        ):
            raise ValueError(f"replay labels/input contract mismatch:{uuid}")
        # `spec` is the renderer's variant data, not a manifest object; in
        # particular it intentionally has no template_id.  Bind its stable
        # input facts against the replayed coverage labels instead.
        if any(
            replay_spec.get(name) != replay_labels.get(label)
            for name, label in (
                ("arity", "input_arity"),
                ("rank", "input_rank"),
                ("mode", "input_mode"),
                ("shape_bucket", "shape_bucket"),
            )
        ):
            raise ValueError(f"replay variant/input label mismatch:{uuid}")
        helpers = replay_proof.get("reachable_helper_classes")
        arity = replay_proof.get("input_arity")
        if (
            not isinstance(helpers, list)
            or "Model" not in helpers
            or type(arity) is not int
            or arity != replay_labels.get("input_arity")
        ):
            raise ValueError(f"static input/helper liveness proof is incomplete:{uuid}")
        family, template = manifest.get("primary_family"), manifest.get("template_id")
        if family not in contract["generated_family_quotas"] or template not in contract["template_recipe"]:
            raise ValueError(f"unknown family/template:{uuid}")
        recipe = manifest.get("recipe_id", contract["template_recipe"][template])
        if recipe != contract["template_recipe"][template]:
            raise ValueError(f"recipe/template mismatch:{uuid}")
        for field, values in seen.items():
            value = manifest.get(field)
            if not isinstance(value, str) or value in values:
                raise ValueError(f"missing or duplicate {field}:{uuid}")
            values.add(value)
        _require_sha(manifest["reference_sha256"], f"reference:{uuid}")
        input_skeleton = manifest.get("input_semantic_skeleton_sha256")
        model_skeleton = manifest.get("model_semantic_skeleton_sha256")
        pair_skeleton = manifest.get("model_input_semantic_skeleton_sha256")
        _require_sha(input_skeleton, f"input semantic skeleton:{uuid}")
        _require_sha(model_skeleton, f"model semantic skeleton:{uuid}")
        _require_sha(pair_skeleton, f"model/input semantic skeleton:{uuid}")
        if manifest.get("final_output_contract") not in (
            {"kind": "single_tensor", "finite_required": True},
            {"kind": "single_dense_tensor", "finite_required": True},
        ):
            raise ValueError(f"invalid output contract:{uuid}")
        input_info = _input_metrics(code, manifest)
        if any(value > REWORK_THRESHOLDS["axis_ratio_max_without_annotation"] for value in input_info["axis_ratios"]):
            raise ValueError(f"unannotated extreme input axis ratio:{uuid}")
        if ".expand(" in code or ".expand_as(" in code:
            raise ValueError(f"zero-stride expand is forbidden:{uuid}")
        if manifest.get("decontamination_status") != "exact_reference_and_normalized_ast_no_match":
            raise ValueError(f"decontamination not bound:{uuid}")
        root = _canonical_json(manifest.get("decontamination_roots"))
        if root == "null" or root == "[]":
            raise ValueError(f"decontamination roots missing:{uuid}")
        roots.add(root)
        commit = manifest.get("git_commit_at_generation")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError(f"invalid generator commit:{uuid}")
        commits.add(commit)
        family_counts[family] += 1
        template_counts[template] += 1
        input_skeletons[input_skeleton] += 1
        model_skeletons[model_skeleton] += 1
        pair_skeletons[pair_skeleton] += 1
        candidate_references.add(manifest["reference_sha256"])
        candidate_normalized.add(manifest["normalized_ast_sha256"])
        by_uuid[uuid] = {"row": row, "manifest": manifest, "index": index, "code": code, "recipe": recipe}
    if dict(family_counts) != contract["generated_family_quotas"]:
        raise ValueError(f"generated family quotas differ:{dict(family_counts)}")
    if any(count == 0 for count in template_counts.values()) or len(template_counts) != len(
        contract["template_recipe"]
    ):
        raise ValueError("static template coverage is incomplete")
    if len(input_skeletons) < REWORK_THRESHOLDS["input_semantic_skeleton_unique_min"]:
        raise ValueError("input semantic skeleton diversity gate failed")
    if (
        max(model_skeletons.values()) > REWORK_THRESHOLDS["model_semantic_skeleton_max_reuse"]
        or max(pair_skeletons.values()) > REWORK_THRESHOLDS["model_input_semantic_skeleton_max_reuse"]
    ):
        raise ValueError("model or model/input semantic skeleton concentration gate failed")
    if len(roots) != 1 or len(commits) != 1:
        raise ValueError("static candidates mix decontamination roots or generator commits")
    root_references, root_normalized, verified_roots = _verify_decontamination_roots(
        json.loads(next(iter(roots))), generator
    )
    if candidate_references & root_references or candidate_normalized & root_normalized:
        raise ValueError("static candidates collide with decontamination roots")
    return (
        table,
        manifests,
        by_uuid,
        {
            "generator_source_sha256": generator_sha,
            "generator_commit": next(iter(commits)),
            "family_counts": dict(family_counts),
            "template_counts": dict(template_counts),
            "input_semantic_skeleton_unique": len(input_skeletons),
            "model_semantic_skeleton_unique": len(model_skeletons),
            "model_input_semantic_skeleton_unique": len(pair_skeletons),
            "model_semantic_skeleton_max_reuse": max(model_skeletons.values()),
            "model_input_semantic_skeleton_max_reuse": max(pair_skeletons.values()),
            "decontamination_roots": verified_roots,
        },
    )


def _binding_sha(record: Mapping[str, Any], field: str, expected: str, uuid: str) -> None:
    direct = record.get(f"{field}_sha256")
    binding = record.get("binding_evidence")
    nested = (
        binding.get(field, {}).get("sha256")
        if isinstance(binding, Mapping) and isinstance(binding.get(field), Mapping)
        else None
    )
    if direct != expected and nested != expected:
        raise ValueError(f"{field} source binding mismatch:{uuid}")


def _memory_guard_ok(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and value.get("enabled") is True
        and value.get("within_limit") is True
        and value.get("over_limit") is False
        and value.get("max_device_memory_gib") == 64.0
    )


def _driver_h20_gpu_evidence_ok(value: Any) -> bool:
    """Validate the adapter driver's GPU identity, including shard routing."""

    return (
        isinstance(value, Mapping)
        and value.get("device") == "cuda:0"
        and "H20" in str(value.get("name"))
        and isinstance(value.get("cuda_visible_devices"), str)
        and bool(value["cuda_visible_devices"])
    )


def _worker_h20_gpu_evidence_ok(value: Any) -> bool:
    """Validate the generic worker schema (which deliberately has no CUDA_VISIBLE_DEVICES field)."""

    capability = value.get("compute_capability") if isinstance(value, Mapping) else None
    cuda_version = value.get("torch_cuda_version") if isinstance(value, Mapping) else None
    return (
        isinstance(value, Mapping)
        and value.get("device") == "cuda:0"
        and "H20" in str(value.get("name"))
        and isinstance(capability, list)
        and len(capability) == 2
        and all(type(part) is int for part in capability)
        and tuple(capability) == (9, 0)
        and isinstance(cuda_version, str)
        and re.fullmatch(r"[1-9][0-9]*\.[0-9]+(?:\.[0-9]+)?", cuda_version) is not None
    )


def _liveness_failed_record_ok(record: Mapping[str, Any]) -> bool:
    """Validate the adapter's real producer schema for pre-worker failures."""

    return liveness_adapter._failure_record_is_attributable(record)


def _liveness_passing_record_ok(record: Mapping[str, Any]) -> bool:
    return not liveness_adapter._authorization_evidence_missing(record)


def _read_ordered_allowlist(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"reference allowlist must be nonempty and unique:{path}")
    return values


def _current_reference_sources(generator: Any) -> dict[str, dict[str, str]]:
    paths = {
        "reference_adapter_source": Path(reference_adapter.__file__).resolve(),
        "generic_reference_core_source": Path(train_mode_contract_module.__file__).resolve(),
        "generator_source": Path(generator.__file__).resolve(),
        "launcher_source": _REPO_ROOT
        / "tools/data/synthesize/canary/operator_structure_10k_method/launch_operator_structure_10k_reference_rank.sh",
    }
    return {name: {"path": str(path), "sha256": _sha256_file(path)} for name, path in paths.items()}


def _require_artifact_pair(value: Any, expected: Mapping[str, str], label: str) -> None:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256"}
        or value.get("path") != expected["path"]
        or value.get("sha256") != expected["sha256"]
    ):
        raise ValueError(f"reference artifact binding mismatch:{label}")


def _verify_reference_record_binding(
    record: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    shard: int,
    host: str,
    candidates: Mapping[str, str],
    manifest: Mapping[str, str],
    allowlist: Mapping[str, str],
    generator: Any,
    sources: Mapping[str, Mapping[str, str]],
    kernelgym: Mapping[str, Any],
    shard_count: int,
    gpus_per_machine: int,
) -> None:
    """Check the lane adapter binding in addition to the generic record schema."""

    uuid = str(record.get("uuid"))
    evidence = record.get("reference_binding_evidence")
    if (
        not isinstance(evidence, Mapping)
        or record.get("candidate_row_index") != item["index"]
        or record.get("row_index") != item["index"]
        or record.get("candidates_sha256") != candidates["sha256"]
        or record.get("manifest_sha256") != manifest["sha256"]
        or record.get("allowlist_sha256") != allowlist["sha256"]
        or record.get("reference_binding_sha256") != _sha256_bytes(_canonical_json(evidence).encode())
        or evidence.get("binding_version") != reference_adapter.RUN_BINDING_VERSION
        or evidence.get("selection_contract_version") != reference_adapter.SELECTION_CONTRACT_VERSION
        or evidence.get("generator_module") != generator.__name__
        or evidence.get("execution_host") != host
        or evidence.get("global_shard_index") != shard
        or evidence.get("global_shard_count") != shard_count
        or evidence.get("candidates") != candidates
        or evidence.get("manifest") != manifest
        or evidence.get("allowlist") != allowlist
        or evidence.get("generic_reference_contract") != record.get("contract_payload")
        or record.get("kernelgym") != kernelgym
        or not _driver_h20_gpu_evidence_ok(record.get("driver_gpu"))
        or record["driver_gpu"].get("cuda_visible_devices") != str(shard % gpus_per_machine)
    ):
        raise ValueError(f"reference record binding/policy mismatch:{uuid}")
    for name, expected in sources.items():
        _source_pair(evidence.get(name), expected, f"reference-record:{uuid}:{name}")
        if record.get(f"{name}_sha256") != expected["sha256"]:
            raise ValueError(f"reference record source digest mismatch:{uuid}:{name}")
    if (
        record.get("validator_source_sha256") != sources["generic_reference_core_source"]["sha256"]
        or record.get("launcher_source_sha256") != sources["launcher_source"]["sha256"]
    ):
        raise ValueError(f"reference generic source digest mismatch:{uuid}")


def _verify_reference_authority(
    directory: Path,
    records: Sequence[Mapping[str, Any]],
    static: Mapping[str, Mapping[str, Any]],
    candidates_path: Path,
    manifest_path: Path,
    allowlist_path: Path,
    generator: Any,
    *,
    expected_uuids: Sequence[str],
) -> dict[str, Any]:
    """Verify the fixed 1x8 canary reference authority."""

    contract_path = directory / "global-reference-contract.json"
    summary_path = directory / "global-reference-summary.json"
    contract = _read_json_object(contract_path, "global reference contract")
    summary = _read_json_object(summary_path, "global reference summary")
    if (
        contract.get("contract_version") != REFERENCE_AUTHORITY_CONTRACT
        or summary.get("contract_version") != REFERENCE_AUTHORITY_SUMMARY
    ):
        raise ValueError("reference authority contract version mismatch")
    expected_candidates = {"path": str(candidates_path.resolve()), "sha256": _sha256_file(candidates_path)}
    expected_manifest = {"path": str(manifest_path.resolve()), "sha256": _sha256_file(manifest_path)}
    expected_allowlist = {"path": str(allowlist_path.resolve()), "sha256": _sha256_file(allowlist_path)}
    machine_count, gpus_per_machine, shard_count = (
        contract.get("machine_count"),
        contract.get("gpus_per_machine"),
        contract.get("global_shard_count"),
    )
    if (
        type(machine_count) is not int
        or type(gpus_per_machine) is not int
        or type(shard_count) is not int
        or machine_count <= 0
        or gpus_per_machine <= 0
        or shard_count != machine_count * gpus_per_machine
        or contract.get("candidate_rows") != STATIC_ROWS
        or contract.get("generator_module") != generator.__name__
    ):
        raise ValueError("reference authority geometry/generator mismatch")
    if (machine_count, gpus_per_machine, shard_count) != (1, 8, 8):
        raise ValueError("canary reference authority must be fixed 1x8/8")
    if contract.get("selection_scope") != "partial":
        raise ValueError(f"reference authority selection scope mismatch:{contract.get('selection_scope')!r}")
    _require_artifact_pair(contract.get("candidates"), expected_candidates, "candidates")
    _require_artifact_pair(contract.get("manifest"), expected_manifest, "manifest")
    _require_artifact_pair(contract.get("allowlist"), expected_allowlist, "allowlist")
    actual_allowlist = _read_ordered_allowlist(allowlist_path)
    if actual_allowlist != list(expected_uuids):
        raise ValueError("reference allowlist differs from the frozen selection")
    if len(actual_allowlist) >= STATIC_ROWS:
        raise ValueError("canary reference allowlist must be a strict subset")
    if summary.get("global_contract", {}).get("sha256") != _sha256_file(contract_path):
        raise ValueError("reference summary does not bind global contract")
    expected_sources = _current_reference_sources(generator)
    source_binding = contract.get("source_binding")
    if not isinstance(source_binding, Mapping) or set(source_binding) != set(REFERENCE_SOURCES):
        raise ValueError("reference authority source-binding schema mismatch")
    for name, expected in expected_sources.items():
        _source_pair(source_binding[name], expected, f"reference-global:{name}")
    kernelgym = contract.get("kernelgym")
    if not isinstance(kernelgym, Mapping):
        raise ValueError("reference authority KernelGym binding missing")
    ranks = contract.get("rank_evidence")
    if not isinstance(ranks, list) or {item.get("machine_rank") for item in ranks if isinstance(item, Mapping)} != set(
        range(machine_count)
    ):
        raise ValueError("reference authority rank evidence mismatch")
    records_by_uuid = {str(record.get("uuid")): record for record in records}
    if len(records_by_uuid) != len(records) or set(records_by_uuid) != set(expected_uuids):
        raise ValueError("reference authority UUID coverage has duplicates or gaps")
    if (
        summary.get("selected") != len(records)
        or summary.get("passed") != sum(record.get("passed") is True for record in records)
        or summary.get("failed") != sum(record.get("passed") is not True for record in records)
        or summary.get("executed", 0) + summary.get("resumed", 0) != len(records)
        or summary.get("rank_count") != machine_count
        or summary.get("shard_count") != shard_count
    ):
        raise ValueError("reference authority aggregate accounting mismatch")
    shards = contract.get("shards")
    if not isinstance(shards, list) or {
        item.get("global_shard_index") for item in shards if isinstance(item, Mapping)
    } != set(range(shard_count)):
        raise ValueError("reference authority shard set mismatch")
    rank_by_id = {int(item["machine_rank"]): item for item in ranks}
    seen: set[str] = set()
    for rank, rank_evidence in rank_by_id.items():
        host = rank_evidence.get("execution_host")
        inventory = rank_evidence.get("gpu_inventory")
        if (
            not isinstance(host, str)
            or not host
            or not isinstance(inventory, list)
            or len(inventory) != gpus_per_machine
            or not all("H20" in str(gpu) for gpu in inventory)
        ):
            raise ValueError(f"reference rank H20 inventory mismatch:{rank}")
        scheduler_pair, summary_pair = rank_evidence.get("scheduler_contract"), rank_evidence.get("launcher_summary")
        for pair, label in ((scheduler_pair, "scheduler"), (summary_pair, "summary")):
            if (
                not isinstance(pair, Mapping)
                or set(pair) != {"path", "sha256"}
                or not isinstance(pair.get("path"), str)
                or not Path(pair["path"]).is_file()
                or _sha256_file(Path(pair["path"])) != pair.get("sha256")
            ):
                raise ValueError(f"reference rank {label} archive mismatch:{rank}")
        scheduler = _read_json_object(Path(scheduler_pair["path"]), f"reference scheduler:{rank}")
        rank_summary = _read_json_object(Path(summary_pair["path"]), f"reference summary:{rank}")
        policy = {
            "machine_rank": rank,
            "machine_count": machine_count,
            "gpus_per_machine": gpus_per_machine,
            "global_shard_count": shard_count,
            "global_shard_indices": list(range(rank * gpus_per_machine, rank * gpus_per_machine + gpus_per_machine)),
            "execution_host": host,
            "candidates": expected_candidates,
            "manifest": expected_manifest,
            "allowlist": expected_allowlist,
            "generator_module": generator.__name__,
            "kernelgym": kernelgym,
            "trials": 5,
            "seed": 42,
            "max_device_memory_gib": 64.0,
            "training": True,
            "expected_mode_class": None,
        }
        if (
            scheduler.get("contract_version") != "operator_structure_10k_reference_rank_scheduler_v1"
            or any(scheduler.get(key) != value for key, value in policy.items())
            or not isinstance(scheduler.get("timeout_seconds"), (int, float))
            or scheduler["timeout_seconds"] <= 0
        ):
            raise ValueError(f"reference scheduler policy mismatch:{rank}")
        scheduler_sources = scheduler.get("source_binding")
        if not isinstance(scheduler_sources, Mapping) or set(scheduler_sources) != set(REFERENCE_SOURCES):
            raise ValueError(f"reference scheduler source schema mismatch:{rank}")
        for name, expected in expected_sources.items():
            _source_pair(scheduler_sources[name], expected, f"reference-scheduler:{rank}:{name}")
        archives = rank_evidence.get("source_archives")
        if not isinstance(archives, Mapping) or set(archives) != set(REFERENCE_SOURCES):
            raise ValueError(f"reference source archive schema mismatch:{rank}")
        for name, expected in expected_sources.items():
            pair = archives[name]
            if (
                not isinstance(pair, Mapping)
                or set(pair) != {"path", "sha256"}
                or not isinstance(pair.get("path"), str)
                or not Path(pair["path"]).is_file()
                or pair.get("sha256") != expected["sha256"]
                or _sha256_file(Path(pair["path"])) != expected["sha256"]
            ):
                raise ValueError(f"reference source archive/current-source mismatch:{rank}:{name}")
        if rank_summary.get("contract_version") != "operator_structure_10k_reference_rank_summary_v1" or any(
            rank_summary.get(key) != scheduler.get(key)
            for key in ("machine_rank", "machine_count", "gpus_per_machine", "global_shard_count", "execution_host")
        ):
            raise ValueError(f"reference rank summary identity mismatch:{rank}")
        if rank_summary.get("source_binding") != scheduler_sources:
            raise ValueError(f"reference rank summary source binding mismatch:{rank}")
        if rank_summary.get("kernelgym") != kernelgym:
            raise ValueError(f"reference rank summary KernelGym binding mismatch:{rank}")
        rank_shards = rank_summary.get("shards")
        if not isinstance(rank_shards, list) or {
            item.get("global_shard_index") for item in rank_shards if isinstance(item, Mapping)
        } != set(range(rank * gpus_per_machine, rank * gpus_per_machine + gpus_per_machine)):
            raise ValueError(f"reference rank shard ownership mismatch:{rank}")
        binding_by_shard = rank_summary.get("reference_binding_by_shard")
        expected_binding_keys = {
            str(index) for index in range(rank * gpus_per_machine, rank * gpus_per_machine + gpus_per_machine)
        }
        if (
            not isinstance(binding_by_shard, Mapping)
            or set(binding_by_shard) != expected_binding_keys
            or any(
                not isinstance(value, str) or SHA256_RE.fullmatch(value) is None for value in binding_by_shard.values()
            )
        ):
            raise ValueError(f"reference rank per-shard binding schema mismatch:{rank}")
        rank_shard_by_index = {int(item["global_shard_index"]): item for item in rank_shards}
        if any(
            rank_shard_by_index[index].get("reference_binding_sha256") != binding_by_shard[str(index)]
            for index in rank_shard_by_index
        ):
            raise ValueError(f"reference rank per-shard binding summary mismatch:{rank}")
        if any(
            rank_summary.get(key) != sum(item.get(key, 0) for item in rank_shards)
            for key in ("selected", "executed", "resumed", "passed", "failed")
        ):
            raise ValueError(f"reference rank accounting mismatch:{rank}")
    for shard_info in shards:
        shard = shard_info["global_shard_index"]
        path = directory / f"shard-{shard:02d}-of-{shard_count:02d}.jsonl"
        if not path.is_file() or shard_info.get("sha256") != _sha256_file(path):
            raise ValueError(f"reference authority shard hash mismatch:{shard}")
        shard_records = _read_jsonl(path)
        if shard_info.get("rows") != len(shard_records):
            raise ValueError(f"reference authority shard row count mismatch:{shard}")
        rank = shard // gpus_per_machine
        rank_evidence = rank_by_id[rank]
        rank_summary = _read_json_object(Path(rank_evidence["launcher_summary"]["path"]), f"reference summary:{rank}")
        rank_shard = next(item for item in rank_summary["shards"] if item["global_shard_index"] == shard)
        if (
            rank_shard.get("selected") != len(shard_records)
            or rank_shard.get("passed") != sum(record.get("passed") is True for record in shard_records)
            or rank_shard.get("failed") != sum(record.get("passed") is not True for record in shard_records)
            or rank_shard.get("selected") != rank_shard.get("executed", 0) + rank_shard.get("resumed", 0)
        ):
            raise ValueError(f"reference shard accounting mismatch:{shard}")
        for record in shard_records:
            uuid = str(record.get("uuid"))
            item = static.get(uuid)
            if (
                uuid in seen
                or item is None
                or item["index"] % shard_count != shard
                or records_by_uuid.get(uuid) != record
            ):
                raise ValueError(f"reference shard ownership/record mismatch:{shard}:{uuid}")
            if record.get("reference_binding_sha256") != rank_shard.get("reference_binding_sha256"):
                raise ValueError(f"reference shard binding mismatch:{shard}:{uuid}")
            _verify_reference_record_binding(
                record,
                item,
                shard=shard,
                host=rank_evidence["execution_host"],
                candidates=expected_candidates,
                manifest=expected_manifest,
                allowlist=expected_allowlist,
                generator=generator,
                sources=expected_sources,
                kernelgym=kernelgym,
                shard_count=shard_count,
                gpus_per_machine=gpus_per_machine,
            )
            seen.add(uuid)
    if seen != set(expected_uuids):
        raise ValueError("reference authority raw shard coverage differs from allowlist")
    return {
        "global_contract": {"path": str(contract_path.resolve()), "sha256": _sha256_file(contract_path)},
        "global_summary": {"path": str(summary_path.resolve()), "sha256": _sha256_file(summary_path)},
        "shard_count": shard_count,
        "rank_count": machine_count,
        "gpus_per_machine": gpus_per_machine,
        "selected_rows": len(expected_uuids),
        "formal": False,
    }


def _index_reference(
    records: Sequence[Mapping[str, Any]],
    static: Mapping[str, Mapping[str, Any]],
    candidates_sha: str,
    *,
    expected_uuids: Sequence[str] | None = None,
) -> dict[str, Mapping[str, Any]]:
    expected = set(static) if expected_uuids is None else set(expected_uuids)
    if not expected or len(records) != len(expected):
        raise ValueError(f"reference row count differs from frozen selection:{len(records)}/{len(expected)}")
    indexed: dict[str, Mapping[str, Any]] = {}
    validator_sha = _sha256_file(_REPO_ROOT / "tools/data/synthesize/validate_train_mode_contract.py")
    fingerprints, launchers, kernelgym_bundles = set(), set(), set()
    for record in records:
        uuid = record.get("uuid")
        if not isinstance(uuid, str) or uuid not in static or uuid in indexed:
            raise ValueError(f"unknown/duplicate reference UUID:{uuid!r}")
        item = static[uuid]
        row_index, source_sha = record.get("row_index"), record.get("source_sha256")
        # Generic validator schema uses row_index/source_sha256.  If a caller
        # also emits lane-local aliases, they are allowed only when identical.
        if (
            row_index != item["index"]
            or source_sha != candidates_sha
            or ("candidate_row_index" in record and record["candidate_row_index"] != row_index)
            or ("candidates_sha256" in record and record["candidates_sha256"] != source_sha)
        ):
            raise ValueError(f"reference candidate join mismatch:{uuid}")
        if (
            record.get("reference_sha256") != item["manifest"]["reference_sha256"]
            or record.get("contract_version") != REFERENCE_CONTRACT
            or record.get("validator_source_sha256") != validator_sha
            or record.get("trials") != 5
            or record.get("seed") != 42
            or record.get("training") is not True
            or record.get("device") != "cuda:0"
            or record.get("max_device_memory_gib") != 64.0
        ):
            raise ValueError(f"reference identity/source/policy mismatch:{uuid}")
        payload = record.get("contract_payload")
        if (
            not isinstance(payload, Mapping)
            or payload.get("contract_version") != REFERENCE_CONTRACT
            or payload.get("expected_mode_class") is not None
        ):
            raise ValueError(f"reference standalone contract payload mismatch:{uuid}")
        for field, expected_value in (
            ("source_sha256", source_sha),
            ("validator_source_sha256", validator_sha),
            ("launcher_source_sha256", record.get("launcher_source_sha256")),
            ("trials", 5),
            ("seed", 42),
            ("training", True),
            ("device", "cuda:0"),
            ("max_device_memory_gib", 64.0),
        ):
            if payload.get(field) != expected_value:
                raise ValueError(f"reference contract payload binding mismatch:{uuid}:{field}")
        launcher = _require_sha(record.get("launcher_source_sha256"), f"reference launcher source:{uuid}")
        fingerprint = _require_sha(record.get("contract_fingerprint"), f"reference contract fingerprint:{uuid}")
        kernelgym = record.get("kernelgym")
        if (
            not isinstance(kernelgym, Mapping)
            or not isinstance(kernelgym.get("root"), str)
            or not isinstance(kernelgym.get("git_commit"), str)
            or re.fullmatch(r"[0-9a-f]{40}", kernelgym["git_commit"]) is None
            or any(
                _require_sha(kernelgym.get(field), f"reference KernelGym:{uuid}:{field}")
                != payload.get(f"kernelgym_{field}")
                for field in KERNELGYM_HASH_FIELDS
            )
        ):
            raise ValueError(f"reference KernelGym bundle mismatch:{uuid}")
        fingerprints.add(fingerprint)
        launchers.add(launcher)
        kernelgym_bundles.add(_canonical_json(kernelgym))
        # A loader/input failure can occur before the worker constructs its
        # per-row ``gpu`` payload.  The adapter therefore records a driver
        # H20 identity for every row; passing rows must also have complete
        # worker H20 and allocator evidence.  This preserves attributable
        # failures without accepting an unproven non-GPU execution.
        if not _driver_h20_gpu_evidence_ok(record.get("driver_gpu")):
            raise ValueError(f"reference driver H20 evidence invalid:{uuid}")
        gpu = record.get("gpu")
        if record.get("passed") is True and (
            not _worker_h20_gpu_evidence_ok(gpu) or not _memory_guard_ok(record.get("memory_guard"))
        ):
            raise ValueError(f"reference passing H20/memory evidence invalid:{uuid}")
        if type(record.get("passed")) is not bool:
            raise ValueError(f"reference pass flag invalid:{uuid}")
        if record["passed"] and (
            record.get("status") != "passed"
            or record.get("reference_forward_calls") != 5
            or record.get("identical_forward_calls") != 5
            or record.get("reference_training") is not True
            or record.get("identical_training") is not True
            or record.get("persistent_model_instances") is not True
        ):
            raise ValueError(f"fresh five-trial reference evidence incomplete:{uuid}")
        indexed[uuid] = record
    if set(indexed) != expected:
        raise ValueError(
            f"reference UUID coverage differs from frozen selection:observed={len(indexed)} expected={len(expected)}"
        )
    if len(fingerprints) != 1 or len(launchers) != 1 or len(kernelgym_bundles) != 1:
        raise ValueError("reference evidence mixes contract fingerprints, launchers, or KernelGym bundles")
    return indexed


def _index_liveness(
    records: Sequence[Mapping[str, Any]],
    static: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
    candidates_sha: str,
    manifest_sha: str,
    generator_sha: str,
    allowlist_path: Path,
) -> dict[str, Mapping[str, Any]]:
    reference_passed = [
        uuid
        for uuid, item in sorted(static.items(), key=lambda pair: pair[1]["index"])
        if references[uuid].get("passed") is True
    ]
    allowlist = [line.strip() for line in allowlist_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if allowlist != reference_passed or len(allowlist) != len(set(allowlist)):
        raise ValueError("liveness allowlist is not the fresh ordered reference-pass UUID set")
    if len(records) != len(reference_passed):
        raise ValueError("liveness rows do not exactly cover fresh reference passes")
    indexed: dict[str, Mapping[str, Any]] = {}
    liveness_sources: dict[str, set[str]] = {
        f"{name}_sha256": set()
        for name in (
            "adapter_source",
            "shared_runtime_core_source",
            "runtime_validation_source",
            "train_mode_contract_source",
            "semantic_operator_generator_source",
            "generator_source",
            "launcher_source",
        )
    }
    for record in records:
        uuid = record.get("uuid")
        if not isinstance(uuid, str) or uuid not in reference_passed or uuid in indexed:
            raise ValueError(f"unknown/duplicate/non-reference liveness UUID:{uuid!r}")
        binding_evidence = record.get("binding_evidence")
        if (
            record.get("candidate_row_index") != static[uuid]["index"]
            or any(
                record.get(field) != static[uuid]["manifest"].get(field)
                for field in ("template_id", "primary_family", "mode_behavior")
            )
            or record.get("candidates_sha256") != candidates_sha
            or record.get("manifest_sha256") != manifest_sha
            or not isinstance(binding_evidence, Mapping)
            or record.get("validation_config") != binding_evidence.get("validation_config")
        ):
            raise ValueError(f"liveness candidate/manifest join mismatch:{uuid}")
        _binding_sha(record, "generator_source", generator_sha, uuid)
        for field, values in liveness_sources.items():
            values.add(_require_sha(record.get(field), f"liveness source:{field}:{uuid}"))
        if type(record.get("passed")) is not bool or not _driver_h20_gpu_evidence_ok(record.get("driver_gpu")):
            raise ValueError(f"liveness status/driver-GPU invalid:{uuid}")
        if record["passed"]:
            trials = record.get("trials")
            if (
                not _liveness_passing_record_ok(record)
                or record.get("status") != "passed"
                or record.get("persistent_model_instances") is not True
                or record.get("training_mode_preserved") is not True
                or record.get("final_output_kind") not in FINAL_OUTPUTS
                or not isinstance(trials, list)
                or len(trials) != 3
            ):
                raise ValueError(f"liveness successful evidence incomplete:{uuid}")
            for trial in trials:
                expected_ops = {
                    item.get("op_id"): item
                    for item in static[uuid]["manifest"].get("declared_ops", [])
                    if isinstance(item, Mapping) and isinstance(item.get("op_id"), str)
                }
                trace = trial.get("trace") if isinstance(trial, Mapping) else None
                per_op = trace.get("per_declared_op") if isinstance(trace, Mapping) else None
                observed = (
                    {item.get("op_id"): item for item in per_op if isinstance(item, Mapping)}
                    if isinstance(per_op, list)
                    else {}
                )
                if (
                    not isinstance(trial, Mapping)
                    or trial.get("output_finite") is not True
                    or trial.get("single_tensor_output") is not True
                    or trial.get("inputs_immutable") is not True
                    or trial.get("control_trace_output_exact") is not True
                    or trial.get("control_trace_state_exact") is not True
                    or not expected_ops
                    or set(observed) != set(expected_ops)
                    or trace.get("final_output_declared_op_ids") != sorted(expected_ops)
                ):
                    raise ValueError(f"liveness trial evidence invalid:{uuid}")
                for op_id, declaration in expected_ops.items():
                    evidence = observed[op_id]
                    if (
                        evidence.get("returned_output_witness") is not True
                        or not isinstance(evidence.get("calls"), int)
                        or evidence["calls"] < declaration.get("min_calls_per_trial", 1)
                    ):
                        raise ValueError(f"declared operator lacks final-output provenance:{uuid}:{op_id}")
        elif not _liveness_failed_record_ok(record):
            raise ValueError(f"liveness failure producer schema invalid:{uuid}")
        indexed[uuid] = record
    if set(indexed) != set(reference_passed):
        raise ValueError("liveness UUID coverage differs from reference-pass set")
    if any(len(values) != 1 for values in liveness_sources.values()):
        raise ValueError("liveness evidence mixes source hashes")
    return indexed


def _read_json_object(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}:{path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object:{path}")
    return value


def _source_pair(value: Any, expected: Mapping[str, str], label: str, *, core: bool = False) -> None:
    required = {"path", "sha256", "contract_version"} if core else {"path", "sha256"}
    if (
        not isinstance(value, Mapping)
        or set(value) != required
        or value.get("path") != expected["path"]
        or value.get("sha256") != expected["sha256"]
    ):
        raise ValueError(f"liveness source binding mismatch:{label}")
    if core and value.get("contract_version") != runtime_core.CONTRACT_VERSION:
        raise ValueError(f"liveness runtime core contract mismatch:{label}")


def _current_liveness_sources(generator: Any) -> dict[str, dict[str, str]]:
    paths = {
        "adapter_source": Path(liveness_adapter.__file__).resolve(),
        "shared_runtime_core_source": Path(runtime_core.__file__).resolve(),
        "runtime_validation_source": Path(runtime_validation_module.__file__).resolve(),
        "train_mode_contract_source": Path(train_mode_contract_module.__file__).resolve(),
        "semantic_operator_generator_source": Path(semantic_operator_generator.__file__).resolve(),
        "generator_source": Path(generator.__file__).resolve(),
        "launcher_source": _REPO_ROOT
        / "tools/data/synthesize/canary/operator_structure_10k_method/launch_operator_structure_10k_liveness_rank.sh",
    }
    return {name: {"path": str(path), "sha256": _sha256_file(path)} for name, path in paths.items()}


def _verify_liveness_record_binding(
    record: Mapping[str, Any],
    item: Mapping[str, Any],
    *,
    shard: int,
    shard_count: int = LIVENESS_SHARDS,
    host: str,
    timeout_seconds: float,
    candidates: Mapping[str, Any],
    manifest: Mapping[str, Any],
    allowlist: Mapping[str, Any],
    generator: Any,
    sources: Mapping[str, Mapping[str, str]],
    reference_passed_binding: Mapping[str, str] | None = None,
) -> None:
    uuid = str(record.get("uuid"))
    evidence = record.get("binding_evidence")
    expected_config = {
        "device": "cuda:0",
        "trials": 3,
        "seed": 17,
        "timeout_seconds": timeout_seconds,
        "max_device_memory_gib": 64.0,
        "persistent_train_mode_models": True,
        "single_tensor_final_output": True,
        "execution_controls": liveness_adapter.EXECUTION_CONTROLS,
    }
    if (
        not isinstance(evidence, Mapping)
        or record.get("contract_version") != liveness_adapter.CONTRACT_VERSION
        or record.get("raw_runtime_core_contract_version") != runtime_core.CONTRACT_VERSION
        or record.get("candidate_row_index") != item["index"]
        or any(
            record.get(field) != item["manifest"].get(field)
            for field in ("template_id", "primary_family", "mode_behavior")
        )
        or record.get("candidates_sha256") != candidates["sha256"]
        or record.get("manifest_sha256") != manifest["sha256"]
        or record.get("allowlist_sha256") != allowlist["sha256"]
        or record.get("validation_config") != expected_config
        or record.get("validation_binding_sha256") != _sha256_bytes(_canonical_json(evidence).encode())
        or evidence.get("contract_version") != liveness_adapter.CONTRACT_VERSION
        or evidence.get("binding_version") != liveness_adapter.RUN_BINDING_VERSION
        or evidence.get("generator_module") != generator.__name__
        or evidence.get("total_rows") != STATIC_ROWS
        or evidence.get("execution_host") != host
        or evidence.get("global_shard_index") != shard
        or evidence.get("global_shard_count") != shard_count
        or evidence.get("candidates_path") != candidates["path"]
        or evidence.get("candidates_sha256") != candidates["sha256"]
        or evidence.get("manifest_path") != manifest["path"]
        or evidence.get("manifest_sha256") != manifest["sha256"]
        or evidence.get("allowlist_path") != allowlist["path"]
        or evidence.get("allowlist_sha256") != allowlist["sha256"]
        or evidence.get("validation_config") != expected_config
    ):
        raise ValueError(f"liveness record binding/policy mismatch:{uuid}")
    if reference_passed_binding is not None and evidence.get("reference_passed_binding") != reference_passed_binding:
        raise ValueError(f"liveness reference-pass binding mismatch:{uuid}")
    for name, expected in sources.items():
        _source_pair(evidence.get(name), expected, f"record:{uuid}:{name}", core=name == "shared_runtime_core_source")
        if record.get(f"{name}_sha256") != expected["sha256"]:
            raise ValueError(f"liveness record source digest mismatch:{uuid}:{name}")
    driver_gpu = record.get("driver_gpu")
    if not _driver_h20_gpu_evidence_ok(driver_gpu) or driver_gpu.get("cuda_visible_devices") != str(shard % 8):
        raise ValueError(f"liveness record driver H20 evidence mismatch:{uuid}")
    if record.get("passed") is True:
        if not _liveness_passing_record_ok(record):
            raise ValueError(f"liveness passing worker H20/memory evidence mismatch:{uuid}")
    elif not _liveness_failed_record_ok(record):
        raise ValueError(f"liveness failed record producer schema mismatch:{uuid}")


def _verify_canary_liveness_authority(
    directory: Path,
    records: Sequence[Mapping[str, Any]],
    static: Mapping[str, Mapping[str, Any]],
    candidates: Path,
    manifest: Path,
    allowlist: Path,
    generator: Any,
    *,
    expected_uuids: Sequence[str],
    reference_passed_binding: Mapping[str, str],
) -> dict[str, Any]:
    """Audit the isolated 1x8/240 liveness authority; never accept it formally."""

    machine_count, gpus_per_machine, shard_count = 1, 8, 8
    contract_path = directory / "global-liveness-contract.json"
    summary_path = directory / "global-liveness-summary.json"
    contract = _read_json_object(contract_path, "canary liveness contract")
    summary = _read_json_object(summary_path, "canary liveness summary")
    expected_candidates = {"path": str(candidates.resolve()), "sha256": _sha256_file(candidates)}
    expected_manifest = {"path": str(manifest.resolve()), "sha256": _sha256_file(manifest)}
    expected_allowlist = {"path": str(allowlist.resolve()), "sha256": _sha256_file(allowlist)}
    merge_path = (
        _REPO_ROOT
        / "tools/data/synthesize/canary/operator_structure_10k_method/merge_operator_structure_10k_evidence.py"
    )
    expected_merge = {"path": str(merge_path), "sha256": _sha256_file(merge_path)}
    summary_contract = summary.get("global_contract")
    if (
        contract.get("contract_version") != "operator_structure_10k_global_liveness_evidence_v1"
        or contract.get("selection_scope") != "partial"
        or contract.get("candidate_rows") != STATIC_ROWS
        or (contract.get("machine_count"), contract.get("gpus_per_machine"), contract.get("global_shard_count"))
        != (machine_count, gpus_per_machine, shard_count)
        or contract.get("generator_module") != generator.__name__
        or contract.get("candidates") != expected_candidates
        or contract.get("manifest") != expected_manifest
        or contract.get("allowlist") != expected_allowlist
        or contract.get("reference_passed_uuids") != expected_allowlist
        or contract.get("reference_passed_binding") != reference_passed_binding
        or contract.get("merge_source") != expected_merge
        or summary.get("contract_version") != "operator_structure_10k_global_liveness_summary_v1"
        or summary.get("selection_scope") != "partial"
        or summary.get("merge_source") != expected_merge
        or not isinstance(summary_contract, Mapping)
        or set(summary_contract) != {"path", "sha256"}
        or not isinstance(summary_contract.get("path"), str)
        or Path(summary_contract["path"]).resolve() != contract_path.resolve()
        or summary_contract.get("sha256") != _sha256_file(contract_path)
    ):
        raise ValueError("canary liveness authority contract/binding mismatch")
    records_by_uuid = {str(record.get("uuid")): record for record in records}
    if len(records_by_uuid) != len(records) or set(records_by_uuid) != set(expected_uuids):
        raise ValueError("canary liveness UUID coverage has duplicates or gaps")
    if (
        summary.get("selected") != len(records)
        or summary.get("passed") != sum(record.get("passed") is True for record in records)
        or summary.get("failed") != sum(record.get("passed") is not True for record in records)
        or summary.get("executed", 0) + summary.get("resumed", 0) != len(records)
        or summary.get("rank_count") != machine_count
        or summary.get("shard_count") != shard_count
    ):
        raise ValueError("canary liveness aggregate accounting mismatch")
    expected_sources = _current_liveness_sources(generator)
    source_binding = contract.get("source_binding")
    if not isinstance(source_binding, Mapping) or set(source_binding) != set(expected_sources):
        raise ValueError("canary liveness source-binding schema mismatch")
    for name, expected in expected_sources.items():
        _source_pair(source_binding[name], expected, f"canary-global:{name}")
    ranks = contract.get("rank_evidence")
    if (
        not isinstance(ranks, list)
        or len(ranks) != 1
        or not isinstance(ranks[0], Mapping)
        or ranks[0].get("machine_rank") != 0
    ):
        raise ValueError("canary liveness rank evidence mismatch")
    rank_evidence = ranks[0]
    authority_root = directory.resolve()
    host, inventory = rank_evidence.get("execution_host"), rank_evidence.get("gpu_inventory")
    if (
        not isinstance(host, str)
        or not host
        or not isinstance(inventory, list)
        or len(inventory) != gpus_per_machine
        or not all("H20" in str(gpu) for gpu in inventory)
    ):
        raise ValueError("canary liveness H20 inventory mismatch")
    for field, label in (("scheduler_contract", "scheduler"), ("launcher_summary", "summary")):
        pair = rank_evidence.get(field)
        if (
            not isinstance(pair, Mapping)
            or set(pair) != {"path", "sha256"}
            or not isinstance(pair.get("path"), str)
            or not Path(pair["path"]).is_file()
            or not Path(pair["path"]).resolve().is_relative_to(authority_root)
            or pair.get("sha256") != _sha256_file(Path(pair["path"]))
        ):
            raise ValueError(f"canary liveness rank {label} archive mismatch")
    scheduler = _read_json_object(Path(rank_evidence["scheduler_contract"]["path"]), "canary liveness scheduler")
    rank_summary = _read_json_object(Path(rank_evidence["launcher_summary"]["path"]), "canary liveness rank summary")
    policy = {
        "machine_rank": 0,
        "machine_count": machine_count,
        "gpus_per_machine": gpus_per_machine,
        "global_shard_count": shard_count,
        "global_shard_indices": list(range(shard_count)),
        "execution_host": host,
        "candidates": expected_candidates,
        "manifest": expected_manifest,
        "allowlist": expected_allowlist,
        "reference_passed_binding": reference_passed_binding,
        "generator_module": generator.__name__,
        "trials": 3,
        "seed": 17,
        "max_device_memory_gib": 64.0,
        "persistent_train_mode_models": True,
        "single_tensor_final_output": True,
        "execution_controls": liveness_adapter.EXECUTION_CONTROLS,
    }
    if (
        scheduler.get("contract_version") != "operator_structure_10k_liveness_rank_scheduler_v1"
        or any(scheduler.get(key) != value for key, value in policy.items())
        or not isinstance(scheduler.get("timeout_seconds"), (int, float))
        or scheduler["timeout_seconds"] <= 0
    ):
        raise ValueError("canary liveness scheduler policy mismatch")
    scheduler_sources = scheduler.get("source_binding")
    if not isinstance(scheduler_sources, Mapping) or set(scheduler_sources) != set(expected_sources):
        raise ValueError("canary liveness scheduler source schema mismatch")
    for name, expected in expected_sources.items():
        _source_pair(scheduler_sources[name], expected, f"canary-scheduler:{name}")
    archives = rank_evidence.get("source_archives")
    if not isinstance(archives, Mapping) or set(archives) != set(expected_sources):
        raise ValueError("canary liveness source archive schema mismatch")
    for name, expected in expected_sources.items():
        pair = archives[name]
        if (
            not isinstance(pair, Mapping)
            or set(pair) != {"path", "sha256"}
            or pair.get("sha256") != expected["sha256"]
            or not isinstance(pair.get("path"), str)
            or not Path(pair["path"]).is_file()
            or not Path(pair["path"]).resolve().is_relative_to(authority_root)
            or _sha256_file(Path(pair["path"])) != expected["sha256"]
        ):
            raise ValueError(f"canary liveness source archive/current-source mismatch:{name}")
    if rank_summary.get("contract_version") != "operator_structure_10k_liveness_rank_summary_v1" or any(
        rank_summary.get(key) != scheduler.get(key)
        for key in ("machine_rank", "machine_count", "gpus_per_machine", "global_shard_count", "execution_host")
    ):
        raise ValueError("canary liveness rank summary identity mismatch")
    summary_sources = rank_summary.get("source_binding")
    if not isinstance(summary_sources, Mapping) or set(summary_sources) != set(expected_sources):
        raise ValueError("canary liveness rank summary source schema mismatch")
    for name, expected in expected_sources.items():
        _source_pair(
            summary_sources[name], expected, f"canary-summary:{name}", core=name == "shared_runtime_core_source"
        )
    rank_shards = rank_summary.get("shards")
    if not isinstance(rank_shards, list) or {
        item.get("global_shard_index") for item in rank_shards if isinstance(item, Mapping)
    } != set(range(shard_count)):
        raise ValueError("canary liveness rank shard ownership mismatch")
    binding_by_shard = rank_summary.get("validation_binding_by_shard")
    if (
        not isinstance(binding_by_shard, Mapping)
        or set(binding_by_shard) != {str(index) for index in range(shard_count)}
        or any(not isinstance(value, str) or SHA256_RE.fullmatch(value) is None for value in binding_by_shard.values())
    ):
        raise ValueError("canary liveness rank binding map mismatch")
    if any(
        rank_summary.get(key) != sum(item.get(key, 0) for item in rank_shards)
        for key in ("selected", "executed", "resumed", "passed", "failed")
    ):
        raise ValueError("canary liveness rank accounting mismatch")
    shards = contract.get("shards")
    if not isinstance(shards, list) or {
        item.get("global_shard_index") for item in shards if isinstance(item, Mapping)
    } != set(range(shard_count)):
        raise ValueError("canary liveness global shard set mismatch")
    summary_by_shard = {int(item["global_shard_index"]): item for item in rank_shards}
    seen: set[str] = set()
    for shard_info in shards:
        shard = shard_info["global_shard_index"]
        path = directory / f"shard-{shard:02d}-of-{shard_count:02d}.jsonl"
        if not path.is_file() or shard_info.get("sha256") != _sha256_file(path):
            raise ValueError(f"canary liveness shard hash mismatch:{shard}")
        shard_records = _read_jsonl(path)
        logged = summary_by_shard[shard]
        if (
            shard_info.get("rows") != len(shard_records)
            or logged.get("selected") != len(shard_records)
            or logged.get("passed") != sum(record.get("passed") is True for record in shard_records)
            or logged.get("failed") != sum(record.get("passed") is not True for record in shard_records)
            or logged.get("selected") != logged.get("executed", 0) + logged.get("resumed", 0)
            or logged.get("reference_binding_sha256", logged.get("validation_binding_sha256"))
            != binding_by_shard[str(shard)]
        ):
            raise ValueError(f"canary liveness shard accounting/binding mismatch:{shard}")
        for record in shard_records:
            uuid = str(record.get("uuid"))
            item = static.get(uuid)
            if (
                uuid in seen
                or item is None
                or item["index"] % shard_count != shard
                or records_by_uuid.get(uuid) != record
            ):
                raise ValueError(f"canary liveness shard ownership/record mismatch:{shard}:{uuid}")
            if record.get("validation_binding_sha256") != binding_by_shard[str(shard)]:
                raise ValueError(f"canary liveness record/shard binding mismatch:{shard}:{uuid}")
            _verify_liveness_record_binding(
                record,
                item,
                shard=shard,
                shard_count=shard_count,
                host=host,
                timeout_seconds=float(scheduler["timeout_seconds"]),
                candidates=expected_candidates,
                manifest=expected_manifest,
                allowlist=expected_allowlist,
                generator=generator,
                sources=expected_sources,
                reference_passed_binding=reference_passed_binding,
            )
            seen.add(uuid)
    if seen != set(expected_uuids):
        raise ValueError("canary liveness shard coverage differs from derived reference passes")
    return {
        "global_contract": {"path": str(contract_path.resolve()), "sha256": _sha256_file(contract_path)},
        "global_summary": {"path": str(summary_path.resolve()), "sha256": _sha256_file(summary_path)},
        "selected_rows": len(expected_uuids),
        "rank_count": 1,
        "shard_count": 8,
        "formal": False,
    }


def _get_inputs(code: str) -> ast.FunctionDef | None:
    tree = ast.parse(code)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_inputs"]
    return functions[-1] if functions else None


def _input_metrics(code: str, manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    function = _get_inputs(code)
    semantic = (
        manifest.get("input_semantic_contract")
        if isinstance(manifest, Mapping) and isinstance(manifest.get("input_semantic_contract"), Mapping)
        else {}
    )
    defaults = {
        "arity": None,
        "ranks": [],
        "numels": [],
        "axis_ratios": [],
        "value_domain": "unknown",
        "layout": "unknown",
        "skeleton": manifest.get("input_semantic_skeleton_sha256") if manifest else None,
    }
    if function is None:
        return defaults
    module_tree = ast.parse(code)
    observations = distribution_profiler._exact_factory_observations(  # noqa: SLF001
        function,
        distribution_profiler._module_environment(module_tree),  # noqa: SLF001
    )
    factories = [base for base, _shape, _reason in observations]
    shapes = [shape for _base, shape, _reason in observations if shape is not None and math.prod(shape) > 0]
    calls = {_call_name(node.func) for node in ast.walk(function) if isinstance(node, ast.Call)}
    value_contract = semantic.get("value_contract")
    value_domain = (
        value_contract
        if isinstance(value_contract, str)
        else (
            "normal_signed"
            if "randn" in factories
            else (
                "discrete"
                if "randint" in factories
                else (
                    "unit_uniform"
                    if "rand" in factories
                    else "positive_counts" if "poisson" in factories else "unknown"
                )
            )
        )
    )
    layout_contract = semantic.get("layout_contract")
    layout = (
        layout_contract
        if isinstance(layout_contract, str)
        else (
            "broadcast_repeat"
            if any(name.endswith("expand") for name in calls)
            else (
                "channel_last_or_permuted"
                if any(name.endswith(suffix) for suffix in ("movedim", "transpose", "permute") for name in calls)
                else (
                    "strided_slice"
                    if any(isinstance(node, ast.Subscript) for node in ast.walk(function))
                    else "contiguous"
                )
            )
        )
    )
    arity = None
    for statement in function.body:
        if isinstance(statement, ast.Return) and isinstance(statement.value, (ast.List, ast.Tuple)):
            arity = len(statement.value.elts)
    ranks = [len(shape) for shape in shapes]
    numels = [math.prod(shape) for shape in shapes]
    ratios = [max(shape) / min(shape) for shape in shapes if shape and min(shape) > 0]
    return {
        "arity": arity if arity is not None else len(shapes) or None,
        "ranks": ranks,
        "numels": numels,
        "axis_ratios": ratios,
        "value_domain": value_domain,
        "layout": layout,
        "skeleton": defaults["skeleton"],
    }


def _materialize_reference_passed(
    output: Path,
    *,
    selected_uuids: Sequence[str],
    indexed: Mapping[str, Mapping[str, Any]],
    static: Mapping[str, Mapping[str, Any]],
    candidates: Path,
    manifest: Path,
    selected_allowlist: Path,
    authority: Mapping[str, Any],
    selection_summary: Path | None,
) -> dict[str, dict[str, str]]:
    """Persist the exact reference-pass set for a subsequent liveness run."""

    ordered = sorted(
        (uuid for uuid in selected_uuids if indexed[uuid].get("passed") is True),
        key=lambda uuid: static[uuid]["index"],
    )
    uuid_path = output.with_name(f"{output.stem}.uuids.txt")
    if not ordered:
        raise ValueError("reference-pass derivation is empty")
    # Preflight both identities before creating either.  A failed retry must
    # never silently reuse or overwrite a partial derivation from another run.
    if output.exists() or uuid_path.exists():
        raise FileExistsError(f"reference-pass derived evidence output already exists:{output}")
    if not output.parent.is_dir() or uuid_path.parent != output.parent:
        raise ValueError(f"reference-pass derived evidence parent must already exist:{output.parent}")
    _write_new_atomically(uuid_path, "".join(f"{uuid}\n" for uuid in ordered))
    uuid_pair = {"path": str(uuid_path.resolve()), "sha256": _sha256_file(uuid_path)}
    payload: dict[str, Any] = {
        "contract_version": REFERENCE_PASSED_BINDING_CONTRACT,
        "selection_scope": "partial",
        "candidate_rows": STATIC_ROWS,
        "selected_rows": len(selected_uuids),
        "reference_passed_rows": len(ordered),
        "ordering": "original_candidate_row_index",
        "candidates": {"path": str(candidates.resolve()), "sha256": _sha256_file(candidates)},
        "manifest": {"path": str(manifest.resolve()), "sha256": _sha256_file(manifest)},
        "selected_allowlist": {"path": str(selected_allowlist.resolve()), "sha256": _sha256_file(selected_allowlist)},
        "reference_authority": {
            "global_contract": authority["global_contract"],
            "global_summary": authority["global_summary"],
        },
        "reference_passed_uuids": uuid_pair,
    }
    if selection_summary is not None:
        payload["selection_summary"] = {
            "path": str(selection_summary.resolve()),
            "sha256": _sha256_file(selection_summary),
        }
    _write_new_atomically(output, json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return {
        "binding": {"path": str(output.resolve()), "sha256": _sha256_file(output)},
        "uuids": uuid_pair,
    }


def _verify_reference_passed_derivation(
    binding_path: Path,
    *,
    expected_selected: Sequence[str],
    expected_passed: Sequence[str],
    candidates: Path,
    manifest: Path,
    selected_allowlist: Path,
    reference_authority: Mapping[str, Any],
    selection_summary: Path | None,
) -> dict[str, dict[str, str]]:
    """Recompute and verify a persisted reference-pass allowlist before liveness."""

    document = _read_json_object(binding_path, "reference-passed binding")
    expected_candidates = {"path": str(candidates.resolve()), "sha256": _sha256_file(candidates)}
    expected_manifest = {"path": str(manifest.resolve()), "sha256": _sha256_file(manifest)}
    expected_selected_pair = {"path": str(selected_allowlist.resolve()), "sha256": _sha256_file(selected_allowlist)}
    if (
        document.get("contract_version") != REFERENCE_PASSED_BINDING_CONTRACT
        or document.get("selection_scope") != "partial"
        or document.get("candidate_rows") != STATIC_ROWS
        or document.get("selected_rows") != len(expected_selected)
        or document.get("reference_passed_rows") != len(expected_passed)
        or document.get("ordering") != "original_candidate_row_index"
        or document.get("candidates") != expected_candidates
        or document.get("manifest") != expected_manifest
        or document.get("selected_allowlist") != expected_selected_pair
        or document.get("reference_authority")
        != {
            "global_contract": reference_authority["global_contract"],
            "global_summary": reference_authority["global_summary"],
        }
    ):
        raise ValueError("reference-passed binding identity mismatch")
    if selection_summary is None:
        raise ValueError("canary reference-passed binding requires a selection summary")
    expected_summary = {"path": str(selection_summary.resolve()), "sha256": _sha256_file(selection_summary)}
    if document.get("selection_summary") != expected_summary:
        raise ValueError("reference-passed binding selection summary mismatch")
    pair = document.get("reference_passed_uuids")
    if not isinstance(pair, Mapping) or set(pair) != {"path", "sha256"} or not isinstance(pair.get("path"), str):
        raise ValueError("reference-passed UUID artifact schema mismatch")
    path = Path(pair["path"])
    if not path.is_file() or pair.get("sha256") != _sha256_file(path):
        raise ValueError("reference-passed UUID artifact hash mismatch")
    if _read_ordered_allowlist(path) != list(expected_passed):
        raise ValueError("reference-passed UUID artifact differs from recomputed reference passes")
    # The binding document and the UUID list are deliberately distinct
    # identities.  Liveness must bind both: the former proves derivation and
    # the latter is the exact replay allowlist.
    return {
        "binding": {"path": str(binding_path.resolve()), "sha256": _sha256_file(binding_path)},
        "uuids": {"path": str(path.resolve()), "sha256": str(pair["sha256"])},
    }


def _verify_liveness_reference_passed_pairs(
    liveness_contract: Mapping[str, Any],
    *,
    allowlist_pair: Mapping[str, str],
    derivation: Mapping[str, Mapping[str, str]],
    label: str,
) -> None:
    """Require the liveness contract to bind both derivation artifacts exactly."""

    if set(derivation) != {"binding", "uuids"}:
        raise ValueError(f"{label} reference-pass derivation schema mismatch")
    binding, uuids = derivation["binding"], derivation["uuids"]
    for name, pair in (("binding", binding), ("uuids", uuids), ("allowlist", allowlist_pair)):
        if (
            not isinstance(pair, Mapping)
            or set(pair) != {"path", "sha256"}
            or not isinstance(pair.get("path"), str)
            or not isinstance(pair.get("sha256"), str)
        ):
            raise ValueError(f"{label} reference-pass {name} pair schema mismatch")
    if (
        liveness_contract.get("reference_passed_binding") != binding
        or liveness_contract.get("reference_passed_uuids") != uuids
        or allowlist_pair != uuids
    ):
        raise ValueError(f"{label} reference-pass derivation binding mismatch")


def verify_canary_reference(args: argparse.Namespace) -> dict[str, Any]:
    """Audit a 240-row reference authority without making it formal evidence.

    The canary shares the canonical 13k candidates and original indices, but
    its UUID allowlist is deliberately partial.  This function never calls
    ``analyze`` and therefore cannot be used to materialize the 10k output.
    """

    from tools.data.synthesize.canary.operator_structure_10k_method import (
        materialize_operator_structure_10k_canary as canary_materializer,
    )

    generator = _load_generator(args.generator_module)
    candidates, manifest_path = args.candidates.resolve(), args.manifest.resolve()
    _, _, static, _ = _static_index(candidates, manifest_path, generator)
    summary = _read_json_object(args.canary_summary.resolve(), "canary selection summary")
    if (
        summary.get("contract") != canary_materializer.CONTRACT
        or summary.get("selected_rows") != canary_materializer.ROWS
    ):
        raise ValueError("canary selection summary contract/count mismatch")
    source = summary.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("canary selection source binding missing")
    _require_artifact_pair(
        source.get("candidates"), {"path": str(candidates), "sha256": _sha256_file(candidates)}, "canary candidates"
    )
    _require_artifact_pair(
        source.get("manifest"), {"path": str(manifest_path), "sha256": _sha256_file(manifest_path)}, "canary manifest"
    )
    generator_pair = source.get("generator")
    if (
        not isinstance(generator_pair, Mapping)
        or set(generator_pair) != {"module", "path", "sha256"}
        or generator_pair.get("module") != generator.__name__
        or generator_pair.get("path") != str(Path(generator.__file__).resolve())
        or generator_pair.get("sha256") != _sha256_file(Path(generator.__file__).resolve())
    ):
        raise ValueError("canary generator source binding mismatch")
    allowlist = args.canary_allowlist.resolve()
    artifact = (
        summary.get("artifacts", {}).get("uuid_allowlist") if isinstance(summary.get("artifacts"), Mapping) else None
    )
    if not isinstance(artifact, Mapping) or artifact.get("sha256") != _sha256_file(allowlist):
        raise ValueError("canary summary/allowlist hash mismatch")
    _, _, records = canary_materializer._load_registry(candidates, manifest_path, generator)
    selected, _ = canary_materializer.select(records)
    expected = [str(record["uuid"]) for record in selected]
    if _read_ordered_allowlist(allowlist) != expected:
        raise ValueError("canary allowlist differs from deterministic frozen selection")
    reference_contract = _read_json_object(
        args.reference_dir.resolve() / "global-reference-contract.json", "partial reference contract"
    )
    if (
        reference_contract.get("contract_version") != REFERENCE_AUTHORITY_CONTRACT
        or reference_contract.get("selection_scope") != "partial"
        or (
            reference_contract.get("machine_count"),
            reference_contract.get("gpus_per_machine"),
            reference_contract.get("global_shard_count"),
        )
        != (1, 8, 8)
    ):
        raise ValueError("canary partial reference contract must be fixed 1x8/8")
    rows, shards = _collect_shards(args.reference_dir.resolve(), int(reference_contract["global_shard_count"]))
    authority = _verify_reference_authority(
        args.reference_dir.resolve(),
        rows,
        static,
        candidates,
        manifest_path,
        allowlist,
        generator,
        expected_uuids=expected,
    )
    indexed = _index_reference(rows, static, _sha256_file(candidates), expected_uuids=expected)
    result: dict[str, Any] = {
        "contract": "operator_structure_10k_canary_reference_audit_v1",
        "review_only": True,
        "training_approved": False,
        "selected_rows": len(indexed),
        "reference_passed": sum(row.get("passed") is True for row in indexed.values()),
        "authority": authority,
        "canary_summary": {
            "path": str(args.canary_summary.resolve()),
            "sha256": _sha256_file(args.canary_summary.resolve()),
        },
        "allowlist": {"path": str(allowlist), "sha256": _sha256_file(allowlist)},
    }
    if args.canary_reference_passed_output is not None:
        result["reference_passed_derivation"] = _materialize_reference_passed(
            args.canary_reference_passed_output.resolve(),
            selected_uuids=expected,
            indexed=indexed,
            static=static,
            candidates=candidates,
            manifest=manifest_path,
            selected_allowlist=allowlist,
            authority=authority,
            selection_summary=args.canary_summary,
        )
    return result


def verify_canary_liveness(args: argparse.Namespace) -> dict[str, Any]:
    """Audit only the 1x8 canary liveness authority against fresh reference passes."""

    from tools.data.synthesize.canary.operator_structure_10k_method import (
        materialize_operator_structure_10k_canary as canary_materializer,
    )

    generator = _load_generator(args.generator_module)
    candidates, manifest_path = args.candidates.resolve(), args.manifest.resolve()
    _, _, static, _ = _static_index(candidates, manifest_path, generator)
    summary = _read_json_object(args.canary_summary.resolve(), "canary selection summary")
    if (
        summary.get("contract") != canary_materializer.CONTRACT
        or summary.get("selected_rows") != canary_materializer.ROWS
    ):
        raise ValueError("canary selection summary contract/count mismatch")
    source = summary.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("canary selection source binding missing")
    _require_artifact_pair(
        source.get("candidates"), {"path": str(candidates), "sha256": _sha256_file(candidates)}, "canary candidates"
    )
    _require_artifact_pair(
        source.get("manifest"), {"path": str(manifest_path), "sha256": _sha256_file(manifest_path)}, "canary manifest"
    )
    generator_pair = source.get("generator")
    if (
        not isinstance(generator_pair, Mapping)
        or set(generator_pair) != {"module", "path", "sha256"}
        or generator_pair.get("module") != generator.__name__
        or generator_pair.get("path") != str(Path(generator.__file__).resolve())
        or generator_pair.get("sha256") != _sha256_file(Path(generator.__file__).resolve())
    ):
        raise ValueError("canary generator source binding mismatch")
    selected_allowlist = args.canary_allowlist.resolve()
    artifact = (
        summary.get("artifacts", {}).get("uuid_allowlist") if isinstance(summary.get("artifacts"), Mapping) else None
    )
    if not isinstance(artifact, Mapping) or artifact.get("sha256") != _sha256_file(selected_allowlist):
        raise ValueError("canary summary/allowlist hash mismatch")
    _, _, registry = canary_materializer._load_registry(candidates, manifest_path, generator)
    expected_selected = [str(record["uuid"]) for record in canary_materializer.select(registry)[0]]
    if _read_ordered_allowlist(selected_allowlist) != expected_selected:
        raise ValueError("canary allowlist differs from deterministic frozen selection")
    reference_contract = _read_json_object(
        args.reference_dir.resolve() / "global-reference-contract.json", "canary reference contract"
    )
    if (
        reference_contract.get("contract_version") != REFERENCE_AUTHORITY_CONTRACT
        or reference_contract.get("selection_scope") != "partial"
        or (
            reference_contract.get("machine_count"),
            reference_contract.get("gpus_per_machine"),
            reference_contract.get("global_shard_count"),
        )
        != (1, 8, 8)
    ):
        raise ValueError("canary reference contract must be fixed partial 1x8/8")
    reference_rows, _ = _collect_shards(args.reference_dir.resolve(), 8)
    reference_authority = _verify_reference_authority(
        args.reference_dir.resolve(),
        reference_rows,
        static,
        candidates,
        manifest_path,
        selected_allowlist,
        generator,
        expected_uuids=expected_selected,
    )
    references = _index_reference(reference_rows, static, _sha256_file(candidates), expected_uuids=expected_selected)
    expected_passed = sorted(
        (uuid for uuid in expected_selected if references[uuid].get("passed") is True),
        key=lambda uuid: static[uuid]["index"],
    )
    reference_passed_derivation = _verify_reference_passed_derivation(
        args.reference_passed_binding.resolve(),
        expected_selected=expected_selected,
        expected_passed=expected_passed,
        candidates=candidates,
        manifest=manifest_path,
        selected_allowlist=selected_allowlist,
        reference_authority=reference_authority,
        selection_summary=args.canary_summary,
    )
    liveness_contract = _read_json_object(
        args.liveness_dir.resolve() / "global-liveness-contract.json", "canary liveness contract"
    )
    liveness_allowlist = liveness_contract.get("allowlist")
    if (
        not isinstance(liveness_allowlist, Mapping)
        or set(liveness_allowlist) != {"path", "sha256"}
        or not isinstance(liveness_allowlist.get("path"), str)
    ):
        raise ValueError("canary liveness allowlist schema mismatch")
    liveness_allowlist_path = Path(liveness_allowlist["path"])
    if (
        not liveness_allowlist_path.is_file()
        or liveness_allowlist.get("sha256") != _sha256_file(liveness_allowlist_path)
        or _read_ordered_allowlist(liveness_allowlist_path) != expected_passed
    ):
        raise ValueError("canary liveness allowlist differs from derived reference passes")
    _verify_liveness_reference_passed_pairs(
        liveness_contract,
        allowlist_pair=liveness_allowlist,
        derivation=reference_passed_derivation,
        label="canary liveness",
    )
    liveness_rows, _ = _collect_shards(args.liveness_dir.resolve(), 8)
    authority = _verify_canary_liveness_authority(
        args.liveness_dir.resolve(),
        liveness_rows,
        static,
        candidates,
        manifest_path,
        liveness_allowlist_path,
        generator,
        expected_uuids=expected_passed,
        reference_passed_binding=reference_passed_derivation["binding"],
    )
    partial_references = {uuid: {"passed": uuid in set(expected_passed)} for uuid in static}
    indexed = _index_liveness(
        liveness_rows,
        static,
        partial_references,
        _sha256_file(candidates),
        _sha256_file(manifest_path),
        _sha256_file(Path(generator.__file__)),
        liveness_allowlist_path,
    )
    return {
        "contract": "operator_structure_10k_canary_liveness_audit_v1",
        "review_only": True,
        "training_approved": False,
        "selected_rows": len(indexed),
        "liveness_passed": sum(record.get("passed") is True for record in indexed.values()),
        "reference_passed": len(expected_passed),
        "reference_authority": reference_authority,
        "liveness_authority": authority,
        "reference_passed_derivation": reference_passed_derivation,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generator-module",
        default="tools.data.synthesize.canary.operator_structure_10k_method.generate_operator_structure_10k",
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--liveness-dir", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--verify-canary-reference", action="store_true")
    mode.add_argument("--verify-canary-liveness", action="store_true")
    parser.add_argument("--canary-summary", type=Path, required=True)
    parser.add_argument("--canary-allowlist", type=Path, required=True)
    parser.add_argument("--canary-reference-passed-output", type=Path)
    parser.add_argument("--reference-passed-binding", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.canary_reference_passed_output is not None and not args.verify_canary_reference:
        raise SystemExit("canary-reference-passed-output requires --verify-canary-reference")
    if args.verify_canary_reference:
        print(json.dumps(verify_canary_reference(args), ensure_ascii=False, sort_keys=True))
        return 0
    if args.liveness_dir is None or args.reference_passed_binding is None:
        raise SystemExit("canary liveness audit requires liveness-dir and reference-passed-binding")
    print(json.dumps(verify_canary_liveness(args), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
