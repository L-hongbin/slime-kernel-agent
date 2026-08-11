#!/usr/bin/env python3
"""Fail-closed finalizer for the 1k KernelBench-gap semantic canary.

This program is deliberately offline: it neither produces candidates nor runs
CUDA.  It only accepts a source-bound, complete static/reference/liveness
evidence family and materializes a review-only partition.
"""

from __future__ import annotations

import argparse
import ast
import collections
import copy
import hashlib
import importlib.util
import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.cleaning.complexity import extract_complexity_features, feature_dict  # noqa: E402
from tools.data.synthesize import validate_train_mode_contract as reference_validator  # noqa: E402
from tools.data.synthesize.kernelbench_gap_method import generate_kernelbench_gap as generator  # noqa: E402
from tools.data.synthesize.kernelbench_gap_method import (  # noqa: E402
    validate_kernelbench_gap_liveness as liveness_validator,
)
from tools.data.synthesize.semantic_operator_method import (  # noqa: E402
    validate_semantic_liveness as shared_runtime_core,
)

ANALYSIS_CONTRACT = "kernelbench_gap_exact_analysis_v1"
MAX_AUTHORIZED_CANDIDATES = 1_000
EXPECTED_FAMILY_QUOTAS = {
    "atomic_low_level": 390,
    "conv_norm_chain": 330,
    "long_single_class": 160,
    "modular_multiclass": 120,
}
ACCEPTED_RUNTIME_STATUS = "kernelgym_reference_and_kernelbench_gap_liveness_passed"
ACCEPTED_GOVERNANCE_STATUS = "kernelbench_gap_review_only_training_not_approved"
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
REQUIRED_ACCEPTED = {
    "atomic_low_level": 350,
    "conv_norm_chain": 300,
    "long_single_class": 140,
    "modular_multiclass": 105,
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


def _require_digest(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _nested(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    result = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not result or not all(isinstance(item, dict) for item in result):
        raise ValueError(f"JSONL must contain nonempty objects:{path}")
    return result


def _collect_shards(directory: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paths = sorted(directory.glob("shard-*-of-*.jsonl"))
    if not paths:
        raise ValueError(f"no shard JSONL found:{directory}")
    rows: list[dict[str, Any]] = []
    hashes: list[dict[str, Any]] = []
    names: set[str] = set()
    for path in paths:
        if path.name in names:
            raise ValueError(f"duplicate shard filename:{path}")
        names.add(path.name)
        shard_rows = _read_jsonl(path)
        rows.extend(shard_rows)
        hashes.append({"path": str(path.resolve()), "sha256": _sha256_file(path), "rows": len(shard_rows)})
    return rows, hashes


def _module_closure(module: Any) -> list[dict[str, str]]:
    """Hash this finalizer's local import closure for reproducible analysis."""

    root_name = str(module.__name__)
    root_file = Path(module.__file__).resolve() if getattr(module, "__file__", None) else None
    pending, seen, paths = [root_name], set(), {}
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        if name == root_name and root_file is not None:
            path = root_file
        else:
            try:
                spec = importlib.util.find_spec(name)
            except (ImportError, ModuleNotFoundError, ValueError):
                continue
            if spec is None or not spec.origin or not spec.origin.endswith(".py"):
                continue
            path = Path(spec.origin).resolve()
        try:
            path.relative_to(_REPO_ROOT)
        except ValueError:
            continue
        paths[path] = name
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            raise ValueError(f"cannot parse source closure:{path}") from exc
        package = name.rpartition(".")[0]
        for node in ast.walk(tree):
            candidates: list[str] = []
            if isinstance(node, ast.Import):
                candidates = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                target = "." * node.level + (node.module or "")
                try:
                    base = importlib.util.resolve_name(target, package) if node.level else (node.module or "")
                except (ImportError, ValueError):
                    base = ""
                if base:
                    candidates.append(base)
                    if node.module:
                        candidates.extend(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
            for candidate in candidates:
                try:
                    imported = importlib.util.find_spec(candidate)
                except (ImportError, ModuleNotFoundError, ValueError):
                    continue
                if not imported or not imported.origin or not imported.origin.endswith(".py"):
                    continue
                try:
                    Path(imported.origin).resolve().relative_to(_REPO_ROOT)
                except ValueError:
                    continue
                pending.append(candidate)
    return [{"module": paths[path], "path": str(path), "sha256": _sha256_file(path)} for path in sorted(paths)]


def _family_template_quotas() -> dict[str, int]:
    quotas: dict[str, int] = {}
    for item in generator.TEMPLATES:
        template_id, variants = getattr(item, "template_id", None), getattr(item, "variants", None)
        if not isinstance(template_id, str) or type(variants) is not int or variants <= 0:
            raise ValueError("generator TEMPLATES must expose template_id and positive variants")
        if template_id in quotas:
            raise ValueError(f"duplicate generator template:{template_id}")
        quotas[template_id] = variants
    if not quotas or sum(quotas.values()) != MAX_AUTHORIZED_CANDIDATES:
        raise ValueError(f"template quotas must sum to 1000:{quotas}")
    return quotas


def _replay_manifest(
    manifest: Mapping[str, Any]
) -> tuple[str, list[Any], Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    """Use the generator's replay helper, never a hand-maintained duplicate registry."""

    replay = getattr(generator, "replay_manifest", None)
    if not callable(replay):
        raise ValueError("generator must export replay_manifest(manifest) for final static replay")
    result = replay(manifest)
    if not isinstance(result, Mapping):
        raise ValueError(f"generator replay result must be a mapping:{manifest.get('uuid')}")
    code = result.get("code")
    declared_ops = result.get("declared_ops")
    labels = result.get("coverage_labels")
    static_proof = result.get("static_proof")
    spec = result.get("spec")
    if (
        not isinstance(code, str)
        or not isinstance(declared_ops, list)
        or not isinstance(labels, Mapping)
        or not isinstance(static_proof, Mapping)
        or not isinstance(spec, Mapping)
    ):
        raise ValueError(f"generator replay result schema invalid:{manifest.get('uuid')}")
    return code, declared_ops, labels, static_proof, spec


def _manifest_contract_valid(manifest: Mapping[str, Any], generator_sha: str, index: int) -> None:
    expected = {
        "manifest_contract_version": getattr(generator, "MANIFEST_VERSION", None),
        "primary_intervention": "semantic_operator",
        "lineage_kind": "standalone_semantic_synthetic",
        "method": "kernelbench_gap_canary",
        "parent_uuid": None,
        "training_approved": False,
        "structured_output_deferred": True,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"manifest contract mismatch:{index}:{field}")
    for field, constant in (
        ("generator_contract_version", "CONTRACT_VERSION"),
        ("registry_version", "REGISTRY_VERSION"),
        ("static_contract_version", "STATIC_CONTRACT_VERSION"),
        ("runtime_contract_version", "RUNTIME_CONTRACT_VERSION"),
    ):
        if hasattr(generator, constant) and manifest.get(field) != getattr(generator, constant):
            raise ValueError(f"manifest generator contract mismatch:{index}:{field}")
    if manifest.get("generator_source_sha256") != generator_sha:
        raise ValueError(f"manifest generator source differs from current source:{index}")
    output = manifest.get("final_output_contract")
    if output not in (
        {"kind": "single_tensor", "finite_required": True},
        {"kind": "single_dense_tensor", "finite_required": True},
    ):
        raise ValueError(f"manifest final output contract invalid:{index}")


def _verify_static(
    candidates_path: Path, manifest_path: Path
) -> tuple[pa.Table, list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    table = pq.read_table(candidates_path)
    rows, manifests = table.to_pylist(), _read_jsonl(manifest_path)
    if (
        getattr(generator, "EXACT_CANARY_ROWS", None) != MAX_AUTHORIZED_CANDIDATES
        or getattr(generator, "MAX_AUTHORIZED_CANDIDATES", None) != MAX_AUTHORIZED_CANDIDATES
        or dict(getattr(generator, "FAMILY_QUOTAS", {})) != EXPECTED_FAMILY_QUOTAS
    ):
        raise ValueError("generator does not preserve the fixed 1k/family-quota contract")
    if len(rows) != MAX_AUTHORIZED_CANDIDATES or len(manifests) != MAX_AUTHORIZED_CANDIDATES:
        raise ValueError(f"static canary must contain exactly 1000 rows:{len(rows)}:{len(manifests)}")
    template_quotas = _family_template_quotas()
    generator_sha = _sha256_file(Path(generator.__file__).resolve())
    seen: dict[str, set[str]] = {
        field: set() for field in ("uuid", "reference_sha256", "normalized_ast_sha256", "row_payload_sha256")
    }
    by_uuid: dict[str, dict[str, Any]] = {}
    family_counts: collections.Counter[str] = collections.Counter()
    template_counts: collections.Counter[str] = collections.Counter()
    skeleton_counts: collections.Counter[str] = collections.Counter()
    template_bindings, root_bindings, commits = set(), set(), set()
    for index, (row, manifest) in enumerate(zip(rows, manifests, strict=True)):
        uuid, code, prompt = (
            _nested(row, "extra_info.uuid"),
            _nested(row, "reward_model.ground_truth"),
            row.get("prompt"),
        )
        if manifest.get("candidate_row_index") != index or manifest.get("uuid") != uuid:
            raise ValueError(f"static row/manifest identity mismatch:{index}")
        _manifest_contract_valid(manifest, generator_sha, index)
        if not isinstance(uuid, str) or not isinstance(code, str) or not isinstance(prompt, list) or len(prompt) != 1:
            raise ValueError(f"candidate schema invalid:{index}")
        content = prompt[0].get("content") if isinstance(prompt[0], Mapping) else None
        if not isinstance(content, str) or not content.endswith(code):
            raise ValueError(f"prompt does not end in exact reference:{index}")
        if (
            manifest.get("reference_sha256") != _sha256_bytes(code.encode())
            or manifest.get("prompt_sha256") != _sha256_bytes(content.encode())
            or manifest.get("row_payload_sha256") != _canonical_sha256(row)
            or manifest.get("normalized_ast_sha256") != generator._normalized_ast_sha256(code)
        ):
            raise ValueError(f"static candidate hash mismatch:{index}")
        expected_code, expected_ops, expected_labels, expected_proof, expected_spec = _replay_manifest(manifest)
        if (
            code != expected_code
            or manifest.get("declared_ops") != expected_ops
            or manifest.get("coverage_labels") != expected_labels
            or manifest.get("static_proof") != expected_proof
            # The registry target is intentionally a subset for composed
            # templates; `required_signature` records the complete realized
            # signature so final analysis can measure what was actually
            # constructed.  Bind both representations independently.
            or manifest.get("template_required_signature") != expected_spec.get("required_signature")
            or manifest.get("required_signature") != expected_proof.get("realized_signature")
            or manifest.get("required_structure") != expected_spec.get("required_structure")
        ):
            raise ValueError(f"closed registry/static replay mismatch:{index}")
        static_again = generator._static_contract(code, expected_ops)
        if static_again != expected_proof:
            raise ValueError(f"static contract replay helper mismatch:{index}")
        if manifest.get("get_inputs_skeleton_sha256") != expected_proof.get("get_inputs_skeleton_sha256"):
            raise ValueError(f"get_inputs skeleton hash replay mismatch:{index}")
        if manifest.get("mode_behavior") not in {"stateless", "train_stateful", "recurrent_state"}:
            raise ValueError(f"mode behavior missing/invalid:{index}")
        skeleton_id = manifest.get("skeleton_id")
        skeleton = manifest.get("template_skeleton")
        if (
            not isinstance(skeleton_id, str)
            or not skeleton_id
            or not isinstance(skeleton, (str, Mapping, list))
            or skeleton_id != expected_spec.get("skeleton_id")
            or skeleton != expected_spec.get("template_skeleton")
            or skeleton_id != expected_labels.get("skeleton_id")
            or skeleton != expected_labels.get("template_skeleton")
        ):
            raise ValueError(f"template skeleton replay mismatch:{index}")
        input_skeleton = manifest.get("get_inputs_skeleton_sha256")
        _require_digest(input_skeleton, f"get_inputs skeleton:{index}")
        for field, values in seen.items():
            value = manifest.get(field)
            if not isinstance(value, str) or value in values:
                raise ValueError(f"static {field} is missing or duplicated:{index}")
            values.add(value)
        family, template = manifest.get("primary_family"), manifest.get("template_id")
        if family not in EXPECTED_FAMILY_QUOTAS or template not in template_quotas:
            raise ValueError(f"unknown family/template:{index}:{family}:{template}")
        family_counts[str(family)] += 1
        template_counts[str(template)] += 1
        skeleton_counts[skeleton_id] += 1
        template_bindings.add(_canonical_json(manifest.get("prompt_template")))
        root_bindings.add(_canonical_json(manifest.get("decontamination_roots")))
        commit = manifest.get("git_commit_at_generation")
        if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
            raise ValueError(f"invalid generator commit binding:{index}")
        commits.add(commit)
        by_uuid[uuid] = {"row_index": index, "row": row, "manifest": manifest}
    if dict(family_counts) != EXPECTED_FAMILY_QUOTAS:
        raise ValueError(f"family quotas differ:{dict(family_counts)}")
    if dict(template_counts) != template_quotas:
        raise ValueError(f"template quotas differ:{dict(template_counts)}")
    if not skeleton_counts or len(commits) != 1 or len(template_bindings) != 1 or len(root_bindings) != 1:
        raise ValueError("mixed/empty static skeleton, generator, template, or root bindings")
    template_binding = json.loads(next(iter(template_bindings)))
    if not isinstance(template_binding, Mapping) or not isinstance(template_binding.get("path"), str):
        raise ValueError("prompt template binding missing")
    template_path = Path(template_binding["path"])
    if not template_path.is_file() or _sha256_file(template_path) != template_binding.get("sha256"):
        raise ValueError("prompt template source drift")
    roots = json.loads(next(iter(root_bindings)))
    if not isinstance(roots, list) or not roots:
        raise ValueError("decontamination root binding missing")
    for root in roots:
        path = Path(root.get("path", "")) if isinstance(root, Mapping) else None
        if (
            path is None
            or not path.is_file()
            or _sha256_file(path) != root.get("sha256")
            or pq.ParquetFile(path).metadata.num_rows != root.get("rows")
        ):
            raise ValueError(f"decontamination root drift:{path}")
    static_summary = {
        "generator_source_sha256": generator_sha,
        "generator_commit": next(iter(commits)),
        "family_counts": dict(sorted(family_counts.items())),
        "template_counts": dict(sorted(template_counts.items())),
        "skeleton_counts": dict(sorted(skeleton_counts.items())),
        "get_inputs_skeleton_unique": len({item["get_inputs_skeleton_sha256"] for item in manifests}),
    }
    return table, manifests, by_uuid, static_summary


def _memory_guard_valid(value: Any) -> bool:
    return not reference_validator.validate_memory_guard_evidence(value)


def _gpu_authorized(record: Mapping[str, Any]) -> bool:
    gpu = record.get("gpu")
    return isinstance(gpu, Mapping) and any(name in str(gpu.get("name")) for name in ("A800", "H20"))


def _verify_reference(
    records: Sequence[Mapping[str, Any]], by_uuid: Mapping[str, Mapping[str, Any]], candidates_path: Path
) -> tuple[dict[str, Mapping[str, Any]], dict[str, str], dict[str, str]]:
    if len(records) != MAX_AUTHORIZED_CANDIDATES:
        raise ValueError(f"reference must contain exactly 1000 records:{len(records)}")
    candidate_sha = _sha256_file(candidates_path)
    validator_sha = _sha256_file(Path(reference_validator.__file__).resolve())
    indexed: dict[str, Mapping[str, Any]] = {}
    failures: dict[str, str] = {}
    bindings, launchers = set(), set()
    for record in records:
        uuid = record.get("uuid")
        if not isinstance(uuid, str) or uuid not in by_uuid or uuid in indexed:
            raise ValueError(f"reference UUID unknown/duplicate:{uuid!r}")
        static = by_uuid[uuid]
        manifest = static["manifest"]
        if (
            record.get("row_index") != static["row_index"]
            or record.get("reference_sha256") != manifest["reference_sha256"]
            or record.get("contract_version") != REFERENCE_CONTRACT
            or record.get("source_sha256") != candidate_sha
            or record.get("validator_source_sha256") != validator_sha
            or record.get("trials") != 5
            or record.get("seed") != 42
            or record.get("training") is not True
            or record.get("device") != "cuda:0"
            or record.get("max_device_memory_gib") != 64.0
        ):
            raise ValueError(f"reference identity/source/policy mismatch:{uuid}")
        payload = record.get("contract_payload")
        if not isinstance(payload, Mapping) or payload.get("expected_mode_class") is not None:
            raise ValueError(f"reference must use the standalone expected-mode-class=any policy:{uuid}")
        kernelgym = record.get("kernelgym")
        if not isinstance(kernelgym, Mapping) or kernelgym.get("git_commit") != EXPECTED_KERNELGYM_COMMIT:
            raise ValueError(f"reference KernelGym commit mismatch:{uuid}")
        if any(kernelgym.get(key) != value for key, value in EXPECTED_KERNELGYM_HASHES.items()):
            raise ValueError(f"reference KernelGym source bundle mismatch:{uuid}")
        if not _gpu_authorized(record) or not _memory_guard_valid(record.get("memory_guard")):
            raise ValueError(f"reference GPU/memory guard invalid:{uuid}")
        fingerprint, launcher = record.get("contract_fingerprint"), record.get("launcher_source_sha256")
        _require_digest(fingerprint, f"reference contract fingerprint:{uuid}")
        _require_digest(launcher, f"reference launcher source:{uuid}")
        bindings.add(fingerprint)
        launchers.add(launcher)
        passed = record.get("passed")
        if type(passed) is not bool:
            raise ValueError(f"reference passed flag invalid:{uuid}")
        if passed:
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
            failures[uuid] = _failure_reason(record)
        indexed[uuid] = record
    if set(indexed) != set(by_uuid) or len(bindings) != 1 or len(launchers) != 1:
        raise ValueError("reference shards mix bindings or do not exactly cover static rows")
    return (
        indexed,
        failures,
        {"contract_fingerprint": next(iter(bindings)), "launcher_source_sha256": next(iter(launchers))},
    )


def _failure_reason(record: Mapping[str, Any]) -> str:
    reason = record.get("failure_signature") or record.get("reason") or record.get("error") or record.get("status")
    return _canonical_json(reason) if isinstance(reason, (Mapping, list)) else str(reason)


def _source_binding_value(
    binding: Mapping[str, Any], name: str, expected_path: Path, expected_sha: str, uuid: str
) -> None:
    value = binding.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"liveness binding evidence missing {name}:{uuid}")
    path, digest = value.get("path"), value.get("sha256")
    if not isinstance(path, str) or Path(path).resolve() != expected_path.resolve() or digest != expected_sha:
        raise ValueError(f"liveness binding evidence drift:{name}:{uuid}")


def _read_allowlist(path: Path) -> list[str]:
    values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not values or len(values) != len(set(values)):
        raise ValueError(f"liveness allowlist empty/duplicated:{path}")
    return values


def _verify_liveness(
    records: Sequence[Mapping[str, Any]],
    by_uuid: Mapping[str, Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
    candidates_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, str], dict[str, str]]:
    reference_passed = [
        uuid
        for uuid, static in sorted(by_uuid.items(), key=lambda item: item[1]["row_index"])
        if references[uuid].get("passed") is True
    ]
    if len(records) != len(reference_passed):
        raise ValueError(
            f"liveness count must equal fresh reference-pass count:{len(records)}:{len(reference_passed)}"
        )
    candidate_sha, manifest_sha = _sha256_file(candidates_path), _sha256_file(manifest_path)
    adapter_path, core_path, generator_path = (
        Path(liveness_validator.__file__).resolve(),
        Path(shared_runtime_core.__file__).resolve(),
        Path(generator.__file__).resolve(),
    )
    adapter_sha, core_sha, generator_sha = (
        _sha256_file(adapter_path),
        _sha256_file(core_path),
        _sha256_file(generator_path),
    )
    if getattr(liveness_validator, "CONTRACT_VERSION", None) != "kernelbench_gap_runtime_liveness_v1":
        raise ValueError("unexpected kernelbench-gap liveness contract")
    indexed: dict[str, Mapping[str, Any]] = {}
    failures: dict[str, str] = {}
    bindings, launchers, allow_paths, allow_hashes = set(), set(), set(), set()
    for record in records:
        uuid = record.get("uuid")
        if not isinstance(uuid, str) or uuid not in reference_passed or uuid in indexed:
            raise ValueError(f"liveness UUID unknown/duplicate/not reference-passed:{uuid!r}")
        static, manifest = by_uuid[uuid], by_uuid[uuid]["manifest"]
        if (
            record.get("contract_version") != liveness_validator.CONTRACT_VERSION
            or record.get("raw_runtime_core_contract_version") != shared_runtime_core.CONTRACT_VERSION
            or record.get("candidate_row_index") != static["row_index"]
            or record.get("primary_family") != manifest["primary_family"]
            or record.get("template_id") != manifest["template_id"]
            or record.get("candidates_sha256") != candidate_sha
            or record.get("manifest_sha256") != manifest_sha
            or record.get("adapter_source_sha256") != adapter_sha
            or record.get("shared_runtime_core_source_sha256") != core_sha
            or record.get("generator_source_sha256") != generator_sha
        ):
            raise ValueError(f"liveness identity/source mismatch:{uuid}")
        config = record.get("validation_config")
        if (
            not isinstance(config, Mapping)
            or config.get("device") != "cuda:0"
            or config.get("trials") != 3
            or config.get("seed") != 17
            or config.get("max_device_memory_gib") != 64.0
            or config.get("persistent_train_mode_models") is not True
            or config.get("single_tensor_final_output") is not True
            or config.get("execution_controls") != shared_runtime_core.EXECUTION_CONTROLS
        ):
            raise ValueError(f"liveness policy mismatch:{uuid}")
        binding = record.get("binding_evidence")
        if not isinstance(binding, Mapping):
            raise ValueError(f"liveness binding evidence missing:{uuid}")
        _source_binding_value(binding, "adapter_source", adapter_path, adapter_sha, uuid)
        _source_binding_value(binding, "shared_runtime_core_source", core_path, core_sha, uuid)
        _source_binding_value(binding, "generator_source", generator_path, generator_sha, uuid)
        launcher_binding = binding.get("launcher_source")
        if not isinstance(launcher_binding, Mapping):
            raise ValueError(f"liveness launcher binding missing:{uuid}")
        launcher_path, launcher_sha = launcher_binding.get("path"), launcher_binding.get("sha256")
        if (
            not isinstance(launcher_path, str)
            or not Path(launcher_path).is_file()
            or _sha256_file(Path(launcher_path)) != launcher_sha
        ):
            raise ValueError(f"liveness launcher source drift:{uuid}")
        if record.get("launcher_source_sha256") != launcher_sha:
            raise ValueError(f"liveness launcher record/binding mismatch:{uuid}")
        allow_path, allow_sha = binding.get("allowlist_path"), binding.get("allowlist_sha256")
        if not isinstance(allow_path, str) or not isinstance(allow_sha, str):
            raise ValueError(f"liveness allowlist binding missing:{uuid}")
        _require_digest(record.get("validation_binding_sha256"), f"liveness binding:{uuid}")
        _require_digest(allow_sha, f"liveness allowlist hash:{uuid}")
        bindings.add(record["validation_binding_sha256"])
        launchers.add(launcher_sha)
        allow_paths.add(allow_path)
        allow_hashes.add(allow_sha)
        passed = record.get("passed")
        if type(passed) is not bool:
            raise ValueError(f"liveness passed flag invalid:{uuid}")
        if passed:
            _verify_liveness_pass(record, manifest)
        else:
            if not record.get("failure_stage") or not record.get("failure_signature"):
                raise ValueError(f"liveness failure attribution missing:{uuid}")
            failures[uuid] = _failure_reason(record)
        indexed[uuid] = record
    if (
        set(indexed) != set(reference_passed)
        or len(bindings) != 1
        or len(launchers) != 1
        or len(allow_paths) != 1
        or len(allow_hashes) != 1
    ):
        raise ValueError("liveness shards mix bindings or do not exactly cover the reference pass set")
    allowlist_path = Path(next(iter(allow_paths))).resolve()
    if not allowlist_path.is_file() or _sha256_file(allowlist_path) != next(iter(allow_hashes)):
        raise ValueError("liveness allowlist source drift")
    if _read_allowlist(allowlist_path) != reference_passed:
        raise ValueError("liveness allowlist differs from fresh ordered reference-pass UUIDs")
    return (
        indexed,
        failures,
        {
            "validation_binding_sha256": next(iter(bindings)),
            "launcher_source_sha256": next(iter(launchers)),
            "allowlist_path": str(allowlist_path),
            "allowlist_sha256": next(iter(allow_hashes)),
        },
    )


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}:{path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object:{path}")
    return value


def _verify_liveness_launcher_evidence(
    liveness_dir: Path,
    *,
    liveness_rows: Sequence[Mapping[str, Any]],
    liveness: Mapping[str, Mapping[str, Any]],
    liveness_binding: Mapping[str, str],
    by_uuid: Mapping[str, Mapping[str, Any]],
    reference_passed: set[str],
    candidates_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    """Bind formal analysis to the required eight-shard launcher evidence."""

    shard_count = 8
    expected_names = {f"shard-{index:02d}-of-{shard_count:02d}.jsonl" for index in range(shard_count)}
    paths = {path.name: path for path in liveness_dir.glob("shard-*-of-*.jsonl")}
    if set(paths) != expected_names:
        raise ValueError(f"liveness shard filenames must be exactly eight launcher shards:{sorted(paths)}")
    scheduler_path = liveness_dir / "scheduler-contract.json"
    summary_path = liveness_dir / "launcher-summary.json"
    if not scheduler_path.is_file() or not summary_path.is_file():
        raise ValueError("formal liveness evidence requires scheduler-contract.json and launcher-summary.json")

    scheduler = _read_json_object(scheduler_path, "liveness scheduler contract")
    summary = _read_json_object(summary_path, "liveness launcher summary")
    candidate_sha, manifest_sha = _sha256_file(candidates_path), _sha256_file(manifest_path)
    sample = next(iter(liveness.values()))
    binding = sample.get("binding_evidence")
    if not isinstance(binding, Mapping):
        raise ValueError("liveness records lack binding evidence")
    expected_source = binding.get("launcher_source")
    if not isinstance(expected_source, Mapping):
        raise ValueError("liveness records lack launcher source binding")
    raw_sources = {
        "adapter_source": binding.get("adapter_source"),
        "shared_runtime_core_source": binding.get("shared_runtime_core_source"),
        "generator_source": binding.get("generator_source"),
        "launcher_source": expected_source,
    }
    if not all(isinstance(value, Mapping) for value in raw_sources.values()):
        raise ValueError("liveness records have incomplete source binding")
    expected_sources = {
        name: {"path": value.get("path"), "sha256": value.get("sha256")} for name, value in raw_sources.items()
    }
    for label, document, contract in (
        ("scheduler", scheduler, "kernelbench_gap_liveness_scheduler_v1"),
        ("launcher summary", summary, "kernelbench_gap_liveness_launcher_summary_v1"),
    ):
        if document.get("contract_version") != contract or document.get("shard_count") != shard_count:
            raise ValueError(f"liveness {label} contract/shard count mismatch")
        if document.get("source_binding") != expected_sources:
            raise ValueError(f"liveness {label} source binding mismatch")
    if scheduler.get("candidates") != {"path": str(candidates_path), "sha256": candidate_sha}:
        raise ValueError("liveness scheduler candidate binding mismatch")
    if scheduler.get("manifest") != {"path": str(manifest_path), "sha256": manifest_sha}:
        raise ValueError("liveness scheduler manifest binding mismatch")
    allowlist = {"path": liveness_binding["allowlist_path"], "sha256": liveness_binding["allowlist_sha256"]}
    if scheduler.get("allowlist") != allowlist:
        raise ValueError("liveness scheduler allowlist binding mismatch")
    if scheduler.get("trials") != 3 or scheduler.get("seed") != 17:
        raise ValueError("liveness scheduler trial/seed mismatch")
    if summary.get("validation_binding_sha256") != liveness_binding["validation_binding_sha256"]:
        raise ValueError("liveness launcher summary validation binding mismatch")

    by_shard: dict[int, list[Mapping[str, Any]]] = {index: [] for index in range(shard_count)}
    for record in liveness_rows:
        uuid = record.get("uuid")
        if not isinstance(uuid, str) or uuid not in by_uuid:
            raise ValueError(f"liveness row has unknown UUID while checking launcher shards:{uuid!r}")
        index = int(by_uuid[uuid]["row_index"])
        by_shard[index % shard_count].append(record)
    expected_uuid_sets = {
        index: {uuid for uuid in reference_passed if int(by_uuid[uuid]["row_index"]) % shard_count == index}
        for index in range(shard_count)
    }
    for index in range(shard_count):
        path = paths[f"shard-{index:02d}-of-{shard_count:02d}.jsonl"]
        records = _read_jsonl(path)
        observed = {str(record.get("uuid")) for record in records}
        if len(observed) != len(records) or observed != expected_uuid_sets[index]:
            raise ValueError(f"liveness shard UUID partition mismatch:{index}")
        if any(int(by_uuid[str(record["uuid"])]["row_index"]) % shard_count != index for record in records):
            raise ValueError(f"liveness shard row-index ownership mismatch:{index}")
        if {_canonical_sha256(record) for record in records} != {
            _canonical_sha256(record) for record in by_shard[index]
        }:
            raise ValueError(f"liveness shard/raw-record binding mismatch:{index}")
    shards = summary.get("shards")
    if not isinstance(shards, list) or len(shards) != shard_count:
        raise ValueError("liveness launcher summary must contain eight shard summaries")
    per_shard = {item.get("shard_index"): item for item in shards if isinstance(item, Mapping)}
    if set(per_shard) != set(range(shard_count)):
        raise ValueError("liveness launcher summary shard indexes mismatch")
    for index, shard in per_shard.items():
        rows = by_shard[index]
        passed = sum(record.get("passed") is True for record in rows)
        if (
            shard.get("shard_count") != shard_count
            or shard.get("selected") != len(rows)
            or shard.get("passed") != passed
            or shard.get("failed") != len(rows) - passed
            or shard.get("executed", -1) + shard.get("resumed", -1) != len(rows)
        ):
            raise ValueError(f"liveness launcher summary shard accounting mismatch:{index}")
    if (
        summary.get("selected") != len(reference_passed)
        or summary.get("passed") != sum(record.get("passed") is True for record in liveness.values())
        or summary.get("failed") != len(reference_passed) - summary.get("passed")
        or summary.get("executed", -1) + summary.get("resumed", -1) != len(reference_passed)
    ):
        raise ValueError("liveness launcher summary aggregate accounting mismatch")
    return {
        "status": "bound_eight_shard_launcher",
        "scheduler_contract": {"path": str(scheduler_path), "sha256": _sha256_file(scheduler_path)},
        "launcher_summary": {"path": str(summary_path), "sha256": _sha256_file(summary_path)},
        "shard_count": shard_count,
        "shards": [
            {
                "path": str(paths[f"shard-{index:02d}-of-{shard_count:02d}.jsonl"]),
                "sha256": _sha256_file(paths[f"shard-{index:02d}-of-{shard_count:02d}.jsonl"]),
                "rows": len(by_shard[index]),
            }
            for index in range(shard_count)
        ],
    }


def _verify_liveness_pass(record: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    if (
        record.get("status") != "passed"
        or record.get("persistent_model_instances") is not True
        or record.get("training_mode_preserved") is not True
        or record.get("final_output_kind") not in ("single_tensor", "single_dense_tensor")
        or record.get("execution_controls") != shared_runtime_core.EXECUTION_CONTROLS
        or not _gpu_authorized(record)
        or not _memory_guard_valid(record.get("memory_guard"))
    ):
        raise ValueError(f"liveness passing summary incomplete:{record.get('uuid')}")
    gpu = record.get("gpu")
    if not isinstance(gpu, Mapping) or gpu.get("device") != "cuda:0":
        raise ValueError(f"liveness CUDA device evidence invalid:{record.get('uuid')}")
    expected_mutation = manifest.get("mode_behavior") == "train_stateful"
    if (
        record.get("mode_behavior") != manifest.get("mode_behavior")
        or record.get("state_mutation_expected") is not expected_mutation
        or (expected_mutation and record.get("state_mutation_observed") is not True)
        or (not expected_mutation and record.get("state_mutation_observed") is not False)
    ):
        raise ValueError(f"liveness state-mode evidence invalid:{record.get('uuid')}")
    aliases = record.get("registered_object_alias_evidence")
    if not isinstance(aliases, Mapping) or set(aliases) != {"module_attribute_paths", "tensor_attribute_paths"}:
        raise ValueError(f"liveness registered-object alias evidence invalid:{record.get('uuid')}")
    for values in aliases.values():
        if not isinstance(values, list) or not all(isinstance(value, str) and value for value in values):
            raise ValueError(f"liveness registered-object alias paths invalid:{record.get('uuid')}")
    trials = record.get("trials")
    if not isinstance(trials, list) or len(trials) != 3:
        raise ValueError(f"liveness trial count invalid:{record.get('uuid')}")
    expected_ops = {
        item["op_id"]: {
            "minimum_calls": item["min_calls_per_trial"],
            "identities": {
                f"{identity['schema']}.{identity.get('overload') or '<default>'}"
                for identity in item["runtime_identities"]
            },
        }
        for item in manifest["declared_ops"]
        if isinstance(item, Mapping)
    }
    if not expected_ops:
        raise ValueError(f"manifest declared ops missing:{record.get('uuid')}")
    for ordinal, trial in enumerate(trials):
        trace = trial.get("trace") if isinstance(trial, Mapping) else None
        if (
            not isinstance(trace, Mapping)
            or trial.get("ordinal") != ordinal
            or trial.get("seed") != 17 + ordinal * 10_007
            or trial.get("single_tensor_output") is not True
            or trial.get("output_finite") is not True
            or trial.get("control_trace_output_exact") is not True
            or trial.get("control_trace_state_exact") is not True
            or trial.get("inputs_immutable") is not True
            or trace.get("final_output_declared_op_ids") != sorted(expected_ops)
        ):
            raise ValueError(f"liveness trial contract invalid:{record.get('uuid')}:{ordinal}")
        per_op = trace.get("per_declared_op")
        if (
            not isinstance(per_op, list)
            or len(per_op) != len(expected_ops)
            or {item.get("op_id") for item in per_op if isinstance(item, Mapping)} != set(expected_ops)
        ):
            raise ValueError(f"liveness op provenance mismatch:{record.get('uuid')}:{ordinal}")
        for evidence in per_op:
            op_id = evidence.get("op_id") if isinstance(evidence, Mapping) else None
            matches = evidence.get("matched_identities") if isinstance(evidence, Mapping) else None
            if (
                not isinstance(evidence, Mapping)
                or evidence.get("returned_output_witness") is not True
                or type(evidence.get("calls")) is not int
                or evidence["calls"] < expected_ops[op_id]["minimum_calls"]
                or evidence.get("minimum_calls") != expected_ops[op_id]["minimum_calls"]
                or not isinstance(matches, Mapping)
                or not matches
                or not set(matches).issubset(expected_ops[op_id]["identities"])
                or not all(type(count) is int and count > 0 for count in matches.values())
                or sum(matches.values()) != evidence["calls"]
            ):
                raise ValueError(f"liveness op witness invalid:{record.get('uuid')}:{ordinal}")


def _operator_family_flags(manifest: Mapping[str, Any]) -> set[str]:
    raw = manifest.get("coverage_labels")
    labels = raw.get("operator_families", []) if isinstance(raw, Mapping) else []
    if isinstance(labels, str):
        labels = [labels]
    flags = {str(label).lower() for label in labels} if isinstance(labels, list) else set()
    declared = " ".join(
        str(item.get("op_id", "")) for item in manifest.get("declared_ops", []) if isinstance(item, Mapping)
    ).lower()
    checks = {
        "convolution": ("conv",),
        "normalization": ("norm", "rms"),
        "matmul_linear": ("matmul", "linear", ".mm", "bmm"),
        "indexing_scatter": ("index", "gather", "scatter", "topk", "take"),
        "loss_distance": ("loss", "distance", "mse", "cross_entropy"),
    }
    return {family for family, needles in checks.items() if family in flags or any(key in declared for key in needles)}


def _code_metrics(code: str, manifest: Mapping[str, Any]) -> dict[str, Any]:
    tree = ast.parse(code)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    # These are the exact complexity-feature names used to state difference 3
    # in the generator's own structural contracts.  Do not substitute the
    # number of declared ATen groups for forward-call complexity.
    features = feature_dict(extract_complexity_features(code))
    return {
        "operator_count": int(features.get("forward_call_count", len(manifest["declared_ops"]))),
        "source_lines": int(features.get("source_line_count", len(code.splitlines()))),
        "registered_module_constructors": int(features.get("init_nn_constructor_count", 0)),
        "top_level_class_count": int(features.get("top_level_class_count", len(classes))),
        "input_skeleton": manifest["get_inputs_skeleton_sha256"],
        "operator_families": sorted(_operator_family_flags(manifest)),
    }


def _distribution_metrics(manifests: Iterable[Mapping[str, Any]], accepted: set[str] | None = None) -> dict[str, Any]:
    selected = [item for item in manifests if accepted is None or item["uuid"] in accepted]
    metrics = [_code_metrics(str(item["_code"]), item) for item in selected]
    count = len(metrics)

    def fractions(predicate: Any) -> float:
        return 0.0 if count == 0 else round(100.0 * sum(predicate(item) for item in metrics) / count, 4)

    family_counts: collections.Counter[str] = collections.Counter()
    for item in metrics:
        family_counts.update(item["operator_families"])
    templates = collections.Counter(item["input_skeleton"] for item in metrics)
    return {
        "sample_count": count,
        "difference_3_structure_pct": {
            "input_related_operator_count_le_1": fractions(lambda item: item["operator_count"] <= 1),
            "input_related_operator_count_ge_10": fractions(lambda item: item["operator_count"] >= 10),
            "source_lines_ge_50": fractions(lambda item: item["source_lines"] >= 50),
            "registered_module_constructors_ge_5": fractions(lambda item: item["registered_module_constructors"] >= 5),
            "multiple_top_level_classes": fractions(lambda item: item["top_level_class_count"] > 1),
        },
        "difference_4_operator_family_pct": {
            family: 0.0 if count == 0 else round(100.0 * family_counts[family] / count, 4)
            for family in ("convolution", "normalization", "matmul_linear", "indexing_scatter", "loss_distance")
        },
        "difference_5_input_factory_templates": {
            "unique": len(templates),
            "max_reuse": max(templates.values(), default=0),
            "duplicate_samples": sum(value for value in templates.values() if value > 1),
            "counts": dict(sorted(templates.items())),
        },
    }


def _failure_audit(
    manifests: Sequence[Mapping[str, Any]],
    references: Mapping[str, Mapping[str, Any]],
    liveness: Mapping[str, Mapping[str, Any]],
    accepted: set[str],
) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    for manifest in manifests:
        uuid = str(manifest["uuid"])
        if uuid in accepted:
            continue
        live = liveness.get(uuid)
        raw = live if live is not None and live.get("passed") is False else references[uuid]
        audit.append(
            {
                "uuid": uuid,
                "row_index": manifest["candidate_row_index"],
                "family": manifest["primary_family"],
                "template_id": manifest["template_id"],
                "skeleton_id": manifest["skeleton_id"],
                "failure_stage": raw.get("failure_stage") or ("liveness" if live is raw else "reference"),
                "failure_signature": raw.get("failure_signature")
                or _canonical_sha256({"reason": _failure_reason(raw)}),
                "reason": _failure_reason(raw),
                "reference_record_sha256": _canonical_sha256(references[uuid]),
                "liveness_record_sha256": _canonical_sha256(live) if live is not None else None,
            }
        )
    return audit


def _failure_report(
    manifests: Sequence[Mapping[str, Any]],
    reference_passed: set[str],
    accepted: set[str],
    audit: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# KernelBench-gap canary failure and bias audit",
        "",
        "Counts are samples. Families are exclusive; failure reasons are source-bound raw evidence signatures.",
        "",
        "| family | generated | reference pass | liveness/accepted |",
        "| --- | ---: | ---: | ---: |",
    ]
    for family in EXPECTED_FAMILY_QUOTAS:
        rows = [item for item in manifests if item["primary_family"] == family]
        lines.append(
            f"| {family} | {len(rows)} | {sum(item['uuid'] in reference_passed for item in rows)} | {sum(item['uuid'] in accepted for item in rows)} |"
        )
    lines.extend(["", "| template | generated | reference pass | liveness/accepted |", "| --- | ---: | ---: | ---: |"])
    for template in sorted({str(item["template_id"]) for item in manifests}):
        rows = [item for item in manifests if item["template_id"] == template]
        lines.append(
            f"| {template} | {len(rows)} | {sum(item['uuid'] in reference_passed for item in rows)} | {sum(item['uuid'] in accepted for item in rows)} |"
        )
    by_failure = collections.Counter((str(item["failure_stage"]), str(item["failure_signature"])) for item in audit)
    lines.extend(["", "## Failure signatures", "", "| stage | signature | samples |", "| --- | --- | ---: |"])
    for (stage, signature), count in sorted(by_failure.items()):
        lines.append(f"| {stage} | `{signature}` | {count} |")
    lines.append("")
    return "\n".join(lines)


def analyze(
    candidates_path: Path,
    manifest_path: Path,
    reference_dir: Path,
    liveness_dir: Path,
    output_dir: Path,
    *,
    allow_unbound_liveness_launcher: bool = False,
) -> dict[str, Any]:
    candidates_path, manifest_path, output_dir = (
        candidates_path.resolve(),
        manifest_path.resolve(),
        output_dir.resolve(),
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"analysis output directory is not empty:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    table, manifests, by_uuid, static_summary = _verify_static(candidates_path, manifest_path)
    # Keep code outside the materialized manifest but available to the distribution parser.
    for item in manifests:
        item["_code"] = by_uuid[str(item["uuid"])]["row"]["reward_model"]["ground_truth"]
    reference_rows, reference_shards = _collect_shards(reference_dir)
    references, reference_failures, reference_binding = _verify_reference(reference_rows, by_uuid, candidates_path)
    liveness_rows, liveness_shards = _collect_shards(liveness_dir)
    liveness, liveness_failures, liveness_binding = _verify_liveness(
        liveness_rows, by_uuid, references, candidates_path, manifest_path
    )
    reference_passed = {uuid for uuid, item in references.items() if item.get("passed") is True}
    launcher_evidence = (
        {
            "status": "debug_opt_out_unbound_launcher",
            "reason": "--allow-unbound-liveness-launcher was explicitly supplied",
        }
        if allow_unbound_liveness_launcher
        else _verify_liveness_launcher_evidence(
            liveness_dir,
            liveness_rows=liveness_rows,
            liveness=liveness,
            liveness_binding=liveness_binding,
            by_uuid=by_uuid,
            reference_passed=reference_passed,
            candidates_path=candidates_path,
            manifest_path=manifest_path,
        )
    )
    accepted = {uuid for uuid, item in liveness.items() if item.get("passed") is True}
    if not accepted.issubset(reference_passed):
        raise ValueError("liveness accepted UUIDs are not a reference-pass subset")
    family_accepted = collections.Counter(by_uuid[uuid]["manifest"]["primary_family"] for uuid in accepted)
    missing = [family for family in EXPECTED_FAMILY_QUOTAS if family_accepted[family] == 0]
    below = {
        family: family_accepted[family]
        for family, minimum in REQUIRED_ACCEPTED.items()
        if family_accepted[family] < minimum
    }
    if missing or below:
        raise ValueError(f"accepted mandatory coverage failed: zero={missing}, below_threshold={below}")
    accepted_indices = [index for index, item in enumerate(manifests) if item["uuid"] in accepted]
    accepted_table = table.take(pa.array(accepted_indices, type=pa.int64()))
    accepted_path = output_dir / "accepted.parquet"
    accepted_manifest_path = output_dir / "accepted.manifest.jsonl"
    failure_audit_path = output_dir / "failure_audit.jsonl"
    failure_report_path = output_dir / "failure_bias_report.md"
    raw_hash_path = output_dir / "raw_artifact_sha256.json"
    pq.write_table(accepted_table, accepted_path, compression="zstd")
    accepted_manifests: list[dict[str, Any]] = []
    for item in manifests:
        uuid = str(item["uuid"])
        if uuid not in accepted:
            continue
        updated = copy.deepcopy(item)
        updated.pop("_code", None)
        updated.update(
            {
                "reference_runtime_status": "passed",
                "kernelbench_gap_liveness_status": "passed",
                "runtime_status": ACCEPTED_RUNTIME_STATUS,
                "governance_status": ACCEPTED_GOVERNANCE_STATUS,
                "training_approved": False,
                "runtime_evidence": {
                    "reference_contract_fingerprint": references[uuid]["contract_fingerprint"],
                    "reference_record_sha256": _canonical_sha256(references[uuid]),
                    "liveness_binding_sha256": liveness[uuid]["validation_binding_sha256"],
                    "liveness_record_sha256": _canonical_sha256(liveness[uuid]),
                    "adapter_source_sha256": liveness[uuid]["adapter_source_sha256"],
                    "shared_runtime_core_source_sha256": liveness[uuid]["shared_runtime_core_source_sha256"],
                    "generator_source_sha256": liveness[uuid]["generator_source_sha256"],
                    "launcher_source_sha256": liveness[uuid]["launcher_source_sha256"],
                },
            }
        )
        accepted_manifests.append(updated)
    with accepted_manifest_path.open("w", encoding="utf-8") as handle:
        for item in accepted_manifests:
            handle.write(_canonical_json(item) + "\n")
    audit = _failure_audit(manifests, references, liveness, accepted)
    with failure_audit_path.open("w", encoding="utf-8") as handle:
        for item in audit:
            handle.write(_canonical_json(item) + "\n")
    failure_report_path.write_text(_failure_report(manifests, reference_passed, accepted, audit), encoding="utf-8")
    source_closure = {
        "analyzer": _module_closure(sys.modules[__name__]),
        "generator": _module_closure(generator),
        "reference_validator": _module_closure(reference_validator),
        "liveness_adapter": _module_closure(liveness_validator),
        "shared_runtime_core": _module_closure(shared_runtime_core),
    }
    raw_hashes = {
        "candidates": {"path": str(candidates_path), "sha256": _sha256_file(candidates_path)},
        "manifest": {"path": str(manifest_path), "sha256": _sha256_file(manifest_path)},
        "reference_shards": reference_shards,
        "liveness_shards": liveness_shards,
        "liveness_launcher_evidence": launcher_evidence,
        "source_closure": source_closure,
    }
    raw_hash_path.write_text(json.dumps(raw_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    distribution = {
        "generated": _distribution_metrics(manifests),
        "reference_passed": _distribution_metrics(manifests, reference_passed),
        "accepted": _distribution_metrics(manifests, accepted),
    }
    summary = {
        "analysis_contract": ANALYSIS_CONTRACT,
        "generated": len(manifests),
        "reference_passed": len(reference_passed),
        "reference_failed": len(manifests) - len(reference_passed),
        "liveness_executed": len(liveness),
        "liveness_passed": len(accepted),
        "liveness_failed": len(liveness) - len(accepted),
        "accepted": len(accepted),
        "required_accepted_by_family": REQUIRED_ACCEPTED,
        "family_counts": {
            family: {
                "generated": EXPECTED_FAMILY_QUOTAS[family],
                "reference_passed": sum(
                    by_uuid[uuid]["manifest"]["primary_family"] == family for uuid in reference_passed
                ),
                "liveness_passed": family_accepted[family],
            }
            for family in EXPECTED_FAMILY_QUOTAS
        },
        "static": static_summary,
        "runtime_bindings": {
            "reference": reference_binding,
            "liveness": liveness_binding,
            "liveness_launcher": launcher_evidence,
        },
        "analyzer_source_sha256": _sha256_file(Path(__file__).resolve()),
        "failure_counts": {"reference": len(reference_failures), "liveness": len(liveness_failures)},
        "distribution_differences_3_to_5": distribution,
        "runtime_status": ACCEPTED_RUNTIME_STATUS,
        "governance_status": ACCEPTED_GOVERNANCE_STATUS,
        "training_approved": False,
        "artifacts": {
            "accepted": {"path": str(accepted_path), "sha256": _sha256_file(accepted_path)},
            "accepted_manifest": {"path": str(accepted_manifest_path), "sha256": _sha256_file(accepted_manifest_path)},
            "failure_audit": {"path": str(failure_audit_path), "sha256": _sha256_file(failure_audit_path)},
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
    parser.add_argument(
        "--allow-unbound-liveness-launcher",
        action="store_true",
        help="debug only: do not require the formal eight-shard launcher evidence",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    analyze(
        args.candidates,
        args.manifest,
        args.reference_dir,
        args.liveness_dir,
        args.output_dir,
        allow_unbound_liveness_launcher=args.allow_unbound_liveness_launcher,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
