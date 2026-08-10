#!/usr/bin/env python3
"""Fail-closed exact join and diversity audit for the frontier 1k canary.

This is deliberately an offline artifact verifier.  It does not regenerate
rows, execute CUDA, or turn a review-only lane into training data.
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

from tools.data.synthesize import validate_train_mode_contract as reference_validator  # noqa: E402
from tools.data.synthesize.frontier_operator_method import generate_frontier_operator as generator  # noqa: E402
from tools.data.synthesize.frontier_operator_method import (  # noqa: E402
    validate_frontier_liveness as liveness_validator,
)

ANALYSIS_CONTRACT = "frontier_operator_exact_analysis_v1"
MAX_AUTHORIZED_CANDIDATES = 1_000
ACCEPTED_RUNTIME_STATUS = "kernelgym_reference_and_frontier_operator_liveness_passed"
ACCEPTED_GOVERNANCE_STATUS = "frontier_operator_review_only_training_not_approved"
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
ALLOWED_STATUSES = frozenset({"passed", "reference_failed", "unsupported", "failed", "timeout", "protocol_error"})
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
AXES = ("family", "template", "scenario_source", "sparse_format", "topology", "state", "modality", "precision")


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


def _require_sha256(value: Any, label: str) -> str:
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


def _first(value: Mapping[str, Any], *paths: str, default: Any = None) -> Any:
    for path in paths:
        item = _nested(value, path, default=None)
        if item is not None:
            return item
    return default


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


def _module_closure(module: Any) -> list[dict[str, str]]:
    """Hash repo-local imported Python sources transitively for audit provenance.

    This is output provenance, not a new generator/validator artifact contract:
    existing raw records continue to bind their actual source fields below.
    """

    root_name = str(module.__name__)
    root_file = Path(module.__file__).resolve() if getattr(module, "__file__", None) else None
    pending = [root_name]
    seen: set[str] = set()
    paths: dict[Path, str] = {}
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
            imported: list[str] = []
            if isinstance(node, ast.Import):
                imported = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                target = ("." * node.level) + (node.module or "")
                try:
                    base = importlib.util.resolve_name(target, package) if node.level else (node.module or "")
                except (ImportError, ValueError):
                    base = ""
                if base:
                    imported.append(base)
                    if node.module:
                        imported.extend(f"{base}.{alias.name}" for alias in node.names if alias.name != "*")
            for candidate in imported:
                try:
                    candidate_spec = importlib.util.find_spec(candidate)
                except (ImportError, ModuleNotFoundError, ValueError):
                    continue
                if candidate_spec and candidate_spec.origin and candidate_spec.origin.endswith(".py"):
                    try:
                        Path(candidate_spec.origin).resolve().relative_to(_REPO_ROOT)
                    except ValueError:
                        continue
                    pending.append(candidate)
    return [{"module": paths[path], "path": str(path), "sha256": _sha256_file(path)} for path in sorted(paths)]


def _axis_values(manifest: Mapping[str, Any], axis: str) -> list[str]:
    if axis == "family":
        raw = manifest.get("primary_family")
    elif axis == "template":
        raw = manifest.get("template_id")
    elif axis == "scenario_source":
        source = manifest.get("scenario_source")
        if not isinstance(source, Mapping):
            raise ValueError(f"scenario_source missing/invalid:{manifest.get('uuid')}")
        raw = source.get("source_url")
    else:
        labels = manifest.get("coverage_labels")
        if not isinstance(labels, Mapping):
            raise ValueError(f"coverage_labels missing/invalid:{manifest.get('uuid')}")
        raw = labels.get(axis)
    if raw is None:
        raise ValueError(f"coverage axis missing:{axis}:{manifest.get('uuid')}")
    values = raw if isinstance(raw, list) else [raw]
    if not values:
        raise ValueError(f"coverage axis empty:{axis}:{manifest.get('uuid')}")
    labels = [_canonical_json(value) if isinstance(value, (Mapping, list)) else str(value) for value in values]
    if any(not label for label in labels) or len(labels) != len(set(labels)):
        raise ValueError(f"coverage axis invalid/duplicated:{axis}:{manifest.get('uuid')}")
    return labels


def _scenario_source_valid(value: Any) -> bool:
    return (
        isinstance(value, Mapping)
        and set(("source_url", "source_kind", "implementation_note", "source_registry_version")).issubset(value)
        and all(
            isinstance(value[key], str) and value[key]
            for key in ("source_url", "source_kind", "implementation_note", "source_registry_version")
        )
    )


def _declared_identities(manifest: Mapping[str, Any]) -> set[str]:
    declared = manifest.get("declared_ops")
    if not isinstance(declared, list) or not declared:
        raise ValueError(f"declared_ops missing/invalid:{manifest.get('uuid')}")
    identities: set[str] = set()
    for declared_op in declared:
        if not isinstance(declared_op, Mapping) or not isinstance(declared_op.get("op_id"), str):
            raise ValueError(f"declared op invalid:{manifest.get('uuid')}")
        runtime = declared_op.get("runtime_identities")
        if not isinstance(runtime, list) or not runtime:
            raise ValueError(f"declared runtime identities missing:{manifest.get('uuid')}")
        for identity in runtime:
            if not isinstance(identity, Mapping) or not isinstance(identity.get("schema"), str):
                raise ValueError(f"declared runtime identity invalid:{manifest.get('uuid')}")
            overload = identity.get("overload") or "<default>"
            identities.add(f"{identity['schema']}.{overload}")
    return identities


def _verify_unique_semantic_coordinate_cells(manifests: Sequence[Mapping[str, Any]]) -> int:
    """Reject shape-only variant clones using requested semantic coordinates.

    ``variant`` is a registry row ordinal, not a semantic coordinate.  The
    identity is therefore canonical JSON of the template id and requested
    coordinates after removing that ordinal; templates remain part of the
    identity because their coordinate vocabularies are intentionally distinct.
    """

    by_template: dict[str, set[str]] = collections.defaultdict(set)
    quotas = {item.template_id: item.variants for item in generator.TEMPLATES}
    for manifest in manifests:
        template_id = manifest.get("template_id")
        solver = manifest.get("constraint_solver")
        if not isinstance(template_id, str) or template_id not in quotas or not isinstance(solver, Mapping):
            raise ValueError(f"semantic coordinate cell schema invalid:{manifest.get('uuid')}")
        requested = solver.get("requested_coordinates")
        if not isinstance(requested, Mapping) or requested.get("variant") != manifest.get("template_variant"):
            raise ValueError(f"semantic coordinate variant binding invalid:{manifest.get('uuid')}")
        identity_coordinates = dict(requested)
        identity_coordinates.pop("variant")
        identity = _canonical_json({"template_id": template_id, "requested_coordinates": identity_coordinates})
        if identity in by_template[template_id]:
            raise ValueError(
                f"duplicate semantic coordinate cell excluding variant:{template_id}:{manifest.get('uuid')}"
            )
        by_template[template_id].add(identity)
    if set(by_template) != set(quotas) or any(
        len(by_template[template]) != quota for template, quota in quotas.items()
    ):
        observed = {template: len(cells) for template, cells in sorted(by_template.items())}
        raise ValueError(f"template semantic coordinate cell quotas differ:{observed}")
    total = sum(len(cells) for cells in by_template.values())
    if total != MAX_AUTHORIZED_CANDIDATES:
        raise ValueError(f"semantic coordinate cells must total exactly 1000:{total}")
    return total


def _verify_static(
    candidates_path: Path, manifest_path: Path
) -> tuple[pa.Table, list[dict[str, Any]], dict[str, dict[str, Any]], int]:
    table = pq.read_table(candidates_path)
    rows = table.to_pylist()
    manifests = _read_jsonl(manifest_path)
    expected_rows = getattr(generator, "EXACT_CANARY_ROWS", MAX_AUTHORIZED_CANDIDATES)
    if (
        expected_rows != MAX_AUTHORIZED_CANDIDATES
        or getattr(generator, "MAX_AUTHORIZED_CANDIDATES", None) != MAX_AUTHORIZED_CANDIDATES
    ):
        raise ValueError("generator does not preserve the exact authorized 1k cap")
    if len(rows) != MAX_AUTHORIZED_CANDIDATES or len(manifests) != MAX_AUTHORIZED_CANDIDATES:
        raise ValueError(f"static canary count must be exactly 1000:{len(rows)}:{len(manifests)}")
    generator_sha = _sha256_file(Path(generator.__file__).resolve())
    seen: dict[str, set[str]] = {
        key: set() for key in ("uuid", "reference_sha256", "normalized_ast_sha256", "row_payload_sha256")
    }
    by_uuid: dict[str, dict[str, Any]] = {}
    family_counts: collections.Counter[str] = collections.Counter()
    template_counts: collections.Counter[str] = collections.Counter()
    commits: set[str] = set()
    template_bindings: set[str] = set()
    roots_bindings: set[str] = set()
    for index, (row, manifest) in enumerate(zip(rows, manifests, strict=True)):
        uuid = _nested(row, "extra_info.uuid")
        code = _nested(row, "reward_model.ground_truth")
        prompt = _nested(row, "prompt")
        if manifest.get("candidate_row_index") != index or manifest.get("uuid") != uuid:
            raise ValueError(f"static identity mismatch:{index}")
        for name in (
            "MANIFEST_VERSION",
            "CONTRACT_VERSION",
            "REGISTRY_VERSION",
            "STATIC_CONTRACT_VERSION",
            "RUNTIME_CONTRACT_VERSION",
        ):
            manifest_key = {
                "MANIFEST_VERSION": "manifest_contract_version",
                "CONTRACT_VERSION": "generator_contract_version",
                "REGISTRY_VERSION": "registry_version",
                "STATIC_CONTRACT_VERSION": "static_contract_version",
                "RUNTIME_CONTRACT_VERSION": "runtime_contract_version",
            }[name]
            if manifest.get(manifest_key) != getattr(generator, name):
                raise ValueError(f"static contract mismatch:{index}:{manifest_key}")
        if manifest.get("generator_source_sha256") != generator_sha:
            raise ValueError(f"generator source differs from static artifact:{index}")
        commit = manifest.get("git_commit_at_generation")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise ValueError(f"generator commit binding missing/invalid:{index}")
        commits.add(commit)
        if (
            manifest.get("lineage_kind") != "standalone_frontier_synthetic"
            or manifest.get("parent_uuid") is not None
            or manifest.get("training_approved") is not False
            or manifest.get("structured_output_deferred") is not True
            or manifest.get("final_output_contract") != {"kind": "single_dense_tensor", "finite_required": True}
        ):
            raise ValueError(f"governance/output/lineage mismatch:{index}")
        if not isinstance(uuid, str) or not isinstance(code, str) or not isinstance(prompt, list) or len(prompt) != 1:
            raise ValueError(f"candidate schema mismatch:{index}")
        if not prompt[0].get("content", "").endswith(code):
            raise ValueError(f"prompt does not end in exact reference:{index}")
        content = prompt[0]["content"]
        prefix = content[: -len(code)]
        if manifest.get("prompt_sha256") != _sha256_bytes(content.encode()) or manifest.get(
            "prompt_prefix_sha256"
        ) != _sha256_bytes(prefix.encode()):
            raise ValueError(f"prompt provenance hash mismatch:{index}")
        if _sha256_bytes(code.encode()) != manifest.get("reference_sha256"):
            raise ValueError(f"reference hash mismatch:{index}")
        if generator._normalized_ast_sha256(code) != manifest.get("normalized_ast_sha256"):
            raise ValueError(f"normalized AST hash mismatch:{index}")
        if _canonical_sha256(row) != manifest.get("row_payload_sha256"):
            raise ValueError(f"row payload hash mismatch:{index}")
        rendered = generator._render(str(manifest.get("template_id")), int(manifest.get("template_variant")))
        if len(rendered) != 4:
            raise ValueError(f"frontier registry render must return code/ops/labels/constraint:{index}")
        expected_code, expected_ops, expected_labels, expected_constraint = rendered
        if (
            code != expected_code
            or manifest.get("declared_ops") != expected_ops
            or manifest.get("coverage_labels") != expected_labels
            or manifest.get("constraint_solver") != expected_constraint
        ):
            raise ValueError(f"closed registry replay mismatch:{index}")
        if manifest.get("static_proof") != generator._static_contract(code, expected_ops):
            raise ValueError(f"static dependency replay mismatch:{index}")
        solver = manifest.get("constraint_solver")
        if (
            not isinstance(solver, Mapping)
            or solver.get("contract_version") != "frontier_constraint_cells_v1"
            or solver.get("backend") != "deterministic_constructive_arithmetic"
        ):
            raise ValueError(f"solver envelope mismatch:{index}")
        if solver.get("template_id") != manifest.get("template_id") or solver.get("variant") != manifest.get(
            "template_variant"
        ):
            raise ValueError(f"solver identity mismatch:{index}")
        invariants = solver.get("invariants")
        if (
            not isinstance(invariants, list)
            or not invariants
            or any(
                not isinstance(item, Mapping)
                or item.get("passed") is not True
                or not isinstance(item.get("name"), str)
                for item in invariants
            )
        ):
            raise ValueError(f"solver invariants missing/failed:{index}")
        if solver.get("invariant_payload_sha256") != _canonical_sha256(invariants):
            raise ValueError(f"solver invariant hash mismatch:{index}")
        if not _scenario_source_valid(manifest.get("scenario_source")):
            raise ValueError(f"scenario source mismatch:{index}")
        for axis in AXES:
            _axis_values(manifest, axis)
        _declared_identities(manifest)
        for field in seen:
            value = manifest.get(field)
            if not isinstance(value, str) or value in seen[field]:
                raise ValueError(f"static {field} not unique/valid:{index}")
            seen[field].add(value)
        family_counts[str(manifest["primary_family"])] += 1
        template_counts[str(manifest["template_id"])] += 1
        template_bindings.add(_canonical_json(manifest.get("prompt_template")))
        roots_bindings.add(_canonical_json(manifest.get("decontamination_roots")))
        by_uuid[uuid] = {"row_index": index, "row": row, "manifest": manifest}
    if len(commits) != 1:
        raise ValueError("static artifact mixes generator commit bindings")
    if len(template_bindings) != 1 or len(roots_bindings) != 1:
        raise ValueError("static artifact mixes prompt/decontamination provenance")
    template_binding = json.loads(next(iter(template_bindings)))
    if not isinstance(template_binding, Mapping) or not isinstance(template_binding.get("path"), str):
        raise ValueError("prompt template binding is invalid")
    template_path = Path(template_binding["path"])
    if not template_path.is_file() or _sha256_file(template_path) != template_binding.get("sha256"):
        raise ValueError("prompt template artifact changed")
    for root in json.loads(next(iter(roots_bindings))):
        if not isinstance(root, Mapping) or not isinstance(root.get("path"), str):
            raise ValueError("decontamination root binding invalid")
        path = Path(root["path"])
        if (
            not path.is_file()
            or _sha256_file(path) != root.get("sha256")
            or pq.ParquetFile(path).metadata.num_rows != root.get("rows")
        ):
            raise ValueError(f"decontamination root changed:{path}")
    if dict(family_counts) != generator.FAMILY_QUOTAS:
        raise ValueError(f"exclusive family quotas differ:{dict(family_counts)}")
    if dict(template_counts) != {item.template_id: item.variants for item in generator.TEMPLATES}:
        raise ValueError(f"template quotas differ:{dict(template_counts)}")
    unique_semantic_coordinate_cells = _verify_unique_semantic_coordinate_cells(manifests)
    return table, manifests, by_uuid, unique_semantic_coordinate_cells


def _record_value(record: Mapping[str, Any], modern: str, legacy: str | None = None) -> Any:
    if modern in record:
        return record[modern]
    return record.get(legacy) if legacy else None


def _record_reason(record: Mapping[str, Any]) -> str:
    reason = _first(record, "reason", "error", "failure_reason")
    if reason is None:
        reasons = record.get("failure_reasons")
        reason = reasons if reasons is not None else record.get("status", "unknown")
    return _canonical_json(reason) if isinstance(reason, (Mapping, list)) else str(reason)


def _normalize_record(record: Mapping[str, Any], *, stage: str) -> dict[str, Any]:
    """Expose the common contract envelope without replacing raw evidence."""

    passed = record.get("passed")
    raw_status = record.get("status")
    status = (
        raw_status
        if raw_status in ALLOWED_STATUSES
        else ("reference_failed" if stage == "reference" and passed is False else "failed")
    )
    reason = _record_reason(record)
    source_sha = (
        _first(record, "candidates_sha256", "source_sha256")
        if stage == "liveness"
        else _first(record, "source_sha256", "candidates_sha256")
    )
    return {
        "uuid": record.get("uuid"),
        "row_index": _record_value(record, "row_index", "candidate_row_index"),
        "family": _record_value(record, "family", "primary_family"),
        "template_id": record.get("template_id"),
        "source_sha256": source_sha,
        "manifest_sha256": record.get("manifest_sha256"),
        "validator_sha256": _record_value(record, "validator_sha256", "validator_source_sha256"),
        "launcher_sha256": _record_value(record, "launcher_sha256", "launcher_source_sha256"),
        "contract_binding_sha256": _first(
            record, "contract_binding_sha256", "validation_binding_sha256", "contract_fingerprint"
        ),
        "status": status,
        "raw_status": raw_status,
        "passed": passed,
        "failure_stage": record.get("failure_stage") or (None if passed is True else stage),
        "failure_signature": record.get("failure_signature")
        or (None if passed is True else _canonical_sha256({"stage": stage, "reason": reason})),
        "reason": reason,
        "duration_seconds": record.get("duration_seconds"),
        "evidence": record.get("evidence", record),
    }


def _verify_common_record(
    record: Mapping[str, Any],
    *,
    stage: str,
    static: Mapping[str, Any],
    candidates_sha: str,
    manifest_sha: str,
    expected_validator_sha: str,
    require_family: bool = True,
) -> dict[str, Any]:
    normalized = _normalize_record(record, stage=stage)
    uuid = normalized["uuid"]
    if not isinstance(uuid, str) or uuid != static["manifest"]["uuid"]:
        raise ValueError(f"{stage} UUID mismatch:{uuid!r}")
    if normalized["row_index"] != static["row_index"]:
        raise ValueError(f"{stage} row index mismatch:{uuid}")
    if require_family and normalized["family"] != static["manifest"]["primary_family"]:
        raise ValueError(f"{stage} family mismatch:{uuid}")
    if not require_family and normalized["family"] not in (None, static["manifest"]["primary_family"]):
        raise ValueError(f"{stage} family mismatch:{uuid}")
    if normalized["template_id"] not in (None, static["manifest"]["template_id"]):
        raise ValueError(f"{stage} template mismatch:{uuid}")
    if normalized["source_sha256"] != candidates_sha or normalized["manifest_sha256"] not in (None, manifest_sha):
        raise ValueError(f"{stage} candidates/manifest binding mismatch:{uuid}")
    if normalized["validator_sha256"] != expected_validator_sha:
        raise ValueError(f"{stage} validator source mismatch:{uuid}")
    for field in ("launcher_sha256", "contract_binding_sha256"):
        _require_sha256(normalized[field], f"{stage}.{field}:{uuid}")
    if normalized["status"] not in ALLOWED_STATUSES or type(normalized["passed"]) is not bool:
        raise ValueError(f"{stage} status/passed invalid:{uuid}")
    if normalized["passed"] is True and normalized["status"] != "passed":
        raise ValueError(f"{stage} passing record has non-passed status:{uuid}")
    if normalized["passed"] is False and normalized["status"] == "passed":
        raise ValueError(f"{stage} failed record has passed status:{uuid}")
    if normalized["passed"] is False and (not normalized["failure_stage"] or not normalized["failure_signature"]):
        raise ValueError(f"{stage} failure attribution missing:{uuid}")
    return normalized


def _gpu_authorized(record: Mapping[str, Any]) -> bool:
    gpu = _first(record, "gpu", "evidence.gpu", default={})
    return isinstance(gpu, Mapping) and any(name in str(gpu.get("name")) for name in ("A800", "H20"))


def _memory_guard_valid(value: Any) -> bool:
    return not reference_validator.validate_memory_guard_evidence(value)


def _verify_reference(
    records: Sequence[Mapping[str, Any]],
    by_uuid: Mapping[str, Mapping[str, Any]],
    candidates_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, dict[str, Any]]]:
    if len(records) != MAX_AUTHORIZED_CANDIDATES:
        raise ValueError(f"reference result count must exactly equal 1000:{len(records)}")
    candidate_sha, manifest_sha = _sha256_file(candidates_path), _sha256_file(manifest_path)
    validator_sha = _sha256_file(Path(reference_validator.__file__).resolve())
    indexed: dict[str, Mapping[str, Any]] = {}
    normalized: dict[str, dict[str, Any]] = {}
    launchers: set[str] = set()
    bindings: set[str] = set()
    indices: set[int] = set()
    for record in records:
        uuid = record.get("uuid")
        if not isinstance(uuid, str) or uuid not in by_uuid or uuid in indexed:
            raise ValueError(f"reference UUID missing/duplicate/unknown:{uuid!r}")
        item = _verify_common_record(
            record,
            stage="reference",
            static=by_uuid[uuid],
            candidates_sha=candidate_sha,
            manifest_sha=manifest_sha,
            expected_validator_sha=validator_sha,
            require_family=False,
        )
        if record.get("contract_version") != REFERENCE_CONTRACT:
            raise ValueError(f"reference contract version mismatch:{uuid}")
        if record.get("reference_sha256") != by_uuid[uuid]["manifest"]["reference_sha256"]:
            raise ValueError(f"reference exact code identity mismatch:{uuid}")
        if (
            record.get("trials") != 5
            or record.get("seed") != 42
            or record.get("training") is not True
            or record.get("max_device_memory_gib") != 64.0
        ):
            raise ValueError(f"reference policy mismatch:{uuid}")
        kernelgym = record.get("kernelgym")
        if not isinstance(kernelgym, Mapping) or kernelgym.get("git_commit") != EXPECTED_KERNELGYM_COMMIT:
            raise ValueError(f"KernelGym authority commit mismatch:{uuid}")
        if any(kernelgym.get(key) != value for key, value in EXPECTED_KERNELGYM_HASHES.items()):
            raise ValueError(f"KernelGym evaluator bundle mismatch:{uuid}")
        if not _memory_guard_valid(record.get("memory_guard")):
            raise ValueError(f"reference memory guard invalid:{uuid}")
        if not _gpu_authorized(record):
            raise ValueError(f"reference GPU is not authorized A800/H20:{uuid}")
        if item["passed"] is True:
            if (
                record.get("kernelgym_correctness") is not True
                or record.get("reference_forward_calls") != 5
                or record.get("identical_forward_calls") != 5
                or record.get("reference_training") is not True
                or record.get("identical_training") is not True
                or _first(record, "persistent_model_instances", "evidence.persistent_model_instances") is not True
            ):
                raise ValueError(f"reference pass lacks train-mode/persistence evidence:{uuid}")
        indexed[uuid], normalized[uuid] = record, item
        indices.add(item["row_index"])
        launchers.add(item["launcher_sha256"])
        bindings.add(item["contract_binding_sha256"])
    if indices != set(range(MAX_AUTHORIZED_CANDIDATES)):
        raise ValueError("reference records do not exactly cover row indices 0..999")
    if len(launchers) != 1 or len(bindings) != 1:
        raise ValueError("reference run mixes launcher or binding hashes")
    return indexed, normalized


def _read_allowlist(path: Path) -> list[str]:
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(f"empty liveness allowlist:{path}")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                parsed.append(json.loads(line))
            except json.JSONDecodeError:
                parsed.append(line)
    if isinstance(parsed, Mapping):
        parsed = parsed.get("uuids")
    if not isinstance(parsed, list):
        raise ValueError(f"allowlist is not a UUID list:{path}")
    values = [item.get("uuid") if isinstance(item, Mapping) else item for item in parsed]
    if not all(isinstance(item, str) for item in values) or len(values) != len(set(values)):
        raise ValueError(f"allowlist UUID values invalid:{path}")
    return list(values)


def _binding_evidence(record: Mapping[str, Any]) -> Mapping[str, Any]:
    evidence = record.get("binding_evidence")
    if not isinstance(evidence, Mapping):
        evidence = _nested(record, "evidence.binding_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError(f"liveness binding evidence missing:{record.get('uuid')}")
    return evidence


def _realized_identities(record: Mapping[str, Any]) -> set[str]:
    """Return only declared dispatch identities proven to reach the final Tensor."""

    identities: set[str] = set()
    trials = _first(record, "trials", "evidence.trials", default=[])
    if not isinstance(trials, list) or not trials:
        raise ValueError(f"liveness trials missing:{record.get('uuid')}")
    for trial in trials:
        trace = trial.get("trace") if isinstance(trial, Mapping) else None
        if not isinstance(trace, Mapping):
            raise ValueError(f"liveness trace missing:{record.get('uuid')}")
        per_op = trace.get("per_declared_op", [])
        if not isinstance(per_op, list):
            raise ValueError(f"liveness declared-op trace invalid:{record.get('uuid')}")
        for item in per_op:
            matches = item.get("matched_identities") if isinstance(item, Mapping) else None
            if (
                isinstance(item, Mapping)
                and item.get("returned_output_witness") is True
                and isinstance(matches, Mapping)
            ):
                identities.update(str(key) for key, value in matches.items() if type(value) is int and value > 0)
    if not identities:
        raise ValueError(f"liveness realized ATen identities missing:{record.get('uuid')}")
    return identities


def _verify_liveness_pass(record: Mapping[str, Any], manifest: Mapping[str, Any]) -> set[str]:
    """Recheck the dispatch/provenance witnesses instead of trusting a pass bit."""

    trials = _first(record, "trials", "evidence.trials", default=[])
    if not isinstance(trials, list) or len(trials) != 3:
        raise ValueError(f"liveness must carry exactly three trials:{record.get('uuid')}")
    expected: dict[str, tuple[set[str], int]] = {}
    for declared in manifest["declared_ops"]:
        expected[str(declared["op_id"])] = (
            {
                f"{identity['schema']}.{identity.get('overload') or '<default>'}"
                for identity in declared["runtime_identities"]
            },
            int(declared["min_calls_per_trial"]),
        )
    for ordinal, trial in enumerate(trials):
        trace = trial.get("trace") if isinstance(trial, Mapping) else None
        if not isinstance(trace, Mapping):
            raise ValueError(f"liveness trace missing:{record.get('uuid')}:{ordinal}")
        per_op = trace.get("per_declared_op")
        if (
            not isinstance(per_op, list)
            or len(per_op) != len(expected)
            or {item.get("op_id") for item in per_op if isinstance(item, Mapping)} != set(expected)
        ):
            raise ValueError(f"liveness declared-op witness set mismatch:{record.get('uuid')}:{ordinal}")
        final_ids = trace.get("final_output_declared_op_ids")
        if final_ids is not None and final_ids != sorted(expected):
            raise ValueError(f"liveness returned-output declared-op mismatch:{record.get('uuid')}:{ordinal}")
        for item in per_op:
            matches = item.get("matched_identities")
            op_id = item.get("op_id")
            if (
                not isinstance(matches, Mapping)
                or not matches
                or not set(matches).issubset(expected[op_id][0])
                or not all(type(count) is int and count > 0 for count in matches.values())
                or type(item.get("calls")) is not int
                or item["calls"] < expected[op_id][1]
                or item.get("returned_output_witness") is not True
            ):
                raise ValueError(f"liveness declared-op witness invalid:{record.get('uuid')}:{ordinal}:{op_id}")
        if (
            _first(trial, "single_tensor_output", "trace.single_tensor_output") is not True
            or _first(trial, "output_finite", "trace.output_finite") is not True
        ):
            raise ValueError(f"liveness trial output evidence invalid:{record.get('uuid')}:{ordinal}")
    return _realized_identities(record)


def _verify_liveness(
    records: Sequence[Mapping[str, Any]],
    by_uuid: Mapping[str, Mapping[str, Any]],
    reference_normalized: Mapping[str, Mapping[str, Any]],
    candidates_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, dict[str, Any]], dict[str, set[str]]]:
    candidate_sha, manifest_sha = _sha256_file(candidates_path), _sha256_file(manifest_path)
    validator_sha = _sha256_file(Path(liveness_validator.__file__).resolve())
    semantic_validator_path = Path(liveness_validator._SEMANTIC_VALIDATOR_PATH).resolve()
    if _sha256_file(semantic_validator_path) != liveness_validator.SEMANTIC_VALIDATOR_SOURCE_SHA256:
        raise ValueError("frontier liveness wrapper semantic-validator closure is stale")
    generator_sha = _sha256_file(Path(generator.__file__).resolve())
    reference_passed = [
        uuid
        for uuid, record in sorted(reference_normalized.items(), key=lambda item: item[1]["row_index"])
        if record["passed"] is True
    ]
    if len(records) != len(reference_passed):
        raise ValueError(
            f"liveness result count must equal fresh reference-pass set:{len(records)}:{len(reference_passed)}"
        )
    indexed: dict[str, Mapping[str, Any]] = {}
    normalized: dict[str, dict[str, Any]] = {}
    realized: dict[str, set[str]] = {}
    launchers: set[str] = set()
    bindings: set[str] = set()
    allowlist_paths: set[str] = set()
    allowlist_hashes: set[str] = set()
    for record in records:
        uuid = record.get("uuid")
        if not isinstance(uuid, str) or uuid not in reference_passed or uuid in indexed:
            raise ValueError(f"liveness UUID missing/duplicate/not-reference-passed:{uuid!r}")
        item = _verify_common_record(
            record,
            stage="liveness",
            static=by_uuid[uuid],
            candidates_sha=candidate_sha,
            manifest_sha=manifest_sha,
            expected_validator_sha=validator_sha,
        )
        if record.get("contract_version") != liveness_validator.CONTRACT_VERSION:
            raise ValueError(f"liveness contract version mismatch:{uuid}")
        if (
            record.get("generator_source_sha256") != generator_sha
            or record.get("semantic_validator_source_sha256") != liveness_validator.SEMANTIC_VALIDATOR_SOURCE_SHA256
        ):
            raise ValueError(f"liveness wrapper transitive source binding mismatch:{uuid}")
        binding_evidence = _binding_evidence(record)
        path, digest = binding_evidence.get("allowlist_path"), binding_evidence.get("allowlist_sha256")
        if not isinstance(path, str) or not path or not _require_sha256(digest, f"allowlist hash:{uuid}"):
            raise ValueError(f"liveness allowlist binding invalid:{uuid}")
        allowlist_paths.add(path)
        allowlist_hashes.add(digest)
        if item["passed"] is True:
            if not _gpu_authorized(record):
                raise ValueError(f"liveness GPU is not authorized A800/H20:{uuid}")
            if _first(record, "persistent_model_instances", "evidence.persistent_model_instances") is not True:
                raise ValueError(f"liveness pass lacks persistent-model evidence:{uuid}")
            realized[uuid] = _verify_liveness_pass(record, by_uuid[uuid]["manifest"])
        indexed[uuid], normalized[uuid] = record, item
        launchers.add(item["launcher_sha256"])
        bindings.add(item["contract_binding_sha256"])
    if set(indexed) != set(reference_passed):
        raise ValueError("liveness UUID set is not the fresh reference-pass exact set")
    if len(launchers) != 1 or len(bindings) != 1 or len(allowlist_paths) != 1 or len(allowlist_hashes) != 1:
        raise ValueError("liveness run mixes launcher/binding/allowlist hashes")
    allowlist_path = Path(next(iter(allowlist_paths))).resolve()
    if not allowlist_path.is_file() or _sha256_file(allowlist_path) != next(iter(allowlist_hashes)):
        raise ValueError("liveness allowlist file is missing, changed, or not hash-bound")
    if _read_allowlist(allowlist_path) != reference_passed:
        raise ValueError("liveness allowlist is stale or differs from the fresh reference-pass UUID sequence")
    return indexed, normalized, realized


def _axis_count_rows(
    manifests: Iterable[Mapping[str, Any]],
    reference_passed: set[str],
    liveness_passed: set[str],
) -> dict[str, dict[str, dict[str, int]]]:
    result: dict[str, dict[str, collections.Counter[str]]] = {
        axis: {
            "generated": collections.Counter(),
            "reference_passed": collections.Counter(),
            "liveness_passed": collections.Counter(),
            "accepted": collections.Counter(),
        }
        for axis in AXES
    }
    for manifest in manifests:
        uuid = str(manifest["uuid"])
        for axis in AXES:
            for value in _axis_values(manifest, axis):
                result[axis]["generated"][value] += 1
                if uuid in reference_passed:
                    result[axis]["reference_passed"][value] += 1
                if uuid in liveness_passed:
                    result[axis]["liveness_passed"][value] += 1
                    result[axis]["accepted"][value] += 1
    return {
        axis: {stage: dict(sorted(counter.items())) for stage, counter in stages.items()}
        for axis, stages in result.items()
    }


def _identity_counts(
    manifests: Iterable[Mapping[str, Any]],
    reference_passed: set[str],
    liveness_passed: set[str],
    realized: Mapping[str, set[str]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, collections.Counter[str]] = {
        "generated": collections.Counter(),
        "reference_passed": collections.Counter(),
        "liveness_passed": collections.Counter(),
        "accepted": collections.Counter(),
    }
    for manifest in manifests:
        uuid = str(manifest["uuid"])
        declared = _declared_identities(manifest)
        counts["generated"].update(declared)
        if uuid in reference_passed:
            counts["reference_passed"].update(declared)
        if uuid in liveness_passed:
            counts["liveness_passed"].update(realized[uuid])
            counts["accepted"].update(realized[uuid])
    return {stage: dict(sorted(counter.items())) for stage, counter in counts.items()}


def _failure_audit(
    manifests: Sequence[Mapping[str, Any]],
    reference_raw: Mapping[str, Mapping[str, Any]],
    reference_normalized: Mapping[str, Mapping[str, Any]],
    liveness_raw: Mapping[str, Mapping[str, Any]],
    liveness_normalized: Mapping[str, Mapping[str, Any]],
    accepted: set[str],
) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    for manifest in manifests:
        uuid = str(manifest["uuid"])
        if uuid in accepted:
            continue
        reference = reference_normalized[uuid]
        liveness = liveness_normalized.get(uuid)
        failed = liveness if liveness is not None and liveness["passed"] is False else reference
        raw = liveness_raw.get(uuid) if liveness is not None and liveness["passed"] is False else reference_raw[uuid]
        audit.append(
            {
                "uuid": uuid,
                "row_index": manifest["candidate_row_index"],
                "family": manifest["primary_family"],
                "template_id": manifest["template_id"],
                "coverage_labels": manifest["coverage_labels"],
                "scenario_source": manifest["scenario_source"],
                "failure_stage": failed["failure_stage"],
                "failure_signature": failed["failure_signature"],
                "reason": failed["reason"],
                "reason_sha256": _sha256_bytes(failed["reason"].encode()),
                "reference_record_sha256": _canonical_sha256(reference_raw[uuid]),
                "reference_record": reference_raw[uuid],
                "liveness_record_sha256": _canonical_sha256(liveness_raw[uuid]) if uuid in liveness_raw else None,
                "liveness_record": liveness_raw.get(uuid),
                "failed_raw_record_sha256": _canonical_sha256(raw),
                "failed_raw_record": raw,
            }
        )
    return audit


def _failure_report(
    counts: Mapping[str, Mapping[str, Mapping[str, int]]],
    identity_counts: Mapping[str, Mapping[str, int]],
    audit: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "# Frontier operator canary failure and diversity audit",
        "",
        "Rows remain review-only. Axis rows are multi-label where a coverage label is a list; realized ATen rows use declared identities for generated/reference counts and only `per_declared_op.matched_identities` with `returned_output_witness=true` for liveness/accepted counts.",
    ]
    for axis in AXES:
        lines.extend(
            [
                "",
                f"## {axis}",
                "",
                "| value | generated | reference pass | liveness pass | accepted |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        keys = sorted(set().union(*(counts[axis][stage] for stage in counts[axis])))
        for key in keys:
            lines.append(
                "| {} | {} | {} | {} | {} |".format(
                    key.replace("|", "\\|"),
                    counts[axis]["generated"].get(key, 0),
                    counts[axis]["reference_passed"].get(key, 0),
                    counts[axis]["liveness_passed"].get(key, 0),
                    counts[axis]["accepted"].get(key, 0),
                )
            )
    lines.extend(
        [
            "",
            "## realized_aten_identity",
            "",
            "| identity | generated (declared) | reference pass (declared) | liveness pass (realized) | accepted (realized) |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    identity_keys = sorted(set().union(*identity_counts.values()))
    for key in identity_keys:
        lines.append(
            "| {} | {} | {} | {} | {} |".format(
                key.replace("|", "\\|"),
                identity_counts["generated"].get(key, 0),
                identity_counts["reference_passed"].get(key, 0),
                identity_counts["liveness_passed"].get(key, 0),
                identity_counts["accepted"].get(key, 0),
            )
        )
    signatures = collections.Counter(str(item["failure_signature"]) for item in audit)
    lines.extend(
        [
            "",
            "## Failure attribution",
            "",
            f"Non-accepted rows: {len(audit)}.",
            "",
            "| failure signature | rows |",
            "| --- | ---: |",
        ]
    )
    lines.extend(f"| `{signature}` | {count} |" for signature, count in sorted(signatures.items()))
    lines.append("")
    return "\n".join(lines)


def _runtime_sample_report(
    manifests: Sequence[Mapping[str, Any]],
    reference_raw: Mapping[str, Mapping[str, Any]],
    liveness_raw: Mapping[str, Mapping[str, Any]],
    realized: Mapping[str, set[str]],
    accepted: set[str],
    audit: Sequence[Mapping[str, Any]],
) -> str:
    """Render deterministic, human-reviewable runtime examples per family."""

    lines = [
        "# Frontier operator runtime evidence samples",
        "",
        "This report selects the first accepted row in candidate order for every family and one raw example for every distinct family/failure signature. Realized identities are only declared-op `matched_identities` carrying `returned_output_witness=true`; full dispatch histograms are deliberately excluded. It is review evidence, not training approval.",
    ]
    families = list(generator.FAMILY_QUOTAS)
    for family in families:
        lines.extend(["", f"## {family}", ""])
        sample = next(
            (
                manifest
                for manifest in manifests
                if manifest["primary_family"] == family and manifest["uuid"] in accepted
            ),
            None,
        )
        if sample is None:
            lines.append("No accepted row in this family.")
        else:
            uuid = str(sample["uuid"])
            liveness = liveness_raw[uuid]
            trials = _first(liveness, "trials", "evidence.trials", default=[])
            first_trace = trials[0].get("trace", {}) if isinstance(trials, list) and trials else {}
            per_op = first_trace.get("per_declared_op", []) if isinstance(first_trace, Mapping) else []
            lines.extend(
                [
                    f"Accepted UUID: `{uuid}`; row `{sample['candidate_row_index']}`; template `{sample['template_id']}`.",
                    "",
                    f"Reference record SHA-256: `{_canonical_sha256(reference_raw[uuid])}`.",
                    "",
                    f"Liveness record SHA-256: `{_canonical_sha256(liveness)}`.",
                    "",
                    "Realized returned-output identities:",
                    "",
                    "```json",
                    json.dumps(sorted(realized[uuid]), ensure_ascii=False),
                    "```",
                    "",
                    "First-trial declared-op witnesses:",
                    "",
                    "```json",
                    json.dumps(per_op, ensure_ascii=False, indent=2, sort_keys=True),
                    "```",
                ]
            )
        failures: dict[str, Mapping[str, Any]] = {}
        for item in audit:
            if item["family"] == family:
                failures.setdefault(str(item["failure_signature"]), item)
        lines.extend(["", "Failure-signature examples:", ""])
        if not failures:
            lines.append("None.")
        else:
            for signature, item in sorted(failures.items()):
                lines.append(
                    f"- `{signature}`: UUID `{item['uuid']}`, stage `{item['failure_stage']}`, "
                    f"reason `{item['reason']}`, raw SHA-256 `{item['failed_raw_record_sha256']}`."
                )
    lines.append("")
    return "\n".join(lines)


def analyze(
    candidates_path: Path, manifest_path: Path, reference_dir: Path, liveness_dir: Path, output_dir: Path
) -> dict[str, Any]:
    candidates_path, manifest_path, output_dir = (
        candidates_path.resolve(),
        manifest_path.resolve(),
        output_dir.resolve(),
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"analysis output directory is not empty:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    table, manifests, by_uuid, unique_semantic_coordinate_cells = _verify_static(candidates_path, manifest_path)
    reference_records, reference_shards = _collect_shards(reference_dir)
    references, reference_normalized = _verify_reference(reference_records, by_uuid, candidates_path, manifest_path)
    reference_passed = {uuid for uuid, item in reference_normalized.items() if item["passed"] is True}
    liveness_records, liveness_shards = _collect_shards(liveness_dir)
    liveness, liveness_normalized, realized = _verify_liveness(
        liveness_records, by_uuid, reference_normalized, candidates_path, manifest_path
    )
    accepted = {uuid for uuid, item in liveness_normalized.items() if item["passed"] is True}
    if not accepted.issubset(reference_passed):
        raise ValueError("accepted liveness set is not a reference-pass subset")
    accepted_indices = [index for index, item in enumerate(manifests) if item["uuid"] in accepted]
    accepted_path = output_dir / "accepted.parquet"
    accepted_manifest_path = output_dir / "accepted.manifest.jsonl"
    failure_audit_path = output_dir / "failure_audit.jsonl"
    failure_report_path = output_dir / "failure_bias_report.md"
    runtime_samples_path = output_dir / "runtime_evidence_samples.md"
    raw_hash_path = output_dir / "raw_artifact_sha256.json"
    pq.write_table(table.take(pa.array(accepted_indices, type=pa.int64())), accepted_path, compression="zstd")
    accepted_manifests: list[dict[str, Any]] = []
    for manifest in manifests:
        uuid = str(manifest["uuid"])
        if uuid not in accepted:
            continue
        updated = copy.deepcopy(manifest)
        updated.update(
            {
                "reference_runtime_status": "passed",
                "operator_liveness_status": "passed",
                "runtime_status": ACCEPTED_RUNTIME_STATUS,
                "governance_status": ACCEPTED_GOVERNANCE_STATUS,
                "training_approved": False,
                "runtime_evidence": {
                    "reference_record_sha256": _canonical_sha256(references[uuid]),
                    "liveness_record_sha256": _canonical_sha256(liveness[uuid]),
                    "realized_returned_output_aten_identities": sorted(realized[uuid]),
                    "realized_aten_identity_definition": "union across liveness trials of per_declared_op.matched_identities where returned_output_witness is true",
                },
            }
        )
        accepted_manifests.append(updated)
    with accepted_manifest_path.open("w", encoding="utf-8") as handle:
        for item in accepted_manifests:
            handle.write(_canonical_json(item) + "\n")
    audit = _failure_audit(manifests, references, reference_normalized, liveness, liveness_normalized, accepted)
    with failure_audit_path.open("w", encoding="utf-8") as handle:
        for item in audit:
            handle.write(_canonical_json(item) + "\n")
    counts = _axis_count_rows(manifests, reference_passed, accepted)
    identity_counts = _identity_counts(manifests, reference_passed, accepted, realized)
    failure_report_path.write_text(_failure_report(counts, identity_counts, audit), encoding="utf-8")
    runtime_samples_path.write_text(
        _runtime_sample_report(manifests, references, liveness, realized, accepted, audit), encoding="utf-8"
    )
    closure_files = {
        "analyzer": _module_closure(sys.modules[__name__]),
        "generator": _module_closure(generator),
        "reference_validator": _module_closure(reference_validator),
        "liveness_validator": _module_closure(liveness_validator),
    }
    closure = {
        name: {"files": files, "closure_sha256": _canonical_sha256(files)} for name, files in closure_files.items()
    }
    raw_hashes = {
        "candidates": {"path": str(candidates_path), "sha256": _sha256_file(candidates_path)},
        "manifest": {"path": str(manifest_path), "sha256": _sha256_file(manifest_path)},
        "reference_shards": reference_shards,
        "liveness_shards": liveness_shards,
        "source_closure": closure,
    }
    raw_hash_path.write_text(json.dumps(raw_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {
        "analysis_contract": ANALYSIS_CONTRACT,
        "max_authorized_candidates": MAX_AUTHORIZED_CANDIDATES,
        "generated": len(manifests),
        "reference_executed": len(references),
        "reference_passed": len(reference_passed),
        "reference_failed": len(manifests) - len(reference_passed),
        "liveness_executed": len(liveness),
        "liveness_passed": len(accepted),
        "liveness_failed": len(liveness) - len(accepted),
        "accepted": len(accepted),
        "unique_semantic_coordinate_cells": unique_semantic_coordinate_cells,
        "semantic_coordinate_cell_identity": "canonical JSON of template_id plus constraint_solver.requested_coordinates after excluding variant",
        "coverage_counts": counts,
        "realized_aten_identity_counts": identity_counts,
        "realized_aten_identity_definition": "generated/reference use declared identities; liveness/accepted use only per_declared_op.matched_identities where returned_output_witness is true",
        "final_output_kind": "single_dense_tensor",
        "structured_output_status": "explicitly_deferred",
        "runtime_status": ACCEPTED_RUNTIME_STATUS,
        "governance_status": ACCEPTED_GOVERNANCE_STATUS,
        "training_approved": False,
        "artifacts": {
            "accepted": {"path": str(accepted_path), "sha256": _sha256_file(accepted_path)},
            "accepted_manifest": {"path": str(accepted_manifest_path), "sha256": _sha256_file(accepted_manifest_path)},
            "failure_audit": {"path": str(failure_audit_path), "sha256": _sha256_file(failure_audit_path)},
            "failure_bias_report": {"path": str(failure_report_path), "sha256": _sha256_file(failure_report_path)},
            "runtime_evidence_samples": {
                "path": str(runtime_samples_path),
                "sha256": _sha256_file(runtime_samples_path),
            },
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
