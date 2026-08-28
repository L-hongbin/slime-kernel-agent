#!/usr/bin/env python3
"""Build deterministic semantic-audit packs for random, dtype, and layout children."""

from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

AUDIT_VERSION = "intervention_semantic_audit_selected_children_v2"
DEFAULT_OUTPUT = REPO_ROOT / "local_artifacts/data/synthesize/intervention_semantic_audit"
DEFAULT_PER_METHOD = 100
DEFAULT_PACK_SIZE = 10

RUNS: dict[str, tuple[Path, ...]] = {
    "random": (REPO_ROOT / "Data/prompt_tvm_v4/random_value_from_shape_v5/run.low128k_final",),
    "dtype": (
        REPO_ROOT / "Data/prompt_tvm_v4/serial_augmentation_v1/dtype/run.parameter_free.5000.v1",
        REPO_ROOT / "Data/prompt_tvm_v4/serial_augmentation_v1/dtype/run.module_state.1000.v1",
        REPO_ROOT / "Data/prompt_tvm_v4/serial_augmentation_v1/dtype/run.parameter_free.500.v2",
        REPO_ROOT / "Data/prompt_tvm_v4/serial_augmentation_v1/dtype/run.module_state.5000.v2.semantic_gate",
        REPO_ROOT / "Data/prompt_tvm_v4/serial_augmentation_v1/dtype/run.module_state.5000.v3.semantic_gate",
        REPO_ROOT / "Data/prompt_tvm_v4/serial_augmentation_v1/dtype/run.module_state.1000.v4.semantic_gate",
    ),
    "layout": (
        REPO_ROOT / "Data/prompt_tvm_v4/serial_augmentation_v1/layout/run.primary.5000.v1",
        REPO_ROOT / "Data/prompt_tvm_v4/serial_augmentation_v1/layout/run.semantic_double.5000.v2",
        REPO_ROOT / "Data/prompt_tvm_v4/serial_augmentation_v1/layout/run.semantic_double.1500.v3",
    ),
}

BASE_MANIFESTS = {
    "random": REPO_ROOT
    / "Data/prompt_tvm_v4/serial_augmentation_v1/random_fallback_base.v3.semantic_gate/manifest.jsonl",
    "dtype": REPO_ROOT
    / "Data/prompt_tvm_v4/serial_augmentation_v1/dtype_fallback_base.v2.semantic_gate_double/manifest.jsonl",
    "layout": REPO_ROOT
    / "Data/prompt_tvm_v4/serial_augmentation_v1/layout_fallback_base.v2.semantic_gate_double/manifest.jsonl",
}

RISK_PATTERNS: dict[str, re.Pattern[str]] = {
    "restricted_domain": re.compile(r"\b(?:log|log2|log10|sqrt|rsqrt|acos|asin|reciprocal)\b", re.I),
    "probability_or_loss": re.compile(
        r"\b(?:softmax|cross_entropy|nll_loss|kl_div|binary_cross_entropy|multinomial)\b", re.I
    ),
    "index_or_mask": re.compile(r"\b(?:embedding|gather|scatter|index|masked|where|topk|sort)\b", re.I),
    "precision_sensitive": re.compile(
        r"\b(?:exp|expm1|cumprod|cumsum|prod|det|inverse|linalg|svd|eig|fft|norm)\b", re.I
    ),
    "normalization": re.compile(
        r"\b(?:batch_norm|layer_norm|group_norm|instance_norm|normalize|mean|var|std)\b", re.I
    ),
    "layout_sensitive": re.compile(
        r"\b(?:view|reshape|flatten|contiguous|stride|as_strided|transpose|permute|matmul|mm|conv)\b", re.I
    ),
    "in_place": re.compile(r"\.(?!__)[A-Za-z]\w*_(?!_)\s*\("),
    "threshold_or_clamp": re.compile(r"\b(?:clamp|relu|threshold|minimum|maximum)\b|(?:<=|>=|==|!=)", re.I),
}


@dataclass(frozen=True)
class AuditItem:
    method: str
    run_id: str
    child_uuid: str
    parent_uuid: str
    source_family: str
    operator_bucket: str
    operator_count: int
    family: str
    coherence_class: str
    factory_count: int
    input_bytes: int
    estimated_peak_bytes: int
    upstream_kind: str
    risk_tags: tuple[str, ...]
    layout_max_ndim: int
    layout_max_stride_ratio: float
    intervention_metadata: Mapping[str, Any]
    parent_code: str
    child_code: str
    stable_hash: str


def _nested(value: Any, path: str, default: Any = None) -> Any:
    current = value
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _stable_hash(*parts: object) -> str:
    return hashlib.sha256(":".join(str(part) for part in parts).encode()).hexdigest()


def _load_jsonl(path: Path, key: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            identity = record.get(key)
            if not isinstance(identity, str) or not identity or identity in result:
                raise ValueError(f"invalid or duplicate {key} at {path}:{line_number}")
            result[identity] = record
    return result


def _parquet_rows_by_uuid(path: Path) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
        for row in batch.to_pylist():
            uuid = _nested(row, "extra_info.uuid")
            if not isinstance(uuid, str) or uuid in result:
                raise ValueError(f"invalid or duplicate UUID in {path}: {uuid!r}")
            result[uuid] = row
    return result


def _reference(row: Mapping[str, Any]) -> str:
    code = _nested(row, "reward_model.ground_truth")
    if not isinstance(code, str):
        raise ValueError("row has no reference source")
    return code


def _method_family(method: str, manifest: Mapping[str, Any]) -> str:
    if method == "random":
        return str(manifest["assigned_target"])
    if method == "dtype":
        return str(manifest["assigned_target"])
    return str(manifest["assigned_family"])


def _upstream_kind(parent_uuid: str) -> str:
    for prefix in ("dtype_", "value_", "shapeai_", "shapesolver_"):
        if parent_uuid.startswith(prefix):
            return prefix.removesuffix("_")
    return "other"


def _risk_tags(code: str) -> tuple[str, ...]:
    return tuple(name for name, pattern in RISK_PATTERNS.items() if pattern.search(code))


def _layout_geometry(manifest: Mapping[str, Any]) -> tuple[int, float]:
    metadata = manifest.get("expected_layout_metadata")
    if not isinstance(metadata, list):
        return 0, 0.0
    maximum_rank = 0
    maximum_ratio = 0.0
    for record in metadata:
        if not isinstance(record, Mapping):
            continue
        shape = record.get("logical_shape")
        strides = record.get("expected_strides")
        if isinstance(shape, list):
            maximum_rank = max(maximum_rank, len(shape))
        if isinstance(strides, list):
            positive = [abs(int(value)) for value in strides if type(value) is int and value]
            if positive:
                maximum_ratio = max(maximum_ratio, max(positive) / min(positive))
    return maximum_rank, maximum_ratio


def _intervention_metadata(method: str, manifest: Mapping[str, Any]) -> dict[str, Any]:
    shared = (
        "primary_intervention",
        "transformed_factory_count",
        "shape_changed",
        "value_changed",
        "dtype_changed",
        "layout_changed",
        "parent_runtime_status",
        "child_runtime_status",
        "liveness_status",
        "semantic_promotion_status",
        "semantic_gate",
    )
    method_fields = {
        "random": ("assigned_target", "realized_intervention", "value_labels"),
        "dtype": (
            "assigned_target",
            "coherence_class",
            "factory_specs_before",
            "explicit_model_casts",
            "model_parameter_count",
            "model_buffer_count",
            "realized_intervention",
        ),
        "layout": (
            "assigned_family",
            "eligible_families",
            "target_factory_indices",
            "expected_layout_metadata",
            "realized_intervention",
        ),
    }
    return {key: manifest[key] for key in (*shared, *method_fields[method]) if key in manifest}


def _load_run(method: str, run_dir: Path) -> list[AuditItem]:
    manifests = _load_jsonl(run_dir / "runtime/accepted.manifest.jsonl", "child_uuid")
    parents = _parquet_rows_by_uuid(run_dir / "parents.parquet")
    children = _parquet_rows_by_uuid(run_dir / "runtime/accepted.parquet")
    if set(children) != set(manifests):
        raise ValueError(f"accepted parquet/manifest UUID mismatch: {run_dir}")
    result: list[AuditItem] = []
    for child_uuid, child_row in children.items():
        manifest = manifests[child_uuid]
        parent_uuid = manifest.get("parent_uuid")
        if not isinstance(parent_uuid, str) or parent_uuid not in parents:
            raise ValueError(f"accepted child has no selected parent: {child_uuid}:{parent_uuid}")
        parent_code = _reference(parents[parent_uuid])
        child_code = _reference(child_row)
        if _sha256_text(parent_code) != manifest.get("parent_reference_sha256"):
            raise ValueError(f"parent source hash mismatch: {parent_uuid}")
        if _sha256_text(child_code) != manifest.get("child_reference_sha256"):
            raise ValueError(f"child source hash mismatch: {child_uuid}")
        maximum_rank, maximum_stride_ratio = _layout_geometry(manifest)
        result.append(
            AuditItem(
                method=method,
                run_id=run_dir.name,
                child_uuid=child_uuid,
                parent_uuid=parent_uuid,
                source_family=str(manifest.get("source_family", "unknown")),
                operator_bucket=str(manifest.get("operator_bucket", "unknown")),
                operator_count=int(manifest.get("operator_count", 0)),
                family=_method_family(method, manifest),
                coherence_class=str(manifest.get("coherence_class", "n/a")),
                factory_count=int(manifest.get("factory_count", 0)),
                input_bytes=int(manifest.get("input_bytes_before") or 0),
                estimated_peak_bytes=int(manifest.get("estimated_peak_bytes") or 0),
                upstream_kind=_upstream_kind(parent_uuid),
                risk_tags=_risk_tags(parent_code),
                layout_max_ndim=maximum_rank,
                layout_max_stride_ratio=maximum_stride_ratio,
                intervention_metadata=_intervention_metadata(method, manifest),
                parent_code=parent_code,
                child_code=child_code,
                stable_hash=_stable_hash(AUDIT_VERSION, method, child_uuid),
            )
        )
    return result


def load_population() -> dict[str, list[AuditItem]]:
    result: dict[str, list[AuditItem]] = {}
    for method, runs in RUNS.items():
        items = [item for run in runs for item in _load_run(method, run)]
        if len(items) != len({item.child_uuid for item in items}):
            raise ValueError(f"duplicate accepted child across {method} runs")
        manifest_rows = [
            json.loads(line)
            for line in BASE_MANIFESTS[method].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        child_field = "random_candidate_uuid" if method == "random" else "stage_child_uuid"
        selected_rows = {
            str(row[child_field]): row
            for row in manifest_rows
            if row.get("selected_layer") == method and isinstance(row.get(child_field), str)
        }
        selected = set(selected_rows)
        by_uuid = {item.child_uuid: item for item in items}
        missing = selected - set(by_uuid)
        if missing:
            raise ValueError(f"selected {method} children missing accepted evidence: {sorted(missing)[:3]}")
        selected_items: list[AuditItem] = []
        for uuid in sorted(selected):
            item = by_uuid[uuid]
            current_gate = selected_rows[uuid].get("semantic_gate")
            if not isinstance(current_gate, Mapping) or current_gate.get("status") != "passed":
                raise ValueError(f"selected {method} child lacks current-policy semantic promotion: {uuid}")
            metadata = dict(item.intervention_metadata)
            metadata["runtime_semantic_gate"] = metadata.get("semantic_gate")
            metadata["semantic_gate"] = dict(current_gate)
            metadata["semantic_gate_revalidated_at_composition"] = True
            selected_items.append(replace(item, intervention_metadata=metadata))
        result[method] = selected_items
    return result


def _risk_orders(method: str, items: Sequence[AuditItem]) -> tuple[list[AuditItem], ...]:
    common = (
        sorted(items, key=lambda item: (-len(item.risk_tags), item.stable_hash)),
        sorted(items, key=lambda item: (-item.factory_count, item.stable_hash)),
        sorted(items, key=lambda item: (-item.operator_count, item.stable_hash)),
    )
    if method == "random":
        return common + (sorted(items, key=lambda item: (-item.input_bytes, item.stable_hash)),)
    if method == "dtype":
        return common + (
            sorted(items, key=lambda item: (item.coherence_class != "module_state", item.stable_hash)),
            sorted(items, key=lambda item: (-item.estimated_peak_bytes, item.stable_hash)),
        )
    return common + (
        sorted(items, key=lambda item: (-item.layout_max_ndim, item.stable_hash)),
        sorted(items, key=lambda item: (-item.layout_max_stride_ratio, item.stable_hash)),
    )


def select_discovery_sample(method: str, items: Sequence[AuditItem], count: int) -> list[AuditItem]:
    if count > len(items):
        raise ValueError(f"sample count exceeds {method} population")
    selected: dict[str, AuditItem] = {}

    def add(item: AuditItem) -> None:
        if len(selected) < count:
            selected.setdefault(item.child_uuid, item)

    if method == "layout":
        for item in items:
            if item.family == "expand_zero_stride":
                add(item)

    anchor_count = max(8, count // 10)
    for ordered in _risk_orders(method, items):
        for item in ordered[:anchor_count]:
            add(item)

    strata: dict[tuple[str, str], list[AuditItem]] = collections.defaultdict(list)
    for item in items:
        fields = {
            "run": item.run_id,
            "family": item.family,
            "source": item.source_family,
            "operator_bucket": item.operator_bucket,
            "coherence": item.coherence_class,
            "factory_count": str(item.factory_count),
            "upstream_kind": item.upstream_kind,
        }
        for name, value in fields.items():
            strata[(name, value)].append(item)
        for tag in item.risk_tags:
            strata[("risk_tag", tag)].append(item)
    ordered_strata = sorted(strata, key=lambda key: (_stable_hash("stratum", method, *key), key))
    for values in strata.values():
        values.sort(key=lambda item: item.stable_hash)
    cursor = 0
    while len(selected) < count:
        progress = False
        for key in ordered_strata:
            options = strata[key]
            if cursor < len(options):
                before = len(selected)
                add(options[cursor])
                progress |= len(selected) > before
                if len(selected) == count:
                    break
        if not progress and cursor >= max(len(values) for values in strata.values()):
            break
        cursor += 1
    for item in sorted(items, key=lambda item: item.stable_hash):
        add(item)
    return sorted(selected.values(), key=lambda item: item.stable_hash)


def _diff(parent: str, child: str) -> str:
    return "\n".join(
        difflib.unified_diff(
            parent.splitlines(),
            child.splitlines(),
            fromfile="parent.py",
            tofile="child.py",
            lineterm="",
        )
    )


def _method_instructions(method: str) -> list[str]:
    common = [
        "逐条阅读 parent reference 和完整 diff；GPU-valid 只证明可执行和有限 trial 下有输出影响。",
        "评级：A=语义闭合且有明确覆盖价值；B=可运行但代表性、domain、coupling 或有效作用可疑；C=reference 可直接证明 child 引入语义冲突；D=证据不足。",
        "必须区分 child 新引入问题与 parent 原有问题。抽样 risk tags 只是静态选样特征，不是预先判错。",
    ]
    specific = {
        "random": "重点检查 value family 是否符合 use-site domain，特别是 category/count、符号、概率、index、loss、normalization 和 threshold。",
        "dtype": "重点检查 FP16/BF16 的数值范围、常量/阈值、reduction/exp/norm、registered state、显式 cast 和输出任务语义。",
        "layout": "重点检查 stride/offset/zero-stride 是否到达有意义的 consumer，是否立刻 materialize，以及 view/reshape/in-place/alias 和 axis 语义。",
    }
    return common + [specific[method]]


def _render_pack(method: str, pack_id: int, items: Sequence[AuditItem]) -> str:
    lines = [f"# {method} semantic audit pack {pack_id:02d}", ""]
    lines.extend(f"- {line}" for line in _method_instructions(method))
    lines.extend(
        [
            "",
            "输出一个 Markdown 表格：child UUID、A/B/C/D、问题来源（child/parent/uncertain）、问题标签、一句话理由；随后只对 B/C/D 写必要证据。",
            "",
        ]
    )
    for index, item in enumerate(items, start=1):
        lines.extend(
            [
                f"## {index}. `{item.child_uuid}`",
                "",
                f"- Parent: `{item.parent_uuid}`; upstream: `{item.upstream_kind}`",
                f"- Run: `{item.run_id}`; family: `{item.family}`; coherence: `{item.coherence_class}`",
                f"- Source: `{item.source_family}`; operator bucket/count: `{item.operator_bucket}` / `{item.operator_count}`",
                f"- Factories: `{item.factory_count}`; input bytes: `{item.input_bytes}`; estimated peak bytes: `{item.estimated_peak_bytes}`",
                f"- Layout max ndim/stride ratio: `{item.layout_max_ndim}` / `{item.layout_max_stride_ratio:.3f}`",
                f"- Sampling risk tags: `{', '.join(item.risk_tags) or 'none'}`",
                "",
                "### Intervention metadata",
                "",
                "```json",
                json.dumps(item.intervention_metadata, indent=2, sort_keys=True),
                "```",
                "",
                "### Parent reference",
                "",
                "```python",
                item.parent_code.rstrip(),
                "```",
                "",
                "### Parent → child diff",
                "",
                "```diff",
                _diff(item.parent_code, item.child_code),
                "```",
                "",
                "### Child reference",
                "",
                "```python",
                item.child_code.rstrip(),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def _write_manifest(path: Path, items: Sequence[AuditItem]) -> None:
    columns = (
        "method",
        "child_uuid",
        "parent_uuid",
        "run_id",
        "family",
        "coherence_class",
        "source_family",
        "operator_bucket",
        "operator_count",
        "factory_count",
        "input_bytes",
        "estimated_peak_bytes",
        "upstream_kind",
        "risk_tags",
        "stable_hash",
    )
    lines = ["\t".join(columns)]
    for item in items:
        values = {
            "method": item.method,
            "child_uuid": item.child_uuid,
            "parent_uuid": item.parent_uuid,
            "run_id": item.run_id,
            "family": item.family,
            "coherence_class": item.coherence_class,
            "source_family": item.source_family,
            "operator_bucket": item.operator_bucket,
            "operator_count": str(item.operator_count),
            "factory_count": str(item.factory_count),
            "input_bytes": str(item.input_bytes),
            "estimated_peak_bytes": str(item.estimated_peak_bytes),
            "upstream_kind": item.upstream_kind,
            "risk_tags": ",".join(item.risk_tags),
            "stable_hash": item.stable_hash,
        }
        lines.append("\t".join(values[column] for column in columns))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(output: Path, per_method: int, pack_size: int) -> dict[str, Any]:
    if per_method <= 0 or pack_size <= 0 or per_method % pack_size:
        raise ValueError("per-method must be positive and divisible by pack-size")
    population = load_population()
    output.mkdir(parents=True, exist_ok=True)
    assignments = output / "assignments"
    assignments.mkdir(exist_ok=True)
    summary: dict[str, Any] = {"audit_version": AUDIT_VERSION, "methods": {}}
    for method, items in population.items():
        sample = select_discovery_sample(method, items, per_method)
        method_dir = output / method
        method_dir.mkdir(exist_ok=True)
        _write_manifest(method_dir / "manifest.tsv", sample)
        for pack_index, start in enumerate(range(0, len(sample), pack_size)):
            pack = sample[start : start + pack_size]
            (assignments / f"{method}_{pack_index:02d}.md").write_text(
                _render_pack(method, pack_index, pack), encoding="utf-8"
            )
        summary["methods"][method] = {
            "population": len(items),
            "sample": len(sample),
            "packs": len(sample) // pack_size,
            "family_population": dict(collections.Counter(item.family for item in items)),
            "family_sample": dict(collections.Counter(item.family for item in sample)),
            "source_sample": dict(collections.Counter(item.source_family for item in sample)),
            "risk_tag_sample": dict(collections.Counter(tag for item in sample for tag in item.risk_tags)),
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-method", type=int, default=DEFAULT_PER_METHOD)
    parser.add_argument("--pack-size", type=int, default=DEFAULT_PACK_SIZE)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    print(json.dumps(run(args.output, args.per_method, args.pack_size), indent=2, sort_keys=True))
