#!/usr/bin/env python3
"""Generate a small, deterministic loss/distance coverage lane.

The lane covers the three loss cells present in KernelBench: cross entropy,
Smooth L1, and triplet margin.  KernelBench source is used only as a
decontamination root; every task is rendered from the project-owned registry.
"""

from __future__ import annotations

import argparse
import ast
import collections
import dataclasses
import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.cleaning.complexity import (  # noqa: E402
    extract_complexity_features,
    extract_operator_signature,
    feature_dict,
)
from tools.data.cleaning.external import _template  # noqa: E402
from tools.data.cleaning.static_analysis import _call_name  # noqa: E402

CONTRACT_VERSION = "loss_distance_closed_registry_v1"
MANIFEST_VERSION = "loss_distance_manifest_v1"
REGISTRY_VERSION = "loss_distance_kernelbench_support_3x4_v1"
RUNTIME_CONTRACT_VERSION = "aten_dispatch_return_provenance_v1"
METHOD = "loss_distance_canary"
DATA_SOURCE = "project_generated_loss_distance_v1"
EXACT_ROWS = 12


@dataclasses.dataclass(frozen=True)
class TemplateSpec:
    template_id: str
    primary_family: str
    kind: str
    input_arity: int
    source_call: str
    variants: int = 4


TEMPLATES = (
    TemplateSpec("LD01", "loss_cross_entropy", "cross_entropy", 2, "F.cross_entropy"),
    TemplateSpec("LD02", "loss_smooth_l1", "smooth_l1", 2, "F.smooth_l1_loss"),
    TemplateSpec("LD03", "loss_triplet_margin", "triplet_margin", 3, "F.triplet_margin_loss"),
)
TEMPLATE_BY_ID = {item.template_id: item for item in TEMPLATES}


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


def _normalized_ast_sha256(code: str) -> str:
    return _sha256_bytes(ast.dump(ast.parse(code), include_attributes=False).encode())


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True).strip()


def _op(op_id: str, source_call: str, *identities: tuple[str, str]) -> dict[str, Any]:
    return {
        "op_id": op_id,
        "source_calls": [source_call],
        "runtime_identities": [{"schema": schema, "overload": overload} for schema, overload in identities],
        "min_calls_per_trial": 1,
        "must_reach_returned_output": True,
    }


def _continuous_input(name: str, batch: int, width: int, variant: int) -> list[str]:
    if variant == 0:
        return [f"{name} = torch.rand({batch}, {width})"]
    if variant == 1:
        return [f"storage_{name} = torch.rand({batch + 1}, {width})", f"{name} = storage_{name}[1:]"]
    if variant == 2:
        return [f"storage_{name} = torch.rand({batch}, {width * 2})", f"{name} = storage_{name}[:, ::2]"]
    return [
        f"storage_{name} = torch.rand(1, {width})",
        f"{name} = storage_{name}.expand({batch}, {width}).clone()",
    ]


def _source(spec: TemplateSpec, variant: int) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    batch = 4 + variant
    width = (24, 32, 40, 48)[variant]
    hidden = width * 2
    classes = (7, 11, 13, 17)[variant]
    inputs: list[str] = []
    if spec.kind == "cross_entropy":
        inputs.extend(_continuous_input("features", batch, width, variant))
        inputs.extend(
            [
                f"targets = torch.randint(0, {classes}, ({batch},), dtype=torch.int64)",
                "return [features, targets]",
            ]
        )
        model_forward = [
            "encoded = self.encoder(features)",
            "encoded = self.pre_norm(encoded)",
            "refined = F.gelu(self.refine1(encoded))",
            "gate = torch.sigmoid(self.gate(encoded))",
            "encoded = encoded + self.refine2(refined * gate)",
            "encoded = F.silu(encoded)",
            "logits = self.classifier(encoded)",
            'return F.cross_entropy(logits, targets, reduction="mean")',
        ]
        model_init = [
            f"self.encoder = FeatureProjector({width}, {hidden})",
            f"self.pre_norm = nn.LayerNorm({width})",
            f"self.refine1 = nn.Linear({width}, {hidden})",
            f"self.refine2 = nn.Linear({hidden}, {width})",
            f"self.gate = nn.Linear({width}, {hidden})",
            f"self.classifier = nn.Linear({width}, {classes})",
        ]
        declared = [_op("cross_entropy", spec.source_call, ("aten::_log_softmax", ""), ("aten::nll_loss_forward", ""))]
        input_contract = {
            "arguments": ["floating_features", "int64_class_indices"],
            "target_range": [0, classes - 1],
            "reduction": "mean",
            "output_rank": 0,
        }
    elif spec.kind == "smooth_l1":
        inputs.extend(_continuous_input("features", batch, width, variant))
        inputs.extend(_continuous_input("targets", batch, width, (variant + 1) % 4))
        inputs.append("return [features, targets]")
        model_forward = [
            "prediction = self.encoder(features)",
            "prediction = self.pre_norm(prediction)",
            "refined = F.gelu(self.refine1(prediction))",
            "gate = torch.sigmoid(self.gate(prediction))",
            "prediction = prediction + self.refine2(refined * gate)",
            "prediction = F.silu(prediction)",
            "target = self.target_norm(targets)",
            "target = target + 0.0625 * torch.sin(target)",
            'return F.smooth_l1_loss(prediction, target, beta=0.5, reduction="mean")',
        ]
        model_init = [
            f"self.encoder = FeatureProjector({width}, {hidden})",
            f"self.pre_norm = nn.LayerNorm({width})",
            f"self.refine1 = nn.Linear({width}, {hidden})",
            f"self.refine2 = nn.Linear({hidden}, {width})",
            f"self.gate = nn.Linear({width}, {hidden})",
            f"self.target_norm = nn.LayerNorm({width})",
        ]
        declared = [_op("smooth_l1", spec.source_call, ("aten::smooth_l1_loss", ""))]
        input_contract = {
            "arguments": ["floating_prediction_features", "floating_targets"],
            "shape_relation": "equal",
            "reduction": "mean",
            "output_rank": 0,
        }
    else:
        for ordinal, name in enumerate(("anchor", "positive", "negative")):
            inputs.extend(_continuous_input(name, batch, width, (variant + ordinal) % 4))
        inputs.append("return [anchor, positive, negative]")
        model_forward = [
            "anchor = F.normalize(self.encoder(anchor), dim=-1)",
            "positive = F.normalize(self.encoder(positive), dim=-1)",
            "negative = F.normalize(self.encoder(negative), dim=-1)",
            "anchor = self.pre_norm(anchor)",
            "refined = F.gelu(self.refine1(anchor))",
            "gate = torch.sigmoid(self.gate(anchor))",
            "anchor = anchor + self.refine2(refined * gate)",
            "anchor = self.output_norm(anchor)",
            'return F.triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2, reduction="mean")',
        ]
        model_init = [
            f"self.encoder = FeatureProjector({width}, {hidden})",
            f"self.pre_norm = nn.LayerNorm({width})",
            f"self.refine1 = nn.Linear({width}, {hidden})",
            f"self.refine2 = nn.Linear({hidden}, {width})",
            f"self.gate = nn.Linear({width}, {hidden})",
            f"self.output_norm = nn.LayerNorm({width})",
        ]
        declared = [_op("triplet_margin", spec.source_call, ("aten::clamp_min", ""))]
        input_contract = {
            "arguments": ["floating_anchor", "floating_positive", "floating_negative"],
            "shape_relation": "all_equal",
            "reduction": "mean",
            "output_rank": 0,
        }
    code = "\n".join(
        [
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            "",
            "class FeatureProjector(nn.Module):",
            "    def __init__(self, width, hidden):",
            "        super().__init__()",
            "        self.input_norm = nn.LayerNorm(width)",
            "        self.up = nn.Linear(width, hidden)",
            "        self.gate = nn.Linear(width, hidden)",
            "        self.down = nn.Linear(hidden, width)",
            "        self.output_norm = nn.LayerNorm(width)",
            "",
            "    def forward(self, x):",
            "        residual = x",
            "        normalized = self.input_norm(x)",
            "        value = F.gelu(self.up(normalized))",
            "        gate = torch.sigmoid(self.gate(normalized))",
            "        mixed = value * gate",
            "        projected = self.down(mixed)",
            "        projected = projected + residual",
            "        return self.output_norm(projected)",
            "",
            "class Model(nn.Module):",
            "    def __init__(self):",
            "        super().__init__()",
            *(f"        {line}" for line in model_init),
            "",
            f"    def forward(self, {', '.join(['features', 'targets'] if spec.input_arity == 2 else ['anchor', 'positive', 'negative'])}):",
            *(f"        {line}" for line in model_forward),
            "",
            "def get_inputs():",
            *(f"    {line}" for line in inputs),
            "",
            "def get_init_inputs():",
            "    return []",
            "",
        ]
    )
    return code, declared, input_contract


def _static_contract(code: str, spec: TemplateSpec, input_contract: Mapping[str, Any]) -> dict[str, Any]:
    tree = ast.parse(code)
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    if set(classes) != {"FeatureProjector", "Model"}:
        raise ValueError(f"unexpected class set:{sorted(classes)}")
    model_forwards = [
        node for node in classes["Model"].body if isinstance(node, ast.FunctionDef) and node.name == "forward"
    ]
    get_inputs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_inputs"]
    if len(model_forwards) != 1 or len(get_inputs) != 1:
        raise ValueError("Model.forward and get_inputs must be unique")
    calls = {_call_name(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    if spec.source_call not in calls:
        raise ValueError(f"loss source call absent:{spec.source_call}")
    factories = {_call_name(node.func) for node in ast.walk(get_inputs[0]) if isinstance(node, ast.Call)}
    expected_factories = {"torch.rand", "torch.randint"} if spec.kind == "cross_entropy" else {"torch.rand"}
    if not expected_factories.issubset(factories):
        raise ValueError(f"input factory contract missing:{spec.template_id}:{sorted(factories)}")
    signature = list(extract_operator_signature(code))
    complexity = feature_dict(extract_complexity_features(code))
    minimums = {
        "forward_call_count": 10,
        "source_line_count": 50,
        "init_nn_constructor_count": 5,
        "top_level_class_count": 2,
    }
    if any(int(complexity[name]) < minimum for name, minimum in minimums.items()):
        raise ValueError(f"high-complexity contract missing:{spec.template_id}:{complexity}")
    return {
        "operator_signature": signature,
        "complexity": {name: int(complexity[name]) for name in minimums},
        "input_factories": sorted(factories),
        "input_arity": spec.input_arity,
        "input_contract": dict(input_contract),
        "reachable_top_level_classes": ["FeatureProjector", "Model"],
        "single_scalar_tensor_output": True,
    }


def replay_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    spec = TEMPLATE_BY_ID[str(manifest["template_id"])]
    variant = int(manifest["template_variant"])
    if not 0 <= variant < spec.variants:
        raise ValueError("template variant out of range")
    code, declared_ops, input_contract = _source(spec, variant)
    static = _static_contract(code, spec, input_contract)
    return {"code": code, "declared_ops": declared_ops, "input_contract": input_contract, "static_proof": static}


def _schema() -> pa.Schema:
    message = pa.struct([pa.field("content", pa.string()), pa.field("role", pa.string())])
    return pa.schema(
        [
            pa.field("data_source", pa.large_string()),
            pa.field("prompt", pa.list_(message)),
            pa.field("ability", pa.large_string()),
            pa.field(
                "reward_model", pa.struct([pa.field("ground_truth", pa.string()), pa.field("style", pa.string())])
            ),
            pa.field(
                "extra_info",
                pa.struct(
                    [
                        pa.field("entry_point", pa.string()),
                        pa.field("level", pa.string()),
                        pa.field("module_name", pa.string()),
                        pa.field("ops", pa.string()),
                        pa.field("original_prompt", pa.list_(message)),
                        pa.field("repo_name", pa.string()),
                        pa.field("type", pa.string()),
                        pa.field("uuid", pa.string()),
                    ]
                ),
            ),
        ]
    )


def _comparison_hashes(paths: Sequence[Path]) -> tuple[set[str], set[str], list[dict[str, Any]]]:
    references: set[str] = set()
    ast_hashes: set[str] = set()
    bindings: list[dict[str, Any]] = []
    for raw in paths:
        path = raw.expanduser().resolve()
        table = pq.read_table(path, columns=["reward_model"])
        for item in table.column(0).to_pylist():
            code = item["ground_truth"]
            references.add(_sha256_bytes(code.encode()))
            ast_hashes.add(_normalized_ast_sha256(code))
        bindings.append({"path": str(path), "sha256": _sha256_file(path), "rows": table.num_rows})
    return references, ast_hashes, bindings


def generate(output_dir: Path, template_path: Path, comparison_paths: Sequence[Path]) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    _, prompt_prefix = _template(template_path)
    comparison_refs, comparison_asts, comparison_bindings = _comparison_hashes(comparison_paths)
    generator_sha = _sha256_file(Path(__file__).resolve())
    template_binding = {"path": str(template_path.resolve()), "sha256": _sha256_file(template_path)}
    pending: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for spec in TEMPLATES:
        for variant in range(spec.variants):
            code, declared_ops, input_contract = _source(spec, variant)
            static = _static_contract(code, spec, input_contract)
            reference_sha = _sha256_bytes(code.encode())
            normalized_sha = _normalized_ast_sha256(code)
            if reference_sha in comparison_refs or normalized_sha in comparison_asts:
                raise ValueError(f"decontamination collision:{spec.template_id}:{variant}")
            uuid = "lossdist_" + _sha256_bytes(f"{CONTRACT_VERSION}|{spec.template_id}|{variant}".encode())[:24]
            content = prompt_prefix + code
            row = {
                "data_source": DATA_SOURCE,
                "prompt": [{"content": content, "role": "user"}],
                "ability": "kernel_optimization",
                "reward_model": {"ground_truth": code, "style": "rule"},
                "extra_info": {
                    "entry_point": "Model",
                    "level": "coverage",
                    "module_name": "Model",
                    "ops": json.dumps([spec.source_call]),
                    "original_prompt": [{"content": content, "role": "user"}],
                    "repo_name": "loss_distance_generator_v1",
                    "type": "semantic_operator_synthetic",
                    "uuid": uuid,
                },
            }
            manifest = {
                "manifest_contract_version": MANIFEST_VERSION,
                "generator_contract_version": CONTRACT_VERSION,
                "registry_version": REGISTRY_VERSION,
                "runtime_contract_version": RUNTIME_CONTRACT_VERSION,
                "method": METHOD,
                "uuid": uuid,
                "template_id": spec.template_id,
                "template_variant": variant,
                "primary_family": spec.primary_family,
                "lineage_kind": "standalone_semantic_synthetic",
                "parent_uuid": None,
                "primary_intervention": "semantic_operator",
                "final_output_contract": {"kind": "single_tensor", "finite_required": True},
                "loss_input_contract": input_contract,
                "mode_behavior": "stateless",
                "training_mode": True,
                "declared_ops": declared_ops,
                "static_proof": static,
                "reference_sha256": reference_sha,
                "normalized_ast_sha256": normalized_sha,
                "prompt_sha256": _sha256_bytes(content.encode()),
                "row_payload_sha256": _canonical_sha256(row),
                "prompt_template": template_binding,
                "decontamination_roots": comparison_bindings,
                "decontamination_status": "exact_reference_and_normalized_ast_no_match",
                "provenance": {
                    "source_family": "project_generated_internal_review",
                    "generator": str(Path(__file__).resolve()),
                    "license": "internal-review-only",
                    "kernelbench_source_used_as_template": False,
                },
                "git_commit_at_generation": _git_commit(),
                "generator_source_sha256": generator_sha,
                "static_status": "passed",
                "reference_runtime_status": "pending",
                "operator_liveness_status": "pending",
                "materialization_status": "review_only",
                "structured_output_deferred": True,
                "training_approved": False,
            }
            pending.append((_sha256_bytes(f"{REGISTRY_VERSION}|{uuid}".encode()), row, manifest))
    pending.sort(key=lambda item: item[0])
    rows, manifests = [], []
    for index, (_, row, manifest) in enumerate(pending):
        manifest["candidate_row_index"] = index
        rows.append(row)
        manifests.append(manifest)
    if len(rows) != EXACT_ROWS or collections.Counter(item["primary_family"] for item in manifests) != {
        "loss_cross_entropy": 4,
        "loss_smooth_l1": 4,
        "loss_triplet_margin": 4,
    }:
        raise ValueError("loss registry quota mismatch")
    output_dir.mkdir(parents=True)
    candidates_path = output_dir / "candidates.parquet"
    manifest_path = output_dir / "manifest.jsonl"
    review_path = output_dir / "review_samples.md"
    pq.write_table(pa.Table.from_pylist(rows, schema=_schema()), candidates_path, compression="zstd")
    manifest_path.write_text("".join(_canonical_json(item) + "\n" for item in manifests), encoding="utf-8")
    review_lines = ["# Loss/distance canary review samples", ""]
    for spec in TEMPLATES:
        index = next(i for i, item in enumerate(manifests) if item["template_id"] == spec.template_id)
        review_lines.extend(
            [f"## {spec.template_id}", "", "```python", rows[index]["reward_model"]["ground_truth"], "```", ""]
        )
    review_path.write_text("\n".join(review_lines), encoding="utf-8")
    summary = {
        "contract_version": CONTRACT_VERSION,
        "rows": len(rows),
        "family_counts": dict(collections.Counter(item["primary_family"] for item in manifests)),
        "kernelbench_support": ["cross_entropy", "smooth_l1_loss", "triplet_margin_loss"],
        "review_only": True,
        "training_approved": False,
        "artifacts": {
            path.name: {"path": str(path), "sha256": _sha256_file(path)}
            for path in (candidates_path, manifest_path, review_path)
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--template-parquet", type=Path, required=True)
    parser.add_argument(
        "--comparison-artifact", type=Path, action="append", required=True, dest="comparison_artifacts"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(_canonical_json(generate(args.output_dir, args.template_parquet, tuple(args.comparison_artifacts))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
