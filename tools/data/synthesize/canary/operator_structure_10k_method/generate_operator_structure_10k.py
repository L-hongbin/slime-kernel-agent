#!/usr/bin/env python3
"""Closed, review-only registry for the 13k operator/structure candidate lane.

The registry is deliberately self-contained.  It creates candidates only; a
separate finalizer owns the deterministic selection of the 10k final mixture.
KernelBench and the earlier 1k lane are decontamination roots, never templates.
"""

from __future__ import annotations

import argparse
import ast
import collections
import copy
import dataclasses
import hashlib
import json
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.cleaning.complexity import extract_complexity_features, extract_operator_signature, feature_dict
from tools.data.cleaning.external import _template
from tools.data.cleaning.static_analysis import _call_name


CONTRACT_VERSION = "operator_structure_10k_deterministic_generator_v2"
MANIFEST_VERSION = "operator_structure_10k_parentless_manifest_v2"
REGISTRY_VERSION = "operator_structure_10k_closed_registry_48_templates_v2"
STATIC_CONTRACT_VERSION = "operator_structure_10k_static_contract_v2"
RUNTIME_CONTRACT_VERSION = "aten_dispatch_return_provenance_v1"
ORDER_VERSION = "sha256_operator_structure_10k_contract_template_variant_v1"
DATA_SOURCE = "project_generated_operator_structure_10k_v2"
METHOD = "operator_structure_10k"
EXACT_CANDIDATE_ROWS = 13_000
MAX_AUTHORIZED_CANDIDATES = 13_000
FINAL_SELECTED_ROWS = 10_000

FAMILY_CANDIDATE_QUOTAS = {
    "matmul_linear_graph": 4_100,
    "conv_norm_graph": 3_300,
    "reduction_pool_graph": 2_000,
    "modular_multiclass_graph": 2_400,
    "layout_fusion_graph": 1_200,
}
FAMILY_FINAL_QUOTAS = {
    "matmul_linear_graph": 3_100,
    "conv_norm_graph": 2_600,
    "reduction_pool_graph": 1_450,
    "modular_multiclass_graph": 1_900,
    "layout_fusion_graph": 950,
}

_HISTORICAL_1K_ROOT = (
    _REPO_ROOT
    / "local_artifacts/data/synthesize/kernelbench_gap_canary1000/run.primary.v2/analysis/final/accepted.parquet"
)
_HISTORICAL_1K_SOURCE_HASHES = {
    _REPO_ROOT
    / "tools/data/synthesize/kernelbench_gap_method/generate_kernelbench_gap.py": "6d2cae2d3d78f72b55093fc8fe8796e00e3605e397cade152a3129249df22ca4",
    _REPO_ROOT
    / "tools/data/synthesize/kernelbench_gap_method/validate_kernelbench_gap_liveness.py": "bfa268b9f91c05ba2192c95ca23fd594415ff9e2955cf9077692c0ac6ec45944",
    _REPO_ROOT
    / "tools/data/synthesize/kernelbench_gap_method/launch_kernelbench_gap_liveness_shards.sh": "0c45b11c6e97d05a0a6ad09e283cf2ceb53edd2d41ae3808bed36f19cb407d51",
    _REPO_ROOT
    / "tools/data/synthesize/kernelbench_gap_method/analyze_kernelbench_gap_run.py": "0e7268a5d74c99721e9d7caf33ca63d5f8b2dc435a937826f245f65017572076",
}


@dataclasses.dataclass(frozen=True)
class TemplateSpec:
    template_id: str
    family: str
    variants: int
    kind: str
    helper_count: int
    uses_batchnorm: bool = False


def _template_specs() -> tuple[TemplateSpec, ...]:
    specs: list[TemplateSpec] = []
    mm_variants = (350,) * 8 + (300,) * 3 + (400,)
    for index, variants in enumerate(mm_variants, 1):
        specs.append(TemplateSpec(f"MM{index:02d}", "matmul_linear_graph", variants, "matmul", 0))
    cn_variants = (300,) * 4 + (275,) * 4 + (250,) * 4
    for index, variants in enumerate(cn_variants, 1):
        specs.append(TemplateSpec(f"CN{index:02d}", "conv_norm_graph", variants, "conv", 0, True))
    for index in range(1, 9):
        specs.append(TemplateSpec(f"RP{index:02d}", "reduction_pool_graph", 250, "reduction", 0))
    for index in range(1, 11):
        specs.append(TemplateSpec(f"MC{index:02d}", "modular_multiclass_graph", 240, "modular", 2))
    for index in range(1, 7):
        specs.append(TemplateSpec(f"LF{index:02d}", "layout_fusion_graph", 200, "layout", 0, True))
    return tuple(specs)


TEMPLATES = _template_specs()
TEMPLATE_BY_ID = {spec.template_id: spec for spec in TEMPLATES}
if sum(spec.variants for spec in TEMPLATES) != EXACT_CANDIDATE_ROWS:
    raise RuntimeError("candidate quota does not sum to 13,000")

_HIGH_CALL_TEMPLATES = frozenset({"MM09", "MM10", "MM11", "MM12", "CN09", "CN10", "CN11", "CN12", "RP08"})


def _has_high_call_tail(spec: TemplateSpec) -> bool:
    return spec.template_id in _HIGH_CALL_TEMPLATES


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


def _verify_historical_1k_sources() -> dict[str, str]:
    observed = {str(path): _sha256_file(path) for path in _HISTORICAL_1K_SOURCE_HASHES}
    mismatches = {
        str(path): {"expected": expected, "observed": observed[str(path)]}
        for path, expected in _HISTORICAL_1K_SOURCE_HASHES.items()
        if observed[str(path)] != expected
    }
    if mismatches:
        raise ValueError(f"historical 1k source hash drift:{mismatches}")
    return observed


def _op(
    op_id: str, source_calls: Sequence[str], identities: Sequence[tuple[str, str]], *, min_calls: int = 1
) -> dict[str, Any]:
    return {
        "op_id": op_id,
        "source_calls": list(source_calls),
        "runtime_identities": [{"schema": schema, "overload": overload} for schema, overload in identities],
        "min_calls_per_trial": min_calls,
        "must_reach_returned_output": True,
    }


_BATCH_NORM_IDENTITIES = (
    ("aten::native_batch_norm", ""),
    ("aten::_native_batch_norm_legit", ""),
    ("aten::_native_batch_norm_legit_functional", ""),
    ("aten::cudnn_batch_norm", ""),
)


def _variant_spec(spec: TemplateSpec, variant: int) -> dict[str, Any]:
    topology = variant % 4
    architecture = (int(spec.template_id[-2:]) - 1) * 4 + topology
    activation_names = ("gelu", "silu", "relu", "tanh")
    activation_plan = tuple(activation_names[(architecture // (4**offset)) % 4] for offset in range(3))
    rank = (2, 3)[(variant // 4) % 2] if spec.family in {"matmul_linear_graph", "modular_multiclass_graph"} else 4
    if spec.family == "reduction_pool_graph":
        rank = (2, 3, 4)[(variant // 4) % 3]
    if spec.family in {"matmul_linear_graph", "modular_multiclass_graph"}:
        shape_group = variant // 8
        hidden = 24 + 4 * (shape_group % 10)
        batch = 2 + shape_group // 10
        sequence = 12 + shape_group % 10
        spatial = 12
    elif spec.family in {"conv_norm_graph", "layout_fusion_graph"}:
        shape_group = variant // 4
        hidden = 24 + 4 * (shape_group % 15)
        batch = 2 + shape_group // 15
        sequence = 12
        spatial = 12 + shape_group % 4
    else:
        shape_group = variant // 12
        hidden = 24 + 4 * shape_group
        batch = 2 + shape_group // 10
        sequence = 12 + shape_group % 10
        spatial = 12 + shape_group % 4
    mode_cycle = ("signed", "unsigned", "paired", "offset", "strided", "movedim", "positive_divisor")
    mode = mode_cycle[(variant // 3 + int(spec.template_id[-2:])) % len(mode_cycle)]
    if not spec.uses_batchnorm and not (spec.family == "reduction_pool_graph" and rank == 4) and variant % 32 == 7:
        mode = "repeat_clone"
    if rank == 2 and mode == "movedim":
        mode = "strided"
    if rank != 4 and spec.family in {"conv_norm_graph", "layout_fusion_graph"}:
        raise AssertionError("convolution layouts require rank four")
    arity = 2 if mode in {"paired", "positive_divisor"} else 1
    return {
        "topology": topology,
        "rank": rank,
        "hidden": hidden,
        "batch": batch,
        "sequence": sequence,
        "spatial": spatial,
        "mode": mode,
        "arity": arity,
        "shape_bucket": ("small", "medium", "compact")[variant % 3],
        "activation": activation_plan[0],
        "activation_plan": activation_plan,
        "architecture": architecture,
        "input_recipe": (variant // 4 + int(spec.template_id[-2:])) % 16,
        "input_post_style": (variant // 64) % 7,
        "template_skeleton": f"{spec.family}:{spec.kind}:topology{topology}:rank{rank}",
        "skeleton_id": f"{spec.template_id}:t{topology}:r{rank}:{mode}",
    }


def _mode_behavior(spec: TemplateSpec, rendered: Mapping[str, Any]) -> str:
    if spec.uses_batchnorm or (spec.family == "reduction_pool_graph" and int(rendered["rank"]) == 4):
        return "train_stateful"
    return "stateless"


def _activation(name: str, value: str) -> str:
    return {
        "gelu": f"F.gelu({value})",
        "silu": f"F.silu({value})",
        "relu": f"F.relu({value})",
        "tanh": f"torch.tanh({value})",
    }[name]


def _input_factory(rendered: Mapping[str, Any], *, conv: bool) -> str:
    rank, batch, hidden, sequence, spatial = (
        int(rendered["rank"]),
        int(rendered["batch"]),
        int(rendered["hidden"]),
        int(rendered["sequence"]),
        int(rendered["spatial"]),
    )
    if conv:
        shape = f"({batch}, {hidden}, {spatial}, {spatial})"
    elif rank == 2:
        shape = f"({batch}, {hidden})"
    elif rank == 3:
        shape = f"({batch}, {sequence}, {hidden})"
    else:
        shape = f"({batch}, {hidden}, {spatial}, {spatial})"
    mode = str(rendered["mode"])
    recipe = int(rendered["input_recipe"])
    distribution_style = int(rendered["input_post_style"])
    lines = ["def get_inputs():", f"    shape = {shape}"]
    if recipe == 0:
        lines.extend(
            [
                "    base = torch.randn(shape)",
                "    scale = torch.sigmoid(torch.randn(shape)) + 0.5",
                "    base = base * scale",
            ]
        )
    elif recipe == 1:
        lines.extend(
            [
                "    noise = torch.randn(shape)",
                "    scale = torch.sigmoid(torch.randn(shape))",
                "    base = noise * (scale + 0.5)",
            ]
        )
    elif recipe == 2:
        lines.extend(
            ["    anchor = torch.randn(shape)", "    delta = torch.randn(shape) * 0.125", "    base = anchor + delta"]
        )
    elif recipe == 3:
        lines.append("    base = torch.tanh(torch.randn(shape))")
    elif recipe == 4:
        lines.append("    base = F.gelu(torch.randn(shape))")
    elif recipe == 5:
        lines.append("    base = torch.sin(torch.randn(shape))")
    elif recipe == 6:
        lines.append("    base = torch.randn(shape).clamp(-2.0, 2.0)")
    elif recipe == 7:
        lines.append("    base = torch.randn(shape) + 0.125 * torch.randn(shape)")
    elif recipe == 8:
        lines.append("    base = torch.randn(shape) * torch.randn(shape)")
    elif recipe == 9:
        lines.append("    base = torch.randn(shape) / (torch.sigmoid(torch.randn(shape)) + 0.5)")
    elif recipe == 10:
        lines.extend(["    seed = torch.empty(shape)", "    base = torch.randn_like(seed)"])
    elif recipe == 11:
        lines.append("    base = torch.normal(mean=0.0, std=1.0, size=shape)")
    elif recipe == 12:
        lines.append("    base = torch.randn(shape).abs()")
    elif recipe == 13:
        lines.append("    base = torch.log1p(torch.exp(torch.randn(shape)))")
    elif recipe == 14:
        lines.append("    base = torch.cos(torch.randn(shape))")
    else:
        lines.append("    base = torch.sinh(torch.randn(shape) * 0.25)")
    if distribution_style == 0:
        lines.append("    base = base + torch.sin(base)")
    elif distribution_style == 1:
        lines.append("    base = F.silu(base)")
    elif distribution_style == 2:
        lines.append("    base = base * torch.sigmoid(base)")
    elif distribution_style == 3:
        lines.append("    base = torch.tanh(base) + 0.125 * base")
    elif distribution_style == 4:
        lines.append("    base = base / (1.0 + base.abs())")
    elif distribution_style == 5:
        lines.append("    base = base + torch.cos(base)")
    else:
        lines.append("    base = F.relu(base) - 0.25 * base")
    if mode == "signed":
        # Some base recipes are one-sided (for example abs/softplus).  Center
        # after every recipe so the signed label proves an actual two-sided
        # value domain instead of merely meaning "not passed through sigmoid".
        lines.append("    x = base - base.mean()")
    elif mode == "unsigned":
        lines.append("    x = torch.sigmoid(base)")
    elif mode == "offset":
        lines.extend(
            [
                "    backing_shape = (shape[0] + 1, *shape[1:])",
                "    prefix = torch.randn((1, *shape[1:]))",
                "    backing = torch.cat([prefix, base], dim=0)",
                "    x = backing[1:]",
            ]
        )
    elif mode == "strided":
        lines.extend(
            [
                "    interleaved = torch.stack([base, torch.zeros_like(base)], dim=-1)",
                "    x = interleaved[..., 0]",
            ]
        )
    elif mode == "movedim":
        lines.append("    x = base.movedim(1, -1)")
    elif mode == "repeat_clone":
        lines.extend(
            [
                "    repeated = base[..., :1]",
                "    x = torch.cat([repeated, repeated, base[..., 2:]], dim=-1).clone()",
            ]
        )
    elif mode in {"paired", "positive_divisor"}:
        lines.append("    x = base")
    else:
        raise ValueError(f"unknown input mode:{mode}")
    if mode == "paired":
        lines.extend(["    y = torch.randn(shape) * 0.125", "    return [x, y]"])
    elif mode == "positive_divisor":
        lines.extend(["    divisor = torch.sigmoid(torch.randn(shape)) + 0.5", "    return [x, divisor]"])
    else:
        lines.append("    return [x]")
    # KernelGym's production loader requires both direct-input factories.  The
    # initialization factory is deliberately the minimal synchronous, empty
    # positional-argument factory; `_static_contract` proves that it cannot
    # hide setup work or a second return path.
    lines.extend(["", "def get_init_inputs():", "    return []"])
    return "\n".join(lines)


def _preamble() -> str:
    return "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"


def _pair_prefix(rendered: Mapping[str, Any]) -> list[str]:
    mode = str(rendered["mode"])
    prefix: list[str] = []
    if mode == "movedim":
        prefix.append("        x = x.movedim(-1, 1)")
    if mode == "paired":
        prefix.extend(["        if y is not None:", "            x = x + 0.125 * y"])
    elif mode == "positive_divisor":
        prefix.extend(["        if y is not None:", "            x = x / y"])
    return prefix


def _matrix_attention_lines(
    query: str, key: str, value: str, rendered: Mapping[str, Any], *, target: str
) -> list[str]:
    """Return two explicit rank-specific matrix products whose result reaches target."""

    rank = int(rendered["rank"])
    if rank == 2:
        score = f"torch.mm({query}, {key}.transpose(0, 1))"
        mixed = f"torch.mm(weights, {value})"
    elif rank == 3:
        score = f"torch.bmm({query}, {key}.transpose(1, 2))"
        mixed = f"torch.bmm(weights, {value})"
    else:
        raise ValueError(f"matrix-attention only supports rank 2/3, got {rank}")
    return [
        f"        scores = {score}",
        f"        weights = torch.softmax(scores / ({query}.shape[-1] ** 0.5), dim=-1)",
        f"        {target} = {target} + 0.05 * {mixed}",
    ]


def _linear_model(spec: TemplateSpec, rendered: Mapping[str, Any], *, modular: bool = False) -> str:
    h, topology = int(rendered["hidden"]), int(rendered["topology"])
    activation_one, activation_two, activation_three = tuple(rendered["activation_plan"])
    argument = "self, x, y=None" if int(rendered["arity"]) == 2 else "self, x"
    act1, act2, act3 = (
        _activation(activation_one, "z"),
        _activation(activation_two, "u"),
        _activation(activation_three, "z"),
    )
    matrix_lines = _matrix_attention_lines("z", "z", "z", rendered, target="z")
    mixer_matrix_lines = _matrix_attention_lines("q", "k", "v", rendered, target="v")
    deep_init = (
        [
            "        self.deep_proj1 = nn.Linear(self.width, self.width)",
            "        self.deep_norm1 = nn.LayerNorm(self.width)",
            "        self.deep_proj2 = nn.Linear(self.width, self.width)",
            "        self.deep_norm2 = nn.LayerNorm(self.width)",
            "        self.deep_gate = nn.Linear(self.width, self.width)",
        ]
        if _has_high_call_tail(spec)
        else []
    )
    deep_forward = (
        [
            "        z = self.deep_proj1(z)",
            "        z = self.deep_norm1(z)",
            f"        z = {_activation(activation_two, 'z')}",
            "        z = self.deep_proj2(z)",
            "        z = self.deep_norm2(z)",
            f"        z = {_activation(activation_three, 'z')}",
            "        z = z * torch.sigmoid(self.deep_gate(z))",
        ]
        if _has_high_call_tail(spec)
        else []
    )
    if modular:
        return "\n".join(
            [
                _preamble().rstrip(),
                "",
                "class ProjectionStage(nn.Module):",
                "    def __init__(self, width):",
                "        super().__init__()",
                "        self.left = nn.Linear(width, width)",
                "        self.norm = nn.LayerNorm(width)",
                "        self.right = nn.Linear(width, width)",
                "    def forward(self, x):",
                "        x = self.left(x)",
                f"        x = {_activation(activation_one, 'x')}",
                "        x = self.norm(x)",
                "        return self.right(x)",
                "",
                "class MixerStage(nn.Module):",
                "    def __init__(self, width):",
                "        super().__init__()",
                "        self.query = nn.Linear(width, width)",
                "        self.key = nn.Linear(width, width)",
                "        self.value = nn.Linear(width, width)",
                "    def forward(self, x):",
                "        q = self.query(x)",
                "        k = self.key(x)",
                "        v = self.value(x)",
                *mixer_matrix_lines,
                "        return v",
                "",
                "class Model(nn.Module):",
                "    def __init__(self):",
                "        super().__init__()",
                f"        self.width = {h}",
                "        self.stage_a = ProjectionStage(self.width)",
                "        self.stage_b = MixerStage(self.width)",
                "        self.gate = nn.Linear(self.width, self.width)",
                "        self.norm = nn.LayerNorm(self.width)",
                "        self.bridge = nn.Linear(self.width, self.width)",
                "        self.bridge_norm = nn.LayerNorm(self.width)",
                "        self.refine = nn.Linear(self.width, self.width)",
                "        self.refine_norm = nn.LayerNorm(self.width)",
                "        self.final = nn.Linear(self.width, self.width)",
                "        self.out = nn.Linear(self.width, self.width)",
                f"    def forward({argument}):",
                *_pair_prefix(rendered),
                "        residual = x",
                "        z = self.stage_a(x)",
                "        z = self.stage_b(z)",
                "        z = self.gate(z)",
                f"        z = {_activation(activation_two, 'z')}",
                "        z = self.norm(z + residual)",
                "        z = self.bridge(z)",
                "        z = self.bridge_norm(z)",
                f"        z = {_activation(activation_three, 'z')}",
                "        z = self.refine(z)",
                "        z = self.refine_norm(z)",
                f"        z = {_activation(activation_one, 'z')}",
                "        z = self.final(z)",
                "        return self.out(z)",
                "",
                _input_factory(rendered, conv=False),
            ]
        )
    residual = "z + residual" if topology % 2 == 0 else "z - 0.125 * residual"
    return "\n".join(
        [
            _preamble().rstrip(),
            "",
            "class Model(nn.Module):",
            "    def __init__(self):",
            "        super().__init__()",
            f"        self.width = {h}",
            "        self.in_proj = nn.Linear(self.width, self.width)",
            "        self.pre_norm = nn.LayerNorm(self.width)",
            "        self.norm1 = nn.LayerNorm(self.width)",
            "        self.mid_proj = nn.Linear(self.width, self.width)",
            "        self.norm2 = nn.LayerNorm(self.width)",
            "        self.gate = nn.Linear(self.width, self.width)",
            "        self.post_proj = nn.Linear(self.width, self.width)",
            "        self.norm3 = nn.LayerNorm(self.width)",
            "        self.final_proj = nn.Linear(self.width, self.width)",
            "        self.final_norm = nn.LayerNorm(self.width)",
            "        self.tail_proj = nn.Linear(self.width, self.width)",
            "        self.tail_norm = nn.LayerNorm(self.width)",
            *deep_init,
            "        self.out_proj = nn.Linear(self.width, self.width)",
            f"    def forward({argument}):",
            *_pair_prefix(rendered),
            "        residual = self.pre_norm(x)",
            "        z = self.in_proj(x)",
            f"        z = {act1}",
            "        z = self.norm1(z)",
            "        u = self.mid_proj(z)",
            f"        u = {act2}",
            "        v = self.gate(u)",
            "        v = torch.sigmoid(v)",
            "        z = u * v",
            *matrix_lines,
            f"        z = {residual}",
            "        z = self.norm2(z)",
            f"        z = {act3}",
            "        z = self.post_proj(z)",
            "        z = self.norm3(z)",
            f"        z = {act1}",
            "        z = self.final_proj(z)",
            "        z = self.final_norm(z)",
            "        z = self.tail_proj(z)",
            "        z = self.tail_norm(z)",
            *deep_forward,
            "        return self.out_proj(z)",
            "",
            _input_factory(rendered, conv=False),
        ]
    )


def _conv_model(spec: TemplateSpec, rendered: Mapping[str, Any], *, layout: bool = False) -> str:
    h, topology = int(rendered["hidden"]), int(rendered["topology"])
    activation_one, activation_two, activation_three = tuple(rendered["activation_plan"])
    argument = "self, x, y=None" if int(rendered["arity"]) == 2 else "self, x"
    skip = "z + residual" if topology % 2 == 0 else "z - 0.125 * residual"
    layout_enter = ["        z = z.transpose(-1, -2)"] if layout else []
    layout_exit = ["        z = z.transpose(-1, -2)"] if layout else []
    pooling_lines = (
        [
            "        pooled = F.avg_pool2d(z, kernel_size=3, stride=1, padding=1)",
            "        z = z + pooled",
        ]
        if spec.family == "reduction_pool_graph"
        else []
    )
    deep_init = (
        [
            "        self.deep_conv1 = nn.Conv2d(self.width, self.width, 3, padding=1)",
            "        self.deep_norm1 = nn.GroupNorm(4, self.width)",
            "        self.deep_conv2 = nn.Conv2d(self.width, self.width, 3, padding=1)",
            "        self.deep_norm2 = nn.GroupNorm(4, self.width)",
            "        self.deep_mix = nn.Conv2d(self.width, self.width, 1)",
        ]
        if _has_high_call_tail(spec)
        else []
    )
    deep_forward = (
        [
            "        z = self.deep_conv1(z)",
            "        z = self.deep_norm1(z)",
            f"        z = {_activation(activation_one, 'z')}",
            "        z = self.deep_conv2(z)",
            "        z = self.deep_norm2(z)",
            f"        z = {_activation(activation_two, 'z')}",
            "        z = self.deep_mix(z)",
            f"        z = {_activation(activation_three, 'z')}",
        ]
        if _has_high_call_tail(spec)
        else []
    )
    return "\n".join(
        [
            _preamble().rstrip(),
            "",
            "class Model(nn.Module):",
            "    def __init__(self):",
            "        super().__init__()",
            f"        self.width = {h}",
            "        self.conv1 = nn.Conv2d(self.width, self.width, 3, padding=1)",
            "        self.bn1 = nn.BatchNorm2d(self.width)",
            "        self.conv2 = nn.Conv2d(self.width, self.width, 3, padding=1)",
            "        self.bn2 = nn.BatchNorm2d(self.width)",
            "        self.conv3 = nn.Conv2d(self.width, self.width, 1)",
            "        self.norm3 = nn.GroupNorm(4, self.width)",
            "        self.conv4 = nn.Conv2d(self.width, self.width, 3, padding=1)",
            "        self.bn4 = nn.BatchNorm2d(self.width)",
            "        self.conv5 = nn.Conv2d(self.width, self.width, 1)",
            "        self.norm4 = nn.GroupNorm(4, self.width)",
            "        self.mix = nn.Conv2d(self.width, self.width, 1)",
            "        self.final_norm = nn.GroupNorm(4, self.width)",
            "        self.tail = nn.Conv2d(self.width, self.width, 1)",
            "        self.tail_norm = nn.GroupNorm(4, self.width)",
            *deep_init,
            "        self.out = nn.Conv2d(self.width, self.width, 1)",
            f"    def forward({argument}):",
            *_pair_prefix(rendered),
            "        residual = x",
            "        z = self.conv1(x)",
            "        z = self.bn1(z)",
            f"        z = {_activation(activation_one, 'z')}",
            "        z = self.conv2(z)",
            "        z = self.bn2(z)",
            f"        z = {_activation(activation_two, 'z')}",
            *layout_enter,
            "        z = self.conv3(z)",
            "        z = self.norm3(z)",
            *pooling_lines,
            "        z = self.conv4(z)",
            "        z = self.bn4(z)",
            f"        z = {_activation(activation_three, 'z')}",
            *layout_exit,
            "        z = self.conv5(z)",
            "        z = self.norm4(z)",
            "        z = self.mix(z)",
            "        z = self.final_norm(z)",
            "        z = self.tail(z)",
            "        z = self.tail_norm(z)",
            *deep_forward,
            f"        z = {skip}",
            # A fixed smooth tail avoids the relu -> residual -> relu
            # idempotence that can make one of the two relus numerically dead
            # for one-sided inputs.
            "        z = torch.tanh(z)",
            "        return self.out(z)",
            "",
            _input_factory(rendered, conv=True),
        ]
    )


def _reduction_model(spec: TemplateSpec, rendered: Mapping[str, Any]) -> str:
    if int(rendered["rank"]) == 4:
        return _conv_model(spec, rendered, layout=False)
    h = int(rendered["hidden"])
    activation_one, activation_two, activation_three = tuple(rendered["activation_plan"])
    argument = "self, x, y=None" if int(rendered["arity"]) == 2 else "self, x"
    deep_init = (
        [
            "        self.deep_proj1 = nn.Linear(self.width, self.width)",
            "        self.deep_norm1 = nn.LayerNorm(self.width)",
            "        self.deep_proj2 = nn.Linear(self.width, self.width)",
            "        self.deep_norm2 = nn.LayerNorm(self.width)",
            "        self.deep_gate = nn.Linear(self.width, self.width)",
        ]
        if _has_high_call_tail(spec)
        else []
    )
    deep_forward = (
        [
            "        z = self.deep_proj1(z)",
            "        z = self.deep_norm1(z)",
            f"        z = {_activation(activation_one, 'z')}",
            "        z = self.deep_proj2(z)",
            "        z = self.deep_norm2(z)",
            f"        z = {_activation(activation_two, 'z')}",
            "        z = z * torch.sigmoid(self.deep_gate(z))",
        ]
        if _has_high_call_tail(spec)
        else []
    )
    return "\n".join(
        [
            _preamble().rstrip(),
            "",
            "class Model(nn.Module):",
            "    def __init__(self):",
            "        super().__init__()",
            f"        self.width = {h}",
            "        self.proj1 = nn.Linear(self.width, self.width)",
            "        self.norm1 = nn.LayerNorm(self.width)",
            "        self.proj2 = nn.Linear(self.width, self.width)",
            "        self.norm2 = nn.LayerNorm(self.width)",
            "        self.gate = nn.Linear(self.width, self.width)",
            "        self.post = nn.Linear(self.width, self.width)",
            "        self.norm3 = nn.LayerNorm(self.width)",
            "        self.refine = nn.Linear(self.width, self.width)",
            "        self.norm4 = nn.LayerNorm(self.width)",
            "        self.bridge = nn.Linear(self.width, self.width)",
            "        self.bridge_norm = nn.LayerNorm(self.width)",
            "        self.tail = nn.Linear(self.width, self.width)",
            "        self.tail_norm = nn.LayerNorm(self.width)",
            *deep_init,
            "        self.out = nn.Linear(self.width, self.width)",
            f"    def forward({argument}):",
            *_pair_prefix(rendered),
            "        residual = x",
            "        z = self.proj1(x)",
            f"        z = {_activation(activation_one, 'z')}",
            "        z = self.norm1(z)",
            "        summary = z.mean(dim=-2, keepdim=True)",
            "        z = z + summary",
            "        z = self.proj2(z)",
            "        z = self.norm2(z)",
            "        gate = torch.sigmoid(self.gate(z))",
            "        z = z * gate",
            "        z = z + residual",
            f"        z = {_activation(activation_two, 'z')}",
            "        z = self.post(z)",
            "        z = self.norm3(z)",
            "        z = self.refine(z)",
            "        z = self.norm4(z)",
            "        z = self.bridge(z)",
            "        z = self.bridge_norm(z)",
            f"        z = {_activation(activation_three, 'z')}",
            "        z = self.tail(z)",
            "        z = self.tail_norm(z)",
            *deep_forward,
            "        return self.out(z)",
            "",
            _input_factory(rendered, conv=False),
        ]
    )


def _render(template_id: str, variant: int) -> tuple[str, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    spec = TEMPLATE_BY_ID[template_id]
    if not 0 <= variant < spec.variants:
        raise ValueError(f"variant out of range:{template_id}:{variant}")
    rendered = _variant_spec(spec, variant)
    matrix_declared = (
        _op("matrix_product", ("torch.mm",), (("aten::mm", ""),), min_calls=2)
        if int(rendered["rank"]) == 2
        else _op("matrix_product", ("torch.bmm",), (("aten::bmm", ""),), min_calls=2)
    )
    if spec.family == "matmul_linear_graph":
        code = _linear_model(spec, rendered)
        declared = [
            _op("linear", ("nn.Linear",), (("aten::addmm", ""), ("aten::linear", "")), min_calls=4),
            matrix_declared,
        ]
    elif spec.family == "conv_norm_graph":
        code = _conv_model(spec, rendered)
        declared = [
            _op("convolution", ("nn.Conv2d",), (("aten::convolution", ""),), min_calls=4),
            _op("batch_norm", ("nn.BatchNorm2d",), _BATCH_NORM_IDENTITIES, min_calls=3),
            _op("group_norm", ("nn.GroupNorm",), (("aten::native_group_norm", ""),), min_calls=3),
        ]
    elif spec.family == "reduction_pool_graph":
        code = _reduction_model(spec, rendered)
        declared = [
            _op(
                "projection",
                ("nn.Linear", "nn.Conv2d"),
                (("aten::addmm", ""), ("aten::linear", ""), ("aten::convolution", "")),
                min_calls=3,
            ),
            (
                _op("pooling", ("torch.nn.functional.avg_pool2d",), (("aten::avg_pool2d", ""),))
                if int(rendered["rank"]) == 4
                else _op("reduction", ("tensor.mean",), (("aten::mean", "dim"),))
            ),
        ]
    elif spec.family == "modular_multiclass_graph":
        code = _linear_model(spec, rendered, modular=True)
        declared = [
            _op("projection", ("nn.Linear",), (("aten::addmm", ""), ("aten::linear", "")), min_calls=4),
            matrix_declared,
            _op("layer_norm", ("nn.LayerNorm",), (("aten::native_layer_norm", ""),), min_calls=3),
        ]
    elif spec.family == "layout_fusion_graph":
        code = _conv_model(spec, rendered, layout=True)
        declared = [
            _op("convolution", ("nn.Conv2d",), (("aten::convolution", ""),), min_calls=4),
            _op("layout", ("tensor.transpose",), (("aten::transpose", "int"),)),
            _op("batch_norm", ("nn.BatchNorm2d",), _BATCH_NORM_IDENTITIES, min_calls=3),
            _op("group_norm", ("nn.GroupNorm",), (("aten::native_group_norm", ""),), min_calls=3),
        ]
    else:
        raise AssertionError(spec.family)
    labels = {
        "template_skeleton": rendered["template_skeleton"],
        "skeleton_id": rendered["skeleton_id"],
        "input_arity": rendered["arity"],
        "input_mode": rendered["mode"],
        "input_rank": rendered["rank"],
        "shape_bucket": rendered["shape_bucket"],
    }
    return code, declared, labels, rendered


def _input_return_arity(function: ast.FunctionDef) -> int:
    returns = [node for node in ast.walk(function) if isinstance(node, ast.Return)]
    if len(returns) != 1 or not isinstance(returns[0].value, ast.List) or not returns[0].value.elts:
        raise ValueError("get_inputs must have one literal non-empty list return")
    return len(returns[0].value.elts)


def _require_empty_init_factory(function: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
    """Prove the KernelGym init factory is the one permitted empty factory."""

    arguments = function.args
    if isinstance(function, ast.AsyncFunctionDef) or any(
        (
            arguments.posonlyargs,
            arguments.args,
            arguments.kwonlyargs,
            arguments.defaults,
            arguments.kw_defaults,
            arguments.vararg is not None,
            arguments.kwarg is not None,
        )
    ):
        raise ValueError("get_init_inputs must be synchronous and parameterless")
    if len(function.body) != 1 or not isinstance(function.body[0], ast.Return):
        raise ValueError("get_init_inputs must contain exactly one direct return")
    returned = function.body[0].value
    if not isinstance(returned, ast.List) or returned.elts:
        raise ValueError("get_init_inputs must directly return an empty list")


def _assigned_self_attrs(function: ast.FunctionDef | None) -> set[str]:
    if function is None:
        return set()
    attrs: set[str] = set()
    for node in ast.walk(function):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self":
                attrs.add(target.attr)
    return attrs


def _called_self_attrs(function: ast.FunctionDef) -> set[str]:
    calls: set[str] = set()
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            calls.add(node.func.attr)
    return calls


def _direct_input_bytes(rendered: Mapping[str, Any]) -> int:
    dimensions = [int(rendered["batch"]), int(rendered["hidden"])]
    if int(rendered["rank"]) == 3:
        dimensions.insert(1, int(rendered["sequence"]))
    elif int(rendered["rank"]) == 4:
        dimensions.extend([int(rendered["spatial"]), int(rendered["spatial"])])
    elements = 1
    for dimension in dimensions:
        elements *= dimension
    return elements * 4 * int(rendered["arity"])


class _SemanticAstNormalizer(ast.NodeTransformer):
    """Erase literals and alpha-rename local names while retaining operators."""

    _PRESERVED_NAMES = frozenset({"torch", "nn", "F", "self", "super"})

    def __init__(self) -> None:
        self._names: dict[str, str] = {}
        self._self_attrs: dict[str, str] = {}

    def visit_Constant(self, node: ast.Constant) -> ast.Constant:
        if node.value is None or isinstance(node.value, bool):
            return ast.copy_location(ast.Constant(value=node.value), node)
        return ast.copy_location(ast.Constant(value=type(node.value).__name__), node)

    def visit_Name(self, node: ast.Name) -> ast.Name:
        if node.id in self._PRESERVED_NAMES:
            return node
        replacement = self._names.setdefault(node.id, f"v{len(self._names)}")
        return ast.copy_location(ast.Name(id=replacement, ctx=node.ctx), node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
        node = self.generic_visit(node)
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            replacement = self._self_attrs.setdefault(node.attr, f"m{len(self._self_attrs)}")
            node.attr = replacement
        return node


def _semantic_ast_dump(nodes: Sequence[ast.AST]) -> str:
    normalizer = _SemanticAstNormalizer()
    normalized = ast.Module(body=[normalizer.visit(copy.deepcopy(node)) for node in nodes], type_ignores=[])
    return ast.dump(ast.fix_missing_locations(normalized), annotate_fields=True, include_attributes=False)


def _semantic_ast_hash(nodes: Sequence[ast.AST]) -> str:
    return _sha256_bytes(_semantic_ast_dump(nodes).encode())


def _input_semantic_contract(code: str) -> dict[str, Any]:
    tree = ast.parse(code)
    factories = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_inputs"]
    if len(factories) != 1:
        raise ValueError("input semantic contract requires one get_inputs")
    factory = factories[0]
    shape_assignments = [
        node
        for node in factory.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "shape" for target in node.targets)
    ]
    if len(shape_assignments) != 1:
        raise ValueError("get_inputs requires exactly one shape assignment")
    normalizer = _SemanticAstNormalizer()
    shape_expression = ast.dump(
        normalizer.visit(ast.parse(ast.unparse(shape_assignments[0].value)).body[0].value),
        annotate_fields=True,
        include_attributes=False,
    )
    calls = sorted({_call_name(node.func) for node in ast.walk(factory) if isinstance(node, ast.Call)})
    return {
        "normalized_factory_ast_sha256": _semantic_ast_hash([factory]),
        "factory_calls": calls,
        "return_arity": _input_return_arity(factory),
        "normalized_shape_expression": shape_expression,
    }


def _semantic_skeletons(code: str) -> tuple[str, str, str]:
    tree = ast.parse(code)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    factories = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_inputs"]
    if len(factories) != 1:
        raise ValueError("semantic skeleton requires one get_inputs")
    model_hash = _semantic_ast_hash(classes)
    input_hash = _input_semantic_contract(code)["normalized_factory_ast_sha256"]
    return model_hash, input_hash, _canonical_sha256({"model_ast": model_hash, "input_ast": input_hash})


def _signed_int(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        return -int(node.operand.value)
    return None


def _semantic_intervention_proof(
    tree: ast.Module,
    declared_ops: Sequence[Mapping[str, Any]],
    rendered: Mapping[str, Any] | TemplateSpec | None,
) -> dict[str, Any]:
    """Reject the known structural forms that only look like interventions."""

    op_ids = {str(item["op_id"]) for item in declared_ops}
    model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Model")
    forward = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "forward")
    proof: dict[str, Any] = {}

    if "layout" in op_ids:
        transpose_positions: list[int] = []
        transpose_axes: list[list[int]] = []
        for index, statement in enumerate(forward.body):
            for call in (node for node in ast.walk(statement) if isinstance(node, ast.Call)):
                if _call_name(call.func) == "z.transpose":
                    transpose_positions.append(index)
                    transpose_axes.append(
                        [value for argument in call.args if (value := _signed_int(argument)) is not None]
                    )
        if len(transpose_positions) != 2 or transpose_axes != [[-1, -2], [-1, -2]]:
            raise ValueError("layout intervention requires two matching spatial transposes")
        between_calls = {
            _call_name(call.func)
            for statement in forward.body[transpose_positions[0] + 1 : transpose_positions[1]]
            for call in ast.walk(statement)
            if isinstance(call, ast.Call)
        }
        if "self.conv4" not in between_calls:
            raise ValueError("layout transposes must surround a spatial convolution")
        proof["layout"] = {
            "transpose_axes": transpose_axes,
            "intervening_spatial_call": "self.conv4",
            "adjacent_inverse_rejected": True,
        }

    if "reduction" in op_ids:
        mean_calls = [
            call
            for call in ast.walk(forward)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "mean"
        ]
        axes = [_signed_int(keyword.value) for call in mean_calls for keyword in call.keywords if keyword.arg == "dim"]
        if len(mean_calls) != 1 or axes != [-2]:
            raise ValueError("reduction must use the non-normalized batch/sequence axis")
        proof["reduction"] = {"mean_axis": -2, "normalized_feature_axis_rejected": True}

    if "pooling" in op_ids:
        pool_positions = [
            index
            for index, statement in enumerate(forward.body)
            if any(
                isinstance(call, ast.Call) and _call_name(call.func) == "F.avg_pool2d" for call in ast.walk(statement)
            )
        ]
        if len(pool_positions) != 1:
            raise ValueError("pooling intervention requires one avg_pool2d call")
        later_loads = {
            node.id
            for statement in forward.body[pool_positions[0] + 1 :]
            for node in ast.walk(statement)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        if "pooled" not in later_loads:
            raise ValueError("pooling result must participate in the returned path")
        proof["pooling"] = {"source_call": "F.avg_pool2d", "result_reused": True}

    if isinstance(rendered, Mapping) and rendered.get("mode") == "signed":
        factories = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_inputs"]
        centered = any(
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "x" for target in node.targets)
            and isinstance(node.value, ast.BinOp)
            and isinstance(node.value.op, ast.Sub)
            and isinstance(node.value.left, ast.Name)
            and node.value.left.id == "base"
            and isinstance(node.value.right, ast.Call)
            and _call_name(node.value.right.func) == "base.mean"
            for node in ast.walk(factories[0])
        )
        if not centered:
            raise ValueError("signed input mode must center the realized base distribution")
        proof["signed_input"] = {"centering": "base_minus_global_mean"}

    if isinstance(rendered, Mapping) and rendered.get("mode") == "strided":
        factories = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_inputs"]
        stacks = [
            call
            for call in ast.walk(factories[0])
            if isinstance(call, ast.Call) and _call_name(call.func) == "torch.stack"
        ]
        slices = [
            node
            for node in ast.walk(factories[0])
            if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "interleaved"
        ]
        if len(stacks) != 1 or len(slices) != 1:
            raise ValueError("strided mode must use one interleaved stack view")
        elements = stacks[0].args[0] if stacks[0].args else None
        if (
            not isinstance(elements, ast.List)
            or len(elements.elts) != 2
            or not isinstance(elements.elts[0], ast.Name)
            or elements.elts[0].id != "base"
            or not isinstance(elements.elts[1], ast.Call)
            or _call_name(elements.elts[1].func) != "torch.zeros_like"
        ):
            raise ValueError("strided mode must preserve every base value once")
        proof["strided_input"] = {"logical_values": "all_base_values_once", "noncontiguous_view": True}

    if isinstance(rendered, Mapping) and rendered.get("mode") == "repeat_clone":
        factories = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_inputs"]
        source = ast.unparse(factories[0])
        if "repeated = base[..., :1]" not in source or "base[..., 2:]" not in source or ".repeat(" in source:
            raise ValueError("repeat-clone must duplicate one feature slice, not the batch")
        proof["repeat_clone_input"] = {"repeated_axis": -1, "repeated_slices": 1, "batch_repeat_rejected": True}

    return proof


def _static_contract(
    code: str, declared_ops: Sequence[Mapping[str, Any]], spec: Mapping[str, Any] | TemplateSpec | None = None
) -> dict[str, Any]:
    tree = ast.parse(code)
    allowed_imports = {"torch", "torch.nn", "torch.nn.functional"}
    for node in tree.body:
        if isinstance(node, ast.Import) and any(alias.name not in allowed_imports for alias in node.names):
            raise ValueError("external import is forbidden")
        if isinstance(node, ast.ImportFrom):
            raise ValueError("from import is forbidden")
    forbidden = {
        "copy_",
        "scaled_dot_product_attention",
        "flatten",
        "eval",
        "exec",
        "__import__",
        "getattr",
        "setattr",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name.split(".")[-1] in forbidden or name in forbidden:
                raise ValueError(f"forbidden call:{name}")
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr.endswith("_")
                and not (node.func.attr.startswith("__") and node.func.attr.endswith("__"))
            ):
                raise ValueError(f"in-place call:{name}")
    class_nodes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    class_counts = collections.Counter(node.name for node in class_nodes)
    duplicates = sorted(name for name, count in class_counts.items() if count != 1)
    if duplicates:
        raise ValueError(f"duplicate top-level class:{duplicates}")
    classes = {node.name: node for node in class_nodes}
    if "Model" not in classes:
        raise ValueError("expected exactly one Model class")
    top_level_functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if [node.name for node in top_level_functions] != ["get_inputs", "get_init_inputs"]:
        raise ValueError("top-level factories must be get_inputs then get_init_inputs")
    if any(not isinstance(node, (ast.Import, ast.ClassDef, ast.FunctionDef)) for node in tree.body):
        raise ValueError("top-level executable statement is forbidden")
    reachable = {"Model"}
    pending = ["Model"]
    while pending:
        class_name = pending.pop()
        node = classes[class_name]
        method_nodes = [item for item in node.body if isinstance(item, ast.FunctionDef)]
        method_counts = collections.Counter(item.name for item in method_nodes)
        if any(count != 1 for count in method_counts.values()):
            raise ValueError(f"duplicate class method:{class_name}")
        methods = {item.name: item for item in method_nodes}
        init, forward = methods.get("__init__"), methods.get("forward")
        direct_returns = (
            [index for index, item in enumerate(forward.body) if isinstance(item, ast.Return)] if forward else []
        )
        if (
            forward is None
            or sum(isinstance(item, ast.Return) for item in ast.walk(forward)) != 1
            or len(direct_returns) != 1
        ):
            raise ValueError(f"class requires exactly one forward return:{class_name}")
        if direct_returns[0] != len(forward.body) - 1:
            raise ValueError(f"forward contains statement after return:{class_name}")
        unused = _assigned_self_attrs(init) - _called_self_attrs(forward)
        unused -= {"width"}
        if unused:
            raise ValueError(f"unreachable registered state:{class_name}:{sorted(unused)}")
        for call in ast.walk(init) if init is not None else ():
            if (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Name)
                and call.func.id in classes
                and call.func.id not in reachable
            ):
                reachable.add(call.func.id)
                pending.append(call.func.id)
    if set(classes) != reachable:
        raise ValueError(f"unreachable helper class:{sorted(set(classes) - reachable)}")
    get_inputs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_inputs"]
    if len(get_inputs) != 1 or isinstance(get_inputs[0], ast.AsyncFunctionDef):
        raise ValueError("get_inputs must be a unique synchronous top-level function")
    init_factories = [node for node in top_level_functions if node.name == "get_init_inputs"]
    if len(init_factories) != 1:
        raise ValueError("get_init_inputs must be a unique top-level function")
    _require_empty_init_factory(init_factories[0])
    factory_calls = {_call_name(node.func) for node in ast.walk(get_inputs[0]) if isinstance(node, ast.Call)}
    if not {"torch.randn", "torch.randn_like", "torch.normal"}.intersection(factory_calls):
        raise ValueError("get_inputs must construct nonconstant random inputs")
    calls = {_call_name(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    signature = extract_operator_signature(code)
    for op in declared_ops:
        if not set(op["source_calls"]).intersection(calls | set(signature)):
            raise ValueError(f"declared op lacks source witness:{op['op_id']}")
    features = feature_dict(extract_complexity_features(code))
    return {
        "single_value_return": True,
        "factory_calls": sorted(factory_calls),
        "input_arity": _input_return_arity(get_inputs[0]),
        "init_factory": {"synchronous": True, "parameter_count": 0, "direct_empty_list_return": True},
        "reachable_helper_classes": sorted(reachable),
        "top_level_class_count": len(classes),
        "realized_signature": list(signature),
        "complexity": features,
        "declared_source_witnesses": {
            op["op_id"]: sorted(set(op["source_calls"]) & (calls | set(signature))) for op in declared_ops
        },
        "semantic_intervention_proof": _semantic_intervention_proof(tree, declared_ops, spec),
    }


def _enforce_spec(
    code: str, spec: TemplateSpec, static: Mapping[str, Any], rendered: Mapping[str, Any] | None = None
) -> None:
    features = static["complexity"]
    if features["source_line_count"] < 50:
        raise ValueError(f"source too short:{spec.template_id}:{features['source_line_count']}")
    if features["forward_call_count"] < 10 or len(static["realized_signature"]) < 10:
        raise ValueError(
            f"input-dependent complexity too low:{spec.template_id}:{features['forward_call_count']}/{len(static['realized_signature'])}"
        )
    if features["init_nn_constructor_count"] < 5:
        raise ValueError(f"registered constructor quota missing:{spec.template_id}")
    if len(static["declared_source_witnesses"]) < 2 or any(
        not value for value in static["declared_source_witnesses"].values()
    ):
        raise ValueError(f"declared operation quota missing:{spec.template_id}")
    if spec.helper_count and len(static["reachable_helper_classes"]) < spec.helper_count + 1:
        raise ValueError(f"reachable helper quota missing:{spec.template_id}")
    if rendered is not None:
        if _direct_input_bytes(rendered) > 64 * 1024 * 1024:
            raise ValueError("input bytes exceed 64MiB")
        dimensions = [
            int(rendered["batch"]),
            int(rendered["hidden"]),
            int(rendered["sequence"]),
            int(rendered["spatial"]),
        ]
        non_one = [value for value in dimensions if value > 1]
        if max(non_one) / min(non_one) > 1_000:
            raise ValueError("dimension ratio exceeds 1e3")
        if _mode_behavior(spec, rendered) == "train_stateful" and rendered["mode"] == "repeat_clone":
            raise ValueError("batch norm cannot receive repeat-clone input")


def _semantic_schema() -> pa.Schema:
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
    normalized: set[str] = set()
    bindings: list[dict[str, Any]] = []
    for raw in paths:
        path = raw.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["reward_model"], batch_size=512):
            for item in batch.column(0).to_pylist():
                code = item.get("ground_truth") if isinstance(item, Mapping) else None
                if not isinstance(code, str) or not code.strip():
                    raise ValueError(f"comparison has empty reference:{path}")
                references.add(_sha256_bytes(code.encode()))
                normalized.add(_normalized_ast_sha256(code))
        bindings.append({"path": str(path), "sha256": _sha256_file(path), "rows": parquet.metadata.num_rows})
    return references, normalized, bindings


def build_records(
    *,
    prompt_prefix: str = "",
    comparison_reference_hashes: set[str] | None = None,
    comparison_ast_hashes: set[str] | None = None,
    template_binding: Mapping[str, Any] | None = None,
    comparison_bindings: Sequence[Mapping[str, Any]] = (),
    git_commit: str | None = None,
    generator_sha256: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    historical_hashes = _verify_historical_1k_sources()
    references = comparison_reference_hashes if comparison_reference_hashes is not None else set()
    ast_hashes = comparison_ast_hashes if comparison_ast_hashes is not None else set()
    source_hash = generator_sha256 or _sha256_file(Path(__file__).resolve())
    commit = git_commit or _git_commit()
    pending: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    seen_reference: dict[str, tuple[str, int]] = {}
    seen_normalized_ast: dict[str, tuple[str, int]] = {}
    for spec in TEMPLATES:
        for variant in range(spec.variants):
            code, declared_ops, labels, rendered = _render(spec.template_id, variant)
            static = _static_contract(code, declared_ops, rendered)
            _enforce_spec(code, spec, static, rendered)
            reference_sha256 = _sha256_bytes(code.encode())
            normalized_ast_sha256 = _normalized_ast_sha256(code)
            if reference_sha256 in references or normalized_ast_sha256 in ast_hashes:
                raise ValueError(f"decontamination collision:{spec.template_id}:{variant}")
            model_skeleton_sha256, input_skeleton_sha256, model_input_skeleton_sha256 = _semantic_skeletons(code)
            input_semantic_contract = _input_semantic_contract(code)
            if input_skeleton_sha256 != input_semantic_contract["normalized_factory_ast_sha256"]:
                raise ValueError("input semantic hash is not contract-derived")
            coordinate = (spec.template_id, variant)
            for field, value, seen in (
                ("reference_sha256", reference_sha256, seen_reference),
                ("normalized_ast_sha256", normalized_ast_sha256, seen_normalized_ast),
            ):
                previous = seen.setdefault(value, coordinate)
                if previous != coordinate:
                    raise ValueError(f"non-unique {field}:{previous}:{coordinate}")
            uuid = "os10k_" + _sha256_bytes(f"{CONTRACT_VERSION}|{spec.template_id}|{variant}".encode())[:24]
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
                    "ops": json.dumps(sorted({name for item in declared_ops for name in item["source_calls"]})),
                    "original_prompt": [{"content": content, "role": "user"}],
                    "repo_name": "operator_structure_10k_generator_v2",
                    "type": "semantic_operator_synthetic",
                    "uuid": uuid,
                },
            }
            manifest = {
                "manifest_contract_version": MANIFEST_VERSION,
                "generator_contract_version": CONTRACT_VERSION,
                "registry_version": REGISTRY_VERSION,
                "static_contract_version": STATIC_CONTRACT_VERSION,
                "runtime_contract_version": RUNTIME_CONTRACT_VERSION,
                "method": METHOD,
                "uuid": uuid,
                "template_id": spec.template_id,
                "template_variant": variant,
                "template_skeleton": rendered["template_skeleton"],
                "skeleton_id": rendered["skeleton_id"],
                "primary_family": spec.family,
                "lineage_kind": "standalone_semantic_synthetic",
                "parent_uuid": None,
                "primary_intervention": "semantic_operator",
                "final_output_contract": {"kind": "single_tensor", "finite_required": True},
                "training_mode": True,
                "mode_behavior": _mode_behavior(spec, rendered),
                "declared_ops": declared_ops,
                "static_proof": static,
                "coverage_labels": labels,
                "input_semantic_contract": input_semantic_contract,
                "input_semantic_skeleton_sha256": input_skeleton_sha256,
                "model_semantic_skeleton_sha256": model_skeleton_sha256,
                "model_input_semantic_skeleton_sha256": model_input_skeleton_sha256,
                "reference_sha256": reference_sha256,
                "normalized_ast_sha256": normalized_ast_sha256,
                "prompt_sha256": _sha256_bytes(content.encode()),
                "row_payload_sha256": _canonical_sha256(row),
                "prompt_template": dict(template_binding or {}),
                "decontamination_roots": [dict(item) for item in comparison_bindings],
                "historical_1k_source_hashes": historical_hashes,
                "decontamination_status": "exact_reference_and_normalized_ast_no_match",
                "provenance": {
                    "source_family": "project_generated_internal_review",
                    "generator": str(Path(__file__).resolve()),
                    "license": "internal-review-only",
                    "kernelbench_source_used_as_template": False,
                },
                "git_commit_at_generation": commit,
                "generator_source_sha256": source_hash,
                "static_status": "passed",
                "reference_runtime_status": "pending",
                "operator_liveness_status": "pending",
                "materialization_status": "review_only",
                "training_approved": False,
                "structured_output_deferred": True,
            }
            pending.append((_sha256_bytes(f"{ORDER_VERSION}|{spec.template_id}|{variant}".encode()), row, manifest))
    pending.sort(key=lambda item: item[0])
    rows, manifests = [], []
    for index, (_, row, manifest) in enumerate(pending):
        manifest["candidate_row_index"] = index
        rows.append(row)
        manifests.append(manifest)
    _enforce_registry(manifests)
    return rows, manifests


def _enforce_registry(manifests: Sequence[Mapping[str, Any]]) -> None:
    family_counts = collections.Counter(str(item["primary_family"]) for item in manifests)
    if len(manifests) != EXACT_CANDIDATE_ROWS or dict(family_counts) != FAMILY_CANDIDATE_QUOTAS:
        raise ValueError(f"candidate family quota mismatch:{len(manifests)}:{dict(family_counts)}")
    exact_fields = ("reference_sha256", "normalized_ast_sha256", "uuid", "row_payload_sha256")
    for field in exact_fields:
        values = [str(item[field]) for item in manifests]
        if len(values) != len(set(values)):
            raise ValueError(f"non-unique {field}")
    model_counts = collections.Counter(str(item["model_semantic_skeleton_sha256"]) for item in manifests)
    input_counts = collections.Counter(str(item["input_semantic_skeleton_sha256"]) for item in manifests)
    pair_counts = collections.Counter(str(item["model_input_semantic_skeleton_sha256"]) for item in manifests)
    if len(model_counts) < 160 or len(input_counts) < 1_000:
        raise ValueError(f"semantic skeleton coverage too low:model={len(model_counts)} input={len(input_counts)}")
    if max(model_counts.values()) > 96 or max(pair_counts.values()) > 32:
        raise ValueError(
            f"semantic skeleton concentration too high:model={max(model_counts.values())} pair={max(pair_counts.values())}"
        )
    ranks = collections.Counter(int(item["coverage_labels"]["input_rank"]) for item in manifests)
    if ranks[2] + ranks[3] < len(manifests) // 4:
        raise ValueError(f"rank 2/3 coverage too low:{dict(ranks)}")
    repeats = sum(item["coverage_labels"]["input_mode"] == "repeat_clone" for item in manifests)
    if repeats > len(manifests) * 0.05:
        raise ValueError(f"repeat clone quota exceeded:{repeats}")


def replay_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    code, declared_ops, labels, rendered = _render(str(manifest["template_id"]), int(manifest["template_variant"]))
    spec = TEMPLATE_BY_ID[str(manifest["template_id"])]
    static = _static_contract(code, declared_ops, rendered)
    _enforce_spec(code, spec, static, rendered)
    _, input_hash, _ = _semantic_skeletons(code)
    input_contract = _input_semantic_contract(code)
    if manifest.get("reference_sha256") and manifest["reference_sha256"] != _sha256_bytes(code.encode()):
        raise ValueError("manifest source does not replay")
    if manifest.get("input_semantic_skeleton_sha256") and manifest["input_semantic_skeleton_sha256"] != input_hash:
        raise ValueError("manifest input semantic skeleton does not replay")
    if manifest.get("input_semantic_contract") and manifest["input_semantic_contract"] != input_contract:
        raise ValueError("manifest input semantic contract does not replay")
    if manifest.get("mode_behavior") != _mode_behavior(spec, rendered):
        raise ValueError("manifest mode behavior does not replay")
    return {
        "code": code,
        "declared_ops": declared_ops,
        "coverage_labels": labels,
        "static_proof": static,
        "spec": rendered,
        "input_semantic_contract": input_contract,
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")


def generate(output_dir: Path, template_path: Path, comparison_paths: Sequence[Path]) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty:{output_dir}")
    _, prompt_prefix = _template(template_path)
    references, normalized, bindings = _comparison_hashes(comparison_paths)
    rows, manifests = build_records(
        prompt_prefix=prompt_prefix,
        comparison_reference_hashes=references,
        comparison_ast_hashes=normalized,
        template_binding={
            "path": str(template_path),
            "sha256": _sha256_file(template_path),
            "rows": pq.ParquetFile(template_path).metadata.num_rows,
        },
        comparison_bindings=bindings,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = output_dir / "candidates.parquet"
    manifest_path = output_dir / "manifest.jsonl"
    pq.write_table(pa.Table.from_pylist(rows, schema=_semantic_schema()), candidates, compression="zstd")
    _write_jsonl(manifest_path, manifests)
    summary = {
        "contract_version": CONTRACT_VERSION,
        "method": METHOD,
        "candidate_rows": len(rows),
        "final_selected_rows": FINAL_SELECTED_ROWS,
        "candidate_family_quotas": FAMILY_CANDIDATE_QUOTAS,
        "final_family_quotas": FAMILY_FINAL_QUOTAS,
        "training_approved": False,
        "artifacts": {
            "candidates": {"path": str(candidates), "sha256": _sha256_file(candidates)},
            "manifest": {"path": str(manifest_path), "sha256": _sha256_file(manifest_path)},
        },
    }
    _write_json(output_dir / "summary.json", summary)
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
    print(
        json.dumps(
            generate(args.output_dir, args.template_parquet, tuple(args.comparison_artifacts)),
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
