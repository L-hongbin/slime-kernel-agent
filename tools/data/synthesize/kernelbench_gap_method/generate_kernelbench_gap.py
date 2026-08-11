#!/usr/bin/env python3
"""Construct the deterministic 1k KernelBench-gap canary.

The registry deliberately covers low-level PyTorch cells and generic module
composition.  It never reads KernelBench source as a template: KernelBench is
only a decontamination root checked after rendering.
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
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.data.cleaning.complexity import extract_complexity_features, extract_operator_signature, feature_dict
from tools.data.cleaning.external import _template
from tools.data.synthesize.augment_prompt_tasks import _call_name, _normalized_ast_sha256

CONTRACT_VERSION = "kernelbench_gap_deterministic_generator_v1"
MANIFEST_VERSION = "kernelbench_gap_parentless_manifest_v1"
REGISTRY_VERSION = "kernelbench_gap_closed_registry_33_cells_v1"
ORDER_VERSION = "sha256_kernelbench_gap_contract_template_variant_v1"
STATIC_CONTRACT_VERSION = "kernelbench_gap_static_contract_v1"
RUNTIME_CONTRACT_VERSION = "aten_dispatch_return_provenance_v1"
DATA_SOURCE = "project_generated_kernelbench_gap_v1"
METHOD = "kernelbench_gap_canary"
MAX_AUTHORIZED_CANDIDATES = 1_000
EXACT_CANARY_ROWS = 1_000

FAMILY_QUOTAS = {
    "atomic_low_level": 390,
    "conv_norm_chain": 330,
    "long_single_class": 160,
    "modular_multiclass": 120,
}

_DEFAULT_TEMPLATE = _REPO_ROOT / "Data/prompt_tvm_v4/train.review.parquet"
_DEFAULT_COMPARISON_ROOTS = (
    _REPO_ROOT / "Data/prompt_tvm_v3/drkernel_rl_thinking.parquet",
    _REPO_ROOT / "Data/prompt_tvm_v4/train.review.parquet",
    _REPO_ROOT / "Data/prompt_tvm_v4/synthesis/accepted/extreme_ops_v3.parquet",
    *tuple(_REPO_ROOT / f"Data/kernelbench-level{level}-validation-tvm-v2/train.parquet" for level in (1, 2, 3)),
    _REPO_ROOT
    / "local_artifacts/data_handoffs/prompt_tvm_v4_semantic_operator_canary1000/analysis/final/accepted.parquet",
    _REPO_ROOT
    / "local_artifacts/data_handoffs/prompt_tvm_v4_frontier_operator_canary1000/analysis/final/accepted.parquet",
)


@dataclasses.dataclass(frozen=True)
class TemplateSpec:
    template_id: str
    family: str
    variants: int
    kind: str
    required_signature: tuple[str, ...]
    required_structure: tuple[tuple[str, int], ...] = ()
    mode_behavior: str = "stateless"


def _spec(
    template_id: str, family: str, variants: int, kind: str, *tokens: str, structure: dict[str, int] | None = None
) -> TemplateSpec:
    stateful = (
        "bn" in kind
        or "batchnorm" in kind
        or kind.startswith("long")
        or kind in {"convnorm_stage", "residual_projection"}
    )
    return TemplateSpec(
        template_id,
        family,
        variants,
        kind,
        tuple(tokens),
        tuple(sorted((structure or {}).items())),
        "train_stateful" if stateful else "stateless",
    )


TEMPLATES = (
    # 390 atomic rows.  AT14/15 are the only two-input cells.
    _spec("AT01", "atomic_low_level", 50, "deconv3d", "nn.ConvTranspose3d"),
    _spec("AT02", "atomic_low_level", 20, "conv3d", "nn.Conv3d"),
    _spec("AT03", "atomic_low_level", 20, "deconv2d", "nn.ConvTranspose2d"),
    _spec("AT04", "atomic_low_level", 20, "conv2d", "nn.Conv2d"),
    _spec("AT05", "atomic_low_level", 30, "bn2d", "nn.BatchNorm2d"),
    _spec("AT06", "atomic_low_level", 15, "bn1d", "nn.BatchNorm1d"),
    _spec("AT07", "atomic_low_level", 15, "bn3d", "nn.BatchNorm3d"),
    _spec("AT08", "atomic_low_level", 8, "groupnorm", "nn.GroupNorm"),
    _spec("AT09", "atomic_low_level", 7, "instancenorm", "nn.InstanceNorm2d"),
    _spec("AT10", "atomic_low_level", 10, "dropout", "nn.Dropout"),
    _spec("AT11", "atomic_low_level", 20, "maxpool3d", "nn.MaxPool3d"),
    _spec("AT12", "atomic_low_level", 20, "avgpool3d", "nn.AvgPool3d"),
    _spec("AT13", "atomic_low_level", 10, "adaptiveavgpool3d", "nn.AdaptiveAvgPool3d"),
    _spec("AT14", "atomic_low_level", 20, "div", "operator.div"),
    _spec("AT15", "atomic_low_level", 20, "sub", "operator.sub"),
    _spec("AT16", "atomic_low_level", 10, "sigmoid", "torch.sigmoid"),
    _spec("AT17", "atomic_low_level", 20, "flatten", "torch.flatten"),
    _spec("AT18", "atomic_low_level", 20, "contiguous", "tensor.contiguous"),
    _spec("AT19", "atomic_low_level", 20, "unfold", "tensor.unfold"),
    _spec(
        "AT20",
        "atomic_low_level",
        20,
        "channel_permute",
        "tensor.contiguous",
        "tensor.transpose",
        "tensor.view",
        "tensor.view",
    ),
    _spec("AT21", "atomic_low_level", 15, "forward_randn", "operator.add", "torch.randn"),
    # 330 short convolution/normalization chains.
    _spec("CN01", "conv_norm_chain", 100, "conv2d_bn_pool", "nn.Conv2d", "nn.BatchNorm2d", "nn.MaxPool2d"),
    _spec("CN02", "conv_norm_chain", 80, "depthwise_bn_pointwise", "nn.Conv2d", "nn.BatchNorm2d"),
    _spec("CN03", "conv_norm_chain", 70, "conv3d_bn_pool", "nn.Conv3d", "nn.BatchNorm3d", "nn.MaxPool3d"),
    _spec("CN04", "conv_norm_chain", 35, "deconv2d_groupnorm", "nn.ConvTranspose2d", "nn.GroupNorm"),
    _spec("CN05", "conv_norm_chain", 25, "deconv3d_batchnorm", "nn.ConvTranspose3d", "nn.BatchNorm3d"),
    _spec("CN06", "conv_norm_chain", 20, "conv1d_batchnorm", "nn.Conv1d", "nn.BatchNorm1d"),
    # Long code has one top-level class; modular code has reachable helpers.
    _spec(
        "LG01",
        "long_single_class",
        80,
        "long2d",
        "nn.Conv2d",
        "nn.BatchNorm2d",
        structure={"source_line_count": 50, "init_nn_constructor_count": 5, "forward_call_count": 10},
    ),
    _spec(
        "LG02",
        "long_single_class",
        40,
        "long3d",
        "nn.Conv3d",
        "nn.BatchNorm3d",
        structure={"source_line_count": 50, "init_nn_constructor_count": 5, "forward_call_count": 10},
    ),
    _spec(
        "LG03",
        "long_single_class",
        40,
        "long1d",
        "nn.Conv1d",
        "nn.BatchNorm1d",
        structure={"source_line_count": 50, "init_nn_constructor_count": 5, "forward_call_count": 10},
    ),
    _spec(
        "MC01",
        "modular_multiclass",
        60,
        "convnorm_stage",
        "nn.Conv2d",
        "nn.BatchNorm2d",
        structure={
            "top_level_class_count": 2,
            "source_line_count": 50,
            "init_nn_constructor_count": 5,
            "forward_call_count": 10,
        },
    ),
    _spec(
        "MC02",
        "modular_multiclass",
        40,
        "residual_projection",
        "nn.Conv2d",
        "nn.BatchNorm2d",
        structure={
            "top_level_class_count": 2,
            "source_line_count": 50,
            "init_nn_constructor_count": 5,
            "forward_call_count": 10,
        },
    ),
    _spec(
        "MC03",
        "modular_multiclass",
        20,
        "feature_spatial_mixer",
        "nn.Conv2d",
        "nn.GroupNorm",
        structure={
            "top_level_class_count": 3,
            "source_line_count": 50,
            "init_nn_constructor_count": 5,
            "forward_call_count": 10,
        },
    ),
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


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True).strip()


_BATCH_NORM = (
    ("aten::native_batch_norm", ""),
    ("aten::_native_batch_norm_legit", ""),
    ("aten::_native_batch_norm_legit_functional", ""),
    ("aten::cudnn_batch_norm", ""),
)
_INSTANCE_NORM = (
    ("aten::native_batch_norm", ""),
    ("aten::_native_batch_norm_legit", ""),
    ("aten::_native_batch_norm_legit_functional", ""),
    ("aten::cudnn_batch_norm", ""),
    ("aten::instance_norm", ""),
)


def _op(op_id: str, source_calls: Sequence[str], *aten: tuple[str, str], min_calls: int = 1) -> dict[str, Any]:
    return {
        "op_id": op_id,
        "source_calls": list(source_calls),
        "runtime_identities": [{"schema": schema, "overload": overload} for schema, overload in aten],
        "min_calls_per_trial": min_calls,
        "must_reach_returned_output": True,
    }


def _module(init: Sequence[str], forward: Sequence[str], inputs: Sequence[str], signature: str = "self, x") -> str:
    return "\n".join(
        [
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            "",
            "class Model(nn.Module):",
            "    def __init__(self):",
            "        super().__init__()",
            *(f"        {line}" for line in init),
            "",
            f"    def forward({signature}):",
            *(f"        {line}" for line in forward),
            "",
            "def get_inputs():",
            *(f"    {line}" for line in inputs),
            "",
            "def get_init_inputs():",
            "    return []",
            "",
        ]
    )


def _dims(variant: int) -> tuple[int, int, int, int]:
    # The combined period is 300, larger than every per-template quota.  This
    # keeps each variant's executed workload distinct without relying on
    # comments, docstrings, or dead constants to manufacture uniqueness.
    return 2 + variant % 3, (4, 8, 12, 16)[variant % 4], 8 + 2 * (variant % 25), 12 + 2 * (variant % 25)


_ACTIVE_RENDER_KEY = ""


def _single(shape: str) -> list[str]:
    """Return one of five materially different ``torch.rand`` input layouts."""
    key = int(_sha256_bytes(f"{_ACTIVE_RENDER_KEY}|{shape}".encode())[:8], 16)
    source, binding, returning = key % 5, (key // 5) % 4, (key // 20) % 3
    dims = [part.strip() for part in shape.split(",")]

    def bound_shape(values: Sequence[str], name: str) -> tuple[list[str], str]:
        tuple_shape = "(" + ", ".join(values) + ",)"
        if binding == 0:
            return [], ", ".join(values)
        if binding == 1:
            return [], tuple_shape
        if binding == 2:
            return [f"{name} = {tuple_shape}"], name
        return [f"{name} = [{', '.join(values)}]"], f"tuple({name})"

    if source == 0:
        prefix, shape_expr = bound_shape(dims, "shape")
        prefix.append(f"x = torch.rand({shape_expr})")
    elif source == 1:
        wide = [*dims[:-1], f"{dims[-1]} * 2"]
        prefix, source_expr = bound_shape(wide, "source_shape")
        prefix.extend([f"x = torch.rand({source_expr})", "x = x[..., ::2]"])
    elif source == 2:
        channels_last = [dims[0], *dims[2:], dims[1]]
        prefix, source_expr = bound_shape(channels_last, "source_shape")
        prefix.extend([f"x = torch.rand({source_expr})", "x = x.movedim(-1, 1)"])
    elif source == 3:
        expanded = ["1", *dims[1:]]
        prefix, source_expr = bound_shape(expanded, "source_shape")
        target_prefix, target_expr = bound_shape(dims, "target_shape")
        prefix.extend(target_prefix)
        prefix.extend([f"x = torch.rand({source_expr})", f"x = x.expand({target_expr})"])
    else:
        offset = [f"{dims[0]} + 1", *dims[1:]]
        prefix, source_expr = bound_shape(offset, "source_shape")
        prefix.extend([f"x = torch.rand({source_expr})", "x = x[1:]"])
    if returning == 0:
        return [*prefix, "return [x]"]
    if returning == 1:
        return [*prefix, "inputs = [x]", "return inputs"]
    return [*prefix, "inputs = []", "inputs.append(x)", "return inputs"]


def _render(template_id: str, variant: int) -> tuple[str, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Render one closed-registry cell; no task source is read as a template."""
    spec = TEMPLATE_BY_ID[template_id]
    if not 0 <= variant < spec.variants:
        raise ValueError(f"variant out of range:{template_id}:{variant}")
    global _ACTIVE_RENDER_KEY
    _ACTIVE_RENDER_KEY = f"{template_id}:{variant}"
    b, c, s, length = _dims(variant)
    ops: list[dict[str, Any]]
    arity = 1
    if spec.kind == "deconv3d":
        code = _module(
            [f"self.op = nn.ConvTranspose3d({c}, {c + 4}, 3, padding=1)"],
            ["return self.op(x)"],
            _single(f"{b}, {c}, 4, {s}, {s}"),
        )
        ops = [_op("conv_transpose3d", ["self.op"], ("aten::convolution", ""))]
    elif spec.kind == "conv3d":
        code = _module(
            [f"self.op = nn.Conv3d({c}, {c + 4}, 3, padding=1)"],
            ["return self.op(x)"],
            _single(f"{b}, {c}, 4, {s}, {s}"),
        )
        ops = [_op("conv3d", ["self.op"], ("aten::convolution", ""))]
    elif spec.kind == "deconv2d":
        code = _module(
            [f"self.op = nn.ConvTranspose2d({c}, {c + 4}, 3, padding=1)"],
            ["return self.op(x)"],
            _single(f"{b}, {c}, {s}, {s}"),
        )
        ops = [_op("conv_transpose2d", ["self.op"], ("aten::convolution", ""))]
    elif spec.kind == "conv2d":
        code = _module(
            [f"self.op = nn.Conv2d({c}, {c + 4}, 3, padding=1)"], ["return self.op(x)"], _single(f"{b}, {c}, {s}, {s}")
        )
        ops = [_op("conv2d", ["self.op"], ("aten::convolution", ""))]
    elif spec.kind in {"bn1d", "bn2d", "bn3d"}:
        rank = spec.kind[-2]
        suffix = {"1": f"{b}, {c}, {length}", "2": f"{b}, {c}, {s}, {s}", "3": f"{b}, {c}, 4, {s}, {s}"}[rank]
        name = f"BatchNorm{rank}d"
        code = _module([f"self.op = nn.{name}({c})"], ["return self.op(x)"], _single(suffix))
        ops = [_op("batch_norm", ["self.op"], *_BATCH_NORM)]
    elif spec.kind == "groupnorm":
        code = _module([f"self.op = nn.GroupNorm(4, {c})"], ["return self.op(x)"], _single(f"{b}, {c}, {s}, {s}"))
        ops = [_op("group_norm", ["self.op"], ("aten::native_group_norm", ""))]
    elif spec.kind == "instancenorm":
        code = _module(
            [f"self.op = nn.InstanceNorm2d({c}, affine=True)"], ["return self.op(x)"], _single(f"{b}, {c}, {s}, {s}")
        )
        ops = [_op("instance_norm", ["self.op"], *_INSTANCE_NORM)]
    elif spec.kind == "dropout":
        code = _module(["self.op = nn.Dropout(p=0.25)"], ["return self.op(x)"], _single(f"{b}, {c}, {s}, {s}"))
        ops = [_op("dropout", ["self.op"], ("aten::native_dropout", ""))]
    elif spec.kind in {"maxpool3d", "avgpool3d", "adaptiveavgpool3d"}:
        ctor, identity = {
            "maxpool3d": ("nn.MaxPool3d(2)", ("aten::max_pool3d_with_indices", "")),
            "avgpool3d": ("nn.AvgPool3d(2)", ("aten::avg_pool3d", "")),
            "adaptiveavgpool3d": ("nn.AdaptiveAvgPool3d((2, 3, 3))", ("aten::_adaptive_avg_pool3d", "")),
        }[spec.kind]
        code = _module([f"self.op = {ctor}"], ["return self.op(x)"], _single(f"{b}, {c}, 4, {s}, {s}"))
        ops = [_op(spec.kind, ["self.op"], identity)]
    elif spec.kind in {"div", "sub"}:
        expression = "x / y" if spec.kind == "div" else "x - y"
        inputs = [
            f"x = torch.rand({b}, {c}, {s}, {s})",
            (
                f"y = torch.rand({b}, {c}, {s}, {s})"
                if spec.kind == "sub"
                else f"y = torch.rand({b}, {c}, {s}, {s}) + 0.25"
            ),
            "return [x, y]",
        ]
        code = _module([], [f"return {expression}"], inputs, "self, x, y")
        ops = [_op(spec.kind, [f"operator.{spec.kind}"], (f"aten::{spec.kind}", "Tensor"))]
        arity = 2
    elif spec.kind == "sigmoid":
        code = _module([], ["return torch.sigmoid(x)"], _single(f"{b}, {c}, {s}, {s}"))
        ops = [_op("sigmoid", ["torch.sigmoid"], ("aten::sigmoid", ""))]
    elif spec.kind == "flatten":
        code = _module([], ["return torch.flatten(x, 1)"], _single(f"{b}, {c}, {s}, {s}"))
        ops = [_op("flatten", ["torch.flatten"], ("aten::view", ""), ("aten::reshape", ""))]
    elif spec.kind == "contiguous":
        code = _module([], ["return x.contiguous()"], [f"return [torch.rand({b}, {s}, {c}).transpose(1, 2)]"])
        ops = [_op("contiguous", ["x.contiguous"], ("aten::clone", ""), ("aten::copy_", ""))]
    elif spec.kind == "unfold":
        code = _module([], ["return x.unfold(2, 2, 2)"], _single(f"{b}, {c}, {length}"))
        ops = [_op("unfold", ["x.unfold"], ("aten::unfold", ""))]
    elif spec.kind == "channel_permute":
        code = _module(
            [],
            [
                f"y = x.view({b}, 2, {c // 2}, {s}, {s})",
                "y = y.transpose(1, 2)",
                "y = y.contiguous()",
                f"return y.view({b}, {c}, {s}, {s})",
            ],
            [f"return [torch.rand({b}, {c}, {s}, {s})]"],
        )
        ops = [
            _op("views", ["x.view", "y.view"], ("aten::view", ""), min_calls=2),
            _op("transpose", ["y.transpose"], ("aten::transpose", "int")),
            _op("contiguous", ["y.contiguous"], ("aten::clone", "")),
        ]
    elif spec.kind == "forward_randn":
        code = _module(
            [],
            ["noise = torch.randn(x.shape, device=x.device, dtype=x.dtype)", "return x + noise"],
            _single(f"{b}, {c}, {s}, {s}"),
        )
        ops = [
            _op("forward_randn", ["torch.randn"], ("aten::randn", "")),
            _op("add_input", ["operator.add"], ("aten::add", "Tensor")),
        ]
    elif spec.kind == "conv2d_bn_pool":
        code = _module(
            [
                f"self.conv = nn.Conv2d({c}, {c + 4}, 3, padding=1)",
                f"self.norm = nn.BatchNorm2d({c + 4})",
                "self.pool = nn.MaxPool2d(2)",
            ],
            ["y = self.conv(x)", "y = self.norm(y)", "y = torch.relu(y)", "return self.pool(y)"],
            _single(f"{b}, {c}, {s}, {s}"),
        )
        ops = [
            _op("conv", ["self.conv"], ("aten::convolution", "")),
            _op("batch_norm", ["self.norm"], *_BATCH_NORM),
            _op("relu", ["torch.relu"], ("aten::relu", "")),
            _op("max_pool", ["self.pool"], ("aten::max_pool2d_with_indices", "")),
        ]
    elif spec.kind == "depthwise_bn_pointwise":
        code = _module(
            [
                f"self.depthwise = nn.Conv2d({c}, {c}, 3, padding=1, groups={c})",
                f"self.norm = nn.BatchNorm2d({c})",
                f"self.pointwise = nn.Conv2d({c}, {c + 4}, 1)",
            ],
            ["y = self.depthwise(x)", "y = self.norm(y)", "y = F.silu(y)", "return self.pointwise(y)"],
            _single(f"{b}, {c}, {s}, {s}"),
        )
        ops = [
            _op("convolutions", ["self.depthwise", "self.pointwise"], ("aten::convolution", ""), min_calls=2),
            _op("batch_norm", ["self.norm"], *_BATCH_NORM),
            _op("silu", ["F.silu"], ("aten::silu", "")),
        ]
    elif spec.kind == "conv3d_bn_pool":
        code = _module(
            [
                f"self.conv = nn.Conv3d({c}, {c + 4}, 3, padding=1)",
                f"self.norm = nn.BatchNorm3d({c + 4})",
                "self.pool = nn.MaxPool3d(2)",
            ],
            ["y = self.conv(x)", "y = self.norm(y)", "return self.pool(torch.relu(y))"],
            _single(f"{b}, {c}, 4, {s}, {s}"),
        )
        ops = [
            _op("conv3d", ["self.conv"], ("aten::convolution", "")),
            _op("batch_norm", ["self.norm"], *_BATCH_NORM),
            _op("relu", ["torch.relu"], ("aten::relu", "")),
            _op("max_pool3d", ["self.pool"], ("aten::max_pool3d_with_indices", "")),
        ]
    elif spec.kind == "deconv2d_groupnorm":
        code = _module(
            [f"self.up = nn.ConvTranspose2d({c}, {c + 4}, 3, padding=1)", f"self.norm = nn.GroupNorm(4, {c + 4})"],
            ["return F.gelu(self.norm(self.up(x)))"],
            _single(f"{b}, {c}, {s}, {s}"),
        )
        ops = [
            _op("conv_transpose2d", ["self.up"], ("aten::convolution", "")),
            _op("group_norm", ["self.norm"], ("aten::native_group_norm", "")),
            _op("gelu", ["F.gelu"], ("aten::gelu", "")),
        ]
    elif spec.kind == "deconv3d_batchnorm":
        code = _module(
            [f"self.up = nn.ConvTranspose3d({c}, {c + 4}, 3, padding=1)", f"self.norm = nn.BatchNorm3d({c + 4})"],
            ["return F.silu(self.norm(self.up(x)))"],
            _single(f"{b}, {c}, 4, {s}, {s}"),
        )
        ops = [
            _op("conv_transpose3d", ["self.up"], ("aten::convolution", "")),
            _op("batch_norm3d", ["self.norm"], *_BATCH_NORM),
            _op("silu", ["F.silu"], ("aten::silu", "")),
        ]
    elif spec.kind == "conv1d_batchnorm":
        code = _module(
            [
                f"self.conv = nn.Conv1d({c}, {c + 4}, 3, padding=1)",
                f"self.norm = nn.BatchNorm1d({c + 4})",
                "self.pool = nn.MaxPool1d(2)",
            ],
            ["y = self.norm(self.conv(x))", "return self.pool(F.gelu(y))"],
            _single(f"{b}, {c}, {length}"),
        )
        ops = [
            _op("conv1d", ["self.conv"], ("aten::convolution", "")),
            _op("batch_norm1d", ["self.norm"], *_BATCH_NORM),
            _op("gelu", ["F.gelu"], ("aten::gelu", "")),
            _op(
                "max_pool1d",
                ["self.pool"],
                ("aten::max_pool1d_with_indices", ""),
                ("aten::max_pool2d_with_indices", ""),
            ),
        ]
    elif spec.kind.startswith("long"):
        rank = spec.kind[-2]
        if rank == "2":
            code = _long_2d(b, c, s)
            ops = [
                _op(
                    "convolutions",
                    [
                        "self.conv1",
                        "self.conv2",
                        "self.proj",
                        "self.refine1",
                        "self.refine2",
                        "self.tail",
                        "self.final1",
                        "self.final2",
                        "self.final3",
                        "self.final4",
                    ],
                    ("aten::convolution", ""),
                    min_calls=10,
                ),
                _op("batch_norm", ["self.norm1", "self.norm2", "self.norm3"], *_BATCH_NORM, min_calls=3),
            ]
        elif rank == "3":
            code = _long_3d(b, c, s)
            ops = [
                _op(
                    "convolutions",
                    [
                        "self.conv1",
                        "self.conv2",
                        "self.proj",
                        "self.refine1",
                        "self.refine2",
                        "self.tail",
                        "self.final1",
                        "self.final2",
                        "self.final3",
                        "self.final4",
                        "self.final5",
                    ],
                    ("aten::convolution", ""),
                    min_calls=11,
                ),
                _op("batch_norm", ["self.norm1", "self.norm2", "self.norm3", "self.norm4"], *_BATCH_NORM, min_calls=4),
            ]
        else:
            code = _long_1d(b, c, length)
            ops = [
                _op(
                    "convolutions",
                    [
                        "self.conv1",
                        "self.conv2",
                        "self.proj",
                        "self.refine1",
                        "self.refine2",
                        "self.tail",
                        "self.final1",
                        "self.final2",
                        "self.final3",
                        "self.final4",
                        "self.final5",
                    ],
                    ("aten::convolution", ""),
                    min_calls=11,
                ),
                _op("batch_norm", ["self.norm1", "self.norm2", "self.norm3", "self.norm4"], *_BATCH_NORM, min_calls=4),
            ]
    elif spec.kind == "convnorm_stage":
        code = _inject_single_input(_multiclass_stage(b, c, s), f"{b}, {c}, {s}, {s}")
        ops = [
            _op(
                "stage_convolutions",
                ["self.stem", "self.head", "self.stage", "self.refine", "self.out"],
                ("aten::convolution", ""),
                min_calls=6,
            ),
            _op(
                "batch_norm",
                ["self.norm", "self.stage", "self.refine_norm", "self.tail_norm"],
                *_BATCH_NORM,
                min_calls=5,
            ),
        ]
    elif spec.kind == "residual_projection":
        code = _inject_single_input(_multiclass_residual(b, c, s), f"{b}, {c}, {s}, {s}")
        ops = [
            _op(
                "residual_convolutions",
                ["self.stem", "self.tail", "self.block", "self.refine", "self.final", "self.post1", "self.post2"],
                ("aten::convolution", ""),
                min_calls=8,
            ),
            _op(
                "batch_norm",
                ["self.norm", "self.block", "self.refine_norm", "self.final_norm"],
                *_BATCH_NORM,
                min_calls=4,
            ),
        ]
    elif spec.kind == "feature_spatial_mixer":
        code = _inject_single_input(_multiclass_mixer(b, c, s), f"{b}, {c}, {s}, {s}")
        ops = [
            _op(
                "mixer_convolutions",
                ["self.stem", "self.tail", "self.gate", "self.mixer", "self.refine", "self.final"],
                ("aten::convolution", ""),
                min_calls=7,
            ),
            _op(
                "group_norm",
                ["self.norm", "self.gate", "self.refine_norm"],
                ("aten::native_group_norm", ""),
                min_calls=3,
            ),
        ]
    else:
        raise ValueError(f"unknown kind:{spec.kind}")
    labels = {
        "template_id": template_id,
        "variant": variant,
        "primary_family": spec.family,
        "input_arity": arity,
        "input_factory": "torch.rand",
        "final_output_kind": "single_tensor",
        "skeleton_id": f"{spec.kind}:{arity}",
        "template_skeleton": spec.kind,
    }
    return (
        code,
        ops,
        labels,
        {
            "template_id": template_id,
            "variant": variant,
            "kind": spec.kind,
            "skeleton_id": labels["skeleton_id"],
            "template_skeleton": spec.kind,
            "required_signature": list(spec.required_signature),
            "required_structure": dict(spec.required_structure),
        },
    )


def _long_2d(b: int, c: int, s: int) -> str:
    return _module(
        [
            f"self.conv1 = nn.Conv2d({c}, {c}, 3, padding=1)",
            f"self.norm1 = nn.BatchNorm2d({c})",
            f"self.conv2 = nn.Conv2d({c}, {c}, 3, padding=1)",
            f"self.norm2 = nn.BatchNorm2d({c})",
            f"self.proj = nn.Conv2d({c}, {c}, 1)",
            f"self.refine1 = nn.Conv2d({c}, {c}, 3, padding=1)",
            f"self.norm3 = nn.BatchNorm2d({c})",
            f"self.refine2 = nn.Conv2d({c}, {c}, 1)",
            f"self.tail = nn.Conv2d({c}, {c}, 3, padding=1)",
            *(f"self.final{i} = nn.Conv2d({c}, {c}, 1)" for i in range(1, 5)),
            "self.pool = nn.MaxPool2d(2)",
            "self.up = nn.Upsample(scale_factor=2, mode='nearest')",
            "self.act1 = nn.ReLU()",
            "self.act2 = nn.GELU()",
        ],
        [
            "a = self.conv1(x)",
            "a = self.norm1(a)",
            "a = self.act1(a)",
            "b = self.conv2(a)",
            "b = self.norm2(b)",
            "b = self.act2(b)",
            "r = self.proj(x)",
            "y = b + r",
            "y = self.pool(y)",
            "y = self.up(y)",
            "z = self.refine1(y)",
            "z = self.norm3(z)",
            "z = self.act2(z)",
            "z = self.refine2(z)",
            "y = self.tail(z) + y",
            "y = self.final1(y)",
            "y = self.act1(y)",
            "y = self.final2(y)",
            "y = self.act2(y)",
            "y = self.final3(y)",
            "y = self.act1(y)",
            "return self.final4(y)",
        ],
        _single(f"{b}, {c}, {s}, {s}"),
    )


def _long_3d(b: int, c: int, s: int) -> str:
    return _module(
        [
            f"self.conv1 = nn.Conv3d({c}, {c}, 3, padding=1)",
            f"self.norm1 = nn.BatchNorm3d({c})",
            f"self.conv2 = nn.Conv3d({c}, {c}, 3, padding=1)",
            f"self.norm2 = nn.BatchNorm3d({c})",
            f"self.proj = nn.Conv3d({c}, {c}, 1)",
            f"self.refine1 = nn.Conv3d({c}, {c}, 3, padding=1)",
            f"self.norm3 = nn.BatchNorm3d({c})",
            f"self.refine2 = nn.Conv3d({c}, {c}, 1)",
            f"self.tail = nn.Conv3d({c}, {c}, 3, padding=1)",
            f"self.final1 = nn.Conv3d({c}, {c}, 1)",
            f"self.norm4 = nn.BatchNorm3d({c})",
            *(f"self.final{i} = nn.Conv3d({c}, {c}, 3, padding=1)" for i in range(2, 6)),
            "self.pool = nn.MaxPool3d(2)",
            "self.act1 = nn.ReLU()",
            "self.act2 = nn.SiLU()",
        ],
        [
            "a = self.conv1(x)",
            "a = self.norm1(a)",
            "a = self.act1(a)",
            "b = self.conv2(a)",
            "b = self.norm2(b)",
            "b = self.act2(b)",
            "r = self.proj(x)",
            "y = b + r",
            "y = self.pool(y)",
            "z = self.refine1(y)",
            "z = self.norm3(z)",
            "z = self.act2(z)",
            "z = self.refine2(z)",
            "y = self.tail(z) + y",
            "y = self.final1(y)",
            "y = self.norm4(y)",
            "y = self.final2(y)",
            "y = self.act1(y)",
            "y = self.final3(y)",
            "y = self.act2(y)",
            "y = self.final4(y)",
            "y = self.act1(y)",
            "return self.final5(y)",
        ],
        _single(f"{b}, {c}, 4, {s}, {s}"),
    )


def _long_1d(b: int, c: int, length: int) -> str:
    return _module(
        [
            f"self.conv1 = nn.Conv1d({c}, {c}, 3, padding=1)",
            f"self.norm1 = nn.BatchNorm1d({c})",
            f"self.conv2 = nn.Conv1d({c}, {c}, 3, padding=1)",
            f"self.norm2 = nn.BatchNorm1d({c})",
            f"self.proj = nn.Conv1d({c}, {c}, 1)",
            f"self.refine1 = nn.Conv1d({c}, {c}, 3, padding=1)",
            f"self.norm3 = nn.BatchNorm1d({c})",
            f"self.refine2 = nn.Conv1d({c}, {c}, 1)",
            f"self.tail = nn.Conv1d({c}, {c}, 3, padding=1)",
            f"self.final1 = nn.Conv1d({c}, {c}, 1)",
            f"self.norm4 = nn.BatchNorm1d({c})",
            *(f"self.final{i} = nn.Conv1d({c}, {c}, 3, padding=1)" for i in range(2, 6)),
            "self.pool = nn.MaxPool1d(2)",
            "self.act1 = nn.ReLU()",
            "self.act2 = nn.GELU()",
        ],
        [
            "a = self.conv1(x)",
            "a = self.norm1(a)",
            "a = self.act1(a)",
            "b = self.conv2(a)",
            "b = self.norm2(b)",
            "b = self.act2(b)",
            "r = self.proj(x)",
            "y = b + r",
            "y = self.pool(y)",
            "z = self.refine1(y)",
            "z = self.norm3(z)",
            "z = self.act2(z)",
            "z = self.refine2(z)",
            "y = self.tail(z) + y",
            "y = self.final1(y)",
            "y = self.norm4(y)",
            "y = self.final2(y)",
            "y = self.act1(y)",
            "y = self.final3(y)",
            "y = self.act2(y)",
            "y = self.final4(y)",
            "y = self.act1(y)",
            "return self.final5(y)",
        ],
        _single(f"{b}, {c}, {length}"),
    )


def _inject_single_input(code: str, shape: str) -> str:
    marker = f"def get_inputs():\n    return [torch.rand({shape})]"
    replacement = "def get_inputs():\n" + "\n".join(f"    {line}" for line in _single(shape))
    if marker not in code:
        raise ValueError("multiclass template lacks the expected input factory")
    return code.replace(marker, replacement, 1)


def _multiclass_stage(b: int, c: int, s: int) -> str:
    return "\n".join(
        [
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            "",
            "class ConvNormStage(nn.Module):",
            "    def __init__(self, channels):",
            "        super().__init__()",
            "        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)",
            "        self.norm1 = nn.BatchNorm2d(channels)",
            "        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)",
            "        self.norm2 = nn.BatchNorm2d(channels)",
            "",
            "    def forward(self, x):",
            "        y = torch.relu(self.norm1(self.conv1(x)))",
            "        return torch.relu(self.norm2(self.conv2(y)))",
            "",
            "class Model(nn.Module):",
            "    def __init__(self):",
            "        super().__init__()",
            f"        self.stem = nn.Conv2d({c}, {c}, 1)",
            f"        self.norm = nn.BatchNorm2d({c})",
            f"        self.stage = ConvNormStage({c})",
            f"        self.head = nn.Conv2d({c}, {c}, 3, padding=1)",
            f"        self.refine = nn.Conv2d({c}, {c}, 3, padding=1)",
            f"        self.refine_norm = nn.BatchNorm2d({c})",
            f"        self.tail_norm = nn.BatchNorm2d({c})",
            f"        self.out = nn.Conv2d({c}, {c}, 1)",
            "        self.pool = nn.MaxPool2d(2)",
            "        self.up = nn.Upsample(scale_factor=2, mode='nearest')",
            "        self.act = nn.GELU()",
            "",
            "    def forward(self, x):",
            "        y = self.stem(x)",
            "        y = self.norm(y)",
            "        y = self.act(y)",
            "        y = self.stage(y)",
            "        y = self.head(y)",
            "        y = self.refine(y)",
            "        y = self.refine_norm(y)",
            "        y = F.gelu(y)",
            "        y = self.tail_norm(y)",
            "        y = self.out(y)",
            "        y = self.pool(y)",
            "        y = self.up(y)",
            "        return y",
            "",
            "def get_inputs():",
            f"    return [torch.rand({b}, {c}, {s}, {s})]",
            "",
            "def get_init_inputs():",
            "    return []",
            "",
        ]
    )


def _multiclass_residual(b: int, c: int, s: int) -> str:
    return "\n".join(
        [
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            "",
            "class ResidualProjection(nn.Module):",
            "    def __init__(self, channels):",
            "        super().__init__()",
            "        self.left = nn.Conv2d(channels, channels, 3, padding=1)",
            "        self.norm = nn.BatchNorm2d(channels)",
            "        self.right = nn.Conv2d(channels, channels, 1)",
            "",
            "    def forward(self, x):",
            "        y = torch.relu(self.norm(self.left(x)))",
            "        return y + self.right(x)",
            "",
            "class Model(nn.Module):",
            "    def __init__(self):",
            "        super().__init__()",
            f"        self.stem = nn.Conv2d({c}, {c}, 1)",
            f"        self.norm = nn.BatchNorm2d({c})",
            f"        self.block = ResidualProjection({c})",
            f"        self.tail = nn.Conv2d({c}, {c}, 3, padding=1)",
            f"        self.refine = nn.Conv2d({c}, {c}, 3, padding=1)",
            f"        self.refine_norm = nn.BatchNorm2d({c})",
            f"        self.final = nn.Conv2d({c}, {c}, 1)",
            f"        self.final_norm = nn.BatchNorm2d({c})",
            f"        self.post1 = nn.Conv2d({c}, {c}, 3, padding=1)",
            f"        self.post2 = nn.Conv2d({c}, {c}, 1)",
            "        self.pool = nn.AvgPool2d(2)",
            "        self.up = nn.Upsample(scale_factor=2, mode='nearest')",
            "        self.act = nn.SiLU()",
            "",
            "    def forward(self, x):",
            "        y = self.act(self.norm(self.stem(x)))",
            "        y = self.block(y)",
            "        y = self.tail(y)",
            "        y = self.act(self.refine_norm(self.refine(y)))",
            "        y = self.final_norm(self.final(y))",
            "        y = self.act(self.post1(y))",
            "        y = self.post2(y)",
            "        y = self.up(self.pool(y))",
            "        return y",
            "",
            "def get_inputs():",
            f"    return [torch.rand({b}, {c}, {s}, {s})]",
            "",
            "def get_init_inputs():",
            "    return []",
            "",
        ]
    )


def _multiclass_mixer(b: int, c: int, s: int) -> str:
    return "\n".join(
        [
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            "",
            "class FeatureGate(nn.Module):",
            "    def __init__(self, channels):",
            "        super().__init__()",
            "        self.proj = nn.Conv2d(channels, channels, 1)",
            "        self.norm = nn.GroupNorm(4, channels)",
            "",
            "    def forward(self, x):",
            "        return torch.sigmoid(self.norm(self.proj(x)))",
            "",
            "class SpatialMixer(nn.Module):",
            "    def __init__(self, channels):",
            "        super().__init__()",
            "        self.depth = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)",
            "        self.point = nn.Conv2d(channels, channels, 1)",
            "",
            "    def forward(self, x):",
            "        return self.point(F.gelu(self.depth(x)))",
            "",
            "class Model(nn.Module):",
            "    def __init__(self):",
            "        super().__init__()",
            f"        self.stem = nn.Conv2d({c}, {c}, 1)",
            f"        self.norm = nn.GroupNorm(4, {c})",
            f"        self.gate = FeatureGate({c})",
            f"        self.mixer = SpatialMixer({c})",
            f"        self.tail = nn.Conv2d({c}, {c}, 3, padding=1)",
            f"        self.refine = nn.Conv2d({c}, {c}, 3, padding=1)",
            f"        self.refine_norm = nn.GroupNorm(4, {c})",
            f"        self.final = nn.Conv2d({c}, {c}, 1)",
            "        self.pool = nn.MaxPool2d(2)",
            "        self.up = nn.Upsample(scale_factor=2, mode='nearest')",
            "",
            "    def forward(self, x):",
            "        y = self.norm(self.stem(x))",
            "        g = self.gate(y)",
            "        y = self.mixer(y)",
            "        y = y * g",
            "        y = self.tail(y)",
            "        y = self.refine(y)",
            "        y = self.refine_norm(y)",
            "        y = F.gelu(y)",
            "        y = self.final(y)",
            "        y = self.pool(y)",
            "        return self.up(y)",
            "",
            "def get_inputs():",
            f"    return [torch.rand({b}, {c}, {s}, {s})]",
            "",
            "def get_init_inputs():",
            "    return []",
            "",
        ]
    )


class _ConstantAndNameEraser(ast.NodeTransformer):
    def __init__(self) -> None:
        self.names: dict[str, str] = {}

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in {"torch", "True", "False", "None"}:
            return node
        if node.id not in self.names:
            self.names[node.id] = f"v{len(self.names)}"
        return ast.copy_location(ast.Name(id=self.names[node.id], ctx=node.ctx), node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        return ast.copy_location(ast.Constant(value="<const>"), node)


def _get_inputs_skeleton(code: str) -> tuple[str, str]:
    tree = ast.parse(code)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_inputs"]
    if len(functions) != 1:
        raise ValueError("expected exactly one top-level get_inputs")
    normalized = _ConstantAndNameEraser().visit(ast.fix_missing_locations(functions[0]))
    dump = ast.dump(normalized, annotate_fields=True, include_attributes=False)
    return dump, _sha256_bytes(dump.encode())


def _coarse_input_profile(code: str) -> str:
    tree = ast.parse(code)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_inputs")
    calls = {_call_name(node.func) for node in ast.walk(function) if isinstance(node, ast.Call)}
    slices = [node.slice for node in ast.walk(function) if isinstance(node, ast.Subscript)]
    if "x.expand" in calls:
        source = "broadcast_expand"
    elif "x.movedim" in calls:
        source = "channel_last_view"
    elif any(
        isinstance(item, ast.Tuple)
        and any(isinstance(part, ast.Slice) and part.step is not None for part in item.elts)
        for item in slices
    ):
        source = "strided_slice"
    elif any(isinstance(item, ast.Slice) and item.lower is not None for item in slices):
        source = "offset_slice"
    else:
        source = "direct_rand"
    arity = _input_return_arity(function)
    binding = (
        "append"
        if any(isinstance(node, ast.Call) and _call_name(node.func).endswith("append") for node in ast.walk(function))
        else (
            "local"
            if any(
                isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "inputs" for target in node.targets)
                for node in ast.walk(function)
            )
            else "direct"
        )
    )
    ranks = _torch_rand_ndims(function)
    return f"{source}|arity={arity}|ranks={','.join(map(str, ranks))}|return={binding}"


def _torch_rand_ndims(function: ast.FunctionDef) -> list[int]:
    """Resolve tensor ndim for the closed shape-binding grammar in `_single`."""
    bound_ranks: dict[str, int] = {}
    for statement in function.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, (ast.List, ast.Tuple))
        ):
            bound_ranks[statement.targets[0].id] = len(statement.value.elts)

    ranks: set[int] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Call) or _call_name(node.func) != "torch.rand":
            continue
        if len(node.args) > 1:
            ranks.add(len(node.args))
            continue
        if len(node.args) != 1:
            raise ValueError("torch.rand rank is not statically bound")
        argument = node.args[0]
        if isinstance(argument, (ast.List, ast.Tuple)):
            ranks.add(len(argument.elts))
        elif isinstance(argument, ast.Name) and argument.id in bound_ranks:
            ranks.add(bound_ranks[argument.id])
        elif (
            isinstance(argument, ast.Call)
            and _call_name(argument.func) == "tuple"
            and len(argument.args) == 1
            and isinstance(argument.args[0], ast.Name)
            and argument.args[0].id in bound_ranks
        ):
            ranks.add(bound_ranks[argument.args[0].id])
        else:
            raise ValueError("torch.rand shape binding is outside the closed grammar")
    if not ranks:
        raise ValueError("get_inputs has no resolved torch.rand rank")
    return sorted(ranks)


def _input_return_arity(function: ast.FunctionDef) -> int:
    """Fail closed on a small, explicit local-list return grammar."""
    locals_: dict[str, int] = {}
    returns: list[int] = []
    for statement in function.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and isinstance(statement.value, ast.List)
        ):
            locals_[statement.targets[0].id] = len(statement.value.elts)
        elif (
            isinstance(statement, ast.Expr)
            and isinstance(statement.value, ast.Call)
            and isinstance(statement.value.func, ast.Attribute)
            and statement.value.func.attr == "append"
            and isinstance(statement.value.func.value, ast.Name)
        ):
            name = statement.value.func.value.id
            if name not in locals_ or len(statement.value.args) != 1 or statement.value.keywords:
                raise ValueError("unsupported local list append")
            locals_[name] += 1
        elif isinstance(statement, ast.Return):
            if isinstance(statement.value, ast.List):
                returns.append(len(statement.value.elts))
            elif isinstance(statement.value, ast.Name) and statement.value.id in locals_:
                returns.append(locals_[statement.value.id])
            else:
                raise ValueError("get_inputs must return a literal or locally-built list")
    if len(returns) != 1 or returns[0] < 1:
        raise ValueError("get_inputs must have one non-empty list return")
    return returns[0]


def _reachable_helper_classes(tree: ast.Module) -> set[str]:
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    if "Model" not in classes:
        raise ValueError("missing Model")
    reachable = {"Model"}
    pending = ["Model"]
    while pending:
        name = pending.pop()
        node = classes[name]
        for call in (item for item in ast.walk(node) if isinstance(item, ast.Call)):
            callee = _call_name(call.func)
            if callee in classes and callee not in reachable:
                reachable.add(callee)
                pending.append(callee)
    return reachable


def _static_contract(
    code: str, declared_ops: Sequence[Mapping[str, Any]], spec: Mapping[str, Any] | TemplateSpec | None = None
) -> dict[str, Any]:
    tree = ast.parse(code)
    models = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Model"]
    if len(models) != 1:
        raise ValueError("expected exactly one top-level Model")
    forwards = [node for node in models[0].body if isinstance(node, ast.FunctionDef) and node.name == "forward"]
    if len(forwards) != 1:
        raise ValueError("Model must define exactly one forward")
    if sum(isinstance(node, ast.Return) for node in ast.walk(forwards[0])) != 1:
        raise ValueError("Model.forward must have exactly one return")
    allowed_imports = {"torch", "torch.nn", "torch.nn.functional"}
    for node in tree.body:
        if isinstance(node, ast.Import) and any(alias.name not in allowed_imports for alias in node.names):
            raise ValueError("external import is forbidden")
        if isinstance(node, ast.ImportFrom):
            raise ValueError("from import is forbidden")
    if any(
        isinstance(node, ast.Call) and _call_name(node.func) in {"eval", "exec", "__import__", "getattr", "setattr"}
        for node in ast.walk(tree)
    ):
        raise ValueError("dynamic execution is forbidden")
    signatures = set(extract_operator_signature(code))
    calls = {_call_name(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    get_inputs = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_inputs"]
    if len(get_inputs) != 1:
        raise ValueError("get_inputs must be a unique top-level synchronous function")
    factory_names = {_call_name(node.func) for node in ast.walk(get_inputs[0]) if isinstance(node, ast.Call)}
    if "torch.randn" in factory_names or "torch.rand" not in factory_names:
        raise ValueError("get_inputs must use torch.rand and not torch.randn")
    input_arity = _input_return_arity(get_inputs[0])
    for declared in declared_ops:
        source_calls = set(declared["source_calls"])
        if not source_calls & (calls | signatures):
            raise ValueError(f"declared op lacks source witness:{declared['op_id']}")
    reachable = _reachable_helper_classes(tree)
    all_classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    if not all_classes.issubset(reachable):
        raise ValueError(f"unreachable helper class:{sorted(all_classes - reachable)}")
    skeleton, skeleton_hash = _get_inputs_skeleton(code)
    return {
        "single_value_return": True,
        "factory_calls": sorted(factory_names),
        "input_arity": input_arity,
        "reachable_helper_classes": sorted(reachable),
        "top_level_class_count": len(all_classes),
        "get_inputs_skeleton": skeleton,
        "get_inputs_skeleton_sha256": skeleton_hash,
        "coarse_input_profile": _coarse_input_profile(code),
        "realized_signature": list(extract_operator_signature(code)),
        "declared_source_witnesses": {
            item["op_id"]: sorted(set(item["source_calls"]) & (calls | signatures)) for item in declared_ops
        },
    }


def _enforce_spec(code: str, spec: TemplateSpec, static: Mapping[str, Any]) -> None:
    signature = tuple(extract_operator_signature(code))
    missing = set(spec.required_signature) - set(signature)
    if missing:
        raise ValueError(f"required signature missing:{spec.template_id}:{sorted(missing)}")
    features = feature_dict(extract_complexity_features(code))
    for name, minimum in spec.required_structure:
        if features[name] < minimum:
            raise ValueError(f"structural quota missing:{spec.template_id}:{name}={features[name]}<{minimum}")
    if spec.family == "modular_multiclass" and len(static["reachable_helper_classes"]) < 2:
        raise ValueError(f"multi-class helper is not reachable:{spec.template_id}")
    if spec.family == "atomic_low_level" and signature != spec.required_signature:
        raise ValueError(f"atomic signature mismatch:{spec.template_id}:{signature}!={spec.required_signature}")


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
    ast_hashes: set[str] = set()
    bindings: list[dict[str, Any]] = []
    for raw in paths:
        path = raw.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        parquet = pq.ParquetFile(path)
        if "reward_model" not in parquet.schema_arrow.names:
            raise ValueError(f"comparison lacks reward_model:{path}")
        parse_failures = 0
        for batch in parquet.iter_batches(columns=["reward_model"], batch_size=256):
            for item in batch.column(0).to_pylist():
                code = item.get("ground_truth") if isinstance(item, Mapping) else None
                if not isinstance(code, str) or not code.strip():
                    raise ValueError(f"comparison has empty reference:{path}")
                references.add(_sha256_bytes(code.encode()))
                try:
                    ast_hashes.add(_normalized_ast_sha256(code))
                except SyntaxError:
                    parse_failures += 1
        bindings.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "rows": parquet.metadata.num_rows,
                "normalized_ast_parse_failures": parse_failures,
            }
        )
    return references, ast_hashes, bindings


def build_records(
    *,
    prompt_prefix: str,
    comparison_reference_hashes: set[str],
    comparison_ast_hashes: set[str],
    template_binding: Mapping[str, Any],
    comparison_bindings: Sequence[Mapping[str, Any]],
    git_commit: str,
    generator_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pending: list[tuple[str, dict[str, Any], dict[str, Any]]] = []
    for spec in TEMPLATES:
        for variant in range(spec.variants):
            code, declared_ops, labels, rendered_spec = _render(spec.template_id, variant)
            static = _static_contract(code, declared_ops)
            _enforce_spec(code, spec, static)
            reference_sha256 = _sha256_bytes(code.encode())
            ast_sha256 = _normalized_ast_sha256(code)
            if reference_sha256 in comparison_reference_hashes or ast_sha256 in comparison_ast_hashes:
                raise ValueError(f"decontamination collision:{spec.template_id}:{variant}")
            uuid = "kbgap_" + _sha256_bytes(f"{CONTRACT_VERSION}|{spec.template_id}|{variant}".encode())[:24]
            content = prompt_prefix + code
            source_calls = sorted({name for op in declared_ops for name in op["source_calls"]})
            row = {
                "data_source": DATA_SOURCE,
                "prompt": [{"content": content, "role": "user"}],
                "ability": "kernel_optimization",
                "reward_model": {"ground_truth": code, "style": "rule"},
                "extra_info": {
                    "entry_point": "Model",
                    "level": "coverage",
                    "module_name": "Model",
                    "ops": json.dumps(source_calls),
                    "original_prompt": [{"content": content, "role": "user"}],
                    "repo_name": "kernelbench_gap_generator_v1",
                    "type": "semantic_operator_synthetic",
                    "uuid": uuid,
                },
            }
            manifest = {
                "manifest_contract_version": MANIFEST_VERSION,
                "generator_contract_version": CONTRACT_VERSION,
                "registry_version": REGISTRY_VERSION,
                "order_version": ORDER_VERSION,
                "static_contract_version": STATIC_CONTRACT_VERSION,
                "runtime_contract_version": RUNTIME_CONTRACT_VERSION,
                "method": METHOD,
                "uuid": uuid,
                "template_id": spec.template_id,
                "template_variant": variant,
                "template_skeleton": rendered_spec["template_skeleton"],
                "skeleton_id": rendered_spec["skeleton_id"],
                "primary_family": spec.family,
                "lineage_kind": "standalone_semantic_synthetic",
                "parent_uuid": None,
                "primary_intervention": "semantic_operator",
                "final_output_contract": {"kind": "single_tensor", "finite_required": True},
                "training_mode": True,
                "declared_ops": declared_ops,
                "required_signature": list(spec.required_signature),
                "required_structure": dict(spec.required_structure),
                "static_proof": static,
                "coverage_labels": labels,
                "reference_sha256": reference_sha256,
                "normalized_ast_sha256": ast_sha256,
                "get_inputs_skeleton_sha256": static["get_inputs_skeleton_sha256"],
                "prompt_sha256": _sha256_bytes(content.encode()),
                "row_payload_sha256": _canonical_sha256(row),
                "prompt_template": dict(template_binding),
                "decontamination_roots": [dict(x) for x in comparison_bindings],
                "decontamination_status": "exact_reference_and_normalized_ast_no_match",
                "provenance": {
                    "source_family": "project_generated_internal_review",
                    "generator": str(Path(__file__).resolve()),
                    "license": "internal-review-only",
                    "provenance_status": "project_generated_bound",
                    "kernelbench_source_used_as_template": False,
                },
                "git_commit_at_generation": git_commit,
                "generator_source_sha256": generator_sha256,
                "static_status": "passed",
                "reference_runtime_status": "pending",
                "operator_liveness_status": "pending",
                "materialization_status": "review_only",
                "training_approved": False,
                "structured_output_deferred": True,
            }
            manifest["mode_behavior"] = spec.mode_behavior
            manifest["required_signature"] = static["realized_signature"]
            manifest["template_required_signature"] = list(spec.required_signature)
            manifest["coarse_input_profile"] = static["coarse_input_profile"]
            pending.append((_sha256_bytes(f"{ORDER_VERSION}|{spec.template_id}|{variant}".encode()), row, manifest))
    pending.sort(key=lambda item: item[0])
    rows, manifests = [], []
    for index, (_, row, manifest) in enumerate(pending):
        manifest["candidate_row_index"] = index
        rows.append(row)
        manifests.append(manifest)
    return rows, manifests


def replay_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Re-render one manifest row for static/runtime analyzers without files."""
    code, ops, labels, spec = _render(str(manifest["template_id"]), int(manifest["template_variant"]))
    static = _static_contract(code, ops)
    template_spec = TEMPLATE_BY_ID[str(manifest["template_id"])]
    _enforce_spec(code, template_spec, static)
    if spec["skeleton_id"] != manifest.get("skeleton_id") or labels["skeleton_id"] != manifest.get("skeleton_id"):
        raise ValueError("manifest skeleton does not replay")
    if template_spec.mode_behavior != manifest.get("mode_behavior"):
        raise ValueError("manifest mode behavior does not replay")
    if list(static["realized_signature"]) != manifest.get("required_signature"):
        raise ValueError("manifest required signature does not replay")
    return {"code": code, "declared_ops": ops, "coverage_labels": labels, "static_proof": static, "spec": spec}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")


def _review_markdown(rows: Sequence[Mapping[str, Any]], manifests: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# KernelBench-gap canary samples",
        "",
        "The registry is generated from project-owned templates. KernelBench is a decontamination root, never a code template.",
        "",
        "| primary family | quota |",
        "| --- | ---: |",
        *(f"| {name} | {count} |" for name, count in FAMILY_QUOTAS.items()),
        "",
    ]
    for family in FAMILY_QUOTAS:
        selected = [
            (row, manifest)
            for row, manifest in zip(rows, manifests, strict=True)
            if manifest["primary_family"] == family
        ]
        for row, manifest in (selected[0], selected[-1]):
            lines.extend(
                [
                    f"## {manifest['template_id']} variant {manifest['template_variant']}",
                    "",
                    f"Skeleton: `{manifest['skeleton_id']}`",
                    "",
                    "```python",
                    row["reward_model"]["ground_truth"].rstrip(),
                    "```",
                    "",
                ]
            )
    return "\n".join(lines)


def generate(output_dir: Path, template_path: Path, comparison_paths: Sequence[Path]) -> dict[str, Any]:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    template_path = template_path.expanduser().resolve()
    _, prompt_prefix = _template(template_path)
    template_binding = {
        "path": str(template_path),
        "sha256": _sha256_file(template_path),
        "rows": pq.ParquetFile(template_path).metadata.num_rows,
    }
    references, ast_hashes, bindings = _comparison_hashes(comparison_paths)
    rows, manifests = build_records(
        prompt_prefix=prompt_prefix,
        comparison_reference_hashes=references,
        comparison_ast_hashes=ast_hashes,
        template_binding=template_binding,
        comparison_bindings=bindings,
        git_commit=_git_commit(),
        generator_sha256=_sha256_file(Path(__file__).resolve()),
    )
    counts = collections.Counter(item["primary_family"] for item in manifests)
    arity = collections.Counter(item["coverage_labels"]["input_arity"] for item in manifests)
    skeletons = collections.Counter(item["get_inputs_skeleton_sha256"] for item in manifests)
    coarse_profiles = collections.Counter(item["coarse_input_profile"] for item in manifests)
    if len(rows) != EXACT_CANARY_ROWS or dict(counts) != FAMILY_QUOTAS or dict(arity) != {1: 960, 2: 40}:
        raise ValueError(f"quota mismatch:rows={len(rows)} families={dict(counts)} arity={dict(arity)}")
    if len(skeletons) < 100 or max(skeletons.values()) > 20 or len(coarse_profiles) < 20:
        raise ValueError(
            f"input skeleton gate failed:unique={len(skeletons)} max_group={max(skeletons.values())} coarse={len(coarse_profiles)}"
        )
    for field in ("uuid", "reference_sha256", "normalized_ast_sha256", "row_payload_sha256"):
        values = [item[field] for item in manifests]
        if len(values) != len(set(values)):
            raise ValueError(f"non-unique {field}")
    candidates, manifest_path, review, summary_path = (
        output_dir / "candidates.parquet",
        output_dir / "manifest.jsonl",
        output_dir / "review_samples.md",
        output_dir / "summary.json",
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=_semantic_schema()), candidates, compression="zstd")
    _write_jsonl(manifest_path, manifests)
    review.write_text(_review_markdown(rows, manifests), encoding="utf-8")
    summary = {
        "contract_version": CONTRACT_VERSION,
        "method": METHOD,
        "registry_version": REGISTRY_VERSION,
        "rows": len(rows),
        "family_counts_exclusive_primary": dict(counts),
        "input_arity_counts": dict(arity),
        "input_factory_requirement": "torch.rand_only",
        "input_skeleton_gate": {
            "unique_minimum": 100,
            "unique_observed": len(skeletons),
            "largest_group_maximum": 20,
            "largest_group_observed": max(skeletons.values()),
            "coarse_profile_minimum": 20,
            "coarse_profile_observed": len(coarse_profiles),
        },
        "training_approved": False,
        "decontamination_roots": bindings,
        "artifacts": {
            "candidates": {"path": str(candidates), "sha256": _sha256_file(candidates)},
            "manifest": {"path": str(manifest_path), "sha256": _sha256_file(manifest_path)},
            "review_samples": {"path": str(review), "sha256": _sha256_file(review)},
        },
    }
    _write_json(summary_path, summary)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--template-parquet", type=Path, default=_DEFAULT_TEMPLATE)
    parser.add_argument("--comparison-artifact", type=Path, action="append", dest="comparison_artifacts")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    paths = tuple(args.comparison_artifacts) if args.comparison_artifacts else _DEFAULT_COMPARISON_ROOTS
    print(json.dumps(generate(args.output_dir, args.template_parquet, paths), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
