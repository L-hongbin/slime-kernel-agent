#!/usr/bin/env python3
"""Generate an exact 1k review-only semantic/operator canary.

This lane is parentless.  Every task comes from a closed deterministic
template registry, has a single Tensor as its final forward result, and binds
both a static source-call contract and an execution-side ATen contract.  The
runtime contract is proved separately before any row can be materialized.
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

from tools.data.cleaning.external import _template
from tools.data.synthesize.augment_prompt_tasks import _call_name, _normalized_ast_sha256

CONTRACT_VERSION = "semantic_operator_single_tensor_generator_v1"
MANIFEST_VERSION = "semantic_operator_parentless_manifest_v1"
REGISTRY_VERSION = "semantic_operator_closed_registry_33_templates_v1"
ORDER_VERSION = "sha256_contract_template_variant_v1"
STATIC_CONTRACT_VERSION = "straight_line_return_dependency_v1"
RUNTIME_CONTRACT_VERSION = "aten_dispatch_return_provenance_v1"
DATA_SOURCE = "project_generated_semantic_operator_v1"
MAX_AUTHORIZED_CANDIDATES = 5_000
EXACT_CANARY_ROWS = 1_000
_DEFAULT_TEMPLATE = _REPO_ROOT / "Data/prompt_tvm_v4/train.review.parquet"
_DEFAULT_COMPARISON_ROOTS = (
    _DEFAULT_TEMPLATE,
    _REPO_ROOT / "Data/prompt_tvm_v4/synthesis/accepted/extreme_ops_v3.parquet",
)

FAMILY_QUOTAS = {
    "attention_recurrent_stateful": 180,
    "conv_norm": 180,
    "heterogeneous_natural_dag": 160,
    "matmul_reduction": 180,
    "pool_index": 150,
    "shape_branch": 150,
}


@dataclasses.dataclass(frozen=True)
class TemplateSpec:
    template_id: str
    family: str
    variants: int
    mode_behavior: str


TEMPLATES = (
    *(
        TemplateSpec(f"CN0{i}", "conv_norm", 30, "train_stateful" if i in {1, 3, 4, 5} else "stateless")
        for i in range(1, 7)
    ),
    *(TemplateSpec(f"MR0{i}", "matmul_reduction", 30, "stateless") for i in range(1, 7)),
    *(TemplateSpec(f"PI0{i}", "pool_index", 30, "stateless") for i in range(1, 6)),
    *(TemplateSpec(f"SB0{i}", "shape_branch", 30, "stateless") for i in range(1, 6)),
    *(
        TemplateSpec(
            f"AR0{i}",
            "attention_recurrent_stateful",
            30,
            "train_stateful" if i == 6 else ("recurrent_state" if i in {4, 5} else "stateless"),
        )
        for i in range(1, 7)
    ),
    *(
        TemplateSpec(f"HD0{i}", "heterogeneous_natural_dag", 32, "train_stateful" if i == 1 else "stateless")
        for i in range(1, 6)
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
    return _sha256_bytes(_canonical_json(value).encode("utf-8"))


def _git_commit() -> str:
    value = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True, stderr=subprocess.DEVNULL
    ).strip()
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise RuntimeError(f"invalid Git commit:{value!r}")
    return value


def _identity(schema: str, overload: str = "") -> dict[str, str]:
    if not schema.startswith("aten::") or not isinstance(overload, str):
        raise ValueError(f"invalid ATen identity:{schema!r}:{overload!r}")
    return {"schema": schema, "overload": overload}


def _op(
    op_id: str,
    source_calls: Sequence[str],
    identities: Sequence[tuple[str, str]],
    *,
    min_calls: int = 1,
) -> dict[str, Any]:
    return {
        "op_id": op_id,
        "source_calls": list(source_calls),
        "runtime_identities": [_identity(*item) for item in identities],
        "min_calls_per_trial": min_calls,
        "must_reach_returned_output": True,
    }


def _module_code(
    *,
    init_lines: Sequence[str],
    forward_signature: str,
    forward_lines: Sequence[str],
    input_lines: Sequence[str],
) -> str:
    init = ["        super().__init__()", *(f"        {line}" for line in init_lines)]
    forward = [f"        {line}" for line in forward_lines]
    inputs = [f"    {line}" for line in input_lines]
    return (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n\n"
        "class Model(nn.Module):\n"
        "    def __init__(self):\n"
        + "\n".join(init)
        + "\n\n"
        + f"    def forward({forward_signature}):\n"
        + "\n".join(forward)
        + "\n\n"
        + "def get_inputs():\n"
        + "\n".join(inputs)
        + "\n\n"
        + "def get_init_inputs():\n"
        + "    return []\n"
    )


def _common_30(variant: int) -> tuple[int, int, int]:
    if not 0 <= variant < 30:
        raise ValueError(f"30-cell variant out of range:{variant}")
    channels = (4, 8, 12, 16, 20, 24)[variant % 6]
    spatial = (8, 10, 12, 14, 16)[variant // 6]
    batch = 2 + (variant % 3)
    return batch, channels, spatial


def _common_32(variant: int) -> tuple[int, int, int]:
    if not 0 <= variant < 32:
        raise ValueError(f"32-cell variant out of range:{variant}")
    channels = (4, 8, 12, 16, 20, 24, 28, 32)[variant % 8]
    spatial = (8, 10, 12, 14)[variant // 8]
    batch = 2 + (variant % 3)
    return batch, channels, spatial


def _render(template_id: str, variant: int) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    """Return code, declared op contracts, and exact shape/topology labels."""

    if template_id.startswith("HD"):
        batch, channels, spatial = _common_32(variant)
    else:
        batch, channels, spatial = _common_30(variant)

    convolution = [("aten::convolution", "")]
    batch_norm = [
        ("aten::native_batch_norm", ""),
        ("aten::_native_batch_norm_legit", ""),
        ("aten::_native_batch_norm_legit_functional", ""),
        ("aten::cudnn_batch_norm", ""),
    ]
    group_norm = [("aten::native_group_norm", "")]
    layer_norm = [("aten::native_layer_norm", "")]
    max_pool = [("aten::max_pool2d_with_indices", ""), ("aten::max_pool1d_with_indices", "")]
    adaptive_pool = [("aten::_adaptive_avg_pool2d", ""), ("aten::adaptive_avg_pool2d", "")]
    attention = [
        ("aten::scaled_dot_product_attention", ""),
        ("aten::_scaled_dot_product_flash_attention", ""),
        ("aten::_scaled_dot_product_efficient_attention", ""),
        ("aten::_scaled_dot_product_cudnn_attention", ""),
        ("aten::_scaled_dot_product_flash_attention_for_cpu", ""),
    ]

    if template_id == "CN01":
        out_channels = channels + 4
        code = _module_code(
            init_lines=[
                f"self.conv = nn.Conv2d({channels}, {out_channels}, 3, padding=1)",
                f"self.bn = nn.BatchNorm2d({out_channels})",
            ],
            forward_signature="self, x",
            forward_lines=["y = self.conv(x)", "y = self.bn(y)", "return torch.relu(y)"],
            input_lines=[f"return [torch.randn({batch}, {channels}, {spatial}, {spatial})]"],
        )
        ops = [
            _op("conv", ["self.conv"], convolution),
            _op("batch_norm", ["self.bn"], batch_norm),
            _op("relu", ["torch.relu"], [("aten::relu", "")]),
        ]
    elif template_id == "CN02":
        out_channels = channels + 4
        code = _module_code(
            init_lines=[
                f"self.depthwise = nn.Conv2d({channels}, {channels}, 3, padding=1, groups={channels})",
                f"self.norm = nn.GroupNorm(4, {channels})",
                f"self.pointwise = nn.Conv2d({channels}, {out_channels}, 1)",
            ],
            forward_signature="self, x",
            forward_lines=["y = self.depthwise(x)", "y = self.norm(y)", "y = F.silu(y)", "return self.pointwise(y)"],
            input_lines=[f"return [torch.randn({batch}, {channels}, {spatial}, {spatial})]"],
        )
        ops = [
            _op("convolutions", ["self.depthwise", "self.pointwise"], convolution, min_calls=2),
            _op("group_norm", ["self.norm"], group_norm),
            _op("silu", ["F.silu"], [("aten::silu", "")]),
        ]
    elif template_id == "CN03":
        out_channels = channels + 4
        length = spatial * 2 + 3
        code = _module_code(
            init_lines=[
                f"self.conv = nn.Conv1d({channels}, {out_channels}, 3, padding=1)",
                f"self.bn = nn.BatchNorm1d({out_channels})",
            ],
            forward_signature="self, x",
            forward_lines=["y = self.conv(x)", "y = self.bn(y)", "return F.gelu(y)"],
            input_lines=[f"return [torch.randn({batch}, {channels}, {length})]"],
        )
        ops = [
            _op("conv", ["self.conv"], convolution),
            _op("batch_norm", ["self.bn"], batch_norm),
            _op("gelu", ["F.gelu"], [("aten::gelu", "")]),
        ]
    elif template_id == "CN04":
        out_channels = channels + 4
        code = _module_code(
            init_lines=[
                f"self.conv = nn.ConvTranspose2d({channels}, {out_channels}, 3, stride=2, padding=1, output_padding=1)",
                f"self.norm = nn.InstanceNorm2d({out_channels}, affine=True, track_running_stats=True)",
            ],
            forward_signature="self, x",
            forward_lines=["y = self.conv(x)", "y = self.norm(y)", "return torch.tanh(y)"],
            input_lines=[f"return [torch.randn({batch}, {channels}, {spatial}, {spatial})]"],
        )
        ops = [
            _op("conv_transpose", ["self.conv"], convolution),
            _op("instance_norm", ["self.norm"], batch_norm),
            _op("tanh", ["torch.tanh"], [("aten::tanh", "")]),
        ]
    elif template_id == "CN05":
        in_channels = (2, 3, 4, 5, 6, 8)[variant % 6]
        out_channels = in_channels + 2
        depth = 4 + variant // 6
        code = _module_code(
            init_lines=[
                f"self.conv = nn.Conv3d({in_channels}, {out_channels}, 3, padding=1)",
                f"self.bn = nn.BatchNorm3d({out_channels})",
            ],
            forward_signature="self, x",
            forward_lines=["y = self.conv(x)", "y = self.bn(y)", "return F.relu(y)"],
            input_lines=[f"return [torch.randn({batch}, {in_channels}, {depth}, {spatial}, {spatial})]"],
        )
        ops = [
            _op("conv", ["self.conv"], convolution),
            _op("batch_norm", ["self.bn"], batch_norm),
            _op("relu", ["F.relu"], [("aten::relu", "")]),
        ]
    elif template_id == "CN06":
        code = _module_code(
            init_lines=[f"self.conv = nn.Conv2d({channels}, {channels}, 3, padding=1)"],
            forward_signature="self, x",
            forward_lines=[
                "y = self.conv(x)",
                f"y = F.layer_norm(y, [{channels}, {spatial}, {spatial}])",
                "return F.gelu(y)",
            ],
            input_lines=[f"return [torch.randn({batch}, {channels}, {spatial}, {spatial})]"],
        )
        ops = [
            _op("conv", ["self.conv"], convolution),
            _op("layer_norm", ["F.layer_norm"], layer_norm),
            _op("gelu", ["F.gelu"], [("aten::gelu", "")]),
        ]

    elif template_id == "MR01":
        m, k, n = spatial, channels + 3, spatial + 3
        code = _module_code(
            init_lines=[],
            forward_signature="self, x, w",
            forward_lines=["y = torch.matmul(x, w)", "return torch.sum(y, dim=-1)"],
            input_lines=[f"return [torch.randn({m}, {k}), torch.randn({k}, {n})]"],
        )
        ops = [
            _op("matmul", ["torch.matmul"], [("aten::mm", ""), ("aten::matmul", "")]),
            _op("sum", ["torch.sum"], [("aten::sum", "dim_IntList")]),
        ]
    elif template_id == "MR02":
        m, k, n = spatial, channels + 3, spatial + 2
        code = _module_code(
            init_lines=[],
            forward_signature="self, x, w",
            forward_lines=["y = torch.bmm(x, w)", "return torch.mean(y, dim=1)"],
            input_lines=[f"return [torch.randn({batch}, {m}, {k}), torch.randn({batch}, {k}, {n})]"],
        )
        ops = [_op("bmm", ["torch.bmm"], [("aten::bmm", "")]), _op("mean", ["torch.mean"], [("aten::mean", "dim")])]
    elif template_id == "MR03":
        m, k, n = spatial, channels + 3, spatial + 2
        code = _module_code(
            init_lines=[],
            forward_signature="self, x, w",
            forward_lines=["y = torch.einsum('bmk,bkn->bmn', x, w)", "return torch.amax(y, dim=-1)"],
            input_lines=[f"return [torch.randn({batch}, {m}, {k}), torch.randn({batch}, {k}, {n})]"],
        )
        ops = [
            _op("einsum_matmul", ["torch.einsum"], [("aten::bmm", "")]),
            _op("amax", ["torch.amax"], [("aten::amax", "")]),
        ]
    elif template_id == "MR04":
        features, out_features, length = channels + 3, channels + 7, spatial
        code = _module_code(
            init_lines=[f"self.linear = nn.Linear({features}, {out_features})"],
            forward_signature="self, x",
            forward_lines=["y = self.linear(x)", "return torch.logsumexp(y, dim=-1)"],
            input_lines=[f"return [torch.randn({batch}, {length}, {features})]"],
        )
        ops = [
            _op("linear", ["self.linear"], [("aten::addmm", ""), ("aten::linear", "")]),
            _op("logsumexp", ["torch.logsumexp"], [("aten::logsumexp", "")]),
        ]
    elif template_id == "MR05":
        m, k, n = spatial, channels + 3, spatial + 4
        code = _module_code(
            init_lines=[
                f"self.weight = nn.Parameter(torch.randn({k}, {n}) * 0.1)",
                f"self.bias = nn.Parameter(torch.zeros({n}))",
            ],
            forward_signature="self, x",
            forward_lines=[
                "y = torch.addmm(self.bias, x, self.weight)",
                "y = torch.softmax(y, dim=-1)",
                "return torch.sum(y[:, ::2], dim=-1)",
            ],
            input_lines=[f"return [torch.randn({m}, {k})]"],
        )
        ops = [
            _op("addmm", ["torch.addmm"], [("aten::addmm", "")]),
            _op("softmax", ["torch.softmax"], [("aten::_softmax", "")]),
            _op("sum", ["torch.sum"], [("aten::sum", "dim_IntList")]),
        ]
    elif template_id == "MR06":
        m, k, n = spatial, channels + 3, spatial + 2
        code = _module_code(
            init_lines=[],
            forward_signature="self, x, w",
            forward_lines=["y = torch.mm(x, w)", "y = torch.mean(y, dim=-1)", "return torch.clamp(y, -2.0, 2.0)"],
            input_lines=[f"return [torch.randn({m}, {k}) / {k ** 0.5:.8f}, torch.randn({k}, {n})]"],
        )
        ops = [
            _op("mm", ["torch.mm"], [("aten::mm", "")]),
            _op("mean", ["torch.mean"], [("aten::mean", "dim")]),
            _op("clamp", ["torch.clamp"], [("aten::clamp", "")]),
        ]

    elif template_id == "PI01":
        pooled = spatial // 2
        code = _module_code(
            init_lines=[],
            forward_signature="self, x, index",
            forward_lines=["y = F.max_pool2d(x, 2)", "return torch.gather(y, 3, index)"],
            input_lines=[
                f"index = torch.arange({pooled - 1}, -1, -1, dtype=torch.long).view(1, 1, 1, {pooled}).expand({batch}, {channels}, {pooled}, {pooled})",
                f"return [torch.randn({batch}, {channels}, {spatial}, {spatial}), index]",
            ],
        )
        ops = [_op("max_pool", ["F.max_pool2d"], max_pool), _op("gather", ["torch.gather"], [("aten::gather", "")])]
    elif template_id == "PI02":
        code = _module_code(
            init_lines=[],
            forward_signature="self, x, index",
            forward_lines=["y = F.avg_pool2d(x, 2)", "return torch.index_select(y, 1, index)"],
            input_lines=[
                f"index = torch.arange({channels - 1}, -1, -1, dtype=torch.long)",
                f"return [torch.randn({batch}, {channels}, {spatial}, {spatial}), index]",
            ],
        )
        ops = [
            _op("avg_pool", ["F.avg_pool2d"], [("aten::avg_pool2d", "")]),
            _op("index_select", ["torch.index_select"], [("aten::index_select", "")]),
        ]
    elif template_id == "PI03":
        target = 3 + (variant % 2)
        code = _module_code(
            init_lines=[],
            forward_signature="self, x, index",
            forward_lines=[
                f"y = F.adaptive_avg_pool2d(x, ({target}, {target}))",
                "return torch.take_along_dim(y, index, dim=2)",
            ],
            input_lines=[
                f"index = torch.arange({target - 1}, -1, -1, dtype=torch.long).view(1, 1, {target}, 1).expand({batch}, {channels}, {target}, {target})",
                f"return [torch.randn({batch}, {channels}, {spatial}, {spatial}), index]",
            ],
        )
        ops = [
            _op("adaptive_avg_pool", ["F.adaptive_avg_pool2d"], adaptive_pool),
            _op("take_along_dim", ["torch.take_along_dim"], [("aten::gather", ""), ("aten::take_along_dim", "")]),
        ]
    elif template_id == "PI04":
        length, pooled = spatial * 2, spatial
        code = _module_code(
            init_lines=[],
            forward_signature="self, x, index",
            forward_lines=["y = F.max_pool1d(x, 2)", "return torch.gather(y, 2, index)"],
            input_lines=[
                f"index = torch.arange({pooled - 1}, -1, -1, dtype=torch.long).view(1, 1, {pooled}).expand({batch}, {channels}, {pooled})",
                f"return [torch.randn({batch}, {channels}, {length}), index]",
            ],
        )
        ops = [_op("max_pool", ["F.max_pool1d"], max_pool), _op("gather", ["torch.gather"], [("aten::gather", "")])]
    elif template_id == "PI05":
        pooled = spatial // 2
        code = _module_code(
            init_lines=[],
            forward_signature="self, x, index",
            forward_lines=[
                "y = F.avg_pool2d(x, 2)",
                "base = torch.zeros_like(y)",
                "return torch.scatter(base, 3, index, y)",
            ],
            input_lines=[
                f"index = torch.arange({pooled - 1}, -1, -1, dtype=torch.long).view(1, 1, 1, {pooled}).expand({batch}, {channels}, {pooled}, {pooled})",
                f"return [torch.randn({batch}, {channels}, {spatial}, {spatial}), index]",
            ],
        )
        ops = [
            _op("avg_pool", ["F.avg_pool2d"], [("aten::avg_pool2d", "")]),
            _op("scatter", ["torch.scatter"], [("aten::scatter", "src")]),
        ]

    elif template_id == "SB01":
        half = channels // 2
        code = _module_code(
            init_lines=[
                f"self.left = nn.Conv2d({half}, {half}, 3, padding=1)",
                f"self.right = nn.Conv2d({half}, {half}, 1)",
            ],
            forward_signature="self, x",
            forward_lines=[
                "left, right = torch.chunk(x, 2, dim=1)",
                "left = self.left(left)",
                "right = F.gelu(self.right(right))",
                "return torch.cat([left, right], dim=1)",
            ],
            input_lines=[f"return [torch.randn({batch}, {channels}, {spatial}, {spatial})]"],
        )
        ops = [
            _op("split", ["torch.chunk"], [("aten::split", "Tensor"), ("aten::split_with_sizes", "")]),
            _op("branch_convs", ["self.left", "self.right"], convolution, min_calls=2),
            _op("gelu_branch", ["F.gelu"], [("aten::gelu", "")]),
            _op("merge", ["torch.cat"], [("aten::cat", "")]),
        ]
    elif template_id == "SB02":
        code = _module_code(
            init_lines=[],
            forward_signature="self, x",
            forward_lines=[
                "left = F.avg_pool2d(x, 2)",
                "right = F.max_pool2d(x, 2)",
                "return torch.cat([left, right], dim=1)",
            ],
            input_lines=[f"return [torch.randn({batch}, {channels}, {spatial}, {spatial})]"],
        )
        ops = [
            _op("avg_branch", ["F.avg_pool2d"], [("aten::avg_pool2d", "")]),
            _op("max_branch", ["F.max_pool2d"], max_pool),
            _op("merge", ["torch.cat"], [("aten::cat", "")]),
        ]
    elif template_id == "SB03":
        code = _module_code(
            init_lines=[],
            forward_signature="self, x",
            forward_lines=[
                "patches = x.unfold(2, 2, 2)",
                "patches = patches.unfold(3, 2, 2)",
                "left = torch.mean(patches, dim=(-1, -2))",
                "right = F.max_pool2d(x, 2)",
                "return torch.add(left, right)",
            ],
            input_lines=[f"return [torch.randn({batch}, {channels}, {spatial}, {spatial})]"],
        )
        ops = [
            _op("unfold", ["x.unfold", "patches.unfold"], [("aten::unfold", "")], min_calls=2),
            _op("patch_mean", ["torch.mean"], [("aten::mean", "dim")]),
            _op("pool_branch", ["F.max_pool2d"], max_pool),
            _op("merge", ["torch.add"], [("aten::add", "Tensor")]),
        ]
    elif template_id == "SB04":
        code = _module_code(
            init_lines=[],
            forward_signature="self, x",
            forward_lines=[
                "pooled = F.avg_pool2d(x, 2)",
                f"left = F.interpolate(pooled, size=({spatial}, {spatial}), mode='bilinear', align_corners=False)",
                "right = F.pad(x[:, :, 1:-1, 1:-1], (1, 1, 1, 1))",
                "return torch.add(left, right)",
            ],
            input_lines=[f"return [torch.randn({batch}, {channels}, {spatial}, {spatial})]"],
        )
        ops = [
            _op("pool_branch", ["F.avg_pool2d"], [("aten::avg_pool2d", "")]),
            _op("upsample_branch", ["F.interpolate"], [("aten::upsample_bilinear2d", "")]),
            _op("pad_branch", ["F.pad"], [("aten::constant_pad_nd", "")]),
            _op("merge", ["torch.add"], [("aten::add", "Tensor")]),
        ]
    elif template_id == "SB05":
        code = _module_code(
            init_lines=[],
            forward_signature="self, x",
            forward_lines=[
                "left = x.flatten(2)",
                "left = left.transpose(1, 2)",
                "right = x.permute(0, 2, 3, 1)",
                f"right = right.reshape({batch}, {spatial * spatial}, {channels})",
                "right = torch.tanh(right)",
                "return torch.add(left, right)",
            ],
            input_lines=[f"return [torch.randn({batch}, {channels}, {spatial}, {spatial})]"],
        )
        ops = [
            _op("transpose_branch", ["left.transpose"], [("aten::transpose", "int")]),
            _op("tanh_branch", ["torch.tanh"], [("aten::tanh", "")]),
            _op("merge", ["torch.add"], [("aten::add", "Tensor")]),
        ]

    elif template_id == "AR01":
        heads, length = 2, 4 + variant // 6
        head_dim = 4 * (1 + (variant % 6) // 2)
        code = _module_code(
            init_lines=[],
            forward_signature="self, q, k, v",
            forward_lines=["return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)"],
            input_lines=[
                f"return [torch.randn({batch}, {heads}, {length}, {head_dim}), torch.randn({batch}, {heads}, {length}, {head_dim}), torch.randn({batch}, {heads}, {length}, {head_dim})]"
            ],
        )
        ops = [_op("scaled_dot_product_attention", ["F.scaled_dot_product_attention"], attention)]
    elif template_id == "AR02":
        length, dim = 4 + variant // 6, channels + 4
        code = _module_code(
            init_lines=[],
            forward_signature="self, q, k, v",
            forward_lines=[
                f"scores = torch.bmm(q, k.transpose(1, 2)) * {dim ** -0.5:.8f}",
                "probs = torch.softmax(scores, dim=-1)",
                "return torch.bmm(probs, v)",
            ],
            input_lines=[
                f"return [torch.randn({batch}, {length}, {dim}), torch.randn({batch}, {length}, {dim}), torch.randn({batch}, {length}, {dim})]"
            ],
        )
        ops = [
            _op("attention_bmm", ["torch.bmm"], [("aten::bmm", "")], min_calls=2),
            _op("attention_softmax", ["torch.softmax"], [("aten::_softmax", "")]),
        ]
    elif template_id == "AR03":
        embed, length = channels * 2, 4 + variant // 6
        code = _module_code(
            init_lines=[f"self.attn = nn.MultiheadAttention({embed}, 2, dropout=0.0, batch_first=True)"],
            forward_signature="self, x",
            forward_lines=["out, _ = self.attn(x, x, x, need_weights=False)", "return out"],
            input_lines=[f"return [torch.randn({batch}, {length}, {embed})]"],
        )
        ops = [_op("multihead_attention", ["self.attn"], attention + [("aten::_native_multi_head_attention", "")])]
    elif template_id == "AR04":
        features, hidden, length = channels + 2, channels + 6, 4 + variant // 6
        code = _module_code(
            init_lines=[f"self.gru = nn.GRU({features}, {hidden}, batch_first=True)"],
            forward_signature="self, x",
            forward_lines=["out, _ = self.gru(x)", "return out"],
            input_lines=[f"return [torch.randn({batch}, {length}, {features})]"],
        )
        ops = [_op("gru", ["self.gru"], [("aten::gru", "input"), ("aten::_cudnn_rnn", "")])]
    elif template_id == "AR05":
        features, hidden, length = channels + 2, channels + 6, 4 + variant // 6
        code = _module_code(
            init_lines=[f"self.lstm = nn.LSTM({features}, {hidden}, batch_first=True)"],
            forward_signature="self, x",
            forward_lines=["out, _ = self.lstm(x)", "return out"],
            input_lines=[f"return [torch.randn({batch}, {length}, {features})]"],
        )
        ops = [_op("lstm", ["self.lstm"], [("aten::lstm", "input"), ("aten::_cudnn_rnn", "")])]
    elif template_id == "AR06":
        features, hidden, length = channels + 4, channels + 8, 4 + variant // 6
        code = _module_code(
            init_lines=[
                f"self.bn = nn.BatchNorm1d({features})",
                f"self.gru = nn.GRU({features}, {hidden}, batch_first=True)",
            ],
            forward_signature="self, x",
            forward_lines=["y = self.bn(x.transpose(1, 2)).transpose(1, 2)", "out, _ = self.gru(y)", "return out"],
            input_lines=[f"return [torch.randn({batch}, {length}, {features})]"],
        )
        ops = [
            _op("batch_norm", ["self.bn"], batch_norm),
            _op("gru", ["self.gru"], [("aten::gru", "input"), ("aten::_cudnn_rnn", "")]),
        ]

    elif template_id == "HD01":
        hidden = channels + 8
        code = _module_code(
            init_lines=[
                f"self.conv = nn.Conv2d({channels}, {channels}, 3, padding=1)",
                f"self.bn = nn.BatchNorm2d({channels})",
                f"self.linear = nn.Linear({channels * 4}, {hidden})",
            ],
            forward_signature="self, x",
            forward_lines=[
                "y = F.relu(self.bn(self.conv(x)))",
                "y = F.max_pool2d(y, 2)",
                "y = F.adaptive_avg_pool2d(y, (2, 2)).flatten(1)",
                "y = self.linear(y)",
                "return F.log_softmax(y, dim=-1)",
            ],
            input_lines=[f"return [torch.randn({batch}, {channels}, {spatial}, {spatial})]"],
        )
        ops = [
            _op("conv", ["self.conv"], convolution),
            _op("batch_norm", ["self.bn"], batch_norm),
            _op("max_pool", ["F.max_pool2d"], max_pool),
            _op("adaptive_pool", ["F.adaptive_avg_pool2d"], adaptive_pool),
            _op("linear", ["self.linear"], [("aten::addmm", ""), ("aten::linear", "")]),
            _op("log_softmax", ["F.log_softmax"], [("aten::_log_softmax", "")]),
        ]
    elif template_id == "HD02":
        vocab, embed, length = 37 + variant, channels + 4, spatial
        code = _module_code(
            init_lines=[
                f"self.embedding = nn.Embedding({vocab}, {embed})",
                f"self.norm = nn.LayerNorm({embed})",
                f"self.q = nn.Linear({embed}, {embed})",
                f"self.k = nn.Linear({embed}, {embed})",
                f"self.v = nn.Linear({embed}, {embed})",
            ],
            forward_signature="self, token_ids",
            forward_lines=[
                "x = self.norm(self.embedding(token_ids))",
                "q = self.q(x)",
                "k = self.k(x)",
                "v = self.v(x)",
                f"scores = torch.bmm(q, k.transpose(1, 2)) * {embed ** -0.5:.8f}",
                "probs = torch.softmax(scores, dim=-1)",
                "return torch.bmm(probs, v)",
            ],
            input_lines=[f"return [torch.randint(0, {vocab}, ({batch}, {length}), dtype=torch.long)]"],
        )
        ops = [
            _op("embedding", ["self.embedding"], [("aten::embedding", "")]),
            _op("layer_norm", ["self.norm"], layer_norm),
            _op("linears", ["self.q", "self.k", "self.v"], [("aten::addmm", ""), ("aten::linear", "")], min_calls=3),
            _op("attention_bmm", ["torch.bmm"], [("aten::bmm", "")], min_calls=2),
            _op("softmax", ["torch.softmax"], [("aten::_softmax", "")]),
        ]
    elif template_id == "HD03":
        code = _module_code(
            init_lines=[f"self.conv = nn.Conv1d({channels}, {channels}, 3, padding=1)"],
            forward_signature="self, x",
            forward_lines=[
                "y = F.gelu(self.conv(x))",
                "left = F.avg_pool1d(y, 2)",
                "right = F.max_pool1d(y, 2)",
                "return torch.cat([left, right], dim=1)",
            ],
            input_lines=[f"return [torch.randn({batch}, {channels}, {spatial * 2})]"],
        )
        ops = [
            _op("conv", ["self.conv"], convolution),
            _op("gelu", ["F.gelu"], [("aten::gelu", "")]),
            _op("avg_pool", ["F.avg_pool1d"], [("aten::avg_pool1d", ""), ("aten::avg_pool2d", "")]),
            _op("max_pool", ["F.max_pool1d"], max_pool),
            _op("merge", ["torch.cat"], [("aten::cat", "")]),
        ]
    elif template_id == "HD04":
        embed, length, selected = channels + 4, spatial + 2, 3 + variant % 3
        code = _module_code(
            init_lines=[f"self.linear = nn.Linear({embed}, {embed + 4})", f"self.norm = nn.LayerNorm({embed + 4})"],
            forward_signature="self, x, index",
            forward_lines=[
                f"expanded = index.unsqueeze(-1).expand({batch}, {selected}, {embed})",
                "y = torch.gather(x, 1, expanded)",
                "y = self.norm(self.linear(y))",
                "return torch.mean(y, dim=1)",
            ],
            input_lines=[
                f"index = torch.arange({selected - 1}, -1, -1, dtype=torch.long).view(1, {selected}).expand({batch}, {selected})",
                f"return [torch.randn({batch}, {length}, {embed}), index]",
            ],
        )
        ops = [
            _op("gather", ["torch.gather"], [("aten::gather", "")]),
            _op("linear", ["self.linear"], [("aten::addmm", ""), ("aten::linear", "")]),
            _op("layer_norm", ["self.norm"], layer_norm),
            _op("mean", ["torch.mean"], [("aten::mean", "dim")]),
        ]
    elif template_id == "HD05":
        code = _module_code(
            init_lines=[
                f"self.conv1 = nn.Conv2d({channels}, {channels}, 3, padding=1)",
                f"self.norm = nn.GroupNorm(4, {channels})",
                f"self.conv2 = nn.Conv2d({channels * 2}, {channels + 4}, 1)",
            ],
            forward_signature="self, x",
            forward_lines=[
                "y = F.relu(self.norm(self.conv1(x)))",
                "pooled = F.max_pool2d(y, 2)",
                f"up = F.interpolate(pooled, size=({spatial}, {spatial}), mode='nearest')",
                "merged = torch.cat([y, up], dim=1)",
                "return self.conv2(merged)",
            ],
            input_lines=[f"return [torch.randn({batch}, {channels}, {spatial}, {spatial})]"],
        )
        ops = [
            _op("convs", ["self.conv1", "self.conv2"], convolution, min_calls=2),
            _op("group_norm", ["self.norm"], group_norm),
            _op("max_pool", ["F.max_pool2d"], max_pool),
            _op("upsample", ["F.interpolate"], [("aten::upsample_nearest2d", "")]),
            _op("merge", ["torch.cat"], [("aten::cat", "")]),
        ]
    else:
        raise ValueError(f"unknown template:{template_id}")

    labels = {
        "batch": batch,
        "channels": channels,
        "spatial": spatial,
        "template_id": template_id,
        "variant": variant,
        "topology": "branch_merge" if template_id.startswith(("SB", "HD")) else "chain",
        "final_output_kind": "single_tensor",
    }
    if template_id == "AR01":
        labels["attention_head_dim"] = head_dim
    return code, ops, labels


_ALLOCATION_CALLS = frozenset(
    {
        "torch.empty_like",
        "torch.full_like",
        "torch.ones_like",
        "torch.rand_like",
        "torch.randn_like",
        "torch.zeros_like",
    }
)


def _expr_tags(
    expression: ast.AST, environment: Mapping[str, frozenset[str]], call_to_ids: Mapping[str, frozenset[str]]
) -> frozenset[str]:
    if isinstance(expression, ast.Name):
        return environment.get(expression.id, frozenset())
    if isinstance(expression, ast.Call):
        name = _call_name(expression.func) or ""
        inputs = [*expression.args, *(keyword.value for keyword in expression.keywords)]
        if isinstance(expression.func, ast.Attribute):
            inputs.append(expression.func.value)
        argument_tags = frozenset().union(*(_expr_tags(item, environment, call_to_ids) for item in inputs))
        if name in _ALLOCATION_CALLS:
            argument_tags = frozenset()
        return argument_tags | call_to_ids.get(name, frozenset())
    if isinstance(expression, (ast.List, ast.Tuple, ast.Set)):
        return frozenset().union(*(_expr_tags(item, environment, call_to_ids) for item in expression.elts))
    if isinstance(expression, ast.Dict):
        return frozenset().union(
            *(
                _expr_tags(item, environment, call_to_ids)
                for item in [*expression.keys, *expression.values]
                if item is not None
            )
        )
    return frozenset().union(
        *(_expr_tags(child, environment, call_to_ids) for child in ast.iter_child_nodes(expression))
    )


def _bind_target(target: ast.AST, tags: frozenset[str], environment: dict[str, frozenset[str]]) -> None:
    if isinstance(target, ast.Name):
        environment[target.id] = tags
    elif isinstance(target, (ast.Tuple, ast.List)):
        for item in target.elts:
            _bind_target(item, tags, environment)
    else:
        raise ValueError(f"unsupported forward assignment target:{ast.dump(target, include_attributes=False)}")


def _static_contract(code: str, declared_ops: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tree = ast.parse(code)
    model = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Model"), None)
    if model is None:
        raise ValueError("generated reference lacks top-level Model")
    forward = next((node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "forward"), None)
    if forward is None:
        raise ValueError("generated Model lacks forward")
    forbidden = (
        ast.If,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.Try,
        ast.With,
        ast.AsyncWith,
        ast.Lambda,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )
    if any(isinstance(node, forbidden) for node in ast.walk(forward)):
        raise ValueError("generated forward is not straight-line")
    if any(isinstance(node, (ast.AugAssign, ast.Delete, ast.NamedExpr)) for node in ast.walk(forward)):
        raise ValueError("generated forward mutates or deletes an expression")
    calls = collections.Counter(_call_name(node.func) for node in ast.walk(forward) if isinstance(node, ast.Call))
    if any(name and (name.endswith("_") or ".__" in name) for name in calls):
        raise ValueError(f"generated forward contains in-place/dunder call:{sorted(calls)}")
    call_to_ids: dict[str, set[str]] = collections.defaultdict(set)
    for item in declared_ops:
        for name in item["source_calls"]:
            call_to_ids[name].add(str(item["op_id"]))
        if not any(calls.get(name, 0) for name in item["source_calls"]):
            raise ValueError(f"declared source op is absent:{item['op_id']}:{item['source_calls']}")
    environment: dict[str, frozenset[str]] = {}
    returned: list[frozenset[str]] = []
    for statement in forward.body:
        if isinstance(statement, ast.Assign):
            tags = _expr_tags(
                statement.value, environment, {key: frozenset(value) for key, value in call_to_ids.items()}
            )
            for target in statement.targets:
                _bind_target(target, tags, environment)
        elif isinstance(statement, ast.AnnAssign):
            tags = (
                _expr_tags(statement.value, environment, {key: frozenset(value) for key, value in call_to_ids.items()})
                if statement.value
                else frozenset()
            )
            _bind_target(statement.target, tags, environment)
        elif isinstance(statement, ast.Expr):
            _expr_tags(statement.value, environment, {key: frozenset(value) for key, value in call_to_ids.items()})
        elif isinstance(statement, ast.Return):
            if statement.value is None or isinstance(statement.value, (ast.Tuple, ast.List, ast.Dict, ast.Set)):
                raise ValueError("final forward output is not syntactically single-valued")
            returned.append(
                _expr_tags(statement.value, environment, {key: frozenset(value) for key, value in call_to_ids.items()})
            )
        else:
            raise ValueError(f"unsupported straight-line statement:{type(statement).__name__}")
    if len(returned) != 1:
        raise ValueError(f"generated forward must contain one return:{len(returned)}")
    expected = {str(item["op_id"]) for item in declared_ops}
    missing = expected - set(returned[0])
    if missing:
        raise ValueError(f"declared ops do not reach syntactic return:{sorted(missing)}")
    return {
        "contract_version": STATIC_CONTRACT_VERSION,
        "forward_statement_count": len(forward.body),
        "source_call_histogram": {str(key): value for key, value in sorted(calls.items()) if key},
        "returned_declared_op_ids": sorted(returned[0]),
        "single_value_return": True,
        "straight_line": True,
    }


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
    for raw_path in paths:
        path = raw_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        parquet = pq.ParquetFile(path)
        if "reward_model" not in parquet.schema_arrow.names:
            raise ValueError(f"comparison artifact lacks reward_model:{path}")
        syntax_error_rows = 0
        for batch in parquet.iter_batches(batch_size=256, columns=["reward_model"]):
            for row in batch.to_pylist():
                code = (
                    row["reward_model"].get("ground_truth") if isinstance(row.get("reward_model"), Mapping) else None
                )
                if not isinstance(code, str) or not code.strip():
                    raise ValueError(f"comparison artifact has empty reference:{path}")
                references.add(_sha256_bytes(code.encode("utf-8")))
                try:
                    ast_hashes.add(_normalized_ast_sha256(code))
                except SyntaxError:
                    # Invalid comparison references cannot have the same valid
                    # normalized AST as a generated row; their exact text hash
                    # remains in the all-row comparison set.
                    syntax_error_rows += 1
        bindings.append(
            {
                "path": str(path),
                "sha256": _sha256_file(path),
                "rows": parquet.metadata.num_rows,
                "normalized_ast_parse_failures": syntax_error_rows,
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
    prompt_prefix_sha256 = _sha256_bytes(prompt_prefix.encode("utf-8"))
    for spec in TEMPLATES:
        for variant in range(spec.variants):
            code, declared_ops, labels = _render(spec.template_id, variant)
            static = _static_contract(code, declared_ops)
            reference_sha256 = _sha256_bytes(code.encode("utf-8"))
            ast_sha256 = _normalized_ast_sha256(code)
            if reference_sha256 in comparison_reference_hashes:
                raise ValueError(f"generated exact-reference collision:{spec.template_id}:{variant}")
            if ast_sha256 in comparison_ast_hashes:
                raise ValueError(f"generated normalized-AST collision:{spec.template_id}:{variant}")
            stable_key = f"{CONTRACT_VERSION}|{spec.template_id}|{variant}"
            uuid = "semop_" + _sha256_bytes(stable_key.encode("utf-8"))[:24]
            content = prompt_prefix + code
            prompt = [{"content": content, "role": "user"}]
            source_calls = sorted({name for item in declared_ops for name in item["source_calls"]})
            row = {
                "data_source": DATA_SOURCE,
                "prompt": prompt,
                "ability": "kernel_optimization",
                "reward_model": {"ground_truth": code, "style": "rule"},
                "extra_info": {
                    "entry_point": "Model",
                    "level": "coverage",
                    "module_name": "Model",
                    "ops": json.dumps(source_calls, ensure_ascii=False),
                    "original_prompt": prompt,
                    "repo_name": "semantic_operator_generator_v1",
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
                "uuid": uuid,
                "template_id": spec.template_id,
                "template_variant": variant,
                "primary_family": spec.family,
                "mode_behavior": spec.mode_behavior,
                "lineage_kind": "standalone_semantic_synthetic",
                "parent_uuid": None,
                "primary_intervention": "semantic_operator",
                "final_output_contract": {"kind": "single_tensor", "finite_required": True},
                "training_mode": True,
                "declared_ops": declared_ops,
                "static_proof": static,
                "coverage_labels": labels,
                "reference_sha256": reference_sha256,
                "normalized_ast_sha256": ast_sha256,
                "prompt_sha256": _sha256_bytes(content.encode("utf-8")),
                "row_payload_sha256": _canonical_sha256(row),
                "prompt_template": dict(template_binding),
                "prompt_prefix_sha256": prompt_prefix_sha256,
                "decontamination_roots": [dict(item) for item in comparison_bindings],
                "decontamination_status": "exact_reference_all_and_normalized_ast_parseable_no_match",
                "provenance": {
                    "source_family": "project_generated_internal_review",
                    "generator": str(Path(__file__).resolve()),
                    "license": "internal-review-only",
                    "provenance_status": "project_generated_bound",
                },
                "git_commit_at_generation": git_commit,
                "generator_source_sha256": generator_sha256,
                "static_status": "passed",
                "reference_runtime_status": "pending",
                "operator_liveness_status": "pending",
                "materialization_status": "review_only",
                "training_approved": False,
                "structured_output_deferred": True,
                "structured_output_defer_reason": "KernelGym authority is Tensor-only at final comparison",
            }
            order_key = _sha256_bytes(f"{ORDER_VERSION}|{spec.template_id}|{variant}".encode())
            pending.append((order_key, row, manifest))
    pending.sort(key=lambda item: item[0])
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for index, (_, row, manifest) in enumerate(pending):
        manifest = dict(manifest)
        manifest["candidate_row_index"] = index
        rows.append(row)
        manifests.append(manifest)
    return rows, manifests


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")


def _review_markdown(rows: Sequence[Mapping[str, Any]], manifests: Sequence[Mapping[str, Any]]) -> str:
    examples: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for family in FAMILY_QUOTAS:
        selected = [
            (row, manifest)
            for row, manifest in zip(rows, manifests, strict=True)
            if manifest["primary_family"] == family
        ]
        examples.extend([selected[0], selected[-1]])
    lines = [
        "# Semantic/operator static canary samples",
        "",
        "All rows are parentless, deterministic, review-only, and return one Tensor. Structured final outputs are deferred.",
        "",
        "| family | quota |",
        "| --- | ---: |",
        *(f"| {family} | {quota} |" for family, quota in sorted(FAMILY_QUOTAS.items())),
        "",
    ]
    for row, manifest in examples:
        lines.extend(
            [
                f"## {manifest['template_id']} variant {manifest['template_variant']} ({manifest['primary_family']})",
                "",
                f"UUID: `{manifest['uuid']}`",
                "",
                "Declared runtime ops: " + ", ".join(item["op_id"] for item in manifest["declared_ops"]),
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
    template_path = template_path.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _, prompt_prefix = _template(template_path)
    template_parquet = pq.ParquetFile(template_path)
    template_binding = {
        "path": str(template_path),
        "sha256": _sha256_file(template_path),
        "rows": template_parquet.metadata.num_rows,
    }
    comparison_reference_hashes, comparison_ast_hashes, comparison_bindings = _comparison_hashes(comparison_paths)
    generator_sha256 = _sha256_file(Path(__file__).resolve())
    rows, manifests = build_records(
        prompt_prefix=prompt_prefix,
        comparison_reference_hashes=comparison_reference_hashes,
        comparison_ast_hashes=comparison_ast_hashes,
        template_binding=template_binding,
        comparison_bindings=comparison_bindings,
        git_commit=_git_commit(),
        generator_sha256=generator_sha256,
    )
    if len(rows) != EXACT_CANARY_ROWS or len(rows) > MAX_AUTHORIZED_CANDIDATES:
        raise ValueError(f"exact canary size mismatch:{len(rows)}")
    family_counts = collections.Counter(item["primary_family"] for item in manifests)
    template_counts = collections.Counter(item["template_id"] for item in manifests)
    if dict(family_counts) != FAMILY_QUOTAS:
        raise ValueError(f"family quotas differ:{dict(family_counts)}")
    expected_template_counts = {item.template_id: item.variants for item in TEMPLATES}
    if dict(template_counts) != expected_template_counts:
        raise ValueError(f"template quotas differ:{dict(template_counts)}")
    for field in ("uuid", "reference_sha256", "normalized_ast_sha256", "row_payload_sha256"):
        values = [item[field] for item in manifests]
        if len(values) != len(set(values)):
            raise ValueError(f"generated {field} is not unique")

    candidates_path = output_dir / "candidates.parquet"
    manifest_path = output_dir / "manifest.jsonl"
    decisions_path = output_dir / "decisions.jsonl"
    review_path = output_dir / "review_samples.md"
    summary_path = output_dir / "summary.json"
    pq.write_table(pa.Table.from_pylist(rows, schema=_semantic_schema()), candidates_path, compression="zstd")
    _write_jsonl(manifest_path, manifests)
    _write_jsonl(
        decisions_path,
        [
            {
                "decision": "single_tensor_only",
                "selected": "B",
                "structured_output_status": "explicitly_deferred",
                "reason": "authority correctness path dereferences output.shape and compares Tensor only",
            },
            {
                "decision": "generation_method",
                "selected": "deterministic_closed_registry_solver",
                "llm_generation_used": False,
                "reason": "the coverage grid and executable references are fully enumerable and statically checkable",
            },
        ],
    )
    review_path.write_text(_review_markdown(rows, manifests), encoding="utf-8")
    summary = {
        "contract_version": CONTRACT_VERSION,
        "registry_version": REGISTRY_VERSION,
        "rows": len(rows),
        "family_counts_exclusive_primary": dict(sorted(family_counts.items())),
        "template_counts": dict(sorted(template_counts.items())),
        "mode_behavior_counts": dict(sorted(collections.Counter(item["mode_behavior"] for item in manifests).items())),
        "final_output_kind": "single_tensor",
        "structured_output_status": "explicitly_deferred",
        "training_approved": False,
        "artifacts": {
            "candidates": {"path": str(candidates_path), "sha256": _sha256_file(candidates_path)},
            "manifest": {"path": str(manifest_path), "sha256": _sha256_file(manifest_path)},
            "decisions": {"path": str(decisions_path), "sha256": _sha256_file(decisions_path)},
            "review_samples": {"path": str(review_path), "sha256": _sha256_file(review_path)},
        },
        "prompt_template": template_binding,
        "decontamination_roots": comparison_bindings,
        "generator_source_sha256": generator_sha256,
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
    comparisons = tuple(args.comparison_artifacts) if args.comparison_artifacts else _DEFAULT_COMPARISON_ROOTS
    summary = generate(args.output_dir, args.template_parquet, comparisons)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
