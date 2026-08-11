#!/usr/bin/env python3
"""Generate the exact-1k review-only frontier operator scenario canary.

The generator is deliberately constructive: each closed-registry cell maps a
template/variant pair to legal shapes, indices and values.  It does not use an
LLM, retry sampling, or the untracked ``lhb`` prior-art directory.
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

CONTRACT_VERSION = "frontier_operator_constructive_generator_v3"
MANIFEST_VERSION = "frontier_operator_parentless_manifest_v1"
REGISTRY_VERSION = "frontier_operator_closed_registry_35_cells_v3"
ORDER_VERSION = "sha256_frontier_contract_template_variant_v1"
STATIC_CONTRACT_VERSION = "straight_line_return_dependency_v1"
CONSTRAINT_CONTRACT_VERSION = "frontier_constraint_cells_v1"
RUNTIME_CONTRACT_VERSION = "aten_dispatch_return_provenance_v1"
DATA_SOURCE = "project_generated_frontier_operator_v3"
MAX_AUTHORIZED_CANDIDATES = 1_000
EXACT_CANARY_ROWS = 1_000
_DEFAULT_TEMPLATE = _REPO_ROOT / "Data/prompt_tvm_v4/train.review.parquet"
_DEFAULT_COMPARISON_ROOTS = (
    _DEFAULT_TEMPLATE,
    _REPO_ROOT / "Data/prompt_tvm_v4/synthesis/accepted/extreme_ops_v3.parquet",
    _REPO_ROOT / "local_artifacts/data_handoffs/prompt_tvm_v4_semantic_operator_canary1000/candidates.parquet",
)

SOURCES = {
    "pytorch_sparse": (
        "https://docs.pytorch.org/docs/stable/sparse",
        "official_documentation",
        "Construct COO/CSR inside forward and densify only at the final boundary.",
    ),
    "mixtral": (
        "https://arxiv.org/abs/2401.04088",
        "original_paper",
        "Token-choice top-2 routed expert execution expressed with dense gather/scatter primitives.",
    ),
    "expert_choice": (
        "https://arxiv.org/abs/2202.09368",
        "original_paper",
        "Expert-choice routing selects tokens independently for each expert; the portable template exposes the choice count as a coordinate.",
    ),
    "deepseek": (
        "https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py",
        "official_repository",
        "MLA, group-limited routing, and a shared-expert path are modeled with ordinary dense PyTorch primitives; the bounded assignment mask is an explicit synthetic stress axis, not a claim about the source implementation.",
    ),
    "mamba": (
        "https://github.com/state-spaces/mamba",
        "official_repository",
        "Selective state-space scans are finite render-time-unrolled recurrences with causal convolution.",
    ),
    "flash_attention": (
        "https://github.com/Dao-AILab/flash-attention",
        "official_repository",
        "GQA, causal/sliding attention, append cache, and paged gather are expressed with framework operators.",
    ),
    "vllm": (
        "https://github.com/vllm-project/vllm",
        "official_repository",
        "Block-table KV gather models page addressing without a serving scheduler.",
    ),
    "pyg": (
        "https://pytorch-geometric.readthedocs.io/en/stable/generated/torch_geometric.nn.conv.MessagePassing.html",
        "official_documentation",
        "CSR-like message aggregation and segment attention use legal offsets and scatter/index-add.",
    ),
    "transformer_engine": (
        "https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html",
        "official_documentation",
        "QDQ recipes model integer ranges and scales with portable fake quantization.",
    ),
    "qwen_vl": (
        "https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py",
        "official_repository",
        "Variable grid/window layout and image sampling are represented with standard PyTorch geometry operators.",
    ),
    "fno": (
        "https://arxiv.org/abs/2010.08895",
        "original_paper",
        "Fourier neural operators motivate RFFT-domain learned spectral filtering.",
    ),
    "pytorch_stft": (
        "https://docs.pytorch.org/docs/stable/generated/torch.stft.html",
        "official_documentation",
        "A portable window/unfold/RFFT construction models the documented STFT framing semantics without the internal in-place transpose path.",
    ),
    "pytorch_linalg": (
        "https://docs.pytorch.org/docs/stable/linalg.html",
        "official_documentation",
        "Construct a positive-definite Gram matrix and exercise Cholesky factorization followed by solve.",
    ),
    "llama": (
        "https://github.com/meta-llama/llama",
        "official_repository",
        "RMSNorm and SwiGLU-style gated residual blocks model a modern dense LLM layer.",
    ),
    "t5": (
        "https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py",
        "official_repository",
        "Encoder-decoder cross-attention and a configurable gated-GELU feed-forward path motivate the portable cross-attention block.",
    ),
    "pytorch_nested": (
        "https://docs.pytorch.org/docs/stable/nested.html",
        "official_documentation",
        "Packed ragged sequences motivate explicit offsets, empty segments, and segment reductions without padding.",
    ),
    "dlrm": (
        "https://arxiv.org/abs/1906.00091",
        "original_paper",
        "Ragged embedding-bag aggregation and dense feature interaction model recommendation workloads.",
    ),
    "faiss": (
        "https://github.com/facebookresearch/faiss",
        "official_repository",
        "Similarity search motivates top-k corpus selection followed by gathered weighted reduction.",
    ),
    "pytorch_embedding_bag": (
        "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.embedding_bag.html",
        "official_documentation",
        "Offsets, duplicate IDs, and empty bags follow the functional embedding-bag contract.",
    ),
    "pytorch_quantization": (
        "https://docs.pytorch.org/docs/stable/quantization.html",
        "official_documentation",
        "Portable fake-QDQ paths exercise per-channel, groupwise packed, and dynamic integer range policies.",
    ),
    "pytorch_grid_sample": (
        "https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html",
        "official_documentation",
        "Normalized coordinates, padding modes, and align-corners policies exercise grid sampling.",
    ),
    "pytorch_pixel_shuffle": (
        "https://docs.pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html",
        "official_documentation",
        "Pixel shuffle and unshuffle model reversible spatial/channel rearrangement.",
    ),
}

FAMILY_QUOTAS = {
    "sparse_storage_compute": 160,
    "moe_routing": 140,
    "selective_state_space": 120,
    "attention_cache": 120,
    "modern_llm_block": 100,
    "ragged_graph_segment": 120,
    "quantization_qdq": 80,
    "vision_geometry": 60,
    "spectral_scientific": 60,
    "retrieval_recommender": 40,
}


@dataclasses.dataclass(frozen=True)
class TemplateSpec:
    template_id: str
    family: str
    variants: int
    source_key: str
    scenario_name: str
    mode_behavior: str = "stateless"


TEMPLATES = (
    TemplateSpec("SP01", "sparse_storage_compute", 40, "pytorch_sparse", "COO SpMM"),
    TemplateSpec("SP02", "sparse_storage_compute", 40, "pytorch_sparse", "CSR SpMM/addmm"),
    TemplateSpec("SP03", "sparse_storage_compute", 40, "pytorch_sparse", "CSR sampled_addmm"),
    TemplateSpec("SP04", "sparse_storage_compute", 40, "pytorch_sparse", "sparse softmax / 2:4 pattern"),
    TemplateSpec("MO01", "moe_routing", 35, "mixtral", "token-choice top-k"),
    TemplateSpec("MO02", "moe_routing", 35, "deepseek", "group-limited top-k"),
    TemplateSpec("MO03", "moe_routing", 35, "expert_choice", "expert-choice routing"),
    TemplateSpec("MO04", "moe_routing", 35, "deepseek", "shared-expert routing with bounded assignment mask"),
    TemplateSpec("SS01", "selective_state_space", 40, "mamba", "selective recurrence"),
    TemplateSpec("SS02", "selective_state_space", 40, "mamba", "causal-conv gated scan"),
    TemplateSpec("SS03", "selective_state_space", 40, "mamba", "chunk/SSD-style scan"),
    TemplateSpec("AC01", "attention_cache", 30, "flash_attention", "RoPE GQA/MQA"),
    TemplateSpec("AC02", "attention_cache", 30, "flash_attention", "unequal-length sliding/ALiBi"),
    TemplateSpec("AC03", "attention_cache", 30, "flash_attention", "append KV cache"),
    TemplateSpec("AC04", "attention_cache", 30, "vllm", "paged block-table gather"),
    TemplateSpec("LL01", "modern_llm_block", 25, "deepseek", "MLA prefill"),
    TemplateSpec("LL02", "modern_llm_block", 25, "deepseek", "compressed MLA decode"),
    TemplateSpec("LL03", "modern_llm_block", 25, "llama", "RMSNorm+SwiGLU"),
    TemplateSpec("LL04", "modern_llm_block", 25, "t5", "cross-attention/gated GELU"),
    TemplateSpec("RG01", "ragged_graph_segment", 30, "pytorch_nested", "offset-packed ragged reduction"),
    TemplateSpec("RG02", "ragged_graph_segment", 30, "pyg", "indexed message passing"),
    TemplateSpec("RG03", "ragged_graph_segment", 30, "pyg", "segment-softmax GAT"),
    TemplateSpec("RG04", "ragged_graph_segment", 30, "pytorch_embedding_bag", "embedding-bag offsets"),
    TemplateSpec("QD01", "quantization_qdq", 20, "pytorch_quantization", "per-channel INT8"),
    TemplateSpec("QD02", "quantization_qdq", 20, "pytorch_quantization", "groupwise packed INT4"),
    TemplateSpec("QD03", "quantization_qdq", 20, "transformer_engine", "block fake-FP8-like QDQ"),
    TemplateSpec("QD04", "quantization_qdq", 20, "pytorch_quantization", "dynamic activation INT8"),
    TemplateSpec("VG01", "vision_geometry", 20, "qwen_vl", "patch/window reorder"),
    TemplateSpec("VG02", "vision_geometry", 20, "pytorch_grid_sample", "grid warp"),
    TemplateSpec("VG03", "vision_geometry", 20, "pytorch_pixel_shuffle", "pixel unshuffle/shuffle"),
    TemplateSpec("SF01", "spectral_scientific", 20, "fno", "RFFT spectral convolution"),
    TemplateSpec("SF02", "spectral_scientific", 20, "pytorch_stft", "windowed RFFT (STFT-like) projection"),
    TemplateSpec("SF03", "spectral_scientific", 20, "pytorch_linalg", "SPD factor/solve"),
    TemplateSpec("RR01", "retrieval_recommender", 20, "dlrm", "embedding-bag interaction"),
    TemplateSpec("RR02", "retrieval_recommender", 20, "faiss", "similarity top-k gather"),
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
    if not schema.startswith("aten::"):
        raise ValueError(f"invalid ATen identity:{schema!r}")
    return {"schema": schema, "overload": overload}


def _op(op_id: str, source_calls: Sequence[str], identities: Sequence[tuple[str, str]]) -> dict[str, Any]:
    return {
        "op_id": op_id,
        "source_calls": list(source_calls),
        "runtime_identities": [_identity(*item) for item in identities],
        "min_calls_per_trial": 1,
        "must_reach_returned_output": True,
    }


def _code(forward_signature: str, lines: Sequence[str], inputs: Sequence[str]) -> str:
    return (
        "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n\n    def forward("
        + forward_signature
        + "):\n"
        + "\n".join("        " + line for line in lines)
        + "\n\ndef get_inputs():\n"
        + "\n".join("    " + line for line in inputs)
        + "\n\ndef get_init_inputs():\n    return []\n"
    )


def _coords(template_id: str, variant: int) -> dict[str, Any]:
    spec = TEMPLATE_BY_ID[template_id]
    if not 0 <= variant < spec.variants:
        raise ValueError(f"variant out of range:{template_id}:{variant}")
    # Each schedule has an explicit coordinate that is consumed by its own
    # template.  Do not share a family-wide policy name between structurally
    # different templates: requested and realized labels must describe the
    # rendered computation, rather than a nearby family member.
    common = {"variant": variant}
    if template_id.startswith("SP"):
        if template_id == "SP04" and variant % 2:
            return {
                **common,
                "sparse_mode": "semi_structured_2to4",
                "rows": 64,
                "cols": 64,
                "rhs_cols": (8, 12, 16, 20, 24)[variant // 8],
                "semi_pair": (variant // 2) % 4,
            }
        rows = (8, 12, 16, 20, 24)[variant % 5]
        cols = (8, 12, 16, 20)[(variant // 5) % 4]
        base = {
            **common,
            "rows": rows,
            "cols": cols,
            "degree": (2, 3, 4, 5)[(variant // 10) % 4],
            "index_pattern": ("regular", "long_tail", "band", "strided")[variant % 4],
            "index_offset": variant % cols,
        }
        return {**base, "sparse_mode": "coo_softmax"} if template_id == "SP04" else base
    if template_id.startswith("MO"):
        return {
            **common,
            "tokens": (8, 12, 16, 20, 24)[variant % 5],
            "experts": (4, 6, 8, 10)[(variant // 5) % 4],
            "topk": 1 + (variant % 2),
            "router_bias_policy": ("uniform", "head_heavy", "alternating", "tail_heavy")[variant % 4],
            "router_temperature": (0.75, 1.25)[variant // 20],
        }
    if template_id.startswith("SS"):
        base = {
            **common,
            "length": (8, 12, 16, 20, 24)[variant % 5],
            "width": (4, 6, 8, 10)[(variant // 5) % 4],
            "input_gain": (0.8, 1.2)[variant // 20],
        }
        if template_id == "SS01":
            return {
                **base,
                "decay_policy": (
                    "decay_scale_035",
                    "decay_scale_140",
                    "alternating_base_080_step_085",
                    "decay_scale_110",
                )[variant % 4],
            }
        if template_id == "SS02":
            return {
                **base,
                "gate_bias": (-0.5, -0.15, 0.15, 0.5)[variant % 4],
                "carry_decay": (0.65, 0.75, 0.85, 0.93)[(variant // 5) % 4],
            }
        return {**base, "chunk_split": ("early", "middle", "late", "middle_half_carry")[variant % 4]}
    if template_id.startswith("AC"):
        base = {
            **common,
            "q_len": (2, 3, 4, 5, 6)[variant % 5],
            "kv_len": (6, 8, 10, 12)[(variant // 5) % 4],
            "heads": (2, 4)[variant % 2],
            "kv_heads": 1 if variant % 3 else 2,
            "query_scale": (0.8, 1.2)[variant // 20],
        }
        if template_id == "AC01":
            return {**base, "rope_base": (4.0, 8.0, 16.0, 32.0)[variant % 4]}
        if template_id == "AC02":
            return {
                **base,
                "sliding_window": (2, 3, 4, 5)[variant % 4],
                "alibi_slope": (0.0, 0.05, 0.1, 0.2)[(variant // 5) % 4],
            }
        if template_id == "AC03":
            return {
                **base,
                "append_policy": ("cache_then_new", "new_then_cache", "scaled_cache_075", "scaled_cache_090")[
                    variant % 4
                ],
            }
        return {**base, "page_order": list(((0, 1), (1, 0), (0, 0), (1, 1))[variant % 4])}
    if template_id.startswith("LL"):
        base = {
            **common,
            "tokens": (4, 6, 8, 10, 12)[variant % 5],
            "hidden": (8, 12, 16, 20)[(variant // 5) % 4],
            "input_scale": (0.8, 1.2)[variant // 20],
            "residual_scale": (0.7, 0.85, 1.0, 1.15)[variant % 4],
        }
        return {**base, "rank": (2, 4)[variant % 2]} if template_id in {"LL01", "LL02"} else base
    if template_id.startswith("RG"):
        base = {**common, "nodes": (8, 12, 16, 20, 24)[variant % 5], "feature_scale": (0.8, 1.2)[variant // 20]}
        if template_id in {"RG01", "RG04"}:
            base.update(
                {
                    "segments": (2, 3, 4, 5)[(variant // 5) % 4],
                    "partition_policy": ("balanced", "front_loaded", "tail_loaded", "empty_middle")[variant % 4],
                }
            )
        if template_id in {"RG02", "RG03"}:
            base["edge_policy"] = ("ring", "hub", "strided", "duplicate_heavy")[(variant // 5) % 4]
            base["edge_phase"] = variant % base["nodes"]
        if template_id == "RG04":
            base["id_policy"] = ("sequential", "head_reuse", "strided", "duplicate_heavy")[(variant // 5) % 4]
            base["id_offset"] = variant % (base["nodes"] + 3)
        return base
    if template_id.startswith("QD"):
        base = {**common, "rows": (4, 8, 12, 16)[variant % 4], "width": (8, 16, 24, 32, 40)[variant % 5]}
        if template_id in {"QD01", "QD04"}:
            return {**base, "quant_bound": (63, 95, 111, 127)[variant % 4]}
        if template_id == "QD02":
            return {
                **base,
                "group_size": (4, 8)[variant % 2],
                "pack_order": ("high_low", "low_high", "pair_rotated", "rotated")[variant % 4],
            }
        return {
            **base,
            "block_size": 4 if base["width"] % 8 else (4 if (variant // 5) % 2 else 8),
            "fp8_bound": (192, 240)[variant % 2],
        }
    if template_id.startswith("VG"):
        base = {**common, "height": (4, 6, 8, 10)[variant % 4], "width": (4, 6, 8, 10, 12)[variant % 5]}
        if template_id == "VG02":
            return {
                **base,
                "padding_policy": ("zeros", "border", "reflection")[variant % 3],
                "interpolation_mode": ("bilinear", "nearest", "bicubic")[(variant // 3) % 3],
                "align_corners": bool((variant // 4) % 2),
                "grid_scale": (0.6, 0.8, 1.0, 1.2)[(variant // 5) % 4],
            }
        return {**base, "reorder_policy": ("roll_plus_1", "roll_minus_1", "roll_plus_2", "roll_plus_3")[variant % 4]}
    if template_id.startswith("SF"):
        base = {**common, "length": (16, 20, 24, 28)[variant % 4], "width": (4, 6, 8, 10, 12)[variant % 5]}
        if template_id == "SF01":
            return {**base, "filter_gain": (0.5, 0.8, 1.1, 1.4)[variant % 4]}
        if template_id == "SF02":
            return {**base, "filter_gain": (0.5, 0.8, 1.1, 1.4)[variant % 4], "hop": (2, 3, 4, 5)[variant % 4]}
        return {**base, "jitter": (0.03, 0.07, 0.11, 0.17)[variant % 4]}
    base = {**common, "items": (8, 12, 16, 20)[variant % 4], "width": (4, 6, 8, 10, 12)[variant % 5]}
    if template_id == "RR01":
        return {
            **base,
            "bag_split_policy": ("balanced", "front_loaded", "tail_loaded", "three_bag")[variant % 4],
            "id_offset": variant % (base["items"] + 4),
        }
    return {
        **base,
        "topk": (1, 2, 3, 4)[variant % 4],
        "corpus_skew": ("flat", "head_bias", "tail_bias", "alternating")[variant % 4],
    }


def _invariant(name: str, expression: str, witness: Any) -> dict[str, Any]:
    return {"name": name, "expression": expression, "witness": witness, "passed": True}


def _constraint(
    template_id: str,
    variant: int,
    requested: Mapping[str, Any],
    realized: Mapping[str, Any],
    invariants: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    inv = [dict(item) for item in invariants]
    if not inv or any(item["passed"] is not True for item in inv):
        raise ValueError(f"invalid invariant set:{template_id}:{variant}")
    return {
        "contract_version": CONSTRAINT_CONTRACT_VERSION,
        "backend": "deterministic_constructive_arithmetic",
        "template_id": template_id,
        "variant": variant,
        "requested_coordinates": dict(requested),
        "realized_coordinates": dict(realized),
        "invariants": inv,
        "invariant_payload_sha256": _canonical_sha256(inv),
    }


def _sparse_indices(
    rows: int, cols: int, degree: int, offset: int, pattern: str
) -> tuple[list[int], list[int], list[int]]:
    crow = [0]
    col: list[int] = []
    for row in range(rows):
        if pattern == "regular":
            values = [(row + slot + offset) % cols for slot in range(degree)]
        elif pattern == "long_tail":
            values = [(offset + (row // 3) * degree + slot) % cols for slot in range(degree)]
        elif pattern == "band":
            values = [(row + offset - degree // 2 + slot) % cols for slot in range(degree)]
        elif pattern == "strided":
            values = [(offset + row * 3 + slot * (cols - 1)) % cols for slot in range(degree)]
        else:
            raise ValueError(f"unknown sparse pattern:{pattern}")
        values = sorted(values)
        if len(values) != degree:
            raise ValueError(f"sparse row degree collision:{rows}:{cols}:{degree}:{row}")
        col.extend(values)
        crow.append(len(col))
    return crow, [row for row in range(rows) for _ in range(degree)], col


def _render(template_id: str, variant: int) -> tuple[str, list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Construct one closed registry cell and prove its typed coordinates."""
    c = _coords(template_id, variant)
    inv: list[dict[str, Any]] = [
        _invariant(
            "variant_bounds",
            "0 <= variant < template_quota",
            {"variant": variant, "quota": TEMPLATE_BY_ID[template_id].variants},
        )
    ]
    ops: list[dict[str, Any]]
    labels: dict[str, Any] = {
        "template_id": template_id,
        "variant": variant,
        "final_output_kind": "single_dense_tensor",
    }
    if template_id.startswith("SP"):
        if template_id == "SP04" and c["sparse_mode"] == "semi_structured_2to4":
            r, k, rhs_cols = c["rows"], c["cols"], c["rhs_cols"]
            inv += [
                _invariant("sparse_layout", "layout=semi_structured_2to4", "semi_structured_2to4"),
                _invariant("sparse_shape", "rows=cols=64 for A800 cuSPARSELt alignment", [r, k]),
                _invariant(
                    "two_of_four",
                    "exactly two values kept in every logical four-wide block",
                    {"rows": r, "blocks_per_row": k // 4, "kept_per_block": 2, "hardware_alignment": [r, k]},
                ),
                _invariant(
                    "sparse_mm_rhs", "rhs rows equal sparse cols and rhs_cols>0", {"rhs_rows": k, "rhs_cols": rhs_cols}
                ),
            ]
            pair_masks = ((1.0, 1.0, 0.0, 0.0), (1.0, 0.0, 1.0, 0.0), (1.0, 0.0, 0.0, 1.0), (0.0, 1.0, 1.0, 0.0))
            mask_values = pair_masks[c["semi_pair"]]
            labels.update(
                {
                    "sparse_layout": "semi_structured_2to4",
                    "sparse_mode": c["sparse_mode"],
                    "nnz": r * k // 2,
                    "rhs_cols": rhs_cols,
                    "semi_pair": c["semi_pair"],
                    "modality": "sparse",
                    "topology": "mask_2to4>semi_structured_compress>packed_anchor>mm",
                }
            )
            code = _code(
                "self, values, rhs",
                [
                    f"base = values[:{r * k}].reshape({r}, {k // 4}, 4).to(torch.float16)",
                    f"mask = torch.tensor({list(mask_values)}, device=values.device, dtype=base.dtype).reshape(1, 1, 4)",
                    f"dense = (base * mask).reshape({r}, {k})",
                    "semi = torch.sparse.to_sparse_semi_structured(dense)",
                    "compressed_values = semi.packed[:, :32]",
                    "compression_anchor = compressed_values.to(torch.float32).mean()",
                    "out = torch.mm(semi, rhs.to(torch.float16)).to(torch.float32)",
                    "return out + compression_anchor * 1e-6",
                ],
                [f"return [torch.randn({r * k}), torch.randn({k}, {rhs_cols})]"],
            )
            ops = [
                _op(
                    "sparse_semi_structured_construct",
                    ["torch.sparse.to_sparse_semi_structured"],
                    [("aten::_to_sparse_semi_structured", ""), ("aten::_cslt_compress", "")],
                ),
                _op("sparse_semi_structured_mm", ["torch.mm"], [("aten::mm", "")]),
            ]
        else:
            r, k, d = c["rows"], c["cols"], c["degree"]
            crow, row, col = _sparse_indices(r, k, d, c["index_offset"], c["index_pattern"])
            nnz = len(col)
            layout = "coo" if template_id in {"SP01", "SP04"} else "csr"
            inv += [
                _invariant("sparse_layout", "layout in {coo,csr}", layout),
                _invariant("sparse_shape", "rows>0 and cols>0", [r, k]),
                _invariant("nnz", "nnz=rows*degree", {"nnz": nnz, "expected": r * d}),
                _invariant("index_bounds", "0<=row<rows and 0<=col<cols", {"row_max": max(row), "col_max": max(col)}),
            ]
            if layout == "csr":
                inv.append(_invariant("csr_crow", "crow[0]=0, monotone, crow[-1]=nnz", {"crow": crow}))
            else:
                inv.append(
                    _invariant(
                        "coo_sorted_rows", "row indices are nondecreasing and columns are sorted within rows", True
                    )
                )
            labels.update(
                {
                    "sparse_layout": layout,
                    "nnz": nnz,
                    "index_pattern": c["index_pattern"],
                    "index_offset": c["index_offset"],
                    "modality": "sparse",
                }
            )
        if template_id == "SP01":
            labels["topology"] = "sparse_coo_construct>coalesce>spmm>tanh"
            code = _code(
                "self, values, dense",
                [
                    f"idx = torch.tensor([{row}, {col}], device=values.device, dtype=torch.long)",
                    f"s = torch.sparse_coo_tensor(idx, values[:{nnz}], size=({r}, {k}), device=values.device).coalesce()",
                    "out = torch.sparse.mm(s, dense)",
                    "return torch.tanh(out)",
                ],
                [f"return [torch.randn({nnz}), torch.randn({k}, 6)]"],
            )
            ops = [
                _op(
                    "sparse_coo_construct",
                    ["torch.sparse_coo_tensor"],
                    [("aten::_sparse_coo_tensor_with_dims_and_tensors", "")],
                ),
                _op("sparse_mm", ["torch.sparse.mm"], [("aten::_sparse_addmm", "")]),
            ]
        elif template_id == "SP02":
            labels["topology"] = "sparse_csr_construct>parallel_spmm_addmm>merge>relu"
            code = _code(
                "self, values, dense, bias",
                [
                    f"crow = torch.tensor({crow}, device=values.device, dtype=torch.long)",
                    f"col = torch.tensor({col}, device=values.device, dtype=torch.long)",
                    f"s = torch.sparse_csr_tensor(crow, col, values[:{nnz}], size=({r}, {k}), device=values.device)",
                    "a = torch.sparse.mm(s, dense)",
                    "b = torch.sparse.addmm(bias, s, dense)",
                    "return torch.relu(a + b)",
                ],
                [f"return [torch.randn({nnz}), torch.randn({k}, 6), torch.randn({r}, 6)]"],
            )
            ops = [
                _op(
                    "sparse_csr_construct",
                    ["torch.sparse_csr_tensor"],
                    [("aten::sparse_compressed_tensor", "comp_plain_value_size")],
                ),
                _op("sparse_mm", ["torch.sparse.mm"], [("aten::_sparse_addmm", "")]),
                _op("sparse_addmm", ["torch.sparse.addmm"], [("aten::_sparse_addmm", "")]),
            ]
        elif template_id == "SP03":
            labels["topology"] = "sparse_csr_pattern>sampled_addmm>dense_boundary>sigmoid"
            code = _code(
                "self, values, left, right",
                [
                    f"crow = torch.tensor({crow}, device=values.device, dtype=torch.long)",
                    f"col = torch.tensor({col}, device=values.device, dtype=torch.long)",
                    f"pat = torch.sparse_csr_tensor(crow, col, values[:{nnz}], size=({r}, {k}), device=values.device)",
                    "sampled = torch.sparse.sampled_addmm(pat, left, right)",
                    "return torch.sigmoid(sampled.to_dense())",
                ],
                [f"return [torch.ones({nnz}), torch.randn({r}, 5), torch.randn(5, {k})]"],
            )
            ops = [
                _op(
                    "sparse_csr_construct",
                    ["torch.sparse_csr_tensor"],
                    [("aten::sparse_compressed_tensor", "comp_plain_value_size")],
                ),
                _op("sparse_sampled_addmm", ["torch.sparse.sampled_addmm"], [("aten::sparse_sampled_addmm", "")]),
            ]
        elif c["sparse_mode"] == "coo_softmax":
            labels.update(
                {
                    "sparse_layout": "coo",
                    "sparse_mode": c["sparse_mode"],
                    "sparse_operator": "sparse_softmax",
                    "topology": "sparse_coo_construct>coalesce>sparse_softmax>dense_boundary",
                }
            )
            code = _code(
                "self, values",
                [
                    f"idx = torch.tensor([{row}, {col}], device=values.device, dtype=torch.long)",
                    f"s = torch.sparse_coo_tensor(idx, values[:{nnz}], size=({r}, {k}), device=values.device).coalesce()",
                    "p = torch.sparse.softmax(s, dim=1)",
                    "return p.to_dense()",
                ],
                [f"return [torch.randn({nnz})]"],
            )
            ops = [
                _op(
                    "sparse_coo_construct",
                    ["torch.sparse_coo_tensor"],
                    [("aten::_sparse_coo_tensor_with_dims_and_tensors", "")],
                ),
                _op("sparse_softmax", ["torch.sparse.softmax"], [("aten::_sparse_softmax", "")]),
            ]
    elif template_id.startswith("MO"):
        t, e, topk = c["tokens"], c["experts"], c["topk"]
        capacity = max(2, (t * topk + e - 1) // e)
        if template_id == "MO03":
            inv += [
                _invariant("expert_choice_topk", "1<=topk<=tokens", {"topk": topk, "tokens": t}),
                _invariant("expert_choice_token_indices", "0<=chosen_token<tokens", {"tokens": t}),
            ]
        else:
            inv += [
                _invariant("token_choice_topk", "1<=topk<=experts", {"topk": topk, "experts": e}),
                _invariant("routing_indices", "0<=route<experts", {"experts": e}),
            ]
        if template_id == "MO04":
            inv.append(_invariant("capacity", "each routed expert keeps at most capacity assignments", capacity))
        bias_values = {
            "uniform": [0.0 for _ in range(e)],
            "head_heavy": [0.35 if index < max(1, e // 3) else -0.05 for index in range(e)],
            "alternating": [0.2 if index % 2 == 0 else -0.1 for index in range(e)],
            "tail_heavy": [-0.05 if index < max(1, e // 2) else 0.3 for index in range(e)],
        }[c["router_bias_policy"]]
        bias_line = f"routing_bias = torch.tensor({bias_values}, device=x.device, dtype=x.dtype)"
        labels.update(
            {
                "topk": topk,
                "experts": e,
                "router_bias_policy": c["router_bias_policy"],
                "router_temperature": c["router_temperature"],
                "modality": "moe",
            }
        )
        if template_id == "MO01":
            labels["topology"] = "router_matmul+bias>topk>expert_gather>batched_matmul>weighted_reduce"
            lines = [
                bias_line,
                f"scores = (x @ router + routing_bias) / {c['router_temperature']}",
                f"weight, route = torch.topk(scores, k={topk}, dim=-1)",
                "picked = expert.index_select(0, route.reshape(-1)).reshape(x.shape[0], route.shape[1], x.shape[1], x.shape[1])",
                "mixed = torch.matmul(picked, x.unsqueeze(1).unsqueeze(-1)).squeeze(-1)",
                "return (torch.softmax(weight, dim=-1).unsqueeze(-1) * mixed).sum(dim=1)",
            ]
        elif template_id == "MO02":
            labels["topology"] = "router_matmul+bias>group_select>masked_topk>expert_gather>weighted_reduce"
            lines = [
                bias_line,
                f"scores = (x @ router + routing_bias) / {c['router_temperature']}",
                "groups = scores.reshape(scores.shape[0], 2, scores.shape[1] // 2)",
                "group_best = groups.amax(dim=-1)",
                "_, chosen_group = torch.topk(group_best, k=1, dim=-1)",
                "mask = torch.zeros_like(groups).scatter(1, chosen_group.unsqueeze(-1), 1.0).reshape_as(scores)",
                "masked_scores = scores.masked_fill(mask == 0, -1e4)",
                f"weight, route = torch.topk(masked_scores, k={topk}, dim=-1)",
                "picked = expert.index_select(0, route.reshape(-1)).reshape(x.shape[0], route.shape[1], x.shape[1], x.shape[1])",
                "mixed = torch.matmul(picked, x.unsqueeze(1).unsqueeze(-1)).squeeze(-1)",
                "return (torch.softmax(weight, dim=-1).unsqueeze(-1) * mixed).sum(dim=1)",
            ]
        elif template_id == "MO03":
            labels["topology"] = "router_matmul+bias>expert_choice_topk>token_gather>expert_einsum>broadcast_merge"
            lines = [
                bias_line,
                f"scores = (x @ router + routing_bias) / {c['router_temperature']}",
                f"_, token_for_expert = torch.topk(scores.transpose(0, 1), k={topk}, dim=-1)",
                f"selected = x.index_select(0, token_for_expert.reshape(-1)).reshape(scores.shape[1], {topk}, x.shape[1])",
                "transformed = torch.einsum('eij,etj->eti', expert, selected)",
                "expert_weight = torch.softmax(scores.transpose(0, 1), dim=-1)",
                "bias = torch.matmul(expert_weight, x).mean(dim=0, keepdim=True)",
                "return transformed.mean(dim=(0, 1)).unsqueeze(0).expand_as(x) + bias.expand_as(x)",
            ]
        else:
            labels["topology"] = (
                "router_matmul+bias>topk>per_expert_capacity_mask>expert_gather+shared_expert>normalized_merge"
            )
            labels["capacity"] = capacity
            lines = [
                bias_line,
                f"scores = (x @ router + routing_bias) / {c['router_temperature']}",
                f"weight, route = torch.topk(scores, k={topk}, dim=-1)",
                "route = route.remainder(expert.shape[0])",
                "route_flat = route.reshape(-1)",
                "expert_axis = torch.arange(expert.shape[0], device=x.device).reshape(1, -1)",
                "assignment = (route_flat.unsqueeze(-1) == expert_axis).to(x.dtype)",
                "position = torch.cumsum(assignment, dim=0)",
                "slot = position.gather(1, route_flat.unsqueeze(-1)).squeeze(-1)",
                f"keep = (slot <= {capacity}).reshape_as(weight).to(weight.dtype)",
                "picked = expert.index_select(0, route_flat).reshape(x.shape[0], route.shape[1], x.shape[1], x.shape[1])",
                "mixed = torch.matmul(picked, x.unsqueeze(1).unsqueeze(-1)).squeeze(-1)",
                "shared_out = x @ shared",
                "combine = torch.softmax(weight, dim=-1) * keep",
                "return shared_out + (combine.unsqueeze(-1) * mixed).sum(dim=1) / combine.sum(dim=1, keepdim=True).clamp_min(1e-5)",
            ]
        signature = "self, x, router, expert, shared" if template_id == "MO04" else "self, x, router, expert"
        inputs = (
            [f"return [torch.randn({t}, 8), torch.randn(8, {e}), torch.randn({e}, 8, 8), torch.randn(8, 8)]"]
            if template_id == "MO04"
            else [f"return [torch.randn({t}, 8), torch.randn(8, {e}), torch.randn({e}, 8, 8)]"]
        )
        code = _code(signature, lines, inputs)
        ops = [
            _op("moe_topk_route", ["torch.topk"], [("aten::topk", "")]),
            _op(
                "moe_expert_gather",
                ["x.index_select" if template_id == "MO03" else "expert.index_select"],
                [("aten::index_select", "")],
            ),
            _op("moe_weighted_combine", ["torch.softmax"], [("aten::_softmax", "")]),
        ]
    elif template_id.startswith("SS"):
        length, width = c["length"], c["width"]
        inv += [
            _invariant("scan_length", "length>=8", length),
            _invariant("state_width", "width>0", width),
            _invariant("causal_state", "all recurrence steps use a preceding state", True),
        ]
        labels.update({"scan_length": length, "input_gain": c["input_gain"], "modality": "state_space"})
        if template_id == "SS01":
            decay_scale = {
                "decay_scale_035": 0.35,
                "decay_scale_140": 1.4,
                "alternating_base_080_step_085": 0.8,
                "decay_scale_110": 1.1,
            }[c["decay_policy"]]
            labels.update(
                {"decay_policy": c["decay_policy"], "topology": "input>selective_decay_recurrence>stack>temporal_mean"}
            )
            lines = [
                f"x_scaled = x * {c['input_gain']}",
                f"delta = torch.sigmoid(x_scaled) * {decay_scale}",
                "state0 = torch.zeros_like(x_scaled[:, 0])",
            ]
            for index in range(length):
                alternating = 0.85 if c["decay_policy"] == "alternating_base_080_step_085" and index % 2 else 1.0
                lines.append(
                    f"state{index + 1} = state{index} * torch.exp(-delta[:, {index}] * {alternating}) + x_scaled[:, {index}]"
                )
            lines.append(
                "return torch.stack(["
                + ", ".join(f"state{index + 1}" for index in range(length))
                + "], dim=1).mean(dim=1)"
            )
        elif template_id == "SS02":
            labels.update(
                {
                    "gate_bias": c["gate_bias"],
                    "carry_decay": c["carry_decay"],
                    "topology": "causal_pad>depthwise_conv>biased_gate>causal_recurrence>temporal_sum",
                }
            )
            lines = [
                f"x_scaled = x * {c['input_gain']}",
                "causal = F.pad(x_scaled.transpose(1, 2), (2, 0))",
                "conv = F.conv1d(causal, kernel, groups=x_scaled.shape[-1]).transpose(1, 2)",
                f"gate = torch.sigmoid(conv + {c['gate_bias']})",
                "s0 = torch.zeros_like(gate[:, 0])",
            ]
            for index in range(length):
                lines.append(f"s{index + 1} = s{index} * {c['carry_decay']} + gate[:, {index}]")
            lines.append(
                "return torch.stack([" + ", ".join(f"s{index + 1}" for index in range(length)) + "], dim=1).sum(dim=1)"
            )
        else:
            split_by_policy = {
                "early": max(1, length // 3),
                "middle": length // 2,
                "late": length - max(1, length // 3),
                "middle_half_carry": length // 2,
            }
            split = split_by_policy[c["chunk_split"]]
            carry_scale = 0.5 if c["chunk_split"] == "middle_half_carry" else 1.0
            labels.update(
                {
                    "chunk_split": c["chunk_split"],
                    "chunk_boundary": split,
                    "topology": "input>tanh>left_prefix_scan+right_prefix_scan_with_carry>concat>temporal_mean",
                }
            )
            lines = [
                f"x_scaled = x * {c['input_gain']}",
                "a = torch.tanh(x_scaled)",
                f"left = torch.cumsum(a[:, :{split}], dim=1)",
                f"right = torch.cumsum(a[:, {split}:], dim=1) + left[:, -1:] * {carry_scale}",
                "return torch.cat([left, right], dim=1).mean(dim=1)",
            ]
        inputs = (
            [f"return [torch.randn(2, {length}, {width}), torch.randn({width}, 1, 3)]"]
            if template_id == "SS02"
            else [f"return [torch.randn(2, {length}, {width})]"]
        )
        code = _code("self, x, kernel" if template_id == "SS02" else "self, x", lines, inputs)
        scan_source, scan_identity = (
            ("torch.exp", ("aten::exp", ""))
            if template_id == "SS01"
            else (
                ("torch.sigmoid", ("aten::sigmoid", ""))
                if template_id == "SS02"
                else ("torch.cumsum", ("aten::cumsum", ""))
            )
        )
        ops = [
            _op("selective_scan", [scan_source], [scan_identity]),
            _op(
                "causal_conv" if template_id == "SS02" else "state_combine",
                ["F.conv1d" if template_id == "SS02" else "torch.stack" if template_id == "SS01" else "torch.cat"],
                [
                    (
                        ("aten::convolution", "")
                        if template_id == "SS02"
                        else ("aten::stack", "") if template_id == "SS01" else ("aten::cat", "")
                    )
                ],
            ),
        ]
    elif template_id.startswith("AC"):
        ql, kl, heads, kvh = c["q_len"], c["kv_len"], c["heads"], min(c["kv_heads"], c["heads"])
        if heads % kvh:
            kvh = 1
        ratio = heads // kvh
        inv += [
            _invariant("attention_heads", "heads%kv_heads=0", {"heads": heads, "kv_heads": kvh}),
            _invariant("gqa_ratio", "ratio>=1", ratio),
            _invariant("sequence_bounds", "0<q_len<=kv_len+cache", {"q_len": ql, "kv_len": kl}),
            _invariant("page_table", "all page ids are in cache range", {"max_page": 1, "pages": 2}),
        ]
        labels.update(
            {"q_len": ql, "kv_len": kl, "gqa_ratio": ratio, "query_scale": c["query_scale"], "modality": "attention"}
        )
        if template_id == "AC01":
            labels.update(
                {"rope_base": c["rope_base"], "topology": "rope_rotation>kv_head_repeat>causal_sdpa>head_flatten"}
            )
            lines = [
                f"q = q * {c['query_scale']}",
                "pos = torch.arange(q.shape[2], device=q.device, dtype=q.dtype)",
                "freq = torch.tensor([1.0, 0.1], device=q.device, dtype=q.dtype)",
                f"angle = pos.view(1, 1, -1, 1) * freq.view(1, 1, 1, -1) / {c['rope_base']}",
                "cosine = torch.cos(angle)",
                "sine = torch.sin(angle)",
                "q_even = q[..., 0::2]",
                "q_odd = q[..., 1::2]",
                "q_rot = torch.stack([q_even * cosine - q_odd * sine, q_even * sine + q_odd * cosine], dim=-1).flatten(-2)",
                f"k_rep = k.repeat_interleave({ratio}, dim=1)",
                f"v_rep = v.repeat_interleave({ratio}, dim=1)",
                "out = F.scaled_dot_product_attention(q_rot, k_rep, v_rep, is_causal=True)",
                "return out.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)",
            ]
        elif template_id == "AC02":
            labels.update(
                {
                    "sliding_window": c["sliding_window"],
                    "alibi_slope": c["alibi_slope"],
                    "topology": "kv_head_repeat>qk_matmul>sliding_alibi_mask>softmax>value_matmul>head_reduce",
                }
            )
            lines = [
                f"q = q * {c['query_scale']}",
                f"k_rep = k.repeat_interleave({ratio}, dim=1)",
                f"v_rep = v.repeat_interleave({ratio}, dim=1)",
                "scores = torch.matmul(q, k_rep.transpose(-1, -2)) / q.shape[-1] ** 0.5",
                "pos_q = torch.arange(q.shape[2], device=q.device).view(-1, 1)",
                "pos_k = torch.arange(k.shape[2], device=q.device).view(1, -1)",
                f"mask = (pos_k <= pos_q + k.shape[2] - q.shape[2]) & (pos_k >= pos_q + k.shape[2] - q.shape[2] - {c['sliding_window']})",
                f"alibi = (pos_k - pos_q).to(q.dtype) * {c['alibi_slope']}",
                "scores = scores + alibi",
                "masked = scores.masked_fill(~mask, -1e4)",
                "return torch.matmul(torch.softmax(masked, dim=-1), v_rep).mean(dim=1)",
            ]
        elif template_id == "AC03":
            order = c["append_policy"]
            labels.update({"append_policy": order, "topology": "cache_append>kv_head_repeat>sdpa>head_reduce"})
            if order == "cache_then_new":
                k_line, v_line = "new_k = torch.cat([cache_k, k], dim=2)", "new_v = torch.cat([cache_v, v], dim=2)"
            elif order == "new_then_cache":
                k_line, v_line = "new_k = torch.cat([k, cache_k], dim=2)", "new_v = torch.cat([v, cache_v], dim=2)"
            elif order == "scaled_cache_075":
                k_line, v_line = (
                    "new_k = torch.cat([cache_k * 0.75, k], dim=2)",
                    "new_v = torch.cat([cache_v * 0.75, v], dim=2)",
                )
            else:
                k_line, v_line = (
                    "new_k = torch.cat([cache_k * 0.9, k], dim=2)",
                    "new_v = torch.cat([cache_v * 0.9, v], dim=2)",
                )
            lines = [
                f"q = q * {c['query_scale']}",
                k_line,
                v_line,
                f"k_rep = new_k.repeat_interleave({ratio}, dim=1)",
                f"v_rep = new_v.repeat_interleave({ratio}, dim=1)",
                "out = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=False)",
                "return out.mean(dim=1)",
            ]
        else:
            page_ids = list(c["page_order"])
            labels.update(
                {
                    "page_order": page_ids,
                    "topology": "page_table_index_select>page_flatten>kv_head_repeat>sdpa>head_reduce",
                }
            )
            lines = [
                f"q = q * {c['query_scale']}",
                f"page_ids = torch.tensor({page_ids}, device=pages_k.device, dtype=torch.long)",
                "gathered_k = pages_k.index_select(0, page_ids).reshape(pages_k.shape[1], pages_k.shape[2] * 2, pages_k.shape[3]).unsqueeze(0)",
                "gathered_v = pages_v.index_select(0, page_ids).reshape(pages_v.shape[1], pages_v.shape[2] * 2, pages_v.shape[3]).unsqueeze(0)",
                f"k_rep = gathered_k.repeat_interleave({ratio}, dim=1)",
                f"v_rep = gathered_v.repeat_interleave({ratio}, dim=1)",
                "return F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=False).mean(dim=1)",
            ]
        if template_id == "AC04":
            inputs = [
                f"return [torch.randn(1, {heads}, {ql}, 4), torch.randn(2, {kvh}, {kl // 2 if kl >= 6 else 3}, 4), torch.randn(2, {kvh}, {kl // 2 if kl >= 6 else 3}, 4)]"
            ]
            signature = "self, q, pages_k, pages_v"
        elif template_id == "AC03":
            inputs = [
                f"return [torch.randn(1, {heads}, {ql}, 4), torch.randn(1, {kvh}, {ql}, 4), torch.randn(1, {kvh}, {ql}, 4), torch.randn(1, {kvh}, {kl}, 4), torch.randn(1, {kvh}, {kl}, 4)]"
            ]
            signature = "self, q, k, v, cache_k, cache_v"
        else:
            inputs = [
                f"return [torch.randn(1, {heads}, {ql}, 4), torch.randn(1, {kvh}, {kl}, 4), torch.randn(1, {kvh}, {kl}, 4)]"
            ]
            signature = "self, q, k, v"
        code = _code(signature, lines, inputs)
        if template_id == "AC01":
            ops = [
                _op(
                    "gqa_attention",
                    ["F.scaled_dot_product_attention"],
                    [
                        ("aten::_scaled_dot_product_efficient_attention", ""),
                        ("aten::_scaled_dot_product_flash_attention", ""),
                        ("aten::bmm", ""),
                        ("aten::_safe_softmax", ""),
                    ],
                ),
                _op("rope", ["torch.cos"], [("aten::cos", "")]),
            ]
        elif template_id == "AC02":
            ops = [
                _op("sliding_attention_math", ["torch.softmax"], [("aten::_softmax", "")]),
                _op("sliding_mask", ["scores.masked_fill"], [("aten::masked_fill", "Scalar")]),
            ]
        elif template_id == "AC03":
            ops = [
                _op(
                    "append_cache_attention",
                    ["F.scaled_dot_product_attention"],
                    [
                        ("aten::_scaled_dot_product_efficient_attention", ""),
                        ("aten::_scaled_dot_product_flash_attention", ""),
                        ("aten::bmm", ""),
                        ("aten::_safe_softmax", ""),
                    ],
                ),
                _op("append_kv_cache", ["torch.cat"], [("aten::cat", "")]),
            ]
        else:
            ops = [
                _op(
                    "paged_cache_attention",
                    ["F.scaled_dot_product_attention"],
                    [
                        ("aten::_scaled_dot_product_efficient_attention", ""),
                        ("aten::_scaled_dot_product_flash_attention", ""),
                        ("aten::bmm", ""),
                        ("aten::_safe_softmax", ""),
                    ],
                ),
                _op("page_table_gather", ["pages_k.index_select"], [("aten::index_select", "")]),
            ]
    elif template_id.startswith("LL"):
        t, h = c["tokens"], c["hidden"]
        rank = c.get("rank")
        inv.append(_invariant("residual_shape", "all block branches have hidden width", h))
        if rank is not None:
            inv.append(_invariant("hidden_rank", "0<rank<=hidden", {"rank": rank, "hidden": h}))
        residual_scale = c["residual_scale"]
        labels.update(
            {"hidden": h, "input_scale": c["input_scale"], "residual_scale": residual_scale, "modality": "llm_block"}
        )
        if rank is not None:
            labels["rank"] = rank
        if template_id == "LL01":
            labels["topology"] = "low_rank_kv_projection+q_projection>attention_scores>softmax>residual_merge"
            lines = [
                f"x_scaled = x * {c['input_scale']}",
                "latent = torch.matmul(x_scaled, down)",
                "kv = torch.matmul(latent, up)",
                "q = torch.matmul(x_scaled, q_proj)",
                "score = torch.matmul(q, kv.transpose(-1, -2)) / q.shape[-1] ** 0.5",
                f"return x_scaled * {residual_scale} + torch.matmul(torch.softmax(score, dim=-1), kv)",
            ]
        elif template_id == "LL02":
            labels["topology"] = "cache_low_rank_compress+restore>decode_q_projection>attention_scores>residual_merge"
            lines = [
                f"x_scaled = x * {c['input_scale']}",
                "compressed = torch.matmul(cache, down)",
                "restored = torch.matmul(compressed, up)",
                "q = torch.matmul(x_scaled, q_proj)",
                "score = torch.matmul(q, restored.transpose(-1, -2)) / q.shape[-1] ** 0.5",
                f"return torch.matmul(torch.softmax(score, dim=-1), restored) + x_scaled * {residual_scale}",
            ]
        elif template_id == "LL03":
            labels["topology"] = "rms_norm>fused_projection>swiglu_gate>residual_merge"
            lines = [
                f"x_scaled = x * {c['input_scale']}",
                f"norm = F.rms_norm(x_scaled, ({h},), eps=1e-5)",
                "gate, value = torch.matmul(norm, fused).chunk(2, dim=-1)",
                f"return x_scaled * {residual_scale} + F.silu(gate) * value",
            ]
        else:
            labels["topology"] = "cross_q_projection>cross_attention_scores>softmax>geglu_gate>residual_merge"
            lines = [
                f"x_scaled = x * {c['input_scale']}",
                "q = torch.matmul(x_scaled, q_proj)",
                "score = torch.matmul(q, context.transpose(-1, -2)) / q.shape[-1] ** 0.5",
                "cross = torch.matmul(torch.softmax(score, dim=-1), context)",
                "gate, value = torch.matmul(cross, fused).chunk(2, dim=-1)",
                f"return x_scaled * {residual_scale} + F.gelu(gate) * value",
            ]
        if template_id == "LL02":
            inputs = [
                f"return [torch.randn(1, {t}, {h}), torch.randn(1, {t + 2}, {h}), torch.randn({h}, {rank}), torch.randn({rank}, {h}), torch.randn({h}, {h})]"
            ]
        elif template_id == "LL04":
            inputs = [
                f"return [torch.randn(1, {t}, {h}), torch.randn(1, {t + 2}, {h}), torch.randn({h}, {h}), torch.randn({h}, {2 * h})]"
            ]
        elif template_id == "LL03":
            inputs = [f"return [torch.randn(1, {t}, {h}), torch.randn({h}, {2 * h})]"]
        else:
            inputs = [
                f"return [torch.randn(1, {t}, {h}), torch.randn({h}, {rank}), torch.randn({rank}, {h}), torch.randn({h}, {h})]"
            ]
        signature = (
            "self, x, down, up, q_proj"
            if template_id == "LL01"
            else (
                "self, x, cache, down, up, q_proj"
                if template_id == "LL02"
                else "self, x, fused" if template_id == "LL03" else "self, x, context, q_proj, fused"
            )
        )
        code = _code(signature, lines, inputs)
        projection_identities = (
            [("aten::mm", ""), ("aten::bmm", "")]
            if template_id in {"LL01", "LL02", "LL04"}
            else [("aten::rms_norm", ""), ("aten::_fused_rms_norm", "")]
        )
        ops = [
            _op(
                "llm_projection",
                ["torch.matmul" if template_id in {"LL01", "LL02", "LL04"} else "F.rms_norm"],
                projection_identities,
            ),
            _op(
                "llm_gate",
                ["F.silu" if template_id == "LL03" else "torch.softmax"],
                [("aten::silu", "") if template_id == "LL03" else ("aten::_softmax", "")],
            ),
        ]
    elif template_id.startswith("RG"):
        n = c["nodes"]
        offsets: list[int] | None = None
        edge_src: list[int] | None = None
        edge_dst: list[int] | None = None
        labels.update({"nodes": n, "feature_scale": c["feature_scale"], "modality": "ragged_graph"})
        if template_id in {"RG01", "RG04"}:
            seg = c["segments"]
            if c["partition_policy"] == "balanced":
                counts = [n // seg + (index < n % seg) for index in range(seg)]
            elif c["partition_policy"] == "front_loaded":
                counts = [n - (seg - 1)] + [1 for _ in range(seg - 1)]
            elif c["partition_policy"] == "tail_loaded":
                counts = [1 for _ in range(seg - 1)] + [n - (seg - 1)]
            else:
                empty_index = seg // 2
                nonempty = seg - 1
                counts = [0 for _ in range(seg)]
                remainder = n % nonempty
                for rank, index in enumerate(item for item in range(seg) if item != empty_index):
                    counts[index] = n // nonempty + (rank < remainder)
            offsets = [0]
            for count in counts:
                offsets.append(offsets[-1] + count)
            inv.append(
                _invariant("offset_monotonic", "offsets monotone and terminal=n", {"offsets": offsets, "terminal": n})
            )
            labels.update({"segments": seg, "partition_policy": c["partition_policy"]})
        if template_id in {"RG02", "RG03"}:
            edge_src = [index % n for index in range(2 * n)]
            if c["edge_policy"] == "ring":
                edge_dst = [(index + 1 + c["edge_phase"]) % n for index in range(2 * n)]
            elif c["edge_policy"] == "hub":
                edge_dst = [0 if index % 3 else (index + c["edge_phase"]) % n for index in range(2 * n)]
            elif c["edge_policy"] == "strided":
                edge_dst = [(index * 3 + c["edge_phase"]) % n for index in range(2 * n)]
            else:
                edge_dst = [((index // 2) + c["edge_phase"]) % n for index in range(2 * n)]
            inv.append(_invariant("node_edge_bounds", "0<=edge<n", {"nodes": n, "edge_max": max(edge_src + edge_dst)}))
            labels.update({"edge_policy": c["edge_policy"], "edge_phase": c["edge_phase"]})
        if template_id == "RG01":
            labels["topology"] = "offset_tensor>prefix_sum>paired_offset_gather>segment_normalize"
            code = _code(
                "self, values",
                [
                    f"values_scaled = values * {c['feature_scale']}",
                    f"offsets = torch.tensor({offsets}, device=values.device, dtype=torch.long)",
                    "prefix = torch.cat([torch.zeros_like(values_scaled[:1]), values_scaled.cumsum(dim=0)], dim=0)",
                    "sums = prefix.index_select(0, offsets[1:]) - prefix.index_select(0, offsets[:-1])",
                    "counts = (offsets[1:] - offsets[:-1]).clamp_min(1).to(values.dtype).unsqueeze(-1)",
                    "return sums / counts",
                ],
                [f"return [torch.randn({n}, 6)]"],
            )
            ops = [
                _op(
                    "ragged_offsets",
                    ["torch.cumsum", "prefix.index_select"],
                    [("aten::cumsum", ""), ("aten::index_select", "")],
                )
            ]
        elif template_id == "RG02":
            assert edge_src is not None and edge_dst is not None
            labels["topology"] = (
                "edge_index_tensor>source_gather>destination_routing_reduce>unique_index_add>degree_normalize"
            )
            code = _code(
                "self, x",
                [
                    f"x_scaled = x * {c['feature_scale']}",
                    f"src = torch.tensor({edge_src}, device=x.device, dtype=torch.long)",
                    f"dst = torch.tensor({edge_dst}, device=x.device, dtype=torch.long)",
                    "messages = x_scaled.index_select(0, src)",
                    "node_ids = torch.arange(x.shape[0], device=x.device, dtype=torch.long)",
                    "routing = (node_ids.unsqueeze(-1) == dst.unsqueeze(0)).to(x.dtype)",
                    "grouped = torch.matmul(routing, messages)",
                    "out0 = torch.zeros_like(x_scaled)",
                    "out = out0.index_add(0, node_ids, grouped)",
                    "degree = routing.sum(dim=1)",
                    "return out / degree.clamp_min(1).unsqueeze(-1)",
                ],
                [f"return [torch.randn({n}, 6)]"],
            )
            ops = [
                _op("graph_message_gather", ["x_scaled.index_select"], [("aten::index_select", "")]),
                _op("graph_segment_reduce", ["out0.index_add"], [("aten::index_add", "")]),
            ]
        elif template_id == "RG03":
            assert edge_src is not None and edge_dst is not None
            labels["topology"] = "edge_index_tensor>pair_score>exp>destination_routing_softmax>unique_index_add"
            code = _code(
                "self, x",
                [
                    f"x_scaled = x * {c['feature_scale']}",
                    f"src = torch.tensor({edge_src}, device=x.device, dtype=torch.long)",
                    f"dst = torch.tensor({edge_dst}, device=x.device, dtype=torch.long)",
                    "source_features = x_scaled.index_select(0, src)",
                    "score = (source_features * x_scaled.index_select(0, dst)).sum(dim=-1)",
                    "exp_score = torch.exp(score - score.amax())",
                    "node_ids = torch.arange(x.shape[0], device=x.device, dtype=torch.long)",
                    "routing = (node_ids.unsqueeze(-1) == dst.unsqueeze(0)).to(x.dtype)",
                    "denom = torch.matmul(routing, exp_score.unsqueeze(-1)).squeeze(-1)",
                    "alpha = exp_score / denom.index_select(0, dst).clamp_min(1e-5)",
                    "grouped = torch.matmul(routing, source_features * alpha.unsqueeze(-1))",
                    "out0 = torch.zeros_like(x_scaled)",
                    "return out0.index_add(0, node_ids, grouped)",
                ],
                [f"return [torch.randn({n}, 6)]"],
            )
            ops = [
                _op("segment_attention_score", ["torch.exp"], [("aten::exp", "")]),
                _op("segment_attention_normalize", ["denom.index_select"], [("aten::index_select", "")]),
                _op("segment_attention_reduce", ["out0.index_add"], [("aten::index_add", "")]),
            ]
        else:
            assert offsets is not None
            labels["topology"] = "ragged_ids_offsets>embedding_bag>dense_interaction_projection"
            labels["id_policy"] = c["id_policy"]
            ids = [
                (
                    index // 2
                    if c["id_policy"] == "duplicate_heavy"
                    else (
                        index // 3
                        if c["id_policy"] == "head_reuse"
                        else index * 2 if c["id_policy"] == "strided" else index
                    )
                )
                + c["id_offset"]
                for index in range(n)
            ]
            ids = [value % (n + 3) for value in ids]
            code = _code(
                "self, weight",
                [
                    f"weight_scaled = weight * {c['feature_scale']}",
                    f"ids = torch.tensor({ids}, device=weight.device, dtype=torch.long)",
                    f"offsets = torch.tensor({offsets}, device=weight.device, dtype=torch.long)",
                    "bag = F.embedding_bag(ids, weight_scaled, offsets, include_last_offset=True, mode='mean')",
                    "return bag @ weight_scaled[:bag.shape[-1]].transpose(0, 1)",
                ],
                [f"return [torch.randn({n + 3}, 6)]"],
            )
            ops = [_op("embedding_bag_offsets", ["F.embedding_bag"], [("aten::_embedding_bag_forward_only", "")])]
    elif template_id.startswith("QD"):
        rows, width = c["rows"], c["width"]
        inv.append(_invariant("scale_positive", "all constructed scales are >0", 1.0))
        labels.update(
            {"quantization": template_id, "width": width, "precision": "fake_quantized", "modality": "quantization"}
        )
        if template_id == "QD01":
            inv.append(
                _invariant(
                    "integer_range",
                    "quantized values are clipped to [-bound,bound]",
                    [-c["quant_bound"], c["quant_bound"]],
                )
            )
            labels.update(
                {
                    "per_channel_bound": c["quant_bound"],
                    "topology": "per_channel_absmax>round_clip_qdq>dense_projection",
                }
            )
            lines = [
                f"scale = x.abs().amax(dim=0, keepdim=True).clamp_min(1e-5) / {c['quant_bound']}.0",
                f"q = torch.round(x / scale).clamp(-{c['quant_bound']}, {c['quant_bound']})",
                "return torch.matmul(q * scale, weight)",
            ]
        elif template_id == "QD02":
            group = c["group_size"]
            order = c["pack_order"]
            inv += [
                _invariant("group_divisibility", "width%group_size=0", {"width": width, "group_size": group}),
                _invariant("integer_range", "INT4 values are clipped to [-8,7]", [-8, 7]),
            ]
            labels.update(
                {
                    "group_size": group,
                    "pack_order": order,
                    "topology": "group_absmax>int4_round_clip>bitwise_pack>bitwise_unpack>qdq_projection",
                }
            )
            rotation = 1 if order == "rotated" else 2 if order == "pair_rotated" else 0
            prepack = f"q4 = torch.roll(q4, shifts={rotation}, dims=-1)" if rotation else "q4 = q4"
            postunpack = (
                f"unpacked = torch.roll(unpacked, shifts={-rotation}, dims=-1)" if rotation else "unpacked = unpacked"
            )
            if order == "low_high":
                pack_lines = [
                    "packed = torch.bitwise_or(torch.bitwise_left_shift(odd, 4), even)",
                    "unpacked = torch.stack([torch.bitwise_and(packed, 15) - 8, torch.bitwise_right_shift(packed, 4) - 8], dim=-1).reshape_as(groups)",
                ]
            else:
                pack_lines = [
                    "packed = torch.bitwise_or(torch.bitwise_left_shift(even, 4), odd)",
                    "unpacked = torch.stack([torch.bitwise_right_shift(packed, 4) - 8, torch.bitwise_and(packed, 15) - 8], dim=-1).reshape_as(groups)",
                ]
            lines = [
                f"groups = x.reshape(x.shape[0], {width // group}, {group})",
                "scale = groups.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5) / 7.0",
                "q4 = torch.round(groups / scale).clamp(-8, 7).to(torch.int32)",
                prepack,
                "even = q4[..., 0::2] + 8",
                "odd = q4[..., 1::2] + 8",
                *pack_lines,
                postunpack,
                "return torch.matmul((unpacked.to(x.dtype) * scale).reshape_as(x), weight)",
            ]
        elif template_id == "QD03":
            block_size, fp8_bound = c["block_size"], c["fp8_bound"]
            inv += [
                _invariant("block_divisibility", "width%block_size=0", {"width": width, "block_size": block_size}),
                _invariant("integer_range", "fake-FP8 values are clipped to [-bound,bound]", [-fp8_bound, fp8_bound]),
            ]
            labels.update(
                {
                    "fp8_block_size": block_size,
                    "fp8_bound": fp8_bound,
                    "topology": "block_absmax>fake_fp8_round_clip>qdq_projection",
                }
            )
            lines = [
                f"block = x.reshape(x.shape[0], -1, {block_size})",
                f"scale = block.abs().amax(dim=-1, keepdim=True).clamp_min(1e-5) / {fp8_bound}.0",
                f"fake_fp8 = torch.round(block / scale).clamp(-{fp8_bound}, {fp8_bound})",
                "return torch.matmul((fake_fp8 * scale).reshape_as(x), weight)",
            ]
        else:
            inv.append(
                _invariant(
                    "integer_range",
                    "quantized values are clipped to [-bound,bound]",
                    [-c["quant_bound"], c["quant_bound"]],
                )
            )
            labels.update(
                {
                    "dynamic_bound": c["quant_bound"],
                    "topology": "tensor_absmax>dynamic_round_clip_qdq>dense_projection",
                }
            )
            lines = [
                f"scale = x.abs().amax().clamp_min(1e-5) / {c['quant_bound']}.0",
                f"q = torch.round(x / scale).clamp(-{c['quant_bound']}, {c['quant_bound']})",
                "return torch.matmul(q * scale, weight)",
            ]
        code = _code("self, x, weight", lines, [f"return [torch.randn({rows}, {width}), torch.randn({width}, 6)]"])
        ops = [
            _op("qdq_scale", ["torch.round"], [("aten::round", "")]),
            _op("qdq_projection", ["torch.matmul"], [("aten::mm", "")]),
        ]
        if template_id == "QD02":
            ops.append(
                _op(
                    "packed_int4_bitwise",
                    ["torch.bitwise_left_shift", "torch.bitwise_or", "torch.bitwise_right_shift", "torch.bitwise_and"],
                    [
                        ("aten::bitwise_left_shift", "Tensor_Scalar"),
                        ("aten::bitwise_or", "Tensor"),
                        ("aten::bitwise_right_shift", "Tensor_Scalar"),
                        ("aten::bitwise_and", "Scalar"),
                    ],
                )
            )
    elif template_id.startswith("VG"):
        h, w = c["height"], c["width"]
        inv.append(_invariant("spatial_shape", "height,width are positive", [h, w]))
        labels.update({"height": h, "width": w, "modality": "vision"})
        if template_id == "VG01":
            shifts = {"roll_plus_1": 1, "roll_minus_1": -1, "roll_plus_2": 2, "roll_plus_3": 3}[c["reorder_policy"]]
            labels.update(
                {
                    "window_reorder": c["reorder_policy"],
                    "topology": "image_unfold>window_permute>window_roll_reorder>projection",
                }
            )
            lines = [
                "patch = x.unfold(2, 2, 2).unfold(3, 2, 2)",
                "window = patch.permute(0, 2, 3, 1, 4, 5).reshape(x.shape[0], -1, x.shape[1] * 4)",
                f"reordered = torch.roll(window, shifts={shifts}, dims=1)",
                "return reordered @ proj",
            ]
            inputs = [f"return [torch.randn(2, 3, {h // 2 * 2}, {w // 2 * 2}), torch.randn(12, 6)]"]
        elif template_id == "VG02":
            grid_scale = c["grid_scale"]
            inv.append(
                _invariant(
                    "geometry_domain",
                    "finite normalized grid; padding policy handles coordinates outside [-1,1]",
                    [-grid_scale, grid_scale],
                )
            )
            labels.update(
                {
                    "padding_mode": c["padding_policy"],
                    "interpolation_mode": c["interpolation_mode"],
                    "grid_scale": grid_scale,
                    "align_corners": c["align_corners"],
                    "topology": "normalized_grid>meshgrid>scaled_grid>grid_sample",
                }
            )
            lines = [
                "grid = torch.linspace(-1.0, 1.0, steps=4, device=x.device, dtype=x.dtype)",
                "yy, xx = torch.meshgrid(grid, grid, indexing='ij')",
                f"sample_grid = (torch.stack([xx, yy], dim=-1) * {grid_scale}).unsqueeze(0).expand(x.shape[0], -1, -1, -1)",
                f"return F.grid_sample(x, sample_grid, mode='{c['interpolation_mode']}', padding_mode='{c['padding_policy']}', align_corners={c['align_corners']})",
            ]
            inputs = [f"return [torch.randn(2, 3, {h}, {w})]"]
        else:
            channel_shift = {"roll_plus_1": 1, "roll_minus_1": -1, "roll_plus_2": 2, "roll_plus_3": 3}[
                c["reorder_policy"]
            ]
            labels.update(
                {
                    "packed_channel_reorder": c["reorder_policy"],
                    "topology": "pixel_unshuffle>packed_channel_roll>channel_mix>pixel_shuffle",
                }
            )
            lines = [
                "packed = F.pixel_unshuffle(x, 2)",
                f"reordered = torch.roll(packed, shifts={channel_shift}, dims=1)",
                "mixed = torch.einsum('bchw,cd->bdhw', reordered, weight)",
                "restored = F.pixel_shuffle(mixed, 2)",
                "return restored",
            ]
            inputs = [f"return [torch.randn(2, 3, {h // 2 * 2}, {w // 2 * 2}), torch.randn(12, 12)]"]
        code = _code(
            "self, x, proj" if template_id == "VG01" else "self, x" if template_id == "VG02" else "self, x, weight",
            lines,
            inputs,
        )
        geometry_identities = (
            [("aten::grid_sampler_2d", ""), ("aten::cudnn_grid_sampler", "")]
            if template_id == "VG02"
            else [("aten::unfold", "")] if template_id == "VG01" else [("aten::pixel_unshuffle", "")]
        )
        ops = [
            _op(
                "vision_geometry",
                [
                    (
                        "F.grid_sample"
                        if template_id == "VG02"
                        else "x.unfold" if template_id == "VG01" else "F.pixel_unshuffle"
                    )
                ],
                geometry_identities,
            )
        ]
    elif template_id.startswith("SF"):
        length, width = c["length"], c["width"]
        inv.append(_invariant("spectral_length", "length is even and >=16", length))
        labels.update({"length": length, "width": width, "modality": "spectral"})
        if template_id == "SF01":
            labels.update({"filter_gain": c["filter_gain"], "topology": "rfft>complex_filter_gain>irfft"})
            lines = [
                "freq = torch.fft.rfft(x, dim=-1)",
                f"filtered = freq * kernel * {c['filter_gain']}",
                "return torch.fft.irfft(filtered, n=x.shape[-1], dim=-1)",
            ]
            inputs = [
                f"return [torch.randn(2, {width}, {length}), torch.randn(1, {width}, {length // 2 + 1}, dtype=torch.complex64)]"
            ]
            signature = "self, x, kernel"
        elif template_id == "SF02":
            labels.update(
                {
                    "stft_hop": c["hop"],
                    "window_gain": c["filter_gain"],
                    "topology": "manual_frame_unfold>windowed_rfft>magnitude_projection",
                }
            )
            lines = [
                f"window = torch.hann_window(8, device=x.device, dtype=x.dtype) * {c['filter_gain']}",
                f"frames = x.unfold(-1, 8, {c['hop']})",
                "spec = torch.fft.rfft(frames * window, dim=-1)",
                "return torch.matmul(spec.abs(), proj)",
            ]
            inputs = [f"return [torch.randn({width}, {length}), torch.randn(5, 6)]"]
            signature = "self, x, proj"
        else:
            inv.append(_invariant("conditioning", "diagonal jitter >0", c["jitter"]))
            labels.update(
                {"diagonal_jitter": c["jitter"], "topology": "gram_matmul>jittered_spd>cholesky>triangular_solve"}
            )
            lines = [
                f"gram = x.transpose(-1, -2) @ x + {c['jitter']} * torch.eye(x.shape[-1], device=x.device, dtype=x.dtype)",
                "chol = torch.linalg.cholesky(gram)",
                "return torch.cholesky_solve(rhs, chol)",
            ]
            inputs = [f"return [torch.randn({length // 2}, {width}), torch.randn({width}, 3)]"]
            signature = "self, x, rhs"
        code = _code(signature, lines, inputs)
        ops = [
            _op(
                "spectral_or_solve",
                ["torch.fft.rfft" if template_id in {"SF01", "SF02"} else "torch.linalg.cholesky"],
                [("aten::_fft_r2c", "") if template_id in {"SF01", "SF02"} else ("aten::linalg_cholesky_ex", "")],
            )
        ]
    else:
        items, width = c["items"], c["width"]
        inv += [
            _invariant(
                "retrieval_ids", "all ids are within embedding table", {"items": items, "table_rows": items + 4}
            )
        ]
        labels.update({"items": items, "width": width, "modality": "retrieval"})
        if template_id == "RR01":
            if c["bag_split_policy"] == "balanced":
                offsets = [0, items // 2, items]
            elif c["bag_split_policy"] == "front_loaded":
                offsets = [0, max(1, (2 * items) // 3), items]
            elif c["bag_split_policy"] == "tail_loaded":
                offsets = [0, max(1, items // 3), items]
            else:
                offsets = [0, max(1, items // 3), max(2, (2 * items) // 3), items]
            # All RR01 cells include intentional duplicate IDs; the split policy
            # changes which duplicate runs land in each bag.
            ids = [
                (
                    (index // 2) + c["id_offset"]
                    if c["bag_split_policy"] != "tail_loaded"
                    else (items - 1 - index) // 2 + c["id_offset"]
                )
                % (items + 4)
                for index in range(items)
            ]
            inv += [
                _invariant("ragged_offsets", "offsets begin at zero, are monotone, and end at ids length", offsets),
                _invariant("duplicate_ids", "at least one retrieval ID is repeated", len(set(ids)) < len(ids)),
            ]
            labels.update(
                {
                    "bag_split_policy": c["bag_split_policy"],
                    "bag_offsets": offsets,
                    "topology": "ragged_ids_offsets>embedding_bag>self_interaction_concat",
                }
            )
            code = _code(
                "self, table",
                [
                    f"ids = torch.tensor({ids}, device=table.device, dtype=torch.long)",
                    f"offsets = torch.tensor({offsets}, device=table.device, dtype=torch.long)",
                    "bag = F.embedding_bag(ids, table, offsets, include_last_offset=True, mode='sum')",
                    "return torch.cat([bag, bag * bag], dim=-1)",
                ],
                [f"return [torch.randn({items + 4}, {width})]"],
            )
            ops = [_op("retrieval_embedding_bag", ["F.embedding_bag"], [("aten::_embedding_bag_forward_only", "")])]
        else:
            topk = c["topk"]
            corpus_rows = items + 4
            inv.append(
                _invariant("retrieval_topk", "1<=topk<=corpus_rows", {"topk": topk, "corpus_rows": corpus_rows})
            )
            bias_values = {
                "flat": [0.0 for _ in range(corpus_rows)],
                "head_bias": [0.25 if index < corpus_rows // 2 else -0.05 for index in range(corpus_rows)],
                "tail_bias": [-0.05 if index < corpus_rows // 2 else 0.25 for index in range(corpus_rows)],
                "alternating": [0.15 if index % 2 else -0.1 for index in range(corpus_rows)],
            }[c["corpus_skew"]]
            labels.update(
                {
                    "retrieval_topk": topk,
                    "corpus_skew": c["corpus_skew"],
                    "topology": "query_corpus_matmul+bias>topk>corpus_gather>softmax_weighted_reduce",
                }
            )
            code = _code(
                "self, query, corpus",
                [
                    "score = query @ corpus.transpose(-1, -2)",
                    f"corpus_bias = torch.tensor({bias_values}, device=query.device, dtype=query.dtype)",
                    "score = score + corpus_bias",
                    f"weight, ids = torch.topk(score, k={topk}, dim=-1)",
                    f"gathered = corpus.index_select(0, ids.reshape(-1)).reshape(query.shape[0], {topk}, corpus.shape[1])",
                    "return (torch.softmax(weight, dim=-1).unsqueeze(-1) * gathered).sum(dim=1)",
                ],
                [f"return [torch.randn({items // 2}, {width}), torch.randn({corpus_rows}, {width})]"],
            )
            ops = [
                _op("retrieval_topk", ["torch.topk"], [("aten::topk", "")]),
                _op("retrieval_gather", ["corpus.index_select"], [("aten::index_select", "")]),
            ]
    labels.setdefault("sparse_format", labels.get("sparse_layout", "none"))
    labels.setdefault(
        "state",
        (
            "state_carry"
            if template_id.startswith("SS")
            else "cache_addressed" if template_id in {"AC03", "AC04"} else "none"
        ),
    )
    labels.setdefault(
        "precision",
        (
            "fake_int8_int4_fp8"
            if template_id.startswith("QD")
            else (
                "fp32_complex_intermediate"
                if template_id in {"SF01", "SF02"}
                else (
                    "fp16_sparse_semi_structured"
                    if template_id == "SP04" and c["sparse_mode"] == "semi_structured_2to4"
                    else "fp32"
                )
            )
        ),
    )
    required_axes = {"sparse_format", "topology", "state", "modality", "precision"}
    if required_axes - labels.keys():
        raise ValueError(f"coverage axes missing:{template_id}:{sorted(required_axes - labels.keys())}")
    requested = dict(c)
    realized = dict(c)
    realized.update(labels)
    if code.count("        return ") != 1:
        raise ValueError(f"expected exactly one forward return:{template_id}:{variant}")
    return code, ops, labels, _constraint(template_id, variant, requested, realized, inv)


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
        tags = frozenset().union(*(_expr_tags(item, environment, call_to_ids) for item in inputs))
        if name in _ALLOCATION_CALLS:
            tags = frozenset()
        return tags | call_to_ids.get(name, frozenset())
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
        raise ValueError(f"unsupported assignment target:{ast.dump(target, include_attributes=False)}")


def _static_contract(code: str, declared_ops: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    tree = ast.parse(code)
    model = next((node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Model"), None)
    forward = (
        next((node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "forward"), None)
        if model
        else None
    )
    if forward is None:
        raise ValueError("missing Model.forward")
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
        raise ValueError("forward is not straight-line")
    calls = collections.Counter(_call_name(node.func) for node in ast.walk(forward) if isinstance(node, ast.Call))
    if any(name and (name.endswith("_") or ".__" in name) for name in calls):
        raise ValueError(f"in-place or dunder call:{sorted(calls)}")
    call_to_ids: dict[str, set[str]] = collections.defaultdict(set)
    for item in declared_ops:
        for name in item["source_calls"]:
            call_to_ids[name].add(str(item["op_id"]))
        if not any(calls.get(name, 0) for name in item["source_calls"]):
            raise ValueError(f"declared source op absent:{item['op_id']}:{item['source_calls']}")
    frozen = {key: frozenset(value) for key, value in call_to_ids.items()}
    environment: dict[str, frozenset[str]] = {}
    returned: list[frozenset[str]] = []
    for statement in forward.body:
        if isinstance(statement, ast.Assign):
            tags = _expr_tags(statement.value, environment, frozen)
            for target in statement.targets:
                _bind_target(target, tags, environment)
        elif isinstance(statement, ast.Expr):
            _expr_tags(statement.value, environment, frozen)
        elif isinstance(statement, ast.Return):
            if statement.value is None or isinstance(statement.value, (ast.Tuple, ast.List, ast.Dict, ast.Set)):
                raise ValueError("final return is not single valued")
            returned.append(_expr_tags(statement.value, environment, frozen))
        else:
            raise ValueError(f"unsupported forward statement:{type(statement).__name__}")
    if len(returned) != 1:
        raise ValueError(f"forward return count:{len(returned)}")
    expected = {str(item["op_id"]) for item in declared_ops}
    missing = expected - set(returned[0])
    if missing:
        raise ValueError(f"declared ops do not reach return:{sorted(missing)}")
    return {
        "contract_version": STATIC_CONTRACT_VERSION,
        "forward_statement_count": len(forward.body),
        "source_call_histogram": {str(key): value for key, value in sorted(calls.items()) if key},
        "returned_declared_op_ids": sorted(returned[0]),
        "single_value_return": True,
        "straight_line": True,
    }


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
        if not path.is_file():
            raise FileNotFoundError(path)
        parquet = pq.ParquetFile(path)
        if "reward_model" not in parquet.schema_arrow.names:
            raise ValueError(f"comparison lacks reward_model:{path}")
        parse_failures = 0
        for batch in parquet.iter_batches(batch_size=256, columns=["reward_model"]):
            for item in batch.to_pylist():
                code = (
                    item["reward_model"].get("ground_truth") if isinstance(item.get("reward_model"), Mapping) else None
                )
                if not isinstance(code, str) or not code.strip():
                    raise ValueError(f"empty comparison reference:{path}")
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
    prefix_hash = _sha256_bytes(prompt_prefix.encode())
    for spec in TEMPLATES:
        semantic_coordinate_cells: set[str] = set()
        source_url, source_kind, note = SOURCES[spec.source_key]
        scenario_source = {
            "source_url": source_url,
            "source_kind": source_kind,
            "implementation_note": note,
            "source_registry_version": REGISTRY_VERSION,
        }
        for variant in range(spec.variants):
            code, declared_ops, labels, constraint = _render(spec.template_id, variant)
            semantic_coordinates = dict(constraint["requested_coordinates"])
            semantic_coordinates.pop("variant", None)
            coordinate_cell = _canonical_json(semantic_coordinates)
            if coordinate_cell in semantic_coordinate_cells:
                raise ValueError(f"duplicate semantic coordinate cell:{spec.template_id}:{variant}:{coordinate_cell}")
            semantic_coordinate_cells.add(coordinate_cell)
            static = _static_contract(code, declared_ops)
            ref_hash, ast_hash = _sha256_bytes(code.encode()), _normalized_ast_sha256(code)
            if ref_hash in comparison_reference_hashes or ast_hash in comparison_ast_hashes:
                raise ValueError(f"decontamination collision:{spec.template_id}:{variant}")
            uuid = "frontier_" + _sha256_bytes(f"{CONTRACT_VERSION}|{spec.template_id}|{variant}".encode())[:24]
            content, prompt = prompt_prefix + code, [{"content": prompt_prefix + code, "role": "user"}]
            row = {
                "data_source": DATA_SOURCE,
                "prompt": prompt,
                "ability": "kernel_optimization",
                "reward_model": {"ground_truth": code, "style": "rule"},
                "extra_info": {
                    "entry_point": "Model",
                    "level": "coverage",
                    "module_name": "Model",
                    "ops": json.dumps(sorted({call for op in declared_ops for call in op["source_calls"]})),
                    "original_prompt": prompt,
                    "repo_name": "frontier_operator_generator_v3",
                    "type": "frontier_operator_synthetic",
                    "uuid": uuid,
                },
            }
            manifest = {
                "manifest_contract_version": MANIFEST_VERSION,
                "generator_contract_version": CONTRACT_VERSION,
                "registry_version": REGISTRY_VERSION,
                "order_version": ORDER_VERSION,
                "static_contract_version": STATIC_CONTRACT_VERSION,
                "constraint_contract_version": CONSTRAINT_CONTRACT_VERSION,
                "runtime_contract_version": RUNTIME_CONTRACT_VERSION,
                "uuid": uuid,
                "template_id": spec.template_id,
                "template_variant": variant,
                "primary_family": spec.family,
                "scenario_name": spec.scenario_name,
                "mode_behavior": spec.mode_behavior,
                "lineage_kind": "standalone_frontier_synthetic",
                "parent_uuid": None,
                "primary_intervention": "frontier_operator_scenario",
                "final_output_contract": {"kind": "single_dense_tensor", "finite_required": True},
                "training_mode": True,
                "declared_ops": declared_ops,
                "static_proof": static,
                "constraint_solver": constraint,
                "scenario_source": scenario_source,
                "coverage_labels": labels,
                "reference_sha256": ref_hash,
                "normalized_ast_sha256": ast_hash,
                "prompt_sha256": _sha256_bytes(content.encode()),
                "row_payload_sha256": _canonical_sha256(row),
                "prompt_template": dict(template_binding),
                "prompt_prefix_sha256": prefix_hash,
                "decontamination_roots": [dict(item) for item in comparison_bindings],
                "decontamination_status": "exact_reference_all_and_normalized_ast_parseable_no_match",
                "provenance": {
                    "source_family": "project_generated_frontier_review",
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
                "structured_output_defer_reason": "KernelGym authority final comparison is Tensor-only",
            }
            pending.append((_sha256_bytes(f"{ORDER_VERSION}|{spec.template_id}|{variant}".encode()), row, manifest))
        if len(semantic_coordinate_cells) != spec.variants:
            raise ValueError(f"semantic coordinate quota mismatch:{spec.template_id}:{len(semantic_coordinate_cells)}")
    pending.sort(key=lambda item: item[0])
    rows, manifests = [], []
    for index, (_, row, manifest) in enumerate(pending):
        manifest["candidate_row_index"] = index
        rows.append(row)
        manifests.append(manifest)
    return rows, manifests


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(_canonical_json(record) + "\n")


def _review_markdown(rows: Sequence[Mapping[str, Any]], manifests: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Frontier operator static canary samples",
        "",
        "The 1,000 rows are deterministic, parentless, review-only, and have one finite dense Tensor as final output.",
        "",
    ]
    for spec in TEMPLATES:
        row, manifest = next(
            (row, manifest)
            for row, manifest in zip(rows, manifests, strict=True)
            if manifest["template_id"] == spec.template_id
        )
        lines += [
            f"## {spec.template_id}: {spec.scenario_name}",
            "",
            f"UUID: `{manifest['uuid']}`",
            "",
            "Coordinates:",
            "",
            "```json",
            json.dumps(manifest["constraint_solver"]["realized_coordinates"], sort_keys=True),
            "```",
            "",
            "```python",
            row["reward_model"]["ground_truth"].rstrip(),
            "```",
            "",
        ]
    return "\n".join(lines)


def generate(output_dir: Path, template_path: Path, comparison_paths: Sequence[Path]) -> dict[str, Any]:
    output_dir, template_path = output_dir.expanduser().resolve(), template_path.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty:{output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _, prefix = _template(template_path)
    template_file = pq.ParquetFile(template_path)
    template_binding = {
        "path": str(template_path),
        "sha256": _sha256_file(template_path),
        "rows": template_file.metadata.num_rows,
    }
    ref_hashes, ast_hashes, comparison_bindings = _comparison_hashes(comparison_paths)
    source_hash = _sha256_file(Path(__file__).resolve())
    rows, manifests = build_records(
        prompt_prefix=prefix,
        comparison_reference_hashes=ref_hashes,
        comparison_ast_hashes=ast_hashes,
        template_binding=template_binding,
        comparison_bindings=comparison_bindings,
        git_commit=_git_commit(),
        generator_sha256=source_hash,
    )
    family_counts, template_counts = collections.Counter(
        item["primary_family"] for item in manifests
    ), collections.Counter(item["template_id"] for item in manifests)
    if len(rows) != EXACT_CANARY_ROWS or len(rows) > MAX_AUTHORIZED_CANDIDATES or dict(family_counts) != FAMILY_QUOTAS:
        raise ValueError(f"exact frontier quota mismatch: rows={len(rows)} families={dict(family_counts)}")
    if dict(template_counts) != {spec.template_id: spec.variants for spec in TEMPLATES}:
        raise ValueError("template quota mismatch")
    for field in ("uuid", "reference_sha256", "normalized_ast_sha256", "row_payload_sha256"):
        values = [item[field] for item in manifests]
        if len(values) != len(set(values)):
            raise ValueError(f"nonunique generated field:{field}")
    candidates, manifest, decisions, review, summary = (
        output_dir / name
        for name in ("candidates.parquet", "manifest.jsonl", "decisions.jsonl", "review_samples.md", "summary.json")
    )
    pq.write_table(pa.Table.from_pylist(rows, schema=_schema()), candidates, compression="zstd")
    _write_jsonl(manifest, manifests)
    _write_jsonl(
        decisions,
        [
            {
                "decision": "generation_method",
                "selected": "deterministic_constructive_solver",
                "solver_backend": "deterministic_constructive_arithmetic",
                "llm_generation_used": False,
                "reason": "the closed scenario matrix admits direct invariant-checked construction",
            },
            {
                "decision": "authority_boundary",
                "selected": "single_dense_tensor_final",
                "structured_output_status": "explicitly_deferred",
                "reason": "intermediates may be sparse, tuple, stateful, or complex; final authority remains Tensor-only",
            },
        ],
    )
    review.write_text(_review_markdown(rows, manifests), encoding="utf-8")
    result = {
        "contract_version": CONTRACT_VERSION,
        "constraint_solver_contract_version": CONSTRAINT_CONTRACT_VERSION,
        "registry_version": REGISTRY_VERSION,
        "rows": len(rows),
        "unique_semantic_coordinate_cells": len(rows),
        "semantic_coordinate_identity_excludes": ["variant"],
        "family_counts_exclusive_primary": dict(sorted(family_counts.items())),
        "template_counts": dict(sorted(template_counts.items())),
        "final_output_kind": "single_dense_tensor",
        "training_approved": False,
        "artifacts": {
            name: {"path": str(path), "sha256": _sha256_file(path)}
            for name, path in {
                "candidates": candidates,
                "manifest": manifest,
                "decisions": decisions,
                "review_samples": review,
            }.items()
        },
        "prompt_template": template_binding,
        "decontamination_roots": comparison_bindings,
        "generator_source_sha256": source_hash,
    }
    _write_json(summary, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--template-parquet", type=Path, default=_DEFAULT_TEMPLATE)
    parser.add_argument("--comparison-artifact", type=Path, action="append", dest="comparison_artifacts")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    comparisons = tuple(args.comparison_artifacts) if args.comparison_artifacts else _DEFAULT_COMPARISON_ROOTS
    print(
        json.dumps(generate(args.output_dir, args.template_parquet, comparisons), ensure_ascii=False, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
