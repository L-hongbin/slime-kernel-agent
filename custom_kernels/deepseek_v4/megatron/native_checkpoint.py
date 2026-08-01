"""R2 checkpoint utilities for DeepSeek-V4-Flash FP8 -> Megatron preparation.

The real checkpoint at ``sgl-project/DeepSeek-V4-Flash-FP8`` is a native/HF-style
sharded safetensors checkpoint.  Megatron training cannot consume that path
directly; it needs a Megatron checkpoint, normally ``torch_dist``.  This module is
the narrow conversion/audit layer between the native checkpoint names and the
current TP=PP=EP=1 V4 mcore state-dict names.

This is deliberately streaming-friendly for inspection: it reads individual
tensors with ``safe_open`` and never loads a full safetensors shard.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass

import torch
from safetensors import safe_open

DEFAULT_V4_FLASH_FP8_CKPT = "/nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8"
EXPERT_RE = re.compile(r"^layers\.(\d+)\.ffn\.experts\.(\d+)\.w([123])\.weight$")
# The MTP head's routed experts (``mtp.<d>.ffn.experts.<e>.w{1,2,3}.weight``). Only
# consumed when MTP conversion is explicitly enabled; otherwise mtp.* is ignored.
MTP_EXPERT_RE = re.compile(r"^mtp\.(\d+)\.ffn\.experts\.(\d+)\.w([123])\.weight$")

# Attention/FFN leaf maps shared by the layer and MTP mappings (kept in one place so
# the two subtrees can never drift).
_ATTN_LEAF = {
    "wq_a": "q_a_proj",
    "wq_b": "q_b_proj",
    "wkv": "kv_proj",
    "wo_a": "o_a_proj",
    "wo_b": "o_b_proj",
    "q_norm": "q_a_norm",
    "kv_norm": "kv_norm",
}
_SHARED_EXPERT_LEAF = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}


def native_mtp_key_to_mcore(key: str) -> str | None:
    """Map one non-expert native ``mtp.<d>.*`` key to the V4 mcore ``mtp.*`` subtree.

    Destination is ``V4LanguageModel.mtp`` (a single ``V4MultiTokenPredictionLayer``),
    so the MTP layer index ``<d>`` is dropped (only one MTP head is built).  Expert
    weights and ``*.scale`` tensors are handled elsewhere / consumed by FP8 dequant.
    Returns None for anything that is not an MTP non-expert weight.
    """
    m = re.fullmatch(r"mtp\.(\d+)\.(.+)", key)
    if not m:
        return None
    rest = m.group(2)
    if rest.endswith(".scale") or MTP_EXPERT_RE.match(key):
        return None

    # MTP-local (not inside the transformer layer): the split projections, the two
    # fusion norms, the final norm, and the head-side hc collapse.
    mtp_local = {
        "enorm.weight": "enorm.weight",
        "hnorm.weight": "hnorm.weight",
        "e_proj.weight": "e_proj.weight",
        "h_proj.weight": "h_proj.weight",
        "norm.weight": "norm.weight",
        "hc_head_fn": "hc_head.hc_fn",
        "hc_head_base": "hc_head.hc_base",
        "hc_head_scale": "hc_head.hc_scale",
    }
    if rest in mtp_local:
        return f"mtp.{mtp_local[rest]}"

    # Everything else lives inside the single V4 decoder layer (transformer_layer.*).
    tl = "mtp.transformer_layer."
    if rest == "attn_norm.weight":
        return f"{tl}input_layernorm.weight"
    if rest == "ffn_norm.weight":
        return f"{tl}post_attention_layernorm.weight"
    mm = re.fullmatch(r"hc_attn_(fn|base|scale)", rest)
    if mm:
        return f"{tl}attn_hc.{mm.group(1)}"
    mm = re.fullmatch(r"hc_ffn_(fn|base|scale)", rest)
    if mm:
        return f"{tl}ffn_hc.{mm.group(1)}"
    if rest == "attn.attn_sink":
        return f"{tl}self_attn.sinks"
    mm = re.fullmatch(r"attn\.([^.]+)\.weight", rest)
    if mm and mm.group(1) in _ATTN_LEAF:
        return f"{tl}self_attn.{_ATTN_LEAF[mm.group(1)]}.weight"
    if rest == "ffn.gate.weight":
        return f"{tl}mlp.gate.weight"
    if rest == "ffn.gate.bias":
        return f"{tl}mlp.gate.e_score_correction_bias"
    mm = re.fullmatch(r"ffn\.shared_experts\.(w[123])\.weight", rest)
    if mm:
        return f"{tl}mlp.shared_experts.{_SHARED_EXPERT_LEAF[mm.group(1)]}.weight"
    return None


@dataclass(frozen=True)
class TensorAudit:
    native_key: str
    mcore_key: str | None
    shape: tuple[int, ...]
    dtype: str
    scale_key: str | None = None
    scale_shape: tuple[int, ...] | None = None
    converted_shape: tuple[int, ...] | None = None
    converted_dtype: str | None = None


@dataclass(frozen=True)
class TensorMetadata:
    key: str
    shape: tuple[int, ...]
    dtype: str
    nbytes: int


@dataclass(frozen=True)
class CheckpointAudit:
    checkpoint: str
    indexed_keys: int
    actual_keys: int
    shard_count: int
    stale_index_keys: tuple[str, ...]
    ignored_mtp_keys: int
    ignored_scale_keys: int
    direct_mapped_keys: int
    expert_weight_keys: int
    unmapped_non_mtp_keys: tuple[str, ...]
    tensor_audits: tuple[TensorAudit, ...]
    expert_audits: tuple[TensorAudit, ...]


@dataclass(frozen=True)
class CheckpointSizeEstimate:
    checkpoint: str
    actual_payload_bytes: int
    non_mtp_payload_bytes: int
    ignored_mtp_payload_bytes: int
    ignored_scale_payload_bytes: int
    mapped_direct_tensors: int
    expert_source_tensors: int
    unmapped_non_mtp_tensors: int
    converted_mcore_payload_bytes: int
    converted_mcore_payload_gib: float
    native_actual_payload_gib: float
    native_non_mtp_payload_gib: float
    filesystem_free_gib: float
    output_dtype: str


@dataclass(frozen=True)
class ConversionPlanRankEstimate:
    pp_rank: int
    ep_rank: int
    layers: tuple[int, ...]
    direct_payload_bytes: int
    expert_payload_bytes: int
    total_payload_bytes: int
    total_payload_gib: float


@dataclass(frozen=True)
class ConversionPlanEstimate:
    checkpoint: str
    pp_size: int
    ep_size: int
    layer_splits: tuple[tuple[int, ...], ...]
    rank_estimates: tuple[ConversionPlanRankEstimate, ...]
    max_rank_payload_gib: float
    total_unique_payload_gib: float
    output_dtype: str


class NativeV4Checkpoint:
    """Lightweight reader for the native V4-Flash safetensors checkpoint."""

    def __init__(self, checkpoint_dir: str = DEFAULT_V4_FLASH_FP8_CKPT):
        self.checkpoint_dir = checkpoint_dir
        index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
        with open(index_path, encoding="utf-8") as f:
            self.weight_map: dict[str, str] = json.load(f)["weight_map"]
        self._actual_key_to_file: dict[str, str] | None = None
        self._tensor_metadata: dict[str, TensorMetadata] | None = None

    @property
    def shard_files(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.weight_map.values())))

    def actual_key_to_file(self) -> dict[str, str]:
        """Return actual safetensor contents, not just the sometimes-stale index."""
        if self._actual_key_to_file is None:
            actual = {}
            for shard in self.shard_files:
                path = os.path.join(self.checkpoint_dir, shard)
                with safe_open(path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        actual[key] = shard
            self._actual_key_to_file = actual
        return self._actual_key_to_file

    def tensor_metadata(self) -> dict[str, TensorMetadata]:
        """Return actual tensor shape/dtype/bytes from safetensors headers only."""
        if self._tensor_metadata is None:
            metadata = {}
            for shard in self.shard_files:
                path = os.path.join(self.checkpoint_dir, shard)
                with safe_open(path, framework="pt", device="cpu") as f:
                    for key in f.keys():
                        tensor_slice = f.get_slice(key)
                        shape = tuple(tensor_slice.get_shape())
                        dtype = tensor_slice.get_dtype()
                        metadata[key] = TensorMetadata(
                            key=key,
                            shape=shape,
                            dtype=dtype,
                            nbytes=numel(shape) * dtype_nbytes(dtype),
                        )
            self._tensor_metadata = metadata
        return self._tensor_metadata

    def stale_index_keys(self) -> tuple[str, ...]:
        actual = self.actual_key_to_file()
        return tuple(sorted(k for k in self.weight_map if k not in actual))

    def has_tensor(self, key: str) -> bool:
        return key in self.actual_key_to_file()

    def get_tensor(self, key: str) -> torch.Tensor:
        actual = self.actual_key_to_file()
        if key not in actual:
            raise KeyError(f"{key} is not present in actual safetensors contents")
        path = os.path.join(self.checkpoint_dir, actual[key])
        with safe_open(path, framework="pt", device="cpu") as f:
            return f.get_tensor(key)


def _is_float8_tensor(tensor: torch.Tensor) -> bool:
    return str(tensor.dtype).startswith("torch.float8_")


def numel(shape: Iterable[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total


def dtype_nbytes(dtype: str | torch.dtype) -> int:
    dtype_name = str(dtype)
    if dtype_name.startswith("torch."):
        dtype_name = dtype_name.removeprefix("torch.").upper()
    aliases = {
        "BOOL": 1,
        "U8": 1,
        "I8": 1,
        "UINT8": 1,
        "INT8": 1,
        "F8_E4M3": 1,
        "F8_E4M3FN": 1,
        "F8_E5M2": 1,
        "FLOAT8_E4M3FN": 1,
        "FLOAT8_E5M2": 1,
        "I16": 2,
        "INT16": 2,
        "F16": 2,
        "FLOAT16": 2,
        "BF16": 2,
        "BFLOAT16": 2,
        "I32": 4,
        "INT32": 4,
        "F32": 4,
        "FLOAT32": 4,
        "I64": 8,
        "INT64": 8,
        "F64": 8,
        "FLOAT64": 8,
    }
    if dtype_name not in aliases:
        raise KeyError(f"unsupported dtype for size estimate: {dtype}")
    return aliases[dtype_name]


def is_floating_dtype_name(dtype: str) -> bool:
    return dtype.startswith("F") or dtype in {"BF16", "FLOAT16", "FLOAT32", "FLOAT64", "BFLOAT16"}


def fp8_scale_key(weight_key: str) -> str:
    if not weight_key.endswith(".weight"):
        raise ValueError(f"FP8 dequant expects a *.weight key, got {weight_key}")
    return weight_key[: -len(".weight")] + ".scale"


def dequant_fp8_block(
    quantized: torch.Tensor,
    scales: torch.Tensor,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize a V4 per-block FP8 matrix.

    Matches the current Transformers finegrained-FP8 dequant convention: infer the
    block size from ``weight.shape[-2:] / scale.shape[-2:]`` and multiply the FP8
    values by the corresponding per-block scale.
    """
    if quantized.ndim < 2 or scales.ndim < 2:
        raise ValueError(f"expected >=2D weight/scale, got {quantized.shape=} {scales.shape=}")
    quantized_fp32 = quantized.to(torch.float32)
    rows, cols = quantized_fp32.shape[-2:]
    scale_rows, scale_cols = scales.shape[-2:]
    if rows % scale_rows or cols % scale_cols:
        raise ValueError(
            f"Weight shape ({rows}, {cols}) is not divisible by scale grid " f"({scale_rows}, {scale_cols})"
        )
    block_m = rows // scale_rows
    block_n = cols // scale_cols
    original_shape = quantized_fp32.shape
    q = quantized_fp32.reshape(-1, scale_rows, block_m, scale_cols, block_n)
    s = scales.to(torch.float32).reshape(-1, scale_rows, scale_cols).unsqueeze(-1).unsqueeze(2)
    return (q * s).reshape(original_shape).to(output_dtype)


def read_native_weight(
    checkpoint: NativeV4Checkpoint,
    native_key: str,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, str | None, torch.Tensor | None]:
    """Read one native weight, dequantizing FP8 ``*.weight`` tensors when needed."""
    tensor = checkpoint.get_tensor(native_key)
    if not _is_float8_tensor(tensor):
        return tensor, None, None
    scale_key = fp8_scale_key(native_key)
    if not checkpoint.has_tensor(scale_key):
        raise KeyError(f"FP8 tensor {native_key} is missing required scale tensor {scale_key}")
    scales = checkpoint.get_tensor(scale_key)
    return dequant_fp8_block(tensor, scales, output_dtype=output_dtype), scale_key, scales


def native_key_to_mcore(key: str, *, include_mtp: bool = False) -> str | None:
    """Map one non-expert native checkpoint key to the current V4 mcore key.

    Scale tensors are consumed by FP8 dequant and intentionally do not map to mcore.
    MTP tensors are ignored by default (``include_mtp=False``) so existing
    conversions stay byte-identical; pass ``include_mtp=True`` (only when the mcore
    model was built with the MTP head) to map the ``mtp.*`` subtree.
    """
    if key.startswith("mtp."):
        return native_mtp_key_to_mcore(key) if include_mtp else None
    if key.endswith(".scale") or EXPERT_RE.match(key):
        return None
    if key == "embed.weight":
        return "embedding.word_embeddings.weight"
    if key == "head.weight":
        return "output_layer.weight"
    if key == "norm.weight":
        return "norm.weight"
    top = {
        "hc_head_fn": "hc_head.hc_fn",
        "hc_head_base": "hc_head.hc_base",
        "hc_head_scale": "hc_head.hc_scale",
    }
    if key in top:
        return top[key]

    m = re.fullmatch(r"layers\.(\d+)\.hc_attn_(fn|base|scale)", key)
    if m:
        return f"layers.{m.group(1)}.attn_hc.{m.group(2)}"
    m = re.fullmatch(r"layers\.(\d+)\.hc_ffn_(fn|base|scale)", key)
    if m:
        return f"layers.{m.group(1)}.ffn_hc.{m.group(2)}"

    m = re.fullmatch(r"layers\.(\d+)\.attn_norm\.weight", key)
    if m:
        return f"layers.{m.group(1)}.input_layernorm.weight"
    m = re.fullmatch(r"layers\.(\d+)\.ffn_norm\.weight", key)
    if m:
        return f"layers.{m.group(1)}.post_attention_layernorm.weight"

    m = re.fullmatch(r"layers\.(\d+)\.attn\.attn_sink", key)
    if m:
        return f"layers.{m.group(1)}.self_attn.sinks"
    attn_leaf = {
        "wq_a": "q_a_proj",
        "wq_b": "q_b_proj",
        "wkv": "kv_proj",
        "wo_a": "o_a_proj",
        "wo_b": "o_b_proj",
        "q_norm": "q_a_norm",
        "kv_norm": "kv_norm",
    }
    m = re.fullmatch(r"layers\.(\d+)\.attn\.([^.]+)\.weight", key)
    if m and m.group(2) in attn_leaf:
        return f"layers.{m.group(1)}.self_attn.{attn_leaf[m.group(2)]}.weight"

    compressor_leaf = {
        "ape": "position_bias",
        "wkv.weight": "kv_proj.weight",
        "wgate.weight": "gate_proj.weight",
        "norm.weight": "kv_norm.weight",
    }
    m = re.fullmatch(r"layers\.(\d+)\.attn\.compressor\.(.+)", key)
    if m and m.group(2) in compressor_leaf:
        return f"layers.{m.group(1)}.self_attn.compressor.{compressor_leaf[m.group(2)]}"

    indexer_leaf = {
        "wq_b.weight": "q_b_proj.weight",
        "weights_proj.weight": "weights_proj.weight",
        "compressor.ape": "position_bias",
        "compressor.wkv.weight": "kv_proj.weight",
        "compressor.wgate.weight": "gate_proj.weight",
        "compressor.norm.weight": "kv_norm.weight",
    }
    m = re.fullmatch(r"layers\.(\d+)\.attn\.indexer\.(.+)", key)
    if m and m.group(2) in indexer_leaf:
        return f"layers.{m.group(1)}.self_attn.compressor.indexer.{indexer_leaf[m.group(2)]}"

    m = re.fullmatch(r"layers\.(\d+)\.ffn\.gate\.weight", key)
    if m:
        return f"layers.{m.group(1)}.mlp.gate.weight"
    m = re.fullmatch(r"layers\.(\d+)\.ffn\.gate\.tid2eid", key)
    if m:
        return f"layers.{m.group(1)}.mlp.gate.tid2eid"
    m = re.fullmatch(r"layers\.(\d+)\.ffn\.gate\.bias", key)
    if m:
        return f"layers.{m.group(1)}.mlp.gate.e_score_correction_bias"

    shared = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
    m = re.fullmatch(r"layers\.(\d+)\.ffn\.shared_experts\.(w[123])\.weight", key)
    if m:
        return f"layers.{m.group(1)}.mlp.shared_experts.{shared[m.group(2)]}.weight"

    return None


def native_key_to_hf(key: str) -> str | None:
    """Map one non-expert native checkpoint key to HF ``DeepseekV4ForCausalLM``.

    The native V4-Flash checkpoint is not a vanilla Transformers state dict: top
    level names are shortened, attention/FFN subtrees use the training-export
    names, and FP8 ``*.scale`` tensors are consumed during dequantization.  Expert
    weights still need the w1/w3 concatenation handled by ``read_expert_tensors``.
    """
    if key.startswith("mtp.") or key.endswith(".scale") or EXPERT_RE.match(key):
        return None
    if key == "embed.weight":
        return "model.embed_tokens.weight"
    if key == "head.weight":
        return "lm_head.weight"
    if key == "norm.weight":
        return "model.norm.weight"
    top = {
        "hc_head_fn": "model.hc_head.hc_fn",
        "hc_head_base": "model.hc_head.hc_base",
        "hc_head_scale": "model.hc_head.hc_scale",
    }
    if key in top:
        return top[key]

    m = re.fullmatch(r"layers\.(\d+)\.hc_attn_(fn|base|scale)", key)
    if m:
        return f"model.layers.{m.group(1)}.attn_hc.{m.group(2)}"
    m = re.fullmatch(r"layers\.(\d+)\.hc_ffn_(fn|base|scale)", key)
    if m:
        return f"model.layers.{m.group(1)}.ffn_hc.{m.group(2)}"

    m = re.fullmatch(r"layers\.(\d+)\.attn_norm\.weight", key)
    if m:
        return f"model.layers.{m.group(1)}.input_layernorm.weight"
    m = re.fullmatch(r"layers\.(\d+)\.ffn_norm\.weight", key)
    if m:
        return f"model.layers.{m.group(1)}.post_attention_layernorm.weight"

    m = re.fullmatch(r"layers\.(\d+)\.attn\.attn_sink", key)
    if m:
        return f"model.layers.{m.group(1)}.self_attn.sinks"
    attn_leaf = {
        "wq_a": "q_a_proj",
        "wq_b": "q_b_proj",
        "wkv": "kv_proj",
        "wo_a": "o_a_proj",
        "wo_b": "o_b_proj",
        "q_norm": "q_a_norm",
        "kv_norm": "kv_norm",
    }
    m = re.fullmatch(r"layers\.(\d+)\.attn\.([^.]+)\.weight", key)
    if m and m.group(2) in attn_leaf:
        return f"model.layers.{m.group(1)}.self_attn.{attn_leaf[m.group(2)]}.weight"

    compressor_leaf = {
        "ape": "position_bias",
        "wkv.weight": "kv_proj.weight",
        "wgate.weight": "gate_proj.weight",
        "norm.weight": "kv_norm.weight",
    }
    m = re.fullmatch(r"layers\.(\d+)\.attn\.compressor\.(.+)", key)
    if m and m.group(2) in compressor_leaf:
        return f"model.layers.{m.group(1)}.self_attn.compressor.{compressor_leaf[m.group(2)]}"

    indexer_leaf = {
        "wq_b.weight": "q_b_proj.weight",
        "weights_proj.weight": "weights_proj.weight",
        "compressor.ape": "position_bias",
        "compressor.wkv.weight": "kv_proj.weight",
        "compressor.wgate.weight": "gate_proj.weight",
        "compressor.norm.weight": "kv_norm.weight",
    }
    m = re.fullmatch(r"layers\.(\d+)\.attn\.indexer\.(.+)", key)
    if m and m.group(2) in indexer_leaf:
        return f"model.layers.{m.group(1)}.self_attn.compressor.indexer.{indexer_leaf[m.group(2)]}"

    m = re.fullmatch(r"layers\.(\d+)\.ffn\.gate\.weight", key)
    if m:
        return f"model.layers.{m.group(1)}.mlp.gate.weight"
    m = re.fullmatch(r"layers\.(\d+)\.ffn\.gate\.tid2eid", key)
    if m:
        return f"model.layers.{m.group(1)}.mlp.gate.tid2eid"
    m = re.fullmatch(r"layers\.(\d+)\.ffn\.gate\.bias", key)
    if m:
        return f"model.layers.{m.group(1)}.mlp.gate.e_score_correction_bias"

    shared = {"w1": "gate_proj", "w2": "down_proj", "w3": "up_proj"}
    m = re.fullmatch(r"layers\.(\d+)\.ffn\.shared_experts\.(w[123])\.weight", key)
    if m:
        return f"model.layers.{m.group(1)}.mlp.shared_experts.{shared[m.group(2)]}.weight"

    return None


def expert_native_keys(layer: int, expert: int, *, mtp: bool = False) -> tuple[str, str, str]:
    root = f"mtp.{layer}.ffn.experts.{expert}" if mtp else f"layers.{layer}.ffn.experts.{expert}"
    return (f"{root}.w1.weight", f"{root}.w3.weight", f"{root}.w2.weight")


def _is_packed_int8_tensor(tensor: torch.Tensor) -> bool:
    return tensor.dtype in (torch.int8, torch.uint8)


def read_expert_tensors(
    checkpoint: NativeV4Checkpoint,
    layer: int,
    expert: int,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
    mtp: bool = False,
) -> dict[str, torch.Tensor]:
    """Read one expert and assemble mcore's gate_up/down expert slices."""
    w1_key, w3_key, w2_key = expert_native_keys(layer, expert, mtp=mtp)
    root = f"mtp.{layer}.mlp.experts" if mtp else f"layers.{layer}.mlp.experts"
    probe = checkpoint.get_tensor(w1_key)
    if _is_packed_int8_tensor(probe):
        return read_expert_tensors_packed(checkpoint, layer, expert, mtp=mtp)
    w1, _, _ = read_native_weight(checkpoint, w1_key, output_dtype=output_dtype)
    w3, _, _ = read_native_weight(checkpoint, w3_key, output_dtype=output_dtype)
    w2, _, _ = read_native_weight(checkpoint, w2_key, output_dtype=output_dtype)
    return {
        f"{root}.gate_up_proj[{expert}]": torch.cat([w1, w3], dim=0),
        f"{root}.down_proj[{expert}]": w2,
    }


def read_expert_tensors_packed(
    checkpoint: NativeV4Checkpoint,
    layer: int,
    expert: int,
    *,
    mtp: bool = False,
) -> dict[str, torch.Tensor]:
    """Read one OFFICIAL packed-MXFP4 expert verbatim (no dequant).

    Official layout per matrix: weight I8 [O, K/2] (E2M1 nibble pairs, low nibble
    = even K element) + sibling ``.scale`` F8_E8M0 [O, K/32] (per-32 power-of-two
    scales). Both are carried as raw uint8 bytes; packing is along K (dim 1), so
    the mcore [w1;w3] concat along dim 0 is unaffected."""
    w1_key, w3_key, w2_key = expert_native_keys(layer, expert, mtp=mtp)
    root = f"mtp.{layer}.mlp.experts" if mtp else f"layers.{layer}.mlp.experts"

    def _packed_pair(weight_key: str) -> tuple[torch.Tensor, torch.Tensor]:
        w = checkpoint.get_tensor(weight_key)
        if not _is_packed_int8_tensor(w):
            raise ValueError(f"expected packed int8 expert weight for {weight_key}, got {w.dtype}")
        scale_key = fp8_scale_key(weight_key)
        if not checkpoint.has_tensor(scale_key):
            raise KeyError(f"packed MXFP4 tensor {weight_key} is missing scale tensor {scale_key}")
        sf = checkpoint.get_tensor(scale_key)
        if sf.dtype != torch.float8_e8m0fnu:
            raise ValueError(f"expected F8_E8M0 scales for {scale_key}, got {sf.dtype}")
        O, Kh = w.shape
        if sf.shape != (O, Kh * 2 // 32):
            raise ValueError(f"scale shape {tuple(sf.shape)} does not match packed weight {tuple(w.shape)}")
        return w.view(torch.uint8), sf.view(torch.uint8)

    w1, s1 = _packed_pair(w1_key)
    w3, s3 = _packed_pair(w3_key)
    w2, s2 = _packed_pair(w2_key)
    return {
        f"{root}.gate_up_proj_fp4[{expert}]": torch.cat([w1, w3], dim=0),
        f"{root}.gate_up_proj_sf[{expert}]": torch.cat([s1, s3], dim=0),
        f"{root}.down_proj_fp4[{expert}]": w2,
        f"{root}.down_proj_sf[{expert}]": s2,
    }


def _model_named_tensors(model) -> dict[str, torch.Tensor]:
    targets = {name: param for name, param in model.named_parameters()}
    targets.update({name: buf for name, buf in model.named_buffers()})
    return targets


def _local_expert_global_ids(expert_module, local_expert_count: int) -> tuple[int, ...]:
    """Return global expert ids stored in a local grouped-expert module."""
    start = int(getattr(expert_module, "local_expert_start", 0))
    end = int(getattr(expert_module, "local_expert_end", start + local_expert_count))
    if end - start != local_expert_count:
        raise ValueError(
            f"expert module local range [{start}, {end}) does not match "
            f"local tensor expert count {local_expert_count}"
        )
    return tuple(range(start, end))


def remap_mcore_layer_key(mcore_key: str, layer_map: dict[int, int] | None) -> str | None:
    if not layer_map:
        return mcore_key
    m = re.match(r"^layers\.(\d+)\.", mcore_key)
    if not m:
        return mcore_key
    source_layer = int(m.group(1))
    if source_layer not in layer_map:
        return None
    return f"layers.{layer_map[source_layer]}.{mcore_key[m.end():]}"


def load_native_checkpoint_into_mcore_model(
    model,
    checkpoint_dir: str = DEFAULT_V4_FLASH_FP8_CKPT,
    *,
    layer_map: dict[int, int] | None = None,
    strict: bool = True,
    include_mtp: bool | None = None,
) -> dict[str, int | tuple[str, ...]]:
    """Copy native V4 checkpoint tensors into an instantiated mcore model.

    The model controls the conversion slice: for example, a 1-layer model only
    receives `layers.0.*` tensors plus top-level embedding/head/norm/HC head.
    Expert tensors are copied expert-by-expert into the destination storage to
    avoid materializing a whole layer's expert block on CPU.

    ``include_mtp`` controls whether the ``mtp.*`` subtree is mapped: ``None``
    (default) auto-detects it from whether the model has an MTP head, so a model
    built WITHOUT MTP loads byte-identically to before.
    """
    if include_mtp is None:
        include_mtp = getattr(model, "mtp", None) is not None
    checkpoint = NativeV4Checkpoint(checkpoint_dir)
    targets = _model_named_tensors(model)
    modules = dict(model.named_modules())
    loaded: set[str] = set()
    direct_count = 0
    expert_slice_count = 0

    for native_key in sorted(checkpoint.actual_key_to_file()):
        mcore_key = native_key_to_mcore(native_key, include_mtp=include_mtp)
        # Layer remap applies to numbered decoder layers only; MTP keys carry no
        # remappable layer index (single ``mtp.*`` head) and are passed through.
        if mcore_key is not None and not mcore_key.startswith("mtp."):
            mcore_key = remap_mcore_layer_key(mcore_key, layer_map)
        if mcore_key is None or mcore_key not in targets:
            continue
        dest = targets[mcore_key]
        output_dtype = dest.dtype if dest.dtype.is_floating_point else torch.bfloat16
        tensor, _, _ = read_native_weight(checkpoint, native_key, output_dtype=output_dtype)
        dest.data.copy_(tensor.to(device=dest.device, dtype=dest.dtype))
        loaded.add(mcore_key)
        direct_count += 1

    # Expert targets come in two families: bf16 grouped Parameters
    # (``gate_up_proj``/``down_proj``) or packed-MXFP4 uint8 buffers
    # (``gate_up_proj_fp4``/``_sf``/``down_proj_fp4``/``_sf``,
    # V4_FP4_FROZEN_EXPERTS=1). The model's buffers define which family is
    # expected; read_expert_tensors{,_packed} must supply matching slices —
    # a bf16 model fed a packed checkpoint (or vice versa) fails loudly on the
    # slice-key lookup below.
    expert_targets = [k for k in targets if k.endswith("mlp.experts.gate_up_proj")]
    packed_expert_targets = [k for k in targets if k.endswith("mlp.experts.gate_up_proj_fp4")]
    target_to_source = {target: source for source, target in (layer_map or {}).items()}
    for gate_key in sorted(expert_targets) + sorted(packed_expert_targets):
        packed = gate_key.endswith(".gate_up_proj_fp4")
        gate_attr = ".gate_up_proj_fp4" if packed else ".gate_up_proj"
        module_name = gate_key[: -len(gate_attr)]
        is_mtp_experts = gate_key.startswith("mtp.")
        if is_mtp_experts:
            if not include_mtp:
                continue
            # Single MTP head -> native root ``mtp.0.ffn.experts.*``.
            source_root = "mtp.0.mlp.experts"
            source_layer = 0
        else:
            m = re.match(r"^layers\.(\d+)\.mlp\.experts\.", gate_key)
            if not m:
                continue
            target_layer = int(m.group(1))
            source_layer = target_to_source.get(target_layer, target_layer)
            source_root = f"layers.{source_layer}.mlp.experts"
        if packed:
            dest_names = ("gate_up_proj_fp4", "gate_up_proj_sf", "down_proj_fp4", "down_proj_sf")
        else:
            dest_names = ("gate_up_proj", "down_proj")
        dests = {n: targets[f"{module_name}.{n}"] for n in dest_names}
        gate_dest = dests[dest_names[0]]
        expert_module = modules.get(module_name)
        global_expert_ids = _local_expert_global_ids(expert_module, gate_dest.shape[0])
        for local_expert_idx, global_expert_id in enumerate(global_expert_ids):
            slices = read_expert_tensors(
                checkpoint,
                source_layer,
                global_expert_id,
                output_dtype=gate_dest.dtype if gate_dest.dtype.is_floating_point else torch.bfloat16,
                mtp=is_mtp_experts,
            )
            for n in dest_names:
                dest = dests[n]
                dest.data[local_expert_idx].copy_(
                    slices[f"{source_root}.{n}[{global_expert_id}]"].to(device=dest.device, dtype=dest.dtype)
                )
                expert_slice_count += 1
        for n in dest_names:
            loaded.add(f"{module_name}.{n}")

    missing = tuple(
        sorted(k for k in targets if k not in loaded and not (k.startswith("rotary_emb.") or ".rotary_emb." in k))
    )
    if strict and missing:
        raise AssertionError(f"mcore tensors not loaded from native checkpoint: {missing}")
    return {
        "direct_tensors": direct_count,
        "expert_slices": expert_slice_count,
        "missing": missing,
    }


def assemble_layer_experts(
    checkpoint: NativeV4Checkpoint,
    layer: int,
    *,
    num_experts: int,
    output_dtype: torch.dtype = torch.bfloat16,
) -> dict[str, torch.Tensor]:
    """Assemble a full layer's grouped expert tensors.

    This intentionally materializes one layer's full expert block and is too large
    to call accidentally for the full 43-layer checkpoint.
    """
    gate_up = []
    down = []
    for expert in range(num_experts):
        slices = read_expert_tensors(checkpoint, layer, expert, output_dtype=output_dtype)
        gate_up.append(slices[f"layers.{layer}.mlp.experts.gate_up_proj[{expert}]"])
        down.append(slices[f"layers.{layer}.mlp.experts.down_proj[{expert}]"])
    return {
        f"layers.{layer}.mlp.experts.gate_up_proj": torch.stack(gate_up, dim=0),
        f"layers.{layer}.mlp.experts.down_proj": torch.stack(down, dim=0),
    }


def _audit_one_tensor(
    checkpoint: NativeV4Checkpoint,
    native_key: str,
    mcore_key: str | None,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> TensorAudit:
    raw = checkpoint.get_tensor(native_key)
    converted, scale_key, scale = read_native_weight(checkpoint, native_key, output_dtype=output_dtype)
    return TensorAudit(
        native_key=native_key,
        mcore_key=mcore_key,
        shape=tuple(raw.shape),
        dtype=str(raw.dtype),
        scale_key=scale_key,
        scale_shape=tuple(scale.shape) if scale is not None else None,
        converted_shape=tuple(converted.shape),
        converted_dtype=str(converted.dtype),
    )


def audit_checkpoint(
    checkpoint_dir: str = DEFAULT_V4_FLASH_FP8_CKPT,
    *,
    sample_layers: Iterable[int] = (0, 2, 42),
    sample_experts: Iterable[int] = (0, 255),
    output_dtype: torch.dtype = torch.bfloat16,
) -> CheckpointAudit:
    checkpoint = NativeV4Checkpoint(checkpoint_dir)
    actual = checkpoint.actual_key_to_file()
    stale = checkpoint.stale_index_keys()

    ignored_mtp = sum(1 for k in actual if k.startswith("mtp."))
    ignored_scale = sum(1 for k in actual if k.endswith(".scale"))
    direct_mapped = 0
    expert_weight = 0
    unmapped = []
    for key in actual:
        if key.startswith("mtp.") or key.endswith(".scale"):
            continue
        if EXPERT_RE.match(key):
            expert_weight += 1
            continue
        if native_key_to_mcore(key) is None:
            unmapped.append(key)
        else:
            direct_mapped += 1

    tensor_samples = [
        "embed.weight",
        "head.weight",
        "hc_head_fn",
        "norm.weight",
    ]
    for layer in sample_layers:
        tensor_samples.extend(
            [
                f"layers.{layer}.attn.wq_a.weight",
                f"layers.{layer}.attn.wkv.weight",
                f"layers.{layer}.attn.wo_a.weight",
                f"layers.{layer}.ffn.gate.weight",
                f"layers.{layer}.ffn.shared_experts.w1.weight",
            ]
        )
        for optional in (
            f"layers.{layer}.attn.compressor.wkv.weight",
            f"layers.{layer}.attn.indexer.wq_b.weight",
            f"layers.{layer}.ffn.gate.tid2eid",
            f"layers.{layer}.ffn.gate.bias",
        ):
            if checkpoint.has_tensor(optional):
                tensor_samples.append(optional)

    audits = []
    seen = set()
    for key in tensor_samples:
        if key in seen or not checkpoint.has_tensor(key):
            continue
        seen.add(key)
        audits.append(
            _audit_one_tensor(
                checkpoint,
                key,
                native_key_to_mcore(key),
                output_dtype=output_dtype,
            )
        )

    expert_audits = []
    for layer in sample_layers:
        for expert in sample_experts:
            w1_key, w3_key, w2_key = expert_native_keys(layer, expert)
            if not all(checkpoint.has_tensor(k) for k in (w1_key, w3_key, w2_key)):
                continue
            slices = read_expert_tensors(checkpoint, layer, expert, output_dtype=output_dtype)
            for mcore_key, tensor in slices.items():
                expert_audits.append(
                    TensorAudit(
                        native_key=f"layers.{layer}.ffn.experts.{expert}.w[1/3/2]",
                        mcore_key=mcore_key,
                        shape=tuple(tensor.shape),
                        dtype=str(tensor.dtype),
                        converted_shape=tuple(tensor.shape),
                        converted_dtype=str(tensor.dtype),
                    )
                )

    return CheckpointAudit(
        checkpoint=checkpoint_dir,
        indexed_keys=len(checkpoint.weight_map),
        actual_keys=len(actual),
        shard_count=len(checkpoint.shard_files),
        stale_index_keys=stale,
        ignored_mtp_keys=ignored_mtp,
        ignored_scale_keys=ignored_scale,
        direct_mapped_keys=direct_mapped,
        expert_weight_keys=expert_weight,
        unmapped_non_mtp_keys=tuple(sorted(unmapped)),
        tensor_audits=tuple(audits),
        expert_audits=tuple(expert_audits),
    )


def audit_to_dict(audit: CheckpointAudit) -> dict:
    data = asdict(audit)
    data["stale_index_key_counts"] = dict(Counter(k.rsplit(".", 1)[-1] for k in audit.stale_index_keys))
    return data


def estimate_converted_checkpoint_size(
    checkpoint_dir: str = DEFAULT_V4_FLASH_FP8_CKPT,
    *,
    output_dtype: torch.dtype = torch.bfloat16,
) -> CheckpointSizeEstimate:
    """Estimate native payload and converted Megatron model payload from metadata only.

    This does not account for distributed-checkpoint file/container overhead, optimizer
    state, rotary buffers, or duplicated replicated ranks. It answers the gating
    question for R2: how large the mapped model parameter payload becomes after
    converting floating native tensors to the dtype Megatron will load.
    """
    checkpoint = NativeV4Checkpoint(checkpoint_dir)
    metadata = checkpoint.tensor_metadata()
    output_dtype_bytes = dtype_nbytes(output_dtype)

    actual_payload = 0
    non_mtp_payload = 0
    ignored_mtp_payload = 0
    ignored_scale_payload = 0
    mapped_direct = 0
    expert_source = 0
    unmapped_non_mtp = 0
    converted_payload = 0

    for key, meta in metadata.items():
        actual_payload += meta.nbytes
        if key.startswith("mtp."):
            ignored_mtp_payload += meta.nbytes
            continue
        non_mtp_payload += meta.nbytes
        if key.endswith(".scale"):
            ignored_scale_payload += meta.nbytes
            continue

        if EXPERT_RE.match(key):
            expert_source += 1
            converted_payload += numel(meta.shape) * output_dtype_bytes
            continue

        if native_key_to_mcore(key) is None:
            unmapped_non_mtp += 1
            continue

        mapped_direct += 1
        if is_floating_dtype_name(meta.dtype):
            converted_payload += numel(meta.shape) * output_dtype_bytes
        else:
            converted_payload += meta.nbytes

    gib = 1024**3
    usage = shutil.disk_usage(checkpoint_dir)
    return CheckpointSizeEstimate(
        checkpoint=checkpoint_dir,
        actual_payload_bytes=actual_payload,
        non_mtp_payload_bytes=non_mtp_payload,
        ignored_mtp_payload_bytes=ignored_mtp_payload,
        ignored_scale_payload_bytes=ignored_scale_payload,
        mapped_direct_tensors=mapped_direct,
        expert_source_tensors=expert_source,
        unmapped_non_mtp_tensors=unmapped_non_mtp,
        converted_mcore_payload_bytes=converted_payload,
        converted_mcore_payload_gib=converted_payload / gib,
        native_actual_payload_gib=actual_payload / gib,
        native_non_mtp_payload_gib=non_mtp_payload / gib,
        filesystem_free_gib=usage.free / gib,
        output_dtype=str(output_dtype),
    )


def contiguous_pp_layer_ids(
    num_layers: int,
    pp_size: int,
    *,
    num_layers_in_first_pipeline_stage: int | None = None,
    num_layers_in_last_pipeline_stage: int | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Compute non-interleaved contiguous PP layer ids for metadata planning."""
    if pp_size < 1:
        raise ValueError(f"pp_size must be >= 1, got {pp_size}")
    first = num_layers_in_first_pipeline_stage
    last = num_layers_in_last_pipeline_stage
    remaining = num_layers
    middle_stages = pp_size
    if first is not None:
        remaining -= first
        middle_stages -= 1
    if last is not None:
        remaining -= last
        middle_stages -= 1
    if remaining < 0:
        raise ValueError("first/last pipeline layer counts exceed total layers")
    middle = 0
    if middle_stages:
        if remaining % middle_stages:
            raise ValueError("remaining layers must divide middle pipeline stages")
        middle = remaining // middle_stages
    counts = []
    for pp_rank in range(pp_size):
        if pp_rank == 0 and first is not None:
            counts.append(first)
        elif pp_rank == pp_size - 1 and last is not None:
            counts.append(last)
        else:
            counts.append(middle)
    if sum(counts) != num_layers:
        raise AssertionError((counts, num_layers))
    offset = 0
    splits = []
    for count in counts:
        splits.append(tuple(range(offset, offset + count)))
        offset += count
    return tuple(splits)


def estimate_pp_ep_conversion_plan(
    checkpoint_dir: str = DEFAULT_V4_FLASH_FP8_CKPT,
    *,
    pp_size: int,
    ep_size: int,
    num_layers: int = 43,
    num_layers_in_first_pipeline_stage: int | None = None,
    num_layers_in_last_pipeline_stage: int | None = None,
    output_dtype: torch.dtype = torch.bfloat16,
) -> ConversionPlanEstimate:
    """Estimate per-rank payload for a PP x EP conversion job from metadata only."""
    if ep_size < 1:
        raise ValueError(f"ep_size must be >= 1, got {ep_size}")
    checkpoint = NativeV4Checkpoint(checkpoint_dir)
    metadata = checkpoint.tensor_metadata()
    output_dtype_bytes = dtype_nbytes(output_dtype)
    layer_splits = contiguous_pp_layer_ids(
        num_layers,
        pp_size,
        num_layers_in_first_pipeline_stage=num_layers_in_first_pipeline_stage,
        num_layers_in_last_pipeline_stage=num_layers_in_last_pipeline_stage,
    )
    layer_to_pp = {layer: pp_rank for pp_rank, layers in enumerate(layer_splits) for layer in layers}
    direct_by_pp = [0 for _ in range(pp_size)]
    expert_by_pp_ep = [[0 for _ in range(ep_size)] for _ in range(pp_size)]
    expert_ids = set()
    total_unique = 0

    def converted_nbytes(meta: TensorMetadata) -> int:
        if is_floating_dtype_name(meta.dtype):
            return numel(meta.shape) * output_dtype_bytes
        return meta.nbytes

    for key, meta in metadata.items():
        if key.startswith("mtp.") or key.endswith(".scale"):
            continue
        size = converted_nbytes(meta)
        expert_match = EXPERT_RE.match(key)
        if expert_match:
            layer = int(expert_match.group(1))
            expert = int(expert_match.group(2))
            if layer not in layer_to_pp:
                raise ValueError(f"expert layer {layer} not covered by PP split")
            expert_ids.add(expert)
            total_unique += size
            continue
        mcore_key = native_key_to_mcore(key)
        if mcore_key is None:
            continue
        layer_match = re.match(r"^layers\.(\d+)\.", mcore_key)
        pp_rank = layer_to_pp[int(layer_match.group(1))] if layer_match else 0
        # Top-level post-process weights live on the last PP stage.
        if not layer_match and mcore_key in {
            "output_layer.weight",
            "norm.weight",
            "hc_head.hc_fn",
            "hc_head.hc_base",
            "hc_head.hc_scale",
        }:
            pp_rank = pp_size - 1
        direct_by_pp[pp_rank] += size
        total_unique += size

    if not expert_ids:
        raise ValueError("no routed experts found in checkpoint metadata")
    num_experts = max(expert_ids) + 1
    if num_experts % ep_size:
        raise ValueError(f"num_experts {num_experts} must divide ep_size {ep_size}")
    experts_per_ep_rank = num_experts // ep_size

    for key, meta in metadata.items():
        expert_match = EXPERT_RE.match(key)
        if not expert_match:
            continue
        layer = int(expert_match.group(1))
        expert = int(expert_match.group(2))
        if key.startswith("mtp."):
            continue
        pp_rank = layer_to_pp[layer]
        ep_rank = expert // experts_per_ep_rank
        expert_by_pp_ep[pp_rank][ep_rank] += converted_nbytes(meta)

    gib = 1024**3
    ranks = []
    for pp_rank, layers in enumerate(layer_splits):
        for ep_rank in range(ep_size):
            direct = direct_by_pp[pp_rank]
            expert = expert_by_pp_ep[pp_rank][ep_rank]
            total = direct + expert
            ranks.append(
                ConversionPlanRankEstimate(
                    pp_rank=pp_rank,
                    ep_rank=ep_rank,
                    layers=layers,
                    direct_payload_bytes=direct,
                    expert_payload_bytes=expert,
                    total_payload_bytes=total,
                    total_payload_gib=total / gib,
                )
            )
    return ConversionPlanEstimate(
        checkpoint=checkpoint_dir,
        pp_size=pp_size,
        ep_size=ep_size,
        layer_splits=layer_splits,
        rank_estimates=tuple(ranks),
        max_rank_payload_gib=max(rank.total_payload_gib for rank in ranks),
        total_unique_payload_gib=total_unique / gib,
        output_dtype=str(output_dtype),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit DeepSeek-V4-Flash native FP8 checkpoint mapping to mcore.")
    parser.add_argument("--checkpoint", default=DEFAULT_V4_FLASH_FP8_CKPT)
    parser.add_argument("--sample-layers", type=int, nargs="+", default=[0, 2, 42])
    parser.add_argument("--sample-experts", type=int, nargs="+", default=[0, 255])
    parser.add_argument("--output-json", default=None)
    parser.add_argument(
        "--estimate-size-json",
        default=None,
        help="Write a metadata-only converted-size estimate JSON next to the audit.",
    )
    parser.add_argument("--plan-json", default=None, help="Write a PP x EP conversion plan estimate JSON.")
    parser.add_argument("--plan-pp-size", type=int, default=4)
    parser.add_argument("--plan-ep-size", type=int, default=4)
    parser.add_argument("--plan-first-layers", type=int, default=None)
    parser.add_argument("--plan-last-layers", type=int, default=None)
    args = parser.parse_args(argv)

    audit = audit_checkpoint(
        args.checkpoint,
        sample_layers=args.sample_layers,
        sample_experts=args.sample_experts,
    )
    data = audit_to_dict(audit)
    rendered = json.dumps(data, indent=2, sort_keys=True)
    if args.output_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.output_json)), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(rendered + "\n")
    if args.estimate_size_json:
        estimate = estimate_converted_checkpoint_size(args.checkpoint)
        estimate_rendered = json.dumps(asdict(estimate), indent=2, sort_keys=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.estimate_size_json)), exist_ok=True)
        with open(args.estimate_size_json, "w", encoding="utf-8") as f:
            f.write(estimate_rendered + "\n")
    if args.plan_json:
        plan = estimate_pp_ep_conversion_plan(
            args.checkpoint,
            pp_size=args.plan_pp_size,
            ep_size=args.plan_ep_size,
            num_layers_in_first_pipeline_stage=args.plan_first_layers,
            num_layers_in_last_pipeline_stage=args.plan_last_layers,
        )
        plan_rendered = json.dumps(asdict(plan), indent=2, sort_keys=True)
        os.makedirs(os.path.dirname(os.path.abspath(args.plan_json)), exist_ok=True)
        with open(args.plan_json, "w", encoding="utf-8") as f:
            f.write(plan_rendered + "\n")
    print(rendered)
    if audit.unmapped_non_mtp_keys:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
