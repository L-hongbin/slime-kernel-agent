"""Utilities for preserving Qwen3.6 MTP/EAGLE tensors across HF saves.

The public HF model class used by ``AutoModelForImageTextToText`` does not
instantiate the extra ``mtp.*`` tensors found in the checkpoint. A plain
``save_pretrained`` therefore silently drops them. SGLang does use those
tensors for EAGLE, so rotation/smoothing producers must copy them back into the
saved BF16 checkpoint before the RTN W8A8 writer runs.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


MTP_SHARD_NAME = "model-mtp.safetensors"


def _load_weight_map(checkpoint_dir: Path) -> dict[str, str]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        return dict(json.loads(index_path.read_text())["weight_map"])

    safetensors_files = sorted(p for p in checkpoint_dir.glob("*.safetensors") if p.name != MTP_SHARD_NAME)
    if len(safetensors_files) != 1:
        raise FileNotFoundError(
            f"Could not resolve safetensors in {checkpoint_dir}; expected an index or one safetensors file."
        )
    shard = safetensors_files[0].name
    with safe_open(safetensors_files[0], framework="pt", device="cpu") as f:
        return {name: shard for name in f.keys()}


def _dtype_nbytes(dtype_name: str) -> int:
    normalized = dtype_name.upper().replace("TORCH.", "")
    if normalized in {"BOOL", "U8", "I8", "UINT8", "INT8"}:
        return 1
    if normalized in {"F16", "BF16", "I16", "U16", "FLOAT16", "BFLOAT16", "INT16", "UINT16"}:
        return 2
    if normalized in {"F32", "I32", "U32", "FLOAT32", "INT32", "UINT32"}:
        return 4
    if normalized in {"F64", "I64", "U64", "FLOAT64", "INT64", "UINT64"}:
        return 8
    raise ValueError(f"unknown safetensors dtype {dtype_name!r}")


def _slice_nbytes(handle, name: str) -> int:
    tensor_slice = handle.get_slice(name)
    return math.prod(tensor_slice.get_shape()) * _dtype_nbytes(tensor_slice.get_dtype())


def _recalculate_total_size(checkpoint_dir: Path, weight_map: dict[str, str]) -> int:
    names_by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        names_by_shard[shard].append(name)

    total = 0
    for shard, names in names_by_shard.items():
        with safe_open(checkpoint_dir / shard, framework="pt", device="cpu") as f:
            keys = set(f.keys())
            missing = sorted(set(names) - keys)
            if missing:
                raise KeyError(f"{checkpoint_dir / shard} missing index tensors: {missing[:5]}")
            total += sum(_slice_nbytes(f, name) for name in names)
    return total


def _read_prefixed_tensors(checkpoint_dir: Path, prefix: str) -> dict[str, torch.Tensor]:
    weight_map = _load_weight_map(checkpoint_dir)
    names_by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in weight_map.items():
        if name.startswith(prefix):
            names_by_shard[shard].append(name)

    tensors: dict[str, torch.Tensor] = {}
    for shard, names in names_by_shard.items():
        with safe_open(checkpoint_dir / shard, framework="pt", device="cpu") as f:
            for name in sorted(names):
                tensors[name] = f.get_tensor(name).contiguous()
    return tensors


def inject_mtp_tensors(checkpoint_dir: str | Path, source_checkpoint: str | Path, prefix: str = "mtp.") -> int:
    """Copy ``mtp.*`` tensors from ``source_checkpoint`` into ``checkpoint_dir``.

    The tensors are written to a dedicated shard, and the HF safetensors index is
    updated to point all ``mtp.*`` names at that shard. Existing non-MTP tensors
    are left untouched.
    """

    checkpoint_dir = Path(checkpoint_dir)
    source_checkpoint = Path(source_checkpoint)
    tensors = _read_prefixed_tensors(source_checkpoint, prefix)
    if not tensors:
        raise ValueError(f"{source_checkpoint} has no tensors with prefix {prefix!r}")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, checkpoint_dir / MTP_SHARD_NAME)

    weight_map = _load_weight_map(checkpoint_dir)
    weight_map = {name: shard for name, shard in weight_map.items() if not name.startswith(prefix)}
    for name in tensors:
        weight_map[name] = MTP_SHARD_NAME

    total_size = _recalculate_total_size(checkpoint_dir, weight_map)
    (checkpoint_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2, sort_keys=True) + "\n"
    )
    return len(tensors)


def _right_rotate(weight: torch.Tensor, transform: torch.Tensor, device: torch.device) -> torch.Tensor:
    out = weight.to(device=device, dtype=torch.float64) @ transform
    return out.to(device="cpu", dtype=weight.dtype).contiguous()


def _left_rotate(weight: torch.Tensor, transform: torch.Tensor, device: torch.device) -> torch.Tensor:
    out = transform.T @ weight.to(device=device, dtype=torch.float64)
    return out.to(device="cpu", dtype=weight.dtype).contiguous()


def _fuse_norm_columns(weight: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
    return (weight.to(torch.float64) * norm.to(torch.float64).view(1, -1)).to(weight.dtype).contiguous()


def _gemma_effective_norm(norm: torch.Tensor) -> torch.Tensor:
    return (norm.to(torch.float64) + 1.0).to(dtype=norm.dtype)


def _gemma_identity_norm_like(norm: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(norm)


def build_quarot_r1_transform(
    hidden_size: int,
    transform_type: str = "random-hadamard",
    device: str | torch.device | None = None,
) -> torch.Tensor:
    """Rebuild llmcompressor's cached R1 transform matrix.

    llmcompressor's Hadamard factories cache one matrix per hidden size. For
    ``random-hadamard`` the factory uses a fresh ``torch.Generator()`` without a
    seed, so rebuilding with the same default generator reproduces the matrix
    used by the body R1 transform. The returned matrix is normalized by
    ``sqrt(hidden_size)`` because ``HadamardTransform.forward`` applies the same
    normalization at use time.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    if transform_type == "random-hadamard":
        from compressed_tensors.transform.utils.hadamard import random_hadamard_matrix

        transform = random_hadamard_matrix(
            hidden_size,
            dtype=torch.float64,
            device=device,
            gen=torch.Generator(),
        )
    elif transform_type == "hadamard":
        from compressed_tensors.transform.utils.hadamard import deterministic_hadamard_matrix

        transform = deterministic_hadamard_matrix(hidden_size, dtype=torch.float64, device=device)
    else:
        raise ValueError(f"MTP rotation only supports hadamard/random-hadamard, got {transform_type!r}")

    return transform / math.sqrt(hidden_size)


def rotate_qwen35_mtp_tensors(
    tensors: dict[str, torch.Tensor],
    transform: torch.Tensor,
    body_final_norm: torch.Tensor | None = None,
    lm_head_weight: torch.Tensor | None = None,
    final_norm_mode: str = "ratio",
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Apply the same R1 residual-basis transform to Qwen3.5/3.6 MTP tensors.

    This mirrors the main model's SpinQuant R1 choices:
    - projections that read residual state are right-rotated
    - projections that write residual state are left-rotated
    - RMSNorm scales immediately before linears are fused into those linears

    ``mtp.norm`` is special because SGLang normally shares the top-level
    ``lm_head`` with the target model. Under QuaRot the target head has already
    fused the target final norm, while MTP has its own final norm. Qwen3.5/3.6
    uses Gemma-style RMSNorm, whose effective scale is ``1 + weight`` and whose
    identity weight is zero. The exact mode is ``separate_lm_head``: emit
    ``mtp.lm_head.weight = lm_head * (1 + mtp.norm) * R`` and set ``mtp.norm``
    to zero. That requires a matching SGLang patch so the draft model keeps the
    separate MTP head instead of overwriting it with the target head.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    transform = transform.to(device=device, dtype=torch.float64)

    out = {name: tensor.detach().cpu().contiguous().clone() for name, tensor in tensors.items()}
    hidden = transform.shape[0]

    fc = out["mtp.fc.weight"]
    if fc.shape != (hidden, 2 * hidden):
        raise ValueError(f"unexpected mtp.fc shape {tuple(fc.shape)} for hidden={hidden}")
    fc = fc.to(torch.float64)
    fc[:, :hidden] *= _gemma_effective_norm(out["mtp.pre_fc_norm_embedding.weight"]).to(torch.float64).view(1, -1)
    fc[:, hidden:] *= _gemma_effective_norm(out["mtp.pre_fc_norm_hidden.weight"]).to(torch.float64).view(1, -1)
    fc = (transform.T @ fc.to(device=device)).to(torch.float64)
    fc_left = fc[:, :hidden] @ transform
    fc_right = fc[:, hidden:] @ transform
    out["mtp.fc.weight"] = torch.cat([fc_left, fc_right], dim=1).to(device="cpu", dtype=tensors["mtp.fc.weight"].dtype)
    out["mtp.pre_fc_norm_embedding.weight"] = _gemma_identity_norm_like(out["mtp.pre_fc_norm_embedding.weight"])
    out["mtp.pre_fc_norm_hidden.weight"] = _gemma_identity_norm_like(out["mtp.pre_fc_norm_hidden.weight"])

    input_norm = _gemma_effective_norm(out["mtp.layers.0.input_layernorm.weight"])
    for proj in ("q_proj", "k_proj", "v_proj"):
        key = f"mtp.layers.0.self_attn.{proj}.weight"
        out[key] = _right_rotate(_fuse_norm_columns(out[key], input_norm), transform, device)
    out["mtp.layers.0.input_layernorm.weight"] = _gemma_identity_norm_like(out["mtp.layers.0.input_layernorm.weight"])
    out["mtp.layers.0.self_attn.o_proj.weight"] = _left_rotate(
        out["mtp.layers.0.self_attn.o_proj.weight"], transform, device
    )

    post_norm = _gemma_effective_norm(out["mtp.layers.0.post_attention_layernorm.weight"])
    for proj in ("gate_proj", "up_proj"):
        key = f"mtp.layers.0.mlp.{proj}.weight"
        out[key] = _right_rotate(_fuse_norm_columns(out[key], post_norm), transform, device)
    out["mtp.layers.0.post_attention_layernorm.weight"] = _gemma_identity_norm_like(
        out["mtp.layers.0.post_attention_layernorm.weight"]
    )
    out["mtp.layers.0.mlp.down_proj.weight"] = _left_rotate(
        out["mtp.layers.0.mlp.down_proj.weight"], transform, device
    )

    if final_norm_mode == "ones":
        out["mtp.norm.weight"] = _gemma_identity_norm_like(out["mtp.norm.weight"])
    elif final_norm_mode == "ratio":
        if body_final_norm is None:
            raise ValueError("body_final_norm is required for final_norm_mode='ratio'")
        out["mtp.norm.weight"] = (
            (
                _gemma_effective_norm(out["mtp.norm.weight"]).float()
                / _gemma_effective_norm(body_final_norm).float().clamp(min=1e-6)
            )
            - 1.0
        ).to(out["mtp.norm.weight"].dtype)
    elif final_norm_mode == "separate_lm_head":
        if lm_head_weight is None:
            raise ValueError("lm_head_weight is required for final_norm_mode='separate_lm_head'")
        out["mtp.lm_head.weight"] = _right_rotate(
            _fuse_norm_columns(
                lm_head_weight.detach().cpu().contiguous(),
                _gemma_effective_norm(out["mtp.norm.weight"]),
            ),
            transform,
            device,
        )
        out["mtp.norm.weight"] = _gemma_identity_norm_like(out["mtp.norm.weight"])
    elif final_norm_mode == "keep":
        pass
    else:
        raise ValueError(f"unknown final_norm_mode {final_norm_mode!r}")

    return {name: tensor.contiguous() for name, tensor in out.items()}


def inject_rotated_mtp_tensors(
    checkpoint_dir: str | Path,
    source_checkpoint: str | Path,
    transform: torch.Tensor,
    final_norm_mode: str = "separate_lm_head",
    prefix: str = "mtp.",
) -> int:
    """Copy MTP tensors from ``source_checkpoint``, rotate them, and inject them."""

    checkpoint_dir = Path(checkpoint_dir)
    source_checkpoint = Path(source_checkpoint)
    tensors = _read_prefixed_tensors(source_checkpoint, prefix)
    if not tensors:
        raise ValueError(f"{source_checkpoint} has no tensors with prefix {prefix!r}")

    body_norm = None
    if final_norm_mode == "ratio":
        body_norms = _read_prefixed_tensors(source_checkpoint, "model.language_model.norm.")
        body_norm = body_norms.get("model.language_model.norm.weight")
        if body_norm is None:
            raise ValueError(f"{source_checkpoint} has no model.language_model.norm.weight")

    lm_head_weight = None
    if final_norm_mode == "separate_lm_head":
        lm_heads = _read_prefixed_tensors(source_checkpoint, "lm_head.")
        lm_head_weight = lm_heads.get("lm_head.weight")
        if lm_head_weight is None:
            raise ValueError(f"{source_checkpoint} has no lm_head.weight")

    rotated = rotate_qwen35_mtp_tensors(
        tensors,
        transform=transform,
        body_final_norm=body_norm,
        lm_head_weight=lm_head_weight,
        final_norm_mode=final_norm_mode,
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_file(rotated, checkpoint_dir / MTP_SHARD_NAME)

    weight_map = _load_weight_map(checkpoint_dir)
    weight_map = {name: shard for name, shard in weight_map.items() if not name.startswith(prefix)}
    for name in rotated:
        weight_map[name] = MTP_SHARD_NAME

    total_size = _recalculate_total_size(checkpoint_dir, weight_map)
    (checkpoint_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2, sort_keys=True) + "\n"
    )
    return len(rotated)
