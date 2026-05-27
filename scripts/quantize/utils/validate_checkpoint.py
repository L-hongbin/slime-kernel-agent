"""Validate a compressed-tensors W8A8 RTN checkpoint before rollout.

This is a pre-rollout sanity check for the failure mode where a checkpoint is
syntactically loadable by SGLang but its INT8 weights are not a real RTN
quantization of the source BF16 weights. In that case generation can degrade
into garbage even though unit tests for the online tensor quantizer pass.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open


@dataclass
class TensorCheck:
    name: str
    shape: tuple[int, ...]
    dtype: str
    scale_shape: tuple[int, ...]
    unique_count: int
    saturated_frac: float
    zero_frac: float
    rel_l2: float | None = None


def _load_weight_map(checkpoint_dir: Path) -> dict[str, Path]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        return {name: checkpoint_dir / shard for name, shard in index["weight_map"].items()}

    safetensors_files = sorted(checkpoint_dir.glob("*.safetensors"))
    if len(safetensors_files) == 1:
        with safe_open(safetensors_files[0], framework="pt", device="cpu") as f:
            return {name: safetensors_files[0] for name in f.keys()}

    raise FileNotFoundError(
        f"Could not resolve safetensors weight map in {checkpoint_dir}; "
        "expected model.safetensors.index.json or exactly one *.safetensors file."
    )


def _get_tensor(weight_map: dict[str, Path], name: str) -> torch.Tensor:
    path = weight_map[name]
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def _dequant_symmetric_int8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    q_fp32 = q.to(torch.float32)
    scale_fp32 = scale.to(torch.float32)
    if scale.numel() == 1:
        return q_fp32 * scale_fp32.reshape(1)
    if scale.ndim == 1 and scale.shape[0] == q.shape[0]:
        return q_fp32 * scale_fp32[:, None]
    if scale.ndim == 2 and scale.shape == (q.shape[0], 1):
        return q_fp32 * scale_fp32
    if scale.ndim == 2 and scale.shape[0] == q.shape[0] and q.shape[1] % scale.shape[1] == 0:
        out_features, in_features = q.shape
        num_groups = scale.shape[1]
        group_size = in_features // num_groups
        return (q_fp32.reshape(out_features, num_groups, group_size) * scale_fp32.unsqueeze(-1)).reshape(
            out_features, in_features
        )
    raise ValueError(f"unsupported W8A8 scale shape {tuple(scale.shape)} for q shape {tuple(q.shape)}")


def check_w8a8_checkpoint(
    quantized_checkpoint: Path,
    *,
    reference_checkpoint: Path | None = None,
    max_tensors: int = 32,
    max_saturated_frac: float = 0.20,
    max_rel_l2: float = 0.08,
) -> list[TensorCheck]:
    q_map = _load_weight_map(quantized_checkpoint)
    ref_map = _load_weight_map(reference_checkpoint) if reference_checkpoint is not None else None

    quantized_weight_names = sorted(
        name for name in q_map if name.endswith(".weight") and name.replace(".weight", ".weight_scale") in q_map
    )
    if not quantized_weight_names:
        raise AssertionError(f"No int8 weights with .weight_scale found in {quantized_checkpoint}")

    checks: list[TensorCheck] = []
    failures: list[str] = []
    for name in quantized_weight_names[:max_tensors]:
        q = _get_tensor(q_map, name)
        scale = _get_tensor(q_map, name.replace(".weight", ".weight_scale"))

        if q.dtype not in (torch.int8, torch.uint8):
            failures.append(f"{name}: expected int8/uint8 weight, got {q.dtype}")
            continue
        if q.ndim != 2:
            failures.append(f"{name}: expected 2D weight, got shape {tuple(q.shape)}")
            continue

        q_flat = q.flatten()
        unique_count = int(torch.unique(q_flat).numel())
        if q.dtype == torch.int8:
            saturated = ((q_flat == 127) | (q_flat == -127) | (q_flat == -128)).float().mean().item()
        else:
            saturated = ((q_flat == 0) | (q_flat == 255)).float().mean().item()
        zero_frac = (q_flat == 0).float().mean().item()

        rel_l2 = None
        if ref_map is not None and name in ref_map:
            ref = _get_tensor(ref_map, name).to(torch.float32)
            deq = _dequant_symmetric_int8(q, scale)
            if tuple(deq.shape) != tuple(ref.shape):
                failures.append(f"{name}: dequant shape {tuple(deq.shape)} != reference shape {tuple(ref.shape)}")
            else:
                rel_l2 = ((deq - ref).norm() / ref.norm().clamp(min=1e-12)).item()

        checks.append(
            TensorCheck(
                name=name,
                shape=tuple(q.shape),
                dtype=str(q.dtype),
                scale_shape=tuple(scale.shape),
                unique_count=unique_count,
                saturated_frac=float(saturated),
                zero_frac=float(zero_frac),
                rel_l2=None if rel_l2 is None else float(rel_l2),
            )
        )

        if unique_count <= 3:
            failures.append(f"{name}: only {unique_count} unique int8 values; looks ternary, not RTN INT8")
        if saturated > max_saturated_frac:
            failures.append(f"{name}: saturated_frac={saturated:.4f} > {max_saturated_frac:.4f}")
        if rel_l2 is not None and rel_l2 > max_rel_l2:
            failures.append(f"{name}: rel_l2={rel_l2:.4f} > {max_rel_l2:.4f}")

    if failures:
        formatted = "\n".join(f"- {failure}" for failure in failures[:20])
        raise AssertionError(f"W8A8 checkpoint sanity check failed:\n{formatted}")

    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quantized-checkpoint", required=True, type=Path)
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--max-tensors", type=int, default=32)
    parser.add_argument("--max-saturated-frac", type=float, default=0.20)
    parser.add_argument("--max-rel-l2", type=float, default=0.08)
    args = parser.parse_args()

    checks = check_w8a8_checkpoint(
        args.quantized_checkpoint,
        reference_checkpoint=args.reference_checkpoint,
        max_tensors=args.max_tensors,
        max_saturated_frac=args.max_saturated_frac,
        max_rel_l2=args.max_rel_l2,
    )
    print(f"OK: checked {len(checks)} W8A8 tensors in {args.quantized_checkpoint}")
    for check in checks[:10]:
        rel = "n/a" if check.rel_l2 is None else f"{check.rel_l2:.6f}"
        print(
            f"{check.name}: shape={check.shape} dtype={check.dtype} "
            f"scale={check.scale_shape} unique={check.unique_count} "
            f"saturated={check.saturated_frac:.6f} zero={check.zero_frac:.6f} rel_l2={rel}"
        )


if __name__ == "__main__":
    main()
