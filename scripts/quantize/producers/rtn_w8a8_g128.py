"""Offline RTN W8A8-G128 checkpoint writer for SGLang `blockwise_int8`.

This emits SGLang's blockwise INT8 format:
  - quantized Linear weights stay in `.weight` as int8
  - block scales are stored as `.weight_scale_inv` with shape
    `(ceil(out / block_n), ceil(in / block_k))`
  - activations are dynamically quantized per token group by SGLang

The default `--target non_linear_attn_mtp` matches the current EAGLE rollout
scope: quantize body + MTP MLP/self_attn, while leaving mamba-style
`linear_attn` and `mtp.fc` in BF16.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.quantize.producers.rtn_w8a8 import should_quantize_weight
from scripts.quantize.utils.validate_checkpoint import check_w8a8_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path, help="Source BF16 HF checkpoint directory")
    parser.add_argument("--output-path", required=True, type=Path, help="Destination W8A8-G128 checkpoint directory")
    parser.add_argument(
        "--target",
        choices=["all-linear", "mlp", "mlp_mtp", "non_linear_attn", "non_linear_attn_mtp"],
        default="non_linear_attn_mtp",
        help=(
            "Text-side Linear weights to quantize. The default quantizes MLP + self_attn "
            "in both the body and MTP draft layer, and skips linear_attn."
        ),
    )
    parser.add_argument("--block-size", type=int, nargs=2, default=[128, 128], metavar=("BLOCK_N", "BLOCK_K"))
    parser.add_argument("--force", action="store_true", help="Remove an existing output directory before writing")
    parser.add_argument("--validation-max-tensors", type=int, default=32)
    parser.add_argument("--validation-max-saturated-frac", type=float, default=0.20)
    parser.add_argument("--validation-max-rel-l2", type=float, default=0.08)
    return parser.parse_args()


def _load_weight_map(checkpoint_dir: Path) -> tuple[dict[str, str], list[str]]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map = dict(index["weight_map"])
        shards = sorted(set(weight_map.values()))
        return weight_map, shards

    safetensors_files = sorted(path.name for path in checkpoint_dir.glob("*.safetensors"))
    if len(safetensors_files) == 1:
        shard = safetensors_files[0]
        with safe_open(checkpoint_dir / shard, framework="pt", device="cpu") as f:
            return {name: shard for name in f.keys()}, [shard]

    raise FileNotFoundError(
        f"Could not resolve safetensors in {checkpoint_dir}; expected an index or one safetensors file."
    )


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


@torch.no_grad()
def quantize_layer_blockwise_int8(
    weight: torch.Tensor,
    *,
    block_size: tuple[int, int] = (128, 128),
    scale_dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_features, in_features = weight.shape
    block_n, block_k = block_size
    if block_n <= 0 or block_k <= 0:
        raise ValueError(f"block_size entries must be positive, got {block_size}")

    w_fp32 = weight.to(torch.float32)
    q = torch.empty_like(w_fp32, dtype=torch.int8)
    n_blocks = (out_features + block_n - 1) // block_n
    k_blocks = (in_features + block_k - 1) // block_k
    scale = torch.empty((n_blocks, k_blocks), dtype=scale_dtype)

    if out_features % block_n == 0 and in_features % block_k == 0:
        # Fast path for Qwen3.6 Linear shapes. Avoid per-block Python loops,
        # which become very slow for G64 on large SmoothQuant shards.
        view = w_fp32.reshape(n_blocks, block_n, k_blocks, block_k).permute(0, 2, 1, 3)
        scale_fp32 = (view.abs().amax(dim=(2, 3)) / 127.0).clamp(min=1e-8)
        q_view = torch.round(view / scale_fp32[:, :, None, None]).clamp(-127, 127).to(torch.int8)
        q.copy_(q_view.permute(0, 2, 1, 3).reshape(out_features, in_features))
        return q.contiguous(), scale_fp32.to(scale_dtype).contiguous()

    for row_block in range(n_blocks):
        row_start = row_block * block_n
        row_end = min(row_start + block_n, out_features)
        for col_block in range(k_blocks):
            col_start = col_block * block_k
            col_end = min(col_start + block_k, in_features)
            block = w_fp32[row_start:row_end, col_start:col_end]
            block_scale = (block.abs().amax() / 127.0).clamp(min=1e-8)
            q[row_start:row_end, col_start:col_end] = torch.round(block / block_scale).clamp(-127, 127).to(torch.int8)
            scale[row_block, col_block] = block_scale.to(scale_dtype)

    return q.contiguous(), scale.contiguous()


def build_quantization_config(target: str, block_size: tuple[int, int] = (128, 128)) -> dict:
    ignored_layers = [
        "lm_head",
        "visual",
        "vision_tower",
        "mm_projector",
        "audio_tower",
    ]
    if target not in ("mlp_mtp", "non_linear_attn_mtp"):
        ignored_layers.append("mtp")
    if target in ("mlp", "mlp_mtp"):
        ignored_layers.extend(["self_attn", "linear_attn"])
        if target == "mlp_mtp":
            ignored_layers.append("mtp.fc")
    elif target in ("non_linear_attn", "non_linear_attn_mtp"):
        ignored_layers.append("linear_attn")
        if target == "non_linear_attn_mtp":
            ignored_layers.append("mtp.fc")

    return {
        "quant_method": "blockwise_int8",
        "activation_scheme": "dynamic",
        "weight_block_size": [int(block_size[0]), int(block_size[1])],
        "ignored_layers": ignored_layers,
    }


def _copy_non_weight_files(model_path: Path, output_path: Path, target: str, block_size: tuple[int, int]) -> None:
    for src in model_path.iterdir():
        if not src.is_file():
            continue
        if src.suffix == ".safetensors" or src.name == "model.safetensors.index.json":
            continue
        dst = output_path / src.name
        shutil.copy2(src, dst)

    config_path = output_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{model_path} does not contain config.json")
    config = json.loads(config_path.read_text())
    config["quantization_config"] = build_quantization_config(target, block_size)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


@torch.no_grad()
def quantize_checkpoint(
    model_path: str | Path,
    output_path: str | Path,
    *,
    target: str = "non_linear_attn_mtp",
    block_size: tuple[int, int] = (128, 128),
    force: bool = False,
) -> int:
    model_path = Path(model_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path == model_path:
        raise ValueError("--output-path must be different from --model-path")
    if output_path.exists():
        if not force:
            raise FileExistsError(f"{output_path} already exists; pass --force to overwrite it")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    weight_map, shards = _load_weight_map(model_path)
    shard_to_names: dict[str, list[str]] = {shard: [] for shard in shards}
    for name, shard in weight_map.items():
        shard_to_names[shard].append(name)

    out_weight_map: dict[str, str] = {}
    total_size = 0
    quantized_count = 0

    for shard in shards:
        out_tensors: dict[str, torch.Tensor] = {}
        with safe_open(model_path / shard, framework="pt", device="cpu") as f:
            for name in sorted(shard_to_names[shard]):
                tensor = f.get_tensor(name)
                if should_quantize_weight(name, tuple(tensor.shape), target):
                    q, scale = quantize_layer_blockwise_int8(tensor, block_size=block_size)
                    scale_name = name.replace(".weight", ".weight_scale_inv")
                    out_tensors[name] = q
                    out_tensors[scale_name] = scale
                    out_weight_map[name] = shard
                    out_weight_map[scale_name] = shard
                    total_size += _tensor_nbytes(q) + _tensor_nbytes(scale)
                    quantized_count += 1
                else:
                    tensor = tensor.contiguous()
                    out_tensors[name] = tensor
                    out_weight_map[name] = shard
                    total_size += _tensor_nbytes(tensor)

        save_file(out_tensors, output_path / shard)
        print(f"[rtn-g128] wrote {shard}: tensors={len(out_tensors)} quantized_so_far={quantized_count}", flush=True)

    (output_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": out_weight_map}, indent=2, sort_keys=True)
        + "\n"
    )
    _copy_non_weight_files(model_path, output_path, target, block_size)
    return quantized_count


def main() -> None:
    args = parse_args()
    block_size = (int(args.block_size[0]), int(args.block_size[1]))
    quantized_count = quantize_checkpoint(
        args.model_path,
        args.output_path,
        target=args.target,
        block_size=block_size,
        force=args.force,
    )
    print(f"[rtn-g128] quantized {quantized_count} tensors; validating saved checkpoint", flush=True)
    checks = check_w8a8_checkpoint(
        args.output_path,
        reference_checkpoint=args.model_path,
        max_tensors=args.validation_max_tensors,
        max_saturated_frac=args.validation_max_saturated_frac,
        max_rel_l2=args.validation_max_rel_l2,
    )
    print(f"[rtn-g128] validation OK: checked {len(checks)} tensors", flush=True)


if __name__ == "__main__":
    main()
