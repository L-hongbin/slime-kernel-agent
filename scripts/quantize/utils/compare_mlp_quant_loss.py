"""Compare local MLP quantization loss against the BF16 checkpoint.

This is a cheap pre-eval probe. It does not load SGLang and does not generate
tokens. For each requested layer it compares:

    MLP_variant(RMSNorm_variant(x)) vs MLP_bf16(RMSNorm_bf16(x))

The probe intentionally dequantizes compressed-tensors weights itself instead
of importing compressed_tensors. On .16, importing that package can pull in a
system torchvision build that is incompatible with the venv torch build.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def parse_layers(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def load_weight_map(root: Path) -> dict[str, str]:
    index = root / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text())["weight_map"]

    single = root / "model.safetensors"
    if not single.exists():
        raise FileNotFoundError(f"no safetensors index or single shard found under {root}")
    with safe_open(single, framework="pt", device="cpu") as handle:
        return {key: single.name for key in handle.keys()}


def load_tensor(root: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(root / weight_map[name], framework="pt", device="cpu") as handle:
        return handle.get_tensor(name)


def layer_prefix(layer: int) -> str:
    return f"model.language_model.layers.{layer}"


def tensor_name(layer: int, suffix: str) -> str:
    return f"{layer_prefix(layer)}.{suffix}"


def unpack_int32(
    value: torch.Tensor,
    num_bits: int,
    shape: torch.Size | tuple[int, ...],
    *,
    packed_dim: int = 1,
) -> torch.Tensor:
    if value.dtype is not torch.int32:
        raise TypeError(f"expected int32 packed tensor, got {value.dtype}")
    if value.ndim != 2:
        raise ValueError(f"expected 2D packed tensor, got shape={tuple(value.shape)}")

    pack_factor = 32 // num_bits
    mask = (1 << num_bits) - 1
    shape = tuple(int(dim) for dim in shape)

    if packed_dim == 1:
        unpacked = torch.empty(
            (value.shape[0], value.shape[1] * pack_factor),
            dtype=torch.int32,
            device=value.device,
        )
        for idx in range(pack_factor):
            unpacked[:, idx::pack_factor] = (value >> (num_bits * idx)) & mask
        unpacked = unpacked[:, : shape[1]]
    elif packed_dim == 0:
        unpacked = torch.empty(
            (value.shape[0] * pack_factor, value.shape[1]),
            dtype=torch.int32,
            device=value.device,
        )
        for idx in range(pack_factor):
            unpacked[idx::pack_factor, :] = (value >> (num_bits * idx)) & mask
        unpacked = unpacked[: shape[0], :]
    else:
        raise ValueError(f"packed_dim must be 0 or 1, got {packed_dim}")

    offset = 1 << (num_bits - 1)
    return (unpacked - offset).to(torch.int8)


def dequant_w4(root: Path, weight_map: dict[str, str], base: str) -> torch.Tensor:
    packed = load_tensor(root, weight_map, f"{base}.weight_packed")
    scale = load_tensor(root, weight_map, f"{base}.weight_scale").float()
    shape = load_tensor(root, weight_map, f"{base}.weight_shape").tolist()
    q_weight = unpack_int32(packed, 4, shape, packed_dim=1).float()

    rows, cols = int(shape[0]), int(shape[1])
    groups = scale.shape[-1]
    group_size = cols // groups
    q_grouped = q_weight.view(rows, groups, group_size)

    zp_name = f"{base}.weight_zero_point"
    if zp_name in weight_map:
        packed_zp = load_tensor(root, weight_map, zp_name)
        zp = unpack_int32(packed_zp, 4, (rows, groups), packed_dim=0).float()
        dequant = (q_grouped - zp.unsqueeze(-1)) * scale.unsqueeze(-1)
    else:
        dequant = q_grouped * scale.unsqueeze(-1)

    return dequant.reshape(rows, cols)


def dequant_w8(root: Path, weight_map: dict[str, str], base: str) -> torch.Tensor:
    weight = load_tensor(root, weight_map, f"{base}.weight").float()
    scale = load_tensor(root, weight_map, f"{base}.weight_scale").float()
    zp_name = f"{base}.weight_zero_point"
    if zp_name in weight_map:
        zero_point = load_tensor(root, weight_map, zp_name).float()
        return (weight - zero_point) * scale
    return weight * scale


def load_bf16_weight(root: Path, weight_map: dict[str, str], base: str) -> torch.Tensor:
    return load_tensor(root, weight_map, f"{base}.weight").float()


def fake_rtn_w4(weight: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    rows, cols = weight.shape
    if cols % group_size != 0:
        raise ValueError(f"{tuple(weight.shape)} columns are not divisible by {group_size}")
    grouped = weight.float().view(rows, cols // group_size, group_size)
    scale = grouped.abs().amax(dim=-1, keepdim=True) / 7.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quant = torch.round(grouped / scale).clamp(-8, 7)
    return (quant * scale).reshape(rows, cols)


def rmsnorm_qwen(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return normed * (1.0 + weight)


def mlp_forward(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    weights: dict[str, torch.Tensor],
    eps: float,
) -> torch.Tensor:
    h = rmsnorm_qwen(x, norm_weight, eps)
    gate = torch.nn.functional.linear(h, weights["gate_proj"])
    up = torch.nn.functional.linear(h, weights["up_proj"])
    hidden = torch.nn.functional.silu(gate) * up
    return torch.nn.functional.linear(hidden, weights["down_proj"])


def load_mlp_weights(
    kind: str,
    root: Path,
    weight_map: dict[str, str],
    layer: int,
) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {}
    for projection in PROJECTIONS:
        base = tensor_name(layer, f"mlp.{projection}")
        if kind == "bf16":
            weights[projection] = load_bf16_weight(root, weight_map, base)
        elif kind == "w8":
            weights[projection] = dequant_w8(root, weight_map, base)
        elif kind == "w4":
            weights[projection] = dequant_w4(root, weight_map, base)
        else:
            raise ValueError(f"unsupported weight kind: {kind}")
    return weights


def move_weights(weights: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device=device) for name, value in weights.items()}


def rel_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    diff = torch.linalg.vector_norm((candidate - reference).float())
    base = torch.linalg.vector_norm(reference.float())
    return (diff / base).item()


def print_table(rows: list[dict[str, Any]], columns: list[str]) -> None:
    widths = {col: max(len(col), *(len(str(row[col])) for row in rows)) for col in columns}
    print("| " + " | ".join(col.ljust(widths[col]) for col in columns) + " |")
    print("| " + " | ".join("-" * widths[col] for col in columns) + " |")
    for row in rows:
        print("| " + " | ".join(str(row[col]).ljust(widths[col]) for col in columns) + " |")


def text_config(config: dict[str, Any]) -> dict[str, Any]:
    for key in ("text_config", "language_config", "model_config"):
        nested = config.get(key)
        if isinstance(nested, dict) and "hidden_size" in nested:
            return nested
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--w8", required=True, type=Path)
    parser.add_argument("--w4-sym", required=True, type=Path)
    parser.add_argument("--w4-asym", required=True, type=Path)
    parser.add_argument("--layers", default="0,1,11,31,63")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--rms-eps", type=float, default=1e-6)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    roots = {
        "bf16": args.source,
        "w8": args.w8,
        "w4_sym": args.w4_sym,
        "w4_asym": args.w4_asym,
    }
    weight_maps = {name: load_weight_map(root) for name, root in roots.items()}
    layers = parse_layers(args.layers)

    config = text_config(json.loads((args.source / "config.json").read_text()))
    hidden_size = int(config["hidden_size"])
    rms_eps = float(config.get("rms_norm_eps", args.rms_eps))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    rows: list[dict[str, Any]] = []
    raw: list[dict[str, Any]] = []
    for layer in layers:
        x = torch.randn(args.batch_size, hidden_size, generator=generator, dtype=torch.float32).to(device)

        ref_norm = (
            load_tensor(
                args.source,
                weight_maps["bf16"],
                tensor_name(layer, "post_attention_layernorm.weight"),
            )
            .float()
            .to(device)
        )
        ref_weights = move_weights(load_mlp_weights("bf16", args.source, weight_maps["bf16"], layer), device)
        ref = mlp_forward(x, ref_norm, ref_weights, rms_eps)

        rtn_weights = {name: fake_rtn_w4(weight.cpu()).to(device) for name, weight in ref_weights.items()}
        rtn = mlp_forward(x, ref_norm, rtn_weights, rms_eps)
        del rtn_weights

        result: dict[str, float] = {
            "layer": float(layer),
            "w4_rtn_sym_sim": rel_l2(ref, rtn),
        }
        del rtn

        for label, kind in (("w8a8_rtn", "w8"), ("w4_awq_sym", "w4"), ("w4_awq_asym", "w4")):
            root = roots["w8"] if label == "w8a8_rtn" else roots[label.replace("w4_awq_", "w4_")]
            weight_map = weight_maps["w8"] if label == "w8a8_rtn" else weight_maps[label.replace("w4_awq_", "w4_")]
            norm = (
                load_tensor(root, weight_map, tensor_name(layer, "post_attention_layernorm.weight")).float().to(device)
            )
            weights = move_weights(load_mlp_weights(kind, root, weight_map, layer), device)
            candidate = mlp_forward(x, norm, weights, rms_eps)
            result[label] = rel_l2(ref, candidate)
            del norm, weights, candidate

        raw.append(result)
        rows.append(
            {
                "layer": layer,
                "W8A8 RTN": f"{result['w8a8_rtn']:.6f}",
                "W4 RTN sim": f"{result['w4_rtn_sym_sim']:.6f}",
                "W4 AWQ sym": f"{result['w4_awq_sym']:.6f}",
                "W4 AWQ asym": f"{result['w4_awq_asym']:.6f}",
            }
        )
        del ref_weights, ref, ref_norm, x
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mean_row: dict[str, Any] = {"layer": "mean"}
    for out_col, raw_col in (
        ("W8A8 RTN", "w8a8_rtn"),
        ("W4 RTN sim", "w4_rtn_sym_sim"),
        ("W4 AWQ sym", "w4_awq_sym"),
        ("W4 AWQ asym", "w4_awq_asym"),
    ):
        mean_row[out_col] = f"{sum(item[raw_col] for item in raw) / len(raw):.6f}"
    rows.append(mean_row)

    metadata = {
        "source": str(args.source),
        "w8": str(args.w8),
        "w4_sym": str(args.w4_sym),
        "w4_asym": str(args.w4_asym),
        "layers": layers,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "rms_eps": rms_eps,
        "device": str(device),
    }
    print(json.dumps(metadata, indent=2, sort_keys=True))
    print_table(rows, ["layer", "W8A8 RTN", "W4 RTN sim", "W4 AWQ sym", "W4 AWQ asym"])

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps({"metadata": metadata, "rows": raw}, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
