"""Plot MLP quantization loss on real UltraChat hidden states.

This script uses the BF16 source checkpoint and a held-out UltraChat slice. It
captures real post-attention hidden states, then compares BF16 MLP outputs with
fake-quantized variants:

* W8: per-output-channel symmetric RTN weights
* W8G64: per-group symmetric RTN weights, group size 64
* W8G128: per-group symmetric RTN weights, group size 128
* W8B64: 2D blockwise symmetric RTN weights, block size 64x64
* W8B128: 2D blockwise symmetric RTN weights, block size 128x128
* A8: per-token symmetric dynamic activation quantization
* A8G64: per-token-group symmetric dynamic activation quantization, group size 64
* A8G128: per-token-group symmetric dynamic activation quantization, group size 128
* W8-A8: W8 weights + A8 activations
* W8G64-A8G64: W8G64 weights + A8G64 activations
* W8G128-A8G128: W8G128 weights + A8G128 activations
* W8B64-A8G64: W8B64 weights + A8G64 activations
* W8B128-A8G128: W8B128 weights + A8G128 activations
* W4AWQASYM: actual W4A16 AWQ ASYM checkpoint weights, group size 128

The plotted metric is absolute L2 over a fixed sampled token matrix for each
layer, not relative L2.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.quantize.producers.awq_w4a16 import patch_transformers_broken_torchvision
from scripts.quantize.utils.compare_mlp_quant_loss import dequant_w4

PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path, help="BF16 HF checkpoint")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--split", default="train_sft")
    parser.add_argument(
        "--code-eval",
        type=Path,
        default=None,
        help="If set, source code-domain inputs (prompt+turn0 response) from this BF16 eval_0.pt "
        "instead of streaming UltraChat. Used for the code-domain quant-loss analysis.",
    )
    parser.add_argument("--skip-prompts", type=int, default=256, help="Skip accepted prompts before sampling")
    parser.add_argument("--max-prompts", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-tokens-per-layer", type=int, default=256)
    parser.add_argument("--layers", default="all", help="'all' or comma-separated layer indices")
    parser.add_argument("--w8-group-sizes", default="64,128")
    parser.add_argument("--a8-group-sizes", default="64,128")
    parser.add_argument("--block-sizes", default="64", help="Comma-separated 2D weight block sizes for W8B")
    parser.add_argument("--w4-group-size", type=int, default=128)
    parser.add_argument(
        "--w4-awq-asym-path",
        type=Path,
        default=Path("checkpoints/quantized/AWQ/Qwen3.6-27B-AWQ-W4A16-asym-mlp"),
        help="Actual W4A16 AWQ ASYM checkpoint to include in the W4 plot",
    )
    parser.add_argument("--hidden-cache", type=Path, help="Optional torch cache for captured hidden samples")
    parser.add_argument("--no-reuse-hidden-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--rms-eps", type=float, default=1e-6)
    parser.add_argument("--compute-device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--proxy",
        default=os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY"),
        help="Optional HTTP(S) proxy for dataset streaming",
    )
    return parser.parse_args()


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


def text_config(config: dict[str, Any]) -> dict[str, Any]:
    for key in ("text_config", "language_config", "model_config"):
        nested = config.get(key)
        if isinstance(nested, dict) and "hidden_size" in nested:
            return nested
    return config


def layer_prefix(layer: int) -> str:
    return f"model.language_model.layers.{layer}"


def tensor_name(layer: int, suffix: str) -> str:
    return f"{layer_prefix(layer)}.{suffix}"


def parse_layers(value: str, num_layers: int) -> list[int]:
    if value == "all":
        return list(range(num_layers))
    return [int(item) for item in value.split(",") if item.strip()]


def parse_group_sizes(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def load_source_model(model_path: Path):
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    patch_transformers_broken_torchvision()
    from transformers import AutoModel, AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as exc:
        print(f"[quant-loss] AutoModelForImageTextToText failed ({exc!r}); falling back to AutoModel", flush=True)
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()
    return model, tokenizer


def build_ultrachat_prompts(args: argparse.Namespace, tokenizer: Any) -> list[dict[str, Any]]:
    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy

    from datasets import load_dataset

    dataset = load_dataset(args.dataset, split=args.split, streaming=True)
    rows: list[dict[str, Any]] = []
    accepted = 0
    for row in dataset:
        messages = row.get("messages")
        if not messages:
            continue
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if not isinstance(text, str) or not text.strip():
            continue
        if accepted < args.skip_prompts:
            accepted += 1
            continue
        rows.append(
            {
                "text": text,
                "source": args.dataset,
                "split": args.split,
                "prompt_id": row.get("prompt_id"),
                "stream_accepted_index": accepted,
            }
        )
        accepted += 1
        if len(rows) >= args.max_prompts:
            break
    if len(rows) < args.max_prompts:
        raise RuntimeError(f"only collected {len(rows)} prompts, expected {args.max_prompts}")
    return rows


def _code_eval_texts(code_eval: Path, min_response_chars: int = 200) -> list[dict[str, Any]]:
    """Yield code-domain text rows (prompt + turn0 response) from a BF16 eval_0.pt."""
    payload = torch.load(code_eval, map_location="cpu", weights_only=False)
    rows: list[dict[str, Any]] = []
    for sample in payload["samples"]:
        meta = sample.get("metadata") or {}
        turns = meta.get("turns") or []
        if not turns:
            continue
        turn = turns[0]
        response = turn.get("response") or ""
        if len(response) < min_response_chars:
            continue
        prompt = turn.get("prompt_snapshot") or sample.get("prompt") or ""
        rows.append(
            {
                "text": prompt + response,
                "source": str(code_eval),
                "sample_index": int(sample.get("index", len(rows))),
                "prompt_chars": len(prompt),
                "response_chars": len(response),
            }
        )
    return rows


def build_code_prompts(args: argparse.Namespace, tokenizer: Any) -> list[dict[str, Any]]:
    """Code-domain analog of build_ultrachat_prompts: real DrKernel BF16 rollouts."""
    rows = _code_eval_texts(args.code_eval)[: args.max_prompts]
    if len(rows) < args.max_prompts:
        raise RuntimeError(
            f"only collected {len(rows)} code rollouts from {args.code_eval}, expected {args.max_prompts}"
        )
    return rows


def build_packed_code_text(
    args: argparse.Namespace, tokenizer: Any, target_tokens: int
) -> tuple[str, list[dict[str, Any]], int]:
    """Pack code-domain BF16 rollouts into one long text until target_tokens (for long-context probes)."""
    separator = "\n\n<|endoftext|>\n\n"
    pieces: list[str] = []
    rows: list[dict[str, Any]] = []
    token_count = 0
    for row in _code_eval_texts(args.code_eval):
        pieces.append(row["text"])
        packed = separator.join(pieces)
        token_count = len(tokenizer(packed, add_special_tokens=False)["input_ids"])
        rows.append(
            {
                "sample_index": row["sample_index"],
                "piece_chars": len(row["text"]),
                "packed_tokens_after_append": token_count,
            }
        )
        if token_count >= target_tokens:
            return packed, rows, token_count
    raise RuntimeError(f"only built {token_count} tokens from {args.code_eval}, target={target_tokens}")


def find_module(model: torch.nn.Module, suffix: str) -> torch.nn.Module:
    candidates = (
        f"model.language_model.layers.{suffix}",
        f"language_model.layers.{suffix}",
        f"model.layers.{suffix}",
    )
    modules = dict(model.named_modules())
    for name in candidates:
        module = modules.get(name)
        if module is not None:
            return module

    suffix_with_dot = f".layers.{suffix}"
    for name, module in modules.items():
        if name.endswith(suffix_with_dot):
            return module
    raise KeyError(f"could not find module suffix layers.{suffix}")


def capture_real_hidden_samples(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    layers: list[int],
    max_seq_length: int,
    max_tokens_per_layer: int,
    seed: int,
) -> tuple[dict[int, torch.Tensor], dict[int, int], int]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length,
    )
    attention_mask = encoded["attention_mask"].detach().cpu().bool()
    total_valid_tokens = int(attention_mask.sum().item())

    captured: dict[int, torch.Tensor] = {}
    token_counts: dict[int, int] = {}
    handles = []

    def make_hook(layer: int):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + layer)

        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            hidden = inputs[0].detach().to("cpu", dtype=torch.float32)
            valid = hidden[attention_mask]
            if valid.ndim != 2:
                valid = valid.reshape(-1, hidden.shape[-1])
            token_counts[layer] = int(valid.shape[0])
            if valid.shape[0] > max_tokens_per_layer:
                indices = torch.randperm(valid.shape[0], generator=generator)[:max_tokens_per_layer]
                valid = valid[indices]
            captured[layer] = valid.contiguous()

        return hook

    for layer in layers:
        module = find_module(model, f"{layer}.post_attention_layernorm")
        handles.append(module.register_forward_pre_hook(make_hook(layer)))

    first_device = next(model.parameters()).device
    encoded = {name: value.to(first_device) for name, value in encoded.items()}
    with torch.no_grad():
        try:
            _ = model(**encoded, use_cache=False)
        except TypeError:
            _ = model(**encoded)

    for handle in handles:
        handle.remove()

    missing = sorted(set(layers) - set(captured))
    if missing:
        raise RuntimeError(f"did not capture hidden states for layers: {missing}")
    return captured, token_counts, total_valid_tokens


def rmsnorm_qwen(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return normed * (1.0 + weight)


def qdq_weight_int8_channel(weight: torch.Tensor) -> torch.Tensor:
    scale = weight.float().abs().amax(dim=1, keepdim=True) / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quant = torch.round(weight.float() / scale).clamp(-127, 127)
    return quant * scale


def qdq_weight_int8_group(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    rows, cols = weight.shape
    if cols % group_size != 0:
        raise ValueError(f"{tuple(weight.shape)} columns are not divisible by group_size={group_size}")
    grouped = weight.float().view(rows, cols // group_size, group_size)
    scale = grouped.abs().amax(dim=-1, keepdim=True) / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quant = torch.round(grouped / scale).clamp(-127, 127)
    return (quant * scale).reshape(rows, cols)


def qdq_int8_block_2d(x: torch.Tensor, block_size: int) -> torch.Tensor:
    rows, cols = x.shape
    matrix = x.float()
    pad_rows = (-rows) % block_size
    pad_cols = (-cols) % block_size
    if pad_rows or pad_cols:
        matrix = torch.nn.functional.pad(matrix, (0, pad_cols, 0, pad_rows))
    padded_rows, padded_cols = matrix.shape
    blocked = matrix.view(padded_rows // block_size, block_size, padded_cols // block_size, block_size)
    scale = blocked.abs().amax(dim=(1, 3), keepdim=True) / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quant = torch.round(blocked / scale).clamp(-127, 127)
    restored = (quant * scale).reshape(padded_rows, padded_cols)
    return restored[:rows, :cols].contiguous()


def qdq_weight_int8_block(weight: torch.Tensor, block_size: int) -> torch.Tensor:
    return qdq_int8_block_2d(weight, block_size)


def qdq_weight_int4_group(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    rows, cols = weight.shape
    if cols % group_size != 0:
        raise ValueError(f"{tuple(weight.shape)} columns are not divisible by group_size={group_size}")
    grouped = weight.float().view(rows, cols // group_size, group_size)
    scale = grouped.abs().amax(dim=-1, keepdim=True) / 7.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quant = torch.round(grouped / scale).clamp(-8, 7)
    return (quant * scale).reshape(rows, cols)


def qdq_activation_int8_token(x: torch.Tensor) -> torch.Tensor:
    scale = x.float().abs().amax(dim=-1, keepdim=True) / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quant = torch.round(x.float() / scale).clamp(-127, 127)
    return quant * scale


def qdq_activation_int8_token_group(x: torch.Tensor, group_size: int) -> torch.Tensor:
    cols = x.shape[-1]
    if cols % group_size != 0:
        raise ValueError(f"{tuple(x.shape)} last dimension is not divisible by group_size={group_size}")
    grouped = x.float().reshape(*x.shape[:-1], cols // group_size, group_size)
    scale = grouped.abs().amax(dim=-1, keepdim=True) / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quant = torch.round(grouped / scale).clamp(-127, 127)
    return (quant * scale).reshape_as(x)


def load_mlp_weights(root: Path, weight_map: dict[str, str], layer: int) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {}
    for projection in PROJECTIONS:
        weights[projection] = load_tensor(root, weight_map, tensor_name(layer, f"mlp.{projection}.weight")).float()
    return weights


def load_awq_w4_mlp_weights(
    root: Path, weight_map: dict[str, str], layer: int, device: torch.device
) -> dict[str, torch.Tensor]:
    weights: dict[str, torch.Tensor] = {}
    for projection in PROJECTIONS:
        base = tensor_name(layer, f"mlp.{projection}")
        weights[projection] = dequant_w4(root, weight_map, base).to(device=device)
    return weights


def to_device(weights: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device=device) for name, value in weights.items()}


def quantize_weights(
    weights: dict[str, torch.Tensor],
    weight_mode: str | None,
    w8_group_size: int,
    w4_group_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    quantized: dict[str, torch.Tensor] = {}
    for name, weight in weights.items():
        if weight_mode is None:
            quantized[name] = weight.to(device=device)
        elif weight_mode == "int8_channel":
            quantized[name] = qdq_weight_int8_channel(weight).to(device=device)
        elif weight_mode == "int8_group":
            quantized[name] = qdq_weight_int8_group(weight, w8_group_size).to(device=device)
        elif weight_mode == "int8_block":
            quantized[name] = qdq_weight_int8_block(weight, w8_group_size).to(device=device)
        elif weight_mode == "int4_group":
            quantized[name] = qdq_weight_int4_group(weight, w4_group_size).to(device=device)
        else:
            raise ValueError(f"unsupported weight_mode={weight_mode}")
    return quantized


def mlp_forward_variant(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    weights: dict[str, torch.Tensor],
    rms_eps: float,
    *,
    activation_mode: str | None,
    a8_group_size: int,
) -> torch.Tensor:
    h = rmsnorm_qwen(x, norm_weight, rms_eps)
    if activation_mode == "int8_token":
        h_for_gate_up = qdq_activation_int8_token(h)
    elif activation_mode == "int8_token_group":
        h_for_gate_up = qdq_activation_int8_token_group(h, a8_group_size)
    elif activation_mode is None:
        h_for_gate_up = h
    else:
        raise ValueError(f"unsupported activation_mode={activation_mode}")

    gate = torch.nn.functional.linear(h_for_gate_up, weights["gate_proj"])
    up = torch.nn.functional.linear(h_for_gate_up, weights["up_proj"])
    hidden = torch.nn.functional.silu(gate) * up
    if activation_mode == "int8_token":
        hidden = qdq_activation_int8_token(hidden)
    elif activation_mode == "int8_token_group":
        hidden = qdq_activation_int8_token_group(hidden, a8_group_size)
    return torch.nn.functional.linear(hidden, weights["down_proj"])


def loss_row(reference: torch.Tensor, candidate: torch.Tensor) -> tuple[float, float, float]:
    diff = (candidate - reference).float()
    abs_l2 = torch.linalg.vector_norm(diff).item()
    mean_token_abs_l2 = torch.linalg.vector_norm(diff, dim=-1).mean().item()
    max_token_abs_l2 = torch.linalg.vector_norm(diff, dim=-1).max().item()
    return abs_l2, mean_token_abs_l2, max_token_abs_l2


PREFERRED_LABELS = (
    "W8",
    "W8G64",
    "W8G128",
    "W8B64",
    "W8B128",
    "A8",
    "A8G64",
    "A8G128",
    "W8-A8",
    "W8G64-A8G64",
    "W8G128-A8G128",
    "W8B64-A8G64",
    "W8B128-A8G128",
    "W4AWQASYM",
)


def metric_labels(rows: list[dict[str, Any]]) -> list[str]:
    suffix = "_abs_l2"
    detail_suffixes = ("_mean_token_abs_l2", "_max_token_abs_l2")
    present = {
        key[: -len(suffix)]
        for row in rows
        for key in row
        if key.endswith(suffix) and not key.endswith(detail_suffixes)
    }
    labels = [label for label in PREFERRED_LABELS if label in present]
    labels.extend(sorted(present - set(labels)))
    return labels


def quant_group_or_block_size(label: str) -> int:
    for marker in ("G", "B"):
        if marker not in label:
            continue
        tail = label.split(marker, 1)[1]
        digits = []
        for char in tail:
            if not char.isdigit():
                break
            digits.append(char)
        if digits:
            return int("".join(digits))
    return 1


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = metric_labels(rows)
    columns = [
        "layer",
        "sampled_tokens",
        "total_valid_tokens",
    ]
    columns.extend(f"{label}_abs_l2" for label in labels)
    columns.extend(f"{label}_mean_token_abs_l2" for label in labels)
    columns.extend(f"{label}_max_token_abs_l2" for label in labels)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})


def plot_rows(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    plot_specs = [
        (
            "w_only_abs_l2.png",
            "W-only MLP Quantization Loss",
            (
                ("W8", "W8_abs_l2"),
                ("W8G64", "W8G64_abs_l2"),
                ("W8G128", "W8G128_abs_l2"),
                ("W8B64", "W8B64_abs_l2"),
                ("W8B128", "W8B128_abs_l2"),
            ),
        ),
        (
            "a_only_abs_l2.png",
            "A-only MLP Quantization Loss",
            (
                ("A8", "A8_abs_l2"),
                ("A8G64", "A8G64_abs_l2"),
                ("A8G128", "A8G128_abs_l2"),
            ),
        ),
        (
            "wa_only_abs_l2.png",
            "WA MLP Quantization Loss",
            (
                ("W8-A8", "W8-A8_abs_l2"),
                ("W8G64-A8G64", "W8G64-A8G64_abs_l2"),
                ("W8G128-A8G128", "W8G128-A8G128_abs_l2"),
                ("W8B64-A8G64", "W8B64-A8G64_abs_l2"),
                ("W8B128-A8G128", "W8B128-A8G128_abs_l2"),
            ),
        ),
        (
            "w4_abs_l2.png",
            "W4 MLP Quantization Loss",
            (
                ("W8-A8", "W8-A8_abs_l2"),
                ("W4A16 AWQ ASYM G128", "W4AWQASYM_abs_l2"),
            ),
        ),
    ]
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _plot_rows_with_pillow(output_dir, rows, plot_specs)
        return

    layers = [row["layer"] for row in rows]
    for filename, title, series in plot_specs:
        fig, ax = plt.subplots(figsize=(11, 5.8), dpi=180)
        for label, key in series:
            if key not in rows[0]:
                continue
            ax.plot(layers, [row[key] for row in rows], marker="o", markersize=2.8, linewidth=1.4, label=label)
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Absolute L2 ||Y_quant - Y_bf16||_2")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        ax.grid(True, which="minor", alpha=0.12)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename)
        plt.close(fig)


def _plot_rows_with_pillow(
    output_dir: Path, rows: list[dict[str, Any]], plot_specs: list[tuple[str, str, Any]]
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1800, 950
    left, right, top, bottom = 145, 55, 95, 130
    plot_w = width - left - right
    plot_h = height - top - bottom
    colors = (
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#393b79",
        "#637939",
    )
    font = ImageFont.load_default()
    layers = [int(row["layer"]) for row in rows]
    x_min, x_max = min(layers), max(layers)

    def x_pos(layer: int) -> float:
        if x_max == x_min:
            return left + plot_w / 2
        return left + (layer - x_min) / (x_max - x_min) * plot_w

    for filename, title, series in plot_specs:
        series = tuple((label, key) for label, key in series if key in rows[0])
        values = [float(row[key]) for _label, key in series for row in rows]
        positive_values = [value for value in values if value > 0]
        y_min = min(positive_values) * 0.8 if positive_values else 1e-6
        y_max = max(positive_values) * 1.25 if positive_values else 1.0
        log_min = math.log10(y_min)
        log_max = math.log10(y_max)

        def y_pos(
            value: float,
            *,
            y_min: float = y_min,
            log_min: float = log_min,
            log_max: float = log_max,
        ) -> float:
            value = max(value, y_min)
            return top + (log_max - math.log10(value)) / (log_max - log_min) * plot_h

        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)
        draw.text((left, 35), title, fill="black", font=font)
        draw.text((left, height - 55), "Layer", fill="black", font=font)
        draw.text((12, 35), "Absolute L2 ||Y_quant - Y_bf16||_2", fill="black", font=font)

        # Grid, axes, and y labels.
        for idx in range(6):
            value = 10 ** (log_min + (log_max - log_min) * idx / 5)
            y = y_pos(value)
            draw.line((left, y, left + plot_w, y), fill="#dddddd", width=1)
            draw.text((25, y - 6), f"{value:.2e}", fill="black", font=font)
        draw.line((left, top, left, top + plot_h), fill="black", width=2)
        draw.line((left, top + plot_h, left + plot_w, top + plot_h), fill="black", width=2)

        x_ticks = sorted(set([0, 8, 16, 24, 32, 40, 48, 56, 63]) & set(range(x_min, x_max + 1)))
        for layer in x_ticks:
            x = x_pos(layer)
            draw.line((x, top + plot_h, x, top + plot_h + 6), fill="black", width=1)
            draw.text((x - 8, top + plot_h + 12), str(layer), fill="black", font=font)

        legend_x = left + plot_w - 260
        legend_y = top + 20
        for series_idx, (label, key) in enumerate(series):
            color = colors[series_idx % len(colors)]
            points = [(x_pos(int(row["layer"])), y_pos(float(row[key]))) for row in rows]
            if len(points) >= 2:
                draw.line(points, fill=color, width=4)
            for x, y in points:
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
            y_legend = legend_y + series_idx * 28
            draw.line((legend_x, y_legend + 7, legend_x + 35, y_legend + 7), fill=color, width=4)
            draw.text((legend_x + 45, y_legend), label, fill="black", font=font)

        image.save(output_dir / filename)


def main() -> None:
    args = parse_args()
    if args.compute_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--compute-device=cuda requested but CUDA is not available")
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = text_config(json.loads((args.model_path / "config.json").read_text()))
    layers = parse_layers(args.layers, int(config["num_hidden_layers"]))
    w8_group_sizes = parse_group_sizes(args.w8_group_sizes)
    a8_group_sizes = parse_group_sizes(args.a8_group_sizes)
    block_sizes = parse_group_sizes(args.block_sizes)
    rms_eps = float(config.get("rms_norm_eps", args.rms_eps))
    hidden_cache = args.hidden_cache or args.output_dir / "hidden_samples.pt"
    w4_awq_asym_path = args.w4_awq_asym_path if args.w4_awq_asym_path and args.w4_awq_asym_path.exists() else None

    captured: dict[int, torch.Tensor]
    token_counts: dict[int, int]
    total_valid_tokens: int
    if hidden_cache.exists() and not args.no_reuse_hidden_cache:
        print(f"[quant-loss] loading hidden cache: {hidden_cache}", flush=True)
        cache = torch.load(hidden_cache, map_location="cpu", weights_only=False)
        cache_meta = cache["metadata"]
        if cache_meta["layers"] != layers:
            raise ValueError(f"hidden cache layers mismatch: {cache_meta['layers']} != {layers}")
        captured = {int(k): v for k, v in cache["captured"].items()}
        token_counts = {int(k): int(v) for k, v in cache["token_counts"].items()}
        total_valid_tokens = int(cache["total_valid_tokens"])
    else:
        print(f"[quant-loss] loading source model: {args.model_path}", flush=True)
        model, tokenizer = load_source_model(args.model_path)
        prompt_rows = (
            build_code_prompts(args, tokenizer) if args.code_eval else build_ultrachat_prompts(args, tokenizer)
        )
        prompt_path = args.output_dir / "calibration_prompts.jsonl"
        with prompt_path.open("w") as handle:
            for row in prompt_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(
            f"[quant-loss] wrote held-out UltraChat prompts: {prompt_path} "
            f"(skip_prompts={args.skip_prompts}, count={len(prompt_rows)})",
            flush=True,
        )

        print(
            f"[quant-loss] capturing layers={len(layers)}, max_tokens_per_layer={args.max_tokens_per_layer}",
            flush=True,
        )
        captured, token_counts, total_valid_tokens = capture_real_hidden_samples(
            model,
            tokenizer,
            [row["text"] for row in prompt_rows],
            layers,
            args.max_seq_length,
            args.max_tokens_per_layer,
            args.seed,
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        torch.save(
            {
                "metadata": {
                    "model_path": str(args.model_path),
                    "dataset": args.dataset,
                    "split": args.split,
                    "skip_prompts": args.skip_prompts,
                    "max_prompts": args.max_prompts,
                    "max_seq_length": args.max_seq_length,
                    "max_tokens_per_layer": args.max_tokens_per_layer,
                    "layers": layers,
                    "seed": args.seed,
                },
                "captured": captured,
                "token_counts": token_counts,
                "total_valid_tokens": total_valid_tokens,
                "prompt_rows": prompt_rows,
            },
            hidden_cache,
        )
        print(f"[quant-loss] wrote hidden cache: {hidden_cache}", flush=True)

    device = torch.device(args.compute_device)
    weight_map = load_weight_map(args.model_path)
    w4_awq_weight_map = load_weight_map(w4_awq_asym_path) if w4_awq_asym_path is not None else None
    rows: list[dict[str, Any]] = []
    for layer in layers:
        print(f"[quant-loss] layer {layer}", flush=True)
        x = captured[layer].to(device=device)
        norm_weight = load_tensor(args.model_path, weight_map, tensor_name(layer, "post_attention_layernorm.weight"))
        norm_weight = norm_weight.float().to(device=device)
        base_weights_cpu = load_mlp_weights(args.model_path, weight_map, layer)

        ref_weights = to_device(base_weights_cpu, device)
        with torch.no_grad():
            reference = mlp_forward_variant(
                x,
                norm_weight,
                ref_weights,
                rms_eps,
                activation_mode=None,
                a8_group_size=1,
            )

        variants = {
            "W8": ("int8_channel", None),
            "A8": (None, "int8_token"),
            "W8-A8": ("int8_channel", "int8_token"),
        }
        for group_size in w8_group_sizes:
            variants[f"W8G{group_size}"] = ("int8_group", None)
        for group_size in a8_group_sizes:
            variants[f"A8G{group_size}"] = (None, "int8_token_group")
        for block_size in block_sizes:
            variants[f"W8B{block_size}"] = ("int8_block", None)
        for group_size in sorted(set(w8_group_sizes) & set(a8_group_sizes)):
            variants[f"W8G{group_size}-A8G{group_size}"] = ("int8_group", "int8_token_group")
        for block_size in sorted(set(block_sizes) & set(a8_group_sizes)):
            variants[f"W8B{block_size}-A8G{block_size}"] = ("int8_block", "int8_token_group")
        if w4_awq_asym_path is not None:
            variants["W4AWQASYM"] = ("w4_awq_asym", None)

        row: dict[str, Any] = {
            "layer": layer,
            "sampled_tokens": int(x.shape[0]),
            "total_valid_tokens": int(token_counts[layer]),
        }
        for label, (weight_mode, activation_mode) in variants.items():
            group_size = quant_group_or_block_size(label)
            variant_norm_weight = norm_weight
            if weight_mode == "w4_awq_asym":
                assert w4_awq_asym_path is not None and w4_awq_weight_map is not None
                weights = load_awq_w4_mlp_weights(w4_awq_asym_path, w4_awq_weight_map, layer, device)
                variant_norm_weight = (
                    load_tensor(
                        w4_awq_asym_path,
                        w4_awq_weight_map,
                        tensor_name(layer, "post_attention_layernorm.weight"),
                    )
                    .float()
                    .to(device=device)
                )
            else:
                weights = quantize_weights(base_weights_cpu, weight_mode, group_size, args.w4_group_size, device)
            with torch.no_grad():
                candidate = mlp_forward_variant(
                    x,
                    variant_norm_weight,
                    weights,
                    rms_eps,
                    activation_mode=activation_mode,
                    a8_group_size=group_size,
                )
            abs_l2, mean_token_abs_l2, max_token_abs_l2 = loss_row(reference, candidate)
            row[f"{label}_abs_l2"] = abs_l2
            row[f"{label}_mean_token_abs_l2"] = mean_token_abs_l2
            row[f"{label}_max_token_abs_l2"] = max_token_abs_l2
            del weights, candidate
            if variant_norm_weight is not norm_weight:
                del variant_norm_weight
            if device.type == "cuda":
                torch.cuda.empty_cache()

        rows.append(row)
        del x, norm_weight, base_weights_cpu, ref_weights, reference
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    csv_path = args.output_dir / "quant_loss_by_layer.csv"
    json_path = args.output_dir / "quant_loss_by_layer.json"
    metadata = {
        "model_path": str(args.model_path),
        "dataset": args.dataset,
        "split": args.split,
        "skip_prompts": args.skip_prompts,
        "max_prompts": args.max_prompts,
        "max_seq_length": args.max_seq_length,
        "max_tokens_per_layer": args.max_tokens_per_layer,
        "total_valid_tokens_before_sampling": total_valid_tokens,
        "hidden_cache": str(hidden_cache),
        "layers": layers,
        "w8": "per-output-channel symmetric RTN",
        "w8_group_sizes": w8_group_sizes,
        "w8_block_sizes": block_sizes,
        "a8": "per-token symmetric dynamic activation quantization before each Linear",
        "a8_group_sizes": a8_group_sizes,
        "w4": f"per-group symmetric RTN, group_size={args.w4_group_size}",
        "w4_awq_asym_path": str(w4_awq_asym_path) if w4_awq_asym_path is not None else None,
        "w4_awq_asym": "actual W4A16 AWQ ASYM checkpoint weights, group_size=128, using checkpoint post_attention_layernorm",
        "code_eval": str(args.code_eval) if args.code_eval else None,
        "input_source": "code_eval(BF16 DrKernel rollouts)" if args.code_eval else "ultrachat",
        "metric": "absolute L2 over the sampled output matrix; mean/max per-token absolute L2 also saved",
        "rms_eps": rms_eps,
        "seed": args.seed,
        "compute_device": args.compute_device,
    }
    json_path.write_text(json.dumps({"metadata": metadata, "rows": rows}, indent=2, sort_keys=True) + "\n")
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    write_csv(csv_path, rows)
    plot_rows(args.output_dir, rows)

    print(f"[quant-loss] wrote {csv_path}", flush=True)
    print(f"[quant-loss] wrote {json_path}", flush=True)
    print(f"[quant-loss] wrote {args.output_dir / 'w_only_abs_l2.png'}", flush=True)
    print(f"[quant-loss] wrote {args.output_dir / 'a_only_abs_l2.png'}", flush=True)
    print(f"[quant-loss] wrote {args.output_dir / 'wa_only_abs_l2.png'}", flush=True)
    print(f"[quant-loss] wrote {args.output_dir / 'w4_abs_l2.png'}", flush=True)


if __name__ == "__main__":
    main()
