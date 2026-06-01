"""Compare MLP quantization loss on real BF16 hidden states.

This is a stronger pre-eval sanity check than random hidden vectors. It runs a
small BF16 forward pass, captures the input to selected
``post_attention_layernorm`` modules, then compares:

    MLP_variant(RMSNorm_variant(real_hidden)) vs
    MLP_bf16(RMSNorm_bf16(real_hidden))

It still does not start SGLang, generate rollout samples, or call the reward
server.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.quantize.producers.awq_w4a16 import patch_transformers_broken_torchvision
from scripts.quantize.utils.compare_mlp_quant_loss import (
    load_mlp_weights,
    load_tensor,
    load_weight_map,
    mlp_forward,
    move_weights,
    parse_layers,
    rel_l2,
    tensor_name,
    text_config,
)

DEFAULT_PROMPTS = [
    "Write a CUDA kernel that adds two float arrays and explain the indexing.",
    "Optimize this PyTorch operation into a fused CUDA kernel: y = relu(x * scale + bias).",
    "Given a matrix multiplication kernel, identify the memory coalescing issue and propose a fix.",
    "Implement a warp-level reduction for 32 float values using CUDA shuffle instructions.",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Source BF16 HF checkpoint")
    parser.add_argument("--w8", required=True, type=Path, help="W8A8 checkpoint used as reference scale")
    parser.add_argument("--w4-sym", required=True, type=Path, help="Symmetric W4A16 checkpoint")
    parser.add_argument("--w4-asym", required=True, type=Path, help="Asymmetric W4A16 checkpoint")
    parser.add_argument("--layers", default="0,1,11,31,63")
    parser.add_argument("--prompt-jsonl", type=Path, help="Optional JSONL with a text-like field")
    parser.add_argument("--prompt-field", default="text")
    parser.add_argument("--max-prompts", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--max-tokens-per-layer", type=int, default=128)
    parser.add_argument("--rms-eps", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--compute-device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def load_prompts(path: Path | None, field: str, limit: int) -> list[str]:
    if path is None:
        return DEFAULT_PROMPTS[:limit]

    prompts: list[str] = []
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            value = row.get(field)
            if isinstance(value, str) and value.strip():
                prompts.append(value)
            if len(prompts) >= limit:
                break
    if not prompts:
        raise ValueError(f"no prompts found in {path} field={field!r}")
    return prompts


def load_source_model(model_path: Path):
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")
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
        print(f"[real-hidden] AutoModelForImageTextToText failed ({exc!r}); falling back to AutoModel", flush=True)
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()
    return model, tokenizer


def find_module(model: torch.nn.Module, suffix: str) -> torch.nn.Module:
    exact_candidates = [
        f"model.language_model.layers.{suffix}",
        f"language_model.layers.{suffix}",
        f"model.layers.{suffix}",
    ]
    modules = dict(model.named_modules())
    for name in exact_candidates:
        module = modules.get(name)
        if module is not None:
            return module

    suffix_with_dot = f".layers.{suffix}"
    for name, module in modules.items():
        if name.endswith(suffix_with_dot):
            return module
    raise KeyError(f"could not find module suffix layers.{suffix}")


def capture_real_hidden(
    model: torch.nn.Module,
    tokenizer: Any,
    prompts: list[str],
    layers: list[int],
    max_seq_length: int,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            captured[layer] = inputs[0].detach().to("cpu", dtype=torch.float32)

        return hook

    for layer in layers:
        module = find_module(model, f"{layer}.post_attention_layernorm")
        handles.append(module.register_forward_pre_hook(make_hook(layer)))

    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length,
    )
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
    return captured, encoded["attention_mask"].detach().cpu().bool()


def sample_valid_tokens(
    hidden: torch.Tensor,
    attention_mask: torch.Tensor,
    max_tokens: int,
    generator: torch.Generator,
) -> torch.Tensor:
    valid = hidden[attention_mask]
    if valid.ndim != 2:
        valid = valid.reshape(-1, hidden.shape[-1])
    if valid.shape[0] > max_tokens:
        indices = torch.randperm(valid.shape[0], generator=generator)[:max_tokens]
        valid = valid[indices]
    return valid


def print_table(rows: list[dict[str, Any]], columns: list[str]) -> None:
    widths = {col: max(len(col), *(len(str(row[col])) for row in rows)) for col in columns}
    print("| " + " | ".join(col.ljust(widths[col]) for col in columns) + " |")
    print("| " + " | ".join("-" * widths[col] for col in columns) + " |")
    for row in rows:
        print("| " + " | ".join(str(row[col]).ljust(widths[col]) for col in columns) + " |")


def main() -> None:
    args = parse_args()
    if args.compute_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--compute-device=cuda requested but CUDA is not available")

    layers = parse_layers(args.layers)
    prompts = load_prompts(args.prompt_jsonl, args.prompt_field, args.max_prompts)

    print(f"[real-hidden] loading BF16 source model: {args.source}", flush=True)
    model, tokenizer = load_source_model(args.source)
    print(f"[real-hidden] capturing layers={layers}, prompts={len(prompts)}", flush=True)
    captured, attention_mask = capture_real_hidden(model, tokenizer, prompts, layers, args.max_seq_length)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    roots = {
        "bf16": args.source,
        "w8": args.w8,
        "w4_sym": args.w4_sym,
        "w4_asym": args.w4_asym,
    }
    weight_maps = {name: load_weight_map(root) for name, root in roots.items()}
    config = text_config(json.loads((args.source / "config.json").read_text()))
    rms_eps = float(config.get("rms_norm_eps", args.rms_eps))
    device = torch.device(args.compute_device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)

    raw_rows: list[dict[str, Any]] = []
    display_rows: list[dict[str, Any]] = []
    for layer in layers:
        x = sample_valid_tokens(captured[layer], attention_mask, args.max_tokens_per_layer, generator).to(device)
        ref_norm = (
            load_tensor(args.source, weight_maps["bf16"], tensor_name(layer, "post_attention_layernorm.weight"))
            .float()
            .to(device)
        )
        ref_weights = move_weights(load_mlp_weights("bf16", args.source, weight_maps["bf16"], layer), device)
        ref = mlp_forward(x, ref_norm, ref_weights, rms_eps)

        result: dict[str, Any] = {"layer": layer, "num_tokens": int(x.shape[0])}
        for label, root_key, kind in (
            ("w8a8_rtn", "w8", "w8"),
            ("w4_awq_sym", "w4_sym", "w4"),
            ("w4_awq_asym", "w4_asym", "w4"),
        ):
            root = roots[root_key]
            weight_map = weight_maps[root_key]
            norm = (
                load_tensor(root, weight_map, tensor_name(layer, "post_attention_layernorm.weight")).float().to(device)
            )
            weights = move_weights(load_mlp_weights(kind, root, weight_map, layer), device)
            candidate = mlp_forward(x, norm, weights, rms_eps)
            result[label] = rel_l2(ref, candidate)
            del norm, weights, candidate

        raw_rows.append(result)
        display_rows.append(
            {
                "layer": layer,
                "tokens": result["num_tokens"],
                "W8A8 RTN": f"{result['w8a8_rtn']:.6f}",
                "W4 AWQ sym": f"{result['w4_awq_sym']:.6f}",
                "W4 AWQ asym": f"{result['w4_awq_asym']:.6f}",
            }
        )
        del x, ref_norm, ref_weights, ref
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mean_row: dict[str, Any] = {"layer": "mean", "tokens": "-"}
    for out_col, raw_col in (
        ("W8A8 RTN", "w8a8_rtn"),
        ("W4 AWQ sym", "w4_awq_sym"),
        ("W4 AWQ asym", "w4_awq_asym"),
    ):
        mean_row[out_col] = f"{sum(float(item[raw_col]) for item in raw_rows) / len(raw_rows):.6f}"
    display_rows.append(mean_row)

    metadata = {
        "source": str(args.source),
        "w8": str(args.w8),
        "w4_sym": str(args.w4_sym),
        "w4_asym": str(args.w4_asym),
        "layers": layers,
        "max_prompts": args.max_prompts,
        "max_seq_length": args.max_seq_length,
        "max_tokens_per_layer": args.max_tokens_per_layer,
        "rms_eps": rms_eps,
        "compute_device": str(device),
        "prompts": prompts,
    }
    print(json.dumps({key: value for key, value in metadata.items() if key != "prompts"}, indent=2, sort_keys=True))
    print_table(display_rows, ["layer", "tokens", "W8A8 RTN", "W4 AWQ sym", "W4 AWQ asym"])

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({"metadata": metadata, "rows": raw_rows}, indent=2, sort_keys=True) + "\n"
        )


if __name__ == "__main__":
    main()
