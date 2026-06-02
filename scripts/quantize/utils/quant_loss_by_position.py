"""Compare MLP quantization loss by token-position bucket.

This is a focused diagnostic for checking whether long-context positions make
W4A16 AWQ ASYM degrade more than W8A8 on real UltraChat hidden states.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.quantize.utils.plot_real_calib_quant_loss import (
    build_code_prompts,
    build_ultrachat_prompts,
    find_module,
    load_awq_w4_mlp_weights,
    load_mlp_weights,
    load_source_model,
    load_tensor,
    load_weight_map,
    loss_row,
    mlp_forward_variant,
    parse_layers,
    quantize_weights,
    tensor_name,
    text_config,
    to_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    parser.add_argument("--split", default="train_sft")
    parser.add_argument(
        "--code-eval",
        type=Path,
        default=None,
        help="If set, source code-domain inputs from this BF16 eval_0.pt instead of UltraChat.",
    )
    parser.add_argument("--skip-prompts", type=int, default=256)
    parser.add_argument("--max-prompts", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--layers", default="0,1,11,31,63")
    parser.add_argument("--buckets", default="0:512,512:1024,1024:2048,2048:4096")
    parser.add_argument("--max-tokens-per-bucket", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260601)
    parser.add_argument("--compute-device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--w4-awq-asym-path",
        type=Path,
        default=Path("checkpoints/quantized/AWQ/Qwen3.6-27B-AWQ-W4A16-asym-mlp"),
    )
    parser.add_argument(
        "--proxy",
        default=os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY"),
    )
    return parser.parse_args()


def parse_buckets(spec: str) -> list[tuple[int, int, str]]:
    buckets: list[tuple[int, int, str]] = []
    for item in spec.split(","):
        left, right = item.split(":", 1)
        start, end = int(left), int(right)
        if end <= start:
            raise ValueError(f"invalid bucket {item!r}")
        buckets.append((start, end, f"{start}-{end}"))
    return buckets


def capture_bucketed_hidden(
    model: torch.nn.Module,
    tokenizer,
    prompts: list[str],
    layers: list[int],
    buckets: list[tuple[int, int, str]],
    max_seq_length: int,
    max_tokens_per_bucket: int,
    seed: int,
) -> tuple[dict[tuple[int, str], torch.Tensor], dict[tuple[int, str], int], int]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_seq_length,
    )
    attention_mask = encoded["attention_mask"].detach().cpu().bool()
    positions = torch.arange(attention_mask.shape[1]).unsqueeze(0).expand_as(attention_mask)
    bucket_masks = {label: attention_mask & (positions >= start) & (positions < end) for start, end, label in buckets}
    total_valid_tokens = int(attention_mask.sum().item())

    captured: dict[tuple[int, str], torch.Tensor] = {}
    token_counts: dict[tuple[int, str], int] = {}
    handles = []

    def make_hook(layer: int):
        def hook(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            hidden = inputs[0].detach().to("cpu", dtype=torch.float32)
            for bucket_index, (_start, _end, label) in enumerate(buckets):
                valid = hidden[bucket_masks[label]]
                if valid.ndim != 2:
                    valid = valid.reshape(-1, hidden.shape[-1])
                token_counts[(layer, label)] = int(valid.shape[0])
                if valid.shape[0] > max_tokens_per_bucket:
                    generator = torch.Generator(device="cpu")
                    generator.manual_seed(seed + layer * 1009 + bucket_index)
                    indices = torch.randperm(valid.shape[0], generator=generator)[:max_tokens_per_bucket]
                    valid = valid[indices]
                captured[(layer, label)] = valid.contiguous()

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

    return captured, token_counts, total_valid_tokens


def main() -> None:
    args = parse_args()
    if args.compute_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--compute-device=cuda requested but CUDA is not available")
    if args.proxy:
        os.environ["HTTP_PROXY"] = args.proxy
        os.environ["HTTPS_PROXY"] = args.proxy
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = text_config(json.loads((args.model_path / "config.json").read_text()))
    layers = parse_layers(args.layers, int(config["num_hidden_layers"]))
    buckets = parse_buckets(args.buckets)
    rms_eps = float(config.get("rms_norm_eps", 1e-6))

    print(f"[position-loss] loading source model: {args.model_path}", flush=True)
    model, tokenizer = load_source_model(args.model_path)
    prompt_rows = build_code_prompts(args, tokenizer) if args.code_eval else build_ultrachat_prompts(args, tokenizer)
    prompt_path = args.output_dir / "calibration_prompts.jsonl"
    with prompt_path.open("w") as handle:
        for row in prompt_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[position-loss] wrote prompts: {prompt_path}", flush=True)

    print(f"[position-loss] capturing layers={layers}, buckets={[b[2] for b in buckets]}", flush=True)
    captured, token_counts, total_valid_tokens = capture_bucketed_hidden(
        model,
        tokenizer,
        [row["text"] for row in prompt_rows],
        layers,
        buckets,
        args.max_seq_length,
        args.max_tokens_per_bucket,
        args.seed,
    )
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    device = torch.device(args.compute_device)
    source_weight_map = load_weight_map(args.model_path)
    awq_weight_map = load_weight_map(args.w4_awq_asym_path)
    rows = []
    for layer in layers:
        print(f"[position-loss] layer {layer}", flush=True)
        norm_weight = load_tensor(
            args.model_path, source_weight_map, tensor_name(layer, "post_attention_layernorm.weight")
        )
        norm_weight = norm_weight.float().to(device=device)
        base_weights_cpu = load_mlp_weights(args.model_path, source_weight_map, layer)
        ref_weights = to_device(base_weights_cpu, device)
        w8a8_weights = quantize_weights(base_weights_cpu, "int8_channel", 1, 128, device)
        w8b128_weights = quantize_weights(base_weights_cpu, "int8_block", 128, 128, device)
        awq_weights = load_awq_w4_mlp_weights(args.w4_awq_asym_path, awq_weight_map, layer, device)
        awq_norm_weight = (
            load_tensor(
                args.w4_awq_asym_path,
                awq_weight_map,
                tensor_name(layer, "post_attention_layernorm.weight"),
            )
            .float()
            .to(device=device)
        )

        for _start, _end, label in buckets:
            x_cpu = captured[(layer, label)]
            if x_cpu.numel() == 0:
                continue
            x = x_cpu.to(device=device)
            with torch.no_grad():
                reference = mlp_forward_variant(
                    x, norm_weight, ref_weights, rms_eps, activation_mode=None, a8_group_size=1
                )
                w8a8 = mlp_forward_variant(
                    x, norm_weight, w8a8_weights, rms_eps, activation_mode="int8_token", a8_group_size=1
                )
                w8b128 = mlp_forward_variant(
                    x,
                    norm_weight,
                    w8b128_weights,
                    rms_eps,
                    activation_mode="int8_token_group",
                    a8_group_size=128,
                )
                awq = mlp_forward_variant(
                    x, awq_norm_weight, awq_weights, rms_eps, activation_mode=None, a8_group_size=1
                )
            row = {
                "layer": layer,
                "bucket": label,
                "sampled_tokens": int(x.shape[0]),
                "total_bucket_tokens": int(token_counts[(layer, label)]),
            }
            for metric_name, candidate in (("W8-A8", w8a8), ("W8B128-A8G128", w8b128), ("W4AWQASYM", awq)):
                abs_l2, mean_token_abs_l2, max_token_abs_l2 = loss_row(reference, candidate)
                row[f"{metric_name}_abs_l2"] = abs_l2
                row[f"{metric_name}_mean_token_abs_l2"] = mean_token_abs_l2
                row[f"{metric_name}_max_token_abs_l2"] = max_token_abs_l2
            rows.append(row)
            del x, reference, w8a8, w8b128, awq
            if device.type == "cuda":
                torch.cuda.empty_cache()

        del norm_weight, base_weights_cpu, ref_weights, w8a8_weights, w8b128_weights, awq_weights, awq_norm_weight
        gc.collect()

    csv_path = args.output_dir / "quant_loss_by_position.csv"
    columns = [
        "layer",
        "bucket",
        "sampled_tokens",
        "total_bucket_tokens",
        "W8-A8_abs_l2",
        "W8B128-A8G128_abs_l2",
        "W4AWQASYM_abs_l2",
        "W8-A8_mean_token_abs_l2",
        "W8B128-A8G128_mean_token_abs_l2",
        "W4AWQASYM_mean_token_abs_l2",
        "W8-A8_max_token_abs_l2",
        "W8B128-A8G128_max_token_abs_l2",
        "W4AWQASYM_max_token_abs_l2",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "model_path": str(args.model_path),
        "w4_awq_asym_path": str(args.w4_awq_asym_path),
        "dataset": args.dataset,
        "split": args.split,
        "skip_prompts": args.skip_prompts,
        "max_prompts": args.max_prompts,
        "max_seq_length": args.max_seq_length,
        "layers": layers,
        "buckets": [label for _start, _end, label in buckets],
        "max_tokens_per_bucket": args.max_tokens_per_bucket,
        "total_valid_tokens_before_sampling": total_valid_tokens,
        "metric": "absolute L2 over sampled MLP output matrix by token-position bucket",
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"[position-loss] wrote {csv_path}", flush=True)


if __name__ == "__main__":
    main()
