#!/usr/bin/env python3
"""Probe whether W4 AWQ errors propagate into code-token logits.

This diagnostic uses real DrKernel W4 full-eval failures.  For responses that
contain the wrong TVM FFI token ``kTVMFloat``, it constructs the same
prompt+prefix immediately before that token and compares:

* hidden-state drift after selected transformer blocks
* sequence logprob for the wrong candidate ``kTVMFloat``
* sequence logprob for the expected candidate ``kTVMFFIFloat``

The model is loaded once from the BF16 source checkpoint.  Then MLP modules are
patched in-place with dequantized W8A8-MLP or W4A16-AWQ weights, so all variants
share the same non-MLP BF16 path.  This is a controlled Transformers-side probe,
not a replacement for full SGLang eval.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
import types
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.quantize.producers.awq_w4a16 import patch_transformers_broken_torchvision
from scripts.quantize.utils.compare_mlp_quant_loss import (
    dequant_w4,
    dequant_w8,
    load_bf16_weight,
    load_tensor,
    load_weight_map,
    tensor_name,
    text_config,
)


DEFAULT_SOURCE = Path("/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B")
DEFAULT_W8_MLP = Path("checkpoints/quantized/SmoothQuant/Qwen3.6-27B-SQ-W8A8-RTN-mlp-mtp-a0p5-ultrachat")
DEFAULT_W4_ASYM = Path("checkpoints/quantized/AWQ/Qwen3.6-27B-AWQ-W4A16-asym-mlp")
DEFAULT_EVAL = Path(
    "checkpoints/Qwen3.6-27B-AWQ-W4A16-asym-mlp/"
    "20260531_233411_awq.w4a16.asym_mlp.sglzpfix.100x8.eagle.rm39_ctx65536_n8_summ1600/"
    "dumps/rollout_data/eval_0.pt"
)
DEFAULT_OUT = Path("checkpoints/quantized/analysis/w4_awq_error_propagation_20260601")
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--w8-mlp", type=Path, default=DEFAULT_W8_MLP)
    parser.add_argument("--w4-asym", type=Path, default=DEFAULT_W4_ASYM)
    parser.add_argument("--w4-eval", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-examples", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--hidden-tail-tokens", type=int, default=128)
    parser.add_argument("--layers", default="0,1,3,7,15,31,47,63")
    parser.add_argument("--wrong-token", default="kTVMFloat")
    parser.add_argument("--expected-token", default="kTVMFFIFloat")
    parser.add_argument("--skip-candidate-logprobs", action="store_true")
    parser.add_argument("--rms-eps", type=float, default=1e-6)
    return parser.parse_args()


def parse_layers(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def load_model_and_tokenizer(model_path: Path):
    patch_transformers_broken_torchvision()
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


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


def set_parameter(module: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    old = getattr(module, name)
    device = old.device
    dtype = old.dtype
    setattr(module, name, torch.nn.Parameter(value.to(device=device, dtype=dtype), requires_grad=False))


def fake_a8_per_token(x: torch.Tensor) -> torch.Tensor:
    x_float = x.float()
    scale = x_float.abs().amax(dim=-1, keepdim=True) / 127.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quant = torch.round(x_float / scale).clamp(-128, 127)
    return (quant * scale).to(dtype=x.dtype)


def make_mlp_forward(use_a8: bool):
    def forward(self: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        linear_in = fake_a8_per_token(x) if use_a8 else x
        gate = self.gate_proj(linear_in)
        up = self.up_proj(linear_in)
        hidden = torch.nn.functional.silu(gate) * up
        down_in = fake_a8_per_token(hidden) if use_a8 else hidden
        return self.down_proj(down_in)

    return forward


def patch_mlp_variant(
    model: torch.nn.Module,
    root: Path,
    weight_map: dict[str, str],
    *,
    kind: str,
    num_layers: int,
    use_a8: bool,
) -> None:
    for layer in range(num_layers):
        norm = find_module(model, f"{layer}.post_attention_layernorm")
        norm_name = tensor_name(layer, "post_attention_layernorm.weight")
        set_parameter(norm, "weight", load_tensor(root, weight_map, norm_name).float())

        mlp = find_module(model, f"{layer}.mlp")
        mlp.forward = types.MethodType(make_mlp_forward(use_a8), mlp)
        for projection in PROJECTIONS:
            linear = find_module(model, f"{layer}.mlp.{projection}")
            base = tensor_name(layer, f"mlp.{projection}")
            if kind == "bf16":
                weight = load_bf16_weight(root, weight_map, base)
            elif kind == "w8":
                weight = dequant_w8(root, weight_map, base)
            elif kind == "w4":
                weight = dequant_w4(root, weight_map, base)
            else:
                raise ValueError(f"unsupported variant kind: {kind}")
            set_parameter(linear, "weight", weight)
            del weight
        if torch.cuda.is_available() and layer % 8 == 7:
            torch.cuda.empty_cache()


def select_error_cases(
    eval_path: Path,
    tokenizer: Any,
    wrong_token: str,
    expected_token: str,
    max_examples: int,
    max_seq_length: int,
) -> list[dict[str, Any]]:
    payload = torch.load(eval_path, map_location="cpu", weights_only=False)
    cases: list[dict[str, Any]] = []
    for sample in payload["samples"]:
        meta = sample.get("metadata") or {}
        turns = meta.get("turns") or []
        if not turns:
            continue
        turn = turns[0]
        response = turn.get("response") or ""
        pos = response.find(wrong_token)
        if pos < 0:
            continue
        prompt = turn.get("prompt_snapshot") or sample.get("prompt") or ""
        prefix = prompt + response[:pos]
        wrong_ids = tokenizer.encode(wrong_token, add_special_tokens=False)
        expected_ids = tokenizer.encode(expected_token, add_special_tokens=False)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        keep_prefix = max_seq_length - max(len(wrong_ids), len(expected_ids)) - 1
        if keep_prefix <= 8:
            raise ValueError("--max-seq-length is too small for candidate scoring")
        truncated = max(0, len(prefix_ids) - keep_prefix)
        if truncated:
            prefix_ids = prefix_ids[-keep_prefix:]
            prefix = tokenizer.decode(prefix_ids)
        cases.append(
            {
                "sample_idx": int(sample["index"]),
                "problem_id": meta.get("problem_id"),
                "name": meta.get("name"),
                "prefix": prefix,
                "prefix_tokens": len(prefix_ids),
                "truncated_left_tokens": truncated,
                "wrong_pos_chars": pos,
                "w4_compiled": bool((turn.get("kernelgym") or {}).get("compiled")),
                "w4_correct": bool((turn.get("kernelgym") or {}).get("correctness")),
            }
        )
        if len(cases) >= max_examples:
            break
    if not cases:
        raise RuntimeError(f"no turn-1 responses containing {wrong_token!r} in {eval_path}")
    return cases


def register_hidden_hooks(
    model: torch.nn.Module,
    layers: list[int],
    tail_tokens: int,
    sink: dict[int, torch.Tensor],
) -> list[Any]:
    handles = []

    def make_hook(layer: int):
        def hook(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: Any) -> None:
            value = output[0] if isinstance(output, tuple) else output
            sink[layer] = value[:, -tail_tokens:, :].detach().to("cpu", dtype=torch.float32)

        return hook

    for layer in layers:
        handles.append(find_module(model, str(layer)).register_forward_hook(make_hook(layer)))
    return handles


def forward_prefix(
    model: torch.nn.Module,
    tokenizer: Any,
    prefix: str,
    layers: list[int],
    tail_tokens: int,
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    encoded = tokenizer(prefix, return_tensors="pt", add_special_tokens=False)
    first_device = next(model.parameters()).device
    encoded = {key: value.to(first_device) for key, value in encoded.items()}
    hidden: dict[int, torch.Tensor] = {}
    handles = register_hidden_hooks(model, layers, tail_tokens, hidden)
    with torch.no_grad():
        output = model(**encoded, use_cache=False, return_dict=True)
    for handle in handles:
        handle.remove()
    logits = output.logits[:, -1, :].detach().to("cpu", dtype=torch.float32)
    del output, encoded
    return hidden, logits


def sequence_logprob(
    model: torch.nn.Module,
    tokenizer: Any,
    prefix: str,
    candidate: str,
) -> tuple[float, int]:
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    candidate_ids = tokenizer.encode(candidate, add_special_tokens=False)
    input_ids = torch.tensor([prefix_ids + candidate_ids], dtype=torch.long)
    first_device = next(model.parameters()).device
    input_ids = input_ids.to(first_device)
    with torch.no_grad():
        output = model(input_ids=input_ids, use_cache=False, return_dict=True)
        logits = output.logits.float()
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        total = 0.0
        for offset, token_id in enumerate(candidate_ids):
            logit_pos = len(prefix_ids) - 1 + offset
            total += float(log_probs[0, logit_pos, token_id].item())
    del output, input_ids, logits, log_probs
    return total, len(candidate_ids)


def rel_l2(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    diff = torch.linalg.vector_norm(candidate - reference)
    base = torch.linalg.vector_norm(reference)
    return float((diff / base).item())


def topk_overlap(a: torch.Tensor, b: torch.Tensor, k: int) -> float:
    a_ids = set(torch.topk(a[0], k).indices.tolist())
    b_ids = set(torch.topk(b[0], k).indices.tolist())
    return len(a_ids & b_ids) / float(k)


def kl_to_reference(reference_logits: torch.Tensor, candidate_logits: torch.Tensor) -> float:
    ref_logp = torch.nn.functional.log_softmax(reference_logits, dim=-1)
    cand_logp = torch.nn.functional.log_softmax(candidate_logits, dim=-1)
    ref_p = ref_logp.exp()
    return float((ref_p * (ref_logp - cand_logp)).sum(dim=-1).mean().item())


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_layers(args.layers)

    config = text_config(json.loads((args.source / "config.json").read_text()))
    num_layers = int(config["num_hidden_layers"])

    print(f"[prop] loading BF16 model: {args.source}", flush=True)
    model, tokenizer = load_model_and_tokenizer(args.source)
    cases = select_error_cases(
        args.w4_eval,
        tokenizer,
        args.wrong_token,
        args.expected_token,
        args.max_examples,
        args.max_seq_length,
    )
    (args.output_dir / "cases.json").write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n")
    print(f"[prop] selected {len(cases)} cases", flush=True)

    source_map = load_weight_map(args.source)
    w8_map = load_weight_map(args.w8_mlp)
    w4_map = load_weight_map(args.w4_asym)

    variants = [
        ("bf16", args.source, source_map, "bf16", False),
        ("w8a8_mlp_proxy", args.w8_mlp, w8_map, "w8", True),
        ("w4_awq_asym", args.w4_asym, w4_map, "w4", False),
    ]

    baseline_hidden: dict[int, dict[int, torch.Tensor]] = {}
    baseline_logits: dict[int, torch.Tensor] = {}
    hidden_rows: list[dict[str, Any]] = []
    logit_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []

    for variant, root, weight_map, kind, use_a8 in variants:
        print(f"[prop] patching variant={variant}", flush=True)
        patch_mlp_variant(model, root, weight_map, kind=kind, num_layers=num_layers, use_a8=use_a8)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for case in cases:
            sample_idx = int(case["sample_idx"])
            print(f"[prop] forward variant={variant} sample={sample_idx}", flush=True)
            hidden, logits = forward_prefix(model, tokenizer, case["prefix"], layers, args.hidden_tail_tokens)

            if not args.skip_candidate_logprobs:
                wrong_lp, wrong_n = sequence_logprob(model, tokenizer, case["prefix"], args.wrong_token)
                expected_lp, expected_n = sequence_logprob(model, tokenizer, case["prefix"], args.expected_token)
                candidate_rows.append(
                    {
                        "sample_idx": sample_idx,
                        "variant": variant,
                        "prefix_tokens": case["prefix_tokens"],
                        "wrong_token": args.wrong_token,
                        "expected_token": args.expected_token,
                        "wrong_logprob": wrong_lp,
                        "expected_logprob": expected_lp,
                        "expected_minus_wrong": expected_lp - wrong_lp,
                        "wrong_num_tokens": wrong_n,
                        "expected_num_tokens": expected_n,
                    }
                )

            if variant == "bf16":
                baseline_hidden[sample_idx] = hidden
                baseline_logits[sample_idx] = logits
                continue

            for layer in layers:
                hidden_rows.append(
                    {
                        "sample_idx": sample_idx,
                        "variant": variant,
                        "layer": layer,
                        "hidden_tail_tokens": args.hidden_tail_tokens,
                        "hidden_rel_l2": rel_l2(baseline_hidden[sample_idx][layer], hidden[layer]),
                    }
                )
            logit_rows.append(
                {
                    "sample_idx": sample_idx,
                    "variant": variant,
                    "logits_kl_to_bf16": kl_to_reference(baseline_logits[sample_idx], logits),
                    "top1_same": int(
                        torch.argmax(baseline_logits[sample_idx], dim=-1).item() == torch.argmax(logits, dim=-1).item()
                    ),
                    "top10_overlap": topk_overlap(baseline_logits[sample_idx], logits, 10),
                    "top50_overlap": topk_overlap(baseline_logits[sample_idx], logits, 50),
                }
            )
            del hidden, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    write_csv(args.output_dir / "hidden_drift_by_layer.csv", hidden_rows)
    write_csv(args.output_dir / "logit_drift.csv", logit_rows)
    write_csv(args.output_dir / "candidate_logprobs.csv", candidate_rows)

    summary = {
        "metadata": {
            "source": str(args.source),
            "w8_mlp": str(args.w8_mlp),
            "w4_asym": str(args.w4_asym),
            "w4_eval": str(args.w4_eval),
            "layers": layers,
            "max_examples": args.max_examples,
            "max_seq_length": args.max_seq_length,
            "hidden_tail_tokens": args.hidden_tail_tokens,
            "skip_candidate_logprobs": args.skip_candidate_logprobs,
        },
        "cases": cases,
        "hidden_drift_rows": hidden_rows,
        "logit_drift_rows": logit_rows,
        "candidate_logprob_rows": candidate_rows,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")

    by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in hidden_rows:
        by_variant.setdefault(str(row["variant"]), []).append(row)
    lines = ["# W4 AWQ Error Propagation Probe", ""]
    lines.append("## Mean Hidden Drift By Layer")
    lines.append("")
    lines.append("| layer | W8A8 MLP proxy rel-L2 | W4 AWQ ASYM rel-L2 | W4/W8 |")
    lines.append("|---:|---:|---:|---:|")
    for layer in layers:
        w8_vals = [
            float(row["hidden_rel_l2"]) for row in by_variant.get("w8a8_mlp_proxy", []) if int(row["layer"]) == layer
        ]
        w4_vals = [
            float(row["hidden_rel_l2"]) for row in by_variant.get("w4_awq_asym", []) if int(row["layer"]) == layer
        ]
        w8_mean = sum(w8_vals) / len(w8_vals) if w8_vals else math.nan
        w4_mean = sum(w4_vals) / len(w4_vals) if w4_vals else math.nan
        ratio = w4_mean / w8_mean if w8_mean and math.isfinite(w8_mean) else math.nan
        lines.append(f"| {layer} | {w8_mean:.6f} | {w4_mean:.6f} | {ratio:.2f} |")
    if candidate_rows:
        lines.extend(["", "## Candidate Logprobs", ""])
        lines.append("| sample | variant | logp(expected) - logp(wrong) |")
        lines.append("|---:|---|---:|")
        for row in candidate_rows:
            lines.append(f"| {row['sample_idx']} | {row['variant']} | {float(row['expected_minus_wrong']):.4f} |")
    lines.extend(["", "## Logit Drift", ""])
    lines.append("| sample | variant | KL to BF16 | top1 same | top10 overlap | top50 overlap |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for row in logit_rows:
        lines.append(
            f"| {row['sample_idx']} | {row['variant']} | {float(row['logits_kl_to_bf16']):.6f} | "
            f"{row['top1_same']} | {float(row['top10_overlap']):.2f} | {float(row['top50_overlap']):.2f} |"
        )
    (args.output_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[prop] wrote {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
