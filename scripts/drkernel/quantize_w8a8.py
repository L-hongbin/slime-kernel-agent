"""Quantize a Qwen3.5/3.6-family checkpoint to compressed-tensors W8A8-INT8.

Per-channel static INT8 weights + per-token dynamic INT8 activations via
GPTQ (no SmoothQuant — llmcompressor has no Qwen3.5 SmoothQuant mapping).

Default target = SwiGLU MLP only (`mlp.gate_proj` / `mlp.up_proj` /
`mlp.down_proj`). Attention (`self_attn`, `linear_attn`), embeddings
(`embed_tokens`), and `lm_head` stay BF16. Rationale: MLP is the largest
single weight family and has the smallest accuracy-per-bit cost in
SwiGLU; attention and embeddings degrade more under INT8.

Two model-load modes (pick via `--multimodal`):

- default (`AutoModelForCausalLM.from_pretrained`): faster, lower RAM
  during quantization. **Side effect**: rewrites the saved
  `architectures` tag from `<X>ForConditionalGeneration` to
  `<X>ForCausalLM`. For Qwen3.5/3.6 this breaks sglang load (the dense
  causal entry is unregistered and the registered code path has a
  hardcoded `num_experts` access — see
  `handoffs/in_progress/HANDOFF_DRKERNEL_W8A8_ROLLOUT.md`). Apply the
  sglang patch in this dir to unlock load.
- `--multimodal`: load via `Qwen3_5ForConditionalGeneration` (or the
  AutoModelForImageTextToText path). Vision tower weights are loaded
  in BF16 and pass through unchanged (the MLP-only target naturally
  skips them — vision tower has no `.mlp.gate/up/down_proj` subtree).
  The saved `architectures` tag stays as the multimodal entry that
  sglang already has registered; no sglang patch needed.

Usage:
    source /tmp/w8a8-venv/bin/activate
    python scripts/drkernel/quantize_w8a8.py \\
        --model-path /nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B \\
        --calibration-path /tmp/calib.jsonl \\
        --output-path /nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B-W8A8-MLP-ct \\
        --multimodal
"""

from __future__ import annotations

import argparse
import json
import os

import torch

# Reduce HF logging noise
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True, help="HF checkpoint dir to quantize")
    ap.add_argument("--calibration-path", required=True, help="JSONL with one prompt per line under key 'text'")
    ap.add_argument("--output-path", required=True, help="Where to save the compressed-tensors W8A8 checkpoint")
    ap.add_argument(
        "--num-calibration-samples", type=int, default=512, help="Default 512 (llmcompressor recommends 512-1024)"
    )
    ap.add_argument(
        "--max-seq-length", type=int, default=4096, help="Truncate calibration prompts to this (default 4096)"
    )
    ap.add_argument(
        "--target",
        choices=["mlp", "all-linear"],
        default="mlp",
        help="What to quantize. 'mlp' (default) = SwiGLU MLP only (gate/up/down_proj); attn/embed/lm_head stay BF16. 'all-linear' = every Linear except lm_head.",
    )
    ap.add_argument(
        "--multimodal",
        action="store_true",
        help="Load via AutoModelForImageTextToText so vision tower weights are preserved and architectures tag stays as <X>ForConditionalGeneration (avoids sglang load issues).",
    )
    return ap.parse_args()


def build_recipe(target: str):
    """Return an llmcompressor recipe list. MLP-only matches Qwen SwiGLU triplet
    via regex; the hybrid linear_attn layers have no .mlp subtree so they're
    naturally skipped. `ignore=['lm_head']` is redundant under 'mlp' (regex
    won't match anyway) but kept as defense in depth."""
    from llmcompressor.modifiers.quantization import GPTQModifier

    if target == "mlp":
        targets = [r"re:.*\.mlp\.(gate_proj|up_proj|down_proj)$"]
    elif target == "all-linear":
        targets = "Linear"
    else:
        raise ValueError(target)
    return [GPTQModifier(targets=targets, scheme="W8A8", ignore=["lm_head"])]


def load_model(model_path: str, multimodal: bool):
    """Load the model in either CausalLM or multimodal mode.

    CausalLM mode loses the vision tower (only ~12% of total params for Qwen
    3.6-27B) and rewrites the architectures tag — but is faster to load and
    uses less host RAM during quantization. Multimodal mode preserves both."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if multimodal:
        # AutoModelForImageTextToText handles modern multimodal models
        # (Qwen3-VL, etc.) without requiring image processor / torchvision
        # at load time. Falls back to direct class load if the auto resolver
        # fails on this arch.
        try:
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"[quantize] AutoModelForImageTextToText failed ({e!r}); falling back to AutoModel.from_pretrained")
            from transformers import AutoModel

            model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    return model, tokenizer


def main():
    args = parse_args()

    print(f"[quantize] loading model from {args.model_path} (multimodal={args.multimodal})", flush=True)
    model, tokenizer = load_model(args.model_path, args.multimodal)
    print(f"[quantize] model loaded, root dtype={next(model.parameters()).dtype}", flush=True)

    print(f"[quantize] loading calibration from {args.calibration_path}", flush=True)
    calibration_texts = []
    with open(args.calibration_path) as f:
        for line in f:
            calibration_texts.append(json.loads(line)["text"])
            if len(calibration_texts) >= args.num_calibration_samples:
                break
    print(f"[quantize] using {len(calibration_texts)} calibration prompts", flush=True)

    from datasets import Dataset

    calibration_ds = Dataset.from_dict({"text": calibration_texts})

    recipe = build_recipe(args.target)
    print(f"[quantize] target={args.target} recipe={recipe}", flush=True)

    # Pass `processor=tokenizer` (not `tokenizer=tokenizer`); llmcompressor
    # `oneshot` treats these as mutually exclusive and the multimodal model
    # otherwise triggers an AutoProcessor lookup that needs torchvision.
    from llmcompressor import oneshot

    oneshot(
        model=model,
        processor=tokenizer,
        dataset=calibration_ds,
        recipe=recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=len(calibration_texts),
        output_dir=args.output_path,
    )

    print(f"[quantize] saving to {args.output_path}", flush=True)
    model.save_pretrained(args.output_path, save_compressed=True)
    tokenizer.save_pretrained(args.output_path)
    print(f"[quantize] done. compressed-tensors W8A8 ckpt at {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
