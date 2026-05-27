"""SmoothQuant + W8A8-INT8 producer for Qwen3.5/3.6 hybrid models.

Pipeline:
  1. Apply SmoothQuant pre-shift (with explicit Qwen3.5 layernorm-to-projection
     mapping). This moves activation outliers from per-token activations into
     per-channel weight scales, making INT8 weight quantization tighter.
  2. (Optional, default ON) Apply RTN-INT8 weight quantization on the smoothed
     model.
  3. Save **two** outputs:
       (a) `--bf16-output-path` — smoothed BF16 checkpoint (no INT8 quantization;
            useful as a "smoothing only" diagnostic baseline).
       (b) `--w8a8-output-path` — smoothed + INT8 quantized checkpoint.

Why both outputs:
  - Smoothing alone changes the model's numerics. Comparing smoothed-BF16 vs
    unrotated-BF16 accuracy tells you whether smoothing itself introduced any
    quality loss. Comparing smoothed-W8A8 vs smoothed-BF16 isolates the cost
    of the INT8 step on top of smoothing.

Mappings (Qwen3.5 hybrid):
  - `post_attention_layernorm` → `mlp.gate_proj`, `mlp.up_proj`
    (down_proj is post-act, no clean layernorm source — not smoothed)
  - `input_layernorm` → `self_attn.{q,k,v}_proj` (full-attention layers)
    AND `linear_attn.{in_proj_a,in_proj_b,in_proj_qkv,in_proj_z}`
    (linear-attention / Mamba layers — note: both layer families share the
    `input_layernorm` naming so the regex catches both)

Usage (on .22):
    source /tmp/w8a8-venv/bin/activate
    python scripts/quantize/producers/smoothquant_w8a8.py \\
        --model-path /nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B \\
        --calibration-path /tmp/calib.jsonl \\
        --bf16-output-path /nfs/FM/chenshuailin/projects/kernel_agents/slime/checkpoints/quantized/Qwen3.6-27B-smooth-bf16 \\
        --w8a8-output-path /nfs/FM/chenshuailin/projects/kernel_agents/slime/checkpoints/quantized/Qwen3.6-27B-smooth-w8a8 \\
        --multimodal
"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import torch

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")

# -----------------------------------------------------------------------------
# Qwen3.5/3.6 hybrid SmoothQuant mappings
# -----------------------------------------------------------------------------
# Each entry is `[ [linear_re_list], layernorm_re ]`. SmoothQuantModifier scans
# the model graph, picks per-channel scales from the listed linears' input
# activations, then folds the inverse scale into the layernorm output (so the
# math is exact, no quality loss in principle — only floating-point drift).
#
# `re:` prefix is llmcompressor's regex marker. We name the full
# `model.language_model.layers.*` prefix explicitly so the mapping does NOT
# accidentally catch the vision tower's layernorms/linears.
#
# down_proj is deliberately omitted: its input is silu(gate) * up, not a clean
# layernorm output, so the SmoothQuant identity doesn't apply cleanly.
QWEN35_HYBRID_MAPPINGS = [
    # MLP: gate + up share post_attention_layernorm
    [
        [
            r"re:model\.language_model\.layers\.\d+\.mlp\.gate_proj$",
            r"re:model\.language_model\.layers\.\d+\.mlp\.up_proj$",
        ],
        r"re:model\.language_model\.layers\.\d+\.post_attention_layernorm$",
    ],
    # Full-attention QKV share input_layernorm
    [
        [
            r"re:model\.language_model\.layers\.\d+\.self_attn\.q_proj$",
            r"re:model\.language_model\.layers\.\d+\.self_attn\.k_proj$",
            r"re:model\.language_model\.layers\.\d+\.self_attn\.v_proj$",
        ],
        r"re:model\.language_model\.layers\.\d+\.input_layernorm$",
    ],
    # Linear-attention / Mamba: all in_proj_* share input_layernorm of the
    # linear_attn layer (same naming convention as self_attn but different
    # layer indices — Qwen3.5 hybrid uses input_layernorm in both).
    [
        [
            r"re:model\.language_model\.layers\.\d+\.linear_attn\.in_proj_a$",
            r"re:model\.language_model\.layers\.\d+\.linear_attn\.in_proj_b$",
            r"re:model\.language_model\.layers\.\d+\.linear_attn\.in_proj_qkv$",
            r"re:model\.language_model\.layers\.\d+\.linear_attn\.in_proj_z$",
        ],
        r"re:model\.language_model\.layers\.\d+\.input_layernorm$",
    ],
]

# Always-ignore for multimodal vision tower / projector / audio.
IGNORE_PATTERNS = [
    "lm_head",
    "re:.*\\.visual\\..*",
    "re:.*vision_tower.*",
    "re:.*mm_projector.*",
    "re:.*\\.audio_tower\\..*",
    "re:.*\\.mtp\\..*",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model-path", required=True, help="Source BF16 HF checkpoint dir")
    ap.add_argument(
        "--calibration-path",
        required=True,
        help="JSONL with one prompt per line under key 'text'",
    )
    ap.add_argument(
        "--bf16-output-path",
        required=True,
        help="Where to save the smoothed BF16 checkpoint (no INT8 quantization)",
    )
    ap.add_argument(
        "--w8a8-output-path",
        required=True,
        help="Where to save the smoothed + INT8 quantized checkpoint",
    )
    ap.add_argument(
        "--num-calibration-samples",
        type=int,
        default=128,
        help="SmoothQuant typically needs less calibration than GPTQ; 128 is a reasonable default",
    )
    ap.add_argument(
        "--max-seq-length",
        type=int,
        default=4096,
        help="Truncate calibration prompts to this length (default 4096)",
    )
    ap.add_argument(
        "--smoothing-strength",
        type=float,
        default=0.5,
        help="SmoothQuant alpha parameter (0=no smoothing, 1=all into weights; default 0.5 per paper)",
    )
    ap.add_argument(
        "--target",
        choices=["mlp", "all-linear"],
        default="mlp",
        help="Which Linear modules to quantize after smoothing. 'mlp' (default) only quantizes the MLP triplet. 'all-linear' quantizes every Linear except lm_head/visual/audio. Smoothing mappings always cover MLP + attention regardless.",
    )
    ap.add_argument(
        "--multimodal",
        action="store_true",
        help="Load via AutoModelForImageTextToText to preserve the multimodal architectures tag",
    )
    return ap.parse_args()


def load_model(model_path: str, multimodal: bool):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if multimodal:
        try:
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as e:
            print(
                f"[smoothquant] AutoModelForImageTextToText failed ({e!r}); falling back to AutoModel",
                flush=True,
            )
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


def load_calibration(path: str, n_max: int) -> list[str]:
    texts: list[str] = []
    with open(path) as f:
        for line in f:
            texts.append(json.loads(line)["text"])
            if len(texts) >= n_max:
                break
    return texts


def main() -> None:
    args = parse_args()

    print(
        f"[smoothquant] loading model from {args.model_path} (multimodal={args.multimodal})",
        flush=True,
    )
    model, tokenizer = load_model(args.model_path, args.multimodal)
    print(
        f"[smoothquant] model loaded, root dtype={next(model.parameters()).dtype}",
        flush=True,
    )

    calibration_texts = load_calibration(args.calibration_path, args.num_calibration_samples)
    print(
        f"[smoothquant] using {len(calibration_texts)} calibration prompts (alpha={args.smoothing_strength})",
        flush=True,
    )

    from datasets import Dataset

    calibration_ds = Dataset.from_dict({"text": calibration_texts})

    # ------------------------------------------------------------------
    # Stage 1: SmoothQuant only (produces the smoothed BF16 ckpt)
    # ------------------------------------------------------------------
    from llmcompressor import oneshot
    from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

    smooth_recipe = [
        SmoothQuantModifier(
            smoothing_strength=args.smoothing_strength,
            mappings=QWEN35_HYBRID_MAPPINGS,
            ignore=IGNORE_PATTERNS,
        )
    ]
    print(f"[smoothquant] stage 1: smoothing with {len(QWEN35_HYBRID_MAPPINGS)} mappings", flush=True)

    oneshot(
        model=model,
        processor=tokenizer,
        dataset=calibration_ds,
        recipe=smooth_recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=len(calibration_texts),
        output_dir=None,  # don't save yet; we'll save manually below
    )

    print(f"[smoothquant] stage 1 done, saving smoothed BF16 to {args.bf16_output_path}", flush=True)
    os.makedirs(args.bf16_output_path, exist_ok=True)
    model.save_pretrained(args.bf16_output_path, save_compressed=False)
    tokenizer.save_pretrained(args.bf16_output_path)
    # Copy chat template & preprocessor configs if present (multimodal models)
    for fname in (
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ):
        src = os.path.join(args.model_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, args.bf16_output_path)

    # ------------------------------------------------------------------
    # Stage 2: RTN-INT8 on top of the smoothed weights
    # ------------------------------------------------------------------
    from llmcompressor.modifiers.quantization import QuantizationModifier

    if args.target == "mlp":
        targets = [r"re:.*\.mlp\.(gate_proj|up_proj|down_proj)$"]
    else:
        targets = "Linear"

    quant_recipe = [
        QuantizationModifier(
            targets=targets,
            scheme="W8A8",
            ignore=IGNORE_PATTERNS,
        )
    ]
    print(f"[smoothquant] stage 2: W8A8 RTN quant on target={args.target}", flush=True)

    oneshot(
        model=model,
        processor=tokenizer,
        dataset=calibration_ds,
        recipe=quant_recipe,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=len(calibration_texts),
        output_dir=args.w8a8_output_path,
    )

    print(f"[smoothquant] stage 2 done, saving W8A8 to {args.w8a8_output_path}", flush=True)
    model.save_pretrained(args.w8a8_output_path, save_compressed=True)
    tokenizer.save_pretrained(args.w8a8_output_path)
    for fname in (
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    ):
        src = os.path.join(args.model_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, args.w8a8_output_path)

    print("[smoothquant] all done.", flush=True)
    print(f"  smoothed BF16 -> {args.bf16_output_path}", flush=True)
    print(f"  smoothed W8A8 -> {args.w8a8_output_path}", flush=True)


if __name__ == "__main__":
    main()
