"""Offline RTN W8A8-INT8 quantization for Qwen3.5/3.6-family checkpoints.

Companion to the online RTN path slime ships per RL step in
`slime/backends/megatron_utils/megatron_to_hf/processors/quantizer_compressed_tensors.py`.
This script produces the starting ckpt sglang loads at engine boot; slime then
overwrites the weights at each weight-sync with its own RTN INT8 packing of the
Megatron actor's current weights.

RTN = Round-To-Nearest = pure weight-statistic min-max quantization, no
calibration data, no Hessian, no iterative refinement. Runs in seconds (modulo
disk write) instead of the ~3-6h GPTQ takes.

Why offline RTN and not just rotated BF16 + a `quantization_config` tag:
- sglang's `compressed_tensors_w8a8_int8` scheme expects raw int8 `.weight`
  tensors on disk; pointing it at BF16 safetensors with a W8A8 config block
  causes the loader to either reject or silently mis-quantize during weight
  registration.
- Doing offline RTN once gives a valid sglang-loadable ckpt, then slime's
  per-step weight-sync overwrites it with the correct rotated+RTN INT8
  derived from the current Megatron actor.

Activations are per-token dynamic INT8 — sglang computes the input scale at
forward time, so no input-side calibration is needed.

Usage:
    python scripts/quantize/quantize_w8a8_rtn_llmcompressor.py \\
        --model-path /nfs/.../Qwen3.6-27B \\
        --output-path /nfs/.../Qwen3.6-27B-W8A8-RTN

By default the script uses `--multimodal` load + `all-linear` targets (every
text-side Linear except `lm_head`, vision tower preserved BF16). This matches
the sglang multimodal entry that's wired up in production.
"""

from __future__ import annotations

import argparse
import os

import torch

# Reduce HF logging noise
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")


def _patch_torch_accelerator_for_compressed_tensors():
    """compressed_tensors.offload.dispatch.get_device_memory calls
    `torch.accelerator.get_memory_info(idx)` and `torch.accelerator.current_device_index()`,
    which exist in newer torch builds but not in 2.9.1 (only `current_accelerator`,
    `device_count`, `current_device_idx` are available). Shim them with cuda equivalents
    so the data_free pipeline can dispatch the model without an AttributeError.
    """
    if not hasattr(torch, "accelerator"):
        return
    if not hasattr(torch.accelerator, "get_memory_info"):

        def _get_memory_info(idx):
            free, total = torch.cuda.mem_get_info(int(idx))
            return (free, total)

        torch.accelerator.get_memory_info = _get_memory_info
    if not hasattr(torch.accelerator, "current_device_index"):
        torch.accelerator.current_device_index = lambda: int(torch.cuda.current_device())


_patch_torch_accelerator_for_compressed_tensors()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True, help="HF checkpoint dir to quantize")
    ap.add_argument("--output-path", required=True, help="Where to save the compressed-tensors W8A8-RTN checkpoint")
    ap.add_argument(
        "--target",
        choices=["mlp", "all-linear"],
        default="all-linear",
        help="What to quantize. 'all-linear' (default for RTN) = every Linear except lm_head/vision. "
        "'mlp' = SwiGLU MLP triplet only.",
    )
    ap.add_argument(
        "--no-multimodal",
        action="store_true",
        help="Load via AutoModelForCausalLM. Default is multimodal load to keep the "
        "`architectures=*ForConditionalGeneration` tag intact so sglang uses its "
        "registered multimodal entry (avoids the dense-entry bugs documented in "
        "handoffs/in_progress/HANDOFF_DRKERNEL_W8A8_ROLLOUT.md).",
    )
    return ap.parse_args()


def build_recipe(target: str):
    """Plain RTN, no GPTQ — no calibration data needed.

    Uses `QuantizationModifier` directly. `GPTQModifier` would also do RTN with
    `block_size=0` / no calibration, but `QuantizationModifier` is the clean
    API for the "just apply the scheme, don't iterate Hessian" path.
    """
    from llmcompressor.modifiers.quantization import QuantizationModifier

    if target == "mlp":
        targets = [r"re:.*\.mlp\.(gate_proj|up_proj|down_proj)$"]
    elif target == "all-linear":
        targets = ["Linear"]
    else:
        raise ValueError(target)

    ignore = [
        "lm_head",
        # Vision / multimodal subgraphs — keep BF16. Defensive ignores even when
        # `targets` already excludes them (some llmcompressor traversals still
        # hook these modules to attach observers).
        "re:.*\\.visual\\..*",
        "re:.*vision_tower.*",
        "re:.*mm_projector.*",
        "re:.*\\.audio_tower\\..*",
        # MTP heads (Qwen3.5/3.6 multi-token-prediction) — speculative-decoding
        # specific; quantizing them is a separate concern.
        "re:.*\\.mtp\\..*",
    ]
    return [QuantizationModifier(targets=targets, scheme="W8A8", ignore=ignore)]


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
    multimodal = not args.no_multimodal

    print(f"[quantize] loading model from {args.model_path} (multimodal={multimodal})", flush=True)
    model, tokenizer = load_model(args.model_path, multimodal)
    print(f"[quantize] model loaded, root dtype={next(model.parameters()).dtype}", flush=True)

    recipe = build_recipe(args.target)
    print(f"[quantize] target={args.target} recipe={recipe}", flush=True)

    # No dataset → no calibration → pure RTN. llmcompressor's `oneshot` is
    # robust to a missing dataset for `QuantizationModifier`-only recipes
    # because per-channel weight quant + per-token *dynamic* activation quant
    # depend only on the static weight tensor and runtime activations.
    from llmcompressor import oneshot

    oneshot(
        model=model,
        processor=tokenizer,
        recipe=recipe,
        output_dir=args.output_path,
    )

    print(f"[quantize] saving to {args.output_path}", flush=True)
    model.save_pretrained(args.output_path, save_compressed=True)
    tokenizer.save_pretrained(args.output_path)
    print(f"[quantize] done. compressed-tensors W8A8-RTN ckpt at {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
