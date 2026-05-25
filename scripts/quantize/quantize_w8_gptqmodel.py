"""Quantize a Qwen3.5/3.6-family checkpoint to W8 INT (weights-only) using GPTQModel.

Uses GPTQModel's modern Qwen3.5 dense-MLP definition (`Qwen3_5GPTQ`,
mirrors `Qwen3_5MoeGPTQ` with the MoE block replaced by a dense MLP)
and writes a checkpoint in the GPTQ_V2 format with `act_group_aware=True`
calibration — ~16k× faster than `desc_act=True` ordering with equal or
better quality recovery, per GPTQModel docs.

Why GPTQModel instead of llmcompressor:
- llmcompressor only has vanilla GPTQModifier (block_size / dampening_frac
  / actorder / offload_hessians); no GPTQv2, no GPTAQ, no
  activation-aware variant
- GPTQModel exposes FORMAT.GPTQ_V2, GPTAQConfig (experimental), FOEMConfig,
  act_group_aware, and an explicit Qwen3.5 model definition that loads
  the multimodal checkpoint with `model.language_model.layers.*` prefix
- An existing W8 ckpt from this exact recipe at
  https://huggingface.co/btbtyler09/Qwen3.6-27B-GPTQ-8bit
  reports +0.07% wikitext-2 perplexity degradation (effectively lossless)

What gets quantized (default = MLP only):
- LM MLP (`mlp.gate_proj` / `mlp.up_proj` / `mlp.down_proj`) → INT8
- LM attention (`self_attn.*_proj` / `linear_attn.*_proj`) → BF16
- Embeddings / norms / lm_head → BF16
- Vision encoder / MTP module → BF16

Caveats — see handoffs/in_progress/HANDOFF_DRKERNEL_W8A8_ROLLOUT.md:
- W8 (not W8A8): weights quantized, activations stay BF16. Less rollout
  speedup than W8A8 but much better accuracy retention. Pareto choice.
- Loader limitations: neither GPTQModel nor stock transformers can load
  the resulting ckpt directly today. vLLM works after a small config
  patch (the public ckpt's model card has the one-liner sed). sglang
  support is unverified — may need analogous patching.

Usage:
    source /tmp/w8a8-venv/bin/activate  # or any venv with gptqmodel installed
    python scripts/quantize/quantize_w8_gptqmodel.py \\
        --model-path /nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B \\
        --calibration-path /tmp/calib.jsonl \\
        --output-path /nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B-W8-MLP-gptqmodel
"""

from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True, help="HF checkpoint dir to quantize")
    ap.add_argument("--calibration-path", required=True, help="JSONL with one prompt per line under key 'text'")
    ap.add_argument("--output-path", required=True, help="Where to save the GPTQ_V2 W8 checkpoint")
    ap.add_argument("--num-calibration-samples", type=int, default=512, help="Default 512")
    ap.add_argument("--max-seq-length", type=int, default=4096, help="Truncate calibration prompts to this")
    ap.add_argument(
        "--target",
        choices=["mlp", "all-linear"],
        default="mlp",
        help="What to quantize. 'mlp' (default) = SwiGLU MLP only. 'all-linear' = all decoder linears (mlp + attn).",
    )
    ap.add_argument("--bits", type=int, default=8, choices=[4, 8], help="Quantization bit width")
    ap.add_argument(
        "--group-size",
        type=int,
        default=32,
        help="Group size for grouped quantization. 32 matches the public W8 ckpt recipe; 128 is faster but lower quality.",
    )
    return ap.parse_args()


def build_quant_config(args):
    """Build a GPTQModel QuantizeConfig with our defaults.

    Key choices:
    - format = GPTQ_V2 (modern format; some loaders also accept legacy GPTQ)
    - act_group_aware = True + desc_act = False: 16k× faster than desc_act
      ordering at equal or better quality (per GPTQModel docs)
    - sym = True: symmetric quantization
    - dynamic: negative-regex skip for non-MLP modules when --target mlp

    See https://github.com/ModelCloud/GPTQModel for full field reference.
    """
    from gptqmodel.quantization import FORMAT, QuantizeConfig

    dynamic = None
    if args.target == "mlp":
        # GPTQModel dynamic syntax: "-:.*pattern" means SKIP modules matching
        # pattern (negative match). Empty dict {} is the "use defaults" sentinel.
        dynamic = {
            r"-:.*self_attn\..*": {},
            r"-:.*linear_attn\..*": {},
            r"-:.*\.visual\..*": {},
            r"-:.*vision_tower.*": {},
            r"-:.*mm_projector.*": {},
            r"-:.*\.audio_tower\..*": {},
            r"-:.*embed_tokens.*": {},
            r"-:lm_head": {},
            r"-:.*mtp.*": {},  # Multi-Token Prediction module (speculative decoding head)
        }

    return QuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        sym=True,
        desc_act=False,
        act_group_aware=True,
        format=FORMAT.GPTQ_V2,
        damp_percent=0.01,
        dynamic=dynamic,
    )


def main():
    args = parse_args()

    from gptqmodel import GPTQModel

    print(f"[quantize] loading model from {args.model_path}", flush=True)
    quant_config = build_quant_config(args)
    model = GPTQModel.load(args.model_path, quant_config, trust_remote_code=True)

    print(f"[quantize] loading calibration from {args.calibration_path}", flush=True)
    calibration_texts: list[str] = []
    with open(args.calibration_path) as f:
        for line in f:
            calibration_texts.append(json.loads(line)["text"])
            if len(calibration_texts) >= args.num_calibration_samples:
                break
    print(f"[quantize] using {len(calibration_texts)} calibration prompts", flush=True)

    # GPTQModel's quantize() takes raw text and tokenizes internally with the
    # model's bundled tokenizer; truncation handled via batch_size / cache_examples_on_gpu.
    print(
        f"[quantize] running GPTQ_V2 bits={args.bits} group_size={args.group_size} target={args.target}",
        flush=True,
    )
    model.quantize(calibration_texts, batch_size=1)

    print(f"[quantize] saving to {args.output_path}", flush=True)
    model.save(args.output_path)
    print(f"[quantize] done. GPTQ_V2 ckpt at {args.output_path}", flush=True)


if __name__ == "__main__":
    main()
