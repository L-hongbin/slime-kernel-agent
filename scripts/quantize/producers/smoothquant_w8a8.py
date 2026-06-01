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
import sys
from pathlib import Path

import torch

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
# Always-ignore for multimodal vision tower / projector / audio.
BASE_IGNORE_PATTERNS = [
    "lm_head",
    "re:.*\\.visual\\..*",
    "re:.*vision_tower.*",
    "re:.*mm_projector.*",
    "re:.*\\.audio_tower\\..*",
]

MTP_IGNORE_PATTERNS = [
    # MTP / EAGLE draft head must stay BF16. sglang builds the draft model with
    # prefix "mtp", so its Linear modules are named `mtp.layers.0.*` (mtp at the
    # START, no leading dot). sglang matches ignore via re.match (anchored), so a
    # `\.mtp\.` pattern never matches and every draft Linear would wrongly get
    # the W8A8 scheme with no weight_scale -> garbage draft logits, ~0.1 accept
    # rate. `.*mtp\.` matches `mtp.layers...` while leaving the target body
    # `model.language_model.layers...` (no `mtp.` substring) quantized.
    "re:.*mtp\\..*",
]


def _layer_indices_by_type(model_path: str) -> tuple[list[int], list[int]]:
    """Return (full_attn_indices, linear_attn_indices) from the model config.

    Qwen3.5 hybrid lists `text_config.layer_types` with `full_attention` or
    `linear_attention` per layer. SmoothQuant's mapping resolver pairs
    layernorm-to-projection by layer index, so we must restrict the layernorm
    regex to ONLY the layer indices that have the corresponding projection
    family — otherwise the resolver finds 64 input_layernorms but only 16
    self_attn.q_projs and refuses to resolve.

    Asserts the layout matches the expected Qwen3.5 27B hybrid (16 full +
    48 linear = 64 layers) so a mismatched config fails fast here instead of
    later inside the SmoothQuant resolver with a confusing error.
    """
    cfg = json.load(open(f"{model_path}/config.json"))
    layer_types = cfg.get("text_config", cfg).get("layer_types") or []
    full_attn = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    linear_attn = [i for i, t in enumerate(layer_types) if t == "linear_attention"]
    n_total = len(layer_types)
    n_other = n_total - len(full_attn) - len(linear_attn)
    if n_other:
        raise ValueError(
            f"unexpected layer_types in {model_path}/config.json: "
            f"{n_other} entries are neither 'full_attention' nor 'linear_attention'"
        )
    if not full_attn:
        raise ValueError(
            f"no full_attention layers in {model_path}/config.json; "
            f"non_linear_attn / all-linear targets need at least one full-attn layer"
        )
    if not linear_attn:
        raise ValueError(
            f"no linear_attention layers in {model_path}/config.json; "
            f"this script assumes a Qwen3.5-style hybrid layout"
        )
    return full_attn, linear_attn


def _idx_regex(indices: list[int]) -> str:
    """`[3,7,11]` -> `(3|7|11)` (non-grouping not supported in SmoothQuant's regex parser)."""
    return "(" + "|".join(str(i) for i in indices) + ")"


def build_mappings_for_target(target: str, model_path: str) -> list:
    """Return SmoothQuant mappings scoped to the target's layer set.

    We only smooth modules that we plan to quantize — smoothing a layer that
    stays BF16 just slightly perturbs its weights without any quantization
    benefit. We also restrict each layernorm regex to the layer indices that
    actually have the corresponding projection family (full_attention layers
    for self_attn, linear_attention layers for linear_attn/Mamba), so the
    SmoothQuant resolver can pair layernorm-to-proj exactly.

    MLP mapping covers all 64 target-model layers (every layer has MLP). The
    explicit *_mtp targets add the single MTP draft layer to the same smoothing
    and quantization scope.
    """
    full_attn, linear_attn = _layer_indices_by_type(model_path)
    all_layers_re = r"\d+"

    # MLP — gate + up share post_attention_layernorm. Every layer has MLP.
    mapping_mlp = [
        [
            rf"re:model\.language_model\.layers\.{all_layers_re}\.mlp\.gate_proj$",
            rf"re:model\.language_model\.layers\.{all_layers_re}\.mlp\.up_proj$",
        ],
        rf"re:model\.language_model\.layers\.{all_layers_re}\.post_attention_layernorm$",
    ]
    # Full-attention QKV share input_layernorm — only the full-attention layers.
    full_idx = _idx_regex(full_attn)
    mapping_self_attn = [
        [
            rf"re:model\.language_model\.layers\.{full_idx}\.self_attn\.q_proj$",
            rf"re:model\.language_model\.layers\.{full_idx}\.self_attn\.k_proj$",
            rf"re:model\.language_model\.layers\.{full_idx}\.self_attn\.v_proj$",
        ],
        rf"re:model\.language_model\.layers\.{full_idx}\.input_layernorm$",
    ]
    # Linear-attention / Mamba in_proj_* — only the linear-attention layers.
    lin_idx = _idx_regex(linear_attn)
    mapping_linear_attn = [
        [
            rf"re:model\.language_model\.layers\.{lin_idx}\.linear_attn\.in_proj_a$",
            rf"re:model\.language_model\.layers\.{lin_idx}\.linear_attn\.in_proj_b$",
            rf"re:model\.language_model\.layers\.{lin_idx}\.linear_attn\.in_proj_qkv$",
            rf"re:model\.language_model\.layers\.{lin_idx}\.linear_attn\.in_proj_z$",
        ],
        rf"re:model\.language_model\.layers\.{lin_idx}\.input_layernorm$",
    ]
    mapping_mtp_mlp = [
        [
            r"re:mtp\.layers\.\d+\.mlp\.gate_proj$",
            r"re:mtp\.layers\.\d+\.mlp\.up_proj$",
        ],
        r"re:mtp\.layers\.\d+\.post_attention_layernorm$",
    ]
    mapping_mtp_self_attn = [
        [
            r"re:mtp\.layers\.\d+\.self_attn\.q_proj$",
            r"re:mtp\.layers\.\d+\.self_attn\.k_proj$",
            r"re:mtp\.layers\.\d+\.self_attn\.v_proj$",
        ],
        r"re:mtp\.layers\.\d+\.input_layernorm$",
    ]

    if target == "mlp":
        return [mapping_mlp]
    if target == "mlp_mtp":
        return [mapping_mlp, mapping_mtp_mlp]
    if target == "non_linear_attn":
        return [mapping_mlp, mapping_self_attn]
    if target == "non_linear_attn_mtp":
        return [mapping_mlp, mapping_self_attn, mapping_mtp_mlp, mapping_mtp_self_attn]
    if target == "all-linear":
        return [mapping_mlp, mapping_self_attn, mapping_linear_attn]
    raise ValueError(f"unknown target {target!r}")


def quant_targets_and_ignore(target: str) -> tuple:
    """Return (targets, ignore) for the QuantizationModifier matching the smoothing scope."""
    mlp_re = r"re:.*\.mlp\.(gate_proj|up_proj|down_proj)$"
    self_attn_re = r"re:.*\.self_attn\.(q_proj|k_proj|v_proj|o_proj)$"
    base_ignore = list(BASE_IGNORE_PATTERNS)
    if target not in ("mlp_mtp", "non_linear_attn_mtp"):
        base_ignore += MTP_IGNORE_PATTERNS
    else:
        base_ignore.append(r"re:^mtp\.fc(\..*)?$")
    if target in ("mlp", "mlp_mtp"):
        targets = [mlp_re]
        # leave attention BF16
        base_ignore += [r"re:.*\.self_attn\..*", r"re:.*\.linear_attn\..*"]
    elif target in ("non_linear_attn", "non_linear_attn_mtp"):
        targets = [mlp_re, self_attn_re]
        # leave linear_attn (mamba) BF16
        base_ignore += [r"re:.*\.linear_attn\..*"]
    elif target == "all-linear":
        targets = "Linear"
    else:
        raise ValueError(f"unknown target {target!r}")
    return targets, base_ignore


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
        choices=["mlp", "mlp_mtp", "non_linear_attn", "non_linear_attn_mtp", "all-linear"],
        default="non_linear_attn",
        help=(
            "Which Linear modules to quantize after smoothing. "
            "'mlp' = MLP triplet only (192 modules); "
            "'mlp_mtp' = MLP triplet plus MTP draft-head MLP; "
            "'non_linear_attn' (default) = MLP + self_attn QKVO, skip mamba/linear_attn (256 modules); "
            "'non_linear_attn_mtp' = non_linear_attn plus MTP draft-head MLP + self_attn; "
            "'all-linear' = every Linear except lm_head/visual/audio (496 modules). "
            "The smoothing mappings are scoped to match the target — modules left BF16 are not smoothed either."
        ),
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
    domain_adapted_markers = 0
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            if "problem_id" in row or row.get("source") in {"drkernel", "kernelbench"}:
                domain_adapted_markers += 1
            texts.append(row["text"])
            if len(texts) >= n_max:
                break
    if domain_adapted_markers:
        print(
            "[smoothquant] WARNING: calibration file appears to come from "
            "DrKernel/KernelBench eval dumps. This is domain-adapted calibration "
            "and must not be used for general quality or lossless-SmoothQuant claims.",
            flush=True,
        )
    return texts


def patch_llmcompressor_transformers5() -> None:
    """Provide the Transformers <=4 init hook name expected by llmcompressor."""
    import transformers.modeling_utils as modeling_utils

    if hasattr(modeling_utils, "TORCH_INIT_FUNCTIONS"):
        return
    names = (
        "uniform_",
        "normal_",
        "xavier_uniform_",
        "xavier_normal_",
        "kaiming_uniform_",
        "kaiming_normal_",
        "orthogonal_",
    )
    modeling_utils.TORCH_INIT_FUNCTIONS = {
        name: getattr(torch.nn.init, name) for name in names if hasattr(torch.nn.init, name)
    }


def patch_transformers5_no_split_modules(model) -> None:
    """Backfill the Transformers <=4 method that llmcompressor still calls."""
    if hasattr(model, "_get_no_split_modules"):
        return

    def _get_no_split_modules(device_map=None):
        return list(getattr(model, "_no_split_modules", []) or [])

    model._get_no_split_modules = _get_no_split_modules


def main() -> None:
    args = parse_args()

    print(
        f"[smoothquant] loading model from {args.model_path} (multimodal={args.multimodal})",
        flush=True,
    )
    model, tokenizer = load_model(args.model_path, args.multimodal)
    patch_transformers5_no_split_modules(model)
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
    patch_llmcompressor_transformers5()
    from llmcompressor import oneshot
    from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier
    from scripts.quantize.patches.llmcompressor_qwen3_5_smoothquant import patch_smoothquant_qwen3_5_rmsnorm

    patch_smoothquant_qwen3_5_rmsnorm()

    smooth_mappings = build_mappings_for_target(args.target, args.model_path)
    smooth_recipe = [
        SmoothQuantModifier(
            smoothing_strength=args.smoothing_strength,
            mappings=smooth_mappings,
            ignore=quant_targets_and_ignore(args.target)[1],
        )
    ]
    print(
        f"[smoothquant] stage 1: target={args.target}, smoothing with {len(smooth_mappings)} mapping group(s)",
        flush=True,
    )

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

    from scripts.quantize.utils.mtp_checkpoint import inject_mtp_tensors

    injected = inject_mtp_tensors(args.bf16_output_path, args.model_path)
    print(
        f"[smoothquant] restored {injected} MTP tensor(s) into smoothed BF16 checkpoint",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Stage 2: RTN-INT8 on top of the smoothed BF16 ckpt
    #
    # Important: we DON'T use llmcompressor's QuantizationModifier here.
    # On 2026-05-27 we found that the two-stage oneshot pattern
    # (SmoothQuant -> save BF16 -> QuantizationModifier on in-memory model)
    # silently produces a BROKEN W8A8 ckpt: int8 weights saturated at
    # ±127 (std ~126 instead of healthy ~30), scales saved as bf16
    # instead of fp32, model generates multilingual token salad.
    # Sglang loads it fine, no exception is raised, no warning emitted.
    #
    # Instead we call our proven pure-PyTorch RTN writer on the saved
    # smoothed-BF16 ckpt. That path is unit-tested
    # (tests/utils/test_quantize_w8a8_rtn_local.py) and ships with a
    # built-in saturation/scale-shape validator.
    # ------------------------------------------------------------------
    # Free GPU memory before invoking the RTN writer (which reads
    # safetensors shard-by-shard from disk into CPU and writes new shards).
    import gc

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    from scripts.quantize.producers.rtn_w8a8 import quantize_checkpoint

    print(
        f"[smoothquant] stage 2: pure-PyTorch RTN on smoothed-BF16 -> W8A8 "
        f"(target={args.target}, source={args.bf16_output_path})",
        flush=True,
    )
    quantize_checkpoint(
        model_path=args.bf16_output_path,
        output_path=args.w8a8_output_path,
        target=args.target,
        force=True,
    )

    # Final post-save validation — guard against silent corruption.
    from scripts.quantize.utils.validate_checkpoint import check_w8a8_checkpoint

    check_w8a8_checkpoint(
        Path(args.w8a8_output_path),
        reference_checkpoint=Path(args.bf16_output_path),
        max_tensors=32,
        # 0.05 saturated_frac threshold per codex review 2026-05-27. The
        # broken llmcompressor pipeline produced ~99% saturated; tightening
        # to 5% (was 20%) catches less-extreme corruption too. A correctly-
        # quantized RTN ckpt has well under 1% saturation, so 5% has plenty
        # of headroom for the per-channel max → ±127 boundary case.
        max_saturated_frac=0.05,
        max_rel_l2=0.10,
    )

    print("[smoothquant] all done.", flush=True)
    print(f"  smoothed BF16 -> {args.bf16_output_path}", flush=True)
    print(f"  smoothed W8A8 -> {args.w8a8_output_path}", flush=True)


if __name__ == "__main__":
    main()
