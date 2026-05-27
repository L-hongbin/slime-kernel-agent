"""Apply offline-only SpinQuant rotation to a Qwen3.5/3.6 multimodal model.

Saves BF16 (no quantization). The point of this script is to validate that the
rotation math is preserved across Qwen3.5's hybrid attention (linear_attn +
self_attn) BEFORE adding GPTQ W8A8 on top.

**Loads via `AutoModelForImageTextToText`** so the vision tower is preserved
in BF16 and the saved `architectures` tag stays as
`Qwen3_5ForConditionalGeneration` — the entry sglang has actually registered
and tested. (The dense `Qwen3_5ForCausalLM` entry has at least four
independent bugs in current sglang; see W8A8 handoff.)

R1 rotation is scoped strictly to `model.language_model.*` projections —
vision tower modules are not touched, so they pass through as BF16.
- self_attn.q/k/v/o on the 16 full-attention layers
- linear_attn.in_proj_{a,b,qkv,z} + out_proj on the 48 mamba-style layers
- mlp.{gate,up,down}_proj on every layer
- embed_tokens + lm_head (top level)

Codex's prior audit: R1 must propagate through every projection that
touches the residual stream. Internal SSM tensors (conv1d, A_log, dt_bias,
norm inside linear_attn) operate in post-projection basis and must NOT be
rotated.

Usage on .22:
    source /tmp/w8a8-venv/bin/activate
    python scripts/quantize/rotation/rotate_bf16.py \\
        --model-path /nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B \\
        --output-path /nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B-rotated-mm-bf16
"""

from __future__ import annotations

import argparse
import os
import re

import torch
from llmcompressor.modifiers.transform.spinquant.mappings import SpinQuantMapping
from llmcompressor.modifiers.transform.spinquant.norm_mappings import NormMapping
from transformers import AutoTokenizer

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")


# Qwen3.5/3.6 hybrid arch mapping. Layout (verified against
# Qwen3.6-27B BF16 ckpt, 64 layers total):
#   - 16 full-attention layers [3,7,11,...,63]: self_attn.{q,k,v,o}_proj
#   - 48 linear_attn layers (mamba-style GatedDeltaNet):
#       in_proj_a/in_proj_b/in_proj_qkv/in_proj_z (residual → projected)
#       out_proj (projected → residual)
#       conv1d/A_log/dt_bias/norm (internal SSM state, do NOT rotate)
#   - All 64 layers: mlp.{gate,up,down}_proj (SwiGLU)
#   - Top level: model.embed_tokens, model.norm, lm_head
#
# R1 must propagate through every projection that touches the residual stream.
# llmcompressor's SpinQuantMapping schema has fixed slots {attn_q, attn_k,
# attn_v, attn_o} for "things on the input side of attention" — for our
# hybrid arch we cram the linear_attn input projections into those slots
# via regex disjunction. attn_o picks up both self_attn.o_proj and
# linear_attn.out_proj on the output side.
#
# We start with rotations=["R1"] only. R2 is per-head V/O rotation which
# only makes sense for full attention; linear_attn has no head concept.
# Even for full_attention layers the R2 win on W8A8 is small (R2's
# headline benefit is for low-bit attention quant).
def build_qwen35_mapping():
    """Construct the SpinQuantMapping for Qwen3.5/3.6 hybrid arch (multimodal layout).

    All projection patterns are scoped with `language_model\.` prefix so that
    the vision tower (`model.visual.*`) and MTP module (`mtp.*`) are
    completely untouched by rotation — they pass through as BF16.

    Done lazily so the script's --help works without llmcompressor installed.
    """

    return SpinQuantMapping(
        embedding=r"re:.*language_model\.embed_tokens$",
        # Both attention block kinds — llmcompressor walks this to find the
        # parent module hosting q/k/v/o. Scope to language_model.
        attn=r"re:.*language_model\.layers\.\d+\.(self_attn|linear_attn)$",
        # Three "attn input" slots — cram all 5 input projection families in:
        # self_attn has 3 (q/k/v); linear_attn has 4 (a/b/qkv/z). Distribute:
        attn_q=r"re:.*language_model\.layers\.\d+\.(self_attn\.q_proj|linear_attn\.in_proj_qkv)$",
        attn_k=r"re:.*language_model\.layers\.\d+\.(self_attn\.k_proj|linear_attn\.in_proj_a|linear_attn\.in_proj_b)$",
        attn_v=r"re:.*language_model\.layers\.\d+\.(self_attn\.v_proj|linear_attn\.in_proj_z)$",
        # One "attn output" slot — covers self_attn.o_proj AND linear_attn.out_proj.
        attn_o=r"re:.*language_model\.layers\.\d+\.(self_attn\.o_proj|linear_attn\.out_proj)$",
        mlp_in=[
            r"re:.*language_model\.layers\.\d+\.mlp\.gate_proj$",
            r"re:.*language_model\.layers\.\d+\.mlp\.up_proj$",
        ],
        mlp_out=r"re:.*language_model\.layers\.\d+\.mlp\.down_proj$",
        # lm_head is at the top level of the multimodal model, not nested
        # under language_model. Hf save_pretrained writes it as just "lm_head".
        lm_head="lm_head",
    )


def build_qwen35_norm_mappings(model):
    """NormMappings as concrete per-layer module paths.

    Why per-layer absolute paths instead of regex: llmcompressor's
    `match_modules_set` (called from `_fuse_norms`) groups matches by
    lowest common parent while streaming `model.named_modules()`. For
    Llama every layer matches every q/k/v pattern, so groups close per
    layer cleanly. For Qwen3.5 hybrid, regex like `re:.*input_layernorm$`
    matches in layers without the requested projection family (e.g. a
    self_attn-targeted mapping hits linear_attn layers' input_layernorm
    too) — groups never close, norms accumulate across layers, trips
    `assert len(norm) == 1`. Enumerating absolute paths per layer
    sidesteps the streaming-group ambiguity entirely.

    (Diagnosis from codex review of the script, 2026-05-25.)
    """

    named = dict(model.named_modules())
    # Only consider language_model layers — multimodal vision tower has its
    # own norm structure (norm1, norm2 inside each visual block) and must
    # not be rotated.
    layer_prefixes = sorted(
        {
            n.removesuffix(".input_layernorm")
            for n in named
            if n.endswith(".input_layernorm") and "language_model" in n
        },
        key=lambda s: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", s)],
    )

    out: list = []
    for layer in layer_prefixes:
        input_norm = f"{layer}.input_layernorm"
        self_linears = [f"{layer}.self_attn.{p}" for p in ("q_proj", "k_proj", "v_proj")]
        linear_linears = [f"{layer}.linear_attn.{p}" for p in ("in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z")]
        has_self = all(n in named for n in self_linears)
        has_linear = all(n in named for n in linear_linears)
        if has_self == has_linear:
            raise ValueError(f"{layer}: expected exactly one attention projection family (self xor linear)")
        out.append(NormMapping(norm=input_norm, linears=self_linears if has_self else linear_linears))

        post_norm = f"{layer}.post_attention_layernorm"
        mlp_linears = [f"{layer}.mlp.gate_proj", f"{layer}.mlp.up_proj"]
        missing = [n for n in [post_norm, *mlp_linears] if n not in named]
        if missing:
            raise ValueError(f"{layer}: missing modules {missing}")
        out.append(NormMapping(norm=post_norm, linears=mlp_linears))

    # Final pre-lm_head norm — for multimodal layout, this is at
    # `model.language_model.norm`. Vision tower has its own `model.visual.merger.norm`
    # which must be excluded.
    final_norm_candidates = [
        n
        for n in named
        if n.endswith(".norm") and "language_model" in n and ".layers." not in n and "linear_attn" not in n
    ]
    if len(final_norm_candidates) != 1:
        raise ValueError(f"could not uniquely identify final language-model norm: {final_norm_candidates}")
    out.append(NormMapping(norm=final_norm_candidates[0], linears=["lm_head"]))
    return out


def build_qwen35_r1_transform_config():
    """Direct R1 transform_config bypassing SpinQuantModifier's mapping inference.

    Why: per codex review, `SpinQuantModifier.on_initialize()` overwrites
    `self.mappings` and `self.norm_mappings` unless `transform_config` is
    already provided. Passing explicit `mappings`/`norm_mappings` kwargs
    without `transform_config` is silently ignored.

    R1 scheme:
    - weight_output (R applied to row/out axis): embed_tokens (writes
      residual), self_attn.o_proj + linear_attn.out_proj (write residual),
      mlp.down_proj (writes residual)
    - weight_input inverse=True (R.T applied to col/in axis): all attn
      input projections (self_attn q/k/v, linear_attn in_proj_a/b/qkv/z),
      mlp.gate/up_proj (read residual), lm_head (reads residual)
    """
    from compressed_tensors.transform import TransformArgs, TransformConfig, TransformScheme

    # All target patterns are scoped with `language_model\.` so the vision
    # tower (`model.visual.*`) and MTP module (`mtp.*`) are completely
    # untouched. lm_head is at the top level — no language_model prefix.
    return TransformConfig(
        config_groups={
            "R1": TransformScheme(
                type="random-hadamard",
                randomize=False,
                requires_grad=False,
                head_dim=None,
                apply=[
                    TransformArgs(
                        targets=[
                            r"re:.*language_model\.embed_tokens$",
                            r"re:.*language_model\.layers\.\d+\.(self_attn\.o_proj|linear_attn\.out_proj)$",
                            r"re:.*language_model\.layers\.\d+\.mlp\.down_proj$",
                        ],
                        location="weight_output",
                    ),
                    TransformArgs(
                        targets=[
                            r"re:.*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj)|linear_attn\.(in_proj_a|in_proj_b|in_proj_qkv|in_proj_z)|mlp\.(gate_proj|up_proj))$",
                            "lm_head",
                        ],
                        location="weight_input",
                        inverse=True,
                    ),
                ],
            )
        }
    )


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True, help="HF BF16 checkpoint dir to rotate")
    ap.add_argument("--output-path", required=True, help="Where to save the rotated BF16 checkpoint")
    ap.add_argument(
        "--rotations",
        default="R1",
        help="Comma-separated subset of {R1,R2}. Default R1 only. R2 is per-head and questionable for linear_attn layers. R3/R4 are online and not supported here.",
    )
    ap.add_argument(
        "--transform-type",
        default="random-hadamard",
        choices=["hadamard", "random-hadamard", "random-matrix"],
        help="Hidden_size=5120 for Qwen3.6-27B is not a power of 2, so plain hadamard may need block_size. random-hadamard is the safest default.",
    )
    ap.add_argument("--transform-block-size", type=int, default=None, help="Override hidden_size auto-selection")
    return ap.parse_args()


def main():
    args = parse_args()
    rotations = [r.strip() for r in args.rotations.split(",") if r.strip()]

    print(f"[rotate] loading multimodal model from {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    # Multimodal load preserves the vision tower (passes through as BF16,
    # SpinQuant regex doesn't match its modules) and keeps the saved
    # `architectures` tag as Qwen3_5ForConditionalGeneration — the entry
    # sglang has actually registered + production-tested.
    try:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"[rotate] AutoModelForImageTextToText failed ({e!r}); falling back to AutoModel.from_pretrained")
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
    print(
        f"[rotate] model loaded, class={type(model).__name__}, root dtype={next(model.parameters()).dtype}", flush=True
    )

    # Two pieces needed:
    # 1) NormMappings as per-layer absolute paths — sidesteps the
    #    match_modules_set group-closure ambiguity that trips assert
    #    len(norm) == 1 on hybrid arch with regex-only mappings
    # 2) transform_config built directly — SpinQuantModifier.on_initialize
    #    overwrites self.mappings / self.norm_mappings unless
    #    transform_config is already provided, so explicit mappings/
    #    norm_mappings kwargs are otherwise silently ignored
    mapping = build_qwen35_mapping()
    norm_mappings = build_qwen35_norm_mappings(model)
    transform_config = build_qwen35_r1_transform_config()
    from llmcompressor.modifiers.transform.spinquant import mappings as spinquant_mappings
    from llmcompressor.modifiers.transform.spinquant import norm_mappings as spinquant_norm_mappings

    arch_name = type(model).__name__  # e.g. Qwen3_5ForCausalLM
    spinquant_mappings.SPINQUANT_MAPPING_REGISTRY[arch_name] = mapping
    spinquant_norm_mappings.NORM_MAPPING_REGISTRY[arch_name] = norm_mappings
    print(
        f"[rotate] registered SpinQuant + norm mappings for {arch_name}, {len(norm_mappings)} NormMappings", flush=True
    )

    from llmcompressor import oneshot
    from llmcompressor.modifiers.transform import SpinQuantModifier

    # llmcompressor's SpinQuantModifier.on_start unconditionally calls
    # _center_embeddings(model), which subtracts each embedding row's
    # hidden-dim mean BEFORE applying R1. That's an additive per-token
    # perturbation that does NOT cancel through residual rotation +
    # RMSNorm (RMSNorm is scale-invariant but NOT translation-invariant).
    # Result: the saved checkpoint is no longer mathematically equivalent
    # to the original BF16 model.
    #
    # Codex audit (2026-05-25) of our first rotated BF16 100x8 ckpt
    # showing T3 correct -4.3pp identified centering as the root cause.
    # Disable it here for the BF16-equivalence experiment. If we add
    # quantization later we can re-evaluate whether to re-enable it
    # (centering is meant to reduce channel-mean outliers in downstream
    # weight quantization; cost is the rotation-only baseline isn't
    # math-identical to original BF16 anymore).
    class _NoCenterSpinQuantModifier(SpinQuantModifier):
        def _center_embeddings(self, model):
            return

    modifier = _NoCenterSpinQuantModifier(
        rotations=rotations,
        transform_type=args.transform_type,
        mappings=mapping,
        norm_mappings=norm_mappings,
        transform_config=transform_config,
        transform_block_size=args.transform_block_size,
    )

    print(f"[rotate] running SpinQuant rotations={rotations} type={args.transform_type}", flush=True)
    # SpinQuant rotation is deterministic (or random-hadamard seeded); no
    # calibration dataset needed for R1/R2.
    oneshot(model=model, recipe=[modifier])

    # Multimodal load preserves the `Qwen3_5ForConditionalGeneration`
    # architecture tag; no layers_block_type alias hack needed (sglang
    # multimodal entry handles layer-type dispatch correctly).

    print(f"[rotate] saving rotated BF16 to {args.output_path}", flush=True)
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)

    # Copy multimodal preprocessor / vocab / merges from source dir.
    # save_pretrained writes config + tokenizer but not image_processor /
    # video_processor / merges. sglang's multimodal entry refuses to load
    # without preprocessor_config.json.
    import shutil
    from pathlib import Path

    src = Path(args.model_path)
    dst = Path(args.output_path)
    for fname in (
        "preprocessor_config.json",
        "video_preprocessor_config.json",
        "configuration.json",  # ModelScope cooperatively
        "merges.txt",
        "vocab.json",
        "chat_template.jinja",  # in case save_pretrained didn't write it
    ):
        s = src / fname
        d = dst / fname
        if s.is_file() and not d.is_file():
            shutil.copy2(s, d)
            print(f"[rotate] copied multimodal config: {fname}", flush=True)

    print("[rotate] done", flush=True)


if __name__ == "__main__":
    main()
