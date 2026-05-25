"""Apply offline-only SpinQuant rotation to a Qwen3.5/3.6 text-only model.

Saves BF16 (no quantization). The point of this script is to validate that the
rotation math is preserved across Qwen3.5's hybrid attention (linear_attn +
self_attn) BEFORE adding GPTQ W8A8 on top. If the rotated BF16 model produces
the same KernelBench accuracy as the unrotated BF16 model (within noise), we
know:
  - the custom Qwen3.5 SpinQuant mapping correctly enumerates every
    residual-facing projection (so R1 rotation cancels through every layer)
  - llmcompressor's fusion of RMSNorm scales into the rotated projections
    worked for the hybrid architecture
  - sglang loads the rotated weights correctly (loader ignores
    transform_config; rotation is already baked into the saved weights)

If accuracy degrades significantly, the mapping is wrong somewhere — most
likely the linear_attn projections aren't covered correctly. Codex's prior
audit identified the four residual-facing input projections of linear_attn
(`in_proj_a`, `in_proj_b`, `in_proj_qkv`, `in_proj_z`) plus `out_proj` as
the things R1 must propagate through. Internal SSM tensors (conv1d, A_log,
dt_bias, norm) operate in post-projection basis and must NOT be rotated.

Loads text-only (`AutoModelForCausalLM.from_pretrained`) so the vision
tower isn't instantiated. This rewrites `architectures` to
`Qwen3_5ForCausalLM` in the saved config; sglang load then requires
`scripts/quantize/sglang_qwen3_5_dense_entry.patch` to be applied.

Usage on .22:
    source /tmp/w8a8-venv/bin/activate
    python scripts/quantize/rotate_bf16_llmcompressor.py \\
        --model-path /nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B \\
        --output-path /nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B-rotated-bf16
"""

from __future__ import annotations

import argparse
import os

import torch

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
    """Construct the SpinQuantMapping for Qwen3.5/3.6 hybrid arch.

    Done lazily so the script's --help works without llmcompressor installed.
    """
    from llmcompressor.modifiers.transform.spinquant.mappings import SpinQuantMapping

    return SpinQuantMapping(
        embedding="re:.*embed_tokens$",
        # Both attention block kinds — llmcompressor walks this to find the
        # parent module hosting q/k/v/o.
        attn="re:.*(self_attn|linear_attn)$",
        # Three "attn input" slots — cram all 5 input projections in:
        # self_attn has 3 (q/k/v); linear_attn has 4 (a/b/qkv/z). Distribute:
        attn_q=r"re:.*(self_attn\.q_proj|linear_attn\.in_proj_qkv)$",
        attn_k=r"re:.*(self_attn\.k_proj|linear_attn\.in_proj_a|linear_attn\.in_proj_b)$",
        attn_v=r"re:.*(self_attn\.v_proj|linear_attn\.in_proj_z)$",
        # One "attn output" slot — covers self_attn.o_proj AND linear_attn.out_proj.
        attn_o=r"re:.*(self_attn\.o_proj|linear_attn\.out_proj)$",
        mlp_in=[r"re:.*mlp\.gate_proj$", r"re:.*mlp\.up_proj$"],
        mlp_out=r"re:.*mlp\.down_proj$",
        lm_head="lm_head",
    )


def build_qwen35_norm_mappings():
    """NormMappings describe RMSNorm scale fusion targets.

    For SpinQuant to fuse RMSNorm's per-channel scale into the downstream
    projection (so rotation can be applied cleanly), each NormMapping
    declares: this norm's output feeds these linears.

    Qwen3.5 hybrid: each of the 64 layers has input_layernorm (feeding
    attention) and post_attention_layernorm (feeding MLP). The 16
    self_attn layers' input_layernorm feeds q/k/v_proj; the 48
    linear_attn layers' input_layernorm feeds in_proj_a/b/qkv/z.
    Express as two separate NormMapping entries with the same norm
    pattern but different linear targets — match_modules_set will pair
    each norm with whichever linear set actually exists in its layer.
    """
    from llmcompressor.modifiers.transform.spinquant.norm_mappings import NormMapping

    return [
        # self_attn layers (16): input_layernorm → q/k/v_proj
        NormMapping(
            norm=r"re:.*input_layernorm$",
            linears=[
                r"re:.*self_attn\.q_proj$",
                r"re:.*self_attn\.k_proj$",
                r"re:.*self_attn\.v_proj$",
            ],
        ),
        # linear_attn layers (48): input_layernorm → in_proj_a/b/qkv/z
        NormMapping(
            norm=r"re:.*input_layernorm$",
            linears=[
                r"re:.*linear_attn\.in_proj_a$",
                r"re:.*linear_attn\.in_proj_b$",
                r"re:.*linear_attn\.in_proj_qkv$",
                r"re:.*linear_attn\.in_proj_z$",
            ],
        ),
        # All 64 layers: post_attention_layernorm → mlp.up/gate_proj
        NormMapping(
            norm=r"re:.*post_attention_layernorm$",
            linears=[
                r"re:.*mlp\.up_proj$",
                r"re:.*mlp\.gate_proj$",
            ],
        ),
        # Final norm before lm_head. Use a tight regex so it doesn't
        # accidentally match linear_attn.norm (internal SSM norm).
        NormMapping(
            norm=r"re:.*language_model\.norm$",
            linears=["lm_head"],
        ),
    ]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True, help="HF BF16 checkpoint dir to rotate")
    ap.add_argument("--output-path", required=True, help="Where to save the rotated BF16 checkpoint")
    ap.add_argument(
        "--rotations",
        default="R1",
        help="Comma-separated subset of {R1,R2}. Default R1 only. R2 is per-head and "
        "questionable for linear_attn layers. R3/R4 are online and not supported here.",
    )
    ap.add_argument(
        "--transform-type",
        default="random-hadamard",
        choices=["hadamard", "random-hadamard", "random-matrix"],
        help="Hidden_size=5120 for Qwen3.6-27B is not a power of 2, so plain hadamard may "
        "need block_size. random-hadamard is the safest default.",
    )
    ap.add_argument("--transform-block-size", type=int, default=None, help="Override hidden_size auto-selection")
    return ap.parse_args()


def main():
    args = parse_args()
    rotations = [r.strip() for r in args.rotations.split(",") if r.strip()]

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[rotate] loading text-only model from {args.model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"[rotate] model loaded, root dtype={next(model.parameters()).dtype}", flush=True)

    # Register the Qwen3.5 mapping in the SpinQuant registry so internal
    # `infer_mapping_from_model` resolves it by class name. Belt-and-braces:
    # we also pass it explicitly via `mappings=` below.
    mapping = build_qwen35_mapping()
    norm_mappings = build_qwen35_norm_mappings()
    from llmcompressor.modifiers.transform.spinquant import mappings as spinquant_mappings
    from llmcompressor.modifiers.transform.spinquant import norm_mappings as spinquant_norm_mappings

    arch_name = type(model).__name__  # e.g. Qwen3_5ForCausalLM
    spinquant_mappings.SPINQUANT_MAPPING_REGISTRY[arch_name] = mapping
    spinquant_norm_mappings.NORM_MAPPING_REGISTRY[arch_name] = norm_mappings
    print(f"[rotate] registered SpinQuant + norm mappings for {arch_name}", flush=True)

    from llmcompressor import oneshot
    from llmcompressor.modifiers.transform import SpinQuantModifier

    modifier = SpinQuantModifier(
        rotations=rotations,
        transform_type=args.transform_type,
        mappings=mapping,
        norm_mappings=norm_mappings,
        transform_block_size=args.transform_block_size,
    )

    print(f"[rotate] running SpinQuant rotations={rotations} type={args.transform_type}", flush=True)
    # SpinQuant rotation is deterministic (or random-hadamard seeded); no
    # calibration dataset needed for R1/R2.
    oneshot(model=model, recipe=[modifier])

    print(f"[rotate] saving rotated BF16 to {args.output_path}", flush=True)
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print("[rotate] done", flush=True)


if __name__ == "__main__":
    main()
