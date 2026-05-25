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
import re

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
    from llmcompressor.modifiers.transform.spinquant.norm_mappings import NormMapping

    named = dict(model.named_modules())
    layer_prefixes = sorted(
        {n.removesuffix(".input_layernorm") for n in named if n.endswith(".input_layernorm")},
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

    # Final pre-lm_head norm: handle the multimodal-wrapped / flat name variations.
    final_norm = next((n for n in ("language_model.norm", "model.norm", "norm") if n in named), None)
    if final_norm is None:
        candidates = [n for n in named if (n == "norm" or n.endswith(".norm")) and ".layers." not in n]
        if len(candidates) != 1:
            raise ValueError(f"could not uniquely identify final pre-lm_head norm: {candidates}")
        final_norm = candidates[0]
    out.append(NormMapping(norm=final_norm, linears=["lm_head"]))
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
                            r"re:.*embed_tokens$",
                            r"re:.*(self_attn\.o_proj|linear_attn\.out_proj)$",
                            r"re:.*mlp\.down_proj$",
                        ],
                        location="weight_output",
                    ),
                    TransformArgs(
                        targets=[
                            r"re:.*(self_attn\.(q_proj|k_proj|v_proj)|linear_attn\.(in_proj_a|in_proj_b|in_proj_qkv|in_proj_z)|mlp\.(gate_proj|up_proj))$",
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

    modifier = SpinQuantModifier(
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

    print(f"[rotate] saving rotated BF16 to {args.output_path}", flush=True)
    model.save_pretrained(args.output_path)
    tokenizer.save_pretrained(args.output_path)
    print("[rotate] done", flush=True)


if __name__ == "__main__":
    main()
