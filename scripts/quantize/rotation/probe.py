"""Forward-divergence probe: measure per-layer hidden state and logit divergence
between an original BF16 model and a modified-weight model.

Run multiple times with different ckpts / variants, save outputs to .pt files,
then compare with compare_rotation_probe.py."""

import argparse
import torch

PROMPTS = [
    "Hello, world. The quick brown fox jumps over the lazy dog.",
    "import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n\n    def forward(self, x):\n        return x.relu()",
    "def fibonacci(n: int) -> int:\n    if n < 2:\n        return n\n",
    "The Cuda kernel for a 2D convolution operation",
    "什么是 attention 机制？",
    "List 5 ways to optimize a matrix multiplication kernel:",
    "Quantum entanglement is a phenomenon where",
    "SELECT user_id, COUNT(*) FROM events WHERE",
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="HF ckpt to load")
    ap.add_argument("--output", required=True, help="output .pt file")
    ap.add_argument(
        "--variant",
        default="raw",
        choices=[
            "raw",
            "roundtrip_only",
            "fuse_bf16",
            "fuse_fp32",
            "pipeline_fp32",
            "pipeline_fp64",
            "pipeline_fused_bf16",
            "pipeline_fused_fp64_to_bf16",
            "pipeline_restore_norm_bf16",
            "pipeline_restore_norm_fp32",
            "pipeline_restore_norm_fp64",
            "pipeline_restore_norm_fp64_to_bf16",
            "raw_fp32",
            "raw_fp64",
        ],
        help="weight modification to apply before forward (raw = no modification). "
        "raw_fp32/fp64: just cast model to target dtype with NO other modifications "
        "(control for fp32/fp64 forward arithmetic baseline). "
        "pipeline_fp32/fp64: cast model to target dtype + apply offset-norm fusion + Hadamard rotation, "
        "all in target dtype storage. Forward also runs in target dtype. "
        "pipeline_fused_bf16: apply the fp32 fusion+rotation pipeline, then cast to bf16 for forward. "
        "pipeline_fused_fp64_to_bf16: same fused pipeline using fp64 intermediates, then cast to bf16. "
        "pipeline_restore_norm_bf16: apply the fp32 fusion+rotation pipeline, inverse-fuse original "
        "Qwen offset-norm weights back into the RMSNorm modules, then cast to bf16 for forward. "
        "pipeline_restore_norm_fp32/fp64: same inverse-fuse layout, kept in fp32/fp64 for forward. "
        "pipeline_restore_norm_fp64_to_bf16: inverse-fuse using fp64 intermediates, then cast to bf16.",
    )
    return ap.parse_args()


def apply_roundtrip_only(model):
    """Apply offset-norm temp BF16 round-trip to every Qwen3_5RMSNorm: w ← BF16(BF16(1+w)-1)."""
    n = 0
    for mod in model.modules():
        if mod.__class__.__name__ == "Qwen3_5RMSNorm":
            w = mod.weight.data
            tmp_bf16 = (1.0 + w.float()).bfloat16()
            restored = (tmp_bf16.float() - 1.0).bfloat16()
            mod.weight.data.copy_(restored)
            n += 1
    print(f"[roundtrip_only] modified {n} norms", flush=True)


def find_norm_linear_pairs(model):
    """Return list of (norm_module, [linear_modules]) pairs matching our residual fusion mappings."""
    named = dict(model.named_modules())
    pairs = []
    for n in named:
        if not (n.endswith(".input_layernorm") and "language_model" in n):
            continue
        norm = named[n]
        prefix = n.removesuffix(".input_layernorm")
        # Try self_attn projections first
        s_keys = [f"{prefix}.self_attn.{p}" for p in ("q_proj", "k_proj", "v_proj")]
        l_keys = [f"{prefix}.linear_attn.{p}" for p in ("in_proj_a", "in_proj_b", "in_proj_qkv", "in_proj_z")]
        if all(k in named for k in s_keys):
            pairs.append((norm, [named[k] for k in s_keys]))
        elif all(k in named for k in l_keys):
            pairs.append((norm, [named[k] for k in l_keys]))
        # post_attention_layernorm → mlp.{gate,up}
        post_norm_name = f"{prefix}.post_attention_layernorm"
        if post_norm_name in named:
            mlp_keys = [f"{prefix}.mlp.gate_proj", f"{prefix}.mlp.up_proj"]
            if all(k in named for k in mlp_keys):
                pairs.append((named[post_norm_name], [named[k] for k in mlp_keys]))
    # final norm → lm_head
    for n in named:
        if n.endswith(".norm") and "language_model" in n and ".layers." not in n and "linear_attn" not in n:
            if "lm_head" in named:
                pairs.append((named[n], [named["lm_head"]]))
            break
    return pairs


def apply_fuse(model, fp32_temp: bool):
    """Emulate offset-norm conversion + fuse_norm_linears, without rotation.
    fp32_temp=True keeps the temp scale (1+w) in FP32; False uses BF16 (current llmcompressor)."""
    pairs = find_norm_linear_pairs(model)
    label = "fuse_fp32" if fp32_temp else "fuse_bf16"
    print(f"[{label}] found {len(pairs)} norm-linear fusion pairs", flush=True)
    n_norms_touched = 0
    for norm, linears in pairs:
        w_fp32 = norm.weight.data.float()
        s = 1.0 + w_fp32  # (1+w) in FP32
        if not fp32_temp:
            s = s.bfloat16().float()  # round-trip via BF16
        for lin in linears:
            # Linear.weight shape (out, in). Absorb s on channel (input) axis: W[:, j] *= s[j]
            lin.weight.data = (lin.weight.data.float() * s.unsqueeze(0)).bfloat16()
        # After fusion, norm becomes identity. For offset-norm: (1 + 0) = 1 = identity, so set w = 0
        norm.weight.data.zero_()
        n_norms_touched += 1
    # Also round-trip un-fused Qwen3_5RMSNorm (q_norm, k_norm) since
    # llmcompressor's norm_calibration_context touches ALL of them, not just our fused ones.
    n_unfused_touched = 0
    fused_norms = {id(p[0]) for p in pairs}
    for mod in model.modules():
        if mod.__class__.__name__ == "Qwen3_5RMSNorm" and id(mod) not in fused_norms:
            w = mod.weight.data
            if fp32_temp:
                # FP32 temp: (1+w).fp32 - 1 = w exactly, then cast to BF16
                # If original was BF16, this is also BF16 → no change. Skip.
                pass
            else:
                tmp_bf16 = (1.0 + w.float()).bfloat16()
                restored = (tmp_bf16.float() - 1.0).bfloat16()
                mod.weight.data.copy_(restored)
            n_unfused_touched += 1
    print(
        f"[{label}] fused {n_norms_touched} norms into linears, perturbed {n_unfused_touched} unfused norms",
        flush=True,
    )


def apply_pipeline(model, target_dtype):
    """Full pipeline (offset-norm fuse + Hadamard rotation) staying in target_dtype.

    Cast model to target_dtype FIRST, then apply fusion + rotation with no BF16
    cast anywhere. Tests whether BF16 storage truncation (rather than the
    operations themselves) is the dominant cost.

    target_dtype: torch.float32 or torch.float64
    """
    import math
    import re

    label = f"pipeline_{str(target_dtype).rsplit('.', 1)[-1]}"

    # Cast all params to target_dtype
    print(f"[{label}] casting model to {target_dtype}", flush=True)
    for p in model.parameters():
        p.data = p.data.to(target_dtype)
    for b in model.buffers():
        if b.is_floating_point():
            b.data = b.data.to(target_dtype)

    # Step 1: offset-norm fusion in target dtype (no BF16 cast)
    pairs = find_norm_linear_pairs(model)
    for norm, linears in pairs:
        w = norm.weight.data.to(target_dtype)
        s = 1.0 + w
        for lin in linears:
            lin.weight.data = lin.weight.data.to(target_dtype) * s.unsqueeze(0)
        norm.weight.data.zero_()
    print(f"[{label}] fused {len(pairs)} norm pairs in {target_dtype}", flush=True)

    # Step 2: build Hadamard matrix (always in fp64 internally, then we cast at apply time)
    from compressed_tensors.transform.utils.hadamard import random_hadamard_matrix

    hidden = model.config.text_config.hidden_size if hasattr(model.config, "text_config") else model.config.hidden_size
    # Build R on CPU first to avoid device_map allocation surprises, then move per-module
    gen = torch.Generator(device="cpu").manual_seed(42)
    R_fp64 = random_hadamard_matrix(hidden, dtype=torch.float64, device=torch.device("cpu"), gen=gen)
    R_fp64 = R_fp64 / math.sqrt(hidden)  # orthogonalize
    print(f"[{label}] built Hadamard R of size {hidden}x{hidden} in fp64", flush=True)

    OUT_PATTERNS = [
        r".*language_model\.embed_tokens$",
        r".*language_model\.layers\.\d+\.(self_attn\.o_proj|linear_attn\.out_proj)$",
        r".*language_model\.layers\.\d+\.mlp\.down_proj$",
    ]
    IN_PATTERNS = [
        r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj)|linear_attn\.(in_proj_a|in_proj_b|in_proj_qkv|in_proj_z)|mlp\.(gate_proj|up_proj))$",
        r"^lm_head$",
    ]

    n_out = n_in = 0
    for name, mod in dict(model.named_modules()).items():
        if isinstance(mod, torch.nn.Embedding):
            if any(re.match(p, name) for p in OUT_PATTERNS):
                R = R_fp64.to(device=mod.weight.device)
                # Embedding weight (vocab, hidden), output side: W @ R
                W64 = mod.weight.data.to(torch.float64) @ R
                mod.weight.data = W64.to(target_dtype)
                n_out += 1
        elif isinstance(mod, torch.nn.Linear):
            R = R_fp64.to(device=mod.weight.device)
            if any(re.match(p, name) for p in OUT_PATTERNS):
                # Linear weight (out, in), output side: W_new = R.T @ W
                W64 = R.T @ mod.weight.data.to(torch.float64)
                mod.weight.data = W64.to(target_dtype)
                n_out += 1
            elif any(re.match(p, name) for p in IN_PATTERNS):
                # Linear weight (out, in), input side inverse: W_new = W @ R
                W64 = mod.weight.data.to(torch.float64) @ R
                mod.weight.data = W64.to(target_dtype)
                n_in += 1
    print(f"[{label}] rotated {n_out} output-side + {n_in} input-side modules in {target_dtype}", flush=True)


def apply_pipeline_restore_norm(model, target_dtype, final_dtype=None):
    """Rotate, then inverse-fuse Qwen offset norms before forward.

    This tests the proposed checkpoint layout:
      1. keep the original Qwen3_5RMSNorm residual weights near zero,
      2. compensate every directly following Linear by dividing its input
         columns by the same scale, and
      3. store/run the final model in target_dtype or final_dtype.

    A naive "put the norm weights back" would double-apply the scale. The
    inverse-fuse step below keeps the transformed model algebraically equivalent
    to the fused rotated pipeline before any final storage cast.
    """

    label = f"pipeline_restore_norm_{str(final_dtype or target_dtype).rsplit('.', 1)[-1]}"
    print(f"[{label}] stashing original fused norm weights", flush=True)
    pairs = find_norm_linear_pairs(model)
    original_norm_weights = {id(norm): norm.weight.data.to(target_dtype).clone() for norm, _ in pairs}

    apply_pipeline(model, target_dtype=target_dtype)

    print(f"[{label}] inverse-fusing original norms into rotated linears", flush=True)
    for norm, linears in pairs:
        w = original_norm_weights[id(norm)].to(norm.weight.device)
        s = 1.0 + w
        for lin in linears:
            lin.weight.data = lin.weight.data.to(target_dtype) / s.to(lin.weight.device).unsqueeze(0)
        norm.weight.data = w.to(norm.weight.device)

    if final_dtype is None:
        return

    print(f"[{label}] casting restored model to {final_dtype}", flush=True)
    for p in model.parameters():
        if p.is_floating_point():
            p.data = p.data.to(final_dtype)
    for b in model.buffers():
        if b.is_floating_point():
            b.data = b.data.to(final_dtype)


def cast_model_bf16(model, label):
    print(f"[{label}] casting model to bf16", flush=True)
    for p in model.parameters():
        if p.is_floating_point():
            p.data = p.data.bfloat16()
    for b in model.buffers():
        if b.is_floating_point():
            b.data = b.data.bfloat16()


def main():
    args = parse_args()
    print(f"[probe] loading {args.model_path} (variant={args.variant})", flush=True)

    from transformers import AutoTokenizer

    try:
        from transformers import AutoModelForImageTextToText as AutoModelCls
    except ImportError:
        from transformers import AutoModel as AutoModelCls

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelCls.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[probe] model loaded, class={type(model).__name__}", flush=True)

    if args.variant == "roundtrip_only":
        apply_roundtrip_only(model)
    elif args.variant == "fuse_bf16":
        apply_fuse(model, fp32_temp=False)
    elif args.variant == "fuse_fp32":
        apply_fuse(model, fp32_temp=True)
    elif args.variant == "pipeline_fp32":
        apply_pipeline(model, target_dtype=torch.float32)
    elif args.variant == "pipeline_fp64":
        apply_pipeline(model, target_dtype=torch.float64)
    elif args.variant == "pipeline_fused_bf16":
        apply_pipeline(model, target_dtype=torch.float32)
        cast_model_bf16(model, "pipeline_fused_bf16")
    elif args.variant == "pipeline_fused_fp64_to_bf16":
        apply_pipeline(model, target_dtype=torch.float64)
        cast_model_bf16(model, "pipeline_fused_fp64_to_bf16")
    elif args.variant == "pipeline_restore_norm_bf16":
        apply_pipeline_restore_norm(model, target_dtype=torch.float32, final_dtype=torch.bfloat16)
    elif args.variant == "pipeline_restore_norm_fp32":
        apply_pipeline_restore_norm(model, target_dtype=torch.float32)
    elif args.variant == "pipeline_restore_norm_fp64":
        apply_pipeline_restore_norm(model, target_dtype=torch.float64)
    elif args.variant == "pipeline_restore_norm_fp64_to_bf16":
        apply_pipeline_restore_norm(model, target_dtype=torch.float64, final_dtype=torch.bfloat16)
    elif args.variant == "raw_fp32":
        # Control: just upcast, no modifications. Tests pure fp32 forward vs bf16 forward.
        print("[raw_fp32] casting model to fp32 (no other modifications)", flush=True)
        for p in model.parameters():
            p.data = p.data.float()
        for b in model.buffers():
            if b.is_floating_point():
                b.data = b.data.float()
    elif args.variant == "raw_fp64":
        print("[raw_fp64] casting model to fp64 (no other modifications)", flush=True)
        for p in model.parameters():
            p.data = p.data.double()
        for b in model.buffers():
            if b.is_floating_point():
                b.data = b.data.double()
    elif args.variant == "raw":
        pass

    # Forward pass with output_hidden_states
    print(f"[probe] running forward on {len(PROMPTS)} prompts", flush=True)
    inputs = tok(PROMPTS, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inputs = {k: v.to(next(model.parameters()).device) for k, v in inputs.items()}
    with torch.no_grad():
        # For multimodal model, only pass text inputs (no pixel_values etc.)
        out = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            output_hidden_states=True,
            use_cache=False,
        )
    # out.hidden_states is a tuple of (num_layers+1) tensors: input embedding + after each layer
    hs = torch.stack([h.detach().float().cpu() for h in out.hidden_states])
    logits = out.logits.detach().float().cpu()
    torch.save(
        {
            "hidden_states": hs,
            "logits": logits,
            "prompts": PROMPTS,
            "input_ids": inputs["input_ids"].cpu(),
            "variant": args.variant,
            "model_path": args.model_path,
        },
        args.output,
    )
    print(f"[probe] saved {args.output}: hs shape {tuple(hs.shape)}, logits {tuple(logits.shape)}", flush=True)


if __name__ == "__main__":
    main()
