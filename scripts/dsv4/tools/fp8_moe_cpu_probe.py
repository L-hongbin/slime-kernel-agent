#!/usr/bin/env python3
"""CPU-side FP8 MoE numerical probe for DeepSeek-V4-Flash.

This intentionally avoids importing the local training module.  It compares
source-derived quantization recipes against one real expert from the serialized
FP8 checkpoint.  It is not a replacement for a CUDA DeepGEMM/sgl_kernel probe.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from safetensors.torch import load_file


DEFAULT_CKPT = Path("/nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8")
BLK = 128
FP8_MAX = 448.0
SWIGLU_LIMIT = 10.0


def load_tensor(checkpoint: Path, key: str) -> torch.Tensor:
    with (checkpoint / "model.safetensors.index.json").open() as f:
        index = json.load(f)["weight_map"]
    path = checkpoint / index[key]
    return load_file(str(path), device="cpu")[key]


def dequant_block(w_fp8: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    o, i = w_fp8.shape
    return w_fp8.to(torch.float32).reshape(o // BLK, BLK, i // BLK, BLK).mul(scale[:, None, :, None]).reshape(o, i)


def mcore_manual_block_quant(w: torch.Tensor):
    o, i = w.shape
    ob, ib = o // BLK, i // BLK
    wb = w.to(torch.float32).reshape(ob, BLK, ib, BLK)
    scale = wb.abs().amax(dim=(1, 3)).clamp_min(1e-8) / FP8_MAX
    q = (wb / scale[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).reshape(o, i)
    return q, scale


def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    return torch.exp2(torch.ceil(torch.log2(x.abs())))


def sglang_token_group_quant_formula(x: torch.Tensor, *, ue8m0: bool = False, eps: float = 1e-10):
    m, k = x.shape
    xv = x.to(torch.float32).reshape(m, k // BLK, BLK)
    scale = xv.abs().amax(dim=2).clamp_min(eps) / FP8_MAX
    if ue8m0:
        scale = ceil_to_ue8m0(scale)
    q = (xv / scale[:, :, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).reshape(m, k)
    return q, scale


def dequant_token_group(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    m, k = q.shape
    return q.to(torch.float32).reshape(m, k // BLK, BLK).mul(scale[:, :, None]).reshape(m, k)


def summarize_quant(name, q_ref, s_ref, q_cmp, s_cmp):
    q_diff = q_ref.to(torch.float32) != q_cmp.to(torch.float32)
    scale_abs = (s_ref - s_cmp).abs()
    scale_rel = scale_abs / s_ref.abs().clamp_min(1e-12)
    print(
        f"{name}: q_mismatch={q_diff.float().mean().item():.6f} "
        f"q_max_abs_value_delta={(q_ref.to(torch.float32)-q_cmp.to(torch.float32)).abs().max().item():.6g} "
        f"scale_max_abs={scale_abs.max().item():.6g} "
        f"scale_max_rel={scale_rel.max().item():.6g}"
    )
    print("  ref_q[:8]=", q_ref.flatten()[:8].to(torch.float32).tolist())
    print("  cmp_q[:8]=", q_cmp.flatten()[:8].to(torch.float32).tolist())
    print("  ref_s[:8]=", s_ref.flatten()[:8].tolist())
    print("  cmp_s[:8]=", s_cmp.flatten()[:8].tolist())


def fp8_linear(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    *,
    act_recipe: str,
) -> torch.Tensor:
    if act_recipe == "sglang_linear":
        x_q, x_s = sglang_token_group_quant_formula(x, ue8m0=False)
    elif act_recipe == "deepgemm_linear":
        from deep_gemm import per_token_cast_to_fp8

        x_q, x_s = per_token_cast_to_fp8(x, use_ue8m0=False)
    elif act_recipe == "deepgemm_ue8m0":
        from deep_gemm import per_token_cast_to_fp8

        x_q, x_s = per_token_cast_to_fp8(x, use_ue8m0=True)
    else:
        raise ValueError(act_recipe)
    return (dequant_token_group(x_q, x_s) @ dequant_block(w_q, w_s).T).to(torch.bfloat16)


def moe_forward(x, w13_q, w13_s, w2_q, w2_s, *, act_recipe):
    gate_up = fp8_linear(x, w13_q, w13_s, act_recipe=act_recipe)
    gate, up = gate_up.chunk(2, dim=-1)
    gate = gate.clamp(max=SWIGLU_LIMIT)
    up = up.clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT)
    hidden = (torch.nn.functional.silu(gate.to(torch.float32)) * up.to(torch.float32)).to(torch.bfloat16)
    return fp8_linear(hidden, w2_q, w2_s, act_recipe=act_recipe)


def compare_outputs(name, ref, cmp):
    diff = (ref.to(torch.float32) - cmp.to(torch.float32)).abs()
    print(
        f"{name}: mean_abs={diff.mean().item():.6g} "
        f"p99={diff.flatten().quantile(0.99).item():.6g} max_abs={diff.max().item():.6g}"
    )
    print("  ref[:8]=", ref.flatten()[:8].to(torch.float32).tolist())
    print("  cmp[:8]=", cmp.flatten()[:8].to(torch.float32).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--expert", type=int, default=0)
    args = parser.parse_args()

    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    torch.manual_seed(1234)

    base = f"layers.{args.layer}.ffn.experts.{args.expert}"
    w1_q = load_tensor(args.checkpoint, f"{base}.w1.weight")
    s1 = load_tensor(args.checkpoint, f"{base}.w1.scale")
    w3_q = load_tensor(args.checkpoint, f"{base}.w3.weight")
    s3 = load_tensor(args.checkpoint, f"{base}.w3.scale")
    w2_q = load_tensor(args.checkpoint, f"{base}.w2.weight")
    s2 = load_tensor(args.checkpoint, f"{base}.w2.scale")
    print(f"checkpoint={args.checkpoint}")
    print(f"layer={args.layer} expert={args.expert}")
    print("w1", tuple(w1_q.shape), w1_q.dtype, tuple(s1.shape), s1.dtype)
    print("w3", tuple(w3_q.shape), w3_q.dtype, tuple(s3.shape), s3.dtype)
    print("w2", tuple(w2_q.shape), w2_q.dtype, tuple(s2.shape), s2.dtype)

    w13_q = torch.cat([w1_q, w3_q], dim=0)
    s13 = torch.cat([s1, s3], dim=0)
    w13_bf16 = dequant_block(w13_q, s13).to(torch.bfloat16)
    w2_bf16 = dequant_block(w2_q, s2).to(torch.bfloat16)

    w13_m_q, s13_m = mcore_manual_block_quant(w13_bf16)
    w2_m_q, s2_m = mcore_manual_block_quant(w2_bf16)
    summarize_quant("weight w13 original_vs_mcore_requant", w13_q, s13, w13_m_q, s13_m)
    summarize_quant("weight w2 original_vs_mcore_requant", w2_q, s2, w2_m_q, s2_m)

    from deep_gemm import per_block_cast_to_fp8, per_token_cast_to_fp8

    w13_dg_q, s13_dg = per_block_cast_to_fp8(w13_bf16.float(), use_ue8m0=False)
    summarize_quant(
        "weight w13 original_vs_deepgemm_requant_linear",
        w13_q,
        s13,
        w13_dg_q,
        s13_dg,
    )
    w13_dg_u_q, s13_dg_u = per_block_cast_to_fp8(w13_bf16.float(), use_ue8m0=True)
    summarize_quant(
        "weight w13 original_vs_deepgemm_requant_ue8m0",
        w13_q,
        s13,
        w13_dg_u_q,
        s13_dg_u,
    )

    x = (torch.randn(8, w1_q.shape[1]) * 0.7).to(torch.bfloat16)
    q_sgl, sc_sgl = sglang_token_group_quant_formula(x, ue8m0=False)
    q_dg, sc_dg = per_token_cast_to_fp8(x, use_ue8m0=False)
    q_dg_u, sc_dg_u = per_token_cast_to_fp8(x, use_ue8m0=True)
    summarize_quant("activation sglang_formula_vs_deepgemm_linear", q_sgl, sc_sgl, q_dg, sc_dg)
    summarize_quant("activation sglang_formula_vs_deepgemm_ue8m0", q_sgl, sc_sgl, q_dg_u, sc_dg_u)

    tiny = (torch.randn(2, w1_q.shape[1]) * 1e-5).to(torch.bfloat16)
    tq_sgl, ts_sgl = sglang_token_group_quant_formula(tiny, ue8m0=False)
    tq_dg, ts_dg = per_token_cast_to_fp8(tiny, use_ue8m0=False)
    summarize_quant(
        "tiny_activation_sglang_formula_vs_deepgemm_linear",
        tq_sgl,
        ts_sgl,
        tq_dg,
        ts_dg,
    )

    x_small = x[:4]
    gate_up_ref = x_small.to(torch.float32) @ w13_bf16.to(torch.float32).T
    gate, up = gate_up_ref.to(torch.bfloat16).chunk(2, dim=-1)
    ref_hidden = (
        torch.nn.functional.silu(gate.clamp(max=SWIGLU_LIMIT).to(torch.float32))
        * up.clamp(min=-SWIGLU_LIMIT, max=SWIGLU_LIMIT).to(torch.float32)
    ).to(torch.bfloat16)
    bf16_ref = (ref_hidden.to(torch.float32) @ w2_bf16.to(torch.float32).T).to(torch.bfloat16)
    sgl_like = moe_forward(x_small, w13_q, s13, w2_q, s2, act_recipe="sglang_linear")
    local_linear = moe_forward(x_small, w13_m_q, s13_m, w2_m_q, s2_m, act_recipe="deepgemm_linear")
    local_ue8m0 = moe_forward(x_small, w13_m_q, s13_m, w2_m_q, s2_m, act_recipe="deepgemm_ue8m0")
    local_weights_sgl_act = moe_forward(x_small, w13_m_q, s13_m, w2_m_q, s2_m, act_recipe="sglang_linear")
    orig_weights_dg_ue8m0_act = moe_forward(x_small, w13_q, s13, w2_q, s2, act_recipe="deepgemm_ue8m0")

    compare_outputs("output bf16_ref_vs_sglang_like", bf16_ref, sgl_like)
    compare_outputs("output bf16_ref_vs_local_linear", bf16_ref, local_linear)
    compare_outputs("output bf16_ref_vs_local_ue8m0", bf16_ref, local_ue8m0)
    compare_outputs("output sglang_like_vs_local_linear", sgl_like, local_linear)
    compare_outputs("output sglang_like_vs_local_ue8m0", sgl_like, local_ue8m0)
    compare_outputs("output sglang_like_vs_local_weights_only", sgl_like, local_weights_sgl_act)
    compare_outputs(
        "output sglang_like_vs_ue8m0_activation_only",
        sgl_like,
        orig_weights_dg_ue8m0_act,
    )


if __name__ == "__main__":
    main()
