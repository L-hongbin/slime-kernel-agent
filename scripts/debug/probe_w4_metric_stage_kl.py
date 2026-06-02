#!/usr/bin/env python3
"""Disambiguate 'is L2 a bad metric' vs 'is the MLP-output stage too local' for W4 vs W8-A8.

On the SAME unbiased held-out UltraChat prompts behind the 39.86 / 38.32 MLP-output
L2 numbers, run the FULL model with MLP patched to three variants and compare, at
the FINAL logits stage, both L2 and KL to BF16:

  variants (mirroring the L2 table exactly):
    bf16        : source MLP
    w8a8        : source MLP weights, per-output-channel symmetric INT8 RTN, + per-token A8 fake-quant
    w4_awq_asym : AWQ asym dequant MLP weights, A16 (no activation quant)

  metrics per variant (mean over sampled valid positions, vs bf16):
    final_logits_L2  : || logits_q - logits_bf16 ||_2          (same L2 currency, but at the OUTPUT stage)
    final_logits_KL  : KL(softmax(bf16) || softmax(q))         (ranking/shape-aware)
    top1_agree, top50_overlap

Read together with the MLP-output L2 table (W4=39.86 ~= W8-A8=38.32), this tells us:
  * if final_logits_L2 already separates W4 from W8A8 -> the STAGE was the issue (local MLP L2 misses propagation)
  * if final_logits_L2 stays similar but KL separates  -> the METRIC was the issue (L2 ignores ranking/shape)
Both can contribute; the numbers give the decomposition.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch

REPO = Path("/nfs/FM/chenshuailin/projects/kernel_agents/slime")
sys.path.insert(0, str(REPO))

import types

from scripts.quantize.producers.awq_w4a16 import patch_transformers_broken_torchvision
from scripts.quantize.utils.compare_mlp_quant_loss import (
    dequant_w4,
    load_bf16_weight,
    load_tensor,
    load_weight_map,
    tensor_name,
    text_config,
)

PROJ = ("gate_proj", "up_proj", "down_proj")
DEFAULT_SOURCE = Path("/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B")
DEFAULT_W4 = REPO / "checkpoints/quantized/AWQ/Qwen3.6-27B-AWQ-W4A16-asym-mlp"
DEFAULT_PROMPTS = REPO / "checkpoints/quantized/analysis/real_calib_quant_loss_20260601/calibration_prompts.jsonl"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--w4-asym", type=Path, default=DEFAULT_W4)
    ap.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--max-prompts", type=int, default=16)
    ap.add_argument("--pos-per-prompt", type=int, default=128)
    ap.add_argument("--seed", type=int, default=20260602)
    ap.add_argument(
        "--output-json",
        type=Path,
        default=REPO / "checkpoints/quantized/analysis/w4_asym_zp_sensitivity/metric_stage_kl.json",
    )
    return ap.parse_args()


def find_module(model, suffix):
    mods = dict(model.named_modules())
    for name in (f"model.language_model.layers.{suffix}", f"language_model.layers.{suffix}", f"model.layers.{suffix}"):
        if name in mods:
            return mods[name]
    for name, m in mods.items():
        if name.endswith(f".layers.{suffix}"):
            return m
    raise KeyError(suffix)


def set_param(module, name, value):
    old = getattr(module, name)
    setattr(module, name, torch.nn.Parameter(value.to(device=old.device, dtype=old.dtype), requires_grad=False))


def fake_a8(x):
    xf = x.float()
    s = xf.abs().amax(-1, keepdim=True) / 127.0
    s = torch.where(s == 0, torch.ones_like(s), s)
    return (torch.round(xf / s).clamp(-127, 127) * s).to(x.dtype)


def qdq_w8_channel(w):
    wf = w.float()
    s = wf.abs().amax(1, keepdim=True) / 127.0
    s = torch.where(s == 0, torch.ones_like(s), s)
    return torch.round(wf / s).clamp(-127, 127) * s


def make_fwd(use_a8):
    def fwd(self, x):
        li = fake_a8(x) if use_a8 else x
        h = torch.nn.functional.silu(self.gate_proj(li)) * self.up_proj(li)
        return self.down_proj(fake_a8(h) if use_a8 else h)

    return fwd


def patch(model, num_layers, kind, src, smap, w4root, w4map):
    use_a8 = kind == "w8a8"
    for L in range(num_layers):
        norm = find_module(model, f"{L}.post_attention_layernorm")
        if kind == "w4_awq_asym":
            set_param(
                norm, "weight", load_tensor(w4root, w4map, tensor_name(L, "post_attention_layernorm.weight")).float()
            )
        else:
            set_param(
                norm, "weight", load_tensor(src, smap, tensor_name(L, "post_attention_layernorm.weight")).float()
            )
        mlp = find_module(model, f"{L}.mlp")
        mlp.forward = types.MethodType(make_fwd(use_a8), mlp)
        for p in PROJ:
            lin = find_module(model, f"{L}.mlp.{p}")
            base = tensor_name(L, f"mlp.{p}")
            if kind == "bf16":
                w = load_bf16_weight(src, smap, base)
            elif kind == "w8a8":
                w = qdq_w8_channel(load_bf16_weight(src, smap, base))
            elif kind == "w4_awq_asym":
                w = dequant_w4(w4root, w4map, base)
            set_param(lin, "weight", w)
            del w
        if torch.cuda.is_available() and L % 8 == 7:
            torch.cuda.empty_cache()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    patch_transformers_broken_torchvision()
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    cfg = text_config(json.loads((args.source / "config.json").read_text()))
    num_layers = int(cfg["num_hidden_layers"])
    tok = AutoTokenizer.from_pretrained(args.source, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    prompts = []
    for line in args.prompts.read_text().splitlines():
        if line.strip():
            prompts.append(json.loads(line)["text"])
        if len(prompts) >= args.max_prompts:
            break
    print(f"[kl] {len(prompts)} held-out prompts", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(
        args.source, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    smap = load_weight_map(args.source)
    w4map = load_weight_map(args.w4_asym)
    dev = next(model.parameters()).device

    # tokenize + choose sampled valid positions per prompt (deterministic)
    enc = [
        tok(p, return_tensors="pt", truncation=True, max_length=args.max_seq_length, add_special_tokens=False)
        for p in prompts
    ]
    pos_idx = []
    g = torch.Generator().manual_seed(args.seed)
    for e in enc:
        n = e["input_ids"].shape[1]
        k = min(args.pos_per_prompt, n - 1)
        sel = (
            torch.randperm(n - 1, generator=g)[:k].sort().values
        )  # exclude last (no next-token target needed; we use logits at pos)
        pos_idx.append(sel)

    def run_variant(kind):
        patch(model, num_layers, kind, args.source, smap, args.w4_asym, w4map)
        gc.collect()
        torch.cuda.empty_cache()
        out = []
        for e, sel in zip(enc, pos_idx, strict=False):
            ids = e["input_ids"].to(dev)
            with torch.no_grad():
                lg = model(input_ids=ids, use_cache=False, return_dict=True).logits[0]  # (seq, V)
            out.append(lg[sel].float().cpu())
            del lg
            torch.cuda.empty_cache()
        return out

    print("[kl] variant bf16", flush=True)
    base = run_variant("bf16")
    base_lp = [torch.log_softmax(x, -1) for x in base]

    results = {}
    for kind in ("w8a8", "w4_awq_asym"):
        print(f"[kl] variant {kind}", flush=True)
        cur = run_variant(kind)
        l2s, kls, top1, top50 = [], [], [], []
        for b, blp, c in zip(base, base_lp, cur, strict=False):
            l2s.append(torch.linalg.vector_norm(c - b, dim=-1))  # per-pos logit L2
            clp = torch.log_softmax(c, -1)
            bp = blp.exp()
            kls.append((bp * (blp - clp)).sum(-1))  # KL(bf16||q) per pos
            top1.append((b.argmax(-1) == c.argmax(-1)).float())
            kk = 50
            bt = b.topk(kk, -1).indices
            ct = c.topk(kk, -1).indices
            ov = torch.stack([torch.isin(bt[i], ct[i]).float().mean() for i in range(bt.shape[0])])
            top50.append(ov)
        L2 = torch.cat(l2s)
        KL = torch.cat(kls)
        T1 = torch.cat(top1)
        T50 = torch.cat(top50)
        results[kind] = {
            "n_positions": int(L2.numel()),
            "final_logits_L2_mean": float(L2.mean()),
            "final_logits_L2_median": float(L2.median()),
            "final_logits_KL_mean": float(KL.mean()),
            "final_logits_KL_median": float(KL.median()),
            "top1_agree": float(T1.mean()),
            "top50_overlap": float(T50.mean()),
        }
        del cur, l2s, kls
        gc.collect()
        torch.cuda.empty_cache()

    w8, w4 = results["w8a8"], results["w4_awq_asym"]
    summary = {
        "mlp_output_L2": {"w8a8": 38.32, "w4_awq_asym": 39.86, "ratio_w4_over_w8": 39.86 / 38.32},
        "final_logits_L2_ratio_w4_over_w8": w4["final_logits_L2_mean"] / w8["final_logits_L2_mean"],
        "final_logits_KL_ratio_w4_over_w8": w4["final_logits_KL_mean"] / w8["final_logits_KL_mean"],
    }
    print("\n=== RESULTS (same held-out inputs as the 39.86/38.32 table) ===")
    print(f"{'metric':28s} {'w8a8':>12s} {'w4_asym':>12s} {'w4/w8':>8s}")
    print(f"{'MLP-output L2 (prior)':28s} {38.32:12.3f} {39.86:12.3f} {39.86/38.32:8.3f}")
    print(
        f"{'final-logits L2 mean':28s} {w8['final_logits_L2_mean']:12.3f} {w4['final_logits_L2_mean']:12.3f} {summary['final_logits_L2_ratio_w4_over_w8']:8.3f}"
    )
    print(
        f"{'final-logits KL mean':28s} {w8['final_logits_KL_mean']:12.4f} {w4['final_logits_KL_mean']:12.4f} {summary['final_logits_KL_ratio_w4_over_w8']:8.3f}"
    )
    print(f"{'top1 agree':28s} {w8['top1_agree']:12.3f} {w4['top1_agree']:12.3f}")
    print(f"{'top50 overlap':28s} {w8['top50_overlap']:12.3f} {w4['top50_overlap']:12.3f}")

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {"results": results, "summary": summary, "n_prompts": len(prompts), "pos_per_prompt": args.pos_per_prompt},
            indent=2,
        )
        + "\n"
    )
    print("wrote", args.output_json)


if __name__ == "__main__":
    main()
