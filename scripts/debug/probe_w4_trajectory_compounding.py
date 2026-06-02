#!/usr/bin/env python3
"""Validate (or falsify) the autoregressive-compounding explanation.

All prior metrics are teacher-forced on BF16's own (good) tokens and are per-token
averages; they show only ~1.5x W4/W8-A8 and cannot capture trajectory drift. The
explanation says W4's damage compounds when W4 conditions on its OWN drifting prefix.

Falsifiable test (matched prompts, same model pair bf16-vs-w4, only the conditioning
trajectory differs):
  * w4_self  : condition on W4's OWN eval response, measure KL(bf16||w4) per position
  * bf16_self: condition on BF16's OWN eval response, measure KL(bf16||w4) per position
Prediction if the explanation holds: w4_self KL >> bf16_self KL, and grows with
response position. If w4_self ~= bf16_self, the compounding story is unsupported.

Both trajectories use the SAME prompt (so the only difference is the self-generated
continuation). Forwards only, no generation. KL is top-1k (==full for forward KL).
"""
from __future__ import annotations

import argparse
import gc
import json
import statistics as st
import sys
from pathlib import Path

import torch

REPO = Path("/nfs/FM/chenshuailin/projects/kernel_agents/slime")
sys.path.insert(0, str(REPO))
from scripts.debug.probe_w4_metric_stage_kl import patch
from scripts.quantize.utils.compare_mlp_quant_loss import load_weight_map, text_config

SRC = Path("/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B")
W4 = REPO / "checkpoints/quantized/AWQ/Qwen3.6-27B-AWQ-W4A16-asym-mlp"
W4_EVAL = REPO / (
    "checkpoints/Qwen3.6-27B-AWQ-W4A16-asym-mlp/"
    "20260531_233411_awq.w4a16.asym_mlp.sglzpfix.100x8.eagle.rm39_ctx65536_n8_summ1600/dumps/rollout_data/eval_0.pt"
)
BF16_EVAL = REPO / (
    "checkpoints/Qwen3.6-27B/"
    "20260529_132454_newSlimeKG.tp4.eagle.rm16_ctx65536_n8_summ1600/dumps/rollout_data/eval_0.pt"
)
BUCKETS = [(0, 64), (64, 256), (256, 1024), (1024, 3072)]


def first_by_problem(payload):
    out = {}
    for s in payload["samples"]:
        meta = s.get("metadata") or {}
        turns = meta.get("turns") or []
        if not turns:
            continue
        pid = meta.get("problem_id") or meta.get("name")
        if pid is None or pid in out:
            continue
        resp = turns[0].get("response") or ""
        prompt = turns[0].get("prompt_snapshot") or s.get("prompt") or ""
        if len(resp) < 200:
            continue
        out[pid] = {"prompt": prompt, "response": resp}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-samples", type=int, default=10)
    ap.add_argument("--ctx-prompt", type=int, default=1024)
    ap.add_argument("--max-resp", type=int, default=3072)
    ap.add_argument("--per-bucket", type=int, default=64)
    ap.add_argument("--topk-kl", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260602)
    ap.add_argument(
        "--output-json",
        type=Path,
        default=REPO / "checkpoints/quantized/analysis/w4_asym_zp_sensitivity/trajectory_compounding.json",
    )
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    from scripts.quantize.producers.awq_w4a16 import patch_transformers_broken_torchvision

    patch_transformers_broken_torchvision()
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    cfg = text_config(json.loads((SRC / "config.json").read_text()))
    num_layers = int(cfg["num_hidden_layers"])
    tok = AutoTokenizer.from_pretrained(SRC, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    w4d = first_by_problem(torch.load(W4_EVAL, map_location="cpu", weights_only=False))
    bfd = first_by_problem(torch.load(BF16_EVAL, map_location="cpu", weights_only=False))
    common = [p for p in w4d if p in bfd][: args.max_samples]
    print(f"[traj] {len(common)} matched problems", flush=True)

    g = torch.Generator().manual_seed(args.seed)
    items = []
    for pid in common:
        p_ids = tok.encode(w4d[pid]["prompt"], add_special_tokens=False)[-args.ctx_prompt :]
        for kind, resp in [("w4_self", w4d[pid]["response"]), ("bf16_self", bfd[pid]["response"])]:
            r_ids = tok.encode(resp, add_special_tokens=False)[: args.max_resp]
            if len(r_ids) < 64:
                continue
            ids = p_ids + r_ids
            rs = len(p_ids)
            sel = []
            for lo, hi in BUCKETS:
                cand = [rs + i for i in range(lo, min(hi, len(r_ids) - 1))]
                if not cand:
                    continue
                if len(cand) > args.per_bucket:
                    pick = torch.randperm(len(cand), generator=g)[: args.per_bucket].tolist()
                    cand = [cand[j] for j in sorted(pick)]
                sel.extend([(p, f"{lo}-{hi}") for p in cand])
            if sel:
                items.append({"pid": pid, "kind": kind, "ids": torch.tensor([ids]), "sel": sel})

    model = AutoModelForImageTextToText.from_pretrained(
        SRC, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    ).eval()
    smap = load_weight_map(SRC)
    w4map = load_weight_map(W4)
    dev = next(model.parameters()).device

    def fwd_logits(it):
        ids = it["ids"].to(dev)
        with torch.no_grad():
            lg = model(input_ids=ids, use_cache=False, return_dict=True).logits[0]
        pos = torch.tensor([p for p, _ in it["sel"]])
        out = lg[pos].float().cpu()
        del lg
        torch.cuda.empty_cache()
        return out

    print("[traj] phase A: BF16", flush=True)
    patch(model, num_layers, "bf16", SRC, smap, W4, w4map)
    gc.collect()
    torch.cuda.empty_cache()
    bf_store = [fwd_logits(it) for it in items]

    print("[traj] phase B: W4", flush=True)
    patch(model, num_layers, "w4_awq_asym", SRC, smap, W4, w4map)
    gc.collect()
    torch.cuda.empty_cache()
    K = args.topk_kl
    rows = []
    for it, bf in zip(items, bf_store, strict=False):
        w4lg = fwd_logits(it)
        b = bf.to(dev)
        c = w4lg.to(dev)
        blp = torch.log_softmax(b, -1)
        clp = torch.log_softmax(c, -1)
        bp = blp.exp()
        idx = b.topk(K, -1).indices
        pp = torch.gather(bp, -1, idx)
        pp = pp / pp.sum(-1, keepdim=True)
        qq = torch.gather(clp.exp(), -1, idx)
        qq = qq / qq.sum(-1, keepdim=True)
        kl = (pp * (pp.clamp_min(1e-12).log() - qq.clamp_min(1e-12).log())).sum(-1)
        agree = (b.argmax(-1) == c.argmax(-1)).float()
        for j, (_p, bkt) in enumerate(it["sel"]):
            rows.append({"kind": it["kind"], "bucket": bkt, "kl": float(kl[j]), "agree": float(agree[j])})
        del b, c, blp, clp, bp
        torch.cuda.empty_cache()

    def agg(kind, bucket=None):
        v = [r for r in rows if r["kind"] == kind and (bucket is None or r["bucket"] == bucket)]
        if not v:
            return None
        return {
            "n": len(v),
            "kl_mean": st.mean(r["kl"] for r in v),
            "kl_median": st.median(r["kl"] for r in v),
            "agree": st.mean(r["agree"] for r in v),
        }

    buckets = [f"{lo}-{hi}" for lo, hi in BUCKETS]
    out = {"by_bucket": {}, "overall": {}}
    print("\n=== KL_top1k(bf16||w4) by response-position bucket ===")
    print(
        f"{'bucket':10s} {'W4-self KL':>11s} {'BF16-self KL':>13s} {'ratio':>6s} {'W4 agree':>9s} {'BF16 agree':>11s}"
    )
    for bkt in buckets:
        a = agg("w4_self", bkt)
        b = agg("bf16_self", bkt)
        if a and b:
            print(
                f"{bkt:10s} {a['kl_mean']:11.4f} {b['kl_mean']:13.4f} {a['kl_mean']/b['kl_mean']:6.2f} {a['agree']:9.3f} {b['agree']:11.3f}"
            )
            out["by_bucket"][bkt] = {"w4_self": a, "bf16_self": b, "ratio": a["kl_mean"] / b["kl_mean"]}
    ao = agg("w4_self")
    bo = agg("bf16_self")
    print(
        f"\nOVERALL  W4-self KL={ao['kl_mean']:.4f} (agree {ao['agree']:.3f})  "
        f"BF16-self KL={bo['kl_mean']:.4f} (agree {bo['agree']:.3f})  ratio={ao['kl_mean']/bo['kl_mean']:.2f}"
    )
    print("Reference: codedomain BF16-trajectory top1k KL mean = 0.0116 (w4 vs bf16)")
    out["overall"] = {
        "w4_self": ao,
        "bf16_self": bo,
        "ratio": ao["kl_mean"] / bo["kl_mean"],
        "n_problems": len(common),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", args.output_json)


if __name__ == "__main__":
    main()
