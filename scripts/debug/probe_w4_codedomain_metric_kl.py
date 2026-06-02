#!/usr/bin/env python3
"""Code-domain version of the metric/stage probe: re-measure W4 vs W8-A8 quant loss
on real DrKernel BF16 rollouts, with final-logits L2, full KL, and top-1k KL.

Motivation (handoff_w4a16_awq.md): on generic UltraChat the best teacher-forced
metric (final-logits KL) gave only W4/W8=1.36x, far below the DrKernel T1 eval gap
(~1.7-1.8x). Hypotheses for the gap: (a) domain (code tokens), (b) KL diluted by
the 248k-vocab tail. This probe tests both:
  * inputs = BF16's OWN generated kernel code (teacher-forced), positions sampled
    from the RESPONSE (generated-code) region only.
  * metrics: final-logits L2, full KL(bf16||q), and top-1k KL (renormalized over
    BF16's top-1000 tokens) -- the top-1k KL removes the long tail.

variants mirror the L2/UltraChat table exactly:
  bf16        : source MLP
  w8a8        : source MLP weights per-output-channel symmetric INT8 RTN + per-token A8
  w4_awq_asym : AWQ asym dequant MLP weights, A16
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

from scripts.debug.probe_w4_metric_stage_kl import patch  # reuse exact variant patching
from scripts.quantize.utils.compare_mlp_quant_loss import load_weight_map, text_config

DEFAULT_SOURCE = Path("/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B")
DEFAULT_W4 = REPO / "checkpoints/quantized/AWQ/Qwen3.6-27B-AWQ-W4A16-asym-mlp"
DEFAULT_BF16_EVAL = REPO / (
    "checkpoints/Qwen3.6-27B/"
    "20260529_132454_newSlimeKG.tp4.eagle.rm16_ctx65536_n8_summ1600/dumps/rollout_data/eval_0.pt"
)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--w4-asym", type=Path, default=DEFAULT_W4)
    ap.add_argument("--bf16-eval", type=Path, default=DEFAULT_BF16_EVAL)
    ap.add_argument("--max-prompts", type=int, default=24)
    ap.add_argument("--ctx-prompt-tokens", type=int, default=512)
    ap.add_argument("--max-resp-tokens", type=int, default=3584)
    ap.add_argument("--pos-per-prompt", type=int, default=160)
    ap.add_argument("--topk-kl", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260602)
    ap.add_argument(
        "--output-json",
        type=Path,
        default=REPO / "checkpoints/quantized/analysis/w4_asym_zp_sensitivity/codedomain_metric_kl.json",
    )
    return ap.parse_args()


def build_code_sequences(eval_path, tok, max_prompts, ctx_prompt, max_resp, pos_per_prompt, seed):
    payload = torch.load(eval_path, map_location="cpu", weights_only=False)
    g = torch.Generator().manual_seed(seed)
    seqs = []
    for sample in payload["samples"]:
        meta = sample.get("metadata") or {}
        turns = meta.get("turns") or []
        if not turns:
            continue
        turn = turns[0]
        resp = turn.get("response") or ""
        prompt = turn.get("prompt_snapshot") or sample.get("prompt") or ""
        if len(resp) < 200:
            continue
        p_ids = tok.encode(prompt, add_special_tokens=False)[-ctx_prompt:]
        r_ids = tok.encode(resp, add_special_tokens=False)[:max_resp]
        if len(r_ids) < 64:
            continue
        ids = p_ids + r_ids
        resp_start = len(p_ids)
        # sample positions in the response region [resp_start, len-1)
        avail = len(ids) - 1 - resp_start
        k = min(pos_per_prompt, avail)
        sel = (resp_start + torch.randperm(avail, generator=g)[:k]).sort().values
        seqs.append(
            {
                "ids": torch.tensor([ids], dtype=torch.long),
                "pos": sel,
                "idx": int(sample["index"]),
                "n_resp": len(r_ids),
            }
        )
        if len(seqs) >= max_prompts:
            break
    if not seqs:
        raise RuntimeError("no usable code sequences")
    return seqs


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    from scripts.quantize.producers.awq_w4a16 import patch_transformers_broken_torchvision

    patch_transformers_broken_torchvision()
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    cfg = text_config(json.loads((args.source / "config.json").read_text()))
    num_layers = int(cfg["num_hidden_layers"])
    tok = AutoTokenizer.from_pretrained(args.source, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    seqs = build_code_sequences(
        args.bf16_eval,
        tok,
        args.max_prompts,
        args.ctx_prompt_tokens,
        args.max_resp_tokens,
        args.pos_per_prompt,
        args.seed,
    )
    npos = sum(len(s["pos"]) for s in seqs)
    print(f"[code-kl] {len(seqs)} BF16 code rollouts, {npos} response positions", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(
        args.source, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    smap = load_weight_map(args.source)
    w4map = load_weight_map(args.w4_asym)
    dev = next(model.parameters()).device

    def run_variant(kind):
        patch(model, num_layers, kind, args.source, smap, args.w4_asym, w4map)
        gc.collect()
        torch.cuda.empty_cache()
        out = []
        for s in seqs:
            ids = s["ids"].to(dev)
            with torch.no_grad():
                lg = model(input_ids=ids, use_cache=False, return_dict=True).logits[0]
            out.append(lg[s["pos"].to(dev)].float().cpu())
            del lg
            torch.cuda.empty_cache()
        return out

    print("[code-kl] variant bf16", flush=True)
    base = run_variant("bf16")

    K = args.topk_kl
    results = {}
    for kind in ("w8a8", "w4_awq_asym"):
        print(f"[code-kl] variant {kind}", flush=True)
        cur = run_variant(kind)
        L2, KLf, KLt, T1, T50 = [], [], [], [], []
        PB, PQ, DROP, LDROP = [], [], [], []  # bf16 top1 prob, quant prob on bf16 top1, prob drop, logprob drop
        for b_cpu, c_cpu in zip(base, cur, strict=False):
            b = b_cpu.to(dev)
            c = c_cpu.to(dev)
            L2.append(torch.linalg.vector_norm(c - b, dim=-1).cpu())
            blp = torch.log_softmax(b, -1)
            clp = torch.log_softmax(c, -1)
            bp = blp.exp()
            KLf.append((bp * (blp - clp)).sum(-1).cpu())  # full KL(bf16||q)
            # top-1k KL: restrict to bf16 top-K indices, renormalize both, KL
            idx = b.topk(K, -1).indices
            pp = torch.gather(bp, -1, idx)
            pp = pp / pp.sum(-1, keepdim=True)
            cp = clp.exp()
            qq = torch.gather(cp, -1, idx)
            qq = qq / qq.sum(-1, keepdim=True)
            KLt.append((pp * (pp.clamp_min(1e-12).log() - qq.clamp_min(1e-12).log())).sum(-1).cpu())
            # top-1 token (BF16 argmax): prob BF16 vs quant put on it
            i1 = b.argmax(-1, keepdim=True)
            pb = bp.gather(-1, i1).squeeze(-1)
            pq = cp.gather(-1, i1).squeeze(-1)
            PB.append(pb.cpu())
            PQ.append(pq.cpu())
            DROP.append((pb - pq).cpu())
            LDROP.append((blp.gather(-1, i1).squeeze(-1) - clp.gather(-1, i1).squeeze(-1)).cpu())
            T1.append((b.argmax(-1) == c.argmax(-1)).float().cpu())
            bt = b.topk(50, -1).indices
            ct = c.topk(50, -1).indices
            T50.append(torch.stack([torch.isin(bt[i], ct[i]).float().mean() for i in range(bt.shape[0])]).cpu())
            del b, c, blp, clp, bp, cp
            torch.cuda.empty_cache()
        L2 = torch.cat(L2)
        KLf = torch.cat(KLf)
        KLt = torch.cat(KLt)
        T1 = torch.cat(T1)
        T50 = torch.cat(T50)
        PB = torch.cat(PB)
        PQ = torch.cat(PQ)
        DROP = torch.cat(DROP)
        LDROP = torch.cat(LDROP)

        def summ(t):
            return {
                "mean": float(t.mean()),
                "median": float(t.median()),
                "p99": float(torch.quantile(t, 0.99)),
                "p999": float(torch.quantile(t, 0.999)),
                "max": float(t.max()),
            }

        results[kind] = {
            "n_positions": int(L2.numel()),
            "final_logits_L2": summ(L2),
            "full_KL": summ(KLf),
            "top1k_KL": summ(KLt),
            "top1_agree": float(T1.mean()),
            "top50_overlap": float(T50.mean()),
            "top1_bf16_prob_mean": float(PB.mean()),
            "top1_quant_prob_mean": float(PQ.mean()),
            "top1_prob_retained": float(PQ.mean() / PB.mean()),
            "top1_prob_drop": summ(DROP),  # p_bf16(top1) - p_quant(top1), per position
            "top1_logprob_drop": summ(LDROP),  # logp_bf16(top1) - logp_quant(top1)
        }
        del cur
        gc.collect()
        torch.cuda.empty_cache()

    w8, w4 = results["w8a8"], results["w4_awq_asym"]
    AGGS = ["mean", "median", "p99", "p999", "max"]
    summary = {
        met: {a: (w4[met][a] / w8[met][a] if w8[met][a] else float("nan")) for a in AGGS}
        for met in ("final_logits_L2", "full_KL", "top1k_KL")
    }
    print("\n=== CODE-DOMAIN (BF16 DrKernel rollouts, response positions): W4/W8-A8 ratio by aggregate ===")
    print(f"{'metric':16s} " + " ".join(f"{a:>9s}" for a in AGGS))
    for met in ("final_logits_L2", "full_KL", "top1k_KL"):
        print(f"{met:16s} " + " ".join(f"{summary[met][a]:9.3f}" for a in AGGS))
    print("\n--- raw top1k_KL (the eval-relevant metric) ---")
    print(f"{'agg':10s} {'w8a8':>12s} {'w4_asym':>12s} {'w4/w8':>8s}")
    for a in AGGS:
        print(f"{a:10s} {w8['top1k_KL'][a]:12.5f} {w4['top1k_KL'][a]:12.5f} {summary['top1k_KL'][a]:8.3f}")
    print(
        f"\ntop1 agree w8={w8['top1_agree']:.3f} w4={w4['top1_agree']:.3f} | top50 overlap w8={w8['top50_overlap']:.3f} w4={w4['top50_overlap']:.3f}"
    )
    print("\n=== TOP-1 TOKEN PROBABILITY (on the BF16 argmax token) ===")
    print(f"BF16 prob on its own top1 (mean):           {w8['top1_bf16_prob_mean']:.4f}")
    print(
        f"prob kept on BF16 top1   w8a8={w8['top1_quant_prob_mean']:.4f}  w4={w4['top1_quant_prob_mean']:.4f}  (retained: w8={w8['top1_prob_retained']:.3f} w4={w4['top1_prob_retained']:.3f})"
    )
    print("top1 prob DROP (p_bf16 - p_quant) by aggregate, and W4/W8 ratio:")
    print(f"{'agg':10s} {'w8a8':>10s} {'w4_asym':>10s} {'w4/w8':>8s}")
    for a in AGGS:
        r = w4["top1_prob_drop"][a] / w8["top1_prob_drop"][a] if w8["top1_prob_drop"][a] else float("nan")
        print(f"{a:10s} {w8['top1_prob_drop'][a]:10.5f} {w4['top1_prob_drop'][a]:10.5f} {r:8.3f}")
    print(
        f"top1 logprob drop (nats) mean: w8={w8['top1_logprob_drop']['mean']:.4f}  w4={w4['top1_logprob_drop']['mean']:.4f}  ratio={w4['top1_logprob_drop']['mean']/w8['top1_logprob_drop']['mean']:.3f}"
    )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(
            {
                "results": results,
                "summary": summary,
                "n_prompts": len(seqs),
                "pos_per_prompt": args.pos_per_prompt,
                "topk_kl": K,
                "bf16_eval": str(args.bf16_eval),
            },
            indent=2,
        )
        + "\n"
    )
    print("wrote", args.output_json)


if __name__ == "__main__":
    main()
