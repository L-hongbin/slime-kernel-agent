"""Compare hidden_states / logits across variants saved by rotation_probe.py."""

import argparse
import torch


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, help="reference .pt (e.g. raw original)")
    ap.add_argument("--variant", action="append", required=True, help="other .pt files to compare")
    return ap.parse_args()


def rel_l2(a, b):
    diff = (a - b).float()
    base = a.float()
    return (diff.norm() / (base.norm() + 1e-8)).item()


def main():
    args = parse_args()
    print(f"=== loading baseline {args.baseline} ===")
    base = torch.load(args.baseline, weights_only=False, map_location="cpu")
    base_hs = base["hidden_states"]
    base_logits = base["logits"]
    num_layers = base_hs.shape[0]
    print(f"baseline shape: hs {tuple(base_hs.shape)}, logits {tuple(base_logits.shape)}")
    print(f"baseline variant={base['variant']} model={base['model_path']}")
    print()

    for vpath in args.variant:
        v = torch.load(vpath, weights_only=False, map_location="cpu")
        v_hs = v["hidden_states"]
        v_logits = v["logits"]
        print(f"=== {vpath} (variant={v['variant']}) ===")
        # Per-layer relative L2
        per_layer = []
        for L in range(num_layers):
            r = rel_l2(v_hs[L], base_hs[L])
            per_layer.append(r)
        # Print summary: embedding, every 8 layers, last
        print("per-layer rel L2 divergence (baseline vs this variant):")
        print(f"  embedding (L0): {per_layer[0]*100:.3f}%")
        for L in range(8, num_layers, 8):
            print(f"  after layer {L:2d}: {per_layer[L]*100:.3f}%")
        print(f"  final (L{num_layers-1}): {per_layer[-1]*100:.3f}%")
        # Logit divergence
        logit_rl2 = rel_l2(v_logits, base_logits)
        # KL on last token
        last_baseline = torch.softmax(base_logits[:, -1, :], dim=-1)
        last_v = torch.log_softmax(v_logits[:, -1, :], dim=-1)
        kl = (last_baseline * (torch.log(last_baseline + 1e-12) - last_v)).sum(-1).mean().item()
        # Top-1 agreement on last token
        top_b = base_logits[:, -1, :].argmax(-1)
        top_v = v_logits[:, -1, :].argmax(-1)
        top1_agree = (top_b == top_v).float().mean().item()
        print(f"logit rel L2: {logit_rl2*100:.3f}%, last-token KL: {kl:.5f}, top1 agree: {top1_agree*100:.1f}%")
        print()


if __name__ == "__main__":
    main()
