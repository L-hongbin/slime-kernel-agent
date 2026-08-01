#!/usr/bin/env python3
"""Quantify the "effectively REINFORCE today" finding for the keep-old-actor handoff.

BEFORE keep-old-actor (this config): the loss sets old_log_probs = the SAME current
actor recompute as the numerator, so ppo_kl ≡ 0 -> ratio = exp(-ppo_kl) = 1.0 for
every token and pg_clipfrac ≡ 0 (proven in slime/utils/ppo_utils.py compute_policy_loss).
No importance weighting, no clipping = vanilla policy gradient.

AFTER keep-old-actor: old_log_probs = behavioral θ_V, numerator = current θ_{V+1}, so
the token ratio is exp(logp_θV − logp_θ{V+1}) = a real one-gradient-step drift. This
script MEASURES that distribution on the real truncated-4 V4 model: score θ, take ONE
gradient step on the LoRA adapter (AdamW at the production LR, a proxy for Muon's update
magnitude), score θ' under the SAME (recorded) MoE routing — matching production where
the old and current forwards both replay the sglang routing — and report the ratio
distribution + PPO clip fraction at eps_clip=[0.2, 0.28].

Illustrative (truncated-4, AdamW proxy, one step, random whitened advantages); the real
per-step distribution is visible via the existing ppo_kl / pg_clipfrac / tis metrics once
enabled. Run on ONE GPU.
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO = "/nfs/FM/chenshuailin/projects/kernel_agents/slime-v4flash-lora"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="/nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8")
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--fp8", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lrs", type=str, default="1e-4,3e-4")
    ap.add_argument("--adapter-scale", type=float, default=0.02)
    ap.add_argument("--master-port", type=int, default=29713)
    return ap.parse_args()


def main():
    args = parse_args()
    os.environ["V4_LORA_DIM"] = "16"
    os.environ["V4_LORA_ALPHA"] = "32"
    os.environ["V4_LORA_DROPOUT"] = "0.0"
    os.environ["V4_LORA_SHARED_EXPERT"] = "1"
    os.environ["V4_LORA_ADAPTER_ONLY_CKPT"] = "1"
    if args.fp8:
        os.environ["V4_FP8_FROZEN_EXPERTS"] = "1"
        os.environ["V4_FP8_SHARED_EXPERT"] = "1"
        os.environ["V4_FP8_ATTENTION"] = "1"

    sys.path.insert(0, REPO)
    from custom_kernels.deepseek_v4.megatron.lora import apply_v4_lora
    from custom_kernels.deepseek_v4.megatron.model_provider import build_v4_mcore_model
    from custom_kernels.deepseek_v4.megatron.native_checkpoint import load_native_checkpoint_into_mcore_model
    from megatron.core import parallel_state, tensor_parallel
    from transformers import AutoConfig

    from slime.backends.megatron_utils.lora_old_actor import enumerate_adapter_params
    from slime.utils.routing_replay import RoutingReplay

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(args.master_port))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.cuda.set_device(0)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0, device_id=torch.device("cuda:0"))
    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(1, 1, expert_model_parallel_size=1)
    tensor_parallel.model_parallel_cuda_manual_seed(0)
    device = torch.device("cuda")

    cfg = AutoConfig.from_pretrained(args.ckpt, trust_remote_code=True)
    N = args.num_layers
    cfg.layer_types = list(cfg.layer_types)[:N]
    cfg.mlp_layer_types = list(cfg.mlp_layer_types)[:N]
    cfg.num_hidden_layers = N
    cfg._attn_implementation = "eager"

    mc = build_v4_mcore_model(
        cfg,
        params_dtype=torch.bfloat16,
        init_weights=False,
        expert_model_parallel_size=1,
        expert_model_parallel_rank=0,
    )
    mc = mc.bfloat16().cuda()
    load_native_checkpoint_into_mcore_model(mc, args.ckpt, layer_map={i: i for i in range(N)}, strict=True)
    mc = apply_v4_lora(mc, dim=16, alpha=32, dropout=0.0)
    if args.fp8:
        from custom_kernels.deepseek_v4.megatron.mcore_model import quantize_lora_adapter_fp8

        for module in mc.modules():
            if hasattr(module, "quantize_frozen_experts_fp8"):
                module.quantize_frozen_experts_fp8()
            elif hasattr(module, "quantize_fp8"):
                module.quantize_fp8()
        _attn_targets = ("q_a_proj", "q_b_proj", "kv_proj", "o_b_proj")
        for name, module in mc.named_modules():
            if (
                any(name.endswith("self_attn." + t) for t in _attn_targets)
                and hasattr(module, "linear_in")
                and hasattr(module, "weight")
            ):
                from custom_kernels.deepseek_v4.megatron.mcore_model import fp8_weight_recipe_from_config

                _blk, _w_ue8m0 = fp8_weight_recipe_from_config(cfg)
                quantize_lora_adapter_fp8(module, _blk, use_ue8m0=_w_ue8m0)

    model = [mc]
    adapter_params = [p for _, p in enumerate_adapter_params(model)]
    routers = list(RoutingReplay.all_routing_replays)
    print(f"[ratio] adapter params={len(adapter_params)} routing_replay routers={len(routers)}")

    torch.manual_seed(0)
    vocab = int(cfg.vocab_size)
    T = args.seq_len
    input_ids = torch.randint(0, vocab, (1, T), device=device, dtype=torch.long)
    plen = T // 2
    resp = torch.arange(plen, T, device=device)
    tgt = input_ids[0, resp]

    def set_adapter(seed, scale):
        g = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for _, p in enumerate_adapter_params(model):
                p.copy_((torch.randn(list(p.shape), generator=g) * scale).to(p.dtype).to(p.device))

    def resp_logp(grad: bool):
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            logits = mc(input_ids)
            logp = torch.log_softmax(logits[0].float(), dim=-1)
            return logp[resp - 1, tgt]

    def report(ratio, lr):
        r = ratio.detach().float().cpu().numpy()
        lr_ratio = np.log(r)
        for eps_lo, eps_hi in [(0.2, 0.28)]:
            clip = np.mean((r < 1 - eps_lo) | (r > 1 + eps_hi))
        pct = np.percentile(r, [1, 50, 99])
        print(
            f"[ratio] lr={lr:g}: ratio mean={r.mean():.4f} std={r.std():.4f} "
            f"p1/p50/p99={pct[0]:.3f}/{pct[1]:.3f}/{pct[2]:.3f} "
            f"min={r.min():.3f} max={r.max():.3f} | mean|log r|={np.abs(lr_ratio).mean():.4f} "
            f"| pg_clipfrac@[0.8,1.28]={clip:.3f}"
        )

    print(
        "[ratio] BEFORE keep-old-actor: ppo_kl == 0 by construction -> ratio == 1.0000 for every "
        "token, pg_clipfrac == 0.000 (vanilla policy gradient / REINFORCE)."
    )

    for lr in [float(x) for x in args.lrs.split(",")]:
        set_adapter(seed=1234, scale=args.adapter_scale)  # a mid-training-like θ_V
        # old forward: record routing (θ_V), get behavioral logp + REINFORCE loss
        RoutingReplay.clear_all()
        os.environ["ROUTING_REPLAY_STAGE"] = "record" if routers else "fallthrough"
        logp_old_t = resp_logp(grad=True)
        logp_old = logp_old_t.detach()
        adv = torch.randn(resp.numel(), device=device)  # whitened advantages (std 1)
        loss = -(adv * logp_old_t).mean()
        opt = torch.optim.AdamW(adapter_params, lr=lr)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()  # θ_V -> θ_{V+1}
        # new forward under the SAME recorded routing (production replays sglang routing for both)
        os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward" if routers else "fallthrough"
        logp_new = resp_logp(grad=False)
        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
        RoutingReplay.clear_all()
        # Production direction (ppo_utils.compute_policy_loss): ratio =
        # exp(current − old) = π_new/π_old. The clip band [1-eps_low, 1+eps_high]
        # is asymmetric, so the direction matters for clip fractions.
        ratio = torch.exp(logp_new - logp_old)  # the PPO ratio AFTER keep-old-actor
        report(ratio, lr)

    print("[ratio] DONE")


if __name__ == "__main__":
    main()
