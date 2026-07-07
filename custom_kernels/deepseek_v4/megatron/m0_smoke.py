"""M0 smoke: build a tiny V4-Flash mcore-scaffold model and run a no-NaN forward.

Gate (from the M0 handoff):
  * MUST keep hidden_size=4096, head_dim=512, hc_mult=4 (A1 fixes hd=512; B1 mHC
    hard-codes H=4 / HIDDEN=4096 / SINKHORN=20).
  * shrink num_hidden_layers, n_routed_experts, num_attention_heads, seq_len.
  * cover >=1 of each attention layer_type (sliding / CSA / HCA) + both routers
    (hash_moe + moe).
  * random input_ids on GPU 0; assert forward completes, shape correct, no NaN/Inf.

Run:  CUDA_VISIBLE_DEVICES=0 python -m custom_kernels.deepseek_v4.megatron.m0_smoke
  or  cd custom_kernels/deepseek_v4 && CUDA_VISIBLE_DEVICES=0 python megatron/m0_smoke.py
"""

import os
import sys

import torch

# allow `python megatron/m0_smoke.py` from custom_kernels/deepseek_v4/ as well as `-m`.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from megatron.decoder import V4Model  # type: ignore
else:
    from .decoder import V4Model

from transformers.models.deepseek_v4 import DeepseekV4Config
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Experts,
    DeepseekV4HashRouter,
    DeepseekV4HyperHead,
    DeepseekV4Indexer,
    DeepseekV4TopKRouter,
)


def tiny_config() -> DeepseekV4Config:
    return DeepseekV4Config(
        # --- FIXED (structural to the kernels) ---
        hidden_size=4096,
        head_dim=512,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        q_lora_rank=1024,
        o_groups=8,
        o_lora_rank=1024,
        sliding_window=128,
        # --- shrunk ---
        vocab_size=512,
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=1,
        moe_intermediate_size=256,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        index_n_heads=4,
        index_head_dim=64,
        index_topk=512,
        # one of each attention type + both routers (hash then moe).
        layer_types=[
            "sliding_attention",
            "compressed_sparse_attention",
            "heavily_compressed_attention",
        ],
        mlp_layer_types=["hash_moe", "hash_moe", "moe"],
    )


@torch.no_grad()
def init_weights(model, cfg, dev):
    """Sane random init for M0.

    Our V4* modules already init their own params (Linears default-init; norms=1;
    sinks/position_bias=0; mHC fn~N(0,std)/base=0/scale=1).  But the HF-owned
    submodules we reuse (routers, experts, indexer, hc_head) are built with
    ``torch.empty`` and only get sane values from HF's ``_init_weights`` — which we
    don't run here.  So initialise exactly those, mirroring HF:1202-1234.
    """
    std = cfg.initializer_range
    for m in model.modules():
        if isinstance(m, (DeepseekV4TopKRouter, DeepseekV4HashRouter)):
            torch.nn.init.normal_(m.weight, mean=0.0, std=std)
            if isinstance(m, DeepseekV4TopKRouter):
                torch.nn.init.zeros_(m.e_score_correction_bias)
            if isinstance(m, DeepseekV4HashRouter):
                # frozen token-id -> expert-id table; random valid expert ids.
                m.tid2eid = torch.randint(0, cfg.n_routed_experts, m.tid2eid.shape, device=dev, dtype=torch.long)
        elif isinstance(m, DeepseekV4Experts):
            torch.nn.init.normal_(m.gate_up_proj, mean=0.0, std=std)
            torch.nn.init.normal_(m.down_proj, mean=0.0, std=std)
        elif isinstance(m, DeepseekV4Indexer):
            torch.nn.init.zeros_(m.position_bias)
        elif isinstance(m, DeepseekV4HyperHead):
            torch.nn.init.normal_(m.hc_fn, mean=0.0, std=std)
            torch.nn.init.zeros_(m.hc_base)
            torch.nn.init.ones_(m.hc_scale)


def main():
    assert torch.cuda.is_available(), "M0 smoke needs a GPU (CUDA_VISIBLE_DEVICES=0)"
    dev = torch.device("cuda")
    torch.manual_seed(0)

    cfg = tiny_config()
    print(
        f"[cfg] layers={cfg.num_hidden_layers} heads={cfg.num_attention_heads} "
        f"experts={cfg.n_routed_experts} hidden={cfg.hidden_size} hd={cfg.head_dim} "
        f"hc_mult={cfg.hc_mult}"
    )
    print(f"[cfg] layer_types={cfg.layer_types}")
    print(f"[cfg] mlp_layer_types={cfg.mlp_layer_types}")
    print(f"[cfg] compress_rates={cfg.compress_rates}")

    model = V4Model(cfg).to(dev).to(torch.bfloat16)
    init_weights(model, cfg, dev)
    # mHC params stay fp32 (HF _keep_in_fp32_modules); cast them back. hc_head's
    # params were re-init'd above in bf16, so .float() must come after init.
    for layer in model.layers:
        for hc in (layer.attn_hc, layer.ffn_hc):
            hc.float()
    model.hc_head.float()

    B, S = 2, 128
    input_ids = torch.randint(0, cfg.vocab_size, (B, S), device=dev)

    with torch.no_grad():
        out = model(input_ids)

    print(f"[fwd] out shape = {tuple(out.shape)} dtype={out.dtype}")
    exp = (B, S, cfg.hidden_size)
    assert tuple(out.shape) == exp, f"shape mismatch: got {tuple(out.shape)} want {exp}"
    has_nan = torch.isnan(out).any().item()
    has_inf = torch.isinf(out).any().item()
    print(
        f"[fwd] any NaN={has_nan}  any Inf={has_inf}  "
        f"min={out.float().min().item():.4f} max={out.float().max().item():.4f} "
        f"mean={out.float().mean().item():.4f} std={out.float().std().item():.4f}"
    )
    assert not has_nan, "OUTPUT HAS NaN"
    assert not has_inf, "OUTPUT HAS Inf"
    print("\nM0 SMOKE PASS: tiny V4-Flash scaffold forward completed, shape OK, no NaN/Inf.")


if __name__ == "__main__":
    main()
