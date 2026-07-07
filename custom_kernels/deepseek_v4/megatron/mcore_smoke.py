"""Build + no-NaN forward smoke for the V4 mcore LanguageModel (TP=PP=EP=1).

Inits a 1-rank Megatron parallel state, builds the model via the provider, runs a
forward, asserts fp32 [B,S,V] logits + no NaN/Inf.  Mirrors m0_smoke but for the
mcore LanguageModule surface.

Run:  CUDA_VISIBLE_DEVICES=0 python -m custom_kernels.deepseek_v4.megatron.mcore_smoke
"""

import os

import torch


def _init_dist_1rank():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
    from megatron.core import parallel_state

    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
        )
    from megatron.core import tensor_parallel

    tensor_parallel.model_parallel_cuda_manual_seed(0)


def main():
    assert torch.cuda.is_available(), "needs a GPU (CUDA_VISIBLE_DEVICES=0)"
    torch.cuda.set_device(0)
    _init_dist_1rank()
    torch.manual_seed(0)

    from .m0_smoke import init_weights as init_hf_owned
    from .m0_smoke import tiny_config
    from .mcore_model import V4GroupedExperts, V4HashRouter, V4TopKRouter
    from .model_provider import build_v4_mcore_model

    cfg = tiny_config()
    dev = torch.device("cuda")
    model = build_v4_mcore_model(cfg, params_dtype=torch.bfloat16).to(dev).to(torch.bfloat16)

    # init the HF-style empty params (routers/experts/hc_head) + V4 custom routers/experts.
    std = cfg.initializer_range
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, (V4TopKRouter, V4HashRouter)):
                torch.nn.init.normal_(m.weight, mean=0.0, std=std)
                if isinstance(m, V4HashRouter):
                    m.tid2eid = torch.randint(0, cfg.n_routed_experts, m.tid2eid.shape, device=dev, dtype=torch.long)
            elif isinstance(m, V4GroupedExperts):
                torch.nn.init.normal_(m.gate_up_proj, mean=0.0, std=std)
                torch.nn.init.normal_(m.down_proj, mean=0.0, std=std)
    init_hf_owned(model, cfg, dev)  # hc_head (+ any remaining HF-owned modules)

    # restore the fp32-kept modules (mHC + hc_head + top-k correction bias) — the same
    # helper slime's Float16Module path invokes via the model's .bfloat16() override.
    model.restore_fp32_modules()
    model.eval()

    B, S = 2, 128
    input_ids = torch.randint(0, cfg.vocab_size, (B, S), device=dev)
    with torch.no_grad():
        out = model(input_ids)

    print(f"[fwd] logits shape={tuple(out.shape)} dtype={out.dtype}")
    exp = (B, S, cfg.vocab_size)
    assert tuple(out.shape) == exp, f"shape {tuple(out.shape)} != {exp}"
    assert out.dtype == torch.float32, f"logits must be fp32, got {out.dtype}"
    assert not torch.isnan(out).any(), "NaN in logits"
    assert not torch.isinf(out).any(), "Inf in logits"
    print(f"[fwd] no NaN/Inf  min={out.min():.3f} max={out.max():.3f} std={out.std():.4f}")
    print("\nMCORE SMOKE PASS: V4 LanguageModel builds + forward returns fp32 [B,S,V].")


if __name__ == "__main__":
    main()
