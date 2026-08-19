"""Validate the V4 LoRA wiring (TP=PP=EP=1) on GPU 0.

Checks (the M-LoRA gate):
  (a) trainable params == ONLY LoRA adapters; base is frozen (requires_grad=False).
  (b) forward is UNCHANGED by LoRA at init (zero-init linear_out => identity) — the
      LoRA model's logits match the no-LoRA model's logits bit-for-bit.
  (c) a backward populates grads ONLY on LoRA params; NO grad on frozen base params
      (sinks/position_bias/kv_norm/experts/router/embedding/output_layer).
  (d) audit the wrapped modules: the 6 target linears/layer ARE wrapped, o_a_proj is
      NOT wrapped, experts/router/shared-MLP are NOT wrapped.

Run:  CUDA_VISIBLE_DEVICES=0 python -m custom_kernels.deepseek_v4.megatron.lora_validate
"""

import os

import torch


def _init_dist_1rank():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29595")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
    from megatron.core import parallel_state, tensor_parallel

    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(1, 1, expert_model_parallel_size=1)
    tensor_parallel.model_parallel_cuda_manual_seed(0)


def _build(cfg, dev):
    from .m0_smoke import init_weights as init_hf_owned
    from .mcore_model import V4GroupedExperts, V4HashRouter, V4TopKRouter
    from .model_provider import build_v4_mcore_model

    m = build_v4_mcore_model(cfg).to(dev)
    std = cfg.initializer_range
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, (V4TopKRouter, V4HashRouter)):
                torch.nn.init.normal_(mod.weight, mean=0.0, std=std)
                if isinstance(mod, V4HashRouter):
                    mod.tid2eid = torch.randint(
                        0, cfg.n_routed_experts, mod.tid2eid.shape, device=dev, dtype=torch.long
                    )
            elif isinstance(mod, V4GroupedExperts):
                torch.nn.init.normal_(mod.gate_up_proj, mean=0.0, std=std)
                torch.nn.init.normal_(mod.down_proj, mean=0.0, std=std)
    init_hf_owned(m, cfg, dev)
    m = m.bfloat16()  # routes through override -> restore_fp32_modules
    return m


def main():
    assert torch.cuda.is_available(), "needs GPU (CUDA_VISIBLE_DEVICES=0)"
    torch.cuda.set_device(0)
    _init_dist_1rank()
    torch.manual_seed(0)

    from .lora import apply_v4_lora, audit_lora
    from .m0_smoke import tiny_config

    cfg = tiny_config()
    dev = torch.device("cuda")
    B, S = 2, 128
    input_ids = torch.randint(0, cfg.vocab_size, (B, S), device=dev)

    # Build ONE model, snapshot its no-LoRA logits, then apply LoRA to the SAME model
    # and compare (so the comparison isolates LoRA, not two independent random inits).
    torch.manual_seed(0)
    m = _build(cfg, dev).eval()
    with torch.no_grad():
        logits_ref = m(input_ids)
    m = apply_v4_lora(m, dim=16, alpha=32, dropout=0.0)
    m.eval()

    aud = audit_lora(m)
    n_layers = cfg.num_hidden_layers
    ok = True

    # (d) audit wrapped set — require the EXACT expected FQN set (a missing target OR an
    # accidental extra wrap, e.g. compressor.indexer.kv_proj, must FAIL — not just len>0).
    attn_targets = ["q_a_proj", "q_b_proj", "kv_proj", "o_b_proj"]
    expected = set()
    for i, lt in enumerate(cfg.layer_types):
        for t in attn_targets:
            expected.add(f"layers.{i}.self_attn.{t}")
        if lt != "sliding_attention":  # CSA/HCA layers have a compressor
            expected.add(f"layers.{i}.self_attn.compressor.kv_proj")
            expected.add(f"layers.{i}.self_attn.compressor.gate_proj")
    got = set(aud["wrapped"])
    missing = expected - got
    extra = got - expected
    print(f"[audit] wrapped={len(got)} expected={len(expected)}")
    if missing:
        print("   MISSING (target not wrapped):", sorted(missing))
    if extra:
        print("   EXTRA (unintended wrap):", sorted(extra))
    set_ok = got == expected
    o_a_ok = len(aud["o_a_wrapped"]) == 0
    print(f"[audit] exact-set match={set_ok}  o_a wrapped={aud['o_a_wrapped']} ({'OK' if o_a_ok else 'BAD'})")
    ok = ok and set_ok and o_a_ok

    # (a) trainable == only LoRA.
    print(
        f"[params] trainable={aud['n_trainable']:,}  frozen={aud['n_frozen']:,}  "
        f"trainable_are_lora_only={aud['trainable_are_lora_only']}"
    )
    ok = ok and aud["trainable_are_lora_only"] and aud["n_trainable"] > 0
    # sanity: a few representative base params are frozen.
    base_frozen = []
    for n, p in m.named_parameters():
        if any(
            k in n
            for k in (
                "sinks",
                "position_bias",
                "experts.gate_up_proj",
                "embedding.word_embeddings",
                "output_layer.weight",
                "hc_head",
                "e_score_correction_bias",
            )
        ):
            base_frozen.append((n, p.requires_grad))
    leaked = [n for n, rg in base_frozen if rg]
    print(f"[params] representative base params frozen: {'OK' if not leaked else 'LEAK: ' + str(leaked)}")
    ok = ok and not leaked

    # (b) forward unchanged (zero-init LoRA == identity).
    with torch.no_grad():
        logits_lora = m(input_ids)
    rel = ((logits_lora.float() - logits_ref.float()).norm() / logits_ref.float().norm().clamp_min(1e-12)).item()
    fwd_ok = rel < 1e-5
    print(f"[forward] LoRA-vs-noLoRA logits rel={rel:.2e} -> {'OK (identity)' if fwd_ok else 'BAD'}")
    ok = ok and fwd_ok

    # (c) backward grads ONLY on LoRA params: assert frozen base params get grad IS NONE
    # (stronger than abs().sum()==0 — a leaked grad that happens to be ~0 would still be a
    # bug), AND assert EVERY adapter is LIVE.  Note the LoRA cold-start: linear_out (B) is
    # zero-init, so at init linear_in's (A) grad routes through B's zero weight and is 0;
    # only after one opt step (B != 0) does A receive a gradient.  So: at init every B must
    # get a nonzero grad; after one step every A must too.
    opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=1e-2)
    m.train()

    def _grad_counts():
        base_with_grad, b_no_grad, a_no_grad = [], [], []
        n_b = n_a = 0
        for n, p in m.named_parameters():
            if "linear_out" in n:
                n_b += 1
                if p.grad is None or p.grad.abs().sum() == 0:
                    b_no_grad.append(n)
            elif "linear_in" in n:
                n_a += 1
                if p.grad is None or p.grad.abs().sum() == 0:
                    a_no_grad.append(n)
            elif p.grad is not None:  # frozen base param must not even allocate a grad
                base_with_grad.append(n)
        return base_with_grad, b_no_grad, a_no_grad, n_b, n_a

    m.zero_grad(set_to_none=True)
    m(input_ids).float().pow(2).mean().backward()
    base_leak, b_no, a_no, n_b, n_a = _grad_counts()
    print(
        f"[backward init] B(linear_out) without grad={len(b_no)}/{n_b} (must be 0); "
        f"A(linear_in) without grad={len(a_no)}/{n_a} (==all at init, cold-start); "
        f"base params with a grad tensor={len(base_leak)} (must be 0)"
    )
    if base_leak:
        print("   BASE GRAD LEAK (grad is not None):", base_leak[:10])
    opt.step()  # B becomes nonzero
    m.zero_grad(set_to_none=True)
    m(input_ids).float().pow(2).mean().backward()
    base_leak2, _, a_no2, _, _ = _grad_counts()
    print(
        f"[backward step1] A(linear_in) without grad={len(a_no2)}/{n_a} (must be 0 now); "
        f"base leak={len(base_leak2)}"
    )
    bwd_ok = not base_leak and not base_leak2 and len(b_no) == 0 and n_b > 0 and len(a_no2) == 0 and n_a > 0
    ok = ok and bwd_ok

    print("\nLORA VALIDATE " + ("PASS" if ok else "FAIL"))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
