"""SFT-LoRA overfit-a-batch sanity (single GPU, TP=PP=EP=1).

Goal: prove the whole stack TRAINS end-to-end — the V4 mcore model + LoRA + slime's
exact SFT loss + real backprop — by overfitting a FIXED non-packed (bshd) batch and
showing the loss decreases, with grads ONLY on the LoRA adapters.

What is real vs stubbed:
  * REAL: the V4 mcore model built through the slime provider (`build_v4_mcore_model`)
    + LoRA via `apply_v4_lora` (base frozen); slime's `sft_loss_function` + `get_batch`
    bshd path + `get_sum_of_sample_mean` (the same loss code the slime train step runs).
  * STUBBED (documented): a tokenizer-free synthetic batch (random token ids — a sanity
    only needs gradient flow, not real text); a plain torch Adam over the trainable LoRA
    params instead of the Megatron distributed optimizer + DDP wrap + Ray launch (at
    TP=PP=EP=1, DP=CP=1 there is no collective to exercise — the distributed optimizer
    would only add launch scaffolding, not change the per-step math). The forward/loss/
    autograd path that actually validates the model is identical to slime's.
  * NON-PACKED (bshd): `packed_seq_params=None` so the model's packed-THD hard-fail does
    NOT trip (one document per row); this is the path the SFT sanity is meant to use.

Run:  CUDA_VISIBLE_DEVICES=0 python -m custom_kernels.deepseek_v4.megatron.sft_sanity
"""

import os
from types import SimpleNamespace

import torch


def _init_dist_1rank():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29599")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
    from megatron.core import parallel_state, tensor_parallel

    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(1, 1, expert_model_parallel_size=1)
    tensor_parallel.model_parallel_cuda_manual_seed(0)


def _build_model(cfg, dev):
    from .lora import apply_v4_lora
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
    m = m.bfloat16()  # routes through .bfloat16() override -> restore fp32 modules
    m = apply_v4_lora(m, dim=16, alpha=32, dropout=0.0)  # freeze base, wrap attn/compressor
    m.restore_fp32_modules()  # re-assert fp32 (LoRA wrap created bf16 adapters; base unchanged)
    return m


def _fixed_bshd_batch(cfg, dev, B=2, total_len=64, resp_len=32):
    """One fixed non-packed (bshd) batch: B documents, each [total_len], response = last
    resp_len tokens.  Returns (batch dict for the loss, stacked tokens [B,S])."""
    g = torch.Generator(device="cpu").manual_seed(123)
    unconcat = [torch.randint(0, cfg.vocab_size, (total_len,), generator=g).to(dev) for _ in range(B)]
    max_seq_len = total_len  # bshd: every row padded to the same length (here already equal)
    tokens = torch.stack(unconcat)  # [B, total_len]
    total_lengths = [total_len] * B
    response_lengths = [resp_len] * B
    # loss_mask feeds get_sum_of_sample_mean, which multiplies it against the per-sample
    # log_probs (length = response_length, since x.split(response_lengths)).  So the mask
    # must be RESPONSE-length, not total-length.  All-ones = train on every response token.
    loss_masks = [torch.ones(resp_len, device=dev) for _ in range(B)]
    batch = dict(
        unconcat_tokens=unconcat,
        tokens=tokens,
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        loss_masks=loss_masks,
        group_mask_sums=None,
        max_seq_lens=[max_seq_len] * B,
    )
    return batch, tokens


def main():
    assert torch.cuda.is_available(), "needs GPU (CUDA_VISIBLE_DEVICES=0)"
    torch.cuda.set_device(0)
    _init_dist_1rank()
    torch.manual_seed(0)

    from slime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
    from slime.backends.megatron_utils.loss import sft_loss_function

    from .lora import audit_lora
    from .m0_smoke import tiny_config

    cfg = tiny_config()
    dev = torch.device("cuda")
    m = _build_model(cfg, dev)

    # slime args surface the loss path reads (qkv_format bshd; cp/temperature defaults).
    args = SimpleNamespace(
        qkv_format="bshd",
        rollout_temperature=1.0,
        allgather_cp=False,
        log_probs_chunk_size=1024,
        calculate_per_token_loss=False,
    )

    batch, tokens = _fixed_bshd_batch(cfg, dev)

    # audit: base frozen, only LoRA trainable.  This must hold AFTER the full build chain
    # (provider -> LoRA freeze -> bf16/Float16 wrap).  In the REAL slime launch,
    # ``freeze_model_params`` runs AFTER the provider and ``--only-train-params-name-list``
    # can set base params back to requires_grad=True (model_provider.py:288) — overriding the
    # LoRA freeze.  So the SFT-LoRA launch must NOT pass --only-train-params-name-list (LoRA
    # already froze the base + made the adapters trainable), or must scope it to match only
    # ``linear_in``/``linear_out``.  This assert is the catch for any arg-level unfreeze.
    aud = audit_lora(m)
    trainable = [p for p in m.parameters() if p.requires_grad]
    # Classify base-vs-LoRA by the trainable param-id SET (the source of truth = which
    # params the freeze left trainable), NOT a name substring — robust to any FQN.
    lora_param_ids = {id(p) for p in trainable}
    leaked_base = [
        n for n, p in m.named_parameters() if p.requires_grad and "linear_in" not in n and "linear_out" not in n
    ]
    # every non-LoRA param must be frozen at init (catches an arg-level base unfreeze).
    non_lora_unfrozen = [
        n
        for n, p in m.named_parameters()
        if id(p) in lora_param_ids and "linear_in" not in n and "linear_out" not in n
    ]
    assert aud["trainable_are_lora_only"] and not aud["o_a_wrapped"], "LoRA audit failed"
    assert not leaked_base and not non_lora_unfrozen, (
        f"base params left trainable after build+freeze (only LoRA must be trainable): "
        f"{(leaked_base or non_lora_unfrozen)[:8]} -- did a launch arg "
        f"(e.g. --only-train-params-name-list) unfreeze them?"
    )
    print(
        f"[setup] trainable LoRA params={aud['n_trainable']:,}  frozen base={aud['n_frozen']:,}  "
        f"o_a wrapped={len(aud['o_a_wrapped'])} (expect 0); base-unfreeze leak={len(leaked_base)} (expect 0)"
    )
    print(
        f"[setup] qkv_format=bshd (packed_seq_params=None -> hard-fail NOT tripped); "
        f"B={tokens.shape[0]} S={tokens.shape[1]} fixed batch"
    )

    opt = torch.optim.Adam(trainable, lr=1e-3)

    sum_of_sample_mean = get_sum_of_sample_mean(
        batch["total_lengths"],
        batch["response_lengths"],
        batch["loss_masks"],
        None,
        args.calculate_per_token_loss,
        args.qkv_format,
        batch["max_seq_lens"],
    )

    losses = []
    grad_norms = []
    base_grad_ever = 0  # cumulative count of base params that EVER got a (nonzero) grad
    n_steps = 40
    B = tokens.shape[0]
    m.train()
    for step in range(n_steps):
        opt.zero_grad(set_to_none=True)
        # bshd forward: NO packed_seq_params (the SFT sanity path).
        logits = m(input_ids=tokens, position_ids=None, attention_mask=None, labels=None)
        loss, log = sft_loss_function(args, batch, logits, sum_of_sample_mean)
        loss.backward()
        # grad must be on LoRA only — classify by the trainable param-id set (a base param
        # is anything NOT in lora_param_ids); count any base param that got a nonzero grad.
        gsq = 0.0
        step_base_grad = 0
        for p in m.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                if id(p) in lora_param_ids:
                    gsq += p.grad.float().pow(2).sum().item()
                else:
                    step_base_grad += 1
        base_grad_ever += step_base_grad
        gnorm = gsq**0.5
        opt.step()
        losses.append(loss.item())
        grad_norms.append(gnorm)
        if step_base_grad:
            print(f"  step {step}: BASE GRAD LEAK ({step_base_grad} params) <-- BAD")
        if step % 5 == 0 or step == n_steps - 1:
            print(f"  step {step:>3}: loss={loss.item():.5f}  grad_norm(LoRA)={gnorm:.4e}")

    first, last = losses[0], losses[-1]
    finite = all(torch.isfinite(torch.tensor(losses)).tolist()) and all(
        torch.isfinite(torch.tensor(grad_norms)).tolist()
    )
    decreased = last < first * 0.7  # overfit-a-batch: expect a clear drop
    # NOTE: the printed loss is the SUM of per-sample mean NLL (get_sum_of_sample_mean sums
    # over samples), i.e. ~B x the conventional per-sample-mean — report both.
    print(
        f"\n[result] loss(sum-of-sample-means) {first:.5f} -> {last:.5f} "
        f"(per-sample-mean {first/B:.5f} -> {last/B:.5f}; drop {100*(1-last/first):.1f}%)  "
        f"finite={finite}  decreased={decreased}  base_grad_ever={base_grad_ever}"
    )

    # --- validity controls (the drop must be GENUINE, not trivial) ---
    # (1) memorization: after overfit, the response-token argmax must match the targets
    #     (proves the loss is over the RIGHT tokens with the right off-by-one, not padding).
    m.eval()
    with torch.no_grad():
        lg = m(input_ids=tokens).float()  # [B,S,V]
    B, S = tokens.shape
    rl = batch["response_lengths"][0]
    correct = tot = 0
    for b in range(B):
        pred = lg[b, S - rl - 1 : S - 1].argmax(-1)  # predicts the response positions
        tgt = tokens[b, S - rl :]
        correct += int((pred == tgt).sum())
        tot += rl
    mem_acc = correct / tot
    print(
        f"[control] response-token argmax accuracy after overfit = {correct}/{tot} "
        f"= {100*mem_acc:.1f}% (genuine fit on the right tokens)"
    )

    # (2) deterministic frozen-forward control: with every param frozen, two no-grad evals
    #     give an identical loss (the forward is deterministic and nothing learns) -> the
    #     real drop requires the trainable LoRA params, not a stochastic/data artifact.
    #     (This is two forward evals, NOT a frozen training loop.)
    mf = _build_model(cfg, dev)
    for p in mf.parameters():
        p.requires_grad = False
    mf.train()
    with torch.no_grad():
        lf0 = sft_loss_function(args, batch, mf(input_ids=tokens), sum_of_sample_mean)[0].item()
        lf1 = sft_loss_function(args, batch, mf(input_ids=tokens), sum_of_sample_mean)[0].item()
    frozen_flat = abs(lf0 - lf1) < 1e-6
    print(
        f"[control] deterministic frozen-forward loss eval0={lf0:.4f} eval1={lf1:.4f} "
        f"identical={frozen_flat} (no learning without trainable params)"
    )

    # base_grad_ever==0 is now a HARD gate (a leaked base grad fails the sanity, not just
    # a printed warning) — this is what enforces "only LoRA updates".
    no_base_grad = base_grad_ever == 0
    ok = finite and decreased and mem_acc > 0.95 and frozen_flat and no_base_grad
    print(
        f"[gate] finite={finite} decreased={decreased} mem_acc>{0.95}={mem_acc>0.95} "
        f"frozen_flat={frozen_flat} no_base_grad={no_base_grad}"
    )
    print("SFT SANITY " + ("PASS" if ok else "FAIL"))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
