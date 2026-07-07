"""R1 single-process driver: the slime train wrapper on the V4 mcore model.

Exercises exactly what the stub sanity (sft_sanity.py) STUBBED — the real
Float16Module bf16 wrap + DDP contiguous grad buffers + Megatron distributed optimizer
(fp32 master copies for ONLY the LoRA params) + real torch_dist `--load` + slime's real
`train` / `train_one_step` / `forward_backward_func` / `sft_loss_function` — driven
directly (no Ray) on a FIXED synthetic bshd batch. Single GPU, TP=PP=EP=1.

Why no Ray: this node can have a separate idle eval Ray cluster. This driver sidesteps
shared-cluster risk while still calling slime's `train()` wrapper after mpu/dist init.
It does not validate Ray placement groups or `train.py` orchestration.

Validates (the R1 gate):
  * non-OOM, loss finite, fixed-batch loss DECREASES over ~40 steps (overfit);
  * optimizer param groups contain ONLY LoRA params (count == 1.61M / 32 adapters) —
    the dist-optimizer built fp32 masters only for the trainable LoRA;
  * base requires_grad=False throughout; unexpected trainable base params are hook-audited;
  * Float16Module wrap kept the fp32 modules fp32 (mHC / hc_head / correction-bias);
  * cross-check the loss curve against the stub (sft_sanity 13.98->~0.002).

Run via the launcher (same args as the bootstrap, + --load the bootstrap dir):
  python -m custom_kernels.deepseek_v4.megatron.r1_train <slime args...>
"""


def _round_up(value, multiple):
    return (value + multiple - 1) // multiple * multiple


def _fixed_rollout_data(cfg_vocab=512, B=2, total_len=64, resp_len=32, steps=40, max_seq_len=None, seed=123):
    """Build the rollout_data dict that slime train() consumes.

    The sample tensors are fixed and reused every step; ``micro_batch_indices`` repeats
    the same [0..B-1] microbatch so the wrapper can run ``steps`` train_one_step calls
    over one DataIterator, matching the actor/train contract for DP=1.
    """
    import torch

    g = torch.Generator(device="cpu").manual_seed(seed)
    dev = torch.cuda.current_device()
    tokens, loss_masks = [], []
    for _ in range(B):
        tokens.append(torch.randint(0, cfg_vocab, (total_len,), generator=g, dtype=torch.long).to(dev))
        loss_masks.append(torch.ones(resp_len, dtype=torch.int, device=dev))
    max_seq_len = max_seq_len or total_len
    return {
        "tokens": tokens,  # list[B] of [total_len] long (GPU)
        "loss_masks": loss_masks,  # list[B] of [resp_len] int (GPU)
        "total_lengths": [total_len] * B,
        "response_lengths": [resp_len] * B,
        "max_seq_lens": [max_seq_len] * B,  # bshd: actor-style padded length
        "rewards": [0.0] * B,
        "raw_reward": [0.0] * B,
        "truncated": [0] * B,
        "sample_indices": list(range(B)),
        "group_ids": list(range(B)),
        "group_mask_sums": torch.tensor([resp_len] * B, dtype=torch.float32, device=dev),
        # One microbatch per training step, all B samples, DP=1.
        "num_microbatches": [1] * steps,
        "global_batch_sizes": [B] * steps,
        # DataIterator.get_next consumes a flat sample-index list per microbatch.
        "micro_batch_indices": [list(range(B)) for _ in range(steps)],
    }


def main():
    import torch

    from slime.utils.arguments import parse_args

    args = parse_args()
    args.rank = 0
    args.world_size = 1
    args.local_rank = 0
    torch.cuda.set_device(0)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)

    from slime.backends.megatron_utils.data import get_data_iterator
    from slime.backends.megatron_utils.initialize import init
    from slime.backends.megatron_utils.model import initialize_model_and_optimizer, train

    init(args)

    # real build (provider+LoRA) + DDP + dist-optimizer + LOAD the bootstrap torch_dist ckpt.
    model, optimizer, opt_param_scheduler, iteration = initialize_model_and_optimizer(args, role="actor")
    model[0].role = "actor"
    # The dist_checkpointing load is STRICT by default — it raises on any missing/unexpected
    # sharded tensor.  Reaching here (no exception) => the LoRA-wrapped model's
    # sharded_state_dict keys matched the saved ckpt exactly (0 missing / 0 unexpected).
    print(
        f"[r1_train] loaded checkpoint at iteration={iteration} (strict dist-ckpt load OK "
        f"=> 0 missing/unexpected model keys through the LoRA-wrapped sharded_state_dict)",
        flush=True,
    )

    # --- invariant asserts (the R1 gate, on the REAL wrapped model) ---
    from megatron.core.distributed import DistributedDataParallel as DDP
    from megatron.core.transformer.module import Float16Module

    inner = model[0]
    while isinstance(inner, (DDP, Float16Module)):
        inner = inner.module
    # (1) optimizer param groups hold ONLY LoRA params.  The DISTRIBUTED optimizer keeps
    # its own fp32 MASTER copies (distinct tensor objects/ids from the model's bf16
    # params), so identity-compare is wrong — compare the total numel against the model's
    # trainable (LoRA) numel: if the optimizer's param count == the LoRA trainable count,
    # the dist-optimizer skipped the frozen base (built masters for LoRA only).
    n_opt = sum(p.numel() for grp in optimizer.param_groups for p in grp["params"])
    lora_named = [(n, p) for n, p in inner.named_parameters() if "linear_in" in n or "linear_out" in n]
    base_named = [(n, p) for n, p in inner.named_parameters() if "linear_in" not in n and "linear_out" not in n]
    n_lora = sum(p.numel() for _, p in lora_named)
    n_trainable = sum(p.numel() for _, p in inner.named_parameters() if p.requires_grad)
    base_frozen = all(not p.requires_grad for _, p in base_named)
    # IDENTITY check (codex R1 P2): numel-equality alone could pass under an equal-numel
    # swap.  The dist-optimizer is built from the SOURCE params where requires_grad=True;
    # assert that trainable-param SET (by id) == LoRA-param SET (by id), exactly.  (The
    # dist-optimizer's fp32 MASTERS are distinct objects — we check the SOURCES, i.e. the
    # model's trainable params the optimizer was built from, not the masters.)
    lora_ids_set = {id(p) for _, p in lora_named}
    trainable_ids_set = {id(p) for _, p in inner.named_parameters() if p.requires_grad}
    opt_sources_only_lora = (trainable_ids_set == lora_ids_set) and len(lora_ids_set) > 0
    opt_only_lora = (n_opt == n_lora == n_trainable) and n_lora > 0 and opt_sources_only_lora
    print(
        f"[r1_train] optimizer params={n_opt:,} model-LoRA={n_lora:,} model-trainable={n_trainable:,} "
        f"(all equal == LoRA only); numel_ok={n_opt == n_lora == n_trainable}; "
        f"identity(trainable-set==LoRA-set)={opt_sources_only_lora}; opt_only_lora={opt_only_lora}; "
        f"base_frozen={base_frozen}",
        flush=True,
    )

    # (2) Float16Module wrap kept the fp32 modules fp32 (mHC / hc_head / correction-bias).
    fp32_ok = True
    for layer in inner.layers:
        if layer.attn_hc.fn.dtype != torch.float32 or layer.ffn_hc.fn.dtype != torch.float32:
            fp32_ok = False
        g = layer.mlp.gate
        if hasattr(g, "e_score_correction_bias") and g.e_score_correction_bias.dtype != torch.float32:
            fp32_ok = False
    if inner.hc_head.hc_fn.dtype != torch.float32:
        fp32_ok = False
    qa_bf16 = any(p.dtype == torch.bfloat16 for n, p in inner.named_parameters() if n.endswith("q_a_proj.weight"))
    print(
        f"[r1_train] fp32-restore-after-Float16Module: mHC/hc_head/bias fp32={fp32_ok}; "
        f"a normal linear is bf16={qa_bf16}",
        flush=True,
    )
    assert opt_only_lora and base_frozen and fp32_ok and qa_bf16, "R1 invariant asserts FAILED"

    base_grad_hook_hits = []
    hook_handles = []
    for n, p in base_named:
        if p.requires_grad:
            hook_handles.append(p.register_hook(lambda grad, name=n: base_grad_hook_hits.append(name)))

    # --- the real train wrapper over a fixed batch (40 steps) ---
    n_steps = 40
    from megatron.core import mpu

    pad_size = mpu.get_tensor_model_parallel_world_size() * args.data_pad_size_multiplier
    rollout_data = _fixed_rollout_data(
        cfg_vocab=args.vocab_size,
        steps=n_steps,
        max_seq_len=_round_up(64, pad_size),
    )
    di = get_data_iterator(rollout_data)
    captured = []

    from slime.utils import logging_utils

    orig_log = logging_utils.log

    def capture_log(log_args, metrics, step_key):
        if "train/loss" in metrics:
            captured.append(
                (
                    int(metrics.get("train/step", len(captured))),
                    float(metrics["train/loss"]),
                    float(metrics.get("train/grad_norm", float("nan"))),
                )
            )
        return orig_log(log_args, metrics, step_key)

    logging_utils.log = capture_log
    try:
        train(
            0,
            model,
            optimizer,
            opt_param_scheduler,
            di,
            rollout_data["num_microbatches"],
            rollout_data["global_batch_sizes"],
        )
    finally:
        logging_utils.log = orig_log
        for handle in hook_handles:
            handle.remove()

    losses = [loss for _, loss, _ in captured]
    grad_norms = [grad for _, _, grad in captured]
    for i, (_, lv, gv) in enumerate(captured):
        if i % 5 == 0 or i == n_steps - 1:
            print(f"  step {i:>3}: loss={lv:.5f} grad_norm={gv:.4e}", flush=True)

    import math

    assert len(losses) == n_steps, f"expected {n_steps} captured train/loss values, got {len(losses)}"
    first, last = losses[0], losses[-1]
    finite = all(math.isfinite(x) for x in losses) and all(math.isfinite(x) for x in grad_norms)
    decreased = last < first * 0.7
    base_frozen_after = all(not p.requires_grad for _, p in base_named)
    print(
        f"\n[r1_train result] loss {first:.5f} -> {last:.5f} (drop {100*(1-last/max(first,1e-9)):.1f}%) "
        f"finite={finite} decreased={decreased} base_frozen_after={base_frozen_after} "
        f"base_grad_hook_hits={len(base_grad_hook_hits)}",
        flush=True,
    )
    ok = finite and decreased and base_frozen_after and not base_grad_hook_hits
    print("R1 TRAIN " + ("PASS" if ok else "FAIL"), flush=True)
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
