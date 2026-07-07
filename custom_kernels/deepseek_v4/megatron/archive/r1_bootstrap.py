"""R1 bootstrap: write a random-init torch_dist checkpoint of the LoRA-wrapped V4 model.

The real slime launch (`--load`) needs a megatron torch_dist checkpoint matching the
model's `sharded_state_dict`.  The HF-load path needs a registered V4 AutoBridge (we
don't have one — that's why we use the custom provider), and slime's load wrapper asserts
`--load` exists+non-empty before megatron's load (which itself would tolerate a missing
ckpt and start from random).  So we produce a valid `iter_0000000` once:

  build the model through the SAME provider+LoRA path (so the saved keys are the
  adapter-wrapped layout) -> slime `save(0, model, opt, sched)` -> `--load` it in the run.

This is a single-process (1-GPU) bootstrap: it does the minimal slime/Megatron init the
actor would do, builds via `setup_model_and_optimizer` (which BUILDS but does NOT load —
only `initialize_model_and_optimizer` loads), then saves.  `--pretrained-checkpoint`
(a dummy) satisfies the build-time assert (model.py:216) without triggering a load.

Run (driven by the launcher, with the same arg set as the real run + --save):
  CUDA_VISIBLE_DEVICES=0 python -m custom_kernels.deepseek_v4.megatron.r1_bootstrap <slime args...>
"""

import os


def main():
    # 1-rank torchrun-style env (single GPU); the launcher sets these too.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29610")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")

    import torch

    from slime.utils.arguments import parse_args

    args = parse_args()
    # parse_args sets rank/world_size from env; ensure single-rank.
    args.rank = 0
    args.world_size = 1
    args.local_rank = 0
    # The bootstrap checkpoint only needs model weights. Muon can create empty
    # chained optimizer branches for frozen/nonlinear params, and Megatron's
    # optimizer checkpoint path does not tolerate those.
    args.no_save_optim = True
    args.no_save_rng = True

    torch.cuda.set_device(0)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)

    from slime.backends.megatron_utils.initialize import init
    from slime.backends.megatron_utils.model import save, setup_model_and_optimizer

    init(args)

    # Build the model (provider + LoRA, per --custom-model-provider-path / --v4-lora-dim);
    # setup_model_and_optimizer BUILDS + wraps DDP + makes the dist-optimizer but does NOT
    # load a checkpoint (that is initialize_model_and_optimizer's job).
    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(args, role="actor")

    # quick audit so the bootstrap log shows the LoRA wiring took (base frozen, only LoRA).
    n_train = sum(p.numel() for p in model[0].parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model[0].parameters() if not p.requires_grad)
    print(f"[r1_bootstrap] built model: trainable={n_train:,} frozen={n_frozen:,}", flush=True)

    save(0, model, optimizer, opt_param_scheduler)
    print(f"[r1_bootstrap] saved iter_0000000 to {args.save}", flush=True)

    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
