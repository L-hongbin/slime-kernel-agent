"""R3 full V4LanguageModel EP smoke.

This is one level above ``r3_deepep_ep_smoke.py``: it builds the tiny 3-layer
V4LanguageModel with EP>1, Megatron ``flex/deepep`` MoE dispatch, and Megatron
DDP, then runs a full-model forward/backward. It is still PP1/single-node; it
does not prove the converted PP3 actor checkpoint can train.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig

from ..m0_smoke import tiny_config
from ..model_provider import build_v4_mcore_model
from .r3_deepep_ep_smoke import _finite_param_grad


def _init_dist(order: str):
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=world_size,
            expert_tensor_parallel_size=1,
            order=order,
        )
    tensor_parallel.model_parallel_cuda_manual_seed(1234)
    return rank, world_size


def _has_finite_expert_grad(model) -> bool:
    for name, param in model.named_parameters():
        if "mlp.experts." in name and _finite_param_grad(param):
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--order", default="tp-cp-ep-dp-pp")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for R3 model EP smoke")
    rank, world_size = _init_dist(args.order)
    if world_size < 2:
        raise RuntimeError("R3 model EP smoke requires at least 2 ranks")

    try:
        cfg = tiny_config()
        ep_rank = parallel_state.get_expert_model_parallel_rank()
        torch.manual_seed(args.seed)
        model = build_v4_mcore_model(
            cfg,
            params_dtype=torch.bfloat16,
            expert_model_parallel_size=world_size,
            expert_model_parallel_rank=ep_rank,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="deepep",
            moe_router_dtype="fp32",
        ).cuda()
        model = model.bfloat16()
        model.restore_fp32_modules()
        ddp_config = DistributedDataParallelConfig(grad_reduce_in_fp32=True)
        ddp_model = DistributedDataParallel(model.config, ddp_config, model)

        gen = torch.Generator(device="cpu").manual_seed(args.seed + 1)
        input_ids = torch.randint(
            0,
            cfg.vocab_size,
            (args.batch_size, args.seq_len),
            generator=gen,
            dtype=torch.long,
        ).cuda()
        logits = ddp_model(input_ids=input_ids)
        finite_logits = torch.isfinite(logits).all()
        gathered = [torch.empty_like(logits) for _ in range(world_size)]
        torch.distributed.all_gather(gathered, logits.detach())
        max_rank_diff = torch.stack([(g.float() - gathered[0].float()).abs().max() for g in gathered]).max()

        loss = logits.float().square().mean()
        loss.backward()
        dense_buffers = len(ddp_model.buffers)
        expert_buffers = len(ddp_model.expert_parallel_buffers)
        expert_grad_ok = torch.tensor(int(_has_finite_expert_grad(ddp_model.module)), device=logits.device)
        finite_logits_i = finite_logits.to(torch.int32)
        torch.distributed.all_reduce(finite_logits_i, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(max_rank_diff, op=torch.distributed.ReduceOp.MAX)
        torch.distributed.all_reduce(expert_grad_ok, op=torch.distributed.ReduceOp.MIN)

        passed = bool(
            finite_logits_i.item() == 1
            and torch.isfinite(loss).item()
            and max_rank_diff.item() == 0.0
            and dense_buffers > 0
            and expert_buffers > 0
            and expert_grad_ok.item() == 1
        )
        if rank == 0:
            print(
                json.dumps(
                    {
                        "passed": passed,
                        "world_size": world_size,
                        "dispatcher": "flex/deepep",
                        "model": "tiny V4LanguageModel",
                        "batch_size": args.batch_size,
                        "seq_len": args.seq_len,
                        "logits_shape": list(logits.shape),
                        "loss": float(loss.item()),
                        "finite_logits": bool(finite_logits_i.item()),
                        "max_rank_diff": float(max_rank_diff.item()),
                        "ddp_dense_buffers": dense_buffers,
                        "ddp_expert_buffers": expert_buffers,
                        "expert_grad_ok": bool(expert_grad_ok.item()),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if not passed:
            raise RuntimeError("R3 full-model EP smoke failed")
    finally:
        if parallel_state.is_initialized():
            parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
