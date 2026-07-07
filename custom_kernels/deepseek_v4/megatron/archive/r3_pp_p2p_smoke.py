"""R3 actual Megatron PP P2P smoke for V4's 4D intermediate stream.

This runs two local ranks with PP=2 and Megatron's non-interleaved pipeline
schedule. The important part is ``adjust_tensor_shapes_fn``: stock Megatron PP
communicates ``[S,B,H]`` tensors, while V4 stages send ``[B,S,hc_mult,H]``.
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.pipeline_parallel import get_forward_backward_func

from ..m0_smoke import tiny_config
from ..model_provider import build_v4_mcore_model
from .r3_pp_slice_smoke import _copy_stage_from_full


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
            pipeline_model_parallel_size=world_size,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
            order=order,
        )
    tensor_parallel.model_parallel_cuda_manual_seed(1234)
    return rank, world_size


def _make_data_iter(input_ids):
    yielded = False

    class _Iter:
        def __iter__(self):
            return self

        def __next__(self):
            nonlocal yielded
            if yielded:
                raise StopIteration
            yielded = True
            return input_ids

    return _Iter()


def _forward_step(data_iterator, model):
    input_ids = next(data_iterator)
    output = model(input_ids=input_ids)

    def collect(output_tensor, non_loss_data=False):
        del non_loss_data
        return output_tensor.detach()

    return output, collect


def _max_float(value: float, device) -> float:
    tensor = torch.tensor(float(value), device=device, dtype=torch.float32)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return float(tensor.item())


def _bool_all(value: bool, device) -> bool:
    tensor = torch.tensor(int(value), device=device, dtype=torch.int32)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
    return bool(tensor.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--first-stage-layers", type=int, default=1)
    parser.add_argument("--order", default="tp-cp-ep-dp-pp")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for R3 PP P2P smoke")
    rank, world_size = _init_dist(args.order)
    if world_size != 2:
        raise RuntimeError("R3 PP P2P smoke currently expects exactly 2 ranks")

    try:
        cfg = tiny_config()
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        pre_process = pp_rank == 0
        post_process = pp_rank == world_size - 1
        torch.manual_seed(args.seed)
        full = build_v4_mcore_model(cfg, params_dtype=torch.bfloat16).cuda().bfloat16()
        full.restore_fp32_modules()

        torch.manual_seed(args.seed + 1 + rank)
        stage = (
            build_v4_mcore_model(
                cfg,
                pre_process=pre_process,
                post_process=post_process,
                params_dtype=torch.bfloat16,
                pipeline_model_parallel_size=world_size,
                num_layers_in_first_pipeline_stage=args.first_stage_layers,
            )
            .cuda()
            .bfloat16()
        )
        stage.restore_fp32_modules()
        copied, skipped = _copy_stage_from_full(stage, full)

        gen = torch.Generator(device="cpu").manual_seed(args.seed + 3)
        input_ids = torch.randint(
            0,
            cfg.vocab_size,
            (args.batch_size, args.seq_len),
            generator=gen,
            dtype=torch.long,
        ).cuda()

        def adjust_tensor_shapes(recv_shapes, send_shapes):
            del recv_shapes, send_shapes
            shape = (args.batch_size, args.seq_len, cfg.hc_mult, cfg.hidden_size)
            return [shape], [shape]

        forward_backward = get_forward_backward_func(pp_size=world_size, vp_size=None)
        outputs = forward_backward(
            forward_step_func=_forward_step,
            data_iterator=_make_data_iter(input_ids),
            model=[stage],
            num_microbatches=1,
            seq_length=args.seq_len,
            micro_batch_size=args.batch_size,
            decoder_seq_length=args.seq_len,
            forward_only=True,
            collect_non_loss_data=True,
            adjust_tensor_shapes_fn=adjust_tensor_shapes,
        )

        local_passed = True
        max_abs = 0.0
        rel = 0.0
        logits_shape = None
        if post_process:
            if len(outputs) != 1:
                raise RuntimeError(f"expected one output on last PP stage, got {len(outputs)}")
            pp_logits = outputs[0]
            with torch.no_grad():
                full_logits = full(input_ids=input_ids)
            diff = (pp_logits.float() - full_logits.float()).abs()
            max_abs = float(diff.max().item())
            rel = float((diff.max() / full_logits.float().abs().amax().clamp_min(1e-6)).item())
            logits_shape = list(pp_logits.shape)
            local_passed = max_abs == 0.0

        device = input_ids.device
        passed = _bool_all(local_passed and not skipped, device)
        max_abs_all = _max_float(max_abs, device)
        rel_all = _max_float(rel, device)
        if rank == 0:
            print(
                json.dumps(
                    {
                        "passed": passed,
                        "world_size": world_size,
                        "pp": world_size,
                        "batch_size": args.batch_size,
                        "seq_len": args.seq_len,
                        "p2p_shape": [args.batch_size, args.seq_len, cfg.hc_mult, cfg.hidden_size],
                        "rank0_layer_ids": list(stage.layer_ids),
                        "copied_rank0": len(copied),
                        "skipped_rank0": skipped,
                        "last_logits_shape": logits_shape,
                        "max_abs": max_abs_all,
                        "rel": rel_all,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if not passed:
            raise RuntimeError("R3 PP P2P smoke failed")
    finally:
        if parallel_state.is_initialized():
            parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
