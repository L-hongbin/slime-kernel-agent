"""Create a small Megatron ``torch_dist`` slice from the real V4-Flash checkpoint.

This is an R2 schema/materialization gate, not the full 43-layer conversion.
It builds a real-shape V4 mcore model with selected layers, loads native FP8
safetensors through ``native_checkpoint.py``, and saves a release-format
distributed checkpoint using Megatron Core dist-checkpointing.

For preflight, ``--logical-ep-size`` / ``--logical-ep-rank`` materialize only
one EP expert shard inside a single-rank process. That is not a runnable
multi-rank training checkpoint by itself; it validates per-rank payload,
global expert-id offsets, and sharded-state metadata before launching a real
PP/EP conversion job.
"""

from __future__ import annotations

import argparse
import os

import torch
from megatron.core import dist_checkpointing, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.serialization import get_default_save_sharded_strategy
from megatron.training.checkpointing import get_checkpoint_tracker_filename
from transformers import AutoConfig

from .model_provider import build_v4_mcore_model
from .native_checkpoint import (
    DEFAULT_V4_FLASH_FP8_CKPT,
    contiguous_pp_layer_ids,
    load_native_checkpoint_into_mcore_model,
)


def init_dist_1rank(master_port: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(master_port))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.cuda.set_device(0)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=1,
            rank=0,
            device_id=torch.device("cuda:0"),
        )
    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
        )
    tensor_parallel.model_parallel_cuda_manual_seed(0)


def init_dist_from_env(master_port: int, *, pp_size: int, ep_size: int, order: str):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(master_port))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=world_size,
            rank=rank,
            device_id=torch.device(f"cuda:{local_rank}"),
        )
    if not parallel_state.is_initialized():
        if world_size == 1:
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                expert_model_parallel_size=1,
            )
        else:
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=pp_size,
                expert_model_parallel_size=ep_size,
                expert_tensor_parallel_size=1,
                order=order,
            )
    tensor_parallel.model_parallel_cuda_manual_seed(0)
    return rank, world_size, local_rank


def sliced_config(checkpoint_dir: str, source_layers: list[int]):
    cfg = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
    cfg.num_hidden_layers = len(source_layers)
    cfg.layer_types = [cfg.layer_types[i] for i in source_layers]
    cfg.mlp_layer_types = [cfg.mlp_layer_types[i] for i in source_layers]
    return cfg


def save_release_torch_dist(model, save_dir: str):
    release_dir = os.path.join(save_dir, "release")
    os.makedirs(release_dir, exist_ok=True)
    metadata = {"dp_cp_group": parallel_state.get_data_parallel_group(with_context_parallel=True)}
    state_dict = {
        "checkpoint_version": 3.0,
        "iteration": 1,
        "model": model.sharded_state_dict(metadata=metadata),
    }
    dist_checkpointing.save(
        state_dict,
        release_dir,
        get_default_save_sharded_strategy("torch_dist"),
        validate_access_integrity=True,
    )
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        with open(get_checkpoint_tracker_filename(save_dir), "w", encoding="utf-8") as f:
            f.write("release")
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    return release_dir


def resolve_source_layers(args, *, pp_rank: int, world_size: int) -> list[int]:
    if args.source_layers is not None:
        return list(args.source_layers)
    if world_size == 1 or args.pp_size == 1:
        return list(range(args.num_layers))
    splits = contiguous_pp_layer_ids(
        args.num_layers,
        args.pp_size,
        num_layers_in_first_pipeline_stage=args.plan_first_layers,
        num_layers_in_last_pipeline_stage=args.plan_last_layers,
    )
    return list(splits[pp_rank])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a small V4 real-weight torch_dist slice.")
    parser.add_argument("--checkpoint", default=DEFAULT_V4_FLASH_FP8_CKPT)
    parser.add_argument("--save", required=True)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument(
        "--source-layers",
        type=int,
        nargs="+",
        default=None,
        help="Native checkpoint layer ids to load into local layers 0..N-1. Defaults to 0..num_layers-1.",
    )
    parser.add_argument(
        "--preserve-global-layer-ids",
        action="store_true",
        help=(
            "Build only --source-layers but save their ShardedTensor keys as global "
            "layers.<id> instead of the sliced local layers.0..N-1 namespace."
        ),
    )
    parser.add_argument("--master-port", type=int, default=29620)
    parser.add_argument("--pp-size", type=int, default=1, help="Actual PP size for torchrun conversion.")
    parser.add_argument("--ep-size", type=int, default=1, help="Actual EP size for torchrun conversion.")
    parser.add_argument("--parallel-order", default="tp-cp-ep-dp-pp")
    parser.add_argument("--plan-first-layers", type=int, default=None)
    parser.add_argument("--plan-last-layers", type=int, default=None)
    parser.add_argument(
        "--logical-ep-size",
        type=int,
        default=1,
        help=(
            "Materialize only one logical EP expert shard inside this 1-rank "
            "preflight process. This does not initialize a real EP process group."
        ),
    )
    parser.add_argument(
        "--logical-ep-rank",
        type=int,
        default=0,
        help="Logical EP rank to materialize when --logical-ep-size > 1.",
    )
    args = parser.parse_args(argv)

    rank, world_size, _ = init_dist_from_env(
        args.master_port,
        pp_size=args.pp_size,
        ep_size=args.ep_size,
        order=args.parallel_order,
    )
    if world_size > 1 and world_size != args.pp_size * args.ep_size:
        raise ValueError(
            f"torchrun WORLD_SIZE={world_size} must equal pp_size*ep_size="
            f"{args.pp_size * args.ep_size} for this R2 conversion preflight"
        )
    pp_rank = parallel_state.get_pipeline_model_parallel_rank() if world_size > 1 else 0
    ep_size = parallel_state.get_expert_model_parallel_world_size() if world_size > 1 else args.logical_ep_size
    ep_rank = parallel_state.get_expert_model_parallel_rank() if world_size > 1 else args.logical_ep_rank

    source_layers = resolve_source_layers(args, pp_rank=pp_rank, world_size=world_size)
    if not source_layers:
        raise ValueError("--num-layers must be >= 1")
    if args.source_layers is not None and args.num_layers != 1 and args.num_layers != len(source_layers):
        raise ValueError("--num-layers must be omitted/1 or equal to len(--source-layers)")
    layer_map = {source_layer: target_layer for target_layer, source_layer in enumerate(source_layers)}
    try:
        if args.preserve_global_layer_ids or world_size > 1:
            cfg = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=True)
            layer_ids = source_layers
        else:
            cfg = sliced_config(args.checkpoint, source_layers)
            layer_ids = None
        pre_process = args.pp_size == 1 or pp_rank == 0
        post_process = args.pp_size == 1 or pp_rank == args.pp_size - 1
        model = build_v4_mcore_model(
            cfg,
            pre_process=pre_process,
            post_process=post_process,
            params_dtype=torch.bfloat16,
            init_weights=False,
            layer_ids=layer_ids,
            expert_model_parallel_size=ep_size,
            expert_model_parallel_rank=ep_rank,
            pipeline_model_parallel_size=args.pp_size,
        )
        model = model.cuda().bfloat16()
        stats = load_native_checkpoint_into_mcore_model(model, args.checkpoint, layer_map=layer_map, strict=True)
        print(
            f"rank={rank} world={world_size} pp={pp_rank}/{args.pp_size} "
            f"ep={ep_rank}/{ep_size} source layers: {source_layers} -> local layers: {layer_map}",
            flush=True,
        )
        print(f"rank={rank} loaded native tensors: {stats}", flush=True)
        release_dir = save_release_torch_dist(model, args.save)
        if rank == 0:
            print(f"saved torch_dist release: {release_dir}", flush=True)
    finally:
        if parallel_state.is_initialized():
            parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
