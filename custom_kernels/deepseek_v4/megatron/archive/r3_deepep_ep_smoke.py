"""R3 V4 MoE DeepEP smoke test.

Run with:
  python -m torch.distributed.run --nproc_per_node=2 \
    -m custom_kernels.deepseek_v4.megatron.r3_deepep_ep_smoke

This is a small correctness gate for the V4 MoE adapter: Megatron's
``flex/deepep`` dispatcher does token dispatch/combine across EP ranks, while
``V4GroupedExperts.forward_dispatched`` performs the local V4 clamp+SwiGLU expert
math.  Rank-local EP2 output is compared against an EP1 reference with identical
weights.
"""

from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import torch

from megatron.core import parallel_state, tensor_parallel
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.transformer.moe.moe_utils import get_default_pg_collection

from ..mcore_model import V4MoELayer
from ..model_provider import make_transformer_config


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
    return rank, world_size, local_rank


def _tiny_hf_config(args):
    return SimpleNamespace(
        initializer_range=0.02,
        num_hidden_layers=1,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        head_dim=args.hidden_size // args.num_attention_heads,
        moe_intermediate_size=args.intermediate_size,
        intermediate_size=args.intermediate_size,
        max_position_embeddings=64,
        num_local_experts=args.num_experts,
        num_experts_per_tok=args.topk,
        mlp_layer_types=[args.router_type],
        hidden_act="silu",
        swiglu_limit=10.0,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.0,
        vocab_size=128,
        mlp_bias=False,
    )


def _randn(shape, gen, scale=0.02):
    return torch.randn(shape, generator=gen, dtype=torch.float32) * scale


def _make_weights(cfg, seed: int):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    tid = torch.arange(cfg.vocab_size, dtype=torch.long).unsqueeze(1)
    offsets = torch.arange(cfg.num_experts_per_tok, dtype=torch.long).unsqueeze(0)
    weights = {
        "router": _randn((cfg.num_local_experts, cfg.hidden_size), gen),
        "router_bias": torch.linspace(-0.01, 0.01, cfg.num_local_experts, dtype=torch.float32),
        "tid2eid": (tid + offsets) % cfg.num_local_experts,
        "gate_up": _randn((cfg.num_local_experts, 2 * cfg.intermediate_size, cfg.hidden_size), gen),
        "down": _randn((cfg.num_local_experts, cfg.hidden_size, cfg.intermediate_size), gen),
        "shared_gate": _randn((cfg.intermediate_size, cfg.hidden_size), gen),
        "shared_up": _randn((cfg.intermediate_size, cfg.hidden_size), gen),
        "shared_down": _randn((cfg.hidden_size, cfg.intermediate_size), gen),
    }
    return weights


def _copy_common_weights(layer: V4MoELayer, weights: dict[str, torch.Tensor]):
    device = next(layer.parameters()).device
    with torch.no_grad():
        layer.gate.weight.copy_(weights["router"].to(device=device, dtype=layer.gate.weight.dtype))
        if hasattr(layer.gate, "e_score_correction_bias"):
            layer.gate.e_score_correction_bias.data = weights["router_bias"].to(device=device)
        if hasattr(layer.gate, "tid2eid"):
            layer.gate.tid2eid.copy_(weights["tid2eid"].to(device=device))
        layer.shared_experts.gate_proj.weight.copy_(
            weights["shared_gate"].to(device=device, dtype=layer.shared_experts.gate_proj.weight.dtype)
        )
        layer.shared_experts.up_proj.weight.copy_(
            weights["shared_up"].to(device=device, dtype=layer.shared_experts.up_proj.weight.dtype)
        )
        layer.shared_experts.down_proj.weight.copy_(
            weights["shared_down"].to(device=device, dtype=layer.shared_experts.down_proj.weight.dtype)
        )


def _copy_ep_weights(layer: V4MoELayer, weights: dict[str, torch.Tensor]):
    device = next(layer.parameters()).device
    start = layer.experts.local_expert_start
    end = layer.experts.local_expert_end
    with torch.no_grad():
        layer.experts.gate_up_proj.copy_(
            weights["gate_up"][start:end].to(device=device, dtype=layer.experts.gate_up_proj.dtype)
        )
        layer.experts.down_proj.copy_(
            weights["down"][start:end].to(device=device, dtype=layer.experts.down_proj.dtype)
        )


def _copy_full_expert_weights(layer: V4MoELayer, weights: dict[str, torch.Tensor]):
    device = next(layer.parameters()).device
    with torch.no_grad():
        layer.experts.gate_up_proj.copy_(weights["gate_up"].to(device=device, dtype=layer.experts.gate_up_proj.dtype))
        layer.experts.down_proj.copy_(weights["down"].to(device=device, dtype=layer.experts.down_proj.dtype))


def _build_ep_layer(cfg, mcore_cfg, world_size, ep_rank):
    layer = V4MoELayer(
        cfg,
        0,
        expert_model_parallel_size=world_size,
        expert_model_parallel_rank=ep_rank,
        mcore_config=mcore_cfg,
        pg_collection=get_default_pg_collection(),
    ).cuda()
    return layer.bfloat16()


def _build_ref_layer(cfg):
    layer = V4MoELayer(cfg, 0).cuda()
    return layer.bfloat16()


def _finite_param_grad(param: torch.nn.Parameter) -> bool:
    grad = getattr(param, "main_grad", None)
    if grad is None:
        grad = param.grad
    return grad is not None and torch.isfinite(grad).all().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--intermediate-size", type=int, default=64)
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--num-attention-heads", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--router-type", choices=("moe", "hash_moe"), default="moe")
    parser.add_argument("--ddp-audit", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--order", default="tp-cp-ep-dp-pp")
    parser.add_argument("--atol", type=float, default=5e-2)
    parser.add_argument("--no-require-cross-rank-route", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DeepEP smoke")
    rank, world_size, _ = _init_dist(args.order)
    if world_size < 2:
        raise RuntimeError("DeepEP smoke requires at least 2 ranks")
    if args.num_experts % world_size != 0:
        raise ValueError("--num-experts must divide WORLD_SIZE")

    try:
        cfg = _tiny_hf_config(args)
        mcore_cfg = make_transformer_config(
            cfg,
            params_dtype=torch.bfloat16,
            expert_model_parallel_size=world_size,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="deepep",
            moe_router_dtype="fp32",
        )
        ep_rank = parallel_state.get_expert_model_parallel_rank()
        weights = _make_weights(cfg, args.seed)

        ep_layer = _build_ep_layer(cfg, mcore_cfg, world_size, ep_rank)
        _copy_common_weights(ep_layer, weights)
        _copy_ep_weights(ep_layer, weights)
        forward_layer = ep_layer
        ddp_dense_buffers = -1
        ddp_expert_buffers = -1
        if args.ddp_audit:
            ddp_config = DistributedDataParallelConfig(grad_reduce_in_fp32=True)
            forward_layer = DistributedDataParallel(mcore_cfg, ddp_config, ep_layer)
            ep_layer = forward_layer.module
            ddp_dense_buffers = len(forward_layer.buffers)
            ddp_expert_buffers = len(forward_layer.expert_parallel_buffers)

        ref_layer = _build_ref_layer(cfg)
        _copy_common_weights(ref_layer, weights)
        _copy_full_expert_weights(ref_layer, weights)

        gen = torch.Generator(device="cpu").manual_seed(args.seed + 1)
        hidden_cpu = torch.randn(args.batch_size, args.seq_len, args.hidden_size, generator=gen, dtype=torch.float32)
        hidden = hidden_cpu.cuda().bfloat16().requires_grad_(True)
        ref_hidden = hidden_cpu.cuda().bfloat16().requires_grad_(True)
        input_ids = None
        if args.router_type == "hash_moe":
            input_ids = (
                torch.arange(args.batch_size * args.seq_len, device=hidden.device)
                .reshape(args.batch_size, args.seq_len)
                .remainder(cfg.vocab_size)
            )

        with torch.no_grad():
            _, _, route_indices = ep_layer.gate(hidden, input_ids)
            route_counts = torch.bincount(route_indices.reshape(-1), minlength=cfg.num_local_experts).to(torch.int64)
            counts_by_ep_rank = route_counts.view(world_size, -1).sum(dim=1)
            cross_route_ok = torch.tensor(int(bool((counts_by_ep_rank > 0).all().item())), device=hidden.device)

        out = forward_layer(hidden, input_ids=input_ids)
        ref_out = ref_layer(ref_hidden, input_ids=input_ids)
        local_tokens_per_expert = (
            ep_layer.token_dispatcher._comm_manager.get_number_of_tokens_per_expert()
            .detach()
            .to(device=hidden.device, dtype=torch.int64)
        )
        gathered_tokens_per_expert = [torch.empty_like(local_tokens_per_expert) for _ in range(world_size)]
        torch.distributed.all_gather(gathered_tokens_per_expert, local_tokens_per_expert)
        diff = (out.float() - ref_out.float()).abs()
        max_abs = diff.max()
        ref_scale = ref_out.float().abs().max().clamp_min(1e-6)
        max_rel = max_abs / ref_scale

        loss = out.float().square().mean()
        loss.backward()
        grad_ok = torch.tensor(
            int(
                hidden.grad is not None
                and torch.isfinite(hidden.grad).all().item()
                and _finite_param_grad(ep_layer.gate.weight)
                and _finite_param_grad(ep_layer.experts.gate_up_proj)
                and _finite_param_grad(ep_layer.experts.down_proj)
            ),
            device=hidden.device,
        )
        ddp_ok = torch.tensor(
            int(not args.ddp_audit or (ddp_dense_buffers > 0 and ddp_expert_buffers > 0)),
            device=hidden.device,
        )
        torch.distributed.all_reduce(max_abs, op=torch.distributed.ReduceOp.MAX)
        torch.distributed.all_reduce(max_rel, op=torch.distributed.ReduceOp.MAX)
        torch.distributed.all_reduce(grad_ok, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(cross_route_ok, op=torch.distributed.ReduceOp.MIN)
        torch.distributed.all_reduce(ddp_ok, op=torch.distributed.ReduceOp.MIN)

        require_cross_route = not args.no_require_cross_rank_route
        passed = bool(
            (max_abs <= args.atol).item()
            and grad_ok.item() == 1
            and ddp_ok.item() == 1
            and (cross_route_ok.item() == 1 or not require_cross_route)
        )
        if rank == 0:
            print(
                json.dumps(
                    {
                        "passed": passed,
                        "world_size": world_size,
                        "router_type": args.router_type,
                        "dispatcher": "flex/deepep",
                        "max_abs": float(max_abs.item()),
                        "max_rel": float(max_rel.item()),
                        "grad_ok": bool(grad_ok.item()),
                        "ddp_audit": args.ddp_audit,
                        "ddp_dense_buffers": ddp_dense_buffers,
                        "ddp_expert_buffers": ddp_expert_buffers,
                        "ddp_ok": bool(ddp_ok.item()),
                        "route_counts": route_counts.cpu().tolist(),
                        "tokens_per_expert_by_rank": [t.cpu().tolist() for t in gathered_tokens_per_expert],
                        "cross_route_ok": bool(cross_route_ok.item()),
                        "require_cross_route": require_cross_route,
                        "atol": args.atol,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if not passed:
            raise RuntimeError("V4 DeepEP EP smoke failed")
    finally:
        if parallel_state.is_initialized():
            parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
