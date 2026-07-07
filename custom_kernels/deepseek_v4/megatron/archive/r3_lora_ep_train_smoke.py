"""R3.5 EP+DeepEP+LoRA+optimizer train smoke.

This is the bridge between the isolated R3 DeepEP model smoke and a full slime
launcher run. It builds a tiny V4LanguageModel twice on each rank:

* an EP1 reference model with all experts local;
* an EP=world_size model with Megatron ``flex/deepep`` token dispatch.

The EP model copies the exact local expert slices from the EP1 reference, applies
V4 LoRA, wraps Megatron DDP, then runs one optimizer step (Muon by default). The
gate is intentionally PP1/single-node; PP>1 actor transport is still separate.
"""

from __future__ import annotations

import argparse
import json
import os

import torch
import torch.nn.functional as F

from megatron.core import parallel_state, tensor_parallel
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.muon import get_megatron_muon_optimizer

from ..lora import apply_v4_lora, audit_lora
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


def _named_tensors(module):
    tensors = {name: param for name, param in module.named_parameters()}
    tensors.update({name: buf for name, buf in module.named_buffers()})
    return tensors


@torch.no_grad()
def _copy_ep_slice_from_full(ep_model, full_model):
    """Copy matching tensors plus expert-axis slices from an EP1 full model."""
    full = _named_tensors(full_model)
    copied = []
    sliced = []
    for name, dest in _named_tensors(ep_model).items():
        if name not in full:
            continue
        src = full[name]
        if tuple(src.shape) == tuple(dest.shape):
            dest.copy_(src.to(device=dest.device, dtype=dest.dtype))
            copied.append(name)
            continue
        if name.endswith("mlp.experts.gate_up_proj") or name.endswith("mlp.experts.down_proj"):
            expert_module_name = name.rsplit(".", 1)[0]
            expert_module = dict(ep_model.named_modules())[expert_module_name]
            start = expert_module.local_expert_start
            end = expert_module.local_expert_end
            dest.copy_(src[start:end].to(device=dest.device, dtype=dest.dtype))
            sliced.append(name)
    return copied, sliced


def _ce_loss(logits, input_ids):
    return F.cross_entropy(
        logits[:, :-1, :].float().reshape(-1, logits.shape[-1]),
        input_ids[:, 1:].reshape(-1),
    )


def _has_finite_trainable_grad(model) -> bool:
    ok = False
    for _name, param in model.named_parameters():
        if param.requires_grad and _finite_param_grad(param):
            ok = True
    return ok


def _expert_allreduce_false(model) -> bool:
    found = False
    for name, param in model.named_parameters():
        if ".mlp.experts." not in name:
            continue
        found = True
        if getattr(param, "allreduce", None) is not False:
            return False
    return found


def _expert_params_frozen(model) -> bool:
    found = False
    for name, param in model.named_parameters():
        if ".mlp.experts." not in name:
            continue
        found = True
        if param.requires_grad:
            return False
    return found


def _mhc_fp32_ok(model) -> bool:
    modules = []
    for layer in model.layers:
        modules.extend([layer.attn_hc, layer.ffn_hc])
    if model.post_process:
        modules.append(model.hc_head)
    for module in modules:
        for param in module.parameters(recurse=True):
            if param.dtype != torch.float32:
                return False
    return True


def _sharded_expert_state_ok(model, world_size: int) -> bool:
    state = model.sharded_state_dict()
    expert_values = [
        value
        for key, value in state.items()
        if ".mlp.experts." in key and (key.endswith("gate_up_proj") or key.endswith("down_proj"))
    ]
    if not expert_values:
        return False
    expected_local = model.hf_config.num_local_experts // world_size
    for value in expert_values:
        if value.global_shape[0] != model.hf_config.num_local_experts:
            return False
        if value.local_shape[0] != expected_local:
            return False
        if value.axis_fragmentations[0] != world_size:
            return False
    return True


def _trainable_snapshot(model):
    return {name: param.detach().float().cpu() for name, param in model.named_parameters() if param.requires_grad}


def _trainable_delta(model, before) -> float:
    total = 0.0
    for name, param in model.named_parameters():
        if not param.requires_grad or name not in before:
            continue
        total += float((param.detach().float().cpu() - before[name]).abs().sum().item())
    return total


def _build_optimizer(kind: str, model, lr: float):
    opt_cfg = OptimizerConfig(
        optimizer=kind,
        lr=lr,
        min_lr=lr,
        weight_decay=0.0,
        clip_grad=1.0,
        bf16=True,
        fp16=False,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=False,
    )
    if "muon" in kind:
        return get_megatron_muon_optimizer(
            config=opt_cfg,
            model_chunks=[model],
            use_gloo_process_groups=True,
            layer_wise_distributed_optimizer=False,
        )
    return get_megatron_optimizer(
        config=opt_cfg,
        model_chunks=[model],
        use_gloo_process_groups=True,
    )


def _bool_all(value: bool, device) -> bool:
    tensor = torch.tensor(int(value), device=device, dtype=torch.int32)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MIN)
    return bool(tensor.item())


def _max_float(value: float, device) -> float:
    tensor = torch.tensor(float(value), device=device, dtype=torch.float32)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX)
    return float(tensor.item())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--lora-dim", type=int, default=4)
    parser.add_argument("--lora-alpha", type=int, default=8)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--optimizer", choices=("muon", "adam"), default="muon")
    parser.add_argument("--order", default="tp-cp-ep-dp-pp")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for R3.5 LoRA EP train smoke")
    rank, world_size = _init_dist(args.order)
    if world_size < 2:
        raise RuntimeError("R3.5 LoRA EP train smoke requires at least 2 ranks")

    try:
        cfg = tiny_config()
        ep_rank = parallel_state.get_expert_model_parallel_rank()
        torch.manual_seed(args.seed)
        full_model = build_v4_mcore_model(
            cfg,
            params_dtype=torch.bfloat16,
            expert_model_parallel_size=1,
            expert_model_parallel_rank=0,
            moe_router_dtype="fp32",
        ).cuda()
        full_model = full_model.bfloat16()
        full_model.restore_fp32_modules()

        torch.manual_seed(args.seed + 1)
        ep_model = build_v4_mcore_model(
            cfg,
            params_dtype=torch.bfloat16,
            expert_model_parallel_size=world_size,
            expert_model_parallel_rank=ep_rank,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="deepep",
            moe_router_dtype="fp32",
        ).cuda()
        ep_model = ep_model.bfloat16()
        ep_model.restore_fp32_modules()
        copied, sliced = _copy_ep_slice_from_full(ep_model, full_model)
        expert_allreduce_false = _expert_allreduce_false(ep_model)
        sharded_expert_state_ok = _sharded_expert_state_ok(ep_model, world_size)

        torch.manual_seed(args.seed + 2)
        full_model = apply_v4_lora(
            full_model,
            dim=args.lora_dim,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        torch.manual_seed(args.seed + 2)
        ep_model = apply_v4_lora(
            ep_model,
            dim=args.lora_dim,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
        )
        full_model = full_model.bfloat16()
        full_model.restore_fp32_modules()
        ep_model = ep_model.bfloat16()
        ep_model.restore_fp32_modules()
        lora_audit = audit_lora(ep_model)
        expert_params_frozen = _expert_params_frozen(ep_model)
        mhc_fp32_ok = _mhc_fp32_ok(ep_model)

        ddp_config = DistributedDataParallelConfig(grad_reduce_in_fp32=True)
        ddp_model = DistributedDataParallel(ep_model.config, ddp_config, ep_model)
        optimizer = _build_optimizer(args.optimizer, ddp_model, args.lr)

        gen = torch.Generator(device="cpu").manual_seed(args.seed + 3)
        input_ids = torch.randint(
            0,
            cfg.vocab_size,
            (args.batch_size, args.seq_len),
            generator=gen,
            dtype=torch.long,
        ).cuda()

        with torch.no_grad():
            ref_logits = full_model(input_ids=input_ids)
            ep_logits = ddp_model(input_ids=input_ids)
            ref_loss = _ce_loss(ref_logits, input_ids)
            ep_loss = _ce_loss(ep_logits, input_ids)
            max_ref_diff = (ep_logits.float() - ref_logits.float()).abs().max()

        ddp_model.zero_grad_buffer()
        optimizer.zero_grad()
        before = _trainable_snapshot(ddp_model.module)
        train_logits = ddp_model(input_ids=input_ids)
        loss = _ce_loss(train_logits, input_ids)
        loss.backward()
        grad_ok_local = _has_finite_trainable_grad(ddp_model.module)
        found_inf = optimizer.prepare_grads()
        found_inf_bool = bool(found_inf.item()) if isinstance(found_inf, torch.Tensor) else bool(found_inf)
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()
        param_delta_local = _trainable_delta(ddp_model.module, before)

        with torch.no_grad():
            post_loss = _ce_loss(ddp_model(input_ids=input_ids), input_ids)

        dense_buffers = len(ddp_model.buffers)
        expert_buffers = len(ddp_model.expert_parallel_buffers)
        device = input_ids.device
        max_ref_diff_all = _max_float(float(max_ref_diff.item()), device)
        loss_diff_all = _max_float(float((ep_loss - ref_loss).abs().item()), device)
        param_delta_all = _max_float(param_delta_local, device)
        finite_loss_all = _bool_all(
            torch.isfinite(ref_loss).item()
            and torch.isfinite(ep_loss).item()
            and torch.isfinite(loss).item()
            and torch.isfinite(post_loss).item(),
            device,
        )
        grad_ok_all = _bool_all(grad_ok_local, device)
        lora_ok = (
            bool(lora_audit["wrapped"]) and not lora_audit["o_a_wrapped"] and lora_audit["trainable_are_lora_only"]
        )
        lora_ok_all = _bool_all(lora_ok, device)
        expert_audit_ok_all = _bool_all(
            expert_allreduce_false and expert_params_frozen and sharded_expert_state_ok and mhc_fp32_ok,
            device,
        )
        buffers_ok_all = _bool_all(dense_buffers > 0, device)
        update_ok_all = _bool_all(update_successful and not found_inf_bool, device)

        passed = bool(
            finite_loss_all
            and grad_ok_all
            and lora_ok_all
            and expert_audit_ok_all
            and buffers_ok_all
            and update_ok_all
            and max_ref_diff_all == 0.0
            and loss_diff_all == 0.0
            and param_delta_all > 0.0
        )
        if rank == 0:
            print(
                json.dumps(
                    {
                        "passed": passed,
                        "world_size": world_size,
                        "dispatcher": "flex/deepep",
                        "optimizer": args.optimizer,
                        "lora_dim": args.lora_dim,
                        "batch_size": args.batch_size,
                        "seq_len": args.seq_len,
                        "copied_tensors": len(copied),
                        "sliced_expert_tensors": len(sliced),
                        "lora_wrapped": len(lora_audit["wrapped"]),
                        "lora_o_a_wrapped": lora_audit["o_a_wrapped"],
                        "trainable_are_lora_only": lora_audit["trainable_are_lora_only"],
                        "expert_allreduce_false": expert_allreduce_false,
                        "expert_params_frozen": expert_params_frozen,
                        "sharded_expert_state_ok": sharded_expert_state_ok,
                        "mhc_fp32_ok": mhc_fp32_ok,
                        "n_trainable": lora_audit["n_trainable"],
                        "n_frozen": lora_audit["n_frozen"],
                        "ref_loss": float(ref_loss.item()),
                        "ep_loss": float(ep_loss.item()),
                        "train_loss": float(loss.item()),
                        "post_loss": float(post_loss.item()),
                        "loss_diff": loss_diff_all,
                        "max_ref_diff": max_ref_diff_all,
                        "grad_ok": grad_ok_all,
                        "found_inf": found_inf_bool,
                        "update_successful": bool(update_successful),
                        "grad_norm": (
                            float(grad_norm.item()) if isinstance(grad_norm, torch.Tensor) else float(grad_norm)
                        ),
                        "num_zeros_in_grad": None if num_zeros_in_grad is None else int(num_zeros_in_grad),
                        "param_delta": param_delta_all,
                        "ddp_dense_buffers": dense_buffers,
                        "ddp_expert_buffers": expert_buffers,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if not passed:
            raise RuntimeError("R3.5 LoRA EP train smoke failed")
    finally:
        if parallel_state.is_initialized():
            parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
