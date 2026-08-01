#!/usr/bin/env python3
"""Compare a frozen Megatron forward with captured SGLang rollout logits.

The input is the raw ``.pt`` emitted by ``dspark_full_vocab_probe.py``.  This
program rebuilds the same token prefixes with the native DeepSeek-V4 Megatron
implementation, loads the official checkpoint directly, and compares paired
full-vocabulary rows.  It is a diagnostic forward only: no optimizer or
checkpoint write is involved.  Optional zero-initialized LoRA wrapping permits
testing the production FP8 base-weight paths without changing model outputs.

Launch one process per EP rank, for example::

    torchrun --standalone --nproc-per-node 8 \
      scripts/dsv4/diagnostics/mismatch/train_rollout_full_vocab_probe.py \
      --checkpoint /nfs/FM/checkpoints/DeepSeek-V4-Flash \
      --sglang-raw /nfs/FM/probe/full_vocab.pt \
      --output /nfs/FM/probe/train_rollout_full_vocab.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import types
from pathlib import Path
from typing import Any

# torchrun ranks must not share TileLang's temporary cache directory.  The
# cache implementation may recreate that directory during compilation.
os.environ.setdefault(
    "TILELANG_CACHE_DIR",
    f"/tmp/tilelang_cache_rank{os.environ.get('LOCAL_RANK', '0')}",
)

_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_MEGATRON_ROOT = Path(os.environ.get("MEGATRON_LM_PATH", "/root/Megatron-LM")).resolve()
if not _MEGATRON_ROOT.is_dir():
    raise RuntimeError(f"Megatron-LM checkout not found: {_MEGATRON_ROOT}")
if str(_MEGATRON_ROOT) not in sys.path:
    sys.path.insert(0, str(_MEGATRON_ROOT))

import numpy as np
import torch
import torch.distributed as dist
from custom_kernels.deepseek_v4.megatron.model_provider import build_v4_mcore_model
from custom_kernels.deepseek_v4.megatron.native_checkpoint import load_native_checkpoint_into_mcore_model
from custom_kernels.deepseek_v4.megatron.slice_torch_dist import init_dist_from_env
from transformers import AutoConfig


def _summary(values: torch.Tensor) -> dict[str, float | int]:
    x = values.detach().double().cpu().numpy()
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "mean_abs": float(np.abs(x).mean()),
        "positive": int((x > 0).sum()),
        "negative": int((x < 0).sum()),
        "zero": int((x == 0).sum()),
        "p01": float(np.quantile(x, 0.01)),
        "p10": float(np.quantile(x, 0.10)),
        "p90": float(np.quantile(x, 0.90)),
        "p99": float(np.quantile(x, 0.99)),
        "max_abs": float(np.abs(x).max()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_record(path: Path, record_index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    blob = torch.load(path, map_location="cpu", weights_only=True)
    records = blob.get("records") if isinstance(blob, dict) else None
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} has no records list")
    if record_index < 0 or record_index >= len(records):
        raise IndexError(f"record index {record_index} outside [0, {len(records)})")
    record = records[record_index]
    required = {"prompt_ids", "response_ids", "generation_logits", "scorer_logits"}
    missing = required - set(record)
    if missing:
        raise ValueError(f"record {record_index} is missing {sorted(missing)}")
    generation = record["generation_logits"]
    scorer = record["scorer_logits"]
    if scorer.ndim == 3:
        if scorer.shape[0] != 1:
            raise ValueError(f"expected one scorer repeat, got shape {tuple(scorer.shape)}")
        scorer = scorer[0]
    response_ids = [int(value) for value in record["response_ids"]]
    if generation.shape != scorer.shape or generation.shape[0] != len(response_ids):
        raise ValueError(
            f"SGLang row mismatch: generation={tuple(generation.shape)} "
            f"scorer={tuple(scorer.shape)} response={len(response_ids)}"
        )
    metadata = {
        "prompt_index": int(record.get("prompt_index", record_index)),
        "prompt": record.get("prompt"),
        "prompt_ids": [int(value) for value in record["prompt_ids"]],
        "response_ids": response_ids,
    }
    routed_experts = record.get("routed_experts")
    if routed_experts is not None:
        if not isinstance(routed_experts, torch.Tensor) or routed_experts.ndim != 3:
            raise ValueError(
                f"record {record_index} routed_experts must be a rank-3 tensor, "
                f"got {type(routed_experts).__name__}"
            )
        metadata["routed_experts"] = routed_experts.int().contiguous()
    tensors = {
        "generation_logits": generation.float().contiguous(),
        "scorer_logits": scorer.float().contiguous(),
    }
    return metadata, tensors


def _paired_metrics(
    train_logits: torch.Tensor,
    rollout_logits: torch.Tensor,
    response_ids: torch.Tensor,
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if train_logits.shape != rollout_logits.shape:
        raise ValueError(
            f"paired logits shape mismatch: train={tuple(train_logits.shape)} "
            f"rollout={tuple(rollout_logits.shape)}"
        )
    train_logp = torch.log_softmax(train_logits.float(), dim=-1)
    rollout_logp = torch.log_softmax(rollout_logits.float(), dim=-1)
    train_p = train_logp.exp()
    rollout_p = rollout_logp.exp()
    rows = torch.arange(response_ids.numel(), device=train_logits.device)
    train_token_logp = train_logp[rows, response_ids]
    rollout_token_logp = rollout_logp[rows, response_ids]
    train_token_p = train_p[rows, response_ids]
    rollout_token_p = rollout_p[rows, response_ids]

    token_logp_delta = train_token_logp - rollout_token_logp
    token_p_delta = train_token_p - rollout_token_p
    token_ratio = token_logp_delta.exp()
    train_entropy = -(train_p * train_logp).sum(dim=-1)
    rollout_entropy = -(rollout_p * rollout_logp).sum(dim=-1)
    entropy_rollout_minus_train = rollout_entropy - train_entropy
    tv = 0.5 * (train_p - rollout_p).abs().sum(dim=-1)
    train_top_prob, train_top_id = train_p.max(dim=-1)
    rollout_top_prob, rollout_top_id = rollout_p.max(dim=-1)

    summary = {
        "returned_token_logprob_train_minus_rollout": _summary(token_logp_delta),
        "returned_token_raw_probability_train_minus_rollout": _summary(token_p_delta),
        "returned_token_probability_ratio_train_over_rollout": _summary(token_ratio),
        "returned_token_probability_ratio_minus_one": _summary(token_ratio - 1),
        "entropy_rollout_minus_train": _summary(entropy_rollout_minus_train),
        "total_variation": _summary(tv),
        "top1_probability_train_minus_rollout": _summary(train_top_prob - rollout_top_prob),
        "top1_id_agreement_fraction": float((train_top_id == rollout_top_id).float().mean()),
        "logit_train_minus_rollout": _summary(train_logits.float() - rollout_logits.float()),
    }
    per_row = {
        "train_token_logp": train_token_logp,
        "rollout_token_logp": rollout_token_logp,
        "train_token_probability": train_token_p,
        "rollout_token_probability": rollout_token_p,
        "train_entropy": train_entropy,
        "rollout_entropy": rollout_entropy,
        "tv": tv,
        "train_top_probability": train_top_prob,
        "rollout_top_probability": rollout_top_prob,
        "train_top_id": train_top_id,
        "rollout_top_id": rollout_top_id,
    }
    return summary, per_row


def _cross_rank_signature(logits: torch.Tensor, response_ids: torch.Tensor) -> list[list[float]]:
    rows = torch.arange(response_ids.numel(), device=logits.device)
    values = logits.float()
    logp = torch.log_softmax(values, dim=-1)[rows, response_ids]
    signature = torch.stack(
        (
            values.sum(dtype=torch.float64),
            values.square().sum(dtype=torch.float64),
            logp.sum(dtype=torch.float64),
            logp.square().sum(dtype=torch.float64),
        )
    )
    gathered = [torch.empty_like(signature) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, signature)
    return [[float(value) for value in item.cpu()] for item in gathered]


def _routing_alignment(model: torch.nn.Module, rollout_routes: torch.Tensor) -> dict[str, Any]:
    """Compare naturally selected learned-MoE expert sets with rollout routes."""

    from slime.utils.routing_replay import get_rollout_routing_replay_for_layer

    per_layer = []
    exact_rows = 0
    total_rows = 0
    overlap_ids = 0
    total_ids = 0
    for layer_id in range(rollout_routes.shape[1]):
        replay = get_rollout_routing_replay_for_layer(model, layer_id)
        if replay is None:
            continue
        if len(replay.top_indices_list) != 1:
            raise RuntimeError(f"layer {layer_id} recorded {len(replay.top_indices_list)} routing tensors, expected 1")
        trainer = replay.top_indices_list[0].long().cpu()
        rollout = rollout_routes[:, layer_id].long().cpu()
        if trainer.shape != rollout.shape:
            raise RuntimeError(
                f"layer {layer_id} routing shape differs: trainer={tuple(trainer.shape)} "
                f"rollout={tuple(rollout.shape)}"
            )
        exact = torch.sort(trainer, dim=-1).values.eq(torch.sort(rollout, dim=-1).values).all(dim=-1)
        overlap = trainer.unsqueeze(-1).eq(rollout.unsqueeze(-2)).any(dim=-1).sum()
        layer_exact = int(exact.sum())
        layer_rows = int(exact.numel())
        layer_overlap = int(overlap)
        layer_ids = int(trainer.numel())
        exact_rows += layer_exact
        total_rows += layer_rows
        overlap_ids += layer_overlap
        total_ids += layer_ids
        per_layer.append(
            {
                "layer_id": layer_id,
                "exact_set_row_fraction": layer_exact / layer_rows,
                "expert_id_overlap_fraction": layer_overlap / layer_ids,
            }
        )
    return {
        "learned_moe_layers": len(per_layer),
        "exact_set_row_fraction": exact_rows / total_rows,
        "expert_id_overlap_fraction": overlap_ids / total_ids,
        "first_layer_with_nonexact_set": next(
            (row["layer_id"] for row in per_layer if row["exact_set_row_fraction"] < 1.0),
            None,
        ),
        "per_layer": per_layer,
    }


def _apply_precision_alignment(
    model: torch.nn.Module,
    checkpoint_config: Any,
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Optionally reproduce the production LoRA + frozen-base FP8 paths."""

    any_fp8 = any(
        (
            args.fp8_shared_expert,
            args.fp8_attention,
            args.checkpoint_fp8_shared_expert,
            args.checkpoint_fp8_attention,
        )
    )
    if any_fp8 and not args.apply_zero_lora:
        raise ValueError("FP8 shared/attention probes require --apply-zero-lora")
    if args.fp8_shared_expert and args.checkpoint_fp8_shared_expert:
        raise ValueError("choose requantized or checkpoint-native shared-expert FP8")
    if args.fp8_attention and args.checkpoint_fp8_attention:
        raise ValueError("choose requantized or checkpoint-native attention FP8")

    stats = {
        "zero_lora": False,
        "shared_experts_fp8": 0,
        "attention_adapters_fp8": 0,
        "checkpoint_fp8_shared_experts": 0,
        "checkpoint_fp8_attention_adapters": 0,
    }
    if not args.apply_zero_lora:
        return model, stats

    os.environ["V4_LORA_SHARED_EXPERT"] = "1"
    from custom_kernels.deepseek_v4.megatron.lora import apply_v4_lora, audit_lora

    model = apply_v4_lora(
        model,
        dim=args.zero_lora_dim,
        alpha=args.zero_lora_alpha,
        dropout=0.0,
    ).eval()
    lora_audit = audit_lora(model)
    stats["zero_lora"] = True
    stats["lora_wrapped_modules"] = len(lora_audit["wrapped"])
    stats["lora_trainable_params"] = lora_audit["n_trainable"]

    if args.fp8_shared_expert:
        for module in model.modules():
            if type(module).__name__ == "V4SharedExpertMLP":
                module.quantize_fp8()
                stats["shared_experts_fp8"] += 1

    if args.fp8_attention:
        from custom_kernels.deepseek_v4.megatron.mcore_model import (
            fp8_weight_recipe_from_config,
            quantize_lora_adapter_fp8,
        )

        block_size, use_weight_ue8m0 = fp8_weight_recipe_from_config(checkpoint_config)
        targets = ("q_a_proj", "q_b_proj", "kv_proj", "o_b_proj")
        for name, module in model.named_modules():
            if (
                any(name.endswith("self_attn." + target) for target in targets)
                and hasattr(module, "linear_in")
                and hasattr(module, "weight")
            ):
                quantize_lora_adapter_fp8(
                    module,
                    block_size,
                    use_ue8m0=use_weight_ue8m0,
                )
                stats["attention_adapters_fp8"] += 1

    if args.checkpoint_fp8_shared_expert or args.checkpoint_fp8_attention:
        from custom_kernels.deepseek_v4.megatron.mcore_model import (
            _fp8_lora_adapter_forward,
            fp8_weight_recipe_from_config,
        )
        from custom_kernels.deepseek_v4.megatron.native_checkpoint import NativeV4Checkpoint, fp8_scale_key

        block_size, _ = fp8_weight_recipe_from_config(checkpoint_config)
        native = NativeV4Checkpoint(str(args.checkpoint))

        def install(adapter: torch.nn.Module, native_key: str) -> None:
            weight = native.get_tensor(native_key)
            scale = native.get_tensor(fp8_scale_key(native_key))
            if weight.dtype != torch.float8_e4m3fn or scale.dtype != torch.float8_e8m0fnu:
                raise TypeError(f"{native_key} expected E4M3/E8M0, got {weight.dtype}/{scale.dtype}")
            if tuple(weight.shape) != tuple(adapter.weight.shape):
                raise ValueError(f"{native_key} shape {tuple(weight.shape)} != adapter {tuple(adapter.weight.shape)}")
            del adapter.weight
            adapter.register_buffer("weight_fp8", weight.cuda().contiguous(), persistent=False)
            # DeepGEMM's ue8m0 mode represents powers-of-two scales as FP32
            # values; the checkpoint stores the same values as E8M0 bytes.
            adapter.register_buffer(
                "weight_scale",
                scale.float().cuda().contiguous(),
                persistent=False,
            )
            adapter._fp8 = True
            adapter._FP8_BLOCK = block_size
            adapter.forward = types.MethodType(_fp8_lora_adapter_forward, adapter)

        if args.checkpoint_fp8_shared_expert:
            shared_leaves = {
                "gate_proj": "w1",
                "up_proj": "w3",
                "down_proj": "w2",
            }
            for name, module in model.named_modules():
                match = re.fullmatch(r"layers\.(\d+)\.mlp\.shared_experts", name)
                if not match:
                    continue
                layer_id = int(match.group(1))
                for attr, native_leaf in shared_leaves.items():
                    install(
                        getattr(module, attr),
                        f"layers.{layer_id}.ffn.shared_experts.{native_leaf}.weight",
                    )
                module._FP8_BLOCK = block_size
                module._fp8 = True
                stats["checkpoint_fp8_shared_experts"] += 1

        if args.checkpoint_fp8_attention:
            attention_leaves = {
                "q_a_proj": "wq_a",
                "q_b_proj": "wq_b",
                "kv_proj": "wkv",
                "o_b_proj": "wo_b",
            }
            for name, module in model.named_modules():
                match = re.fullmatch(
                    r"layers\.(\d+)\.self_attn\.(q_a_proj|q_b_proj|kv_proj|o_b_proj)",
                    name,
                )
                if not match:
                    continue
                layer_id = int(match.group(1))
                install(
                    module,
                    f"layers.{layer_id}.attn.{attention_leaves[match.group(2)]}.weight",
                )
                stats["checkpoint_fp8_attention_adapters"] += 1

    if args.fp8_shared_expert and stats["shared_experts_fp8"] == 0:
        raise RuntimeError("no V4SharedExpertMLP modules were FP8-quantized")
    if args.fp8_attention and stats["attention_adapters_fp8"] == 0:
        raise RuntimeError("no LoRA-wrapped attention modules were FP8-quantized")
    if args.checkpoint_fp8_shared_expert and stats["checkpoint_fp8_shared_experts"] == 0:
        raise RuntimeError("no shared experts received checkpoint-native FP8 buffers")
    if args.checkpoint_fp8_attention and stats["checkpoint_fp8_attention_adapters"] == 0:
        raise RuntimeError("no attention adapters received checkpoint-native FP8 buffers")
    torch.cuda.empty_cache()
    return model, stats


def run(args: argparse.Namespace) -> None:
    if args.routing_replay and args.capture_natural_routing:
        raise ValueError("--routing-replay and --capture-natural-routing are mutually exclusive")
    if args.routing_replay or args.capture_natural_routing:
        os.environ["ENABLE_ROUTING_REPLAY"] = "1"
        os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward" if args.routing_replay else "record"
    else:
        os.environ.setdefault("ENABLE_ROUTING_REPLAY", "0")

    rank, world_size, local_rank = init_dist_from_env(
        args.master_port,
        pp_size=1,
        ep_size=args.ep_size,
        order="tp-cp-ep-dp-pp",
    )
    if world_size != args.ep_size:
        raise ValueError(f"WORLD_SIZE={world_size} must equal --ep-size={args.ep_size}")

    metadata_records: list[dict[str, Any]] | None = None
    sglang_records: list[dict[str, torch.Tensor]] | None = None
    if rank == 0:
        loaded = [_load_record(args.sglang_raw, index) for index in args.record_indices]
        metadata_records = [item[0] for item in loaded]
        sglang_records = [item[1] for item in loaded]
        transport = metadata_records
    else:
        transport = None
    objects = [transport]
    dist.broadcast_object_list(objects, src=0)
    metadata_records = objects[0]
    if not isinstance(metadata_records, list) or not metadata_records:
        raise RuntimeError("rank 0 did not broadcast probe metadata")

    torch.cuda.set_device(local_rank)
    config = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=True)
    model = build_v4_mcore_model(
        config,
        params_dtype=torch.bfloat16,
        expert_model_parallel_size=args.ep_size,
        expert_model_parallel_rank=rank,
        moe_token_dispatcher_type=args.moe_dispatcher,
        moe_flex_dispatcher_backend=args.moe_flex_backend,
        moe_router_dtype="fp32",
        moe_deepep_num_sms=args.moe_deepep_num_sms,
    )
    model = model.cuda().bfloat16().eval()
    load_stats = load_native_checkpoint_into_mcore_model(
        model,
        str(args.checkpoint),
        layer_map={index: index for index in range(config.num_hidden_layers)},
        strict=True,
    )
    model, precision_alignment = _apply_precision_alignment(model, config, args)
    dist.barrier()

    torch.cuda.reset_peak_memory_stats()
    output_records: list[dict[str, Any]] = []
    raw_records: list[dict[str, Any]] = []
    aggregate_train: list[torch.Tensor] = []
    aggregate_generation: list[torch.Tensor] = []
    aggregate_scorer: list[torch.Tensor] = []
    aggregate_response_ids: list[torch.Tensor] = []

    for slot, metadata in enumerate(metadata_records):
        prompt_ids = metadata["prompt_ids"]
        response_ids_list = metadata["response_ids"]
        model_input_ids = (prompt_ids + response_ids_list)[:-1]
        response_start = len(prompt_ids) - 1
        response_count = len(response_ids_list)
        input_ids = torch.tensor(model_input_ids, dtype=torch.long, device="cuda").unsqueeze(0)
        response_ids = torch.tensor(response_ids_list, dtype=torch.long, device="cuda")

        if args.routing_replay or args.capture_natural_routing:
            routed_experts = metadata.get("routed_experts")
            if not isinstance(routed_experts, torch.Tensor):
                raise ValueError(f"record {args.record_indices[slot]} has no routed_experts")
            if routed_experts.shape[0] != len(model_input_ids):
                raise ValueError(
                    f"routing rows {routed_experts.shape[0]} != trainer input rows " f"{len(model_input_ids)}"
                )
            if routed_experts.shape[1] != config.num_hidden_layers:
                raise ValueError(
                    f"routing layers {routed_experts.shape[1]} != model layers " f"{config.num_hidden_layers}"
                )
            from slime.utils.routing_replay import RoutingReplay, record_rollout_routing_replay_for_layer

            RoutingReplay.clear_all()
            if args.routing_replay:
                replay_offset = 0
                for layer_id in range(config.num_hidden_layers):
                    replay_offset = record_rollout_routing_replay_for_layer(
                        model,
                        layer_id,
                        routed_experts[:, layer_id],
                        replay_offset,
                    )

        with torch.no_grad():
            logits = model(input_ids)
        if args.routing_replay:
            RoutingReplay.check_fully_consumed(context=f"record {args.record_indices[slot]}", model_modules=model)
        routing_alignment_by_rank = None
        if args.capture_natural_routing:
            local_alignment = _routing_alignment(model, routed_experts)
            routing_alignment_by_rank = [None for _ in range(world_size)]
            dist.all_gather_object(routing_alignment_by_rank, local_alignment)
        train_rows = logits[0, response_start : response_start + response_count].contiguous()
        if train_rows.shape[0] != response_count:
            raise RuntimeError(f"trainer returned {train_rows.shape[0]} response rows, expected {response_count}")
        signatures = _cross_rank_signature(train_rows, response_ids)

        if rank == 0:
            assert sglang_records is not None
            sglang_tensors = sglang_records[slot]
            generation = sglang_tensors["generation_logits"].to(device="cuda")
            scorer = sglang_tensors["scorer_logits"].to(device="cuda")
            generation_summary, generation_rows = _paired_metrics(train_rows, generation, response_ids)
            scorer_summary, scorer_rows = _paired_metrics(train_rows, scorer, response_ids)
            sglang_internal_summary, _ = _paired_metrics(generation, scorer, response_ids)
            output_records.append(
                {
                    "record_index": args.record_indices[slot],
                    "prompt_index": metadata["prompt_index"],
                    "prompt_tokens": len(prompt_ids),
                    "response_tokens": response_count,
                    "trainer_input_tokens": len(model_input_ids),
                    "cross_rank_signatures": signatures,
                    "natural_routing_alignment_by_rank": routing_alignment_by_rank,
                    "train_vs_sglang_generation": generation_summary,
                    "train_vs_sglang_full_prefix": scorer_summary,
                    "sglang_generation_vs_full_prefix_control": sglang_internal_summary,
                }
            )
            raw_records.append(
                {
                    "record_index": args.record_indices[slot],
                    "metadata": metadata,
                    "trainer_logits": train_rows.detach().cpu(),
                    "sglang_generation_logits": sglang_tensors["generation_logits"],
                    "sglang_scorer_logits": sglang_tensors["scorer_logits"],
                    "train_vs_generation_rows": {key: value.detach().cpu() for key, value in generation_rows.items()},
                    "train_vs_scorer_rows": {key: value.detach().cpu() for key, value in scorer_rows.items()},
                }
            )
            aggregate_train.append(train_rows)
            aggregate_generation.append(generation)
            aggregate_scorer.append(scorer)
            aggregate_response_ids.append(response_ids)

        del logits, train_rows, input_ids, response_ids
        dist.barrier()

    peak_gib = torch.cuda.max_memory_allocated() / 1024**3

    if rank == 0:
        train_all = torch.cat(aggregate_train, dim=0)
        generation_all = torch.cat(aggregate_generation, dim=0)
        scorer_all = torch.cat(aggregate_scorer, dim=0)
        response_ids_all = torch.cat(aggregate_response_ids, dim=0)
        generation_summary, _ = _paired_metrics(train_all, generation_all, response_ids_all)
        scorer_summary, _ = _paired_metrics(train_all, scorer_all, response_ids_all)
        sglang_internal_summary, _ = _paired_metrics(generation_all, scorer_all, response_ids_all)

        output = {
            "scope": (
                "Frozen Megatron forward versus previously captured corrected SGLang "
                "rollout/full-prefix logits; trainer routing is "
                + ("replayed from rollout." if args.routing_replay else "selected naturally.")
            ),
            "checkpoint": str(args.checkpoint),
            "sglang_raw": str(args.sglang_raw),
            "sglang_raw_sha256": _sha256(args.sglang_raw),
            "record_indices": args.record_indices,
            "routing_mode": (
                "rollout_replay"
                if args.routing_replay
                else "captured_natural" if args.capture_natural_routing else "natural"
            ),
            "response_tokens": int(response_ids_all.numel()),
            "topology": {
                "world_size": world_size,
                "ep_size": args.ep_size,
                "tp_size": 1,
                "pp_size": 1,
                "cp_size": 1,
                "moe_dispatcher": args.moe_dispatcher,
                "moe_flex_backend": args.moe_flex_backend,
            },
            "environment": {
                key: os.environ.get(key)
                for key in (
                    "V4_FP4_FROZEN_EXPERTS",
                    "V4_FP4_EXPERT_GEMM",
                    "V4_FUSED_SILU_QUANT",
                    "V4_QUANT_DIV_ALIGN",
                    "V4_SHARED_EXPERT_ALIGN_ACT",
                    "V4_MHC_POST_FP32_COMBINE",
                    "TILELANG_CACHE_DIR",
                    "ENABLE_ROUTING_REPLAY",
                    "ROUTING_REPLAY_STAGE",
                )
            },
            "load_stats": load_stats,
            "precision_alignment": precision_alignment,
            "rank_peak_allocated_gib": peak_gib,
            "aggregate": {
                "train_vs_sglang_generation": generation_summary,
                "train_vs_sglang_full_prefix": scorer_summary,
                "sglang_generation_vs_full_prefix_control": sglang_internal_summary,
            },
            "records": output_records,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        raw_output = args.output.with_suffix(".pt")
        torch.save({"records": raw_records}, raw_output)
        print(json.dumps(output, indent=2, sort_keys=True), flush=True)

    dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sglang-raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-indices", type=int, nargs="+", default=[0])
    parser.add_argument("--routing-replay", action="store_true")
    parser.add_argument("--capture-natural-routing", action="store_true")
    parser.add_argument("--apply-zero-lora", action="store_true")
    parser.add_argument("--zero-lora-dim", type=int, default=32)
    parser.add_argument("--zero-lora-alpha", type=float, default=32.0)
    parser.add_argument("--fp8-shared-expert", action="store_true")
    parser.add_argument("--fp8-attention", action="store_true")
    parser.add_argument("--checkpoint-fp8-shared-expert", action="store_true")
    parser.add_argument("--checkpoint-fp8-attention", action="store_true")
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--moe-dispatcher", choices=("flex", "alltoall", "allgather"), default="flex")
    parser.add_argument("--moe-flex-backend", default="deepep")
    parser.add_argument("--moe-deepep-num-sms", type=int, default=20)
    parser.add_argument("--master-port", type=int, default=29723)
    args = parser.parse_args()
    try:
        run(args)
    finally:
        from megatron.core import parallel_state

        if parallel_state.is_initialized():
            parallel_state.destroy_model_parallel()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
