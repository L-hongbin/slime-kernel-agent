"""Verify a V4 R2 torch_dist checkpoint against the native checkpoint.

This is a reviewable R2 gate for conversion artifacts. It loads the
Megatron distributed checkpoint into the same PP/EP rank layout used to save it,
then compares representative direct tensors and the first/last local expert
against the native safetensors source.
"""

from __future__ import annotations

import argparse
import socket
from datetime import datetime, timezone
from pathlib import Path

import torch
from megatron.core import dist_checkpointing, parallel_state
from transformers import AutoConfig

from .model_provider import build_v4_mcore_model
from .native_checkpoint import DEFAULT_V4_FLASH_FP8_CKPT, NativeV4Checkpoint, read_expert_tensors, read_native_weight
from .slice_torch_dist import init_dist_from_env, resolve_source_layers


def _release_dir(load_dir: str) -> str:
    path = Path(load_dir)
    if path.name == "release":
        return str(path)
    return str(path / "release")


def _report_header() -> str:
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return f"R2 real torch_dist verification (generated_at={generated_at}, rank0_host={socket.gethostname()})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a V4 R2 torch_dist checkpoint.")
    parser.add_argument("--checkpoint", default=DEFAULT_V4_FLASH_FP8_CKPT)
    parser.add_argument("--load", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--source-layers", type=int, nargs="+", default=None)
    parser.add_argument("--pp-size", type=int, default=1)
    parser.add_argument("--ep-size", type=int, default=1)
    parser.add_argument("--plan-first-layers", type=int, default=None)
    parser.add_argument("--plan-last-layers", type=int, default=None)
    parser.add_argument("--master-port", type=int, default=29652)
    parser.add_argument("--parallel-order", default="tp-cp-ep-dp-pp")
    parser.add_argument(
        "--fp4-experts",
        action="store_true",
        default=False,
        help="Verify a packed-MXFP4 conversion (sets V4_FP4_FROZEN_EXPERTS=1 before the model build).",
    )
    args = parser.parse_args(argv)

    if args.fp4_experts:
        import os

        os.environ["V4_FP4_FROZEN_EXPERTS"] = "1"

    rank, world_size, _ = init_dist_from_env(
        args.master_port,
        pp_size=args.pp_size,
        ep_size=args.ep_size,
        order=args.parallel_order,
    )
    if world_size > 1 and world_size != args.pp_size * args.ep_size:
        raise ValueError(
            f"torchrun WORLD_SIZE={world_size} must equal pp_size*ep_size=" f"{args.pp_size * args.ep_size}"
        )
    try:
        pp_rank = parallel_state.get_pipeline_model_parallel_rank() if world_size > 1 else 0
        ep_rank = parallel_state.get_expert_model_parallel_rank() if world_size > 1 else 0
        ep_size = parallel_state.get_expert_model_parallel_world_size() if world_size > 1 else args.ep_size
        source_layers = resolve_source_layers(args, pp_rank=pp_rank, world_size=world_size)
        if not source_layers:
            raise ValueError("--source-layers must not be empty")

        cfg = AutoConfig.from_pretrained(args.checkpoint, trust_remote_code=True)
        pre_process = args.pp_size == 1 or pp_rank == 0
        post_process = args.pp_size == 1 or pp_rank == args.pp_size - 1
        model = (
            build_v4_mcore_model(
                cfg,
                pre_process=pre_process,
                post_process=post_process,
                params_dtype=torch.bfloat16,
                init_weights=False,
                layer_ids=source_layers,
                expert_model_parallel_size=ep_size,
                expert_model_parallel_rank=ep_rank,
                pipeline_model_parallel_size=args.pp_size,
            )
            .cuda()
            .bfloat16()
        )

        metadata = {"dp_cp_group": parallel_state.get_data_parallel_group(with_context_parallel=True)}
        template = {
            "checkpoint_version": 3.0,
            "iteration": 1,
            "model": model.sharded_state_dict(metadata=metadata),
        }
        # torch-DCP silently NO-OPs requests for keys absent from the checkpoint
        # metadata (observed 2026-07-16: a layer-namespace mismatch left every
        # tensor at its zero/garbage init while load_state_dict reported no
        # missing keys). Assert full key coverage up front instead.
        from megatron.core.dist_checkpointing.mapping import ShardedTensor as _ST

        def _sharded_keys(node):
            if isinstance(node, _ST):
                yield node.key
            elif isinstance(node, dict):
                for v in node.values():
                    yield from _sharded_keys(v)
            elif isinstance(node, (list, tuple)):
                for v in node:
                    yield from _sharded_keys(v)

        ckpt_keys = set(dist_checkpointing.load_tensors_metadata(_release_dir(args.load)).keys())
        requested = set(_sharded_keys(template["model"]))
        absent = sorted(requested - ckpt_keys)
        if absent:
            raise AssertionError(
                f"{len(absent)} requested tensors are ABSENT from the checkpoint "
                f"(namespace mismatch? wrong --source-layers / missing "
                f"--preserve-global-layer-ids on the save side?): {absent[:8]}..."
            )
        loaded = dist_checkpointing.load(template, _release_dir(args.load))
        result = model.load_state_dict(loaded["model"], strict=False)

        native = NativeV4Checkpoint(args.checkpoint)
        source_layer = source_layers[0]
        q_native, _, _ = read_native_weight(
            native, f"layers.{source_layer}.attn.wq_a.weight", output_dtype=torch.bfloat16
        )
        q_loaded = model.layers[0].self_attn.q_a_proj.weight.detach().cpu()

        experts = model.layers[0].mlp.experts
        local_count = experts.num_experts
        first = experts.local_expert_start
        last = experts.local_expert_end - 1
        first_native = read_expert_tensors(native, source_layer, first, output_dtype=torch.bfloat16)
        last_native = read_expert_tensors(native, source_layer, last, output_dtype=torch.bfloat16)
        packed = getattr(experts, "_experts_fp4", False)
        if packed:
            # Packed conversions must round-trip the official bytes EXACTLY —
            # compare all four uint8 tensors bitwise (max abs diff of uint8 views),
            # assert resident dtype, and reject invalid E8M0 scale bytes anywhere
            # in the rank's shard (0x00 = unpopulated, 0xFF = NaN; codex finding 9).
            expert_lines = []
            for name in ("gate_up_proj_fp4", "gate_up_proj_sf", "down_proj_fp4", "down_proj_sf"):
                loaded_buf = getattr(experts, name).detach().cpu()
                if loaded_buf.dtype != torch.uint8:
                    raise AssertionError(f"{name} resident dtype {loaded_buf.dtype} != torch.uint8")
                if name.endswith("_sf"):
                    bad = int((loaded_buf == 0x00).sum()) + int((loaded_buf == 0xFF).sum())
                    expert_lines.append(f"{name}_invalid_scale_bytes={bad}")
                    if bad:
                        # ENFORCED, not report-only (codex impl review finding 1):
                        # 0x00 = unpopulated buffer / wrong family, 0xFF = NaN scale.
                        # The conversion job must exit nonzero, not print "complete".
                        raise AssertionError(
                            f"{name} has {bad} invalid E8M0 scale bytes (0x00/0xFF) in this "
                            "rank's shard — conversion verify FAILED"
                        )
                for label, gid, nat in (("first", first, first_native), ("last", last, last_native)):
                    idx = 0 if label == "first" else -1
                    ref = nat[f"layers.{source_layer}.mlp.experts.{name}[{gid}]"]
                    diff = (loaded_buf[idx].to(torch.int16) - ref.to(torch.int16)).abs().max().item()
                    expert_lines.append(f"expert{gid}_{name}_bytediff={diff}")
        else:
            gate = experts.gate_up_proj.detach().cpu()
            down = experts.down_proj.detach().cpu()
            expert_lines = [
                "expert"
                f"{first}_gate_up_diff="
                f"{(gate[0] - first_native[f'layers.{source_layer}.mlp.experts.gate_up_proj[{first}]']).abs().max().item()}",
                "expert"
                f"{first}_down_diff="
                f"{(down[0] - first_native[f'layers.{source_layer}.mlp.experts.down_proj[{first}]']).abs().max().item()}",
                "expert"
                f"{last}_gate_up_diff="
                f"{(gate[-1] - last_native[f'layers.{source_layer}.mlp.experts.gate_up_proj[{last}]']).abs().max().item()}",
                "expert"
                f"{last}_down_diff="
                f"{(down[-1] - last_native[f'layers.{source_layer}.mlp.experts.down_proj[{last}]']).abs().max().item()}",
            ]

        lines = [
            f"rank={rank} pp_rank={pp_rank}/{args.pp_size} ep_rank={ep_rank}/{ep_size} "
            f"source_layers={source_layers}",
            f"local_experts={local_count} global_experts={first}..{last} packed_fp4={packed}",
            f"load_state_dict_missing={list(result.missing_keys)}",
            f"load_state_dict_unexpected={list(result.unexpected_keys)}",
            f"q_a_proj_diff={(q_loaded - q_native).abs().max().item()}",
            *expert_lines,
        ]

        rank_text = "\n".join(lines)
        print(rank_text, flush=True)
        gathered = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered, rank_text)
        if rank == 0:
            rendered = [
                _report_header(),
                f"input_native={args.checkpoint}",
                f"slice_checkpoint={args.load}",
                f"world_size={world_size} pp_size={args.pp_size} ep_size={args.ep_size}",
            ]
            for r, rank_output in enumerate(gathered):
                rendered.append("")
                rendered.append(f"## rank {r}")
                rendered.append(str(rank_output).strip())
            rendered.append("")
            rendered.append("note: this verifies representative direct and expert tensors for each loaded rank.")
            output = Path(args.output)
            output.write_text("\n".join(rendered) + "\n", encoding="utf-8")
        torch.distributed.barrier()
    finally:
        if parallel_state.is_initialized():
            parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
