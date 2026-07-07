"""R3 PP-slice model semantics smoke.

This is not a Megatron P2P scheduler smoke. It validates the V4 model-stage
contract that PP>1 needs before the real scheduler can work:

* non-first stages consume the received ``[B,S,hc,H]`` stream via
  ``set_input_tensor``;
* every stage can recompute V4 RoPE locally from the same ``input_ids`` /
  ``position_ids``;
* local ``layers.N`` modules map to the intended global layer ids.

The remaining production PP gate is Megatron's P2P tensor shape, because stock
non-interleaved PP expects ``[S,B,H]`` while V4 sends ``[B,S,hc,H]``.
"""

from __future__ import annotations

import argparse
import json

import torch

from megatron.core import parallel_state, tensor_parallel

from ..m0_smoke import tiny_config
from ..model_provider import build_v4_mcore_model


def _named_tensors(module):
    tensors = {name: param for name, param in module.named_parameters()}
    tensors.update({name: buf for name, buf in module.named_buffers()})
    return tensors


@torch.no_grad()
def _copy_stage_from_full(stage, full):
    full_tensors = _named_tensors(full)
    copied = []
    skipped = []
    for name, dest in _named_tensors(stage).items():
        source_name = name
        if name.startswith("layers."):
            parts = name.split(".", 2)
            local_idx = int(parts[1])
            global_idx = stage.layer_ids[local_idx]
            source_name = f"layers.{global_idx}.{parts[2]}"
        src = full_tensors.get(source_name)
        if src is None or tuple(src.shape) != tuple(dest.shape):
            skipped.append(name)
            continue
        dest.copy_(src.to(device=dest.device, dtype=dest.dtype))
        copied.append(name)
    return copied, skipped


def _init_single_process_groups(master_port: int):
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{master_port}",
            world_size=1,
            rank=0,
            device_id=torch.device("cuda:0"),
        )
    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
            expert_tensor_parallel_size=1,
        )
    tensor_parallel.model_parallel_cuda_manual_seed(1234)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--split-after-layer", type=int, default=1)
    parser.add_argument("--master-port", type=int, default=29547)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for R3 PP-slice smoke")
    torch.cuda.set_device(0)
    _init_single_process_groups(args.master_port)
    dev = torch.device("cuda")
    try:
        cfg = tiny_config()
        if not 0 < args.split_after_layer < cfg.num_hidden_layers:
            raise ValueError(
                f"--split-after-layer must be in [1, {cfg.num_hidden_layers - 1}], " f"got {args.split_after_layer}"
            )
        first_layers = tuple(range(args.split_after_layer))
        last_layers = tuple(range(args.split_after_layer, cfg.num_hidden_layers))

        torch.manual_seed(args.seed)
        full = build_v4_mcore_model(cfg, params_dtype=torch.bfloat16).to(dev).bfloat16()
        full.restore_fp32_modules()

        torch.manual_seed(args.seed + 1)
        stage0 = (
            build_v4_mcore_model(
                cfg,
                pre_process=True,
                post_process=False,
                params_dtype=torch.bfloat16,
                layer_ids=first_layers,
            )
            .to(dev)
            .bfloat16()
        )
        stage0.restore_fp32_modules()
        torch.manual_seed(args.seed + 2)
        stage1 = (
            build_v4_mcore_model(
                cfg,
                pre_process=False,
                post_process=True,
                params_dtype=torch.bfloat16,
                layer_ids=last_layers,
            )
            .to(dev)
            .bfloat16()
        )
        stage1.restore_fp32_modules()
        copied0, skipped0 = _copy_stage_from_full(stage0, full)
        copied1, skipped1 = _copy_stage_from_full(stage1, full)

        gen = torch.Generator(device="cpu").manual_seed(args.seed + 3)
        input_ids = torch.randint(
            0,
            cfg.vocab_size,
            (args.batch_size, args.seq_len),
            generator=gen,
            dtype=torch.long,
        ).to(dev)
        with torch.no_grad():
            full_logits = full(input_ids=input_ids)
            hidden = stage0(input_ids=input_ids)
            stage1.set_input_tensor(hidden)
            pp_logits = stage1(input_ids=input_ids)
            max_abs = (pp_logits.float() - full_logits.float()).abs().max()
            rel = max_abs / full_logits.float().abs().amax().clamp_min(1e-6)
            passed = bool(max_abs.item() == 0.0)

        print(
            json.dumps(
                {
                    "passed": passed,
                    "batch_size": args.batch_size,
                    "seq_len": args.seq_len,
                    "first_layers": list(first_layers),
                    "last_layers": list(last_layers),
                    "hidden_shape": list(hidden.shape),
                    "logits_shape": list(pp_logits.shape),
                    "copied_stage0": len(copied0),
                    "copied_stage1": len(copied1),
                    "skipped_stage0": skipped0,
                    "skipped_stage1": skipped1,
                    "max_abs": float(max_abs.item()),
                    "rel": float(rel.item()),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not passed:
            raise RuntimeError("R3 PP-slice smoke failed")
    finally:
        if parallel_state.is_initialized():
            parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
