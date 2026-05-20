from __future__ import annotations

import argparse
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slime.utils.ppo_utils import _compute_opsm_mask_batched, _compute_opsm_mask_loop


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _build_inputs(
    *,
    batch_size: int,
    min_response_len: int,
    max_response_len: int,
    opsm_delta: float,
    opsm_aggregation: str,
    opsm_upper: float | None,
    opsm_token_veto_threshold: float | None,
    advantage_mode: str,
    force_token_veto: bool,
    device: torch.device,
    seed: int,
) -> tuple[Namespace, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    lengths = torch.randint(
        min_response_len,
        max_response_len + 1,
        (batch_size,),
        device=device,
        generator=generator,
    ).tolist()

    full_log_probs = []
    full_old_log_probs = []
    advantages = []
    loss_masks = []
    for length in lengths:
        old_log_prob = torch.randn(length, device=device, generator=generator) * 0.2
        log_prob = old_log_prob - torch.randn(length, device=device, generator=generator) * 0.01
        loss_mask = torch.ones(length, device=device)

        full_old_log_probs.append(old_log_prob)
        full_log_probs.append(log_prob)
        if advantage_mode == "broadcast":
            advantage_value = torch.randn((), device=device, generator=generator)
            advantages.append(torch.ones(length, device=device) * advantage_value)
        else:
            advantages.append(torch.randn(length, device=device, generator=generator))
        loss_masks.append(loss_mask)

    if force_token_veto:
        if opsm_token_veto_threshold is None:
            raise ValueError("--force-token-veto requires --opsm-token-veto-threshold.")
        full_log_probs[0][0] = full_old_log_probs[0][0] + torch.log(
            torch.tensor(opsm_token_veto_threshold, device=device)
        ) - 1.0

    return (
        Namespace(
            opsm_delta=opsm_delta,
            opsm_lower=opsm_delta,
            opsm_upper=opsm_upper,
            opsm_aggregation=opsm_aggregation,
            opsm_token_veto_threshold=opsm_token_veto_threshold,
            opsm_use_advantage=True,
            max_turns=4,
        ),
        full_log_probs,
        full_old_log_probs,
        advantages,
        loss_masks,
    )


def _time_fn(fn, *, warmup: int, iters: int, device: torch.device, inputs: tuple) -> float:
    for _ in range(warmup):
        fn(*inputs)
    _synchronize(device)

    start = time.perf_counter()
    for _ in range(iters):
        fn(*inputs)
    _synchronize(device)
    return (time.perf_counter() - start) / iters


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare OPSM loop and batched mask implementations.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-response-len", type=int, default=128)
    parser.add_argument("--max-response-len", type=int, default=2048)
    parser.add_argument("--opsm-delta", type=float, default=1e-4)
    parser.add_argument("--opsm-aggregation", choices=["kl", "geometric", "turns_geometric"], default="kl")
    parser.add_argument("--opsm-upper", type=float, default=None)
    parser.add_argument("--opsm-token-veto-threshold", type=float, default=None)
    parser.add_argument("--force-token-veto", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--advantage-mode", choices=["broadcast", "token"], default="broadcast")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    inputs = _build_inputs(
        batch_size=args.batch_size,
        min_response_len=args.min_response_len,
        max_response_len=args.max_response_len,
        opsm_delta=args.opsm_delta,
        opsm_aggregation=args.opsm_aggregation,
        opsm_upper=args.opsm_upper,
        opsm_token_veto_threshold=args.opsm_token_veto_threshold,
        advantage_mode=args.advantage_mode,
        force_token_veto=args.force_token_veto,
        device=device,
        seed=args.seed,
    )

    loop_mask, loop_clipfrac = _compute_opsm_mask_loop(*inputs)
    batched_mask, batched_clipfrac = _compute_opsm_mask_batched(*inputs)
    mask_equal = torch.equal(loop_mask, batched_mask)
    clipfrac_diff = (loop_clipfrac - batched_clipfrac).abs().item()

    loop_time = _time_fn(_compute_opsm_mask_loop, warmup=args.warmup, iters=args.iters, device=device, inputs=inputs)
    batched_time = _time_fn(
        _compute_opsm_mask_batched,
        warmup=args.warmup,
        iters=args.iters,
        device=device,
        inputs=inputs,
    )

    print(f"device: {device}")
    print(f"batch_size: {args.batch_size}")
    print(f"response_len: [{args.min_response_len}, {args.max_response_len}]")
    print(f"opsm_aggregation: {args.opsm_aggregation}")
    print(f"opsm_token_veto_threshold: {args.opsm_token_veto_threshold}")
    print(f"force_token_veto: {args.force_token_veto}")
    print(f"advantage_mode: {args.advantage_mode}")
    print(f"mask_equal: {mask_equal}")
    print(f"clipfrac_diff: {clipfrac_diff:.8g}")
    print(f"loop_time_ms: {loop_time * 1000:.3f}")
    print(f"batched_time_ms: {batched_time * 1000:.3f}")
    print(f"speedup: {loop_time / batched_time:.3f}x")

    if not mask_equal or clipfrac_diff > 1e-6:
        raise AssertionError("Batched OPSM mask result differs from loop implementation.")


if __name__ == "__main__":
    main()
