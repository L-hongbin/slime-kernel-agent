"""Benchmark the current OPSM mask implementation."""

from __future__ import annotations

import argparse
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slime.utils.ppo_utils import compute_opsm_mask


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _build_inputs(
    *,
    batch_size: int,
    min_response_len: int,
    max_response_len: int,
    opsm_delta: float,
    advantage_mode: str,
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

    return Namespace(opsm_delta=opsm_delta), full_log_probs, full_old_log_probs, advantages, loss_masks


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
    parser = argparse.ArgumentParser(description="Benchmark the current OPSM mask implementation.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-response-len", type=int, default=128)
    parser.add_argument("--max-response-len", type=int, default=2048)
    parser.add_argument("--opsm-delta", type=float, default=1e-4)
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
        advantage_mode=args.advantage_mode,
        device=device,
        seed=args.seed,
    )

    opsm_mask, opsm_clipfrac = compute_opsm_mask(*inputs)
    opsm_time = _time_fn(compute_opsm_mask, warmup=args.warmup, iters=args.iters, device=device, inputs=inputs)

    print(f"device: {device}")
    print(f"batch_size: {args.batch_size}")
    print(f"response_len: [{args.min_response_len}, {args.max_response_len}]")
    print(f"opsm_delta: {args.opsm_delta}")
    print(f"advantage_mode: {args.advantage_mode}")
    print(f"kept_tokens: {opsm_mask.sum().item():.0f}")
    print(f"opsm_clipfrac: {opsm_clipfrac.item():.8g}")
    print(f"opsm_time_ms: {opsm_time * 1000:.3f}")


if __name__ == "__main__":
    main()
