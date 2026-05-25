#!/usr/bin/env python3
"""Check sequence_mis loop and batch modes on deterministic tensors.

Run from the repository root:
    python examples/kernel_agent/test/check_sequence_mis_consistency.py
"""

from __future__ import annotations

import sys
import time
import types
from argparse import Namespace
from pathlib import Path

import torch


def _install_megatron_stub_if_needed() -> None:
    try:
        from megatron.core import mpu  # noqa: F401

        return
    except Exception:
        pass

    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    mpu_mod = types.ModuleType("megatron.core.mpu")
    mpu_mod.get_context_parallel_world_size = lambda: 1
    mpu_mod.get_context_parallel_rank = lambda: 0
    mpu_mod.get_context_parallel_group = lambda: None
    core_mod.mpu = mpu_mod
    megatron_mod.core = core_mod
    sys.modules.setdefault("megatron", megatron_mod)
    sys.modules.setdefault("megatron.core", core_mod)
    sys.modules.setdefault("megatron.core.mpu", mpu_mod)


def _import_kernel_filter():
    _install_megatron_stub_if_needed()
    repo_root = Path(__file__).resolve().parents[3]
    kernel_agent_dir = repo_root / "examples" / "kernel_agent"
    for item in (repo_root, kernel_agent_dir):
        item_str = str(item)
        if item_str not in sys.path:
            sys.path.insert(0, item_str)

    import kernel_filter

    # This script validates sequence-level logic, not CP transport.
    kernel_filter.all_gather_with_cp = lambda tensor, _total_length, _response_length: tensor
    return kernel_filter


def _import_targets():
    return _import_kernel_filter().sequence_mis


def _build_inputs() -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    # diff = train_log_prob - rollout_log_prob. Values are chosen to create
    # keep, lower-bound reject, upper-bound reject, padding, and veto cases.
    diffs = [
        [0.0, 0.0, 0.0],
        [-0.22314355, -0.22314355],  # exp(mean(diff)) ~= 0.8
        [0.18232156, 0.18232156, 0.18232156, 0.18232156],  # ~= 1.2
        [0.0],
        [-11.512925, 0.0, 0.0],  # catastrophic token for veto threshold 1e-4
        [0.0, 0.0],
    ]
    mask_values = [
        [1, 1, 1],
        [1, 1],
        [1, 1, 0, 1],
        [0],
        [1, 1, 1],
        [1, 0],
    ]

    train_log_probs = [torch.tensor(item, dtype=torch.float32) for item in diffs]
    rollout_log_probs = [torch.zeros_like(item) for item in train_log_probs]
    loss_masks = [torch.tensor(item, dtype=torch.float32) for item in mask_values]
    advantages = [-torch.ones_like(item) for item in loss_masks]
    return train_log_probs, rollout_log_probs, loss_masks, advantages


def _actual_masks_from_sequence_mis(
    sequence_mis,
    *,
    aggregation: str | None,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    lower: float,
    upper: float,
    token_veto_threshold: float | None,
    mode: str,
    advantages: list[torch.Tensor] | None = None,
    use_advantage: bool = False,
) -> list[torch.Tensor]:
    args = Namespace(
        sequence_mis_aggregation=aggregation,
        sequence_mis_lower=lower,
        sequence_mis_upper=upper,
        sequence_mis_token_veto_threshold=token_veto_threshold,
        sequence_mis_mode=mode,
        sequence_mis_batch_size=8,
        n_samples_per_prompt=2,
        max_turns=3,
        sequence_mis_use_advantage=use_advantage,
    )
    rollout_data = {
        "log_probs": [item.clone() for item in train_log_probs],
        "rollout_log_probs": [item.clone() for item in rollout_log_probs],
        "loss_masks": [item.clone() for item in loss_masks],
        "total_lengths": [len(item) for item in loss_masks],
        "response_lengths": [len(item) for item in loss_masks],
    }
    if advantages is not None:
        rollout_data["advantages"] = [item.clone() for item in advantages]
    sequence_mis(args, rollout_id=0, rollout_data=rollout_data)
    return rollout_data["loss_masks"]


def assert_sequence_mis_loop_matches_batch(aggregation: str, *, token_veto_threshold: float | None = None) -> None:
    sequence_mis = _import_targets()
    train_log_probs, rollout_log_probs, loss_masks, _advantages = _build_inputs()
    lower, upper = (-0.1, 0.1) if aggregation == "kl" else (0.9, 1.1)

    expected = _actual_masks_from_sequence_mis(
        sequence_mis,
        aggregation=aggregation,
        train_log_probs=train_log_probs,
        rollout_log_probs=rollout_log_probs,
        loss_masks=loss_masks,
        lower=lower,
        upper=upper,
        token_veto_threshold=token_veto_threshold,
        mode="loop",
    )
    actual = _actual_masks_from_sequence_mis(
        sequence_mis,
        aggregation=aggregation,
        train_log_probs=train_log_probs,
        rollout_log_probs=rollout_log_probs,
        loss_masks=loss_masks,
        lower=lower,
        upper=upper,
        token_veto_threshold=token_veto_threshold,
        mode="batch",
    )

    for i, (actual_mask, expected_mask) in enumerate(zip(actual, expected, strict=True)):
        if not torch.equal(actual_mask, expected_mask):
            raise AssertionError(
                f"{aggregation} loop/batch mismatch at sample {i}: "
                f"batch={actual_mask.tolist()} loop={expected_mask.tolist()}"
            )


def test_sequence_mis_loop_matches_batch() -> None:
    for aggregation in ("kl", "geometric", "turns_geometric"):
        assert_sequence_mis_loop_matches_batch(aggregation)
        assert_sequence_mis_loop_matches_batch(aggregation, token_veto_threshold=1e-4)


def test_sequence_mis_token_veto_without_aggregation() -> None:
    sequence_mis = _import_targets()
    train_log_probs, rollout_log_probs, loss_masks, _advantages = _build_inputs()

    actual = _actual_masks_from_sequence_mis(
        sequence_mis,
        aggregation=None,
        train_log_probs=train_log_probs,
        rollout_log_probs=rollout_log_probs,
        loss_masks=loss_masks,
        lower=float("-inf"),
        upper=float("inf"),
        token_veto_threshold=1e-4,
        mode="batch",
    )
    expected = [item.clone() for item in loss_masks]
    expected[4] = torch.zeros_like(expected[4])

    for i, (actual_mask, expected_mask) in enumerate(zip(actual, expected, strict=True)):
        if not torch.equal(actual_mask, expected_mask):
            raise AssertionError(
                f"token-veto-only mismatch at sample {i}: "
                f"actual={actual_mask.tolist()} expected={expected_mask.tolist()}"
            )


def test_sequence_mis_use_advantage_protects_positive_advantage() -> None:
    sequence_mis = _import_targets()
    train_log_probs, rollout_log_probs, loss_masks, advantages = _build_inputs()
    advantages = [item.clone() for item in advantages]
    advantages[1] = torch.ones_like(advantages[1])

    for mode in ("loop", "batch"):
        actual = _actual_masks_from_sequence_mis(
            sequence_mis,
            aggregation="geometric",
            train_log_probs=train_log_probs,
            rollout_log_probs=rollout_log_probs,
            loss_masks=loss_masks,
            lower=0.9,
            upper=1.1,
            token_veto_threshold=None,
            mode=mode,
            advantages=advantages,
            use_advantage=True,
        )
        if not torch.equal(actual[1], loss_masks[1]):
            raise AssertionError(f"use_advantage should keep positive-advantage sample in {mode} mode")
        if not torch.equal(actual[2], torch.zeros_like(loss_masks[2])):
            raise AssertionError(f"use_advantage should still reject negative-advantage sample in {mode} mode")


def _build_benchmark_inputs(
    *,
    num_groups: int = 48,
    max_turns: int = 3,
    min_len: int = 16,
    max_len: int = 256,
    device: torch.device | str = "cpu",
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    generator = torch.Generator().manual_seed(1234)
    total = num_groups * max_turns
    train_log_probs = []
    rollout_log_probs = []
    loss_masks = []
    for i in range(total):
        length = int(torch.randint(min_len, max_len + 1, (1,), generator=generator).item())
        rollout = torch.randn(length, generator=generator) * 0.05
        # Keep the distribution near 1.0 with a few deliberate outliers/veto cases.
        diff = torch.randn(length, generator=generator) * 0.03
        if i % 17 == 0:
            diff = diff + 0.2
        if i % 19 == 0:
            diff = diff - 0.2
        if i % 23 == 0:
            diff[0] = -11.512925
        train = rollout + diff
        mask = torch.ones(length, dtype=torch.float32)
        if i % 11 == 0 and length > 4:
            mask[-3:] = 0
        train_log_probs.append(train.float().to(device))
        rollout_log_probs.append(rollout.float().to(device))
        loss_masks.append(mask.to(device))
    return train_log_probs, rollout_log_probs, loss_masks


def _run_sequence_mis_once(
    sequence_mis,
    *,
    aggregation: str,
    mode: str,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
) -> list[torch.Tensor]:
    lower, upper = (-0.1, 0.1) if aggregation == "kl" else (0.9, 1.1)
    args = Namespace(
        sequence_mis_aggregation=aggregation,
        sequence_mis_lower=lower,
        sequence_mis_upper=upper,
        sequence_mis_token_veto_threshold=1e-4,
        sequence_mis_mode=mode,
        sequence_mis_batch_size=8,
        n_samples_per_prompt=2,
        max_turns=3,
    )
    rollout_data = {
        "log_probs": [item.clone() for item in train_log_probs],
        "rollout_log_probs": [item.clone() for item in rollout_log_probs],
        "loss_masks": [item.clone() for item in loss_masks],
        "total_lengths": [len(item) for item in loss_masks],
        "response_lengths": [len(item) for item in loss_masks],
    }
    sequence_mis(args, rollout_id=0, rollout_data=rollout_data)
    return rollout_data["loss_masks"]


def _time_sequence_mis(
    sequence_mis,
    *,
    aggregation: str,
    mode: str,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    iters: int = 20,
    device: torch.device | str = "cpu",
) -> float:
    # Warmup keeps import/cache noise out of the tiny benchmark.
    _run_sequence_mis_once(
        sequence_mis,
        aggregation=aggregation,
        mode=mode,
        train_log_probs=train_log_probs,
        rollout_log_probs=rollout_log_probs,
        loss_masks=loss_masks,
    )
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        _run_sequence_mis_once(
            sequence_mis,
            aggregation=aggregation,
            mode=mode,
            train_log_probs=train_log_probs,
            rollout_log_probs=rollout_log_probs,
            loss_masks=loss_masks,
        )
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return (time.perf_counter() - start) / iters


def benchmark_loop_vs_batch() -> None:
    sequence_mis = _import_targets()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_log_probs, rollout_log_probs, loss_masks = _build_benchmark_inputs(device=device)
    print(f"\nloop vs batch sequence_mis benchmark ({device.type.upper()}, CP gather patched to no-op)")
    print("aggregation        loop_ms    batch_ms   speedup")
    for aggregation in ("kl", "geometric", "turns_geometric"):
        loop_masks = _run_sequence_mis_once(
            sequence_mis,
            aggregation=aggregation,
            mode="loop",
            train_log_probs=train_log_probs,
            rollout_log_probs=rollout_log_probs,
            loss_masks=loss_masks,
        )
        batch_masks = _run_sequence_mis_once(
            sequence_mis,
            aggregation=aggregation,
            mode="batch",
            train_log_probs=train_log_probs,
            rollout_log_probs=rollout_log_probs,
            loss_masks=loss_masks,
        )
        for i, (loop_mask, batch_mask) in enumerate(zip(loop_masks, batch_masks, strict=True)):
            if not torch.equal(loop_mask, batch_mask):
                raise AssertionError(
                    f"loop/batch mismatch for {aggregation} at sample {i}: "
                    f"loop={loop_mask.tolist()} batch={batch_mask.tolist()}"
                )

        loop_time = _time_sequence_mis(
            sequence_mis,
            aggregation=aggregation,
            mode="loop",
            train_log_probs=train_log_probs,
            rollout_log_probs=rollout_log_probs,
            loss_masks=loss_masks,
            device=device,
        )
        batch_time = _time_sequence_mis(
            sequence_mis,
            aggregation=aggregation,
            mode="batch",
            train_log_probs=train_log_probs,
            rollout_log_probs=rollout_log_probs,
            loss_masks=loss_masks,
            device=device,
        )
        speedup = loop_time / batch_time if batch_time > 0 else float("inf")
        print(f"{aggregation:16s} {loop_time * 1000:8.3f} {batch_time * 1000:9.3f} {speedup:8.2f}x")


def main() -> None:
    test_sequence_mis_loop_matches_batch()
    test_sequence_mis_token_veto_without_aggregation()
    test_sequence_mis_use_advantage_protects_positive_advantage()
    print(
        "sequence_mis loop/batch modes match for kl, geometric, turns_geometric, "
        "batch mode supports token-veto-only mode, and use_advantage protects positive advantages."
    )
    benchmark_loop_vs_batch()


if __name__ == "__main__":
    main()
