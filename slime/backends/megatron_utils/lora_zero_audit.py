"""Fail-closed proof that a freshly loaded LoRA has zero functional delta.

The V4 ``LinearAdapter`` computes ``B(A(x))`` with parameters named
``linear_in.weight`` (A) and ``linear_out.weight`` (B).  Exact-zero B therefore
makes the adapter contribution identically zero regardless of A's random seed.

This module is intentionally independent of Megatron. The actor imports it
only when ``--assert-zero-lora-out`` is enabled, so ordinary training does no
parameter scan and pays no distributed-collective cost.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist

PASS_MARKER = "V4_ENTROPY_AB_ZERO_LORA_OUT_OK"
FAIL_MARKER = "V4_ENTROPY_AB_ZERO_LORA_OUT_FAIL"

_LORA_IN_SUFFIX = ".linear_in.weight"
_LORA_OUT_SUFFIX = ".linear_out.weight"


@dataclass(frozen=True)
class LoRAZeroAuditStats:
    """Linearly reducible LoRA A/B statistics for one or all train ranks."""

    linear_in_tensors: int = 0
    linear_in_numel: int = 0
    linear_in_nonfinite: int = 0
    linear_in_sum: float = 0.0
    linear_in_sum_sq: float = 0.0
    linear_in_max_abs: float = 0.0
    linear_out_tensors: int = 0
    linear_out_numel: int = 0
    linear_out_nonzero: int = 0
    linear_out_nonfinite: int = 0
    linear_out_sum_sq: float = 0.0
    linear_out_max_abs: float = 0.0

    @property
    def linear_in_mean(self) -> float:
        return self.linear_in_sum / self.linear_in_numel if self.linear_in_numel else float("nan")

    @property
    def linear_in_l2(self) -> float:
        return math.sqrt(max(self.linear_in_sum_sq, 0.0))

    @property
    def linear_in_rms(self) -> float:
        if not self.linear_in_numel:
            return float("nan")
        return math.sqrt(max(self.linear_in_sum_sq, 0.0) / self.linear_in_numel)

    @property
    def linear_out_l2(self) -> float:
        return math.sqrt(max(self.linear_out_sum_sq, 0.0))


def _iter_unique_trainable_lora_params(model: Sequence[torch.nn.Module]):
    """Yield each live trainable LoRA A/B parameter once per process."""

    seen: set[int] = set()
    for model_chunk in model:
        for name, param in model_chunk.named_parameters():
            if not param.requires_grad or id(param) in seen:
                continue
            if name.endswith(_LORA_IN_SUFFIX):
                side = "in"
            elif name.endswith(_LORA_OUT_SUFFIX):
                side = "out"
            else:
                continue
            seen.add(id(param))
            yield side, param.detach()


def collect_local_lora_zero_stats(model: Sequence[torch.nn.Module]) -> LoRAZeroAuditStats:
    """Collect exact B-zero checks and cheap A moments without copying weights.

    Reductions run on each parameter's resident device.  Only three tiny scalar
    buffers per device are copied back to the CPU after the scan.
    """

    # counts: A tensors/numel/nonfinite, B tensors/numel/nonzero/nonfinite
    count_accumulators: dict[torch.device, torch.Tensor] = {}
    # sums: A sum/sum_sq, B sum_sq
    sum_accumulators: dict[torch.device, torch.Tensor] = {}
    # maxima: A max_abs, B max_abs
    max_accumulators: dict[torch.device, torch.Tensor] = {}

    with torch.no_grad():
        for side, value in _iter_unique_trainable_lora_params(model):
            device = value.device
            counts = count_accumulators.setdefault(device, torch.zeros(7, dtype=torch.int64, device=device))
            sums = sum_accumulators.setdefault(device, torch.zeros(3, dtype=torch.float64, device=device))
            maxima = max_accumulators.setdefault(device, torch.zeros(2, dtype=torch.float64, device=device))

            finite_count = torch.isfinite(value).sum(dtype=torch.int64)
            value_l2 = torch.linalg.vector_norm(value, ord=2, dtype=torch.float32).to(torch.float64)
            value_max_abs = torch.linalg.vector_norm(value, ord=float("inf"), dtype=torch.float32).to(torch.float64)

            if side == "in":
                counts[0] += 1
                counts[1] += value.numel()
                counts[2] += value.numel() - finite_count
                sums[0] += value.sum(dtype=torch.float32).to(torch.float64)
                sums[1] += value_l2.square()
                maxima[0] = torch.maximum(maxima[0], value_max_abs)
            else:
                counts[3] += 1
                counts[4] += value.numel()
                counts[5] += torch.count_nonzero(value)
                counts[6] += value.numel() - finite_count
                sums[2] += value_l2.square()
                maxima[1] = torch.maximum(maxima[1], value_max_abs)

    counts = [0] * 7
    sums = [0.0] * 3
    maxima = [0.0] * 2
    for device in count_accumulators:
        device_counts = count_accumulators[device].cpu().tolist()
        device_sums = sum_accumulators[device].cpu().tolist()
        device_maxima = max_accumulators[device].cpu().tolist()
        counts = [left + int(right) for left, right in zip(counts, device_counts, strict=True)]
        sums = [left + float(right) for left, right in zip(sums, device_sums, strict=True)]
        maxima = [max(left, float(right)) for left, right in zip(maxima, device_maxima, strict=True)]

    return LoRAZeroAuditStats(
        linear_in_tensors=counts[0],
        linear_in_numel=counts[1],
        linear_in_nonfinite=counts[2],
        linear_in_sum=sums[0],
        linear_in_sum_sq=sums[1],
        linear_in_max_abs=maxima[0],
        linear_out_tensors=counts[3],
        linear_out_numel=counts[4],
        linear_out_nonzero=counts[5],
        linear_out_nonfinite=counts[6],
        linear_out_sum_sq=sums[2],
        linear_out_max_abs=maxima[1],
    )


def reduce_lora_zero_stats(
    local: LoRAZeroAuditStats,
    *,
    process_group=None,
) -> LoRAZeroAuditStats:
    """All-reduce local statistics so every train rank makes one decision."""

    if not (dist.is_available() and dist.is_initialized()):
        return local

    counts = torch.tensor(
        [
            local.linear_in_tensors,
            local.linear_in_numel,
            local.linear_in_nonfinite,
            local.linear_out_tensors,
            local.linear_out_numel,
            local.linear_out_nonzero,
            local.linear_out_nonfinite,
        ],
        dtype=torch.int64,
    )
    sums = torch.tensor(
        [local.linear_in_sum, local.linear_in_sum_sq, local.linear_out_sum_sq],
        dtype=torch.float64,
    )
    maxima = torch.tensor(
        [local.linear_in_max_abs, local.linear_out_max_abs],
        dtype=torch.float64,
    )
    dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=process_group)
    dist.all_reduce(sums, op=dist.ReduceOp.SUM, group=process_group)
    dist.all_reduce(maxima, op=dist.ReduceOp.MAX, group=process_group)

    return LoRAZeroAuditStats(
        linear_in_tensors=int(counts[0].item()),
        linear_in_numel=int(counts[1].item()),
        linear_in_nonfinite=int(counts[2].item()),
        linear_in_sum=float(sums[0].item()),
        linear_in_sum_sq=float(sums[1].item()),
        linear_in_max_abs=float(maxima[0].item()),
        linear_out_tensors=int(counts[3].item()),
        linear_out_numel=int(counts[4].item()),
        linear_out_nonzero=int(counts[5].item()),
        linear_out_nonfinite=int(counts[6].item()),
        linear_out_sum_sq=float(sums[2].item()),
        linear_out_max_abs=float(maxima[1].item()),
    )


def _format_stats(stats: LoRAZeroAuditStats) -> str:
    return (
        f"linear_out_tensors={stats.linear_out_tensors} "
        f"linear_out_numel={stats.linear_out_numel} "
        f"linear_out_nonzero={stats.linear_out_nonzero} "
        f"linear_out_nonfinite={stats.linear_out_nonfinite} "
        f"linear_out_max_abs={stats.linear_out_max_abs:.9e} "
        f"linear_out_l2={stats.linear_out_l2:.9e} "
        f"linear_in_tensors={stats.linear_in_tensors} "
        f"linear_in_numel={stats.linear_in_numel} "
        f"linear_in_nonfinite={stats.linear_in_nonfinite} "
        f"linear_in_mean={stats.linear_in_mean:.9e} "
        f"linear_in_rms={stats.linear_in_rms:.9e} "
        f"linear_in_max_abs={stats.linear_in_max_abs:.9e} "
        f"linear_in_l2={stats.linear_in_l2:.9e}"
    )


def assert_zero_lora_out(
    model: Sequence[torch.nn.Module],
    *,
    process_group=None,
    should_print: bool = False,
    print_fn: Callable[[str], None] = print,
) -> LoRAZeroAuditStats:
    """Prove all trainable LoRA B tensors are exactly zero on every rank.

    Every rank completes the same reductions before validation.  Consequently a
    nonzero B or non-finite A on any one rank makes every rank raise instead of
    allowing peers to enter later model collectives and hang.
    """

    stats = reduce_lora_zero_stats(collect_local_lora_zero_stats(model), process_group=process_group)
    failures = []
    if stats.linear_in_tensors == 0 or stats.linear_out_tensors == 0:
        failures.append("missing trainable LoRA A/B parameters")
    if stats.linear_in_tensors != stats.linear_out_tensors:
        failures.append("LoRA A/B tensor counts differ")
    if stats.linear_in_nonfinite:
        failures.append("LoRA A contains non-finite values")
    if stats.linear_out_nonfinite:
        failures.append("LoRA B contains non-finite values")
    if stats.linear_out_nonzero:
        failures.append("LoRA B is not exactly zero")
    if stats.linear_out_max_abs != 0.0 or stats.linear_out_l2 != 0.0:
        failures.append("LoRA B zero moments are inconsistent")

    formatted = _format_stats(stats)
    if failures:
        raise RuntimeError(f"[{FAIL_MARKER}] {'; '.join(failures)}; {formatted}")
    if should_print:
        print_fn(f"[{PASS_MARKER}] {formatted}")
    return stats


__all__ = [
    "FAIL_MARKER",
    "PASS_MARKER",
    "LoRAZeroAuditStats",
    "assert_zero_lora_out",
    "collect_local_lora_zero_stats",
    "reduce_lora_zero_stats",
]
