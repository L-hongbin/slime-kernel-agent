"""Measure replay-mask construction and TP log-prob chunking without loading the model."""

import argparse
import ast
import json
import os
import statistics
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from slime.backends.megatron_utils.loss import _fill_topp_mask_rows
from slime.utils.ppo_utils import calculate_log_probs_and_entropy


def measure(operation, repetitions):
    operation()
    torch.cuda.synchronize()
    durations = []
    for _ in range(repetitions):
        torch.cuda.synchronize()
        started = time.perf_counter()
        operation()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
    return statistics.median(durations)


def benchmark_mask(args):
    source = ast.parse(args.baseline_loss.read_text())
    function = next(
        node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "_fill_topp_mask_rows"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(args.baseline_loss), "exec"), namespace)
    samples = torch.load(args.rollout, map_location="cpu", weights_only=False)["samples"]
    sample = max(samples, key=lambda item: item["response_length"])
    ids = sample["rollout_top_p_token_ids"]
    offsets = sample["rollout_top_p_token_offsets"]
    ids = ids.tolist() if torch.is_tensor(ids) else list(ids)
    offsets = offsets.tolist() if torch.is_tensor(offsets) else list(offsets)
    length = sample["response_length"]
    results = []
    for rank in range(4):
        original = torch.ones(length, 62080, dtype=torch.bool, device="cuda")
        batched = torch.ones_like(original)
        common = (0, 0, length, rank * 62080, (rank + 1) * 62080)
        baseline = partial(namespace["_fill_topp_mask_rows"], original, ids, offsets, *common)
        ids_array, offsets_array = np.asarray(ids, dtype=np.int64), np.asarray(offsets, dtype=np.int64)
        optimized = partial(_fill_topp_mask_rows, batched, ids_array, offsets_array, *common)
        baseline_seconds = measure(baseline, args.repetitions)
        optimized_seconds = measure(optimized, args.repetitions)
        assert torch.equal(original, batched)
        results.append(
            dict(
                tp_rank=rank,
                response_tokens=length,
                baseline_s=baseline_seconds,
                optimized_s=optimized_seconds,
                speedup=baseline_seconds / optimized_seconds,
            )
        )
        del original, batched
    return {"mask": results}


def benchmark_loss(args):
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    torch.manual_seed(1234 + rank)
    vocab = 248320 // dist.get_world_size()
    logits = torch.randn(args.tokens, vocab, device="cuda", requires_grad=True)
    tokens = torch.arange(args.tokens, device="cuda") % 248320
    keep = torch.zeros_like(logits, dtype=torch.bool)
    keep[:, :8] = True
    output = []
    expected_log_probs = torch.empty(args.tokens, 1, device="cuda")
    expected_gradient = torch.empty_like(logits)
    for chunk in (512, 1024, 2048):

        def operation(chunk_size=chunk):
            logits.grad = None
            log_probs, entropy = calculate_log_probs_and_entropy(
                logits,
                tokens,
                dist.group.WORLD,
                with_entropy=True,
                with_entropy_grad=False,
                chunk_size=chunk_size,
                log_prob_keep_mask=keep,
            )
            log_probs.mean().backward()
            return log_probs.detach(), entropy.detach()

        logits.grad = None
        torch.cuda.synchronize()
        starting_allocation = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        duration = measure(operation, args.repetitions)
        peak_allocation = torch.cuda.max_memory_allocated()
        log_probs, _ = operation()
        if chunk == 512:
            expected_log_probs.copy_(log_probs)
            expected_gradient.copy_(logits.grad)
        else:
            torch.testing.assert_close(log_probs, expected_log_probs, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(logits.grad, expected_gradient, atol=1e-8, rtol=1e-6)
        elapsed = torch.tensor(duration, device="cuda")
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        output.append(
            dict(
                chunk=chunk,
                max_rank_seconds=elapsed.item(),
                peak_allocated_gib=peak_allocation / 2**30,
                temporary_peak_gib=(peak_allocation - starting_allocation) / 2**30,
            )
        )
    dist.destroy_process_group()
    return {"loss": output, "tokens": args.tokens} if rank == 0 else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("mask", "loss"))
    parser.add_argument("--rollout", type=Path)
    parser.add_argument("--baseline-loss", type=Path)
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = benchmark_mask(arguments) if arguments.mode == "mask" else benchmark_loss(arguments)
    if result is not None:
        arguments.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
