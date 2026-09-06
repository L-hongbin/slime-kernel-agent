import logging
import time
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def capture_replay_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    """Before-train-step hook for fixed-replay gradient/logprob comparisons.

    Captures the optimizer's norm-contributing gradients after its step (and
    clipping), before train_one_step clears buffers. Compute timing excludes
    the CPU copies and serialization. This opt-in diagnostic can write tens
    of GiB per rank and must only run with an explicit train dump destination.
    """
    if not args.debug_train_only or not args.save_debug_train_data:
        raise ValueError("capture_replay_step requires --debug-train-only and --save-debug-train-data")
    from slime.backends.megatron_utils import loss as loss_module

    rank = torch.distributed.get_rank()
    path = Path(args.save_debug_train_data.format(rollout_id=rollout_id, rank=rank))
    path = path.with_name(f"{path.stem}_step{step_id}_capture.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    original_logprobs = loss_module.get_log_probs_and_entropy
    original_step = optimizer.step
    captures = []

    def capture_logprobs(*positional, **kwargs):
        result = original_logprobs(*positional, **kwargs)
        # Detached response-sized tensors retain no activation graph. Delay
        # their host transfer until after the timed compute window.
        captures.append(
            {
                "total_lengths": kwargs["total_lengths"],
                "response_lengths": kwargs["response_lengths"],
                "log_probs": [lp.detach() for lp in result[1]["log_probs"]],
            }
        )
        return result

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    def capture_step(*positional, **kwargs):
        try:
            result = original_step(*positional, **kwargs)
            torch.cuda.synchronize()
            compute_seconds = time.perf_counter() - started
            peak_allocated_bytes = torch.cuda.max_memory_allocated()
            peak_reserved_bytes = torch.cuda.max_memory_reserved()
            gradients = [g.detach().cpu().clone() for g in optimizer.get_main_grads_for_grad_norm()]
            for capture in captures:
                capture["log_probs"] = [lp.cpu().clone() for lp in capture["log_probs"]]
            torch.save(
                {
                    "rollout_id": rollout_id,
                    "step_id": step_id,
                    "rank": rank,
                    "compute_seconds": compute_seconds,
                    "peak_allocated_bytes": peak_allocated_bytes,
                    "peak_reserved_bytes": peak_reserved_bytes,
                    "gradients": gradients,
                    "gradient_stage": "post_optimizer_step_before_buffer_clear",
                    "logprob_microbatches": captures,
                },
                path,
            )
            logger.info(
                "REPLAY_CAPTURE rank=%s seconds=%.3f grad_elements=%s path=%s",
                rank,
                compute_seconds,
                sum(g.numel() for g in gradients),
                path,
            )
            return result
        finally:
            optimizer.step = original_step
            loss_module.get_log_probs_and_entropy = original_logprobs

    loss_module.get_log_probs_and_entropy = capture_logprobs
    optimizer.step = capture_step


def save_debug_train_data(args, *, rollout_id, rollout_data):
    if (path_template := args.save_debug_train_data) is not None:
        rank = torch.distributed.get_rank()
        path = Path(path_template.format(rollout_id=rollout_id, rank=rank))
        logger.info(f"Save debug train data to {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            dict(
                rollout_id=rollout_id,
                rank=rank,
                rollout_data=rollout_data,
            ),
            path,
        )
