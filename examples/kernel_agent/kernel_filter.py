import json
import logging
from typing import Any

import torch

from slime.backends.megatron_utils.cp_utils import all_gather_with_cp
from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.sequence_mis import MISMATCH_EXCEED_THRESHOLDS, compute_sequence_mis
from slime.utils.types import Sample

try:
    from .config import CUDA_AGENT_CONFIGS
    from .kernel_reward import get_kernel_group_filter_rewards
except ImportError:
    from config import CUDA_AGENT_CONFIGS
    from kernel_reward import get_kernel_group_filter_rewards

logger = logging.getLogger(__name__)
_FILTER_CONFIG_LOGGED = False


def _low_variance_audit_record(args, samples: list[Sample], filter_rewards: list[float]) -> dict[str, Any]:
    """Build one structured record that joins filtered sample IDs to rewards."""

    def metadata_for(sample: Sample) -> dict[str, Any]:
        return sample.metadata if isinstance(sample.metadata, dict) else {}

    def stable_id(sample: Sample) -> str | int | None:
        metadata = metadata_for(sample)
        for key in ("uuid", "uid", "id", "index", "source_row"):
            value = metadata.get(key)
            if value is not None:
                return value if isinstance(value, (str, int)) else str(value)
        return sample.index

    first = samples[0]
    first_metadata = metadata_for(first)
    return {
        "rollout_step": first_metadata.get("start_rollout_id"),
        "prompt_group_index": first.group_index,
        "samples": [
            {
                "id": stable_id(sample),
                "sample_index": sample.index,
                "rollout_id": sample.rollout_id if sample.rollout_id is not None else sample.index,
                # Both fields are settled before filtering. LASER-D filters
                # with the length bonus; other modes prefer task_reward.
                "filter_reward": float(filter_reward),
                "reward": float(sample.get_reward_value(args)),
            }
            for sample, filter_reward in zip(samples, filter_rewards, strict=True)
        ],
    }


def filter_cuda_kernel_group(args, samples: list[Sample], **kwargs: Any) -> DynamicFilterOutput:
    global _FILTER_CONFIG_LOGGED

    filter_config = CUDA_AGENT_CONFIGS.get("filter", {})
    reject_low_variance_groups = bool(filter_config.get("reject_low_variance_groups", True))
    reject_small_groups = bool(filter_config.get("reject_small_groups", True))

    target_group_size = getattr(args, "target_group_size", None) or filter_config.get("target_group_size")
    target_group_size = target_group_size or args.n_samples_per_prompt
    min_group_size = getattr(args, "min_group_size", None)
    if min_group_size is None:
        min_group_size = filter_config.get("min_group_size")

    if min_group_size is None:
        min_group_size = target_group_size // 2 + 1

    if reject_small_groups and min_group_size <= target_group_size // 2:
        min_required = target_group_size // 2 + 1
        padding_ratio = (target_group_size - min_group_size) / target_group_size * 100
        raise ValueError(
            f"min_group_size ({min_group_size}) must be > target_group_size // 2 ({target_group_size // 2}) "
            f"to avoid excessive padding overhead. Minimum required: {min_required}. "
            f"With min_group_size={min_group_size}, padding to target_group_size={target_group_size} "
            f"would result in >{padding_ratio:.0f}% padding per group."
        )

    reward_std_threshold = getattr(args, "reward_std_threshold", None)
    if reward_std_threshold is None:
        reward_std_threshold = filter_config.get("reward_std_threshold", 1e-3)
    reward_std_threshold = float(reward_std_threshold)

    if not _FILTER_CONFIG_LOGGED:
        logger.info(
            "[kernel_agent][filter] config: reject_low_variance_groups=%s reject_small_groups=%s "
            "target_group_size=%s min_group_size=%s reward_std_threshold=%s",
            reject_low_variance_groups,
            reject_small_groups,
            target_group_size,
            min_group_size,
            reward_std_threshold,
        )
        _FILTER_CONFIG_LOGGED = True

    valid_samples = [sample for sample in samples if not sample.remove_sample]
    if reject_small_groups and len(valid_samples) < min_group_size:
        logger.info(
            "[kernel_agent][filter] drop group: valid_group_size=%s min_group_size=%s target_group_size=%s",
            len(valid_samples),
            min_group_size,
            target_group_size,
        )
        return DynamicFilterOutput(
            keep=False,
            reason=f"group_size_lt_min_{min_group_size}",
        )

    if reject_low_variance_groups:
        # Keep DAPO's pre-penalty variance policy. LASER-D's correctness-gated
        # bonus is a learning signal even in all-correct groups and must survive
        # filtering; its collector has already written the per-sample bonus.
        rewards = get_kernel_group_filter_rewards(
            args, valid_samples, [sample.get_reward_value(args) for sample in valid_samples]
        )
        reward_std = torch.tensor(rewards, dtype=torch.float64).std(unbiased=False).item()
        if reward_std < reward_std_threshold:
            audit_record = _low_variance_audit_record(args, valid_samples, rewards)
            logger.info(
                "[kernel_agent][filter] drop group: reward_std=%.6g threshold=%.6g "
                "valid_group_size=%s rewards=%s audit=%s",
                reward_std,
                reward_std_threshold,
                len(valid_samples),
                rewards,
                json.dumps(audit_record, ensure_ascii=False, sort_keys=True, default=str),
            )
            return DynamicFilterOutput(
                keep=False,
                reason=f"reward_std_lt_{reward_std_threshold:g}",
            )

    return DynamicFilterOutput(keep=True)


def sequence_mis(args, rollout_id: int, rollout_data: dict[str, Any]) -> dict[str, float]:
    """Apply sequence-level mask importance sampling before dynamic micro-batching.

    This hook is intended for ``--rollout-data-postprocess-path``. It uses the actor
    log-probs recomputed by Megatron and the rollout log-probs from SGLang, then
    masks whole response sequences whose importance ratio is outside the configured
    bounds. Samples stay in the batch; only their ``loss_masks`` are zeroed.

    ``ratio_source`` (from ``--sequence-mis-config``) selects the ratio pair:
      - ``"rollout"`` (default): ``log_probs`` (megatron recompute) vs
        ``rollout_log_probs`` (sglang) — the train/infer cross-engine pair.
      - ``"old_actor"``: ``cur_log_probs`` (current-actor recompute) vs ``log_probs``
        (behavioral old-actor recompute) — a SAME-STACK drift ratio (both megatron),
        so cross-engine numerics don't drive rejection. Requires ``--keep-old-actor``
        and the train actor to have populated ``cur_log_probs`` (incompatible with
        routing replay; enforced in ``slime_validate_args``).
    """

    # Live-forward sample admission is performed in policy_loss_function. Keep
    # existing launchers with this hook compatible without filtering twice.
    aggregation = getattr(args, "sequence_mis_aggregation", None)
    if getattr(args, "sequence_mis_actor_logprob", "dynamic" if aggregation == "binary_kl" else "static") == "dynamic":
        return {}

    if "log_probs" not in rollout_data:
        logger.info(
            "[kernel_agent][sequence_mis] skip rollout_id=%s because rollout_data['log_probs'] is unavailable "
            "on this rank.",
            rollout_id,
        )
        return {}
    if "rollout_log_probs" not in rollout_data:
        raise ValueError(
            "sequence_mis requires rollout_data['rollout_log_probs']. "
            "Make sure the rollout function stores sample.rollout_log_probs."
        )

    ratio_source = getattr(args, "sequence_mis_ratio_source", "rollout")
    if ratio_source == "old_actor":
        if "cur_log_probs" not in rollout_data:
            raise ValueError(
                "[kernel_agent][sequence_mis] ratio_source='old_actor' requires "
                "rollout_data['cur_log_probs'] (the current-actor recompute). It is populated by "
                "the train actor only when --keep-old-actor takes the LoRA old-actor path."
            )
        # Same-stack drift ratio: current (numerator) vs old-actor (denominator), both megatron.
        train_log_probs = rollout_data["cur_log_probs"]
        rollout_log_probs = rollout_data["log_probs"]
    else:
        train_log_probs = rollout_data["log_probs"]
        rollout_log_probs = rollout_data["rollout_log_probs"]
    loss_masks = rollout_data["loss_masks"]
    total_lengths = rollout_data["total_lengths"]
    response_lengths = rollout_data["response_lengths"]
    qkv_format = getattr(args, "qkv_format", "thd")
    max_seq_lens = rollout_data.get("max_seq_lens")
    if qkv_format == "bshd":
        if max_seq_lens is None:
            raise ValueError(
                "[kernel_agent][sequence_mis] qkv_format='bshd' requires "
                "rollout_data['max_seq_lens'] so CP gather uses the forward layout."
            )
        if len(max_seq_lens) != len(response_lengths):
            raise ValueError(
                "[kernel_agent][sequence_mis] max_seq_lens length mismatch: "
                f"max_seq_lens={len(max_seq_lens)}, response_lengths={len(response_lengths)}."
            )
    use_advantage = bool(getattr(args, "sequence_mis_use_advantage", False))
    advantages = rollout_data.get("advantages") if use_advantage else None
    if use_advantage:
        assert (
            advantages is not None
        ), "[kernel_agent][sequence_mis] use_advantage requires rollout_data['advantages']."
        if len(advantages) != len(loss_masks):
            raise ValueError(
                "[kernel_agent][sequence_mis] advantages length mismatch: "
                f"advantages={len(advantages)}, loss_masks={len(loss_masks)}."
            )

    if not (
        len(train_log_probs)
        == len(rollout_log_probs)
        == len(loss_masks)
        == len(total_lengths)
        == len(response_lengths)
    ):
        raise ValueError(
            "[kernel_agent][sequence_mis] rollout_data length mismatch: "
            f"log_probs={len(train_log_probs)}, rollout_log_probs={len(rollout_log_probs)}, "
            f"loss_masks={len(loss_masks)}, total_lengths={len(total_lengths)}, "
            f"response_lengths={len(response_lengths)}."
        )

    full_actor, full_reference, full_advantages = [], [], []
    for i, (actor, reference) in enumerate(zip(train_log_probs, rollout_log_probs, strict=True)):
        layout = {"qkv_format": qkv_format, "max_seq_len": None if max_seq_lens is None else max_seq_lens[i]}
        full_actor.append(all_gather_with_cp(actor.detach(), total_lengths[i], response_lengths[i], **layout))
        full_reference.append(all_gather_with_cp(reference.detach(), total_lengths[i], response_lengths[i], **layout))
        if use_advantage:
            full_advantages.append(
                all_gather_with_cp(advantages[i].detach(), total_lengths[i], response_lengths[i], **layout)
            )
    result = compute_sequence_mis(
        args,
        full_actor,
        full_reference,
        loss_masks,
        full_advantages if use_advantage else None,
        collect_stats=True,
    )
    loss_masks, stats = result.loss_masks, result.stats
    mode = getattr(args, "sequence_mis_mode", "batch") or "batch"
    rollout_data["loss_masks"] = loss_masks
    valid_sequences = stats["valid_sequences"]
    rejected = stats["rejected"]
    mean_ratio = stats["ratio_sum"] / max(valid_sequences, 1)
    rollout_data["seq_mis/reject_rate"] = (rejected, valid_sequences)
    rollout_data["seq_mis/advantage_protected_rate"] = (stats["advantage_protected"], valid_sequences)
    rollout_data["seq_mis/ratio_mean"] = (stats["ratio_sum"], valid_sequences)

    # Post-MIS effective batch: sequences/tokens that still carry a non-zero
    # loss mask. ``effective_tokens == 0`` is exactly the silent no-op train step
    # (whole batch rejected -> loss/grad all zero), so surface it explicitly.
    valid_tokens = stats["valid_tokens"]
    reject_rate = rejected / max(valid_sequences, 1)
    effective_sequences = int(valid_sequences - rejected)
    effective_tokens = sum(float(m.sum().item()) for m in loss_masks)
    mean_abs_log_ratio = stats["abs_log_ratio_sum"] / max(valid_tokens, 1.0)
    exceed_fracs = {
        threshold: stats[f"tok_exceed_{threshold}"] / max(valid_tokens, 1.0)
        for threshold in MISMATCH_EXCEED_THRESHOLDS
    }
    logger.info(
        "[kernel_agent][sequence_mis] rollout_id=%s mode=%s rejected=%s/%s reject_rate=%.6f "
        "advantage_protected=%s ratio_mean=%.6g ratio_min=%.6g ratio_max=%.6g "
        "effective_sequences=%s effective_tokens=%s valid_tokens=%s "
        "mismatch_mean_abs_log_ratio=%.6g mismatch_max_abs_log_ratio=%.6g mismatch_exceed_frac=%s",
        rollout_id,
        mode,
        int(stats["rejected"]),
        int(stats["valid_sequences"]),
        reject_rate,
        int(stats["advantage_protected"]),
        mean_ratio,
        0.0 if stats["min_ratio"] == float("inf") else stats["min_ratio"],
        stats["max_ratio"],
        effective_sequences,
        int(effective_tokens),
        int(valid_tokens),
        mean_abs_log_ratio,
        stats["abs_log_ratio_max"],
        {f"|lr|>{threshold}": round(frac, 6) for threshold, frac in exceed_fracs.items()},
    )

    # Surface aggregate mismatch/rejection signals to the metric logger so the
    # fp32-lm_head experiment is comparable across runs. ``log_rollout_data``
    # treats 0-d tensors as per-(DP rank) scalars (count=1) and gathers them as a
    # mean across DP ranks, which is the right aggregation for these rates.
    if loss_masks:
        metric_device = loss_masks[0].device
        wandb_metrics = {
            "mis_reject_rate": reject_rate,
            "mis_effective_token_frac": effective_tokens / max(valid_tokens, 1.0),
            "mis_mean_abs_log_ratio": mean_abs_log_ratio,
            "mis_max_abs_log_ratio": stats["abs_log_ratio_max"],
        }
        for threshold, frac in exceed_fracs.items():
            wandb_metrics[f"mis_tok_frac_abs_log_ratio_gt_{threshold}"] = frac
        for name, value in wandb_metrics.items():
            rollout_data[name] = torch.tensor(float(value), device=metric_device)

    return stats
