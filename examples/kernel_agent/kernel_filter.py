import json
import logging
from typing import Any

import torch
from torch.nn.utils.rnn import pad_sequence

from slime.backends.megatron_utils.cp_utils import all_gather_with_cp
from slime.rollout.filter_hub.base_types import DynamicFilterOutput
from slime.utils.types import Sample

try:
    from .config import CUDA_AGENT_CONFIGS
except ImportError:
    from config import CUDA_AGENT_CONFIGS

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
                "group_id": sample.group_id,
                # filter_reward is the pre-overlong-penalty task reward used
                # for the variance decision; reward is the effective reward
                # that would otherwise reach advantage computation.
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
        # Variance is judged on the PRE-PENALTY task reward when the overlong
        # penalty recorded one (metadata["task_reward"]): lengthy all-fail
        # groups must be dropped exactly as without the penalty; the penalty
        # only shapes advantages of groups that survive (user directive
        # 2026-07-18).
        rewards = [
            (
                sample.metadata.get("task_reward")
                if isinstance(sample.metadata, dict) and "task_reward" in sample.metadata
                else sample.get_reward_value(args)
            )
            for sample in valid_samples
        ]
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


def _get_sequence_mis_aggregation(args) -> str | None:
    aggregation = getattr(args, "sequence_mis_aggregation", None)
    if aggregation is None:
        return None
    if aggregation not in {"kl", "geometric", "mirrorpop", "turns_geometric", "turns_mirrorpop"}:
        raise ValueError(
            "[kernel_agent][sequence_mis] aggregation must be one of "
            f"['kl', 'geometric', 'mirrorpop', 'turns_geometric', 'turns_mirrorpop'], got {aggregation!r}."
        )
    return aggregation


def _get_sequence_mis_group_size(args, aggregation: str | None) -> int | None:
    if aggregation not in {"turns_geometric", "turns_mirrorpop"}:
        return None
    group_size = int(getattr(args, "sequence_mis_group_size", None) or getattr(args, "n_samples_per_prompt", 1) or 1)
    if group_size <= 0:
        raise ValueError(f"[kernel_agent][sequence_mis] group_size must be positive, got {group_size}.")
    return group_size


def _get_sequence_mis_batch_size(args, aggregation: str | None, max_turns: int | None, group_size: int | None) -> int:
    batch_size = int(getattr(args, "sequence_mis_batch_size", 8) or 8)
    if batch_size <= 0:
        raise ValueError(f"[kernel_agent][sequence_mis] sequence_mis_batch_size must be positive, got {batch_size}.")
    if aggregation in {"turns_geometric", "turns_mirrorpop"}:
        assert max_turns is not None
        if batch_size % max_turns != 0:
            batch_size = ((batch_size + max_turns - 1) // max_turns) * max_turns
    return batch_size


def _has_positive_advantage(advantage: torch.Tensor | None, mask: torch.Tensor) -> bool:
    if advantage is None:
        return False
    advantage = advantage.float()
    if advantage.shape == mask.shape:
        return bool(((advantage > 0) & mask.bool()).any().item())
    return bool((advantage > 0).any().item())


def _sequence_mis_mask_and_log_ratios(
    *,
    index: int,
    full_train_log_prob: torch.Tensor,
    full_rollout_log_prob: torch.Tensor,
    loss_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if full_train_log_prob.shape != full_rollout_log_prob.shape:
        raise ValueError(
            "[kernel_agent][sequence_mis] log_prob shape mismatch at sample "
            f"{index}: train={tuple(full_train_log_prob.shape)}, rollout={tuple(full_rollout_log_prob.shape)}."
        )
    if full_train_log_prob.shape != loss_mask.shape:
        raise ValueError(
            "[kernel_agent][sequence_mis] loss_mask shape mismatch at sample "
            f"{index}: log_prob={tuple(full_train_log_prob.shape)}, loss_mask={tuple(loss_mask.shape)}."
        )
    mask = loss_mask.float()
    return mask, (full_train_log_prob.float() - full_rollout_log_prob.float()) * mask


def _apply_mis(
    *,
    loss_masks: list[torch.Tensor],
    index: int,
    mask: torch.Tensor,
    sequence_value: torch.Tensor | None,
    is_token_veto: bool,
    advantage_protected: bool,
    lower_bound: float,
    upper_bound: float,
    stats: dict[str, float],
) -> None:
    if mask.sum() <= 0:
        return

    stats["valid_sequences"] += 1
    if sequence_value is None:
        if advantage_protected:
            stats["advantage_protected"] += 1
        elif is_token_veto:
            loss_masks[index] = torch.zeros_like(loss_masks[index])
            stats["rejected"] += 1
        return

    ratio_value = float(sequence_value.item())
    stats["ratio_sum"] += ratio_value
    stats["min_ratio"] = min(stats["min_ratio"], ratio_value)
    stats["max_ratio"] = max(stats["max_ratio"], ratio_value)

    if advantage_protected:
        stats["advantage_protected"] += 1
    elif is_token_veto or ratio_value < lower_bound or ratio_value > upper_bound:
        loss_masks[index] = torch.zeros_like(loss_masks[index])
        stats["rejected"] += 1


def _gather_sequence_mis_chunk(
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    advantages: list[torch.Tensor] | None,
    total_lengths: list[int],
    response_lengths: list[int],
    qkv_format: str,
    max_seq_lens: list[int] | None,
    start: int,
    end: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[bool]]:
    log_ratios = []
    masks = []
    advantage_protected_flags = []
    for i in range(start, end):
        max_seq_len = None if max_seq_lens is None else int(max_seq_lens[i])
        full_train_log_prob = all_gather_with_cp(
            train_log_probs[i],
            int(total_lengths[i]),
            int(response_lengths[i]),
            qkv_format=qkv_format,
            max_seq_len=max_seq_len,
        )
        full_rollout_log_prob = all_gather_with_cp(
            rollout_log_probs[i],
            int(total_lengths[i]),
            int(response_lengths[i]),
            qkv_format=qkv_format,
            max_seq_len=max_seq_len,
        )

        mask, sample_log_ratios = _sequence_mis_mask_and_log_ratios(
            index=i,
            full_train_log_prob=full_train_log_prob,
            full_rollout_log_prob=full_rollout_log_prob,
            loss_mask=loss_masks[i],
        )
        masks.append(mask)
        log_ratios.append(sample_log_ratios)
        advantage = None if advantages is None else advantages[i]
        advantage_protected_flags.append(_has_positive_advantage(advantage, mask))
    return log_ratios, masks, advantage_protected_flags


# Per-token |train_log_prob - rollout_log_prob| exceedance buckets. These measure
# the raw train/rollout logits mismatch magnitude independent of the MIS bounds, so
# they stay comparable when the bounds or aggregation change (e.g. the fp32-lm_head
# experiment that aims to shrink this distribution).
_MISMATCH_EXCEED_THRESHOLDS = (0.02, 0.05)


def _accumulate_mismatch_stats(stats: dict[str, float], log_ratios: torch.Tensor, mask: torch.Tensor) -> None:
    """Accumulate raw per-token log-prob mismatch magnitude over valid tokens.

    ``log_ratios`` is already zeroed outside ``mask`` by every caller, so padded /
    un-masked positions contribute 0 to the sum and never trip the exceedance
    counts (all thresholds are > 0). Works for both 1-D (loop) and 2-D padded
    (batch) tensors.
    """

    valid_count = float(mask.sum().item())
    if valid_count == 0:
        return
    abs_log_ratios = log_ratios.abs()
    stats["abs_log_ratio_sum"] += float(abs_log_ratios.sum().item())
    stats["valid_tokens"] += valid_count
    stats["abs_log_ratio_max"] = max(stats["abs_log_ratio_max"], float(abs_log_ratios.max().item()))
    for threshold in _MISMATCH_EXCEED_THRESHOLDS:
        stats[f"tok_exceed_{threshold}"] += float((abs_log_ratios > threshold).sum().item())


def _loop_sequence_mis(
    *,
    aggregation: str | None,
    max_turns: int | None,
    group_size: int | None,
    lower_bound: float,
    upper_bound: float,
    log_veto_threshold: torch.Tensor | None,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    advantages: list[torch.Tensor] | None,
    total_lengths: list[int],
    response_lengths: list[int],
    qkv_format: str,
    max_seq_lens: list[int] | None,
    stats: dict[str, float],
) -> None:
    temp_turns: list[tuple[int, torch.Tensor, torch.Tensor, bool, bool]] = []
    temp_log_ratio_sum = None
    temp_abs_log_ratio_sum = None
    temp_valid_token_count = None

    with torch.no_grad():
        for i in range(len(train_log_probs)):
            max_seq_len = None if max_seq_lens is None else int(max_seq_lens[i])
            full_train_log_prob = all_gather_with_cp(
                train_log_probs[i],
                int(total_lengths[i]),
                int(response_lengths[i]),
                qkv_format=qkv_format,
                max_seq_len=max_seq_len,
            )
            full_rollout_log_prob = all_gather_with_cp(
                rollout_log_probs[i],
                int(total_lengths[i]),
                int(response_lengths[i]),
                qkv_format=qkv_format,
                max_seq_len=max_seq_len,
            )

            mask, log_ratios = _sequence_mis_mask_and_log_ratios(
                index=i,
                full_train_log_prob=full_train_log_prob,
                full_rollout_log_prob=full_rollout_log_prob,
                loss_mask=loss_masks[i],
            )
            _accumulate_mismatch_stats(stats, log_ratios, mask)
            valid_token_count = torch.clamp_min(mask.sum(), 1)
            is_token_veto = bool(
                log_veto_threshold is not None and ((log_ratios < log_veto_threshold) & mask.bool()).any().item()
            )
            advantage = None if advantages is None else advantages[i]
            advantage_protected = _has_positive_advantage(advantage, mask)

            if aggregation == "kl":
                sequence_value = -log_ratios.sum() / valid_token_count
                _apply_mis(
                    loss_masks=loss_masks,
                    index=i,
                    mask=mask,
                    sequence_value=sequence_value,
                    is_token_veto=is_token_veto,
                    advantage_protected=advantage_protected,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    stats=stats,
                )
            elif aggregation == "geometric":
                sequence_value = torch.exp(torch.clamp(log_ratios.sum() / valid_token_count, min=-20.0, max=20.0))
                _apply_mis(
                    loss_masks=loss_masks,
                    index=i,
                    mask=mask,
                    sequence_value=sequence_value,
                    is_token_veto=is_token_veto,
                    advantage_protected=advantage_protected,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    stats=stats,
                )
            elif aggregation == "mirrorpop":
                sequence_value = log_ratios.abs().sum() / valid_token_count
                _apply_mis(
                    loss_masks=loss_masks,
                    index=i,
                    mask=mask,
                    sequence_value=sequence_value,
                    is_token_veto=is_token_veto,
                    advantage_protected=advantage_protected,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    stats=stats,
                )
            elif aggregation in {"turns_geometric", "turns_mirrorpop"}:
                assert max_turns is not None
                temp_turns.append((i, mask, log_ratios, is_token_veto, advantage_protected))
                temp_log_ratio_sum = (
                    log_ratios.sum() if temp_log_ratio_sum is None else temp_log_ratio_sum + log_ratios.sum()
                )
                temp_abs_log_ratio_sum = (
                    log_ratios.abs().sum()
                    if temp_abs_log_ratio_sum is None
                    else temp_abs_log_ratio_sum + log_ratios.abs().sum()
                )
                temp_valid_token_count = (
                    mask.sum() if temp_valid_token_count is None else temp_valid_token_count + mask.sum()
                )
                if len(temp_turns) == max_turns:
                    valid_token_count = torch.clamp_min(temp_valid_token_count, 1)
                    if aggregation == "turns_mirrorpop":
                        group_value = temp_abs_log_ratio_sum / valid_token_count
                    else:
                        group_value = torch.exp(
                            torch.clamp(
                                temp_log_ratio_sum / valid_token_count,
                                min=-20.0,
                                max=20.0,
                            )
                        )
                    for index, turn_mask, _turn_log_ratios, turn_is_token_veto, turn_advantage_protected in temp_turns:
                        _apply_mis(
                            loss_masks=loss_masks,
                            index=index,
                            mask=turn_mask,
                            sequence_value=group_value,
                            is_token_veto=turn_is_token_veto,
                            advantage_protected=turn_advantage_protected,
                            lower_bound=lower_bound,
                            upper_bound=upper_bound,
                            stats=stats,
                        )
                    temp_turns = []
                    temp_log_ratio_sum = None
                    temp_abs_log_ratio_sum = None
                    temp_valid_token_count = None
            elif aggregation is None:
                _apply_mis(
                    loss_masks=loss_masks,
                    index=i,
                    mask=mask,
                    sequence_value=None,
                    is_token_veto=is_token_veto,
                    advantage_protected=advantage_protected,
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                    stats=stats,
                )
            else:
                raise ValueError(f"[kernel_agent][sequence_mis] unknown aggregation: {aggregation}")


def _batch_sequence_mis(
    *,
    aggregation: str | None,
    max_turns: int | None,
    group_size: int | None,
    lower_bound: float,
    upper_bound: float,
    log_veto_threshold: torch.Tensor | None,
    batch_size: int,
    train_log_probs: list[torch.Tensor],
    rollout_log_probs: list[torch.Tensor],
    loss_masks: list[torch.Tensor],
    advantages: list[torch.Tensor] | None,
    total_lengths: list[int],
    response_lengths: list[int],
    qkv_format: str,
    max_seq_lens: list[int] | None,
    stats: dict[str, float],
) -> None:
    with torch.no_grad():
        for start in range(0, len(train_log_probs), batch_size):
            end = min(start + batch_size, len(train_log_probs))
            log_ratios, masks, advantage_protected_flags = _gather_sequence_mis_chunk(
                train_log_probs,
                rollout_log_probs,
                loss_masks,
                advantages,
                total_lengths,
                response_lengths,
                qkv_format,
                max_seq_lens,
                start,
                end,
            )
            log_ratios_padded = pad_sequence(log_ratios, batch_first=True, padding_value=0)
            masks_padded = pad_sequence(masks, batch_first=True, padding_value=0)
            _accumulate_mismatch_stats(stats, log_ratios_padded, masks_padded)
            valid_token_counts = torch.clamp_min(masks_padded.sum(dim=1), 1)
            is_token_veto = torch.zeros(len(log_ratios), dtype=torch.bool, device=masks_padded.device)
            if log_veto_threshold is not None:
                is_token_veto = ((log_ratios_padded < log_veto_threshold) & masks_padded.bool()).any(dim=1)

            if aggregation == "kl":
                sequence_values = -log_ratios_padded.sum(dim=1) / valid_token_counts
            elif aggregation == "geometric":
                sequence_values = torch.exp(
                    torch.clamp(log_ratios_padded.sum(dim=1) / valid_token_counts, min=-20.0, max=20.0)
                )
            elif aggregation == "mirrorpop":
                sequence_values = log_ratios_padded.abs().sum(dim=1) / valid_token_counts
            elif aggregation in {"turns_geometric", "turns_mirrorpop"}:
                assert max_turns is not None
                if len(log_ratios) % max_turns != 0:
                    raise ValueError(
                        "[kernel_agent][sequence_mis] internal batch for turns_geometric or turns_mirrorpop is incomplete: "
                        f"batch_size={len(log_ratios)}, max_turns={max_turns}."
                    )
                trajectory_count = len(log_ratios) // max_turns
                grouped_log_ratios = log_ratios_padded.reshape(trajectory_count, max_turns, log_ratios_padded.size(1))
                grouped_masks = masks_padded.reshape(trajectory_count, max_turns, masks_padded.size(1))
                grouped_valid_token_counts = torch.clamp_min(grouped_masks.sum(dim=(1, 2)), 1)
                if aggregation == "turns_mirrorpop":
                    group_values = grouped_log_ratios.abs().sum(dim=(1, 2)) / grouped_valid_token_counts
                else:
                    group_values = torch.exp(
                        torch.clamp(
                            grouped_log_ratios.sum(dim=(1, 2)) / grouped_valid_token_counts,
                            min=-20.0,
                            max=20.0,
                        )
                    )
                sequence_values = group_values.unsqueeze(1).expand(trajectory_count, max_turns).reshape(-1)
            elif aggregation is None:
                sequence_values = [None] * len(log_ratios)
            else:
                raise ValueError(f"[kernel_agent][sequence_mis] unknown aggregation: {aggregation}")

            valid_sequences = masks_padded.sum(dim=1) > 0
            stats["valid_sequences"] += int(valid_sequences.sum().item())
            advantage_protected = torch.tensor(advantage_protected_flags, dtype=torch.bool, device=masks_padded.device)
            valid_advantage_protected = valid_sequences & advantage_protected
            stats["advantage_protected"] += int(valid_advantage_protected.sum().item())

            if aggregation is None:
                rejected = valid_sequences & ~advantage_protected & is_token_veto
            else:
                ratio_stats_mask = valid_sequences
                if ratio_stats_mask.any():
                    valid_sequence_values = sequence_values[ratio_stats_mask]
                    stats["ratio_sum"] += float(valid_sequence_values.sum().item())
                    stats["min_ratio"] = min(stats["min_ratio"], float(valid_sequence_values.min().item()))
                    stats["max_ratio"] = max(stats["max_ratio"], float(valid_sequence_values.max().item()))

                ratio_rejected = (sequence_values < lower_bound) | (sequence_values > upper_bound)
                rejected = valid_sequences & ~advantage_protected & (is_token_veto | ratio_rejected)

            rejected_offsets = torch.nonzero(rejected, as_tuple=False).flatten().tolist()
            stats["rejected"] += len(rejected_offsets)
            for offset in rejected_offsets:
                loss_masks[start + offset] = torch.zeros_like(loss_masks[start + offset])


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

    aggregation = _get_sequence_mis_aggregation(args)
    lower_bound = getattr(args, "sequence_mis_lower", None)
    lower_bound = float("-inf") if lower_bound is None else float(lower_bound)
    upper_bound = getattr(args, "sequence_mis_upper", None)
    upper_bound = float("inf") if upper_bound is None else float(upper_bound)

    if lower_bound >= upper_bound:
        raise ValueError(
            "[kernel_agent][sequence_mis] invalid bounds: " f"lower_bound={lower_bound}, upper_bound={upper_bound}."
        )

    token_veto_threshold = getattr(args, "sequence_mis_token_veto_threshold", None)
    token_veto_threshold = None if token_veto_threshold is None else float(token_veto_threshold)
    if token_veto_threshold is not None and token_veto_threshold <= 0:
        raise ValueError(
            "[kernel_agent][sequence_mis] token veto threshold must be positive, " f"got {token_veto_threshold}."
        )

    if aggregation in {"turns_geometric", "turns_mirrorpop"}:
        max_turns = getattr(args, "max_turns", None)
        if max_turns is None:
            raise ValueError(
                "[kernel_agent][sequence_mis] --max-turns must be set for turns_geometric or turns_mirrorpop aggregation."
            )
        max_turns = int(max_turns)
        if max_turns <= 0:
            raise ValueError(f"[kernel_agent][sequence_mis] --max-turns must be positive, got {max_turns}.")
    else:
        max_turns = None
    group_size = _get_sequence_mis_group_size(args, aggregation)

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

    if max_turns is not None and len(train_log_probs) % max_turns != 0:
        raise ValueError(
            "[kernel_agent][sequence_mis] turns_geometric and turns_mirrorpop require complete trajectory-major groups: "
            f"got {len(train_log_probs)} samples, max_turns={max_turns}. "
            "For multi-turn rollout, consider enabling --filter-by-last-turn and --padding-turns."
        )

    log_veto_threshold = None
    if token_veto_threshold is not None:
        log_veto_threshold = torch.log(
            torch.tensor(token_veto_threshold, device=loss_masks[0].device, dtype=torch.float32)
        )

    if aggregation is None:
        logger.warning(
            "[kernel_agent][sequence_mis] sequence_mis_aggregation is not set; only token veto will be applied."
        )

    stats = {
        "rejected": 0.0,
        "valid_sequences": 0.0,
        "ratio_sum": 0.0,
        "min_ratio": float("inf"),
        "max_ratio": 0.0,
        "advantage_protected": 0.0,
        "abs_log_ratio_sum": 0.0,
        "valid_tokens": 0.0,
        "abs_log_ratio_max": 0.0,
    }
    for threshold in _MISMATCH_EXCEED_THRESHOLDS:
        stats[f"tok_exceed_{threshold}"] = 0.0
    mode = getattr(args, "sequence_mis_mode", "batch") or "batch"
    if mode == "loop":
        _loop_sequence_mis(
            aggregation=aggregation,
            max_turns=max_turns,
            group_size=group_size,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            log_veto_threshold=log_veto_threshold,
            train_log_probs=train_log_probs,
            rollout_log_probs=rollout_log_probs,
            loss_masks=loss_masks,
            advantages=advantages,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            qkv_format=qkv_format,
            max_seq_lens=max_seq_lens,
            stats=stats,
        )
    elif mode == "batch":
        _batch_sequence_mis(
            aggregation=aggregation,
            max_turns=max_turns,
            group_size=group_size,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            log_veto_threshold=log_veto_threshold,
            batch_size=_get_sequence_mis_batch_size(args, aggregation, max_turns, group_size),
            train_log_probs=train_log_probs,
            rollout_log_probs=rollout_log_probs,
            loss_masks=loss_masks,
            advantages=advantages,
            total_lengths=total_lengths,
            response_lengths=response_lengths,
            qkv_format=qkv_format,
            max_seq_lens=max_seq_lens,
            stats=stats,
        )
    else:
        raise ValueError(f"[kernel_agent][sequence_mis] sequence_mis_mode must be 'loop' or 'batch', got {mode!r}.")

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
        for threshold in _MISMATCH_EXCEED_THRESHOLDS
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
