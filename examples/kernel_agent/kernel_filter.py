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
            reason=f"group_size_lt_min_{len(valid_samples)}_{min_group_size}",
        )

    if reject_low_variance_groups:
        rewards = [sample.get_reward_value(args) for sample in valid_samples]
        reward_std = torch.tensor(rewards, dtype=torch.float64).std(unbiased=False).item()
        if reward_std < reward_std_threshold:
            logger.info(
                "[kernel_agent][filter] drop group: reward_std=%.6g threshold=%.6g valid_group_size=%s rewards=%s",
                reward_std,
                reward_std_threshold,
                len(valid_samples),
                rewards,
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
    if aggregation not in {"kl", "geometric", "turns_geometric"}:
        raise ValueError(
            "[kernel_agent][sequence_mis] aggregation must be one of "
            f"['kl', 'geometric', 'turns_geometric'], got {aggregation!r}."
        )
    return aggregation


def _get_sequence_mis_group_size(args, aggregation: str | None) -> int | None:
    if aggregation != "turns_geometric":
        return None
    group_size = int(getattr(args, "sequence_mis_group_size", None) or getattr(args, "n_samples_per_prompt", 1) or 1)
    if group_size <= 0:
        raise ValueError(f"[kernel_agent][sequence_mis] group_size must be positive, got {group_size}.")
    return group_size


def _get_sequence_mis_batch_size(args, aggregation: str | None, max_turns: int | None, group_size: int | None) -> int:
    batch_size = int(getattr(args, "sequence_mis_batch_size", 8) or 8)
    if batch_size <= 0:
        raise ValueError(f"[kernel_agent][sequence_mis] sequence_mis_batch_size must be positive, got {batch_size}.")
    if aggregation == "turns_geometric":
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
    start: int,
    end: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[bool]]:
    log_ratios = []
    masks = []
    advantage_protected_flags = []
    for i in range(start, end):
        full_train_log_prob = all_gather_with_cp(train_log_probs[i], int(total_lengths[i]), int(response_lengths[i]))
        full_rollout_log_prob = all_gather_with_cp(
            rollout_log_probs[i], int(total_lengths[i]), int(response_lengths[i])
        )

        if full_train_log_prob.shape != full_rollout_log_prob.shape:
            raise ValueError(
                "[kernel_agent][sequence_mis] log_prob shape mismatch at sample "
                f"{i}: train={tuple(full_train_log_prob.shape)}, rollout={tuple(full_rollout_log_prob.shape)}."
            )
        if full_train_log_prob.shape != loss_masks[i].shape:
            raise ValueError(
                "[kernel_agent][sequence_mis] loss_mask shape mismatch at sample "
                f"{i}: log_prob={tuple(full_train_log_prob.shape)}, loss_mask={tuple(loss_masks[i].shape)}."
            )

        mask = loss_masks[i].float()
        masks.append(mask)
        log_ratios.append((full_train_log_prob.float() - full_rollout_log_prob.float()) * mask)
        advantage = None if advantages is None else advantages[i]
        advantage_protected_flags.append(_has_positive_advantage(advantage, mask))
    return log_ratios, masks, advantage_protected_flags


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
    stats: dict[str, float],
) -> None:
    temp_turns: list[tuple[int, torch.Tensor, torch.Tensor, bool, bool]] = []
    temp_log_ratio_sum = None
    temp_valid_token_count = None

    with torch.no_grad():
        for i in range(len(train_log_probs)):
            full_train_log_prob = all_gather_with_cp(
                train_log_probs[i], int(total_lengths[i]), int(response_lengths[i])
            )
            full_rollout_log_prob = all_gather_with_cp(
                rollout_log_probs[i], int(total_lengths[i]), int(response_lengths[i])
            )

            if full_train_log_prob.shape != full_rollout_log_prob.shape:
                raise ValueError(
                    "[kernel_agent][sequence_mis] log_prob shape mismatch at sample "
                    f"{i}: train={tuple(full_train_log_prob.shape)}, rollout={tuple(full_rollout_log_prob.shape)}."
                )
            if full_train_log_prob.shape != loss_masks[i].shape:
                raise ValueError(
                    "[kernel_agent][sequence_mis] loss_mask shape mismatch at sample "
                    f"{i}: log_prob={tuple(full_train_log_prob.shape)}, loss_mask={tuple(loss_masks[i].shape)}."
                )

            mask = loss_masks[i].float()
            log_ratios = (full_train_log_prob.float() - full_rollout_log_prob.float()) * mask
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
            elif aggregation == "turns_geometric":
                assert max_turns is not None
                temp_turns.append((i, mask, log_ratios, is_token_veto, advantage_protected))
                temp_log_ratio_sum = (
                    log_ratios.sum() if temp_log_ratio_sum is None else temp_log_ratio_sum + log_ratios.sum()
                )
                temp_valid_token_count = (
                    mask.sum() if temp_valid_token_count is None else temp_valid_token_count + mask.sum()
                )
                if len(temp_turns) == max_turns:
                    group_value = torch.exp(
                        torch.clamp(
                            temp_log_ratio_sum / torch.clamp_min(temp_valid_token_count, 1),
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
                start,
                end,
            )
            log_ratios_padded = pad_sequence(log_ratios, batch_first=True, padding_value=0)
            masks_padded = pad_sequence(masks, batch_first=True, padding_value=0)
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
            elif aggregation == "turns_geometric":
                assert max_turns is not None
                if len(log_ratios) % max_turns != 0:
                    raise ValueError(
                        "[kernel_agent][sequence_mis] internal batch for turns_geometric is incomplete: "
                        f"batch_size={len(log_ratios)}, max_turns={max_turns}."
                    )
                trajectory_count = len(log_ratios) // max_turns
                grouped_log_ratios = log_ratios_padded.reshape(trajectory_count, max_turns, log_ratios_padded.size(1))
                grouped_masks = masks_padded.reshape(trajectory_count, max_turns, masks_padded.size(1))
                grouped_valid_token_counts = torch.clamp_min(grouped_masks.sum(dim=(1, 2)), 1)
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


def sequence_mis(args, rollout_id: int, rollout_data: dict[str, Any]) -> None:
    """Apply sequence-level mask importance sampling before dynamic micro-batching.

    This hook is intended for ``--rollout-data-postprocess-path``. It uses the actor
    log-probs recomputed by Megatron and the rollout log-probs from SGLang, then
    masks whole response sequences whose importance ratio is outside the configured
    bounds. Samples stay in the batch; only their ``loss_masks`` are zeroed.
    """

    if "log_probs" not in rollout_data:
        logger.info(
            "[kernel_agent][sequence_mis] skip rollout_id=%s because rollout_data['log_probs'] is unavailable "
            "on this rank.",
            rollout_id,
        )
        return
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

    if aggregation == "turns_geometric":
        max_turns = getattr(args, "max_turns", None)
        if max_turns is None:
            raise ValueError("[kernel_agent][sequence_mis] --max-turns must be set for turns_geometric aggregation.")
        max_turns = int(max_turns)
        if max_turns <= 0:
            raise ValueError(f"[kernel_agent][sequence_mis] --max-turns must be positive, got {max_turns}.")
    else:
        max_turns = None
    group_size = _get_sequence_mis_group_size(args, aggregation)

    train_log_probs = rollout_data["log_probs"]
    rollout_log_probs = rollout_data["rollout_log_probs"]
    loss_masks = rollout_data["loss_masks"]
    total_lengths = rollout_data["total_lengths"]
    response_lengths = rollout_data["response_lengths"]
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

    if not (len(train_log_probs) == len(rollout_log_probs) == len(loss_masks) == len(response_lengths)):
        raise ValueError(
            "[kernel_agent][sequence_mis] rollout_data length mismatch: "
            f"log_probs={len(train_log_probs)}, rollout_log_probs={len(rollout_log_probs)}, "
            f"loss_masks={len(loss_masks)}, response_lengths={len(response_lengths)}."
        )

    if max_turns is not None and len(train_log_probs) % max_turns != 0:
        raise ValueError(
            "[kernel_agent][sequence_mis] turns_geometric requires complete trajectory-major groups: "
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
    }
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
            stats=stats,
        )
    else:
        raise ValueError(f"[kernel_agent][sequence_mis] sequence_mis_mode must be 'loop' or 'batch', got {mode!r}.")

    rollout_data["loss_masks"] = loss_masks
    mean_ratio = stats["ratio_sum"] / max(stats["valid_sequences"], 1)
    logger.info(
        "[kernel_agent][sequence_mis] rollout_id=%s mode=%s rejected=%s/%s reject_rate=%.6f "
        "advantage_protected=%s ratio_mean=%.6g ratio_min=%.6g ratio_max=%.6g",
        rollout_id,
        mode,
        int(stats["rejected"]),
        int(stats["valid_sequences"]),
        stats["rejected"] / max(stats["valid_sequences"], 1),
        int(stats["advantage_protected"]),
        mean_ratio,
        0.0 if stats["min_ratio"] == float("inf") else stats["min_ratio"],
        stats["max_ratio"],
    )
