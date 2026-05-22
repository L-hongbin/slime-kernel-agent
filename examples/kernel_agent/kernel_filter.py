import logging
from typing import Any

import torch

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
            "[cuda_agent][filter] config: reject_low_variance_groups=%s reject_small_groups=%s "
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
            "[cuda_agent][filter] drop group: valid_group_size=%s min_group_size=%s target_group_size=%s",
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
                "[cuda_agent][filter] drop group: reward_std=%.6g threshold=%.6g valid_group_size=%s rewards=%s",
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
