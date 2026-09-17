import math
import random
from typing import Any

import torch

try:
    from .config import CUDA_AGENT_CONFIGS
    from .utils import (
        COMPILATION_ERROR,
        CORRECTNESS_ERROR,
        IMPORT_ERROR,
        KERNEL_EVAL_TIMEOUT,
        PRECHECK_ERROR,
        RUNTIME_ERROR,
        SYNTAX_ERROR,
        VALIDATION_ERROR,
    )
except ImportError:
    from config import CUDA_AGENT_CONFIGS
    from utils import (
        COMPILATION_ERROR,
        CORRECTNESS_ERROR,
        IMPORT_ERROR,
        KERNEL_EVAL_TIMEOUT,
        PRECHECK_ERROR,
        RUNTIME_ERROR,
        SYNTAX_ERROR,
        VALIDATION_ERROR,
    )


def _timing_cv_and_trials(metadata: dict[str, Any], prefix: str) -> tuple[float, int]:
    mean = metadata.get(f"{prefix}_mean_ms")
    std = metadata.get(f"{prefix}_std_ms")
    num_trials = metadata.get(f"{prefix}_num_trials")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in (mean, std, num_trials)):
        raise ValueError(f"{prefix} timing metadata must include numeric mean_ms, std_ms, and num_trials")

    mean = float(mean)
    std = float(std)
    num_trials = int(num_trials)
    if not math.isfinite(mean) or mean <= 0.0:
        raise ValueError(f"{prefix}_mean_ms must be positive and finite")
    if not math.isfinite(std) or std < 0.0:
        raise ValueError(f"{prefix}_std_ms must be non-negative and finite")
    if num_trials <= 0:
        raise ValueError(f"{prefix}_num_trials must be positive")
    return std / mean, num_trials


def _compute_speedup_log_standard_error(metadata: dict[str, Any] | None, config: dict[str, Any]) -> float:
    if not isinstance(metadata, dict):
        raise ValueError("lcb_improvement requires timing metadata")
    kernel_cv, kernel_trials = _timing_cv_and_trials(metadata, "kg_kernel_perf")
    reference_keys = {
        "kg_reference_perf_mean_ms",
        "kg_reference_perf_std_ms",
        "kg_reference_perf_num_trials",
    }
    if reference_keys.issubset(metadata):
        reference_cv, reference_trials = _timing_cv_and_trials(metadata, "kg_reference_perf")
    elif metadata.get("cached") is True:
        # KernelGYM's reference cache currently stores only the mean runtime.
        # Treat that shared baseline as fixed; the configured log-noise floor
        # is the place to account for cache age and cross-block drift.
        reference_cv, reference_trials = 0.0, 1
    else:
        raise ValueError("lcb_improvement requires reference timing metadata unless the reference runtime is cached")

    log_std_floor = float(config.get("speedup_uncertainty_log_std_floor", 0.0))
    if not math.isfinite(log_std_floor) or log_std_floor < 0.0:
        raise ValueError("speedup_uncertainty_log_std_floor must be non-negative and finite")
    return math.sqrt(kernel_cv**2 / kernel_trials + reference_cv**2 / reference_trials + log_std_floor**2)


def _calculate_performance_score(
    speedup: float,
    mode: str,
    metadata: dict[str, Any] | None,
    config: dict[str, Any],
) -> float:
    """Map a raw evaluator speedup to the scalar performance score."""

    mode = str(mode).strip().lower()
    allowed_modes = {"legacy", "improvement", "lcb_improvement"}
    if mode not in allowed_modes:
        raise ValueError(f"speedup_score_mode must be one of {sorted(allowed_modes)}, got {mode!r}")

    upper_bound = float(config["speedup_reward_upper_bound"])
    lower_bound = float(config["speedup_reward_lower_bound"])
    if mode == "legacy":
        performance_score = min(speedup, upper_bound)
        return 0.0 if performance_score < lower_bound else performance_score

    if not math.isfinite(speedup) or speedup <= 0.0:
        return 0.0
    if not math.isfinite(upper_bound) or upper_bound <= 1.0:
        raise ValueError("speedup_reward_upper_bound must be greater than 1.0 for improvement modes")

    if mode == "lcb_improvement":
        speedup_log_standard_error = _compute_speedup_log_standard_error(metadata, config)
        z_score = float(config.get("speedup_uncertainty_z_score", 1.96))
        if not math.isfinite(z_score) or z_score < 0.0:
            raise ValueError("speedup_uncertainty_z_score must be non-negative and finite")
        speedup *= math.exp(-z_score * speedup_log_standard_error)

    if speedup < lower_bound:
        return 0.0
    return min(max((speedup - 1.0) / (upper_bound - 1.0), 0.0), 1.0)


_REWARD_COMPONENT_KEYS = (
    "correctness",
    "performance",
    "coverage",
    "failed",
    "length",
)
_KERNEL_SCORE_KEYS = ("correctness", "performance", "coverage")


def _compute_dynamic_auxiliary_gate(num_correct: int, group_size: int, *, args=None) -> float:
    """Return the shared speedup/coverage gate for one valid reward group."""

    mode = getattr(args, "dynamic_reward_gate", None)
    if mode is None:
        return 1.0
    if mode == "sqrt":
        if group_size <= 1 or num_correct <= 1:
            return 0.0
        return math.sqrt(max((num_correct - 1) / (group_size - 1), 0.0))
    if mode in {"piecewise", "piecewise-sqrt"}:
        # Adapt Coda's thresholded gates (arXiv:2603.08659, Eq. 4), not its
        # length reward: downweight auxiliary objectives on hard groups and
        # upweight them on easy groups, with a neutral middle region.
        if group_size <= 1:
            return 1.0  # Insufficient group evidence: preserve the base weights.
        success_rate = num_correct / group_size
        hard_threshold, easy_threshold = getattr(args, "difficulty_thresholds", [1 / 3, 2 / 3])
        gate_min, gate_max = getattr(args, "dynamic_reward_gate_range", [0.8, 1.2])
        # Curve the distance from the neutral region, not the final gate:
        # piecewise-sqrt strengthens both tails while preserving their bounds.
        if success_rate < hard_threshold:
            distance = (hard_threshold - success_rate) / hard_threshold
            strength = math.sqrt(distance) if mode == "piecewise-sqrt" else distance
            return 1.0 - (1.0 - gate_min) * strength
        if success_rate > easy_threshold:
            distance = (success_rate - easy_threshold) / (1.0 - easy_threshold)
            strength = math.sqrt(distance) if mode == "piecewise-sqrt" else distance
            return 1.0 + (gate_max - 1.0) * strength
        return 1.0
    raise ValueError(f"Unknown dynamic reward gate: {mode!r}")


def _sample_is_correct(sample) -> bool:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    env_extra_info = metadata.get("env_extra_info")
    if isinstance(env_extra_info, dict) and "correctness" in env_extra_info:
        return bool(env_extra_info["correctness"]) and not bool(env_extra_info.get("decoy_kernel", False))

    kernel_score = metadata.get("kernel_score")
    if isinstance(kernel_score, dict) and "correctness" in kernel_score:
        return float(kernel_score["correctness"]) > 0.0

    components = metadata.get("reward_component")
    return isinstance(components, dict) and float(components.get("correctness", 0.0)) > 0.0


def annotate_group_difficulty(samples) -> tuple[int, int]:
    """Annotate one reward group with shared correctness/difficulty statistics.

    Multi-turn callers pass one ``(group_index, turn_idx)`` group at a time, so
    the resulting metadata describes the difficulty of that specific turn.
    Removed, padded, and aborted samples do not contribute to the denominator.
    Every kernel or pad sample receives the annotation for logging and
    downstream reward processing.
    """

    kernel_samples = []
    valid_samples = []
    for sample in samples:
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        if metadata.get("role", "kernel") not in {"kernel", "pad"}:
            continue
        kernel_samples.append(sample)
        if sample.remove_sample or sample.status == sample.Status.ABORTED or metadata.get("is_pad_turn"):
            continue
        valid_samples.append(sample)

    num_valid = len(valid_samples)
    num_correct = sum(_sample_is_correct(sample) for sample in valid_samples)
    correct_rate = num_correct / num_valid if num_valid else 0.0
    difficulty = 1.0 - correct_rate if num_valid else 0.0

    annotation = {
        "group_num_correct": num_correct,
        "group_num_valid": num_valid,
        "group_correct_rate": correct_rate,
        "group_difficulty": difficulty,
    }
    for sample in kernel_samples:
        sample.metadata = dict(sample.metadata or {})
        sample.metadata.update(annotation)

    return num_correct, num_valid


def _apply_dynamic_group_reward_weights(
    samples,
    rewards,
    config: dict[str, Any],
    *,
    args=None,
    record_metrics: bool = False,
) -> list[float]:
    """Rebuild rewards from their components with a group auxiliary gate."""

    rewards = [float(reward) for reward in rewards]
    if len(samples) != len(rewards):
        raise ValueError("samples and rewards must have the same length")

    num_correct, group_size = annotate_group_difficulty(samples)
    gate = _compute_dynamic_auxiliary_gate(num_correct, group_size, args=args)
    dynamic_rewards = []
    for sample, reward in zip(samples, rewards, strict=True):
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        components = metadata.get("reward_component")
        kernel_score = metadata.get("kernel_score")
        if (
            not isinstance(components, dict)
            or any(key not in components for key in _REWARD_COMPONENT_KEYS)
            or not isinstance(kernel_score, dict)
            or any(key not in kernel_score for key in _KERNEL_SCORE_KEYS)
        ):
            dynamic_rewards.append(reward)
            continue

        correctness_score, performance_score, coverage_score = (float(kernel_score[key]) for key in _KERNEL_SCORE_KEYS)
        correctness_reward = float(components["correctness"])
        failed_reward = components["failed"]
        length_score = float(components["length"])
        failed_reward = None if failed_reward is None else float(failed_reward)

        dynamic_reward = failed_reward
        base_performance_reward = float(components["performance"])
        if dynamic_reward is None:
            weight_performance = float(config["init_performance_weight"])
            if config.get("performance_reward_requires_correctness", False):
                weight_performance *= correctness_score
            base_performance_reward = weight_performance * performance_score
            weight_performance *= gate
            weight_coverage = float(config["coverage_reward_weight"]) * gate
            if not config["coverage_reward_enable"]:
                weight_coverage = 0.0
            performance_reward = weight_performance * performance_score
            coverage_reward = weight_coverage * coverage_score
            components["performance"] = performance_reward
            components["coverage"] = coverage_reward
            dynamic_reward = correctness_reward + performance_reward + coverage_reward
        metadata["task_reward"] = dynamic_reward
        metadata["length_score"] = length_score
        if record_metrics:
            # Compare against the ungated contribution, not an earlier filter
            # preview or a previous invocation of reward post-processing.
            metadata["dynamic_reward"] = {
                "gate": gate,
                "performance_reward_delta": float(components["performance"]) - base_performance_reward,
            }
        dynamic_rewards.append(dynamic_reward + length_score)
    return dynamic_rewards


def _apply_overlong_penalty(args, sample, metadata: dict[str, Any]) -> float:
    """Replace the length component, never subtract repeatedly from a shaped reward."""
    response_len = int(getattr(sample, "response_length", 0) or 0)
    tokens = getattr(sample, "tokens", None)
    prompt_len = max(0, len(tokens) - response_len) if isinstance(tokens, (list, tuple)) else 0
    response_cap = int(getattr(args, "rollout_max_response_len", 0) or 0)
    context_cap = int(getattr(args, "rollout_max_context_len", 0) or 0)
    effective_cap = response_cap
    if getattr(args, "overlong_use_effective_response_cap", False) and context_cap > 0:
        effective_cap = min(response_cap, max(1, context_cap - prompt_len))

    penalty = 0.0
    buffer_len = int(getattr(args, "overlong_buffer_len", 2048))
    factor = float(getattr(args, "overlong_penalty_factor", 1.0))
    if buffer_len > 0 and factor > 0 and effective_cap > 0:
        window = min(buffer_len, effective_cap)
        exceed = response_len - (effective_cap - window)
        penalty = factor * min(1.0, max(0, exceed) / window)

    metadata["length_score"] = -penalty
    metadata["overlong_prompt_len"] = prompt_len
    metadata["overlong_effective_response_cap"] = effective_cap
    metadata["reward_component"]["length"] = -penalty
    return float(metadata["task_reward"]) - penalty


def post_process_rollout_rewards(args, samples, *, stage: str = "rollout") -> list[float]:
    """Shape and write back single-turn rewards; never compute returns or advantages.

    The sample stage applies per-response length shaping. The rollout stage settles dynamic weights, length shaping,
    and failure-stage fallback before filtering complete prompt/turn groups.
    Reapplication is idempotent; removed samples are always untouched.
    """
    if stage not in {"sample", "rollout"}:
        raise ValueError("Reward post-processing stage must be sample or rollout")
    if stage == "rollout":
        for sample in samples:
            if isinstance(sample.metadata, dict):
                sample.metadata.pop("dynamic_reward", None)
    config = CUDA_AGENT_CONFIGS["reward"]
    dynamic_weight = stage == "rollout" and getattr(args, "dynamic_reward_gate", None) is not None
    failed_group_reward = stage == "rollout" and bool(config["apply_failed_group_reward"])
    overlong_penalty = getattr(args, "overlong_penalty", None)
    if overlong_penalty not in {None, "dapo", "laser-d"}:
        raise ValueError("--overlong-penalty must be None, dapo, or laser-d")
    reward_key = getattr(args, "reward_key", None)
    rewards = [float(sample.reward[reward_key] if reward_key else sample.reward) for sample in samples]
    if not dynamic_weight and not failed_group_reward and overlong_penalty is None:
        return rewards

    reward_groups: dict[object, list[int]] = {}
    eligible_indices = []
    for idx, sample in enumerate(samples):
        metadata = sample.metadata or {}
        if (
            sample.remove_sample
            or sample.status == sample.Status.ABORTED
            or metadata.get("role", "kernel") != "kernel"
            or metadata.get("is_pad_turn")
        ):
            continue
        eligible_indices.append(idx)
        if dynamic_weight or failed_group_reward:
            group_key: object = sample.group_index
            if getattr(args, "use_multi_turn", False):
                turn_idx = metadata.get("turn_idx")
                assert turn_idx is not None, "--use-multi-turn requires sample.metadata['turn_idx']"
                group_key = group_key, int(turn_idx)
            reward_groups.setdefault(group_key, []).append(idx)

    for group_indices in reward_groups.values():
        if not dynamic_weight:
            continue
        group_samples = [samples[idx] for idx in group_indices]
        group_values = _apply_dynamic_group_reward_weights(
            group_samples,
            [rewards[idx] for idx in group_indices],
            config,
            args=args,
            record_metrics=True,
        )
        for idx, reward in zip(group_indices, group_values, strict=True):
            rewards[idx] = reward

    for idx in eligible_indices:
        sample = samples[idx]
        metadata = sample.metadata or {}
        if overlong_penalty == "dapo":
            if "reward_component" not in metadata or "task_reward" not in metadata:
                raise ValueError("overlong-penalty requires kernel task_reward and reward_component metadata")
            rewards[idx] = _apply_overlong_penalty(args, sample, metadata)
        elif overlong_penalty == "laser-d":
            if (
                sample.status not in {sample.Status.COMPLETED, sample.Status.TRUNCATED}
                or metadata.get("role", "kernel") != "kernel"
                or metadata.get("is_pad_turn")
            ):
                continue
            record = metadata.get("laser_d")
            if not isinstance(record, dict):
                if stage == "sample":
                    # Individual generation does not yet know the complete group's difficulty.
                    continue
                raise ValueError("LASER-D requires a budget snapshot from the pre-filter rollout collector")
            bonus = (
                float(getattr(args, "laser_d_length_score", 0.5))
                if _sample_is_correct(sample) and sample.response_length <= record["budget"]
                else 0.0
            )
            metadata["reward_component"]["length"] = bonus
            metadata["length_score"] = bonus
            record["length_score"] = bonus
            rewards[idx] = float(metadata["task_reward"]) + bonus

    if failed_group_reward:
        for group_indices in reward_groups.values():
            group_values = _apply_failed_group_reward(
                args,
                [samples[idx] for idx in group_indices],
                [rewards[idx] for idx in group_indices],
                float(config["failed_score"]),
                float(config["init_correct_weight"]),
            )
            for idx, reward in zip(group_indices, group_values, strict=True):
                rewards[idx] = reward

    for idx in eligible_indices:
        sample = samples[idx]
        # Keep dictionary-valued rewards and their unrelated fields intact.
        if reward_key:
            sample.reward = {**sample.reward, reward_key: rewards[idx]}
        else:
            sample.reward = rewards[idx]
    return rewards


def _annotate_conditional_truncation_mask(args, sample) -> None:
    """Select a sample for the MicroCoder-GRPO Conditional Truncation Mask (CTM).

    Implements CTM from Breaking Training Bottlenecks: Effective and Stable
    Reinforcement Learning for Coding Models (arXiv:2603.07777). Eligible
    responses reach the maximum length, are non-incorrect (correct or
    incomplete), and do not repeat the preceding 128-token window at the tail;
    they are randomly selected with probability rho for final advantage masking.
    The paper compares rho=0.1, 0.2, and 0.3; slime defaults to rho=0.1. This
    hook only records the selection. The training backend zeros advantages
    after OPD and advantage normalization, without changing their statistics.

    Kernel-specific restriction: runtime and later-stage failures remain
    learnable even when generation was truncated. Precheck/compile failures
    can still qualify as incomplete responses under the conditions below.
    """
    sample.metadata = dict(sample.metadata or {})
    for key in (
        "conditional_truncation_masking_eligible",
        "conditional_truncation_masked",
        "conditional_truncation_mask_prob",
        "conditional_truncation_repeat_window",
    ):
        sample.metadata.pop(key, None)
    if sample.remove_sample:
        return
    if sample.loss_mask is not None and sum(sample.loss_mask) == 0:
        return

    # Paper CTM eligibility: max length, non-incorrect (correct or truncated),
    # no repeated tail, then Bernoulli masking.
    max_response_len = int(getattr(args, "rollout_max_response_len", getattr(args, "max_new_tokens", 0)) or 0)
    response_length = int(sample.response_length or 0)
    if max_response_len <= 0 or response_length != max_response_len:
        return

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    env_extra_info = metadata.get("env_extra_info")
    env_result = metadata.get("env_result")
    env_state = env_result.get("env_state", env_result) if isinstance(env_result, dict) else {}
    env_state = env_state if isinstance(env_state, dict) else {}
    env_metadata = env_state.get("metadata")
    env_metadata = env_metadata if isinstance(env_metadata, dict) else {}
    extra = env_extra_info if isinstance(env_extra_info, dict) else {}

    # Error evidence takes precedence over both TRUNCATED and correctness=True:
    # a correct kernel may still fail during performance/profiling. Read the
    # original result as well as compact feedback; reward-stage scores may be
    # disabled and must not control CTM eligibility.
    error = env_state.get("error") or env_state.get("error_code")
    if error in {RUNTIME_ERROR, CORRECTNESS_ERROR} or env_state.get("error_code") in {
        RUNTIME_ERROR,
        CORRECTNESS_ERROR,
    }:
        return
    if any(
        info.get(key)
        for info in (env_state, env_metadata, extra)
        for key in (
            "runtime_error",
            "correctness_runtime_error",
            "correctness_output_mismatch",
            "performance_error",
            "profiling_error",
        )
    ):
        return
    compiled = env_state.get("compiled", extra.get("compilation"))
    if compiled is True and (
        error
        or env_state.get("error_message")
        or env_metadata.get("error")
        or env_state.get("status") in {"failed", "timeout"}
        or env_state.get("success") is False
        or env_state.get("correctness", extra.get("correctness")) is False
    ):
        return
    if isinstance(env_extra_info, dict):
        if bool(env_extra_info.get("decoy_kernel")):
            return
        is_mask_candidate = env_extra_info.get("correctness") is True or sample.status == sample.Status.TRUNCATED
    else:
        is_mask_candidate = sample.status == sample.Status.TRUNCATED
    if not is_mask_candidate:
        return

    repeat_window = int(getattr(args, "conditional_truncation_repeat_window", 128))
    response_tokens = sample.tokens[-response_length:] if response_length > 0 else []
    has_repeated_tail = (
        repeat_window > 0
        and len(response_tokens) >= 2 * repeat_window
        and response_tokens[-repeat_window:] == response_tokens[-2 * repeat_window : -repeat_window]
    )
    if has_repeated_tail:
        return

    mask_prob = float(getattr(args, "conditional_truncation_mask_prob", 0.1))
    sample.metadata["conditional_truncation_masking_eligible"] = True
    if random.random() >= mask_prob:
        return

    sample.metadata["conditional_truncation_masked"] = True
    sample.metadata["conditional_truncation_mask_prob"] = mask_prob
    sample.metadata["conditional_truncation_repeat_window"] = repeat_window


def reward_post_process_by_group(args, samples):
    # Collectors have settled and written rewards before filtering. Never
    # recompute shaping from a filtered subset at the training boundary.
    raw_rewards = [float(sample.get_reward_value(args)) for sample in samples]
    if args.advantage_estimator in {"argmaxrl", "tailrl"}:
        for sample in samples:
            if getattr(args, "use_conditional_truncation_mask", False):
                _annotate_conditional_truncation_mask(args, sample)
        # RolloutManager computes complete-group ArgMaxRL/TailRL advantages before DP
        # partitioning. Preserve the settled aggregate reward, without centering,
        # std normalization, or TRLOO future-return folding.
        return raw_rewards, raw_rewards
    if args.advantage_estimator == "trloo" and getattr(args, "use_multi_turn", False):
        for idx, sample in enumerate(samples):
            metadata = sample.metadata or {}
            if sample.remove_sample:
                continue
            if "return_reward" not in metadata:
                raise ValueError(
                    "TRLOO requires metadata['return_reward'] computed from the complete trajectory "
                    "before reward post-processing; regenerate legacy rollout data missing this field"
                )
            # Only the current turn uses shaped reward; future credit was frozen
            # from original task_reward before dynamic weighting/failure fallback.
            raw_rewards[idx] += float(metadata["return_reward"])
    rewards = [None] * len(raw_rewards)
    use_conditional_truncation_mask = getattr(args, "use_conditional_truncation_mask", False)

    idx_to_group_index: dict[int, object] = {}
    reward_groups: dict[object, list[int]] = {}
    for idx, sample in enumerate(samples):
        if use_conditional_truncation_mask:
            _annotate_conditional_truncation_mask(args, sample)
        if sample.remove_sample:
            rewards[idx] = raw_rewards[idx]
            continue

        group_index = sample.group_index
        if getattr(args, "use_multi_turn", False):
            turn_idx = sample.metadata.get("turn_idx") if sample.metadata is not None else None
            assert turn_idx is not None, "--use-multi-turn requires sample.metadata['turn_idx']"
            group_index = group_index, int(turn_idx)
        idx_to_group_index[idx] = group_index
        reward_groups.setdefault(group_index, []).append(idx)

    group_stats: dict[object, dict[str, float]] = {}
    for group_index, group_indices in reward_groups.items():
        group_samples = [samples[idx] for idx in group_indices]
        annotate_group_difficulty(group_samples)
        group_reward_values = [raw_rewards[idx] for idx in group_indices]
        group_rewards = torch.tensor(group_reward_values, dtype=torch.float)
        group_stats[group_index] = {
            "mean": group_rewards.mean().item(),
            "std": group_rewards.std().item() if len(group_reward_values) > 1 else 0.0,
            "size": len(group_reward_values),
        }

    for idx, raw_reward in enumerate(raw_rewards):
        if rewards[idx] is not None:
            continue

        group_index = idx_to_group_index[idx]
        stats = group_stats[group_index]
        reward = raw_reward - stats["mean"]

        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
            reward = reward / (stats["std"] + 1e-6)

        if args.advantage_estimator in ["rloo", "trloo"]:
            # Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
            # Each contiguous group of ``n_samples_per_prompt`` samples is treated as one
            # prompt group. The leave-one-out baseline for a sample is the mean reward of
            # the other samples in that group. For singleton groups, no baseline is used.
            group_len = stats["size"]
            if group_len == 1:
                reward = 0.0
            else:
                reward = reward * group_len / (group_len - 1)

        rewards[idx] = reward

    return raw_rewards, rewards


def get_kernel_group_filter_rewards(args, samples, rewards) -> list[float]:
    """Shared reward basis for failure-group detection and read-only filtering."""
    if getattr(args, "overlong_penalty", None) == "laser-d":
        return [float(reward) for reward in rewards]
    return [
        float(sample.metadata.get("task_reward", reward) if isinstance(sample.metadata, dict) else reward)
        for sample, reward in zip(samples, rewards, strict=True)
    ]


def _apply_failed_group_reward(
    args,
    samples,
    rewards,
    failed_score: float,
    init_correct_weight: float,
) -> list[float]:
    """Replace default-scored all-failed groups before filtering.

    Detection uses the filter's reward basis, requiring exact equality to the
    weighted default failure score. DAPO penalties are retained, but not tested.
    """
    rewards = [float(reward) for reward in rewards]
    if not rewards or any(_sample_is_correct(sample) for sample in samples):
        return rewards
    filter_rewards = get_kernel_group_filter_rewards(args, samples, rewards)
    default_failed_reward = failed_score * init_correct_weight
    if any(reward != default_failed_reward for reward in filter_rewards):
        return rewards

    metadata = [sample.metadata if isinstance(sample.metadata, dict) else {} for sample in samples]
    if any("kernel_failed_score" not in item for item in metadata):
        return rewards

    failed_rewards = [float(item["kernel_failed_score"]) * init_correct_weight for item in metadata]
    for item, failed_reward in zip(metadata, failed_rewards, strict=True):
        components = item.setdefault("reward_component", {})
        # Replace only task contributions; retain length shaping and frozen future credit.
        for key in components:
            if key not in {"length", "return_reward"}:
                components[key] = 0.0
        components["failed"] = failed_reward
        item["task_reward"] = failed_reward
        item["failed_group_reward"] = True
    return [
        failed_reward + (reward - filter_reward)
        for failed_reward, reward, filter_reward in zip(failed_rewards, rewards, filter_rewards, strict=True)
    ]


def generate_rollout(args, rollout_id, data_source, evaluation=False):
    """Kernel reward settlement before the standard SGLang group filter."""
    from slime.rollout.sglang_rollout import generate_rollout as default_generate_rollout
    from slime.rollout.sglang_rollout import generate_rollout_async
    from slime.utils.async_utils import run

    from .length_reward import LaserDBudgetController

    if evaluation:
        return default_generate_rollout(args, rollout_id, data_source, evaluation=True)
    controller = (
        LaserDBudgetController(args, data_source, rollout_id)
        if getattr(args, "overlong_penalty", None) == "laser-d"
        else None
    )

    def process_group(group):
        annotate_group_difficulty(group)
        if controller is not None:
            controller.observe(group)
        post_process_rollout_rewards(args, group)

    output, aborted = run(
        generate_rollout_async(args, rollout_id, data_source.get_samples, on_group_completed=process_group)
    )
    if aborted:
        data_source.add_samples(aborted)
    if controller is not None:
        output.metrics = {**(output.metrics or {}), **controller.finish()}
    return output


def _resolve_kernel_failed_score(env_state: dict[str, Any], config: dict[str, Any]) -> tuple[float, str]:
    """Classify a failed kernel and return its configured score and reason."""
    correctness = bool(env_state.get("correctness", False))
    compiled = env_state.get("compiled")
    metadata = env_state.get("metadata") if isinstance(env_state.get("metadata"), dict) else {}
    error = env_state.get("error")
    error_message = str(env_state.get("error_message") or "")
    lower_error_message = error_message.lower()
    kernel_failed_score = config["kernel_failed_score"]

    output_mismatch_score = float(kernel_failed_score.get("output_mismatch", 0.0))
    if output_mismatch_score < 0.0:
        raise ValueError("kernel_failed_score['output_mismatch'] must be non-negative")
    if output_mismatch_score >= 1.0:
        raise ValueError("kernel_failed_score['output_mismatch'] must be lower than correctness_score (1.0)")

    precheck_error_codes = {PRECHECK_ERROR, VALIDATION_ERROR, SYNTAX_ERROR, IMPORT_ERROR}
    output_mismatch = (
        env_state.get("status") == "completed"
        and not bool(env_state.get("decoy_kernel", False))
        and not correctness
        and error != RUNTIME_ERROR
        and bool(metadata.get("correctness_output_mismatch", False))
    )
    if output_mismatch and compiled is not True:
        if output_mismatch_score > 0.0:
            raise AssertionError(
                "KernelGym contract violation: correctness_output_mismatch=true requires compiled=true"
            )
        output_mismatch = False

    if bool(env_state.get("decoy_kernel", False)):
        stage = "decoy"
    elif metadata.get("client_precheck") or error in precheck_error_codes or "pre-check error" in lower_error_message:
        stage = "precheck"
    elif error == RUNTIME_ERROR or metadata.get("runtime_error") or metadata.get("correctness_runtime_error"):
        stage = "runtime"
    elif error == KERNEL_EVAL_TIMEOUT and compiled is True:
        stage = "runtime"
    elif output_mismatch:
        stage = "output_mismatch"
    elif (
        error == CORRECTNESS_ERROR
        or metadata.get("correctness_candidate_forward_completed")
        or (env_state.get("status") == "completed" and compiled is True)
    ):
        stage = "correctness"
    elif compiled is True:
        stage = "runtime"
    elif error == COMPILATION_ERROR or compiled is False or "kernel compilation error" in lower_error_message:
        stage = "compilation"
    else:
        stage = "other"

    return float(kernel_failed_score.get(stage, config["failed_score"])), stage


def _compute_coverage(result: dict[str, Any], config: dict[str, Any]) -> dict[str, float]:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    num_custom_kernel = result.get("num_custom_kernel", metadata.get("num_custom_kernel", 0)) or 0
    num_total_kernels = result.get("num_total_kernels", metadata.get("num_total_kernels", 0)) or 0
    custom_time = (
        result.get(
            "custom_kernel_cuda_time_in_profiling_us",
            metadata.get("custom_kernel_cuda_time_in_profiling_us", 0),
        )
        or 0
    )
    total_time = (
        result.get(
            "total_kernel_run_time_in_profiling_us",
            metadata.get("total_kernel_run_time_in_profiling_us", 0),
        )
        or 0
    )

    number_coverage = float(num_custom_kernel) / float(num_total_kernels) if num_total_kernels else 0.0
    time_coverage = float(custom_time) / float(total_time) if total_time else 0.0
    coverage = time_coverage if config["coverage_reward_type"] == "time_coverage" else number_coverage
    return {
        "coverage": coverage,
        "num_custom_kernel": num_custom_kernel,
        "num_total_kernels": num_total_kernels,
        "custom_kernel_cuda_time_in_profiling_us": custom_time,
        "total_kernel_run_time_in_profiling_us": total_time,
    }


def calculate_kernel_reward(
    env_state: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Calculate base kernel scores without rollout reward post-processing."""
    default_failed_score = float(config["failed_score"])
    apply_kernel_failed_score = bool(config.get("apply_kernel_failed_score", False))
    if apply_kernel_failed_score and bool(config.get("apply_failed_group_reward", False)):
        raise ValueError("apply_kernel_failed_score and apply_failed_group_reward cannot both be enabled")

    completed = env_state.get("status") == "completed"
    decoy_kernel = bool(env_state.get("decoy_kernel", False))
    correctness = bool(env_state.get("correctness", False)) if completed else False
    compiled = bool(env_state.get("compiled", False)) if completed else False
    kernel_failed = not (completed and compiled and correctness and not decoy_kernel)
    speedup = float(env_state.get("speedup") or 0.0) if completed else 0.0
    speedup_log_standard_error = None
    correctness_score = float(correctness)
    performance_score = 0.0
    coverage_score = 0.0
    correctness_reward = 0.0
    performance_reward = 0.0
    coverage_reward = 0.0
    failed_reward = None
    kernel_failed_score = None
    kernel_failed_score_tag = None
    coverage_info = {
        "coverage": 0.0,
        "num_custom_kernel": 0,
        "num_total_kernels": 0,
        "custom_kernel_cuda_time_in_profiling_us": 0,
        "total_kernel_run_time_in_profiling_us": 0,
    }

    if kernel_failed:
        if apply_kernel_failed_score or config.get("apply_failed_group_reward", False):
            kernel_failed_score, kernel_failed_score_tag = _resolve_kernel_failed_score(env_state, config)
        else:
            kernel_failed_score = default_failed_score
            kernel_failed_score_tag = "disabled"
        failed_score = kernel_failed_score if apply_kernel_failed_score else default_failed_score
        task_reward = failed_score * float(config["init_correct_weight"])
        failed_reward = task_reward
    else:
        speedup_score_mode = str(config.get("speedup_score_mode", "legacy")).strip().lower()
        if speedup_score_mode == "lcb_improvement" and math.isfinite(speedup) and speedup > 0.0:
            speedup_log_standard_error = _compute_speedup_log_standard_error(env_state.get("metadata"), config)
        performance_score = _calculate_performance_score(
            speedup,
            speedup_score_mode,
            env_state.get("metadata"),
            config,
        )

        correctness_reward = float(config["init_correct_weight"])
        weight_performance = float(config["init_performance_weight"])
        if config.get("performance_reward_requires_correctness", False):
            weight_performance *= correctness_score
        performance_reward = weight_performance * performance_score
        task_reward = correctness_reward + performance_reward

        coverage_info = _compute_coverage(env_state, config)
        coverage_score = coverage_info["coverage"]
        weight_coverage = float(config["coverage_reward_weight"]) if config["coverage_reward_enable"] else 0.0
        coverage_reward = weight_coverage * coverage_score
        task_reward += coverage_reward

    details = {
        **env_state,
        "reward": task_reward,
        "raw_task_reward": task_reward,
        "task_reward": task_reward,
        "length_score": 0.0,
        "overlong_prompt_len": 0,
        "overlong_effective_response_cap": 0,
        "speedup": speedup,
        "success": not kernel_failed,
        "correctness": correctness,
        "compiled": compiled,
        "decoy_kernel": decoy_kernel,
        "profiling": env_state.get("profiling"),
        "kernel_failed_score": kernel_failed_score,
        "kernel_failed_score_tag": kernel_failed_score_tag,
        "kernel_score": {
            "correctness": correctness_score,
            "performance": performance_score,
            "coverage": coverage_score,
        },
        "reward_component": {
            "correctness": correctness_reward,
            "performance": performance_reward,
            "coverage": coverage_reward,
            "failed": failed_reward,
            "length": 0.0,
        },
        "speedup_log_standard_error": speedup_log_standard_error,
        **coverage_info,
    }
    return details
