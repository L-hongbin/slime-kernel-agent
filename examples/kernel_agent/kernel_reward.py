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


def calculate_reward(env_result: dict[str, Any], config: dict[str, Any]) -> float:
    env_state = env_result.get("env_state") if isinstance(env_result, dict) else {}
    return calculate_reward_speedup(env_state, config)["reward"]


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


def _compute_speedup_reward_value(
    speedup: float,
    mode: str,
    metadata: dict[str, Any] | None,
    config: dict[str, Any],
) -> float:
    """Map a raw evaluator speedup to the scalar performance score."""

    mode = str(mode).strip().lower()
    allowed_modes = {"legacy", "improvement", "lcb_improvement"}
    if mode not in allowed_modes:
        raise ValueError(f"speedup_reward_mode must be one of {sorted(allowed_modes)}, got {mode!r}")

    upper_bound = float(config["speedup_reward_upper_bound"])
    lower_bound = float(config["speedup_reward_lower_bound"])
    if mode == "legacy":
        speedup_reward = min(speedup, upper_bound)
        return 0.0 if speedup_reward < lower_bound else speedup_reward

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
    "reward_correctness_component",
    "reward_performance_component",
    "reward_coverage_component",
    "reward_partial_component",
    "reward_penalty_component",
)


def _compute_dynamic_auxiliary_gate(num_correct: int, group_size: int) -> float:
    """Return the shared speedup/coverage gate for one valid reward group."""

    if group_size <= 1 or num_correct <= 1:
        return 0.0
    return math.sqrt(max((num_correct - 1) / (group_size - 1), 0.0))


def _sample_is_correct(sample) -> bool:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    env_extra_info = metadata.get("env_extra_info")
    if isinstance(env_extra_info, dict) and "correctness" in env_extra_info:
        return bool(env_extra_info["correctness"]) and not bool(env_extra_info.get("decoy_kernel", False))

    components = metadata.get("reward_components")
    return isinstance(components, dict) and float(components.get("reward_correctness_component", 0.0)) > 0.0


def _apply_dynamic_group_reward_weights(
    samples,
    rewards,
    config: dict[str, Any],
) -> list[float]:
    """Rescale speedup and coverage components with one gate shared by the group.

    ``rewards`` may already contain additive shaping such as an overlong
    penalty. The residual relative to the recorded task components is kept
    unchanged, so only speedup and coverage are dynamically reweighted.
    """

    rewards = [float(reward) for reward in rewards]
    if not config.get("enable_dynamic_reward_weight", False):
        return rewards
    if len(samples) != len(rewards):
        raise ValueError("samples and rewards must have the same length")

    group_size = len(samples)
    num_correct = sum(_sample_is_correct(sample) for sample in samples)
    gate = _compute_dynamic_auxiliary_gate(num_correct, group_size)
    dynamic_rewards = []
    for sample, reward in zip(samples, rewards, strict=True):
        metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
        components = metadata.get("reward_components")
        if not isinstance(components, dict) or any(key not in components for key in _REWARD_COMPONENT_KEYS):
            dynamic_rewards.append(reward)
            continue

        correctness, performance, coverage, partial, penalty = (
            float(components[key]) for key in _REWARD_COMPONENT_KEYS
        )
        static_task_reward = correctness + performance + coverage + partial + penalty
        additive_residual = reward - static_task_reward
        dynamic_task_reward = correctness + gate * (performance + coverage) + partial + penalty
        dynamic_rewards.append(dynamic_task_reward + additive_residual)
    return dynamic_rewards


def _dynamic_raw_rewards(args, samples, config: dict[str, Any]) -> list[float]:
    """Apply per-turn group weights and rebuild TRLOO trajectory returns."""

    turn_rewards = [0.0 if sample.remove_sample else float(sample.get_reward_value(args)) for sample in samples]
    reward_groups: dict[object, list[int]] = {}
    for idx, sample in enumerate(samples):
        if sample.remove_sample:
            continue
        group_key: object = sample.group_index
        if getattr(args, "use_multi_turn", False):
            turn_idx = sample.metadata.get("turn_idx") if isinstance(sample.metadata, dict) else None
            assert turn_idx is not None, "--use-multi-turn requires sample.metadata['turn_idx']"
            group_key = group_key, int(turn_idx)
        reward_groups.setdefault(group_key, []).append(idx)

    for group_indices in reward_groups.values():
        group_samples = [samples[idx] for idx in group_indices]
        group_values = _apply_dynamic_group_reward_weights(
            group_samples,
            [turn_rewards[idx] for idx in group_indices],
            config,
        )
        for idx, reward in zip(group_indices, group_values, strict=True):
            turn_rewards[idx] = reward

    if args.advantage_estimator != "trloo" or not getattr(args, "use_multi_turn", False):
        return turn_rewards

    gamma = float(getattr(args, "multi_turn_gamma", 1.0))
    trajectory_groups: dict[tuple[object, object], list[int]] = {}
    for idx, sample in enumerate(samples):
        trajectory_id = sample.rollout_id if sample.rollout_id is not None else sample.index
        trajectory_groups.setdefault((sample.group_index, trajectory_id), []).append(idx)

    raw_rewards = [0.0] * len(samples)
    for trajectory_indices in trajectory_groups.values():
        trajectory_indices.sort(
            key=lambda idx: int(samples[idx].metadata.get("turn_idx", 0)),
            reverse=True,
        )
        cumulative_reward = 0.0
        for idx in trajectory_indices:
            turn_reward = 0.0 if samples[idx].remove_sample else turn_rewards[idx]
            cumulative_reward = turn_reward + gamma * cumulative_reward
            raw_rewards[idx] = cumulative_reward
    return raw_rewards


def _apply_conditional_truncation_mask(args, sample, advantage: float) -> float:
    """Apply the MicroCoder-GRPO Conditional Truncation Mask (CTM).

    Implements CTM from Breaking Training Bottlenecks: Effective and Stable
    Reinforcement Learning for Coding Models (arXiv:2603.07777). Eligible
    responses reach the maximum length, are non-incorrect (correct or
    incomplete), and do not repeat the preceding 128-token window at the tail;
    their post-processed advantages are randomly zeroed with probability rho.
    The paper compares rho=0.1, 0.2, and 0.3; slime defaults to rho=0.1. This
    hook runs after group reward normalization so masked samples do not alter
    other samples' advantages.
    """
    if sample.remove_sample:
        return advantage
    if sample.loss_mask is not None and sum(sample.loss_mask) == 0:
        return advantage

    # Paper CTM eligibility: max length, non-incorrect (correct or truncated),
    # no repeated tail, then Bernoulli masking.
    max_response_len = int(getattr(args, "rollout_max_response_len", getattr(args, "max_new_tokens", 0)) or 0)
    response_length = int(sample.response_length or 0)
    if max_response_len <= 0 or response_length != max_response_len:
        return advantage

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    env_extra_info = metadata.get("env_extra_info")
    if isinstance(env_extra_info, dict):
        if bool(env_extra_info.get("decoy_kernel")):
            return advantage
        is_mask_candidate = env_extra_info.get("correctness") is True or sample.status == sample.Status.TRUNCATED
    else:
        is_mask_candidate = sample.status == sample.Status.TRUNCATED
    if not is_mask_candidate:
        return advantage

    repeat_window = int(getattr(args, "conditional_truncation_repeat_window", 128))
    response_tokens = sample.tokens[-response_length:] if response_length > 0 else []
    has_repeated_tail = (
        repeat_window > 0
        and len(response_tokens) >= 2 * repeat_window
        and response_tokens[-repeat_window:] == response_tokens[-2 * repeat_window : -repeat_window]
    )
    if has_repeated_tail:
        return advantage

    mask_prob = float(getattr(args, "conditional_truncation_mask_prob", 0.1))
    sample.metadata = dict(sample.metadata or {})
    sample.metadata["conditional_truncation_masking_eligible"] = True
    if random.random() >= mask_prob:
        return advantage

    sample.metadata["conditional_truncation_masked"] = True
    sample.metadata["conditional_truncation_mask_prob"] = mask_prob
    sample.metadata["conditional_truncation_repeat_window"] = repeat_window
    return 0.0


def reward_post_process_by_group(args, samples):
    reward_config = CUDA_AGENT_CONFIGS["reward"]
    if reward_config.get("enable_dynamic_reward_weight", False):
        raw_rewards = _dynamic_raw_rewards(args, samples, reward_config)
    elif args.advantage_estimator == "trloo":
        raw_rewards = [sample.metadata["multi_turn_reward"] for sample in samples]
    else:
        raw_rewards = [sample.get_reward_value(args) for sample in samples]
    rewards = [None] * len(raw_rewards)
    use_conditional_truncation_mask = getattr(args, "use_conditional_truncation_mask", False)

    idx_to_group_index: dict[int, object] = {}
    reward_groups: dict[object, list[int]] = {}
    for idx, sample in enumerate(samples):
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
    apply_failed_group_reward = bool(reward_config["apply_failed_group_reward"])
    failed_score = float(reward_config["failed_score"])
    for group_index, group_indices in reward_groups.items():
        group_samples = [samples[idx] for idx in group_indices]
        group_reward_values = [raw_rewards[idx] for idx in group_indices]
        if apply_failed_group_reward:
            group_reward_values = _apply_failed_group_reward(group_samples, group_reward_values, failed_score)
        for idx, reward in zip(group_indices, group_reward_values, strict=True):
            raw_rewards[idx] = reward
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

        if use_conditional_truncation_mask:
            reward = _apply_conditional_truncation_mask(args, samples[idx], reward)
        rewards[idx] = reward

    return raw_rewards, rewards


def _apply_failed_group_reward(samples, rewards, failed_score: float) -> list[float]:
    """Use saved penalty scores when every group reward equals the failure score."""
    rewards = [float(reward) for reward in rewards]
    if not rewards or any(reward != failed_score for reward in rewards):
        return rewards

    metadata = [sample.metadata if isinstance(sample.metadata, dict) else {} for sample in samples]
    if any("penalty_score" not in item for item in metadata):
        return rewards

    return [float(item["penalty_score"]) for item in metadata]


def _resolve_penalty_score(env_state: dict[str, Any], config: dict[str, Any]) -> float:
    """Classify evaluation progress and return its failed-group penalty."""
    correctness = bool(env_state.get("correctness", False))
    compiled = env_state.get("compiled")
    if compiled is True and correctness and not bool(env_state.get("decoy_kernel", False)):
        return 0.0

    metadata = env_state.get("metadata") if isinstance(env_state.get("metadata"), dict) else {}
    error = env_state.get("error")
    error_message = str(env_state.get("error_message") or "")
    lower_error_message = error_message.lower()
    precheck_error_codes = {PRECHECK_ERROR, VALIDATION_ERROR, SYNTAX_ERROR, IMPORT_ERROR}

    if bool(env_state.get("decoy_kernel", False)):
        stage = "decoy"
    elif metadata.get("client_precheck") or error in precheck_error_codes or "pre-check error" in lower_error_message:
        stage = "precheck"
    elif error == RUNTIME_ERROR or metadata.get("runtime_error") or metadata.get("correctness_runtime_error"):
        stage = "runtime"
    elif error == KERNEL_EVAL_TIMEOUT and compiled is True:
        stage = "runtime"
    elif (
        error == CORRECTNESS_ERROR
        or metadata.get("correctness_output_mismatch")
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

    return float(config["penalty_score"].get(stage, config["failed_score"]))


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


def _resolve_output_mismatch_partial_credit(
    env_state: dict[str, Any],
    config: dict[str, Any],
) -> tuple[float, bool, str]:
    """Return the reviewed output-mismatch partial reward and audit reason.

    KernelGym sets ``correctness_output_mismatch`` only after the candidate
    forward returns, CUDA synchronization completes, and shape/value comparison
    fails. Compilation is therefore a consistency assertion, not the rewarded
    event. Runtime failures, timeouts, decoys, and generic compiled-but-wrong
    results remain zero-reward failures.
    """

    partial_reward = float(config.get("output_mismatch_partial_reward", 0.0))
    if partial_reward < 0.0:
        raise ValueError("output_mismatch_partial_reward must be non-negative")
    if partial_reward == 0.0:
        return 0.0, False, "disabled"

    correct_reward_floor = float(config["init_correct_weight"])
    if partial_reward >= correct_reward_floor:
        raise ValueError(
            "output_mismatch_partial_reward must be lower than init_correct_weight "
            f"({partial_reward} >= {correct_reward_floor})"
        )

    status = env_state.get("status")
    if status != "completed":
        return 0.0, False, "timeout" if status == "timeout" else "env_not_completed"
    if bool(env_state.get("decoy_kernel", False)):
        return 0.0, False, "decoy"
    if bool(env_state.get("correctness", False)):
        return 0.0, False, "already_correct"

    metadata = env_state.get("metadata") if isinstance(env_state.get("metadata"), dict) else {}
    if env_state.get("error") == RUNTIME_ERROR:
        return 0.0, False, "runtime_error"
    if not bool(metadata.get("correctness_output_mismatch", False)):
        if not bool(env_state.get("compiled", False)):
            return 0.0, False, "not_compiled"
        if not bool(metadata.get("correctness_candidate_forward_completed", False)):
            return 0.0, False, "candidate_forward_not_completed"
        return 0.0, False, "no_output_mismatch"

    if env_state.get("compiled") is not True:
        raise AssertionError("KernelGym contract violation: correctness_output_mismatch=true requires compiled=true")
    return partial_reward, True, "applied"


def calculate_reward_speedup(env_state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    partial_reward, partial_applied, partial_reason = _resolve_output_mismatch_partial_credit(env_state, config)
    penalty_score = _resolve_penalty_score(env_state, config)
    failed_score = float(config["failed_score"])
    apply_penalty_score = bool(config.get("apply_penalty_score", False))
    if apply_penalty_score and bool(config.get("apply_failed_group_reward", False)):
        raise ValueError("apply_penalty_score and apply_failed_group_reward cannot both be enabled")

    if env_state.get("status") != "completed":
        reward = penalty_score if apply_penalty_score else failed_score
        return {
            **env_state,
            "reward": reward,
            "score": reward,
            "speedup": 0.0,
            "success": False,
            "correctness": False,
            "compiled": False,
            "partial_credit_output_mismatch": partial_applied,
            "partial_credit_output_mismatch_reason": partial_reason,
            "penalty_score": penalty_score,
            "reward_correctness_component": 0.0,
            "reward_performance_component": 0.0,
            "reward_coverage_component": 0.0,
            "reward_partial_component": 0.0,
            "reward_penalty_component": reward,
            "speedup_log_standard_error": None,
        }

    if env_state.get("decoy_kernel", False):
        reward = penalty_score if apply_penalty_score else failed_score
        return {
            **env_state,
            "reward": reward,
            "score": reward,
            "decoy_kernel": True,
            "success": False,
            "partial_credit_output_mismatch": partial_applied,
            "partial_credit_output_mismatch_reason": partial_reason,
            "penalty_score": penalty_score,
            "reward_correctness_component": 0.0,
            "reward_performance_component": 0.0,
            "reward_coverage_component": 0.0,
            "reward_partial_component": 0.0,
            "reward_penalty_component": reward,
            "speedup_log_standard_error": None,
        }

    correctness = bool(env_state.get("correctness", False))
    compiled = bool(env_state.get("compiled", False))
    speedup = env_state.get("speedup", 0.0)
    speedup = 0.0 if speedup is None else float(speedup)
    speedup_reward_mode = str(config.get("speedup_reward_mode", "legacy")).strip().lower()
    speedup_log_standard_error = None
    if speedup_reward_mode == "lcb_improvement" and math.isfinite(speedup) and speedup > 0.0:
        speedup_log_standard_error = _compute_speedup_log_standard_error(env_state.get("metadata"), config)

    speedup_reward = _compute_speedup_reward_value(
        speedup,
        speedup_reward_mode,
        env_state.get("metadata"),
        config,
    )

    correctness_reward = 0.0
    performance_reward = 0.0
    coverage_reward = 0.0
    partial_component = 0.0
    penalty_component = 0.0
    if apply_penalty_score and not (compiled and correctness):
        reward = penalty_score
        penalty_component = reward
    elif not compiled:
        reward = failed_score
        penalty_component = reward
    elif partial_applied:
        reward = partial_reward
        partial_component = reward
    else:
        correctness_reward = float(config["init_correct_weight"]) if correctness else failed_score
        performance_reward = float(config["init_performance_weight"]) * speedup_reward
        if config.get("performance_reward_requires_correctness", False):
            performance_reward *= float(correctness)
        reward = correctness_reward + performance_reward

    coverage_info = {
        "coverage": 0.0,
        "num_custom_kernel": 0,
        "num_total_kernels": 0,
        "custom_kernel_cuda_time_in_profiling_us": 0,
        "total_kernel_run_time_in_profiling_us": 0,
    }
    if correctness:
        coverage_info = _compute_coverage(env_state, config)
        if config["coverage_reward_enable"]:
            coverage_reward = float(config["coverage_reward_weight"]) * coverage_info["coverage"]
            reward += coverage_reward

    return {
        **env_state,
        "reward": reward,
        "score": reward,
        "speedup": speedup,
        "success": compiled and correctness,
        "correctness": correctness,
        "compiled": compiled,
        "profiling": env_state.get("profiling"),
        "partial_credit_output_mismatch": partial_applied,
        "partial_credit_output_mismatch_reason": partial_reason,
        "penalty_score": penalty_score,
        "reward_correctness_component": correctness_reward,
        "reward_performance_component": performance_reward,
        "reward_coverage_component": coverage_reward,
        "reward_partial_component": partial_component,
        "reward_penalty_component": penalty_component,
        "speedup_reward": speedup_reward,
        "speedup_log_standard_error": speedup_log_standard_error,
        **coverage_info,
    }
