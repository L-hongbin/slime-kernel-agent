import random
from typing import Any

import torch

try:
    from .utils import COMPILATION_ERROR, IMPORT_ERROR, PRECHECK_ERROR, SYNTAX_ERROR, VALIDATION_ERROR
except ImportError:
    from utils import COMPILATION_ERROR, IMPORT_ERROR, PRECHECK_ERROR, SYNTAX_ERROR, VALIDATION_ERROR


def calculate_reward(env_result: dict[str, Any], config: dict[str, Any]) -> float:
    env_state = env_result.get("env_state") if isinstance(env_result, dict) else {}
    return calculate_reward_speedup(env_state, config)["reward"]


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
    if args.advantage_estimator == "trloo":
        raw_rewards = [sample.metadata["multi_turn_reward"] for sample in samples]
    else:
        raw_rewards = [sample.get_reward_value(args) for sample in samples]
    rewards = [None] * len(raw_rewards)
    use_conditional_truncation_mask = getattr(args, "use_conditional_truncation_mask", False)

    idx_to_group_index: dict[int, object] = {}
    reward_groups: dict[object, list[float]] = {}
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
        reward_groups.setdefault(group_index, []).append(raw_rewards[idx])

    group_stats: dict[object, dict[str, float]] = {}
    for group_index, group_reward_values in reward_groups.items():
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


def _resolve_failure_reward(env_state: dict[str, Any], config: dict[str, Any]) -> float:
    metadata = env_state.get("metadata") if isinstance(env_state.get("metadata"), dict) else {}
    error = env_state.get("error")
    error_message = str(env_state.get("error_message") or "")
    lower_error_message = error_message.lower()
    precheck_error_codes = {PRECHECK_ERROR, VALIDATION_ERROR, SYNTAX_ERROR, IMPORT_ERROR}
    if config["apply_precheck_fail_penalty"] and (
        metadata.get("client_precheck") or error in precheck_error_codes or "pre-check error" in lower_error_message
    ):
        return float(config["precheck_fail_penalty"])
    if config["apply_compilation_fail_penalty"] and (
        error == COMPILATION_ERROR or "kernel compilation error" in lower_error_message
    ):
        return float(config["compilation_fail_penalty"])
    return float(config["penalty_score"])


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
    if metadata.get("runtime_error") or metadata.get("correctness_runtime_error"):
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

    if env_state.get("status") != "completed":
        reward = _resolve_failure_reward(env_state, config)
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
        }

    if env_state.get("decoy_kernel", False):
        reward = float(config["penalty_score"])
        return {
            **env_state,
            "reward": reward,
            "score": reward,
            "decoy_kernel": True,
            "success": False,
            "partial_credit_output_mismatch": partial_applied,
            "partial_credit_output_mismatch_reason": partial_reason,
        }

    correctness = bool(env_state.get("correctness", False))
    compiled = bool(env_state.get("compiled", False))
    speedup = env_state.get("speedup", 0.0)
    speedup = 0.0 if speedup is None else float(speedup)

    reward_speedup = min(speedup, float(config["speedup_reward_upper_bound"]))
    if reward_speedup < float(config["speedup_reward_lower_bound"]):
        reward_speedup = 0.0

    if not compiled and config["apply_compilation_fail_penalty"]:
        reward = float(config["compilation_fail_penalty"])
    elif partial_applied:
        reward = partial_reward
    else:
        correctness_reward = float(config["init_correct_weight"]) * float(correctness)
        performance_reward = float(config["init_performance_weight"]) * reward_speedup
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
        **coverage_info,
    }
