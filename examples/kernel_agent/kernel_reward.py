from typing import Any

import torch

try:
    from .utils import COMPILATION_ERROR, IMPORT_ERROR, PRECHECK_ERROR, SYNTAX_ERROR, VALIDATION_ERROR
except ImportError:
    from utils import COMPILATION_ERROR, IMPORT_ERROR, PRECHECK_ERROR, SYNTAX_ERROR, VALIDATION_ERROR


def calculate_reward(env_result: dict[str, Any], config: dict[str, Any]) -> float:
    env_state = env_result.get("env_state") if isinstance(env_result, dict) else {}
    return calculate_reward_speedup(env_state, config)["reward"]


def reward_post_process_by_group(args, samples):
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    rewards = list(raw_rewards)

    idx_to_group_index: dict[int, object] = {}
    reward_groups: dict[object, list[float]] = {}
    for idx, sample in enumerate(samples):
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
            "std": group_rewards.std().item(),
            "size": len(group_reward_values),
        }

    for idx, raw_reward in enumerate(raw_rewards):
        group_index = idx_to_group_index[idx]
        stats = group_stats[group_index]
        reward = raw_reward - stats["mean"]

        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
            reward = reward / (stats["std"] + 1e-6)

        if args.advantage_estimator == "rloo":
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
    custom_time = result.get(
        "custom_kernel_cuda_time_in_profiling_us",
        metadata.get("custom_kernel_cuda_time_in_profiling_us", 0),
    ) or 0
    total_time = result.get(
        "total_kernel_run_time_in_profiling_us",
        metadata.get("total_kernel_run_time_in_profiling_us", 0),
    ) or 0

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


def calculate_reward_speedup(env_state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
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
        }

    if env_state.get("decoy_kernel", False):
        reward = float(config["penalty_score"])
        return {
            **env_state,
            "reward": reward,
            "score": reward,
            "decoy_kernel": True,
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
    else:
        reward = float(config["init_correct_weight"]) * float(correctness) + float(
            config["init_performance_weight"]
        ) * reward_speedup

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
            reward += float(config["coverage_reward_weight"]) * coverage_info["coverage"]

    return {
        **env_state,
        "reward": reward,
        "score": reward,
        "speedup": speedup,
        "success": compiled and correctness,
        "correctness": correctness,
        "compiled": compiled,
        "profiling": env_state.get("profiling"),
        **coverage_info,
    }
