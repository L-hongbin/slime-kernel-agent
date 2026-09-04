import logging
from typing import Any

import numpy as np
import torch

from slime.observability import logging_utils
from slime.observability.metric_utils import (
    compute_pass_rate,
    compute_rollout_step,
    compute_statistics,
    dict_add_prefix,
    has_repetition,
)
from slime.utils.misc import group_by, load_function
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

_SGLANG_REQUEST_PERF_FIELDS = (
    ("request/e2e_latency", "e2e_latency"),
    ("request/queue_time", "queue_time"),
    ("decode/throughput", "decode_throughput"),
)
_SGLANG_PREFILL_PERF_FIELDS = (
    ("prefill/bootstrap_queue_duration", "pd_prefill_bootstrap_queue_duration"),
    ("prefill/bootstrap_duration", "pd_prefill_bootstrap_duration"),
    ("prefill/alloc_wait_duration", "pd_prefill_alloc_wait_duration"),
    ("prefill/forward_duration", "pd_prefill_forward_duration"),
    ("prefill/transfer_queue_duration", "pd_prefill_transfer_queue_duration"),
    ("prefill/transfer_speed_gb_s", "pd_transfer_speed_gb_s"),
    ("prefill/transfer_total_mb", "pd_transfer_total_mb"),
    ("prefill/retry_count", "pd_prefill_retry_count"),
)
_SGLANG_DECODE_PERF_FIELDS = (
    ("decode/prealloc_duration", "pd_decode_prealloc_duration"),
    ("decode/bootstrap_duration", "pd_decode_bootstrap_duration"),
    ("decode/alloc_wait_duration", "pd_decode_alloc_wait_duration"),
    ("decode/transfer_duration", "pd_decode_transfer_duration"),
    ("decode/forward_duration", "pd_decode_forward_duration"),
)


def compute_metrics_from_samples(args, samples):
    response_lengths = [sample.effective_response_length for sample in samples]

    log_dict = {}
    log_dict |= dict_add_prefix(compute_statistics(response_lengths), "response_len/")
    log_dict |= _compute_kernel_agent_metrics(samples)
    if getattr(args, "use_multi_turn", False):
        log_dict |= _compute_kernel_multi_turn_metrics(args, samples)
    log_dict |= _compute_zero_std_metrics(args, samples)
    log_dict |= _compute_spec_metrics(args, samples)
    log_dict |= _compute_prefix_cache_metrics(samples)
    log_dict |= _compute_reward_cat_metrics(args, samples)
    log_dict |= _compute_top_p_kept_vocab_metrics(samples)
    if getattr(args, "log_response_diversity", False):
        log_dict |= _compute_response_diversity(args, samples)
    log_dict["repetition_frac"] = np.mean([int(has_repetition(s.response)) for s in samples]).item()
    log_dict["truncated_ratio"] = np.mean([int(s.status == Sample.Status.TRUNCATED) for s in samples]).item()
    return log_dict


def _iter_response_diversity_groups(args, samples):
    if any(sample.rollout_id is not None or sample.group_index is not None for sample in samples):
        groups = {}
        for sample in samples:
            metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
            turn_idx = metadata.get("turn_idx")
            if sample.group_index is not None:
                # Compare responses generated for the same prompt and turn.
                # A multi-turn trajectory has one rollout_id per response, so
                # rollout_id-first grouping would incorrectly measure each
                # trajectory in isolation instead of response diversity.
                group_key = ("group", sample.group_index, turn_idx)
            elif sample.rollout_id is not None:
                group_key = ("rollout", sample.rollout_id, turn_idx)
            else:
                group_key = ("sample", sample.index, turn_idx)
            groups.setdefault(group_key, []).append(sample)
        return groups.values()

    group_size = max(int(getattr(args, "n_samples_per_prompt", 1) or 1), 1)
    return (samples[index : index + group_size] for index in range(0, len(samples), group_size))


def _compute_response_diversity(args, samples) -> dict[str, float]:
    token_bits = 32
    max_token = (1 << token_bits) - 1
    tail_mask = (1 << (token_bits * 3)) - 1
    diversities = []

    for group in _iter_response_diversity_groups(args, samples):
        total = 0
        unique = set()
        for sample in group:
            response_length = int(sample.response_length or 0)
            tokens = sample.tokens[-response_length:] if response_length >= 4 else []
            if len(tokens) < 4:
                continue
            total += len(tokens) - 3
            first = tokens[:4]
            if any(token < 0 or token > max_token for token in first):
                raise ValueError(f"token id exceeds response diversity token_bits={token_bits}")
            key = (((first[0] << token_bits) | first[1]) << token_bits | first[2]) << token_bits | first[3]
            unique.add(key)
            for token in tokens[4:]:
                if token < 0 or token > max_token:
                    raise ValueError(f"token id {token} exceeds response diversity token_bits={token_bits}")
                key = ((key & tail_mask) << token_bits) | token
                unique.add(key)
        if total > 0:
            diversities.append(len(unique) / total)

    return {"response_diversity": float(np.mean(diversities).item()) if diversities else 0.0}


_KERNEL_FAST_THRESHOLDS = (1.0, 1.2, 1.5, 2.0, 3.0)


def _compute_kernel_multi_turn_metrics(args, samples):
    values_by_turn = {}
    sample_trajectories = {}
    for sample in samples:
        metadata = sample.metadata or {}
        env_extra_info = metadata.get("env_extra_info")
        if "turn_idx" not in metadata or not isinstance(env_extra_info, dict):
            continue
        try:
            turn_idx = int(metadata["turn_idx"])
        except (TypeError, ValueError):
            continue

        speedup = env_extra_info.get("speedup")
        if isinstance(speedup, bool) or not isinstance(speedup, (int, float)):
            continue
        correctness = bool(env_extra_info.get("correctness")) and not bool(env_extra_info.get("decoy_kernel"))
        compilation = bool(env_extra_info.get("compilation"))
        turn_result = {
            "correctness": correctness,
            "compilation": compilation,
            "speedup": float(speedup),
        }
        if sample.index is not None:
            sample_trajectories.setdefault(sample.index, {})[turn_idx] = turn_result

        values = values_by_turn.setdefault(
            turn_idx,
            {
                "correctness": [],
                "compilation": [],
                "speedup": [],
                "fast": {threshold: [] for threshold in _KERNEL_FAST_THRESHOLDS},
            },
        )
        values["correctness"].append(float(correctness))
        values["compilation"].append(float(compilation))
        values["speedup"].append(float(speedup))
        for threshold in _KERNEL_FAST_THRESHOLDS:
            values["fast"][threshold].append(float(correctness and speedup >= threshold))

    metrics = {}
    for turn_idx, values in sorted(values_by_turn.items()):
        prefix = f"kernel/turn{turn_idx}"
        for key in ("correctness", "compilation", "speedup"):
            if values[key]:
                metrics[f"{prefix}/{key}"] = np.mean(values[key]).item()
        for threshold, threshold_values in values["fast"].items():
            if threshold_values:
                metrics[f"{prefix}/fast@{threshold:g}"] = np.mean(threshold_values).item()
    metrics |= _compute_kernel_trajectory_metrics(args, sample_trajectories)
    return metrics


def _compute_kernel_trajectory_metrics(args, sample_trajectories):
    max_turns = int(getattr(args, "max_turns", 1) or 1)
    if max_turns <= 1:
        return {}

    first_turn_correct = []
    last_turn_correct = []
    improved_samples = 0
    regressed_samples = 0
    best_by_turn = {
        turn_count: {"correctness": [], "compilation": [], "speedup": []} for turn_count in range(1, max_turns + 1)
    }
    for trajectory in sample_trajectories.values():
        sorted_turns = [trajectory[turn_idx] for turn_idx in sorted(trajectory)]
        if not sorted_turns:
            continue
        first_correct = bool(sorted_turns[0]["correctness"])
        last_correct = bool(sorted_turns[-1]["correctness"])
        first_turn_correct.append(float(first_correct))
        last_turn_correct.append(float(last_correct))
        improved_samples += int(not first_correct and last_correct)
        regressed_samples += int(first_correct and not last_correct)

        for turn_count, values in best_by_turn.items():
            visible_turns = sorted_turns[:turn_count]
            values["correctness"].append(float(any(turn["correctness"] for turn in visible_turns)))
            values["compilation"].append(float(any(turn["compilation"] for turn in visible_turns)))
            values["speedup"].append(max(float(turn["speedup"]) for turn in visible_turns))

    if not first_turn_correct:
        return {}
    metrics = {
        "kernel/correct_improvement/first_turn_correct_rate": np.mean(first_turn_correct).item(),
        "kernel/correct_improvement/last_turn_correct_rate": np.mean(last_turn_correct).item(),
        "kernel/correct_improvement/improved_samples": improved_samples,
        "kernel/correct_improvement/regressed_samples": regressed_samples,
        "kernel/correct_improvement/net_improvement": improved_samples - regressed_samples,
    }
    for turn_count, values in best_by_turn.items():
        for key, metric_values in values.items():
            if metric_values:
                metrics[f"kernel/trajectory/best_by_turn_{turn_count}/{key}"] = np.mean(metric_values).item()
    return metrics


def _compute_kernel_agent_metrics(samples):
    bool_keys = {
        "correctness",
        "compilation",
        "decoy_kernel",
        "correctness_candidate_forward_completed",
        "correctness_output_mismatch",
    }
    coverage_keys = {"time_coverage", "num_coverage"}
    values_by_key = {}
    decoy_reason_count = {}
    incorrect_backend_probe_skip_reason_count = {}
    overlong_penalty_values = []
    time_values = {
        "model_time": [],
        "env_time": [],
        "detail_env_time/compile_time": [],
        "detail_env_time/kernel_runtime": [],
        "detail_env_time/profile_time": [],
        "detail_env_time/ncu_profile_time_s": [],
        "detail_env_time/runtime_sanitizer_time_s": [],
        "detail_env_time/refer_runtime": [],
    }
    total_count = len(samples)
    coverage_rs_masked_count = 0
    conditional_truncation_masked_count = 0
    correct_count = 0
    coverage_rs_correct_masked_count = 0
    precheck_count = 0
    precheck_passed_count = 0
    env_status_count = 0
    env_timeout_count = 0
    kernel_eval_client_timeout_count = 0
    non_pad_count = 0
    generate_guard_timeout_count = 0
    incorrect_backend_probe_attempted_count = 0
    incorrect_backend_probe_valid_count = 0
    incorrect_backend_probe_custom_kernel_observed_count = 0
    incorrect_backend_probe_decoy_detected_count = 0
    partial_credit_applied_count = 0
    partial_credit_rejected_decoy_count = 0
    partial_credit_rejected_runtime_error_count = 0
    partial_credit_rejected_timeout_count = 0

    def record_reason(counter: dict[str, int], value) -> None:
        if isinstance(value, str) and value:
            key = "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")
            if key:
                counter[key] = counter.get(key, 0) + 1

    for sample in samples:
        metadata = sample.metadata or {}
        partial_reason = metadata.get("partial_credit_output_mismatch_reason")
        if metadata.get("partial_credit_output_mismatch") is True:
            partial_credit_applied_count += 1
        elif partial_reason == "decoy":
            partial_credit_rejected_decoy_count += 1
        elif partial_reason == "runtime_error":
            partial_credit_rejected_runtime_error_count += 1
        elif partial_reason == "timeout":
            partial_credit_rejected_timeout_count += 1
        overlong_penalty = metadata.get("overlong_penalty")
        if isinstance(overlong_penalty, (int, float)) and not isinstance(overlong_penalty, bool):
            overlong_penalty_values.append(float(overlong_penalty))
        coverage_masked = sample.remove_sample and metadata.get("remove_reason") == "coverage_rs"
        conditional_masked = bool(metadata.get("conditional_truncation_masked"))
        coverage_rs_masked_count += int(coverage_masked)
        conditional_truncation_masked_count += int(conditional_masked)

        if not metadata.get("is_pad_turn"):
            non_pad_count += 1
            generate_guard_timeout_count += int(
                sample.status == Sample.Status.ABORTED and metadata.get("abort_reason") == "wall_clock_timeout"
            )

        model_time = metadata.get("model_time")
        if isinstance(model_time, (int, float)) and not isinstance(model_time, bool):
            time_values["model_time"].append(float(model_time))

        env_extra_info = metadata.get("env_extra_info")
        if not isinstance(env_extra_info, dict):
            continue
        env_time = metadata.get("env_time")
        if (
            env_extra_info.get("precheck") != "failed"
            and isinstance(env_time, (int, float))
            and not isinstance(env_time, bool)
        ):
            time_values["env_time"].append(float(env_time))
            detail_env_time = env_extra_info.get("detail_env_time")
            if isinstance(detail_env_time, dict):
                for key in (
                    "compile_time",
                    "kernel_runtime",
                    "profile_time",
                    "ncu_profile_time_s",
                    "runtime_sanitizer_time_s",
                    "refer_runtime",
                ):
                    value = detail_env_time.get(key)
                    if isinstance(value, (int, float)) and not isinstance(value, bool):
                        time_values[f"detail_env_time/{key}"].append(float(value))

        env_result = metadata.get("env_result")
        env_state = env_result.get("env_state") if isinstance(env_result, dict) else {}
        env_state = env_state if isinstance(env_state, dict) else {}
        status = env_state.get("status")
        if status is not None:
            env_status_count += 1
            if status == "timeout":
                env_timeout_count += 1
                error_message = str(env_state.get("error_message") or env_state.get("error") or "")
                kernel_eval_client_timeout_count += int("client-side" in error_message)

        precheck = env_extra_info.get("precheck")
        if precheck in {"passed", "failed"}:
            precheck_count += 1
            precheck_passed_count += int(precheck == "passed")

        record_reason(decoy_reason_count, env_extra_info.get("decoy_reason"))
        record_reason(
            incorrect_backend_probe_skip_reason_count,
            env_extra_info.get("incorrect_backend_probe_skip_reason"),
        )
        incorrect_backend_probe_attempted_count += int(env_extra_info.get("incorrect_backend_probe_attempted") is True)
        incorrect_backend_probe_valid_count += int(env_extra_info.get("incorrect_backend_probe_valid") is True)
        incorrect_backend_probe_custom_kernel_observed_count += int(
            env_extra_info.get("incorrect_backend_probe_custom_kernel_observed") is True
        )
        incorrect_backend_probe_decoy_detected_count += int(
            env_extra_info.get("incorrect_backend_probe_decoy_detected") is True
        )

        is_correct = bool(env_extra_info.get("correctness")) and not bool(env_extra_info.get("decoy_kernel"))
        correct_count += int(is_correct)
        coverage_rs_correct_masked_count += int(is_correct and coverage_masked)

        for key, value in env_extra_info.items():
            if key in bool_keys and isinstance(value, bool):
                values_by_key.setdefault(key, []).append(float(value))
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                values_by_key.setdefault(key, []).append(float(value))

    metrics = {}
    for key, values in values_by_key.items():
        prefix = "coverage" if key in coverage_keys else "env_extra_info"
        stats = {"mean": np.mean(values).item()} if key in bool_keys else compute_statistics(values)
        for stat_key in ("mean",) if key in bool_keys else ("min", "max", "mean"):
            metrics[f"{prefix}/{key}/{stat_key}"] = stats[stat_key]
    if total_count:
        metrics["sample_mask/coverage_rs_masked_fraction"] = coverage_rs_masked_count / total_count
        metrics["sample_mask/conditional_truncation_masked_fraction"] = (
            conditional_truncation_masked_count / total_count
        )
    if correct_count:
        metrics["sample_mask/coverage_rs_correct_masked_fraction"] = coverage_rs_correct_masked_count / correct_count
    if precheck_count:
        metrics["kernel/precheck_pass_rate"] = precheck_passed_count / precheck_count
    if env_status_count:
        metrics["kernel/eval_timeout_count"] = env_timeout_count
        metrics["kernel/eval_timeout_ratio"] = env_timeout_count / env_status_count
        metrics["kernel/eval_client_timeout_count"] = kernel_eval_client_timeout_count
        metrics["kernel/eval_client_timeout_ratio"] = kernel_eval_client_timeout_count / env_status_count
    if non_pad_count:
        metrics["kernel/generate_guard_timeout_count"] = generate_guard_timeout_count
        metrics["kernel/generate_guard_timeout_ratio"] = generate_guard_timeout_count / non_pad_count
        metrics["kernel/partial_credit/applied_rate"] = partial_credit_applied_count / non_pad_count
        metrics["kernel/partial_credit/rejected_decoy_count"] = partial_credit_rejected_decoy_count
        metrics["kernel/partial_credit/rejected_runtime_error_count"] = partial_credit_rejected_runtime_error_count
        metrics["kernel/partial_credit/rejected_timeout_count"] = partial_credit_rejected_timeout_count
        for reason, count in decoy_reason_count.items():
            metrics[f"kernel/decoy_reason/{reason}_count"] = count
        for reason, count in incorrect_backend_probe_skip_reason_count.items():
            metrics[f"kernel/incorrect_backend_probe/skip_{reason}_count"] = count
        metrics["kernel/incorrect_backend_probe/attempted_count"] = incorrect_backend_probe_attempted_count
        metrics["kernel/incorrect_backend_probe/attempted_ratio"] = (
            incorrect_backend_probe_attempted_count / non_pad_count
        )
        if incorrect_backend_probe_attempted_count:
            metrics["kernel/incorrect_backend_probe/valid_count"] = incorrect_backend_probe_valid_count
            metrics["kernel/incorrect_backend_probe/valid_ratio_of_attempted"] = (
                incorrect_backend_probe_valid_count / incorrect_backend_probe_attempted_count
            )
        if incorrect_backend_probe_valid_count:
            metrics["kernel/incorrect_backend_probe/custom_kernel_observed_ratio_of_valid"] = (
                incorrect_backend_probe_custom_kernel_observed_count / incorrect_backend_probe_valid_count
            )
            metrics["kernel/incorrect_backend_probe/decoy_detected_ratio_of_valid"] = (
                incorrect_backend_probe_decoy_detected_count / incorrect_backend_probe_valid_count
            )
    if overlong_penalty_values:
        metrics["kernel/overlong_penalty/mean"] = np.mean(overlong_penalty_values).item()
    for key, values in time_values.items():
        if not values:
            continue
        metrics[f"kernel/time/{key}/mean"] = np.mean(values).item()
        metrics[f"kernel/time/{key}/sum"] = np.sum(values).item()
        metrics[f"kernel/time/{key}/count"] = len(values)
        metrics[f"kernel/time/{key}/p50"] = np.percentile(values, 50).item()
        metrics[f"kernel/time/{key}/p90"] = np.percentile(values, 90).item()
        metrics[f"kernel/time/{key}/p95"] = np.percentile(values, 95).item()
        metrics[f"kernel/time/{key}/max"] = np.max(values).item()
    return metrics


def compute_perf_metrics_from_samples(args, samples, rollout_time):
    non_generation_time = [sample.non_generation_time for sample in samples]

    log_dict = {}
    log_dict["rollout_time"] = rollout_time
    if max(non_generation_time) > 0:
        log_dict |= dict_add_prefix(compute_statistics(non_generation_time), "non_generation_time/")

    def token_perf(response_lengths, non_generation_time, key=""):
        max_response_length = max(response_lengths)
        if args.rollout_num_gpus:
            log_dict[f"{key}tokens_per_gpu_per_sec"] = sum(response_lengths) / rollout_time / args.rollout_num_gpus
        log_dict[f"longest_{key}sample_tokens_per_sec"] = max_response_length / rollout_time

        if max(non_generation_time) == 0:
            return

        non_generation_time = [
            t for t, length in zip(non_generation_time, response_lengths, strict=True) if length == max_response_length
        ]
        mean_non_generation_time = sum(non_generation_time) / len(non_generation_time)

        log_dict[f"longest_{key}sample_non_generation_time"] = mean_non_generation_time
        log_dict[f"longest_{key}sample_tokens_per_sec_without_non_generation"] = max_response_length / (
            rollout_time - mean_non_generation_time
        )

    token_perf([sample.response_length for sample in samples], non_generation_time, key="")
    token_perf([sample.effective_response_length for sample in samples], non_generation_time, key="effective_")
    log_dict |= _compute_sglang_request_perf_metrics(samples)

    return log_dict


def _compute_sglang_request_perf_metrics(all_samples: list[Sample]):
    attrs_by_request = list(_iter_sglang_generate_attrs(all_samples))
    if not attrs_by_request:
        return {}

    values_by_metric: dict[str, list[float]] = {}
    profiled_request_count = 0

    def add_value(metric_key: str, source_key: str, attrs: dict) -> bool:
        value = attrs.get(source_key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
            return False
        values_by_metric.setdefault(metric_key, []).append(float(value))
        return True

    for attrs in attrs_by_request:
        request_has_perf = False

        for metric_key, source_key in _SGLANG_REQUEST_PERF_FIELDS:
            request_has_perf |= add_value(metric_key, source_key, attrs)

        for metric_key, source_key in _SGLANG_PREFILL_PERF_FIELDS:
            request_has_perf |= add_value(metric_key, source_key, attrs)

        for metric_key, source_key in _SGLANG_DECODE_PERF_FIELDS:
            request_has_perf |= add_value(metric_key, source_key, attrs)

        if request_has_perf:
            profiled_request_count += 1

    metrics: dict[str, float] = {}
    for key, values in values_by_metric.items():
        if not values:
            continue
        metrics |= dict_add_prefix(compute_statistics(values), f"{key}/")

    return metrics


def _iter_sglang_generate_attrs(all_samples: list[Sample]):
    for sample in all_samples:
        trace = getattr(sample, "trace", None)
        if not isinstance(trace, dict):
            continue
        for event in trace.get("events") or []:
            if event.get("type") != "span_end" or event.get("name") != "sglang_generate":
                continue
            attrs = event.get("attrs")
            if isinstance(attrs, dict):
                yield attrs


def _compute_zero_std_metrics(args, all_samples: list[Sample]):
    # only compute in GRPO-like algorithms where one prompt has multiple responses
    if args.advantage_estimator == "ppo":
        return {}

    def _is_zero_std(samples: list[Sample]):
        rewards = [sample.get_reward_value(args) for sample in samples]
        return len(rewards) == 0 or all(rewards[0] == r for r in rewards)

    all_sample_groups = group_by(all_samples, lambda s: s.group_index)
    interesting_sample_groups = [g for g in all_sample_groups.values() if _is_zero_std(g)]

    interesting_rewards = [str(round(g[0].get_reward_value(args), 1)) for g in interesting_sample_groups]

    return {f"zero_std/count_{reward}": len(items) for reward, items in group_by(interesting_rewards).items()}


def _compute_top_p_kept_vocab_metrics(all_samples: list[Sample]):
    total_kept = 0
    total_tokens = 0
    for sample in all_samples:
        offsets = sample.rollout_top_p_token_offsets
        if offsets is None or sample.response_length == 0:
            continue
        offsets = torch.as_tensor(offsets, dtype=torch.int64)
        if offsets.numel() == 0:
            continue
        assert (
            offsets.numel() == sample.response_length + 1
        ), f"top-p token offsets length {offsets.numel()} != response length + 1 {sample.response_length + 1}"
        if sample.remove_sample:
            continue
        if sample.loss_mask is None:
            total_kept += int(offsets[-1] - offsets[0])
            total_tokens += sample.response_length
            continue
        loss_mask = torch.as_tensor(sample.loss_mask, dtype=torch.bool, device=offsets.device)
        assert (
            loss_mask.numel() == sample.response_length
        ), f"loss mask length {loss_mask.numel()} != response length {sample.response_length}"
        total_kept += int(torch.diff(offsets)[loss_mask].sum())
        total_tokens += int(loss_mask.sum())
    if total_tokens == 0:
        return {}
    return {"top_p_kept_vocab_per_token": total_kept / total_tokens}


def _compute_spec_metrics(args, all_samples: list[Sample]):
    if getattr(args, "sglang_speculative_algorithm", None) is None:
        return {}
    num_samples = len(all_samples)
    metrics = {}
    metrics["spec_accept_rate"] = sum(sample.spec_info.spec_accept_rate for sample in all_samples) / num_samples
    metrics["spec_accept_length"] = sum(sample.spec_info.spec_accept_length for sample in all_samples) / num_samples
    return metrics


def _compute_prefix_cache_metrics(all_samples: list[Sample]):
    num_samples = len(all_samples)
    metrics = {}
    total_cached_tokens = sum(sample.prefix_cache_info.cached_tokens for sample in all_samples)
    total_prompt_tokens = sum(sample.prefix_cache_info.total_prompt_tokens for sample in all_samples)

    metrics["prefix_cache_hit_rate"] = total_cached_tokens / total_prompt_tokens if total_prompt_tokens > 0 else 0.0
    metrics["avg_cached_tokens_per_sample"] = total_cached_tokens / num_samples
    return metrics


def _compute_reward_cat_metrics(args, all_samples: list[Sample]):
    reward_cat_key = args.log_reward_category
    if reward_cat_key is None:
        return {}

    samples_of_reward_cat = group_by(all_samples, lambda s: s.reward[reward_cat_key])

    return {f"error_cat/{reward_cat}": len(s) / len(all_samples) for reward_cat, s in samples_of_reward_cat.items()}


def log_eval_rollout_data(rollout_id, args, data, extra_metrics: dict[str, Any] | None = None):
    if args.custom_eval_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_eval_rollout_log_function_path)
        if custom_log_func(rollout_id, args, data, extra_metrics):
            return

    log_dict = extra_metrics or {}
    for key in data.keys():
        rewards = data[key]["rewards"]
        log_dict[f"eval/{key}"] = sum(rewards) / len(rewards)
        if (samples := data[key].get("samples")) is not None:
            log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), f"eval/{key}/")
        if "truncated" in data[key]:
            truncated = data[key]["truncated"]
            log_dict[f"eval/{key}-truncated_ratio"] = sum(truncated) / len(truncated)
        if args.log_passrate:
            log_dict |= dict_add_prefix(
                compute_pass_rate(
                    flat_rewards=rewards,
                    group_size=args.n_samples_per_eval_prompt,
                ),
                f"eval/{key}-",
            )

    logger.info(f"eval {rollout_id}: {log_dict}")

    step = compute_rollout_step(args, rollout_id)
    log_dict["eval/step"] = step
    if args.wandb_always_use_train_step:
        log_dict["train/step"] = step
        log_dict["rollout/step"] = step
    logging_utils.log(args, log_dict, step_key="eval/step")

    return log_dict


def log_rollout_data(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
    if args.custom_rollout_log_function_path is not None:
        custom_log_func = load_function(args.custom_rollout_log_function_path)
        if custom_log_func(rollout_id, args, samples, rollout_extra_metrics, rollout_time):
            return

    if args.load_debug_rollout_data:
        return

    log_dict = {**(rollout_extra_metrics or {})}
    log_dict |= dict_add_prefix(compute_metrics_from_samples(args, samples), "rollout/")
    log_dict |= dict_add_prefix(compute_perf_metrics_from_samples(args, samples, rollout_time), "perf/")
    logger.info(f"perf {rollout_id}: {log_dict}")
    step = compute_rollout_step(args, rollout_id)
    log_dict["rollout/step"] = step
    if args.wandb_always_use_train_step:
        log_dict["train/step"] = step
    logging_utils.log(args, log_dict, step_key="rollout/step")
