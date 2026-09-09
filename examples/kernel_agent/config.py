import os

# Rollout request settings.
rollout_request_max_retries = max(1, int(os.environ.get("ROLLOUT_REQUEST_MAX_RETRIES", 60)))
# Rollout logging settings.
log_rollout_info = bool(int(os.environ.get("CUDA_AGENT_LOG_ROLLOUT_INFO", 1)))
log_rollout_info_rate = float(os.environ.get("CUDA_AGENT_LOG_ROLLOUT_INFO_RATE", 0.01))
log_multi_turn_info = bool(int(os.environ.get("CUDA_AGENT_LOG_MULTI_TURN_TEXT", 1)))
log_rollout_stats_only = bool(int(os.environ.get("CUDA_AGENT_LOG_ROLLOUT_STATS_ONLY", 1)))
log_first_rollout = bool(int(os.environ.get("CUDA_AGENT_LOG_FIRST_ROLLOUT", 1)))
log_slowest_info = bool(int(os.environ.get("CUDA_AGENT_LOG_SLOWEST_INFO", 1)))
log_slowest_step_window = int(os.environ.get("CUDA_AGENT_LOG_SLOWEST_STEP_WINDOW", 100))
log_slowest_min_delta_seconds = float(os.environ.get("CUDA_AGENT_LOG_SLOWEST_MIN_DELTA_SECONDS", 10.0))
# KernelGym request lifecycle settings.
kernel_eval_heartbeat_interval = float(os.environ.get("CUDA_AGENT_KERNEL_EVAL_HEARTBEAT_INTERVAL", 60.0))
kernel_eval_task_timeout = float(os.environ.get("CUDA_AGENT_KERNEL_EVAL_TASK_TIMEOUT", 300.0))
# Correctness and performance trial settings.
num_correct_trials = int(os.environ.get("CUDA_AGENT_NUM_CORRECT_TRIALS", 5))
num_perf_trials = int(os.environ.get("CUDA_AGENT_NUM_PERF_TRIALS", 50))
# Warmup iterations before timed trials, and number of high/low trials trimmed
# from each end before the mean is computed. Both reference and kernel are timed
# under these identical settings (KernelGYM defaults: num_warmup=3, trim=0).
num_warmup = int(os.environ.get("CUDA_AGENT_NUM_WARMUP", 30))
perf_trim_count = int(os.environ.get("CUDA_AGENT_PERF_TRIM_COUNT", 0))
# Reuse a cached reference runtime (keyed by uuid) instead of re-timing the
# reference every turn. Gives a stable speedup denominator across turns/samples
# of the same problem. Requires uuid in the payload; only applied when present.
use_reference_cache = bool(int(os.environ.get("CUDA_AGENT_USE_REFERENCE_CACHE", 1)))
# Adaptive kernel-perf trials: run at least perf_min_trials, then continue only
# while timing CV > perf_cv_threshold, up to num_perf_trials. Default OFF (opt-in);
# enable with CUDA_AGENT_ADAPTIVE_PERF_TRIALS=1.
adaptive_perf_trials = bool(int(os.environ.get("CUDA_AGENT_ADAPTIVE_PERF_TRIALS", 0)))
perf_min_trials = int(os.environ.get("CUDA_AGENT_PERF_MIN_TRIALS", 20))
perf_cv_threshold = float(os.environ.get("CUDA_AGENT_PERF_CV_THRESHOLD", 0.05))
# Reference perf trials; None -> reuse num_perf_trials on the server.
refer_num_perf_trials = (
    int(os.environ["CUDA_AGENT_REFER_NUM_PERF_TRIALS"]) if os.environ.get("CUDA_AGENT_REFER_NUM_PERF_TRIALS") else None
)
# Correctness-stage timeout overrides; None -> use the server's config/formula.
# correctness_timeout: explicit budget in seconds. enabled: per-request on/off.
correctness_timeout = (
    float(os.environ["CUDA_AGENT_CORRECTNESS_TIMEOUT"]) if os.environ.get("CUDA_AGENT_CORRECTNESS_TIMEOUT") else None
)
_cte = os.environ.get("CUDA_AGENT_CORRECTNESS_TIMEOUT_ENABLED")
correctness_timeout_enabled = None if _cte is None else bool(int(_cte))
# Reward settings.
# Speedup reward mapping. ``legacy`` preserves the historical clipped raw
# speedup. ``improvement`` maps [1x, upper_bound] to [0, 1].
# ``lcb_improvement`` applies a lower confidence bound to speedup first, using
# timing statistics returned by KernelGYM, and then uses the same mapping.
speedup_reward_mode = os.environ.get("CUDA_AGENT_SPEEDUP_REWARD_MODE", "legacy").strip().lower()
if speedup_reward_mode not in {"legacy", "improvement", "lcb_improvement"}:
    raise ValueError("CUDA_AGENT_SPEEDUP_REWARD_MODE must be one of: legacy, improvement, lcb_improvement")
speedup_uncertainty_z_score = float(os.environ.get("CUDA_AGENT_SPEEDUP_UNCERTAINTY_Z_SCORE", 1.96))
speedup_uncertainty_log_std_floor = float(os.environ.get("CUDA_AGENT_SPEEDUP_UNCERTAINTY_LOG_STD_FLOOR", 0.0))
enable_dynamic_reward_weight = bool(int(os.environ.get("CUDA_AGENT_ENABLE_DYNAMIC_REWARD_WEIGHT", "0")))
if speedup_uncertainty_z_score < 0.0:
    raise ValueError("CUDA_AGENT_SPEEDUP_UNCERTAINTY_Z_SCORE must be non-negative")
if speedup_uncertainty_log_std_floor < 0.0:
    raise ValueError("CUDA_AGENT_SPEEDUP_UNCERTAINTY_LOG_STD_FLOOR must be non-negative")
# Optional reward for a candidate that compiled, completed its forward pass,
# and reached KernelGym's shape/value comparison but produced a wrong output.
# Default off so launchers keep their historical reward policy. Set a positive
# value explicitly to enable the reviewed output-mismatch partial reward.
output_mismatch_partial_reward = float(os.environ.get("CUDA_AGENT_OUTPUT_MISMATCH_PARTIAL_REWARD", 0.0))
performance_reward_requires_correctness = bool(
    int(os.environ.get("CUDA_AGENT_PERFORMANCE_REWARD_REQUIRES_CORRECTNESS", "0"))
)
apply_failed_group_reward = bool(int(os.environ.get("CUDA_AGENT_APPLY_FAILED_GROUP_REWARD", "0")))
apply_penalty_score = bool(int(os.environ.get("CUDA_AGENT_APPLY_PENALTY_SCORE", "0")))
if apply_penalty_score and apply_failed_group_reward:
    raise ValueError("CUDA_AGENT_APPLY_PENALTY_SCORE and CUDA_AGENT_APPLY_FAILED_GROUP_REWARD cannot both be enabled")
# KernelGYM diagnostics and validation features controlled by each request.
# NCU, Compute Sanitizer, correctness input perturbations, and adaptive perf
# trials are opt-in because they add latency or change the evaluated inputs.
enable_profiling = bool(int(os.environ.get("CUDA_AGENT_ENABLE_PROFILING", 1)))
enable_ncu = bool(int(os.environ.get("CUDA_AGENT_ENABLE_NCU", 0)))
enable_compute_sanitizer = bool(int(os.environ.get("CUDA_AGENT_ENABLE_COMPUTE_SANITIZER", 0)))
compute_sanitizer_mode = os.environ.get("CUDA_AGENT_COMPUTE_SANITIZER_MODE", "error_based").strip().lower()
if compute_sanitizer_mode not in {"error_based", "full"}:
    raise ValueError("CUDA_AGENT_COMPUTE_SANITIZER_MODE must be 'error_based' or 'full'")
enable_correctness_input_perturbations = bool(
    int(os.environ.get("CUDA_AGENT_ENABLE_CORRECTNESS_INPUT_PERTURBATIONS", 0))
)
# Memory usage guard settings.
_memory_ratio_threshold = os.environ.get("CUDA_AGENT_MEMORY_RATIO_THRESHOLD", "1.8").strip()
memory_ratio_threshold = (
    None if _memory_ratio_threshold.lower() in {"", "none", "null"} else float(_memory_ratio_threshold)
)
if memory_ratio_threshold is not None and memory_ratio_threshold <= 1.0:
    raise ValueError("CUDA_AGENT_MEMORY_RATIO_THRESHOLD must be greater than 1.0, or null to disable")
# Combined configuration consumed by the rollout pipeline.
CUDA_AGENT_CONFIGS = {
    # Rollout request settings.
    "rollout_request_max_retries": rollout_request_max_retries,
    # Rollout logging settings.
    "max_feedback_chars": 0,
    "log_multi_turn_sample_rate": 0.01,
    "log_rollout_info": log_rollout_info,
    "log_rollout_info_rate": log_rollout_info_rate,
    "log_multi_turn_info": log_multi_turn_info,
    "log_rollout_stats_only": log_rollout_stats_only,
    "log_first_rollout": log_first_rollout,
    "log_slowest_info": log_slowest_info,
    "log_slowest_step_window": log_slowest_step_window,
    "log_slowest_min_delta_seconds": log_slowest_min_delta_seconds,
    "slowest_tracker_timeout": 2.0,
    # Rollout filtering settings.
    "filter": {
        "reject_low_variance_groups": True,
        "reject_small_groups": True,
        "target_group_size": None,
        "min_group_size": None,
        "reward_std_threshold": 1e-3,
    },
    "env": {
        # KernelGym request lifecycle settings.
        "kernel_eval_function_path": None,
        "kernel_eval_max_retries": 3,
        "kernel_eval_task_timeout": kernel_eval_task_timeout,
        "kernel_eval_client_timeout": 2400,
        "kernel_eval_poll_interval": 1.0,
        "kernel_eval_heartbeat_interval": kernel_eval_heartbeat_interval,
        # KernelGym scheduling settings. Eval jobs can lower these independently
        # when sharing the backend pool with training; training defaults to 32.
        "kernel_eval_worker_max_concurrency": int(os.environ.get("KERNEL_EVAL_WORKER_MAX_CONCURRENCY", "32")),
        "kernel_eval_rate_limit": int(os.environ.get("KERNEL_EVAL_RATE_LIMIT", "32")),
        "kernel_eval_priority": os.environ.get("KERNEL_EVAL_PRIORITY", "normal"),
        "kernel_eval_acquire_timeout": 2400,
        # Correctness and performance trial settings.
        "num_correct_trials": num_correct_trials,
        # 2026-07-11 (user direction): 30 warmup + 50 timed trials (was 3+100).
        "num_perf_trials": num_perf_trials,
        "num_warmup": num_warmup,
        "perf_trim_count": perf_trim_count,
        # Reference cache and adaptive performance settings.
        "use_reference_cache": use_reference_cache,
        "adaptive_perf_trials": adaptive_perf_trials,
        "perf_min_trials": perf_min_trials,
        "perf_cv_threshold": perf_cv_threshold,
        "refer_num_perf_trials": refer_num_perf_trials,
        # Correctness timeout settings.
        "correctness_timeout": correctness_timeout,
        "correctness_timeout_enabled": correctness_timeout_enabled,
        # Diagnostic and validation settings.
        "verbose_errors": True,
        "enable_profiling": enable_profiling,
        "enable_ncu": enable_ncu,
        "enable_compute_sanitizer": enable_compute_sanitizer,
        "compute_sanitizer_mode": compute_sanitizer_mode,
        "enable_correctness_input_perturbations": enable_correctness_input_perturbations,
        "simplify_error": True,
        # Memory and decoy-kernel guards.
        "memory_ratio_threshold": memory_ratio_threshold,
        "detect_decoy_kernel": True,
        # Compilation pipeline settings.
        "split_compile_and_execute": True,
        "enable_compile_artifact_cache": True,
    },
    # Reward settings.
    "reward": {
        "init_correct_weight": 0.5,
        "init_performance_weight": 0.5,
        "speedup_reward_mode": speedup_reward_mode,
        "speedup_reward_upper_bound": 2.0,
        "speedup_reward_lower_bound": 0.0,
        "speedup_uncertainty_z_score": speedup_uncertainty_z_score,
        "speedup_uncertainty_log_std_floor": speedup_uncertainty_log_std_floor,
        # Keep the configured 0.5 maxima, but gate both auxiliary objectives by
        # sqrt(max((num_correct - 1) / (group_size - 1), 0)).
        "enable_dynamic_reward_weight": enable_dynamic_reward_weight,
        "failed_score": 0.0,
        "apply_penalty_score": apply_penalty_score,
        "apply_failed_group_reward": apply_failed_group_reward,
        # These scores are only used when every valid sample in a reward group
        # has task reward equal to failed_score and apply_failed_group_reward is
        # enabled. They rank progress without changing non-failure groups.
        "penalty_score": {
            "precheck": -1.0,
            "compilation": -0.75,
            "runtime": -0.5,
            "correctness": -0.25,
            "decoy": -1.0,
            "other": -1.0,
        },
        "coverage_reward_enable": True,
        "coverage_reward_type": "time_coverage",
        "coverage_reward_weight": 0.5,
        "output_mismatch_partial_reward": output_mismatch_partial_reward,
        "performance_reward_requires_correctness": performance_reward_requires_correctness,
    },
}
