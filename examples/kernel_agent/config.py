import os

log_rollout_info = bool(int(os.environ.get("CUDA_AGENT_LOG_ROLLOUT_INFO", 1)))
log_rollout_info_rate = float(os.environ.get("CUDA_AGENT_LOG_ROLLOUT_INFO_RATE", 0.01))
# When True, [rollout_info] logs only stats (summary + per-turn metrics) and skips the
# verbose text: prompt, response_think, response_content, format_feedback, and [messages].
log_rollout_stats_only = bool(int(os.environ.get("CUDA_AGENT_LOG_ROLLOUT_STATS_ONLY", 0)))
kernel_eval_heartbeat_interval = float(os.environ.get("CUDA_AGENT_KERNEL_EVAL_HEARTBEAT_INTERVAL", 60.0))
kernel_eval_task_timeout = float(os.environ.get("CUDA_AGENT_KERNEL_EVAL_TASK_TIMEOUT", 600.0))
num_correct_trials = float(os.environ.get("CUDA_AGENT_NUM_CORRECT_TRIALS", 5.0))
num_perf_trials = float(os.environ.get("CUDA_AGENT_NUM_PERF_TRIALS", 100.0))
# Warmup iterations before timed trials, and number of high/low trials trimmed
# from each end before the mean is computed. Both reference and kernel are timed
# under these identical settings (KernelGYM defaults: num_warmup=3, trim=0).
num_warmup = int(os.environ.get("CUDA_AGENT_NUM_WARMUP", 3))
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

_memory_ratio_threshold = os.environ.get("CUDA_AGENT_MEMORY_RATIO_THRESHOLD", "1.8").strip()
memory_ratio_threshold = (
    None if _memory_ratio_threshold.lower() in {"", "none", "null"} else float(_memory_ratio_threshold)
)
if memory_ratio_threshold is not None and memory_ratio_threshold <= 1.0:
    raise ValueError("CUDA_AGENT_MEMORY_RATIO_THRESHOLD must be greater than 1.0, or null to disable")

CUDA_AGENT_CONFIGS = {
    "max_feedback_chars": 0,
    "log_rollout_info": log_rollout_info,
    "log_rollout_info_rate": log_rollout_info_rate,
    "log_rollout_stats_only": log_rollout_stats_only,
    "log_slowest_step_window": 50,
    "log_slowest_min_delta_seconds": 5.0,
    "slowest_tracker_timeout": 2.0,
    "filter": {
        "reject_low_variance_groups": True,
        "reject_small_groups": True,
        "target_group_size": None,
        "min_group_size": None,
        "reward_std_threshold": 1e-3,
    },
    "env": {
        "kernel_eval_function_path": None,
        "kernel_eval_max_retries": 3,
        "kernel_eval_task_timeout": kernel_eval_task_timeout,
        "kernel_eval_client_timeout": 2400,
        "kernel_eval_poll_interval": 1.0,
        "kernel_eval_heartbeat_interval": kernel_eval_heartbeat_interval,
        "kernel_eval_worker_max_concurrency": 32,
        "kernel_eval_rate_limit": 32,
        "kernel_eval_acquire_timeout": 2400,
        "num_correct_trials": num_correct_trials,
        "num_perf_trials": num_perf_trials,
        "num_warmup": num_warmup,
        "perf_trim_count": perf_trim_count,
        "use_reference_cache": use_reference_cache,
        "adaptive_perf_trials": adaptive_perf_trials,
        "perf_min_trials": perf_min_trials,
        "perf_cv_threshold": perf_cv_threshold,
        "refer_num_perf_trials": refer_num_perf_trials,
        "correctness_timeout": correctness_timeout,
        "correctness_timeout_enabled": correctness_timeout_enabled,
        "verbose_errors": True,
        "enable_profiling": enable_profiling,
        "enable_ncu": enable_ncu,
        "enable_compute_sanitizer": enable_compute_sanitizer,
        "compute_sanitizer_mode": compute_sanitizer_mode,
        "enable_correctness_input_perturbations": enable_correctness_input_perturbations,
        "memory_ratio_threshold": memory_ratio_threshold,
        "detect_decoy_kernel": True,
        "split_compile_and_execute": True,
        "enable_compile_artifact_cache": True,
    },
    "reward": {
        "init_correct_weight": 0.5,
        "init_performance_weight": 0.5,
        "speedup_reward_upper_bound": 5.0,
        "speedup_reward_lower_bound": 0.0,
        "penalty_score": 0,
        "compilation_fail_penalty": 0,
        "precheck_fail_penalty": 0,
        "apply_compilation_fail_penalty": False,
        "apply_precheck_fail_penalty": False,
        "coverage_reward_enable": True,
        "coverage_reward_type": "time_coverage",
        "coverage_reward_weight": 0.5,
    },
}
