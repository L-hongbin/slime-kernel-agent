import os

CUDA_AGENT_CONFIGS = {
    "max_feedback_chars": 0,
    "log_multi_turn_sample_rate": 0.01,
    "log_multi_turn_full_text": False,
    "log_slowest_step_window": 10,
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
        "kernel_eval_task_timeout": 300,
        "kernel_eval_client_timeout": 2400,
        "kernel_eval_poll_interval": 1.0,
        "kernel_eval_heartbeat_interval": 60.0,
        # Eval jobs can lower these independently when sharing the KernelGym
        # backend pool with training. Training keeps the historical default 32.
        "kernel_eval_worker_max_concurrency": int(os.environ.get("KERNEL_EVAL_WORKER_MAX_CONCURRENCY", "32")),
        "kernel_eval_rate_limit": int(os.environ.get("KERNEL_EVAL_RATE_LIMIT", "32")),
        "kernel_eval_priority": os.environ.get("KERNEL_EVAL_PRIORITY", "normal"),
        "kernel_eval_acquire_timeout": 2400,
        "num_correct_trials": 5,
        # 2026-07-11 (user direction): 30 warmup + 50 timed trials (was 3+100).
        "num_perf_trials": 50,
        "num_warmup": 30,
        "verbose_errors": True,
        "enable_profiling": True,
        "detect_decoy_kernel": True,
        "split_compile_and_execute": True,
        "enable_compile_artifact_cache": True,
    },
    "reward": {
        "init_correct_weight": 0.5,
        "init_performance_weight": 0.5,
        "speedup_reward_upper_bound": 3.0,
        "speedup_reward_lower_bound": 0.0,
        "penalty_score": 0,
        "compilation_fail_penalty": 0,
        "precheck_fail_penalty": 0,
        "apply_compilation_fail_penalty": True,
        "apply_precheck_fail_penalty": True,
        "coverage_reward_enable": True,
        "coverage_reward_type": "time_coverage",
        "coverage_reward_weight": 0.5,
    },
}
