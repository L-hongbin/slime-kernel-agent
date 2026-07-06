CUDA_AGENT_CONFIGS = {
    "max_feedback_chars": 0,
    "log_multi_turn_sample_rate": 0.01,
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
        # 600s per eval task (user-directed "large enough timeout" for the L2/L3 re-runs,
        # 2026-07-04): at 300s, 3.6% of L2-T1 and 19% of L3-T1 samples were execution-phase
        # task timeouts; 600s resolved all 66 L3 cases in the isolated re-test.
        "kernel_eval_task_timeout": 600,
        # Client wait scaled with the task budget: 32 in-flight / 4 GPU workers can queue
        # up to ~8 rounds x 600s in the worst case.
        "kernel_eval_client_timeout": 4800,
        "kernel_eval_poll_interval": 1.0,
        "kernel_eval_heartbeat_interval": 60.0,
        "kernel_eval_worker_max_concurrency": 32,
        "kernel_eval_rate_limit": 32,
        "kernel_eval_acquire_timeout": 4800,
        "num_correct_trials": 5,
        "num_perf_trials": 100,
        "verbose_errors": True,
        "enable_profiling": True,
        "detect_decoy_kernel": True,
        # split-compile=False: keep compile+load co-resident in ONE worker process so PyTorch's
        # per-process JIT extension versioner stays consistent → fixes the cross-process
        # `<name>_vN.so: cannot open` false COMPILATION_ERROR (verified root cause). The server-side
        # SPLIT_COMPILE_AND_EXECUTE flag cannot override a client True, so it must be set here.
        "split_compile_and_execute": False,
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
