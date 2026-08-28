"""Unit tests for scripts/dsv4/_dsv4_task_args.sh::build_dsv4_task_args.

Pins the RL launcher's arg wiring (borrowed from run.t1.qwen3.6.27B.fasync.sh)
so a regression in the task-arg assembly is caught without a cluster run:
  - smoke_sft stays byte-identical to the validated R6 SFT smoke,
  - rl mode emits policy_loss + trloo advantages + eps-clip (never sft_loss /
    --disable-compute-advantages-and-returns),
  - REWARD_MODE=random isolates the RL math (--rm-type random, no KernelGym),
  - REWARD_MODE=drkernel wires the real custom generate/reward/filter fns.
"""

import subprocess
from pathlib import Path

import pytest

NUM_GPUS = 0

REPO = Path(__file__).resolve().parents[2]
HELPER = REPO / "scripts" / "dsv4" / "_dsv4_task_args.sh"


def _task_args_result(**env):
    """Source the helper, run build_dsv4_task_args, print TASK_ARGS one per line."""
    base = {
        "REPO": str(REPO),
        "PROMPT_DATA": "/data/x.parquet",
        "INPUT_KEY": "prompt",
        "LABEL_KEY": "reward_model",
        "METADATA_KEY": "extra_info",
        "MAX_CONTEXT_LEN": "16384",
        "MAX_RESPONSE_LEN": "8192",
        "ROLLOUT_MAX_PROMPT_LEN": "8192",
        "ROLLOUT_TEMPERATURE": "1",
        "ROLLOUT_TOP_P": "1",
        "ADVANTAGE_ESTIMATOR": "trloo",
        "EPS_CLIP": "0.2",
        "EPS_CLIP_HIGH": "0.28",
        "ENTROPY_COEF": "0.00",
        "KERNEL_ENV_URL": "http://127.0.0.1:20211",
        "KERNEL_BACKEND": "tvm_ffi",
        "USE_WANDB": "0",
        "DEBUG_DIR": "/scratch/debug",
        "USE_ROLLOUT_ROUTING_REPLAY": "1",
    }
    base.update(env)
    script = f'set -euo pipefail\nsource "{HELPER}"\nbuild_dsv4_task_args\nprintf "%s\\n" "${{TASK_ARGS[@]}}"\n'
    return subprocess.run(
        ["bash", "-c", script],
        env={"PATH": "/usr/bin:/bin", **base},
        capture_output=True,
        text=True,
    )


def _task_args(**env):
    out = _task_args_result(**env)
    assert out.returncode == 0, out.stderr
    return out.stdout.splitlines()


def _joined(lines):
    return " ".join(lines)


def test_smoke_sft_unchanged():
    a = _task_args(TASK_MODE="smoke_sft")
    j = _joined(a)
    assert "sft_loss" in j
    assert "--disable-compute-advantages-and-returns" in a
    assert "--rm-type" in a and "random" in a
    # tiny smoke windows, not the RL context length
    assert "512" in a and "16" in a
    assert "policy_loss" not in j
    assert "--advantage-estimator" not in a


def test_rl_random_isolates_math():
    a = _task_args(TASK_MODE="rl", REWARD_MODE="random")
    j = _joined(a)
    assert "policy_loss" in j
    assert a.count("--advantage-estimator") == 1
    assert "trloo" in a
    assert "--eps-clip" in a and "0.2" in a
    assert "--eps-clip-high" in a and "0.28" in a
    # RL math isolation: random reward, no KernelGym custom fns
    assert "--rm-type" in a and "random" in a
    assert "--custom-rm-path" not in a
    assert "--kernel-env-url" not in a
    # RL must not carry the SFT-only flags
    assert "sft_loss" not in j
    assert "--disable-compute-advantages-and-returns" not in a
    # real context window flows through
    assert "16384" in a
    # prompt-len pinned so prompt+response never exceeds the SGLang context
    assert "--rollout-max-prompt-len" in a and "8192" in a


def test_rl_pg_reduction_defaults_to_global_token_mean_and_allows_explicit_diagnostic_override():
    default_args = _task_args(TASK_MODE="rl", REWARD_MODE="random")
    assert "--calculate-per-token-loss" in default_args
    assert "--custom-pg-loss-reducer-function-path" not in default_args

    completion_equal = _task_args(
        TASK_MODE="rl",
        REWARD_MODE="random",
        CALCULATE_PER_TOKEN_LOSS="0",
        CUSTOM_PG_LOSS_REDUCER_FUNCTION_PATH=(
            "examples.kernel_agent.diagnostic_reducers.get_completion_mean_pg_loss_reducer"
        ),
    )
    assert "--calculate-per-token-loss" not in completion_equal
    reducer_pos = completion_equal.index("--custom-pg-loss-reducer-function-path")
    assert completion_equal[reducer_pos + 1].endswith("get_completion_mean_pg_loss_reducer")


def test_rl_pg_reduction_rejects_invalid_boolean_value():
    result = _task_args_result(
        TASK_MODE="rl",
        REWARD_MODE="random",
        CALCULATE_PER_TOKEN_LOSS="yes",
    )
    assert result.returncode == 2
    assert "CALCULATE_PER_TOKEN_LOSS must be 0 or 1" in result.stderr


def test_predictive_topk_dppo_args_are_conditional_and_paper_defaults_are_wired():
    default_args = _task_args(TASK_MODE="rl", REWARD_MODE="random")
    assert "--dppo-predictive-top-k" not in default_args
    assert "--dppo-predictive-tail-estimator" not in default_args

    predictive = _task_args(
        TASK_MODE="rl",
        REWARD_MODE="random",
        POLICY_LOSS_MODE="dppo_topk_kl_predictive",
        USE_ROLLOUT_LOGPROBS="1",
        MAX_CONTEXT_LEN="12288",
        MAX_RESPONSE_LEN="12288",
        ROLLOUT_MAX_PROMPT_LEN="8192",
        EPS_CLIP="0.20",
        EPS_CLIP_HIGH="0.20",
        EPS_CLIP_C="5",
    )
    assert predictive[predictive.index("--policy-loss-mode") + 1] == "dppo_topk_kl_predictive"
    assert predictive[predictive.index("--dppo-predictive-top-k") + 1] == "20"
    assert predictive[predictive.index("--dppo-predictive-tail-estimator") + 1] == "aggregated"
    assert predictive[predictive.index("--eps-clip") + 1] == "0.20"
    assert predictive[predictive.index("--eps-clip-high") + 1] == "0.20"
    assert predictive[predictive.index("--eps-clip-c") + 1] == "5"
    assert predictive[predictive.index("--rollout-max-context-len") + 1] == "12288"
    assert predictive[predictive.index("--rollout-max-response-len") + 1] == "12288"
    assert "--use-rollout-logprobs" in predictive


def test_predictive_topk_dppo_args_accept_explicit_estimator_override():
    predictive = _task_args(
        TASK_MODE="rl",
        REWARD_MODE="random",
        POLICY_LOSS_MODE="dppo_topk_kl_predictive",
        DPPO_PREDICTIVE_TOP_K="7",
        DPPO_PREDICTIVE_TAIL_ESTIMATOR="uniform",
    )
    assert predictive[predictive.index("--dppo-predictive-top-k") + 1] == "7"
    assert predictive[predictive.index("--dppo-predictive-tail-estimator") + 1] == "uniform"


def test_dis_args_use_rollout_anchor_without_dppo_support_or_ratio_cap():
    dis = _task_args(
        TASK_MODE="rl",
        REWARD_MODE="random",
        POLICY_LOSS_MODE="dis",
        USE_ROLLOUT_LOGPROBS="1",
        EPS_CLIP="0.80",
        EPS_CLIP_HIGH="3.0",
        DIS_RATIO_LEVEL="sequence",
    )
    assert dis[dis.index("--policy-loss-mode") + 1] == "dis"
    assert dis[dis.index("--eps-clip") + 1] == "0.80"
    assert dis[dis.index("--eps-clip-high") + 1] == "3.0"
    assert dis[dis.index("--dis-ratio-level") + 1] == "sequence"
    assert "--use-rollout-logprobs" in dis
    assert "--eps-clip-c" not in dis
    assert "--dppo-predictive-top-k" not in dis
    assert "--dppo-predictive-tail-estimator" not in dis


def test_rl_drkernel_wires_real_task():
    a = _task_args(TASK_MODE="rl", REWARD_MODE="drkernel")
    j = _joined(a)
    assert "policy_loss" in j
    assert "examples.kernel_agent.generate_with_cuda_agent.generate" in a
    assert "examples.kernel_agent.generate_with_cuda_agent.reward_func" in a
    assert "examples.kernel_agent.kernel_filter.filter_cuda_kernel_group" in a
    assert "--kernel-env-url" in a and "http://127.0.0.1:20211" in a
    assert "--use-multi-turn" in a
    # REQUIRED with turns_geometric sequence-mis: slime_validate_args raises
    # without it (arguments.py). Gating it on MAX_TURNS>1 was launch-breaking
    # (codex milestone review 2026-07-04) — must always be present in drkernel mode.
    assert "--enable-turns-dp-partitions" in a
    assert any("turns_geometric" in x for x in a)
    # drkernel uses the real reward, not the random stand-in
    assert "--rm-type" not in a
    # thinking chat template for the real task
    assert '{"enable_thinking":true}' in a


def test_dynamic_batch_uses_explicit_slime_token_limits():
    args = _task_args(
        TASK_MODE="rl",
        REWARD_MODE="random",
        MAX_TOKENS_PER_GPU="12288",
        LOG_PROBS_MAX_TOKENS_PER_GPU="8192",
    )
    assert "--use-dynamic-batch-size" in args
    assert args[args.index("--max-tokens-per-gpu") + 1] == "12288"
    assert args[args.index("--log-probs-max-tokens-per-gpu") + 1] == "8192"


def test_padded_length_descending_train_order_is_explicit_and_validated():
    default_args = _task_args(TASK_MODE="rl", REWARD_MODE="random")
    assert "--sort-train-microbatches-by-padded-length-desc" not in default_args

    descending_args = _task_args(
        TASK_MODE="rl",
        REWARD_MODE="random",
        SORT_TRAIN_MICROBATCHES_BY_PADDED_LENGTH_DESC="1",
    )
    assert "--sort-train-microbatches-by-padded-length-desc" in descending_args

    invalid = _task_args_result(
        TASK_MODE="rl",
        REWARD_MODE="random",
        SORT_TRAIN_MICROBATCHES_BY_PADDED_LENGTH_DESC="yes",
    )
    assert invalid.returncode == 2
    assert "SORT_TRAIN_MICROBATCHES_BY_PADDED_LENGTH_DESC must be 0 or 1" in invalid.stderr


def test_overlong_penalty_is_explicit_drkernel_cli():
    args = _task_args(
        TASK_MODE="rl",
        REWARD_MODE="drkernel",
        OVERLONG_PENALTY="1",
        OVERLONG_BUFFER_LEN="1024",
        OVERLONG_PENALTY_FACTOR="0.5",
    )
    assert "--overlong-penalty" in args
    assert args[args.index("--overlong-buffer-len") + 1] == "1024"
    assert args[args.index("--overlong-penalty-factor") + 1] == "0.5"


def test_entropy_diagnostics_are_explicit_cli_flags():
    args = _task_args(
        TASK_MODE="rl",
        REWARD_MODE="random",
        ENTROPY_COMMON_PROBE="1",
        ASSERT_ZERO_LORA_OUT="1",
    )
    assert "--entropy-common-probe" in args
    assert "--assert-zero-lora-out" in args


def test_rl_wandb_toggle():
    off = _task_args(TASK_MODE="rl", REWARD_MODE="random", USE_WANDB="0")
    assert "--use-wandb" not in off
    on = _task_args(
        TASK_MODE="rl",
        REWARD_MODE="random",
        USE_WANDB="1",
        WANDB_PROJECT="slime",
        WANDB_GROUP="v4grp",
    )
    assert "--use-wandb" in on
    assert "slime" in on and "v4grp" in on


def test_routing_replay_toggle():
    on = _task_args(TASK_MODE="rl", REWARD_MODE="random", USE_ROLLOUT_ROUTING_REPLAY="1")
    assert "--use-rollout-routing-replay" in on
    off = _task_args(TASK_MODE="rl", REWARD_MODE="random", USE_ROLLOUT_ROUTING_REPLAY="0")
    assert "--use-rollout-routing-replay" not in off


def test_rollout_dataset_resume_root_is_explicitly_wired():
    default_args = _task_args(TASK_MODE="rl", REWARD_MODE="random")
    assert "--rollout-dataset-load" not in default_args

    resume_args = _task_args(
        TASK_MODE="rl",
        REWARD_MODE="random",
        ROLLOUT_DATASET_LOAD="/dataset/resume",
    )
    pos = resume_args.index("--rollout-dataset-load")
    assert resume_args[pos + 1] == "/dataset/resume"


def test_forge_replay_keeps_normal_rl_task_but_overrides_rollout_function():
    args = _task_args(
        TASK_MODE="rl",
        REWARD_MODE="drkernel",
        LOAD_FORGE_ROLLOUT_DATA="/replay/rollout_0.pt",
    )

    function_pos = args.index("--rollout-function-path")
    assert args[function_pos + 1] == "slime.rollout.forge_load.generate_rollout"
    data_pos = args.index("--load-forge-rollout-data")
    assert args[data_pos + 1] == "/replay/rollout_0.pt"
    # Keep the original DrKernel train-data/reward semantics.  The forge rollout
    # function bypasses generation; these args still drive reward normalization,
    # turn-aware DP partitioning and the policy loss on the captured samples.
    assert "--custom-reward-post-process-path" in args
    assert "--enable-turns-dp-partitions" in args
    assert "--use-rollout-routing-replay" in args


@pytest.mark.parametrize(
    "conflict",
    [
        {"LOAD_DEBUG_ROLLOUT_DATA": "/debug/rollout_0.pt"},
        {"DEBUG_TRAIN_ONLY": "1"},
        {"DEBUG_ROLLOUT_ONLY": "1"},
    ],
)
def test_forge_replay_rejects_modes_that_disable_training_or_sglang(conflict):
    result = _task_args_result(
        TASK_MODE="rl",
        REWARD_MODE="drkernel",
        LOAD_FORGE_ROLLOUT_DATA="/replay/rollout_0.pt",
        **conflict,
    )
    assert result.returncode == 2
    assert "LOAD_FORGE_ROLLOUT_DATA keeps SGLang + weight sync live" in result.stderr


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
