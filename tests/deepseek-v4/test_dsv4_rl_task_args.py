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

REPO = Path(__file__).resolve().parents[2]
HELPER = REPO / "scripts" / "dsv4" / "_dsv4_task_args.sh"


def _task_args(**env):
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
    out = subprocess.run(
        ["bash", "-c", script],
        env={"PATH": "/usr/bin:/bin", **base},
        capture_output=True,
        text=True,
    )
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
