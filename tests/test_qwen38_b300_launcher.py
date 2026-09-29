import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "examples/kernel_agent/run_qwen38_b300_piecewise.sh"
pytestmark = pytest.mark.unit


def dry_run(tmp_path, **overrides):
    environment = {
        "PATH": os.environ["PATH"],
        "HOME": str(tmp_path),
        "CONFIG_DRY_RUN": "1",
        "B300_RUNTIME": str(tmp_path / "runtime"),
        "EXP_ROOT": str(tmp_path / "experiment"),
    }
    environment.update(overrides)
    return subprocess.run(["bash", str(LAUNCHER)], env=environment, capture_output=True, text=True, check=False)


def argument(tokens, name):
    return tokens[tokens.index(name) + 1]


def test_b300_defaults_preserve_approved_r9_configuration(tmp_path):
    result = dry_run(tmp_path)
    assert result.returncode == 0, result.stderr
    tokens = shlex.split(result.stdout)
    expected = {
        "--actor-num-nodes": "1",
        "--actor-num-gpus-per-node": "4",
        "--rollout-num-gpus": "4",
        "--tensor-model-parallel-size": "4",
        "--pipeline-model-parallel-size": "1",
        "--context-parallel-size": "1",
        "--rollout-num-gpus-per-engine": "1",
        "--rollout-batch-size": "16",
        "--n-samples-per-prompt": "16",
        "--global-batch-size": "128",
        "--max-tokens-per-gpu": "16384",
        "--log-probs-max-tokens-per-gpu": "16384",
        "--log-probs-chunk-size": "2048",
        "--recompute-granularity": "full",
        "--recompute-method": "block",
        "--recompute-num-layers": "8",
        "--num-rollout": "80",
        "--save-interval": "10",
        "--sglang-max-running-requests": "64",
        "--sglang-mem-fraction-static": "0.85",
        "--sglang-max-mamba-cache-size": "320",
        "--attention-backend": "flash",
        "--sglang-attention-backend": "trtllm_mha",
        "--rollout-max-context-len": "120000",
        "--rollout-max-response-len": "32000",
        "--lr": "1e-6",
        "--advantage-estimator": "trloo",
        "--policy-loss-mode": "dppo_binary_tv",
        "--dynamic-reward-gate": "piecewise",
        "--sglang-speculative-num-steps": "3",
        "--kernel-env-url": "http://192.168.112.55:20111",
    }
    for name, value in expected.items():
        assert argument(tokens, name) == value
    if "--sglang-cuda-graph-max-bs-decode" in tokens:
        assert argument(tokens, "--sglang-cuda-graph-max-bs-decode") == "64"
        assert argument(tokens, "--sglang-sampling-mask-max-tokens") == "32768"
        assert "--sglang-cuda-graph-max-bs" not in tokens
    else:
        assert argument(tokens, "--sglang-cuda-graph-max-bs") == "64"
        assert "--sglang-sampling-mask-max-tokens" not in tokens
    train_environment = json.loads(argument(tokens, "--train-env-vars"))
    assert train_environment["SLIME_COMM_MEMORY_CHECK_INTERVAL"] == "64"
    assert "--use-tensorboard" in tokens
    assert "--enable-fp32-lm-head" in tokens
    assert "--enable-mtp-training" in tokens
    assert "--log-device-memory-used" in tokens
    for forbidden in ("--use-wandb", "--save-hf", "--no-save-optim", "--colocate", "--load-debug-rollout-data"):
        assert forbidden not in tokens
    assert argument(tokens, "--load") == argument(tokens, "--hf-checkpoint")
    train_env = json.loads(argument(tokens, "--train-env-vars"))
    assert train_env["PYTHONPATH"].split(":")[0] == str(tmp_path / "runtime/train_packages")
    assert not (tmp_path / "runtime").exists()
    assert "\r" not in result.stdout


def test_recompute_can_be_disabled_for_explicit_comparisons(tmp_path):
    result = dry_run(tmp_path, RECOMPUTE_NUM_LAYERS="0")
    assert result.returncode == 0, result.stderr
    assert not any(token.startswith("--recompute-") for token in shlex.split(result.stdout))


@pytest.mark.parametrize(
    ("overrides", "expected_interval"),
    [({"SAVE_INTERVAL_STEPS": "40"}, "20"), ({"ROLLOUT_BATCH_SIZE": "8"}, "20")],
)
def test_save_interval_tracks_optimizer_steps(tmp_path, overrides, expected_interval):
    result = dry_run(tmp_path, **overrides)
    assert result.returncode == 0, result.stderr
    assert argument(shlex.split(result.stdout), "--save-interval") == expected_interval


@pytest.mark.parametrize(
    "overrides",
    [
        {"SAVE_INTERVAL_STEPS": "0"},
        {"SAVE_INTERVAL_STEPS": "3"},
        {"ROLLOUT_BATCH_SIZE": "15"},
        {"N_SAMPLES_PER_PROMPT": "0"},
        {"NUM_ROLLOUT": "-1"},
        {"ROLLOUT_TP_SIZE": "3"},
        {"ROLLOUT_TP_SIZE": "abc"},
        {"RECOMPUTE_NUM_LAYERS": "-1"},
        {"RECOMPUTE_NUM_LAYERS": "65"},
        {"RECOMPUTE_NUM_LAYERS": "abc"},
        {"VALIDATION_NO_SAVE": "1"},
        {"VALIDATION_NO_SAVE": "yes"},
        {"VALIDATION_ROLLOUT_ONLY": "yes"},
        {"VALIDATION_ROLLOUT_ONLY": "1", "LOAD_DEBUG_ROLLOUT_DATA": "/tmp/replay.pt"},
    ],
)
def test_invalid_batch_and_save_settings_fail_before_submission(tmp_path, overrides):
    result = dry_run(tmp_path, **overrides)
    assert result.returncode != 0
    assert result.stderr
    assert not result.stdout


def test_resume_and_attention_overrides_are_explicit(tmp_path):
    resume_path = str(tmp_path / "previous/checkpoints")
    result = dry_run(tmp_path, TRAIN_LOAD_PATH=resume_path, TRAIN_ATTENTION_BACKEND="fused")
    assert result.returncode == 0, result.stderr
    tokens = shlex.split(result.stdout)
    assert argument(tokens, "--load") == resume_path
    assert argument(tokens, "--attention-backend") == "fused"


def test_saved_rollout_replay_is_opt_in(tmp_path):
    replay_path = str(tmp_path / "rollout_0.pt")
    result = dry_run(tmp_path, LOAD_DEBUG_ROLLOUT_DATA=replay_path)
    assert result.returncode == 0, result.stderr
    assert argument(shlex.split(result.stdout), "--load-debug-rollout-data") == replay_path


def test_recompute_can_be_explicitly_reenabled(tmp_path):
    result = dry_run(tmp_path, RECOMPUTE_NUM_LAYERS="40")
    assert result.returncode == 0, result.stderr
    tokens = shlex.split(result.stdout)
    assert argument(tokens, "--recompute-num-layers") == "40"
    assert argument(tokens, "--recompute-granularity") == "full"
    assert argument(tokens, "--recompute-method") == "block"


def test_replay_benchmark_can_skip_checkpoint_io(tmp_path):
    result = dry_run(tmp_path, VALIDATION_NO_SAVE="1", LOAD_DEBUG_ROLLOUT_DATA=str(tmp_path / "rollout_0.pt"))
    assert result.returncode == 0, result.stderr
    tokens = shlex.split(result.stdout)
    assert "--save" not in tokens
    assert "--save-interval" not in tokens


def test_loss_chunk_size_can_be_benchmarked_without_changing_microbatch_budget(tmp_path):
    result = dry_run(tmp_path, LOG_PROBS_CHUNK_SIZE="1024")
    assert result.returncode == 0, result.stderr
    tokens = shlex.split(result.stdout)
    assert argument(tokens, "--log-probs-chunk-size") == "1024"
    assert argument(tokens, "--max-tokens-per-gpu") == "16384"


def test_rollout_only_validation_is_opt_in(tmp_path):
    result = dry_run(tmp_path, VALIDATION_ROLLOUT_ONLY="1")
    assert result.returncode == 0, result.stderr
    assert "--debug-rollout-only" in shlex.split(result.stdout)
    assert "--debug-rollout-only" not in shlex.split(dry_run(tmp_path).stdout)


def test_ray_start_dry_run_does_not_stop_existing_services(tmp_path):
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "examples/kernel_agent/start_qwen38_b300_ray.sh")],
        env={"PATH": os.environ["PATH"], "CONFIG_DRY_RUN": "1", "B300_RUNTIME": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    tokens = shlex.split(result.stdout)
    assert tokens[:2] == ["ray", "start"]
    assert argument(tokens, "--num-gpus") == "8"
    assert argument(tokens, "--dashboard-port") == "8269"


def test_socket_tmpdir_is_short_even_with_long_artifact_path(tmp_path):
    environment_script = REPO_ROOT / "examples/kernel_agent/qwen38_b300_env.sh"
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; printf "%s" "$TMPDIR"', "bash", str(environment_script)],
        env={"PATH": os.environ["PATH"], "CONFIG_DRY_RUN": "1", "B300_RUNTIME": str(tmp_path / ("long" * 40))},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == "/tmp/slime_qwen38_b300"


def test_launcher_rejects_overlong_socket_tmpdir(tmp_path):
    result = dry_run(tmp_path, B300_TMPDIR="/tmp/" + "long" * 30)
    assert result.returncode != 0
    assert "Unix socket" in result.stderr
