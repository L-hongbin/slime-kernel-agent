"""Static contract tests for the V4 gate launchers' argument wiring.

The R6 full-loop NaN root cause was a launcher flag (``--rollout-temperature 0``)
flowing into the train-side log-prob path — invisible to code-level unit tests
because only the launcher supplied the bad value. These tests pin the launcher
contracts statically so a config regression fails in CI, not 15 minutes into a
multi-node run:

- sampling defaults are on-policy and positive (temperature > 0, top-p wired),
- no launcher hardcodes ``--rollout-temperature 0`` again,
- the dump verifier is called with the actual expected sample count,
- the train-actor env (--train-env-vars) carries PYTHONPATH (megatron.training
  is not importable in Ray actors without it),
- the committed smoke prompt data satisfies the launchers' batch sizes.
"""

import json
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
R6 = REPO / "scripts" / "dsv4" / "full_loop_smoke.sh"
# The task-arg assembly (loss type, rollout-temperature wiring, etc.) was
# extracted into this sourceable helper so it can be unit-tested in isolation;
# the r6_text fixture reads both so these assertions still cover it.
R6_TASK_ARGS = REPO / "scripts" / "dsv4" / "_dsv4_task_args.sh"
R4 = REPO / "scripts" / "dsv4" / "rollout_smoke.sh"
R3 = REPO / "scripts" / "dsv4" / "train_smoke.sh"
ALL_LAUNCHERS = [R6, R4, R3]


def _default(script_text: str, var: str) -> str | None:
    """Extract the default from a `VAR=${VAR:-default}` line."""
    m = re.search(rf"^{var}=\$\{{{var}:-([^}}]*)\}}", script_text, flags=re.M)
    return m.group(1) if m else None


@pytest.fixture(scope="module")
def r6_text() -> str:
    return R6.read_text() + "\n" + R6_TASK_ARGS.read_text()


def test_r6_sampling_defaults_are_on_policy(r6_text):
    temperature = _default(r6_text, "ROLLOUT_TEMPERATURE")
    top_p = _default(r6_text, "ROLLOUT_TOP_P")
    assert temperature is not None and float(temperature) > 0, (
        "R6 must default rollout temperature > 0: temperature 0 (greedy) divides "
        "logits by zero in the train-side log-prob path -> NaN loss."
    )
    assert (
        float(temperature) == 1.0 and top_p is not None and float(top_p) == 1.0
    ), "R6 smoke should default to on-policy sampling (temperature=1, top_p=1)."
    assert '--rollout-temperature "${ROLLOUT_TEMPERATURE}"' in r6_text
    assert '--rollout-top-p "${ROLLOUT_TOP_P}"' in r6_text


@pytest.mark.parametrize("script", ALL_LAUNCHERS, ids=lambda p: p.name)
def test_no_launcher_hardcodes_temperature_zero(script):
    for line in script.read_text().splitlines():
        stripped = line.split("#", 1)[0]
        assert not re.search(r"--rollout-temperature\s+0(\.0*)?(\s|\\|$)", stripped), (
            f"{script.name} hardcodes --rollout-temperature 0; training rejects it "
            f"(NaN loss). Use temperature 1 or the ROLLOUT_TEMPERATURE variable."
        )


def test_r6_verifier_gets_expected_sample_count(r6_text):
    calls = re.findall(r"verify_rollout_dump.py[^\n]*", r6_text)
    assert calls, "R6 must verify the rollout dump after the loop"
    for call in calls:
        assert "--expected-samples" in call, (
            "verify_rollout_dump.py defaults to R4's 2 samples; R6 must pass "
            "--expected-samples (ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT)."
        )
        assert "ROLLOUT_BATCH_SIZE" in call and "N_SAMPLES_PER_PROMPT" in call


def test_r6_train_actor_env_carries_pythonpath(r6_text):
    assert "--train-env-vars" in r6_text, (
        "R6 must pass --train-env-vars: Ray train actors need explicit env "
        "(raylet env is not inherited reliably in direct-driver mode)."
    )
    m = re.search(r"TRAIN_ENV_VARS_JSON=\$\(python[^)]*?keys = \[(.*?)\]", r6_text, flags=re.S)
    assert m, "R6 should build TRAIN_ENV_VARS_JSON from an explicit key list"
    keys = m.group(1)
    for required in ["PYTHONPATH", "PATH", "NCCL_SOCKET_IFNAME", "NO_PROXY"]:
        assert f'"{required}"' in keys, (
            f"--train-env-vars must forward {required}: PYTHONPATH makes "
            "megatron.training importable in actors; the others keep NCCL/proxy sane."
        )


def test_r6_uses_sft_loss_with_routing_replay(r6_text):
    assert "--loss-type sft_loss" in r6_text
    assert "--calculate-per-token-loss" in r6_text
    assert "--use-rollout-routing-replay" in r6_text
    assert _default(r6_text, "USE_ROLLOUT_ROUTING_REPLAY") == "1"


def test_r4_rollout_only_sampling_default_positive():
    text = R4.read_text()
    m = re.search(r"--rollout-temperature\s+\"\$\{ROLLOUT_TEMPERATURE:-([^}]*)\}\"", text)
    assert m and float(m.group(1)) > 0, (
        "R4 should default rollout temperature > 0 (temp 0 is only legal because "
        "R4 is --debug-rollout-only; defaulting positive keeps configs uniform)."
    )


def test_smoke_prompt_data_satisfies_batch_sizes(r6_text):
    rollout_batch_size = int(_default(r6_text, "ROLLOUT_BATCH_SIZE"))
    full_loop = REPO / "Data" / "dsv4_full_loop_smoke.jsonl"
    rollout_only = REPO / "Data" / "dsv4_rollout_smoke.jsonl"
    for path, need in [(full_loop, rollout_batch_size), (rollout_only, 2)]:
        assert path.is_file(), f"{path} missing: the smoke launchers preflight it"
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        assert len(lines) >= need, f"{path.name} has {len(lines)} rows < required {need}"
        for ln in lines:
            row = json.loads(ln)
            assert (
                "input" in row and "label" in row
            ), f"{path.name} rows must have the launchers' --input-key/--label-key fields"
