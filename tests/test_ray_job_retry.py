import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

NUM_GPUS = 0
REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "examples/kernel_agent/eval"
HELPER = EVAL_DIR / "ray_job_retry.sh"


def _run_submit(tmp_path, outcomes, *, attempts="3", interval="0"):
    ray = tmp_path / "ray"
    ray.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "calls = Path(os.environ['CALLS'])\n"
        "previous = calls.read_text().splitlines() if calls.exists() else []\n"
        "with calls.open('a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        "outcome = json.loads(os.environ['OUTCOMES'])[len(previous)]\n"
        "print(outcome.get('stdout', ''), flush=True)\n"
        "print(outcome.get('stderr', ''), file=sys.stderr, flush=True)\n"
        "sys.exit(outcome.get('status', 0))\n"
    )
    ray.chmod(0o755)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    calls = tmp_path / "calls.jsonl"
    arguments = [
        "--address=http://localhost:8265",
        '--runtime-env-json={"env_vars": {"A": "two words"}}',
        "--",
        "echo",
        "a b",
    ]
    result = subprocess.run(
        [
            "bash",
            "-c",
            'set -Eeuo pipefail; source "$1"; shift; ray_job_submit_with_retry "$@"',
            "bash",
            str(HELPER),
            *arguments,
        ],
        env={
            **os.environ,
            "PATH": f"{tmp_path}:{os.environ['PATH']}",
            "TMPDIR": str(scratch),
            "CALLS": str(calls),
            "OUTCOMES": json.dumps(outcomes),
            "RAY_SUBMIT_MAX_ATTEMPTS": attempts,
            "RAY_SUBMIT_RETRY_INTERVAL": interval,
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    recorded = [json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
    assert all(call == ["job", "submit", *arguments] for call in recorded)
    assert list(scratch.iterdir()) == []
    return result, len(recorded)


@pytest.mark.parametrize(
    "message",
    ["No available agent to submit job", "Connection refused", "Failed to establish a new connection"],
)
def test_startup_race_retries_then_succeeds(tmp_path, message):
    result, calls = _run_submit(
        tmp_path,
        [{"status": 1, "stderr": message}, {"stdout": "evaluation finished", "stderr": "success warning"}],
    )
    assert result.returncode == 0
    assert calls == 2
    assert message in result.stderr
    assert "success warning" in result.stderr
    assert "evaluation finished" in result.stdout


def test_success_does_not_retry(tmp_path):
    result, calls = _run_submit(tmp_path, [{"stdout": "done"}])
    assert result.returncode == 0
    assert calls == 1


def test_retry_exhaustion_preserves_last_exit_status(tmp_path):
    result, calls = _run_submit(tmp_path, [{"status": status, "stderr": "Connection refused"} for status in (1, 3, 7)])
    assert result.returncode == 7
    assert calls == 3


@pytest.mark.parametrize(
    "outcome",
    [
        {"status": 8, "stderr": "evaluation failed"},
        {"status": 8, "stderr": "invalid Ray arguments"},
        {"status": 8, "stdout": "evaluation: Connection refused"},
        {"status": 8, "stdout": "Job 'test' submitted successfully", "stderr": "Connection refused"},
        {"status": 8, "stderr": "Job 'test' submitted successfully\nConnection refused"},
    ],
)
def test_job_failures_and_post_submission_disconnects_do_not_retry(tmp_path, outcome):
    result, calls = _run_submit(tmp_path, [outcome])
    assert result.returncode == 8
    assert calls == 1


@pytest.mark.parametrize("attempts,interval", [("0", "0"), ("abc", "0"), ("-1", "0"), ("1", "-1"), ("1", "oops")])
def test_invalid_retry_settings_fail_before_submission(tmp_path, attempts, interval):
    result, calls = _run_submit(tmp_path, [], attempts=attempts, interval=interval)
    assert result.returncode == 2
    assert calls == 0


@pytest.mark.parametrize("script", sorted(EVAL_DIR.glob("*.sh")), ids=lambda path: path.name)
def test_eval_shell_syntax_and_retry_source(script):
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text()
    assert "ray job submit --address=" not in text
    if "ray_job_submit_with_retry --address=" in text:
        # Resolve each launcher's SCRIPT_DIR convention without running evaluation.
        setup = next(line for line in text.splitlines() if line.startswith("SCRIPT_DIR="))
        setup = setup.replace("${BASH_SOURCE[0]}", "$0")
        source = next(
            line
            for line in text.splitlines()
            if line.startswith('source "${SCRIPT_DIR}/') and "ray_job_retry.sh" in line
        )
        subprocess.run(
            ["bash", "-c", setup + "\n" + source + "\ndeclare -F ray_job_submit_with_retry", str(script)],
            check=True,
            capture_output=True,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
