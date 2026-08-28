"""CPU coverage for KernelGym startup retry backoff."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts import check_kernelgym_health as health


NUM_GPUS = 0
RESOURCE_CHECK = REPO_ROOT / "scripts/check_host_resources.sh"


@pytest.mark.unit
def test_health_retry_uses_exponential_backoff_with_cap(monkeypatch):
    sleeps = []

    class FakeClient:
        def __init__(self, base_url, timeout_s):
            self.base_url = base_url
            self.calls = 0

        async def check_health(self):
            self.calls += 1
            if self.calls < 5:
                raise health.KernelGymRequestError("restarting")
            return {"status": "healthy"}

        async def close(self):
            return None

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(health, "KernelGymHealthClient", FakeClient)
    monkeypatch.setattr(health.asyncio, "sleep", fake_sleep)
    args = SimpleNamespace(
        url="http://kernelgym",
        timeout=1.0,
        attempts=5,
        interval=2.0,
        backoff_factor=2.0,
        max_interval=10.0,
        workers_status=False,
    )

    assert health.asyncio.run(health.run(args)) == 0
    assert sleeps == [2.0, 4.0, 8.0, 10.0]


@pytest.mark.unit
def test_health_backoff_rejects_factor_below_one():
    args = SimpleNamespace(
        url="http://kernelgym",
        timeout=1.0,
        attempts=2,
        interval=1.0,
        backoff_factor=0.5,
        max_interval=10.0,
        workers_status=False,
    )

    with pytest.raises(ValueError, match="backoff-factor"):
        health.asyncio.run(health.run(args))


@pytest.mark.unit
def test_zero_attempt_limit_retries_until_healthy(monkeypatch):
    sleeps = []

    class FakeClient:
        def __init__(self, base_url, timeout_s):
            self.base_url = base_url
            self.calls = 0

        async def check_health(self):
            self.calls += 1
            if self.calls < 4:
                raise health.KernelGymRequestError("still restarting")
            return {"status": "healthy"}

        async def close(self):
            return None

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(health, "KernelGymHealthClient", FakeClient)
    monkeypatch.setattr(health.asyncio, "sleep", fake_sleep)
    args = SimpleNamespace(
        url="http://kernelgym",
        timeout=1.0,
        attempts=0,
        interval=2.0,
        backoff_factor=2.0,
        max_interval=3600.0,
        workers_status=False,
    )

    assert health.asyncio.run(health.run(args)) == 0
    assert sleeps == [2.0, 4.0, 8.0]


def _write_fake_nvidia_smi(path: Path, *, busy: bool) -> None:
    compute_apps = 'echo "GPU-test, 123, busy, 100"' if busy else "exit 0"
    path.write_text(
        f"""#!/bin/bash
case "$*" in
  "-L") echo "GPU 0: NVIDIA H20 (UUID: GPU-test)" ;;
  *"--query-gpu="*) echo "0, NVIDIA H20, 0, 97871, 0" ;;
  *"--query-compute-apps="*) {compute_apps} ;;
  *) exit 2 ;;
esac
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_host_preflight_forwards_kernelgym_retry_options(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_nvidia_smi(bin_dir / "nvidia-smi", busy=False)
    captured = tmp_path / "health-args"
    health_script = tmp_path / "health.py"
    health_script.write_text(
        f"from pathlib import Path\nimport sys\nPath({str(captured)!r}).write_text('\\n'.join(sys.argv[1:]))\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            str(RESOURCE_CHECK),
            "--expected-gpus",
            "1",
            "--skip-cpu-health",
            "--kernelgym-health-script",
            str(health_script),
            "--kernelgym-health-attempts",
            "0",
            "--kernelgym-health-interval",
            "2",
            "--kernelgym-health-backoff",
            "2",
            "--kernelgym-health-max-interval",
            "3600",
            "--python-bin",
            sys.executable,
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert captured.read_text().splitlines() == [
        "--url",
        "http://127.0.0.1:20211",
        "--attempts",
        "0",
        "--interval",
        "2",
        "--backoff-factor",
        "2",
        "--max-interval",
        "3600",
    ]


def test_host_preflight_rejects_busy_gpu_before_waiting_for_kernelgym(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _write_fake_nvidia_smi(bin_dir / "nvidia-smi", busy=True)
    marker = tmp_path / "kernelgym-was-called"
    health_script = tmp_path / "health.py"
    health_script.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "bash",
            str(RESOURCE_CHECK),
            "--expected-gpus",
            "1",
            "--skip-cpu-health",
            "--kernelgym-health-script",
            str(health_script),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "GPU compute processes are already running" in result.stderr
    assert not marker.exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
