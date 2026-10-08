"""CPU coverage for KernelGym startup retry backoff."""

from __future__ import annotations

import os
import subprocess
import sys
from copy import deepcopy
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


@pytest.mark.unit
@pytest.mark.parametrize(
    "case,expected,message",
    [
        ("matching", 0, "kernelgym_version matches"),
        ("different_gpu", 1, "GPU mismatch"),
        ("different_memory", 1, "GPU mismatch"),
        ("missing_gpu", 1, "missing GPU information"),
        ("unavailable", 1, "GPU unavailable"),
        ("mixed_gpu", 1, "GPU mismatch"),
        ("different_version", 1, "kernelgym_version mismatch"),
        ("different_dirty", 1, "kernelgym_version mismatch"),
        ("one_missing_version", 0, "SKIP:"),
        ("all_missing_version", 0, "SKIP:"),
        ("null_version", 0, "SKIP:"),
        ("missing_with_other_mismatch", 0, "SKIP:"),
        ("unhealthy", 1, "not healthy"),
    ],
)
def test_multiple_health_urls_over_local_http(case, expected, message, capsys):
    from aiohttp import web

    # Same schema as a real local KernelGym /health response.
    first = {
        "status": "healthy",
        "gpu_status": {
            "cuda:0": {"name": "NVIDIA H100 PCIe", "memory_total": "79.6GB", "memory_used": "1.1GB", "available": True}
        },
        "kernelgym_version": {"git_commit": "abc", "git_branch": "main", "git_dirty": False},
    }
    second = deepcopy(first)
    gpu = second["gpu_status"].pop("cuda:0")
    gpu["memory_used"] = "23.0GB"
    gpu["memory_total"] = "79.60 GB"
    second["gpu_status"] = {"cuda:3": gpu, "cuda:7": deepcopy(gpu)}
    payloads = [first, second]
    if case == "different_gpu":
        gpu["name"] = "NVIDIA A100"
    elif case == "different_memory":
        gpu["memory_total"] = "40GB"
    elif case == "missing_gpu":
        second.pop("gpu_status")
    elif case == "unavailable":
        gpu["available"] = False
    elif case == "mixed_gpu":
        second["gpu_status"]["cuda:7"]["name"] = "NVIDIA A100"
    elif case == "different_version":
        second["kernelgym_version"]["git_commit"] = "def"
    elif case == "different_dirty":
        second["kernelgym_version"]["git_dirty"] = True
    elif case == "one_missing_version":
        second.pop("kernelgym_version")
    elif case == "all_missing_version":
        for payload in payloads:
            payload.pop("kernelgym_version")
    elif case == "null_version":
        second["kernelgym_version"] = {"git_commit": None, "git_branch": None, "git_dirty": None}
    elif case == "missing_with_other_mismatch":
        second["kernelgym_version"]["git_commit"] = "def"
        third = deepcopy(first)
        third.pop("kernelgym_version")
        payloads.append(third)
    elif case == "unhealthy":
        second["status"] = "unhealthy"

    async def scenario():
        runners, urls, requested = [], [], []
        try:
            for index, payload in enumerate(payloads):

                async def handler(request, payload=payload, index=index):
                    requested.append((index, request.path))
                    if request.path == "/health":
                        return web.json_response(payload)
                    return web.json_response({"workers": {}})

                app = web.Application()
                app.router.add_get("/health", handler)
                app.router.add_get("/workers/status", handler)
                runner = web.AppRunner(app)
                await runner.setup()
                runners.append(runner)
                site = web.TCPSite(runner, "127.0.0.1", 0)
                await site.start()
                port = runner.addresses[0][1]
                urls.append(f"http://127.0.0.1:{port}")
            args = health.parse_args(["--urls", *urls, "--attempts", "1", "--timeout", "2", "--workers-status"])
            assert await health.run(args) == expected
            assert all((i, "/health") in requested for i in range(len(payloads)))
            if expected == 0:
                assert all((i, "/workers/status") in requested for i in range(len(payloads)))
        finally:
            for runner in runners:
                await runner.cleanup()

    health.asyncio.run(scenario())
    captured = capsys.readouterr()
    assert message in captured.out + captured.err


@pytest.mark.unit
def test_health_url_cli_and_environment_precedence(monkeypatch):
    monkeypatch.setenv("KERNELGYM_URLS", "http://a,http://b")
    assert health.parse_args([]).urls == ["http://a", "http://b"]
    single = health.parse_args(["--url", "http://single"])
    assert single.urls is None and single.url == "http://single"
    assert health.parse_args(["--urls", "http://explicit"]).urls == ["http://explicit"]
    with pytest.raises(SystemExit):
        health.parse_args(["--url", "http://a", "--urls", "http://b"])


@pytest.mark.unit
def test_single_url_keeps_legacy_health_contract(monkeypatch):
    class FakeClient:
        def __init__(self, base_url, timeout_s):
            self.base_url = base_url

        async def check_health(self):
            return {"status": "healthy"}

        async def close(self):
            pass

    monkeypatch.setattr(health, "KernelGymHealthClient", FakeClient)
    args = health.parse_args(["--url", "http://legacy", "--attempts", "1"])
    assert health.asyncio.run(health.run(args)) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
