from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

NUM_GPUS = 0

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import multi_node_train


@pytest.mark.unit
def test_scp_opts_convert_ssh_port_option():
    ssh_opts = ["-p", "23422", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]

    assert multi_node_train.scp_opts_from_ssh_opts(ssh_opts) == [
        "-P",
        "23422",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
    ]


@pytest.mark.unit
def test_scp_opts_convert_compact_ssh_port_option():
    assert multi_node_train.scp_opts_from_ssh_opts(["-p23422"]) == ["-P23422"]


@pytest.mark.unit
def test_scp_opts_env_override(monkeypatch):
    monkeypatch.setenv("MULTI_NODE_SCP_OPTS", "-P 12345 -o BatchMode=yes")

    assert multi_node_train.scp_opts_from_env(["-p", "23422"]) == ["-P", "12345", "-o", "BatchMode=yes"]


@pytest.mark.unit
def test_env_flag_enabled(monkeypatch):
    monkeypatch.delenv("MULTI_NODE_KERNELGYM_HEALTH_CHECK", raising=False)
    assert multi_node_train.env_flag_enabled("MULTI_NODE_KERNELGYM_HEALTH_CHECK")

    monkeypatch.setenv("MULTI_NODE_KERNELGYM_HEALTH_CHECK", "0")
    assert not multi_node_train.env_flag_enabled("MULTI_NODE_KERNELGYM_HEALTH_CHECK")


@pytest.mark.unit
def test_kernelgym_health_command_uses_env(monkeypatch):
    monkeypatch.setenv("KERNELGYM_URL", "http://kernelgym:20111")
    monkeypatch.setenv("KERNELGYM_HEALTH_TIMEOUT", "7")
    monkeypatch.setenv("KERNELGYM_HEALTH_ATTEMPTS", "3")
    monkeypatch.setenv("KERNELGYM_HEALTH_INTERVAL", "2")

    command = multi_node_train.kernelgym_health_command(REPO_ROOT)

    assert command == [
        "python3",
        str(REPO_ROOT / "scripts/check_kernelgym_health.py"),
        "--url",
        "http://kernelgym:20111",
        "--timeout",
        "7",
        "--attempts",
        "3",
        "--interval",
        "2",
    ]


@pytest.mark.unit
def test_parse_gpu_occupancy_ignores_blank_lines():
    stdout = "\nGPU-abc, 1234, python, 4096\n\nGPU-def, 2345, ray::Worker, 512\n"

    processes = multi_node_train.parse_gpu_occupancy(stdout)

    assert processes == [
        multi_node_train.GpuProcess("GPU-abc", "1234", "python", "4096"),
        multi_node_train.GpuProcess("GPU-def", "2345", "ray::Worker", "512"),
    ]


@pytest.mark.unit
def test_run_gpu_occupancy_probe_raises_when_process_exists(monkeypatch):
    def fake_run(command, text, stdout, stderr):
        return SimpleNamespace(returncode=0, stdout="GPU-abc, 1234, python, 4096\n", stderr="")

    monkeypatch.setattr(multi_node_train.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="GPU occupied on node 0"):
        multi_node_train.run_gpu_occupancy_probe("node 0", multi_node_train.gpu_occupancy_command())


@pytest.mark.unit
def test_run_gpu_occupancy_probe_raises_when_probe_fails(monkeypatch):
    def fake_run(command, text, stdout, stderr):
        return SimpleNamespace(returncode=1, stdout="", stderr="nvidia-smi: command not found")

    monkeypatch.setattr(multi_node_train.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="GPU occupancy check failed for node 0: nvidia-smi"):
        multi_node_train.run_gpu_occupancy_probe("node 0", multi_node_train.gpu_occupancy_command())


@pytest.mark.unit
def test_check_gpu_free_for_hosts_checks_local_and_remote(monkeypatch):
    calls = []

    def fake_run(command, text, stdout, stderr):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setenv("MULTI_NODE_SSH_OPTS", "-o BatchMode=yes")
    monkeypatch.setattr(multi_node_train.subprocess, "run", fake_run)

    hosts = [
        multi_node_train.Host("head", "10.0.0.1"),
        multi_node_train.Host("worker", "10.0.0.2"),
    ]
    multi_node_train.check_gpu_free_for_hosts(hosts, node_idx=0)

    assert calls == [
        multi_node_train.gpu_occupancy_command(),
        ["ssh", "-o", "BatchMode=yes", "worker", multi_node_train.remote_gpu_occupancy_command()],
    ]


@pytest.mark.unit
def test_check_gpu_free_for_hosts_can_be_disabled(monkeypatch):
    def fail_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called")

    monkeypatch.setenv("MULTI_NODE_GPU_OCCUPANCY_CHECK", "0")
    monkeypatch.setattr(multi_node_train.subprocess, "run", fail_run)

    multi_node_train.check_gpu_free_for_hosts([multi_node_train.Host("head", "10.0.0.1")], node_idx=0)


@pytest.mark.unit
def test_check_current_node_gpu_free_can_be_disabled(monkeypatch):
    def fail_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called")

    monkeypatch.setenv("MULTI_NODE_GPU_OCCUPANCY_CHECK", "false")
    monkeypatch.setattr(multi_node_train.subprocess, "run", fail_run)

    multi_node_train.check_current_node_gpu_free("node 1 worker")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
