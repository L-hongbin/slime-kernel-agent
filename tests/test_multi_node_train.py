from pathlib import Path
import sys

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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
