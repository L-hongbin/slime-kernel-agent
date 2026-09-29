import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.b300 import check_qwen38_b300_runtime as runtime  # noqa: E402

NUM_GPUS = 0
pytestmark = pytest.mark.unit


@pytest.mark.parametrize("rollout_bytes", [b"same NCCL runtime", b"different NCCL runtime", None])
def test_nccl_runtime_alignment_checks_library_contents(tmp_path, monkeypatch, rollout_bytes):
    train = tmp_path / "train"
    rollout = tmp_path / "rollout"
    (train / "torch").mkdir(parents=True)
    (rollout / "torch").mkdir(parents=True)
    relative = Path("nvidia/nccl/lib/libnccl.so.2")
    (train / relative).parent.mkdir(parents=True)
    (train / relative).write_bytes(b"same NCCL runtime")
    if rollout_bytes is not None:
        (rollout / relative).parent.mkdir(parents=True)
        (rollout / relative).write_bytes(rollout_bytes)
    monkeypatch.setenv("SLIME_TRAIN_PACKAGES", str(train))
    monkeypatch.setenv("SLIME_ROLLOUT_PYTHONPATH", f"{tmp_path / 'without_torch'}:{rollout}")
    if rollout_bytes is None:
        with pytest.raises(FileNotFoundError):
            runtime._check_nccl_runtime_alignment()
    elif rollout_bytes != b"same NCCL runtime":
        with pytest.raises(RuntimeError, match="must be byte-identical"):
            runtime._check_nccl_runtime_alignment()
    else:
        runtime._check_nccl_runtime_alignment()


def test_nccl_runtime_alignment_requires_rollout_torch(tmp_path, monkeypatch):
    monkeypatch.setenv("SLIME_TRAIN_PACKAGES", str(tmp_path / "train"))
    monkeypatch.setenv("SLIME_ROLLOUT_PYTHONPATH", str(tmp_path / "without_torch"))
    with pytest.raises(RuntimeError, match="does not contain a Torch runtime"):
        runtime._check_nccl_runtime_alignment()


def test_package_version_prefers_rollout_overlay(tmp_path, monkeypatch):
    metadata = tmp_path / "sglang-0.5.20.dist-info" / "METADATA"
    metadata.parent.mkdir()
    metadata.write_text("Metadata-Version: 2.1\nName: sglang\nVersion: 0.5.20\n")
    monkeypatch.setenv("SLIME_ROLLOUT_PYTHONPATH", str(tmp_path))
    monkeypatch.setattr(runtime.importlib.metadata, "version", lambda name: "0.5.15.post1")
    assert runtime._check_sglang_version() == "0.5.20"
    monkeypatch.setenv("SLIME_ROLLOUT_PYTHONPATH", "")
    assert runtime._check_sglang_version() == "0.5.15.post1"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
