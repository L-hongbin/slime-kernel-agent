from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.archive_rolling_node_local_checkpoint import (
    HostSpec,
    RollingArchiver,
    SshTransport,
    file_manifest,
    parse_host,
    select_finalized_iterations,
    validate_checkpoint_dir,
)

NUM_GPUS = 0


class LocalTransport:
    def __init__(self, roots: dict[str, Path]) -> None:
        self.roots = roots
        self.removed: list[tuple[str, int]] = []

    def read_tracker(self, host: HostSpec, checkpoint_dir: str) -> int | None:
        path = self.roots[host.label] / "latest_checkpointed_iteration.txt"
        return int(path.read_text()) if path.is_file() else None

    def list_iterations(self, host: HostSpec, checkpoint_dir: str) -> set[int]:
        return {
            int(path.name.removeprefix("iter_")) for path in self.roots[host.label].glob("iter_*") if path.is_dir()
        }

    def remote_manifest(self, host: HostSpec, checkpoint_dir: str, iteration: int):
        return file_manifest(self.roots[host.label] / f"iter_{iteration:07d}")

    def pull_iteration(self, host: HostSpec, checkpoint_dir: str, iteration: int, destination: Path) -> None:
        shutil.copytree(self.roots[host.label] / f"iter_{iteration:07d}", destination)

    def pull_state(self, host: HostSpec, checkpoint_dir: str, destination: Path) -> None:
        destination.mkdir(parents=True)
        tracker = self.roots[host.label] / "latest_checkpointed_iteration.txt"
        if tracker.is_file():
            shutil.copy2(tracker, destination / tracker.name)

    def remove_iteration(self, host: HostSpec, checkpoint_dir: str, iteration: int) -> None:
        shutil.rmtree(self.roots[host.label] / f"iter_{iteration:07d}")
        self.removed.append((host.label, iteration))


def make_checkpoint(root: Path, iteration: int, payload: bytes) -> None:
    directory = root / f"iter_{iteration:07d}"
    directory.mkdir(parents=True)
    (directory / "rank.distcp").write_bytes(payload)
    (root / "latest_checkpointed_iteration.txt").write_text(str(iteration), encoding="utf-8")


def test_path_and_host_validation():
    assert parse_host("node70=root@10.11.2.170") == HostSpec("node70", "root@10.11.2.170")
    assert parse_host("node70=local") == HostSpec("node70", "local")
    assert validate_checkpoint_dir("/nfs/FM/x/experiments/run/checkpoints").endswith("/checkpoints")
    for value in ["node70", "bad label=node70", "node70=$(oops)"]:
        with pytest.raises(argparse.ArgumentTypeError):
            parse_host(value)
    for path in ["relative/checkpoints", "/nfs/FM/checkpoints", "/nfs/FM/x/experiments/../checkpoints"]:
        with pytest.raises(ValueError):
            validate_checkpoint_dir(path)
    with pytest.raises(ValueError, match="must not be inside"):
        RollingArchiver(
            hosts=[HostSpec("node70", "node70"), HostSpec("node69", "node69")],
            checkpoint_dir="/nfs/FM/x/experiments/qwen38/checkpoints",
            archive_dir=Path("/nfs/FM/x/experiments/qwen38/checkpoints/archive"),
            transport=LocalTransport({}),
            keep_archives=1,
            min_free_bytes=0,
        )


def test_ssh_transport_local_target_uses_direct_filesystem(tmp_path: Path):
    checkpoint_dir = tmp_path / "experiments/qwen38/checkpoints"
    checkpoint_dir.mkdir(parents=True)
    make_checkpoint(checkpoint_dir, 19, b"local-node70-shard")
    host = HostSpec("node70", "local")
    transport = SshTransport(ssh_port=23538)

    assert transport.read_tracker(host, str(checkpoint_dir)) == 19
    assert transport.list_iterations(host, str(checkpoint_dir)) == {19}
    assert transport.remote_manifest(host, str(checkpoint_dir), 19) == file_manifest(checkpoint_dir / "iter_0000019")

    pulled = tmp_path / "pulled/iter_0000019"
    transport.pull_iteration(host, str(checkpoint_dir), 19, pulled)
    assert (pulled / "rank.distcp").read_bytes() == b"local-node70-shard"

    state = tmp_path / "state"
    transport.pull_state(host, str(checkpoint_dir), state)
    assert (state / "latest_checkpointed_iteration.txt").read_text() == "19"

    transport.remove_iteration(host, str(checkpoint_dir), 19)
    assert not (checkpoint_dir / "iter_0000019").exists()


def test_selects_only_iterations_present_on_every_host_and_finalized():
    assert select_finalized_iterations(
        {"node70": None, "node69": 39},
        {"node70": {19, 39, 59}, "node69": {19, 39}},
    ) == [19, 39]
    assert (
        select_finalized_iterations(
            {"node70": None, "node69": None},
            {"node70": {19}, "node69": {19}},
        )
        == []
    )


def test_archives_hashes_then_prunes_exact_node_local_iteration(tmp_path: Path):
    roots = {"node70": tmp_path / "remote70", "node69": tmp_path / "remote69"}
    for root in roots.values():
        root.mkdir()
    make_checkpoint(roots["node70"], 19, b"node70-full-optimizer-shard")
    make_checkpoint(roots["node69"], 19, b"node69-full-optimizer-shard")
    hosts = [HostSpec("node70", "node70"), HostSpec("node69", "node69")]
    transport = LocalTransport(roots)
    archive = tmp_path / "archive"
    archiver = RollingArchiver(
        hosts=hosts,
        checkpoint_dir="/nfs/FM/x/experiments/qwen38/checkpoints",
        archive_dir=archive,
        transport=transport,
        keep_archives=1,
        min_free_bytes=0,
    )

    assert archiver.poll_once() == 1
    final = archive / "iter_0000019"
    manifest = json.loads((final / "manifest.json").read_text())
    assert manifest["verified"] is True
    assert manifest["total_bytes"] == len(b"node70-full-optimizer-shard") + len(b"node69-full-optimizer-shard")
    assert (final / "node70/iter_0000019/rank.distcp").read_bytes() == b"node70-full-optimizer-shard"
    assert (final / "node69/iter_0000019/rank.distcp").read_bytes() == b"node69-full-optimizer-shard"
    assert set(transport.removed) == {("node70", 19), ("node69", 19)}
    assert not (roots["node70"] / "iter_0000019").exists()
    assert not (roots["node69"] / "iter_0000019").exists()


def test_source_is_not_pruned_when_archive_verification_fails(tmp_path: Path, monkeypatch):
    roots = {"node70": tmp_path / "remote70", "node69": tmp_path / "remote69"}
    for root in roots.values():
        root.mkdir()
    make_checkpoint(roots["node70"], 19, b"node70")
    make_checkpoint(roots["node69"], 19, b"node69")
    transport = LocalTransport(roots)
    original_pull = transport.pull_iteration

    def corrupt_pull(host, checkpoint_dir, iteration, destination):
        original_pull(host, checkpoint_dir, iteration, destination)
        if host.label == "node69":
            (destination / "rank.distcp").write_bytes(b"corrupt")

    monkeypatch.setattr(transport, "pull_iteration", corrupt_pull)
    archiver = RollingArchiver(
        hosts=[HostSpec("node70", "node70"), HostSpec("node69", "node69")],
        checkpoint_dir="/nfs/FM/x/experiments/qwen38/checkpoints",
        archive_dir=tmp_path / "archive",
        transport=transport,
        keep_archives=1,
        min_free_bytes=0,
    )

    with pytest.raises(RuntimeError, match="SHA256 verification failed"):
        archiver.poll_once()
    assert transport.removed == []
    assert (roots["node70"] / "iter_0000019").is_dir()
    assert (roots["node69"] / "iter_0000019").is_dir()


def test_retain_source_mode_never_prunes_verified_node_local_iterations(tmp_path: Path):
    roots = {"node70": tmp_path / "remote70", "node69": tmp_path / "remote69"}
    for root in roots.values():
        root.mkdir()
    make_checkpoint(roots["node70"], 19, b"node70")
    make_checkpoint(roots["node69"], 19, b"node69")
    transport = LocalTransport(roots)
    archive = tmp_path / "archive"
    archiver = RollingArchiver(
        hosts=[HostSpec("node70", "node70"), HostSpec("node69", "node69")],
        checkpoint_dir="/nfs/FM/x/experiments/qwen38/checkpoints",
        archive_dir=archive,
        transport=transport,
        keep_archives=1,
        min_free_bytes=0,
        retain_source=True,
    )

    assert archiver.poll_once() == 1
    manifest = json.loads((archive / "iter_0000019/manifest.json").read_text())
    assert manifest["source_retained"] is True
    assert transport.removed == []
    assert (roots["node70"] / "iter_0000019").is_dir()
    assert (roots["node69"] / "iter_0000019").is_dir()

    # Re-observing an already verified archive must also retain the sources.
    assert archiver.poll_once() == 1
    assert transport.removed == []


def test_keeps_only_latest_verified_archive(tmp_path: Path):
    roots = {"node70": tmp_path / "remote70", "node69": tmp_path / "remote69"}
    for root in roots.values():
        root.mkdir()
    hosts = [HostSpec("node70", "node70"), HostSpec("node69", "node69")]
    transport = LocalTransport(roots)
    archive = tmp_path / "archive"
    archiver = RollingArchiver(
        hosts=hosts,
        checkpoint_dir="/nfs/FM/x/experiments/qwen38/checkpoints",
        archive_dir=archive,
        transport=transport,
        keep_archives=1,
        min_free_bytes=0,
    )
    for iteration in (19, 39):
        make_checkpoint(roots["node70"], iteration, f"node70-{iteration}".encode())
        make_checkpoint(roots["node69"], iteration, f"node69-{iteration}".encode())
        assert archiver.poll_once() == 1

    assert not (archive / "iter_0000019").exists()
    assert (archive / "iter_0000039/manifest.json").is_file()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
