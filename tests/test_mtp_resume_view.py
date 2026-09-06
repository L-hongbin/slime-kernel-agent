import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed.checkpoint as dcp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_mtp_resume_view import EMBEDDING_KEY, MANIFEST, fetch_range, prepare_view

NUM_GPUS = 0


@pytest.fixture
def checkpoint(tmp_path):
    owner = tmp_path / "owner"
    target = tmp_path / "target"
    owner_iter = owner / "iter_0000000"
    target_iter = target / "iter_0000000"
    embedding = torch.arange(32, dtype=torch.bfloat16).view(4, 8)
    dcp.save({EMBEDDING_KEY: embedding, "decoder.weight": torch.ones(4096)}, checkpoint_id=owner_iter)
    target_iter.mkdir(parents=True)
    shutil.copy2(owner_iter / ".metadata", target_iter / ".metadata")
    (target / "rollout").mkdir()
    (target / "rollout" / "state.txt").write_text("sampler state")
    (target / "latest_checkpointed_iteration.txt").write_text("0")

    def fetch(source_dir, name, offset, length, metadata_sha, output):
        with (owner_iter / name).open("rb") as stream:
            stream.seek(offset)
            output.write_bytes(stream.read(length))
        return "owner"

    return owner, target, embedding, fetch


def test_remote_embedding_ranges_load_without_copying_the_owner_shard(checkpoint, tmp_path):
    owner, target, embedding, fetch = checkpoint
    original_metadata = (target / "iter_0000000" / ".metadata").read_bytes()
    view = tmp_path / "load_view"
    result = prepare_view(target, view, 0, fetch)
    assert result["materialized_bytes"] < sum(f.stat().st_size for f in (owner / "iter_0000000").glob("*.distcp"))
    assert (target / "iter_0000000" / ".metadata").read_bytes() == original_metadata
    assert not list((target / "iter_0000000").glob("*.distcp"))
    assert (view / "rollout" / "state.txt").read_text() == "sampler state"
    restored = {EMBEDDING_KEY: torch.empty_like(embedding)}
    dcp.load(restored, checkpoint_id=view / "iter_0000000")
    torch.testing.assert_close(restored[EMBEDDING_KEY], embedding, rtol=0, atol=0)


def test_owner_view_uses_local_storage_links(checkpoint, tmp_path):
    owner, _, embedding, _ = checkpoint

    def unexpected_fetch(*args):
        raise AssertionError("Local embedding must not use SSH")

    view = tmp_path / "owner_view"
    result = prepare_view(owner, view, 0, unexpected_fetch)
    assert result["materialized_bytes"] == 0
    assert all(f.is_symlink() for f in (view / "iter_0000000").glob("*.distcp"))
    restored = {EMBEDDING_KEY: torch.empty_like(embedding)}
    dcp.load(restored, checkpoint_id=view / "iter_0000000")
    torch.testing.assert_close(restored[EMBEDDING_KEY], embedding, rtol=0, atol=0)


def test_failed_rebuild_preserves_previous_view_and_canonical_checkpoint(checkpoint, tmp_path):
    owner, target, _, fetch = checkpoint
    view = tmp_path / "view"
    prepare_view(target, view, 0, fetch)
    original_manifest = (view / MANIFEST).read_bytes()
    original_metadata = (target / "iter_0000000" / ".metadata").read_bytes()

    def partial_fetch(source_dir, name, offset, length, metadata_sha, output):
        output.write_bytes(b"partial")
        return "owner"

    with pytest.raises(RuntimeError, match="Incomplete embedding"):
        prepare_view(target, view, 0, partial_fetch)
    assert (view / MANIFEST).read_bytes() == original_manifest
    assert (target / "iter_0000000" / ".metadata").read_bytes() == original_metadata
    assert list((owner / "iter_0000000").glob("*.distcp"))


def test_rebuild_does_not_reuse_stale_embedding_data(checkpoint, tmp_path):
    owner, target, embedding, fetch = checkpoint
    view = tmp_path / "view"
    prepare_view(target, view, 0, fetch)
    updated = embedding + 16
    dcp.save({EMBEDDING_KEY: updated, "decoder.weight": torch.ones(4096)}, checkpoint_id=owner / "iter_0000000")
    shutil.copy2(owner / "iter_0000000" / ".metadata", target / "iter_0000000" / ".metadata")
    prepare_view(target, view, 0, fetch)
    restored = {EMBEDDING_KEY: torch.empty_like(embedding)}
    dcp.load(restored, checkpoint_id=view / "iter_0000000")
    torch.testing.assert_close(restored[EMBEDDING_KEY], updated, rtol=0, atol=0)


def test_refuses_unrecognized_or_in_checkpoint_destination(checkpoint, tmp_path):
    _, target, _, fetch = checkpoint
    view = tmp_path / "unrelated"
    view.mkdir()
    (view / "keep.txt").write_text("keep")
    with pytest.raises(ValueError, match="unrecognized"):
        prepare_view(target, view, 0, fetch)
    assert (view / "keep.txt").read_text() == "keep"
    with pytest.raises(ValueError, match="outside"):
        prepare_view(target, target / "view", 0, fetch)


@pytest.mark.parametrize("failure", [subprocess.TimeoutExpired("ssh", 600), OSError("transport unavailable")])
def test_source_fallback_discards_partial_transfer_after_timeout(monkeypatch, tmp_path, failure):
    attempts = []
    payload = b"serialized embedding tensor"

    def run(command, *, stdout, stderr, timeout):
        attempts.append(command)
        if len(attempts) == 1:
            stdout.write(b"partial previous source")
            raise failure
        stdout.write(payload)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    output = tmp_path / "replica"
    owner = fetch_range([("head", 23538), ("owner", 23538)], tmp_path, "__0.distcp", 0, len(payload), "sha", output)
    assert owner == "owner" and len(attempts) == 2
    assert output.read_bytes() == payload


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
