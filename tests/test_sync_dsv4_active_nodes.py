import hashlib
import os
import shutil
import socket
import subprocess
from pathlib import Path

import pytest


NUM_GPUS = 0

REPO_ROOT = Path(__file__).resolve().parents[1]
SYNC_FILES = (
    "rsync_project.sh",
    "runtime_fingerprint.sh",
    "sync_dsv4_active_nodes.sh",
)


def _write_executable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(0o755)


def _fixture_repo(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    sync_dir = root / "scripts" / "sync"
    sync_dir.mkdir(parents=True)
    for name in SYNC_FILES:
        shutil.copy2(REPO_ROOT / "scripts" / "sync" / name, sync_dir / name)

    for directory in ("slime", "custom_kernels", "tests", "examples"):
        (root / directory).mkdir()
        (root / directory / "keep.txt").write_text(f"{directory}\n")
    (root / "train.py").write_text("# train\n")
    (root / "train_async.py").write_text("# train async\n")
    tile = tmp_path / "TileKernels"
    tile.mkdir()
    (tile / "tile.py").write_text("# tile\n")
    for checkout in (root, tile):
        subprocess.run(["git", "init", "-q", str(checkout)], check=True)
        subprocess.run(["git", "-C", str(checkout), "config", "user.email", "sync-test@example.invalid"], check=True)
        subprocess.run(["git", "-C", str(checkout), "config", "user.name", "Sync Test"], check=True)
        subprocess.run(["git", "-C", str(checkout), "add", "."], check=True)
        subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "fixture"], check=True)
    return root


@pytest.fixture
def fake_transport(tmp_path: Path) -> tuple[Path, Path]:
    fake_bin = tmp_path / "fake-bin"
    call_log = tmp_path / "calls.log"
    _write_executable(
        fake_bin / "ssh",
        r"""#!/usr/bin/env bash
set -euo pipefail
host=$1
shift
if [[ "$*" == *"__SLIME_SYNC_PROBE__"* ]]; then
  printf '__SLIME_SYNC_PROBE__ rsync=%s tar=%s\n' "${FAKE_REMOTE_RSYNC:-0}" "${FAKE_REMOTE_TAR:-1}"
  exit 0
fi
if [[ "$*" == *"runtime_fingerprint.sh --verify-manifest"* ]]; then
  expected=${*##* }
  if [[ "$*" == *"TileKernels"* ]]; then
    exec bash "${FAKE_FINGERPRINT_SCRIPT:?}" --verify-manifest "${FAKE_REMOTE_TILE_ROOT:?}/${host}" "${expected}"
  fi
  exec bash "${FAKE_FINGERPRINT_SCRIPT:?}" --verify-manifest "${FAKE_REMOTE_ROOT:?}/${host}" "${expected}"
fi
if [[ "$*" == *"sha256sum --"*".slime-v4-source-provenance"* ]]; then
  if [[ "$*" == *"TileKernels"* ]]; then
    exec sha256sum -- "${FAKE_REMOTE_TILE_ROOT:?}/${host}/.slime-v4-source-provenance"
  fi
  exec sha256sum -- "${FAKE_REMOTE_ROOT:?}/${host}/.slime-v4-source-provenance"
fi
exec bash -c "$*"
""",
    )
    _write_executable(
        fake_bin / "rsync",
        r"""#!/usr/bin/env bash
set -euo pipefail
destination=${!#}
printf '%s\n' "${destination}" >> "${FAKE_CALL_LOG:?}"
if [[ -n "${FAKE_FAIL_TARGET:-}" && "${destination}" == "${FAKE_FAIL_TARGET}:"* ]]; then
  exit 23
fi
exit 0
""",
    )
    _write_executable(
        fake_bin / "hostname",
        r"""#!/usr/bin/env bash
set -euo pipefail
if [[ -n "${FAKE_SOURCE_HOST:-}" ]]; then
  printf '%s\n' "${FAKE_SOURCE_HOST}"
else
  exec /bin/hostname "$@"
fi
""",
    )
    return fake_bin, call_log


def _env(fake_bin: Path, call_log: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        PATH=f"{fake_bin}:/usr/bin:/bin",
        FAKE_CALL_LOG=str(call_log),
        SYNC_V4_EXPECTED_SOURCE_HOST=socket.gethostname().split(".", 1)[0],
    )
    return env


def _provenance(checkout: Path, fingerprint_helper: Path, *, tree: bool) -> str:
    manifest_flag = "--tree-manifest" if tree else "--manifest"
    manifest = subprocess.check_output([str(fingerprint_helper), manifest_flag, str(checkout)])
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "--verify", "HEAD"], text=True
    ).strip()
    return (
        "format=slime-v4-source-provenance-v1\n"
        f"git_revision={revision}\n"
        f"runtime_source_manifest_sha256={hashlib.sha256(manifest).hexdigest()}\n"
    )


def test_tar_fallback_preserves_project_excludes(tmp_path: Path, fake_transport) -> None:
    fake_bin, call_log = fake_transport
    root = _fixture_repo(tmp_path)
    destination = tmp_path / "remote checkout"

    (root / "included.py").write_text("included\n")
    (root / ".git" / "secret").write_text("no\n")
    (root / "experiments").mkdir()
    (root / "experiments" / "large.bin").write_text("no\n")
    (root / "local_artifacts").mkdir()
    (root / "local_artifacts" / "large.bin").write_text("no\n")
    (root / "slime" / "__pycache__").mkdir()
    (root / "slime" / "__pycache__" / "bad.pyc").write_text("no\n")
    (root / "examples" / "kernel_agent" / "logs").mkdir(parents=True)
    (root / "examples" / "kernel_agent" / "logs" / "run.txt").write_text("no\n")
    (root / "custom_kernels" / "build.log").write_text("no\n")
    (root / "custom_kernels" / "plugin.egg-info").mkdir()
    (root / "custom_kernels" / "plugin.egg-info" / "PKG-INFO").write_text("no\n")

    result = subprocess.run(
        [str(root / "scripts/sync/rsync_project.sh"), "fake-host", str(destination)],
        text=True,
        capture_output=True,
        env={**_env(fake_bin, call_log), "FAKE_REMOTE_RSYNC": "0"},
    )
    assert result.returncode == 0, result.stderr
    assert "backend=tar" in result.stdout
    assert (destination / "included.py").read_text() == "included\n"
    assert not (destination / ".git").exists()
    assert not (destination / "experiments").exists()
    assert not (destination / "local_artifacts").exists()
    assert not (destination / "slime/__pycache__").exists()
    assert not (destination / "examples/kernel_agent/logs").exists()
    assert not (destination / "custom_kernels/build.log").exists()
    assert not (destination / "custom_kernels/plugin.egg-info").exists()


def test_tar_fallback_refuses_delete_semantics(tmp_path: Path, fake_transport) -> None:
    fake_bin, call_log = fake_transport
    root = _fixture_repo(tmp_path)
    destination = tmp_path / "must-not-exist"
    result = subprocess.run(
        [str(root / "scripts/sync/rsync_project.sh"), "fake-host", str(destination)],
        text=True,
        capture_output=True,
        env={
            **_env(fake_bin, call_log),
            "FAKE_REMOTE_RSYNC": "0",
            "RSYNC_PROJECT_DELETE": "1",
        },
    )
    assert result.returncode == 2
    assert "requires rsync at both ends" in result.stderr
    assert not destination.exists()


def test_active_sync_dry_run_is_concurrent_whitelisted_and_fail_closed(tmp_path: Path, fake_transport) -> None:
    fake_bin, call_log = fake_transport
    root = _fixture_repo(tmp_path)
    script = root / "scripts/sync/sync_dsv4_active_nodes.sh"
    env = {
        **_env(fake_bin, call_log),
        "FAKE_REMOTE_RSYNC": "1",
        "SYNC_V4_TILEKERNELS_DIR": str(tmp_path / "TileKernels"),
    }

    rejected = subprocess.run(
        [str(script), "--dry-run", "--target", "node54"],
        text=True,
        capture_output=True,
        env=env,
    )
    assert rejected.returncode == 2
    assert "refusing non-active target: node54" in rejected.stderr
    assert not call_log.exists()

    successful = subprocess.run([str(script), "--dry-run"], text=True, capture_output=True, env=env)
    assert successful.returncode == 0, successful.stderr
    destinations = set(call_log.read_text().splitlines())
    assert {item.split(":", 1)[0] for item in destinations} == {
        "node69_slime",
        "node53_dspark",
        "node70_dspark",
    }
    assert len(call_log.read_text().splitlines()) == 6

    call_log.unlink()
    failed = subprocess.run(
        [str(script), "--dry-run"],
        text=True,
        capture_output=True,
        env={**env, "FAKE_FAIL_TARGET": "node53_dspark"},
    )
    assert failed.returncode == 1
    assert "one or more targets failed" in failed.stderr
    assert {item.split(":", 1)[0] for item in call_log.read_text().splitlines()} == {
        "node69_slime",
        "node53_dspark",
        "node70_dspark",
    }


def test_node64_dspark_dry_run_is_allowed_but_never_self_copies(tmp_path: Path, fake_transport) -> None:
    fake_bin, call_log = fake_transport
    root = _fixture_repo(tmp_path)
    script = root / "scripts/sync/sync_dsv4_active_nodes.sh"
    env = {
        **_env(fake_bin, call_log),
        "FAKE_SOURCE_HOST": "node64",
        "SYNC_V4_EXPECTED_SOURCE_HOST": "node64",
        "SYNC_V4_TILEKERNELS_DIR": str(tmp_path / "TileKernels"),
    }

    result = subprocess.run(
        [str(script), "--dry-run", "--target", "node64_dspark"],
        text=True,
        capture_output=True,
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "node64_dspark shares node64 source paths; dry-run skips self-copy" in result.stdout
    assert not call_log.exists()


def test_check_compares_streamed_deterministic_fingerprints(tmp_path: Path, fake_transport) -> None:
    fake_bin, call_log = fake_transport
    root = _fixture_repo(tmp_path)
    tile = tmp_path / "TileKernels"
    remote_root = tmp_path / "remote-roots"
    remote_tile_root = tmp_path / "remote-tile-roots"
    fingerprint_helper = root / "scripts/sync/runtime_fingerprint.sh"
    main_provenance = _provenance(root, fingerprint_helper, tree=False)
    tile_provenance = _provenance(tile, fingerprint_helper, tree=True)
    for host in ("node69_slime", "node53_dspark", "node70_dspark"):
        shutil.copytree(root, remote_root / host)
        shutil.copytree(tile, remote_tile_root / host)
        (remote_root / host / ".slime-v4-source-provenance").write_text(main_provenance)
        (remote_tile_root / host / ".slime-v4-source-provenance").write_text(tile_provenance)

    env = {
        **_env(fake_bin, call_log),
        "FAKE_REMOTE_ROOT": str(remote_root),
        "FAKE_REMOTE_TILE_ROOT": str(remote_tile_root),
        "FAKE_FINGERPRINT_SCRIPT": str(fingerprint_helper),
        "SYNC_V4_TILEKERNELS_DIR": str(tile),
    }
    script = root / "scripts/sync/sync_dsv4_active_nodes.sh"

    # Non-delete syncs may leave historical files. They are outside the source
    # manifest and therefore must not make a correctly synced checkout fail.
    (remote_root / "node53_dspark/tests/stale_remote_only.py").write_text("stale\n")
    matching = subprocess.run([str(script), "--check"], text=True, capture_output=True, env=env)
    assert matching.returncode == 0, matching.stderr
    assert "all 3 target(s) match" in matching.stdout

    (remote_root / "node70_dspark/train.py").write_text("# divergent\n")
    mismatch = subprocess.run([str(script), "--check"], text=True, capture_output=True, env=env)
    assert mismatch.returncode == 1
    assert "content mismatch: train.py" in mismatch.stdout

    (remote_root / "node70_dspark/train.py").write_text("# train\n")
    (remote_root / "node69_slime/train_async.py").unlink()
    missing = subprocess.run([str(script), "--check"], text=True, capture_output=True, env=env)
    assert missing.returncode == 1
    assert "missing or wrong-type file: train_async.py" in missing.stdout


def test_fingerprint_ignores_excluded_artifacts_but_tracks_runtime_content(tmp_path: Path) -> None:
    root = _fixture_repo(tmp_path)
    helper = root / "scripts/sync/runtime_fingerprint.sh"

    before = subprocess.check_output([str(helper), str(root)], text=True).strip()
    (root / "slime/__pycache__").mkdir()
    (root / "slime/__pycache__/ignored.pyc").write_text("noise\n")
    (root / "tests/debug.log").write_text("noise\n")
    (root / "tests/ignored.log").mkdir()
    (root / "tests/ignored.log/content.txt").write_text("noise\n")
    (root / "examples/ignored.pyc").symlink_to("keep.txt")
    excluded = subprocess.check_output([str(helper), str(root)], text=True).strip()
    assert excluded == before

    (root / "train.py").write_text("# changed\n")
    changed = subprocess.check_output([str(helper), str(root)], text=True).strip()
    assert changed != before


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
