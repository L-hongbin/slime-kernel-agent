#!/usr/bin/env python3
"""Archive node-local distributed checkpoints with verified copy semantics.

Megatron assumes ``--save`` is shared, while the Qwen3.8 actor nodes mount
``/nfs/FM`` from different local disks.  A full optimizer checkpoint consumes
roughly 236 GiB on each actor host, so the next save cannot coexist there.
This watcher pulls every finalized iteration to an archive host, verifies every
file by SHA256 and atomically publishes the archive.  The default rolling mode
then removes the exact remote ``iter_XXXXXXX`` directories; ``--retain-source``
keeps every node-local source checkpoint intact for non-destructive archival.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


LOG = logging.getLogger("rolling-checkpoint-archive")
ITER_RE = re.compile(r"^iter_(\d{7})$")


@dataclass(frozen=True)
class HostSpec:
    label: str
    target: str


def parse_host(value: str) -> HostSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("host must be LABEL=SSH_TARGET")
    label, target = value.split("=", 1)
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
        raise argparse.ArgumentTypeError(f"unsafe host label: {label!r}")
    if not re.fullmatch(r"[A-Za-z0-9_.@:-]+", target):
        raise argparse.ArgumentTypeError(f"unsafe SSH target: {target!r}")
    return HostSpec(label=label, target=target)


def validate_checkpoint_dir(path: str) -> str:
    if not path.startswith("/") or any(char.isspace() for char in path):
        raise ValueError(f"checkpoint dir must be an absolute whitespace-free path: {path!r}")
    if "/experiments/" not in path or not path.endswith("/checkpoints") or ".." in path:
        raise ValueError(f"refusing unsafe checkpoint dir: {path!r}")
    return path.rstrip("/")


def file_manifest(root: Path) -> dict[str, dict[str, int | str]]:
    result: dict[str, dict[str, int | str]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(16 * 1024 * 1024):
                digest.update(chunk)
        result[path.relative_to(root).as_posix()] = {
            "size": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    return result


def manifest_bytes(manifest: dict[str, dict[str, int | str]]) -> int:
    return sum(int(record["size"]) for record in manifest.values())


def select_finalized_iterations(
    trackers: dict[str, int | None],
    iterations: dict[str, set[int]],
) -> list[int]:
    finalized = [value for value in trackers.values() if value is not None]
    if not finalized or not iterations:
        return []
    common = set.intersection(*iterations.values())
    return sorted(iteration for iteration in common if iteration <= max(finalized))


class Transport(Protocol):
    def read_tracker(self, host: HostSpec, checkpoint_dir: str) -> int | None: ...

    def list_iterations(self, host: HostSpec, checkpoint_dir: str) -> set[int]: ...

    def remote_manifest(
        self, host: HostSpec, checkpoint_dir: str, iteration: int
    ) -> dict[str, dict[str, int | str]]: ...

    def pull_iteration(self, host: HostSpec, checkpoint_dir: str, iteration: int, destination: Path) -> None: ...

    def pull_state(self, host: HostSpec, checkpoint_dir: str, destination: Path) -> None: ...

    def remove_iteration(self, host: HostSpec, checkpoint_dir: str, iteration: int) -> None: ...


class SshTransport:
    def __init__(self, *, ssh_port: int) -> None:
        self.ssh_port = ssh_port

    def _ssh(self, host: HostSpec, argv: list[str]) -> str:
        if host.target == "local":
            result = subprocess.run(
                argv,
                check=True,
                text=True,
                capture_output=True,
            )
            return result.stdout
        result = subprocess.run(
            [
                "ssh",
                "-p",
                str(self.ssh_port),
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=15",
                host.target,
                shlex.join(argv),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        return result.stdout

    def _python(self, host: HostSpec, code: str, *args: str) -> str:
        return self._ssh(host, ["python3", "-c", code, *args])

    def read_tracker(self, host: HostSpec, checkpoint_dir: str) -> int | None:
        code = (
            "from pathlib import Path; import sys; "
            "p=Path(sys.argv[1])/'latest_checkpointed_iteration.txt'; "
            "print(p.read_text().strip() if p.is_file() else '')"
        )
        value = self._python(host, code, checkpoint_dir).strip()
        return int(value) if value.isdigit() else None

    def list_iterations(self, host: HostSpec, checkpoint_dir: str) -> set[int]:
        code = (
            "from pathlib import Path; import json,re,sys; "
            "root=Path(sys.argv[1]); pat=re.compile(r'^iter_(\\d{7})$'); "
            "print(json.dumps(sorted(int(m.group(1)) for p in root.glob('iter_*') "
            "if p.is_dir() and (m:=pat.fullmatch(p.name)))))"
        )
        return set(json.loads(self._python(host, code, checkpoint_dir)))

    def remote_manifest(self, host: HostSpec, checkpoint_dir: str, iteration: int) -> dict[str, dict[str, int | str]]:
        code = r"""
from pathlib import Path
import hashlib, json, sys
root = Path(sys.argv[1])
out = {}
for path in sorted(item for item in root.rglob('*') if item.is_file()):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    out[path.relative_to(root).as_posix()] = {'size': path.stat().st_size, 'sha256': digest.hexdigest()}
print(json.dumps(out, sort_keys=True))
"""
        source = f"{checkpoint_dir}/iter_{iteration:07d}"
        return json.loads(self._python(host, code, source))

    def _rsync(self, host: HostSpec, source: str, destination: Path, *extra: str) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        if host.target == "local":
            subprocess.run(
                [
                    "rsync",
                    "-a",
                    "--partial",
                    *extra,
                    f"{source.rstrip('/')}/",
                    f"{destination}/",
                ],
                check=True,
            )
            return
        subprocess.run(
            [
                "rsync",
                "-a",
                "--partial",
                *extra,
                "-e",
                f"ssh -p {self.ssh_port} -o BatchMode=yes -o ConnectTimeout=15",
                f"{host.target}:{source.rstrip('/')}/",
                f"{destination}/",
            ],
            check=True,
        )

    def pull_iteration(self, host: HostSpec, checkpoint_dir: str, iteration: int, destination: Path) -> None:
        self._rsync(host, f"{checkpoint_dir}/iter_{iteration:07d}", destination)

    def pull_state(self, host: HostSpec, checkpoint_dir: str, destination: Path) -> None:
        self._rsync(host, checkpoint_dir, destination, "--exclude=iter_*")

    def remove_iteration(self, host: HostSpec, checkpoint_dir: str, iteration: int) -> None:
        expected = f"{checkpoint_dir}/iter_{iteration:07d}"
        code = r"""
from pathlib import Path
import re, shutil, sys
root = Path(sys.argv[1])
target = Path(sys.argv[2])
if target.parent != root or re.fullmatch(r'iter_\d{7}', target.name) is None:
    raise SystemExit(f'refusing unsafe deletion: {target}')
if target.exists():
    shutil.rmtree(target)
if target.exists():
    raise SystemExit(f'deletion did not complete: {target}')
"""
        self._python(host, code, checkpoint_dir, expected)


class RollingArchiver:
    def __init__(
        self,
        *,
        hosts: list[HostSpec],
        checkpoint_dir: str,
        archive_dir: Path,
        transport: Transport,
        keep_archives: int,
        min_free_bytes: int,
        retain_source: bool = False,
    ) -> None:
        if len(hosts) < 2:
            raise ValueError("at least two node-local checkpoint hosts are required")
        if keep_archives < 1:
            raise ValueError("keep_archives must be positive")
        self.hosts = hosts
        self.checkpoint_dir = validate_checkpoint_dir(checkpoint_dir)
        self.archive_dir = archive_dir.resolve()
        if str(self.archive_dir).startswith(f"{self.checkpoint_dir}/"):
            raise ValueError("archive_dir must not be inside the node-local checkpoint dir")
        self.transport = transport
        self.keep_archives = keep_archives
        self.min_free_bytes = min_free_bytes
        self.retain_source = retain_source
        self.archive_dir.mkdir(parents=True, exist_ok=True)

    def _parallel(self, function, items):
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.hosts)) as pool:
            futures = {pool.submit(function, item): item for item in items}
            return {futures[future].label: future.result() for future in concurrent.futures.as_completed(futures)}

    def _write_status(self, payload: dict) -> None:
        temporary = self.archive_dir / ".watcher_status.json.tmp"
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, self.archive_dir / "watcher_status.json")

    def candidates(self) -> list[int]:
        trackers = self._parallel(lambda host: self.transport.read_tracker(host, self.checkpoint_dir), self.hosts)
        iterations = self._parallel(lambda host: self.transport.list_iterations(host, self.checkpoint_dir), self.hosts)
        candidates = select_finalized_iterations(trackers, iterations)
        self._write_status(
            {
                "state": "poll",
                "time": time.time(),
                "trackers": trackers,
                "iterations": {key: sorted(value) for key, value in iterations.items()},
                "candidates": candidates,
            }
        )
        return candidates

    def archive_iteration(self, iteration: int) -> None:
        name = f"iter_{iteration:07d}"
        final = self.archive_dir / name
        if final.is_dir() and (final / "manifest.json").is_file():
            if self.retain_source:
                LOG.info("%s is already verified; retaining node-local source copies", name)
            else:
                LOG.info("%s is already verified; pruning any remaining node-local copies", name)
                self._parallel(
                    lambda host: self.transport.remove_iteration(host, self.checkpoint_dir, iteration), self.hosts
                )
            self.prune_old_archives()
            return

        staging = self.archive_dir / f".{name}.partial"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        LOG.info("hashing finalized node-local shards for %s", name)
        remote_manifests = self._parallel(
            lambda host: self.transport.remote_manifest(host, self.checkpoint_dir, iteration), self.hosts
        )
        required = sum(manifest_bytes(value) for value in remote_manifests.values())
        free = shutil.disk_usage(self.archive_dir).free
        if free - required < self.min_free_bytes:
            raise RuntimeError(
                f"archive filesystem has {free} bytes free, needs {required} plus {self.min_free_bytes} headroom"
            )

        def pull(host: HostSpec) -> None:
            host_root = staging / host.label
            self.transport.pull_iteration(host, self.checkpoint_dir, iteration, host_root / name)
            self.transport.pull_state(host, self.checkpoint_dir, host_root / "checkpoint_state")

        LOG.info("copying %s bytes for %s", required, name)
        self._parallel(pull, self.hosts)
        for host in self.hosts:
            local = file_manifest(staging / host.label / name)
            if local != remote_manifests[host.label]:
                raise RuntimeError(f"SHA256 verification failed for {name} from {host.label}")

        manifest = {
            "iteration": iteration,
            "verified": True,
            "verified_at": time.time(),
            "checkpoint_dir": self.checkpoint_dir,
            "hosts": {host.label: host.target for host in self.hosts},
            "files": remote_manifests,
            "total_bytes": required,
            "source_retained": self.retain_source,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(staging, final)
        LOG.info("published verified archive %s", final)

        if self.retain_source:
            LOG.info("retained node-local copies of %s after verification", name)
        else:
            self._parallel(
                lambda host: self.transport.remove_iteration(host, self.checkpoint_dir, iteration), self.hosts
            )
            LOG.info("removed node-local copies of %s after verification", name)
        self.prune_old_archives()
        self._write_status(
            {
                "state": "archived",
                "time": time.time(),
                "iteration": iteration,
                "archive": str(final),
                "total_bytes": required,
                "source_retained": self.retain_source,
            }
        )

    def prune_old_archives(self) -> None:
        archives = sorted(
            (path for path in self.archive_dir.iterdir() if path.is_dir() and ITER_RE.fullmatch(path.name)),
            key=lambda path: int(ITER_RE.fullmatch(path.name).group(1)),
        )
        for path in archives[: -self.keep_archives]:
            LOG.info("pruning superseded verified archive %s", path)
            shutil.rmtree(path)

    def poll_once(self) -> int:
        processed = 0
        for iteration in self.candidates():
            self.archive_iteration(iteration)
            processed += 1
        # Also repair a crash after atomic publication/source deletion but
        # before superseded archives were pruned. In that state no remote
        # iteration remains, so candidate-based cleanup alone cannot revisit it.
        self.prune_old_archives()
        return processed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", action="append", type=parse_host, required=True)
    parser.add_argument("--ssh-port", type=int, default=23538)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--keep-archives", type=int, default=1)
    parser.add_argument("--min-free-gib", type=int, default=64)
    parser.add_argument("--poll-interval", type=int, default=60)
    parser.add_argument(
        "--retain-source",
        action="store_true",
        help="copy and verify checkpoints without deleting node-local iter_* source directories",
    )
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.poll_interval < 10:
        raise SystemExit("poll interval must be at least 10 seconds")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    archiver = RollingArchiver(
        hosts=args.host,
        checkpoint_dir=args.checkpoint_dir,
        archive_dir=args.archive_dir,
        transport=SshTransport(ssh_port=args.ssh_port),
        keep_archives=args.keep_archives,
        min_free_bytes=args.min_free_gib * 1024**3,
        retain_source=args.retain_source,
    )
    lock_path = archiver.archive_dir / ".watcher.lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"another watcher holds {lock_path}") from exc
        while True:
            try:
                processed = archiver.poll_once()
                LOG.info("poll complete: processed=%s", processed)
            except Exception:
                LOG.exception("checkpoint archive poll failed; source checkpoints were not pruned")
                archiver._write_status({"state": "error", "time": time.time()})
                if args.once:
                    raise
            if args.once:
                return
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
