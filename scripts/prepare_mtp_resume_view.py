#!/usr/bin/env python3
"""Build a node-local DCP load view for Qwen's shared MTP embedding.

MCore saves the embedding once on the first PP stage. The last stage needs
the same tensor for MTP, but its node-local checkpoint lacks the owning shard
files. Materialize only those serialized tensor ranges, then redirect their
storage entries in a disposable load view. Original checkpoints stay canonical
for saving, archival and conversion.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import pickle
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

EMBEDDING_KEY = "embedding.word_embeddings.weight"
MANIFEST = "mtp_resume_view.json"
REMOTE_READ = """
import hashlib,sys
from pathlib import Path
root=Path(sys.argv[1]); name=sys.argv[2]; offset=int(sys.argv[3]); length=int(sys.argv[4])
assert hashlib.sha256((root/'.metadata').read_bytes()).hexdigest()==sys.argv[5], 'checkpoint metadata differs'
assert Path(name).name==name, 'invalid storage path'
with (root/name).open('rb') as source:
    assert (root/name).stat().st_size>=offset+length, 'incomplete storage file'
    source.seek(offset)
    while length:
        block=source.read(min(length,1024*1024))
        if not block: raise RuntimeError('short storage read')
        sys.stdout.buffer.write(block)
        length-=len(block)
"""


def fetch_range(sources, source_dir, relative_path, offset, length, metadata_sha256, output):
    errors = []
    for host, port in sources:
        command = shlex.join(
            [
                "python3",
                "-c",
                REMOTE_READ,
                str(source_dir),
                relative_path,
                str(offset),
                str(length),
                metadata_sha256,
            ]
        )
        with output.open("wb") as target, tempfile.TemporaryFile() as stderr:
            try:
                proc = subprocess.run(
                    ["ssh", "-p", str(port), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", host, command],
                    stdout=target,
                    stderr=stderr,
                    timeout=600,
                )
            except (subprocess.TimeoutExpired, OSError) as exc:
                errors.append(f"{host}:{port}: {exc}")
                continue
            stderr.seek(0)
            detail = stderr.read().decode(errors="replace")
        if proc.returncode == 0 and output.stat().st_size == length:
            return host
        errors.append(f"{host}:{port}: {detail[-500:]}")
    raise RuntimeError(f"Cannot read {relative_path}[{offset}:{offset + length}]: {errors}")


def prepare_view(checkpoint_root: Path, output_root: Path, iteration: int, fetch) -> dict:
    checkpoint_root = checkpoint_root.resolve()
    output_root = output_root.absolute()
    if output_root == checkpoint_root or checkpoint_root in output_root.parents:
        raise ValueError("The disposable load view must be outside the canonical checkpoint directory")
    iteration_name = f"iter_{iteration:07d}"
    source_dir = checkpoint_root / iteration_name
    metadata_bytes = (source_dir / ".metadata").read_bytes()
    metadata_sha = hashlib.sha256(metadata_bytes).hexdigest()
    metadata = pickle.loads(metadata_bytes)
    embedding = [(index, info) for index, info in metadata.storage_data.items() if index.fqn == EMBEDDING_KEY]
    if not embedding:
        raise ValueError(f"No shared embedding found in {source_dir}")
    if output_root.exists():
        marker = output_root / MANIFEST
        if not marker.is_file() or json.loads(marker.read_text()).get("checkpoint_root") != str(checkpoint_root):
            raise ValueError(f"Refusing to replace an unrecognized load-view directory: {output_root}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        view = temporary / iteration_name
        view.mkdir()
        for entry in source_dir.iterdir():
            if entry.name != ".metadata":
                (view / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
        # Rollout data-source state lives beside the iteration directory.
        for entry in checkpoint_root.iterdir():
            if entry.name == "latest_checkpointed_iteration.txt" or entry.name.startswith("iter_"):
                continue
            (temporary / entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
        chunks = []
        for chunk, (index, info) in enumerate(embedding):
            if Path(info.relative_path).name != info.relative_path:
                raise ValueError(f"Unexpected DCP storage path: {info.relative_path}")
            local = source_dir / info.relative_path
            if local.exists():
                if local.stat().st_size < info.offset + info.length:
                    raise RuntimeError(f"Incomplete local checkpoint storage: {local}")
                continue
            name = f"mtp_embedding_{chunk:03d}.distcp"
            target = view / name
            owner = fetch(source_dir, info.relative_path, info.offset, info.length, metadata_sha, target)
            if target.stat().st_size != info.length:
                raise RuntimeError(f"Incomplete embedding replica: {target}")
            metadata.storage_data[index] = dataclasses.replace(info, relative_path=name, offset=0)
            digest = hashlib.sha256()
            with target.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            chunks.append(
                {
                    "source": owner,
                    "storage_file": info.relative_path,
                    "offset": info.offset,
                    "length": info.length,
                    "replica": name,
                    "sha256": digest.hexdigest(),
                }
            )
        if (source_dir / ".metadata").read_bytes() != metadata_bytes:
            raise RuntimeError("Checkpoint metadata changed during view preparation")
        (view / ".metadata").write_bytes(pickle.dumps(metadata))
        (temporary / "latest_checkpointed_iteration.txt").write_text(str(iteration))
        manifest = {
            "checkpoint_root": str(checkpoint_root),
            "iteration": iteration,
            "canonical_metadata_sha256": metadata_sha,
            "materialized_bytes": sum(x["length"] for x in chunks),
            "chunks": chunks,
        }
        (temporary / MANIFEST).write_text(json.dumps(manifest, indent=2))
        # The only replaced directory is a recognized, regenerable cache;
        # rmtree does not follow its links into the canonical checkpoint.
        if output_root.exists():
            shutil.rmtree(output_root)
        os.replace(temporary, output_root)
        return manifest
    except BaseException:
        shutil.rmtree(temporary)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--source", nargs=2, action="append", metavar=("SSH_HOST", "SSH_PORT"), required=True)
    args = parser.parse_args()

    def fetch(*values):
        return fetch_range(args.source, *values)

    print(json.dumps(prepare_view(args.checkpoint_root, args.output_root, args.iteration, fetch), indent=2))


if __name__ == "__main__":
    main()
