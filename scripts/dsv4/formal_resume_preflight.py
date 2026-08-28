#!/usr/bin/env python3
"""Resolve and validate the formal DS-V4 PP1/CP2 resume checkpoint.

The target save directory is node-local.  Call ``snapshot`` once locally and
once through ssh, then pass both JSON objects to ``resolve``.  Keeping remote
I/O outside this helper makes the resume decision a small, deterministic
function that is straightforward to unit test.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MIGRATION_ITERATION = 59
EXPECTED_LORA_TENSORS = 766
EXPECTED_LORA_NUMEL = 114_933_760

ITERATION_DIR_RE = re.compile(r"^iter_([0-9]{7})$")
SHARED_ITERATION_FILES = (
    ".metadata",
    "common.pt",
    "metadata.json",
    "v4_lora_scaling.json",
    "v4_adapter_checkpoint.json",
)
MARKER_COUNT_FIELDS = {
    "lora_param_tensors": EXPECTED_LORA_TENSORS,
    "lora_param_numel": EXPECTED_LORA_NUMEL,
    "optimizer_param_tensors": EXPECTED_LORA_TENSORS,
    "optimizer_param_numel": EXPECTED_LORA_NUMEL,
    "fp32_master_tensors": EXPECTED_LORA_TENSORS,
    "fp32_master_numel": EXPECTED_LORA_NUMEL,
    "optimizer_state_tensors": EXPECTED_LORA_TENSORS,
    "optimizer_state_numel": EXPECTED_LORA_NUMEL,
}
NATIVE_TOPOLOGY = {
    "tensor_model_parallel_size": 1,
    "pipeline_model_parallel_size": 1,
    "context_parallel_size": 2,
    "expert_model_parallel_size": 8,
    "world_size": 16,
    "cp_partition_mode": "contiguous",
}


class PreflightError(RuntimeError):
    """A resume state is incomplete, inconsistent, or unsafe to load."""


@dataclass(frozen=True)
class TargetSnapshot:
    """Filesystem facts needed to decide migration versus native resume."""

    root: str
    root_exists: bool
    latest_present: bool
    latest_text: str | None
    iter_dir_names: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["iter_dir_names"] = list(self.iter_dir_names)
        return data

    @classmethod
    def from_json_dict(cls, data: Mapping[str, Any]) -> TargetSnapshot:
        required = {
            "root",
            "root_exists",
            "latest_present",
            "latest_text",
            "iter_dir_names",
        }
        missing = required - set(data)
        if missing:
            raise PreflightError(f"snapshot is missing fields: {sorted(missing)}")
        if not isinstance(data["root"], str):
            raise PreflightError("snapshot root must be a string")
        if type(data["root_exists"]) is not bool:
            raise PreflightError("snapshot root_exists must be a boolean")
        if type(data["latest_present"]) is not bool:
            raise PreflightError("snapshot latest_present must be a boolean")
        latest_text = data["latest_text"]
        if latest_text is not None and not isinstance(latest_text, str):
            raise PreflightError("snapshot latest_text must be a string or null")
        names = data["iter_dir_names"]
        if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
            raise PreflightError("snapshot iter_dir_names must be a list of strings")
        if not data["latest_present"] and latest_text is not None:
            raise PreflightError("snapshot without a latest marker cannot have latest_text")
        return cls(
            root=data["root"],
            root_exists=data["root_exists"],
            latest_present=data["latest_present"],
            latest_text=latest_text,
            iter_dir_names=tuple(names),
        )


@dataclass(frozen=True)
class ResumePlan:
    mode: str
    checkpoint_iteration: int
    start_rollout_id: int
    load_rng: int


def inspect_target(root: Path) -> TargetSnapshot:
    """Inspect only the target root's latest marker and immediate iter dirs.

    A missing root is equivalent to an empty root.  Unrelated entries such as
    ``rollout/`` do not make a target non-empty, while every directory whose
    name starts with ``iter_`` is reported so migration cannot ignore a partial
    or malformed checkpoint.
    """

    root = root.expanduser()
    if root.exists() and not root.is_dir():
        raise PreflightError(f"target root exists but is not a directory: {root}")
    root_exists = root.is_dir()
    latest_path = root / "latest_checkpointed_iteration.txt"
    latest_present = latest_path.exists()
    latest_text: str | None = None
    if latest_present:
        if not latest_path.is_file():
            latest_text = "<not-a-file>"
        else:
            try:
                latest_text = latest_path.read_text().strip()
            except OSError as exc:
                latest_text = f"<unreadable:{exc}>"

    iter_dir_names: tuple[str, ...] = ()
    if root_exists:
        try:
            iter_dir_names = tuple(
                sorted(entry.name for entry in root.iterdir() if entry.is_dir() and entry.name.startswith("iter_"))
            )
        except OSError as exc:
            raise PreflightError(f"cannot inspect target root {root}: {exc}") from exc

    return TargetSnapshot(
        root=str(root),
        root_exists=root_exists,
        latest_present=latest_present,
        latest_text=latest_text,
        iter_dir_names=iter_dir_names,
    )


def _numeric_latest(snapshot: TargetSnapshot, label: str) -> int:
    text = snapshot.latest_text
    if text is None or re.fullmatch(r"[0-9]+", text) is None:
        raise PreflightError(f"{label} latest marker is not numeric: {text!r} (root={snapshot.root})")
    return int(text)


def resolve_resume_plan(
    local: TargetSnapshot,
    remote: TargetSnapshot,
    *,
    allow_migration: bool = True,
    minimum_native_iteration: int = MIGRATION_ITERATION + 1,
) -> ResumePlan:
    """Resolve a consistent migration or native formal DS-V4 resume state.

    The defaults preserve the historical PP2->PP1 migration contract.  A new
    zero-based formal generation uses ``allow_migration=False`` and
    ``minimum_native_iteration=0`` so an empty root is never silently mapped
    back to iter59 and its first native checkpoint can be resumed.
    """

    if type(allow_migration) is not bool:
        raise PreflightError("allow_migration must be a boolean")
    if type(minimum_native_iteration) is not int or minimum_native_iteration < 0:
        raise PreflightError(
            "minimum_native_iteration must be a non-negative integer, " f"got {minimum_native_iteration!r}"
        )

    snapshots = (("local", local), ("remote", remote))
    if all(not snapshot.latest_present and not snapshot.iter_dir_names for _label, snapshot in snapshots):
        if not allow_migration:
            raise PreflightError(
                "native-only target has no checkpoint on either node; "
                "use the explicit fresh-start launch instead of resuming"
            )
        return ResumePlan(
            mode="migration",
            checkpoint_iteration=MIGRATION_ITERATION,
            start_rollout_id=MIGRATION_ITERATION + 1,
            load_rng=0,
        )

    if all(not snapshot.latest_present for _label, snapshot in snapshots):
        raise PreflightError(
            "target iteration directories exist without latest markers: "
            f"local={list(local.iter_dir_names)} remote={list(remote.iter_dir_names)}"
        )
    if not all(snapshot.latest_present for _label, snapshot in snapshots):
        raise PreflightError(
            "local/remote target latest markers disagree: "
            f"local={local.latest_present} remote={remote.latest_present}"
        )

    local_latest = _numeric_latest(local, "local")
    remote_latest = _numeric_latest(remote, "remote")
    if local_latest != remote_latest:
        raise PreflightError(
            "local/remote target latest iterations disagree: " f"local={local_latest} remote={remote_latest}"
        )
    if local_latest < minimum_native_iteration:
        raise PreflightError(
            "native target latest iteration must be >= " f"{minimum_native_iteration}, " f"got {local_latest}"
        )

    expected_dir = f"iter_{local_latest:07d}"
    missing_latest_dir = [label for label, snapshot in snapshots if expected_dir not in snapshot.iter_dir_names]
    if missing_latest_dir:
        raise PreflightError(
            f"latest marker {local_latest} has no {expected_dir} directory on: " + ", ".join(missing_latest_dir)
        )

    return ResumePlan(
        mode="native",
        checkpoint_iteration=local_latest,
        start_rollout_id=local_latest + 1,
        load_rng=1,
    )


def _read_numeric_latest(root: Path) -> int:
    latest_path = root / "latest_checkpointed_iteration.txt"
    try:
        text = latest_path.read_text().strip()
    except OSError as exc:
        raise PreflightError(f"cannot read latest marker {latest_path}: {exc}") from exc
    if re.fullmatch(r"[0-9]+", text) is None:
        raise PreflightError(f"latest marker is not numeric: {latest_path}: {text!r}")
    return int(text)


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot read {label} JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"{label} JSON must contain an object: {path}")
    return value


def _validate_marker(iteration_dir: Path) -> dict[str, Any]:
    marker_path = iteration_dir / "v4_adapter_checkpoint.json"
    marker = _load_json_object(marker_path, "adapter marker")
    for field, expected in MARKER_COUNT_FIELDS.items():
        value = marker.get(field)
        if type(value) is not int or value != expected:
            raise PreflightError(f"adapter marker {field} must be {expected}, got {value!r}: {marker_path}")
    for field in ("saved_optimizer", "saved_rng"):
        if marker.get(field) is not True:
            raise PreflightError(f"adapter marker {field} must be true, got {marker.get(field)!r}: {marker_path}")
    return marker


def _common_arg(args: object, name: str) -> Any:
    if isinstance(args, Mapping):
        return args.get(name)
    return getattr(args, name, None)


def _validate_common(
    common_path: Path,
    iteration: int,
    *,
    require_native_topology: bool,
) -> dict[str, Any]:
    try:
        import torch

        common = torch.load(common_path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise PreflightError(f"cannot load common checkpoint {common_path}: {exc}") from exc
    if not isinstance(common, dict):
        raise PreflightError(f"common checkpoint must contain a dict: {common_path}")
    common_iteration = common.get("iteration")
    if type(common_iteration) is not int or common_iteration != iteration:
        raise PreflightError(
            f"common checkpoint iteration must be {iteration}, got {common_iteration!r}: " f"{common_path}"
        )

    topology: dict[str, Any] = {}
    if require_native_topology:
        args = common.get("args")
        if args is None:
            raise PreflightError(f"common checkpoint has no args: {common_path}")
        mismatches = {}
        for field, expected in NATIVE_TOPOLOGY.items():
            value = _common_arg(args, field)
            topology[field] = value
            if value != expected:
                mismatches[field] = {"expected": expected, "actual": value}
        if mismatches:
            raise PreflightError(
                "native checkpoint topology mismatch: " + json.dumps(mismatches, sort_keys=True) + f" ({common_path})"
            )
    return topology


def _validate_storage_manifest(metadata_path: Path, shards: list[Path]) -> None:
    """Prove that every DCP storage extent is locally readable.

    Counting files is insufficient: a checkpoint can have 32 unrelated shard
    names while ``.metadata`` points at a missing file, or contain a truncated
    file whose last tensor extent lies beyond EOF.  A fresh process resolves
    reads exclusively through these metadata entries, so this is the same
    contract the loader needs.
    """

    try:
        with metadata_path.open("rb") as stream:
            metadata = pickle.load(stream)
        storage_entries = list(metadata.storage_data.values())
    except (
        OSError,
        EOFError,
        ImportError,
        AttributeError,
        pickle.UnpicklingError,
    ) as exc:
        raise PreflightError(f"cannot read torch_dist storage metadata {metadata_path}: {exc}") from exc
    if not storage_entries:
        raise PreflightError(f"torch_dist storage metadata is empty: {metadata_path}")

    actual = {path.name: path for path in shards}
    referenced: set[str] = set()
    invalid_paths: list[str] = []
    invalid_extents: list[str] = []
    for entry in storage_entries:
        relative_path = getattr(entry, "relative_path", None)
        offset = getattr(entry, "offset", None)
        length = getattr(entry, "length", None)
        if not isinstance(relative_path, str) or not relative_path or Path(relative_path).name != relative_path:
            invalid_paths.append(repr(relative_path))
            continue
        referenced.add(relative_path)
        if type(offset) is not int or type(length) is not int or offset < 0 or length <= 0:
            invalid_extents.append(f"{relative_path}: offset={offset!r} length={length!r}")
            continue
        shard = actual.get(relative_path)
        if shard is not None:
            try:
                shard_size = shard.stat().st_size
            except OSError as exc:
                raise PreflightError(f"cannot stat torch_dist shard {shard}: {exc}") from exc
            if offset + length > shard_size:
                invalid_extents.append(f"{relative_path}: end={offset + length} size={shard_size}")

    if invalid_paths:
        raise PreflightError(f"invalid storage paths in {metadata_path}: {invalid_paths[:8]}")
    missing = sorted(referenced - set(actual))
    extra = sorted(set(actual) - referenced)
    if missing or extra:
        raise PreflightError(
            "torch_dist metadata/shard set mismatch: "
            f"missing={missing[:8]} extra={extra[:8]} ({metadata_path.parent})"
        )
    if invalid_extents:
        raise PreflightError(f"invalid or out-of-bounds storage extents in {metadata_path}: " f"{invalid_extents[:8]}")


def probe_checkpoint(
    root: Path,
    iteration: int,
    *,
    mode: str,
    expected_distcp: int | None = None,
    minimum_native_iteration: int = MIGRATION_ITERATION + 1,
) -> dict[str, Any]:
    """Validate one node-local checkpoint copy before formal DS-V4 launch."""

    if mode not in ("migration", "native"):
        raise PreflightError(f"unsupported checkpoint mode: {mode!r}")
    if type(iteration) is not int or iteration < 0:
        raise PreflightError(f"iteration must be a non-negative integer, got {iteration!r}")
    if type(minimum_native_iteration) is not int or minimum_native_iteration < 0:
        raise PreflightError(
            "minimum_native_iteration must be a non-negative integer, " f"got {minimum_native_iteration!r}"
        )
    # torch_dist metadata names every storage file globally.  The checkpoint
    # root is node-local, so a fresh process on either train node must still be
    # able to open the full referenced set; keeping only that node's 16 writer
    # shards makes load planning select files that do not exist locally.
    mode_default_distcp = 32
    if mode == "migration" and iteration != MIGRATION_ITERATION:
        raise PreflightError(f"migration checkpoint iteration must be {MIGRATION_ITERATION}, got {iteration}")
    if mode == "native" and iteration < minimum_native_iteration:
        raise PreflightError("native checkpoint iteration must be >= " f"{minimum_native_iteration}, got {iteration}")
    if expected_distcp is None:
        expected_distcp = mode_default_distcp
    if type(expected_distcp) is not int or expected_distcp <= 0:
        raise PreflightError(f"expected_distcp must be a positive integer, got {expected_distcp!r}")
    if expected_distcp != mode_default_distcp:
        raise PreflightError(
            f"{mode} checkpoint must have {mode_default_distcp} distcp files per node, " f"not {expected_distcp}"
        )

    root = root.expanduser()
    if not root.is_dir():
        raise PreflightError(f"checkpoint root is not a directory: {root}")
    latest = _read_numeric_latest(root)
    if latest != iteration:
        raise PreflightError(f"checkpoint root latest iteration is {latest}, expected {iteration}: {root}")
    iteration_dir = root / f"iter_{iteration:07d}"
    if not iteration_dir.is_dir():
        raise PreflightError(f"checkpoint iteration directory is missing: {iteration_dir}")

    missing_or_empty = [
        filename
        for filename in SHARED_ITERATION_FILES
        if not (iteration_dir / filename).is_file() or (iteration_dir / filename).stat().st_size <= 0
    ]
    if missing_or_empty:
        raise PreflightError(f"checkpoint shared files missing or empty in {iteration_dir}: {missing_or_empty}")

    shards = sorted(iteration_dir.glob("*.distcp"))
    if len(shards) != expected_distcp:
        raise PreflightError(f"expected {expected_distcp} distcp files, found {len(shards)} in {iteration_dir}")
    empty_shards = [path.name for path in shards if path.stat().st_size <= 0]
    if empty_shards:
        raise PreflightError(f"empty distcp files in {iteration_dir}: {empty_shards[:8]}")
    _validate_storage_manifest(iteration_dir / ".metadata", shards)

    marker = _validate_marker(iteration_dir)
    topology = _validate_common(
        iteration_dir / "common.pt",
        iteration,
        require_native_topology=mode == "native",
    )
    return {
        "status": "PASS",
        "mode": mode,
        "root": str(root),
        "iteration": iteration,
        "iteration_dir": str(iteration_dir),
        "distcp_count": len(shards),
        "lora_param_tensors": marker["lora_param_tensors"],
        "lora_param_numel": marker["lora_param_numel"],
        "saved_optimizer": marker["saved_optimizer"],
        "saved_rng": marker["saved_rng"],
        "topology": topology,
    }


def _decode_snapshot(text: str, label: str) -> TargetSnapshot:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PreflightError(f"invalid {label} snapshot JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"{label} snapshot JSON must contain an object")
    return TargetSnapshot.from_json_dict(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot = subparsers.add_parser("snapshot", help="inspect one target root")
    snapshot.add_argument("--root", required=True, type=Path)

    resolve = subparsers.add_parser("resolve", help="resolve two snapshot JSON objects")
    resolve.add_argument("--local-snapshot", required=True)
    resolve.add_argument("--remote-snapshot", required=True)
    resolve.add_argument(
        "--native-only",
        action="store_true",
        help="reject an empty target instead of selecting the historical iter59 migration",
    )
    resolve.add_argument(
        "--minimum-native-iteration",
        type=int,
        default=MIGRATION_ITERATION + 1,
    )

    probe = subparsers.add_parser("probe", help="validate one checkpoint copy")
    probe.add_argument("--root", required=True, type=Path)
    probe.add_argument("--iteration", required=True, type=int)
    probe.add_argument("--mode", required=True, choices=("migration", "native"))
    probe.add_argument("--expected-distcp", type=int)
    probe.add_argument(
        "--minimum-native-iteration",
        type=int,
        default=MIGRATION_ITERATION + 1,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "snapshot":
            result = inspect_target(args.root).to_json_dict()
        elif args.command == "resolve":
            local = _decode_snapshot(args.local_snapshot, "local")
            remote = _decode_snapshot(args.remote_snapshot, "remote")
            result = asdict(
                resolve_resume_plan(
                    local,
                    remote,
                    allow_migration=not args.native_only,
                    minimum_native_iteration=args.minimum_native_iteration,
                )
            )
        else:
            result = probe_checkpoint(
                args.root,
                args.iteration,
                mode=args.mode,
                expected_distcp=args.expected_distcp,
                minimum_native_iteration=args.minimum_native_iteration,
            )
    except PreflightError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
