"""Adapter-only (LoRA) checkpointing for V4-Flash — Bridge-style state-dict filter.

For LoRA only the adapter params change; the frozen base reloads cold from the
converted ``torch_dist`` ``--load``.  Muon is built from trainable parameters and
therefore should own only LoRA fp32 masters and momentum.  Before saving, this
module proves that ownership relation by object identity and rejects an oversized
node-local checkpoint after the write.

Megatron-Bridge supports adapter-only save via ``adapter_key_filter`` /
``apply_peft_adapter_filter_to_state_dict`` (peft/base.py) — but slime saves
through ``megatron.training.checkpointing.save_checkpoint`` (the full-model path),
so we replicate the filter here: wrap each model chunk's ``sharded_state_dict`` to
keep only adapter params on save, and to request only adapter params on resume.

V4 uses ``LinearAdapter`` (LoRA params ``linear_in.weight`` / ``linear_out.weight``,
base frozen at ``.weight``), so the filter is keyed on ``requires_grad``, not on a
naming heuristic.  The filter logs kept/full sizes and aborts before the write if
it did not actually shrink the model state.
"""

import hashlib
import json
import logging
import os
import pickle
import re
import socket
from contextlib import contextmanager
from pathlib import Path

try:
    from megatron.core.pipeline_parallel.utils import unwrap_model
except ImportError:
    from megatron.core.utils import unwrap_model

logger = logging.getLogger(__name__)

_MODULE_PREFIX = re.compile(r"^(?:module\.)+")

# Adapter scaling metadata written next to each adapter-only checkpoint iteration.
# The adapter tensors alone cannot reveal the forward multiplier (alpha/r vs
# rsLoRA's alpha/sqrt(r)); the scale is reconstructed from CLI args at model build, so
# a resume with different LoRA scaling arguments would
# silently train (and serve) a different effective delta. This marker makes the
# mismatch fail loud on resume.
_SCALING_MARKER = "v4_lora_scaling.json"
_CHECKPOINT_MARKER = "v4_adapter_checkpoint.json"

_LORA_PARAM_SUFFIXES = (".linear_in.weight", ".linear_out.weight")
_DEFAULT_MAX_NODE_CHECKPOINT_BYTES = 2 * 1024**3
_DEFAULT_REPLICA_CHUNK_BYTES = 16 * 1024**2

# A training process group is stable for the lifetime of a worker.  Cache its
# node topology and the node-leader Gloo group so checkpointing every five steps
# does not leak a fresh process group at every save.
_NODE_TOPOLOGY_CACHE = {}
_NODE_LEADER_GROUP_CACHE = {}


def adapter_only_ckpt_enabled(args) -> bool:
    """Save adapter-only state exactly when the model trains LoRA.

    ``lora_dim`` is the model-independent contract: every LoRA-trained model
    saves only its adapter state, independent of the custom model provider.
    """
    return int(getattr(args, "lora_dim", 0) or 0) > 0


def _is_lora_param_name(name: str) -> bool:
    return _norm(name).endswith(_LORA_PARAM_SUFFIXES)


def _named_trainable_params(model) -> dict[int, tuple[str, object]]:
    """Return unique live trainable parameters, keyed by object identity."""
    params: dict[int, tuple[str, object]] = {}
    for chunk in model:
        for name, param in unwrap_model(chunk).named_parameters():
            if getattr(param, "requires_grad", False):
                params.setdefault(id(param), (_norm(name), param))
    return params


def _optimizer_leaf_model_params(optimizer):
    """Yield the live model Parameters owned by one Megatron optimizer leaf.

    ``Float16OptimizerWithFloat16Params`` replaces its inner optimizer's bf16
    parameters with fp32 master copies.  Its ``float16_groups`` retain the live
    model objects and are therefore the authoritative identity mapping.  FP32
    leaves use the inner optimizer parameters directly.
    """
    if getattr(optimizer, "is_stub_optimizer", False):
        return
    if hasattr(optimizer, "float16_groups"):
        for group in optimizer.float16_groups:
            yield from group
        for group in getattr(optimizer, "fp32_from_fp32_groups", ()):
            yield from group
        return
    get_parameters = getattr(optimizer, "get_parameters", None)
    if callable(get_parameters):
        yield from get_parameters()
        return
    inner = getattr(optimizer, "optimizer", None)
    if inner is not None:
        for group in getattr(inner, "param_groups", ()):
            yield from group.get("params", ())


def _optimizer_model_params(optimizer):
    """Yield live model Parameters from a (possibly chained) optimizer."""
    chained = getattr(optimizer, "chained_optimizers", None)
    if chained is not None:
        for child in chained:
            yield from _optimizer_model_params(child)
        return
    yield from _optimizer_leaf_model_params(optimizer)


def _optimizer_leaves(optimizer):
    """Yield non-chained optimizer leaves, including stubs."""
    chained = getattr(optimizer, "chained_optimizers", None)
    if chained is not None:
        for child in chained:
            yield from _optimizer_leaves(child)
        return
    if optimizer is not None:
        yield optimizer


def validate_lora_optimizer_state(model, optimizer) -> dict[str, int]:
    """Fail unless the optimizer owns exactly the live LoRA parameters.

    This is the safety boundary for DS-V4 adapter-only checkpointing with
    optimizer saving enabled. Muon should only be constructed from trainable
    LoRA matrices; checking object identity catches both a frozen-base leak and
    an adapter omitted from the optimizer before torch_dist serializes fp32
    master parameters or momentum.
    """
    trainable = _named_trainable_params(model)
    non_lora = sorted(name for name, _param in trainable.values() if not _is_lora_param_name(name))
    if non_lora:
        raise RuntimeError(
            "LoRA-only optimizer checkpoint refused: non-LoRA trainable parameters found: " + ", ".join(non_lora[:10])
        )

    owned: dict[int, object] = {}
    for param in _optimizer_model_params(optimizer):
        owned.setdefault(id(param), param)

    trainable_ids = set(trainable)
    owned_ids = set(owned)
    unexpected = owned_ids - trainable_ids
    missing = trainable_ids - owned_ids
    if unexpected or missing:
        unexpected_shapes = [tuple(getattr(owned[param_id], "shape", ())) for param_id in unexpected]
        missing_names = [trainable[param_id][0] for param_id in missing]
        raise RuntimeError(
            "LoRA-only optimizer checkpoint refused: optimizer/model parameter identity mismatch "
            f"(unexpected_optimizer_params={len(unexpected)} shapes={unexpected_shapes[:10]}, "
            f"missing_lora_params={len(missing)} names={missing_names[:10]})"
        )
    if not owned:
        raise RuntimeError("LoRA-only optimizer checkpoint refused: optimizer owns no LoRA parameters")

    fp32_master_tensors = 0
    fp32_master_numel = 0
    fp32_master_bytes = 0
    optimizer_state_tensors = 0
    optimizer_state_numel = 0
    optimizer_state_bytes = 0
    for leaf in _optimizer_leaves(optimizer):
        if getattr(leaf, "is_stub_optimizer", False):
            continue
        float16_groups = getattr(leaf, "float16_groups", None)
        if float16_groups is None:
            continue
        master_groups = getattr(leaf, "fp32_from_float16_groups", None)
        if master_groups is None or len(master_groups) != len(float16_groups):
            raise RuntimeError(
                "LoRA-only optimizer checkpoint refused: bf16 model groups do not have "
                "a one-to-one fp32 master-group mapping"
            )

        expected_inner_params = []
        for model_group, master_group in zip(float16_groups, master_groups, strict=True):
            if len(model_group) != len(master_group):
                raise RuntimeError("LoRA-only optimizer checkpoint refused: bf16/fp32 master group lengths differ")
            for model_param, master_param in zip(model_group, master_group, strict=True):
                if tuple(model_param.shape) != tuple(master_param.shape):
                    raise RuntimeError(
                        "LoRA-only optimizer checkpoint refused: bf16/fp32 master shapes differ "
                        f"({tuple(model_param.shape)} != {tuple(master_param.shape)})"
                    )
                fp32_master_tensors += 1
                fp32_master_numel += int(master_param.numel())
                fp32_master_bytes += int(master_param.numel() * master_param.element_size())
                expected_inner_params.append(master_param)
        for group in getattr(leaf, "fp32_from_fp32_groups", ()):
            expected_inner_params.extend(group)

        inner = getattr(leaf, "optimizer", None)
        if inner is None:
            raise RuntimeError(
                "LoRA-only optimizer checkpoint refused: non-stub optimizer leaf has no inner optimizer"
            )
        actual_inner_params = [
            param for group in getattr(inner, "param_groups", ()) for param in group.get("params", ())
        ]
        expected_inner_ids = {id(param) for param in expected_inner_params}
        actual_inner_ids = {id(param) for param in actual_inner_params}
        if expected_inner_ids != actual_inner_ids or len(expected_inner_params) != len(actual_inner_params):
            raise RuntimeError(
                "LoRA-only optimizer checkpoint refused: fp32 masters and inner optimizer params differ "
                f"(expected={len(expected_inner_params)} actual={len(actual_inner_params)})"
            )

        for state_param, state in getattr(inner, "state", {}).items():
            if id(state_param) not in expected_inner_ids:
                raise RuntimeError(
                    "LoRA-only optimizer checkpoint refused: optimizer state belongs to an " "untracked parameter"
                )
            for state_name, value in state.items():
                if not hasattr(value, "numel") or not hasattr(value, "element_size"):
                    continue
                state_numel = int(value.numel())
                if state_numel not in (1, int(state_param.numel())):
                    raise RuntimeError(
                        "LoRA-only optimizer checkpoint refused: optimizer state tensor has an "
                        f"unexpected size ({state_name}={state_numel}, param={state_param.numel()})"
                    )
                optimizer_state_tensors += 1
                optimizer_state_numel += state_numel
                optimizer_state_bytes += int(state_numel * value.element_size())

    numel = sum(int(param.numel()) for param in owned.values())
    float16_numel = sum(
        int(param.numel())
        for leaf in _optimizer_leaves(optimizer)
        if not getattr(leaf, "is_stub_optimizer", False)
        for group in (getattr(leaf, "float16_groups", ()) or ())
        for param in group
    )
    if fp32_master_numel != float16_numel:
        raise RuntimeError(
            "LoRA-only optimizer checkpoint refused: fp32 master numel does not match bf16 LoRA "
            f"numel ({fp32_master_numel} != {float16_numel})"
        )
    stats = {
        "optimizer_param_tensors": len(owned),
        "optimizer_param_numel": numel,
        "fp32_master_tensors": fp32_master_tensors,
        "fp32_master_numel": fp32_master_numel,
        "fp32_master_bytes": fp32_master_bytes,
        "optimizer_state_tensors": optimizer_state_tensors,
        "optimizer_state_numel": optimizer_state_numel,
        "optimizer_state_bytes": optimizer_state_bytes,
    }
    logger.info(
        "LoRA-only optimizer checkpoint validated: %d LoRA tensors/%d params, "
        "%d fp32-master bytes, %d optimizer-state bytes",
        stats["optimizer_param_tensors"],
        stats["optimizer_param_numel"],
        stats["fp32_master_bytes"],
        stats["optimizer_state_bytes"],
    )
    return stats


def validate_loaded_lora_optimizer_state(model, optimizer, marker: dict) -> dict[str, int]:
    """Prove that a stateful adapter resume restored the saved Muon state.

    ``validate_lora_optimizer_state`` establishes parameter ownership and
    measures the live fp32 masters / optimizer tensors.  After checkpoint load,
    compare those measurements with the node-local marker written by the save.
    This closes the gap where a checkpoint could advertise optimizer state but
    a loader regression silently leaves the newly constructed Muon state empty.
    """
    if not marker or not marker.get("saved_optimizer", False):
        raise RuntimeError("stateful LoRA resume cannot validate optimizer restore without a saved-optimizer marker")

    stats = validate_lora_optimizer_state(model, optimizer)
    fields = (
        "optimizer_param_tensors",
        "optimizer_param_numel",
        "fp32_master_tensors",
        "fp32_master_numel",
        "fp32_master_bytes",
        "optimizer_state_tensors",
        "optimizer_state_numel",
        "optimizer_state_bytes",
    )
    mismatches = {
        field: (int(marker.get(field, -1)), int(stats[field]))
        for field in fields
        if int(marker.get(field, -1)) != int(stats[field])
    }
    if mismatches:
        detail = ", ".join(f"{field}: checkpoint={saved} live={live}" for field, (saved, live) in mismatches.items())
        raise RuntimeError(f"LoRA optimizer state restore mismatch: {detail}")
    logger.info(
        "LoRA optimizer state restore verified: %d tensors/%d bytes",
        stats["optimizer_state_tensors"],
        stats["optimizer_state_bytes"],
    )
    return stats


def validate_adapter_checkpoint_size(
    save_dir: str,
    iteration: int,
    max_node_bytes: int = _DEFAULT_MAX_NODE_CHECKPOINT_BYTES,
) -> int:
    """Reject a node-local adapter checkpoint that looks like a full-model save.

    The ownership audit runs before serialization; this post-write guard closes
    the measurement loop and catches regressions in the actual torch_dist output.
    ``/nfs`` is node-local, so each node measures only the shards it must retain.
    """
    import torch

    dist = torch.distributed
    rank, leaders = _distributed_node_topology()
    total = None
    limit = None
    local_error = None
    if not dist.is_initialized() or rank in leaders:
        try:
            limit = int(max_node_bytes)
            if limit <= 0:
                raise ValueError("must be positive")
            it_dir = os.path.join(save_dir, f"iter_{iteration:07d}")
            total = sum(
                os.path.getsize(os.path.join(root, filename))
                for root, _dirs, files in os.walk(it_dir)
                for filename in files
            )
            if total > limit:
                raise RuntimeError(
                    "adapter-only checkpoint exceeded the node-local safety limit: "
                    f"{total} bytes > {limit} bytes in {it_dir}; refusing to continue because "
                    "the save may contain frozen base-model state"
                )
        except (OSError, RuntimeError, ValueError) as exc:
            local_error = f"adapter checkpoint size validation failed on {socket.gethostname()}: {exc}"

    if dist.is_initialized():
        reports = [None] * dist.get_world_size()
        dist.all_gather_object(reports, (total, limit, local_error) if rank in leaders else None)
        errors = list(dict.fromkeys(report[2] for report in reports if report is not None and report[2]))
        if errors:
            raise RuntimeError("; ".join(errors))
        totals = [int(report[0]) for report in reports if report is not None]
        limits = [int(report[1]) for report in reports if report is not None]
        total = max(totals)
        limit = min(limits)
        dist.barrier()
    elif local_error:
        raise RuntimeError(local_error)

    logger.info(
        "adapter-only replicated checkpoint size validated: max_node=%d bytes " "(limit=%d, save_dir=%s)",
        total,
        limit,
        save_dir,
    )
    return int(total)


def _fsync_directory(path: str) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: str, data: bytes) -> None:
    """Durably replace one small file without exposing partial contents."""
    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        _fsync_directory(parent)
    finally:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass


def _distributed_node_topology():
    """Return ``(rank, node_leaders)`` using hostnames as node identities."""
    import torch

    dist = torch.distributed
    if not dist.is_initialized():
        return 0, (0,)
    world_group = dist.group.WORLD
    cache_key = id(world_group)
    cached = _NODE_TOPOLOGY_CACHE.get(cache_key)
    if cached is not None:
        return dist.get_rank(), cached

    hostnames = [None] * dist.get_world_size()
    dist.all_gather_object(hostnames, socket.gethostname())
    first_rank_by_host = {}
    for global_rank, hostname in enumerate(hostnames):
        first_rank_by_host.setdefault(hostname, global_rank)
    leaders = tuple(first_rank_by_host.values())
    _NODE_TOPOLOGY_CACHE[cache_key] = leaders
    return dist.get_rank(), leaders


def _world_error(local_error: str | None) -> str | None:
    """Collect an error from any rank so all ranks fail instead of hanging."""
    import torch

    dist = torch.distributed
    if not dist.is_initialized():
        return local_error
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, local_error)
    messages = list(dict.fromkeys(error for error in errors if error))
    return "; ".join(messages) if messages else None


def _write_bytes_per_node(path: str, data: bytes, label: str) -> None:
    """Have one rank per node atomically write ``data``, then report globally."""
    import torch

    dist = torch.distributed
    rank, leaders = _distributed_node_topology()
    local_error = None
    if not dist.is_initialized() or rank in leaders:
        try:
            _atomic_write_bytes(path, data)
        except Exception as exc:
            local_error = f"{label} write failed on {socket.gethostname()}: {exc}"
    error = _world_error(local_error)
    if error:
        raise RuntimeError(error)
    if dist.is_initialized():
        dist.barrier()


def _safe_checkpoint_target(root: str, relative_path: str) -> str:
    if not isinstance(relative_path, str) or not relative_path:
        raise RuntimeError(f"invalid empty/non-string checkpoint path: {relative_path!r}")
    root_abs = os.path.abspath(root)
    target = os.path.abspath(os.path.join(root_abs, relative_path))
    if os.path.commonpath((root_abs, target)) != root_abs:
        raise RuntimeError(f"checkpoint metadata path escapes iteration directory: {relative_path!r}")
    return target


def _expected_distcp_files(iteration_dir: str) -> dict[str, int]:
    """Read the authoritative distcp filename/length set from ``.metadata``."""
    metadata_path = os.path.join(iteration_dir, ".metadata")
    with open(metadata_path, "rb") as fh:
        metadata = pickle.load(fh)
    storage_data = getattr(metadata, "storage_data", None)
    if not storage_data:
        raise RuntimeError(f"torch_dist metadata has no storage_data: {metadata_path}")

    expected = {}
    for info in storage_data.values():
        relative_path = getattr(info, "relative_path", None)
        if not isinstance(relative_path, str) or not relative_path.endswith(".distcp"):
            raise RuntimeError(f"unexpected torch_dist storage path in {metadata_path}: {relative_path!r}")
        _safe_checkpoint_target(iteration_dir, relative_path)
        try:
            end = int(info.offset) + int(info.length)
        except (AttributeError, TypeError, ValueError) as exc:
            raise RuntimeError(f"torch_dist metadata lacks a valid offset/length for {relative_path!r}") from exc
        if end <= 0:
            raise RuntimeError(f"invalid torch_dist file extent for {relative_path!r}: {end}")
        expected[relative_path] = max(expected.get(relative_path, 0), end)
    return expected


def _local_distcp_manifest(iteration_dir: str) -> dict[str, dict[str, object]]:
    """Hash all local shards; hashes make duplicate filenames fail closed."""
    manifest = {}
    for root, _dirs, files in os.walk(iteration_dir):
        for filename in files:
            if not filename.endswith(".distcp"):
                continue
            path = os.path.join(root, filename)
            relative_path = os.path.relpath(path, iteration_dir)
            _safe_checkpoint_target(iteration_dir, relative_path)
            digest = hashlib.sha256()
            with open(path, "rb") as fh:
                while chunk := fh.read(8 * 1024**2):
                    digest.update(chunk)
            manifest[relative_path] = {
                "size": os.path.getsize(path),
                "sha256": digest.hexdigest(),
            }
    return manifest


def _plan_distcp_replication(
    expected: dict[str, int],
    manifests: list[dict[str, dict[str, object]]],
    leader_ranks: tuple[int, ...],
) -> dict[str, dict[str, object]]:
    """Validate the cross-node union and choose one canonical owner per shard."""
    if len(manifests) != len(leader_ranks):
        raise RuntimeError("node-leader manifest count does not match node-leader rank count")
    expected_names = set(expected)
    union_names = set().union(*(set(manifest) for manifest in manifests))
    if union_names != expected_names:
        missing = sorted(expected_names - union_names)
        extra = sorted(union_names - expected_names)
        raise RuntimeError(
            "node-local distcp union does not equal .metadata storage set "
            f"(missing={missing[:10]}, extra={extra[:10]})"
        )

    plan = {}
    for relative_path in sorted(expected):
        holders = [index for index, manifest in enumerate(manifests) if relative_path in manifest]
        entries = [manifests[index][relative_path] for index in holders]
        wrong_sizes = sorted({int(entry["size"]) for entry in entries} - {expected[relative_path]})
        if wrong_sizes:
            raise RuntimeError(
                f"distcp shard size disagrees with .metadata for {relative_path}: "
                f"expected={expected[relative_path]} observed={wrong_sizes}"
            )
        hashes = {str(entry["sha256"]) for entry in entries}
        if len(hashes) != 1:
            raise RuntimeError(f"same-name distcp shard hash conflict for {relative_path}: {sorted(hashes)}")
        owner_index = holders[0]
        plan[relative_path] = {
            "size": expected[relative_path],
            "sha256": hashes.pop(),
            "owner_rank": leader_ranks[owner_index],
            "holder_ranks": tuple(leader_ranks[index] for index in holders),
        }
    return plan


def _get_node_leader_group(leaders: tuple[int, ...]):
    import torch

    dist = torch.distributed
    if len(leaders) == 1:
        return None
    cache_key = (id(dist.group.WORLD), leaders)
    if cache_key not in _NODE_LEADER_GROUP_CACHE:
        if not dist.is_gloo_available():
            raise RuntimeError("adapter checkpoint replication requires the Gloo backend")
        # All global ranks call new_group in the same order. Only node leaders
        # subsequently participate in its CPU collectives.
        _NODE_LEADER_GROUP_CACHE[cache_key] = dist.new_group(
            ranks=list(leaders), backend="gloo", group_desc="v4-adapter-node-leaders"
        )
    return _NODE_LEADER_GROUP_CACHE[cache_key]


def _leader_all_gather(dist, group, world_size: int, value):
    if world_size == 1:
        return [value]
    gathered = [None] * world_size
    dist.all_gather_object(gathered, value, group=group)
    return gathered


def _leader_sync_error(torch, dist, group, world_size: int, local_error: str | None):
    if world_size == 1:
        return local_error
    ok = torch.tensor([0 if local_error else 1], dtype=torch.int32)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=group)
    if int(ok.item()) == 1:
        return None
    errors = _leader_all_gather(dist, group, world_size, local_error)
    messages = list(dict.fromkeys(error for error in errors if error))
    return "; ".join(messages)


def _replicate_one_distcp_file(
    *,
    iteration_dir: str,
    relative_path: str,
    spec: dict[str, object],
    rank: int,
    leaders: tuple[int, ...],
    group,
    chunk_bytes: int,
) -> str | None:
    """Stream one shard between node leaders with synchronized I/O failures."""
    import torch

    dist = torch.distributed
    leader_count = len(leaders)
    owner_rank = int(spec["owner_rank"])
    holder_ranks = tuple(int(value) for value in spec["holder_ranks"])
    size = int(spec["size"])
    expected_hash = str(spec["sha256"])
    path = _safe_checkpoint_target(iteration_dir, relative_path)
    needs_write = rank not in holder_ranks
    tmp_path = f"{path}.replicating.{os.getpid()}"
    source_fh = None
    destination_fh = None
    local_error = None
    try:
        try:
            buffer = torch.empty(min(chunk_bytes, size), dtype=torch.uint8)
            if rank == owner_rank:
                source_fh = open(path, "rb")
            if needs_write:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                destination_fh = open(tmp_path, "wb")
        except (OSError, RuntimeError) as exc:
            local_error = f"opening {relative_path} on {socket.gethostname()} failed: {exc}"
        error = _leader_sync_error(torch, dist, group, leader_count, local_error)
        if error:
            return error

        digest = hashlib.sha256()
        offset = 0
        while offset < size:
            count = min(chunk_bytes, size - offset)
            view = buffer[:count]
            local_error = None
            if rank == owner_rank:
                try:
                    read = source_fh.readinto(view.numpy())
                    if read != count:
                        raise OSError(f"short read: expected {count}, got {read}")
                except OSError as exc:
                    local_error = f"reading {relative_path} at offset {offset} failed: {exc}"
            error = _leader_sync_error(torch, dist, group, leader_count, local_error)
            if error:
                return error

            if leader_count > 1:
                dist.broadcast(view, src=owner_rank, group=group)
            payload = memoryview(view.numpy())
            digest.update(payload)
            local_error = None
            if needs_write:
                try:
                    written = destination_fh.write(payload)
                    if written != count:
                        raise OSError(f"short write: expected {count}, got {written}")
                except OSError as exc:
                    local_error = f"writing {relative_path} at offset {offset} failed: {exc}"
            error = _leader_sync_error(torch, dist, group, leader_count, local_error)
            if error:
                return error
            offset += count

        local_error = None
        try:
            if digest.hexdigest() != expected_hash:
                raise OSError(f"stream hash {digest.hexdigest()} does not match manifest {expected_hash}")
            if rank == owner_rank and source_fh.read(1):
                raise OSError("source grew after manifest collection")
            if needs_write:
                destination_fh.flush()
                os.fsync(destination_fh.fileno())
                destination_fh.close()
                destination_fh = None
                os.replace(tmp_path, path)
                _fsync_directory(os.path.dirname(path))
                if os.path.getsize(path) != size:
                    raise OSError(f"committed size is not {size}")
        except OSError as exc:
            local_error = f"committing replicated shard {relative_path} failed: {exc}"
        return _leader_sync_error(torch, dist, group, leader_count, local_error)
    finally:
        if source_fh is not None:
            source_fh.close()
        if destination_fh is not None:
            destination_fh.close()
        try:
            os.unlink(tmp_path)
        except FileNotFoundError:
            pass


def _replicate_distcp_as_node_leader(
    iteration_dir: str,
    rank: int,
    leaders: tuple[int, ...],
    group,
    chunk_bytes: int,
) -> str | None:
    import torch

    dist = torch.distributed
    leader_count = len(leaders)
    local_error = None
    expected = None
    manifest = None
    try:
        expected = _expected_distcp_files(iteration_dir)
        manifest = _local_distcp_manifest(iteration_dir)
    except Exception as exc:
        local_error = f"checkpoint manifest scan failed on {socket.gethostname()}: {exc}"
    error = _leader_sync_error(torch, dist, group, leader_count, local_error)
    if error:
        return error

    expected_by_node = _leader_all_gather(dist, group, leader_count, expected)
    if any(candidate != expected_by_node[0] for candidate in expected_by_node[1:]):
        return "node-local .metadata files disagree on the expected distcp shard set"
    manifests = _leader_all_gather(dist, group, leader_count, manifest)
    try:
        plan = _plan_distcp_replication(expected_by_node[0], manifests, leaders)
    except RuntimeError as exc:
        return str(exc)

    copied_bytes = 0
    for relative_path, spec in plan.items():
        if len(spec["holder_ranks"]) == leader_count:
            continue
        error = _replicate_one_distcp_file(
            iteration_dir=iteration_dir,
            relative_path=relative_path,
            spec=spec,
            rank=rank,
            leaders=leaders,
            group=group,
            chunk_bytes=chunk_bytes,
        )
        if error:
            return error
        if rank not in spec["holder_ranks"]:
            copied_bytes += int(spec["size"])

    # The initial local files were hashed, and every received stream was checked
    # against that hash before atomic replacement. Finish with an exact local
    # filename/length check against .metadata.
    local_error = None
    try:
        post_manifest = _local_distcp_manifest(iteration_dir)
        actual = {name: int(entry["size"]) for name, entry in post_manifest.items()}
        wrong_hash = sorted(
            name for name in set(post_manifest) & set(plan) if post_manifest[name]["sha256"] != plan[name]["sha256"]
        )
        if actual != expected or wrong_hash:
            raise RuntimeError(
                f"post-replication local shard set is incomplete (actual={len(actual)} "
                f"expected={len(expected)} wrong_hash={wrong_hash[:10]})"
            )
    except Exception as exc:
        local_error = f"post-replication validation failed on {socket.gethostname()}: {exc}"
    error = _leader_sync_error(torch, dist, group, leader_count, local_error)
    if error:
        return error
    logger.info(
        "adapter checkpoint replicated: %d global distcp files, %d bytes copied to node %s",
        len(expected),
        copied_bytes,
        socket.gethostname(),
    )
    return None


def replicate_ckpt_metadata_per_node(save_dir: str, iteration: int) -> None:
    """Atomically broadcast rank-0 torch_dist metadata to each node leader."""
    import torch

    dist = torch.distributed
    if not dist.is_initialized():
        return
    iteration_dir = os.path.join(save_dir, f"iter_{iteration:07d}")
    filenames = (".metadata", "common.pt", "metadata.json")
    rank, leaders = _distributed_node_topology()
    payload = None
    local_error = None
    if rank == 0:
        try:
            payload = {filename: Path(iteration_dir, filename).read_bytes() for filename in filenames}
        except OSError as exc:
            local_error = f"rank-0 checkpoint metadata read failed: {exc}"
    error = _world_error(local_error)
    if error:
        raise RuntimeError(error)
    box = [payload]
    dist.broadcast_object_list(box, src=0)
    payload = box[0]
    local_error = None
    if rank in leaders:
        try:
            for filename in filenames:
                _atomic_write_bytes(os.path.join(iteration_dir, filename), payload[filename])
        except (KeyError, OSError) as exc:
            local_error = f"checkpoint metadata replication failed on {socket.gethostname()}: {exc}"
    error = _world_error(local_error)
    if error:
        raise RuntimeError(error)
    dist.barrier()


def replicate_distcp_shards_per_node(save_dir: str, iteration: int, *, chunk_bytes: int | None = None) -> None:
    """Give every node the complete DCP shard union required by native load.

    Each node initially retains only files written by its local ranks, while the
    global ``.metadata`` references all ranks. One leader per node hashes its
    local manifest, leaders reject missing/extra/conflicting shards, then a Gloo
    CPU group streams only missing files through bounded buffers and atomically
    installs them. All errors are returned through a world collective so worker
    ranks do not remain stuck behind a barrier after a leader-side I/O failure.
    """
    import torch

    dist = torch.distributed
    if not dist.is_initialized():
        return
    if chunk_bytes is None:
        chunk_bytes = _DEFAULT_REPLICA_CHUNK_BYTES
    if chunk_bytes <= 0:
        raise RuntimeError("checkpoint replica chunk size must be positive")

    rank, leaders = _distributed_node_topology()
    group = _get_node_leader_group(leaders)
    local_error = None
    if rank in leaders:
        try:
            local_error = _replicate_distcp_as_node_leader(
                os.path.join(save_dir, f"iter_{iteration:07d}"),
                rank,
                leaders,
                group,
                chunk_bytes,
            )
        except Exception as exc:  # keep non-leaders from hanging on a world barrier
            local_error = (
                f"unexpected checkpoint replication failure on {socket.gethostname()}: " f"{type(exc).__name__}: {exc}"
            )
    error = _world_error(local_error)
    if error:
        raise RuntimeError(error)
    dist.barrier()


def write_latest_marker_per_node(save_dir: str, iteration: int) -> None:
    """Two-phase, atomic per-node publication after all checkpoint validation."""
    latest = os.path.join(save_dir, "latest_checkpointed_iteration.txt")
    candidate = os.path.join(save_dir, f".latest_checkpointed_iteration.{iteration}.pending")
    _write_bytes_per_node(
        candidate,
        f"{iteration}\n".encode(),
        "latest checkpoint candidate",
    )

    def publish():
        os.replace(candidate, latest)
        _fsync_directory(save_dir)

    _run_on_node_leaders("latest checkpoint publication", publish)


def _run_on_node_leaders(label: str, action) -> None:
    """Run a filesystem action once per node and make every rank see failures."""
    import torch

    dist = torch.distributed
    rank, leaders = _distributed_node_topology()
    local_error = None
    if not dist.is_initialized() or rank in leaders:
        try:
            action()
        except Exception as exc:
            local_error = f"{label} failed on {socket.gethostname()}: {exc}"
    error = _world_error(local_error)
    if error:
        raise RuntimeError(error)
    if dist.is_initialized():
        dist.barrier()


def _quarantine_existing(path: str, tag: str) -> str:
    """Recoverably move an unpublished partial checkpoint out of the way."""
    parent = os.path.dirname(path)
    basename = os.path.basename(path)
    for attempt in range(100):
        suffix = f".{tag}.{os.getpid()}" + (f".{attempt}" if attempt else "")
        quarantine = os.path.join(parent, f".{basename}{suffix}")
        if not os.path.exists(quarantine):
            os.replace(path, quarantine)
            _fsync_directory(parent)
            return quarantine
    raise RuntimeError(f"could not allocate a quarantine name for {path}")


def adapter_checkpoint_staging_dir(save_dir: str, iteration: int) -> str:
    return os.path.join(save_dir, f".pending_adapter_iter_{iteration:07d}")


def prepare_adapter_checkpoint_staging(save_dir: str, iteration: int) -> str:
    """Prepare a hidden unpublished root for one Megatron save.

    Upstream Megatron publishes its tracker as part of ``save_checkpoint``. If it
    writes directly to the final root, global rank 0 exposes an iteration before
    the peer node has the remote shards. Saving as a non-persistent *global*
    checkpoint under this hidden root keeps both its iteration directory and its
    early tracker unpublished until replication and validation have completed.
    """
    staging_dir = adapter_checkpoint_staging_dir(save_dir, iteration)
    target_dir = os.path.join(save_dir, f"iter_{iteration:07d}")

    def prepare():
        os.makedirs(save_dir, exist_ok=True)
        if os.path.exists(staging_dir):
            quarantined = _quarantine_existing(staging_dir, "abandoned")
            logger.warning("adapter checkpoint quarantined stale staging dir: %s", quarantined)
        if os.path.exists(target_dir):
            latest_path = os.path.join(save_dir, "latest_checkpointed_iteration.txt")
            latest = None
            try:
                latest = Path(latest_path).read_text().strip()
            except OSError:
                pass
            if latest == str(iteration):
                raise RuntimeError(f"refusing to overwrite published adapter checkpoint {target_dir}")
            quarantined = _quarantine_existing(target_dir, "incomplete")
            logger.warning("adapter checkpoint quarantined unpublished target dir: %s", quarantined)
        os.makedirs(staging_dir, exist_ok=False)

    _run_on_node_leaders("adapter checkpoint staging preparation", prepare)
    return staging_dir


@contextmanager
def adapter_checkpoint_staging(args, iteration: int):
    """Route a Megatron save into an unpublished per-iteration staging root.

    For async saves, Megatron's writer captures the concrete checkpoint path
    before this context restores the non-persistent arguments.  Publication is
    still deferred until :func:`finalize_adapter_checkpoint` runs from the same
    async request's finalize chain.
    """
    save_dir = args.save
    staging_dir = prepare_adapter_checkpoint_staging(save_dir, iteration)
    old_type = getattr(args, "non_persistent_ckpt_type", None)
    old_dir = getattr(args, "non_persistent_global_ckpt_dir", None)
    args.non_persistent_ckpt_type = "global"
    args.non_persistent_global_ckpt_dir = staging_dir
    try:
        yield staging_dir
    finally:
        args.non_persistent_ckpt_type = old_type
        args.non_persistent_global_ckpt_dir = old_dir


@contextmanager
def register_adapter_async_finalize(finalize_fn):
    """Attach ``finalize_fn`` to the next Megatron async save request.

    ``save_checkpoint`` builds the :class:`AsyncRequest`, adds its own distcp /
    tracker callbacks, and finally invokes the module-level
    ``schedule_async_save`` function.  Intercepting that public scheduling seam
    lets slime append the adapter replication + atomic-publication callback
    before the request is frozen, without reaching into Megatron's private
    async queue or patching the external Megatron checkout.

    The context is deliberately single-use.  A missing or duplicate schedule
    means the adapter save no longer has the lifecycle we rely on and must fail
    before training proceeds with an unpublished checkpoint.
    """
    from megatron.training import checkpointing

    original_schedule = checkpointing.schedule_async_save
    scheduled = 0

    def schedule_with_adapter_finalize(async_request):
        nonlocal scheduled
        scheduled += 1
        if scheduled > 1:
            raise RuntimeError("adapter async-save registered more than one request in a single save")
        async_request.add_finalize_fn(finalize_fn)
        return original_schedule(async_request)

    checkpointing.schedule_async_save = schedule_with_adapter_finalize
    completed = False
    try:
        yield
        completed = True
    finally:
        checkpointing.schedule_async_save = original_schedule

    if completed and scheduled != 1:
        raise RuntimeError("adapter async-save did not schedule exactly one Megatron async request")


def _validate_complete_distcp_dir(iteration_dir: str) -> None:
    expected = _expected_distcp_files(iteration_dir)
    actual = {}
    for root, _dirs, files in os.walk(iteration_dir):
        for filename in files:
            if not filename.endswith(".distcp"):
                continue
            path = os.path.join(root, filename)
            relative_path = os.path.relpath(path, iteration_dir)
            _safe_checkpoint_target(iteration_dir, relative_path)
            actual[relative_path] = os.path.getsize(path)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        wrong = sorted(name for name in set(actual) & set(expected) if actual[name] != expected[name])
        raise RuntimeError(
            "committed distcp set does not exactly match .metadata "
            f"(missing={missing[:10]}, extra={extra[:10]}, wrong_size={wrong[:10]})"
        )


def commit_adapter_checkpoint_staging(save_dir: str, staging_dir: str, iteration: int) -> None:
    """Atomically promote a validated staged iteration on every node.

    ``latest_checkpointed_iteration.txt`` is intentionally not touched here; the
    caller publishes it only after this function has succeeded globally.
    """
    staged_iteration = os.path.join(staging_dir, f"iter_{iteration:07d}")
    target_iteration = os.path.join(save_dir, f"iter_{iteration:07d}")

    def commit():
        if not os.path.isdir(staged_iteration):
            raise RuntimeError(f"staged iteration directory is missing: {staged_iteration}")
        if os.path.exists(target_iteration):
            raise RuntimeError(f"adapter checkpoint target already exists: {target_iteration}")
        _validate_complete_distcp_dir(staged_iteration)
        os.replace(staged_iteration, target_iteration)
        _fsync_directory(save_dir)
        _validate_complete_distcp_dir(target_iteration)
        try:
            os.unlink(os.path.join(staging_dir, "latest_checkpointed_iteration.txt"))
        except FileNotFoundError:
            pass
        try:
            os.rmdir(staging_dir)
        except OSError as exc:
            # The iteration itself is already atomically promoted. A harmless
            # hidden staging residue must not make a complete checkpoint unusable.
            logger.warning("adapter checkpoint could not remove empty staging root %s: %s", staging_dir, exc)

    _run_on_node_leaders("adapter checkpoint staging commit", commit)


def collect_adapter_scaling(model, *, rslora: bool = False) -> dict | None:
    """Read the LoRA scaling config off the LIVE adapter modules (source of truth).

    Returns ``{"dim", "alpha", "scale", "rslora"}`` or ``None`` when the model has
    no LoRA adapters. ``scale`` is the operative forward multiplier
    (``adapter.scale``, also what the weight-sync/export chain reads); ``rslora``
    records the parsed CLI setting for the audit trail. Raises if adapters
    disagree (uniform scaling is a V4 invariant the export chain relies on).
    """
    dims: set[int] = set()
    alphas: set[float] = set()
    scales: set[float] = set()
    for chunk in model:
        for _name, module in unwrap_model(chunk).named_modules():
            if not all(hasattr(module, attr) for attr in ("linear_in", "linear_out", "scale")):
                continue
            dims.add(int(module.dim))
            alphas.add(float(module.alpha))
            scales.add(float(module.scale))
    if not scales:
        return None
    if len(dims) != 1 or len(alphas) != 1 or len(scales) != 1:
        raise RuntimeError(
            f"inconsistent LoRA adapter scaling: dims={sorted(dims)} alphas={sorted(alphas)} scales={sorted(scales)}"
        )
    return {
        "dim": dims.pop(),
        "alpha": alphas.pop(),
        "scale": scales.pop(),
        "rslora": bool(rslora),
    }


def write_adapter_scaling_marker(save_dir: str, iteration: int, model, *, rslora: bool = False) -> None:
    """Atomically write the scaling marker once per node."""
    meta = collect_adapter_scaling(model, rslora=rslora)
    if meta is None:
        return
    it_dir = os.path.join(save_dir, f"iter_{iteration:07d}")
    data = (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode()
    _write_bytes_per_node(os.path.join(it_dir, _SCALING_MARKER), data, "adapter scaling marker")


def write_adapter_checkpoint_marker(
    save_dir: str,
    iteration: int,
    model,
    *,
    optimizer_stats: dict[str, int] | None,
    saved_optimizer: bool,
    saved_rng: bool,
) -> None:
    """Record exactly which resume components landed in an adapter checkpoint.

    Every rank writes the same small marker to its node-local ``/nfs``.  Resume
    can then distinguish a weights-only snapshot from one that also contains
    Muon momentum/fp32 masters and RNG state instead of relying on missing-key
    errors deep in Megatron's loader.
    """
    trainable = _named_trainable_params(model)
    marker = {
        "schema_version": 2,
        "saved_optimizer": bool(saved_optimizer),
        "saved_rng": bool(saved_rng),
        "lora_param_tensors": len(trainable),
        "lora_param_numel": sum(int(param.numel()) for _name, param in trainable.values()),
        "optimizer_param_tensors": int((optimizer_stats or {}).get("optimizer_param_tensors", 0)),
        "optimizer_param_numel": int((optimizer_stats or {}).get("optimizer_param_numel", 0)),
        "fp32_master_tensors": int((optimizer_stats or {}).get("fp32_master_tensors", 0)),
        "fp32_master_numel": int((optimizer_stats or {}).get("fp32_master_numel", 0)),
        "fp32_master_bytes": int((optimizer_stats or {}).get("fp32_master_bytes", 0)),
        "optimizer_state_tensors": int((optimizer_stats or {}).get("optimizer_state_tensors", 0)),
        "optimizer_state_numel": int((optimizer_stats or {}).get("optimizer_state_numel", 0)),
        "optimizer_state_bytes": int((optimizer_stats or {}).get("optimizer_state_bytes", 0)),
    }
    it_dir = os.path.join(save_dir, f"iter_{iteration:07d}")
    data = (json.dumps(marker, indent=2, sort_keys=True) + "\n").encode()
    _write_bytes_per_node(os.path.join(it_dir, _CHECKPOINT_MARKER), data, "adapter component marker")


def finalize_adapter_checkpoint(
    final_save_dir: str,
    staging_dir: str,
    iteration: int,
    model,
    *,
    optimizer_stats: dict[str, int] | None,
    saved_optimizer: bool,
    saved_rng: bool,
    max_node_bytes: int = _DEFAULT_MAX_NODE_CHECKPOINT_BYTES,
    rslora: bool = False,
) -> None:
    """Replicate, validate, and atomically publish one staged adapter save.

    The function is intentionally identical for sync and async saves.  In sync
    mode the caller invokes it immediately after ``save_checkpoint``.  In async
    mode it is appended to Megatron's ``AsyncRequest.finalize_fns``, so it cannot
    observe partial distcp shards and any failure propagates through
    ``maybe_finalize_async_save`` on every training rank.
    """
    replicate_ckpt_metadata_per_node(staging_dir, iteration)
    replicate_distcp_shards_per_node(staging_dir, iteration)
    # Record the operative LoRA scale and exactly which stateful components
    # landed before the checkpoint becomes visible to a future resume.
    write_adapter_scaling_marker(staging_dir, iteration, model, rslora=rslora)
    write_adapter_checkpoint_marker(
        staging_dir,
        iteration,
        model,
        optimizer_stats=optimizer_stats,
        saved_optimizer=saved_optimizer,
        saved_rng=saved_rng,
    )
    validate_adapter_checkpoint_size(staging_dir, iteration, max_node_bytes)
    commit_adapter_checkpoint_staging(final_save_dir, staging_dir, iteration)
    write_latest_marker_per_node(final_save_dir, iteration)

    import torch

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.info(
            "adapter checkpoint COMMITTED: iteration=%s save_dir=%s",
            iteration,
            final_save_dir,
        )


def validate_adapter_checkpoint_components(
    ckpt_path: str,
    *,
    load_optimizer: bool,
    load_rng: bool,
) -> dict | None:
    """Fail clearly if a requested resume component was not saved."""
    it_dir = _resolve_iter_dir(ckpt_path)
    marker_path = os.path.join(it_dir, _CHECKPOINT_MARKER) if it_dir else None
    if marker_path is None or not os.path.isfile(marker_path):
        if load_optimizer or load_rng:
            raise RuntimeError(
                f"adapter resume requested optimizer/RNG state but {marker_path or _CHECKPOINT_MARKER} "
                "is missing; this legacy checkpoint cannot prove those components were saved"
            )
        logger.warning(
            "adapter resume: no %s under %s; treating it as a legacy weights-only checkpoint",
            _CHECKPOINT_MARKER,
            ckpt_path,
        )
        return None
    with open(marker_path) as f:
        marker = json.load(f)
    if load_optimizer and not marker.get("saved_optimizer", False):
        raise RuntimeError(f"adapter resume requested optimizer state, but checkpoint is weights-only: {marker_path}")
    if load_rng and not marker.get("saved_rng", False):
        raise RuntimeError(f"adapter resume requested RNG state, but checkpoint did not save RNG: {marker_path}")
    if marker.get("saved_optimizer", False):
        lora_numel = int(marker.get("lora_param_numel", -1))
        optimizer_numel = int(marker.get("optimizer_param_numel", -2))
        if lora_numel <= 0 or optimizer_numel != lora_numel:
            raise RuntimeError(
                f"invalid LoRA optimizer marker {marker_path}: "
                f"lora_param_numel={lora_numel} optimizer_param_numel={optimizer_numel}"
            )
        if int(marker.get("schema_version", 1)) >= 2:
            fp32_master_numel = int(marker.get("fp32_master_numel", -3))
            if fp32_master_numel != lora_numel:
                raise RuntimeError(
                    f"invalid LoRA fp32-master marker {marker_path}: "
                    f"lora_param_numel={lora_numel} fp32_master_numel={fp32_master_numel}"
                )
    logger.info(
        "adapter resume component marker verified: optimizer=%s rng=%s (%s)",
        marker.get("saved_optimizer", False),
        marker.get("saved_rng", False),
        marker_path,
    )
    return marker


def _resolve_iter_dir(ckpt_path: str) -> str | None:
    """The iter_XXXXXXX dir for an adapter checkpoint path (the path itself if it
    already is one, else via latest_checkpointed_iteration.txt)."""
    base = os.path.basename(os.path.normpath(ckpt_path))
    if re.fullmatch(r"iter_\d{7}", base):
        return ckpt_path
    marker = os.path.join(ckpt_path, "latest_checkpointed_iteration.txt")
    try:
        with open(marker) as f:
            iteration = int(f.read().strip())
    except (OSError, ValueError):
        return None
    return os.path.join(ckpt_path, f"iter_{iteration:07d}")


def validate_adapter_scaling_marker(ckpt_path: str, model, *, rslora: bool = False) -> None:
    """Fail loud when the checkpoint's recorded LoRA scaling differs from the live
    model's (rebuilt from the registered LoRA CLI arguments).

    A missing marker only warns (checkpoints written before the marker existed);
    a present-but-mismatched marker raises — resuming rsLoRA-trained adapters
    under classic scaling (or vice versa) silently rescales every delta by
    sqrt(r) and corrupts the run.
    """
    it_dir = _resolve_iter_dir(ckpt_path)
    marker_path = os.path.join(it_dir, _SCALING_MARKER) if it_dir else None
    if marker_path is None or not os.path.isfile(marker_path):
        logger.warning(
            "adapter resume: no %s found under %s — cannot verify the checkpoint was "
            "trained with the current LoRA scaling. Proceeding; ensure the LoRA "
            "arguments match the original run.",
            _SCALING_MARKER,
            ckpt_path,
        )
        return
    with open(marker_path) as f:
        saved = json.load(f)
    live = collect_adapter_scaling(model, rslora=rslora)
    if live is None:
        raise RuntimeError("adapter resume: checkpoint has a scaling marker but the live model has no LoRA adapters")
    mismatches = {
        key: (saved.get(key), live[key]) for key in ("dim", "alpha", "scale", "rslora") if saved.get(key) != live[key]
    }
    if mismatches:
        raise RuntimeError(
            f"adapter resume scaling mismatch vs {marker_path}: "
            + ", ".join(
                f"{key}: ckpt={saved_value!r} live={live_value!r}"
                for key, (saved_value, live_value) in mismatches.items()
            )
            + f" (ckpt rslora={saved.get('rslora')!r}, live rslora={live['rslora']!r}). "
            "Set --lora-dim/--lora-alpha/--lora-rslora to the values the checkpoint "
            "was trained with; resuming with a different scaling would "
            "silently rescale every adapter delta."
        )
    logger.info(
        "adapter resume: scaling marker verified (dim=%s alpha=%s scale=%s rslora=%s)",
        live["dim"],
        live["alpha"],
        live["scale"],
        live["rslora"],
    )


def _norm(key) -> str:
    if isinstance(key, tuple):
        key = key[0]
    return _MODULE_PREFIX.sub("", str(key))


def trainable_param_keys(model) -> set[str]:
    keys: set[str] = set()
    for chunk in model:
        for name, param in chunk.named_parameters():
            if param.requires_grad:
                keys.add(_norm(name))
    return keys


def _is_adapter_key(norm_key: str, trainable: set[str]) -> bool:
    if "_extra_state" in norm_key:
        return False
    if norm_key in trainable:
        return True
    return ".adapter." in norm_key or norm_key.endswith(".adapters")


def _sharded_bytes(value) -> int:
    """Local byte size of a Megatron ShardedTensor/ShardedObject entry (0 if unknown)."""
    data = getattr(value, "data", None)
    if data is not None and hasattr(data, "numel"):
        try:
            return data.numel() * data.element_size()
        except Exception:
            return 0
    return 0


def _log_sizes(label: str, sd: dict) -> int:
    total = sum(_sharded_bytes(v) for v in sd.values())
    logger.info("adapter-ckpt[%s]: %d keys, %.3f GB local", label, len(sd), total / 1e9)
    return total


@contextmanager
def adapter_only_model_save(model):
    """Filter each model chunk's ``sharded_state_dict`` to adapter params only.

    Always logs full-vs-kept byte sizes so the adapter-only shrink is visible,
    and asserts a small non-empty subset was kept before writing.
    """
    # save_checkpoint calls unwrap_model(model) before generate_state_dict, so we
    # must wrap the UNWRAPPED module's sharded_state_dict (wrapping the DDP wrapper
    # is bypassed and the full base gets written).
    modules = [unwrap_model(chunk) for chunk in model]
    trainable = trainable_param_keys(modules)
    if not trainable:
        raise RuntimeError("adapter-only save: no trainable params (LoRA not applied / base not frozen)")
    originals = []
    for chunk in modules:
        orig = chunk.sharded_state_dict

        def make_filtered(orig_fn):
            def filtered(*a, **kw):
                sd = orig_fn(*a, **kw)
                kept = {k: v for k, v in sd.items() if _is_adapter_key(_norm(k), trainable)}
                # Always log sizes so the shrink (or lack of it) is visible; the
                # debug flag only controls whether we abort before writing.
                full_bytes = _log_sizes("full", sd)
                kept_bytes = _log_sizes("kept", kept)
                logger.info(
                    "adapter-ckpt: kept %d/%d keys, %.2f MB of %.2f GB (%.4f%%)",
                    len(kept),
                    len(sd),
                    kept_bytes / 1e6,
                    full_bytes / 1e9,
                    100.0 * kept_bytes / max(full_bytes, 1),
                )
                assert (
                    0 < len(kept) < len(sd)
                ), f"adapter-only filter kept {len(kept)}/{len(sd)} keys; refusing to write a corrupt checkpoint"
                return kept

            return filtered

        chunk.sharded_state_dict = make_filtered(orig)
        originals.append((chunk, orig))

    logger.info("adapter-only save: %d trainable adapter param tensors, base frozen/skipped", len(trainable))
    try:
        yield trainable
    finally:
        for chunk, orig in originals:
            chunk.sharded_state_dict = orig


@contextmanager
def adapter_only_model_load(model):
    """Filter each model chunk's ``sharded_state_dict`` to adapter params only so a
    resume load requests only adapter keys (base already loaded cold from --load)."""
    modules = [unwrap_model(chunk) for chunk in model]
    trainable = trainable_param_keys(modules)
    if not trainable:
        raise RuntimeError("adapter-only resume: no trainable params found")

    originals = []
    for chunk in modules:
        orig = chunk.sharded_state_dict

        def make_filtered(orig_fn):
            def filtered(*a, **kw):
                sd = orig_fn(*a, **kw)
                kept = {k: v for k, v in sd.items() if _is_adapter_key(_norm(k), trainable)}
                assert kept, "adapter-only load filtered out every key; key-space mismatch"
                return kept

            return filtered

        chunk.sharded_state_dict = make_filtered(orig)
        originals.append((chunk, orig))
    try:
        yield trainable
    finally:
        for chunk, orig in originals:
            chunk.sharded_state_dict = orig
