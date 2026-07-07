"""Adapter-only (LoRA) checkpointing for V4-Flash — Bridge-style state-dict filter.

A full V4 ``--save`` writes ~259GB (frozen 284B base + DistributedOptimizer fp32
master params for the whole grad buffer). For LoRA only the tiny adapter params
change; the base reloads cold from the converted ``torch_dist`` ``--load``.

Megatron-Bridge supports adapter-only save via ``adapter_key_filter`` /
``apply_peft_adapter_filter_to_state_dict`` (peft/base.py) — but slime saves
through ``megatron.training.checkpointing.save_checkpoint`` (the full-model path),
so we replicate the filter here: wrap each model chunk's ``sharded_state_dict`` to
keep only adapter params on save, and to request only adapter params on resume.

V4 uses ``LinearAdapter`` (LoRA params ``linear_in.weight`` / ``linear_out.weight``,
base frozen at ``.weight``), so the filter is keyed on ``requires_grad``, not on a
sizes and aborts before the write, to verify the filter actually shrinks the save.
"""

import logging
import os
import re
from contextlib import contextmanager

try:
    from megatron.core.pipeline_parallel.utils import unwrap_model
except ImportError:
    from megatron.core.utils import unwrap_model

logger = logging.getLogger(__name__)

_MODULE_PREFIX = re.compile(r"^(?:module\.)+")


def adapter_only_ckpt_enabled(args) -> bool:
    return os.environ.get("V4_LORA_ADAPTER_ONLY_CKPT", "0") == "1"


def replicate_ckpt_metadata_per_node(save_dir: str, iteration: int) -> None:
    """Broadcast the torch_dist metadata files to every node.

    torch_dist writes ``.metadata`` / ``common.pt`` / ``metadata.json`` only from
    global rank 0, but /nfs is per-node-local, so other nodes' checkpoint dirs have
    only their data shards and the resume's format detection + load fail
    ("unknown checkpoint format"). Broadcast the (small) metadata bytes from rank 0
    and write them on any node that lacks them.
    """
    import torch

    if not torch.distributed.is_initialized():
        return
    it_dir = os.path.join(save_dir, f"iter_{iteration:07d}")
    files = [".metadata", "common.pt", "metadata.json"]
    rank = torch.distributed.get_rank()
    payload = None
    if rank == 0:
        payload = {}
        for fn in files:
            p = os.path.join(it_dir, fn)
            if os.path.isfile(p):
                with open(p, "rb") as fh:
                    payload[fn] = fh.read()
    box = [payload]
    torch.distributed.broadcast_object_list(box, src=0)
    payload = box[0]
    if payload and rank != 0:
        os.makedirs(it_dir, exist_ok=True)
        for fn, data in payload.items():
            p = os.path.join(it_dir, fn)
            if not os.path.exists(p):
                with open(p, "wb") as fh:
                    fh.write(data)
    torch.distributed.barrier()


def write_latest_marker_per_node(save_dir: str, iteration: int) -> None:
    """Write latest_checkpointed_iteration.txt on THIS node.

    Megatron writes it only from global rank 0, but /nfs is per-node-local, so
    other nodes' copies of the checkpoint dir lack it and the resume's
    ``_is_megatron_checkpoint`` detection routes them to the HF loader. Every rank
    writes the same tiny value to its node-local dir (last-writer-wins, harmless).
    """
    try:
        with open(os.path.join(save_dir, "latest_checkpointed_iteration.txt"), "w") as f:
            f.write(str(iteration))
    except OSError as exc:  # pragma: no cover
        logger.warning("adapter-ckpt: could not write latest marker in %s: %s", save_dir, exc)


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
