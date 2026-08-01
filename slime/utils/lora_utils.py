"""Lightweight LoRA-serving helpers shared by the (megatron) weight-sync path and
the (megatron-free) sglang rollout path.

These three symbols intentionally live here — not in
``slime.backends.megatron_utils.update_weight.lora_adapter_sync`` — because the
rollout process must decide whether to attach a ``lora_path`` to each ``/generate``
payload without importing megatron / the HF converter. ``lora_adapter_sync``
re-exports them for backward compatibility.
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Sequence

# Base name / prefix for the served LoRA adapter. The sync uses QeRL-style
# ALTERNATING adapter names derived from this prefix (``slime_lora_0`` /
# ``slime_lora_1``, see ``lora_adapter_name``): each sync loads the NEW name while
# the OLD one is still resident, then unloads the OLD one. This never
# unload-then-reloads a single mem-pool slot the captured cuda graph references
# (the same-slot reload triggers cudaErrorIllegalAddress on the first forward),
# and it keeps the previous adapter live for any in-flight request until the new
# one is switched in. Requires ``--sglang-max-loras-per-batch >= 2`` (two slots
# resident during the swap). The rollout learns the ACTIVE name from the engine
# each step rather than assuming a constant, so ``lora_path`` always tracks it.
SLIME_LORA_ADAPTER_NAME = "slime_lora"

# Number of distinct adapter names cycled through. Two is the minimum for a
# load-new-before-unload-old double buffer; it also bounds the sglang-side name/id
# registry (unlike a monotonically-growing ``slime_lora_{step}`` scheme).
_LORA_NUM_ALTERNATING_NAMES = 2


def lora_plus_lambda(args: Namespace) -> float | None:
    """Validate the configured LoRA+ learning-rate ratio ``eta_B = lambda * eta_A``.

    Returns ``None`` when LoRA+ is OFF: unset, empty, or exactly 1.0 (a ratio of 1
    is the vanilla single-LR setup, so the optimizer wiring stays byte-identical).
    Malformed / non-positive values raise immediately (a silently-dropped LR knob
    was this project's past "RL not learning" failure mode — fail loud instead).

    The megatron side (``slime.backends.megatron_utils.model``) consumes this to
    build a separate optimizer param group for the LoRA B matrices
    (``linear_out.weight``) with ``max_lr/min_lr`` scaled by lambda. Kept here
    (megatron-free) so tests and the launcher wiring can import it lightly.
    """
    raw = getattr(args, "lora_plus_lambda", None)
    if raw is None:
        return None
    try:
        lam = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"--lora-plus-lambda={raw!r} is not a float") from exc
    if lam <= 0:
        raise ValueError(f"--lora-plus-lambda must be > 0, got {raw!r}")
    if lam == 1.0:
        return None
    return lam


def use_lora_weight_sync(args: Namespace) -> bool:
    """True when the LoRA-adapter sync path is requested by CLI.

    Default OFF: full/merge weight sync is unchanged and byte-identical when this
    returns False.
    """
    return bool(getattr(args, "use_lora_weight_sync", False))


def lora_adapter_name(version: int) -> str:
    """Alternating adapter name for sync ``version`` (the updater's weight_version).

    ``version`` and ``version + 1`` always map to DIFFERENT names, so a sync never
    loads a name that is still resident from the immediately-preceding sync — the
    precondition for the load-new-before-unload-old double buffer. ``version`` and
    ``version + 2`` share a name, but by then that name's earlier occupant has been
    unloaded (a full sync ago) and its mem-pool slot freed.
    """
    return f"{SLIME_LORA_ADAPTER_NAME}_{version % _LORA_NUM_ALTERNATING_NAMES}"


def all_alternating_lora_names() -> set[str]:
    """Every adapter name the alternating scheme can produce. Used to clear stale
    adapters a prior run may have left resident on a persistent/external engine
    before the first sync (otherwise the first load can collide with a resident
    name and the mem-pool may have no free slot)."""
    return {lora_adapter_name(v) for v in range(_LORA_NUM_ALTERNATING_NAMES)}


def raise_on_failed_lora_load(results: Sequence[object], adapter_name: str) -> None:
    """Raise if any engine's ``load_lora_adapter_from_tensors`` result reports
    failure. sglang can return HTTP 200 with ``{"success": False}``; the caller
    must NOT proceed to unload the old (still-serving) adapter on a failed load,
    or engines would be left serving base-only / a stale adapter. ``None`` results
    (non-leader ranks that no-op) are treated as success."""
    failed = [r for r in results if isinstance(r, dict) and not r.get("success", True)]
    if failed:
        raise RuntimeError(
            f"LoRA adapter load {adapter_name!r} failed on {len(failed)}/{len(results)} "
            f"engine(s): {failed}. Not unloading the previous adapter (old adapter "
            f"stays live); aborting the sync."
        )


def plan_lora_swap(new_name: str, prev_name: str | None) -> list[tuple[str, str]]:
    """Ordered load/unload plan for one alternating adapter swap.

    Returns ``[("load", new_name), ("unload", prev_name)?]`` — LOAD the new adapter
    FIRST (into the free second mem-pool slot, while the old adapter is still
    resident and referenced by the captured cuda graph), THEN unload the old one.
    The unload step is omitted on the very first sync (``prev_name is None``) and if
    the previous name somehow equals the new one (defensive; alternation guarantees
    they differ).
    """
    plan: list[tuple[str, str]] = [("load", new_name)]
    if prev_name is not None and prev_name != new_name:
        plan.append(("unload", prev_name))
    return plan


def rollout_lora_path(args: Namespace, active_name: str | None = None) -> str | None:
    """Adapter name to put in a rollout ``/generate`` payload's ``lora_path``.

    ``--rollout-lora-name`` selects a statically preloaded adapter (for example an
    evaluation checkpoint supplied through ``--sglang-lora-paths NAME=PATH``).
    Otherwise, returns ``active_name`` (the adapter the engine currently serves,
    learned from the engine each rollout step) when dynamic LoRA-adapter sync is
    active. Returns ``None`` for base-only / non-LoRA serving. ``None`` is also
    returned when dynamic serving is on but no adapter has been loaded yet: the base
    model is served, which is correct because the initial adapter is zero-init
    (delta ``B @ A == 0``)."""
    static_name = getattr(args, "rollout_lora_name", None)
    if static_name:
        return static_name
    if not use_lora_weight_sync(args):
        return None
    return active_name
