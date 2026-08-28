"""Adapter-only "old actor" for V4 LoRA RL (the cheap ``--keep-old-actor``).

Upstream ``--keep-old-actor`` keeps a SECOND full model instance and switches to
it for the behavioral log-prob (scoring) forward, so the PPO ratio and TIS use the
policy that actually sampled the batch instead of the just-trained policy. For our
284B fp8 frozen-base + ~45M bf16 LoRA adapter, a second full model is absurd: the
"old actor" differs from the live model ONLY by the trainable adapter tensors
(``linear_in``/``linear_out`` = PEFT ``lora_A``/``lora_B``). This module snapshots
just those tensors (~90MB), swaps them into the LIVE model for the scoring forward,
and restores the live adapter afterwards. The frozen base is never touched.

Version bookkeeping (async lag-1, ``update_weights_interval == 1``)
------------------------------------------------------------------
Let ``θ_n`` be the adapter after ``n`` gradient steps. The engine serves version
``v`` == ``θ_{v-1}`` (the first, pre-loop, ``update_weights`` push makes ``θ_0`` ==
version 1). A batch generated under engine version ``v`` was sampled from ``θ_{v-1}``.

In ``train_async.py`` iteration ``k`` trains the batch whose ``gen_weight_version``
== ``k`` (== 1 for ``k in {0, 1}``); at entry the live adapter is ``θ_k`` (``k``
gradient steps done) while the behavioral policy is ``θ_{k-1}`` — exactly one step
behind. So a SINGLE snapshot suffices: at the start of the first ``train_actor``
seed it from the live adapter; each iteration score with it, then AFTER restoring
the live adapter and BEFORE the gradient step refresh it to the live adapter tagged
with the current ``weight_updater.weight_version``. The refreshed snapshot (``θ_k``,
version ``k+1``) is precisely the behavioral policy of the next batch (gen version
``k+1``). ``score_with_snapshot`` asserts the batch's ``gen_weight_version`` matches
the snapshot version so a partial-rollout / buffered carry-over can never be scored
with the wrong behavior weights.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager

import torch

from slime.backends.megatron_utils.update_weight.lora_adapter_sync import is_adapter_param_name

logger = logging.getLogger(__name__)


def should_recompute_old_actor_log_probs(args) -> bool:
    """Return whether the pre-train old-policy scoring forward is required.

    The debug-force flag is only for matched frozen-replay experiments: it
    makes a rollout-denominator arm execute the same frozen-old-actor forward
    as a recompute-denominator arm without enabling TIS/mismatch loss logic.
    """

    return bool(
        not getattr(args, "use_rollout_logprobs", False)
        or getattr(args, "get_mismatch_metrics", False)
        or getattr(args, "debug_force_old_actor_logprob_recompute", False)
    )


def enumerate_adapter_params(model) -> list[tuple[str, torch.nn.Parameter]]:
    """Return ``(local_name, Parameter)`` for every trainable LoRA adapter param.

    Enumerates the LIVE parameter objects (not gathered copies) so ``copy_`` swaps
    hit the tensors the forward/optimizer use. Names are per-vp-stage-local and only
    need to be stable within this process (snapshot and restore run on the same rank
    against the same objects), so no PP/TP global renaming is required.
    """
    out: list[tuple[str, torch.nn.Parameter]] = []
    for vp_stage, module in enumerate(model):
        for name, param in module.named_parameters():
            if getattr(param, "requires_grad", False) and is_adapter_param_name(name):
                out.append((f"vp{vp_stage}.{name}", param))
    return out


class LoRAOldActorSnapshot:
    """A single version-tagged snapshot of the trainable LoRA adapter tensors.

    Not a full model: only the ``requires_grad`` adapter params are held. The
    snapshot lives on ``device`` (``"cpu"`` pinned by default — matches the
    weights_backuper, avoids competing with the training model's GPU budget at long
    context, and is unaffected by ``offload_train``'s ``torch_memory_saver`` pause).
    """

    def __init__(self, model, *, device: str = "cpu", pin_memory: bool = True) -> None:
        self._params = enumerate_adapter_params(model)
        self._device = torch.device(device)
        self._pin_memory = pin_memory and self._device.type == "cpu"
        # Persistent behavioral snapshot (θ_{k-1}) + its version tag.
        self._snapshot: dict[str, torch.Tensor] | None = None
        self._snapshot_version: int | None = None
        # Transient buffer holding the live adapter (θ_k) across the scoring swap so
        # the restore is bit-exact regardless of what the snapshot holds.
        self._scratch: dict[str, torch.Tensor] | None = None

    # -- introspection -------------------------------------------------------
    def has_params(self) -> bool:
        return len(self._params) > 0

    @property
    def num_tensors(self) -> int:
        return len(self._params)

    @property
    def num_bytes(self) -> int:
        """Bytes of ONE adapter copy. Steady-state footprint is twice this
        (snapshot + swap scratch), pinned on CPU by default."""
        return sum(p.numel() * p.element_size() for _, p in self._params)

    @property
    def version(self) -> int | None:
        return self._snapshot_version

    # -- buffer management ---------------------------------------------------
    def _alloc_like(self) -> dict[str, torch.Tensor]:
        buffers: dict[str, torch.Tensor] = {}
        for name, param in self._params:
            if self._pin_memory:
                # Matches the weights_backuper allocation (pinned host staging).
                buffers[name] = torch.empty_like(param, device=self._device, pin_memory=True)
            else:
                buffers[name] = torch.empty_like(param, device=self._device)
        return buffers

    def _needs_sync(self) -> bool:
        # A non_blocking copy that crosses the host/device boundary must be synced
        # before the destination is read or reused.
        return self._device.type == "cuda" or any(p.is_cuda for _, p in self._params)

    @torch.no_grad()
    def _snapshot_live_into(self, dst: dict[str, torch.Tensor]) -> None:
        for name, param in self._params:
            dst[name].copy_(param.detach(), non_blocking=True)
        if self._needs_sync():
            torch.cuda.synchronize()

    @torch.no_grad()
    def _load_into_live(self, src: dict[str, torch.Tensor]) -> None:
        for name, param in self._params:
            param.data.copy_(src[name], non_blocking=True)
        if self._needs_sync():
            torch.cuda.synchronize()

    # -- public API ----------------------------------------------------------
    @torch.no_grad()
    def refresh(self, version: int) -> None:
        """Snapshot the current live adapter and tag it ``version``.

        Called (a) once to seed the behavioral snapshot for the first batch and
        (b) after every scoring pass, before the gradient step, so the snapshot
        tracks the policy that produced the just-pushed engine weights.
        """
        if self._snapshot is None:
            self._snapshot = self._alloc_like()
        self._snapshot_live_into(self._snapshot)
        self._snapshot_version = int(version)

    @contextmanager
    def score_with_snapshot(self, expected_version: int | None = None):
        """Swap the behavioral adapter into the live model for a scoring forward.

        Saves the live adapter, loads the snapshot, yields (caller runs the
        forward), and ALWAYS restores the live adapter (try/finally) so a crash
        mid-scoring can never leave the training path on the old adapter. Only
        ``.data`` is mutated in place — the ``Parameter`` objects (and thus the
        optimizer state keyed on them) are untouched.

        ``expected_version`` (a batch's ``gen_weight_version``) is asserted equal to
        the snapshot version so a stale/buffered batch fails loudly instead of being
        scored with the wrong behavior weights.
        """
        assert self._snapshot is not None, "refresh() must seed the snapshot before scoring"
        if expected_version is not None and int(expected_version) != self._snapshot_version:
            raise RuntimeError(
                f"LoRA old-actor version mismatch: snapshot holds v{self._snapshot_version} "
                f"but the batch was generated under v{int(expected_version)}. Scoring with the "
                "wrong behavior weights would corrupt the importance ratio; aborting."
            )
        if self._scratch is None:
            self._scratch = self._alloc_like()
        # Save live (θ_k) -> scratch, then load snapshot (θ_{k-1}) -> live.
        self._snapshot_live_into(self._scratch)
        self._load_into_live(self._snapshot)
        try:
            yield
        finally:
            # Restore live adapter (θ_k), bit-exact, even on exception.
            self._load_into_live(self._scratch)


def maybe_refresh_lora_old_actor_snapshot(
    snapshot: LoRAOldActorSnapshot,
    *,
    version: int,
    freeze_after_seed: bool,
) -> bool:
    """Seed or advance an adapter old-actor snapshot.

    Returns ``True`` when a copy was taken.  ``freeze_after_seed`` is the
    explicit fixed-debug-replay behavior: the first call captures the initial
    actor, while every later call is a no-op so repeated optimizer steps keep
    scoring against the one behavior policy that produced the frozen dump.

    This helper is deliberately policy-free; CLI validation restricts the
    freeze mode to ``debug_train_only + load_debug_rollout_data + keep_old_actor``.
    """
    if snapshot.version is None:
        snapshot.refresh(version=version)
        return True
    if freeze_after_seed:
        return False
    snapshot.refresh(version=version)
    return True


def resolve_batch_gen_version(gen_versions, snapshot_version, *, gloo_group=None):
    """Collective-safe FAIL-CLOSED check that a batch's ``gen_weight_version``s
    match the snapshot.

    ``gen_versions`` is this rank's per-sample list. This is only ever called when
    keep-old-actor is active, where every scored sample MUST carry provenance: a
    missing list, ``None`` entries (unstamped/debug samples), mixed versions, or a
    version disagreeing with the snapshot are all hard errors. Scoring an
    unstamped sample under the snapshot would silently corrupt the importance
    ratio — replaying debug data requires disabling --keep-old-actor.

    The check is done under an all-reduce over ``gloo_group`` so that if ANY rank's
    batch is off (e.g. a buffered carry-over lands on one DP rank only), EVERY rank
    raises together — a lone rank raising into a collective forward would otherwise
    hang the job instead of failing loudly. When distributed is unavailable (unit
    tests), it falls back to a local check.
    """
    versions = list(gen_versions or [])
    unstamped = sum(1 for v in versions if v is None)
    present = sorted({int(v) for v in versions if v is not None})
    local_bad = 1 if (not versions or unstamped or len(present) != 1 or present[0] != int(snapshot_version)) else 0

    import torch.distributed as dist

    global_bad = local_bad
    if dist.is_available() and dist.is_initialized():
        flag = torch.tensor([local_bad], dtype=torch.int64)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=gloo_group)
        global_bad = int(flag.item())

    if global_bad:
        raise RuntimeError(
            f"LoRA old-actor version check failed (fail-closed): snapshot holds "
            f"v{int(snapshot_version)} but this rank's batch carries gen_weight_version(s) "
            f"{present or '<none>'} with {unstamped}/{len(versions)} unstamped sample(s). "
            "Under keep-old-actor every scored sample must carry the snapshot's behavioral "
            "version; missing/mixed provenance means a buffered, debug-loaded, or partial "
            "rollout would be scored with the wrong behavior weights. Aborting instead of "
            "silently corrupting the importance ratio (disable --keep-old-actor to replay "
            "unstamped data)."
        )
    return present[0]
