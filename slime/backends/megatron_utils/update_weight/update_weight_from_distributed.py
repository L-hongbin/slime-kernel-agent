from __future__ import annotations

import logging
import re
import socket
import time
from argparse import Namespace
from collections.abc import Callable, Iterator, Mapping, Sequence

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle
from tqdm import tqdm

from slime.utils import accelerator
from slime.utils.distributed_utils import get_gloo_group, init_process_group
from slime.utils.http_utils import _wrap_ipv6

from ..megatron_to_hf import convert_to_hf
from ..sglang import DeltaSpec
from .common import _maybe_v4_global_name, all_gather_param, named_params_and_buffers
from .lora_adapter_sync import (
    all_alternating_lora_names,
    build_lora_adapter_state_dict,
    is_adapter_param_name,
    lora_adapter_name,
    plan_lora_swap,
    raise_on_failed_lora_load,
    use_lora_weight_sync,
)

logger = logging.getLogger(__name__)

_V4_MODEL_NAME_MARKERS = ("deepseekv4", "deepseek_v4")
_LORA_IN_SUFFIX = ".linear_in.weight"
_LORA_OUT_SUFFIX = ".linear_out.weight"
_WEIGHT_SUFFIX = ".weight"
_V4_SGLANG_COMPRESSOR_WEIGHT = re.compile(r"^layers\.\d+\.attn\.(?:indexer\.)?compressor\.(wkv|wgate)\.weight$")
_V4_SGLANG_WQKV_A_WEIGHT = re.compile(r"^layers\.\d+\.attn\.(wq_a|wkv)\.(weight|weight_scale_inv)$")


def _is_deepseekv4_model_name(model_name: str) -> bool:
    return any(marker in model_name for marker in _V4_MODEL_NAME_MARKERS)


def _lora_base_weight_name(name: str) -> str | None:
    if name.endswith(_LORA_IN_SUFFIX):
        return name[: -len(_LORA_IN_SUFFIX)] + _WEIGHT_SUFFIX
    if name.endswith(_LORA_OUT_SUFFIX):
        return name[: -len(_LORA_OUT_SUFFIX)] + _WEIGHT_SUFFIX
    return None


def _merge_lora_weight(
    base_weight: torch.Tensor,
    lora_in_weight: torch.Tensor,
    lora_out_weight: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if base_weight.ndim != 2 or lora_in_weight.ndim != 2 or lora_out_weight.ndim != 2:
        raise ValueError(
            "V4 LoRA merge expects 2D linear weights, got "
            f"base={tuple(base_weight.shape)} in={tuple(lora_in_weight.shape)} "
            f"out={tuple(lora_out_weight.shape)}"
        )
    if lora_in_weight.shape[1] != base_weight.shape[1] or lora_out_weight.shape[0] != base_weight.shape[0]:
        raise ValueError(
            "V4 LoRA merge shape mismatch: "
            f"base={tuple(base_weight.shape)} in={tuple(lora_in_weight.shape)} "
            f"out={tuple(lora_out_weight.shape)}"
        )
    merged = base_weight.float()
    merged = merged + lora_out_weight.float().matmul(lora_in_weight.float()) * float(scale)
    return merged.to(dtype=base_weight.dtype)


def _v4_sglang_compressor_pair_key(name: str) -> tuple[str, str] | None:
    match = _V4_SGLANG_COMPRESSOR_WEIGHT.fullmatch(name)
    if match is None:
        return None
    return name.rsplit(".", 2)[0], match.group(1)


def _v4_incomplete_sglang_compressor_pairs(named_tensors: Sequence[tuple[str, torch.Tensor]]) -> dict[str, set[str]]:
    pairs: dict[str, set[str]] = {}
    for name, _ in named_tensors:
        parsed = _v4_sglang_compressor_pair_key(name)
        if parsed is None:
            continue
        key, side = parsed
        pairs.setdefault(key, set()).add(side)
    return {key: sides for key, sides in pairs.items() if sides != {"wkv", "wgate"}}


def _v4_sglang_wqkv_a_pair_key(name: str) -> tuple[str, str] | None:
    match = _V4_SGLANG_WQKV_A_WEIGHT.fullmatch(name)
    if match is None:
        return None
    side = "q" if match.group(1) == "wq_a" else "kv"
    key = name.replace(".wq_a.", ".wqkv_a.").replace(".wkv.", ".wqkv_a.")
    return key, side


def _v4_incomplete_sglang_wqkv_a_pairs(named_tensors: Sequence[tuple[str, torch.Tensor]]) -> dict[str, set[str]]:
    pairs: dict[str, set[str]] = {}
    for name, _ in named_tensors:
        parsed = _v4_sglang_wqkv_a_pair_key(name)
        if parsed is None:
            continue
        key, side = parsed
        pairs.setdefault(key, set()).add(side)
    return {key: sides for key, sides in pairs.items() if sides != {"q", "kv"}}


def _v4_incomplete_sglang_loader_pairs(named_tensors: Sequence[tuple[str, torch.Tensor]]) -> dict[str, set[str]]:
    incomplete = _v4_incomplete_sglang_compressor_pairs(named_tensors)
    incomplete.update(_v4_incomplete_sglang_wqkv_a_pairs(named_tensors))
    return incomplete


def _build_v4_lora_base_scales(args: Namespace, model: Sequence[torch.nn.Module]) -> dict[str, float]:
    scales: dict[str, float] = {}
    for model_module in model:
        for module_name, module in model_module.named_modules():
            if not all(hasattr(module, attr) for attr in ("linear_in", "linear_out", "scale")):
                continue
            probe = getattr(module, "weight", None)
            if probe is None:
                continue
            base_name = f"{module_name}.weight" if module_name else "weight"
            if not base_name.startswith("module.module."):
                base_name = "module." + base_name
            global_base_name = _maybe_v4_global_name(args, model_module, base_name, probe, expert_offset=None)
            scales[global_base_name or base_name] = float(module.scale)
    return scales


class UpdateWeightFromDistributed:
    """
    Update distributed engines through a device process group. Each PP rank: group "slime-pp_{pp_rank}",
    only DP=TP=0 transfers. Non-expert (TP) and expert (EP) params separate.
    Subclasses override ``_send_weights`` / ``_on_chunk`` to inject per-mode behaviour.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        """
        Initialize. Groups created in connect_rollout_engines.
        """
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self._model_update_groups = None
        self.update_weight_metrics: dict[str, float] = {}
        self._v4_lora_base_scales: dict[str, float] | None = None
        # Name of the LoRA adapter loaded on the engines by the previous sync (for
        # unload-then-reload; sglang forbids re-loading an existing adapter name).
        self._lora_prev_adapter_name: str | None = None

    def pop_metrics(self) -> dict[str, float]:
        """
        Return and clear ``update_weight_metrics``. Drained by the actor onto the rollout/step log.

        The metrics are produced on GLOBAL RANK 0 (the adapter gather/swap runs
        there), but wandb logging happens on the PRIMARY rank (tp0/dp0 of the
        LAST pipeline stage — a different node under PP>1). Broadcast rank 0's
        dict so the primary rank pops real values; without this the lora/*
        panels were silently empty (bit r9o 2026-07-12). Collective-safe: every
        rank calls pop_metrics via log_perf_data each step.
        """
        payload = [self.update_weight_metrics]
        dist.broadcast_object_list(payload, src=0, group=get_gloo_group())
        self.update_weight_metrics = {}
        return payload[0]

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
        engine_parallel_configs: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        """
        Bind rollout engines and create the device communication group for full-weight sync.

        Adapter-only LoRA sync sends tensors through Ray RPCs and never consumes
        ``_model_update_groups``.  Keep the native PP-source ownership metadata,
        but avoid making every rollout GPU join an otherwise-unused NCCL group.
        The legacy full-weight path retains the existing group lifecycle.
        """
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock
        self._engine_gpu_counts = engine_gpu_counts

        # For TP:
        #   1. AllGather parameters to rank 0
        #   2. Broadcast parameters from rank 0 to all sglang engines
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        )
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        if self._is_pp_src_rank:
            self._group_name = f"slime-pp_{pp_rank}"

        if self._is_pp_src_rank:
            if use_lora_weight_sync(self.args):
                if self._model_update_groups is not None:
                    disconnect_rollout_engines_from_distributed(
                        self._group_name, self._model_update_groups, self.rollout_engines
                    )
                self._model_update_groups = None
                logger.info(
                    "[%s] LoRA adapter-only sync: skipping unused native weight-update NCCL group",
                    self._group_name,
                )
                return
            if self._model_update_groups is not None:
                disconnect_rollout_engines_from_distributed(
                    self._group_name, self._model_update_groups, self.rollout_engines
                )
            self._model_update_groups = connect_rollout_engines_from_distributed(
                self.args,
                self._group_name,
                rollout_engines,
                engine_gpu_counts=engine_gpu_counts,
            )

    def disconnect_rollout_engines(self) -> None:
        if not getattr(self, "_is_pp_src_rank", False) or self._model_update_groups is None:
            return
        disconnect_rollout_engines_from_distributed(self._group_name, self._model_update_groups, self.rollout_engines)
        self._model_update_groups = None

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        Pause → flush → _send_weights → continue. Progress on PP source.
        """
        self.weight_version += 1

        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])

            # int4/fp4 pre_process
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                )
        dist.barrier(group=get_gloo_group())

        if use_lora_weight_sync(self.args):
            self._update_weights_lora_adapter()
        else:
            pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_pp_src_rank else None
            self._send_weights(pbar)

        if dist.get_rank() == 0:
            # int4/fp4 post_process
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
            self._finish_lora_adapter_swap()
        dist.barrier(group=get_gloo_group())

    def _finish_lora_adapter_swap(self) -> None:
        """Best-effort unload of the previous alternating adapter, AFTER
        generation resumed (rank0 only). Bounded by the engine-side timeout; on
        timeout the adapter stays resident and the next swap retries first."""
        pending = getattr(self, "_lora_pending_unload", None)
        if pending is None:
            return
        try:
            ray.get([engine.unload_lora_adapter.remote(lora_name=pending) for engine in self.rollout_engines])
            self._lora_pending_unload = None
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "deferred lora unload of %s failed (%s); leaving resident, will retry next swap",
                pending,
                exc,
            )

    def _send_weights(self, pbar: tqdm | None) -> None:
        """
        Non-expert (TP) pass → barrier → expert (EP) pass → barrier. Each iterator
        yields broadcast-ready chunks (bucketing happens internally); subclasses
        override ``_on_chunk`` to inject per-chunk behaviour.
        """
        for chunk_iter in (self._iter_non_expert_chunks(), self._iter_expert_chunks()):
            for hf_chunk in chunk_iter:
                self._on_chunk(hf_chunk)
                self._update_bucket_weights_from_distributed(hf_chunk, pbar=pbar)
            dist.barrier(group=get_gloo_group())

    def _on_chunk(self, hf_chunk: list[tuple[str, torch.Tensor]]) -> None:
        """
        Hook for each HF chunk in ``_send_weights`` before its broadcast. No-op by default.
        """

    def _get_v4_lora_base_scales(self) -> dict[str, float]:
        if self._v4_lora_base_scales is None:
            if _is_deepseekv4_model_name(self.model_name):
                self._v4_lora_base_scales = _build_v4_lora_base_scales(self.args, self.model)
            else:
                self._v4_lora_base_scales = {}
        return self._v4_lora_base_scales

    def _uses_v4_lora_only_sync(self) -> bool:
        return bool(self._get_v4_lora_base_scales())

    def _collect_local_adapter_named_tensors(self) -> list[tuple[str, torch.Tensor]]:
        """Gather trainable LoRA params, materializing CPU payloads only on PP sources.

        ``all_gather_param`` is still called by every rank in the same parameter
        order because TP shards require the collective.  The native
        ``_is_pp_src_rank`` ownership rule (DP-with-CP rank 0 and TP rank 0)
        selects one canonical payload owner per PP stage; all DP/CP/EP replicas
        discard the gathered GPU view instead of copying it to CPU.  Routed-expert
        adapters are rejected until this path has an EP gather.
        """
        adapter_named_tensors = [
            (name, param)
            for name, param in named_params_and_buffers(self.args, self.model)
            if getattr(param, "requires_grad", False) and is_adapter_param_name(name)
        ]
        routed_expert_names = [name for name, _param in adapter_named_tensors if ".experts." in name]
        if routed_expert_names:
            raise RuntimeError(
                "LoRA adapter-only sync does not yet support routed-expert LoRA parameters "
                "('.experts.'): their EP shards require an expert-model-parallel gather "
                "before canonical PP-source materialization. Replicated '.shared_experts.' "
                f"adapters remain supported. Found {len(routed_expert_names)}, e.g. {routed_expert_names[0]!r}."
            )

        collected: list[tuple[str, torch.Tensor]] = []
        for name, param in adapter_named_tensors:
            full = all_gather_param(name, param)
            if not self._is_pp_src_rank:
                continue
            collected.append((name, full.detach().to("cpu", copy=True)))
        return collected

    @torch.no_grad()
    def _update_weights_lora_adapter(self) -> None:
        """Ship only the trainable LoRA adapter to the engines (base + adapter served).

        Every rank joins the per-parameter shard collectives, but only the canonical
        source of each PP stage materializes a CPU payload.  PP=1 builds directly
        on global rank 0; PP>1 gathers the stage-source payloads (and empty payloads
        from all other ranks) to global rank 0.  Rank 0 then assembles a PEFT state
        dict + config and drives the existing alternating adapter hot-swap endpoint.
        """
        scales = self._get_v4_lora_base_scales()
        if not scales:
            raise RuntimeError(
                "--use-lora-weight-sync requires a V4 LoRA model (no trainable adapter "
                "modules with weight/linear_in/linear_out/scale were found)."
            )
        scale = next(iter(scales.values()))

        # Phase timers (perf/lora_swap/*): the swap sits on the async loop's
        # critical path (engine idle during it), so decompose where the ~5s goes
        # before optimizing — gather (collect+gloo), build (dedupe+state dict),
        # swap (ship to engines + load + unload).
        import time as _time

        _t0 = _time.perf_counter()
        local = self._collect_local_adapter_named_tensors()

        gloo_group = get_gloo_group()
        global_rank = dist.get_rank()
        pp_size = mpu.get_pipeline_model_parallel_world_size()
        if pp_size < 1:
            raise RuntimeError(f"invalid pipeline model parallel size {pp_size}")

        gathered: list[list[tuple[str, torch.Tensor]] | None] | None
        if pp_size == 1:
            # With one PP stage the native source rule must select global rank 0.
            # All ranks have already completed the per-parameter collectives above,
            # so nonzero ranks can wait at update_weights()'s trailing Gloo barrier
            # without serializing an empty object through another collective.
            expected_source = global_rank == 0
            if self._is_pp_src_rank != expected_source:
                raise RuntimeError(
                    "PP=1 LoRA sync requires global rank 0 to be the sole canonical "
                    f"PP source, but rank {global_rank} has _is_pp_src_rank={self._is_pp_src_rank}"
                )
            gathered = [local] if global_rank == 0 else None
        else:
            world_size = dist.get_world_size(gloo_group)
            gathered = [None] * world_size if global_rank == 0 else None
            dist.gather_object(local, object_gather_list=gathered, dst=0, group=gloo_group)
        _t_gather = _time.perf_counter()

        if global_rank != 0:
            return

        if gathered is None:
            raise RuntimeError("global rank 0 did not receive LoRA adapter payloads")

        # PP stage sources normally own disjoint global layer names.  Keep the
        # historical first-wins merge for tied/overlapping names.
        merged: dict[str, torch.Tensor] = {}
        for chunk in gathered:
            if chunk is None:
                raise RuntimeError("LoRA adapter gather returned an incomplete payload list")
            for name, tensor in chunk:
                merged.setdefault(name, tensor)

        state_dict, config_dict = build_lora_adapter_state_dict(self.args, list(merged.items()), scale=scale)
        _t_build = _time.perf_counter()

        engines = list(self.rollout_engines)
        if not engines:
            return

        self._apply_lora_adapter_swap(engines, state_dict, config_dict)
        _t_swap = _time.perf_counter()
        # wandb sections group by the first path segment: one "lora" group with
        # the adapter-shape panel (lora/lora_adapter/*) and the swap-phase panel
        # (lora/lora_swap/*).
        self.update_weight_metrics["lora/lora_adapter/num_tensors"] = float(len(state_dict))
        self.update_weight_metrics["lora/lora_adapter/rank"] = float(config_dict["r"])
        self.update_weight_metrics["lora/lora_adapter/bytes"] = float(
            sum(t.numel() * t.element_size() for t in state_dict.values())
        )
        self.update_weight_metrics["lora/lora_swap/gather_time"] = _t_gather - _t0
        self.update_weight_metrics["lora/lora_swap/build_time"] = _t_build - _t_gather
        self.update_weight_metrics["lora/lora_swap/engine_swap_time"] = _t_swap - _t_build
        self.update_weight_metrics["lora/lora_swap/total_time"] = _t_swap - _t0
        logger.info(
            "[lora_swap] gather=%.2fs build=%.2fs engine_swap=%.2fs total=%.2fs",
            _t_gather - _t0,
            _t_build - _t_gather,
            _t_swap - _t_build,
            _t_swap - _t0,
        )

    def _apply_lora_adapter_swap(self, engines, state_dict, config_dict) -> None:
        """Execute one alternating adapter swap on ``engines`` and advance
        ``_lora_prev_adapter_name``.

        QeRL-style ALTERNATING adapters: load a NEW name while the PREVIOUS adapter
        is still resident (double buffer), then unload the old one. ``plan_lora_swap``
        returns ``[("load", new)]`` on the first sync and ``[("load", new),
        ("unload", prev)]`` afterwards — load ALWAYS precedes unload so the
        cuda-graph-referenced old slot is never reload-reused in place (the
        same-slot reload is what triggers cudaErrorIllegalAddress). Needs
        ``--sglang-max-loras-per-batch >= 2`` (two mem-pool slots during the swap).
        Generation is paused+flushed for the whole sync, so the swap is
        request-free; the ordering is belt-and-suspenders for any in-flight ref.

        Split out from the gather so the swap ordering + first-sync restart cleanup
        is unit-testable without the megatron gather (see test_dsv4_lora_serve.py).
        """
        adapter_name = lora_adapter_name(self.weight_version)

        # First sync of this updater (fresh process / resume / external engine):
        # the engines may still hold adapters from a prior run in either
        # alternating slot. Clear all alternating names best-effort so the first
        # load never collides with a resident name and the mem-pool has a free
        # slot. Unloading an absent name is a harmless no-op on a fresh engine.
        if self._lora_prev_adapter_name is None:
            for stale in all_alternating_lora_names():
                for engine in engines:
                    try:
                        ray.get(engine.unload_lora_adapter.remote(lora_name=stale))
                    except Exception:
                        pass

        # A pending unload that timed out after the previous swap: retry before
        # loading (the old adapter's refs drained during the intervening rollout,
        # so this is instant now; also frees the mem-pool slot the load may need).
        pending = getattr(self, "_lora_pending_unload", None)
        if pending is not None:
            for engine in engines:
                try:
                    ray.get(engine.unload_lora_adapter.remote(lora_name=pending))
                except Exception:
                    pass
            self._lora_pending_unload = None

        for op, name in plan_lora_swap(adapter_name, self._lora_prev_adapter_name):
            if op == "load":
                results = ray.get(
                    [
                        engine.load_lora_adapter_from_tensors.remote(
                            lora_name=name,
                            tensors=state_dict,
                            config_dict=config_dict,
                            weight_version=str(self.weight_version),
                        )
                        for engine in engines
                    ]
                )
                # Fail loudly on a partial/failed load BEFORE unloading the old
                # (still-serving) adapter — otherwise engines are left serving
                # base-only / a stale adapter and _lora_prev_adapter_name would
                # advance past a name that never went live.
                raise_on_failed_lora_load(results, name)
            else:  # "unload"
                # DEFERRED: sglang's unload waits for the old adapter's request
                # refs to drain, which can never happen while generation is
                # paused — executing it here hung rank0 indefinitely and
                # collapsed the sync barrier (formal r7f/r7g 2026-07-10). The
                # unload now runs in _finish_lora_adapter_swap AFTER
                # continue_generation.
                self._lora_pending_unload = name
        self._lora_prev_adapter_name = adapter_name

    def _iter_non_expert_chunks(self) -> Iterator[list[tuple[str, torch.Tensor]]]:
        """
        Yield broadcast-sized HF chunks of non-expert params: TP all-gather +
        HF convert per param, then bucket up to ``--update-weight-buffer-size``.
        Empty on non-PP-src ranks (they still join all_gather_param).
        """
        buffer_size = 0
        buffer: list[tuple[str, torch.Tensor]] = []
        named_tensors = list(named_params_and_buffers(self.args, self.model))
        v4_lora_base_scales = self._get_v4_lora_base_scales()
        v4_lora_only_sync = bool(v4_lora_base_scales)
        v4_sglang_loader_pair_protection = _is_deepseekv4_model_name(self.model_name)
        tensor_by_name = dict(named_tensors) if v4_lora_only_sync else {}
        for name, param in named_tensors:
            if ".experts." in name:
                continue

            if v4_lora_only_sync:
                if _lora_base_weight_name(name) is not None:
                    continue
                if name not in v4_lora_base_scales:
                    continue

                prefix = name[: -len(_WEIGHT_SUFFIX)]
                lora_in_name = prefix + _LORA_IN_SUFFIX
                lora_out_name = prefix + _LORA_OUT_SUFFIX
                if lora_in_name not in tensor_by_name or lora_out_name not in tensor_by_name:
                    raise KeyError(
                        f"V4 LoRA base {name!r} is missing adapter tensors " f"{lora_in_name!r}/{lora_out_name!r}"
                    )

                base_param = all_gather_param(name, param)
                lora_in = all_gather_param(lora_in_name, tensor_by_name[lora_in_name])
                lora_out = all_gather_param(lora_out_name, tensor_by_name[lora_out_name])
                if not self._is_pp_src_rank:
                    continue
                param = _merge_lora_weight(base_param, lora_in, lora_out, v4_lora_base_scales[name])
            else:
                param = all_gather_param(name, param)

            if not self._is_pp_src_rank:
                continue
            hf_chunk = convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
            chunk_bytes = sum(t.numel() * t.element_size() for _, t in hf_chunk)
            if (
                buffer
                and buffer_size + chunk_bytes > self.args.update_weight_buffer_size
                and not (v4_sglang_loader_pair_protection and _v4_incomplete_sglang_loader_pairs(buffer))
            ):
                yield buffer
                buffer = []
                buffer_size = 0
            buffer.extend(hf_chunk)
            buffer_size += chunk_bytes
        if buffer:
            if v4_sglang_loader_pair_protection:
                incomplete = _v4_incomplete_sglang_loader_pairs(buffer)
                if incomplete:
                    raise ValueError(
                        "DeepSeek-V4 SGLang loader requires fused raw-name weight pairs "
                        f"in the same online update chunk; incomplete pairs: {incomplete}"
                    )
            yield buffer

    def _iter_expert_chunks(
        self,
        params: Iterator[tuple[str, torch.Tensor]] | None = None,
    ) -> Iterator[list[tuple[str, torch.Tensor]]]:
        """
        Yield one HF chunk per EP-weighted batch of expert params: TP gather +
        buffer until threshold, then EP gather + HF convert. ``params`` lets
        callers restrict the iterator to a subset; by default all expert
        parameters on this rank are used.
        """
        if self._uses_v4_lora_only_sync():
            return
        if params is None:
            params = ((n, p) for n, p in named_params_and_buffers(self.args, self.model) if ".experts." in n)
        buffer_size = 0
        batch: list[tuple[str, torch.Tensor]] = []
        for name, param in params:
            param = all_gather_param(name, param)
            param_size = param.numel() * param.element_size()
            if (
                buffer_size + param_size
            ) * mpu.get_expert_model_parallel_world_size() > self.args.update_weight_buffer_size:
                hf_chunk = self._ep_gather_and_convert(batch)
                if hf_chunk:
                    yield hf_chunk
                batch = []
                buffer_size = 0
            batch.append((name, param))
            buffer_size += param_size
        if batch:
            hf_chunk = self._ep_gather_and_convert(batch)
            if hf_chunk:
                yield hf_chunk

    def _ep_gather_and_convert(self, named_tensors: list[tuple[str, torch.Tensor]]) -> list[tuple[str, torch.Tensor]]:
        """
        EP all-gather a buffered batch + HF convert on PP source. Returns HF tensors on
        PP source, [] elsewhere. Clears ``named_tensors``.
        """
        names = [name for name, _ in named_tensors]
        all_names = [None] * mpu.get_expert_model_parallel_world_size()
        dist.all_gather_object(all_names, names, group=mpu.get_expert_model_parallel_group())

        for names in all_names:
            assert len(named_tensors) == len(names), f"mismatch names length: {len(named_tensors)} != {len(names)}"

        all_gathered_params = [[] for _ in range(mpu.get_expert_model_parallel_world_size())]
        handles = []
        for i, (_name, param) in enumerate(named_tensors):
            params = [
                torch.empty_like(param.data, device=accelerator.current_device())
                for _ in range(mpu.get_expert_model_parallel_world_size())
            ]
            handle = dist.all_gather(params, param.data, group=mpu.get_expert_model_parallel_group(), async_op=True)
            handles.append(handle)
            for ep_rank, names in enumerate(all_names):
                all_gathered_params[ep_rank].append((names[i], params[ep_rank]))
        for handle in handles:
            handle.wait()

        named_tensors.clear()
        if not self._is_pp_src_rank:
            return []

        all_gathered_params = sum(all_gathered_params, [])
        converted_hf_tensors = []
        for name, param in all_gathered_params:
            converted_hf_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
        return converted_hf_tensors

    def _update_bucket_weights_from_distributed(
        self,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        pbar: tqdm | None = None,
        load_format: str | None = None,
        delta: DeltaSpec | None = None,
    ) -> None:
        """
        Lock → transfer → clear → unlock → pbar++. Lock prevents communication deadlock.
        Delta sync passes ``load_format="delta"`` with its decoding specification.
        """
        # Lock the rollout engines to prevent communication deadlock.
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)

        try:
            refs = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                self.weight_version,
                self.rollout_engines,
                converted_named_tensors,
                load_format=load_format,
                delta=delta,
            )
            try:
                ray.get(refs)
            except Exception as exc:
                names = [name for name, _ in converted_named_tensors]
                preview = names[:8]
                suffix = " ..." if len(names) > len(preview) else ""
                raise RuntimeError(
                    f"Distributed weight update failed for group {self._group_name} "
                    f"with {len(names)} tensors: {preview}{suffix}"
                ) from exc
            converted_named_tensors.clear()
        finally:
            ray.get(self.rollout_engine_lock.release.remote())
        pbar.update(1)


def connect_rollout_engines_from_distributed(
    args: Namespace,
    group_name: str,
    rollout_engines: Sequence[ActorHandle],
    engine_gpu_counts: Sequence[int] | None = None,
) -> dist.ProcessGroup:
    """
    Create a device process group: training rank 0 + all engine GPUs. Blocks until joined.

    ``engine_gpu_counts`` gives the number of GPUs per engine.  When engines
    have heterogeneous TP sizes (e.g. prefill TP=2, decode TP=4), each engine
    occupies a different number of ranks in the process group.
    """
    if engine_gpu_counts is None:
        engine_gpu_counts = [args.rollout_num_gpus_per_engine] * len(rollout_engines)

    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    world_size = sum(engine_gpu_counts) + 1  # +1 for training rank 0

    # Compute cumulative rank offsets: engine i starts at cumulative[i] + 1.
    cumulative = [0]
    for c in engine_gpu_counts:
        cumulative.append(cumulative[-1] + c)

    backend = accelerator.weight_update_backend()
    refs = [
        engine.init_weights_update_group.remote(
            master_address=master_address,
            master_port=master_port,
            rank_offset=cumulative[i] + 1,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
        )
        for i, engine in enumerate(rollout_engines)
    ]
    model_update_groups = init_process_group(
        backend=backend,
        init_method=f"tcp://{_wrap_ipv6(master_address)}:{master_port}",
        world_size=world_size,
        rank=0,
        group_name=group_name,
    )
    ray.get(refs)
    return model_update_groups


def disconnect_rollout_engines_from_distributed(group_name, model_update_groups, rollout_engines):
    """
    Destroy the weight-update process group on training and engines.
    """
    refs = [engine.destroy_weights_update_group.remote(group_name) for engine in rollout_engines]
    dist.destroy_process_group(model_update_groups)
    ray.get(refs)


def update_weights_from_distributed(
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
    load_format: str | None = None,
    delta: DeltaSpec | None = None,
) -> list[ObjectRef]:
    """
    Send metadata through Ray and tensors through the configured transport.
    """
    request_kwargs = {
        "names": [name for name, _ in converted_named_tensors],
        "dtypes": [param.dtype for _, param in converted_named_tensors],
        "shapes": [param.shape for _, param in converted_named_tensors],
        "group_name": group_name,
        "weight_version": str(weight_version),
    }
    if load_format is not None:
        request_kwargs["load_format"] = load_format
    if delta is not None:
        request_kwargs["delta"] = delta
    refs = [engine.update_weights_from_distributed.remote(**request_kwargs) for engine in rollout_engines]
    handles = []
    for _, param in converted_named_tensors:
        handles.append(dist.broadcast(param.data, 0, group=group, async_op=True))
    for handle in handles:
        handle.wait()

    return refs


def post_process_weights(
    restore_weights_before_load: bool,
    post_process_quantization: bool,
    rollout_engines: Sequence[ActorHandle],
):
    """
    Trigger post-process for int4/fp4 quantization on all rollout engines.
    """
    ray.get(
        [
            engine.post_process_weights.remote(
                restore_weights_before_load=restore_weights_before_load,
                post_process_quantization=post_process_quantization,
            )
            for engine in rollout_engines
        ]
    )
