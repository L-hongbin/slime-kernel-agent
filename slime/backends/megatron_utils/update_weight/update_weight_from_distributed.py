from __future__ import annotations

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

from slime.utils.distributed_utils import get_gloo_group, init_process_group

from ..megatron_to_hf import convert_to_hf
from ..sglang import DeltaSpec
from .common import _maybe_v4_global_name, all_gather_param, named_params_and_buffers

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
            if not all(hasattr(module, attr) for attr in ("weight", "linear_in", "linear_out", "scale")):
                continue
            base_name = f"{module_name}.weight" if module_name else "weight"
            if not base_name.startswith("module.module."):
                base_name = "module." + base_name
            global_base_name = _maybe_v4_global_name(args, model_module, base_name, module.weight, expert_offset=None)
            scales[global_base_name or base_name] = float(module.scale)
    return scales


class UpdateWeightFromDistributed:
    """
    Update distributed engines via NCCL. Each PP rank: group "slime-pp_{pp_rank}",
    only DP=TP=0 broadcasts. Non-expert (TP) and expert (EP) params separate.
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

    def pop_metrics(self) -> dict[str, float]:
        """
        Return and clear ``update_weight_metrics``. Drained by the actor onto the rollout/step log.
        """
        out, self.update_weight_metrics = self.update_weight_metrics, {}
        return out

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """
        Create NCCL "slime-pp_{pp_rank}" if PP source (DP=TP=0). Lock prevents concurrent broadcasts.
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
            if self._model_update_groups is not None:
                disconnect_rollout_engines_from_distributed(
                    self.args, self._group_name, self._model_update_groups, self.rollout_engines
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
        disconnect_rollout_engines_from_distributed(
            self.args, self._group_name, self._model_update_groups, self.rollout_engines
        )
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
        dist.barrier(group=get_gloo_group())

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
        callers restrict the iter to a subset (used by delta-sync sub-passes);
        defaults to all expert params on this rank.
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
                torch.empty_like(param.data, device=torch.cuda.current_device())
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
        Lock → broadcast → clear → unlock → pbar++. Lock prevents NCCL deadlock.
        Delta sync passes ``load_format="delta"`` + a ``DeltaSpec`` describing the
        per-param decoding of the (__positions__, __values__) bucket tensors.
        """
        # lock the rollout engines to prevent dead lock on broadcast.
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
    Create NCCL group: training rank 0 + all engine GPUs. Blocks until joined.

    ``engine_gpu_counts`` gives the number of GPUs per engine.  When engines
    have heterogeneous TP sizes (e.g. prefill TP=2, decode TP=4), each engine
    occupies a different number of ranks in the NCCL group.
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

    refs = [
        engine.init_weights_update_group.remote(
            master_address=master_address,
            master_port=master_port,
            rank_offset=cumulative[i] + 1,
            world_size=world_size,
            group_name=group_name,
            backend="nccl",
        )
        for i, engine in enumerate(rollout_engines)
    ]
    model_update_groups = init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_address}:{master_port}",
        world_size=world_size,
        rank=0,
        group_name=group_name,
    )
    ray.get(refs)
    return model_update_groups


def disconnect_rollout_engines_from_distributed(args, group_name, model_update_groups, rollout_engines):
    """
    Destroy NCCL on training and engines.
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
    Send metadata (Ray), broadcast tensors (NCCL rank 0 → engines).
    Delta sync passes ``load_format="delta"`` + ``delta`` (DeltaSpec).
    """
    refs = [
        engine.update_weights_from_distributed.remote(
            names=[name for name, _ in converted_named_tensors],
            dtypes=[param.dtype for _, param in converted_named_tensors],
            shapes=[param.shape for _, param in converted_named_tensors],
            group_name=group_name,
            weight_version=str(weight_version),
            load_format=load_format,
            delta=delta,
        )
        for engine in rollout_engines
    ]

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
