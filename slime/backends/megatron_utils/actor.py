import logging
import os
import random
from argparse import Namespace
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import ray
import torch
import torch.distributed as dist

from .path_bootstrap import ensure_megatron_lm_on_sys_path

ensure_megatron_lm_on_sys_path()

from megatron.core import mpu
from torch_memory_saver import torch_memory_saver
from transformers import AutoConfig, AutoTokenizer

from slime.observability import train_data_utils, train_metric_utils
from slime.observability.logging_utils import init_tracking
from slime.observability.profile_utils import TrainProfiler
from slime.observability.timer import Timer, inverse_timer, timer, with_defer
from slime.observability.train_metric_utils import ENTROPY_COMMON_PROBE_MASK_KEY
from slime.ray.train_actor import TrainRayActor
from slime.utils import accelerator
from slime.utils.data import process_rollout_data
from slime.utils.distributed_utils import get_gloo_group
from slime.utils.memory_utils import clear_memory, print_memory
from slime.utils.misc import Box
from slime.utils.reloadable_process_group import (
    destroy_process_groups,
    monkey_patch_torch_dist,
    register_default_process_group,
    reload_process_groups,
)
from slime.utils.routing_replay import RoutingReplay, record_rollout_routing_replay_for_layer
from slime.utils.types import RolloutBatch

from ...utils.tensor_backper import TensorBackuper
from .checkpoint import load_checkpoint
from .cp_utils import prepare_routed_experts_for_routing_replay, slice_log_prob_with_cp
from .data import (
    DataIterator,
    compute_bshd_max_seq_lens,
    get_data_iterator,
    summarize_bshd_padding,
)
from .hf_checkpoint_saver import save_hf_model_to_path
from .initialize import init, is_megatron_main_rank
from .lora_old_actor import (
    LoRAOldActorSnapshot,
    enumerate_adapter_params,
    maybe_refresh_lora_old_actor_snapshot,
    resolve_batch_gen_version,
    should_recompute_old_actor_log_probs,
)
from .loss import (
    compute_advantages_and_returns,
    drain_captured_log_probs,
    enable_log_prob_capture,
    get_log_probs_and_entropy,
    get_values,
)
from .model import forward_only, initialize_model_and_optimizer, save, train
from .update_weight import create_weight_updater
from .update_weight.common import named_params_and_buffers

logging.getLogger("megatron").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def _slice_predictive_support_with_cp(
    support: np.ndarray,
    *,
    total_length: int,
    response_length: int,
    qkv_format: str,
    max_seq_len: int | None,
    dtype: torch.dtype,
    device,
) -> torch.Tensor:
    """Slice a compact [response, support] numpy matrix along response/CP."""
    if not isinstance(support, np.ndarray):
        raise TypeError(f"predictive support must be numpy.ndarray, got {type(support).__name__}")
    support_tensor = torch.from_numpy(support)
    support_tensor = slice_log_prob_with_cp(
        support_tensor,
        total_length,
        response_length,
        qkv_format,
        max_seq_len,
    )
    return support_tensor.to(device=device, dtype=dtype)


_PROTECTED_TRAIN_ENV_KEYS = {
    "CUDA_VISIBLE_DEVICES",
    "LOCAL_RANK",
    "MASTER_ADDR",
    "MASTER_PORT",
    "RANK",
    "WORLD_SIZE",
}


def _apply_train_env_vars(args: Namespace) -> None:
    train_env_vars = getattr(args, "train_env_vars", {}) or {}
    for key, value in train_env_vars.items():
        if key in _PROTECTED_TRAIN_ENV_KEYS or value is None:
            continue
        os.environ[str(key)] = str(value)


def _snapshot_entropy_common_probe_masks(rollout_data: RolloutBatch, *, enabled: bool) -> bool:
    """Freeze the pre-postprocess response masks for exact entropy comparison.

    This diagnostic is deliberately fail-closed: a fixed-batch arm must not
    silently overwrite a stale snapshot or compare populations after one of
    the per-sample fields has drifted out of alignment.  The snapshot is made
    before ``DataIterator`` construction and before sequence-MIS can replace
    rejected samples' masks.
    """
    if not enabled:
        return False
    if ENTROPY_COMMON_PROBE_MASK_KEY in rollout_data:
        raise RuntimeError(
            f"entropy common-probe snapshot already exists under {ENTROPY_COMMON_PROBE_MASK_KEY!r}; "
            "refusing to overwrite a duplicate/stale probe population"
        )

    required_fields = ("loss_masks", "response_lengths", "total_lengths", "tokens")
    missing = [key for key in required_fields if rollout_data.get(key) is None]
    if missing:
        raise RuntimeError(f"entropy common-probe snapshot is missing rollout fields: {missing}")

    loss_masks = rollout_data["loss_masks"]
    response_lengths = rollout_data["response_lengths"]
    total_lengths = rollout_data["total_lengths"]
    tokens = rollout_data["tokens"]
    if not isinstance(loss_masks, (list, tuple)):
        raise RuntimeError(
            f"entropy common-probe loss_masks must be a per-sample list/tuple, got {type(loss_masks).__name__}"
        )

    sample_count = len(loss_masks)
    field_lengths = {
        "loss_masks": sample_count,
        "response_lengths": len(response_lengths),
        "total_lengths": len(total_lengths),
        "tokens": len(tokens),
    }
    if len(set(field_lengths.values())) != 1:
        raise RuntimeError(f"entropy common-probe rollout field length mismatch: {field_lengths}")
    if sample_count == 0:
        raise RuntimeError("entropy common-probe snapshot requires at least one sample")

    frozen_masks: list[torch.Tensor] = []
    for index, (loss_mask, response_length, total_length, token_ids) in enumerate(
        zip(loss_masks, response_lengths, total_lengths, tokens, strict=True)
    ):
        if not isinstance(loss_mask, torch.Tensor) or loss_mask.ndim != 1:
            shape = tuple(loss_mask.shape) if isinstance(loss_mask, torch.Tensor) else None
            raise RuntimeError(
                f"entropy common-probe loss_masks[{index}] must be a 1-D tensor, "
                f"got type={type(loss_mask).__name__} shape={shape}"
            )
        response_length = int(response_length)
        total_length = int(total_length)
        if response_length < 0 or total_length <= 0 or response_length > total_length:
            raise RuntimeError(
                f"entropy common-probe invalid lengths at sample {index}: "
                f"response_length={response_length}, total_length={total_length}"
            )
        if loss_mask.numel() != response_length:
            raise RuntimeError(
                f"entropy common-probe mask/response length mismatch at sample {index}: "
                f"mask={loss_mask.numel()}, response_length={response_length}"
            )
        if not isinstance(token_ids, torch.Tensor) or token_ids.numel() != total_length:
            token_count = token_ids.numel() if isinstance(token_ids, torch.Tensor) else None
            raise RuntimeError(
                f"entropy common-probe token/total length mismatch at sample {index}: "
                f"tokens={token_count}, total_length={total_length}"
            )
        frozen_masks.append(loss_mask.detach().clone())

    rollout_data[ENTROPY_COMMON_PROBE_MASK_KEY] = frozen_masks
    return True


@contextmanager
def _cold_base_load_for_adapter_resume(args: Namespace, enabled: bool):
    """Force the converted frozen-base load to be weights-only.

    Stateful adapter resume is a two-source load: ``args.load`` points at the
    converted base checkpoint, while ``args.lora_adapter_resume_load`` points at
    the adapter+optimizer+RNG checkpoint.  The user's load flags belong to the
    second source; applying them to the base first asks a model-only conversion
    checkpoint for optimizer/RNG state and fails before the overlay is reached.
    """
    old = (args.no_load_optim, args.no_load_rng)
    if enabled:
        args.no_load_optim = True
        args.no_load_rng = True
    try:
        yield
    finally:
        args.no_load_optim, args.no_load_rng = old


class MegatronTrainRayActor(TrainRayActor):
    @with_defer(lambda: Timer().start("train_wait"))
    def init(
        self,
        args: Namespace,
        role: str,
        with_ref: bool = False,
        with_opd_teacher: bool = False,
    ) -> int | None:
        _apply_train_env_vars(args)
        if args.debug_rollout_only:
            self.args = args
            return 0

        monkey_patch_torch_dist()
        super().init(args, role, with_ref, with_opd_teacher)
        # Destroying and recreating WORLD invalidates raw dist.group.WORLD references cached by external code.
        # Set SLIME_DESTROY_WORLD_PROCESS_GROUP=0 when such references may outlive a train sleep/wake cycle.
        if os.getenv("SLIME_DESTROY_WORLD_PROCESS_GROUP", "1").lower() not in {"0", "false", "no"}:
            register_default_process_group(timeout=timedelta(minutes=args.distributed_timeout_minutes))
        else:
            logger.info("Default WORLD process-group destruction is disabled")

        init(args)

        if is_megatron_main_rank():
            init_tracking(args, primary=False, role=role)

        self.prof = TrainProfiler(args)

        # read config and tokenizer serialized to prevent concurrent writing bug.
        for i in range(args.num_gpus_per_node):
            if i == dist.get_rank() % args.num_gpus_per_node:
                self.hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
                self.tokenizer = AutoTokenizer.from_pretrained(self.args.hf_checkpoint, trust_remote_code=True)
            dist.barrier(group=get_gloo_group())

        dist.barrier(group=get_gloo_group())

        if args.offload_train:
            if (x := args.train_memory_margin_bytes) > 0:
                logger.info(f"Set torch_memory_saver.memory_margin_bytes to {x}")
                torch_memory_saver.memory_margin_bytes = x

        # V4 LoRA adapter-only resume: base loaded cold from --load above; overlay
        # saved adapters (+ Muon optim) from the adapter checkpoint and continue
        # the rollout counter from its iteration.
        from .adapter_ckpt import adapter_only_ckpt_enabled

        adapter_resume_path = args.lora_adapter_resume_load
        adapter_resume_active = bool(adapter_resume_path and adapter_only_ckpt_enabled(args) and role == "actor")
        adapter_resume_load_optim = not args.no_load_optim
        adapter_resume_load_rng = not args.no_load_rng
        with _cold_base_load_for_adapter_resume(args, adapter_resume_active):
            self.model, self.optimizer, self.opt_param_scheduler, loaded_rollout_id = initialize_model_and_optimizer(
                args, role
            )

        if adapter_resume_active:
            loaded_rollout_id = self.load_adapter_resume(
                adapter_resume_path,
                load_optim=adapter_resume_load_optim,
                load_rng=adapter_resume_load_rng,
            )

        # Fixed-batch entropy A/B only: prove the freshly loaded actor is
        # functionally the frozen base before any snapshot, publish, or optimizer
        # update.  LinearAdapter contributes B(A(x)); exact-zero trainable B on
        # every rank therefore makes the adapter delta identically zero even when
        # random A initialization differs.  Keep the import behind the gate so
        # ordinary training does not scan parameters or enter extra collectives.
        if role == "actor" and getattr(args, "assert_zero_lora_out", False):
            from .lora_zero_audit import assert_zero_lora_out

            assert_zero_lora_out(
                self.model,
                process_group=get_gloo_group(),
                should_print=is_megatron_main_rank(),
                print_fn=lambda message: print(message, flush=True),
            )

        vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size() or 1
        if vpp_size > 1:
            from megatron.core.utils import get_model_config

            microbatch_group_size_per_vp_stage = get_model_config(self.model[0]).microbatch_group_size_per_vp_stage
        else:
            microbatch_group_size_per_vp_stage = 1
        self.train_parallel_config = {
            "dp_size": mpu.get_data_parallel_world_size(with_context_parallel=False),
            "cp_size": mpu.get_context_parallel_world_size(),
            "vpp_size": vpp_size,
            "microbatch_group_size_per_vp_stage": microbatch_group_size_per_vp_stage,
        }

        start_rollout_id = loaded_rollout_id + 1

        if role == "critic":
            if self.args.offload_train:
                self.sleep()
            return start_rollout_id

        self.weights_backuper = TensorBackuper(
            source_getter=lambda: named_params_and_buffers(self.args, self.model),
        )
        self._active_model_tag: str | None = "actor"
        self.weights_backuper.backup("actor")

        if with_ref:
            self.load_other_checkpoint("ref", args.ref_load)

        # Load teacher model for Megatron-based on-policy distillation
        if with_opd_teacher:
            self.load_other_checkpoint("teacher", args.opd_teacher_load)

        # LoRA specialization of --keep-old-actor: when trainable adapter params
        # exist, the "old actor" differs from the live model ONLY by the adapter, so
        # we snapshot/swap just those (~90MB) instead of instantiating a second full
        # model. Set for the actor role only; consumers use getattr for safety.
        self.lora_old_actor = None
        if self.args.keep_old_actor:
            if self._should_use_lora_old_actor():
                # Snapshot contents refresh every training step but the published
                # version tag only advances at an update boundary; with an interval
                # above one the contents would drift ahead of the tag and the
                # version assert could pass while scoring with the wrong weights.
                assert getattr(self.args, "update_weights_interval", 1) == 1, (
                    "--keep-old-actor (LoRA adapter snapshot) requires "
                    "--update-weights-interval 1: the snapshot version tag is only "
                    f"correct when every step publishes; got interval "
                    f"{self.args.update_weights_interval}."
                )
                device = "cpu"
                self.lora_old_actor = LoRAOldActorSnapshot(self.model, device=device)
                logger.info(
                    "LoRA old-actor (adapter-only --keep-old-actor): %d adapter tensors, %.1f MB on %s; "
                    "NOT instantiating a second full model.",
                    self.lora_old_actor.num_tensors,
                    self.lora_old_actor.num_bytes / 1e6,
                    device,
                )
            else:
                if getattr(self.args, "debug_freeze_old_actor_snapshot", False):
                    raise RuntimeError(
                        "--debug-freeze-old-actor-snapshot requires the adapter-only LoRA old-actor path, "
                        "but no trainable LoRA adapter parameters were found."
                    )
                # Upstream full-model path: a second model instance for scoring.
                self.load_other_checkpoint("old_actor", args.load)
                # Create rollout_actor as a copy of current actor
                if args.update_weights_interval == 1:
                    self.weights_backuper.backup("rollout_actor")

        if self.args.vocab_size is None:
            # Prefer HF config vocab_size (which may include model-native padding)
            # over tokenizer vocab_size, which may be smaller.
            hf_vocab = getattr(self.hf_config, "vocab_size", None)
            self.args.vocab_size = hf_vocab if hf_vocab is not None else self.tokenizer.vocab_size

        self.weight_updater = create_weight_updater(
            self.args,
            self.model,
            weights_getter=lambda: self.weights_backuper.get("actor"),
            model_name=type(self.hf_config).__name__.lower() if self.args.model_name is None else self.args.model_name,
            quantization_config=getattr(self.hf_config, "quantization_config", None),
        )

        # empty cache after initialization
        clear_memory()

        if self.args.offload_train:
            # recover to actor in the end.
            self._switch_model("actor")
            self.sleep()

        self.rollout_engines = None

        self.rollout_data_postprocess = None
        if self.args.rollout_data_postprocess_path is not None:
            from slime.utils.misc import load_function

            self.rollout_data_postprocess = load_function(self.args.rollout_data_postprocess_path)

        self.prof.on_init_end()

        return start_rollout_id

    @timer
    def sleep(self) -> None:
        assert self.args.offload_train

        clear_memory(clear_host_memory=True)
        print_memory("before offload model")
        if (
            self.role == "actor"
            and self.args.use_critic
            and not self.args.colocate
            and hasattr(self.weight_updater, "disconnect_rollout_engines")
        ):
            self.weight_updater.disconnect_rollout_engines()
        destroy_process_groups()

        torch_memory_saver.pause()

        print_memory("after offload model")

    @timer
    def wake_up(self) -> None:
        assert self.args.offload_train
        print_memory("before wake_up model")

        torch_memory_saver.resume()

        clear_memory()
        reload_process_groups()

        if mpu.get_pipeline_model_parallel_world_size() > 2:
            # Megatron's patched batched pipeline P2P uses the default WORLD
            # group.  After reload, PP=4 starts with only the first two stages
            # entering batch_isend_irecv(), but PyTorch requires every rank when
            # that is the first NCCL operation on a group.  Prime WORLD here,
            # after the memory saver is resumed, so later stages cannot miss its
            # lazy initialization.  Sleep still destroys it completely.
            dist.barrier(device_ids=[accelerator.current_device()])
        if self.role == "actor":
            self._switch_model("actor")
        print_memory("after wake_up model")

    def _get_rollout_data(self, rollout_data_ref: Box) -> RolloutBatch:
        # Fetch data through ray on CPU, not sure if this will be performance bottleneck.
        # Both first pp stage and the last pp stage will receive the data.
        rollout_data = process_rollout_data(
            rollout_data_ref,
            mpu.get_data_parallel_rank(with_context_parallel=False),
            mpu.get_data_parallel_world_size(with_context_parallel=False),
        )
        # TODO: this is ugly, move to somewhere else?
        # move tokens to GPU in advance
        device = accelerator.current_device()
        rollout_data["tokens"] = [
            t.to(device=device, dtype=torch.long, non_blocking=True) for t in rollout_data["tokens"]
        ]
        rollout_data["loss_masks"] = [
            t.to(device=device, dtype=torch.int, non_blocking=True) for t in rollout_data["loss_masks"]
        ]
        if "rollout_mask_sums" in rollout_data:
            # Promote precomputed per-rollout mask totals to GPU tensors here
            # (matching loss_masks) so the loss reducer can just divide.
            rollout_data["rollout_mask_sums"] = rollout_data["rollout_mask_sums"].to(
                device=device, dtype=torch.float32, non_blocking=True
            )
        if "multimodal_train_inputs" in rollout_data:
            # Move multimodal training tensors to GPU in advance
            rollout_data["multimodal_train_inputs"] = [
                (
                    {
                        key: value.to(device=device, non_blocking=True) if isinstance(value, torch.Tensor) else value
                        for key, value in mm_dict.items()
                    }
                    if mm_dict is not None
                    else None
                )
                for mm_dict in rollout_data["multimodal_train_inputs"]
            ]

        if self.args.qkv_format == "bshd":
            # Pad each scheduled microbatch only to its own real maximum. BSHD
            # stacks the samples in an MBS, so all members of one MBS intentionally
            # share a width. Under contiguous CP, the helper also enforces the B1
            # invariant l_local=max_seq_len/cp_size is 128-aligned. PP>1 retains a
            # safe rollout-wide fallback because its P2P shapes are fixed for one
            # Megatron pipeline schedule; formal DS-V4 is PP1 and takes the dynamic
            # path.
            pad_size = mpu.get_tensor_model_parallel_world_size() * self.args.data_pad_size_multiplier
            rollout_data["max_seq_lens"] = compute_bshd_max_seq_lens(
                rollout_data["total_lengths"],
                rollout_data["micro_batch_indices"],
                pad_size=pad_size,
                cp_size=mpu.get_context_parallel_world_size(),
                cp_partition_mode=getattr(self.args, "cp_partition_mode", "zigzag"),
                pipeline_model_parallel_size=mpu.get_pipeline_model_parallel_world_size(),
            )

            # V4 PP needs the hidden-stream tensor shape to match this padded
            # per-rollout length. PP1 does not consume this override; PP>1's
            # fallback above makes every microbatch share the same maximum.
            self.args._v4_pp_current_seq_len = max(rollout_data["max_seq_lens"])

            # Log one line per DP shard (CP rank 0, last PP stage) so a GPU
            # canary can positively prove which sequence widths were used.
            # Including dp_rank prevents Ray log de-duplication from hiding
            # distinct local histograms.
            if (
                mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_context_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
            ):
                padding_summary = summarize_bshd_padding(
                    rollout_data["total_lengths"],
                    rollout_data["max_seq_lens"],
                )
                microbatch_width_order = [
                    max(rollout_data["max_seq_lens"][sample_index] for sample_index in microbatch)
                    for microbatch in rollout_data["micro_batch_indices"]
                ]
                microbatch_width_orders_by_step = []
                microbatch_cursor = 0
                for step_num_microbatches in rollout_data["num_microbatches"]:
                    step_end = microbatch_cursor + step_num_microbatches
                    microbatch_width_orders_by_step.append(microbatch_width_order[microbatch_cursor:step_end])
                    microbatch_cursor = step_end
                microbatch_widths_descending = all(
                    all(left >= right for left, right in zip(step_widths, step_widths[1:], strict=False))
                    for step_widths in microbatch_width_orders_by_step
                )
                logger.info(
                    "V4_ACTUAL_LENGTH_PADDING dp_rank=%s pad_multiplier=%s cp=%s pp=%s "
                    "samples=%s unique_widths=%s width_hist=%s raw_slots=%s padded_slots=%s "
                    "rollout_wide_slots=%s padding_overhead_pct=%.2f saved_vs_rollout_wide_pct=%.2f "
                    "microbatch_width_order=%s microbatch_widths_descending=%s",
                    mpu.get_data_parallel_rank(with_context_parallel=False),
                    self.args.data_pad_size_multiplier,
                    mpu.get_context_parallel_world_size(),
                    mpu.get_pipeline_model_parallel_world_size(),
                    padding_summary["samples"],
                    padding_summary["unique_widths"],
                    padding_summary["width_hist"],
                    padding_summary["raw_slots"],
                    padding_summary["padded_slots"],
                    padding_summary["rollout_wide_slots"],
                    padding_summary["padding_overhead_pct"],
                    padding_summary["saved_vs_rollout_wide_pct"],
                    ";".join(
                        ",".join(str(width) for width in step_widths)
                        for step_widths in microbatch_width_orders_by_step
                    ),
                    microbatch_widths_descending,
                )
        for key in ["rollout_log_probs", "teacher_log_probs"]:
            if key not in rollout_data:
                continue
            rollout_data[key] = [
                slice_log_prob_with_cp(
                    log_prob,
                    total_length,
                    response_length,
                    self.args.qkv_format,
                    rollout_data["max_seq_lens"][i] if self.args.qkv_format == "bshd" else None,
                ).to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
                for i, (log_prob, total_length, response_length) in enumerate(
                    zip(
                        rollout_data[key],
                        rollout_data["total_lengths"],
                        rollout_data["response_lengths"],
                        strict=False,
                    )
                )
            ]
        predictive_support_dtypes = {
            "rollout_topk_token_ids": torch.long,
            "rollout_topk_log_probs": torch.float32,
            "rollout_topk_valid_mask": torch.bool,
        }
        for key, dtype in predictive_support_dtypes.items():
            if key not in rollout_data:
                continue
            rollout_data[key] = [
                _slice_predictive_support_with_cp(
                    support,
                    total_length=total_length,
                    response_length=response_length,
                    qkv_format=self.args.qkv_format,
                    max_seq_len=(rollout_data["max_seq_lens"][i] if self.args.qkv_format == "bshd" else None),
                    dtype=dtype,
                    device=torch.cuda.current_device(),
                )
                for i, (support, total_length, response_length) in enumerate(
                    zip(
                        rollout_data[key],
                        rollout_data["total_lengths"],
                        rollout_data["response_lengths"],
                        strict=True,
                    )
                )
            ]
        if "rollout_routed_experts" in rollout_data:
            rollout_data["rollout_routed_experts"] = [
                torch.from_numpy(r) for r in rollout_data["rollout_routed_experts"]
            ]
        return rollout_data

    def _switch_model(self, target_tag: str) -> None:
        if target_tag not in self.weights_backuper.backup_tags:
            raise ValueError(f"Cannot switch to unknown model tag: {target_tag}")
        self.weights_backuper.restore(target_tag)
        self._active_model_tag = target_tag

    def _should_use_lora_old_actor(self) -> bool:
        """Take the adapter-only --keep-old-actor path iff trainable LoRA adapter
        params exist. Non-LoRA models keep the upstream full-model old-actor
        behavior."""
        return len(enumerate_adapter_params(self.model)) > 0

    def fill_routing_replay(self, data_iterator, num_microbatches, rollout_data):
        if "rollout_routed_experts" not in rollout_data:
            raise ValueError(
                "rollout_routed_experts is required in rollout_data when use_rollout_routing_replay is set."
            )

        from megatron.core.transformer.transformer_block import get_num_layers_to_build
        from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

        from slime.utils.routing_replay import RoutingReplay

        for iterator in data_iterator:
            iterator.reset()

        for _ in range(sum(num_microbatches)):
            batch = data_iterator[0].get_next(["rollout_routed_experts", "tokens", "max_seq_lens"])
            rollout_routed_experts = prepare_routed_experts_for_routing_replay(
                batch["rollout_routed_experts"],
                batch["tokens"],
                num_experts=self.args.num_experts,
                data_pad_size_multiplier=self.args.data_pad_size_multiplier,
                sequence_parallel=self.args.sequence_parallel,
                allgather_cp=self.args.allgather_cp,
                qkv_format=self.args.qkv_format,
                max_seq_lens=batch["max_seq_lens"],
            )

            routing_replay_offset = 0
            for vp_stage, model in enumerate(self.model):
                model_module = model.module
                config = model_module.config
                num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)
                offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
                for layer_id in range(offset, offset + num_layers_to_build):
                    # skip dense layer
                    if isinstance(config.moe_layer_freq, int):
                        if layer_id % config.moe_layer_freq != 0:
                            continue
                    elif isinstance(config.moe_layer_freq, list):
                        assert len(config.moe_layer_freq) == config.num_layers
                        if config.moe_layer_freq[layer_id] == 0:
                            continue
                    layer_routed_experts = rollout_routed_experts[:, layer_id]
                    routing_replay_offset = record_rollout_routing_replay_for_layer(
                        model_module,
                        layer_id,
                        layer_routed_experts,
                        routing_replay_offset,
                    )
            # The global registry can include routers from separately built
            # ref/teacher models. They remain empty in this actor pass and must
            # not make the active-model fill look incomplete.
            registered_with_records = sum(
                bool(replay.top_indices_list) for replay in RoutingReplay.all_routing_replays
            )
            if routing_replay_offset != registered_with_records:
                raise AssertionError(
                    "rollout routing replay did not fill all active-model replays: "
                    f"recorded={routing_replay_offset}, "
                    f"registered_with_records={registered_with_records}, "
                    f"registered_total={len(RoutingReplay.all_routing_replays)}"
                )

        del rollout_data["rollout_routed_experts"]

        for iterator in data_iterator:
            iterator.reset()

    def compute_log_prob(
        self,
        data_iterator: list[DataIterator],
        num_microbatches: list[int],
        store_prefix: str = "",
    ) -> dict[str, list[torch.Tensor]]:
        with timer(f"{store_prefix}log_probs"):
            return forward_only(
                get_log_probs_and_entropy,
                self.args,
                self.model,
                data_iterator,
                num_microbatches,
                store_prefix=store_prefix,
                use_rollout_top_p_replay=True,
            )

    def train(self, rollout_id: int, rollout_data_ref: Box, external_data=None):
        if self.args.debug_rollout_only:
            return None

        if self.args.offload_train:
            self.wake_up()

        with timer("data_preprocess"):
            rollout_data = self._get_rollout_data(rollout_data_ref)

        if self.role == "critic":
            result = self.train_critic(rollout_id, rollout_data)
        else:
            self.train_actor(rollout_id, rollout_data, external_data=external_data)
            result = None

        # A normal resident actor is finalized explicitly by the driver, after
        # its train refs have resolved.  Keeping that collective out of this
        # method makes the save/train overlap boundary observable and prevents
        # an implicit poll from racing a subsequently queued train call.  An
        # offloaded actor is the exception: sleep() destroys its process groups,
        # so any pending async checkpoint must be finalized first.
        if self.args.offload_train and self.args.async_save:
            self.finalize_async_save(rollout_id)

        if self.args.offload_train:
            del rollout_data
            self.sleep()

        return result

    def finalize_async_save(
        self,
        iteration: int,
        terminate: bool = False,
        wake_if_offloaded: bool = False,
    ) -> None:
        """Finalize a pending async checkpoint on the Ray actor's main thread."""
        if self.args.debug_rollout_only or not self.args.async_save:
            return

        from megatron.training.async_utils import maybe_finalize_async_save

        woke_for_cleanup = bool(wake_if_offloaded and self.args.offload_train)
        timer_name = (
            "async_save_finalize" if getattr(self, "role", "actor") == "actor" else f"{self.role}_async_save_finalize"
        )
        if woke_for_cleanup:
            self.wake_up()
        try:
            if dist.get_rank() == 0:
                logger.info("V4_ASYNC_SAVE_FINALIZE_START iteration=%s", iteration)
            with timer(timer_name):
                # This is intentionally blocking. Ray invokes actor methods
                # serially, so the collective cannot overlap this actor's train
                # method. Include the global barrier so this measures complete
                # worker shutdown rather than only the rank-local join.
                finalize_kwargs = {"blocking": True}
                if terminate:
                    finalize_kwargs["terminate"] = True
                maybe_finalize_async_save(**finalize_kwargs)
                if terminate:
                    dist.barrier()
            if dist.get_rank() == 0 and terminate:
                logger.info(
                    "V4_ASYNC_SAVE_WORKERS_TERMINATED iteration=%s world_size=%s",
                    iteration,
                    dist.get_world_size(),
                )
            if dist.get_rank() == 0:
                logger.info("V4_ASYNC_SAVE_FINALIZE_END iteration=%s", iteration)
            train_metric_utils.log_named_perf_timers(iteration, self.args, timer_name)
        finally:
            if woke_for_cleanup:
                self.sleep()

    def train_critic(self, rollout_id: int, rollout_data: RolloutBatch):
        """Train critic and return CPU values (used as old-values for the next actor train)."""
        data_iterator = get_data_iterator(rollout_data)
        num_microbatches = rollout_data["num_microbatches"]
        global_batch_sizes = rollout_data["global_batch_sizes"]

        # Compute current critic values (used as old_values for value loss and for actor advantages).
        rollout_data.update(forward_only(get_values, self.args, self.model, data_iterator, num_microbatches))

        compute_advantages_and_returns(self.args, rollout_data)

        self.args.loss_type = "value_loss"
        train(
            rollout_id,
            self.model,
            self.optimizer,
            self.opt_param_scheduler,
            data_iterator,
            num_microbatches,
            global_batch_sizes,
        )

        if mpu.is_pipeline_last_stage() and "values" in rollout_data:
            from slime.backends.megatron_utils.data import tensors_to_cpu

            return {"values": tensors_to_cpu(rollout_data["values"])}
        return {}

    def train_actor(self, rollout_id: int, rollout_data: RolloutBatch, external_data=None) -> None:
        _snapshot_entropy_common_probe_masks(
            rollout_data,
            enabled=getattr(self.args, "entropy_common_probe", False),
        )

        # Diagnostic (env-gated, no-op unless enabled): after update_weights and
        # before the first train forward/backward, scan live model params for
        # NaN/Inf to tell param-corruption from a backward-compute NaN.
        if os.environ.get("SLIME_DEBUG_CHECK_PARAMS", "0") == "1":
            try:
                bad = [
                    (n, tuple(p.shape))
                    for mdl in self.model
                    for n, p in mdl.named_parameters()
                    if not torch.isfinite(p.data).all()
                ]
                # Value checksum (not just finiteness): compare live full-loop vs
                # debug-replay to detect params changed in place by update_weights.
                csum = 0.0
                for mdl in self.model:
                    for _, p in mdl.named_parameters():
                        csum += p.data.float().abs().sum().item()
                print(
                    f"[SLIME_DEBUG_CHECK_PARAMS] rollout_id={rollout_id} rank={dist.get_rank()} "
                    f"pp={mpu.get_pipeline_model_parallel_rank()} nonfinite_params={len(bad)} "
                    f"param_abs_sum={csum:.8e} first={bad[:5]}",
                    flush=True,
                )
            except Exception as _e:  # never let the diagnostic break training
                print(f"[SLIME_DEBUG_CHECK_PARAMS] error: {_e}", flush=True)
        # Diagnostic (env-gated): autograd anomaly detection raises at the first
        # backward op that produces NaN, with a traceback to the forward op that
        # created it — pinpoints where the full-loop NaN originates.
        if os.environ.get("SLIME_DEBUG_ANOMALY", "0") == "1":
            torch.autograd.set_detect_anomaly(True, check_nan=True)
            if dist.get_rank() == 0:
                print("[SLIME_DEBUG_ANOMALY] autograd anomaly detection ON", flush=True)
        # Create data iterator for log_probs and train.
        data_iterator = get_data_iterator(rollout_data)
        num_microbatches = rollout_data["num_microbatches"]
        global_batch_sizes = rollout_data["global_batch_sizes"]

        if self.args.use_rollout_routing_replay:
            self.fill_routing_replay(data_iterator, num_microbatches, rollout_data)

        with inverse_timer("train_wait"), timer("train"):
            if self.args.compute_advantages_and_returns:
                if "ref" in self.weights_backuper.backup_tags:
                    if self.args.use_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    self._switch_model("ref")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="ref_",
                        )
                    )

                # Forward teacher model to get teacher_log_probs for Megatron-based OPD
                if "teacher" in self.weights_backuper.backup_tags:
                    if self.args.use_routing_replay:
                        os.environ["ROUTING_REPLAY_STAGE"] = "fallthrough"
                    self._switch_model("teacher")
                    rollout_data.update(
                        self.compute_log_prob(
                            data_iterator,
                            num_microbatches,
                            store_prefix="teacher_",
                        )
                    )

                lora_old = getattr(self, "lora_old_actor", None)
                if lora_old is not None:
                    # LoRA old-actor: the live model already holds the current
                    # adapter; the behavioral adapter is swapped in only around the
                    # scoring forward (below). Seed the snapshot on the first step
                    # (θ_0 == the just-pushed initial adapter). Live training normally
                    # refreshes it after scoring; explicit fixed-debug replay keeps this
                    # first snapshot instead so every repeat has the same behavior anchor.
                    if lora_old.version is None:
                        freeze_old_snapshot = bool(getattr(self.args, "debug_freeze_old_actor_snapshot", False))
                        if freeze_old_snapshot and self.weight_updater.weight_version != 0:
                            raise RuntimeError(
                                "Fixed debug replay requires the initial LoRA old-actor snapshot to be v0, "
                                f"but weight_updater is already v{self.weight_updater.weight_version}. "
                                "The prepared dump is stamped gen_weight_version=0; refusing a mismatched anchor."
                            )
                        maybe_refresh_lora_old_actor_snapshot(
                            lora_old,
                            version=self.weight_updater.weight_version,
                            freeze_after_seed=freeze_old_snapshot,
                        )
                        if freeze_old_snapshot and is_megatron_main_rank():
                            logger.warning(
                                "FIXED DEBUG REPLAY: seeded LoRA old-actor snapshot once at v%s; "
                                "snapshot refresh is disabled, so every repeat scores against the initial "
                                "behavior policy.",
                                lora_old.version,
                            )
                else:
                    self._switch_model("old_actor" if self.args.keep_old_actor else "actor")
                can_reuse_log_probs_in_loss = (
                    len(num_microbatches) == 1
                    and self.args.loss_type == "policy_loss"
                    and self.args.kl_coef == 0
                    and not self.args.use_rollout_logprobs
                    and not self.args.get_mismatch_metrics
                    and not self.args.use_critic
                    and not self.args.keep_old_actor
                    and not self.args.use_opd
                    and (not self.args.use_routing_replay or self.args.use_rollout_routing_replay)
                    and self.args.advantage_estimator != "gspo"
                )
                if should_recompute_old_actor_log_probs(self.args) and not can_reuse_log_probs_in_loss:
                    if (
                        getattr(self.args, "debug_force_old_actor_logprob_recompute", False)
                        and self.args.use_rollout_logprobs
                        and is_megatron_main_rank()
                    ):
                        logger.warning(
                            "FIXED DEBUG REPLAY: forcing frozen-old-actor log-prob recompute while "
                            "policy loss keeps rollout_log_probs as its denominator."
                        )
                    if self.args.use_routing_replay:
                        if self.args.use_rollout_routing_replay:
                            os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
                        else:
                            os.environ["ROUTING_REPLAY_STAGE"] = "record"
                    if lora_old is not None:
                        # Assert the batch's behavioral version matches the snapshot
                        # (collective, so a lone stale DP rank fails loudly instead of
                        # hanging the forward), then score under the old adapter and
                        # restore the live adapter (score_with_snapshot's finally).
                        expected_v = resolve_batch_gen_version(
                            rollout_data.get("gen_weight_versions"),
                            lora_old.version,
                            gloo_group=get_gloo_group(),
                        )
                        with lora_old.score_with_snapshot(expected_version=expected_v):
                            rollout_data.update(
                                self.compute_log_prob(
                                    data_iterator,
                                    num_microbatches,
                                    store_prefix="",
                                )
                            )
                    else:
                        rollout_data.update(
                            self.compute_log_prob(
                                data_iterator,
                                num_microbatches,
                                store_prefix="",
                            )
                        )
                    if (
                        lora_old is not None
                        and getattr(self.args, "sequence_mis_ratio_source", "rollout") == "old_actor"
                    ):
                        # Same-stack MIS: also recompute under the CURRENT live
                        # adapter (θ_k, already restored above) so the postprocess can
                        # form a pure-drift ratio (cur/old, both megatron) instead of
                        # the cross-engine megatron-vs-sglang pair. Validated
                        # incompatible with routing replay, so this is a plain forward.
                        rollout_data.update(
                            self.compute_log_prob(
                                data_iterator,
                                num_microbatches,
                                store_prefix="cur_",
                            )
                        )
                    if self.args.use_rollout_routing_replay:
                        RoutingReplay.clear_all_forward()

                if self.args.use_critic:
                    if external_data is not None and mpu.is_pipeline_last_stage():
                        values = external_data.get("values")
                        if values is not None:
                            from slime.backends.megatron_utils.data import tensors_to_gpu

                            rollout_data["values"] = tensors_to_gpu(values)
                if lora_old is not None:
                    # The live adapter (θ_k) was already restored by
                    # score_with_snapshot. In live training, refresh the behavioral
                    # snapshot BEFORE the gradient step, tagged with the version it was
                    # pushed to the engines as — it is exactly the next batch's policy.
                    # Fixed debug replay deliberately leaves the initial snapshot intact.
                    refreshed = maybe_refresh_lora_old_actor_snapshot(
                        lora_old,
                        version=self.weight_updater.weight_version,
                        freeze_after_seed=bool(getattr(self.args, "debug_freeze_old_actor_snapshot", False)),
                    )
                    if not refreshed and is_megatron_main_rank():
                        logger.info(
                            "FIXED DEBUG REPLAY: keeping LoRA old-actor snapshot frozen at v%s "
                            "after rollout_id=%s.",
                            lora_old.version,
                            rollout_id,
                        )
                elif self._active_model_tag != "actor":
                    self._switch_model("actor")

                # Calculate adv and returns. Need to performed before training (instead of on the fly),
                # because we may need normalize the whole rollout.
                compute_advantages_and_returns(self.args, rollout_data)

            if self.rollout_data_postprocess is not None:
                self.rollout_data_postprocess(self.args, rollout_id, rollout_data)

            train_metric_utils.log_rollout_data(
                rollout_id,
                self.args,
                rollout_data,
            )

            # Train
            if self.args.use_routing_replay:
                os.environ["ROUTING_REPLAY_STAGE"] = "replay_backward"
            # When dumping train debug data but the actor log_probs were not
            # recomputed separately (can_reuse_log_probs_in_loss / use_rollout_logprobs),
            # snapshot them from the training forward so the dump still carries
            # per-sample log_probs — at no extra forward pass.
            capture_log_probs = self.args.save_debug_train_data is not None and "log_probs" not in rollout_data
            if capture_log_probs:
                enable_log_prob_capture()
            with timer("actor_train"):
                train(
                    rollout_id,
                    self.model,
                    self.optimizer,
                    self.opt_param_scheduler,
                    data_iterator,
                    num_microbatches,
                    global_batch_sizes,
                )
            if capture_log_probs:
                captured = drain_captured_log_probs()
                # `captured` is non-empty only on the last PP stage running a loss
                # that snapshots log_probs (policy_loss), and then covers every
                # local sample. Key it by this rank's `partition` to land in local
                # sample order; skip otherwise (nothing to place).
                if captured:
                    rollout_data["log_probs"] = [captured[pos] for pos in rollout_data["partition"]]

            self.prof.step(rollout_id=rollout_id)

        train_data_utils.save_debug_train_data(self.args, rollout_id=rollout_id, rollout_data=rollout_data)

        if self.args.use_routing_replay:
            # Codex-review invariant: verify every recorded routing entry was
            # consumed by the train forward and by exactly the decoder layers
            # that activation checkpointing replays in backward (uniform=all,
            # block=first K local layers, off=none).
            RoutingReplay.check_fully_consumed(
                context=f"rollout {rollout_id} post-train",
                model_modules=self.model,
            )
            RoutingReplay.clear_all()

        # update the cpu actor weight to the latest model
        self.weights_backuper.backup("actor")

        # Update ref model if needed
        if (
            self.args.ref_update_interval is not None
            and (rollout_id + 1) % self.args.ref_update_interval == 0
            and "ref" in self.weights_backuper.backup_tags
        ):
            with timer("ref_model_update"):
                if is_megatron_main_rank():
                    logger.info(f"Updating ref model at rollout_id {rollout_id}")
                self.weights_backuper.backup("ref")

        train_metric_utils.log_perf_data(
            rollout_id,
            self.args,
            extra_metrics=self.weight_updater.pop_metrics(),
        )

    def save_model(self, rollout_id: int, force_sync: bool = False) -> None:
        if self.args.debug_rollout_only:
            return

        # Offloaded groups wake in a separate all-actor phase owned by
        # RayTrainGroup, so one rank's wake failure cannot let healthy ranks
        # enter checkpoint collectives alone.

        timer_name = "save_model" if self.role == "actor" else f"{self.role}_save_model"
        with timer(timer_name):
            if self.args.async_save:
                from megatron.training.async_utils import maybe_finalize_async_save

                maybe_finalize_async_save(blocking=True)

            save(rollout_id, self.model, self.optimizer, self.opt_param_scheduler)

        # Report schedule/D2H time at the checkpoint's own rollout. This helper
        # is best-effort and cannot interrupt checkpoint lifecycle control.
        train_metric_utils.log_named_perf_timers(rollout_id, self.args, timer_name)

        if force_sync and self.args.async_save:
            self.finalize_async_save(rollout_id, terminate=True)

        if self.args.offload_train:
            # A non-final save launched a new background request above. It must
            # be joined before sleep() destroys process groups.
            if self.args.async_save and not force_sync:
                self.finalize_async_save(rollout_id)

    def prepare_save_model(self) -> None:
        """Wake an offloaded actor before the group enters checkpoint collectives."""
        if self.args.offload_train:
            self.wake_up()

    def finish_save_model(self, rollout_id: int) -> None:
        """Run rank-local save postprocessing after checkpoint ownership transfers."""
        try:
            if self.args.save_hf is not None and self.role == "actor":
                save_hf_model_to_path(
                    self.args,
                    Path(self.args.save_hf.format(rollout_id=rollout_id)),
                    self.model,
                )
        finally:
            # All checkpoint collectives completed before rank-local HF work.
            # Keep every offloaded actor in the same asleep state even when one
            # rank's HF export fails, so driver cleanup can wake the full group.
            if self.args.offload_train:
                self.sleep()

    @timer
    def update_weights(self) -> None:
        if self.args.debug_train_only or self.args.debug_rollout_only:
            return

        if self.args.use_fault_tolerance:
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.recover_updatable_engines.remote())
            dist.barrier(group=get_gloo_group())

        (
            rollout_engines,
            rollout_engine_lock,
            num_new_engines,
            engine_gpu_counts,
            engine_gpu_offsets,
            engine_parallel_configs,
        ) = ray.get(self.rollout_manager.get_updatable_engines_and_lock.remote())

        reconnect_rollout_engines = self.args.offload_train and self.args.use_critic and not self.args.colocate

        if not rollout_engines and not reconnect_rollout_engines:
            if dist.get_rank() == 0:
                logger.info("No updatable SGLang engines are running; skip weight update.")
            return

        if reconnect_rollout_engines:
            self.wake_up()
        elif self.args.offload_train:
            reload_process_groups()

        if num_new_engines > 0 or reconnect_rollout_engines:
            self.weight_updater.connect_rollout_engines(
                rollout_engines,
                rollout_engine_lock,
                engine_gpu_counts=engine_gpu_counts,
                engine_gpu_offsets=engine_gpu_offsets,
                engine_parallel_configs=engine_parallel_configs,
            )
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                ray.get(self.rollout_manager.clear_updatable_num_new_engines.remote())

        with torch_memory_saver.disable() if self.args.offload_train else nullcontext():
            print_memory("before update_weights")
            self.weight_updater.update_weights()
            print_memory("after update_weights")

            if self.args.ci_test and len(rollout_engines) > 0:
                engine = random.choice(rollout_engines)
                engine_version = ray.get(engine.get_weight_version.remote())
                if str(engine_version) != str(self.weight_updater.weight_version):
                    raise RuntimeError(
                        f"Weight version mismatch! Engine: {engine_version}, Updater: {self.weight_updater.weight_version}"
                    )

            if getattr(self.args, "keep_old_actor", False) and getattr(self, "lora_old_actor", None) is None:
                # Full-model queue (upstream). The LoRA old-actor path keeps its own
                # adapter-only snapshot, refreshed in train_actor before the gradient
                # step, so it does not participate in this weights_backuper queue.
                if self.args.update_weights_interval == 1:
                    logger.info("updating model queue: rollout_actor -> old_actor, actor -> rollout_actor")
                    # Queue-style update: rollout_actor params -> old_actor, actor params -> rollout_actor
                    # First copy rollout_actor to old_actor
                    self.weights_backuper.copy(src_tag="rollout_actor", dst_tag="old_actor")
                    # Then copy current actor to rollout_actor
                    self.weights_backuper.backup("rollout_actor")
                else:
                    self.weights_backuper.backup("old_actor")

        if reconnect_rollout_engines:
            self.sleep()
        elif self.args.offload_train:
            destroy_process_groups()

    def load_other_checkpoint(self, model_tag: str, path: str) -> None:
        old_args = (
            self.args.load,
            self.args.no_load_optim,
            self.args.no_load_rng,
            self.args.finetune,
            self.args.ckpt_step,
        )
        self.args.load = path
        self.args.no_load_optim = True
        self.args.no_load_rng = True
        self.args.finetune = True

        if model_tag == "ref" and self.args.ref_ckpt_step is not None:
            self.args.ckpt_step = self.args.ref_ckpt_step
        elif model_tag == "teacher" and self.args.opd_teacher_ckpt_step is not None:
            self.args.ckpt_step = self.args.opd_teacher_ckpt_step

        _, _ = load_checkpoint(
            self.model,
            None,
            None,
            checkpointing_context={},
        )
        (
            self.args.load,
            self.args.no_load_optim,
            self.args.no_load_rng,
            self.args.finetune,
            self.args.ckpt_step,
        ) = old_args

        self.weights_backuper.backup(model_tag)
        self._active_model_tag = model_tag

    def load_adapter_resume(self, path: str, load_optim: bool = True, load_rng: bool = True) -> int:
        """Resume V4 LoRA training from an adapter-only checkpoint.

        The frozen base was already loaded cold from ``--load`` (torch_dist) in
        ``initialize_model_and_optimizer``; this overlays the saved adapter
        weights, optional Muon optimizer state, and optional RNG state from an
        adapter-only ``--save`` dir. Returns the checkpoint's iteration so the
        rollout counter continues instead of restarting at 0. The model
        sharded_state_dict is filtered to adapter keys so Megatron's load only
        requests the (present) adapter tensors, not the absent base.
        """
        from .adapter_ckpt import (
            adapter_only_model_load,
            validate_adapter_checkpoint_components,
            validate_adapter_scaling_marker,
            validate_loaded_lora_optimizer_state,
            validate_lora_optimizer_state,
        )

        # Fail loud if the checkpoint's recorded LoRA scaling (alpha/r vs rsLoRA's
        # alpha/sqrt(r), dim, alpha) differs from what the CLI args rebuilt at model
        # build — a silent mismatch rescales every adapter delta on resume.
        validate_adapter_scaling_marker(path, self.model, rslora=self.args.lora_rslora)
        if load_optim:
            # The same identity gate used before save also protects load: never
            # map checkpoint state onto an optimizer that owns frozen base params
            # or omits a live LoRA matrix.
            validate_lora_optimizer_state(self.model, self.optimizer)
        component_marker = validate_adapter_checkpoint_components(path, load_optimizer=load_optim, load_rng=load_rng)

        old = (self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune)
        self.args.load = path
        self.args.no_load_optim = not load_optim
        self.args.no_load_rng = not load_rng
        self.args.finetune = False  # a real resume: keep the checkpoint iteration
        try:
            with adapter_only_model_load(self.model):
                iteration, _ = load_checkpoint(
                    self.model,
                    self.optimizer if load_optim else None,
                    self.opt_param_scheduler if load_optim else None,
                    checkpointing_context={},
                    skip_load_to_model_and_opt=False,
                )
        finally:
            self.args.load, self.args.no_load_optim, self.args.no_load_rng, self.args.finetune = old
        if load_optim:
            validate_loaded_lora_optimizer_state(self.model, self.optimizer, component_marker)
        logger.info("V4 LoRA adapter resume: loaded adapters from %s at iteration %d", path, iteration)
        return iteration
