import dataclasses
import gc
import logging
import math
import os
from argparse import Namespace
from collections.abc import Callable, Sequence
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path

import torch
from megatron.core import mpu
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.muon import get_megatron_muon_optimizer
from megatron.core.optimizer.optimizer import MegatronOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.utils import get_model_config
from megatron.training.global_vars import get_args
from megatron.training.training import get_model
from tqdm import tqdm

try:
    from megatron.core.pipeline_parallel.utils import unwrap_model
except ImportError:
    from megatron.core.utils import unwrap_model
from slime.observability import logging_utils, train_metric_utils
from slime.observability.train_metric_utils import (
    ENTROPY_COMMON_PROBE_MASK_KEY,
    add_derived_dppo_metrics,
    add_derived_entropy_metrics,
    add_entropy_common_probe_metric,
    format_train_metric_key,
)
from slime.utils.memory_utils import clear_memory

from .checkpoint import load_checkpoint, save_checkpoint
from .data import DataIterator, get_batch
from .loss import ROLLOUT_TOP_P_TOKEN_KEYS, get_rollout_top_p_logprob_kwargs, loss_function
from .model_provider import get_model_provider_func
from .stateless_adam import StatelessAdam

logger = logging.getLogger(__name__)


def _disable_tqdm_for_non_main_rank() -> bool:
    return not (
        mpu.get_data_parallel_rank(with_context_parallel=True) == 0
        and mpu.get_tensor_model_parallel_rank() == 0
        and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
    )


def _should_update_microbatch_pbar(model) -> bool:
    if _disable_tqdm_for_non_main_rank():
        return False

    while hasattr(model, "module"):
        model = model.module
    vp_stage = getattr(model, "vp_stage", None)
    if mpu.get_virtual_pipeline_model_parallel_world_size() is not None and vp_stage is not None:
        return mpu.is_pipeline_last_stage(ignore_virtual=False, vp_stage=vp_stage)
    return mpu.is_pipeline_last_stage(ignore_virtual=True)


def _v4_current_pp_seq_length(args: Namespace) -> int:
    """The sequence length the CURRENT forward pass actually communicates.

    slime pads each rollout's tokens to a per-step ``max_seq_len`` (actor.py, the
    padded max over ``total_lengths``), stored on ``args._v4_pp_current_seq_len``.
    That value — not the static ``args.seq_length`` — is what ``get_batch``
    produces and what RoPE/position_ids use, so it is the length PP must
    communicate. Falls back to ``args.seq_length`` when unset (e.g. before the
    first rollout, or non-rollout forwards)."""
    override = getattr(args, "_v4_pp_current_seq_len", None)
    if override:
        return int(override)
    return int(args.seq_length)


def _v4_pp_sequence_length(args: Namespace) -> int:
    """Sequence length that V4 PP stages communicate for the current batch.

    In slime's ``bshd`` path, ``get_batch`` pads token/loss-mask tensors to the
    per-step ``max_seq_len``.  PP hidden-state communication must use the same
    padded sequence length; otherwise non-first stages recompute RoPE for the
    padded input ids but receive a shorter hidden stream (the RL log-prob forward
    crashed here: q S=128 from a stale args.seq_length vs cos/sin S=384 from the
    real rollout data).
    """
    seq_length = _v4_current_pp_seq_length(args)
    if getattr(args, "qkv_format", None) == "bshd":
        tp_size = int(getattr(args, "tensor_model_parallel_size", 1) or 1)
        pad_multiplier = int(getattr(args, "data_pad_size_multiplier", 1) or 1)
        pad_size = max(1, tp_size * pad_multiplier)
        seq_length = ((seq_length + pad_size - 1) // pad_size) * pad_size
    return seq_length


def _v4_pp_adjust_tensor_shapes_fn(args: Namespace, model: Sequence[DDP]):
    """Return Megatron PP tensor-shape adapter for V4's 4D hc stream, if needed.

    Megatron's non-interleaved PP schedule defaults to communicating [S, B, H].
    V4 stages pass the hyper-connection stream [B, S, hc_mult, H], so PP>1 needs
    explicit recv/send shapes. Interleaved PP still has no adjust hook upstream.
    """
    if getattr(args, "pipeline_model_parallel_size", 1) <= 1:
        return None
    if getattr(args, "virtual_pipeline_model_parallel_size", None) is not None:
        raise ValueError("V4LanguageModel does not support virtual pipeline parallelism yet")

    hf_config = None
    for module in unwrap_model(model):
        hf_config = getattr(module, "hf_config", None)
        if hf_config is not None and hasattr(hf_config, "hc_mult"):
            break
    if hf_config is None or not hasattr(hf_config, "hc_mult"):
        return None

    hc_mult = int(hf_config.hc_mult)
    hidden_size = int(hf_config.hidden_size)
    pp_seq_length = _v4_pp_sequence_length(args)

    def adjust_tensor_shapes(_recv_shapes, _send_shapes):
        shape = (args.micro_batch_size, pp_seq_length, hc_mult, hidden_size)
        return [shape], [shape]

    return adjust_tensor_shapes


def _wrap_forward_step_with_microbatch_pbar(forward_step_func, pbar):
    if pbar is None:
        return forward_step_func

    def wrapped_forward_step(*args, **kwargs):
        result = forward_step_func(*args, **kwargs)
        model = args[1] if len(args) > 1 else kwargs.get("model")
        if model is not None and _should_update_microbatch_pbar(model):
            pbar.update(1)
        return result

    return wrapped_forward_step


def _with_rollout_top_p_token_keys(args: Namespace, keys: Sequence[str]) -> list[str]:
    if args.rollout_top_p == 1.0:
        return list(keys)
    return [*keys, *ROLLOUT_TOP_P_TOKEN_KEYS]


def _iter_critic_output_layers(model: Sequence[DDP]):
    for chunk_id, module in enumerate(unwrap_model(model)):
        output_layer = getattr(module, "output_layer", None)
        if output_layer is not None:
            yield chunk_id, output_layer


try:
    from megatron.training.checkpointing import get_load_checkpoint_path_by_args
except ImportError:

    def get_load_checkpoint_path_by_args(args, load_arg="load"):
        from megatron.training.checkpointing import (
            get_checkpoint_name,
            get_checkpoint_tracker_filename,
            isfile,
            read_metadata,
        )

        """Get the checkpoint path based on the arguments."""
        load_dir = getattr(args, load_arg)
        iteration, release = -1, False
        tracker_filename = "because load directory is not defined"
        if load_dir is not None:
            tracker_filename = get_checkpoint_tracker_filename(load_dir)
            if isfile(tracker_filename):
                iteration, release = read_metadata(tracker_filename)
            else:
                load_dir, checkpoint_step = os.path.split(load_dir)
                if checkpoint_step == "release" or checkpoint_step.startswith("iter_"):
                    release = checkpoint_step == "release"
                    if not release:
                        iteration = int(checkpoint_step.split("_")[1])

        # Allow user to specify the loaded iteration.
        if getattr(args, "ckpt_step", None):
            iteration = args.ckpt_step

        return get_checkpoint_name(load_dir, iteration, release, return_base_dir=True)


def _critic_output_layer_needs_reinit(args: Namespace, model: Sequence[DDP], role: str) -> bool:
    if role != "critic" or args.load is None:
        return False

    from megatron.core.dist_checkpointing.serialization import load_tensors_metadata

    checkpoint_path = Path(get_load_checkpoint_path_by_args(args))
    if not (checkpoint_path / ".metadata").is_file():
        return False

    checkpoint_metadata = load_tensors_metadata(str(checkpoint_path))
    for _chunk_id, output_layer in _iter_critic_output_layers(model):
        for name in ("weight", "bias"):
            param = getattr(output_layer, name, None)
            if param is None:
                continue

            param_name = f"output_layer.{name}"
            ckpt_tensor_metadata = next(
                (
                    tensor_metadata
                    for key, tensor_metadata in checkpoint_metadata.items()
                    if key == param_name or key.endswith(f".{param_name}")
                ),
                None,
            )
            expected_shape = tuple(param.shape)
            checkpoint_shape = tuple(ckpt_tensor_metadata.global_shape) if ckpt_tensor_metadata is not None else None
            if checkpoint_shape == expected_shape:
                continue

            reason = (
                "missing from checkpoint metadata"
                if checkpoint_shape is None
                else f"shape mismatch checkpoint={checkpoint_shape} runtime={expected_shape}"
            )
            logger.warning(
                "Will reinitialize critic %s after checkpoint load because it is %s",
                param_name,
                reason,
            )
            return True

    return False


@torch.no_grad()
def _reinitialize_critic_output_layer(args: Namespace, model: Sequence[DDP]) -> None:
    init_method_std = getattr(args, "init_method_std", None)
    if init_method_std is None:
        init_method_std = 0.02
    for _chunk_id, output_layer in _iter_critic_output_layers(model):
        output_layer.weight.data.normal_(mean=0.0, std=init_method_std)
        if output_layer.bias is not None:
            output_layer.bias.data.zero_()


def get_optimizer_param_scheduler(args: Namespace, optimizer: MegatronOptimizer) -> OptimizerParamScheduler:
    """Create and configure the optimizer learning-rate/weight-decay scheduler.

    This configures iteration-based schedules derived from the global batch size
    and run-time arguments.

    Args:
        args (Namespace): Training/runtime arguments (argparse namespace).
        optimizer (MegatronOptimizer): Megatron optimizer bound to the model.

    Returns:
        OptimizerParamScheduler: Initialized scheduler bound to ``optimizer``.
    """
    # Iteration-based training. ``train_iters`` is an estimate of the total
    # number of training steps — it's only used to size Megatron's LR decay
    # schedule (and ``lr_decay_iters`` defaults to it). With variable per-rollout
    # sample counts (dynamic sampling / filtering / custom step splitter) the
    # *actual* total can drift; the schedule still tracks the true progress via
    # ``opt_param_scheduler.num_steps`` (samples consumed, also persisted across
    # resume), so the worst case is the cosine/linear schedule reaches its
    # plateau slightly early or late. Pass ``--lr-decay-iters`` explicitly if you
    # need exact decay control.
    args.train_iters = _estimate_train_iters(
        args.num_rollout,
        args.rollout_batch_size,
        args.n_samples_per_prompt,
        args.global_batch_size,
    )
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    lr_decay_steps = args.lr_decay_iters * args.global_batch_size
    wd_incr_steps = args.train_iters * args.global_batch_size
    wsd_decay_steps = None
    if args.lr_wsd_decay_iters is not None:
        wsd_decay_steps = args.lr_wsd_decay_iters * args.global_batch_size
    if args.lr_warmup_fraction is not None:
        lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
    else:
        lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size

    opt_param_scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=args.lr_warmup_init,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        use_checkpoint_opt_param_scheduler=args.use_checkpoint_opt_param_scheduler,
        override_opt_param_scheduler=args.override_opt_param_scheduler,
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=args.lr_wsd_decay_style,
    )

    return opt_param_scheduler


def _noop_init_state_fn(*args, **kwargs) -> None:
    return None


def _disable_distributed_optimizer_state_initialization(optimizer: MegatronOptimizer) -> None:
    for megatron_optimizer in getattr(optimizer, "chained_optimizers", [optimizer]):
        if megatron_optimizer.__class__.__name__ == "DistributedOptimizer":
            megatron_optimizer.init_state_fn = _noop_init_state_fn


@contextmanager
def _patch_megatron_adam(adam_cls):
    import megatron.core.optimizer as megatron_optimizer
    import megatron.core.optimizer.distrib_optimizer as megatron_distrib_optimizer

    missing = object()
    old_adam = megatron_optimizer.Adam
    old_cpu_adam = getattr(megatron_optimizer, "CPUAdam", missing)
    old_distrib_adam = megatron_distrib_optimizer.Adam
    try:
        megatron_optimizer.Adam = adam_cls
        if old_cpu_adam is not missing:
            megatron_optimizer.CPUAdam = adam_cls
        megatron_distrib_optimizer.Adam = adam_cls
        yield
    finally:
        megatron_optimizer.Adam = old_adam
        if old_cpu_adam is not missing:
            megatron_optimizer.CPUAdam = old_cpu_adam
        megatron_distrib_optimizer.Adam = old_distrib_adam


def _estimate_train_iters(
    num_rollout: int,
    rollout_batch_size: int,
    n_samples_per_prompt: int,
    global_batch_size: int,
) -> int:
    total_samples = num_rollout * rollout_batch_size * n_samples_per_prompt
    if total_samples <= 0:
        return 0
    return max(1, math.ceil(total_samples / global_batch_size))


_LORA_B_PARAM_SUFFIX = ".linear_out.weight"


def _v4_lora_plus_config_overrides(config: OptimizerConfig, model: Sequence[DDP], lam: float):
    """Megatron ``config_overrides`` implementing LoRA+ (eta_B = lambda * eta_A).

    Builds on the standard overrides (bias/1D weight-decay skip — preserved so the
    non-LoRA behavior is unchanged) and adds a ``ParamKey`` matching the LoRA B
    matrices (``*.linear_out.weight``; only megatron-bridge ``LinearAdapter``
    modules own a ``linear_out`` child, and only adapter params are trainable, so
    the glob cannot catch base weights). The override sets:

    * ``max_lr`` / ``min_lr`` scaled by lambda — these are the per-group knobs
      ``OptimizerParamScheduler.get_lr`` actually honors (it IGNORES ``lr_mult``;
      the scheduler multiplies nothing), so the B group tracks every decay style
      at exactly lambda x the A group (min_lr scaled too, keeping the ratio exact
      through cosine/linear floors; during a warmup ramp the ratio approaches
      lambda as lr leaves ``init_lr``).
    * ``lr_mult = lambda`` — NOT for the scheduler (ignored there) but because it
      is part of Megatron's param-group identity tuple
      (``param_group_identifier_keys = (wd_mult, lr_mult, is_expert_parallel,
      is_decoupled_lr)`` — note ``max_lr`` is NOT in it). Without a distinct
      ``lr_mult`` the A and B groups would collide into ONE identifier on
      optimizer-state save/resume (``_filter_and_reorder_param_groups`` keys a
      dict on the tuple), silently cross-mapping Muon momentum between groups.
      With it, resume round-trips, and resuming with a DIFFERENT lambda (or
      toggling LoRA+ across a resume) fails loud with a missing-group ValueError
      instead of silently keeping the old ratio.

    Raises if no trainable LoRA B param exists (lambda set on a non-LoRA run is a
    config error, not a no-op).
    """
    from megatron.core.optimizer import get_standard_config_overrides
    from megatron.core.optimizer.optimizer_config import ParamKey

    n_b = 0
    for chunk in model:
        for name, param in chunk.named_parameters():
            if param.requires_grad and name.endswith(_LORA_B_PARAM_SUFFIX):
                n_b += 1
    if n_b == 0:
        raise RuntimeError(
            f"--lora-plus-lambda={lam} is set but the model has no trainable LoRA "
            f"B params (*{_LORA_B_PARAM_SUFFIX}); LoRA+ requires --lora-dim > 0."
        )

    overrides = get_standard_config_overrides(config)
    overrides[ParamKey(name=f"*{_LORA_B_PARAM_SUFFIX}")] = {
        "max_lr": lam * config.lr,
        "min_lr": lam * (config.min_lr or 0.0),
        "lr_mult": float(lam),
    }
    return overrides


def _log_v4_lora_plus_groups(optimizer: MegatronOptimizer, model: Sequence[DDP], lam: float) -> None:
    """Verify + log the LoRA+ split once at init (after the scheduler's step(0)).

    Fails loud if the optimizer does not contain exactly the expected B group:
    the B (lr_mult == lambda) group's param shapes must equal the model's
    trainable ``linear_out`` shapes (shape multiset survives the fp32
    master-param clone inside Float16OptimizerWithFloat16Params, unlike ids),
    and its lr must be exactly lambda x the default group's lr.
    """
    expected_b_shapes = sorted(
        tuple(p.shape)
        for chunk in model
        for name, p in chunk.named_parameters()
        if p.requires_grad and name.endswith(_LORA_B_PARAM_SUFFIX)
    )
    b_groups = [g for g in optimizer.param_groups if g.get("lr_mult") == lam and g["params"]]
    a_groups = [g for g in optimizer.param_groups if g.get("lr_mult", 1.0) == 1.0 and g["params"]]
    if len(b_groups) != 1 or not a_groups:
        raise RuntimeError(
            f"LoRA+ group split failed: found {len(b_groups)} B group(s) with "
            f"lr_mult=={lam} and {len(a_groups)} default group(s) "
            f"(param_groups={[(g.get('lr_mult'), len(g['params'])) for g in optimizer.param_groups]})"
        )
    b_group = b_groups[0]
    got_b_shapes = sorted(tuple(p.shape) for p in b_group["params"])
    if got_b_shapes != expected_b_shapes:
        raise RuntimeError(
            f"LoRA+ B group does not contain exactly the LoRA linear_out params: "
            f"expected {len(expected_b_shapes)} tensors, got {len(got_b_shapes)}"
        )
    eta_a = a_groups[0]["lr"]
    eta_b = b_group["lr"]
    if eta_b != lam * eta_a:
        raise RuntimeError(f"LoRA+ effective LRs wrong at init: eta_A={eta_a} eta_B={eta_b} lambda={lam}")
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        logger.info(
            "[lora+] optimizer param groups split: eta_A=%.3e (%d tensors), eta_B=%.3e "
            "(%d tensors), lambda=%g (--lora-plus-lambda)",
            eta_a,
            sum(len(g["params"]) for g in a_groups),
            eta_b,
            len(b_group["params"]),
            lam,
        )


def setup_model_and_optimizer(
    args: Namespace,
    role: str = "actor",
) -> tuple[list[DDP], MegatronOptimizer | None, OptimizerParamScheduler | None]:
    """Build model(s), wrap with DDP, and construct optimizer and scheduler.

    Args:
        args (Namespace): Training/runtime arguments (argparse namespace).
        role (str): Logical role of the model (e.g., "actor", "critic").
        no_wd_decay_cond (Callable[..., bool] | None): Predicate to exclude
            parameters from weight decay.
        scale_lr_cond (Callable[..., bool] | None): Predicate to scale LR for
            selected parameter groups.
        lr_mult (float): Global learning-rate multiplier for the optimizer.

    Returns:
        tuple[list[DDP], MegatronOptimizer | None, OptimizerParamScheduler | None]:
            - List of model chunks wrapped by ``DDP``.
            - The constructed ``MegatronOptimizer`` instance.
            - The learning-rate/weight-decay scheduler tied to the optimizer.
    """
    assert not args.moe_use_upcycling
    assert args.load is not None or args.pretrained_checkpoint is not None

    model = get_model(get_model_provider_func(args, role), ModelType.encoder_or_decoder)

    if args.num_rollout == 0:
        args.no_load_optim = True
        return model, None, None

    # Optimizer
    kwargs = {}
    for f in dataclasses.fields(OptimizerConfig):
        if hasattr(args, f.name):
            kwargs[f.name] = getattr(args, f.name)
    config = OptimizerConfig(**kwargs)
    config.timers = None

    # LoRA+ (--lora-plus-lambda, default unset/1.0 = OFF): separate param group
    # for the LoRA B matrices with eta_B = lambda * eta_A. When off,
    # config_overrides stays None and the optimizer construction below is
    # byte-identical to the pre-LoRA+ behavior (None triggers megatron's own
    # get_standard_config_overrides fallback inside _get_param_groups).
    from slime.utils.lora_utils import lora_plus_lambda

    lora_plus_lam = lora_plus_lambda(args)
    config_overrides = (
        _v4_lora_plus_config_overrides(config, model, lora_plus_lam) if lora_plus_lam is not None else None
    )

    if args.use_stateless_adam:
        assert config.optimizer == "adam", "Stateless Adam only supports --optimizer adam."
        assert args.no_save_optim, "Stateless Adam does not save Adam moment states. Please set --no-save-optim."

    if "muon" in config.optimizer:
        optimizer = get_megatron_muon_optimizer(
            config=config,
            model_chunks=model,
            config_overrides=config_overrides,
            use_gloo_process_groups=args.enable_gloo_process_groups,
            layer_wise_distributed_optimizer="dist" in config.optimizer,
        )
    else:
        optimizer_context = _patch_megatron_adam(StatelessAdam) if args.use_stateless_adam else nullcontext()
        with optimizer_context:
            optimizer = get_megatron_optimizer(
                config=config,
                model_chunks=model,
                config_overrides=config_overrides,
                use_gloo_process_groups=args.enable_gloo_process_groups,
            )
        if args.use_stateless_adam:
            _disable_distributed_optimizer_state_initialization(optimizer)
    opt_param_scheduler = get_optimizer_param_scheduler(args, optimizer)
    if lora_plus_lam is not None:
        # scheduler __init__ ran step(0), so per-group lrs are live — verify + log.
        _log_v4_lora_plus_groups(optimizer, model, lora_plus_lam)
    return model, optimizer, opt_param_scheduler


def enable_forward_pre_hook(model_chunks: Sequence[DDP]) -> None:
    """Enable forward pre-hooks for provided DDP-wrapped model chunks.

    Args:
        model_chunks (Sequence[DDP]): Sequence of DDP modules to enable hooks on.
    """
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        if model_chunk.remove_forward_pre_hook_handles:
            continue
        model_chunk.enable_forward_pre_hook()


def disable_forward_pre_hook(model_chunks: Sequence[DDP], param_sync: bool = True) -> None:
    """Disable forward pre-hooks for provided DDP-wrapped model chunks.

    Args:
        model_chunks (Sequence[DDP]): Sequence of DDP modules to disable hooks on.
        param_sync (bool): Whether to synchronize parameters when disabling.
    """
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        if not model_chunk.remove_forward_pre_hook_handles:
            continue
        model_chunk.disable_forward_pre_hook(param_sync=param_sync)


@torch.no_grad()
def forward_only(
    f: Callable[..., dict[str, list[torch.Tensor]]],
    args: Namespace,
    model: Sequence[DDP],
    data_iterator: Sequence[DataIterator],
    num_microbatches: Sequence[int],
    store_prefix: str = "",
    use_rollout_top_p_replay: bool = False,
) -> dict[str, list[torch.Tensor]]:
    """Run forward passes only and collect non-loss outputs (e.g., logprobs).

    The model is put into evaluation mode, a forward-only pipeline pass is
    executed, and relevant outputs are aggregated and returned.

    Args:
        f (Callable[..., dict[str, list[torch.Tensor]]]): Post-forward callback used to
            compute and package outputs to collect. This should accept a logits
            tensor as its first positional argument and additional keyword-only
            arguments; see ``get_log_probs_and_entropy``/``get_values`` in
            ``megatron_utils.loss`` for examples. It will be partially applied
            so that the callable returned from the internal forward step only
            requires the logits tensor.
        args (Namespace): Runtime arguments.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding batches for inference.
        num_microbatches (Sequence[int]): Number of microbatches per rollout step.
        store_prefix (str): Prefix to prepend to stored output keys.
        use_rollout_top_p_replay (bool): Whether to pass rollout top-p token sets
            to the post-forward log-prob callback when top-p rollout is enabled.

    Returns:
        dict[str, list[torch.Tensor]]: Aggregated outputs keyed by ``store_prefix + key``.
    """

    # reset data iterator
    for iterator in data_iterator:
        iterator.reset()

    config = get_model_config(model[0])
    batch_keys = [
        "tokens",
        "loss_masks",
        "multimodal_train_inputs",
        "total_lengths",
        "response_lengths",
        "max_seq_lens",
    ]
    if use_rollout_top_p_replay:
        batch_keys = _with_rollout_top_p_token_keys(args, batch_keys)

    def forward_step(
        data_iterator: DataIterator, model: GPTModel, return_schedule_plan: bool = False
    ) -> tuple[torch.Tensor, Callable[[torch.Tensor], dict[str, list[torch.Tensor]]]]:
        """Forward step used by Megatron's pipeline engine.

        Args:
            data_iterator (DataIterator): Input data iterator.
            model (GPTModel): The GPT model chunk to execute.

        Returns:
            tuple[torch.Tensor, Callable[[torch.Tensor], dict[str, list[torch.Tensor]]]]:
            Output tensor(s) and a callable that computes and packages results
            to be collected by the engine.
        """

        assert not return_schedule_plan, "forward_only step should never return schedule plan"

        # Get the batch.
        batch = get_batch(
            data_iterator,
            batch_keys,
            args.data_pad_size_multiplier,
            args.qkv_format,
            args.allgather_cp,
        )
        unconcat_tokens = batch["unconcat_tokens"]
        tokens = batch["tokens"]
        packed_seq_params = batch["packed_seq_params"]
        total_lengths = batch["total_lengths"]
        response_lengths = batch["response_lengths"]
        forward_kwargs = {
            "input_ids": tokens,
            "position_ids": None,
            "attention_mask": None,
            "labels": None,
            "packed_seq_params": packed_seq_params,
            "loss_mask": batch["full_loss_masks"],
        }
        if batch["multimodal_train_inputs"] is not None:
            forward_kwargs.update(batch["multimodal_train_inputs"])
        output_tensor = model(**forward_kwargs)

        output_kwargs = {
            "args": args,
            "unconcat_tokens": unconcat_tokens,
            "total_lengths": total_lengths,
            "response_lengths": response_lengths,
            "with_entropy": args.use_rollout_entropy,
        }
        if use_rollout_top_p_replay:
            output_kwargs.update(get_rollout_top_p_logprob_kwargs(args, batch))

        return output_tensor, partial(f, **output_kwargs)

    # Turn on evaluation mode which disables dropout.
    for model_module in model:
        model_module.eval()

    if args.custom_megatron_before_log_prob_hook_path:
        from slime.utils.misc import load_function

        custom_before_log_prob_hook = load_function(args.custom_megatron_before_log_prob_hook_path)
        custom_before_log_prob_hook(args, model, store_prefix)

    forward_backward_func = get_forward_backward_func()
    # V4's PP hidden stream is 4D [B, S, hc_mult, H]; PP>1 needs the same
    # recv/send shape adapter here as the training forward (build_train_step),
    # otherwise stage 1's log-prob forward allocates a 3D recv buffer and the mHC
    # kernel fails with "not enough values to unpack (expected 4, got 3)". Returns
    # None for PP<=1 / non-V4 models, so this is a no-op elsewhere.
    adjust_tensor_shapes_fn = _v4_pp_adjust_tensor_shapes_fn(args, model)
    # When the V4 adapter is active it fixes the PP tensor shape, so Megatron's
    # own get_tensor_shapes must be told the same (current-rollout) seq length,
    # not the static args.seq_length. No-op for non-V4 / PP<=1 (adapter is None).
    log_prob_seq_length = _v4_current_pp_seq_length(args) if adjust_tensor_shapes_fn is not None else args.seq_length
    # Don't care about timing during evaluation
    config.timers = None
    forward_data_store = []
    num_steps_per_rollout = len(num_microbatches)
    microbatch_pbar = tqdm(
        total=sum(num_microbatches),
        desc=f"{(store_prefix or getattr(model[0], 'role', 'actor')).rstrip('_')} forward",
        unit="microbatch",
        dynamic_ncols=True,
        leave=False,
        disable=_disable_tqdm_for_non_main_rank(),
    )
    forward_step_with_progress = _wrap_forward_step_with_microbatch_pbar(forward_step, microbatch_pbar)
    for step_id in range(num_steps_per_rollout):
        forward_data_store += forward_backward_func(
            forward_step_func=forward_step_with_progress,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=num_microbatches[step_id],
            seq_length=log_prob_seq_length,
            micro_batch_size=args.micro_batch_size,
            forward_only=True,
            adjust_tensor_shapes_fn=adjust_tensor_shapes_fn,
        )
    microbatch_pbar.close()

    # Move model back to the train mode.
    for model_module in model:
        model_module.train()

    rollout_data = {}
    # Store the results on the last stage
    if mpu.is_pipeline_last_stage():
        keys = forward_data_store[0].keys()
        for key in keys:
            values = []
            for value in forward_data_store:
                assert isinstance(value[key], list)
                values += value[key]

            if args.use_dynamic_batch_size:
                # TODO: This is ugly... Find a better way to make the data have the same order.
                # TODO: move this out of the loop.
                origin_values = [None] * len(values)
                origin_indices = sum(data_iterator[0].micro_batch_indices, [])
                for value, origin_index in zip(values, origin_indices, strict=False):
                    origin_values[origin_index] = value
                values = origin_values
            rollout_data[f"{store_prefix}{key}"] = values
    return rollout_data


def train_one_step(
    args: Namespace,
    rollout_id: int,
    step_id: int,
    data_iterator: Sequence[DataIterator],
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
    num_microbatches: int,
    step_global_batch_size: int,
    microbatch_pbar=None,
) -> tuple[dict[str, float], float, dict[str, float]]:
    """Execute a single pipeline-parallel training step.

    Runs forward/backward over ``num_microbatches``, applies optimizer step and
    one scheduler step when gradients are valid.

    Args:
        args (Namespace): Runtime arguments.
        rollout_id (int): Rollout identifier.
        step_id (int): Step index within the current rollout.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding training batches.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
        num_microbatches (int): Number of microbatches to process.
        step_global_batch_size (int): Rollout count for this training step
            (total across DP; one "rollout" = one execution of one of the
            ``n_samples_per_prompt`` rollouts, which may emit >1 training
            sample under compact / subagent). Used both as the loss
            normalizer inside the closure and as the LR scheduler
            ``increment``. In the common case (1 rollout = 1 sample) this
            equals the per-step sample count, so behavior is unchanged.

    Returns:
        tuple[dict[str, float], float, dict[str, float]]: Reduced loss
        dictionary (last stage only), gradient norm, and optional gradient
        diagnostic metrics for logging.
    """
    args = get_args()

    # Set grad to zero.
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    if args.custom_megatron_before_train_step_hook_path:
        from slime.utils.misc import load_function

        custom_before_train_step_hook = load_function(args.custom_megatron_before_train_step_hook_path)
        custom_before_train_step_hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler)

    def forward_step(data_iterator: DataIterator, model: GPTModel, return_schedule_plan: bool = False) -> tuple[
        torch.Tensor,
        Callable[[torch.Tensor], tuple[torch.Tensor, int, dict[str, torch.Tensor | list[str]]]],
    ]:
        """Forward step used by Megatron's pipeline engine during training.

        Args:
            data_iterator (DataIterator): Input data iterator.
            model (GPTModel): The GPT model chunk to execute.

        Returns:
            tuple[torch.Tensor, Callable[[torch.Tensor], tuple[torch.Tensor, int, dict[str, torch.Tensor | list[str]]]]]:
            Output tensor(s) and the loss function, which returns
            (loss, num_elems, {"keys": list[str], "values": torch.Tensor}).
        """

        # Get the batch.
        batch = get_batch(
            data_iterator,
            _with_rollout_top_p_token_keys(
                args,
                [
                    "tokens",
                    "multimodal_train_inputs",
                    "packed_seq_params",
                    "total_lengths",
                    "response_lengths",
                    "loss_masks",
                    ENTROPY_COMMON_PROBE_MASK_KEY,
                    "log_probs",
                    "ref_log_probs",
                    "values",
                    "advantages",
                    "returns",
                    "rollout_log_probs",
                    "rollout_topk_token_ids",
                    "rollout_topk_log_probs",
                    "rollout_topk_valid_mask",
                    "max_seq_lens",
                    "teacher_log_probs",
                    "rollout_mask_sums",
                    # Only present when dumping train debug data; lets the loss
                    # snapshot each sample's log_probs keyed by rollout position.
                    *(["partition"] if args.save_debug_train_data is not None else []),
                ],
            ),
            args.data_pad_size_multiplier,
            args.qkv_format,
            args.allgather_cp,
        )

        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            old_stage = os.environ["ROUTING_REPLAY_STAGE"]
            os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"

        if return_schedule_plan:
            assert not args.enable_mtp_training, "MTP training should not be enabled when using combined 1f1b"
            position_ids = None
            output_tensor = model.build_schedule_plan(
                input_ids=batch["tokens"],
                position_ids=position_ids,
                attention_mask=None,
                labels=None,
                packed_seq_params=batch["packed_seq_params"],
                loss_mask=batch["full_loss_masks"],
            )
        else:
            forward_kwargs = {
                "input_ids": batch["tokens"],
                "position_ids": None,
                "attention_mask": None,
                "labels": None,
                "packed_seq_params": batch["packed_seq_params"],
                "loss_mask": batch["full_loss_masks"],
            }

            if batch["multimodal_train_inputs"] is not None:
                forward_kwargs.update(batch["multimodal_train_inputs"])

            if args.enable_mtp_training:
                forward_kwargs["mtp_kwargs"] = {"mtp_labels": batch["tokens"]}

            output_tensor = model(**forward_kwargs)

        if os.environ.get("ENABLE_ROUTING_REPLAY", "0") == "1":
            os.environ["ROUTING_REPLAY_STAGE"] = old_stage

        return output_tensor, partial(loss_function, args, batch, num_microbatches, step_global_batch_size)

    # Forward pass.
    forward_backward_func = get_forward_backward_func()
    adjust_tensor_shapes_fn = _v4_pp_adjust_tensor_shapes_fn(args, model)
    # See forward_only: when the V4 adapter fixes the PP shape, Megatron's
    # get_tensor_shapes must use the same current-rollout seq length.
    train_seq_length = _v4_current_pp_seq_length(args) if adjust_tensor_shapes_fn is not None else args.seq_length
    losses_reduced = forward_backward_func(
        forward_step_func=_wrap_forward_step_with_microbatch_pbar(forward_step, microbatch_pbar),
        data_iterator=data_iterator,
        model=model,
        num_microbatches=num_microbatches,
        seq_length=train_seq_length,
        micro_batch_size=args.micro_batch_size,
        decoder_seq_length=args.decoder_seq_length,
        forward_only=False,
        adjust_tensor_shapes_fn=adjust_tensor_shapes_fn,
    )

    valid_step = True
    grad_norm = float("nan")
    if not getattr(args, "check_for_nan_in_loss_and_grad", True):
        found_inf_flag = optimizer.prepare_grads()
        if found_inf_flag:
            valid_step = False
        else:
            grad_norm = optimizer.get_grad_norm()
            if isinstance(grad_norm, torch.Tensor):
                valid_step = not (torch.isnan(grad_norm) or torch.isinf(grad_norm))
            else:
                valid_step = not (math.isnan(grad_norm) or math.isinf(grad_norm))

    # CI check: verify only MTP parameters have non-zero gradients when truncation happens
    # This check must happen before optimizer.step() as gradients may be modified during step
    if args.ci_test and args.enable_mtp_training:
        from slime.backends.megatron_utils.ci_utils import check_mtp_only_grad

        check_mtp_only_grad(model, step_id)

    if valid_step:
        # Update parameters.
        update_successful, grad_norm, num_zeros_in_grad = optimizer.step()

        # Update learning rate. Use the per-step global_batch_size when dynamic
        # batching is on so the scheduler's samples-seen counter tracks reality.
        assert update_successful
        opt_param_scheduler.step(increment=step_global_batch_size)

    # release grad
    for model_chunk in model:
        model_chunk.zero_grad_buffer()
    optimizer.zero_grad()

    if mpu.is_pipeline_last_stage(ignore_virtual=True):
        loss_reduced = train_metric_utils.reduce_train_step_metrics(
            losses_reduced,
            calculate_per_token_loss=args.calculate_per_token_loss,
            step_global_batch_size=step_global_batch_size,
            cp_size=mpu.get_context_parallel_world_size(),
            dp_with_cp_group=mpu.get_data_parallel_group(with_context_parallel=True),
        )
        return loss_reduced, grad_norm
    return {}, grad_norm


def should_disable_forward_pre_hook(args: Namespace) -> bool:
    """Block forward pre-hook for certain configurations."""
    return args.use_distributed_optimizer and args.overlap_param_gather


def train(
    rollout_id: int,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
    data_iterator: Sequence[DataIterator],
    num_microbatches: Sequence[int],
    global_batch_sizes: Sequence[int],
) -> None:
    """Run training over a rollout consisting of multiple steps.

    The model is switched to train mode, training hooks are configured, and
    ``train_one_step`` is invoked for each step in the rollout.

    Args:
        rollout_id (int): Rollout identifier.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
        data_iterator (Sequence[DataIterator]): Iterable(s) yielding training batches.
        num_microbatches (Sequence[int]): Microbatches per step in the rollout.
        global_batch_sizes (Sequence[int]): Rollout count per step (total
            across DP; one "rollout" = one execution of one of the
            ``n_samples_per_prompt`` rollouts of a prompt). Same length as
            ``num_microbatches``; consumed by ``train_one_step`` for loss
            scaling and LR scheduler increments. Equals per-step sample count
            in the common case (1 rollout = 1 sample).
    """
    args = get_args()

    assert len(num_microbatches) == len(global_batch_sizes), (
        f"num_microbatches and global_batch_sizes must have the same length, "
        f"got {len(num_microbatches)} vs {len(global_batch_sizes)}"
    )

    for iterator in data_iterator:
        iterator.reset()

    # Turn on training mode which enables dropout.
    for model_module in model:
        model_module.train()

    # Setup some training config params.
    config = get_model_config(model[0])
    config.grad_scale_func = optimizer.scale_loss
    config.timers = None
    if args.overlap_grad_reduce:
        config.no_sync_func = None
        config.grad_sync_func = None
    if args.overlap_param_gather:
        config.param_sync_func = None
    if isinstance(model[0], DDP) and args.overlap_grad_reduce:
        assert config.no_sync_func is None, (
            "When overlap_grad_reduce is True, config.no_sync_func must be None; "
            "a custom no_sync_func is not supported when overlapping grad-reduce"
        )
        config.no_sync_func = [model_chunk.no_sync for model_chunk in model]
        if len(model) == 1:
            config.no_sync_func = config.no_sync_func[0]
        if args.align_grad_reduce:
            config.grad_sync_func = [model_chunk.start_grad_sync for model_chunk in model]
            if len(model) == 1:
                config.grad_sync_func = config.grad_sync_func[0]
    if args.overlap_param_gather and args.align_param_gather:
        config.param_sync_func = [model_chunk.start_param_sync for model_chunk in model]
        if len(model) == 1:
            config.param_sync_func = config.param_sync_func[0]
    config.finalize_model_grads_func = finalize_model_grads

    pre_hook_enabled = False

    if args.reset_optimizer_states:
        if (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
        ):
            logger.info("Reset optimizer states")
        for chained_optimizer in optimizer.chained_optimizers:
            for group in chained_optimizer.optimizer.param_groups:
                if "step" in group:
                    group["step"] = 0
            for state in chained_optimizer.optimizer.state.values():
                if "step" in state:
                    if isinstance(state["step"], torch.Tensor):
                        state["step"].zero_()
                    else:
                        state["step"] = 0
                if "exp_avg" in state:
                    state["exp_avg"].zero_()
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].zero_()

    if args.manual_gc:
        # Disable the default garbage collector and perform the collection manually.
        # This is to align the timing of garbage collection across ranks.
        assert args.manual_gc_interval >= 0, "Manual garbage collection interval should be larger than or equal to 0"
        gc.disable()
        gc.collect()

    # Disable forward pre-hook to start training to ensure that errors in checkpoint loading
    # or random initialization don't propagate to all ranks in first all-gather (which is a
    # no-op if things work correctly).
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model, param_sync=False)
        # Also remove param_sync_func temporarily so that sync calls made in
        # `forward_backward_func` are no-ops.
        param_sync_func = config.param_sync_func
        config.param_sync_func = None
        pre_hook_enabled = False

    num_steps_per_rollout = len(num_microbatches)
    microbatch_pbar = tqdm(
        total=sum(num_microbatches),
        desc=f"{getattr(model[0], 'role', 'actor')} train",
        unit="microbatch",
        dynamic_ncols=True,
        leave=False,
        disable=_disable_tqdm_for_non_main_rank(),
    )

    # Run training iterations till done.
    for step_id in range(num_steps_per_rollout):

        # Run training step.
        loss_dict, grad_norm = train_one_step(
            args,
            rollout_id,
            step_id,
            data_iterator,
            model,
            optimizer,
            opt_param_scheduler,
            num_microbatches[step_id],
            global_batch_sizes[step_id],
            microbatch_pbar=microbatch_pbar,
        )

        if step_id == 0:
            # Enable forward pre-hook after training step has successfully run. All subsequent
            # forward passes will use the forward pre-hook / `param_sync_func` in
            # `forward_backward_func`.
            if should_disable_forward_pre_hook(args):
                enable_forward_pre_hook(model)
                config.param_sync_func = param_sync_func
                pre_hook_enabled = True

        if args.enable_mtp_training:
            from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

            mtp_loss_scale = 1 / num_microbatches[step_id]
            tracker = MTPLossLoggingHelper.tracker
            if "values" in tracker:
                values = tracker["values"]
                if tracker.get("reduce_group") is not None:
                    torch.distributed.all_reduce(values, group=tracker.get("reduce_group"))
                if tracker.get("avg_group") is not None:
                    torch.distributed.all_reduce(values, group=tracker["avg_group"], op=torch.distributed.ReduceOp.AVG)
                # Multi-head MTP: tracker["values"] is [num_mtp_layers]; aggregate below.
                mtp_losses = tracker["values"] * mtp_loss_scale
                MTPLossLoggingHelper.clean_loss_in_tracker()

                # CI check: verify MTP loss is within expected bounds
                if args.ci_test:
                    from slime.backends.megatron_utils.ci_utils import check_mtp_loss

                    check_mtp_loss(mtp_losses.sum().item())

        # per train step log.
        if (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == mpu.get_pipeline_model_parallel_world_size() - 1
        ):
            accumulated_step_id = rollout_id * num_steps_per_rollout + step_id
            role = getattr(model[0], "role", "actor")
            role_tag = "" if role == "actor" else f"{role}-"
            add_entropy_common_probe_metric(
                loss_dict,
                required=getattr(args, "entropy_common_probe", False),
            )
            add_derived_dppo_metrics(loss_dict)
            add_derived_entropy_metrics(loss_dict)
            log_dict = {
                format_train_metric_key(key, role_tag): val.mean().item() if isinstance(val, torch.Tensor) else val
                for key, val in loss_dict.items()
            }
            log_dict[f"train/{role_tag}grad_norm"] = grad_norm
            if args.enable_mtp_training:
                for _i in range(mtp_losses.shape[0]):
                    log_dict[f"train/{role_tag}mtp_{_i + 1}_loss"] = mtp_losses[_i].item()
                log_dict[f"train/{role_tag}mtp_loss"] = mtp_losses.sum().item()
            for param_group_id, param_group in enumerate(optimizer.param_groups):
                log_dict[f"train/{role_tag}lr-pg_{param_group_id}"] = opt_param_scheduler.get_lr(param_group)

            # Per-step gbs — uneven step sizes are easy to miss without this.
            log_dict[f"train/{role_tag}global_batch_size"] = global_batch_sizes[step_id]
            log_dict["train/step"] = accumulated_step_id
            if args.wandb_always_use_train_step:
                log_dict["rollout/step"] = accumulated_step_id
            logging_utils.log(args, log_dict, step_key="train/step")

            if args.ci_test and "train/train_rollout_logprob_abs_diff" in log_dict:
                threshold = args.ci_train_rollout_logprob_abs_diff_threshold
                assert log_dict["train/train_rollout_logprob_abs_diff"] <= threshold, f"{threshold=} {log_dict=}"

            if args.ci_test and not args.ci_disable_kl_checker:
                if step_id == 0 and "train/ppo_kl" in log_dict and "train/pg_clipfrac" in log_dict:
                    # TODO: figure out why KL is not exactly zero when using PPO loss with KL clipping, and whether this is expected behavior or a bug.
                    assert log_dict["train/ppo_kl"] < 1e-8, f"{log_dict=}"
                # R3 replays rollout routing for the actor path, while ref
                # log-probs are computed with normal routing. The initial
                # actor/ref KL is therefore not expected to be exactly zero.
                if (
                    accumulated_step_id == 0
                    and not getattr(args, "use_rollout_routing_replay", False)
                    and "train/kl_loss" in log_dict
                ):
                    assert log_dict["train/kl_loss"] < 1e-8, f"{log_dict=}"

            logger.info(f"{role_tag}step {accumulated_step_id}: {log_dict}")

            if args.ci_save_grad_norm is not None:
                ci_save_grad_norm_path = args.ci_save_grad_norm.format(
                    role=role,
                    rollout_id=rollout_id,
                    step_id=step_id,
                )
                torch.save(grad_norm, ci_save_grad_norm_path)
            elif args.ci_load_grad_norm is not None:
                ci_load_grad_norm_path = args.ci_load_grad_norm.format(
                    role=role,
                    rollout_id=rollout_id,
                    step_id=step_id,
                )
                expected_grad_norm = torch.load(ci_load_grad_norm_path)
                assert math.isclose(
                    grad_norm,
                    expected_grad_norm,
                    rel_tol=0.01,
                    abs_tol=0.01,
                ), f"grad norm mismatch: {grad_norm} != {expected_grad_norm}"
    microbatch_pbar.close()
    # Close out pre-hooks if using distributed optimizer and overlapped param gather.
    if pre_hook_enabled:
        disable_forward_pre_hook(model)


def save(
    iteration: int,
    model: Sequence[DDP],
    optimizer: MegatronOptimizer,
    opt_param_scheduler: OptimizerParamScheduler,
) -> None:
    """Persist a training checkpoint safely with forward hooks disabled.

    Args:
        iteration (int): Current global iteration number.
        model (Sequence[DDP]): Sequence of DDP-wrapped model chunks.
        optimizer (MegatronOptimizer): Optimizer instance.
        opt_param_scheduler (OptimizerParamScheduler): LR/WD scheduler.
    """
    from .adapter_ckpt import (
        adapter_checkpoint_staging,
        adapter_only_ckpt_enabled,
        adapter_only_model_save,
        finalize_adapter_checkpoint,
        register_adapter_async_finalize,
        validate_lora_optimizer_state,
    )

    args = get_args()
    if should_disable_forward_pre_hook(args):
        disable_forward_pre_hook(model)

    def _do_save(*, non_persistent_ckpt=False):
        save_checkpoint(
            iteration,
            model,
            optimizer,
            opt_param_scheduler,
            num_floating_point_operations_so_far=0,
            checkpointing_context=None,
            train_data_iterator=None,
            preprocess_common_state_dict_fn=None,
            non_persistent_ckpt=non_persistent_ckpt,
        )

    if adapter_only_ckpt_enabled(args):
        # LoRA-only save: filter the model state dict to adapter params (frozen
        # base reloads cold from --load on resume). Always logs full-vs-kept byte
        # sizes so the adapter-only shrink is visible.
        optimizer_stats = None
        if not args.no_save_optim:
            optimizer_stats = validate_lora_optimizer_state(model, optimizer)
        if not args.save:
            raise RuntimeError("adapter-only checkpointing requires args.save")
        final_save_dir = args.save
        saved_optimizer = not args.no_save_optim
        saved_rng = not args.no_save_rng
        # Upstream Megatron writes latest_checkpointed_iteration.txt before our
        # node-local files can be unioned. Route the save to a hidden
        # non-persistent-global root, replicate and validate there, atomically
        # promote the iteration on each node, and publish final latest markers
        # only after every node has committed.  With --async-save, the post-write
        # work is registered on Megatron's AsyncRequest and therefore runs only
        # after all distcp writer processes have completed.
        with adapter_checkpoint_staging(args, iteration) as staging_dir:

            def _finalize_adapter_save():
                finalize_adapter_checkpoint(
                    final_save_dir,
                    staging_dir,
                    iteration,
                    model,
                    optimizer_stats=optimizer_stats,
                    saved_optimizer=saved_optimizer,
                    saved_rng=saved_rng,
                    max_node_bytes=args.lora_checkpoint_max_node_bytes,
                    rslora=args.lora_rslora,
                )

            with adapter_only_model_save(model):
                if args.async_save:
                    with register_adapter_async_finalize(_finalize_adapter_save):
                        _do_save(non_persistent_ckpt=True)
                else:
                    _do_save(non_persistent_ckpt=True)
        if args.async_save:
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                logger.info(
                    "adapter checkpoint SCHEDULED: iteration=%s staging_dir=%s",
                    iteration,
                    staging_dir,
                )
        else:
            _finalize_adapter_save()
    else:
        _do_save()

    if should_disable_forward_pre_hook(args):
        enable_forward_pre_hook(model)


def initialize_model_and_optimizer(
    args: Namespace, role: str = "actor"
) -> tuple[list[DDP], MegatronOptimizer | None, OptimizerParamScheduler | None, int]:
    """Initialize model(s), optimizer, scheduler, and load from checkpoint.

    Args:
        args (Namespace): Runtime arguments.
        role (str): Logical role of the model (e.g., "actor", "critic").

    Returns:
        tuple[list[DDP], MegatronOptimizer | None, OptimizerParamScheduler | None, int]:
            DDP-wrapped model chunks, optimizer, scheduler, and iteration index.
    """

    if torch.version.hip:
        import megatron.core.dist_checkpointing.strategies.filesystem_async as filesystem_async_module

        from slime.utils.rocm_checkpoint_writer import ROCmFileSystemWriterAsync

        filesystem_async_module.FileSystemWriterAsync = ROCmFileSystemWriterAsync
        print("[ROCm] Applied FileSystemWriterAsync patch for HIP compatibility")

    model, optimizer, opt_param_scheduler = setup_model_and_optimizer(args, role)
    model[0].role = role
    reinit_critic_output_layer = _critic_output_layer_needs_reinit(args, model, role)
    clear_memory()
    iteration, _ = load_checkpoint(
        model,
        optimizer,
        opt_param_scheduler,
        checkpointing_context={},
    )
    if reinit_critic_output_layer:
        _reinitialize_critic_output_layer(args, model)
        if (args.fp16 or args.bf16) and optimizer is not None:
            optimizer.reload_model_params()
    clear_memory()

    _maybe_verify_v4_frozen_experts_fp4(model)

    return model, optimizer, opt_param_scheduler, iteration


def _maybe_verify_v4_frozen_experts_fp4(model):
    """Packed-MXFP4 frozen experts (V4_FP4_FROZEN_EXPERTS=1): the OFFICIAL
    checkpoint's packed expert bytes were loaded VERBATIM into uint8 buffers by
    the torch_dist load (no post-load quantization step exists in this mode).
    Verify the buffers were actually populated — a checkpoint-family mixup leaves
    the zero-init scales in place, which would silently make every expert a
    near-no-op. The packed buffers never enter adapter saves, so full
    saves/ref/teacher loads remain forbidden."""
    import os

    if os.environ.get("V4_FP4_FROZEN_EXPERTS", "0") != "1":
        return
    n = 0
    for chunk in model:
        for module in chunk.modules():
            if hasattr(module, "verify_fp4_loaded") and getattr(module, "_experts_fp4", False):
                module.verify_fp4_loaded()
                n += 1
    if n == 0:
        raise RuntimeError(
            "V4_FP4_FROZEN_EXPERTS=1 but no packed-MXFP4 V4GroupedExperts modules "
            "were found — the model was built without the flag (it is read at "
            "module construction) or this is not a V4 model"
        )
    import torch

    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
    print(f"[V4_FP4_FROZEN_EXPERTS] rank={rank} verified {n} packed-MXFP4 expert modules", flush=True)
