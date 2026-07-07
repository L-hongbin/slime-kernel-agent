import logging
import os
import re
from pathlib import Path

from .path_bootstrap import ensure_megatron_lm_on_sys_path

ensure_megatron_lm_on_sys_path()

# TODO: may need to copy those 2 functions and do refactoring.
from megatron.training.checkpointing import load_checkpoint as _load_checkpoint_megatron
from megatron.training.checkpointing import save_checkpoint
from megatron.training.global_vars import get_args

from slime.utils import megatron_bridge_utils

try:
    # Here we patch out the `validate_non_overlapping_shards_metadata` in both functions
    # because it is really slow for large models with many shards.
    # TODO: find a less hacky way to do this.
    import torch.distributed as dist
    import torch.distributed._shard.sharding_spec as shard_spec
    from torch.distributed._shard.sharded_tensor import ShardedTensor
    from torch.distributed._shard.sharded_tensor.metadata import ShardedTensorMetadata
    from torch.distributed._shard.sharded_tensor.shard import Shard
    from torch.distributed._shard.sharded_tensor.utils import _parse_and_validate_remote_device
    from torch.distributed._shard.sharding_spec.api import EnumerableShardingSpec

    def __post_init__(self):
        pass

    EnumerableShardingSpec.__post_init__ = __post_init__

    @classmethod
    def _init_from_local_shards_and_global_metadata(  # type: ignore[override]
        cls,
        local_shards: list[Shard],
        sharded_tensor_metadata: ShardedTensorMetadata,
        process_group=None,
        init_rrefs=False,
        sharding_spec=None,
    ) -> ShardedTensor:
        """
        Initialize a ShardedTensor with local shards and a global
        ShardedTensorMetadata built on each rank.

        Warning: This API is experimental and subject to change. It does
                 not do cross rank validations, and fully rely on the user
                 for the correctness of sharded_tensor_metadata on each rank
        """
        process_group = cls._normalize_pg(process_group)
        current_rank = dist.get_rank()  # intentional to get global rank

        shards_metadata = sharded_tensor_metadata.shards_metadata

        local_shard_metadatas = []

        # collect local shard metadatas from the global sharded_tensor_metadata
        for shard_metadata in shards_metadata:  # type: ignore[attr-defined]
            rank, local_device = _parse_and_validate_remote_device(process_group, shard_metadata.placement)

            if current_rank == rank:
                local_shard_metadatas.append(shard_metadata)

        shards_metadata = sharded_tensor_metadata.shards_metadata
        tensor_properties = sharded_tensor_metadata.tensor_properties

        if sharding_spec is None:
            spec = shard_spec._infer_sharding_spec_from_shards_metadata(shards_metadata)
        else:
            spec = sharding_spec

        sharded_tensor = ShardedTensor.__new__(
            ShardedTensor,
            spec,
            sharded_tensor_metadata.size,
            dtype=tensor_properties.dtype,
            layout=tensor_properties.layout,
            pin_memory=tensor_properties.pin_memory,
            requires_grad=tensor_properties.requires_grad,
        )

        # done validation, add local_shards
        sharded_tensor._local_shards = local_shards
        sharded_tensor._prepare_init(process_group=process_group, init_rrefs=init_rrefs)

        # run post initialization, i.e. map registration, rpc initialization
        sharded_tensor._post_init()
        return sharded_tensor

    ShardedTensor._init_from_local_shards_and_global_metadata = _init_from_local_shards_and_global_metadata

except ImportError:
    pass

logger = logging.getLogger(__name__)


def _patch_chained_optimizer_synchronize_steps() -> None:
    """Make ChainedOptimizer._synchronize_steps tolerate stub sub-optimizers.

    Muon + LoRA builds a ChainedOptimizer where one sub-optimizer has no params
    and is a stub (``optimizer.optimizer is None``, ``is_stub_optimizer``). The
    upstream ``_synchronize_steps`` iterates ``optimizer.optimizer.param_groups``
    with no stub guard (unlike the rest of the class), so any save/load with
    optimizer state crashes with ``'NoneType' object has no attribute
    'param_groups'``. Skip stubs (they have no step to synchronize).
    """
    try:
        from megatron.core.optimizer.optimizer import ChainedOptimizer
    except Exception:
        return

    def _synchronize_steps(self):
        steps = []
        for optimizer in self.chained_optimizers:
            inner = getattr(optimizer, "optimizer", None)
            if inner is None:  # stub sub-optimizer (empty param groups)
                continue
            for param_group in inner.param_groups:
                if len(param_group["params"]) > 0 and "step" in param_group:
                    steps.append(param_group["step"])
        steps = list(set(steps))
        assert len(steps) <= 1, f"steps: {steps}"
        step = steps[0] if len(steps) == 1 else None
        for optimizer in self.chained_optimizers:
            inner = getattr(optimizer, "optimizer", None)
            if inner is None:
                continue
            for param_group in inner.param_groups:
                if len(param_group["params"]) > 0 and "step" in param_group:
                    param_group["step"] = step
        return step

    ChainedOptimizer._synchronize_steps = _synchronize_steps


def _patch_stub_optimizer_state_dict() -> None:
    """Make a stub sub-optimizer's (empty param group, ``optimizer is None``)
    state-dict methods no-op instead of dereferencing ``self.optimizer``.

    Muon + LoRA chains a real optimizer with a param-less stub. Megatron's
    ``Float16OptimizerWithFloat16Params.{state_dict,sharded_state_dict}`` call
    ``self.optimizer.state_dict()`` with no stub guard, crashing every optimizer
    save/load ("'NoneType' object has no attribute 'state_dict'"). A stub holds
    no optimizer state, so return an empty dict for it on both save and load.
    """
    try:
        from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params
    except Exception:
        return

    def _is_stub(self) -> bool:
        return getattr(self, "is_stub_optimizer", False) or getattr(self, "optimizer", None) is None

    for _name in ("state_dict", "sharded_state_dict"):
        _orig = getattr(Float16OptimizerWithFloat16Params, _name)

        def _make(orig):
            def _wrapped(self, *args, **kwargs):
                if _is_stub(self):
                    return {}
                return orig(self, *args, **kwargs)

            return _wrapped

        setattr(Float16OptimizerWithFloat16Params, _name, _make(_orig))


_patch_chained_optimizer_synchronize_steps()
_patch_stub_optimizer_state_dict()

__all__ = ["save_checkpoint"]


def load_checkpoint(ddp_model, optimizer, opt_param_scheduler, checkpointing_context, skip_load_to_model_and_opt):
    # ref: how megatron `load_checkpoint` gets directory
    args = get_args()
    load_path = args.load

    assert Path(load_path).exists() and _is_dir_nonempty(
        load_path
    ), f"{args.load=} does not exist or is an empty directory. Did you specify the wrong folder?"

    if _is_megatron_checkpoint(load_path):
        return _load_checkpoint_megatron(
            ddp_model=ddp_model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=skip_load_to_model_and_opt,
        )
    else:
        return _load_checkpoint_hf(
            ddp_model=ddp_model,
            optimizer=optimizer,
            args=args,
            load_path=load_path,
        )


def _is_megatron_checkpoint(path: str | Path) -> bool:
    return (Path(path) / "latest_checkpointed_iteration.txt").is_file() or bool(
        re.fullmatch(r"iter_\d{7}", Path(path).name)
    )


def _load_checkpoint_hf(ddp_model, optimizer, args, load_path: str):
    assert args.megatron_to_hf_mode == "bridge", "Only bridge mode is supported for loading HF checkpoint"
    from megatron.bridge import AutoBridge

    import slime_plugins.megatron_bridge  # noqa: F401

    logger.info(f"Load checkpoint from HuggingFace model into Megatron (path={load_path})")

    with megatron_bridge_utils.patch_megatron_model(ddp_model):
        bridge = megatron_bridge_utils.patch_auto_bridge_hf_config(
            AutoBridge.from_hf_pretrained(load_path, trust_remote_code=True)
        )
        bridge.load_hf_weights(ddp_model)

    # Copied from Megatron-core :: load_checkpoint (with simplifications)
    if (args.fp16 or args.bf16) and optimizer is not None:
        assert not args.load_main_params_from_ckpt
        optimizer.reload_model_params()

    # We can see `successfully loaded checkpoint from ... [ t 1/2, p 1/1 ] at iteration 0`
    # when loading Megatron, thus it is 0
    iteration = 0
    num_floating_point_operations_so_far = 0
    return iteration, num_floating_point_operations_so_far


def _is_dir_nonempty(path):
    with os.scandir(path) as it:
        return any(it)
