import copy
import logging
import socket
from typing import Any

import ray
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .actor_group import RayTrainGroup
from .utils import add_default_ray_env_vars

logger = logging.getLogger(__name__)


@ray.remote(num_gpus=1)
class InfoActor:
    def get_ip_and_gpu_id(self):
        return ray.util.get_node_ip_address(), ray.get_gpu_ids()[0]


def sort_key(x):
    index, node_identifier, gpu_id = x
    # Sort by node IP number and then by GPU ID
    try:
        # try to parse it as an IP address.
        ip_address = node_identifier
        node_ip_parts = list(map(int, ip_address.split(".")))
    except ValueError:
        # Try to resolve the hostname to an IP address.
        try:
            ip_address = socket.gethostbyname(node_identifier)
            node_ip_parts = list(map(int, ip_address.split(".")))
        except (socket.gaierror, TypeError):
            # Instead, we convert each character of the original identifier string
            # to its ASCII value. This provides a stable and consistent numerical
            # representation that allows for sorting.
            node_ip_parts = [ord(c) for c in node_identifier]

    return (node_ip_parts, int(gpu_id))


def _make_gpu_bundles(num_gpus: int, resource_name: str | None = None) -> list[dict[str, float]]:
    bundles = []
    for _ in range(num_gpus):
        bundle: dict[str, float] = {"GPU": 1, "CPU": 1}
        if resource_name:
            bundle[resource_name] = 1
        bundles.append(bundle)
    return bundles


def _empty_placement_group() -> tuple[Any, list[int], list[int]]:
    return None, [], []


def _create_placement_group(num_gpus, *, role: str = "", resource_name: str | None = None):
    """Create a placement group with the specified number of GPUs."""
    if num_gpus == 0:
        return _empty_placement_group()

    bundles = _make_gpu_bundles(num_gpus, resource_name=resource_name)
    pg = placement_group(bundles, strategy="PACK")
    num_bundles = len(bundles)

    # Wait for the placement group to be scheduled. Poll rather than a bare
    # ray.get(pg.ready()) so the wait is observable: when it can't be placed yet
    # (a node's GPUs haven't registered with the GCS, or an autoscaler is still
    # bringing nodes up) log the GPU counts periodically instead of hanging with no
    # output. The wait stays unbounded, so autoscaling clusters — where a pending
    # placement group is what drives scale-up — are unaffected.
    ready_ref = pg.ready()
    elapsed = 0
    log_interval = 30
    while not ray.wait([ready_ref], timeout=log_interval)[0]:
        elapsed += log_interval
        total = ray.cluster_resources().get("GPU", 0)
        available = ray.available_resources().get("GPU", 0)
        logger.info(
            f"Waiting for placement group of {num_gpus} GPUs (elapsed {elapsed}s): "
            f"{total:g} GPUs registered with Ray, {available:g} available."
        )

    # use info actor to get the GPU id
    info_actors = []
    for i in range(num_bundles):
        info_actors.append(
            InfoActor.options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_bundle_index=i,
                ),
            ).remote()
        )
    gpu_ids = ray.get([actor.get_ip_and_gpu_id.remote() for actor in info_actors])
    for actor in info_actors:
        ray.kill(actor)

    bundle_infos = [(i, gpu_ids[i][0], gpu_ids[i][1]) for i in range(num_bundles)]
    sorted_bundle_infos = sorted(bundle_infos, key=sort_key)
    pg_reordered_bundle_indices = [info[0] for info in sorted_bundle_infos]
    # Map from logical index -> physical GPU ID
    pg_reordered_gpu_ids = [gpu_ids[info[0]][1] for info in sorted_bundle_infos]

    for i in range(num_bundles):
        actual_bundle_index = pg_reordered_bundle_indices[i]
        role_prefix = f"{role} " if role else ""
        logger.info(
            f"  {role_prefix}bundle {i:4}, actual_bundle_index: {actual_bundle_index:4}, "
            f"node: {gpu_ids[actual_bundle_index][0]}, gpu: {gpu_ids[actual_bundle_index][1]}"
        )

    return pg, pg_reordered_bundle_indices, pg_reordered_gpu_ids


def _get_placement_group_layout(args) -> tuple[int, int]:
    actor_num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node

    if args.debug_train_only:
        return actor_num_gpus, 0

    if args.rollout_external:
        if args.debug_rollout_only:
            return actor_num_gpus, 0
        return actor_num_gpus, actor_num_gpus

    if args.debug_rollout_only:
        return args.rollout_num_gpus, 0

    if args.colocate:
        return max(actor_num_gpus, args.rollout_num_gpus), 0

    return actor_num_gpus + args.rollout_num_gpus, actor_num_gpus


def _use_role_placement_resources(args) -> bool:
    return bool(getattr(args, "actor_placement_resource", None) or getattr(args, "rollout_placement_resource", None))


def _create_role_placement_groups(args, actor_num_gpus: int, rollout_num_gpus: int):
    if args.colocate:
        raise ValueError("--actor-placement-resource/--rollout-placement-resource do not support --colocate")

    actor_resource = getattr(args, "actor_placement_resource", None)
    rollout_resource = getattr(args, "rollout_placement_resource", None)

    if actor_num_gpus > 0:
        logger.info(
            "Creating actor placement group with %s GPUs%s...",
            actor_num_gpus,
            f" requiring Ray resource {actor_resource!r}" if actor_resource else "",
        )
        actor_pg = _create_placement_group(actor_num_gpus, role="actor", resource_name=actor_resource)
    else:
        actor_pg = _empty_placement_group()

    if rollout_num_gpus > 0:
        logger.info(
            "Creating rollout placement group with %s GPUs%s...",
            rollout_num_gpus,
            f" requiring Ray resource {rollout_resource!r}" if rollout_resource else "",
        )
        rollout_pg = _create_placement_group(rollout_num_gpus, role="rollout", resource_name=rollout_resource)
    else:
        rollout_pg = _empty_placement_group()

    result = {
        "actor": actor_pg,
        "rollout": rollout_pg,
    }
    result["critic"] = result["actor"] if args.use_critic else None
    return result


def _get_role_placement_group_counts(args) -> tuple[int, int]:
    """Return separate actor/rollout GPU counts for resource-constrained placement."""
    actor_num_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node

    if args.debug_train_only:
        return actor_num_gpus, 0
    if getattr(args, "rollout_external", False):
        # External SGLang servers do not consume local Ray GPU bundles.
        return (0 if args.debug_rollout_only else actor_num_gpus), 0
    if args.debug_rollout_only:
        return 0, args.rollout_num_gpus
    if args.colocate:
        # Kept explicit for completeness; _create_role_placement_groups emits
        # the user-facing unsupported-mode error.
        return actor_num_gpus, actor_num_gpus
    return actor_num_gpus, args.rollout_num_gpus


def create_placement_groups(args):
    """Create placement groups for actor, critic, and rollout engines."""

    if _use_role_placement_resources(args):
        actor_num_gpus, rollout_num_gpus = _get_role_placement_group_counts(args)
        return _create_role_placement_groups(args, actor_num_gpus, rollout_num_gpus)

    num_gpus, rollout_offset = _get_placement_group_layout(args)
    logger.info(f"Creating placement group with {num_gpus} GPUs...")
    pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids = _create_placement_group(num_gpus)
    rollout_pg_reordered_bundle_indices = actor_pg_reordered_bundle_indices[rollout_offset:]
    rollout_pg_reordered_gpu_ids = actor_pg_reordered_gpu_ids[rollout_offset:]

    result = {
        "actor": (pg, actor_pg_reordered_bundle_indices, actor_pg_reordered_gpu_ids),
        "rollout": (pg, rollout_pg_reordered_bundle_indices, rollout_pg_reordered_gpu_ids),
    }

    result["critic"] = result["actor"] if args.use_critic else None

    return result


def _num_gpus_per_train_actor(args):
    return 0.4 if args.colocate else 1


def allocate_train_group(
    args,
    num_nodes,
    num_gpus_per_node,
    pg,
    role="actor",
    with_ref=False,
    with_opd_teacher=False,
    actor_cls=None,
):
    return RayTrainGroup(
        args=args,
        num_nodes=num_nodes,
        num_gpus_per_node=num_gpus_per_node,
        pg=pg,
        num_gpus_per_actor=_num_gpus_per_train_actor(args),
        role=role,
        with_ref=with_ref,
        with_opd_teacher=with_opd_teacher,
        actor_cls=actor_cls,
    )


def create_actor_model(args, pgs, rollout_manager, actor_cls=None):
    actor_args = args
    if args.megatron_config_path is not None:
        from slime.utils.arguments import parse_megatron_role_args

        actor_args = parse_megatron_role_args(args, args.megatron_config_path, role="actor")

    actor_model_kwargs = {}
    if actor_cls is not None:
        actor_model_kwargs["actor_cls"] = actor_cls
    actor_model = allocate_train_group(
        args=actor_args,
        num_nodes=args.actor_num_nodes,
        num_gpus_per_node=args.actor_num_gpus_per_node,
        pg=pgs["actor"],
        with_ref=actor_args.kl_coef != 0 or actor_args.use_kl_loss,
        with_opd_teacher=actor_args.use_opd and actor_args.opd_type == "megatron",
        **actor_model_kwargs,
    )
    actor_start_rollout_ids = actor_model.create(rollout_manager=rollout_manager)
    return actor_model, actor_start_rollout_ids


def create_training_models(args, pgs, rollout_manager, actor_cls=None):
    actor_model, actor_start_rollout_ids = create_actor_model(args, pgs, rollout_manager, actor_cls=actor_cls)

    critic_model = None
    if args.use_critic and args.num_rollout != 0:
        from slime.utils.arguments import parse_megatron_role_args

        critic_args = (
            parse_megatron_role_args(args, args.megatron_config_path, role="critic")
            if args.megatron_config_path is not None
            else copy.deepcopy(args)
        )
        if args.megatron_config_path is None:
            critic_args.disable_param_buffers_cpu_backup = False

        critic_model = allocate_train_group(
            args=critic_args,
            num_nodes=args.critic_num_nodes,
            num_gpus_per_node=args.critic_num_gpus_per_node,
            pg=pgs["critic"],
            role="critic",
        )
        critic_start_rollout_ids = critic_model.create(rollout_manager=rollout_manager)

    # TODO how to decide rollout start id when critic is involved? For now we just require user to specify it via args.
    if critic_model is not None:
        start_rollout_ids = critic_start_rollout_ids
    else:
        start_rollout_ids = actor_start_rollout_ids

    assert len(set(start_rollout_ids)) == 1

    if args.start_rollout_id is None:
        args.start_rollout_id = start_rollout_ids[0]

    if args.rollout_global_dataset and rollout_manager is not None:
        ray.get(rollout_manager.load.remote(args.start_rollout_id - 1))

    return actor_model, critic_model


def create_rollout_manager(args, pg):
    from .rollout import RolloutManager

    rollout_manager_options = {
        "num_cpus": 1,
        "num_gpus": 0,
        "runtime_env": {"env_vars": add_default_ray_env_vars()},
    }
    if getattr(args, "rollout_data_transport", "object-store") == "nixl":
        rollout_manager_options["enable_tensor_transport"] = True
    rollout_manager = RolloutManager.options(**rollout_manager_options).remote(args, pg)

    # calculate num_rollout from num_epoch
    num_rollout_per_epoch = None
    if args.num_rollout is None:
        num_rollout_per_epoch = ray.get(rollout_manager.get_num_rollout_per_epoch.remote())
        args.num_rollout = num_rollout_per_epoch * args.num_epoch
        assert args.num_rollout > 0

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="snapshot"))
        ray.get(rollout_manager.check_weights.remote(action="reset_tensors"))

    if args.offload_rollout:
        ray.get(rollout_manager.offload.remote())

    return rollout_manager, num_rollout_per_epoch
