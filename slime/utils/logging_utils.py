import logging
import os

import wandb

from . import wandb_utils
from .tensorboard_utils import _TensorboardAdapter

_LOGGER_CONFIGURED = False
_TRACKING_ACTOR = None
_OWNS_TRACKING_ACTOR = False


class _CentralTrackingActor:
    def __init__(self, args):
        self.args = args
        wandb_utils.init_wandb_primary(self.args)

    def get_wandb_run_id(self):
        return getattr(self.args, "wandb_run_id", None)

    def update_open_metrics(self, router_addr):
        wandb_utils.reinit_wandb_primary_with_open_metrics(self.args, router_addr)

    def log(self, metrics, step_key: str):
        if self.args.use_wandb:
            wandb.log(metrics)

        if self.args.use_tensorboard:
            metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
            _TensorboardAdapter(self.args).log(data=metrics_except_step, step=metrics[step_key])

    def finish(self):
        if self.args.use_wandb:
            try:
                if wandb.run is not None:
                    wandb.finish()
            except Exception:
                logging.getLogger(__name__).exception("Failed to finish wandb run")


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def init_tracking(args, primary: bool = True, **kwargs):
    if _use_centralized_tracking(args):
        if primary:
            _init_central_tracking(args)
        return

    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)


def update_tracking_open_metrics(args, router_addr):
    if _use_centralized_tracking(args):
        actor = _get_central_tracking_actor(args)
        if actor is not None:
            import ray

            ray.get(actor.update_open_metrics.remote(router_addr))
        return

    wandb_utils.reinit_wandb_primary_with_open_metrics(args, router_addr)


def finish_tracking(args):
    global _OWNS_TRACKING_ACTOR, _TRACKING_ACTOR

    if _use_centralized_tracking(args):
        if _OWNS_TRACKING_ACTOR and _TRACKING_ACTOR is not None:
            import ray

            ray.get(_TRACKING_ACTOR.finish.remote())
            ray.kill(_TRACKING_ACTOR)
            _TRACKING_ACTOR = None
            _OWNS_TRACKING_ACTOR = False
        return

    if not args.use_wandb:
        return
    try:
        if wandb.run is not None:
            wandb.finish()
    except Exception:
        logging.getLogger(__name__).exception("Failed to finish wandb run")


# TODO further refactor, e.g. put TensorBoard init to the "init" part
def log(args, metrics, step_key: str):
    if _use_centralized_tracking(args):
        actor = _get_central_tracking_actor(args)
        if actor is not None:
            import ray

            ray.get(actor.log.remote(metrics, step_key))
            return

    if args.use_wandb:
        wandb.log(metrics)

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])


def _use_centralized_tracking(args) -> bool:
    return bool(getattr(args, "wandb_centralized", False)) and (args.use_wandb or args.use_tensorboard)


def _init_central_tracking(args):
    global _OWNS_TRACKING_ACTOR, _TRACKING_ACTOR

    import ray

    if getattr(args, "tracking_actor_name", None) is None:
        args.tracking_actor_name = f"slime-tracking-{os.getpid()}"

    actor_options = {"name": args.tracking_actor_name}
    try:
        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        actor_options["scheduling_strategy"] = NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(),
            soft=False,
        )
    except Exception:
        logging.getLogger(__name__).warning("Failed to pin central tracking actor to the driver node.")

    TrackingActor = ray.remote(num_cpus=0)(_CentralTrackingActor)
    _TRACKING_ACTOR = TrackingActor.options(**actor_options).remote(args)
    _OWNS_TRACKING_ACTOR = True
    args.wandb_run_id = ray.get(_TRACKING_ACTOR.get_wandb_run_id.remote())


def _get_central_tracking_actor(args):
    global _TRACKING_ACTOR

    if _TRACKING_ACTOR is not None:
        return _TRACKING_ACTOR

    actor_name = getattr(args, "tracking_actor_name", None)
    if actor_name is None:
        return None

    import ray

    _TRACKING_ACTOR = ray.get_actor(actor_name)
    return _TRACKING_ACTOR
