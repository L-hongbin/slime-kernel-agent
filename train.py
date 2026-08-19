import logging
from contextlib import contextmanager

import ray

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger, finish_tracking, init_tracking, update_tracking_open_metrics
from slime.utils.misc import should_run_periodic_action


class AsyncCheckpointFinalizer:
    """Coordinate checkpoint finalization with an in-flight rollout.

    A save first enters ``_preparing`` while its all-rank RPC is unresolved.
    Only a uniformly successful return promotes it to ``_pending``; an
    unconfirmed/partial distributed save must never enter normal finalize
    collectives because Megatron requires identical async queues on every rank.
    Confirmed finalizers are dispatched only while a rollout future is in
    flight and are awaited before another training actor operation starts.

    When async saving is disabled, lifecycle methods are no-ops. Offloaded
    training disables rollout overlap but still tracks a confirmed persistent
    worker so exceptional driver exits can wake and terminate it safely.
    """

    def __init__(self, enabled, ray_get=None, *, track_workers=None, offload_train=False):
        self.enabled = enabled
        self.track_workers = enabled if track_workers is None else track_workers
        self.offload_train = offload_train
        self._ray_get = ray.get if ray_get is None else ray_get
        self._preparing = []
        self._pending = []
        self._inflight_refs = []
        self._active_models = []

    def _remember_active_model(self, model, iteration):
        for index, (active_model, _active_iteration) in enumerate(self._active_models):
            if active_model is model:
                self._active_models[index] = (model, iteration)
                return
        self._active_models.append((model, iteration))

    def _forget_active_model(self, model):
        self._active_models = [
            (active_model, iteration) for active_model, iteration in self._active_models if active_model is not model
        ]

    def begin_save(self, model, iteration, *, force_sync):
        """Register lifecycle ownership before dispatching save RPCs.

        The preparing state deliberately cannot be finalized. Megatron's async
        queue requires the same requests on every rank, which is known only
        after the group RPC returns successfully.
        """
        if not self.track_workers:
            return
        if self._inflight_refs:
            raise RuntimeError("cannot schedule another async save while finalization is in flight")
        if any(preparing_model is model for preparing_model, _ in self._preparing):
            raise RuntimeError("model already has a preparing async save")
        if any(pending_model is model for pending_model, _ in self._pending):
            raise RuntimeError("model already has a pending async save")
        self._preparing.append((model, iteration))

    def complete_save(self, model, iteration, *, force_sync):
        """Acknowledge a save RPC that returned successfully on every rank."""
        if not self.track_workers:
            return
        matched = any(
            preparing_model is model and preparing_iteration == iteration
            for preparing_model, preparing_iteration in self._preparing
        )
        if not matched:
            raise RuntimeError("completed async save was not in preparing state")
        self._preparing = [
            (preparing_model, preparing_iteration)
            for preparing_model, preparing_iteration in self._preparing
            if preparing_model is not model
        ]
        if force_sync:
            # save_model(force_sync=True) finalized and terminated the worker.
            self._forget_active_model(model)
            return
        self._remember_active_model(model, iteration)
        if self.enabled:
            self._pending.append((model, iteration))

    def cancel_save(self, model):
        """Forget an unconfirmed group save without entering finalize collectives."""
        self._preparing = [
            (preparing_model, preparing_iteration)
            for preparing_model, preparing_iteration in self._preparing
            if preparing_model is not model
        ]
        # A partial new request makes even an older live worker unsafe to close
        # via normal distributed finalize. Managed cluster teardown owns it.
        self._pending = [
            (pending_model, pending_iteration)
            for pending_model, pending_iteration in self._pending
            if pending_model is not model
        ]
        self._forget_active_model(model)

    def record_save(self, model, iteration, *, force_sync):
        """Compatibility helper for already-completed save calls and unit tests."""
        self.begin_save(model, iteration, force_sync=force_sync)
        self.complete_save(model, iteration, force_sync=force_sync)

    def dispatch_during_rollout(self, rollout_future):
        """Dispatch pending finalizers iff a rollout future is available."""
        if not self._pending or rollout_future is None:
            return False
        if self._inflight_refs:
            raise RuntimeError("async save finalization is already in flight")

        self._dispatch_pending()
        return True

    def _dispatch_pending(self):
        """Dispatch each pending model, retaining entries that did not submit."""
        while self._pending:
            model, iteration = self._pending[0]
            refs = model.async_finalize_async_save(iteration)
            # Pop only after the actor-group call successfully returned its
            # refs. If (for example) actor dispatch succeeds but critic dispatch
            # fails, the actor refs remain tracked and the critic stays pending.
            self._pending.pop(0)
            if refs:
                self._inflight_refs.extend(refs)

    def _wait_inflight(self):
        if not self._inflight_refs:
            return
        refs = self._inflight_refs
        # A failed finalize is terminal for this checkpoint request; clear the
        # coordinator state before propagating it so callers cannot accidentally
        # re-submit collectives that may already have run on some ranks.
        self._inflight_refs = []
        self._ray_get(refs)

    def wait_before_actor_operation(self):
        """Wait for dispatched finalizers before train/update/save collectives."""
        if self._pending:
            raise RuntimeError("pending async save was not dispatched during a rollout")
        self._wait_inflight()

    def resolve_rollout(self, rollout_future, *, wait_for_finalizers=True):
        """Resolve rollout and always drain finalizers if generation fails.

        The async driver can defer the successful-path wait until after it has
        launched the following rollout, preserving overlap when update interval
        is greater than one. Error paths always drain immediately.
        """
        self.dispatch_during_rollout(rollout_future)
        try:
            rollout_data_ref = self._ray_get(rollout_future)
        except BaseException as rollout_error:
            # Always collect checkpoint completion/errors even when generation
            # fails. If both sides fail, keep the rollout failure as the primary
            # exception while retaining the finalize failure as its cause.
            try:
                self.wait_before_actor_operation()
            except BaseException as finalize_error:
                if hasattr(rollout_error, "add_note"):
                    rollout_error.add_note(f"async checkpoint finalization also failed: {finalize_error!r}")
                raise rollout_error from finalize_error
            raise
        if wait_for_finalizers:
            self.wait_before_actor_operation()
        return rollout_data_ref

    def drain_pending_now(self):
        """Best-effort drain and worker termination for a driver error path."""
        cleanup_errors = []

        if self._preparing:
            # Never finalize an async queue whose all-rank enqueue consistency is
            # unknown. This is a fatal distributed-save path; cluster teardown
            # must abort the actors/workers instead.
            unconfirmed_models = [model for model, _iteration in self._preparing]
            self._preparing = []
            self._pending = [
                (model, iteration)
                for model, iteration in self._pending
                if not any(model is unconfirmed for unconfirmed in unconfirmed_models)
            ]
            self._active_models = [
                (model, iteration)
                for model, iteration in self._active_models
                if not any(model is unconfirmed for unconfirmed in unconfirmed_models)
            ]
            cleanup_errors.append(
                RuntimeError("unconfirmed distributed checkpoint was not finalized; cluster teardown required")
            )

        # Finish already-submitted actor groups, but continue with any critic or
        # other model left pending even if those refs report an error.
        try:
            self._wait_inflight()
        except BaseException as error:
            cleanup_errors.append(error)

        while self._pending:
            model, iteration = self._pending[0]
            try:
                refs = model.async_finalize_async_save(
                    iteration,
                    terminate=True,
                    wake_if_offloaded=self.offload_train,
                )
            except BaseException as error:
                cleanup_errors.append(error)
                self._pending.pop(0)
                continue

            self._pending.pop(0)
            self._forget_active_model(model)
            if refs:
                self._inflight_refs.extend(refs)
            try:
                self._wait_inflight()
            except BaseException as error:
                cleanup_errors.append(error)

        # Models whose finalize was already dispatched above still have a live
        # persistent worker. Terminate them even when another model failed.
        active_models = self._active_models
        self._active_models = []
        for model, iteration in active_models:
            try:
                refs = model.async_finalize_async_save(
                    iteration,
                    terminate=True,
                    wake_if_offloaded=self.offload_train,
                )
                if refs:
                    self._inflight_refs.extend(refs)
                self._wait_inflight()
            except BaseException as error:
                cleanup_errors.append(error)

        if cleanup_errors:
            first_error = cleanup_errors[0]
            if hasattr(first_error, "add_note"):
                for extra_error in cleanup_errors[1:]:
                    first_error.add_note(f"additional checkpoint cleanup failure: {extra_error!r}")
            raise first_error

    @contextmanager
    def drain_on_error(self):
        """Preserve a driver exception while collecting checkpoint completion."""
        try:
            yield
        except BaseException as driver_error:
            try:
                self.drain_pending_now()
            except BaseException as finalize_error:
                if hasattr(driver_error, "add_note"):
                    driver_error.add_note(f"async checkpoint cleanup also failed: {finalize_error!r}")
                raise driver_error from finalize_error
            raise

    def assert_requests_drained(self):
        if self._preparing or self._pending or self._inflight_refs:
            raise RuntimeError("async checkpoint finalization remains pending at driver shutdown")

    def assert_idle(self):
        self.assert_requests_drained()
        if self._active_models:
            raise RuntimeError("persistent async checkpoint worker remains active at driver shutdown")


def save_model_with_lifecycle(finalizer, model, iteration, *, force_sync):
    """Run checkpoint, promote only all-rank success, then do local postprocessing."""
    finalizer.begin_save(model, iteration, force_sync=force_sync)
    try:
        model.save_model(iteration, force_sync=force_sync)
    except BaseException:
        finalizer.cancel_save(model)
        raise
    finalizer.complete_save(model, iteration, force_sync=force_sync)
    model.finish_save_model(iteration)


def train(args):
    configure_logger()
    # allocate the GPUs
    logger = logging.getLogger(__name__)
    logger.info("train: creating placement groups")
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    logger.info("train: creating rollout manager")
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    if args.debug_train_only:
        logger.info("debug-train-only: skipping rollout metrics router setup")
    else:
        # Update primary W&B with SGLang metrics endpoint now that servers are up.
        logger.info("train: waiting for rollout metrics router address")
        router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
        logger.info("train: rollout metrics router address is %s", router_addr)
        update_tracking_open_metrics(args, router_addr)

    if args.debug_rollout_only:
        # Rollout-only debugging should not allocate Megatron actors. The
        # placement group only contains rollout GPU bundles in this mode.
        if args.num_rollout == 0 and args.eval_interval is not None:
            ray.get(rollout_manager.eval.remote(rollout_id=0))

        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
                logger.info("debug-rollout-only: starting eval rollout_id=%s", rollout_id)
                ray.get(rollout_manager.eval.remote(rollout_id))

            logger.info("debug-rollout-only: starting generate rollout_id=%s", rollout_id)
            ray.get(rollout_manager.generate.remote(rollout_id))
            logger.info("debug-rollout-only: finished generate rollout_id=%s", rollout_id)

            if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
                logger.info("debug-rollout-only: starting periodic eval rollout_id=%s", rollout_id)
                ray.get(rollout_manager.eval.remote(rollout_id))

        logger.info("debug-rollout-only: disposing rollout manager")
        ray.get(rollout_manager.dispose.remote())
        finish_tracking(args)
        return

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)
    checkpoint_finalizer = AsyncCheckpointFinalizer(
        # Offloaded actors must finalize inside train() before sleep destroys
        # their process groups; dispatching from the driver while they are
        # asleep would both duplicate that join and run without live groups.
        enabled=getattr(args, "async_save", False) and not getattr(args, "offload_train", False),
        ray_get=ray.get,
        track_workers=getattr(args, "async_save", False),
        offload_train=getattr(args, "offload_train", False),
    )

    if args.offload_rollout:
        ray.get(rollout_manager.onload_weights.remote())

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        ray.get(rollout_manager.onload_kv.remote())

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train(actor_trains_this_step):
        # Each model auto-offloads after train() when offload_train is set,
        # so we only need clear_memory for the non-offload case.
        if not args.offload_train:
            if not args.use_critic or actor_trains_this_step:
                actor_model.clear_memory()
            else:
                critic_model.clear_memory()

    def save(rollout_id):
        actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps
        force_sync = rollout_id == args.num_rollout - 1
        if actor_trains_this_step:
            save_model_with_lifecycle(
                checkpoint_finalizer,
                actor_model,
                rollout_id,
                force_sync=force_sync,
            )
        if args.use_critic:
            save_model_with_lifecycle(
                checkpoint_finalizer,
                critic_model,
                rollout_id,
                force_sync=force_sync,
            )
        if args.rollout_global_dataset:
            ray.get(rollout_manager.save.remote(rollout_id))

    # train loop.
    with checkpoint_finalizer.drain_on_error():
        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
                ray.get(rollout_manager.eval.remote(rollout_id))

            rollout_future = rollout_manager.generate.remote(rollout_id)
            rollout_data_ref = checkpoint_finalizer.resolve_rollout(rollout_future)

            if args.offload_rollout:
                ray.get(rollout_manager.offload.remote())

            actor_trains_this_step = (not args.use_critic) or rollout_id >= args.num_critic_only_steps

            if args.use_critic:
                value_refs = critic_model.async_train(rollout_id, rollout_data_ref)
                if actor_trains_this_step:
                    ray.get(actor_model.async_train(rollout_id, rollout_data_ref, external_data=value_refs))
                else:
                    ray.get(value_refs)
            else:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))

            if should_run_periodic_action(
                rollout_id,
                args.save_interval,
                num_rollout_per_epoch,
                args.num_rollout,
            ):
                save(rollout_id)

            offload_train(actor_trains_this_step)
            if args.offload_rollout:
                ray.get(rollout_manager.onload_weights.remote())
            actor_model.update_weights()

            if args.offload_rollout:
                ray.get(rollout_manager.onload_kv.remote())

            if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
                ray.get(rollout_manager.eval.remote(rollout_id))

        checkpoint_finalizer.assert_idle()
    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
