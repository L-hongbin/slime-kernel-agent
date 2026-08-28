import ray
from train import AsyncCheckpointFinalizer, save_model_with_lifecycle

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.observability.logging_utils import configure_logger, finish_tracking, init_tracking, update_tracking_open_metrics
from slime.utils.misc import should_run_periodic_action


# The framework supports other asynchronous approaches such as fully async (which is shown in examples/full_async).
def train(args):
    assert not args.colocate, "Colocation is not supported for async training."
    configure_logger()
    release_train = getattr(args, "release_train", False)
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # Update primary W&B with SGLang metrics endpoint now that servers are up.
    # Debug-train-only uses replayed data and has no live rollout metrics router.
    if not getattr(args, "debug_train_only", False):
        router_addr = ray.get(rollout_manager.get_metrics_router_addr.remote())
        update_tracking_open_metrics(args, router_addr)

    if getattr(args, "debug_rollout_only", False):
        # Match the synchronous driver contract: rollout-only debugging owns
        # no actor placement bundles and must return before Megatron allocation.
        if args.num_rollout == 0 and args.eval_interval is not None:
            ray.get(rollout_manager.eval.remote(rollout_id=0))

        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            if args.eval_interval is not None and rollout_id == 0 and not args.skip_eval_before_train:
                ray.get(rollout_manager.eval.remote(rollout_id))

            ray.get(rollout_manager.generate.remote(rollout_id))

            if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
                ray.get(rollout_manager.eval.remote(rollout_id))

        ray.get(rollout_manager.dispose.remote())
        finish_tracking(args)
        return

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)
    checkpoint_finalizer = AsyncCheckpointFinalizer(
        # Offloaded actors finalize inside train() before destroying process
        # groups, so the resident-actor overlap coordinator must stay disabled.
        enabled=getattr(args, "async_save", False) and not getattr(args, "offload_train", False),
        ray_get=ray.get,
        track_workers=getattr(args, "async_save", False),
        offload_train=getattr(args, "offload_train", False),
    )

    # Always push actor weights to rollout once weights are loaded.
    actor_model.update_weights()

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    # async train loop. weight_version counts completed update_weights() calls
    # (1 = the initial post-load push above); each generate() is stamped with
    # the version its sampling runs under so per-sample weight staleness
    # (trained_version - generated_version) is measurable end to end.
    weight_version = 1
    # A load-only resume gate intentionally has an empty range. Do not create an
    # unconsumed generation future: its eventual exception would otherwise be
    # reported by Ray after checkpoint restoration has already succeeded.
    rollout_data_next_future = None
    if args.start_rollout_id < args.num_rollout:
        rollout_data_next_future = rollout_manager.generate.remote(
            args.start_rollout_id,
            weight_version=weight_version,
        )
    rollout_data_curr_ref = None
    with checkpoint_finalizer.drain_on_error():
        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            # Resolve the current generation but, on success, defer the
            # checkpoint join until after rollout N+1 is launched. On failure,
            # resolve_rollout still drains finalizers immediately.
            if rollout_data_next_future is not None:
                rollout_data_curr_ref = checkpoint_finalizer.resolve_rollout(
                    rollout_data_next_future,
                    wait_for_finalizers=False,
                )
                rollout_data_next_future = None

            save_this_rollout = release_train or should_run_periodic_action(
                rollout_id,
                args.save_interval,
                num_rollout_per_epoch,
                args.num_rollout,
            )
            if save_this_rollout and args.rollout_global_dataset:
                # Snapshot post-rollout-N state before generate(N+1) can advance
                # the shared data source. If train/save then fails, state N may
                # exist ahead of the latest model checkpoint, but resume derives
                # the requested state id from that model checkpoint and ignores
                # the ahead file. Retrying N overwrites the same logical state.
                ray.get(rollout_manager.save.remote(rollout_id))

            # Start the next rollout early, before joining a finalizer that was
            # already overlapped with the generation consumed above.
            if rollout_id + 1 < args.num_rollout:
                rollout_data_next_future = rollout_manager.generate.remote(
                    rollout_id + 1,
                    weight_version=weight_version,
                )

            # A prior save may already have been dispatched against the rollout
            # consumed above. If it could not be dispatched earlier, the next
            # rollout has now been launched, so dispatch it before blocking.
            # Both sides finish before the next actor train.
            checkpoint_finalizer.dispatch_during_rollout(rollout_data_next_future)
            checkpoint_finalizer.wait_before_actor_operation()

            if release_train:
                actor_model.create()

            if args.use_critic:
                actor_trains_this_step = rollout_id >= args.num_critic_only_steps
                value_refs = critic_model.async_train(rollout_id, rollout_data_curr_ref)
                if actor_trains_this_step:
                    ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref, external_data=value_refs))
                else:
                    ray.get(value_refs)
            else:
                ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))

            if save_this_rollout:
                force_sync = release_train or rollout_id == args.num_rollout - 1
                if (not args.use_critic) or rollout_id >= args.num_critic_only_steps:
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

                # Normally rollout N+1 was started before train N, so finalize N
                # can overlap it immediately. The final save is force-sync and
                # therefore does not register here.
                checkpoint_finalizer.dispatch_during_rollout(rollout_data_next_future)

            if release_train or (rollout_id + 1) % args.update_weights_interval == 0:
                # Sync generation before update_weights so a live rollout never
                # observes an in-place weight update.
                rollout_data_curr_ref = (
                    checkpoint_finalizer.resolve_rollout(x) if (x := rollout_data_next_future) is not None else None
                )
                rollout_data_next_future = None
                # resolve_rollout already joined dispatched refs. Keep the guard
                # for the no-future path and prohibit collective overlap.
                checkpoint_finalizer.wait_before_actor_operation()
                actor_model.update_weights()
                weight_version += 1

            if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
                ray.get(rollout_manager.eval.remote(rollout_id))

        checkpoint_finalizer.assert_idle()
    ray.get(rollout_manager.dispose.remote())
    finish_tracking(args)


if __name__ == "__main__":
    args = parse_args()
    train(args)
