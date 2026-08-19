"""CPU contracts for overlapping async checkpoint finalization with rollout."""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

NUM_GPUS = 0

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import train
import train_async


@dataclass
class _Ref:
    wait_event: str
    value: object = None
    dependencies: list = field(default_factory=list)
    error: BaseException | None = None


class _FakeRay:
    def __init__(self, events):
        self.events = events

    def get(self, value):
        if isinstance(value, list):
            return [self.get(item) for item in value]
        if isinstance(value, _Ref):
            for dependency in value.dependencies:
                self.get(dependency)
            self.events.append(value.wait_event)
            if value.error is not None:
                raise value.error
            return value.value
        return value


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeRolloutManager:
    def __init__(self, events, *, fail_metrics_router=False, fail_eval_at=None):
        self.events = events
        self.fail_metrics_router = fail_metrics_router
        self.fail_eval_at = fail_eval_at
        self.generate = _RemoteMethod(self._generate)
        self.save = _RemoteMethod(self._save)
        self.eval = _RemoteMethod(self._eval)
        self.dispose = _RemoteMethod(self._dispose)
        self.get_metrics_router_addr = _RemoteMethod(self._get_metrics_router_addr)

    def _generate(self, rollout_id, weight_version=None):
        suffix = "" if weight_version is None else f":weight-{weight_version}"
        self.events.append(f"rollout-dispatch:{rollout_id}{suffix}")
        return _Ref(f"rollout-wait:{rollout_id}", value=f"rollout-data-{rollout_id}")

    def _save(self, rollout_id):
        self.events.append(f"dataset-save:{rollout_id}")
        return None

    def _eval(self, rollout_id):
        self.events.append(f"eval:{rollout_id}")
        if rollout_id == self.fail_eval_at:
            raise RuntimeError(f"eval failed at {rollout_id}")
        return None

    def _dispose(self):
        self.events.append("dispose")
        return None

    def _get_metrics_router_addr(self):
        if self.fail_metrics_router:
            raise AssertionError("debug-train-only must not request a metrics router")
        return "router"


class _FakeModel:
    def __init__(self, name, events, *, fail_save_at=None):
        self.name = name
        self.events = events
        self.update_count = 0
        self.fail_save_at = fail_save_at

    def update_weights(self):
        self.update_count += 1
        self.events.append(f"{self.name}-update:{self.update_count}")

    def async_train(self, rollout_id, rollout_data_ref, external_data=None):
        self.events.append(f"{self.name}-train-dispatch:{rollout_id}")
        dependencies = list(external_data) if isinstance(external_data, list) else []
        return [
            _Ref(
                f"{self.name}-train-wait:{rollout_id}",
                value={"values": []} if self.name == "critic" else None,
                dependencies=dependencies,
            )
        ]

    def save_model(self, rollout_id, force_sync=False):
        self.events.append(f"{self.name}-save:{rollout_id}:force-{force_sync}")
        if rollout_id == self.fail_save_at:
            raise RuntimeError(f"{self.name} save failed at {rollout_id}")

    def finish_save_model(self, rollout_id):
        self.events.append(f"{self.name}-save-finish:{rollout_id}")

    def async_finalize_async_save(self, iteration, terminate=False, wake_if_offloaded=False):
        operation = "finalize-terminate" if terminate else "finalize"
        if wake_if_offloaded:
            operation += "-wake"
        self.events.append(f"{self.name}-{operation}-dispatch:{iteration}")
        return [_Ref(f"{self.name}-{operation}-wait:{iteration}")]

    def clear_memory(self):
        self.events.append(f"{self.name}-clear")


def _index(events, event):
    return events.index(event)


def _install_driver_runtime(
    monkeypatch,
    module,
    *,
    use_critic,
    fail_metrics_router=False,
    fail_eval_at=None,
    fail_actor_save_at=None,
):
    events = []
    fake_ray = _FakeRay(events)
    rollout_manager = _FakeRolloutManager(
        events,
        fail_metrics_router=fail_metrics_router,
        fail_eval_at=fail_eval_at,
    )
    actor_model = _FakeModel("actor", events, fail_save_at=fail_actor_save_at)
    critic_model = _FakeModel("critic", events) if use_critic else None

    monkeypatch.setattr(module, "ray", SimpleNamespace(get=fake_ray.get))
    monkeypatch.setattr(module, "configure_logger", lambda: None)
    monkeypatch.setattr(module, "init_tracking", lambda _args: None)
    monkeypatch.setattr(module, "finish_tracking", lambda _args: None)
    monkeypatch.setattr(module, "update_tracking_open_metrics", lambda _args, _addr: None)
    monkeypatch.setattr(
        module,
        "create_placement_groups",
        lambda _args: {"rollout": object(), "actor": object(), "critic": object()},
    )
    monkeypatch.setattr(
        module,
        "create_rollout_manager",
        lambda _args, _pg: (rollout_manager, None),
    )
    monkeypatch.setattr(
        module,
        "create_training_models",
        lambda _args, _pgs, _manager: (actor_model, critic_model),
    )
    return events, rollout_manager, actor_model, critic_model


def test_finalizer_dispatches_after_rollout_launch_and_waits_rollout_first():
    events = ["rollout-dispatch:8"]
    fake_ray = _FakeRay(events)
    actor = _FakeModel("actor", events)
    coordinator = train.AsyncCheckpointFinalizer(enabled=True, ray_get=fake_ray.get)
    coordinator.record_save(actor, 7, force_sync=False)

    result = coordinator.resolve_rollout(_Ref("rollout-wait:8", value="rollout-data-8"))

    assert result == "rollout-data-8"
    assert events == [
        "rollout-dispatch:8",
        "actor-finalize-dispatch:7",
        "rollout-wait:8",
        "actor-finalize-wait:7",
    ]
    coordinator.assert_requests_drained()


def test_rollout_failure_still_drains_dispatched_finalize():
    events = ["rollout-dispatch:8"]
    fake_ray = _FakeRay(events)
    actor = _FakeModel("actor", events)
    coordinator = train.AsyncCheckpointFinalizer(enabled=True, ray_get=fake_ray.get)
    coordinator.record_save(actor, 7, force_sync=False)
    rollout_error = RuntimeError("rollout failed")

    with pytest.raises(RuntimeError, match="rollout failed") as exc_info:
        coordinator.resolve_rollout(_Ref("rollout-wait:8", error=rollout_error))

    assert exc_info.value is rollout_error
    assert events == [
        "rollout-dispatch:8",
        "actor-finalize-dispatch:7",
        "rollout-wait:8",
        "actor-finalize-wait:7",
    ]
    coordinator.assert_requests_drained()


def test_dual_failure_preserves_rollout_error_and_finalize_cause():
    events = ["rollout-dispatch:8"]
    fake_ray = _FakeRay(events)
    finalize_error = ValueError("finalize failed")

    class _FailingFinalizeModel(_FakeModel):
        def async_finalize_async_save(self, iteration, terminate=False, wake_if_offloaded=False):
            assert not terminate
            assert not wake_if_offloaded
            self.events.append(f"{self.name}-finalize-dispatch:{iteration}")
            return [
                _Ref(
                    f"{self.name}-finalize-wait:{iteration}",
                    error=finalize_error,
                )
            ]

    actor = _FailingFinalizeModel("actor", events)
    coordinator = train.AsyncCheckpointFinalizer(enabled=True, ray_get=fake_ray.get)
    coordinator.record_save(actor, 7, force_sync=False)
    rollout_error = RuntimeError("rollout failed")

    with pytest.raises(RuntimeError, match="rollout failed") as exc_info:
        coordinator.resolve_rollout(_Ref("rollout-wait:8", error=rollout_error))

    assert exc_info.value is rollout_error
    assert exc_info.value.__cause__ is finalize_error
    assert "async checkpoint finalization also failed" in "\n".join(getattr(exc_info.value, "__notes__", []))
    coordinator.assert_requests_drained()


def test_partial_actor_critic_dispatch_cleanup_drains_both_and_terminates_workers():
    events = []
    fake_ray = _FakeRay(events)
    actor = _FakeModel("actor", events)

    class _CriticFailsFirstDispatch(_FakeModel):
        def async_finalize_async_save(self, iteration, terminate=False, wake_if_offloaded=False):
            if not terminate:
                self.events.append(f"{self.name}-finalize-dispatch:{iteration}")
                raise RuntimeError("critic dispatch failed")
            return super().async_finalize_async_save(
                iteration,
                terminate=terminate,
                wake_if_offloaded=wake_if_offloaded,
            )

    critic = _CriticFailsFirstDispatch("critic", events)
    coordinator = train.AsyncCheckpointFinalizer(enabled=True, ray_get=fake_ray.get)
    coordinator.record_save(actor, 7, force_sync=False)
    coordinator.record_save(critic, 7, force_sync=False)

    with pytest.raises(RuntimeError, match="critic dispatch failed"):
        coordinator.dispatch_during_rollout(_Ref("rollout-wait:8"))
    coordinator.drain_pending_now()

    assert events == [
        "actor-finalize-dispatch:7",
        "critic-finalize-dispatch:7",
        "actor-finalize-wait:7",
        "critic-finalize-terminate-dispatch:7",
        "critic-finalize-terminate-wait:7",
        "actor-finalize-terminate-dispatch:7",
        "actor-finalize-terminate-wait:7",
    ]
    coordinator.assert_idle()


def test_sync_driver_error_drains_and_terminates_pending_save(monkeypatch):
    events, _manager, _actor, _critic = _install_driver_runtime(
        monkeypatch,
        train,
        use_critic=False,
        fail_metrics_router=True,
        fail_eval_at=0,
    )
    args = SimpleNamespace(
        async_save=True,
        debug_train_only=True,
        debug_rollout_only=False,
        colocate=False,
        num_rollout=3,
        start_rollout_id=0,
        eval_interval=1,
        skip_eval_before_train=True,
        offload_rollout=False,
        offload_train=False,
        check_weight_update_equal=False,
        use_critic=False,
        num_critic_only_steps=0,
        rollout_global_dataset=False,
        save_interval=1,
    )

    with pytest.raises(RuntimeError, match="eval failed at 0"):
        train.train(args)

    assert _index(events, "actor-save:0:force-False") < _index(events, "eval:0")
    assert _index(events, "eval:0") < _index(events, "actor-finalize-terminate-dispatch:0")
    assert _index(events, "actor-finalize-terminate-dispatch:0") < _index(
        events,
        "actor-finalize-terminate-wait:0",
    )


def test_unconfirmed_group_save_failure_never_enters_finalize_collectives(monkeypatch):
    events, _manager, _actor, _critic = _install_driver_runtime(
        monkeypatch,
        train,
        use_critic=False,
        fail_metrics_router=True,
        fail_actor_save_at=0,
    )
    args = SimpleNamespace(
        async_save=True,
        debug_train_only=True,
        debug_rollout_only=False,
        colocate=False,
        num_rollout=3,
        start_rollout_id=0,
        eval_interval=None,
        skip_eval_before_train=False,
        offload_rollout=False,
        offload_train=False,
        check_weight_update_equal=False,
        use_critic=False,
        num_critic_only_steps=0,
        rollout_global_dataset=False,
        save_interval=1,
    )

    with pytest.raises(RuntimeError, match="actor save failed at 0"):
        train.train(args)

    assert "actor-save:0:force-False" in events
    assert not any("finalize" in event for event in events)


def test_offloaded_driver_error_wakes_actor_to_terminate_worker(monkeypatch):
    events, _manager, _actor, _critic = _install_driver_runtime(
        monkeypatch,
        train,
        use_critic=False,
        fail_metrics_router=True,
        fail_eval_at=0,
    )
    args = SimpleNamespace(
        async_save=True,
        debug_train_only=True,
        debug_rollout_only=False,
        colocate=False,
        num_rollout=3,
        start_rollout_id=0,
        eval_interval=1,
        skip_eval_before_train=True,
        offload_rollout=False,
        offload_train=True,
        check_weight_update_equal=False,
        use_critic=False,
        num_critic_only_steps=0,
        rollout_global_dataset=False,
        save_interval=1,
    )

    with pytest.raises(RuntimeError, match="eval failed at 0"):
        train.train(args)

    assert "actor-finalize-terminate-wake-dispatch:0" in events
    assert "actor-finalize-terminate-wake-wait:0" in events


def test_sync_driver_finalizes_actor_and_critic_during_next_rollout(monkeypatch):
    events, _manager, _actor, _critic = _install_driver_runtime(
        monkeypatch,
        train,
        use_critic=True,
        fail_metrics_router=True,
    )
    args = SimpleNamespace(
        async_save=True,
        debug_train_only=True,
        debug_rollout_only=False,
        colocate=False,
        num_rollout=4,
        start_rollout_id=0,
        eval_interval=None,
        skip_eval_before_train=False,
        offload_rollout=False,
        offload_train=False,
        check_weight_update_equal=False,
        use_critic=True,
        num_critic_only_steps=0,
        rollout_global_dataset=False,
        save_interval=2,
    )

    train.train(args)

    # save_interval=2 creates one non-final pending save at iteration 1. Both
    # model groups finalize only after rollout 2 has started, and both joins
    # finish before either group starts training iteration 2.
    assert _index(events, "rollout-dispatch:2") < _index(events, "actor-finalize-dispatch:1")
    assert _index(events, "actor-finalize-dispatch:1") < _index(events, "rollout-wait:2")
    assert _index(events, "critic-finalize-dispatch:1") < _index(events, "rollout-wait:2")
    assert _index(events, "rollout-wait:2") < _index(events, "actor-finalize-wait:1")
    assert _index(events, "actor-finalize-wait:1") < _index(events, "critic-train-dispatch:2")
    assert _index(events, "critic-finalize-wait:1") < _index(events, "critic-train-dispatch:2")

    assert "actor-save:0:force-False" not in events
    assert "critic-save:0:force-False" not in events
    assert "actor-save:3:force-True" in events
    assert "critic-save:3:force-True" in events
    assert not any(event.endswith("finalize-dispatch:3") for event in events)
    assert events[-1] == "dispose"


def test_sync_driver_does_not_dispatch_for_offloaded_training_actors(monkeypatch):
    events, _manager, _actor, _critic = _install_driver_runtime(
        monkeypatch,
        train,
        use_critic=False,
        fail_metrics_router=True,
    )
    args = SimpleNamespace(
        async_save=True,
        debug_train_only=True,
        debug_rollout_only=False,
        num_rollout=2,
        start_rollout_id=0,
        eval_interval=None,
        skip_eval_before_train=False,
        offload_rollout=False,
        offload_train=True,
        check_weight_update_equal=False,
        use_critic=False,
        num_critic_only_steps=0,
        rollout_global_dataset=False,
        save_interval=1,
    )

    train.train(args)

    # Offloaded actors join pending saves inside their next train call before
    # sleep() tears down process groups. The driver must not queue a duplicate
    # finalize call against an actor that is currently asleep.
    assert "actor-save:0:force-False" in events
    assert not any("finalize-dispatch" in event for event in events)
    assert events[-1] == "dispose"


def test_async_driver_joins_finalize_before_interval_one_update(monkeypatch):
    events, _manager, _actor, _critic = _install_driver_runtime(
        monkeypatch,
        train_async,
        use_critic=False,
        fail_metrics_router=True,
    )
    args = SimpleNamespace(
        async_save=True,
        debug_train_only=True,
        colocate=False,
        num_rollout=3,
        start_rollout_id=0,
        check_weight_update_equal=False,
        use_critic=False,
        num_critic_only_steps=0,
        rollout_global_dataset=False,
        save_interval=1,
        update_weights_interval=1,
        eval_interval=None,
    )

    train_async.train(args)

    # The next rollout is live before finalize dispatch. update interval 1 then
    # consumes that rollout, joins the checkpoint refs, and only then updates.
    assert _index(events, "rollout-dispatch:1:weight-1") < _index(events, "actor-finalize-dispatch:0")
    assert _index(events, "actor-finalize-dispatch:0") < _index(events, "rollout-wait:1")
    assert _index(events, "rollout-wait:1") < _index(events, "actor-finalize-wait:0")
    assert _index(events, "actor-finalize-wait:0") < _index(events, "actor-update:2")

    # interval 1 left no current future at the start of iteration 1. The driver
    # launches rollout 2 before it dispatches the newly scheduled save 1.
    assert _index(events, "rollout-dispatch:2:weight-2") < _index(events, "actor-finalize-dispatch:1")
    assert _index(events, "actor-finalize-wait:1") < _index(events, "actor-update:3")
    assert _index(events, "actor-finalize-wait:0") < _index(events, "actor-train-dispatch:1")

    # The final force-synchronous save is never registered as pending.
    assert "actor-save:2:force-True" in events
    assert "actor-finalize-dispatch:2" not in events
    assert events[-1] == "dispose"


def test_async_interval_two_launches_following_rollout_before_finalize_join(monkeypatch):
    events, _manager, _actor, _critic = _install_driver_runtime(
        monkeypatch,
        train_async,
        use_critic=False,
        fail_metrics_router=True,
    )
    args = SimpleNamespace(
        async_save=True,
        debug_train_only=True,
        colocate=False,
        num_rollout=4,
        start_rollout_id=0,
        check_weight_update_equal=False,
        use_critic=False,
        num_critic_only_steps=0,
        rollout_global_dataset=False,
        save_interval=1,
        update_weights_interval=2,
        eval_interval=None,
        offload_train=False,
    )

    train_async.train(args)

    # finalize(0) overlaps rollout1. If it outlives rollout1, rollout2 still
    # launches before the join, avoiding an interval>1 pipeline bubble.
    assert _index(events, "rollout-wait:1") < _index(events, "rollout-dispatch:2:weight-1")
    assert _index(events, "rollout-dispatch:2:weight-1") < _index(
        events,
        "actor-finalize-wait:0",
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
