"""CPU-only contracts for the V4 persistent async-checkpoint worker."""

import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

NUM_GPUS = 0

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _run_save_model(monkeypatch, *, force_sync, offload_train=False):
    import megatron.training.async_utils as async_utils

    import slime.backends.megatron_utils.actor as actor_module

    events = []

    def finalize(**kwargs):
        events.append(("finalize", kwargs))

    def save(rollout_id, model, optimizer, opt_param_scheduler):
        events.append(
            (
                "save",
                rollout_id,
                model,
                optimizer,
                opt_param_scheduler,
            )
        )

    def barrier():
        events.append(("barrier",))

    def info(message, *args):
        events.append(("log", message % args if args else message))

    monkeypatch.setattr(async_utils, "maybe_finalize_async_save", finalize)
    monkeypatch.setattr(actor_module, "save", save)
    monkeypatch.setattr(actor_module.dist, "barrier", barrier)
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(actor_module.dist, "get_world_size", lambda: 16)
    monkeypatch.setattr(actor_module.logger, "info", info)
    monkeypatch.setattr(actor_module, "timer", lambda name: _recording_timer(events, name))

    def report_perf(iteration, _args, *names):
        events.append(("perf", iteration, names))

    monkeypatch.setattr(actor_module, "log_named_perf_timers", report_perf)

    actor = SimpleNamespace(
        args=SimpleNamespace(
            async_save=True,
            debug_rollout_only=False,
            offload_train=offload_train,
            save_hf=None,
        ),
        model=object(),
        optimizer=object(),
        opt_param_scheduler=object(),
        role="actor",
        wake_up=lambda: events.append(("wake_up",)),
        sleep=lambda: events.append(("sleep",)),
    )
    actor.finalize_async_save = MethodType(actor_module.MegatronTrainRayActor.finalize_async_save, actor)
    actor_module.MegatronTrainRayActor.prepare_save_model(actor)
    actor_module.MegatronTrainRayActor.save_model(
        actor,
        rollout_id=7,
        force_sync=force_sync,
    )
    actor_module.MegatronTrainRayActor.finish_save_model(actor, rollout_id=7)
    return events


@contextmanager
def _recording_timer(events, name):
    events.append(("timer_start", name))
    try:
        yield
    finally:
        events.append(("timer_end", name))


def _make_train_actor(actor_module, *, offload_train, events):
    train_active = {"value": False}
    caller_thread = threading.get_ident()

    def train_actor(rollout_id, rollout_data, external_data=None):
        train_active["value"] = True
        events.append(("train", rollout_id, threading.get_ident()))
        train_active["value"] = False

    actor = SimpleNamespace(
        args=SimpleNamespace(
            async_save=True,
            debug_rollout_only=False,
            offload_train=offload_train,
        ),
        role="actor",
        wake_up=lambda: events.append(("wake_up",)),
        _get_rollout_data=lambda rollout_data_ref: {"payload": rollout_data_ref},
        train_actor=train_actor,
        sleep=lambda: events.append(("sleep",)),
    )
    actor.finalize_async_save = MethodType(actor_module.MegatronTrainRayActor.finalize_async_save, actor)
    return actor, train_active, caller_thread


def test_explicit_finalize_is_blocking_timed_and_runs_on_calling_thread(monkeypatch):
    import megatron.training.async_utils as async_utils

    import slime.backends.megatron_utils.actor as actor_module

    events = []
    caller_thread = threading.get_ident()

    def finalize(**kwargs):
        events.append(("finalize", kwargs, threading.get_ident()))

    monkeypatch.setattr(async_utils, "maybe_finalize_async_save", finalize)
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(actor_module.logger, "info", lambda message, *args: events.append(("log", message % args)))
    monkeypatch.setattr(actor_module, "timer", lambda name: _recording_timer(events, name))
    monkeypatch.setattr(
        actor_module,
        "log_named_perf_timers",
        lambda iteration, _args, *names: events.append(("perf", iteration, names)),
    )

    actor = SimpleNamespace(args=SimpleNamespace(async_save=True, debug_rollout_only=False))
    actor_module.MegatronTrainRayActor.finalize_async_save(actor, iteration=11)

    assert events == [
        ("log", "V4_ASYNC_SAVE_FINALIZE_START iteration=11"),
        ("timer_start", "async_save_finalize"),
        ("finalize", {"blocking": True}, caller_thread),
        ("timer_end", "async_save_finalize"),
        ("log", "V4_ASYNC_SAVE_FINALIZE_END iteration=11"),
        ("perf", 11, ("async_save_finalize",)),
    ]


def test_actor_group_async_finalize_returns_refs_without_waiting():
    from slime.ray.actor_group import RayTrainGroup

    calls = []

    class RemoteFinalize:
        def __init__(self, rank):
            self.rank = rank

        def remote(self, iteration, terminate=False, wake_if_offloaded=False):
            calls.append((self.rank, iteration, terminate, wake_if_offloaded))
            return f"ref-{self.rank}-{iteration}"

    group = RayTrainGroup.__new__(RayTrainGroup)
    group._actor_handlers = [SimpleNamespace(finalize_async_save=RemoteFinalize(rank)) for rank in range(3)]

    refs = group.async_finalize_async_save(iteration=12)

    assert refs == ["ref-0-12", "ref-1-12", "ref-2-12"]
    assert calls == [(0, 12, False, False), (1, 12, False, False), (2, 12, False, False)]


@pytest.mark.parametrize("wake_fails", (False, True))
def test_offload_group_waits_for_every_wake_before_checkpoint_dispatch(monkeypatch, wake_fails):
    import slime.ray.actor_group as actor_group_module
    from slime.ray.actor_group import RayTrainGroup

    events = []

    class _Ref:
        def __init__(self, event, error=None):
            self.event = event
            self.error = error

    def fake_get(refs):
        for ref in refs:
            events.append(ref.event)
            if ref.error is not None:
                raise ref.error
        return [None] * len(refs)

    class _Prepare:
        def __init__(self, rank):
            self.rank = rank

        def remote(self):
            events.append(f"wake-dispatch:{self.rank}")
            error = RuntimeError("wake failed") if wake_fails and self.rank == 1 else None
            return _Ref(f"wake-wait:{self.rank}", error=error)

    class _Save:
        def __init__(self, rank):
            self.rank = rank

        def remote(self, rollout_id, force_sync=False):
            events.append(f"save-dispatch:{self.rank}:{rollout_id}:{force_sync}")
            return _Ref(f"save-wait:{self.rank}")

    monkeypatch.setattr(actor_group_module.ray, "get", fake_get)
    group = RayTrainGroup.__new__(RayTrainGroup)
    group.args = SimpleNamespace(offload_train=True)
    group._actor_handlers = [
        SimpleNamespace(prepare_save_model=_Prepare(rank), save_model=_Save(rank)) for rank in range(3)
    ]

    if wake_fails:
        with pytest.raises(RuntimeError, match="wake failed"):
            group.save_model(rollout_id=9, force_sync=False)
        assert not any(event.startswith("save-dispatch") for event in events)
    else:
        group.save_model(rollout_id=9, force_sync=False)
        last_wake_wait = max(index for index, event in enumerate(events) if event.startswith("wake-wait"))
        first_save_dispatch = min(index for index, event in enumerate(events) if event.startswith("save-dispatch"))
        assert last_wake_wait < first_save_dispatch


def test_resident_train_has_no_implicit_async_finalize(monkeypatch):
    import slime.backends.megatron_utils.actor as actor_module

    events = []
    actor, _, caller_thread = _make_train_actor(actor_module, offload_train=False, events=events)
    actor.finalize_async_save = lambda iteration: pytest.fail("resident train must not finalize implicitly")
    monkeypatch.setattr(actor_module, "timer", lambda name: _recording_timer(events, name))

    result = actor_module.MegatronTrainRayActor.train(actor, rollout_id=13, rollout_data_ref="batch")

    assert result is None
    assert ("train", 13, caller_thread) in events
    assert not any(event[0] == "sleep" for event in events)


def test_offload_train_blocking_finalize_finishes_after_train_and_before_sleep(monkeypatch):
    import megatron.training.async_utils as async_utils

    import slime.backends.megatron_utils.actor as actor_module

    events = []
    actor, train_active, caller_thread = _make_train_actor(actor_module, offload_train=True, events=events)

    def finalize(**kwargs):
        assert not train_active["value"], "async-save collective overlapped train"
        events.append(("finalize", kwargs, threading.get_ident()))

    monkeypatch.setattr(async_utils, "maybe_finalize_async_save", finalize)
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(actor_module, "timer", lambda name: _recording_timer(events, name))
    monkeypatch.setattr(actor_module, "log_named_perf_timers", lambda *_args: None)

    actor_module.MegatronTrainRayActor.train(actor, rollout_id=14, rollout_data_ref="batch")

    train_index = events.index(("train", 14, caller_thread))
    finalize_index = events.index(("finalize", {"blocking": True}, caller_thread))
    sleep_index = events.index(("sleep",))
    assert train_index < finalize_index < sleep_index


def test_nonfinal_async_save_drains_previous_request_without_terminating(monkeypatch):
    events = _run_save_model(monkeypatch, force_sync=False)

    assert events[0:2] == [
        ("timer_start", "save_model"),
        ("finalize", {"blocking": True}),
    ]
    assert events[2][0:2] == ("save", 7)
    assert events[3:] == [
        ("timer_end", "save_model"),
        ("perf", 7, ("save_model",)),
    ]


def test_final_async_save_flushes_new_request_and_terminates_worker(monkeypatch):
    events = _run_save_model(monkeypatch, force_sync=True)

    assert events[0:2] == [
        ("timer_start", "save_model"),
        ("finalize", {"blocking": True}),
    ]
    assert events[2][0:2] == ("save", 7)
    assert events[3:7] == [
        ("timer_end", "save_model"),
        ("perf", 7, ("save_model",)),
        ("log", "V4_ASYNC_SAVE_FINALIZE_START iteration=7"),
        ("timer_start", "async_save_finalize"),
    ]
    assert events[7] == (
        "finalize",
        {"blocking": True, "terminate": True},
    )
    assert events[8:] == [
        ("barrier",),
        ("timer_end", "async_save_finalize"),
        ("log", "V4_ASYNC_SAVE_WORKERS_TERMINATED iteration=7 world_size=16"),
        ("log", "V4_ASYNC_SAVE_FINALIZE_END iteration=7"),
        ("perf", 7, ("async_save_finalize",)),
    ]


def test_offload_save_finalizes_new_request_before_sleep(monkeypatch):
    events = _run_save_model(monkeypatch, force_sync=False, offload_train=True)

    assert events[0:3] == [
        ("wake_up",),
        ("timer_start", "save_model"),
        ("finalize", {"blocking": True}),
    ]
    assert events[3][0:2] == ("save", 7)
    assert events[4:] == [
        ("timer_end", "save_model"),
        ("perf", 7, ("save_model",)),
        ("log", "V4_ASYNC_SAVE_FINALIZE_START iteration=7"),
        ("timer_start", "async_save_finalize"),
        ("finalize", {"blocking": True}),
        ("timer_end", "async_save_finalize"),
        ("log", "V4_ASYNC_SAVE_FINALIZE_END iteration=7"),
        ("perf", 7, ("async_save_finalize",)),
        ("sleep",),
    ]


def test_offload_finish_sleeps_even_when_rank_local_hf_export_fails(monkeypatch):
    import slime.backends.megatron_utils.actor as actor_module
    import slime.backends.megatron_utils.model as model_module

    events = []
    monkeypatch.setattr(
        model_module,
        "save_hf_model",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("HF export failed")),
    )
    actor = SimpleNamespace(
        args=SimpleNamespace(save_hf="/tmp/not-written", offload_train=True),
        role="actor",
        model=object(),
        sleep=lambda: events.append(("sleep",)),
    )

    with pytest.raises(RuntimeError, match="HF export failed"):
        actor_module.MegatronTrainRayActor.finish_save_model(actor, rollout_id=7)

    assert events == [("sleep",)]


def test_checkpoint_perf_reporting_is_best_effort_and_selective(monkeypatch):
    from slime.utils import train_metric_utils
    from slime.utils.timer import Timer

    timer_instance = Timer()
    timer_instance.reset()
    timer_instance.add("save_model", 1.25)
    timer_instance.add("update_weights", 3.5)
    monkeypatch.setattr(
        train_metric_utils.logging_utils,
        "log",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("tracking failed")),
    )

    train_metric_utils.log_named_perf_timers_raw(
        rollout_id=7,
        args=SimpleNamespace(wandb_always_use_train_step=False),
        is_primary_rank=True,
        timer_names=("save_model",),
    )

    assert "save_model" not in timer_instance.timers
    assert timer_instance.timers["update_weights"] == 3.5
    timer_instance.reset()


@pytest.mark.parametrize(
    "launcher",
    ("scripts/dsv4/_dsv4_launch_core.sh", "scripts/dsv4/train_smoke.sh"),
)
def test_async_save_launchers_enable_the_persistent_worker(launcher):
    source = (REPO / launcher).read_text()

    assert "SAVE_ARGS+=(--async-save --use-persistent-ckpt-worker)" in source
    assert "SAVE_ARGS+=(--async-save)\n" not in source


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
