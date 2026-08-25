import importlib
from argparse import Namespace

import pytest


NUM_GPUS = 0


class _RemoteMethod:
    def __init__(self, func):
        self._func = func

    def remote(self, *args, **kwargs):
        return self._func(*args, **kwargs)


class _FakeRolloutManager:
    def __init__(self):
        self.calls = []
        self.get_metrics_router_addr = _RemoteMethod(lambda: "http://router")
        self.generate = _RemoteMethod(self._generate)
        self.eval = _RemoteMethod(self._eval)
        self.dispose = _RemoteMethod(self._dispose)

    def _generate(self, rollout_id):
        self.calls.append(("generate", rollout_id))

    def _eval(self, rollout_id):
        self.calls.append(("eval", rollout_id))

    def _dispose(self):
        self.calls.append(("dispose", None))


class _FakeActorModel:
    def __init__(self):
        self.calls = []

    def update_weights(self):
        self.calls.append(("update_weights", None))

    def async_train(self, rollout_id, rollout_data_ref, external_data=None):
        self.calls.append(("async_train", rollout_id, rollout_data_ref, external_data))
        return "train-ref"

    def clear_memory(self):
        self.calls.append(("clear_memory", None))

    def save_model(self, rollout_id, force_sync=False):
        self.calls.append(("save_model", rollout_id, force_sync))


def _args(**overrides):
    values = dict(
        colocate=False,
        debug_rollout_only=True,
        debug_train_only=False,
        num_rollout=2,
        start_rollout_id=0,
        eval_interval=None,
        skip_eval_before_train=False,
    )
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize("driver_module_name", ["train", "train_async"])
def test_debug_rollout_only_skips_training_model_allocation(monkeypatch, driver_module_name):
    train_module = importlib.import_module(driver_module_name)

    manager = _FakeRolloutManager()

    monkeypatch.setattr(train_module, "configure_logger", lambda: None)
    monkeypatch.setattr(train_module, "create_placement_groups", lambda args: {"rollout": object()})
    monkeypatch.setattr(train_module, "init_tracking", lambda args, primary=True, role=None: None)
    monkeypatch.setattr(train_module, "create_rollout_manager", lambda args, pg: (manager, None))
    monkeypatch.setattr(train_module, "update_tracking_open_metrics", lambda args, addr: None)
    monkeypatch.setattr(train_module, "finish_tracking", lambda args: None)
    monkeypatch.setattr(train_module.ray, "get", lambda value: value)

    def fail_create_training_models(*args, **kwargs):
        raise AssertionError("debug_rollout_only must not allocate training models")

    monkeypatch.setattr(train_module, "create_training_models", fail_create_training_models)

    train_module.train(_args())

    assert manager.calls == [
        ("generate", 0),
        ("generate", 1),
        ("dispose", None),
    ]


@pytest.mark.parametrize("driver_module_name", ["train", "train_async"])
def test_debug_rollout_only_preserves_eval_cadence(monkeypatch, driver_module_name):
    train_module = importlib.import_module(driver_module_name)

    manager = _FakeRolloutManager()

    monkeypatch.setattr(train_module, "configure_logger", lambda: None)
    monkeypatch.setattr(train_module, "create_placement_groups", lambda args: {"rollout": object()})
    monkeypatch.setattr(train_module, "init_tracking", lambda args, primary=True, role=None: None)
    monkeypatch.setattr(train_module, "create_rollout_manager", lambda args, pg: (manager, None))
    monkeypatch.setattr(train_module, "update_tracking_open_metrics", lambda args, addr: None)
    monkeypatch.setattr(train_module, "finish_tracking", lambda args: None)
    monkeypatch.setattr(train_module.ray, "get", lambda value: value)
    monkeypatch.setattr(train_module, "create_training_models", lambda *args, **kwargs: None)

    train_module.train(_args(num_rollout=1, eval_interval=1))

    assert manager.calls == [
        ("eval", 0),
        ("generate", 0),
        ("eval", 0),
        ("dispose", None),
    ]


def test_debug_train_only_skips_rollout_metrics_router(monkeypatch):
    import train as train_module

    manager = _FakeRolloutManager()
    actor = _FakeActorModel()

    def fail_metrics_addr():
        raise AssertionError("debug_train_only must not wait for rollout metrics router")

    manager.get_metrics_router_addr = _RemoteMethod(fail_metrics_addr)

    monkeypatch.setattr(train_module, "configure_logger", lambda: None)
    monkeypatch.setattr(
        train_module, "create_placement_groups", lambda args: {"rollout": object(), "actor": object(), "critic": None}
    )
    monkeypatch.setattr(train_module, "init_tracking", lambda args, primary=True, role=None: None)
    monkeypatch.setattr(train_module, "create_rollout_manager", lambda args, pg: (manager, None))
    monkeypatch.setattr(
        train_module,
        "update_tracking_open_metrics",
        lambda args, addr: (_ for _ in ()).throw(
            AssertionError("debug_train_only must not update rollout metrics endpoint")
        ),
    )
    monkeypatch.setattr(train_module, "create_training_models", lambda args, pgs, rollout_manager: (actor, None))
    monkeypatch.setattr(train_module, "finish_tracking", lambda args: None)
    monkeypatch.setattr(train_module.ray, "get", lambda value: value)

    train_module.train(
        _args(
            debug_rollout_only=False,
            debug_train_only=True,
            use_critic=False,
            offload_rollout=False,
            offload_train=False,
            check_weight_update_equal=False,
            rollout_global_dataset=False,
            save_interval=None,
            num_rollout=1,
        )
    )

    assert manager.calls == [
        ("generate", 0),
        ("dispose", None),
    ]
    assert actor.calls == [
        ("update_weights", None),
        ("async_train", 0, None, None),
        ("clear_memory", None),
        ("update_weights", None),
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
