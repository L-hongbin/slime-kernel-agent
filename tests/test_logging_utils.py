from types import SimpleNamespace

import pytest

from slime.observability import logging_utils

NUM_GPUS = 0


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _ActorHandle:
    def __init__(self, actor):
        self._actor = actor

    def __getattr__(self, name):
        attr = getattr(self._actor, name)
        if callable(attr):
            return _RemoteMethod(attr)
        return attr


class _RemoteClass:
    def __init__(self, ray, cls):
        self._ray = ray
        self._cls = cls
        self._name = None

    def options(self, name=None, **kwargs):
        self._name = name
        return self

    def remote(self, *args, **kwargs):
        handle = _ActorHandle(self._cls(*args, **kwargs))
        if self._name is not None:
            self._ray.actors[self._name] = handle
        return handle


class _FakeRay:
    def __init__(self):
        self.actors = {}
        self.killed = []

    def remote(self, *args, **kwargs):
        if args and isinstance(args[0], type):
            return _RemoteClass(self, args[0])

        def decorate(cls):
            return _RemoteClass(self, cls)

        return decorate

    def get(self, value):
        return value

    def get_actor(self, name):
        return self.actors[name]

    def kill(self, actor):
        self.killed.append(actor)

    def get_runtime_context(self):
        return SimpleNamespace(get_node_id=lambda: "node-1")


class _FakeWandb:
    def __init__(self):
        self.logged = []
        self.finished = 0
        self.run = object()

    def log(self, metrics):
        self.logged.append(metrics)

    def finish(self):
        self.finished += 1


@pytest.fixture
def reset_tracking_globals():
    old_actor = logging_utils._TRACKING_ACTOR
    old_owns = logging_utils._OWNS_TRACKING_ACTOR
    logging_utils._TRACKING_ACTOR = None
    logging_utils._OWNS_TRACKING_ACTOR = False
    try:
        yield
    finally:
        logging_utils._TRACKING_ACTOR = old_actor
        logging_utils._OWNS_TRACKING_ACTOR = old_owns


def _args(**overrides):
    values = {
        "use_wandb": True,
        "use_tensorboard": False,
        "wandb_centralized": True,
        "wandb_run_id": None,
        "tracking_actor_name": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.unit
def test_centralized_primary_initializes_single_tracking_actor(monkeypatch, reset_tracking_globals):
    fake_ray = _FakeRay()
    init_calls = []

    def init_primary(args):
        init_calls.append(args)
        args.wandb_run_id = "run-123"

    monkeypatch.setitem(__import__("sys").modules, "ray", fake_ray)
    monkeypatch.setattr(logging_utils.wandb_utils, "init_wandb_primary", init_primary)

    args = _args()
    logging_utils.init_tracking(args, primary=True)

    assert args.wandb_run_id == "run-123"
    assert args.tracking_actor_name.startswith("slime-tracking-")
    assert list(fake_ray.actors) == [args.tracking_actor_name]
    assert init_calls


@pytest.mark.unit
def test_centralized_secondary_does_not_init_wandb(monkeypatch, reset_tracking_globals):
    init_secondary_calls = []

    monkeypatch.setattr(
        logging_utils.wandb_utils,
        "init_wandb_secondary",
        lambda *args, **kwargs: init_secondary_calls.append(args),
    )

    logging_utils.init_tracking(_args(wandb_run_id="run-123", tracking_actor_name="tracker"), primary=False)

    assert init_secondary_calls == []
    assert logging_utils._TRACKING_ACTOR is None


@pytest.mark.unit
def test_centralized_log_is_forwarded_to_tracking_actor(monkeypatch, reset_tracking_globals):
    fake_ray = _FakeRay()
    fake_wandb = _FakeWandb()

    def init_primary(args):
        args.wandb_run_id = "run-123"

    monkeypatch.setitem(__import__("sys").modules, "ray", fake_ray)
    monkeypatch.setattr(logging_utils, "wandb", fake_wandb)
    monkeypatch.setattr(logging_utils.wandb_utils, "init_wandb_primary", init_primary)

    args = _args()
    logging_utils.init_tracking(args, primary=True)
    logging_utils.log(
        args,
        {"train/loss": 1.5, "train/pg_loss": 1.5, "train/step": 7},
        step_key="train/step",
    )

    assert fake_wandb.logged == [{"train/pg_loss": 1.5, "train/step": 7}]


@pytest.mark.unit
def test_redundant_tracking_metrics_are_filtered_without_mutating_payload(monkeypatch, reset_tracking_globals):
    fake_wandb = _FakeWandb()
    monkeypatch.setattr(logging_utils, "wandb", fake_wandb)

    redundant = {key: float(index) for index, key in enumerate(logging_utils._REDUNDANT_WANDB_METRICS)}
    metrics = {
        **redundant,
        "rollout/truncated": 0.25,
        "rollout/response_len/mean": 1024.0,
        "rollout/kernel/turn0/compilation": 0.8,
        "rollout/kernel/turn0/speedup": 1.2,
        "train/pg_loss": -0.1,
        "rollout/step": 3,
    }
    original = metrics.copy()

    logging_utils.log(_args(wandb_centralized=False), metrics, step_key="rollout/step")

    assert metrics == original
    assert fake_wandb.logged == [
        {
            "rollout/truncated": 0.25,
            "rollout/response_len/mean": 1024.0,
            "rollout/kernel/turn0/compilation": 0.8,
            "rollout/kernel/turn0/speedup": 1.2,
            "train/pg_loss": -0.1,
            "rollout/step": 3,
        }
    ]


@pytest.mark.unit
def test_redundant_wandb_filter_does_not_change_tensorboard_payload(monkeypatch, reset_tracking_globals):
    tensorboard_logs = []

    class _FakeTensorboardAdapter:
        def __init__(self, args):
            pass

        def log(self, data, step):
            tensorboard_logs.append((data, step))

    monkeypatch.setattr(logging_utils, "_TensorboardAdapter", _FakeTensorboardAdapter)
    metrics = {"train/loss": 1.5, "train/pg_loss": 1.5, "train/step": 7}

    logging_utils.log(
        _args(use_wandb=False, use_tensorboard=True, wandb_centralized=False),
        metrics,
        step_key="train/step",
    )

    assert tensorboard_logs == [({"train/loss": 1.5, "train/pg_loss": 1.5}, 7)]


@pytest.mark.unit
def test_exp_metrics_are_printed_and_sent_as_a_separate_tensorboard_payload(
    monkeypatch, reset_tracking_globals, caplog
):
    tensorboard_logs = []

    class _FakeTensorboardAdapter:
        def __init__(self, args):
            pass

        def log(self, data, step):
            tensorboard_logs.append((data, step))

    monkeypatch.setattr(logging_utils, "_TensorboardAdapter", _FakeTensorboardAdapter)
    caplog.set_level("INFO", logger=logging_utils.logger.name)

    logging_utils.log_exp_metrics(
        _args(use_wandb=False, use_tensorboard=True, wandb_centralized=False),
        {"exp/train/dppo/binary_tv/mean": 0.2},
        step_key="train/step",
        step=7,
        context="train 7",
    )

    assert tensorboard_logs == [({"exp/train/dppo/binary_tv/mean": 0.2}, 7)]
    assert "exp train 7" in caplog.text


@pytest.mark.unit
def test_exp_logger_rejects_regular_metrics(reset_tracking_globals):
    with pytest.raises(ValueError, match="non-exp keys"):
        logging_utils.log_exp_metrics(
            _args(use_wandb=False, use_tensorboard=False),
            {"train/loss": 1.0},
            step_key="train/step",
            step=1,
            context="train 1",
        )


@pytest.mark.unit
def test_redundant_tracking_metric_inventory_is_exact():
    assert logging_utils._REDUNDANT_WANDB_METRICS == {
        "lora/lora_adapter/bytes",
        "lora/lora_adapter/num_tensors",
        "lora/lora_adapter/rank",
        "perf/effective_tokens_per_gpu_per_sec",
        "perf/longest_effective_sample_tokens_per_sec",
        "rollout/coverage/num_coverage/max",
        "rollout/coverage/num_coverage/min",
        "rollout/coverage/time_coverage/max",
        "rollout/coverage/time_coverage/min",
        "rollout/env_extra_info/compilation/mean",
        "rollout/env_extra_info/speedup/mean",
        "rollout/env_extra_info/speedup/min",
        "rollout/kernel/time/env_time/sum",
        "rollout/kernel/time/model_time/sum",
        "rollout/kl",
        "rollout/response_lengths",
        "rollout/returns",
        "rollout/truncated_ratio",
        "rollout/turn_indices",
        "train/loss",
    }


@pytest.mark.unit
def test_centralized_finish_only_kills_from_owner(monkeypatch, reset_tracking_globals):
    fake_ray = _FakeRay()
    fake_wandb = _FakeWandb()

    def init_primary(args):
        args.wandb_run_id = "run-123"

    monkeypatch.setitem(__import__("sys").modules, "ray", fake_ray)
    monkeypatch.setattr(logging_utils, "wandb", fake_wandb)
    monkeypatch.setattr(logging_utils.wandb_utils, "init_wandb_primary", init_primary)

    owner_args = _args()
    logging_utils.init_tracking(owner_args, primary=True)
    actor = logging_utils._TRACKING_ACTOR

    logging_utils._OWNS_TRACKING_ACTOR = False
    logging_utils.finish_tracking(_args(wandb_run_id="run-123", tracking_actor_name=owner_args.tracking_actor_name))
    assert fake_ray.killed == []
    assert fake_wandb.finished == 0

    logging_utils._OWNS_TRACKING_ACTOR = True
    logging_utils._TRACKING_ACTOR = actor
    logging_utils.finish_tracking(owner_args)
    assert fake_ray.killed == [actor]
    assert fake_wandb.finished == 1


@pytest.mark.unit
def test_centralized_tracking_requires_explicit_flag(monkeypatch, reset_tracking_globals):
    init_primary_calls = []

    monkeypatch.setattr(
        logging_utils.wandb_utils,
        "init_wandb_primary",
        lambda *args, **kwargs: init_primary_calls.append(args),
    )

    logging_utils.init_tracking(_args(wandb_centralized=False), primary=True)

    assert len(init_primary_calls) == 1
    assert logging_utils._TRACKING_ACTOR is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
