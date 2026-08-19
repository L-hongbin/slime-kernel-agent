"""CPU contracts for rollout dataset checkpoint loading and async save alignment."""

import argparse
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

NUM_GPUS = 0

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import train_async
from slime.rollout import data_source as data_source_module
from slime.rollout.data_source import RolloutDataSource
from slime.utils.arguments import get_slime_extra_args_provider

RECONSTRUCT = REPO / "scripts" / "dsv4" / "tools" / "reconstruct_r21_dataset_state.py"


def test_rollout_dataset_load_cli_defaults_to_none_and_accepts_path():
    parser = get_slime_extra_args_provider()(argparse.ArgumentParser())

    default_args = parser.parse_args(["--rollout-batch-size", "1"])
    assert default_args.rollout_dataset_load is None

    explicit_args = parser.parse_args(["--rollout-batch-size", "1", "--rollout-dataset-load", "/dataset-checkpoint"])
    assert explicit_args.rollout_dataset_load == "/dataset-checkpoint"


def _uninitialized_data_source(args):
    source = RolloutDataSource.__new__(RolloutDataSource)
    source.args = args
    source.sample_offset = 0
    source.epoch_id = 0
    source.sample_group_index = 0
    source.sample_index = 0
    source.metadata = {}
    source.dataset = None
    return source


@pytest.mark.parametrize(
    ("include_dataset_load", "dataset_load", "expected_root"),
    [
        (True, "/dataset-checkpoint", "/dataset-checkpoint"),
        (True, None, "/model-checkpoint"),
        (False, None, "/model-checkpoint"),
    ],
)
def test_data_source_load_uses_explicit_root_with_backward_compatible_fallback(
    monkeypatch,
    include_dataset_load,
    dataset_load,
    expected_root,
):
    args_dict = {
        "rollout_global_dataset": True,
        "rollout_shuffle": False,
        "load": "/model-checkpoint",
    }
    if include_dataset_load:
        args_dict["rollout_dataset_load"] = dataset_load
    source = _uninitialized_data_source(SimpleNamespace(**args_dict))

    loaded_paths = []
    monkeypatch.setattr(data_source_module.os.path, "exists", lambda _path: True)

    def fake_torch_load(path):
        loaded_paths.append(path)
        return {
            "sample_offset": 2640,
            "epoch_id": 0,
            "sample_group_index": 2640,
            "sample_index": 42240,
            "metadata": {"source": "iter59"},
        }

    monkeypatch.setattr(data_source_module.torch, "load", fake_torch_load)

    source.load(59)

    assert loaded_paths == [f"{expected_root}/rollout/global_dataset_state_dict_59.pt"]
    assert (source.sample_offset, source.sample_group_index, source.sample_index) == (2640, 2640, 42240)
    assert source.metadata == {"source": "iter59"}


def test_explicit_dataset_root_fails_loudly_when_state_is_missing(monkeypatch):
    source = _uninitialized_data_source(
        SimpleNamespace(
            rollout_global_dataset=True,
            rollout_shuffle=False,
            load="/model-checkpoint",
            rollout_dataset_load="/dataset-checkpoint",
        )
    )
    existing_model_state = "/model-checkpoint/rollout/global_dataset_state_dict_59.pt"
    loaded_paths = []
    monkeypatch.setattr(data_source_module.os.path, "exists", lambda path: path == existing_model_state)
    monkeypatch.setattr(data_source_module.torch, "load", lambda path: loaded_paths.append(path))

    with pytest.raises(FileNotFoundError, match="Refusing to reset the prompt cursor silently"):
        source.load(59)

    assert loaded_paths == []
    assert source.sample_offset == 0


def test_r21_iter59_dataset_state_reconstruction_is_exact_and_audited(tmp_path):
    source = tmp_path / "global_dataset_state_dict_54.pt"
    output = tmp_path / "resume" / "rollout" / "global_dataset_state_dict_59.pt"
    torch.save(
        {
            "sample_offset": 2512,
            "epoch_id": 0,
            "sample_group_index": 2512,
            "sample_index": 40192,
            "metadata": {},
        },
        source,
    )

    subprocess.run(
        [sys.executable, str(RECONSTRUCT), "--source-state54", str(source), "--output-state59", str(output)],
        check=True,
        text=True,
        capture_output=True,
    )
    assert torch.load(output, map_location="cpu", weights_only=False) == {
        "sample_offset": 2640,
        "epoch_id": 0,
        "sample_group_index": 2640,
        "sample_index": 42240,
        "metadata": {},
    }


def test_r21_dataset_reconstruction_refuses_an_unaudited_source(tmp_path):
    source = tmp_path / "wrong_state54.pt"
    output = tmp_path / "state59.pt"
    torch.save({"sample_offset": 0}, source)

    result = subprocess.run(
        [sys.executable, str(RECONSTRUCT), "--source-state54", str(source), "--output-state59", str(output)],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "differs from audited values" in result.stderr
    assert not output.exists()


class _RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class _FakeRolloutManager:
    def __init__(self, events):
        self.events = events
        self.dataset_state = None
        self.saved_states = []
        self.generate = _RemoteMethod(self._generate)
        self.save = _RemoteMethod(self._save)
        self.get_metrics_router_addr = _RemoteMethod(lambda: None)
        self.dispose = _RemoteMethod(lambda: self.events.append("dispose"))
        self.eval = _RemoteMethod(lambda rollout_id: self.events.append(f"eval:{rollout_id}"))

    def _generate(self, rollout_id, weight_version=None):
        self.dataset_state = f"post-rollout-{rollout_id}"
        self.events.append(f"generate:{rollout_id}:weight-{weight_version}")
        return f"rollout-data-{rollout_id}"

    def _save(self, rollout_id):
        snapshot = (rollout_id, self.dataset_state)
        self.saved_states.append(snapshot)
        self.events.append(f"dataset-save:{rollout_id}:{self.dataset_state}")


class _FakeActorModel:
    def __init__(self, events, fail_train_at=None):
        self.events = events
        self.fail_train_at = fail_train_at
        self.saved_models = []

    def update_weights(self):
        self.events.append("update-weights")

    def async_train(self, rollout_id, rollout_data_ref, external_data=None):
        self.events.append(f"train:{rollout_id}:{rollout_data_ref}")
        if rollout_id == self.fail_train_at:
            raise RuntimeError(f"injected train failure at {rollout_id}")
        return None

    def save_model(self, rollout_id, force_sync=False):
        self.saved_models.append(rollout_id)
        self.events.append(f"model-save:{rollout_id}")

    def finish_save_model(self, rollout_id):
        self.events.append(f"model-save-finish:{rollout_id}")


def _install_fake_async_runtime(
    monkeypatch,
    *,
    num_rollout=2,
    start_rollout_id=0,
    fail_train_at=None,
):
    events = []
    rollout_manager = _FakeRolloutManager(events)
    actor_model = _FakeActorModel(events, fail_train_at=fail_train_at)

    monkeypatch.setattr(train_async, "ray", SimpleNamespace(get=lambda value: value))
    monkeypatch.setattr(train_async, "configure_logger", lambda: None)
    monkeypatch.setattr(train_async, "init_tracking", lambda _args: None)
    monkeypatch.setattr(train_async, "finish_tracking", lambda _args: None)
    monkeypatch.setattr(train_async, "update_tracking_open_metrics", lambda _args, _addr: None)
    monkeypatch.setattr(train_async, "create_placement_groups", lambda _args: {"rollout": object()})
    monkeypatch.setattr(
        train_async,
        "create_rollout_manager",
        lambda _args, _pg: (rollout_manager, None),
    )
    monkeypatch.setattr(
        train_async,
        "create_training_models",
        lambda _args, _pgs, _manager: (actor_model, None),
    )

    args = SimpleNamespace(
        colocate=False,
        check_weight_update_equal=False,
        start_rollout_id=start_rollout_id,
        num_rollout=num_rollout,
        use_critic=False,
        rollout_global_dataset=True,
        save_interval=1,
        update_weights_interval=1,
        eval_interval=None,
    )
    return args, rollout_manager, actor_model, events


def test_async_dataset_state_is_saved_before_next_rollout_advances_it(monkeypatch):
    args, rollout_manager, actor_model, events = _install_fake_async_runtime(monkeypatch)

    train_async.train(args)

    assert rollout_manager.saved_states == [
        (0, "post-rollout-0"),
        (1, "post-rollout-1"),
    ]
    assert actor_model.saved_models == [0, 1]
    assert events.index("dataset-save:0:post-rollout-0") < events.index("generate:1:weight-1")
    assert events.count("dataset-save:0:post-rollout-0") == 1


def test_async_failure_may_leave_ahead_dataset_state_but_not_model_checkpoint(monkeypatch):
    args, rollout_manager, actor_model, _events = _install_fake_async_runtime(
        monkeypatch,
        num_rollout=1,
        fail_train_at=0,
    )

    with pytest.raises(RuntimeError, match="injected train failure"):
        train_async.train(args)

    # Resume chooses the state id from the latest completed model checkpoint,
    # so this pre-model state is intentionally harmless and ignored on retry.
    assert rollout_manager.saved_states == [(0, "post-rollout-0")]
    assert actor_model.saved_models == []


def test_async_empty_resume_range_does_not_launch_unconsumed_generation(monkeypatch):
    args, _rollout_manager, actor_model, events = _install_fake_async_runtime(
        monkeypatch,
        num_rollout=65,
        start_rollout_id=65,
    )

    train_async.train(args)

    assert not any(event.startswith("generate:") for event in events)
    assert actor_model.saved_models == []
    assert events[-1] == "dispose"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
