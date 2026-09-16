from types import SimpleNamespace

import pytest
import torch

from slime.ray.rollout import RolloutManager, _compute_turn_generation_metrics
from slime.utils.types import Sample

NUM_GPUS = 0


def manager(tmp_path):
    cls = RolloutManager.__ray_metadata__.modified_class
    instance = cls.__new__(cls)
    instance.args = SimpleNamespace(save_debug_rollout_data=str(tmp_path / "rollout_{rollout_id}.pt"))
    return instance


def test_capture_first_and_latest_atomically(tmp_path, monkeypatch):
    monkeypatch.setenv("SLIME_SAVE_DEBUG_ROLLOUT_MAX_ID", "0")
    monkeypatch.setenv("SLIME_SAVE_DEBUG_ROLLOUT_KEEP_LATEST", "1")
    instance = manager(tmp_path)
    for step in (0, 1, 2):
        instance._save_debug_rollout_data([Sample(response=str(step))], step, evaluation=False)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["rollout_0.pt", "rollout_latest.pt"]
    assert torch.load(tmp_path / "rollout_0.pt", weights_only=False)["rollout_id"] == 0
    latest = torch.load(tmp_path / "rollout_latest.pt", weights_only=False)
    assert latest["rollout_id"] == 2
    assert latest["samples"][0]["response"] == "2"


def test_capture_limit_without_latest_preserves_only_first(tmp_path, monkeypatch):
    monkeypatch.setenv("SLIME_SAVE_DEBUG_ROLLOUT_MAX_ID", "0")
    monkeypatch.setenv("SLIME_SAVE_DEBUG_ROLLOUT_KEEP_LATEST", "0")
    instance = manager(tmp_path)
    instance._save_debug_rollout_data([], 0, evaluation=False)
    instance._save_debug_rollout_data([], 1, evaluation=False)
    assert [path.name for path in tmp_path.iterdir()] == ["rollout_0.pt"]


def test_eval_capture_does_not_use_train_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("SLIME_SAVE_DEBUG_ROLLOUT_MAX_ID", "0")
    instance = manager(tmp_path)
    instance._save_debug_rollout_data({"validation": {"samples": [Sample()]}}, 9, evaluation=True)
    assert (tmp_path / "rollout_eval_9.pt").is_file()


@pytest.mark.parametrize("maximum", ["-1", "nan", "1.2"])
def test_invalid_capture_limit_fails_before_writing(tmp_path, monkeypatch, maximum):
    monkeypatch.setenv("SLIME_SAVE_DEBUG_ROLLOUT_MAX_ID", maximum)
    with pytest.raises(ValueError, match="non-negative integer"):
        manager(tmp_path)._save_debug_rollout_data([], 0, evaluation=False)
    assert not list(tmp_path.iterdir())


def test_turn_generation_counts_precheck_removed_and_padding():
    samples = [
        Sample(status=Sample.Status.TRUNCATED, response_length=9, metadata={"turn_idx": 0}),
        Sample(
            status=Sample.Status.TRUNCATED,
            response_length=7,
            remove_sample=True,
            loss_mask=[0],
            metadata={"turn_idx": 0, "env_extra_info": {"speedup": None}},
        ),
        Sample(status=Sample.Status.COMPLETED, response_length=0, metadata={"turn_idx": 0, "is_pad_turn": True}),
        Sample(status=Sample.Status.COMPLETED, response_length=2, metadata={"turn_idx": 1}),
        Sample(status=Sample.Status.TRUNCATED, metadata={}),
    ]
    metrics = _compute_turn_generation_metrics(samples)
    assert metrics["generation/turn0/samples"] == 3
    assert metrics["generation/turn0/truncated_count"] == 2
    assert metrics["generation/turn0/truncated_ratio"] == pytest.approx(2 / 3)
    assert metrics["generation/turn0/pad_samples"] == 1
    assert metrics["generation/turn1/truncated_ratio"] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
