import inspect
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

NUM_GPUS = 0

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from slime.backends.megatron_utils.actor import (
    MegatronTrainRayActor,
    _apply_train_env_vars,
    _cold_base_load_for_adapter_resume,
    _snapshot_entropy_common_probe_masks,
)
from slime.utils.train_metric_utils import ENTROPY_COMMON_PROBE_MASK_KEY


def _probe_rollout_data():
    return {
        "tokens": [torch.arange(5), torch.arange(7)],
        "loss_masks": [torch.tensor([1, 0, 1]), torch.tensor([1, 1, 0, 1])],
        "response_lengths": [3, 4],
        "total_lengths": [5, 7],
    }


def test_apply_train_env_vars_sets_runtime_env_without_overwriting_rank(monkeypatch):
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.delenv("DSV4_TEST_RUNTIME_SWITCH", raising=False)

    args = SimpleNamespace(
        train_env_vars={
            "RANK": "0",
            "LOCAL_RANK": "0",
            "DSV4_TEST_RUNTIME_SWITCH": "enabled",
        }
    )

    _apply_train_env_vars(args)

    assert os.environ["DSV4_TEST_RUNTIME_SWITCH"] == "enabled"
    assert os.environ["RANK"] == "7"
    assert os.environ["LOCAL_RANK"] == "3"


def test_adapter_resume_load_flags_apply_only_to_overlay():
    args = SimpleNamespace(no_load_optim=False, no_load_rng=False)

    with _cold_base_load_for_adapter_resume(args, enabled=True):
        assert args.no_load_optim is True
        assert args.no_load_rng is True

    assert args.no_load_optim is False
    assert args.no_load_rng is False


def test_cold_base_load_restores_flags_after_failure():
    args = SimpleNamespace(no_load_optim=False, no_load_rng=True)

    try:
        with _cold_base_load_for_adapter_resume(args, enabled=True):
            raise RuntimeError("synthetic load failure")
    except RuntimeError:
        pass

    assert args.no_load_optim is False
    assert args.no_load_rng is True


def test_entropy_common_probe_snapshot_is_gated_cloned_and_non_aliasing():
    rollout_data = _probe_rollout_data()

    assert _snapshot_entropy_common_probe_masks(rollout_data, enabled=False) is False
    assert ENTROPY_COMMON_PROBE_MASK_KEY not in rollout_data

    originals = [mask.clone() for mask in rollout_data["loss_masks"]]
    assert _snapshot_entropy_common_probe_masks(rollout_data, enabled=True) is True
    frozen = rollout_data[ENTROPY_COMMON_PROBE_MASK_KEY]
    for original_mask, live_mask, frozen_mask in zip(originals, rollout_data["loss_masks"], frozen, strict=True):
        torch.testing.assert_close(frozen_mask, original_mask)
        assert frozen_mask.data_ptr() != live_mask.data_ptr()

    # Model the postprocess changing the live population: the probe remains the
    # exact pre-MIS masks because it owns independent storage.
    rollout_data["loss_masks"][0].zero_()
    torch.testing.assert_close(frozen[0], originals[0])


def test_entropy_common_probe_snapshot_rejects_duplicate_missing_and_misaligned_data():
    missing = _probe_rollout_data()
    del missing["loss_masks"]
    with pytest.raises(RuntimeError, match="missing rollout fields"):
        _snapshot_entropy_common_probe_masks(missing, enabled=True)

    misaligned = _probe_rollout_data()
    misaligned["response_lengths"][0] = 2
    with pytest.raises(RuntimeError, match="mask/response length mismatch"):
        _snapshot_entropy_common_probe_masks(misaligned, enabled=True)

    duplicate = _probe_rollout_data()
    _snapshot_entropy_common_probe_masks(duplicate, enabled=True)
    with pytest.raises(RuntimeError, match="already exists"):
        _snapshot_entropy_common_probe_masks(duplicate, enabled=True)


def test_entropy_common_probe_snapshot_precedes_iterator_and_postprocess():
    source = inspect.getsource(MegatronTrainRayActor.train_actor)
    snapshot_at = source.index("_snapshot_entropy_common_probe_masks(")
    iterator_at = source.index("get_data_iterator(rollout_data)")
    postprocess_at = source.index("self.rollout_data_postprocess(self.args, rollout_id, rollout_data)")
    assert snapshot_at < iterator_at < postprocess_at


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
