from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slime.ray.rollout import RolloutManager
from slime.utils.types import Sample

NUM_GPUS = 0


def _make_manager(*, use_multi_turn: bool = False, enable_turns_dp_partitions: bool = False):
    manager_cls = RolloutManager.__ray_metadata__.modified_class
    manager = manager_cls.__new__(manager_cls)
    manager.custom_convert_samples_to_train_data_func = None
    manager.custom_reward_post_process_func = lambda _args, samples: (
        [sample.reward for sample in samples],
        [sample.reward for sample in samples],
    )
    manager.args = SimpleNamespace(
        enable_turns_dp_partitions=enable_turns_dp_partitions,
        reward_key=None,
        use_multi_turn=use_multi_turn,
        use_rollout_routing_replay=True,
        num_layers=1,
        moe_router_topk=1,
    )
    return manager


def _make_sample(index: int, reward: float, turn_idx: int | None = None) -> Sample:
    metadata = {}
    if turn_idx is not None:
        metadata["turn_idx"] = turn_idx
    return Sample(
        index=index,
        group_index=0,
        reward=reward,
        metadata=metadata,
        tokens=[1, 2],
        response_length=1,
    )


def _routed() -> np.ndarray:
    return np.zeros((1, 1, 1), dtype=np.int32)


def test_convert_filters_missing_routing_replay_samples_before_rewards():
    manager = _make_manager()
    samples = [
        _make_sample(0, 1.0),
        _make_sample(1, 2.0),
        _make_sample(2, 4.0),
    ]
    samples[0].rollout_routed_experts = _routed()
    samples[2].rollout_routed_experts = _routed()

    train_data = manager._convert_samples_to_train_data(samples)

    assert train_data["sample_indices"] == [0, 2]
    assert train_data["raw_reward"] == [1.0, 4.0]
    assert train_data["rewards"] == [1.0, 4.0]
    assert len(train_data["rollout_routed_experts"]) == 2


def test_convert_filters_whole_trajectory_when_turn_partition_routing_is_missing():
    manager = _make_manager(use_multi_turn=True, enable_turns_dp_partitions=True)
    samples = [
        _make_sample(0, 1.0, turn_idx=0),
        _make_sample(0, 2.0, turn_idx=1),
        _make_sample(1, 3.0, turn_idx=0),
        _make_sample(1, 4.0, turn_idx=1),
    ]
    samples[0].rollout_routed_experts = _routed()
    samples[2].rollout_routed_experts = _routed()
    samples[3].rollout_routed_experts = _routed()

    train_data = manager._convert_samples_to_train_data(samples)

    assert train_data["sample_indices"] == [1, 1]
    assert train_data["turn_indices"] == [0, 1]
    assert train_data["raw_reward"] == [3.0, 4.0]
    assert len(train_data["rollout_routed_experts"]) == 2


def test_convert_raises_when_all_routing_replay_samples_are_missing():
    manager = _make_manager()

    with pytest.raises(ValueError, match="All rollout samples were missing rollout_routed_experts"):
        manager._convert_samples_to_train_data([_make_sample(0, 1.0)])


def test_generate_rollout_rejects_groups_missing_routing_for_refill():
    # The acceptance-loop check (reject-and-refill) that keeps the train batch exact:
    # a group with ANY None routed record is rejected; empty (pad-turn) arrays pass.
    from slime.rollout.sglang_rollout import _groups_missing_routing_replay

    args_on = SimpleNamespace(use_rollout_routing_replay=True)
    args_off = SimpleNamespace(use_rollout_routing_replay=False)

    healthy = [_make_sample(0, 1.0), _make_sample(0, 2.0)]
    for s in healthy:
        s.rollout_routed_experts = _routed()
    pad_like = _make_sample(0, 0.0)
    pad_like.rollout_routed_experts = _routed()[:0]  # empty, not None (pad turn)
    broken = [_make_sample(1, 1.0), _make_sample(1, 2.0)]
    broken[0].rollout_routed_experts = _routed()  # second sample missing

    assert _groups_missing_routing_replay(args_on, [healthy]) is False
    assert _groups_missing_routing_replay(args_on, [healthy + [pad_like]]) is False
    assert _groups_missing_routing_replay(args_on, [broken]) is True
    # multi-turn trajectory: list of turn-groups, missing record in any turn rejects
    assert _groups_missing_routing_replay(args_on, [healthy, broken]) is True
    # replay off: never rejects
    assert _groups_missing_routing_replay(args_off, [broken]) is False
    # flat list[Sample] groups (single-turn path) also handled
    assert _groups_missing_routing_replay(args_on, [[broken[1]]]) is True


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("turn_partitions", [False, True])
def test_prompt_metadata_conversion_and_dp_transport(monkeypatch, enabled, turn_partitions):
    import slime.ray.rollout as rollout_module

    manager = _make_manager(use_multi_turn=True, enable_turns_dp_partitions=turn_partitions)
    manager.args.calculate_per_prompt_loss = enabled
    manager.args.use_rollout_routing_replay = False
    manager.args.global_batch_size = 4
    manager.args.micro_batch_size = 1
    manager.args.use_dynamic_batch_size = False
    manager.args.balance_data = False
    manager.args.balance_by_flops = False
    manager.train_parallel_config = {
        "dp_size": 2,
        "cp_size": 1,
        "vpp_size": 1,
        "microbatch_group_size_per_vp_stage": 1,
    }
    samples = [_make_sample(i, float(i), turn) for i in range(4) for turn in range(2)]
    for sample in samples:
        sample.rollout_id = sample.index
    for sample in samples[-2:]:
        sample.group_index = 1
    data = manager._convert_samples_to_train_data(samples)
    assert ("group_indices" in data) is enabled
    if enabled:
        assert data["group_indices"] == [0] * 6 + [1] * 2
    monkeypatch.setattr(rollout_module.ray, "put", lambda data: data)
    shards = [box.inner for box in manager._split_train_data_by_dp(data)]
    for shard in shards:
        assert ("prompt_mask_sums" in shard) is enabled
        if enabled:
            assert shard["prompt_mask_sums"].tolist() == [6.0 if i < 6 else 2.0 for i in shard["partition"]]
            assert shard["prompt_loss_scales"].tolist() == [2.0] * len(shard["partition"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
