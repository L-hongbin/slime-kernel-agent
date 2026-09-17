import copy
import json
import logging
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.kernel_agent.kernel_filter import filter_cuda_kernel_group
from examples.kernel_agent.kernel_reward import post_process_rollout_rewards
from slime.utils.types import Sample

NUM_GPUS = 0


@pytest.mark.parametrize("method", [None, "dapo", "laser-d"])
@pytest.mark.parametrize("has_task_reward", [False, True])
def test_filter_reward_selection_is_read_only(method, has_task_reward):
    args = Namespace(
        n_samples_per_prompt=2, min_group_size=2, target_group_size=2, reward_key="score", overlong_penalty=method
    )
    samples = [
        Sample(
            index=index,
            group_index=0,
            reward={"score": float(index), "other": 9.0},
            metadata={"task_reward": 0.0} if has_task_reward else {},
        )
        for index in range(2)
    ]
    before = copy.deepcopy([sample.to_dict() for sample in samples])
    result = filter_cuda_kernel_group(args, samples)
    assert result.keep is (method == "laser-d" or not has_task_reward)
    assert [sample.to_dict() for sample in samples] == before


def test_low_variance_drop_logs_ids_and_corresponding_rewards(caplog):
    args = Namespace(
        min_group_size=2,
        n_samples_per_prompt=2,
        reward_key=None,
        reward_std_threshold=1e-3,
        target_group_size=2,
    )
    samples = [
        Sample(
            group_index=41,
            index=100,
            rollout_id=100,
            reward=0.25,
            metadata={"uuid": "prompt-uuid", "start_rollout_id": 12, "task_reward": 0.5},
        ),
        Sample(
            group_index=41,
            index=101,
            rollout_id=101,
            reward=0.125,
            metadata={"uuid": "prompt-uuid", "start_rollout_id": 12, "task_reward": 0.5},
        ),
    ]

    with caplog.at_level(logging.INFO, logger="examples.kernel_agent.kernel_filter"):
        result = filter_cuda_kernel_group(args, samples)

    assert not result.keep
    assert result.reason == "reward_std_lt_0.001"
    message = next(record.message for record in caplog.records if " audit=" in record.message)
    audit = json.loads(message.split(" audit=", 1)[1])
    assert audit == {
        "prompt_group_index": 41,
        "rollout_step": 12,
        "samples": [
            {
                "filter_reward": 0.5,
                "rollout_id": 100,
                "id": "prompt-uuid",
                "reward": 0.25,
                "sample_index": 100,
            },
            {
                "filter_reward": 0.5,
                "rollout_id": 101,
                "id": "prompt-uuid",
                "reward": 0.125,
                "sample_index": 101,
            },
        ],
    }


@pytest.mark.parametrize("gate,keep", [(None, True), ("sqrt", False), ("piecewise", True), ("piecewise-sqrt", True)])
def test_filter_respects_gate_switch_without_writing_reward(monkeypatch, gate, keep):
    from examples.kernel_agent.config import CUDA_AGENT_CONFIGS

    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "apply_failed_group_reward", False)
    # Isolate the gate from the independent correctness prerequisite for speedup rewards.
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "performance_reward_requires_correctness", False)
    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "init_performance_weight", 0.5)
    args = Namespace(
        min_group_size=2,
        n_samples_per_prompt=2,
        reward_key=None,
        reward_std_threshold=0.001,
        target_group_size=2,
        dynamic_reward_gate=gate,
    )
    samples = []
    for i, performance in enumerate([0.0, 1.0]):
        reward = performance * 0.5
        samples.append(
            Sample(
                group_index=0,
                index=i,
                reward=reward,
                metadata={
                    "task_reward": reward,
                    "kernel_score": {"correctness": 0.0, "performance": performance, "coverage": 0.0},
                    "reward_component": {
                        "correctness": 0.0,
                        "performance": reward,
                        "coverage": 0.0,
                        "failed": None,
                        "length": 0.0,
                    },
                },
            )
        )
    post_process_rollout_rewards(args, samples)
    before = copy.deepcopy([sample.to_dict() for sample in samples])
    result = filter_cuda_kernel_group(args, samples)
    assert result.keep is keep
    assert [sample.to_dict() for sample in samples] == before


@pytest.mark.parametrize("mode,keep", [("sqrt", False), ("piecewise", True), ("piecewise-sqrt", True)])
def test_filter_reads_settled_dynamic_gate_without_mutation(monkeypatch, mode, keep):
    from examples.kernel_agent.config import CUDA_AGENT_CONFIGS

    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "init_performance_weight", 0.5)
    args = Namespace(
        min_group_size=2,
        n_samples_per_prompt=2,
        target_group_size=2,
        reward_key=None,
        reward_std_threshold=0.27,
        dynamic_reward_gate=mode,
    )
    samples = [
        Sample(
            group_index=0,
            reward=0.5 + 0.5 * performance,
            metadata={
                "task_reward": 0.5 + 0.5 * performance,
                "kernel_score": {"correctness": 1.0, "performance": performance, "coverage": 0.0},
                "reward_component": {
                    "correctness": 0.5,
                    "performance": 0.5 * performance,
                    "coverage": 0.0,
                    "failed": None,
                    "length": 0.0,
                },
            },
        )
        for performance in (0.0, 1.0)
    ]
    post_process_rollout_rewards(args, samples)
    before = copy.deepcopy([sample.to_dict() for sample in samples])
    assert filter_cuda_kernel_group(args, samples).keep is keep
    assert [sample.to_dict() for sample in samples] == before
    assert all("dynamic_reward" in sample.metadata for sample in samples)


@pytest.mark.parametrize("gate", [None, "sqrt", "piecewise-sqrt"])
def test_laser_d_bonus_keeps_all_correct_groups_with_different_lengths(monkeypatch, gate):
    from examples.kernel_agent.config import CUDA_AGENT_CONFIGS
    from examples.kernel_agent.length_reward import LaserDBudgetController

    monkeypatch.setitem(CUDA_AGENT_CONFIGS["reward"], "init_performance_weight", 0.5)
    args = Namespace(
        min_group_size=2,
        n_samples_per_prompt=2,
        target_group_size=2,
        reward_key=None,
        reward_std_threshold=0.1,
        dynamic_reward_gate=gate,
        overlong_penalty="laser-d",
        rollout_max_response_len=16,
        laser_d_min_length=4,
    )
    samples = [
        Sample(
            group_index=0,
            reward=1.0,
            response_length=length,
            status=Sample.Status.COMPLETED,
            metadata={
                "task_reward": 1.0,
                "kernel_score": {"correctness": 1.0, "performance": 1.0, "coverage": 0.0},
                "reward_component": {
                    "correctness": 0.5,
                    "performance": 0.5,
                    "coverage": 0.0,
                    "failed": None,
                    "length": 0.0,
                },
            },
        )
        for length in (4, 8)
    ]
    from types import SimpleNamespace

    LaserDBudgetController(args, SimpleNamespace(metadata={}), 0).observe(samples)
    post_process_rollout_rewards(args, samples)
    assert filter_cuda_kernel_group(args, samples).keep
    assert [sample.reward - sample.metadata["task_reward"] for sample in samples] == [0.5, 0.0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
