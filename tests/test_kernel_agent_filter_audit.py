import json
import logging
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.kernel_agent.kernel_filter import filter_cuda_kernel_group
from slime.utils.types import Sample

NUM_GPUS = 0


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
