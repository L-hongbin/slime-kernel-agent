"""CPU contracts for fixed-batch KernelAgent diagnostic reducers."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.kernel_agent.diagnostic_reducers import get_completion_mean_pg_loss_reducer

NUM_GPUS = 0


def test_completion_mean_gives_short_and_long_completions_equal_weight():
    response_lengths = [2, 4]
    masks = [torch.ones(2), torch.ones(4)]
    token_losses = torch.tensor([1.0, 3.0, 2.0, 2.0, 2.0, 2.0])

    with patch(
        "slime.backends.megatron_utils.cp_utils.mpu.get_context_parallel_world_size",
        return_value=1,
    ):
        reducer = get_completion_mean_pg_loss_reducer(
            total_lengths=response_lengths,
            response_lengths=response_lengths,
            loss_masks=masks,
            calculate_per_token_loss=False,
        )

    # mean([1,3]) + mean([2,2,2,2]); the outer sample normalizer divides by 2.
    assert reducer(token_losses).item() == pytest.approx(4.0)


def test_completion_mean_rejects_token_normalizer_contract():
    with pytest.raises(ValueError, match="normalizes by samples rather than tokens"):
        get_completion_mean_pg_loss_reducer(
            total_lengths=[2],
            response_lengths=[2],
            loss_masks=[torch.ones(2)],
            calculate_per_token_loss=True,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
