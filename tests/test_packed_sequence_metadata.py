import pytest
import torch

from slime.backends.megatron_utils import data as data_module
from slime.utils.sequence_metadata import get_cpu_sequence_boundaries

pytestmark = pytest.mark.unit
NUM_GPUS = 0


@pytest.mark.parametrize("cp_size,cp_rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("allgather_cp", [False, True])
def test_packed_microbatch_builds_metadata_without_scalar_readback(monkeypatch, cp_size, cp_rank, allgather_cp):
    monkeypatch.setattr(data_module.mpu, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_world_size", lambda: cp_size)
    monkeypatch.setattr(data_module.mpu, "get_context_parallel_rank", lambda: cp_rank)
    monkeypatch.setattr(data_module.accelerator, "current_device", lambda: torch.device("cpu"))
    rollout_data = {
        "tokens": [torch.arange(5), torch.arange(7)],
        "loss_masks": [torch.ones(2), torch.ones(3)],
        "total_lengths": [5, 7],
        "response_lengths": [2, 3],
    }
    monkeypatch.setattr(torch.Tensor, "item", lambda self: pytest.fail("unexpected scalar readback"))
    batch = data_module.get_batch(
        data_module.DataIterator(rollout_data, [[0, 1]]),
        list(rollout_data),
        pad_multiplier=2,
        allgather_cp=allgather_cp,
    )
    packed = batch["packed_seq_params"]
    expected = (0, 8, 16) if cp_size == 2 and not allgather_cp else (0, 5, 12, 16)
    assert get_cpu_sequence_boundaries(packed.cu_seqlens_q) == expected
    assert packed.cu_seqlens_q is packed.cu_seqlens_kv
    assert packed.max_seqlen_q == (8 if cp_size == 2 and not allgather_cp else 7)
    assert batch["tokens"].shape == batch["full_loss_masks"].shape == (1, 16 // cp_size)
