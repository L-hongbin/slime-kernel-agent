"""Fixed-replay capture ownership and exception recovery."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from slime.backends.megatron_utils import loss as loss_module

NUM_GPUS = 0


@pytest.mark.parametrize("write_fails", [False, True])
def test_replay_capture_keeps_gradient_values_and_restores_hooks(tmp_path, monkeypatch, write_fails):
    from slime.utils import train_dump_utils

    gradient = torch.tensor([2.0, 3.0])
    logprob = torch.tensor([-1.0, -2.0], requires_grad=True)

    def original_step():
        return True, 1.0, 0

    def original_logprobs(*a, **kw):
        return None, {"log_probs": [logprob]}

    optimizer = SimpleNamespace(step=original_step, get_main_grads_for_grad_norm=lambda: [gradient])
    monkeypatch.setattr(loss_module, "get_log_probs_and_entropy", original_logprobs)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 100)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 200)
    args = SimpleNamespace(debug_train_only=True, save_debug_train_data=str(tmp_path / "train_{rank}.pt"))
    train_dump_utils.capture_replay_step(args, 0, 0, None, optimizer, None)
    loss_module.get_log_probs_and_entropy(None, total_lengths=[5], response_lengths=[2])
    if write_fails:

        def fail_save(*a, **kw):
            raise OSError("disk full")

        monkeypatch.setattr(torch, "save", fail_save)
        with pytest.raises(OSError, match="disk full"):
            optimizer.step()
    else:
        assert optimizer.step() == (True, 1.0, 0)
        gradient.zero_()
        with torch.no_grad():
            logprob.zero_()
        capture = torch.load(tmp_path / "train_0_step0_capture.pt", weights_only=True)
        assert capture["gradients"][0].tolist() == [2.0, 3.0]
        assert capture["logprob_microbatches"][0]["log_probs"][0].tolist() == [-1.0, -2.0]
        assert capture["gradient_stage"] == "post_optimizer_step_before_buffer_clear"
    assert optimizer.step is original_step
    assert loss_module.get_log_probs_and_entropy is original_logprobs


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
