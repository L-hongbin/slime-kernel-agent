"""Single-GPU FP32 equivalence and BF16 numerical audit across GDN chunk boundaries.

Uses random small Qwen3.5 GDN weights, not a deployed 27B checkpoint. The CPU
suite separately exercises CP loss layouts; this test runs the real CUDA
recurrence, fused vocabulary loss and backward without starting rollout.
"""

import copy
import json
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from test_trloo_trajectory_packing import _args, _loss_and_grad, _manager, _samples

NUM_GPUS = 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GDN kernel required")
def test_gdn_chunked_three_turn_backward(tmp_path, monkeypatch):
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    from megatron.core import parallel_state
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    class GDNPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            config = Qwen3_5TextConfig(
                hidden_size=128,
                linear_key_head_dim=128,
                linear_value_head_dim=128,
                linear_num_key_heads=1,
                linear_num_value_heads=2,
            )
            self.use_bf16 = False
            self.embedding = torch.nn.Embedding(32, 128)
            self.gdn = Qwen3_5GatedDeltaNet(config, layer_idx=0)
            # HF disables both fast kernels when causal-conv1d is absent.
            # Select FLA explicitly; PyTorch's causal convolution stays valid.
            self.gdn.chunk_gated_delta_rule = chunk_gated_delta_rule
            self.head = torch.nn.Linear(128, 32)
            with torch.no_grad():
                self.gdn.A_log.uniform_(0, 1)
                self.gdn.dt_bias.fill_(1)

        def forward(self, tokens):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.use_bf16):
                x = self.embedding(tokens)
                if self.use_bf16:
                    x = x.to(torch.bfloat16)
                x = x.unsqueeze(0)
                return self.head(x + self.gdn(x)).squeeze(0).float()

    torch.cuda.set_device(0)
    dist.init_process_group("nccl", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    parallel_state.initialize_model_parallel()
    try:
        torch.manual_seed(421)
        model = GDNPolicy().cuda().to(torch.bfloat16).float()
        assert model.gdn.chunk_gated_delta_rule.__module__.startswith("fla."), "Must exercise the fused GDN kernel"
        samples = _samples(model, repair=True, filtered=True, response_lengths=(127, 157, 193))
        args = _args(eps_clip=0.15, eps_clip_high=0.15)
        split = _manager(args)._convert_samples_to_train_data(copy.deepcopy(samples))
        args.pack_multi_turn_trajectories = True
        packed = _manager(args)._convert_samples_to_train_data(copy.deepcopy(samples))
        with torch.no_grad():
            delta = torch.randn_like(model.head.weight, dtype=torch.bfloat16) * 0.01
            model.head.weight.copy_((model.head.weight.to(torch.bfloat16) + delta).float())
        old = _loss_and_grad(model, split, args, monkeypatch, 1)
        new = _loss_and_grad(model, packed, args, monkeypatch, 1)
        relative_grad_error = float((old[1] - new[1]).float().norm() / old[1].float().norm())
        max_logprob_error = max(abs(a - b) for a, b in zip(old[3], new[3], strict=True))
        print(
            json.dumps(
                {
                    "kernel": model.gdn.chunk_gated_delta_rule.__module__,
                    "split_tokens": sum(map(len, split["tokens"])),
                    "packed_tokens": sum(map(len, packed["tokens"])),
                    "split_loss": old[0],
                    "packed_loss": new[0],
                    "relative_grad_error": relative_grad_error,
                    "max_logprob_error": max_logprob_error,
                }
            )
        )
        assert old[0] == pytest.approx(new[0], abs=2e-6, rel=1e-4)
        assert torch.isfinite(new[1]).all()
        assert relative_grad_error < 2e-4
        assert max_logprob_error < 1e-5

        # Hold weights, tokens, masks and behavior distributions fixed. Only
        # forward/backward activation precision changes. This is a numerical
        # audit: shared BF16 adjoints need not round like separate backwards.
        model.use_bf16 = True
        old_bf16 = _loss_and_grad(model, split, args, monkeypatch, 1)
        new_bf16 = _loss_and_grad(model, packed, args, monkeypatch, 1)
        bf16_error = float((old_bf16[1] - new_bf16[1]).norm() / old_bf16[1].norm())
        split_vs_fp32 = float((old_bf16[1] - old[1]).norm() / old[1].norm())
        packed_vs_fp32 = float((new_bf16[1] - new[1]).norm() / new[1].norm())
        print(
            json.dumps(
                {
                    "precision_audit": "same weights and behavior data; BF16 activations / FP32 grad buffers",
                    "split_loss": old_bf16[0],
                    "packed_loss": new_bf16[0],
                    "relative_grad_error": bf16_error,
                    "split_grad_error_vs_fp32": split_vs_fp32,
                    "packed_grad_error_vs_fp32": packed_vs_fp32,
                    "grad_cosine": float(torch.nn.functional.cosine_similarity(old_bf16[1], new_bf16[1], dim=0)),
                }
            )
        )
        assert torch.isfinite(old_bf16[1]).all() and torch.isfinite(new_bf16[1]).all()
    finally:
        parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-s"]))
