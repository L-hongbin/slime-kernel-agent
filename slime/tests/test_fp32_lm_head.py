import os
from types import SimpleNamespace

import pytest
import torch

os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/slime_flashinfer_test_cache")

from slime.backends.megatron_utils.fp32_lm_head import enable_fp32_lm_head


class _SingleRankGroup:
    def size(self) -> int:
        return 1

    def rank(self) -> int:
        return 0


class _TinyOutputLayer(torch.nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight)
        self.weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)
        self.bias = None
        self.input_size = weight.shape[1]
        self.output_size_per_partition = weight.shape[0]
        self.skip_bias_add = False
        self.allreduce_dgrad = False
        self.sequence_parallel = False
        self.explicit_expert_comm = False
        self.disable_grad_reduce = True
        self.gradient_accumulation_fusion = True
        self.gather_output = False
        self.config = SimpleNamespace(defer_embedding_wgrad_compute=False, wgrad_deferral_limit=None)
        self.tp_group = _SingleRankGroup()


def _supports_cpu_bf16_matmul() -> bool:
    try:
        torch.ones(2, 2, dtype=torch.bfloat16).matmul(torch.ones(2, 2, dtype=torch.bfloat16))
    except RuntimeError:
        return False
    return True


@pytest.mark.unit
def test_fp32_lm_head_logits_are_fp32_and_closer_to_fp32_reference():
    if not _supports_cpu_bf16_matmul():
        pytest.skip("CPU bfloat16 matmul is not available in this PyTorch build.")

    torch.manual_seed(1234)
    hidden = (torch.randn(5, 3, 7) * 3).bfloat16()
    weight = (torch.randn(11, 7) * 3).bfloat16()
    layer = _TinyOutputLayer(weight)
    enable_fp32_lm_head(SimpleNamespace(output_layer=layer))

    logits, _ = layer(hidden)

    reference = hidden.float().matmul(layer.weight.float().t())
    bf16_logits = hidden.matmul(layer.weight.detach().t()).float()

    assert logits.dtype == torch.float32
    assert torch.allclose(logits, reference, atol=0, rtol=0)
    assert (logits - reference).abs().max() < (bf16_logits - reference).abs().max()


@pytest.mark.unit
def test_fp32_lm_head_accumulates_fp32_main_grad():
    torch.manual_seed(5678)
    hidden = (torch.randn(4, 2, 6) * 2).bfloat16().requires_grad_(True)
    weight = (torch.randn(9, 6) * 2).bfloat16()
    grad_output = torch.randn(4, 2, 9)
    layer = _TinyOutputLayer(weight)
    enable_fp32_lm_head(SimpleNamespace(output_layer=layer))

    logits, _ = layer(hidden)
    loss = (logits * grad_output).sum()
    loss.backward()

    expected_main_grad = grad_output.reshape(-1, 9).t().matmul(hidden.detach().float().reshape(-1, 6))
    expected_hidden_grad = grad_output.matmul(layer.weight.detach().float()).to(torch.bfloat16)

    assert logits.dtype == torch.float32
    assert layer.weight.main_grad.dtype == torch.float32
    assert torch.allclose(layer.weight.main_grad, expected_main_grad, atol=1e-5, rtol=1e-5)
    assert layer.weight.grad is None
    assert hidden.grad is not None
    assert hidden.grad.dtype == torch.bfloat16
    assert torch.allclose(hidden.grad, expected_hidden_grad, atol=0, rtol=0)
