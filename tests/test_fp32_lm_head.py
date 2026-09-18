import os
from types import SimpleNamespace

import pytest
import torch

NUM_GPUS = 0

os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/slime_flashinfer_test_cache")

from slime.backends.megatron_utils.model_provider import (  # noqa: E402
    _enable_actor_fp32_lm_head,
    _fp32_lm_head_requested,
)


class _SingleRankGroup:
    def size(self) -> int:
        return 1


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


def test_legacy_enable_actor_fp32_lm_head_uses_safe_projection_patch():
    torch.manual_seed(1234)
    hidden = torch.randn(4, 3, dtype=torch.bfloat16)
    weight = torch.randn(2, 3, dtype=torch.bfloat16)
    model = SimpleNamespace(output_layer=_TinyOutputLayer(weight))

    returned_model = _enable_actor_fp32_lm_head(model)
    logits, bias = model.output_layer(hidden)

    assert returned_model is model
    assert bias is None
    assert model.output_layer.weight.dtype == torch.bfloat16
    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits, hidden.float().matmul(weight.float().t()))


def test_legacy_enable_actor_fp32_lm_head_is_idempotent():
    model = SimpleNamespace(output_layer=_TinyOutputLayer(torch.randn(2, 3, dtype=torch.bfloat16)))

    _enable_actor_fp32_lm_head(model)
    first_forward = model.output_layer.forward
    _enable_actor_fp32_lm_head(model)

    assert model.output_layer.forward == first_forward


def test_fp32_lm_head_uses_enable_flag():
    assert _fp32_lm_head_requested(SimpleNamespace(enable_fp32_lm_head=True))
    assert not _fp32_lm_head_requested(SimpleNamespace(fp32_lm_head=True))


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("frozen_weight", [False, True])
@pytest.mark.parametrize("sequence_length", [0, 5])
def test_fp32_lm_head_strided_sequence_gradients(batch_size, frozen_weight, sequence_length):
    torch.manual_seed(1234)
    hidden = torch.randn(3, batch_size, sequence_length, dtype=torch.bfloat16).permute(2, 1, 0).requires_grad_()
    weight = torch.randn(7, 3, dtype=torch.bfloat16)
    model = SimpleNamespace(output_layer=_TinyOutputLayer(weight))
    _enable_actor_fp32_lm_head(model)
    output_weight = model.output_layer.weight.detach() if frozen_weight else model.output_layer.weight
    logits, _ = model.output_layer(hidden, weight=output_weight)
    grad_output = torch.randn(7, batch_size, sequence_length).permute(2, 1, 0)
    if sequence_length:
        assert not hidden.is_contiguous() and not grad_output.is_contiguous()

    reference_hidden = hidden.detach().float().requires_grad_()
    reference_weight = weight.float().requires_grad_(not frozen_weight)
    reference = torch.nn.functional.linear(reference_hidden, reference_weight)
    logits.backward(grad_output)
    reference.backward(grad_output)

    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits, reference)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad.to(hidden.dtype))
    if frozen_weight:
        assert model.output_layer.weight.grad is None
        assert torch.count_nonzero(model.output_layer.weight.main_grad) == 0
    else:
        torch.testing.assert_close(model.output_layer.weight.main_grad, reference_weight.grad)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
