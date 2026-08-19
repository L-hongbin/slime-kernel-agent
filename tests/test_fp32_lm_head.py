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


@pytest.mark.parametrize("flag_name", ["fp32_lm_head", "enable_fp32_lm_head"])
def test_fp32_lm_head_accepts_current_and_legacy_flag_names(flag_name):
    assert _fp32_lm_head_requested(SimpleNamespace(**{flag_name: True}))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
