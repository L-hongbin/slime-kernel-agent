import os
from types import SimpleNamespace

import pytest
import torch

NUM_GPUS = 0

os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/slime_flashinfer_test_cache")

from slime.backends.megatron_utils.fp32_lm_head import enable_flatten_lm_head, enable_fp32_lm_head  # noqa: E402
from slime.backends.megatron_utils.model_provider import (  # noqa: E402
    _configure_actor_lm_head,
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


@pytest.mark.parametrize("impl", ["normal", "flatten"])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("frozen_weight", [False, True])
@pytest.mark.parametrize("sequence_length", [0, 5])
def test_fp32_lm_head_strided_sequence_gradients(batch_size, frozen_weight, sequence_length, impl):
    torch.manual_seed(1234)
    hidden = torch.randn(3, batch_size, sequence_length, dtype=torch.bfloat16).permute(2, 1, 0).requires_grad_()
    weight = torch.randn(7, 3, dtype=torch.bfloat16)
    model = SimpleNamespace(output_layer=_TinyOutputLayer(weight))
    enable_fp32_lm_head(model, impl=impl)
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


@pytest.mark.parametrize("impl", ["normal", "flatten"])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("frozen_weight", [False, True])
def test_fp32_lm_head_megatron_cross_entropy_gradients(monkeypatch, batch_size, frozen_weight, impl):
    from megatron.core.tensor_parallel import cross_entropy

    monkeypatch.setattr(cross_entropy, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(cross_entropy, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(cross_entropy, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, **kwargs: None)

    torch.manual_seed(1234)
    hidden = torch.randn(3, batch_size, 5, dtype=torch.bfloat16).permute(2, 1, 0).requires_grad_()
    weight = torch.randn(7, 3, dtype=torch.bfloat16)
    target = torch.randint(7, (5, batch_size))
    model = SimpleNamespace(output_layer=_TinyOutputLayer(weight))
    enable_fp32_lm_head(model, impl=impl)
    output_weight = model.output_layer.weight.detach() if frozen_weight else model.output_layer.weight
    logits, _ = model.output_layer(hidden, weight=output_weight)
    # MTP uses Megatron's CE, which mutates FP32 logits in place and saves them for backward.
    loss = cross_entropy.vocab_parallel_cross_entropy(logits, target).mean()
    loss.backward()

    reference_hidden = hidden.detach().float().requires_grad_()
    reference_weight = weight.float().requires_grad_(not frozen_weight)
    reference_logits = torch.nn.functional.linear(reference_hidden, reference_weight)
    reference_loss = torch.nn.functional.cross_entropy(reference_logits.reshape(-1, 7), target.reshape(-1))
    reference_loss.backward()

    torch.testing.assert_close(loss, reference_loss)
    torch.testing.assert_close(hidden.grad, reference_hidden.grad.to(hidden.dtype))
    if frozen_weight:
        assert model.output_layer.weight.grad is None
        assert torch.count_nonzero(model.output_layer.weight.main_grad) == 0
    else:
        torch.testing.assert_close(model.output_layer.weight.main_grad, reference_weight.grad)


def _native_output_layer(weight):
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear

    # Exercise the real native forward and _forward_impl without initializing
    # distributed parameter allocation; these tests use a single local shard.
    layer = ColumnParallelLinear.__new__(ColumnParallelLinear)
    torch.nn.Module.__init__(layer)
    layer.weight = torch.nn.Parameter(weight)
    layer.bias = None
    layer.input_size = weight.shape[1]
    layer.output_size_per_partition = weight.shape[0]
    layer.skip_bias_add = False
    layer.allreduce_dgrad = False
    layer.sequence_parallel = False
    layer.explicit_expert_comm = False
    layer.disable_grad_reduce = True
    layer.gradient_accumulation_fusion = False
    layer.gather_output = False
    layer.tp_group = _SingleRankGroup()
    layer.config = SimpleNamespace(
        defer_embedding_wgrad_compute=False, wgrad_deferral_limit=None, _cpu_offloading_context=None
    )
    return layer


@pytest.mark.parametrize("impl", ["normal", "flatten"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("frozen", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "strided"])
def test_native_head_layout_dtype_and_gradients(impl, dtype, frozen, layout):
    torch.manual_seed(1234)
    hidden = torch.randn(3, 2, 5, dtype=dtype).permute(2, 1, 0)
    grad = torch.randn(7, 2, 5, dtype=dtype).permute(2, 1, 0)
    if layout == "contiguous":
        hidden, grad = hidden.contiguous(), grad.contiguous()
    hidden.requires_grad_()
    layer = _native_output_layer(torch.randn(7, 3, dtype=dtype))
    layer.weight.requires_grad_(not frozen)
    original_forward = layer.forward
    original_impl = layer._forward_impl
    _configure_actor_lm_head(SimpleNamespace(output_layer=layer), SimpleNamespace(lm_head_impl=impl), "actor", True)
    assert layer.forward == original_forward
    if impl == "normal":
        assert layer._forward_impl == original_impl
    result, _ = layer(hidden)
    assert result.dtype == dtype
    result.backward(grad)

    ref_hidden = hidden.detach().clone().requires_grad_()
    ref_weight = layer.weight.detach().clone().requires_grad_(not frozen)
    reference = torch.nn.functional.linear(ref_hidden, ref_weight)
    reference.backward(grad)
    torch.testing.assert_close(result, reference)
    torch.testing.assert_close(hidden.grad, ref_hidden.grad)
    if frozen:
        assert layer.weight.grad is None
    else:
        torch.testing.assert_close(layer.weight.grad, ref_weight.grad)


@pytest.mark.parametrize("fp32", [False, True])
@pytest.mark.parametrize("frozen", [False, True])
def test_flatten_uses_2d_mm_in_forward_and_dgrad(monkeypatch, fp32, frozen):
    calls = []
    original_mm = torch.mm

    def tracked_mm(left, right, **kwargs):
        calls.append((tuple(left.shape), tuple(right.shape), left.dtype))
        return original_mm(left, right, **kwargs)

    monkeypatch.setattr(torch, "mm", tracked_mm)
    layer = _native_output_layer(torch.randn(7, 3, dtype=torch.bfloat16))
    layer.weight.requires_grad_(not frozen)
    model = SimpleNamespace(output_layer=layer)
    args = SimpleNamespace(lm_head_impl="flatten", enable_fp32_lm_head=fp32)
    _configure_actor_lm_head(model, args, "actor", True)
    hidden = torch.randn(3, 2, 5, dtype=torch.bfloat16).permute(2, 1, 0).requires_grad_()
    output, _ = layer(hidden)
    output.backward(torch.randn_like(output))
    dtype = torch.float32 if fp32 else torch.bfloat16
    assert calls[:2] == [((10, 3), (3, 7), dtype), ((10, 7), (7, 3), dtype)]


@pytest.mark.parametrize("impl", ["normal", "flatten"])
def test_lm_head_argument_choices_and_default(impl):
    import argparse

    from slime.utils.arguments import get_slime_extra_args_provider

    parser = get_slime_extra_args_provider()(argparse.ArgumentParser())
    assert parser.parse_args(["--rollout-batch-size", "1"]).lm_head_impl == "normal"
    assert parser.parse_args(["--rollout-batch-size", "1", "--lm-head-impl", impl]).lm_head_impl == impl
    with pytest.raises(SystemExit):
        parser.parse_args(["--rollout-batch-size", "1", "--lm-head-impl", "auto"])


@pytest.mark.parametrize("role,post_process", [("critic", True), ("actor", False)])
def test_lm_head_modes_skip_critic_and_non_output_stages(role, post_process):
    model = SimpleNamespace()
    _configure_actor_lm_head(
        model, SimpleNamespace(lm_head_impl="flatten", enable_fp32_lm_head=True), role, post_process
    )
    assert not hasattr(model, "output_layer")


def test_flatten_native_head_rejects_unsupported_layer():
    with pytest.raises(TypeError, match="ColumnParallelLinear"):
        enable_flatten_lm_head(SimpleNamespace(output_layer=torch.nn.Linear(3, 7)))


@pytest.mark.parametrize("fp32", [False, True])
@pytest.mark.parametrize("defer_limit", [0, 1])
def test_flatten_preserves_deferred_weight_gradient_buffers(fp32, defer_limit):
    layer = _native_output_layer(torch.randn(7, 3))
    layer.config.defer_embedding_wgrad_compute = True
    layer.config.wgrad_deferral_limit = defer_limit
    layer.grad_output_buffer = []
    layer.embedding_activation_buffer = []
    args = SimpleNamespace(lm_head_impl="flatten", enable_fp32_lm_head=fp32)
    _configure_actor_lm_head(SimpleNamespace(output_layer=layer), args, "actor", True)
    hidden = torch.randn(5, 2, 3, requires_grad=True)
    for i in range(2):
        layer.weight.grad = None
        output, _ = layer(hidden)
        output.sum().backward()
        deferred = defer_limit == 0 or i < defer_limit
        assert (layer.weight.grad is None) == deferred
    expected_count = 2 if defer_limit == 0 else 1
    assert len(layer.grad_output_buffer) == expected_count
    assert len(layer.embedding_activation_buffer) == expected_count
    assert layer.embedding_activation_buffer[0].shape == (5, 2, 3)
    assert layer.grad_output_buffer[0].shape == (5, 2, 7)


@pytest.mark.parametrize("fp32,impl", [(False, "flatten"), (True, "normal"), (True, "flatten")])
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("frozen", [False, True])
def test_head_collective_gradients(monkeypatch, fp32, impl, sequence_parallel, frozen):
    from slime.backends.megatron_utils import fp32_lm_head as head

    calls = []

    class Work:
        def wait(self):
            calls.append("wait")

    class Buffer:
        def get_tensor(self, shape, dtype, name):
            return torch.empty(shape, dtype=dtype)

    def gather(output, tensor, group, async_op=False):
        calls.append("gather_async" if async_op else "gather")
        output.copy_(torch.cat([tensor, tensor], dim=0))
        return Work()

    def scatter(output, tensor, group, async_op=False):
        calls.append("scatter")
        first, second = tensor.chunk(2, dim=0)
        output.copy_(first + second)
        return Work()

    def all_reduce(tensor, group, async_op=False):
        calls.append("all_reduce")
        tensor.mul_(2)
        return Work()

    monkeypatch.setattr(head, "get_global_memory_buffer", lambda: Buffer())
    monkeypatch.setattr(head, "dist_all_gather_func", gather)
    monkeypatch.setattr(head, "dist_reduce_scatter_func", scatter)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    layer = _native_output_layer(torch.randn(7, 3))
    layer.weight.requires_grad_(not frozen)
    layer.sequence_parallel = sequence_parallel
    layer.allreduce_dgrad = not sequence_parallel
    layer.tp_group = SimpleNamespace(size=lambda: 2)
    _configure_actor_lm_head(
        SimpleNamespace(output_layer=layer),
        SimpleNamespace(lm_head_impl=impl, enable_fp32_lm_head=fp32),
        "actor",
        True,
    )
    hidden = torch.randn(5, 2, 3, requires_grad=True)
    output, _ = layer(hidden)
    grad = torch.randn_like(output)
    output.backward(grad)
    expected_input = grad.matmul(layer.weight.detach())
    if sequence_parallel:
        first, second = expected_input.chunk(2, dim=0)
        expected_input = first + second
        full_input = torch.cat([hidden.detach(), hidden.detach()], dim=0)
        assert calls.count("gather") == 1
        assert calls.count("gather_async") == int(not frozen)
        assert calls.count("scatter") == 1
    else:
        expected_input *= 2
        full_input = hidden.detach()
        assert calls.count("all_reduce") == 1
    assert calls.count("wait") == 1 + int(sequence_parallel and not frozen)
    torch.testing.assert_close(hidden.grad, expected_input)
    if not frozen:
        torch.testing.assert_close(layer.weight.grad, grad.reshape(-1, 7).t().mm(full_input.reshape(-1, 3)))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
