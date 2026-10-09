"""Normal/flatten vocab projections and optional FP32 math for Megatron LM heads.

Parameter dtypes and checkpoint layouts are preserved. Native normal heads use
Megatron unchanged; native flatten heads replace only the local projection hook.
FP32 heads upcast projection operands while retaining the original parameters.
"""

from __future__ import annotations

from types import MethodType

import torch
from megatron.core.parallel_state import get_global_memory_buffer
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, dist_all_gather_func, dist_reduce_scatter_func
from megatron.core.tensor_parallel.mappings import (
    copy_to_tensor_model_parallel_region,
    gather_from_tensor_model_parallel_region,
)
from megatron.core.utils import prepare_input_tensors_for_wgrad_compute

try:
    import fused_weight_gradient_mlp_cuda

    _FUSED_WGRAD_AVAILABLE = True
except ImportError:
    fused_weight_gradient_mlp_cuda = None
    _FUSED_WGRAD_AVAILABLE = False

try:
    from transformer_engine.pytorch.module.base import get_dummy_wgrad

    _HAVE_TE = True
except ImportError:
    get_dummy_wgrad = None
    _HAVE_TE = False


class _LmHeadLinear(torch.autograd.Function):
    """Column-parallel projection with selectable dtype/layout and main_grad support."""

    @staticmethod
    def forward(
        ctx,
        input_: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        gradient_accumulation_fusion: bool,
        allreduce_dgrad: bool,
        sequence_parallel: bool,
        grad_output_buffer: list[torch.Tensor] | None,
        wgrad_deferral_limit: int | None,
        tp_group,
        fp32: bool,
        impl: str,
    ) -> torch.Tensor:
        if gradient_accumulation_fusion and weight.requires_grad and hasattr(weight, "main_grad"):
            main_grad = weight.main_grad
        else:
            main_grad = None

        ctx.save_for_backward(input_, weight)
        ctx.main_grad = main_grad
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.sequence_parallel = sequence_parallel
        ctx.grad_output_buffer = grad_output_buffer
        ctx.wgrad_deferral_limit = wgrad_deferral_limit
        ctx.tp_group = tp_group
        ctx.compute_dtype = torch.float32 if fp32 else input_.dtype
        ctx.impl = impl

        if sequence_parallel:
            dim_size = list(input_.size())
            dim_size[0] *= tp_group.size()
            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
            dist_all_gather_func(all_gather_buffer, input_, group=tp_group)
            total_input = all_gather_buffer
        else:
            total_input = input_

        if impl == "flatten":
            input_2d = total_input.reshape(-1, total_input.shape[-1]).to(ctx.compute_dtype)
            # Megatron/MTP cross entropy mutates logits in place. Return an owned
            # tensor, not a view created inside this custom autograd Function.
            output = torch.empty(
                (*total_input.shape[:-1], weight.shape[0]), dtype=ctx.compute_dtype, device=input_2d.device
            )
            torch.mm(input_2d, weight.to(ctx.compute_dtype).t(), out=output.view(-1, weight.shape[0]))
        else:
            output = torch.matmul(total_input.to(ctx.compute_dtype), weight.to(ctx.compute_dtype).t())
        if bias is not None:
            output = output + bias.to(ctx.compute_dtype)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input_, weight = ctx.saved_tensors
        tp_group = ctx.tp_group

        if ctx.gradient_accumulation_fusion and weight.requires_grad:
            weight.main_grad = ctx.main_grad

        compute_wgrad = weight.requires_grad
        if compute_wgrad and ctx.grad_output_buffer is not None:
            if ctx.wgrad_deferral_limit == 0 or len(ctx.grad_output_buffer) < ctx.wgrad_deferral_limit:
                ctx.grad_output_buffer.append(grad_output)
                compute_wgrad = False

        if compute_wgrad and ctx.sequence_parallel:
            dim_size = list(input_.size())
            dim_size[0] *= tp_group.size()
            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
            gather_handle = dist_all_gather_func(all_gather_buffer, input_, group=tp_group, async_op=True)
            total_input = all_gather_buffer
        else:
            gather_handle = None
            total_input = input_

        if ctx.impl == "flatten":
            grad_output_2d = grad_output.reshape(-1, grad_output.shape[-1]).to(ctx.compute_dtype)
            grad_input = torch.mm(grad_output_2d, weight.to(ctx.compute_dtype))
            grad_input = grad_input.reshape(*grad_output.shape[:-1], weight.shape[1]).to(input_.dtype)
        else:
            grad_input = torch.matmul(grad_output.to(ctx.compute_dtype), weight.to(ctx.compute_dtype)).to(input_.dtype)

        if gather_handle is not None:
            gather_handle.wait()

        dgrad_handle = None
        if ctx.allreduce_dgrad:
            dgrad_handle = torch.distributed.all_reduce(grad_input, group=tp_group, async_op=True)

        if ctx.sequence_parallel:
            sub_grad_input = torch.empty_like(input_, requires_grad=False)
            dgrad_handle = dist_reduce_scatter_func(sub_grad_input, grad_input, group=tp_group, async_op=True)

        grad_weight = None
        if weight.requires_grad:
            if ctx.gradient_accumulation_fusion:
                if compute_wgrad:
                    _accumulate_main_grad(total_input.to(ctx.compute_dtype), grad_output.to(ctx.compute_dtype), weight)
                grad_weight = _dummy_grad_weight(weight, input_.dtype)
                if hasattr(weight, "grad_added_to_main_grad"):
                    weight.grad_added_to_main_grad = True
            elif compute_wgrad:
                grad_output_2d, total_input_2d = prepare_input_tensors_for_wgrad_compute(
                    grad_output.to(ctx.compute_dtype), total_input.to(ctx.compute_dtype)
                )
                grad_weight = grad_output_2d.t().matmul(total_input_2d)

        grad_bias = None
        if ctx.use_bias:
            grad_bias = grad_output.reshape(-1, grad_output.shape[-1]).sum(dim=0)

        if dgrad_handle is not None:
            dgrad_handle.wait()

        if ctx.sequence_parallel:
            return sub_grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None
        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None


def _accumulate_main_grad(total_input: torch.Tensor, grad_output: torch.Tensor, weight: torch.Tensor) -> None:
    grad_output_2d, total_input_2d = prepare_input_tensors_for_wgrad_compute(grad_output, total_input)
    main_grad = weight.main_grad
    if main_grad is None:
        raise RuntimeError("LM head requires weight.main_grad for gradient accumulation fusion.")

    if (
        _FUSED_WGRAD_AVAILABLE
        and total_input_2d.is_cuda
        and grad_output_2d.is_cuda
        and main_grad.is_cuda
        and (
            main_grad.dtype == torch.float32
            or (main_grad.dtype in (torch.float16, torch.bfloat16) and total_input_2d.dtype == main_grad.dtype)
        )
    ):
        if main_grad.dtype == torch.float32:
            fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(total_input_2d, grad_output_2d, main_grad)
        else:
            fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(total_input_2d, grad_output_2d, main_grad)
    else:
        main_grad.add_(grad_output_2d.t().matmul(total_input_2d).to(main_grad.dtype))


def _dummy_grad_weight(weight: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor | None:
    if not hasattr(weight, "grad_added_to_main_grad"):
        return None

    if getattr(weight, "zero_out_wgrad", False):
        if _HAVE_TE and weight.is_cuda:
            return get_dummy_wgrad(list(weight.main_grad.shape), input_dtype, zero=True)
        return torch.zeros(weight.main_grad.shape, dtype=input_dtype, device=weight.device, requires_grad=False)

    if _HAVE_TE and weight.is_cuda:
        return get_dummy_wgrad(list(weight.main_grad.shape), input_dtype)
    return torch.empty(weight.main_grad.shape, dtype=input_dtype, device=weight.device, requires_grad=False)


def _fp32_lm_head_forward(
    self,
    input_: torch.Tensor,
    weight: torch.Tensor | None = None,
    runtime_gather_output: bool | None = None,
    output_cross_entropy_loss: bool = False,
    labels: torch.Tensor | None = None,
    reduction: str = "none",
    ignore_index: int = -100,
):
    del labels, reduction, ignore_index
    if output_cross_entropy_loss:
        raise NotImplementedError("--enable-fp32-lm-head requires cross_entropy_loss_fusion=False.")

    if weight is None:
        if self.weight is None:
            raise RuntimeError("output_layer weight is not allocated and no weight was supplied.")
        weight = self.weight
    else:
        expected_shape = (self.output_size_per_partition, self.input_size)
        if weight.shape != expected_shape:
            raise RuntimeError(f"supplied weight's shape is {tuple(weight.shape)}, not {expected_shape} as expected")

    bias = self.bias if not self.skip_bias_add else None

    if self.allreduce_dgrad or self.sequence_parallel or self.explicit_expert_comm or self.disable_grad_reduce:
        input_parallel = input_
    else:
        input_parallel = copy_to_tensor_model_parallel_region(input_, group=self.tp_group)

    if self.config.defer_embedding_wgrad_compute:
        if (
            self.config.wgrad_deferral_limit == 0
            or len(self.embedding_activation_buffer) < self.config.wgrad_deferral_limit
        ):
            self.embedding_activation_buffer.append(input_parallel.float())

    allreduce_dgrad = False if self.explicit_expert_comm else self.allreduce_dgrad
    output_parallel = _LmHeadLinear.apply(
        input_parallel,
        weight,
        bias,
        self.gradient_accumulation_fusion,
        allreduce_dgrad,
        False if self.explicit_expert_comm else self.sequence_parallel,
        self.grad_output_buffer if self.config.defer_embedding_wgrad_compute else None,
        self.config.wgrad_deferral_limit if self.config.defer_embedding_wgrad_compute else None,
        self.tp_group,
        True,
        self._slime_lm_head_impl,
    )

    gather_output = self.gather_output if runtime_gather_output is None else runtime_gather_output
    if gather_output:
        output = gather_from_tensor_model_parallel_region(output_parallel, group=self.tp_group)
    else:
        output = output_parallel
    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias


def enable_fp32_lm_head(model: torch.nn.Module, impl: str = "normal") -> None:
    """Patch ``model.output_layer`` so only the vocab projection GEMM runs in fp32."""
    if impl not in ("normal", "flatten"):
        raise ValueError(f"Unsupported LM head implementation: {impl}")
    output_layer = getattr(model, "output_layer", None)
    if output_layer is None:
        raise RuntimeError("--enable-fp32-lm-head was requested on a post-process model without output_layer.")
    output_layer._slime_lm_head_impl = impl
    if getattr(output_layer, "_slime_fp32_lm_head_enabled", False):
        return
    output_layer.forward = MethodType(_fp32_lm_head_forward, output_layer)
    output_layer._slime_fp32_lm_head_enabled = True


def _flatten_lm_head_forward_impl(
    self,
    input,
    weight,
    bias,
    gradient_accumulation_fusion,
    allreduce_dgrad,
    sequence_parallel,
    grad_output_buffer,
    wgrad_deferral_limit,
    tp_group,
):
    return _LmHeadLinear.apply(
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
        tp_group,
        False,
        "flatten",
    )


def enable_flatten_lm_head(model: torch.nn.Module) -> None:
    """Replace only the local projection of a native ColumnParallelLinear head."""
    output_layer = getattr(model, "output_layer", None)
    if not isinstance(output_layer, ColumnParallelLinear):
        raise TypeError("--lm-head-impl flatten requires a Megatron ColumnParallelLinear output_layer")
    if not callable(getattr(output_layer, "_forward_impl", None)):
        raise TypeError(
            "This Megatron ColumnParallelLinear does not expose the _forward_impl hook required for flatten"
        )
    output_layer._forward_impl = MethodType(_flatten_lm_head_forward_impl, output_layer)
    output_layer._slime_lm_head_impl = "flatten"
