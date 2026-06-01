import logging
import math
import re

import torch
import torch.nn as nn

from scripts.quantize.utils.int8_quantization import quantize_layer_int8

try:
    import fake_int4_quant_cuda
except ImportError:
    fake_int4_quant_cuda = None

logger = logging.getLogger(__name__)


__all__ = ["quantize_params_compressed_tensors"]


class WQLinear_GEMM(nn.Module):
    def __init__(self, w_bit, group_size, in_features, out_features, bias, dev, training=False):
        super().__init__()

        if w_bit not in [4]:
            raise NotImplementedError("Only 4-bit are supported for now.")

        self.in_features = in_features
        self.out_features = out_features
        self.w_bit = w_bit
        self.group_size = group_size if group_size != -1 else in_features
        self.training = training

        # quick sanity check (make sure alignment)
        assert self.in_features % self.group_size == 0
        assert out_features % (32 // self.w_bit) == 0

        self.register_buffer(
            "qweight",
            torch.zeros(
                (in_features, out_features // (32 // self.w_bit)),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (in_features // self.group_size, out_features // (32 // self.w_bit)),
                dtype=torch.int32,
                device=dev,
            ),
        )
        self.register_buffer(
            "scales",
            torch.zeros(
                (in_features // self.group_size, out_features),
                dtype=torch.float16,
                device=dev,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(
                    (out_features),
                    dtype=torch.float16,
                    device=dev,
                ),
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(cls, linear, w_bit, group_size, init_only=False, scales=None, zeros=None):
        awq_linear = cls(
            w_bit,
            group_size,
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            linear.weight.device,
        )
        if init_only:  # just prepare for loading sd
            return awq_linear

        # need scales and zeros info for real quantization
        assert scales is not None and zeros is not None

        awq_linear.scales = scales.clone().half()
        if linear.bias is not None:
            awq_linear.bias = linear.bias.clone().half()

        pack_num = 32 // awq_linear.w_bit
        device = torch.device(f"cuda:{torch.cuda.current_device()}")

        repeat_scales = scales.to(device).t().repeat_interleave(group_size, 1)
        if isinstance(zeros, torch.Tensor):
            repeat_zeros = zeros.to(device).t().repeat_interleave(group_size, 1)
        else:
            repeat_zeros = zeros
        intweight = torch.round(linear.weight.to(device) / repeat_scales + repeat_zeros).to(torch.int).t().contiguous()
        intweight = intweight.to(dtype=torch.int32)
        del repeat_scales

        intweight = intweight.reshape(-1, intweight.shape[1] // pack_num, pack_num)

        new_order_map = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=device) * awq_linear.w_bit
        intweight = intweight << new_order_map
        intweight = torch.sum(intweight, dim=-1).to(torch.int32)
        awq_linear.qweight = intweight

        if isinstance(zeros, torch.Tensor):
            zeros = zeros.to(dtype=torch.int32, device=device)
            zeros = zeros.reshape(-1, zeros.shape[1] // pack_num, pack_num)
            zeros = zeros << new_order_map
            qzeros = torch.sum(zeros, dim=-1).to(torch.int32)

        else:
            value = 0
            for i in range(pack_num):
                value |= zeros << (i * awq_linear.w_bit)
            qzeros = (
                torch.ones(
                    (scales.shape[0], scales.shape[1] // pack_num),
                    dtype=torch.int32,
                    device=device,
                )
                * value
            )

        awq_linear.qzeros = qzeros

        return awq_linear


def pack_to_int32(
    value,
    num_bits,
    packed_dim=1,
    sym=False,
):
    # if value.dtype is not torch.int8:
    #     raise ValueError("Tensor must be quantized to torch.int8 before packing")

    if num_bits > 8:
        raise ValueError("Packing is only supported for less than 8 bits")

    if num_bits < 1:
        raise ValueError(f"num_bits must be at least 1, got {num_bits}")

    # Convert to unsigned range for packing, matching quantization offset
    if sym:
        offset = 1 << (num_bits - 1)
        value = (value + offset).to(torch.uint8)
    device = value.device

    pack_factor = 32 // num_bits

    if packed_dim == 0:
        value = value.transpose(0, 1)

    rows, cols = value.shape
    padded_cols = math.ceil(cols / pack_factor) * pack_factor
    pad_len = padded_cols - cols

    if pad_len > 0:
        value = torch.nn.functional.pad(value, (0, pad_len))

    num_groups = padded_cols // pack_factor

    # Use int32 here
    reshaped = value.view(rows, num_groups, pack_factor).to(torch.int32)
    bit_shifts = torch.arange(pack_factor, device=device, dtype=torch.int32) * num_bits
    packed = (reshaped << bit_shifts).sum(dim=2, dtype=torch.int32)

    if packed_dim == 0:
        packed = packed.transpose(0, 1)

    return packed


def round_to_quantized_type_dtype(
    tensor,
    dtype,
    cast_to_original_dtype=False,
):
    original_dtype = tensor.dtype
    iinfo = torch.iinfo(dtype)
    rounded = torch.round(torch.clamp(tensor, iinfo.min, iinfo.max)).to(dtype)
    if cast_to_original_dtype:
        return rounded.to(original_dtype)
    return rounded


@torch.no_grad()
def quantize(
    x,
    scale,
    zero_point,
    dtype=torch.int8,
):
    group_size = x.shape[-1] // scale.shape[-1]
    output_dtype = dtype
    output = torch.zeros_like(x).to(output_dtype)

    reshaped_dims = (
        math.ceil(x.shape[-1] / group_size),
        group_size,
    )
    x = x.unflatten(-1, reshaped_dims)

    scaled = x / scale.unsqueeze(-1)

    if zero_point is not None:
        zero_point = zero_point.unsqueeze(-1)
        scaled += zero_point.to(x.dtype)

    # clamp and round
    output = round_to_quantized_type_dtype(tensor=scaled, dtype=dtype)

    output = output.flatten(start_dim=-2)
    output = output.to(output_dtype)

    return output


def if_quant(name, patterns):
    for pattern in patterns:
        if re.search(pattern, name):
            return True
    return False


def pack_layer(weight, group_size, sym=True):
    w, scale, zp = fake_int4_quant_cuda.fake_int4_quant_cuda(weight, (1, group_size), sym)
    w = w.view(weight.shape[0], 1, weight.shape[1] // group_size, group_size)
    scale = scale.view(weight.shape[0], 1, weight.shape[1] // group_size, 1)
    zp = zp.view(weight.shape[0], 1, weight.shape[1] // group_size, 1)
    if sym:
        w = w * scale
    else:
        w = (w - zp) * scale
    w = w.view(weight.shape)
    scale = scale.view(weight.shape[0], -1).contiguous()
    if not sym:
        zp = zp.view(weight.shape[0], -1)
        zeros = zp.t().contiguous().to(torch.float32)
        zeros = zeros.to(dtype=torch.int32, device=w.device)
        zeros = zeros.reshape(-1, zeros.shape[1] // 8, 8)
        new_order_map = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7], device=zeros.device) * 4
        zeros = zeros << new_order_map
        packed_zp = torch.sum(zeros, dim=-1).to(torch.int32)
    else:
        zp = None
        packed_zp = None

    quantized_weight = quantize(
        x=w,
        scale=scale,
        zero_point=zp,
        dtype=torch.int8 if sym else torch.uint8,
    )
    packed_weight = pack_to_int32(quantized_weight, 4, sym=sym)
    return packed_weight, scale, packed_zp


def quantize_params_compressed_tensors(converted_named_params, quantization_config):
    group_cfg = quantization_config["config_groups"]["group_0"]
    w_cfg = group_cfg["weights"]
    num_bits = w_cfg.get("num_bits", 4)
    is_symmetric = w_cfg["symmetric"]
    strategy = w_cfg.get("strategy")  # "channel" | "group" | "tensor"
    group_size = w_cfg.get("group_size")
    ignore_rules = quantization_config.get("ignore", [])

    fmt = quantization_config.get("format")
    if fmt is None:
        # INT8 -> raw int8 (`int-quantized`); INT4 -> packed (`pack-quantized`) since
        # the only wired INT4 path is the WNA16-style CUDA kernel.
        fmt = "int-quantized" if num_bits == 8 else "pack-quantized"

    if num_bits not in (4, 8):
        raise NotImplementedError(
            f"compressed-tensors quantize: unsupported num_bits={num_bits}; only 4 and 8 are wired."
        )

    use_raw_int8 = num_bits == 8 and fmt == "int-quantized"

    has_input_act_quant = group_cfg.get("input_activations") is not None
    if use_raw_int8 and strategy == "group" and has_input_act_quant:
        # SGLang's compressed_tensors_w8a8_int8 only loads per-channel / per-tensor
        # weight_scale. A per-group `int-quantized` weight + INT8 activations checkpoint
        # would silently produce shapes the W8A8 loader cannot consume.
        raise NotImplementedError(
            "compressed-tensors W8A8 INT8 (input_activations set) does not support "
            "per-group weight strategy: sglang's compressed_tensors_w8a8_int8 scheme "
            "registers weight_scale as (output_size, 1) per-channel or "
            "(num_partitions,) per-tensor only. Use strategy='channel' or 'tensor', "
            "or drop input_activations for a weight-only WNA16-style config."
        )

    results = []

    for name, param in converted_named_params:
        is_ignored = any(
            (r.startswith("re:") and re.match(r[3:], name)) or r == name or name.startswith(r) for r in ignore_rules
        )

        if is_ignored or not name.endswith(".weight") or param.dim() < 2:
            results.append((name, param))
            continue

        if use_raw_int8:
            qw, s, zp = quantize_layer_int8(param, group_size, strategy, is_symmetric)
            results.append((name, qw))
            results.append((name.replace(".weight", ".weight_scale"), s))
            if zp is not None:
                results.append((name.replace(".weight", ".weight_zero_point"), zp))
        else:
            # INT4 packed (WNA16) path; CUDA kernel only supports INT4.
            if num_bits != 4:
                raise NotImplementedError(
                    f"compressed-tensors pack-quantized num_bits={num_bits} not supported "
                    "(fake_int4_quant_cuda kernel is INT4-only; set format='int-quantized' for INT8)"
                )
            qw, s, zp = pack_layer(param, group_size, is_symmetric)
            qweight_name = name.replace(".weight", ".weight_packed")
            scale_name = name.replace(".weight", ".weight_scale")
            weight_shape = torch.tensor(param.shape, dtype=torch.int32, device="cuda")
            weight_shape_name = name.replace(".weight", ".weight_shape")
            if zp is not None:
                zp_name = name.replace(".weight", ".weight_zero_point")
                results.append((zp_name, zp))
            results.append((qweight_name, qw))
            results.append((scale_name, s))
            results.append((weight_shape_name, weight_shape))

    return results
