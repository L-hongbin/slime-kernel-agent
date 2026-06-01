"""Small INT8 RTN helpers shared by offline and online quantization paths."""

from __future__ import annotations

import torch


@torch.no_grad()
def quantize_layer_int8(weight, group_size, strategy, sym=True, scale_dtype=torch.float32):
    """RTN INT8 quantization producing the `int-quantized` raw-int8 layout.

    Output matches what sglang's `compressed_tensors_w8a8_int8` scheme loads:
      - weight: torch.int8 (sym) or torch.uint8 (asym), shape (out, in)
      - weight_scale: scale_dtype, shape depends on strategy
            "channel": (out, 1)
            "tensor":  (1,)
            "group":   (out, in/group_size)
      - weight_zero_point: torch.int32, same shape as weight_scale, or None for sym

    Symmetric clamp range is [-127, 127] to stay consistent with
    scale = absmax / 127; -128 is unreachable when scale is computed from absmax.
    """
    out_features, in_features = weight.shape
    w_fp32 = weight.to(torch.float32)

    if strategy == "tensor":
        if sym:
            absmax = w_fp32.abs().amax()
            scale_scalar = (absmax / 127.0).clamp(min=1e-8)
            q = torch.round(w_fp32 / scale_scalar).clamp(-127, 127).to(torch.int8)
            scale = scale_scalar.reshape(1).to(scale_dtype).contiguous()
            return q.contiguous(), scale, None
        wmin = w_fp32.amin()
        wmax = w_fp32.amax()
        scale_scalar = ((wmax - wmin) / 255.0).clamp(min=1e-8)
        zp_f = torch.round(-wmin / scale_scalar).clamp(0, 255)
        q = (torch.round(w_fp32 / scale_scalar) + zp_f).clamp(0, 255).to(torch.uint8)
        scale = scale_scalar.reshape(1).to(scale_dtype).contiguous()
        zp = zp_f.reshape(1).to(torch.int32).contiguous()
        return q.contiguous(), scale, zp

    if strategy == "channel" or group_size is None or group_size <= 0:
        eff_group_size = in_features
    else:
        if in_features % group_size != 0:
            raise ValueError(
                f"in_features={in_features} not divisible by group_size={group_size} "
                f"(weight shape={tuple(weight.shape)})"
            )
        eff_group_size = group_size
    num_groups = in_features // eff_group_size

    w_grouped = w_fp32.reshape(out_features, num_groups, eff_group_size)

    if sym:
        absmax = w_grouped.abs().amax(dim=-1, keepdim=True)
        scale = (absmax / 127.0).clamp(min=1e-8)
        q = torch.round(w_grouped / scale).clamp(-127, 127).to(torch.int8)
        q = q.reshape(out_features, in_features).contiguous()
        scale = scale.reshape(out_features, num_groups).to(scale_dtype).contiguous()
        return q, scale, None

    wmin = w_grouped.amin(dim=-1, keepdim=True)
    wmax = w_grouped.amax(dim=-1, keepdim=True)
    scale = ((wmax - wmin) / 255.0).clamp(min=1e-8)
    zp_f = torch.round(-wmin / scale).clamp(0, 255)
    q = (torch.round(w_grouped / scale) + zp_f).clamp(0, 255).to(torch.uint8)
    q = q.reshape(out_features, in_features).contiguous()
    scale = scale.reshape(out_features, num_groups).to(scale_dtype).contiguous()
    zp = zp_f.reshape(out_features, num_groups).to(torch.int32).contiguous()
    return q, scale, zp
