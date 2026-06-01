"""Runtime patch for llmcompressor SmoothQuant on Qwen3.5 offset RMSNorm.

Qwen3_5RMSNorm stores an offset parameter and applies it as `(1 + weight)`.
llmcompressor's SmoothQuant implementation treats all smooth layers as if the
module output scale is directly `weight`, so it updates RMSNorm weights with
`weight / scales`.  For Qwen3_5RMSNorm this produces `1 + weight / scales`,
which is not equal to the required `(1 + weight) / scales`.
"""

from __future__ import annotations

from types import MethodType

import torch
from torch.nn import Module


def is_qwen3_5_offset_rmsnorm(module: Module) -> bool:
    return module.__class__.__name__ == "Qwen3_5RMSNorm"


def smooth_weight_for_module(module: Module, scales: torch.Tensor) -> torch.Tensor:
    """Return the SmoothQuant-smoothed smooth-layer weight for a module.

    For ordinary norm layers llmcompressor's original `weight / scales` is
    correct.  Qwen3_5RMSNorm is special because its forward multiplier is
    `(1 + weight)`, so the parameter update must preserve:

        norm(x) * (1 + w') == norm(x) * (1 + w) / scales

    which gives `w' = (1 + w) / scales - 1`.
    """
    if module.weight.ndim != 1:
        return module.weight / scales.view(-1, 1)
    if is_qwen3_5_offset_rmsnorm(module):
        return (module.weight + 1.0) / scales - 1.0
    return module.weight / scales


def patch_smoothquant_qwen3_5_rmsnorm() -> None:
    """Patch llmcompressor SmoothQuantModifier in-place.

    The patch is intentionally narrow: only the smooth-layer weight update is
    changed, and only `Qwen3_5RMSNorm` uses the offset-aware formula.  Balance
    layer scaling, bias handling, distributed activation-scale reduction, and
    calibration-data cleanup remain identical to llmcompressor's implementation.
    """
    from compressed_tensors.offload import update_offload_parameter
    from compressed_tensors.offload.dist_utils import is_distributed
    from llmcompressor.modifiers.transform.smoothquant.base import MINIMUM_SMOOTHING_SCALE, SmoothQuantModifier
    from loguru import logger

    if getattr(SmoothQuantModifier, "_slime_qwen3_5_rmsnorm_patch", False):
        return

    @torch.no_grad()
    def _apply_smoothing(self: SmoothQuantModifier, model: Module):
        if is_distributed():
            self._reduce_activation_scales()

        for mapping in self.resolved_mappings_:
            if mapping.smooth_name not in self.scales_:
                continue
            logger.info(f"Smoothing with {mapping.smooth_name}")

            activation_scales = (
                self.scales_[mapping.smooth_name].max_channel_vals - self.scales_[mapping.smooth_name].min_channel_vals
            )
            smooth_layer = mapping.smooth_layer
            balance_layers = mapping.balance_layers

            scales = self._calculate_smoothing_scales(balance_layers, activation_scales)
            scales = torch.maximum(scales, torch.Tensor([MINIMUM_SMOOTHING_SCALE]).to(scales.device))

            @torch.no_grad()
            def smooth(module):
                if module in balance_layers:  # noqa: B023
                    update_offload_parameter(module, "weight", module.weight * scales.view(1, -1))  # noqa: B023
                elif module == smooth_layer:  # noqa: B023
                    update_offload_parameter(module, "weight", smooth_weight_for_module(module, scales))  # noqa: B023

                    if hasattr(module, "bias") and module.bias is not None:
                        update_offload_parameter(module, "bias", module.bias / scales)  # noqa: B023

            for layer in balance_layers:
                smooth(layer)
            smooth(smooth_layer)

            del self.scales_[mapping.smooth_name]

    SmoothQuantModifier._apply_smoothing = MethodType(_apply_smoothing, SmoothQuantModifier).__func__
    SmoothQuantModifier._slime_qwen3_5_rmsnorm_patch = True
