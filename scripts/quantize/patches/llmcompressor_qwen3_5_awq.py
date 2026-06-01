"""Runtime patch for llmcompressor AWQ on Qwen3.5 offset RMSNorm.

Qwen3_5RMSNorm stores an offset parameter and applies it as ``(1 + weight)``.
llmcompressor's AWQ smoothing treats smooth-layer output scale as ``weight``,
so it updates RMSNorm weights with ``weight / scales``.  For Qwen3_5RMSNorm
that produces ``1 + weight / scales``, not the required ``(1 + weight) / scales``.
"""

from __future__ import annotations

from types import MethodType

import torch
from torch.nn import Module


def is_qwen3_5_offset_rmsnorm(module: Module) -> bool:
    return module.__class__.__name__ == "Qwen3_5RMSNorm"


def awq_smooth_weight_for_module(module: Module, scales: torch.Tensor) -> torch.Tensor:
    """Return the AWQ-smoothed smooth-layer weight for a module."""
    if module.weight.ndim != 1:
        weight = module.weight.clone()
        weight[-scales.size(0) :].div_(scales.view(-1, 1))
        return weight
    if is_qwen3_5_offset_rmsnorm(module):
        return (module.weight + 1.0) / scales - 1.0
    return module.weight / scales


def patch_awq_qwen3_5_rmsnorm() -> None:
    """Patch llmcompressor AWQModifier in-place for Qwen3_5RMSNorm smoothing."""
    from compressed_tensors.offload import update_offload_parameter
    from compressed_tensors.utils import align_modules
    from llmcompressor.modifiers.awq.base import AWQModifier
    from llmcompressor.modifiers.utils.hooks import HooksMixin
    from llmcompressor.utils.helpers import calibration_forward_context
    from loguru import logger
    from tqdm import tqdm

    if getattr(AWQModifier, "_slime_qwen3_5_rmsnorm_patch", False):
        return

    @torch.no_grad()
    def _apply_smoothing(self: AWQModifier, model: Module) -> None:
        mappings_to_smooth = [
            mapping for mapping in self._resolved_mappings if mapping.smooth_name in self._smooth_activation_stats
        ]
        for mapping in tqdm(mappings_to_smooth, desc="Smoothing"):
            smooth_layer = mapping.smooth_layer
            balance_layers = mapping.balance_layers
            parent_module = mapping.parent

            with (
                align_modules([parent_module, smooth_layer, *balance_layers]),
                calibration_forward_context(model),
                HooksMixin.disable_hooks(),
            ):
                fp16_outputs = self._run_samples(parent_module)
                if len(fp16_outputs) == 0 or all(f.numel() == 0 for f in fp16_outputs):
                    logger.info(
                        f"Skipping smooth_layer {mapping.smooth_name}, no activations "
                        "found to scale. This can occasionally occur in MoE models "
                        "when certain experts are not activated by calibration samples."
                    )
                    del self._smooth_activation_stats[mapping.smooth_name]
                    continue
                if not all([fp16_output.isfinite().all() for fp16_output in fp16_outputs]):
                    logger.warning(
                        f"Skipping smooth_layer {mapping.smooth_name}, NaN or inf "
                        "outputs found during forward pass of the parent module "
                        f"{mapping.parent_name}. The model is either generating NaN "
                        "output with provided calibration data set, or the mappings "
                        "are incorrectly set and modifying the model in undesired ways."
                    )
                    del self._smooth_activation_stats[mapping.smooth_name]
                    continue

                orig_layer_weights = {
                    balance_layer: balance_layer.weight.clone() for balance_layer in mapping.balance_layers
                }

                best_scales = self._compute_best_scale(mapping, fp16_outputs, orig_layer_weights)

                @torch.no_grad()
                def smooth(module: Module, orig_weights: dict[Module, torch.Tensor]) -> None:
                    scales = best_scales.to(module.weight.device)  # noqa: B023
                    if module in balance_layers:  # noqa: B023
                        update_offload_parameter(
                            module,
                            "weight",
                            orig_weights[module].to(module.weight.device) * scales.view(1, -1),
                        )
                    elif module == smooth_layer:  # noqa: B023
                        update_offload_parameter(
                            module,
                            "weight",
                            awq_smooth_weight_for_module(module, scales),
                        )
                        if hasattr(module, "bias") and module.bias is not None:
                            update_offload_parameter(
                                module,
                                "bias",
                                module.bias.div_(scales),
                            )

                for layer in balance_layers:
                    smooth(layer, orig_layer_weights)
                smooth(smooth_layer, orig_layer_weights)

                del self._smooth_activation_stats[mapping.smooth_name]
                del orig_layer_weights

        for cache in self._parent_args_cache.values():
            cache.batch_intermediates.clear()
        self._assert_all_activations_consumed()

    AWQModifier._apply_smoothing = MethodType(_apply_smoothing, AWQModifier).__func__
    AWQModifier._slime_qwen3_5_rmsnorm_patch = True
