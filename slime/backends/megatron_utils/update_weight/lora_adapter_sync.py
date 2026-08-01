"""LoRA-adapter weight sync: push only the trainable adapter to sglang.

Default full/merge sync (``update_weight_from_distributed.py``) sends the *merged*
base+LoRA linear (a full 2D weight per wrapped module) every RL step. This module
implements the alternative: extract only the trainable (``requires_grad=True``)
adapter tensors, rename them to PEFT/HF convention, and hand them to sglang's
``load_lora_adapter_from_tensors`` path so the engine serves ``base + adapter``.

Gated behind ``--use-lora-weight-sync`` (default OFF);
when off, none of this runs and the merge path is byte-identical.

Scope: "trainable" is defined as ``requires_grad=True`` on the megatron side.
Current attention and replicated ``shared_experts`` LoRA targets flow through the
canonical PP-source sync. Routed-expert LoRA names containing ``.experts.`` are
rejected until the distributed sync implements an EP gather; silently selecting
one EP shard would produce an incomplete adapter. The megatron→serving name map is
the *same* converter used for the base weights (``convert_deepseekv4_to_hf``), so
supported adapter keys always agree with the served module they modify.

SGLANG-SIDE PORT (see handoffs/deepseek-v4/lora_serve_design.md): the sglang
DeepSeek-V4 model registers these 6 modules for LoRA — ``get_hidden_dim`` +
``get_stacked_multiply`` + ``supported_lora_modules`` on ``DeepseekV4ForCausalLM``,
the V4 leaf names in ``_KNOWN_LORA_TARGET_MODULES``/``REPLICATED_LINEAR_LORA_NAMES``/
``SUPPORTED_LORA_TARGET_MODULES``. Launch with ``SGLANG_OPT_FUSE_WQA_WKV=0`` so
``wq_a``/``wkv`` are separate targets, and the compressor ``wkv``/``wgate`` fuse to
the served ``wkv_gate`` (``lora.py`` ``normalize_wkv_gate``) whose forward-bypassing
fused kernel gets an explicit ``apply_lora`` delta (``compressor.py``). Requires
attn-tp==1 (dp-attention). Everything on the slime side is unit-tested offline
(``tests/deepseek-v4/test_dsv4_lora_serve.py``).
"""

from __future__ import annotations

from argparse import Namespace
from collections.abc import Sequence

import torch

from slime.utils.lora_utils import (  # re-exported for backward compatibility
    SLIME_LORA_ADAPTER_NAME,
    all_alternating_lora_names,
    lora_adapter_name,
    plan_lora_swap,
    raise_on_failed_lora_load,
    rollout_lora_path,
    use_lora_weight_sync,
)

from ..megatron_to_hf.deepseekv4 import convert_deepseekv4_to_hf

__all__ = [
    "SLIME_LORA_ADAPTER_NAME",
    "all_alternating_lora_names",
    "lora_adapter_name",
    "plan_lora_swap",
    "raise_on_failed_lora_load",
    "rollout_lora_path",
    "use_lora_weight_sync",
    "adapter_side_and_base",
    "is_adapter_param_name",
    "megatron_adapter_name_to_peft",
    "peft_target_module_leaf",
    "build_lora_adapter_state_dict",
]

_LORA_IN_SUFFIX = ".linear_in.weight"
_LORA_OUT_SUFFIX = ".linear_out.weight"
_WEIGHT_SUFFIX = ".weight"

# PEFT weight-key convention consumed by sglang's LoRAAdapter loader. sglang only
# needs a ``layers.<N>.`` substring (get_layer_id) plus the wrapped module's leaf
# name; the ``base_model.model.`` prefix matches PEFT's on-disk adapter naming.
_PEFT_PREFIX = "base_model.model."


def adapter_side_and_base(name: str) -> tuple[str | None, str | None]:
    """Return (``"A"``/``"B"``, base-module-name) for a LoRA adapter param, else (None, None).

    ``linear_in`` is PEFT ``lora_A`` (down-projection), ``linear_out`` is ``lora_B``
    (up-projection). Base module name is the wrapped linear without the adapter leaf,
    e.g. ``...self_attn.q_a_proj``.
    """
    if name.endswith(_LORA_IN_SUFFIX):
        return "A", name[: -len(_LORA_IN_SUFFIX)]
    if name.endswith(_LORA_OUT_SUFFIX):
        return "B", name[: -len(_LORA_OUT_SUFFIX)]
    return None, None


def is_adapter_param_name(name: str) -> bool:
    side, _ = adapter_side_and_base(name)
    return side is not None


def megatron_adapter_name_to_peft(args: Namespace, name: str, param: torch.Tensor | None = None) -> str | None:
    """Map a megatron global adapter param name to its PEFT/sglang key.

    ``module.module.layers.3.self_attn.q_a_proj.linear_in.weight``
        -> ``base_model.model.layers.3.attn.wq_a.lora_A.weight``

    Returns ``None`` if ``name`` is not a LoRA adapter param. Raises if the base
    module does not map cleanly through the V4 converter (surfaces a naming drift
    instead of silently shipping an unusable key).
    """
    side, base = adapter_side_and_base(name)
    if side is None:
        return None

    base_weight_name = base + _WEIGHT_SUFFIX
    probe = param if param is not None else torch.empty(0)
    converted = convert_deepseekv4_to_hf(args, base_weight_name, probe)
    if len(converted) != 1:
        raise ValueError(
            f"LoRA base {base_weight_name!r} did not map to exactly one served weight "
            f"(got {[n for n, _ in converted]}); adapter cannot be routed."
        )
    serving_name = converted[0][0]
    if not serving_name.endswith(_WEIGHT_SUFFIX):
        raise ValueError(f"Unexpected served weight name {serving_name!r} for LoRA base {base_weight_name!r}")
    serving_base = serving_name[: -len(_WEIGHT_SUFFIX)]
    return f"{_PEFT_PREFIX}{serving_base}.lora_{side}.weight"


def peft_target_module_leaf(peft_name: str) -> str:
    """Leaf module name for a PEFT adapter key (the sglang ``--lora-target-modules`` token).

    ``base_model.model.layers.3.attn.wq_a.lora_A.weight`` -> ``wq_a``.
    """
    parts = peft_name.split(".")
    # drop the trailing ``lora_A``/``lora_B`` and ``weight``.
    return parts[-3]


def build_lora_adapter_state_dict(
    args: Namespace,
    named_tensors: Sequence[tuple[str, torch.Tensor]],
    *,
    scale: float,
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    """Assemble a PEFT state dict + adapter config from gathered adapter tensors.

    ``named_tensors`` are (megatron-global-name, full-tensor) pairs for the trainable
    adapter params (already TP/PP-gathered). Non-adapter names are ignored.

    ``scale`` is the LoRA forward multiplier read off the live adapters (uniform
    across V4 adapters): ``alpha / r`` classic, ``alpha / sqrt(r)`` under rsLoRA
    (``--lora-rslora``). ``r`` is read from each ``lora_A`` tensor's leading dim;
    ``lora_alpha`` is derived so ``lora_alpha / r == scale`` EXACTLY — sglang has no
    ``use_rslora``, it always computes ``scaling = lora_alpha / r`` (plain float
    division of the raw config value, ``sglang/srt/lora/lora.py``), so rsLoRA is
    served via an effective alpha ``alpha * sqrt(r)`` (e.g. 128 at r=16/alpha=32).
    The roundtrip is verified and raises if ``lora_alpha / r != scale`` (float
    exactness holds for power-of-two r; production r=16 is exact for both modes).
    """
    state_dict: dict[str, torch.Tensor] = {}
    target_modules: set[str] = set()
    ranks: set[int] = set()

    for name, tensor in named_tensors:
        peft_name = megatron_adapter_name_to_peft(args, name, tensor)
        if peft_name is None:
            continue
        state_dict[peft_name] = tensor
        target_modules.add(peft_target_module_leaf(peft_name))
        if peft_name.endswith(".lora_A.weight"):
            ranks.add(int(tensor.shape[0]))

    if not state_dict:
        raise ValueError("build_lora_adapter_state_dict: no LoRA adapter tensors found")
    if len(ranks) != 1:
        raise ValueError(f"LoRA adapters have inconsistent ranks {sorted(ranks)}; expected one")

    r = next(iter(ranks))
    # Emit lora_alpha = scale * r so the serving side recovers the trainer scaling
    # bit-for-bit (sglang: scaling = lora_alpha / r). Keep the historical integer
    # type when the value is integral (classic r16/alpha32 -> 32, rsLoRA
    # r16/alpha32 -> 128) so existing adapter_config.json output is byte-identical
    # in the default configuration; otherwise ship the float (sglang reads the raw
    # config value with no int coercion, and json serializes floats fine).
    lora_alpha = scale * r
    if float(lora_alpha).is_integer():
        lora_alpha = int(lora_alpha)
    if (lora_alpha / r) != scale:
        raise ValueError(
            f"exported lora_alpha={lora_alpha!r} does not reproduce the trainer "
            f"scaling exactly on the serving side: lora_alpha/r = {lora_alpha / r!r} "
            f"!= scale = {scale!r} (r={r}). Serving would silently apply a different "
            "adapter multiplier than training. Pick a rank whose scaling round-trips "
            "in float64 (any power-of-two r is safe for both classic and rsLoRA)."
        )
    config_dict = {
        "peft_type": "lora",
        "r": r,
        "lora_alpha": lora_alpha,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": sorted(target_modules),
    }
    return state_dict, config_dict
