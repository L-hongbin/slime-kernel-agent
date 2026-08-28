"""Tests for DS-V4 shared-expert LoRA: frozen fp8 base +
bf16 adapter on ``mlp.shared_experts.{gate,up,down}_proj``.

Covers the four layers of the feature:
  1. train-side target gating + wrap audit (flag off => bit-identical old set),
  2. the fp8/adapter interaction in ``V4SharedExpertMLP`` (the `_lin` bypass
     hazard: quantizing the base must NOT drop the LoRA delta),
  3. adapter export name mapping (megatron -> native serving names w1/w3/w2),
  4. the sglang fork contracts (w1/w3/w2 normalization, tp1-replicated buffer
     sizing + slicing, FusedMoE exclusion, waterfill+LoRA hard raise).

CPU tests run everywhere; fp8 tests need CUDA + deep_gemm; sglang tests need
the serving fork importable (the cluster containers). Run:
    python -m pytest tests/deepseek-v4/test_dsv4_lora_shared_expert.py -q
"""

from __future__ import annotations

import sys

import pytest
import torch
from custom_kernels.deepseek_v4.megatron.lora import (
    DSV4_LORA_SHARED_EXPERT_TARGET_MODULES,
    DSV4_LORA_TARGET_MODULES,
    apply_v4_lora,
    audit_lora,
    default_lora_target_modules,
)
from custom_kernels.deepseek_v4.megatron.m0_smoke import tiny_config
from custom_kernels.deepseek_v4.megatron.mcore_model import V4SharedExpertMLP
from torch import nn

_SHARED_LEAVES = ("gate_proj", "up_proj", "down_proj")


# --------------------------------------------------------------- target gating


def test_default_targets_unchanged_when_flag_off():
    """With the flag off the effective target set is BIT-IDENTICAL to the legacy
    attention/compressor set — existing runs are unaffected."""
    assert default_lora_target_modules() == DSV4_LORA_TARGET_MODULES


def test_flag_adds_shared_expert_targets():
    targets = default_lora_target_modules(shared_expert=True)
    assert targets[: len(DSV4_LORA_TARGET_MODULES)] == DSV4_LORA_TARGET_MODULES
    assert targets[len(DSV4_LORA_TARGET_MODULES) :] == DSV4_LORA_SHARED_EXPERT_TARGET_MODULES
    # full-path wildcards scoped to decoder layers only (the leaf-collision
    # lesson + the MTP exclusion: mtp.transformer_layer.mlp.shared_experts must
    # NOT match — its adapters have no exportable serving name).
    for pat in DSV4_LORA_SHARED_EXPERT_TARGET_MODULES:
        assert pat.startswith("*layers.*.mlp.shared_experts."), pat


# ---------------------------------------------------------------- wrap audit


class _TinyHolder(nn.Module):
    """Minimal module tree exposing the REAL FQN shape ``layers.<i>.mlp.
    shared_experts.<leaf>`` plus decoys that must NOT be wrapped (router
    ``gate``, a routed-expert-like linear, an indexer-like gate_proj, and the
    MTP head's shared expert — its adapters would have no exportable serving
    name and the NextN serving tree is never LoRA-wrapped)."""

    def __init__(self, cfg, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.Module()
            layer.mlp = nn.Module()
            layer.mlp.shared_experts = V4SharedExpertMLP(cfg)
            layer.mlp.gate = nn.Linear(cfg.hidden_size, cfg.n_routed_experts, bias=False)  # router decoy
            layer.mlp.experts = nn.Module()
            layer.mlp.experts.gate_proj = nn.Linear(8, 8, bias=False)  # expert decoy
            layer.indexer = nn.Module()
            layer.indexer.gate_proj = nn.Linear(8, 8, bias=False)  # indexer decoy
            self.layers.append(layer)
        # MTP decoy: mirrors mcore's ``mtp.transformer_layer.mlp.shared_experts``
        self.mtp = nn.Module()
        self.mtp.transformer_layer = nn.Module()
        self.mtp.transformer_layer.mlp = nn.Module()
        self.mtp.transformer_layer.mlp.shared_experts = V4SharedExpertMLP(cfg)


def _wrapped_names(model):
    return set(audit_lora(model)["wrapped"])


def test_wrap_audit_exact_shared_expert_set():
    cfg = tiny_config()
    m = apply_v4_lora(_TinyHolder(cfg), dim=4, alpha=8, dropout=0.0, shared_expert=True)
    expected = {f"layers.{i}.mlp.shared_experts.{leaf}" for i in range(2) for leaf in _SHARED_LEAVES}
    assert _wrapped_names(m) == expected
    aud = audit_lora(m)
    assert aud["trainable_are_lora_only"]
    # r=4: each wrapped linear contributes A(4*in) + B(out*4).
    h, inter = cfg.hidden_size, cfg.intermediate_size
    per_layer = 2 * (4 * h + inter * 4) + (4 * inter + h * 4)
    assert aud["n_trainable"] == 2 * per_layer


def test_wrap_off_by_default_leaves_shared_expert_frozen():
    cfg = tiny_config()
    m = apply_v4_lora(_TinyHolder(cfg), dim=4, alpha=8, dropout=0.0)
    assert _wrapped_names(m) == set()  # holder has no attention/compressor
    assert all(not p.requires_grad for p in m.parameters())


# ------------------------------------------------- init identity + grad flow


def test_wrapped_shared_expert_is_identity_at_init():
    """LinearAdapter zero-inits B, so the wrapped forward equals the unwrapped
    forward exactly at init (forward parity preserved on wrap)."""
    torch.manual_seed(0)
    cfg = tiny_config()
    holder = _TinyHolder(cfg, n_layers=1)
    mlp = holder.layers[0].mlp.shared_experts
    x = torch.randn(2, 5, cfg.hidden_size)
    with torch.no_grad():
        ref = mlp(x)
    apply_v4_lora(holder, dim=4, alpha=8, dropout=0.0, shared_expert=True)
    with torch.no_grad():
        got = holder.layers[0].mlp.shared_experts(x)
    assert torch.equal(got, ref)


def test_grads_flow_to_adapter_only():
    torch.manual_seed(1)
    cfg = tiny_config()
    holder = apply_v4_lora(
        _TinyHolder(cfg, n_layers=1),
        dim=4,
        alpha=8,
        dropout=0.0,
        shared_expert=True,
    )
    mlp = holder.layers[0].mlp.shared_experts
    # B is zero-init => dL/dA would be zero; make B nonzero so both sides get grads.
    with torch.no_grad():
        for leaf in _SHARED_LEAVES:
            getattr(mlp, leaf).linear_out.weight.normal_(0, 0.02)
    x = torch.randn(2, 5, cfg.hidden_size)
    mlp(x).sum().backward()
    for leaf in _SHARED_LEAVES:
        ad = getattr(mlp, leaf)
        assert ad.linear_in.weight.grad is not None and ad.linear_in.weight.grad.abs().sum() > 0, leaf
        assert ad.linear_out.weight.grad is not None and ad.linear_out.weight.grad.abs().sum() > 0, leaf
        assert not ad.weight.requires_grad and ad.weight.grad is None, leaf


# --------------------------------------------------------- export name mapping


_MEG_BASE = "module.module.decoder.layers.3.mlp.shared_experts"
_EXPORT_CASES = [
    (f"{_MEG_BASE}.gate_proj", "layers.3.ffn.shared_experts.w1", "w1"),
    (f"{_MEG_BASE}.up_proj", "layers.3.ffn.shared_experts.w3", "w3"),
    (f"{_MEG_BASE}.down_proj", "layers.3.ffn.shared_experts.w2", "w2"),
]


@pytest.mark.parametrize("base,serving_base,leaf", _EXPORT_CASES)
def test_megatron_adapter_name_to_peft_shared_expert(base, serving_base, leaf):
    from slime.backends.megatron_utils.update_weight.lora_adapter_sync import (
        megatron_adapter_name_to_peft,
        peft_target_module_leaf,
    )

    for side, suffix in (("A", ".linear_in.weight"), ("B", ".linear_out.weight")):
        peft = megatron_adapter_name_to_peft(None, base + suffix, torch.empty(0))
        assert peft == f"base_model.model.{serving_base}.lora_{side}.weight"
        assert peft_target_module_leaf(peft) == leaf


def test_build_adapter_state_dict_includes_shared_expert_targets():
    from slime.backends.megatron_utils.update_weight.lora_adapter_sync import build_lora_adapter_state_dict

    r, h, inter = 4, 64, 32
    named = [
        (f"{_MEG_BASE}.gate_proj.linear_in.weight", torch.zeros(r, h)),
        (f"{_MEG_BASE}.gate_proj.linear_out.weight", torch.zeros(inter, r)),
        (f"{_MEG_BASE}.up_proj.linear_in.weight", torch.zeros(r, h)),
        (f"{_MEG_BASE}.up_proj.linear_out.weight", torch.zeros(inter, r)),
        (f"{_MEG_BASE}.down_proj.linear_in.weight", torch.zeros(r, inter)),
        (f"{_MEG_BASE}.down_proj.linear_out.weight", torch.zeros(h, r)),
        ("module.module.decoder.layers.3.self_attn.q_a_proj.linear_in.weight", torch.zeros(r, h)),
        ("module.module.decoder.layers.3.self_attn.q_a_proj.linear_out.weight", torch.zeros(h, r)),
    ]
    sd, cfg = build_lora_adapter_state_dict(None, named, scale=2.0)
    assert {"w1", "w2", "w3"} <= set(cfg["target_modules"])
    assert "base_model.model.layers.3.ffn.shared_experts.w1.lora_A.weight" in sd
    assert cfg["r"] == r and cfg["lora_alpha"] == 8


# ----------------------------------------------------------- sglang fork side


def test_sglang_fork_shared_expert_contracts():
    """Run the serving-fork contract checks in a CLEAN interpreter. In-process
    imports flake: the train-side kernel imports load tilelang's
    libcudart_stub.so, which shadows real cudart symbols sglang later resolves
    via ctypes (undefined symbol: cudaDeviceReset). A fresh process imports
    sglang before tilelang ever loads. See _sglang_shared_expert_checks.py for
    the individual contracts (normalization, tp1 sizing/slicing, FusedMoE
    exclusion, waterfill raise)."""
    import importlib.util
    import subprocess
    from pathlib import Path

    if importlib.util.find_spec("sglang") is None:
        pytest.skip("serving fork not importable here")
    script = Path(__file__).parent / "_sglang_shared_expert_checks.py"
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "ALL SGLANG SHARED-EXPERT CONTRACTS PASS" in proc.stdout


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
