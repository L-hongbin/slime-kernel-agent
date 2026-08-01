"""Tests for the ``--lora-rslora`` and ``--lora-plus-lambda`` arguments.

rsLoRA: trainer forward scale alpha/sqrt(r) instead of alpha/r, applied by
mutating ``adapter.scale`` right after the wrap (all attention/compressor and
shared-expert adapters read ``self.scale`` at call time). Serving
equivalence rides the exported ``lora_alpha = scale * r``: sglang always computes
``scaling = lora_alpha / r`` (no use_rslora in the fork), so the roundtrip must be
float-exact and is verified at export.

LoRA+: eta_B = lambda * eta_A via a separate Megatron optimizer param group for
the LoRA B (``linear_out``) matrices, keyed on max_lr/min_lr (the only per-group
knobs OptimizerParamScheduler honors) plus a distinct lr_mult (the param-group
identity key for optimizer-state save/resume).

CPU tests run everywhere; the param-group integration test initializes a
single-rank gloo group. Run:
    python -m pytest tests/deepseek-v4/test_dsv4_lora_rslora_loraplus.py -q
"""

from __future__ import annotations

import json
import math
import os

import pytest
import torch
from custom_kernels.deepseek_v4.megatron.lora import apply_v4_lora, audit_lora, lora_scaling
from torch import nn

# ---------------------------------------------------------------------------
# tiny V4-shaped holder (attention/compressor FQNs the default targets match)
# ---------------------------------------------------------------------------

_ATTN_LEAVES = ("q_a_proj", "q_b_proj", "kv_proj", "o_b_proj")
_COMPRESSOR_LEAVES = ("kv_proj", "gate_proj")


class _AttnHolder(nn.Module):
    def __init__(self, hidden=32, n_layers=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            layer = nn.Module()
            layer.self_attn = nn.Module()
            for leaf in _ATTN_LEAVES:
                setattr(layer.self_attn, leaf, nn.Linear(hidden, hidden, bias=False))
            layer.self_attn.compressor = nn.Module()
            for leaf in _COMPRESSOR_LEAVES:
                setattr(layer.self_attn.compressor, leaf, nn.Linear(hidden, hidden, bias=False))
            self.layers.append(layer)


def _wrapped_adapters(model):
    from megatron.bridge.peft.lora_layers import LinearAdapter

    return {name: m for name, m in model.named_modules() if isinstance(m, LinearAdapter)}


def _make_wrapped(*, rslora: bool, dim=16, alpha=32, hidden=32, seed=0):
    torch.manual_seed(seed)
    return apply_v4_lora(
        _AttnHolder(hidden=hidden),
        dim=dim,
        alpha=alpha,
        dropout=0.0,
        rslora=rslora,
    )


# ---------------------------------------------------------------------------
# rsLoRA: scale values + forward
# ---------------------------------------------------------------------------


def test_lora_scaling_helper():
    assert lora_scaling(16, 32, rslora=False) == 2.0
    assert lora_scaling(16, 32, rslora=True) == 8.0  # 32 / sqrt(16), exact
    assert lora_scaling(8, 16, rslora=False) == 2.0
    assert lora_scaling(8, 16, rslora=True) == 16 / math.sqrt(8)
    assert lora_scaling(16, 32) == 2.0


def test_default_off_scale_is_classic_alpha_over_r():
    """The default leaves the bridge's scale = alpha/dim
    untouched (the observable of default-off byte-identity — no adapter attribute
    or forward path differs from the pre-rsLoRA code)."""
    torch.manual_seed(0)
    m = apply_v4_lora(_AttnHolder(), dim=16, alpha=32, dropout=0.0)
    adapters = _wrapped_adapters(m)
    assert len(adapters) == 6
    for name, ad in adapters.items():
        assert ad.scale == 32 / 16 == 2.0, name
    assert audit_lora(m)["adapter_scales"] == [2.0]


def test_rslora_scale_exact_alpha_over_sqrt_r():
    m = _make_wrapped(rslora=True)
    adapters = _wrapped_adapters(m)
    assert len(adapters) == 6
    for name, ad in adapters.items():
        assert ad.scale == 8.0, name  # 32/sqrt(16), exact float
        assert ad.dim == 16 and ad.alpha == 32
    assert audit_lora(m)["adapter_scales"] == [8.0]


def test_forward_identity_between_modes_at_init():
    """B is zero-init, so the wrapped forward equals the base forward in BOTH
    modes at init — enabling rsLoRA does not perturb step-0 parity."""
    x = torch.randn(3, 5, 32)
    m_off = _make_wrapped(rslora=False, seed=7)
    m_on = _make_wrapped(rslora=True, seed=7)
    with torch.no_grad():
        assert torch.equal(m_off.layers[0].self_attn.q_a_proj(x), m_on.layers[0].self_attn.q_a_proj(x))


def test_rslora_forward_delta_is_exactly_scaled():
    """With identical weights (same seed), the adapter delta under rsLoRA is
    bitwise sqrt(r) x the classic delta (multiplying by a power of two only
    shifts the exponent), and the full forward equals base + scale * B(A(x))."""
    m_off = _make_wrapped(rslora=False, seed=3)
    m_on = _make_wrapped(rslora=True, seed=3)
    x = torch.randn(4, 32)
    ad_off = m_off.layers[0].self_attn.q_a_proj
    ad_on = m_on.layers[0].self_attn.q_a_proj
    with torch.no_grad():
        ad_off.linear_out.weight.normal_(0, 0.05)
        ad_on.linear_out.weight.copy_(ad_off.linear_out.weight)
        # zero the frozen base so the forward IS the scaled adapter delta
        # (0 + lora_res is exact; reconstructing via (base+lora)-base is not).
        ad_off.weight.zero_()
        ad_on.weight.zero_()
        delta_off = ad_off(x)
        delta_on = ad_on(x)
        lora_core = ad_off.linear_out(ad_off.linear_in(x))
    assert torch.equal(delta_off, lora_core * 2.0)
    assert torch.equal(delta_on, lora_core * 8.0)
    assert torch.equal(delta_on, delta_off * 4.0)  # 4 = sqrt(16) ratio, exact


# ---------------------------------------------------------------------------
# rsLoRA: export / serving roundtrip (the alpha_eff trick)
# ---------------------------------------------------------------------------


def _fake_adapter_tensors(rank: int, hidden: int = 32):
    base = "module.module.decoder.layers.3.self_attn.q_a_proj"
    return [
        (base + ".linear_in.weight", torch.randn(rank, hidden)),
        (base + ".linear_out.weight", torch.randn(hidden, rank)),
    ]


def _sglang_scaling(config_dict):
    """Reproduce EXACTLY what the sglang fork does with the exported config:
    LoRAConfig reads the raw values (lora_config.py:43-44, no coercion) and
    LoRAAdapter computes ``scaling = lora_alpha / r`` (lora.py:71)."""
    return config_dict["lora_alpha"] / config_dict["r"]


def test_export_classic_scaling_unchanged():
    from slime.backends.megatron_utils.update_weight.lora_adapter_sync import build_lora_adapter_state_dict

    scale = lora_scaling(16, 32, rslora=False)
    _sd, cfg = build_lora_adapter_state_dict(None, _fake_adapter_tensors(16), scale=scale)
    assert cfg["r"] == 16
    assert cfg["lora_alpha"] == 32 and isinstance(cfg["lora_alpha"], int)  # byte-identical default config
    assert _sglang_scaling(cfg) == scale == 2.0


def test_export_rslora_alpha_eff_roundtrip_exact():
    from slime.backends.megatron_utils.update_weight.lora_adapter_sync import build_lora_adapter_state_dict

    scale = lora_scaling(16, 32, rslora=True)  # 8.0 exact
    _sd, cfg = build_lora_adapter_state_dict(None, _fake_adapter_tensors(16), scale=scale)
    assert cfg["lora_alpha"] == 128 and isinstance(cfg["lora_alpha"], int)  # 32 * sqrt(16)
    # serving reproduces the trainer scale bit-for-bit
    assert _sglang_scaling(cfg) == scale == 8.0
    # and the config survives a JSON write/parse (the on-disk exporter path)
    assert _sglang_scaling(json.loads(json.dumps(cfg))) == scale


def test_export_non_integral_alpha_ships_float_when_exact():
    from slime.backends.megatron_utils.update_weight.lora_adapter_sync import build_lora_adapter_state_dict

    # scale * r = 1.5 (non-integral) but 1.5 / 2 == 0.75 exactly -> float alpha OK
    _sd, cfg = build_lora_adapter_state_dict(None, _fake_adapter_tensors(2), scale=0.75)
    assert cfg["lora_alpha"] == 1.5 and isinstance(cfg["lora_alpha"], float)
    assert _sglang_scaling(cfg) == 0.75


def test_export_inexact_roundtrip_raises():
    from slime.backends.megatron_utils.update_weight.lora_adapter_sync import build_lora_adapter_state_dict

    # 0.1 * 3 = 0.30000000000000004; /3 = 0.10000000000000002 != 0.1 -> must raise,
    # not silently serve a different multiplier than training.
    scale = 0.1
    assert (scale * 3) / 3 != scale  # premise of the test
    with pytest.raises(ValueError, match="does not reproduce the trainer scaling"):
        build_lora_adapter_state_dict(None, _fake_adapter_tensors(3), scale=scale)


def test_live_module_scale_feeds_export_chain():
    """The sync reads ``float(module.scale)`` off the LIVE adapters
    (_build_v4_lora_base_scales) — mutating scale at apply time is sufficient for
    both the merge path and the adapter-only export."""
    m = _make_wrapped(rslora=True)
    scales = {float(ad.scale) for ad in _wrapped_adapters(m).values()}
    assert scales == {8.0}


# ---------------------------------------------------------------------------
# rsLoRA: resume scaling marker
# ---------------------------------------------------------------------------


def test_scaling_marker_roundtrip_and_mismatch(tmp_path):
    from slime.backends.megatron_utils.adapter_ckpt import (
        collect_adapter_scaling,
        validate_adapter_scaling_marker,
        write_adapter_scaling_marker,
    )

    m_rs = _make_wrapped(rslora=True)
    meta = collect_adapter_scaling([m_rs], rslora=True)
    assert meta == {"dim": 16, "alpha": 32.0, "scale": 8.0, "rslora": True}

    save_dir = tmp_path / "ckpt"
    save_dir.mkdir()
    (save_dir / "latest_checkpointed_iteration.txt").write_text("7")
    write_adapter_scaling_marker(str(save_dir), 7, [m_rs], rslora=True)
    marker = save_dir / "iter_0000007" / "v4_lora_scaling.json"
    assert marker.is_file()

    # same-scaling resume validates (via save dir AND direct iter dir)
    validate_adapter_scaling_marker(str(save_dir), [m_rs], rslora=True)
    validate_adapter_scaling_marker(str(save_dir / "iter_0000007"), [m_rs], rslora=True)

    # resuming the rsLoRA checkpoint with classic scaling must fail loud
    m_classic = _make_wrapped(rslora=False)
    with pytest.raises(RuntimeError, match="scaling mismatch"):
        validate_adapter_scaling_marker(str(save_dir), [m_classic])


def test_scaling_marker_missing_only_warns(tmp_path, caplog):
    from slime.backends.megatron_utils.adapter_ckpt import validate_adapter_scaling_marker

    m = _make_wrapped(rslora=False)
    save_dir = tmp_path / "old_ckpt"
    save_dir.mkdir()
    (save_dir / "latest_checkpointed_iteration.txt").write_text("3")
    (save_dir / "iter_0000003").mkdir()
    with caplog.at_level("WARNING"):
        validate_adapter_scaling_marker(str(save_dir), [m])  # no marker -> warn, no raise
    assert any("cannot verify" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# LoRA+: env parsing
# ---------------------------------------------------------------------------


def test_lora_plus_lambda_parsing():
    from argparse import Namespace

    from slime.utils.lora_utils import lora_plus_lambda

    for off in (None, 1, 1.0):
        assert lora_plus_lambda(Namespace(lora_plus_lambda=off)) is None, repr(off)
    assert lora_plus_lambda(Namespace(lora_plus_lambda=4)) == 4.0
    assert lora_plus_lambda(Namespace(lora_plus_lambda=8.5)) == 8.5
    for bad in (0, -2, "abc"):
        with pytest.raises(ValueError):
            lora_plus_lambda(Namespace(lora_plus_lambda=bad))


# ---------------------------------------------------------------------------
# LoRA+: override construction + param-group split
# ---------------------------------------------------------------------------


def _optimizer_config(lr=1e-4, min_lr=0.0):
    from megatron.core.optimizer import OptimizerConfig

    return OptimizerConfig(lr=lr, min_lr=min_lr, optimizer="adam", bf16=False, fp16=False)


def test_lora_plus_overrides_shape_and_no_lora_raises(monkeypatch):
    from slime.backends.megatron_utils.model import _v4_lora_plus_config_overrides

    config = _optimizer_config()
    m = _make_wrapped(rslora=False)
    overrides = _v4_lora_plus_config_overrides(config, [m], 4.0)
    b_keys = [k for k in overrides if getattr(k, "name", None) == "*.linear_out.weight"]
    assert len(b_keys) == 1
    ov = overrides[b_keys[0]]
    assert ov["max_lr"] == 4.0 * config.lr
    assert ov["min_lr"] == 0.0
    assert ov["lr_mult"] == 4.0  # distinct param-group identity for save/resume

    plain = _AttnHolder()  # no LoRA wrap -> lambda set is a config error
    with pytest.raises(RuntimeError, match="no trainable LoRA"):
        _v4_lora_plus_config_overrides(config, [plain], 4.0)


def _ensure_single_rank_dist():
    if torch.distributed.is_initialized():
        return True
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29511")
    try:
        torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
        return True
    except Exception:  # noqa: BLE001
        return False


def test_lora_plus_param_groups_contain_exactly_b_tensors(monkeypatch):
    """End-to-end through megatron's _get_param_groups (what both the Muon and the
    adam constructions call): exactly the linear_out tensors land in the lambda
    group, everything else trainable in the default group, no frozen base params
    anywhere."""
    if not _ensure_single_rank_dist():
        pytest.skip("could not initialize a single-rank gloo process group")
    from megatron.core.optimizer import _get_param_groups

    from slime.backends.megatron_utils.model import _v4_lora_plus_config_overrides

    lam = 4.0
    config = _optimizer_config(lr=1e-4, min_lr=1e-6)
    m = _make_wrapped(rslora=False)
    overrides = _v4_lora_plus_config_overrides(config, [m], lam)
    groups = _get_param_groups([m], config, overrides)

    b_params = {id(p) for n, p in m.named_parameters() if n.endswith(".linear_out.weight")}
    a_params = {id(p) for n, p in m.named_parameters() if n.endswith(".linear_in.weight")}
    assert len(b_params) == len(a_params) == 6

    lam_groups = [g for g in groups if g.get("lr_mult") == lam and g["params"]]
    default_groups = [g for g in groups if g.get("lr_mult") == 1.0 and g["params"]]
    assert len(lam_groups) == 1
    assert {id(p) for p in lam_groups[0]["params"]} == b_params
    assert lam_groups[0]["max_lr"] == lam * config.lr
    assert lam_groups[0]["min_lr"] == lam * config.min_lr
    assert {id(p) for g in default_groups for p in g["params"]} == a_params
    for g in default_groups:
        assert g["max_lr"] == config.lr
    # nothing else (frozen base) snuck into any group
    all_group_params = {id(p) for g in groups for p in g["params"]}
    assert all_group_params == a_params | b_params


def test_lora_plus_scheduler_applies_group_max_lr(monkeypatch):
    """OptimizerParamScheduler.get_lr honors per-group max_lr/min_lr (and ignores
    lr_mult): under the production 'constant' style the B group runs at exactly
    lambda * lr from step 0."""
    if not _ensure_single_rank_dist():
        pytest.skip("could not initialize a single-rank gloo process group")
    from types import SimpleNamespace

    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler

    lam, lr = 4.0, 1e-4
    fake_opt = SimpleNamespace(
        param_groups=[
            {
                "params": [torch.nn.Parameter(torch.zeros(1))],
                "lr_mult": 1.0,
                "wd_mult": 1.0,
                "max_lr": lr,
                "min_lr": 0.0,
            },
            {
                "params": [torch.nn.Parameter(torch.zeros(1))],
                "lr_mult": lam,
                "wd_mult": 1.0,
                "max_lr": lam * lr,
                "min_lr": 0.0,
            },
        ]
    )
    OptimizerParamScheduler(
        fake_opt,
        init_lr=0.0,
        max_lr=lr,
        min_lr=0.0,
        lr_warmup_steps=0,
        lr_decay_steps=100,
        lr_decay_style="constant",
        start_wd=0.01,
        end_wd=0.01,
        wd_incr_steps=100,
        wd_incr_style="constant",
        use_checkpoint_opt_param_scheduler=False,
        override_opt_param_scheduler=True,
    )
    eta_a = fake_opt.param_groups[0]["lr"]
    eta_b = fake_opt.param_groups[1]["lr"]
    assert eta_a == lr
    assert eta_b == lam * lr == lam * eta_a


def test_log_v4_lora_plus_groups_verifies_split(monkeypatch):
    from types import SimpleNamespace

    from slime.backends.megatron_utils.model import _log_v4_lora_plus_groups

    lam, lr = 4.0, 1e-4
    m = _make_wrapped(rslora=False)
    a = [p for n, p in m.named_parameters() if n.endswith(".linear_in.weight")]
    b = [p for n, p in m.named_parameters() if n.endswith(".linear_out.weight")]
    good = SimpleNamespace(
        param_groups=[
            {"params": a, "lr_mult": 1.0, "lr": lr},
            {"params": b, "lr_mult": lam, "lr": lam * lr},
        ]
    )
    _log_v4_lora_plus_groups(good, [m], lam)  # passes

    # wrong contents (an A tensor in the B group) must fail loud
    bad = SimpleNamespace(
        param_groups=[
            {"params": a[1:], "lr_mult": 1.0, "lr": lr},
            {"params": b + a[:1], "lr_mult": lam, "lr": lam * lr},
        ]
    )
    with pytest.raises(RuntimeError, match="does not contain exactly"):
        _log_v4_lora_plus_groups(bad, [m], lam)

    # wrong LR ratio must fail loud
    bad_lr = SimpleNamespace(
        param_groups=[
            {"params": a, "lr_mult": 1.0, "lr": lr},
            {"params": b, "lr_mult": lam, "lr": 2 * lr},
        ]
    )
    with pytest.raises(RuntimeError, match="effective LRs wrong"):
        _log_v4_lora_plus_groups(bad_lr, [m], lam)
