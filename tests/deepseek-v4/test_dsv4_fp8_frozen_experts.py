"""D: frozen experts in FP8 (V4_FP8_FROZEN_EXPERTS=1) — cuts the ~63GB/GPU rest
state (fp8 ckpt held as bf16) so ctx-16k fits. Experts are frozen + served fp8 by
sglang at rollout, so fp8-in-training also reduces the train/rollout mismatch.
Pins: quant/dequant parity, idempotence, memory halving, sharded_state_dict guard."""

import torch
from custom_kernels.deepseek_v4.megatron.mcore_model import V4GroupedExperts


class _Cfg:
    num_local_experts = 4
    hidden_size = 256
    intermediate_size = 256
    hidden_act = "silu"
    swiglu_limit = 7.0


def _mk():
    e = V4GroupedExperts(_Cfg())
    e.gate_up_proj.data = e.gate_up_proj.data.bfloat16()  # real model: params_dtype=bf16
    e.down_proj.data = e.down_proj.data.bfloat16()
    with torch.no_grad():
        e.gate_up_proj.normal_(0, 0.02)
        e.down_proj.normal_(0, 0.02)
    e.gate_up_proj.requires_grad_(False)  # real model: LoRA freezes base experts
    e.down_proj.requires_grad_(False)
    return e


def test_dequant_parity_within_fp8_tolerance():
    e = _mk()
    ref_gu = e.gate_up_proj.clone()
    ref_dn = e.down_proj.clone()
    e.quantize_frozen_experts_fp8()
    for idx in range(_Cfg.num_local_experts):
        gu = e._expert_weight("gate_up_proj", idx)
        dn = e._expert_weight("down_proj", idx)
        # e4m3 ~2 decimal digits: relative error should be small per element
        rel_gu = (gu - ref_gu[idx]).abs().amax() / ref_gu[idx].abs().amax()
        rel_dn = (dn - ref_dn[idx]).abs().amax() / ref_dn[idx].abs().amax()
        assert rel_gu < 0.1, f"gate_up expert {idx} rel err {rel_gu}"
        assert rel_dn < 0.1, f"down expert {idx} rel err {rel_dn}"
        assert gu.dtype == torch.bfloat16 and dn.dtype == torch.bfloat16


def test_memory_halves_and_params_dropped():
    e = _mk()
    bf16_bytes = e.gate_up_proj.numel() * 2 + e.down_proj.numel() * 2
    e.quantize_frozen_experts_fp8()
    # bf16 Parameters are gone; fp8 buffers are ~half the bytes (+ tiny scales)
    assert not hasattr(e, "gate_up_proj") or e.gate_up_proj is None
    fp8_bytes = e.gate_up_proj_fp8.numel() * 1 + e.down_proj_fp8.numel() * 1
    assert fp8_bytes <= bf16_bytes // 2 + 1
    # blockwise [128,128] scales: gate_up [E, 2I/128, H/128]
    assert e.gate_up_proj_scale.shape == (
        _Cfg.num_local_experts,
        (2 * _Cfg.intermediate_size) // 128,
        _Cfg.hidden_size // 128,
    )
    # frozen experts must not appear in named_parameters (no optimizer state)
    assert not any("proj" in n for n, _ in e.named_parameters())


def test_idempotent():
    e = _mk()
    e.quantize_frozen_experts_fp8()
    s1 = e.gate_up_proj_scale.clone()
    e.quantize_frozen_experts_fp8()  # second call is a no-op
    assert torch.equal(e.gate_up_proj_scale, s1)


def test_sharded_state_dict_skips_when_quantized():
    e = _mk()
    assert e.sharded_state_dict(prefix="x.")  # real entries before quantize (load path)
    e.quantize_frozen_experts_fp8()
    assert e.sharded_state_dict(prefix="x.") == {}  # skipped after (adapter-only saves)


def test_forward_dispatched_runs_quantized():
    e = _mk()
    e.quantize_frozen_experts_fp8()
    N = 6
    hs = torch.randn(N, _Cfg.hidden_size, dtype=torch.bfloat16)
    tpe = torch.tensor([2, 1, 2, 1])
    out = e.forward_dispatched(hs, tpe)
    assert out.shape == hs.shape and torch.isfinite(out).all()


def test_frozen_fp8_matmul_forward_parity_and_grad():
    """The custom _FrozenFp8ExpertLinear must (a) match a bf16 dequant+linear in
    the forward and (b) produce a correct grad_x, WITHOUT retaining the bf16 weight
    (it saves only the fp8 buffers). Guards the ctx-16k OOM fix."""
    import torch
    from custom_kernels.deepseek_v4.megatron.mcore_model import _dequant_block_fp8, _FrozenFp8ExpertLinear

    e = _mk()
    e.quantize_frozen_experts_fp8()
    blk = e._FP8_BLOCK
    x = torch.randn(6, _Cfg.hidden_size, dtype=torch.bfloat16, requires_grad=True)
    wf = e.gate_up_proj_fp8[0]
    sc = e.gate_up_proj_scale[0]
    # custom Function
    y = _FrozenFp8ExpertLinear.apply(x, wf, sc, blk)
    y.sum().backward()
    gx = x.grad.clone()
    # reference: dequant to bf16, plain autograd linear
    x2 = x.detach().clone().requires_grad_(True)
    w_ref = _dequant_block_fp8(wf, sc, blk)
    y2 = torch.nn.functional.linear(x2, w_ref)
    y2.sum().backward()
    assert torch.allclose(y, y2, atol=1e-3), "forward mismatch vs bf16 dequant+linear"
    assert torch.allclose(gx, x2.grad, atol=1e-3), "grad_x mismatch"
    # the expert weight is fp8-stored (the Function saves fp8 buffers, not bf16)
    assert wf.dtype == torch.float8_e4m3fn


def test_expert_matmul_dispatches_on_quant_state():
    """_expert_matmul uses plain F.linear pre-quant and the custom Function post."""
    import torch

    e = _mk()
    x = torch.randn(4, _Cfg.hidden_size, dtype=torch.bfloat16)
    y_bf16 = e._expert_matmul(x, "gate_up_proj", 0)  # pre-quant path
    e.quantize_frozen_experts_fp8()
    y_fp8 = e._expert_matmul(x, "gate_up_proj", 0)  # custom-Function path
    assert y_bf16.shape == y_fp8.shape == (4, 2 * _Cfg.intermediate_size)
    # fp8-precision weights -> close but not identical
    assert (y_bf16 - y_fp8).abs().amax() / y_bf16.abs().amax() < 0.1
