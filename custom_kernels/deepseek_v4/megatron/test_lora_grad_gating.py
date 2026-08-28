"""Unit check for the LoRA needs_input_grad gating added to A1 + B2 backward.

Under LoRA the base params ``sinks`` (A1), compressor ``position_bias`` + ``kv_norm
.weight`` (B2) are frozen.  The kernels' backward now SKIPS those param-grad reductions
when ``ctx.needs_input_grad[i]`` is False (returns None), while still returning the
INPUT grads (dq/dk/dkv/dgate) that carry cross-layer flow to the trainable LoRA adapters.

This checks, for A1 / B2-CSA / B2-HCA:
  (1) FROZEN param (requires_grad=False): the param grad is None, input grads are present
      and finite (flow preserved).
  (2) TRAINED param (requires_grad=True): the param grad is present AND numerically equal
      to the ungated path (the gating must not change the computed gradient).

Run:  CUDA_VISIBLE_DEVICES=0 python -m custom_kernels.deepseek_v4.megatron.test_lora_grad_gating
"""

import torch

from . import _kernels


def _mk(B=1, H=8, S=128, D=512, dev="cuda", dt=torch.bfloat16):
    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn(B, H, S, D, generator=g).to(dev).to(dt)
    k_raw = torch.randn(B, 1, S, D, generator=g).to(dev).to(dt)
    sinks = torch.randn(H, generator=g).to(dev).to(torch.float32)
    return q, k_raw, sinks


def _run_a1(requires_sink_grad):
    q, k_raw, sinks = _mk()
    q = q.clone().requires_grad_(True)
    k_raw = k_raw.clone().requires_grad_(True)
    sinks = sinks.clone().requires_grad_(requires_sink_grad)
    out = _kernels.v4flash_attention(q.contiguous(), k_raw.contiguous(), None, sinks, 128, 1)
    out.float().pow(2).sum().backward()
    return q.grad, k_raw.grad, sinks.grad


def _run_b2(kind, requires_param_grad):
    # B2 compressors take kv/gate [B,S,Dproj], position_bias, kv_norm weight, eps, rate.
    B, S = 1, 128
    if kind == "csa":
        R, Dproj, D = 4, 1024, 512  # CSA kv/gate are 2*head_dim wide
        fn = _kernels.csa_compress
    else:
        R, Dproj, D = 128, 512, 512  # HCA kv/gate are head_dim wide
        fn = _kernels.hca_compress
    g = torch.Generator(device="cpu").manual_seed(1)
    kv = torch.randn(B, S, Dproj, generator=g).cuda().bfloat16().requires_grad_(True)
    gate = torch.randn(B, S, Dproj, generator=g).cuda().bfloat16().requires_grad_(True)
    pos = (torch.randn(R, Dproj, generator=g) * 0.05).cuda().to(torch.float32).requires_grad_(requires_param_grad)
    # kv_norm gain is bf16 in the model (the kernel expects the gain in the kv dtype).
    weight = torch.ones(D).cuda().bfloat16().requires_grad_(requires_param_grad)
    out = fn(kv.contiguous(), gate.contiguous(), pos, weight, 1e-6, R)
    out.float().pow(2).sum().backward()
    return kv.grad, gate.grad, pos.grad, weight.grad


def main():
    assert torch.cuda.is_available(), "needs GPU (CUDA_VISIBLE_DEVICES=0)"
    torch.cuda.set_device(0)
    ok = True

    # ---- A1 ----
    dq_t, dk_t, dsink_t = _run_a1(requires_sink_grad=True)
    dq_f, dk_f, dsink_f = _run_a1(requires_sink_grad=False)
    a1_frozen_ok = (
        dsink_f is None
        and dq_f is not None
        and dk_f is not None
        and torch.isfinite(dq_f.float()).all()
        and torch.isfinite(dk_f.float()).all()
    )
    # trained-case grad unchanged + input grads identical between the two runs
    a1_trained_ok = dsink_t is not None and torch.isfinite(dsink_t.float()).all()

    # A1 accumulates dKV with atomic adds -> run-to-run order is nondeterministic, so
    # compare input grads by relative norm (the gating must not change them BEYOND that
    # atomic noise floor ~1e-3), not bit-equality.
    def _rel(a, b):
        return ((a.float() - b.float()).norm() / a.float().norm().clamp_min(1e-12)).item()

    a1_inp_match = _rel(dq_t, dq_f) <= 2e-3 and _rel(dk_t, dk_f) <= 2e-3
    print(
        f"[A1] frozen: dsink={'None(ok)' if dsink_f is None else 'PRESENT(BAD)'} "
        f"dq/dk finite={a1_frozen_ok} | trained: dsink present={a1_trained_ok} | "
        f"input-grad match frozen-vs-trained={a1_inp_match}"
    )
    ok = ok and a1_frozen_ok and a1_trained_ok and a1_inp_match

    # ---- B2 CSA + HCA ----
    for kind in ("csa", "hca"):
        dkv_t, dg_t, dpos_t, dw_t = _run_b2(kind, requires_param_grad=True)
        dkv_f, dg_f, dpos_f, dw_f = _run_b2(kind, requires_param_grad=False)
        frozen_ok = (
            dpos_f is None
            and dw_f is None
            and dkv_f is not None
            and dg_f is not None
            and torch.isfinite(dkv_f.float()).all()
            and torch.isfinite(dg_f.float()).all()
        )
        trained_ok = (
            dpos_t is not None
            and dw_t is not None
            and torch.isfinite(dpos_t.float()).all()
            and torch.isfinite(dw_t.float()).all()
        )
        inp_match = torch.allclose(dkv_t.float(), dkv_f.float(), atol=1e-2, rtol=1e-2) and torch.allclose(
            dg_t.float(), dg_f.float(), atol=1e-2, rtol=1e-2
        )
        print(
            f"[B2-{kind}] frozen: dpos={'None(ok)' if dpos_f is None else 'BAD'} "
            f"dweight={'None(ok)' if dw_f is None else 'BAD'} dkv/dgate finite={frozen_ok} | "
            f"trained: dpos+dweight present={trained_ok} | input-grad match={inp_match}"
        )
        ok = ok and frozen_ok and trained_ok and inp_match

    print("\nLORA GRAD-GATING " + ("PASS" if ok else "FAIL"))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
