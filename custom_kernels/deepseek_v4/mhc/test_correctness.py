"""Correctness tests for the tilelang mHC HyperConnection kernel.

Four checks, each printing a clear PASS/FAIL:

  1. FORMULA gradcheck (fp64): the *analytic* backward formulas (the exact math
     the kernel implements — Sinkhorn VJP, RMSNorm-collapse, param grads) run in
     fp64 and are compared to autograd through the fp64 reference. This proves
     the math is exact, independent of the kernel's fp32 tensor-core rounding.
  2. FORWARD fp32: kernel vs fp32 reference (post / comb / collapsed).
  3. BACKWARD fp32: kernel grads vs autograd through the fp32 reference.
  4. FORWARD+BACKWARD bf16: real-dtype path, looser (bf16) tolerance.

Run: CUDA_VISIBLE_DEVICES=1 python test_correctness.py
"""

import kernel as K
import reference as R
import torch

DEV = "cuda"
H, D, MIX, M = K.H, K.HIDDEN, K.MIX, K.HIDDEN * K.H
EPS = K.HC_EPS

_results = []


def _check(name, err, tol):
    ok = err <= tol
    _results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<26} max-rel-err={err:.2e}  tol={tol:.0e}")
    return ok


def _mk(B, S, dtype, rg=True, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(B, S, H, D, device=DEV, dtype=dtype, generator=g, requires_grad=rg)
    fn = (torch.randn(MIX, M, device=DEV, generator=g) / M**0.5).to(torch.float32).requires_grad_(rg)
    base = (torch.randn(MIX, device=DEV, generator=g) * 0.1).requires_grad_(rg)
    scale = (torch.rand(3, device=DEV, generator=g) + 0.5).requires_grad_(rg)
    return x, fn, base, scale


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / (b.float().abs().max() + 1e-12)).item()


# ---------------------------------------------------------------- 1. fp64 formula
def test_formula_fp64():
    print("\n[1] FORMULA gradcheck (fp64, analytic vs autograd-through-reference)")
    torch.manual_seed(0)
    B, S = 2, 5
    N = B * S
    x = torch.randn(B, S, H, D, device=DEV, dtype=torch.float64, requires_grad=True)
    fn = (torch.randn(MIX, M, device=DEV, dtype=torch.float64) / M**0.5).requires_grad_()
    base = (torch.randn(MIX, device=DEV, dtype=torch.float64) * 0.1).requires_grad_()
    scale = (torch.rand(3, device=DEV, dtype=torch.float64) + 0.5).requires_grad_()

    post, comb, coll = R.hyper_connection_forward(x, fn, base, scale, cdtype=torch.float64)
    gp, gc, gl = torch.randn_like(post), torch.randn_like(comb), torch.randn_like(coll)
    ((post * gp).sum() + (comb * gc).sum() + (coll * gl).sum()).backward()
    ax, afn, ab, asc = x.grad, fn.grad, base.grad, scale.grad

    # analytic backward in fp64 (same formulas as kernel.py, pure torch)
    xf = x.detach().reshape(N, M)
    inv = torch.rsqrt(xf.square().mean(-1) + R.RMS_NORM_EPS)
    raw = inv[:, None] * (xf @ fn.detach().t())
    d_pre = torch.einsum("nd,nhd->nh", gl.reshape(N, D), xf.view(N, H, D))
    d_raw = K._middle_bwd(raw, base.detach(), scale.detach(), d_pre, gp.reshape(N, H), gc.reshape(N, H, H))
    p = (d_raw * raw).sum(-1)
    d_flat = d_raw @ fn.detach()
    d_xf = inv[:, None] * d_flat - (inv**2 * p / M)[:, None] * xf
    d_x = d_xf.view(N, H, D) + d_pre.new_zeros(N, H, D)  # collapse term added below
    s0, s1, s2 = scale.detach().unbind(0)
    pre = torch.sigmoid(raw[:, :H] * s0 + base.detach()[:H]) + EPS
    d_x = d_xf.view(N, H, D) + pre[:, :, None] * gl.reshape(N, D)[:, None, :]
    d_fn = (d_raw * inv[:, None]).t() @ xf
    d_base = torch.cat([d_raw[:, :H].sum(0) / s0, d_raw[:, H : 2 * H].sum(0) / s1, d_raw[:, 2 * H :].sum(0) / s2])
    d_scale = torch.stack(
        [
            (d_raw[:, :H] * raw[:, :H]).sum() / s0,
            (d_raw[:, H : 2 * H] * raw[:, H : 2 * H]).sum() / s1,
            (d_raw[:, 2 * H :] * raw[:, 2 * H :]).sum() / s2,
        ]
    )
    _check("d_x", _rel(d_x.reshape_as(ax), ax), 1e-9)
    _check("d_fn", _rel(d_fn, afn), 1e-9)
    _check("d_base", _rel(d_base, ab), 1e-9)
    _check("d_scale", _rel(d_scale, asc), 1e-9)


# ---------------------------------------------------------------- 2/3. fp32
def test_fp32():
    print("\n[2/3] FORWARD + BACKWARD fp32 (kernel vs reference + autograd)")
    xr, fr, br, sr = _mk(2, 256, torch.float32, seed=1)
    xk, fk, bk, sk = _mk(2, 256, torch.float32, seed=1)
    pr, cr, lr = R.hyper_connection_forward(xr, fr, br, sr)
    pk, ck, lk = K.hyper_connection(xk, fk, bk, sk)
    _check("fwd post", _rel(pk, pr), 3e-3)
    _check("fwd comb", _rel(ck, cr), 3e-3)
    _check("fwd collapsed", _rel(lk, lr), 3e-3)
    gp, gc, gl = torch.randn_like(pr), torch.randn_like(cr), torch.randn_like(lr)
    ((pr * gp).sum() + (cr * gc).sum() + (lr * gl).sum()).backward()
    ((pk * gp).sum() + (ck * gc).sum() + (lk * gl).sum()).backward()
    _check("bwd d_x", _rel(xk.grad, xr.grad), 3e-3)
    _check("bwd d_fn", _rel(fk.grad, fr.grad), 3e-3)
    _check("bwd d_base", _rel(bk.grad, br.grad), 3e-3)
    _check("bwd d_scale", _rel(sk.grad, sr.grad), 3e-3)


# ---------------------------------------------------------------- 4. bf16
def test_bf16():
    print("\n[4] FORWARD + BACKWARD bf16 at training shape (S=2048)")
    xr, fr, br, sr = _mk(1, 2048, torch.bfloat16, seed=2)
    xk, fk, bk, sk = _mk(1, 2048, torch.bfloat16, seed=2)
    pr, cr, lr = R.hyper_connection_forward(xr, fr, br, sr)
    pk, ck, lk = K.hyper_connection(xk, fk, bk, sk)
    _check("fwd post", _rel(pk, pr), 2e-2)
    _check("fwd comb", _rel(ck, cr), 2e-2)
    _check("fwd collapsed", _rel(lk, lr), 2e-2)
    gp, gc = torch.randn_like(pr), torch.randn_like(cr)
    gl = torch.randn_like(lr.float())
    ((pr * gp).sum() + (cr * gc).sum() + (lr.float() * gl).sum()).backward()
    ((pk * gp).sum() + (ck * gc).sum() + (lk.float() * gl).sum()).backward()
    _check("bwd d_x", _rel(xk.grad, xr.grad), 3e-2)
    _check("bwd d_fn", _rel(fk.grad, fr.grad), 2e-2)


# -------------------------------------------- 5. sglang-forward + our-backward
def test_sglang_path():
    print("\n[5] sglang-fwd + our-bwd: FORWARD + BACKWARD bf16 (S=2048)")
    xr, fr, br, sr = _mk(1, 2048, torch.bfloat16, seed=2)
    xk, fk, bk, sk = _mk(1, 2048, torch.bfloat16, seed=2)
    pr, cr, lr = R.hyper_connection_forward(xr, fr, br, sr)
    pk, ck, lk = K.hyper_connection_sglang(xk, fk, bk, sk)
    # sglang forward vs reference (same math; bf16/tf32 rounding)
    _check("sgl fwd post", _rel(pk, pr), 2e-2)
    _check("sgl fwd comb", _rel(ck, cr), 2e-2)
    _check("sgl fwd collapsed", _rel(lk, lr), 2e-2)
    gp, gc = torch.randn_like(pr), torch.randn_like(cr)
    gl = torch.randn_like(lr.float())
    ((pr * gp).sum() + (cr * gc).sum() + (lr.float() * gl).sum()).backward()
    ((pk * gp).sum() + (ck * gc).sum() + (lk.float() * gl).sum()).backward()
    # backward = our analytic backward (gradient of reference math)
    _check("sgl bwd d_x", _rel(xk.grad, xr.grad), 3e-2)
    _check("sgl bwd d_fn", _rel(fk.grad, fr.grad), 2e-2)
    _check("sgl bwd d_base", _rel(bk.grad, br.grad), 3e-2)
    _check("sgl bwd d_scale", _rel(sk.grad, sr.grad), 3e-2)


if __name__ == "__main__":
    test_formula_fp64()
    test_fp32()
    test_bf16()
    test_sglang_path()
    print("\n" + ("=" * 52))
    print("OVERALL:", "ALL PASS" if all(_results) else f"{_results.count(False)} FAIL / {len(_results)}")
    print("=" * 52)
