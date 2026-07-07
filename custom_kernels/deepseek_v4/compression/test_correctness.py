"""Correctness tests for the V4-Flash compression pool tilelang kernel (B2).

Checks, for BOTH HCA (W=128) and CSA (overlap, ratio=4), head_dim=512, several n_win:
  * forward vs fp32 reference  (tight fp32; bf16 within tolerance at real shapes)
  * backward vs torch autograd through the fp32 reference (dkv, dgate, dweight, dpos_bias)

Run:  CUDA_VISIBLE_DEVICES=2 python test_correctness.py
"""

import reference as ref
import torch
from kernel import csa_compress, hca_compress

DEV = "cuda"
D = 512
EPS = 1e-6

# fp32: tight. bf16: loose (compute is fp32 but inputs/outputs round to bf16).
TOL = {torch.float32: dict(atol=1e-4, rtol=1e-4), torch.bfloat16: dict(atol=3e-2, rtol=3e-2)}


def _stats(a, b):
    a = a.float()
    b = b.float()
    denom = b.abs().max().clamp_min(1e-6)
    return (a - b).abs().max().item(), ((a - b).abs().max() / denom).item()


def _check(name, got, exp, dt):
    mae, mre = _stats(got, exp)
    tol = TOL[dt]
    ok = torch.allclose(got.float(), exp.float(), **tol)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:18s} max_abs={mae:.3e} max_rel={mre:.3e}")
    return ok


def run_case(kind, dt, B, n_win, seed=0):
    torch.manual_seed(seed)
    rate = 128 if kind == "HCA" else 4
    S = n_win * rate + (rate // 2)  # add a partial trailing window to exercise truncation
    proj = D if kind == "HCA" else 2 * D
    kv = torch.randn(B, S, proj, device=DEV, dtype=dt) * 0.5
    gate = torch.randn(B, S, proj, device=DEV, dtype=dt) * 0.5
    pb = torch.randn(rate, proj, device=DEV, dtype=dt) * 0.3
    w = 1.0 + 0.1 * torch.randn(D, device=DEV, dtype=dt)

    tensors = [kv, gate, pb, w]
    k_k, g_k, pb_k, w_k = [t.detach().clone().requires_grad_(True) for t in tensors]
    k_r, g_r, pb_r, w_r = [t.detach().float().clone().requires_grad_(True) for t in tensors]

    fn_k = hca_compress if kind == "HCA" else csa_compress
    fn_r = ref.hca_compress_ref if kind == "HCA" else ref.csa_compress_ref
    out_k = fn_k(k_k, g_k, pb_k, w_k, EPS, rate)
    out_r = fn_r(k_r, g_r, pb_r, w_r, EPS, rate)

    print(f"{kind} {dt} B={B} n_win={n_win} S={S} -> compressed {tuple(out_k.shape)}")
    ok = _check("forward", out_k, out_r, dt)

    go = torch.randn_like(out_k)
    out_k.backward(go)
    out_r.backward(go.float())
    ok &= _check("dkv", k_k.grad, k_r.grad, dt)
    ok &= _check("dgate", g_k.grad, g_r.grad, dt)
    ok &= _check("dpos_bias", pb_k.grad, pb_r.grad, dt)
    ok &= _check("dweight", w_k.grad, w_r.grad, dt)
    return ok


def main():
    allok = True
    for kind in ("HCA", "CSA"):
        for dt in (torch.float32, torch.bfloat16):
            for B, n_win in [(1, 3), (2, 5), (1, 16)]:
                allok &= run_case(kind, dt, B, n_win)
            print()
    print("=" * 60)
    print("ALL PASS" if allok else "SOME FAILED")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
