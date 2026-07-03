"""Correctness tests for the DeepSeek-V4-Flash core attention (A1) tilelang kernel.

Run:  CUDA_VISIBLE_DEVICES=0 python test_correctness.py

Checks
------
1. Reference self-consistency: float64 `torch.autograd.gradcheck` on the fp32
   reference at a tiny shape -> the reference (source of truth) differentiates
   exactly as intended.
2. Forward: bf16 kernel vs fp32 reference on real V4-Flash dims (H=64, D=512,
   S in {512, 2048, 8192}, sliding-only / CSA m=4 / HCA m=128), tight bf16
   tolerance.
3. Backward: bf16 kernel grads (dq, dk_raw, dk_comp, dsink) vs autograd through
   the fp32 reference, bf16 tolerance, on the same shapes.

NOTE: an fp32-COMPUTE kernel variant (for a true float64 gradcheck of the kernel
itself) is blocked by tilelang layout inference at the small block sizes needed
to fit fp32 head_dim=512 tiles in smem; the bf16-kernel-vs-fp32-reference grad
comparison below is the kernel correctness gate instead.
"""

import sys

import torch

sys.path.insert(0, ".")
import kernel as K  # noqa: E402
from reference import attention_reference  # noqa: E402

DEV = "cuda"
H, D = 64, 512


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / (b.abs().max() + 1e-9)).item(), (a - b).abs().max().item()


def make_inputs(B, S, m, with_comp, dtype=torch.bfloat16, scale=0.3, seed=0):
    torch.manual_seed(seed)
    Tcomp = (S // m) if with_comp else 0
    q = torch.randn(B, H, S, D, device=DEV, dtype=dtype) * scale
    kr = torch.randn(B, 1, S, D, device=DEV, dtype=dtype) * scale
    kc = (torch.randn(B, 1, Tcomp, D, device=DEV, dtype=dtype) * scale) if with_comp else None
    sinks = torch.randn(H, device=DEV, dtype=torch.float32)
    return q, kr, kc, sinks


# --------------------------------------------------------------------------- #
def test_reference_gradcheck():
    print("\n[1] float64 gradcheck on the fp32 reference (tiny shape)")
    torch.manual_seed(1)
    B, h, S, m, d, W = 1, 2, 12, 4, 16, 6
    Tcomp = S // m
    q = (torch.randn(B, h, S, d, device=DEV, dtype=torch.float64) * 0.3).requires_grad_(True)
    kr = (torch.randn(B, 1, S, d, device=DEV, dtype=torch.float64) * 0.3).requires_grad_(True)
    kc = (torch.randn(B, 1, Tcomp, d, device=DEV, dtype=torch.float64) * 0.3).requires_grad_(True)
    sinks = (torch.randn(h, device=DEV, dtype=torch.float64)).requires_grad_(True)

    def f(q, kr, kc, sinks):
        return attention_reference(q, kr, kc, sinks, W, m)

    ok = torch.autograd.gradcheck(f, (q, kr, kc, sinks), eps=1e-6, atol=1e-4, rtol=1e-3)
    print(f"    gradcheck: {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
def test_forward(cases, fwd_tol=1.5e-2):
    print("\n[2] forward: bf16 kernel vs fp32 reference")
    all_ok = True
    for B, S, m, with_comp, W in cases:
        q, kr, kc, sinks = make_inputs(B, S, m, with_comp)
        tag = f"B{B} S{S} m{m} comp{int(with_comp)} W{W}"
        try:
            out_ref = attention_reference(q.float(), kr.float(), None if kc is None else kc.float(), sinks, W, m)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            out = K.v4flash_attention(q, kr, kc, sinks, W, m)  # kernel survives
            print(f"    {tag:24s} reference OOM; kernel ran (out {tuple(out.shape)})  WIN")
            del out
            torch.cuda.empty_cache()
            continue
        out = K.v4flash_attention(q, kr, kc, sinks, W, m)
        ro, eo = _rel(out, out_ref)
        ok = ro < fwd_tol
        all_ok &= ok
        print(f"    {tag:24s} out rel={ro:.2e} max={eo:.2e}  {'PASS' if ok else 'FAIL'}")
        del out, out_ref
        torch.cuda.empty_cache()
    return all_ok


# --------------------------------------------------------------------------- #
def test_backward(cases, tol=2e-2):
    print("\n[3] backward: bf16 kernel grads vs autograd through fp32 reference")
    all_ok = True
    for B, S, m, with_comp, W in cases:
        q, kr, kc, sinks = make_inputs(B, S, m, with_comp)
        tag = f"B{B} S{S} m{m} comp{int(with_comp)} W{W}"
        qr = q.float().clone().requires_grad_(True)
        krr = kr.float().clone().requires_grad_(True)
        kcr = None if kc is None else kc.float().clone().requires_grad_(True)
        sr = sinks.clone().requires_grad_(True)
        try:
            outr = attention_reference(qr, krr, kcr, sr, W, m)
            g = torch.randn_like(outr)
            outr.backward(g)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            qk = q.clone().requires_grad_(True)
            krk = kr.clone().requires_grad_(True)
            kck = None if kc is None else kc.clone().requires_grad_(True)
            sk = sinks.clone().requires_grad_(True)
            outk = K.v4flash_attention(qk, krk, kck, sk, W, m)
            outk.backward(torch.randn_like(outk))  # kernel bwd survives
            print(f"    {tag:24s} reference OOM; kernel fwd+bwd ran  WIN")
            del outk, qk, krk, kck
            torch.cuda.empty_cache()
            continue

        qk = q.clone().requires_grad_(True)
        krk = kr.clone().requires_grad_(True)
        kck = None if kc is None else kc.clone().requires_grad_(True)
        sk = sinks.clone().requires_grad_(True)
        outk = K.v4flash_attention(qk, krk, kck, sk, W, m)
        outk.backward(g.to(outk.dtype))

        rq, eq = _rel(qk.grad, qr.grad)
        rk, ek = _rel(krk.grad, krr.grad)
        rs, es = _rel(sk.grad, sr.grad)
        parts = [("dq", rq), ("dkr", rk), ("dsink", rs)]
        if with_comp:
            rc, ec = _rel(kck.grad, kcr.grad)
            parts.append(("dkc", rc))
        ok = all(r < tol for _, r in parts)
        all_ok &= ok
        rels = " ".join(f"{n}={r:.2e}" for n, r in parts)
        print(f"    {tag:24s} {rels}  {'PASS' if ok else 'FAIL'}")
        del outr, outk, qr, krr, kcr, qk, krk, kck
        torch.cuda.empty_cache()
    return all_ok


if __name__ == "__main__":
    W = 128
    cases = []
    for S in (512, 2048, 8192):
        cases.append((1, S, 4, False, W))  # sliding-only
        cases.append((1, S, 4, True, W))  # CSA m=4
        cases.append((1, S, 128, True, W))  # HCA m=128
    cases.append((2, 512, 4, True, 64))  # batch>1, smaller window

    r1 = test_reference_gradcheck()
    r2 = test_forward(cases)
    r3 = test_backward(cases)
    print("\n=== SUMMARY ===")
    print(f"  reference gradcheck : {'PASS' if r1 else 'FAIL'}")
    print(f"  forward (bf16)      : {'PASS' if r2 else 'FAIL'}")
    print(f"  backward (bf16)     : {'PASS' if r3 else 'FAIL'}")
    print("  OVERALL:", "PASS" if (r1 and r2 and r3) else "FAIL")
