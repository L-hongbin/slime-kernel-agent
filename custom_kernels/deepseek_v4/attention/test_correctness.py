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


def make_topk_mask(B, S, Tcomp, k, seed=0):
    """Random CSA-style top-k selection mask [B,S,Tcomp] (bool, True=selected).

    When ``Tcomp <= k`` the whole compressed axis is selected (mask no-op: the
    sparse kernel must then match the dense kernel exactly).  Otherwise each
    (batch, query) keeps a random size-``k`` subset -- the structural causal
    threshold is ANDed in by both kernel and reference, so an arbitrary subset is
    a valid test of the top-k masking path.
    """
    if Tcomp <= k:
        return torch.ones(B, S, Tcomp, dtype=torch.bool, device=DEV)
    g = torch.Generator().manual_seed(seed)
    idx = torch.rand(B, S, Tcomp, generator=g).topk(k, dim=-1).indices
    mask = torch.zeros(B, S, Tcomp, dtype=torch.bool)
    mask.scatter_(2, idx, True)
    return mask.to(DEV)


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


# --------------------------------------------------------------------------- #
def test_sparse(cases, fwd_tol=1.5e-2, bwd_tol=2e-2):
    """Sparse (top-k over compressed) kernel vs reference, fwd + all grads.

    Each case is (B, S, m, W, k): the CSA lightning-indexer top-k budget is ``k``.
    Covers Tcomp>k (real selection) and Tcomp<=k (mask no-op == dense).
    """
    print("\n[4] sparse top-k: bf16 kernel vs fp32 reference (fwd + grads)")
    all_ok = True
    for B, S, m, W, k in cases:
        q, kr, kc, sinks = make_inputs(B, S, m, with_comp=True)
        Tcomp = kc.shape[2]
        mask = make_topk_mask(B, S, Tcomp, k)
        tag = f"B{B} S{S} m{m} Tcomp{Tcomp} k{k} W{W}"

        qr = q.float().clone().requires_grad_(True)
        krr = kr.float().clone().requires_grad_(True)
        kcr = kc.float().clone().requires_grad_(True)
        sr = sinks.clone().requires_grad_(True)
        try:
            outr = attention_reference(qr, krr, kcr, sr, W, m, comp_topk_mask=mask)
            g = torch.randn_like(outr)
            outr.backward(g)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"    {tag:36s} reference OOM; skipped")
            continue

        qk = q.clone().requires_grad_(True)
        krk = kr.clone().requires_grad_(True)
        kck = kc.clone().requires_grad_(True)
        sk = sinks.clone().requires_grad_(True)
        outk = K.v4flash_attention(qk, krk, kck, sk, W, m, comp_topk_mask=mask)
        outk.backward(g.to(outk.dtype))

        ro, _ = _rel(outk, outr)
        rq, _ = _rel(qk.grad, qr.grad)
        rk, _ = _rel(krk.grad, krr.grad)
        rc, _ = _rel(kck.grad, kcr.grad)
        rs, _ = _rel(sk.grad, sr.grad)
        ok = ro < fwd_tol and max(rq, rk, rc, rs) < bwd_tol
        all_ok &= ok
        print(
            f"    {tag:36s} out={ro:.2e} dq={rq:.2e} dkr={rk:.2e} dkc={rc:.2e} ds={rs:.2e}  "
            f"{'PASS' if ok else 'FAIL'}"
        )
        del outr, outk, qr, krr, kcr, qk, krk, kck
        torch.cuda.empty_cache()
    return all_ok


# --------------------------------------------------------------------------- #
# Context-parallel (CP2) parity: per-rank q_pos0/raw_halo kernels vs whole-seq  #
# --------------------------------------------------------------------------- #
def _maxdiff(a, b):
    """Max |a-b| over the float-cast; 0.0 iff bitwise-identical (bf16 casts exact)."""
    return (a.float() - b.float()).abs().max().item()


def _fwd_lse(q, k_raw, k_comp, sinks, window, m, q_pos0=0, raw_halo=0, block_M=64, block_N=64, comp_topk_mask=None):
    """Padded forward returning (out[:, :, :S], lse[:, :, :S]).

    Mirrors ``K._V4FlashAttn.forward``'s padding / kvt construction so the CP test can
    read Lse (the public wrapper returns only ``out``). Routes through K's real fwd
    caches with the (q_pos0, raw_halo)-inclusive key, so it exercises BOTH that the
    builders USE the constants (design M1) and that the fwd cache key includes them
    (M2): calling this for rank0 (q_pos0=0) then rank1 (q_pos0=L) with the SAME S would
    silently reuse rank0's kernel if either were missing, and Lse parity would fail.
    """
    B, H, S, D = q.shape
    Tcomp = 0 if k_comp is None else k_comp.shape[2]
    Sp = K._ceil(S, max(block_M, block_N)) * max(block_M, block_N)
    Sr = raw_halo + Sp
    Tcp = K._ceil(Tcomp, block_N) * block_N if Tcomp > 0 else 0
    KV = Sr + Tcp
    dev, dt = q.device, q.dtype
    sparse = comp_topk_mask is not None and Tcomp > 0

    q_pad = q.new_zeros(B, H, Sp, D)
    q_pad[:, :, :S] = q
    kvt = q.new_zeros(B, 1, KV, D)
    kvt[:, :, : raw_halo + S] = k_raw
    if Tcomp > 0:
        kvt[:, :, Sr : Sr + Tcomp] = k_comp
    sinks_f = sinks.float().contiguous()

    key = (B, H, S, Tcomp, Sp, Tcp, window, m, D, block_M, block_N, str(dt), K._USE_WS_FWD, q_pos0, raw_halo)
    if sparse:
        mask_pad = torch.zeros(B, Sp, Tcp, dtype=torch.uint8, device=dev)
        mask_pad[:, :S, :Tcomp] = comp_topk_mask.to(torch.uint8)
        builder = K._build_fwd_ws_sparse if K._USE_WS_FWD else K._build_fwd_sparse
        fwd = K._get(
            K._FWD_SPARSE_CACHE,
            builder,
            key,
            B,
            H,
            S,
            Tcomp,
            Sp,
            Tcp,
            window,
            m,
            D,
            block_M,
            block_N,
            q_pos0,
            raw_halo,
        )
        out_pad, lse_pad = fwd(q_pad, kvt, sinks_f, mask_pad)
    else:
        builder = K._build_fwd_ws if K._USE_WS_FWD else K._build_fwd
        fwd = K._get(
            K._FWD_CACHE, builder, key, B, H, S, Tcomp, Sp, Tcp, window, m, D, block_M, block_N, q_pos0, raw_halo
        )
        out_pad, lse_pad = fwd(q_pad, kvt, sinks_f)
    return out_pad[:, :, :S], lse_pad[:, :, :S]


def test_cp_parity(cases, bwd_close_tol=2e-2, Bc=1, Hc=8, halo=128):
    """CP2==CP1 rung-1: two per-rank kernels reproduce the whole-sequence kernel.

    Emulates CP2 in ONE process on a length-S sequence split at L=S/2:
      rank0 = kernel on queries [0:L],  raw [0:L],       q_pos0=0, raw_halo=0
      rank1 = kernel on queries [L:S],  raw [L-halo:S],  q_pos0=L, raw_halo=halo
    (the compressed axis is GLOBAL / replicated on both ranks). Expected identities:
      * out, lse, dq  : BITWISE-equal to the whole-seq kernel's slices (each query's
        receptive field is fully local and the visited kv blocks match in value+order).
      * dk_raw        : whole-seq == rank0 over [0:L] + rank1's local rows into [L:S]
        + rank1's HALO rows folded back into [L-halo:L]  (within bf16 tol -- the split
        changes fp reduction/rounding order, not the summed terms).
      * dk_comp, dsink: summed across ranks == whole-seq  (within bf16 tol).
    Hc is intentionally < production 64 to keep the compile sweep light; the kernel is
    per-head (grid dim H) so head count does not affect the parity logic.
    """
    print("\n[5] CP2 parity: per-rank q_pos0/raw_halo kernels vs whole-sequence kernel")
    all_ok = True
    for S, m, with_comp, W, sparse in cases:
        L = S // 2
        assert L % 64 == 0, f"l_local {L} must be block-aligned for this test"
        torch.manual_seed(0)
        scale = 0.3
        q_g = torch.randn(Bc, Hc, S, D, device=DEV, dtype=torch.bfloat16) * scale
        kr_g = torch.randn(Bc, 1, S, D, device=DEV, dtype=torch.bfloat16) * scale
        Tcomp = (S // m) if with_comp else 0
        kc_g = (torch.randn(Bc, 1, Tcomp, D, device=DEV, dtype=torch.bfloat16) * scale) if with_comp else None
        sinks = torch.randn(Hc, device=DEV, dtype=torch.float32)
        mask_g = make_topk_mask(Bc, S, Tcomp, 128) if (sparse and with_comp) else None
        m0 = None if mask_g is None else mask_g[:, :L, :].contiguous()
        m1 = None if mask_g is None else mask_g[:, L:, :].contiguous()
        tag = f"S{S} L{L} m{m} comp{int(with_comp)} W{W} sp{int(sparse)}"

        def sl(a, lo, hi):
            return a[:, :, lo:hi].contiguous()

        # ---- forward: out + Lse BITWISE parity (direct builder path reads Lse) ----
        o_ref, l_ref = _fwd_lse(q_g, kr_g, kc_g, sinks, W, m, 0, 0, comp_topk_mask=mask_g)
        o0, l0 = _fwd_lse(sl(q_g, 0, L), sl(kr_g, 0, L), kc_g, sinks, W, m, 0, 0, comp_topk_mask=m0)
        o1, l1 = _fwd_lse(sl(q_g, L, S), sl(kr_g, L - halo, S), kc_g, sinks, W, m, L, halo, comp_topk_mask=m1)
        fwd_bit = max(
            _maxdiff(o0, o_ref[:, :, :L]),
            _maxdiff(o1, o_ref[:, :, L:]),
            _maxdiff(l0, l_ref[:, :, :L]),
            _maxdiff(l1, l_ref[:, :, L:]),
        )

        # ---- backward parity through the public wrapper (also checks _FWD_CACHE key) ----
        def leaf(a):
            return None if a is None else a.clone().requires_grad_(True)

        qg, krg, kcg, sg = leaf(q_g), leaf(kr_g), leaf(kc_g), leaf(sinks)
        og = K.v4flash_attention(qg, krg, kcg, sg, W, m, comp_topk_mask=mask_g)
        gg = torch.randn_like(og)
        og.backward(gg)

        q0, k0, c0, s0 = leaf(sl(q_g, 0, L)), leaf(sl(kr_g, 0, L)), leaf(kc_g), leaf(sinks)
        o0w = K.v4flash_attention(q0, k0, c0, s0, W, m, comp_topk_mask=m0, q_pos0=0, raw_halo=0)
        o0w.backward(sl(gg, 0, L))

        q1, k1, c1, s1 = leaf(sl(q_g, L, S)), leaf(sl(kr_g, L - halo, S)), leaf(kc_g), leaf(sinks)
        o1w = K.v4flash_attention(q1, k1, c1, s1, W, m, comp_topk_mask=m1, q_pos0=L, raw_halo=halo)
        o1w.backward(sl(gg, L, S))

        # wrapper out BITWISE (M2: _FWD_CACHE key must include q_pos0/raw_halo)
        out_bit = max(_maxdiff(o0w, og.detach()[:, :, :L]), _maxdiff(o1w, og.detach()[:, :, L:]))
        # dq BITWISE
        dq_bit = max(_maxdiff(q0.grad, qg.grad[:, :, :L]), _maxdiff(q1.grad, qg.grad[:, :, L:]))
        # dk_raw: fold rank1's halo rows [0:halo] back into global [L-halo:L]
        dkr = torch.zeros_like(krg.grad, dtype=torch.float32)
        dkr[:, :, :L] += k0.grad.float()
        dkr[:, :, L - halo : L] += k1.grad[:, :, :halo].float()
        dkr[:, :, L:] += k1.grad[:, :, halo:].float()
        rkr = _rel(dkr, krg.grad.float())[0]
        rds = _rel(s0.grad + s1.grad, sg.grad)[0]
        rkc = _rel(c0.grad.float() + c1.grad.float(), kcg.grad.float())[0] if with_comp else 0.0

        ok = (fwd_bit == 0.0) and (out_bit == 0.0) and (dq_bit == 0.0) and max(rkr, rkc, rds) < bwd_close_tol
        all_ok &= ok
        print(
            f"    {tag:32s} bitwise[out/lse={fwd_bit:.1e} wrap_out={out_bit:.1e} dq={dq_bit:.1e}] "
            f"close[dkr={rkr:.1e} dkc={rkc:.1e} ds={rds:.1e}]  {'PASS' if ok else 'FAIL'}"
        )
        del q_g, kr_g, kc_g, og, o0w, o1w, o_ref, o0, o1
        torch.cuda.empty_cache()
    return all_ok


def test_cp_alignment_guard():
    """The kernel wrapper rejects a raw axis that is not block_N-aligned.

    The CP halo is 128 (not the literal window-1 = 127) precisely so that raw_halo AND
    raw_halo+Sp stay divisible by block_N (=64); a 127-halo must raise. (128-alignment
    of l_local itself -- the S=1152 -> l_local=576 concern -- is a data-layer/compressor
    requirement enforced upstream; the attention kernel only needs block_N alignment,
    which the S=1152 case in test_cp_parity confirms it accepts.)
    """
    print("\n[6] CP alignment guard: non-block-aligned raw_halo must assert")
    B, Hc, S = 1, 8, 512
    q = torch.randn(B, Hc, S, D, device=DEV, dtype=torch.bfloat16) * 0.3
    kr = torch.randn(B, 1, 127 + S, D, device=DEV, dtype=torch.bfloat16) * 0.3  # 127-halo
    sinks = torch.randn(Hc, device=DEV, dtype=torch.float32)
    try:
        K.v4flash_attention(q, kr, None, sinks, 128, 4, q_pos0=S, raw_halo=127)
        print("    raw_halo=127 did NOT raise  FAIL")
        return False
    except AssertionError as e:
        print(f"    raw_halo=127 raised AssertionError  PASS  ({str(e)[:56]})")
        return True


if __name__ == "__main__":
    W = 128
    cases = []
    for S in (512, 2048, 8192):
        cases.append((1, S, 4, False, W))  # sliding-only
        cases.append((1, S, 4, True, W))  # CSA m=4
        cases.append((1, S, 128, True, W))  # HCA m=128
    cases.append((2, 512, 4, True, 64))  # batch>1, smaller window

    # Sparse cases (B, S, m, W, k): Tcomp = S//m.
    sparse_cases = [
        (1, 2048, 4, W, 128),  # Tcomp=512 > k=128   (real top-k selection)
        (1, 4096, 4, W, 512),  # Tcomp=1024 > k=512  (formal CSA budget; ref fits)
        (1, 512, 4, W, 512),  # Tcomp=128 <= k=512   (mask no-op == dense)
        (2, 512, 4, 64, 40),  # batch>1, small window, Tcomp=128 > k=40
    ]

    # CP2 parity cases (S_total, m, with_comp, W, sparse): split at L=S/2, halo=128.
    cp_cases = [
        (1024, 4, True, W, False),  # CSA m=4, dense-over-compressed
        (1024, 128, True, W, False),  # HCA m=128
        (1024, 4, False, W, False),  # sliding-window only (no compressed axis)
        (1024, 4, True, W, True),  # sparse top-k over compressed
        (1152, 4, True, W, False),  # odd shape: L=576 (64-aligned, NOT 128-aligned)
    ]

    r1 = test_reference_gradcheck()
    r2 = test_forward(cases)
    r3 = test_backward(cases)
    r4 = test_sparse(sparse_cases)
    r5 = test_cp_parity(cp_cases)
    r6 = test_cp_alignment_guard()
    print("\n=== SUMMARY ===")
    print(f"  reference gradcheck : {'PASS' if r1 else 'FAIL'}")
    print(f"  forward (bf16)      : {'PASS' if r2 else 'FAIL'}")
    print(f"  backward (bf16)     : {'PASS' if r3 else 'FAIL'}")
    print(f"  sparse top-k (bf16) : {'PASS' if r4 else 'FAIL'}")
    print(f"  CP2 parity (bf16)   : {'PASS' if r5 else 'FAIL'}")
    print(f"  CP2 align guard     : {'PASS' if r6 else 'FAIL'}")
    print("  OVERALL:", "PASS" if (r1 and r2 and r3 and r4 and r5 and r6) else "FAIL")
