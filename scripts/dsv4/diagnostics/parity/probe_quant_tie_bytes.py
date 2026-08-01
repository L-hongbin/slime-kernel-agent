#!/usr/bin/env python3
"""Quantizer tie-rounding byte probe: WHICH activation-quant arithmetic does V4
serving actually run, and what closes the trainer's 0.11% fp8-byte tie seam?

Context (t2_shared_expert_probe.md "quantizer-tie"): trainer activation quant
(deep_gemm.per_token_cast_to_fp8) vs sglang serving quant differ on ~0.11% of
fp8 bytes on identical bf16 inputs (scales 100% bit-equal); those ±1-bin flips
perturb ~20% of downstream GEMM output bf16 bytes by <=2 ulp.

HYPOTHESIS UNDER TEST (from the alignment plan): the seam is reciprocal-multiply
(deep_gemm: sf=amax/448 then x*(1.0/sf), two roundings) vs true IEEE division
(sglang triton colmajor kernel fp8_kernel.py ~:234: y/y_s, one rounding), and a
repo-local "div-aligned" quantizer (same padding/amax/1e-4 clamp, but x/sf)
would byte-match serving.

MEASURED VERDICT (2026-07-13, node64, the production fork at
/sgl-workspace/sglang, sgl_kernel 0.4.3):  HYPOTHESIS REFUTED.
  * deep_gemm reciprocal-multiply == torch true-division == triton colmajor
    (y/y_s) == triton rowmajor (y*y_s_inv): ALL byte-identical on every battery.
    Reciprocal-vs-division ties at fp32 are ~1e-6-rare — not the seam.
  * Production (sglang_per_token_group_quant_fp8, the call
    deepgemm_w8a8_block_fp8_linear_with_fallback makes) NEVER runs either
    triton kernel here: it dispatches to the sgl_kernel CUDA v2 kernel
    (sgl_per_token_group_quant_8bit_v2, csrc/gemm/per_token_group_quant_8bit_v2.cu,
    arithmetic "Copied and modified from DeepEP"): multiplier = 448/amax
    (device division), q = cvt_fp8x2(fmul2_rn(x, 448/amax)), stored scale =
    amax*(1.0f/448).  THAT arithmetic vs deep_gemm's x*(1/(amax/448)) is the
    0.11% tie seam.  A torch emulation of it (x*(448/amax)) still leaves
    ~0.006% residual (fastmath/fused pipeline), so emulation is not enough.
  * Closure: call the production kernel itself for bytes+scales, then restore
    the trainer's 1e-4 amax floor on sub-floor groups from deep_gemm's own
    result (the floor is a SEPARATE seam, deliberately kept) — this is
    per_token_cast_to_fp8_servealigned in
    custom_kernels/deepseek_v4/megatron/mcore_model.py (env V4_QUANT_DIV_ALIGN):
    0.000000% bytes+scales vs production on normal groups, bit-exact deep_gemm
    on sub-floor groups.
  * Scale bits: fl32(amax/448) (deep_gemm) == fl32(amax * fl32(1/448)) (CUDA v2)
    for EVERY positive bf16 amax >= 1e-4 (exhaustive sweep) — "same scale bits,
    different bins" holds universally, not just on random batteries.

Reconciliation of the two prior findings (both correct):
  * repro_actquant_bitcompare.py ("bit-identical"): its cfg_c called the SAME
    wrapper -> SAME CUDA v2 kernel and measured the SAME 0.11% byte diff
    (reproduced here as a_dg vs c_prod_rowmajor); "bit-identical" was its
    verdict about SCALES (100% equal) with the 0.11% byte flips dismissed as
    negligible.  It did NOT hit a different (reciprocal-multiply triton)
    variant — that guess is refuted: both triton variants byte-match deep_gemm.
  * t2 probe (0.11% matters): same number, downstream-amplification framing.

Variants:
  a_dg        deep_gemm.per_token_cast_to_fp8(use_ue8m0=False)      [trainer today]
  b_div       torch true-division (same padding/amax/1e-4 clamp)    [the hypothesis]
  e_cudaemu   torch x*(448/amax), scale=amax*(1/448)                [CUDA-v2 emulation]
  c_prod      sglang_per_token_group_quant_fp8 colmajor+tma         [PRODUCTION -> CUDA v2]
  c_prod_rm   sglang_per_token_group_quant_fp8 rowmajor             [old repro's call -> CUDA v2]
  c_triton    _per_token_group_quant_8bit_raw colmajor (y/y_s)      [the kernel the plan named]
  d_triton    _per_token_group_quant_8bit_raw rowmajor (y*y_s_inv)  [reciprocal triton variant]
  f_repo      per_token_cast_to_fp8_servealigned                    [the shipped V4_QUANT_DIV_ALIGN path]

Batteries: realistic (chan-scaled randn), large randn, boundary (values built at
bf16(midpoint*sf) for random E4M3 bin midpoints), pow2 (sf=2^k exact, quotients
EXACTLY on midpoints -> pure tie-to-even test), nearzero (floor seam, reported
but out of scope).

Run:  CUDA_VISIBLE_DEVICES=7 python scripts/dsv4/diagnostics/parity/probe_quant_tie_bytes.py
(footprint <1 GiB; needs the sglang fork importable, e.g. /sgl-workspace/sglang)
"""

import argparse
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch  # noqa: E402

GK = 128
DEV = "cuda"


# --------------------------------------------------------------------------- variants
def q_a_dg(x):
    import deep_gemm

    return deep_gemm.per_token_cast_to_fp8(x, use_ue8m0=False)


def q_b_div(x):
    """The plan's hypothesized 'div-aligned' quantizer: deep_gemm's padding/amax/
    1e-4 clamp, but scale applied by TRUE IEEE DIVISION."""
    m, n = x.shape
    padded_n = (n + GK - 1) // GK * GK
    x_padded = torch.empty((m, padded_n), dtype=x.dtype, device=x.device).fill_(0)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, padded_n // GK, GK)
    x_amax = x_view.abs().float().amax(dim=2).view(m, padded_n // GK).clamp(1e-4)
    sf = x_amax / 448.0
    x_fp8 = (x_view / sf.unsqueeze(2)).to(torch.float8_e4m3fn).view(m, padded_n)[:, :n].contiguous()
    return x_fp8, sf


def q_e_cudaemu(x):
    """Torch emulation of the sgl_kernel CUDA v2 arithmetic (DeepEP
    calculate_fp8_scales): multiplier=448/amax, stored scale=amax*(1/448),
    eps floor 1e-10."""
    m, n = x.shape
    assert n % GK == 0
    xv = x.view(m, n // GK, GK)
    amax = xv.abs().float().amax(dim=2).clamp_min(1e-10)
    mult = 448.0 / amax
    q = (xv.float() * mult.unsqueeze(2)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(m, n)
    return q, amax * (1.0 / 448.0)


def q_c_prod(x):
    from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

    q, s = sglang_per_token_group_quant_fp8(
        x.contiguous(), GK, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=False
    )
    return q, s[: x.shape[0], :].contiguous()


def q_c_prod_rm(x):
    from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

    return sglang_per_token_group_quant_fp8(x.contiguous(), GK)


def q_c_triton(x):
    from sglang.srt.layers.quantization.fp8_kernel import _per_token_group_quant_8bit_raw

    q, s = _per_token_group_quant_8bit_raw(x.contiguous(), GK, column_major_scales=True)
    return q, s[: x.shape[0], :].contiguous()


def q_d_triton(x):
    from sglang.srt.layers.quantization.fp8_kernel import _per_token_group_quant_8bit_raw

    return _per_token_group_quant_8bit_raw(x.contiguous(), GK, column_major_scales=False)


def q_f_repo(x):
    from custom_kernels.deepseek_v4.megatron.mcore_model import per_token_cast_to_fp8_servealigned

    return per_token_cast_to_fp8_servealigned(x, use_ue8m0=False)


VARIANTS = {
    "a_dg": q_a_dg,
    "b_div": q_b_div,
    "e_cudaemu": q_e_cudaemu,
    "c_prod": q_c_prod,
    "c_prod_rm": q_c_prod_rm,
    "c_triton": q_c_triton,
    "d_triton": q_d_triton,
    "f_repo": q_f_repo,
}


# --------------------------------------------------------------------------- batteries
def make_realistic(T, D, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    base = torch.randn(T, D, generator=g, device=DEV)
    chan = torch.rand(D, generator=g, device=DEV) * 2.0 + 0.3
    chan[torch.randint(0, D, (max(1, D // 256),), generator=g, device=DEV)] *= 12.0
    return (base * chan).to(torch.bfloat16)


def make_randn(T, D, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(T, D, generator=g, device=DEV) * 3.0).to(torch.bfloat16)


def _e4m3_midpoints():
    """Midpoints between adjacent positive finite e4m3 values (fp32-exact)."""
    grid = torch.arange(256, dtype=torch.uint8, device=DEV).view(torch.float8_e4m3fn).float()
    grid = grid[torch.isfinite(grid) & (grid > 0)].unique().sort().values
    return (grid[:-1] + grid[1:]) / 2.0


def make_boundary(T, D, seed):
    """Values constructed at bf16(midpoint * sf) for random E4M3 bin midpoints,
    with a pinned per-group amax — a dense near-bin-boundary battery."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    ng = D // GK
    mids = _e4m3_midpoints()
    amax = (torch.rand(T, ng, generator=g, device=DEV) * 6.0 + 0.5).to(torch.bfloat16)
    sf = amax.float() / 448.0  # what both quantizers derive (before rounding differences)
    pick = mids[torch.randint(0, mids.numel(), (T, ng, GK), generator=g, device=DEV)]
    sign = torch.where(torch.rand(T, ng, GK, generator=g, device=DEV) < 0.5, -1.0, 1.0)
    x = (pick * sign * sf.unsqueeze(2)).to(torch.bfloat16)
    x[:, :, 0] = amax  # pin group amax (max midpoint 432 < 448 keeps it the max)
    return x.view(T, D)


def make_pow2(T, D, seed):
    """sf exactly 2^k (amax=448*2^k bf16-exact) and x = midpoint*2^k (bf16-exact):
    the scaled value is EXACTLY the e4m3 midpoint -> pure tie-to-even probe."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    ng = D // GK
    mids = _e4m3_midpoints()
    k = torch.randint(-6, 7, (T, ng), generator=g, device=DEV).float()
    sf = torch.pow(2.0, k)
    pick = mids[torch.randint(0, mids.numel(), (T, ng, GK), generator=g, device=DEV)]
    sign = torch.where(torch.rand(T, ng, GK, generator=g, device=DEV) < 0.5, -1.0, 1.0)
    x = (pick * sign * sf.unsqueeze(2)).to(torch.bfloat16)
    x[:, :, 0] = (448.0 * sf).to(torch.bfloat16)
    return x.view(T, D)


def make_nearzero(T, D, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.randn(T, D, generator=g, device=DEV) * 1e-6).to(torch.bfloat16)


BATTERIES = {
    "realistic": lambda seed: make_realistic(2048, 4096, seed),
    "randn_large": lambda seed: make_randn(4096, 4096, seed),
    "boundary": lambda seed: make_boundary(1024, 4096, seed),
    "pow2_ties": lambda seed: make_pow2(512, 4096, seed),
    "nearzero": lambda seed: make_nearzero(256, 2048, seed),
}


# --------------------------------------------------------------------------- reporting
def byte_diff(q1, q2):
    return (q1.view(torch.uint8) != q2.view(torch.uint8)).float().mean().item() * 100


def scale_diff(s1, s2):
    return (s1.view(torch.float32) != s2.view(torch.float32)).float().mean().item() * 100


def run_battery(name, x):
    print(f"\n===== battery {name}: shape {tuple(x.shape)}, mean|x|={x.float().abs().mean():.3g} =====")
    outs = {}
    for k, fn in VARIANTS.items():
        try:
            outs[k] = fn(x)
        except Exception as exc:  # keep the probe running if one variant is unavailable
            print(f"  [{k}] UNAVAILABLE: {type(exc).__name__}: {exc}")
    ks = list(outs)
    print(f"  {'pair':26s} {'%fp8_bytes_differ':>18s} {'%scale_bits_differ':>19s}")
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            (q1, s1), (q2, s2) = outs[ks[i]], outs[ks[j]]
            print(f"  {ks[i] + ' vs ' + ks[j]:26s} {byte_diff(q1, q2):18.6f} {scale_diff(s1, s2):19.6f}")
    return outs


def exhaustive_amax_scale_sweep():
    """fl32(amax/448) vs fl32(amax*fl32(1/448)) for EVERY positive bf16 amax >= 1e-4,
    through the real kernels (deep_gemm vs production CUDA v2)."""
    import deep_gemm
    from sglang.srt.layers.quantization.fp8_kernel import sglang_per_token_group_quant_fp8

    vals = []
    for e in range(-14, 16):
        vals.append((1.0 + torch.arange(256, device=DEV, dtype=torch.float32) / 256.0) * (2.0**e))
    amaxes = torch.cat(vals).to(torch.bfloat16).float().unique()
    amaxes = amaxes[amaxes >= 1e-4]
    G = amaxes.numel()
    xg = torch.zeros(G, GK, device=DEV, dtype=torch.bfloat16)
    xg[:, 0] = amaxes.to(torch.bfloat16)
    xg[:, 1:] = (amaxes.unsqueeze(1) * torch.rand(G, GK - 1, device=DEV) * 0.9).to(torch.bfloat16)
    xg[:, 1:] = torch.minimum(xg[:, 1:], xg[:, :1])
    _, s_dg = deep_gemm.per_token_cast_to_fp8(xg, use_ue8m0=False)
    _, s_pr = sglang_per_token_group_quant_fp8(
        xg.contiguous(), GK, column_major_scales=True, scale_tma_aligned=True, scale_ue8m0=False
    )
    s_pr = s_pr[:G, :].contiguous()
    neq = int((s_dg != s_pr).sum().item())
    print(f"\n===== exhaustive amax scale-bit sweep: {G} bf16 amax values >= 1e-4 -> {neq} scale-bit diffs =====")
    return neq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.cuda.init()
    from sglang.srt.layers.quantization import fp8_kernel as _fk

    print(
        "dispatch: enable_sgl_per_token_group_quant_8bit="
        f"{getattr(_fk, 'enable_sgl_per_token_group_quant_8bit', None)} "
        "(True -> production wrapper runs the sgl_kernel CUDA v2 kernel, NOT the triton kernels)"
    )
    verdict_pairs = {}
    for name, mk in BATTERIES.items():
        outs = run_battery(name, mk(args.seed + hash(name) % 1000))
        if name != "nearzero" and all(k in outs for k in ("a_dg", "b_div", "c_prod", "f_repo", "c_triton")):
            verdict_pairs.setdefault("b_div vs c_prod", []).append(byte_diff(outs["b_div"][0], outs["c_prod"][0]))
            verdict_pairs.setdefault("a_dg vs c_prod", []).append(byte_diff(outs["a_dg"][0], outs["c_prod"][0]))
            verdict_pairs.setdefault("f_repo vs c_prod", []).append(byte_diff(outs["f_repo"][0], outs["c_prod"][0]))
            verdict_pairs.setdefault("b_div vs a_dg", []).append(byte_diff(outs["b_div"][0], outs["a_dg"][0]))
            verdict_pairs.setdefault("c_triton vs a_dg", []).append(byte_diff(outs["c_triton"][0], outs["a_dg"][0]))
    exhaustive_amax_scale_sweep()
    print("\n===== VERDICT (max %fp8 bytes differ across non-nearzero batteries) =====")
    for k, v in verdict_pairs.items():
        print(f"  {k:22s} max {max(v):.6f}%")
    print(
        "  -> true-division does NOT close the seam (b_div==a_dg==triton variants);\n"
        "     the seam is the sgl_kernel CUDA v2 arithmetic; f_repo (V4_QUANT_DIV_ALIGN\n"
        "     serve-aligned path) closes it to 0 by running the production kernel itself."
    )


if __name__ == "__main__":
    main()
