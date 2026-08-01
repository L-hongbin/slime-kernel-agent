"""T3 tests for the fixed fp32 partial/LSE/sink merge.

Covers the sglang-fork patch scripts/dsv4/patches/sglang_dsv4_fp32_partial_merge.patch
(flash_mla_sm120_triton.py): the sm120 sparse-decode path
keeps the raw (SWA) and compressed (c4/c128) attention partials in fp32, fuses
the LSE merge + attention-sink fold in fp32, and performs ONE final bf16 store —
removing the three bf16 seams the trainer's A1 kernel does not have. The
original deep probe is archived under `local_artifacts/deepseek-v4/retired_scripts/diagnostics/parity/`.

Runs on the node53 dev container (1 GPU, patched local sglang fork). Validates:

  1. the fixed path defaults ON; the retained legacy helper composition remains
     available only as an internal A/B oracle.
  2. fixed path == manual fp32 composition (dispatch pins).
  3. fixed-path LSE remains at fp32-rounding equivalence with the legacy oracle
     (the merge/sink LSE math does not depend on the partial-output dtype).
  4. the fixed path closes the seam error: vs a fp64 single-state oracle its error is
     at the single-bf16-store floor, strictly below the legacy oracle, for CSA/HCA/sliding
     shapes on BOTH bf16 and fp8 KV pools.
  5. no NaN/Inf and correct shapes/dtypes on edge cases (zero-length topk,
     all-invalid indices, no sink, no extra cache).
  6. the fixed path is CUDA-graph capturable: capture, mutate inputs, replay ==
     eager rerun bitwise (production always serves with CUDA graphs).

Invoke: pytest -q tests/deepseek-v4/test_dsv4_fp32_partial_merge.py
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU + patched sglang fork")

NOPE, ROPE, D = 448, 64, 512
SWA_WINDOW = 128
C4_TOPK = 512
PAGE = 64
H = 64
SM_SCALE = 1.0 / (D**0.5)


def _mod():
    from sglang.srt.layers.attention import flash_mla_sm120_triton as m

    return m


@pytest.fixture
def flag_guard():
    """Restore the module-level gate after each test that flips it."""
    m = _mod()
    orig = m._FP32_PARTIAL_MERGE
    yield m
    m._FP32_PARTIAL_MERGE = orig


# --------------------------------------------------------------------------- #
# Input builders for production shapes; the original deep probe is archived.
# --------------------------------------------------------------------------- #
def _bf16_cache(latent, page=PAGE):
    n = latent.shape[0]
    pages = (n + page - 1) // page
    c = torch.zeros(pages * page, D, dtype=torch.bfloat16, device=latent.device)
    c[:n] = latent
    return c.view(pages, page, 1, D).contiguous()


def _fp8_cache(latent, page=PAGE):
    from sglang.srt.layers.attention.dsv4.quant_k_cache import quant_to_nope_fp8_rope_bf16_pack_triton

    n = latent.shape[0]
    pages = (n + page - 1) // page
    pack = quant_to_nope_fp8_rope_bf16_pack_triton(latent.contiguous())
    data = torch.zeros(pages * page, 576, dtype=torch.uint8, device=latent.device)
    scal = torch.zeros(pages * page, 8, dtype=torch.uint8, device=latent.device)
    data[:n, :NOPE] = pack.k_nope_fp8.view(torch.uint8).view(n, NOPE)
    data[:n, NOPE:] = pack.k_rope_bf16.view(torch.uint8).view(n, ROPE * 2)
    scal[:n, :7] = pack.scale_k_nope_ue8m0.view(torch.uint8).view(n, -1)[:, :7]
    buf = torch.cat([data.view(pages, page * 576), scal.view(pages, page * 8)], dim=1).contiguous()
    return buf.view(pages, page, 1, 584)


def _dequant_fp8_values(latent):
    from sglang.srt.layers.attention.dsv4.quant_k_cache import quant_to_nope_fp8_rope_bf16_pack_triton

    n = latent.shape[0]
    pack = quant_to_nope_fp8_rope_bf16_pack_triton(latent.contiguous())
    scale = torch.exp2(pack.scale_k_nope_ue8m0.double() - 127)
    out = torch.empty(n, D, dtype=torch.float64, device=latent.device)
    out[:, :NOPE] = (pack.k_nope_fp8.double().view(n, NOPE // 64, 64) * scale.view(n, -1, 1)).reshape(n, NOPE)
    out[:, NOPE:] = pack.k_rope_bf16.double()
    return out


def _scenario(case: str, kv: str, seed: int = 7, S: int = 2048, P: int = 16):
    """Build (q, raw cache+idx+len, extra cache+idx+len, sink, oracle values)."""
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    latent = torch.randn(S, D, generator=g, device=dev).to(torch.bfloat16)
    q = (torch.randn(P, 1, H, D, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    sink = torch.randn(H, generator=g, device=dev).float()
    pos = torch.linspace(256, S - 1, P, device=dev).long()

    raw_idx = torch.full((P, 1, SWA_WINDOW), -1, dtype=torch.int32, device=dev)
    raw_len = torch.empty(P, dtype=torch.int32, device=dev)
    for j, i in enumerate(pos.tolist()):
        lo = max(0, i - SWA_WINDOW + 1)
        raw_idx[j, 0, : i - lo + 1] = torch.arange(lo, i + 1, dtype=torch.int32, device=dev)
        raw_len[j] = i - lo + 1

    m = {"sliding": 0, "csa": 4, "hca": 128}[case]
    comp = idx_c = len_c = None
    if m:
        t_comp = S // m
        comp = torch.randn(t_comp, D, generator=g, device=dev).to(torch.bfloat16)
        width = C4_TOPK if m == 4 else max(64, ((t_comp + 63) // 64) * 64)
        idx_c = torch.full((P, 1, width), -1, dtype=torch.int32, device=dev)
        len_c = torch.zeros(P, dtype=torch.int32, device=dev)
        gc = torch.Generator(device="cpu").manual_seed(seed + m)
        for j, i in enumerate(pos.tolist()):
            k = min((i + 1) // m, t_comp, width)
            if k <= 0:
                continue
            sel = torch.randperm(min((i + 1) // m, t_comp), generator=gc)[:k]
            idx_c[j, 0, :k] = sel.sort().values.to(torch.int32).to(dev)
            len_c[j] = k

    build = _bf16_cache if kv == "bf16" else _fp8_cache
    eff = latent.double() if kv == "bf16" else _dequant_fp8_values(latent)
    eff_c = None
    if m:
        eff_c = comp.double() if kv == "bf16" else _dequant_fp8_values(comp)
    return {
        "q": q,
        "raw_cache": build(latent),
        "raw_idx": raw_idx,
        "raw_len": raw_len,
        "comp_cache": build(comp) if m else None,
        "comp_idx": idx_c,
        "comp_len": len_c,
        "sink": sink,
        "eff_raw": eff,
        "eff_comp": eff_c,
    }


def _oracle(sc):
    """fp64 single-state softmax over raw+comp selected tokens (+ sink if
    present; sink=None exercises the no-sink merge branch)."""
    q = sc["q"].squeeze(1).double()
    P = q.shape[0]
    out = torch.empty(P, H, D, dtype=torch.float64, device=q.device)
    for b in range(P):
        vals = [sc["eff_raw"][sc["raw_idx"][b, 0, : sc["raw_len"][b]].long()]]
        if sc["eff_comp"] is not None and int(sc["comp_len"][b]) > 0:
            vals.append(sc["eff_comp"][sc["comp_idx"][b, 0, : sc["comp_len"][b]].long()])
        kv = torch.cat(vals, 0)
        s = (q[b] @ kv.T) * SM_SCALE
        if sc["sink"] is not None:
            s_all = torch.cat([s, sc["sink"].double().view(H, 1)], dim=1)
            mx = s_all.max(dim=1, keepdim=True).values
            e = torch.exp(s_all - mx)
            out[b] = (e[:, :-1] / e.sum(dim=1, keepdim=True)) @ kv
        else:
            mx = s.max(dim=1, keepdim=True).values
            e = torch.exp(s - mx)
            out[b] = (e / e.sum(dim=1, keepdim=True)) @ kv
    return out.unsqueeze(1)


def _run(sc):
    m = _mod()
    return m.flash_mla_sparse_decode_triton(
        sc["q"],
        sc["raw_cache"],
        sc["raw_idx"],
        sc["raw_len"],
        sc["sink"],
        D,
        SM_SCALE,
        sc["comp_cache"],
        sc["comp_idx"],
        sc["comp_len"],
    )


def _rel_rms(x, ref):
    return ((x.double() - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-30)).item()


# --------------------------------------------------------------------------- #
# 1. default OFF + env-off == legacy composition                              #
# --------------------------------------------------------------------------- #
def test_fixed_on_and_legacy_composition(flag_guard):
    m = flag_guard
    # The working container may still have the unpatched SGLang wheel. Pin the
    # post-patch production mode explicitly; the patch-source contract below
    # verifies that deployed copies default to this mode.
    m._FP32_PARTIAL_MERGE = True
    sc = _scenario("csa", "bf16")
    m._FP32_PARTIAL_MERGE = False
    out, lse = _run(sc)
    # compose the legacy pipeline by hand from the module's own helpers
    o1, l1 = m._run_triton_sparse_decode(sc["q"], sc["raw_cache"], sc["raw_idx"], sc["raw_len"], SM_SCALE)
    o2, l2 = m._run_triton_sparse_decode(sc["q"], sc["comp_cache"], sc["comp_idx"], sc["comp_len"], SM_SCALE)
    mo, ml = m._merge_partial_attn(o1, l1, o2, l2)
    so, sl = m._apply_attn_sink(mo, ml, sc["sink"])
    assert torch.equal(out, so)
    assert torch.equal(lse, sl.permute(0, 2, 1))
    assert out.dtype == torch.bfloat16


def test_patch_fixes_partial_merge_on_without_an_environment_gate():
    patch = (
        Path(__file__).resolve().parents[2] / "scripts/dsv4/patches/sglang_dsv4_fp32_partial_merge.patch"
    ).read_text()
    assert "+_FP32_PARTIAL_MERGE = True" in patch
    assert "SGLANG_DSV4_FP32_PARTIAL_MERGE" not in patch


# --------------------------------------------------------------------------- #
# 2/3. env-on dispatch pin + LSE bit-invariance                               #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", ["sliding", "csa", "hca"])
def test_env_on_dispatch_and_lse_invariance(flag_guard, case):
    m = flag_guard
    sc = _scenario(case, "bf16")

    m._FP32_PARTIAL_MERGE = False
    out_off, lse_off = _run(sc)
    m._FP32_PARTIAL_MERGE = True
    out_on, lse_on = _run(sc)

    # dispatch pin: env-on equals the manual fp32 composition
    o1, l1 = m._run_triton_sparse_decode(
        sc["q"], sc["raw_cache"], sc["raw_idx"], sc["raw_len"], SM_SCALE, out_dtype=torch.float32
    )
    assert o1.dtype == torch.float32
    o2 = l2 = None
    if sc["comp_cache"] is not None:
        o2, l2 = m._run_triton_sparse_decode(
            sc["q"], sc["comp_cache"], sc["comp_idx"], sc["comp_len"], SM_SCALE, out_dtype=torch.float32
        )
    mo, ml = m._merge_partials_and_sink_fp32(o1, l1, o2, l2, sc["sink"])
    assert torch.equal(out_on, mo)
    assert torch.equal(lse_on, ml.permute(0, 2, 1))
    assert out_on.dtype == torch.bfloat16 and out_on.shape == out_off.shape

    # The LSE never depends on the partial-output dtype. Separate Triton launches
    # may differ by a few fp32 ulps, so test numerical rather than launch-order
    # bit identity.
    torch.testing.assert_close(lse_on, lse_off, rtol=0, atol=2e-6)


# --------------------------------------------------------------------------- #
# 4. env-on closes the seam error vs the fp64 oracle                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("kv", ["bf16", "fp8"])
@pytest.mark.parametrize("case", ["sliding", "csa", "hca"])
def test_env_on_closes_seam_error(flag_guard, case, kv):
    m = flag_guard
    sc = _scenario(case, kv)
    oracle = _oracle(sc)  # [P,1,H,D] fp64 on the pool's effective values

    m._FP32_PARTIAL_MERGE = False
    out_off, _ = _run(sc)
    m._FP32_PARTIAL_MERGE = True
    out_on, _ = _run(sc)

    e_off = _rel_rms(out_off, oracle)
    e_on = _rel_rms(out_on, oracle)
    # the single unavoidable bf16 store's error floor
    floor = _rel_rms(oracle.to(torch.bfloat16), oracle)

    assert torch.isfinite(out_on.float()).all()
    # env-on must sit at the single-store floor (kernel fp32 residual ~2e-7)
    assert e_on <= 1.15 * floor, (case, kv, e_on, floor)
    # and strictly below env-off, by the probe-measured margins:
    # sliding ~29%, csa/hca ~41% seam share of the oracle distance
    min_gain = 0.15 if case == "sliding" else 0.25
    assert e_on <= (1.0 - min_gain) * e_off, (case, kv, e_on, e_off)


# --------------------------------------------------------------------------- #
# 5. edge cases: no NaN/Inf, shapes, dtypes                                   #
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# 4b. oracle-equivalence on the no-extra / no-sink merge branches              #
#     (review gap closure 2026-07-13: these branches were previously tested    #
#     for shape/finiteness only; the no-extra+sink combination was already     #
#     oracle-tested via case="sliding" above)                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "with_extra,case",
    [(True, "csa"), (False, "sliding")],
    ids=["extra-nosink", "noextra-nosink"],
)
def test_env_on_oracle_equivalence_no_sink_branches(flag_guard, with_extra, case):
    m = flag_guard
    sc = _scenario(case, "bf16", seed=17)
    sc["sink"] = None  # exercise the sink-free merge branch
    assert (sc["comp_cache"] is not None) == with_extra
    oracle = _oracle(sc)

    m._FP32_PARTIAL_MERGE = False
    out_off, _ = _run(sc)
    m._FP32_PARTIAL_MERGE = True
    out_on, _ = _run(sc)
    assert torch.isfinite(out_on.float()).all()
    assert out_on.dtype == torch.bfloat16 and out_on.shape == out_off.shape

    e_on = _rel_rms(out_on, oracle)
    e_off = _rel_rms(out_off, oracle)
    floor = _rel_rms(oracle.to(torch.bfloat16), oracle)
    # env-on must sit at the single-bf16-store floor on these branches too
    assert e_on <= 1.15 * floor, (case, with_extra, e_on, floor)
    # and never be worse than env-off; with an extra cache the merge seams
    # exist even sink-free, so demand the probe-scale improvement there
    if with_extra:
        assert e_on <= 0.75 * e_off, (case, e_on, e_off)
    else:
        # single partial, no sink: env-on is one bf16 round of the same fp32
        # accumulator — allow bit-noise-level equality but no regression
        assert e_on <= 1.02 * e_off, (case, e_on, e_off)


@pytest.mark.parametrize("env_on", [False, True])
def test_edge_cases_no_nan(flag_guard, env_on):
    m = flag_guard
    m._FP32_PARTIAL_MERGE = env_on
    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(3)
    latent = torch.randn(256, D, generator=g, device=dev).to(torch.bfloat16)
    cache = _bf16_cache(latent)
    q = (torch.randn(4, 1, H, D, generator=g, device=dev) * 0.5).to(torch.bfloat16)
    sink = torch.randn(H, generator=g, device=dev).float()

    idx = torch.full((4, 1, 64), -1, dtype=torch.int32, device=dev)
    idx[0, 0, :16] = torch.arange(16, dtype=torch.int32, device=dev)
    idx[1, 0, :1] = 0  # single token
    # rows 2,3: zero valid tokens (all -1 / zero length)
    ln = torch.tensor([16, 1, 0, 0], dtype=torch.int32, device=dev)

    eidx = torch.full((4, 1, 64), -1, dtype=torch.int32, device=dev)
    eidx[0, 0, :4] = torch.arange(4, dtype=torch.int32, device=dev)
    eln = torch.tensor([4, 0, 0, 0], dtype=torch.int32, device=dev)

    for extra in (None, cache):
        for s in (None, sink):
            out, lse = m.flash_mla_sparse_decode_triton(
                q,
                cache,
                idx,
                ln,
                s,
                D,
                SM_SCALE,
                extra,
                eidx if extra is not None else None,
                eln if extra is not None else None,
            )
            assert out.shape == (4, 1, H, D) and out.dtype == torch.bfloat16
            assert not torch.isnan(out.float()).any()
            assert not torch.isinf(out.float()).any()
            # zero-valid rows must produce zero output, not garbage
            assert out[2:].abs().sum().item() == 0.0


# --------------------------------------------------------------------------- #
# 6. CUDA-graph capture (production always serves with graphs on)             #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("env_on", [False, True])
def test_cuda_graph_capture(flag_guard, env_on):
    m = flag_guard
    m._FP32_PARTIAL_MERGE = env_on
    sc = _scenario("csa", "bf16", seed=11)

    # static input buffers (as the production capture uses)
    static = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in sc.items()}

    # eager warmup (autotune must not run during capture)
    _ = _run(static)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        g_out, g_lse = _run(static)

    # new inputs -> copy into static buffers -> replay must equal eager
    sc2 = _scenario("csa", "bf16", seed=12)
    for k in ("q", "raw_cache", "raw_idx", "raw_len", "comp_cache", "comp_idx", "comp_len", "sink"):
        static[k].copy_(sc2[k])
    graph.replay()
    torch.cuda.synchronize()
    eager_out, eager_lse = _run(sc2)
    assert torch.equal(g_out, eager_out)
    assert torch.equal(g_lse, eager_lse)
