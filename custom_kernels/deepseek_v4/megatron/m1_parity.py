"""M1 parity: prove the M0 torch ``V4Model`` FORWARD matches HF
``DeepseekV4ForCausalLM`` (eager) on a tiny config, layer by layer.

This is the #1 project-risk gate: V4 numerical alignment of our 3-kernel scaffold
against the HF reference.  We:

  1. Build a tiny HF ``DeepseekV4ForCausalLM`` (eager) and run its ``_init_weights``
     so every param/buffer is sane (routers, experts, indexer, mHC, hc_head,
     sinks=0, position_bias=0, valid random ``tid2eid``).
  2. Build the M0 ``V4Model`` and **copy HF's ``model.*`` state_dict into it** (the
     88 scaffold keys == 88 HF keys, identical shapes — codex-verified; a direct
     ``load_state_dict``).
  3. Run HF in **fp32** (reference truth) and M0 in **bf16** (A1/B2 builders are
     bf16-only) on the *same* ``input_ids``, no cache, contiguous positions; also a
     **HF-bf16** same-dtype run to isolate op divergence from the fp32->bf16 gap.
  4. Capture per-layer intermediates via forward hooks on all three and print a
     per-layer / per-tensor max-rel/cos/p99/worst-token table, asserting each within
     a bf16 tolerance.

Hardening folded from the M1 codex review:
  * **nonzero-perturb variant** — HF ``_init_weights`` zeros ``sinks`` and the
    compressor ``position_bias``; M1 would otherwise exercise those paths only at 0.
    A second pass sets nonzero shared sinks + nonzero compressor/indexer
    position_bias in BOTH HF and M0 (identical values) and re-checks parity.
  * **top-k flip provenance** — a 3-way router-index table (hf_fp32 / hf_bf16 / m0)
    + on the flipped tokens the router-INPUT delta (m0 vs hf_bf16) and the top-k
    margins, proving the flip is pure bf16 rounding (router input matches at floor)
    not a router-input seam bug.
  * **sparse-bug gates** — each non-MoE seam is gated on max_abs / p99 / worst-token
    abs error, not just global rel/cos (a localized 1-token/1-head error passes a
    global metric).
  * **automated negative controls** — zeroing the output conjugate-RoPE ``sin`` AND
    permuting the grouped ``o_a`` rows must both BREAK parity (teeth on >1 seam).
  * **CSA stress** (optional) — ``index_topk < compressed_len`` so HF CSA does a real
    top-k while A1 stays dense; the resulting divergence is the *expected* dense
    approximation, reported but NOT a parity failure.

Run:  CUDA_VISIBLE_DEVICES=0 python -m custom_kernels.deepseek_v4.megatron.m1_parity
"""

import os
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from megatron.decoder import V4Model  # type: ignore
    from megatron.m0_smoke import tiny_config  # type: ignore
else:
    from .decoder import V4Model
    from .m0_smoke import tiny_config

from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4CSACompressor,
    DeepseekV4ForCausalLM,
    DeepseekV4HashRouter,
    DeepseekV4HCACompressor,
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
    DeepseekV4Indexer,
)


# --------------------------------------------------------------------------------------
# error metrics
# --------------------------------------------------------------------------------------
def err(ref: torch.Tensor, got: torch.Tensor):
    """Scale-aware error of ``got`` vs ``ref`` (both cast to fp32).

    Returns a dict with:
      rel       : GLOBAL relative error ``||got-ref|| / ||ref||`` (Frobenius). The
                  right scale-aware metric; an elementwise ``diff/|ref|`` explodes on
                  near-zero entries even when the tensor is well aligned.
      cos       : cosine similarity over the flattened tensors.
      max_abs   : max absolute elementwise error (catches a single hot element).
      p99_abs   : 99th-percentile absolute error (catches a localized cluster that a
                  global norm dilutes).
      worst_tok : max over the leading "token" axis of the per-token L2 error
                  normalised by the per-token ref L2 (catches a 1-token blowup).
      worst_chan: max over the LAST axis of the per-channel abs error (catches a
                  1-head / 1-channel error).
    """
    ref = ref.detach().float()
    got = got.detach().float().reshape(ref.shape)
    diff = got - ref
    abs_d = diff.abs()
    rel = (diff.norm() / ref.norm().clamp_min(1e-12)).item()
    cos = torch.nn.functional.cosine_similarity(ref.reshape(1, -1), got.reshape(1, -1)).item()
    max_abs = abs_d.max().item()
    p99_abs = torch.quantile(abs_d.flatten().float(), 0.99).item()
    # worst-token: flatten everything but the last dim into "rows", L2 per row.
    rows_d = diff.reshape(-1, diff.shape[-1]).norm(dim=-1)
    rows_r = ref.reshape(-1, ref.shape[-1]).norm(dim=-1).clamp_min(1e-12)
    worst_tok = (rows_d / rows_r).max().item()
    # worst-channel: abs error reduced over all but the last dim.
    worst_chan = abs_d.reshape(-1, abs_d.shape[-1]).max(dim=0).values.max().item()
    return dict(rel=rel, cos=cos, max_abs=max_abs, p99_abs=p99_abs, worst_tok=worst_tok, worst_chan=worst_chan)


# --------------------------------------------------------------------------------------
# hook plumbing: capture the per-layer seams on all three models.
#
#   - post-attn-HC stream : ffn_hc INPUT  [B,S,hc_mult,hidden] (== after the attn mix)
#   - post-FFN-HC stream  : decoder layer OUTPUT [B,S,hc_mult,hidden]
#   - compressor out      : compressor [B,1,T,head_dim] (CSA/HCA layers only)
#   - attn out pre-o_proj : o_a_proj INPUT (post conjugate-RoPE, pre output projection)
#   - mlp_out / router_idx / router_in : MoE output, top-k expert indices, router input
# --------------------------------------------------------------------------------------
class Captures:
    def __init__(self):
        self.d = {}

    def put(self, key, t):
        self.d[key] = t.detach().float()


def _restore_hf_fp32_modules(model):
    """Cast exactly the modules M0 keeps in fp32 (mHC sites + hc_head) back to fp32
    after a bf16 cast, so the HF-bf16 reference mirrors the M0 dtype layout EXACTLY.

    Note: M0 keeps the RMSNorms in bf16 (only ``attn_hc``/``ffn_hc``/``hc_head`` are
    restored to fp32 in m0_smoke).  HF's own ``_keep_in_fp32_modules`` additionally
    keeps the norms fp32, but a faithful same-dtype comparison must match what M0
    actually runs — and a pure-eager HF with fp32 norms feeding bf16 linears even
    crashes (fp32-norm output into a bf16 ``q_a_proj``).  So we restore ONLY the mHC
    modules here; the weighted RMSNorms (which already cast their output back to the
    bf16 input dtype) stay bf16 in both models.
    """
    for m in model.modules():
        if isinstance(m, (DeepseekV4HyperConnection, DeepseekV4HyperHead)):
            m.float()


def _register(layers, cap, tag):
    handles = []

    def layer_out_hook(idx):
        def h(mod, inp, out):
            cap.put(f"{tag}.L{idx}.post_ffn_hc", out)

        return h

    def ffn_hc_in_hook(idx):
        def h(mod, inp, out):
            cap.put(f"{tag}.L{idx}.post_attn_hc", inp[0])

        return h

    def o_a_in_hook(idx):
        def h(mod, inp, out):
            cap.put(f"{tag}.L{idx}.attn_pre_oproj", inp[0])

        return h

    def comp_out_hook(idx):
        def h(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            cap.put(f"{tag}.L{idx}.compressor", t)

        return h

    def mlp_out_hook(idx):
        def h(mod, inp, out):
            cap.put(f"{tag}.L{idx}.mlp_out", out)

        return h

    def router_hook(idx):
        def h(mod, inp, out):
            # router input is the post-attn-layernorm collapsed stream [B,S,hidden].
            cap.put(f"{tag}.L{idx}.router_in", inp[0])
            cap.d[f"{tag}.L{idx}.router_idx"] = out[2].detach()
            cap.put(f"{tag}.L{idx}.router_logits", out[0])

        return h

    for idx, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(layer_out_hook(idx)))
        handles.append(layer.ffn_hc.register_forward_hook(ffn_hc_in_hook(idx)))
        handles.append(layer.self_attn.o_a_proj.register_forward_hook(o_a_in_hook(idx)))
        handles.append(layer.mlp.register_forward_hook(mlp_out_hook(idx)))
        handles.append(layer.mlp.gate.register_forward_hook(router_hook(idx)))
        comp = layer.self_attn.compressor
        if comp is not None:
            handles.append(comp.register_forward_hook(comp_out_hook(idx)))
    return handles


# --------------------------------------------------------------------------------------
# model construction with optional perturbations / mutations
# --------------------------------------------------------------------------------------
def _perturb_nonzero(model, cfg, std=0.05):
    """Set nonzero shared sinks + nonzero compressor/indexer position_bias in-place,
    so the sink path and the gated-pool position_bias path are exercised away from 0.

    Uses deterministic per-shape values (a fixed generator) so HF and M0 get the SAME
    perturbation — they are applied by-name independently to each model.
    """
    g = torch.Generator(device="cpu").manual_seed(1234)
    for m in model.modules():
        if isinstance(m, DeepseekV4Attention):
            # shared per-head sink, nonzero (gpt-oss style denom-only logit).
            v = torch.randn(m.sinks.shape, generator=g) * std
            m.sinks.data.copy_(v.to(m.sinks.dtype).to(m.sinks.device))
        elif isinstance(m, (DeepseekV4HCACompressor, DeepseekV4CSACompressor, DeepseekV4Indexer)):
            v = torch.randn(m.position_bias.shape, generator=g) * std
            m.position_bias.data.copy_(v.to(m.position_bias.dtype).to(m.position_bias.device))


def _mutate_oa_permute(model):
    """Negative control: permute the rows of every grouped ``o_a_proj`` weight.  This
    is a structurally valid weight (same shape, no NaN) but wrong math, so parity MUST
    break — proving the harness has teeth on a non-RoPE seam."""
    g = torch.Generator(device="cpu").manual_seed(7)
    for m in model.modules():
        from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4GroupedLinear

        if isinstance(m, DeepseekV4GroupedLinear):
            w = m.weight.data
            perm = torch.randperm(w.shape[0], generator=g).to(w.device)
            m.weight.data.copy_(w[perm])


def build_models(cfg, dev, *, perturb_nonzero=False, mutate=None, zero_rope_sin=False):
    """Build (hf_fp32, hf_bf16, m0) with HF weights copied into M0.

    perturb_nonzero : apply nonzero sinks + position_bias to all three (same values).
    mutate          : callable(model)->None applied to M0 ONLY (negative control).
    zero_rope_sin   : monkeypatch M0 attention's apply_rotary_pos_emb to drop sin
                      (negative control on the RoPE seam).
    """
    hf = DeepseekV4ForCausalLM(cfg)
    for m in hf.modules():
        if isinstance(m, DeepseekV4HashRouter):
            m.tid2eid = torch.randint(0, cfg.n_routed_experts, m.tid2eid.shape, dtype=torch.long)
    if perturb_nonzero:
        _perturb_nonzero(hf, cfg)
    hf_fp32 = hf.to(dev).to(torch.float32).eval()
    hf_state = hf_fp32.model.state_dict()

    hf_bf16 = DeepseekV4ForCausalLM(cfg)
    hf_bf16.load_state_dict(hf_fp32.state_dict(), strict=True)
    hf_bf16 = hf_bf16.to(dev).to(torch.bfloat16).eval()
    _restore_hf_fp32_modules(hf_bf16)

    m0 = V4Model(cfg)
    m0.load_state_dict(hf_state, strict=True)  # nonzero sinks/position_bias carried by state_dict
    m0 = m0.to(dev).to(torch.bfloat16).eval()
    for layer in m0.layers:
        for hc in (layer.attn_hc, layer.ffn_hc):
            hc.float()
    m0.hc_head.float()
    if mutate is not None:
        mutate(m0)
    return hf_fp32, hf_bf16, m0


def run_and_capture(hf_fp32, hf_bf16, m0, input_ids, position_ids, zero_rope_sin=False):
    cap = Captures()
    h = (
        _register(hf_fp32.model.layers, cap, "hf")
        + _register(hf_bf16.model.layers, cap, "hfb")
        + _register(m0.layers, cap, "m0")
    )

    patched = None
    if zero_rope_sin:
        from custom_kernels.deepseek_v4.megatron import attention as A

        orig = A.apply_rotary_pos_emb

        def buggy(x, cos, sin, unsqueeze_dim=1):
            return orig(x, cos, sin * 0.0, unsqueeze_dim)

        A.apply_rotary_pos_emb = buggy
        patched = (A, orig)

    try:
        with torch.no_grad():
            hf_hidden = hf_fp32.model(
                input_ids=input_ids, position_ids=position_ids, use_cache=False
            ).last_hidden_state
            hfb_hidden = hf_bf16.model(
                input_ids=input_ids, position_ids=position_ids, use_cache=False
            ).last_hidden_state
            m0_hidden = m0(input_ids, position_ids=position_ids)
    finally:
        if patched is not None:
            patched[0].apply_rotary_pos_emb = patched[1]
        for hh in h:
            hh.remove()
    return cap, hf_hidden, hfb_hidden, m0_hidden


# tolerances (M0 vs HF-bf16, same dtype) -------------------------------------------------
TOL_REL = 2e-2  # global rel err at the bf16 noise floor over a 3-layer stack
TOL_COS = 0.9995
TOL_COS_MOE = 0.997  # MoE-downstream seams: top-k flips inflate max_abs, cos stays ~1
TOL_MAXABS = 0.30  # per-element max abs error on non-MoE seams (catches a hot elt)
TOL_P99 = 0.06  # 99th pct abs error on non-MoE seams (bf16 floor ~0.05 at L2)
TOL_WORST_TOK = 0.10  # worst single-token relative L2 on non-MoE seams


def report(cfg, cap, hf_hidden, hfb_hidden, m0_hidden, *, label, expect_dense_approx=False):
    """Print the per-seam table + MoE/router diagnostics; return the failure list.

    expect_dense_approx: CSA stress variant — CSA-layer divergence is the expected
    dense approximation, reported but not counted as a failure.
    """
    print("\n" + "=" * 122)
    print(f"[{label}]")
    print(
        f"{'layer / tensor':<22}{'layer/mlp type':<38}"
        f"{'M0-vs-HFfp32':>14}{'M0-vs-HFbf16: rel    cos   maxabs   p99   wTok':>48}"
    )
    print("-" * 122)
    seams = ["compressor", "attn_pre_oproj", "post_attn_hc", "post_ffn_hc"]
    moe_downstream = {"post_ffn_hc"}
    failures = []
    # In CSA-stress mode the dense-approx diverges at the first CSA layer and
    # contaminates the residual stream for every layer at or after it, so waive from
    # there on (not just the CSA layer's own seams).
    csa_layers = [i for i, t in enumerate(cfg.layer_types) if t == "compressed_sparse_attention"]
    first_csa = min(csa_layers) if csa_layers else cfg.num_hidden_layers
    for idx in range(cfg.num_hidden_layers):
        lt, mt = cfg.layer_types[idx], cfg.mlp_layer_types[idx]
        is_csa = idx >= first_csa
        for seam in seams:
            kgot, kref32, krefb = (f"m0.L{idx}.{seam}", f"hf.L{idx}.{seam}", f"hfb.L{idx}.{seam}")
            if kref32 not in cap.d:
                continue
            e32 = err(cap.d[kref32], cap.d[kgot])
            eb = err(cap.d[krefb], cap.d[kgot])
            if seam in moe_downstream:
                ok = eb["cos"] >= TOL_COS_MOE
            else:
                ok = (
                    eb["cos"] >= TOL_COS
                    and eb["rel"] <= TOL_REL
                    and eb["max_abs"] <= TOL_MAXABS
                    and eb["p99_abs"] <= TOL_P99
                    and eb["worst_tok"] <= TOL_WORST_TOK
                )
            waived = expect_dense_approx and is_csa and not ok
            flag = "" if ok else ("  (dense-approx, waived)" if waived else "  <-- FAIL")
            if not ok and not waived:
                failures.append((idx, seam, eb))
            print(
                f"{f'L{idx} {seam}':<22}{lt + '/' + mt:<38}"
                f"{e32['rel']:>7.4f}{e32['cos']:>7.4f}"
                f"{eb['rel']:>9.4f}{eb['cos']:>7.4f}{eb['max_abs']:>8.3f}"
                f"{eb['p99_abs']:>7.3f}{eb['worst_tok']:>7.3f}{flag}"
            )
    e32 = err(hf_hidden, m0_hidden)
    eb = err(hfb_hidden, m0_hidden)
    ok = eb["cos"] >= TOL_COS_MOE
    waived = expect_dense_approx and not ok
    if not ok and not waived:
        failures.append((-1, "final_hidden", eb))
    flag = "" if ok else ("  (dense-approx, waived)" if waived else "  <-- FAIL")
    print("-" * 122)
    print(
        f"{'FINAL hidden':<22}{'':<38}"
        f"{e32['rel']:>7.4f}{e32['cos']:>7.4f}"
        f"{eb['rel']:>9.4f}{eb['cos']:>7.4f}{eb['max_abs']:>8.3f}"
        f"{eb['p99_abs']:>7.3f}{eb['worst_tok']:>7.3f}{flag}"
    )
    print("=" * 122)

    # ---- MoE 3-way router-index table + flip provenance ----
    print(
        "\n[MoE] 3-way router agreement (vs hf_fp32) + flip provenance "
        "(router-INPUT delta on flipped tokens, M0 vs HF-bf16):"
    )
    moe_clean = True
    for idx in range(cfg.num_hidden_layers):
        mt = cfg.mlp_layer_types[idx]
        ih32 = cap.d[f"hf.L{idx}.router_idx"]
        ihb = cap.d[f"hfb.L{idx}.router_idx"]
        im = cap.d[f"m0.L{idx}.router_idx"]
        N = ih32.shape[0]

        def agree(a, b):
            return (a.sort(-1).values == b.sort(-1).values).all(-1)

        a_hfb = agree(ih32, ihb)  # hf_bf16 vs hf_fp32 (the dtype's own flips)
        a_m0 = agree(ih32, im)  # m0 vs hf_fp32
        a_m0b = agree(ihb, im)  # m0 vs hf_bf16 (same dtype)
        flip_m0b = ~a_m0b

        # On the tokens where M0 flips vs HF-bf16, how big is the router-INPUT delta?
        ri_h = cap.d[f"hfb.L{idx}.router_in"].reshape(N, -1)
        ri_m = cap.d[f"m0.L{idx}.router_in"].reshape(N, -1)
        ri_e_all = err(ri_h, ri_m)
        if flip_m0b.any():
            ri_e_flip = err(ri_h[flip_m0b], ri_m[flip_m0b])
            # top-k margin on the flipped tokens: gap between the kth and (k+1)th
            # router score in HF-bf16 (small margin => a near-tie that bf16 can flip).
            lg = cap.d[f"hfb.L{idx}.router_logits"][flip_m0b].float()  # [n_flip, E]
            # score_fn(logits) is monotone-ish; use logits order for the margin proxy.
            top = lg.topk(min(cfg.num_experts_per_tok + 1, lg.shape[-1]), dim=-1).values
            margins = (
                (top[:, cfg.num_experts_per_tok - 1] - top[:, cfg.num_experts_per_tok])
                if top.shape[-1] > cfg.num_experts_per_tok
                else top.new_zeros(top.shape[0])
            )
            flip_in_rel = ri_e_flip["rel"]
            margin_med = margins.median().item()
        else:
            flip_in_rel = 0.0
            margin_med = float("nan")
        flip = int(flip_m0b.sum())
        # matched-router MoE output (must be at bf16 floor; carries accumulated upstream
        # bf16 error in the MoE input, so judged on cos + per-token max_abs).
        mh = cap.d[f"hfb.L{idx}.mlp_out"].reshape(N, -1)
        mm = cap.d[f"m0.L{idx}.mlp_out"].reshape(N, -1)
        em = err(mh[a_m0b], mm[a_m0b]) if a_m0b.any() else dict(cos=float("nan"), max_abs=float("nan"))
        # PASS: flip rate bounded, matched-tok MoE at floor, AND the router-input on the
        # flipped tokens matches at the bf16 floor (=> flip is rounding, not a seam bug).
        ok = flip <= max(2, int(0.05 * N)) and em["cos"] >= TOL_COS and em["max_abs"] < 0.2 and flip_in_rel <= TOL_REL
        # in CSA-stress mode the dense-approx perturbs the MoE input, flipping more
        # top-k tokens — that's the expected approximation, not a parity failure.
        if expect_dense_approx:
            ok = True
        moe_clean = moe_clean and ok
        flag = "" if ok else "  <-- FAIL"
        print(
            f"  L{idx} {mt:<9} flips: m0-vs-fp32={int((~a_m0).sum()):>3} "
            f"hfbf16-vs-fp32={int((~a_hfb).sum()):>3} m0-vs-hfbf16={flip:>3}/{N} | "
            f"router-in rel(all)={ri_e_all['rel']:.4f} rel(flipped)={flip_in_rel:.4f} "
            f"margin_med={margin_med:.4f} | matched MoE cos={em['cos']:.5f} "
            f"maxabs={em['max_abs']:.4f}{flag}"
        )
    if not moe_clean:
        failures.append((-2, "moe_router_boundary", None))

    e_dtype = err(hf_hidden, hfb_hidden)
    print(
        f"\n[dtype gap] HF-bf16 vs HF-fp32 final hidden: rel={e_dtype['rel']:.5f} "
        f"cos={e_dtype['cos']:.6f} (reference floor for the M0-vs-HFfp32 column)"
    )
    return failures


# --------------------------------------------------------------------------------------
def main():
    assert torch.cuda.is_available(), "M1 parity needs a GPU (CUDA_VISIBLE_DEVICES=0)"
    dev = torch.device("cuda")
    torch.manual_seed(0)

    cfg = tiny_config()
    assert cfg.hidden_size == 4096 and cfg.head_dim == 512 and cfg.hc_mult == 4
    assert cfg.hc_sinkhorn_iters == 20
    assert abs(cfg.hc_eps - 1e-6) < 1e-12 and abs(cfg.rms_norm_eps - 1e-6) < 1e-12

    B, S = 2, 128
    csa_entries = S // cfg.compress_rates["compressed_sparse_attention"]
    hca_entries = S // cfg.compress_rates["heavily_compressed_attention"]
    assert csa_entries <= cfg.index_topk and hca_entries <= cfg.index_topk
    print(
        f"[cfg] B={B} S={S} layers={cfg.num_hidden_layers} heads={cfg.num_attention_heads} "
        f"experts={cfg.n_routed_experts} hidden={cfg.hidden_size} hd={cfg.head_dim}"
    )
    print(f"[cfg] layer_types={cfg.layer_types}  mlp_layer_types={cfg.mlp_layer_types}")
    print(
        f"[cfg] CSA entries={csa_entries} HCA entries={hca_entries} index_topk={cfg.index_topk} "
        f"(tiny-seq CSA top-k is a NO-OP -> HF CSA == dense)"
    )

    input_ids = torch.randint(0, cfg.vocab_size, (B, S), device=dev)
    position_ids = torch.arange(S, device=dev).unsqueeze(0).expand(B, -1)

    all_failures = {}

    # ---- (1) baseline: HF _init_weights (sinks=0, position_bias=0) ----
    hf32, hfb, m0 = build_models(cfg, dev)
    cap, hh, hbh, mh = run_and_capture(hf32, hfb, m0, input_ids, position_ids)
    all_failures["baseline (sinks=0, pos_bias=0)"] = report(
        cfg, cap, hh, hbh, mh, label="baseline (HF _init_weights: sinks=0, position_bias=0)"
    )

    # ---- (2) nonzero perturbation: shared sinks + compressor/indexer position_bias ----
    hf32, hfb, m0 = build_models(cfg, dev, perturb_nonzero=True)
    # sanity: the perturbation actually took (nonzero in both HF and M0).
    sink_nz = any(l.self_attn.sinks.abs().sum() > 0 for l in m0.layers)
    pb_nz = any(
        getattr(l.self_attn.compressor, "position_bias", torch.zeros(1)).abs().sum() > 0
        for l in m0.layers
        if l.self_attn.compressor is not None
    )
    assert sink_nz and pb_nz, "nonzero perturbation did not take"
    cap, hh, hbh, mh = run_and_capture(hf32, hfb, m0, input_ids, position_ids)
    all_failures["nonzero (sinks + position_bias)"] = report(
        cfg, cap, hh, hbh, mh, label="NONZERO sinks + compressor/indexer position_bias"
    )

    # ---- (3a) negative control: zero the output conjugate-RoPE sin (must FAIL) ----
    hf32, hfb, m0 = build_models(cfg, dev)
    cap, hh, hbh, mh = run_and_capture(hf32, hfb, m0, input_ids, position_ids, zero_rope_sin=True)
    neg_rope = report(cfg, cap, hh, hbh, mh, label="NEG-CONTROL: zero RoPE sin (MUST break)")

    # ---- (3b) negative control: permute grouped o_a rows (must FAIL) ----
    hf32, hfb, m0 = build_models(cfg, dev, mutate=_mutate_oa_permute)
    cap, hh, hbh, mh = run_and_capture(hf32, hfb, m0, input_ids, position_ids)
    neg_oa = report(cfg, cap, hh, hbh, mh, label="NEG-CONTROL: permute grouped o_a rows (MUST break)")

    # ---- (4) CSA stress: index_topk < compressed_len -> HF does a real top-k ----
    cfg_stress = tiny_config()
    cfg_stress.index_topk = 4  # << csa_entries=32 -> HF CSA gathers only 4 entries/query
    hf32, hfb, m0 = build_models(cfg_stress, dev)
    cap, hh, hbh, mh = run_and_capture(hf32, hfb, m0, input_ids, position_ids)
    report(
        cfg_stress,
        cap,
        hh,
        hbh,
        mh,
        label="CSA STRESS: index_topk=4 < 32 (expected dense-approx divergence, NOT a fail)",
        expect_dense_approx=True,
    )

    # ---- verdict ----
    print("\n" + "#" * 122)
    ok = True
    for name, fails in all_failures.items():
        status = "PASS" if not fails else f"FAIL ({len(fails)} seam(s))"
        print(f"  parity[{name}]: {status}")
        if fails:
            ok = False
            for idx, seam, e in fails:
                if e is None:
                    print(f"      L{idx} {seam}")
                else:
                    print(
                        f"      L{idx} {seam}: rel={e['rel']:.4f} cos={e['cos']:.5f} "
                        f"max_abs={e['max_abs']:.3f} p99={e['p99_abs']:.3f} wTok={e['worst_tok']:.3f}"
                    )
    # negative controls MUST have produced failures.
    rope_broke = len(neg_rope) > 0
    oa_broke = len(neg_oa) > 0
    print(
        f"  neg-control[zero RoPE sin]:   {'BROKE parity (good)' if rope_broke else 'DID NOT BREAK -- harness has no teeth!'}"
    )
    print(
        f"  neg-control[permute o_a]:     {'BROKE parity (good)' if oa_broke else 'DID NOT BREAK -- harness has no teeth!'}"
    )
    if not (rope_broke and oa_broke):
        ok = False
    print("#" * 122)

    print(f"\nlayer types exercised: {set(cfg.layer_types)}  routers: {set(cfg.mlp_layer_types)}")
    if not ok:
        print("\nM1 PARITY FAIL.")
        sys.exit(1)
    print(
        "\nM1 PARITY PASS (hardened): baseline + nonzero-sinks/position_bias both match "
        "HF eager at the bf16 op-floor (rel/cos/max_abs/p99/worst-token gated) across all "
        "layer types + routers; the L2 top-k flip is pure bf16 rounding (router input "
        "matches at floor); both negative controls break parity; CSA-stress divergence is "
        "the expected dense approximation."
    )


if __name__ == "__main__":
    main()
