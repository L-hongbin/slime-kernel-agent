"""M-impl parity: prove the **mcore V4 LanguageModel** FORWARD matches HF
``DeepseekV4ForCausalLM`` (eager) at TP=PP=EP=1, on the tiny config, layer by layer.

This is the milestone gate: the mcore model (custom ``LanguageModule`` + custom
``V4MoELayer`` with input_ids-threaded V4 routers + exact clamp+SwiGLU experts) must
reproduce HF to the bf16 floor on every non-MoE seam, with the same top-k-flip /
negative-control / nonzero-perturb structure the M0 harness (``m1_parity.py``) used.

Differences vs ``m1_parity.py``:
  * the "got" model is the mcore ``V4LanguageModel`` (not the M0 ``V4Model``);
  * HF weights are copied in via the HF->mcore key remap (``load_hf_into_mcore``),
    NOT a bare ``load_state_dict`` (the embedding/output_layer/lm_head are renamed);
  * the model returns fp32 ``[B,S,V]`` logits, so the FINAL seam is compared on the
    pre-lm_head hidden state (captured via a hook on ``norm``) to keep the seam table
    identical to M1 (the M1 reference is HF's ``model.last_hidden_state``).
  * hooks bind to ``layer.mlp.gate`` (the V4 router) which returns ``(logits, weights,
    indices)`` exactly like HF, so the router diagnostics are unchanged.

Run:  CUDA_VISIBLE_DEVICES=0 python -m custom_kernels.deepseek_v4.megatron.m_impl_parity
"""

import os
import sys

import torch

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from megatron import m1_parity as M1  # type: ignore
    from megatron.m0_smoke import tiny_config  # type: ignore
    from megatron.model_provider import build_v4_mcore_model  # type: ignore
else:
    from .m0_smoke import tiny_config
    from .model_provider import build_v4_mcore_model
    from . import m1_parity as M1

from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4Attention,
    DeepseekV4CSACompressor,
    DeepseekV4ForCausalLM,
    DeepseekV4GroupedLinear,
    DeepseekV4HCACompressor,
    DeepseekV4Indexer,
)


# --------------------------------------------------------------------------------------
# 1-rank Megatron init (the mcore LanguageModule needs parallel_state + the MP rng).
# --------------------------------------------------------------------------------------
def init_dist_1rank():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29580")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend="nccl", world_size=1, rank=0)
    from megatron.core import parallel_state, tensor_parallel

    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
        )
    tensor_parallel.model_parallel_cuda_manual_seed(0)


# --------------------------------------------------------------------------------------
# HF -> mcore weight remap (the key map; identity except 2 renames + strip "model.").
# Verified: 0 shape mismatches, 0 unmapped HF params, only mcore extra = output_layer
# ._extra_state (TE bookkeeping, skipped).  See M_IMPL_NOTES.md key-map table.
# --------------------------------------------------------------------------------------
def hf_key_to_mcore(k: str):
    if k == "model.embed_tokens.weight":
        return "embedding.word_embeddings.weight"
    if k == "lm_head.weight":
        return "output_layer.weight"
    if k == "model.norm.weight":
        return "norm.weight"
    if k.startswith("model.hc_head."):
        return k[len("model.") :]
    if k.startswith("model.layers."):
        return k[len("model.") :]
    if k.startswith("model.rotary_emb"):
        return None  # rebuilt by mcore model's own DeepseekV4RotaryEmbedding
    raise KeyError(f"unmapped HF key: {k}")


def load_hf_into_mcore(mcore_model, hf_full_state):
    """Copy a full HF ``DeepseekV4ForCausalLM`` state_dict into the mcore model via the
    remap. strict on the mapped subset; asserts every mcore param/buffer is filled."""
    remapped = {}
    for k, v in hf_full_state.items():
        tgt = hf_key_to_mcore(k)
        if tgt is None:
            continue
        remapped[tgt] = v
    msd = mcore_model.state_dict()
    # every mcore key (except _extra_state) must be covered.
    missing = [k for k in msd if k not in remapped and not k.endswith("_extra_state")]
    assert not missing, f"mcore keys not covered by HF remap: {missing}"
    # load (allow _extra_state to stay None).
    incompatible = mcore_model.load_state_dict(remapped, strict=False)
    unexpected = [k for k in incompatible.unexpected_keys]
    real_missing = [k for k in incompatible.missing_keys if not k.endswith("_extra_state")]
    assert not unexpected, f"unexpected remap keys: {unexpected}"
    assert not real_missing, f"missing after remap: {real_missing}"


# --------------------------------------------------------------------------------------
# mutations / perturbations on the mcore model (V4-module-typed; mirror M1's HF-typed).
# --------------------------------------------------------------------------------------
def _perturb_nonzero_mcore(model, std=0.05):
    g = torch.Generator(device="cpu").manual_seed(1234)
    for m in model.modules():
        if isinstance(m, DeepseekV4Attention) or m.__class__.__name__ == "V4Attention":
            v = torch.randn(m.sinks.shape, generator=g) * std
            m.sinks.data.copy_(v.to(m.sinks.dtype).to(m.sinks.device))
        elif isinstance(m, (DeepseekV4HCACompressor, DeepseekV4CSACompressor, DeepseekV4Indexer)) or (
            hasattr(m, "position_bias") and m.__class__.__name__ in ("V4HCACompressor", "V4CSACompressor")
        ):
            v = torch.randn(m.position_bias.shape, generator=g) * std
            m.position_bias.data.copy_(v.to(m.position_bias.dtype).to(m.position_bias.device))


def _mutate_oa_permute_mcore(model):
    g = torch.Generator(device="cpu").manual_seed(7)
    for m in model.modules():
        if isinstance(m, DeepseekV4GroupedLinear):
            w = m.weight.data
            perm = torch.randperm(w.shape[0], generator=g).to(w.device)
            m.weight.data.copy_(w[perm])


def _mutate_experts_permute_mcore(model):
    """Negative control on the NEW MoE compute (V4GroupedExperts): permute each
    expert's down_proj OUTPUT rows.  Structurally valid (same shape, no NaN) but wrong
    per-expert math, so the MoE output -> post_ffn_hc -> FINAL seam MUST break.  The
    M1 negative controls only touched REUSED M0 code (RoPE / o_a); this gives the
    harness teeth on the routed-expert path itself."""
    from .mcore_model import V4GroupedExperts

    g = torch.Generator(device="cpu").manual_seed(11)
    for m in model.modules():
        if isinstance(m, V4GroupedExperts):
            w = m.down_proj.data  # [E, H, I]
            perm = torch.randperm(w.shape[1], generator=g).to(w.device)
            m.down_proj.data.copy_(w[:, perm, :])


def _mutate_router_permute_mcore(model):
    """Negative control on the NEW router compute (V4TopKRouter/V4HashRouter): permute
    the router gate weight rows (== relabel which expert each score belongs to).  This
    changes which experts win the top-k (TopK) / which scores weight the hash-selected
    experts, so the MoE output MUST diverge from HF — proving the router scoring path
    is actually exercised, not silently bypassed."""
    from .mcore_model import V4HashRouter, V4TopKRouter

    g = torch.Generator(device="cpu").manual_seed(13)
    for m in model.modules():
        if isinstance(m, (V4TopKRouter, V4HashRouter)):
            w = m.weight.data  # [E, H]
            perm = torch.randperm(w.shape[0], generator=g).to(w.device)
            m.weight.data.copy_(w[perm])


# ---- hardening (mirror M1's nonzero-perturb, but on the NEW MoE math) ----------------
# correction-bias std: sqrtsoftplus scores have mean ~0.9, so std=0.02 is a realistic
# small nudge that still changes the top-k for ~23% of tokens vs zero-bias (exercises
# the bias-add path) WITHOUT creating a degenerate near-tie flip-storm (std=0.5 flips
# 123/256 tokens at the bf16 boundary — a test artifact, not a model bug).
_CORRECTION_BIAS_STD = 0.02
# clamp-stress: lower swiglu_limit (instead of scaling weights) so the clamp bites a
# large fraction (~0.44 at limit=1.0) at NORMAL activation scale — keeps the MoE output
# normal-magnitude so the bf16 relative floor still applies (scaling gate_up x6 instead
# blows the output scale x6 and trips the relative gates on dynamic range, not on clamp
# correctness — the disable-clamp neg-control proves the clamp itself is exercised).
_CLAMP_STRESS_LIMIT = 1.0


def clamp_stress_cfg():
    """A tiny_config clone with a small swiglu_limit so the expert clamp is dominant."""
    c = tiny_config()
    c.swiglu_limit = _CLAMP_STRESS_LIMIT
    return c


def _perturb_correction_bias(hf, cfg, std=_CORRECTION_BIAS_STD):
    """Pre-snapshot: set HF TopKRouter ``e_score_correction_bias`` NONZERO (so it flows
    into mcore identically).  The bias is added to the scores before the top-k argmax,
    so a nonzero bias CHANGES which experts are selected — this exercises the bias path
    away from 0 (HF ``_init_weights`` zeros it).  Both models must still match."""
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4TopKRouter

    g = torch.Generator(device="cpu").manual_seed(2024)
    for m in hf.modules():
        if isinstance(m, DeepseekV4TopKRouter):
            v = torch.randn(m.e_score_correction_bias.shape, generator=g) * std
            m.e_score_correction_bias.data.copy_(
                v.to(m.e_score_correction_bias.dtype).to(m.e_score_correction_bias.device)
            )


def _perturb_correction_bias_strong(hf, cfg):
    """Pre-snapshot bias for the zero-bias NEG-CONTROL only: a LARGE bias (std=0.5 ≈
    half the ~0.9 sqrtsoftplus score scale) so that zeroing it in mcore flips a large,
    deterministic fraction of the top-k -> the control reliably BREAKS parity.  This is
    NOT a must-pass variant (the realistic std=0.02 bias is the must-pass one); a big
    bias here just guarantees the control has teeth regardless of the random draw."""
    _perturb_correction_bias(hf, cfg, std=0.5)


def _mutate_zero_bias_mcore(model):
    """Neg-control: zero the mcore TopK router ``e_score_correction_bias`` (while HF
    keeps it nonzero from the pre-snapshot perturb).  The top-k selection then differs
    from HF -> MoE output MUST break.  Teeth on the bias-add path specifically."""
    from .mcore_model import V4TopKRouter

    for m in model.modules():
        if isinstance(m, V4TopKRouter):
            m.e_score_correction_bias.data.zero_()


def _mutate_disable_clamp_mcore(model):
    """Neg-control: disable the mcore expert clamp (limit -> +inf) while HF keeps
    swiglu_limit.  Under clamp-stress (saturated activations) the expert output MUST
    diverge from HF -> breaks parity.  Teeth on the clamp(swiglu_limit) path."""
    from .mcore_model import V4GroupedExperts

    for m in model.modules():
        if isinstance(m, V4GroupedExperts):
            m.limit = float("inf")


# --------------------------------------------------------------------------------------
# build (hf_fp32, hf_bf16, mcore) with HF weights copied into mcore.
# --------------------------------------------------------------------------------------
def build_models(cfg, dev, *, perturb_nonzero=False, pre_snapshot_perturb=None, mutate=None, lora=False):
    """pre_snapshot_perturb(hf, cfg): applied to HF BEFORE the state snapshot, so it
    flows into the mcore model via the weight copy (both models get identical values —
    use for "must still PASS" variants like nonzero correction-bias / clamp-stress).
    mutate(mcore): applied to the mcore model ONLY after load (negative controls)."""
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HashRouter

    hf = DeepseekV4ForCausalLM(cfg)
    for m in hf.modules():
        if isinstance(m, DeepseekV4HashRouter):
            m.tid2eid = torch.randint(0, cfg.n_routed_experts, m.tid2eid.shape, dtype=torch.long)
    if perturb_nonzero:
        M1._perturb_nonzero(hf, cfg)
    if pre_snapshot_perturb is not None:
        pre_snapshot_perturb(hf, cfg)
    hf_fp32 = hf.to(dev).to(torch.float32).eval()
    hf_full_state = hf_fp32.state_dict()

    hf_bf16 = DeepseekV4ForCausalLM(cfg)
    hf_bf16.load_state_dict(hf_full_state, strict=True)
    hf_bf16 = hf_bf16.to(dev).to(torch.bfloat16).eval()
    M1._restore_hf_fp32_modules(hf_bf16)
    # HF keeps e_score_correction_bias fp32 (_keep_in_fp32_modules_strict); restore it
    # so the same-dtype HF-bf16 reference matches the mcore fp32-bias layout (matters
    # for the nonzero-correction-bias parity pass — a bf16 bias shifts the top-k).
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4TopKRouter

    for m in hf_bf16.modules():
        if isinstance(m, DeepseekV4TopKRouter):
            m.e_score_correction_bias.data = m.e_score_correction_bias.data.float()

    # init_weights=False: load_hf_into_mcore copies the EXACT HF weights right after, so the
    # provider's default random init would be wasted (and must not perturb the copy).
    mc = build_v4_mcore_model(cfg, params_dtype=torch.bfloat16, init_weights=False)
    load_hf_into_mcore(mc, hf_full_state)  # carries nonzero sinks/pos_bias/tid2eid
    if lora:
        # Apply LoRA on the loaded model (zero-init linear_out => identity at init, so
        # the LoRA forward must still match HF — proves the wrapped linears didn't shift
        # the math).  Applied before the bf16 cast so the adapters cast with the model.
        from .lora import apply_v4_lora

        mc = apply_v4_lora(mc, dim=16, alpha=32, dropout=0.0)
    mc = mc.to(dev).to(torch.bfloat16).eval()
    # restore the fp32-kept modules (mHC + hc_head + top-k correction bias) — exactly
    # what the model's own .bfloat16() override would do under slime's Float16Module.
    mc.restore_fp32_modules()
    if mutate is not None:
        mutate(mc)
    return hf_fp32, hf_bf16, mc


# --------------------------------------------------------------------------------------
# run + capture.  The mcore model returns fp32 [B,S,V] logits; the M1 seam table
# compares the pre-lm_head hidden state, so we hook ``norm`` (its OUTPUT == the
# last_hidden_state HF returns) and compare that as the FINAL seam.
# --------------------------------------------------------------------------------------
def run_and_capture(hf_fp32, hf_bf16, mc, input_ids, position_ids, zero_rope_sin=False):
    cap = M1.Captures()
    h = (
        M1._register(hf_fp32.model.layers, cap, "hf")
        + M1._register(hf_bf16.model.layers, cap, "hfb")
        + M1._register(mc.layers, cap, "m0")  # tag "m0" so M1.report reads m0.* keys
    )

    # capture the mcore pre-lm_head hidden (norm output) as the FINAL comparison tensor.
    mc_final = {}

    def norm_hook(mod, inp, out):
        mc_final["h"] = out.detach().float()

    h.append(mc.norm.register_forward_hook(norm_hook))

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
            _ = mc(input_ids, position_ids=position_ids)  # fills mc_final via the norm hook
    finally:
        if patched is not None:
            patched[0].apply_rotary_pos_emb = patched[1]
        for hh in h:
            hh.remove()
    return cap, hf_hidden, hfb_hidden, mc_final["h"]


# --------------------------------------------------------------------------------------
def main():
    assert torch.cuda.is_available(), "M-impl parity needs a GPU (CUDA_VISIBLE_DEVICES=0)"
    init_dist_1rank()
    dev = torch.device("cuda")
    torch.cuda.set_device(0)
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
    print("[model] mcore V4 LanguageModel @ TP=PP=EP=1  vs  HF DeepseekV4ForCausalLM")

    input_ids = torch.randint(0, cfg.vocab_size, (B, S), device=dev)
    position_ids = torch.arange(S, device=dev).unsqueeze(0).expand(B, -1)

    all_failures = {}

    # (1) baseline.
    hf32, hfb, mc = build_models(cfg, dev)
    cap, hh, hbh, mh = run_and_capture(hf32, hfb, mc, input_ids, position_ids)
    all_failures["baseline (sinks=0, pos_bias=0)"] = M1.report(
        cfg, cap, hh, hbh, mh, label="baseline (HF _init_weights: sinks=0, position_bias=0)"
    )

    # (2) nonzero perturbation.
    hf32, hfb, mc = build_models(cfg, dev, perturb_nonzero=True)
    sink_nz = any(l.self_attn.sinks.abs().sum() > 0 for l in mc.layers)
    pb_nz = any(
        getattr(l.self_attn.compressor, "position_bias", torch.zeros(1)).abs().sum() > 0
        for l in mc.layers
        if l.self_attn.compressor is not None
    )
    assert sink_nz and pb_nz, "nonzero perturbation did not take on the mcore model"
    cap, hh, hbh, mh = run_and_capture(hf32, hfb, mc, input_ids, position_ids)
    all_failures["nonzero (sinks + position_bias)"] = M1.report(
        cfg, cap, hh, hbh, mh, label="NONZERO sinks + compressor/indexer position_bias"
    )

    # (2b) NEW-MoE hardening: nonzero e_score_correction_bias (changes top-k ordering;
    # both models get the same bias -> must still PASS).
    hf32, hfb, mc = build_models(cfg, dev, pre_snapshot_perturb=_perturb_correction_bias)
    bias_nz = any(getattr(l.mlp.gate, "e_score_correction_bias", torch.zeros(1)).abs().sum() > 0 for l in mc.layers)
    assert bias_nz, "nonzero correction-bias did not take on the mcore model"
    cap, hh, hbh, mh = run_and_capture(hf32, hfb, mc, input_ids, position_ids)
    all_failures["nonzero e_score_correction_bias"] = M1.report(
        cfg, cap, hh, hbh, mh, label="NONZERO e_score_correction_bias (top-k bias path)"
    )

    # (2c) NEW-MoE hardening: clamp-stress via a small swiglu_limit (clamp bites ~44%
    # of activations at normal scale; both models clamp identically -> must still PASS).
    cfg_cs = clamp_stress_cfg()
    hf32, hfb, mc = build_models(cfg_cs, dev)
    assert all(l.mlp.experts.limit == _CLAMP_STRESS_LIMIT for l in mc.layers), "clamp limit not set"
    cap, hh, hbh, mh = run_and_capture(hf32, hfb, mc, input_ids, position_ids)
    all_failures["clamp-stress (small swiglu_limit)"] = M1.report(
        cfg_cs, cap, hh, hbh, mh, label="CLAMP-STRESS: swiglu_limit=1.0 (clamp dominant)"
    )

    # (2d) LoRA applied (zero-init => identity): the LoRA-wrapped V4 attention/compressor
    # linears must STILL match HF (proves the LinearAdapter wrap preserves the math and
    # didn't accidentally wrap o_a / change shapes).  Must PASS.
    hf32, hfb, mc = build_models(cfg, dev, lora=True)
    cap, hh, hbh, mh = run_and_capture(hf32, hfb, mc, input_ids, position_ids)
    all_failures["LoRA applied (zero-init identity)"] = M1.report(
        cfg, cap, hh, hbh, mh, label="LoRA APPLIED (zero-init linear_out => identity)"
    )

    # (3a) neg-control: zero RoPE sin (must FAIL).
    hf32, hfb, mc = build_models(cfg, dev)
    cap, hh, hbh, mh = run_and_capture(hf32, hfb, mc, input_ids, position_ids, zero_rope_sin=True)
    neg_rope = M1.report(cfg, cap, hh, hbh, mh, label="NEG-CONTROL: zero RoPE sin (MUST break)")

    # (3b) neg-control: permute grouped o_a rows (must FAIL).
    hf32, hfb, mc = build_models(cfg, dev, mutate=_mutate_oa_permute_mcore)
    cap, hh, hbh, mh = run_and_capture(hf32, hfb, mc, input_ids, position_ids)
    neg_oa = M1.report(cfg, cap, hh, hbh, mh, label="NEG-CONTROL: permute grouped o_a rows (MUST break)")

    # (3c) neg-control on the NEW MoE EXPERT compute: permute each expert's down_proj
    # output rows (must FAIL — teeth on V4GroupedExperts, which M1 never exercised).
    hf32, hfb, mc = build_models(cfg, dev, mutate=_mutate_experts_permute_mcore)
    cap, hh, hbh, mh = run_and_capture(hf32, hfb, mc, input_ids, position_ids)
    neg_experts = M1.report(cfg, cap, hh, hbh, mh, label="NEG-CONTROL: permute expert down_proj rows (MUST break)")

    # (3d) neg-control on the NEW ROUTER compute: permute router gate weight rows
    # (must FAIL — teeth on V4TopKRouter/V4HashRouter scoring, which M1 never exercised).
    hf32, hfb, mc = build_models(cfg, dev, mutate=_mutate_router_permute_mcore)
    cap, hh, hbh, mh = run_and_capture(hf32, hfb, mc, input_ids, position_ids)
    neg_router = M1.report(cfg, cap, hh, hbh, mh, label="NEG-CONTROL: permute router gate rows (MUST break)")

    # (3e) neg-control on the BIAS path: HF keeps a (large) nonzero correction-bias,
    # mcore zeros it -> top-k differs -> MUST break (proves the bias-add is exercised).
    # Uses a strong bias so the control is deterministic, not a flaky near-boundary flip.
    hf32, hfb, mc = build_models(
        cfg, dev, pre_snapshot_perturb=_perturb_correction_bias_strong, mutate=_mutate_zero_bias_mcore
    )
    cap, hh, hbh, mh = run_and_capture(hf32, hfb, mc, input_ids, position_ids)
    neg_bias = M1.report(cfg, cap, hh, hbh, mh, label="NEG-CONTROL: zero mcore correction-bias (MUST break)")

    # (3f) neg-control on the CLAMP path: under clamp-stress (small limit), mcore
    # disables the clamp (limit->inf) while HF clamps -> expert output differs -> break.
    hf32, hfb, mc = build_models(clamp_stress_cfg(), dev, mutate=_mutate_disable_clamp_mcore)
    cap, hh, hbh, mh = run_and_capture(hf32, hfb, mc, input_ids, position_ids)
    neg_clamp = M1.report(
        clamp_stress_cfg(), cap, hh, hbh, mh, label="NEG-CONTROL: disable mcore swiglu clamp (MUST break)"
    )

    # verdict.
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
    neg_controls = [
        ("zero RoPE sin", len(neg_rope) > 0),
        ("permute o_a", len(neg_oa) > 0),
        ("permute expert down_proj", len(neg_experts) > 0),
        ("permute router gate", len(neg_router) > 0),
        ("zero correction-bias", len(neg_bias) > 0),
        ("disable swiglu clamp", len(neg_clamp) > 0),
    ]
    for tag, broke in neg_controls:
        print(
            f"  neg-control[{tag}]:".ljust(42) + f"{'BROKE parity (good)' if broke else 'DID NOT BREAK -- no teeth!'}"
        )
    if not all(broke for _, broke in neg_controls):
        ok = False
    print("#" * 122)

    if not ok:
        print("\nM-IMPL PARITY FAIL.")
        sys.exit(1)
    print(
        "\nM-IMPL PARITY PASS: the mcore V4 LanguageModel (TP=PP=EP=1) matches HF eager at the "
        "bf16 op-floor across FIVE must-pass variants (baseline; nonzero sinks/position_bias; "
        "nonzero e_score_correction_bias; clamp-stress saturated SwiGLU; LoRA-applied zero-init "
        "identity), the top-k flip is pure bf16 rounding, and all SIX negative controls break "
        "parity (RoPE + o_a on reused M0 seams; expert down_proj + router gate + zero-correction-"
        "bias + disable-clamp on the NEW V4 MoE compute). HF->mcore remap is identity."
    )


if __name__ == "__main__":
    main()
