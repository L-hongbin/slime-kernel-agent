"""Tests for the DeepSeek-V4-Flash MTP (multi-token-prediction) head.

Covers three behaviours the train-side MTP gate depends on:
  1. a shape-level forward of ``V4MultiTokenPredictionLayer`` on a tiny config
     (torch-reference attention/mHC so it runs on CPU), including the roll;
  2. the ``mtp.*`` -> mcore converter key mapping (and that it is still ignored
     when the gate is off), plus that every mapped key exists on the built module;
  3. that the MTP router is EXCLUDED from the rollout routing replay (it routes
     live: no ``RoutingReplay`` is constructed, no ``routing_replay`` attr).
"""

import importlib.util
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_native_checkpoint_module():
    module_path = REPO_ROOT / "custom_kernels/deepseek_v4/megatron/native_checkpoint.py"
    spec = importlib.util.spec_from_file_location("test_dsv4_mtp_native_checkpoint", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _tiny_v4_config():
    # hidden_size/head_dim/hc_mult are structural to the A1/B1 kernels (see m0_smoke);
    # everything else is shrunk.  The MTP head builds a single decoder layer (idx 0).
    from transformers.models.deepseek_v4 import DeepseekV4Config

    return DeepseekV4Config(
        hidden_size=4096,
        head_dim=512,
        hc_mult=4,
        hc_sinkhorn_iters=20,
        q_lora_rank=1024,
        o_groups=8,
        o_lora_rank=1024,
        sliding_window=128,
        vocab_size=512,
        num_hidden_layers=3,
        num_attention_heads=8,
        num_key_value_heads=1,
        moe_intermediate_size=256,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        index_n_heads=4,
        index_head_dim=64,
        index_topk=512,
        layer_types=["sliding_attention", "compressed_sparse_attention", "heavily_compressed_attention"],
        mlp_layer_types=["hash_moe", "hash_moe", "moe"],
    )


def _use_torch_reference_kernels(monkeypatch):
    """Point the A1 attention + B1 mHC module globals at their torch references so the
    forward runs on CPU regardless of prior kernel-bound imports (the globals are
    looked up at call time inside V4Attention / V4HyperConnection)."""
    import custom_kernels.deepseek_v4.megatron.attention as attention
    import custom_kernels.deepseek_v4.megatron.decoder as decoder
    from custom_kernels.deepseek_v4.attention.reference import attention_reference
    from custom_kernels.deepseek_v4.mhc.reference import hyper_connection_forward

    def _torch_attn(q, k_raw, k_comp, sinks, window, m, comp_topk_mask=None):
        return attention_reference(q, k_raw, k_comp, sinks, window, m, comp_topk_mask=comp_topk_mask).to(q.dtype)

    monkeypatch.setattr(attention, "_v4flash_attention", _torch_attn)
    monkeypatch.setattr(decoder, "_HC", hyper_connection_forward)


def test_dsv4_mtp_layer_forward_shapes_and_roll(monkeypatch):
    _use_torch_reference_kernels(monkeypatch)
    from custom_kernels.deepseek_v4.megatron.model_provider import init_v4_module_weights
    from custom_kernels.deepseek_v4.megatron.mtp import V4MultiTokenPredictionLayer
    from custom_kernels.deepseek_v4.megatron.rope import DeepseekV4RotaryEmbedding

    cfg = _tiny_v4_config()
    torch.manual_seed(0)
    mtp = V4MultiTokenPredictionLayer(cfg)
    init_v4_module_weights(mtp, cfg)

    # Composed correctly: no-compressor sliding attention + learned (non-hash) MoE.
    assert mtp.transformer_layer.self_attn.layer_type == "sliding_attention"
    assert mtp.transformer_layer.self_attn.compressor is None
    assert mtp.transformer_layer.mlp.is_hash is False

    B, S, H = 1, 4, cfg.hidden_size
    emb = torch.nn.Embedding(cfg.vocab_size, H)
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    rotary = DeepseekV4RotaryEmbedding(cfg)
    x = torch.randn(B, S, H)
    position_embeddings = {
        "main": rotary(x, position_ids=position_ids, layer_type="main"),
        "compress": rotary(x, position_ids=position_ids, layer_type="compress"),
    }
    hidden_states = torch.randn(B, S, cfg.hc_mult, H)  # the main model's pre-collapse hc stream

    out, rolled_input_ids, rolled_position_ids = mtp(
        input_ids, position_ids, hidden_states, position_embeddings, lambda ids, pos: emb(ids)
    )

    assert tuple(out.shape) == (B, S, H)
    assert torch.isfinite(out).all()
    # Depth-1 roll: id at position i becomes token i+1, last position zero-filled.
    assert torch.equal(rolled_input_ids[:, :-1], input_ids[:, 1:])
    assert int(rolled_input_ids[0, -1]) == 0
    assert torch.equal(rolled_position_ids[:, :-1], position_ids[:, 1:])


def test_dsv4_mtp_converter_key_mapping():
    module = _load_native_checkpoint_module()

    # Off by default (existing conversions stay byte-identical).
    assert module.native_key_to_mcore("mtp.0.attn.wq_a.weight") is None
    assert module.native_key_to_mcore("mtp.0.enorm.weight") is None

    # Non-MTP keys are unaffected by the include_mtp switch.
    assert module.native_key_to_mcore("layers.2.attn.wq_a.weight", include_mtp=True) == (
        "layers.2.self_attn.q_a_proj.weight"
    )

    cases = {
        "mtp.0.enorm.weight": "mtp.enorm.weight",
        "mtp.0.hnorm.weight": "mtp.hnorm.weight",
        "mtp.0.e_proj.weight": "mtp.e_proj.weight",
        "mtp.0.h_proj.weight": "mtp.h_proj.weight",
        "mtp.0.norm.weight": "mtp.norm.weight",
        "mtp.0.hc_head_fn": "mtp.hc_head.hc_fn",
        "mtp.0.hc_head_base": "mtp.hc_head.hc_base",
        "mtp.0.hc_head_scale": "mtp.hc_head.hc_scale",
        "mtp.0.attn_norm.weight": "mtp.transformer_layer.input_layernorm.weight",
        "mtp.0.ffn_norm.weight": "mtp.transformer_layer.post_attention_layernorm.weight",
        "mtp.0.hc_attn_fn": "mtp.transformer_layer.attn_hc.fn",
        "mtp.0.hc_attn_base": "mtp.transformer_layer.attn_hc.base",
        "mtp.0.hc_ffn_scale": "mtp.transformer_layer.ffn_hc.scale",
        "mtp.0.attn.attn_sink": "mtp.transformer_layer.self_attn.sinks",
        "mtp.0.attn.wq_a.weight": "mtp.transformer_layer.self_attn.q_a_proj.weight",
        "mtp.0.attn.wkv.weight": "mtp.transformer_layer.self_attn.kv_proj.weight",
        "mtp.0.attn.wo_a.weight": "mtp.transformer_layer.self_attn.o_a_proj.weight",
        "mtp.0.attn.wo_b.weight": "mtp.transformer_layer.self_attn.o_b_proj.weight",
        "mtp.0.attn.q_norm.weight": "mtp.transformer_layer.self_attn.q_a_norm.weight",
        "mtp.0.attn.kv_norm.weight": "mtp.transformer_layer.self_attn.kv_norm.weight",
        "mtp.0.ffn.gate.weight": "mtp.transformer_layer.mlp.gate.weight",
        "mtp.0.ffn.gate.bias": "mtp.transformer_layer.mlp.gate.e_score_correction_bias",
        "mtp.0.ffn.shared_experts.w1.weight": "mtp.transformer_layer.mlp.shared_experts.gate_proj.weight",
        "mtp.0.ffn.shared_experts.w2.weight": "mtp.transformer_layer.mlp.shared_experts.down_proj.weight",
        "mtp.0.ffn.shared_experts.w3.weight": "mtp.transformer_layer.mlp.shared_experts.up_proj.weight",
    }
    for native, mcore in cases.items():
        assert module.native_key_to_mcore(native, include_mtp=True) == mcore, native

    # Scale + routed-expert keys stay unmapped even with the gate on (consumed by
    # FP8 dequant / the expert-assembly loop respectively).
    assert module.native_key_to_mcore("mtp.0.attn.wq_a.scale", include_mtp=True) is None
    assert module.native_key_to_mcore("mtp.0.ffn.experts.5.w1.weight", include_mtp=True) is None
    assert module.MTP_EXPERT_RE.match("mtp.0.ffn.experts.5.w1.weight") is not None


def test_dsv4_mtp_converter_targets_exist_on_built_module():
    """Every mapped MTP direct key must correspond to a real param/buffer on the built
    MTP module, so the converter and the module structure cannot silently drift."""
    module = _load_native_checkpoint_module()
    from custom_kernels.deepseek_v4.megatron.mtp import V4MultiTokenPredictionLayer

    cfg = _tiny_v4_config()
    mtp = V4MultiTokenPredictionLayer(cfg)
    have = {f"mtp.{n}" for n, _ in mtp.named_parameters()}
    have |= {f"mtp.{n}" for n, _ in mtp.named_buffers()}

    native_direct_keys = [
        "mtp.0.enorm.weight",
        "mtp.0.hnorm.weight",
        "mtp.0.e_proj.weight",
        "mtp.0.h_proj.weight",
        "mtp.0.norm.weight",
        "mtp.0.hc_head_fn",
        "mtp.0.hc_head_base",
        "mtp.0.hc_head_scale",
        "mtp.0.attn_norm.weight",
        "mtp.0.ffn_norm.weight",
        "mtp.0.hc_attn_fn",
        "mtp.0.hc_attn_base",
        "mtp.0.hc_attn_scale",
        "mtp.0.hc_ffn_fn",
        "mtp.0.hc_ffn_base",
        "mtp.0.hc_ffn_scale",
        "mtp.0.attn.attn_sink",
        "mtp.0.attn.wq_a.weight",
        "mtp.0.attn.wq_b.weight",
        "mtp.0.attn.wkv.weight",
        "mtp.0.attn.wo_a.weight",
        "mtp.0.attn.wo_b.weight",
        "mtp.0.attn.q_norm.weight",
        "mtp.0.attn.kv_norm.weight",
        "mtp.0.ffn.gate.weight",
        "mtp.0.ffn.gate.bias",
        "mtp.0.ffn.shared_experts.w1.weight",
        "mtp.0.ffn.shared_experts.w2.weight",
        "mtp.0.ffn.shared_experts.w3.weight",
    ]
    for native in native_direct_keys:
        mcore = module.native_key_to_mcore(native, include_mtp=True)
        assert mcore is not None, native
        assert mcore in have, f"{native} -> {mcore} not found on module"

    # The grouped routed experts are loaded via the expert-assembly loop; their target
    # storage exists as gate_up_proj/down_proj on the built module.
    assert "mtp.transformer_layer.mlp.experts.gate_up_proj" in have
    assert "mtp.transformer_layer.mlp.experts.down_proj" in have


def test_dsv4_mtp_forward_activation_checkpoint_matches_eager_and_backprops(monkeypatch):
    """With full recompute the MTP decoder layer is recomputed in backward; the forward
    output must match eager and the backward must still reach the MTP params."""
    _use_torch_reference_kernels(monkeypatch)
    from custom_kernels.deepseek_v4.megatron.model_provider import init_v4_module_weights
    from custom_kernels.deepseek_v4.megatron.mtp import V4MultiTokenPredictionLayer
    from custom_kernels.deepseek_v4.megatron.rope import DeepseekV4RotaryEmbedding

    cfg = _tiny_v4_config()
    torch.manual_seed(0)
    mtp = V4MultiTokenPredictionLayer(cfg)
    init_v4_module_weights(mtp, cfg)
    mtp.train()

    B, S, H = 1, 4, cfg.hidden_size
    emb = torch.nn.Embedding(cfg.vocab_size, H)
    input_ids = torch.randint(0, cfg.vocab_size, (B, S))
    position_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
    rotary = DeepseekV4RotaryEmbedding(cfg)
    x = torch.randn(B, S, H)
    pe = {
        "main": rotary(x, position_ids=position_ids, layer_type="main"),
        "compress": rotary(x, position_ids=position_ids, layer_type="compress"),
    }

    def run():
        hidden = torch.randn(B, S, cfg.hc_mult, H, generator=torch.Generator().manual_seed(7)).requires_grad_(True)
        out, _, _ = mtp(input_ids, position_ids, hidden, pe, lambda ids, pos: emb(ids))
        return out

    cfg.recompute_granularity = None
    out_eager = run()
    cfg.recompute_granularity = "full"
    out_ckpt = run()

    assert torch.allclose(out_eager, out_ckpt, atol=1e-5, rtol=1e-4)
    out_ckpt.sum().backward()
    g = mtp.transformer_layer.self_attn.q_a_proj.weight.grad
    assert g is not None and g.abs().sum() > 0  # recompute reached the MTP attention


def test_dsv4_mtp_expert_ep_sharding_wired_in_model_sharded_state_dict():
    """Regression guard: V4LanguageModel.sharded_state_dict must explicitly EP-shard the
    MTP experts (plain nn.Module -> default recursion won't call the custom EP path)."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "custom_kernels/deepseek_v4/megatron/mcore_model.py").read_text()
    assert "mtp.transformer_layer.mlp.experts." in src
    # And the underlying EP-aware sharded_state_dict is the shared V4GroupedExperts one
    # (already covered by test_dsv4_model_provider's EP shard test), invoked on the MTP.
    assert "self.mtp.transformer_layer.mlp.experts.sharded_state_dict(" in src


def test_roll_mtp_labels_and_masks_depth1_alignment():
    """Depth-1 MTP labels/mask must match Megatron's roll sequence: a provided loss_mask
    rolls LEFT by 2 (once pre-loop + once per depth), labels always roll LEFT by 2."""
    from custom_kernels.deepseek_v4.megatron.mtp import roll_mtp_labels_and_masks

    B, S = 2, 6
    ids = torch.arange(1, B * S + 1).reshape(B, S)

    # Provided loss_mask -> rolled -2 (matches the label roll).
    lm = torch.ones(B, S)
    lm[:, 0] = 0
    ((labels_d, mask_d, num_tokens),) = roll_mtp_labels_and_masks(ids, lm, 1)
    exp_labels = torch.roll(ids, -2, -1)
    exp_labels[:, -2:] = 0
    exp_mask = torch.roll(lm, -2, -1)
    exp_mask[:, -2:] = 0
    assert torch.equal(labels_d, exp_labels)
    assert torch.equal(mask_d, exp_mask)
    assert int(num_tokens) == int(mask_d.sum())

    # loss_mask=None -> ones created AFTER the pre-loop label roll, so it only gets the
    # single in-loop roll (-1); this mirrors Megatron exactly (gpt_model.py:635-637).
    ((_, mask_none, ntok_none),) = roll_mtp_labels_and_masks(ids, None, 1)
    exp_none = torch.ones(B, S)
    exp_none[:, -1:] = 0
    assert torch.equal(mask_none, exp_none)
    assert int(ntok_none) == B * (S - 1)


def test_apply_mtp_loss_populates_tracker_and_seeds_backward():
    """apply_mtp_loss must log a finite loss to MTPLossLoggingHelper and, via
    MTPLossAutoScaler on main_hidden, backprop the MTP loss into the MTP head."""
    from types import SimpleNamespace

    import torch.nn.functional as F
    from custom_kernels.deepseek_v4.megatron.mtp import apply_mtp_loss
    from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler, MTPLossLoggingHelper

    if "values" in MTPLossLoggingHelper.tracker:
        MTPLossLoggingHelper.clean_loss_in_tracker()
    MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0))

    B, S, H, V = 2, 6, 8, 50
    ids = torch.randint(0, V, (B, S))
    mtp_hidden = torch.randn(B, S, H, requires_grad=True)
    main_hidden = torch.randn(B, S, H, requires_grad=True)
    head = torch.nn.Linear(H, V, bias=False)

    def ce_fn(labels, logits_bsv):
        return F.cross_entropy(logits_bsv.reshape(-1, V), labels.reshape(-1), reduction="none").reshape(labels.shape)

    cfg = SimpleNamespace(mtp_num_layers=1, mtp_loss_scaling_factor=0.2, calculate_per_token_loss=False)
    wrapped = apply_mtp_loss(
        mtp_hidden=mtp_hidden,
        main_hidden=main_hidden,
        mtp_labels=ids,
        loss_mask=None,
        output_logits_fn=lambda h: head(h),
        ce_fn=ce_fn,
        config=cfg,
        training=True,
        avg_group=None,
    )
    tracker = MTPLossLoggingHelper.tracker
    assert "values" in tracker and torch.isfinite(tracker["values"]).all() and float(tracker["values"][0]) > 0

    # The main-loss backward on the wrapped hidden must seed the MTP loss backward.
    wrapped.square().mean().backward()
    assert head.weight.grad is not None and head.weight.grad.abs().sum() > 0
    assert mtp_hidden.grad is not None and mtp_hidden.grad.abs().sum() > 0
    MTPLossLoggingHelper.clean_loss_in_tracker()


def test_apply_mtp_loss_with_detached_head_trains_mtp_but_not_head():
    """Production passes a DETACHED output head weight (the MTP loss must not train the
    output head).  With a detached-weight projection, the head weight gets NO grad while
    the MTP hidden still does — guarding the detach contract of the loss path."""
    from types import SimpleNamespace

    import torch.nn.functional as F
    from custom_kernels.deepseek_v4.megatron.mtp import apply_mtp_loss
    from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler, MTPLossLoggingHelper

    if "values" in MTPLossLoggingHelper.tracker:
        MTPLossLoggingHelper.clean_loss_in_tracker()
    MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0))

    B, S, H, V = 2, 6, 8, 50
    ids = torch.randint(0, V, (B, S))
    mtp_hidden = torch.randn(B, S, H, requires_grad=True)
    main_hidden = torch.randn(B, S, H, requires_grad=True)
    head_w = torch.randn(V, H, requires_grad=True)  # the shared output head weight

    def ce_fn(labels, logits_bsv):
        return F.cross_entropy(logits_bsv.reshape(-1, V), labels.reshape(-1), reduction="none").reshape(labels.shape)

    cfg = SimpleNamespace(mtp_num_layers=1, mtp_loss_scaling_factor=0.2, calculate_per_token_loss=False)
    wrapped = apply_mtp_loss(
        mtp_hidden=mtp_hidden,
        main_hidden=main_hidden,
        mtp_labels=ids,
        loss_mask=None,
        output_logits_fn=lambda h: h @ head_w.detach().t(),  # DETACHED head, like production
        ce_fn=ce_fn,
        config=cfg,
        training=True,
        avg_group=None,
    )
    wrapped.square().mean().backward()
    assert head_w.grad is None  # detached -> the MTP loss does NOT train the output head
    assert mtp_hidden.grad is not None and mtp_hidden.grad.abs().sum() > 0
    MTPLossLoggingHelper.clean_loss_in_tracker()


def test_make_transformer_config_sets_mtp_fields_only_when_enabled():
    module = importlib.import_module("custom_kernels.deepseek_v4.megatron.model_provider")
    from types import SimpleNamespace

    hf = SimpleNamespace(
        initializer_range=0.02,
        num_hidden_layers=2,
        hidden_size=8,
        num_attention_heads=2,
        head_dim=4,
        moe_intermediate_size=16,
        max_position_embeddings=32,
        num_local_experts=8,
        num_experts_per_tok=2,
    )
    off = module.make_transformer_config(hf)
    assert off.mtp_num_layers is None  # unchanged when MTP off
    on = module.make_transformer_config(hf, enable_mtp=True, mtp_loss_scaling_factor=0.2)
    assert on.mtp_num_layers == 1
    assert on.mtp_loss_scaling_factor == 0.2


def test_dsv4_stage_builds_embedding_covers_pp_topologies():
    """The MTP head embeds rolled ids on the LAST stage, so that stage must build the
    shared embedding (in addition to the first stage).  At PP=1 first==last."""
    mm = importlib.import_module("custom_kernels.deepseek_v4.megatron.mcore_model")
    f = mm._v4_stage_builds_embedding
    # PP=1: single stage is both first and last -> always builds.
    assert f(pre_process=True, post_process=True, enable_mtp=False) is True
    assert f(pre_process=True, post_process=True, enable_mtp=True) is True
    # PP>1 first stage: always builds.
    assert f(pre_process=True, post_process=False, enable_mtp=True) is True
    # PP>1 last stage: builds ONLY when MTP is enabled (for the rolled-id embedding).
    assert f(pre_process=False, post_process=True, enable_mtp=True) is True
    assert f(pre_process=False, post_process=True, enable_mtp=False) is False
    # PP>1 middle stage: never builds.
    assert f(pre_process=False, post_process=False, enable_mtp=True) is False


def test_dsv4_overrides_setup_embeddings_to_skip_tied_machinery():
    """Regression guard: V4 must NOT fall back to LanguageModule's tied-embedding setup
    (it assumes shared/trainable weights + vp_stage/embd group), which config.mtp_num_layers
    would otherwise activate.  V4's embedding is untied + frozen."""
    from megatron.core.models.common.language_module.language_module import LanguageModule

    mm = importlib.import_module("custom_kernels.deepseek_v4.megatron.mcore_model")
    assert mm.V4LanguageModel.setup_embeddings_and_output_layer is not LanguageModule.setup_embeddings_and_output_layer


def test_dsv4_mtp_router_excluded_from_routing_replay(monkeypatch):
    """With routing replay ON, the MTP MoE router must NOT register a RoutingReplay
    (which would pollute the global list + shift the main-layer offset) and must have
    no ``routing_replay`` attr, so it routes live."""
    monkeypatch.setenv("ENABLE_ROUTING_REPLAY", "1")
    import custom_kernels.deepseek_v4.megatron.mcore_model as mm

    from slime.utils.routing_replay import RoutingReplay

    cfg = _tiny_v4_config()

    before = len(RoutingReplay.all_routing_replays)
    main_moe = mm.V4MoELayer(cfg, 2)  # layer 2 == "moe": a main learned router
    after_main = len(RoutingReplay.all_routing_replays)
    mtp_moe = mm.V4MoELayer(cfg, 2, mlp_type_override="moe", enable_routing_replay=False)
    after_mtp = len(RoutingReplay.all_routing_replays)

    assert after_main - before == 1, "main learned router should register a replay"
    assert after_mtp - after_main == 0, "MTP router must NOT register a replay"
    assert hasattr(main_moe.gate, "routing_replay")
    assert not hasattr(mtp_moe.gate, "routing_replay")
