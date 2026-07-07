"""slime ``custom_model_provider_path``-compatible provider for the V4 mcore model.

slime's ``model_provider.py`` (``_get_model_provider_func``, :69) calls a custom
provider as ``provider(pre_process=..., post_process=..., vp_stage=...)`` and expects
back a GPTModel-like module.  This builds the V4 ``LanguageModule`` at TP=PP=EP=1.

Wire it via ``--custom-model-provider-path
custom_kernels.deepseek_v4.megatron.model_provider.v4_model_provider`` (the provider reads the
HF ``DeepseekV4Config`` from ``--hf-checkpoint`` and a Megatron ``TransformerConfig``
from the slime/Megatron args, exactly like the stock provider builds its config).

For the standalone parity harness (no slime args), use ``build_v4_mcore_model(hf_config)``
which builds a minimal valid ``TransformerConfig`` directly.
"""

import torch
from megatron.core.transformer.transformer_block import get_num_layers_to_build
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from .mcore_model import V4LanguageModel


def _init_method_std(std):
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def make_transformer_config(
    hf_config,
    *,
    params_dtype=torch.bfloat16,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    expert_model_parallel_size=1,
    moe_token_dispatcher_type=None,
    moe_flex_dispatcher_backend="deepep",
    moe_router_dtype="fp32",
    moe_deepep_num_sms=20,
    num_layers_in_first_pipeline_stage=None,
    num_layers_in_last_pipeline_stage=None,
) -> TransformerConfig:
    """Build a minimal-but-valid Megatron ``TransformerConfig`` for the V4
    ``LanguageModule`` base + embedding + output_layer at TP=PP=EP=1.

    The V4 compute modules read shapes from the HF config, not this; this config only
    drives the LanguageModule base class, the (replicated) embedding/output_layer, and
    the process-group plumbing.  kv_channels is pinned to head_dim so the post-init
    validation (num_attention_heads % tp == 0; kv_channels default) is satisfied.
    """
    if tensor_model_parallel_size != 1:
        raise ValueError(
            "V4LanguageModel currently requires tensor_model_parallel_size=1: "
            "V4 attention/compressor/mHC are replicated and not TP-sharded."
        )
    std = hf_config.initializer_range
    moe_kwargs = {}
    if expert_model_parallel_size > 1:
        moe_kwargs = {
            "num_moe_experts": hf_config.num_local_experts,
            "moe_router_topk": hf_config.num_experts_per_tok,
            "moe_ffn_hidden_size": hf_config.moe_intermediate_size,
            "moe_token_dispatcher_type": moe_token_dispatcher_type or "flex",
            "moe_flex_dispatcher_backend": moe_flex_dispatcher_backend,
            "moe_router_dtype": moe_router_dtype,
            "moe_deepep_num_sms": moe_deepep_num_sms,
        }
    pp_kwargs = {}
    if pipeline_model_parallel_size > 1:
        pp_kwargs = {
            "pipeline_dtype": params_dtype,
            "num_layers_in_first_pipeline_stage": num_layers_in_first_pipeline_stage,
            "num_layers_in_last_pipeline_stage": num_layers_in_last_pipeline_stage,
        }
    cfg = TransformerConfig(
        num_layers=hf_config.num_hidden_layers,
        hidden_size=hf_config.hidden_size,
        num_attention_heads=hf_config.num_attention_heads,
        kv_channels=hf_config.head_dim,
        ffn_hidden_size=hf_config.moe_intermediate_size,
        # parallel sizes. V4 compute is custom; these keep Megatron metadata aligned.
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
        expert_tensor_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        # dtype / init.
        params_dtype=params_dtype,
        bf16=(params_dtype == torch.bfloat16),
        fp16=False,
        init_method=_init_method_std(std),
        output_layer_init_method=_init_method_std(std),
        # misc surface the base/embedding read.
        add_bias_linear=False,
        gated_linear_unit=False,
        perform_initialization=True,
        use_cpu_initialization=False,
        # HF DeepseekV4 has NO dropout; Megatron defaults these to 0.1, which would
        # silently corrupt a real (training-mode) forward.  Pin to 0.
        hidden_dropout=0.0,
        attention_dropout=0.0,
        **moe_kwargs,
        **pp_kwargs,
    )
    # max_position_embeddings is not a TransformerConfig field; stash it for the model.
    cfg.max_position_embeddings = hf_config.max_position_embeddings
    return cfg


def resolve_v4_layer_ids(
    config: TransformerConfig, *, vp_stage: int | None = None, pp_rank: int | None = None
) -> tuple[int, ...]:
    """Return global V4 decoder layer ids for one PP/VP stage.

    This is a thin wrapper around Megatron Core's own TransformerBlock layer
    partitioning helpers. It keeps V4's custom ``ModuleList`` construction aligned
    with the checkpoint global layer ids that mcore uses for PP.
    """
    offset = get_transformer_layer_offset(config, vp_stage=vp_stage, pp_rank=pp_rank)
    count = get_num_layers_to_build(config, vp_stage=vp_stage, pp_rank=pp_rank)
    return tuple(range(offset, offset + count))


@torch.no_grad()
def init_v4_module_weights(model, hf_config):
    """Initialize the V4 modules' params that default to ``torch.empty`` (HF builds these
    via ``_init_weights``, which the mcore build path does NOT run).  Without this the
    routers / experts / hc_head are uninitialized (zeros/garbage) and the forward NaNs —
    the M0_NOTES "uninit router weight -> NaN" trap, now hit through the real provider.

    Mirrors HF ``DeepseekV4PreTrainedModel._init_weights`` for the trainable tensors that
    should start from normal/zero/one values. The one deliberate R1 smoke divergence is
    ``V4HashRouter.tid2eid``: random valid ids avoid an all-expert-0 random-init route, while
    a real checkpoint load overwrites it. Megatron already inits the embedding +
    output_layer via ``init_method``; norms/sinks/position_bias/mHC self-init in their
    __init__."""
    import torch.nn as nn
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4HyperHead, DeepseekV4Indexer

    from .mcore_model import V4GroupedExperts, V4HashRouter, V4TopKRouter

    std = hf_config.initializer_range
    for m in model.modules():
        if isinstance(m, (V4TopKRouter, V4HashRouter)):
            nn.init.normal_(m.weight, mean=0.0, std=std)
            if isinstance(m, V4TopKRouter):
                m.e_score_correction_bias.zero_()
            if isinstance(m, V4HashRouter):
                m.tid2eid = torch.randint(
                    0,
                    hf_config.n_routed_experts,
                    m.tid2eid.shape,
                    device=m.weight.device,
                    dtype=torch.long,
                )
        elif isinstance(m, V4GroupedExperts):
            nn.init.normal_(m.gate_up_proj, mean=0.0, std=std)
            nn.init.normal_(m.down_proj, mean=0.0, std=std)
        elif isinstance(m, DeepseekV4HyperHead):
            nn.init.normal_(m.hc_fn, mean=0.0, std=std)
            nn.init.zeros_(m.hc_base)
            nn.init.ones_(m.hc_scale)
        elif isinstance(m, DeepseekV4Indexer):
            nn.init.zeros_(m.position_bias)
    return model


def build_v4_mcore_model(
    hf_config,
    *,
    pre_process=True,
    post_process=True,
    params_dtype=torch.bfloat16,
    init_weights=True,
    layer_ids: list[int] | tuple[int, ...] | None = None,
    expert_model_parallel_size=1,
    expert_model_parallel_rank=0,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    moe_token_dispatcher_type=None,
    moe_flex_dispatcher_backend="deepep",
    moe_router_dtype="fp32",
    moe_deepep_num_sms=20,
    num_layers_in_first_pipeline_stage=None,
    num_layers_in_last_pipeline_stage=None,
    vp_stage=None,
) -> V4LanguageModel:
    """Build the V4 mcore model from an HF ``DeepseekV4Config``.

    ``init_weights`` (default True) runs ``init_v4_module_weights`` so the torch.empty
    router/expert/hc_head params are sane (else a random-init forward NaNs).  Pass
    ``init_weights=False`` when the caller immediately overwrites via load_state_dict
    (the parity harness does, to keep its exact HF-copied weights)."""
    cfg = make_transformer_config(
        hf_config,
        params_dtype=params_dtype,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size,
        expert_model_parallel_size=expert_model_parallel_size,
        moe_token_dispatcher_type=moe_token_dispatcher_type,
        moe_flex_dispatcher_backend=moe_flex_dispatcher_backend,
        moe_router_dtype=moe_router_dtype,
        moe_deepep_num_sms=moe_deepep_num_sms,
        num_layers_in_first_pipeline_stage=num_layers_in_first_pipeline_stage,
        num_layers_in_last_pipeline_stage=num_layers_in_last_pipeline_stage,
    )
    if layer_ids is None and pipeline_model_parallel_size > 1:
        layer_ids = resolve_v4_layer_ids(cfg, vp_stage=vp_stage)
    model = V4LanguageModel(
        config=cfg,
        hf_config=hf_config,
        pre_process=pre_process,
        post_process=post_process,
        layer_ids=layer_ids,
        expert_model_parallel_size=expert_model_parallel_size,
        expert_model_parallel_rank=expert_model_parallel_rank,
    )
    if init_weights:
        init_v4_module_weights(model, hf_config)
    return model


def _v4_lora_cfg(args):
    """Resolve LoRA config from args (if registered) else env vars.  slime's Megatron
    parser uses ``ignore_unknown_args=True``, so a bare ``--v4-lora-dim`` CLI flag is
    silently dropped — the launcher therefore passes these via env vars
    (``V4_LORA_DIM`` / ``V4_LORA_ALPHA`` / ``V4_LORA_DROPOUT``).  args take precedence
    if present (e.g. a future custom-arg provider registers them)."""
    import os

    dim = getattr(args, "v4_lora_dim", None)
    if dim is None:
        dim = os.environ.get("V4_LORA_DIM", 0)
    dim = int(dim or 0)
    alpha = getattr(args, "v4_lora_alpha", None)
    if alpha is None:
        alpha = os.environ.get("V4_LORA_ALPHA", 2 * dim)
    dropout = getattr(args, "v4_lora_dropout", None)
    if dropout is None:
        dropout = os.environ.get("V4_LORA_DROPOUT", 0.0)
    return dim, int(alpha or 2 * dim), float(dropout or 0.0)


def v4_model_provider(pre_process=True, post_process=True, vp_stage=None):
    """slime entry point. Reads the HF config from megatron args (--hf-checkpoint).

    If LoRA dim > 0 (``--v4-lora-dim`` or env ``V4_LORA_DIM``), applies LoRA (freezes
    base, wraps the V4 attention + compressor linears, excludes ``o_a_proj``) on the
    built model BEFORE slime's Float16Module/DDP wrap — so the optimizer sees only the
    trainable LoRA adapters.
    """
    from megatron.core import parallel_state
    from megatron.training import get_args
    from transformers import AutoConfig

    args = get_args()
    hf_config = AutoConfig.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    ep_size = getattr(args, "expert_model_parallel_size", 1)
    ep_rank = parallel_state.get_expert_model_parallel_rank() if ep_size > 1 else 0
    model = build_v4_mcore_model(
        hf_config,
        pre_process=pre_process,
        post_process=post_process,
        params_dtype=getattr(args, "params_dtype", torch.bfloat16),
        expert_model_parallel_size=ep_size,
        expert_model_parallel_rank=ep_rank,
        tensor_model_parallel_size=getattr(args, "tensor_model_parallel_size", 1),
        pipeline_model_parallel_size=getattr(args, "pipeline_model_parallel_size", 1),
        moe_token_dispatcher_type=getattr(args, "moe_token_dispatcher_type", None),
        moe_flex_dispatcher_backend=getattr(args, "moe_flex_dispatcher_backend", "deepep"),
        moe_router_dtype=getattr(args, "moe_router_dtype", "fp32") or "fp32",
        moe_deepep_num_sms=getattr(args, "moe_deepep_num_sms", 20),
        num_layers_in_first_pipeline_stage=getattr(args, "decoder_first_pipeline_num_layers", None),
        num_layers_in_last_pipeline_stage=getattr(args, "decoder_last_pipeline_num_layers", None),
        vp_stage=vp_stage,
    )

    lora_dim, lora_alpha, lora_dropout = _v4_lora_cfg(args)
    if lora_dim > 0:
        from .lora import apply_v4_lora

        model = apply_v4_lora(model, dim=lora_dim, alpha=lora_alpha, dropout=lora_dropout)
    return model


__all__ = [
    "make_transformer_config",
    "resolve_v4_layer_ids",
    "build_v4_mcore_model",
    "v4_model_provider",
]
