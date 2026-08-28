import re


_LORA_ADAPTER_MARKERS = (".linear_in.", ".linear_out.")


def _strip_module_prefix(name: str) -> str:
    for prefix in ("module.module.", "module."):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def _is_lora_adapter_param(name: str) -> bool:
    return any(marker in name for marker in _LORA_ADAPTER_MARKERS)


def _convert_top_level(name, param):
    top_level = {
        "embedding.word_embeddings.weight": "embed.weight",
        "output_layer.weight": "head.weight",
        "norm.weight": "norm.weight",
        "decoder.final_layernorm.weight": "norm.weight",
        "hc_head.hc_fn": "hc_head_fn",
        "hc_head.hc_base": "hc_head_base",
        "hc_head.hc_scale": "hc_head_scale",
    }
    if name in top_level:
        return [(top_level[name], param)]
    return None


def _convert_self_attn(layer_idx: str, rest: str, param):
    attn_leaf = {
        "sinks": "attn.attn_sink",
        "q_a_proj.weight": "attn.wq_a.weight",
        "q_b_proj.weight": "attn.wq_b.weight",
        "kv_proj.weight": "attn.wkv.weight",
        "o_a_proj.weight": "attn.wo_a.weight",
        "o_b_proj.weight": "attn.wo_b.weight",
        "q_a_norm.weight": "attn.q_norm.weight",
        "kv_norm.weight": "attn.kv_norm.weight",
    }
    if rest in attn_leaf:
        return [(f"layers.{layer_idx}.{attn_leaf[rest]}", param)]

    compressor_leaf = {
        "position_bias": "attn.compressor.ape",
        "kv_proj.weight": "attn.compressor.wkv.weight",
        "gate_proj.weight": "attn.compressor.wgate.weight",
        "kv_norm.weight": "attn.compressor.norm.weight",
    }
    if rest.startswith("compressor."):
        leaf = rest[len("compressor.") :]
        if leaf in compressor_leaf:
            return [(f"layers.{layer_idx}.{compressor_leaf[leaf]}", param)]

    indexer_leaf = {
        "q_b_proj.weight": "attn.indexer.wq_b.weight",
        "weights_proj.weight": "attn.indexer.weights_proj.weight",
        "position_bias": "attn.indexer.compressor.ape",
        "kv_proj.weight": "attn.indexer.compressor.wkv.weight",
        "gate_proj.weight": "attn.indexer.compressor.wgate.weight",
        "kv_norm.weight": "attn.indexer.compressor.norm.weight",
    }
    if rest.startswith("compressor.indexer."):
        leaf = rest[len("compressor.indexer.") :]
        if leaf in indexer_leaf:
            return [(f"layers.{layer_idx}.{indexer_leaf[leaf]}", param)]

    return None


def _expert_range_from_rest(rest: str) -> tuple[str, int, int] | None:
    match = re.fullmatch(r"mlp\.experts\.(gate_up_proj|down_proj)\.expert_range(\d+)-(\d+)", rest)
    if not match:
        return None
    kind, start, end = match.groups()
    return kind, int(start), int(end)


def _convert_expert_range(layer_idx: str, rest: str, param):
    parsed = _expert_range_from_rest(rest)
    if parsed is None:
        return None

    kind, start, end = parsed
    if end - start != param.shape[0]:
        raise ValueError(
            f"Expert range {start}-{end} for layer {layer_idx} has {end - start} experts, "
            f"but tensor has first dimension {param.shape[0]}"
        )

    outputs = []
    for local_idx, expert_idx in enumerate(range(start, end)):
        if kind == "gate_up_proj":
            gate_weight, up_weight = param[local_idx].chunk(2, dim=0)
            outputs.append((f"layers.{layer_idx}.ffn.experts.{expert_idx}.w1.weight", gate_weight))
            outputs.append((f"layers.{layer_idx}.ffn.experts.{expert_idx}.w3.weight", up_weight))
        elif kind == "down_proj":
            outputs.append((f"layers.{layer_idx}.ffn.experts.{expert_idx}.w2.weight", param[local_idx]))
        else:
            raise ValueError(f"Unknown DeepSeek V4 expert tensor kind: {kind}")
    return outputs


def _convert_mlp(layer_idx: str, rest: str, param):
    expert_outputs = _convert_expert_range(layer_idx, rest, param)
    if expert_outputs is not None:
        return expert_outputs

    mlp_leaf = {
        "mlp.gate.weight": "ffn.gate.weight",
        "mlp.gate.tid2eid": "ffn.gate.tid2eid",
        "mlp.gate.e_score_correction_bias": "ffn.gate.bias",
        "mlp.shared_experts.gate_proj.weight": "ffn.shared_experts.w1.weight",
        "mlp.shared_experts.up_proj.weight": "ffn.shared_experts.w3.weight",
        "mlp.shared_experts.down_proj.weight": "ffn.shared_experts.w2.weight",
    }
    if rest in mlp_leaf:
        return [(f"layers.{layer_idx}.{mlp_leaf[rest]}", param)]
    return None


def convert_deepseekv4_to_hf(args, name, param):
    del args
    name = _strip_module_prefix(name)
    if _is_lora_adapter_param(name):
        return []

    top_level = _convert_top_level(name, param)
    if top_level is not None:
        return top_level

    match = re.fullmatch(r"(?:decoder\.)?layers\.(\d+)\.(.+)", name)
    if not match:
        raise ValueError(f"Unknown DeepSeek V4 parameter name: {name}")
    layer_idx, rest = match.groups()

    hc_leaf = {
        "attn_hc.fn": "hc_attn_fn",
        "attn_hc.base": "hc_attn_base",
        "attn_hc.scale": "hc_attn_scale",
        "ffn_hc.fn": "hc_ffn_fn",
        "ffn_hc.base": "hc_ffn_base",
        "ffn_hc.scale": "hc_ffn_scale",
    }
    if rest in hc_leaf:
        return [(f"layers.{layer_idx}.{hc_leaf[rest]}", param)]

    norm_leaf = {
        "input_layernorm.weight": "attn_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
    }
    if rest in norm_leaf:
        return [(f"layers.{layer_idx}.{norm_leaf[rest]}", param)]

    if rest.startswith("self_attn."):
        converted = _convert_self_attn(layer_idx, rest[len("self_attn.") :], param)
        if converted is not None:
            return converted

    converted = _convert_mlp(layer_idx, rest, param)
    if converted is not None:
        return converted

    raise ValueError(f"Unknown DeepSeek V4 parameter name: {name}")
