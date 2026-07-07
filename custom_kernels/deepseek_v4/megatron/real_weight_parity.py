"""Real-weight R2 forward parity for a single V4-Flash layer.

This gate complements the full ``PP3_EP8`` load-back check: the full artifact
proves distributed checkpoint materialization, while this script proves that a
real native FP8 layer can be dequantized into both HF eager and the mcore V4
model and produce matching forward outputs at EP=1.

It intentionally keeps the model sliced to one source layer.  That avoids the
EP>1 dispatcher gap and avoids loading the full 43-layer checkpoint, while still
using real hidden/expert/vocab shapes and real checkpoint weights.
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import socket
from pathlib import Path

import torch
from transformers import AutoConfig
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    ROPE_INIT_FUNCTIONS,
    DeepseekV4ForCausalLM,
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
    DeepseekV4RotaryEmbedding,
    DeepseekV4TopKRouter,
)

from .m1_parity import Captures, _register, err
from .model_provider import build_v4_mcore_model
from .native_checkpoint import (
    DEFAULT_V4_FLASH_FP8_CKPT,
    NativeV4Checkpoint,
    load_native_checkpoint_into_mcore_model,
    native_key_to_hf,
    read_expert_tensors,
    read_native_weight,
)


def sliced_config(checkpoint_dir: str, source_layer: int):
    cfg = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
    original_layer_types = list(cfg.layer_types)
    original_mlp_layer_types = list(cfg.mlp_layer_types)
    cfg.num_hidden_layers = 1
    cfg.layer_types = [original_layer_types[source_layer]]
    cfg.mlp_layer_types = [original_mlp_layer_types[source_layer]]
    # Keep the HF reference deterministic and aligned with the tiny parity gates.
    cfg._attn_implementation = "eager"
    return cfg, original_layer_types[source_layer], original_mlp_layer_types[source_layer]


def _model_named_tensors(model) -> dict[str, torch.Tensor]:
    targets = {name: param for name, param in model.named_parameters()}
    targets.update({name: buf for name, buf in model.named_buffers()})
    return targets


def _remap_hf_layer_key(hf_key: str, layer_map: dict[int, int]) -> str | None:
    if not layer_map:
        return hf_key
    m = re.match(r"^model\.layers\.(\d+)\.", hf_key)
    if not m:
        return hf_key
    source_layer = int(m.group(1))
    if source_layer not in layer_map:
        return None
    return f"model.layers.{layer_map[source_layer]}.{hf_key[m.end():]}"


def _restore_hf_fp32_modules(model):
    for module in model.modules():
        if isinstance(module, (DeepseekV4HyperConnection, DeepseekV4HyperHead)):
            module.float()
        if isinstance(module, DeepseekV4TopKRouter):
            module.e_score_correction_bias.data = module.e_score_correction_bias.data.float()
    return model


def _restore_hf_rotary_buffers(model):
    """Initialize non-persistent RoPE buffers after ``meta -> to_empty``.

    ``DeepseekV4RotaryEmbedding`` registers inv-freq buffers as non-persistent, so
    they are absent from state_dict and from the native checkpoint.  A meta-built
    model followed by ``to_empty`` leaves those buffers as uninitialized memory
    unless we explicitly recompute them from config.
    """
    for module in model.modules():
        if not isinstance(module, DeepseekV4RotaryEmbedding):
            continue
        for layer_type in module.layer_types:
            rope_init_fn = module.compute_default_rope_parameters
            if module.rope_type[layer_type] != "default":
                rope_init_fn = ROPE_INIT_FUNCTIONS[module.rope_type[layer_type]]
            inv_freq, attention_scaling = rope_init_fn(
                module.config,
                device=getattr(module, f"{layer_type}_inv_freq").device,
                layer_type=layer_type,
            )
            for suffix in ("inv_freq", "original_inv_freq"):
                dest = getattr(module, f"{layer_type}_{suffix}")
                dest.data.copy_(inv_freq.to(device=dest.device, dtype=dest.dtype))
            setattr(module, f"{layer_type}_attention_scaling", attention_scaling)
    return model


def build_empty_hf_model(cfg, device: torch.device):
    with torch.device("meta"):
        model = DeepseekV4ForCausalLM(cfg)
    model = model.to_empty(device=device)
    model = model.to(dtype=torch.bfloat16)
    model = _restore_hf_rotary_buffers(model)
    return _restore_hf_fp32_modules(model).eval()


def init_dist_1rank(master_port: int):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(master_port))
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.cuda.set_device(0)
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            backend="nccl",
            world_size=1,
            rank=0,
            device_id=torch.device("cuda:0"),
        )

    from megatron.core import parallel_state, tensor_parallel

    if not parallel_state.is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
        )
    tensor_parallel.model_parallel_cuda_manual_seed(0)


def load_native_checkpoint_into_hf_model(
    model,
    checkpoint_dir: str,
    *,
    layer_map: dict[int, int],
    strict: bool = True,
) -> dict[str, int | tuple[str, ...]]:
    """Copy native V4 tensors into a sliced HF ``DeepseekV4ForCausalLM`` model."""
    checkpoint = NativeV4Checkpoint(checkpoint_dir)
    targets = _model_named_tensors(model)
    loaded: set[str] = set()
    direct_count = 0
    expert_slice_count = 0

    for native_key in sorted(checkpoint.actual_key_to_file()):
        hf_key = native_key_to_hf(native_key)
        hf_key = _remap_hf_layer_key(hf_key, layer_map) if hf_key is not None else None
        if hf_key is None or hf_key not in targets:
            continue
        dest = targets[hf_key]
        output_dtype = dest.dtype if dest.dtype.is_floating_point else torch.bfloat16
        tensor, _, _ = read_native_weight(checkpoint, native_key, output_dtype=output_dtype)
        dest.data.copy_(tensor.to(device=dest.device, dtype=dest.dtype))
        loaded.add(hf_key)
        direct_count += 1

    target_to_source = {target: source for source, target in layer_map.items()}
    expert_targets = [k for k in targets if k.endswith("mlp.experts.gate_up_proj")]
    for gate_key in sorted(expert_targets):
        m = re.fullmatch(r"model\.layers\.(\d+)\.mlp\.experts\.gate_up_proj", gate_key)
        if not m:
            continue
        target_layer = int(m.group(1))
        source_layer = target_to_source.get(target_layer, target_layer)
        down_key = f"model.layers.{target_layer}.mlp.experts.down_proj"
        gate_dest = targets[gate_key]
        down_dest = targets[down_key]
        for expert_idx in range(gate_dest.shape[0]):
            slices = read_expert_tensors(
                checkpoint,
                source_layer,
                expert_idx,
                output_dtype=gate_dest.dtype if gate_dest.dtype.is_floating_point else torch.bfloat16,
            )
            gate_dest.data[expert_idx].copy_(
                slices[f"layers.{source_layer}.mlp.experts.gate_up_proj[{expert_idx}]"].to(
                    device=gate_dest.device, dtype=gate_dest.dtype
                )
            )
            down_dest.data[expert_idx].copy_(
                slices[f"layers.{source_layer}.mlp.experts.down_proj[{expert_idx}]"].to(
                    device=down_dest.device, dtype=down_dest.dtype
                )
            )
            expert_slice_count += 2
        loaded.add(gate_key)
        loaded.add(down_key)

    missing = tuple(
        sorted(
            k for k in targets if k not in loaded and not (k.startswith("model.rotary_emb.") or ".rotary_emb." in k)
        )
    )
    if strict and missing:
        raise AssertionError(f"HF tensors not loaded from native checkpoint: {missing}")
    return {
        "direct_tensors": direct_count,
        "expert_slices": expert_slice_count,
        "missing": missing,
    }


def _metric_line(name: str, metric: dict[str, float]) -> str:
    return (
        f"{name:<20} rel={metric['rel']:.6f} cos={metric['cos']:.8f} "
        f"max_abs={metric['max_abs']:.6f} p99={metric['p99_abs']:.6f} "
        f"worst_tok={metric['worst_tok']:.6f}"
    )


def _router_stats(cap: Captures) -> dict[str, object]:
    hf_idx = cap.d["hf.L0.router_idx"]
    mc_idx = cap.d["m0.L0.router_idx"]
    agree = (hf_idx.sort(-1).values == mc_idx.sort(-1).values).all(-1)
    total = int(agree.numel())
    matched = int(agree.sum().item())
    flat_agree = agree.reshape(-1)
    hf_router_in = cap.d["hf.L0.router_in"].reshape(total, -1)
    mc_router_in = cap.d["m0.L0.router_in"].reshape(total, -1)
    stats: dict[str, object] = {
        "matched": matched,
        "total": total,
        "flips": total - matched,
        "router_in_all": err(hf_router_in, mc_router_in),
    }
    if matched < total:
        flipped = ~flat_agree
        stats["router_in_flipped"] = err(hf_router_in[flipped], mc_router_in[flipped])
    else:
        stats["router_in_flipped"] = None
    if matched > 0 and "hf.L0.mlp_out" in cap.d and "m0.L0.mlp_out" in cap.d:
        hf_mlp = cap.d["hf.L0.mlp_out"].reshape(total, -1)
        mc_mlp = cap.d["m0.L0.mlp_out"].reshape(total, -1)
        stats["matched_mlp"] = err(hf_mlp[flat_agree], mc_mlp[flat_agree])
    else:
        stats["matched_mlp"] = None
    return stats


def run_parity(args) -> tuple[bool, list[str]]:
    init_dist_1rank(args.master_port)
    device = torch.device("cuda")
    torch.cuda.set_device(0)
    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats(device)

    cfg, layer_type, mlp_layer_type = sliced_config(args.checkpoint, args.source_layer)
    if args.seq_len % cfg.compress_rates.get(layer_type, 1) and layer_type != "sliding_attention":
        raise ValueError(
            f"--seq-len={args.seq_len} must divide {layer_type} compress rate "
            f"{cfg.compress_rates[layer_type]} for this stateless dense-index parity gate"
        )

    layer_map = {args.source_layer: 0}
    hf = build_empty_hf_model(cfg, device)
    hf_stats = load_native_checkpoint_into_hf_model(hf, args.checkpoint, layer_map=layer_map, strict=True)

    mc = build_v4_mcore_model(
        cfg,
        params_dtype=torch.bfloat16,
        init_weights=False,
        expert_model_parallel_size=1,
        expert_model_parallel_rank=0,
    )
    mc = mc.cuda().bfloat16().eval()
    mc_stats = load_native_checkpoint_into_mcore_model(mc, args.checkpoint, layer_map=layer_map, strict=True)

    input_ids = torch.randint(0, cfg.vocab_size, (args.batch_size, args.seq_len), device=device, dtype=torch.long)
    position_ids = torch.arange(args.seq_len, device=device).unsqueeze(0).expand(args.batch_size, -1)

    cap = Captures()
    handles = _register(hf.model.layers, cap, "hf") + _register(mc.layers, cap, "m0")
    mc_hidden: dict[str, torch.Tensor] = {}

    def norm_hook(_mod, _inp, out):
        mc_hidden["value"] = out.detach()

    handles.append(mc.norm.register_forward_hook(norm_hook))
    try:
        with torch.no_grad():
            hf_hidden = hf.model(
                input_ids=input_ids,
                position_ids=position_ids,
                use_cache=False,
            ).last_hidden_state
            hf_logits = hf.lm_head(hf_hidden).float()
            mc_logits = mc(input_ids, position_ids=position_ids)
    finally:
        for handle in handles:
            handle.remove()

    if "value" not in mc_hidden:
        raise AssertionError("mcore final hidden hook did not fire")

    metrics = {}
    for seam in ("compressor", "attn_pre_oproj", "post_attn_hc", "mlp_out", "post_ffn_hc"):
        hf_key = f"hf.L0.{seam}"
        mc_key = f"m0.L0.{seam}"
        if hf_key in cap.d and mc_key in cap.d:
            metrics[seam] = err(cap.d[hf_key], cap.d[mc_key])
    metrics["final_hidden"] = err(hf_hidden, mc_hidden["value"])
    metrics["logits"] = err(hf_logits, mc_logits)
    router = _router_stats(cap)
    router_ok = int(router["matched"])
    router_total = int(router["total"])
    router_flips = int(router["flips"])

    finite = all(torch.isfinite(t.float()).all().item() for t in (hf_hidden, hf_logits, mc_hidden["value"], mc_logits))
    failures = []
    for name, metric in metrics.items():
        if name == "mlp_out" and router_flips:
            if metric["cos"] < args.min_moe_cos:
                failures.append(f"{name}: rel={metric['rel']:.6f} cos={metric['cos']:.8f}")
            continue
        if metric["cos"] < args.min_cos or metric["rel"] > args.max_rel:
            failures.append(f"{name}: rel={metric['rel']:.6f} cos={metric['cos']:.8f}")
    max_router_flips = max(
        args.router_flip_abs_min,
        int(args.max_router_flip_frac * router_total),
    )
    if router_flips > max_router_flips:
        failures.append(f"router agreement {router_ok}/{router_total}")
    flipped_in = router["router_in_flipped"]
    if flipped_in is not None and flipped_in["rel"] > args.max_rel:
        failures.append(f"router flipped-input rel={flipped_in['rel']:.6f}")
    if not finite:
        failures.append("non-finite hidden/logit tensor")

    peak_gib = torch.cuda.max_memory_allocated(device) / (1024**3)
    lines = [
        "R2 real-weight HF eager vs mcore forward parity",
        f"date=2026-07-01 host={socket.gethostname()} pid={os.getpid()}",
        f"checkpoint={args.checkpoint}",
        f"source_layer={args.source_layer} -> local_layer=0",
        f"layer_type={layer_type} mlp_layer_type={mlp_layer_type}",
        f"batch_size={args.batch_size} seq_len={args.seq_len} seed={args.seed}",
        (
            f"thresholds: min_cos={args.min_cos} max_rel={args.max_rel} "
            f"min_moe_cos={args.min_moe_cos} max_router_flips={max_router_flips}"
        ),
        f"hf_load_stats={hf_stats}",
        f"mcore_load_stats={mc_stats}",
        f"router_agreement={router_ok}/{router_total}",
        _metric_line("router_in_all", router["router_in_all"]),
        f"input_ids_sample={input_ids[0, : min(args.seq_len, 16)].detach().cpu().tolist()}",
        f"cuda_peak_allocated_gib={peak_gib:.3f}",
        "",
        "## metrics",
    ]
    if router["router_in_flipped"] is not None:
        lines.append(_metric_line("router_in_flipped", router["router_in_flipped"]))
    if router["matched_mlp"] is not None:
        lines.append(_metric_line("matched_mlp", router["matched_mlp"]))
    for name in sorted(metrics):
        lines.append(_metric_line(name, metrics[name]))
    lines.append("")
    if failures:
        lines.append("VERDICT=FAIL")
        lines.extend(f"failure={item}" for item in failures)
    else:
        lines.append("VERDICT=PASS")
    lines.append("note=single-layer real-shape EP=1 gate; full PP3_EP8 conversion/load-back is verified separately.")

    del hf, mc, hf_hidden, hf_logits, mc_logits, mc_hidden
    gc.collect()
    torch.cuda.empty_cache()

    return not failures, lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Real-weight V4 FP8 -> HF/mcore forward parity.")
    parser.add_argument("--checkpoint", default=DEFAULT_V4_FLASH_FP8_CKPT)
    parser.add_argument("--source-layer", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--master-port", type=int, default=29681)
    parser.add_argument("--min-cos", type=float, default=0.997)
    parser.add_argument("--min-moe-cos", type=float, default=0.997)
    parser.add_argument("--max-rel", type=float, default=0.05)
    parser.add_argument("--max-router-flip-frac", type=float, default=0.05)
    parser.add_argument("--router-flip-abs-min", type=int, default=2)
    parser.add_argument(
        "--output",
        default="handoffs/deepseek-v4/r2_logs/real_weight_layer2_parity.txt",
        help="Reviewable text report path.",
    )
    args = parser.parse_args(argv)

    try:
        ok, lines = run_parity(args)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines), flush=True)
        return 0 if ok else 1
    finally:
        from megatron.core import parallel_state

        if parallel_state.is_initialized():
            parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
