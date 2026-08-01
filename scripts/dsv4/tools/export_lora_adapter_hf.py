#!/usr/bin/env python3
"""Export an adapter-only Megatron distcp checkpoint to an HF/PEFT LoRA directory.

Reads a ``torch.distributed.checkpoint`` ("distcp") directory that slime writes when
DS-V4 adapter-only checkpointing (only the trainable LoRA adapter tensors — no frozen
base, no optimizer state) and emits a PEFT adapter directory
(``adapter_model.safetensors`` + ``adapter_config.json``) that sglang can load via
``/load_lora_adapter``.

Ground truth for the megatron -> serving name map is the SAME code path the live
weight-sync uses: ``slime.backends.megatron_utils.update_weight.lora_adapter_sync``
(``megatron_adapter_name_to_peft`` / ``build_lora_adapter_state_dict``), which routes
each adapter's base module through the V4 ``convert_deepseekv4_to_hf`` converter. So the
exported keys are byte-identical to the names sglang expects — the same contract that
``scripts/dsv4/diagnostics/lora/lora_reload_repro_driver.py::build_fake_adapter`` emits.

Scaling: the trainer uses r16/alpha32 (``megatron.bridge`` ``LinearAdapter`` applies
delta = scale * B @ A with scale = alpha/r classic, alpha/sqrt(r) under rsLoRA). The
exported ``lora_B`` tensors are RAW (unscaled); the adapter_config.json carries
``lora_alpha`` so sglang reproduces the same effective delta via its own
``scaling = lora_alpha / r``. This matches ``build_lora_adapter_state_dict`` exactly
(``lora_alpha = scale * r``). **If the checkpoint was trained with ``--lora-rslora``
you MUST pass ``--rslora``** — the tensors alone cannot reveal the scaling mode, and
exporting an rsLoRA-trained adapter without it serves a 4x-too-weak delta (at r=16).

CPU-only, single-process: the adapter tensors are small dense 2D matrices (replicated
across DP, one PP owner each), so a ``no_dist`` DCP load reconstructs them without a
process group. Run with ``CUDA_VISIBLE_DEVICES=""``.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader

# Direct execution sets sys.path[0] to scripts/dsv4/tools, which can otherwise resolve an
# older site-packages ``slime`` instead of this checkout. The exporter and the live
# sync must use the same source tree because their Megatron->PEFT name map is the
# serving contract.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from slime.backends.megatron_utils.update_weight.lora_adapter_sync import (
    build_lora_adapter_state_dict,
    is_adapter_param_name,
)

# --- authoritative expected contract (read off build_fake_adapter with shared experts) ---
# Module-path suffixes (layer index stripped) present under base_model.model.layers.{L}.
# MANDATORY leaves appear on every decoder layer; OPTIONAL (compressor) leaves appear only
# on layers that carry a compressor (compress_ratios[L] != 0).
_MANDATORY_SUFFIXES = (
    "attn.wq_a",
    "attn.wq_b",
    "attn.wkv",
    "attn.wo_b",
    "ffn.shared_experts.w1",
    "ffn.shared_experts.w2",
    "ffn.shared_experts.w3",
)
_OPTIONAL_SUFFIXES = (
    "attn.compressor.wkv",
    "attn.compressor.wgate",
)
_ALL_SUFFIXES = set(_MANDATORY_SUFFIXES) | set(_OPTIONAL_SUFFIXES)
_EXPECTED_TARGET_MODULES = {"wq_a", "wq_b", "wkv", "wo_b", "wgate", "w1", "w2", "w3"}

_PEFT_NAME_RE = re.compile(r"^base_model\.model\.layers\.(\d+)\.(.+)\.lora_([AB])\.weight$")


def _read_adapter_metadata(reader: FileSystemReader):
    """Return {megatron_key -> (size_tuple, dtype)} for every LoRA adapter tensor."""
    md = reader.read_metadata()
    out = {}
    for key, entry in md.state_dict_metadata.items():
        # Exact-resume adapter checkpoints may also contain Muon state and fp32
        # masters whose nested parameter suffixes look like LoRA leaves. They are
        # optimizer artifacts, not served model tensors. The model entries are the
        # unprefixed layer keys (766 for the formal rank-32/shared-expert recipe).
        if key.startswith("optimizer."):
            continue
        if not is_adapter_param_name(key):
            continue
        size = getattr(entry, "size", None)
        props = getattr(entry, "properties", None)
        if size is None or props is None:
            # A LoRA-named non-tensor entry would be a red flag; surface it.
            raise ValueError(f"adapter key {key!r} is not a tensor entry ({type(entry).__name__})")
        out[key] = (tuple(size), props.dtype)
    return out


def _load_adapter_tensors(reader: FileSystemReader, meta: dict) -> dict[str, torch.Tensor]:
    """Single-process (no_dist) DCP load of exactly the adapter tensors."""
    state = {k: torch.empty(size, dtype=dtype) for k, (size, dtype) in meta.items()}
    dcp.load(state, storage_reader=reader, no_dist=True)
    return state


def _detect_rank(named_tensors) -> int:
    ranks = set()
    for name, tensor in named_tensors:
        # lora_A == megatron ".linear_in.weight" has shape (rank, in_dim).
        if name.endswith(".linear_in.weight"):
            ranks.add(int(tensor.shape[0]))
    if len(ranks) != 1:
        raise ValueError(f"could not determine a single LoRA rank; saw {sorted(ranks)}")
    return next(iter(ranks))


def _stack_norm(tensors) -> float:
    total = 0.0
    for t in tensors:
        total += t.float().pow(2).sum().item()
    return total**0.5


def _validate(state_dict: dict[str, torch.Tensor], config_dict: dict, *, expected_rank: int, expected_alpha):
    problems = []

    # (b) every exported name matches an allowed pattern; mandatory leaves present per layer.
    per_layer_suffix_sides: dict[int, dict[str, set[str]]] = {}
    for name in state_dict:
        m = _PEFT_NAME_RE.match(name)
        if not m:
            problems.append(f"name does not match PEFT pattern: {name}")
            continue
        layer, suffix, side = int(m.group(1)), m.group(2), m.group(3)
        if suffix not in _ALL_SUFFIXES:
            problems.append(f"unexpected module suffix {suffix!r} in {name}")
            continue
        per_layer_suffix_sides.setdefault(layer, {}).setdefault(suffix, set()).add(side)

    layers = sorted(per_layer_suffix_sides)
    for layer in layers:
        got = per_layer_suffix_sides[layer]
        for suffix in _MANDATORY_SUFFIXES:
            if suffix not in got:
                problems.append(f"layer {layer}: missing mandatory {suffix}")
        for suffix, sides in got.items():
            if sides != {"A", "B"}:
                problems.append(f"layer {layer}: {suffix} has sides {sorted(sides)} (need both A and B)")

    # rank / config
    if config_dict.get("r") != expected_rank:
        problems.append(f"config r={config_dict.get('r')} != expected {expected_rank}")
    if config_dict.get("lora_alpha") != expected_alpha:
        problems.append(f"config lora_alpha={config_dict.get('lora_alpha')} != expected {expected_alpha}")
    if config_dict.get("peft_type") != "lora":
        problems.append(f"config peft_type={config_dict.get('peft_type')!r} != 'lora'")
    tm = set(config_dict.get("target_modules", []))
    if tm != _EXPECTED_TARGET_MODULES:
        problems.append(f"target_modules {sorted(tm)} != expected {sorted(_EXPECTED_TARGET_MODULES)}")

    # (c) B tensors non-zero (trained adapter).
    b_norm = _stack_norm([t for n, t in state_dict.items() if n.endswith(".lora_B.weight")])
    if not (b_norm > 0.1):
        problems.append(f"total lora_B L2 norm {b_norm:.4f} <= 0.1 (adapter looks untrained/zero)")

    return problems, layers, per_layer_suffix_sides, b_norm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="adapter-only distcp directory (contains .metadata + __*.distcp)")
    ap.add_argument("--out", required=True, help="output PEFT adapter directory")
    ap.add_argument("--lora-alpha", type=int, default=32, help="trainer LoRA alpha (r16/alpha32 default)")
    ap.add_argument("--expected-rank", type=int, default=16, help="rank assertion (0 to skip)")
    ap.add_argument(
        "--rslora",
        action="store_true",
        help="checkpoint was trained with --lora-rslora (trainer scale alpha/sqrt(r)); "
        "exports the effective lora_alpha = alpha*sqrt(r) so sglang's lora_alpha/r "
        "reproduces the trainer scaling exactly",
    )
    args = ap.parse_args()

    print(f"[export] reading distcp metadata from {args.ckpt}", flush=True)
    reader = FileSystemReader(args.ckpt)
    meta = _read_adapter_metadata(reader)
    if not meta:
        print("[export] ERROR: no LoRA adapter tensors found in checkpoint metadata", file=sys.stderr)
        return 2
    print(f"[export] adapter tensor entries in checkpoint: {len(meta)}", flush=True)

    print("[export] loading adapter tensors (CPU, no_dist single-process)...", flush=True)
    loaded = _load_adapter_tensors(reader, meta)
    named_tensors = [(k, loaded[k]) for k in sorted(loaded)]

    rank = _detect_rank(named_tensors)
    if args.rslora:
        import math

        scale = args.lora_alpha / math.sqrt(rank)
        print(
            f"[export] detected rank={rank}, alpha={args.lora_alpha}, RSLORA -> scale(alpha/sqrt(r))={scale}",
            flush=True,
        )
    else:
        scale = args.lora_alpha / rank
        print(f"[export] detected rank={rank}, alpha={args.lora_alpha} -> scale(alpha/r)={scale}", flush=True)

    ns = SimpleNamespace()
    state_dict, config_dict = build_lora_adapter_state_dict(ns, named_tensors, scale=scale)

    # The exported (effective) alpha: scale * r — equals the trainer alpha for
    # classic scaling, alpha*sqrt(r) for rsLoRA (e.g. 128 at r=16/alpha=32).
    expected_alpha = scale * rank
    if float(expected_alpha).is_integer():
        expected_alpha = int(expected_alpha)

    # ---- validation (artifact-gated: refuse to write on any failure) ----
    problems, layers, per_layer, b_norm = _validate(
        state_dict, config_dict, expected_rank=args.expected_rank, expected_alpha=expected_alpha
    )

    a_norm = _stack_norm([t for n, t in state_dict.items() if n.endswith(".lora_A.weight")])
    n_compressor_layers = sum(1 for L in layers if "attn.compressor.wkv" in per_layer[L])

    print("\n========== EXPORT SUMMARY ==========", flush=True)
    print(f"exported tensors        : {len(state_dict)}", flush=True)
    print(f"decoder layers          : {len(layers)} (range {layers[0]}..{layers[-1]})", flush=True)
    print(f"compressor layers        : {n_compressor_layers}", flush=True)
    print(f"rank / lora_alpha       : {config_dict['r']} / {config_dict['lora_alpha']}", flush=True)
    print(f"target_modules          : {config_dict['target_modules']}", flush=True)
    print(f"global L2 norm  A-stack : {a_norm:.4f}", flush=True)
    print(f"global L2 norm  B-stack : {b_norm:.4f}", flush=True)
    print("sample names:", flush=True)
    for n in list(state_dict)[:6]:
        print(f"  {n}  shape={tuple(state_dict[n].shape)}", flush=True)
    print("per-tensor shapes for 3 layers:", flush=True)
    sample_layers = [layers[0], layers[len(layers) // 2], layers[-1]]
    for L in sample_layers:
        print(f"  --- layer {L} ---", flush=True)
        for n in sorted(k for k in state_dict if k.startswith(f"base_model.model.layers.{L}.")):
            print(f"    {n}  shape={tuple(state_dict[n].shape)}", flush=True)

    if problems:
        print("\n[export] VALIDATION FAILED:", flush=True)
        for p in problems:
            print(f"  - {p}", flush=True)
        print("[export] refusing to write a checkpoint that fails the contract.", flush=True)
        return 3
    print("\n[export] VALIDATION PASSED", flush=True)

    # ---- write ----
    os.makedirs(args.out, exist_ok=True)
    from safetensors.torch import save_file

    save_file(
        {k: v.detach().cpu().contiguous() for k, v in state_dict.items()},
        os.path.join(args.out, "adapter_model.safetensors"),
    )
    with open(os.path.join(args.out, "adapter_config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"[export] wrote {args.out}/adapter_model.safetensors ({len(state_dict)} tensors)", flush=True)
    print(f"[export] wrote {args.out}/adapter_config.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
