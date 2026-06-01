"""Offline RTN W8A8-INT8 checkpoint writer using the repo-local INT8 helper.

This avoids the llmcompressor save path for W8A8 RTN. It streams the source HF
safetensors shard-by-shard, quantizes text-side Linear weights via
`scripts.quantize.utils.int8_quantization`, and validates the saved checkpoint
against the BF16 source before returning.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.quantize.utils.int8_quantization import quantize_layer_int8
from scripts.quantize.utils.validate_checkpoint import check_w8a8_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path, help="Source BF16 HF checkpoint directory")
    parser.add_argument("--output-path", required=True, type=Path, help="Destination W8A8 checkpoint directory")
    parser.add_argument(
        "--target",
        choices=["all-linear", "mlp", "mlp_mtp", "non_linear_attn", "non_linear_attn_mtp"],
        default="all-linear",
        help=(
            "Text-side Linear weights to quantize. "
            "'all-linear' = MLP + self_attn + linear_attn (496 modules). "
            "'mlp' = MLP only (192 modules). "
            "'mlp_mtp' = same as mlp but ALSO quantizes the MTP draft head's mlp. "
            "'non_linear_attn' = MLP + self_attn, skip mamba-style linear_attn (256 modules); "
            "keeps the MTP/EAGLE draft head in BF16. "
            "'non_linear_attn_mtp' = same as non_linear_attn but ALSO quantizes the "
            "MTP draft head's mlp + self_attn (mtp.fc and mamba linear_attn stay BF16)."
        ),
    )
    parser.add_argument("--force", action="store_true", help="Remove an existing output directory before writing")
    parser.add_argument("--validation-max-tensors", type=int, default=32)
    parser.add_argument("--validation-max-saturated-frac", type=float, default=0.20)
    parser.add_argument("--validation-max-rel-l2", type=float, default=0.08)
    return parser.parse_args()


def _load_weight_map(checkpoint_dir: Path) -> tuple[dict[str, str], list[str]]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map = dict(index["weight_map"])
        shards = sorted(set(weight_map.values()))
        return weight_map, shards

    safetensors_files = sorted(path.name for path in checkpoint_dir.glob("*.safetensors"))
    if len(safetensors_files) == 1:
        shard = safetensors_files[0]
        with safe_open(checkpoint_dir / shard, framework="pt", device="cpu") as f:
            return {name: shard for name in f.keys()}, [shard]

    raise FileNotFoundError(
        f"Could not resolve safetensors in {checkpoint_dir}; expected an index or one safetensors file."
    )


def should_quantize_weight(name: str, shape: tuple[int, ...], target: str) -> bool:
    if not name.endswith(".weight") or len(shape) != 2:
        return False
    is_body = name.startswith("model.language_model.layers.")
    # MTP / EAGLE draft head weights are stored under `mtp.layers.*` (plus the
    # standalone `mtp.fc` / `mtp.norm` / `mtp.pre_fc_norm_*`). Only the
    # *_mtp targets quantize the corresponding draft-head scope; everything
    # else keeps it BF16.
    is_mtp_layer = name.startswith("mtp.layers.")
    if target == "mlp_mtp":
        if not (is_body or is_mtp_layer):
            return False
        return ".mlp." in name
    if target == "non_linear_attn_mtp":
        # MLP + self_attn for BOTH the model body and the MTP draft layer; skip
        # linear_attn (mamba) and mtp.fc (the [embed;hidden] projection, which is
        # neither mlp nor self_attn).
        if not (is_body or is_mtp_layer):
            return False
        return ".mlp." in name or ".self_attn." in name
    if not is_body:
        return False
    if target == "mlp":
        return ".mlp." in name
    if target == "non_linear_attn":
        # Quantize MLP + self_attn (softmax attention); skip linear_attn (mamba).
        return ".mlp." in name or ".self_attn." in name
    return True


def build_quantization_config(target: str) -> dict:
    # SGLang fuses gate_proj+up_proj into a single MergedColumnParallelLinear
    # named `gate_up_proj` at load time. The compressed-tensors loader then
    # asks "is `mlp.gate_up_proj` covered?" against `targets`, so the regex
    # must accept the fused name as well, even though the safetensors-side
    # weights are written under the unfused `gate_proj` / `up_proj` keys.
    # SGLang fuses gate_proj+up_proj into MergedColumnParallelLinear named
    # `gate_up_proj`, and self_attn q/k/v into QKVParallelLinear named
    # `qkv_proj`. The compressed-tensors loader looks up these fused module
    # names against `targets`, so any partial-quant target must include the
    # fused alias even though safetensors stores the unfused per-proj weights.
    mlp_target_re = r"re:.*\.mlp\.(gate_proj|up_proj|down_proj|gate_up_proj)$"
    self_attn_target_re = r"re:.*\.self_attn\.(q_proj|k_proj|v_proj|o_proj|qkv_proj)$"
    if target == "all-linear":
        targets = ["Linear"]
    elif target in ("mlp", "mlp_mtp"):
        targets = [mlp_target_re]
    elif target in ("non_linear_attn", "non_linear_attn_mtp"):
        # Both quantize MLP + self_attn. The mtp variant differs only in NOT
        # ignoring the draft head below: the same `mlp_target_re` /
        # `self_attn_target_re` already match the draft modules `mtp.layers.0.
        # mlp.gate_up_proj` / `mtp.layers.0.self_attn.qkv_proj` (sglang builds
        # the draft layer with a `self_attn` prefix), so no extra target needed.
        targets = [mlp_target_re, self_attn_target_re]
    else:
        raise ValueError(f"Unknown target {target!r}")
    # sglang compressed-tensors loader requires every Linear module to either
    # match `targets` or `ignore`. For target="mlp", the self_attn and
    # linear_attn (Qwen3.5 mamba-style) Linear submodules are neither, so they
    # must be added explicitly to `ignore` or sglang refuses to load with
    # `ValueError: Unable to find matching target for ...linear_attn.in_proj_qkvz`.
    ignore = [
        "lm_head",
        "re:.*\\.visual\\..*",
        "re:.*vision_tower.*",
        "re:.*mm_projector.*",
        "re:.*\\.audio_tower\\..*",
    ]
    if target not in ("mlp_mtp", "non_linear_attn_mtp"):
        # MTP / EAGLE draft head must stay BF16 for all targets EXCEPT
        # the explicit *_mtp targets. sglang builds the draft model with prefix "mtp"
        # (see Qwen3_5ForCausalLMMTP), so its quant-aware Linear modules are
        # named `mtp.layers.0.*` -- mtp at the START with no leading dot. sglang
        # matches ignore via re.match (anchored), so a pattern requiring
        # `\.mtp\.` never matches `mtp.layers...` and every draft Linear would
        # wrongly get the W8A8 scheme with no weight_scale in the ckpt,
        # producing garbage draft logits and ~0.1 accept rate. Use `.*mtp\.`
        # (no required leading dot) so `mtp.layers...` matches while the target
        # body `model.language_model.layers...` (no `mtp.` substring) does not.
        ignore.append("re:.*mtp\\..*")
    if target in ("mlp", "mlp_mtp"):
        ignore.extend(
            [
                "re:.*\\.self_attn\\..*",
                "re:.*\\.linear_attn\\..*",
            ]
        )
        if target == "mlp_mtp":
            ignore.append("re:^mtp\\.fc(\\..*)?$")
    elif target == "non_linear_attn":
        ignore.append("re:.*\\.linear_attn\\..*")
    elif target == "non_linear_attn_mtp":
        # Quantize body + MTP mlp/self_attn; skip only mamba linear_attn. The
        # draft head is intentionally NOT in `ignore` here so its mlp/self_attn
        # get the W8A8 scheme (the ckpt now carries their weight_scale).
        ignore.append("re:.*\\.linear_attn\\..*")
        ignore.append("re:^mtp\\.fc(\\..*)?$")
    return {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "quantization_status": "compressed",
        "global_compression_ratio": None,
        "kv_cache_scheme": None,
        "sparsity_config": None,
        "transform_config": None,
        "version": "local-rtn",
        "config_groups": {
            "group_0": {
                "targets": targets,
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "channel",
                    "dynamic": False,
                    "group_size": None,
                    "observer": "memoryless_minmax",
                    "observer_kwargs": {},
                },
                "input_activations": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "token",
                    "dynamic": True,
                    "group_size": None,
                    "observer": None,
                    "observer_kwargs": {},
                },
                "output_activations": None,
            }
        },
        "ignore": ignore,
    }


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _copy_non_weight_files(model_path: Path, output_path: Path, target: str) -> None:
    for src in model_path.iterdir():
        if not src.is_file():
            continue
        if src.suffix == ".safetensors" or src.name == "model.safetensors.index.json":
            continue
        dst = output_path / src.name
        shutil.copy2(src, dst)

    config_path = output_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{model_path} does not contain config.json")
    config = json.loads(config_path.read_text())
    config["quantization_config"] = build_quantization_config(target)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


@torch.no_grad()
def quantize_checkpoint(model_path: str | Path, output_path: str | Path, target: str, force: bool = False) -> int:
    model_path = Path(model_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path == model_path:
        raise ValueError("--output-path must be different from --model-path")
    if output_path.exists():
        if not force:
            raise FileExistsError(f"{output_path} already exists; pass --force to overwrite it")
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)

    weight_map, shards = _load_weight_map(model_path)
    shard_to_names: dict[str, list[str]] = {shard: [] for shard in shards}
    for name, shard in weight_map.items():
        shard_to_names[shard].append(name)

    out_weight_map: dict[str, str] = {}
    total_size = 0
    quantized_count = 0

    for shard in shards:
        out_tensors: dict[str, torch.Tensor] = {}
        with safe_open(model_path / shard, framework="pt", device="cpu") as f:
            for name in sorted(shard_to_names[shard]):
                tensor = f.get_tensor(name)
                if should_quantize_weight(name, tuple(tensor.shape), target):
                    q, scale, zero_point = quantize_layer_int8(
                        tensor,
                        group_size=None,
                        strategy="channel",
                        sym=True,
                    )
                    if zero_point is not None:
                        raise AssertionError(f"unexpected zero point for symmetric INT8: {name}")
                    out_tensors[name] = q
                    scale_name = name.replace(".weight", ".weight_scale")
                    out_tensors[scale_name] = scale
                    out_weight_map[name] = shard
                    out_weight_map[scale_name] = shard
                    total_size += _tensor_nbytes(q) + _tensor_nbytes(scale)
                    quantized_count += 1
                else:
                    tensor = tensor.contiguous()
                    out_tensors[name] = tensor
                    out_weight_map[name] = shard
                    total_size += _tensor_nbytes(tensor)

        save_file(out_tensors, output_path / shard)
        print(f"[local-rtn] wrote {shard}: tensors={len(out_tensors)} quantized_so_far={quantized_count}", flush=True)

    (output_path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": out_weight_map}, indent=2, sort_keys=True)
        + "\n"
    )
    _copy_non_weight_files(model_path, output_path, target)
    return quantized_count


def main() -> None:
    args = parse_args()
    quantized_count = quantize_checkpoint(args.model_path, args.output_path, args.target, force=args.force)
    print(f"[local-rtn] quantized {quantized_count} tensors; validating saved checkpoint", flush=True)
    checks = check_w8a8_checkpoint(
        args.output_path,
        reference_checkpoint=args.model_path,
        max_tensors=args.validation_max_tensors,
        max_saturated_frac=args.validation_max_saturated_frac,
        max_rel_l2=args.validation_max_rel_l2,
    )
    print(f"[local-rtn] validation OK: checked {len(checks)} tensors", flush=True)


if __name__ == "__main__":
    main()
