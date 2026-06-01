"""Static gate for AWQ W4A16 checkpoints before rollout/eval.

This catches the cheap failures first: wrong compressed-tensors metadata,
missing packed INT4 MLP tensors, and accidental quantization of self-attn,
linear-attn, lm_head, or MTP weights.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.quantize.producers.awq_w4a16 import (
    MLP_TARGET_RE,
    awq_targets_and_ignore,
    should_restore_awq_preserved_tensor,
)


@dataclass(frozen=True)
class AwqW4A16Report:
    quantized_mlp: int
    quantized_total: int
    mtp_tensors: int
    preserved_checked: int = 0
    preserved_sampled: int = 0


def _load_weight_names(checkpoint_dir: Path) -> set[str]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        return set(index["weight_map"])

    safetensors_files = sorted(checkpoint_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"no safetensors files found under {checkpoint_dir}")

    names: set[str] = set()
    for path in safetensors_files:
        with safe_open(path, framework="pt", device="cpu") as f:
            names.update(f.keys())
    return names


def _load_weight_map(checkpoint_dir: Path) -> dict[str, str]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        return dict(index["weight_map"])

    safetensors_files = sorted(checkpoint_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"no safetensors files found under {checkpoint_dir}")
    if len(safetensors_files) != 1:
        raise FileNotFoundError(f"{checkpoint_dir} has multiple safetensors files but no index")
    with safe_open(safetensors_files[0], framework="pt", device="cpu") as f:
        return {name: safetensors_files[0].name for name in f.keys()}


def _load_tensor(checkpoint_dir: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(checkpoint_dir / weight_map[name], framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def _numel(shape: list[int]) -> int:
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


def _sample_slices(shape: list[int], *, max_sample_elements: int = 8192) -> list[tuple[slice, ...]]:
    if not shape:
        return [()]

    if _numel(shape) <= max_sample_elements:
        return [tuple(slice(None) for _ in shape)]

    first_dim = shape[0]
    if len(shape) == 1:
        chunk = min(first_dim, max_sample_elements)
        starts = [0, max(0, first_dim // 2 - chunk // 2), max(0, first_dim - chunk)]
        seen: set[int] = set()
        slices = []
        for start in starts:
            if start in seen:
                continue
            seen.add(start)
            slices.append((slice(start, start + chunk),))
        return slices

    tail_shape = shape[1:]
    rows = [0, first_dim // 2, first_dim - 1]
    unique_rows = []
    seen_rows: set[int] = set()
    for row in rows:
        if row < 0 or row in seen_rows:
            continue
        seen_rows.add(row)
        unique_rows.append(row)

    if _numel(tail_shape) <= max_sample_elements:
        tail_slices = [tuple(slice(None) for _ in tail_shape)]
    else:
        tail_slices = _sample_slices(tail_shape, max_sample_elements=max_sample_elements)

    return [(slice(row, row + 1), *tail) for row in unique_rows for tail in tail_slices]


def _sampled_tensor_drift(
    checkpoint_dir: Path,
    reference_checkpoint: Path,
    ckpt_map: dict[str, str],
    ref_map: dict[str, str],
    name: str,
    shape: list[int],
    *,
    max_preserved_rel_l2: float,
) -> str | None:
    selectors = _sample_slices(shape)
    with safe_open(checkpoint_dir / ckpt_map[name], framework="pt", device="cpu") as cur_file:
        cur_slice = cur_file.get_slice(name)
        with safe_open(reference_checkpoint / ref_map[name], framework="pt", device="cpu") as ref_file:
            ref_slice = ref_file.get_slice(name)
            for selector in selectors:
                cur = cur_slice[selector].float()
                ref = ref_slice[selector].float()
                if torch.equal(cur, ref):
                    continue
                denom = ref.norm().clamp(min=1e-12)
                rel_l2 = ((cur - ref).norm() / denom).item()
                if rel_l2 > max_preserved_rel_l2:
                    return f"{name}: sampled_rel_l2={rel_l2:.6g}, slice={selector}"
    return None


def _quantized_weight_bases(names: set[str]) -> set[str]:
    bases: set[str] = set()
    for name in names:
        if name.endswith(".weight_packed"):
            bases.add(name.removesuffix(".weight_packed") + ".weight")
        elif name.endswith(".qweight"):
            bases.add(name.removesuffix(".qweight") + ".weight")
    return bases


def _load_quantization_config(checkpoint_dir: Path) -> dict:
    config_path = checkpoint_dir / "config.json"
    cfg = json.loads(config_path.read_text())
    qcfg = cfg.get("quantization_config")
    if not isinstance(qcfg, dict):
        raise AssertionError(f"{config_path} has no quantization_config")
    return qcfg


def _group_weight_configs(qcfg: dict) -> list[dict]:
    groups = qcfg.get("config_groups") or {}
    if not isinstance(groups, dict) or not groups:
        raise AssertionError("quantization_config.config_groups is missing or empty")
    weights = []
    for name, group in groups.items():
        wcfg = group.get("weights")
        if not isinstance(wcfg, dict):
            raise AssertionError(f"{name} has no weights config")
        weights.append(wcfg)
    return weights


def _sglang_compressed_tensors_paths(sglang_source_root: Path) -> tuple[Path, Path]:
    candidates = [
        sglang_source_root / "sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py",
        sglang_source_root / "python/sglang/srt/layers/quantization/compressed_tensors/compressed_tensors.py",
    ]
    dispatch_path = next((path for path in candidates if path.exists()), candidates[0])
    scheme_path = dispatch_path.parent / "schemes/compressed_tensors_wNa16.py"
    return dispatch_path, scheme_path


def check_sglang_wna16_asym_support(sglang_source_root: Path) -> None:
    dispatch_path, scheme_path = _sglang_compressed_tensors_paths(sglang_source_root)
    if not dispatch_path.exists():
        raise AssertionError(f"SGLang compressed-tensors dispatcher not found: {dispatch_path}")
    if not scheme_path.exists():
        raise AssertionError(f"SGLang WNA16 scheme not found: {scheme_path}")

    dispatch = dispatch_path.read_text()
    scheme = scheme_path.read_text()
    if "symmetric=weight_quant.symmetric" not in dispatch:
        raise AssertionError(
            "SGLang WNA16 dispatcher does not pass weight_quant.symmetric into "
            "CompressedTensorsWNA16; asymmetric checkpoints may be loaded as symmetric"
        )
    if "return is_channel_group and input_quant_none and is_symmetric and is_static" in dispatch:
        raise AssertionError(
            "SGLang WNA16 dispatcher still rejects asymmetric weight quantization " "in _is_wNa16_group_channel"
        )
    for required in (
        "WNA16_ZP_SUPPORTED_TYPES_MAP",
        "if not self.symmetric",
        '"weight_zero_point"',
    ):
        if required not in scheme:
            raise AssertionError(f"SGLang WNA16 scheme missing asymmetric zero-point support: {required}")


def check_awq_w4a16_checkpoint(
    checkpoint_dir: Path,
    *,
    min_quantized_mlp: int = 1,
    require_mtp: bool = False,
    require_symmetric: bool = False,
    require_sglang_asym_support: bool = False,
    sglang_source_root: Path | None = None,
    reference_checkpoint: Path | None = None,
    max_preserved_rel_l2: float = 1e-7,
    max_preserved_elements: int = 50_000_000,
) -> AwqW4A16Report:
    qcfg = _load_quantization_config(checkpoint_dir)
    if qcfg.get("quant_method") != "compressed-tensors":
        raise AssertionError(f"quant_method={qcfg.get('quant_method')!r}, expected 'compressed-tensors'")

    for wcfg in _group_weight_configs(qcfg):
        if wcfg.get("num_bits") != 4:
            raise AssertionError(f"weights.num_bits={wcfg.get('num_bits')!r}, expected 4")
        if require_symmetric and wcfg.get("symmetric") is not True:
            raise AssertionError(
                "asymmetric W4A16 compressed-tensors weights are not allowed "
                "for this SGLang eval path; expected weights.symmetric=true"
            )
        if require_sglang_asym_support and wcfg.get("symmetric") is False:
            check_sglang_wna16_asym_support(
                sglang_source_root or Path(os.environ.get("SGLANG_SOURCE_ROOT", "/sgl-workspace/sglang/python"))
            )

    targets, ignore = awq_targets_and_ignore()
    group_targets = [
        target for group in (qcfg.get("config_groups") or {}).values() for target in (group.get("targets") or [])
    ]
    if MLP_TARGET_RE not in group_targets:
        raise AssertionError(f"MLP target regex missing from quantization_config: {targets}")
    missing_ignore = [item for item in ignore if item not in (qcfg.get("ignore") or [])]
    if missing_ignore:
        raise AssertionError(f"quantization_config.ignore missing entries: {missing_ignore}")

    names = _load_weight_names(checkpoint_dir)
    q_bases = _quantized_weight_bases(names)
    if not q_bases:
        raise AssertionError("no packed INT4 weights found; expected .weight_packed or .qweight tensors")

    mlp_re = re.compile(r".*\.mlp\.(gate_proj|up_proj|down_proj|gate_up_proj)\.weight$")
    q_mlp = sorted(name for name in q_bases if mlp_re.match(name))
    if len(q_mlp) < min_quantized_mlp:
        raise AssertionError(f"only {len(q_mlp)} quantized MLP weights; expected at least {min_quantized_mlp}")

    forbidden = [
        name
        for name in sorted(q_bases)
        if ".self_attn." in name
        or ".linear_attn." in name
        or name.startswith("mtp.")
        or name == "lm_head.weight"
        or name.endswith(".lm_head.weight")
    ]
    if forbidden:
        shown = ", ".join(forbidden[:10])
        raise AssertionError(f"forbidden W4A16 quantized weights found: {shown}")

    mtp_tensors = sum(1 for name in names if name.startswith("mtp."))
    if require_mtp and mtp_tensors == 0:
        raise AssertionError("require_mtp=True but no mtp.* tensors are present")

    preserved_checked = 0
    preserved_sampled = 0
    if reference_checkpoint is not None:
        ref_map = _load_weight_map(reference_checkpoint)
        ckpt_map = _load_weight_map(checkpoint_dir)
        drifted: list[str] = []
        for name in sorted(ckpt_map):
            if name not in ref_map or not should_restore_awq_preserved_tensor(name):
                continue
            with safe_open(checkpoint_dir / ckpt_map[name], framework="pt", device="cpu") as f:
                tensor_slice = f.get_slice(name)
                shape = list(tensor_slice.get_shape())
                numel = _numel(shape)
            if numel > max_preserved_elements:
                preserved_checked += 1
                preserved_sampled += 1
                sampled_drift = _sampled_tensor_drift(
                    checkpoint_dir,
                    reference_checkpoint,
                    ckpt_map,
                    ref_map,
                    name,
                    shape,
                    max_preserved_rel_l2=max_preserved_rel_l2,
                )
                if sampled_drift:
                    drifted.append(sampled_drift)
                continue

            ref = _load_tensor(reference_checkpoint, ref_map, name).float()
            cur = _load_tensor(checkpoint_dir, ckpt_map, name).float()
            denom = ref.norm().clamp(min=1e-12)
            rel_l2 = ((cur - ref).norm() / denom).item()
            preserved_checked += 1
            if rel_l2 > max_preserved_rel_l2:
                drifted.append(f"{name}: rel_l2={rel_l2:.6g}")

        if drifted:
            shown = "; ".join(drifted[:10])
            raise AssertionError(
                "non-target BF16 tensors drifted from reference checkpoint; "
                f"checked={preserved_checked}, examples: {shown}"
            )

    return AwqW4A16Report(
        quantized_mlp=len(q_mlp),
        quantized_total=len(q_bases),
        mtp_tensors=mtp_tensors,
        preserved_checked=preserved_checked,
        preserved_sampled=preserved_sampled,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--min-quantized-mlp", type=int, default=1)
    parser.add_argument("--require-mtp", action="store_true")
    parser.add_argument(
        "--require-symmetric",
        action="store_true",
        help="Reject W4A16_ASYM checkpoints before SGLang eval.",
    )
    parser.add_argument(
        "--require-sglang-asym-support",
        action="store_true",
        help="For asymmetric checkpoints, require the local SGLang WNA16 zero-point runtime patch.",
    )
    parser.add_argument(
        "--sglang-source-root",
        type=Path,
        default=Path(os.environ.get("SGLANG_SOURCE_ROOT", "/sgl-workspace/sglang/python")),
    )
    parser.add_argument("--reference-checkpoint", type=Path)
    parser.add_argument("--max-preserved-rel-l2", type=float, default=1e-7)
    parser.add_argument("--max-preserved-elements", type=int, default=50_000_000)
    args = parser.parse_args()

    report = check_awq_w4a16_checkpoint(
        args.checkpoint,
        min_quantized_mlp=args.min_quantized_mlp,
        require_mtp=args.require_mtp,
        require_symmetric=args.require_symmetric,
        require_sglang_asym_support=args.require_sglang_asym_support,
        sglang_source_root=args.sglang_source_root,
        reference_checkpoint=args.reference_checkpoint,
        max_preserved_rel_l2=args.max_preserved_rel_l2,
        max_preserved_elements=args.max_preserved_elements,
    )
    print(
        "OK: AWQ W4A16 checkpoint "
        f"{args.checkpoint} has {report.quantized_mlp} quantized MLP weights, "
        f"{report.quantized_total} quantized weights total, "
        f"{report.mtp_tensors} mtp tensor(s), "
        f"{report.preserved_checked} preserved tensor(s) checked "
        f"({report.preserved_sampled} sampled large tensor(s))."
    )


if __name__ == "__main__":
    main()
