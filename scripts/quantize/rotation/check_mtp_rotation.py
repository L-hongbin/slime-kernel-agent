"""Static sanity check for QuaRot-rotated Qwen3.5/3.6 MTP tensors.

This catches the cheap failures before launching a full SGLang eval:
- missing ``mtp.*`` tensors after HF ``save_pretrained``
- MTP tensors copied back unrotated instead of R1-rotated
- missing exact ``mtp.lm_head.weight`` for QuaRot+EAGLE
- draft MTP head accidentally identical to the target shared head
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors import safe_open

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.quantize.utils.mtp_checkpoint import (  # noqa: E402
    _gemma_effective_norm,
    _load_weight_map,
    _read_prefixed_tensors,
    build_quarot_r1_transform,
    rotate_qwen35_mtp_tensors,
)


REQUIRED_MTP_KEYS = (
    "mtp.fc.weight",
    "mtp.pre_fc_norm_embedding.weight",
    "mtp.pre_fc_norm_hidden.weight",
    "mtp.layers.0.input_layernorm.weight",
    "mtp.layers.0.self_attn.q_proj.weight",
    "mtp.layers.0.self_attn.k_proj.weight",
    "mtp.layers.0.self_attn.v_proj.weight",
    "mtp.layers.0.self_attn.o_proj.weight",
    "mtp.layers.0.post_attention_layernorm.weight",
    "mtp.layers.0.mlp.gate_proj.weight",
    "mtp.layers.0.mlp.up_proj.weight",
    "mtp.layers.0.mlp.down_proj.weight",
    "mtp.norm.weight",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, type=Path, help="Rotated checkpoint to check")
    ap.add_argument("--source-model", required=True, type=Path, help="Original BF16 checkpoint carrying source mtp.*")
    ap.add_argument("--hidden-size", type=int, default=5120)
    ap.add_argument(
        "--transform-type",
        choices=["random-hadamard", "hadamard"],
        default="random-hadamard",
        help="Must match rotate_bf16.py",
    )
    ap.add_argument(
        "--final-norm-mode",
        choices=["separate-lm-head", "ratio", "ones", "keep"],
        default="separate-lm-head",
    )
    ap.add_argument("--head-sample-rows", type=int, default=16)
    ap.add_argument("--rtol", type=float, default=2e-2)
    ap.add_argument("--atol", type=float, default=2e-2)
    ap.add_argument("--max-int8-rel-l2", type=float, default=0.10)
    return ap.parse_args()


def _read_tensor(checkpoint: Path, name: str) -> torch.Tensor:
    weight_map = _load_weight_map(checkpoint)
    shard = weight_map[name]
    with safe_open(checkpoint / shard, framework="pt", device="cpu") as f:
        return f.get_tensor(name).contiguous()


def _read_rows(checkpoint: Path, name: str, rows: int) -> torch.Tensor:
    weight_map = _load_weight_map(checkpoint)
    shard = weight_map[name]
    with safe_open(checkpoint / shard, framework="pt", device="cpu") as f:
        tensor_slice = f.get_slice(name)
        shape = tensor_slice.get_shape()
        if len(shape) != 2:
            raise ValueError(f"{name} is not 2D: shape={shape}")
        return tensor_slice[: min(rows, shape[0]), :].contiguous()


def _iter_existing_names(checkpoint: Path, weight_map: dict[str, str], names: list[str]):
    for name in names:
        if name in weight_map:
            yield name, _read_tensor(checkpoint, name)


def _read_weight_for_compare(checkpoint: Path, name: str, weight_map: dict[str, str]) -> tuple[torch.Tensor, bool]:
    tensor = _read_tensor(checkpoint, name)
    if tensor.dtype != torch.int8:
        return tensor, False

    scale_name = name.replace(".weight", ".weight_scale")
    if scale_name not in weight_map:
        raise KeyError(f"{name} is int8 but {scale_name} is missing")
    scale = _read_tensor(checkpoint, scale_name).float()
    return tensor.float() * scale, True


def _close_or_record(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    failures: list[str],
    rtol: float,
    atol: float,
    max_int8_rel_l2: float,
    is_int8: bool = False,
) -> None:
    actual_f = actual.float()
    expected_f = expected.to(dtype=actual.dtype).float()
    if is_int8:
        rel_l2 = torch.linalg.vector_norm(actual_f - expected.float()) / torch.linalg.vector_norm(
            expected.float()
        ).clamp(min=1e-6)
        rel_l2_value = rel_l2.item()
        if rel_l2_value <= max_int8_rel_l2:
            return
        failures.append(f"{name}: int8 dequant rel_l2={rel_l2_value:.6g} > {max_int8_rel_l2:.6g}")
        return

    if torch.allclose(actual_f, expected_f, rtol=rtol, atol=atol):
        return
    max_abs = (actual_f - expected_f).abs().max().item()
    denom = expected_f.abs().clamp(min=1e-6)
    max_rel = ((actual_f - expected_f).abs() / denom).max().item()
    failures.append(f"{name}: mismatch max_abs={max_abs:.6g} max_rel={max_rel:.6g}")


def main() -> None:
    args = parse_args()
    final_norm_mode = args.final_norm_mode.replace("-", "_")
    weight_map = _load_weight_map(args.checkpoint)
    source_weight_map = _load_weight_map(args.source_model)
    failures: list[str] = []

    required = list(REQUIRED_MTP_KEYS)
    if final_norm_mode == "separate_lm_head":
        required.append("mtp.lm_head.weight")
    missing = [name for name in required if name not in weight_map]
    if missing:
        raise SystemExit(f"[check-mtp-rotation] missing required checkpoint tensors: {missing}")
    source_missing = [name for name in REQUIRED_MTP_KEYS if name not in source_weight_map]
    if "lm_head.weight" not in source_weight_map:
        source_missing.append("lm_head.weight")
    if source_missing:
        raise SystemExit(f"[check-mtp-rotation] missing required source tensors: {source_missing}")

    transform = build_quarot_r1_transform(args.hidden_size, transform_type=args.transform_type)
    body_required = [
        "model.language_model.embed_tokens.weight",
        "model.language_model.norm.weight",
        "lm_head.weight",
    ]
    missing_body = [name for name in body_required if name not in weight_map or name not in source_weight_map]
    if missing_body:
        raise SystemExit(f"[check-mtp-rotation] missing required body tensors: {missing_body}")

    source_embed_rows = _read_rows(
        args.source_model, "model.language_model.embed_tokens.weight", args.head_sample_rows
    )
    actual_embed_rows = _read_rows(args.checkpoint, "model.language_model.embed_tokens.weight", args.head_sample_rows)
    expected_embed_rows = source_embed_rows.to(device=transform.device, dtype=torch.float64) @ transform
    _close_or_record(
        "model.language_model.embed_tokens.weight[:sample]",
        actual_embed_rows,
        expected_embed_rows.cpu(),
        failures,
        args.rtol,
        args.atol,
        args.max_int8_rel_l2,
    )

    source_body_head_rows = _read_rows(args.source_model, "lm_head.weight", args.head_sample_rows)
    actual_body_head_rows = _read_rows(args.checkpoint, "lm_head.weight", args.head_sample_rows)
    source_body_norm = _read_tensor(args.source_model, "model.language_model.norm.weight")
    expected_body_head_rows = (
        source_body_head_rows.to(device=transform.device, dtype=torch.float64)
        * _gemma_effective_norm(source_body_norm).to(device=transform.device, dtype=torch.float64).view(1, -1)
    ) @ transform
    _close_or_record(
        "lm_head.weight[:sample]",
        actual_body_head_rows,
        expected_body_head_rows.cpu(),
        failures,
        args.rtol,
        args.atol,
        args.max_int8_rel_l2,
    )

    body_identity_names = [
        "model.language_model.norm.weight",
        *[
            f"model.language_model.layers.{idx}.{norm}.weight"
            for idx in range(64)
            for norm in ("input_layernorm", "post_attention_layernorm")
        ],
    ]
    mtp_identity_names = [
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.layers.0.input_layernorm.weight",
        "mtp.layers.0.post_attention_layernorm.weight",
        "mtp.norm.weight",
    ]
    for name, actual in _iter_existing_names(args.checkpoint, weight_map, body_identity_names + mtp_identity_names):
        _close_or_record(
            name,
            actual,
            torch.zeros_like(actual),
            failures,
            args.rtol,
            args.atol,
            args.max_int8_rel_l2,
        )

    source_mtp = _read_prefixed_tensors(args.source_model, "mtp.")
    body_norm = None
    if final_norm_mode == "ratio":
        body_norm = _read_tensor(args.source_model, "model.language_model.norm.weight")

    linears_mode = "ones" if final_norm_mode == "separate_lm_head" else final_norm_mode
    expected_mtp = rotate_qwen35_mtp_tensors(
        source_mtp,
        transform=transform,
        body_final_norm=body_norm,
        final_norm_mode=linears_mode,
    )
    for name, expected in sorted(expected_mtp.items()):
        if name == "mtp.lm_head.weight":
            continue
        actual, is_int8 = _read_weight_for_compare(args.checkpoint, name, weight_map)
        _close_or_record(name, actual, expected, failures, args.rtol, args.atol, args.max_int8_rel_l2, is_int8)

    if final_norm_mode == "separate_lm_head":
        source_head_rows = _read_rows(args.source_model, "lm_head.weight", args.head_sample_rows)
        actual_mtp_head_rows = _read_rows(args.checkpoint, "mtp.lm_head.weight", args.head_sample_rows)
        norm = _gemma_effective_norm(source_mtp["mtp.norm.weight"]).to(device=transform.device, dtype=torch.float64)
        expected_head_rows = (
            source_head_rows.to(device=transform.device, dtype=torch.float64) * norm.view(1, -1)
        ) @ transform
        _close_or_record(
            "mtp.lm_head.weight[:sample]",
            actual_mtp_head_rows,
            expected_head_rows.cpu(),
            failures,
            args.rtol,
            args.atol,
            args.max_int8_rel_l2,
        )

        target_head_rows = _read_rows(args.checkpoint, "lm_head.weight", args.head_sample_rows)
        if torch.allclose(actual_mtp_head_rows.float(), target_head_rows.float(), rtol=1e-4, atol=1e-4):
            failures.append(
                "mtp.lm_head.weight[:sample] is identical to target lm_head.weight; SGLang will share the wrong head"
            )

    if failures:
        joined = "\n  - ".join(failures)
        raise SystemExit(f"[check-mtp-rotation] FAILED\n  - {joined}")

    print(
        f"[check-mtp-rotation] OK checkpoint={args.checkpoint} mode={final_norm_mode} "
        f"transform={args.transform_type} checked_mtp_tensors={len(expected_mtp)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
