#!/usr/bin/env python3
"""Quantify how much the per-group zero-point matters for the W4A16 ASYM checkpoint.

Motivation: the SGLang WNA16 ASYM zero-point patch must be applied for the asym
checkpoint to load correctly. Without it, SGLang loads the asym checkpoint as
symmetric (uint4b8) and dequantizes as (q_raw - 8) * scale instead of the correct
(q_raw - zp_raw) * scale. This probe measures, per MLP weight:

* the distribution of zp_raw (the stored unsigned 0..15 zero point)
* the symmetric-misload weight error vs the correct asym weight
  ||(zp_raw - 8) * scale|| / ||W_asym||                (runtime-bug-induced error)
* both reconstructions vs the BF16 source weight
  ||W_asym - W_bf16|| / ||W_bf16||  and  ||W_sym - W_bf16|| / ||W_bf16||

If zp_raw ~= 8 everywhere, the zero-point barely matters (sym ~= asym). If zp_raw
is systematically off 8, the unpatched runtime adds a large per-group bias.

Pure read-only. Uses torch only for bf16 source loading + norms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def load_weight_map(root: Path) -> dict[str, str]:
    index = root / "model.safetensors.index.json"
    if index.exists():
        return json.loads(index.read_text())["weight_map"]
    single = root / "model.safetensors"
    with safe_open(single, framework="pt", device="cpu") as h:
        return {k: single.name for k in h.keys()}


def get(root: Path, wm: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(root / wm[name], framework="pt", device="cpu") as h:
        return h.get_tensor(name)


def unpack_int32(value: torch.Tensor, num_bits: int, shape, packed_dim: int) -> torch.Tensor:
    pack = 32 // num_bits
    mask = (1 << num_bits) - 1
    shape = tuple(int(d) for d in shape)
    v = value.to(torch.int64)  # arithmetic-shift-safe; & mask isolates the nibble
    if packed_dim == 1:
        out = torch.empty((v.shape[0], v.shape[1] * pack), dtype=torch.int64)
        for i in range(pack):
            out[:, i::pack] = (v >> (num_bits * i)) & mask
        out = out[:, : shape[1]]
    else:
        out = torch.empty((v.shape[0] * pack, v.shape[1]), dtype=torch.int64)
        for i in range(pack):
            out[i::pack, :] = (v >> (num_bits * i)) & mask
        out = out[: shape[0], :]
    return out  # raw unsigned 0..15


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asym", type=Path, default=Path("checkpoints/quantized/AWQ/Qwen3.6-27B-AWQ-W4A16-asym-mlp"))
    ap.add_argument("--source", type=Path, default=Path("/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B"))
    ap.add_argument("--layers", default="0,15,31,47,63")
    ap.add_argument(
        "--output-json",
        type=Path,
        default=Path("checkpoints/quantized/analysis/w4_asym_zp_sensitivity/zp_sensitivity.json"),
    )
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    awm = load_weight_map(args.asym)
    swm = load_weight_map(args.source)

    rows = []
    print(
        f"{'layer.proj':22s} {'zp!=8%':>8s} {'zp_mean':>8s} {'zp_std':>7s} "
        f"{'sym_vs_asym':>11s} {'asym_vs_bf16':>12s} {'sym_vs_bf16':>11s} {'inflation':>9s}"
    )
    for layer in layers:
        for proj in PROJECTIONS:
            base = f"model.language_model.layers.{layer}.mlp.{proj}"
            packed = get(args.asym, awm, f"{base}.weight_packed")
            scale = get(args.asym, awm, f"{base}.weight_scale").float()
            shape = get(args.asym, awm, f"{base}.weight_shape").tolist()
            rows_n, cols = int(shape[0]), int(shape[1])
            groups = scale.shape[-1]
            gsz = cols // groups

            q = unpack_int32(packed, 4, shape, packed_dim=1).view(rows_n, groups, gsz).float()
            zp_raw = unpack_int32(
                get(args.asym, awm, f"{base}.weight_zero_point"), 4, (rows_n, groups), packed_dim=0
            ).float()  # 0..15, per (row, group)

            scl = scale.unsqueeze(-1)
            w_asym = ((q - zp_raw.unsqueeze(-1)) * scl).reshape(rows_n, cols)
            w_sym = ((q - 8.0) * scl).reshape(rows_n, cols)  # uint4b8 misload
            w_bf16 = get(args.source, swm, f"{base}.weight").float()

            def rel(a, b):
                return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b))

            sym_vs_asym = rel(w_sym, w_asym)
            asym_vs_bf16 = rel(w_asym, w_bf16)
            sym_vs_bf16 = rel(w_sym, w_bf16)
            zp_ne8 = float((zp_raw != 8).float().mean()) * 100.0
            row = {
                "layer": layer,
                "proj": proj,
                "zp_ne8_pct": zp_ne8,
                "zp_mean": float(zp_raw.mean()),
                "zp_std": float(zp_raw.std()),
                "zp_min": float(zp_raw.min()),
                "zp_max": float(zp_raw.max()),
                "sym_vs_asym_relL2": sym_vs_asym,
                "asym_vs_bf16_relL2": asym_vs_bf16,
                "sym_vs_bf16_relL2": sym_vs_bf16,
                "bug_inflation_x": sym_vs_bf16 / asym_vs_bf16 if asym_vs_bf16 else float("nan"),
            }
            rows.append(row)
            print(
                f"{f'{layer}.{proj}':22s} {zp_ne8:7.2f}% {row['zp_mean']:8.2f} {row['zp_std']:7.2f} "
                f"{sym_vs_asym:11.4f} {asym_vs_bf16:12.4f} {sym_vs_bf16:11.4f} {row['bug_inflation_x']:8.2f}x"
            )

    # aggregate
    import statistics as st

    agg = {
        "mean_zp_ne8_pct": st.mean(r["zp_ne8_pct"] for r in rows),
        "mean_sym_vs_asym_relL2": st.mean(r["sym_vs_asym_relL2"] for r in rows),
        "mean_asym_vs_bf16_relL2": st.mean(r["asym_vs_bf16_relL2"] for r in rows),
        "mean_sym_vs_bf16_relL2": st.mean(r["sym_vs_bf16_relL2"] for r in rows),
        "mean_bug_inflation_x": st.mean(r["bug_inflation_x"] for r in rows),
    }
    print("\nAGGREGATE:", json.dumps(agg, indent=2))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({"rows": rows, "aggregate": agg, "asym": str(args.asym), "source": str(args.source)}, indent=2)
        + "\n"
    )
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
