#!/usr/bin/env python3
"""Numerically verify SGLang's patched WNA16 ASYM runtime dequant == offline dequant_w4.

Closes the last open link in handoff_w4a16_awq.md: all "loss" numbers (39.86 etc.)
come from the offline dequant_w4; this checks that the REAL SGLang marlin INT4
kernel, after the WNA16 ASYM zero-point patch, reconstructs the same effective
weight.

Method: build one CompressedTensorsWNA16 group/asym linear via SGLang, load the
checkpoint's MLP linear params, run process_weights_after_loading (marlin repack),
then read back the effective weight W_eff by feeding an fp16 identity:
    y = I @ W_eff^T  ->  W_eff = y^T
Compare W_eff to:
  * offline asym dequant  (q-zp)*scale   -- should match (rel-L2 ~ fp tolerance)
  * offline sym misload   (q-8)*scale    -- should NOT match (confirms patch active)

Run on .16 with the patch applied. Needs GPU (marlin is sm80+).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path("/nfs/FM/chenshuailin/projects/kernel_agents/slime")
sys.path.insert(0, str(REPO))
sys.path.insert(0, "/sgl-workspace/sglang/python")

import scripts.debug.probe_w4_asym_zp_sensitivity as P
from scripts.quantize.utils.compare_mlp_quant_loss import dequant_w4, load_tensor, load_weight_map
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import CompressedTensorsWNA16


def build_runtime_weight(asym: Path, awm: dict, base: str, device="cuda") -> torch.Tensor:
    packed = load_tensor(asym, awm, f"{base}.weight_packed")  # (N, K/8) int32
    scale = load_tensor(asym, awm, f"{base}.weight_scale")  # (N, K/128)
    zp = load_tensor(asym, awm, f"{base}.weight_zero_point")  # (N/8, K/128) int32
    shape = load_tensor(asym, awm, f"{base}.weight_shape")  # [N, K]
    N, K = int(shape[0]), int(shape[1])
    gsz = K // scale.shape[-1]
    params_dtype = torch.float16

    scheme = CompressedTensorsWNA16(strategy="group", num_bits=4, group_size=gsz, symmetric=False, actorder=None)
    layer = torch.nn.Module()
    scheme.create_weights(
        layer,
        output_size=N,
        input_size=K,
        output_partition_sizes=[N],
        input_size_per_partition=K,
        params_dtype=params_dtype,
        weight_loader=lambda *a, **k: None,
    )
    # populate created params with the on-disk tensors
    layer.weight_packed.data.copy_(packed)
    layer.weight_scale.data.copy_(scale.to(params_dtype))
    layer.weight_zero_point.data.copy_(zp)
    layer.weight_shape.data.copy_(shape.to(torch.int64))
    layer.to(device)
    scheme.process_weights_after_loading(layer)

    # identity readout of the effective weight the marlin kernel applies
    eye = torch.eye(K, dtype=params_dtype, device=device)
    with torch.no_grad():
        y = scheme.apply_weights(layer, eye, bias=None)  # (K, N) = W_eff^T
    return y.t().contiguous().float(), N, K, gsz  # (N, K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asym", type=Path, default=REPO / "checkpoints/quantized/AWQ/Qwen3.6-27B-AWQ-W4A16-asym-mlp")
    ap.add_argument("--layers", default="0,31,63")
    ap.add_argument("--projs", default="gate_proj,up_proj,down_proj")
    ap.add_argument(
        "--output-json",
        type=Path,
        default=REPO / "checkpoints/quantized/analysis/w4_asym_zp_sensitivity/runtime_fidelity.json",
    )
    args = ap.parse_args()
    awm = load_weight_map(args.asym)

    def rel(a, b):
        return float(torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(b))

    rows = []
    print(
        f"{'layer.proj':18s} {'shape(N,K)':>14s} {'runtime_vs_ASYM':>16s} {'runtime_vs_SYMmisload':>22s} {'verdict':>10s}"
    )
    for layer in [int(x) for x in args.layers.split(",")]:
        for proj in args.projs.split(","):
            base = f"model.language_model.layers.{layer}.mlp.{proj}"
            w_rt, N, K, gsz = build_runtime_weight(args.asym, awm, base)
            w_asym = dequant_w4(args.asym, awm, base).float().to(w_rt.device)  # (q-zp)*scale
            # symmetric misload reference
            packed = load_tensor(args.asym, awm, f"{base}.weight_packed")
            scale = load_tensor(args.asym, awm, f"{base}.weight_scale").float()
            shape = load_tensor(args.asym, awm, f"{base}.weight_shape").tolist()
            groups = scale.shape[-1]
            q = P.unpack_int32(packed, 4, shape, 1).view(N, groups, K // groups).float()
            w_sym = ((q - 8.0) * scale.unsqueeze(-1)).reshape(N, K).to(w_rt.device)

            r_asym = rel(w_rt, w_asym)
            r_sym = rel(w_rt, w_sym)
            verdict = (
                "MATCH-ASYM" if r_asym < 0.05 and r_asym < r_sym else ("MATCH-SYM!" if r_sym < 0.05 else "DIVERGE")
            )
            rows.append(
                {
                    "layer": layer,
                    "proj": proj,
                    "N": N,
                    "K": K,
                    "runtime_vs_asym_relL2": r_asym,
                    "runtime_vs_sym_relL2": r_sym,
                    "verdict": verdict,
                }
            )
            print(f"{f'{layer}.{proj}':18s} {f'({N},{K})':>14s} {r_asym:16.5f} {r_sym:22.5f} {verdict:>10s}")
            del w_rt, w_asym, w_sym
            torch.cuda.empty_cache()

    agg = {
        "max_runtime_vs_asym_relL2": max(r["runtime_vs_asym_relL2"] for r in rows),
        "all_match_asym": all(r["verdict"] == "MATCH-ASYM" for r in rows),
    }
    print("\nAGG:", json.dumps(agg, indent=2))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"rows": rows, "aggregate": agg}, indent=2) + "\n")
    print("wrote", args.output_json)


if __name__ == "__main__":
    main()
