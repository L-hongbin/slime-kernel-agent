#!/usr/bin/env python3
"""Print the safetensors dtype of a representative routed-expert weight.

Checkpoint-family probe: the official mixed DeepSeek-V4-Flash checkpoint
(packed MXFP4 experts, I8) and the secondary uniform-FP8 checkpoint (F8_E4M3)
have byte-identical config.json/model.safetensors.index.json — only per-tensor
dtype distinguishes them. Used by launcher preflights (design doc
handoffs/deepseek-v4/fp4_w4a16_design.md).

Usage: probe_expert_dtype.py CHECKPOINT_DIR [TENSOR_KEY]
Prints the dtype string (e.g. I8, F8_E4M3) on stdout; exits 2 on read failure.
"""

import json
import os
import struct
import sys


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: probe_expert_dtype.py CHECKPOINT_DIR [TENSOR_KEY]", file=sys.stderr)
        return 2
    ckpt = sys.argv[1]
    key = sys.argv[2] if len(sys.argv) > 2 else "layers.3.ffn.experts.0.w1.weight"
    try:
        idx = json.load(open(os.path.join(ckpt, "model.safetensors.index.json")))
        shard = os.path.join(ckpt, idx["weight_map"][key])
        with open(shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        print(header[key]["dtype"])
    except Exception as exc:  # noqa: BLE001 — preflight tool, fail with context
        print(f"probe failed for {ckpt} ({key}): {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
