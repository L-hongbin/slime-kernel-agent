"""Inject R1-rotated MTP tensors into an already rotated Qwen3.6 checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.quantize.utils.mtp_checkpoint import build_quarot_r1_transform, inject_rotated_mtp_tensors


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-model", required=True, type=Path, help="Original BF16 checkpoint carrying mtp.*")
    ap.add_argument("--checkpoint", required=True, type=Path, help="Rotated BF16 checkpoint to repair")
    ap.add_argument("--hidden-size", type=int, default=5120)
    ap.add_argument(
        "--transform-type",
        choices=["random-hadamard", "hadamard"],
        default="random-hadamard",
        help="Must match the body R1 transform used by rotate_bf16.py",
    )
    ap.add_argument(
        "--final-norm-mode",
        choices=["separate-lm-head", "ratio", "ones", "keep"],
        default="separate-lm-head",
        help="How to handle mtp.norm; separate-lm-head is the exact QuaRot+EAGLE mode",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    transform = build_quarot_r1_transform(args.hidden_size, transform_type=args.transform_type)
    final_norm_mode = args.final_norm_mode.replace("-", "_")
    injected = inject_rotated_mtp_tensors(
        checkpoint_dir=args.checkpoint,
        source_checkpoint=args.source_model,
        transform=transform,
        final_norm_mode=final_norm_mode,
    )
    print(
        f"[repair-mtp-rotation] injected {injected} rotated MTP tensor(s) into {args.checkpoint} "
        f"(final_norm_mode={final_norm_mode}, transform_type={args.transform_type})",
        flush=True,
    )


if __name__ == "__main__":
    main()
