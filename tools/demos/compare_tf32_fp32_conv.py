#!/usr/bin/env python3
"""Compare CUDA conv outputs with TF32 enabled vs true-fp32/fp64 references.

This is a small reproduction script for the KernelBench TF32 oracle issue:
on cc>=8.0 GPUs, cuDNN fp32 convolutions may use TF32 by default. If that
TF32 output is used as the correctness oracle with a 1e-4 tolerance, true-fp32
custom kernels can be judged wrong for not matching the noisier TF32 result.

Example:
  python tools/compare_tf32_fp32_conv.py
  python tools/compare_tf32_fp32_conv.py --include-matmul
"""

from __future__ import annotations

import argparse
import contextlib
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class ErrorStats:
    max_abs: float
    max_rel: float
    mean_abs: float
    rmse: float
    allclose_1e_4: bool
    allclose_1e_3: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare TF32 conv results against true-fp32 and fp64 references.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--in-channels", type=int, default=64)
    parser.add_argument("--out-channels", type=int, default=64)
    parser.add_argument("--height", type=int, default=56)
    parser.add_argument("--width", type=int, default=56)
    parser.add_argument("--kernel-size", type=int, default=3)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--padding", type=int, default=1)
    parser.add_argument("--dilation", type=int, default=1)
    parser.add_argument("--groups", type=int, default=1)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on. This script is meaningful only on CUDA.",
    )
    parser.add_argument(
        "--include-matmul",
        action="store_true",
        help="Also run a matmul comparison with matmul.allow_tf32 toggled.",
    )
    parser.add_argument("--matmul-m", type=int, default=2048)
    parser.add_argument("--matmul-k", type=int, default=4096)
    parser.add_argument("--matmul-n", type=int, default=2048)
    return parser.parse_args()


@contextlib.contextmanager
def tf32_mode(torch, *, cudnn: bool, matmul: bool) -> Iterator[None]:
    old_cudnn = getattr(torch.backends.cudnn, "allow_tf32", None)
    old_matmul = getattr(torch.backends.cuda.matmul, "allow_tf32", None)
    old_precision = None
    old_global_fp32 = getattr(torch.backends, "fp32_precision", None)
    old_cudnn_fp32 = getattr(torch.backends.cudnn, "fp32_precision", None)
    old_cudnn_conv_fp32 = getattr(getattr(torch.backends.cudnn, "conv", None), "fp32_precision", None)
    old_matmul_fp32 = getattr(torch.backends.cuda.matmul, "fp32_precision", None)
    if hasattr(torch, "get_float32_matmul_precision"):
        old_precision = torch.get_float32_matmul_precision()
    try:
        if old_cudnn is not None:
            torch.backends.cudnn.allow_tf32 = cudnn
        if old_matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = matmul

        # PyTorch 2.9+ is moving from allow_tf32 flags to fp32_precision
        # settings. Set these when available so the script keeps working on
        # newer runtimes while still supporting older ones.
        if old_global_fp32 is not None:
            torch.backends.fp32_precision = "tf32" if (cudnn or matmul) else "ieee"
        if old_cudnn_fp32 is not None:
            torch.backends.cudnn.fp32_precision = "tf32" if cudnn else "ieee"
        cudnn_conv = getattr(torch.backends.cudnn, "conv", None)
        if old_cudnn_conv_fp32 is not None and cudnn_conv is not None:
            cudnn_conv.fp32_precision = "tf32" if cudnn else "ieee"
        if old_matmul_fp32 is not None:
            torch.backends.cuda.matmul.fp32_precision = "tf32" if matmul else "ieee"
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if matmul else "highest")
        yield
    finally:
        if old_cudnn is not None:
            torch.backends.cudnn.allow_tf32 = old_cudnn
        if old_matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = old_matmul
        if old_global_fp32 is not None:
            torch.backends.fp32_precision = old_global_fp32
        if old_cudnn_fp32 is not None:
            torch.backends.cudnn.fp32_precision = old_cudnn_fp32
        cudnn_conv = getattr(torch.backends.cudnn, "conv", None)
        if old_cudnn_conv_fp32 is not None and cudnn_conv is not None:
            cudnn_conv.fp32_precision = old_cudnn_conv_fp32
        if old_matmul_fp32 is not None:
            torch.backends.cuda.matmul.fp32_precision = old_matmul_fp32
        if old_precision is not None and hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(old_precision)


def error_stats(torch, got, ref) -> ErrorStats:
    got64 = got.detach().to(torch.float64)
    ref64 = ref.detach().to(torch.float64)
    diff = got64 - ref64
    abs_diff = diff.abs()
    rel_diff = abs_diff / ref64.abs().clamp_min(1e-12)
    return ErrorStats(
        max_abs=float(abs_diff.max().item()),
        max_rel=float(rel_diff.max().item()),
        mean_abs=float(abs_diff.mean().item()),
        rmse=float(torch.sqrt((diff * diff).mean()).item()),
        allclose_1e_4=bool(torch.allclose(got, ref, rtol=1e-4, atol=1e-4)),
        allclose_1e_3=bool(torch.allclose(got, ref, rtol=1e-3, atol=1e-3)),
    )


def print_stats(title: str, stats: ErrorStats) -> None:
    print(title)
    print(f"  max_abs      {stats.max_abs:.6e}")
    print(f"  max_rel      {stats.max_rel:.6e}")
    print(f"  mean_abs     {stats.mean_abs:.6e}")
    print(f"  rmse         {stats.rmse:.6e}")
    print(f"  allclose 1e-4 {stats.allclose_1e_4}")
    print(f"  allclose 1e-3 {stats.allclose_1e_3}")


def make_conv_inputs(torch, args: argparse.Namespace):
    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed)
    if args.in_channels % args.groups != 0:
        raise ValueError("--in-channels must be divisible by --groups")
    if args.out_channels % args.groups != 0:
        raise ValueError("--out-channels must be divisible by --groups")

    x64 = torch.randn(
        args.batch,
        args.in_channels,
        args.height,
        args.width,
        device=args.device,
        dtype=torch.float64,
        generator=generator,
    )
    w64 = torch.randn(
        args.out_channels,
        args.in_channels // args.groups,
        args.kernel_size,
        args.kernel_size,
        device=args.device,
        dtype=torch.float64,
        generator=generator,
    )
    b64 = torch.randn(
        args.out_channels,
        device=args.device,
        dtype=torch.float64,
        generator=generator,
    )
    return x64, w64, b64


def run_conv(torch, args: argparse.Namespace) -> None:
    import torch.nn.functional as F

    x64, w64, b64 = make_conv_inputs(torch, args)
    x32, w32, b32 = x64.float(), w64.float(), b64.float()
    conv_kwargs = dict(
        stride=args.stride,
        padding=args.padding,
        dilation=args.dilation,
        groups=args.groups,
    )

    with torch.no_grad():
        # The fp64 result is a high-precision reference; cast to fp32 so
        # allclose uses the same dtype scale as KernelBench fp32 checks.
        y64 = F.conv2d(x64, w64, b64, **conv_kwargs).float()
        with tf32_mode(torch, cudnn=False, matmul=False):
            y_fp32 = F.conv2d(x32, w32, b32, **conv_kwargs)
        with tf32_mode(torch, cudnn=True, matmul=True):
            y_tf32 = F.conv2d(x32, w32, b32, **conv_kwargs)
        torch.cuda.synchronize()

    print("\n[conv2d]")
    print(
        "shape "
        f"x={tuple(x32.shape)} w={tuple(w32.shape)} "
        f"stride={args.stride} padding={args.padding} groups={args.groups}"
    )
    print_stats("TF32-on conv vs fp64 reference", error_stats(torch, y_tf32, y64))
    print_stats("TF32-off fp32 conv vs fp64 reference", error_stats(torch, y_fp32, y64))
    print_stats("TF32-on conv vs TF32-off fp32 conv", error_stats(torch, y_tf32, y_fp32))


def run_matmul(torch, args: argparse.Namespace) -> None:
    generator = torch.Generator(device=args.device)
    generator.manual_seed(args.seed + 1)
    a64 = torch.randn(
        args.matmul_m,
        args.matmul_k,
        device=args.device,
        dtype=torch.float64,
        generator=generator,
    )
    b64 = torch.randn(
        args.matmul_k,
        args.matmul_n,
        device=args.device,
        dtype=torch.float64,
        generator=generator,
    )
    a32, b32 = a64.float(), b64.float()

    with torch.no_grad():
        ref = (a64 @ b64).float()
        with tf32_mode(torch, cudnn=False, matmul=False):
            fp32 = a32 @ b32
        with tf32_mode(torch, cudnn=False, matmul=True):
            tf32 = a32 @ b32
        torch.cuda.synchronize()

    print("\n[matmul]")
    print(f"shape a={tuple(a32.shape)} b={tuple(b32.shape)}")
    print_stats("TF32-on matmul vs fp64 reference", error_stats(torch, tf32, ref))
    print_stats("TF32-off fp32 matmul vs fp64 reference", error_stats(torch, fp32, ref))
    print_stats("TF32-on matmul vs TF32-off fp32 matmul", error_stats(torch, tf32, fp32))


def main() -> None:
    args = parse_args()

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; TF32 comparison requires a CUDA GPU.")

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("This script is meaningful only on CUDA devices.")

    print("torch", torch.__version__)
    print("device", torch.cuda.get_device_name(device))
    print("capability", torch.cuda.get_device_capability(device))
    print("initial cudnn.allow_tf32", getattr(torch.backends.cudnn, "allow_tf32", None))
    print(
        "initial matmul.allow_tf32",
        getattr(torch.backends.cuda.matmul, "allow_tf32", None),
    )
    print("initial backends.fp32_precision", getattr(torch.backends, "fp32_precision", None))
    print(
        "initial cudnn.fp32_precision",
        getattr(torch.backends.cudnn, "fp32_precision", None),
    )
    print(
        "initial cudnn.conv.fp32_precision",
        getattr(getattr(torch.backends.cudnn, "conv", None), "fp32_precision", None),
    )
    print(
        "initial cuda.matmul.fp32_precision",
        getattr(torch.backends.cuda.matmul, "fp32_precision", None),
    )
    if hasattr(torch, "get_float32_matmul_precision"):
        print("initial float32_matmul_precision", torch.get_float32_matmul_precision())

    run_conv(torch, args)
    if args.include_matmul:
        run_matmul(torch, args)


if __name__ == "__main__":
    main()
