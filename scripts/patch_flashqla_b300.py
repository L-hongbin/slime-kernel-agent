"""Route SM103 through FlashQLA 0.1.2's Blackwell kernels without changing the native compile target."""

import argparse
from pathlib import Path

OLD = 'elif tilelang.contrib.nvcc.get_target_compute_version() == "10.0":'
NEW = 'elif tilelang.contrib.nvcc.get_target_compute_version() in ("10.0", "10.3"):'


def patch_package(root: Path, *, check_only: bool = False):
    if '__version__ = "0.1.2"' not in (root / "__init__.py").read_text():
        raise RuntimeError("The B300 dispatch patch requires FlashQLA 0.1.2")
    paths = [root / "ops/gated_delta_rule/chunk" / name for name in ("__init__.py", "cp_context.py")]
    sources = [path.read_text() for path in paths]
    for path, source in zip(paths, sources, strict=True):
        counts = source.count(OLD), source.count(NEW)
        if counts not in {(1, 0), (0, 1)}:
            raise RuntimeError(f"Unexpected FlashQLA architecture dispatch in {path}: {counts}")
        if check_only and counts != (0, 1):
            raise RuntimeError(f"Missing SM103 dispatch in {path}")
    if not check_only:
        for path, source in zip(paths, sources, strict=True):
            if OLD in source:
                backup = path.with_suffix(path.suffix + ".before-b300")
                if not backup.exists():
                    backup.write_text(source)
                path.write_text(source.replace(OLD, NEW))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    patch_package(args.path, check_only=args.check)
    print(f"FlashQLA 0.1.2 SM103 dispatch verified: {args.path}")
