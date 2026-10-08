"""Reuse versioned CPU sequence boundaries in FlashQLA 0.1.2's CP planner."""

import argparse
from pathlib import Path

OLD = "    raw_cu_seqlens = raw_cu_seqlens.tolist()"
NEW = (
    "    from slime.utils.sequence_metadata import get_cpu_sequence_boundaries\n\n"
    "    raw_cu_seqlens = get_cpu_sequence_boundaries(raw_cu_seqlens)"
)


def patch_package(root: Path, *, check_only: bool = False):
    if '__version__ = "0.1.2"' not in (root / "__init__.py").read_text():
        raise RuntimeError("The CPU metadata patch requires FlashQLA 0.1.2")
    path = root / "ops/gated_delta_rule/chunk/cp_context.py"
    source = path.read_text()
    counts = source.count(OLD), source.count(NEW)
    if counts not in {(1, 0), (0, 1)}:
        raise RuntimeError(f"Unexpected FlashQLA CP metadata code in {path}: {counts}")
    if check_only and counts != (0, 1):
        raise RuntimeError(f"Missing CPU metadata patch in {path}")
    if not check_only and OLD in source:
        backup = path.with_suffix(path.suffix + ".before-cpu-metadata")
        if not backup.exists():
            backup.write_text(source)
        path.write_text(source.replace(OLD, NEW))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    patch_package(args.path, check_only=args.check)
    print(f"FlashQLA CPU sequence metadata verified: {args.path}")
