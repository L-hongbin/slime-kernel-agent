"""Keep an empty MTP supervision mask from producing a zero-token division."""

import argparse
from pathlib import Path

OLD = "                mtp_loss = loss_mask * mtp_loss\n"
NEW = OLD + "                num_tokens = num_tokens.clamp_min(1)\n"


def patch_file(path: Path, *, check_only: bool = False) -> None:
    source = path.read_text()
    if source.count(OLD) != 1 or source.count(NEW) not in (0, 1):
        raise RuntimeError(f"Unexpected MTP loss layout in {path}")
    if check_only and NEW not in source:
        raise RuntimeError("MTP empty-mask protection is missing")
    if not check_only and NEW not in source:
        backup = path.with_suffix(path.suffix + ".before-mtp-empty-mask")
        if not backup.exists():
            backup.write_text(source)
        path.write_text(source.replace(OLD, NEW))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    patch_file(args.path, check_only=args.check)
    print(f"Megatron MTP empty-mask protection verified: {args.path}")
