#!/usr/bin/env python3
"""Make MTP hidden-state gradient isolation independent of tensor view layout."""

import argparse
from pathlib import Path

OLD = """        hidden_states = make_viewless_tensor(
            inp=hidden_states, requires_grad=True, keep_graph=False
        )

        return input_ids, position_ids, decoder_input, hidden_states
"""
NEW = OLD.replace("inp=hidden_states,", "inp=hidden_states.detach(),")


def patch_file(path: Path, *, check_only: bool = False) -> None:
    source = path.read_text()
    counts = source.count(OLD), source.count(NEW)
    if counts not in {(1, 0), (0, 1)}:
        raise RuntimeError(f"Unexpected MTP embedding preprocessing layout: {counts}")
    if "decoder_input = decoder_input.detach()" not in source:
        raise RuntimeError("MTP embedding gradient isolation is missing")
    if check_only and counts != (0, 1):
        raise RuntimeError("MTP hidden-state detach is not installed")
    if not check_only and counts == (1, 0):
        path.write_text(source.replace(OLD, NEW))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    patch_file(args.path, check_only=args.check)
    print(f"MTP hidden-state detach verified: {args.path}")
