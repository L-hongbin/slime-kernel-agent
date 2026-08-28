#!/usr/bin/env python3
"""Stop audited FLA helper kernels recompiling for every varlen token count.

The deployed FLA 0.4.1/0.4.2 source layouts pass ``NB`` (a bucket derived from
the current token count) as an autotune key and a ``tl.constexpr`` to their
gated RMSNorm and short-convolution kernels. The kernel bodies never read
``NB``. Specializing on it therefore generates and autotunes identical Triton
programs for nearly every dynamic training microbatch.

This fail-closed source patch removes only that unused specialization input.
The candidate configurations, launch grids, tensor shapes, and arithmetic are
unchanged. Unknown or partially modified source is rejected.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


DEFAULT_FUSED_NORM_PATH = Path("/usr/local/lib/python3.12/dist-packages/fla/modules/fused_norm_gate.py")
DEFAULT_CONVOLUTION_PATH = Path("/usr/local/lib/python3.12/dist-packages/fla/modules/convolution.py")


@dataclass(frozen=True)
class Replacement:
    old: str
    new: str
    count: int


@dataclass(frozen=True)
class SourceLayout:
    name: str
    replacements: tuple[Replacement, ...]
    marker: str
    marker_count: int = 2


_KERNEL_MARKER = "# slime: NB is unused; do not specialize varlen kernels by token-count bucket."
_FUSED_NORM_MARKER = _KERNEL_MARKER
_CONVOLUTION_MARKER = _KERNEL_MARKER
_CONVOLUTION_OPS_MARKER = "# slime: the Triton kernel has no unused NB specialization argument."

_FUSED_NORM_REPLACEMENTS = (
    Replacement(
        "    key=['D', 'NB', 'IS_RMS_NORM', 'STORE_RESIDUAL_OUT', 'HAS_RESIDUAL', 'HAS_WEIGHT'],",
        f"    {_FUSED_NORM_MARKER}\n"
        "    key=['D', 'IS_RMS_NORM', 'STORE_RESIDUAL_OUT', 'HAS_RESIDUAL', 'HAS_WEIGHT'],",
        1,
    ),
    Replacement(
        "    key=['D', 'NB', 'IS_RMS_NORM', 'HAS_DRESIDUAL', 'HAS_WEIGHT'],",
        f"    {_FUSED_NORM_MARKER}\n" "    key=['D', 'IS_RMS_NORM', 'HAS_DRESIDUAL', 'HAS_WEIGHT'],",
        1,
    ),
    Replacement("    NB: tl.constexpr,\n", "", 2),
    Replacement("        NB = triton.cdiv(T, 2048)\n", "", 2),
    Replacement("            NB=NB,\n", "", 2),
)

_FUSED_NORM_REPLACEMENTS_042 = (
    Replacement(
        '    key=["D", "NB", "IS_RMS_NORM", "STORE_RESIDUAL_OUT", "HAS_RESIDUAL", "HAS_WEIGHT"],',
        f"    {_FUSED_NORM_MARKER}\n"
        '    key=["D", "IS_RMS_NORM", "STORE_RESIDUAL_OUT", "HAS_RESIDUAL", "HAS_WEIGHT"],',
        1,
    ),
    Replacement(
        '    key=["D", "NB", "IS_RMS_NORM", "HAS_DRESIDUAL", "HAS_WEIGHT"],',
        f"    {_FUSED_NORM_MARKER}\n" '    key=["D", "IS_RMS_NORM", "HAS_DRESIDUAL", "HAS_WEIGHT"],',
        1,
    ),
    Replacement("    NB: tl.constexpr,\n", "", 2),
    Replacement("        NB = triton.cdiv(T, 2048 * 32)\n", "", 2),
    Replacement("            NB=NB,\n", "", 2),
)

_CONVOLUTION_REPLACEMENTS = (
    Replacement(
        "    key=['D', 'W', 'NB'],",
        f"    {_CONVOLUTION_MARKER}\n    key=['D', 'W'],",
        2,
    ),
    Replacement("    NB: tl.constexpr,\n", "", 2),
    Replacement("    NB = triton.cdiv(B*T, 1024)\n", "", 2),
    Replacement("        NB=NB,\n", "", 2),
)

_CONVOLUTION_KERNEL_REPLACEMENTS_042 = (
    Replacement(
        "    key=['D', 'W', 'NB'],",
        f"    {_CONVOLUTION_MARKER}\n    key=['D', 'W'],",
        2,
    ),
    Replacement("    NB: tl.constexpr,\n", "", 2),
)

_CONVOLUTION_OPS_REPLACEMENTS_042 = (
    Replacement(
        "    NB = triton.cdiv(B*T, 1024)\n",
        f"    {_CONVOLUTION_OPS_MARKER}\n",
        2,
    ),
    Replacement("        NB=NB,\n", "", 2),
)


_FUSED_NORM_LAYOUTS = (
    SourceLayout(
        "FLA 0.4.1 gated norm",
        _FUSED_NORM_REPLACEMENTS,
        _FUSED_NORM_MARKER,
    ),
    SourceLayout(
        "deployed FLA 0.4.2 gated norm",
        _FUSED_NORM_REPLACEMENTS_042,
        _FUSED_NORM_MARKER,
    ),
)
_MONOLITHIC_CONVOLUTION_LAYOUT = SourceLayout(
    "FLA 0.4.1 monolithic convolution",
    _CONVOLUTION_REPLACEMENTS,
    _CONVOLUTION_MARKER,
)
_SPLIT_CONVOLUTION_KERNEL_LAYOUT = SourceLayout(
    "FLA 0.4.2 split convolution kernels",
    _CONVOLUTION_KERNEL_REPLACEMENTS_042,
    _CONVOLUTION_MARKER,
)
_SPLIT_CONVOLUTION_OPS_LAYOUT = SourceLayout(
    "FLA 0.4.2 split convolution ops",
    _CONVOLUTION_OPS_REPLACEMENTS_042,
    _CONVOLUTION_OPS_MARKER,
)


def _state(source: str, layout: SourceLayout) -> str:
    old_counts = [source.count(item.old) for item in layout.replacements]
    expected = [item.count for item in layout.replacements]
    new_counts = [source.count(item.new) if item.new else None for item in layout.replacements]
    expected_new = [item.count if item.new else None for item in layout.replacements]
    marker_count = source.count(layout.marker)
    if old_counts == expected and marker_count == 0:
        return "unpatched"
    if (
        old_counts == [0] * len(layout.replacements)
        and new_counts == expected_new
        and marker_count == layout.marker_count
    ):
        return "patched"
    raise RuntimeError(
        f"FLA source does not match audited layout {layout.name!r}: "
        f"old_counts={old_counts}, expected_old={expected}, "
        f"new_counts={new_counts}, expected_new={expected_new}, "
        f"marker_count={marker_count}, expected_markers={layout.marker_count}"
    )


def _patch_file(
    path: Path,
    layout: SourceLayout,
    *,
    check_only: bool,
) -> str:
    source = path.read_text(encoding="utf-8")
    state = _state(source, layout)
    if check_only:
        if state != "patched":
            raise RuntimeError(f"FLA varlen autotune patch is not installed in {path}")
        return state

    if state == "unpatched":
        for item in layout.replacements:
            source = source.replace(item.old, item.new)
        if _state(source, layout) != "patched":
            raise RuntimeError(f"FLA varlen autotune patch failed post-write verification for {path}")
        path.write_text(source, encoding="utf-8")
    return "patched"


def _patch_file_profiles(
    path: Path,
    layouts: tuple[SourceLayout, ...],
    *,
    check_only: bool,
) -> str:
    source = path.read_text(encoding="utf-8")
    errors = []
    for layout in layouts:
        try:
            _state(source, layout)
        except RuntimeError as exc:
            errors.append(str(exc))
            continue
        return _patch_file(path, layout, check_only=check_only)
    raise RuntimeError(f"FLA source does not match any audited 0.4.1/0.4.2 layout in {path}: " + " | ".join(errors))


def patch_files(
    fused_norm_path: Path = DEFAULT_FUSED_NORM_PATH,
    convolution_path: Path = DEFAULT_CONVOLUTION_PATH,
    convolution_kernels_path: Path | None = None,
    convolution_ops_path: Path | None = None,
    *,
    check_only: bool = False,
) -> tuple[str, str]:
    fused_state = _patch_file_profiles(
        fused_norm_path,
        _FUSED_NORM_LAYOUTS,
        check_only=check_only,
    )
    monolithic_source = convolution_path.read_text(encoding="utf-8")
    if _CONVOLUTION_MARKER in monolithic_source or any(
        item.old in monolithic_source for item in _CONVOLUTION_REPLACEMENTS
    ):
        convolution_state = _patch_file(
            convolution_path,
            _MONOLITHIC_CONVOLUTION_LAYOUT,
            check_only=check_only,
        )
    else:
        kernels_path = convolution_kernels_path or convolution_path.parent / "conv/triton/kernels.py"
        ops_path = convolution_ops_path or convolution_path.parent / "conv/triton/ops.py"
        _patch_file(
            kernels_path,
            _SPLIT_CONVOLUTION_KERNEL_LAYOUT,
            check_only=check_only,
        )
        _patch_file(
            ops_path,
            _SPLIT_CONVOLUTION_OPS_LAYOUT,
            check_only=check_only,
        )
        convolution_state = "patched"
    return fused_state, convolution_state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fused-norm-path", type=Path, default=DEFAULT_FUSED_NORM_PATH)
    parser.add_argument("--convolution-path", type=Path, default=DEFAULT_CONVOLUTION_PATH)
    parser.add_argument("--convolution-kernels-path", type=Path)
    parser.add_argument("--convolution-ops-path", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    states = patch_files(
        args.fused_norm_path,
        args.convolution_path,
        args.convolution_kernels_path,
        args.convolution_ops_path,
        check_only=args.check,
    )
    print("FLA varlen autotune NB specialization: " f"fused_norm_gate={states[0]} convolution={states[1]}")


if __name__ == "__main__":
    main()
