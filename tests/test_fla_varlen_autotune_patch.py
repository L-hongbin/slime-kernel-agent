"""CPU contract tests for the fail-closed FLA varlen autotune patch."""

import sys
from pathlib import Path

import pytest

NUM_GPUS = 0
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.patch_fla_varlen_autotune_nb import (
    _CONVOLUTION_KERNEL_REPLACEMENTS_042,
    _CONVOLUTION_MARKER,
    _CONVOLUTION_OPS_MARKER,
    _CONVOLUTION_OPS_REPLACEMENTS_042,
    _CONVOLUTION_REPLACEMENTS,
    _FUSED_NORM_MARKER,
    _FUSED_NORM_REPLACEMENTS,
    _FUSED_NORM_REPLACEMENTS_042,
    patch_files,
)


def _fixture(replacements) -> str:
    return "\n".join(item.old * item.count for item in replacements)


@pytest.mark.unit
def test_patch_is_exact_idempotent_and_removes_unused_nb_specialization(tmp_path: Path):
    fused_norm = tmp_path / "fused_norm_gate.py"
    convolution = tmp_path / "convolution.py"
    fused_norm.write_text(_fixture(_FUSED_NORM_REPLACEMENTS), encoding="utf-8")
    convolution.write_text(_fixture(_CONVOLUTION_REPLACEMENTS), encoding="utf-8")

    assert patch_files(fused_norm, convolution) == ("patched", "patched")
    fused_source = fused_norm.read_text(encoding="utf-8")
    convolution_source = convolution.read_text(encoding="utf-8")

    assert fused_source.count(_FUSED_NORM_MARKER) == 2
    assert convolution_source.count(_CONVOLUTION_MARKER) == 2
    for source, replacements in [
        (fused_source, _FUSED_NORM_REPLACEMENTS),
        (convolution_source, _CONVOLUTION_REPLACEMENTS),
    ]:
        for item in replacements:
            assert item.old not in source
        assert "NB: tl.constexpr" not in source
        assert "NB=NB" not in source

    assert patch_files(fused_norm, convolution) == ("patched", "patched")
    assert patch_files(fused_norm, convolution, check_only=True) == ("patched", "patched")


@pytest.mark.unit
def test_patch_fails_closed_on_unknown_fla_source(tmp_path: Path):
    fused_norm = tmp_path / "fused_norm_gate.py"
    convolution = tmp_path / "convolution.py"
    fused_norm.write_text("unexpected source", encoding="utf-8")
    convolution.write_text(_fixture(_CONVOLUTION_REPLACEMENTS), encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not match any audited 0.4.1/0.4.2 layout"):
        patch_files(fused_norm, convolution)


@pytest.mark.unit
def test_patch_check_fails_closed_when_patched_source_is_corrupted(tmp_path: Path):
    fused_norm = tmp_path / "fused_norm_gate.py"
    convolution = tmp_path / "convolution.py"
    fused_norm.write_text(_fixture(_FUSED_NORM_REPLACEMENTS), encoding="utf-8")
    convolution.write_text(_fixture(_CONVOLUTION_REPLACEMENTS), encoding="utf-8")
    patch_files(fused_norm, convolution)

    source = fused_norm.read_text(encoding="utf-8")
    source = source.replace(
        _FUSED_NORM_REPLACEMENTS[0].new,
        f"    {_FUSED_NORM_MARKER}\n    corrupted_key = True",
        1,
    )
    fused_norm.write_text(source, encoding="utf-8")

    assert source.count(_FUSED_NORM_MARKER) == 2
    with pytest.raises(RuntimeError, match="new_counts"):
        patch_files(fused_norm, convolution, check_only=True)


@pytest.mark.unit
def test_split_fla_042_layout_is_exact_and_idempotent(tmp_path: Path):
    fused_norm = tmp_path / "fused_norm_gate.py"
    convolution = tmp_path / "convolution.py"
    kernels = tmp_path / "conv/triton/kernels.py"
    ops = tmp_path / "conv/triton/ops.py"
    kernels.parent.mkdir(parents=True)
    fused_norm.write_text(_fixture(_FUSED_NORM_REPLACEMENTS_042), encoding="utf-8")
    convolution.write_text("from fla.modules.conv import ShortConvolution\n", encoding="utf-8")
    kernels.write_text(_fixture(_CONVOLUTION_KERNEL_REPLACEMENTS_042), encoding="utf-8")
    ops.write_text(_fixture(_CONVOLUTION_OPS_REPLACEMENTS_042), encoding="utf-8")

    assert patch_files(fused_norm, convolution, kernels, ops) == ("patched", "patched")
    assert fused_norm.read_text(encoding="utf-8").count(_FUSED_NORM_MARKER) == 2
    assert kernels.read_text(encoding="utf-8").count(_CONVOLUTION_MARKER) == 2
    assert ops.read_text(encoding="utf-8").count(_CONVOLUTION_OPS_MARKER) == 2
    assert patch_files(fused_norm, convolution, kernels, ops, check_only=True) == ("patched", "patched")


@pytest.mark.unit
def test_installed_fla_layout_is_still_patchable(tmp_path: Path):
    installed = Path("/usr/local/lib/python3.12/dist-packages/fla/modules")
    if not installed.is_dir():
        pytest.skip("FLA is not installed in this CPU test environment")

    fused_norm = tmp_path / "fused_norm_gate.py"
    convolution = tmp_path / "convolution.py"
    fused_norm.write_text((installed / fused_norm.name).read_text(encoding="utf-8"), encoding="utf-8")
    convolution.write_text((installed / convolution.name).read_text(encoding="utf-8"), encoding="utf-8")

    installed_kernels = installed / "conv/triton/kernels.py"
    installed_ops = installed / "conv/triton/ops.py"
    if installed_kernels.is_file() and installed_ops.is_file():
        kernels = tmp_path / "conv/triton/kernels.py"
        ops = tmp_path / "conv/triton/ops.py"
        kernels.parent.mkdir(parents=True)
        kernels.write_text(installed_kernels.read_text(encoding="utf-8"), encoding="utf-8")
        ops.write_text(installed_ops.read_text(encoding="utf-8"), encoding="utf-8")
        assert patch_files(fused_norm, convolution, kernels, ops) == (
            "patched",
            "patched",
        )
    else:
        assert patch_files(fused_norm, convolution) == ("patched", "patched")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
