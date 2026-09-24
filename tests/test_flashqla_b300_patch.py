from pathlib import Path

import pytest

from scripts.patch_flashqla_b300 import NEW, OLD, patch_package

pytestmark = pytest.mark.unit


def make_package(tmp_path: Path):
    root = tmp_path / "flash_qla"
    target = root / "ops/gated_delta_rule/chunk"
    target.mkdir(parents=True)
    (root / "__init__.py").write_text('__version__ = "0.1.2"\n')
    for name in ("__init__.py", "cp_context.py"):
        (target / name).write_text(OLD + '\n    ARCH = "SM100"\n')
    return root, target


def test_b300_patch_is_narrow_and_idempotent(tmp_path):
    root, target = make_package(tmp_path)
    with pytest.raises(RuntimeError, match="Missing SM103"):
        patch_package(root, check_only=True)
    patch_package(root)
    patch_package(root)
    patch_package(root, check_only=True)
    for name in ("__init__.py", "cp_context.py"):
        assert (target / name).read_text() == NEW + '\n    ARCH = "SM100"\n'
        assert (target / (name + ".before-b300")).read_text() == OLD + '\n    ARCH = "SM100"\n'


def test_b300_patch_rejects_other_versions(tmp_path):
    root, _ = make_package(tmp_path)
    (root / "__init__.py").write_text('__version__ = "0.1.3"\n')
    with pytest.raises(RuntimeError, match="requires FlashQLA 0.1.2"):
        patch_package(root)


def test_b300_patch_validates_all_files_before_writing(tmp_path):
    root, target = make_package(tmp_path)
    (target / "cp_context.py").write_text("unsupported_layout = True\n")
    with pytest.raises(RuntimeError, match="Unexpected FlashQLA"):
        patch_package(root)
    assert OLD in (target / "__init__.py").read_text()
    assert not (target / "__init__.py.before-b300").exists()
