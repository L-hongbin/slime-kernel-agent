import pytest
import torch

from scripts.patch_flashqla_cpu_metadata import NEW, OLD, patch_package
from slime.utils.sequence_metadata import cache_cpu_sequence_boundaries, get_cpu_sequence_boundaries

pytestmark = pytest.mark.unit
NUM_GPUS = 0


def test_cpu_metadata_avoids_readback(monkeypatch):
    boundaries = torch.tensor([0, 8, 24], dtype=torch.int32)
    cache_cpu_sequence_boundaries(boundaries, [0, 8, 24])
    monkeypatch.setattr(torch.Tensor, "tolist", lambda self: pytest.fail("unexpected tensor readback"))
    assert get_cpu_sequence_boundaries(boundaries) == (0, 8, 24)


def test_cpu_metadata_invalidates_after_mutation():
    boundaries = torch.tensor([0, 8, 24])
    assert get_cpu_sequence_boundaries(boundaries) == (0, 8, 24)
    boundaries[-1] = 32
    assert get_cpu_sequence_boundaries(boundaries) == (0, 8, 32)


def test_cpu_metadata_does_not_reuse_clone_cache():
    boundaries = torch.tensor([0, 8, 24])
    cache_cpu_sequence_boundaries(boundaries, [0, 8, 24])
    cloned = boundaries.clone()
    cloned[-1] = 32
    assert get_cpu_sequence_boundaries(cloned) == (0, 8, 32)
    assert get_cpu_sequence_boundaries(boundaries) == (0, 8, 24)


def test_cpu_metadata_validates_shape():
    with pytest.raises(ValueError):
        cache_cpu_sequence_boundaries(torch.tensor([0, 8]), [0, 8, 24])


def test_flashqla_cpu_patch_is_versioned_and_idempotent(tmp_path):
    (tmp_path / "__init__.py").write_text('__version__ = "0.1.2"')
    target = tmp_path / "ops/gated_delta_rule/chunk/cp_context.py"
    target.parent.mkdir(parents=True)
    target.write_text(OLD)
    with pytest.raises(RuntimeError, match="Missing"):
        patch_package(tmp_path, check_only=True)
    patch_package(tmp_path)
    patch_package(tmp_path)
    patch_package(tmp_path, check_only=True)
    assert target.read_text() == NEW
    (tmp_path / "__init__.py").write_text('__version__ = "0.1.3"')
    with pytest.raises(RuntimeError, match="0.1.2"):
        patch_package(tmp_path)
