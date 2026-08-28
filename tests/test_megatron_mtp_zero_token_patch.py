"""Unit coverage for the fail-closed Megatron MTP zero-token source patch."""

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.patch_megatron_mtp_zero_token import _PATCHED_ROLL_BLOCK, _REPLACEMENTS, _ROLL_BLOCK, patch_file

NUM_GPUS = 0


def _fixture_source() -> str:
    return _ROLL_BLOCK + "\n".join(old for old, _new in _REPLACEMENTS)


@pytest.mark.unit
def test_patch_is_exact_and_idempotent(tmp_path: Path):
    target = tmp_path / "gpt_model.py"
    target.write_text(_fixture_source(), encoding="utf-8")

    assert patch_file(target) == "patched"
    patched = target.read_text(encoding="utf-8")
    assert patched.count(_PATCHED_ROLL_BLOCK) == 1
    for old, new in _REPLACEMENTS:
        assert old not in patched
        assert patched.count(new) == 1

    assert patch_file(target) == "patched"
    assert patch_file(target, check_only=True) == "patched"


@pytest.mark.unit
def test_patch_fails_closed_on_unknown_source(tmp_path: Path):
    target = tmp_path / "gpt_model.py"
    target.write_text("unexpected source", encoding="utf-8")

    with pytest.raises(RuntimeError, match="does not match the audited layout"):
        patch_file(target)


@pytest.mark.unit
def test_zero_token_masked_loss_stays_finite_and_zero():
    masked_loss = torch.tensor(0.0)
    num_tokens = torch.tensor(0.0)

    safe_loss = masked_loss / num_tokens.clamp_min(1.0)

    assert torch.isfinite(safe_loss)
    assert safe_loss.item() == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
