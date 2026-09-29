import textwrap

import pytest
import torch

from scripts.patch_megatron_mtp_empty_mask import NEW, OLD, patch_file

pytestmark = pytest.mark.unit
NUM_GPUS = 0


def test_empty_mtp_mask_patch_is_idempotent(tmp_path):
    path = tmp_path / "gpt_model.py"
    path.write_text(OLD)
    with pytest.raises(RuntimeError, match="missing"):
        patch_file(path, check_only=True)
    patch_file(path)
    patch_file(path)
    patch_file(path, check_only=True)
    assert path.read_text() == NEW
    path.write_text("unexpected layout")
    with pytest.raises(RuntimeError, match="Unexpected"):
        patch_file(path)


@pytest.mark.parametrize("mask", [[0.0, 0.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
def test_patched_mtp_normalization_preserves_nonempty_gradients(mask):
    namespace = {}
    source = (
        "def normalize(mtp_loss, loss_mask):\n"
        "    num_tokens = loss_mask.sum()\n"
        + textwrap.indent(textwrap.dedent(NEW), "    ")
        + "    return mtp_loss.sum() / num_tokens\n"
    )
    exec(compile(source, "patched_mtp_loss", "exec"), namespace)
    losses = torch.tensor([0.2, 0.5, 0.7], requires_grad=True)
    loss_mask = torch.tensor(mask)
    actual = namespace["normalize"](losses, loss_mask)
    (gradient,) = torch.autograd.grad(actual, losses)
    assert torch.isfinite(actual) and torch.isfinite(gradient).all()
    if sum(mask):
        torch.testing.assert_close(actual, (losses * loss_mask).sum() / loss_mask.sum(), rtol=0, atol=0)
        torch.testing.assert_close(gradient, loss_mask / loss_mask.sum(), rtol=0, atol=0)
    else:
        assert actual == 0
        assert not gradient.any()
