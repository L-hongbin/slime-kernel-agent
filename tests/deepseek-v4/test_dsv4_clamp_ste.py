"""SwiGLU clamp uses exact clamp values with an identity STE backward."""

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from custom_kernels.deepseek_v4.megatron.mcore_model import _clamp


NUM_GPUS = 0


@pytest.mark.parametrize(
    ("minimum", "maximum"),
    [(-1.0, 1.0), (None, 1.0), (-1.0, None)],
)
def test_clamp_forward_matches_torch_and_backward_is_identity(minimum, maximum):
    x = torch.tensor([-2.0, -0.5, 0.5, 2.0], requires_grad=True)

    y = _clamp(x, min=minimum, max=maximum)

    assert torch.equal(y, x.detach().clamp(min=minimum, max=maximum))
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones_like(x))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
