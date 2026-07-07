import importlib
import sys
from pathlib import Path

import torch


def _run_forward_backward(fn, args):
    cloned = [arg.detach().clone().requires_grad_(arg.requires_grad) if torch.is_tensor(arg) else arg for arg in args]
    out = fn(*cloned)
    out.sum().backward()
    grads = [arg.grad.detach().clone() if torch.is_tensor(arg) and arg.grad is not None else None for arg in cloned]
    return out.detach(), grads


def test_v4_compressor_torch_fallback_matches_reference(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    monkeypatch.setenv("V4_COMPRESS_TORCH", "1")
    import custom_kernels.deepseek_v4.megatron.compressor as compressor
    from custom_kernels.deepseek_v4.compression import reference

    compressor = importlib.reload(compressor)

    try:
        torch.manual_seed(7)
        eps = 1e-6

        hca_args = (
            torch.randn(1, 8, 4, requires_grad=True),
            torch.randn(1, 8, 4, requires_grad=True),
            torch.randn(4, 4, requires_grad=True),
            torch.randn(4, requires_grad=True),
            eps,
            4,
        )
        hca_got, hca_got_grads = _run_forward_backward(compressor._hca_compress, hca_args)
        hca_exp, hca_exp_grads = _run_forward_backward(reference.hca_compress_ref, hca_args)
        assert torch.allclose(hca_got, hca_exp)
        for got, exp in zip(hca_got_grads[:4], hca_exp_grads[:4], strict=True):
            assert torch.allclose(got, exp)

        csa_args = (
            torch.randn(1, 8, 8, requires_grad=True),
            torch.randn(1, 8, 8, requires_grad=True),
            torch.randn(2, 8, requires_grad=True),
            torch.randn(4, requires_grad=True),
            eps,
            2,
        )
        csa_got, csa_got_grads = _run_forward_backward(compressor._csa_compress, csa_args)
        csa_exp, csa_exp_grads = _run_forward_backward(reference.csa_compress_ref, csa_args)
        assert torch.allclose(csa_got, csa_exp)
        for got, exp in zip(csa_got_grads[:4], csa_exp_grads[:4], strict=True):
            assert torch.allclose(got, exp)
    finally:
        monkeypatch.delenv("V4_COMPRESS_TORCH", raising=False)
        importlib.reload(compressor)
