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


def test_v4_attention_torch_fallback_matches_reference(monkeypatch):
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    monkeypatch.setenv("V4_ATTENTION_TORCH", "1")
    import custom_kernels.deepseek_v4.megatron.attention as attention
    from custom_kernels.deepseek_v4.attention.reference import attention_reference

    attention = importlib.reload(attention)

    try:
        torch.manual_seed(11)
        args = (
            torch.randn(1, 3, 5, 4, requires_grad=True),
            torch.randn(1, 1, 5, 4, requires_grad=True),
            torch.randn(1, 1, 2, 4, requires_grad=True),
            torch.randn(3, requires_grad=True),
            4,
            2,
        )
        got, got_grads = _run_forward_backward(attention._v4flash_attention, args)
        exp, exp_grads = _run_forward_backward(attention_reference, args)
        assert torch.allclose(got, exp)
        for got_grad, exp_grad in zip(got_grads[:4], exp_grads[:4], strict=True):
            assert torch.allclose(got_grad, exp_grad)

        q, k_raw, k_comp, sinks, window, m = args
        got_bf16 = attention._v4flash_attention(
            q.detach().to(torch.bfloat16),
            k_raw.detach().to(torch.bfloat16),
            k_comp.detach().to(torch.bfloat16),
            sinks.detach(),
            window,
            m,
        )
        assert got_bf16.dtype == torch.bfloat16
    finally:
        monkeypatch.delenv("V4_ATTENTION_TORCH", raising=False)
        importlib.reload(attention)
