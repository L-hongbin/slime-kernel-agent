"""Validate FA4 training attention at Qwen3.8 TP4/CP1 local head dimensions."""

import argparse
import importlib.metadata
import json
import os
import time

import torch
import torch.nn.functional as functional


def check_backend(backend_name="flash"):
    os.environ.update(
        NVTE_FLASH_ATTN=str(int(backend_name in ("flash", "auto"))),
        NVTE_FUSED_ATTN=str(int(backend_name in ("fused", "auto"))),
        NVTE_UNFUSED_ATTN=str(int(backend_name in ("unfused", "auto"))),
    )
    from transformer_engine.pytorch.attention.dot_product_attention.utils import AttentionParams, get_attention_backend

    assert torch.cuda.get_device_capability() == (10, 3), "Expected B300 SM103"
    version = importlib.metadata.version("flash-attn-4")
    assert version == "4.0.0b15", version
    params = AttentionParams(
        qkv_layout="thd_thd_thd",
        num_heads=6,
        num_gqa_groups=1,
        head_dim_qk=256,
        head_dim_v=256,
        max_seqlen_q=120000,
        max_seqlen_kv=120000,
        attn_mask_type="padding_causal",
        window_size=(-1, 0),
        core_attention_bias_shape=None,
        core_attention_bias_requires_grad=False,
        context_parallel=False,
        cp_size=1,
        is_training=True,
    )
    backend = get_attention_backend(params)
    assert backend[0] or backend[2] or backend[4], f"No {backend_name} backend for B300 TP4/CP1 head_dim=256"
    if backend_name == "flash":
        assert backend[0] and not backend[2], backend
        assert str(backend[1]).startswith("4."), backend
    print(f"Requested {backend_name}, installed FA4 {version}, selected backend: {backend}", flush=True)


def reference_attention(inputs, lengths):
    outputs = []
    offset = 0
    for length in lengths:
        query, key, value = [tensor[offset : offset + length].transpose(0, 1) for tensor in inputs]
        outputs.append(
            functional.scaled_dot_product_attention(
                query, key.repeat_interleave(6, dim=0), value.repeat_interleave(6, dim=0), is_causal=True
            ).transpose(0, 1)
        )
        offset += length
    return torch.cat(outputs).flatten(1)


def check_attention(lengths, *, compare):
    from transformer_engine.pytorch import DotProductAttention

    module = DotProductAttention(
        num_attention_heads=6,
        num_gqa_groups=1,
        kv_channels=256,
        qkv_format="thd",
        attn_mask_type="padding_causal",
        attention_dropout=0.0,
    ).train()
    inputs = [
        torch.randn(sum(lengths), heads, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for heads in (6, 1, 1)
    ]
    cu_seqlens = torch.tensor([0, *lengths], device="cuda", dtype=torch.int32).cumsum(0, dtype=torch.int32)
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = module(
        *inputs,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max(lengths),
        max_seqlen_kv=max(lengths),
    )
    gradient = torch.randn_like(output)
    output.backward(gradient)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    actual_tensors = [output, *(tensor.grad for tensor in inputs)]
    for tensor in actual_tensors:
        assert tensor is not None and torch.isfinite(tensor).all()
    errors = {}
    if compare:
        reference_inputs = [tensor.detach().float().requires_grad_() for tensor in inputs]
        expected = reference_attention(reference_inputs, lengths)
        expected.backward(gradient.float())
        expected_tensors = [expected, *(tensor.grad for tensor in reference_inputs)]
        for name, actual, reference in zip(
            ("output", "query_grad", "key_grad", "value_grad"), actual_tensors, expected_tensors, strict=True
        ):
            assert torch.isfinite(reference).all(), name
            error = (actual.float() - reference).norm() / reference.norm().clamp_min(1e-8)
            errors[name] = error.item()
            assert error < 0.03, f"{name}: relative L2 error {error.item()}"
    print(
        json.dumps({"lengths": lengths, "forward_backward_seconds": elapsed, "relative_l2_errors": errors}), flush=True
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--backend", choices=("flash", "fused", "auto", "unfused"), default="flash")
    args = parser.parse_args()
    check_backend(args.backend)
    if args.check_only:
        return
    torch.manual_seed(1234)
    for lengths in ([256, 128], [2048, 512]):
        check_attention(lengths, compare=True)
    for lengths in ([32768, 512], [120000]):
        check_attention(lengths, compare=False)
    print("PASS: FA4 TP4/CP1 BF16 attention forward/backward, including the 120k context limit", flush=True)


if __name__ == "__main__":
    main()
