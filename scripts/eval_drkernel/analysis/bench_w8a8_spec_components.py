#!/usr/bin/env python3
"""Microbench W8A8 + EAGLE component costs for Qwen3.6-27B.

Run on the SGLang host with:

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/sgl-workspace/sglang/python \
    /usr/bin/python3 bench_w8a8_spec_components.py
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

# F821: ruff misreports tensors defined in the enclosing bench function as
# undefined inside the timing closures. B023: those closures are invoked
# immediately by time_cuda() in the same iteration, so the loop-variable
# capture is intentional. Both are false positives for this diagnostic bench.
# ruff: noqa: F821, B023


SGLANG_PYTHON = "/sgl-workspace/sglang/python"
if SGLANG_PYTHON not in sys.path:
    sys.path.insert(0, SGLANG_PYTHON)

from sgl_kernel import int8_scaled_mm  # noqa: E402
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel  # noqa: E402
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import causal_conv1d_update  # noqa: E402
from sglang.srt.layers.attention.mamba.mamba_state_scatter_triton import (  # noqa: E402
    fused_mamba_state_scatter_with_mask,
)
from sglang.srt.layers.quantization.int8_kernel import per_token_quant_int8  # noqa: E402

DTYPE = torch.bfloat16


@dataclass
class Timing:
    name: str
    batch: int
    tokens_per_req: int
    ms: float
    note: str = ""


def sync() -> None:
    torch.cuda.synchronize()


def time_cuda(fn: Callable[[], object], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        out = fn()
        if isinstance(out, torch.Tensor):
            out.flatten()[0].item()
    sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    sync()
    return start.elapsed_time(end) / iters


def clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def int8_weight(n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    # SGLang creates [N, K] int8 params, then process_weights_after_loading
    # stores layer.weight = weight.t(), i.e. a column-major [K, N] view.
    w_base = torch.empty((n, k), device="cuda", dtype=torch.int8)
    w_base.random_(-127, 127)
    w = w_base.t()
    scale = torch.ones((n, 1), device="cuda", dtype=torch.float32) * 0.01
    return w, scale


def bf16_weight(n: int, k: int) -> torch.Tensor:
    return torch.empty((n, k), device="cuda", dtype=DTYPE).normal_(0, 0.02)


def bench_quant_linear(
    name: str,
    m: int,
    k: int,
    n: int,
    batch: int,
    tokens_per_req: int,
    warmup: int,
    iters: int,
) -> list[Timing]:
    x = torch.empty((m, k), device="cuda", dtype=DTYPE).normal_(0, 1)
    w_i8, w_scale = int8_weight(n, k)
    w_bf16 = bf16_weight(n, k)
    x_q, x_scale = per_token_quant_int8(x)
    sync()

    def quant_only():
        return per_token_quant_int8(x)

    def int8_mm_only():
        return int8_scaled_mm(x_q, w_i8, x_scale, w_scale, out_dtype=DTYPE)

    def int8_apply():
        q, s = per_token_quant_int8(x)
        return int8_scaled_mm(q, w_i8, s, w_scale, out_dtype=DTYPE)

    def bf16_mm():
        return torch.matmul(x, w_bf16.t())

    rows = [
        Timing(
            f"{name}.per_token_quant_int8",
            batch,
            tokens_per_req,
            time_cuda(quant_only, warmup, iters),
            f"M={m}, K={k}",
        ),
        Timing(
            f"{name}.int8_scaled_mm_only",
            batch,
            tokens_per_req,
            time_cuda(int8_mm_only, warmup, iters),
            f"M={m}, K={k}, N={n}",
        ),
        Timing(
            f"{name}.w8a8_apply_quant_plus_mm",
            batch,
            tokens_per_req,
            time_cuda(int8_apply, warmup, iters),
            f"M={m}, K={k}, N={n}",
        ),
        Timing(
            f"{name}.bf16_matmul",
            batch,
            tokens_per_req,
            time_cuda(bf16_mm, warmup, iters),
            f"M={m}, K={k}, N={n}",
        ),
    ]
    del x, w_i8, w_scale, w_bf16, x_q, x_scale
    clear_cuda()
    return rows


def bench_gdn_core(
    batch: int,
    tokens_per_req: int,
    warmup: int,
    iters: int,
    hk: int,
    hv: int,
    dk: int,
    dv: int,
    conv_kernel: int,
) -> list[Timing]:
    kernel = TritonGDNKernel()
    pool = batch + 8
    q_dim = hk * dk
    k_dim = hk * dk
    v_dim = hv * dv
    conv_dim = q_dim + k_dim + v_dim

    cache_indices = torch.arange(batch, device="cuda", dtype=torch.int32)
    query_start_decode = torch.arange(0, batch + 1, device="cuda", dtype=torch.int32)
    query_start_verify = torch.arange(
        0,
        batch * tokens_per_req + 1,
        step=tokens_per_req,
        device="cuda",
        dtype=torch.int32,
    )
    intermediate_indices = torch.arange(pool, device="cuda", dtype=torch.int32)

    a1 = torch.empty((batch, hv), device="cuda", dtype=DTYPE).normal_()
    b1 = torch.empty((batch, hv), device="cuda", dtype=DTYPE).normal_()
    mixed1 = torch.empty((batch, conv_dim), device="cuda", dtype=DTYPE).normal_()
    q1 = torch.empty((1, batch, hk, dk), device="cuda", dtype=DTYPE).normal_()
    k1 = torch.empty((1, batch, hk, dk), device="cuda", dtype=DTYPE).normal_()
    v1 = torch.empty((1, batch, hv, dv), device="cuda", dtype=DTYPE).normal_()

    total_verify_tokens = batch * tokens_per_req
    a4 = torch.empty((total_verify_tokens, hv), device="cuda", dtype=DTYPE).normal_()
    b4 = torch.empty((total_verify_tokens, hv), device="cuda", dtype=DTYPE).normal_()
    q4 = torch.empty((1, total_verify_tokens, hk, dk), device="cuda", dtype=DTYPE).normal_()
    k4 = torch.empty((1, total_verify_tokens, hk, dk), device="cuda", dtype=DTYPE).normal_()
    v4 = torch.empty((1, total_verify_tokens, hv, dv), device="cuda", dtype=DTYPE).normal_()

    ssm_states = torch.empty((pool, hv, dv, dk), device="cuda", dtype=DTYPE).normal_()
    intermediate_states = torch.empty((pool, tokens_per_req, hv, dv, dk), device="cuda", dtype=DTYPE)
    conv_states = torch.empty((pool, conv_dim, conv_kernel - 1), device="cuda", dtype=DTYPE).normal_()
    conv_weight = torch.empty((conv_dim, conv_kernel), device="cuda", dtype=DTYPE).normal_()
    intermediate_conv = torch.empty((pool, tokens_per_req, conv_dim, conv_kernel - 1), device="cuda", dtype=DTYPE)
    mixed_verify = torch.empty((batch, conv_dim, tokens_per_req), device="cuda", dtype=DTYPE).normal_()
    A_log = torch.empty((hv,), device="cuda", dtype=torch.float32).normal_()
    dt_bias = torch.empty((hv,), device="cuda", dtype=DTYPE).normal_()

    def packed_decode_recurrent():
        return kernel.packed_decode(
            mixed_qkv=mixed1,
            a=a1,
            b=b1,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=dk**-0.5,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            num_v_heads=hv,
            head_v_dim=dv,
        )

    def target_verify_recurrent():
        return kernel.target_verify(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q4,
            k=k4,
            v=v4,
            a=a4,
            b=b4,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_verify,
            intermediate_states_buffer=intermediate_states,
            intermediate_state_indices=intermediate_indices,
            cache_steps=tokens_per_req,
            retrieve_parent_token=None,
        )

    def decode_conv():
        return causal_conv1d_update(
            mixed1,
            conv_states,
            conv_weight,
            None,
            "silu",
            conv_state_indices=cache_indices,
        )

    def verify_conv():
        return causal_conv1d_update(
            mixed_verify,
            conv_states,
            conv_weight,
            None,
            "silu",
            conv_state_indices=cache_indices,
            intermediate_conv_window=intermediate_conv,
            intermediate_state_indices=intermediate_indices[:batch],
            retrieve_next_token=None,
            retrieve_next_sibling=None,
            retrieve_parent_token=None,
        )

    def nonpacked_decode_recurrent():
        return kernel.decode(
            q=q1,
            k=k1,
            v=v1,
            a=a1,
            b=b1,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_decode,
        )

    # Compile all variants before timing; the first Triton run can dominate.
    for fn in (
        packed_decode_recurrent,
        nonpacked_decode_recurrent,
        target_verify_recurrent,
        decode_conv,
        verify_conv,
    ):
        fn()
    sync()

    rows = [
        Timing(
            "gdn.conv_decode_1tok",
            batch,
            1,
            time_cuda(decode_conv, warmup, iters),
            f"conv_dim={conv_dim}, kernel={conv_kernel}",
        ),
        Timing(
            "gdn.conv_verify_4tok",
            batch,
            tokens_per_req,
            time_cuda(verify_conv, warmup, iters),
            f"conv_dim={conv_dim}, kernel={conv_kernel}, saves intermediate windows",
        ),
        Timing(
            "gdn.recurrent_decode_1tok_packed",
            batch,
            1,
            time_cuda(packed_decode_recurrent, warmup, iters),
            f"HV={hv}, K={dk}, V={dv}, updates state",
        ),
        Timing(
            "gdn.recurrent_decode_1tok_nonpacked",
            batch,
            1,
            time_cuda(nonpacked_decode_recurrent, warmup, iters),
            f"HV={hv}, K={dk}, V={dv}, updates state",
        ),
        Timing(
            "gdn.recurrent_verify_4tok",
            batch,
            tokens_per_req,
            time_cuda(target_verify_recurrent, warmup, iters),
            f"HV={hv}, K={dk}, V={dv}, caches {tokens_per_req} states",
        ),
    ]
    clear_cuda()
    return rows


def bench_sdpa(
    batch: int,
    tokens_per_req: int,
    context_len: int,
    warmup: int,
    iters: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> list[Timing]:
    rows: list[Timing] = []
    for t in (1, tokens_per_req):
        q = torch.empty((batch, num_heads, t, head_dim), device="cuda", dtype=DTYPE).normal_()
        k = torch.empty(
            (batch, num_kv_heads, context_len + t, head_dim),
            device="cuda",
            dtype=DTYPE,
        ).normal_()
        v = torch.empty_like(k)

        def sdpa():
            return F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)

        rows.append(
            Timing(
                f"full_attention.sdpa_gqa_{t}tok",
                batch,
                t,
                time_cuda(sdpa, warmup, iters),
                f"Heads={num_heads}, KVHeads={num_kv_heads}, D={head_dim}, ctx={context_len}; PyTorch SDPA, not paged RadixAttention",
            )
        )
        del q, k, v
        clear_cuda()
    return rows


def bench_mamba_commit(
    batch: int,
    tokens_per_req: int,
    warmup: int,
    iters: int,
    num_layers: int,
    hv: int,
    dk: int,
    dv: int,
    conv_dim: int,
    conv_kernel: int,
) -> list[Timing]:
    pool = batch + 8
    dst_indices = torch.arange(batch, device="cuda", dtype=torch.int32)
    step_indices = torch.full((batch,), tokens_per_req - 1, device="cuda", dtype=torch.int32)
    ssm_states = torch.empty((num_layers, pool, hv, dv, dk), device="cuda", dtype=DTYPE)
    ssm_intermediate = torch.empty(
        (num_layers, pool, tokens_per_req, hv, dv, dk),
        device="cuda",
        dtype=DTYPE,
    )
    conv_states = torch.empty((num_layers, pool, conv_dim, conv_kernel - 1), device="cuda", dtype=DTYPE)
    conv_intermediate = torch.empty(
        (num_layers, pool, tokens_per_req, conv_dim, conv_kernel - 1),
        device="cuda",
        dtype=DTYPE,
    )

    def commit_ssm():
        return fused_mamba_state_scatter_with_mask(ssm_states, ssm_intermediate, dst_indices, step_indices)

    def commit_conv():
        return fused_mamba_state_scatter_with_mask(conv_states, conv_intermediate, dst_indices, step_indices)

    def commit_both():
        commit_ssm()
        commit_conv()

    commit_both()
    sync()
    rows = [
        Timing(
            "gdn.commit_verify_ssm_all_layers",
            batch,
            tokens_per_req,
            time_cuda(commit_ssm, warmup, iters),
            f"layers={num_layers}, HV={hv}, K={dk}, V={dv}",
        ),
        Timing(
            "gdn.commit_verify_conv_all_layers",
            batch,
            tokens_per_req,
            time_cuda(commit_conv, warmup, iters),
            f"layers={num_layers}, conv_dim={conv_dim}, kernel={conv_kernel}",
        ),
        Timing(
            "gdn.commit_verify_ssm_plus_conv_all_layers",
            batch,
            tokens_per_req,
            time_cuda(commit_both, warmup, iters),
            "matches update_mamba_state_after_mtp_verify first two scatter calls",
        ),
    ]
    clear_cuda()
    return rows


def summarize(rows: list[Timing], batch: int, tokens_per_req: int) -> dict[str, float]:
    by_name = {r.name: r.ms for r in rows}

    # Qwen3.6-27B shape from text_config:
    # 48 linear-attn layers, 16 full-attn layers, 64 dense MLPs.
    gdn_decode = by_name["gdn.conv_decode_1tok"] + by_name["gdn.recurrent_decode_1tok_packed"]
    gdn_verify = by_name["gdn.conv_verify_4tok"] + by_name["gdn.recurrent_verify_4tok"]

    mlp_decode = by_name["mlp_gate_up.w8a8_apply_quant_plus_mm"] + by_name["mlp_down.w8a8_apply_quant_plus_mm"]
    mlp_verify = (
        by_name["mlp_gate_up_verify.w8a8_apply_quant_plus_mm"] + by_name["mlp_down_verify.w8a8_apply_quant_plus_mm"]
    )
    full_proj_decode = by_name["full_qkv.w8a8_apply_quant_plus_mm"] + by_name["full_o.w8a8_apply_quant_plus_mm"]
    full_proj_verify = (
        by_name["full_qkv_verify.w8a8_apply_quant_plus_mm"] + by_name["full_o_verify.w8a8_apply_quant_plus_mm"]
    )
    gdn_proj_decode = by_name["gdn_qkvz.bf16_matmul"] + by_name["gdn_ba.bf16_matmul"] + by_name["gdn_out.bf16_matmul"]
    gdn_proj_verify = (
        by_name["gdn_qkvz_verify.bf16_matmul"]
        + by_name["gdn_ba_verify.bf16_matmul"]
        + by_name["gdn_out_verify.bf16_matmul"]
    )

    quant_overhead_decode = 64 * (
        by_name["mlp_gate_up.per_token_quant_int8"] + by_name["mlp_down.per_token_quant_int8"]
    ) + 16 * (by_name["full_qkv.per_token_quant_int8"] + by_name["full_o.per_token_quant_int8"])
    quant_overhead_verify = 64 * (
        by_name["mlp_gate_up_verify.per_token_quant_int8"] + by_name["mlp_down_verify.per_token_quant_int8"]
    ) + 16 * (by_name["full_qkv_verify.per_token_quant_int8"] + by_name["full_o_verify.per_token_quant_int8"])
    full_sdpa_decode = by_name["full_attention.sdpa_gqa_1tok"]
    full_sdpa_verify = by_name["full_attention.sdpa_gqa_4tok"]
    mamba_commit_verify = by_name["gdn.commit_verify_ssm_plus_conv_all_layers"]

    summary = {
        "projected_decode_ms_quantized_mlp_all_64": 64 * mlp_decode,
        "projected_verify_ms_quantized_mlp_all_64": 64 * mlp_verify,
        "projected_decode_ms_full_attn_quantized_proj_all_16": 16 * full_proj_decode,
        "projected_verify_ms_full_attn_quantized_proj_all_16": 16 * full_proj_verify,
        "projected_decode_ms_full_attn_sdpa_all_16": 16 * full_sdpa_decode,
        "projected_verify_ms_full_attn_sdpa_all_16": 16 * full_sdpa_verify,
        "projected_decode_ms_gdn_bf16_proj_all_48": 48 * gdn_proj_decode,
        "projected_verify_ms_gdn_bf16_proj_all_48": 48 * gdn_proj_verify,
        "projected_decode_ms_gdn_conv_scan_all_48": 48 * gdn_decode,
        "projected_verify_ms_gdn_conv_scan_all_48": 48 * gdn_verify,
        "projected_verify_ms_gdn_commit_after_accept_all_48": mamba_commit_verify,
        "projected_decode_ms_quant_overhead_only": quant_overhead_decode,
        "projected_verify_ms_quant_overhead_only": quant_overhead_verify,
        "observed_batch": float(batch),
        "tokens_per_req": float(tokens_per_req),
    }
    decode_total = (
        summary["projected_decode_ms_quantized_mlp_all_64"]
        + summary["projected_decode_ms_full_attn_quantized_proj_all_16"]
        + summary["projected_decode_ms_full_attn_sdpa_all_16"]
        + summary["projected_decode_ms_gdn_bf16_proj_all_48"]
        + summary["projected_decode_ms_gdn_conv_scan_all_48"]
    )
    verify_total = (
        summary["projected_verify_ms_quantized_mlp_all_64"]
        + summary["projected_verify_ms_full_attn_quantized_proj_all_16"]
        + summary["projected_verify_ms_full_attn_sdpa_all_16"]
        + summary["projected_verify_ms_gdn_bf16_proj_all_48"]
        + summary["projected_verify_ms_gdn_conv_scan_all_48"]
        + summary["projected_verify_ms_gdn_commit_after_accept_all_48"]
    )
    summary["projected_decode_ms_sum_listed_components"] = decode_total
    summary["projected_verify_ms_sum_listed_components"] = verify_total
    summary["projected_verify_over_decode_ratio_listed_components"] = verify_total / decode_total
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=80)
    parser.add_argument("--tokens-per-req", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--sdpa-context", type=int, default=2048)
    parser.add_argument("--tp", type=int, default=4)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    batch = args.batch
    t = args.tokens_per_req
    tp = args.tp
    m1 = batch
    m4 = batch * t

    rows: list[Timing] = []
    # Actual text_config shapes for Qwen3.6-27B / qwen3_5_text.
    hidden = 5120
    intermediate = 17408
    gdn_key_heads = 16
    gdn_value_heads = 48
    gdn_key_dim = 128
    gdn_value_dim = 128
    full_heads = 24
    full_kv_heads = 4
    head_dim = 256
    assert full_heads % tp == 0
    assert full_kv_heads % tp == 0
    assert gdn_key_heads % tp == 0
    assert gdn_value_heads % tp == 0
    assert intermediate % tp == 0
    gdn_key_heads_rank = gdn_key_heads // tp
    gdn_value_heads_rank = gdn_value_heads // tp
    gdn_qkvz = 2 * gdn_key_heads_rank * gdn_key_dim + 2 * gdn_value_heads_rank * gdn_value_dim
    gdn_ba = 2 * gdn_value_heads_rank
    gdn_out_in = gdn_value_heads_rank * gdn_value_dim
    full_heads_rank = full_heads // tp
    full_kv_heads_rank = max(1, full_kv_heads // tp)
    full_qkv = (full_heads_rank * 2 + 2 * full_kv_heads_rank) * head_dim
    full_o_in = full_heads_rank * head_dim
    intermediate_rank = intermediate // tp

    linear_shapes = [
        ("mlp_gate_up", m1, hidden, 2 * intermediate_rank),
        ("mlp_down", m1, intermediate_rank, hidden),
        ("full_qkv", m1, hidden, full_qkv),
        ("full_o", m1, full_o_in, hidden),
        ("gdn_qkvz", m1, hidden, gdn_qkvz),
        ("gdn_ba", m1, hidden, gdn_ba),
        ("gdn_out", m1, gdn_out_in, hidden),
        ("mlp_gate_up_verify", m4, hidden, 2 * intermediate_rank),
        ("mlp_down_verify", m4, intermediate_rank, hidden),
        ("full_qkv_verify", m4, hidden, full_qkv),
        ("full_o_verify", m4, full_o_in, hidden),
        ("gdn_qkvz_verify", m4, hidden, gdn_qkvz),
        ("gdn_ba_verify", m4, hidden, gdn_ba),
        ("gdn_out_verify", m4, gdn_out_in, hidden),
    ]
    for name, m, k, n in linear_shapes:
        rows.extend(bench_quant_linear(name, m, k, n, batch, 1 if m == m1 else t, args.warmup, args.iters))

    rows.extend(
        bench_gdn_core(
            batch=batch,
            tokens_per_req=t,
            warmup=args.warmup,
            iters=args.iters,
            hk=gdn_key_heads_rank,
            hv=gdn_value_heads_rank,
            dk=gdn_key_dim,
            dv=gdn_value_dim,
            conv_kernel=4,
        )
    )
    rows.extend(
        bench_mamba_commit(
            batch=batch,
            tokens_per_req=t,
            warmup=args.warmup,
            iters=args.iters,
            num_layers=48,
            hv=gdn_value_heads_rank,
            dk=gdn_key_dim,
            dv=gdn_value_dim,
            conv_dim=(2 * gdn_key_heads_rank * gdn_key_dim + gdn_value_heads_rank * gdn_value_dim),
            conv_kernel=4,
        )
    )
    rows.extend(
        bench_sdpa(
            batch=batch,
            tokens_per_req=t,
            context_len=args.sdpa_context,
            warmup=max(3, args.warmup // 2),
            iters=max(10, args.iters // 2),
            num_heads=full_heads_rank,
            num_kv_heads=full_kv_heads_rank,
            head_dim=head_dim,
        )
    )

    result = {
        "host": os.uname().nodename,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "batch": batch,
        "tokens_per_req": t,
        "tp": tp,
        "timings": [asdict(r) for r in rows],
        "summary": summarize(rows, batch, t),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
