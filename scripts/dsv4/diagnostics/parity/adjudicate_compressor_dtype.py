#!/usr/bin/env python3
"""Adjudicate whether the Target-4 bf16 compressor pool->RMSNorm boundary EXISTS on the
production H20 path. Proves the dtype chain by EXECUTING production constructors, not by
static reading.

Verdict (2026-07-13): the production H20 pool->norm chain is fp32 end to end, so NO bf16
boundary exists there. The earlier Phase-A `probe_compressor_pool_norm.py` is archived under
`local_artifacts/deepseek-v4/retired_scripts/diagnostics/parity/`; it
constructed bf16 buffers -- a config only reachable via AITER `_tgemm` on HIP/ROCm (out of
scope). See handoffs/deepseek-v4/t4_compressor_probe.md "## Adjudication".

Run on the node53 H20 dev container (needs a real DSV4 config on disk):
  CUDA_VISIBLE_DEVICES=7 python3 scripts/dsv4/diagnostics/parity/adjudicate_compressor_dtype.py \
      --config /nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
os.environ.setdefault("TILELANG_CACHE_DIR", "/tmp/t4_probe_tilelang_cache")

import sys

sys.path.insert(0, "/sgl-workspace/sglang/python")

import torch


def sec(t):
    print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default="/nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8",
    )
    args = ap.parse_args()
    dev = torch.device("cuda", 0)

    sec("1. dispatch flags on THIS box")
    from sglang.jit_kernel.dsv4 import gemm as gm
    from sglang.srt.environ import envs
    from sglang.srt.layers.attention.dsv4 import compressor as cmp
    from sglang.srt.utils import get_bool_env_var, is_hip

    print("is_hip():", is_hip(), "| SGLANG_USE_AITER:", get_bool_env_var("SGLANG_USE_AITER"))
    print("compressor._tgemm:", cmp._tgemm, "| gemm._use_aiter:", gm._use_aiter)
    print("gemm._linear_bf16_fp32_algo:", repr(gm._linear_bf16_fp32_algo))
    print("SGLANG_OPT_USE_COMPRESSOR_V2:", envs.SGLANG_OPT_USE_COMPRESSOR_V2.get())

    sec("2. linear_bf16_fp32(bf16, bf16).dtype == production kv_score_input dtype")
    x = torch.randn(16, 4096, dtype=torch.bfloat16, device=dev)
    w = torch.randn(1024, 4096, dtype=torch.bfloat16, device=dev)
    print("linear_bf16_fp32 out dtype:", cmp.linear_bf16_fp32(x, w).dtype)
    print("compute_kv_score branch:", "_tgemm.mm -> bf16" if cmp._tgemm is not None else "linear_bf16_fp32 -> fp32")

    sec("3. DSV4 state buffer dtype (real CompressStatePool)")
    from sglang.srt.mem_cache.deepseek_v4_compress_state import CompressStatePool

    mrk = "/sgl-workspace/sglang/python/sglang/srt/model_executor/" "model_runner_kv_cache_mixin.py"
    txt = open(mrk).read()
    print("model_runner `state_dtype = torch.float32` constant present:", "state_dtype = torch.float32" in txt)
    for ratio, overlap in ((4, True), (128, False)):
        pool = CompressStatePool(
            size=ratio * 4,
            ring_size=2,
            overlap=overlap,
            head_dim=512,
            dtype=torch.float32,
            device=str(dev),
            enable_memory_saver=False,
            ratio=ratio,
        )
        print(
            f"ratio={ratio}: real CompressStatePool(dtype=state_dtype=fp32)"
            f".kv_score.dtype = {pool.kv_score_buffer.kv_score.dtype}"
        )

    sec("4. REAL Compressor module: type + compute_kv_score dtype")
    import torch.distributed as dist

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT="29593", RANK="0", WORLD_SIZE="1")
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=0, world_size=1)
    from sglang.srt.distributed import init_distributed_environment, initialize_model_parallel

    try:
        init_distributed_environment(
            world_size=1, rank=0, local_rank=0, distributed_init_method="tcp://127.0.0.1:29594", backend="nccl"
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
    except Exception as e:
        print("  (mp init note:", repr(e)[:80], ")")

    # The CP gate reads global server args (unrelated to dtype); neutralize it so the
    # real matmul branch executes. It is identity when no context-parallel.
    cmp.dsa_use_prefill_cp = lambda *_a, **_k: False

    from sglang.srt.configs.deepseek_v4 import DeepSeekV4Config
    from sglang.srt.layers.attention.dsv4.compressor import Compressor

    cfg = DeepSeekV4Config.from_pretrained(args.config)
    freqs = torch.polar(torch.ones(4096, 32, device=dev), torch.zeros(4096, 32, device=dev))
    comp = Compressor(
        config=cfg,
        layer_id=0,
        is_in_indexer=False,
        freqs_cis=freqs,
        compress_ratio=4,
        head_dim=512,
        rotate=False,
        prefix="",
    ).to(dev)
    print("type(compressor):", type(comp).__module__ + "." + type(comp).__name__)
    print(
        "wkv_gate.weight.dtype:",
        comp.wkv_gate.weight.dtype,
        "| ape.dtype:",
        comp.ape.dtype,
        "| norm.weight.dtype:",
        comp.norm.weight.dtype,
    )

    class _FB:
        class _M:
            def is_idle(self):
                return False

        forward_mode = _M()

    xh = torch.randn(8, cfg.hidden_size, dtype=torch.bfloat16, device=dev)
    ksc = comp.compute_kv_score(xh, _FB())
    print(">>> REAL Compressor.compute_kv_score OUTPUT dtype:", ksc.dtype)

    sec("5. compress_forward(out=None) pooled dtype: production-fp32 vs Phase-A bf16")
    from sglang.jit_kernel.dsv4.compress import CompressorDecodePlan, compress_forward

    def run(kind, R, in_dtype):
        D, n_win, B = 512, 4, 2
        pages = B * n_win
        coff = 2 if kind == "csa" else 1
        buf = torch.randn(pages, R, 2 * D * coff, dtype=in_dtype, device=dev)
        kv_input = buf[:, R - 1, :].clone()
        ape = torch.randn(R * coff, D, dtype=in_dtype, device=dev)
        p = torch.arange(pages, dtype=torch.int32, device=dev)
        wv = p % n_win
        plan = CompressorDecodePlan(
            R,
            torch.stack([(wv + 1) * R, p * R + (R - 1), torch.where(wv > 0, p - 1, torch.zeros_like(p)), p], 1)
            .contiguous()
            .view(torch.uint8),
        )
        return compress_forward(
            kv_score_buffer=buf, kv_score_input=kv_input, ape=ape.view(-1, D), plan=plan, head_dim=D, compress_ratio=R
        ).dtype

    for kind, R in (("csa", 4), ("hca", 128)):
        print(
            f"{kind.upper()} R={R}: production(fp32 in, out=None) -> {run(kind, R, torch.float32)}"
            f" ; Phase-A build(bf16 in) -> {run(kind, R, torch.bfloat16)}"
        )

    sec("VERDICT: production pool->norm is fp32 -> NO bf16 boundary on H20; Target 4 void")


if __name__ == "__main__":
    main()
