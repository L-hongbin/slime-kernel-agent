#!/usr/bin/env python3
"""Microbenchmark: why is flash_mla_sparse_decode_triton slow on H20?

Times the Triton sparse-decode kernel (bf16 and fp8 gather branches) against
the compiled fp8 flash_mla_with_kvcache at matched DSv4 decode shapes, sweeps
the head count to expose per-head KV re-gather (grid=(B,H), no MQA grouping),
and reports achieved bandwidth vs the unique-KV roofline.

Run inside the old production fork container on ONE approved GPU:
  CUDA_VISIBLE_DEVICES=0 python3 triton_decode_microbench.py
"""

import json
import sys

import torch

D_QK = 576  # q width (512 latent + 64 rope)
D_V = 512
PAGE_SIZE = 256
NUM_PAGES = 64  # 16k tokens of cache — plenty for topk sampling
H20_BW_GBS = 4000.0  # H20 HBM3 ~4 TB/s


def make_bf16_cache(device):
    return torch.randn(NUM_PAGES, PAGE_SIZE, D_V, dtype=torch.bfloat16, device=device) * 0.05


def make_fp8_cache(device):
    # packed 584 B/token: 576 data + 8 scale bytes, scale section after data
    buf = torch.randint(0, 255, (NUM_PAGES, PAGE_SIZE * 584), dtype=torch.uint8, device=device)
    return buf.view(NUM_PAGES, PAGE_SIZE, 584)


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def main():
    device = "cuda:0"
    torch.manual_seed(0)
    from sglang.srt.layers.attention.flash_mla_sm120_triton import _run_triton_sparse_decode

    results = []
    bf16_cache = make_bf16_cache(device)
    fp8_cache = make_fp8_cache(device)

    total_tokens = NUM_PAGES * PAGE_SIZE

    for B in (4, 16):
        for topk in (128, 512, 640):
            idx = torch.randint(0, total_tokens, (B, topk), dtype=torch.int32, device=device)
            tl_full = torch.full((B,), topk, dtype=torch.int32, device=device)
            for H in (1, 8, 64):
                q = torch.randn(B, 1, H, D_QK, dtype=torch.bfloat16, device=device) * 0.05
                for name, cache in (("triton_bf16", bf16_cache), ("triton_fp8", fp8_cache)):
                    ms = bench(lambda q=q, c=cache, i=idx, t=tl_full: _run_triton_sparse_decode(q, c, i, t, 0.09))
                    bytes_per_tok = 1024 if name == "triton_bf16" else 584
                    uniq_gb = B * topk * bytes_per_tok / 1e9
                    row = {
                        "kernel": name,
                        "B": B,
                        "H": H,
                        "topk": topk,
                        "ms": round(ms, 4),
                        "unique_kv_MB": round(uniq_gb * 1e3, 2),
                        "eff_bw_GBs_unique": round(uniq_gb / (ms / 1e3), 1),
                        "eff_bw_GBs_xH": round(uniq_gb * H / (ms / 1e3), 1),
                        "roofline_us_unique": round(uniq_gb / H20_BW_GBS * 1e6, 1),
                    }
                    results.append(row)
                    print(json.dumps(row), flush=True)

    # Compiled fp8 flash-MLA reference at the same shapes (H=64 only; MQA kernel)
    try:
        import sgl_kernel.flash_mla as flash_mla

        meta = flash_mla.get_mla_metadata()[0]
        cache4 = fp8_cache.view(NUM_PAGES, PAGE_SIZE, 1, 584)
        for B in (4, 16):
            for topk in (128, 512, 640):
                idx = torch.randint(0, total_tokens, (B, 1, topk), dtype=torch.int32, device=device)
                tl_full = torch.full((B,), topk, dtype=torch.int32, device=device)
                q = torch.randn(B, 1, 64, D_QK, dtype=torch.bfloat16, device=device) * 0.05

                def call(q=q, i=idx, t=tl_full):
                    return flash_mla.flash_mla_with_kvcache(
                        q=q,
                        k_cache=cache4,
                        head_dim_v=D_V,
                        block_table=None,
                        cache_seqlens=None,
                        tile_scheduler_metadata=meta,
                        softmax_scale=0.09,
                        is_fp8_kvcache=True,
                        indices=i,
                        topk_length=t,
                        attn_sink=None,
                    )

                ms = bench(call)
                uniq_gb = B * topk * 584 / 1e9
                row = {
                    "kernel": "compiled_fp8",
                    "B": B,
                    "H": 64,
                    "topk": topk,
                    "ms": round(ms, 4),
                    "unique_kv_MB": round(uniq_gb * 1e3, 2),
                    "eff_bw_GBs_unique": round(uniq_gb / (ms / 1e3), 1),
                    "roofline_us_unique": round(uniq_gb / H20_BW_GBS * 1e6, 1),
                }
                results.append(row)
                print(json.dumps(row), flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"compiled_fp8 reference failed: {type(e).__name__}: {e}", file=sys.stderr)

    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/triton_decode_microbench.json"
    json.dump(results, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
