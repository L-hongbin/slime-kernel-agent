# V4 FP8 MoE expert compute — efficiency results

Bench: `custom_kernels/deepseek_v4/megatron/bench_fp8_moe.py`
(`PYTHONPATH=$PWD python custom_kernels/deepseek_v4/megatron/bench_fp8_moe.py`).
Re-run and update this file whenever the fp8 MoE compute path changes
(repo convention: kernel change ⇒ rerun fwd+bwd efficiency).

## What is measured

`V4GroupedExperts.forward_dispatched` at EP8 shapes (32 local experts, hidden
4096, moe-intermediate 2048), two modes:
- **fp8_gemm** (`V4_FP8_EXPERT_GEMM=1`): deep_gemm `m_grouped_fp8_gemm_nt_contiguous`
  — the same kernel sglang serves these experts with — forward; bf16 grad_x
  backward. No bf16 weight retained.
- **bf16_loop** (baseline): per-expert `F.linear` on the dequantized bf16 weight
  (also via the no-retention custom Function).

## Result (1× H20, 2026-07-05, e4m3, block [128,128])

| total tokens | mode      | fwd ms | fwd+bwd ms | peak GB |
| ---: | :--- | ---: | ---: | ---: |
| 8192  | fp8_gemm  | 6.11  | 19.41 | 1.34 |
| 8192  | bf16_loop | 13.85 | 34.58 | 1.41 |
| 16384 | fp8_gemm  | 7.95  | 25.82 | 1.83 |
| 16384 | bf16_loop | 17.00 | 44.31 | 1.91 |
| 32768 | fp8_gemm  | 13.60 | 38.88 | 2.74 |
| 32768 | bf16_loop | 23.60 | 65.03 | 2.92 |

**fp8_gemm is ~2.0× faster forward and ~1.7× faster fwd+bwd** than the per-expert
bf16 loop (one grouped kernel vs a Python loop of many small matmuls), at slightly
lower isolated peak memory.

## Comparison vs sglang (same kernel, our wrapper overhead)

Our forward uses the SAME kernel sglang serves these experts with
(`m_grouped_fp8_gemm_nt_contiguous`), so kernel throughput is identical. Per
gate_up matmul at T=16384, E=32 (1× H20):

| component | ms | vs sglang |
| :--- | ---: | :--- |
| deep_gemm kernel (sglang's ceiling) | 2.10 | identical (shared kernel) |
| per-token fp8 quant + TMA-align | 1.37 | inherent to fp8; sglang pays it too |
| pad/scatter to 128 (Python loop) | 0.89 | **our overhead** — sglang pads on-device, fused with dispatch |

We are at kernel parity with sglang. The one fixable gap is the Python
pad/scatter loop (~0.9 ms/matmul, ~0.8 s/step); vectorizing it on-device
(cumsum + scatter) would close most of it. Follow-up optimization, not a
correctness issue.

## Notes / caveats

- The large MEMORY win of fp8 (the ctx-16k OOM fix) is NOT this isolated single-
  call peak — both modes here use the no-retention custom Function. It shows at
  full-run scale, where whole-layer activation recompute otherwise retains every
  active expert's bf16 weight across the layer backward (~2GB). See
  `custom_kernels/deepseek_v4/megatron/mcore_model.py` `_Fp8GroupedExpertMatmul` /
  `_FrozenFp8ExpertLinear`.
- Numerical parity (validated separately): fp8 forward vs bf16 rel-err ~0.03–0.05
  (fp8 e4m3 precision, matches sglang rollout); grad_x rel-err ~0.03.
- Backward grad_x is bf16 (training-only, no sglang counterpart; avoids fp8-grad
  noise). A full-fp8 backward (deep_gemm nn variant) is a possible later speedup.
