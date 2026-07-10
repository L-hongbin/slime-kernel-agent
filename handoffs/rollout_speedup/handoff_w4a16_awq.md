# W4A16 AWQ 交接

最后更新：2026-06-02 CST。

本文只记录 Qwen3.6-27B 的 W4A16/AWQ 方向。W8A8 只作为必要基线引用；QuaRot、SmoothQuant 和 INT8 blockwise 细节不要继续塞进这里。

## 当前结论

- 当前有效主线：`W4A16 AWQ ASYM MLP-only`，checkpoint `checkpoints/quantized/AWQ/Qwen3.6-27B-AWQ-W4A16-asym-mlp`。
- 质量：full eval score `0.35375`，不是生产候选。final score 与 W8A8 SQ+RTN MLP (`0.37375`) 的差距不显著；真正稳健异常是 T1 code validity，W4 T1 compile/correct `20.0% / 10.9%`，W8A8 MLP 是 `36.4% / 21.2%`。
- runtime：ASYM zero-point patch 是必需项。不打 patch 时 W4 ASYM 会被当 symmetric 误载，忽略 zero point 约造成 `53%` 权重 rel-L2 破坏；patched marlin runtime 与离线 dequant 对齐，抽样 rel-L2 约 `3e-5`。
- 解释状态：code-domain final-logits top-1k KL 只有 `1.47x`，p99 `1.61x`；trajectory compounding ratio `0.99`，已证伪“自回归累积放大”。当前离线指标不能闭合 T1 掉点，需要受控 A/B 和 prefix/API-token logprob。

## Checkpoint 与运行口径

| 项 | 值 |
|---|---|
| checkpoint | `checkpoints/quantized/AWQ/Qwen3.6-27B-AWQ-W4A16-asym-mlp` |
| eval run | `checkpoints/Qwen3.6-27B-AWQ-W4A16-asym-mlp/20260531_233411_awq.w4a16.asym_mlp.sglzpfix.100x8.eagle.rm39_ctx65536_n8_summ1600` |
| rollout / reward | `.16` rollout，`.39` KernelGym reward；8 GPU worker + 24 CPU worker |
| SGLang patch | `scripts/quantize/patches/sglang_compressed_tensors_wna16_asym.patch` |
| eval wrapper | `scripts/eval_drkernel/rollout_speedup_ablation/debug.27b.tp4.eagle.awq_w4a16.sh` |

重跑 W4 ASYM 前必须检查 `.16` live SGLang 是否仍有 `symmetric=weight_quant.symmetric` 传参；镜像重装曾丢 patch。

## 量化范围与校准

| 模块 | 状态 |
|---|---|
| body MLP `gate/up/down` | W4A16 AWQ ASYM |
| `self_attn` / `linear_attn` / `lm_head` | BF16 |
| `mtp.*` | BF16 |
| 视觉/音频相关模块 | BF16 |

| 配置 | 值 |
|---|---|
| group size | 128 |
| zero point | true |
| AWQ search | `duo_scaling=both`, `n_grid=40` |
| 校准集 | UltraChat 256 rows, max seq length 2048 |
| code 数据用途 | 只用于量化误差分析，不用于构建 checkpoint |

这不是 all-linear W4，也没有量化 MTP 或 linear-attn；结果表里必须写成 `W4A16 AWQ ASYM MLP-only`。

## Full Eval 结果

分母均为 800；BF16/W8A8 是 `.22/.40` 生产基线，W4 是 `.16/.39`，不是同机同 RM paired A/B。

| Target | 方法 | 范围 | score | wall | response mean/median | srv tok/s | accept len |
|---|---|---|---:|---:|---:|---:|---:|
| BF16 | - | - | 0.38375 | 1:24:37 | 8366.2 / 7968.5 | 2994 | 3.24 |
| W8A8 | RTN | Non-LA | 0.40750 | 1:16:12 | 8262.8 / 7720.0 | 3162 | 3.22 |
| W8A8 | SQ+RTN | MLP-only | 0.37375 | 1:17:55 | 8013.6 / 7659.5 | 3160 | 3.23 |
| W4A16 | AWQ ASYM | MLP-only | 0.35375 | 1:25:49 | 7878.5 / 7185.0 | 2607 | 3.25 |

`0.37375` 是 W8A8 SQ+RTN MLP 的 run log/sample reward 口径；`302/800=0.37750` 是 T3 raw correctness，不是 eval metric。

## Turn-Level 与错误形态

| Target | Compile T1/T2/T3 | Correct T1/T2/T3 | Fast@1.0 T1/T2/T3 | Fast@1.2 T1/T2/T3 |
|---|---:|---:|---:|---:|
| BF16 | 39.8 / 62.1 / 65.4 | 23.2 / 34.6 / 38.8 | 9.6 / 12.6 / 14.6 | 4.9 / 7.5 / 9.9 |
| W8A8 SQ+RTN MLP | 36.4 / 63.2 / 65.6 | 21.2 / 32.4 / 37.8 | 8.5 / 13.5 / 17.2 | 4.2 / 7.0 / 11.0 |
| W4A16 AWQ ASYM | 20.0 / 57.8 / 65.2 | 10.9 / 21.0 / 35.5 | 3.0 / 7.5 / 14.1 | 1.0 / 4.2 / 8.2 |

T1 结构拆解：

| Model | T1 correct | T1 compile | T1 all sections | T1 resp mean/median/p90 |
|---|---:|---:|---:|---:|
| W8A8 SQ+RTN MLP | 21.2% | 36.4% | 99.5% | 17333.1 / 12927.5 / 37795.7 |
| W4A16 AWQ ASYM | 10.9% | 20.0% | 99.2% | 16112.0 / 12611.5 / 33544.3 |

W4 的 T1 掉点不是 section 格式消失，也不是输出更长；主要是编译通过率和正确率掉。错误类别更偏 API/name token：

| Model | T1 wrong/fail top categories |
|---|---|
| W8A8 SQ+RTN MLP | `compile_other=234`, `compiled_wrong=121`, `no_member=114`, `kTVMFloat=69`, `other_wrong=59` |
| W4A16 AWQ ASYM | `kTVMFloat=190`, `compile_other=175`, `no_member=124`, `not_declared=110`, `compiled_wrong=73` |

Paired transition: W8A8 MLP T1 correct -> W4 wrong `149`，反向 `66`；W8A8 MLP compiled -> W4 not compiled `236`。

## 量化误差分析

### 已闭合

- runtime fidelity：patched SGLang marlin INT4 kernel 的有效权重与离线 `dequant_w4` 对齐，抽样 rel-L2 约 `3e-5`；unpatched symmetric-misload 为 `0.39-0.72`。
- code-domain 本地 MLP L2：`W4A16 AWQ ASYM G128=43.27`，`W8-A8=37.33`，W4/W8-A8=`1.16x`。同口径只看权重轴，W4 权重误差是 W8 权重误差的 `9.17x` (`43.27` vs `4.72`)。
- long-context local L2：8k -> 64k 的 W4/W8-A8 稳定在 `1.08-1.12x`，没有随位置爆炸。

### 未闭合

- output-stage metrics 只支持温和差距：code-domain final-logits L2 `1.23x`，top-1k KL mean `1.47x`，p99 `1.61x`。
- top-1 token probability/head 指标不支持“大幅更差”：W4 对 BF16 top-1 token 的平均保留反而略高于 W8-A8。
- autoregressive compounding 已证伪：沿 W4 自己轨迹的 KL 与沿 BF16 轨迹相同，overall ratio `0.99`，且不随位置增长。

当前结论：受控分布指标显示 W4 比 W8-A8 差，但只到约 `1.5x`；它们不能解释 T1 compile/correct 的 `~1.7-1.8x` 掉点，更不能证明当前跨机 final score 差距显著。下一步需要 prefix/API-token 级指标。

## 不再使用的结论

- 不再说“propagation probe 证明 W4 自回归误差累积”。该 probe 选的是 W4 failure prefix，`~2.1x` 是 selection bias。
- 不再说“final score 已经显著更差”。W4 vs W8A8 SQ+RTN MLP paired final reward: W4-only `113`，W8-only `129`，McNemar continuity-corrected `z≈0.96`。
- 不再用 UltraChat-only local MLP loss 判断 DrKernel full eval。UltraChat 只用于当前 checkpoint 校准；分析口径已切到 code-domain BF16 rollout。

## 下一步

1. 做同机、同 RM、同 seed/prompt slots 的 controlled A/B eval：W4 vs W8A8 MLP vs BF16。输出 sample/problem 级 final reward、T1 compile/correct 和错误类别转移。
2. 对 T1 失败前缀做 prefix-conditioned logit probe：`kTVMFloat`、`kTVMFFIFloat`、常见 API/name token、section marker、code fence、stop token 的 rank/logprob。
3. 只有 A/B 和 prefix probe 指出方向后，再考虑校准集、group size、只量化 `down_proj`、只量化 `gate/up` 或代码域校准。

## 证据路径

| 类型 | 路径 |
|---|---|
| producer | `scripts/quantize/producers/awq_w4a16.py` |
| checkpoint gate | `scripts/quantize/utils/check_awq_w4a16.py` |
| eval wrapper | `scripts/eval_drkernel/rollout_speedup_ablation/debug.27b.tp4.eagle.awq_w4a16.sh` |
| SGLang ASYM patch | `scripts/quantize/patches/sglang_compressed_tensors_wna16_asym.patch` |
| W4/W8 saved eval gap analyzer | `scripts/eval_drkernel/analysis/analyze_w4_awq_eval_gap.py` |
| error propagation probe | `scripts/eval_drkernel/analysis/probe_w4_awq_error_propagation.py` |
| zero-point sensitivity | `scripts/eval_drkernel/analysis/probe_w4_asym_zp_sensitivity.py` |
| runtime fidelity | `scripts/eval_drkernel/analysis/probe_sglang_wna16_runtime_fidelity.py` |
| code-domain KL probe | `scripts/eval_drkernel/analysis/probe_w4_codedomain_metric_kl.py` |
| trajectory compounding probe | `scripts/eval_drkernel/analysis/probe_w4_trajectory_compounding.py` |
| code fake-quant loss artifacts | `checkpoints/quantized/analysis/real_code_quant_loss_20260602/` |
| long-context artifacts | `checkpoints/quantized/analysis/real_code_quant_loss_{longctx_20260602,position_20260602,64k_20260602}/` |
| runtime/metric artifacts | `checkpoints/quantized/analysis/w4_asym_zp_sensitivity/` |
| eval gap artifacts | `checkpoints/quantized/analysis/w4_awq_eval_gap_20260601/` |
| ASYM full eval | `checkpoints/Qwen3.6-27B-AWQ-W4A16-asym-mlp/20260531_233411_awq.w4a16.asym_mlp.sglzpfix.100x8.eagle.rm39_ctx65536_n8_summ1600` |
