# Qwen3.6 rollout 低精度实验

本文统一维护 FP8、W8A8 与 W4A16 的 Qwen3.6-27B / DrKernel 评测。留存实验来自 2026 年 5–6 月，默认包含 EAGLE/MTP；它们不替代当前 Qwen3.8 的 no-spec 运行合同

`Non-LA` 表示量化 body+MTP 的 MLP/self-attn，保留 `linear_attn` 和 `mtp.fc` BF16；blockwise64/128 指 `[64,64]` / `[128,128]`，不能与 row-wise group size 混淆

W8A8 RTN Non-LA per-channel 在这组完整 run 中提供了较好的 score/wall 对照。AWQ 的 T1 代码有效性明显较低，但跨机器的 final score 和整体 wall 不支持通用“量化无损”“AWQ 不可用”或显著性结论。配对条件与无效 run 的排除理由必须和数字一起阅读

## FP8

### 留存结果

- FP8 run 的 score 为 `0.39250`，T3 correct 为 `39.6%`；与下表 BF16 的数值接近
- 观测 wall time 为 `1:20:10`，但 device/backend/reward 配置与输出长度未对齐，不能作为纯 FP8 加速比

### Run

- Run dir: `checkpoints/Qwen3.6-27B-FP8/20260602_022840_newSlimeKG.tp4.eagle.rm16.C96.H20.linear-fi.fp8_ctx65536_n8_summ1600`
- Checkpoint: `/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B-FP8`
- Runtime: H20, TP4, C96, EAGLE draft tokens `4`, context `65536`, `linear_attn_backend=flashinfer`, `attention_backend=fa3`, `sampling_backend=flashinfer`
- Evidence: `run.log`, `eval_config.resolved.yaml`, `dumps/rollout_data/eval_0.pt`

### 效率对比

| Target | reward score | wall time | srv tok/s(req>=80) | accept len |
| ------ | -----------: | --------: | -----------------: | ---------: |
| BF16   |      0.38125 |   1:36:09 |               3006 |      3.234 |
| FP8    |      0.39250 |   1:20:10 |               3759 |      3.248 |

`srv tok/s(req>=80)` 只代表 high-concurrency decode 局部，不含 prefill、reward、
排队和 tail。FP8 run 的 `response_len` mean/median/max/min 是
`7998.0 / 7443.5 / 57550 / 0`，有 1 个 final response length 为 0 的样本

### 精度对比

分母均为 800；`Fast@x` 也是 in-all

| Target |   Compile T1/T2/T3 |   Correct T1/T2/T3 | Fast@1.0 T1/T2/T3 | Fast@1.2 T1/T2/T3 |
| ------ | -----------------: | -----------------: | ----------------: | ----------------: |
| BF16   | 43.6 / 62.4 / 65.9 | 25.2 / 35.6 / 39.0 | 8.8 / 14.6 / 16.5 |  4.0 / 8.0 / 10.2 |
| FP8    | 48.9 / 71.0 / 74.5 | 26.8 / 36.0 / 39.6 | 9.9 / 13.5 / 15.0 |   4.9 / 8.0 / 8.8 |

FP8 accept length `3.248`，prefix cache hit rate `0.428`，平均 cached tokens/sample
`4381`，没有看到 EAGLE collapse 或明显 truncation collapse

这些 run 缺少同设备、同 backend、同 reward、同 seed 的配对比较；保留原始指标和协议，不作通用精度无损或硬件速度承诺


## W8A8 完整评测

| Target | 方法 | 范围 | 粒度 | score | wall | 全 turn resp | srv tok/s | accept len |
|---|---|---|---|---:|---:|---:|---:|---:|
| BF16 | - | - | - | 0.38375 | 1:24:37 | 26.56M | 2994 | 3.24 |
| W8A8 | RTN | Non-LA | per-channel | 0.40750 | 1:16:12 | 26.47M | 3162 | 3.22 |
| W8A8 | RTN | Non-LA | blockwise128 | 0.38875 | 1:39:56 | 30.27M | 2767 | 3.23 |
| W8A8 | RTN | Non-LA | blockwise64 | 0.38125 | 1:39:02 | 27.55M | 2611 | 3.24 |
| W8A8 | SQ+RTN | MLP-only | per-channel | 0.37375 | 1:17:55 | 26.87M | 3160 | 3.23 |
| W8A8 | SQ+RTN | Non-LA | per-channel | - | 1:16:47 | 26.92M | 3169 | 3.24 |
| W8A8 | SQ+RTN | Non-LA | blockwise64 | 0.37500 | 1:35:25 | 27.23M | 2597 | 3.24 |
| W8A8 | QuaRot+RTN | Non-LA | per-channel | 0.37250 | 1:24:12 | 28.12M | 3086 | 3.15 |

`srv tok/s` 是 decode log 中 `req>=80` 的 median throughput，只代表高并发 decode 局部，不包含 prefill、reward/eval、排队和低并发 tail。all-linear run 在 `798/800` 时中断，没有完整 `eval_0.pt`，已从完整结果表排除

### 逐轮质量

单位为 `%`，分母为 800

| Target | 方法 | 范围 | 粒度 | Compile T1/T2/T3 | Correct T1/T2/T3 | Fast@1.0 T1/T2/T3 | Fast@1.2 T1/T2/T3 |
|---|---|---|---|---:|---:|---:|---:|
| BF16 | - | - | - | 39.8 / 62.1 / 65.4 | 23.2 / 34.6 / 38.8 | 9.6 / 12.6 / 14.6 | 4.9 / 7.5 / 9.9 |
| W8A8 | RTN | Non-LA | per-channel | 34.0 / 60.5 / 62.5 | 19.6 / 32.6 / 40.8 | 8.2 / 15.0 / 17.0 | 4.0 / 8.1 / 9.6 |
| W8A8 | RTN | Non-LA | blockwise128 | 46.5 / 65.6 / 67.8 | 27.0 / 34.9 / 39.2 | 11.2 / 15.1 / 15.4 | 6.4 / 8.0 / 7.2 |
| W8A8 | RTN | Non-LA | blockwise64 | 43.1 / 65.0 / 68.2 | 24.9 / 35.1 / 38.6 | 10.2 / 12.6 / 15.2 | 5.2 / 6.4 / 8.9 |
| W8A8 | SQ+RTN | MLP-only | per-channel | 36.4 / 63.2 / 65.6 | 21.2 / 32.4 / 37.8 | 8.5 / 13.5 / 17.2 | 4.2 / 7.0 / 11.0 |
| W8A8 | SQ+RTN | Non-LA | per-channel | 34.2 / 61.5 / 65.0 | 18.8 / 28.9 / 35.4 | 7.6 / 12.4 / 15.8 | 3.1 / 7.1 / 8.9 |
| W8A8 | SQ+RTN | Non-LA | blockwise64 | 39.6 / 61.6 / 67.4 | 24.2 / 33.1 / 38.2 | 9.8 / 14.8 / 17.2 | 4.5 / 8.0 / 9.9 |
| W8A8 | QuaRot+RTN | Non-LA | per-channel | 36.4 / 66.4 / 65.8 | 21.9 / 33.2 / 37.5 | 9.5 / 13.9 / 16.6 | 3.5 / 6.8 / 8.5 |

读法：per-channel RTN 在这组完整 run 中的 score 和 wall 优于所列 BF16 baseline。B128/B64 的 T1/T2 指标看起来不差，但端到端 score/wall 不胜出，不能用 turn-level 表面指标替代 full eval

### 机制与限制

1. W8A8 + EAGLE 没塌。BF16 fixed-request EAGLE gain `1.252x`；W8A8 all/Non-LA 分别 `1.119x` / `1.160x`。W8A8 gain 小，是因为 target no-spec 已变快，EAGLE 可继续省的 target 成本变少
2. blockwise128 的早期慢有两个已修问题：缺 A800 `blockwise_int8` tuned configs，以及默认 `mem_fraction_static=0.85 + chunked_prefill=8192/max_prefill=16384` 触发高水位 OOM。修复后 B128 full eval 无 OOM，但仍因输出变长而不胜出
3. blockwise128 的 wall 长主要由输出分布膨胀驱动：相对 per-channel 全 turn 多 `3.80M` tokens，turn1 净多 `2.82M`；blockwise 独有长样本的 `### CUDA_KERNELS` marker 明显更晚或缺失
4. blockwise64 总 token 低于 B128，但仍有同类 late-marker 长尾，并叠加较低 high-concurrency decode (`2611` vs per-channel `3162`) 和 tail drain
5. “blockwise proxy loss 更低”只对 held-out UltraChat / teacher-forced / MLP-only local L2 成立，不能推导 full rollout 更短。真实 eval 量化的是 Non-LA+MTP 的 263 个 INT8 Linear；`linear_attn` 和 `mtp.fc` 保持 BF16
6. QuaRot 修复了旧 RMSNorm/MTP accept collapse，但 score、wall、accept 和 T3 correct 都低于 per-channel RTN。QuaRot 还有额外 `lm_head` 成本：旋转后主模型和 MTP 的 head 权重不同，不能复用同一份 `lm_head`

旧 QuaRot accept=1.00 来自错误的 RMSNorm/MTP checkpoint，不能用于评价修复后的方法。RowG128 只完成 gate/smoke，未形成 full-eval 结果

### W8A8 证据

脚本路径记录历史实验来源，其中一次性 producer/probe 已退出维护；复核应先定位原 revision 和保存的产物，不能把下表当作当前可执行命令清单

| 类型 | 路径 |
|---|---|
| RTN producer | `scripts/quantize/producers/rtn_w8a8.py` |
| blockwise producer | `scripts/quantize/producers/rtn_w8a8_g128.py` |
| SmoothQuant producer | `scripts/quantize/producers/smoothquant_w8a8.py` |
| MTP checkpoint 工具 | `scripts/quantize/utils/mtp_checkpoint.py` |
| QuaRot MTP sanity | `scripts/quantize/rotation/check_mtp_rotation.py` |
| SGLang MTP patch | `scripts/quantize/patches/sglang_qwen3_5_mtp_separate_lm_head.patch` |
| fixed-request probe | `checkpoints/Qwen3.6-27B/w8a8_eagle_probe_20260530/` |
| BF16 EAGLE eval | `checkpoints/Qwen3.6-27B/20260529_132454_newSlimeKG.tp4.eagle.rm16_ctx65536_n8_summ1600` |
| RTN Non-LA ckpt | `checkpoints/quantized/RTN/Qwen3.6-27B-W8A8-RTN-nonla-mtp` |
| RTN all partial eval | `checkpoints/Qwen3.6-27B-W8A8-RTN/20260530_062219_newSlimeKG.tp4.eagle.rm16.w8a8all_ctx65536_n8_summ1600` |
| B128 fixed eval | `checkpoints/Qwen3.6-27B-W8A8-G128-RTN-nonla-mtp/20260531_150558_w8a8.g128.nonla_mtp.sglcfg.mem82.cp4096.100x8.eagle_ctx65536_n8_summ1600` |
| B64 RTN eval | `checkpoints/Qwen3.6-27B-W8A8-G64-RTN-nonla-mtp/20260601_032830_rtn.g64.nonla_mtp.sglcfg64.mem82.cp4096.100x8.eagle.rm16_ctx65536_n8_summ1600` |
| B64 SQ eval | `checkpoints/Qwen3.6-27B-SQ-W8A8-G64-RTN-nonla-mtp-a0p5-ultrachat/20260601_013236_smooth.g64.nonla_mtp.sglcfg64.mem82.cp4096.100x8.eagle.rm16_ctx65536_n8_summ1600` |
| QuaRot completed eval | `checkpoints/Qwen3.6-27B-QR-W8A8-nonla-mtp/20260531_002353_quarot.nonla_mtp.gemmafix_ctx65536_n8_summ1600` |
| B64/B128 structure analysis | `checkpoints/Qwen3.6-27B/blockwise_length_logic_20260601/` |
| B128 length analysis | `checkpoints/Qwen3.6-27B/g128_length_analysis_20260601/` |
| B64 RTN length analysis | `checkpoints/Qwen3.6-27B/rtn_g64_length_analysis_20260601/` |
| first-token logprob probe | `scripts/eval_drkernel/analysis/probe_blockwise_format_logits.py` |


## W4A16 AWQ ASYM MLP-only

| 项 | 值 |
|---|---|
| checkpoint | `checkpoints/quantized/AWQ/Qwen3.6-27B-AWQ-W4A16-asym-mlp` |
| eval run | `checkpoints/Qwen3.6-27B-AWQ-W4A16-asym-mlp/20260531_233411_awq.w4a16.asym_mlp.sglzpfix.100x8.eagle.rm39_ctx65536_n8_summ1600` |
| rollout / reward | `.16` rollout，`.39` KernelGym reward；8 GPU worker + 24 CPU worker |
| SGLang patch | `scripts/quantize/patches/sglang_compressed_tensors_wna16_asym.patch` |
| eval wrapper | `scripts/eval_drkernel/rollout_speedup_ablation/debug.27b.tp4.eagle.awq_w4a16.sh` |

重跑 W4 ASYM 前必须检查 `.16` live SGLang 是否仍有 `symmetric=weight_quant.symmetric` 传参；镜像重装曾丢 patch

### 量化范围与校准

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

这不是 all-linear W4，也没有量化 MTP 或 linear-attn；结果表里必须写成 `W4A16 AWQ ASYM MLP-only`

### Full-eval 与逐轮结果

分母均为 800；BF16/W8A8 是 `.22/.40` 生产基线，W4 是 `.16/.39`，不是同机同 RM paired A/B

| Target | 方法 | 范围 | score | wall | response mean/median | srv tok/s | accept len |
|---|---|---|---:|---:|---:|---:|---:|
| BF16 | - | - | 0.38375 | 1:24:37 | 8366.2 / 7968.5 | 2994 | 3.24 |
| W8A8 | RTN | Non-LA | 0.40750 | 1:16:12 | 8262.8 / 7720.0 | 3162 | 3.22 |
| W8A8 | SQ+RTN | MLP-only | 0.37375 | 1:17:55 | 8013.6 / 7659.5 | 3160 | 3.23 |
| W4A16 | AWQ ASYM | MLP-only | 0.35375 | 1:25:49 | 7878.5 / 7185.0 | 2607 | 3.25 |

`0.37375` 是 W8A8 SQ+RTN MLP 的 run log/sample reward 口径；`302/800=0.37750` 是 T3 raw correctness，不是 eval metric

### 错误形态

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

Paired transition: W8A8 MLP T1 correct -> W4 wrong `149`，反向 `66`；W8A8 MLP compiled -> W4 not compiled `236`

### 量化误差分析

#### 已闭合

- runtime fidelity：patched SGLang marlin INT4 kernel 的有效权重与离线 `dequant_w4` 对齐，抽样 rel-L2 约 `3e-5`；unpatched symmetric-misload 为 `0.39-0.72`
- code-domain 本地 MLP L2：`W4A16 AWQ ASYM G128=43.27`，`W8-A8=37.33`，W4/W8-A8=`1.16x`。同口径只看权重轴，W4 权重误差是 W8 权重误差的 `9.17x` (`43.27` vs `4.72`)
- long-context local L2：8k -> 64k 的 W4/W8-A8 稳定在 `1.08-1.12x`，没有随位置爆炸

#### 未闭合

- output-stage metrics 只支持温和差距：code-domain final-logits L2 `1.23x`，top-1k KL mean `1.47x`，p99 `1.61x`
- top-1 token probability/head 指标不支持“大幅更差”：W4 对 BF16 top-1 token 的平均保留反而略高于 W8-A8
- autoregressive compounding 已证伪：沿 W4 自己轨迹的 KL 与沿 BF16 轨迹相同，overall ratio `0.99`，且不随位置增长

当前结论：受控分布指标显示 W4 比 W8-A8 差，但只到约 `1.5x`；它们不能解释 T1 compile/correct 的 `~1.7-1.8x` 掉点，更不能证明当前跨机 final score 差距显著。下一步需要 prefix/API-token 级指标

W4 与 W8A8 MLP 的 paired final reward 差异未达到原文所声称的显著程度：W4-only 113、W8-only 129，McNemar continuity-corrected z 约 0.96。更明确的观察是 T1 compile/correct 差异；code-domain KL 与 trajectory compounding probe 尚不能闭合其机制

### 尚未闭合的验证

1. 做同机、同 RM、同 seed/prompt slots 的 controlled A/B eval：W4 vs W8A8 MLP vs BF16。输出 sample/problem 级 final reward、T1 compile/correct 和错误类别转移
2. 对 T1 失败前缀做 prefix-conditioned logit probe：`kTVMFloat`、`kTVMFFIFloat`、常见 API/name token、section marker、code fence、stop token 的 rank/logprob
3. 只有 A/B 和 prefix probe 指出方向后，再考虑校准集、group size、只量化 `down_proj`、只量化 `gate/up` 或代码域校准

### W4A16 证据

下面同样记录历史工具与产物身份，不保证一次性脚本仍在当前分支中

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
