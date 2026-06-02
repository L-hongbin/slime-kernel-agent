# DrKernel W8A8 Rollout 交接

最后更新：2026-06-02 CST。

默认口径：除非特别说明，本文所有实验都使用 MTP；量化范围也覆盖对应 MTP 模块。`Non-LA` 表示量化 body+MTP 的 MLP/self-attn，保留 `linear_attn` 和 `mtp.fc` BF16。`blockwise64/128` 指 SGLang `blockwise_int8` 的 `[64,64] / [128,128]`，不是 row-wise G64/G128。

## 当前结论

- 可用基线：`W8A8 RTN Non-LA per-channel + EAGLE`。score `0.40750`，wall `1:16:12`，高于且快于 BF16 EAGLE (`0.38375`, `1:24:37`)。
- EAGLE 有效：W8A8 fixed-request 仍有 `1.119-1.160x`，accept 正常；gain 小于 BF16 是因为 W8A8 no-spec target decode 已更快。
- 不替换基线：all-linear RTN run 在 `798/800` 崩溃，仅 partial；B64/B128 blockwise、SmoothQuant、QuaRot 都没有同时超过 per-channel RTN 的 score 和 wall。
- 当前解释：blockwise64/128 和 QuaRot wall 变长主要来自输出 token 变多；QuaRot 另有额外开销，旋转后主模型和 MTP 的 `lm_head` 权重不再相同，不能继续共享一份 `lm_head`。

## 端到端结果

| Target | 方法 | 范围 | 粒度 | score | wall | 全 turn resp | srv tok/s | accept len |
|---|---|---|---|---:|---:|---:|---:|---:|
| BF16 | - | - | - | 0.38375 | 1:24:37 | 26.56M | 2994 | 3.24 |
| W8A8 | RTN | Non-LA | per-channel | 0.40750 | 1:16:12 | 26.47M | 3162 | 3.22 |
| W8A8 | RTN | all | per-channel | ~0.318* | 1:10:45* | 24.87M* | 3264 | - |
| W8A8 | RTN | Non-LA | blockwise128 | 0.38875 | 1:39:56 | 30.27M | 2767 | 3.23 |
| W8A8 | RTN | Non-LA | blockwise64 | 0.38125 | 1:39:02 | 27.55M | 2611 | 3.24 |
| W8A8 | SQ+RTN | MLP-only | per-channel | 0.37375 | 1:17:55 | 26.87M | 3160 | 3.23 |
| W8A8 | SQ+RTN | Non-LA | per-channel | - | 1:16:47 | 26.92M | 3169 | 3.24 |
| W8A8 | SQ+RTN | Non-LA | blockwise64 | 0.37500 | 1:35:25 | 27.23M | 2597 | 3.24 |
| W8A8 | QuaRot+RTN | Non-LA | per-channel | 0.37250 | 1:24:12 | 28.12M | 3086 | 3.15 |

`srv tok/s` 是 decode log 中 `req>=80` 的 median throughput，只代表高并发 decode 局部，不包含 prefill、reward/eval、排队和低并发 tail。`*` all-linear run 在 `798/800` 时 detokenizer 被 kill，没有完整 `eval_0.pt`；该行只保留 partial 口径，不作为生产比较结论。

## Turn-Level 质量

单位为 `%`，分母为 800。all-linear 没有完整 dump，Fast@1.0/1.2 无法从 run.log 复算。

| Target | 方法 | 范围 | 粒度 | Compile T1/T2/T3 | Correct T1/T2/T3 | Fast@1.0 T1/T2/T3 | Fast@1.2 T1/T2/T3 |
|---|---|---|---|---:|---:|---:|---:|
| BF16 | - | - | - | 39.8 / 62.1 / 65.4 | 23.2 / 34.6 / 38.8 | 9.6 / 12.6 / 14.6 | 4.9 / 7.5 / 9.9 |
| W8A8 | RTN | Non-LA | per-channel | 34.0 / 60.5 / 62.5 | 19.6 / 32.6 / 40.8 | 8.2 / 15.0 / 17.0 | 4.0 / 8.1 / 9.6 |
| W8A8 | RTN | all | per-channel | 35.5 / 59.0 / 61.2 | 18.8 / 28.1 / 31.9 | - | - |
| W8A8 | RTN | Non-LA | blockwise128 | 46.5 / 65.6 / 67.8 | 27.0 / 34.9 / 39.2 | 11.2 / 15.1 / 15.4 | 6.4 / 8.0 / 7.2 |
| W8A8 | RTN | Non-LA | blockwise64 | 43.1 / 65.0 / 68.2 | 24.9 / 35.1 / 38.6 | 10.2 / 12.6 / 15.2 | 5.2 / 6.4 / 8.9 |
| W8A8 | SQ+RTN | MLP-only | per-channel | 36.4 / 63.2 / 65.6 | 21.2 / 32.4 / 37.8 | 8.5 / 13.5 / 17.2 | 4.2 / 7.0 / 11.0 |
| W8A8 | SQ+RTN | Non-LA | per-channel | 34.2 / 61.5 / 65.0 | 18.8 / 28.9 / 35.4 | 7.6 / 12.4 / 15.8 | 3.1 / 7.1 / 8.9 |
| W8A8 | SQ+RTN | Non-LA | blockwise64 | 39.6 / 61.6 / 67.4 | 24.2 / 33.1 / 38.2 | 9.8 / 14.8 / 17.2 | 4.5 / 8.0 / 9.9 |
| W8A8 | QuaRot+RTN | Non-LA | per-channel | 36.4 / 66.4 / 65.8 | 21.9 / 33.2 / 37.5 | 9.5 / 13.9 / 16.6 | 3.5 / 6.8 / 8.5 |

读法：per-channel RTN 的 T3 correct/fast 最好，且 wall 最短。B128/B64 的 T1/T2 指标看起来不差，但端到端 score/wall 不胜出，不能用 turn-level 表面指标替代 full eval。

## 解释链

1. W8A8 + EAGLE 没塌。BF16 fixed-request EAGLE gain `1.252x`；W8A8 all/Non-LA 分别 `1.119x` / `1.160x`。W8A8 gain 小，是因为 target no-spec 已变快，EAGLE 可继续省的 target 成本变少。
2. blockwise128 的早期慢有两个已修问题：缺 A800 `blockwise_int8` tuned configs，以及默认 `mem_fraction_static=0.85 + chunked_prefill=8192/max_prefill=16384` 触发高水位 OOM。修复后 B128 full eval 无 OOM，但仍因输出变长而不胜出。
3. blockwise128 的 wall 长主要由输出分布膨胀驱动：相对 per-channel 全 turn 多 `3.80M` tokens，turn1 净多 `2.82M`；blockwise 独有长样本的 `### CUDA_KERNELS` marker 明显更晚或缺失。
4. blockwise64 总 token 低于 B128，但仍有同类 late-marker 长尾，并叠加较低 high-concurrency decode (`2611` vs per-channel `3162`) 和 tail drain。
5. “blockwise proxy loss 更低”只对 held-out UltraChat / teacher-forced / MLP-only local L2 成立，不能推导 full rollout 更短。真实 eval 量化的是 Non-LA+MTP 的 263 个 INT8 Linear；`linear_attn` 和 `mtp.fc` 保持 BF16。
6. QuaRot 修复了旧 RMSNorm/MTP accept collapse，但 score、wall、accept 和 T3 correct 都低于 per-channel RTN。QuaRot 还有额外 `lm_head` 成本：旋转后主模型和 MTP 的 head 权重不同，不能复用同一份 `lm_head`。

## 不再使用的结论

- 旧 20260530 QuaRot Non-LA/MLP accept=`1.00` 的结果无效，只说明旧 checkpoint 的 RMSNorm/MTP 处理错了；不要作为 QuaRot 质量结论。
- all-linear RTN 100x8 没有完整 eval dump，不要把 partial score 当完成 run。
- RowG128 只完成 checkpoint/gate/smoke，full eval 和 tuning 已停止；partial `[1,128]` configs 不进入结论。
- W4A16 AWQ 已拆到 `handoffs/in_progress/handoff_w4a16_awq.md`，本文不再维护 W4 细节。

## 下一步

1. 当前可用 INT8 基线继续用 `W8A8 RTN Non-LA per-channel + EAGLE`。
2. blockwise 若继续追，只做 prefix-conditioned probe：`### CUDA_KERNELS`、code fence、stop token、格式 token rank/logprob；不要再用 MLP-only L2 proxy 推断 full eval。
3. QuaRot 若继续追，先做 logit/KL 与 draft-target agreement 小样本 sanity；不要重跑旧 20260530 坏 checkpoint。
4. SmoothQuant 暂停扩展；MLP-only 和 Non-LA 都没有超过 RTN Non-LA per-channel。

## 证据路径

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
