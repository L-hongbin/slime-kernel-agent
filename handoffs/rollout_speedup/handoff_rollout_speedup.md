# Rollout 加速 — 总览

> **阅读指南**
> 本文是 rollout 加速各方向的**导航文档**。每个方向只列核心结论和关键数据，详细实验记录请点击各小节的"详见"链接。

---

## 背景

- **模型**：Qwen3.6-27B（16 full-attn + 48 linear-attn/mamba, hybrid）
- **评测 workload**：KernelBench L1, 800-sample eval, 多轮 kernel 编辑（max_turns=3）
- **硬件**：8× A800-80GB（除非另行说明）
- **推理框架**：SGLang 0.5.12.post1 + EAGLE 投机采样

### Wall time 瓶颈在哪？

<!-- 理解加速方向之前，需先理解 wall time 的组成 -->

| 阶段                                   | 受 cache 影响？ | 典型耗时                                               | wall 占比 |
| -------------------------------------- | :-------------: | ------------------------------------------------------ | :-------: |
| **Prefill**（prompt 编码）             |        ✅        | 几千 token @ 5k–20k tok/s ≈ 亚秒~2s                    |  **小**   |
| **Decode**（逐 token 生成）            |        ❌        | response 4–8k tok @ ~1400–1850 tok/s ≈ 数秒            |  **大**   |
| **KernelGym 评测**（编译 + benchmark） |        ❌        | 编译 + 30 warmup + 50 perf trials ≈ 数秒~数十秒/kernel |  **大**   |

<!-- **核心结论：wall time 由 decode 吞吐 + KernelGym 外部评测共同主导**，prefill 是配角。因此 prefix cache（只加速 prefill）对 wall 无效，而提升 decode 吞吐的方向（量化、投机解码）才是有效杠杆。 -->

---

## 加速效果总览

同一 KernelBench L1 800-sample eval workload 下，各方向叠加后的累积效果：

| 方向                | 关键改动               | Wall time | 累积加速 | 吞吐 (tok/s) | Score |
| ------------------- | ---------------------- | --------: | -------: | -----------: | ----: |
| Baseline            | —                      |   1:58:37 |    1.00× |        ~3730 | 0.375 |
| + SpecDec (EAGLE)   | draft=4                |   1:36:09 |    1.23× |        ~4600 | 0.381 |
| + Reward worker ×16 | 8→16                   |   1:24:37 |    1.40× |         5231 | 0.384 |
| + W8A8 量化         | RTN Non-LA per-channel |   1:16:12 |    1.56× |         5790 | 0.408 |

注意：各行是**叠加关系**（每行 = 上一行 + 本行改动），不是独立实验。
<!-- > - Baseline 的 1.00× 指 TP4、prefix cache 已启用但对 wall 无效。
> - Score 为 reward score，判优看 score 不看 wall 单点。
> - Reward worker 行排在 W8A8 之前，是因为现有 W8A8 run 已默认带 rm16 —— rm16 是其前置条件。 -->

---

## 1：Prefix Cache

**详见**：[handoff_prefix_cache.md](./handoff_prefix_cache.md)

### 做了什么

试图通过提高 SGLang prefix-cache 命中率给多轮 rollout 加速。尝试了两条路线：
1. **TP2 vs TP4**：减少 engine 数量提高命中率，但 TP4 本身更慢
2. **preserve_thinking**：修复 no-norm chat template 的 thinking 归一化 + hicache ratio 调优，命中率提到 ~53%

### 结论

**Cache hit rate ≠ 推理效率。** 命中率从 21.5% 到 53% 差距巨大，但 wall time 全在 ~2h、s/it 全 ~8。原因是 prefill 只占 wall 的很小部分（见上方"瓶颈"表），提升 cache 命中对 decode 和 KernelGym 评测无影响。

**缓存不是加速杠杆，不再继续。**

---

## 2：投机解码 (SpecDec)

**详见**：[handoff_specdec_drkernel.md](./handoff_specdec_drkernel.md)

### 做了什么

使用 EAGLE 投机采样加速 decode 阶段。当前配置：`num-steps=3, topk=1, num-draft-tokens=4`。

### 结论

| 配置                |   Wall time | Score | Accept length |
| ------------------- | ----------: | ----: | ------------: |
| 无 spec（baseline） |     1:58:37 | 0.375 |             — |
| EAGLE               | **1:36:09** | 0.381 |          3.23 |

- **Wall 加速 1.23×，score 无损**，是目前所有方向里最干净的加速。
- `accept_length ≈ 3.23` 已逼近 `num_draft_tokens=4` 天花板

<!-- ### 未尽路线（按优先级）

| 优先级 | 方案                                           | 需要训练？ | 预期收益              |
| :----: | ---------------------------------------------- | :--------: | --------------------- |
| **P0** | EAGLE **tree 化**（topk=4/8, steps=4/5）       |     否     | accept 3.2 → 4~5      |
| **P0** | **NGRAM / suffix**（特别适合多轮复刻代码场景） |     否     | 复刻段 accept 10~18   |
|   P2   | **EAGLE3** + 域内 draft                        |   需训练   | accept 4~6+，上限最高 | --> |

---

## 3：Reward Server 并发

**详见**：[handoff_reward_server_concurrency.md](./handoff_reward_server_concurrency.md)

### 做了什么

将 KernelGym reward worker 从 8 增加到 16（前提：需先修 KernelGym compile-cache bug，见 `KernelGYM-reward-only/bug_report.md`）。

### 结论

| 配置                 |   Wall time | Score | 编译率 |
| -------------------- | ----------: | ----: | -----: |
| 8 worker（baseline） |     1:36:09 | 0.381 |  57.6% |
| 16 worker            | **1:24:37** | 0.384 |  56.1% |

- **Wall 加速 1.14×，score 和编译率均持平。**
- 两个 run 的 decode token 量相同 → 692s wall 下降**全部来自 RM 评测提速**。

### 天花板与风险

- Worker 翻倍消除了 reward 队列阻塞，但瓶颈随即转移到 decode（GPU 算力是硬上限），继续堆 worker 边际收益递减。
<!-- - **未修 bug 时堆 worker 会触发并发竞态**，导致编译大面积失败、score 暴跌，是净亏。 -->

---

## 4：低精度推理 (W8A8 / FP8 / INT4)

**详见**
- → [handoff_low_precision.md](./handoff_low_precision.md)（FP8 / INT8 / INT4 低精度加速 + 精度总表）
- → [handoff_drkernel_w8a8_rollout.md](./handoff_drkernel_w8a8_rollout.md)（W8A8 实测 + 复现 recipe）
- → [handoff_w8a8_speedup_survey.md](./handoff_w8a8_speedup_survey.md)（未尝试技术的 ranked 决策矩阵）

### 已验证的量化方案

| 方案                            |   Wall time |     Score | 结论                                         |
| ------------------------------- | ----------: | --------: | -------------------------------------------- |
| BF16（baseline）                |     1:24:37 |     0.384 | —                                            |
| **W8A8 RTN Non-LA per-channel** | **1:16:12** | **0.408** | ✅ **推荐：最快且精度无损**                   |
| FP8                             |     1:20:10 |     0.393 | ✅ 可用，但 A800 无 FP8 tensor core，仅限 H20 |
| W8A8 RTN all（含 linear-attn）  |     1:10:45 |     0.318 | ❌ 精度大幅下降                               |
| W4A16 AWQ                       |     1:25:49 |     0.354 | ❌ 精度大幅下降                               |

### 关键发现

- **不能量化 linear-attn 层**：all-linear 方案 score 从 0.384 跌到 0.318。
<!-- - **A800 只能走 INT8 路线**：无 FP8 tensor core。H20 可走 FP8。
- W8A8 RTN Non-LA per-channel 是当前**效率与精度的最佳平衡点**。 -->

---

## 补充对比：A100 vs A800 vs H20 设备差异

**详见**：[handoff_rollout_device_efficiency.md](./handoff_rollout_device_efficiency.md)

| 对比         | Wall 差异 | 吞吐差异 |          Score | 结论                               |
| ------------ | --------: | -------: | -------------: | ---------------------------------- |
| A100 vs A800 |      2.9% |    ~6.9% |       均 0.375 | 推理效率一致                       |
| A800 vs H20  |     ~6.5% |     同档 | 0.384 vs 0.381 | 吞吐同档，H20 可用 flashinfer 后端 |

> 注意：A100/A800 使用 SGLang triton linear-attn backend，H20 使用 flashinfer。两边后端不同是硬件限制（flashinfer 的 linear-attn 不支持 A800）。

<!-- ---

## 总结论与下一步

### 有效杠杆（按收益排序）

1. **W8A8 量化**（+1.11× 边际加速）：当前最佳方案 RTN Non-LA per-channel，score 无损。Survey 里还有 PD-disaggregation / fully-async rollout 等未尝试方向。
2. **EAGLE 投机解码**（1.23× 边际加速）：配置已饱和，tree 化 / NGRAM / EAGLE3 可继续挖掘。
3. **Reward worker ×16**（+1.14× 边际加速）：已基本到顶，不再投入。

### 无效杠杆

- **Prefix cache**：对 wall 无效，不继续。

### 判优原则

> **⚠️ 任何加速方案的判优看 score，不看 cache hit 或 wall 单点。**
>
> Wall 被 decode + KernelGym 共同主导，区分度低；曾有 rm16 被 wall 数字误导的先例。 -->
