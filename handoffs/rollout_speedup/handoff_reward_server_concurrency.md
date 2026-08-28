# Rollout 加速：Reward Server 并发

> **环境**：
> 模型 Qwen3.6-27B，BF16 精度 + EAGLE 投机采样 

## 1. 结论

**Reward worker (GPU) 8→16：获得 ~1.14× wall 提速，收益不显著，本方向无功而返。**

| Run | Worker | Wall | s/it |
|-----|-------:|-----:|-----:|
| baseline（8 worker） | 8 | 1:36:09 | 7.21 |
| 16 worker | 16 | **1:24:37** | **6.35** |

<!-- spec_accept_length 3.23→3.24，也持平。 -->

**为什么 2× worker 只换来 1.14× wall？**

Worker 翻倍成功消除了 reward 队列阻塞（decode 并发从中位数 32 跃升到 90），但瓶颈随即**转移到 decode**——GPU 算力是硬天花板，并发 3× 只能多吐 14% 的 token。继续堆 reward worker 的边际收益会越来越小。

---

## 2. Decode 并发分析

![decode concurrency rm8 vs rm16](../images/decode_concurrency_rm8_vs_rm16.png)

<!-- > 生成脚本：`scripts/analysis/plot_decode_concurrency.py` -->

| 指标 | 8 Worker | 16 Worker |
|------|--------:|---------:|
| Decode 并发中位数 | 32 | **90** |
| Decode 并发均值 | 41.8 | 66.0 |
| Gen throughput 均值 | 2033 tok/s | 2316 tok/s |

**8 worker** 中段长时间饿到 ~20–40 并发；**16 worker** 全程钉在 ~90/96（近满载）。
虽然 16 worker 消除了 reward 队列阻塞，但 decode 并发并没有显著提升，说明 Eagle 投机采样下，32的中位并发数仍可以吃完几乎所有算力。

<!-- ### 为什么并发 3× 却只有 1.14× 加速？

decode 并发从 32 涨到 90（≈3×），但 gen throughput 仅从 2033 涨到 2316 tok/s（1.14×）。根源是 **GPU 算力天花板**：
- continuous batching 下，batch size 32→90 时计算已趋近 GPU 峰值，更大 batch 几乎无法进一步提升 per-token 吞吐，受限于显存带宽和 SM 占用。
- 两个 run 的尾部都有 ramp-down 阶段（sample 逐渐完成，并发自然回落），进一步稀释整体收益。 -->

---

## 3. 补充实验：SGLang 最大并发 96 vs 64 vs 32

**动机**：§2 中 8 worker（C96）的并发曲线存在部分时间段 rollout 并发数降到 10 以下的波谷（见上图 rm8 曲线中段）。猜测是高并发上限导致 burst 式资源争抢，尝试降低 `--sglang-max-running-requests` 来平滑并发、消除波谷。

固定 **reward worker=8**、TP=4、EAGLE、800-sample eval，比较 max running requests 从 96 降到 64 / 32。

**结论：降低 SGLang 最大并发不仅牺牲了峰值 decode throughput，也没有避免 rollout 并发波谷，只会更慢。**

<!-- 注意：C64/C32 目录名带 `.rm16`，但本次对比按 reward server 实际配置归类为 8 worker。 -->

![sglang concurrency c32 c64 c96](../images/sglang_concurrency_c32_c64_c96.png)

> 生成脚本：`scripts/analysis/plot_sglang_concurrency_c32_c64_c96.py`

### 核心效率指标

| Run | SGLang max req | Wall | s/it | 相对 C96 |
|-----|---------------:|-----:|-----:|---------:|
| C96（baseline） | 96 | **1:36:09** | **7.21** | **1.00×** |
| C64 | 64 | 1:50:08 | 8.26 | 0.87× |
| C32 | 32 | 2:07:26 | 9.56 | 0.75× |

### Decode 饱和度指标

| 指标 | C96 | C64 | C32 |
|------|----:|----:|----:|
| 并发中位数 | **32** | 22 | 27 |
| 并发均值 | **41.8** | 30.4 | 23.4 |
| 并发 P90 | **92.1** | 64 | 32 |
| 达到 ≥90% 上限占比 | 15.2% | 22.5% | **44.9%** |
| Gen throughput 均值 | **2033** | 1759 | 1722 |
| Gen throughput P90 | **3003** | 2771 | 2177 |

<!-- Gen throughput P50: C96=2024, C64=1738, C32=1939; P99: C96=3325, C64=2958, C32=2262 -->

<!-- C64/C32 在 8 worker 条件下都没有换来 wall 收益：它们更常贴近自身较低上限，但 decode 并发均值和 token throughput 上界都低于 C96。C32 的 P50 throughput 高于 C64，但 P90/P99 被 32 并发上限明显压低，整体 wall 仍继续变慢到 2:07:26。这里的判断不是 16 worker 下的 decode-saturated 场景，而是在同样 RM-bound 的 8 worker 场景里，降低 SGLang max running requests 仍然会损失吞吐。 -->

C96 的 decode 并发均值和 throughput 上界均高于 C64/C32。降低 max req 使引擎更频繁贴近自身较低的上限，整体吞吐下滑。

---

## 4. 参考 Run

| 标签 | 路径 |
|------|------|
| 8 Worker（baseline） | `checkpoints/Qwen3.6-27B/20260529_075505_newSlimeKG.tp4.eagle_ctx65536_n8_summ1600/` |
| 16 Worker | `checkpoints/Qwen3.6-27B/20260529_132454_newSlimeKG.tp4.eagle.rm16_ctx65536_n8_summ1600/` |
| C64 | `checkpoints/Qwen3.6-27B/20260531_051603_newSlimeKG.tp4.eagle.rm16.C64_ctx65536_n8_summ1600/` |
| C32 | `checkpoints/Qwen3.6-27B/20260531_073240_newSlimeKG.tp4.eagle.rm16.C32_ctx65536_n8_summ1600/` |
