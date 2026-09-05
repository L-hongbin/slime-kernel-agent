# Rollout 加速实验索引

留存结果来自 Qwen3.6/DrKernel 的特定评测配置。相同 score 不是统计等价证明，跨设备、backend、reward worker 或输出长度的 wall 变化也不是单变量加速比。当前 Qwen3.8 训练合同见[训练配置](../qwen38/training.md)

## 各问题的唯一入口

| 问题 | 结论边界 | 证据 |
| --- | --- | --- |
| Prefix cache、history 与 routing | 命中率提高不保证 end-to-end wall 下降；history 策略会改变长度和生成语义 | [Prefix cache](handoff_prefix_cache.md) |
| EAGLE / speculative decoding | 保留该模型的实测；不能外推到当前 no-spec Qwen3.8 | [Speculative decoding](handoff_specdec_drkernel.md) |
| Reward stage 与并发 | 分开看 candidate timer、reference timing、队列和 decode；未分解项不等于编译时间 | [Reward 耗时与并发](handoff_reward_speedup.md) |
| FP8、W8A8、W4A16 | 完整 run、partial run、同条件与跨条件比较分开；模型质量结论限于观测协议 | [低精度实验](handoff_low_precision.md) |

## 设备与 backend 的观测对照

A100/A800 当时使用 Triton linear-attn，H20 使用 FlashInfer。以下保留原始 run 数字，设备与 backend 的影响没有分离

| 比较 | request cap | wall | reward score | 解释范围 |
| --- | --- | --- | --- | --- |
| A800 / A100 | 64 / 64 | 6608s / 6421s | 0.375 / 0.375 | 单次观测接近，未证明统计等价 |
| A800 / H20 | 96 / 96 | 5077s / 4746s | 0.384 / 0.381 | 同时改变设备和 linear-attn backend |

| 指标                         |  A800 (C64) |  A100 (C64) |
| ---------------------------- | ----------: | ----------: |
| wall(800 eval)               |       6608s |       6421s |
| s/it                         |        8.26 |        8.03 |
| response_len/mean            |      8285.6 |      8606.3 |
| reward score                 |       0.375 |       0.375 |
| decode running-req median    |          22 |          26 |
| gen throughput mean / median | 1759 / 1738 | 1817 / 1882 |
| spec_accept_length           |      3.2367 |      3.2389 |
| prefix_cache_hit_rate        |      45.94% |      45.85% |

| 指标                         | A800 (C96) triton | H20 (C96) flashinfer |
| ---------------------------- | ----------------: | -------------------: |
| wall(800 eval)               |             5077s |            **4746s** |
| s/it                         |              6.35 |             **5.93** |
| response_len/mean            |            8366.2 |               7993.4 |
| reward score                 |             0.384 |                0.381 |
| decode running-req median    |            **90** |                   34 |
| gen throughput mean / median |   **2316 / 2915** |          2158 / 2470 |
| spec_accept_length           |            3.2390 |               3.2432 |
| prefix_cache_hit_rate        |            40.51% |               42.41% |

原始 run identity：A100 C64 `20260531_052414_...C64.A100`；A800 C64 `20260531_051603_...C64`；A800 C96 `20260529_132454_...rm16`；H20 C96 `20260602_042421_...C96.H20.linear-fi`。它们位于 `checkpoints/Qwen3.6-27B/` 的原始节点目录，不能据文档断言文件当前仍可访问

H20 的 `20260601_231948` run 有 106 条 KernelGymRequestError、仅 747 条 reach-T3，已从对照排除；该故障 run 不能用于设备或精度结论
