# Rollout 推理效率设备对比: A100 vs A800 EAGLE

挂在 `handoffs/in_progress/handoff_rollout_speedup.md` 的补充 handoff。目的:对比同一组
drkernel Qwen3.6-27B 800-sample eval 在不同 rollout 设备上的推理效率差异。

## 结论

同为 TP4、2 个 SGLang engine、EAGLE、rm16、max-running-requests=64、ctx65536:

- A100 run 的 800-eval wall 是 **1:47:01(6421s)**,A800 run 是 **1:50:08(6608s)**,
  wall 差异只有 **2.9%**。
- A100 run 输出更长(response mean 8606 vs 8286,+3.9%)。按总 response token 归一后,
  A100 是 **1072 tok/s**,A800 是 **1003 tok/s**,差异约 **6.9%**。
- SGLang decode 日志同量级:gen throughput mean **1817 vs 1759 tok/s(+3.3%)**,
  median **1882 vs 1738 tok/s(+8.3%)**,p90 基本相同(**2768 vs 2771 tok/s**)。
- 质量与投机解码行为基本等价:score 都是 **0.375**,spec_accept_length 都是 **~3.24**,
  prefix-cache hit 都是 **~45.9%**。

判断:这两条 run 不支持 A100 和 A800 存在可操作的推理效率差异。考虑到输出长度、并发轨迹、
reward endpoint 和长尾样本都有轻微差异,上述 3-7% 的差别应视为同一量级噪声;结论是
**A100 和 A800 在这组 rollout 配置下推理效率一致**。

## 对比对象

| 口径 | A100 run | A800 run |
|---|---|---|
| checkpoint | `checkpoints/Qwen3.6-27B/20260531_052414_newSlimeKG.tp4.eagle.rm16.C64.A100_ctx65536_n8_summ1600` | `checkpoints/Qwen3.6-27B/20260531_051603_newSlimeKG.tp4.eagle.rm16.C64_ctx65536_n8_summ1600` |
| rollout host | `172.80.0.6` | `192.168.16.22` |
| reward URL | `http://192.168.16.39:20111` | `http://192.168.16.40:20111` |
| SGLang shape | 8 GPU, 2x TP4 engine | 8 GPU, 2x TP4 engine |
| max req / graph BS | 64 / 64 | 64 / 64 |
| specdec | EAGLE, steps=3, topk=1, draft=4 | EAGLE, steps=3, topk=1, draft=4 |
| mem fraction | 0.85 | 0.85 |

硬件型号口径 caveat: A100 标签来自 run 目录名;A800 标签来自当前 rollout-speedup hub 的环境 anchor
和无 `A100` 后缀的 baseline。两份 `run.log` 都记录了 8 GPU/NVLink/host,但没有直接打印
`nvidia-smi --query-gpu=name`。后续严格硬件 A/B 应在 launcher 里写入 GPU 型号。

## 核心指标

| 指标 | A100 | A800 | 差异 |
|---|---:|---:|---:|
| wall(800 eval) | 6421s | 6608s | 2.9% |
| s/it | 8.03 | 8.26 | 2.9% |
| response_len/mean | 8606.3 | 8285.6 | 1.039x |
| estimated response tokens | 6.885M | 6.628M | 1.039x |
| effective response tok/s | 1072.3 | 1003.1 | 6.9% |
| score | 0.375 | 0.375 | equal |
| spec_accept_length | 3.2389 | 3.2367 | equal |
| prefix_cache_hit_rate | 45.85% | 45.94% | equal |
| truncated_ratio | 1.125% | 0.875% | A100 higher |
| compile success rate | 1362/2332 = 58.4% | 1349/2374 = 56.8% | similar |

## Decode 日志指标

| 指标 | A100 | A800 | 变化 |
|---|---:|---:|---:|
| decode log count | 621 | 661 | - |
| running-req mean | **32.3** | 30.4 | +6.5% |
| running-req median | **26** | 22 | +18.2% |
| running-req p90 | 64 | 64 | equal |
| logs at >=90% cap | 21.3% | **22.5%** | similar |
| gen throughput mean | **1817 tok/s** | 1759 tok/s | +3.3% |
| gen throughput median | **1882 tok/s** | 1738 tok/s | +8.3% |
| gen throughput p90 | 2768 tok/s | **2771 tok/s** | equal |
| decode accept_len mean | 3.217 | 3.249 | similar |
| decode accept_rate mean | 0.739 | 0.750 | similar |
| full token usage mean | **0.222** | 0.202 | +9.8% |

解读:两边并发轨迹和吞吐都在同一档。median running-req 只有 22-26,接近上限的日志只有 ~21-23%,
所以这不是单纯 GPU 算力上限实验;queueing、长尾样本和 reward 交错会明显影响可见 wall。

## 下一步

1. 做严格硬件 A/B 前,先把 launcher 加上 `nvidia-smi --query-gpu=name --format=csv,noheader`
   和 driver/CUDA 版本打印,避免只靠目录名识别设备。
2. 若要隔离纯推理效率,用相同 prompt dump 跑纯 SGLang `ignore_eos` bench,或至少固定同一 reward endpoint。
3. 若目标是实际 rollout wall,当前数据说明单靠 A100 替换 A800 没有明确收益;应继续从 reward overlap、
   specdec tree/NGRAM 等 rollout 路径优化。

## 证据

- A100 run log: `checkpoints/Qwen3.6-27B/20260531_052414_newSlimeKG.tp4.eagle.rm16.C64.A100_ctx65536_n8_summ1600/run.log`
- A800 run log: `checkpoints/Qwen3.6-27B/20260531_051603_newSlimeKG.tp4.eagle.rm16.C64_ctx65536_n8_summ1600/run.log`
