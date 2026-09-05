# GDN 后端 logprob 失配与 sequence MIS 空训练步

Qwen3.6-27B 的历史 RL run 在 sequence MIS `[0.999,1.001]` 窄阈值下拒绝约 96% 的序列，出现有效训练 token 全为零的步骤。同权重、零 staleness 的隔离对照中，将 rollout GDN 后端由 FlashInfer 改为 Triton 后，序列 reject 降为零，token logprob 差异明显减小

这条证据确认了该软件版本下后端选择对失配的影响。真实输入 replay 也复现了差异，但未通过单独修复状态别名、舍入或归一化路径的配对实验确定唯一底层原因；上游类似 issue 和跨层误差增长只能提供机制线索

## 同权重对照

两次均开启 EAGLE，rollout_id=0，两侧使用 base 权重。序列 reject 与逐 token 越阈比例是不同统计量

| | flashinfer(baseline) | **triton** |
| --- | --- | --- |
| reject_rate | 0.9647 (246/255) | **0.0000 (0/253)** |
| effective_sequences | 9 | **253(全部)** |
| mean\|lr\| | 0.0284 | **0.00453**(↓6×) |
| max\|lr\| | 9.22 | **1.67**(↓5.5×) |
| 结构性"自信 token 分歧" | 有(缩进 token sglang 99.5% / megatron 0.3%,5.85 nats) | **消失**(worst=` int` 1.7 nats) |

Triton 后仍有非零 token logprob 差异，reject=0 只表示该批序列聚合值满足门禁，不能写成数值完全一致。本文不推断新版本 FlashInfer 的表现，也不依赖 SGLang 默认后端；复核时应显式选择后端

## 真实输入与合成输入的区别

- 同 prompt、greedy 输出的一致前缀 token 对照：FlashInfer/Triton per-token logprob 平均绝对差 0.0175，最大 0.052，94.4% token 越过 0.001
- 固定 2048-token 真实序列 teacher-forced 前向：hidden-state RMS 差从 L0=0、L20=1.4e-3、L32=4.5e-3 增至 L63=0.084；rescore logprob 平均绝对差 0.0386、最大 4.78。该实验同时说明问题在 prefill/rescore 中可见
- layer4 捕获的真实 q/k/v replay：decode output 最大差 3.2e-3，而所测合成输入的两后端差异为零。真实 gating 中 39.5% head 的 alpha>0.99、14.7%>0.999，合成输入未覆盖相同范围
- 合成 N=2000 多步、T=1024 prefill 和 MTP committed-state 对照差异很小；它们只验证所覆盖的输入，不能据此排除全部 kernel 数值问题

这些记录支持误差随真实模型前向传播积累；没有独立性证据支持把各层当作随机游走，也没有证明某个上游状态池别名问题就是本 run 的唯一根因

## 已做的排除与边界

| 对照 | 观察 | 能支持的结论 |
|---|---|---|
| 双侧 fp32 lm-head | reject 仍约 0.984 | 单独改变 lm-head 精度未解决此问题 |
| SSM/conv/gate dtype 核对 | 两侧 SSM 为 fp32，conv 为 bf16，gate 为 fp32 | 未发现这些配置字段的 dtype 不一致 |
| Megatron FLA 与 FlashQLA 重跑 | reject/mean/worst-K 接近 | 该次训练 kernel 切换未消除失配 |
| 两后端均开 EAGLE | Triton 对照 reject=0 | EAGLE 不是该故障的充分解释 |
| 无 CUDA Graph 对照 | 状态差异仍可复现 | 故障不依赖 CUDA Graph 才出现 |

## MIS 空步的工程含义

`sequence_mis` 在已收集的 batch 上修改 loss mask；dynamic filter 收足 group 不保证 MIS 后仍有有效 token。如果所有 mask 被置零，防止除零的 loss 实现仍可能让优化器执行一个没有有效策略梯度的步骤，step 计数、动量和调度器行为需另行检查

统计入口为 [kernel_filter.py](../../examples/kernel_agent/kernel_filter.py)，保留 `mis_reject_rate`、log-ratio 及 effective sequence/token 指标；一致性检查为 [test_sequence_mis_consistency.py](../../slime/tests/test_sequence_mis_consistency.py)。本案例不表示当前 Qwen3.8 predictive-DPPO recipe 使用相同 hard MIS 门禁，当前训练契约见 [Qwen3.8 training](../qwen38/training.md)

## signed log-ratio 的正确读法

固定 prefix，令采样分布为 p、评分分布为 q，则 `E_p[log q − log p] = −KL(p || q) ≤ 0`。这个恒等式针对期望 log-ratio；实际日志还需核对 temperature、截断采样、mask 和序列聚合是否对应同一分布定义

它不能推出每个 token 或每条序列的 ratio 都小于 1。在共同完整支持集上，算术期望 `E_p[q/p] = 1`；sequence MIS 常用的 `exp(mean(log_ratio))` 又是几何聚合，不能与这个算术期望混用。因此 `ratio_mean≈0.995` 和该批 `ratio_max<1` 是经验观察，不能由 Gibbs 不等式推出“永远只在下界拒绝”或“训练策略变尖必然使该偏置增大”

## 复核入口

[整模型 logprob 对照](../../examples/kernel_agent/test/evidence_logprob.py) 保留为独立验证器，需要两份模型加载资源。先同步代码、模型、后端配置和 tokenizer，再记录 fingerprint；本次文档整理未重跑模型

历史证据位于节点本地 `/ms/FM/chenshuailin/mis_debug/`：`rollout_0.pt`、`worst_tokens_rollout0.jsonl`、`worst_tokens_triton.jsonl`、`lp_{flashinfer,triton}.json`、`xlayer_lp_{fi,tri}.pt`、`gdn_real_capture_v3.pt`。其可用性取决于原节点保留情况
