# scripts/analysis

离线分析脚本入口。这里的脚本只读 eval dump、`run.log` 或已有输出，不负责启动
Ray/SGLang/KernelGym 作业；实验结论以对应 handoff 为准。

## Eval Dump / Turn-Level

- `per_turn_acc.py`: 读取 `eval_0.pt`，统计 turn-level compile/correct/fast 指标。
- `per_turn_acc_nopt.py`: torch-free 的 `eval_0.pt` 指标读取器，适合只需要快速看
  turn-level 质量的环境。
- `per_turn_len.py`: 统计各 turn 输出长度、截断和长尾分布。

## SGLang Decode / Concurrency

- `summarize_sglang_decode_log.py`: 解析 SGLang `Decode batch` 行，汇总同 batch
  throughput、request 数和 speculative accept 指标。
- `plot_decode_concurrency.py`: 画单个 decode log 的并发/吞吐曲线。
- `plot_sglang_concurrency_c32_c64_c96.py`: 对比 C32/C64/C96 decode 并发和吞吐；
  reward worker 并发 handoff 中的图来自这个脚本。

## Thinking / Prefix Cache

- `preserve_thinking_budget.py`: 检查 thinking 预算和保留策略对 prompt 的影响。
- `preserve_thinking_prefix_match.py`: 对比 thinking 归一化前后的 prefix-cache
  匹配情况。

## 结论入口

- `handoffs/rollout_speedup/handoff_rollout_speedup.md`: rollout 加速总览。
- `handoffs/rollout_speedup/handoff_reward_server_concurrency.md`: reward worker
  并发和 C32/C64/C96 decode 对比。
