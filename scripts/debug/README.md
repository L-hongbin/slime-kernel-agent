# scripts/debug

调试启动脚本和一次性证据脚本入口。启动脚本可能拉起 Ray/SGLang/KernelGym；
运行前按 `RUNTIME.md` 和对应 handoff 做配置 sanity check。

## Eval Wrappers

- `debug.9b.sh`: Qwen3.5-9B DrKernel eval/debug wrapper。
- `debug.27b.sh`: Qwen3.6-27B BF16 baseline wrapper。
- `debug.27b.tp4.eagle.sh`: Qwen3.6-27B TP4 + EAGLE wrapper。
- `debug.27b.tp4.eagle.A100.sh`: A100 对照 wrapper。
- `debug.27b.w8a8.sh`: W8A8 RTN smoke/eval wrapper。
- `debug.27b.tp4.eagle.w8a8.sh`: W8A8 + EAGLE full-eval wrapper。
- `debug.27b.tp4.eagle.awq_w4a16.sh`: AWQ W4A16 + EAGLE wrapper，包含
  pre-eval checkpoint/runtime gate。

## Fixed-Shape Benches

- `bench_sglang_wall.py`: 固定请求形状的 SGLang HTTP wall benchmark。
- `bench_w8a8_spec_components.py`: W8A8 + EAGLE component-cost microbench。

## INT8 / Blockwise Diagnostics

- `analyze_g128_lengths.py`: paired BF16/per-channel/B128 eval dump 长度分析；
  历史文件名中的 `g128` 对应 blockwise B128。
- `analyze_blockwise_output_structure.py`: 分析 blockwise 独有长尾、late marker
  和结构差异。
- `probe_blockwise_format_logits.py`: target-only first-token logprob probe，用于定位
  blockwise early-basin shift。

## W4A16 AWQ Diagnostics

- `analyze_w4_awq_eval_gap.py`: 汇总 W4A16 full-eval 与中间指标 gap。
- `probe_w4_awq_error_propagation.py`: W4 hidden/error propagation probe。
- `probe_w4_asym_zp_sensitivity.py`: ASYM zero-point 敏感性 probe。
- `probe_sglang_wna16_runtime_fidelity.py`: SGLang WNA16 ASYM runtime fidelity probe。
- `probe_w4_codedomain_metric_kl.py`: code-domain metric/logit KL probe。
- `probe_w4_metric_stage_kl.py`: metric-stage KL probe。
- `probe_w4_trajectory_compounding.py`: trajectory compounding probe。

## Prompt Sanity

- `render_prompt_check.py`: 渲染并检查 DrKernel prompt 模板，适合改模板后做人工复核。

## 结论入口

- `handoffs/rollout_speedup/handoff_low_precision.md`: FP8 / INT8 / INT4 low-precision
  rollout 总结。
- `handoffs/rollout_speedup/handoff_drkernel_w8a8_rollout.md`: W8A8 当前详细 handoff。
- `handoffs/rollout_speedup/handoff_w4a16_awq.md`: W4A16 AWQ 当前详细 handoff。
