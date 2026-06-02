# scripts/drkernel

DrKernel / KernelGym 评测产物工具入口。

- `summarize_kernelgym_eval.py`: 汇总 KernelGym eval artifact 或 run dir，抽取
  compile/correct/fast 等指标，并在 `dumps/rollout_data/` 下写出可人工复核的
  `sample_*.txt`。
