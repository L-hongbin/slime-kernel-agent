# INDEX

## Stable Docs

- `AGENTS.md`: repository-level rules and collaboration policy.
- `RUNTIME.md`: stable runtime facts for nodes, endpoints, shared paths, data paths,
  and common run entrypoints; experiment-specific settings stay in handoffs.

## Handoff Hubs

- `handoffs/in_progress/handoff_drkernel_slime_plan.md`: DrKernel-on-slime
  plan/status hub; links the active template, quantization, rollout, and training
  follow-ups.
- `handoffs/rollout_speedup/handoff_rollout_speedup.md`: rollout 加速总入口；
  归档 prefix cache、low precision、SpecDec、reward concurrency、device efficiency
  等方向。
- `handoffs/complete/handoff_bf16_baseline_jump_root_cause.md`: May24→May28
  BF16 baseline jump root cause; resolved by the SGLang Qwen3.5 GDN stride fix.

## DrKernel Plugin

- `slime_plugins/drkernel/README.md`: DrKernel custom rollout plugin code,
  prompt templates, KernelGym RM, extraction, and design docs.

## Scripts

- `scripts/eval_drkernel/README.md`: eval/debug launch wrappers (incl. H20 + summarizer,
  merged from former `scripts/debug/` and `scripts/drkernel/`), fixed-shape benches, and
  one-off low-precision evidence probes.
- `scripts/analysis/README.md`: offline eval dump, run-log, concurrency, and
  prefix-cache analysis scripts.
- `scripts/quantize/README.md`: quantization producers, checkpoint gates, runtime
  patches, calibration builders, and quantization evidence utilities.
- `scripts/data/convert_verl_to_slime.py`: VERL parquet to slime
  `ground_truth`/`extra_info` converter.

## Run Configs

- `scripts/debug.sh`: current debug train/eval configuration for converted
  DrKernel/KernelBench parquet data.
- `scripts/eval_kernelbench_level1.yaml`: slime eval config for converted
  KernelBench L1 validation data.

## Tests And Review Artifacts

- `tests/utils/`: unit coverage for data conversion, eval config, DrKernel prompt
  rendering/extraction/RM/throttle, SGLang context caps, and quantization utilities.
- `checkpoints/drkernel_prompt_debug/formatted_prompt_examples.txt`: generated
  real-data prompt examples for manual review after prompt-template changes.
