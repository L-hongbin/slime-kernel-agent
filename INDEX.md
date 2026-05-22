# INDEX

## Data Conversion

- `scripts/data/convert_verl_to_slime.py`: converts VERL parquet records to the `ground_truth` + `extra_info` schema used by `scripts/debug.sh`.
- `tests/utils/test_convert_verl_to_slime_data.py`: unit coverage for the converter output and slime `Dataset` loading with `prompt_key=ground_truth`, `metadata_key=extra_info`, and chat-template rendering.

## DrKernel Plugin

- `handoffs/in_progress/HANDOFF_DRKERNEL_SLIME_PLAN.md`: current migration plan and handoff for implementing DrKernel on slime, including current rollout status and next KernelGYM reward step.
- `slime_plugins/drkernel/design-docs/dynamic_prompt_templates.md`: design for fragment-based dynamic prompt templates in the DrKernel custom rollout.
- `slime_plugins/drkernel/extract.py`: extracts DrKernel/KernelGym-ready kernel submissions from assistant responses.
- `slime_plugins/drkernel/kernelgym_rm.py`: async KernelGym HTTP client and optional single-turn custom RM wrapper.
- `slime_plugins/drkernel/rollout.py`: DrKernel custom rollout and prompt renderer; template selection is restricted only by `sample.metadata["template_allowed"]`.
- `slime_plugins/drkernel/prompt_templates/single_turn_v1.yaml`: first dynamic prompt profile set for DrKernel single-turn rollout templates.
- `scripts/drkernel/summarize_kernelgym_eval.py`: CLI to summarize KernelGym eval artifacts (file or run dir); writes review `sample_*.txt` under `dumps/rollout_data/`.
- `tests/utils/test_drkernel_prompt_templates.py`: unit coverage and real-data formatted prompt dump for DrKernel prompt rendering.
- `tests/utils/test_drkernel_eval_summary.py`: unit coverage for KernelGym eval summary metric extraction.
- `checkpoints/drkernel_prompt_debug/formatted_prompt_examples.txt`: generated real-data prompt examples for manual review.

## Run Configs

- `scripts/debug.sh`: current debug training/eval configuration for converted drkernel/kernelbench parquet data.
- `scripts/eval_kernelbench_level1.yaml`: slime eval config for converted KernelBench L1 validation data.
- `checkpoints/Qwen3.5-9B/run_20260515_122920.log`: successful Qwen3.5-9B DrKernel eval rerun on `.67`; `raysubmit_v7whRX3AfMxtL3YY`, KernelBench L1 score `0.035`.
- `tests/utils/test_eval_config.py`: unit coverage for eval config prompt/response/context length fields.
