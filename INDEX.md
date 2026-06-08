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
- `handoffs/in_progress/handoff_train_step_efficiency.md`: colocate RL step 端到端
  效率瓶颈分析（27B TP4/CP2，单步 ~10.75min，rollout:train≈45:55）；瓶颈含 rollout
  长尾、训练 full recompute、薄微批 + dynamo 退回 eager。
- `handoffs/complete/handoff_bf16_baseline_jump_root_cause.md`: May24→May28
  BF16 baseline jump root cause; resolved by the SGLang Qwen3.5 GDN stride fix.
- `handoffs/in_progress/handoff_lora_support.md`: slime LoRA 训练支持评估；结论
  可行（复用 Megatron-Bridge `peft` 的 LoRA/LoRAMerge），含路线 A/B、改动点、风险。
- `handoffs/in_progress/handoff_megatron_fsdp_support.md`: 评估把 Megatron-DDP 换成
  Megatron-FSDP（`--use-megatron-fsdp`）所需开发；核心改动在权重同步 un-shard + checkpoint
  `fsdp_dtensor` 格式，含三选项消歧、改动点、风险与工作量。
- `handoffs/complete/handoff_launch_speedup.md`: **训练启动慢排查（已完成）**。根因 = 宿主中挖矿
  病毒占满 CPU；清理后启动 ~24min→~4min。含干净机器耗时分解、`numa_balancing=0` 必须保留的实验依据
  （附录）、启动前宿主健康检查脚本 `check_host_health.sh`。已合并原 NUMA 子文档。
- `handoffs/complete/handoff_malware_cleanup_20260608.md`: node64/node69 宿主
  病毒清理记录；含挖矿链、node64 Perl/httpd 后门、cron/profile 清理、复核结果、取证文件路径和遗留风险。

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
- `scripts/convert/qwen3.6-27B-torch-dist.sh`: offline Qwen3.6-27B HF to
  Megatron `torch_dist` conversion, parametric over TP/PP (`TP`/`PP` env →
  `torch_dist_tp${TP}_pp${PP}`). Passes `--mtp-num-layers 1` so the MTP head is
  converted; without it the 15 `mtp.*` HF weights are silently dropped.
- `scripts/train_drkernel/check_kernelgym_health.py`: standalone KernelGym
  `/health` preflight for DrKernel training runs.

## Run Configs

- `setup_env.sh`: repo setup entrypoint; installs slime/debugpy and verifies
  FlashInfer GDN Cutlass DSL dependencies.
- `multi_node_train.py`: generic Ray multi-node launcher. Reads local
  `HOSTFILE`/`hostfile`, treats the first node as head, ssh-starts worker nodes,
  checksum-syncs small launch inputs to workers, waits for all Ray nodes, and
  then runs the target train script with the Ray cluster already up.
- `scripts/train_drkernel/debug.t1.27b.fp8.tp4.cp2.pp2.eagle.offload.sh`:
  Qwen3.6-27B-FP8 DrKernel training smoke (`TP4×PP2×CP2`); launch multi-node as
  `python3 multi_node_train.py scripts/train_drkernel/debug.t1.27b.fp8.tp4.cp2.pp2.eagle.offload.sh`.
- `scripts/debug.sh`: current debug train/eval configuration for converted
  DrKernel/KernelBench parquet data.
- `scripts/eval_kernelbench_level1.yaml`: slime eval config for converted
  KernelBench L1 validation data.

## Tests And Review Artifacts

- `tests/utils/`: unit coverage for data conversion, eval config, DrKernel prompt
  rendering/extraction/RM/throttle, SGLang context caps, and quantization utilities.
- `checkpoints/drkernel_prompt_debug/formatted_prompt_examples.txt`: generated
  real-data prompt examples for manual review after prompt-template changes.
