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
  效率瓶颈分析（权威 run：20260609_134303 bf16+gradf32+TIS+RLOO，41 step 稳态，
  step median ~15.7min：actor train 8.3min、rollout/wait 7min）；abort/drain 已基本消失
  （median 1s），新暴露 checkpoint save 每 10 step ~384s；hash 问题独立成章（历史 FP8 A/B）。
- `handoffs/in_progress/handoff_checkpoint_save_efficiency.md`: save step ~384s 尾巴拆解
  （torch_dist ~285s + HF ~90s + ~9s wrapper）；2026-06-10 9B TP4×CP2×PP1×DP1 A/B 已验证
  persistent-worker 才能启用 async，且 `dp_reshardable` 首次前台 save 7.6s vs fully+worker 139.5s；
  27B 脚本已切默认 dp_reshardable，并新增 `LOAD_DIR` 支持从旧 fully ckpt resume 后转存新 dp ckpt。
- `handoffs/in_progress/handoff_megatron_train_accel.md`: Megatron 训练加速开关清单（含 TL;DR 决策表）；
  已开（overlap-param-gather/async-save/save-hf）、待评估（PP 气泡 VPP/layout 最有潜力、recompute 放松、
  manual-gc 收益边际需实测）、实验项（FP8 `--fp8-format`/CUDA-graph）；
  结论 **TP comm overlap 与 slime always-varlen 不兼容，搁置**；cross-entropy fusion/MoE overlap 不适用。
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
- `handoffs/complete/handoff_malware_cleanup_20260608.md`: node62/node64/node69/node70
  宿主病毒清理记录；含挖矿链、node64 Perl/httpd 后门、cron/profile 清理、复核结果、取证文件路径和遗留风险。

## DrKernel Plugin

- `slime_plugins/drkernel/README.md`: DrKernel custom rollout plugin code,
  prompt templates, KernelGym RM, extraction, and design docs.

## Scripts

- `scripts/eval_drkernel/README.md`: eval/debug launch wrappers (incl. H20 + summarizer,
  merged from former `scripts/debug/` and `scripts/drkernel/`), fixed-shape benches, and
  one-off low-precision evidence probes.
- `scripts/eval_drkernel/eval.27b.t1.tp4.eagle.H20.sh`: single-turn KernelBench L1
  eval of a trained HF checkpoint (`EVAL_HF_CKPT=.../hf/iter_N`), one H20 node per
  checkpoint for parallel per-iteration evals.
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
- `scripts/check_kernelgym_health.py`: standalone KernelGym
  `/health` preflight for DrKernel training runs.
- `tools/summarize_run_perf.py`: parse slime `run.log` perf dicts and print
  ASCII-table average/median/max for step time, actor train TFLOPS, and rollout time.

## Run Configs

- `setup_env.sh`: repo setup entrypoint; installs slime/debugpy and verifies
  FlashInfer GDN Cutlass DSL dependencies.
- `multi_node_train.py`: generic Ray multi-node launcher. Reads local
  `HOSTFILE`/`hostfile`, treats the first node as head, ssh-starts worker nodes,
  checks all nodes for active GPU compute processes, checksum-syncs small launch
  inputs to workers, waits for all Ray nodes, and then runs the target train
  script with the Ray cluster already up.
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
