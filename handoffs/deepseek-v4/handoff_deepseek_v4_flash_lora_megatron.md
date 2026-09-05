# DeepSeek-V4-Flash on slime/Megatron

本文是 DeepSeek-V4-Flash 训练、rollout 与评估材料的导航页，不复制专题文档中的实现细节、实验数字或运行状态

## 稳定边界

- 训练侧使用自定义 Megatron `LanguageModule`，承载四路 mHC hidden stream、CSA/HCA compressor、DeepSeek-V4 attention 和 EP-sharded routed experts
- attention、compressor、mHC、router 与 shared expert 在 EP rank 上复制，routed experts 沿 expert 轴分片
- LoRA 只覆盖明确支持的 attention、compressor 和可选 shared-expert 模块；adapter checkpoint、optimizer state 与 rollout serving 均有独立契约
- rollout 使用 SGLang/DSpark patch series；speculative decoding、LoRA serving、routing replay 与 probability parity 分别验收
- 正式运行参数、节点与 checkpoint identity 由 `RUNTIME.md` 和 managed launcher preflight 决定，不在本文维护某次 run 的快照

## Canonical 文档

| 责任 | 唯一维护入口 |
|---|---|
| 自定义模型与 kernel 边界 | `handoffs/deepseek-v4/dsv4_kernel_inventory.md` |
| Packed-MXFP4 storage、转换与 W4A16 compute | `handoffs/deepseek-v4/fp4_w4a16_design.md` |
| Megatron 分片、CP/EP、checkpoint 与 restore lineage | `handoffs/deepseek-v4/dsv4_megatron_sharding_contract.md` |
| LoRA training、old actor、TIS/MIS、rsLoRA 与 LoRA+ | `handoffs/deepseek-v4/lora_training_features.md` |
| LoRA adapter serving 与 CUDA Graph contract | `handoffs/deepseek-v4/lora_serve_design.md` |
| MTP/speculative decoding | `handoffs/deepseek-v4/mtp_speculative_decoding.md` |
| Train↔rollout probability parity | `handoffs/deepseek-v4/train_rollout_mismatch.md` |
| Predictive mask 与 entropy 方向 | `handoffs/deepseek-v4/predictive_entropy_collapse_20260721.md` |
| DPPO tail contraction | `handoffs/deepseek-v4/dppo_tail_contraction_20260723.md` |
| Dynamic filter 与零 reward 归因 | `handoffs/deepseek-v4/dynamic_filter_zero_reward_analysis_20260816.md` |
| 0731 fresh entropy 审计 | `handoffs/deepseek-v4/entropy_0731_v4_fresh_audit_20260816.md` |
| H200 train/H20 rollout 可行性 | `handoffs/deepseek-v4/h200_train_h20_rollout_feasibility_20260722.md` |
| KernelBench L1 LoRA 训练曲线 | `handoffs/deepseek-v4/kernelbench_l1_lora_curve_20260724.md` |
| KernelBench 单轮/三轮端点与失败归因 | `handoffs/deepseek-v4/kernelbench_eval.md` |

专题文档只维护自己的 contract。运行过程、一次性排障和节点快照应进入 ignored `local_artifacts/deepseek-v4/`，不回填为第二份设计说明

## 代码入口

| 责任 | 入口 |
|---|---|
| Formal lifecycle 与 fail-closed preflight | `scripts/dsv4/launch_formal_managed.sh` |
| Formal recipe | `scripts/dsv4/run.deepseek_v4_flash.fp4.formal.rl.sh` |
| 共享 launch core | `scripts/dsv4/_dsv4_launch_core.sh` |
| RL task arguments | `scripts/dsv4/_dsv4_task_args.sh` |
| 运行说明 | `scripts/dsv4/README.md`、`RUNTIME.md` |
| Megatron model、LoRA 与 checkpoint | `custom_kernels/deepseek_v4/megatron/`、`slime/backends/megatron_utils/` |
| Policy loss 与 routing replay | `slime/backends/megatron_utils/loss.py`、`slime/utils/ppo_utils.py`、`slime/utils/routing_replay.py` |
| DSpark patch series | `scripts/dsv4/patches/dspark_port_series/` |
| Diagnostics 与 studies | `scripts/dsv4/diagnostics/`、`scripts/dsv4/studies/` |
| Focused regression tests | `tests/deepseek-v4/` |

## 接手顺序

1. 阅读 `RUNTIME.md` 与 `scripts/dsv4/README.md`，确认当前节点、环境、checkpoint 和服务入口
2. 根据改动范围阅读上表中对应的 canonical contract，不从历史 run 推断当前配置
3. 先运行相关 focused tests，再执行 runtime fingerprint 与 managed `--prepare-only` preflight
4. rollout 或 train 单侧有疑问时，优先使用 `--debug-rollout-only` 或 `--debug-train-only` 隔离验证
5. 只有单侧门禁通过后才运行完整 loop；正式启动与 resume 统一走 managed lifecycle

## 证据规则

- 代码正确性由 focused tests 与可复现 diagnostics 共同证明
- runtime readiness 由当前 preflight、service health、checkpoint manifest 和 fingerprint 证明
- 训练结论必须引用明确 run identity、step 区间和本地 evidence 文件
- 旧日志中的节点、拓扑、LR、并行度和服务状态均不自动继承到新 run
- 新结论写入对应 canonical 文档；本页只在责任边界或入口变化时更新
