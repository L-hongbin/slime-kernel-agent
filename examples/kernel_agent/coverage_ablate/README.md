# Coverage Reward 与 PRS 消融实验 (Qwen3.8 B300)

本目录提供在 B300 上训练 Qwen3.8 时使用的覆盖率辅助奖励（Coverage Reward）与概率拒绝采样（PRS, Probabilistic Rejection Sampling）的实验配置。**其中 `effrefcov`（效率折扣参考时间覆盖率）为最终方案**，其余配置用于基线对比与消融分析。

## 方案对照

- **最终方案定义**：`effrefcov` 将参考时间覆盖率乘以效率折扣因子（仅在候选实现的端到端耗时长于参考实现时进行衰减：`discount = min(1.0, reference / candidate)`），而 PRS 保持使用原始未折扣的 `reference_time_coverage`。
- **关键机制与事实**：
  - 表中辅助奖励权重仅作用于 coverage 或 speed 辅助项；权重为 0 时仅关闭辅助奖励，不影响基础奖励及有效性过滤。
  - 所有启用 PRS 的方案统一使用阈值 `0.3` 与线性过渡宽度（factor）`0.1`。
  - **重计算层数历史差异**：基线 `qwen38_b300_baseline_t1` 当前默认值为 24（历史已完成的实际训练使用了显式覆盖 `RECOMPUTE_NUM_LAYERS=32`，早期 8 层曾发生 OOM）；历史 `refcov` 脚本默认仍为 8；其余方案默认均为 32。24 层尚未经过完整训练验证；受这些配置差异影响，对照时还需核对实际启动参数。

| 启动脚本 (Launcher) | 方案定位 | 辅助奖励类型 (Auxiliary reward type) | 辅助权重 | PRS Coverage Key | 重计算层数 |
| --- | --- | --- | --- | --- | --- |
| `qwen38_b300_baseline_t1_effrefcov.sh` | **最终方案** | `efficiency_reference_time_coverage` | 0.5 | `reference_time_coverage` | 32 |
| `qwen38_b300_baseline_t1.sh` | 基线 | `time_coverage` | 0.5 | `time_coverage` | 24 (历史运行 32) |
| `qwen38_b300_baseline_t1_refcov.sh` | 对照 | `reference_time_coverage` | 0.5 | `time_coverage` | 8 |
| `qwen38_b300_baseline_t1_nocov_noprs.sh` | 消融 | `efficiency_reference_time_coverage` | 0 | 禁用 (Disabled) | 32 |
| `qwen38_b300_baseline_t1_nocov_prs.sh` | 消融 | `capped_speed_auxiliary` | 0 | `reference_time_coverage` | 32 |
| `qwen38_b300_baseline_t1_refcov_prs.sh` | 消融 | `reference_time_coverage` | 0.5 | `reference_time_coverage` | 32 |
| `qwen38_b300_baseline_t1_speedaux_prs.sh` | 消融 | `capped_speed_auxiliary` | 0.5 | `reference_time_coverage` | 32 |
| `qwen38_b300_baseline_t1_timecov025_prs.sh` | 消融 | `gated_time_coverage` | 0.25 | `reference_time_coverage` | 32 |

## 运行环境与启动命令

本组实验依赖预配置的 B300 双容器环境（训练与 Rollout 容器隔离，分别分配 `slime_actor` 与 `slime_rollout` 资源，共享配置见 `examples/kernel_agent/qwen38_b300_env.sh`，诊断工具见 `scripts/b300/README.md`）。在仓库根目录下确认环境与资源后，最终方案的完整启动命令如下：

```bash
bash examples/kernel_agent/coverage_ablate/qwen38_b300_baseline_t1_effrefcov.sh
```

如需执行其他基线或消融对比实验，替换为对应的脚本路径即可。各方案的输出路径、Ray submission ID 与 TensorBoard 记录均保持相互独立。

## 评测入口

本地仓库已版本化追踪各方案对应的 GEPA-V2 评测脚本与校验工具。评测套件入口位于 `examples/kernel_agent/eval/`（例如最终方案对应的 `examples/kernel_agent/eval/qwen38_b300_baseline_t1_effrefcov_gepav2_all.sh`），产物校验与导出工具见 `scripts/qwen38_*_eval_artifacts.py`。训练脚本本身不自动触发评测流程。
