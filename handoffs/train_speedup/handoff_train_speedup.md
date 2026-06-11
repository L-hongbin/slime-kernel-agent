# 训练加速 — 总览

> **阅读指南**
> 本文是 slime 训练侧加速各方向的**导航文档**。每个方向只列核心结论和关键数据，详细实验记录请点击各小节的"详见"链接。
> Rollout/推理侧加速见 [../rollout_speedup/handoff_rollout_speedup.md](../rollout_speedup/handoff_rollout_speedup.md)。

---

## 背景

- **模型**：Qwen3.6-27B（**dense**，64 层，hidden 5120，ffn 17408，词表 248320）+ GDN 线性注意力 + MTP head（1 层）。无 MoE。
- **并行**：TP4 × PP2（手动 34/30 切分，`--decoder-last-pipeline-num-layers 30/31`）× CP2 × DP2 = 16 GPU；colocate + `--optimizer-cpu-offload`。
- **硬件**：4 节点 H20-96GB；TP4+CP2 在单节点（NVLink），PP 跨节点（IB）。
- **算法**：RLOO + TIS，EAGLE 投机解码；bf16 权重 + `--accumulate-allreduce-grads-in-fp32`。
- **权威 run**：[`checkpoints/Qwen3.6-27B/20260609_134303.t1.27B.bf16.TP4.PP2.CP2.tis.eagle.colocate.offload.ctx16384.gradf32.H20/run.log`](https://wandb.ai/shuailin_chen/slime/runs/ttl1ctxe?nw=nwuser1106982578)（41 个完整 train step，约 11h43m）。
- **脚本**：`scripts/train_drkernel/t1.27b.bf16.tis.tp4.cp2.pp2.eagle.colocate.offload.gradf32.sh`

### 两个一直成立的前提

<!-- 后文反复用到，先讲清楚 -->

1. **Megatron 性能开关一部分自动生成**：除显式 `add_argument` 外，`ArgumentGroupFactory(TrainingConfig)` 和 `core_transformer_config_from_args` 会从 dataclass 字段自动生成 flag（如 `--manual-gc`），grep 不到字面 `add_argument` 但合法。⚠️ 自动生成的名字可能反直觉（FP8 是 `--fp8-format` 不是 `--fp8`）。
2. **slime 用自己的训练循环**（`slime/ray/train_actor.py` + `backends/megatron_utils/actor.py`），不走 Megatron 的 `training.py` pretrain loop——只在 Megatron 主循环里生效的开关在 slime 里需逐一核对接线（`--manual-gc-interval`、`--log-memory-interval` 等都失效）。

---

## 单步 wall time 花在哪？

<!-- 理解加速方向之前，先理解 step_time 的组成。数据来自上面的 bf16 权威 run。 -->

`perf/step_time = perf/train_wait_time + perf/train_time`。稳态 step median **949s（约 15.8 分钟）**。

| 阶段 | 稳态 median | wall 占比 | 能否 overlap |
| --- | ---: | :---: | --- |
| **actor train** | **498s（8.3 min）** | **最大单项** | — |
| **rollout**（含 pending 长尾，计入 `train_wait`） | **390s（6.5 min）** | **大** | async 模式可与 train overlap |
| **checkpoint save**（每 10 step 周期性） | **~384s** | **周期性尾巴** | 前台 save 可与下一步 train overlap |
| offload + update_weights + wake_up + data_preprocess | ~22s | **2.3%（小）** | — |

<!-- **核心结论：训练 wall 由 actor train + rollout 主导**，offload/权重同步是配角（22s）。因此 async 模式的收益主要来自 rollout 和 train 的 overlap，而非 offload/weightsync 开销的减少。checkpoint save 是新暴露的周期性尾巴，已单独成章。 -->

> save step（10/20/30/40）的 `step_time` median/mean 升到 **1315s / 1385s**，非 save step 为 **943s / 953s**。

---

## 加速方向总览

| 方向 | 状态 | 关键结论 |
| --- | --- | --- |
| 单步效率瓶颈分析 | ✅ 已定位 | 瓶颈 = actor train（8.3min）+ rollout（6.5min）；offload/weightsync 仅占 2.3% |
| Megatron 加速开关 | ⏳ 部分落地 | 已开 overlap-param-gather/async-save；**PP 气泡（VPP/layout）最有潜力**；TP comm overlap 与 varlen 不兼容搁置 |
| Checkpoint save 效率 | ✅ 已优化 | `dp_reshardable` 前台 save 7.6s vs `fully_reshardable` 139.5s（9B A/B，18.3×）；27B 脚本已切默认 dp |

---

## 1：单步效率瓶颈分析

**详见**：[handoff_train_step_efficiency.md](./handoff_train_step_efficiency.md)

### 做了什么

基于 bf16 权威 run 的 41 个稳态 step，拆解 `step_time = train_wait_time + train_time`，按 save / non-save step 分组量化各阶段耗时，并核对 rollout 长尾、router 健康度、recompute 配置。

### 结论

| 指标 | 非 save step median / mean | save step median / mean |
| --- | ---: | ---: |
| step_time | 943s / 953s | 1315s / 1385s |
| actor_train_time | **498s / 500s（最大单项）** | 498s / 499s |
| rollout_time | 390s / 421s | 410s / 465s |
| save_model_time | 0 | **385s / 384s** |

- **瓶颈是 actor train 和 rollout 两个大块**；offload + update_weights + wake_up + data_preprocess 合计仅 ~22s（2.3%）。
- **async 模式的收益主要来自 rollout 和 train 的 overlap**，不是 offload/weightsync 开销的减少。
- actor_train_tflops 稳态 ~30.5，偏低；`recompute full block 25 层` 偏保守，**减重计算前需先量 actor train 峰值显存**（Megatron `--log-memory-interval` 在 slime 不触发，需外部 `nvidia-smi -l 1` 对齐窗口）。
- Router 固定用 `round_robin`：历史 FP8 A/B 显示 `consistent_hashing` 会放大 backlog/502/disconnect/drain（DrKernel 是大批单轮长生成，亲和路由反而有害）。

---

## 2：Megatron 训练加速开关

**详见**：[handoff_megatron_train_accel.md](./handoff_megatron_train_accel.md)

### 做了什么

系统盘点 Megatron 训练侧的性能开关：哪些已开、哪些值得加、哪些已搁置，以及各自在 slime 自有训练循环里**是否真生效**。

### 结论

| 开关 | 状态 | 建议 |
| --- | --- | --- |
| `--overlap-param-gather` | ✅ 已开 | 保持；smoke 校验数值正确性 |
| `--async-save` / `--save-hf` | ✅ 已加 | 保持；先验证 FP8 HF 导出可加载 |
| **PP 气泡（VPP / layout / defer-wgrad）** | ⏳ 待评估 | **最有潜力**，与 varlen 兼容 |
| 重计算放松（recompute） | ✅ 已完成 | 已量峰值显存并放松 recompute |
| `--manual-gc` | ⏳ 可选 | 低风险但 RL 体制收益**边际**，实测再决定 |
| FP8 训练 / 训练期 CUDA graph | 🧪 实验 | 高风险，仅长期实验 |
| `--tp-comm-overlap` | ❌ 搁置 | 与 slime **always-varlen 根本性不兼容** |
| cross-entropy fusion / MoE overlap | ❌ 不适用 | RL 路径不走 / 模型 dense |

### 关键发现

- **PP 气泡是最有潜力的结构性优化**：PP=2 + 末段背负大词表输出层 + 交叉熵 + MTP head 是气泡来源。优先 `--num-virtual-stages-per-pipeline-rank`（避开 decoder first/last 断言）或 `--pipeline-model-parallel-layout`（替代写死的 34/30）。
- **`--tp-comm-overlap` 搁置**：`_initialize_tp_communicators()` 用预分配固定形状 buffer，而 slime 强制 packed varlen/THD（每 micro-batch 变长）→ 形状对不上报错或静默回退。硬上要改回定长 BSHD+padding、关动态批，等于放弃 packed 优势，不划算。
- **`--manual-gc` 收益边际**：它是稠密预训练（每秒上千紧同步 iter）的优化；slime RL step 是分钟级，单次 GC 暂停被 27B GEMM 掩盖，能抠回的 jitter 大概率边际。

---

## 3：Checkpoint Save 效率

**详见**：[handoff_checkpoint_save_efficiency.md](./handoff_checkpoint_save_efficiency.md)

### 做了什么

拆解 save step 多出的 ~384s 尾巴，定位耗时来源，并用 9B 单节点最小 A/B 隔离 checkpoint 路径，落地到 27B 脚本。

### 结论

历史 27B run 的 save 拆解（`save_model_time ≈ 384s`，全部计入 save step 的 `train_wait`）：

| save 阶段 | 耗时 | 占比 |
| --- | ---: | ---: |
| torch_dist save（模型+优化器） | ~285s | ~74% |
| HF 导出（91 shard） | ~90s | ~23% |
| wrapper（wake_up/sleep 等） | ~9s | ~2-3% |

9B TP4×CP2×PP1×DP1 A/B（首次前台 `save_model`）：

| Case | optim ckpt 格式 | persistent worker | 首次前台 save | 结论 |
| --- | --- | :---: | ---: | --- |
| `fr1_pw0` | `fully_reshardable` | 否 | 186.9s | 缺 worker 时 `--async-save` 被禁用 |
| `fr1_pw1` | `fully_reshardable` | 是 | 139.5s | worker 可异步写盘，但 fully 前台仍慢 |
| `fr0_pw1` | `dp_reshardable` | 是 | **7.6s** | 换默认 dp 后前台 save 降 **18.3×** |

### 关键发现

- **`--use-persistent-ckpt-worker` 是 `--async-save` 生效的必要条件**：无 worker 时 Megatron 打印 `Disabling --async-save`。
- **`fully_reshardable` 是前台 save 慢的主因**：保存时 gather/transform optimizer buffer，前台成本高；`dp_reshardable`（默认）各 rank 并行保存已有 shard。只有明确需要改并行拓扑 resume 时才付 fully 的成本。
- **27B 脚本已切回默认 `dp_reshardable`**，删除 `--dist-ckpt-optim-fully-reshardable` 及其 mem-efficient flag；新增 `LOAD_DIR`/`CKPT_STEP` 支持从旧 fully ckpt resume 后转存成新 dp ckpt（load 看 checkpoint 自带 metadata，save 看当前命令行 flag）。
- 当前脚本**保持 `--save-hf` 注释**，避免每次 save 额外 ~90s HF 导出。

---

## 总结论与下一步

### 有效杠杆（按潜力排序）

1. **PP 气泡优化**（§2，最有潜力）：VPP / `--pipeline-model-parallel-layout` 收缩 PP=2 末段不均的结构性气泡，与 varlen 兼容，待量收益。
2. **重计算放松**（§2）：先量 actor train 峰值显存，有余量则减 recompute / 增大 token batch 直接提 actor train 吞吐。
3. **Checkpoint save**（§3，已落地）：切 `dp_reshardable` 后前台 save 7.6s，save 尾巴基本消除。
4. **rollout/train overlap**（§1）：async 模式让 rollout 与 train 重叠，是 rollout 这 6.5min 的主要回收路径。

### 无效 / 搁置杠杆

- **offload/weightsync 优化**：仅占 2.3%，即使完全优化掉收益也很小。
- **`--tp-comm-overlap`**：与 slime always-varlen 不兼容，搁置。
- **`--manual-gc`**：RL 分钟级 step 下收益边际。

### 判优原则

> **⚠️ 量峰值显存别信日志边界值**：Megatron `--log-memory-interval` 在 slime 循环不触发，`Memory-Usage before/after offload/wake_up` 是边界瞬时值不是训练峰值。测 actor train 峰值需外部 `nvidia-smi -l 1` 对齐 `actor_train` 窗口，或给训练循环加 `torch.cuda.max_memory_allocated()` 日志。
