# 训练 RL step 效率瓶颈分析

> **阅读指南**
> 本文分析 **colocate 模式下单个 RL step 的端到端效率**（rollout 生成 + 训练前反向），
> 区分 **结构性瓶颈（放大规模仍存在）** 与 **smoke 产物（故意调小的配置）**。
> Rollout/推理侧的加速方向详见
> [../rollout_speedup/handoff_rollout_speedup.md](../rollout_speedup/handoff_rollout_speedup.md)。

---

## 背景

- **模型**：Qwen3.6-27B（dense, 64 层, hidden 5120）+ MTP head（`--enable-mtp-training`, 1 层）
- **硬件**：8× H20-96GB（单机），colocate
- **并行**：TP4 × CP2 × PP1，**DP=1**；rollout = 2× TP4 engine
- **推理**：SGLang + EAGLE（draft=4, steps=3, topk=1）
- **数据来源 run**：`checkpoints/Qwen3.6-27B/20260607_035845_t1.TP4.CP2.eagle.bf16.H20_ctx16384/run.log`
- **启动脚本**：`scripts/train_drkernel/debug.t1.27b.bf16.tp4.cp2.eagle.sh`
- ⚠️ **这是 smoke 测试**：`num_rollout=3`、`rollout_batch_size=4`、`n_samples_per_prompt=8`
  （gbs=32）、`CTX_LEN=16384`。部分配置故意调小，下文已标注。

---

## 单步时间分解（rollout 0）

RL step = `train_wait` + `train`，colocate 下两者**完全串行**
（`offload_train=True` / `offload_rollout=True`，生成时 8 张训练卡 idle，训练时推理引擎已 offload）。

| 阶段 | 耗时 | 占比 | 来源 |
| --- | ---: | ---: | --- |
| update_weights（权重推到 SGLang） | 4.0s | 0.6% | `timer.py` update_weights |
| **rollout 生成** | **281.7s** | **44%** | `perf/rollout_time` |
| sleep/wake_up/preprocess（offload 切换） | ~2.1s | 0.3% | `timer.py` |
| **actor_train（前反向, 32 微批）** | **353.8s** | **55%** | `timer.py` actor_train |
| **合计 / step** | **~645s ≈ 10.75 min** | 100% | rollout:train ≈ 45:55 |

> 两半各占一半且串行 —— 见瓶颈 4。

---

## 瓶颈排序（含证据与可调项）

### 🔴 1. Rollout 长尾：>60% 的生成时间在排空 1–7 条 straggler（结构性）

SGLang decode 日志：
- **满批阶段** `#running-req: 29`，吞吐 2500–2725 tok/s/engine（双引擎 ~5000+），仅持续约前 90s。
- **长尾阶段** `04:08:05 → 04:10:59`（约 **174s，占生成 63%**），`#running-req` 一路 7→2→1，
  吞吐崩到 **230–510 tok/s**，GPU 近乎空转。

佐证指标（`perf 0`）：
- `truncated_ratio = 0.25`（25% 样本打满 16383 被截断 —— 即 straggler）
- `longest_sample_tokens_per_sec = 53.4` vs `tokens_per_gpu_per_sec = 179.8`（最长样本每 token 慢 3.4×）

**根因**：同 prompt 内 8 个 sample 长度方差极大，少数超长（被截断）样本拖住整组 batch。
**这是最高 ROI 的优化点。**
可调：partial rollout、组内可判定后提前 abort 剩余样本、over-sampling 比例与 dynamic-filter 联动、
动态/更短 `max_response_len`、长度惩罚。

> ✅ **现成杠杆（首选试这个）= partial rollout（`--partial-rollout`）**，上游 slime 已原生实现（即 APRIL
> [RLsys-Foundation/APRIL](https://github.com/RLsys-Foundation/APRIL) 的开关，非 fork）。**本 run `partial_rollout=False`
> （args dump line 664），当前是关的。**
>
> **原理（slime 源码）**：over-sampling 调度收够 `rollout_batch_size` 个有效 group 后，`abort()`
> 给所有 SGLang worker 发 `/abort_request{abort_all}` 杀掉在飞的长尾请求（`sglang_rollout.py:379-420`）。
> - 关：被 abort 的半成品**直接丢弃**（`:405 if not args.partial_rollout: continue`），已生成的几千 token 算力白烧。
> - 开：半成品收进 `aborted_samples` → `data_source.add_samples()`（`:657-659`）塞回 buffer，下一步从
>   `sample.tokens`（prompt+已生成）**断点续写**（`:238`），并打 `start_rollout_id` 标记。净效果：长尾样本切成几段
>   跨 step 接力完成，算力都留住 → 敢更激进 over-provision + 提早 abort 削墙钟。
> - ⚠️ **off-policy 取舍**：续写那段是旧权重生成的。默认全当 on-policy（接受 staleness）；加
>   `--mask-offpolicy-in-partial-rollout`（`:274-275`）把旧段 `loss_mask=0`、只对新生成 token 计 loss（严格 on-policy 但丢旧段梯度）。
> - 它**不是单独把长尾变快**：本步要的 group 仍需各自 8 个样本全跑完（`:474`）；价值是让"提早 abort + 多 over-provision"不浪费。
> - 官方报告 rollout 吞吐 +17–35%、收敛快 15–20%、精度 +2–5%。⚠️ 3 步 smoke 看不出（跨 step 接力），要多步真实训练里量。

### 🔴 2. 训练 full activation recompute —— 纯开销 ~+33% compute（结构性）

脚本：`--recompute-granularity full --recompute-method uniform --recompute-num-layers 1`
→ **每层全量重算**，反向额外重跑一遍前向。

粗算训练 MFU **~20–23%**（≈448k tokens / 354s / 8×H20，27B dense，含 recompute 的 ~8N FLOP/token；
H20 BF16 峰值 ~148 TFLOPS）。

为何开着：**`DP=1` 时 `--use-distributed-optimizer` 不分片**（ZeRO 需 DP>1），优化器状态全副本压每卡，
内存紧 → 用 full recompute 换显存。
**待验证**：先量训练峰值显存；若有余量改 **selective recompute / 只重算部分层**，可直接砍训练时间一大块。
> ⚠️ 量峰值显存的开关有坑：Megatron 的 `--log-memory-interval` / `--log-device-memory-used`（会打 `max allocated`）
> 只在 Megatron 自己的 `training_log` 里触发，而 **slime 用自己的训练循环、不走那条路，这两个开关在 slime 里不生效**。
> slime 自带的 `print_memory`（`slime/utils/memory_utils.py`）只在 offload/wake_up/update_weights 等空闲边界打印、
> 且只报瞬时 `allocated/reserved`、不报峰值。可行手段：① 训练阶段外部 `nvidia-smi -l 1` 看 `memory.used`（注意 colocate
> 时序，要在 `actor_train` 窗口看）；② 给 `actor.py:train_actor` 反向后加一行 `torch.cuda.max_memory_allocated()`
> 峰值日志（最准，需小改代码 + sanity check）。
> **上游核对（2026-06-07）**：拉 `THUDM/slime@main` 的 `memory_utils.py` 与 `megatron_utils/actor.py` 比对，
> 上游 `available_memory()` 同样只报瞬时值、训练循环内同样无内存日志 —— **上游也没有此能力，无可拉取**，要峰值得自己加（手段 ②）。

### 🟠 3. 微批太"薄" + torch.dynamo 反复重编译退回 eager（结构性）

- `--max-tokens-per-gpu 9216`，样本均长 ~14k token、CP=2 后每 rank ~7k → **基本一条样本一个微批 = 32 微批**，
  GEMM 的 M 维很小，GPU 利用率低。
- 日志多次 `torch._dynamo hit config.recompile_limit (8)`（`bias_dropout_add_fused_train`），
  动态 shape 触发反复重编译、超限后**退回 eager**，融合 kernel 失效。
- 体现在微批曲线：前 8 个微批 24.9→18.5→14.4→…→10s 的 warmup 尾巴，**每个 step 重交一次学费**。

可调：提高/分桶 dynamo recompile limit、对动态 shape 做 padding/bucketing、评估提高 `max-tokens-per-gpu`
（需先解决瓶颈 2 的显存）。

### 🟠 4. Colocate 串行 —— 两半互不重叠（结构性，框架固有）

生成 282s 内训练卡全 idle；训练 354s 内推理引擎 offload。当前 `rollout_batch_size=4`（smoke）下 rollout 本身很小，
**长尾**把这半段拉得不成比例。

### 🟡 5. 启动一次性开销 ~6 min（smoke 占比大，长跑可摊薄）

`Capture cuda graph end. Time elapsed: 184–199 s`（主图）+ draft 图 + 模型加载，启动到首步 ~6 min。
对 3 步 smoke 占比巨大。smoke 想快可缩小 `--sglang-cuda-graph-max-bs` 捕获的 bs 列表。

### 🟡 6. 其它观察（待核）

- `rollout/spec_accept_rate = 0.0` 但 `spec_accept_length = 3.10`、decode accept len 2.6–3.5
  → **EAGLE 实际有效**（~3× draft 接受），`spec_accept_rate` 字段疑似指标 bug，需单独核对。
- `prefix_cache_hit_rate = 0.318`、`avg_cached_tokens_per_sample = 428`，而 prompt 长 ~4400–5100、
  同 prompt 8 sample 本应高度共享前缀 → 命中率偏低，`consistent_hashing` 路由 + 前缀缓存可能没吃满。

---

## 建议优先级（待办）

1. [ ] **治 rollout 长尾**（瓶颈 1）—— 投入产出最高，砍 generation 的 ~60% 尾部空转。
2. [ ] **核训练峰值显存（外部 nvidia-smi 或给 train 循环加峰值日志；`--log-memory-interval` 在 slime 不生效），放松 full recompute**（瓶颈 2）。
3. [ ] **缓解 dynamo 退回 + 评估提高 `max-tokens-per-gpu`**（瓶颈 3）。
4. [ ] 核对 `spec_accept_rate=0.0` 是否指标 bug；排查 prefix 命中率偏低（瓶颈 6）。

## 可生成的证据（按需）

- 同 prompt 8 样本逐条响应长度 / 截断分布 dump → 量化长尾。
- 量训练峰值显存（外部 `nvidia-smi -l 1` 看 `actor_train` 窗口，或给 `train_actor` 加 `max_memory_allocated` 日志）
  → 判断能否降 recompute。（注：`--log-memory-interval` 在 slime 训练循环不触发，见瓶颈 2 的 ⚠️。）
- 按 `CLAUDE.md` 流程派 codex（xhigh）复核本结论。

---

## slime 内置加速手段速查（doc 调研 2026-06-07）

> 来源：`docs/zh/advanced/{speculative-decoding,pd-disaggregation,sglang-config,delta-weight-sync,megatron-config}.md`、
> `docs/zh/get_started/{usage,qa}.md` + `slime/utils/arguments.py`。按"对哪个瓶颈有用"归类，标注本 run 现状。

### 训练侧（瓶颈 2/3）

| 手段 | 开关 | 本 run 现状 | 说明 / 取舍 |
|---|---|---|---|
| 降重计算 | `--recompute-granularity selective` | full（全重算） | usage.md：selective 少重算一些；显存够就换，直接省训练时间。先量峰值显存（见瓶颈 2 ⚠️）。 |
| 增大训练微批 | `--max-tokens-per-gpu` ↑ | 9216（≈ `max_resp/cp` 保守地板） | qa.md：先按 `max_response_len/cp_size` 防 OOM，稳定后调大提效。需配合降重计算腾显存。 |
| log-prob 前向用更大批 | `--log-probs-max-tokens-per-gpu` ↑ | 9216（=训练批） | log-prob 前向无 backward、显存更省，可设得比训练批大 → ref/old-actor 两趟前向的微批更少更快。 |
| 跳过 old log-prob 前向 | `--use-rollout-logprobs`（+`--use-tis`） | 关 | 直接用 sglang 生成时的 logprob 当 IS 比，省掉一整趟 Megatron 前向。⚠️ sglang↔megatron logprob 有数值差，需 TIS 做 off-policy 校正。进阶项，先验证。 |

### Rollout 侧（瓶颈 1/4/6）

| 手段 | 开关 | 本 run 现状 | 说明 / 取舍 |
|---|---|---|---|
| partial rollout | `--partial-rollout` | 关 | 见瓶颈 1：回收被 abort 的长尾半成品续跑（首选）。 |
| 投机采样 + 在线 MTP | `--sglang-speculative-algorithm EAGLE` + `--enable-mtp-training` | **已开** | speculative-decoding.md：RL 中在线训 MTP，draft 随 policy 更新、接受率不衰减。本 run accept_len~3.1 有效，保持。 |
| PD 分离 | `--prefill-num-servers` / `--sglang-config` | 关 | decode 主导、长 context、多轮时收益大；本 run 单轮单机 8 卡 colocate 收益有限，优先级低。 |
| 低精度 rollout | W8A8 / FP8 | 关 | 提 decode 吞吐。详见 [低精度 handoff](../rollout_speedup/handoff_low_precision.md) / [W8A8 综述](../rollout_speedup/handoff_w8a8_speedup_survey.md)。 |
| 路由 prefix 命中 | `consistent_hashing`（已用）/ `--router-balance-abs-threshold` | hit 0.32 | 见瓶颈 6。强制均衡（threshold 0）会伤 prefix 命中，需权衡。 |

### 不适用 / 已排除

- **Delta 权重同步**（`--update-weight-mode delta`）：doc 明确 **colocate 下被参数校验拒绝**（CUDA IPC 只传 ~64B 句柄，
  delta 的 wire 节省为零、簿记纯亏）。本 run update_weights 仅 4s，无需。
- **`--megatron-config-path`**：目前只支持 PPO 的 actor/critic 角色覆盖，GRPO/rloo 不适用。

---

## 关联文档

- [../rollout_speedup/handoff_rollout_speedup.md](../rollout_speedup/handoff_rollout_speedup.md) —— rollout/推理加速总入口
- [../rollout_speedup/handoff_rollout_device_efficiency.md](../rollout_speedup/handoff_rollout_device_efficiency.md) —— 推理设备利用率
- [handoff_drkernel_slime_plan.md](./handoff_drkernel_slime_plan.md) —— DrKernel-on-slime 计划/状态 hub
