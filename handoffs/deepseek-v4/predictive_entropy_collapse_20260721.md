# DeepSeek-V4 predictive mask 与 entropy 方向

本文解释 predictive DPPO mask 为什么在观测实验中系统性提高 entropy，并区分它与 DIS 的作用。Train↔rollout probability parity 的定义、测量与残差归因统一由 `handoffs/deepseek-v4/train_rollout_mismatch.md` 维护；本文不再重复 runtime 修复史或跨引擎 kernel 清单

## 结论

- predictive mask 只删除继续增大 rollout/train KL 的局部更新，不直接约束 entropy
- 被删除更新的平均 entropy 方向决定 mask 的净效应；实验中被删更新持续是降熵方向，因此剩余梯度更增熵
- 更严格的 KL threshold 会更早 mask 更多 token，观测到的 entropy 上升也更快
- 每轮 rollout anchor 随当前策略移动，local KL gate 没有把策略拉回固定 reference 的恢复力
- DIS 按 sampled-token ratio 越界与否删梯度，gate 不含 advantage 符号，因此没有固定的 entropy 方向
- 跨引擎 mismatch 会改变 ratio 与 mask，但不能单独解释 predictive mask 的方向选择

## Predictive mask 的一阶机制

在一个 response 位置上，记 rollout behavior distribution 为 `b`，trainer current distribution 为 `p`，sampled token 为 `k`，advantage 为 `A`

```text
r = p_k / b_k
w = min(r, ratio_cap)
g_k = A * w * (e_k - p)
```

Predictive DPPO 估计 `KL(b || p)` 沿更新方向的一阶变化。去掉不影响符号的正权重 `w` 后，代码中的方向项为

```text
d_k = (p_k - b_k) + sum_i p_i * (b_i - p_i)
```

当 predictive KL 已超过 threshold `delta`，且 `A*d_k > 0` 时，该位置的梯度被删除。这个条件只回答更新是否继续增大 rollout/train KL，不回答更新会让 `p` 变平还是变尖

同一更新对 entropy 的一阶方向为

```text
<grad_z H(p), g_k> = A * w * h_k
h_k = sum_i p_i^2 * (log p_i + H) - p_k * (log p_k + H)
```

令 `M_k=1` 表示该位置被 mask，则应用 mask 前后的 entropy 方向差为

```text
delta_H_kept - delta_H_all = -eta * sum_{k: M_k=1} A_k * w_k * h_k
```

因此，只要 `E[A*w*h_k | M=1] < 0`，被删除位置平均是降熵更新，mask 就会系统性提高剩余更新的 entropy 方向。这是选择偏差，不是额外加入 entropy bonus

## 直接证据

### Step-wise 一阶方向

在同一条 predictive-DPPO 训练曲线上，step 180–272 的 93 个 step 中，`entropy/first_order_mask_delta` 93 次均为正。未 mask 的一阶 entropy 方向接近零且略为负，应用 mask 后变为正

一个后续 step 的同批 logits 结果为

| 方向 | 值 |
|---|---:|
| 不应用 mask | `-1.11e-5` |
| 应用 mask | `+3.09e-6` |
| mask 单独造成的方向变化 | `+1.42e-5` |

该指标在逐位置 logit 空间精确计算一阶方向，但不等于共享参数、Muon 和完整 optimizer step 后的实际 entropy 变化

### Threshold sweep

保持初始 checkpoint、batch 规模、LR、LoRA 与 Muon recipe 不变，只改变 predictive KL threshold

| `delta` | step 0 entropy | step 3 entropy | 累计增幅 | step 0 超阈值 token | step 0 实际 mask |
|---:|---:|---:|---:|---:|---:|
| `0.05` | `0.535` | `1.214` | `+0.679` | `19.37%` | `9.99%` |
| `0.10` | `0.539` | `1.003` | `+0.464` | `7.05%` | `3.88%` |
| `0.15` | `0.540` | `0.845` | `+0.304` | `3.46%` | `1.97%` |

threshold 越严格，mask 越多，entropy 上升越快。冻结 batch 的单变量 A/B 进一步排除了在线 batch 差异：`delta=0.20` 时两次更新的 entropy 增幅为 `0.07597`，关闭 mask 时为 `0.01615`，相差 `4.70x`

关闭 mask 后 entropy 仍会上升，因此 predictive mask 是该实验的主要放大器，不是唯一来源

### 固定 prompt

相同 16 个 prompt、sampling seed 与 rollout 路径下，后期 checkpoint 相比早期 checkpoint 的 tail mass 从 `0.00705` 增至 `0.01076`，support-logit standard deviation 从 `2.909` 降至 `2.742`，sampled entropy 从 `0.811` 增至 `0.884`

生成数步后两个策略的 prefix 会分叉，因此该实验能证明后期 checkpoint 在固定 prompt 集合上产生更平的生成分布，不能替代 same-prefix full-vocabulary entropy 比较

## 为什么 local KL gate 不能阻止长期漂移

predictive mask 的 behavior anchor 是当轮 rollout policy。每次 optimizer update 后，下一轮 rollout 又来自更新后的策略，因此 anchor 随策略一起移动。即使每一步 train 都接近当轮 rollout，多个小步仍可累计远离初始策略

若目标是限制长期 entropy 或 policy drift，需要固定-reference KL、明确 entropy 约束，或具有恢复力的 smooth penalty。Hard deletion 只能移除局部方向，不能提供拉回 reference 的梯度

## 与 DIS 的区别

DIS 根据 sampled-token ratio 是否越界决定是否保留梯度

```text
r_t = p_train(x_t | h_t) / p_rollout(x_t | h_t)
M_t_DIS = 1[r_t <= 1-eps_low or r_t >= 1+eps_high]
```

DIS 删除梯度造成的 entropy 方向变化仍由被删位置的 `A_t*r_t*h_t` 平均符号决定。与 predictive mask 不同，DIS gate 不使用 advantage 符号；同一个高-ratio 位置的外推梯度和纠正梯度都会被删除

因此 DIS 既可能使剩余梯度更增熵，也可能更降熵。它同样没有固定-reference 恢复力；token 越界后，只有后续 rollout、共享参数更新或 batch 组成变化才能让它重新进入有效区间

一条旧 DIS curve 同时改变了 Muon scale、optimizer 配置、batch 长度与截断分布，没有完成只切换 DIS 的单变量 A/B，不能用来断言 DIS 的固定 entropy 方向

## 验收与边界

- 必须同时记录 unmasked、kept 与 mask-delta 的 entropy 一阶方向
- 必须按 advantage 正负、upper/lower ratio 与 KL-over-threshold 分桶
- threshold sweep 应固定 checkpoint、batch、optimizer state 与 sampling artifact
- fixed-prompt generation 只能证明行为变化，same-prefix full-vocabulary forward 才能证明分布变化
- train↔rollout mismatch 必须先通过独立 parity contract 验收，避免把坏 denominator 当成 mask 因果
- 一阶 logit-space 方向不能替代真实 optimizer-step A/B

最小后续实验是在同一冻结 batch 与 optimizer state 上比较 predictive mask、无 mask、smooth KL penalty 和 fixed-reference KL；若继续研究 DIS，则单独开关 DIS，并保持 Muon、ratio cap、batch 与 reward 完全相同

## 实现与证据入口

- `slime/backends/megatron_utils/loss.py`：predictive mask 与 entropy 一阶指标
- `tests/test_predictive_mask_gradient_entropy_audit.py`：方向公式与 mask-delta 契约
- `tests/test_predictive_mask_policy_loss.py`：policy-loss mask 语义
- `scripts/dsv4/studies/entropy/run_entropy_fixed_batch_arm.sh`：冻结 replay A/B
- `local_artifacts/deepseek-v4/r2_logs/predictive_entropy_rise_iter219_269_20260725.txt`：step-wise 与 fixed-prompt evidence
- `local_artifacts/deepseek-v4/r2_logs/dppo_threshold_fresh0_sweep_20260721.txt`：threshold sweep
- `local_artifacts/deepseek-v4/r2_logs/entropy_ab_wave1_mask_vs_nomask_20260721.txt`：冻结-batch mask A/B
- `local_artifacts/deepseek-v4/r2_logs/dis_muon_boom_bust_and_restart_20260722.txt`：DIS curve 的 confound evidence
