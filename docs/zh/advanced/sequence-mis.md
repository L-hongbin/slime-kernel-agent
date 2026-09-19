# Sample 级 Sequence MIS

## 训练 forward 的 Bernoulli KL 筛选

Megatron policy loss 支持直接使用本次 actor 训练 forward 的 logprob，对每个训练 sample 进行筛选：

```bash
--sequence-mis-config '{"aggregation":"binary_kl"}'
```

默认不启用。启用 `binary_kl` 后，`upper` 默认为 `0.05`，可显式覆盖为有限、非负数，例如
`--sequence-mis-config '{"aggregation":"binary_kl","upper":0.02}'`。
默认值用于与 DPPO Binary-TV `0.2` 叠加时的初始尝试，尚未经过真实训练调优。
不需要 `--rollout-data-postprocess-path`、`--keep-old-actor` 或额外的 actor forward。
旧脚本即使保留 `examples.kernel_agent.kernel_filter.sequence_mis` 后处理入口，该入口在此模式下也会跳过，避免重复筛选。

计算方法参考 [FlashREINFORCE](https://github.com/yifanzhang-pro/FlashREINFORCE/blob/master/flashreinforce/loss.py)：

```text
p = exp(实际 rollout_logprob)
q = exp(当前 actor 训练 forward 的 logprob)
d = p * log(p/q) + (1-p) * log((1-p)/(1-q))
sample_kl = sum(d * 原始 loss_mask) / sum(原始 loss_mask)
sample_kl > upper → 当前 sample 的 loss_mask 全部置零
```

- 概率以 FP32 计算并 clamp 到 `[1e-6, 1-1e-6]`；gate 不参与反向传播。
- 只统计有效生成 token，排除上下文、工具反馈和 padding；CP 下先汇总完整 sample 的 logprob。
- 多轮训练拆分后，每个 turn 是独立 sample，分别筛选。**不是论文的整条多轮轨迹统一筛选。**
- 按共享 loss mask 语义，拒绝 sample 同时屏蔽主 policy loss、entropy 和 reference KL 项。
- 原始 token / rollout / prompt 分母保持不变；不删除 sample，不重算 reward 或 advantage。
- mask 只更新当前 loss 调用中的 batch 副本，不污染原始训练数据；checkpoint 重算及后续训练 forward 会重新判断。
- 缺少实际 rollout logprob 时明确报错。此模式仅支持 `ratio_source="rollout"`，不支持旧模式的 `lower`、`delta`、token veto 或 advantage protection。
- 此功能只增加 rejection gate，不自动改变 IS 权重、PPO clipping、advantage estimator 或训练更新次数，不等同于完整 FlashREINFORCE。

## 指标

开启后，随现有训练指标输出：

- `seq_mis_binary_kl`：筛选前的 sample 平均 Bernoulli KL。
- `seq_mis_masked_fraction`：将拒绝标记广播到该 sample token 后，用筛选前的 mask 聚合。

指标沿用现有训练 reducer：PerToken 按原始 token 数归一化；非 PerToken 按 rollout 归一化，轨迹内按有效 token 数加权。
因此，多轮场景中的 `seq_mis_masked_fraction` 不是简单的“被拒 turn 数 / turn 数”。
独立的 entropy common-probe 诊断保留其原始固定 mask，不受该筛选影响。

## 与旧 actor 侧 seq-MIS 的区别

旧 `kl/geometric/mirrorpop/turns_*` 模式继续使用训练前后处理入口及原有行为。
`binary_kl` 在训练 loss 内执行，始终比较当前训练 forward 和实际 rollout 的概率，不使用预先重算的旧 actor 概率冒充当前策略。
