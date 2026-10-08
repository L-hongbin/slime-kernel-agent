# Sample 级 Sequence MIS

## 公共计算入口

static 后处理和 dynamic loss 都调用 `slime.utils.sequence_mis.compute_sequence_mis`。
调用方准备完整 response 的 actor/reference logprob（CP 分片需先 gather）和原始 loss mask。
公共函数返回 `loss_masks`、每条 sample 的 `keep` 和 `scores`，可通过 `collect_stats=True` 获取静态诊断统计。
函数不修改输入 mask，不参与反向传播，也不依赖 Megatron。所有聚合方法共用分块计算；
`sequence_mis_mode=loop/batch` 仅改变分块大小，多轮模式始终保留完整轨迹分组。
旧的 `examples.kernel_agent.kernel_filter.sequence_mis` 路径继续作为 static 后处理适配入口。

## 选择 actor logprob 来源

`--sequence-mis-config` 中的 `actor_logprob` 与 `aggregation` 分开配置：

| actor_logprob | 概率来源 | 筛选时机 |
| --- | --- | --- |
| `static` | 训练前重算的 actor logprob | rollout 数据后处理时筛选一次，后续更新复用 mask |
| `dynamic` | 当前训练 forward 的 logits 计算出的 logprob | 每个 microbatch、每次 forward 重新筛选 |

为兼容旧脚本，不指定时 `binary_kl` 默认 `dynamic`，其他聚合方法默认 `static`。
`kl`、`geometric`、`mirrorpop`、`binary_kl` 都支持两种来源。
`turns_geometric` 和 `turns_mirrorpop` 只支持 `static`，因为完整轨迹可能跨 microbatch。

例如，训练前按 binary-KL 过滤：

```bash
--sequence-mis-config '{"aggregation":"binary_kl","actor_logprob":"static","upper":0.05}' \
--rollout-data-postprocess-path examples.kernel_agent.kernel_filter.sequence_mis
```

`static` 必须配置上述后处理入口（或在自定义后处理中调用它）。该模式会要求训练前重算，
即使开启 `--use-rollout-logprobs` 或当前批次本可直接复用 loss forward，也不能跳过。

使用实时 forward 做 geometric 过滤：

```bash
--sequence-mis-config '{"aggregation":"geometric","actor_logprob":"dynamic","lower":0.9,"upper":1.1}'
```

`dynamic` 当前支持 Megatron policy loss；不需要后处理入口，保留该入口也会自动跳过，避免重复过滤。
传统聚合的 `token_veto_threshold` 和 `use_advantage` 在 dynamic 模式下仍然有效。
`ratio_source` 继续决定参考概率：默认 `rollout`，传统聚合也支持 `old_actor`（仍需 `--keep-old-actor`，
并保留与 routing replay 不兼容的限制）。dynamic 使用 old_actor 时，分子来自当前 forward，
不再额外重算一次当前 actor。

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

不指定 `actor_logprob` 时保留旧行为。指定后，执行阶段由 `actor_logprob` 决定，
聚合方法由 `aggregation` 决定；static 的预计算概率不会随当前批次内的参数更新而变化。
非 binary-KL 的 dynamic 模式输出 `seq_mis_score` 和 `seq_mis_masked_fraction`；
static 模式继续通过后处理入口输出 `seq_mis/` 指标。
