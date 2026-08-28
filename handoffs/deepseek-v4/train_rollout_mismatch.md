# DeepSeek-V4 train↔rollout probability parity contract

本文定义同一策略在 SGLang rollout 与 Megatron trainer 之间的概率一致性口径、已验证事实和诊断门禁。Policy loss、old actor 与 TIS 的训练语义见 `handoffs/deepseek-v4/lora_training_features.md`；predictive mask 对 entropy 的影响见 `handoffs/deepseek-v4/predictive_entropy_collapse_20260721.md`

## 结论

- train↔rollout mismatch 必须先拆成观测接口、SGLang 内部一致性、跨引擎数值路径和策略版本漂移四层
- DSPARK verifier-window 修复后的一个 H200 artifact 证明 SGLang generation 与同引擎 full-prefix scorer 可以逐 bit 一致
- 在同一 artifact 上固定 rollout expert ID 后，Megatron 与 SGLang 仍有 `0.043693` sampled-token mean absolute log-prob gap 和 `0.019976` full-vocabulary TV
- routing replay 只固定 expert ID，不固定 gate weight、router 输入、expert arithmetic、attention 或 collective
- 旧 EAGLE、bf16 KV 和已删除 FP8 alignment flag 的测量不能作为当前 FP4/DSPARK recipe 的预期 floor
- online sampled-token 指标混合了策略漂移与跨引擎数值差异；完整 paired-logit probe 才能判定纯 parity

## 四层因果分解

| 层 | 比较对象 | 主要问题 | 必须先满足的门禁 |
|---|---|---|---|
| 观测接口 | generation hook 与 API 返回值 | API 是否记录了实际采样分布 | sampled token 与 logprob 长度、token ID、精度一致 |
| SGLang 内部 | generation-time logits 与同引擎 full-prefix scorer | verifier state、prefix、kernel path 是否一致 | 完整词表逐行对齐，最好逐 bit 相同 |
| 跨引擎 | SGLang 与 Megatron 的同权重、同 prefix forward | kernel、dtype、routing、collective 是否一致 | 相同 checkpoint、adapter、tokens、expert IDs 与 runtime fingerprint |
| 策略版本 | current trainer 与 rollout behavior policy | optimizer step 或 async lag 造成的真实 policy drift | old-actor version tag 或冻结 replay |

上层没有关闭时，不得把观测到的差异归因给下层。例如 SGLang 自身 generation/scorer 不一致时，Megatron 对 stored rollout logprob 的差异不是纯跨引擎残差；current trainer 对 rollout 的差异也不是同版本数值误差

## 度量

对 rollout 返回 token `x_t` 定义

```text
delta_logp = log p_train(x_t | h_t) - log p_rollout(x_t | h_t)
delta_p    =     p_train(x_t | h_t) -     p_rollout(x_t | h_t)
ratio      = exp(delta_logp)
```

paired full-vocabulary 诊断至少报告以下统计

| 度量 | 作用 |
|---|---|
| mean `delta_logp` | sampled-token 系统方向 |
| mean absolute `delta_logp` | sampled-token 差异量级 |
| mean/absolute `delta_p` | 原始概率尺度的差异 |
| `H(rollout)-H(train)` | 分布平坦度方向 |
| total variation | 完整分布差异，不依赖 sampled token |
| top-1 ID agreement | 决策边界稳定性 |
| position/length buckets | 判断误差是否随 decode 深度增长 |

online 指标的解释必须区分来源

- `train/train_rollout_logprob_abs_diff` 比较当前 trainer forward 与 batch 中的 rollout logprob，包含真实策略更新带来的漂移
- `tis_abs = |exp(megatron_old_logprob-sglang_rollout_logprob)-1|` 只有在 old actor 与 `gen_weight_version` 匹配时才表示同版本 cross-engine mismatch
- PPO/DIS/predictive ratio 使用哪个 denominator 由 policy-loss 配置决定，不能把 objective ratio 直接当 parity metric

## 已验证事实

### SGLang 内部一致性

DSPARK target verifier 曾遗漏完整 verify window 的状态重写，使 generation-time probability 与同引擎 full-prefix scorer 不一致。`scripts/dsv4/patches/dspark_port_series/0014-*` 修复后，记录的 H200 artifact 在 384 个位置、49,643,520 个 logits 上逐 bit 相同；API sampled logprob 只剩一个浮点舍入量级的单点差异

这项证据证明该 runtime 与 artifact 的 SGLang 内部层已关闭，不证明所有未来 runtime 自动满足，也不证明 Megatron 与 SGLang 相同。每次 runtime 或 speculative patch 变化后都必须重跑内部自检

### 跨引擎 residual

在通过内部自检的 H200 artifact 上，Megatron 使用相同 checkpoint、prefix，并 replay rollout expert IDs，paired full-vocabulary 结果为

| 度量 | 值 |
|---|---:|
| sampled-token mean `delta_logp` | `+0.004859` |
| sampled-token mean absolute `delta_logp` | `0.043693` |
| sampled-token mean absolute `delta_p` | `0.010473` |
| mean `H(rollout)-H(train)` | `-0.002505` |
| mean total variation | `0.019976` |
| top-1 ID agreement | `97.40%` |

该结果证明 same-expert-ID residual 存在，但不能确定其他 checkpoint、batch、并行拓扑或 runtime 的稳定方向。旧 H0 的 signed `-0.037949` 与 absolute `0.115774` 包含当时 SGLang 内部不一致，不能与上表做 paired 前后比较

## Routing replay 边界

rollout 保存 learned-MoE 层每个 token 的 top-6 expert ID，Megatron forward 使用这些 ID 选取自己计算的 router scores

- 已固定：每层执行的 expert ID 集合
- 未固定：六个 expert 的 gate weight
- 未固定：进入 router 的 hidden state
- 未固定：W4A16 expert 内部 arithmetic 与 combine reduction
- 未固定：attention、shared expert、mHC、lm head 与 collective

因此，routing replay 能消除离散 expert-set 分叉，但不能把整个 MoE 层变成相同函数。下一层分解应先比较 IDs-only replay 与 IDs+gate-weight replay，再检查 expert 输入、单 expert 输出和 combine 前后结果

## 已排除的解法

- 将 trainer 的逐 expert BF16 `F.linear` 换成 grouped BF16 GEMM 没有改善 TV
- 强行启用当前 W4A16 路径不执行的旧 shared-expert/attention FP8 alignment flag 会新增数值路径，不能作为修复
- 历史 dense-vs-sparse indexer 实验测得影响远小于主 residual，不能解释当前差异
- bf16 KV 属于另一套容量与 kernel contract；当前 DSpark launch core 对 bf16 KV fail closed，不能借用旧 bf16-KV 数字描述正式 FP8-KV 路径

## 诊断与验收流程

1. 固定 checkpoint、adapter、tokenizer、chat template、prompt、response token、sampling 参数和 runtime fingerprint
2. 在 SGLang 同时保存 generation logits、full-prefix scorer logits、sampled token 和 routed expert IDs
3. 先要求 SGLang 内部 paired logits 逐 bit 相同；失败时停止跨引擎归因
4. 用 `train_rollout_full_vocab_probe.py` 在 Megatron 重建相同 prefix，并 replay rollout expert IDs
5. 保存完整词表 logits，报告 sampled signed/absolute gap、raw probability、entropy、TV、top-1 与位置分桶
6. 若 residual 仍存在，按 gate weight、expert arithmetic、attention/shared path、collective 的顺序做单变量分解
7. 只有同一 checkpoint、batch 与 runtime 的 paired A/B 才能量化修复效果

诊断只读 checkpoint，不得写 optimizer 或训练状态。生成 artifact 必须记录输入 SHA256、模型与 adapter identity、patch-series fingerprint、精度、并行拓扑和 expert-replay 设置

## 实现与证据入口

- `scripts/dsv4/diagnostics/mismatch/dspark_full_vocab_probe.py`：SGLang generation/scorer 内部自检
- `scripts/dsv4/diagnostics/mismatch/train_rollout_full_vocab_probe.py`：Megatron/SGLang paired full-vocabulary 比较
- `scripts/dsv4/patches/dspark_port_series/0014-*`：DSPARK verifier-window 修复
- `slime/utils/routing_replay.py`：expert-ID replay
- `slime/backends/megatron_utils/loss.py`：online sampled-token mismatch 指标
- `local_artifacts/deepseek-v4/r2_logs/train_rollout_full_vocab_h200_20260723.txt`：当前 paired H200 evidence
- `local_artifacts/deepseek-v4/r2_logs/tim_h200_dspark_root_fix_20260722.txt`：SGLang 内部 closure evidence
