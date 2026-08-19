# DeepSeek-V4 LoRA training-side contract

本文定义训练侧 LoRA 的两组独立机制：behavior-policy anchor 的 adapter-only old actor，以及 adapter 参数化与优化器分组的 rsLoRA、LoRA+。正式启用值以 launcher 为准；serving 映射与热切换见 `handoffs/deepseek-v4/lora_serve_design.md`，模型切分与当前 target set 见 `handoffs/deepseek-v4/dsv4_megatron_sharding_contract.md`

## 结论

- `--keep-old-actor` 在 LoRA 训练中只保存 adapter，不实例化第二份完整模型
- old-actor snapshot 只支持 `--update-weights-interval 1`，每个样本必须携带唯一且匹配的 `gen_weight_version`
- `--use-tis` 与 adapter old actor 配合时衡量 Megatron 与 SGLang 的同版本差异；它与 `--use-rollout-logprobs` 互斥
- sequence MIS 的 `ratio_source=old_actor` 比较同一 Megatron stack 的 current 与 old policy，但与 routing replay 互斥
- rsLoRA 把 forward scale 从 `alpha/r` 改为 `alpha/sqrt(r)`，checkpoint 和 serving 必须保留同一有效 scale
- LoRA+ 只提高 B 矩阵的学习率，并用独立 optimizer group 保证 save/resume 身份不混淆

## Behavior-policy anchor

### 分母选择

| 配置 | policy ratio 的 behavior anchor | 约束 |
|---|---|---|
| 不使用 rollout logprobs，且不开 old actor | 当前 actor 的 Megatron recompute，ratio 可能退化为 1 | 不能代表真实一步策略漂移 |
| `--keep-old-actor` | 生成该 batch 的旧 adapter 在 Megatron 上 recompute | LoRA 路径使用 adapter-only snapshot |
| `--use-rollout-logprobs` | SGLang 返回的 behavior-policy logprob | 与 TIS 互斥；通常不再执行 old-actor recompute |
| `--get-mismatch-metrics` | loss anchor 不变，额外执行 Megatron recompute | 仅用于观测 cross-engine mismatch |

`should_recompute_old_actor_log_probs` 只在 loss 或 mismatch 指标需要 Megatron old-policy 时触发 forward。直接使用 rollout logprobs 的 DPPO、DIS 等模式已有 behavior anchor，不应再叠加 TIS

### Adapter-only snapshot 协议

1. 枚举所有 `requires_grad` 且名称为 `linear_in.weight` 或 `linear_out.weight` 的 adapter parameter
2. 在 CPU pinned memory 保存一份 behavioral snapshot，并分配一份 swap scratch
3. 首次训练前用已发布的 adapter 初始化 snapshot 与 version tag
4. scoring 前把 live adapter 写入 scratch，再把 snapshot 原位 copy 到 live parameter
5. scoring 完成后在 `finally` 中 bit-exact 恢复 live adapter
6. gradient step 前用当前 live adapter 刷新 snapshot，供下一批 lag-1 rollout 使用

原位 copy 不替换 `Parameter` 对象，因此 optimizer state 的 key 保持不变。snapshot 内容与发布版本只有在每步都 update weights 时一致，初始化阶段会硬拒绝 `update_weights_interval != 1`

`resolve_batch_gen_version` 对本 rank 的空列表、缺失值、混合版本或版本不符全部 fail closed，并通过 Gloo all-reduce 让所有 DP rank 一起失败，避免单 rank 在后续 collective forward 中挂死。没有版本标记的 debug rollout 必须关闭 old actor，或使用受 CLI 门禁限制的 fixed-replay 诊断模式

### TIS 与 sequence MIS

TIS 对 token policy loss 乘以下列截断权重

```text
tis = exp(megatron_old_logprob - sglang_rollout_logprob)
weight = clamp(tis, tis_clip_low, tis_clip)
```

配合 adapter old actor 时，两项指向同一 behavior version，TIS 只描述 train/rollout stack 差异。没有 old actor 时，Megatron recompute 可能包含策略漂移，不能解释为纯 cross-engine mismatch

sequence MIS 支持两个 ratio source

| `ratio_source` | 比较对象 | 适用边界 |
|---|---|---|
| `rollout` | Megatron old-policy 与 SGLang rollout-policy | 默认 cross-engine 检查，可与 routing replay 共用 scoring forward |
| `old_actor` | Megatron current-policy 与 Megatron old-policy | 纯策略漂移，需要额外 current forward，禁止 routing replay |

`ratio_source=old_actor` 必须同时启用 old actor。它的额外 forward 会重复消费 routing replay buffer，因此参数校验在启动前直接拒绝该组合

## rsLoRA 与 LoRA+

### 参数化

| 模式 | adapter forward | optimizer LR |
|---|---|---|
| classic LoRA | `x + alpha/r * B(A(x))` | A、B 使用同一 group LR |
| rsLoRA | `x + alpha/sqrt(r) * B(A(x))` | A、B 默认仍使用同一 group LR |
| LoRA+ | scaling 由 classic 或 rsLoRA 决定 | B group 使用 `lambda * LR` |

`apply_v4_lora` 完成 Megatron Bridge wrap 后，rsLoRA 统一改写每个 `LinearAdapter.scale`。attention、compressor、shared expert 的 trainer forward，以及 full-merge 与 adapter-only export，都读取这一个 live scale

SGLang 仍按 `lora_alpha/r` 计算，因此 export 写入 `lora_alpha = live_scale * r`，并在生成 config 时验证 roundtrip 完全一致。离线导出 rsLoRA checkpoint 必须向 `scripts/dsv4/tools/export_lora_adapter_hf.py` 传 `--rslora`

adapter checkpoint 写入 `v4_lora_scaling.json`，记录 rank、alpha、rsLoRA mode 与有效 scale。resume 时 marker 与重建模型不一致会失败；旧 checkpoint 缺少 marker 时只告警，不能据此证明 scaling 匹配

### LoRA+ optimizer group

`--lora-plus-lambda` 未设置或等于 1 时关闭 LoRA+。启用后，所有 trainable `*.linear_out.weight` 进入唯一 B group

- `max_lr` 与 `min_lr` 乘以 lambda，使 scheduler 全程保持 B/A LR 比例
- `lr_mult=lambda` 只作为 Megatron optimizer-state group identity，不参与 scheduler 计算
- 初始化后校验 B group 的 tensor shape multiset 与模型中的 LoRA B parameter 完全相等
- 初始化后校验有效 `eta_B == lambda * eta_A`

Megatron 的 optimizer-state group identity 不包含 `max_lr`，因此缺少独立 `lr_mult` 会让 A、B group 在保存或恢复时碰撞。改变 lambda、开启或关闭 LoRA+ 后继续加载旧 optimizer state 会因 group 不匹配而失败；需要新 optimizer state 时显式关闭 optimizer resume

### 组合约束

- rsLoRA 改变所有 adapter delta 的有效倍率，不能在同一 adapter lineage 的 resume 中切换
- LoRA+ 只改变 B group 的 optimizer step，不改变 serving config
- rsLoRA 与 LoRA+ 可以组合，但 forward scale 与 B-group LR 的效果会相乘，学习率需要作为新 recipe 独立验证
- old-actor snapshot 只 copy tensor data，与 classic 或 rsLoRA scale、LoRA+ group 划分正交
- 改变 rank 会同时改变 adapter shape、rsLoRA scale 与 serving pool 上限，必须作为新 lineage 处理

## Fail-closed 门禁

| 条件 | 结果 |
|---|---|
| LoRA 相关参数设置但 `lora_dim <= 0` | 参数校验失败 |
| old actor 的 `update_weights_interval != 1` | actor 初始化失败 |
| old-actor batch 缺失或混合 `gen_weight_version` | 所有 DP rank 一起失败 |
| `use_rollout_logprobs` 与 TIS 同时开启 | 参数校验失败 |
| `ratio_source=old_actor` 未开启 old actor | 参数校验失败 |
| `ratio_source=old_actor` 与 routing replay 同时开启 | 参数校验失败 |
| rsLoRA scaling marker 与模型不一致 | adapter resume 失败 |
| LoRA+ 没有精确匹配的 B group 或 LR 比例错误 | optimizer 初始化失败 |

## 验证与代码入口

```bash
python -m pytest \
  tests/deepseek-v4/test_dsv4_lora_old_actor.py \
  tests/deepseek-v4/test_dsv4_lora_rslora_loraplus.py \
  tests/test_lora_arguments.py -q
```

关键实现入口如下

- `slime/backends/megatron_utils/lora_old_actor.py`：snapshot、swap、version gate
- `slime/backends/megatron_utils/actor.py`：scoring 时序、refresh、resume scaling gate
- `slime/backends/megatron_utils/loss.py`：TIS 权重
- `slime/utils/arguments.py`：组合门禁
- `custom_kernels/deepseek_v4/megatron/lora.py`：target wrap 与 rsLoRA scale
- `slime/backends/megatron_utils/update_weight/lora_adapter_sync.py`：serving scale roundtrip
- `slime/backends/megatron_utils/model.py`：LoRA+ optimizer group 与启动校验
- `slime/backends/megatron_utils/adapter_ckpt.py`：scaling marker
- `scripts/dsv4/_dsv4_task_args.sh`：launcher 参数映射

GPU/full-loop 验证应覆盖 old-actor scoring 与 restore、至少两个连续 version、TIS/ratio 指标有限、LoRA+ Muon group 的实际 LR、adapter save/resume，以及 rsLoRA export 后的 trainer/serving scale 一致性
