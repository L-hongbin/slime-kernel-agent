# Checkpoint Save 效率与 offload/TMS 保存安全

这份文档回答训练同学实际需要判断的两个问题：

1. checkpoint save 应该怎么配，才能避免周期性 save 拖慢训练。
2. offload/TMS 训练下，怎样保证保存出来的 model tensor 是可靠的。

## 一句话结论

- 常规同拓扑续训使用默认 `dp_reshardable`，不要为普通训练 checkpoint 打开 `fully_reshardable`；后者的前台保存成本高一个数量级。
- 继续保留 `--async-save`。非 final save 只需要完成前台准备并 schedule 写盘，真正写盘可以和下一步训练重叠。
- 保存异常的风险点在 offload actor 生命周期：模型在 train/save/update_weights 之间反复 `wake_up()` / `sleep()`；如果状态不准，TMS pause 前的 CUDA allocation 状态可能不一致，model tensor 保存记录就可能错误。
- 当前修复不是强制同步保存，而是让 offload actor 的 `sleep()` / `wake_up()` 幂等，并在 `destroy_process_groups()` 后、`torch_memory_saver.pause()` 前清 CUDA cache。

<!--
已确认有一份 offload/TMS/async-save 配置下保存出的 torch_dist checkpoint 包含坏 model tensor，不能作为正常 resume 或评测输入。

`--use-persistent-ckpt-worker` 是本次排查中的一个容易误导的细节：
当前 slime 的自定义初始化没有调用 Megatron 的 `init_persistent_async_worker()`，所以这个 flag 在当前路径里主要用于通过 Megatron 参数校验，避免 `--async-save` 被关掉；它不是这组 save time 差异的主因。
-->

## 保存耗时

9B TP4 x CP2 x PP1 x DP1 的 `train_only` 保存实测：

| Case | optim ckpt 格式 | worker flag | async 实际状态 | iter0 前台 `save_model` | final `save_model` | 结论 |
|---|---|---:|---:|---:|---:|---|
| `fr1_pw0` | `fully_reshardable` | 否 | 被禁用 | 186.9s | 179.0s | 缺 worker flag 时 `--async-save` 不生效 |
| `fr1_pw1` | `fully_reshardable` | 是 | 生效 | 139.5s | 201.0s | async 可重叠写盘，但 fully 前台路径仍很慢 |
| `fr0_pw1` | `dp_reshardable` | 是 | 生效 | **7.6s** | 124.1s | 换默认 dp 后首次前台 save 降 18.3x |
| `ckptfix_train_only_tis` | `dp_reshardable` | 是 | 生效 | **8.1s** | 129.9s | 当前 actor sleep/wake fix 验证；保存成功，tensor 直读通过 |

看训练吞吐时主要看非 final 的前台 `save_model`。final `save_model` 会等待未完成的 async 写盘 finalize，所以通常明显更长。

## Async save 怎么看

`--async-save` 生效后，普通训练 step 里的 `save_model` 不等价于完整写盘时间。它主要包含前台准备、构造 `state_dict`、把写盘任务排进 async 队列；真正写盘可以和下一步训练重叠。

| 项 | 前台 `save_model` | final `save_model` |
|---|---|---|
| 发生位置 | 普通训练 step 中的 checkpoint save | 作业结束前最后一次 checkpoint save |
| async 生效时包含什么 | 前台准备、state_dict 生成、schedule async 写盘 | 前台准备 + 等待所有未完成 async 写盘 finalize |
| 是否能和训练重叠 | 能 | 不能 |
| 用途 | 判断 checkpoint 对训练吞吐的影响 | 判断作业结束前还要等多久 |
| 当前 fix 9B 实测 | iter0 前台 `save_model=8.1s` | final `save_model=129.9s` |

因此评估训练吞吐时看非 final save；评估退出/收尾耗时时再看 final save。

## Checkpoint 格式怎么选

| 项 | `dp_reshardable` | `fully_reshardable` |
|---|---|---|
| 开启方式 | 默认；不加 `--dist-ckpt-optim-fully-reshardable` | 加 `--dist-ckpt-optim-fully-reshardable` |
| 保存格式 | 贴近 DistributedOptimizer 内部 bucket/shard | 转成更通用的 per-param canonical optimizer state |
| save 行为 | 各 rank 并行保存已有 shard | 保存时 gather/transform optimizer buffer，前台成本高 |
| 瞬时显存/内存 | 较低，无大规模 gather 中间态 | 较高；可能产生 GPU/CPU 中间 buffer |
| resume 能力 | 适合同 TP/PP 拓扑续训，主要支持 DP 维度 reshard | 更适合跨 TP/PP/EP/DP 拓扑 resume |
| 本次实测 | dp+worker 首次前台 save 7.6s | fully+worker 首次前台 save 139.5s |
| 适用建议 | 常规同拓扑训练 checkpoint 默认用它 | 只有明确需要改并行拓扑 resume 时才付这个成本 |

`--distrib-optim-fully-reshardable-mem-efficient` 只在 fully 路径下有效。它可以降低 fully save/load 的 host/device memory 压力，但不会把 fully 的前台 gather/transform 变成 dp 的快路径。

## 保存异常的错误链

这类错误的关键不是“哪个评测读错了”，而是 checkpoint 的 model tensor 记录本身已经不可信。

<!--
已确认的异常 run:
checkpoints/Qwen3.6-27B/20260610_142800.t1.27B.bf16.TP4.PP2.CP2.tis.eagle.colocate.offload.ctx16384.gradf32.H20
-->

触发组合：

```text
offload_train=true
async_save=true
torch_dist checkpoint
torch_memory_saver LD_PRELOAD enabled
训练/保存/update_weights 之间反复 wake_up() / sleep()
```

TMS 指 `torch_memory_saver`。它通过 preload hook 管理 PyTorch CUDA tensor 的底层显存分配，在 `offload_train` 下负责 pause/resume 和 CPU backup。

错误链：

```text
offload_train 下模型参数由 TMS 管理
-> 训练前 wake_up()，训练后 sleep()
-> save_model() 也需要先 wake_up() 再取 state_dict，保存后再 sleep()
-> async save 会把写盘和下一步训练重叠，下一步训练又会 wake_up()
-> 如果 sleep/wake 不是幂等的，或者 process group teardown 后没有清 CUDA cache 就 TMS pause
-> TMS 看到的 CUDA allocation 状态可能和 actor 期望状态不一致
-> model tensor 保存记录可能错误
```

这次坏 checkpoint 的关键信号：

```text
iter_0000049:
  866 个 tensor
  46 个 tensor rel_l2 >= 0.1
  41 个 tensor 全零
  无 missing key / shape mismatch / nonfinite

iter_0000059:
  缺 .metadata 和 metadata.json
  不作为可靠 checkpoint 使用
```

同一个 checkpoint 里，model BF16 记录坏，但 optimizer/master fp32 记录仍接近 base。这说明错误发生在 model checkpoint 保存内容上，不是后续读取或评测造成。

<!--
代表性坏 tensor，保留给需要复核原始证据的人：

decoder.layers.29.self_attention.linear_attn.in_proj_a.weight
  nonzero 0 / 245760

decoder.layers.2.self_attention.input_layernorm.weight
  nonzero 0 / 5120

decoder.layers.14.self_attention.linear_attn.conv1d.weight
  nonzero 0 / 40960

decoder.layers.32.self_attention.linear_attn.conv1d.weight

model BF16 record:
  nonzero 16384 / 40960
  relL2 vs base 0.9009426236152649

optimizer/master fp32 record:
  nonzero 40960 / 40960
  relL2 vs base 0.0001776839781086892
-->

## 修复是什么

修复文件：

```text
slime/backends/megatron_utils/actor.py
```

修复逻辑：

- actor 初始化后维护 `_offload_sleeping`。
- `sleep()` 如果已经 sleeping，直接返回。
- `wake_up()` 如果已经 active，直接返回。
- `sleep()` 在 `destroy_process_groups()` 后、`torch_memory_saver.pause()` 前执行 `clear_memory()`。
- `save_model()` 仍保持原来的 offload 生命周期：保存前 `wake_up()`，保存后 `sleep()`。
- 不强制同步保存；继续保留 `--async-save`。

<!--
上游关联：

- [THUDM/slime #1888](https://github.com/THUDM/slime/pull/1888)：保存前后显式 wake/sleep。
- [THUDM/slime #1895](https://github.com/THUDM/slime/issues/1895)：TMS pause/free 与重复 sleep 相关问题。
-->

## 正确性检查

当前修复已经用 9B `train_only` 跑过同类 offload/TMS/async-save 路径：`iter_0000000` 和 `iter_0000001` 都成功落盘。直读新 checkpoint 的 6 个 model tensor，结果全部 finite、全非零，`rel_l2_vs_base` 在 0 到 `1.039e-5` 范围内；没有复现全零或异常大偏离。

<!--
9B 验证 run 信息：

- Ray job: `raysubmit_QXEjvbHzDqAbfiKn`
- Run dir: `checkpoints/Qwen3.5-9B/20260611_063141.debug.t1.9B.bf16.TP4.PP1.CP2.tis.eagle.colocate.offload.ctx8192.gradf32.H20.ckptfix.train_only.tis/`

抽查 tensor：

验证方式：走 `train_only`，用固定 debug rollout dump 跳过真实 rollout 服务，只验证训练、offload、TMS、async save、finalize 和 torch_dist 读回。

关键配置：`debug_train_only=True`, `offload_train=True`, `use_tis=True`, `async_save=True`, `use_persistent_ckpt_worker=True`, `dist_ckpt_optim_fully_reshardable=False`, `dynamic_sampling_filter_path=None`

保存结果：`iter_0000000` 和 `iter_0000001` 都成功落盘，latest iteration 为 1。

Tensor 直读校验：从 `iter_0000001` 直读 6 个 model tensor，全部 finite、全非零；`rel_l2_vs_base` 范围 0 到 `1.039e-5`。

decoder.layers.0.self_attention.input_layernorm.weight
  nonzero 4096 / 4096, rel_l2_vs_base 7.154e-07

decoder.layers.2.self_attention.input_layernorm.weight
  nonzero 4096 / 4096, rel_l2_vs_base 0

decoder.layers.0.self_attention.linear_attn.conv1d.weight
  nonzero 32768 / 32768, rel_l2_vs_base 9.765e-06

decoder.layers.5.self_attention.linear_attn.conv1d.weight
  nonzero 32768 / 32768, rel_l2_vs_base 1.039e-05

decoder.layers.0.self_attention.linear_attn.in_proj_a.weight
  nonzero 131072 / 131072, rel_l2_vs_base 6.327e-07

decoder.layers.5.self_attention.linear_attn.in_proj_a.weight
  nonzero 131072 / 131072, rel_l2_vs_base 1.953e-06
-->
