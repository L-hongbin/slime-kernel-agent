# Checkpoint Save 效率分析

本文记录周期性 checkpoint save 的耗时拆解、9B 单节点 A/B 实测结论，以及当前 27B 训练脚本的落地改动。
与 [handoff_train_step_efficiency.md](./handoff_train_step_efficiency.md) 互补：那篇讲单 RL step 时间花在哪，本篇专门解释 save step 的尾巴。

## TL;DR

| 结论 | 证据 | 行动 |
|---|---|---|
| 历史 27B run 每 10 step 的 save 阻塞主循环约 384s | `save_model_time≈382-386s`，全部计入 save step 的 `train_wait` | 已定位为 checkpoint 路径问题 |
| 384s 里 torch_dist 是大头，HF 导出是次大头 | torch_dist ~285s，HF ~90s，wrapper ~9s | 当前脚本已关闭 `--save-hf` |
| `--use-persistent-ckpt-worker` 是 `--async-save` 生效的必要条件 | 无 worker 时 Megatron 打印 `Disabling --async-save`；有 worker 时日志出现 `scheduled an async checkpoint save` | 当前脚本保留 worker |
| `fully_reshardable` 是前台 save 慢的主因 | 9B TP4 x CP2 x PP1 x DP1 实测：fully+worker 首次前台 save 139.5s；dp+worker 7.6s | 当前脚本切回默认 `dp_reshardable` |
| 旧 fully ckpt 可以 resume 后转存成新 dp ckpt | load 看 checkpoint 自带 metadata，save 看当前命令行 flag | 从旧 ckpt resume 时必须让 `--load` 指向旧目录，`--save` 指向新目录 |

## 9B A/B 实测

基于 `scripts/train_drkernel/debug.t1.9b.bf16.tis.tp4.cp2.pp2.eagle.colocate.offload.gradf32.sh` 做了
TP4 x CP2 x PP1 x DP1 的最小实测。为隔离 checkpoint save，使用 synthetic train-only rollout：
`NUM_ROLLOUT=2`、`SAVE_INTERVAL=1`、`ROLLOUT_BATCH_SIZE=1`、`N_SAMPLES_PER_PROMPT=2`、`GLOBAL_BATCH_SIZE=2`、
`ENABLE_WANDB=0`、`--save-hf` 关闭。

| Case | optim ckpt 格式 | persistent worker | async 实际状态 | iter0 前台 `save_model` | final `save_model` | 结论 |
|---|---|---:|---:|---:|---:|---|
| `fr1_pw0` | `fully_reshardable` | 否 | 被禁用 | 186.9s | 179.0s | 缺 worker 时 `--async-save` 不生效 |
| `fr1_pw1` | `fully_reshardable` | 是 | 生效 | 139.5s | 201.0s | worker 可异步写盘，但 fully 前台路径仍很慢 |
| `fr0_pw1` | `dp_reshardable` | 是 | 生效 | **7.6s** | 124.1s | 换默认 dp 后首次前台 save 降 18.3x |

对应 run log 仍保留在这些轻量 run 目录中：

| Case | run log |
|---|---|
| `fr1_pw0` | `checkpoints/Qwen3.5-9B/20260610_092507.debug.t1.9B.bf16.TP4.PP1.CP2.tis.eagle.colocate.offload.ctx16384.gradf32.H20.ckptab.fr1.pw0/run.log` |
| `fr1_pw1` | `checkpoints/Qwen3.5-9B/20260610_093836.debug.t1.9B.bf16.TP4.PP1.CP2.tis.eagle.colocate.offload.ctx16384.gradf32.H20.ckptab.fr1.pw1/run.log` |
| `fr0_pw1` | `checkpoints/Qwen3.5-9B/20260610_094917.debug.t1.9B.bf16.TP4.PP1.CP2.tis.eagle.colocate.offload.ctx16384.gradf32.H20.ckptab.fr0.pw1/run.log` |

实测注意事项：

- 三个 Ray job 都在 job 内日志中显示 succeeded；Ray CLI 结尾仍可能打印 `could not determine terminal Ray job status ... unknown`。
- 大 checkpoint payload `iter_*` 已删除，run 目录只保留 `run.log`、`latest_checkpointed_iteration.txt` 和 tiny rollout state。
- 本轮是 DP1，因此不能量化 DP>1 的 gather scaling；但已经足够证明 fully 格式即使在 DP1 也有显著前台成本。

## 前台 Save vs Final Save

| 项 | 前台 `save_model` | final `save_model` |
|---|---|---|
| 发生位置 | 普通训练 step 中的 checkpoint save | 作业结束前最后一次 checkpoint save |
| async 生效时包含什么 | 前台准备、state_dict 生成、把 async 写盘任务排出去 | 前台准备 + 必须等待所有未完成 async 写盘 finalize |
| 是否能和训练重叠 | 能。写盘可以和下一步训练重叠 | 不能。已经没有下一步训练可重叠 |
| 用途 | 判断 checkpoint 对训练吞吐的影响 | 判断作业结束前还要等多久 |
| 本次 dp+worker 实测 | iter0 前台 `save_model=7.6s` | final `save_model=124.1s` |

因此评估训练吞吐时看非 final 的前台 `save_model`；评估退出/收尾耗时时再看 final `save_model`。

## `dp_reshardable` vs `fully_reshardable`

| 项 | `dp_reshardable` | `fully_reshardable` |
|---|---|---|
| 开启方式 | 默认；不加 `--dist-ckpt-optim-fully-reshardable` | 加 `--dist-ckpt-optim-fully-reshardable` |
| 保存格式 | 贴近 DistributedOptimizer 内部 bucket/shard | 转成更通用的 per-param canonical optimizer state |
| save 行为 | 各 rank 并行保存已有 shard | 保存时 gather/transform optimizer buffer，前台成本高 |
| 瞬时显存/内存 | 较低，无大规模 gather 中间态 | 较高；可能产生 GPU/CPU 中间 buffer |
| resume 能力 | 适合同 TP/PP 拓扑续训，主要支持 DP 维度 reshard | 更适合跨 TP/PP/EP/DP 拓扑 resume |
| 本次实测 | dp+worker 首次前台 save 7.6s | fully+worker 首次前台 save 139.5s |
| 适用建议 | 常规同拓扑训练 checkpoint 默认用它 | 只有明确需要改并行拓扑 resume 时才付这个成本 |

`--distrib-optim-fully-reshardable-mem-efficient` 只在 fully 路径下有效。它用 Gloo/DP rank0 方式降低 fully save/load
的 host/device memory 压力，但不会把 fully 的前台 gather/transform 变成 dp 的快路径；这次 fully 实测已经开了该 flag，
前台 save 仍是 139.5s。切回 `dp_reshardable` 后，这个 flag 应一起删除。

## 历史 27B Run 拆解

- Run: `checkpoints/Qwen3.6-27B/20260609_134303.t1.27B.bf16.TP4.PP2.CP2.tis.eagle.colocate.offload.ctx16384.gradf32.H20/run.log`
- 拓扑: 4 节点 H20，TP4 x PP2 x CP2 x DP2
- 配置: colocate、`--optimizer-cpu-offload`、`--save-interval 10`、`--save-hf` 开启、`fully_reshardable` 开启、
  `use_persistent_ckpt_worker=False`
- save 发生在 iter 9/19/29/39，`perf/save_model_time≈382-386s`

| save 阶段 | iter 9 | iter 19 | iter 29 | iter 39 | 占比 |
|---|---:|---:|---:|---:|---:|
| torch_dist save（模型+优化器） | 284.8s | 283.6s | 286.6s | 285.8s | ~74% |
| HF 导出（91 shard） | 92s | 90s | 89s | 91s | ~23% |
| wrapper（wake_up/sleep 等） | ~8s | ~9s | ~9s | ~9s | ~2-3% |
| 合计（`save_model_time`） | 385.1s | 382.3s | 384.3s | 385.8s | 100% |

torch_dist 18GB/节点 ÷ 285s 约 63 MB/s，明显不像纯本地盘写入瓶颈；结合 9B A/B，主因应是 fully optimizer
checkpoint 的前台 gather/transform。

## 当前脚本状态

目标脚本：`scripts/train_drkernel/t1.27b.bf16.tis.tp4.cp2.pp2.eagle.colocate.offload.gradf32.sh`

| 项 | 当前状态 | 说明 |
|---|---|---|
| `--dist-ckpt-optim-fully-reshardable` | 已删除 | 后续新保存默认写成 `dp_reshardable` |
| `--distrib-optim-fully-reshardable-mem-efficient` | 已删除 | 只对 fully 有效，dp 路径不需要 |
| `--async-save` | 保留 | 配合 persistent worker 后生效 |
| `--use-persistent-ckpt-worker` | 保留 | `--async-save` 必需 |
| `--save-hf` | 保持注释 | 避免每次 save 额外 ~90s HF 导出 |
| `LOAD_DIR` | 新增可选覆盖 | 从旧 ckpt resume 时用 `LOAD_DIR=/path/to/old/run`，新 ckpt 仍写到新的 `SAVE_DIR` |
| `CKPT_STEP` | 新增可选覆盖 | 父目录没有 `latest_checkpointed_iteration.txt` 时，用 `CKPT_STEP=39` 指定加载 `iter_0000039` |

从旧 fully 中间 ckpt 转到新 dp ckpt 的推荐方式：

```bash
LOAD_DIR=/path/to/old/fully/run \
CKPT_STEP=39 \
bash scripts/train_drkernel/t1.27b.bf16.tis.tp4.cp2.pp2.eagle.colocate.offload.gradf32.sh
```

期望日志顺序：

```text
sharded_state_dict metadata loaded from the checkpoint: {'distrib_optim_sharding_type': 'fully_reshardable', ...}
Loading distributed optimizer sharded state of type fully_reshardable
...
Storing distributed optimizer sharded state of type dp_reshardable
```

如果后续要从新 dp ckpt 继续训，保持同 TP/PP 拓扑；不要指望新 dp ckpt 支持改 TP/PP 后 resume。

## 关联文档

- [handoff_train_step_efficiency.md](./handoff_train_step_efficiency.md) - 单 RL step 时间分解
- [handoff_megatron_train_accel.md](./handoff_megatron_train_accel.md) - Megatron 训练侧加速开关总表
