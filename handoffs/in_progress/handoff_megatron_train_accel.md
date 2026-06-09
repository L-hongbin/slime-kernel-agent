# Megatron 训练加速配置分析

> 本文 = **Megatron 训练侧加速开关清单**：哪些已开、哪些值得加、哪些已搁置，及各自在 slime 里是否真生效。
> 与 [handoff_train_step_efficiency.md](./handoff_train_step_efficiency.md)（讲"时间花在哪"）互补——本篇讲"哪些旋钮能拧"。
> Rollout/推理侧见 [../rollout_speedup/handoff_rollout_speedup.md](../rollout_speedup/handoff_rollout_speedup.md)。

## TL;DR

| 开关 | 状态 | 收益 | 建议 |
|---|---|---|---|
| `--overlap-param-gather` | ✅ 已开 | 中 | 保持；smoke 校验数值正确性即可 |
| `--async-save` / `--save-hf` | ✅ 本轮加 | 中（长跑） | 保持；先验证 FP8 HF 导出可加载 |
| PP 气泡（VPP / layout / defer-wgrad） | ⏳ 待评估 | **中高** | **最有潜力**，与 varlen 兼容，见 §3.A |
| 重计算放松（recompute） | ⏳ 待评估 | 中高 | 先量峰值显存，有余量就减重计算，见 §3.B |
| `--manual-gc` | ⏳ 可选 | **边际** | 低风险但 RL 体制收益小，**实测再决定**，见 §3.C |
| FP8 训练 / 训练期 CUDA graph | 🧪 实验 | 高/未知 | 高风险，仅长期实验，见 §3.D |
| `--tp-comm-overlap` | ❌ 搁置 | — | 与 slime varlen **不兼容**，见 §4 |
| cross-entropy fusion / MoE overlap | ❌ 不适用 | — | RL 路径不走 / dense 模型，见 §4 |

---

## 1. 背景

- **模型**：Qwen3.6-27B（**dense**，64 层，hidden 5120，ffn 17408，词表 248320）+ GDN 线性注意力 + MTP head（1 层）。无 MoE。
- **并行**：TP4 × PP2（手动 34/30 切分）× CP2 = 16 GPU = 2 节点；colocate + offload。
- **硬件**：H20-96GB；TP4+CP2 在单节点（NVLink），PP 跨 2 节点（IB）。
- **脚本**：`scripts/train_drkernel/debug.t1.27b.fp8.tp4.cp2.pp2.eagle.colocate.offload.sh`
- **两个一直成立的前提**（后文反复用到）：
  1. Megatron 性能开关一部分是显式 `add_argument`，一部分由 `ArgumentGroupFactory(TrainingConfig)`（`Megatron-LM/megatron/training/arguments.py:2033`）和 `core_transformer_config_from_args`（`:1329-1345`）从 dataclass 字段**自动生成**——很多 flag（`--manual-gc` 等）grep 不到字面 `add_argument` 但合法。⚠️ 注意自动生成的名字可能与直觉不符（如 FP8 是 `--fp8-format` 不是 `--fp8`）。
  2. **slime 用自己的训练循环**（`slime/ray/train_actor.py` + `backends/megatron_utils/actor.py`），不走 Megatron 的 `training.py` pretrain loop——只在 Megatron 主循环里生效的开关，在 slime 里需逐一核对接线。

---

## 2. 当前基线（已启用）

| 类别 | 开关 |
|---|---|
| 并行 | `--tensor-model-parallel-size 4` `--sequence-parallel` `--pipeline-model-parallel-size 2`（`--decoder-last-pipeline-num-layers 30`）`--context-parallel-size 2` |
| 重计算 | `--recompute-granularity selective` |
| 动态批 | `--use-dynamic-batch-size` `--max-tokens-per-gpu 8192` `--log-probs-max-tokens-per-gpu 16384` |
| 分布式优化器 | `--use-distributed-optimizer` `--overlap-grad-reduce` `--overlap-param-gather` |
| 优化器精度/卸载 | `--use-precision-aware-optimizer` `--optimizer-cpu-offload` `--overlap-cpu-optimizer-d2h-h2d` |
| 梯度通信 | `--grad-reduce-in-bf16` |
| 注意力 / log-prob | `--attention-backend flash` · `--log-probs-chunk-size 10000` |
| checkpoint | `--async-save` + `--save-hf`（见 §3.E） |

> **算子融合默认就开**（rope / swiglu / bias-dropout / masked-softmax / grad-accum-fusion / persist-layernorm），除非 `--no-*`，已白拿。

`--overlap-param-gather`（脚本 `:202`，slime 接线 `model.py:613-615`、`:677-680`）：分布式优化器参数 all-gather 与前向重叠，配 `--overlap-grad-reduce`。与 `--optimizer-cpu-offload` 同用一般兼容，建议 smoke 校验数值（参数 CPU H2D 回来再 gather 的时序）。

---

## 3. 值得评估的加速项

### A. PP 气泡优化（最有潜力，全部与 varlen 兼容）

PP=2 + 末段背负大词表输出层 + 交叉熵 + MTP head，是当前结构性气泡来源。下列开关都不与 packed varlen 冲突：

| 开关 | 作用 | 注意 |
|---|---|---|
| `--num-virtual-stages-per-pipeline-rank` | VPP，交错调度收缩气泡 | **优先**：不均匀 PP 下灵活，**不触发** decoder first/last 断言（`arguments.py:528-552`） |
| `--num-layers-per-virtual-pipeline-stage` | VPP 旧式入口 | 会与手动 `--decoder-last-pipeline-num-layers 30` 不均匀切分打架，需重算层分配 |
| `--pipeline-model-parallel-layout` | 显式排布各 stage 的 embedding/loss/transformer/MTP 层 | 直接对症末段不均（`arguments.py:2255-2262`），可替代写死的 34/30 |
| `--defer-embedding-wgrad-compute` | embedding wgrad 推迟进气泡 | 适用已有的 `--untie-embeddings-and-output-weights`；低风险小收益 |
| `--overlap-p2p-communication-warmup-flush` | PP P2P warmup/flush 重叠 | 与 varlen P2P 兼容（`p2p_communication.py:301-307`），但**仅 VPP 开启 P2P overlap 后才有意义**（`model_parallel_config.py:342-348`、`:458-462`） |
| `--account-for-embedding-in-pipeline-split` / `--account-for-loss-in-pipeline-split` | 自动把 embedding/loss 算进 PP 负载 | 非加速，是手动 34/30 的稳健替代 |

### B. 重计算放松（用显存换速度）

当前 `selective`；优化器已 offload 到 CPU、腾出 GPU 显存。若峰值显存有余量，**减少重计算**（甚至关）可直接提速，或 `--recompute-modules` 精细指定。
⚠️ 量峰值显存有坑：Megatron `--log-memory-interval` 在 slime 循环**不触发**（见 step-efficiency handoff 瓶颈 2），需外部 `nvidia-smi -l 1` 看 `actor_train` 窗口，或给 train 循环加 `torch.cuda.max_memory_allocated()` 日志。

### C. `--manual-gc`（低风险，但 RL 体制下收益边际，需实测）

- **原理**：`gc.disable()` 停掉**循环垃圾回收器**的自动阈值触发（**不影响引用计数**，后者照常即时释放），改为在对齐时点手动 `gc.collect()`，消除某 rank 偶发 GC 暂停在跨节点同步点放大成的气泡。
- **slime 接线**：开关时 `train()` 内**每步**都 `gc.disable()+gc.collect()`（`model.py:750-755`，函数 `:661`）。`--manual-gc-interval` 在 slime **无效**（slime 不跑 Megatron 主循环里那段周期回收，`training.py:2321-2323`）。回收不会漏：colocate+offload 下每步另有 wake（`actor.py:202`）+ sleep（`:180`）两次 `clear_memory()`→`gc.collect()`，合计三次。
- **为何收益边际**：manual-gc 是**稠密预训练**（每秒上千紧同步 iter，单 rank GC 暂停在每次同步处拖住全体）的优化。slime RL step 是**分钟级**，单次 GC 暂停（几~几十 ms）相对 27B GEMM 很小且被算力掩盖；同步远不如预训练频密 → 能抠回的 jitter 大概率边际。
- **为何 slime 默认 `False`（≠没用）**：① 收益边际属调优旋钮；② 框架默认偏鲁棒——默认开自动 GC 则始终有兜底，不依赖"手动 collect 点 cadence 够好"这一假设（唯 async 非 offload 模式外层缺额外 `clear_memory()`）；③ 与上游一致（`training_config.py:70` 即 `manual_gc=False`）。
- **结论**：本脚本（colocate+offload）下加它**低风险、不漏收**，但**吞吐收益预期小**——smoke 实测开关前后 `actor_train` 时间均值/方差/p99 再决定是否常开，别想当然算净赚。

### D. 实验性 / 高风险（仅长期实验）

- **FP8 训练** `--fp8-format hybrid`（注意名字；+ `--fp8-recipe`、`--fp8-param-gather`，见 `transformer_config.py:437-439`）：`--hf-checkpoint` 本是 FP8、H20 支持 FP8 GEMM，可提速 ~1.3–1.5×，但 RL 稳定性 + GDN 后端兼容性未知。
- **训练期 CUDA graph**（`--rl-training-cuda-graphs` / `--external-cuda-graph`）：省 kernel launch，但 GDN + 动态批脆（rollout 侧已踩过对齐坑）。

### E. checkpoint（本轮已落地，附说明）

- `--async-save`：dist-ckpt（torch_dist）D2H 暂存后**后台写 NFS**，与训练重叠；仅 torch_dist 受益，`--save-hf` 仍同步。最后一步 `force_sync=True` 阻塞刷完（`actor.py:576-584`）。⚠️ 本 smoke 只在末步存且 force_sync→等价同步，收益要在有中间 save-interval 的长跑里才体现。
- `--save-hf ${SAVE_DIR}/hf/iter_{rollout_id}`：默认 `--megatron-to-hf-mode raw`→`save_hf_model_direct`（`hf_checkpoint_saver.py:21`），量化配置继承自 `--hf-checkpoint`（FP8）。⚠️ **不是单文件/纯 rank0 聚合**：权重重建用 **PP broadcast + TP all-gather 跨 rank**（`hf_weight_iterator_direct.py:67-78`、`:103-106`；`common.py:59-66`、`:94-103`），rank0 落盘，产物是**分片 safetensors + index**（`hf_checkpoint_saver.py:127-149`）。
- **保存触发**：`should_run_periodic_action` 在末步（`rollout_id==num_rollout-1`）**无条件 True**（`misc.py:92-93`），故 `--save-interval 10` > `NUM_ROLLOUT=3` 时末步仍存一次。
- ⚠️ FP8 + raw HF 导出此前未在这套 27B FP8 配置实跑过，正式长跑前先 smoke 验证 `iter_2` 能生成并被加载。

---

## 4. 已搁置 / 不适用

### `--tp-comm-overlap` — 与 slime varlen 根本性不兼容，搁置

- **前置全满足**：TE 2.10.0（`te>=2.7` 分支，bootstrap 默认 nccl）；`--sequence-parallel` 已开（Megatron 断言要求）；`CUDA_DEVICE_MAX_CONNECTIONS=1` 已设（Megatron 断言要求）；TP4 走 NVLink；slime 已接线（`initialize.py:95`）。
- **决定性阻塞**：`_initialize_tp_communicators()`（`initialize.py:242`）用**预分配固定形状** `input_shape=[(seq_length×mbs)/CP, hidden]` 调 `initialize_ub`，要求进 GEMM 的张量匹配该 buffer。而 slime 强制 **packed varlen/THD**：`--qkv-format` 默认 `thd`（`slime/utils/arguments.py:109-113`）+ 无条件 `variable_seq_lengths=True`（`backends/megatron_utils/arguments.py:77-78`），每 micro-batch 变长；且 `seq_length` 默认 4096（`arguments.py:154`）与 `CTX_LEN=16384` 不符。
- **结果**：形状对不上 → 首个 forward 报错，或 TE 静默回退到非重叠（buffer 白占显存、零收益）。
- **硬上代价大、不划算**：要改 slime 数据路径为定长 BSHD+padding、关动态批、备 `--tp-comm-overlap-cfg`，等于放弃 packed 训练优势。

### 误区（别浪费时间）

- **`--cross-entropy-loss-fusion` / `--cross-entropy-fusion-impl`**：RL 路径不走 Megatron 的 LM 交叉熵；slime 策略损失用自己的 `fused_vocab_parallel_cross_entropy`（`slime/utils/ppo_utils.py:153`）+ `--log-probs-chunk-size`，大词表已处理。
- **MoE 类 overlap**：模型 dense，不适用。

---

## 5. 待办

1. [ ] **PP 气泡（§3.A，最高潜力）**：评估 `--num-virtual-stages-per-pipeline-rank`（优先，避开 decoder 断言）或 `--pipeline-model-parallel-layout` 替代写死的 34/30，量气泡收益。
2. [ ] **核训练峰值显存 → 放松 recompute（§3.B）**：与 step-efficiency handoff 瓶颈 2 同一动作。
3. [ ] **`--manual-gc`（§3.C，可选）**：smoke 实测 `actor_train` 时间分布再决定是否常开。
4. [ ] FP8 / 训练期 CUDA graph（§3.D）：仅长期实验，先验稳定性。
5. [x] `--overlap-param-gather` 已开；`--async-save`/`--save-hf` 本轮落地；TP comm overlap 搁置（§4）。

## 关联文档

- [handoff_train_step_efficiency.md](./handoff_train_step_efficiency.md) —— 单步耗时分解 / rollout 长尾 / recompute / dynamo
- [handoff_drkernel_slime_plan.md](./handoff_drkernel_slime_plan.md) —— DrKernel-on-slime 计划/状态 hub
- [../rollout_speedup/handoff_rollout_speedup.md](../rollout_speedup/handoff_rollout_speedup.md) —— rollout/推理加速总入口
