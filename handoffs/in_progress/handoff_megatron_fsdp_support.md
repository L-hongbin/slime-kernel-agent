# Slime Megatron-FSDP 训练支持：可行性评估与开发清单

> [!NOTE]
> **一句话结论**：slime 当前训练后端把模型用 **Megatron DDP** 包起来（`megatron.core.distributed.DistributedDataParallel`），
> 没有走任何 FSDP 路径。换成 **Megatron-FSDP**（`--use-megatron-fsdp`）在**模型构建/训练 step 层面工作量很小**
> （Megatron-Core 0.16 已自带，`get_model` 会自动按 flag 改包 FSDP）——
> 真正的开发量集中在 **① 权重同步给 SGLang** 和 **② checkpoint 格式**，因为这两处都硬编码了 "DDP + 每卡持有完整 TP 分片" 的假设，
> 而 FSDP 会把参数在 DP 维再切一刀。本文为**评估 + 开发清单**，尚未动工。

---

## ⚠️ 先消歧：你说的 "megatron fsdp" 是哪一个？

环境里的 Megatron-Core `0.16.0rc0`（`/root/Megatron-LM`）实际上提供**两套** FSDP，外加一段 slime 上游历史，共三个候选：

| 选项 | 是什么 | flag | 关键约束 |
| :--- | :--- | :--- | :--- |
| **A. Megatron-FSDP**（本文主线，推荐） | NVIDIA 自研、与 Megatron 并行栈深度整合的 FSDP（旧名 custom FSDP）。`megatron.core.distributed.fsdp.mcore_fsdp_adapter.FullyShardedDataParallel` | `--use-megatron-fsdp`（旧 `--use-custom-fsdp` 已 deprecated） | **要求 `--ckpt-format fsdp_dtensor`**；不支持 hybrid CP；TP/CP 同用需设 `CUDA_DEVICE_MAX_CONNECTIONS=1`。**支持** PP/EP/TP 组合 |
| B. Torch-FSDP2 | Megatron 对 PyTorch 原生 FSDP2 的封装。`TorchFullyShardedDataParallel` | `--use-torch-fsdp2` | **不支持 PP、不支持 EP、不支持 MCore 分布式优化器**；要求 `torch_dist`/`torch_dcp` ckpt + `--untie-embeddings-and-output-weights`；暂不支持 fp16 |
| C. 复活上游被删的 FSDP 后端 | slime **上游曾有**独立 torch-FSDP 后端，已在 `e4faf63 Remove FSDP support (#1664)` 整体删除（本仓库继承了删除后的状态） | 旧 `--train-backend fsdp` | 是**另起一个后端**，不是包一层；上游既已删除，维护性差，不建议 |

> 「`megatron-ddp` → `megatron fsdp`」最自然的读法是 **选项 A**：训练后端不变，只把模型的 DDP 包装换成 Megatron 自己的 FSDP 包装。
> 本文按选项 A 展开；若你其实想要 B 或 C，请告知，路线会不同。

---

## 📊 现状盘点 (Dashboard)

| 维度 | 状态 | 说明 |
| :--- | :---: | :--- |
| slime 训练后端 | 仅 Megatron | `actor_group.py` 只 import `MegatronTrainRayActor`；`train_backend` 实际只认 `megatron`（`arguments.py:1527/1911`） |
| 模型包装 | **Megatron DDP** | `model.py:223` 走 Megatron `get_model`，内部按 args 包 `DistributedDataParallel`；全仓 `isinstance(..., DDP)` 假设遍布 |
| Megatron-FSDP 可用性 | ✅ 现成 | `megatron.core.distributed.fsdp` 已随 0.16 安装，`FullyShardedDataParallel` 可直接用 |
| `--use-megatron-fsdp` 能否解析 | ✅ 能 | slime 复用 Megatron 官方 parser（`arguments.py:181` `_megatron_parse_args(..., ignore_unknown_args=True)`），flag 天然可解析 |
| 权重同步 | ❌ 仅 TP-aware | `update_weight/common.py` 的 `all_gather_param` 只在 **TP group** all-gather，假设每卡已持有完整 DP 副本 |
| Checkpoint | ❌ 仅 torch_dist | `checkpoint.py` 直接转调 Megatron `save_checkpoint`；无 `fsdp_dtensor` 适配 |

---

## 🔑 决定性发现：换 FSDP「包装」很便宜，换 FSDP「数据布局」很贵

**便宜的部分（模型/训练 step）**：slime 不自己包 DDP，而是调 Megatron 的 `get_model`
（`model.py:223`，`from megatron.training.training import get_model`）。Megatron 内部会根据
`args.use_megatron_fsdp` 自动改用 `FullyShardedDataParallel` 包装、构建 FSDP 版优化器、装好
grad reduce-scatter / param all-gather hook。所以 **forward/backward/optimizer step 基本零改动**。

**贵的部分（数据布局假设）**：FSDP 把每个参数在 **DP 维**进一步分片（flat shard / DTensor），
而 slime 的两条出口管线都假设 "每张卡持有这一份参数的完整 TP 分片"：

1. **权重同步给 SGLang**（RL 每 `--update-weights-interval` 步一次）：
   `update_weight/common.py:118 named_params_and_buffers` 直接遍历 `model_module.named_parameters()`，
   再交给 `all_gather_param`（`common.py:15`）**只在 TP group** all-gather 后做 HF 转换。
   FSDP 下 `param.data` 只是该参数的一个 DP 分片 → **必须先在 FSDP/DP 维 un-shard（all-gather 或 `DTensor.full_tensor()`）还原成完整参数，再走现有 TP gather + megatron→HF**。这是最大、最易错的一块。

2. **关键的类型假设会直接崩**：Megatron-FSDP 的 `FullyShardedDataParallel(_BaseDataParallel)`
   **不是** `DistributedDataParallel` 的子类（DDP 也只是 `_BaseDataParallel` 的兄弟类）。
   于是这些 `assert isinstance(..., DDP)` / `isinstance(model[0], DDP)` 会失败或走错分支：
   `model.py:249`、`model.py:261`（forward pre-hook 开关）、`model.py:664`（overlap_grad_reduce 的 `no_sync_func` 装配）。需改成 `_BaseDataParallel` 或按能力检测。

---

## 🛠️ 开发清单（精确到文件）

| # | 改动 | 文件 | 内容 |
| :-: | :--- | :--- | :--- |
| 1 | 暴露/默认 flag | [`slime/backends/megatron_utils/arguments.py`](file:///nfs/FM/chenshuailin/projects/kernel_agents/slime/slime/backends/megatron_utils/arguments.py) | 确认 `--use-megatron-fsdp`、`--ckpt-format fsdp_dtensor`、`--fsdp-double-buffer` 等能透传；在 `validate_args` 里加 FSDP 组合校验（PP/EP/CP 限制、ckpt 格式强制） |
| 2 | 放宽 DDP 类型假设 | [`model.py:249/261/664`](file:///nfs/FM/chenshuailin/projects/kernel_agents/slime/slime/backends/megatron_utils/model.py) | `isinstance(..., DDP)` → `isinstance(..., _BaseDataParallel)`；forward pre-hook / `no_sync_func` 装配按 FSDP 能力分支（FSDP 的 hook 语义与 DDP 不同，需核对 `enable_forward_pre_hook` 是否存在/同义） |
| 3 | **权重同步 un-shard（核心）** | [`update_weight/common.py`](file:///nfs/FM/chenshuailin/projects/kernel_agents/slime/slime/backends/megatron_utils/update_weight/common.py)（`named_params_and_buffers` / `all_gather_param` / `all_gather_params_async`）+ `update_weight_from_tensor.py` / `update_weight_from_distributed{,_delta}.py` | 在现有 TP all-gather **之前**插入 FSDP un-shard：把每个 FSDP 分片在 DP/FSDP group 还原为完整张量（优先用 FSDP adapter 暴露的 unshard API / DTensor `full_tensor()`，避免手搓）。命名映射（`module.module.decoder.layers...`）需在 FSDP flatten 后仍成立 |
| 4 | Checkpoint 格式 | [`checkpoint.py`](file:///nfs/FM/chenshuailin/projects/kernel_agents/slime/slime/backends/megatron_utils/checkpoint.py)、[`model.py:save_hf_model`](file:///nfs/FM/chenshuailin/projects/kernel_agents/slime/slime/backends/megatron_utils/model.py)（~882）、`hf_checkpoint_saver.py` | Megatron-FSDP 要求 `fsdp_dtensor` 格式存取；现走 `torch_dist`。需让 save/load 与 `save_hf_model`（导出 HF）兼容 FSDP 的 sharded state dict（HF 导出同样要先 un-shard） |
| 5 | offload / memory saver | [`actor.py`](file:///nfs/FM/chenshuailin/projects/kernel_agents/slime/slime/backends/megatron_utils/actor.py)、`ray/actor_group.py:62` | `offload_train` + `torch_memory_saver` 当前按 DDP 的 param/grad buffer 设计；需验证 FSDP 的 `ParamAndGradBuffer` 在 offload/reload、`zero_grad_buffer`（`model.py:466/598`）下行为正确 |
| 6 | 启动脚本 + 环境 | `scripts/train_drkernel/*.sh` | 加 `--use-megatron-fsdp --ckpt-format fsdp_dtensor`；TP/CP 同用时设 `CUDA_DEVICE_MAX_CONNECTIONS=1`；去掉与 FSDP 冲突的 flag（如某些 `--use-distributed-optimizer` 组合） |

---

## ⚠️ 风险与待验证点

| 风险 | 描述 | 应对 |
| :--- | :--- | :--- |
| **权重同步正确性** | FSDP un-shard + TP gather + megatron→HF 三段拼接，任一段分片布局错则同步出的权重静默错误 | **强制 sanity：同步前后，训练侧与 SGLang 侧对同一 prompt 的 logprob/logits 必须一致**（对标 AGENTS.md「行为敏感改动补单测/sanity」） |
| 并行组合限制 | Megatron-FSDP 不支持 hybrid CP；TP/CP+FSDP 需 `CUDA_DEVICE_MAX_CONNECTIONS=1`；与分布式优化器、overlap_param_gather 的兼容矩阵需逐一确认 | 首版锁定 **dense + TP-only(或TP=1) + 无 PP/CP** 的最小组合打通，再逐步加 PP/EP |
| ckpt 互通 | 历史 checkpoint 是 `torch_dist`，FSDP 用 `fsdp_dtensor`，两者不直接互读 | 评估是否需要一次性转换脚本，或 FSDP 跑只从 HF base 冷启 |
| MoE/EP | 专家分片 + FSDP 分片叠加，命名与 un-shard 更复杂 | 初版**只支持 dense**，MoE 留后续 |
| offload 交互 | `torch_memory_saver` LD_PRELOAD 路径假设 DDP buffer 语义 | 单独验证 FSDP 下 offload→rollout→reload→train 一轮显存正确回收 |
| 收益是否达预期 | 目标是省显存 / 用更少卡训更大模型；需实测 vs 现有 `--use-distributed-optimizer` 的边际收益 | 跑通后对比 per-GPU 峰值显存与吞吐，再决定是否设为默认 |

---

## ⏱️ 工作量估计

* **最小打通**（单机，dense，TP-only/无 PP-CP，train step 跑通 + 一次权重同步对齐 + HF 导出）：**3–6 天**。核心耗时 = 权重同步 un-shard + 同步一致性 sanity + checkpoint 格式适配。
* **并行组合扩展**（加 PP/EP/CP，逐组合验证 + ckpt 互通/转换 + 文档）：再 **3–5 天**。
* **MoE/EP 专家层 FSDP**：额外 **1 周+**，命名与 un-shard 最复杂。

> 对比基线：LoRA handoff 那条路是"接现成能力"；本条是"换数据布局假设"，正确性验证成本更高，但功能改动面更聚焦（主要就是 update_weight + checkpoint 两块）。

---

## ✅ 建议与下一步

1. **先确认方向**：本文按 **选项 A（Megatron-FSDP）** 写；若你要的是 torch-FSDP2 或复活上游后端，路线不同，先定。
2. 选 A 的话，按**最小组合**（dense / TP-only / 无 PP-CP / `fsdp_dtensor`）打通：改动 #1 #2 #3 #4。
3. 用「**同步前后 train↔SGLang logprob 一致**」锁正确性，这是本任务唯一不可省的 gate。
4. 跑通后对比显存/吞吐确认收益，再扩并行组合与 MoE；首跑前做参数+组合校验 sanity，跑完导出 logs 由 **Codex xhigh** review 再下结论（对标 AGENTS.md）。

---

## 📎 关键代码索引

* 模型构建/包装：`slime/backends/megatron_utils/model.py:223`（Megatron `get_model`，按 `use_megatron_fsdp` 自动改包）；DDP 类型假设 `model.py:249/261/664`
* 权重同步：`slime/backends/megatron_utils/update_weight/common.py`（`named_params_and_buffers` / `all_gather_param` / `all_gather_params_async`）+ `update_weight_from_tensor.py` / `update_weight_from_distributed{,_delta}.py`
* megatron→HF：`slime/backends/megatron_utils/megatron_to_hf/__init__.py`
* Checkpoint：`slime/backends/megatron_utils/checkpoint.py`、`model.py:save_hf_model`(~882)、`hf_checkpoint_saver.py`
* 参数解析：`slime/backends/megatron_utils/arguments.py:181`（复用 Megatron parser，`ignore_unknown_args=True`）
* Megatron-FSDP 实现：`megatron/core/distributed/fsdp/mcore_fsdp_adapter.py`（`FullyShardedDataParallel`）+ `fsdp/src/README.md`；torch-FSDP2：`torch_fully_sharded_data_parallel.py`
* 组合约束来源：`megatron/training/arguments.py:598/605-621/656-696/920-957/1041-1048`
