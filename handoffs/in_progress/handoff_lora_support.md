# Slime LoRA 训练支持：可行性评估与接入方案

> [!NOTE]
> **一句话结论**：slime 目前**不支持** LoRA 训练，但**可行且工作量可控**——因为 slime 已依赖的
> **Megatron-Bridge 自带一套 TP/EP 感知的生产级 LoRA/PEFT 实现**，且 slime 已在用 bridge 路径。
> 工作从"自研 LoRA"降级为"把现成能力接进 slime 的训练循环与权重同步链路"。
> 本文为**评估 + 方案**，尚未动工。

---

## 📊 现状盘点 (Dashboard)

| 维度 | 状态 | 说明 |
| :--- | :---: | :--- |
| slime 原生 LoRA | ❌ 无 | 无 `--lora-*` 参数；无 `peft` 依赖；代码里 `q_lora_rank/kv_lora_rank` 是 GLM5 **MLA 架构**的低秩投影，**不是** PEFT adapter |
| 选择性冻结（弱替代） | ✅ 有 | `--only-train-params-name-list` / `--freeze-params-name-list`（正则匹配，见 `model_provider.py:269-283`）——手动选层训练，**省显存远不如 LoRA** |
| Megatron-Bridge LoRA | ✅ 现成 | `megatron/bridge/peft/`（`LoRA` / `LoRAMerge` / TP-EP adapter / `peft_bridge.py`）已安装可用 |
| SGLang LoRA serving | ✅ 现成 | SGLang `0.5.12.post1` 自带 LoRA serving（路线 B 需要） |

---

## 🔑 决定性发现：Megatron-Bridge 自带 LoRA

环境已安装 `/usr/local/lib/python3.12/dist-packages/megatron/bridge/peft/`，提供：

* **`LoRA` PEFT 类**（`peft/lora.py`）：用法 `model = LoRA(target_modules=[...], dim=32, alpha=32)(model)`。
  自动冻结 base、按模块名匹配注入 adapter，**支持 model chunk 列表（PP 友好）**。
  关键字段：`target_modules / exclude_modules / dim(默认32) / alpha(默认32) / dropout / lora_A_init_method(xavier) / lora_B_init_method(zero) / lora_dtype`。
* **TP/EP 感知 adapter**：`ParallelLinearAdapter` / `TELinearAdapter` / `TEFusedLoRALinear`，以及 MoE 专家 LoRA（`GroupedExpertLinearAdapter` / `LoRATopKRouter`）。
  **张量并行/专家并行切分这一最棘手的正确性问题，bridge 已处理好。**
* **`LoRAMerge`**：把 adapter 合并进 base 权重（`W_eff = W + (α/r)·B·A`）。
* **`MegatronPeftBridge`（`models/conversion/peft_bridge.py`）**：Megatron adapter 权重 ↔ HF PEFT 格式互转
  （`convert_adapter_weights_to_peft_state` / `build_adapter_config_dict` / Column/Row parallel 映射）——正是同步给 SGLang 所需的 megatron→HF 转换。

> ⚠️ 注意：`megatron.core`（`/root/Megatron-LM`）本身**无** LoRA/PEFT；LoRA 能力来自 **Megatron-Bridge** 包，不是 core。

---

## 🧭 两条路线（核心差异 = 权重如何同步给 SGLang）

slime 在 RL 中每隔 `--update-weights-interval` 把训练权重同步给 SGLang（见 `actor.py:595-653` + `update_weight/`）。

### 路线 A — LoRA 训练 + 同步前合并成全量权重（✅ 推荐先做）

* 训练：base 冻结、只有 adapter 进优化器 → **省优化器状态 + 梯度显存**（LoRA 主要收益）。
* 同步：用 `LoRAMerge` 算出合并后全量权重，再走 slime **现有** `convert_to_hf` + NCCL/IPC 同步。
* **SGLang 完全不改**，只看到"更新后的全量权重"。
* 代价：同步带宽 / 推理端与现状相同（仍全量），只在**训练侧**省显存——但这通常正是诉求（更少卡训更大模型）。
* 风险最低、改动集中，作为第一版打通。

### 路线 B — 只同步 adapter + SGLang 原生 LoRA serving（收益最大，工作量更大）

* SGLang 只加载一次 base（静态），每步只收 adapter 增量。
* 需要：`MegatronPeftBridge` 产 HF PEFT adapter → slime update_weight 路径改为只发 adapter → 对接 SGLang LoRA 热更 API → SGLang 启动加 `--enable-lora` 等。
* 同步 payload 极小；但 slime `update_weight_*` 围绕"全模型 param 遍历"写，且 **SGLang 在 RL 下动态热更 LoRA 的成熟度是主要风险**。

---

## 🛠️ 路线 A 改动点（精确到文件）

| # | 改动 | 文件 | 内容 |
| :-: | :--- | :--- | :--- |
| 1 | 新增 CLI | [`slime/utils/arguments.py`](file:///nfs/FM/chenshuailin/projects/kernel_agents/slime/slime/utils/arguments.py) | `--lora-enable / --lora-rank / --lora-alpha / --lora-target-modules / --lora-dropout` |
| 2 | 注入 LoRA | [`model_provider.py`](file:///nfs/FM/chenshuailin/projects/kernel_agents/slime/slime/backends/megatron_utils/model_provider.py)（构建返回处）或走 `--custom-model-provider-path` | `model = LoRA(...)(model)`；bridge 路径下尤其顺 |
| 3 | 优化器 | [`model.py:setup_model_and_optimizer`](file:///nfs/FM/chenshuailin/projects/kernel_agents/slime/slime/backends/megatron_utils/model.py)（约 199-239） | 基本不改——`get_megatron_optimizer` 按 `requires_grad` 收集，LoRA 已冻结 base。**需实测**分布式优化器对"大部分 param frozen"是否有假设问题 |
| 4 | 同步前合并 | [`update_weight/update_weight_from_tensor.py`](file:///nfs/FM/chenshuailin/projects/kernel_agents/slime/slime/backends/megatron_utils/update_weight/update_weight_from_tensor.py) / `update_weight_from_distributed.py` 的权重迭代器 | 调 `LoRAMerge` 得合并后全量张量，再走现有 `convert_to_hf` |
| 5 | Checkpoint | [`checkpoint.py`](file:///nfs/FM/chenshuailin/projects/kernel_agents/slime/slime/backends/megatron_utils/checkpoint.py) / `model.py:save_hf_model`（约 882-930） | 初版只存"合并后全量"；后续可用 `MegatronPeftBridge` 存 adapter |
| 6 | 加载 base | `checkpoint.py:_load_checkpoint_hf` | 先 bridge 加载 HF base，**再**注入 LoRA（顺序要对） |

---

## ⚠️ 风险与待验证点

| 风险 | 描述 | 应对 |
| :--- | :--- | :--- |
| 分布式优化器 + 几乎全冻结 | Megatron distributed optimizer 在"只剩少量 LoRA param 可训"时 bucket/通信行为未知 | 实测；必要时关 `--use-distributed-optimizer` 或调 bucket |
| 合并的 TP 正确性 | `B·A` 合并须在与 base 一致的分片布局下做（bridge adapter 已 TP 感知，合并在 all_gather 之前按分片本地完成） | **强制 sanity check：合并前后 logits 必须一致** |
| MoE 模型 | 专家 LoRA（GroupedExpert）更复杂 | 初版**只支持 dense / 非专家层**，MoE 留后续 |
| bridge 版本契合 | `LoRA(...)` 与 slime 当前 bridge provider 的构造顺序需对齐 | 在 `--megatron-to-hf-mode bridge` 路径优先验证 |

---

## ⏱️ 工作量估计

* **路线 A 打通**（单卡 / 小 TP，dense 模型，merge-then-sync）：**2–4 天**。核心 = 参数接线 + 合并迭代器 + 合并正确性 sanity check。
* **多 TP/PP 验证 + checkpoint adapter 存取 + 文档**：再 **2–3 天**。
* **路线 B**（SGLang adapter 热更）：额外 **1–2 周**，主要耗在 SGLang 动态 LoRA 对接与稳定性。

---

## ✅ 建议与下一步

1. 先做**路线 A**：复用 bridge `LoRA` + `LoRAMerge`，最小改动打通
   "LoRA 训练 → 合并 → 现有同步链路 → SGLang rollout"。
2. 用"**合并前后输出一致**"的 sanity check 锁住正确性（对标 AGENTS.md「行为敏感改动补单测/sanity」）。
3. 观察收益（训练显存下降、奖励曲线正常）后，再评估是否值得做路线 B 省同步带宽。
4. 首跑前 Sanity（参数校验 + 冻结比例 + 合并一致性抽样），跑完导出 logs 由 **Codex xhigh** review 再下结论。

---

## 📎 关键代码索引

* 模型构建：`slime/backends/megatron_utils/model_provider.py`（`model_provider`，custom hook `--custom-model-provider-path`；冻结逻辑 269-283）
* 优化器：`slime/backends/megatron_utils/model.py:199-239`（`get_megatron_optimizer`）
* 权重同步：`slime/backends/megatron_utils/actor.py:595-653` + `update_weight/{update_weight_from_tensor,update_weight_from_distributed}.py`
* megatron→HF：`slime/backends/megatron_utils/megatron_to_hf/__init__.py`
* Bridge LoRA：`megatron/bridge/peft/{lora,lora_layers,adapter_wrapper,canonical_lora}.py` + `models/conversion/peft_bridge.py`
