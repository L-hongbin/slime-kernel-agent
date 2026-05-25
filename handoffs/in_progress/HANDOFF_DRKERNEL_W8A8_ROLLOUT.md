# DrKernel W8A8-INT8 Rollout 加速尝试（已 ABANDON）

记录于 2026-05-25。本文档独立可读，复现性导向：把整次尝试从环境到失败原因到 retry 路径全部固化，下次想重启时不用从零开始。

## 目标

把 27B SGLang rollout 从 BF16 切到 W8A8-INT8（INT8 weights + INT8 activations），缩短 KernelBench Level1 多轮 eval 的 wall time。

**预期收益**：rollout 阶段 ~1.5-2× 加速。Phase 1 一次 100×8 eval ≈ 70 min，最多省 ~35 min/run。

**最终结论**：**ABANDON**。量化技术上成功（28GB compressed-tensors W8A8 ckpt 已生成），但 SGLang 加载失败两连击，需要 sglang-side 改动或额外的 weight surgery，ROI 不够。

## 环境（已搭建，可复用）

`/tmp/w8a8-venv/` on **192.168.16.22**：

- transformers **5.3.0**（27B 是 `Qwen3_5ForConditionalGeneration` 多模态架构，4.57 系列不识别）
- llmcompressor: **git main**（pypi 0.10 pin transformers<=4.57.6，与上面冲突）
- compressed-tensors: **git main**（pypi 旧版报 `ModuleNotFoundError: compressed_tensors.distributed`）
- torchvision：**已卸载**（装上会触发 `RuntimeError: operator torchvision::nms does not exist` 的 ABI mismatch；但卸载后 transformers 加载 Qwen3.6 model 不再尝试 import vision processor → OK）

依赖安装顺序教训：先 venv 内 `pip install transformers==5.3.0`，再 git+main 装 llmcompressor / compressed-tensors，最后 `pip uninstall torchvision`。

## 实施步骤（按时间顺序）

### Step 1 — Calibration dataset 提取（成功）

脚本：`/tmp/build_w8a8_calibration.py` on .22

从 v2_3 noenv n=4 eval dump 抽取 T0/T1/T2 rendered prompts：

```
SRC = checkpoints/Qwen3.6-27B/20260524_124613_ctx65536_n4_summ1600_v2_3_noenv/dumps/rollout_data/eval_0.pt
OUT = /tmp/calibration_drkernel_v2_3_noenv_n4.jsonl
```

输出 1200 条 prompts（每条带 `text`/`turn`/`problem_id`），代表生产 rollout 的分布。GPTQ 取前 512 条作为 calibration set。

### Step 2 — GPTQ 量化（成功，~3h）

脚本：`/tmp/quantize_qwen36_w8a8.py` on .22

关键配置：

```python
MODEL_PATH = "/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B"
OUT_PATH = "/tmp/Qwen3.6-27B-W8A8-ct"
CALIB_PATH = "/tmp/calibration_drkernel_v2_3_noenv_n4.jsonl"
CALIB_N = 512
CALIB_MAX_LEN = 4096

from llmcompressor.modifiers.quantization import GPTQModifier
from llmcompressor import oneshot
from datasets import Dataset

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
)

calibration_ds = Dataset.from_dict({"text": calibration_texts})

# SmoothQuant has no arch mapping for Qwen3.6 (qwen3_5 model_type) → skip
recipe = [GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])]

oneshot(
    model=model,
    processor=tokenizer,                # 不要同时传 tokenizer + processor，会被 oneshot 拒绝
    dataset=calibration_ds,             # 必须 HF Dataset，不能 list
    recipe=recipe,
    max_seq_length=CALIB_MAX_LEN,
    num_calibration_samples=512,
    output_dir=OUT_PATH,
)
model.save_pretrained(OUT_PATH, save_compressed=True)
tokenizer.save_pretrained(OUT_PATH)
```

跑了 ~3h 在 .22 8×A800（layer-by-layer GPTQ，65 layers × ~3 min/layer）。

**坑（已避开，记在这里给下次）**：
- llmcompressor 0.10 pypi 强制 transformers<=4.57.6，必须 git main
- `pip` 0.14 太旧装不上 git main 的 compressed-tensors，要升 pip
- SmoothQuantModifier 对 Qwen3.6 (`qwen3_5` model_type) 没有 preset mapping → `Error resolving mappings for given architecture`。drop 掉只用 GPTQ
- `oneshot` 不接受同时传 `tokenizer` + `processor`（互斥）→ 只传 `processor=tokenizer`
- `oneshot` 要 HF `Dataset.from_dict({"text": ...})`，传 list 会报 `'list' object has no attribute 'column_names'`
- torchvision 装上就 ABI mismatch；卸了反而 OK（transformers 在 trust_remote_code=True 路径下不强求 vision processor）

### Step 3 — Checkpoint 落盘（成功）

输出：

```
/tmp/Qwen3.6-27B-W8A8-ct/                              # .22 local
/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B-W8A8-ct/  # NFS copy (28GB, 36s scp)
```

config.json 关键字段：
- `architectures: ["Qwen3_5ForCausalLM"]` ← **关键变化**：从原 BF16 的 `Qwen3_5ForConditionalGeneration` 被 llmcompressor 改成 ForCausalLM（因为它走 `AutoModelForCausalLM.from_pretrained` 加载多模态原模型时只保留 LM）
- `quantization_config.format: int-quantized`
- `quantization_config.input_activations`: int8, token-dynamic
- `quantization_config.weights`: int8, per-channel static, `actorder: static`
- `quant_method: compressed-tensors`

权重命名（用 safetensors 工具核实，1347 keys）：
- `model.language_model.layers.*` 前缀（多模态命名约定，**没有** `model.layers.*` 平铺）
- `lm_head.weight` 顶层
- **没有** `model.visual.*` keys（vision tower 被 llmcompressor 丢弃）

### Step 4 — SGLang 加载 smoke（**失败 ×2**）

环境：sglang 0.5.10.post1 at `/sgl-workspace/sglang/python/sglang/` on .22

启动命令：

```
CUDA_VISIBLE_DEVICES=0,1 python3 -m sglang.launch_server \
  --model-path /nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B-W8A8-ct \
  --tp-size 2 --port 30099 --host 127.0.0.1 \
  --context-length 8192 --mem-fraction-static 0.7 --trust-remote-code
```

#### Blocker 1 — `Qwen3_5ForCausalLM` 不在 EntryClass

```
ValueError: Qwen3_5ForCausalLM has no SGlang implementation and the
Transformers implementation is not compatible with SGLang.
```

根因：`/sgl-workspace/sglang/python/sglang/srt/models/qwen3_5.py` 文件末尾：

```python
class Qwen3_5ForCausalLM(nn.Module): ...           # 类定义存在
class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM): ...
class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration): ...
class Qwen3_5MoeForConditionalGeneration(Qwen3VLForConditionalGeneration): ...

EntryClass = [Qwen3_5MoeForConditionalGeneration, Qwen3_5ForConditionalGeneration]
# Qwen3_5ForCausalLM 不在列
```

SGLang 启动时按 `EntryClass` 注册 `arch_name → model_class`，未注册的 arch 走 transformers fallback → 报错。

**临时 patch**：`sed -i` 把 `Qwen3_5ForCausalLM` 加进 `EntryClass`。**已 revert**（cp 备份恢复）。

#### Blocker 2 — `get_model_config_for_expert_location` 硬编码 MoE

补完 EntryClass 后继续报：

```
AttributeError: 'Qwen3_5TextConfig' object has no attribute 'num_experts'
  File "/sgl-workspace/sglang/python/sglang/srt/models/qwen3_5.py", line 1097,
       in get_model_config_for_expert_location
    num_logical_experts=config.num_experts,
```

根因：`Qwen3_5ForCausalLM.get_model_config_for_expert_location` 假设 MoE config，直接访问 `config.num_experts`。dense 27B 没有该字段。SGLang 启动期会无条件调用这个 classmethod 算 expert 分布做 TP 分片。

修起来涉及 sglang 类层多个 method（不止这一个 hard-code），影响面到所有走 `Qwen3_5ForCausalLM` 入口的人。

**结论**：暴露了"SGLang 这个 dense entry 实际是死代码" —— Qwen3_5ForCausalLM 类代码齐全但从没被当独立入口跑通过（否则启动期 expert-location 立刻爆 AttributeError，PR 不可能 merge）。llmcompressor 把多模态模型量化成 ForCausalLM 是常规操作，撞上 ecosystem mismatch。

## Codex 二审（独立核实，2026-05-25）

prompt: `/tmp/codex_phase2_abandon.txt`
output: `/tmp/codex_p2_out.log`

Verdict：**AGREE-ABANDON**。

核心补充信息：
1. **Path D（codex 推荐）**：拿原 BF16 full multimodal config.json + 移植 W8A8 quantization_config + 保留 `architectures=["Qwen3_5ForConditionalGeneration"]` + 指向同一份 W8A8 safetensors。我后续验证发现：
   - 权重命名匹配（W8A8 weights 已是 `model.language_model.*` 前缀）✅
   - 但 sglang `Qwen3VLForConditionalGeneration.__init__` 在 `qwen3_vl.py:1085` **无条件**实例化 `self.visual = Qwen3VLMoeVisionModel(...)`，不受 `language_only` 影响 ❌
   - sglang `--language-only` server arg 只影响 mooncake transfer engine 初始化（`model_runner.py:1054`），不跳过 vision 权重加载 ❌
   - 加载时 weight loader 会找 `model.visual.*` keys → 缺失 → 加载失败 ❌
2. **FP8 KV cache warning**：sglang Qwen3.5 docs 提到默认 scale=1.0 会伤 reasoning-heavy 精度，不要轻易加 `--kv-cache-dtype fp8` 到加速 sweep
3. llmcompressor 没有"保留原 multimodal architecture 但只 save AutoModelForCausalLM 抽取"的 save 选项

## 残留 Artifacts（保留可复用）

| 路径 | 主机 | 大小 | 说明 |
|---|---|---|---|
| `/tmp/Qwen3.6-27B-W8A8-ct/` | .22 | 28 GB | compressed-tensors W8A8 ckpt（local，可能被 /tmp 清理） |
| `/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B-W8A8-ct/` | NFS | 28 GB | 同上，NFS 持久化 |
| `/tmp/quantize_qwen36_w8a8.py` | .22 | 2.5 KB | 量化脚本（含坑解释） |
| `/tmp/build_w8a8_calibration.py` | .22 | 1 KB | calibration 抽取脚本 |
| `/tmp/calibration_drkernel_v2_3_noenv_n4.jsonl` | .22 | ~5 MB | 1200 prompts |
| `/tmp/w8a8-venv/` | .22 | ~3 GB | venv（llmcompressor + ct git main + transformers 5.3.0） |
| `/tmp/codex_phase2_abandon.txt` | local | 6 KB | codex review prompt |
| `/tmp/codex_p2_out.log` | local | 4 KB | codex review output |

## 重启 Phase 2 的建议路径（按 ROI 排序）

如果未来要 retry：

### 路径 D'（修订版，最便宜）— **推荐**

预计 4-6h。步骤：

1. 用 `safetensors` 工具从 `/nfs/.../Qwen3.6-27B/` 抽取 `model.visual.*` 全部 keys（BF16）
2. concat 到 W8A8 safetensors（`/nfs/.../Qwen3.6-27B-W8A8-ct/model.safetensors`），生成 `model-merged.safetensors`
3. 写 overlay config：copy 原 BF16 full `config.json` + 移植 W8A8 `quantization_config` 块 + 关键：在 `quantization_config.ignore` 列表加入 vision tower 各 module 路径（`re:.*model\.visual\..*`），否则 sglang quant 加载器会试图把 vision 权重当 INT8 处理
4. 指 sglang 加载新 overlay 目录，走 `Qwen3VLForConditionalGeneration` 已注册路径
5. smoke 测试 + 100×8 eval

风险：sglang 的 weight loader 可能对 quantization_config 的 ignore pattern 处理有 bug；ignore vision 后能否正常 forward 也需要验证（vision tower 在 multimodal forward 路径里被调用，但 KernelBench 都是纯 text prompts，不该走到 visual forward）。

### 路径 A（最干净但贵）

预计 3-5h calibration + debug。重做量化：

1. 加载多模态：`Qwen3_5ForConditionalGeneration.from_pretrained(...)`（不是 AutoModelForCausalLM）
2. GPTQModifier `targets` 限制到 LM 层：`targets=["re:.*model\\.language_model\\..*Linear"]`，vision tower 保留 BF16
3. save 时保留原 `architectures=["Qwen3_5ForConditionalGeneration"]`

未验证 llmcompressor 能否正确处理多模态 model 的偏部量化。需要看 llmcompressor docs / 源码确认 `targets` 的 regex 在 multimodal model 上行为。

### 路径 B（不推荐）

fork sglang 给 `Qwen3_5ForCausalLM` 写 dense-aware `get_model_config_for_expert_location`（return None 或跳过 expert-location 初始化），并把它加入 EntryClass。维护成本高，每次 sglang 升级有 regression 风险。

### 路径 C（被动）

等 sglang 上游支持 dense Qwen3_5 entry。可以提 issue 推进。

## 经验教训

1. **架构标签很关键**：llmcompressor 的 `AutoModelForCausalLM.from_pretrained` 路径会**重写** `architectures` 字段（从 `Qwen3_5ForConditionalGeneration` 改成 `Qwen3_5ForCausalLM`）。这是后续所有 sglang 加载问题的根。如果 llmcompressor 提供"保留原 architectures"选项就能省事，但目前没有
2. **"类存在 ≠ 类可用"**：sglang 的 model class 存在不代表它被注册或测试过。debug 时先看 `EntryClass` 列表
3. **MoE 假设泄漏到 dense 路径**：hybrid 架构里 dense 实现常会被 expert-location 代码假设有 MoE config。是常见但难察的 bug
4. **`--language-only` 不是"跳过 vision 加载"开关**：sglang 这个 flag 只用于 mooncake disaggregation 场景，对 weight loader 不生效。下次见到 `language_only` / `text_only` 这类 flag 要先 grep 源码确认实际作用
5. **量化前先做 sglang load smoke**：用一个 tiny model（比如 1B dense）走完整 quantize → load 流程，验证整条 pipeline。3h 量化结果发现加载不了，时间损失大
6. **/tmp 不在 NFS 上**：slime ray cluster 通过 NFS 共享，量化 output 要 cp 到 NFS 才能被远程节点加载。本次 cp 36s

## 不依赖 W8A8 的加速建议（codex 给出的 top-3）

如果纯 wall-time 是目标，这些每个 1 次 100×8 即可验证（~70 min/次），未跑过：

1. **Speculative decoding NEXTN**：`--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`（前文已注意到 sglang 注册了 `qwen3_5_mtp.py`，模型可能有 NEXTN draft 头可用）
2. **`--mamba-scheduler-strategy extra_buffer --page-size 64`**（Qwen3.5 hybrid attn 含 mamba layers，调度策略可能有空间）
3. **显存压榨**：`--max-running-requests 96/128 + --mem-fraction-static 0.92/0.94 + --schedule-policy lpm`（如果 n=8 同 prompt 共享前缀，lpm policy 可能命中更多 prefix cache）

## 相关文件 / commit

- 本次尝试无 slime 代码变更（纯 OOB 实验，sglang patch 已 revert）
- 上下文：`handoffs/in_progress/HANDOFF_DRKERNEL_SLIME_PLAN.md` 的 `## 2026-05-24/25 Phase 1+2 实验日志` 段也提及 W8A8 abandon，但简化版；本文档是详细可复现版
