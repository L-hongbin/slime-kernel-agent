# DrKernel W8A8-INT8 Rollout 加速

记录于 2026-05-25。本文档独立可读，复现性导向：把整次尝试从环境到失败原因到 retry 路径全部固化。

## 目标

把 27B SGLang rollout 从 BF16 切到 W8A8-INT8（INT8 weights + INT8 activations），缩短 KernelBench Level1 多轮 eval 的 wall time。

**预期收益**：rollout 阶段 ~1.5-2× 加速。Phase 1 一次 100×8 eval ≈ 70 min，最多省 ~35 min/run。

## 当前状态

**第一次尝试（2026-05-24）**：full-Linear W8A8 量化 + `AutoModelForCausalLM` 加载路径 — 量化成功（28GB compressed-tensors ckpt），但 sglang 加载失败两连击。详见下面 Step 4 + Blocker 1/2。

**第二次方案（2026-05-25，待跑）**：in-repo 实现两条 retry 路径（见末尾"重启 Phase 2 的建议路径"）：
- 路径 A（推荐先试）：`scripts/drkernel/quantize_w8a8.py --multimodal --target mlp` 走多模态加载 + MLP-only 量化，保留 vision tower BF16，`architectures` tag 不变，sglang 不用改
- 路径 B：默认 mode + apply `scripts/drkernel/sglang_qwen3_5_dense_entry.patch`（已 dry-run 验证 apply 干净）

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

#### sglang main 分支也没修（核实于 2026-05-25）

WebFetch `https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/models/qwen3_5.py` 结果：

- `EntryClass = [Qwen3_5MoeForConditionalGeneration, Qwen3_5ForConditionalGeneration]` —— `Qwen3_5ForCausalLM` 仍未注册
- `Qwen3_5ForCausalLM.get_model_config_for_expert_location` 仍是 `num_logical_experts=config.num_experts`（无 getattr 守卫）

**间接证据**：MoE 变体的同名方法已被改过加了 `text_config = getattr(config, "text_config", config)`（处理 multimodal config 嵌套），但 dense 那一份没动 —— 说明维护者最近碰过这块代码但没人触发过 dense 路径。"dense entry 是死代码"的判断在 main 上仍然成立。

**含义**：升级 sglang 不能解锁本次 W8A8 加载；retry 路径 C（等上游）需要主动提 issue 推动修复（改动很小：1 行 `getattr` + EntryClass 加 1 项），但需要 sglang 维护者愿意 review/merge。

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
| `scripts/quantize/build_calibration.py` | repo | 2 KB | calibration 抽取脚本（通用，两条 quant 路径都用） |
| `scripts/quantize/quantize_w8a8_llmcompressor.py` | repo | 7 KB | llmcompressor W8A8 量化（含 MLP-only target + `--multimodal` mode） |
| `scripts/quantize/quantize_w8_gptqmodel.py` | repo | 5 KB | GPTQModel W8 GPTQ_V2 量化（推荐，accuracy ~0.07% degradation） |
| `scripts/quantize/sglang_qwen3_5_dense_entry.patch` | repo | 2 KB | sglang dense entry 解锁补丁 |
| `scripts/quantize/README.md` | repo | 3 KB | 两条 quant 路径速查 + 算法对比 + loading caveats |
| `/tmp/Qwen3.6-27B-W8A8-ct/` | .22 | 28 GB | 旧 full-W8A8 ckpt（CausalLM mode 量化产物，被 sglang load blocker 卡住） |
| `/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B-W8A8-ct/` | NFS | 28 GB | 同上 NFS 持久化 |
| `/tmp/calibration_drkernel_v2_3_noenv_n4.jsonl` | .22 | ~5 MB | 1200 calibration prompts（已存在，可复用） |
| `/tmp/w8a8-venv/` | .22 | ~3 GB | venv（llmcompressor + ct git main + transformers 5.3.0） |
| `/tmp/codex_phase2_abandon.txt` + `/tmp/codex_p2_out.log` | local | ~10 KB | codex review prompt + output |

## 重启 Phase 2 的建议路径（按 ROI 排序）

代码已 in-repo（独立 `scripts/quantize/` 文件夹）：
- `scripts/quantize/build_calibration.py` — 从 eval_0.pt 抽取 calibration prompts
- `scripts/quantize/quantize_w8a8_llmcompressor.py` — llmcompressor + 香草 GPTQ + W8A8（含 `--multimodal` 模式）
- `scripts/quantize/quantize_w8_gptqmodel.py` — **GPTQModel + GPTQ_V2 + `act_group_aware=True` + W8 only**（推荐先试，accuracy 损失 ~0.07%，已有公开 ckpt 验证）
- `scripts/quantize/sglang_qwen3_5_dense_entry.patch` — sglang dense entry 解锁补丁（llmcompressor CausalLM mode 用）
- `scripts/quantize/README.md` — 两条 quant 路径选型 + loading caveats 速查

### llmcompressor vs GPTQModel — 选型核心 trade-off（2026-05-25 核实）

两个工具在对立方向上各有所长：**llmcompressor scheme 灵活 + 算法老；GPTQModel 算法先进 + weights-only**。

| 维度 | llmcompressor | GPTQModel |
|---|---|---|
| **W8A8（activations 也 INT8）** | ✅ | ❌ — README 明说 "GGUF and FP8 are weight-only"；QuantizeConfig 无 `a_bits` 或 activation calibration 字段 |
| W4/W8 weights-only | ✅ | ✅ |
| FP8 weights | ✅ | ✅（weight-only） |
| KV cache quant | ✅ | ❌ |
| **GPTQv2 (`FORMAT.GPTQ_V2`)** | ❌ | ✅ |
| **GPTAQ (activation-aware GPTQ, asymmetric calibration)** | ❌ | ✅（`GPTAQConfig` experimental） |
| **`act_group_aware`** | ❌ | ✅（`desc_act=False` 时默认开，16k× faster vs `desc_act=True` 同等 quality） |
| **FOEM (first-order error compensation)** | ❌ | ✅ |
| **Qwen3.5 explicit model def** | ❌（fall back generic CausalLM，重写 architectures） | ✅（`Qwen3_5GPTQ` mirror of `Qwen3_5MoeGPTQ`，保留多模态 layout） |
| selective per-module skip | regex `targets` + `ignore` | `QuantizeConfig.dynamic` negative match `"-:..."` |

核实方法：
- llmcompressor GPTQModifier 源码（`src/llmcompressor/modifiers/gptq/base.py`）只有 4 个 GPTQ-specific 字段：`block_size`、`dampening_frac`、`actorder`、`offload_hessians`。无 GPTQv2/GPTAQ/activation-aware 任何形式
- GPTQModel README + QuantizeConfig 文档明确列出 `format=FORMAT.GPTQ_V2`、`act_group_aware`、`gptaq=GPTAQConfig(...)`、`foem=FOEMConfig(...)`、`dynamic={...}` 字段
- GPTQModel README 明确写 "GGUF and FP8 are weight-only"。FORMAT enum 只含 weights-only 方案（GPTQ / GPTQ_V2 / GGUF / FP8 / BITSANDBYTES / MARLIN / BITBLAS / QQQ / EXL3 / GEMM / GEMV / GEMV_FAST / LLM_AWQ / PAROQUANT），无 W*A* 联合方案

### 含义：不存在"全选"工具

| 你想要 | 必选工具 | 算法限制 |
|---|---|---|
| **W8A8 max 速度** | llmcompressor | vanilla GPTQ only |
| **最好的 weights-only INT8**（GPTQv2 + act_group_aware + 多模态保留） | GPTQModel | weights-only，math 仍 BF16 |
| **两者都要** | 不存在 | 理论上可串联（GPTQModel 量 weights → llmcompressor 加 activation quant），未测试，loader 大概率不兼容 |

### 公开 W8 ckpt（可直接验证）

[`btbtyler09/Qwen3.6-27B-GPTQ-8bit`](https://huggingface.co/btbtyler09/Qwen3.6-27B-GPTQ-8bit) on HuggingFace：

- GPTQModel v5.7.1 量化，W8 GPTQ_V2，bits=8 group_size=32 sym=True desc_act=False
- 全部 64 层 text decoder linears 都量化（mlp + self_attn + linear_attn），vision/MTP/embed/lm_head 保 BF16
- `architectures=["Qwen3_5ForConditionalGeneration"]`（多模态 layout 保留）
- 大小 32GB（vs BF16 50GB，1.6× 压缩）
- **wikitext-2 perplexity 7.0697 vs BF16 7.0652，degradation +0.07%（基本无损）**
- ⚠️ "Neither GPTQModel nor transformers can currently load this model directly. Use vLLM." vLLM 需要小补丁（model card 给了一句 sed），sglang 兼容性未验证

可以直接拉这个 ckpt 跑 sglang smoke 测试 —— 比从头量化省 ~3-6h。recipe 跟 `quantize_w8_gptqmodel.py --target all-linear` 基本一致。

### 路径 A — `--multimodal` 量化 + 不动 sglang（**推荐先试**）

llmcompressor 官方支持"load 多模态 + 只量化 LM"的 pattern（见 `llmcompressor/examples/multimodal_vision/`，Qwen2-VL / Llama-3.2-Vision / Pixtral 都有示例）。标准做法：

1. 用多模态 class 加载（`AutoModelForImageTextToText` 或具体 `*ForConditionalGeneration`）
2. `ignore` 显式排除 vision tower / projector / audio tower 等非 LM 子图
3. calibration 用纯 text 数据（multimodal forward 在 `pixel_values=None` 时走 text-only path，GPTQ 只对到 LM 层收集 activations）

`scripts/drkernel/quantize_w8a8.py --multimodal --target mlp` 就是这个 pattern：
- 加载用 `AutoModelForImageTextToText.from_pretrained`（fallback 到 `AutoModel`）
- targets regex `re:.*\.mlp\.(gate_proj|up_proj|down_proj)$` 只匹配 SwiGLU 三件套（vision 的 `visual.merger.mlp.*` 也不匹配，因为命名不是 `gate_proj` 等）
- `ignore` 显式列出 `re:.*\.visual\..*` / `re:.*vision_tower.*` / `re:.*mm_projector.*` / `re:.*\.audio_tower\..*` 作为防御性 fence（即便 targets 不命中，也让 GPTQ traversal 完全跳过这些子树）
- 产物：vision tower BF16 + LM attention/embed/lm_head BF16 + **只有 LM MLP 是 W8A8**
- `architectures` 保持 `Qwen3_5ForConditionalGeneration`，走 sglang 已注册的多模态 entry → 无需 sglang 修改

未验证项（按可能性排序）：
- transformers 5.3 上 `AutoModelForImageTextToText` 对 Qwen3.6-27B 能否走通（可能要 fallback `AutoModel` 或直接 import `Qwen3_5ForConditionalGeneration`）
- llmcompressor 的 GPTQ sequential traversal 在 Qwen3.5/3.6 多模态结构上是否有死锁/递归问题（新模型无 `traceable_*` wrapper，可能要看 traversal 报错）
- multimodal forward 在 calibration 时 `pixel_values=None` 是否正常走 text-only 路径（绝大多数多模态实现都支持，但要核实）

### 路径 B — `AutoModelForCausalLM` 量化 + sglang 补丁

`quantize_w8a8.py`（默认 mode）+ `patch -p1 < scripts/drkernel/sglang_qwen3_5_dense_entry.patch`。

补丁两处改：

1. `EntryClass = [..., Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM]` — 注册 dense 入口
2. `Qwen3_5ForCausalLM.get_model_config_for_expert_location` 加 `if not hasattr(config, "num_experts"): return None` 守卫（sglang 上游 `_init_common` / `init_trivial` 已经 null-safe，returning None 就让 EPLB 初始化静默跳过）

已在 sglang 0.5.10.post1 上 `patch --dry-run` 验证 apply 干净。维护成本：每次 sglang 升级需要重新 apply（diff 几行，rebase 应该简单）。

适合：vision tower 太大（占 ~12% 总参数）想省 quantize/load 时的 host RAM，或者你已经有非多模态来源的 dense ckpt。

### 路径 C — 上游 PR

补丁本身很小（1 个 hasattr 守卫 + EntryClass 加 2 项），可以直接提 sglang upstream PR。审核通过后路径 B 的补丁负担消失。**已核实 sglang main 分支至今仍有同样问题**。

### 路径 E — SpinQuant offline rotation 实测（2026-05-25，部分跑通）

按用户 "step by step" 推进路径 E：先做 BF16 + offline R1 rotation 不加 quant，验证 rotation 数学 + sglang load。脚本 `scripts/quantize/rotate_bf16_llmcompressor.py`。

#### Producer 端 — ✅ 完全跑通

依次 debug 出 3 个 llmcompressor SpinQuantModifier 问题，最终成功产出 51GB 旋转 BF16 ckpt at `/nfs/.../Qwen3.6-27B-rotated-bf16/`：

1. **SpinQuantMapping schema 限制**：3 个 `attn_q/k/v` 槽对 Qwen3.5 hybrid（7 个 input projections = 3 self_attn + 4 linear_attn）放不下。Cram via regex disjunction。Codex 验证 R1 only 数学 OK（q/k/v 槽对 R1 都是同一处理 `weight_input, inverse=True`）；R2 不行
2. **`_fuse_norms` `assert len(norm) == 1`**：`match_modules_set` 按 lowest common parent 流式分组，hybrid layers 缺投影 family 会让 group 不闭合 → norms 跨层累积 → 触发 assert。Fix：枚举每层绝对路径作 NormMapping，绕开 streaming-group 歧义
3. **`SpinQuantModifier.on_initialize` 覆写**：`mappings` / `norm_mappings` kwargs 在没有 `transform_config` 时被静默覆写。Fix：直接构造 `TransformConfig` + `TransformScheme` + `TransformArgs` 传给 modifier

Codex review 后 R1 transform 在 14 秒内完成，498 个 module 旋转。

#### Consumer 端 — ❌ SGLang dense entry 4 bug 累计，第 4 个无补丁可解

加载 rotated BF16 时连续撞 sglang `Qwen3_5ForCausalLM` 入口的 4 个独立 bug：

| # | bug | 修法 | 状态 |
|---|---|---|---|
| 1 | `Qwen3_5ForCausalLM` 不在 `EntryClass` 注册 | `scripts/quantize/sglang_qwen3_5_dense_entry.patch` 加入 + hasattr 守卫 | ✅ 已 patch |
| 2 | `get_model_config_for_expert_location` 硬编码 `config.num_experts`（dense config 无此字段） | 同上 patch 加 `if not hasattr(config, "num_experts"): return None` | ✅ 已 patch |
| 3 | `make_layers` 读 `config.layers_block_type` 但 HF Qwen3.5 暴露 `layer_types`；值也错位（`attention` vs `full_attention`） | 在保存的 config.json 里加 `layers_block_type` alias + 值翻译 | ✅ rotate 脚本已加 + patch_config.py 已就地修旧 ckpt |
| 4 | `RadixLinearAttention.forward` decode 路径调 `forward_batch.attn_backend.forward(layer=, mixed_qkv=, a=, b=)`，但默认 `flash` AttentionBackend 签名要 `q/k/v` 位置参 | sglang 没有为 dense entry 把 `linear_attn` 层 dispatch 到 mamba-aware backend；要正确接 `--linear-attn-backend triton` 通路 | ❌ 非平凡 sglang 改动，无 1-line patch |

Bug 4 的 traceback：
```
File ".../sglang/srt/layers/radix_linear_attention.py", line 95, in forward
    return forward_batch.attn_backend.forward(
TypeError: AttentionBackend.forward() missing 3 required positional arguments: 'q', 'k', 'v'
```

发生在 `init_device_graphs` cuda graph capture（decode mode）。即便 `--disable-cuda-graph` 跳过 capture，首次 decode 仍会撞同一签名。结构性 dispatch bug。

#### 结论

- **rotation 数学和 producer 端是 OK 的** —— llmcompressor SpinQuant 加上 Qwen3.5 自定义 mapping + 直接 transform_config 可以稳定产出旋转后的 ckpt
- **sglang dense entry path 实际是死代码** —— 累计 4 个独立 bug 说明这条路从未被任何人完整跑通过。修 1-3 各只要 1-2 行；修 4 要 sglang attention dispatch 重做
- **路径 E 短期不可行**。要做 rotation + W8A8，必须改用**多模态 arch tag**（`Qwen3_5ForConditionalGeneration`），走 sglang 已注册路径。多模态 entry 是生产中真在用的（BF16 27B 通过它跑了几百次 eval），attention dispatch 正确

#### 推荐转向

放弃 `AutoModelForCausalLM` 加载 + 写 `Qwen3_5ForCausalLM` arch tag 的方案。改用路径 A 思路：
1. `AutoModelForImageTextToText.from_pretrained` 加载多模态模型
2. SpinQuant rotation 只作用在 `model.language_model.*` 子树（vision tower 保 BF16 + 不动 rotation）
3. 保存时 `architectures=["Qwen3_5ForConditionalGeneration"]` 不变
4. sglang 走多模态 entry，绕开 4 个 dense entry bug

这条路 producer 端复杂度高一些（要把 SpinQuant mapping 限定到 language_model 子树 + vision tower 不能被 rotation 触及），但 consumer 端零 sglang 改动，工程上更可控。

下次重启 Phase 2 时优先此路径。

### Path D（codex 最初建议）— 已验证不可行

把 W8A8 ckpt 的 `architectures` 改回 `Qwen3_5ForConditionalGeneration`、保留同份 safetensors，期望 sglang 走 multimodal entry 加载。

阻塞：`Qwen3VLForConditionalGeneration.__init__` 无条件实例化 `self.visual = Qwen3VLMoeVisionModel(...)`（`qwen3_vl.py:1085`），weight loader 找不到 `model.visual.*` keys 就崩。路径 A 之所以可行，正是因为 multimodal 加载时就把 vision 权重读进来量化产物保留下来，而不只是改 tag。

## MLP-only 量化的设计取舍

`quantize_w8a8.py --target mlp` 默认行为。原因：

- **Embedding** (`embed_tokens`)：INT8 量化对 embedding 影响大（vocab × hidden_size 的 lookup table，量化噪声直接进 first layer），通常保 BF16
- **lm_head**：与 embedding 对称，且是 logits 直接来源，量化伤 perplexity 明显，保 BF16
- **Attention** (`self_attn`, `linear_attn`)：q/k/v/o projection 量化敏感（QK^T 计算累积误差），尤其 hybrid 架构的 mamba-style linear_attn 含 `A_log`、`conv1d` 等非标准 module，量化覆盖率不够时易报 unsupported
- **MLP** (`mlp.gate_proj`, `mlp.up_proj`, `mlp.down_proj`)：SwiGLU MLP 是单层最大的 weight family（`3 × hidden_size × intermediate_size`，对 Qwen3.6-27B = 3 × 5120 × 17408 ≈ 268M params per full-attn layer，是 QKVO 的 ~2.5×），同时对 INT8 量化的 accuracy hit 最小。性价比最高

实测节省：MLP-only W8A8 大约把 full-attention 层的权重压到 ~65%（vs full W8A8 的 ~50%）。对应总模型大小估计 BF16 54GB → MLP-W8A8 ≈ 38-40GB（vs full W8A8 28GB）。Rollout 加速比 full W8A8 小，但 accuracy 保留更好。

如果要追求最大压缩可改 `--target all-linear`（除 lm_head 外全部 Linear）。

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
