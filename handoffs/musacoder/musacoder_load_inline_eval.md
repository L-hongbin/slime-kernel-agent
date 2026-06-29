# MusaCoder 评测

## 背景

**MusaCoder-27B** 要求模型输出一个完整的 Python `load_inline` 代码块：单个 code block，内含 CUDA kernel、C++ binding 和 PyTorch wrapper。

**slime** 当前的 `cuda_agent` 后端要求模型按三段式输出：

- `### CUDA_KERNELS`
- `### APPLY_BINDINGS`
- `### MODEL_NEW`

两种格式不兼容。如果直接把 MusaCoder 放进原先pipeline评测，很多样本会在格式校验或 binding 阶段失败。这不是模型能力本身的问题，而是输出格式不匹配

## Takeaway

- `TF32 + 1e-4` 会低估 MusaCoder；关闭 TF32 后，pybind 一段式 correctness 从 **61.25%** 回到 **87.38%**
- correctness 应关闭 TF32、保持 `1e-4`；performance/profile 再恢复 PyTorch 默认 TF32 路径
- 反直觉：模型没有跨输出风格泛化，反而跨binding方法泛化
 

# 问题根源：精度口径对比


在 Ampere 及更新 GPU 上，PyTorch 对不同算子的默认策略并不一致：

| 路径 | PyTorch 默认 float32 计算策略 | 对 correctness 的影响 |
|---|---|---|
| `nn.Conv*` / cuDNN conv | 默认开启 TF32 | reference conv 可能是 TF32 近似结果 |
| `torch.mm` / `matmul` / `bmm` | 默认关闭 TF32 | reference matmul 通常是真 fp32 |

**注意：这里的 TF32/FP32 不是输入输出 tensor 的 `dtype`。这些 tensor dtype 仍然是 `torch.float32`；变化的是 PyTorch backend 对 float32 算子的内部计算精度策略**

## 1. 几种口径的结果

binding方法：MusaCoder 原生 pybind `load_inline` 一段式
100 题 × 8 samples = 800。

| ref | tolerance | correctness | compile | 说明 |
|---|---|---:|---:|---|
| - | - | **95.75%** | - | 官方报告精度 |
| TF32 | 1e-4 | **61.25%** | 98.12% | 旧 KernelGym/KernelBench 路径：conv reference 继承 PyTorch 默认 TF32 |
| TF32 | 1e-2 | **88.00%** | 98.12% | - |
| TF32 | 1e-3 | **87.75%** | 98.12% | - |
| FP32 | 1e-4 | **87.38%** | 98.25% | correctness 关闭 TF32，profile 前恢复默认 |

> KernelBench 一开始使用的是统一的 `atol=rtol=1e-2`。后来 PR #80 `Precision Support + TileLang Integration` 于 2025-11-05 merge，把 eval 从固定 `atol=rtol=1e-02` 改成按 precision 查 tolerance：fp32 使用 `1e-4`，fp16/bf16 使用 `1e-2`


原因分析：

1. 在 compute capacity >= 8.0 的 GPU 上，PyTorch/cuDNN conv 默认启用 TF32
2. 旧 correctness 路径没有关闭 TF32，`nn.Conv*` reference 实际是 TF32
3. TF32 conv reference 相对 fp32 的误差约为 `3e-4` 量级，已经超过 `1e-4`
4. 结果是：即使 custom kernel 语义正确，仍有概率被判成 output mismatch

## 2. 为什么不是简单地把 tolerance 放宽

正确做法不是永久使用 `1e-2`，而是把 correctness oracle 和 performance baseline 解耦：

| 评测目标 | reference |
|---|---|
| correctness | 强制关闭 TF32 |
| performance/profile | 恢复默认状态，conv 可走 TF32 |

`1e-2` 能掩盖 TF32 reference 的噪声，因此适合解释历史 KernelBench 结果；但它也会放过一部分真正的数值错误。更干净的方案是：**correctness 用 `FP32 + 1e-4`，profile 恢复默认 TF32 性能路径**。



# binding 方案对比（固定修正后 correctness 口径）

## 1. 三种 binding/格式

| 格式 | 输出结构 | 说明 |
|---|---|---|
| pybind 一段式 | 单个 Python code block，使用 `torch.utils.cpp_extension.load_inline` | MusaCoder 论文原生格式 |
| TVM-FFI 一段式 | 单个 Python code block，使用 `tvm_ffi.cpp.load_inline` | - |
| pybind 三段式 | `CUDA_KERNELS` + `APPLY_BINDINGS` + `MODEL_NEW` | - |

以下结果均使用修正后标准口径：

- correctness：`FP32 + 1e-4`。
- profile：恢复 PyTorch 默认状态。
- 100 题 × 8 samples = 800。

| metric | pybind 一段式 | TVM-FFI 一段式 | pybind 三段式 |
|---|---:|---:|---:|
| compile | **98.25%** | **90.38%** | **70.13%** |
| correctness | **87.38%** | **82.75%** | **59.50%** |
| problem any@8 correct | 99% | 100% | 97% |
| fast@1.0 | 13.12% | 13.50% | 11.88% |
| fast@1.2 | 9.50% | 9.75% | 8.38% |

MusaCoder 是按论文的 `load_inline` 一段式格式训练的。强行要求它输出三段式，会引入额外的 section 格式错误、binding 拼接错误和 wrapper 不匹配。这个损耗不是 CUDA kernel 能力本身造成的。

**相比不同的输出结构，反而不同的binding方法对模型的影响更小，这是意料之外的：模型没有跨输出风格泛化，反而跨binding方法泛化**


# 复现差距分析

<!-- **结论**：load_inline 一段式、单轮、FP32+1e-4 TF32-off 下，我们 L1 复现 Avg@8 = **87.38%**（官方现行放大shape口径），与官方 L1 报告的 ~95.75% 差约 8 点。目前能定量归因的只有一层：

1. **输入shape口径 ~+3 点**（已实证）：官方 2025-07 把 level1 输入整体放大；放大前的小shape下同一批 kernel 正确率高约 3 点（87.38% → 90.38%）。
2. **残余 ~5 点（未解释）**：即便最优的小shape单轮 90.38% 仍低于 ~95.75%，对应一批真实的模型实现错误（见"剩余失败解剖"），尚未定位单一主因。

口径前提（避免误判）：我们的数据集shape与官方 `main` 逐题一致，**不是我们放大的**；best-of-8 我们不输（Pass@8 99% ≥ 官方 Overall 93.2），差距只在 Avg@8 的逐样本一致性。注意"95.75% = 官方 L1 Avg@8"是推测（HF card 只给出 Overall 跨 L1/2/3 的 88.60），硬比前需核实。 -->

## 错题分布


| 失败类别 | 数量 | 机制 | shape引起? |
|---|---:|---|---|
| **illegal memory access** | **14** | **`int` 线性索引在 ≥2³¹ 元素处溢出** | **是** |
| invalid configuration | 9 | grid 维度超限（展平尾维 > 65535） | 是 |
| output_mismatch（reduction/loss/norm） | 29 | 大规模 reduction 朴素 fp32 累加漂移超 1e-4 | 混合 |
| output_mismatch（conv/matmul/pool） | 20 | 部分算法错、部分大 K 漂移 | 混合 |
| compile_fail | 14 | 真编译错 | 否 |
| binding_sig | 11 | wrapper 接口类型错 | 否 |
| py_runtime | 4 | wrapper 逻辑错 | 否 |

**其中 illegal_mem 14 + grid 9 = 23 个是纯shape引起的 int32 索引溢出与 grid 超限**——这类失败在小输入下不会出现（int 不溢出、grid 不超限）。
注意到[KernelBench v0.1](https://scalingintelligence.stanford.edu/blogs/kernelbenchv01/)版本才将测试输入tensor的shape放大（为了避免launch overhead对加速比测试的影响），因此怀疑MusaCoder使用旧版的shape进行的测试，如下表所示：

<!-- ## 输入shape效应：+3% -->

<!-- 新旧shape都用 `torch.rand`、其余口径相同（load_inline、TF32-off、1e-4、单轮、avg@8、800 样本）： -->

| 数据集 | correct avg@8 |
|---|---:|
| 新shape（官方现行，大输入） | 87.38% |
| **旧shape（放大前，小输入）** | **90.38%** |
| 官方报告 | 95.75% |

<!-- - **纯shape效应 = 缩小输入 +3.0 点**。官方放大shape确实压低正确率；官方 ~95.75% 口径未完全核实（见文末注释），硬比前需确认。
- 逐题：shape救回结构性失败——pid45 AvgPool2D 0→8（`int` 线性索引在 2³² 元素处溢出 → illegal memory access）、pid87 conv_pointwise 1→8、pid100 HingeLoss 1→8、pid95 CrossEntropy 3→7；小shape也引入个别 cumsum 形状回归（pid92/93）。

（数据集构建、randn 分布混淆弯路、原始失败枚举见文末注释。） -->

## 剩余错题分布（小shape，77 题）

逐一核对均为真实模型错（期望形状皆算子合法输出，非 harness 假失败）：

| 类别 | 数量 | 性质 |
|---|---:|---|
| shape_mismatch | 29 | 输出形状算错：conv_transpose 输出shape公式、loss reduce、cumsum_exclusive |
| output_mismatch | 31 | 真数值错（多为硬错，详见下方说明） |
| binding_sig | 7 | wrapper 接口类型错（conv / InstanceNorm / AvgPool1D） |
| compile_fail | 5 | 真编译错 |
| py_runtime / 越界 / grid | 5 | wrapper 逻辑、conv3D 越界、4D matmul grid |

- **数值错绝大多数是真算错、非精度**：31 个放宽到 1e-2（100× 容差）只翻正 3 个，28 个仍错。典型：**BatchNorm（诊断方向已更正，见下）**——评测 harness（官方 KernelBench `src/eval.py` 与 KernelGym fork 的 `correctness.py` 均**从不调用 `.eval()`**）在 **train 模式**下跑 reference，`nn.BatchNorm2d` 用的是**当前 batch 的 mean/var**（按 N,H,W 归约、有偏方差），不是 running buffer。所以正确实现必须现算 batch 统计量；模型的系统性错误恰恰**相反**：它读 `running_mean/running_var`（eval/推理语义）→ Output mismatch（pid33 7/8 错）。唯一做对的样本 idx_0256 正是用 `self.training=True` 分支现算 batch 统计量。（实证：ref vs train 公式误差 4.8e-7，ref vs eval 公式误差 1.7；手写 train-mode 一段式 kernel 经真实 KernelGym backend 打分 correct + 1.69× @ 小 shape 16×64×256×256。）注意：train-mode BatchNorm 在语义上不寻常（真实推理应是 eval/running-stats），这是 KernelBench 不调 `.eval()` 的约定，不是我们打分的 bug。其余真数值错：scan 族（cumsum/cumprod）语义实现错；转置 matmul 转置处理错。
<!-- - **错误结构**：约 46 个带明确可执行报错（shape 29 "expected X got Y" + compile 5 + binding 7 + runtime 5），属首轮即暴露、信息充分的类型；另有 BatchNorm（harness 跑 train 模式 → 须现算 batch 统计量，模型多误用 running stats，见正文更正）、loss 形状歧义；其余为纯能力硬错（scan 语义、转置 matmul ~10+）。 -->

<!-- > **评分坑（重要）**：live `--kernel-backend cuda_agent` 的 client 端 precheck/抽取是三段式专用，对 load_inline 单段式响应会判空 → reward=0、env_time≈0、reward server 全程空闲。load_inline 的分**必须靠对 dump 重打分**（`rescore_musacoder_dump.py --backend load_inline`），不能看 live eval 的 reward。详见 RUNTIME.md 的 PITFALL。 -->

<!--
复现差距分析——细节/弯路注释（不放进正文主线）：

A. 数据集与运行
- 三套数据集（均 load_inline 格式、同 `musa_coder.jinja` + `backend_display="pybind load_inline"` + 同 one-shot）：
  - 新shape（官方现行）：`Data/kernelbench-level1-validation-musa-coder-load-inline/`，rand。
  - 旧shape randn：`Data/...-oldsize/`，来源 KernelBench 放大提交 `b08d959`(2025-07-02) 的父提交 `21fbe5a`(2025-06-02)，100 题、小shape、原生 torch.randn。
  - 旧shape rand（干净shape消融）：`Data/...-oldsize-rand/`，对旧 reference 仅 `torch.randn(`→`torch.rand(`（sed，123 处，几何/randint/其它不变；全树 diff 无杂质；经 codex xhigh 复核 Valid-with-caveats）。
- 构建脚本：`tools/build_oldsize_raw.py`（旧 .py → raw parquet）+ `tools/convert_prompt_with_template.py`；旧版 clone 在 `../KernelBench-oldsize`（含 `level1_rand/`）。
- 运行：在 .22 生成 → dump `eval_0.pt`；在 .16 host-net 容器 `csl_slime_0614` 用 `rescore_musacoder_dump.py --backend load_inline`（TF32-off、1e-4、32 worker）打分。结果 JSON：`.../kb_l1_oldsize.../dumps/rescore_oldsize_tf32off_800.json`、`.../kb_l1_oldsize_rand.../dumps/rescore_oldsize_rand_tf32off_800.json`。
- 两题组成差异（matched-98 排除）：pid50（旧 Product_reduction ↔ 新 conv_standard_2D）、pid97（旧 CosineSimilarity ↔ 新 SDPA）；因题集编辑发生在shape放大之后，无单一 commit 同时具备"当前题集 + 小shape"。matched-98 与 all-100 几乎相同，故影响可忽略。

B. randn 分布混淆弯路（为何正文统一用 rand）
- 最初的旧shape数据集沿用放大前默认 `torch.randn`，结果 correct avg@8 = **83.50%**，反而低于新shape 87.38%——这是分布混淆，不是shape效应。
- 实证（per-sample error）：randn（均值 0、含负值）下大 K matmul/reduction 求和发生数值失配，pid2/6/16/18 由 8/8 跌到 0/8 `Output mismatch`（compiled=True）；换 rand（全正）后全部翻正。
- 另注：旧 randn 的 cumsum 回归（pid92/93）经核是形状错（`Output shape mismatch`），与分布无关。
- 因此干净的shape消融必须把分布固定为 rand（与官方现行一致）；正文三组对比即用 rand。KernelBench 官方也在 2025-07-21（commit 6b1cea2）把默认从多分布 `rand_mix` 改成普通 `torch.rand`（改回原因 commit/PR 无书面说明，不臆测）。

C. 大shape 101 失败分布的补充（正文"大shape下的失败分布"表已含主体）
- grid 超限例：pid36 RMSNorm 用 blockIdx.y 索引展平尾维 262144 > 65535。
- 来源 `rescore_tf32off_fp32ref_800.json` + 逐样本 review。
- 这批里约 52 个首轮带明确报错（compile/binding/runtime/部分 mismatch）。

D. 口径校正记录（避免重蹈）
- 95.75% 的口径未完全核实（L1 vs Overall、pass@k vs avg@k、评测协议），硬比前需确认；HF card 只给 Overall Avg@8 88.60。
- "我们放大了输入shape"被证伪：shape == 官方 main（pid45 16×64×2048×2048=2³²、pid87 1024×1024、pid100 batch=32768 均逐题一致）。
- "缩小shape把分数找回来"不是合法对齐——除非证明官方用的是放大前旧版（时间线上 2025-07 放大早于 2026-06 论文 ~11 月，官方大概率用放大shape）。
- 不要看 live eval 的 reward（client precheck 对 load_inline 判 0）；只认 dump 重打分。
-->

<!--
补充注释：以下信息来自详细版 handoff，仅用于后续复查，不放进正文主线。

1. KernelGym 实现与运行状态
- KernelGym worktree: `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-load-inline`
- branch: `feature/musacoder-load-inline`
- 关键实现：`kernelgym/backend/kernelbench/load_inline_backend.py`、`kernelgym/backend/kernelbench/dispatcher.py`、`kernelgym/toolkit/kernelbench/binding_detection.py`、`kernelgym/toolkit/kernelbench/load_inline_decoy.py`、`kernelgym/toolkit/kernelbench/correctness.py`
- 评分脚本：`scripts/rescore_musacoder_dump.py`、`scripts/score_one_sample.py`、`scripts/subdivide_mismatch.py`
- 设计文档：`docs/design-doc/MUSACODER_LOAD_INLINE_BACKEND.md`
- 状态：backend、decoy、correctness TF32-off/profile restore 已本地测试和 800-sample 重测；尚未 commit/merge，尚未部署到 `.21` 服务。

1. MusaCoder dump / extractor 注意事项
- 最终 pybind dump: `experiments/EvalFAsync.cuda_agent.MusaCoder-27B.CTX32768/MusaCoder-27B.kb_l1_musa_coder.load_inline.cuda_agent/dumps/rollout_data/eval_0.pt`
- dump 中 full response 可能含多个 python fence，因为 thinking 里会有 draft snippet；正确 extractor 应从 `</think>` 后提取，或取最后一个完整 ```python block，并去掉结尾 `<|im_end|>`。
- 最终 prompt 使用官方 KernelBench `add` kernel 的 single-block `ModelNew` one-shot；早期 two-block one-shot 与 “Output ONLY the code block” 不一致，已废弃。
- prompt 第一行曾有 `PyTorchinternals` 转录错误，已修复为 `PyTorch internals`；parquet 需要重新生成才会生效。
- parquet 转换使用 source field `ground_truth`，不是 `prompt`。

1. TVM-FFI one-piece API 约束
- `tvm_ffi.cpp.load_inline(functions=[...])` 会自动 export 函数，inline C++ 里不要再写 `TVM_FFI_DLL_EXPORT_TYPED_FUNC(...)`。
- `load_inline` 不接受 `verbose` 参数。
- 调用时需要 `with tvm_ffi.use_torch_stream():`。
- prompt 需要同时 `import tvm_ffi` 和 `from tvm_ffi import cpp as tvm_ffi_cpp`。
- 评分仍走 `backend=load_inline`，因为输出是单个 Python module；旧 `tvm_ffi` 后端期望 three-section 源格式，不适合这个 prompt。

1. 精度问题的实验证据
- 旧 pybind `TF32 + 1e-4`: `.19:/tmp/rescore_800.json`，490/800 correct。
- mismatch 分桶：`.19:/tmp/mismatch_subdiv.json`，257 个 output mismatch 中 214 个是 `near_miss<=1e-2`；其中 212 个在 1e-3 诊断重跑中通过。
- TF32-off near-miss 对照：`.19:/tmp/nearmiss_tf32off_1e4.json`，214 个 near-miss 中 208 个翻转为 correct，且 208 个全部是 conv；剩余 6 个是非 conv 的朴素大 reduction/HuberLoss 类误差。
- 完整 `FP32 + 1e-4` 重测 JSON：
  - pybind: `experiments/EvalFAsync.cuda_agent.MusaCoder-27B.CTX32768/MusaCoder-27B.kb_l1_musa_coder.load_inline.cuda_agent/dumps/rescore_tf32off_fp32ref_800.json`
  - TVM-FFI: `experiments/EvalFAsync.cuda_agent.MusaCoder-27B.CTX32768/MusaCoder-27B.kb_l1_musa_coder.tvm_ffi_inline.v2.cuda_agent/dumps/rescore_tf32off_fp32ref_800.json`
  - three-section: `experiments/EvalFAsync.cuda_agent.MusaCoder-27B.CTX32768/MusaCoder-27B.kb_l1_musa_coder.cuda_agent.fixedprompt/dumps/rescore_tf32off_fp32ref_800.json`

1. 比较 caveat
- 与论文或外部结果比较时，必须确认 KernelBench 版本/tolerance、PyTorch 版本、GPU/TF32 状态、pass@k vs sample avg@8、执行后端、problem set 版本。
- 历史 `TF32 + 1e-2` 是依据旧 mismatch 分桶推导的 correctness 口径，不是完整 profile 重跑；不能拿它的 fast@p 与完整重测直接比较。
- 早期 three-section 诊断结果：TF32-on `1e-4` 为 39.25%，TF32-on `1e-3` 为 59.62%；最终正文使用的是修正后 `FP32 + 1e-4` 的 59.50%。
-->

# one_shot_example 消融

## takeaway
**问题**： 前文中，prompt 里的 one-shot-example 是最简单的 elementwise_add。替换成更复杂/更难的example能否让模型更好呢？
**结论**： 尝试了两个one-shot-example，结果都没有改善，反而略有下降。复杂 one-shot 把失败从数值/形状错转移到语法错误

## 实验结果
尝试了两个 one-shot-examle：
1. Claude-Opus-4.8写的 **BatchNorm** 类 kernel；
2. MusaCoder写的，rollout8次，唯一一次正确的rollout结果， 题型：**conv_transposed**

下表为测试结果，由于从测试集中选取了one-shot-example，因此额外计算了泄漏题相关正确率
| one-shot | ALL correct（含泄漏） | NO-LEAK correct | 泄漏题自身正确率 |
|---|---|---|---|
| baseline（elementwise，不泄漏） | 90.38% | — | — |
| BatchNorm | 88.12% | 88.13% | 1/8→7/8 |
| conv_transposed | 89.50% | 89.90% | 1/8→4/8 |

<!-- （NO-LEAK 只对 ablation 行有意义：把该 one-shot 泄漏的那道题排除后的正确率；其 Δ 是对"同样排除该题后的 baseline"相减——两次该 baseline 恰好都 91.16%。baseline 自身 one-shot=elementwise 不泄漏，故不列 NO-LEAK。另：泄漏题被带到 7/8、4/8 也没把含泄漏 ALL 拉回 baseline（仍 <90.38%），净损失为真，非泄漏口径造成的假象。） -->


### 按领域 correct 精度

统一排除两道泄漏题（pid33/73），总共98题：

| one-shot| conv (33题) | matmul (18题) | reduce/norm/loss/pool (34题) | other（激活类）(13题) |
|---|---:|---:|---:|---:|
| baseline | **97.3%** | 96.5% | **81.2%** | 100% |
| BatchNorm | 90.2% | 93.8% | **80.9%** | 100% |
| conv_transposed | **92.4%** | 96.5% | 82.7% | 100% |

<!-- 读法：归约示例（BatchNorm）没抬高归约类（81.2%→80.9%，持平）；conv 示例（convT3d）没救回 conv（97.3%→**92.4%**，反降）。两套 one-shot 的损失都最集中在 conv（BatchNorm 甚至把 conv 砸得比 conv 示例自己还狠：90.2% < 92.4%），matmul/reduce/other 基本持平 → 没救回自家领域、损失以 conv 为主且与示例领域无关。

（题数：KernelBench L1 共 **100** 题，按算子分 conv 34 / matmul 18 / reduce-norm-loss-pool 35 / other 13——conv 本就是最大一类；表中 33/18/34/13 = 排掉 2 道泄漏题 pid33/73 后的 98 题。） -->

### 错误类型分布（ALL 800，含泄漏）

| 失败类型 | baseline | BatchNorm | conv_transposed |
|---|---:|---:|---:|
| output_mismatch（数值错） | 31 | 24 | 21 |
| shape_mismatch（形状错） | 29 | 26 | 25 |
| binding_sig（wrapper 接口签名错） | 7 | **15** | **21** |
| cuda_runtime（CUDA 越界 / launch 配置） | 3 | **11** | 4 |
| compile_fail | 5 | 8 | 5 |
| py_runtime（wrapper 逻辑错） | 2 | **11** | 8 |
| **合计 fails** | **77** | **95** | **84** |

### 结论
1. **one-shot example并没有提高同领域的算子精度**
2. **复杂 one-shot 把失败从数值/形状错转移到语法错误**：模型照抄了复杂示例那套多参数 wrapper 传参与 kernel-launch 写法，却把接口签名、参数解包、launch 配置搞错


<!-- 1. **"更复杂的 one-shot 更好"不成立**——两个都没改善，平凡 elementwise 反而是更稳的中性模板。强弱有别：BatchNorm 的 −3pp 较硬（≈3.5 SD），conv 版 −1.3pp 落在单次噪声内（z≈−0.86）→ **只能说这两个没改善，不能断言"更难必然有害"**。 -->
<!-- 2. **不是领域锚定**：示例没救回自己的领域（归约示例没抬高归约类、conv 示例没救回 conv）。净损失**以 conv 为主**，但"集中于 conv"作为机制偏强（仅 BatchNorm 硬、强回退不跨 run 重合）→ **需多 seed 才能坐实**。 -->
<!-- 3. **泄漏强度看可照抄性**：好抄的融合 kernel（BatchNorm）把正确性和 fast 写法都带走（1→7、抄到 ~1.68×）；难抄的复杂 conv 只带走部分且是慢实现（1→4、~0.007×、fast 无增益）。 -->

<!-- 复现/细节（不放主线）：
- 错误分类口径：not compiled→compile_fail；error 含 "shape mismatch"→shape_mismatch；"Output mismatch"→output_mismatch；"incompatible function arguments"→binding_sig；"CUDA error…"→cuda_runtime；其余 compiled-but-wrong→py_runtime。脚本即时从三个 rescore JSON 的 results[].{compiled,correctness,error} 统计。
- 三个数据集（同 raw、同 musa_coder.jinja、同 backend_display="pybind load_inline"）：baseline `…-oldsize-rand`、BatchNorm 版 `…-oldsize-rand-bn1shot`、conv 版 `…-oldsize-rand-convT3dshot`。rollout 在 .22：`SKIP_KERNELGYM_HEALTH=1` 跳过健康检查（cuda_agent precheck 拒 load_inline → live reward=0 是预期，靠事后 rescore）；rescore 在 .22：load_inline backend、`KERNELGYM_CORRECTNESS_DISABLE_TF32=1`、8 worker。EVAL_DATA 必须用绝对路径（Data/ 被 gitignore，ray 上传 working-dir 会丢相对路径）。
- 实验1 = 泄漏题 pid33（BatchNorm，小 shape 1/8 correct、0/8 fast）。动机串起"harness 跑 train 模式 → BN 须现算 batch 统计量"（见正文更正）；据此手写 train-mode 融合 kernel（双精度逐通道归约 + float4），真实 backend 打分 correct + 1.69×（16×64×256×256）。one-shot/自测：`Data/…-oldsize-rand/one_shot_example_load_inline_batchnorm.txt`、`test_batchnorm_oneshot.py`；rescore：`…bn1shot…/dumps/rescore_bn1shot_tf32off_800.json`。compile 99.37%→99.00%（几乎不变 → 不是格式崩，是 kernel 逻辑被带偏）。
- 实验2 = 泄漏题 pid73（`73_conv_transposed_3D…grouped`，winner idx581，correct 但 0.0072×）的通过 rollout 原文。one-shot：`Data/…-oldsize-rand/one_shot_example_load_inline_convT3d.txt`；rescore：`…convT3dshot…/dumps/rescore_convT3d_tf32off_800.json`。
- 逐类聚合（excl 泄漏题，conv/matmul 拆开，codex xhigh 复核重算）：BatchNorm conv −19 / matmul −4 / reduce −1 / other 0（合计 −24）；conv 版 conv −13 / matmul 0 / reduce +3 / other 0（合计 −10）。粗 SD：两者 conv/matmul 合计 −23≈3.5SD、−13≈2.3SD；reduce 类两者均在噪声内。
- 噪声判定：单次 rollout，逐题 Δ 含二项采样噪声（4/8 处每题 SD~2）。跨 run "至少 1 样本 conv 回退" 重合 10 vs 期望 6.6（hypergeom p≈0.037 弱显著），但 "≥2 样本强回退" 无跨 run 重合，最大 |Δ| 题也不集中在 conv。
- 早先曾误读 BatchNorm 结果为"领域锚定（归约类被抬高）"：挑了几道 +Δ 归约题、忽略同类 −Δ，实际归约类净 −1 持平；被实验2证伪。
- codex 复核记录：`/nfs/FM/chenshuailin/staging_oneshot_conv1x1/codex_review_ablation_out.txt`（数字 VERIFIED；早期 3a/3b OVERSTATED → 已按其建议 hedge）。
-->


<!-- 小 shape 三轮结果已注释（只展示大 shape 版本；此节保留备查）。

# 三轮 multi-turn 测试结果（load_inline live feedback）

## takeaway
**问题**：在单轮 baseline 之上，给模型 3 轮 live KernelGym feedback，能否提高最终可用 kernel 的正确率和速度？

**结论**：多轮显著提高 **best-by-turn** 指标，但 final-turn-only 会回落，说明模型常能在某一轮修好/优化，但继续优化时也会把正确性改崩。

口径：100 题 × 8 = 800 trajectories，最多 3 turns，分母统一 800。**Compile / Correct 为可信口径**；**Fast@ 为 live 打分（num_perf=100），与离线 baseline 仅作近似比较**。

| 指标 | T1 | T2 | T3 | Best-by-turn |
|---|---:|---:|---:|---:|
| Compile | 97.88% | 94.38% | 94.75% | 99.38% |
| Correct | 88.38% | 82.38% | 80.75% | 97.62% |
| Fast@1.0 | 20.75% | 24.25% | 26.62% | 32.25% |
| Fast@1.2 | 13.50% | 16.50% | 19.12% | 22.12% |

计数：

| 指标 | T1 | T2 | T3 | Best-by-turn |
|---|---:|---:|---:|---:|
| Compile | 783/800 | 755/800 | 758/800 | 795/800 |
| Correct | 707/800 | 659/800 | 646/800 | 781/800 |
| Fast@1.0 | 166/800 | 194/800 | 213/800 | 258/800 |
| Fast@1.2 | 108/800 | 132/800 | 153/800 | 177/800 |

对照单轮 baseline（同数据集 oldsize-rand，denom 800，离线 rescore）：

| 指标 | 单轮 baseline | 3-turn T1 | 3-turn Best |
|---|---:|---:|---:|
| Correct | 90.38% | 88.38% | 97.62% |
| Fast@1.0 | 22.75% | 20.75% | 32.25% |

要点：

1. **Correct（可比、可信）**：best-by-turn 把正确率从 baseline 90.38% 提到 **97.62%**；T1 88.38% 与 baseline 90.38% 基本持平（差距在采样波动内，无首轮退化）。
2. **Fast@（仅近似）**：best-by-turn 把 fast@1.0 从 22.75% 提到 32.25%，方向明确，数值不作严格结论。
3. **best-by-turn 含 pass@3 成分**：3 轮=3 次机会，+7pp 里"feedback 帮助"与"多采样"未隔离。
4. **final-turn 会回落**（T3 correct 80.75%）：模型在已正确后继续优化常把正确性改崩 → 真实使用须按 live reward 选 best turn，不能盲取最后一轮。

> 环境假失败源于 load_inline 并发 versioner bug（详见 `musacoder_multiturn_handoff.md` §5）；已重跑清除主签名 `.so cannot open`，残留约 5 个假失败（~0.6pp，已可忽略），故上表绝对数可直接采纳。

证据文件：

- 原始 v2 dump：`/nfs/FM/chenshuailin/projects/kernel_agents/slime-musacoder-mt/experiments/EvalMT3.load_inline.MusaCoder-27B.CTX40960/MusaCoder-27B.mt3_full_v2.load_inline/dumps/rollout_data/eval_0.pt`
- repaired 指标 summary：`/nfs/FM/chenshuailin/staging_oneshot_conv1x1/mt3_v2_repaired_metrics_summary.json`

小 shape 节注释结束。-->

<!--
.so cannot open 补充测试记录（注释，仅供复查）：

A. 原始 v2 不是完全 clean
- v2 配置：`SGLANG_MAX_RUNNING_REQUESTS=8`，2 engines，总 in-flight 约 16。
- 原始 v2 dump：2398 samples，800 trajectories；T1=800，T2=799，T3=799。
- 原始 live 指标被 `load_inline` 并发编译/导入 race 污染：`.so cannot open shared object file` 共 86 samples，84 groups。
- 分布：T1=20，T2=31，T3=35。
- 原始 raw v2 live：T1 correct 86.25%，T2 80.00%，T3 77.00%，Best correct 97.62%；Best fast@1.0 32.00%。
- 这些 `.so cannot open` 是环境假失败，不应当算模型真实 compile_fail；但它也会污染后续 turn 的 feedback，所以不能只做 offline rescore 了事。

B. 第一次补充重跑：84 个污染 trajectory，完整 3 turns
- 临时数据集：`/nfs/FM/chenshuailin/projects/kernel_agents/slime-dev-csl-2/Data/kernelbench-level1-validation-musa-coder-load-inline-oldsize-rand-mt3-v2-polluted84-rerun/train.parquet`
- 首先用 `SGLANG_MAX_RUNNING_REQUESTS=1` 跑，太慢（约 6/84 用 31min）。
- 用户要求改成 `SGLANG_MAX_RUNNING_REQUESTS=8` 后重启。
- sglang8 tag：`MusaCoder-27B.mt3_v2_polluted84_rerun.nokgfix.sglang8`
- sglang8 dump：`/nfs/FM/chenshuailin/projects/kernel_agents/slime-musacoder-mt/experiments/EvalMT3.load_inline.MusaCoder-27B.CTX40960/MusaCoder-27B.mt3_v2_polluted84_rerun.nokgfix.sglang8/dumps/rollout_data/eval_0.pt`
- sglang8 完成：84/84，耗时 1:01:22，250 samples；T1=84，T2=83，T3=83。
- sglang8 仍残留 `.so cannot open`：12 samples，11 groups；分布 T1=4，T2=6，T3=2。

C. 第二次补充重跑：sglang8 仍污染的 11 个 trajectory，完整 3 turns
- 临时数据集：`/nfs/FM/chenshuailin/projects/kernel_agents/slime-dev-csl-2/Data/kernelbench-level1-validation-musa-coder-load-inline-oldsize-rand-mt3-v2-polluted84-rerun-sglang8-polluted-rerun/train.parquet`
- 配置：`SGLANG_MAX_RUNNING_REQUESTS=4`。
- tag：`MusaCoder-27B.mt3_v2_sglang8_polluted11_rerun.nokgfix.sglang4`
- dump：`/nfs/FM/chenshuailin/projects/kernel_agents/slime-musacoder-mt/experiments/EvalMT3.load_inline.MusaCoder-27B.CTX40960/MusaCoder-27B.mt3_v2_sglang8_polluted11_rerun.nokgfix.sglang4/dumps/rollout_data/eval_0.pt`
- 完成：11/11，耗时 16:51，33 samples；T1=11，T2=11，T3=11。
- `.so cannot open`：0；missing env_result：0。

D. 合并规则
- 以原始 v2 full dump 的 800 groups 为底。
- 将原始 v2 中 84 个 `.so cannot open` 污染 groups 替换为 sglang8 84-subset 的对应完整 trajectory。
- 将 sglang8 中仍污染的 11 groups 再替换为 sglang4 11-subset 的对应完整 trajectory。
- 合并后 repaired full v2：`.so cannot open` = 0 samples / 0 groups。
- 仍有两个 trajectory 不满 3 turns：原始 group 484 只有 T1；替换后 group 614 只有 T1。因此 T2/T3 present=798/800。

E. 运行中遇到的基础设施问题
- sglang1 -> sglang8 重启时，旧 sglang scheduler orphan 占用每 GPU 约 70GB，导致新 sglang8 OOM；清理 `.22` 上 orphan SGLang/Ray GPU 进程后重启成功。
- Ray 有时 `Failed to get node info ... Deadline Exceeded`；处理方式是 `ray stop --force` 后清残留 GCS/raylet/dashboard 子进程，并在必要时换 `RAY_TEMP_DIR`。
- Gloo 自动 fallback 到不存在的 `bond0` 会失败；显式设置 `LOCAL_GLOO_SOCKET_IFNAME=ens22f0np0` 后启动成功。
- 这些操作均未停止 `.21` KernelGym reward 服务。
-->

# 三轮 multi-turn 测试结果（**大 shape**）

多轮**显著抬高 best-by-turn**——correct 从 T1 86.50% 提到 **96.50%**。但per-turn的精度返回有所下降，第二轮和第三轮的精度差不多，均不如第一轮


| 指标 | T1 | T2 | T3 | Best-by-turn |
|---|---:|---:|---:|---:|
| Compile | 97.38% | 93.38% | 94.38% | 99.12% |
| **Correct** | **86.50%** | 75.25% | 75.25% | **96.50%** |
| Fast@1.0 | 14.62% | 18.25% | 18.38% | 25.00% |
| Fast@1.2 | 11.00% | 13.75% | 13.00% | 17.12% |

<!-- 要点：
1. **多轮把 best-by-turn 抬到 96.50%**（vs T1 86.50%，+10 点）——模型能在 3 轮里某一轮把 kernel 修好/优化。注意含 pass@3 成分（3 轮=3 次机会），部分增益来自多采样而非纯 feedback。
2. **final-turn 会回落**：T3 correct 75.25% < T1 86.50%——模型在已正确后继续优化常把正确性改崩 → 真实使用须按 live reward 选 best turn，不能盲取最后一轮。
3. **fast 也随多轮上升**：fast@1.0 14.62%(T1)→25.00%(best)、fast@1.2 11.00%→17.12%。
4. **T1 单轮基线**：clean T1 correct 86.50% ≈ 单轮 baseline 87.38%（同口径），说明首轮即接近单轮水平、多轮增益主要体现在 best-by-turn。
5. **失败以真实模型错为主**：illegal-memory（int32 索引溢出等）仅少数（合并后 3 条），被 per-task fresh process 隔离，非 infra 假象。 -->

<!-- 证据：merged dump `…/MusaCoder-27B.mt3_largeshape_nosplit.load_inline/dumps/rollout_data/eval_0_merged_clean.pt`（已 normalize group_id/index，800 组）；re-run dump `…mt3_largeshape_rerun179…/eval_0.pt`（cannot-open=0 实证）。 -->

## best-by-turn 仍失败的样本分析（28/800）

best-by-turn correct=96.50% → **28 条 trajectory 3 轮都没做对**，且**只集中在 9 道题**。代表轮取每条「最后一个能编译的 turn」（无则记 compile_fail）来分类：

| 失败类别 | 数量 | 说明 |
|---|---:|---|
| output_mismatch | 14 | - |
| illegal_memory | 7 | - |
| compile_fail | 7 | 3 轮都没编译过 |

按题（9 题，每题含其失败的 sample 数）：

| pid | 题目 | 失败数 | 主类别 / 机制 |
|---|---|---:|---|
| 73 | conv_transposed_3D（非对称） | 7 | output_mismatch |
| 100 | HingeLoss | 7 | 5 illegal_memory（int32 溢出）+ 2 output_mismatch |
| 68 | conv_transposed_3D（方形） | 5 | compile_fail |
| 45 | AvgPool2D | 2 | illegal_memory |
| 4 | Matrix-vector mult | 2 | output_mismatch |
| 97 | ScaledDotProductAttention | 2 | compile_fail |
| 89 | cumsum | 1 | output_mismatch |
| 96 | HuberLoss | 1 | output_mismatch |
| 98 | KLDivLoss | 1 | output_mismatch |

<!-- 要点：
- **失败高度集中**：28 条只来自 9 题；**conv_transposed_3D（pid68+73 共 12 条）、HingeLoss（7）** 是 MusaCoder 3 轮都搞不定的核心难点。
- **多轮 feedback 对"真算法/数值硬错"帮助有限**：50%（14 条）是编译通过但结果错，3 轮迭代没能修正（转置卷积语义、scan、loss reduce 等）。
- **25%（7 条）是大 shape 的 int32 索引溢出**（pid45/100），属实现缺陷、非 infra 假失败（已与 cannot-open 区分）。
- **25%（7 条）3 轮都没编译过**（pid68/97）。 -->
