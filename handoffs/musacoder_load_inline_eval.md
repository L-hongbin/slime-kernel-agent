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

<!--
补充注释：以下信息来自详细版 handoff，仅用于后续复查，不放进正文主线。

1. KernelGym 实现与运行状态
- KernelGym worktree: `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-load-inline`
- branch: `feature/musacoder-load-inline`
- 关键实现：`kernelgym/backend/kernelbench/load_inline_backend.py`、`kernelgym/backend/kernelbench/dispatcher.py`、`kernelgym/toolkit/kernelbench/binding_detection.py`、`kernelgym/toolkit/kernelbench/load_inline_decoy.py`、`kernelgym/toolkit/kernelbench/correctness.py`
- 评分脚本：`scripts/rescore_musacoder_dump.py`、`scripts/score_one_sample.py`、`scripts/subdivide_mismatch.py`
- 设计文档：`docs/design-doc/MUSACODER_LOAD_INLINE_BACKEND.md`
- 状态：backend、decoy、correctness TF32-off/profile restore 已本地测试和 800-sample 重测；尚未 commit/merge，尚未部署到 `.21` 服务。

2. MusaCoder dump / extractor 注意事项
- 最终 pybind dump: `experiments/EvalFAsync.cuda_agent.MusaCoder-27B.CTX32768/MusaCoder-27B.kb_l1_musa_coder.load_inline.cuda_agent/dumps/rollout_data/eval_0.pt`
- dump 中 full response 可能含多个 python fence，因为 thinking 里会有 draft snippet；正确 extractor 应从 `</think>` 后提取，或取最后一个完整 ```python block，并去掉结尾 `<|im_end|>`。
- 最终 prompt 使用官方 KernelBench `add` kernel 的 single-block `ModelNew` one-shot；早期 two-block one-shot 与 “Output ONLY the code block” 不一致，已废弃。
- prompt 第一行曾有 `PyTorchinternals` 转录错误，已修复为 `PyTorch internals`；parquet 需要重新生成才会生效。
- parquet 转换使用 source field `ground_truth`，不是 `prompt`。

3. TVM-FFI one-piece API 约束
- `tvm_ffi.cpp.load_inline(functions=[...])` 会自动 export 函数，inline C++ 里不要再写 `TVM_FFI_DLL_EXPORT_TYPED_FUNC(...)`。
- `load_inline` 不接受 `verbose` 参数。
- 调用时需要 `with tvm_ffi.use_torch_stream():`。
- prompt 需要同时 `import tvm_ffi` 和 `from tvm_ffi import cpp as tvm_ffi_cpp`。
- 评分仍走 `backend=load_inline`，因为输出是单个 Python module；旧 `tvm_ffi` 后端期望 three-section 源格式，不适合这个 prompt。

4. 精度问题的实验证据
- 旧 pybind `TF32 + 1e-4`: `.19:/tmp/rescore_800.json`，490/800 correct。
- mismatch 分桶：`.19:/tmp/mismatch_subdiv.json`，257 个 output mismatch 中 214 个是 `near_miss<=1e-2`；其中 212 个在 1e-3 诊断重跑中通过。
- TF32-off near-miss 对照：`.19:/tmp/nearmiss_tf32off_1e4.json`，214 个 near-miss 中 208 个翻转为 correct，且 208 个全部是 conv；剩余 6 个是非 conv 的朴素大 reduction/HuberLoss 类误差。
- 完整 `FP32 + 1e-4` 重测 JSON：
  - pybind: `experiments/EvalFAsync.cuda_agent.MusaCoder-27B.CTX32768/MusaCoder-27B.kb_l1_musa_coder.load_inline.cuda_agent/dumps/rescore_tf32off_fp32ref_800.json`
  - TVM-FFI: `experiments/EvalFAsync.cuda_agent.MusaCoder-27B.CTX32768/MusaCoder-27B.kb_l1_musa_coder.tvm_ffi_inline.v2.cuda_agent/dumps/rescore_tf32off_fp32ref_800.json`
  - three-section: `experiments/EvalFAsync.cuda_agent.MusaCoder-27B.CTX32768/MusaCoder-27B.kb_l1_musa_coder.cuda_agent.fixedprompt/dumps/rescore_tf32off_fp32ref_800.json`

5. 比较 caveat
- 与论文或外部结果比较时，必须确认 KernelBench 版本/tolerance、PyTorch 版本、GPU/TF32 状态、pass@k vs sample avg@8、执行后端、problem set 版本。
- 历史 `TF32 + 1e-2` 是依据旧 mismatch 分桶推导的 correctness 口径，不是完整 profile 重跑；不能拿它的 fast@p 与完整重测直接比较。
- 早期 three-section 诊断结果：TF32-on `1e-4` 为 39.25%，TF32-on `1e-3` 为 59.62%；最终正文使用的是修正后 `FP32 + 1e-4` 的 59.50%。
-->
