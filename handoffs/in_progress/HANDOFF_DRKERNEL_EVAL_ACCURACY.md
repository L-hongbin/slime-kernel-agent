# DrKernel KernelBench Level1 多轮 Eval 精度汇总

生成于 2026-05-25。覆盖 `checkpoints/Qwen3.5-9B/` 和 `checkpoints/Qwen3.6-27B/` 下所有有效 eval dump（`dumps/rollout_data/eval_0.pt`，n ≥ 100，metadata.turns 非空）。

## 指标定义

每一列都是**比率 = 命中样本数 / 总样本数 n**（不是 per-turn-available 的子集）。具体：

- **Compile T{t}**：第 t 轮 `kernelgym.compiled == True` 的样本数 / n
- **Correct T{t}**：第 t 轮 `compiled AND correctness AND NOT decoy_kernel` 的样本数 / n
- **Fast@1.0 T{t}**：第 t 轮 `compiled AND correct AND NOT decoy AND speedup ≥ 1.0` 的样本数 / n （**in_all**：分母是 total，不是 correct）
- **Fast@1.2 T{t}**：同上，阈值 1.2x

所有 fast@x 都是 **in_all** 口径，与 LHB DIR1 的 fast_in_all 一致；不是 fast_in_correct。

## 元数据列含义

- **First-turn template**：first-turn 时实际使用的 `slime_plugins/drkernel/prompt_templates/backends/*.jinja` 文件，从 rendered prompt 内容指纹检测出来（不是从 yaml 配置抽，因为 yaml 中 backend id 跨版本不变；具体文件是在不同时间替换的）
- **Env GPU/NVCC**：first-turn prompt 顶部是否含 `Target environment:` block（注入了 KernelGYM compile host 的 GPU="NVIDIA GeForce RTX 4090 (SM 8.9, Ada Lovelace)" + NVCC="CUDA 12.9 (nvcc, targeting sm_89)"）。Y = 注入，N = 不注入

## 模板演化

| 文件 | 简介 |
|---|---|
| `tvm_ffi_module.jinja` (v1) | LHB byte-identical baseline。30+ 条 verbose guidance，含 "Use TVM-FFI, not pybind11" + 完整的 `TVM_FFI_ICHECK` example pattern (7 行 ICHECK + `input.numel()` rank-agnostic) |
| `tvm_ffi_module_v2.jinja` | 加 `Common accessors:` API doc + `CUB or Thrust` allow + numel caveat。删 v1 verbose 复述部分 |
| `tvm_ffi_module_v2_1.jinja` | v2 减 CUB/Thrust，保留 accessor doc |
| `tvm_ffi_module_v2_2.jinja` | 纯 cleanup of v1，无新增。删 5 类冗余（"no PyTorch ops" 重复、launcher-signature restatement、negative API list、f32-specific ICHECK、4 句 MODEL_NEW 参数保留） |
| `tvm_ffi_module_v2_3.jinja` | 等价于 v2_2，只删掉了文件末尾泄漏的 jinja 注释 |
| `tvm_ffi_module_v2_4.jinja` | 基于 v2_3 的 reword/merge/trim，删通用 TVM-FFI/CUDA 知识 + 重复 bullets + rationale 文字，净 -728 chars。example 块不变。**尚未测试**（pending：9B + 27B 各 1 次 100×8） |

注意：v2_3 的 C++ example 比 v1 简化了——只剩 1 行 device ICHECK + 用 1D-only `input.shape()[0]`（而非 rank-agnostic `input.numel()`）。这是后面 9B 显著退化的根因（见 Phase 1+2 实验日志中 9B 分析）。

---

### Qwen3.5-9B

| Run | n | First-turn template | Env GPU/NVCC | Compile T1 | Compile T2 | Compile T3 | Correct T1 | Correct T2 | Correct T3 | Fast@1.0 T1 | Fast@1.0 T2 | Fast@1.0 T3 | Fast@1.2 T1 | Fast@1.2 T2 | Fast@1.2 T3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260522_134217_ctx65536_n8 | 800 | tvm_ffi_module.jinja (v1) | N | 5.2% | 10.1% | 19.5% | 2.6% | 3.5% | 5.5% | 2.4% | 2.8% | 4.6% | 0.0% | 0.0% | 0.0% |
| 20260524_235411_ctx32768_n8_v2_3_noenv_9b_n8 | 800 | tvm_ffi_module_v2_3.jinja | N | 0.5% | 3.8% | 11.4% | 0.0% | 1.1% | 2.0% | 0.0% | 1.0% | 1.1% | 0.0% | 0.0% | 0.0% |

#### 9B 观察

- 同 n=800, 同 noenv 下：v2_3 比 v1 **每一列都差**（compile T3 19.5% → 11.4%，correct T3 5.5% → 2.0%，fast@1.0 T3 4.6% → 1.1%）
- 与 27B Phase 1 结论"v2_3 noenv 是最佳"相反 — **template 效果是模型规模依赖的**
- 退化机制（已通过错误分类 + 响应模式分析验证）：v2_3 的 C++ example 简化成 1D-only 后，9B 倾向 verbatim 复制 example，对多维 op 写出错误的 size 参数；同时删掉的 6 个 ICHECK pattern 让 9B 不知道怎么写 op-specific validation，`icheck_use_issue` 错误率从 27.4% 上升到 34.9%

---

### Qwen3.6-27B

| Run | n | First-turn template | Env GPU/NVCC | Compile T1 | Compile T2 | Compile T3 | Correct T1 | Correct T2 | Correct T3 | Fast@1.0 T1 | Fast@1.0 T2 | Fast@1.0 T3 | Fast@1.2 T1 | Fast@1.2 T2 | Fast@1.2 T3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 20260523_001207_ctx65536_n8 | 800 | tvm_ffi_module.jinja (v1) | N | 44.6% | 46.6% | 47.2% | 27.1% | 27.6% | 30.0% | 12.6% | 11.8% | 12.2% | 0.6% | 1.1% | 1.4% |
| 20260523_054450_ctx65536_n8_summ800 | 800 | tvm_ffi_module.jinja (v1) | N | 44.1% | 44.8% | 45.4% | 27.6% | 26.2% | 26.2% | 13.8% | 10.9% | 11.0% | 0.9% | 1.2% | 1.0% |
| 20260523_072056_ctx65536_n8_summ1600 | 800 | tvm_ffi_module.jinja (v1) | N | 43.8% | 48.9% | 47.9% | 28.0% | 29.9% | 27.6% | 14.1% | 9.9% | 9.8% | 0.8% | 0.8% | 1.0% |
| 20260523_090038_ctx65536_n8_summ3200 | 800 | tvm_ffi_module.jinja (v1) | N | 41.2% | 44.4% | 45.5% | 26.6% | 26.2% | 27.3% | 14.2% | 9.6% | 10.2% | 0.6% | 0.8% | 1.6% |
| 20260524_142451_ctx65536_n8_summ1600_v2_3_env_n8 | 800 | tvm_ffi_module_v2_3.jinja | Y | 35.0% | 48.8% | 51.4% | 16.6% | 24.2% | 27.9% | 6.0% | 7.6% | 9.0% | 1.0% | 1.0% | 1.4% |
| 20260524_154851_ctx65536_n8_summ1600_v2_3_noenv_n8 | 800 | tvm_ffi_module_v2_3.jinja | N | 35.4% | 49.9% | 50.1% | 20.1% | 29.6% | 29.4% | 9.4% | 12.1% | 11.0% | 1.1% | 2.0% | 1.6% |

#### 27B 观察

- **v1 (LHB byte-identical) vs v2_3 noenv n=800**：fast@1.0 / fast@1.2 持平到略升（fast@1.2 T3 1.4% → 1.6%）；correct 持平（30% → 29.4%）。Paired McNemar (n=800) 显示 T1 fast@1.0 v2_3 noenv 显著更好 (p=0.0045)
- **env block 影响**：env_n8 vs noenv_n8 paired McNemar，env 始终略差（n=400 时 reward p=0.0029，n=800 收窄；T1 fast@1.0 始终显著差 p=0.0045）。Mechanism：env block 触发 hardware cargo-culting（692/800 提 sm_89），但不转化为更高 fast@x
- **fast@1.2 T3 的天花板 ~1.5-1.6%** 在所有 v2_3 variants 上一致，与 LHB DIR1 报告的 6.6% 仍有显著 gap。这是 model + training-data 限制（模型在 tvm_ffi 后端从未写出激进优化），不是模板问题
- v2 / v2_1 / v2_2 中间版本：加了 accessor doc / numel caveat 等"额外解释"，反而 compile T1 略降到 30%-36% 区间（vs v1 的 40%+ 和 v2_3 的 35%+）

---

## 跨模型小结

| 维度 | 27B | 9B |
|---|---|---|
| v1 → v2_3 effect | T1 fast@1.0 显著改善 (paired p=0.0045) | 全线下降 ~11x |
| env block on/off | env 略差（cargo-cult 但无收益） | 还没测；不打算测（9B 太弱） |
| 推荐 default | v2_3 + noenv | 暂时**继续用 v1**，或者出 v2_4（基于 v2_3 但把 C++ example ICHECK 块恢复到 v1 完整版本） |

## 可选下一步

1. **v2_4 ablation**：以 v2_3 为基础，只把 C++ example 部分换成 v1 那段（含完整 7 ICHECK + `input.numel()`）。预测：9B 回到 v1 水平甚至更好（其它 cleanup 仍有效），27B 在 fast@x 上不太可能退步（context 长几百 chars，noise ≈ 不变）。验证成本：9B + 27B 各 1 次 100×8 ≈ 1.5h
2. **per-model 模板选择**：在 `rollout.py` profile 上加 model-size 维度，9B 走 v1，27B 走 v2_3。短期省事但长期维护成本高
3. **训练优先级**：如果短期只训 27B，保持 v2_3 default 即可；如果会训 9B，必须先解决退化问题

## 排除的 dump

只保留 n=800（n_samples_per_eval_prompt=8 × 100 prompts），剔除了：

- 9B：`20260522_092718/` (n=2 smoke)、`20260522_124300/` (n=1)
- 27B：所有 `_n1_*` (n=100, n_per_prompt=1) 和 `_n4_*` (n=400, n_per_prompt=4) 中间 ablation 运行（共 14 个）；`20260522_061122/`、`20260522_064910/` metadata 缺失

## 数据来源

所有数字直接从 `eval_0.pt` 的 `samples[*].metadata.turns[t].kernelgym` 计算。生成脚本：`/tmp/build_handoff.py` on .22（保存一份在 `scripts/drkernel/` 之外的临时位置；下次重跑可参照本文件第 38-65 行的"指标定义"和 yaml 中 `chosen_prompt_slots` 字段自行实现）。
