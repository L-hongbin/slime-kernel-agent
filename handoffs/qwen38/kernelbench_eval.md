# Qwen3.8 KernelBench L1–L3 评测

## FastCredit025 step100：64K / 128K / 192K

本轮 6000 轮记录的预算截断率为 **0.00%**。相对 32K/48K/64K，L3 Best Correct 从 51.50% 提高到 59.50%，第三轮 Correct 从 31.50% 提高到 41.25%；L1/L2 Best Correct 则分别变化 −1.00 / −0.25 个百分点，Fast 指标没有全面提升。这些是同一 checkpoint 的单次评测点估计，不据此确定训练目标的根因

| 三轮总 context | 全部轮次截断（%） |
|---|---:|
| 24K/32K/40K | 25.57 |
| 32K/48K/64K | 7.30 |
| 64K/128K/192K | 0.00 |

本轮只测试 `Qwen3.8-27B-FastCredit025-Step100`，完成 100 次更新的 `iter_0000099`；该 checkpoint 训练时的来源 bonus 仅在正确且实测 speedup ≥ 1 时生效，系数 0.25、无最终回报封顶，评测本身不分配 component reward。没有重测 TRLOO 或恢复训练。2000 条轨迹、6000 轮反馈完整，独立 raw env_state 计数与维护 summarizer 一致，作业成功结束，耗时 154.95 分钟；收尾记录确认本次使用的 24 张 H20 已释放

### 64K / 128K / 192K 各轮质量

表格从原始计数统一四舍五入到两位小数。分母 L1/L2/L3 为 800/800/400 条轨迹；每题 8 条三轮轨迹，所有任务级失败保留在分母。Best 为每条轨迹任一轮达到各自指标，不是 pass@8 或最后一轮指标

#### L1

| 轮次 | Compile（%） | Correct（%） | Fast@1.0（%） | Fast@1.2（%） |
|---|---:|---:|---:|---:|
| T1 | 96.63 | 89.38 | 23.25 | 19.00 |
| T2 | 93.13 | 82.63 | 35.88 | 24.00 |
| T3 | 89.88 | 81.13 | 37.88 | 27.13 |
| Best | 100.00 | 98.50 | 44.63 | 28.63 |

| 轮次 | 截断（%） | 未关闭 thinking（%） | Thinking 中位 token | Answer 中位 token | Prompt 中位 token | 可生成预算中位 token | 剩余未用预算中位 token |
|---|---:|---:|---:|---:|---:|---:|---:|
| T1 | 0.00 | 0.13 | 6587.50 | 1345.50 | 1357.00 | 64179.00 | 56097.00 |
| T2 | 0.00 | 1.88 | 8174.00 | 1517.50 | 9767.00 | 121305.00 | 111123.50 |
| T3 | 0.00 | 0.75 | 6511.50 | 1669.00 | 20304.50 | 176299.50 | 165363.00 |

#### L2

| 轮次 | Compile（%） | Correct（%） | Fast@1.0（%） | Fast@1.2（%） |
|---|---:|---:|---:|---:|
| T1 | 92.88 | 64.50 | 10.13 | 9.25 |
| T2 | 93.13 | 68.00 | 21.38 | 13.25 |
| T3 | 89.00 | 65.75 | 26.75 | 17.50 |
| Best | 100.00 | 83.50 | 30.63 | 18.50 |

| 轮次 | 截断（%） | 未关闭 thinking（%） | Thinking 中位 token | Answer 中位 token | Prompt 中位 token | 可生成预算中位 token | 剩余未用预算中位 token |
|---|---:|---:|---:|---:|---:|---:|---:|
| T1 | 0.00 | 0.13 | 10856.00 | 1967.50 | 1397.50 | 64138.50 | 51102.00 |
| T2 | 0.00 | 1.25 | 7856.50 | 2094.00 | 14778.00 | 116294.00 | 105374.00 |
| T3 | 0.00 | 1.00 | 9376.50 | 2207.50 | 26050.50 | 170553.50 | 157811.00 |

#### L3

| 轮次 | Compile（%） | Correct（%） | Fast@1.0（%） | Fast@1.2（%） |
|---|---:|---:|---:|---:|
| T1 | 71.75 | 31.50 | 3.00 | 2.50 |
| T2 | 78.00 | 39.25 | 4.50 | 3.50 |
| T3 | 81.00 | 41.25 | 7.75 | 5.25 |
| Best | 94.50 | 59.50 | 8.50 | 6.25 |

| 轮次 | 截断（%） | 未关闭 thinking（%） | Thinking 中位 token | Answer 中位 token | Prompt 中位 token | 可生成预算中位 token | 剩余未用预算中位 token |
|---|---:|---:|---:|---:|---:|---:|---:|
| T1 | 0.00 | 5.00 | 21403.00 | 3993.50 | 1922.50 | 63613.50 | 37754.50 |
| T2 | 0.00 | 1.50 | 8604.50 | 4393.50 | 28172.50 | 102899.50 | 86454.50 |
| T3 | 0.00 | 0.25 | 9352.00 | 4811.50 | 44987.50 | 151616.50 | 135649.00 |

### FastCredit025 三套预算对照

| Level | 三轮总 context | Best Correct（%） | Best Fast@1.0（%） | T3 Correct（%） | 全部轮次截断（%） |
|---|---|---:|---:|---:|---:|
| L1 | 24K/32K/40K | 97.88 | 42.38 | 67.13 | 14.79 |
| L1 | 32K/48K/64K | 99.50 | 45.25 | 79.75 | 3.33 |
| L1 | 64K/128K/192K | 98.50 | 44.63 | 81.13 | 0.00 |
| L2 | 24K/32K/40K | 81.25 | 25.50 | 49.00 | 19.38 |
| L2 | 32K/48K/64K | 83.75 | 29.88 | 61.38 | 3.83 |
| L2 | 64K/128K/192K | 83.50 | 30.63 | 65.75 | 0.00 |
| L3 | 24K/32K/40K | 40.00 | 4.75 | 21.75 | 59.50 |
| L3 | 32K/48K/64K | 51.50 | 7.50 | 31.50 | 22.17 |
| L3 | 64K/128K/192K | 59.50 | 8.50 | 41.25 | 0.00 |

### 实际预算与输出边界

65536 / 131072 / 196608 是包含 prompt 与保留历史的三轮总 context，并非每轮独立输出额度。server、rollout、eval 的全局 context/response cap 均为 196608；MTP 在全局边界保留 4 tokens，实际三轮有效上限为 65536 / 131072 / 196604。全部 6000 轮均核验实际 prompt、请求 `max_new_tokens` 与 response tokens，未发现客户端仍受旧 64K/128K cap 限制，见[逐轮预算与长度审计](../../local_artifacts/qwen38/fastcredit_step100_context64k128k192k_20260914/fastcredit025/context_length_audit.json)

本轮三个 turn 的最大实际总长度分别为 55323 / 83862 / 106045，最少剩余请求预算分别为 10213 / 47210 / 90559 tokens。301 轮总长度超过 64K，没有一轮超过 128K；因此本轮消除了已观测的预算截断，但没有证明第三轮 192K 是必需条件。原生 192K 服务边界另由 196588 输入加 16 输出的 MTP/CUDA Graph 诊断验证，见[实际使用汇总](../../local_artifacts/qwen38/fastcredit_step100_context64k128k192k_20260914/observed_budget_summary.json)与[边界诊断](../../local_artifacts/qwen38/fastcredit_step100_context64k128k192k_20260914/boundary_smoke.json)

Thinking/answer 仍按实际响应 token IDs 的首个 `</think>` 分界，排除 special tokens；未关闭时全部计入 thinking，answer 计为 0。表格为包含失败轮次的中位数，完整均值与 P90 保存在长度审计。本轮 68 轮未关闭 thinking，均为 completed；“未关闭 thinking”“预算截断”“代码未完整交付”分别统计，不能混用

L3 首轮 thinking 中位长度与上一套预算接近（21403.00 对 21601.50 tokens），answer 中位长度则从 2647.00 增至 3993.50 tokens。预算放宽后能看到更多完整代码和修复尝试，但完成生成仍不保证正确或更快

### 代表性输出与任务级失败

人工复核 9 条不同轨迹的三轮响应结构、前后文、关键代码和反馈，共 27 轮；完整输出与逐例记录见[人工复核](../../local_artifacts/qwen38/fastcredit_step100_context64k128k192k_20260914/fastcredit025/manual_review.md)

- L1 group360：首轮候选因 FP16 降精度被原 precheck 拒绝；后两轮正确但慢。第三轮总 context 92437，充分输出并不等于达到性能目标
- L2 group723：三轮都正确，第三轮输出 39427 tokens，但三轮 speedup 都小于 1；继续长时间优化推导没有带来达标结果
- L3 group225（Swin MLP）：首轮数值错误，后两轮正确但慢。第三轮 prompt 83248 加输出 22797，达到本轮最大实际总 context 106045
- L3 group79：首轮已经关闭 thinking，却在 CUDA 实现中途以 `<|im_end|>` 结束，缺少完整 ModelNew；输出 29604 / 请求 63289，仍有 33685 tokens 未用。模型下一轮自称“truncated”不代表预算截断，实际保存状态与 token 证据显示是在有余量时结束，见[尾部 token 核对](../../local_artifacts/qwen38/fastcredit_step100_context64k128k192k_20260914/fastcredit025/manual_token_end_audit.json)

| Level | WorkerProcessCrashed 次数 |
|---|---:|
| L1 | 1 |
| L2 | 7 |
| L3 | 5 |

共 13 轮 worker crash，全部保留正常分母，没有选择性删除或重测。L1 group655 的 host binding 直接读取在 GPU 上创建的 meta；L2 group18 的 host binding 解引用 GPU scalar 指针，与各自 native 错误栈吻合。后者改为 GPU kernel 内读取后正确，前者消除 crash 后仍触发原 decoy 检查；不能把“没有 crash”直接视为 Correct。源码片段见[候选代码复核](../../local_artifacts/qwen38/fastcredit_step100_context64k128k192k_20260914/fastcredit025/worker_crash_source_review.json)，全量错误见[worker crash 审计](../../local_artifacts/qwen38/fastcredit_step100_context64k128k192k_20260914/fastcredit025/worker_crash_audit.json)。未做独立 replay，其余 crash 的根因尚未逐项闭合

### 运行配置与必要调整

复用上一轮已修复的冻结 `runtime_repo`，生成/反馈/计分代码不变。模型与 tokenizer 原生上限均为 262144，没有新增 RoPE scaling、修改权重、换精度或 backend。保持 canonical KernelBench L1/L2/L3、每题 8 条轨迹、三轮、BF16、TP4、原生 MTP3、CUDA Graph、FA3/Triton、temperature 1、top-p 1、top-k −1、medium、保留历史 thinking、finalize none、原 prompt/feedback 和 KernelGym 计分口径

预检发现一台候选机器仍有其它任务占用，本轮只使用确认空闲的三个节点，6 个 TP4 引擎共 24 张 H20。相对上一轮的必要运行调整如下；这些差异不应混入训练机制的因果解释

| 项目 | 上一轮 | 本轮 |
|---|---:|---:|
| 引擎数 / H20 数 | 8 / 32 | 6 / 24 |
| 每引擎 max-running / 客户端并发 | 64 / 256 | 32 / 128 |
| CUDA Graph max batch | 64 | 32 |
| Prefill chunk tokens | 8192 | 4096 |
| 轨迹生成 guard 秒 | 3600 | 21600 |
| Router request / queue timeout 秒 | 14400 / 2400 | 21600 / 21600 |

HTTP 客户端原有无限总超时保持不变，KernelGym 算子与客户端 timeout 未修改。无生成 abort、推理请求超时、缺失反馈或 replacement 字符；所有任务级失败按原口径计分

三节点权重、tokenizer/config、数据和代码哈希均核验，实际 Ray 执行包也已逐文件核对。先通过实际 192K MTP 边界检查，再通过真实 `evaluation=True` 三轮客户端 canary；canary 每轮 256 输出 tokens 仅用于生成、precheck 反馈和 metadata 链路检查，不计入正式分数，也未泄漏到正式 cap。正式作业 `qwen38-fastcredit025-step100-ctx64k128k192k-20260914` 仅含本模型，Controller 已正常结束，未创建后续模型任务

[配置与证据入口](../../local_artifacts/qwen38/fastcredit_step100_context64k128k192k_20260914/README.md)索引了原始 argv、实际引擎参数、BF16 解析、节点同步、预算核对和人工复核；[资源释放证据](../../local_artifacts/qwen38/fastcredit_step100_context64k128k192k_20260914/cleanup_evidence.json)确认本次 24 张卡已释放、训练继续停止，未干预其它容器

## Step100：32K / 48K / 64K 对照

扩大三轮总 context 后，两模型的预算截断都减少，但 **FastCredit025 的额外截断没有完全消除**。全部轮次的截断差由旧预算下的 13.17 个百分点缩小到 6.42 个百分点；剩余差异主要集中在 L3。本次只能确认这些 checkpoint 在两套预算下的实际表现，不据此确定训练目标造成该行为的根因

| 模型 | 旧 24K/32K/40K 截断（%） | 新 32K/48K/64K 截断（%） |
|---|---:|---:|
| TRLOO baseline | 12.40 | 0.88 |
| FastCredit025 | 25.57 | 7.30 |

新预算下，FastCredit025 的三轮最佳 Correct 在 L1/L2 较高、L3 略低，三档最佳 Fast@1.0 均较高；但第三轮 Correct 仍分别比 baseline 低 5.38 / 4.13 / 7.00 个百分点。因此“保留已验证最佳候选”和“直接取最后一轮”仍给出不同结论。这些是单次评测点估计，未做显著性检验或训练重复

两模型均为完成 100 次更新的 `iter_0000099`：原 TRLOO baseline 与 FastCredit025（正确且实测 speedup ≥ 1 才加来源 bonus，系数 0.25、无最终回报封顶）。两模型各完成 2,000 条轨迹、6,000 轮记录，独立 raw env_state 计数与维护 summarizer 一致，无生成 abort 或缺失反馈。此次评测未恢复训练，收尾记录确认本次使用的 32 张 H20 已释放

### 新预算下的各轮质量

表格从原始计数统一四舍五入到两位小数。三轮总 context 上限为 32768 / 49152 / 65536，包含 prompt 与全部保留历史。第三轮为原生 MTP 保留 4 个 token，因此实际请求的有效总上限为 65532。每档每题 8 条三轮轨迹；分母 L1/L2/L3 为 800/800/400 条，失败保留在分母。Best 是每条轨迹任一轮达到相应指标，区别于最后一轮和 pass@8

#### L1

| 模型 | 轮次 | Compile（%） | Correct（%） | Fast@1.0（%） | Fast@1.2（%） |
|---|---|---:|---:|---:|---:|
| TRLOO baseline | T1 | 94.50 | 84.88 | 22.25 | 14.38 |
| TRLOO baseline | T2 | 94.38 | 81.75 | 32.00 | 20.25 |
| TRLOO baseline | T3 | 94.13 | 85.13 | 36.13 | 24.25 |
| TRLOO baseline | Best | 100.00 | 98.13 | 41.88 | 25.00 |
| FastCredit025 | T1 | 95.88 | 87.38 | 24.00 | 18.50 |
| FastCredit025 | T2 | 89.25 | 82.38 | 35.50 | 23.88 |
| FastCredit025 | T3 | 86.38 | 79.75 | 41.00 | 27.88 |
| FastCredit025 | Best | 99.88 | 99.50 | 45.25 | 28.88 |

| 模型 | 轮次 | 截断（%） | 未关闭 thinking（%） | Thinking 中位 token | Answer 中位 token | 可生成预算中位 token | 剩余未用预算中位 token |
|---|---|---:|---:|---:|---:|---:|---:|
| TRLOO baseline | T1 | 0.00 | 1.00 | 6139.00 | 1335.50 | 31411.00 | 23758.00 |
| TRLOO baseline | T2 | 0.13 | 1.88 | 5083.00 | 1509.00 | 39797.50 | 32641.50 |
| TRLOO baseline | T3 | 0.13 | 0.63 | 4125.50 | 1658.50 | 48685.50 | 40576.00 |
| FastCredit025 | T1 | 0.38 | 0.25 | 6401.50 | 1352.50 | 31411.00 | 23543.00 |
| FastCredit025 | T2 | 4.13 | 5.00 | 8108.00 | 1506.50 | 39591.50 | 29598.50 |
| FastCredit025 | T3 | 5.50 | 4.13 | 6421.00 | 1541.50 | 45593.50 | 34081.50 |

#### L2

| 模型 | 轮次 | Compile（%） | Correct（%） | Fast@1.0（%） | Fast@1.2（%） |
|---|---|---:|---:|---:|---:|
| TRLOO baseline | T1 | 91.00 | 62.13 | 3.88 | 3.75 |
| TRLOO baseline | T2 | 91.63 | 64.88 | 16.00 | 8.88 |
| TRLOO baseline | T3 | 91.25 | 65.50 | 20.88 | 12.00 |
| TRLOO baseline | Best | 100.00 | 82.25 | 23.00 | 12.38 |
| FastCredit025 | T1 | 92.63 | 65.25 | 9.38 | 7.88 |
| FastCredit025 | T2 | 90.13 | 66.25 | 22.00 | 13.50 |
| FastCredit025 | T3 | 82.50 | 61.38 | 26.00 | 17.13 |
| FastCredit025 | Best | 99.88 | 83.75 | 29.88 | 17.88 |

| 模型 | 轮次 | 截断（%） | 未关闭 thinking（%） | Thinking 中位 token | Answer 中位 token | 可生成预算中位 token | 剩余未用预算中位 token |
|---|---|---:|---:|---:|---:|---:|---:|
| TRLOO baseline | T1 | 0.13 | 1.13 | 10814.00 | 2002.00 | 31370.50 | 18414.00 |
| TRLOO baseline | T2 | 0.13 | 1.63 | 5403.00 | 2133.50 | 34430.00 | 26334.00 |
| TRLOO baseline | T3 | 0.25 | 0.88 | 5479.50 | 2202.50 | 42342.00 | 33569.50 |
| FastCredit025 | T1 | 0.50 | 0.25 | 10759.00 | 1955.00 | 31370.50 | 18379.00 |
| FastCredit025 | T2 | 4.13 | 3.50 | 8674.00 | 2023.50 | 34410.00 | 22991.00 |
| FastCredit025 | T3 | 6.88 | 5.88 | 9195.00 | 2077.50 | 39004.50 | 25686.00 |

#### L3

| 模型 | 轮次 | Compile（%） | Correct（%） | Fast@1.0（%） | Fast@1.2（%） |
|---|---|---:|---:|---:|---:|
| TRLOO baseline | T1 | 45.25 | 25.50 | 2.50 | 1.75 |
| TRLOO baseline | T2 | 71.00 | 34.25 | 4.25 | 3.25 |
| TRLOO baseline | T3 | 80.75 | 38.50 | 6.25 | 4.75 |
| TRLOO baseline | Best | 89.00 | 52.50 | 6.75 | 4.75 |
| FastCredit025 | T1 | 50.50 | 24.75 | 3.00 | 2.00 |
| FastCredit025 | T2 | 61.50 | 30.25 | 5.50 | 3.75 |
| FastCredit025 | T3 | 66.50 | 31.50 | 6.50 | 5.00 |
| FastCredit025 | Best | 85.50 | 51.50 | 7.50 | 5.75 |

| 模型 | 轮次 | 截断（%） | 未关闭 thinking（%） | Thinking 中位 token | Answer 中位 token | 可生成预算中位 token | 剩余未用预算中位 token |
|---|---|---:|---:|---:|---:|---:|---:|
| TRLOO baseline | T1 | 6.50 | 34.25 | 19196.00 | 2105.50 | 30845.50 | 9417.00 |
| TRLOO baseline | T2 | 4.50 | 4.75 | 3459.00 | 3904.00 | 25612.50 | 15697.00 |
| TRLOO baseline | T3 | 0.75 | 1.25 | 5495.00 | 4251.00 | 31706.00 | 20510.50 |
| FastCredit025 | T1 | 34.75 | 22.00 | 21601.50 | 2647.00 | 30845.50 | 4089.50 |
| FastCredit025 | T2 | 19.50 | 6.75 | 6415.50 | 3883.50 | 20128.50 | 8206.00 |
| FastCredit025 | T3 | 12.25 | 4.75 | 7748.00 | 4165.50 | 24159.00 | 9706.50 |

### 与旧 24K / 32K / 40K 预算比较

| Level | 模型 | 旧 Best Correct（%） | 新 Best Correct（%） | 旧 Best Fast@1.0（%） | 新 Best Fast@1.0（%） | 旧全部轮次截断（%） | 新全部轮次截断（%） |
|---|---|---:|---:|---:|---:|---:|---:|
| L1 | TRLOO baseline | 97.50 | 98.13 | 43.38 | 41.88 | 5.04 | 0.08 |
| L1 | FastCredit025 | 97.88 | 99.50 | 42.38 | 45.25 | 14.79 | 3.33 |
| L2 | TRLOO baseline | 81.25 | 82.25 | 22.25 | 23.00 | 7.79 | 0.17 |
| L2 | FastCredit025 | 81.25 | 83.75 | 25.50 | 29.88 | 19.38 | 3.83 |
| L3 | TRLOO baseline | 44.00 | 52.50 | 8.00 | 6.75 | 36.33 | 3.92 |
| L3 | FastCredit025 | 40.00 | 51.50 | 4.75 | 7.50 | 59.50 | 22.17 |


#### 旧预算逐轮明细

以下均为此前已完成的 24K/32K/40K 结果；质量指标读取原始计数，长度用旧 dump 按本轮相同 token 分界口径重算。新预算对应逐轮明细见上方 L1–L3 表。下表长度均为中位数，失败仍保留在分母

旧预算 L1

| 模型 | 轮次 | Compile（%） | Correct（%） | Fast@1.0（%） | Fast@1.2（%） | 截断（%） | Thinking 中位 token | Answer 中位 token |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TRLOO baseline | T1 | 94.38 | 85.00 | 21.88 | 15.38 | 0.25 | 6131.50 | 1323.00 |
| TRLOO baseline | T2 | 89.38 | 79.50 | 34.88 | 21.00 | 6.63 | 4567.50 | 1476.00 |
| TRLOO baseline | T3 | 86.00 | 78.38 | 35.88 | 23.38 | 8.25 | 3719.50 | 1541.00 |
| FastCredit025 | T1 | 96.13 | 89.25 | 22.50 | 18.00 | 1.38 | 6388.00 | 1374.00 |
| FastCredit025 | T2 | 76.13 | 71.13 | 34.88 | 22.63 | 19.63 | 7971.00 | 1303.00 |
| FastCredit025 | T3 | 72.25 | 67.13 | 34.63 | 25.75 | 23.38 | 5289.50 | 1320.50 |

旧预算 L2

| 模型 | 轮次 | Compile（%） | Correct（%） | Fast@1.0（%） | Fast@1.2（%） | 截断（%） | Thinking 中位 token | Answer 中位 token |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TRLOO baseline | T1 | 88.75 | 59.38 | 3.63 | 3.38 | 2.50 | 10473.50 | 1936.50 |
| TRLOO baseline | T2 | 84.50 | 59.88 | 13.63 | 7.75 | 10.63 | 5115.50 | 1975.00 |
| TRLOO baseline | T3 | 82.75 | 61.13 | 20.13 | 11.75 | 10.25 | 4753.50 | 2056.50 |
| FastCredit025 | T1 | 85.75 | 62.25 | 8.38 | 7.63 | 7.00 | 10965.00 | 1903.50 |
| FastCredit025 | T2 | 72.75 | 53.63 | 18.50 | 12.25 | 22.63 | 7333.50 | 1828.00 |
| FastCredit025 | T3 | 64.88 | 49.00 | 21.50 | 14.75 | 28.50 | 7469.00 | 1803.00 |

旧预算 L3

| 模型 | 轮次 | Compile（%） | Correct（%） | Fast@1.0（%） | Fast@1.2（%） | 截断（%） | Thinking 中位 token | Answer 中位 token |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TRLOO baseline | T1 | 34.25 | 19.00 | 3.75 | 3.00 | 44.50 | 18921.50 | 1372.00 |
| TRLOO baseline | T2 | 50.00 | 25.00 | 5.25 | 4.50 | 33.50 | 1593.50 | 3413.00 |
| TRLOO baseline | T3 | 55.00 | 29.25 | 7.00 | 5.50 | 31.00 | 2220.00 | 3428.50 |
| FastCredit025 | T1 | 26.75 | 19.50 | 1.50 | 1.00 | 66.50 | 20998.00 | 528.50 |
| FastCredit025 | T2 | 35.50 | 18.00 | 2.50 | 1.50 | 59.25 | 6267.00 | 2158.50 |
| FastCredit025 | T3 | 36.25 | 21.75 | 3.50 | 2.75 | 52.75 | 4184.50 | 2720.00 |

旧长度完整均值、中位数、P90 与推导请求预算见 [baseline](../../local_artifacts/qwen38/step100_context32k48k64k_20260914/attempt2/historical_baseline/context_length_audit.json)和[FastCredit025](../../local_artifacts/qwen38/step100_context32k48k64k_20260914/attempt2/historical_fastcredit025/context_length_audit.json)；新旧 Best 指标分别见本节比较表、上方新预算质量表及下方保留的旧协议章节

### 长度与实际预算

表中的 thinking/answer 长度来自保存的响应 token IDs，以首个 `</think>` 分界，并去除 special tokens；没有关闭 token 时，全部计入 thinking、answer 为 0。统计包含全部轮次。未关闭 thinking 不等于预算截断，也不等于没有可提取的 CUDA 代码：两模型分别有 205 / 51 轮在未关闭 thinking 的情况下正常结束生成。对 baseline 的这类记录核对了文本与 token，未发现文本有关闭标记而 token 统计漏认的情况；抽查可见 EOS 停在推导或代码中途，见[关闭标记复核](../../local_artifacts/qwen38/step100_context32k48k64k_20260914/attempt2/baseline/thinking_delimiter_review.json)

FastCredit025 在反馈后的 thinking 中位长度三档均更长，answer 中位长度则接近或更短；这描述输出分布，不单独建立训练机制的因果链。均值、P90、实际 prompt 长度以及逐条记录见 [baseline 长度审计](../../local_artifacts/qwen38/step100_context32k48k64k_20260914/attempt2/baseline/context_length_audit.json)与[FastCredit025 长度审计](../../local_artifacts/qwen38/step100_context32k48k64k_20260914/attempt2/fastcredit025/context_length_audit.json)

可生成预算是该轮实际请求的 `max_new_tokens`；剩余未用预算是请求额度减去实际 response tokens。12,000 轮均验证 `max_new_tokens = max(0, min(configured_output_cap, min(turn_cap, server_cap − 4) − prompt_tokens))`，未发现越界或仍被 40K engine 限制。T1/T2 保留完整的 32768/49152 总上限，MTP 的四 token 余量只限制第三轮最末边界。两模型所有 53 / 438 个截断轮次均耗尽对应请求额度，见[baseline 截断预算检查](../../local_artifacts/qwen38/step100_context32k48k64k_20260914/attempt2/baseline/truncation_budget_check.json)与[FastCredit025 检查](../../local_artifacts/qwen38/step100_context32k48k64k_20260914/attempt2/fastcredit025/truncation_budget_check.json)

旧预算的长度使用旧 dump 统一重算；旧请求预算从保存的 token 长度和原预算函数推导，新预算则由生成时直接记录的 metadata 核验，两者证据强度明确区分

### 实际输出与任务级失败

两模型各人工复核 9 条不同轨迹的三轮响应头尾、结构和反馈，共 54 轮；样例用于检查真实行为，不代表全体错误类型比例。完整文本与复核记录见 [baseline](../../local_artifacts/qwen38/step100_context32k48k64k_20260914/attempt2/baseline/manual_review.md)和[FastCredit025](../../local_artifacts/qwen38/step100_context32k48k64k_20260914/attempt2/fastcredit025/manual_review.md)

- Baseline L1 group539：前两轮 ConvTranspose3d 正确但慢，第三轮在 prompt 40422 的基础上用尽 25110 输出 tokens，达到有效总长度 65532 后停在 CUDA 实现中途
- FastCredit025 L1 group50：前两轮 GEMM 正确但慢，第三轮继续分析 shared-memory bank conflict，耗尽 18274 输出 tokens 后未交付完整代码
- FastCredit025 L2 group0：第三轮仍有 39420 输出 tokens，但继续进行优化推导直到用完预算，没有完整 ModelNew；这说明增加到 64K 仍不能消除所有长 thinking 和交付失败
- FastCredit025 L3 group34：首轮 AlexNet 正确但慢，第二轮优化后编译失败，第三轮修复 cuDNN API 时截断在 ModelNew.forward 中途
- 反馈后修复也实际存在，例如 baseline L3 group6 修正 `dtype()` 后正确，FastCredit025 L1 group33 修正 `data_ptr()` 后正确

| 模型 | L1 WorkerProcessCrashed 次数 | L2 次数 | L3 次数 |
|---|---:|---:|---:|
| TRLOO baseline | 2 | 12 | 3 |
| FastCredit025 | 2 | 20 | 4 |

这些任务按正常口径计失败并保留分母，没有选择性删除或重测。抽查 baseline L1 group261 / L3 group16、FastCredit025 L1 group466 的最终代码，可见 host C++ 解引用 CUDA tensor 的 `data_ptr`，与相应 native 函数中的崩溃栈吻合；这是所查候选的代码缺陷证据，不将其外推到全部 native crash。未做独立 replay，其余崩溃的根因尚未逐项闭合。两模型各有一轮含 replacement 字符，细节保存在长度审计的 anomalies 中

### 配置、执行与偏离

两模型均用四节点八个 TP4 engine、BF16、原生 MTP3、CUDA Graph、FA3/Triton，server / rollout / eval 全局 cap 均为 65536；客户端总并发 256、每引擎 max-running 64，未降低并发。canonical 数据、每题八条轨迹、temperature 1、top-p 1、top-k −1、medium、保留历史 thinking、finalize none、原 KernelGym URL 和计分口径一致

本轮统一使用 FastCredit025 已验证的冻结评测源码；历史 prompt/feedback 模板同哈希，源码差异主要是本轮关闭的 component 分支。Mamba 参数名统一为 `radix-cache-strategy=extra_buffer`。旧 baseline 曾用三节点六引擎，本轮两模型均用八引擎；与旧结果比较时保留这一运行差异，不将时间差单独解释成预算或算法收益

首跑 baseline 因新增预算诊断引用了未传入内部函数的 `evaluation` 变量而失败，已停止并全部作废。两份旧协议冻结源码与归档 manifest 匹配，均无该新增块，因此旧分数不受此 NameError 影响，见[源码复核](../../local_artifacts/qwen38/step100_context32k48k64k_20260914/old_protocol_nameerror_audit.json)。修复、四节点重同步及真实三轮 GPU 诊断通过后，两模型均从本轮 `attempt2` 完整重跑；本文仅使用后缀 `-r2` 的有效作业。另一次 65516 输入加 16 输出 tokens 的 64K GPU 边界检查通过，两项诊断均不计入正式分数。有效作业分别耗时 80.24 / 105.67 分钟，没有使用 eager 或恢复训练。Controller 已正常收尾，`completed` 仅含 baseline、fastcredit025，没有第三个模型任务，见[终态回执](../../local_artifacts/qwen38/step100_context32k48k64k_20260914/attempt2/comparison_terminal.json)。收尾复查时，本次 controller 和 Ray 作业均已退出；一张 GPU 上的占用已核实来自其它容器的新任务，其余 31 张卡空闲，不属于本次评测残留，见[收尾与资源归属核查](../../local_artifacts/qwen38/step100_context32k48k64k_20260914/attempt2/final_closeout_recheck.json)

两模型权重、配置、数据和实际执行包均核验一致；[配置与完整证据入口](../../local_artifacts/qwen38/step100_context32k48k64k_20260914/attempt2/README.md)索引了 checkpoint 来源、实际 engine 参数、运行回执、旧长度重算、作废记录和清理证据。模型身份分别见 [baseline](../../local_artifacts/qwen38/step100_context32k48k64k_20260914/attempt2/baseline/identity.json)与[FastCredit025](../../local_artifacts/qwen38/step100_context32k48k64k_20260914/attempt2/fastcredit025/identity.json)

## FastCredit025 step100

本节记录三轮总 context 为 24K/32K/40K 的已完成评测。与同预算的原 TRLOO baseline 相比，L1/L2 的三轮最佳 Correct 接近，L2 的 Fast 指标更高；L3 的 Compile、Correct 和 Fast 均更低。首轮正确率的点估计有所提高，但第三轮正确率三档都低于 baseline，整体截断也更多，当前没有显示出全面优于 baseline 的结果

### 三轮最佳与 baseline 对照

每条轨迹任一轮达到指标即计入三轮最佳，各指标分别统计；分母为该 level 的全部轨迹，失败不从分母中移除。Correct 排除 decoy，Fast 还要求正确且实测 speedup 达到阈值。该口径不是 pass@8，也不代表最后一轮答案

| Level | 模型 | Best Compile（%） | Best Correct（%） | Best Fast@1.0（%） | Best Fast@1.2（%） |
|---|---|---:|---:|---:|---:|
| L1 | TRLOO baseline | 99.88 | 97.50 | 43.38 | 24.63 |
| L1 | FastCredit025 | 99.63 | 97.88 | 42.38 | 26.75 |
| L2 | TRLOO baseline | 99.75 | 81.25 | 22.25 | 12.75 |
| L2 | FastCredit025 | 99.63 | 81.25 | 25.50 | 15.88 |
| L3 | TRLOO baseline | 76.00 | 44.00 | 8.00 | 6.00 |
| L3 | FastCredit025 | 65.25 | 40.00 | 4.75 | 3.00 |

### 最后一轮质量与截断

三轮最佳会保留前面已经做对的结果，因此还要看最终交付质量。本次首轮 Correct 略高，但第三轮 Correct 更低；仅看 Best Correct 会掩盖这一差异

| Level | Baseline 首轮 Correct（%） | 本次首轮 Correct（%） | Baseline 第三轮 Correct（%） | 本次第三轮 Correct（%） |
|---|---:|---:|---:|---:|
| L1 | 85.00 | 89.25 | 78.38 | 67.13 |
| L2 | 59.38 | 62.25 | 61.13 | 49.00 |
| L3 | 19.00 | 19.50 | 29.25 | 21.75 |

| 指标 | TRLOO baseline（%） | FastCredit025（%） |
|---|---:|---:|
| 全部生成轮次的截断比例 | 12.40 | 25.57 |

人工抽查的 L1 `group_id=53` 展示了这个问题：第一轮错误调用 `ShapeView.has_value()` 导致编译失败，第二轮修正后正确；第三轮为优化性能继续推导 shared-memory bank conflict，最终思考截断、没有交付完整实现。该样例说明“已经修好又未完成输出”确实存在，不代表这一模式的全体占比；[完整三轮源码与反馈](../../local_artifacts/component_reward_training/fast_credit_step100_eval_20260914/manual_examples.json)保存在本地证据目录

这些是本次运行的点估计，尚未做题级配对 bootstrap 或多次训练重复，暂不判断统计显著性，也不把截断增长单独归因于 component credit。抽查还发现 L2 `group_id=16` 的第一、三轮返回 `WorkerProcessCrashed`，原因未闭合；作业成功、没有生成 abort，不等于所有任务级失败都已排除环境因素

### 协议与完成情况

本轮训练使用正确且 speedup ≥ 1 门槛、来源系数 0.25、取消最终回报封顶，完成 100 次更新后停止，保留 step80/100 源 checkpoint。评测沿用相同 canonical L1–L3 数据、每题 8 条轨迹、三轮 24K/32K/40K、BF16/MTP3、temperature 1、medium、finalize none；baseline 使用 24 H20，本次使用 32 H20

评测于 2026-09-14 06:06（UTC+9）成功结束，耗时约 85 分钟，全部轨迹与三轮反馈收齐，无生成 abort 或缺失 env_result。raw env_state 的独立计数与维护 summarizer 一致；收尾回执确认此次四节点评测 GPU 已释放，未恢复训练。配置、运行边界和验收入口见[本轮评测记录](../../local_artifacts/component_reward_training/fast_credit_step100_eval_20260914/README.md)，逐轮数据见[原始计数](../../local_artifacts/component_reward_training/fast_credit_step100_eval_20260914/results_audit.json)与[完整汇总](../../local_artifacts/component_reward_training/fast_credit_step100_eval_20260914/summary.txt)

本节表格按原始计数统一四舍五入到两位小数；原始汇总使用不同的半数舍入规则，个别末位可能不同，底层计数没有变化

## Source-component reward step100

本节对应历史 source-component `replace` 目标，区别于前述 additive FastCredit。训练按用户指令停止后，评测完成 100 次更新的 `iter_0000099`。四节点的八个 TP4 引擎完成全部 2,000 条三轮轨迹、6,000 轮记录，作业成功且独立计数与维护汇总一致，耗时约 93 分钟。收尾记录确认本次使用的 32 张 H20 已释放，未恢复训练

分母 L1/L2/L3 为全部 800/800/400 条轨迹，失败保留在分母。三轮最佳按每条轨迹任一轮达到相应指标统计，不是最后一轮或 pass@8；本表保留三位小数

| Level | 首轮 Correct (%) | 第三轮 Correct (%) | 三轮最佳 Correct (%) | 三轮最佳 Fast@1.0 (%) | 三轮最佳 Fast@1.2 (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| L1 | 50.375 | 34.375 | 81.875 | 37.750 | 24.125 |
| L2 | 24.625 | 23.625 | 50.375 | 15.875 | 7.750 |
| L3 | 6.500 | 10.750 | 18.500 | 4.250 | 3.000 |

沿用下节基线的三轮 24K/32K/40K、BF16/MTP3、temperature1、canonical 数据、每题 8 samples 和 finalize none。三轮最佳 Correct 相比基线分别下降 15.625/30.875/25.500 个百分点，截断从 12.40% 增至 46.63%。人工抽查可见长推导和代码未完成；模型/数据身份、权重导出、执行包和计数核对未发现错误，训练目标与退化之间的因果链尚未建立

详细各轮分数、输出抽查、3 条含 replacement 字符的响应及并发边界见 [source-component step100 报告](../../local_artifacts/component_reward_training/step100_eval_20260912/results.md)。本次没有生成 abort、缺失反馈或 OOM；截断属于有计分记录的失败输出

## v4_1 三轮 TRLOO step100

用户要求停止三轮 packed TRLOO 训练并测试 step100，评测读取 `iter_0000099`。模型、optimizer 源 checkpoint 均保留，训练未恢复；评测作业 `qwen38-v4-1-trloo-step100-kernelbench-3turn-3nodes-20260908` 成功结束，使用三个节点共 24 张 H20，结束后已释放。canonical L1/L2/L3 各 100/100/50 题、每题 8 条三轮轨迹，2000 条轨迹、6000 条记录完整，无缺失反馈或生成 abort

| Level | 首轮 Correct (%) | 第三轮 Correct (%) | 三轮最佳 Correct (%) | 三轮最佳 Fast@1.0 (%) | 三轮最佳 Fast@1.2 (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| L1 | 85.00 | 78.38 | 97.50 | 43.38 | 24.63 |
| L2 | 59.38 | 61.13 | 81.25 | 22.25 | 12.75 |
| L3 | 19.00 | 29.25 | 44.00 | 8.00 | 6.00 |

三轮最佳是每条轨迹任一轮达到指标，分母保持每档全部轨迹，并非 pass@8 或最后一轮质量。L1 有 135 条首轮正确但第三轮错误的轨迹，82 条首轮错误但第三轮正确；因此最佳分数依赖保留并选择已验证候选，不能直接当作最后输出效果

本次使用 BF16、原生 MTP3、三轮 24K/32K/40K、temperature1、FA3/Triton、CUDA Graph、tvm_ffi、finalize none，区别于以下单轮 no-spec 结果。为导出当前 checkpoint，串行转换器补充 GDN 分段 DCP 的合并，27 项 CPU 测试、完整张量形状/有限性审计与源张量抽查通过。配置、扩容边界、原始数据与独立计数由[step100 三轮评测报告](../../local_artifacts/qwen38/trloo_v4_1_step100_eval_20260908/results.md)统一维护

## DataV2 / DataV4 / 新 reward 单轮结论

新 DataV4 lineage 明显强于旧版 DataV2 lineage。最可比的同 step80 结果中，逐 trajectory Correct 在 L1/L2/L3 分别从 **78.250%/38.375%/6.000%** 提升到 **87.250%/60.625%/16.750%**，即 **+9.000/+22.250/+10.750 pct-pt**。按题目配对的 20,000 次 bootstrap 95% CI 分别为 `[+4.750,+13.375]`、`[+16.375,+28.250]`、`[+5.000,+17.250]` pct-pt，三档提升都稳定为正

DataV4 内部从 step80 到 step100 又在 L1/L2 Correct 上提升 **2.875/4.125 pct-pt**，两档 paired-bootstrap CI 为正；L3 变化 **-0.250 pct-pt**，属于统计持平。在该组 16K checkpoint 中，**DataV4 step100** 的 L1/L2 Correct 最高。相较旧版当时推荐的 step40，它的 L1/L2/L3 Correct 分别高 **15.625/19.375/5.750 pct-pt**

新 lineage 的 Compile 与截断率改善比 Fast 更一致。同 step80 下，DataV4 的三档 Compile 提升 `12.750/39.750/15.500` pct-pt，截断率降低 `13.750/42.750/16.750` pct-pt；但 L1/L2 Fast@1.0 分别下降 `3.125/1.750` pct-pt。该 lineage 的正确交付率更高，性能指标没有同步提升。Fast 的小幅差异还会受不同 KernelGym 节点、reference cache 与计时波动影响

两条 lineage 都从同一 BF16 base 重新训练，核心训练配置相同，但 DataV4 是 fresh run，训练 prompt 数据和后续随机 rollout 历史同时变化。因此该对比能确认 **DataV4 lineage 的实际 checkpoint 更强**，不能把全部增益严格归因到某一类新增数据行

2026-08-27 又完成了新 reward、24K lineage 的 step80 全量评测。相对旧 DataV4 step80，它的 L1/L2/L3 Correct 差值为 `-2.38/0.00/+3.75`，Compile 差值为 `-0.62/+5.12/+15.50`，截断率差值为 `-0.75/-6.00/-45.75`；三档 Fast@1.0 差值均为负。由于训练 context、训练 temperature、partial reward、PRS/coverage-RS 和长度惩罚同时变化，这是一组配置 bundle 对照，不是单变量 reward 因果消融。旧 DataV4 step100 仍是 L1/L2 Correct 最优点；新 reward step80 则在本文对照中具有最高的 L3 Correct、Compile 与最低截断率

## 统一结果

指标均为单轮逐 trajectory 口径；比例列单位为 `%`

| lineage | checkpoint | level | n | Compile (%) | Correct (%) | Fast@1.0 (%) | Fast@1.2 (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| DataV2 | step40 (`iter_0000039`) | L1 | 800 | 88.750 | 74.500 | 18.750 | 14.875 |
| DataV2 | step40 (`iter_0000039`) | L2 | 800 | 66.750 | 45.375 | 1.875 | 1.500 |
| DataV2 | step40 (`iter_0000039`) | L3 | 400 | 14.250 | 10.750 | 1.500 | 0.750 |
| DataV2 | step80 (`iter_0000079`) | L1 | 800 | 84.125 | 78.250 | 23.625 | 19.375 |
| DataV2 | step80 (`iter_0000079`) | L2 | 800 | 48.250 | 38.375 | 4.750 | 4.000 |
| DataV2 | step80 (`iter_0000079`) | L3 | 400 | 7.000 | 6.000 | 1.000 | 1.000 |
| DataV4 | step80 (`iter_0000079`) | L1 | 800 | 96.875 | **87.250** | 20.500 | 15.125 |
| DataV4 | step80 (`iter_0000079`) | L2 | 800 | 88.000 | **60.625** | 3.000 | 2.625 |
| DataV4 | step80 (`iter_0000079`) | L3 | 400 | 22.500 | **16.750** | 1.750 | 0.750 |
| DataV4 | step100 (`iter_0000099`) | L1 | 800 | 97.875 | **90.125** | 21.375 | 16.875 |
| DataV4 | step100 (`iter_0000099`) | L2 | 800 | 91.375 | **64.750** | 2.500 | 2.375 |
| DataV4 | step100 (`iter_0000099`) | L3 | 400 | 24.000 | **16.500** | 1.500 | 1.250 |

`Correct` 是数值正确且不是 decoy kernel；`Fast@x` 要求先 Correct，再达到相对 PyTorch reference 的对应 speedup

## 新 reward、24K step80

| level | Compile (%) | Correct (%) | Fast@1.0 (%) | Fast@1.2 (%) | 截断率 (%) | pass@8 (%) |
|---|---:|---:|---:|---:|---:|---:|
| L1 | 96.25 | 84.88 | 18.12 | 14.88 | 0.25 | 99.00 |
| L2 | 93.12 | 60.62 | 0.62 | 0.50 | 1.00 | 85.00 |
| L3 | 38.00 | 20.50 | 0.75 | 0.00 | 30.00 | 46.00 |

与旧 DataV4 step80 的相同步数对照如下。Correct 的 95% CI 使用 problem-paired 20,000-draw bootstrap，seed `20260827`；两次正式 eval 的题目、单轮采样、temperature 1.0、medium、tvm_ffi、no-spec 和每题 8 samples 相同，context 分别跟随 checkpoint 原生配置使用 16K 与 24K

| level | Compile 差值 (pp) | Correct 差值 (pp) | Fast@1.0 差值 (pp) | Fast@1.2 差值 (pp) | 截断率差值 (pp) | Correct 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---:|
| L1 | -0.62 | -2.38 | -2.38 | -0.25 | -0.75 | [-5.25,+0.62] |
| L2 | +5.12 | 0.00 | -2.38 | -2.12 | -6.00 | [-4.12,+4.00] |
| L3 | +15.50 | +3.75 | -1.00 | -0.75 | -45.75 | [0.00,+7.50] |

新 dump 的三档题目集合完整，每题恰好 8 个样本；独立重算与 summary 完全一致。全局没有空响应、重复响应、缺失指标、非有限 speedup、`correct && !compiled` 或两份环境状态不一致。人工检查了每档一个正确完成样本和一个失败/截断样本：成功样本都有完整 CUDA/TVM-FFI 与 `ModelNew`；L2/L3 失败样本在最终代码完成前被 context 截断，L1 失败样本已生成可编译但错误的完整实现、随后在自检修订中截断。六条样本均没有乱码或复制响应

## 新旧数据：同 step80 配对比较

| level | Compile (pp) | Correct (pp) | Fast@1.0 (pp) | Fast@1.2 (pp) | 截断率 (pp) | Correct paired-bootstrap 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---:|
| L1 | +12.750 | **+9.000** | -3.125 | -4.250 | -13.750 | **[+4.750,+13.375]** |
| L2 | +39.750 | **+22.250** | -1.750 | -1.375 | -42.750 | **[+16.375,+28.250]** |
| L3 | +15.500 | **+10.750** | +0.750 | -0.250 | -16.750 | **[+5.000,+17.250]** |

Bootstrap 以 problem 为重采样单位，每题内部保留 8 个 samples，20,000 draws，seed `20260824`；三档差值为正的 bootstrap 概率均为 `1.0`。评测 prompt、采样数和推理协议相同，因此这是本次新旧训练数据最直接的 checkpoint 对比

问题级 pass@8 也一致改善：DataV2 step80 的 L1/L2/L3 为 `98/100`、`71/100`、`9/50`，DataV4 step80 为 `100/100`、`84/100`、`14/50`

## 两条 lineage 各自的训练内变化

### DataV2 step40 → step80

| level | Compile (pp) | Correct (pp) | Fast@1.0 (pp) | Fast@1.2 (pp) | 截断率 (pp) | Correct paired-bootstrap 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---:|
| L1 | -4.625 | +3.750 | +4.875 | +4.500 | +9.375 | `[-1.375,+8.750]` |
| L2 | -18.500 | **-7.000** | +2.875 | +2.500 | +27.375 | `[-12.250,-1.750]` |
| L3 | -7.250 | **-4.750** | -0.500 | +0.250 | +9.250 | `[-8.250,-1.750]` |

旧版 step80 的 L2/L3 明确回退。它在完成样本上的条件正确率实际更高，但生成显著变长，大量响应在代码或 binding 完成前耗尽 16K context，截断样本几乎全部失败；这使总体 Compile/Correct 下滑。旧版当时因此推荐 step40

### DataV4 step80 → step100

| level | Compile (pp) | Correct (pp) | Fast@1.0 (pp) | Fast@1.2 (pp) | 截断率 (pp) | Correct paired-bootstrap 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---:|
| L1 | +1.000 | **+2.875** | +0.875 | +1.750 | -0.375 | **[+0.250,+5.750]** |
| L2 | +3.375 | **+4.125** | -0.500 | -0.250 | -1.875 | **[+0.375,+7.875]** |
| L3 | +1.500 | -0.250 | -0.250 | +0.500 | -2.750 | `[-2.250,+1.750]` |

同样使用 problem-paired 20,000-draw bootstrap；差值为正的概率为 L1 `0.98030`、L2 `0.98315`、L3 `0.35870`。DataV4 到 step100 没有复现旧版 step80 的 L2/L3 correctness 崩落

## 截断与输出质量

| lineage / checkpoint | L1 截断率 (%) | L2 截断率 (%) | L3 截断率 (%) | L1/L2/L3 response length median (tokens) |
|---|---:|---:|---:|---:|
| DataV2 step40 | 5.375 | 22.375 | 83.250 | 5845.5 / 10978.5 / 14329 |
| DataV2 step80 | 14.750 | 49.750 | 92.500 | 8496 / 14796 / 14379 |
| DataV4 step80 | 1.000 | 7.000 | 75.750 | 5444.5 / 9931 / 14306 |
| DataV4 step100 | 0.625 | 5.125 | 73.000 | 5168 / 9329.5 / 14254 |
| DataV4 新 reward step80 | 0.250 | 1.000 | 30.000 | 6530.5 / 11426 / 19107 |

旧 DataV4 16K lineage 几乎消除了 L1 截断并大幅缓解 L2，但 L3 仍有约四分之三响应触及预算。新 reward 24K step80 把 L3 截断率降到 30.00%，仍明显高于 L1/L2，长组合任务的收尾依然是主要瓶颈。SGLang context 包含 prompt，因此有效生成预算小于配置 cap，截断响应本身不必达到 16K 或 24K tokens

前四个 dump 合计恰好 8,000 trajectories，新 reward step80 另有 2,000 trajectories；每个 dump 中各 level 的每个 problem id 都恰好出现 8 次。独立审计没有发现 missing metrics、`correct && !compiled`、非有限 speedup、env state 不一致、空响应、Unicode replacement、非法控制字符或 dump 内重复响应。DataV2 step40 有一个候选 kernel 触发 300 秒执行超时，其余错误均来自候选代码本身而非评测服务断连

每个 checkpoint、每个 level 都人工检查了一条正确完成样本和一条失败截断样本，前四个 checkpoint 共 24 条，新 reward step80 再检查 6 条。它们都是连贯的英文算子分析和 CUDA/TVM-FFI 实现；成功样本结构完整，失败样本主要停在推演、CUDA 源码或 binding 中间，没有随机多语种词流或乱码

## 统一评测协议与 lineage

- canonical validation：L1 100 题、L2 100 题、L3 50 题，每题 8 samples，即 `800/800/400` trajectories/checkpoint
- `tvm_ffi`、`max_turns=1`、temperature `1.0`、top-p `1.0`、top-k `-1`、medium reasoning、context/response `16384`
- BF16 HF serving、no-spec、FA3 attention、Triton linear/GDN、带 padding CUDA graph；没有使用 eager
- 新 reward step80 使用 context/response cap `24576`。SGLang context 是 prompt 与 response 的总长度，因此单条样本的实际 response budget 是 `24576 - prompt_tokens`，不是额外再生成完整 24K；其余四个 checkpoint 同理使用 `16384 - prompt_tokens`
- L1/L2/L3 eval 数据 SHA256：`e034d42fe5e8ed0fac0e580bb9070379f719080b05d666accbac59abc081435f`、`11c1858d88be14ebc7fa766390f46a0db1ad872e921ddc61411e7df4d2e64cfe`、`6b3c85f2e57f307036b8c38ff26891e8b88ac44afb702a7c7f539d2b8307755e`
- DataV2 lineage：`FAsync.NoSpec.DefaultCG.DPPOPredictive.LongestFirst.matched.medium.Temp1.1.tvm_ffi.Qwen3.8-27B.BF16Train.FP8Rollout.CTX16384`；训练数据为 `Data/prompt_tvm_v2/drkernel_rl_thinking.parquet`（71,996 rows，SHA256 `17b948be017e8e57e6e87dcae15eca3acd5b9feab141be49dd4589e74ab14d25`）
- DataV4 lineage：`FAsync.NoSpec.DefaultCG.DPPOPredictive.LongestFirst.matched.medium.Temp1.1.DataV4.tvm_ffi.Qwen3.8-27B.BF16Train.FP8Rollout.CTX16384`，从 BF16 base iteration 0 fresh start，没有加载 DataV2 checkpoint；训练数据为 `Data/prompt_tvm_v4/release/train.parquet`（39,636 rows，SHA256 `189da56dca3acbb03b5532b360ec1eb9adde75c173952149a8515dcebfb08f79`）
- 新 reward lineage：`FAsync.NoSpec.DefaultCG.DPPOPredictive.LongestFirst.matched.medium.Temp1.0.Mismatch0p25.NoPRS.Len4096Pen0p2.DataV4.tvm_ffi.Qwen3.8-27B.BF16Train.FP8Rollout.CTX24576`；同样从 BF16 base iteration 0 fresh start，使用相同 DataV4 release，增加严格 output-mismatch partial reward `0.25`、关闭 PRS/coverage-RS，并使用最后 4096 response budget 最大 `0.2` 的线性长度惩罚

DataV2 step40 来自 node69/node70 归档 `/nfs/LOCAL/chenshuailin/checkpoints/qwen38_temp11_failed_enospc_20260823/iter_0000039`，DataV2 step80 来自旧实验的 `checkpoints/iter_0000079`。DataV4 step80 来自 node70 已验证的两机归档 `/nfs/LOCAL/chenshuailin/checkpoints/qwen38_datav4_rolling_20260823/iter_0000079`，step100 来自 DataV4 实验的两机完整 `iter_0000099`。DataV4 两个 checkpoint gather 后均为 32 个 shard、`489,654,266,558` shard bytes；转换后均为 851 tensors、11 个 safetensors、`53,792,108,344` weight bytes，HF 位于 `/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.8-27B-RL-Temp1.1-DataV4/step{80,100}`。源 Megatron checkpoint 未删除

新 reward step80 来自 node70 的 rolling archive `/nfs/LOCAL/chenshuailin/checkpoints/qwen38_datav4_mismatch0p25_ctx24576_rolling_20260825/iter_0000079`。归档 manifest 已验证，包含 node70/node69 两端共 32 个 shard；rolling keep1 已在发布 step80 后删除 step60 和两端 source half。转换得到 851 tensors、11 个 safetensors，部署 HF 位于 `/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.8-27B-RL-DataV4-Mismatch0p25-CTX24576/step80`

## 证据与产物

所有 `experiments` 路径都位于运行 snapshot `/nfs/FM/chenshuailin/projects/kernel_agents/slime-dev-csl-2-qwen38-rl-20260819`：

- DataV2 step40（node53）：`experiments/EvalFAsync.NoSpec.tvm_ffi.Qwen3.8-27B.CTX16384/step40/summary.20260823.052114.txt`，summary SHA256 `5061574f0c0dc320ff24195156c535a55b04b9e25afdfd8add25a7b46dbfeb27`；dump SHA256 `5fed1617620dfd28d0696ec5aeed265a81bd80d0fd07b5e6003acf57fe905ca0`
- DataV2 step80（node64）：`experiments/EvalFAsync.NoSpec.tvm_ffi.Qwen3.8-27B.CTX16384/step80/summary.20260823.053359.txt`，summary SHA256 `d5f17b7052d8c350678d4e096e1df3e590696062cdbe514ab3931f0662b673b2`；dump SHA256 `3e476a3f6bba6a2b3e037dbbec057185df85229278f7012aef647875ea3d55c7`
- DataV4 step80（node53，Ray job `raysubmit_26TLAYqePXX1Z3K3`）：`experiments/EvalFAsync.NoSpec.tvm_ffi.Qwen3.8-27B.CTX16384/datav4_step80/summary.20260824.061105.txt`，summary SHA256 `29fe60fec7631e8363fc3d771e2ec7852f7ccfcaacc92e0baba44d0abd094b60`；dump `317,624,807` bytes，SHA256 `fc096ea6716baf923cf50fb9535016582a9e6d181e332ef8f1d46e6158873846`；audit SHA256 `5246d4b0031eb37333c748ba7739c4de44fa3151bf4c72294f3585f4798643a6`
- DataV4 step100（node69，Ray job `raysubmit_JeUgSQZNwiUYcZBv`）：`experiments/EvalFAsync.NoSpec.tvm_ffi.Qwen3.8-27B.CTX16384/datav4_step100/summary.20260824.062754.txt`，summary SHA256 `57e7d812b45c271450be8e49ccf69e13f59bf34de68b00fb7c00015bf031d702`；dump `307,024,999` bytes，SHA256 `9b5644acd1749d80fde274d9d6fee6836128dd344ab341b6ec8ab0f16475f1b0`；audit SHA256 `16f6c53be2665634ad88cae90c6a08b4ecfa6010e6fc947aa722439f3a8e864f`
- 新 reward step80（node69，Ray job `raysubmit_GndRsHUTkJweCZSk`）：`experiments/EvalFAsync.NoSpec.tvm_ffi.Qwen3.8-27B.CTX24576/newreward_datav4_step80_ctx24576/summary.20260827.121940.txt`，summary SHA256 `26fd118c1ec754d65031185b17d43ffbf3ceb65bd5bf3fd9e1c9de0e574e1afe`；dump `391,431,335` bytes，SHA256 `f2913fc1b27f27e85bf544de5e1f1b15109146f842c0b8ab9cccf0ffc645fbcd`；audit SHA256 `ebd16cf1d79ae1858db759dd835b37f077502ad2e523bdbae89c7680896f9d02`；人工样本 SHA256 `769652018f2f38daa8a9dd8e7d6258c7f2d298ddbc0e332a77ddd02add36b1ff`
- 新旧 reward/config step80 配对比较：`local_artifacts/qwen38/newreward_step80_eval_20260827/oldreward16k_vs_newreward24k_step80_compare.20260827.json`，SHA256 `9a1c287390617aeae946a6ed2124f1774883ee2cd02311c23c812f5d34e1364b`
- DataV4 step80→100 比较报告 SHA256：`79ed80582d0cd27722aadae6930a359de96b890650f738b35c11614bc9487d2a`
- DataV2 step80→DataV4 step80 比较证据：`local_artifacts/qwen38/eval_step40_80/legacy_step80_vs_datav4_step80_correct_bootstrap_20260824.json`，SHA256 `7a137cb4dd6fc72d77be0dff11b8cc5b0434443492efc53d8252de7e84e1c19b`
- 独立审计脚本：`local_artifacts/qwen38/eval_step40_80/audit_l123_dump.py`、`compare_l123_audits.py`

五个正式 Ray eval 都成功结束，评测后的 serving/Ray 进程和 GPU 占用已释放，KernelGym 健康且队列为空。DataV4 step100 与新 reward step80 原计划使用 node64，但该机被另一任务占满 8 张 H20；未终止对方任务，按用户要求改到同型号 node69，推理协议与 A800 KernelGym 计分池不变

新 reward step80 的第一次启动在任何样本生成前因 Ray 临时目录过长触发 AF_UNIX 路径限制。将该次 eval 专用的 `RAY_TEMP_DIR` 改为 `/nfs/LOCAL/rq38e80` 后重新启动并完整通过；这只改变 Ray 本地运行目录，没有改变模型、数据、采样或计分协议

DataV4 部署期间还发现 `gather_convert_deploy.sh` 通过 stdin 远程执行时，嵌套 ssh 会吞掉后续脚本文本，导致首次向 node53 同步为空目录。已给嵌套 ssh 增加 `-n` 并重新部署，文件清单、大小和转换结果全部通过核对，没有删除或覆盖源 checkpoint
