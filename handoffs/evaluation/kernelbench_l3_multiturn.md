# KernelBench L3 多轮评测与失败机制

这轮最值得优先处理的是交付完整性、跨文件接口一致性，以及数值执行环境的披露。Qwen 经常在优化已经正确的实现时交付不完整正文；DeepSeek 较少卡在交付阶段，但会反复写错 launcher 的声明、定义和调用之间的对应关系。进入运行阶段后，错误又分成真正的越界、仍未完成的语义实现，以及 reference 数值模式不一致，单看 Wrong Answer 或最大绝对误差无法区分

受控 replay 已验证：补一处 C linkage、补一个 shared-memory 读边界、对齐一次 cuBLAS TF32 math mode，都能让对应 Qwen 候选通过正确性检查；DeepSeek 一个连续编译失败的候选，只改 launcher 原型就能进入数值检查，但随后同时暴露 Wrong Answer 和 ATen policy violation。另一个 DeepSeek attention 候选需要依次修正 GEMM 契约、batch 覆盖与 head layout，才最终正确，完整验证了多层错误逐级暴露的机制

本文件以 constructor 修复后的完整 dump 为主要证据；较早三模型观测和无效实验的排除原因见文末。正文不分析 Fast@x；所有比例在表头标注单位，数值保留两位小数。T1–T5 表示人类阅读轮次，原始 `turn_idx` 为 0–4

## 推理与评测配置

| 配置 | Qwen3.8-27B-FP8 | DeepSeek-V4-Flash-0731 |
| --- | --- | --- |
| 权重 | 官方 release FP8 | 官方 release 0731 FP8 |
| reasoning effort | medium | low |
| 历史 | exact token-prefix | 官方 chat template 重渲染 |
| serving | 2 × TP4，engine 并发 32 | 2 × TP4，engine 并发 32 |
| KV cache 请求/解析值 | auto / auto | auto / fp8_e4m3 |
| speculative decoding / eager | 关闭 / 关闭 | 关闭 / 关闭 |
| 数据 | KernelBench L3，50 题 | 同左 |
| 每题 rollout / 最多轮数 | 8 / 5 | 8 / 5 |
| 每轮最大新增 response | 32768 token | 同左 |
| 累计 context 上限 | 32768 / 65536 / 98304 / 131072 / 163840 | 同左 |
| temperature / top-p / top-k | 1.0 / 1.0 / -1 | 同左 |
| KernelGym | 已部署 dev_csl，原 URL `http://127.0.0.1:20211` | 同左 |
| task / client / generate guard timeout | 300s / 2400s / 18000s | 同左 |
| Runtime Sanitizer | error_based，每 check 独立 60s | 同左 |
| 返回候选 | finalize_mode=none，继续至五轮 | 同左 |

实际输入是 `Data/kernelbench-level3-validation-tvm-v2/train.parquet`，SHA256 为 `6b3c85f2e57f307036b8c38ff26891e8b88ac44afb702a7c7f539d2b8307755e`，没有使用候选 `prompt_v4_1` 数据。两份 config 的数据 hash 和主要 agent 代码 fingerprint 一致；分析时的 `generate_with_cuda_agent.py`、`utils.py`、`config.py` hash 与保存的运行 fingerprint 一致，因此该次分析可以按原实现复建 model feedback；后续复现必须使用保存的代码版本，不能直接假定当前工作区仍与旧 fingerprint 相同

完整配置、运行标识和 dump 位于[本轮实验目录](../../local_artifacts/official_l3_retest_postconstructor_20260904/run_manifest.json)，逐模型配置为 [Qwen](../../local_artifacts/official_l3_retest_postconstructor_20260904/qwen38/eval_config.json) 和 [DeepSeek](../../local_artifacts/official_l3_retest_postconstructor_20260904/dsv4/eval_config.json)

## 数据范围与结论边界

两模型均保留完整五轮 trajectory，题目、rollout、turn coverage 均为 100.00，missing `env_result` 与 aborted 均为 0.00。所有 turn 参与统计；人工检查覆盖交付失败、反复编译失败、Correct→Fail→Correct、持续 Wrong Answer、sanitizer、policy 和断连样本，保留完整参考实现、各轮源码与重建 feedback

但结果中仍有已知 submit disconnect：Qwen 和 DeepSeek 分别影响全部 trajectory 的 1.00 和 2.50。之前启动的整题补跑已停止，没有产生可替换的完整 dump。本文件保持原始 observed 指标，冻结候选 replay 单独作为证据；不把 replay 后的分数填回旧轨迹，也不把旧后续轮视为正确反馈下的反事实结果

此前的[断连候选 replay](../../local_artifacts/official_l3_retest_postconstructor_20260904/disconnected_candidate_replay.json) 已发现 DeepSeek `g12 T5` 实际可以正确，而原 trajectory 从未记为 Correct；`g137 T4` 也可正确，但该轨迹已有其他正确轮。当前再次验证 `g181 T3`，返回 `RUNTIME_ERROR` 并保留候选触发 cuDNN 错误后退出的证据，符合用户已部署的 RESOURCE_ERROR 分类修复。候选失败与服务重启引起的断连不能合并为一个 infra 桶

## 总体结果与模型差异

| 模型 | Best Compile (%) | Best Correct (%) | Terminal Correct (%) | 至少一条 trajectory 正确的题目 (%) | 已至少编译一次的轨迹中 Best Correct (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8 | 84.75 | 38.25 | 19.50 | 82.00 | 45.13 |
| DeepSeek | 97.00 | 43.50 | 21.75 | 76.00 | 44.85 |

DeepSeek 有更高的编译覆盖；在已经至少编译一次的轨迹中，两者 Correct 比例接近。但这个条件样本经过各自能力筛选，不能据此断言两模型语义能力相同。Qwen 覆盖更多题目，DeepSeek 在部分题族上更集中地成功

| 对比：DeepSeek 减 Qwen | observed 差值 (%) | 题级配对 bootstrap 95% CI (%) |
| --- | ---: | ---: |
| Best Compile | +12.25 | [+7.00,+18.25] |
| Best Correct | +5.25 | [-2.75,+13.50] |

bootstrap 以题为 cluster 重采样，每题已观测到的 n=8 比率固定，因此不额外模拟题内 rollout 噪声。该区间也没有消除断连反馈、不同 history/reasoning 设置或题族组成的影响，不构成纯模型能力或 sanitizer 因果效应的估计

| 题族 | Qwen Best Compile (%) | Qwen Best Correct (%) | DeepSeek Best Compile (%) | DeepSeek Best Correct (%) |
| --- | ---: | ---: | ---: | ---: |
| MLP | 100.00 | 95.83 | 100.00 | 75.00 |
| NetVLAD | 100.00 | 12.50 | 100.00 | 81.25 |
| CNN | 81.50 | 45.50 | 94.50 | 49.50 |
| RNN | 95.00 | 36.25 | 100.00 | 48.75 |
| Transformer/attention | 68.75 | 12.50 | 98.44 | 7.81 |
| Mamba | 100.00 | 0.00 | 100.00 | 0.00 |

两模型都至少解决的题目占 72.00，仅 Qwen 解决占 10.00，仅 DeepSeek 解决占 4.00，两者均未解决占 14.00。Qwen 独有成功包括 DenseNet201、EfficientNetB2、LSTMBidirectional、MiniGPTBlock、ReLUSelfAttention；DeepSeek 独有成功集中在两道 bidirectional GRU。单个平均分掩盖了这些互补性

## 多轮搜索没有保住已经获得的正确解

四个队列互斥，分母为每个模型的全部 trajectory；`keep-correct` 包括首次在 T5 才正确、尚无继续退化机会的轨迹

| 模型 | 从未 Compile (%) | Compile 过但从未 Correct (%) | Correct 后至少失败一次 (%) | 首次 Correct 后保持 Correct (%) |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8 | 15.25 | 46.50 | 27.25 | 11.00 |
| DeepSeek | 3.00 | 53.50 | 31.50 | 12.00 |

| 现象与分母 | Qwen (%) | DeepSeek (%) |
| --- | ---: | ---: |
| 曾正确后失败 / ever-correct trajectory | 71.24 | 72.41 |
| 最后一轮失败 / ever-correct trajectory | 49.02 | 50.00 |
| Correct→Fail→Correct / 全部 trajectory | 9.75 | 12.25 |
| 退化后又恢复 / 曾正确后失败的 trajectory | 35.78 | 38.89 |
| 首次失败后一轮即恢复 / 发生恢复的 trajectory | 89.74 | 85.71 |
| 恢复后最终又失败 / 发生恢复的 trajectory | 12.82 | 20.41 |
| 恢复为历史某个正确版本，忽略注释与格式 / 发生恢复的 trajectory | 0.00 | 2.04 |

一次恢复并不保证 T5 仍正确，恢复后再失败的轨迹也计入 terminal loss。相邻 Correct 轮之后仍为 Correct 的比例只有 44.39 / 39.21；保存历史正确候选有直接观测依据。至于约束编辑范围是否能提高 Best Correct，需要另做实验

恢复通常没有原样取回历史实现。按实际 parser 提取三段源码，忽略 C/C++ 注释和空白、Python 按 AST 比较，同时保留 include、字面量与运算符后，只有 DeepSeek `g15` 回到一个历史正确版本；它回到的还是较早版本，非当时历史最佳候选。按源码逐字比较，两者恢复比例都为 0.00

仅观察真实 cuBLAS/cuDNN 调用是否形成 A→B→A，可以确认 DeepSeek `g1/g84/g151/g269/g292` 存在库调用路径来回切换，占所有恢复轨迹的 10.20、全部 trajectory 的 1.25；Qwen 该判据下为 0.00。这只是可复算的窄判据，不包括两种自写 CUDA 算法之间的切换，不能作为所有实现切换的完整频率

实现恢复按[源码一致性结果](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/recovery_code_identity.json)和下面逐例的代码证据

## Qwen3.8-27B-FP8

## 主要损失发生在交付和重新交付

| 全 turn 结果 | 比例 (%) |
| --- | ---: |
| Precheck failure | 34.70 |
| Compilation/load failure | 26.20 |
| Wrong Answer | 15.45 |
| Correct | 14.15 |
| Runtime failure | 6.00 |
| Decoy/policy failure | 3.30 |
| Infrastructure disconnect | 0.20 |

| Precheck cause | 桶内比例 (%) | cause 的具体含义 |
| --- | ---: | --- |
| missing_or_invalid_modelnew | 91.50 | parser 没有得到包含 ModelNew 的完整 Python section |
| framework_compute_bypass | 4.03 | 实际 PyTorch tensor compute 被禁止 |
| tvm_ffi_host_cuda_leak | 1.87 | host binding 带入 CUDA runtime header/type |
| python_syntax_error | 1.44 | 已提取的 Python 正文存在语法错误 |
| tvm_ffi_call_not_exported | 1.01 | Python 调用与 native export 名称不对应 |
| fallback_or_try_bypass | 0.14 | 不符合协议的 fallback/try 路径 |

全部 trajectory 中，74.25 至少一轮出现 `missing_or_invalid_modelnew`。这些失败里，61.73 没有生成 `</think>`；36.38 的原始文本已经出现 `class ModelNew`，但 parser 没有提取到可用的完整 MODEL_NEW。这两个现象可能重叠，不能相加

“没有 ModelNew”经常是交付没有完成。`g33 T3` 已写到 AlexNet 后续层调用，停在 `self.conv5`；`g48 T3` 已写出 ModelNew，停在 `nn.Max`；`g233` 五轮都被这一类别拒绝，实际停止位置依次落在规划文字、binding、规划文字、Python 和 CUDA。相同的错误标签覆盖了不同的中断位置，重写整个网络不断重新付出前半段的 token 成本

`g0 T3` 已正确，T4 花较长正文讨论 GEMM tile 与线程映射，却没有结束 thinking、没有提交三段代码，T5 才重新完成交付。它属于“尝试优化后没有交付候选”，不能说成一个新的 CUDA 实现已经编译后做错。首次 Correct 后退化的 Qwen 轨迹中，53.21 就落在 missing ModelNew 这一类

保存的 engine length-truncated turn 比例为 0.15；但缺少 ModelNew 的样本中，99.37 的文本以 `<|im_end|>` 结束，其他还有 `<|endoftext|>` 或真正 length termination。生成代码直接保存 `output['text']`，并按 engine 的 length 类型设置 turn status；后处理的 `finish_reason=max_turns` 是整条轨迹结束原因，无法还原更细的每请求停止信息。因此可以确认低“length 截断率”与严重交付不完整同时存在，尚不能仅凭 dump 把提前终止归因到模型、FP8 或 serving 的某一环节

这也解释了为何继续增加累计 context 并不自动解决问题：T2–T5 的 missing ModelNew turn 比例依次为 24.50、22.75、27.75、32.50，后期重新变高。可观察到优化反馈后重新规划和大段生成，未验证“历史越长导致退化”的因果关系

可复核源码与正文：[g0](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/manual_review/qwen38_g000/)、[g33](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/manual_review/qwen38_g033/)、[g48](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/manual_review/qwen38_g048/)、[g233](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/manual_review/qwen38_g233/)

## 编译接口失败中，有可一次局部修复的错误

以下分母为 evaluator 标记的 compilation/load failure；primary cause 是互斥的规则归类，存在多个诊断时不代表该 cause 是唯一 bug

| cause | 桶内比例 (%) | 具体原因 |
| --- | ---: | --- |
| tvm_ffi_api_or_binding | 45.23 | DLDataType、dtype/stride accessor、FFI tensor 契约用错 |
| type_or_signature_mismatch | 19.27 | 参数数量、顺序、指针或 descriptor 类型不匹配 |
| undefined_identifier | 13.55 | 拼错或当前库中不存在的 API/enum |
| linker_or_missing_symbol | 7.82 | 声明/定义、C linkage、符号对应关系错误 |
| other_compiler_diagnostic | 7.25 | 其余已识别的 native 编译诊断 |
| missing_header_or_include | 2.67 | include 路径或 header 不存在 |
| resource_limit | 1.15 | 编译器报告寄存器或 shared-memory 资源限制 |
| Python construction/import | 0.57 | 未定义名称或未调用 Module 初始化，被服务归入 compilation |
| cuda_api_or_type_misuse | 0.57 | CUDA API/type 用法不符 |
| duplicate_or_redefinition | 0.57 | 重复声明或定义 |
| syntax_or_delimiter | 0.57 | native 语法/分隔符错误 |
| missing_standard_header | 0.57 | 使用 printf 等符号但缺相应标准头 |
| host_device_boundary | 0.19 | host 与 device 调用边界错误 |

`g0 T1` 同时含 `cuBLASHandle_t` 拼写、DLDataType 比较和 stride accessor 问题，T2 修过之后才进入 linker，暴露 `.cpp` 用 `extern "C"` 声明而 `.cu` 普通 C++ 定义的问题。冻结 T2 在当前服务重放，仍是 undefined reference；只给 `.cu` 定义增加 `extern "C"`，正确性全部通过。该改动没有替换 GEMM 算法，证明这个候选当时已经只差一处 linkage 修正

[原候选 replay](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/replay/qwen38_g0_t1_frozen.result.json)与[单点修复 replay](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/replay/qwen38_g0_t1_extern_c_only.result.json)把“语法/API 修复后才看到 linker”的顺序与“局部修复已经足够”区分开来。一次 compiler 日志可以有多条诊断，但 linker 和运行时的潜在错误仍可能要等前级通过才出现

## 做对后又做错：优化引入的 API 和内存错误

`g8` 在修掉 binding 后进入 illegal access，随后 T3/T4 正确；T5 为改用 cuBLASLt 引入不存在的 `CUBLASLT_EPILOGUE_BIAS_RELU`，再次编译失败。`g42` 的 T2 正确，T3 在调整 convolution algorithm 时引入 `CUDNN_SEARCH_FAST` 与不可用 API，T4 恢复，T5 再优化时 Wrong Answer。它们表明历史正确解不能约束下一份完整重写中的接口选择

`g28` 的路径更具体：T1 正确 → T2 未完成交付 → T3 正确 → T4 未完成交付 → T5 shared-memory 越界。T5 的 `fc_tiled_kernel` 用 `ceil(M/8)` 为线程分配多个输出通道，在输出写入时检查了 `m < M`，却在 `s_w[m * W_PAD + k]` 读取时没有检查，尾部线程先越界再到写保护

sanitizer 给出 `generated.cu:176` 的 invalid shared read，并定位到尾部线程。冻结候选复现相同错误；只在这次 shared-memory read 前加 `m < M`，全部正确性检查通过。这是定位有效且可直接验证的 sanitizer 反馈，不必重新采样整份 LeNet

证据：[g8](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/manual_review/qwen38_g008/)、[g42](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/manual_review/qwen38_g042/)、[g28 T5 源码](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/manual_review/qwen38_g028/T5_generated.cu)、[边界修复 replay](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/replay_local_contracts/qwen38_g28_t4_shared_read_bound_only.result.json)

## 五轮 Wrong Answer 中，存在数值模式已经成为唯一阻碍的候选

`g394` 的 ReLUSelfAttention 五轮都是 Wrong Answer，max difference 从几百降到约 0.135 后停住。T4 已实现完整 QKV、causal mask、ReLU attention 与输出合并，T5 也主动怀疑浮点误差，但继续修改 reduction 没有改变结果

原始 metadata 明确记录 `fp32_math_mode=tf32`、`correctness_tf32_enabled=true`。候选 cuBLAS handle 没有设 TF32 math mode。受控对照保持原始 reference、输入协议和候选其他代码不变，仅加入 `cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH)`，结果从 Wrong Answer 变为全部正确性检查通过

这证明该 frozen candidate 的判定对 math mode 敏感。原 forward 逻辑并未在这次实验中改动，不能继续把它列成“attention 算法仍未实现”。这里也不推断所有相近误差的实现都正确：ReLUSelfAttention 中，Qwen 和 DeepSeek 分别有 75.00、87.50 的 trajectory 至少一轮落在相近最大误差区间，但其他候选仍需独立验证

关键反馈缺口可以闭环：首轮 prompt 没有披露 TF32；实际保存的 metadata 有该信息；与运行 hash 一致的 `build_model_feedback` 又没有保留这两个字段。因此该样本下一轮看得到最大/平均误差，看不到正在匹配的 reference math mode。披露有效 math mode、dtype 和容差属于环境事实，符合此前“不提供修复建议”的要求

证据：[原 model feedback](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/manual_review/qwen38_g394/T4_model_feedback.json)、[冻结候选](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/replay_precision/qwen38_g394_t3_frozen.result.json)、[仅 TF32 改动](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/replay_precision/qwen38_g394_t3_tf32_math_mode_only.result.json)

## Runtime failure 还包含模型容器与参数布局错误

| cause | Runtime 与 infrastructure 桶内比例 (%) | 具体原因与边界 |
| --- | ---: | --- |
| python_wrapper_exception | 44.35 | 属性访问、wrapper 调用或 FFI 断言失败，需要继续查看异常正文 |
| illegal_memory_access | 25.81 | 已观察到非法地址或相应 sanitizer 证据 |
| shape_or_type_exception | 16.13 | 输入、参数、输出的 shape/type 契约不符 |
| worker_native_crash | 9.68 | 子进程退出，部分有候选库错误；不能仅按 crash 名称判断 infra |
| service_disconnect | 3.23 | HTTP 提交断连，独立于候选正确性 |
| runtime_timeout | 0.81 | 候选运行未在指定期限内完成 |

例如 `g326 T3` 把 GRU 的命名属性 `weight_ih_l0` 等当成 `weight_ih_l[layer]` 数组访问，触发明确的 AttributeError。同一份 Python 还用 `h0[layer]` / `h0[L + layer]` 划分双向 hidden state，而层与方向通常应交错索引；后者是代码审查发现的后续风险，尚未 replay 验证，不能把它计成已暴露根因

这类错误解释了“写了 state_dict-compatible module”为何仍不足以保证运行正确：参数存在、如何找到参数、如何打包到 native ABI，是不同的约束。[g326 源码](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/manual_review/qwen38_g326/T3_model.py)

## DeepSeek-V4-Flash-0731

## 能持续交付代码，主要困难转移到运行与语义

| 全 turn 结果 | 比例 (%) |
| --- | ---: |
| Compilation/load failure | 33.45 |
| Wrong Answer | 22.75 |
| Runtime failure | 17.25 |
| Correct | 15.70 |
| Decoy/policy failure | 5.45 |
| Precheck failure | 4.90 |
| Infrastructure disconnect | 0.50 |

| Precheck cause | 桶内比例 (%) | cause 的具体含义 |
| --- | ---: | --- |
| tvm_ffi_host_cuda_leak | 35.71 | host binding 引入 CUDA runtime header/type |
| framework_compute_bypass | 22.45 | 已定位到实际 PyTorch tensor compute |
| missing_or_invalid_modelnew | 11.22 | Python section/class 没有按契约交付 |
| tvm_ffi_missing_extension_call | 11.22 | 没有可识别的 extension 调用 |
| tvm_ffi_missing_export | 11.22 | 没有可识别的 native export |
| python_syntax_error | 4.08 | 提取后的 Python 有语法错误 |
| tvm_ffi_call_not_exported | 3.06 | 调用与 export 名称不匹配 |
| tvm_ffi_missing_header | 1.02 | binding 缺少要求的 TVM-FFI header |

这轮检查到的 framework 拒绝已经能指向 `torch.sqrt`、softmax、layer_norm 等实际调用；合法 ReLU/BatchNorm constructor 也出现在正确候选中。以 `g62` 为例，T3 正确，T4 为 fold BN 在 Python `_fold_bn` 中计算 `torch.sqrt` 而被拒绝，T5 又正确。它是优化跨过了已明确的 compute policy 边界，与此前 constructor 误杀的因果链不同

## 连续五轮编译失败：声明少一个参数，而且越改仍少一个

| cause | Compilation/load 桶内比例 (%) | 具体原因 |
| --- | ---: | --- |
| type_or_signature_mismatch | 22.42 | launcher、cuDNN/cuBLAS 参数数量与类型不符 |
| undefined_identifier | 18.39 | 不存在或不可见的 API、enum、kernel 名称 |
| tvm_ffi_api_or_binding | 15.25 | tensor accessor、dtype/device、FFI 契约错误 |
| missing_standard_header | 14.20 | printf/fprintf/stderr 等错误处理代码缺标准头 |
| other_compiler_diagnostic | 9.87 | 其他 native 编译诊断 |
| linker_or_missing_symbol | 8.97 | 声明、定义或 linkage 不对应 |
| missing_header_or_include | 4.04 | header/path 不存在 |
| Python construction/import | 3.14 | 缺 helper class、外部模块或构造错误，被归入 compilation |
| duplicate_or_redefinition | 1.79 | 重复声明/定义 |
| syntax_or_delimiter | 0.75 | native 语法错误 |
| template_or_macro_misuse | 0.60 | template/macro 契约错误 |
| host_device_boundary | 0.30 | host/device 调用边界错误 |
| cuda_api_or_type_misuse | 0.15 | CUDA API/type 错误 |
| resource_limit | 0.15 | 编译器报告资源限制 |

按 evaluator 标签，五轮都 compilation/load failure 的 trajectory 占 1.25，分别为 `g54/g123/g124/g144/g356`；其中 `g54 T5` 是 `InceptionModule` 未定义，已经属于 Python construction，因此严格“五轮都有 native compiler/linker 失败”的比例是 1.00

`g123` 最能解释“模型看懂报错，却一直没修好”。CUDA 定义和 binding 调用都包含完整的 dilation 参数与 stream，binding 的前置声明始终少一个 `int`。编译器按当前 translation unit 的声明检查，后面的整数于是落到 `void*` 参数位，同时产生 `invalid conversion from int to void*` 和 `too many arguments`

T4/T5 的 reasoning 已经明确讨论参数数量，并逐项数 input、weight、output、尺寸、dilation、stream；最终输出仍保留少一个参数的声明。问题落在“理解 diagnostic → 同时更新三份接口描述”的执行一致性上，单纯再提醒一次同样的错误未必有帮助

冻结 `g123 T5` 再次失败；只把 binding 中这一条声明替换成该候选自身 CUDA 定义的签名，编译即通过，随后出现数值不匹配与 `aten::cat` policy violation。replay 直接证明了两个层次：到 T5 时的 native 编译失败只需修正这一条声明就能解除；声明修好后仍远未完成题目。早期轮次还包含其他编译错误，不能由这个对照倒推它们都只有这一处问题

`g124` 也有相同 ABI 形态；独立审查还在 `g356` 观察到模型删减 dilation 参数后，声明继续比定义/调用少一项。对全部 DeepSeek compilation/load turn，`int→void*` 与“too many arguments”共同出现占 8.67；这是诊断组合频率，尚未逐个证明都由同一种 dilation 漏项导致

证据：[g123 完整轨迹](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/manual_review/dsv4_g123/)、[原型单点修复结果](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/replay_local_contracts/dsv4_g123_t4_conv_prototype_only.result.json)

## 相同 runtime message 后面，可以连续藏着不同 GEMM 错误

`g399` 的 ReLUSelfAttention 在 T2–T4 都返回 `cuda_attention_forward failed`。人工读 CUDA 可见它在修改 QKV projection 的 cuBLAS transpose/leading dimension，却没有一致地重建后续 QKᵀ 和 attention×V 的 row-major/column-major 对应关系

受控对照使用冻结 T2，每次只改变列出的代码或 math mode

| 候选 | 当前服务 replay 结果 | 能确认的机制 |
| --- | --- | --- |
| 原 T2 | Runtime error | QKV SGEMM 的参数契约不符，原始 sanitizer 输出含 parameter 8 错误 |
| 只修 QKV projection | 相同 wrapper Runtime error | 前级修复后，后续 GEMM 契约仍不符 |
| 修正三个 GEMM 的维度、transpose 和 leading dimension | Wrong Answer | native 调用能够完成，仍有数值/布局层面的错误 |
| 上述版本再对齐 TF32 math mode | 仍 Wrong Answer | 此候选剩余问题不能用 Qwen 的 TF32 单因解释 |
| 再修正 split/merge kernel 的 batch 覆盖 | 仍 Wrong Answer | 完整 batch 被处理，但 Q/K/V layout 仍与 GEMM 不符 |
| 再修正 Q/K/V 的 head-major 存储布局 | 全部正确性检查通过 | 同一 frozen candidate 的完整错误链闭合 |

后续人工检查定位到两个原始 runtime error 遮挡的具体 bug。split 和 merge 的 launcher 按整个 batch 发线程，kernel 内部却只用 `T * C` 作为总量并提前 return，后续 batch 的输出没有写入；Q/K/V 又按 `(B,T,head,dim)` 连续写入，后面的 batched GEMM 按 `(B,head,T,dim)` 读取。只修 batch 覆盖仍不正确，再对齐这个 layout 才通过全部检查

因此这份完整代码里同时存在多处 GEMM 参数错误、batch 覆盖错误和布局错误，早期却统一表现为 wrapper 的 `cuda_attention_forward failed`。与 Qwen `g394` 的 math-mode-only 成功放在一起看，能够避免把所有 attention 错误统一归因于 precision，也能说明为何仅反复修日志中的第一处错误会耗掉多轮

还有一个具体反馈缺口：sanitizer 的原始输出保留了 SGEMM 参数错误，压缩后的 model feedback 丢弃 raw output，主要留下 wrapper 失败和各 check 的 clean/target-failed 状态。对 `g399 T2` 重建的 feedback 中没有 SGEMM 参数信息，下一轮难以知道哪个 native 调用首先失败。适合保留的是已经发生的库调用错误事实，无需添加修复建议

证据：[g399 源码和 model feedback](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/manual_review/dsv4_g399/)、[仅 projection 修复](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/replay/dsv4_g399_t1_fix_projection_only.result.json)、[三个 GEMM 修复](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/replay/dsv4_g399_t1_fix_all_three_gemm_contracts.result.json)、[再修 batch 与 head layout 后正确](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/replay_attention_dataflow/dsv4_g399_t1_gemms_tf32_batch_and_head_layout.result.json)

## Correct→Fail→Correct：有回退，也有改好后再次引入错误

`g1` 的 MLP 在 T3 用自写 tiled GEMM 正确；T4 因速度反馈换为 cuBLAS 路径，出现 illegal access；T5 又回到自写 GEMM并正确。五轮的 ModelNew 相同，主要变化在 CUDA，所以“Python/binding 没改”不能推出这是局部 kernel 修复

`g15` 在更早的正确实现和后来的优化实现之间，最终恢复了较早版本；规范化后的源码与历史版本相同。这个例子支持“有回退到旧实现”，但它在全部恢复中只占很小一部分

`g42` 的 T2 正确，T3 换 cuDNN 后输出不匹配，T4 修复 output slice/descriptor 后正确，T5 又切到自写实现并因缺 `FLT_MAX` 声明而编译失败。这里恢复轮保留了新路径继续修复，并非回到首次正确版本；随后再次优化又丢掉了 terminal correctness

这些行为有共同诱因：协议每轮都要求 full improved implementation，正确反馈中仍提供性能与 profiling，模型继续尝试优化。原始 reasoning 明确写了切换理由，能够确认样本层面的优化动机；不能从这种观察直接推出某个固定采样温度或特定 history 模式导致了退化

证据：[g1](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/manual_review/dsv4_g001/)、[g15](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/manual_review/dsv4_g015/)、[g42](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/manual_review/dsv4_g042/)

## 持续 Wrong Answer 需要区分变化的错误与数值不稳定

DeepSeek 五轮均 Wrong Answer 的 trajectory 占 2.00，包括 VisionAttention、LSTM、GRU、Mamba 与 ReLUSelfAttention。相邻 Wrong Answer 的最大误差有时下降、有时反弹；即使数值相同，也只能说明同一次输入/比较上最坏误差相同，不能证明输出相同或源码未变

`g247` 的 VisionAttention 曾把最大误差从约 2.19 降到约 0.00113，随后两轮又回到前者。`g377` 的 Mamba 后两轮已明确实现 diagonal/off-diagonal 项、chunk state 更新与输出，最大误差仍很大。reference 本身对随机 A 做累积求和后取 exp，状态数值可能被放大；没有 reference/output 的尺度、归一化误差和中间张量比较，不能把巨大绝对误差直接解释为“只返回垃圾”或“缺整段 recurrence”

Mamba 两模型的 Best Compile 均为 100.00、Best Correct 均为 0.00，足以确认困难位于 native 构建以后。现有证据还不足以为其全部 Wrong Answer 分配经过验证的语义/数值根因比例。尤其 `initial_states` 未支持这类代码风险，若本次 get_inputs 没有触发该分支，就不能记作已发生的失败原因

证据：[g247](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/manual_review/dsv4_g247/)、[g377 reference 与代码](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/manual_review/dsv4_g377/)、[所有持续 Wrong Answer 轨迹](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/statistics.json)

## Runtime failure 的具体组成

| cause | Runtime 与 infrastructure 桶内比例 (%) | 具体原因与边界 |
| --- | ---: | --- |
| illegal_memory_access | 32.11 | CUDA 非法访问及其 sanitizer 证据 |
| worker_native_crash | 30.42 | 子进程异常退出；需结合候选库错误、exit/abort、stderr 判断 |
| python_wrapper_exception | 23.94 | 模型属性、FFI、native 返回值断言等 Python 层异常 |
| shape_or_type_exception | 8.17 | 参数维度、stride、shape 或 Python/native 参数契约不符 |
| service_disconnect | 2.82 | 提交连接中断，与候选算法分开归类 |
| runtime_timeout | 2.54 | 候选未在期限内完成 |

`g181 T3` 的 stderr 在 worker 退出前给出 `CUDNN_STATUS_BAD_PARAM`，同轨迹后续退出栈还包含候选 conv2d 中的 exit 与 profiler 析构。当前服务 replay 将其归为 `RUNTIME_ERROR`，仍然能复现候选失败。这个案例不能作为“KernelGym 又出现资源错误”的证据；CUPTI/profiler 栈同时出现也不能取代更早的候选库错误作为已验证根因

## 反馈能提供什么、还缺什么

## 多个错误共存，单一 outcome 会丢掉第二条修复线索

| 现象与分母 | Qwen (%) | DeepSeek (%) |
| --- | ---: | ---: |
| 至少两条不同的真实 compiler/linker 诊断 / compilation-load turn | 63.55 | 64.42 |
| 相邻均编译失败且诊断集合相同 / 相邻均编译失败 | 0.95 | 8.93 |
| 相邻均编译失败且至少共享一条诊断 / 相邻均编译失败 | 23.81 | 32.14 |
| 五轮同一失败细类 / 全部 trajectory | 5.00 | 2.25 |
| 同时有 output mismatch / decoy turn | 71.21 | 72.48 |

编译器常在一次失败里返回多条诊断；这些诊断又可能是一个根因的级联结果，例如额外整数参数同时触发类型和数量错误。因此“日志有多个 error”不等于“多个独立 bug”，也不能笼统说 GYM 每次只返回一个 compiler error

真正的串行遮挡发生在阶段之间：结构不完整时还没编译、编译失败时还没运行、native 调用失败时还没比较输出。`g123` 的原型修复与 `g399` 的分阶段修复已经通过 replay 验证这种遮挡。同时 policy 检查与 output comparison 也会共同留下结果，单个 `decoy` 标签会隐藏仍然存在的数值错误

例如 Qwen `g307 T3` 同时存在数值不匹配与 `aten::t` 禁止。把 `.t()` 等价改写为 `.transpose(0, 1)` 后，当前 evaluator 不再标记 decoy，但仍是 Wrong Answer。这个对照证明当前 allowlist 对这两种转置写法存在判定差异，也证明放行该操作不会自动修好数值结果。纯 `aten::t` 禁止占 Qwen/DeepSeek 全 turn 的 0.10 / 0.25，不能用这个窄问题解释整体 Correct 差距

[转置拼写对照](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/replay/qwen38_g307_t2_transpose_spelling.result.json)

## Sanitizer 的诊断能力取决于它实际观测到了什么

| sanitizer 状态，占全 turn | Qwen (%) | DeepSeek (%) |
| --- | ---: | ---: |
| 没有 sanitizer metadata | 61.80 | 44.85 |
| skipped | 36.70 | 49.85 |
| issues_found | 1.00 | 3.45 |
| error | 0.50 | 0.75 |
| clean | 0.00 | 1.05 |
| partial | 0.00 | 0.05 |

error_based 没有触发的候选不能计作 sanitizer clean；普通 Wrong Answer 也不一定触发检查。两模型进入 sanitizer 的样本经过前级失败类型筛选，不能用上述触发率比较真实内存安全能力

| issues_found 里的诊断质量 | Qwen (%) | DeepSeek (%) |
| --- | ---: | ---: |
| 至少一条 issue 含源码位置 | 30.00 | 8.70 |
| 原始工具输出还含 Internal Sanitizer Error | 70.00 | 69.57 |

Qwen `g28` 是有源码位置且可局部修复的例子；Qwen `g8 T2` 和 DeepSeek `g1 T4` 则主要返回 cudaLaunchKernel/cudaDeviceSynchronize/cudaFree 等 API 上的 illegal-address 信息，没有给出第一处非法内存指令。对 `g8 T2` 当前 replay 仍有这类工具内部错误及 API 诊断，说明“启用了 sanitizer”并不保证得到源码级定位。kernel filter、vendor library 与工具异常可能参与其中，现有证据不能单独证明是哪一个造成定位缺失

DeepSeek 的 clean 结果中，33.33 至少一个 check 同时标记 `target_application_failed=true`。`g399` 的检查可以零 memory error，但应用已经因 SGEMM 参数错误提前返回，因此 `clean` 只描述被执行的 sanitizer 检查结果，不能当成整个程序完成、数值正确或所有路径安全的证明。当前 model feedback 保留了 target-failed 字段，这部分信息没有丢失

输入重放也需要按 check 区分：这轮 DeepSeek 的 memcheck/synccheck/racecheck 记录声明精确重放，出现的 initcheck 则都声明 `input_values_exactly_replayed=false`，使用 CPU 生成后传到 GPU 的路径。这个区别已经保留在 model feedback 中；initcheck 的 clean 不能作为同一个失败输入上没有问题的反证

## 有证据支持的后续改进

| 优先处理项 | 证据 | 可披露的事实或可验证的改动 |
| --- | --- | --- |
| 数值执行环境 | Qwen g394 仅对齐 TF32 就正确，原 feedback 未披露该模式 | 有效 dtype、reference math mode、atol/rtol、比较方式；不附“应该如何修改”的建议 |
| 交付终止原因 | Qwen 多处正文在代码中结束，trajectory finish_reason 无法定位 | 每请求 engine finish_reason、stop token、实际 max_new_tokens、prompt/output token 数；将交付失败与 length termination 分开统计 |
| 跨文件 ABI 的事实反馈 | DeepSeek g123 理解错误但连续输出少一项的声明 | 对应 declaration/definition/call 的源位置、参数列表；若扩展检查需单独评估复杂度，当前未实现 |
| Native 首个错误保留 | DeepSeek g399 wrapper 消息相同，原始 SGEMM 错误没有进入 feedback | 库调用失败的函数、status/参数编号、实际源码位置；不靠通用 wrapper exception 代替 |
| 历史正确候选保存 | 约一半 ever-correct 的 terminal 已失败 | 独立保存经过验证的 incumbent；另外实验验证编辑策略，不把它的收益写成已测结果 |
| Sanitizer 有效性边界 | issues_found 大量缺源码；clean 可伴随应用提前失败 | 并列保留检查完成度、工具错误、target failure；不要把某个 clean 布尔值升级成全局结论 |
| 同时保留数值与 policy 结果 | 大多数 decoy turn 还包含 output mismatch | 维持多维结果，避免只用主 error type 告诉模型“唯一问题已定位” |

这些是原始评测的分析结论和后续实验入口。现在首轮模板和 v4.1 派生数据已加入 TF32 notice，详见[发版合同](../data/synthesize/training_data_release.md)；本文评测使用 v2 L3 数据，不能把后来的提示词或代码修复回填到这些结果中

## 复核与可复现产物

全量统计由[分析脚本](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/analyze.py)从逐轨迹完整证据重算，[statistics.json](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/statistics.json)保留计数与分母；完整 raw dump 不变。关键案例的[可读源码与 feedback](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/manual_review/)可以和原 response 对照，模型输入中的 JSON 仍按原代码无缩进，人工导出保留缩进

本次 replay 由[同一入口](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/replay_cases.py)执行，[基础对照](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/replay/manifest.json)、[数值模式对照](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/replay_precision/manifest.json)、[局部接口与边界对照](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/replay_local_contracts/manifest.json)、[attention 数据流对照](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/replay_attention_dataflow/manifest.json)均保存完整 request/result、原始/变体 response hash、唯一 task ID。变体只服务于因果诊断，不作为模型生成能力或新 rollout 精度

Grok-4.6-xhigh 完成了[只读独立审查](../../local_artifacts/official_l3_retest_postconstructor_20260904/deep_analysis/cross_review_result.json)。它复算了总体和 DeepSeek 队列，指出 Python construction 混入 compile、ABI 声明漏参、恢复判据与 parser 的偏差；这些发现已结合原始源码复核并纳入本文。审查中“巨大 Mamba 误差意味着垃圾输出”“未支持但未触发的 initial_states 是当前失败原因”和笼统“sanitizer 不能定位”的表述证据不足，本文未采用；保留了数值尺度未知和成功定位样本的反证


## 较早三模型观测的使用边界

下面是 2026-09-02 保存的原始评分：每模型 50 题 × 8 trajectory × 5 轮，Qwen3.8 为 medium、Qwen3.6 为 default、DeepSeek-0731 为 low。Qwen 使用 exact token-prefix，DeepSeek 重渲染历史。evaluator 为 `c936513`，当时没有 Runtime Sanitizer，且存在 receiver 无约束的 framework-compute 误报；这些数字用于定位旧实验，不作为修复后协议的 clean baseline

### 原始观测值

| model | Best Compile (%) | Best Correct (%) | Terminal Correct (%) | Correct Pass@8 (%) |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.8-27B | 84.50 | 39.50 | 22.75 | 76.00 |
| Qwen3.6-27B | 88.00 | 21.25 | 9.00 | 58.00 |
| DeepSeek-V4-Flash-0731 | 90.25 | 40.50 | 23.25 | 80.00 |

### Qwen3.6 的逐轮结果

| turn | Turn Compile (%) | Turn Correct (%) | Best Correct (%) | Truncated (%) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 20.75 | 3.25 | 3.25 | 3.25 |
| 2 | 50.50 | 6.50 | 8.75 | 0.00 |
| 3 | 66.75 | 6.75 | 13.50 | 0.00 |
| 4 | 69.25 | 9.25 | 18.00 | 0.00 |
| 5 | 72.00 | 9.00 | 21.25 | 0.00 |

较早三模型的逐题、逐 turn、题族和错误文本由 `local_artifacts/qwen38/official_l3_5turn_eval_20260901/combined_error_analysis/all_turn_outcomes.jsonl` 及同目录分析产物维护。DeepSeek 最终 merged dump 位于该实验目录的 `final_inputs/dsv4/dumps/rollout_data/eval_0.pt`。旧 Qwen3.8/Qwen3.6 原始 `.pt` 位于已结束的临时 node-local mount namespace，保留下来的 manifest、hash、audit 和分层正文只能支撑对应审计，不能声称仍可全量源码复放

## 被排除的实验与已确认的 evaluator 回归

- Constructor 污染重测：旧 AST checker 把 `nn.ReLU/GELU/LayerNorm/BatchNorm*` 构造误判为 compute。冻结响应重放中，Qwen 和 DeepSeek 各自 observed framework turn 的 96.52% / 95.95% 只含 constructor；这证明判定污染，不证明这些候选随后一定正确
- 错误 feedback 会改变后续整条轨迹。离线放行一个 frozen response 无法补出其 compile/runtime 结果，也不能恢复正确 feedback 下的后续 response，因此不使用简单 rescoring 修正五轮准确率
- 一份 DeepSeek generic-runtime dump 不满足该实验约定的 DSpark runtime。它只可用于冻结源码的 checker/telemetry 检查，已从模型精度比较排除
- 72 条 affected trajectory 的定向重跑仍有 constructor 误报：53 个 framework turn 中 39 个仅构造 module，14 个包含实际 tensor compute。rich source-location feedback 已送达，问题在 checker 判定。运行末段从 2×TP4 热插到 4×TP4，不能把总耗时视为 16 GPU 全程性能
- v4.1 的唯一 prompt 变化是 TF32 notice，数据协议由发版合同和实际 parquet hash 确定

Constructor 判定应区分实例化容器与对 tensor 执行算子。候选 forward 的 ATen profiler 仍须检查实际 framework compute；具体反馈 schema 统一见[模型反馈](../kernel_agent/feedback.md)

排除依据：`local_artifacts/qwen38/official_l3_5turn_sanitizer_retest_20260903/framework_constructor_false_positive_audit.md`、`local_artifacts/deepseek-v4/live_kernelgym_constructor_matrix_20260904*`；定向重跑的映射、配置、dump 与 `framework_recurrence_audit.json` 位于 `local_artifacts/dsv4_0731_framework_compute_rerun_20260904/`
