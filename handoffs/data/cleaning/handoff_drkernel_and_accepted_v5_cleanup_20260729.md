# 数据清洗交接：DrKernel、鸥波合成、外部开源

## 摘要

本文档记录三类训练数据候选的清洗过程和结果：DrKernel（主数据）、鸥波合成、外部开源（CUDA-Agent-Ops-6K 与 KernelBook），另附 KernelBench 官方测试集集的审计。
三类数据过的是同一套检查：先 CPU 静态检查，再 GPU runtime 检查，最后按 train/eval 表现划分，表现一致的称为 mode-same，不一致的称为 mode-variant；证据不足、暂时无法判定好坏的样本放进 quarantine 单独存档

各训练数据集的清洗结果如下：
| 数据集 | 原始样本数 | 过滤后有效样本 | mode-same |
| --- | ---: | ---: | ---: |
| DrKernel（主数据） | 71,996 | 45,505 | 40,427 |
| 鸥波合成 | 2,326 | 2,004 | 1,644 |
| CUDA-Agent-Ops-6K | 6,000 | 4,995 |
| KernelBook | 18,153 | 11,820 |

KernelBench 测试集集经逐题手动review，最终合格测试集如下：

| Level | 原始题数 | 过滤后最终测试集 | 存在的问题 |
| --- | ---: | ---: | --- |
| Level 1 | 100 | 100 | - |
| Level 2 | 100 | 93 | 删除 7 题：官方输入和容差下输出恒定或近恒定（如 GroupNorm 后取全局 mean、大 softmax 的变化落在容差内），常量替代就能通过正确性比较，可绕过核心计算 |
| Level 3 | 50 | 37 | 删除 13 题：12 题同为恒定/近恒定问题（ViT `batch_first` 误用导致 patch 输入不汇聚、MobileNetV2 输出约 `2.7e-21`、UNet 重复 softmax 路径不随输入改变等），1 题 `forward` 内随机初始化 LSTM 状态，oracle 不可复现 |

## DrKernel

### 数据漏斗总览

| 阶段 | 样本数 | 占比 |
| --- | ---: | ---: |
| 原始数据 | 71,996 | 100.00% |
| 通过静态检查 | 59,483 | 82.62% |
| GPU 执行通过 | 45,505 | 63.20% |
| mode-same | 40,427 | 56.15% |
| mode-variant | 5,078 | 7.05% |


### CPU 静态检查

上 GPU 之前，先在 CPU 上用 Python AST 扫一遍源码，避免浪费 GPU 算力。
静态检查共剔除约 12.5k 个样本，按 primary reason（每个样本只记第一个命中的致命原因）划分：

| 静态 primary reason | 样本数 | 规则 |
| --- | ---: | --- |
| `random_forward` | 3,798 | `forward` 里有未设种子的随机调用（dropout、采样、Gumbel-softmax、分数池化等） |
| `unused_forward_module` | 3,151 | `__init__` 里建的 `nn.*` 模块在 `forward` 里用不到 |
| `dead_forward_computation` | 2,747 | dead code（代码会执行，但结果被覆盖或丢弃、到不了返回值）：含调用的局部赋值属于此类 |
| `unused_forward_argument` | 1,002 | 声明的 forward 参数从未被读取 |
| `forward_uses_global_instead_of_init_argument` | 894 | `__init__` 收了参数没用，`forward` 用了同名全局变量 |
| `forward_output_independent` | 604 | 能证明输出与所有输入无关 |
| `forward_has_no_inputs` | 196 | `forward` 没有任何输入参数 |
| `dead_forward_rng_effect` | 70 | dead code 消耗了 RNG |
| `dead_forward_stateful_effect` | 32 | dead code 调用了会改状态的操作（如 running 统计更新） |
| `identity_forward` | 11 | 函数体只有 `return <某个输入>` |
| `dead_forward_unknown_effect` | 6 | dead code 调用了副作用无法确定的函数 |
| `semantic_duplicate` | 6 | 与更早的样本完全重复（语料内部去重） |
| `explicit_uuid_denylist` | 3 | 三个经人工确认的问题 UUID |

<!-- 一个口径说明：静态候选数加静态拒绝数比原始数据少 7 个样本。这 7 个样本是 GPU 分片启动时用更新版静态检查重扫拦下的（6 个分数池化、1 个 Gumbel-softmax 新被识别为 `random_forward`），没上 GPU。静态拒绝总数以 `skipped_static_reject` 一项的数字为准。 -->

### GPU runtime 检查

过了静态检查的样本在 GPU 上实际执行，检查规则如下：

- 使用 3 个种子，比较容差为 `rtol=1e-4`、`atol=1e-5`
- 每个种子下，同一份输入执行 3 次 forward，要求结果一致且有限
- 最多采集 20 个自然输入，要求输出至少有 `1e-4` 比例的元素发生变化
- 单独扰动每个 forward 参数，要求每个参数都能影响输出
- 连续 40 次 `get_inputs()` 完全相同，判为固定输入
- 自然采样无法证明输入敏感性时，继续使用合成输入，包括缩放、强仿射、值域反射、全零和 NaN/Inf 边界；仍无法证明的样本进入 quarantine

| Runtime verdict | 样本数 | 规则 | 处理 |
| --- | ---: | --- | --- |
| `passed` | 45,505 | - | mode 划分前合格 |
| `fixed_input_values` | 4,028 | 连续 40 次 `get_inputs()` 完全相同，可以背答案 | 拒绝 |
| `input_sensitivity_inconclusive` | 3,592 | 输入在变但输出不变；静态上又存在输入到输出的路径，无法定论 | quarantine |
| `synthetic_sensitivity_only` | 3,534 | 自然输入下输出不变，只有合成探针能改变输出 | quarantine |
| `forward_argument_sensitivity_inconclusive` | 1,417 | 某个参数怎么扰动都不影响输出 | quarantine |
| `Failed` | 334 | 超时、OOM、异常、缺模块等 | 资源或代码失败 |
| `no_same_output` | 449 | 同一份输入跑 3 次，结果不一致 | 拒绝 |
| `natural_output_low_activity` | 297 | 输出有变化但比例不到 `1e-4` | quarantine |
| `non_finite_output` | 184 | 输出出现 NaN/Inf | 拒绝 |
| `different_input_not_changed` | 128 | 输入变、输出不变，静态上也没有输入依赖 | 拒绝 |
| `unstable_input_structure` | 8 | `get_inputs()` 返回的结构不稳定 | 拒绝 |
<!-- | `skipped_static_reject` | 12,520 | 静态检查拒绝 | 没上 GPU | -->

<!-- 这张表已经覆盖延长预算后的最终 verdict，所有样本数相加仍为 71,996。`Failed` 不等于语义脏数据：最终仍有 40 个样本完整审计超时、28 个 OOM，其他失败来自依赖或代码异常；这些资源和执行失败继续与语义拒绝分开。 -->

### train/eval 划分

部分题目（比如BatchNorm、Dropout等），天然在train/eval下有不同的行为。当前KernelGym标准流程中，没有声明是train还是eval模式（虽然实际使用eval模式）。
因此，这类样本存在歧义。对全部有效样本，分别在 train 和 eval 模式下跑，比较输出、内部状态和随机性：

| mode 组合（互斥） | 样本数 |
| --- | ---: |
| train/eval 一致（mode-same） | 40,427 |
| 输出与状态均不同 | 2,824 |
| 输出不同且 train 随机 | 1,976 |
| 输出、状态、随机性均不同 | 123 |
| 仅状态不同 | 105 |
| 仅输出不同 | 43 |
| 仅 train 随机 | 7 |

mode-variant 主要是 BatchNorm running 状态更新和 dropout 造成的，抽样复核全部支持划分结果（见人工复核一节）。

### 人工复核

<!-- 人工共读 316 个互不重复的样本，其中 60 个用于复杂度校准（见复杂度配比文档），其余 256 个是清洗处置复核：72 个覆盖常规保留和排除类别，139 个覆盖全部 quarantine 类别，45 个覆盖延长预算后的最终 verdict 和失败类别。所有样本都在读码前按确定性规则选定，是刻意分层抽样，不能当总体错误率。 -->

#### 第一轮：保留与排除各类别（72 个样本）

| 类别 | 阅读数 | 结论 |
| --- | ---: | --- |
| mode-variant | 10 | BatchNorm/dropout 证据支持划分 |
| 固定输入 | 5 | `get_inputs()` 确实返回常量 |
| 仅合成敏感 | 5 | 自然分布下输出确实不动 |
| 参数/输入敏感性无法定论 | 10 | 支持 quarantine |
| 低自然活跃度 | 4 | 谓词类操作只改少数元素，与测量一致 |
| runtime 失败 | 6 | 程序合理但吃资源 |
| 其他 runtime 排除 | 12 | 代码里能看到对应问题 |
| 静态排除 | 20 | 代码里能看到对应问题 |

<!-- 第一轮没有发现需要推翻处置的样本。复杂度相关的 60 个样本的复核和 `nn.Identity` 计数缺陷修正不在本节重复展开。 -->

#### 第二轮：全部 quarantine 类别（139 个样本）

<!-- 抽样方法：排除第一轮已读的样本，每层内按 `SHA-256(salt || row_index || uuid)` 排序选取；固定 salt 记录在复核 JSON 中。大层读 20 个样本，不足 20 个全读。 -->

| Quarantine 层 | 总量 | 阅读数 | 结论 |
| --- | ---: | ---: | --- |
| `input_sensitivity_inconclusive` | 3,592 | 20 | 18 个样本是分布导致的常量路径，2 个容差坍缩；全部支持 quarantine |
| `synthetic_sensitivity_only` | 3,534 | 20 | 全是退化的谓词/归约/饱和路径，只有极端探针能够到 |
| `forward_argument_sensitivity_inconclusive` | 1,417 | 20 | 13 个确实不敏感的参数，5 个耦合/可选参数（见下），2 个有歧义 |
| `natural_output_low_activity` | 297 | 20 | 全是稀疏谓词或近常量变换，与测量一致 |
| 仅 dead code RNG 副作用 | 161 | 20 | 消耗 RNG 的结果确实被覆盖或丢弃 |
| 仅 dead code 状态副作用 | 91 | 20 | BatchNorm 的结果未被使用，但训练模式下仍可能更新 running 状态 |
| dead code RNG + 状态副作用 | 7 | 7（全部） | 两种机制同时存在 |
| `dead_forward_unknown_effect` | 9 | 9（全部） | 被丢弃的 MultiheadAttention/LSTM 分支，副作用不能假设为无 |
| `explicit_uuid_denylist` | 3 | 3（全部） | 与人工确认的问题记录一致 |
| **合计** | **9,111** | **139** |  |

最值得恢复的是 `forward_argument_sensitivity_inconclusive` 里的 7 个样本：5 个是高/宽、起止/步长这类耦合参数，单独扰动一个会失败；或可选参数固定为 `None`，没有合法的替换值。另 2 个看着合法但现有探针证明不了影响。这些仍留在 quarantine，以后做 schema 感知的校验器（耦合参数一起扰动、生成类型正确的可选参数值）时再恢复。

<!-- 代表性例子：`cuda_llm_310675` 算 `bitwise_xor(x, x)`，输出恒为常量；`cuda_llm_391471` 的可选 attention mask 固定为 `None`；`cuda_llm_549004` 算 `sqrt(abs(sign(x)))`，对 `randn` 输入恒为 1，只有精确的零能改变输出 -->

<!-- #### 最终 verdict 复核（45 个样本）

另有 45 个互不重复的样本覆盖最终的 `passed`、四类 runtime quarantine、固定输入、输入无关、timeout、OOM 和代码异常。读完完整源码后，45/45 都支持最终处置；资源失败样本继续与语义拒绝分开。

复核证据（每个样本的行号、UUID、代码）在：

- `local_artifacts/data_handoffs/drkernel_gpu_v5_manual_review_packet_20260730.txt`
- `local_artifacts/data_handoffs/drkernel_gpu_v5_manual_review_20260730.json`
- `local_artifacts/data_handoffs/drkernel_gpu_v5_quarantine_expanded_review_20260730.json`
- `local_artifacts/h200_timeout_recheck_20260731/review/result_breakdown_and_sample_manifest.json` -->

## 鸥波合成

<!-- 数据存放在 `Data/prompt_tvm_v3/oubo_generate_accepted/`，过的是与 DrKernel 相同的整套检查。 -->

### 数据漏斗总览

| 阶段 | 样本数 | 占比 |
| --- | ---: | ---: |
| 原始数据 | 2,326 | 100.00% |
| GPU 执行通过 | 2,004 | 86.16% |
| mode-same | 1,644 | 70.68% |
| 近似去重后训练候选 | 1,641 | 70.55% |
| mode-variant | 360 | 15.48% |

### CPU 静态检查

与 DrKernel 使用同一套 AST 检查。下表按 primary reason 互斥计数，只列非零类别
共 233 个样本未送 GPU，其中 228 个静态拒绝、5 个是仍可能消耗随机数或改变状态的 dead code，进入 quarantine

| 静态 primary reason | 样本数 | 处理 | 判断规则 |
| --- | ---: | --- | --- |
| `dead_forward_computation` | 131 | 127 拒绝，4 quarantine | 含调用的结果被覆盖、丢弃或到不了返回值；4 个样本同时存在 RNG、状态或未知副作用 |
| `random_forward` | 54 | 拒绝 | `forward` 含未受契约控制的随机调用 |
| `unused_forward_module` | 26 | 拒绝 | `__init__` 创建的 `nn.*` 模块没有参与 `forward` 输出 |
| `execution_mode_dependent_forward` | 9 | 拒绝 | `forward` 显式依赖编译、trace 或执行模式，无法形成稳定 oracle |
| `non_kernel_forward_serialization` | 8 | 拒绝 | 核心行为是模型/对象序列化，不是可优化的 tensor kernel 计算 |
| `unused_forward_argument` | 2 | 拒绝 | 声明的 forward 参数从未被读取 |
| `unsafe_reference_code` | 2 | 拒绝 | reference 含不允许执行的危险代码 |
| `dead_forward_rng_effect` | 1 | quarantine | 被丢弃的分支仍消耗 RNG，不能假设没有行为影响 |
| **剔除样本合计** | **233** | **228 拒绝，5 quarantine** | — |

### GPU runtime 检查

<!-- 通过静态检查的 2,093 个样本使用与 DrKernel 相同的三种子 runtime 检查。下表同样只列非零类别；最终 2,004 个样本通过，52 个进入 runtime quarantine，37 个拒绝或执行失败。 -->

| Runtime verdict | 样本数 | 处理 | 判断规则 |
| --- | ---: | --- | --- |
| `passed` | 2,004 | 保留 | 三个种子下输出有限、同输入稳定、自然输入敏感，且每个 forward 参数均能证明影响输出 |
| `forward_argument_sensitivity_inconclusive` | 47 | quarantine | 至少一个 forward 参数在现有合法扰动下无法证明会影响输出 |
| `Failed` | 16 | 拒绝/失败池 | 10 个样本存在 CPU/GPU device mismatch，6 个触发 accelerator error |
| `unstable_input_structure` | 15 | 拒绝 | 多次 `get_inputs()` 返回的容器、叶子数量或结构不一致 |
| `no_same_output` | 4 | 拒绝 | 同一输入重复执行得到不同输出 |
| `synthetic_sensitivity_only` | 3 | quarantine | 自然输入下输出不变，只有合成边界探针能改变输出 |
| `non_finite_output` | 2 | 拒绝 | 输出出现 NaN 或 Inf |
| `input_sensitivity_inconclusive` | 2 | quarantine | 静态上存在输入到输出路径，但自然输入和合成探针都无法证明输出变化 |
<!-- | **实际执行合计** | **2,093** | **2,004 保留，52 quarantine，37 拒绝/失败** | — | -->

<!-- 完整 2,326 个样本的口径为：2,004 保留、57 quarantine、265 拒绝；其中 quarantine 包含静态层 5 个和 runtime 层 52 个。 -->

### train/eval 划分

| mode 划分 | 样本数 |
| --- | ---: |
| mode-same | 1,644 |
| mode-variant | 360 |

<!-- 主目录保存 2,003 个有效样本，另外 1 个 mode-same 在 GPU 审计的独立分区。 -->

### 近似去重

<!-- 近似去重只作用于 1,644 个 mode-same 训练候选样本，DrKernel、CUDA-Agent-Ops-6K 和 KernelBook 的 mode-same 样本优先保留，鸥波内部则保留更早的样本。单独使用 token 或 AST 阈值会把共享 `nn.Module` 骨架、但核心算子不同的任务误判为相似：原始阈值命中 41 对，逐对检查算子清单并阅读其中 13 对完整 `Model` 与输入代码后，确认 38 对属于这种误判。 -->
判断规则：标准化 `ops` 集合完全相同，并同时满足 Python token Jaccard 大于 0.8，或入口类 AST 结构相似度大于 0.9。

| 结果 | 样本数 | 处置 |
| --- | ---: | --- |
| 与 DrKernel 近似重复 | 1 | 删除鸥波样本，保留 DrKernel 样本 |
| 鸥波内部近似重复 | 2 | 删除较晚的鸥波样本 |
| 最终训练候选 | 1,641 |  |

<!-- 最终文件的 SHA-256 是 `b590256457d408f673aff3bc560e2606b1f144f66eb4063d28a1a877875900d3`。逐样本决策、基线哈希和人工复核结论在 `local_artifacts/data_handoffs/oubo_near_dedup_20260731/`。来源和许可证限制没有改变，这 1,641 个样本仍不能自动并入训练。 -->

### 人工复核

抽 40 个样本人工复核，覆盖保留 mode-same、mode-variant、runtime quarantine 和静态拒绝四类，全部支持最终处置。

<!-- 这批数据单独存放在上述目录，上游来源未确认。 -->

## 外部开源（CUDA-Agent-Ops-6K / KernelBook）

只处理能提供可执行 PyTorch `Model`、`get_inputs()` 和 `get_init_inputs()` 的 CUDA-Agent-Ops-6K 与 KernelBook。FastKernels 和 TritonBench 没有可直接复用的同形 oracle 和输入函数，因此没有混入。
沿用与 DrKernel 相同的审计规则，并添加去重。

### 数据漏斗总览

<!-- `all_valid` 是通过完整 runtime 审计的集合；quarantine 是未保留样本的子集，不能与未保留数相加。 -->

| 阶段 | CUDA-Agent | KernelBook |
| --- | ---: | ---: |
| 转换后原始样本 | 6,000 | 18,153 |
| 通过静态检查并送 GPU | 5,211 | 13,033 |
| GPU 执行通过 | 4,995（83.25%） | 11,820（65.11%） |
| 未保留 | 1,005 | 6,333 |
| 其中 quarantine | 155 | 534 |

### CPU 静态检查

<!-- 两份数据都先执行与 DrKernel 相同的 AST 检查；重复定义仍取 Python 最后生效的版本，重写样本后再过其余全部检查。CUDA-Agent 有 3 个样本按此规则重写 reference 和 UUID，GPU 通过后以 `repairs.uuid` 记录。下表按 primary reason 计数，每个样本只属于一个类别；CUDA-Agent 的 2 个 `dead_forward_stateful_effect` 和 1 个 `execution_mode_dependent_forward` 进入静态 quarantine，KernelBook 的 3 个 `dead_forward_rng_effect` 同样隔离，其余为静态拒绝，全部不送 GPU。 -->

| 静态 primary reason | CUDA-Agent | KernelBook |
| --- | ---: | ---: |
| `semantic_duplicate` | 437 | 4,209 |
| `semantic_duplicate_against` | 64 | 0 |
| `unused_forward_module` | 143 | 169 |
| `unused_forward_argument` | 49 | 361 |
| `dead_forward_computation` | 45 | 35 |
| `forward_uses_global_instead_of_init_argument` | 37 | 0 |
| `random_forward` | 10 | 188 |
| `unsafe_reference_code` | 0 | 97 |
| `forward_has_no_inputs` | 1 | 37 |
| `reference_contains_code_fence` | 0 | 14 |
| `forward_output_independent` | 0 | 5 |
| `dead_forward_stateful_effect` | 2 | 0 |
| `dead_forward_rng_effect` | 0 | 3 |
| `execution_mode_dependent_forward` | 1 | 1 |
| `non_kernel_forward_serialization` | 0 | 1 |
| **剔除样本合计** | **789** | **5,120** |

### GPU runtime 检查

<!-- 下表的分母是实际执行的静态候选。`passed` 样本随后进入互斥的 mode 分区；四类证据不足的 verdict 进入 quarantine；固定输入、输出不随输入改变、同输入不稳定和非有限输出直接拒绝；`Failed` 进入资源或依赖恢复池。 -->

| Runtime verdict | CUDA-Agent | KernelBook | 处理 |
| --- | ---: | ---: | --- |
| `passed` | 4,995 | 11,820 | - |
| `Failed` | 17 | 471 | 资源或代码失败 |
| `forward_argument_sensitivity_inconclusive` | 70 | 150 | quarantine |
| `input_sensitivity_inconclusive` | 9 | 198 | quarantine |
| `synthetic_sensitivity_only` | 36 | 175 | quarantine |
| `natural_output_low_activity` | 37 | 8 | quarantine |
| `fixed_input_values` | 13 | 26 | 拒绝 |
| `different_input_not_changed` | 2 | 55 | 拒绝 |
| `no_same_output` | 6 | 85 | 拒绝 |
| `non_finite_output` | 26 | 45 | 拒绝 |
| **实际执行合计** | **5,211** | **13,033** | — |

<!-- CUDA-Agent 的 17 个失败主要是 OOM（12 个），其余为 timeout 和 `AcceleratorError`。KernelBook 的 471 个失败绝大多数是缺少 `_paritybench_helpers`（454 个），少数是构造时下载外部权重或 `TypeError` 等代码异常。这些资源、依赖和代码失败都没有被描述成语义脏数据，也没有进入当前训练候选。 -->

### train/eval 划分

| mode 划分 | CUDA-Agent | KernelBook |
| --- | ---: | ---: |
| `mode_same_observed` | **4,551（75.85%）** | **10,897（60.03%）** |
| `mode_variant` | 444 | 917 |
| `mode_inconclusive` | 0 | 6 |

<!-- mode-variant 的计算本身有效，但 train/eval 的输出、状态或随机性不同，单独保存。KernelBook 的 6 个 mode-inconclusive 样本在 runtime 主检查中通过，但 train/eval 审计无法复制模型，只留在 `all_valid`，不进 `mode_same_observed` 分区。 -->

### 人工复核

<!-- 人工复核共 202 个互不重复的真实 reference：CUDA-Agent 96 个，KernelBook 106 个。两边都覆盖每个静态 primary reason、每个 runtime 拒绝 verdict、每个 quarantine verdict、每个失败异常类别，以及 mode-same 和 mode-variant；KernelBook 的 6 个 mode-inconclusive 全部阅读。 -->

| 复核层 | CUDA-Agent | KernelBook | 结论 |
| --- | ---: | ---: | --- |
| 全部人工阅读 | 96 | 106 | 没发现错误放入 mode-same 的样本 |
| 每类 runtime quarantine | 5 | 5 | 饱和、稀疏谓词、自然分布退化、可选或耦合参数等机制与审计一致 |
| mode-same | 10 | 10 | 输出对输入有实质依赖，抽样支持保留 |
| mode-variant | 10 | 10 | 主要为 BatchNorm/SyncBatchNorm 状态或 dropout/attention 随机性差异 |
| mode-inconclusive | 0 | 6（全读） | 3 个样本自定义 `Parameter.fast`，另 3 个分别含线程锁、generator、自定义 autograd 对象；模型复制失败，继续隔离 |
<!-- | 延长预算后的最终 verdict | 16 | 6 | 覆盖通过、timeout、OOM、依赖失败和新增 quarantine，全部支持最终处置 | -->

<!-- 抽检同时确认了保守误拒边界。CUDA-Agent 的少量 `random_forward` 来自当前输入走不到的随机分支或仅用于取类型的随机调用；部分 `no_same_output` 可能是 GPU tie-breaking/归约不确定性。KernelBook 有 2 个 `forward_output_independent` 样本通过 helper 的内部状态间接影响输出，静态规则没有证明这条路径。它们都继续留在拒绝或恢复池，因为手工读码不足以替代可复现的自动释放条件。KernelBook 的 454 个缺 helper 样本值得在补齐、固定并记录依赖后重跑。 -->

<!-- 逐样本代码、选择标签、mode 细节和 runtime 错误保存在 `local_artifacts/external_gpu_v5_20260730/manual_review_packet.json`；最终 verdict 的补充复核证据位于 `local_artifacts/h200_timeout_recheck_20260731/review/`。完整审计记录分别为 `local_artifacts/external_gpu_v5_20260730/cuda_agent/full.audit.jsonl`、`local_artifacts/external_gpu_v5_20260730/kernelbook/full.audit.jsonl` 和 300 秒 GPU 审计记录。 -->

<!-- ### 数据位置

两份结果位于 `Data/prompt_tvm_v3/external/`，没有追加到主训练 parquet。CUDA-Agent 的 converted parquet 和 provenance 位于 `external/cuda_agent_ops_6k/source/`；KernelBook 对应文件位于 `external/kernelbook/source/`。每个 parquet 的 UUID 唯一；主目录和独立分区互不重复，按来源合并后得到上文的最终有效集合。 -->

## KernelBench

KernelBench 的处理方式与训练数据不同：自动审计（内部语义去重、CPU 静态检查、GPU runtime，参数与主数据一致）只负责找候选问题，每道题是否留在测试集由逐题人工裁决决定。
最终没问题的测试集为 Level 1 共 100 题、Level 2 共 93 题、Level 3 共 37 题。

<!-- 产物分四类：`all_valid.parquet` 是自动审计通过集合；`manual_restored.parquet` 是自动未通过、但人工确认仍是有效测试题的集合；`excluded.parquet` 是人工确认存在缺陷的集合；`train.parquet` 是最终测试集，即自动通过与人工恢复的并集。`mode_variant.parquet` 只是诊断分区。交付 parquet 的 schema 与原始测试集集一致；源数据没有 UUID 和 `ops` 字段，审计记录里生成了这两个字段供对齐，但没有塞回交付 parquet，以兼容现有评测 launcher。 -->

### 数据漏斗

| 阶段 | Level 1 | Level 2 | Level 3 |
| --- | ---: | ---: | ---: |
| 原始题数 | 100 | 100 | 50 |
| 自动审计通过 | 83 | 91 | 32 |
| 自动未通过，人工恢复 | 17 | 2 | 5 |
| 人工确认排除 | 0 | 7 | 13 |
| **最终测试集** | **100** | **93** | **37** |

### CPU 静态检查

<!-- 8 个样本命中静态规则：Level 2 两个（未使用的 `self.bias`；`forward` 直接读全局 min/max 而不用构造参数），Level 3 六个（`forward` 内随机初始化 LSTM 状态、辅助 FC 成为 dead code、未使用的 `mask`/`c_proj` 等）。八个样本的源码全部人工阅读，规则命中与实际代码一致；但命中训练数据的静态规则不等于测试题不可用，最终只排除恒零的全局 min/max 和随机 LSTM oracle 两个，其余六个恢复。 -->

| 静态 primary reason | Level 1 | Level 2 | Level 3 |
| --- | ---: | ---: | ---: |
| `unused_forward_module` | 0 | 1 | 1 |
| `forward_uses_global_instead_of_init_argument` | 0 | 1 | 0 |
| `random_forward` | 0 | 0 | 1 |
| `dead_forward_computation` | 0 | 0 | 2 |
| `unused_forward_argument` | 0 | 0 | 2 |
| **静态拒绝合计** | **0** | **2** | **6** |

### GPU runtime 检查

<!-- Runtime 参数与主数据一致：三个种子、同输入三次、最多 20 个自然输入、40 次固定输入检查、逐参数敏感性、完整 train/eval 审计、`rtol=1e-4`、`atol=1e-5`、每个样本 60 秒超时。表中每列之和等于实际送 GPU 的静态候选数。 -->

| Runtime verdict | Level 1 | Level 2 | Level 3 | 处理 |
| --- | ---: | ---: | ---: | --- |
| `passed` | 83 | 91 | 32 | - |
| `input_sensitivity_inconclusive` | 0 | 4 | 11 | quarantine |
| `synthetic_sensitivity_only` | 0 | 1 | 1 | quarantine |
| `natural_output_low_activity` | 0 | 2 | 0 | quarantine |
| `Failed` | 17 | 0 | 0 | 全量人工裁决；Level 1 全恢复 |
| **实际执行合计** | **100** | **98** | **44** | — |

Level 1 的 17 个 `Failed` 全是 60 秒超时。逐题复核确认它们都是正常的官方算子题，只是整套多探针审计在极大 shape 上跑不完：例如矩阵向量乘的矩阵是 `2048 × 1,048,576`

Level 2/3 的 19 个 quarantine 样本全部完整阅读，恢复其中一个。其余确认在官方输入和容差下可以绕过核心计算：
1. GroupNorm 后取全局 mean，差异约 `3e-8`；
2. 大 softmax 变化约 `1.6e-6`，落在绝对容差内；
3. ViT 把 `[B,S,E]` 送进默认 `batch_first=False` 的 encoder 后只取 class token，patch 输入无法汇聚；
4. MobileNetV2 输出约 `2.7e-21`，等等

### train/eval 诊断

下表仅统计自动审计通过集合。BatchNorm 或 dropout 的 train/eval 差异是正常模块语义，不作为删题理由，mode-variant 全部保留在最终 `train.parquet`。

| 互斥 mode 组合 | Level 1 | Level 2 | Level 3 |
| --- | ---: | ---: | ---: |
| `train_eval_same` | 82 | 79 | 21 |
| 输出和状态均不同 | 1 | 11 | 11 |
| 输出不同且 train 随机 | 0 | 1 | 0 |
| mode-inconclusive | 0 | 0 | 0 |

24 个 mode-variant 样本全部人工阅读，标记均成立：Level 1 唯一一个是 BatchNorm；Level 2/3 的"输出和状态均不同"都是 BatchNorm running-state 路径，Level 2 另有一个 dropout 导致 train-only 随机。

### 人工复核

共 98 个互不重复的样本。44 个自动未通过样本是全量逐题裁决，不是抽样：每题完整阅读可执行逻辑，结合自动审计、输入 shape 和 GPU 数值复跑，给出明确的恢复或排除理由。24 个 mode-variant 全部阅读。另按固定 SHA-256 规则从每个 level 的 mode-same 抽 10 个样本。后两部分是诊断证据，不能当总体错误率。

| 复核层 | Level 1 | Level 2 | Level 3 | 结论 |
| --- | ---: | ---: | ---: | --- |
| 全部自动未通过样本 | 17 | 9 | 18 | **44/44 完整逐题复核** |
| 其中人工恢复 | 17 | 2 | 5 | 资源超时、非致命附属状态、可选参数或仍有逐元素敏感性 |
| 其中最终排除 | 0 | 7 | 13 | 容差级恒定/近恒定输出或随机 oracle |
| 全部 mode-variant | 1 | 12 | 11 | 标记成立，但不作为测试集排除条件 |
| mode-same 确定性抽样 | 10 | 10 | 10 | 30 个样本都有明确输入依赖和合理计算 |
| **互不重复合计** | **28** | **31** | **39** | **98 个样本** |

| 人工裁决机制 | Level 1 | Level 2 | Level 3 | 处理 |
| --- | ---: | ---: | ---: | --- |
| 整套审计资源超时，但单题逻辑正常 | 17 | 0 | 0 | 恢复 |
| 未使用的附属参数/模块或成为 dead code 的辅助 FC，不影响核心计算 | 0 | 1 | 3 | 恢复 |
| 可选且官方输入不提供的 `mask=None` | 0 | 0 | 2 | 恢复 |
| 变化稀疏但逐元素错误仍会失败 | 0 | 1 | 0 | 恢复 |
| 官方输入与容差下恒定或近恒定，可绕过核心计算 | 0 | 7 | 12 | 排除 |
| `forward` 内部随机导致 oracle 规则不可复现 | 0 | 0 | 1 | 排除 |

<!-- ### 使用边界

完整性校验：三层自动审计分别覆盖 100/100/50 题，没有调度占位；`train.parquet` 与 `excluded.parquet` 互斥，按原始顺序合并后恰好覆盖各自官方全集；`manual_restored.parquet` 全部来自自动未通过集合；交付 parquet 的 schema 与原始测试集集一致。

Level 1 保持官方 100 题，可维持现有分母。Level 2/3 的 93/37 题是人工裁决后的清洁诊断集；用它们汇报成绩时必须同时报告原始 100/50 官方全集结果，不能与历史官方分数直接比较。复核证据文件见文末产物清单。 -->

<!-- ## 执行与完整性

正文的最终 verdict 以有效 UUID 对齐：71,996 个样本的主审计记录提供完整分母，361 个样本的延长预算审计记录覆盖其中对应记录。两份记录的覆盖关系逐样本校验；主目录与独立通过分区的 UUID 交集为零，按来源合并后的有效样本数分别为 45,505、2,004、4,995 和 11,820。

DrKernel 候选数据切成 24 个分片，在 node64、node69、node70 上跑，合并前核验了全部分片的计数和哈希，最终记录条数与原始数据一致。所有 runtime 结论都来自这轮重新执行，没有复用旧的 CPU 结果。

外部数据另行执行：CUDA-Agent 在 node64 上使用 8 张 GPU，8 个分片分别耗时 88–109 分钟；KernelBook 在 node53、node69、node70 上使用 24 张 GPU，24 个分片分别耗时 90–120 分钟。每张 GPU 跑 4 个隔离 worker。运行器按完整 batch 才写审计和进度，而每个分片小于 batch，导致分片完成前长期显示 0%，这是进度可观测性缺陷，不代表 GPU 空闲。

这轮外部运行还暴露了一个编排错误：正式的 KernelBook 24-shard 结果已经齐全后，旧的 8-shard KernelBook 命令仍排在 CUDA-Agent 会话后方，其中 7 个重复任务继续占用 node64，延迟了最终判断。重复任务已停止，最终物化只使用正式 24-shard 结果。第一次合并还因 `seq -w 0 7` 生成一位分片号而失败关闭，没有写出部分结果；改用显式两位编号后合并成功。以上偏差没有改变任何最终样本级 verdict。

KernelBench Level 2/3 分别在 4-worker GPU 配置下用时约 7 和 4 分钟。Level 1 首次沿用 4 worker 时，大输入并发造成显存争用，处理到第 64 个样本时已出现 28 个混合 OOM/超时，该轮立即停止并全部作废。第一次分片恢复又发现 `skipped_by_option` 会被清洗器自动重跑、任务范围超出声明索引，同样停止且没有发布任何结果。最终改为 8 张 GPU、每卡一个 worker、八个互斥索引段，最长分片约 14 分钟；合并前逐样本核对声明索引、入口点、语义哈希和有效 UUID，并确认 100 个样本恰好覆盖一次。隔离执行留下的 17 个 `Failed` 仍保留在自动审计记录中，但人工裁决层全部恢复，避免把整套审计的资源预算误写成测试题有效性。

发布前修掉了 4 个清洗问题：监控误报节点完成进度（只影响显示）；合并脚本参数传错，失败关闭没写输出，重跑修复；审计聚合统计口径错误，`AuditSummaryTest` 已覆盖；runtime overlay 缺 `reasons`/`flags` 字段，`ShardMergeTest` 已覆盖。复杂度计数缺陷及其验证见独立的复杂度配比文档。

专项清理测试约 110 个全部通过，resume、数据集状态、任务参数测试约 70 个全部通过，启动脚本通过 `bash -n`。 -->

## 复现

```bash
python3 -m tools.data.cleaning.subsets \
  --input Data/prompt_tvm_v3/analysis/drkernel_rl_thinking.all_valid.parquet \
  --audit-jsonl local_artifacts/data_handoffs/drkernel_gpu_v5_full_20260730.audit.jsonl \
  --output /tmp/drkernel_mode_same.parquet \
  --summary-json /tmp/drkernel_mode_same.summary.json \
  --keep-status kept \
  --require-any-mode-flag train_eval_same \
  --align-by-effective-uuid \
  --overwrite

python3 tests/tools/data/test_ops_data_cleaning.py
```

<!-- 复杂度物化和一致性校验命令见独立的 [复杂度配比文档](handoff_drkernel_complexity_mixture_20260731.md)。 -->

## 遗留问题

- gpu-v5 的 dead-state 检查有一个覆盖缺口：`unused_forward_module` 只跟踪 `self.x = nn.*(...)` 或 `torch.nn.*(...)`，`unused_forward_argument` 只检查 `forward` 形参；它不检查“构造参数只写入普通 `self.*` 属性、且该属性无法从 `forward` 及其 helper 到达”的情况。`kernelbook_508_aa2f4ab6a65c3b1f` 中，`raw_msg_dim`、`memory_dim`、`time_dim` 只计算未参与 `forward` 的整数属性 `self.out_channels`，但四个 `forward` tensor 都影响 `torch.cat` 输出，因此静态和 GPU sensitivity 均通过。后续应增加从 `__init__` 参数到普通实例属性、再到 `forward` 可达读取的 AST 数据流检查，并对纯 metadata 的保留口径单独定义。
- 40 个样本仍超出 300 秒完整审计预算，28 个样本在完整审计中 OOM。这 40 个样本的单次 smoke 全部通过；使用前需要先定义目标时延/显存要求。
- 当前 `Failed` 中没有纳入延长预算复核的非-timeout 样本尚未统一复查；继续把 OOM、依赖缺失、代码异常和语义拒绝分开，不能用一个 `Failed` 桶代表脏数据。
- 恢复 `forward_argument_sensitivity_inconclusive` 之前，先做耦合参数和可选参数的探针。
- CUDA-Agent 的下一步是训练混合比例和增益消融实验。
- KernelBook 缺 `_paritybench_helpers` 的 454 个样本，补齐并固定依赖后再考虑恢复。
- 本轮没有测 DrKernel、鸥波合成数据或外部候选与验证集的污染，后续需要单独补。


## 产物清单

### DrKernel

| 产物 | 样本数 | SHA-256 |
| --- | ---: | --- |
| 主训练 parquet | 40,307 | `9e9ffca46022e74c0616f5e272871e76dfd000b6e7937b01685cd0bb11d521e5` |
| 独立 mode-same 分区 | 120 | `e460fc178145bf02de91ea4262bc3e0a7d67ffaf10e97c1476825db43e0bdb4c` |
| 主目录全部有效样本 | 45,346 | `8a12c0bb81485fc708fbc44d16b517bf0dec049ba245068a5a2753cc58e7b484` |
| 独立 `passed` 分区 | 159 | `31814861d1f265169875e7ad62a724c9032650388c27680954675471c2bda464` |
| 主目录 mode-variant parquet | 5,039 | `c1e992f11646a1a15aea3e00434dc6066a1d92f0dbeab4435896e77ddbd7f787` |
| 独立 mode-variant 分区 | 39 | `f0fffc8bc3c69ba93efce648834af814103bcbfb316d9ff194b16d524cc19d21` |
| 主目录完整审计 JSONL | 71,996 | `51052d9ef2043c56176bce047320c07a071fdd9b2b6e69b8754f630102d505b4` |
| 延长预算审计 JSONL | 361 | `fb6bf3fb6ed163ecc38c3f637288189f37f2d2180eafd45543f7de95d5218b5b` |
| 第一轮人工复核 JSON（含 60 个复杂度校准样本） | 132 | `a49c8ca4f34224b233d0e3a5ed995de592fa5634310021c75c8b4dfaf985587d` |
| 第二轮 quarantine 复核 JSON | 139 | `bb75adb9980726340e8a1d2e549fdfd289ba04f5c823b520f8aec518206d7613` |
| 最终 verdict 复核清单（其中 DrKernel 45 个样本） | 67 | `6379f3fa639900d02ed634c2242c03e633556c8faae7e8647fc94f3d4c2c0808` |

### 外部开源

| 产物 | 样本数 | SHA-256 |
| --- | ---: | --- |
| CUDA-Agent 主目录 `all_valid.parquet` | 4,937 | `9b2a95f8edd62ee12465320f135ff60e375e5a8975de8d0db45860e6d59bf8aa` |
| CUDA-Agent 主目录 `mode_same_observed.parquet` | 4,498 | `384a26720c9e7c6665e8f4c5177627197a41b23b81fe090cc7857b61feef6c76` |
| CUDA-Agent 主目录 `mode_variant.parquet` | 439 | `0055255dce4ae3f4fa241d1deace781753658fa978e931782b0d12f92f71d4d0` |
| CUDA-Agent 独立 `passed.parquet` | 58 | `4eef47bd4c73434f08aefad3fd755cf5448120bd2da7a4bae7ba036ddfe6f89d` |
| CUDA-Agent 独立 `mode_same_observed.parquet` | 53 | `822455fcc0b0ceaf49e0073d917c19a92fd630680b78c5acd3dc9fe405a3ff53` |
| CUDA-Agent 独立 `mode_variant.parquet` | 5 | `8fa4c67438897321e2fbcda563f359c7bc37802daa79e6b3279eafff052ed6a0` |
| KernelBook 主目录 `all_valid.parquet` | 11,816 | `5c9b4d78444c8d096221928a53bc79f32a782de22bdba5e21ef1656524f5d0ab` |
| KernelBook 主目录 `mode_same_observed.parquet` | 10,893 | `14e9d7114bd0f093df9b94242c32335bc8f7c211559cdeb041bbd13533397a00` |
| KernelBook `mode_variant.parquet` | 917 | `06ebfabfa519e2b48d6585f491c2c2209d178f795f434f48fc3a071988197a76` |
| KernelBook `mode_inconclusive.parquet` | 6 | `93bdc4425cbfbe8066376ad35f5199cdb2fee249c23878ad000773e0557aef6b` |
| KernelBook 独立 `passed.parquet` / `mode_same_observed.parquet` | 4 | `3221e538bc7eb8d8e51d0275c692673ad3986fc83649f69b4f209a62d0cc7144` |
| CUDA-Agent 完整审计 | 6,000 | `0b2366f88e33969981a64b46b95a35c2dc91716885c1f3e0bbe3babf04a08a4e` |
| KernelBook 完整审计 | 18,153 | `9f1785c7e35eed76e3bbeaf7e3677671f5bfffe10dc7cfdc6328a56c456d856a` |
| 180 个样本的人工复核包 | 180 | `3a237d2df948df06a2d2aa6546658814c6c8c9460810cbb7056cefd552df063f` |
| 67 个样本的跨来源最终 verdict 复核包（其中外部数据 22 个样本） | 67 | `c78f2dfabd1f3b9c32113cb27a7a17d75fd10916e0188db64a4aed96ed5ad389` |

### KernelBench 测试集集

| 产物 | 样本数 | SHA-256 |
| --- | ---: | --- |
| Level 1 `all_valid.parquet` | 83 | `8acdbf40f641408bf6c5c51be0aaa482ca7030b3e76e5e2869a059a82588218a` |
| Level 1 `manual_restored.parquet` | 17 | `9642a5cfce6b8d3a2ed842657ab3299b577370d36e90b1ffcacdd059ea693c1c` |
| Level 1 `excluded.parquet` | 0 | `849f3471bf99cd937d403bbc20bec44ccef7b2c11919c609417665041c2cbedd` |
| Level 1 **`train.parquet`** | **100** | `8797bbb6d5682efde0a099065d14105083f646fa3fec301efdaa2f565985d90f` |
| Level 1 `mode_variant.parquet` | 1 | `628eb495308b22a10df6c71518cdda6a1655c707f1f75c6d46bd127f0ac26eff` |
| Level 2 `all_valid.parquet` | 91 | `24973007c043206546a6eef033147d75656a63bee7522436e2e7f92080431828` |
| Level 2 `manual_restored.parquet` | 2 | `568516864a3fe8052a16378f3fc83abe57784b63b0a717c8e49adf72e4dbc925` |
| Level 2 `excluded.parquet` | 7 | `50eadecfe2f99fd05f3c2e44233c47dc45e81a7fb15a471d2b87e7be3f988b35` |
| Level 2 **`train.parquet`** | **93** | `2b00b8e8f8328d32c3a27947aee9bee8f7faae8fe99591987155e35589a391e3` |
| Level 2 `mode_variant.parquet` | 12 | `a84e7d926662e3a5c4a0fd97b96d8dc130bdb739bed159903f5ffd57a2a15e29` |
| Level 3 `all_valid.parquet` | 32 | `627aeedd3e1908e5799ee4a445ea554b982fb759f72c4aa400cf723d7e37a6c4` |
| Level 3 `manual_restored.parquet` | 5 | `bc1b03de9cbf7ba7a4c820a21c289a75a6fb9d6b917b2264aa3916a8d7da4611` |
| Level 3 `excluded.parquet` | 13 | `6ac475f4e7fd9eb65ca6066dc5e1dd18bb695b22dab84e232385da4bcb8b338a` |
| Level 3 **`train.parquet`** | **37** | `34197b63cefb7354c8318adfcf6ecf8777bbb7bf0c471c00d83d7affaf6965a6` |
| Level 3 `mode_variant.parquet` | 11 | `536914b2ca5c3d42923b097f75378213a12af9ff54a5a73536bf82ef47f19134` |
| Level 1 完整审计 | 100 | `00d5a8da8ef9d51677b93ddbacd331d03490d5e13c5a5908ceeb4698c8b5b5bd` |
| Level 2 完整审计 | 100 | `df0a4580fafad2661c971bb2aa4b350df0d2f01f00ff3bf1638fa68ced81f8d1` |
| Level 3 完整审计 | 50 | `3a91eb90659fc8683ad6653feb3fe352ed50b0c7a73045cbbe5501bd0126f069` |
| 98 个样本的人工复核包（`manual_review_packet.json`） | 98 | `2355a6303a8d6b2fb71c423e1856a9668e2c7726878a2b4363ea3d97fcb9c577` |
| 44 个样本的逐题裁决（`kernelbench_excluded_manual_adjudication_20260730.json`） | 44 | `c183f71fc4297c622b668374ee5d6d5c6139bbc5505b1382670b2344215736f9` |
| 最终物化清单（`manual_testset_manifest.json`） | — | `12636215843d944839492634b56a4dc6324c3bec9f7d914ce42f67c906b7db19` |

最后三个复核证据文件位于 `local_artifacts/kernelbench_gpu_v5_20260730/`。
