# CUDA-Agent train/test 数据差异
当前训练侧正确率超过 90%，但 KernelBench 实测约为 L1 90%、L2 70%、L3 10%。
分析发现，当前 RL 训练用的任务集和 KernelBench 评测集之间存在较大的分布差异。本文分析两个数据集的构成和四个主要差异维度，外加一个训练集内部的问题，供后续决定数据建设方向时参考。


## 差异一：输入规模
观察可解析 tensor，KernelBench 的中位元素数是训练集的 2,048 倍，相差约三个数量级。

| tensor 元素数统计 | 训练集 | KernelBench |
| --- | ---: | ---: |
| 中位数 | 16,384 | 33,554,432 |
| p90 | 3,145,728 | 1,610,612,736 |
| ≥ 1M 元素的占比 | 18.2% | 88.6% |
| ≥ 100M 元素的占比 | 0.5% | 42.2% |

大 tensor 会压测 grid 覆盖、索引位宽、stride、数值累加、超时和显存行为，训练集则较少提供这类信号。静态 shape 解析覆盖训练集约七成、KernelBench 约八成的随机数生成调用；这里统计的是解析成功的 tensor，不代表 runtime FLOPs、显存或中间 workspace。

## 差异二：输入值分布先验

训练集 95.6% 的样本用 `randn` 生成输入，KernelBench 全部用 `rand`。同一条样本可以使用多种随机数生成方法，所以下表是多标签统计、各行不能相加；factory 之后的算术也可能改变最终值分布。

| 随机数生成方法 | 训练集 | KernelBench |
| --- | ---: | ---: |
| `randn` | 38,515 (95.6%) | 0 |
| `rand` | 1,010 (2.5%) | 250 (100%) |
| `randint` | 3,817 (9.5%) | 3 (1.2%) |

## 差异三：程序结构

两边的算子数中位数相近（都是 3 个左右），但长尾分布不同：KernelBench 有大量真正的单算子任务，也有大量长模块；训练集集中在中间，以短的多算子组合为主。

| 结构条件（按样本占比） | 训练集 | KernelBench |
| --- | ---: | ---: |
| 输入相关算子数 ≤ 1 | 2.5% | 36.4% |
| 输入相关算子数 ≥ 10 | 0.8% | 7.2% |
| 源码行数 ≥ 50 | 2.0% | 20.4% |
| 模型拥有的层/模块 ≥ 5 | 1.5% | 12.8% |
| 含多个顶层 class | 0.1% | 5.6% |

取 L3 在源码行数、算子数、层数等八项结构指标上的中位数作为基准，训练集 40,307 个样本中只有 7 个同时达到全部阈值。这个交集只描述结构尾部，不能直接解释 L3 correctness。

## 差异四：算子族分布

训练集并不缺算子多样性（有 3.5 万种不同算子签名，activation 占比与 KernelBench 同为 38.8%），但单个算子族的占比就有明显偏差（见下表）。
当前还只是单轴的统计，再叠加拓扑和 shape 的组合关系（比如卷积接归一化、大 shape 下的 matmul），差距只会更大。

| 算子族（多标签占比） | 训练集 | KernelBench | 比值 |
| --- | ---: | ---: | ---: |
| convolution | 14.3% | 45.2% | 0.32× |
| normalization | 5.6% | 19.6% | 0.28× |
| matmul/linear | 17.5% | 32.4% | 0.54× |
| indexing/scatter | 8.7% | 0.8% | 10.84× |
| loss/distance | 14.9% | 1.2% | 12.44× |

## 问题五：输入生成模板重复度高

训练集 40,307 个样本的 `Model` 类几乎两两不同，但输入生成函数去重后只有约 13k 种，同一份模板最多被 2,654 个样本共用。这说明输入生成大量复用少数模板。这一项只有训练集内部统计，没有 KernelBench 侧的同口径数字，因此不构成双边差异结论。

## 已知与未知

已确立的事实：训练池是单一 cudaLLM/DrKernel lineage；CUDA 解在线生成；训练与评测分数口径不同；正确率从 L1 约 90% 跌到 L3 10%；两边在 tensor 规模、输入先验、结构尾部、算子组合上差异显著。

尚未回答的问题：正确率差距里有多少由 shape 单独造成，多少由输入值/dtype 造成，多少由拓扑、状态和接口复杂度造成。回答这些问题需要对照干预实验，继续加分布统计表没有增量。

v4 的 coverage-first 构建针对这些差异扩展支持度：保留全部 64,315 条 canonical parent，另加 396 条验证通过的输入干预 child 和 512 条验证通过的新语义任务，组成 65,223 个样本的 review union。该 union 仍是 review 状态（`training_approved=false`）；在这段构建记录对应的分析时点，synthesis 侧登记的 training artifact 仍是上述 40,307 个样本的 v3 artifact。它不是下述 Qwen3.8 DataV2→DataV4 消融中旧 checkpoint 的训练输入。

独立 shape 扩增最终由 3 条 static lane 和 10 条模型辅助 lane 组成，共得到 63,979 个 H20 runtime-safe child，覆盖 45,833/64,315 个 canonical parent。下游按 parent 去重得到 31,648-row one-parent-one-shape base。所有 accepted child 都受 source-span shape-only、4 GiB direct-input、1000:1 维度比、paired reference 和 changed-region liveness 约束；这个集合没有并入 65,223-row union，也没有训练收益证据。

最早的 exact-two solver 把 2 的幂 occurrence 固定为 50%，后续 variable/static 和模型辅助 lane 已取消这个硬模式，但最终数据仍有熟悉值、source、operator、capacity 和 runtime acceptance bias。4 GiB input ceiling 与 1000:1 shape balance 也无法约束 attention、卷积等算子的中间显存；与 KernelBench 的结果只能称 support coverage，不能称 distribution matching。

## 产物与参考

- 完整调研与技术交接（英文，本文数据来源）：`handoffs/data/synthesize/README.md`
- 分布对比图：`handoffs/data/synthesize/figures/`
- 当时的 synthesis training artifact：`Data/prompt_tvm_v3/drkernel_rl_thinking.parquet`（40,307 个样本；不是下述 DataV2 checkpoint 输入）
- v4 review union：`Data/prompt_tvm_v4/train.coverage.review.parquet`（65,223 个样本）
- shape 扩增总览：`handoffs/data/synthesize/shape_expansion.md`
- initial exact-two solver evidence：`Data/prompt_tvm_v4/shape_solver_multidim_v4_byte_targets_v1/run.full53896/analysis/`
- 分布测量工具：`tools/data/synthesize/profile_prompt_tvm_distribution.py`

## 数据与训练配置的实测对照

DataV2/DataV4、step80→step100 与 24K 新 reward bundle 的完整 L1/L2/L3 指标、paired 比较和数据完整性证据统一见[Qwen3.8 KernelBench 评测](../../qwen38/kernelbench_eval.md)。新 reward bundle 同时改变 context、temperature 和 reward，不能把结果差异归因于单独的数据或 reward 因子
