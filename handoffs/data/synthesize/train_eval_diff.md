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

## 消融实验（2026-08-24）

<!-- 上述数据建设完成后，使用正式 DataV4 release 做了一次 fresh RL 训练，并与旧版 DataV2 训练做同 step80 对照。旧版 checkpoint 的实际训练输入是 `Data/prompt_tvm_v2/drkernel_rl_thinking.parquet`（71,996 rows，SHA256 `17b948be017e8e57e6e87dcae15eca3acd5b9feab141be49dd4589e74ab14d25`）；DataV4 使用 `Data/prompt_tvm_v4/release/train.parquet`（39,636 rows，SHA256 `189da56dca3acbb03b5532b360ec1eb9adde75c173952149a8515dcebfb08f79`），不是前文 65,223-row review union。前文用于分布分析的 40,307-row v3 artifact 也不是该旧 checkpoint 的训练输入。 -->

<!-- 这是一组实用的数据消融：两条 lineage 都从同一 Qwen3.8-27B BF16 base iteration 0 开始，匹配 temperature 1.1、predictive-DPPO、LongestFirst、medium reasoning、BF16 actor/FP8 rollout、16K context、no-spec 和 tvm_ffi 等主要训练契约；比较相同的 step80，并使用完全相同的 KernelBench 题目和推理协议。由于两次 fresh run 的随机 rollout 历史也不同，它能回答“DataV4 lineage 的实际 checkpoint 是否更强”，但不能把全部差值严格归因到某一类新增数据或单个构建步骤。 -->

<!-- 评测为单轮 `temperature=1.0`，L1/L2/L3 分别是 100/100/50 题，每题 8 samples，即 800/800/400 trajectories。Correct 是数值正确且排除 decoy；置信区间以 problem 为重采样单位，20,000 次 paired bootstrap，seed `20260824`。 -->

下表中，DataV2 为旧版训练数据，DataV4 为新版训练数据

### 四个 checkpoint 总览

| lineage | checkpoint | L1 Correct (%) | L1 Fast@1.0 (%) | L2 Correct (%) | L2 Fast@1.0 (%) | L3 Correct (%) | L3 Fast@1.0 (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| DataV2 | step40 | 74.50 | 18.75 | 45.38 | 1.88 | 10.75 | 1.50 |
| DataV2 | step80 | 78.25 | 23.62 | 38.38 | 4.75 | 6.00 | 1.00 |
| DataV4 | step80 | 87.25 | 20.50 | 60.62 | 3.00 | 16.75 | 1.75 |
| DataV4 | step100 | **90.12** | **21.38** | **64.75** | **2.50** | **16.50** | **1.50** |

### 新 reward step80（24K 配置消融）

<!-- 这次对照使用相同的 DataV4 release 和相同 step80，但不是单变量 reward 消融。旧配置为训练 context 16K、训练 temperature 1.1 和旧 reward/filter；新配置为训练 context 24K、temperature 1.0、output-mismatch partial reward 0.25、关闭 PRS/coverage-RS，并在最后 4K response budget 施加最大 0.2 的线性长度惩罚。两边正式评测都使用单轮、temperature 1.0、medium、tvm_ffi、no-spec、每题 8 samples；评测 context 分别跟随 checkpoint 原生配置使用 16K 和 24K。因此下表只能回答这组配置 bundle 的实际 checkpoint 变化，不能把差值单独归因于 reward。 -->

| level | 旧 DataV4 step80 Correct (%) | 新 reward step80 Correct (%) | Correct 差值 (%) | 旧 Fast@1.0 (%) | 新 Fast@1.0 (%) | Fast@1.0 差值 (%) | Compile 差值 (%) | 截断率差值 (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 | 87.25 | 84.88 | -2.38 | 20.50 | 18.12 | -2.38 | -0.62 | -0.75 |
| L2 | 60.62 | 60.62 | 0.00 | 3.00 | 0.62 | -2.38 | +5.12 | -6.00 |
| L3 | 16.75 | **20.50** | **+3.75** | 1.75 | 0.75 | -1.00 | +15.50 | -45.75 |

| level | Correct 差值的配对 bootstrap 95% CI (%) |
|---|---:|
| L1 | [-5.25,+0.62] |
| L2 | [-4.12,+4.00] |
| L3 | [0.00,+7.50] |

新配置最清楚的变化不是整体 Correct 普遍上升，而是长任务交付改善：L3 截断率从 75.75 降到 30.00，Compile 从 22.50 升到 38.00，Correct 从 16.75 升到 20.50；L1 Correct 小幅下降，L2 持平，三档 Fast@1.0 都下降。当前证据因此支持“24K 与新 reward/filter bundle 显著缓解 L3 截断并提高可编译交付”，不支持“新 reward 单独提高全部 level 的精度或速度”。

完整协议、checkpoint 来源、独立 dump 审计和人工样本复核继续合并维护在 `handoffs/in_progress/qwen38_legacy_datav4_l123_eval_20260824.md`。本次配对比较证据为 `local_artifacts/qwen38/newreward_step80_eval_20260827/oldreward16k_vs_newreward24k_step80_compare.20260827.json`，SHA256 `9a1c287390617aeae946a6ed2124f1774883ee2cd02311c23c812f5d34e1364b`。

<!-- ### 主消融：DataV2 step80 → DataV4 step80

| level | DataV2 step80 Correct (%) | DataV4 step80 Correct (%) | 差值 (%) | Correct 95% CI (%) | Compile 差值 (%) | 截断率差值 (%) |
|---|---:|---:|---:|---:|---:|---:|
| L1 | 78.25 | **87.25** | **+9.00** | **[+4.75,+13.38]** | +12.75 | -13.75 |
| L2 | 38.38 | **60.62** | **+22.25** | **[+16.38,+28.25]** | +39.75 | -42.75 |
| L3 | 6.00 | **16.75** | **+10.75** | **[+5.00,+17.25]** | +15.50 | -16.75 |

三档 Correct 的 bootstrap 差值为正概率（%）均为 `100.00`。问题级 pass@8（%，L1/L2/L3）也从 DataV2 step80 的 `98.00/71.00/18.00` 提升到 DataV4 step80 的 `100.00/84.00/28.00`。最显著的伴随变化是 Compile 提升和截断减少：L2 截断率（%）从 49.75 降到 7.00，L3 从 92.50 降到 75.75。这支持 DataV4 改善了在固定 16K budget 内交付完整可编译实现的能力，但仍不能把截断下降认定为数据改动影响 Correct 的唯一中介机制。

速度指标没有一致提高。同 step80 的 Fast@1.0 差值（%，L1/L2/L3）为 `-3.12/-1.75/+0.75`，Fast@1.2 为 `-4.25/-1.38/-0.25`。因此本消融支持“正确率与覆盖显著改善”，不支持“生成的正确 kernel 普遍更快”。Fast 的小差值还可能混入不同 KernelGym 节点、reference cache 和计时波动。 -->

<!-- ### 旧推荐点到新推荐点：DataV2 step40 → DataV4 step100

| level | DataV2 step40 Correct (%) | DataV4 step100 Correct (%) | Correct 差值 (%) | Correct 95% CI (%) | Compile 差值 (%) | Fast@1.0 差值 (%) | Fast@1.2 差值 (%) | 截断率差值 (%) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| L1 | 74.50 | **90.12** | **+15.62** | **[+11.12,+20.38]** | +9.12 | +2.62 | +2.00 | -4.75 |
| L2 | 45.38 | **64.75** | **+19.38** | **[+14.62,+24.12]** | +24.62 | +0.62 | +0.88 | -17.25 |
| L3 | 10.75 | **16.50** | **+5.75** | **[+1.75,+10.50]** | +9.75 | 0.00 | +0.50 | -10.25 |

端点比较的 Correct 差值为正概率（%）为 L1/L2 `100.00`、L3 `99.83`。与同 step80 主消融不同，它同时混入 DataV2→DataV4 和额外 60 个训练 step，不能解释为纯数据效应；它回答的是实际旧推荐 checkpoint 到当前新推荐 checkpoint 的总体收益。

### DataV4 训练内变化：step80 → step100

DataV4 继续训练到 step100 后，Correct（%，L1/L2/L3）达到 **90.12/64.75/16.50**。相对 DataV4 step80 的差值（%，L1/L2/L3）为 `+2.88/+4.12/-0.25`；paired-bootstrap 95% CI（%）分别为 `[+0.25,+5.75]`、`[+0.38,+7.88]`、`[-2.25,+1.75]`，其中 L3 统计上持平。当前 checkpoint 选择因此推荐 DataV4 step100。

完整的四 checkpoint 结果、截断分析、人工样本复核、checkpoint 来源和产物哈希统一维护在 `handoffs/in_progress/qwen38_legacy_datav4_l123_eval_20260824.md`。同 step80 的配对比较证据为 `local_artifacts/qwen38/eval_step40_80/legacy_step80_vs_datav4_step80_correct_bootstrap_20260824.json`（SHA256 `7a137cb4dd6fc72d77be0dff11b8cc5b0434443492efc53d8252de7e84e1c19b`）；DataV2 step40→DataV4 step100 的端点比较证据为 `local_artifacts/qwen38/eval_step40_80/datav2_step40_vs_datav4_step100_correct_bootstrap_20260824.json`（SHA256 `9d6f8978156be2aa2a1ae5a3f6e019c500c683fe1b2f829afb01324e8f37da65`）。 -->
