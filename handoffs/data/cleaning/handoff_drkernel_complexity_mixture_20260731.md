# DrKernel 训练数据复杂度配比

## 摘要

DrKernel 主训练 parquet 中 train/eval 行为一致（mode-same）的候选约 40k 个样本，本文回答一个问题：简单任务是否过多、要不要按复杂度删样本。按有意放宽的 Level1-like 口径（宁可多计简单任务），简单任务占约 29%，低于 30% 上限，结论是不删任何样本，保留全部样本和原始顺序。三分类的 Level2/3 结构标签在 DrKernel 上有明显域偏移，只能当诊断看，不能用来删样本或设采样权重。

本文只讨论复杂度定义、配比决策和复现证据。数据清洗规则、quarantine、外部数据见 [DrKernel 与补充数据清洗交接](handoff_drkernel_and_accepted_v5_cleanup_20260729.md)。

## 配比口径

配比约束是"无算子或单算子任务不超过训练候选的 30%"。这里的 Level1-like 是配比统计桶，不等同于官方 KernelBench Level 1 标签。为了不漏计简单任务，任一条件成立就计入 cap 池：

- 算子 taxonomy 判为 Level1；
- 结构特征的 3-NN 投票判为 Level1；
- 去掉显式 `nn.Identity` 后，程序至多剩一个输入相关计算算子。

结构特征只读最终生效的 `Model` AST，包括 forward 调用数、不同调用数、`self.*` 调用数、构造的 `nn.*` 模块数、forward 语句数、控制流数、算子数、类数、AST 深度和源码行数；`get_inputs()` 不参与复杂度特征。参考集是官方 KernelBench Level 1/2/3 共 250 题。

参考集上的留一法准确率约 91%。这只说明特征能在 KernelBench 内部复原大部分标签，不能证明三分类标签能无偏迁移到 DrKernel。

| 真实层级 | 预测 Level 1 | 预测 Level 2 | 预测 Level 3 |
| --- | ---: | ---: | ---: |
| Level 1（100） | 85 | 15 | 0 |
| Level 2（100） | 0 | 97 | 3 |
| Level 3（50） | 0 | 5 | 45 |

## 配比结果与决策

决策只看保守 Level1-like cap 池。三分类结构桶同时保留作诊断，但不用于删样本或设采样权重。

| 度量 | 样本数 | 比例 | 用途 |
| --- | ---: | ---: | --- |
| 严格预测 Level1 | 732 | 1.82% | 诊断 |
| 新找回的 `nn.Identity`/单算子候选 | 148 | 0.37% | 修正漏计边界 |
| 保守 Level1-like cap 池 | 11,605 | 28.79% | 执行 30% 上限 |
| 30% 约束允许的最多 Level1-like 样本 | 12,300 | 假设非 Level1-like 固定时低于 30% | 上限 |
| 距离上限余量 | 695 | — | 可增加的 Level1-like 样本数 |
| 实际删除 | 0 | 0.00% | 最终决策 |

结构三分类给出 Level1/2/3 约 0.7k/33.7k/5.9k 个样本。人工复核显示后两个桶不能解释为真实的"融合难度"或"架构难度"：抽中的 strict Level3 全是一个或两个算子的简单程序。因此最终只使用有意放宽的 Level1-like 上限，保留全部样本及原始顺序。

## 人工复核与规则修正

复杂度复核共读 60 个互不重复的有效 reference。样本在读码前按 SHA-256 排序从固定分层中选取；这是一组刻意覆盖边界的定性检查，不能解释为总体错误率。

| 复核层 | 阅读数 | 读码结论 |
| --- | ---: | --- |
| 严格 Level1 | 12 | 10 个确为单算子；GRU 与藏在 `ModuleDict` 中的 Linear 共 2 个被低估，属于 cap 的保守多计 |
| 结构 Level1 歧义 | 12 | 全是两到三个算子的融合；说明保守池会主动多计简单任务 |
| 结构 Level2 | 8 | 全是合理的多算子融合 |
| 结构 Level3 | 8 | 全是一个或两个算子的简单程序；证明 Level2/3 标签存在域偏移 |
| 普通 mode-same 对照 | 8 | 输入相关且在审计模式下稳定 |
| 新找回的 `nn.Identity`/单算子样本 | 12 | 全部在去掉 Identity 包装后只剩一个真实计算算子 |
| **合计** | **60** | **没有发现违反保留规则的样本** |

复核还发现了一个测量缺陷：结构签名曾把显式 `nn.Identity` 当成第二个计算算子。例如 `cuda_llm_424147` 的签名是 `nn.Identity` 加 `torch.amax`，实际只有一个有效计算算子。修正规则后新增 148 个 Level1-like 候选；预先选定的 12 个抽样全部支持归入保守 cap 池。该修正把简单任务计数从漏计改为更保守的上界，但约 29% 仍低于 30%，所以最终删除数保持为零。

## 复现与证据

先从完整审计中物化 train/eval 行为一致的约 40k 个样本，再运行复杂度分层：

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

python3 tools/data/cleaning/stratify_ops_complexity.py \
  --input /tmp/drkernel_mode_same.parquet \
  --reference Data/external/converted/kernelbench_level1_2_3.reference.parquet \
  --output /tmp/drkernel_complexity_checked.parquet \
  --assignments-jsonl /tmp/drkernel_complexity.assignments.jsonl \
  --summary-json /tmp/drkernel_complexity.summary.json \
  --max-level1-fraction 0.30 \
  --seed 20260729 \
  --overwrite

sha256sum Data/prompt_tvm_v3/drkernel_rl_thinking.parquet \
  /tmp/drkernel_complexity_checked.parquet
```

| 证据 | 样本数 | SHA-256 |
| --- | ---: | --- |
| 最终训练 parquet | 40,307 | `9e9ffca46022e74c0616f5e272871e76dfd000b6e7937b01685cd0bb11d521e5` |
| cap 前 mode-same parquet | 40,307 | `8bcc9dcda616a881e9afd401d4f3de0da074c0706d8cc0a521a7c3dbf746eddf` |
| 逐样本复杂度分配 JSONL | 40,307 | `2d992ed1b63eacbb3f42a3ea86c699bc81d6d0cf7c9f9db4034e492d1fe5e242` |
| 复杂度汇总 JSON | 1 | `146bef9e70541d2d9be039e14389f4fc4851d4fdca9428a1ffc0287725051daa` |
| 第一轮人工复核 JSON（含 60 个复杂度复核样本） | 132 | `a49c8ca4f34224b233d0e3a5ed995de592fa5634310021c75c8b4dfaf985587d` |

逐样本复杂度分配、汇总和人工复核证据位于：

- `local_artifacts/data_handoffs/drkernel_gpu_v5_mode_same_complexity_20260730.assignments.jsonl`
- `local_artifacts/data_handoffs/drkernel_gpu_v5_mode_same_complexity_20260730.summary.json`
- `local_artifacts/data_handoffs/drkernel_gpu_v5_manual_review_20260730.json`
- `local_artifacts/data_handoffs/drkernel_gpu_v5_level1_noop_recovery_review_20260730.txt`

人工复核 JSON 还包含清洗处置样本；本节表格只统计其中为复杂度校准选取的 60 个样本。

## 使用边界与后续方向

- 当前 30% 结论是保守上限判断；它足以回答"是否需要删简单任务"，不支持精确估计每个难度层级的真实占比。
- Level2/3 标签在 DrKernel 上已经观察到域偏移。后续若要按难度重采样，应显式建模复合模块、数据流和架构图，并重新人工校准。
- 本轮没有做训练集与 KernelBench 的污染检查；参考集只用于复杂度 taxonomy 校准。
- 如果后续把 120 个样本的独立 DrKernel mode-same 分区、鸥波合成数据或外部数据合入主训练 parquet，必须在合并后的最终分母上重新计算 30% 上限，不能直接沿用本文的 28.79%。
