# 可变多维 shape 扩展：12,318 条恢复集

## 摘要

这条 lane 用静态 solver 扩展 12,318 个旧 exact-two lane 无法处理的 parent，不调用生成模型。每个 parent 只分配一个确定性随机 `Medium` 或 `Large` target，只生成一个 child；solver 同时修改 2--5 个逻辑 shape slot，并用 source-span identity、FakeTensor、4 GiB 容量上限和 1000:1 维度比例门禁拒绝不可靠结果。

全量 CPU 阶段接受 11,877 个 child，覆盖率 96.42%。所有 child 都通过独立源码重放与 AST/factory/rank 校验；静态分布门禁通过。1,000-parent canary 的 H20 paired reference 与 changed-region liveness 最终接受 713 个 child，也通过 runtime 分布门禁。全量 H20 reference 已按固定 2×8 H20、128 shard 合同启动，但 reference、region 和最终 runtime analyzer 尚未全部完成，因此当前产物仍是 review-only，`training_approved=false`。

| 阶段 | 结果 | 结论 |
| --- | ---: | --- |
| 精确输入选择 | 12,318 parent | `Medium=6,151`，`Large=6,167` |
| 全量 static/FakeTensor | 11,877 child | 96.42%，identity 与 static bias 门禁通过 |
| 1k canary static | 957 child | 95.70% |
| 1k canary H20 最终 eligible | 713 child | 74.50% static retention，runtime bias 门禁通过 |
| 全量 H20 | reference 运行中 | 尚无完整 runtime-eligible 集合 |

## 输入与变换合同

输入来自 full 53,896-parent solver 中因 strict exact-two 约束而跳过的候选。selector 重新证明 slot span、factory set、axis、occurrence 数和 exact storage witness，得到 12,318 个可恢复 parent；它只选择 parent，不生成 child。输入固定在：

`Data/prompt_tvm_v4/shape_solver_variable_multislot_v5/input.recoverable12318/`

目录名保留历史编号；当前生产合同是 `shape_variable_multislot_solver_v7`，generator 是 `same_factory_product_variable_2_to_5_balanced_nonleading_soft_p2_50_v1`。

核心约束如下：

- 每个 parent 只有一个 variant 和一个 child，variant 在 `Medium` 与 `Large` 间确定性分配。
- `Medium` target 为 64--256 MiB；`Large` 为大于 256 MiB且不超过 4 GiB。允许相对 target 误差最多 25%，不再使用 24×/49 GiB 静态门禁。
- 每个 child 修改 2--5 个逻辑 slot；linked slot 即使同步出现在多个 input factory 中，逻辑上仍只计一次。
- 同组 slot 必须作用于相同的非空 direct-input factory set，在每个 factory 中各出现一次且 axis 互异；storage 必须能写成精确的 product 形式。
- 只允许替换已声明的十进制 shape 整数字面量。`Model`、`get_init_inputs()`、input factory 顺序、dtype、rank 和所有未声明源码 span 必须不变。
- parent 与 child 都必须通过 FakeTensor。每个最终 rank≥2 direct factory 必须满足最大维不超过第二大维的 1000 倍。
- 2 的幂只是确定性的 per-slot soft preference，不设“每个 child 恰好几个”的局部模板。最终只要求 changed occurrence 的全局 P2 占比落在 30%--50%。

`balanced_nonleading_no_explicit_batch` scope 指所有被选 occurrence 都在非 leading axis，且 linked symbol 不属于 `batch_size`、`batchsize`、`batch`、`bs`、`n_batch`。solver 在结构允许时偏向该 scope，同时为 general lane 保留候选与最多两次 bounded attempt，防止为了位置偏好大幅损失覆盖率。每个 parent 总共最多做六次 child FakeTensor 尝试。

## 1k H20 canary

权威 canary 位于：

`Data/prompt_tvm_v4/shape_solver_variable_multislot_v6/run.canary1000.balanced_v7/`

目录名同样是历史名称；manifest 合同为 v7。静态阶段接受 957/1,000 个 child。H20 reference 共验证 1,914 行 parent/child：763 对 parent 与 child 同时通过，86 对只有 parent 通过，108 对两边都未通过，没有 child-only pass。

region 只处理 763 个 reference 双通过 child，结果为 713 passed、28 OOM、17 rejected、5 unsupported。17 个 rejected 都是 `expanded_region_no_output_effect`，说明扩出来的区域在三次确定性扰动中没有影响输出；这正是原始参数整体 live、扩展后新增 tail 却 dead 的情形。OOM 和 unsupported 保持 inconclusive，不计入 eligible。

| Canary 指标 | Static | Runtime eligible |
| --- | ---: | ---: |
| Child | 957 | 713 |
| Medium | 480 | 400 |
| Large | 477 | 313 |
| K2 / K3 / K4 | 707 / 249 / 1 | 512 / 200 / 1 |
| Changed occurrences | 2,431 | 1,789 |
| P2 fraction | 36.90% | 37.56% |
| Effective values | 375.48 | 308.64 |
| Top-5 value fraction | 21.39% | 22.14% |
| Explicit-batch child fraction | 74.61% | 72.23% |
| 1000:1 violation | 0 | 0 |

Runtime retention 显著依赖 variant 与来源：`Medium=83.33%`，`Large=65.62%`；`cuda_agent=51.90%`，`drkernel=76.51%`。这说明 4 GiB input cap 和 FakeTensor 不能替代真实 H20：attention、convolution、reduction 的中间量和 workspace 仍可能 OOM。

## 12,318-parent 全量 static 结果

四台机器的 CPU 各用 64 worker，完成 1,024/1,024 个 deterministic shard；merge 后的 canonical run 为：

`Data/prompt_tvm_v4/shape_solver_variable_multislot_v5/run.recoverable12318.balanced_v7/`

| Variant | Selected | Accepted | Coverage |
| --- | ---: | ---: | ---: |
| Medium | 6,151 | 5,920 | 96.24% |
| Large | 6,167 | 5,957 | 96.59% |
| 合计 | 12,318 | 11,877 | 96.42% |

| Logical slots | Child | 占 accepted |
| ---: | ---: | ---: |
| 2 | 8,428 | 70.96% |
| 3 | 3,440 | 28.96% |
| 4 | 9 | 0.08% |
| 5 | 0 | 0% |

K2 仍占多数不是 exact-2 硬编码，而是当前 same-factory/product 证明下，很多 parent 没有兼容的 K3--K5 group，或更高 cardinality 候选无法通过 shape coupling 的 FakeTensor 门禁。solver 没有为了追求 slot 数去改 forward 或程序结构。

441 个未接受 parent 的最终归因是：420 个 child candidate FakeTensor failure、3 个 FakeTensor timeout、17 个所有数值候选都被 1000:1 门禁拒绝、1 个没有可行 product 解。候选级别共观察到 2,177 次 FakeTensor failure 和 12 次 timeout；每个 parent 的 bounded 尝试仍不超过六次。

## 全量偏差与人工复核

全量 11,877 个 child 共改变 30,462 个 direct-factory dimension occurrences。P2 占 36.82%，不同值 6,513 个，effective values 为 891.17，Top-5 占 20.39%；8 的倍数占 43.45%，32 的倍数占 35.42%。容量 P50 为 271.68 MiB，P90 为 3.24 GiB，最大值恰为 4 GiB。每个 child 的最大维度比 P50/P90/P99 为 2.43/33.60/310.93，最大 1000，越界为 0。

balanced scope 接受 2,909 个 child，general lane 接受 8,968 个。75.51% child 触及 leading axis，74.93% 触及显式 batch symbol；按 occurrence 统计分别是 34.07% 和 33.83%。主要原因不是 selector 忽略偏好，而是 7,232 个已接受 child 根本没有兼容的 scope group，只能 general fallback。该偏差必须在全量 H20 筛选后重新统计，不能用 canary 或 static 比例替代最终结果。

独立 analyzer 对 11,877/11,877 个 child 重放 source spans，并重新解析实际 factory/axis diff；`Model`、`get_init_inputs()`、factory sequence、rank、hash、lineage、parent/child FakeTensor 和 1000:1 contract 全部通过。额外的全量 tokenizer diff 显示 0 个 child 修改了非十进制整数字面量。`analysis/stratified_review_samples.md` 中的 11 个 accepted diff 和全部 9 个稀有 K4 child 均人工复核，只改 shape 数值，没有 docstring、等价重写、4D→2D 或 forward 结构修改。

## 全量 H20 合同与完成边界

全量 paired reference 的输入为 23,754 行，已固定以下 scheduler contract：

| 节点 | Rank | GPU | Virtual shards/GPU |
| --- | ---: | ---: | ---: |
| `node53_slime` | 0 | 8×H20 | 8 |
| `node69_slime` | 1 | 8×H20 | 8 |

总 shard 数为 128，单行超时 180 秒，allocator 上限 64 GiB，paired parquet SHA-256 为 `4a9bc3b678508255cfe0a6c6f82304f62459a996de8c40eb9c3df257a935a318`。启动前两台机器的输入与 launcher/validator 源码均逐文件 hash 一致。node64 和 node70 正被与本 lane 无关的任务占用，因此未抢占、未停止其服务；reference topology 一经落盘不能再追加节点。

reference 完成后必须先把两个 node-local shard family 无覆盖地汇集到 canonical run，再执行：

```bash
python -m tools.data.synthesize.verify_shape_solver_runtime_shards \
  Data/prompt_tvm_v4/shape_solver_variable_multislot_v5/run.recoverable12318.balanced_v7 \
  reference \
  --expected-machine-count 2 \
  --expected-gpus-per-machine 8 \
  --expected-virtual-shards-per-gpu 8

python -m tools.data.synthesize.analyze_shape_solver_run \
  Data/prompt_tvm_v4/shape_solver_variable_multislot_v5/run.recoverable12318.balanced_v7
```

第一次 analyzer 只负责从完整 reference 生成 parent+child 双通过 allowlist。region 必须使用该精确 allowlist，并重新检查新增区域的 output effect。region 可以按启动时的空闲节点固定自己的 topology；最终 verifier 必须同时显式传入 region topology 和已冻结的 reference `2×8×8` topology。只有 region exact checker 与 `analyze_shape_solver_run --require-runtime-quality` 都通过后，才会产生全量 runtime review 集；它仍不会自动获得训练批准。

## 证据与哈希

| Artifact | SHA-256 |
| --- | --- |
| 输入 `selected.parquet` | `fc780d5ce11124c732b5edb9b965ceb9f190d43206cd8635ded1fe0a9b1cca1a` |
| 输入 `selection.json` | `7f4a83f974c728d1b2763ef6136a2db8fee8bb31be6955c43fcce27fd0633e63` |
| 1,024-way `shards.json` | `5e5ba62394c42a8fb2260aeddb59dd030b4fd2674d27a944d29805773249d81d` |
| Full `static/children.parquet` | `4982dc0815f5947cd61cfeb01e1d35cfb679ce24e0d7ef96337c942a7abd5e44` |
| Full `static/paired.parquet` | `4a9bc3b678508255cfe0a6c6f82304f62459a996de8c40eb9c3df257a935a318` |
| Full `static/manifest.json` | `d2bf7460444633f6c46e0f8715cf7e6733b0796817ca3da02542f09f9aa09a15` |
| Full `analysis/summary.json` | `751340b58f94f032b68a92abc478b3272745a375e7b76b6d780917085b6e4a08` |
| Full `analysis/shape_bias_audit.json` | `540e68af560ee0619bc456d897dbe72dddd3add27f6c7dfaa8fc52b6eae33740` |
| Full `analysis/stratified_review_samples.md` | `8608a16e717f32745aaa3d84af8bae271979ca39ead9fda90805aedbc08543c2` |
| Canary `static/manifest.json` | `8e0c925aa1527688de3a115cbedc808a5568f4d21a6344ab98ec9ad8d154aefb` |
| Canary runtime `analysis/summary.json` | `dbb6da841793a081a00745d38a40dde8feb7be4a30a81c52572e894fda99f741` |

| Source | SHA-256 |
| --- | --- |
| `solve_variable_shape_delta.py` | `26c2298d6cb7cf5bfbd1320a56b665c2aa0d6d42d4235ee9c48dfd5efa1e3fe9` |
| `launch_shape_solver_cpu_shards.sh` | `3ee9e2f4807b29a1f0d7b7157472e65e1e724d65590934d2008a18febcf87470` |
| `launch_variable_shape_solver_cpu_shards.sh` | `16f1e1697def66020f9e649acdc94f94d79b7ecc0be8df6e4dea07cd57888698` |
| `analyze_shape_solver_run.py` | `839f00e63bcae976130bfc262d7f563d6b8fbdd9b74d4d49d1b10a6d3ebbd8e7` |
| `validate_shape_region_liveness.py` | `988e747beede5508aaaa9c5ef0404ce2d2cb55f8716e193b79aeee00f5b5347b` |
| `verify_shape_solver_runtime_shards.py` | `22e416187b825aafb239b8bcd8e266141d547cb6c336146c6951d3a8835f1a3c` |
| `launch_reference_validation_shards.sh` | `da63f011263caa0114d754090338304b57283002e814092b9de7ba63d46a4a38` |
| `launch_shape_region_validation.sh` | `b1fa357bea9a7978a97325826bb1c4cade31eae657c1b323f89c66fe96cdf29f` |

按要求没有为这些生产数据工具新增单元测试。验证采用 `py_compile`、`bash -n`、CLI smoke、真实 manifest tamper rejection、deterministic replay、完整静态 artifact 和 H20 canary evidence。
