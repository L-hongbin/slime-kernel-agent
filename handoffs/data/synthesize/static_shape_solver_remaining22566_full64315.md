# Shape hard-tail 22,566 与全 64,315 行分布

## 结论

第三条 static-solver lane 处理前两条 shape lane 未覆盖的 22,566 个 parent。它接受 994 个 static child，经完整 H20 paired reference 与 changed-region 验证后保留 852 个 runtime-eligible child。三条 lane 合计得到 23,358 个互不重叠的 runtime-eligible shape child，覆盖 64,315 个 canonical parent 的 36.32%。

为回答“全部 child 替换 parent 后整个 64k 数据集是什么分布”，本轮另构造了一个严格等基数、仅用于分析的假想替换集：23,358 行用对应 child 替换，40,957 行保留原 parent，总数仍为 64,315。它不是训练产物，`training_approved=false`；正式数据合同仍保持 parent immutable、child additive。

全量 tensor-factory occurrence 的 numel 分布从 canonical parent 的 P50/P90 `8,192 / 4,000,000` 提升到 `786,432 / 399,507,456`，但仍低于 KernelBench 的 `33,554,432 / 1,610,612,736`。因此 shape coverage 明显改善，但没有对齐 KernelBench。

## 第三条 hard-tail lane

输入位于：

`Data/prompt_tvm_v4/shape_solver_variable_multislot_v8/run.remaining22566.balanced_v7/`

selector 从旧 exact-two run 的 53,896 个 parent 决策中选择 parent FakeTensor 已通过、exact-two 未接受、且不属于上一条 12,318-parent variable lane 的全部剩余项。计数闭合为：

| 项 | Parent |
| --- | ---: |
| 旧 exact-two static accepted | 17,353 |
| 旧 parent FakeTensor 未通过 | 1,659 |
| 上一条 variable lane | 12,318 |
| 本条 remaining lane | 22,566 |
| 合计 | 53,896 |

solver 保留既有 `Medium`/`Large` target、4 GiB aggregate input 上限、2--5 logical slots、全局 30%--50% P2 门禁和 1000:1 维度比例门禁。只替换已声明的十进制 shape span；`Model`、`get_init_inputs()`、factory 顺序、dtype、rank 和其他源码不变。

### Static 结果

| 结果 | Parent | 占 22,566 |
| --- | ---: | ---: |
| accepted | 994 | 4.40% |
| no exact product profile | 17,862 | 79.15% |
| child candidate FakeTensor failed | 3,702 | 16.40% |
| child candidate FakeTensor timeout | 6 | 0.03% |
| dimension-balance guard | 2 | 0.01% |

低 yield 的主因是这是前两条 lane 筛剩的结构 hard tail：17,862 个 parent 不在当前 same-factory/product 静态证明域内。它不是 4 GiB 上限造成的；容量上限主要影响最终右尾能达到多大。

994 个 static child 中 `Medium=515`、`Large=479`；logical slots 为 K2=112、K3=872、K4=10，没有 K1/K5。5,965 个 changed occurrence 的 P2 占 43.10%；1000:1 违规为 0，child 最大维度比最大为 633.20。独立 analyzer 重放 994/994 个 child，invalid 为 0。

### H20 结果

既存 artifact contract 固定为 `4 logical ranks × 8 GPUs × 4 virtual shards = 128 shards`。用户将 node64 排除后，rank 1/2/3 分别在 node53/node69/node70 运行，rank 0 在 node69 完成 rank 2 后串行复用；即 4 个逻辑 rank 实际只使用 3 台物理 H20，node64 未执行 GPU 验证，其 oscar-hy3 服务未改动。

| 阶段 | 完整性 | 结果 |
| --- | --- | --- |
| paired reference | 128/128 shards，1,988/1,988 rows | 1,798 pass，190 fail；both-pass 880，parent-only 38，neither 76，child-only 0 |
| changed region | 128/128 shards，880/880 children | 852 pass，12 rejected，14 unsupported，2 OOM |

12 个 rejected 全部为 `expanded_region_no_output_effect`，均来自 KernelBook；这是“原 input 有效、但扩出的 tail 被切片/裁剪或其他 forward 逻辑丢弃”的 region-specific dead-tail，不等同于 v5 清洗针对的整体 dead argument。14 个 unsupported 分为 positional-forward 映射失败 4、non-finite output 1、unchanged control 不精确 7、region effect 不一致 2。

最终 852 个 eligible child 为 `Medium=449`、`Large=403`，来源为 KernelBook 744、DrKernel 98、Oubo 8、CUDA-Agent 2。K2/K3/K4 为 81/763/8。5,068 个 changed occurrence 的 P2 占 43.45%，805 个 exact value 的 entropy effective values（`2^H`）为 163.98，Top-5 占 40.49%；Top-5 全是 64/128/256/512/1024 一类 P2 anchor，说明这条 hard-tail lane 仍存在明显二进制吸附。aggregate input bytes P50/P90/max 为 261,750,784 / 3,856,131,686 / 4,294,950,976；1000:1 违规为 0。

## 三条 lane 合并

| Lane | Runtime eligible | Medium | Large |
| --- | ---: | ---: | ---: |
| historical exact-two | 13,743 | 7,112 | 6,631 |
| variable 12,318 | 8,763 | 4,962 | 3,801 |
| remaining 22,566 | 852 | 449 | 403 |
| 合计 | 23,358 | 12,523 | 10,835 |

三条 eligible parent 集合 pairwise disjoint。合并 logical-slot 分布为 K2=19,852、K3=3,489、K4=17，没有 K1/K5。59,087 是 changed-occurrence 原始行数（按 `(child, slot)` 去重后为 50,239），其中 P2 占 44.45%；虽然有 9,569 个不同值，inverse-HHI effective values（`1 / sum(p^2)`）只有 48.95，Top-5/Top-10 占 28.77%/41.04%，前十高频值全部是 2 的幂。这里的 effective-value 口径不同于上文 lane 级 bias audit 的 entropy `2^H`。故“不是固定一半 P2”已经做到，但最终数值分布仍有强 binary-anchor bias。

eligible coverage 还存在明显 source/operator bias：

| Source | Eligible / canonical parent | 替换率 |
| --- | ---: | ---: |
| CUDA-Agent | 642 / 4,995 | 12.85% |
| DrKernel | 14,540 / 45,505 | 31.95% |
| KernelBook | 7,550 / 11,814 | 63.91% |
| Oubo | 626 / 2,001 | 31.28% |

| Operator bucket | Eligible / canonical parent | 替换率 |
| --- | ---: | ---: |
| ≤1 op | 1,650 / 2,810 | 58.72% |
| 2--5 ops | 17,704 / 49,152 | 36.02% |
| ≥6 ops | 4,004 / 12,353 | 32.41% |

假想替换不改变 source 行数或 operator/factory 结构；它只说明不同 source/operator 中有多少行获得了大 shape。因此 KernelBook 的规模先验被改得远多于 CUDA-Agent。

## 整个 64,315 行的假想替换分布

分析产物位于：

`Data/prompt_tvm_v4/shape_full64315_substitution_analysis_v1/`

构造器逐项验证 eligible UUID、children parquet SHA、canonical `parent_uuid`、三条 lane 无交集和固定行数。结果为 23,358 replaced + 40,957 unchanged = 64,315；manifest 有 23,358 行。AST profiler 不执行 reference code，canonical/substituted 均为 0 parse failure、0 missing `get_inputs()`、0 operator-signature failure。operator count、operator-family presence 和 input-factory presence 的 canonical/substituted 摘要逐字节一致。

| Per-tensor numel | Canonical 64,315 | Shape-substituted 64,315 | KernelBench 250 | KB / substituted |
| --- | ---: | ---: | ---: | ---: |
| P50 | 8,192 | 786,432 | 33,554,432 | 42.67× |
| P75 | 196,608 | 53,865,588 | 268,435,456 | 4.98× |
| P90 | 4,000,000 | 399,507,456 | 1,610,612,736 | 4.03× |
| P95 | 16,777,216 | 689,831,936 | 2,146,959,360 | 3.11× |
| P99 | 268,435,456 | 1,017,249,792 | 2,147,483,648 | 2.11× |

| Per-tensor numel tail | Canonical | Shape-substituted | KernelBench |
| --- | ---: | ---: | ---: |
| ≥1M | 19.20% | 49.72% | 88.61% |
| ≥10M | 6.38% | 43.16% | 62.45% |
| ≥100M | 2.41% | 18.87% | 42.19% |

残余差距有两个不同原因：

1. coverage：63.68% parent 仍未替换，而且被替换的 task 也只改静态证明为 live/coupled 的部分 factory/slot。这是中位数和 ≥1M coverage 仍明显不足的主因。
2. 上限：4 GiB 是 aggregate input-byte 门禁。对 float32 单 tensor，它天然限制在约 1.07B elements，而 KernelBench P95/P99 已接近 2.147B elements；最终 substituted P99=1.017B 与这个 ceiling 对齐。因此右尾差距确实部分来自 4 GiB cap。提高上限可能补右尾，但不会解决 40,957 个 unchanged parent 或 static proof-domain 缺口。

## 复核与证据

人工检查 `analysis/review_samples.md` 的 16 个真实 diff：只修改 shape 整数，没有 docstring、forward、程序结构、factory/rank 或非 shape 内容；其中 15 个进入最终 runtime eligible，另 1 个由 H20 门禁排除。全量 source-span identity/analyzer 对 994/994 通过。

关键哈希：

| Artifact | SHA-256 |
| --- | --- |
| remaining selected parquet | `f79907057d3588a651f08e84262c4a591df42a53967ffcedddde12683e55f916` |
| remaining children parquet | `ba138e8f7b9b0ad2d1d9d9e6f8cfbf40d5aae76db2fd4fd69cd64a13b05737f2` |
| remaining paired parquet | `09fa55559b55ca8e9d5a9945fb0e33f6325214e2b7fa2e8478c676659a737bca` |
| remaining static manifest | `8ad63a1f10b2b8a82853805867241e602bc179208c698dce33e23783c3bc9d4d` |
| final remaining summary | `f49181b6aefe5c4b779b54f35ef2ff72a624855c02db2e5589528bf41dbb6e7f` |
| final remaining bias audit | `7003d17d9cb923c3ddf36b6137eed4f7a1e967f75328b29b330921a8b8b00542` |
| reference allowlist | `7560ad3e01220ab9a5a9b40c3d56052f83a6bfafa1bdab911a5de9a50b0ce6db` |
| hypothetical substituted parquet | `869b82e9f4d04c3bc48d680b5c5f908542b83d4ca6567b12cc001920585839a0` |
| substitution manifest | `a0a95ee7895ceb2b0b5c83eec8f6e858a8abd86919cd46e13ddedd90e1561714` |
| full-64k analysis summary | `10bdb93d1ca26ddcd27a8a06efc1e15d8b49097c6bedd2a4023b5d056c6e4fd8` |

reference verifier fingerprint 为 `860181999e53ba32495fdfc347abcec1848b7634e7a8a7b95ff3e31039b5d65b`；region validation binding 为 `c87351969209e598f33ba4e5696d3fe3306ac8f71b664a650a626fada2cac9a8`。按要求没有为生产数据工具新增单元测试；验收使用真实 static artifact、128/128 H20 shard exact verifier、runtime-quality analyzer、全 64k AST profile 和人工 diff。

### Kimi milestone review

Kimi K3 high-thinking session `session_8b3f6470-75fa-4a9e-9411-b3784a7a972c` 对 artifact、哈希、H20 shard、三 lane/64k 闭合、分布统计和真实 diff 做了只读独立复算，最终给出 `VERDICT=PASS`、`SHAPE_COMPLETE=YES`、`BLOCKING_ISSUES=none`。非阻断遗留项是 binary-anchor 与 source/operator coverage bias、64k summary 尚未单独固化所有 merged-lane 统计，以及后续运行应把 per-rank hostname 写入 artifact；这些均不改变本轮 shape 阶段可以收尾的结论。
