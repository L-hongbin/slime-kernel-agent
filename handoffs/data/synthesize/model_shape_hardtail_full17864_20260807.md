# DSV4-Flash full hard-tail shape expansion

## 结论

DeepSeek-V4-Flash-0731 已遍历静态 solver 遗留的 17,864 个 hard-tail parent，生成
Medium/Large 两个候选，并完成静态 source-span、shape-only、容量、dead-slot、维度平衡
和 FakeTensor 门禁。静态阶段保留 32,217 个 child，覆盖 16,167 个 parent。

三台 H20 随后执行 paired parent/child reference。用户在只剩尾部样本时决定停止：最终
保留 48,306/48,384 条唯一 reference 记录，主动跳过 78 条；缺失集合已显式固化，不能
被默认严格分析或后续工具误认为完整门禁。当前有 24,563 个 child 的 parent 与 child
同时通过 reference，覆盖 13,166 个 parent。changed-region liveness 尚未运行，当前结果
仍是 review-only，不能进入训练，也没有执行 shape 重采样。

## 阶段结果

| 阶段 | 结果 |
| --- | ---: |
| hard-tail parent | 17,864 |
| HTTP 200 generation | 17,864/17,864 |
| completion length-limit | 844/17,864（4.72%） |
| static/FakeTensor child | 32,217/35,728（90.18%） |
| 覆盖 parent | 16,167/17,864（90.50%） |
| paired reference 记录 | 48,306/48,384（99.84%） |
| 主动跳过 reference | 78（23 parent、55 child） |
| 受缺失记录影响的 child pair | 101 |
| reference passed record | 38,168/48,306（79.01%） |
| parent+child both pass | 24,563 child / 13,166 parent |
| changed-region liveness | 未运行 |

`generation/responses.jsonl` 有 17,865 个物理记录：17,864 个 parent 各有一条最终
HTTP 200，另有 `kernelbook_17402_4a875b94a84c95d2` 的一条 `http_status=0` 失败尝试；
统计按每个 parent 的最新成功记录计算，不能直接用物理行数代替 parent 数。

停止前 node53、node69 各完成 16,128/16,128；node70 保留 16,050/16,128，随后只停止
`shape_full17864_ref_r2`，退出码 130、`OOMKilled=false`。三台的 96 个 JSONL 与 96 个
log 已聚合到项目目录，`rsync --checksum --dry-run` 对三台均没有文件内容差异。只有
`shard-95-of-96` 没有最终 shard summary；其 426/504 条逐行记录均可解析且 UUID 唯一。

## Reference 因果口径

全部失败不能直接归因于 shape 扩增。16,144 个已观察 parent 中只有 13,559 个通过；
最大失败签名是 KernelGym 对 tuple 输出访问 `.shape`，共 6,600 条（2,295 parent、
4,305 child）。它同时击中原始 parent 和 child，是 harness/题目兼容性问题，不是新增
shape 的增量失败。

可归因口径固定为“parent reference 已通过，且 child reference 已观察”。该口径共有
26,978 个 child，其中 24,563 个通过，条件通过率 91.05%；2,415 个增量失败的状态为：

| 状态 | 数量 |
| --- | ---: |
| cuda_out_of_memory | 1,724 |
| failed | 468 |
| timeout | 128 |
| error | 83 |
| gpu_memory_telemetry_error | 12 |

Medium/Large 的差距主要由内存压力造成：Large 的 2,025 个条件失败中有 1,506 个
OOM，Medium 的 390 个条件失败中有 218 个 OOM。

| 条件通过率 | passed / observed | rate |
| --- | ---: | ---: |
| Medium | 13,100/13,490 | 97.11% |
| Large | 11,463/13,488 | 84.99% |
| `<256 MiB` | 13,024/13,414 | 97.09% |
| `256 MiB–1 GiB` | 2,559/2,726 | 93.87% |
| `1–2 GiB` | 2,998/3,528 | 84.98% |
| `2–4 GiB` | 5,982/7,310 | 81.83% |

## Bias 与完整 64k 分布边界

reference 门禁没有显著制造新的离散数值 bias。静态候选到 both-pass 候选的 changed
occurrence 指标为：power-of-two 11.23% → 11.03%，接近 100-grid 13.83% → 13.68%，
接近 power-of-two 14.70% → 14.36%，Top-10 share 12.00% → 11.81%。但是原有的
十进制/常见二进制锚点仍然明显；最高频值仍是 256、512、128、64、1024、1000。

runtime 会轻微压低 size 分布。both-pass child 的 aggregate input bytes 分位数为
P50 0.238 GiB、P75 1.940 GiB、P90 3.145 GiB、P95 3.576 GiB、P99 3.918 GiB；静态
候选对应 P50 0.250 GiB、P75 2.149 GiB、P90 3.245 GiB、P95 3.633 GiB、P99
3.934 GiB。4 GiB 输入上限不是唯一限制：算子中间量会远大于输入，因此 2–4 GiB
候选仍有较高 OOM 淘汰率。

来源 bias 比 operator-count bias 更明显。在 parent-pass 条件口径下，CUDA-Agent 为
1,361/1,848（73.65%），DrKernel 为 23,102/25,015（92.35%）；operator-count 三档为
90.06%–95.11%。最终重采样必须把 source 和 size/variant 一起作为约束，不能只优化
shape bucket coverage，否则会进一步降低 CUDA-Agent 与 Large 占比。

在重采样前只能给出候选上界，不能声称已有最终 64,315-row 分布。上游三条静态 lane
已改变 23,358 个 parent；若 model lane 对每个 reference-both-pass parent 最多取一个
child，则最多再改变 13,166 个，得到 36,524 changed、27,791 unchanged，unchanged
share 为 43.21%。其中 11,397 个 parent 同时有 Medium/Large，1,703 个只有 Medium，
66 个只有 Large。changed-region liveness 还会进一步降低可用数量。

## 分析合同和治理

`analyze_runtime_canary.py` 新增显式 `--allow-partial-reference`。默认模式仍要求 reference
UUID 集完全相等并 fail closed；只有显式参数才接受预期 UUID 的真子集。部分分析会写出
准确缺失 UUID 和 both-pass child UUID 清单，并记录 shard-family digest。没有放松
`runtime_evidence.py`、最终 runtime verifier 或 resampler 的完整性要求。

因此当前停止点为：

1. 生成、静态/FakeTensor、部分 paired reference 已整理；
2. 没有 changed-region liveness，`runtime/accepted.parquet` 不存在；
3. 没有执行 shape 重采样；
4. `training_approved=false`，任何下游消费都必须等待新的用户决定，并处理 partial
   reference 与 region validation 合同。

按用户要求，生产数据工具没有新增单元测试。执行了 `black`、`py_compile`、真实 96-shard
replay、UUID/JSON 完整性检查和三台 rsync checksum 对照。

Kimi 对原始 shard、355 MiB manifest、parquet 与真实 diff 做了独立重算，结论为
**PASS**，无 mandatory fix；PASS 仅认可证据、因果口径和治理边界，不批准训练或
重采样。其可选观察包括 raw response 多一条失败重试、complete run 可能遗留旧 missing
UUID 文件，以及 cluster-side stop/rsync 元数据无法仅从聚合文件反证。

## 可复核产物

- `Data/prompt_tvm_v4/shape_model_hardtail_v2/run.full17864/selected.parquet`：
  `883a73a88e473fc71a907c8eb65ce31cc760f6140029d996bf75e96268a592f7`。
- `generation/responses.jsonl`：
  `e6f9a2a799ea2eb9b79057ccc487f1b31c94e31340973bef1097ceac67f76a59`。
- `static/children.parquet`：
  `10d0e73b8bc90a02ec82370445d6e70c82aaa5a66178d3b71403c11371058847`。
- `static/paired.parquet`：
  `54d0a9805a6a8bc2f93f9451a028181574a941124231d36a89a7fc6eef553aaf`。
- `static/manifest.json`：
  `de3430a1a1e847b3c3d17ab2977eb68b4d4819f57b4afad83c05d5e63656548b`。
- `analysis/bias.json`：
  `0ef5c140ca52d7d680f7d17666c264860b11eb5a88cad7e850ec201a51cb6ad8`。
- `h20/reference/`：96 shard、280 MiB；shard-family digest
  `2eaafb8e7ea5e658a9688a61566973629f18ad064a670b982716e0903481fc18`。
- `analysis/runtime_diagnostics.json`：完整 partial/reference/bias 诊断。
  SHA-256 `6848e2f20bc0ac2378b2fc30b7abfb9771bb64d7c25e48ee0d41fbaa47e20699`。
- `analysis/reference_missing_uuids.txt`：78 个主动跳过 UUID，SHA-256
  `a12283137e8434d89595caef911b7fc40b5944fb18818442d2f3ba84c1bc3430`。
- `analysis/reference_both_pass_child_uuids.txt`：24,563 个候选 UUID，SHA-256
  `44b386fdd6d246adf8553896f083de9ff89ab9bb79d3fdc72bde7066cf1879aa`。
- `analysis/runtime_review_samples.md`：真实 parent/child diff 与失败样本。
- `review/kimi_full_reference_verdict.md`：Kimi 独立 PASS，SHA-256
  `91498f9e4f9756beb378813b9d5f3a952612ca6163b71ccc72dc3dfeeb658628`。
