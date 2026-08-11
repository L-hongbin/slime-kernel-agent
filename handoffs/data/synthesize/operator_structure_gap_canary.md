# 低层算子与程序结构扩增 canary

## 摘要

现有训练集与 KernelBench 在程序结构、算子族和输入生成模板上有明显差距。本次用 33 个封闭模板构造 1,000 个 parentless 任务，重点补充低层 convolution、normalization、pooling、layout，以及长单类和多 class 结构。全部任务使用规则生成，没有读取 KernelBench 源码，也没有调用 DSV4F

最终有 996 个样本同时通过静态合同、KernelGym H20 reference 和三次算子返回链验证，有效率为 99.6%。四个 family 都超过 canary 门槛，33 个模板也都保留了有效样本。四条失败均来自 channels-last 非连续输入上的 `torch.flatten`，reference 结果正确，但现有运行时证据无法把底层 `clone -> _unsafe_view` 证明为 flatten 返回链，因此保守拒绝

这批数据证明了规则法可以稳定补充低层算子和结构尾部。它是定向 coverage canary，尚未解决整体分布差距：并入 40,307 个训练样本后，convolution 和 normalization 占比只升到 15.70% 和 7.10%，仍低于 KernelBench 的 45.20% 和 19.60%；本批也没有新增 matmul/linear。产物保持 `review_only` 和 `training_approved=false`

## 背景

训练集集中在短的多算子组合，KernelBench 同时包含较多单算子任务和长模块。训练集的 convolution、normalization 和 matmul/linear 覆盖也明显偏低，`get_inputs()` 则大量复用少数模板

| 差异 | 现有训练集 | KernelBench |
| --- | ---: | ---: |
| 输入相关算子数 ≤ 1 | 2.48% | 36.40% |
| 输入相关算子数 ≥ 10 | 0.81% | 7.20% |
| 源码行数 ≥ 50 | 1.98% | 20.40% |
| 模型拥有的 `nn` 模块 ≥ 5 | 1.50% | 12.80% |
| 多个顶层 class | 0.12% | 5.60% |
| convolution | 14.30% | 45.20% |
| normalization | 5.58% | 19.60% |
| matmul/linear | 17.49% | 32.40% |

严格比较调用标签时，KernelBench 有 19 个训练集未出现的标签。最终有效集只命中其中 1 个：`torch.randn` 在 `forward` 中出现 15 次，不是 `get_inputs()` factory；其余 18 个标签仍为 0。这个集合混有架构名、helper 名、变量名和一个提取误差，不能直接当作低层算子清单

| 标签类型 | 数量 | 本次处理 |
| --- | ---: | --- |
| `torch.randn` | 1 | 在 `forward` 中新增 15 个有效样本，且输入值仍可达输出 |
| `self.hidden.copy_` | 1 | 暂缓，需要独立的 stateful-write 验证合同 |
| `self.cls_token.expand`、`self.hidden.to` | 2 | 接收对象名造成的精确标签差异，不通过伪造变量名补齐 |
| `tensor.sqrt` | 1 | 实际来自 `math.sqrt` 的提取误标，不构造假覆盖 |
| 架构、helper 和外部 wrapper 名 | 14 | 不复制 `InceptionModule`、Transformer、builder/helper 等高层实现 |

因此，本次以算子语义、拓扑和可验证结构为目标，不追求 19 个字符串全部命中

## 方法

### 封闭模板

生成器固定了四个互斥 family 和 33 个模板。每个模板都声明源码调用、预期 ATen identity、每次 trial 的最少调用次数，以及目标算子必须到达最终单 Tensor 输出的合同

| family | 构造数 | 最终有效数 | 主要覆盖 |
| --- | ---: | ---: | --- |
| `atomic_low_level` | 390 | 386 | Conv/ConvTranspose 1D–3D、Batch/Group/InstanceNorm、3D pooling、div/sub、sigmoid、dropout、flatten、contiguous、unfold、channel permute、forward `randn` |
| `conv_norm_chain` | 330 | 330 | Conv2d→BN2d→pool、depthwise→BN2d→pointwise、Conv3d→BN3d→pool、deconv→norm、Conv1d→BN1d→pool |
| `long_single_class` | 160 | 160 | 50 行以上、至少 5 个 registered module、至少 10 个 forward call 的 1D/2D/3D 长模块 |
| `modular_multiclass` | 120 | 120 | 2–3 个可达顶层 class、至少 5 个 registered module、至少 10 个 forward call |

多 class 和长模块都能由规则稳定表达，静态 reachability 也能完整证明，所以没有部署 DSV4F。模型生成仍可用于规则难以表达的开放结构，但需要单独的生成、去污染、调用图和人工复核合同

### 输入生成

最终 996 个样本全部以 `torch.rand` 作为 `get_inputs()` 的随机 factory，其中 956 个单输入、40 个双输入。输入构造同时改变 shape、stride、storage offset 或 batch 内数值关系，不使用注释、docstring 和 dead code 制造表面差异

| 输入构造 | 样本数 | 语义 |
| --- | ---: | --- |
| `direct_rand` | 264 | 直接生成连续 tensor，其中 40 个为 div/sub 双输入 |
| `offset_slice` | 188 | 从更大的 tensor 切出非零 storage offset 的连续 view |
| `strided_slice` | 185 | 通过步长切片生成非连续输入 |
| `channel_last_view` | 179 | 末维生成后用 `movedim` 形成非连续 channel-first view |
| `broadcast_repeat` | 180 | 先沿 batch `expand`，再 `clone` 成正 stride 连续 tensor；batch 内数值重复 |

最终输入中没有 zero-stride tensor，364 个输入非连续，188 个输入具有非零 storage offset。完整 `get_inputs()` AST 有 943 种，最大重复组为 3；擦除常量并对局部变量 alpha-renaming 后有 174 种 skeleton，最大重复组为 20

### 静态检查

静态检查同时约束以下内容

- 重新提取完整算子签名，并核对模板要求的调用 multiset
- 从整个 `Model` class 的构造和调用关系递归解析顶层 helper class，拒绝完全不可达的 helper 定义
- 核对唯一同步 `get_inputs()`、输入数量、factory、rank、返回结构和 input skeleton 配额
- 要求唯一顶层 `Model`、唯一 `forward` 和唯一顶层同步 `get_inputs()`；import 仅允许 `torch`、`torch.nn` 和 `torch.nn.functional`
- 禁止 `from import`、`exec`、`eval`、`__import__`、动态 `getattr` 和 `setattr`
- 要求 1,000 份 reference SHA 和 normalized AST 都唯一
- 对现有训练集、v4 review 集、既有 semantic/frontier canary 和 KernelBench 250 题做 exact reference 与 normalized AST 去污染

### GPU 验证

GPU 检查分两层。KernelGym H20 reference 在 train mode 下运行五次，保持模型实例，固定 paired execution 的初始化与 RNG，检查编译、单 Tensor 输出、数值、state 和显存合同。三次 liveness 再检查声明的 ATen identity、最少调用次数和返回链 provenance，同时核对 RNG、mode、parameter/buffer、object/storage alias 与禁止的写操作

运行证据固定为 8 个分片，每个分片 125 个样本。finalizer 会重放静态生成，精确连接 reference、allowlist 和 liveness UUID，并绑定 candidate、manifest、8 个分片、launcher summary、scheduler contract 和源码 SHA。缺文件、分片归属错误、路径或 SHA 漂移都会拒绝物化

## 结果与分析

### 数据漏斗

| 阶段 | 样本数 | 结果 |
| --- | ---: | --- |
| 静态闭合注册表 | 1,000 | 33 个模板和四个 family 配额完整，去污染无碰撞 |
| KernelGym H20 reference | 1,000 | 1,000/1,000 通过，五次 trial 均完成 |
| 三次算子返回链验证 | 1,000 | 996 通过，4 条 flatten provenance 拒绝 |
| 最终有效集 | 996 | 全部 family 超过门槛，33 个模板均有保留 |

| family | 最终有效数 | canary 门槛 | 余量 |
| --- | ---: | ---: | ---: |
| `atomic_low_level` | 386 | 350 | 36 |
| `conv_norm_chain` | 330 | 300 | 30 |
| `long_single_class` | 160 | 140 | 20 |
| `modular_multiclass` | 120 | 105 | 15 |

### 分布

![低层算子与程序结构 canary 的覆盖分布](figures/operator_structure_gap_canary_coverage.png)

| 结构条件 | 现有训练集 | KernelBench | 最终有效 canary | 训练集 + canary |
| --- | ---: | ---: | ---: | ---: |
| 输入相关算子数 ≤ 1 | 2.48% | 36.40% | 35.24% | 3.27% |
| 输入相关算子数 ≥ 10 | 0.81% | 7.20% | 28.11% | 1.46% |
| 源码行数 ≥ 50 | 1.98% | 20.40% | 28.11% | 2.61% |
| 模型拥有的 `nn` 模块 ≥ 5 | 1.50% | 12.80% | 28.11% | 2.15% |
| 多个顶层 class | 0.12% | 5.60% | 12.05% | 0.41% |

| 算子族 | 现有训练集 | KernelBench | 最终有效 canary | 训练集 + canary |
| --- | ---: | ---: | ---: | ---: |
| convolution | 14.30% | 45.20% | 72.29% | 15.70% |
| normalization | 5.58% | 19.60% | 68.78% | 7.10% |
| matmul/linear | 17.49% | 32.40% | 0 | 17.07% |
| indexing/scatter | 8.67% | 0.80% | 0 | 8.46% |
| loss/distance | 14.93% | 1.20% | 0 | 14.57% |

canary 自身覆盖了目标结构，且刻意提高 convolution 和 normalization 的密度。合入四万级训练池后变化仍小，说明 1,000 个样本足以验证方法和支持度，不足以完成 distribution matching。扩量时还需要单独增加 matmul/linear 与 sequence-rank cell，不能继续复制现有 conv/norm 模板

### 手工抽检

抽检覆盖全部 33 个模板，并在四个 family 和五种输入构造之间分层；stateful、非连续输入、重复 batch 和多 class 样本均进入样本池

| 抽样维度 | 规则 | 实际覆盖 |
| --- | --- | ---: |
| 模板 | 每个模板至少 1 个 | 33/33 |
| family | `atomic` 21，其他三个 family 各 8 | 45 个样本 |
| 输入构造 | 五种构造各至少 5 个 | 5/5 |
| 结构与状态 | 覆盖长模块、多 class、stateful 和非连续输入 | 全部命中 |

| 评级 | 样本数 | 含义 |
| --- | ---: | --- |
| A | 38 | 未发现语义问题或明显分布边界 |
| B | 7 | 样本有效，但存在需要记录的输入分布边界 |
| C | 0 | 未发现 child 引入的语义错误 |

六个 B 级样本来自 `broadcast_repeat`，其 batch 内数值完全重复；另一个 B 级 div 样本将分母限制在正数且远离零。两类都能有效测试对应算子，但不覆盖独立 batch、负分母或近零分母。该集合没有 parent，因此 parent 问题不适用；抽检没有证据不足项

## 已知偏差和遗留问题

### Shape 仍小

本批只做安全 canary，rank 3/4/5 的样本数为 115/611/270。profiler 能静态解析 564/1,036 个 factory 调用，其元素数 p50 和 p90 为 16,384 和 110,592，没有已解析样本达到 1M；KernelBench 对应为 33,554,432、1,610,612,736 和 88.6% 达到 1M。该统计是可解析调用的下界。本方法不承担 shape 扩增，不能用它替代已有 shape lane

### 值域和模板仍集中

`get_inputs()` 全部使用 `torch.rand`，修正了训练集过度依赖 `randn` 的方向，但值域仍集中在 `[0, 1)`。`broadcast_repeat` 还引入了 180 个 batch 内相关样本。去常量 skeleton 的最大重复组已经限制为 20，但并入训练集后，原有最大重复组仍为 5,038，输入模板集中问题没有消失

### Flatten 复合路径

四条失败样本都使用下面的核心组合

```python
def forward(self, x):
    return torch.flatten(x, 1)

def get_inputs():
    x = torch.rand(3, 10, 10, 8).movedim(-1, 1)
    return [x]
```

该输入非连续，H20 上的 flatten 走 `aten::clone -> aten::_unsafe_view`。reference 五次执行均通过，但现有 liveness 只把 `aten::view/reshape` 认作 flatten identity，无法证明目标标签到达返回值。当前结果保守拒绝这四条。扩量前可让 AT17 避开 `channel_last_view`，或为复合 materialize/view 路径增加经过审查的 provenance 规则，不能直接放宽 shared runtime core

### Stateful write 尚未覆盖

`self.hidden.copy_` 会修改 registered buffer，现有 shared liveness 对 tainted in-place write 采用 fail-closed 策略。要补这类样本，需要单独核对 control/trace 最终 state、跨 trial 持久化、alias 和返回值依赖的 stateful-write validator

### 尚无训练收益结论

最终数据只证明 reference 可运行、声明算子可达返回值和结构合同成立。尚未做采样权重设计、训练合并或 ablation，不能据此判断 KernelBench correctness 的提升幅度

## 产物和复现入口

### 代码

- 生成器：`tools/data/synthesize/kernelbench_gap_method/generate_kernelbench_gap.py`
- liveness adapter：`tools/data/synthesize/kernelbench_gap_method/validate_kernelbench_gap_liveness.py`
- 8 卡 launcher：`tools/data/synthesize/kernelbench_gap_method/launch_kernelbench_gap_liveness_shards.sh`
- fail-closed finalizer：`tools/data/synthesize/kernelbench_gap_method/analyze_kernelbench_gap_run.py`

### 最终产物

- 有效样本：`local_artifacts/data/synthesize/kernelbench_gap_canary1000/run.primary.v2/analysis/final/accepted.parquet`，SHA-256 `8c9d5c39a5be9ea4a89031cc7099fb150fc2a9ed0a3503a1ab7c50bbeceb7903`
- 逐样本合同：`local_artifacts/data/synthesize/kernelbench_gap_canary1000/run.primary.v2/analysis/final/accepted.manifest.jsonl`，SHA-256 `e1bfb037e543fc74ec56b15e387e4bb7f2b8830dd6923ec9b0c4f1c33b93c71e`
- 最终汇总：`local_artifacts/data/synthesize/kernelbench_gap_canary1000/run.primary.v2/analysis/final/final_summary.json`，SHA-256 `23da9c8e62d3a97a5ce7a0563f5047fb6e623037df1f7498566069c49eb6b451`
- 失败审计：`local_artifacts/data/synthesize/kernelbench_gap_canary1000/run.primary.v2/analysis/final/failure_audit.jsonl`
- 同口径分布：`local_artifacts/data/synthesize/kernelbench_gap_canary1000/run.primary.v2/analysis/final/accepted_distribution_profile.json`
- 手工抽检：`local_artifacts/kernelbench_gap_canary_review_20260811/manual_semantic_audit.md`

### 证据绑定

- candidate SHA-256：`c9e631c0bd2c8c78b9c95d69e9e7b8b5d27ca92baaa269ef78492668b36f352d`
- generator commit：`4fe1c02c700ce110aeec40c5aa3afaf55d9ef341`
- generator SHA-256：`6d2cae2d3d78f72b55093fc8fe8796e00e3605e397cade152a3129249df22ca4`
- reference runtime：H20，PyTorch 2.11.0+cu129，CUDA 12.9，KernelGym commit `26255057463a77b23abac0f3e5eafeeebf2ebbb5`
- reference contract fingerprint：`0265b2a03abd4a9b984d90ec68d7049dd8664e5d815a60f213f6c0f22e5c04dd`
