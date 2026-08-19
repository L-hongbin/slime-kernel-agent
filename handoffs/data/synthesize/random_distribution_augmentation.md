# 随机数生成分布扩增与验证

# 摘要

原训练集的输入几乎都由 `randn` 生成，而 KernelBench 全部使用 `rand`。本方法在 shape 扩增结果上只改变输入随机数生成分布，不改变 shape、dtype、layout、`Model.forward` 和 `get_init_inputs`

确定性 solver 处理约 32k 个 shape child，经静态检查、H20 paired reference、三次 liveness 和 use-site 语义过滤后保留约 23k 个 child。最终池的 100 条风险分层样本评级为 A81/B19、C/D 0，其中 17 条 B 来自 child 的值域或有效作用疑点，2 条来自 parent 原有问题。所有产物保持 `training_approved=false`

# 背景

原训练集 95.6% 的样本含 `randn` 输入，KernelBench 250 道题全部含 `rand` 输入。各方法可以在同一道题中同时出现，因此训练集各行不能相加为总样本数

| 随机数生成方法 | 原训练集 | KernelBench |
| --- | ---: | ---: |
| `randn` | 38,515（95.6%） | 0 |
| `rand` | 1,010（2.5%） | 250（100%） |
| `randint` | 3,817（9.5%） | 3（1.2%） |

# 方法

## 随机数生成分布

每个 parent 通过稳定 hash 分配一种分布。同一个 `get_inputs()` 中符合条件的 `torch.randn` factory 会全部替换，但各 factory 使用独立的稳定 seed

| 随机数生成分布 | 构造方式 | 目标覆盖 | 构造理由 |
| --- | --- | --- | --- |
| `uniform_01` | 用 Normal CDF 把原 `randn` 映射到 `[0,1)` | 非负连续值 | 概率积分变换使标准正态变量 `z` 的 `Φ(z)` 服从 Uniform(0,1)；确定性映射保留原 `randn` 调用和全局 RNG 消耗 |
| `signed_uniform` | 用 `erf` 映射到 `[-1,1)` | 有界正负连续值 | `erf(z/√2)=2Φ(z)-1`，因此结果关于 0 对称并保留正负覆盖，同时不增加全局 RNG 消耗 |
| `poisson_counts` | 从原 draw 计算有界 rate，再用局部 generator 采样 | 非负计数值 | `softplus(draw)` 保证 rate 为正，截到 `[0.125,8]` 避免几乎全零或尾部过大；局部 generator 隔离新增采样的 RNG 消耗 |
| `multinomial_categories` | 沿末维构造概率并采样 category ID | 离散类别值 | 末维 `softmax` 把 draw 变成合法类别概率；按末维长度有放回采样并恢复原 shape，局部 generator 隔离新增 RNG 消耗 |

Poisson 和 multinomial 仍用原浮点 dtype 保存离散值。Multinomial 只接受末维静态且不小于 2、category ID 能由原 dtype 精确表示的输入

## Static solver

Solver 解析完整 reference 的 AST，并按以下顺序处理

1. 定位同步 `get_inputs()`，解析 direct-input factory 与顶层返回值的对应关系
2. 只接受 factory 关系唯一、shape 和实数浮点 dtype 可静态解析的 parent
3. 记录 `get_inputs()` 中全部 `torch.randn` 的源码区间，并用目标分布 wrapper 替换
4. 重新解析 child，核对替换数量、`Model`、`get_init_inputs`、factory signature 和完整 AST

Factory 返回前只能经过简单变量 alias 和 list、tuple、dict 容器。乘加、切片、`.to()`、transpose 或其他函数包装会使 factory 关系不再唯一，solver 直接拒绝

`uniform_01` 和 `signed_uniform` 只对原 draw 做确定性映射。Poisson 和 multinomial 需要新增采样，因此使用与 factory 绑定的局部 generator。以下代码展示 Poisson 的等价结构

```python
# parent
x = torch.randn(shape)

# child
draw = torch.randn(shape)  # 与 parent 一样推进默认 generator
local_generator = torch.Generator(device=draw.device).manual_seed(factory_seed)
rate = torch.clamp(torch.nn.functional.softplus(draw), min=0.125, max=8.0)
x = torch.poisson(rate, generator=local_generator)
```

`factory_seed` 由 parent 稳定 hash 和 factory 顺序生成。新增采样只推进局部 generator，因此 `get_inputs()` 结束后的默认 RNG 状态与 parent 一致

## 静态检查

每个 child 必须同时满足以下合同

- 输入数量、shape、dtype 和 layout 不变
- 原全局 `randn` 调用与 RNG 消耗不变
- 只有 direct-input factory 的随机数生成分布改变
- Poisson rate、multinomial cardinality 和浮点精确表示范围满足约束
- Parent 与 child 的非 factory AST 保持一致

## GPU 检查

Paired reference 为 parent 和 child 使用相同模型初始化与 RNG 状态，各运行五次 trial。Parent 用于确认原题在同一环境中成立，并区分 intervention 问题与 parent 原有的 OOM、timeout、unsupported 或 evaluator failure

Reference 双通过后，child 再运行三次 liveness，检查 device、stride、storage offset、requires-grad、`get_inputs()` 后的 RNG 状态、目标分布 support、未修改输入 control 和输出效应。Poisson 会从 parent 重算 rate 并核对 rate/count histogram；multinomial 按实际 runtime dtype 检查 category ID 的精确表示范围

## Use-site 语义过滤

GPU 检查能证明程序在有限 seed 下可运行且输入变化影响输出，不能证明新分布适合实际 use-site。静态语义过滤从已修改输入追踪完整 `forward`

- `multinomial_categories` 只有在 category ID 可证明只用于结构或离散顺序操作时保留
- `uniform_01` 和 `poisson_counts` 遇到可证明依赖输入正负号的首个数值算子时拒绝
- 无法静态证伪的 normalization、sequence domain、跨输入 coupling 和 parent 轴语义记录为 warning

Category ID 可以用于离散排序

```python
def forward(self, x):
    return torch.argsort(x, dim=-1)
```

Category ID 进入矩阵乘时，编号会被当作具有距离和比例关系的连续数值，因此拒绝

```python
def forward(self, x, weight):
    return torch.matmul(x, weight)
```

对于非负分布，`abs` 会退化成恒等操作，LeakyReLU 的负半轴也不会执行，因此这两种首算子均拒绝

```python
y1 = torch.abs(x)
y2 = torch.nn.functional.leaky_relu(x, negative_slope=0.1)
```

如果输入先经过 Linear 或 convolution，输出可以重新产生负数，后面的 ReLU 不会仅因输入分布非负而被拒绝

# 结果与分析

验证漏斗的主要断层来自 use-site 语义过滤。四种分布在 GPU 阶段都接近全部通过；`multinomial_categories` 进入完整 `forward` 检查后只保留 36 条，说明仅凭 shape、dtype 和 runtime pass 会明显高估 category 输入的有效覆盖

![全量漏斗与各随机数生成分布最终保留率](../../../local_artifacts/data/synthesize/random_distribution_augmentation/random_distribution_acceptance.png)

## 手动抽检

最终 22,908 条 accepted child 中固定抽取 100 条风险分层样本。抽样先覆盖高风险静态标签，再轮询分布、source、operator count、factory count 和输入规模。该抽样用于发现问题，评级比例不能外推为全量缺陷率

A 表示分布与完整 reference 闭合且有明确覆盖价值；B 表示程序可运行，但 domain、coupling、有效作用或 parent 语义存在疑点；C 需要代码直接证明 child 违反明确约束或程序结构；D 表示证据不足

| 随机数生成分布 | 抽样数 | A | B | C | D |
| --- | ---: | ---: | ---: | ---: | ---: |
| `multinomial_categories` | 4 | 4 | 0 | 0 | 0 |
| `poisson_counts` | 26 | 20 | 6 | 0 | 0 |
| `signed_uniform` | 30 | 29 | 1 | 0 | 0 |
| `uniform_01` | 40 | 28 | 12 | 0 | 0 |
| **合计** | **100** | **81** | **19** | **0** | **0** |

| B 主类型 | 数量 | 典型表现 |
| --- | ---: | --- |
| Attention、LSTM 或 logits 的连续值域变窄 | 11 | 非负或计数输入直接进入 Q/K/V、循环初态或 logits |
| 分支、派生标签或算子有效范围坍缩 | 6 | 阈值分支不可达、clamp/ReLU6 饱和、派生标签恒定 |
| Parent 原有问题 | 2 | 负 CrossEntropy weight、`forward` 内覆盖 `Parameter.data` |

19 条 B 中，17 条来自 child 的代表性或有效作用疑点，2 条来自 parent。没有发现 C/D，但全量仍未获得语义证明

## 使用边界

最终 accepted child 已通过构造合同、paired reference、三次 liveness 和当前已知的 use-site 拒绝规则。这些检查不保证任意 seed 下都有效，也不能证明新分布与任务语义等价。所有 row 只能作为人工 review 和受控训练 ablation 的候选

# 遗留问题

## 同一 parent 统一使用一种分布

同一个 `get_inputs()` 中的多个 `torch.randn` 会统一使用 parent 被分配到的分布。例如，图像输入、权重输入和连续 target 可能一起变成 Poisson。各 factory 的局部 seed 不同，但分布相同

如果 `get_inputs()` 同时包含 `torch.randn` 和 `torch.rand`，solver 会拒绝整个 parent，不做部分替换。`torch.randint` 可以与 `torch.randn` 共存并保持不变

后续可以沿 factory 到 consumer 的数据流分别选择分布，或每个 child 只修改一个 factory。两种方案都需要重新执行 paired reference、liveness 和语义抽检

## Multinomial 与矩阵运算

当前门禁拒绝 category ID 进入 matmul 等连续矩阵运算。重新排列 category 编号不会改变类别身份，却会改变矩阵乘结果；solver 无法证明编号的大小、距离和比例具有数值语义。ID 上界还会随末维扩大，矩阵累加可能进一步放大数值

这条规则会过滤 ordinal category、小整数矩阵等可能有价值的覆盖。后续应单独定义 `small_integer_matrix` 或有界计数分布，明确数值范围和 use-site，再进行 GPU 与语义验证

## 原生 `torch.rand` 尚未覆盖

`uniform_01` 保留原 `torch.randn`，再用 Normal CDF 做确定性映射。该方案维持默认 RNG 状态，但没有覆盖原生 `torch.rand` 的实现路径和有限精度端点行为

如果需要覆盖原生 `torch.rand`，可以先消耗并丢弃原 `randn` draw，再用 factory 局部 generator 调用 `torch.rand`。该方案会增加一次随机采样，需要重新验证 RNG、dtype、device、layout、性能和输出有效性

## 静态门禁无法穷尽派生语义

最终池复核仍发现 17 条 child 级 B，包括非负输入经 normalize 后使 cumsum 只走正区间、Poisson 矩阵乘后在 clamp 上界聚集，以及由输入均值派生的二元 target 恒定

简单按首个 consumer 拒绝会误杀正常的卷积、矩阵乘和计数任务。后续应优先增加可可靠证明的阈值可达性、派生标签退化和 logit-role 规则，再重新筛选全量

# Log 和 artifact

## 全量结果

| Gate | Passed | Denominator | Rate |
| --- | ---: | ---: | ---: |
| Paired reference：parent/child both-pass | 31,428 | 31,648 | 99.3049% |
| 三次 liveness raw pass | 31,039 | 31,428 | 98.7623% |
| 语义过滤晋级 | 22,908 | 31,039 | 73.8039% |
| 最终 accepted | 22,908 | 31,648 | 72.3837% |

| 随机数生成分布 | Candidate | Liveness raw pass | Semantic promoted | Final rate |
| --- | ---: | ---: | ---: | ---: |
| `multinomial_categories` | 7,800 | 7,582 | 36 | 0.4615% |
| `poisson_counts` | 7,961 | 7,895 | 7,606 | 95.5408% |
| `signed_uniform` | 7,898 | 7,659 | 7,659 | 96.9739% |
| `uniform_01` | 7,989 | 7,903 | 7,607 | 95.2184% |

Paired reference 的 220 个非 both-pass pair 包括 214 个 parent-pass/child-fail、4 个 both-fail 和 2 个 parent-fail/child-pass。按保守归因，202 个属于可能由 intervention 引起的数值或正确性问题，6 个 OOM 或 memory guard，4 个 timeout，2 个 evaluator/environment 瞬态，其余 6 个来自 parent baseline 或双方 baseline/evaluator

Liveness 拒绝 389 条，其中 273 条输出没有稳定变化，95 条触发 64 GiB 门禁，21 条 unsupported。语义过滤再拒绝 8,131 条，其中 `multinomial_categories` 7,546 条、`poisson_counts` 289 条、`uniform_01` 296 条

GPU raw-pass 的 source 分布为 cuda_agent 1,728、drkernel 28,254、oubo_generated 1,057；operator bucket 为不超过 1 op 696、2–5 ops 27,139、不少于 6 ops 3,204

## 抽检记录

Use-site 语义过滤前，从 31,039 条 H20 liveness raw-pass child 中抽取的 100 条问题发现型样本为 A55/B45。该轮暴露的 category-as-value 和 sign-sensitive 模式进入当前语义门禁

语义过滤后的最终池复核为 A81/B19、C/D 0。最终 36 条 multinomial child 的调用签名已全量核对，只包含结构或离散顺序操作。Accepted parquet/manifest 的首、中、尾 row 也已人工核对，22,908 个 child UUID 唯一且顺序一致

## 运行记录

全量 paired reference 和 liveness 使用 node64、node53、node69、node70 的 4×8 张 NVIDIA H20。每张 GPU 使用 8 个 virtual shard，共 256 个 shard

Exact analyzer 校验 scheduler contract、256 份 JSONL、launcher summary、modulo assignment、源码 SHA、输入 SHA 和 runtime partition。GPU raw evidence 的独立复核结果为 PASS，无 blocker

## 路径和入口

| 用途 | 路径 |
| --- | --- |
| 权威运行目录 | `Data/prompt_tvm_v4/random_value_from_shape_v5/run.low128k_final` |
| 串行扩增 fallback base | `Data/prompt_tvm_v4/serial_augmentation_v1/random_fallback_base.v3.semantic_gate` |
| Solver | `tools/data/synthesize/random_method/solve_value_coverage.py` |
| Liveness validator | `tools/data/synthesize/random_method/validate_value_liveness.py` |
| Liveness launcher | `tools/data/synthesize/random_method/launch_value_liveness_shards.sh` |
| Exact analyzer | `tools/data/synthesize/random_method/analyze_value_run.py` |
| 语义门禁 | `tools/data/synthesize/intervention_semantic_gates.py` |
| 抽检工具 | `tools/data/synthesize/sample_intervention_semantic_audit.py` |
| 绘图脚本 | `tools/data/synthesize/random_method/plot_random_distribution_results.py` |
| 结果图 | `local_artifacts/data/synthesize/random_distribution_augmentation/random_distribution_acceptance.png` |
| 原始池人工抽检 | `local_artifacts/data/synthesize/intervention_semantic_audit/` |
| 最终池审查包 | `local_artifacts/data/synthesize/intervention_semantic_audit_v2/assignments/random_*.md` |
| 最终池逐条评级 | `local_artifacts/data/synthesize/intervention_semantic_audit_v2/random/final_review.md` |
| GPU evidence 复核 | `Data/prompt_tvm_v4/random_value_from_shape_v5/run.low128k_final/review/kimip_final_runtime_verdict.md` |

串行扩增的 base 包含 22,908 条随机数生成分布 child 和 8,740 条 shape fallback

## Artifact hash

权威运行目录下的主要 artifact 如下

| Artifact | SHA-256 |
| --- | --- |
| `runtime/accepted.parquet` | `f432b53c9335dd1de357eda057f1d16c0e2ec7dbb501ac8e34298e38f9869980` |
| `runtime/accepted.manifest.jsonl` | `c70a27f5c0e8287942d6e6b993255ec2e464d81651b102c923f4ce5a50b8f2bd` |
| `runtime/final_summary.json` | `55ffd9563c486dfad5962c615f1c3a4c67c1ac47d43334827c2e475105eeef95` |
| `analysis/semantic_gate_summary.json` | `88386cb0bbc57176f2cb9480241eb2337f2ecc94622e386759aa77b05e637935` |
| `runtime/raw_artifact_sha256.json` | `1cdbe5feb4ea8247ac82433a0030b61ccbcf4dc14c9fba259eb23e557dda7ea5` |
| `analysis/reference_both_pass_child_uuids.txt` | `de604a3227abbbb969059d1b9f4a08fd5bca20314df7fbb4cdacdba4d8dc3219` |
| `analysis/reference_summary.json` | `1086b6ccf646bfd6ad1ad2cc6f5de26e156dd0a8a80e2c71936eb94cede87de8` |
| `analysis/liveness_summary.json` | `1220bd0757a0a0cbad979ac738752864477dce0525953075a3214261d96ce5c3` |
| `analysis/failure_bias_report.md` | `1db0b26077fefb3ece21732b2d4162803eef37f229cad37f3e94eb33551e90bc` |
| `local_artifacts/.../random_distribution_acceptance.png` | `2231271d80f5f6f15de91347483a9324f4039e1914ec796e1258706fd68d72fe` |
