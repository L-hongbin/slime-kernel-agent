# Shape child 语义抽检

## 摘要

本次从 63,979 个 H20-valid shape child 中抽取 264 条互不重复的 parent→child 对照。这是“启发式极端样本 + 分层覆盖样本”，用途是找问题，不是估计全量缺陷率；抽样前并不知道哪些 child 有语义问题。

最终评级为 A=193、B=52、C=19。其中 1 条 C 是 child 新引入的确定性语义问题，18 条是 parent 本来已有、被新 shape 放大或暴露的问题。

因此，GPU reference 和 changed-region 通过不能证明 shape 有任务语义。该次候选池元数据为 `training_approved=false`；后续 release 使用授权与候选审计状态分别记录，见 [数据总览](../../README.md)。1/264 只是在启发式极端、分层覆盖样本中的已确认 child 新增问题命中率，不能外推为 63,979 条的总体比例。

## 抽样规则

### 总体和使用的信息

| 项目 | 规则或实际数量 | 对抽样的影响 |
| --- | --- | --- |
| 候选总体 | 13 个 shape lane 中全部 63,979 个 H20 runtime-valid child | 只要在 runtime-valid 集合中就有资格进入抽样 |
| 方法构成 | Static solver 23,358；DSV4F 40,621 | 这里只是总体构成，没有按该比例设置配额 |
| Parent 去重 | 不做 | 同一个 parent 的不同有效 child 可以分别进入总体和样本 |
| 目标样本数 | 264 | 用于语义问题发现和分类 |
| 用于选样 | input factory shape、input bytes、扩增倍数、changed slot/factory、lane、方法、source、variant、operator family、child UUID | 只描述数值、来源和粗粒度结构 |
| 不用于选样 | 人工评级、既有 review、GPU 之外的语义结论、child 是否“看起来合理” | 抽样前不知道哪些 child 有语义问题 |
| Factory 无法静态解析时 | 最大轴和轴比例退化为使用 `get_inputs()` 内可见的正整数常量；bytes/scale 仍使用 runtime/profile evidence | 该 fallback 可能把非 shape 常量计入极端值，只用于排序，是已知抽样噪声 |
| Operator 提取失败时 | 归入 `other_only` | 不丢弃 child，但 operator-family 分层精度降低 |

### 第一阶段：启发式极端样本

每个指标取前 `max(12, 264 // 12) = 22` 条。指标间按表格顺序去重，排序并列由 `SHA256(audit_version, child_uuid)` 决定。

| 顺序 | 排序指标 | 精确定义 | 取数 | 与前序重复 | 新增 unique child |
| ---: | --- | --- | ---: | ---: | ---: |
| 1 | Tensor 内轴比例 | 对每个 changed tensor 计算 `max(shape) / min(shape)`，child 取最大值，降序 | 22 | 0 | 22 |
| 2 | 最大单轴 | 所有 changed tensor 的最大 dimension value，降序 | 22 | 0 | 22 |
| 3 | Input 扩增倍数 | `aggregate input bytes after / before`，降序 | 22 | 3 | 19 |
| 4 | Direct-input bytes | child 所有 direct input 的 aggregate bytes，降序 | 22 | 0 | 22 |
| 5 | 修改复杂度 | changed factory 数量降序，再按 logical shape slot 数量降序 | 22 | 0 | 22 |
| **合计** | 五个排序共尝试 110 次 | 按 child UUID 去重 | **110** | **3** | **107** |

“极端”只表示上述数值大或修改结构复杂，不表示已经知道它有语义风险。超大单轴、高扩增倍数和多 factory 修改都可能是合理的 kernel 压测。对于无法静态恢复 direct factory 的 child，表中前两个指标使用上一表所述的正整数常量 fallback，并不具备同等精度。

#### “轴比例”的口径

| 口径 | 公式 | 用途 |
| --- | --- | --- |
| 本次语义抽样的轴比例 | 同一个 changed tensor 内 `最大维度 / 最小维度`；rank-1 记为 1 | 让细长、各轴差异很大的 tensor 更容易进入人工审查 |
| 生产 shape contract 的 1,000:1 门禁 | 同一个 rank≥2 tensor 内 `最大维度 / 第二大维度` | 禁止某一个维度比其余所有维度高出三个数量级；不是本次抽样指标 |

| Shape | 抽样轴比例：最大/最小 | 生产门禁比例：最大/第二大 | 说明 |
| --- | ---: | ---: | --- |
| `[4,4096,4,4883]` | `4883/4 = 1220.75` | `4883/4096 ≈ 1.19` | 对抽样而言很细长，但不存在单独一个轴支配其余轴，因此通过 1,000:1 生产门禁 |
| `[4,4,618492,4]` | `618492/4 = 154623` | `618492/4 = 154623` | 一个轴比其余轴大五个数量级，两个口径都会认为极端，并违反生产门禁 |
| `[28123,28159]` | `28159/28123 ≈ 1.0013` | `28159/28123 ≈ 1.0013` | 两个轴接近，不会因轴比例进入极端尾部 |

轴比例只衡量数字 geometry，不知道哪个轴是 batch、channel、sequence 或 spatial，也不直接表示 shape 有意义或无意义。

### 第二阶段：分层轮询补齐

| 分层字段 | 一个 child 如何进入分层 | 是否设置总体比例配额 |
| --- | --- | --- |
| Lane | 生成它的 13 个具体 run/lane 之一 | 否 |
| 方法 | `static_solver` 或 `model_shape`（DSV4F） | 否 |
| Source family | parent 的数据来源 | 否 |
| Variant | 生成记录中的 Medium/Large 等 variant | 否 |
| Logical-slot count | 修改的逻辑 shape slot 数量 | 否 |
| Changed-factory count | 被 shape 修改影响的 input factory 数量 | 否 |
| Operator family | 从完整 reference 提取的粗粒度 multi-label family | 否；一个 child 可同时进入多个 family |

各分层内部按稳定 SHA256 排序，然后依次轮询所有分层的第 1、2、3……条。一个 child 同时命中多个分层时只保留一次。分层没有按总体占比设置 quota，因此稀有分层会获得更多曝光，不能把最终方法/source 比例当成总体比例。

| 选样阶段 | 新增 unique child | 累计 |
| --- | ---: | ---: |
| 五类启发式极端指标 | 107 | 107 |
| 分层轮询补齐 | 157 | 264 |
| 全体稳定 SHA256 fallback | 0 | 264 |

### 样本最终构成

| 项目 | 实际规则或数量 |
| --- | --- |
| 最终方法构成 | DSV4F 224，Static solver 40 |
| 重复 child | 0 |

| 可以从本次抽检得出的结论 | 是否可以 |
| --- | --- |
| GPU-valid child 中是否存在语义问题 | 可以 |
| 存在哪些问题模式 | 可以 |
| 全量 63,979 条的缺陷率 | 不可以 |
| Static solver 与 DSV4F 谁的缺陷率更高 | 不可以 |
| 用 1/264 外推 child 新增问题总数 | 不可以 |

若要估计总体比例，需要另做从 63,979 条中等概率随机抽取、并向审查者隐藏 lane/method 的 blind audit。

## 问题类型与数量

### 最终评级

每个 child 给出一个互斥的主评级；A、B、C、D 四类合计等于 264，不是问题标签计数。

| 评级 | 定义 | 数量 | 占比 | 建议处理 |
| --- | --- | ---: | ---: | --- |
| A | Shape 保留了输入、参数和 forward 的关键语义关系，并具有明确的 kernel 压测价值 | 193 | 73.11% | 可继续进入后续候选流程；仍需服从其他分布和训练门禁 |
| B | Shape 可以运行，也有部分压测价值，但存在人为比例、轴解释、弱 coupling、低有效工作量或资源代表性疑点 | 52 | 19.70% | 不是无效样本；应按问题标签降权、补充门禁或定向复核 |
| C | 明显无意义、破坏任务语义或违反明确约束 | 19 | 7.20% | 最终采样前隔离或回到 parent 清洗处理 |
| D | 仅凭 reference 和 diff 证据不足，无法判断 | 0 | 0.00% | 补充运行或语义证据后再评级 |
| **合计** | 每个 child 恰好一个主评级 | **264** | **100.00%** |  |

| 19 个 C 的来源 | 数量 | 含义 |
| --- | ---: | --- |
| Child 新引入的确定性问题 | 1 | 应在最终采样前隔离 |
| Parent 原有问题被放大或暴露 | 18 | 需要回到 source/parent 清洗处理，不能全归因于 shape 生成 |

### 1 个确认由 child 引入的问题

| 类型 | 数量 | Child |
| --- | ---: | --- |
| 违反 source 明确输入域 | 1 | `shapeai_cf02860b6a846c58094ed4f7` |
| **合计** | **1** |  |

### 18 个 parent 原有问题

| 类型 | 数量 | Child |
| --- | ---: | --- |
| 构造参数只生成未参与 forward 的普通 metadata | 1 | `shapesolver_2b6f776edc8cdda92da6ee16` |
| 同一 loss 在代码中已使用两个不同类别轴 | 1 | `shapeai_61f6e74a0818d51d1325dd95` |
| Loss 权重的 rank/broadcast 已违反 docstring | 1 | `shapeai_0f7e60ff06abf914e863d57d` |
| Attention 的 batch/head/sequence 轴已错置 | 7 | `shapeai_19068c4bebdcd8dc1d1189c3`、`shapeai_20d1cd0a1ad3958c6fa71158`、`shapeai_66e77f8e48dee74c25ce3bb9`、`shapeai_6836982a6ec85a5df6ed48c4`、`shapeai_f867d1ac6aa825e37957f1cf`、`shapesolver_23c5b4f7b2b198020f5c0656`、`shapesolver_b224a5b13adf605eb6ac08f4` |
| 明确要求 3-D GRU 输入但实际为 rank-4 | 1 | `shapeai_7fcc2f731bf7bb4473b37257` |
| Reduction 后 normalization 轴已错误 | 1 | `shapeai_ebba969c5f7eb4082d5a1d9d` |
| 全等轴掩盖 expand/broadcast 错位 | 3 | `shapeai_305722af479e6b2f2260dfb2`、`shapeai_260920ce835238c5f8411684`、`shapeai_ea9221672f3f7d0cfe35d363` |
| Graph 的额外 batch 轴被错误解释或 flatten | 2 | `shapeai_7c114e2b7eec8ef88045a558`、`shapeai_7297cda9888b17391ce120dd` |
| Parent 本身没有实质算子工作 | 1 | `shapesolver_067a0c1427d6f3d82de549cf` |
| **合计** | **18** |  |

### C 级核心代码示例

下面先给出 1 个确认由 child 新引入的问题。代码只保留决定 shape 语义的部分。

#### 违反 reference 明确写出的输入范围

`shapeai_cf02860b6a846c58094ed4f7` 同步修改了 input 和 `input_size`，因此运行不会失配；问题是 reference 明确限定长度小于 10。

```python
# Generates sample inputs with the requirement of size above 1 and less than 10
def get_inputs():
    x = torch.randint(0, 2, (1137654321,), dtype=torch.uint8)
    y = torch.randint(0, 2, (1137654321,), dtype=torch.uint8)
    return [x, y]

def get_init_inputs():
    return [1137654321]
```

下面是三个“parent 原有问题被放大或暴露”的代表例子，不应算作 child 新造出的代码缺陷。

`shapeai_305722af479e6b2f2260dfb2` 的 parent 用 `[32,1]` 的 `linspace_values` 对四维输入做 `expand_as`。广播从右侧对齐，因此 32 实际对应 height 轴，而代码的 `argsort(dim=1)` 和 softmax 把 channel 轴当作目标轴。Parent 的 channel、height、width 都为 32，掩盖了这个错误；child 只把 batch 从 128 扩为 1,828，没有制造该问题。

```python
def forward(self, x):
    x = torch.argsort(x, dim=1)
    linspace_values = torch.linspace(
        self.low, self.high, steps=self.steps
    ).unsqueeze(1).expand_as(x)
    x = torch.median(torch.stack((x, linspace_values)), dim=0).values
    return self.softmax(x)

def get_inputs():
    return [torch.randn(1828, 32, 32, 32)]  # parent 为 [128,32,32,32]
```

`shapesolver_2b6f776edc8cdda92da6ee16` 的构造参数只用于计算普通整数属性 `self.out_channels`，而该属性不参与 forward；这是 parent 原有的 dead init metadata。Child 将四个输入的末维扩为 128 后，只是把这个问题显式暴露出来。

```python
class Model(torch.nn.Module):
    def __init__(self, raw_msg_dim, memory_dim, time_dim):
        self.out_channels = raw_msg_dim + 2 * memory_dim + time_dim

    def forward(self, z_src, z_dst, raw_msg, t_enc):
        return torch.cat([z_src, z_dst, raw_msg, t_enc], dim=-1)
```

```python
class Model(nn.Module):
    def __init__(self):
        self.instance_norm = nn.InstanceNorm2d(3)

    def forward(self, x):
        x = torch.sum(x, dim=1)  # [batch, 3, H, W] -> [batch, H, W]
        return self.instance_norm(x)  # 3-D 输入按 [C, H, W] 解释，把 batch 当 C
```

### 52 个 B 的问题标签

B 表示“仍可作为某种压测，但代表性或语义可疑”，不是确认无效。下表数量是 52 个 B 中带有该标签的样本数；标签允许重叠，所以数量之和为 127，不能相加当作 B 的总数。

| 问题标签 | 数量 | 含义 | 示例 child | 示例 |
| --- | ---: | --- | --- | --- |
| `effective_work` | 41 | tensor 变大，但核心算子的有效工作覆盖、输出保留或任务价值增长不足 | `shapeai_7c4c0733e226ddcba8bc2c40` | 巨大 Linear 输出最终只保留 1024 个对角元素 |
| `runtime_gate_gap` | 35 | direct-input bytes 不能表示参数、中间激活或输出峰值 | `shapeai_05fd98af19f7a556add5ebf3` | direct input 约 1.88 GiB，后续 Conv 输出估算约 7.5 GiB；该 child 已在 H20 实测通过 |
| `axis_semantics` | 16 | 代码对 rank/axis 的解释与变量名或任务描述不同 | `shapeai_f15bdcaa8fc8eb8b0fbc3568` | 3-D interpolate 实际按 `(N,C,L)` 工作，不是描述中的二维 height/width |
| `coupling` | 14 | shape 可广播或运行，但输入、权重、index、metadata 之间的任务关系较弱 | `shapeai_468fa85419a1cd4fe1c2fdb8` | scatter index 的覆盖上界没有随目标轴增长 |
| `aspect_ratio` | 13 | 某些轴相差很大，使特定算子退化或分布不典型 | `shapesolver_dc5d17787874d0ac5fab2aff` | Conv1d 输出的 `(C=128,L=8192)` 上做 `triu`，只清零约 0.8%，接近恒等 |
| `repetition` | 4 | 工作量增长主要来自重复访问或重复相同操作 | `shapeai_c20e4e7410492107ae8e4288` | 大量 edge 重复访问同一小组节点 |
| `other` | 4 | 不能归入上面标签的代数恒等或任务域疑点 | `shapeai_89d2a1c3e413796245bff182` | uint8 XOR 后再 `clamp(0,255)` 恒等 |

`shapesolver_6b409e7409d6bb9f90b27273` 的 `view(-1, 784)` 保持了代码要求的整除关系，但 child 把输入从 `[4,784]` 改成 `[4096,241276]`，使一个输入行被重新切成许多 784-wide 行。代码没有提供证据证明输入首轴必须继续代表原始样本边界，因此该项记为 `axis_semantics + coupling` 的 B，而不是确定破坏语义的 C。

这些标签来自 264 条样本的逐条评级；其中 `runtime_gate_gap` 只表示静态 4 GiB input 门禁的解释边界。相关 child 已完成真实 H20 验证，不能据此反推它们实际 OOM 或运行失败。

## 建议门禁

1. 在最终采样前隔离上表 1 个确认 child；18 个 source-defect 样本按 parent 清洗策略另行处理。
2. 对 `view(-1, constant)`、多处使用不同 softmax/class dim、broadcast/expand、cat metadata 和明确输入域约束增加静态语义检查。不要把“总元素数可整除”当成维度语义成立。
3. 对原始 shape 多个轴相等的 parent，增加 near-neighbor axis probe：把候选轴单独 `+1` 后做 FakeTensor/小规模真实运行，用来暴露只因轴值碰巧相等而成立的 broadcast/view/coupling；它仍不能替代人工语义检查。
4. 将 direct-input bytes 与参数量、关键中间 shape 的估计分开记录。H20 runtime 继续作为最终可运行门禁，但不要把它解释为 workload 合理性证明。
5. 对 scatter/index、reduction、diag、Identity 等增加“核心算子有效工作量随扩增增长”的启发式指标，作为采样降权而非一刀切拒绝。

## 证据入口

- 抽样清单：`local_artifacts/data/synthesize/shape_semantic_audit/manifest.tsv`
- 可复现抽样代码：`tools/data/synthesize/model_shape/sample_shape_semantic_audit.py`
- 边界样本的逐条裁定见下节，属于上述 264 条样本的子集

## 73 条边界样本复核

这 73 条是 264 条审计样本中初始判为 B 的子集，复核结果已纳入全文 A=193、B=52、C=19 的总数，不能与总数相加。73 个 UUID 无遗漏、无重复

统一裁定口径如下：大尺寸、奇数、非 2 的幂、矩形 shape、输出为标量，以及真实执行的大型
reduction/GEMM/attention 本身不构成 B；只有明确的语义、coupling、有效工作覆盖或 workload
代表性疑点才保留 B。C 要求存在可以从 reference 直接证明的任务语义错误或明确约束冲突。

| 最终评级 | 数量 | 含义 |
| --- | ---: | --- |
| A | 12 | 没有 material shape 问题 |
| B | 52 | 存在语义、coupling、有效工作覆盖或 workload 代表性疑点 |
| C | 9 | 存在可由 reference 直接证明的 parent/source 语义问题 |
| D | 0 | 无 |
| **合计** | **73** |  |

### A 级样本

| Child | 裁定依据 |
| --- | --- |
| `shapeai_24dab59d95ec8cfea127d449` | `forward(t,x)` 中 x 末轴 319 匹配 Linear，输出末轴 311 与 t 匹配；逐元素门控合法 |
| `shapeai_d2bab2c569bc05240b821d6d` | `triu` 对 1000×36000 矩形末两轴定义明确，未违反代码约束 |
| `shapeai_d3d3e4d4af45c2df757264dc` | `sinc`、`ones_like` 和 `dist` 消费完整向量；标量归约不是有效工作不足 |
| `shapeai_d6ea517880f653c4de989977` | `dist(x, scalar)` 对完整向量执行合法全局归约 |
| `shapeai_dec0f41f70c3b0d65dadcd87` | 两个前导维可合法作为嵌套 batch，GCN 两次 matmul 的节点和特征轴闭合 |
| `shapeai_defc04f497172e74d443833d` | LSTM 的 sequence、batch、input 和 hidden state 全部闭合；规模大不是语义问题 |
| `shapeai_e06d3e3d62ef0796fdd9e62b` | node feature、邻接矩阵和 projection 构成合法 batched dense GCN |
| `shapeai_f2738e7771279e1dd4fa2790` | query/key 不同长度是合法 cross-attention，channel/head 约束闭合 |
| `shapeai_f33d2762bdfb580fb04ada02` | Linear、target 和逐样本 addcmul 的 batch 轴同步 |
| `shapesolver_23bfc1711afeb648c5e79a86` | softmax 类别轴和 MSE 两端完全同步；矩形空间轴没有额外约束 |
| `shapesolver_6a12b6d2146722f4fa1d9df4` | reduction 明确沿长度 4 的轴执行，其余大轴均参与 norm 计算 |
| `shapesolver_e546b9197ce5838415e1545d` | index 值域、Embedding 和 LogSoftmax 完整闭合；小词表上的大 batch/sequence 仍是有效工作 |

### C 级样本

下列问题都已存在于 parent；child 只是保留、放大或暴露它们，不能计为 shape 扩增新造的问题。

| Child | Parent/source 问题 |
| --- | --- |
| `shapeai_0f7e60ff06abf914e863d57d` | docstring 要求 `(B,anchors)` 权重，实际 parent 用 rank-4 weights 乘 rank-3 loss，靠全 4 轴广播运行 |
| `shapeai_19068c4bebdcd8dc1d1189c3` | SDPA 输入未转为 `[B,heads,sequence,head_dim]`，实际把 heads=8 当 attention length |
| `shapeai_20d1cd0a1ad3958c6fa71158` | `[B,S,E]` 直接送入 SDPA，batch 被当 leading/head 维，`num_heads` 完全未参与 reshape |
| `shapeai_66e77f8e48dee74c25ce3bb9` | 输入按 batch-first 构造，MHA 却保持默认 sequence-first，后续又按 `bth` 解释输出 |
| `shapeai_6836982a6ec85a5df6ed48c4` | 默认 sequence-first MHA 的输出直接交给 batch-first LSTM，同一前两轴被重新解释 |
| `shapeai_f867d1ac6aa825e37957f1cf` | 与上一条同一 parent；MHA 和 LSTM 对 batch/sequence 的解释冲突 |
| `shapeai_7fcc2f731bf7bb4473b37257` | docstring 明确要求 3-D sequence×batch×feature，parent 实际提供 rank-4 输入 |
| `shapesolver_23c5b4f7b2b198020f5c0656` | 名义 `[batch,sequence,E]` 直接送入 sequence-first MHA，batch 和 sequence 对调 |
| `shapesolver_b224a5b13adf605eb6ac08f4` | `[B,S,E]` 被 permute 为 `[B,E,S]` 后送入 SDPA，实际 attention length 是 E，`num_heads` 未使用 |

### B 级边界案例

- `shapesolver_6b409e7409d6bb9f90b27273` 评级为 B。`view(-1,784)` 会重切连续元素，但 reference
  没有足够证据证明输入首轴必须维持原始记录边界，因此只能标记 axis/coupling 疑点，不能确定为 C。
- `shapeai_468fa85419a1cd4fe1c2fdb8` 评级为 B。bmm 完整闭合，但 scatter 的 index 上界仍按
  128 生成，而目标轴已扩至 768，覆盖率随扩增下降。
- `shapeai_f15bdcaa8fc8eb8b0fbc3568` 评级为 B。3-D interpolate 的 API 行为合法，但变量名所称
  height/width 与实际 `(N,C,L)` 解释不一致。
- `shapesolver_e09ae73a969d1794274e443c` 评级为 B。复数 layout/reduction 工作有效，但构造器中的
  channel/config metadata 与 forward 缺少实际 coupling。
