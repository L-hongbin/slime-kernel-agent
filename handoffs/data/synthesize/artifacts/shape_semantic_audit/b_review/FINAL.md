# B 级样本最终裁定

这里记录 73 条语义边界样本的最终裁定。73 个 UUID 无遗漏、无重复。

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

## A 级样本

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

## C 级样本

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

## B 级边界案例

- `shapesolver_6b409e7409d6bb9f90b27273` 评级为 B。`view(-1,784)` 会重切连续元素，但 reference
  没有足够证据证明输入首轴必须维持原始记录边界，因此只能标记 axis/coupling 疑点，不能确定为 C。
- `shapeai_468fa85419a1cd4fe1c2fdb8` 评级为 B。bmm 完整闭合，但 scatter 的 index 上界仍按
  128 生成，而目标轴已扩至 768，覆盖率随扩增下降。
- `shapeai_f15bdcaa8fc8eb8b0fbc3568` 评级为 B。3-D interpolate 的 API 行为合法，但变量名所称
  height/width 与实际 `(N,C,L)` 解释不一致。
- `shapesolver_e09ae73a969d1794274e443c` 评级为 B。复数 layout/reduction 工作有效，但构造器中的
  channel/config metadata 与 forward 缺少实际 coupling。
