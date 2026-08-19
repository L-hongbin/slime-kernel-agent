# 随机数生成分布、dtype、layout 串行扩增

# 摘要

最终产物完整保留 31,648 个 shape-resample 样本，按随机数生成分布、dtype、layout 的顺序串行扩增。某一步失败时保留上一步结果，不会丢失样本。三种 intervention 可以叠加，不能相加当作数据集样本数

最终合成用 `intervention_semantic_promotion_gate_v5` 检查每个 child，GPU evidence 只在 parent UUID 和 reference 完全一致时复用。所有产物仍为 review artifact，`training_approved=false`

| Intervention | 样本数 | 占全部样本 |
| --- | ---: | ---: |
| 随机数生成分布 | 22,908 | 72.3837% |
| dtype | 7,249 | 22.9051% |
| layout | 7,682 | 24.2733% |

# 总体合成结果

最终 intervention 组合分布如下

| 随机数生成分布 | dtype | layout | 样本数 |
| --- | --- | --- | ---: |
| ❌ | ❌ | ❌ | 5,642 |
| ❌ | ❌ | ✅ | 1,830 |
| ❌ | ✅ | ❌ | 939 |
| ❌ | ✅ | ✅ | 329 |
| ✅ | ❌ | ❌ | 12,944 |
| ✅ | ❌ | ✅ | 3,983 |
| ✅ | ✅ | ❌ | 4,441 |
| ✅ | ✅ | ✅ | 1,540 |

# 随机数生成分布扩增

完整方法、验证结果和使用边界见 [随机数生成分布扩增](random_distribution_augmentation.md)

随机数生成分布扩增使用确定性 solver，把原 `randn` 映射为 `uniform_01`、`signed_uniform`、`poisson_counts` 或 `multinomial_categories`

GPU 验证通过 31,039 条。最终保留 22,908 条：

| Family | Candidate | GPU 验证通过 | 最终保留 |
| --- | ---: | ---: | ---: |
| `uniform_01` | 7,989 | 7,903 | 7,607 |
| `signed_uniform` | 7,898 | 7,659 | 7,659 |
| `poisson_counts` | 7,961 | 7,895 | 7,606 |
| `multinomial_categories` | 7,800 | 7,582 | 36 |

# dtype 扩增

dtype 扩增使用确定性 AST solver，将直接输入的 FP32 floating factory 改为 FP16 或 BF16。Solver 要求同一个样本中的浮点输入都能静态解析且原本均为 FP32，然后把所有对应 factory 改成同一个 dtype

Solver 根据模型状态分成两类：

| 类型 | 适用范围 | 修改内容 |
| --- | --- | --- |
| `parameter_free` | 模型没有 registered parameter、buffer 或其他 tensor state | 只修改直接输入 factory 的 dtype |
| `module_state` | 模型状态全部来自可静态识别的内置 `torch.nn` module、parameter 或 buffer | 修改输入 dtype，并在 `Model.__init__` 末尾将 registered floating state 转为同一 dtype；非浮点 state 保持不变 |

`module_state` 先沿用 parent 的 FP32 初始化，再统一转换 registered state，而不在每个 constructor 中指定 dtype，原因如下

1. 低精度 constructor 的随机初始化不保证等于 FP32 初始化后再 cast，还可能改变 RNG 消耗；
2. 不同 module、`nn.Parameter` 和 `register_buffer` 也没有统一的改写形式，逐项修改容易遗漏 nested module 或破坏 tied parameter 和 storage alias

`module_state` 会拒绝 custom module base、`to/_apply` 等自定义转换逻辑、动态 constructor、显式 dtype cast、未注册 tensor state 和无法静态归因的常量

## GPU 验证

GPU 验证包含两条执行路径：

1. 把 parent 的 FP32 输入显式转换到目标 dtype，再运行 child model，与 FP32 parent 输出比较；FP16 使用 `rtol=atol=1e-2`，BF16 使用 `rtol=atol=2e-2`；
2. 直接运行 child 的低精度 factory，确认输入确实采用目标 dtype、目标 dtype 被算子消费，并且没有 FP32 或 complex fallback

之所以分成两条路径，是因为 CUDA 上直接用低精度 `torch.rand` 生成的值不保证逐元素等于 FP32 draw 再 cast。第一条路径验证低精度计算与 parent 的数值关系，第二条路径验证真实 factory 和 dispatch，不能用其中一条代替另一条

Paired reference 先确认 parent 和 child 在相同初始化及 RNG 状态下都能通过通用 reference evaluator。它不证明目标 dtype 被实际消费，也不检查 state 精确转换或 FP32 fallback；这些由上述三轮 dtype-specific GPU 验证完成

主要失败来自 RNG 状态变化、FP32/complex fallback、输出超过低精度容差、未注册 state、非浮点输出变化和 OOM

## 语义过滤

GPU 验证通过后，语义过滤继续拒绝低精度会破坏离散或边界语义的样本：BF16 不能逐整数区分 256 以上的 category ID，FP16 的对应上限是 2,048；`argmin/argmax` 可能因低精度制造并列而改变索引输出；`acos/asin` 的开区间 margin、clamp 上界也必须能由目标 dtype 表示

## 数据漏斗

| 阶段 | 通过样本数 | 占静态候选 |
| --- | ---: | ---: |
| 静态候选 | 17,500 | 100.0000% |
| Paired reference 双通过 | 16,799 | 95.9943% |
| GPU 验证通过 | 8,112 | 46.3543% |
| 语义过滤通过 | 7,608 | 43.4743% |
| 最终纳入 | 7,249 | 41.4229% |

注意：16,799 条 Paired reference 双通过样本中有 8,687 条未通过 GPU 验证：

| GPU 验证拒绝原因 | 样本数 | 占 GPU 验证拒绝样本 |
| --- | ---: | ---: |
| `get_inputs()` 后 RNG 状态变化 | 3,747 | 43.13% |
| 出现 FP32 或 复数 fallback | 2,066 | 23.78% |
| 低精度输出超过容差 | 1,683 | 19.37% |
| 未注册 state 或非浮点输出变化 | 850 | 9.78% |
| OOM | 279 | 3.21% |
| 非有限值、control 不稳定等其他原因 | 62 | 0.71% |

语义过滤通过后未最终纳入的 359 条绑定了不同的 parent reference，旧 GPU evidence 不能复用于最终串行位置，因此这些位置保留上一步样本

## 最终分布

最终 dtype 分布如下：

| 维度 | 类别 | 样本数 |
| --- | --- | ---: |
| 目标 dtype | FP16 | 3,663 |
| 目标 dtype | BF16 | 3,586 |
| 模型状态 | `parameter_free` | 2,229 |
| 模型状态 | `module_state` | 5,020 |
| source | DrKernel | 6,407 |
| source | CUDA-Agent | 644 |
| source | oubo-generated | 198 |

## 手工检查

问题发现型抽检从 GPU 验证通过的样本中按 dtype、模型状态、source、operator 数量和静态风险标签抽取 100 条，汇总裁决为 A76/B24，没有 C/D。该抽样刻意提高高风险结构的比例，不能用来估计全量问题率

| B 类问题 | 样本数 | 处理方式 |
| --- | ---: | --- |
| 上游 category 值超出低精度精确整数范围 | 14 | 超出 BF16/FP16 上限时拒绝 |
| 数值范围、长归约或端点保护风险 | 4 | 可静态证明的 clamp、inverse-trig 边界直接拒绝，长乘积保留 warning |
| Parent 原有问题 | 6 | 标记问题来源，不归因于 dtype child |

## 遗留问题

- 同一个样本中的所有 floating input factory 当前统一改成同一个目标 dtype。这保证了 intervention 单一且便于验证，但没有覆盖不同输入使用不同精度的 mixed-dtype 场景，例如 activation 使用 FP16、累计量或数值敏感输入使用 BF16。后续若扩展 mixed dtype，需要根据输入之间的算子关系分配 dtype，并重新验证类型提升、显式 cast 和多输入算子的 dtype compatibility
- `get_inputs()` 后 RNG 状态发生变化的 child 当前全部拒绝。CUDA 随机 factory 在改变输出 dtype 后可能采用不同的随机数生成路径，因此这条规则排除了部分原生低精度随机输入。后续可以把它们作为独立的 low-precision RNG lane，要求 parent/child 各自可重复、输入分布满足同一合同，并重新定义输出比较方式；不能直接复用当前只改变 dtype 且保持 RNG 状态一致的证据

# Layout 扩增

## 三种扩增方法

下面用 shape 为 `[32, 64]` 的输入说明三种 layout 变换。Child 保持 tensor 的 shape、dtype 和 logical value 不变，只改变底层存储方式

### `transpose_noncontiguous`

Parent 直接生成 contiguous tensor：

```python
x = torch.randn(32, 64)
```

Child 先交换末两维并 materialize，再交换回来：

```python
base = torch.randn(32, 64)
x = base.transpose(-1, -2).contiguous().transpose(-1, -2)
```

最终 shape 仍为 `[32, 64]`，元素顺序不变，但 stride 从 `(64, 1)` 变成 `(1, 32)`，因此 `x.is_contiguous()` 为 `False`

### `slice_storage_offset`

Parent 仍然直接生成 tensor：

```python
x = torch.randn(32, 64)
```

Child 在末维前补一个元素，再切掉补上的位置：

```python
base = torch.randn(32, 64)
storage = torch.cat((torch.zeros_like(base[:, :1]), base), dim=-1)
x = storage.narrow(-1, 1, 64)
```

`x` 的值与 base 相同，但它从底层 storage 的第二个元素开始，`storage_offset()` 为 1，stride 为 `(65, 1)`

### `expand_zero_stride`

该方法只用于 `zeros`、`ones` 或静态 `full` 等沿目标维取值相同的 factory：

```python
x = torch.zeros(32, 64)
```

Child 先保留一行，再扩展回原 shape：

```python
base = torch.zeros(32, 64)
x = base.narrow(0, 0, 1).expand(32, 64)
```

最终 shape 和值不变，但 32 行共享同一行 storage，第 0 维 stride 为 0。输入必须只读，否则多个逻辑位置会写入同一块存储

Layout solver 只改变 input tensor 的物理 layout。`Model.forward` 保持不变；若 forward 与新 layout 不兼容，直接拒绝该修改，例如 non-contiguous 输入无法满足 `.view()` 的 stride compatibility、zero-stride 输入被 in-place 写入，或程序显式读取 `stride()`、`storage_offset()`

## 验证 pipeline

### 静态检查

Solver 先验证 source、manifest 和 hash 绑定，要求 layout child 保持 logical shape、dtype、value、RNG、`Model`、`forward` 和 `get_init_inputs` 不变，并预先计算目标 stride、offset 或 zero-stride

### GPU 验证

Paired reference 在 GPU 上分别运行 parent 和 child，使用相同初始化及 RNG 状态，检查双方都能通过 reference。随后对 child 做三次 GPU 验证，核对 intervention 的实际实现、输出影响和 runtime partition。OOM、timeout、unsupported、worker error、输出关系失败或 intervention 未实际生效都会 fail closed

## 数据漏斗

| 阶段 | 通过样本数 | 占比 |
| --- | ---: | ---: |
| 静态候选 | 11,500 | 100% |
| Paired reference 双通过 | 10,481 | 91% |
| GPU 验证通过 | 8,336 | 72% |
| 最终纳入 | 7,682 | 66% |

GPU 验证通过后未最终纳入的 654 条绑定了不同的 parent reference，旧 evidence 不能复用于最终串行位置，因此这些位置保留上一步样本

## 最终分布

| 维度 | 类别 | 样本数 |
| --- | --- | ---: |
| layout family | `slice_storage_offset` | 4,171 |
| layout family | `transpose_noncontiguous` | 3,508 |
| layout family | `expand_zero_stride` | 3 |
| source | DrKernel | 7,126 |
| source | CUDA-Agent | 420 |
| source | oubo-generated | 136 |

## 抽检结论与边界

| 检查对象 | 样本数 | 选择规则 | 结果 | 结论 |
| --- | ---: | --- | --- | --- |
| GPU 验证通过池的问题发现型抽检 | 100 | 按静态风险标签、factory/operator 数量、ndim 和 stride ratio 排序，并覆盖 family、source、operator bucket；全部 `expand_zero_stride` 强制进入 | A98、B2、C0、D0 | 两条 B 都来自 parent 原有的 batch/sequence 疑点，没有发现 layout child 新引入的语义冲突 |
| 最终 `expand_zero_stride` 全量检查 | 3 | 全部检查 | 3 条均可保留 | Tensor 分别作为只读 mask 或 RNN hidden input 使用，没有 in-place alias 写入 |
| 最终 selected pool 审查包 | 100 | 覆盖三种 layout family、source、operator bucket、factory 数量和静态风险标签 | A95、B5、C0、D0 | 没有发现 child 问题或证据不足；5 条 B 均为 parent 原有问题 |

问题发现型抽样刻意提高高风险结构的比例，A/B 比例不能外推为全量问题率

| 最终审查归因 | 样本数 | 结论 |
| --- | ---: | --- |
| child 问题 | 0 | 未发现 layout 变换引入 logical value、shape、dtype、RNG、alias 或 in-place 冲突 |
| parent 问题 | 5 | 3 条 batch/sequence 轴解释可疑，1 条 reduction/mask coupling 可疑，1 条 loss coupling 可疑 |
| 证据不足 | 0 | 每条都有完整 parent、child、diff、GPU verdict 和实际 layout metadata |

| family | A | B | C | D | 合计 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `slice_storage_offset` | 52 | 3 | 0 | 0 | 55 |
| `transpose_noncontiguous` | 41 | 1 | 0 | 0 | 42 |
| `expand_zero_stride` | 2 | 1 | 0 | 0 | 3 |
| 合计 | 95 | 5 | 0 | 0 | 100 |

静态 policy 给审查包中的 9 条样本记录了 warning。人工核对确认 3 条 batch/sequence 问题；4 条 `cat/stack` warning 只表示 layout 传播距离短，仍评 A；2 条 batch/sequence warning 因模块明确使用 `batch_first=True` 属于误报，其中一条 parent 另有 reduction/mask coupling 问题。误报根因是 AST 规则没有区分 sequence module 构造器与 `self.lstm(x)`、`self.gru(x)` 运行时调用，后者因不带 `batch_first` 参数被错误标记

| 已知边界 | 涉及样本数 | 当前处理 |
| --- | ---: | --- |
| 静态 batch/sequence warning | 49 | 记录为 review-only 标签；最终审查已证实该规则存在误报，不能直接作为 parent 问题数 |
| 首个 semantic consumer 为 `cat/stack` 等 materializer | 125 | 保留并记录 warning；GPU 验证已经证明 transformed layout 被读取 |
| 静态 policy 无法证明的其他语义关系 | - | 保留 review-only 状态，训练前仍需 provenance、去重和受控 ablation |

## 遗留问题

| 问题 | 当前影响 | 后续处理 |
| --- | --- | --- |
| `expand_zero_stride` 最终只有 3 条 | 当前只识别直接的 `zeros`、`ones` 和静态 `full`，zero-stride 覆盖不足 | 扩展静态分析以识别 `zeros_like`、`ones_like` 和其他可证明沿目标维恒定的表达式，并让具备该 eligibility 的 parent 优先选择 `expand_zero_stride` |
| 654 条通过 GPU 验证和语义 promotion 的 child 未最终纳入 | Child 绑定的 parent reference 与最终上游样本不一致，现有证据不能直接复用 | 基于当前上游 reference 重新构造并执行 Paired reference、GPU 验证和语义 promotion |
| 125 条 layout 在首个 consumer 被 materialize | Layout 已被读取，但生命周期短，训练价值可能低于能传播到多个算子的 layout | 在后续采样中单独标记 layout 传播距离，并优先补充能经过多个 consumer 的样本 |
| Layout family 覆盖较窄 | 目前只覆盖 transpose non-contiguous、positive storage offset 和 zero stride | 增加 channels-last、一般 strided slice 和 memory format 等 layout，同时维持 shape、dtype、value 和 `forward` 不变 |
| batch/sequence 静态 warning 存在误报 | `self.lstm(x)`、`self.gru(x)` 运行时调用会被误当成未声明 `batch_first` 的构造器，warning 数不能直接当问题数 | Analyzer 区分 module 构造器与运行时调用，再结合构造参数和输入轴来源判断；保留人工确认后的 parent 问题归因 |

## 产物与复现入口

### 最终产物

权威 final artifact：

`Data/prompt_tvm_v4/serial_augmentation_v1/layout_fallback_base.v2.semantic_gate_double`

| Artifact | 样本数 | SHA-256 |
| --- | ---: | --- |
| `selected.parquet` | 31,648 | `03c254abb4adac15b09507f874a76b8d988f1f2831c283e6a5b1755d35d437f2` |
| `manifest.jsonl` | 31,648 | `a865b3ba94d9928946b77df7000c4616c53696929e5853a82c5815651cfb4f2a` |
| `summary.json` | - | `47adbd69fb63f8c4ed912abf648d449252cc18dfedbecd8057318cc30db02608` |

中间 fallback base：

| Stage | Selected counts | `selected.parquet` SHA-256 | `manifest.jsonl` SHA-256 |
| --- | --- | --- | --- |
| 随机数生成分布 | 22,908 随机数生成分布 + 8,740 shape | `8c9f59797f086be416a319a6728d5254304d75655bf448fb55223cae41340982` | `ee5930d1658839a9dfcbf49636a731d5ad19c54a6033f2710c7d0c7764eb919d` |
| dtype | 7,249 dtype + 24,399 prior | `e0320f1a1f14975bb7d5f874762cadaeaca414c27e1336e233abb30d9ad2ec91` | `93ff43aa95da2282b78547f608d889f24d4505bb8737442c4988c38634690153` |

### Lane log

| Stage/lane | Candidate | Both-pass | GPU 验证通过 | Semantic pass | Final exact reuse |
| --- | ---: | ---: | ---: | ---: | ---: |
| dtype `run.parameter_free.5000.v1` | 5,000 | 4,658 | 2,745 | 2,333 | 2,033 |
| dtype `run.module_state.1000.v1` | 1,000 | 951 | 403 | 373 | 341 |
| dtype `run.parameter_free.500.v2` | 500 | 474 | 263 | 223 | 196 |
| dtype `run.module_state.5000.v2.semantic_gate` | 5,000 | 4,882 | 2,171 | 2,165 | 2,165 |
| dtype `run.module_state.5000.v3.semantic_gate` | 5,000 | 4,869 | 2,098 | 2,084 | 2,084 |
| dtype `run.module_state.1000.v4.semantic_gate` | 1,000 | 965 | 432 | 430 | 430 |
| layout `run.primary.5000.v1` | 5,000 | 4,992 | 3,600 | 3,600 | 2,946 |
| layout `run.semantic_double.5000.v2` | 5,000 | 4,993 | 3,682 | 3,682 | 3,682 |
| layout `run.semantic_double.1500.v3` | 1,500 | 1,496 | 1,054 | 1,054 | 1,054 |

### 代码和审查材料

- `tools/data/synthesize/intervention_semantic_gates.py`：统一语义 promotion；
- `tools/data/synthesize/compose_serial_augmentation.py`：串行 fallback、exact-parent evidence reuse 和最终 current-policy 重检；
- `tools/data/synthesize/dtype_method/`：dtype solver、H20 验证和 analyzer；
- `tools/data/synthesize/layout_method/`：layout solver、H20 验证和 analyzer；
- `tools/data/synthesize/sample_intervention_semantic_audit.py`：最终 selected-child 分层抽样；
- `local_artifacts/data/synthesize/intervention_semantic_audit_v2/`：门禁后 300 条审查包；
- `local_artifacts/data/synthesize/intervention_semantic_audit_v2/random/final_review.md`：最终随机数生成分布 100 条逐条评级；
- `local_artifacts/data/synthesize/intervention_semantic_audit_v2/layout/final_review.md`：最终 layout 100 条逐条评级、问题归因和 B 级证据；

这套生产数据 pipeline 没有新增单元测试。验证依赖 source/hash/static replay、Python compile、shell syntax、真实 H20 paired reference 和 GPU 验证、完整 manifest 合并和人工 diff 检查。
