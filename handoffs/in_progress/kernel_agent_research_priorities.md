# Kernel agent research priorities

Kernel agent 已经能生成、修复并迭代优化固定 contract 下的 CUDA/Triton kernel。下一阶段的核心问题是判断：什么值得优化、哪个变换产生收益、结果在哪些输入与系统中成立，以及 sandbox speedup 能否转化为生产收益

本文只维护研究优先级、最小交付物和验收指标。Evaluator 攻击面与当前实现缺口由 `handoffs/deepseek-v4/step440_step620_kernelbench_failure_attribution_20260729.md` 维护；数据构建、训练 recipe、评测结果和运行状态由各自 handoff 维护

## 决策

按依赖关系推进三个计划

1. P0 Production-grounded verifier and reward
2. P1 Generalization-first kernel portfolio
3. P2 Causal transition-graph learning

P0 是后两项的测量基础。P1 定义 agent 应交付的生产 artifact。P2 改进产生该 artifact 的数据、credit assignment 与搜索效率

## 证据边界

| Evidence | 可支持的结论 |
|---|---|
| [KernelBench-Verified](https://arxiv.org/abs/2607.16241) 将最佳模型在 single-turn best-of-5 protocol 下的 geomean speedup 从标准评测的 `1.43×` 修正为 `0.88×`，且该模型的正确 kernels 中 28% 增加 peak memory | Baseline、隐藏输入分布和资源指标会改变结论 |
| [KernelBench-X](https://arxiv.org/abs/2605.04956) 的 6 个 quantization tasks 在 5 个方法上合计 `0/30` 成功 | 普通 compile+allclose 不能覆盖低精度 numerical contract |
| [FastKernels](https://arxiv.org/abs/2605.23215) 将 kernel 放回 production-like inference framework 后，最强 agent 的 aggregate speedup 为 `0.94×` | 单 kernel sandbox 分数不能代表端到端部署收益 |
| [KernelGenBench](https://arxiv.org/abs/2607.27231) 消耗超过 150 亿 tokens；同一 agent 的成功率可从 NVIDIA 的 87% 降到另一平台的 25%，specialized agent 平均每个成功 operator 约 511 万 tokens | 跨硬件泛化和搜索成本都是一阶约束 |
| [SOL-ExecBench](https://arxiv.org/abs/2603.19173) 从 124 个 production/emerging models 抽取 235 个 kernel problems，并加入 hardware SOL bound、锁频、L2 清理、进程隔离和静态检查 | 性能 reward 需要固定测量协议和硬件 headroom |
| [Dr. Kernel](https://arxiv.org/abs/2602.05885) 的 TRLOO 修复 turn-level GRPO self-inclusion bias | 正确的 advantage estimator 很重要，但仍未定位单个 semantic edit 的因果贡献 |
| [KLineage](https://arxiv.org/abs/2605.28213) 将 optimization skill 表示为带适用位置、条件、效果与 failure guard 的验证对象 | Skill 应是 condition-aware rule，并带执行证据 |
| [µCUTLASS](https://arxiv.org/abs/2603.29010) 表明过低与过高抽象都会损失效率或自由度，并用 DSL+SOL guidance 降低搜索成本 | Agent 与 compiler 需要保留优化杠杆的结构化接口 |

这些结果来自不同 workload、硬件、baseline 和预算，只用于确定研究风险与优先级，不能跨论文直接排序模型

## P0 Production-grounded verifier and reward

### 目标

把 reward 从单个 shape 的 scalar speedup 改成 correctness-gated production utility

```text
task contract
+ semantic correctness
+ numerical contract
+ latency distribution
+ memory/resource cost
+ deployment coverage
+ full-system effect
```

### 最小 contract

每个任务显式声明

- 合法 shape、dtype、layout、stride、alignment 与 storage offset
- aliasing、in-place、determinism、NaN/Inf 和 accumulation 语义
- forward、backward 与需要支持的 higher-order gradient
- 目标 GPU、compiler/runtime 版本和允许使用的 library
- latency/throughput、peak memory、compile time 与精度容差的权重

### Verifier layers

1. Static provenance 与 forbidden-call 检查
2. Public smoke tests，提供最小可调试反馈
3. Hidden distribution、boundary、non-contiguous 与 adversarial tests
4. Sanitizer、race、determinism 与 numerical-stability checks
5. Baseline/SOL calibrated performance harness
6. Production framework integration 与 end-to-end regression

Hidden verifier 只返回分级结果，不暴露逐例答案，避免 multi-turn agent 反向拟合测试集

### Reward representation

保留原始 measurement vector，训练时再映射成 objective

```text
correctness_vector
latency_samples
baseline_samples
peak_memory
compile_cost
SOL_headroom
system_throughput_or_p99
```

每个 task manifest 在看见 candidate 前冻结 vector 到 utility 的映射，correctness failure 的 utility 为 `0`。跨任务先将 baseline utility 归一为 `1`，再报告 `correct_fraction × correct_tasks_geomean_utility`；没有 correct task 时结果为 `0`

接近噪声区间的 candidate 继续采样或返回 uncertain，不用单次测量强制排序

### P0-A harness 验收

先冻结 `verifier_regression_v1` manifest，包含 `step440_step620_kernelbench_failure_attribution_20260729.md` 已确认的全部 bypass artifacts、等量 matched benign kernels、hidden inputs、环境 fingerprint 和 SHA256。实现完成的门禁为

- 已知 attack recall `100%`，matched benign pass `100%`
- PyTorch fallback、旧输出缓存、计时篡改和 hidden-feedback probing 四类 ASR 均为 `0%`
- 每个 artifact 在 3 个隔离进程中得到相同 correctness/uncertain/speedup label
- Candidate 相对 baseline 的 speedup 95% confidence interval 与 `[-1%, +1%]` 有交集时标记 uncertain，不计作 speedup
- 每条结果完整保存 latency samples、peak memory、compile time、SOL headroom 和失败类别

### P0-B production utility gate

初始 integration workload 固定为本仓库 DeepSeek-V4 A1 attention：candidate 替换 `custom_kernels/deepseek_v4/attention/` 后运行现有 correctness suite，并用同一 saved rollout 执行 `--debug-train-only` replay

- Kernel correctness suite 全部通过
- Replay token mask 与 baseline 完全相同，loss absolute difference 不超过 `1e-3`，所有 gradient finite
- 3 个独立 process 各 warmup 3 步、测量 5 步；step-throughput 95% confidence interval 不低于 baseline `-1%`
- Peak allocated memory 不高于 baseline `+5%`
- 无新增 graph break、layout copy 或同步事件

P0-A 完成后才能采集可信 transition graph；P0-B 在 portfolio 形成后验收 production utility

## P1 Generalization-first kernel portfolio

### 目标 artifact

```text
variants[]:
  precondition
  implementation
  correctness_scope
  performance_envelope
dispatch_policy
generic_fallback
integration_contract
```

Specialization 可以存在，但必须显式声明 guard。未命中 guard 时使用已验证 fallback

### 泛化轴

训练和评测至少对以下维度做独立 holdout

- shape 与 batch regime
- dtype、accumulation 与 quantization format
- contiguous、strided 与 aliased layout
- GPU architecture 与 memory hierarchy
- compiler、driver 与 framework version
- input value distribution

### Condition-aware skill

每条可复用 skill 保存

```text
precondition
semantic_action
expected_effect
scope
risk
positive_evidence
negative_evidence
```

Skill library 的 admission 需要跨实例复验。检索同时使用 task contract、profile state 和 code/IR location，避免仅靠自然语言相似度

### 多层表示

Agent 先选择 algorithm、fusion boundary 与 data layout，再选择 schedule、tile、warp、stage 和 low-level implementation。Compiler 暴露结构化 IR、resource estimate 和 legal transformation，不要求模型重复生成样板代码

### P1 验收

对每个 held-out config `c`，在每个 variant 固定 20 次隔离测量后定义

```text
dispatch_regret(c) =
  (latency(dispatch(c)) - latency(best_correct_variant(c)))
  / latency(best_correct_variant(c))
```

Oracle 只能从同一冻结 portfolio 选择，不获得额外 candidate 或 profile budget。通过门禁为

- Held-out config correctness 与 fallback coverage 均为 `100%`
- `p95(dispatch_regret) <= 5%` 且 `max(dispatch_regret) <= 10%`
- 新 GPU/compiler holdout 上所有未命中 guard 的输入都进入 fallback
- Portfolio 相对最佳 single variant 的 paired geomean latency 至少改善 `5%`，95% bootstrap confidence interval 下界大于 `0`
- Peak memory 不高于最佳 single variant `+5%`，且无新增 graph break、layout copy 或 downstream fusion loss

## P2 Causal transition-graph learning

### 数据单元

线性 trajectory 改为带 sibling counterfactual 的 transition graph

```text
task_contract
environment_fingerprint
parent_kernel
parent_profile
semantic_action
child_kernel
compile_result
correctness_vector
latency_distribution
profile_delta
sibling_candidates
accepted_or_rejected
downstream_consequence
```

保存失败 edit 和暂时变慢但为后续优化提供前置结构的 edit，避免只学习单调局部 patch

### Credit assignment

- 将 diff 解析成 algorithm、layout、fusion、schedule、tile 与 implementation actions
- 同一 parent 主动采样多个 sibling edits
- 用 sibling outcome 估计 action-level counterfactual value
- 对跨多步才兑现的布局或接口变更保留 delayed credit
- Token policy 学习实现 action，action policy 学习选择优化方向

TRLOO 继续负责 turn-level unbiased advantage；semantic-action credit 是独立层，不能用整轮 reward 平摊代替

### 搜索调度

- 用 SOL headroom 决定是否继续优化
- 用 static analysis 和 compile cache 淘汰重复候选
- 用 uncertainty 分配 high-fidelity profile 次数
- 保持 population diversity，区分 exploration 与 local repair
- 同时优化 token、wall-clock、GPU-hour 与 high-fidelity evaluation 数量

### P2 验收

先冻结 `transition_graph_v1` 的 task-level train/held-out split 和 `delayed_edit_v1` 人工标注集。所有方法对每个 parent 共享以下上限

```text
32 child proposals
128k generated tokens
8 high-fidelity profiles
1 GPU-hour
```

Success 定义为通过 P0 hidden correctness、相对 baseline latency 至少改善 `5%` 且 95% confidence interval 下界大于 `0`，同时满足 task manifest 冻结的 peak-memory 与 compile-cost 上限。P0-B framework integration 不计入本阶段 success

两个 end-to-end 对照分别使用同模型的 token-only value 和 final-return regression；proposal sampler、search scheduler、parent split 与上述四项预算完全相同

- Held-out parent success rate 至少提高 `5` percentage points，paired bootstrap 95% confidence interval 下界大于 `0`
- Held-out aggregate utility 的 paired bootstrap 95% confidence interval 下界大于 `0`
- 相对两个对照中较强的一方，每个成功 artifact 的 tokens、GPU-hours 和 high-fidelity profile 数至少各下降 `20%`
- Held-out sibling action ranking 的 pairwise accuracy 相对两个 baseline 至少提高 `5` percentage points，95% confidence interval 下界大于 `0`
- `delayed_edit_v1` 上必要前置 edit 的 precision、recall 和 F1 均不低于 `0.70`
- Dataset loader 对每条 edge 的 contract、environment、parent/child、action、correctness、latency 和 downstream consequence 字段完成 `100%` schema validation

## 推荐执行顺序

1. 冻结 `verifier_regression_v1`，完成 P0-A evaluator
2. 用当前 agent 采集 parent/sibling transition graphs，不先改变训练算法
3. 定义 kernel portfolio schema 和 dispatch evaluator，建立 P1 baseline
4. 训练 action-level proposal/value model，做固定预算 P2 A/B
5. 将最佳 portfolio 接入 DeepSeek-V4 A1 workload，执行 P0-B production utility gate

每一步都必须产出可复核 artifact、固定输入 hash、环境 fingerprint 和独立复测结果。后续阶段不得用更大的搜索预算掩盖前一阶段的失败

## 统一报告

| 维度 | 必报指标 |
|---|---|
| Correctness | public/hidden pass、failure category、numerical error、determinism |
| Performance | latency distribution、throughput、SOL gap closure |
| Resources | peak memory、compile time、energy proxy |
| Generalization | held-out shape/dtype/layout/GPU/compiler |
| Deployment | framework-level correctness、throughput/p99、graph/copy regressions |
| Search cost | tokens、wall time、GPU-hours、compile/profile counts |
| Causality | sibling coverage、action ranking、delayed-effect recovery |

跨方法比较必须固定 workload、hardware、baseline、verifier、budget 和 artifact-selection rule

## Non-goals

- 只提高一个公开 benchmark 的 Fast@1
- 继续堆积缺少中间证据的 `PyTorch → final kernel` pairs
- 让模型直接看到 hidden verifier 的逐例失败
- 用单个 kernel 的局部 speedup 代替系统级收益
- 为所有任务固定同一种 DSL 或最低层代码表示
