# 训练数据分布修改总计划

## 摘要

所有 intervention child 统一从 64,315 条 immutable canonical parent 派生，各扩展 lane 独立生成，禁止从其他 child 或已经拼接的 review union 二次派生。每个 child 只承载一个 primary intervention，先用等预算单轴实验测量收益，再决定最终 mixture；当前没有证据支持直接按 KernelBench 频率复制分布，也没有证据支持把多个扩展做 Cartesian product。没有 parent 的 semantic synthetic task 使用独立 generator/provenance root，不能伪装成 intervention child。

执行范围更新（2026-08-05）：另一台 KernelGym 机器只负责小批量构造和验证，任一 non-shape lane 单次最多 5,000 个 parent/child；当前 random/value 以 1,000-parent canary 为交付终点，不在该机器启动 full-lane 扩增。eligible pool 只用于报告可扩展性，不构成继续放量的授权。本轮允许 A800 作为权威 canary 验收环境，只要完整记录 GPU/Torch/CUDA/cuDNN/KernelGym 和 source-bound runtime fingerprint；不要求为了 H20 硬件一致性重跑。

shape 扩展由当前会话继续负责。另一台 KernelGym 机器并行负责 random/value、coherent dtype、layout、semantic/operator 和 source/mode coverage。两边共享同一份 parent、manifest 和验证合同；runtime policy 不一致的结果只算 static candidate，不能直接合并为统一的 runtime-accepted partition。

所有现有 v4 产物都处于 review 状态，`training_approved=false`。当前训练仍使用 40,307 条 v3 数据。最终数据集必须经过 cross-lane 去重、provenance/license 审计、统一 runtime evidence 检查和受控训练 ablation，不能由任一 lane 单独发布。

## 统一基线

三个常被混用的 parquet 含义不同。所有 intervention child 只允许从第二行的 canonical parent 派生。

| 角色 | Artifact | Rows | SHA-256 | 使用规则 |
| --- | --- | ---: | --- | --- |
| 当前训练集 | `Data/prompt_tvm_v3/drkernel_rl_thinking.parquet` | 40,307 | `9e9ffca46022e74c0616f5e272871e76dfd000b6e7937b01685cd0bb11d521e5` | 训练基线；不作为新 child 的 lineage 根 |
| Canonical parent | `Data/prompt_tvm_v4/train.review.parquet` | 64,315 | `b07205fcadc543964cfc7ee5fd9c1e4d011f0f3481447f656e5297e40b4b99f4` | 所有 intervention lane 的唯一 parent 集合 |
| Coverage review checkpoint | `Data/prompt_tvm_v4/train.coverage.review.parquet` | 65,223 | `0b3fa5bd82fbdf5ac943f962998d52d9584c181906ee54a431270f95e3bd4bdd` | 64,315 parent + 396 input child + 512 semantic task；只用于 review，不得继续派生 |

本计划把数据扩展分成两类 lineage：

- mutation lanes：shape、random/value、dtype、layout。child 必须绑定 64,315-row canonical parent，且每个 child 只有一个 primary intervention。
- standalone lanes：semantic/operator synthesis 和新增 source。它们使用 generator/source、provenance、license 和 decontamination root，不填写虚假的 parent UUID。mode-only variant 若由 canonical row 改写，仍按 mutation child 管理。

## 差距口径

KernelBench 只用于诊断覆盖缺口和评测，不作为训练数据来源。下表各指标的分母不同，不能互相换算。

| 指标 | 当前训练 | KernelBench | 结论 |
| --- | ---: | ---: | --- |
| resolved tensor numel p50 | 16,384 | 33,554,432 | 中位数相差 2,048×；这不是 aggregate input bytes 的扩展倍率 |
| resolved tensor numel p90 | 3,145,728 | 1,610,612,736 | KernelBench 的大 tensor 长尾更重 |
| resolved tensor numel ≥1M | 18.2% | 88.6% | 训练侧大 shape 覆盖不足 |
| resolved tensor numel ≥100M | 0.5% | 42.2% | 极大 tensor 覆盖差距明显 |
| row 含 `randn` | 95.6% | 0% | 当前输入值先验高度集中 |
| row 含 `rand` | 2.5% | 100% | 只表示 factory presence，不等于最终值分布 |
| row 含 `randint` | 9.5% | 1.2% | factory presence 是多标签统计，各行不能相加 |

tensor numel、所有 direct input 的 aggregate bytes、allocator peak、operator workspace 和 runtime memory 是五种不同口径。shape lane 的目标按 aggregate direct-input bytes 定义，冻结后的目标 runtime 环境决定实际可行性。旧的 32×、24×/49 GiB 都不再作为 shape 接受门禁。

## 分工和边界

| 工作面 | Owner | 当前目标 | 禁止事项 |
| --- | --- | --- | --- |
| Shape | 当前会话 | 独立产出 shape partition 和统一交付 manifest | 另一台机器不生成、不验证、不调度 shape，也不等待 shape 进度 |
| Random/value | 另一台 KernelGym 机器 | value-only 单轴 child；只生成 `uniform_01`、`signed_uniform`、`poisson_counts`、`multinomial_categories` | 不生成 `boundary_pm1`；不同时改 shape/dtype/layout；不把 Poisson、multinomial 当通用连续输入替换 |
| Dtype | 另一台 KernelGym 机器 | parameter-free/proven-compatible input dtype，或 coherent precision sibling | 不只改 input factory 后让 FP16/BF16 撞 FP32 parameter、buffer 或 operand |
| Layout | 另一台 KernelGym 机器 | 独立覆盖 transposed stride、slice/storage offset、zero stride、memory format | 不和 value/dtype 先做组合；不只依赖源码形式，必须检查 runtime stride |
| Semantic/operator | 另一台 KernelGym 机器 | 补薄弱 atomic family 和 heterogeneous graph cell | 不继续堆 shape-preserving pointwise 长链；不从 KernelBench 复制评测题 |
| Source/mode/provenance | 另一台机器生成，两边共同验收 | 增加合规 lineage 和 mode/interface 覆盖，记录 license/provenance | provenance 不清的数据不得进入 review union |
| Cross-lane merge 和训练实验 | 统一 owner | 去重、统一 evidence、等预算 ablation、确定 mixture | 任一 lane 不得自行标记 `training_approved=true` |

另一台机器可以完成 static 和 A800/H20 canary。A800 或 H20 均可产生 authoritative canary evidence，前提是完整绑定 launcher、validator、KernelGym checkout、Torch/CUDA/GPU 环境及 runtime policy fingerprint；环境不同的结果必须作为独立 evidence partition 管理，不能静默混合。

## 另一台机器负责的构造合同

### Random/value

random/value 只生成 value-only child，shape、dtype、layout、`Model.forward` 和 `get_init_inputs` 保持不变。每个 parent 用 stable hash 指定一个 eligible family，避免同一 parent 展开全部 family 形成 Cartesian amplification。本轮使用确定性 solver，不需要 LLM：

- `uniform_01`：将原 `randn` draw 映射到 `[0, 1)`；
- `signed_uniform`：将原 draw 映射到 `[-1, 1)`；
- `poisson_counts`：由原 draw 构造有界正 rate，并用 stable seed 的局部 generator 采样非负整数值；
- `multinomial_categories`：仅用于静态 last dimension ≥2 的输入，以最后一维构造概率并采样 category id。

Poisson 和 multinomial 的结果仍编码在原浮点 dtype 中，并保持原 shape、layout 和全局 RNG 状态。它们是受约束的 value coverage cell，不是通用连续输入替换；是否可接受必须由 paired parent/child reference、support 检查和 output-liveness 决定，不能仅凭静态构造宣称语义安全。`boundary_pm1` 已从当前合同移除。后续 zero-heavy、sparse、repeated values、bounded magnitude tail 和 NaN/Inf 不属于本轮。

Poisson runtime evidence 必须从 parent 重算同一 rate mapping，报告固定区间的 rate histogram 和 sampled count histogram，并分别校验频次守恒；只报 min/max 不足以验收。Multinomial 必须按实际 runtime dtype 检查 category ID 的精确整数表示上限，并报告 last-dimension cardinality。

2026-08-05 的 1k A800 canary 已完成：1,000 个 candidate 中 850 个 parent/child reference 双通过，792 个再通过 value/output liveness；accepted family 为 Multinomial 205、Poisson 209、signed Uniform 189、Uniform `[0,1)` 189，`boundary_pm1=0`。产物仍为 review-only，详见 `handoffs/data/synthesize/RANDOM_VALUE_CANARY.md`。

旧 pilot 的 306 个 value intervention 中有 260 个 child pass，但其中 208 个同时改了 shape，且没有 paired parent runtime。该结果只能说明构造可执行，不能给出 value intervention 的独立通过率。新 canary 必须 paired 运行 parent/child，并把 evaluator failure、OOM、reference failure 和 intervention-induced failure 分开。

### Dtype

input-only dtype mutation 不再作为通用规则。旧 pilot 只通过 45/92，40 个 failure 来自 FP16/BF16 input 与 FP32 parameter 或 operand 不一致。

新 lane 只接受两类任务：

1. parameter-free 或静态证明全部 operand 已兼容的 input dtype child；
2. coherent precision sibling，同时处理 input、parameter、buffer、dtype-sensitive constant 和 comparison tolerance。

manifest 必须记录每类 tensor 的实际 dtype、显式 cast 和 runtime promotion。child 即使 correctness pass，只要核心路径立即隐式回到 FP32，也不能计入目标低精度 coverage。

2026-08-05 的 parameter-free 1k A800 canary 已完成。本轮使用 deterministic solver，不需要 LLM；1,000 个 candidate 中 830 个 parent/child reference 双通过，638 个再通过三次 dtype/output liveness，accepted 为 BF16 308、FP16 330。

2026-08-06 的 `module_state` coherent 1k A800 canary 也已完成，仍由 solver 构造，不依赖 LLM。child 同时修改 direct FP32 input factory，并在 `Model.__init__` 末尾显式将 registered parameter/buffer 转成 target dtype；三轮 runtime proof 检查 parent-state exact cast、non-floating state、alias、隐藏/未注册 state、构造 RNG、post-forward state 和 FP32/complex dispatch fallback。1,000 个 candidate 中 863 个 reference 双通过，479 个通过 liveness（BF16 218、FP16 261）。两轮结果均为 review-only、`training_approved=false`，详见 `handoffs/data/synthesize/DTYPE_CANARY.md`；custom conversion hook、dtype-sensitive constant、动态或未注册 state 仍不在已证明范围内。

### Layout

每个 layout pattern 单独成 cell，保持 logical shape、值和 dtype 不变。现有 transpose-contiguous-transpose pilot 通过 31/41，10 个 failure 都受 tuple-output evaluator 缺口影响，不能据此判定 layout 无效。

2026-08-06 的 1k A800 canary 已完成前三类 deterministic solver cell：

- transpose/permute 后的非 contiguous stride；
- slice 产生的 stride 和 non-zero storage offset；
- expand 产生的 zero stride；

1,000 个 candidate 中 851 个 parent/child reference 双通过，584 个再通过三轮 raw-direct layout/output liveness：transpose 262、slice 314、expand 8。Static 和 runtime gate 均保持 logical value、shape、dtype、RNG 与原 factory 求值顺序，并检查实际 stride/offset/zero stride、alias、forward 后 immutability、semantic consumer 和 exact output；unknown dispatch 或 intervention 在 consumer 前被 materialize 均 fail-closed。结果为 review-only、`training_approved=false`，详见 `handoffs/data/synthesize/LAYOUT_CANARY.md`。

channels-last 或其他 memory format 仍需逐 operator 证明 memory-format contract，本 canary 未覆盖。后续 runtime gate 仍必须读取实际 size、stride、storage offset 和 contiguity，确认 intervention 没有在进入被测计算前被 `.contiguous()` 或等价 materializer 消除。

### Semantic/operator 和 source/mode

现有 256 个 atomic task 全部通过 reference，但 156 个属于 pointwise，conv 和 matmul 各只有 7 个，pooling 3 个，indexing 4 个。现有 256 个 multi-op graph 也全部通过，但只使用 12 种 shape-preserving pointwise op，单输出、FP32、contiguous 的构造过于保守。

下一轮使用 family quota 和约束求解补充这些 cell：conv+normalization、matmul+reduction、shape-changing branch、attention/recurrent/stateful、structured output，以及 natural multi-class operator graph。每个 declared op 都要在 returned output 的 dependency graph 上；只增加源码长度、等价重写或 dead branch 不计 coverage。

source coverage 通过合规来源和受控生成扩展。Oubo 数据需先补 provenance/license；KernelBook 需固定 license 和 group split；KernelBench 保持 evaluation-only。6,799 个 mode-variant parent 仍需独立 train-mode gate，不能靠新增 child 绕过。

## 统一执行流程

另一台机器上的每个新 non-shape 合同都按下列顺序运行，并先做 1,000-parent canary。该流程不依赖 shape lane 的状态。

现有 non-shape 工具只提供旧 pilot 的实现和证据，不能直接启动新 full run：旧 generator 会混合 shape/value，dtype 仍是 input-only，layout 只有一种 pattern，semantic generator 仍偏 pointwise；旧 selector 也没有 lane filter。另一台机器先实现 lane isolation、各 lane 独立 selector、paired parent/child runner 和 value/dtype/layout liveness gate，再启动 canary。semantic/source 没有 parent，使用 reference correctness、declared-op dependency、provenance 和 decontamination gate。

```text
immutable canonical parents or generator/provenance roots
  -> lane-local deterministic selection/generation
  -> source/AST/static feasibility
  -> 1k canary + manual diff review
  -> paired authoritative runtime reference（本轮 A800 可验收）
  -> intervention-specific liveness
  -> lane bias and failure audit
  -> ≤1k random/value canary handoff
  -> any further scale-up requires separate authorization
  -> cross-lane identity/provenance/evidence merge
  -> equal-budget training ablation
  -> mixture decision and explicit training approval
```

canary 的 acceptance rate 用于估算后续运行成本和暴露构造问题，不设成训练价值代理。本机所有 non-shape 单次构造或验证不超过 5,000；当前 random/value 交付上限为 1,000，不因 eligible pool 大小自动放量。每轮至少人工复核不同 source、operator、target cell、成功与失败的真实 reference diff 和 raw runtime audit。

生产数据路径不新增单元测试，继续依赖 `py_compile`、CLI/launcher syntax、真实 canary、deterministic shard merge、exact manifest checker 和人工样本复核。任何 source-bound generator/validator 改动都会使旧 manifest 或 runtime binding 的 source hash 失效，必须重建相应 evidence。

## Lane artifact 和 manifest 合同

每个 lane 使用独立目录，至少交付：

```text
<lane>/candidates.parquet
<lane>/manifest.jsonl
<lane>/summary.json
<lane>/review_samples.md
<lane>/runtime/raw/
<lane>/runtime/accepted.parquet
```

manifest 一行对应一行 parquet，顺序固定，并包含：

- canonical parent artifact path/SHA、parent UUID 和 source row；parentless semantic synthetic task 改为 generator/provenance root；
- parent/child reference SHA 和 normalized-AST SHA；
- 唯一 `primary_intervention`、assigned target、realized intervention 和 reject reason；
- generator contract/version/source SHA、Git commit 和 dependency SHA；
- source、operator family、mode、shape/value/dtype/layout、memory 等 coverage labels；
- static、parent runtime、child runtime、liveness 和 materialization 的独立状态；
- GPU、Torch、CUDA、KernelGym checkout、launcher/validator SHA 和完整 runtime policy fingerprint；
- provenance/license 和 lineage；
- row-level artifact hash，以及 lane 内 UUID/reference/AST uniqueness 证明。

不要把 row presence、tensor occurrence、logical slot、factory-axis occurrence、child count 和 parent coverage 混成一个 coverage 百分比。每份 summary 必须写明 numerator、denominator、multi-label/exclusive 口径和 measurement stage。

## Bias 和 failure 报告

| Lane | 必报分布 | 必报 failure attribution |
| --- | --- | --- |
| Random/value | family、factory、support、rate/count 分布、category cardinality、source/operator 条件分布；另报 multinomial last-dimension eligibility 和各 family 的 paired/liveness 拒绝率 | evaluator、domain、numerical、OOM、joint contamination |
| Dtype | target dtype、parameter/buffer/constant 一致性、cast/promotion、source/operator | dtype mismatch、unsupported op、tolerance、implicit FP32 fallback |
| Layout | pattern、rank、stride、offset、contiguity、source/operator | intervention erased、unsupported stride、evaluator、reference |
| Semantic/source | family、topology、op count、shape-changing/stateful/structured cells、lineage | invalid graph、dead op、mode/state、provenance/license、evaluator |

失败归因先运行 unchanged parent。parent 和 child 同样失败时，默认归入环境/evaluator/原题问题；只有 parent pass、child fail 且 causal chain 闭合时，才归因于 intervention。

## Cross-lane merge 前的 blocker

### Runtime evidence policy 已分叉

65,223-row checkpoint 中旧 396+512 accepted evidence 绑定的 launcher/validator SHA 与当前工具不同：

| Evidence | Launcher SHA prefix | Validator SHA prefix |
| --- | --- | --- |
| 旧 accepted partition | `908beaf4` | `87f07b34` |
| 当前 source-bound policy | `da63f011` | `691b6262` |

当前 assembler 要求 accepted partition 的 `runtime_policy_fingerprint` 和 GPU/Torch/CUDA 环境一致。推荐在统一当前环境重新验证旧 908 行，再和新 lane 合并。另一种方案是实现 versioned semantic-compatible policy projection；该方案需要显式 review，不能通过忽略 tool hash 或放松检查静默合并。

authoritative runtime evidence 前先由最终 merge owner 冻结 Git commit、container digest、KernelGym checkout、Torch/CUDA、GPU type、launcher、validator 和 evaluator policy。tuple/structured-output 等 evaluator 修复必须在 freeze 前完成。freeze 前的结果只算 canary；旧 908 行也在 freeze 后重验，避免反复产生不兼容 evidence。

### Shape partition 的汇合边界

shape partition 由当前会话单独交付。另一台机器只提交自己的 non-shape lane artifact，不修改 shape schema，也不负责总 union；最终 merge owner 统一适配各 lane schema、去重并验证 evidence。

### KernelGym 和跨机器同步

当前 reference launcher 直接 import 本机 KernelGym checkout，不调用远端 HTTP endpoint。另一台机器只有 endpoint 时无法生成 materializer-compatible evidence。跨机器运行前必须同步 Git commit、未入 Git 的 parquet、KernelGym checkout 和 runtime image，并逐项核对 SHA；路径相同不能替代内容校验。

materialize/merge 还依赖 decontamination baseline `Data/external/converted/kernelbench_level1_2_3.reference.parquet`，SHA-256 为 `b4de490253a0f5a1f5e971f0f723cffd1de2cc2506ceabe97a369804e3a3d7f2`。该文件不在 `Data/prompt_tvm_v4/**`，需要单独同步和校验。

## 训练实验和最终 mixture

先做 equal-budget、single-axis ablation，所有实验使用相同 parent/source budget、训练 token budget、模型 checkpoint 和 evaluation denominator：

| Arm | 新增数据 |
| --- | --- |
| B0 | 当前训练 baseline |
| B1 | shape only |
| B2 | random/value only |
| B3 | coherent dtype only |
| B4 | layout only |
| B5 | semantic/operator only |

单轴结果稳定后再做 cumulative arm 和小规模 joint-cell 实验。最终 mixture weight 依据 KernelBench 分层 correctness、compile/runtime failure、训练内 holdout、source/operator/shape slice 和回归风险决定，不按候选池大小或 KernelBench 原始频率自动设置。

最终分母上重新计算 simple/Level1-like 占比，现有 30% cap 只作为防止简单题稀释的 merge gate。DrKernel 的 complexity heuristic 对其他来源存在域偏移，只用于诊断，不能跨来源直接下采样。

发布条件包括：所有 row 可追溯、cross-lane UUID/reference/AST 唯一、runtime evidence policy 一致、failure attribution 可复核、关键 coverage cell 有 runtime-accepted 样本、单轴 ablation 无不可接受回归，并由人工显式设置 `training_approved=true`。

## 旧实现入口和可复用证据

| 用途 | 路径 | 本轮定位 |
| --- | --- | --- |
| 全局差距摘要 | `handoffs/data/synthesize/train_eval_diff.md` | 诊断依据 |
| v4 构建与旧 pilot | `handoffs/data/synthesize/DATA_CONSTRUCTION.md` | 旧证据 |
| Input intervention generator | `tools/data/synthesize/augment_prompt_tasks.py` | legacy 起点；先拆 lane |
| Coverage selector | `tools/data/synthesize/select_augmentation_coverage_pilot.py` | legacy 起点；先加 lane filter/quota |
| Semantic generator | `tools/data/synthesize/generate_extreme_op_tasks.py` | legacy 起点；不能继续原样扩 pointwise |
| Reference launcher | `tools/data/synthesize/launch_reference_validation_shards.sh` | 需扩 paired evidence binding |
| Runtime validator | `tools/data/synthesize/validate_train_mode_contract.py` | freeze 后绑定 source SHA |
| Materializer | `tools/data/synthesize/materialize_validated_candidates.py` | 需适配新 lane schema |
| Union assembler | `tools/data/synthesize/build_prompt_tvm_v4_coverage.py` | 最终 merge owner 使用 |
| Distribution profiler | `tools/data/synthesize/profile_prompt_tvm_distribution.py` | 可复用审计入口 |

另一台机器开始前应确认上述 non-shape 主链文件已经进入同一个 pushed commit。交接 branch 为 `v4flash-lora`，shape 基础实现 commit 为 `0238144221f12241ab5cef55cc81645d024e63b0`；本计划的 commit SHA 以该文件的 `git log -1` 为准。`Data/**` 不随 Git 分发，必须按本计划记录的 row count 和 SHA 单独同步。

本计划定义协作与验收合同，不批准任何现有或未来 partition 进入训练；`training_approved=false`。
