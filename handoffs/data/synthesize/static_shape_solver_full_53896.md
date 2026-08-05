# 全量 static shape solver 合成与 H20 验证

严格双 slot lane 之后的 12,318-parent 可变 2--5 slot 恢复任务已暂停；当前证据、未完成边界和恢复命令见 [`static_shape_solver_variable_multislot_delta12318.md`](static_shape_solver_variable_multislot_delta12318.md)。

## 摘要

这轮用确定性 static solver 扩大现有任务的输入 shape，不调用生成模型。solver 从 64,315 个 canonical parent 中排除 5,987 个未满足静态可测或选择条件的 parent，以及 4,432 个输入已经达到 128 MiB 的 parent，对剩余 53,896 个 parent 各分配一个 `Medium` 或 `Large` 目标。静态 identity、容量、shape 关系和 FakeTensor 门禁接受 17,353 个 child；四台 H20 的 paired reference 与 changed-region 两阶段验证最终保留 13,743 个。

每个 child 恰好改两个逻辑 shape slot，且只改 `get_inputs()` 中已有的整数 span。`Model`、`get_init_inputs()`、factory 顺序、dtype、rank 和其余源码保持不变。最终所有 rank≥2 direct input factory 都满足 `最大维度 <= 1000 × 次大维度`，因此 `[4, 4, 618492, 4]` 这类单维度比其余最大维度高五个数量级的 shape 无法进入产物。全部静态 child 和 runtime-eligible child 都通过独立重解析，比例门禁零违规。

13,743 个 child 仍是独立 review 集合，`training_approved=false`，没有并入 65,223 行的 v4 coverage union，也没有被任何训练 launcher 选中。当前证据证明源码改动边界、reference 自洽性和扩展区域 liveness；尚未证明它能提高 CUDA policy 的训练效果。

## 构造合同

每个 parent 通过固定 salt 的 SHA-256 分配一个 variant 和一个逐字节随机容量目标。两个 variant 的边界均为闭区间，child 输入总容量至少是 parent 的 2 倍，目标相对误差最多 25%。

| Variant | 目标输入容量 | 分配 parent | 静态 child |
| --- | ---: | ---: | ---: |
| `Medium` | 64–256 MiB | 26,943 | 8,618 |
| `Large` | `[256 MiB + 1 byte, 4 GiB]` | 26,953 | 8,735 |

solver 只接受严格 structural pair：两个 slot 必须作用于同一组非空 input factory，在每个 factory 的不同 axis 上各出现一次，两个值都增大。两个新值中恰好一个是 2 的幂，另一个不是，因此每个 child 的 changed occurrence 中 2 的幂占比局部固定为 50%，不会因分片、失败重试或 runtime 子集筛选而漂移。每个 child 都有两个逻辑 slot，单维改动占比为 0%。

source-span patch 后执行以下 fail-closed 门禁：

- 同时重放两个精确 UTF-8 span，生成的源码必须与 child 逐字节一致。
- `Model` 和 `get_init_inputs()` AST 必须与 parent 完全相同；factory 数量、顺序、rank、调用名和未声明 axis 不得变化。
- parent 和 child 都必须通过 30 秒 FakeTensor forward。
- 每个最终 rank≥2 direct input factory 必须满足 `largest <= 1000 × second_largest`；无法精确解析时拒绝。rank-1 没有次大维度，比例规则不适用。
- fixed-record guard 拒绝 forward 只读取固定索引、扩展后必然产生 dead tail 的 axis。
- 输入容量、2 倍增长、variant 边界和随机 target 误差必须同时满足。

## 静态结果

| 阶段 | Parent / child | 说明 |
| --- | ---: | --- |
| Canonical parent | 64,315 | canonical 候选总数 |
| 已有至少 128 MiB 输入 | 4,432 | 不再扩展 |
| 未满足静态可测或选择条件 | 5,987 | 包含无法解析、无 direct factory 或 0-byte 等情况 |
| Solver 输入 | 53,896 | 每个 parent 只分配一个 variant |
| parent FakeTensor pass | 52,237 | 290 failed、2 timeout、1,367 unsupported |
| 静态 child | 17,353 | 覆盖 32.20% solver 输入 |

未接受的主因是找不到满足合同的 structural pair，共 30,180 个；另有 4,699 个候选在 child FakeTensor forward 失败，5 个超时。所有 17,353 个静态 child 都通过独立 identity verifier，缺失、孤儿、无效 child 均为 0。

目标容量拟合较准，相对误差 P50 为 0.0379%，P90 为 0.0845%，最大为 1.3583%。最终输入容量最大为 4,294,705,152 字节，低于 4 GiB 上限。

静态接受率存在明显 source 和 operator 偏差：

| 分组 | Solver 输入 | 静态 child | 接受率 |
| --- | ---: | ---: | ---: |
| `cuda_agent` | 2,351 | 129 | 5.49% |
| `drkernel` | 38,327 | 7,873 | 20.54% |
| `kernelbook` | 11,788 | 8,565 | 72.66% |
| `oubo_generated` | 1,430 | 786 | 54.97% |
| `convolution` | 10,826 | 3,051 | 28.18% |
| `indexing` | 1,667 | 388 | 23.28% |
| `matrix` | 10,928 | 2,637 | 24.13% |
| `pointwise_other` | 13,156 | 5,721 | 43.49% |
| `reduction_normalization` | 14,938 | 4,599 | 30.79% |
| `shape_layout` | 2,381 | 957 | 40.19% |

这些差异由 structural pair 可解性、target/capacity、1000:1 比例、storage polynomial feasibility 和 FakeTensor 支持范围共同形成，不能当作目标训练 mixture。

## H20 两阶段门禁

H20 验证使用 node64、node53、node69、node70 的 32 张 H20。每个阶段固定为 256 个 virtual shard，每台机器 8 个进程并行处理 8 组 shard。四台机器的输入 parquet、manifest、allowlist、launcher 和 validator hash 一致，最终 exact shard checker 都通过，缺行、重复、跨 shard UUID、坏日志和 contract mismatch 均为 0。

### Paired reference

reference 阶段运行 17,353 组 unchanged parent/child，共 34,706 行。每行使用 production KernelGym correctness、persistent `training=True` model、paired RNG、5 个 trial、180 秒超时和 64 GiB allocator 上限。只有 parent 和 child 同时通过的 pair 才进入 region allowlist。

| 结果 | Pair |
| --- | ---: |
| parent 与 child 都通过 | 14,390 |
| 仅 parent 通过 | 1,292 |
| 仅 child 通过 | 4 |
| 两者都不通过 | 1,667 |

unchanged parent 通过 15,682/17,353，child 通过 14,394/17,353。四个 `child_only` 全部来自 baseline-invalid 的 KernelBook 题，包含未定义或未初始化参数；它们不构成 shape 扩展改善 reference 的证据。`neither` 中 1,241 对是 tuple 输出触发 KernelGym 的 `tuple.shape` evaluator 缺口，属于基础题/评测器问题。

reference 原始 top-level 状态中有 1,168 行 `cuda_out_of_memory`。KernelGym 另有 106 行把 `torch.OutOfMemoryError` 捕获在 `kernelgym_metadata.runtime_error_name` 中，top-level 状态为 `failed`；两组互不重叠，因此 OOM classifier 得到 1,274 个唯一 child 行，parent 行为 0。其中 `parent_only` 1,065 个、`neither` 209 个；`Large` 955 个、`Medium` 319 个。17 行 timeout 中最慢的是 ConvTranspose3d，耗时 184.42 秒。

`parent_only` 是最能排除 baseline-invalid 的 shape-sensitive paired bucket。自动 OOM union classifier 在这个 bucket 中识别出 1,065 个 OOM；其余还包括 output mismatch、shape 关系破坏、worker error 和 timeout。Large 的失败更多：reference child 通过率为 79.89%，Medium 为 86.05%。维度比例只限制输入几何形状，无法限制 attention、卷积或 reduction 的中间 activation 和 workspace；一个约 4 GiB 的 attention 输入在执行时请求了约 2 TiB 显存。FakeTensor 也无法覆盖所有 input-target、batch-matmul、broadcast 和初始化参数之间的动态关系。

### Changed-region liveness

region 阶段只运行 14,390 个 reference 双通过 child。validator 用 seeds `17`、`10024`、`20031` 分别扰动扩展出来的区域；三个 trial 都要观测到一致的 output effect，单行超时为 600 秒。

| 结果 | 全部 | `Medium` | `Large` |
| --- | ---: | ---: | ---: |
| `passed` | 13,743 | 7,112 | 6,631 |
| `rejected` | 249 | 135 | 114 |
| `unsupported` | 231 | 118 | 113 |
| `failed` | 166 | 49 | 117 |
| `timeout` | 1 | 0 | 1 |

表中的 `failed` 是 analyzer 的归一化汇总桶，由原始状态 114 个 OOM、51 个 `error` 和 1 个 `gpu_memory_telemetry_error` 组成。

249 个 `rejected` 都是 `expanded_region_no_output_effect`：在 seeds `17`、`10024`、`20031` 的三个 bounded perturbation trial 中，150 个 child 有一个 slot 的新增区域没有 output effect，99 个两个 slot 都没有。这个证据足以拒绝当前 child，但只有人工读过源码的样本才能进一步证明结构性 dead tail。v5 清洗检查的是原始参数整体是否影响输出；原始长度为 4 且 forward 只读取索引 0–3 的参数在清洗时完全 live，扩到更大长度后才会产生 dead tail。fixed-record guard 能静态识别直接固定索引，但间接索引、切片和算子内部截断仍需 region gate 捕获。

231 个 `unsupported` 包含 67 个三 seed effect 不一致、127 个 unchanged control 非 bit-exact、35 个 factory 无法映射 positional argument，以及 2 个放大后产生 non-finite 的 case。51 个 `error` 中，50 个是 weight norm/非叶 Tensor 使 validator `deepcopy` 失败，另一个是普通 tensor init attribute 没有迁移到 CUDA。唯一 telemetry error 已由源码和 stderr 定位为基础题 one-hot class 上界错误，并暴露 reference 对异步 device assert 的漏检；唯一 timeout 来自基础题在 model init 在线下载 528 MB VGG16。OOM、validator error 和 unsupported evidence 不证明 liveness，telemetry 单例则是已确认的基础题错误；它们均不进入最终集合。

## 最终分布与偏差

runtime-eligible 集合为 13,743/17,353，静态 child 留存率 79.20%，覆盖 53,896 个 solver parent 的 25.50%。每个 parent 最多保留一个 child。

| 分组 | 静态 child | Runtime eligible | 留存率 |
| --- | ---: | ---: | ---: |
| `Medium` | 8,618 | 7,112 | 82.52% |
| `Large` | 8,735 | 6,631 | 75.91% |
| `cuda_agent` | 129 | 61 | 47.29% |
| `drkernel` | 7,873 | 6,265 | 79.58% |
| `kernelbook` | 8,565 | 6,806 | 79.46% |
| `oubo_generated` | 786 | 611 | 77.74% |
| `convolution` | 3,051 | 2,358 | 77.29% |
| `indexing` | 388 | 231 | 59.54% |
| `matrix` | 2,637 | 1,708 | 64.77% |
| `pointwise_other` | 5,721 | 4,764 | 83.27% |
| `reduction_normalization` | 4,599 | 3,888 | 84.54% |
| `shape_layout` | 957 | 794 | 82.97% |

上表的总留存包含 unchanged parent 的 baseline/evaluator 失败，source 和 operator 差异属于观察性偏差，不能全部归因于 shape 扩展。

shape value 分布满足指定的 30%–50% 2 的幂区间，但这是构造合同带来的精确 50%，不代表自然分布。静态集合有 40,884 个 changed occurrence、8,679 个不同新值；其中 20,442 个 2 的幂 occurrence 只覆盖 19 个值，Top-5 占该子集 79.68%。另 20,442 个非 2 的幂 occurrence 覆盖 8,660 个值，Top-5 仅占 0.38%，模 8/16/32 的 `-1` residue 接近均匀基线。runtime 筛选后仍为精确 50%，因为每个 child 局部成对携带一个 2 的幂 slot 和一个非 2 的幂 slot；因此无法从这批数据独立估计 runtime 对 2 的幂的偏好。

| 分布指标 | 静态 17,353 | Runtime eligible 13,743 |
| --- | ---: | ---: |
| changed occurrence | 40,884 | 31,766 |
| unique 新值 | 8,679 | 7,502 |
| effective values | 478.17 | 450.40 |
| 2 的幂 | 50.00% | 50.00% |
| multiple-of-8 | 55.96% | 55.93% |
| leading occurrence | 32.92% | 32.30% |
| 输入容量 P50 | 280.62 MiB | 249.50 MiB |
| 输入容量 P90 | 3,289.50 MiB | 3,203.40 MiB |
| 输入容量最大值 | 4,095.75 MiB | 4,095.75 MiB |

static→eligible 前后的 axis composition 变化较小。观察上，`Large`、`indexing`、`matrix` 和容量右尾留存较低；`cuda_agent` 也低，但静态样本只有 129 个。Medium/Large 与容量带完全共线，axis、source、operator、capacity 和 baseline/evaluator 失败彼此相关，当前数据无法识别各因素的独立因果效应。后续若用于训练，需要显式决定 source/operator reweighting，不能把 13,743 行直接视为无偏样本。

## 维度平衡复核

analyzer 没有信任 solver 保存的 balance evidence，而是重新解析每个 child 的全部 direct input factory，并用整数乘法判断边界。

| 范围 | 样本数 | P50 | P90 | P99 | 最大值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 静态 child 最大比例 | 17,353 | 1.4668 | 5.3333 | 62.1580 | 1000.0000 |
| 静态 rank≥2 final factory | 20,910 | 1.4272 | 4.9610 | 58.9074 | 1000.0000 |
| 静态受改动 factory | 20,442 | 1.4342 | 4.8943 | 57.4625 | 964.6484 |
| Runtime eligible rank≥2 final factory | 16,111 | 1.4333 | 4.6400 | 55.7199 | 1000.0000 |
| Runtime eligible 受改动 factory | 15,883 | 1.4382 | 4.6400 | 54.7388 | 964.6484 |

静态集合共有 20,968 个 direct factory，其中 20,910 个 rank≥2、58 个 rank-1；runtime-eligible 子集对应 16,144、16,111 和 33 个。40,884 个 changed occurrence 全部来自 rank≥2，没有改动 rank-1。唯一恰好等于 1000 的 final factory 是 parent 已有且没有改动的 `[1, 1000]`。所有受改动 factory 都低于边界，最大值为 `[246950, 256]`，比例 964.6484。违规和无法解析均为 0。

## 人工复核

静态分层样本覆盖 source、operator、variant、slot kind、leading/non-leading axis、最接近/最远 target、最大容量和最大 scale。复核样本只在 `get_inputs()` 修改两个声明的 shape slot；未发现 docstring、等价重写、4D 变 2D、随机方法替换或程序结构变化。

| Child | Parent shape → child shape | 观察 |
| --- | --- | --- |
| `shapesolver_07cd195d1e1bd626654ea5c2` | `[16,3,H,W] → [28,4,H,W]` | 全集最远 target，误差仍只有 1.3583% |
| `shapesolver_38ea7775623cd158988f3e57` | 三个 `[200,32,E] → [1838,256,E]` | shared trailing-axis literals 在三个 factories 同步改两个 axis |
| `shapesolver_ddcd2f0efa2613964e73271c` | `[4,4,4,4] → [4,4,1561,2048]` | non-leading spatial 扩展，比例受控 |
| `shapesolver_b1198eabf9b805f64464a367` | `[8,16,32,32,32] → [8,16,32,521,256]` | 5D convolution 输入，rank 不变 |
| `shapesolver_e22ccf22c2846288f4b15361` | `[4,4,4] → [16384,4,16383]` | 最大容量约 4 GiB，两个大维接近平衡 |

H20 原始结果另按 status 和 failure reason 做了确定性抽样。全量 14,390 行都恰好有两个互异 `slot_id`。13,743 个 `passed` 行全部完成三个 trial，seeds 均为 `[17,10024,20031]`，两个 slot 在每个 seed 下均 `output_changed=true`；环境统一为 NVIDIA H20、Torch 2.11.0+cu129、CUDA 12.9。18 个抽样 diff 都只改 `get_inputs()` 的 shape 数值或局部变量；个别旧注释继续描述 parent shape，因为 solver 不修改注释或文档文本。

| Child | Region 结果 | 人工判断 |
| --- | --- | --- |
| `shapesolver_001f04b7c469af2749785f3e` | `rejected` | forward 只读 `x[:diag_dim]` 的前 10 行，新增 batch tail 永远无效 |
| `shapesolver_7b4912b59f08eaec6cb7fc59` | `rejected` | forward 只取 `x[...,0]` 和 `x[...,1]`，新增 width tail 无效 |
| `shapesolver_0469eb00caecd88ce1d9d7af` | `unsupported` | 一个 slot 的三个 seed effect 为 `[false,true,true]`，不满足一致性合同 |
| `shapesolver_0059d8a14f3ab17bda03ba39` | OOM | 输出比较的临时 tensor 超出显存，源码仍只有 shape span 改动 |
| `shapesolver_0e3c58450acd95e0c3e4b982` | `error` | weight norm 非叶 Tensor 使 region validator `deepcopy` 失败 |
| `shapesolver_f7e5a9b64dad3cc78cec2ad0` | telemetry error | 基础题 one-hot class 上界不足，异步 device assert 暴露 reference 缺口 |
| `shapesolver_0fd1cba7ad66eab16dbad543` | `timeout` | model init 在线下载 VGG16，属于基础题外部副作用 |

OOM、unsupported、error 和 timeout 保持原始分类，不被包装成 solver 正确性失败。

## 适用边界

- 结果只验证 reference task；没有生成或运行 CUDA kernel，也没有测训练收益、最终 correctness 或 speedup。
- FakeTensor 是候选剪枝，不是 runtime 证明。H20 reference 和 region evidence 才决定最终 eligibility。
- 1000:1 规则约束 input factory 的维度平衡，不约束 output、intermediate activation、workspace、计算复杂度或 rank-1 的绝对长度。
- region `rejected` 证明新增区域在三个 deterministic seed gate 下都没有 output effect，足以拒绝当前 child；`unsupported`、OOM 和 worker error 只能说明当前 validator 无法接受。
- 最终集合保留明显的 source/operator/capacity selection bias，训练前需要单独审阅 mixture 和去重策略。
- 当前产物与 65,223 行 coverage union 独立，不能直接拼接或替换训练 parquet。

## 执行说明

CPU 阶段在四台机器上并行执行 4,096 个 deterministic shard；每台默认 32 个 worker，线程库限制为单线程。500 和 5,000 parent canary 均先通过相同合同，再启动 53,896 parent 全量。H20 reference 与 region 各用四台机器、32 张卡和 256 个 dynamic queue shard；所有服务和 worker 已结束，没有遗留 GPU process。

分片 merge 最初在每个 4,096-shard 循环里重复计算大文件 hash，导致速度异常；该次运行在 publish 前停止，临时目录删除后改为缓存 immutable hash，再完成全量。第一次同步 region allowlist 时远端 `analysis/` 尚未创建，rsync 没有写入；创建精确目录并校验 hash 后重新同步。node70 没有 `rg`，早期 monitor 低估了完成数，改用 `grep` 后确认运行本身不受影响。

分析器最初只统计 top-level `cuda_out_of_memory`，漏掉 KernelGym 捕获在 metadata 中的 106 个 `torch.OutOfMemoryError`。修复后重跑全部 analysis，最终 `oom_count=1274`。自动生成的 AI 对照表还曾把只尝试 1,000 个 parent 的 canary 除以 53,896 的 full solver 分母；最终报告改为各自 within-run denominator，并明确两种合同不能比较 acceptance efficiency。两次修正都只影响测量口径，没有改变 child 或 H20 evidence。按要求没有为这些生产数据工具新增单元测试；一名独立 reviewer 在该要求重申前误跑了一次已有测试，17/17 通过，但这不作为本轮验收证据。验收依据是 `py_compile`、shell syntax、全量 artifact 重读、两个 exact shard checker 和真实样本人工复核。

## 产物

- `tools/data/synthesize/solve_multidim_shape_coverage.py`：双 slot deterministic solver，SHA-256 `04079b71e6f1c8f8b7037c19d9de931d9db76cd2e11ce5471db14d092dd7037a`。
- `tools/data/synthesize/launch_shape_solver_cpu_shards.sh`：四机 CPU shard launcher，SHA-256 `0199caf77ba7d8fc5848902333cb40a75ff3cf511a8b45e5634f5456cd708c46`。
- `tools/data/synthesize/merge_shape_solver_shards.py`：deterministic merge，SHA-256 `3347dc5482f0c68bcf3f32f1c399c6124425dece07aa33efb2831e248e0081f2`。
- `tools/data/synthesize/launch_shape_region_validation.sh`：四机 changed-region launcher，SHA-256 `b1fa357bea9a7978a97325826bb1c4cade31eae657c1b323f89c66fe96cdf29f`。
- `tools/data/synthesize/validate_shape_region_liveness.py`：region validator，SHA-256 `df6d75809e7f809ae9dcd73acb0bc7ce71ce5bc802622f0796d5328ef4d1352e`。
- `tools/data/synthesize/verify_shape_solver_runtime_shards.py`：reference/region exact shard checker，SHA-256 `3b44ceb04625f2c09552a88e44759cb62aa145f9362e71b350135cca88aa784f`。
- `tools/data/synthesize/analyze_shape_solver_run.py`：identity、runtime、OOM 和 bias analyzer，SHA-256 `1104184e8900aa19e4395c59b33ae2f1673f7cbedf2a298e94d10383618e2b30`。
- `Data/prompt_tvm_v4/shape_solver_multidim_v4_byte_targets_v1/run.full53896/selected.parquet`：53,896 个 solver parent，SHA-256 `01eeaa55e91ed92b44ee6de413d6c2c1bddb8794051e6fa79c4907ae98c67f07`。
- `Data/prompt_tvm_v4/shape_solver_multidim_v4_byte_targets_v1/run.full53896/targets.parquet`：一 parent 一 variant 的 deterministic target ledger，SHA-256 `7203ba5374223e73e64d51c5a864de5266579001f4934872966ab35b3bba1c7f`。
- `Data/prompt_tvm_v4/shape_solver_multidim_v4_byte_targets_v1/run.full53896/static/children.parquet`：17,353 个静态 review child，SHA-256 `39568d106db575da25e5df5f46f30460a78c2277874c3ad93acceca71a107030`。
- `Data/prompt_tvm_v4/shape_solver_multidim_v4_byte_targets_v1/run.full53896/static/paired.parquet`：34,706 行 paired reference 输入，SHA-256 `cdb12932dddee5cb995c5f83b5db5a7941f4ae0d9e8ba44f159a592348a872ba`。
- `Data/prompt_tvm_v4/shape_solver_multidim_v4_byte_targets_v1/run.full53896/static/manifest.json`：完整构造合同和逐 parent decision，SHA-256 `bb9b0a487ac98058056b2fff20991dbe900cc0c13f5815834eaf78d8409546ce`。
- `Data/prompt_tvm_v4/shape_solver_multidim_v4_byte_targets_v1/run.full53896/h20/reference/`：256 个 paired reference shard 与 SHA-bound scheduler evidence。
- `Data/prompt_tvm_v4/shape_solver_multidim_v4_byte_targets_v1/run.full53896/h20/region/`：256 个 changed-region shard 与 SHA-bound scheduler evidence。
- `Data/prompt_tvm_v4/shape_solver_multidim_v4_byte_targets_v1/run.full53896/analysis/region_reference_both_pass_uuids.txt`：14,390 个 region allowlist UUID，SHA-256 `ed04e6990db37d2a1cbccaeb1842aa42fbed8a1112ef1d3d448e5757b3eddff3`。
- `Data/prompt_tvm_v4/shape_solver_multidim_v4_byte_targets_v1/run.full53896/analysis/summary.json`：最终强校验与 runtime eligible UUID ledger，SHA-256 `c4c4bbee827d00665fdff2f48734f75abd5200985656be51c67d29d2e89257ef`。
- `Data/prompt_tvm_v4/shape_solver_multidim_v4_byte_targets_v1/run.full53896/analysis/shape_bias_audit.json`：static/runtime 分布与 balance audit，SHA-256 `e22ac4958328fb63009ffb304b6dd2fe5b3d8707e6e9598cab25881596f68a5d`。
- `Data/prompt_tvm_v4/shape_solver_multidim_v4_byte_targets_v1/run.full53896/analysis/review_samples.md`、`stratified_review_samples.md` 和 `changed_slots.tsv`：可人工审阅的真实 diff、失败样本和全量 changed occurrence。
