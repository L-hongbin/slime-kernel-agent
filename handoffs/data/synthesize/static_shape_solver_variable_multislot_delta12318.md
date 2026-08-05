# 可变多 slot shape solver：12,318 条恢复集 handoff

## 摘要

这条 follow-up lane 处理上一轮 full static solver 中因严格双 slot / 固定 2 的幂模式而未生成 child、但放宽词法位置限制后仍有正 storage witness 的 12,318 个 parent。它不改动已完成的 `run.full53896`，也没有把任何新行并入训练数据。

已完成精确输入选择、2–5 logical slot solver、最终 128-parent static/FakeTensor canary，以及同一 canary 的 8-shard CPU 运行和 deterministic merge。12,318 条全量 CPU solver 也已完成 1,024/1,024 shard manifest：node69_slime rank 0 和 node70_dspark rank 1 各保留 512 个 node-local shard。两边结果尚未汇集和 merge，因此没有 full static 统计或 canonical `run.recoverable12318`；canary 与全量 H20 reference / changed-region 都未启动，`training_approved=false`。

| 阶段 | 状态 | 结果 |
| --- | --- | --- |
| 精确 delta 输入 | 已完成 | 12,318 parent；`Medium=6,151`，`Large=6,167` |
| 1,024-way 全量输入分片 | 已完成 | 只含输入与 global-index ledger，未运行 solver |
| 最终 128-parent static canary | 已完成 | 121 child，94.53% parent coverage |
| 8-shard CPU + merge canary | 已完成 | 四个 parquet 与非分片 canary 逐字节一致 |
| Canary H20 reference / region | 未启动 | 没有 runtime evidence |
| 12,318 条全量 CPU solver | 已完成分片 | 1,024/1,024 manifest；node69/node70 各 512，等待汇集 |
| 全量 CPU merge / analysis | 未启动 | 没有 canonical merged run 或 full static count |
| 全量 H20 reference / region | 未启动 | 没有 runtime-eligible child |

## 精确输入与来源约束

输入来自已完成的 `Data/prompt_tvm_v4/shape_solver_multidim_v4_byte_targets_v1/run.full53896`。selector 只考察旧 manifest 中 reason 为 `no_strict_two_slot_exactly_one_power_solution`、至少有两个 affine slot 且旧 strict bilinear profile 数为 0 的 13,536 个 decision。它只移除“patch token 必须物理位于 `get_inputs()` 函数体内”这一条词法限制；slot span 不重叠、同一非空 factory set、每个 slot 在每个 factory 恰好出现一次、axis 互异、occurrence 数相同以及正的 exact bilinear storage witness 都仍需成立。

重新证明后，12,318 个 parent 有正 witness，另 1,218 个为 `no_positive_profile`。selector 校验旧 selected、selection、static manifest、UUID、reference hash 和依赖源码 hash 后才原子发布结果；该阶段只选 parent，不生成 child。

| 产物 | SHA-256 |
| --- | --- |
| `input.recoverable12318/selected.parquet` | `fc780d5ce11124c732b5edb9b965ceb9f190d43206cd8635ded1fe0a9b1cca1a` |
| `input.recoverable12318/selection.json` | `7f4a83f974c728d1b2763ef6136a2db8fee8bb31be6955c43fcce27fd0633e63` |
| `input.recoverable12318/selection_ledger.tsv` | `1eea28ef2b9a50eabf7f0736be8ca6adbd2d078ed6b093c4058a4f5f82b6a108` |
| 旧 full-run `selected.parquet` | `01eeaa55e91ed92b44ee6de413d6c2c1bddb8794051e6fa79c4907ae98c67f07` |
| 旧 full-run `selection.json` | `e82375a2095bf1a82a0df2bd96676b40ce2bab8d005fe892b8e32bf2f482de34` |
| 旧 full-run `static/manifest.json` | `bb9b0a487ac98058056b2fff20991dbe900cc0c13f5815834eaf78d8409546ce` |

全量 modulo 输入已预先拆成 1,024 shards：`Data/prompt_tvm_v4/shape_solver_variable_multislot_v5/shards.1024`。其 `shards.json` SHA-256 为 `5e5ba62394c42a8fb2260aeddb59dd030b4fd2674d27a944d29805773249d81d`，记录的 source selected / selection hash 分别与上表的 `fc780...` / `7f4a...` 一致。

## 新的 2–5 slot 构造合同

solver contract 为 `shape_variable_multislot_solver_v5`，generator 为 `same_factory_product_variable_2_to_5_soft_p2_50_v3`。每个 parent 仍只复用旧 lane 已分配的一个 byte-granularity target：`Medium` 为 64–256 MiB，`Large` 为 `[256 MiB + 1 byte, 4 GiB]`；child 至少为 parent input bytes 的 2 倍，target 相对误差上限为 25%。

每个候选改 2–5 个 logical shape slot，不再要求 exact-2。一个 linked slot 即使同步出现在多个 input factory 中，logical count 仍只计一次。所有 slot 必须满足：

- 作用于同一个非空 direct-input factory set；
- 在每个 factory 中各出现一次且位于互异 axis；
- patch spans 两两不重叠，新值都严格大于 parent 值；
- module-scope linked name 只有在所有 load 都被证明是 direct input-shape dimension 时才允许；
- 精确 storage 形式为 `constant_bytes + product_coefficient * product(values)`；
- source-span replay 后 `Model`、`get_init_inputs()`、factory 顺序、rank 和未声明 span 保持不变；
- parent 和 child 都通过 FakeTensor；每个最终 rank≥2 direct factory 满足 `largest <= 1000 × second_largest`。

候选优先较大的 structural group；在最大 cardinality 上最多先尝试三个不同候选，随后才尝试较小 cardinality，单 parent 最多进行六次 child FakeTensor 尝试。因此“尽量更多维度”是结构和门禁允许下的 best effort，不是强行把每个 child 做成 3–5 slot。

对 12,318 个输入按当前 group enumerator 做的最大兼容 cardinality 预检如下。7,644 个 parent 的合同内结构上限就是两个 slot；若要让它们也改三个以上维度，必须放宽当前 same-factory/product 证明或允许语义性代码修改，不属于本 lane。

| 最大兼容 logical slots | Parent | 占比 |
| ---: | ---: | ---: |
| 2 | 7,644 | 62.06% |
| 3 | 2,876 | 23.35% |
| 4 | 1,790 | 14.53% |
| 5 | 8 | 0.06% |
| 3–5 合计 | 4,674 | 37.94% |

2 的幂不再有 per-child hard constraint。每个 logical slot 用 `parent_uuid + logical_slot_id` 独立、确定性地得到 50% soft preference；candidate ranking 先比较 0.1% target-error bucket，再比较 preference mismatch，避免为了吸附到 2 的幂而显著牺牲随机 byte target。最终门禁只看 changed factory-axis occurrence 的全局 2 的幂占比是否在闭区间 30%–50%，而且 H20 筛选后必须重新统计；static 通过不保证 runtime 子集仍通过。

## 最终 128-parent canary

权威 canary 是 `Data/prompt_tvm_v4/shape_solver_variable_multislot_v5/run.canary128.v4`。输入为精确 delta 集合的前 128 行，`Medium=60`、`Large=68`；parent FakeTensor 为 128/128，static/FakeTensor 接受 121 个 child，其中 `Medium=56`、`Large=65`。七个失败全部是 child FakeTensor incompatibility，包括 convolution channel 约束、broadcast、fixed reshape/split 和 pixel-shuffle shape 约束；它们不是 H20 结果。

| 接受 child 的 logical slots | Child | 占 121 个 child |
| ---: | ---: | ---: |
| 2 | 71 | 58.68% |
| 3 | 32 | 26.45% |
| 4 | 17 | 14.05% |
| 5 | 1 | 0.83% |
| 3–5 合计 | 50 | 41.32% |

两个 child 在最大 cardinality 的候选未通过后降到较小 group；其余 119 个接受 child 使用其最大兼容 group。独立 analyzer 重放了 121 个 child 的精确 source spans，并重新解析 factory / axis diff、`Model` 和 `get_init_inputs()` AST、hash、lineage、FakeTensor 及 1000:1 balance；invalid、missing、orphan 和 balance violation 都为 0。

target 相对误差 P50 为 0.0330%，P90 为 0.0831%，最大为 1.5483%。121 个 child 共 328 个 changed factory-axis occurrences，其中 113 个为 2 的幂，占 34.45%；static 30%–50% 门禁通过。新值有 206 个 distinct value，effective values 为 140.44，Top-5 占 18.60%；相比旧 exact-2 lane 的局部固定 50%，此 canary 没有固定 per-child 模式。

下表按 logical slot 统计一个 child 有多少个新值是 2 的幂。可见相同 cardinality 下同时存在 0、1、2 或 3 个，而不是“恰好一个是 2 的幂、另一个不是”。

| Logical slots | 0 个 P2 | 1 个 P2 | 2 个 P2 | 3 个 P2 |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 26 | 45 | 0 | 0 |
| 3 | 7 | 16 | 9 | 0 |
| 4 | 2 | 7 | 5 | 3 |
| 5 | 0 | 0 | 1 | 0 |

### 人工复核样本

人工读取的代表性 diff 都只改 module-scope shape integer；没有修改 docstring、程序结构、factory、dtype 或 rank。原源码已有的 shape 注释可能随新数值变旧，但 solver 不改注释文本。

| Child / parent | Logical slots | Parent shape → child shape | P2 新值 | Target 误差 |
| --- | ---: | --- | ---: | ---: |
| `shapesolver_81db59832d86fab765e3e702` / `cuda_agent_ops_6764919bc05ec00942ec` | 5 | `[2,8,512,32,32] → [5,16,1897,86,64]` | 2 | 0.0648% |
| `shapesolver_e1b5bf1d83a057e23cf86eb3` / `cuda_llm_201824` | 4 | `[64,128,32,32] → [128,512,64,132]` | 3 | 0.0795% |
| `shapesolver_2470d5c8c9ddfb488971c3aa` / `cuda_llm_653385` | 3 | `[16,256,256] → [29,481,517]` | 0 | 0.0767% |
| `shapesolver_c1cd6167c665eeef8e9f0a30` / `cuda_llm_789527` | 2 | `[128,512] → [28643,32768]` | 1 | 0.0080% |

共同可审阅证据位于 canary 的 `analysis/summary.md`、`analysis/shape_bias_audit.md`、`analysis/changed_slots.tsv`、`analysis/review_samples.md` 和 `analysis/stratified_review_samples.md`。

## 8-shard CPU / merge 验证

128 行输入还完成了 8-way modulo shard 集成：

- shard input：`shards.canary128.v4.8`，`shards.json` SHA-256 `444d2c746cf1bc353442cb50a17f055fabd14ad2ba81f4786288cddb905d14dd`；
- shard runs：`run.canary128.v4.sharded8`，8/8 manifest 完成；
- merge output：`run.canary128.v4.merged8`，contract 为 `shape_solver_shard_merge_v2`。

合并后 `selected`、`targets`、`children` 和 `paired` 与非分片 `run.canary128.v4` 逐字节相同：

| Artifact | 两个 run 的共同 SHA-256 |
| --- | --- |
| `selected.parquet` | `4a662416213d7075fb448aa8f0fafafd545ed27d64775547d6c1c981c4503ee6` |
| `targets.parquet` | `d49ef464d398e21885c79d345010bebd1fa71d06fafc7677e28c6516ba48caac` |
| `static/children.parquet` | `39e9c6f17ab626c2a3aea2ae0e6d78570116de6ba245b323bd08d7502a69bfca` |
| `static/paired.parquet` | `3a074c75c4e5af15810fc8849ee9a0bfe65c1b2d682571c89630cc1c747c1d3c` |

去除 merge 新增的 `shard_index` / `shard_source_row_index` provenance 后，两边的 128 个 decision 也一致；counts 和 skip reasons 一致。两个 manifest 本身不应字节相同，因为 merged manifest 额外绑定八个 shard manifest、merge tool、输入 shard ledger，并把 p2 observation scope 写为 globally merged static children。

## 已验证与未验证边界

当前证据只证明：精确 delta membership、确定性 target 复用、2–5 slot structural/storage 合同、源码改动边界、parent/child FakeTensor、自独立重解析的 identity 与 1000:1 balance、static shape-value audit，以及 8-shard merge 的 deterministic equivalence。

当前证据没有证明：

- 121 个 canary child 或未来全量 child 能在 H20 上通过 production KernelGym reference；
- changed-region 对 output 有效，或不存在扩展后才产生的 dead tail；
- runtime-eligible 子集仍满足 30%–50% p2 与其他 bias 门禁；
- 12,318 个 parent 的全量 static 接受数、source/operator bias 或耗时；
- 任何 CUDA solution correctness、speedup 或训练收益。

因此不得把 canary 的 121 个 child 或未来 static merge 直接接入训练。只有完整 paired reference、changed-region、exact shard checker 和 `--require-runtime-quality` analyzer 都通过后，才可形成 runtime review 集；即使如此，训练仍需另行批准。

## 后续汇集与验证顺序

### 1. 后续运行前的资源和一致性检查

继续 H20 阶段前先核对同一 commit 的工具、`input.recoverable12318`、`shards.1024` 和 merged artifact hash。曾观察到 node53 上有一套与本 lane 无关的 DeepSeek-V4-Flash TP8 vLLM 服务，占用约 93 GiB/卡；该状态可能已经变化。启动 H20 前必须在每台候选节点检查 `nvidia-smi`、compute process、CPU load、磁盘和 validation lock，不要假设 node53 或其他 H20 仍空闲。

H20 launcher 会 fail-closed 检查可见 GPU idle 状态。一次 reference 或 region run 的 machine count / rank mapping 会写入 scheduler contract，启动后不能在同一证据目录中改 machine count 续跑。如果四台都空闲，可使用 4 台；如果 node53 或用户的新扩展任务仍在使用 GPU，可选其余空闲节点并重新连续编号，但要在启动前固定 mapping。

### 2. 先补最终 canary 的 H20 门禁

在一台确认空闲、可见 8 张 H20 的 KernelGym 容器内运行：

```bash
canary_run=Data/prompt_tvm_v4/shape_solver_variable_multislot_v5/run.canary128.v4.merged8

SYNTH_MACHINE_COUNT=1 \
SYNTH_GPUS_PER_MACHINE=8 \
SYNTH_VIRTUAL_SHARDS_PER_GPU=8 \
SYNTH_TIMEOUT_SECONDS=180 \
SYNTH_EXPECTED_MODE_CLASS=any \
SYNTH_MAX_DEVICE_MEMORY_GIB=64 \
SYNTH_EXPECTED_INPUT_SHA256=3a074c75c4e5af15810fc8849ee9a0bfe65c1b2d682571c89630cc1c747c1d3c \
bash tools/data/synthesize/launch_reference_validation_shards.sh \
  0 "${canary_run}/static/paired.parquet" "${canary_run}/h20/reference"

python -m tools.data.synthesize.verify_shape_solver_runtime_shards \
  "${canary_run}" reference
python -m tools.data.synthesize.analyze_shape_solver_run "${canary_run}"

SHAPE_REGION_MACHINE_COUNT=1 \
SHAPE_REGION_GPUS_PER_MACHINE=8 \
SHAPE_REGION_VIRTUAL_SHARDS_PER_GPU=8 \
SHAPE_REGION_TIMEOUT_SECONDS=600 \
SHAPE_REGION_TRIALS=3 \
SHAPE_REGION_SEED=17 \
bash tools/data/synthesize/launch_shape_region_validation.sh 0 "${canary_run}"

python -m tools.data.synthesize.verify_shape_solver_runtime_shards \
  "${canary_run}" region
python -m tools.data.synthesize.analyze_shape_solver_run \
  "${canary_run}" --require-runtime-quality
```

reference verify 后的第一次 analyzer 会生成 region allowlist。若跨节点运行，必须把 canonical run artifacts、完整 reference evidence 和生成的 allowlist 同步并逐文件校验后再启 region。

### 3. 汇集并合并 12,318 条全量 CPU shards

全量采用已存在的 1,024 个 modulo shards，并已按以下固定参数执行完毕：

| 节点 | Rank | Shard ownership | 完成 manifest |
| --- | ---: | --- | ---: |
| node69_slime | 0 | 偶数 shard | 512 |
| node70_dspark | 1 | 奇数 shard | 512 |

launcher 使用 `SHAPE_CPU_MACHINE_COUNT=2`、`SHAPE_CPU_WORKERS=64` 和 `SHAPE_FAKE_TIMEOUT_SECONDS=30`。两个节点没有残留 solver process；任务正常跑完，没有被停止。输出位于各节点同名的 node-local 路径 `Data/prompt_tvm_v4/shape_solver_variable_multislot_v5/run.recoverable12318.sharded1024`。不要重新运行或覆盖这些目录。

下一步只把 node69 的偶数 shard、node70 的奇数 shard 和各自日志汇集到一个 canonical shard-run root。汇集前后都要核对 1,024 个 manifest、shard ownership、artifact SHA 和 solver source SHA。下面的启动命令只用于说明本轮已使用的配置，不应再次执行：

```bash
machine_rank=0  # node70 使用 1

SHAPE_CPU_MACHINE_COUNT=2 \
SHAPE_CPU_WORKERS=64 \
SHAPE_FAKE_TIMEOUT_SECONDS=30 \
bash tools/data/synthesize/launch_variable_shape_solver_cpu_shards.sh \
  "${machine_rank}" \
  Data/prompt_tvm_v4/shape_solver_variable_multislot_v5/shards.1024 \
  Data/prompt_tvm_v4/shape_solver_variable_multislot_v5/run.recoverable12318.sharded1024
```

专用 wrapper 固定 module 为 `tools.data.synthesize.solve_variable_shape_delta`，并在运行和 resume 时要求 manifest contract 为 `shape_variable_multislot_solver_v5`，防止误用旧 exact-2 solver。两台的 rank-owned shards 完成汇集后再 merge：

```bash
python -m tools.data.synthesize.merge_shape_solver_shards \
  Data/prompt_tvm_v4/shape_solver_variable_multislot_v5/run.recoverable12318 \
  --shard-input-root \
    Data/prompt_tvm_v4/shape_solver_variable_multislot_v5/shards.1024 \
  --shard-run-root \
    Data/prompt_tvm_v4/shape_solver_variable_multislot_v5/run.recoverable12318.sharded1024 \
  --source-selected \
    Data/prompt_tvm_v4/shape_solver_variable_multislot_v5/input.recoverable12318/selected.parquet

python -m tools.data.synthesize.analyze_shape_solver_run \
  Data/prompt_tvm_v4/shape_solver_variable_multislot_v5/run.recoverable12318
```

merge 会拒绝缺 shard、global index 重复、source row/hash 不匹配、solver contract/source hash 漂移和 artifact hash 错误。static analyzer 通过后仍需人工读取 source/operator/cardinality、target 尾部、p2/value concentration、fallback 和失败样本；不能把 128 canary 的 94.53% 接受率外推成全量结论。

### 4. 全量 H20 reference 与 region

全量 H20 命令沿用 canary 的两阶段顺序，把 `canary_run` 换成 `run.recoverable12318`，并把 `SYNTH_MACHINE_COUNT` / `SHAPE_REGION_MACHINE_COUNT` 设为实际锁定的空闲节点数。reference 启动前计算 full `static/paired.parquet` SHA-256，并在所有节点用同一个 `SYNTH_EXPECTED_INPUT_SHA256`；各节点使用连续且唯一的 machine rank。reference evidence 汇集并通过 exact checker 和 analyzer 后，再同步 analyzer 生成的 allowlist，启动 region，最后执行：

```bash
python -m tools.data.synthesize.verify_shape_solver_runtime_shards \
  Data/prompt_tvm_v4/shape_solver_variable_multislot_v5/run.recoverable12318 region
python -m tools.data.synthesize.analyze_shape_solver_run \
  Data/prompt_tvm_v4/shape_solver_variable_multislot_v5/run.recoverable12318 \
  --require-runtime-quality
```

只有这里的 runtime analyzer 才能给出最终 child 数与 runtime bias。不要复用旧 exact-2 lane 的 H20 allowlist、counts 或环境结论。

## 当前源码绑定

| 文件 | SHA-256 |
| --- | --- |
| `prepare_variable_shape_delta.py` | `b1cb3ef5ad3fe0cb1a5f18d40c438285a5232819d4408bfdbe2d3b1072150aa3` |
| `solve_variable_shape_delta.py` | `0ef18ff7650fee1680ee52d3b9694261ae104357565b5db37d267a9ed62d67ea` |
| `solve_multidim_shape_coverage.py` | `04079b71e6f1c8f8b7037c19d9de931d9db76cd2e11ce5471db14d092dd7037a` |
| `solve_shape_coverage.py` | `09bb44a956f3f7cab2a19469df4407baef5909c623f4bf9e9513cc71813bada0` |
| `ai_shape_coverage.py` | `5d8214ceca39dc88db0518c7d90a49120e86638cada61c2cebcd91b160f7c132` |
| `augment_prompt_tasks.py` | `ee5514948a46815f00378085a80935d1cf85068b78333925b35882b4e4626857` |
| `launch_variable_shape_solver_cpu_shards.sh` | `305f1ec6a9646c79491063314e2abd3f0d400598e5551625fefc1f3fa1c9aff1` |
| `launch_shape_solver_cpu_shards.sh` | `8d002ed80b8f363a2ae27204f2af8a0cb7c04dea5d039c9297db54dd6e8d2086` |
| `shard_shape_solver_input.py` | `5896d44dfa2f425ec134912d8701a40363951341312a269e1b6c180b488e4395` |
| `merge_shape_solver_shards.py` | `f35a0ef34354943a9730ca607376034d5f7cacc4754b0731d8f81543f5e922be` |
| `analyze_shape_solver_run.py` | `3636a8f0b720b2d51902fd13f5886fc6babfb1f047603979d7cdfc573c72f133` |
| `validate_train_mode_contract.py` | `691b626236b55e1b330534874a73d8f37e0a559fc35c9e788f4a2855981936fd` |
| `validate_shape_region_liveness.py` | `cd4051e424286c3d8f60b1759daba7fc43b9655c94412b7be425d1d8c4c2536a` |
| `verify_shape_solver_runtime_shards.py` | `3b44ceb04625f2c09552a88e44759cb62aa145f9362e71b350135cca88aa784f` |
| `launch_reference_validation_shards.sh` | `da63f011263caa0114d754090338304b57283002e814092b9de7ba63d46a4a38` |
| `launch_shape_region_validation.sh` | `b1fa357bea9a7978a97325826bb1c4cade31eae657c1b323f89c66fe96cdf29f` |

按要求没有为这些生产数据工具新增单元测试。发布 source-bound checkpoint 时保留了上表已被 canary manifest 绑定的源码字节，没有应用会整体改变源码 SHA-256 的 Black/isort/Ruff 自动改写；`py_compile`、四个 launcher 的 `bash -n`、关键 CLI `--help` 和真实 canary/merge 证据均已通过。后续若格式化或修改任一 source-bound 文件，必须从相应 selection/static/runtime 阶段重新生成证据，不能只改 manifest hash。

本 handoff 记录的是 full CPU shard 完成点和后续可恢复边界，不是 runtime 完成报告；`training_approved=false`。
