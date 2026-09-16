# Runtime 调用组件追踪

离线捕获整次 forward 的 kernel、受支持 cuBLAS GEMM、CUDA copy/fill 调用，以实际 global-memory 访问或成功 API 契约建立 buffer 区域与版本依赖，再做跨轮组件对应

自定义 kernel 保持 opaque；不构建指令/寄存器图，不使用源码相似度或节点数发 reward。调用参数中的指针只用于身份绑定，读写证据来自 memory-only NVBit 或明确 API 契约

## 捕获与提取

可选 `RUNTIME_TRACE_RECORD_FORMAT=regions`：按调用以一 KiB tile 索引精确字节位图，聚合重复／广播读，再导出区间。`runs` 为默认顺序模式，保留线程归属；区域模式的读写交集仍需顺序诊断，不能自动确认原地入口值

`RUNTIME_TRACE_ORDERED_LAUNCHES=0,4` 可在 regions 模式的一次完整重放中，仅对指定 launch 使用顺序记录；同一个 CUfunction 的其它调用仍按区域记录。顺序记录使用额外的整个 forward 预算，最多 `min(CAPACITY, TOTAL_RECORDS)` 条，区域输出仍受原预算限制，实际值写入 `ordered_selection` 事件。KernelGym 适配器从第一张对齐图自动选择足迹完整的非原子原地调用，在剩余执行／报告时间内最多重试一次；只接受完整重放且输入、状态、输出与全部调用接口一致的新图，失败时保留第一张图

区域模式下 `RUNTIME_TRACE_CAPACITY` 须为二的幂，限制每调用 tile 表及导出临时数组；`RUNTIME_TRACE_TOTAL_RECORDS` 限制整个 forward 输出的合并区间数。真实大输入试验使用四百万容量，tile 表约 1.06 GiB，导出临时数组最多 128 MiB。该计数与 raw events 不可直接比较；`record_format` 事件中的 `format`、`capacity_scope`、`total_record_unit` 和 `region_table_bytes` 记录实际含义。调用参数和非 opaque 实现指纹独立于访存预算保留

需要外部 NVBit 1.8、CUDA toolkit 与 H20；构建产物默认进入 ignored `local_artifacts/component_tracking/coarse/build`

```bash
make -C tools/data/runtime_trace NVBIT_ROOT=/absolute/path/to/nvbit_release_x86_64
python -m tools.data.runtime_trace.check_runtime_trace
```

GPU 执行前检查占用，把源码、二进制、payload、配置同步到独立 node-local 目录并写 `manifest.json`，格式为相对路径到 SHA256 的映射。`replay` 启动时逐文件核对；不要在共享部署目录构建或覆盖服务

在 node-local bundle 根目录运行，每次使用新的输出目录与 trace prefix；同一 payload 必须先做不设置 `LD_PRELOAD`、不传 `--tracer` 的控制

```bash
CUDA_VISIBLE_DEVICES=0 LD_PRELOAD="$PWD/runtime_trace.so" \
  RUNTIME_TRACE_START_ENABLED=0 RUNTIME_TRACE_OUTPUT="$PWD/traces/sample" \
  RUNTIME_TRACE_CTAS=-1 RUNTIME_TRACE_CAPACITY=1048576 \
  python -m tools.data.runtime_trace.replay \
    --payload inputs/sample.json --manifest manifest.json \
    --tracer "$PWD/runtime_trace.so" --output results/sample

python -m tools.data.runtime_trace.extract \
  --trace traces/sample --allocations results/sample/allocations.json \
  --context context.json --output graphs/sample.json
```

Reference payload 包含 `reference_code`，可带校验用 `source_sha256`；candidate 另含 `custom_code`，需传 `--kernelgym-root` 指向既有隔离 backend 与其 source manifest。默认初始化 seed42、输入 seed17，保留原始 shape/dtype；reference 使用 train/no_grad，并在 forward 前重置 seed23，candidate 使用 eval/no_grad 且沿用输入初始化后的 RNG 状态

两条路径显式启用 matmul/cudnn TF32。`result.json` 记录 forward 前读取到的实际 backend 字段、seed、配置 hash 和输入/输出指纹；reference 另记所有 named parameter/buffer 的初始内容 hash。跨实现比较先核对这些字段，不能仅凭输入 hash 相同就比较输出或耗时

`context.json` 必须包含同一题目源码的 `task`、实际输入指纹 `input_signature`、运行配置 `environment_signature`。采集的 SASS 实现标识只在区域接口对应成立后用于区分实现；它不证明数学语义或创作来源

## 匹配与回溯

```python
from tools.data.runtime_trace.match import compare
result = compare(previous_graph, best_graph)
```

`lineage` 接受带 `turn/correct/score/graph` 的 JSON 列表，graph 可为相对文件路径；由调用方提供历史正确性与分数，最佳图缺失时不换成较差答案

```bash
python -m tools.data.runtime_trace.lineage --snapshots snapshots.json --output lineage.json
```

结果区分留存观测、配置变化、实现变化、局部对应、配置未知、重复调用歧义和无法一对一对应。完整但没有已观测数据效果的调用单列；结构候选不等于训练可用状态，不自动接 GRPO

匹配结果另外区分 `same_call_role_candidate` 与实现版本 `unchanged/revised/unknown`；身份比较包含 typed views、alias、输入来源、输出角色和相邻消费者。融合版作为新组件，后续同角色修订可以追踪至早期候选；下游接口变化仍可能打断对应。`lineage` 输出身份候选与严格实现留存两种来源，不自动改变训练奖励

## 边界与证据

- 全 forward、默认所有 CTA；每 kernel 最多 1,048,576 条压缩 memory runs，截断和未知不会被视为完整足迹
- runs 模式仅对精确连续地址做无损压缩；regions 模式保留逐字节读写集合与空洞，重复写／原子标记在 tile 内保守扩散，原地顺序明确未知
- 同 stream 与成功 event/device sync 建立顺序；诊断插桩自带的强制同步不进入程序依赖图
- 未知中间调用会使旧版本失效；后续完整覆盖可按区域恢复。无序写入、原子冲突、无法绑定的私有 storage 保留未知
- cuBLAS Sgemm/GemmEx 及两者的 StridedBatched API 以公开逻辑矩阵契约覆盖；batch stride 以元素为单位转换，按 operand 合并精确区域，保留 gap。负 stride／超预算返回未知；cuDNN/cuBLASLt 没有专门契约，库内部物理访存仍 opaque
- 扩展 launch 记录支持的无指针属性（如 cluster 维度及调度策略）；事件、指针、未知属性和不完整列表保留配置未知。workspace 模式区分默认池、自定义区和显式禁用默认池；默认池模式的 `workspace_bytes=0` 仅表示未绑定自定义容量
- 多活跃 CUDA context、多个 launch 主机线程、CUDA graph capture 当前拒绝；未支持 launch/内存 API 显式记 unknown
- 采集会串行化执行，冷 forward 成本只用于诊断，不能作为训练性能分数

## KernelGym 集成入口

`service_replay.py` 接收 KernelGym 在正常 correctness trial 记录的输入/RNG capsule，分别执行 control、trace 和 CPU report；不在正常计时进程 preload tracer。默认重建模型后逐项验证 tensor state 内容 hash，也支持完整 state 恢复。输入 archive 保留 storage alias，临时 storage registry 有上限与原子 checkpoint，失败进程可保留部分图

```bash
python -m tools.data.runtime_trace.export --native /absolute/path/runtime_trace.so --output /node-local/tracer-bundle
```

Bundle 包含逐文件 manifest，KernelGym 的 `KERNELGYM_RUNTIME_GRAPH_BUNDLE` 指向它。源码与操作方配置见 [KernelGym 集成说明](../../../../KernelGYM-component-integration/docs/design-doc/RUNTIME_COMPONENT_GRAPH.md)；部署由主 Agent 负责

服务请求 `record_format` 默认 regions，首次 trace 显式采用该值；区域表容量在进入子进程前检查为二次幂。服务设 `RUNTIME_TRACE_SKIP_VENDOR_MEMORY=1`，跳过 vendor 内部访存和 SASS 反汇编；受支持库的子调用通过 NVBit 导出 cubin，验证 CUDA ELF 区段和实际入口后，用完整二进制 SHA256＋入口定位＋launch 配置记录版本。库调用仍按父级组件计数，未知实现/ABI 保留 unknown

默认每次 trace 最多导出 16 个库模块，每个最多 64 MiB；实际限制记在 native config。模块卸载会使 handle 缓存失效，编号和总次数预算不重置。NVBit 1.8 的 dump 返回值只留作诊断，成功条件来自实际文件验证；文件缺失／截断／入口缺失均不能成为指纹。正常在线 cleanup 同时回收 runs 与 cubin，显式 retain_raw 才保留原始证据

访存满容量后计数饱和，`dropped_runs_exact=false` 表示丢弃量仅为下界；不能拿它计算精确采样覆盖率。`RUNTIME_TRACE_MEMORY_POLICY=interfaces` 保留调用清单和已支持库契约，自定义 kernel 访存未知

CPU 区域减法使用单调扫描，版本相交查询使用有序区间索引，避免 strided 大图退化成平方级扫描。超出 inline 限制时仅省略已不完整节点的足迹及其重复 version 明细；未决依赖保留计数、哈希和显式 unknown，调用单位及完整足迹全部保留。`check_extract_fast_paths` 验证与朴素算法逐项相等，`check_bounded_summary` 验证分奖分母及 unknown 份额不缩水

该路径与此前 coarse 冻结批次分别保留实际版本与证据，不混算旧全指令结果；它返回观测与缺口，不实现 scalar reward

首次了解采集和提图机制，见 [NVBit 提图教程](../../../handoffs/paper/runtime_graph_extraction.md)；奖励方法、实物结果和训练准备集中在 [组件奖励报告](../../../handoffs/paper/component_reward_training.md)
