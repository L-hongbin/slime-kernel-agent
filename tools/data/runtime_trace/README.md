# Runtime 调用组件追踪

离线捕获整次 forward 的 kernel、受支持 cuBLAS GEMM、CUDA copy/fill 调用，以实际 global-memory 访问或成功 API 契约建立 buffer 区域与版本依赖，再做跨轮组件对应

自定义 kernel 保持 opaque；不构建指令/寄存器图，不使用源码相似度或节点数发 reward。调用参数中的指针只用于身份绑定，读写证据来自 memory-only NVBit 或明确 API 契约

## 捕获与提取

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

## 边界与证据

- 全 forward、默认所有 CTA；每 kernel 最多 1,048,576 条压缩 memory runs，截断和未知不会被视为完整足迹
- Warp 内仅对精确连续地址做无损区域压缩；保留读写、覆盖范围、alias 生命周期、输出 view 和版本
- 同 stream 与成功 event/device sync 建立顺序；诊断插桩自带的强制同步不进入程序依赖图
- 未知中间调用会使旧版本失效；后续完整覆盖可按区域恢复。无序写入、原子冲突、无法绑定的私有 storage 保留未知
- cuBLAS Sgemm/GemmEx 以公开逻辑矩阵契约覆盖，不宣称测到库内部每次物理访问；cuDNN/cuBLASLt 没有专门契约
- 扩展 launch 属性只记录数量；缺失或非零时配置未知。workspace 模式区分默认池、自定义区和显式禁用默认池；默认池模式的 `workspace_bytes=0` 仅表示未绑定自定义容量
- 多活跃 CUDA context、多个 launch 主机线程、CUDA graph capture 当前拒绝；未支持 launch/内存 API 显式记 unknown
- 采集会串行化执行，冷 forward 成本只用于诊断，不能作为训练性能分数

实际案例、成本、冻结批次和复现 manifest 见 [组件追踪报告](../../../handoffs/paper/runtime_component_tracking.md)
