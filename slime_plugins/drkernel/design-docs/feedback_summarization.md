# Multi-turn Feedback Summarization

## 目标

让 `format_kernelgym_feedback` 在把 KernelGym `/evaluate` 响应渲染进下一轮 tool_response 之前，先做一道**字段白名单 + 错误文本摘要**的压缩，让 feedback 在模型多轮 prompt 里只占一个可预算的小份额。

## 非目标

- 不动 KernelGym server（`verbose_errors=True` 保留），server 仍返完整 stderr。
- 不动 prompt 模板，`{{ feedback }}` 占位符不变。
- 不引入新的多轮 driver 概念。
- 不修 `problem_id=87` 的 `reference_runtime=-1` worker OOM 问题（server 侧的事）。

## 一、tool_response 的字符预算应当是多少（first principles）

每个 multi-turn 轮的 prompt 大概结构是：

```
[ T0 system + first user prompt ]  ~ 3-6 KB
[ T0 assistant ]                    模型自由生成，典型 5-15 KB（含 thinking + 三段式代码）
[ T1 tool_response ]                ← 本文档要 budget 的对象
[ T1 assistant ]                    再一次 5-15 KB
[ T2 tool_response ]                ← budget
[ T2 assistant ]                    最后一次生成，需要剩下的 context
```

模型 context 假设 64K tokens（约 256 KB 字符级，4:1）。需要给最终 assistant generation 留至少 ~30K tokens（120 KB）让模型完成 thinking + 完整 kernel 代码。剩下的 prompt 全部累计约 130 KB。其中：

- 前 2 个 assistant turn ≈ 20-30 KB
- T0 prompt ≈ 6 KB
- 留给 T1 + T2 tool_response 总计 **~5-10 KB**，单条 tool_response **2-5 KB**

下界：单条 tool_response 至少要装下评估信号（compile / correct / runtime 等结构化字段约 0.2 KB）+ 错误摘要（够 5-10 行 nvcc error 信息约 1-1.5 KB）+ 轻量 metadata（约 0.2 KB）= **~1.5-2 KB 是合理下限**。

上界：超过 ~3 KB 之后，重复 nvcc 命令行 / stack frame 对模型修 bug 没有边际收益，只会挤压后续 generation 空间和稀释关键信号。

**结论**：单条 tool_response 目标在 **1-2.5 KB 字符级**。错误摘要内嵌部分上限设为 **1600 字符**（约 400 tokens），能装下大部分实际场景需要的 error 信号同时给非错误字段留余量。1600 字符做默认值，CLI 可配。

## 二、应该保留 KernelGym 返回的哪些字段

KernelGym `/evaluate` 返回 dict 顶层有 `env_state`、`reward_extra_info` 等。`env_state` 是评估主体，逐字段过一遍：

### 顶层字段

| 字段 | 示例值 | 决策 | 理由 |
|---|---|---|---|
| `task_id` | `"parallel_task_000484_aa8c66eb"` | **去掉** | UUID，模型读不出语义；只对 KernelGym 内部 trace 有用 |
| `status` | `"failed"` / `"completed"` / `"timeout"` | **保留** | 模型读一个词就知道发生了什么 |
| `compiled` | `false` | **保留** | 关键评估信号 |
| `correctness` | `false` | **保留** | 关键评估信号 |
| `decoy_kernel` | `false` | **保留** | 反作弊信号，模型读了知道是不是被判走捷径 |
| `reference_runtime` | `5.25` (ms) | **保留** | baseline 性能，模型知道目标值 |
| `kernel_runtime` | `1.91` (ms) | **保留** | 自己 kernel 的实际耗时 |
| `speedup` | `2.74` | **保留** | 直接对比指标，模型不必自己除 |
| `reward` | `0.5` / `-0.5` | **去掉** | KernelGym 的训练用 scalar，**对模型决策没增量信息**——`compiled / correctness / speedup` 已经全告诉模型了 |
| `success` | `false` | **去掉** | `compiled AND correctness AND NOT decoy_kernel` 的派生值，**冗余** |
| `error_code` | `"COMPILATION_ERROR"` | **保留** | 一个词的错误归类，比 `status` 更细 |
| `error_message` | `"Task processing failed: RuntimeError: ..."`（短）或几十 KB（verbose） | **保留并摘要** | summarize 走 1600 chars 上限 |

### `metadata` 子 dict 字段

`env_state["metadata"]` 里 KernelGym 塞了一堆 server 内部追踪 / profiling 字段。逐字段：

| 字段 | 示例值 | 决策 | 理由 |
|---|---|---|---|
| `error` | 整段 nvcc stderr（几十 KB） | **去掉** | **只作为 summarize 的输入兜底**——它本身不进 payload，被压缩成 `error_message` 字段后挂出去 |
| `compilation_error` | 同 stderr 子集 | **去掉**（同上，作为 summarize 输入） |
| `runtime_error` | 同上 | **去掉**（同上） |
| `correctness_issue` | 同上 | **去掉**（同上） |
| `hardware` | `"NVIDIA GeForce RTX 4090"` | **保留** | 模型偶尔需要知道 GPU arch（compute capability 影响 instruction set） |
| `gpu_name` | `"NVIDIA GeForce RTX 4090"` | **去掉** | 跟 `hardware` 重复 |
| `device` | `"cuda:1"` | **去掉** | 模型不关心 cuda:0 还是 cuda:1 |
| `backend` | `"tvm_ffi"` | **去掉** | 模型已经从 first-turn prompt 里知道是哪种 backend |
| `compilation_error_name` | `"compile_error"` / `"ninja_link_error"` 等 | **保留** | 更细的错误分类，几个字符 |
| `runtime_error_name` | `"builtins.RuntimeError"` / `"torch.AcceleratorError"` 等 | **保留** | 异常类全名，帮模型快速识别错误类型 |
| `correctness_issue_name` | `"Output mismatch"` | **保留** | 正确性失败的简短归类 |
| `max_difference` | `0.12` | **保留** | 输出最大数值偏差，模型可据此判断是不是数值精度问题 |
| `avg_difference` | `0.003` | **保留** | 输出平均偏差 |
| `kg_stage_*` (`kg_stage_completed_s`, `kg_stage_current`, `kg_stage_metadata_path`, ...) | timing trace dict | **去掉** | KernelGym 内部 profiling，模型无法据此改代码 |
| `tm_*` (`tm_enter_monotonic_ns`, `tm_status_hset_s`, ...) | 纳秒/毫秒 timing | **去掉** | KernelGym 服务端任务管理 timing，模型不需要 |

### `metrics` 字段（KernelGym kernel 成功跑起来后才有）

模型如果想"接着优化"，需要知道 kernel 占多少时间、覆盖了多少计算。这一组指标在 kernel 跑起来时才有意义。字段名按 KernelGym 实际返回 schema（注意 `num_custom_kernels` 是复数、`custom_kernel_cuda_time_coverage` 是带前缀的全名）：

| 字段 | 示例值 | 决策 | 理由 |
|---|---|---|---|
| `custom_kernel_cuda_time_coverage` | `"Custom kernel CUDA time: 218103.85us / Total: 218103.85us, Coverage: 100.00%"` | **保留** | 一行人类可读总结，告诉模型 custom kernel 占了总 GPU 时间的百分比 |
| `num_custom_kernels` | `3`（注意复数） | **保留** | 模型生成的 custom kernel 实际被调用了几个 |
| `num_total_kernels` | `12` | **保留** | 总 kernel 调用数，配合算覆盖率 |
| `custom_kernel_cuda_time_in_profiling_us` | `218103.85` (μs) | **保留** | custom kernel 的 CUDA 时间 |
| `total_kernel_cuda_time_in_profiling_us` | `218103.85` (μs) | **保留** | 总 CUDA 时间 |
| `total_kernel_run_time_in_profiling_us` | `218103.85` (μs) | **去掉** | 跟 cuda_time 在简单 case 下几乎一样，省字符 |
| `total_kernel_run_time_in_profiling_us_cpu_cuda` | `218103.85` (μs) | **去掉** | 完全冗余 |
| `num_coverage` | `1` | **去掉** | 跟 `num_custom_kernels` 含义模糊重叠，没明确语义价值 |

**性能指标只在 `compiled AND correctness == True` 时才挂出去**，失败 case 不显示这些（值都是 -1 / 0，无意义）。

### 不出现的字段

- `reward_extra_info`：KernelGym 返回顶层的另一个字典，目前看跟 `env_state` 内容大量重叠且没有独有信号，去掉。

## 三、改后 feedback 长什么样

以 nvcc 编译失败为例，新 `format_kernelgym_feedback` 输出大概几百字节：

```
{"status": "failed", "compiled": false, "correctness": false, "decoy_kernel": false,
 "reference_runtime": -1.0, "kernel_runtime": -1.0, "speedup": 0.0,
 "error_code": "COMPILATION_ERROR",
 "error_message": ".../generated.cu(39): error: a value of type \"cublasStatus_t\" cannot be used ...\n.../generated.cu(40): error: a value of type \"cublasStatus_t\" cannot be used ...\n.../generated.cu(106): error: identifier \"CUDNN_TENSOR_NCDHW\" is undefined\n.../generated.cu(106): error: identifier \"cudnnSetTensor5dDescriptor\" is undefined",
 "metadata": {"hardware": "NVIDIA GeForce RTX 4090", "compilation_error_name": "compile_error"}}
```

`compiled+correctness=True` 时附带 metrics。下面两条都是从 `eval_0.pt` 实测样本经过新 `build_prompt_feedback_payload` 输出的 JSON（单行级别 511 / 518 chars）。

**(a) 正确且略快于 baseline（speedup ≥ 1.0）**：

```
{"status": "completed", "compiled": true, "correctness": true, "decoy_kernel": false,
 "reference_runtime": 9.39, "kernel_runtime": 9.36, "speedup": 1.0032051282051284,
 "metrics": {
   "custom_kernel_cuda_time_coverage": "Custom kernel CUDA time: 93174.91us / Total CUDA time: 93174.91us, Coverage: 100.00%",
   "num_custom_kernels": 1, "num_total_kernels": 1,
   "custom_kernel_cuda_time_in_profiling_us": 93174.91,
   "total_kernel_cuda_time_in_profiling_us": 93174.91},
 "metadata": {"hardware": "NVIDIA GeForce RTX 4090"}}
```

模型读这条能立刻判断："compiled+correct，但仅刚刚好与 baseline 持平（speedup≈1.003），custom kernel 占了 100% 的 CUDA 时间——继续优化空间还有，下一轮可以尝试更激进的 tile / 共享内存策略"。

**(b) 正确但比 baseline 慢（speedup < 1）**：

```
{"status": "completed", "compiled": true, "correctness": true, "decoy_kernel": false,
 "reference_runtime": 2.65, "kernel_runtime": 19.3, "speedup": 0.13730569948186527,
 "metrics": {
   "custom_kernel_cuda_time_coverage": "Custom kernel CUDA time: 192841.79us / Total CUDA time: 192841.79us, Coverage: 100.00%",
   "num_custom_kernels": 1, "num_total_kernels": 1,
   "custom_kernel_cuda_time_in_profiling_us": 192841.789,
   "total_kernel_cuda_time_in_profiling_us": 192841.789},
 "metadata": {"hardware": "NVIDIA GeForce RTX 4090"}}
```

模型读这条能立刻判断："compiled+correct 但比 baseline 慢了 7×，肯定有性能 bug——下一轮需要从根本上重新设计，而不是微调"。

注意几个共同特点：
- **无 `error_message` 字段**——正确性通过时没有错误内容，summarize 也就没产出
- **`error_code: null`** 被白名单过滤掉（值为 None 不挂出去）
- `metadata` 只剩 `hardware` 一项，其它 `gpu_name / device / backend` 全部 drop（drop 是设计明文）
- 总长 ~500 chars，远低于失败 case 时的 ~800-1500 chars（失败 case 多带 `error_message`）

**(c) 正确但 perf 阶段 OOM（基础设施问题，非模型 bug）**：

KernelGym worker（24GB RTX 4090）在 perf 阶段做 `num_warmup=30 + num_perf_trials=50` 次连续 forward + torch profiler buffer + reference model perf 同步跑，对部分大 problem（如 1×1 conv with large channels）显存会撑爆。这种 case correctness 已通过、kernel 本身没错，但 perf 测量失败、`speedup=0 / reference_runtime=-1`。

```
{"status": "completed", "compiled": true, "correctness": true, "decoy_kernel": false,
 "reference_runtime": -1.0, "kernel_runtime": 599.0, "speedup": 0.0,
 "error_message": "CUDA out of memory. Tried to allocate 8.00 GiB. GPU 5 has a total capacity of 23.52 GiB of which 6.62 GiB is free. Including non-PyTorch memory, this process has 16.44 GiB memory in use. Process 2165460 has 448.00 MiB memory in use.",
 "metrics": {...},
 "metadata": {"hardware": "NVIDIA GeForce RTX 4090"}}
```

模型读这条能立刻判断："correctness 过了但 evaluation 基础设施 OOM——不是我代码问题，下一轮不必动 kernel"。这条信号靠 `metadata.error_during_performance` 字段进入 `_ENV_STATE_PRIMARY_ERROR_SOURCES`，summarize 走 `CUDA out of memory[^\n]*` 抓 OOM 头一句、然后截掉 `"See documentation for Memory Management..."` 引用尾巴。

`problem_id=87` 在 27B 100×8×3 turn 跑里 9/800 = 1.1% 命中这种 case。**单独的 follow-up**：这种 sample 应该从 `fast@x` 分母里排除（不是模型能力问题），但当前 metric 计算还在分母里，下个 PR 修。

## KernelGym 响应有两种 shape

生产环境实际 KernelGym `/evaluate` 响应是**顶层就是 env_state**（dump 里 2738 个 sample，0 个是嵌套形态）：

```
response = {"task_id": ..., "status": ..., "compiled": ..., "correctness": ...,
            "metadata": {...}, "error_code": ..., "error_message": ...}
```

少数客户端 / 老版本可能嵌套：

```
response = {"env_state": {"task_id": ..., "compiled": ..., ...}}
```

`format_kernelgym_feedback` 通过 `_looks_like_env_state(dict)` 检查 dict 是否含 `compiled / correctness / status` 任一签名 key，**两种 shape 都识别并都送 `build_prompt_feedback_payload`**——这是设计明文要求。

## 四、错误摘要规则

`summarize_diagnostic_text` 把任意 string（来自 `metadata.compilation_error / metadata.runtime_error / metadata.correctness_issue / metadata.error / state.error_message / state.error` 第一个非空者；`metadata.error` 是最后兜底，因为很多失败 case server 只塞这一个字段）压成 ≤ N 字符的简短表示，多级 fallback：

1. 命中 `Precheck failed: <line>` → 单行
2. 命中 `Syntax error <line>` → 单行
3. 命中 `(Task|Operation) ... timeout` → 单行
4. 命中 `'X' object has no attribute 'Y'` → 单行
5. 否则提取所有 `... error: ...` 编译错误行，**去重保序取前 4 条**
6. 否则提取 `XxxError:` / `XxxException:` 异常行，**去重取最后 3 条**
7. 否则提取含 `failed/error/exception/timeout` 字样的行，去重取最后 3 条
8. fallback：原字符串裸截到 N 字符 + `...<truncated>` 后缀

每个分支的输出都过一遍 `_truncate_text(..., limit=N)`，最终长度 ≤ `N + len("...<truncated>")`（多出约 14 字符的尾标记）。

## 五、默认值 + 可配置

错误摘要长度上限 N 的**默认值 1600 字符**（≈ 400 tokens），在 `slime_plugins/drkernel/args.py` 加一个 CLI flag：

```
--kernelgym-error-summary-chars  type=int, default=1600
```

跑训练 / debug script 时可覆盖。新 helper 读 `args.kernelgym_error_summary_chars` 传给 `summarize_diagnostic_text(limit=...)`。

## 六、集成

只动 `slime_plugins/drkernel/rollout.py`：

- 顶部加 4 个 helper 实现上面两步压缩
- `format_kernelgym_feedback` 在 "env_state 是 dict" 的分支接 `build_prompt_feedback_payload`
- 其他 fallback 路径（extract_error / 非 dict payload）保持
- 外层 `max_chars` 字符兜底截断保留作二级保险

`slime_plugins/drkernel/args.py` 加 `--kernelgym-error-summary-chars`。

slime 上游 0 改动。

## 七、验证

- **字符长度**：feedback 字符数分布 mean ≤ 2 KB，max ≤ 3 KB。
- **错误信息完整性**：对几个典型失败案例（nvcc compile error / runtime CUDA error / correctness mismatch / Precheck syntax error / timeout），检查摘要后的 `error_message` 是否仍然让模型能定位问题。
- **单元测试**：摘要 8 级 fallback 各覆盖 1 个 case；白名单字段过滤覆盖 5-7 个 case（含 extract_error 兜底、空 env_state 等边界）。

## 八、风险

- 1600 字符的摘要上限是经验起步值；极长堆栈或多文件错误时可能漏掉部分 root cause。先用这个值跑，看模型表现再调。CLI 可配，遇到 case 可临时上调。
- 白名单字段是当前 KernelGym `/evaluate` 响应 schema 的快照。如果 KernelGym 改 schema（重命名 / 拆分字段），白名单需要同步更新，否则 feedback 会少信息。
- `verbose_errors=True` 在 server 端继续生成大段 stderr 是浪费——本次只在 client 端把它丢掉，等需要再去改 server。
