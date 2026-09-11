# 从实际执行构建组件图

组件追踪要回答的是：最终答案用了哪些实现，其中哪些在较早轮次就已经出现。为此，我们把一次 kernel 调用或一次库操作作为一个组件，记录组件之间怎样传递数据，再比较不同轮次的组件

本文先用一个小例子说明这张图怎样得到、怎样用于追踪，再讨论 NVIDIA 工具的选择和当前缺口。服务接入、采集参数和复核文件放在文末；奖励分配规则见[组件奖励方案](component_reward_training.md)

## 用两个 kernel 说明节点和边

假设程序先把输入乘以系数，再加 bias 并做 ReLU。下面的 `ops` 代表自定义 CUDA 绑定，整个例子只有两次 kernel launch

```python
def forward(x, bias, a):
    tmp = torch.empty_like(x)
    ops.scale(x, tmp, a, x.numel())
    ops.bias_relu(tmp, bias, x.numel())
    return tmp
```

对应的 CUDA 计算如下，省略绑定和启动代码，假定输入是连续 FP32 向量，两个 kernel 在同一个 stream 上执行

```cuda
__global__ void scale(const float* x, float* tmp, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) tmp[i] = a * x[i];
}

__global__ void bias_relu(float* tmp, const float* bias, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) tmp[i] = fmaxf(tmp[i] + bias[i], 0.0f);
}
```

要得到的图很简单：

```text
x ──→ [scale] ──→ tmp 的第一版数据 ──→ [bias_relu] ──→ tmp 的第二版数据 ──→ 返回值
                                          ↑
                                         bias
```

一个方框对应一次调用。边表示后一个调用读取了前一个调用写出的数据。两版 tmp 使用同一块显存：第一版存乘法结果，第二版存加 bias 和 ReLU 后的结果。记录数据版本，就能区分“地址相同、内容已经更新”的情况

组件粒度也由这个例子确定：`bias_relu` 是一个组件，其中的加法和 ReLU 一起追踪。如果后续把两次调用融合为一次 `fused_scale_bias_relu`，融合版就是一个新组件。我们关心它何时出现、后来是否保留，因此不需要把它再拆成寄存器或算术指令级的图

## 怎样从一次执行得到这张图

### 先记录实际执行的调用

准备好输入和模型参数后，开启记录，执行一次 `forward`，等待 GPU 完成，再停止记录。得到的调用序列在这个例子中就是 scale、bias_relu

每次调用要记录所执行的 kernel、启动配置和所属 stream。这些信息告诉我们“运行了什么”和“程序保证了怎样的执行顺序”。循环执行几次就记录几次调用；if/else 则记录本次输入实际走到的分支

只有调用序列还连不出数据依赖。例如 scale 后面即使启动了另一个 kernel，它也可能只处理独立数据。要确认两者有关系，还要知道读写了什么

### 再记录每个调用读写的存储区域

buffer 是 tensor 背后的存储空间。在本例中，需要区分 x、bias、tmp 三块存储，并得到如下记录

| 调用 | 读取 | 写入 |
|---|---|---|
| scale | x 的元素 | tmp 的元素 |
| bias_relu | tmp 的旧值、bias 的元素 | tmp 的新值 |

对自定义 CUDA kernel，当前用 NVBit 在实际执行的访存指令旁插入记录函数，取得地址、访问长度和读写方向。再结合 tensor 的存储范围，把 GPU 地址解释为“tmp 的哪一段字节”。支持的库操作则可以根据 API 参数推导读写区域，例如 GEMM 的矩阵尺寸、转置和步长

记录区域是为了区分同一块大 buffer 的不同部分。A 写前半段、B 只读后半段时，两者没有通过这些区域传递数据。名字相同或指针属于同一块分配，都不足以确认依赖

跨进程的显存地址也可能改变。记录成“输入 x”“临时 buffer tmp 及其中偏移”，才能比较两次独立执行

### 根据读写关系和执行顺序连边

在这个例子中，scale 先写 tmp，bias_relu 随后读取这版 tmp，而且中间没有其它调用覆盖它。因此建立 `scale → bias_relu`

bias_relu 是原地更新：读旧值和写新值发生在同一 kernel 内。单纯的读集合、写集合只说明两者重叠；要确认它确实读取了调用前的数据，还需要访问顺序。本例的代码是同一线程先读 `tmp[i]` 再写 `tmp[i]`，当前采集器通过针对这类调用补充线程内顺序记录来确认

多 stream 的程序还要检查 event 等同步关系，不能把日志中谁先出现直接当成程序的数据顺序。无法确认的边会标为未知，已有的调用节点仍然可以保留

### 最后确定返回值来自哪里

`forward` 返回 tmp，所以要记录“返回的 tensor 对应 tmp 的哪一段存储”。找到最后写入这段数据的 bias_relu，再沿依赖边向前找到 scale，就得到参与返回值计算的组件集合

如果程序另外计算了一块 scratch，但既不返回它，也没有后续调用读取它，那么沿返回值向前查找时就不会经过它。这样可以排除已确认与输出无关的独立计算

这里的关系是调用之间的存储数据来源。一个 kernel 即使读了某个值，也可能在内部丢弃它；调用级图可能保留这种多余依赖。要证明每个读取都改变了最终数值，需要更细的分析或执行对照，当前组件奖励采用的是“最佳答案保留下来的实现值得 credit”的假设

## 图怎样帮助跨轮追踪组件

假设第一轮使用 scale 和 bias_relu，第二轮改成融合组件 F，第三轮保留 F 并修好了别处的错误。最佳答案是第三轮时，我们希望把 F 的来源找到第二轮

第一步是在第三轮的图里找到 F：它执行了一次融合调用，读取 x 和 bias，写出返回值。第二步是在第二轮寻找对应调用，核对实现、配置和读写接口。接口包括输入输出的角色、形状、类型、布局，以及多个参数是否共享存储

| 跨轮变化 | 当前如何解释 |
|---|---|
| F 的实现、配置和接口相同，而且对应唯一 | 可以记录相同实现的较早出现轮次 |
| 输入输出接口类似，但机器代码改变 | 记录实现发生修订；仅凭相似接口不能确认原实现留存 |
| 两个独立 kernel 变成一个 fused kernel | fused kernel 是新组件 |
| 多个重复调用都符合描述 | 对应有歧义，暂时不能唯一确定来源 |

朴素 GEMM 和 tiled GEMM 可能拥有相同的读写接口，但实现不同。图记录数据关系；实现指纹是从捕获的机器代码等信息计算出的摘要，用来比较实现版本。两者一起用于追踪，无需为每道题专门定义匹配规则，也不要求自动给实现命名为“朴素”或“tiled”

当前奖励代码使用严格实现匹配。允许实现修订的“相同调用角色”匹配只做离线候选分析，还没有用于训练分奖。即使找到组件较早出现的轮次，也只说明观测到了实现留存，不证明模型复制了那份代码，或证明它带来了多少性能收益

## NVBit、Nsight Systems 和 CUPTI 怎样选择

NVBit 是当前验证读写关系的工具，还没有通过同任务对照证明它是最佳生产方案。工具选择取决于缺的是“调用记录”还是“自定义 kernel 的实际读写区域”

| 工具 | 官方提供的主要能力 | 对组件图的用途 | 还需要我们补什么 |
|---|---|---|---|
| Nsight Systems，命令行名 `nsys` | CUDA API、kernel、copy 和同步等执行时间线 | 人工检查调用是否漏记、执行顺序是否合理 | 数据区域依赖、跨轮匹配 |
| CUPTI | 用 Activity / Callback API 编程采集 CUDA 活动和事件 | 构建可集成的调用采集层 | tensor 存储映射、任意自定义 kernel 的读写区域、图与匹配 |
| NVBit | 对实际执行的 GPU 机器指令插桩 | 为自定义 kernel 采集实际访存，必要时采集顺序 | 区域汇总、数据版本、图与匹配；还要控制插桩成本 |

Nsight Systems 适合直接查看执行过程，CUPTI 适合开发自己的采集器。它们的常规 CUDA tracing 记录 kernel 执行和内存复制等活动，不能直接给出任意 kernel 内每次 load/store 所访问的 buffer 区域。CUPTI 的 memory allocation 记录、访存性能指标和 PC sampling 也不能替代这份数据来源记录，见 [Nsight Systems 用户指南](https://docs.nvidia.com/nsight-systems/UserGuide/)和 [CUPTI API 概览](https://docs.nvidia.com/cupti/)

以本例为例，时间线能说明 scale 先执行、bias_relu 后执行。要连出 tmp 这条边，还需要知道前者写 tmp、后者读 tmp 的旧值。这部分可以来自可信的算子接口说明、通用编译分析，或实际访存采集；仅把采集工具换成 CUPTI 并不会自动补齐

NVBit 提供了观察实际访存的入口，但插桩函数本身需要执行 GPU 指令，会增加开销并扰动调度。它还是独立于 CUDA Toolkit 发布的研究工具，见 [NVBit 官方说明](https://github.com/NVlabs/NVBit)。当前位图容量不足、参数结构体解码不全等问题，则属于我们基于它实现的采集器，不能全部归为 NVBit 的固有限制

如果关注 NVIDIA 新的硬件追踪，CUPTI 文档中的 HES 是从 Blackwell 引入的 kernel 时间戳采集机制。它有助于降低时间线采集成本，提供的仍不是任意程序的逐字节数据依赖图，见 [HES 说明](https://docs.nvidia.com/cupti/main/main.html#hardware-event-system-hes)

### 更合适的迭代方向

建议验证分层采集：调用与同步等信息使用较轻的 API 级采集；已知 copy、fill 和受支持库操作按接口规则确定读写；只有无法从这些信息确定读写的自定义 kernel 才使用 NVBit。需要解释原地更新时，再局部增加顺序记录

目前已经使用库接口规则跳过受支持 GEMM 的内部逐次访存，但自定义 kernel 的区域模式仍会对受支持访存指令插桩。CUPTI 调用层替换、进一步选择性插桩及其实际收益尚未完成对照验证，不能写成已经实现的低开销方案

如果训练只需要近似匹配，也可以先用调用和接口筛选候选，将精确采集用于抽查或高价值轨迹。这个选择会改变确认标准，必须测量误匹配率，不能把近似图直接当成已经确认的数据依赖图

## 当前困难是否意味着 runtime 组件追踪不可行

调用级组件图已经有真实样本能够采集和匹配，下面的 RNN 就是一例。当前尚未证明的是：面对大量任意模型生成代码，能否以训练可承受的成本，稳定取得足够完整、足够准确的组件来源

需要分别处理三个问题，不能只用“图不完整”概括

| 问题 | 当前具体情况 | 对判断的影响 |
|---|---|---|
| 采集器没有记录完整 | 大 buffer 超过区域表容量，部分参数结构体没有解码，某些 API 尚未覆盖 | 属于工程覆盖缺口，不能据此认定组件追踪不可行 |
| 候选执行本身有问题 | 已核实的尾部越界读取、cuBLAS 返回失败后继续消费未写入的 buffer | 图应保留错误或未知，换采集工具不会修好候选 |
| 重放是否代表原评测还未确认 | 当前输出按字节哈希比较；原子浮点归约可能数值接近却哈希不同 | 需要检查实际误差和重复执行，不能直接视为正确性失败，也不能直接放行 |

为检查重放是否仍代表原评测，停止采集后会分别记录两类输出信息：

1. 返回 tensor 对应的存储区域，用于把最后一个计算节点接到返回值
2. 输出的 shape、dtype、布局等信息和内容哈希，用于比较原评测与重放是否一致

哈希是 tensor 内容的摘要，内容变一位就会变化。它不是 tensor 的名称，也不是数值正确性的判断。目前 trajectory165、169 的浮点原子归约输出存在哈希不一致，具体数值误差还没有验证完成；trajectory42 则发现了库调用失败后继续执行的问题，原评测为何仍判 Correct 也没有闭环

单次 runtime 图描述本次输入走过的路径。换输入可能走另一条分支，所以“采到了这次调用关系”与“覆盖所有输入下的行为”是两种不同要求。我们的当前目标是观测同一任务条件下的组件留存，先把这个目标的覆盖率、误匹配率和成本测清楚，再决定是否适合接入大规模训练

## 附录：当前实现与复核示例

以下内容用于检查代码和复现，不影响前面的方法定义

### 实际图的简化表示

下面用教学例子的 32 个 FP32 元素展示一条边。它表示 scale 写出的 tmp 数据被 bias_relu 读取；原地读旧值的关系假定已经通过顺序采集确认。字段是实际 schema 的子集，完整节点、buffer 信息和其它边省略

```json
{
  "schema": "coarse-component-graph/v1",
  "edges": [
    {
      "source": "kernel:scale",
      "target": "kernel:bias_relu",
      "buffer": "storage:tmp",
      "version": "storage:tmp@kernel:scale",
      "regions": [[0, 128]],
      "certainty": "proven_region_dependency",
      "evidence": "observed_global_memory_runs"
    }
  ]
}
```

`regions` 使用左闭右开的字节区间，128 字节对应 32 个 FP32 元素。`version` 指出数据由哪个调用写出，`certainty` 表示这条关系的确认状态，`evidence` 说明依据来自哪种记录

### NVIDIA 插桩 API 与区域汇总

下面是[原生采集代码](../../tools/data/runtime_trace/runtime_trace.cu)的关键调用，省略支持性检查。NVBit 提供访存地址和指令属性，`trace_regions` 是我们实现的 GPU 记录函数

```cpp
nvbit_insert_call(ins, "trace_regions", IPOINT_BEFORE);
nvbit_add_call_arg_guard_pred_val(ins);
nvbit_add_call_arg_mref_addr64(ins, 0);
nvbit_add_call_arg_const_val32(ins, ins->getSize());
nvbit_add_call_arg_const_val32(ins, mode);
```

记录函数检查指令是否实际执行，再按地址、长度和读写方向更新集合。当前 `regions` 模式按 1 KiB 地址块建立索引，块内用逐字节位图保留精确范围；重复读取同一字节只置位，不反复追加日志。导出时将连续访问合并为区间

```text
观测区域： [0, 4), [4, 8), [8, 12), [16, 20), [0, 4)
合并结果： [0, 12),                 [16, 20)
```

这种 bucketize 是存储索引，块内仍保留空洞。若只保留粗桶编号，A 写桶的前半段、B 读后半段也会被误判为有关联。因此粗桶可以筛选候选，确认区域依赖仍需精确核对；当前不记录中间 tensor 的全部数值，也没有数值分桶

`runs` 模式额外保留线程归属和访问顺序。原地调用的定向重试从相同初始状态重新执行完整 forward，只对选中的调用增加这种记录；重试通过核对后使用整张新图，不拼接不同执行的局部图。跨线程冲突和原子更新仍可能无法确认来源

| 当前预算 | 含义 |
|---|---|
| 每调用 4M tile | 区域表最多覆盖约 4 GiB 不同地址空间，实际可用容量还受表占用与冲突影响 |
| 区域表约 1 GiB | 读写位图的工作区；导出临时数组另有约百 MiB 上限 |
| 每 forward 最多 4M 区间 | 导出的区域记录上限，与 tile 数是不同计数 |
| 定向重试另限 4M 顺序记录 | 沿用 control＋trace 60 秒执行预算、合计 30 秒 CPU report 预算 |

一次复制同时读取约 4 GB、写入约 4 GB 时，输入和输出地址都会占表项，所以仍可能超过容量。减少落盘日志不代表减少了同等比例的 GPU 插桩工作。这两项成本应分别测量

### 库调用与无关记录怎样处理

成功的受支持 GEMM 按 API 参数计算 A、B、C 的实际区域，包含转置、leading dimension、batch stride 和 alpha/beta 等条件。例如 beta 为零时，C 的旧值无需作为输入。各 batch 的区域求并集时保留 padding 空洞，失败调用和潜在冲突不会获得已确认的库数据依赖

库内部多个 kernel 归入一次父级库操作。实现版本使用实际加载的 cubin、执行入口和 launch 配置记录；cubin 是编译后的 GPU 二进制。NVBit 提供 `nvbit_dump_cubin` 导出接口，CPU 端验证文件与入口后计算哈希，见 [NVBit 发布说明](https://github.com/NVlabs/NVBit/releases/tag/v1.7.6)。当前专用读写规则覆盖 GEMM 及相应 strided-batched 变体，cuBLASLt、cuDNN 尚无同等覆盖

其它记录按其作用处理：

| 记录 | 在图中的用途 |
|---|---|
| 模型构造、编译、输入初始化 | 放在 forward 采集范围之外 |
| 分配、view、存储生命周期 | 用于解释地址归属和别名，不单独计作计算组件 |
| stream、event、同步 | 用于确认程序执行顺序，排除采集器自行插入的同步 |
| copy、fill | 保留读写效果，可能是输出计算的一部分 |
| 绝对地址、进程 handle | 匹配时标准化，保留偏移、布局及共享存储关系 |
| 已确认与返回值无关的调用 | 原图保留，输出组件集合不选取 |
| 截断、未知写入、未覆盖的 API | 保留未知及其影响范围，不能作为噪音直接删掉 |

### 一份实际 RNN 图

下面的路径来自[真实 H20 RNN 顺序采集结果](../../local_artifacts/component_reward_training/rnn_t5_2m.trace.result.json)。clone、concat 等含义由调用记录和样本代码人工核对，图中参数输入为简洁起见省略

```text
hidden → [copy:1 clone] ──┐
                         ↓
x ─────────────→ [kernel:2 concat]
                         ↓
                  [library:4 GEMM]
                         ↓
                  [kernel:9 bias+tanh]
                         ↓
                  [library:11 GEMM]
                         ↓
                  [kernel:16 bias]
                         ↓
                       返回值
```

bias+tanh 原地更新第一次 GEMM 的结果，第二次 GEMM 读取更新后的数据，和前面的 tmp 例子相同。T4 与 T5 比较时，clone、concat、bias+tanh 和末尾 bias 找到了一致的组件描述；这份采集尚缺两次 GEMM 的完整实现版本，因此没有把它们确认为跨轮实现留存

这个例子验证了“记录调用、连接数据来源、比较组件”的可执行性。它没有证明临时退化奖励有效，也没有证明在全部训练数据上能以同样成本完成；相关数量统一放在[组件奖励实验结果](component_reward_training.md#实际-rollout-中有多少有意义的临时退化)

### 评测服务接入与代码入口

当前 KernelGym 负责保存正常评测的输入、注册参数和 buffer、RNG 状态及执行配置，然后在隔离进程中执行普通重放（control）和插桩重放（trace）。这只是当前保存与重放实验的接入位置；构建组件图所需的是可运行的候选及其输入状态，不依赖 KernelGym 这个服务名称

正常评测负责正确性和性能 reward，插桩耗时只用于衡量采集成本。返回 tensor 的存储登记与输出内容核对，在[重放实现](../../tools/data/runtime_trace/service_replay.py)中是两项独立操作：

```python
observer.observe(output, "output:0")       # 返回 tensor 对应哪段存储
result["output"] = fingerprints(output)   # 记录结构信息和内容哈希
```

服务请求的 `record_format` 默认 `regions`，直接运行原生 tracer 未设置环境变量时仍默认 `runs`，复现时应显式指定。当前还有未登记的私有 CUDA 分配、未解码的参数结构体和未支持的访存等覆盖缺口；多个活跃 CUDA context、多个 launch 主机线程及 CUDA Graph capture 当前拒绝采集。未注册的 Python 状态也可能影响重放

| 实现环节 | 入口 |
|---|---|
| 构建和执行命令 | [工具 README](../../tools/data/runtime_trace/README.md) |
| 重放、输入状态与输出核对 | [service_replay.py](../../tools/data/runtime_trace/service_replay.py) |
| 调用与 GPU 访存采集 | [runtime_trace.cu](../../tools/data/runtime_trace/runtime_trace.cu)、[inject_funcs.cu](../../tools/data/runtime_trace/inject_funcs.cu) |
| tensor 存储与生命周期 | [observer.py](../../tools/data/runtime_trace/observer.py) |
| 库 API 参数与父子调用 | [library_trace.cpp](../../tools/data/runtime_trace/library_trace.cpp) |
| 地址绑定、区域和数据版本 | [extract.py](../../tools/data/runtime_trace/extract.py) |
| 跨轮对应与输出组件奖励 | [match.py](../../tools/data/runtime_trace/match.py)、[component_reward.py](../../examples/kernel_agent/component_reward.py) |

当前缺口的源码核对见[尾部越界人工检查](../../local_artifacts/component_reward_training/detail_iteration/unmapped_review.md)；整批观测与未解决案例见[组件奖励报告](component_reward_training.md)。本文负责解释方法，不另维护一份实验进度表
