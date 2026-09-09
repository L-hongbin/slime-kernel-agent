# Runtime 组件追踪：已观测候选与留存边界

这版已经独立实现 runtime 指令采集、部分依赖恢复、图邻域匹配和最佳答案回溯，但**还不能作为组件留存 reward 的检测器**。输出是可审计的对应候选，关键状态未知、采样或截断时不会给出确定留存标签

组件追踪与 state matching 已按用户后续要求独立实现。这里的 collector、提取器、matcher 和回溯代码均由组件追踪线维护；双方只共享第三方 NVBit、验证输入和踩坑经验。早期共用匹配器的结果保留为联合原型，最终证据另存于 [final](../../local_artifacts/component_tracking/final/)，不能混作独立验证

## 从实际执行到候选组件

入口是一次候选程序在固定输入上的执行。NVBit 在 SASS 指令前插入记录，保存指令位置、线程、CTA、谓词结果、active mask，以及能取得的访存地址和常量位。它提供二进制插桩接口；依赖恢复和跨版本匹配由本工具实现，并非 NVIDIA 已经提供完整组件追踪器。[NVBit 官方说明](https://github.com/NVlabs/NVBit)

随后把实际读写连接起来。寄存器边来自受支持指令的输入、输出操作数和动态最后写入者；128-bit 向量操作会更新四个寄存器。全局、共享和局部内存按字节区域处理，覆盖写入只保留仍可到达读取的版本。跨线程关系需要经过已验证的 CTA barrier，跨 kernel 关系需要原有同 stream 顺序；日志先到先后不构成依赖。原地返回还会单独建立输出值节点，避免把输入 storage 和最终输出混成一个版本

这些规则按 SASS 指令和 CUDA 行为定义，不按题目写 signature。无法解释的指令、常量角色、allocation 或同步保持未知。尤其是 CBANK 的 32-bit 值也可能属于运行时 ABI，不能自动解释成模型的整数参数；这里保留原始位供审查，但不据此判定实现相同。[CUDA 指令类别参考](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-binary-utilities/index.html)

完整动态展开会很大，因此图按指令位置归纳重复事件，保存逐线程的谓词、mask 顺序和访存序列。相同线程模式用连续区间压缩；这一步没有丢弃该模式里的事件，但归纳图仍不能替代完整动态图的等价证明

在此基础上，独立 matcher 使用 NetworkX VF2 检查带属性、带方向的局部邻域，并保留重复部件造成的歧义。最佳版本按调用方提供的正确性和分数选择，同分取最早版本；最佳图缺失时不换成较差版本。回溯输出“此前在哪轮观测到对应候选”和“是否连接到已观测输出”，不把一条通用地址计算或 EXIT 的匹配当成实现贡献，也不按节点数量给 reward

## 同一输入的独立交叉核对

两条线分别用自己的 collector 和事件解析器，在同一个 GEMM 可执行文件上跑了九种变化：朴素实现、改名与 helper、tile 8/16、错误的 B 转置、运行时 alpha 变化、相同已执行分支，以及增加独立 scale kernel

九个用例的目标二进制 hash、launch 数、issued/executed 指令数、完整 opcode 计数、按空间与宽度分类的访存事件、输出 hash 和参考误差全部一致。朴素实现共有 35,584 条 issued 事件，其中 33,039 条谓词为真；双方没有把这两个分母混在一起。[本线独立核对结果](../../local_artifacts/component_tracking/final/independent_crosscheck.json)

这证明两套采集器在这些输入上取得了相同底层事实，不能外推成依赖图完全正确或状态等价。对错误转置或 alpha 改变，两边都能如实记录不同输出；但未识别常量角色时，部分图属性仍可能相同，因此不能据图相同直接确认留存或合并状态

在改名的 GEMM 中，候选已经能落到实际乘加，例如 `FFMA R25, R10, R25, R21`，并沿已恢复的依赖连接到输出。它仍是局部观测候选；运行时 alpha 改变的反例说明，同一局部片段可以出现在不同整体计算中。[代表性人工核对](../../local_artifacts/component_tracking/final/manual_review.md)

## 实测反例改变了输出契约

独立复核发现，最初的向量处理只更新两个寄存器，会让后续指令把 `LDS.128` 写入的第三、第四个寄存器误接到旧 producer。现在按操作数宽度展开，并补了向量和 uniform carry 检查。barrier 用例也逐项核对了跨线程写入和读取，没有把不完整的 barrier 当成同步

另一类问题更根本：采样的第一个 CTA 即使与前一轮完全相同，也不能证明其他 CTA 没有替换计算。现在采样和截断会传到指令节点；输出始终标为候选，`certifies_retention=false`、`eligible_for_credit=false`。真实 Qwen RNN 的三个版本完成了输出一致性复放，但只采 concat 的一个 CTA；严格规则下没有给出确定留存结论。历史 raw speedup 只用于演示最佳版本选择，不替代训练的 `q(K)`

还有一个实际插桩错误：FP16 reduction 中的 `ULDC.U8 … c[0][0x621]` 被探针按 32 位常量读取，触发 `CUDA_ERROR_MISALIGNED_ADDRESS`。限制到对齐且宽度至少四字节的常量读取后，同一源码和输入成功完成，输出 hash 与无插桩控制一致。subword 常量仍记未知；修复后的轨迹还有 buffer 截断，不能称为完整依赖捕获。[失败日志](../../local_artifacts/component_tracking/final/raw/fp16_diagnostic.log)、[修复后的控制](../../local_artifacts/component_tracking/final/summary.json)

重复执行同一 BF16 训练参考程序时，曾出现图属性变化。核对发现未注册的中间指针和运行时 CBANK 值被误当成标量；修正后，两次独立进程的节点属性与边属性一致。这个控制验证了记录的稳定性，没有证明这些未知字段的语义相同

## 规模和当前可用范围

下表是单次冷 forward 的诊断 wall time，包含首次插桩等开销，既不是稳态开销，也不是模型正式性能分数。所有控制均使用相同源码、初始化和输入 seed，输出指纹相同

| 工作负载 | 无插桩 (s) | 插桩 (s) | 观测范围 |
|---|---:|---:|---|
| BF16 训练参考，输入 1×32 | 0.120 | 4.263 | 所有 launch、所有 CTA |
| FP16 训练参考，输入 1×64 | 0.173 | 3.809 | 所有 launch，部分事件因上限截断 |
| 额外训练样本，输入 8064×8192 | 0.021 | 1.567 | 每个选定 launch 的第一个 CTA |

很小的 BF16 输入也产生约 83 万条指令事件。完整计数包含大量库实现和线程展开，不能只看输入字节数估算追踪规模。最终图约 6,900 个节点、1.5 万条边；精确线程模式压缩后，格式化 JSON 仍约 16 MB，CPU 提取需要数秒。[准确计数、耗时和输出控制](../../local_artifacts/component_tracking/final/summary.json)

额外训练样本按固定位置选取，没有添加该题的抽取分支；它完成了原始 shape/dtype 的输出一致性检查。实现期间还根据通用复核修正了计数和寄存器契约，因此这不作为严格冻结的全量 held-out 成功率实验

当前适合离线诊断、反例构造和对应候选审查，尚不适合几十 K 训练题的逐 rollout 全量追踪。下一步真正需要补的是通用的参数与 allocation 角色、更多可靠的指令契约，以及在保留依赖证据的前提下降低采集量；单纯增加邻域匹配复杂度不能填补这些缺失。拆分、融合、重新编译后的语义关联也仍未解决

## 实现、审查与复现

- 维护入口：[tools/data/runtime_trace](../../tools/data/runtime_trace/README.md)
- 最佳版本真实回溯：[lineage_real.md](../../local_artifacts/component_tracking/final/lineage_real.md)
- 校准用例候选回溯：[lineage_control.md](../../local_artifacts/component_tracking/final/lineage_control.md)，分数是显式的接口控制值，不是 benchmark speedup
- 完整证据与实现 hash：[summary.json](../../local_artifacts/component_tracking/final/summary.json)
- 独立 Grok 复核：[review_verdict.md](../../local_artifacts/component_tracking/review_verdict.md)，最终修订以本地契约检查和真实复放支持，不能称为已通过完整独立验收
- 初始源码快照与隔离来源：[baseline_manifest.json](../../local_artifacts/component_tracking/baseline_manifest.json)

30 项 CPU 契约检查通过，覆盖 byte overlap、竞争、barrier、向量寄存器、uniform carry、缺失最佳版本、回退、重复部件歧义及采样不能确认留存。另核对了共同目标中 14,336 次否定谓词 branch 的实际下一条指令，没有发现与 guard 不符的跳转

代码位于独立 worktree `slime-component-tracking`、分支 `component-tracking-runtime`，基于 `0567b377`。实验使用 node69 的独立进程；另一条线使用 node70。没有修改部署中的 KernelGym、模型 feedback、训练 reward 或共享运行环境。第三方 NVBit 和所有生成产物留在 ignored `local_artifacts/` 下
