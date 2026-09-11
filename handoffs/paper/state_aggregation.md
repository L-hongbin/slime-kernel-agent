# 状态聚合：怎样比较不同轨迹的程序

状态聚合要回答：两条轨迹都还没做对时，它们是否处在足够相近的计算状态，可以共享比较或学习信号？仅看源码相同、编译通过或 wrong_answer 太粗；只看数据访问区域也不够，因为相同区域可以执行不同计算

当前方向是通用的 runtime 调用图，不为每道题编写 signature。已经有重复执行和小样本交叉核对，但自然轨迹能聚合多少、聚合后是否适合训练比较，仍未得到可靠结论。这条线未接入当前[组件奖励实验](component_reward_training.md)

## 怎样提取状态、寻找对应

先运行候选的完整 forward，记录实际发生的 kernel、库调用、copy／fill，以及输入、输出、临时 buffer 的角色与生命周期。这样能处理本次执行走到的 if/else 和循环；单个输入没有走到的分支仍不在观测范围内

再用实际访存和受支持的库 API 契约恢复读写区域与跨调用依赖。调用顺序、参数中的指针只提供定位信息，不能单独证明数据边。状态描述保留三部分：调用结构、实现与配置、已确认及未知的约束

比较限定在同一任务和可比输入／执行条件下，从边界角色和区域接口寻找调用对应，再核对数据连接与配置。重复接口、临时 buffer 角色不明或融合／拆分都保留歧义，不因 kernel 名字相同就认定相同

这得到的是“观测结构可比”的候选，不是数学等价或可共享整轮 baseline 的证明。当前还没有可用于训练的自动合组规则，也不把状态线的分组直接放入 GRPO／TRLOO

## 能聚合多少

现有数字主要来自校准输入、重复执行和人为变体，不能回答大规模自然轨迹的聚合率

| 检查范围 | 观察结果 | 能说明什么 |
|---|---|---|
| 状态线开发输入 | 16 个输入的重复图均一致 | 这些输入上的重复观测稳定 |
| 增强档观测分组 | 34 份图中，32 份进入 13 个非单例组 | 包含重复执行和人工变体，只是观测组 |
| 插桩输出控制 | 51 对输出一致 | 所测执行未出现取证导致的输出差异 |
| 真实跨轨迹 RNN | 只比较了同题 g257 T4 与 g259 T4；状态线未找到严格调用对应，组件线找到部分接口对应 | 样本不足，不能估计自然覆盖率或据此合组 |

状态匹配与组件追踪分别实现了所需组件，只共享输入和经验。两边在校准样例和真实 RNN 上核对了调用、库参数与读写集合；两套实现一致支持这些具体观测，不等于互相证明了整图语义正确

## 为什么结构相同仍然不够

一个校准反例是错误的 B 转置：它可以读取与原版相同的字节集合，却算出不同结果。只比较区域会把它们放在同一观测组，实现版本或完整矩阵契约才能显示差别

人工检查 attention 时也看到：一条轨迹修好了 PV 的矩阵参数后变为 Correct，另一条虽然也修好了 PV，却交换了 QK 的输入，仍然 wrong_answer。相同的全程序标签掩盖了不同的局部修复状态；局部契约一致，也不表示来自上游的输入已经正确

这两类例子说明，后续合组至少要区分“共享了哪些局部结构”和“当前还有什么错误”。早期按参考环节手工标注的 MLP／attention 记录可以作为校准样例，但逐题适配不能作为几十 K 训练任务的实现路线

## 下一步只验证一个问题

用独立抽样的训练／开发轨迹，测量通用图能找到多少跨轨迹对应，并人工检查这些对应是否保留了关键差异。分开报告成功采集、找到局部对应和能形成候选组，避免混成一个成功率

先聚焦调用级表示和输入输出连接，不回到寄存器级图。库内部实现、未支持访存、临时 buffer 绑定和融合／拆分仍是主要缺口；实际存储字节相同，也不能省略 dtype、layout、math mode 与状态生命周期

只有覆盖率和组内差异检查过关，才进一步验证共享 baseline 是否降低噪声、是否引入偏差。组件来源跟踪可以先独立推进，不需要等待整份程序状态聚合完成

## 实现与证据

- 通用状态线的[工具入口](../../../slime-trace-state-matching/tools/data/state_matching/README.md)、[批次摘要](../../../slime-trace-state-matching/local_artifacts/state_matching/coarse/delivery_summary.json)和[跨实现核对](../../../slime-trace-state-matching/local_artifacts/state_matching/coarse/crosscheck/report.md)
- 主 Agent 的[共同执行条件核对](../../local_artifacts/trace_analysis_coordination/coarse_main_metadata_audit.json)与[审查裁定](../../local_artifacts/trace_analysis_coordination/coarse_grok_adjudication.md)
- 人工校准的[状态卡片](../../local_artifacts/paper/structural_state/final/cards)、[跨轮变化](../../local_artifacts/paper/structural_state/final/transitions.json)和[审核记录](../../local_artifacts/paper/structural_state/manual_review.md)，其中参考环节映射依赖人工适配，不作为通用覆盖率
