# 多轮 kernel 优化：当前实验计划

当前检验一个简单假设：最佳答案中保留下来的优化实现，值得给较早实现它们的轮次分配 reward。FastCredit additive 已完成 100 次更新并停止，checkpoint 已完成多套 context 预算评测，尚未建立全面训练收益；接下来优先核对实际分奖覆盖与训练信号，再决定后续对照

本目录维护三个研究入口：[本文](plan.md)说明实验顺序，[组件奖励](component_reward_training.md)说明实现与验证，[状态聚合](state_aggregation.md)说明另一条研究线；[runtime 图教程](runtime_graph_extraction.md)单独展开采图方法。后续迭代直接更新对应文档

从源码匹配到 runtime 图及优化规则的尝试和失败原因，集中记录在[失败路径复盘](component_tracking_failed_paths.md)，不在实验计划中重复展开。FastCredit 与 Correctness diff 的共同奖励假设、整轮信号风险和对照需求见[多轮 reward 设计分析](多轮reward设计分析.md)

## 当前先做什么

主实验对照是最新的 v4_1 三轮 packed TRLOO，保持模型、数据、训练与 rollout 配置不变，在原累计回报上向合格最佳答案的早期组件来源轮额外加奖。跨轨迹状态聚合暂不启用，避免同时改变 credit 和比较组

当前采用单阶段方案，已实现 58 项源码／跨轮变化模板，失败轮同样参与分析。具体 kernel／库计算调用构成分奖单位，策略标签只是该实现的描述。策略清单、覆盖率和训练进展只在[组件奖励文档](component_reward_training.md)维护

此前局部匹配没有增加自然来源候选，历史结果保留在该文档附录。本次训练使用独立的保守整实现签名，不需要低层访存图或重部署 KernelGym。启动前已核对冻结源码，并分别完成真实 rollout 和训练重放检查；配置、验收与停止回执见[训练配置与执行](component_reward_training.md#训练配置与执行)

人工审计将策略机制识别、合法早期来源匹配和最早轮次判断分别计数；当前已发现别名／赋值链漏报和无关文件 context 引起的实际漏奖。历史训练日志审计缺少完整源码留样；FastCredit 训练已启用首批与最新批限量保存，后续可据此扩大检查，但两个批次不足以代表完整训练流量。具体数字和已核实案例只在组件奖励文档维护

后续验证重点放在三个问题：

- 有多少轨迹真的向早期来源轮加奖；分别看正确与 speedup 门槛、采集缺失、组件不匹配和最佳轮本来就最早的情况
- 正确率和多轮最佳结果是否改善，以及 advantage 尺度、过滤比例是否出现异常
- 计入 CPU 源码分析和训练后，收益是否值得额外计算成本

## 怎样解释后续训练结果

当前 additive 方法保留 baseline 累计回报，只把额外来源 bonus 加入一次，不再次向前传播。它同时引入额外奖励预算、正确与 speedup 门槛和来源分配规则；即使某项指标改善，也不能只据这一组实验归因于组件追踪。替代式与封顶目标的记录单独解释，不能混作 additive 的训练证据

后续对照可在保留 baseline 的前提下，将同等额外奖励全部加给最佳轮，或随机置换来源，区分额外奖励与来源分配的影响；有稳定信号后，再比较把额外计算预算用于普通 rollout，以及等预算单轮／多轮。开发集用于选择配置，最终测试集不用于挑案例或调规则

状态聚合解决的是另一个问题：不同轨迹的哪些状态适合共享比较。它需要独立测量自然轨迹中的覆盖率和组内差异，尚不能直接替换现有同题、同轮次的 TRLOO 分组

## E1.1 留下的有用线索

已有历史轨迹中确实能找到暂时退化后突破、非相邻复用和大改写后突破。这些现象用于提出假设，不能单凭先后关系判断中间修改有贡献

| 案例 | 观察 | 仍缺少的判断 |
|---|---|---|
| Qwen RNN，group 259 | concat 合并投影后变慢，保留 concat 并改用 TF32/GemmEx 后变快 | 不 concat、只改 TF32 是否已经同样有效 |
| DeepSeek GRU，group 304 | 把 timestep 递推移入 kernel 后变慢，随后预计算输入投影才变快 | 保留 host 递推、只做预计算会怎样 |
| Qwen，group 118／99 | 局部恢复旧 launcher，与放弃全融合路线，都可能表现为先退化后恢复 | 需要区分恢复了哪些实现，不能只看整轮成败 |

若后续研究因果交互，可以对前两例构造两个修改的四种组合，固定其余条件做执行对照。编译失败首先是可行性问题，不能赋零性能后宣称存在正交互；程序层面的有用性也不等于该轮修改改善了后续搜索，后者需要独立续跑

## 数据与证据

早期 E1.1 使用两个官方模型合计 800 条已有的 L3 五轮轨迹做静态筛选，并非新采集的自然训练池。来源缺失、签名改写和只看 raw speedup 都会影响筛选，以下比例不作为当前 runtime 覆盖率或因果频率

| 历史筛选指标 (%) | Qwen3.8 | DeepSeek-V4-Flash |
|---|---:|---:|
| 完整三段代码且有语法有效的 ModelNew，按轮次 | 67.75 | 99.25 |
| 命中任一候选现象，按轨迹 | 8.75 | 8.75 |

筛选口径和原始案例保留在[候选清单与 protocol](../../local_artifacts/paper/trajectory_candidates/validated/candidates.md)、[人工审计](../../local_artifacts/paper/trajectory_candidates/manual_audit/review.md)、[结构化证据](../../local_artifacts/paper/trajectory_candidates/manual_audit/cases.json)。工具说明见[轨迹结构](../../tools/data/trajectory_structure/README.md)与[候选筛选](../../tools/data/trajectory_candidates/README.md)
