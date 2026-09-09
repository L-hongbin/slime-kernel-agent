# 按最佳答案的组件来源分配奖励

这项实验把每条轨迹最佳答案的质量奖励分给实现其组件的轮次，观察它能否比现有 TRLOO 更准确地奖励有用的中间尝试。KernelGym 在独立诊断执行中采集调用图，slime 根据图追踪组件并构造逐轮训练信号

核心分奖机制已在真实候选上跑通，slime 与 KernelGym 的实现和安全修复已完成，但还没有重部署服务或启动新训练。当前阻碍是 KernelGym 的连接入口失效，以及两台 rollout 机器仍被其它容器占用；没有停止这些作业，也没有缩减两台 train、两台 rollout 的配置

本轮最直接的实物结果：trajectory240 的原训练三轮 reward 为 `[0, 1.74, 1.85]`，新采集的图确认最佳第三轮的 kernel 在第二轮就已实现，离线分奖变为 `[0, 1.85, 0]`。这是原训练实测 reward 与新 H20 图结合的反事实检查，不是新训练效果或部署后的全流程验收

## 从一个例子说明奖励如何变化

假设第三轮得到了轨迹中的最佳答案，其原始 reward 为 1.2。这个答案有三个参与输出的计算调用，其中两个在第一轮已经实现，第三个在第三轮才出现。第一轮虽然整体没做对，但两个调用最终被最佳答案保留，就应获得相应的奖励

本轮先采用调用等权的启发式，把最佳答案的总奖励分成三份，再按来源轮次归集

| 轮次 | 最佳答案中最早在这一轮出现的调用 | 分配的 credit |
|---|---|---:|
| 第一轮 | 两个 | 0.80 |
| 第二轮 | 没有确认的来源 | 0.00 |
| 第三轮 | 一个 | 0.40 |

分配后的 credit 直接进入现有的同题、同轮次 leave-one-out，之后使用原来的 packing 和 trainer。不能再把未来 credit 累积回前面，否则例子会变成 `[1.2, 0.4, 0.4]`，重新奖励没有留下组件的第二轮，并重复计算同一份预算

原有逐轮 task reward 单独保留。最佳轮按它选择，平分时取最早一轮，不自动选最后一轮；原有负奖励继续保留。正确答案的奖励底线高于 value mismatch 的部分奖励，因此有正确轮时会优先选正确轮，全错时仍遵守原有部分奖励定义

与旧 TRLOO 相比，这同时改变了质量预算的选取和跨轮分配：旧方法累计逐轮 reward，新方法只分配一次最佳质量预算。保持学习率等超参不变，不代表 advantage 尺度相同；后续即使有收益，也不能仅凭这个对照把收益全部归因于组件追踪。仅奖励最佳轮的对照可以进一步拆分这两个因素，本轮尚未另启该训练

## 怎样判断一个调用来自哪一轮

KernelGym 提供真实执行中的调用、buffer 区域、依赖、实现版本和配置。slime 对最佳答案中参与输出计算的调用，查找此前最早能唯一对应的实现；不使用源码相似度，也不把相同访问区域直接当作相同计算

这里的来源表示最早观测到同一实现，不证明模型复制了先前源码。`A→B→A` 中恢复的实现可以对应到第一次 A，但要标为再出现，与持续保留分开。库 API 与其内部 kernel 不重复计数，无数据效果以及确定不参与输出的调用不分奖励

融合、拆分、缺失记录或歧义可能让来源无法确认。这部分预算显式留给最佳轮，记录为未归因份额；最佳图整体不可用时，退回仅奖励最佳轮的策略并单列原因，不能把它统计成组件归因成功。完整轨迹的正 credit 总和始终等于最佳答案的正 reward，不因 kernel 数增加而膨胀；调用等权仍可能影响各轮分配，这是需要检查的启发式偏差

soft finalize 在归因之后处理，保留有正 credit 的有效来源轮，不因为该轮整体错误而删除它。padding、model abort 和其它不可训练样本不恢复。动态过滤也要查看完整 credit 向量，否则最后一轮没有 credit 时可能误删前面有贡献的轨迹；方差阈值和最小 group 大小沿用基线

## KernelGym 与训练链路的分工

KernelGym 的正常 correctness 和性能计时保持原流程。可选诊断在隔离进程内执行，记录相同候选、输入和配置的身份；采集超时或缺口附加在独立字段，不覆盖原评测结论

HTTP 返回有界、自包含的图摘要，完整 trace 另存 artifact。slime 将摘要放入 `Sample.metadata.runtime_graph`，不放入给模型的 feedback，也不改变原始评测 reward。原型中的 [runtime 图](runtime_structure_summary.md)作为实现基础，不重新引入寄存器级分析

## 真实样本揭示的限制与修复

RNN 两个真实候选的六个调用中，三个自定义 kernel 和一个 copy 可以跨轮对应，两个库调用因为实现未检查而保留未知。没有为了提高匹配比例而放宽输出成员资格：早期 1M 记录截断了最后一个 kernel，整条输出路径无法证实，只能把奖励留在最佳轮；补足记录后才恢复四个可匹配单位

训练样本的 FP16 kernel 需要更多逐 warp 记录，最终诊断上限为 4M records、60 秒 control+trace、30 秒 CPU report。原评分的 300 秒、模型 context、采样与训练超参不变。超出容量时仍保守处理：不完整的访存细节可以显式省略，但调用本身不能从奖励分母消失；调用清单缺失时整图 fallback

这套细粒度访存取证仍有明显成本。两份 FP16 样本的附加诊断各约 19 秒，而插桩 forward 仅约 0.89–1.06 秒；RNN 的附加诊断约 23 秒。不能把 forward 的一秒当作每题总开销

含安全修复、CPU 优化前的 H20 canary 进一步分解为 control 2.78 秒、trace 3.90 秒、CPU report 13.03 秒，瓶颈主要在 CPU 整图。对无写入的读取和互不相交的写入增加精确快速路径后，同一份完整真实 trace 的 CPU 对照从 11.37 秒降至 8.16 秒；500 组随机比较和整张真实图内容核对均一致，没有减少访存证据或放松匹配。这是 CPU 路径的配对结果，不是完整训练的吞吐增益，完整轨迹池覆盖率与吞吐仍待验收

另有两个工程问题已通过实物或 CPU 反例关闭：紧凑 JSON 预算与缩进写盘不一致，会让合法图被文件大小上限拒绝，现已统一编码；诊断子进程未回收时，不能只隔离 GPU 却 ACK 任务，现沿既有路径先冻结精确 claim、保留原评分，再隔离通知。普通 `setsid()` 子进程也纳入本次调用的身份标记清理，但不把它宣称为抵抗恶意清环境变量的安全沙箱

## 对照配置与后续验收

| 项目 | 沿用的 TRLOO 基线 |
|---|---|
| 模型与数据 | Qwen3.8-27B，v4_1 |
| Trainer | node69/70，各八卡 H20；BF16，TP4/PP2/CP2/SP |
| Rollout | node53/64，各八卡 H20；四个 TP4 FP8 engine |
| 轨迹／批量 | 三轮，16 prompt × 16 samples，packing |
| Context | 24576 / 32768 / 40960 |
| MTP | rollout 三步，单步 teacher forcing，CE 系数 0.2 |
| 优化器 | Adam，constant lr=1e-6，beta=(0.9,0.98)，weight decay=0.01 |
| 采样与长度惩罚 | temperature=1、top-p=1、top-k=-1，长度惩罚系数为零 |
| Serving | CUDA Graph，128 并发，static fraction=0.80，logprob chunk=256 |

启动前要验证三层：关闭新模式时原路径不变；真实 replay 能给出可人工核对的跨轮 credit，而非全部 fallback；credit 经过滤、LOO 和 packing 后实际进入 trainer 的 token advantage。还要核对诊断开销、采集缺口、隔离超时与 GPU 进程清理，最后做实际 rollout／train 检查

主工作区已复跑奖励、packing 和通信三组检查，分别通过 66、56、12 项。它们验证 credit 守恒、过滤、CP1/2 下 PPO／DPPO 的 packed/split loss 与非零梯度，以及不修改模型 feedback；不能代替真实 HTTP 图和 27B trainer 验收

KernelGym 的正常 worker、关闭流程、诊断、精确 claim 冻结与后代回收聚焦检查通过 56 项。测试不运行真实 Redis 或故意制造 GPU 无法回收的故障，物理异常路径的线上验收仍保留边界

从旧训练 rollout40 导出完整轨迹后，固定选取 24 条三轮样本做部署前对照：每个 prompt group 首条，加八条修复／回退案例。72 轮的正确／非正确判定与原训练一致，包含客户端 precheck 和正常服务端失败；一次 HTTP 提交超时已按原 task ID 找回，没有重新提交。它是开发验证集，不是随机总体估计，[样本选择](../../local_artifacts/component_reward_training/replay_selection.json)与[原始结果](../../local_artifacts/component_reward_training/baseline_replay/summary.json)保留在实验目录

## 实现、部署与证据

线上 KernelGym 的 `dev_csl` 源码比初始本地副本新，正式集成需要基于已核对的线上版本，保留后续的输入生成和错误分类修复；旧本地 base 上的开发测试不作为最终部署验收。用户授权的部署节点是 SSH `127.0.0.1:22021` 对应的八卡 A800，不把 H20 上的原型测试视为 A800 部署验收

A800 小型 native 探针的插桩／无插桩输出一致。初次失败定位到 CUDA 工具不在 PATH；仅设置 NVDISASM 的绝对路径仍不够，加入完整 CUDA 12.9 bin 路径后通过。该检查只验证冻结的 sm80 采集器与简单 kernel，最终集成 bundle 仍须验收，[构建与执行证据](../../local_artifacts/component_reward_training/ampere_probe/run-mc25_i1t/result.json)记录具体 hash 和驱动。探针进程已退出，显存查询仍比暖 worker 基线多约 126 MiB，未将其宣称为全部释放或确认成泄漏，也未为此重置 GPU

外部 Grok 完成了设计与实现两轮审查。已采纳字段一致性、摘要完整性、原子过滤及未回收进程的 claim 冻结问题；未采纳直接把 partial 输出路径视为已证实成员的建议，而是通过重新取证补齐记录。[设计裁定](../../local_artifacts/component_reward_training/design_review_adjudication.md)与[实现裁定](../../local_artifacts/component_reward_training/final_review_adjudication.md)保留具体理由，[真实图人工检查](../../local_artifacts/component_reward_training/h20_manual_review_4m/summary.json)区分原始 reward、重新采集的图和仍未完成的训练验收

资源检查时 node69/70 空闲，node53 的两卡和 node64 的全部卡被其它服务占用。没有停止这些服务，也没有静默缩减为一台 rollout；正式启动仍需满足两台 train、两台 rollout

通信及公式约定保存在 [集成合同](../../local_artifacts/component_reward_training/integration_contract.md)。基线参数来源为 [最近 TRLOO 训练](../qwen38/mtp_training.md)，本轮修改前已核对九个关键代码入口与其 resume40 waitfix 快照一致

[训练配置计划](../../local_artifacts/component_reward_training/training_plan.json)已 dump，明确它不是已提交的任务。当前 baseline 的 serving/result-wait 修复被单独固化，组件变化叠加在其上，原来的不相关工作区改动没有合入本次变更。恢复连接后须重新核对 KernelGym 最新源码，再做 A800 HTTP 对照与两台 train、两台 rollout 的实际提交
