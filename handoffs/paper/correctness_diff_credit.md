# 正确性 diff credit：把正确实现的 credit 分给前轮

目标是在 TRLOO baseline 上提高 kernel 正确率。当前用首个正确答案作为参照，通过代码 diff 找出与它接近的前轮实现，再给这些前轮少量额外 credit；FastCredit 保持关闭

<!-- 正式实验从 baseline 相同的官方原始权重开始，RL step 从 0 计数，optimizer、RNG 和数据游标重新初始化；不加载 TRLOO 或 FastCredit 的 step100。四机新作业已启动，先训练到 step100，尚无训练后正确率改善的结论 -->

实现预验证使用过 TRLOO step100 的诊断留样，分奖覆盖为 3.91%，入选修改已人工检查，正确参照已按原任务精度复测。这个数字不代表当前从 0 训练的覆盖率；此前的 step100 续训也不进入正式对照

## 怎样从代码 diff 得到 credit

以一条“第一轮错误、第二轮正确”的轨迹为例。第一轮已经写出了主要计算，只在交叉熵公式中写错了一个符号：

~~~cpp
// T1：错误
total_loss += -(logit_t - max_val) - logf(sum_exp);

// T2：正确
total_loss += -logit_t + max_val + logf(sum_exp);
~~~

T2 修正了符号，其余实现大多沿用 T1。当前方法把 T1 视为一个可以补偿的、接近正确版本的实现：T2 保留原 TRLOO 奖励，T1 获得额外 credit。这里的“接近”指源码接近经过验证的正确版本，不指输出的数值误差很小

具体分为四步：

1. 找到这条轨迹第一次 completed、kernel 正确且非 decoy 的回答，作为参照版本
2. 抽取参照与前轮实际提交的三段代码：CUDA、binding、ModelNew，分别比较 token
3. 每个代码段都达到接近度要求的前轮进入候选；相同代码版本只保留最早满足条件的一轮
4. 直接复用参照轮原有的有效正确性结果，按合格前轮版本分奖，并记录原评测 task_id

比较时忽略注释和不影响语义的排版，保留标识符、数字、字符串、运算符，以及 Python 的缩进层级和语句边界。缩进的空格数量可以归一化，代码属于哪个循环或分支、语句在哪里结束仍需保留。CUDA、binding、ModelNew 分开判断，避免大量相同的 CUDA 或 binding 代码掩盖 ModelNew 的整体重写

设某段前轮代码有 $N_t$ 个 token，正确版本有 $N_*$ 个，SequenceMatcher 找到的有序匹配 token 数为 $M$，差异比例定义为

$$
d = 1-\frac{M}{\max(N_t,N_*)}
$$

三个代码段各自都要求 $d\leq 0.10$，并且至少存在实际 token 变化。只有注释、空白变化，或代码完全相同但评测结果不同的情况，不因此获得额外项

前轮还需要回答 completed、代码段完整、原 baseline 未移除且存在有效 loss mask。截断、padding、decoy 和没有正确参照的轨迹不发额外 credit。参照轮的原评测需要 completed、compiled=True、correctness=True、decoy_kernel=False；稳定性复测用于离线抽查，不作为分奖前置条件

上述取消强制复测的改动已进入仓库代码，尚未同步到本次训练的冻结执行包；按用户要求，未为此停止或重启训练，运行时仍沿用原来的复测流程。该作业后来因主评测客户端超时退出，情况见后文；这次超时发生在原始评测，不是参照复测。代码改动已通过 CPU 回归与真实留样离线分奖检查，见[代码改动验收](../../local_artifacts/paper/correctness_diff_training/original_verdict_code_review.json)

## 一条轨迹具体分多少

每条有合格前轮的轨迹，额外总预算固定为 $\lambda=0.25$，由不同的合格前轮版本均分。代码越长不会得到更多预算

设合格轮集合为 $S$，原 TRLOO 累计回报为 $G_t$：

$$
c_t =
\begin{cases}
1/|S|,&t\in S\\
0,&t\notin S
\end{cases}
\qquad
Y_t=G_t+\lambda c_t
$$

| 情况 | T1 额外项 | T2 额外项 | T3 额外项 |
|---|---:|---:|---:|
| T2 首次正确，T1 符合条件 | 0.25 | 0.00 | 0.00 |
| T3 首次正确，T1、T2 是不同的合格版本 | 0.125 | 0.125 | 0.00 |
| T3 首次正确，T1、T2 是同一合格版本 | 0.25 | 0.00 | 0.00 |
| 没有正确版本，或没有合格前轮 | 0.00 | 0.00 | 0.00 |

额外项只加到对应轮的 $G_t$ 一次，不再向前累加。原始 task reward、TRLOO 回传、finalize、loss mask 都保持不变；首次正确轮与之后轮次保留原回报。这是对额外预算的定义，不是裁剪原累计回报或总 reward

随后仍使用原来的同题、同轮 leave-one-out 计算 advantage

diff 当前用于决定“哪一轮分奖”，训练仍使用该轮原有的完整 response loss mask，包括 thinking；对应的学习风险集中在后文讨论

## 对照实验怎样设置

实验要分别回答：给接近正确的失败回答补偿是否有用，以及 diff 选轮是否比不看接近度的分配更好

| 组别 | 额外 credit 的分配 |
|---|---|
| TRLOO baseline | 计算同样的诊断信息，但不加奖 |
| 等预算随机对照 | 在兼容轨迹之间打乱整套分奖向量，不按 diff 接近度挑选受奖对象 |
| 正确性 diff credit | 按上述代码接近度与不同版本分奖 |

随机对照只在同一道题、首个正确轮相同、可用前轮集合相同的轨迹之间交换分奖向量。例如把一条轨迹的 $(0.125,0.125,0)$ 整体交给另一条符合条件的轨迹，而不是把两个 0.125 分别乱放。这样可以同时保留各轮总预算、受奖轮数量和每条轨迹的额外预算上限

三组使用相同的原评测有效性条件和前轮入选资格；离线稳定性抽查不改变各组分奖。预算匹配在各组自己的当前 rollout 内完成，不代表策略分化后各组总预算始终相同；实际受奖对象可能相同，需要记录重合程度。下述历史留样的统计仍保留当时的强制复测口径

| 设置 | 当前配置 |
|---|---|
| 共同起点 | Qwen3.8-27B 官方原始权重，与 baseline 的原始初始化一致；全新 optimizer、RNG 和数据游标，RL step0 |
| 训练长度 | 先做 100 次新更新，与已有 baseline step100 的训练步数对齐 |
| 数据 | prompt_v4_1；每批 16 题，每题 16 条轨迹 |
| 三轮总 context | 24K / 32K / 40K |
| 训练 / rollout | BF16 / FP8；保留 MTP3 rollout 与 MTP1 teacher forcing |
| 优化设置 | 原 TRLOO、predictive DPPO、1e-6 学习率、轨迹打包及原入训 mask |
| 新参数 | 额外预算 0.25；各代码段最大差异比例 0.10 |
| 资源 | 前期两台 train、一台 rollout；四机时两台 train、两台 rollout |
| Checkpoint | 每 20 步保存，保留最近两代可恢复 checkpoint；保存里程碑检查容量 |

同一数据起点和种子固定初始顺序；fully-async 的完成顺序与动态过滤会随策略变化，因此还要保存实际接受的题目身份，不能声称三组消费题目逐条完全相同。原 TRLOO 基础 reward 保持不变，仍包含原有性能与 coverage 项；正确性 diff 决定的是新增 credit，不是把整个 reward 改成二元 correctness

## 预验证的训练池发现了什么

本节来自 TRLOO step100 的诊断留样，用于检查实现与分奖覆盖，不是当前 step0 新训练的结果

首先检查了原来的整轮修复包门槛：要求 T1 错、T2 错、T3 首次正确，再检验 T2 的修改是否必要。这个门槛在新池中的实际分奖覆盖为零，因为唯一第三轮首次正确轨迹的第二轮被截断了

| 首次正确的位置 | 占全部轨迹（%） |
|---|---:|
| T1 | 69.53 |
| T2 | 7.81 |
| T3 | 0.39 |
| 没有 completed 的正确回答 | 22.27 |

这说明，应当首先覆盖“第一轮已写出大部分正确实现、第二轮修复后通过”的样本，而不把训练门槛限定在三轮才完成的修复链上

| 指标 | 分母 | 比例（%） |
|---|---|---:|
| 原三轮必要包规则可发奖 | 全部轨迹 | 0.00 |
| 当前 diff 规则静态入选 | 全部轨迹 | 3.91 |
| 正确参照复测通过后的分奖覆盖 | 全部轨迹 | 3.91 |
| 随机对照与 diff 受奖对象重合 | diff 受奖前轮 | 90.00 |

这仍只是一个真实训练批次，不能外推为全训练集覆盖率。随机对照也比看上去更相似：本批只换掉了一组受奖对象。短训练即使有变化，也需要谨慎区分“额外补偿有效”和“diff 筛选本身有效”

本批入选的都是 T1，参照都是 T2；全部入选 diff 已人工检查。按修改实际解决的问题分类：

| 修改解决的问题 | 占入选前轮（%） | 可见改动 |
|---|---:|---|
| CUDA stream 类型导致编译失败 | 20.00 | 给 cudaMemsetAsync 的 void* 转成 cudaStream_t |
| 交叉熵公式错误 | 20.00 | 修正 log 项符号，或保留被 exp 覆盖前的目标 logit |
| GEMM / elementwise 计算语义错误 | 20.00 | 修正 cuBLAS 转置标志，或把 bias 放回 ReLU 之前 |
| 旋转索引方向错误 | 20.00 | 修正 90° 与 270° 对应的坐标映射 |
| 调用不存在的数学函数 | 10.00 | 将 log1f 改为合法的 logf(1 + expf(x)) |
| binding 头文件不符合环境规则 | 10.00 | 移除 binding .cpp 中的 CUDA runtime 头文件 |

正确参照复测覆盖 FP32、BF16 和 FP16 任务，均使用原精度与五次正确性检查，未出现参照复测失败。旧的“所有请求统一 FP32”做法没有沿用

在这批固定数据上加入额外项后，受奖前轮平均 advantage 从 -0.96 变为 -0.72，仍有 70.00% 为负。这证明的是分奖信号已经改变，不是模型已经学得更好

## 当前验证与尚需观察的结果

当前运行的是 qwen38-correctness-diff025-from0-resume20-20260915，按用户授权从本实验自己的 step20 恢复，node69/70 训练、node53/64 rollout。累计完成 82 次更新，恢复后新增六十二次，第 83 次更新进行中。最新 checkpoint 为完整的 step80。近期 loss 和梯度有限，四节点 KernelGym 健康，恢复以来日志未发现生成异常、客户端评测超时或 OOM。已归档较早的 step60 并释放其源副本，空间足够保存 step100；训练参数和分奖规则未变，见[step80 保存与现场检查](../../local_artifacts/paper/correctness_diff_training/fresh0/resume_step20/step80_checkpoint_review.json)

恢复时已在 GPU 上成功加载 iter_0000019、恢复 optimizer 和数据游标，并按 resume 路径加载 RNG；actor 与 MTP 权重已同步到四个 rollout 引擎，见[恢复验收](../../local_artifacts/paper/correctness_diff_training/fresh0/resume_step20/startup_verification.json)

原作业 qwen38-correctness-diff025-from0-20260915 从 baseline 相同的官方 release 开始，finetune=True、no_load_optim=True、no_load_rng=True、start_rollout_id=0，没有沿用旧 RL 状态。此次恢复沿用同一个从 0 实验的 checkpoint，没有加载历史 TRLOO 或 FastCredit step100；仍以累计 100 次更新为目标

原始 BF16 release 在两台训练机的全文件 SHA256 一致，metadata 不含 optimizer，来源 HF 路径为官方 Qwen3.8-27B。fresh 与显式 resume 的初始化回归已通过；新作业使用独立目录，先前诊断的模型更新和数据游标均未沿用，见[初始化约定](../../local_artifacts/paper/correctness_diff_training/fresh0/initialization_contract.json)、[权重校验](../../local_artifacts/paper/correctness_diff_training/fresh0/actor_release_verification.json)与[实际参数](../../local_artifacts/paper/correctness_diff_training/fresh0/formal_diff_config.json)

原作业完成 32 次更新后，下一批 rollout32 的主评测客户端超时触发保护，以 FAILED 退出。该批在转换成训练数据之前被拒绝，没有进行第 33 次更新；故障排查阶段未自动重提，随后由用户明确要求恢复

最新保存为 step80（iter_0000079），两节点分片、全局 metadata 文件范围、optimizer、RNG 和数据游标均已核查。较早的 step60 已完整归档到 node70，逐文件通过源端与归档端 SHA256 比对，归档的数据游标和恢复标记也已核对；随后只释放了 step60 的两机源目录。最近两代是可恢复的 step60 归档与 step80 分布式 checkpoint，之前的 step20、step40 归档保留，见[保存与容量处理](../../local_artifacts/paper/correctness_diff_training/fresh0/resume_step20/step80_checkpoint_review.json)

原作业中断时，最后 12 次已完成更新没有 checkpoint，此次从 step20 重跑；该恢复起点的核查见[step20 保存验收](../../local_artifacts/paper/correctness_diff_training/fresh0/step20_checkpoint_review.json)与[失败后的 checkpoint 复核](../../local_artifacts/paper/correctness_diff_training/fresh0/terminal_failure_20260915_1006.json)

恢复使用原冻结执行包，四节点源码与推理运行时哈希一致；训练及分奖参数逐项比对通过，只增加恢复加载项并更换留样目录。原日志和留样保留，checkpoint 继续保存到同一实验目录，最近两代保留策略不变；尚未启用取消参照复测或提交重试的新行为，也没有重新部署 KernelGym。只读监控已重新接入三小时心跳、保存里程碑与作业终态通知，见[恢复参数](../../local_artifacts/paper/correctness_diff_training/fresh0/resume_step20/formal_diff_config.json)与[运行来源核验](../../local_artifacts/paper/correctness_diff_training/fresh0/resume_step20/runtime_provenance.json)

用户已授权后续可恢复故障的自主续训：训练中断后，主 Agent 先诊断并核查入训污染范围；确认运行条件、四节点同步、资源和最近有效 checkpoint 满足恢复要求后，直接 resume，不再逐次请求确认，分奖与训练超参保持不变。若条件尚未恢复，继续安排只读监控唤醒，待恢复后复核；每次决定和恢复结果均 page-user。监控程序本身仍只发送通知，不执行停训或重启；后续明确的用户停训指令优先，不因此覆盖旧证据、重复提交或影响其它任务

step80 保存后，node69 只剩约 117 GiB，低于下一代约 172 GiB 的写入需求。释放已归档的 step60 源副本后，node69 可用空间恢复到约 289 GiB，node70 约 643 GiB，足够保存 step100。最近两代指两代可恢复状态，不要求源目录与归档各保留一套；本次没有停止训练、修改训练参数或删除共享模型

下表各批次均通过源码与任务绑定、分奖资格、预算及 target 重算检查，截断轮没有获得额外项。恢复后的 rollout30、41、52、61、64、79、81 各人工查看两条受奖轨迹的源码 diff，重算接近度，并抽查实际存在的截断样本。rollout81 还在 CPU 上重算了整批筛选记录，与原记录一致；该检查复用原有参照判定，没有新增评测。各批题目不同，预取数据也不等于已经入训，且未做独立反事实验证，下面的变化不能解释为训练效果

| 留样 rollout | T1 Correct（%） | T3 Correct（%） | Best Correct（%） | 分奖轨迹覆盖（%） | 总截断（%） |
|---|---:|---:|---:|---:|---:|
| 0 | 25.39 | 63.28 | 76.17 | 36.72 | 4.30 |
| 21 | 50.00 | 52.34 | 73.05 | 14.06 | 8.59 |
| 23（恢复后预取） | 42.19 | 56.25 | 69.14 | 12.50 | 10.55 |
| 30（恢复后预取） | 52.73 | 62.50 | 76.95 | 11.33 | 11.72 |
| 41（恢复后预取） | 64.84 | 66.80 | 87.11 | 12.11 | 6.25 |
| 52（恢复后预取） | 31.25 | 53.13 | 65.63 | 10.16 | 3.26 |
| 61（恢复后预取） | 65.23 | 73.44 | 87.11 | 12.89 | 0.39 |
| 64（恢复后预取） | 55.08 | 76.56 | 83.20 | 12.11 | 1.69 |
| 79（恢复后预取） | 53.91 | 61.72 | 71.09 | 9.77 | 0.00 |
| 81（恢复后预取） | 52.34 | 55.86 | 67.58 | 6.64 | 1.04 |

Correct 使用原始 turn 编号、全部轨迹分母及 completed、compiled、correctness、非 decoy 条件。发现现有日志会先跳过 speedup 缺失的轮次，再按剩余轮次顺序计算 best_by_turn，可能把 T2 正确计入首轮；本表已直接从留样重算，不沿用该日志口径。该问题不改变按原始 turn_idx 分配的 credit，本次未修改训练代码

人工复核了两批中的六条完整轨迹。rollout0 g167 的 T2 修好 binding 后，把原先可用的 self.flatten(x) 改成 self.flatten.flatten(x)，引入 AttributeError；T3 恢复原调用后正确，T2 仍获均分的额外项。这是当前规则会把修复与自造错误一起补偿的真实案例，尚未做反事实 replay，见[分奖审计与人工样例](../../local_artifacts/paper/correctness_diff_training/fresh0/step20_rollout_audit.json)

恢复后的 rollout30 中，g10205 的 T3 用临时数组保存新的 LSTM 隐藏状态，再统一写回，修正 T2 循环内原地更新造成的新旧状态混用；T2 获得 0.25，T3 保留原奖励。另有 g10652 的 T2 原评测正确、参照复测却报输出不一致，当前冻结版因此取消该轨迹的额外项，原 TRLOO 回报不变；该不一致尚未额外 replay 归因。源码差分与原始判定保存在上述心跳证据中

目前只启动过从 0 开始的 diff 组，尚未另起新的 baseline 或随机对照训练。用户确认当前链路正常后，已先恢复训练；提交恢复逻辑的缺口仍待单独修改，入训前 fail-fast 保护继续保留。后续观察正确率、截断和分奖表现，当前没有训练后正确率改善的结论

本次运行曾出现四节点访问 KernelGym 超时及返回字段缺失，随后链路自行恢复，远端 API 进程没有重启。缺少 metadata 的响应触发客户端 abort，含 aborted 样本的题目组会在进入输出队列前被整体重排。rollout29、30，以及之后留存的 rollout31 已分别检查，未发现 aborted、客户端超时或 RESOURCE_ERROR 样本进入这些留样，分奖检查通过；这些检查没有覆盖当时尚未收齐的 rollout32，见[访问异常与入训检查](../../local_artifacts/paper/correctness_diff_training/fresh0/infra_access_20260915_0913.json)和[rollout31 心跳检查](../../local_artifacts/paper/correctness_diff_training/fresh0/heartbeat_1789466538.json)

最终导致退出的是 g10085 的 T3 原始评测：断链期间提交请求，提交响应超时；客户端继续轮询，等待约 40 分钟仍没有拿到有效结果，因而返回客户端超时。训练与下一批 rollout 并行，第 32 次更新完成后，主进程等待下一批数据时才收到这次异常。保护逻辑在入训前拒绝整批，避免把“没有评测结果”产生的零分用于原 TRLOO 回报和组内比较；失败现场四节点 /health 已恢复，GPU 随作业退出释放

进一步核对 SSH、转发器和服务端日志后，原因可以拆成三个环节：

| 环节 | 已确认的现象 | 对本次故障的影响 |
|---|---|---|
| 故障时段的 SSH 转发异常 | SSH 连接超时、握手失败，node69 转发器记录后端连接被拒绝；API 与 Redis 进程持续运行 | 可能阻断请求提交，但尚未将目标 task 的 POST 与具体失效的 SSH 连接一一对应 |
| 转发恢复受阻 | node69 的部分端口持续报 Address already in use，新的反向转发反复绑定失败 | 重连不能立即恢复所有后端；旧 SSH 会话未释放端口是合理解释，但当时的端口占用 PID 未留存 |
| Slime 没有恢复未确认的提交 | 三次 POST 失败后只查询原 task；网络恢复后仍不补交，直到客户端期限耗尽 | 将一次提交失败拖成约 40 分钟等待，最终触发整批 fail-fast |

目标 task 在覆盖故障时段的 API 日志中没有完整请求正文记录，相邻 task 有；该 task 在运行期间的 status 查询全部返回 404。当前部署的 KernelGym 会把仍在运行的 workflow 返回为 processing，因此不能继续用旧客户端注释里的“正常运行的父任务也一直 404”解释本例。现有证据高度指向请求未成功提交，而非候选 kernel 长时间执行；但缺少独立的接收事务记录，仍保留这个归因边界

用实际冻结客户端做了无网络的故障模拟：提交阶段均失败、随后恢复连通性时，客户端持续查询不存在的 task 并等到超时；若模拟任务已经被服务端接受、只是提交响应丢失，则可以正常取回结果。这确认了当前恢复逻辑只覆盖后一种情况。后续应补上未确认提交的安全恢复或整题组重采集，保留入训前保护；单纯增加 timeout 或取消保护都不能补出缺失的评测结果

用户指出隧道正常后，从实际 node69 执行容器重新检查，/health 同样正常，而旧 task 的 status 与 results 仍返回 404。当前连通性、故障时段的转发异常、以及单个请求失败的具体原因需要分开判断：历史转发日志不能证明隧道现在仍故障，也不足以单独确定目标 POST 为何失败；已经独立复现的是客户端没有恢复未确认提交的能力

更上游网关为何断线仍未定位，故障排查没有修改 SSH、KernelGym 或训练代码。此次中断不构成方法有效或无效的证据，不能据此将此前所有更新判为污染。任务时间线见[退出检查](../../local_artifacts/paper/correctness_diff_training/fresh0/terminal_failure_20260915_1006.json)，服务端对照、转发日志、当前复查和客户端复现见[提交恢复根因排查](../../local_artifacts/paper/correctness_diff_training/fresh0/submission_recovery_diagnosis_20260915.json)

### 已结束的续训诊断

此前做过从 TRLOO step100 开始的回放和短续训，仅作为诊断保留。一次续训因 KernelGym 访问中断停止；恢复后的 r2 完成一次更新后，按用户要求切换为 step0 新训练而停止。它们不作为从 0 对照的训练结果

诊断首跑前两个批次的完整留样已核查，额外项没有发给截断回答、正确参照或后续轮，也没有重复回传；用于发奖的正确参照均通过复测

| 续训诊断 rollout 批次 | 分奖轨迹覆盖（%） | T1 截断（%） | T2 截断（%） | T3 截断（%） | 总截断（%） |
|---|---:|---:|---:|---:|---:|
| 100 | 5.86 | 8.20 | 5.08 | 8.59 | 7.29 |
| 101 | 4.69 | 8.20 | 14.84 | 10.55 | 11.20 |

两个批次题目不同，且来自 fully-async 采集，不能把这两行之差解释为训练效果。独立正确率计数使用全部轨迹、completed 且非 decoy 的口径，保存在[诊断留样审计](../../local_artifacts/paper/correctness_diff_training/formal_initial_rollout_audit.json)，不混用日志中分母不同的条件正确率

最初两批检查时，KernelGym 健康且没有待排队任务。留样中有两次 300 秒任务超时，以及七次候选运行错误后的 60 秒 memcheck 超时；两类分别保留，相关超时轮均没有额外 credit。尚未独立 replay 这些超时，不能进一步归因

后续心跳发现四节点均无法访问 KernelGym：20211 转发器仍在，但其八个后端端口全部消失，SSH 反向入口也无法完成握手。rollout110 中 g40363 的第三轮等待主评测超过客户端期限，得到合成零分，仍带有效 loss mask 进入第十一次训练计算。没有额外 credit 并不足以防止原 TRLOO 回报受到未完成评测的影响，因此已停止该作业并排除出正式对照；日志与留样保留，不能把该次中断解释为方法有效或无效

三组共用的代码已补上主评测客户端超时整批报错保护，在 LOO 前阻止更新；原有候选 kernel 超时、sanitizer 超时与原 TRLOO off 模式未改。CPU 合约检查和真实受影响批次的三模式离线检查均通过，保护已随新快照同步到四节点并用于重跑，见[故障与保护验证](../../local_artifacts/paper/correctness_diff_training/kernelgym_outage_audit.json)

恢复验收分开检查了两条路径：新生成的两条三轮轨迹均完成正确性检查；随后重放 g36055 的三轮代码并复测正确参照，复现编译失败后的修复，参考测速有效，额外项为 $(0.25,0,0)$。这验证的是执行与分奖链路，没有进行新的 optimizer 更新，见[恢复验收](../../local_artifacts/paper/correctness_diff_training/attempt2/recovery_acceptance.json)

新生成的 BCE 样本还暴露出一个独立问题：正确性通过，但参考测速返回构造参数 TypeError，reference_runtime 为 -1。参数解包不一致目前只是推测，尚未修复或统计其全数据覆盖；因此完整奖励链路使用 g36055 补充验收，未将 BCE 的正确性成功当作测速成功，也未在本次重跑中改动数据或 KernelGym

正式判断以各轮正确率、Best Correct、首次做对位置为主，同时观察截断率、回答长度、diff 覆盖、实际分奖和 advantage 分布。reward 上升或 loss 有限不能替代正确率改善

## 当前方法的问题与验证边界

当前方法用源码接近度，为后来修复成功的失败前轮补偿。判断它是否值得继续，重点是受奖对象是否合适、额外信号鼓励了什么行为，以及最终正确率是否改善。下面分别说明代码已确定的机制、真实样本中已观察到的现象和仍需训练验证的风险

### 源码接近度无法直接表示正确性贡献

局部 typo、关键算法错误和无关改名都按 token 差异计算，规则没有区分它们的语义重要性。入选说明前轮源码接近正确版本，不能据此把该轮每一行代码和每一步推理都认作有用贡献

人工检查确认入选样本包含真实的编译、公式和索引修复，也发现 g36330 的修改中混有与主要修复不同的检查删除。现有证据支持把接近度作为低成本代理信号，尚不足以给出贡献归因的 precision，见[入选样本人工检查](../../local_artifacts/paper/correctness_diff_training/near_manual_review.md)

### 均分没有区分前轮的实际进展

只要 T1、T2 都接近首次正确的 T3，且是不同版本，就均分额外预算。例如 T1 的 dtype 和索引都错，T2 修好 dtype，T3 再修好索引后正确；若两个前轮都符合条件，仍各分一半，T2 不会因为多解决一个错误而获得更多

此前续训诊断人工检查了六条轨迹的三轮代码和反馈，其中 g36747 体现了这个问题：T1 的 reduction 漏加 warp 0，T2 改用 double 累加却仍漏加，T3 将循环起点从 1 改为 0 后正确，T1、T2 各获 0.125。这个案例证明均分按预期执行，没有证明 T2 的精度修改有必要。更一般地，只做无关变量改名也可能形成一个新的合格版本；源码去重能够排除完全相同版本，不能排除没有实际进展的版本

### 新增项可能偏向先错后修

T1 直接正确时没有额外 diff credit；T1 接近正确、T2 修复通过时，T1 可以获得额外补偿。两者都保留原 TRLOO 回报，因此不能只看额外项就断言先错后修的总奖励更高，但新增项本身确实偏向有修复机会的轨迹

此外，补偿在同题、同轮 LOO 之前加入，会提高其他回答面对的比较基准。一个没有加奖的早期正确回答，原 reward 未变，相对 advantage 仍可能下降。这是新增 credit 与原 TRLOO 的相互作用；是否真的导致模型延迟做对，需要联合观察 T1 Correct、Best Correct 和首次正确位置，不能从奖励公式直接断言

### 局部筛选对应的是整轮训练信号

筛选依据可能只是最终代码中的一个符号，但额外 credit 会改变该轮全部有效 response token 的 advantage，包括保留代码、后来被修掉的错误代码，以及无关或冗长的 thinking。它可能同时强化这些内容，或减弱原本对它们的惩罚

因此，当前方法没有把新增信号精确落到正确实现的部分，也无法阻止一轮中的有效修改和新制造的错误一起受奖。整轮 mask 的行为已核实；是否因此增加回答长度、截断或无效修改，仍是待验证的训练风险

### 覆盖偏向局部修补，容易漏掉有用但未完成的实现

各代码段分别设置差异门槛，使较短 binding 的变化也能阻止整轮入选。后轮补充新代码同样增加差异：g36426 的前轮 CUDA token 全部被后轮有序保留，但新增代码使该段差异达到四成以上，仍然出局。当前衡量的是整个实现离正确答案有多近，前轮代码的保留程度只反映其中一部分

在同一诊断批次、固定其余入选条件后，阈值敏感性如下。分母是已有正确参照且满足其他比较条件的前轮，不是全部轨迹，也不是全训练集

| 各代码段差异阈值（%） | 可比较前轮的入选比例（%） |
|---|---:|
| 5.00 | 47.06 |
| 10.00 | 58.82 |
| 20.00 | 64.71 |
| 40.00 | 76.47 |

这些结果由[诊断逐轮记录](../../local_artifacts/paper/correctness_diff_training/near_annotated_capture/ledger.json)中的分段差异重新计算，只说明覆盖随阈值怎样变化，尚未验证哪个阈值最有利于学习。完整三段代码、有效 mask 和正确参照的要求还会进一步限制覆盖；全轨迹都没有做对的难题，仍只能依靠原 TRLOO 获得训练信号

### 现有对照与人工检查还不足以证明有效

阈值与随机对照的预验证来自一个 TRLOO step100 诊断批次；新增的 step0 训练留样用于检查分奖执行，尚未形成完整训练对照，不能把单批覆盖直接外推到整个训练集。随机对照又受到同题、首个正确轮和可用前轮集合的共同限制，大部分奖励没有可替换的对象

| 随机打乱时的情况 | 占 diff 受奖前轮（%） |
|---|---:|
| 所在小组的可用前轮全部入选，交换后仍然受奖 | 80.00 |
| 小组内存在未入选前轮，可以改变受奖对象 | 20.00 |
| 本次打乱后实际仍然受奖，包含上述固定部分 | 90.00 |

可交换部分在这次打乱后保留了一半；按这些分组计算，随机重合的期望也与实际重合一致。因此，高重合主要来自对照可交换空间太小，不能当成随机分配独立验证了 diff 判断

人工检查已经覆盖该批入选修改，但对被拒绝前轮的系统检查仍不足，尚不能完整回答哪些该奖却漏了、哪些入选却不值得奖。已有正确性结果也受有限测试输入约束，离线复测不构成所有输入上的正确性证明。当前证据支持继续验证这一启发式，尚不足以证明它提高了正确率，或优于等预算随机补偿

后续判断应优先检查受奖轮是否带来实际进展、是否偏向先错后修，以及整轮信号是否夹带错误行为；单独提高 diff 阈值不能解决这三个问题

## 先前整轮对照留下的证据

整轮对照仍用于分析，不再作为当前训练的发奖门槛。历史留样中，它能构造对照的覆盖为 1.30%；这些对照发现了三种情况：

- g154：dtype API 修复与对角地址修复都需要，去掉前一个修复会重新编译失败
- g43818：保留在正确答案里的设备处理并非必要，只加最后的 size_t 地址修复就正确
- g76：必要修复包也夹带了新错误；只修第一版的 dtype、不加第二版的 grid 上限，就已经正确

因此，旧整包必要性结果不能当作当前源码接近度规则的 precision。之前的完整分析与数值保留在[历史分析快照](../../local_artifacts/paper/correctness_diff_training/previous_whole_bundle_report.md)，执行证据见[历史训练留样入口](../../local_artifacts/paper/repair_credit_validation/trloo_training_turn_bundles/README.md)；更早评测池的局部 diff 检查见[评测池证据](../../local_artifacts/paper/repair_credit_validation/trloo_step100_ctx32k48k64k/README.md)

## 代码与实验入口

- 在线实现：[correctness_diff_reward.py](../../examples/kernel_agent/correctness_diff_reward.py)
- CPU 合约检查：[test_kernel_agent_correctness_diff_reward.py](../../tests/test_kernel_agent_correctness_diff_reward.py)
- 新池分奖、参照复测与随机对照：[汇总](../../local_artifacts/paper/correctness_diff_training/near_annotated_capture/summary.json)、[逐轮记录](../../local_artifacts/paper/correctness_diff_training/near_annotated_capture/ledger.json)
- 人工检查与训练张量抽查：[检查记录](../../local_artifacts/paper/correctness_diff_training/near_manual_review.md)
- 实验设置与启动入口：[train_arm.sh](../../local_artifacts/paper/correctness_diff_training/train_arm.sh)
- 正式实际 argv 与源码身份：[step0 新训练配置](../../local_artifacts/paper/correctness_diff_training/fresh0/formal_diff_config.json)
- 新采集使用同步 debug-rollout 路径，正式训练使用原 fully-async 路径；大文件存放在实际执行容器的 /nfs/LOCAL，不把宿主终端同名路径当作已经同步
- 只使用既有 KernelGym dev_csl 服务，未修改或重新部署 KernelGym；代码、配置、权重及数据按实际执行节点核验

node53 的原占用提前自行结束，无需杀掉其 GPU 进程。首次四机启动在提交前发现 node64 旧 SGLang 副本缺少 MTP refit 补丁，随后采用实验私有副本，四节点逐文件 hash 与补丁检查均通过；共享运行时与 KernelGym 没有改动

旧整包 hook 的 Grok 审阅、新规则的代理 Kimi 审阅均因连接问题未获得最终结论，原生 Kimi 又因额度限制无法调用。没有记为独立审查通过，也没有更改付费或账户设置；替代检查与边界见[审阅记录](../../local_artifacts/paper/correctness_diff_training/near_review_disposition.json)

只读[正式监控](../../local_artifacts/paper/correctness_diff_training/fresh0/formal_watcher_state.json)检查 step100 保存、作业状态和 KernelGym 健康，每三小时 Queue 心跳；每 20 步保存后通知主 Agent 检查分奖与磁盘容量，链路连续三次不可达会提前通知。监控不会部署服务、停止作业或自动启动后续对照，处置仍由主 Agent 执行
