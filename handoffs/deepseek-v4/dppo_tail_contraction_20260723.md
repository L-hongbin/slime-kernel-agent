# DPPO clipfrac 尾部收缩：固定 prompt 因果实验与 generation 端点边界

更新时间：2026-07-24

## 摘要

正式 r21 的 `pg_clipfrac` 在 step75 到 step95 从 0.1499% 降到 0.0454%。
它不是 clip 方向逐步反转，也不是 rollout 权重落后一个 checkpoint：真正变化的是
rollout 与 trainer 的 Top-K-plus-tail KL 超过 0.15 的 token 数。step95 后该尾部
回升，step164 的 `pg_clipfrac` 为 0.0801%；因此这是一段收缩，不是持续归零。

固定 prompt 实验确认 checkpoint 介入具有真实总效应。同一组 16 个 DrKernel
prompt 和逐 slot seed 下，iter74 与 iter94 的实际 generation `outside@0.15`
分别为 159/84,251=0.1887% 和 38/77,987=0.0487%，下降 74.18%；prompt-cluster
bootstrap 的绝对变化 95% CI 为 [-0.1960, -0.0884] 个百分点。这与正式
step75→95 的 70.59% 下降几乎复现，也不需要改变 prompt 身份或把 oversampling
cap 从 32 改为 24。共同 response 长度、前 1024/1742 token 和 prompt 等权检查
都保留该效应，排除了“iter94 只是少生成了后段 token”及单个 prompt 支配。

证据同时把根因边界改写得更窄，但没有给出单一 kernel 根因。SGLang 把相同
response 做 full-prefix 重算后，iter74/iter94 的尾部只剩 1/84,251 和
2/77,987；generation→full-prefix 的 checkpoint 相关修正几乎逐量解释了实际
收缩。也就是说，已验证的系统级根因边界是 checkpoint 敏感的 production-like
增量/DSPARK generation 端点相对 Megatron replay 的差异，而不是 full-prefix
静态 policy effect。此前在短 prompt、base model、eager 下闭合的 shape-stable
修正组合，在当前 gamma=3、CUDA graph、LoRA-capable server 上不能闭合；精确差异
仍可能位于 CUDA graph、gamma=3 target state、LoRA-capable 路径或尚未覆盖的
shape-sensitive kernel 中。

当前决策是不改 DPPO loss、clip 阈值或权重同步，也不直接打开未做性能验收且未
闭合当前端点的诊断修正。下一实验应先对 base/no-LoRA-capability server 做匹配的
gamma=3 graph on/off，再逐项加入 LoRA capability 和 adapter，并在首个失败臂的
同一 verify boundary 比较 full logits 与 KV/hidden state。reward 质量、任务能力和
GSPO 选择不在本文范围内。

## 1. 指标为什么会突然缩小

Predictive DPPO 对每个 response token 比较 rollout 与 trainer 在
`TopK(rollout, 20) ∪ {sampled token}` 上的分布，并把未保留词表合成一个 tail
bucket。`dppo_outside` 是粗粒度 KL 大于 0.15 的 token 比例；只有 advantage
缩放后的 sampled-token 一阶更新方向还会增大该 KL 时，token 才计入
`pg_clipfrac`。因此：

`pg_clipfrac = dppo_outside × P(clipped direction | outside)`。

steps75-110 中，第二项均值 0.4644、范围 0.4136-0.5166，与 step 的相关系数
仅 -0.0487。变化来自有多少 token 穿过 KL=0.15，而不是 tail token 的更新方向
越来越安全。step75→95 平均 Top-K KL 只下降 18.16%，`dppo_outside` 却下降
70.59%；硬阈值把 mismatch 分布的中等移动放大成 tail count 的大幅变化。

这决定了后续实验的直接 endpoint 必须是实际 generation 的
`outside@0.15`。平均 logprob 差、full-prefix KL 或 route equality 都只能作为
机制控制，不能替代它。

## 2. 在线轨迹排除了什么

### 收缩会反弹，但反弹跨越了采样制度边界

每次 restart 的重复 step 取较新的 lag-0 值：

| 窗口 | oversampling cap | `dppo_outside` | 每 step 斜率 | `pg_clipfrac` | 每 step 斜率 |
|---|---:|---:|---:|---:|---:|
| 75-94 | 32 | 0.3441% → 0.1066% | -0.01234 pct-pt | 0.1499% → 0.0482% | -0.00593 pct-pt |
| 95-164 | 24 | 0.1012% → 0.1707% | +0.00094 pct-pt | 0.0454% → 0.0801% | +0.00048 pct-pt |

step95 后的整体斜率为正，但 step131-150 的 outside 斜率只有
`+0.000123` pct-pt/step、step correlation 0.061，说明后半段已进入平台和批次
波动，而不是持续单调回升。step155 达到阶段峰值 0.1955%，step164 回到
0.1707%；它约为 step75 的一半，但已接近或高于 step85 的量级。

step85 与 step95 的 1,370 项参数 dump 只有 `start_rollout_id` 和
`over_sampling_batch_size` 不同，后者从 32 降到 24。75-94 的收缩全部发生在
cap=32 下，不能由后续改动造成；95 后反弹则混合了 checkpoint、dataset cursor
和候选池效应。若要解释反弹或选择 cap，仍需同 checkpoint、cursor、seed 的
rollout-only 32/24 A/B，不能从在线 before/after 分配因果量。

正式 run 在 step164 checkpoint 提交完成后收到外部 SIGTERM。日志没有在信号前
出现 model、数值、NCCL、CUDA 或 reward exception，host kernel journal 也没有
对应 OOM/Xid；`latest_checkpointed_iteration.txt` 为 164。停止机制已确认是
SIGTERM，发送者未知。这不影响截至 step164 的指标，但不应写成训练自然结束。

### 一步权重时延不是主因

三个 resume 在首个 rollout 前都加载 checkpoint、完成 rollout 权重同步并报告
weight version 1：

| 训练 step | rollout 已加载策略 | `dppo_outside` | `pg_clipfrac` |
|---|---:|---:|---:|
| 75 | iter74 | 0.3441% | 0.1499% |
| 85 | iter84 | 0.1580% | 0.0794% |
| 95 | iter94 | 0.1012% | 0.0454% |

step85 的连续值与 lag-0 restart 值只差约 3.3%，去掉一步时延不会恢复到 step75。
因此不应通过改变同步顺序来处理该收缩。

### 在线相关性不能分开 checkpoint 与 context

steps75-110 的 outside 与平均 Top-K KL、sampled-logprob 绝对差、centered support
差的 level correlation 分别为 0.871、0.621、0.622，但这些量都随 step 自相关。
做一阶差分后，后三者的相关性降到 0.068、-0.144；response length 反而为
-0.721，分窗口去趋势后仍为 -0.528。它可能表示批次构成、token 位置稀释或
serial noise，聚合日志不能区分。

iter74→iter94 的 LoRA A factor 仅变化 0.205%，B factor 与有效 `B@A` 更新均变化
37.3%，cosine 约 0.953。checkpoint state 足够不同，但在线轨迹仍无法判断 tail
变化来自 policy 对固定任务的作用，还是不同 checkpoint 诱导出的 response/context。
这就是固定 prompt 介入要回答的问题。

## 3. 固定 prompt 证明 checkpoint 总效应存在

实验固定 seed 20260723 选择的 16 个真实 DrKernel prompt，覆盖 elementwise、
reduction、normalization、convolution、embedding、batched matmul、padding、split
和 dropout。每个 prompt 使用相同的逐 slot seed，但两个 checkpoint 可以生成不同
continuation。实际 generation 对角线保留正式指标的行为分布；估计按 token 加权，
95% CI 对 16 个 prompt cluster 成对 bootstrap 10,000 次。

| generation endpoint | iter74 | iter94 | iter94−iter74（95% CI） |
|---|---:|---:|---:|
| `outside@0.15` | 0.1887% (159/84,251) | 0.0487% (38/77,987) | -0.1400 pct-pt `[-0.1960,-0.0884]` |
| mean Top-K KL | 0.0096868 | 0.0077778 | -0.0019090 `[-0.0028894,-0.0009308]` |
| mean abs(sampled-logprob delta) | 0.0601435 | 0.0537572 | -0.0063863 `[-0.0121785,-0.0003848]` |

outside 相对下降 74.18%，平均 Top-K KL 下降 19.71%，几乎复现正式
step75→95 的 70.59%/18.16%。固定实验绕过 dynamic filter 和 cap，因此 original
contraction 不需要 prompt 身份或 32→24 候选池改动。

长度与少数 prompt 也不能解释结果：

- prompt 等权 outside 下降 75.87%，绝对变化 CI
  `[-0.2132,-0.0990]` pct-pt；13 个 prompt 下降、2 个相同、1 个上升；
- iter74/iter94 分别有 14/16 与 15/16 个 prompt 出现越界 token，iter74 最大单个
  prompt 只占 11.95% 的事件；
- 每对 response 截到较短长度仍下降 73.43%；
- 所有 prompt 只取前 1024 或 1742 token 时分别下降 81.40% 和 82.50%，CI 均排除 0；
- 越界点从第一 response decile 到最后一 decile 都存在。

人工解码了最大和早期越界点。它们位于正常的 CUDA/C++ 代码或推理文本，例如
dtype check、ConvTranspose2d 说明、`conv_bias_forward` 和 reduction 方案；没有
padding、EOS 错位或损坏样本。固定 prompt 结果因此建立了 checkpoint 介入的真实
总效应。由于 response token 不相同，它仍包含 checkpoint→continuation/length→tail
的中介路径，不能被写成“LoRA 参数对同一 token stream 的直接效应”。

## 4. Full-prefix 2×2 为什么不能完成直接归因

为固定 realized context，每组 response 又由两个 checkpoint 的 SGLang
full-prefix scorer 重算，再由匹配 checkpoint 的 Megatron replay。四格设计原本
希望用同列差值隔离 policy state、同行差值隔离 context：

| `outside@0.15` | context74 | context94 |
|---|---:|---:|
| policy74 | 1/84,251 = 0.00119% | 0/77,987 |
| policy94 | 4/84,251 = 0.00475% | 2/77,987 = 0.00256% |

这四个单元几乎没有 tail。context74 上的 policy effect 为 +0.00356 pct-pt，CI
跨 0；context94 上为 +0.00256 pct-pt，CI 下界为 0。它们既不具有实际 generation
的量级，也不具有负方向。

原因不是 bootstrap power，而是行为端点被替换了。相同 policy 的 generation 与
full-prefix SGLang 在两个 checkpoint 上都 0/16 records 通过严格控制；完整轨迹的
sampled-logprob mean absolute difference 为 0.06584/0.06034，response 每一行的
43 层 route 都至少有一处不同。第一 response token 的平均误差只有
`2.6e-5/1.9e-4`，说明 token indexing 对齐，差异是在增量状态推进后产生。

端点修正的 checkpoint 差值给出直接代数定位：

- iter74 `full-prefix outside - generation outside = -0.00187535`；
- iter94 为 `-0.00046162`；
- 两者之差 `+0.00141373`，CI `[+0.00091864,+0.00195862]`；
- 它几乎抵消实际 generation 的 `-0.00139996`，只留下 full-prefix 对角线的
  `+0.00001378`。

因此，实际收缩由 generation endpoint correction 随 checkpoint 改变所承载。
四格 full-prefix 的 policy/context effect 只能描述另一个端点，不能承担在线 tail
的直接分解。这也是最重要的测量结论：对 production generation 问题，不能用
teacher-forced full-prefix 结果替换目标量。

## 5. 根因已经定位到哪里，证据又停在哪里

Generation 对角线中，Megatron 回放 SGLang generation 捕获的 expert IDs；因此
expert-ID 选择差异不是该跨引擎 tail 的必要条件。没有固定的仍包括 gate weights、
进入 router 的 hidden state、专家与 attention 计算、增量 KV/state，以及两栈的
数值路径。当前证据支持的最窄完整陈述是：

> iter74→iter94 的尾部收缩，是 production-like gamma=3 CUDA-graph DSPARK
> generation 行为分布相对 Megatron fixed-ID replay 的 checkpoint 敏感 mismatch
> 变小；full-prefix SGLang 相对 Megatron 的同类 tail 在两个 checkpoint 上都近零。

此前 H200 base-model 调试在 MHC split、router/TP reduce、C4 compressor 和
FlashMLA tail 上找到过 work-size 数值链，并在短 prompt 上做到 generation 与
full-prefix bit-exact。但保留的 exact launcher 是 gamma=4、无 LoRA capability、
`--disable-cuda-graph`；不能直接证明 formal gamma=3 graph endpoint 已修复。

本次在当前 debug tree 上打开全部已知 shape-stable 开关，并从 live process 环境
逐项核对，仍无法闭合：iter74/iter94 前 1024 token 的同引擎 mean absolute
logprob difference 为 0.09753/0.09386，与 production 的 0.09746/0.09566 基本
相同。没有 active adapter 的 24/22/25-token prompt 在同一 gamma=3 graph server
上也有 sampled mean absolute difference 0.10159；因此 active LoRA 权重和长
prompt 都不是必要条件，但 LoRA-capable server、gamma=3 和 CUDA graph 尚未分开。

为隔离 graph 而做的临时 eager gamma=3 启动在首请求令 scheduler 以 signal 3
退出，没有生成样本。该失败只能说明当前组合不能作为稳定对照，不能证明 CUDA
graph 是或不是低层根因。到这里继续打开更多未匹配开关只会重复混杂；应转为小型
factorial 和首分叉层 instrumentation。

## 6. 当前工程决策与下一实验

不修改以下组件：

- DPPO clip 阈值和 loss：它们忠实地对当前 behavior/trainer mismatch 计数；降低
  tail count 不等于 loss 本身失效；
- rollout 权重同步：lag-0 restart 已排除一步时延为主因；
- oversampling cap：它不参与 75-94 的原始收缩，且对 95 后反弹尚无配对因果量；
- 已知 shape-stable debug 开关：当前 gamma=3 graph 端点未闭合，且 compressor/
  FlashMLA 方案尚未做 production 性能验收。

下一项根因实验按最小差异顺序进行：

1. 在同一 source tree、checkpoint、gamma=3、DP rank、prompt、seed 下，先启动完全
   不带 LoRA capability 的 base server，做 CUDA graph on/off；eager arm 必须先
   修到稳定，不能使用本次 crash 结果。
2. 对通过的 arm 加入 LoRA capability 但不激活 adapter，再加入 iter74/iter94。
   每步先做 3×128 短 prompt，再做真实长 prompt；保存 full logits、routes 和公开
   scalar，不能只看 sampled mean。
3. 在第一个 generation/full-prefix 分叉的 arm，固定同一 verify block，分别用
   historical KV 与 rebuilt KV 重放下一 block；逐层比较 hidden、router scores、
   gate weights、compressor output、KV bytes 和 attention output，找到第一处分叉。
4. 只有 production-like graph endpoint 通过后，才重新跑本次 16-prompt generation
   对角线和 Megatron score，验证 `outside@0.15` 是否消失且两个 checkpoint 不再有
   74% 差异。

若目标改为解释 step95 后反弹，则另做同 checkpoint/cursor/seed 的 rollout-only
cap32/cap24 A/B；它不是原始 contraction 的首要根因实验。

## 7. 证据、入口与执行偏差

- [正式 steps75-164、lag、方向、采样边界与停止原因](../../local_artifacts/deepseek-v4/r2_logs/dppo_clipfrac_tail_formal_75_164_20260724.txt)
- [固定 prompt 2×2、置信区间、长度审计、端点控制与人工样本](../../local_artifacts/deepseek-v4/r2_logs/dppo_tail_fixed_grid_h200_20260724.txt)
- [iter74/iter94 LoRA 状态比较](../../local_artifacts/deepseek-v4/r2_logs/dppo_tail_adapter_iter74_iter94_20260723.txt)
- [固定 prompt 选择与人工检查](../../local_artifacts/deepseek-v4/r2_logs/dppo_tail_fixed_prompts_20260723.txt)
- [探针数学、测试与 topology 边界](../../local_artifacts/deepseek-v4/r2_logs/dppo_tail_probe_validation_20260723.txt)
- [先前 generation/full-prefix work-size 根因与 eager 范围](../../local_artifacts/deepseek-v4/r2_logs/tim_h200_dspark_root_fix_20260722.txt)
- [固定上下文探针](../../scripts/dsv4/studies/entropy/predictive_tail_probe.py)
- [production-like rollout launcher](../../scripts/dsv4/studies/entropy/launch_predictive_tail_probe_server.sh)
- [predictive-DPPO 实现](../../slime/utils/ppo_utils.py)

执行中有四项需要明确保留：

1. 计划的 generation cap 从 4096 提高到 6144，因为正式 mean response 约 5470；
   4096 会系统性截断典型后段。结果是 endpoint 更有代表性，但运行更久。
2. 一次用于打印参数的尝试误启动了 SGLang，当时 H200 上有无关 vLLM eval；发现后
   只停止了新进程，无关任务继续运行，没有使用该尝试的结果。
3. trainer 输出完成且 16 records 可加载后，torchrun parent 在 workers 退出后仍
   挂住；验证产物后只终止了准确的 launcher PID。八个 score artifacts 均完整。
4. 临时 eager gamma=3 arm 首请求 crash，没有科学结果，也没有重复盲跑。所有 probe
   最终停止，H200 八卡均为 4 MiB、0% utilization。
