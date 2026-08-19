# DeepSeek-V4-Flash 三轮 KernelBench 评测与失败归因

更新时间：2026-08-03

## 2026-08-03 更新：model0731 显式 low 重跑已完成

原 model0731 32K endpoint 的 L3 Turn 3 effective truncation 为 21%，超过用户指定的
10% 阈值。为排除 reasoning 配置没有被显式传入的疑问，node53 的 8 张 H20 又完成了
一次独立三轮评测。严格 chat template 只接受 `enable_thinking=true` 和
`reasoning_effort=low`；实际 job 命令、`GenerateState` 和 dump 中保存的 template
共同证明参数被消费。官方 low prefix 为空，预启动对照又确认严格模板与旧默认-low
模板逐字节、逐 token 相同，因此这次运行验证的是配置显式性；两次独立采样的分数差
不能解释成 low reasoning 的因果效果。

这里的严格模板是本次重跑专用的冻结 artifact：
`local_artifacts/deepseek-v4/kernelbench_3turn_20260801/chat_template_model0731_explicit_low.jinja`。
它不等同于仓库通用的 `deepseek_v4_chat_template.jinja`，也不用于证明通用模板已经实现
`reasoning_effort` 分档；它只通过“缺失或非 low 立即报错”证明本次请求显式消费了 low。

正式 job succeeded，三个 level contract 与 `eval_suite` 全部 PASS；800/800/400 条
轨迹没有 wall-clock abort 或缺失 `env_result`，SGLang retract 为 0。运行使用 32,768
context/response/max-prefill、三轮、100/100/50 题、每题 8 条、seed 42、
temperature/top-p=1、KernelGym `low/32/32`，没有 eager mode。一次性容器在结束后
删除，8 张 H20 已释放。

完整 Turn 1/2/3/Best 指标如下，单轮列和 Best 都使用完整轨迹数作分母：

| Level | 指标 | Turn 1 | Turn 2 | Turn 3 | Best |
|---:|---|---:|---:|---:|---:|
| L1 | Compile | 53.62% | 80.38% | 83.50% | 97.12% |
| L1 | Correct | 39.88% | 59.13% | 62.38% | 85.12% |
| L1 | Fast@1.0 | 11.00% | 22.62% | 26.50% | 35.75% |
| L1 | Fast@1.2 | 7.75% | 14.75% | 18.38% | 21.75% |
| L2 | Compile | 35.50% | 67.00% | 70.50% | 90.00% |
| L2 | Correct | 18.00% | 34.12% | 40.50% | 60.25% |
| L2 | Fast@1.0 | 4.38% | 11.00% | 16.25% | 19.38% |
| L2 | Fast@1.2 | 1.38% | 4.12% | 6.88% | 7.50% |
| L3 | Compile | 20.25% | 41.50% | 47.00% | 63.00% |
| L3 | Correct | 3.50% | 5.75% | 8.75% | 13.75% |
| L3 | Fast@1.0 | 0.25% | 0.75% | 2.00% | 2.75% |
| L3 | Fast@1.2 | 0.00% | 0.00% | 0.75% | 0.75% |

用户定义的 Turn 3 effective truncation 仍把实际 T3 `status=truncated` 与因 T1/T2
prompt 过长而根本没有 T3 相加：

| Level | 实际 T3 truncated | 因 T1/2 过长而无 T3 | 合计 |
|---:|---:|---:|---:|
| L1 | 2 | 0 | 2/800（0.25%） |
| L2 | 30 | 1 | 31/800（3.875%） |
| L3 | 72 | 13 | 85/400（21.25%） |

缺失 T3 的 14 条轨迹全部由 `finish_reason=prompt_truncated` 解释。原默认-low 运行
对应 0.50%/3.25%/21.00%；新运行的 L3 仍超过阈值，而且三层变化方向不一致，当前
证据不支持显式 low 降低截断。原生 record-level truncation 为 L1
2/2400（0.0833%）、L2 31/2398（1.2927%）、L3 92/1183（7.7768%）。

分数存在一项新的测量边界。dump 中有 81 条明确传输失败：L1/L2/L3 为 14/58/9 条，
覆盖 14/57/9 条轨迹；旧默认-low 运行的 exact server-disconnected 是 8/11/9 条。
这些响应已有完整生成源码，但 KernelGym 没有返回候选 verdict。上表仍是正式固定分母
观测值，应读作下界；若所有传输失败候选都命中，Best Correct 的最大上修为 L1
0.50、L2 2.875、L3 1.25 个百分点。这只是上限，不是校正分数。候选自己的编译失败、
错误结果、非法地址、subprocess crash 和 300 秒 task limit 仍按模型代码结果处理。
运行后服务 queue 为 0，48 CPU + 16 GPU worker 全部 online；传输失败根因尚未闭合，
也没有对 81 条做部分重评。因此本次分数不能用于声称 reasoning 配置带来提升或下降，
但它不改变 Turn 3 是否生成及 `status` 的结构统计结论。

人工抽检覆盖每层正常、T3 truncated、无 T3 和传输失败轨迹。正常轨迹都保留三段真实
源码及嵌套环境结果；代表性的 T3 truncated 为 L1 group 545、L2 group 14、L3
group 28，无 T3 为 L2 group 665 和 L3 group 50，传输失败为 L1 group 43、L2
group 22、L3 group 399。完整审计见
`local_artifacts/deepseek-v4/kernelbench_3turn_20260801/model0731_explicit_low_audit_20260803.txt`。

正式 dump/summary/run log/orchestrator SHA256 为
`3971bffca940f45848cf899d0632ee3dbe93b0c52911bcd9efdec4cb614bab15` /
`0ae1ef568ec5fc6684afdbc87ec740cbf463f5fc0dbf7b77c4ea1c3cfc821d4e` /
`4c2edf90a05afbe43745f60d9b523048d8097cc85088186facfa3bbcb29d7cec` /
`7fd5cc366bb107eaaf38140822b211ea6cbffd2cb0f0515893b03ef4752102c7`。
首次启动因过长 result tag 使 Ray AF_UNIX socket path
超限，在 0 sample、0 KernelGym task 时退出并隔离；正式运行改用短 Ray temp tag
`m0731low` 后完成。该零样本尝试没有污染正式分母。

## 2026-08-03 更新：32K 三轮补充评测已完成

按后续要求，model0731、step440 和 step620 又在 32K context/response 合同下完成
独立三轮评测。三个 checkpoint 都使用官方 L1/L2/L3 100/100/50 题、每题 8 条轨迹、
seed 42、temperature/top-p=1、同一提示与 evaluator/ATen legality gate；
`rollout-max-context-len`、`rollout-max-response-len`、SGLang context length 和 max
prefill tokens 均为 32,768。model0731 在 node53 运行；node70 在 model0731 运行期间
另起 step440 -> step620 的严格串行队列。三次运行都没有使用 eager mode。

32K 三轮 Best 结果：

| 模型 | Level | Compile | Correct | Fast@1.0 | Fast@1.2 |
|---|---:|---:|---:|---:|---:|
| step440 | 1 | 100.00% | 97.75% | 32.38% | 20.88% |
| step440 | 2 | 99.25% | 69.25% | 4.62% | 4.00% |
| step440 | 3 | 96.50% | 18.50% | 2.00% | 1.25% |
| step620 | 1 | 99.75% | 97.62% | 36.12% | 24.75% |
| step620 | 2 | 98.88% | 88.88% | 13.00% | 11.25% |
| step620 | 3 | 98.25% | 23.00% | 3.75% | 2.00% |
| model0731 | 1 | 97.75% | 86.12% | 33.88% | 20.88% |
| model0731 | 2 | 88.25% | 62.25% | 20.38% | 7.62% |
| model0731 | 3 | 61.75% | 14.75% | 3.75% | 1.50% |

三个 job 均 succeeded，九个 level contract 全部 PASS，`eval_suite=PASS`；合计
6,000 条轨迹没有 wall-clock abort sentinel 或缺失 `env_result`，容器均在结束后
删除。step440 的 L1/L2/L3 turn records 为 2400/2400/1192，step620 为
2400/2400/1198，model0731 为 2396/2396/1171。

正式 `truncated_ratio` 采用 record-level 定义。model0731 的 L1/L2/L3 分别为
3/2396（0.1252%）、28/2396（1.1686%）、75/1171（6.4048%），overall 为
106/5963（1.7776%）。用户随后把决策口径明确为 Turn 3 effective truncation：实际
Turn 3 的 `status=truncated`，以及因 Turn 1/2 已使 prompt 超过 context 而根本没有
Turn 3，都计入；分母固定为完整轨迹数。逐轨迹复核结果如下：

| 模型 | Level | 实际 T3 truncated | 因 T1/2 过长而无 T3 | 合计 Turn 3 truncation |
|---|---:|---:|---:|---:|
| step440 | L1 | 0 | 0 | 0/800 (0.00%) |
| step440 | L2 | 2 | 0 | 2/800 (0.25%) |
| step440 | L3 | 11 | 8 | 19/400 (4.75%) |
| step620 | L1 | 0 | 0 | 0/800 (0.00%) |
| step620 | L2 | 0 | 0 | 0/800 (0.00%) |
| step620 | L3 | 1 | 1 | 2/400 (0.50%) |
| model0731 | L1 | 2 | 2 | 4/800 (0.50%) |
| model0731 | L2 | 23 | 3 | 26/800 (3.25%) |
| model0731 | L3 | 66 | 18 | 84/400 (21.00%) |

所有缺失 T3 的轨迹都以 `finish_reason=prompt_truncated` 结束，没有混入 abort 或其他
提前结束原因。model0731 L3 因此超过 10%，推翻了此前只依据原生 record-level ratio
得出的“不触发”判断。该时点按用户指示先记录结果；随后已经按本文开头的新合同完成
explicit-low 重跑。本段原始 model0731 endpoint 的启动命令只有
`--apply-chat-template-kwargs {"enable_thinking":true}`，不能回写成显式 low。

step440 的 L1/L2/L3 record-level truncation 为 0%、0.0833%、1.0067%；step620 为
0%、0%、0.0835%。32K 明显解除旧 12K 合同中的大量输出截断，但前后是独立随机生成，
不能把所有分数变化唯一归因于 context 长度。

32K 产物哈希：

- step440 dump/summary：
  `320f190d310e93d6cd01d6f0ad3eb95dcb17b51b1a55b11e549dd96b93789704` /
  `85962aa3e46e6d23e866d213b04312ef17ba840f82540c311cf24c2d2b8ddfc5`；
- step620 dump/summary：
  `05d142dddc925214ecff217ed19c4c062ae8f94c5b35ff3e7e73f047a5dea365` /
  `191309ee0fec083a188cf5bbee3883dbd2a7254cbb05e037b1a0177e2df5aa3c`；
- model0731 dump/summary/orchestrator：
  `07ec967eef1f380db49e662685b683f6942d2e21d4cef804965941d9b72d8d09` /
  `713c1ac08dae1d2072cd22ffe5d7b2f3d6a247b620c9e00b6429cd5f0387de2a` /
  `c5459e041c6224cfcf0bb0dc49971a5d8eb0e32618a6eaa2ced7d8e0aecd71d3`；
- node70 串行队列日志：
  `/mnt/data/csl_v4_kernelbench_3turn_20260801/step440_step620.32k.queue.20260802.102917.out`，
  SHA256 `7f4c1835feba0b63dea132706e65214c20a4165f6d725cec4bb8d3f9afd79c29`；
- 32K runner SHA256：
  `c2bee4f0578b5bb41f25793b7a80f802d370697df2d380f9747460602b22574c`。

执行偏差：node70 的冻结评测 snapshot 仍把 context/response 硬编码为 12,288；为避免
修改冻结目录，本次把已支持环境覆盖的 runner 放到 node70 `/tmp`，显式注入
`EVAL_MAX_CONTEXT_LEN=32768` 和 `EVAL_MAX_RESPONSE_LEN=32768`。node70 当时没有
step440 adapter，因此从 node53 复制后用 SHA256
`896a90674898773b00f5af3403b47f9750a19506845c966591c6cee81f9061a5` 核验，再启动
串行队列。两项 workaround 均在启动日志中留有合同证据；正式运行参数和产物未留缺口。

## 摘要

这轮工作先把 node53 上被 0731 权重覆盖的 DSpark 基座从 node64 权威副本原子恢复，
再用同一套已部署 ATen 合规门禁的合同评测 step440、step620 和无 LoRA 的
DeepSeek-V4-Flash-0731。训练按用户指示保持停止，评测使用 32 并发；命令中的
`train.py --debug-rollout-only` 只负责生成和评分，不更新权重。

两个 LoRA checkpoint 的最强结论来自 L2：step620 三轮 BestCorrect 为 80.125%，
step440 为 47.875%。step440 的 raw-correct 实际达到 82.375%，但 283/800 条轨迹
至少一轮调用了禁止的 ATen compute；step620 只有 5/800 条。候选源码和 profiler
metadata 直接确认了 `self.conv(...)`、`self.conv_transpose(...)` 先完成核心计算、
扩展只做后处理的路径。step440 先前的高分主要混入了 PyTorch fallback；新门禁下，
step620 在 Conv 和 ConvTranspose 家族的合规 Correct/Fast 明显领先。

step620 的 L1 BestCorrect 已到 97.000%，Fast@1.0 仍只有 29.250%；L2 为
80.125%/9.000%，L3 为 13.250%/2.250%。人工裁决后仍完全未解的题包括 7/93 道
L2 和 21/37 道 L3；复杂 ConvTranspose 管线、reduction-heavy Matmul、整网 CNN、
Transformer/RNN 和 Mamba 是当前能力边界。L1 题73仍是唯一已证实会改变本次评分
语义的题，step440/620 都是 0/8 correct；历史扫描范围和逐题证据统一见
`step440_step620_kernelbench_failure_attribution_20260729.md` 的 Level1 逐题审计节。

无 LoRA 的 model0731 在同一合同下明显更弱：L1/L2/L3 BestCorrect 为
63.000%/17.250%/2.250%，Fast@1.0 为 26.125%/5.250%/0%。排除已确认坏题后仍有
5/99、36/93、34/37 道题 0/8 correct。它的 config、tokenizer、generation config
和 chat template 与 DSpark 路径逐文件同 hash，但完整权重不同；因此它与 step620
可以作为 checkpoint 端点比较，不能把差值全部解释成 LoRA 的因果增益。model0731
L2/L3 的 record-level 输出截断率达到 59.2%/85.6%，生成完整度也是端点差异的一部分。

正式 dump 只留下 3 条 wall-clock abort sentinel：step440 L2 一条、step620 L2
两条。wrapper 没有保存这些轨迹已经完成的前两轮，因此整条轨迹按零分计入固定分母，
对应 0.125%/0.25% 的保守 infra 边界。其余服务返回的 native crash、非法地址和
undefined symbol 都带候选执行上下文，按模型代码结果处理。新建的 `--init` 容器内
zombie 为 0；model0731 正式跑的 2,000 条轨迹全部有环境结果、0 abort sentinel，
结束后 KernelGym 仍为 64/64 worker online、队列为 0。宿主现存 zombie 来自旧
`sleep infinity` 容器，不是本轮新增长尾。

截至 2026-08-03，训练按用户指示保持停止，使用这组三轮结果作为新 ATen 门禁后的
基线。历史单轮曲线保留原值，但不能跨门禁修复边界直接拼接。训练恢复前还应校准
coverage rejection sampling 对 extension-internal cuBLAS/cuDNN 的误杀边界。

## 测量合同：三轮 Best 才是主 endpoint

L1/L2/L3 使用官方 100/100/50 题，每题 8 条独立轨迹。生成参数为 seed 42、
temperature/top-p=1、12,288 context/response，最多 3 轮；KernelGym 使用同一 A800
worker 池和 `low/32/32`。每个 H20 任务只加载一次模型，依次评测三个 level。

Compile 表示一条轨迹在最多三轮内至少一轮编译成功；Correct 还要求数值正确且
`decoy_kernel=false`；Fast@1.0/@1.2 再要求 speedup 达到阈值。三项始终使用
800/800/400 的完整轨迹分母。单轮 T2/T3 只覆盖上一轮尚未结束的轨迹，不能独立解释
反馈收益；累计 `Best@Tk` 回答“给到第 k 轮后是否曾经解出”。

step440 和 model0731 在 node53、step620 在 node70，均为 8xH20 和 fresh Docker
`--init` 容器。
两个节点使用相同代码 fingerprint、DSpark base、提示和 evaluator；image 的 65 个
base/runtime layers 相同，最终 JIT/cache layer 为宿主本地副本。节点和独立随机生成
使它们不等同于逐 token 配对 A/B；绕过归因依赖候选源码与 ATen metadata，而非只看
分数相关性。

step440/620 使用同一 DSpark base，只改变 LoRA checkpoint；曲线标签遵循
`step = iteration + 1`，分别来自 `iter_0000439` 和 `iter_0000619`。model0731 使用
另一套完整权重；它的非权重配置与提示合同相同，但不构成“同一 base 只去掉 LoRA”的
配对消融。

2026-07-29 的单轮历史点在 ATen legality gate 部署前产生。下文 2026-08-01/02
三轮点全部使用新门禁，可以比较 checkpoint 端点；跨这条修复边界的绝对分数只保留
为历史合同结果。单轮曲线与失败归因分别见
`kernelbench_l1_lora_curve_20260724.md` 和
`step440_step620_kernelbench_failure_attribution_20260729.md`。

## 正式结果：step620 的 L2 合规提升最大

| 模型 | Level | Compile | Correct | Fast@1.0 | Fast@1.2 |
|---|---:|---:|---:|---:|---:|
| step440 | 1 | 797/800 (99.625%) | 767/800 (95.875%) | 229/800 (28.625%) | 160/800 (20.000%) |
| step440 | 2 | 783/800 (97.875%) | 383/800 (47.875%) | 27/800 (3.375%) | 23/800 (2.875%) |
| step440 | 3 | 379/400 (94.750%) | 49/400 (12.250%) | 4/400 (1.000%) | 2/400 (0.500%) |
| step620 | 1 | 797/800 (99.625%) | 776/800 (97.000%) | 234/800 (29.250%) | 172/800 (21.500%) |
| step620 | 2 | 765/800 (95.625%) | 641/800 (80.125%) | 72/800 (9.000%) | 56/800 (7.000%) |
| step620 | 3 | 358/400 (89.500%) | 53/400 (13.250%) | 9/400 (2.250%) | 6/400 (1.500%) |
| model0731 | 1 | 603/800 (75.375%) | 504/800 (63.000%) | 209/800 (26.125%) | 119/800 (14.875%) |
| model0731 | 2 | 258/800 (32.250%) | 138/800 (17.250%) | 42/800 (5.250%) | 19/800 (2.375%) |
| model0731 | 3 | 35/400 (8.750%) | 9/400 (2.250%) | 0/400 (0.000%) | 0/400 (0.000%) |

step620 的 L1 正确性、L2 正确性与速度、L3 正确性与速度都高于 step440；step440
在 L2/L3 Compile 高 2.25/5.25 个百分点。Compile 的方向不能当成合规能力排序：
step440 的可编译候选包含大量 PyTorch fallback；step620 的 record-level response
truncation 记录数又高于 step440。当前证据确认两者的合规性和生成完成度都发生变化，
没有固定 response 反事实能把 Compile 差距唯一归给一项。历史 handoff 没有保留能同时
复现 `prompt_truncated` 终止计数与 turn-record 总数的单一审计产物，因此不发布精确的
record-level truncation 百分点差；这个记账缺口不改变固定 800/800/400 轨迹分母下的
Compile/Correct/Fast 主结果。
step620 相对 model0731 的 L2 Correct 高 62.875 个百分点，L3 高 11 个百分点；
model0731 的 L2 Fast@1 虽高于 step440，主要来自 GEMM/Matmul/BMM 的 27 条轨迹，
不是 convolution 组合能力接近 step620。

三轮反馈对正确性的累计增益如下；在两个 LoRA checkpoint 中，T3 相对 T1 的最大增益
出现在 step620 L2，达到 14.875 个百分点。三套端点整体的最大增益是 model0731 L1
的 28.625 个百分点。

| 模型/Level | BestCorrect@T1 | @T2 | @T3 | BestFast@1@T1 | @T2 | @T3 |
|---|---:|---:|---:|---:|---:|---:|
| step440 L1 | 89.625% | 94.875% | 95.875% | 19.625% | 27.250% | 28.625% |
| step440 L2 | 38.750% | 46.875% | 47.875% | 2.375% | 3.000% | 3.375% |
| step440 L3 | 7.750% | 12.000% | 12.250% | 0.750% | 1.000% | 1.000% |
| step620 L1 | 89.125% | 95.750% | 97.000% | 22.625% | 28.250% | 29.250% |
| step620 L2 | 65.250% | 78.250% | 80.125% | 6.875% | 8.750% | 9.000% |
| step620 L3 | 9.000% | 12.500% | 13.250% | 2.000% | 2.000% | 2.250% |
| model0731 L1 | 34.375% | 58.750% | 63.000% | 13.500% | 24.125% | 26.125% |
| model0731 L2 | 7.375% | 14.250% | 17.250% | 2.125% | 4.250% | 5.250% |
| model0731 L3 | 0.750% | 2.000% | 2.250% | 0.000% | 0.000% | 0.000% |

12k 上下文已实际限制后续轮次。`status=truncated` 表示当前响应耗尽可用输出长度；
step440 的 L1/L2/L3 分别有 682/716/175 条记录，step620 有 740/906/324 条，且
绝大多数随后触发 `MODEL_NEW` 结构检查失败。`finish_reason=prompt_truncated` 表示当前
轮已评分，但追加完整代码和反馈后无法再容纳下一轮：step440 在 T1/T2 后分别发生
55/54、327/300、84/48 次，step620 为 234/227、285/275、62/44 次，三组顺序均为
L1/L2/L3。这是固定 12k 合同下的答案长度边界，会压低可获得的 T2/T3 增益。
model0731 的 L1/L2/L3 分别有 786/944/662 条 record 为 `status=truncated`，占各
level record 的 38.8%/59.2%/85.6%；260/466/237 条轨迹以
`finish_reason=prompt_truncated` 结束。L3 只生成 773 条 turn record，远少于最多
1,200 条，12k 输出/上下文合同明显限制了后续反馈机会。

## step440 的 L2 高分来自哪里

旧 checker 只确认扩展存在和数值正确，可能漏过“PyTorch 完成主体、扩展只做后处理”。
新 KernelGym 在每次 candidate correctness forward 外层用 CPU ATen profiler 记录
dispatcher 调用，并按显式 allowlist 判定。step440 L2 题1 的真实 metadata 记录了
`aten::_convolution`、`aten::conv2d`、`aten::convolution` 和
`aten::cudnn_convolution`，policy reason 为 `DISALLOWED_ATEN_COMPUTE`。

这个门禁把 raw numeric correctness 与合规 Correct 分开：

| 模型 | Level | Raw-correct 轨迹 | Scored-correct 轨迹 | 任一轮 decoy 轨迹 |
|---|---:|---:|---:|---:|
| step440 | 1 | 770 | 767 | 8 |
| step440 | 2 | 659 | 383 | 283 |
| step440 | 3 | 117 | 49 | 69 |
| step620 | 1 | 776 | 776 | 1 |
| step620 | 2 | 646 | 641 | 5 |
| step620 | 3 | 55 | 53 | 2 |
| model0731 | 1 | 505 | 504 | 1 |
| model0731 | 2 | 140 | 138 | 2 |
| model0731 | 3 | 10 | 9 | 1 |

同一轨迹可能一轮命中 decoy、另一轮给出合规正确实现，因此最后一列不等于 raw 与
scored 的差。L2 家族拆分进一步定位了变化：

| 模型 | 家族 | 轨迹 | Raw correct | Scored correct | Decoy-only | Fast@1.0 |
|---|---|---:|---:|---:|---:|---:|
| step440 | GEMM/Matmul/BMM | 296 | 251 | 217 | 34 | 11 |
| step620 | GEMM/Matmul/BMM | 296 | 230 | 230 | 0 | 15 |
| model0731 | GEMM/Matmul/BMM | 296 | 49 | 49 | 0 | 27 |
| step440 | ConvTranspose | 240 | 174 | 20 | 154 | 2 |
| step620 | ConvTranspose | 240 | 163 | 160 | 3 | 29 |
| model0731 | ConvTranspose | 240 | 24 | 24 | 0 | 1 |
| step440 | Conv | 264 | 234 | 146 | 88 | 14 |
| step620 | Conv | 264 | 253 | 251 | 2 | 28 |
| model0731 | Conv | 264 | 67 | 65 | 2 | 14 |

step440 的 raw-correct 与 step620 接近，但 ConvTranspose 有 154 条、Conv 有 88 条
轨迹只靠 decoy 得到正确结果。新门禁下 step620 的合规 Correct 和 Fast 均更高，
因此原先“step440 L2 更强”的解释不成立。

## 题目问题：官方分数与人工裁决集同时报告

### L1：题73会改变评分语义

题73 的错位参数绑定、104 条历史生成对照和全库扫描边界已归档在
`step440_step620_kernelbench_failure_attribution_20260729.md`。本轮新合同下，
step440/620/model0731 在该题分别为 8/8、7/8、6/8 compile，均为 0/8 correct；
这与历史结论一致，但不把其余 99 题提升为已形式证明无问题。

排除题73、仍用其余 99 题 792 条轨迹时，step440 的 L1
Compile/Correct/Fast@1/Fast@1.2 为 99.621%/96.843%/28.914%/20.202%；step620 为
99.747%/97.980%/29.545%/21.717%；model0731 为
75.379%/63.636%/26.389%/15.025%。这只是敏感性，修复题目后仍需重测。

### L2/L3：退化题会高估部分结果

GPU 多探针与逐题人工裁决已经确认 L2 有 7 道、L3 有 13 道题在官方输入/容差下输出
恒定或近恒定，或 oracle 本身不可复现。L2 排除题为 9、23、36、38、42、80、83；
L3 为 3、7、11、12、18、19、20、22、23、24、28、35、45。人工裁决集保留
93/37 题；它们用于诊断，不能替换历史官方 100/50 题分母。

| 模型 | 集合 | 轨迹 | Compile | Correct | Fast@1.0 | Fast@1.2 |
|---|---|---:|---:|---:|---:|---:|
| step440 | L2-93 | 744 | 98.118% | 47.446% | 1.075% | 0.538% |
| step620 | L2-93 | 744 | 95.968% | 80.645% | 7.124% | 4.973% |
| model0731 | L2-93 | 744 | 32.527% | 17.204% | 4.704% | 1.747% |
| step440 | L3-37 | 296 | 93.581% | 14.189% | 1.351% | 0.676% |
| step620 | L3-37 | 296 | 87.838% | 16.892% | 3.041% | 2.027% |
| model0731 | L3-37 | 296 | 10.135% | 2.365% | 0.000% | 0.000% |

L2 的 7 道排除题在 step620 均至少有一条 correct 轨迹，并给 step440/620 各贡献
恰好 19 条 Fast@1.0 和 19 条 Fast@1.2。移除后 step620 仍有 53/744 Fast@1，
step440 只有 8/744；退化题没有解释两者的合规 Fast 差距。model0731 的 L2 排除题
贡献 7 条 Fast@1 和 6 条 Fast@1.2，移除后仍有 35/744 Fast@1。L3 排除题对三个
checkpoint 都没有贡献 Fast，移除主要改变分母。

## 模型真正做不好的题型

排除题目问题后，step620 仍有 7/93 道 L2 题完全未解：题3、11、13、14、18、66、
72。它们集中在多阶段 ConvTranspose3d/2d 与 reduction-heavy Matmul/GEMM。代表性
失败包括题14 可编译但归约结果不正确；题3/11/13/72 要同时保留转置卷积、归一化、
pool/activation 等完整语义，候选经常省略阶段或错误融合。

L3 人工裁决集仍有 21/37 道完全未解：AlexNet、ResNet18/101、DenseNet block/整网、
ShuffleNet、Swin/CvT、LSTM/GRU 变体、MiniGPT 和两道 Mamba。按架构族的三轮 Best
计数如下：

| 模型 | L3 家族 | 轨迹 | Compile | Correct | Fast@1.0 |
|---|---|---:|---:|---:|---:|
| step440 | MLP | 24 | 24 | 23 | 0 |
| step620 | MLP | 24 | 20 | 15 | 0 |
| model0731 | MLP | 24 | 7 | 2 | 0 |
| step440 | CNN | 200 | 188 | 16 | 4 |
| step620 | CNN | 200 | 184 | 25 | 9 |
| model0731 | CNN | 200 | 19 | 2 | 0 |
| step440 | Transformer/attention | 64 | 62 | 2 | 0 |
| step620 | Transformer/attention | 64 | 61 | 7 | 0 |
| model0731 | Transformer/attention | 64 | 1 | 0 | 0 |
| step440 | RNN | 80 | 80 | 8 | 0 |
| step620 | RNN | 80 | 77 | 4 | 0 |
| model0731 | RNN | 80 | 6 | 5 | 0 |
| step440 | Mamba | 16 | 16 | 0 | 0 |
| step620 | Mamba | 16 | 6 | 0 | 0 |
| model0731 | Mamba | 16 | 2 | 0 | 0 |
| step440 | NetVLAD | 16 | 9 | 0 | 0 |
| step620 | NetVLAD | 16 | 10 | 2 | 0 |
| model0731 | NetVLAD | 16 | 0 | 0 | 0 |

L3 的 Fast@1 全部来自 CNN。MLP、Transformer/attention、RNN、Mamba、NetVLAD
都没有 Fast 轨迹。源码抽检显示这不是统一计时缩放：step620 L2 题1 的完整扩展正确
但只有 0.0927x，题50 的完整 ConvTranspose3d+后处理扩展达到 13.91x；L3 题17 的
完整 SqueezeNet Fire 扩展达到 3.22x。L3 题49 的 Mamba 候选直接把输出清零，能编译
但数值错误。模型同时存在语义完整性和性能泛化两个缺口。

model0731 在人工裁决集仍有 5/99 道 L1、36/93 道 L2、34/37 道 L3 完全未解。
L1 的 5 道是三道 3D convolution、`cumsum` 和 scaled-dot-product attention；L3
只解出 GoogleNet Inception module、SqueezeNet Fire module 和 VanillaRNNHidden。
它在 Transformer/attention、Mamba、NetVLAD 上均为 0 correct，L3 所有家族均为
0 Fast@1。L2 的 42 条 Fast@1 中 27 条来自 GEMM/Matmul/BMM，ConvTranspose 只有 1 条。

model0731 的四条 decoy 已逐条读取。L2 题23 的 `self.conv(x)` 和 L3 题13 的
training-mode `self.transition(x)` 是直接 PyTorch fallback；L2 题83 调用原生 dropout，
且题83属于人工裁决排除集。L1 题55 的主体卷积经过自定义扩展，但无 bias 分支在
forward 内创建空 `torch.tensor`，触发 `aten::lift_fresh/detach_`。正式分数按已部署
门禁扣除此轨迹；是否把这两个操作加入 plumbing allowlist 仍需单独校准，最多影响
model0731 L1 的 1/800。

## 基础设施边界与容器生命周期

step440 L2 题56 有一条、step620 L2 题55 有两条严格结构化的
`wall_clock_timeout` abort sentinel，均标记 `turn_idx=2`、
`num_turns_completed=2`、空 response、reward 0、`remove_sample=true`。wall-clock
wrapper 只保存 sentinel，没有保存已完成的 T1/T2 sample；正式分母保留并把整个轨迹
计零，所以 step440/620 最多存在 0.125%/0.25% 的向下不确定性。

runner 的 postcheck 只允许这种精确结构，并限制每个 level 不超过 1%；任何其他缺失
`env_result` 都会失败。除上述 1/2 条 sentinel 外，step440/620 已保存的非 sentinel
turn record 均有 `env_result`；历史记录中的 turn-record 总数与上文 `prompt_truncated`
终止计数无法同时闭合，因此这里不再发布 step440/620 的总 turn-record 数，也不由它
推导精确截断率。KernelGym 返回的 candidate subprocess segfault、undefined symbol、
illegal address 或 launch error 都携带候选调用栈或动态库信息，当前没有证据把它们
归为共享服务故障。
model0731 的 4,391 条 turn record 全部有 env_result；其中未出现 no-idle-worker、
I/O error、连接失败、guard timeout 或 wall-clock sentinel 等 infra signature。

model0731 在正式有效跑之前有两次已隔离尝试。第一次把单轮 3,000 秒 generate
guard 误用于三轮，在 L1 595/800 时已有 189 条 guard timeout；runner 因此改为
显式 9,000 秒。第二次暴露了服务端取消竞态：上一个客户端退出后旧子任务仍在队列；
对旧 ID 发送精确 DELETE 后，16 个新 GPU 任务又在三次 300 秒 idle-worker 等待后同时
返回 `Task failed after 2 retries. Last error: None`。独立唯一代码 smoke 同时在 CPU compile
worker 上复现 `[Errno 5] Input/output error`，所以两次均按 infra-invalid 停止，没有写入
任何模型分数。无效目录后缀分别为 `invalid-guard3000-20260801` 和
`invalid-kernelgym-pool-20260802`。

队列清空后，两台 A800 的 64 个 worker 主进程退出，API/Redis/每台机的 persistent
monitor 保留，并按既有 expected-worker 清单全量重建。源码闭环确认两个永久失活条件：
recycle 已从 canonical list 移除的进程会被 `finally` 重新加入 idle list；
`_get_idle_worker` 又只清理 idle list，死亡 canonical/busy 条目会压制 emergency/top-up。
KernelGym 源码已修正这两点并新增回归测试，连同现有 pool/crash 测试为 15 passed。
评测 runner 的失败路径现会从自己的 host log 提取 ID，取消 parent/`_compile`/
`_kernel`/`_ref` 后再删容器。根修复以 KernelGym 提交 `2625505` 发布并重启服务；
服务恢复 48 CPU + 16 GPU worker 后，16 路不同源码 smoke 全部 compile/correct，并覆盖
两台 A800 的全部 GPU。正式重跑结束后仍为 64/64 online、队列为 0。完整可复核时序见
`local_artifacts/deepseek-v4/kernelbench_3turn_20260801/kernelgym_recovery_evidence.txt`。

每次评测创建一个 fresh `--init` 容器，结束路径最多重试三次删除并用 `docker inspect`
确认不存在。step440/620、正式 model0731 和两次无效 model0731 容器均已删除，它们的
容器内 zombie 为 0。node53
宿主的 95 个 zombie 来自旧 `csl_dspark_r1`，其 PID 1 是 `sleep infinity`，不会 reap
历史 Ray/SGLang children。fresh 容器解决了新评测的 zombie 累积；旧宿主 zombie 只有
重启相应 legacy 容器或其父进程才会消失，逐个 `kill` zombie 本身无效。

## 训练信号：PyTorch compute 已归零，coverage 仍需校准

活跃训练 parquet 的 40,307/40,307 行使用同一提示前缀，明确要求核心计算经过生成
扩展，禁止在 binding 中调用 PyTorch/ATen compute，也禁止在 `ModelNew` 中用
`torch.matmul`、`F.conv2d` 等绕过。

KernelGym 提交 `320c46d` 和 `4f3ee28` 在每个真实 candidate correctness forward
外层运行 CPU ATen profiler。Appendix-J allowlist 之外的 `aten::mm`、
`aten::convolution`、`aten::sum` 等会设置 `decoy_kernel=true` 和
`DISALLOWED_ATEN_COMPUTE`，并跳过性能测试。slime 的 `penalty_score=0`，因此数值
正确但调用这些 PyTorch 官方 compute 算子的训练样本 reward 为 0。allocation、view、
copy 和 dtype/device plumbing 等 allowlist 操作继续合法。

coverage 回答的是“命名的生成 kernel 占设备时间多少”，不能单独证明计算来源。
KernelGym 目前把低于 30% 标为 diagnostic，低于 0.1% 也不硬拒绝，因为 extension
内部的 cuBLAS/cuDNN kernel 尚不能可靠归属。slime 侧同时启用了 time-coverage reward
和 rejection sampling，threshold=0.3、factor=0.1；正确但 coverage<0.3 的样本会被
移出训练。这可能过滤合法的 extension-internal vendor-library 实现。恢复训练前应先用
合法 cuBLAS/cuDNN 正例校准该边界，ATen legality gate 继续承担 categorical 合规判定。

更完整的 lazy-optimization 防线、KernelBench-Verified 和 DeepReinforce 方法见
`step440_step620_kernelbench_failure_attribution_20260729.md`；本报告只保留与本次新门禁
结果直接相关的结论。

## DSpark 权重修复与 model0731

node53 的 DSpark 路径曾被 0731 权重覆盖：污染目录首 shard SHA256 为
`f3668ba4cccf1ca6a7eb84e888fb92c1cdc7204d472ba9db771e6fd3abf6b874`，与独立
`DeepSeek-V4-Flash-0731` 一致；node64 权威 DSpark shard 为
`5176586613905d4beaadbaea1cfecd1693e17f9da4a0ea09d99aab7d3f2f5b7c`。

修复先同步到临时目录，逐文件 checksum 后原子替换。node64/node53 的修复目录均为
递归 77 个文件、166,898,672,920 bytes；首 shard 和 index SHA256 一致。污染副本保留
在 node53：
`/mnt/md1/chenshuailin/checkpoints/deepseek-ai/DeepSeek-V4-Flash-DSpark.polluted-20260801-before-node64-repair`。

独立 0731 模型目录有 48/48 个 shard、无 `.incomplete`；index 含 72,317 个 tensor，
首 shard/index SHA256 为 `f3668ba4...`/
`98efab455cf08dfbbbaaba6f570e1bf10bf927d2b4c3c453a59c2f6f0e3be92b`。评测 launcher
只补入缺失的 `chat_template.jinja`，没有修改权重。

model0731 正式跑在 KernelGym 修复后的服务上完成，L1/L2/L3 分别产生 800/800/400
条轨迹和 2,024/1,594/773 条 turn record，全部含 env_result；runner contract、容器清理
和服务后检均通过。Best Compile/Correct/Fast@1/Fast@1.2 计数分别为
603/504/209/119、258/138/42/19、35/9/0/0。排除人工裁决题后的 L2-93 为
242/128/35/13，L3-37 为 30/7/0/0。

它只有 1/2/1 条 L1/L2/L3 轨迹命中 decoy，raw-correct 为 505/140/10；低分来自大量
不完整、预检失败、编译失败或数值错误的候选，没有出现 step440 那样的大规模 PyTorch
fallback。L2/L3 的高截断率和人工裁决集 36/93、34/37 道完全未解共同说明它在复杂
组合和整网任务上的输出完整性与语义能力都不足。config、tokenizer config、generation
config 和 chat template 均与 DSpark 路径同 hash，这次没有发现模板或运行参数错配。

## 审计与可复核产物

完整 claim-evidence brief 位于
`local_artifacts/deepseek-v4/kernelbench_3turn_20260801/claim_evidence_brief.md`。逐题题目
质量证据分别见 `step440_step620_kernelbench_failure_attribution_20260729.md` 和
`handoffs/data/cleaning/handoff_drkernel_and_accepted_v5_cleanup_20260729.md`。

step440 产物位于 node53：
`/mnt/md1/csl_v4_kernelbench_3turn_20260801/experiments/`
`Eval.KernelBenchL123.DeepSeekV4Flash.step440.12k.turn3.n8.h20/step440/`。

- dump/summary/orchestrator SHA256：
  `e3f880c8856aadd65f3c4df36f2f2edab7d244b9a031cfef245f0f3d2e73bdc8` /
  `044a7bf8b9106552cc38c3a2953632a74b08823ed9dc39cc56f9ab3f04c4f30d` /
  `632b05a7dec94b5ebdbb43eff8dc4e7749a2237c91cb4be5772a7ef668a7bde1`。
- adapter SHA256：
  `896a90674898773b00f5af3403b47f9750a19506845c966591c6cee81f9061a5`。

step620 产物位于 node70：
`/mnt/data/csl_v4_kernelbench_3turn_20260801/experiments/`
`Eval.KernelBenchL123.DeepSeekV4Flash.step620.12k.turn3.n8.h20/step620/`。

- dump/summary/orchestrator SHA256：
  `fd2f6a8c2f9e82496fe90fcb9af4afa7d8ae89f411f1d63d3899c26c1842539e` /
  `dbc47a70dbd82c99a9aeebb4d34fc3714270e80ad911e0be00196d13ee01a10e` /
  `5ad839daeae7bf02d82b2180447e9563b9ea7dcd75c5e1a87a24b64439980108`。
- adapter SHA256：
  `7142cc57e4d79513dda2738bec0d63d6679d23a82e47e58e23e5381107edab62`。

model0731 产物位于 node53：
`/mnt/md1/csl_v4_kernelbench_3turn_20260801/experiments/`
`Eval.KernelBenchL123.DeepSeekV4Flash.model0731.12k.turn3.n8.h20/model0731/`。

- dump/summary/orchestrator SHA256：
  `8bbb14f0b400e50f3a202182d544a6435dedd7b50daa14466d3339d776924077` /
  `6a0009520ff5b3d05ae2de3d36126d20ec43d309c69794a73b9f87c53c4321ae` /
  `4970c3730a8cbc06f1c32155e2875e743686520692dac9ffea3f630257894676`。

三份官方数据 SHA256 为 L1
`e034d42fe5e8ed0fac0e580bb9070379f719080b05d666accbac59abc081435f`、L2
`11c1858d88be14ebc7fa766390f46a0db1ad872e921ddc61411e7df4d2e64cfe`、L3
`6b3c85f2e57f307036b8c38ff26891e8b88ac44afb702a7c7f539d2b8307755e`。runner SHA256
为 `252fc3fe3cc3992bf2753c4bec9ddb3128874d9e649477f7bd6efc01e6eac695`；KernelGym
根修复提交为 `26255057463a77b23abac0f3e5eafeeebf2ebbb5`。

## 执行偏差与剩余风险

- node70 的第一次环境预检发现宿主缺 pandas；随后尝试复制过大的镜像上下文又被
  Docker root 空间拒绝。两次都在 0 样本阶段停止。最终使用已有验证 snapshot、缩小
  的 repo 和宿主已有 torch 完成只读预检，正式 step620 结果未使用失败尝试的产物。
- step440/620 的旧 postcheck 把任何缺失 env_result 都拒绝，因此正式 dump 完成后
  runner 以状态 1 退出。人工确认三条记录都是精确 wall-clock sentinel 后，postcheck
  改为仅允许该结构、每 level 上限 1%；对原 dump 重跑通过。没有改写 dump 或分数。
- step440/620 的历史分析没有留下能同时闭合 `prompt_truncated` 终止计数和总 turn-record
  数的独立审计产物；旧文稿中的 5,467/5,378 总数及由其推导的精确截断率已撤下。
  轨迹级 Best 分母、三条 sentinel 及其保守 infra 边界是独立计数，不受此缺口影响。
- node70/node53 的 snapshot image ID 因宿主 JIT/cache 最后一层不同；base/runtime
  layers 和代码 fingerprint 已核验相同。它仍是跨宿主随机评测边界，不能写成严格
  paired A/B。
- model0731 前两次尝试分别因错误的 3,000 秒 guard 和 KernelGym pool 污染作废；
  只有根修复部署、16-GPU smoke 通过后的第三次结果写入正式表。
- 没有修复题73后重跑，也没有把人工裁决集写回官方分数。所有敏感性结果保持独立。

验证命令通过：summarizer 测试 4 passed、KernelGym pool/crash 测试 15 passed、四个
H20/H200 launcher 的 `bash -n`、suite 中四段 Python heredoc 编译、summarizer/
health-check `py_compile` 和两个仓库的 `git diff --check`。最终 node53 无评测容器，
Ray 端口空闲，H20 无计算进程；KernelGym 64/64 worker online、active/busy/queue 均为 0。
