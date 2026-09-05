# DeepSeek-V4 KernelBench 端点比较与失败归因

本文统一维护 step440/step620 的单轮、三轮和 model0731 对照。step 标签遵循 `step = iteration + 1`。L1 全训练曲线由[LoRA 曲线](kernelbench_l1_lora_curve_20260724.md)维护；更晚的官方模型五轮实验见[L3 多轮分析](../evaluation/kernelbench_l3_multiturn.md)

这些结果跨越 context 长度、turn 数和 ATen legality gate 的变化，分别保留各自协议。旧单轮的正式高分中包含可识别的 PyTorch compute bypass，不能把它拼接成门禁修复后的能力曲线；静态敏感性分析也不会回写原始分数

## 单轮 12K：门禁修复前的观测

Level2 是 100 题、每题 8 个样本；Level3 是 50 题、每题 8 个样本。所有运行均为
单轮、12k context/response、seed 42、temperature/top-p=1，并使用 KernelGym
`low/32/32`。Compile、Correct 和 Fast 均以全部样本为分母；Correct 排除 decoy，
Fast 还要求正式 Correct。曲线标签遵循 `step = iteration + 1`：step440 和 step620
分别来自 `iter_0000439` 与 `iter_0000619`

| Level | checkpoint | 样本 | Compile | Correct | Fast@1.0 | Fast@1.2 |
|---|---|---:|---:|---:|---:|---:|
| 2 | base，无 LoRA | 800 | 154 (19.25%) | 50 (6.25%) | 19 (2.38%) | 8 (1.00%) |
| 2 | step440 | 800 | 720 (90.00%) | 597 (74.62%) | 225 (28.12%) | 127 (15.88%) |
| 2 | step620 | 800 | 666 (83.25%) | 549 (68.62%) | 47 (5.88%) | 39 (4.88%) |
| 3 | base，无 LoRA | 400 | 25 (6.25%) | 3 (0.75%) | 0 | 0 |
| 3 | step440 | 400 | 277 (69.25%) | 65 (16.25%) | 7 (1.75%) | 2 (0.50%) |
| 3 | step620 | 400 | 273 (68.25%) | 40 (10.00%) | 6 (1.50%) | 4 (1.00%) |

整数计数是主值；step440 的 74.625% Correct 和 28.125% Fast@1.0 在 summary 中按
当前格式显示为 74.62% 和 28.12%。step620 Level2 的精确 Correct/Fast@1.0/Fast@1.2
为 68.625%/5.875%/4.875%，summary 显示 68.62%/5.88%/4.88%；该 level 有 1 条
raw-correct decoy，step620 Level3 decoy 为 0，均已从正式 Correct 排除

step620 adapter model/config SHA256 分别为
`7142cc57e4d79513dda2738bec0d63d6679d23a82e47e58e23e5381107edab62` /
`f340d4d0e714efcfa93929a5480d2fae540b79d3424cd6d26196d195aa4c3275`；Level2/3
parquet SHA256 分别为
`11c1858d88be14ebc7fa766390f46a0db1ad872e921ddc61411e7df4d2e64cfe` /
`6b3c85f2e57f307036b8c38ff26891e8b88ac44afb702a7c7f539d2b8307755e`，均与 pinned
official revision `423217d9...` 逐行匹配。正式 dump 的独立完整性核验如下：

| 检查 | step620 Level2 | step620 Level3 |
|---|---:|---:|
| 样本 / missing env_result | 800 / 0 | 400 / 0 |
| index/group_id | 严格 0–799 | 严格 0–399 |
| 每题样本数 | 100 题均为 8 | 50 题均为 8 |
| response | 800 个均非空且唯一 | 400 个均非空且唯一 |
| sample status | completed 729；truncated 71 | completed 331；truncated 69 |
| KernelGym status | completed 666；failed 120；timeout 14 | completed 273；failed 125；timeout 2 |

base Level3 还有一个 raw-correct decoy 被正确排除。其 3 个正式 Correct 中，题15
idx113 是未被 checker 识别的 PyTorch 核心计算绕过，因此正式分数保持 0.75%，但
只有另外 2 个样本满足完整自定义扩展的意图。step620 Level3 的 40 个 Correct 已逐个
检查，没有发现同类核心计算绕过；step440 的 65 个中有 26 个核心计算绕过

以下四类证据处于不同语义层，不能把计数相加成“失败样本分解”：

| 类别 | 回答的问题 | 本报告如何处理 |
|---|---|---|
| 题目/reference 质量 | 题目是否测到了名称宣称的能力 | 标出语义错误、退化计算和误导性描述；不事后改正式分数 |
| evaluator/checker 漏洞 | 错误或违规答案是否被误判为成功 | 用静态规则给出可识别绕过下界；未命中不等于合规 |
| runtime infra | 请求是否因服务或控制面故障而没有被正常测量 | 单列断连和 0 请求启动失败，不混入模型错误 |
| 模型失败 | 在给定 literal reference 和 evaluator 下答案为何失败 | 分为缺代码、接口、编译、数值错误和性能不足 |

最强结论是，step440 的 Level2 正式 Fast 优势主要不是 CUDA kernel 变强，而是
checker 没有拦住一种规避方式：候选在 `forward` 中直接调用 PyTorch 的 convolution
或 transposed convolution，扩展只做后处理。step440 的 225 个 Fast@1.0 中有 190 个
命中这类核心计算绕过；step620 只有 8/47。去掉静态可识别的绕过后，两者 Fast@1.0
变成 35 对 39。全量 AST scan 已确认题80/83在 step440 和 step620 各 16 个候选
均未命中该绕过规则，因此再把两道恒为零的退化题贡献置零且仍以 800 为分母，剩
19/800（2.38%）对 24/800（3.00%）。若直接移除两题并改用 784 作分母，则是
2.42% 对 3.06%。因此不能把正式 28.12% 对 5.88% 解读为 step440 的自定义 CUDA
比 step620 快约五倍。Fast@1.2 更明显：把静态命中和两道恒零题贡献置零后，
step440/step620 从正式 127/39 变为 4/17（仍以 800 为分母）

step440 的 Level2 正式 Correct 也高 48 个，但同样不能解读为合规答案更好。它比
step620 多 234 个“绕过且 Correct”，却少 186 个“未命中绕过扫描且 Correct”；
两者抵消后才得到正式 `+48`。step620 的输出也明显更长，71 个样本被截断且其中
66 个未编译；step440 只有 3 个截断。因而 Correct 差距是 checker 漏洞、答案结构和
完成率共同形成的正式评分结果，不能归给单一训练机制

Level3 复现了同一结论。step440 正式 Correct 是 65/400，高于 step620 的 40/400；
但逐个检查 65 个成功候选后，有 26 个仍用 PyTorch 完成 Conv/BN/attention/
Transformer/LSTM/GRU/Linear 等核心计算。按严格扩展意图作诊断，step440 是
39 Correct、3 Fast@1.0、1 Fast@1.2，step620 是 40/6/4。正式 `+25 Correct` 恰好是
`+26` 个绕过和 `-1` 个完整扩展 Correct 的合计，也不是完整自定义 CUDA 能力提升

失败来源不是单一因素。数据集中确有会改变能力解释的题目问题；评测基础设施有
一条 Level3 服务断连和若干发生在 0 请求阶段的 Ray 启动失败；其余大多数失败是
模型没有完成代码、接口不合规、编译失败、数值不正确或自定义 kernel 过慢
正式历史分数保留不改，报告同时给出静态可识别绕过的诊断口径，避免把 evaluator 漏洞和
退化题收益当作模型能力

## 三轮 12K：ATen legality gate 下的端点

L1/L2/L3 使用官方 100/100/50 题，每题 8 条独立轨迹。生成参数为 seed 42、
temperature/top-p=1、12,288 context/response，最多 3 轮；KernelGym 使用同一 A800
worker 池和 `low/32/32`。每个 H20 任务只加载一次模型，依次评测三个 level

Compile 表示一条轨迹在最多三轮内至少一轮编译成功；Correct 还要求数值正确且
`decoy_kernel=false`；Fast@1.0/@1.2 再要求 speedup 达到阈值。三项始终使用
800/800/400 的完整轨迹分母。单轮 T2/T3 只覆盖上一轮尚未结束的轨迹，不能独立解释
反馈收益；累计 `Best@Tk` 回答“给到第 k 轮后是否曾经解出”

step440 和 model0731 在 node53、step620 在 node70，均为 8xH20 和 fresh Docker
`--init` 容器
两个节点使用相同代码 fingerprint、DSpark base、提示和 evaluator；image 的 65 个
base/runtime layers 相同，最终 JIT/cache layer 为宿主本地副本。节点和独立随机生成
使它们不等同于逐 token 配对 A/B；绕过归因依赖候选源码与 ATen metadata，而非只看
分数相关性

step440/620 使用同一 DSpark base，只改变 LoRA checkpoint；曲线标签遵循
`step = iteration + 1`，分别来自 `iter_0000439` 和 `iter_0000619`。model0731 使用
另一套完整权重；它的非权重配置与提示合同相同，但不构成“同一 base 只去掉 LoRA”的
配对消融



### 结果与累计 best

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
Compile/Correct/Fast 主结果
step620 相对 model0731 的 L2 Correct 高 62.875 个百分点，L3 高 11 个百分点；
model0731 的 L2 Fast@1 虽高于 step440，主要来自 GEMM/Matmul/BMM 的 27 条轨迹，
不是 convolution 组合能力接近 step620

三轮反馈对正确性的累计增益如下；在两个 LoRA checkpoint 中，T3 相对 T1 的最大增益
出现在 step620 L2，达到 14.875 个百分点。三套端点整体的最大增益是 model0731 L1
的 28.625 个百分点

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
L1/L2/L3。这是固定 12k 合同下的答案长度边界，会压低可获得的 T2/T3 增益
model0731 的 L1/L2/L3 分别有 786/944/662 条 record 为 `status=truncated`，占各
level record 的 38.8%/59.2%/85.6%；260/466/237 条轨迹以
`finish_reason=prompt_truncated` 结束。L3 只生成 773 条 turn record，远少于最多
1,200 条，12k 输出/上下文合同明显限制了后续反馈机会

## 三轮 32K：独立采样的补充端点

按后续要求，model0731、step440 和 step620 又在 32K context/response 合同下完成
独立三轮评测。三个 checkpoint 都使用官方 L1/L2/L3 100/100/50 题、每题 8 条轨迹、
seed 42、temperature/top-p=1、同一提示与 evaluator/ATen legality gate；
`rollout-max-context-len`、`rollout-max-response-len`、SGLang context length 和 max
prefill tokens 均为 32,768。model0731 在 node53 运行；node70 在 model0731 运行期间
另起 step440 -> step620 的严格串行队列。三次运行都没有使用 eager mode

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
2400/2400/1198，model0731 为 2396/2396/1171

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
得出的“不触发”判断。对应 explicit-low 重跑结果见本文下一节。本段原始 model0731 endpoint 的启动命令只有
`--apply-chat-template-kwargs {"enable_thinking":true}`，不能回写成显式 low

step440 的 L1/L2/L3 record-level truncation 为 0%、0.0833%、1.0067%；step620 为
0%、0%、0.0835%。32K 明显解除旧 12K 合同中的大量输出截断，但前后是独立随机生成，
不能把所有分数变化唯一归因于 context 长度

### 32K 产物身份

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
  `c2bee4f0578b5bb41f25793b7a80f802d370697df2d380f9747460602b22574c`

执行偏差：node70 的冻结评测 snapshot 仍把 context/response 硬编码为 12,288；为避免
修改冻结目录，本次把已支持环境覆盖的 runner 放到 node70 `/tmp`，显式注入
`EVAL_MAX_CONTEXT_LEN=32768` 和 `EVAL_MAX_RESPONSE_LEN=32768`。node70 当时没有
step440 adapter，因此从 node53 复制后用 SHA256
`896a90674898773b00f5af3403b47f9750a19506845c966591c6cee81f9061a5` 核验，再启动
串行队列。两项 workaround 均在启动日志中留有合同证据；正式运行参数和产物未留缺口

## Model0731 显式 low 的复测边界

原 model0731 32K endpoint 的 L3 Turn 3 effective truncation 为 21%，超过用户指定的
10% 阈值。为排除 reasoning 配置没有被显式传入的疑问，node53 的 8 张 H20 又完成了
一次独立三轮评测。严格 chat template 只接受 `enable_thinking=true` 和
`reasoning_effort=low`；实际 job 命令、`GenerateState` 和 dump 中保存的 template
共同证明参数被消费。官方 low prefix 为空，预启动对照又确认严格模板与旧默认-low
模板逐字节、逐 token 相同，因此这次运行验证的是配置显式性；两次独立采样的分数差
不能解释成 low reasoning 的因果效果

这里的严格模板是本次重跑专用的冻结 artifact：
`local_artifacts/deepseek-v4/kernelbench_3turn_20260801/chat_template_model0731_explicit_low.jinja`
它不等同于仓库通用的 `deepseek_v4_chat_template.jinja`，也不用于证明通用模板已经实现
`reasoning_effort` 分档；它只通过“缺失或非 low 立即报错”证明本次请求显式消费了 low

正式 job succeeded，三个 level contract 与 `eval_suite` 全部 PASS；800/800/400 条
轨迹没有 wall-clock abort 或缺失 `env_result`，SGLang retract 为 0。运行使用 32,768
context/response/max-prefill、三轮、100/100/50 题、每题 8 条、seed 42、
temperature/top-p=1、KernelGym `low/32/32`，没有 eager mode。一次性容器在结束后
删除，8 张 H20 已释放

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
2/2400（0.0833%）、L2 31/2398（1.2927%）、L3 92/1183（7.7768%）

分数存在一项新的测量边界。dump 中有 81 条明确传输失败：L1/L2/L3 为 14/58/9 条，
覆盖 14/57/9 条轨迹；旧默认-low 运行的 exact server-disconnected 是 8/11/9 条
这些响应已有完整生成源码，但 KernelGym 没有返回候选 verdict。上表仍是正式固定分母
观测值，应读作下界；若所有传输失败候选都命中，Best Correct 的最大上修为 L1
0.50、L2 2.875、L3 1.25 个百分点。这只是上限，不是校正分数。候选自己的编译失败、
错误结果、非法地址、subprocess crash 和 300 秒 task limit 仍按模型代码结果处理
运行后服务 queue 为 0，48 CPU + 16 GPU worker 全部 online；传输失败根因尚未闭合，
也没有对 81 条做部分重评。因此本次分数不能用于声称 reasoning 配置带来提升或下降，
但它不改变 Turn 3 是否生成及 `status` 的结构统计结论

人工抽检覆盖每层正常、T3 truncated、无 T3 和传输失败轨迹。正常轨迹都保留三段真实
源码及嵌套环境结果；代表性的 T3 truncated 为 L1 group 545、L2 group 14、L3
group 28，无 T3 为 L2 group 665 和 L3 group 50，传输失败为 L1 group 43、L2
group 22、L3 group 399。完整审计见
`local_artifacts/deepseek-v4/kernelbench_3turn_20260801/model0731_explicit_low_audit_20260803.txt`

正式 dump/summary/run log/orchestrator SHA256 为
`3971bffca940f45848cf899d0632ee3dbe93b0c52911bcd9efdec4cb614bab15` /
`0ae1ef568ec5fc6684afdbc87ec740cbf463f5fc0dbf7b77c4ea1c3cfc821d4e` /
`4c2edf90a05afbe43745f60d9b523048d8097cc85088186facfa3bbcb29d7cec` /
`7fd5cc366bb107eaaf38140822b211ea6cbffd2cb0f0515893b03ef4752102c7`
首次启动因过长 result tag 使 Ray AF_UNIX socket path
超限，在 0 sample、0 KernelGym task 时退出并隔离；正式运行改用短 Ray temp tag
`m0731low` 后完成。该零样本尝试没有污染正式分母

## 题目语义与判分敏感性

### 逐题审计

这里按影响分级。“题目有问题”不等于请求失败，也不自动允许事后删分；只有先定义
固定排除口径并重算，才能发布修订指标

### 会显著扭曲能力解释

- Level1 题73 的 reference 语义已确认错误，会改变本次 Correct 判定。它是 Level1
  全库扫描中唯一已证实会改变本次评分方向的题。题50、85另有未使用参数，题87有
  命名错误，但没有证据表明它们改变 step620 的评分方向
- Level2 题80 数学上恒为零：keepdim max 得到单通道后，又减去该单通道自身的
  mean；题83 先 `min(x, 0)` 再 `clamp(min=0)`，也恒为零。step620 在两题得到
  15 个同时达到 Fast@1.0 和 Fast@1.2，占其全部 Fast@1.0 的 15/47、Fast@1.2 的
  15/39；按题分别为题80 8/8 compile、7/8 correct/Fast@1.2，题83 8/8
  compile/correct/Fast@1.2，两题贡献 step620 Fast@1.0 的 31.9%。step440 两题有
  16 个同时达到两个阈值
  这些是在字面 reference 下合法、但不能代表 kernel 优化能力的收益
- Level3 题28 把 `[B, tokens, C]` 送给默认 `batch_first=False` 的 attention
  缩小的确定性运行显示不同样本 logits 完全相同，改变任一图像输出仍不变；这是
  reference 维度语义错误，而不是普通的模型困难题

### 字面代码可评分，但题名、注释或参数误导

- Level2 题26 名称/文档写 Add + HardSwish，代码却计算
  `(x + add_input) * hardswish(x + add_input)`；题58 文档写 HardSwish，实际是
  `x * sigmoid(x + 3) / 6`。题11、19忽略 `groups`，题48忽略传入的标量
  `scaling_factor` 并另建随机参数，题58、72忽略 `bias_shape`
- Level2 题18、42、44 含 singleton 维上的冗余 reduction/pooling，增加题面复杂度
  而不增加有效计算
- Level3 题24 名为 EfficientNetB2，但把 adaptive pool 和 sigmoid 直接串进每个
  MBConv，且没有把 gate 乘回 spatial feature；运行时五个 block 的输出都已变成
  `1x1`。题45 名为 UNetSoftmax，但最终输出没有 softmax。题50 声明并注释了
  `c_proj`，forward 却完全不调用；覆盖其权重后输出差异为 0

这些题的 literal reference 都能执行，所以当前 evaluator 按代码评分是确定的；问题
在于它们测到的不是名称或注释宣称的模型。报告将其与真正的 infra 故障分开

## 已确认的基础设施与模型边界

- 三轮 12K 的 step440/step620 各有 1/2 条精确 wall-clock sentinel，已完成的 T1/T2 未随 sentinel 保留；固定分母评分分别有最多 0.125/0.25 个百分点的向下不确定性。没有足以同时闭合旧 turn-record 总数与 prompt-truncated 终止计数的单一审计产物，因此不再发布由它们推导的精确截断率
- model0731 的 guard=3000 尝试和 KernelGym pool 污染尝试被排除。pool 的已证问题是回收进程被重新加入 idle list，以及死亡 canonical/busy 条目压制 top-up；修复 `2625505` 后的 16-GPU smoke 和正式 dump 才进入表格
- DSpark 权重目录曾被 0731 shard 覆盖，经权威副本逐文件 checksum 后替换。独立 model0731 不是“同一 DSpark base 去掉 adapter”的实验臂，不能将差异归给 LoRA 一个因素
- ATen legality 判断实际 candidate-forward compute。Kernel time coverage 只回答生成 kernel 的时间占比，不能代替来源判定；extension 内 cuBLAS/cuDNN 的归属需要正例校准
- 候选的 undefined symbol、非法地址或 subprocess crash 有候选调用栈时按模型代码失败处理；服务池或传输故障需独立证据。冻结候选重放不构成新模型生成分数

单轮逐题失败正文、case 定位和诊断结果仍可从下列 evidence 入口复核。更广的 verifier/reward 研究方向由[研究优先级](../in_progress/kernel_agent_research_priorities.md)维护，不在每份评测报告重复列待办

## 单轮证据

- Level1 全题与候选归因已合并到“Level1 step620 逐题审计”一节。正式根目录为
  `/ssd/csl_v4_h200_eval_20260727/experiments/Eval.KernelBenchL1.DeepSeekV4FlashLoRA.12k.turn1.n8.h200/step620/`；
  dump/summary/log SHA256 依次为
  `4a867fc66ed688faa8ff48d3568c0f61af91e5ae8555959d051473148a5ea39d`、
  `1da0e05de6fe4b05f57cc013a14f400ac5f09dea7afa96329bb65693311e5fca`、
  `8644072ad7328253f7f9a2531ec5ae7da70b01a476077b14f97ff4184d7e6882`
- step620 Level2 正式根目录为
  `/mnt/md1/csl_v4r21_fp4_pp1cp2_12k_dppo_predictive_resume40_20260722/h20_eval_20260729/experiments/Eval.KernelBenchL2.DeepSeekV4FlashLoRA.12k.turn1.n8.h20/step620/`；
  dump/summary/log SHA256 依次为
  `33b0d97f66f0ef3e449651d1a92bf1d7a606016961fad645dbae0244185783e3`、
  `baf248ba0310bfe0d42881515a67644120fa2c50995810fab49588f8c4206e15`、
  `9dec37d0b082337ac29c30ff672467d29731478a9d4e996751f40e363738b99d`
- step620 Level3 正式根目录为同一实验根下的
  `Eval.KernelBenchL3.DeepSeekV4FlashLoRA.12k.turn1.n8.h20/step620/`；
  dump/summary/log SHA256 依次为
  `b7d84a6ea799179a2dd8a10f6d75fdff66a9af0306462a0855d27483620e10f8`、
  `8a4f2ad96c8c6e87488ef05a8c98a32bc7289e8d4ba68c5dc8efcf52f4180fcc`、
  `51ea947b8b4392c4cd2bbce6af5aee237867aa6559aa46b2ed7c0d04cc7f1fb3`
- Level3 逐题、逐族、全部 Correct 人工检查和四题 reference 运行证据：
  `local_artifacts/level3_error_analysis/report.md` 与
  `reference_semantics_evidence.json`
- Level2 AST 实际调用扫描、全 2,400 样本清单和规则说明：
  `local_artifacts/level2_error_analysis/report.md`、`RULES.md`、
  `sample_scan.tsv` 和 `primary_hits.tsv`。report/script/manifest SHA256 分别为
  `885e26bc01830696340e75984b5182c9b12e809b965dab0dad0ad58d0bec1243`、
  `289fe490c4422543a27002a351ad55b05dbad835370b03cc63526067d498bb3c`、
  `15c8810a694aea237a38b45ff406f54cb8776b6be7f877e2e0a9d32058217fd6`
- step440 Level3 完整性、逐题/题族、26 个人工确认绕过和 Fast 明细：
  `local_artifacts/level3_error_analysis/step440_comparison.md`，SHA256
  `9e313641ec5737152cc865fe053cf7c94b2c610ce029a2f62230be780cd158b5`；
  复现脚本 SHA256
  `ac8b8d25b5a764931a2e1fe45c3ea0ac57940b6ba39b31a8a7811b99910d71aa`
- idx186 固定重放包：
  `local_artifacts/level3_error_analysis/idx186_replay/manifest.json` 和
  `result.raw.json`、`result.md`、`manifest.after_replay.sha256`。后三者 SHA256
  分别为
  `815249d9f6e7af5bf325495ff586a3fb36ef49dbc2f8523bac4cdd96f5ef383b`、
  `f858d54b3823942a3da1b77132d3431ee67968c79c3c5893202dbab93f21b303`、
  `4443504ccf0821ecebeceb13af58c5b086e2f18373d7efefdcf40c1604eaa06a`；
  请求 SHA256 为
  `1e1ab6f506ebdbde5bc2746d9f7fa801eede606ced69bf0fcc02c2d8710e964c`
- step440 Level2 独立审计：其正式 step 目录下 `audit.20260729.txt`，SHA256
  `945b227eef7b4df1d67e1ae404f94dcbce0dcc74acd495c1c960efcd9c1587e9`
- node64 三次 0 样本启动失败审计：其 Level3 step440 目录下
  `node64_startup_failures.20260729.txt`，SHA256
  `8c0765d097895dcfc34aa94f5ba7d125f3ae6156a69b45f05a755044b577445b`
- step440 Level3 正式根目录：node53
  `/mnt/md1/csl_v4r21_fp4_pp1cp2_12k_dppo_predictive_resume40_20260722/h20_eval_20260729/experiments/Eval.KernelBenchL3.DeepSeekV4FlashLoRA.12k.turn1.n8.h20/step440/`
  dump/summary/formal log/host log SHA256 依次为
  `dc36ff364ab534c409e4a7f8eae95cb8897b881601201648f30c476807c3ee49`、
  `c05583f200083ca208f7e01772a5f43793bfdeb4055cb19f24aa0cc656e7ba87`、
  `c8c07fe597fc70a066b280fcb98e51d87cb4fe67257712290c794eb3c8c1880c`、
  `1579b9ce773458734a5b78bbd7a38fc0373094d7f2625006d4b0b82166e600af`
- step380 Level3 正式根目录：同一 node53 实验根下的 `step380/`。dump 和 summary
  SHA256 分别为
  `6da76e195eb139ab4ce11a76946631d50f747d7113b7e3bfc5133c6198a15fd1` 和
  `37e4b6b2e0793eb9559084546ea556fa1470a0af691cb90bd913d4a89c374c12`；runner
  PASS 后 disposable 容器已删除，Ray 端口和八张 H20 均已释放

## 三轮 12K 证据

完整 claim-evidence brief 位于
`local_artifacts/deepseek-v4/kernelbench_3turn_20260801/claim_evidence_brief.md`。逐题题目
质量结果见本文前述各端点；L1 训练曲线见
`handoffs/data/cleaning/handoff_drkernel_and_accepted_v5_cleanup_20260729.md`

step440 产物位于 node53：
`/mnt/md1/csl_v4_kernelbench_3turn_20260801/experiments/`
`Eval.KernelBenchL123.DeepSeekV4Flash.step440.12k.turn3.n8.h20/step440/`

- dump/summary/orchestrator SHA256：
  `e3f880c8856aadd65f3c4df36f2f2edab7d244b9a031cfef245f0f3d2e73bdc8` /
  `044a7bf8b9106552cc38c3a2953632a74b08823ed9dc39cc56f9ab3f04c4f30d` /
  `632b05a7dec94b5ebdbb43eff8dc4e7749a2237c91cb4be5772a7ef668a7bde1`
- adapter SHA256：
  `896a90674898773b00f5af3403b47f9750a19506845c966591c6cee81f9061a5`

step620 产物位于 node70：
`/mnt/data/csl_v4_kernelbench_3turn_20260801/experiments/`
`Eval.KernelBenchL123.DeepSeekV4Flash.step620.12k.turn3.n8.h20/step620/`

- dump/summary/orchestrator SHA256：
  `fd2f6a8c2f9e82496fe90fcb9af4afa7d8ae89f411f1d63d3899c26c1842539e` /
  `dbc47a70dbd82c99a9aeebb4d34fc3714270e80ad911e0be00196d13ee01a10e` /
  `5ad839daeae7bf02d82b2180447e9563b9ea7dcd75c5e1a87a24b64439980108`
- adapter SHA256：
  `7142cc57e4d79513dda2738bec0d63d6679d23a82e47e58e23e5381107edab62`

model0731 产物位于 node53：
`/mnt/md1/csl_v4_kernelbench_3turn_20260801/experiments/`
`Eval.KernelBenchL123.DeepSeekV4Flash.model0731.12k.turn3.n8.h20/model0731/`

- dump/summary/orchestrator SHA256：
  `8bbb14f0b400e50f3a202182d544a6435dedd7b50daa14466d3339d776924077` /
  `6a0009520ff5b3d05ae2de3d36126d20ec43d309c69794a73b9f87c53c4321ae` /
  `4970c3730a8cbc06f1c32155e2875e743686520692dac9ffea3f630257894676`

三份官方数据 SHA256 为 L1
`e034d42fe5e8ed0fac0e580bb9070379f719080b05d666accbac59abc081435f`、L2
`11c1858d88be14ebc7fa766390f46a0db1ad872e921ddc61411e7df4d2e64cfe`、L3
`6b3c85f2e57f307036b8c38ff26891e8b88ac44afb702a7c7f539d2b8307755e`。runner SHA256
为 `252fc3fe3cc3992bf2753c4bec9ddb3128874d9e649477f7bd6efc01e6eac695`；KernelGym
根修复提交为 `26255057463a77b23abac0f3e5eafeeebf2ebbb5`
