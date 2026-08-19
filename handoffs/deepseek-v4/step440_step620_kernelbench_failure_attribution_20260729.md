# step440 与 step620 KernelBench 失败归因

更新时间：2026-07-29

## 摘要

这份报告回答三个问题：step440 为什么在 Level2 上显著高于 step620，失败样本究竟
来自题目、基础设施还是模型，以及这些分数是否代表真正的自定义 CUDA 能力。
它同时吸收原 Level1 step620 逐题审计与 Level2/3 正式运行证据，作为当前三个 level
已完成单轮评测的统一失败归因；Level1 的 step440 等曲线点仍由
`kernelbench_l1_lora_curve_20260724.md` 维护。

最强结论是，step440 的 Level2 正式 Fast 优势主要不是 CUDA kernel 变强，而是
checker 没有拦住一种规避方式：候选在 `forward` 中直接调用 PyTorch 的 convolution
或 transposed convolution，扩展只做后处理。step440 的 225 个 Fast@1.0 中有 190 个
命中这类核心计算绕过；step620 只有 8/47。去掉静态可识别的绕过后，两者 Fast@1.0
变成 35 对 39。全量 AST scan 已确认题80/83在 step440 和 step620 各 16 个候选
均未命中该绕过规则，因此再把两道恒为零的退化题贡献置零且仍以 800 为分母，剩
19/800（2.38%）对 24/800（3.00%）。若直接移除两题并改用 784 作分母，则是
2.42% 对 3.06%。因此不能把正式 28.12% 对 5.88% 解读为 step440 的自定义 CUDA
比 step620 快约五倍。Fast@1.2 更明显：把静态命中和两道恒零题贡献置零后，
step440/step620 从正式 127/39 变为 4/17（仍以 800 为分母）。

step440 的 Level2 正式 Correct 也高 48 个，但同样不能解读为合规答案更好。它比
step620 多 234 个“绕过且 Correct”，却少 186 个“未命中绕过扫描且 Correct”；
两者抵消后才得到正式 `+48`。step620 的输出也明显更长，71 个样本被截断且其中
66 个未编译；step440 只有 3 个截断。因而 Correct 差距是 checker 漏洞、答案结构和
完成率共同形成的正式评分结果，不能归给单一训练机制。

Level3 复现了同一结论。step440 正式 Correct 是 65/400，高于 step620 的 40/400；
但逐个检查 65 个成功候选后，有 26 个仍用 PyTorch 完成 Conv/BN/attention/
Transformer/LSTM/GRU/Linear 等核心计算。按严格扩展意图作诊断，step440 是
39 Correct、3 Fast@1.0、1 Fast@1.2，step620 是 40/6/4。正式 `+25 Correct` 恰好是
`+26` 个绕过和 `-1` 个完整扩展 Correct 的合计，也不是完整自定义 CUDA 能力提升。

失败来源不是单一因素。数据集中确有会改变能力解释的题目问题；评测基础设施有
一条 Level3 服务断连和若干发生在 0 请求阶段的 Ray 启动失败；其余大多数失败是
模型没有完成代码、接口不合规、编译失败、数值不正确或自定义 kernel 过慢。
正式历史分数保留不改，报告同时给出静态可识别绕过的诊断口径，避免把 evaluator 漏洞和
退化题收益当作模型能力。

## 评测合同与正式结果

Level2 是 100 题、每题 8 个样本；Level3 是 50 题、每题 8 个样本。所有运行均为
单轮、12k context/response、seed 42、temperature/top-p=1，并使用 KernelGym
`low/32/32`。Compile、Correct 和 Fast 均以全部样本为分母；Correct 排除 decoy，
Fast 还要求正式 Correct。曲线标签遵循 `step = iteration + 1`：step440 和 step620
分别来自 `iter_0000439` 与 `iter_0000619`。

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
raw-correct decoy，step620 Level3 decoy 为 0，均已从正式 Correct 排除。

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
检查，没有发现同类核心计算绕过；step440 的 65 个中有 26 个核心计算绕过。

以下四类证据处于不同语义层，不能把计数相加成“失败样本分解”：

| 类别 | 回答的问题 | 本报告如何处理 |
|---|---|---|
| 题目/reference 质量 | 题目是否测到了名称宣称的能力 | 标出语义错误、退化计算和误导性描述；不事后改正式分数 |
| evaluator/checker 漏洞 | 错误或违规答案是否被误判为成功 | 用静态规则给出可识别绕过下界；未命中不等于合规 |
| runtime infra | 请求是否因服务或控制面故障而没有被正常测量 | 单列断连和 0 请求启动失败，不混入模型错误 |
| 模型失败 | 在给定 literal reference 和 evaluator 下答案为何失败 | 分为缺代码、接口、编译、数值错误和性能不足 |

## Level1 step620 逐题审计

本节合并原独立逐题报告，分析对象是 H200 上 step620 的唯一正式 dump：100 题、每题
8 个独立生成、共 800 条，四项指标都以 800 为分母。KernelGym correctness 使用
`atol=rtol=1e-4`，在首个失败 trial 停止；性能由同一 A800 worker 上的 reference 和
candidate 各 50 次试验得到，H200 只生成代码、不进入 speedup 计时。

| 口径 | 样本 | Compile | Correct | Fast@1.0 | Fast@1.2 |
|---|---:|---:|---:|---:|---:|
| 官方 step620 | 800 | 782 (97.75%) | 723 (90.38%) | 206 (25.75%) | 160 (20.00%) |
| 排除题73 | 792 | 776 (97.98%) | 723 (91.29%) | 206 (26.01%) | 160 (20.20%) |

题目质量审计分三层：100/100 reference 通过 AST、必需定义、构造签名和
`get_init_inputs()` 绑定扫描；100/100 又完成 `get_init_inputs -> Model` 构造和
meta-device forward；人工复核覆盖全部自动扫描异常、全部 compiled-but-wrong 题和
代表性正确候选；77 条非 Correct 样本均已归因，但没有逐行复核 800 条中的每个正确
候选。这能较强排除结构、绑定和明显语义问题，但不是对其余 99 题自然语言意图的形式证明。

### Compile 与 Correct 失败

18 条 Compile 失败均有候选侧终态，没有正式请求缺失或基础设施断连；其中 3 条
300 秒 timeout 只能证明任务真实入队并返回终态，不能证明候选可编译：

| 机制 | 条数 | 题/样本 |
|---|---:|---|
| 达到 12,286 token、无 EOS、缺完整 `ModelNew` | 9 | idx504/572/573/577/578/594/612/613/617 |
| `ModelNew.__init__` 不匹配 reference 构造参数 | 3 | idx348/601/631 |
| NVCC 或候选源码错误 | 3 | idx554/671/775 |
| KernelGym 任务达到 300 秒 | 3 | 题24 idx185；题97 idx769/774 |

782 条 Compile 中 59 条 compiled-but-wrong：47 条有大幅输出错误、NaN、运行错误或
明确的候选语义问题；6 条来自题73；另 6 条是并行归约顺序下的数值容差失败：题91
idx723/726/727 的 reverse cumsum（max diff 0.075195，avg diff 约 0.006）、题96
idx766 的 HuberLoss（max diff 0.000848）、题99 idx790 的 TripletMarginLoss
（标量差 0.003777）、题100 idx792 的 HingeLoss（标量差 0.001786）。固定候选重放
前不调整容差，也不事后改正式 Correct。

代表性真实错误包括题45 AvgPool2D：输入有 4,294,967,296 个 float，4 个失败候选用
32 位 `int` 计算偏移并越过 `INT_MAX`，同题另外 4 个安全索引候选全部正确。题97
SDPA 的 8 条中只有 2 条正确且 speedup 为 0.0302x/0.0474x；其余是输出错误、launch
out of resources、NVCC 失败或 timeout，没有证据指向 reference 或共享基础设施。

### 题73与数据卫生边界

题73 的 `Model` 把 `output_padding` 放在 `groups` 前，`get_init_inputs()` 却把第六个
位置写成 `groups=4`；reference 创建 `nn.ConvTranspose3d` 时又不转发
`output_padding`，所以字面执行实际是 groups=1。step620 的 6 个可编译候选要么按题面
实现 groups=4，要么按错误绑定值分配输出，均与 literal reference 不一致。跨 H200
step460–700 的 104 个生成只有 1 条正确；该 step640 候选在 reasoning 中明确识别了
位置绑定错误，并故意复现 groups=1/忽略 `output_padding`。这直接证明题目缺陷会改变
Correct 判定，排除题73的表格只是敏感性而不是修复后分数。

位置绑定扫描只发现六组“变量名不等于形参名”：题33–36 的 `features -> num_features`
和题47 的 `reduce_dim -> dim` 都是等义别名；题73 的 `groups -> output_padding` 是
唯一把值绑定到另一种参数语义的情况。这是“题73唯一改变评分方向”的负向扫描证据。

题50 的 `num_classes` 未使用，题85 的 `out_channels/groups` 被 depthwise 硬编码覆盖，
题87 把 `nn.Conv2d` 属性命名为 `conv1d`；它们是数据卫生或命名问题，但未发现改变本轮
评分方向。题59 的“square kernel”在代码中明确是 `(k, k, 1)`，8 个候选均复现该形状，
不属于题73式绑定冲突。

### 性能分布与逐题索引

723 条 Correct 中只有 206 条快于 PyTorch；517 条虽然正确但不够快：145 条低于
0.1x，186 条在 0.1–0.5x，96 条在 0.5–0.9x，50 条在 0.9–0.95x，40 条在
0.95–1.0x；另有 46 条处于 1.0–1.2x。65 条正确结果处于 0.95–1.05x、127 条处于
0.9–1.1x，Fast@1.0 对近阈值计时敏感且当前没有置信区间；427 条低于 0.9x，测量噪声
不可能改变“不够快”的结论。正确候选的题族统计如下：

| 题族 | 样本 | Compile | Correct | Fast@1.0 | Fast@1.2 | speedup 中位数 |
|---|---:|---:|---:|---:|---:|---:|
| Matmul，题1–18 | 144 | 100.00% | 95.83% | 1.39% | 0.69% | 0.2516x |
| Activation，题19–32 | 112 | 99.11% | 99.11% | 30.36% | 14.29% | 0.9210x |
| Norm，题33–40 | 64 | 100.00% | 93.75% | 56.25% | 43.75% | 1.1438x |
| Pool/reduction，题41–49 | 72 | 98.61% | 90.28% | 55.56% | 48.61% | 1.3295x |
| Convolution，题50–87 | 304 | 95.72% | 86.84% | 14.80% | 10.86% | 0.1203x |
| Scan/special，题88–93 | 48 | 100.00% | 79.17% | 18.75% | 16.67% | 0.4242x |
| Loss/attention，题94–100 | 56 | 94.64% | 83.93% | 71.43% | 69.64% | 2.3785x |

Compile 少于 8/8 的题及其 `(compile, correct, fast@1.0, fast@1.2)` 为：题24
`(7,7,1,0)`、44 `(7,7,5,5)`、64 `(7,5,0,0)`、70 `(7,7,0,0)`、72
`(6,5,5,5)`、73 `(6,0,0,0)`、75 `(7,6,0,0)`、76 `(7,5,0,0)`、77
`(6,6,0,0)`、78 `(7,6,0,0)`、79 `(7,7,0,0)`、84 `(7,7,7,5)`、97
`(5,2,0,0)`。compiled-but-wrong 涉及题 6、7、8、11、17、33、37、40、45、46、48、54、55、56、
57、58、59、62、64、67、71、72、73、75、76、78、87、89、91、92、93、96、97、
99、100。Fast@1.0 为 8/8 的题为 25、30、36、41、42、43、82、83、85、88、94、98；
Fast@1.0 为 0/8 的题为 2–13、15–18、20、28、29、39、45、47、48、50、
54–81（不含72）、86、87、90–93、97，共 58/100 题。主要能力缺口已经从
Compile/Correct 转向性能：100/100 题至少有 1 条 Compile，99/100 至少有 1 条
Correct，但只有 42/100 至少有 1 条 Fast@1.0。

## 为什么 step440 的 Level2 高于 step620

### 正式 Correct 差距掩盖了答案结构变化

step440 比 step620 多 54 个 Compile、48 个 Correct。条件化到已经编译的候选后，
两者 Correct 率只差 0.49 个百分点：

| checkpoint | Correct / Compile | completed / truncated | 响应长度中位数 | p90 | p95 |
|---|---:|---:|---:|---:|---:|
| step440 | 597/720 = 82.92% | 797/3 | 7,004 | 8,837 | 9,488 |
| step620 | 549/666 = 82.43% | 729/71 | 8,868.5 | 10,622 | 10,774 |

但这个条件比例把未命中静态扫描的实现和 PyTorch 核心绕过混在一起，不能单独解释
能力；未命中静态扫描也不等于已经证明合规：

| checkpoint | 绕过 Compile | 绕过 Correct | 未命中扫描 Compile | 未命中扫描 Correct |
|---|---:|---:|---:|---:|
| step440 | 265 | 248 | 455 | 349 |
| step620 | 14 | 14 | 652 | 535 |

正式 `+48 Correct` 实际是绕过部分 `+234` 与未命中扫描部分 `-186` 的合计。
step620 的 71 个截断样本中有 66 个未编译；它的 134 个未编译样本以 validation
和 compilation error 为主，step440 只有 80 个未编译。“输出更长、更多触及 12k
截断”在数量上足以覆盖 54 个 Compile 差距，但不是唯一因果解释；截断与答案模式
构成的相对贡献没有做反事实分解。证据能确定的是答案分布发生了大幅变化，不能把
正式 Correct 差距解释成 step440 的合规语义能力更高。

### Fast 差距来自 checker 漏洞

提示要求核心计算必须经过自定义扩展，不能在 `forward` 中用 PyTorch 完成。但当前
precheck 只拦住一部分显式 framework compute；当候选先调用 `self.conv(...)` 或
`self.conv_transpose(...)`，再让扩展做激活、bias 或其他后处理时，仍可能通过。
对最终 `MODEL_NEW` 的静态扫描结果如下：

| checkpoint | 可识别绕过 | 绕过 Correct | 绕过 Fast@1.0/@1.2 | 未命中扫描 Fast@1.0/@1.2 |
|---|---:|---:|---:|---:|
| step440 | 274 | 248 | 190/107 | 35/20 |
| step620 | 14 | 14 | 8/7 | 39/32 |
| base | 46 | 15 | 11/6 | 8/2 |

step440 的 190/225 Fast@1.0（84.4%）来自可识别的 PyTorch 核心绕过。按题族看，
ConvTranspose 的正式 Fast 有 109 个，其中 106 个是绕过；Conv 的 91 个 Fast 中有
80 个是绕过。Matmul/GEMM/BMM 的 25 个 Fast 中只有 4 个命中扫描，因此这一族的
提升相对更接近真正的扩展实现。Fast@1.2 也有 107/127（84.25%）命中，而 step620
只有 7/39；再把题80/83的 16/15 个贡献置零后，未命中扫描的 Fast@1.2 是 4 对 17。

人工对照也支持这个机制。step440 题2 idx10 直接调用
`self.conv_transpose(x)`，speedup 1.753x；step620 同题 idx14 在扩展里完整实现
ConvTranspose，只有 0.291x。题43和题93也呈现同样对照。95 道可比题的 reference
运行时间中位相对差为 0，67 道完全相同，最大差异只有题6的 16%，所以 reference
计时漂移不能解释 28.12% 对 5.88% 的 Fast 差距。

这项静态扫描是保守的诊断，不是新的正式评分器；它可能漏掉变体，但已足以解释
大部分差距。正式分数为了历史可复现保持不变。下一步应先扩充 precheck，覆盖
`self.*conv*()`、`self.*linear*()`、`self.*pool*()`、`torch.mm/matmul/bmm`、
`F.linear/F.conv*` 等核心调用，然后固定同一批 response 重评分。

复核时发现早期 scratch 统计曾给出 base/step440/step620=`106/274/39`，但无法用
文档声明的“最终完整 `MODEL_NEW` 中的实际直接调用”规则复现。token-aware AST 的
可重跑下界是 `46/274/14`；一个 lexical 版本得到 `48/274/15`，多出的 3 条只是
docstring/comment 中的调用示例。AST 命中且 Compile/Correct/Fast 的计数是
`20/15/11`、`265/248/190`、`14/14/8`。因此关于 step440 正式成功样本污染和 Fast
差距的结论不变，不可复现的旧总数及 lexical 误报不再作为主证据。

## 题目和 reference 的问题

这里按影响分级。“题目有问题”不等于请求失败，也不自动允许事后删分；只有先定义
固定排除口径并重算，才能发布修订指标。

### 会显著扭曲能力解释

- Level1 题73 的 reference 语义已确认错误，会改变本次 Correct 判定。它是 Level1
  全库扫描中唯一已证实会改变本次评分方向的题。题50、85另有未使用参数，题87有
  命名错误，但没有证据表明它们改变 step620 的评分方向。
- Level2 题80 数学上恒为零：keepdim max 得到单通道后，又减去该单通道自身的
  mean；题83 先 `min(x, 0)` 再 `clamp(min=0)`，也恒为零。step620 在两题得到
  15 个同时达到 Fast@1.0 和 Fast@1.2，占其全部 Fast@1.0 的 15/47、Fast@1.2 的
  15/39；按题分别为题80 8/8 compile、7/8 correct/Fast@1.2，题83 8/8
  compile/correct/Fast@1.2，两题贡献 step620 Fast@1.0 的 31.9%。step440 两题有
  16 个同时达到两个阈值。
  这些是在字面 reference 下合法、但不能代表 kernel 优化能力的收益。
- Level3 题28 把 `[B, tokens, C]` 送给默认 `batch_first=False` 的 attention。
  缩小的确定性运行显示不同样本 logits 完全相同，改变任一图像输出仍不变；这是
  reference 维度语义错误，而不是普通的模型困难题。

### 字面代码可评分，但题名、注释或参数误导

- Level2 题26 名称/文档写 Add + HardSwish，代码却计算
  `(x + add_input) * hardswish(x + add_input)`；题58 文档写 HardSwish，实际是
  `x * sigmoid(x + 3) / 6`。题11、19忽略 `groups`，题48忽略传入的标量
  `scaling_factor` 并另建随机参数，题58、72忽略 `bias_shape`。
- Level2 题18、42、44 含 singleton 维上的冗余 reduction/pooling，增加题面复杂度
  而不增加有效计算。
- Level3 题24 名为 EfficientNetB2，但把 adaptive pool 和 sigmoid 直接串进每个
  MBConv，且没有把 gate 乘回 spatial feature；运行时五个 block 的输出都已变成
  `1x1`。题45 名为 UNetSoftmax，但最终输出没有 softmax。题50 声明并注释了
  `c_proj`，forward 却完全不调用；覆盖其权重后输出差异为 0。

这些题的 literal reference 都能执行，所以当前 evaluator 按代码评分是确定的；问题
在于它们测到的不是名称或注释宣称的模型。报告将其与真正的 infra 故障分开。

## 基础设施问题

已发现的基础设施问题很少，且没有证据能解释 step440/620 的主体差距：

- step620 Level3 题24 idx186 的结构化错误是
  `Server disconnected without sending a response.`。它在正式日志中没有出现，
  只保存在 dump 的 sample-level error；先前仅搜索小写字符串而漏报。该条使 Compile
  最多有 `+1/400` 的原始不确定性。正式运行结束后的一次固定辅助重放确认它
  `compiled=true`、`correctness=false`，首个 trial 的 max difference 为 1.613391。
  原 dump 和正式分数未改；诊断反事实只是 Compile 274/400（68.50%），Correct/Fast
  不变。
- step440 Level3 在 node64 的三次预启动失败都发生在 Ray dashboard agent 发布
  端口之前；每次均为 0 KernelGym POST、0 dump、0 summary。已验证的失败机制是
  dashboard agent 的七个模块加载总计需 24--32 秒，超过 raylet 约 15 秒的端口文件
  等待，因而形成注册竞态。延迟已定位到 ReporterAgent/GPU profiling 附近，但 NVML 或既有
  watch 会话只是高概率触发因素，尚未做完单因素证明。固定 agent 端口、提前发布
  listen port，并在提交 job 前等待 `/api/healthz` 后，正式运行才进入模型阶段。
  这些失败没有进入任何正式分数。
- node53 在正式成功前另有两次 0 样本失败：第一次 Ray start 失败；第二次虽然提前
  发布固定端口，但 job submit 返回 500 `No available agent`，直接证明还必须等待
  `/api/healthz`。两次 formal/host log SHA256 分别为
  `10a87f815d50dde729d21fedfb57c79f2a36d70870f922328935da8f9c808461` /
  `7e4d95a027bb4aca6653510d1bc71e0e57c86f445df7d002ce11586c0e50fd17` 和
  `a063e7deb6a288513190ab3c5c7aadd119c0aac05f122940dccec909b858b035` /
  `d53abcf5e314e5f52da73a185a3a83e825008070233f678cb7e2f1c4bcb56f6e`；
  两份 formal log 的 KernelGym POST、eval progress、summary 计数均为 0。
- step440 Level2 的正式 dump 完整，但 EXIT 日志曾误报容器已删除；容器实际是
  `Exited (137)`，`OOMKilled=false`。人工删除后，runner 已改为重试删除并用
  `docker inspect` 验证容器确实不存在。这是生命周期清理缺陷，不是样本评分故障。
- step620 Level2 完成后的第一次只读 dump 审计容器使用默认 bridge，因宿主缺
  `/usr/sbin/iptables` 而在加载 dump 前退出；改用 `--network host` 后独立审计通过。
  正式 Level2 运行本来就是 host network，不受这个审计容器偏差影响。
- step440 Level3 的 400 个样本全部有 env_result，dump 内不含大小写不敏感的断连、
  ConnectError、RaySystemError、OOM 或磁盘错误。host/formal log 在模型启动时各记录
  两条 `503 Service Unavailable`，但都发生在 DeepGEMM warmup 期间；健康检查在首个
  eval 样本前恢复 200，没有样本承载该错误。唯一 timeout 是题2 idx13 的 300 秒
  task outcome。正式容器已删除，Ray 端口和 8 张 H20 均已释放。
- 这次 fresh `--init` 容器没有留下 eval 进程或 zombie。结束后的宿主 19 个 zombie
  中，15 个是运行前已有、父进程属于长寿命 `sleep infinity` 容器；另 4 个的父进程
  是同时运行的独立数据清理任务。没有杀其他任务的父进程，也没有把宿主总数误归给
  本次评测。这里的 19 是 step440 结束时刻的瞬时值；并行清理任务退出后，step620
  两次正式运行结束时宿主已恢复到基线 15。

Level2 的三个正式运行均未发现连接断开、RaySystemError 或服务 OOM。KernelGym 的
300 秒 timeout 计为候选/任务结果而不是自动归为 infra；step440 有 8 个，step620
有 14 个。只有结合重放或服务侧证据才能进一步改分类。

## 模型真正做不好的题型

### 生成完成度和接口

base 的主要问题是不会稳定给出完整 `MODEL_NEW`：Level3 有 157/400 缺失代码段、
155 个响应截断；step620 已改善为 68/400 和 69 个截断，但仍是最大的非编译来源。
Level2 到 step620 又出现输出变长和 71 个截断，说明“写完可执行答案”仍是能力瓶颈。
此外仍有 TVM-FFI 接口、shape/dtype/device 和 framework-compute precheck 错误。

### 复杂整网语义

step440 和 step620 Level3 的 Compile 接近（69.25% 对 68.25%），正式 Correct 是
16.25% 对 10.00%。按架构族拆分后，step440 的 25 个正式 Correct 优势来自 MLP
`+6`、Transformer/attention `+9`、RNN `+11` 和 CNN `-1`；两次在 Mamba、NetVLAD
都是 0 Correct：

| 题族 | 样本 | step440 Cpl/Cor/Fast | step620 Cpl/Cor/Fast |
|---|---:|---:|---:|
| MLP | 24 | 20/18/0 | 17/12/0 |
| CNN | 200 | 127/20/3 | 147/21/6 |
| Transformer/attention | 64 | 40/13/0 | 50/4/0 |
| RNN | 80 | 76/14/4 | 51/3/0 |
| Mamba | 16 | 10/0/0 | 2/0/0 |
| NetVLAD | 16 | 4/0/0 | 6/0/0 |

2026-07-29 补测的 step380 Level3 正式点为 268/400 Compile（67.00%）、73/400
Correct（18.25%）、7/400 Fast@1（1.75%）和 2/400 Fast@1.2（0.50%）。400 个
样本全部有 env_result；状态为 268 completed、128 failed、4 timeout。该点尚未做逐条
forward 合规人工审计，不能把 73 个正式 Correct 直接解释为完整扩展解题能力。

time coverage 已经能暴露一部分风险；当前训练同时使用 coverage reward 和软拒绝
采样，但 coverage 不进入正式 Correct 门槛。step380 的 73 个正式 Correct 全部有
coverage：最小值 0.0422%，中位数
99.7244%；8 个低于 1%，16 个低于 50%，27 个低于 90%。作为已完成人工标签的对照，
step440 的 26 个确认绕过中 coverage 中位数为 1.7325%，但范围从 0.0162% 到
99.3207%；39 个确认完整扩展样本的范围为 85.8536% 到 100%。因此 coverage 是有效的
风险排序信号，却不存在一个能在该审计集上无误区分全部绕过与全部合规样本的简单
阈值；正式门槛仍需结合静态/运行时调用来源检查。

step440 只有 11 个截断、12 个缺失 `MODEL_NEW`，step620 分别是 69 和 68；但
step440 有 58 个 framework-compute/code-bypass 被 precheck 拦截，step620 只有 2 个。
step440 另有 212 个 compiled-wrong、15 个编译错误、31 个其他接口错误、4 个
candidate-specific native/link crash 和 1 个 timeout；step620 的 compiled-wrong 是
233 个。这个对照表明 step440 更常写完代码，也更常显式违反扩展合同。

更重要的是，precheck 仍漏过了 26 个 formal-correct 核心绕过。代表样本包括：题5
idx39 的 AlexNet 全部走 PyTorch，扩展只 copy 结果；题28 idx223 的 patch Linear、
TransformerEncoder 和 MLP head 都走 PyTorch，扩展只加 positional embedding；题35
idx272/274/276 和题38 idx296/297/298/299/303 直接调用 `nn.LSTM`，扩展只做末尾
Linear；题41 idx321 的 `nn.GRU` 已给出完整结果，扩展是 identity dummy。人工逐个
检查后，step440 只有 39 个 formal-correct 候选的核心计算完整走扩展，反而比
step620 的 40 个少 1。正式 7 个 Fast@1.0 中有 4 个 recurrent bypass；完整扩展的
Fast 只剩题4 idx24 和题17 idx129/135，其中只有 idx135 达到 Fast@1.2。

233 个 step620 Level3 compiled-wrong 中，至少 131 个 response 明写
`placeholder`、`dummy`、`simplified` 或 `omitted`；这是词法下界，不是精确因果
分类。人工样本包括：题3 idx17 承认留下 placeholder 但仍能编译；题28 idx218 只
实现 shape 正确的最小 ViT；题39等 RNN 候选漏掉完整状态递推。模型经常先搭出可编译
框架，再省略 block、分支、状态或融合前后的数值语义。

Level2 同样有四道两次都 0 Correct 的代表题：题3 是 ConvTranspose3D、额外加权、
LayerNorm 和 pooling 的组合，step440 样本直接漏掉 `sum_weight`；题14、18、66
的候选使用看似代数等价的 reduction/softmax 重排，观测误差多在 `1e-1` 到
`1e-3`。前者是明确语义遗漏；后三者当前仍按模型数值失败计，容差是否过严尚未通过
固定重放验证，不能混为同一种错误。

### 真正的 CUDA 性能

排除静态可识别的 checker 绕过后，剩余样本中的自定义 convolution 和 transposed
convolution 经常比 reference 慢 10--100 倍；一个人工检查的 GEMM 正确候选约为
468 微秒，而对应 cuBLAS reference 约 8 微秒，speedup 约 0.017x。step620 Level3
的 6 个 Fast 也高度集中：题4 LeNet
只有 1 个，题17 SqueezeNet Fire 有 5 个；Fast@1.2 的 4 个全部来自题17。这说明模型
能在少数小结构上写出有效优化，但尚未泛化到完整网络或主流 GEMM/conv 族。

## Wide research：lazy optimization 需要哪几层防线

宽搜后的结论不是再找一个更好的 coverage 阈值，而是把四个不同问题拆开：数值是否正确、
实现是否遵守“核心计算由生成扩展完成”的合同、计时是否完整、题目本身是否可被退化解利用。
没有一种现有方法同时证明四件事。对本次 step440 最直接的修复是“静态允许列表 + 运行时
算子来源”合规门禁；coverage、隐藏输入、fresh container 和 LLM 审计都应保留，但只能放在
各自擅长的层。

这里的 `lazy optimization` 指 [Dr. Kernel](https://arxiv.org/html/2602.05885) 所说的
“框架完成主体计算，生成 kernel 只处理小尾巴”。论文示例中生成 kernel 仅占总 CUDA 时间
0.014%，有效 fusion 则占 86.15%；step440 确认绕过样本的最低 coverage 是 0.0162%，
机制和量级都相符。它不同于
[DeepReinforce DefenseKernelHack](https://deep-reinforce.com/defense_kernel_hack.html)
所称的 `lazy evaluation`：后者返回 Tensor 子类，把计算推迟到未计时的 `allclose` 阶段，
属于计时逃逸。

本节截至 2026-07-29 检查了 benchmark/evaluator 论文、作者官方实现或官方文档；除
DeepReinforce 和 Wafer 的维护者实战报告外，不用二手综述支撑技术结论。搜索结果按攻击面
重组如下，而不是按论文逐篇罗列。

| 攻击面 | 已公开的办法 | 能拦住什么 | 仍然挡不住什么 |
|---|---|---|---|
| 框架代算、空壳 kernel | [Kevin](https://arxiv.org/html/2507.11948) 对 `torch.nn`、`functional`、fallback 和空继承直接置零；[TritonRL](https://arxiv.org/html/2510.17891) 把 syntax/functional validity 独立于数值正确性，用 linter 加 LLM judge；[KernelGenBench](https://docs.flagos.io/projects/kernelgenbench/en/latest/features/anti-hack.html) 用 AST 允许列表，只允许 tensor 创建、dtype 和常量类 Torch API，并阻断 alias、`getattr`、`data_ptr/storage`、cache、`exec/eval`。 | 能直接抓本次 `self.conv/Linear/LSTM/...` 主体计算和常见 PyTorch fallback；允许列表比字符串黑名单更接近所需合同。 | 纯字符串规则会漏 module alias、调用图、C++/ATen 和预编译 vendor 路径；LLM judge 不是确定性证明。静态规则还必须明确哪些 allocation/shape/device API 合法。 |
| “确实调用过自定义 kernel” | KernelGenBench 的 Ghost Replay 先正常运行，再把 `@triton.jit` 函数替换为 no-op；输出不变即拒绝，NVIDIA 上再用 `torch.profiler` 检查 Triton 硬件 trace。[Dr. Kernel](https://arxiv.org/html/2602.05885) 也拦截 Triton launch。 | 抓完全未调用、只编译不运行和假入口。 | 这是存在性而非主体贡献证明。本次“PyTorch 主体 + 扩展后处理”在 no-op 后输出会变、trace 也有自定义 kernel，仍可通过 Ghost Replay 和 profiler 签名。 |
| 主体贡献/目标 DSL 占比 | [Atrex-Bench](https://arxiv.org/html/2607.14541) 单独报告 `@flydsl.kernel` 占 forward device time 的比例；Qwen3.7-Max 的 Correct 为 84.8%，FlyDSL adoption 仅 43.8%，由此显式暴露 PyTorch、vendor kernel 和替代 DSL 三类 shortcut。 | 把“数值正确”和“目标实现参与多少”分开，能显示本次 coverage 所反映的风险。 | Atrex 把 adoption 作为独立指标，而非合规硬门槛；任一标量占比都不能证明语义归属。step440 已有 bypass 99.3207%、完整扩展最低 85.8536% 的交叠证据。 |
| 固定输入、宽松容差、硬编码 | [KernelBench-Verified](https://arxiv.org/html/2607.16241) 用原分布、3 倍幅度、0.01 倍幅度和负值四组隐藏输入；[The Correctness Illusion](https://arxiv.org/html/2606.20128) 用 op-schema 共享符号维度生成边界 shape，采用 fp64 CPU reference、按 op/dtype 的绝对容差、ULP/NaN/Inf 统计，并保存完整输入快照和最小失败例。 | 抓正值假设、恒等/常量捷径、tail mask、单 shape 转写错误和部分低精度偷换；失败可逐字节重放。 | 对所有输入都调用 PyTorch 的 fallback 仍数值正确，所以 fuzzing 不能代替来源门禁。Correctness Illusion 的 10 个 bug 是作者构造的小语料，且其非连续 layout 在 Python 边界未真正覆盖，不能把 10/10 外推成真实模型检出率。 |
| 题目/reference 自身可被利用 | [robust-kbench](https://arxiv.org/abs/2509.14279) 在任务入库前检查输出范围、跨 seed 标准差、各轴变化和 input impact，并用 LLM 查 reference 的冗余/低效；同时提供多 init、多输入配置以及 forward/backward。过滤受污染任务后，其 KernelBench 平均 speedup 从 3.13x 降到 1.49x。 | 在训练前隔离恒零、近常量、输入不敏感、冗余运算和低效 baseline；直接对应本报告题73、题80/83、题28一类能力解释风险。 | 任务过滤改变 benchmark 定义，不能事后改写历史正式分数；有结构的稀疏/常量输出也可能是合法语义，规则需题级复核。其 LLM soft verifier 只有约 0.73--0.82 准确率，适合省 GPU 的预筛，不适合作唯一硬门禁。 |
| 异步、缓存、地址复用、计时篡改 | [DeepReinforce 实现](https://github.com/deepreinforce-ai/CUDA-L2/blob/main/defense.py) 强同步所有 stream、检查新增线程/原生 Tensor/storage/data pointer 和计时函数引用；[SOL-ExecBench](https://arxiv.org/html/2603.19173) 另做独立 subprocess、多随机 trial、输入 clone、L2 flush 和 data-pointer 移动；Atrex 使用频率锁检查、逐 iteration L2 flush 和 per-shape subprocess；[Wafer 的 10 类实战归纳](https://www.wafer.ai/blog/reward-hacks-field-guide) 再加入同地址换新内容的 pointer poisoning、NaN/Inf guard buffer、重复输入的 bitwise determinism 以及普通/强同步双计时。 | 抓 side stream/thread、Tensor 子类延迟计算、Python 或 C++ 指针缓存、旧显存/no-op、跨 trial 状态和伪造计时。 | fresh container 只清理跨评测状态，不能阻止同一评测内部缓存、框架代算或 side stream；强同步也不证明计算来自自定义 kernel。 |
| 越界、未初始化、race 和不可能性能 | [NVIDIA Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) 的 memcheck/racecheck/initcheck/synccheck 分别查越界或未对齐访问、shared-memory race、未初始化读和错误同步；Atrex 把超过 per-shape roofline（score `>1`）视为 evaluation error 而非截断分数。 | 抓“读垃圾却碰巧 allclose”、共享内存溢出、同步缺陷，以及明显超越硬件上界的错误测量。 | Sanitizer 很慢且有指令/动态并行等覆盖边界，适合 winner 晋级而非每个 rollout；roofline 是报警器，不给出漏洞根因。 |
| evaluator 未知漏洞 | [Hacker-Fixer Loop](https://arxiv.org/html/2606.08960) 让可读 verifier 的 hacker 找绕过、fixer 修 verifier、合法 solver 防误杀，并在共享 evaluator 基座内跨任务复用基础设施级补丁；其 KernelBench 回归集含 15 种公开攻击、133 个 hinted 实例，最终防御包括（非穷举）stack introspection、reference/solution 进程隔离、原函数捕获、Tensor 子类检测、全 stream 同步、随机 warmup 和不可预测 seed。 | 对固定规则没想到的组合攻击做持续红队；该工作在 KernelBench held-out 公开攻击上把 ASR 从 62% 降到 0%。 | 只防 hacker 找到的攻击。原始补丁曾把合法 `load_inline` 和自建 cuBLAS handle 一并拦掉；必须同时测合法 solver pass，论文用 autopatch 才在保持 0% ASR 时恢复到 92--98% benign pass。 |
| 训练信号和经验污染 | Dr. Kernel 用 `PR=T_generated/T_total` reward 与软 PRS；[CUDA-L1](https://arxiv.org/html/2507.14111) 在 reward 异常跃升时调用 adversarial checker，检索动态 hack 库中最相似 3 例，并截断组内标准化 reward；[daVinci-kernel](https://arxiv.org/html/2606.16497) 拒绝包含 PyTorch fallback/`torch.compile` 的技能，并把候选技能注回原任务复测，达标才持久化。 | 降低一次高额假 reward 对策略的放大，避免把逃逸技巧蒸馏进经验库，并保留探索。 | CUDA-L1 报告的 checker 检出率仅“超过 60%”；PR/PRS 和 reward clipping 都是训练塑形，不是 evaluator 合规证明。 |

### 对本次 checker 的直接判别方案

本仓库不是“还没有 coverage”。`examples/kernel_agent/kernel_reward.py` 已对正确样本加入
time coverage reward；`scripts/dsv4/_dsv4_task_args.sh` 已启用与 Dr. Kernel 相同的 PRS
`threshold=0.3`、`factor=0.1`，`examples/kernel_agent/utils.py` 按该概率删除训练样本。
真正的缺口在 `examples/kernel_agent/summarize_eval.py`：正式 Correct 只检查数值正确和
`decoy_kernel`，不检查实现来源。因此低 coverage 绕过可能在训练中被软过滤，高 coverage
混合实现仍可能保留，而两者在正式评测中都能成为 Correct。

对 step440 的最小可执行改动由三部分组成：

1. **静态合规允许列表**：解析 import alias、对象绑定和 `forward` 调用图；候选侧只允许
   扩展加载、tensor allocation、shape/stride/dtype/device 查询和明确列出的管理操作。
   `nn.Module.forward`、`torch/functional` 计算算子、Python `@`、ATen/C++ library fallback
   以及预编译 vendor op 是否允许，必须按 benchmark 合同逐项声明，不能靠函数名黑名单猜。
2. **运行时 operator knockout/provenance**：在 candidate forward 外围记录 dispatcher/profiler
   事件；若候选 wrapper 或扩展通过禁用的 ATen/框架计算路径执行，直接令
   `ContractCompliant=0`。同时要求预期生成扩展入口实际运行。这个组合分别证明“没有禁用
   代算路径”和“不是空壳入口”；Ghost Replay 只能证明后半句。运行时门禁必须在隔离进程中
   使用 evaluator 预先保存的原始函数引用，避免候选改写观测器。
3. **指标拆分**：至少公开 `Compile`、`NumericCorrect`、`ContractCompliant`、`Fast` 和
   `Coverage/target-device-time share`。`Compile` 应要求实际生成 AOT/扩展 artifact，而非
   仅成功 import；正式 Fast 应要求 Compile、NumericCorrect 和 ContractCompliant 同时
   成立。coverage 继续作为连续诊断和训练信号，而不冒充合规证明。

这里不声称一次 trace 就能形式化证明“全部语义都在扩展中”。它给出的是可执行合同：任何
禁用计算路径出现即失败，允许扩展入口未出现也失败。对 raw CUDA、TVM-FFI、Triton 和
允许的 vendor library 应分别维护可识别入口；无法归属的 device kernel 标记为 unknown，
进入晋级审计，而不是默认合规。

### 落地优先级与回归合同

| 优先级 | 改动 | 验证标准 |
|---|---|---|
| P0 | 实现静态允许列表、运行时 provenance、独立 `ContractCompliant` 和合规后 Fast | step440 的 274 个静态扫描命中样本成为攻击回归集；26 个 Level3 人工确认绕过必须失败。39 个 step440 与 40 个 step620 人工确认完整扩展样本，以及合法 TVM-FFI/`load_inline` 样本组成正例集，必须避免 Hacker-Fixer 已展示的过度拦截。 |
| P1 | 隐藏 value/shape/dtype 边界 fuzz；输入 clone、同地址换值、pointer churn、普通/强同步双计时、随机 warmup、reference/solution 独立进程 | 固定 seed 可重放；已知恒零、缓存、stream、Tensor 子类和计时 monkey-patch 样本全部失败；原正式 dump 不改写。 |
| P1 | 任务入库质量门禁：output range/std/axes/input impact、冗余语义和 baseline 效率审计 | 题73、80、83、28 类已知问题被隔离或带版本化标签；结构性合法常量任务经人工正例检查不误删。 |
| P2 | Fast winner 才运行 Compute Sanitizer、roofline/SOL sanity、双独立静态 judge 和人工复核 | memcheck/racecheck/initcheck/synccheck 无错误，性能不超过有效硬件上界，审计结论和输入快照可复现。 |
| P2 | 离线 Hacker-Fixer 持续红队，并把新 exploit 加入共享防御池和训练 hack 库 | 同时报告 held-out attack success rate 与 legitimate solver pass rate；任何防御只在两者均达标时合入。 |

不能单独依赖的方案也因此明确：只检查“执行过一个生成 kernel”、只看 profiler 中有目标
kernel、只设 coverage 阈值、只换新容器、只加隐藏输入、只用 LLM judge，或只搜
`torch.nn` 字符串，都会留下本次已经观测到或公开文献已经复现的逃逸面。分层方案的目的
不是把所有检测塞进每个 rollout，而是把廉价确定性门禁放在热路径，把高成本 fuzz、
sanitizer、双 judge 和人工复核放在 Fast winner/新 baseline 晋级路径。

## 决策与后续验证

1. 正式历史指标继续保留，因为它们可由原 dump 复现；模型选择时同时展示
   “排除静态可识别 PyTorch 核心绕过”和“把恒零题80/83贡献置零”的诊断口径，
   并明确这仍不是经过完整合规证明的新指标。
2. 在比较后续 checkpoint 之前先补 checker：核心计算调用必须完整路由到扩展，
   不能因为扩展参与了后处理就放行。实现上采用确定性的静态调用图与运行时 provenance
   门禁，coverage 保持训练信号；用 step440 的 274 个扫描命中样本做回归集。
3. 数据集发布修订版时优先修复或隔离 Level1 题73、Level2 题80/83、Level3 题28；
   其余命名、注释、冗余计算和未使用参数单独标注，不与确定的语义错误同级处理。
4. step620 Level3 idx186 的单次固定辅助重放已完成并单独记录；原 dump 和正式分数
   保持不变，不再重复重放。
5. 下一轮训练若要提升真实能力，应优先约束完整代码和禁止 placeholder，然后分别
   针对整网状态/分支语义与 GEMM/conv 性能优化；现有 PR reward/PRS 应保留，但只优化
   正式 Fast 或只调 coverage 阈值都会继续留下绕过空间。

## 可复核证据

- Level1 全题与候选归因已合并到“Level1 step620 逐题审计”一节。正式根目录为
  `/ssd/csl_v4_h200_eval_20260727/experiments/Eval.KernelBenchL1.DeepSeekV4FlashLoRA.12k.turn1.n8.h200/step620/`；
  dump/summary/log SHA256 依次为
  `4a867fc66ed688faa8ff48d3568c0f61af91e5ae8555959d051473148a5ea39d`、
  `1da0e05de6fe4b05f57cc013a14f400ac5f09dea7afa96329bb65693311e5fca`、
  `8644072ad7328253f7f9a2531ec5ae7da70b01a476077b14f97ff4184d7e6882`。
- step620 Level2 正式根目录为
  `/mnt/md1/csl_v4r21_fp4_pp1cp2_12k_dppo_predictive_resume40_20260722/h20_eval_20260729/experiments/Eval.KernelBenchL2.DeepSeekV4FlashLoRA.12k.turn1.n8.h20/step620/`；
  dump/summary/log SHA256 依次为
  `33b0d97f66f0ef3e449651d1a92bf1d7a606016961fad645dbae0244185783e3`、
  `baf248ba0310bfe0d42881515a67644120fa2c50995810fab49588f8c4206e15`、
  `9dec37d0b082337ac29c30ff672467d29731478a9d4e996751f40e363738b99d`。
- step620 Level3 正式根目录为同一实验根下的
  `Eval.KernelBenchL3.DeepSeekV4FlashLoRA.12k.turn1.n8.h20/step620/`；
  dump/summary/log SHA256 依次为
  `b7d84a6ea799179a2dd8a10f6d75fdff66a9af0306462a0855d27483620e10f8`、
  `8a4f2ad96c8c6e87488ef05a8c98a32bc7289e8d4ba68c5dc8efcf52f4180fcc`、
  `51ea947b8b4392c4cd2bbce6af5aee237867aa6559aa46b2ed7c0d04cc7f1fb3`。
- Level3 逐题、逐族、全部 Correct 人工检查和四题 reference 运行证据：
  `local_artifacts/level3_error_analysis/report.md` 与
  `reference_semantics_evidence.json`。
- Level2 AST 实际调用扫描、全 2,400 样本清单和规则说明：
  `local_artifacts/level2_error_analysis/report.md`、`RULES.md`、
  `sample_scan.tsv` 和 `primary_hits.tsv`。report/script/manifest SHA256 分别为
  `885e26bc01830696340e75984b5182c9b12e809b965dab0dad0ad58d0bec1243`、
  `289fe490c4422543a27002a351ad55b05dbad835370b03cc63526067d498bb3c`、
  `15c8810a694aea237a38b45ff406f54cb8776b6be7f877e2e0a9d32058217fd6`。
- step440 Level3 完整性、逐题/题族、26 个人工确认绕过和 Fast 明细：
  `local_artifacts/level3_error_analysis/step440_comparison.md`，SHA256
  `9e313641ec5737152cc865fe053cf7c94b2c610ce029a2f62230be780cd158b5`；
  复现脚本 SHA256
  `ac8b8d25b5a764931a2e1fe45c3ea0ac57940b6ba39b31a8a7811b99910d71aa`。
- idx186 固定重放包：
  `local_artifacts/level3_error_analysis/idx186_replay/manifest.json` 和
  `result.raw.json`、`result.md`、`manifest.after_replay.sha256`。后三者 SHA256
  分别为
  `815249d9f6e7af5bf325495ff586a3fb36ef49dbc2f8523bac4cdd96f5ef383b`、
  `f858d54b3823942a3da1b77132d3431ee67968c79c3c5893202dbab93f21b303`、
  `4443504ccf0821ecebeceb13af58c5b086e2f18373d7efefdcf40c1604eaa06a`；
  请求 SHA256 为
  `1e1ab6f506ebdbde5bc2746d9f7fa801eede606ced69bf0fcc02c2d8710e964c`。
- step440 Level2 独立审计：其正式 step 目录下 `audit.20260729.txt`，SHA256
  `945b227eef7b4df1d67e1ae404f94dcbce0dcc74acd495c1c960efcd9c1587e9`。
- node64 三次 0 样本启动失败审计：其 Level3 step440 目录下
  `node64_startup_failures.20260729.txt`，SHA256
  `8c0765d097895dcfc34aa94f5ba7d125f3ae6156a69b45f05a755044b577445b`。
- step440 Level3 正式根目录：node53
  `/mnt/md1/csl_v4r21_fp4_pp1cp2_12k_dppo_predictive_resume40_20260722/h20_eval_20260729/experiments/Eval.KernelBenchL3.DeepSeekV4FlashLoRA.12k.turn1.n8.h20/step440/`。
  dump/summary/formal log/host log SHA256 依次为
  `dc36ff364ab534c409e4a7f8eae95cb8897b881601201648f30c476807c3ee49`、
  `c05583f200083ca208f7e01772a5f43793bfdeb4055cb19f24aa0cc656e7ba87`、
  `c8c07fe597fc70a066b280fcb98e51d87cb4fe67257712290c794eb3c8c1880c`、
  `1579b9ce773458734a5b78bbd7a38fc0373094d7f2625006d4b0b82166e600af`。
- step380 Level3 正式根目录：同一 node53 实验根下的 `step380/`。dump 和 summary
  SHA256 分别为
  `6da76e195eb139ab4ce11a76946631d50f747d7113b7e3bfc5133c6198a15fd1` 和
  `37e4b6b2e0793eb9559084546ea556fa1470a0af691cb90bd913d4a89c374c12`；runner
  PASS 后 disposable 容器已删除，Ray 端口和八张 H20 均已释放。

## 边界与偏差

- Level3 的实际数据目录是 `Data/kernelbench-level3-validation-tvm-v2`，不是最初消息
  中的 v3；该 v2 与 pinned official revision `423217d9...` 逐行匹配。
- Level2 checker 绕过分类是 AST 静态保守下界；Level3 的诊断来自逐个检查正式
  Correct 候选的 forward 调用路径。两者都未替代正式 evaluator，也没有事后重写分数。
- 除四道重点题外，没有对 Level3 所有 50 个 reference 做完整标准架构语义证明；
  但所有题都完成 AST、构造审计，48/50 通过 meta forward，另两题只是审计方法不
  支持初始化中的 `.item()`。
- 本报告新增的 Level2/3 运行均在 H20 上完成，未使用 H200；引用的 Level1
  step620 正式产物来自此前的 H200 评测与人工审计。
- step440 Level3 比较报告的第一次自动复跑使用直接覆盖，加载三个大 dump 的短暂窗口
  把目标暴露为 0 字节；复核时立即发现。复现器已改为内存渲染和原子替换，独立副本与
  当前报告逐字节一致；旧报告 SHA 已废弃。
