# DeepSeek-V4-Flash r21 rsLoRA 的 KernelBench-L1 曲线

更新时间：2026-07-29

## 摘要

r21 predictive-DPPO 的 H20 KernelBench-L1 单轮曲线已完成 step0–460，每 20 step
一个点，每点固定为 100 题 × 8 样本。24 个点均有 800 条结果且
`missing env_result=0`。后半程的 compile/correct 从 step160 的
79.38%/64.38% 上升到 step440 的 97.25%/91.62%；step460 为
97.50%/89.62%，correct 比峰值低 2.00 pct-pt，但单点随机采样不足以把回落
归因于训练更新。恢复训练已产出 step480–560 所需 checkpoint 和 adapter；训练
随后又被续跑；截至 2026-07-29 05:18 JST，checkpoint 已推进到
`iter_0000699`。H200 已完成同 checkpoint 的 step460 随机复测和 step480/500
续测以及 step520/540/560/580/600/620/640/660/680/700。用户已选择把曲线扩展到
后续所有新增的每 20-step 点，并在 step700 后把评测平台改回 H20。step720
checkpoint、adapter 和 H20 disposable runner 已通过门禁；H20 `low/32/32`
正式运行已完成 800/800，但其中 1 条正式 precheck 通过后发生基础设施断连，
用户选择保留原始点并记录最多 0.125 pct-pt 的指标不确定宽度。

最终两点经过独立计数、真实样本抽查和权重重构复核。step460 的 766 个 serving
adapter 张量与 `iter_0000459` 逐 tensor `torch.equal`；Ray job 和外层脚本均
成功退出。H200 先重测了同一 step460：运行与产物通过复核，但 800 个生成
response 与 H20 版本全部不同，因此只能作为同 checkpoint 的独立随机复测，
不能作为 matched-output 的硬件评分对照。step480 的 766 个 serving adapter
张量也与 `iter_0000479` 逐 tensor `torch.equal`，有效运行和产物通过独立复核。
H200 后续每个点使用带 `--init` 的新容器并在退出时删除；已完成两点均没有新增
zombie，主机现存 zombie 均来自更早的非 init 探针容器。

训练暂停后，step660 的 KernelGym 设置先从 `low/4/4` 调整为 `low/16/16`，与
当前 16 个在线 GPU worker 对齐；priority 仍保持 low，避免改变调度语义。当时
正在运行的 step640 保留启动时的 `low/4/4`，没有为并发调整丢弃已有结果。
用户随后要求下一点评测改为 32 并发，因此 disposable runner 从 step680 起默认
`low/32/32`，即每个 GPU worker 约 2 个在途任务；正在运行的 step660 保留
`low/16/16`。若训练恢复，必须通过三个 `KERNEL_EVAL_*` 环境变量显式恢复
`low/4/4`。

## 评测合同与 checkpoint 映射

- 数据：KernelBench Level 1 validation，100 题，每题 8 样本。
- 口径：`tvm_ffi`、ctx/response=12288、max-turns=1、seed=42、temperature=1、
  top-p=1；四项指标分母均为全部 800 条。
- Compile 为 `compiled=true`；Correct 还要求非 decoy；Fast@x 要求 Correct 且
  `speedup >= x`。
- serving：8×H20，DeepSeek-V4-Flash-DSpark base + 指定 PEFT rsLoRA adapter，
  DSpark gamma=3、DP-attention=8、MoE TP8、CUDA graph 开启，并发上限 128。
- checkpoint 标签采用 `step = iteration + 1`；例如 step440/460 分别来自
  `iter_0000439/459`。不要比较相同文件名或导出时间来判断跨 step 权重是否相同。
- adapter 为 rank32、alpha32、rsLoRA；step440/460 均重构 766 个张量并与
  serving 文件逐项相等，源节点和 rollout 节点的 SHA256 也一致。

2026-07-29 对 step620 的逐题审计确认题73存在评分语义缺陷：`groups=4` 在
`get_init_inputs()` 中错位绑定为 `output_padding=4`，而参考实现又忽略
`output_padding`，实际计算 groups=1。官方曲线为保持跨点合同不变仍保留该题，
但不应把它当作模型正确性学习信号。step620 排除整题后的敏感性口径为
97.98%/91.29%/26.01%/20.20%；完整归因见
`step440_step620_kernelbench_failure_attribution_20260729.md` 的 Level1 逐题审计节。

## 主结果

所有指标单位为 %；同一行的四个指标来自同一个 checkpoint。

| step | compile | correct | fast@1.0 | fast@1.2 |
|---:|---:|---:|---:|---:|
| 0 | 41.88 | 27.88 | 7.75 | 5.00 |
| 20 | 61.88 | 43.62 | 13.88 | 8.62 |
| 40 | 49.62 | 41.75 | 14.37 | 10.25 |
| 60 | 76.25 | 64.88 | 14.62 | 10.38 |
| 80 | 81.25 | 67.00 | 15.75 | 11.38 |
| 100 | 87.50 | 69.12 | 14.88 | 10.38 |
| 120 | 78.75 | 68.25 | 17.50 | 11.12 |
| 140 | 79.50 | 66.62 | 17.75 | 12.38 |
| 160 | 79.38 | 64.38 | 14.37 | 10.62 |
| 180 | 85.38 | 73.25 | 16.25 | 12.00 |
| 200 | 85.00 | 73.00 | 15.50 | 11.25 |
| 220 | 87.25 | 75.62 | 16.25 | 11.25 |
| 240 | 88.88 | 78.25 | 16.38 | 12.25 |
| 260 | 88.75 | 78.00 | 15.38 | 11.12 |
| 280 | 91.75 | 79.62 | 16.50 | 12.12 |
| 300 | 91.75 | 80.38 | 18.88 | 14.50 |
| 320 | 95.00 | 86.38 | 18.50 | 14.12 |
| 340 | 97.00 | 88.50 | 19.50 | 15.25 |
| 360 | 96.75 | 88.12 | 18.25 | 15.25 |
| 380 | 96.00 | 88.12 | 20.25 | 17.12 |
| 400 | **97.50** | 89.62 | **21.00** | 16.12 |
| 420 | 96.12 | 89.62 | 19.50 | 15.75 |
| 440 | 97.25 | **91.62** | **21.00** | **17.38** |
| 460 | **97.50** | 89.62 | 19.88 | 16.38 |

step160 之后的主要观察是 compile 和 correct 同向上升，而 fast 指标只缓慢
提高且波动更大。step440→460 的 correct/fast 同时下降，compile 则提高
0.25 pct-pt；这排除了“所有指标都因评测失效而下跌”的简单解释，但不能区分
采样噪声与 checkpoint 更新效果。曲线是固定题集上的随机生成测量，未做跨点
多重比较校正。

早期 step100→160 的 Correct 从 69.12% 降到 64.38%；题目簇 bootstrap 差为
-4.75 pct-pt，95% CI [-9.25, -0.38]。但 step100 是事后曲线峰值，各点截断率又在
1.75%–12.88% 间变化，因此这里只登记“早期局部回落”，不把它直接写成训练更新导致
的能力退化；后续 Correct 也重新上升。

## 最终点验证

step460 的有效运行是 `20260727.012810`，Ray job
`raysubmit_PDqJ6x2zWfYLqQVS` succeeded，外层 `curve_exit_status=0`。
独立复核得到：compile 780、correct 717、fast@1.0 159、fast@1.2 131、decoy
0；index 恰为 0–799，100 题各 8 条，全部 turn_idx=0。sample 状态为 796
completed、4 truncated；env 状态为 780 completed、19 failed、1 timeout。

代表样本抽查覆盖了四类端点：idx0 编译且正确；idx200 的 GELU 是真实 NVCC
编译失败；idx9 编译成功后发生 CUDA illegal memory access；idx690 的 1×1
Conv2d 被 KernelGym 在 300 秒判定 timeout。timeout 和编译失败都保留了完整
`env_result`，因此不是缺失样本。

step460 来自 `out/iter_0000459`。rank32、alpha32、rsLoRA scale
5.65685424949238 的 766 个重构张量与 serving adapter 的 key、shape、dtype、
值逐项相等。node64/node53 的 SHA256 为：config
`f340d4d0e714efcfa93929a5480d2fae540b79d3424cd6d26196d195aa4c3275`，model
`24e5e268636371b41a774c24bc12ceb0412a87fce308ba467f1f68b57e998473`。DP0–DP7
均加载 adapter UUID `cc749dfd6bf35f4887ef26125b5eb081`，有效目录没有旧
step dump 混入。

## H200 桥接边界与续测

H200 上的结果使用相同题集和评测合同，但每个点仍是独立随机生成；下表不能消除
跨 checkpoint 的采样方差。

| step | compile | correct | fast@1.0 | fast@1.2 |
|---:|---:|---:|---:|---:|
| 460 | 97.50 | 92.00 | 21.12 | 17.50 |
| 480 | 97.62 | 90.88 | 21.88 | 17.75 |
| 500 | 98.38 | 90.75 | 21.38 | 17.38 |
| 520 | 98.12 | 90.88 | 21.00 | 16.88 |
| 540 | 98.00 | 91.00 | 21.00 | 18.38 |
| 560 | 96.38 | 88.75 | 23.25 | 19.25 |
| 580 | 97.88 | 89.75 | 23.62 | 19.12 |
| 600 | 97.38 | 88.50 | 22.38 | 18.00 |
| 620 | 97.75 | 90.38 | 25.75 | 20.00 |
| 640 | 98.00 | 89.38 | 23.50 | 18.75 |
| 660 | 98.25 | 88.38 | 23.75 | 18.25 |
| 680 | 97.75 | 88.75 | 23.38 | 18.50 |
| 700 | 98.75 | 90.38 | 22.00 | 18.88 |

H200 独立复测使用相同 step460 adapter、相同 100 题和相同评测合同。独立重算
为 compile 780、correct 736、fast@1.0 169、fast@1.2 140、decoy 0；800 条
index/group_id 均为 0–799，100 题各 8 条，全部 turn_idx=0，且
`missing env_result=0`。env 状态为 780 completed、19 failed、1 timeout。

这次复测不能回答“同一批输出在 H20 与 H200 serving 上是否得到相同结果”。两端
prompt 按 index 800/800 相同，adapter model/config SHA256 也分别相同，但 response
hash 和 response length 都是 0/800 相同；全局以及每题 8 样本内均无 response
重合。相对 H20 的 correct +19、fast@1.0 +10、fast@1.2 +9 混入了完整的随机
生成差异。若要做受控平台对照，应固定同一批 response 后分别重评，而不是比较
这两个随机 rollout dump。

H200 有效运行日志为 `step460/20260727.152519.log`，Ray job
`raysubmit_KvEaMubGGMZkb7bw` succeeded，外层和容器退出状态均为 0；dump SHA256
为 `1a945a4b724224f7ad5064aadc7003d23c5fbe8317b85bf1e58d414015c508bf`。
代表样本覆盖正确快核 idx192（2.23×）、正确慢核 idx0（0.31×）、编译成功但
数值错误 idx31、生成代码遮蔽 CUDA `blockDim` 导致 NVCC 失败的 idx191，以及
KernelGym 300 秒超时的 idx499。所有类别均保留完整 `env_result`。

step480 独立重算为 compile 781、correct 727、fast@1.0 175、fast@1.2 142、
decoy 0；800 条结果齐全且 `missing env_result=0`。sample 状态为 796 completed、
4 truncated；env 状态为 781 completed、17 failed、2 timeout。有效运行日志为
`step480/20260728.000302.log`，Ray job `raysubmit_DTyMWtXXQzt7kCN1` succeeded，
外层和容器退出状态均为 0。dump SHA256 为
`999cbcf3aa56323415e6d44fde694919ac2a6ee1d990da296224f560aa036750`，summary
SHA256 为 `254ef0ac3a48b6a70f98c73bedb7318ec798685bd5f6e68c0e16a8fbbef4f582`。

step480 来自 `iter_0000479`；重构的 766 个张量与 H200 serving adapter 逐项
相等，H200 和源节点的 model SHA256 均为
`b626b5e751b9c94e29500a0df9661fccc36ba658cc6eea3718ba1ba96a6ad075`。
代表样本覆盖正确快核 idx192（2.24×）、正确慢核 idx0（0.34×）、编译成功但
CUDA illegal memory access 的 idx21、CUDA section marker 损坏导致 NVCC 失败的
idx48，以及 KernelGym 300 秒超时的 idx768。此前一次启动在生成样本前因 SSH
反向隧道中断而退出，没有留下 dump；有效运行的时间戳、Ray job 和产物均独立，
未混入该失败尝试。

step500 独立于 summary 的主复核重算为 compile 787、correct 726、fast@1.0
171、fast@1.2 139、decoy 0；env 状态为 787 completed、8 failed、5 timeout，
sample 状态为 800 completed。800 条结果无缺失，index 为 0–799，100 题各 8 条，
turn_idx 均为 0；无空 response、空解析代码、correct-but-not-compiled、
`remove_sample` 或与 reference code 完全相同的生成代码。代表样本覆盖最快正确核
idx793（7.07×）、正确但慢的 idx610（0.00455×），以及候选 `ModelNew` 构造函数
签名错误的 failed idx328。最后 5 个 timeout 均保留规范化 env_result 并计入全量
分母。有效日志为 `step500/20260728.120927.log`，Ray job
`raysubmit_kqnSRhGRZ7WWyp49`、外层和容器退出状态均为 0；新 `--init` 容器
`csl_v4_eval_step500_20260728.200926` 已删除，主机 zombie 仍为历史基线 72。
dump、summary 和日志 SHA256 分别为
`c3be243cd2152673a81f02fe896255f962da049ac59f8342b55684be0f6f5fe3`、
`b9107351b344efb9084f3a077aa194af391f2e002ad552f1beb2cbf6149c2ee0` 和
`e093839bba9860d8cb97ebd8f6a920345a22a1642a69fe358e05a7b42da4aee8`。
独立 reviewer 另行从 dump 重算、用正式 parser 检查全部 800 条代码并抽查真实样本，
同时复核 Ray/runner、容器时间线和 zombie 基线，结论为 PASS；完成通知已通过
`page_user` 发送。

step520 独立重算为 compile 785、correct 727、fast@1.0 168、fast@1.2 135、
decoy 0；env 状态为 785 completed、13 failed、2 timeout，sample 状态为 800
completed。800 条结果无缺失，index/group_id 均为 0–799，100 题各 8 条，
turn_idx 均为 0；无空 response、correct-but-not-compiled 或 `remove_sample`。
唯一 parser caveat 是 idx579 只生成推理而没有规定的 CUDA/binding/ModelNew 三段
代码，被本地 precheck 正确判为 validation failure 并计入全量分母。其余非完成态
均可归因于候选代码的编译/构造签名错误或 2 个明确的 300 秒任务 timeout，不是
运行级错误。代表样本覆盖最快正确核 idx699 MinGPTNewGelu（8.12×）、正确但慢的
idx493 Conv2d（0.00415×）、遮蔽 CUDA `blockDim` 内建而编译失败的 idx307、
precheck 失败的 idx579，以及 SDPA 300 秒 timeout 的 idx769。有效日志为
`step520/20260728.134342.log`，Ray job `raysubmit_ARpTZ6x6yurneYSs`、外层和容器
退出状态均为 0；新 `--init` 容器 `csl_v4_eval_step520_20260728.214341` 已删除，
主机 zombie 仍为历史基线 72。dump、summary 和日志 SHA256 分别为
`c574d9313298eb9c6a24c53f74f1007c71b9f4696c1574736de72d657c21d66c`、
`085266e7663e7b009f0fd7c00e60591ac976500abaefe88cc8a6a5c5550bef00` 和
`a5c2c1d45262cfda1a85ae9d6d82ee48b9ea15856db6dbb99dc7ffa74c8861e9`。
独立 reviewer 复算全部指标、正式 parser、不变量、实样、Ray/runner、容器时间线和
zombie 基线后结论为 PASS；完成通知已通过 `page_user` 发送。队列随后用另一新
`--init` 容器 `csl_v4_eval_step540_20260728.231904` 启动 step540；实际评测进程
环境仍为 KernelGym low/4/4。

step540 独立重算为 compile 784、correct 728、fast@1.0 168、fast@1.2 147、
decoy 0；env 状态为 784 completed、13 failed、3 timeout，sample 状态为 799
completed、1 truncated。800 条结果无缺失，index/group_id 均为 0–799，100 题
各 8 条，turn_idx 均为 0；无空 response、correct-but-not-compiled 或
`remove_sample`。正式 parser 检出 799 条完整三段代码；唯一 caveat 是 idx122
response 截断且缺 ModelNew，被本地 validation 正确拒绝并计入全量分母。另两条
precheck failure、10 条候选代码编译/接口失败和 3 个明确的 300 秒 timeout 都是
样本级结果，不是运行级错误。代表样本覆盖最快正确核 idx702 MinGPTNewGelu
（8.32×）、正确但慢的 idx688 pointwise Conv2d（0.00606×）、构造签名错误的
idx340、截断 precheck 的 idx122，以及 4D tensor matmul 300 秒 timeout 的 idx83。
有效日志为 `step540/20260728.151906.log`，Ray job
`raysubmit_AYu7m7uZaEgFANtm`、外层和容器退出状态均为 0；新 `--init` 容器
`csl_v4_eval_step540_20260728.231904` 已删除，主机 zombie 仍为历史基线 72。
dump、summary 和日志 SHA256 分别为
`6a81ff061343f38917d28893cdc7291d0b7d9427c85f1bee508422574d1828cc`、
`7787164780e1e4050d7289e2e50cb1bb924c3f8ab0aee8d7f19262954bbea200` 和
`c1801f8af5f0622aa54048eb470f9d718fa2fbabe0cc96e9f0d524996dee2351`。
独立 reviewer 复核原始 dump、正式 parser、真实样本、运行链、容器时间线和 zombie
基线后结论为 PASS；完成通知已通过 `page_user` 发送。队列随后用另一新 `--init`
容器 `csl_v4_eval_step560_20260729.010152` 启动 step560；实际评测进程环境仍为
KernelGym low/4/4。

step560 独立重算为 compile 771、correct 710、fast@1.0 186、fast@1.2 154、
decoy 0；env 状态为 771 completed、25 failed、4 timeout，sample 状态为 784
completed、16 truncated。800 条结果无缺失，index/group_id 均为 0–799，100 题
各 8 条，turn_idx 均为 0；无空 response、correct-but-not-compiled 或
`remove_sample`。step540 与 step560 的 800 条 problem、prompt 和 label 逐 index
字节级相同，step560 adapter model/config SHA256 也分别与预期
`a99a638fb9b3a43202fa1945c88e21d52163504161b66e9465459ba8e71a3617` 和
`f340d4d0e714efcfa93929a5480d2fae540b79d3424cd6d26196d195aa4c3275`
完全一致，排除了数据、adapter 和索引错配。

step540→560 的 compile/correct 回落是响应级真实退化，不是测量 bug：17 条
precheck failure 中，16 条恰好达到总 token 12286、无 EOS，在 context 12288
附近截断并未输出完整 ModelNew；唯一 completed 的 idx513 则错误嵌套 section
code fence，正式 parser 无法提取三段代码。17 条 parser 结构异常与 17 条
validation failure 精确重合。compile 784→771 的 -13 可分解为 precheck 多 14、
timeout 多 1、普通编译失败少 2；correct 728→710 又叠加 compiled-but-wrong 多 5。
fast 指标同期反而提高，进一步排除了全局打分失效。是否为稳定模型退化还是
temperature=1 单点抽样波动，需要复测或后续点判断。

代表样本覆盖最快正确核 idx316 LayerNorm（16.50×）、正确但慢的 idx755
CrossEntropyLoss（0.00436×）、NVCC 明确报 shared `Bs` 重复声明的 idx21、
自纠推演中截断的 idx136、嵌套 fence 错误的 idx513，以及明确 300 秒 timeout 的
idx552。有效日志为 `step560/20260728.170154.log`，Ray job
`raysubmit_qu5nAkVFaWQMKiyq`、外层和容器退出状态均为 0；新 `--init` 容器
`csl_v4_eval_step560_20260729.010152` 已删除，主机 zombie 仍为历史基线 72。
dump、summary 和日志 SHA256 分别为
`7dd0533997f9d6ce5ba51f275c1e3c97512cce062d33bd65840b656fe9ee559c`、
`686b6554cb86cc0cc816bb74be3a0b9dc0cf8f00c722fcfd908664ca02880763` 和
`87e7e5be8a94353a2b231c8874b7e177755119a469e482dc31922f21c991de44`。
独立 reviewer 逐条闭合异常分解、正式 parser、adapter 身份、实样和运行链后结论
为 PASS；完成通知已通过 `page_user` 发送。队列随后用另一新 `--init` 容器
`csl_v4_eval_step580_20260729.025344` 启动 step580；实际评测进程环境仍为
KernelGym low/4/4。

step580 独立重算为 compile 783、correct 718、fast@1.0 189、fast@1.2 153、
decoy 0；env 状态为 783 completed、14 failed、3 timeout，sample 状态为 794
completed、6 truncated。800 条结果无缺失，index/group_id 均为 0–799，100 题
各 8 条，turn_idx 均为 0；无空 response、correct-but-not-compiled 或
`remove_sample`。6 条截断响应均达到总 token 12286、无 EOS 且缺少完整三段代码，
被正式 parser 正确拒绝；idx615 虽为 completed 且三段完整，但先给出
`forward: pass` 再覆盖类定义，触发静态 fallback/bypass 防护。3 条 timeout 分别为
idx694、768、770 的 300 秒任务级超时；其余 7 条普通编译失败和 65 条
compiled-but-wrong 均有完整 env 证据，不是运行级错误。

代表样本覆盖最快正确核 idx792 HingeLoss（7.13×）、正确但慢的 idx531 Conv1d
（0.00253×）、NVCC 报 `float2` 未声明的 idx262、截断的 idx455、静态 bypass 的
idx615 和 300 秒 timeout 的 idx694。adapter model/config SHA256 与预期
`47604297bdc69587327eddf30c79889436b09e8339a3e3ddc0bcd43984497c7b` 和
`f340d4d0e714efcfa93929a5480d2fae540b79d3424cd6d26196d195aa4c3275`
完全一致，DP0–DP7 加载同一 adapter UUID。有效日志为
`step580/20260728.185346.log`，Ray job `raysubmit_hDz2x4tsj6V4gXSj`、外层和容器
退出状态均为 0；新 `--init` 容器 `csl_v4_eval_step580_20260729.025344` 已删除，
主机 zombie 仍为历史基线 72。dump、summary 和日志 SHA256 分别为
`f7f0513f65441560165a6c210fa2cadcf94fd7ff4269f7723bb181c278b4426f`、
`006e706785da803d775a0c276119c0b7efedbdc9669c45bf4e7714b02f5ea700` 和
`26f245ee97cdf769e622779d8a5be68cfa916a223f0414fd5db71c43283f7db6`。
独立 reviewer 复核原始 dump、正式 parser、异常分类、adapter 身份、实样、运行链
和容器生命周期后结论为 PASS；完成通知已通过 `page_user` 发送。队列随后用另一
新 `--init` 容器 `csl_v4_eval_step600_20260729.043708` 启动 step600；实际评测
进程环境仍为 KernelGym low/4/4。

step600 独立重算为 compile 779、correct 708、fast@1.0 179、fast@1.2 144、
decoy 0；env 状态为 779 completed、17 failed、4 timeout，sample 状态为 796
completed、4 truncated。800 条结果无缺失，index/group_id 均精确为 0–799，
100 题各 8 条，prompt/label/problem_id 与 parquet 逐条匹配；turn_idx 全 0、
response 非空且 800 条唯一、token/loss-mask 对齐、无 correct-but-not-compiled 或
`remove_sample`。completed 的 779 条精确分解为 correct 708 与 wrong 71；17 条
failed 精确分解为 11 条普通编译/接口失败和 6 条 precheck failure，另有 4 条明确
的 300 秒单任务 timeout（idx291、771、772、774）。

正式 parser 本地复核得到 5 条结构失败：截断的 idx552/569 只有 CUDA、idx573
只有 CUDA+binding；completed 的 idx688 缺 CUDA，idx734 只有 CUDA 且自称 partial
stub。4 条 truncated 都达到总 token 12286 且无 EOS；idx610 与前三条不同，它先
给出一整组 CUDA+binding+ModelNew，再开始第二组修订并被截断，正式 parser 按既定
规则选取最后一组完整代码，环境确实编译且正确，因此该结果有效。环境额外拒绝
idx437：静态规则 `\bpass\b` 命中了合法 CUDA 循环变量
`for (int pass=0; ...)`，人工闭因确认是 static bypass 误报而非 fallback/作弊；
仍严格按正式评测口径计失败，没有事后改分。

代表样本覆盖最快正确核 idx703 MinGPTNewGelu（8.318×）、最慢正确的 idx588
dilated ConvTranspose1D（0.001498×）、因自定义 `float4` 冲突而被 NVCC 明确拒绝的
idx149、结构失败 idx552/688/734、静态误报 idx437，以及 GPU5 上 300 秒 timeout
的 idx291。adapter model/config SHA256 与预期
`7a4f994318ba6c25655c4a957bd4847dd7e3661593ea0e78fba0bf7223e7938f` 和
`f340d4d0e714efcfa93929a5480d2fae540b79d3424cd6d26196d195aa4c3275`
完全一致；DP0–DP7 各且仅加载一次 LoRA UUID
`972c9abf53855727a8f82ee343dc6cda`。

运行中曾出现约 40 分钟进度条不前、H200 利用率为 0 且 `/health_generate` 超时；
事后从完整日志闭因后，不能把它称为 SGLang 死亡：BatchHeartbeat 持续追踪不同
唯一 KernelGym task，最后一批 total request 约 1196–1226 秒、其中 env 等待约
1041–1114 秒，随后 697→798→799→800 批量释放并逐条出现 HTTP/status completed。
运行只有一个 Ray job `raysubmit_uTjGR15XQjiipVQu`、一次 dump save，Ray、外层脚本
和容器均 exit 0，没有重启、重复落盘或 fatal 日志。有效日志为
`step600/20260728.203710.log`；dump、summary、日志 SHA256 分别为
`9f1f70c1c450985683abaa100535c04c5a51cefd61097edb211d32db3f60cd5e`、
`63b792f67c0e89c44692d215bece84fad6f6ff57173394d44ac858e5af4135df` 和
`ef0a42b901e27560b997f91a004dd56a58d2ab34e95c57c18859f5f0aedabc3f`。
独立 reviewer 复算产物、数据、parser、异常、实样和生命周期后结论为 PASS；完成
通知已通过 `page_user` 发送。旧 `--init` 容器已删除且 zombie 仍为历史基线 72，
队列随后以全新 `--init` 容器 `csl_v4_eval_step620_20260729.061919` 启动 step620，
实际环境继续为 KernelGym low/4/4。

step620 独立重算为 compile 782、correct 723、fast@1.0 206、fast@1.2 160、
decoy 0；env 状态为 782 completed、15 failed、3 timeout，sample 状态为 790
completed、10 truncated。800 条结果无缺失，index/group_id 精确为 0–799，100 题
各 8 条，且 prompt/label/problem_id 与 parquet 逐条匹配；response 非空且 800 条
唯一，token 对齐，无 correct-but-not-compiled 或 `remove_sample`。completed 782 条
精确分解为 correct 723 与 wrong 59；failed 15 条精确分解为 9 条 validation 和
6 条普通失败，另有 3 条明确的 300 秒单任务 timeout（idx185、769、774）。

9 条 validation failure（idx504/572/573/577/578/594/612/613/617）均达到总 token
12286、无 EOS，正式 parser 无法取得完整 ModelNew，环境与本地 precheck 逐条一致。
第 10 条截断 idx351 先给出完整 CUDA+binding+ModelNew，随后开始第二组 CUDA 修订并
被截断；正式 parser 按既定规则选前一完整组，环境 precheck/compile/correct 均通过，
因此结果有效。6 条普通失败均有真实候选错误：3 条 ModelNew 构造参数签名不匹配，
idx554 使用非法 `trap` 汇编，idx671 把 `const int64_t*` 赋给可写指针，idx775 的
CUDA 索引缺 `]`。

代表样本覆盖最快正确核 idx696 MinGPTNewGelu（8.2196×）、最慢正确的 idx615
ConvTranspose3D（0.008664×）、构造签名错误 idx348、结构截断 idx504/572、编译
错误 idx554/671/775，以及 GPU5 上 300 秒 timeout 的 idx185。adapter model/config
SHA256 与预期
`7142cc57e4d79513dda2738bec0d63d6679d23a82e47e58e23e5381107edab62` 和
`f340d4d0e714efcfa93929a5480d2fae540b79d3424cd6d26196d195aa4c3275`
完全一致；DP0–DP7 各且仅加载一次 LoRA UUID
`1876366cd0765a57b91199a5ef2b34de`。

长尾期间 BatchHeartbeat 持续推进唯一 task、tokens_in_use 保持 3/4；进度从 683
批量释放到 777，再到 798 和 800。最后一个 task 的累计等待 767.7 秒包含队列和
状态等待，服务最终明确返回 HTTP 200/STATUS timeout；3 个真实 300 秒 timeout 均
显式进入 dump。运行只有一个 Ray job `raysubmit_gGedkjzLFJepRVTL`、一次 dump save，
Ray、外层脚本和容器均 exit 0，无重启、重复落盘或 fatal 日志。有效日志为
`step620/20260728.221921.log`；dump、summary、日志 SHA256 分别为
`4a867fc66ed688faa8ff48d3568c0f61af91e5ae8555959d051473148a5ea39d`、
`1da0e05de6fe4b05f57cc013a14f400ac5f09dea7afa96329bb65693311e5fca` 和
`8644072ad7328253f7f9a2531ec5ae7da70b01a476077b14f97ff4184d7e6882`。
独立 reviewer 复算全部证据后结论为 PASS；完成通知已通过 `page_user` 发送。旧
`--init` 容器已删除且 zombie 仍为历史基线 72，队列随后用全新 `--init` 容器
`csl_v4_eval_step640_20260729.075432` 启动 step640，实际环境继续为 KernelGym
low/4/4。

step640 独立重算为 compile 784、raw correct 716、正式 correct 715、fast@1.0
188、fast@1.2 150；正式 correct 排除了唯一 decoy idx96，因此表中分别为
98.00%/89.38%/23.50%/18.75%。env 状态为 784 completed、13 failed、3 timeout，
sample 状态为 800 completed、无截断。800 条结果无缺失，index/group_id 精确为
0–799，100 题各 8 条，prompt/label/problem_id 与 parquet 逐条匹配；turn_idx 全
0、response 非空且 800 条唯一、token/loss-mask 长度关系正确、无
correct-but-not-compiled 或 `remove_sample`。completed 784 条精确分解为正式
correct 715、compiled-wrong 68 和 decoy 1；failed 13 条精确分解为 4 条 formal
precheck failure 与 9 条普通失败，另有 3 条 300 秒 timeout（idx768/772/773）。

formal parser 对 800 条均取得完整 CUDA+binding+ModelNew；本地 precheck 与环境
逐条一致，仅 idx260/278/625 未调用 `tvm_ffi_extension`，idx627 的 ModelNew 有
Python 缩进语法错误。9 条普通失败为真实候选错误：idx341/536/572/603 的
ModelNew 构造签名不匹配，idx265 重复定义 launcher，idx473 缺分号，idx503 把
整数当指针，idx742 使用非常量 shared 数组维度，idx756 拼错 launcher 名称。
唯一 decoy idx96 的对称矩阵乘法由 cuBLAS `ampere_sgemm` 完成，而声明的
`dummy_kernel` 从未执行，custom kernel 数量/时间覆盖均为 0；因此 KernelGym
按正式口径将它计入 compile、排除出 correct 是一致且可复现的，不是计数错误。

代表样本覆盖最快正式正确核 idx700 MinGPTNewGelu（8.2440×）、最慢正式正确核
idx40 large-K matmul（0.002494×）、decoy idx96、precheck idx260/627、普通编译
失败 idx265 和 timeout idx768。adapter model/config SHA256 与预期
`c415f59d0060a87005de827b8fa31dc42e04128b8d8190bae248ba932e664e9b` 和
`f340d4d0e714efcfa93929a5480d2fae540b79d3424cd6d26196d195aa4c3275`
一致；DP0–DP7 各且仅加载一次 LoRA UUID
`5ae814c5dd6351e49d77f05d1a86b8ac`。

运行只有一个 Ray job `raysubmit_41LCtHWCv85XWqRy`、一次 dump save；Ray、外层
脚本和容器均 exit 0，无 fatal、重启或重复落盘。有效日志为
`step640/20260728.235433.log`；dump、summary、日志 SHA256 分别为
`47b31ef470353b26cef8154038a67f0ecce45f380b94f15e5eb792887fc9274c`、
`8d0adad110fcdd079f6bae0c86eabe44effd99c41a536744790d941b89c7bc56` 和
`fd9106f3fc0621254c647828b8a6e66677ecd27ec70b95130d8bd24392a819bf`。
独立 reviewer 对全量产物、parser、异常、实样、adapter 和生命周期的结论为 PASS，
完成通知已通过 `page_user` 发送。step640 `--init` 容器已删除，zombie 保持历史
基线 72；队列随后用全新
`--init` 容器 `csl_v4_eval_step660_20260729.091334` 启动 step660，Ray runtime
环境已核验为 KernelGym `low/16/16`。

step660 的正式 dump 口径重算为 compile 786、correct 707、fast@1.0 190、
fast@1.2 146、decoy 0，即 98.25%/88.38%/23.75%/18.25%。env 状态为 786
completed、10 failed、4 timeout；sample 状态为 800 completed、无截断。800 条
结果的 index/group_id 精确为 0–799，100 题各 8 条，prompt/label/problem_id 与
parquet 逐条匹配；response 非空且唯一、token/loss-mask 关系正确、无
correct-but-not-compiled 或 `remove_sample`。completed 786 条精确分解为 correct
707 与 compiled-wrong 79；failed 10 条包含 1 条 precheck、8 条真实候选失败和
1 条基础设施瞬断，另有 4 条明确的 300 秒 timeout（idx256/291/768/773）。

formal parser 对 800 条均取得完整 CUDA+binding+ModelNew，本地与环境 precheck
仅 idx485 同时失败：ModelNew 最后调用框架 ConvTranspose3d 而没有调用
`tvm_ffi_extension`。8 条真实普通失败为 idx223 的 float4 下标误用、idx257 缺
`cudaMalloc` 声明、idx259 候选 so 缺 `batchnorm_launcher` 符号、idx316 参数
不足、idx479 未定义 kW/kH/kD、idx515 非法类型/offset、idx685 未定义 tx/ty，
以及 idx786 的 ModelNew 构造签名错误。代表样本覆盖最快正确的 idx701
MinGPTNewGelu（8.2687×）、最慢正确的 idx439 Conv2d（0.008529×）、precheck
idx485、普通失败 idx223、timeout idx256，以及下述瞬断 idx184。

idx184 的 LogSoftmax response 完整、有 EOS、formal precheck 通过，且候选为真实
自定义 `log_softmax_kernel`。它在提交评测后 0.047 秒返回
`Server disconnected without sending a response.`，没有 task_id；日志中对应
`parallel_task_000188_02b64587` 是 799 个 POST 中唯一没有 HTTP response/STATUS
的请求，KernelGym 查询也显示 task 不存在。代码闭因是客户端只重试 timeout 和
connect error，而该 `httpx.RemoteProtocolError` 落入广义异常后被立即包装成
`COMPILATION_ERROR`；候选实际从未编译或执行。因此独立 reviewer 对其余完整性
给 PASS、对 clean formal point 给 CONDITIONAL。用户在 unattended-user-decide
选择 A：保留可复现的正式 dump 原始口径，不整点重跑、不改分，并明确记录最多
0.125 pct-pt 的保守偏差。

adapter model/config SHA256 分别为
`9a27918eacbbeffb2c3eeb4c8e108633b0f2e194511028e6da7a76e19a209db6` 和
`f340d4d0e714efcfa93929a5480d2fae540b79d3424cd6d26196d195aa4c3275`；
DP0–DP7 各且仅加载一次 LoRA UUID `1886bf4d16e4531996c36cfcbb3f025d`。运行只有
一个 Ray job `raysubmit_dynwbYR93N3RKkiF`、一次 dump save；Ray、外层脚本和容器
均 exit 0。有效日志为 `step660/20260729.011336.log`；dump、summary、日志 SHA256
分别为 `e048830f565f442300cfc8eb8e12c00da294f714f5da70d78c15086bec9cab1f`、
`65fea2b40b1cb5deb361f9228a4958ecf602422e4c2e78bdc61e2779e15b3a05` 和
`d0f9e93b689d3c686fd897ddfbce821ae821700b7d9a78a3782a47870952a08e`。
step660 的 800 样本阶段约 26 分钟；接受完成通知已通过 `page_user` 发送。旧容器
已删除，zombie 仍为历史基线 72。队列随后用全新 `--init` 容器
`csl_v4_eval_step680_20260729.094710` 启动 step680，其 Ray runtime 已核验为
KernelGym `low/32/32`。

step680 独立重算为 compile 782、correct 710、fast@1.0 187、fast@1.2 148、
decoy 0，即 97.75%/88.75%/23.38%/18.50%。env 状态为 782 completed、13
failed、5 timeout，sample 状态为 800 completed、无截断。800 条结果的
index/group_id 精确为 0–799，100 题各 8 条，prompt/label/problem_id 与 parquet
逐条匹配；response 非空且唯一、token/loss-mask/logprob 长度关系正确，无
`remove_sample`。completed 782 条精确分解为 correct 710 与 compiled-wrong 72；
failed 13 条精确分解为 4 条 formal precheck failure 与 9 条真实候选失败；5 条
300 秒 timeout 为 idx45/481/768/769/770。

formal parser/precheck 对 idx478/479/575 一致判定缺 CUDA `.cu` section，对
idx755 一致判定 binding.cpp 非法使用 `cudaStream_t`；其余 796 条全部通过。
9 条正式提交失败均由真实候选闭因：idx155/207/645/647/691 为常规编译错误，
idx339/340/574 为 ModelNew 构造签名错误，idx319 的 binding 引用了候选未定义的
`layer_norm_launcher`，导致隔离 worker 子进程 exit127。代表样本覆盖最快正确的
idx702 MinGPTNewGelu（7.3087×）、最慢正确的 idx759 CrossEntropyLoss
（0.004363×）、precheck idx478/755、候选失败 idx319、错误结果 idx43 和 timeout
idx45。

32 并发下 796 个正式 POST start、HTTP response、STATUS 与 dump task_id 四组集合
完全一致，全部 HTTP 200，未重现 step660 的 `RemoteProtocolError`。启动时日志的
三段 Traceback 实为同一个 `HEALTH_CHECK_*` 请求的 CancelledError→TimeoutError→
ValueError 异常链；探针重试随后成功，发生在首个正式请求之前。之后 800 个
SGLang generate 全部 HTTP 200，故该启动探针噪声没有影响正式样本。

adapter model/config SHA256 分别为
`98f1fd4db7df4d2479d424d915731075a67b9e726275242d5f9184d80b16ff59` 和
`f340d4d0e714efcfa93929a5480d2fae540b79d3424cd6d26196d195aa4c3275`；
DP0–DP7 各且仅加载一次 LoRA UUID `342eda044e2a5811b5885c81382f887b`。单一 Ray
job `raysubmit_aMjABF9XLVU8BKrY`、dump、summary、外层脚本和容器退出链全部成功。
有效日志为 `step680/20260729.014712.log`；dump、summary、日志 SHA256 分别为
`0e9395b6369792e381d42d0cb338bcbcdc0b8ba9efd68ffc602450cc994a0556`、
`2e93ce5b2015046b105b4c90e9b41e26001b07837338b344250adac873ecf1d7` 和
`1e9c5492e43cb8fcafa6fec3691514eef6a4b3cd91a067e08c5aee14af3ec955`。
独立 reviewer 结论为 PASS，完成通知已通过 `page_user` 发送。step680 的 800
样本阶段约 23 分 42 秒；旧容器已删除，zombie 仍为历史基线 72。队列随后用全新 `--init` 容器
`csl_v4_eval_step700_20260729.101822` 启动 step700，实际环境继续为
KernelGym `low/32/32`。

step700 的正式 dump 口径重算为 compile 790、correct 723、fast@1.0 176、
fast@1.2 151、decoy 0，即 98.75%/90.38%/22.00%/18.88%。env 状态为 790
completed、7 failed、3 timeout，sample 状态为 800 completed、无截断。800 条
结果的 index/group_id 精确为 0–799，100 题各 8 条，prompt/label/problem_id 与
parquet 逐条匹配；response 非空且唯一、token/loss-mask/logprob 长度关系正确。
completed 790 条精确分解为 correct 723 与 compiled-wrong 67；7 条 failed 包含
3 条 formal precheck、1 条真实候选编译失败和 3 条基础设施瞬断；3 条 300 秒
timeout 为 idx769/770/771。

正式 parser/precheck 对 idx552 缺 CUDA `.cu`、idx759 缺 ModelNew、idx774 未调用
`tvm_ffi_extension` 的判断与环境逐条一致。唯一真实普通失败 idx715 在 CUDA 中
用运行时 `cols` 初始化 `constexpr`，NVCC 明确拒绝。代表样本覆盖最快正确的
idx313 LayerNorm（14.6871×）、最慢正确的 idx753 CrossEntropyLoss（0.004408×）、
错误结果 idx31、precheck idx552/759/774、候选失败 idx715 和 timeout idx769。

low/32/32 下共有 797 个通过 precheck 的 POST start，其中 794 个取得 HTTP 200、
STATUS 和 dump task_id，四者逐 task 状态一致；缺失的 task181/491/579 分别对应
idx198/486/571。三条 response 都完整且 formal precheck 通过，但在 0.015–0.106
秒内返回 `Server disconnected without sending a response.`、无 task_id；在线
查询 status/results 均为 404。独立复核把近因闭合到 LB/反向 SSH/后端 HTTP 链路
在响应头前 EOF，而客户端遗漏对 `httpx.RemoteProtocolError` 的重试，随后又把
通用 failure 误包装成 `COMPILATION_ERROR`。因此三条候选均未实际评测，独立
reviewer 对其余完整性给 PASS、对 clean formal point 给 CONDITIONAL。用户再次
选择 A：保留正式 dump 原始口径、不重跑、不改分，并记录每项最多 0.375 pct-pt
的不确定宽度。

adapter model/config SHA256 分别为
`caf301482c17454cfbda2226ce73841843ae0839bd21adeaad20b9f8b6d141a4` 和
`f340d4d0e714efcfa93929a5480d2fae540b79d3424cd6d26196d195aa4c3275`；
DP0–DP7 各且仅加载一次 LoRA UUID `a70097446a4c5a778b38b040b034c671`。单一 Ray
job `raysubmit_LmYhF9xDL8sQFjkp`、dump、summary、外层脚本和容器退出链全部成功。
有效日志为 `step700/20260729.021824.log`；dump、summary、日志 SHA256 分别为
`5c0ee9d6ef1f539eca54783f504a46eccfe1df3de2b037d4756cda3a6a16f11e`、
`5c897452d92eea1b3182d6f43f6b61420c101510f7a4911e85ff6af4b922cec2` 和
`3a45b7ac47193b6a181d7755ac10d3c9b3f2dc07a492af28d7df1eae4d66784b`。
step700 的 800 样本阶段约 21 分 34 秒；接受完成通知已通过 `page_user` 发送。旧
容器已删除，zombie 仍为历史基线 72，H200 后继队列最终
`h200_eval_queue=PASS` 且没有容器残留。用户随后要求后续点改回 H20；映射为
step720 的 `iter_0000719` 已存在，后续将继续使用每点一个新 `--init` 容器。

node53 的裸 `sglang:dspark-r1` 镜像不含线上容器中已验证的 DSpark 源码修改和约
0.9 GiB JIT 缓存，直接用裸镜像创建一次性容器会引入运行时漂移。为同时满足环境
一致和 zombie 回收，已在 node53 从空闲 `csl_dspark_r1` 固化只读评测快照
`csl/sglang:dspark-r1-eval-snapshot-20260729`，image ID 为
`sha256:fab44f685ffcc1873d3c8f9b0e13094b930f81c32c1fc72d373616bf70d4be0f`。
正式入口为 `scripts/dsv4/run_eval_step.sh h20 curve <STEP>`：每点从该快照创建新的
`--init`/host-network 容器，检查 8 张 H20 空闲、Ray 端口和 KernelGym 健康，
持有单机锁，并在任何退出路径删除容器。现有 rollout 容器未停止、未修改。评测
代码另行隔离在 node53
`/mnt/md1/chenshuailin/projects/kernel_agents/slime-v4flash-lora-eval-20260729`，
只读 Megatron 源码在同级 `Megatron-LM-eval-20260729`；node64/node53 源码
runtime fingerprint 均为
`5ccb61336db38cc9064cf5198d7b4e258be0de4c79083bf936629bdf23c1b40d`。

H20 正式运行前有三次前置启动失败，均发生在生成开始前、没有正式 sample 或
KernelGym task，且每次容器都已删除。依次为：快照没有 Megatron Python 源码而在
参数解析失败；Ray 2.55 不接受 loopback node IP、自动改选 `10.11.2.153` 后因
地址不一致超时；node53 旧活动 checkout 的 SGLang 参数兼容层落后于 node64，
缺少 `sglang_data_parallel_size` fallback。最终 runner 挂载复算 manifest 一致的
Megatron 源码，固定 `10.11.2.153/bond0`，并从上述隔离 eval checkout 启动；针对
最后一个错误的参数 smoke 先通过 DP=8/TP=8。正式 step720 使用新容器
`csl_v4_h20_eval_step720_20260729.033320` 和 Ray job
`raysubmit_z2KaCuQJpa5vEvzJ`；CUDA graph、8 个 base+LoRA DP rank 和 DSpark
draft 均加载成功后才进入 800 样本生成。node53 zombie 在这些退出后保持历史基线
15，没有新增。

## H20 续测

step720 的原始 dump 口径重算为 compile 783、correct 722、fast@1.0 171、
fast@1.2 147、decoy 0，即 97.88%/90.25%/21.38%/18.38%。800 条样本均为
completed、无截断，`env_result` 无缺失；index/group_id 精确为 0--799，100 题
各 8 条，response 非空且 800 条唯一。env 状态为 783 completed、13 failed、
4 timeout；completed 精确分解为 722 correct 与 61 compiled-wrong。13 条 failed
中 7 条是 formal precheck failure、5 条是真实候选 NVCC 编译失败、1 条是基础设施
瞬断。4 条 timeout 为 idx770/772/773/775，均已实际入队，并在 KernelGym GPU
worker 上达到 300 秒任务上限。

low/32/32 下共有 793 个通过 precheck 的 POST start，其中 792 个取得 HTTP 200
和终态；唯一缺失的是 task431/idx440。该条 response 完整、formal precheck
通过，但 KernelGym 在响应头前断开，dump 中无 task_id，错误为
`Server disconnected without sending a response.`，随后被客户端包装成
compile=false。它与 step700 已闭因的 `httpx.RemoteProtocolError` 漏重试路径
相同，因此不是候选编译失败；原始四项指标各有最多 1/800=0.125 pct-pt 的未知
宽度。已通过 `page_user` 提供 A=保留原始点并记录不确定度、B=新容器整点重跑；
用户选择 A，因此 step720 已按原始 dump 正式记点，后继每 20-step 评测解除暂停。

代表样本抽查覆盖了 idx796 的 HingeLoss 最快正确结果（7.15415x）、idx515 的
ConvTranspose2d 最慢正确结果（0.0108315x）、idx39 的 compiled-wrong、idx8 的
缺 `.cu` precheck failure、idx21 的真实 NVCC `size_tb` 未定义失败，以及 idx770
的 SDPA 300 秒 timeout。正式 Ray job `raysubmit_z2KaCuQJpa5vEvzJ` 和 runner 均
成功退出；800 样本阶段为 24 分 42 秒。DP0--DP7 各且仅加载一次 LoRA UUID
`95cccff8665e562981a945f12daca449`。有效日志为
`step720/20260729.033345.log`；dump、summary、日志 SHA256 分别为
`b2000c4039c4601264a189352119d504b23e0e0bf293a5f914959c825fbe6594`、
`2afb5aefd50a21739cbd693e0e8a2c0ebf559437e53a28e454f214fd5161e117` 和
`7e560f47e979a8ae1195ab1d63edddc73a9edce4402c26a9bbf7bb725b2fa767`。一次性
容器已删除，Ray 端口释放、8 卡显存归零，zombie 仍为运行前的 15。

## 产物与运行边界

- 评测根目录（node69 本地）：
  `experiments/Eval.KernelBenchL1.DeepSeekV4FlashLoRA.12k.turn1.n8/`。
- step440：`step440/20260727.005140.log`、对应 summary 和
  `dumps/rollout_data/eval_0.pt`。
- step460：`step460/20260727.012810.log`、对应 summary 和
  `dumps/rollout_data/eval_0.pt`。
- H200 评测根目录：`/ssd/csl_v4_h200_eval_20260727/experiments/`
  `Eval.KernelBenchL1.DeepSeekV4FlashLoRA.12k.turn1.n8.h200/`；step460 的
  summary 为 `step460/summary.20260727.152519.txt`，step480 的 summary 为
  `step480/summary.20260728.000302.txt`。
- 编排入口：`examples/kernel_agent/eval.deepseek-v4-flash-lora-curve.sh`；
  `EVAL_ONLY_STEP` 可只跑一个点，已有完整 summary 默认跳过。
- H200 单点入口：`scripts/dsv4/run_eval_step.sh h200 curve <STEP>`；它拒绝忙 GPU，默认
  KernelGym `low/32/32`（训练活动时显式覆盖为 `low/4/4`），并为每点创建和删除
  一个 `--init` 容器。
- step500/520/540/560 的 H200 adapter 均为 229,967,152 bytes、766 个唯一 tensor，
  config SHA256 均为 `f340d4d0e714efcfa93929a5480d2fae540b79d3424cd6d26196d195aa4c3275`；
  model SHA256 依次为 `532e51c994f006e07e71a5ef6dca5a33e87f3db79b24df46458ca5cacb6ea564`、
  `50ffd5208a3e7400bf4beb92bc566455cce417d8a9eb8ca1ac92b179cfc93b84`、
  `9a7466b4980bbb1f3a7f2cb7c2dd62ace61fc88bf02ba91552150ace453b4648`、
  `a99a638fb9b3a43202fa1945c88e21d52163504161b66e9465459ba8e71a3617`。
- 用户选择继续纳入新 checkpoint 后，step580/600 分别从 `iter_0000579/599`
  导出；二者均通过 766 张量、rank32、alpha32、rsLoRA 合同校验，model SHA256
  分别为 `47604297bdc69587327eddf30c79889436b09e8339a3e3ddc0bcd43984497c7b`
  和 `7a4f994318ba6c25655c4a957bd4847dd7e3661593ea0e78fba0bf7223e7938f`，
  config SHA256 与前述点一致。当时的旧单点评测脚本只枚举到 step560，后改为直接
  分派 `EVAL_ONLY_STEP`，否则未来合法 step 会空跑成功；当时本地与 H200 脚本
  SHA256 均为
  `df591cd4ca67206caf91e209a18def0ae6e4416e0afad0827a7eb40433dea35a`。
- 训练随后原子发布 `iter_0000619`，其 32 个 distcp 分片、766 个 LoRA 参数张量和
  rank32/alpha32/rsLoRA 描述一致。step620 导出同样通过 766 个唯一张量、A/B 各
  383、rank32 合同校验，并已通过 staging 目录原子发布到 H200；model SHA256 为
  `7142cc57e4d79513dda2738bec0d63d6679d23a82e47e58e23e5381107edab62`，
  config SHA256 仍为
  `f340d4d0e714efcfa93929a5480d2fae540b79d3424cd6d26196d195aa4c3275`。
- 训练随后又原子发布 `iter_0000639`，其 32 个 distcp 分片和 LoRA 元数据完整。
  step640 CPU 导出通过 766 个唯一张量、A/B 各 383、rank32 和 rsLoRA 等效
  serving alpha `181.01933598375615` 校验，并通过 staging 目录原子发布到 H200；
  model SHA256 为
  `c415f59d0060a87005de827b8fa31dc42e04128b8d8190bae248ba932e664e9b`，config
  SHA256 仍为
  `f340d4d0e714efcfa93929a5480d2fae540b79d3424cd6d26196d195aa4c3275`。
- 训练又原子发布 `iter_0000659`：32 个非空 distcp 分片、`.metadata`、
  `v4_lora_scaling.json` 和 `v4_adapter_checkpoint.json` 均完整。step660 CPU 导出及
  独立 safetensors 复核通过 766 个唯一张量、A/B 各 383、43 层、rank32、无
  非有限或全零张量，rsLoRA 等效 serving alpha 为 `181.01933598375615`。产物经
  staging 目录复算 hash 后原子发布到 H200；model SHA256 为
  `9a27918eacbbeffb2c3eeb4c8e108633b0f2e194511028e6da7a76e19a209db6`，config
  SHA256 仍为
  `f340d4d0e714efcfa93929a5480d2fae540b79d3424cd6d26196d195aa4c3275`。
- 训练随后原子发布 `iter_0000679`：37 个顶层文件、32 个非空 distcp 分片、
  `.metadata` 和两份 LoRA 元数据均完整。step680 CPU 导出和独立 safetensors
  复核通过 766 个唯一张量、A/B 各 383、43 层、rank32、无非有限或全零张量，
  rsLoRA 等效 serving alpha 为 `181.01933598375615`。产物经隐藏 staging 原子
  发布到 H200，并由宿主 hash 与正在运行的 step560 容器分别复核；model SHA256
  为 `98f1fd4db7df4d2479d424d915731075a67b9e726275242d5f9184d80b16ff59`，config
  SHA256 仍为
  `f340d4d0e714efcfa93929a5480d2fae540b79d3424cd6d26196d195aa4c3275`。
  首次导出命令因误用容器 `/root` 工作目录而立即退出且没有写产物，改用仓库绝对
  路径后成功；原子发布命令中的两个辅助 `awk/find` 检查又因远端引号错误报错，
  但最终目录随后以 `set -e` 从宿主和容器内双重重算 hash、大小、结构和数值并
  全部 PASS，因此没有遗留产物缺口。node69 容器和 node64 的可重建临时副本均已
  显式删除，只保留 H200 正式目录。
- 训练随后原子发布 `iter_0000699`：37 个顶层文件、32 个非空 distcp 分片、
  `.metadata` 和两份 LoRA 元数据均完整。step700 CPU 导出和独立 safetensors
  校验通过 766 个唯一张量、A/B 各 383、43 层、rank32、无非有限或全零张量，
  rsLoRA 等效 serving alpha 为 `181.01933598375615`。产物在隐藏 staging 中先
  通过 hash/大小门禁，再原子发布到 H200；H200 宿主和正在运行的 step580 容器
  双重校验 PASS。model SHA256 为
  `caf301482c17454cfbda2226ce73841843ae0839bd21adeaad20b9f8b6d141a4`，config
  SHA256 仍为
  `f340d4d0e714efcfa93929a5480d2fae540b79d3424cd6d26196d195aa4c3275`。
  首次 staging 大小检查误用 `stat %f`（文件模式）而被本地解析门禁拒绝，未执行
  原子改名；改用 `%n` 后重验并发布，正式产物无缺口。node69 容器和 node64 的
  可重建临时副本均已显式删除，只保留 H200 正式目录。
- `iter_0000719` 的 32 个非空 distcp 分片、`.metadata` 和两份 LoRA 元数据完整。
  step720 在 checkpoint 所在的 node64 做 CPU 导出，验证通过 766 个唯一张量、
  A/B 各 383、43 层、rank32、无非有限或全零张量，rsLoRA 等效 serving alpha
  为 `181.01933598375615`。node64 先以隐藏 staging 原子发布，再同步到 node53
  的隐藏 staging 并复算 hash 后原子发布；model/config SHA256 分别为
  `2b30ed3e85882e7b7336cee9d7dfb2c6994bcaa5891cd4f60ac6ceafc4c2646b` 和
  `f340d4d0e714efcfa93929a5480d2fae540b79d3424cd6d26196d195aa4c3275`。首次尝试
  从 node53 读取 checkpoint 时在 metadata 读取前即退出，因为各节点的
  `/nfs/FM` 实为本地盘且 node53 没有该训练 checkpoint；没有创建输出，随后改为
  正确的 node64 导出、node53 原子同步，未留下半成品。
- 两台 rollout 机器只有在各自拥有独立 serving 实例并同步扩容 KernelGym 后，
  才可能接近按题切分的 2 倍吞吐；共享同一个 KernelGym 不能假设时间减半。
- 手动 `kill` zombie 无效，因为进程已经死亡，必须由父进程 `wait()` 或销毁
  PID namespace。H200 新 runner 已通过 `--init` 和每点销毁容器解决新评测的
  回收问题；截至本次复核，主机 72 个旧 zombie 分别属于早于本次评测的
  `csl_v4_tim_probe_20260722`、`ds4_dump` 和 `csl_v4_tail_probe_20260723`。
- 2026-07-28 step500 启动前，runner 的忙卡预检发现另一个 GLM next20k 容器
  占满 8 卡。其首次生成命令因参数错误退出，任务方随后修正并重新开始有效生成；
  首个 1000 条分片完成后，任务方又动态创建并启动了后续分片，因此不能用某一
  时刻目录中的分片数估算总运行时间。按用户选择等待该任务自然完成，没有停止
  外部容器。该任务最终自然完成动态分片 part0000–part0019 并释放资源；等待期间
  step500 没有产生 Ray job、样本或 dump，因此不存在半成品被误计为有效点的问题。
- 为避免等待窗口依赖交互终端，15:06 JST 已在 H200 启动持锁的串行等待队列
  `scripts/dsv4/wait_and_run_h200_eval_queue.sh 500 520 540 560 580 600`。它只在 8 卡
  都低于显存/利用率门槛且 Ray 端口空闲时调用现有单点 runner；任一点失败即停止，
  不会绕过新 `--init` 容器或 KernelGym low/4/4 合同。队列日志为
  `/ssd/csl_v4_h200_eval_20260727/host_logs/queue.20260728.1504.log`，重复队列锁探针
  已通过。20:09 CST 空闲预检通过后，step500 以新容器
  `csl_v4_eval_step500_20260728.200926` 启动；容器 `HostConfig.Init=true`、PID 1 为
  `docker-init`，Ray job 为 `raysubmit_kqnSRhGRZ7WWyp49`。实际 rollout 进程环境核验为
  `KERNEL_EVAL_PRIORITY=low`、`KERNEL_EVAL_RATE_LIMIT=4`、
  `KERNEL_EVAL_WORKER_MAX_CONCURRENCY=4`。
- 原队列只到 step600。为避免它完成后出现无人值守空档，当时的队列脚本新增可选
  `H200_EVAL_QUEUE_WAIT_FOR_LOCK=1`：后继队列会阻塞等待同一独占锁，旧队列释放后
  才进入 GPU 空闲预检，默认行为仍是遇锁立即退出。更新后的脚本通过 `bash -n`，
  该历史版本在本地和 H200 的 SHA256 均为
  `c2543c1dc5307dac380794465af081e4a0990cadead41b3bd807121cd47473c9`。
  H200 已以该模式预排 step620/640/660/680/700，PID `3861203`，日志为
  `/ssd/csl_v4_h200_eval_20260727/host_logs/queue.20260729.successor.log`；启动复核确认
  它停在 `waiting_for_queue_lock`，不会与正在运行的 step600 重叠。
- 同日只读复核发现原 r21 训练已经恢复：node64/node69 的 checkpoint 目录和
  `latest_checkpointed_iteration.txt=589` 一致，四节点 Ray 均 active、无 recent
  failure，rollout 591 正在生成。原计划中的“全部评测后 resume”因此不再是待执行
  动作；训练保持运行。用户随后选择继续评测 step580（对应 `iter_0000579`）及
  后续所有新增的每 20-step checkpoint。
