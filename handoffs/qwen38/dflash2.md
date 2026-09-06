# Qwen3.8 的 DFlash2 / DSpark 适配分析

当前 MTP1 teacher forcing + MTP3 rollout 已通过[完整训练与恢复验证](mtp_training.md)。两个外部 draft 方案中，DSpark 的现成运行时代码更多，但已有本地输出退化记录和未闭合的 GDN 正确性问题；DFlash2 需要手动合入运行时支持，也要复核共享的 GDN verify 路径。二者随 RL 在线训练都需要新增外部 draft 的训练与发布流程。本次只分析源码、模型配置和社区问题，没有下载大权重、占用 GPU、部署服务或修改运行时

## DFlash2 的现成能力与当前差距

官方已有 [Qwen3.8-27B-DFlash2](https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2) 权重。读取模型 metadata 得到约 1.924B BF16 参数、3.85GB 权重文件；配置为五层 sliding attention、窗口 2048、block size 8，读取 target 的 `[5, 19, 33, 47, 61]` 五层 hidden。block size 8 包含一个验证 anchor 和七个 draft token

DFlash2 在一次 draft forward 中并行预测整个 block，再通过动态卷积和 candidate selector 建立位置间关联。SGLang 的算法参数仍为 `DFLASH`，checkpoint 的 `DFlash2DraftModel` 选择具体实现，见[官方架构说明](https://sgl-project.github.io/SpecForge/concepts/DFlash2.html)

当前四机容器的 SGLang 有 DFlash worker、Qwen target hidden capture、GDN verify 后状态提交，以及 sampled logprob 回传路径，但 `models/dflash.py` 缺少 DFlash2 类、动态卷积与 selector。[支持 PR #35371](https://github.com/sgl-project/sglang/pull/35371) 已合并，涉及五个运行时文件和两个测试文件；不能只把 `NEXTN` 改成 `DFLASH` 就启动该 checkpoint。对本地代码副本执行 patch dry-run，五个运行时文件中模型、worker、feature capture 三处不能直接套用，需保留本地重构后手动合入

当前 target 虽然是 FP8，直接读取本地 safetensors header 确认 `lm_head.weight` 与 embedding 均为 BF16，因此初版 selector 要求 dense head 的限制不直接阻塞这份 checkpoint。换成量化 lm-head 时还要包含[后续支持 #35496](https://github.com/sgl-project/sglang/pull/35496)

## 只换 rollout，需要改什么

| 部位 | 适配内容 | 复杂度 |
|---|---|---|
| SGLang 隔离运行时 | 在现有 overlay 上合入 DFlash2 的模型、selector kernel、worker 和 config 支持，保留当前 FA3/GDN/cache/logprob 修复 | 中 |
| 启动参数 | 使用外部 draft 路径、`DFLASH`、block size 8；删除 NEXTN 的 steps/top-k 组合；关闭原生 MTP 在线训练 | 小 |
| 输入预算 | 现有 helper 已按 draft tokens 在服务 context 上限预留空间，block 8 对应最多 40952；还要实测输入上限与第三轮反馈 | 小 |
| 权重更新 | actor 继续更新 target；固定 draft 不接收 `mtp.*`，并核对共享 embedding/lm-head 的引用、CPU backup 和缓存清理 | 中 |
| 真实训练 payload | 验证 T=1、top-p=1、top-k=-1 下的拒绝采样、sampled logprob、top-20+sampled 支持集及 weight version | 中 |
| 容量与稳定性 | 重新测 TP4、40K、C64/C128；记录实际运行并发、Mamba slots、draft KV、CUDA graph 内存与峰值 | 中，需 GPU 验证 |

selector 自身的 top-k=16 与 rollout 返回的 target top-20 是两套不同的数据。即使 draft 只在 16 个候选中提议，正确的拒绝采样仍应恢复 target 的全词表分布；上线验证要检查实际 q 与 target p 的对应关系，不能为了适配 selector 收紧当前 target 的采样参数

当前 MTP 分布式补丁只识别原生 `Qwen3_5ForCausalLMMTP`，不能拿它给 DFlash2 更新权重。冻结外部 draft 时，最小路径是 actor 不启用 MTP，沿用 target 更新，并单独验证 target 更新后 draft 引用和缓存仍一致

五层 BF16 target 特征在 40K 序列上，未压缩的逻辑数据量约 `40960 × 5 × 5120 × 2 = 2.10GB`。这是捕获/传输数据量的估算，实际显存取决于 prefill chunk、融合投影与 TP 切分，不能直接加到静态显存占用上。原 MTP3 的并发上限也不能移用到 DFlash2

## 在线训练的额外工作

[slime 的文档](https://thudm.github.io/slime/advanced/speculative-decoding.html) 支持原生 MTP 在线训练，外部 draft 训练仍标为 WIP。[SpecForge 已有 DFlash2 训练与导出](https://sgl-project.github.io/SpecForge/basic_usage/training.html)，可以复用其 DFlash feature schema 和训练入口，但还需要与当前 RL 流程连接

- 采集指定五层的 hidden、input IDs 和 response loss mask，保证与 target 权重版本、token history 和采样分布一致
- 训练五层 drafter、动态卷积与 selector；其 block 目标及 selector loss 有独立定义，不能复用单层 MTP 的 CE 路径
- 新增 draft optimizer、checkpoint 和发布版本，将更新安全地送到四个 TP4 rollout engine，并处理旧缓存失效
- 确定训练资源和数据通道。当前 PP37/27 让五个 target feature 分布在两个 actor stage，直接嵌入 Megatron 需要跨 PP 汇集，还要处理 TP/CP；独立 SpecForge capture/trainer 是更清楚的接入边界，但会占用额外计算或减少 rollout 容量

工程估算：固定 draft 的最小可运行适配约 1–2 个工作日；包括当前 40K、多并发、refit 与 logprob 契约的验收，约 3–5 个工作日。同步在线训练是另一个项目，约一至两周起，取决于采用独立 capture 还是接入 Megatron；这些是基于源码差距的估算，不是已完成工时或性能承诺

## 收益与社区问题

官方模型卡的 H200、FA3、最多 4096 new tokens、推荐采样参数下，DFlash2 在 C32 GSM8K 的吞吐约比七步 MTP 高 39%。该对照与我们的 H20 TP4、MTP3、40K、多轮 CUDA 代码生成、全词表采样和 top-20 payload 不同，只能证明值得测试，无法预测本工作负载的收益

当前还有未关闭的 [Qwen3.8 DFlash2 并发状态串扰报告 #36548](https://github.com/sgl-project/sglang/issues/36548)：报告者在 RTX Pro 6000 的并发请求下观察到上下文疑似串入其它请求。该 issue 没有足够复现信息和确认根因，不能断言 H20 同样存在；但本任务正是高并发长上下文，应把不同请求的隔离验证列入上线前测试

算法上，保持 target 采样分布不要求 draft 跟着每次 RL 更新：正确拒绝采样使用当前 target 概率；固定 draft 的主要问题是 acceptance 可能随分布漂移下降。此性质仍依赖运行时正确实现，不能替代并发与状态验证

建议先验证固定官方 draft 的 rollout。通过 40K 边界、混合请求隔离、T=1 的概率和 logprob 契约、target refit 后继续生成，再比较同一真实样本集下的吞吐与容量。观察 RL 更新后的 acceptance 漂移后，再决定是否建设在线 DFlash2 训练

## DSpark 的现成模型与接入差异

[RadixArk/Qwen3.8-27B-DSpark](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark) 提供约 1.857B BF16 参数、3.71GB 的外部 draft。它使用并行 backbone 加轻量逐位置 Markov head 建模 block 内依赖，并训练 confidence head，支持按置信度安排验证预算，见 [DSpark 论文](https://arxiv.org/abs/2607.05147)

| 项目 | 当前公开 DSpark checkpoint | DFlash2 checkpoint |
|---|---|---|
| 参数量 | 1.857B | 1.924B |
| draft attention | 五层 full attention，YaRN | 五层 sliding attention，窗口 2048 |
| target features | `[5, 19, 33, 47, 61]`，hidden size 5120 | 相同 |
| 位置间依赖 | VanillaMarkov，rank 256 | 动态卷积 + top-16 candidate selector，rank 256 |
| 配置中的 block size | 7，表示七个 draft proposals | 8，表示八 token 的 block |
| target verify width | 8 | 8 |
| 训练监督宽度 | `training_block_size=16` | block size 8，含 selector 目标 |
| 当前 SGLang | 已有 `DSparkDraftModel` / `Qwen3DSparkModel`、worker、Markov/confidence 与 logprob 路径 | 缺少 DFlash2 模型、卷积和 selector，前述三处需手动合入 |

这里必须区分字段口径：DSpark 的 `block_size=7` 对应 `--speculative-num-draft-tokens 8`；`--speculative-num-steps 1` 表示一次并行 draft backbone forward，之后 Markov head 仍逐位置产生七个提议。`training_block_size=16` 是该外部 draft 的训练监督宽度，与原生 MTP 的单步 teacher forcing、MTP layer 数量分别定义

固定 DSpark 的最小接入是：在维护 launcher 中增加外部 draft 分支，使用 `DSPARK`、明确 BF16 draft 的 `--speculative-draft-model-quantization unquant`、设置 gamma/verify width，并关闭原生 MTP 在线训练。首先固定 `SGLANG_RAGGED_VERIFY_MODE=static` 做完整验证；通过后再单独评估 confidence-based compact verify、STS 校准与 SPS 成本表，不能直接复制其它硬件的调度表。target refit、共享 lm-head、版本和缓存要求沿用前面的外部 draft 约定

### T=1、TP4 与 40K 的成本

本地 `dspark_draft.py` 只在 `all_greedy` 时把 Markov proposal sampler 放入 draft CUDA graph。当前训练采用 T=1，会执行逐位置的采样分支；backbone 仍可使用 CUDA graph，这与关闭整个服务的 CUDA graphs 是不同的设置。`models/dspark.py` 先汇集完整词表 logits，再执行 Markov 修正；DFlash2 的上游 selector 则支持 sampled graph path，TP 候选选择只汇集各 shard 的 top-k。由此推断，DSpark 在当前 T=1/TP4 下有额外的采样与通信成本，具体影响需要相同负载实测

五层 full attention 的 draft KV 随历史长度增长。按 BF16 KV、8 KV heads、head dim 128、TP4 且正常分片估算，一条 40960-token 独立序列的 draft KV 为 `40960 × 5 × 2 × 8 × 128 × 2 = 800MiB`，即每张卡约 200MiB；64 条都达到该长度时仅这部分约 12.5GiB/卡。该估算不包含 target KV、GDN state、logits、graph workspace，也不代表当前配置能够容纳全部请求；KV 精度、prefix sharing 与 allocator 会改变实际占用

DFlash2 的训练配置使用 2K sliding attention，历史访问范围更小，但 SGLang 是否真的只分配这部分物理 KV 还取决于 compact/window cache 实现，不能直接宣称节省 20 倍显存。给 full-attention DSpark 强行加 2K window 会改变 draft 的上下文条件，需重新测 acceptance

### 正确性与已有失败记录

仓库的[既有 16K 受控 gate](training_validation.md#单轮-serving-与长尾的已证边界) 固定八个 prompt、medium、T=1、top-p 0.95、top-k 20、FP8 TP4、FA3、Triton GDN、CUDA graphs 和并发 1。DSpark 的 8/8 输出未通过 precheck，其中六条早停、两条长重复；同条件 NoSpec 和 MTP 各有 7/8 通过。DSpark 的 sampled logprob 全部有限，说明“接口返回正常”不足以证明生成健康。本次只复核该记录，未重跑，也未将其根因直接归到某个社区 issue；它不能代表后续所有代码或 checkpoint revision

[问题 #35150](https://github.com/sgl-project/sglang/issues/35150) 提供了更具体的控制：在 Qwen3.8 NVFP4、Triton、TP1 上强制拒绝全部 draft，仅提交 target 自己的 token，仍与普通 decode 累积偏离；FP32 SSM state 只能延迟偏离。当前该 issue 仍开放

- [PR #36014](https://github.com/sgl-project/sglang/pull/36014) 定位到 beta 的精度语义：packed decode 将 sigmoid 结果经过 activation dtype，再进入 FP32 更新，而 target verify 保持 FP32。该 PR 的单步 kernel 测试通过，但仍未合并，也明确不声称关闭整个 #35150。本地 overlay 的 beta 行仍是保留 FP32 的旧表达式，未包含这项修复
- [PR #35541](https://github.com/sgl-project/sglang/pull/35541) 还尝试对齐 full-attention decode/verify 路由；当前说明承认组合改动仍未通过 clean packed-decode 对照，因此不能把它当作已完成修复
- #35150 的后续评论也报告 DFlash2/GB10 的类似现象；其与 #36548 并发串扰的关系尚属报告者推测。这提醒我们，GDN target-verify 数值路径可能同时影响多个 draft 算法，不能据算法名称认定某一方天然规避。现有 MTP 训练/恢复与 payload 验证也不等同于逐 token AR 等价性证明

所以 DSpark 的优先检查应是固定 prefix 的单步状态对齐、强制拒绝的长序列控制、正常 T=1 输出健康、并发请求隔离，然后才是吞吐。诊断发现普通 decode/verify 的细微差异后，需要进一步量化采样分布和任务质量影响；既不能忽略已出现的输出退化，也不能仅凭一次 greedy 分叉断言任意生产配置都发生语义损坏

### 在线训练与公开跑分

[SpecForge 的训练与导出入口](https://sgl-project.github.io/SpecForge/basic_usage/training.html) 已支持 DSpark，使用显式 draft config、`training.strategy: dspark` 和相应 feature capture；[配置指南](https://github.com/sgl-project/SpecForge/blob/main/examples/configs/README.md) 定义 CE/L1 与 confidence head 的训练项。接到当前 RL 流程时，仍需处理五层 feature、mask、target 版本、独立 optimizer/checkpoint、draft 发布和失效缓存，无法通过打开 `--enable-mtp-training` 完成。confidence calibration 和 hardware cost table 还需要单独维护

[当前 DSpark 模型卡](https://huggingface.co/RadixArk/Qwen3.8-27B-DSpark)的 v2 吞吐对照已经包含原生 MTP3：在其 H200 TP1、FP8、xhigh、T=1/top-p=0.95/top-k=20、最多 2048 new tokens 的 C32 HumanEval 单元中，DSpark v2 为 2472.3 tokens/s，MTP3 为 2296.9，约高 7.6%；同并发 MT-Bench 则约低 7.5%。这些结果只支持收益依赖 workload。该卡的 acceptance 表另用 GB300 与 NVFP4；此前 DFlash2 卡的 H200 对照使用 4096 new tokens、七步 MTP 以及未明确标成当前 v2 revision 的 DSpark。不能跨表把 DSpark v2 与 DFlash2 直接排名

### 当前建议与工作量

| 目标 | DSpark | DFlash2 |
|---|---|---|
| 固定 draft 的最小接入 | 约 0.5–1 个工作日，已有运行时代码可用 | 约 1–2 个工作日，需手动合入运行时 |
| 当前训练负载验收 | 若输出健康与状态检查通过，约 2–4 个工作日；复现已知漂移后的修复时间另计 | 约 3–5 个工作日；同样需要检查共享 GDN 路径 |
| 随 RL 在线训练 | 独立工程，约一至两周起 | 独立工程，约一至两周起 |

这些是源码分析得到的工程估算。短期保留已完成完整环路验证的 MTP1 训练 + MTP3 rollout；外部 draft 先做固定权重对照。若优先回答“当前 DSpark 是否已经恢复正常”，其接入成本较低，适合先做小规模 correctness 复测；若优先探索 T=1/TP4 的长上下文性能，DFlash2 的局部 attention 和 sampled selector graph 路径值得优先测试。当前证据不足以承诺两者谁在真实训练中更快或更稳定

## 本地证据

- [分析产物](../../local_artifacts/qwen38/dflash2_analysis_20260906/) 保存官方 config、模型 metadata、model card、PR 状态和原始代码
- `patch_dry_run_result.json` 保存补丁冲突；`local_runtime/manifest.json` 标识当前隔离 SGLang 中实际读取的文件；`target_shared_weight_dtypes.json` 保存本地 target 的 shared weight dtype
- HF revision：`50307d4c4cde6860d4eee73e2547cd786fe8e8a4`；DFlash2 SGLang merge：`c14312a66420b75ca9a11bf1817c4db1fa26b097`

- DSpark 的 [分析产物](../../local_artifacts/qwen38/dflash2_analysis_20260906/dspark/) 保存模型 config/card/metadata、SpecForge 文档、问题与 PR 状态、历史 gate 摘录和本地源码 hash；本轮没有 GPU 实验
- DSpark HF revision：`b9a5dbdf03bc999c6c73c426b19c2d9041cea393`；权重 SHA256：`2aff025f45823b40ebe726b9dfa40302f3512bd9a11c3a7347de32a567acd9a7`
