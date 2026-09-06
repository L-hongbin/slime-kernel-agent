# TRLOO 多轮轨迹合并

`--pack-multi-turn-trajectories` 在训练数据转换阶段把同一 trajectory 的多个 turn 合为一条 causal sequence，三个回答段共同参与一次训练 forward/backward。Rollout、过滤和逐轮 TRLOO reward normalization 保留原路径；每个回答段使用自己的 advantage。该开关默认关闭，activation recompute、reference scoring 和多次 optimizer step 仍按原配置执行

本次验收检查训练目标与权重合同、高精度数学等价、真实训练执行和单步计算收益，并继续追查生产 BF16 的巨大梯度差异。只比较 reward 不足以发现 mask、target、loss 分母或模型导数变化。真实回放及固定系数、固定形状对照已完成；长期训练质量及端到端提速未验证

实现位于 [trajectory_packing.py](../../slime/utils/trajectory_packing.py)，由 [RolloutManager](../../slime/ray/rollout.py) 在 reward postprocess 之后、DP schedule 之前调用。新 worktree 为 `/nfs/FM/chenshuailin/projects/kernel_agents/slime-trloo-packed-trajectories`，分支 `feature/trloo-packed-trajectories`，基线为 `dev_csl2` 的 `9e25fd9a`

## 配置

维护入口支持 `PACK_MULTI_TURN_TRAJECTORIES=1`。以下命令只打印三轮配置，使用与原三轮部署相同的 context cap 和 R31；实际执行前仍需同步节点代码、数据及运行环境，按[训练验证](training_validation.md)完成 rollout-only 和 train-only 检查

```bash
CONFIG_DRY_RUN=1 \
PACK_MULTI_TURN_TRAJECTORIES=1 \
MAX_TURNS=3 TURN_MAX_CONTEXT_LENS="24576 32768 40960" \
MAX_CONTEXT_LEN=40960 RECOMPUTE_NUM_LAYERS=31 USE_NODE64_ROLLOUT=0 \
bash examples/kernel_agent/run.t1.qwen3.8.27B.fasync.sh
```

`TURN_MAX_CONTEXT_LENS` 同时控制生成长度和 overlong penalty 的 effective response cap，第二轮仍按 32K 计算。未提供逐轮数组时保留原来的 first-turn cap 行为；数组与 `--first-turn-max-context-len` 互斥。`PACK_MULTI_TURN_TRAJECTORIES=0` 可切回逐轮样本，比较时其余参数和 replay 必须一致

## 等价性合同

- 先按 `(prompt group, turn_idx)` 算 TRLOO advantage，再映射到回答 token；反馈、模板文本和被过滤轮次的 loss mask 为零
- 后续输入必须保留每个有 loss 的回答的原始预测前缀。任意较早 token 改写都会报错，避免 decode/re-encode 或历史裁剪造成训练目标变化
- Qwen 补 `</think>` 时可后移原 `<|im_end|>`。`target_tokens` 将原结束 token 放在原预测位置，实际模型输入继续使用修复后的历史；对应 behavior logprob/top-k 同步映射到该位置
- `token_rewards` 经 actor 的 CP 切片进入 advantage 计算；逐轮奖励折叠和可选 advantage whitening 的统计总体保持不变
- `loss_normalization_counts` 保留原来的 `sum_turn max(loss_mask.sum(), 1)`，包括被过滤或补齐轮次。按 trajectory 归一化时保留原 `group_mask_sums`；训练步仍按原 group 数切分
- 逐轮 reward/truncation 指标通过 `packed_turn_metrics` 保留原分母。训练侧序列长度描述合并后的物理序列；rollout 侧逐轮长度指标仍在合并前记录

当前支持 TRLOO 的 PPO 和 predictive Top-K DPPO policy loss。Sequence MIS、TIS、OPSM、routing replay、LoRA、critic/OPD、自定义 advantage/converter/reducer/postprocessor、MTP 和 allgather-CP 不在本次支持范围，参数检查会拒绝这些组合。要求 attention/hidden dropout 为零

## 验证与边界

[CPU 等价性测试](../../tests/test_trloo_trajectory_packing.py)使用因果小模型和真实训练 converter、CP loss 布局、advantage/reducer/policy loss，仅将 fused vocabulary logprob/entropy kernel 替换为 torch softmax。覆盖 CP=1/2、PPO/DPPO、token/trajectory 归一化、结束 token 后移、过滤/补齐、非等长 turn 数，以及 DP→actor 字段传输。该证据不代表多 GPU attention 通信或 27B optimizer 的实测结果

[单卡 GDN 检查](../../tests/test_trloo_trajectory_packing_gpu.py)使用随机初始化的小型 Qwen3.5 GDN，跨 recurrent chunk 边界比较三轮与合并后的 CUDA forward/backward，并调用真实 Megatron fused loss。本机缺少 `causal-conv1d`，检查显式选择 FLA recurrence，卷积使用 PyTorch；它不覆盖正式部署的 distributed FlashQLA + TP4/PP2/CP2

CPU 检查及相关回归共 123 项通过（含 replay capture 的保存与异常恢复检查），单卡 GDN 检查通过，launcher dry-run 确认了三轮 cap 和 packing 开关。GDN 检查中的固定权重与 behavior 数据精度对照如下；日志和本地源码 hash 见 [GPU 日志](../../local_artifacts/trloo_packing/gdn_gpu.log)与 [provenance](../../local_artifacts/trloo_packing/gpu_provenance.json)，CPU 结果见[回归日志](../../local_artifacts/trloo_packing/formal/updated_cpu.log)

| 指标 | FP32 计算 | BF16 activation、FP32 梯度缓冲 |
|---|---:|---:|
| split / packed loss | -0.03544896 / -0.03544891 | -0.03549813 / -0.03549942 |
| split 与 packed 梯度相对差 | 2.37×10⁻⁵ | 4.33% |
| split 与 packed 梯度 cosine | 接近 1 | 0.99906 |
| split 梯度相对 FP32 的差异 | — | 3.24% |
| packed 梯度相对 FP32 的差异 | — | 4.09% |

该配对实验确认了低精度计算会改变共享前缀 backward 的数值结果；只提高梯度缓冲精度不能消除差异。尚未分离 GEMM、GDN recurrence 和 backward 累加各自的贡献。GPU 测试对 FP32 等价性设严格断言，BF16 部分保留数值审计及有限性检查，不能将通过测试解释为 BF16 位级等价或已通过正式 27B Megatron replay。正式 BF16 的完整 replay 已执行，结果见下文；数值差异如实记录，不将小模型测试扩大解释为 27B 数值等价

`pre-commit` 和 `git diff --check` 通过。外部 `cursor-grok-4.6-xhigh` 只读审查发生 18 次重连/重试，在共享 30 分钟预算结束时退出 124，未产生 final result，余下预算不足以执行 fallback。已改用本地自查，未将其记录为独立审查通过；[原始 event stream](../../local_artifacts/trloo_packing/review_grok.jsonl)保留在本地

真实数据审计使用留存 Qwen L3 evaluation dump 的前 3 轮，共 400 条 trajectory、800 个轮次衔接。该评测路径重渲染历史，在原回答前额外插入空 `</think>` 段，800 个衔接均改变已评分前缀，因此不适合作为合并等价性 replay。可人工查看[原始文本差异](../../local_artifacts/trloo_packing/real_example_review.txt)和[统计](../../local_artifacts/trloo_packing/real_prefix_audit.json)。该旧评测数据已由下述新采集的真实训练 replay 替代

## 正式 27B 固定回放

2026-09-05 在 node69/70/53 的专用容器内完成真实三轮 rollout；训练采用 node69/70 的 H20×16、TP4/PP2/CP2、SP、PP33/31、R31、BF16 与 distributed FlashQLA。三节点 1197 个源码文件 hash 一致；[源码清单](../../local_artifacts/trloo_packing/formal/source_manifest.json)、节点 provenance、各组完整环境 JSON 和原始日志保存在同一 formal 目录。actor checkpoint 使用 `torch_dist_tp4_pp2_distributed_gdn_flashqla/release`

[真实 rollout](../../local_artifacts/trloo_packing/formal/rollout_0.pt) 的 SHA256 为 `6e90bdd2aa0a9f094f07ef93ce457be08e4ecdb444c2611d9365549962315582`，含 2 个 prompt、8 条 trajectory、24 个真实 turn。原始输入共 673543 token，合并后 275815 token，最长 40958；237108 个评分 token 均有非零 advantage，4 处结束 token target 覆盖。有一轮被过滤，原分母的 clamp 使总 normalization count 为 237109。输入、反馈与结束位置已人工抽查，见[审计](../../local_artifacts/trloo_packing/formal/rollout_audit.json)和[真实文本](../../local_artifacts/trloo_packing/formal/real_train_example.txt)

固定同一 checkpoint 和 replay，各执行一步真实 optimizer update；诊断 hook 捕获全部 26895998464 个 norm-contributing FP32 gradient 元素，以及每个评分 token 的实际训练 logprob。梯度在 optimizer step 后、清零前采集；计算时间结束于 optimizer step 后 CUDA synchronize，排除大体积 CPU 拷贝和文件写入。记录的是 torch allocator 峰值，不能解释为 NVML 的物理显存峰值

| 指标 | split0 | split1（重复基线） | packed |
|---|---:|---:|---:|
| microbatch 数 | 24 | 24 | 8 |
| 计算耗时，秒 | 123.610 | 123.474 | 74.514 |
| loss | 0.006592632 | 0.006592632 | 0.006586703 |
| 完整梯度 norm | 0.176830 | 0.176696 | 0.148673 |
| 梯度相对 split0 的 L2 差 | — | 1.34% | 71.35% |
| 梯度与 split0 的 cosine | — | 0.999910 | 0.712365 |
| allocator allocated 峰值，GiB | 42.490 | — | 42.419 |
| allocator reserved 峰值，GiB | 79.789 | — | 62.527 |

packed 的单步计算约快 1.66 倍，训练目标与权重合同检查通过；**生产 BF16 数值不等价**。[完整梯度对照](../../local_artifacts/trloo_packing/formal/packed_gradient_summary.json)的差异远大于[重复基线](../../local_artifacts/trloo_packing/formal/baseline_repeat_summary.json)。两个 split 基线的全部评分 logprob 位级一致，说明其 1.34% 梯度波动发生在 backward 阶段

逐 token 对齐确认 target、behavior logprob 和 advantage 完全相同。packed 的第三轮 28222 个评分 logprob 位级一致，前两轮存在差异；8 条轨迹的首个明显差异均紧跟 split 输入的 SP 分片边界，见[逐 token 对照](../../local_artifacts/trloo_packing/formal/packed_logprob_comparison.json)和[边界证据](../../local_artifacts/trloo_packing/formal/sequence_parallel_boundary_hint.json)。下述单轨迹探针已定位首个差异的原因；不将这一局部根因解释为全部梯度差异或长期学习影响已解决

第一次 packed 提交客户端退出 137，未发现本任务的 Ray submission 或有效训练结果；该次排除。同期 Ray session 被另一 worktree 的任务更换，SIGKILL 来源未证实。后续使用 `REUSE_RAY_CLUSTER=1` 成功重跑；其他任务占用资源时继续离线诊断，不重启共享 Ray。第二次外部 Kimi 审查进程退出 0，但 final result 只有过程陈述，未给出有效审查结论；该结果同样不计作独立审查通过

### 首个差异的因果证据

从相同真实 dump 取 trajectory 0 的三个 turn，继续使用完整 27B checkpoint 和原 TP/PP/CP/SP 配置。该组只有一个 trajectory，TRLOO advantage 为零，**仅用于前向诊断**。源码与配置同步清单为 [probe manifest](../../local_artifacts/trloo_packing/formal/probe_source_manifest.json)；临时 hook 的精确源码作为[本地证据](../../local_artifacts/trloo_packing/formal/trloo_prefix_probe.py)留存，不作为维护入口

原拆分路径自身的 turn0/turn2 前缀对照，以及 split/packed 对照，均给出同一结果：前四层抽样 norm 参数的 TP/CP 副本相同，第一层输入、in_proj 输出、GDN 输出和 out_proj 的四份 TP partial 全部一致；第一次 reduce-scatter 后，在全局 hidden row 3072 开始不同。实际 `calculate_per_token_loss=True`，没有隐藏的 microbatch 分母变化。见[首个差异](../../local_artifacts/trloo_packing/formal/first_divergence.json)、[激活对照](../../local_artifacts/trloo_packing/formal/probe_split_vs_probe_packed_activations.json)和[norm 对照](../../local_artifacts/trloo_packing/formal/probe_split_vs_probe_packed_norms.json)

将该位置的四份 BF16 partial 在 CPU 上按 TP 顺序逐次相加并舍入，顺序 `(2,3,0,1)` 与 `(1,2,3,0)` 分别**位级复现**短、长前缀的全部 5120 通道结果，均为零不匹配。因此首个差异来自 SP 接收 rank 变化带来的 BF16 归约顺序变化。一个实际通道的四份 partial 为 `[0.051025390625, 0.0054931640625, 0.08349609375, 0.000759124755859375]`；两种 BF16 顺序的结果为 `0.1416015625` 和 `0.140625`，FP32 和为 `0.14077377319335938`。见[求和顺序证据](../../local_artifacts/trloo_packing/formal/collective_order_proof.json)

这是原 BF16 TP/SP 路径已有的长度依赖；合并让已评分前缀改用更长的并行形状，从而暴露该数值变化。完整梯度差异平方和的 98.76% 来自前半模型，见[分段统计](../../local_artifacts/trloo_packing/formal/gradient_difference_by_pipeline_stage.json)，但这项分解不能单独证明所有梯度差异都来自第一次归约

针对此数值问题的独立 `cursor-grok-4.6-xhigh` 审查已返回有效 final result，支持上述定位方法，指出 Userbuffers overlap 可能绕过 Python 探针，并要求把 FP32 归约控制视为配置变更。正式配置未开启 TP communication overlap；精度控制也显式断言了该条件。审查范围是数值机制与探针，见[完整结论](../../local_artifacts/trloo_packing/formal/collective_review_verdict.json)，不将其扩大为全部实现或生产配置已通过

### 梯度差异幅度的定量解释

这里的 71.35% 定义为 `||g_packed - g_split||₂ / ||g_split||₂`，不是 71% 的参数出错，也不是梯度 norm 下降 71%。实际 norm 比为 0.84077，方向夹角约 44.57°；差异平方和的 68.40% 来自垂直于原梯度的分量，因此不符合单纯分母错一个倍数的特征。该分解见[定量分析](../../local_artifacts/trloo_packing/formal/gradient_amplification_analysis.json)

Predictive DPPO 的参数梯度可写为 `g = -Σ_t c_t ∇θ logπ_t / Z`，其中 `c_t = A_t · min(ρ_t, 5) · keep_t`；importance weight 和动态 keep mask 均 detach，唯一 autograd 路径是 sampled-token logprob，见 [ppo_utils.py](../../slime/utils/ppo_utils.py)。Reward/advantage 一致约束了 `A_t`，不保证 `ρ_t`、`keep_t` 或网络对参数的导数相同

固定回放给出以下进一步证据：

- 原配置中，21074 个 token 的 logprob 差异超过 0.01，2758 个超过 0.1；最大差异 0.6476 对应约 1.91 倍概率比例，不能用接近零的中位数代表所有 token
- 不计动态 DPPO keep mask，系数 `A·min(ρ,5)` 的相对 L2 差为 2.00%；sampled-logit 对应分量 `c·(1-π_t)` 的相对差为 5.67%，后者也不是全词表 logit 梯度
- 原 split/packed 各屏蔽 25/22 个评分 token。[训练指标](../../local_artifacts/trloo_packing/formal/native_loss_gradient_metrics.json)对应的屏蔽绝对系数质量均约占 0.0129%。即使允许任意选择这 25/22 个位置，`dLoss/dlogprob` 相对 L2 差的保守上界仍为 10.46%，见[上界](../../local_artifacts/trloo_packing/formal/dppo_coefficient_difference_bound.json)。上界使用三角不等式，并以最大的若干系数平方和界定可能删除的范数；共同分母 Z=237109 抵消
- 原始标量系数正负质量分别约为 30959.5 和 -29461.0，净值仅占绝对质量的 2.48%。这说明标量汇总会抵消，但各 token 的参数导数方向不同，不能据此声称参数梯度也有相同的抵消率

因此，简单的 importance weight 整体缩放不能解释全部参数梯度差异。需要考察从 token 输出导数到参数梯度的映射与求和：网络 Jacobian 变化、不同 token 梯度之间的抵消，以及共享前缀一次 backward 相对分开 backward 的低精度舍入，均可能放大差异。前半模型贡献 98.76% 的差异平方和与这种放大相容，但不能单凭该分布认定 FlashQLA backward 有 bug，或分配每种机制对 71.35% 的贡献

固定形状和固定上游导数的控制已补齐。记 A 为原短前缀逐轮训练，F 为同样短前缀、改用 packed 的固定 DPPO 系数，B 为每个 turn 都使用完整 trajectory 输入但只保留本轮 loss mask，C 为一次合并训练。A/F/B 均为 24 个 microbatch，C 为 8 个。F 与 A 的全部评分 logprob 位级一致，F 与 C 的实际 coefficient/keep 位级一致；B/C 的 logprob、coefficient/keep、target、advantage、behavior logprob 全部位级一致。见[固定系数合同](../../local_artifacts/trloo_packing/causal/short_frozen_contracts.json)与[固定形状合同](../../local_artifacts/trloo_packing/causal/full_split_vs_packed_coeff_contracts.json)

[完整梯度分解](../../local_artifacts/trloo_packing/causal/decomposition_four_summary.json)覆盖同样的 26895998464 个 FP32 元素，所有 norm 均小于 clip_grad=1。C 是新一次 packed capture，因此 A/C 为 71.28%，与原 71.35% 接近

| 对照 | 隔离的变化 | 相对 A 的梯度 L2 差 | 该向量项平方 / 总差异平方 |
|---|---|---:|---:|
| A−F | 短前缀形状不变，只换 DPPO 系数 | 11.60% | 2.6497% |
| F−B | 系数固定，改用完整序列计算形状 | 69.72% | 95.6678% |
| B−C | 完整形状固定，逐轮反传改为联合反传 | 1.19% | 0.0277% |
| A−C | 总变化 | 71.28% | 100% |

三组交叉项共占总差异平方的 1.6548%。这是 `A−C=(A−F)+(F−B)+(B−C)` 沿给定控制路径的向量分解，各项不能解释为相互独立的因果百分比。B−C 若以 B 自身 norm 归一化为 **1.4078%**，接近原配置重复基线的 **1.3403%**。合并三次 backward 的作用很小；固定系数后仍有很大的长度/形状相关模型导数变化

实际 DPPO 系数的相对 L2 差为 **2.2807%**，13 个 token 的 keep 状态改变，屏蔽数为 25/22，见[精确系数对照](../../local_artifacts/trloo_packing/causal/coefficient_exact_contrast.json)。它在固定短前缀模型路径上产生 11.60% 的参数梯度变化，证明 token 导数变化确有放大；未据此给每个 kernel 分配放大倍数

通过 optimizer 自身的 model-to-shard 映射，完整梯度已对齐到参数名和全局层号。固定系数 F−B 的分层相对差，从第 49–64 层的约 4.87%，增至第 33–48 层的 22.95%、第 17–32 层的 74.98% 和第 1–16 层的 85.91%，见[深度分布](../../local_artifacts/trloo_packing/causal/gradient_depth_profile.json)。Full attention 的 Q/gate/K/V 权重均有约 76%–85% 的变化，V 权重占其中最大的绝对差异，见[参数块分解](../../local_artifacts/trloo_packing/causal/attention_blocks_summary.json)。这些是参数梯度分布；下文的 hidden-state adjoint 与局部 VJP 控制进一步定位了放大过程，未将全部变化归因于 softmax 或 FlashQLA backward

针对 FP32-TP 反例的数值补审由 `kimi-k3-high` 在只读 ask 模式完成，覆盖 FP32-TP 反例、DPPO 系数上界及梯度幅度解释，见[有效 final result](../../local_artifacts/trloo_packing/formal/gradient_review_kimi_verdict.json)。采纳其“首个根因不等于整体解释、差异是实质方向漂移、需要控制实验”的结论；没有采纳其“约 40 倍参数梯度抵消放大已经证实”“Jacobian 漂移已证实占主导”“改变精度后 norm 变化等于基线重复运行不稳定”等过强表述，理由见[审查采纳记录](../../local_artifacts/trloo_packing/formal/gradient_review_adjudication.json)

一次早期 Grok 补审自行启动嵌套 reviewer，违反任务显式禁止委派约束，已只终止该审查进程树；随后在原 30 分钟预算内切换 Kimi 并取得有效结论。暴露的 skill 缺口是默认 `agentp --print` 仍可调用 shell，继承的 review 工作流可能再次触发 reviewer。当时使用明确 final-reviewer 角色、禁止重启 skill/委派和 `--mode ask` 完成替代；该约束仅用于本次审查。后续 skill 修改已按用户意愿回退，本功能提交不包含 skill 策略变更

### 精度控制的边界与交付范围

在相同 8 条真实 trajectory 上，临时将 TE TP reduce-scatter 改用 MCore 的 BF16 通信、FP32 累加实现，保持 BF16 GEMM。拆分与合并均完成一步 update；[完整梯度对照](../../local_artifacts/trloo_packing/formal/fp32tp_gradient_summary.json)的相对 L2 差为 93.20%、cosine 为 0.37470，未消除整体数值差异。[逐 token 对照](../../local_artifacts/trloo_packing/formal/split_fp32tp_vs_packed_fp32tp_logprobs.json)显示，原 SP 边界差异消失，8 条轨迹的新差异均紧跟 CP 边界；见[边界统计](../../local_artifacts/trloo_packing/formal/fp32tp_boundary_hint.json)。该结果支持多处并行切分会影响数值；随后 CP 探针与独立参考对照进一步定位如下

精度控制第一次因诊断适配传入三维缓冲区、而 MCore helper 需要一维缓冲区而失败。改为展平视图后，先通过[四进程同步/异步 CPU 通信检查](../../local_artifacts/trloo_packing/formal/fp32_collective_shape_check.log)，再重新同步三节点并成功重跑；失败运行不计入结果。成功控制的[源码清单](../../local_artifacts/trloo_packing/formal/fp32tp_source_manifest.json)与[诊断源码](../../local_artifacts/trloo_packing/formal/fp32_tp_replay.py)保留在本地证据目录

通信精度控制仅用于诊断，没有作为生产精度修复保留。改变 TP 求和精度并不等于取得了全模型 FP32 真值，也没有消除长度依赖

### 逐层反传的实测定位

在相同 8 条 replay 上，F/B 各补一次训练，仅捕获 trajectory 0 第一轮的全序列 layer-input activation 和梯度。Checkpoint 的原 microbatch 标识随 ctx 保存并在重算时恢复；CPU 检查验证交错 microbatch 不会串标签。节点源码、依赖和冻结系数 hash 全部匹配，见[执行清单](../../local_artifacts/trloo_packing/causal/depth_source_manifest.json)。两个探针运行的全部 237108 个评分 token 合同都与原 F/B 控制位级一致

前 24077 个输入 token 位级一致，原结束 token 输入改写位置不纳入前缀比较。重建每层全部 CP/SP 分片，确认每个全局位置恰有一个 owner；共 64 层输入加 final norm 输入 65 处。两种形状的所有未来位置梯度均严格为零，排除了所测样本上的未来 loss/causal backward 泄漏。见[完整逐层数据](../../local_artifacts/trloo_packing/causal/depth_summary.json)与[曲线](../../local_artifacts/trloo_packing/causal/hidden_depth_profile.png)

| 层输入 | activation 相对差 | hidden-state adjoint 相对差 |
|---|---:|---:|
| 1 | 0% | 89.76% |
| 16 | 0.32% | 91.67% |
| 32 | 0.67% | 70.70% |
| 40 | 1.16% | 32.10% |
| 48 | 2.16% | 9.94% |
| 56 | 4.23% | 4.45% |
| 64 | 4.04% | 3.45% |
| final norm | 4.01% | 3.11% |

这直接展示了差异沿反传深度增大的过程，比参数梯度的分层统计更接近传播机制。捕获点在 optimizer 最终按 token 分母缩放之前，相对比较使用共同因子；不能把表中原始 hidden adjoint norm 当作归一化 loss 的梯度 norm。为分开本层计算路径变化与上游差异传播，再执行一次 full-shape replay：先保存每层算出的输入梯度，再向前一层返回 short-shape 捕获的输入梯度。这样每层均使用 full-shape activation 和固定的 short-shape upstream；该梯度覆盖仅用于诊断，未保存或部署其更新后的 checkpoint

[局部 VJP 对照](../../local_artifacts/trloo_packing/causal/local_vjp_summary.json)确认 65 处 full-shape activation、全部评分 token 的 logprob/coefficient/keep 与 B 位级一致，所有 suffix gradient 为零。64 个 Transformer layer 在固定各自上游梯度后，本层输入梯度的相对差**最大只有 2.604%**。第一层本地差为 0.473%，完整链路的差为 89.760%；第 36 层对应为 2.604% 和 63.287%，见[对照曲线](../../local_artifacts/trloo_packing/causal/local_vjp_depth_profile.png)

记原 short、full 的 layer-input adjoint 为 `F_i`、`B_i`，局部控制为 `L_i`。实测恒等式 `F_i−B_i=(F_i−L_i)+(L_i−B_i)` 将本层变化与上游差异的传播分开。第 36 层对实际 upstream 差异向量的传播增益约为 2.157，第 48 层约为 1.841；多处 full attention 层反复传播并放大差异，与前半模型巨大的参数梯度变化闭合。低精度 backward 并非严格线性，这些是给定样本、方向和控制路径下的观测增益，不能解释为全局算子范数、所有输入的条件数或各 kernel 的独立因果占比

因此，本次巨大梯度差异的主要因果链为：长短序列改变 BF16 SP/CP 计算分块和舍入，前向 activation/logprob 出现变化；DPPO 系数与网络导数随之改变；较小的本层反传变化在多层复合传播中形成巨大的前部 adjoint 和参数梯度差异。共享前缀从三次 backward 改为一次的单独作用约为重复基线量级。该证据覆盖本次真实 8-trajectory 完整参数对照，以及其中 1 条 trajectory 第一轮的逐层局部控制；没有将其扩大为全部底层 kernel 的正确性认证，也没有证明长期训练质量不变

### SP 与 CP 的计算检查

四卡 SP 检查覆盖实际 MCore/TE RMSNorm→column linear→SiLU→row linear，以及 all-gather/reduce-scatter 的前向和伴随算子。SP 开/关、FP32/BF16、梯度累加 fusion 开/关、512/768 长度且固定有效前缀，共 64 个 rank/config case。整数通信与伴随算子精确通过，FP32 相对独立全权重 torch 参考的最大相对 L2 差为 1.004×10⁻⁶，BF16 最大约 0.449%。见[SP 结果](../../local_artifacts/trloo_packing/causal/sp_block_summary.json)与[诊断程序](../../local_artifacts/trloo_packing/causal/check_sp_block.py)。在这个范围内未发现 SP 索引、转置通信或 norm/projection 梯度缩放错误；该检查不覆盖 GDN recurrence/conv 或 CP attention

去掉首次 BF16 TP 归约漂移后，trajectory 0 的前三层 GDN 采样激活以及第 4 层 full attention 的采样 QKV/norm/Q 位级一致，首次差异出现在该 attention core 的输出、全局 hidden row 6020，恰为短序列 CP 边界。见[逐模块激活](../../local_artifacts/trloo_packing/causal/cp_probe_vs_cp_probe_activations.json)。将两个 CP rank 的 K/V 按原 zigzag 顺序还原，三个长度下直到位置 23552 的完整 K/V 前缀、全部四份 TP head 和所选 Q 位级相同

FP64 独立计算 `softmax(QKᵀ/16)V`，覆盖 11 个 query、4 份 TP head、3 种长度共 132 个结果。相对误差最大约 0.300%；只需单个 CP 块的平均误差约 0.168%，跨两个块约 0.262%，见[参考结果](../../local_artifacts/trloo_packing/causal/cp_attention_reference.json)。这约束了所测 forward 的索引、causal mask 和缩放行为。实际 TE THD 路径以 `zeros_like(q)` 分配 BF16 output，逐次合并 BF16 FlashAttention partial output；源码中的“FP32 out”注释与该路径 dtype 不一致。尚未用实际 partial output 位级重放合并过程，因此分块舍入与 BF16 merge 的各自贡献没有单独定量

进一步在两卡上固定该层真实 Q/K/V，对 11 个 query 施加相同合成 upstream，其余 query 的 upstream 为零；分别执行 CP1/CP2 和三个真实长度。CP2 的所选 forward 输出与原 27B 探针**位级一致**。按所选 query 的合并范数统计，相对同一 FP64 causal attention 导数，三种长度下的最大 dQ/dK/dV 误差分别约 0.798%、0.349%、0.245%；单个 query 的最大 dQ 相对误差约 2.15%。短/长 CP2 对照的 dQ/dK/dV 差为约 0.885%、0.384%、0.140%；CP1 的相同有效前缀输出与 dK/dV 随长度保持一致，dQ 仅有极小尾差。所有未施加 upstream 的 query 梯度，以及 causal prefix 之后的 K/V 梯度均为零。见[真实输入局部 backward](../../local_artifacts/trloo_packing/causal/cp_backward_reference.json)、[执行程序](../../local_artifacts/trloo_packing/causal/check_cp_attention.py)和[FP64 参考](../../local_artifacts/trloo_packing/causal/analyze_cp_backward.py)

该局部控制没有复现全模型的巨大梯度误差。把 softmax backward 的 `D=Σ(dO·O)` 改用实际 BF16 O 后，FP64 导数预测更接近真实 dQ，支持保存的舍入输出会继续影响 backward；这仍是局部机制，未证明它主导全模型差异。当前最强结论是：SP primitive 的数学检查通过，SP/CP 的长度依赖舍入均已观测；固定上游系数后的模型路径变化主导本次完整梯度对照，逐层局部 VJP 控制进一步确认了反传复合传播的放大

新的 `cursor-grok-4.6-xhigh` 只读审查已给出有效 final result，认可 A/F/B/C 对照的主要因果结论，要求保留 SP/CP 局部检查的范围限制。审查误把 F/B 描述为 24/8 microbatch，已纠正为 24/24；24/8 属于 B/C。其他过强表述和采纳理由见[审查原文](../../local_artifacts/trloo_packing/causal/results_review_verdict.json)与[采纳记录](../../local_artifacts/trloo_packing/causal/results_review_adjudication.json)

逐层探针与局部 VJP 控制也完成了 Grok 只读复核，认可“本层小差异加多层传播放大”的限定结论，未发现决定性测量错误，见[最终复核](../../local_artifacts/trloo_packing/causal/local_vjp_review_verdict.json)。复核读取 CPU hook 日志时进程尚在运行，因而只确认了测试源码；随后主代理核验了实际 `replay_short_adjoints=True`、checkpoint/live 混合两层检查的退出码 0 和 PASS 输出。该检查确认先保存原始梯度再返回冻结值，且交错 microbatch 标签正确，见[完整日志](../../local_artifacts/trloo_packing/causal/local_vjp_cpu_hooks.log)与[采纳记录](../../local_artifacts/trloo_packing/causal/local_vjp_review_adjudication.json)

GPU 使用已与并行的 MTP Codex 会话直接协调，通过共享协调文件留存明确回复；每轮检查空闲进程、按约定顺序使用、只 reuse Ray，并在完成后主动通知释放。协调记录见[回复](../../local_artifacts/trloo_packing/causal/gpu_coordination/reply.txt)。局部 CP 首次直接调用 TE 时漏用了 THD 要求的 `padding_causal` 枚举，断言在 attention 执行前失败；修正并同步 hash 后重跑，上述结果仅取成功运行

复现检查：

```bash
python tests/test_trloo_trajectory_packing.py
python tests/test_trloo_trajectory_packing_gpu.py
python -m pytest -q tests/test_train_dump_utils.py \
  tests/test_predictive_mask_rollout_plumbing.py \
  tests/test_predictive_mask_arguments.py tests/test_rollout_reward_post_process.py \
  tests/test_dp_schedule.py tests/test_dppo_diagnostic_metrics.py \
  tests/deepseek-v4/test_overlong_penalty.py
```

输入 token 数收益取决于真实长度分布。若三个完整上下文分别为 24K、32K、40K，合并将 96K 输入 token 降到约 40K；端到端耗时还取决于 pipeline bubble、重计算、rollout/verifier 等，不能据此承诺相同比例的整体提速

维护源码已移除全部一次性数值探针及临时 converter 环境入口，精确诊断源码留在[本地证据](../../local_artifacts/trloo_packing/causal/diagnostic_sources/)。生产改动继续保持 packing 默认关闭，未改变 SP/CP 通信精度或底层 attention/GDN backward；此次分析没有据此宣称 71% 的梯度变化无害
