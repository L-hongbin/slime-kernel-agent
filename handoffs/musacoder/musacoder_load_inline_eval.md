# MusaCoder load_inline 评测与诊断

MusaCoder-27B 在本文记录的 A800/CUDA、单块 `load_inline` 输出、TF32-off correctness 和 `atol=rtol=1e-4` 协议下，L1/L2/L3 三轮 best-by-turn correct 分别为 96.50%、92.50%、54.50%

这些是 2026 年 6–7 月实验的结果，不表示当前服务配置。多轮 best 包含额外尝试机会，不能直接与外部单轮分数比较；官方 prompt、完整软件栈及统计口径未全部对齐，本文不将剩余分差归给某一个因素

## 协议与证据范围

- MusaCoder 原生输出是一个包含 CUDA、C++ binding、Python `ModelNew` 的完整 Python code block；三段式 `CUDA_KERNELS/APPLY_BINDINGS/MODEL_NEW` 是另一个输出协议
- correctness 阶段关闭 TF32，profile 阶段恢复该次实验原有的性能配置；两者必须分别记录，`torch.float32` 输入本身不能证明内部计算禁用了 TF32
- L1/L2 各 100 题 × 8 trajectories，L3 为 50 题 × 8；逐轮与 best 均按完整 trajectory 分母计算，自然提前结束保留已完成轮次
- 早期 `cuda_agent` client precheck 会拒绝单块输出，该组结果来自 dump 的 `load_inline` 离线重打分；后续三轮实验使用 live feedback，不能混读两类 reward
- 原始 dump 可能含 thinking 中的草稿 fence，抽取必须识别最终答案；仅取第一个 Python fence 会改变实际评分代码
- 原始论文转录保留在 [MusaCoder 论文源材料](src/MinerU_markdow.md)，它不替代本实验的配置和评分证据

## L1 精度与输出格式

早期单轮 800 样本的精度对照如下，移除未核实协议的官方分数行。`TF32 + 1e-2/1e-3` 是历史 mismatch 分桶或诊断重打分口径，不能据此推导完整性能指标

| ref | tolerance | correctness | compile | 说明 |
|---|---|---:|---:|---|
| TF32 | 1e-4 | **61.25%** | 98.12% | 旧 KernelGym/KernelBench 路径：conv reference 继承 PyTorch 默认 TF32 |
| TF32 | 1e-2 | **88.00%** | 98.12% | - |
| TF32 | 1e-3 | **87.75%** | 98.12% | - |
| FP32 | 1e-4 | **87.38%** | 98.25% | correctness 关闭 TF32，profile 前恢复默认 |

214 个 TF32-on near-miss 中，关闭 TF32 后有 208 个转为 correct，且这 208 个全部为 conv；其余 6 个来自非 conv 归约或 loss。这支持该批 conv reference 精度设置影响判分，不意味着任意容差失败都属于 harness 错误

固定 FP32 correctness 后的格式对照：

| metric | pybind 一段式 | TVM-FFI 一段式 | pybind 三段式 |
|---|---:|---:|---:|
| compile | **98.25%** | **90.38%** | **70.13%** |
| correctness | **87.38%** | **82.75%** | **59.50%** |
| problem any@8 correct | 99% | 100% | 97% |
| fast@1.0 | 13.12% | 13.50% | 11.88% |
| fast@1.2 | 9.50% | 9.75% | 8.38% |

三段式的 compile/correct 下降更大；这组结果支持保留模型熟悉的输出协议，但没有隔离每一种接口错误的因果贡献

证据：早期 pybind dump 位于 `experiments/EvalFAsync.cuda_agent.MusaCoder-27B.CTX32768/MusaCoder-27B.kb_l1_musa_coder.load_inline.cuda_agent/dumps/`，完整重打分为 `rescore_tf32off_fp32ref_800.json`；TVM-FFI 与三段式对应 sibling run 分别为 `MusaCoder-27B.kb_l1_musa_coder.tvm_ffi_inline.v2.cuda_agent`、`MusaCoder-27B.kb_l1_musa_coder.cuda_agent.fixedprompt`

## L1 shape 与 one-shot 消融

固定 `torch.rand` 分布时，新 shape 单轮 correct 为 87.38%，旧 shape 为 90.38%；旧 shape 沿用 `randn` 的对照仅 83.50%，说明分布混淆不能计为 shape 效应。旧题集取自 `21fbe5a`，与新题集有两道题的内容差异，matched-98 与 all-100 的汇总接近；缩小输入也会改变任务，不能用小输入通过修正大输入分数

大输入的 101 个失败包括 14 个 int32 索引溢出、9 个 grid 超限、49 个数值 mismatch、14 个编译失败、11 个 binding 签名错、4 个 Python runtime 错。索引与 launch 限制属于候选实现对目标输入的缺陷。旧 shape 的 77 个失败为 shape 29、数值 31、binding 7、compile 5、其他 runtime/越界 5；数值失败放宽到 1e-2 只翻正 3 个

在旧 shape 上，用复杂 one-shot 替换 elementwise add 的单次消融如下；NO-LEAK 排除各自示例对应题，其匹配 baseline 均为 91.16%

| one-shot | ALL correct（含泄漏） | NO-LEAK correct | 泄漏题自身正确率 |
|---|---|---|---|
| baseline（elementwise，不泄漏） | 90.38% | — | — |
| BatchNorm | 88.12% | 88.13% | 1/8→7/8 |
| conv_transposed | 89.50% | 89.90% | 1/8→4/8 |

| one-shot| conv (33题) | matmul (18题) | reduce/norm/loss/pool (34题) | other（激活类）(13题) |
|---|---:|---:|---:|---:|
| baseline | **97.3%** | 96.5% | **81.2%** | 100% |
| BatchNorm | 90.2% | 93.8% | **80.9%** | 100% |
| conv_transposed | **92.4%** | 96.5% | 82.7% | 100% |

| 失败类型 | baseline | BatchNorm | conv_transposed |
|---|---:|---:|---:|
| output_mismatch（数值错） | 31 | 24 | 21 |
| shape_mismatch（形状错） | 29 | 26 | 25 |
| binding_sig（wrapper 接口签名错） | 7 | **15** | **21** |
| cuda_runtime（CUDA 越界 / launch 配置） | 3 | **11** | 4 |
| compile_fail | 5 | 8 | 5 |
| py_runtime（wrapper 逻辑错） | 2 | **11** | 8 |
| **合计 fails** | **77** | **95** | **84** |

两个示例均未提高整体正确率，接口与 runtime 失败增加。只有单次生成，不能推成“复杂示例普遍有害”；从测试题取 one-shot 的行也不能作为无泄漏评测结果

证据：旧 shape 三组数据为 `Data/...-oldsize-rand`、`...-oldsize-rand-bn1shot`、`...-oldsize-rand-convT3dshot`；重打分文件分别为 `rescore_oldsize_rand_tf32off_800.json`、`rescore_bn1shot_tf32off_800.json`、`rescore_convT3d_tf32off_800.json`，独立复核记录为 `/nfs/FM/chenshuailin/staging_oneshot_conv1x1/codex_review_ablation_out.txt`

## L1 三轮结果

大 shape 使用 `mt3_largeshape_nosplit` 的 800 组数据，将 179 条受 `.so cannot open` 编译/导入 race 污染的完整 trajectory 替换为重跑结果。替换完整 trajectory 是因为错误反馈也会影响后续生成，离线修正一个分数无法消除这种影响

| 指标 | T1 | T2 | T3 | Best-by-turn |
|---|---:|---:|---:|---:|
| Compile | 97.38% | 93.38% | 94.38% | 99.12% |
| **Correct** | **86.50%** | 75.25% | 75.25% | **96.50%** |
| Fast@1.0 | 14.62% | 18.25% | 18.38% | 25.00% |
| Fast@1.2 | 11.00% | 13.75% | 13.00% | 17.12% |

best 高于 T1，T3 correct 低于 T1；增益包含多采样成分，没有单独证明 feedback 的净因果收益

| pid | 题目 | 失败数 | 主类别 / 机制 |
|---|---|---:|---|
| 73 | conv_transposed_3D（非对称） | 7 | output_mismatch |
| 100 | HingeLoss | 7 | 5 illegal_memory（int32 溢出）+ 2 output_mismatch |
| 68 | conv_transposed_3D（方形） | 5 | compile_fail |
| 45 | AvgPool2D | 2 | illegal_memory |
| 4 | Matrix-vector mult | 2 | output_mismatch |
| 97 | ScaledDotProductAttention | 2 | compile_fail |
| 89 | cumsum | 1 | output_mismatch |
| 96 | HuberLoss | 1 | output_mismatch |
| 98 | KLDivLoss | 1 | output_mismatch |

仍失败的 28 条按最后一个可编译轮次归类为数值 14、illegal memory 7、三轮均编译失败 7

旧 shape 的独立三轮实验保留为不同数据协议：

| 指标 | T1 | T2 | T3 | Best-by-turn |
|---|---:|---:|---:|---:|
| Compile | 97.88% | 94.38% | 94.75% | 99.38% |
| Correct | 88.38% | 82.38% | 80.75% | 97.62% |
| Fast@1.0 | 20.75% | 24.25% | 26.62% | 32.25% |
| Fast@1.2 | 13.50% | 16.50% | 19.12% | 22.12% |

该组以 800 trajectories 为分母，T2/T3 present=798，经过 84 条、再 11 条完整 trajectory 替换后 `.so cannot open` 为零。证据为 `mt3_full_v2`、`mt3_v2_polluted84_rerun.nokgfix.sglang8`、`mt3_v2_sglang8_polluted11_rerun.nokgfix.sglang4` 的 dump，以及 `/nfs/FM/chenshuailin/staging_oneshot_conv1x1/mt3_v2_repaired_metrics_summary.json`

## L2 三轮结果与失败归因

`mt3_level2_t600` 使用 task timeout 600s、trajectory guard 10800s，2400 条逐轮记录完整。历史 300s run 的 T1 correct 为 513/800，t600 为 544/800；这是重新生成的两次 run，不能把全部差值当成 timeout 的配对因果效应

| 指标 | T1 | T2 | T3 | best |
|---|---:|---:|---:|---:|
| correct | 68.00% | 68.75% | 67.00% | 92.50% |
| compile | 98.75% | 96.38% | 95.12% | 100.00% |
| fast@1.0 | 0.88% | 1.50% | 3.12% | 3.75% |

| T1 失败类别 | 300s run | t600 run |
|---|---:|---:|
| output_mismatch，max_diff > 1e-2 | 151 | 160 |
| near-miss，1e-4 < max_diff ≤ 1e-2 | 53 | 48 |
| shape_mismatch | 13 | 22 |
| runtime_exception | 22 | 16 |
| compile_fail | 19 | 6 |
| task_timeout | 29 | 3 |
| HTTP 400 | 0 | 1 |
| 合计 | 287 | 256 |

这些是报错分桶，`max_diff` 大小本身不能证明具体算法根因，near-miss 桶也不等价于在更宽容差下全量重跑通过。t600 的 160 个大误差中有 74 个来自 BatchNorm 题；只凭这一相关性不能把其余全部断定为同一种实现错误

### 可复核的语义案例

- BatchNorm：300s run 的 11 题中模型常读取 `running_mean/running_var`，reference 按 train-mode 使用当前 batch 统计量。将 reference 改为 eval-mode 后，88 个候选由 5 个通过变为 67 个；GroupNorm/InstanceNorm 对照保持 96/136。此实验验证语义差异，但修改了任务，不能替代原协议评分或证明官方使用 eval-mode
- LayerNorm：pid3/34 的 `nn.LayerNorm(out_channels)` 实际归一化最后一维宽度，恰好宽度等于 out_channels；候选按 channel 归一化，属于轴解释错误
- 归约顺序：pid14 先对 weight 求和再点积，改变浮点累加顺序；旧 shape 下 8/8 通过支持规模敏感性，不证明大 shape 满足原数值契约
- shape 置换：pid40/56/78/100 的 32 个冻结候选在旧小输入下 31 个通过；原输入下仍有部分本来就通过。改变 reference 的结果只能诊断输入敏感性，不能称为 31 个原始超时全部救回

旧 shape 重新生成的 800 样本 compile=771/800、correct@1e-4=578/800；由诊断桶得到的宽容差估计为 599/800。冻结候选换旧 reference 的另一组有效样本仅 760 个，排除 5 道计算逻辑有变化的题，correct@1e-4=530/760；两组分母与候选不同，不合并

eval-mode 与宽容差联合重打分只完成 64/240 后中止，因此没有可报告的完整联合分数；也没有依据将 L2 的 eval-mode 通过率迁移为 L3 修正分数

证据位于 `/nfs/FM/chenshuailin/staging_oneshot_conv1x1/`：`l2_normgroup_baseline_sanity20.json`（20/20 与 live 一致）、`l2_normgroup_evalmode_full240.json`、`rescore_evalmode_tol1e2.log`、`l2_shapetest_old.json`、`l2_full_oldshape_tol1e4.json`、`l2_full_oldshape_cases/manifest.json`、`l2_oldshape_1e2_targeted.json`、`l2_oldshape_bn_evalmode.json`（63/80，另一个子集）与 `l2_oldshape_cleanrun_meta.json`

## L3 三轮结果与评测限制

`mt3_level3_t600` 使用 task timeout 600s、trajectory guard 10800s，并修复缺少 `einops` 的环境错误。共 1195 条记录，5 个 trajectory 因 `prompt_truncated` 自然止于 T3 前；没有原始 run 那种 abort 丢失已完成轮次的问题

| 指标 | T1 | T2 | T3 | best |
|---|---:|---:|---:|---:|
| correct | 24.50% | 33.25% | 37.00% | 54.50% |
| compile | 94.25% | 91.00% | 85.00% | 99.25% |
| fast@1.0 | 0.00% | 0.00% | 0.25% | 0.25% |

T1 的 302 个失败为：数值大误差 177、runtime 64、near-miss 35、compile 14、HTTP 400 为 5、timeout 4、shape 3。含 BatchNorm 的 19 题 T1 为 0/152，best 为 91/152；非 BN 题 T1 为 98/248。分组结果说明该题族首轮困难且多轮有恢复，不能仅凭汇总证明每个恢复都来自修正 BN 语义

全轮仍有 40/1195 个 HTTP 400、14/1195 个 task timeout。其可能隐藏的正确候选未完成重打分，不能称为“所有 harness 问题已修复”，也不能把全轮比例当作 T1 分数上界

## 基础设施诊断：修复与 workaround

| 问题 | 隔离证据 | 已验证的处理及边界 |
|---|---|---|
| 3000s trajectory guard 丢失已完成轮次 | 原始 L3 的 56/400 条 trajectory 丢失前面轮次，T1 仅 344 条可见 | guard 放宽到 10800s 后该次重跑 abort=0；这是 workaround，当时 `_abort_result` 丢轮次逻辑未修复 |
| task 预算不足 | 冻结 L2 的 29 个超时候选，600s 重打分后 26 个正确；L3 的 66 个均编译通过，其中 47 正确、19 错误 | 支持预算影响判分，不支持“超时几乎全正确”；600s 也未消除全部超时 |
| reference 缺少 `einops` | 加载错误被吞成 None，下游出现 unpack-None；影响 Mamba2 题 | 安装依赖并显式报告环境错误后 unpack-None 消失；pid48/49 T1 compile=6/8、7/8，correct 仍均为 0/8 |
| 原始响应 10 万字符长度校验 | t600 被拒的 40 条里 39 条截断、11 条无代码 fence、5 条含完整可抽取代码块，类别有重叠 | 该次重跑未修改校验器，5 条完整代码正确性未知；错误提示过大与校验对象是两个问题，不能将缩短错误文本写成修复了拒绝行为 |

原始 L3 的 T1 7.75% 等数字受丢记录及环境问题污染，不作模型基线。600s 是本文已测配置，不外推成所有后端或未来模型的固定预算

超时配对证据：`/nfs/FM/chenshuailin/staging_oneshot_conv1x1/l2_timeout_retest_600s.json`、`l3_timeout_retest_600s.json`

## 耗时测量

以下来自历史 300s run。`perf_est = num_perf_trials × (reference_runtime + kernel_runtime)` 不含 warmup；原计时分析的 `compile_est = kg_kernel_total_s − correctness_trial_s − perf_est` 实际是含加载、编译和未细分工作在内的余量，不能当成纯 nvcc 计时或固定编译成本

| 指标 | L1 | L2 | L2/L1 |
|---|---:|---:|---:|
| response tokens（mean / median） | 5838 / 5751 | 8418 / 8286 | 1.44× |
| env 总时间 s（mean / median / p90，compiled 样本） | 36.4 / 13.6 / 138.0 | 116.0 / 97.7 / 188.9 | 3.2× / 7.2× |
| 未细分余量 est. s（mean / median） | 18.7 / 7.5 | 90.8 / 89.8 | 4.9× / 12× |
| 其中 correctness s（mean） | 1.15 | 1.65 | - |
| 其中 perf est. s（mean，仅 correct 样本） | 18.6 | 34.5 | 1.9× |
| env 时间合计（GPU·h，turn-1 全部） | 7.3 | 24.2 | 3.3× |
| 余量占 env 合计比例 | 51% | **78%** | - |
| 错误样本 env 均值 s | 10.8（中位 0.6，快速失败） | 88.2（中位 87.7，全额编译） | 8.2× |
| 300s 超时样本数（error_message 含 timeout） | 8 | 29 | 3.6× |

L1/L2 墙钟分别约 5h40m/20.4h，同时 engine 数为 2/1、reward worker 为 8/4、平均 response 为 5838/8418 tokens；这些共同变化可以解释瓶颈方向，不能把比例简单相乘得到端到端因果分解

原 L3 只有 344/400 条 T1 可见，env 计时覆盖 compiled 的 239 条（mean/median 93.6/86.5s），未细分余量 mean/median 87.3/85.2s。缺失和超时造成选择偏差，不据此推断完整 L3 的编译分布或固定 90s 成本

TVM-FFI 的另一组 Qwen3.6 L3 数据只有 438 个已评分轮次，其中 correct=61；task median/p90/p99/max 为 0.9/24.8/228.7/275.1s。对这些已完成样本离线截断，180s 会截掉 8 个（6 个 correct）、240s 截掉 4 个（均 correct）、300s 截掉 0 个。原运行已使用 300s timeout，未完成样本不在这张分布里，不能由此证明 300s 对全量请求无误伤

耗时字段为 `kg_kernel_total_s`、`correctness_trial_s`、`reference_runtime`、`kernel_runtime`、`num_perf_trials`、`response_length`；L1 时间统计使用未替换污染 trajectory 的原始 dump，与前文评分合并版不同

## 三轮数据入口

实验根目录为 `/nfs/FM/chenshuailin/projects/kernel_agents/slime-musacoder-mt/experiments/EvalMT3.load_inline.MusaCoder-27B.CTX40960/`，以下 run 下的 `dumps/rollout_data/eval_0.pt` 是原始记录：

- `MusaCoder-27B.mt3_largeshape_nosplit.load_inline`；正式汇总使用同目录 `eval_0_merged_clean.pt`，替换来源为 `mt3_largeshape_rerun179`
- `MusaCoder-27B.mt3_level2.load_inline`、`MusaCoder-27B.mt3_level2_t600.load_inline`
- `MusaCoder-27B.mt3_level3.load_inline`、`MusaCoder-27B.mt3_level3_t600.load_inline`

旧 shape L2 重新生成位于 sibling `EvalMT1.load_inline.MusaCoder-27B.CTX40960/MusaCoder-27B.t1_level2_oldshape.load_inline/dumps/rollout_data/eval_0.pt`。TVM-FFI 耗时来自 `slime-dev-csl-2/experiments/Eval.TVMFFI.Qwen3.6-27B.…turn3.n8/Qwen3.6-27B.level3.kg21_tf32off_correctness.20260621.093912/dumps/rollout_data/eval_0.pt`，后者保留的是历史缩写路径，复核前需在原节点定位完整 run

这些大文件可能只在原节点或 worktree 中可用；本次整理未重新执行 GPU 评测
