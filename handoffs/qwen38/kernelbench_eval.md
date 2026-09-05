# Qwen3.8 DataV2 / DataV4 / 新 reward KernelBench L1–L3 统一评测

## 结论

新 DataV4 lineage 明显强于旧版 DataV2 lineage。最可比的同 step80 结果中，逐 trajectory Correct 在 L1/L2/L3 分别从 **78.250%/38.375%/6.000%** 提升到 **87.250%/60.625%/16.750%**，即 **+9.000/+22.250/+10.750 pct-pt**。按题目配对的 20,000 次 bootstrap 95% CI 分别为 `[+4.750,+13.375]`、`[+16.375,+28.250]`、`[+5.000,+17.250]` pct-pt，三档提升都稳定为正

DataV4 内部从 step80 到 step100 又在 L1/L2 Correct 上提升 **2.875/4.125 pct-pt**，两档 paired-bootstrap CI 为正；L3 变化 **-0.250 pct-pt**，属于统计持平。在该组 16K checkpoint 中，**DataV4 step100** 的 L1/L2 Correct 最高。相较旧版当时推荐的 step40，它的 L1/L2/L3 Correct 分别高 **15.625/19.375/5.750 pct-pt**

新 lineage 的 Compile 与截断率改善比 Fast 更一致。同 step80 下，DataV4 的三档 Compile 提升 `12.750/39.750/15.500` pct-pt，截断率降低 `13.750/42.750/16.750` pct-pt；但 L1/L2 Fast@1.0 分别下降 `3.125/1.750` pct-pt。该 lineage 的正确交付率更高，性能指标没有同步提升。Fast 的小幅差异还会受不同 KernelGym 节点、reference cache 与计时波动影响

两条 lineage 都从同一 BF16 base 重新训练，核心训练配置相同，但 DataV4 是 fresh run，训练 prompt 数据和后续随机 rollout 历史同时变化。因此该对比能确认 **DataV4 lineage 的实际 checkpoint 更强**，不能把全部增益严格归因到某一类新增数据行

2026-08-27 又完成了新 reward、24K lineage 的 step80 全量评测。相对旧 DataV4 step80，它的 L1/L2/L3 Correct 差值为 `-2.38/0.00/+3.75`，Compile 差值为 `-0.62/+5.12/+15.50`，截断率差值为 `-0.75/-6.00/-45.75`；三档 Fast@1.0 差值均为负。由于训练 context、训练 temperature、partial reward、PRS/coverage-RS 和长度惩罚同时变化，这是一组配置 bundle 对照，不是单变量 reward 因果消融。旧 DataV4 step100 仍是 L1/L2 Correct 最优点；新 reward step80 则在本文对照中具有最高的 L3 Correct、Compile 与最低截断率

## 统一结果

指标均为单轮逐 trajectory 口径；比例列单位为 `%`

| lineage | checkpoint | level | n | Compile (%) | Correct (%) | Fast@1.0 (%) | Fast@1.2 (%) |
|---|---|---:|---:|---:|---:|---:|---:|
| DataV2 | step40 (`iter_0000039`) | L1 | 800 | 88.750 | 74.500 | 18.750 | 14.875 |
| DataV2 | step40 (`iter_0000039`) | L2 | 800 | 66.750 | 45.375 | 1.875 | 1.500 |
| DataV2 | step40 (`iter_0000039`) | L3 | 400 | 14.250 | 10.750 | 1.500 | 0.750 |
| DataV2 | step80 (`iter_0000079`) | L1 | 800 | 84.125 | 78.250 | 23.625 | 19.375 |
| DataV2 | step80 (`iter_0000079`) | L2 | 800 | 48.250 | 38.375 | 4.750 | 4.000 |
| DataV2 | step80 (`iter_0000079`) | L3 | 400 | 7.000 | 6.000 | 1.000 | 1.000 |
| DataV4 | step80 (`iter_0000079`) | L1 | 800 | 96.875 | **87.250** | 20.500 | 15.125 |
| DataV4 | step80 (`iter_0000079`) | L2 | 800 | 88.000 | **60.625** | 3.000 | 2.625 |
| DataV4 | step80 (`iter_0000079`) | L3 | 400 | 22.500 | **16.750** | 1.750 | 0.750 |
| DataV4 | step100 (`iter_0000099`) | L1 | 800 | 97.875 | **90.125** | 21.375 | 16.875 |
| DataV4 | step100 (`iter_0000099`) | L2 | 800 | 91.375 | **64.750** | 2.500 | 2.375 |
| DataV4 | step100 (`iter_0000099`) | L3 | 400 | 24.000 | **16.500** | 1.500 | 1.250 |

`Correct` 是数值正确且不是 decoy kernel；`Fast@x` 要求先 Correct，再达到相对 PyTorch reference 的对应 speedup

## 新 reward、24K step80

| level | Compile (%) | Correct (%) | Fast@1.0 (%) | Fast@1.2 (%) | 截断率 (%) | pass@8 (%) |
|---|---:|---:|---:|---:|---:|---:|
| L1 | 96.25 | 84.88 | 18.12 | 14.88 | 0.25 | 99.00 |
| L2 | 93.12 | 60.62 | 0.62 | 0.50 | 1.00 | 85.00 |
| L3 | 38.00 | 20.50 | 0.75 | 0.00 | 30.00 | 46.00 |

与旧 DataV4 step80 的相同步数对照如下。Correct 的 95% CI 使用 problem-paired 20,000-draw bootstrap，seed `20260827`；两次正式 eval 的题目、单轮采样、temperature 1.0、medium、tvm_ffi、no-spec 和每题 8 samples 相同，context 分别跟随 checkpoint 原生配置使用 16K 与 24K

| level | Compile 差值 (pp) | Correct 差值 (pp) | Fast@1.0 差值 (pp) | Fast@1.2 差值 (pp) | 截断率差值 (pp) | Correct 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---:|
| L1 | -0.62 | -2.38 | -2.38 | -0.25 | -0.75 | [-5.25,+0.62] |
| L2 | +5.12 | 0.00 | -2.38 | -2.12 | -6.00 | [-4.12,+4.00] |
| L3 | +15.50 | +3.75 | -1.00 | -0.75 | -45.75 | [0.00,+7.50] |

新 dump 的三档题目集合完整，每题恰好 8 个样本；独立重算与 summary 完全一致。全局没有空响应、重复响应、缺失指标、非有限 speedup、`correct && !compiled` 或两份环境状态不一致。人工检查了每档一个正确完成样本和一个失败/截断样本：成功样本都有完整 CUDA/TVM-FFI 与 `ModelNew`；L2/L3 失败样本在最终代码完成前被 context 截断，L1 失败样本已生成可编译但错误的完整实现、随后在自检修订中截断。六条样本均没有乱码或复制响应

## 新旧数据：同 step80 配对比较

| level | Compile (pp) | Correct (pp) | Fast@1.0 (pp) | Fast@1.2 (pp) | 截断率 (pp) | Correct paired-bootstrap 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---:|
| L1 | +12.750 | **+9.000** | -3.125 | -4.250 | -13.750 | **[+4.750,+13.375]** |
| L2 | +39.750 | **+22.250** | -1.750 | -1.375 | -42.750 | **[+16.375,+28.250]** |
| L3 | +15.500 | **+10.750** | +0.750 | -0.250 | -16.750 | **[+5.000,+17.250]** |

Bootstrap 以 problem 为重采样单位，每题内部保留 8 个 samples，20,000 draws，seed `20260824`；三档差值为正的 bootstrap 概率均为 `1.0`。评测 prompt、采样数和推理协议相同，因此这是本次新旧训练数据最直接的 checkpoint 对比

问题级 pass@8 也一致改善：DataV2 step80 的 L1/L2/L3 为 `98/100`、`71/100`、`9/50`，DataV4 step80 为 `100/100`、`84/100`、`14/50`

## 两条 lineage 各自的训练内变化

### DataV2 step40 → step80

| level | Compile (pp) | Correct (pp) | Fast@1.0 (pp) | Fast@1.2 (pp) | 截断率 (pp) | Correct paired-bootstrap 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---:|
| L1 | -4.625 | +3.750 | +4.875 | +4.500 | +9.375 | `[-1.375,+8.750]` |
| L2 | -18.500 | **-7.000** | +2.875 | +2.500 | +27.375 | `[-12.250,-1.750]` |
| L3 | -7.250 | **-4.750** | -0.500 | +0.250 | +9.250 | `[-8.250,-1.750]` |

旧版 step80 的 L2/L3 明确回退。它在完成样本上的条件正确率实际更高，但生成显著变长，大量响应在代码或 binding 完成前耗尽 16K context，截断样本几乎全部失败；这使总体 Compile/Correct 下滑。旧版当时因此推荐 step40

### DataV4 step80 → step100

| level | Compile (pp) | Correct (pp) | Fast@1.0 (pp) | Fast@1.2 (pp) | 截断率 (pp) | Correct paired-bootstrap 95% CI (pp) |
|---|---:|---:|---:|---:|---:|---:|
| L1 | +1.000 | **+2.875** | +0.875 | +1.750 | -0.375 | **[+0.250,+5.750]** |
| L2 | +3.375 | **+4.125** | -0.500 | -0.250 | -1.875 | **[+0.375,+7.875]** |
| L3 | +1.500 | -0.250 | -0.250 | +0.500 | -2.750 | `[-2.250,+1.750]` |

同样使用 problem-paired 20,000-draw bootstrap；差值为正的概率为 L1 `0.98030`、L2 `0.98315`、L3 `0.35870`。DataV4 到 step100 没有复现旧版 step80 的 L2/L3 correctness 崩落

## 截断与输出质量

| lineage / checkpoint | L1 截断率 (%) | L2 截断率 (%) | L3 截断率 (%) | L1/L2/L3 response length median (tokens) |
|---|---:|---:|---:|---:|
| DataV2 step40 | 5.375 | 22.375 | 83.250 | 5845.5 / 10978.5 / 14329 |
| DataV2 step80 | 14.750 | 49.750 | 92.500 | 8496 / 14796 / 14379 |
| DataV4 step80 | 1.000 | 7.000 | 75.750 | 5444.5 / 9931 / 14306 |
| DataV4 step100 | 0.625 | 5.125 | 73.000 | 5168 / 9329.5 / 14254 |
| DataV4 新 reward step80 | 0.250 | 1.000 | 30.000 | 6530.5 / 11426 / 19107 |

旧 DataV4 16K lineage 几乎消除了 L1 截断并大幅缓解 L2，但 L3 仍有约四分之三响应触及预算。新 reward 24K step80 把 L3 截断率降到 30.00%，仍明显高于 L1/L2，长组合任务的收尾依然是主要瓶颈。SGLang context 包含 prompt，因此有效生成预算小于配置 cap，截断响应本身不必达到 16K 或 24K tokens

前四个 dump 合计恰好 8,000 trajectories，新 reward step80 另有 2,000 trajectories；每个 dump 中各 level 的每个 problem id 都恰好出现 8 次。独立审计没有发现 missing metrics、`correct && !compiled`、非有限 speedup、env state 不一致、空响应、Unicode replacement、非法控制字符或 dump 内重复响应。DataV2 step40 有一个候选 kernel 触发 300 秒执行超时，其余错误均来自候选代码本身而非评测服务断连

每个 checkpoint、每个 level 都人工检查了一条正确完成样本和一条失败截断样本，前四个 checkpoint 共 24 条，新 reward step80 再检查 6 条。它们都是连贯的英文算子分析和 CUDA/TVM-FFI 实现；成功样本结构完整，失败样本主要停在推演、CUDA 源码或 binding 中间，没有随机多语种词流或乱码

## 统一评测协议与 lineage

- canonical validation：L1 100 题、L2 100 题、L3 50 题，每题 8 samples，即 `800/800/400` trajectories/checkpoint
- `tvm_ffi`、`max_turns=1`、temperature `1.0`、top-p `1.0`、top-k `-1`、medium reasoning、context/response `16384`
- BF16 HF serving、no-spec、FA3 attention、Triton linear/GDN、带 padding CUDA graph；没有使用 eager
- 新 reward step80 使用 context/response cap `24576`。SGLang context 是 prompt 与 response 的总长度，因此单条样本的实际 response budget 是 `24576 - prompt_tokens`，不是额外再生成完整 24K；其余四个 checkpoint 同理使用 `16384 - prompt_tokens`
- L1/L2/L3 eval 数据 SHA256：`e034d42fe5e8ed0fac0e580bb9070379f719080b05d666accbac59abc081435f`、`11c1858d88be14ebc7fa766390f46a0db1ad872e921ddc61411e7df4d2e64cfe`、`6b3c85f2e57f307036b8c38ff26891e8b88ac44afb702a7c7f539d2b8307755e`
- DataV2 lineage：`FAsync.NoSpec.DefaultCG.DPPOPredictive.LongestFirst.matched.medium.Temp1.1.tvm_ffi.Qwen3.8-27B.BF16Train.FP8Rollout.CTX16384`；训练数据为 `Data/prompt_tvm_v2/drkernel_rl_thinking.parquet`（71,996 rows，SHA256 `17b948be017e8e57e6e87dcae15eca3acd5b9feab141be49dd4589e74ab14d25`）
- DataV4 lineage：`FAsync.NoSpec.DefaultCG.DPPOPredictive.LongestFirst.matched.medium.Temp1.1.DataV4.tvm_ffi.Qwen3.8-27B.BF16Train.FP8Rollout.CTX16384`，从 BF16 base iteration 0 fresh start，没有加载 DataV2 checkpoint；训练数据为 `Data/prompt_tvm_v4/release/train.parquet`（39,636 rows，SHA256 `189da56dca3acbb03b5532b360ec1eb9adde75c173952149a8515dcebfb08f79`）
- 新 reward lineage：`FAsync.NoSpec.DefaultCG.DPPOPredictive.LongestFirst.matched.medium.Temp1.0.Mismatch0p25.NoPRS.Len4096Pen0p2.DataV4.tvm_ffi.Qwen3.8-27B.BF16Train.FP8Rollout.CTX24576`；同样从 BF16 base iteration 0 fresh start，使用相同 DataV4 release，增加严格 output-mismatch partial reward `0.25`、关闭 PRS/coverage-RS，并使用最后 4096 response budget 最大 `0.2` 的线性长度惩罚

DataV2 step40 来自 node69/node70 归档 `/nfs/LOCAL/chenshuailin/checkpoints/qwen38_temp11_failed_enospc_20260823/iter_0000039`，DataV2 step80 来自旧实验的 `checkpoints/iter_0000079`。DataV4 step80 来自 node70 已验证的两机归档 `/nfs/LOCAL/chenshuailin/checkpoints/qwen38_datav4_rolling_20260823/iter_0000079`，step100 来自 DataV4 实验的两机完整 `iter_0000099`。DataV4 两个 checkpoint gather 后均为 32 个 shard、`489,654,266,558` shard bytes；转换后均为 851 tensors、11 个 safetensors、`53,792,108,344` weight bytes，HF 位于 `/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.8-27B-RL-Temp1.1-DataV4/step{80,100}`。源 Megatron checkpoint 未删除

新 reward step80 来自 node70 的 rolling archive `/nfs/LOCAL/chenshuailin/checkpoints/qwen38_datav4_mismatch0p25_ctx24576_rolling_20260825/iter_0000079`。归档 manifest 已验证，包含 node70/node69 两端共 32 个 shard；rolling keep1 已在发布 step80 后删除 step60 和两端 source half。转换得到 851 tensors、11 个 safetensors，部署 HF 位于 `/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.8-27B-RL-DataV4-Mismatch0p25-CTX24576/step80`

## 证据与产物

所有 `experiments` 路径都位于运行 snapshot `/nfs/FM/chenshuailin/projects/kernel_agents/slime-dev-csl-2-qwen38-rl-20260819`：

- DataV2 step40（node53）：`experiments/EvalFAsync.NoSpec.tvm_ffi.Qwen3.8-27B.CTX16384/step40/summary.20260823.052114.txt`，summary SHA256 `5061574f0c0dc320ff24195156c535a55b04b9e25afdfd8add25a7b46dbfeb27`；dump SHA256 `5fed1617620dfd28d0696ec5aeed265a81bd80d0fd07b5e6003acf57fe905ca0`
- DataV2 step80（node64）：`experiments/EvalFAsync.NoSpec.tvm_ffi.Qwen3.8-27B.CTX16384/step80/summary.20260823.053359.txt`，summary SHA256 `d5f17b7052d8c350678d4e096e1df3e590696062cdbe514ab3931f0662b673b2`；dump SHA256 `3e476a3f6bba6a2b3e037dbbec057185df85229278f7012aef647875ea3d55c7`
- DataV4 step80（node53，Ray job `raysubmit_26TLAYqePXX1Z3K3`）：`experiments/EvalFAsync.NoSpec.tvm_ffi.Qwen3.8-27B.CTX16384/datav4_step80/summary.20260824.061105.txt`，summary SHA256 `29fe60fec7631e8363fc3d771e2ec7852f7ccfcaacc92e0baba44d0abd094b60`；dump `317,624,807` bytes，SHA256 `fc096ea6716baf923cf50fb9535016582a9e6d181e332ef8f1d46e6158873846`；audit SHA256 `5246d4b0031eb37333c748ba7739c4de44fa3151bf4c72294f3585f4798643a6`
- DataV4 step100（node69，Ray job `raysubmit_JeUgSQZNwiUYcZBv`）：`experiments/EvalFAsync.NoSpec.tvm_ffi.Qwen3.8-27B.CTX16384/datav4_step100/summary.20260824.062754.txt`，summary SHA256 `57e7d812b45c271450be8e49ccf69e13f59bf34de68b00fb7c00015bf031d702`；dump `307,024,999` bytes，SHA256 `9b5644acd1749d80fde274d9d6fee6836128dd344ab341b6ec8ab0f16475f1b0`；audit SHA256 `16f6c53be2665634ad88cae90c6a08b4ecfa6010e6fc947aa722439f3a8e864f`
- 新 reward step80（node69，Ray job `raysubmit_GndRsHUTkJweCZSk`）：`experiments/EvalFAsync.NoSpec.tvm_ffi.Qwen3.8-27B.CTX24576/newreward_datav4_step80_ctx24576/summary.20260827.121940.txt`，summary SHA256 `26fd118c1ec754d65031185b17d43ffbf3ceb65bd5bf3fd9e1c9de0e574e1afe`；dump `391,431,335` bytes，SHA256 `f2913fc1b27f27e85bf544de5e1f1b15109146f842c0b8ab9cccf0ffc645fbcd`；audit SHA256 `ebd16cf1d79ae1858db759dd835b37f077502ad2e523bdbae89c7680896f9d02`；人工样本 SHA256 `769652018f2f38daa8a9dd8e7d6258c7f2d298ddbc0e332a77ddd02add36b1ff`
- 新旧 reward/config step80 配对比较：`local_artifacts/qwen38/newreward_step80_eval_20260827/oldreward16k_vs_newreward24k_step80_compare.20260827.json`，SHA256 `9a1c287390617aeae946a6ed2124f1774883ee2cd02311c23c812f5d34e1364b`
- DataV4 step80→100 比较报告 SHA256：`79ed80582d0cd27722aadae6930a359de96b890650f738b35c11614bc9487d2a`
- DataV2 step80→DataV4 step80 比较证据：`local_artifacts/qwen38/eval_step40_80/legacy_step80_vs_datav4_step80_correct_bootstrap_20260824.json`，SHA256 `7a137cb4dd6fc72d77be0dff11b8cc5b0434443492efc53d8252de7e84e1c19b`
- 独立审计脚本：`local_artifacts/qwen38/eval_step40_80/audit_l123_dump.py`、`compare_l123_audits.py`

五个正式 Ray eval 都成功结束，评测后的 serving/Ray 进程和 GPU 占用已释放，KernelGym 健康且队列为空。DataV4 step100 与新 reward step80 原计划使用 node64，但该机被另一任务占满 8 张 H20；未终止对方任务，按用户要求改到同型号 node69，推理协议与 A800 KernelGym 计分池不变

新 reward step80 的第一次启动在任何样本生成前因 Ray 临时目录过长触发 AF_UNIX 路径限制。将该次 eval 专用的 `RAY_TEMP_DIR` 改为 `/nfs/LOCAL/rq38e80` 后重新启动并完整通过；这只改变 Ray 本地运行目录，没有改变模型、数据、采样或计分协议

DataV4 部署期间还发现 `gather_convert_deploy.sh` 通过 stdin 远程执行时，嵌套 ssh 会吞掉后续脚本文本，导致首次向 node53 同步为空目录。已给嵌套 ssh 增加 `-n` 并重新部署，文件清单、大小和转换结果全部通过核对，没有删除或覆盖源 checkpoint
