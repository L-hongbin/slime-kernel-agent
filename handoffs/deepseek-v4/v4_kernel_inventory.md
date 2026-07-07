# DeepSeek-V4-Flash — Megatron 缺失算子盘点 + tilelang 核实现结果

> V4-Flash LoRA on slime/Megatron 的 Phase-1 准备工作。真值来源：
> `transformers/models/deepseek_v4/modeling_deepseek_v4.py` (5.8.1) + `configuration_deepseek_v4.py`。
> 硬件：H20（sm90 Hopper，97GB，8 卡）。tilelang 0.1.8 / torch 2.11+cu129；
> node64 driver 570/CUDA 12.8，node69 driver 595/CUDA 13.2，node70 driver 570/CUDA 12.9。
> 配套：[[handoff_deepseek_v4_flash_lora_megatron]]。
> 各核细节 + 全部数据见 `custom_kernels/deepseek_v4/{attention,mhc,compression}/RESULTS.md`。

## 进度总览（2026-06-30；R6 补记 2026-07-03）

**R6 补记（2026-07-03）**：kernel-on（TileLang A1/B1/B2）R6 全链路 smoke 已 **PASS**
（rollout→权重同步→PP2/EP8 Muon 训练一步，on-policy temperature=1），见
[[handoff_deepseek_v4_flash_lora_megatron]]。下方 R4 时期"先 kernel 压力复现、
不能把 rollout 当下一步生产门"的判断已被超越；当时的 full-loop 失败根因是
`--rollout-temperature 0` 在训练侧 log-prob 除零（非 kernel 问题）。node70 的
TileLang sm90a 加载问题仍然存在（node70 仍不参与 kernel-on 训练）。另：训练侧
mHC 现默认 `V4_MHC_MIXING_ORACLE=1`（DeepSeek 配方：Sinkhorn 只做前向 mixing
oracle，反向 detach）；TileLang `HyperConnectionFn.backward` 的 oracle 模式
（跳过 Sinkhorn VJP 计算）仍是待做优化。

第一步（用 tilelang 实现 Megatron 缺失的 V4 专属算子）的核已**落地 + 正确性验证 + codex 评审通过**；新效率口径下的三处欠账（A1 前向、B1/B2 反向）**已做完一轮优化并 orchestrator GPU 复验**（见下，2026-06-30）。

**R4 kernel repair note（2026-07-02）**：full actor smoke 仍默认走 torch fallback，
不能把 rollout 作为下一步生产门。已先修复 B1 mHC 的确定性 stride bug：`N=1` 时
`raw_p[:, :MIX].contiguous()` 会保留物理 stride `(32,1)`，而 TileLang
`_build_sinkhorn_collapse` 需要 `Raw[N,24]` 的 stride `(24,1)`；`kernel.py::_compact_last_dim`
改为 fresh packed copy。正确性日志 `handoffs/deepseek-v4/r2_logs/r4_mhc_stride_fix_correctness.log`
为 `OVERALL: ALL PASS`；按 kernel 修改要求复测了 forward/backward 效率，日志
`handoffs/deepseek-v4/r2_logs/r4_mhc_stride_fix_bench.log`，`B=1,S=1` GPU repro
见 `handoffs/deepseek-v4/r2_logs/r4_mhc_stride_fix_s1_repro.log`，详表见
`custom_kernels/deepseek_v4/mhc/RESULTS.md`。单进程 smoke-shape battery
`handoffs/deepseek-v4/r2_logs/r4_kernel_singleproc_shape_battery.log` 覆盖 mHC S64、
CSA S64、HCA S128、attention sliding/CSA S64 的 forward/backward 全 PASS。这不等于
attempt6/7/8 的多进程/训练-shape SIGILL 已经全部根因关闭；下一步仍是 kernel 压力复现，
而不是 rollout。

**R4 node70 gate（2026-07-02）**：`PP3_EP8` kernel-on debug-train-only attempt1 在
node70 PP stage2 失败，栈落在 mHC `_build_norm_gemm` 的 `cuModuleLoadData`
(`handoffs/deepseek-v4/r2_logs/r4_pp3_ep8_kernel_on_train_smoke_attempt1.log`)。
最小复现确认不是 Ray 问题：node70 单进程 mHC norm_gemm
`r4_node70_mhc_norm_single.log` segfault，sglang `mhc_pre_gemm_sqrsum_tilelang`
pre-only `r4_node70_sglang_pre_only.log` SIGILL，torch norm 后的
`_build_sinkhorn_collapse` `r4_node70_mhc_torch_norm_hybrid2.log` SIGILL，forward
全 torch 后的 mHC backward d_pre `r4_node70_mhc_torch_norm_middle_tilelang_bwd.log`
segfault。node64/node69 同测 PASS；node62 因 `PATH` 未含 `/usr/local/cuda/bin`
曾误用 pip nvcc 缺 `cuda/atomic`，加 PATH 后 `r4_node62_mhc_norm_single_pathfix.log`
PASS。结论：node70 当前不能用于 kernel-on 训练或 sglang V4 rollout；若要继续保留
node62 rollout，下一训练 fallback 是 node64+node69 `PP2_EP8`。

**R4 PP2 fallback（2026-07-02）**：已转换并验证 `PP2_EP8` torch_dist：
PP2/EP8 转换 verify 为 16/16 ranks PASS（evidence 文件已在 2026-07-03 清理中删除；可用 `scripts/v4/convert_torch_dist.sh` 链式 verify 重新生成），missing/unexpected
为空，代表性 direct/expert diff 全 0。kernel-on debug-train-only smoke 在 node64/69
通过，日志 `handoffs/deepseek-v4/r2_logs/r4_pp2_ep8_kernel_on_train_smoke_attempt2.log`：
checkpoint load 成功，V4 kernels enabled，`actor_train end (259.4s)`，
`train/loss=12.9600830078125`，`train/grad_norm=2.0647118008469465`，Ray job
succeeded。attempt1 的 AF_UNIX path-too-long 是 Ray temp 路径问题；attempt2 用短
`/tmp/v4p2a2` 规避。因 node64 `/nfs/FM` 余量紧，本次 `SAVE_MODEL=0`，未保存新的
全量 smoke checkpoint。下一步可进入 rollout 前，还要独立确认 node62 sglang V4
kernel/serve 与 routing-replay 数据面；node70 仍不参与 kernel-on。

**评测口径（三个独立指标，不混用）**：① **正确性**：fwd+bwd 都 vs **torch reference**（fp64 gradcheck + fp32/bf16）；② **forward 效率**：只 vs **sglang**（A1→FlashMLA、B1→`mhc_pre`、B2→`compress_forward`）——`torch.compile` 仅作背景参考，**不作效率标尺**；③ **backward 效率**：只 vs **本核 forward**（bwd/fwd 延迟比 + bwd%峰值 vs fwd%峰值）——sglang 推理核无反向。**健康基线见下节实测：bwd/fwd ≈ 2.0–2.5×，且 fwd 效率应 ≥ bwd 效率。**

| 核 | 正确性（vs torch） | forward 效率（仅 vs sglang） | backward 效率（仅 vs forward；基线 2.0–2.5×） | 状态 |
|---|---|---|---|---|
| **A1** 注意力（hd512+sink+MQA） | fwd+bwd PASS，fp64 gradcheck 精确（0 误差）；ws 前向 Out/Lse 与基线**位级一致** | **warp-spec(split-D)：0.50×→0.65× FlashMLA**（dense %峰值 40.7→**52.6%**，1.30× 提速）；**fwd 53%峰 > bwd 42%峰 → 倒挂已纠正**；sparse CSA/HCA 1.10–1.12× | bwd/fwd **2.67（m4）/2.53（m128）**（dKV 拆 raw/comp + 紧 q_lo，≈基线） | ✅ 前向 warp-spec 已落地（0.65×，逼近 ~0.7 结构上限）+ 反向达标；无死锁，已 GPU 复验 |
| **B1** mHC（Sinkhorn） | fp64 gradcheck 精确（0 误差），fwd/bwd PASS | `hyper_connection_sglang`：**1.00–1.07× sglang `mhc_pre`**（已追平） | bwd/fwd **12–31× → 5.6–5.8×**（LoRA 4.7–4.9×）：前向存 `raw/inv`、反向不再重算 norm+GEMM；比值仍高于基线是因**前向 SOTA-fast 撑大比值**，反向绝对耗时已合理 | ✅ 反向已修；已 GPU 复验 |
| **B2** 压缩器（CSA/HCA） | fwd+bwd PASS（fp32 ~1e-7、bf16 容差内） | **CSA 1.8×、HCA 持平 0.97×**（vs sglang `compress_forward`，手写 CUDA，pool-only） | bwd/fwd **~5× → 1.39×（CSA32k）/ 2.70×（CSA8k）/ 2.70–2.89×（HCA）**：根因是两处 **torch 侧归约**（dpos_bias 全量上转、dweight 重算），折进 reduction / 核吐 per-row partial 解决——**回到健康区间** | ✅ 反向达标；已 GPU 复验 |

## bwd/fwd 效率基线（Megatron/标准算子实测，H20 bf16）

为校准"反向效率是否合理"，实测了几个标准算子的 bwd/fwd（fwd-only no_grad vs fwd+bwd 差值）：

| 算子 | fwd | bwd | bwd/fwd |
|---|---|---|---|
| Linear 8192×4096×4096 | 2.22ms | 5.25ms | **2.36×** |
| flash_attn S8192 H32 D128 causal | 7.25ms | 18.29ms | **2.52×** |
| RMSNorm 8192×4096 | 0.77ms | 1.88ms | **2.46×** |
| SwiGLU MLP 4096→14336 | 23.4ms | 47.2ms | **2.02×** |

**结论：健康 bwd/fwd ≈ 2.0–2.5×；且这些核 fwd 效率 ≥ bwd 效率。** 据此：A1 延迟比正常但 fwd<bwd 效率倒挂（前向欠优化）；B1 的 7–30× 与 B2 的 ~5× 都明显超标，反向有真实浪费。`bwd/fwd 基线脚本：scratchpad/bwdfwd.py`。

**fp8**：sglang 这三个算子前向**全部 bf16**（FA4 attention 显式 `NotImplementedError` 拒绝 descale；`compress_forward` 无 fp8 入参；mhc 断言 bf16+fp32，`FP8` 常量未用）。V4 的 fp8/fp4 只在 **MoE experts（block-fp8，复用 Megatron TE）** 和 **indexer（fp4，已丢弃）**——**三个核都不需要 fp8 前向**。

## V4-Flash 固定维度（决定核 shape）

| 字段 | 值 | 说明 |
|---|---|---|
| hidden_size | 4096 | |
| num_attention_heads | 64 | |
| num_key_value_heads | 1 | **shared-KV MQA**（K 和 V 是同一张量） |
| head_dim | **512** | **>256 → FA2/FA3/FA4/SDPA/FlexAttn 全部用不了** |
| qk_rope_head_dim | 64 | partial rotary 64/512；RoPE 作用在尾部 64，交错式 |
| q_lora_rank | 1024 | |
| sliding_window | 128 | 每个 attn 层都有滑窗分支 |
| o_groups / o_lora_rank | 8 / 1024 | grouped output projection |
| compress_rate CSA / HCA | 4 / 128 | 窗口大小 |
| hc_mult / sinkhorn_iters / eps | 4 / 20 / 1e-6 | mHC；fp32 |
| index_n_heads / head_dim / topk | 64 / 128 / 512 | indexer —— **32k 下丢弃**（dense-over-compressed） |
| n_routed_experts / per_tok / shared | 256 / 6 / 1 | MoE |
| num_hidden_layers | 43 | attn: 2×HCA 后 CSA/HCA 交替；mlp: 3×hash_moe 后 moe |

## 决定性事实

`head_dim=512` **结构性地挡死所有现成融合注意力核**（FA2/3/4 卡 256；SDPA 无 per-head sink；FlexAttn 无法 resize KV 以容纳块内 compressor 拼接）。V4 官方只能 **eager**。32k 下 eager 每层物化 `[64, ~40k, ~40k]` 分数 —— 显存和速度都不可行。**自研融合 flash attention 是价值最高、也是全项目最长的那根杆。**

## 算子分类

### A —— 必须自研 tilelang 核（无库核可用）

**A1. V4 融合 flash attention**（`DeepseekV4Attention` 核心，modeling:705/743）—— fwd + 手写 bwd(dq,dk,dv)。
- MQA shared-KV（1 个 KV 头广播到 64）、`head_dim=512`、bf16 计算 / fp32 累加。
- per-head **attention sink**（gpt-oss 风格：softmax 分母多一列，无 value 贡献）。
- KV 轴 = `[原始 token（滑窗因果 W=128）++ 压缩条目（因果阈值 (w+1)*m<=qpos+1）]` 拼接；一个"MQA + sink + 加性 bias flash attention, hd=512"通用核服务全部 43 层 / 3 种层型。
- RoPE 在核**外**（核拿 post-RoPE 张量）；输出端共轭 `-sin` 也在外。

### B —— 自研 tilelang 候选

**B1. mHC HyperConnection**（modeling:864 / HyperHead:943）—— fwd + bwd。每 forward 跑 **2×/层 ×43 = 86 次**，fp32。核心是 **Sinkhorn-Knopp 20 迭代**（`[B,S,4,4]` 上的行/列归一）+ collapse。

**B2. 窗口 gated-softmax 压缩 pool**（`(kv*gate.softmax(dim=window)).sum(dim=window)` + RMSNorm；CSA/HCA compressor 核心，modeling:418/534/661）—— fwd + bwd。

### C —— 直接用 torch / 复用 Megatron（不自研，自研会浪费算力）

| 算子 | modeling | 原因 |
|---|---|---|
| 交错 partial RoPE | 335/342 | 廉价 elementwise；torch.compile 足够 |
| GroupedLinear `o_a_proj` | 303 | 块对角 = batched GEMM；cuBLAS/`torch.bmm` 最优 |
| Unweighted RMSNorm | 66 | trivial |
| HashRouter `tid2eid` 查表 | 1040 | 冻结 gather，trivial |
| TopKRouter（sqrtsoftplus + e_score_correction_bias） | 1019 | 复用 Megatron router，加 sqrtsoftplus |
| Experts（256，grouped） | 978 | Megatron GroupedGEMM MoE，FP8，**冻结** —— 复用 |
| Attention sink | 721 | 折进 A1 |
| **Indexer top-k** | 448 | **32k 丢弃**（dense-over-compressed）—— 不实现 |

---

## 实现结果（detail）

### A1 注意力 —— `custom_kernels/deepseek_v4/attention/`
- **正确性**：`test_correctness.py` OVERALL PASS。fp64 gradcheck 通过；bf16 核 vs fp32 reference 在紧容差内（fwd ~2.4–3.2e-3，bwd dq/dk_raw/dk_comp/dsink ~2–4.8e-3），覆盖 S∈{512,2048,8192}、CSA m=4 + HCA m=128、滑窗-only + 带压缩、W∈{128,64}、B∈{1,2}。**S8192 带压缩时 dense fp32 reference OOM，本核 fwd+bwd 正常**。
- **vs torch.compile**（naive dense reference）：fwd+bwd **6.5–36.8×**，且 16k/32k reference OOM 而本核能跑。
- **vs 社区 SOTA（前向，三个独立基线一致）**：
  - **FlashMLA `flash_mla_sparse_fwd`（sglang V4 backend 真正用的核，手写 CUDA/CUTLASS）**：dense-causal 配置下我们 **~0.50×**（~60 vs ~120 TF/s）。
  - FA4 @hd256 **0.49×**；FlashInfer-MLA @hd512 **0.48–0.51×**。
  - **重要 caveat**：FlashMLA 是 **MLA 矩阵吸收（latent 空间，算法上更省）+ 稀疏 top-k + 推理-only（无反向）**；我们是 dense-over-compressed + dense hd512 + sink + 训练 fwd+bwd。所以这个比值偏向 FlashMLA，是"硬件利用率天花板"而非同操作对比。前向数都是 forward-only useful-matmul TF/s（codex 复核过）。
- **反向已优化（本轮重点，bwd 效率 vs fwd 效率）**：把 dKV 拆成 `_build_bwd_dkv_raw` + `_build_bwd_dkv_comp`，压缩核给紧的 `q_lo` 下界（跳过全 mask 的 query block）。codex(xhigh) 评审：数学正确、无重复计数。
  - dKV S8192 m4：72.9 → **40.7 ms（1.79×）**。
  - **bwd/fwd 延迟比：3.9 → 2.67（m4）/ 2.9 → 2.53（m128）** —— 达到 flash 风格 ~2.5× 理想值，低于 3.5× 的 FLOP 量级下限。
  - **硬件效率**：dKV %峰值 22.9→**41.1%**；forward 32%、dQ 43.5%（未动）。**bwd效率/fwd效率 = 1.3–1.4×：反向现在比前向更高效，前向成了瓶颈。**
- **ncu 定位**：三个核 **occupancy 卡 1 CTA/SM**（196KB 动态 smem + 240 寄存器/线程，实测 occupancy ~11.6%）；**非显存/原子瓶颈**（DRAM 3–11%，L2 21–34%）。
- **据 ncu 做的取舍（已记录）**：head-group 原子累加**不做**（原子非瓶颈、还降并行度）；5-GEMM 融合**不做**（会给 dQ 引入 64 头原子，得不偿失，FA2 式独立 dQ 核零原子更优）；前向 autotune **无普适增益**（block_N=128 只在最稠密 shape +8%，其余 −6~22%）。
- **前向 0.5× 的根因 + 后续**：天花板 = occupancy 1 CTA/SM + WGMMA 与 softmax 无法重叠 → 唯一出路是 **warp 专门化**（Phase-2，参考 FlashInfer-MLA 的 12-warp 调度：TMA-load / MMA / softmax / correction 分角色）。调参（threads=256 破 layout、stages=2 无效）已证无效。

### B1 mHC —— `custom_kernels/deepseek_v4/mhc/`
- **hybrid（sglang-free fallback, fp32/bf16）**：tilelang `_build_norm_gemm`
  保存 `raw/inv`（RMSNorm rescale 是 per-row scalar，`raw=inv·(xf@fnᵀ)`，
  不物化 512MB flat 张量）+ fused tilelang `_build_sinkhorn_collapse`
  做 gates / 20-iter Sinkhorn / collapse；手写**解析反向**（每步归一化的
  VJP，非 autograd-through-20-iters，fp64 gradcheck **0 误差**）。vs
  torch.compile：前向胜、fwd+bwd 近似持平、显存 −19–27%。
- **fused Sinkhorn 当前状态**：早期默认 `tilelang.compile` 的 warp-specialization
  曾把 `thread<32` producer/consumer split 编成跨 warp barrier hang；当前实现按
  sglang 的同构 big-fuse 配置编译：
  `TL_DISABLE_WARP_SPECIALIZED=True` + `TL_DISABLE_TMA_LOWER=True`，已 GPU 复验不挂。
  `MHC_TORCH_MIDDLE=1` 仍可回退 torch middle。
- **default trainable path（bf16）**：`hyper_connection_sglang` 现在不是整段调用
  `sglang mhc_pre` 再在 backward 重算 GEMM；它只复用 sglang 的稳定
  `mhc_pre_gemm_sqrsum_tilelang` 得到 `gemm_out[N,24]` 和 `sqrsum[N]`，本地形成
  `raw=inv·gemm_out`，再走同一个 fused Sinkhorn+collapse 并保存 `raw/inv/pre`。
  因此 backward 不再重算 norm+GEMM。
- **效率结果**：`bench_options.py` 中默认 (c) 路径在 S2048/S8192/S16384
  total fwd+bwd 为 **2.27/3.15/5.72 ms**，比旧 sglang-recompute 设计快约
  14–15%（长序列），LoRA frozen-param 路径还会跳过 `d_fn` GEMM。R4 stride
  修复后 `bench.py` 复测见 `custom_kernels/deepseek_v4/mhc/RESULTS.md`：S2048/S8192/S16384
  tilelang fwd+bwd=1.469/3.654/6.741ms，峰值显存仍低于 torch.compile。
- **数值/边界**：`test_correctness.py` 当前 `OVERALL: ALL PASS`，含
  sglang-GEMM path + our backward。`hyper_connection_sglang` 约束仍是 bf16
  residual + fp32 `fn`；fp32/fp64 用 hybrid。R4 另修 `N=1` raw stride bug：
  `r4_mhc_stride_fix_s1_repro.log` 证明 single-token GPU forward/backward PASS。

### B2 压缩器 —— `custom_kernels/deepseek_v4/compression/`
- **正确性**：HCA（W=128）+ CSA（overlap ratio=4）fwd+bwd ALL PASS（fp32 ~1e-7，bf16 容差内）。
- **vs torch.compile**：CSA 融合核（in-kernel 折叠 Ca/Cb gather + position_bias，省掉 `new_kv/new_gate` 物化、~2× DRAM）**1.3–2.0×**、显存近半；HCA fwd+bwd 1.17–1.24×。codex 评审：无 bug，CSA 融合是它建议并实现的。
- **vs 社区 SOTA**：sglang `dsv4.compress_forward`（**手写 CUDA**，`.cuh` nvcc JIT，pool-only）。前向：CSA **1.8×**；HCA 起初 2.6×（外面 torch 物化 gate），套用同样的 in-kernel-gather 融合后 **32k 持平 0.97×**。注意我们多算了 RMSNorm（sglang pool-only），能跟手写 CUDA 这么近已不错。

## sglang V4 核用什么语言（对比参照）
- **主注意力 FlashMLA**（`flash_mla_sparse_fwd`/`_with_kvcache`）= **手写 CUDA/CUTLASS**（预编译 `flashmla_ops.abi3.so`）。
- **压缩器 / fused norm+rope / topk / paged_mqa** = **手写 CUDA**（`jit_kernel/csrc/deepseek_v4/*.cuh`，nvcc JIT）。
- **mHC / indexer fp8 logits** = **TileLang**（和我们同语言）。
- 大量后端/变体/辅助 = **Triton**。所有 GEMM（MoE、mHC prenorm）= **DeepGEMM**。
- 含义：A1 vs FlashMLA 是 **TileLang vs 手写 CUDA+吸收+稀疏**（0.5× 体面）；B2 vs 手写 CUDA（追平/接近，不错）；B1 vs **TileLang（同语言）**——所以直接用它的前向最划算（已采纳）。

## "用 sglang 前向 + 自写反向"策略适用性
- **B1：✅ 已落地**（同语言、同数学、sglang 前向就是我们想要的快版；反向我们已有）。
- **A1：❌ 不适用** —— sglang 没有算我们训练那个函数的前向（FA4 卡 hd256；FlashMLA 是吸收+稀疏 = 另一个函数，且给它写反向更难）。继续优化自研核。
- **B2：⚠️ 不值** —— 已达/超 SOTA 且反向已有，接 sglang(推理布局)摩擦 > 收益。

## 测试/benchmark 契约（每个核都满足）—— 三个独立指标，不混用
1. **正确性（vs torch）**：前向 + 反向都对 HF eager reference 验。fp64 `gradcheck` 验反向公式（有限差分需 double，避免抵消误差，证明解析 VJP 数学精确）；fp32/bf16 kernel vs reference 验核实现（注意：gradcheck 测公式、不测核，核必须另用 fp32/bf16 vs reference 验——B1 死锁核曾"骗过"gradcheck）。
2. **forward 效率（仅 vs sglang）**：A1→FlashMLA `flash_mla_sparse_fwd`、B1→`mhc_pre`、B2→`compress_forward`。前向-only（推理核无反向）；报延迟/TF·s/%峰值 + ours/sglang，并标 apples-to-apples caveat（吸收/稀疏 vs 我们 dense+sink+bias）。**`torch.compile(reference)` 仅作背景参考，不作 forward 效率标尺。**
3. **backward 效率（仅 vs 本核 forward）**：报 bwd/fwd 延迟比 + bwd%峰值 vs fwd%峰值。理想 ~2–2.5×（flash）；若反向 GEMM 量约为前向 3.5× 则有 ~3.5× 结构性下限。sglang/torch.compile 无可用训练反向，不作标尺。
4. **codex(xhigh) 评审**每个核 + GPU 复验（codex 容器无 GPU，它的改动必须本地 GPU 验证）。

## recompute / activation-checkpoint 兼容性（分析，未改代码）

"recompute" 两层含义,要分清:
1. **算子内部 flash 式 recompute**（前向只存极少量,反向重算）——**三个核已内建**:A1 只存 `lse`、反向从 Q/K 重算概率 P(不存 [S×KV]);B2 只存 `pooled`、反向重算 softmax 权重;B1 反向从 `raw` 重算 20-iter Sinkhorn。这是它们 32k 省显存的主力(也是 B1 bwd/fwd 高的原因)。
2. **框架级 activation checkpoint**（Megatron `recompute_granularity` / `checkpoint_core_attention` / `torch.utils.checkpoint`）——**天然兼容,无需改 kernel**:三个核都是规范 `torch.autograd.Function`,前向确定性、无副作用、无 dropout/随机(V4 `attention_dropout=0`);checkpoint 重跑外层 module 前向时会重新调用我们的 `Function.forward`(确定性 → 重跑结果一致 → 反向正确)。

**集成时要处理的 3 点(属集成层,非 kernel 层)**:① 接线——让包裹 `v4flash_attention` 的 mcore module 能被 `checkpoint_core_attention`/`recompute_granularity` 包住(现在 kernel 还是裸算子、未接 module,这是"看起来不支持"的真实来源);② checkpoint 路径梯度一致性验证;③ **A1 的 dKV 用 fp32 原子累加 → 梯度 run-to-run 非位级可复现**(不影响 checkpoint 正确性,因 checkpoint 重跑的是确定性前向;但若 Megatron 开**确定性训练模式**,需要确定性 dKV 变体[split-K/两阶段],ncu 判定非瓶颈故暂未做)。

## LoRA 支持（分析，未改代码）

**结论:三个核对 LoRA 的支持是充分的——LoRA 不在 kernel 里,kernel 只需提供"输入梯度"让梯度跨层流到各 adapter,这点三个核都满足。**

- **LoRA 在 linear 上,不在 kernel 里**:LoRA = 给线性层加 `W+(B@A)·s`(`megatron.bridge.peft.LoRA` 包 q_a/q_b、kv_proj、o_proj 等)。我们的核(A1 注意力 / B2 压缩 pool / B1 mHC)是**激活上的纯计算**,不持有任何 base/adapter 权重,**不需要任何 LoRA 逻辑**。
- **LoRA 训练对 kernel 的真实要求 = 反向要给出输入梯度**(让梯度跨层回流到所有层的 adapter)。三个核都给:A1 给 `dq/dk_raw/dk_comp`(直接喂被 LoRA 适配的 q_b/kv_proj 反向)、B2 给 `dkv/dgate`、B1 给 `d_x`。**即使 CSA/HCA/mHC 自身 frozen,也必须有输入梯度做跨层回流——已具备。**
- **frozen-param 的效率优化(LoRA 专属)**:base frozen 时,这些核**自身参数梯度不需要**,只要输入梯度。
  - **B1 已做**:LoRA 下 `fn/base/scale` frozen → `needs_input_grad` gating **跳过 d_fn 那个重 GEMM**,只算 d_x。
  - **A1 / B2 是机会点(非阻塞)**:A1 的 `dsink`(sinks frozen)、B2 的 `dpos_bias`/`dweight`(frozen)目前仍算,LoRA 下可按 `needs_input_grad` 跳过省算力——**待做的小优化**。
- **dtype 对齐**:核是 **bf16**(fp32 累加),匹配 **bf16 LoRA adapter + bf16 激活**;FP8 只在 base 的 TE 线性 / MoE experts,**不在我们的核里**(与 fp8 结论一致)。无冲突。
- **正好覆盖计划的 LoRA 目标**:handoff 计划把 LoRA 挂在 MLA 注意力线性(q_a/q_b、kv 压缩、o_proj)。A1 的反向 `dq/dk` 喂 q_b/kv adapter,`o_proj` 的梯度从层后流入——**A1 完整支撑这套目标**。

**集成时要验证的点**:① autograd.Function 的输入梯度流要与 `megatron.bridge.peft.LoRA` 包裹 + **EP/PP 切分**(V4 注意力 EP 复制、无纯 TP)下的 adapter 形状对得上;② 给 A1/B2 补 `needs_input_grad` gating(frozen 时跳参数梯度);③ RoPE/grouped-o_proj 在核外,梯度经 autograd 串起(已是)。

## 后续（Phase-2 follow-up）
1. **A1 前向 warp 专门化**（唯一能把前向从 0.5× 往上推的路；occupancy/重叠受限，非调参可解）。可选 TMA loads、autotune（仅密集 shape）。
2. **B1**：若不想依赖 sglang，可自己写**非 warp 专门化**的融合 Sinkhorn+collapse（避开死锁根源）；否则 `hyper_connection_sglang` 已够。
3. **接入 Megatron Phase-1 parity spike**：把三个核接进 mcore 的 V4 模型，逐层对齐 HF。

## FP8 MoE expert compute (deep_gemm grouped fp8)
- Bench: `custom_kernels/deepseek_v4/megatron/bench_fp8_moe.py`; results: `custom_kernels/deepseek_v4/megatron/FP8_MOE_RESULTS.md`
- fp8 grouped fwd ~2x, fwd+bwd ~1.7x faster than the per-expert bf16 loop; matches sglang serving numerics. (2026-07-05)
