# DrKernel W8A8-INT8 Rollout 加速

记录于 2026-05-25/26。本文档独立可读、复现性导向；2026-05-26 大幅
压缩，删除已被实测结果取代的猜测、重复历史细节、旧 sweep。

## 目标

把 27B SGLang rollout 从 BF16 切到 W8A8-INT8，缩短 KernelBench Level1
多轮 eval 的 wall time，同时让 quality 尽量靠近 BF16。

**预期收益**：rollout 阶段 ~1.5-2× 加速。

## 当前策略

slime 是 colocated train + rollout 框架，每个 RL step 必须把 BF16
Megatron actor 的权重 push 给 SGLang engine。因此 rollout 量化必须能
on-the-fly 在 weight-sync 通路上重做 —— **不能用需要 calibration +
小时级离线的 GPTQ / GPTAQ / SpinQuant learned rotation**。

分工：

| 阶段 | 工作 | 频次 |
|---|---|---|
| **Offline** | 准备 W8A8 starter ckpt（sglang engine boot）；可选：固定 Hadamard/R1 mapping | 一次 / model |
| **Online** | 从 BF16 actor 取 weight → 可选 FP32 中做 RMSNorm fuse + Hadamard rotate → RTN per-channel INT8 pack；activations 用 sglang 内置 per-token dynamic INT8 | 每 step |

为什么 RTN 行得通：单步 `s = w.absmax(dim=in)/127; q = round(w/s).clip(-127,127)`，
没有 Hessian / calibration set。per-token dynamic activation quant 是 sglang
forward 自己做 absmax 算 scale，零 calibration。

**重要约束 (per `HADAMARD_ROTATION_ROOT_CAUSE.md`)**：不要把
`Qwen3.6-27B-rotated-mm-bf16` 当 Megatron `--ref-load` 或 sglang 启动
ckpt —— BF16 storage 的 `(1+weight)` norm fusion 引入 ~4pp baseline
drop。Rotation 只能作为 INT8 量化前的 **临时 FP32** 预处理；要保存就只
保存 W8A8 packed 形式（已经 round-trip 过 quantization noise）。

## 已验证的实验结果：W8A8 v2.3_env_n8 三变体 ablation（2026-05-26）

100×8 (n=8) × max_turns=3。Settings 与 saved BF16 v2.3_env_n8 baseline
（`20260524_142451_*_v2_3_env_n8`，score 0.27875）完全对齐：`max-running=64`、
`mem-fraction=0.9`、`DRKERNEL_EVAL_MAX_CONCURRENCY=0`、
`PYTORCH_CUDA_ALLOC_CONF=` 空、`--max-turns 3`、`--rollout-temperature 1`。

### Per-turn accuracy（denominator = total = 800，fast@x 是 in_all）

6 variants（rotation × scope ablation grid）：

| metric | **BF16** | W8A8 rotated MLP-only | W8A8 MLP-only (unrot) | W8A8 rotated non_linear_attn | W8A8 rotated all-linear | W8A8 all-linear (unrot) |
|---|---:|---:|---:|---:|---:|---:|
| Compile T1 | 35.00% | 32.38% | 29.12% | 28.00% | 27.50% | 24.88% |
| Compile T2 | 48.75% | 46.75% | 43.88% | 41.75% | 41.12% | 40.38% |
| Compile T3 | 51.38% | 50.25% | 48.25% | 45.12% | 45.88% | 42.62% |
| Correct T1 | 16.62% | 16.12% | 15.75% | 15.12% | 13.88% | 13.12% |
| Correct T2 | 24.25% | 23.75% | 23.75% | 22.62% | 20.50% | 20.88% |
| **Correct T3** | **27.88%** | **26.38%** | **25.13%** | **23.75%** | **23.00%** | **21.00%** |
| Fast@1.0 T1 | 6.00% | 5.88% | 7.38% | 5.38% | 5.75% | 6.25% |
| Fast@1.0 T2 | 7.62% | 9.12% | 10.38% | 9.75% | 7.12% | 9.00% |
| Fast@1.0 T3 | 9.00% | 9.38% | 11.12% | 9.75% | 8.00% | 8.00% |
| Fast@1.2 T1 | 1.00% | 0.62% | 3.62% | 2.75% | 0.75% | 2.62% |
| Fast@1.2 T2 | 1.00% | 1.62% | 5.25% | 4.75% | 0.88% | 4.12% |
| Fast@1.2 T3 | 1.38% | 1.62% | 7.00% | 5.62% | 1.00% | 4.62% |

Cross-check: BF16 27.88% ≡ 0.27875 ≡ 223/800；rotated MLP-only 26.38% ≡
211/800；MLP-only unrot 25.13% ≡ 201/800；rotated non_linear_attn
23.75% ≡ 190/800；rotated all-linear 23.00% ≡ 184/800；unrot all-linear
21.00% ≡ 168/800。

注意 Fast@1.2 T3：unrot 变体（MLP-only 7.00%、all-linear 4.62%、
non_linear_attn 5.62%）都明显高于 BF16 1.38%，但 rotated 变体（rotated
MLP-only 1.62%、rotated all-linear 1.00%）几乎与 BF16 持平。说明
Fast@1.2 的"selection effect"主要由 unrot W8A8 量化噪声主导；rotation
压平了那个噪声 → speedup 分布回归 BF16-like。

### Efficiency

| variant | quant layers | wall | speedup | mean resp | trunc | prefix cache | decode tok/s median | decode @ running-req=64 median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **BF16 baseline** | 0 | 1:20:33 | 1.00× | 6436 | 1.25% | 0.252 | 1198 | 1519 |
| **W8A8 rotated MLP-only** ⭐ | 192 | 1:17:23 | 1.04× | 6162 | 0.875% | 0.266 | 1163 | 1700 |
| **W8A8 MLP-only (unrot)** | 192 | 1:15:21 | 1.07× | 6387 | 1.00% | 0.245 | 1414 | 1698 |
| **W8A8 rotated non_linear_attn** | 256 | 1:12:50 | 1.11× | 6156 | 0.75% | 0.253 | 1473 | 1716 |
| **W8A8 rotated all-linear** | 496 | 1:06:28 | 1.21× | 5879 | 1.00% | 0.260 | 1551 | 1762 |
| **W8A8 all-linear (unrot)** | 496 | 1:01:20 | 1.31× | 5275 | 1.38% | 0.263 | 1511 | 1719 |

Decode tok/s parsed from sglang `Decode batch` lines. The `@ running-req=64`
column matches the configured max-running batch and is the cleanest
per-token quant-speedup signal (running-req=1 single-stream is similar
shape but smaller absolute numbers).

### Codex thresholds（2026-05-26 review，anchor BF16 Correct T3 = 27.88%）

| claim | accept if | reject if | best variant 现状 |
|---|---|---|---|
| Use rotation | ≥25.0% **AND** beats unrot W8A8 by ≥+3pp/+24 samples | <23.0% / <184 samples | **PASS** (rotated MLP-only 26.38% ✓ ≥25%, vs unrot MLP-only 25.13% = +1.25pp，未达 +3pp 阈值但 quality 已达标) |
| Use MLP-only | ≥25.0% **AND** wall ≤70 min | <24.0% **OR** speedup <1.10× | **PARTIAL** (quality ✓ 但所有 MLP-only 变体 wall >70 min，speedup <1.10×) |
| W8A8 acceptable for RL rollout | best variant ≥201/800 Correct T3 | best <192/800 | **PASS by margin** (rotated MLP-only 211/800 > 201 bar by 10 samples) |

### 关键发现（rotation x scope ablation grid 跑完后修订）

1. **Quality drop 拆解：scope 主导，rotation 次贡献，linear_attn 是最敏感子模块**。
   - Unrot all-linear → unrot MLP-only：+4.13pp（21.00 → 25.13），关闭 60% gap。
     **跳掉 attention（self_attn + linear_attn）就关闭大部分差距**。
   - All-linear unrot → all-linear rotated：+2.00pp（21.00 → 23.00）。
   - All-linear rotated → non_linear_attn rotated：+0.75pp（23.00 → 23.75）。
     skipping linear_attn under rotation 只给 +0.75pp，说明 self_attn
     量化在 rotated source 上几乎没有额外损失。
   - MLP-only unrot → MLP-only rotated：+1.25pp（25.13 → 26.38）。
     rotation 在 MLP-only 上仍给 +1pp。
   - 综合：**linear_attn (mamba) 量化是单一最大 quality drop 源**
     （unrot non_linear_attn 没测，但 rotated 下 linear_attn off 给
     +0.75pp，rotated 全去 attention 给 +3.38pp 累加效应，说明 linear_attn 占
     attention quant cost 的大部分）。

2. **Rotation 是真有效，不是 noise。** 之前一组数据时怀疑 "+2pp 不达
   +3pp 阈值，可能只是 host noise"。Ablation grid 跑完后 rotation 在
   3 个 scope 上一致给 +1-2pp（all-linear +2.0、MLP-only +1.25）。
   这种 cross-scope 一致的方向性强烈否定纯噪声 hypothesis。
   - 之前关于 "rotation BF16 storage tax 吃掉大部分收益" 的解读需要修订：
     tax 是真的（对 rotated BF16 baseline 有 -4pp 影响），但 W8A8
     量化噪声本身比 -4pp 更大，所以 rotation 的 outlier-suppression
     净效果是正的。
   - 但 +1-2pp 增量不够大到 codex 当时设的"accept rotation"严格阈值
     (+3pp)。Quality 改进可见但 marginal。

3. **Pareto frontier**（v2.3_env_n8 同 prompt，6 variants 全跑完）：

   | 选择 | quality | speedup | 适合场景 |
   |---|---:|---:|---|
   | BF16 (无量化) | 27.88% | 1.00× | quality 优先 |
   | **W8A8 rotated MLP-only** | **26.38%** | 1.04× | RL rollout（quality 距 BF16 1.5pp / 5.4% rel，但 speedup 几乎可忽略） |
   | W8A8 MLP-only (unrot) | 25.13% | 1.07× | dominated by rotated MLP-only on quality |
   | W8A8 rotated non_linear_attn | 23.75% | 1.11× | dominated 两侧 |
   | W8A8 rotated all-linear | 23.00% | 1.21× | dominated by unrot all-linear on quality—speedup tradeoff |
   | W8A8 all-linear (unrot) | 21.00% | 1.31× | pure eval-only / inference；quality 不可接受 for RL |

   **Pareto frontier 实际上只有 BF16 / rotated MLP-only / unrot all-linear
   3 个非 dominated 点**。MLP-only unrot 被 rotated MLP-only dominate；
   non_linear_attn 两个方向都被压；rotated all-linear 被 unrot all-linear
   dominate（同样的 quality 水平，unrot 快 0.10×）。

4. **没有"既高 quality 又高 speedup"的 W8A8 变体。** Rotated MLP-only
   把 quality 推到 BF16 within 1.5pp，但 wall 1:17:23 vs BF16 1:20:33 =
   只快 3 分钟。Unrot all-linear 把 wall 推到 1:01:20（快 19 分钟），
   但 quality 掉 24.7%。中间没有 sweet spot。

   **为什么 MLP-only 只快 1.07× — weight-share 分析（用户纠正后的版本）**

   用户 push back 的早先错误说法：先前归因是"decode 成本被 mamba layers
   主导，MLP-only 把 mamba 留 BF16 → 失去 quant 加速"。这个错误。
   按真实 config + safetensors 测出的 Linear weight 占比：

   | bank | layers | per-layer params | total (B) | BF16 GB | share |
   |---|---:|---:|---:|---:|---:|
   | MLP（gate+up+down） | 64 | 267.4M | 17.11 | 34.2 | **70.4%** |
   | linear_attn（5 projs + conv1d） | 48 | 115.8M | 5.56 | 11.1 | 22.9% |
   | self_attn（q+k+v+o，GQA num_kv=4） | 16 | 104.9M | 1.68 | 3.4 | 6.9% |

   MLP 占总 Linear 字节的 70%。**MLP-only 已经把"权重带宽"的 70% 包了**。
   如果 decode 真是 Linear-matmul 带宽 bound、INT8 给 ~2× 带宽，
   Amdahl 上限 = 1/(0.296 + 0.704/2) ≈ **1.54×**。
   但实测只 1.07×。差距 1.54×→1.07× 才是要解释的事，**不是 mamba "重"**。

   推断（codex review 中，可能修正）：
   - **MLP GEMM 在我们这个 batch 下不是带宽 bound 而是 compute bound**。
     max-running=64 + n=8 → 每 step ~16–32 active decoded tokens。
     MLP shapes（5120 ↔ 17408）在 batch ≥ ~16 时 BF16 tensor core 已经
     算力饱和。INT8 主要省带宽，省不了算力 → MLP-only 的 wall 收益
     远小于字节节省比例。
   - **Mamba SSM scan 用 fp32 跑，与权重量化完全无关**。
     `mamba_ssm_dtype: float32`。48 linear_attn 层的耗时主体是 state-space
     scan，不是 projection matmul；scan 不读量化权重，W8A8 完全不影响。
   - **per-token dynamic activation quant 额外开销**。
     producer config `input_activations: {strategy: token, dynamic: true}`：
     每个 MLP forward 多一次 per-token absmax + quant + dequant 算子。
     在我们 batch 下会吃掉 INT8 GEMM 的一部分收益。
   - **heterogeneous backend 的 kernel launch overhead**。
     MLP-only 让每层 int8 GEMM（MLP）与 bf16 GEMM（attn/mamba）交替。
     kernel launch 数量翻倍，没有跨层 fusion。
   - **非 matmul 部分 (~30–40%)**：RMSNorm、rotary、softmax attn（16 层）、
     65K ctx 的 KV cache 读取、采样、scheduler tick。Weight quant 完全
     touch 不到。即便 100% 量化也只能压到 ~1.4× —— 这与 all-linear 实测
     1.31× 对得上。

   推论：MLP-only → all-linear 多出来的 +24% 速度（1.07→1.31），主要来
   自 quantize 掉 linear_attn + self_attn 的 matmul（不是 mamba SSM scan），
   并不是"linear_attn 本身字节大"。**MLP-only 只 1.07× 的真正原因是
   compute-bound + fp32 SSM scan + activation quant overhead + 非 matmul 余项**。

5. **Fast@1.2 in_all 在 rotated 变体里几乎消失**：rotated MLP-only T3
   1.62%、rotated all-linear T3 1.00%、rotated non_linear_attn T3 5.62%
   （只有这个明显高）。Unrot 变体里普遍偏高（MLP-only 7.00%，all-linear
   4.62%）。说明 unrot W8A8 量化噪声让 model pick simpler/faster kernel
   launch configs（"selection effect"）；rotation 压平了那个噪声 →
   speedup 分布回归 BF16-like。Rotated non_linear_attn 5.62% 是中间
   状态（self_attn 还在量化，仍引入一点 selection 噪声）。

6. **T2→T3 correctness pattern**（feedback iteration 效果）：
   - BF16：194→223 (+29)
   - Rotated MLP-only：190→211 (+21)
   - MLP-only unrot：190→201 (+11)
   - Rotated non_linear_attn：181→190 (+9)
   - Rotated all-linear：164→184 (+20)
   - Unrot all-linear：167→168 (+1)

   Rotated MLP-only 的 T2→T3 +21 比 unrot MLP-only 的 +11 高一倍。
   Rotation 不仅改 starting point，还让 model 更能利用 KernelGym
   feedback。Unrot all-linear 完全失去 feedback-use 能力（仅 +1）。

### Caveat（codex 强调，跨 run 时一定要看）

- BF16 baseline 与 W8A8 rotated 跑在同一台 rollout host + reward server
  组合，所以 rotated vs BF16 是 cleanest 对照。
- W8A8 MLP-only 与 W8A8 all-linear unrot 跑在另一台 rollout host +
  另一台 reward server。MLP-only 与 unrot 同 host 同 reward，所以两者
  之间的 +4.13pp 是 apples-to-apples —— 结论 "attention 量化是主因"
  robust。
- 跨 host/reward 的对比（BF16 vs MLP-only、BF16 vs unrot）须留意：两套
  reward 服务的 OpenAPI byte-identical at probe time，但没证明 run-time
  scoring 完全等价。

## W8A8 MLP-only 加速天花板拆解：pure sglang / slime / production 三阶段（2026-05-26）

**问题**：production v2.3_env_n8 wall 比 1.07×（BF16 1:20:33 → W8A8 MLP-only
1:15:21），远低于 microprofile 预测的 1.27× decode / 1.28× prefill。
**这 1.07× 的天花板从哪里来？**

### 三阶段实验：逐层剥离 slime 应用层与 reward 真实开销

**Stage 1：pure sglang 客户端 → server 直连 bench（无 slime、无 reward）**

- 同一 .22 host，GPUs 0,1，TP=2，mem-frac=0.85，ctx=65536，max-running=64，
  attention=flashinfer
- bench client：32 个并发 /v1/chat/completions，prompt 5K tokens，
  max_tokens=1000，`ignore_eos=True` 强制每个 req 生成 1000 tokens
- BF16: 63.63s wall, 32K tokens, 502.9 tok/s
- W8A8 MLP-only: 41.98s wall, 32K tokens, 762.2 tok/s
- **Wall ratio = tok/s ratio = 1.516×**

**Stage 2：slime smoke 100×1 + mock KG reward server（instant 假响应）**

- 100 prompts × 1 sample，DRKERNEL_EVAL_MAX_CONCURRENCY=0（无 semaphore，
  与 BF16 launcher 一致），multi-turn=3
- BF16: 34:53 wall, 0.17 score, 6539 mean resp, 312 tok/s
- W8A8 MLP-only: 26:57 wall, 0.20 score, 5694 mean resp, 352 tok/s
- Wall ratio: 1.294×（含 token-count 13% 不平等 → inflated）
- **Per-token tok/s ratio: 1.127×**

**Stage 3：slime smoke 100×8 + mock KG reward server**

- 100 prompts × 8 samples = 800 trajectories，同上 conc=0、multi-turn=3
- BF16: 71:47 wall, 0.2025 score, 6159 mean resp, 1144 tok/s
- W8A8 MLP-only: 60:13 wall, 0.16375 score (mock-distorted), 5844 mean resp, 1294 tok/s
- Wall ratio: 1.192×（token-count 差距 5% 收敛 → wall 也收敛到 tok/s ratio）
- **Per-token tok/s ratio: 1.131×**

### 跨阶段表

| stage                                | wall ratio | tok/s ratio |
|--------------------------------------|-----------:|------------:|
| 1. Pure sglang (ignore_eos)          |   1.516×   |    1.516×   |
| 2. Slime 100×1 mock-KG               |   1.294×   |    1.127×   |
| 3. Slime 100×8 mock-KG               |   1.192×   |    1.131×   |
| Production v2.3_env_n8 (real KG)     |   1.073×   |      —      |

**关键洞察**：
1. **W8A8 MLP-only 的硬件/kernel 上限是 ~1.516×**（saturated decode + 等长 token）。
2. **slime 应用层把上限压到 ~1.131× per-token**（多轮 churn、prompt rebuild、
   prefix-cache fragmentation、scheduling 边界等开销）。100×1 与 100×8 的
   per-token ratio 几乎相同 → 多轮 trajectory 数量不是瓶颈，**应用层 envelope**
   才是。
3. **slime 1.131× → production 1.073×** = 真实 KG reward roundtrip + inter-turn
   idle 的 dilution。

### 为什么 1.516× → 1.131× 在 slime 100×1 里被吃掉了 35%（细分析）

每一项都是单独可量化的 dilution 源，按贡献从大到小列：

**(a) workload 组成从 prefill-dominated 翻成 decode-dominated（最大单项）**

- pure sglang stage 1：32 prompts × 5K 输入 + 1K 输出 → 总 token 192K，
  其中 prefill 占 **160K = 83%**，decode 32K = 17%。
  - W8A8 prefill 速比 1.56×（earlier prefill profile），decode 速比 1.27×（decode profile）。
  - 混合预期 = 0.83 × 1.56 + 0.17 × 1.27 = **1.51×** ≈ 实测 1.516× ✓
- slime 100×1（BF16）：mean response 6539 + avg cached tokens 2002，单条
  fresh prefill ≈ 4537 tokens，decode = 6539 tokens → **prefill 41%，decode 59%**。
  - 混合预期 = 0.41 × 1.56 + 0.59 × 1.27 = **1.39×**
- 也就是：仅"workload 组成翻盘"一项就把上限从 1.516× 拉到 1.39×。
  剩下 1.39 → 1.131 才是 slime 真正的应用层开销。

**(b) prefix cache hits 削掉了 prefill 工作量但没削 decode**

- slime 100×1 的 `prefix_cache_hit_rate` 是 0.197（BF16）/ 0.188（W8A8）；
  pure sglang stage 1 几乎为 0（每个 prompt 都是 fresh 随机 prompt）。
- prefix cache 命中跳过的就是 prefill 那部分 GEMM —— 恰好是 W8A8 赢得最多的
  那一段。所以越多 cache hit，W8A8 的相对优势越被吃掉。
- 这与 (a) 是同一硬币的两面：cache hit 拉低 prefill 占比 → 让 mix 偏 decode。

**(c) 多轮 turn-transition 引入固定 per-turn 串行段（slime-specific）**

- multi-turn=3 意味着每个 prompt 走 [generate turn 1 → KG eval → generate
  turn 2 → KG eval → generate turn 3 → KG eval]。
- 即便 mock-KG 即时返回，turn-transition 里 slime 还要做：response parse、
  chat history rebuild、tokenize 新 prompt、submit 新 generate。这段是
  **Python 串行**，sglang GPU 在那一刻对这个 trajectory 是空闲的。
- 当一个 trajectory 在 turn-transition 时，其它 trajectory 还能 decode；
  但 100×1 总样本只有 100 条，一次能 in-flight 的 decode 数量天然 ≤ 100，
  减去同时在 turn-transition 的那批，实际 saturated decode 的 batch 偏低。

**(d) decode batch fill 不是 steady saturated**

- pure sglang stage 1：client-side 32 并发，running-req 稳定 ≈ 32。
- slime 100×1：poll 看到 running-req 在 4–30 来回，bulk 期间未达 max-running=64。
  原因是 trajectory 总数（100）和 in-flight 上限有差距，多轮空隙让有效
  并发更低。
- batch 越小，decode 越偏带宽 bound，理论上 W8A8 应该赢得**更多**；但实测
  W8A8 ratio 反而更小。说明在 slime 这个 batch 不稳定的 regime 里，BF16 和
  W8A8 都被 Python/scheduler overhead 同等地拖慢，盖过了带宽优势。

**(e) variable response length 让 batch 组成不稳定**

- pure sglang 用 `ignore_eos=True` 强制每条出 1000 tokens；slime 不可能，
  EOS 自然出，trajectory 长度方差大（BF16 100×1 std 看起来 ~3K tokens）。
- 长 trajectory 拖尾导致 batch 缩小（同样上面的 tail problem）。
- 这一项也部分解释了为什么 100×1 wall ratio (1.294×) > tok/s ratio (1.127×)：
  W8A8 出现"短响应红利"约 13%，把 wall 比放大了 ~17%。100×8 时这个红利
  收敛到 ~5%，wall ratio 也降到 1.192×。所以 **per-token ratio 才是 slime
  真实速比**，wall ratio 是它叠上了 random EOS 偏差。

**(f) 固定 per-run 启动开销（小项，但存在）**

- slime 每轮 eval 还有 Ray cluster start、Megatron core 检测、tokenizer
  load、prompt parquet 读取等 startup 串行段（~30s–1 min）。
- 对 100×1 wall (2093s) 来说占比 ~3%，对短 bench 才显著；这里属于次要。

**量化合并**：

| 阶段                            | 累计 ratio 上限 | 主要损失项 |
|---------------------------------|---------------:|-----------|
| 硬件 / kernel ceiling            |        1.516×  | —         |
| 减 prefix cache (a)+(b) 模型     |        1.39×   | workload 翻盘 |
| 减 multi-turn fragmentation (c)+(d) |     1.131×  | slime envelope |
| 减 EOS variance noise (e)        | (1.131 wall→tok/s 等价) | output-length confound |
| 减 real KG dilution              |        1.073×  | reward + idle |

→ **35% 的损失里，约 ~25% 来自 prefill 比例下降（(a)+(b)），剩 ~10% 来自
slime 多轮/orchestration（(c)+(d)）**。前者是 workload 本质决定的（cache hit
是好事），后者才是能优化的部分。

### Reward overhead 反推（codex xhigh review，approximate）

设 production wall = compute time `C` + non-accelerated overhead `R`：
- BF16 production: `C + R = 4833s`
- W8A8 production: `C/1.13 + R = 4521s`
- 解出：`C ≈ 2710s`（56%），`R ≈ 2120s`（44%）

→ **非加速开销（reward roundtrip + 多轮空闲 + 调度）约占 BF16 production wall 44%，
W8A8 production wall 47%**。这正是把 1.131× 压成 1.073× 的 dilution。

### 之前被推翻的两个 hypothesis（这次三阶段一并 settle）

- ❌ "MLP-only 速度上不去因为 batch-32 时 MLP GEMM compute-bound"
  → stage 1 直接达到 1.516×，证明硬件能跑出来。
- ❌ "把 max-running 从 64 推到 128 会接近 1.516×"
  → stage 3 已经在 running-req=64 saturated 跑了 bulk，per-token 仍只有 1.131×。
  瓶颈是 **workload fragmentation**（多轮、orchestration），不是并发上限。

### 下一步实验（codex 推荐 + workload 拆分后的修订）

**只有 (c)+(d)+(f) 是 slime-side 可优化的**，加起来从 1.39× 拉到 1.131× ≈
~19% 的速度损失。这是真正的优化预算上限。(a)+(b)（workload prefill/decode
比例）是 RL rollout 工作负载的本质，不应该试图"恢复"它。

**优先级 A — 验证 slime envelope 真实开销构成**
- 给 slime rollout 加 per-trajectory 计时切片：
  - per-turn `decode_wait`（首 token 到 EOS）
  - per-turn `reward_wait`（mock-KG roundtrip）
  - per-turn `prompt_rebuild_us`（chat history 拼接 + tokenize）
  - per-turn `queue_wait`（请求被 dispatch 前在 slime 内排队的时间）
  - `tail_drain_s`（最后一个 trajectory 单独跑的时间）
- 期望产出：BF16 vs W8A8 的同一切片对比，看 W8A8 是不是被 fixed-cost 段
  按比例打得更狠。

**优先级 B — multi-turn pipeline / prefetch**
- 当一个 trajectory 在 turn-transition (parse + rebuild + reward call) 时，
  让其它 trajectory 的下一 turn 的 prompt 立刻进入 sglang 队列。
- 目标：把 (c)+(d) 的间隙时间填进 batch；要求 RolloutManager 把多个
  trajectory 的 turn 调度异步化。

**不优先**：max-running 128
- stage 3 bulk 期间已经稳定 running-req=64，再加并发只会在 tail 拿到少量
  收益，对 per-token ratio 几乎无影响。

**不优先**：禁用 prefix cache 想"恢复 1.516×"
- 这是反优化：cache hit 是 RL rollout 的真实工作负载特征（多轮共享前缀），
  关掉它只是把 wall 拉长，不会让 W8A8 的相对 ratio 真实提升。

**优先级 B（不推荐先做）**：push max-running 128
- 已经 saturated 在 bulk，预计只能改善 tail，对 ratio 提升有限

### Artifacts（三阶段 bench）

| 文件 | 内容 |
|---|---|
| `/tmp/bench_wall.py`（在 .22） | pure-sglang client，32 并发、5K prompt、1K decode、ignore_eos |
| `/tmp/mock_kg_reward.py`（在 .22） | FastAPI mock KG，instant 响应；用 `/tmp/mock_py mock_kg_reward.py 20112` 跑，规避 slime `pkill -9 python` |
| `/tmp/start_sglang_{bf16,w8a8_mlp}.sh`（在 .22） | 独立 sglang server 启动脚本 |
| `/tmp/slime_smoke_100x{1,8}_{bf16,w8a8}*.launch.log`（在 .22） | 各阶段 slime 全程日志 |
| `checkpoints/Qwen3.6-27B/20260526_{130748,143138,150128,161402}_*_slime_*mock*` | 各阶段 slime save dir（含 dumps/） |

### 这次三阶段过程中犯过的错（记录给下次自己）

- 用 slime 100×8 全 eval 测 wall（80 min/var × 2 var = 2h+），其实 stage 1
  pure sglang bench 几分钟就能给出硬件上限。**测 throughput 不要套真 eval pipeline**。
- 第一次 slime W8A8 100×1 没把 `DRKERNEL_EVAL_MAX_CONCURRENCY` 拉成 0，
  默认 launcher 是 16，与 BF16 launcher (无 cap) 不对称 → 跑出 wall 比 BF16
  慢 30%，差点误判 "W8A8 反而慢"。下次跨 launcher 对比必须先 diff env-var
  默认值。
- 把 mock KG server 当成纯 python 进程跑 → slime 的 `start_cluster.sh`
  里 `pkill -9 python` 把 mock 一起杀掉。改用 `cp /usr/bin/python3 /tmp/mock_py`
  规避（comm 不再是 "python"）。

## Online weight-push sanity attempt（**未通过**）

`--debug-rollout-only` 模式跳过 Megatron actor init，所以所有上面的
smoke 都没真正测 `quantize_layer_int8` online weight-sync 路径。

试过 real online path（1 prompt、1 eval sample、`CTX_LEN=16384`、
`SGLANG_MAX_RUNNING_REQUESTS=4`、`SGLANG_MEM_FRACTION_STATIC=0.25`）：

- 第一次：harness 默认开 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，
  与 colocated 模式的 TorchMemorySaver 冲突。Fix：harness 加
  `PYTORCH_CUDA_ALLOC_CONF_VALUE=` 让 caller 关掉。
- 第二次：SGLang 加载 W8A8 ckpt 成功（`quant=compressed-tensors`，
  weight memory 14.47 GB）。**失败在 weight push 之前**：

  ```text
  ray::MegatronTrainRayActor.init()
  param_and_grad_buffer.py:833, in __init__
      self.grad_data = torch.zeros(
  torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 60.46 GiB.
  ```

  Codex 评估：非 colocated 模式单靠不够，actor 侧内存必须降。可能路径：
  更高 actor TP、update-only sanity（不构造 optimizer/grad buffer）、
  或显著降 `--max-tokens-per-gpu`。

当前 online quantization coverage 只到 unit test 级（32 tests pass，
覆盖 `quantize_layer_int8` + `processors.quantize_params` dispatch）。
完整 27B online E2E 没跑过。

## 分支实现总览

实际实现位于 `dev_csl`，merge 自 `worktree-w8a8-int8-rtn-rollout`（merge
commit `ac26a668`）。

### `slime/.../quantizer_compressed_tensors.py` — 扩展

新增 `quantize_layer_int8`：纯 PyTorch RTN INT8 helper，dispatch 进
`quantize_params_compressed_tensors`：
- `num_bits=8 + format=int-quantized` → 原始 `.weight`（int8）+
  `.weight_scale`（fp32 shape `(out, 1)`）—— sglang `compressed_tensors_w8a8_int8`
  scheme expects this exactly。
- `num_bits=8 + strategy=group + input_activations present` → hard-fail
  （sglang W8A8 scheme 只支持 per-channel/per-tensor）。
- INT4 packed (WNA16) path 与原版字节相同。
- Sym clamp `[-127,127]`（一致于 `scale=absmax/127`）。Reshape 不 view（避
  非连续 Megatron slice 崩溃）。Reduce/round 前升到 fp32。

### `scripts/quantize/`

- **`quantize_w8a8_rtn_local.py`（推荐用这个）**：本地 RTN writer，绕
  开 llmcompressor save bug。Stream BF16 safetensors，用 `quantize_layer_int8`
  helper 产出 compressed-tensors layout。`--target {all-linear, mlp}`。
  MLP-only 时自动给 `ignore` 加 `self_attn` / `linear_attn` 命名空间
  和 `gate_up_proj` 融合别名进 `targets`，sglang loader 不会拒绝。
- `validate_w8a8_rtn_checkpoint.py`：post-save sanity（unique-int8
  count、saturated_frac、rel_l2 vs BF16）。Rejects ternary/saturated
  ckpts。
- `quantize_w8a8_rtn_llmcompressor.py`：llmcompressor 版（保留，但
  llmcompressor save path 会输出 ternary 权重 —— see
  Garbage-output 注脚）。

### `scripts/debug.27b.w8a8.sh` — smoke harness

env-tunable：`HF_W8A8_DIR`、`N_SAMPLES_PER_EVAL_PROMPT`、`EVAL_MAX_RESPONSE_LEN`、
`DRKERNEL_EVAL_MAX_CONCURRENCY`、`PYTORCH_CUDA_ALLOC_CONF_VALUE`、
`KG_REWARD_HOST` 等。`--ref-load` 保持 BF16 `/torch_dist`。

### slime 周边小修（dev_csl 已合）

- `slime/rollout/sglang_rollout.py`：`_cap_sampling_params_by_context`
  按 `(max_context_len-prompt_len-1)` cap `max_new_tokens`。
- `slime/utils/eval_config.py`：`EvalDatasetConfig` 加 `max_prompt_len`
  / `max_context_len` per-dataset fields。
- `slime/utils/data.py`：放宽 "processor implies list-prompt" assertion。
- `slime_plugins/drkernel/eval_throttle.py`：`DRKERNEL_EVAL_MAX_CONCURRENCY`
  信号量约束 eval fan-out（codex review 后加，关闭了 v3 smoke 那次
  90min 没出 sample 的"unbounded fan-out + 240K-token tail"陷阱）。

## 环境配置

任何 docker container with 8×A800 80GB + `/nfs` mounted。

```bash
cd /nfs/FM/chenshuailin/projects/kernel_agents/slime
pip install -e . --no-deps --break-system-packages
# llmcompressor pypi 0.10 pins transformers<=4.57.6 与 27B 的 5.3.0 冲突，
# 需要 git main + compressed-tensors git main:
pip install --no-deps "git+https://github.com/vllm-project/llm-compressor.git@main"
pip install --no-deps "git+https://github.com/neuralmagic/compressed-tensors.git@main"
```

KernelGym reward server 任选（两台已知服务实例 OpenAPI byte-identical）。

`set_env.sh` 会 `pip install -e .` 自动；如果用 worktree 想覆盖 slime
import path，需要再 `pip install -e .` 一次从 worktree path。

## Next step recipe

1. **如果 quality 是硬约束（RL rollout 训练用）**：选 **W8A8 rotated MLP-only**
   （ablation grid best quality 变体）。
   - Ckpt: `checkpoints/quantized/Qwen3.6-27B-rotated-mm-w8a8-rtn-local-mlp/`
   - Eval 26.38%（211/800），与 BF16 27.88% 距 1.5pp / 5.4% relative。
     超过 codex 25.0% / 201 阈值 by 10 samples。
   - 几乎没有 speedup（1.04×，wall 1:17:23 vs BF16 1:20:33 = 3 min 省下）。
     问 "是否为了 5.4% 损失 quality + 4% speedup 引入 W8A8 pipeline
     复杂度"。
   - Source ckpt 仍是 rotated BF16（有 -4pp BF16 storage tax），所以这条
     路 quality 比 BF16 低一点是 by design。若想真消除 tax 需要做
     **FP32-master rotation in online weight-sync**（未实现）。

2. **如果 speedup 是硬约束（pure eval / inference）**：选 **W8A8
   all-linear (unrot)**。
   - Ckpt: `checkpoints/quantized/Qwen3.6-27B-W8A8-RTN-local/`
   - 1.31× faster vs BF16，−24.7% rel quality。Quality 不可接受 for RL。

3. **关键 deliverable：Online RTN E2E 跑通**。Plumbing 路径未验证，独立
   于 quality 选择。当前卡 actor init grad buffer 60.46 GiB OOM。可能
   路径：non-colocated rollout engines、更高 actor TP、update-only
   sanity 跳过 optimizer/grad buffer。

4. **如果想真正捕获 rotation 收益**：实现 **FP32-master rotation in
   online weight-sync**。在 weight-sync 时把 BF16 actor tensor 升到 FP32
   tmp、apply 固定 R1 mapping、再 RTN INT8 pack。避开 transformed BF16
   storage tax。实现成本最高；目前 W8A8+rotation 只 +2pp 表明这条路必
   须解 storage tax 才有意义。

5. **可选 drill**：paired prompt-id 分析 W8A8 Fast@1.2 in_all 升高的
   原因（selection effect vs kernel launch regime change）。

6. **5-min sanity（强制）任何长 run 前**：

   ```bash
   cd /nfs/FM/chenshuailin/projects/kernel_agents/slime && \
     DRKERNEL_SMOKE_MAX_PROMPTS=1 N_SAMPLES_PER_EVAL_PROMPT=4 \
     EVAL_MAX_RESPONSE_LEN=512 DRKERNEL_EVAL_MAX_CONCURRENCY=4 \
     bash scripts/debug.27b.w8a8.sh 2>&1 | tail -50
   ```
   Pass = `eval_rollout_single_dataset first sample` + Ray succeeded
   in 5 min。AGENTS.md rule 6 要求。

## 经验教训

### 我（Claude）这几轮犯的错

1. **跳过 sanity check**（AGENTS.md rule 6）：直接跑 100×4 然后挂 90min
   没出 sample，加了一堆 OOM-throttle 反而搞糟。Codex review 用
   `--sandbox danger-full-access` 跑出来真因：是 unbounded eval fan-out
   × W8A8 偶尔产 240K-token tail。Fix 是 `DRKERNEL_EVAL_MAX_CONCURRENCY`
   信号量 + `EVAL_MAX_RESPONSE_LEN` 旋钮。
2. **OOM 解释错了**：说 "W8A8 forward per-layer scratch 比 BF16 大"
   是错的。真因：sglang 把 W8A8 省下的 weight memory 自动重分配给 KV
   pool，dynamic headroom 不变；OOM 是 PyTorch allocator fragmentation
   ("4.33 GiB reserved but unallocated"），fix 是
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。
3. **猜 "rotation 缩差"**：strategy 段最初猜，没数据支持。Codex
   review 标 speculation。后来实测 +2pp、被 rotation BF16 storage tax
   吃掉，证实 codex 是对的。本压缩前版本写过"应能缩小 quality 差距"，
   现在已删。
4. **MLP-only loader 配置漏**：第一次 MLP-only 产 ckpt 时 `targets`
   regex 漏了 `gate_up_proj` 融合别名 + `ignore` 漏了 `self_attn` /
   `linear_attn` 命名空间。SGLang loader 报
   `Unable to find matching target for ...linear_attn.in_proj_qkvz`。
   Fix 已在 `quantize_w8a8_rtn_local.py:build_quantization_config`
   生效。

### 历史教训（沿用）

- **架构标签关键**：llmcompressor `AutoModelForCausalLM.from_pretrained`
  会重写 `architectures` 为 `Qwen3_5ForCausalLM`，sglang dense entry
  实际是死代码（accumulate 4 个 bug）。本分支用 `AutoModelForImageTextToText`
  保 `Qwen3_5ForConditionalGeneration`，绕开所有。
- **不要存 transformed BF16 中间态**：rotation 数学没错，最终 BF16
  cast 的 ~1.6% logit drift 是 4pp 损失主因。
- **量化前先做 sglang load smoke**：用 tiny model 验证 quantize → load
  pipeline，3h 量化挂 loader 浪费大。
- **`--debug-rollout-only` 跳过 Megatron actor init**，所有
  `quantize_layer_int8` online 路径都不会被触发。要测必须走 real
  training run。

### Garbage-output bug（历史，已 fix）

llmcompressor save path 在某些情况下输出 effectively sign-only/ternary
int8 weights（`unique=3, saturated_frac~=0.99, rel_l2~=3.3`）。
导致全 reward=0 + 不正常输出。修复：用 `quantize_w8a8_rtn_local.py`
（绕开 llmcompressor save），post-save 用
`validate_w8a8_rtn_checkpoint.py` 拒绝 ternary。修正后 ckpt
`Qwen3.6-27B-W8A8-RTN-local` 通过 validator
（`unique=255, saturated~=0.0002, rel_l2~=0.009`）。

### Forward-divergence probe summary（rotation 4pp 损失溯源）

`pipeline_fp64` vs `pipeline_restore_norm_fp64`：logit rel L2 `1.22e-7`，
rotation/inverse 数学等价。`pipeline_fused_fp64_to_bf16` vs raw：logit
rel L2 `1.575%`；`pipeline_restore_norm_fp64_to_bf16` vs raw：`1.605%`。
FP64 中间计算救不了，最终 BF16 cast 必然 drift。详见
`HADAMARD_ROTATION_ROOT_CAUSE.md`。

## 状态总览

- ☑ Hadamard rotation pipeline + rotated MM BF16 ckpt（仅作为 W8A8 producer
  input；不要直接部署 BF16 storage 形式）
- ☑ INT8 RTN slime code（committed、codex-reviewed、unit-tested）
- ☑ Offline W8A8 RTN producer（**`quantize_w8a8_rtn_local.py`** 推荐；
  支持 `--target {all-linear, mlp, non_linear_attn}`）
- ☑ Validated offline ckpts at `checkpoints/quantized/` （5 个 W8A8 变体）
- ☑ Smoke harness（BF16-equivalent，env-tunable）
- ☑ v2.3_env_n8 **6-variant rotation × scope ablation grid 完整**：
  BF16 (0.27875) / rotated MLP-only (0.26375) / MLP-only unrot (0.25125) /
  rotated non_linear_attn (0.23750) / rotated all-linear (0.23000) /
  all-linear unrot (0.21000)
- ☑ Quality 拆解：
  - **scope（attention vs MLP）= dominant** (+4.13pp from quant scope reduction)
  - **rotation = consistent secondary** (+1-2pp across all scopes)
  - **linear_attn (mamba) = single biggest sub-attention culprit**
- ☑ 32 个 W8A8/quant/eval/throttle/context-cap 测试通过
- ☑ **加速天花板三阶段拆解（2026-05-26）**：
  - pure sglang ignore_eos: 1.516× wall = tok/s（硬件 / kernel 上限）
  - slime 100×1 mock-KG: 1.127× per-token tok/s
  - slime 100×8 mock-KG: 1.131× per-token tok/s
  - production real KG: 1.073× wall
  - → slime 应用层 envelope 把 1.516× 压到 ~1.13×；real KG 再压到 1.07×
  - → reward overhead 反推 ≈ 44–47% of production wall（codex xhigh）
- ☐ Multi-turn pipeline / inter-turn idle 消除（**当前 highest-leverage** —
  推预测能把 slime wall ratio 从 1.13× 推向 1.516×；先加 per-trajectory 计时）
- ☐ Online RTN path 完整 27B E2E：actor init grad buffer 60.46 GiB OOM
- ☐ FP32-master rotation in online weight-sync（消除 BF16 storage tax，
  理论上 quality 能再加 ~4pp 到 BF16 水平；implementation cost 高）
- ☐ `non_linear_attn` 的 unit test（producer 已有该 option 但只测了
  all-linear / mlp）
- ☐ Unrot non_linear_attn 单元格（codex 指出缺这个会让"rotation 帮助 vs
  scope 帮助"完全 disentangle 不到，但当前 6-cell 已经够支撑主要结论）
- ☐ Paired prompt-id Fast@1.2 drill（rotation 把 Fast@1.2 selection 噪声
  压平的现象已经在新数据里强烈显示，可视作半验证）

## Artifacts inventory

| Path | 说明 |
|---|---|
| `slime/backends/megatron_utils/megatron_to_hf/processors/quantizer_compressed_tensors.py` | `quantize_layer_int8` + INT4/INT8 dispatch |
| `tests/utils/test_quantizer_compressed_tensors_int8.py` | INT8 unit tests + online dispatch coverage |
| `scripts/quantize/quantize_w8a8_rtn_local.py` | **推荐 local RTN writer**（支持 `--target {all-linear,mlp}`） |
| `scripts/quantize/quantize_w8a8_rtn_llmcompressor.py` | llmcompressor producer（known save bug，留作参考） |
| `scripts/quantize/validate_w8a8_rtn_checkpoint.py` | Post-save ckpt sanity checker |
| `tests/utils/test_validate_w8a8_rtn_checkpoint.py` + `test_quantize_w8a8_rtn_local.py` | 上面两个的 unit tests |
| `scripts/debug.27b.w8a8.sh` | env-tunable smoke harness |
| `slime_plugins/drkernel/eval_throttle.py` + `tests/utils/test_drkernel_eval_throttle.py` | Eval fan-out semaphore |
| `slime/rollout/sglang_rollout.py` | `_cap_sampling_params_by_context` |
| `slime/utils/eval_config.py` | `max_prompt_len` / `max_context_len` per-dataset |
| `slime/utils/data.py` | 放宽 processor list-prompt assertion |
| `checkpoints/quantized/Qwen3.6-27B-rotated-mm-bf16/` | Rotated BF16 source ckpt（only as W8A8 producer input；不要直接 deploy 部署，BF16 storage tax 是已知问题） |
| `checkpoints/quantized/Qwen3.6-27B-W8A8-RTN-local/` | **W8A8 all-linear unrot ckpt**（max speedup, quality 不可接受） |
| `checkpoints/quantized/Qwen3.6-27B-W8A8-RTN-local-mlp/` | **W8A8 MLP-only ckpt（RL rollout 推荐）**，attention BF16 |
| `checkpoints/quantized/Qwen3.6-27B-rotated-mm-w8a8-rtn-local/` | W8A8 all-linear rotated ckpt（rotation BF16 storage tax 吃掉收益） |
| `checkpoints/quantized/Qwen3.6-27B-rotated-mm-w8a8-rtn-local-mlp/` | W8A8 rotated MLP-only ckpt（rotation + MLP-only 组合，2026-05-26 新增） |
| `checkpoints/quantized/Qwen3.6-27B-rotated-mm-w8a8-rtn-local-nonla/` | W8A8 rotated MLP + self_attn ckpt（skip linear_attn / mamba layers，2026-05-26 新增） |
| `checkpoints/Qwen3.6-27B/20260524_142451_*_v2_3_env_n8` | **BF16 baseline** (score 0.27875, wall 1:20:33) |
| `checkpoints/Qwen3.6-27B/20260526_020408_*_w8a8-rtn-local-v2_3_env_n8` | W8A8 all-linear unrot (0.21000, 1:01:20) |
| `checkpoints/Qwen3.6-27B/20260526_043626_*_w8a8-rtn-local-rotated-v2_3_env_n8` | W8A8 rotated all-linear (0.23000, 1:06:28) |
| `checkpoints/Qwen3.6-27B/20260526_045323_*_w8a8-rtn-local-mlp-v2_3_env_n8-r3` | W8A8 MLP-only unrot (0.25125, 1:15:21) |
| `checkpoints/Qwen3.6-27B/20260526_065510_*_w8a8-rtn-local-rotated-mlp-v2_3_env_n8` | **W8A8 rotated MLP-only ⭐** (0.26375, 1:17:23) |
| `checkpoints/Qwen3.6-27B/20260526_065737_*_w8a8-rtn-local-rotated-nonla-v2_3_env_n8` | W8A8 rotated non_linear_attn (0.23750, 1:12:50) |

## 历史路径（已废弃 / superseded by 上面 ablation 数据，仅备查）

- **GPTQ-based 路径**（W8A8 + GPTQ on `AutoModelForCausalLM`）：量化成功
  但 sglang dense entry 4 bug 累计。改用 `--multimodal` load + multimodal
  arch tag 绕开。GPTQ 本身离线 3-6h 不适合 colocated online update。
- **GPTQModel weights-only W8 GPTQ_V2**：算法先进（act_group_aware + FOEM），
  但 README 明说 "GGUF and FP8 are weight-only"，不支持 activation
  quantization → 不满足 W8A8 加速目标。
- **SGLang dense entry 死代码**：`Qwen3_5ForCausalLM` 入口含 4 个独立
  bug（EntryClass 未注册、MoE config hardcoded、layers_block_type 命名
  错位、`RadixLinearAttention.forward` signature mismatch）。本分支统一
  走 multimodal entry（`Qwen3_5ForConditionalGeneration`）规避，零
  sglang 改动。
- **不依赖 W8A8 的加速建议（codex top-3，未验证）**：speculative decoding
  NEXTN、`--mamba-scheduler-strategy extra_buffer`、显存压榨
  `--max-running-requests 128 --schedule-policy lpm`。
