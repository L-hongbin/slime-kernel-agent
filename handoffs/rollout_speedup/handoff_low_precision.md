# Rollout加速： Low Precision推理

默认口径：除非特别说明，本文所有实验都使用 MTP；量化范围也会覆盖对应的 MTP 模块。只有明确写成 BF16 保留、no-spec 或其它例外时，才不按这个默认口径理解。

---

## FP8

### 当前结论

- 精度：可用。score `0.39250`，T3 correct `39.6%`，与 A800 BF16 rm8 baseline 同档。
- 效率：wall time `1:20:10`，快于 BF16 rm8 的 `1:36:09`；

<!-- ### Run

- Run dir: `checkpoints/Qwen3.6-27B-FP8/20260602_022840_newSlimeKG.tp4.eagle.rm16.C96.H20.linear-fi.fp8_ctx65536_n8_summ1600`
- Checkpoint: `/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B-FP8`
- Runtime: H20, TP4, C96, EAGLE draft tokens `4`, context `65536`, `linear_attn_backend=flashinfer`, `attention_backend=fa3`, `sampling_backend=flashinfer`
- Evidence: `run.log`, `eval_config.resolved.yaml`, `dumps/rollout_data/eval_0.pt` -->

### 效率对比

| Target | reward score | wall time | srv tok/s(req>=80) | accept len |
| ------ | -----------: | --------: | -----------------: | ---------: |
| BF16   |      0.38125 |   1:36:09 |               3006 |      3.234 |
| FP8    |      0.39250 |   1:20:10 |               3759 |      3.248 |

<!--
`srv tok/s(req>=80)` 只代表 high-concurrency decode 局部，不含 prefill、reward、
排队和 tail。FP8 run 的 `response_len` mean/median/max/min 是
`7998.0 / 7443.5 / 57550 / 0`，有 1 个 final response length 为 0 的样本。 -->

### 精度对比

<!-- 分母均为 800；`Fast@x` 也是 in-all。 -->

| Target |   Compile T1/T2/T3 |   Correct T1/T2/T3 | Fast@1.0 T1/T2/T3 | Fast@1.2 T1/T2/T3 |
| ------ | -----------------: | -----------------: | ----------------: | ----------------: |
| BF16   | 43.6 / 62.4 / 65.9 | 25.2 / 35.6 / 39.0 | 8.8 / 14.6 / 16.5 |  4.0 / 8.0 / 10.2 |
| FP8    | 48.9 / 71.0 / 74.5 | 26.8 / 36.0 / 39.6 | 9.9 / 13.5 / 15.0 |   4.9 / 8.0 / 8.8 |

FP8 accept length `3.248`，prefix cache hit rate `0.428`，平均 cached tokens/sample
<!-- `4381`，没有看到 EAGLE collapse 或明显 truncation collapse。 -->

<!-- ### 读法

1. FP8 目前是可用质量方向：score 和 turn-level 质量都与 A800 BF16 rm8 baseline 同档。
2. FP8 不是已经证明的 backend 速度方向：wall time 更短，但 device/backend/reward worker
   与输出 token 数都不对齐，且缺少严格 paired 同 seed 对照。
3. 如果继续推进，应补同机同 RM 同 seed 的 BF16/FP8 paired eval，并输出
   per-problem reward、turn-level compile/correct 和长度分布。 -->

## INT8

### 当前结论

W8A8（不量化linear-attn）的精度接近 BF16，效率优于BF16

### 效率对比

| Target | 方法       | 范围     | 粒度        |   score | wall time | 全 turn resp | srv tok/s | accept len |
| ------ | ---------- | -------- | ----------- | ------: | --------: | -----------: | --------: | ---------: |
| BF16   | -          | -        | -           | 0.38375 |   1:24:37 |       26.56M |      2994 |       3.24 |
| W8A8   | RTN        | Non-LA   | per-channel | 0.40750 |   1:16:12 |       26.47M |      3162 |       3.22 |
| W8A8   | RTN        | all      | per-channel |   0.318 |   1:10:45 |       24.87M |      3264 |          - |
| W8A8   | RTN        | Non-LA   | blockwise64 | 0.38125 |   1:39:02 |       27.55M |      2611 |       3.24 |
| W8A8   | SQ+RTN     | MLP-only | per-channel | 0.37375 |   1:17:55 |       26.87M |      3160 |       3.23 |
| W8A8   | QuaRot+RTN | Non-LA   | per-channel | 0.37250 |   1:24:12 |       28.12M |      3086 |       3.15 |

<!-- `*` all-linear run 在 `798/800` 时 detokenizer 被 kill，没有完整 `eval_0.pt`；该行是 run.log partial 口径。 -->

### 精度对比

<!-- 单位为 `%`；分母均为 800。 -->

| Target | 方法       | 范围     | 粒度        |   Compile T1/T2/T3 |   Correct T1/T2/T3 |  Fast@1.0 T1/T2/T3 | Fast@1.2 T1/T2/T3 |
| ------ | ---------- | -------- | ----------- | -----------------: | -----------------: | -----------------: | ----------------: |
| BF16   | -          | -        | -           | 39.8 / 62.1 / 65.4 | 23.2 / 34.6 / 38.8 |  9.6 / 12.6 / 14.6 |   4.9 / 7.5 / 9.9 |
| W8A8   | RTN        | Non-LA   | per-channel | 34.0 / 60.5 / 62.5 | 19.6 / 32.6 / 40.8 |  8.2 / 15.0 / 17.0 |   4.0 / 8.1 / 9.6 |
| W8A8   | RTN        | all      | per-channel | 35.5 / 59.0 / 61.2 | 18.8 / 28.1 / 31.9 |                  - |                 - |
| W8A8   | RTN        | Non-LA   | blockwise64 | 43.1 / 65.0 / 68.2 | 24.9 / 35.1 / 38.6 | 10.2 / 12.6 / 15.2 |   5.2 / 6.4 / 8.9 |
| W8A8   | SQ+RTN     | MLP-only | per-channel | 36.4 / 63.2 / 65.6 | 21.2 / 32.4 / 37.8 |  8.5 / 13.5 / 17.2 |  4.2 / 7.0 / 11.0 |
| W8A8   | QuaRot+RTN | Non-LA   | per-channel | 36.4 / 66.4 / 65.8 | 21.9 / 33.2 / 37.5 |  9.5 / 13.9 / 16.6 |   3.5 / 6.8 / 8.5 |

all-linear 的 Fast@1.0/1.2 无法从 run.log 复算。

### 解释

1. block64 和 QuaRot 的 wall time 变长，主要是因为输出 token 变多了
2. QuaRot 还有一个额外开销：原先主模型和 MTP 可以共用一份 `lm_head`，但 QuaRot 后主模型和 MTP 的权重不再相同，需要复制一份 `lm_head`

<!-- ### 下一步

1. 生产基线保持 per-channel RTN W8A8 Non-LA。
2. blockwise 若继续追，应先做 prefix-conditioned probe，定位哪些 prefix 下
   `### CUDA_KERNELS`、code fence、stop token 或格式 token rank 被压低；不要再只凭
   MLP proxy 或单条 full eval wall time 判断。
3. QuaRot 修复了旧 RMSNorm/MTP accept collapse，但当前分数和 wall time 都不胜出；除非要单独研究旋转量化质量，否则不作为主线。 -->

## INT4

### 当前结论

AWQ量化下，模型精度大幅下降，不可用

### 效率对比

| Target | 方法     | 范围     |   score | wall time | response len mean/median | srv tok/s | accept len | 接收率 |
| ------ | -------- | -------- | ------: | --------: | -----------------------: | --------: | ---------: | -----: |
| BF16   | -        | -        | 0.38375 |   1:24:37 |          8366.2 / 7968.5 |      2994 |       3.24 |  74.6% |
| W8A8   | SQ+RTN   | MLP-only | 0.37375 |   1:17:55 |          8013.6 / 7659.5 |      3160 |       3.23 |  74.4% |
| W4A16  | AWQ ASYM | MLP-only | 0.35375 |   1:25:49 |          7878.5 / 7185.0 |      2607 |       3.25 |  75.0% |

<!-- W4 run: `checkpoints/Qwen3.6-27B-AWQ-W4A16-asym-mlp/20260531_233411_awq.w4a16.asym_mlp.sglzpfix.100x8.eagle.rm39_ctx65536_n8_summ1600`。
它是 `.16` rollout + `.39` KernelGym reward，不是与 BF16/W8A8 的同机同 RM paired A/B。 -->

### 精度对比

| Target          |   Compile T1/T2/T3 |   Correct T1/T2/T3 | Fast@1.0 T1/T2/T3 | Fast@1.2 T1/T2/T3 |
| --------------- | -----------------: | -----------------: | ----------------: | ----------------: |
| BF16            | 39.8 / 62.1 / 65.4 | 23.2 / 34.6 / 38.8 | 9.6 / 12.6 / 14.6 |   4.9 / 7.5 / 9.9 |
| W8A8 SQ+RTN MLP | 36.4 / 63.2 / 65.6 | 21.2 / 32.4 / 37.8 | 8.5 / 13.5 / 17.2 |  4.2 / 7.0 / 11.0 |
| W4A16 AWQ ASYM  | 20.0 / 57.8 / 65.2 | 10.9 / 21.0 / 35.5 |  3.0 / 7.5 / 14.1 |   1.0 / 4.2 / 8.2 |

<!-- T1 上，W8A8 MLP 正确但 W4 错的样本是反向的 `2.26x`；W8A8 MLP 能编译但 W4
不能编译的样本是反向的 `2.25x`。section 缺失只有个位数，不能解释质量差。

T1 failure categories 也指向 API/name/token 级错误，而不是 length 或格式整体 collapse：

| Model           | T1 wrong/fail top categories                                                                   |
| --------------- | ---------------------------------------------------------------------------------------------- |
| BF16            | `compile_other=298`, `compiled_wrong=132`, `no_member=77`, `other_wrong=72`, `kTVMFloat=17`    |
| W8A8 SQ+RTN MLP | `compile_other=234`, `compiled_wrong=121`, `no_member=114`, `kTVMFloat=69`, `other_wrong=59`   |
| W4A16 AWQ ASYM  | `kTVMFloat=190`, `compile_other=175`, `no_member=124`, `not_declared=110`, `compiled_wrong=73` | --> |

<!-- ### 解释链状态

已闭合的部分：

1. Runtime fidelity 已闭合。SGLang WNA16 ASYM zero-point patch 是 load-bearing；
   不打 patch 时 ASYM checkpoint 会按 symmetric `uint4b8` 误载，MLP 权重约
   `53%` rel-L2 损坏。重打 patch 后，identity probe 读出的真实 marlin INT4
   kernel 有效权重与离线 `dequant_w4` 对齐，9 个抽样 tensor 的 rel-L2 约 `3e-5`。
2. W4 的问题不是 EAGLE collapse：accept length/rate `3.25 / 75.0%` 与 BF16/W8A8
   同档。
3. W4 的问题也不是 response length collapse：W4 final response mean/median
   `7878.5 / 7185.0`，T1 response 长度低于 W8A8 SQ+RTN MLP。

未闭合的部分：

1. code-domain MLP-output L2 只给出 W4/(W8-A8) `1.16x`；final-logits L2 `1.23x`；
   final-logits top-1k KL mean `1.47x`，p99 `1.61x`。这些能说明 W4 在 distribution
   body/tail 上更差，但仍低于 T1 compile/correct 的 `~1.7-1.8x`。
2. 无偏 trajectory compounding probe 反而测到 W4-self/BF16-self KL ratio `0.99`，
   且不随位置增长；所以“误差自回归累积放大”这条解释已被证伪。
3. `kTVMFloat` 是相关性信号，不是 token 级因果闭合。在 W4 已生成的错误 prefix 下，
   BF16/W8/W4 都可能更偏 `kTVMFloat`；当前更合理的表述是 W4 更容易走进旧 API/name
   token basin，但具体前缀触发点还没定位。

一句话：W4A16 AWQ 的 target 对 T1 代码有效性确实有稳健负面影响；但当前 evidence
不能把跨机 final score 差距当成已闭合的结构性结论，也不能用一个 aggregate MLP L2
解释 T1 掉点。下一步应转向同 prompt/seed 的受控 A/B 和 prefix-conditioned logprob。

### 下一步

1. 做同机、同 RM、同 seed/prompt slots 的 W4 vs W8A8 MLP vs BF16 paired eval，
   输出 sample/problem 级 final reward、T1 compile/correct 和错误类别转移。
2. 对 T1 失败前缀做 prefix-conditioned logit probe，重点测 `kTVMFloat`、
   `kTVMFFIFloat`、常见 API/name token、section marker、code fence 和 stop token
   的 rank/logprob。
3. 再考虑校准集、group size、只量化 `down_proj`、只量化 `gate/up` 或代码域校准的小样本
   对照；每个分支先跑 offline probe 和静态 gate，再决定是否 full eval。 -->
