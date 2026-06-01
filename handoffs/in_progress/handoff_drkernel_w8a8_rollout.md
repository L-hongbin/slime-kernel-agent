# DrKernel W8A8 Rollout 交接

最后更新：2026-06-01。

## 结论

- W8A8 + EAGLE 不是“没效果”：accept 正常，fixed-request 仍有 `1.119-1.160x`。
- 量化后 EAGLE 收益变小是合理的：W8A8 已经把 target no-spec decode 变快，EAGLE 能继续省的 target 成本变少；draft、verify、调度和 tail drain 没有同比下降。
- 精度只比较 target model：BF16 vs W8A8。MTP/EAGLE 是 speculative 路径，不单独进精度表。
- 旧 QuaRot + EAGLE 结果无效：Non-LA 和 MLP 两个 20260530 产物 accept 都塌到 `1.00`。根因不是单纯 MTP basis，而是 Qwen/Gemma RMSNorm 语义也错了：有效 scale 是 `1 + weight`，融合后 identity 必须是 `0`，旧产物留成了 `1`。
- 2026-05-31 已修 QuaRot body+MTP 的 Gemma norm 融合，并重建短路径 Non-LA 产物；full eval 已完成，结构问题不再复现旧产物的 `accept=1.00` collapse。
- QuaRot 修复后仍不是当前可用方向：EAGLE run 为 score `0.37250`，wall `1:24:12`，srv tok/s `3086`，final accept `3.145`；弱于 RTN Non-LA 的 `0.40750` / `1:16:12` / `3162` / `3.22`。
- QuaRot no-EAGLE 对照已补：同 checkpoint 关 EAGLE 后 score `0.35750`，wall `1:45:44`，srv tok/s `2560`。因此 QuaRot 的差距不是 EAGLE 拖慢；EAGLE 在该 checkpoint 上仍把 wall 从 `1:45:44` 拉到 `1:24:12`，约 `1.26x`。
- W8A8-G128 的异常不是“G128 本来算得这么慢”：SGLang 侧先缺 A800 blockwise-int8 kernel configs，导致 pure SGLang decode 慢于 BF16；补齐四个 config 后 fixed decode 从 `54.41s/1807 tok/s` 到 `46.02s/2136 tok/s`，快于 BF16 `50.10s/1962 tok/s`。full eval 又暴露默认 `mem_fraction_static=0.85 + chunked_prefill=8192/max_prefill=16384` 在高水位会 logits OOM，旧 G128 日志有 162 个 OOM/router error；改为 `mem=0.82, chunked=max_prefill=4096` 后 full eval 无 OOM，score `0.38875`，wall `1:39:56`，srv tok/s `2767`，accept `3.227`。仍慢于 per-channel RTN Non-LA baseline (`0.40750` / `1:16:12`)，所以不替换基线。
- 2026-06-01 补了当前态 pure-SGLang 固定输出复核：TP4、C32、512 output tokens，G128 no-spec `11.356s / 1443 tok/s`，BF16 no-spec `12.178s / 1345 tok/s`；G128 EAGLE `8.390s / 1953 tok/s`，BF16 EAGLE `9.996s / 1639 tok/s`。这证明短上下文 fixed-output 的 SGLang target/spec decode 侧已不再是 G128 慢于 BF16。
- 2026-06-01 继续补了 BF16 / per-channel W8A8 / G128 的 800 题逐样本逐 turn 对齐：G128 wall 变长的直接原因是生成分布变长，且主要集中在 turn1 长尾，而不是“G128 算子天然慢”。但这不是可接受的“合理波动”，而是 G128 checkpoint 的输出病态：G128 全 turn completion `30.27M`，per-channel `26.47M`，BF16 `26.56M`；G128 相对 per-channel 多 `3.80M` tokens，其中 turn1 贡献 `2.82M`。turn1 `>32768` tokens 的样本数：G128 `168`，per-channel `101`，BF16 `88`；G128 turn1 `>45000` 有 `71` 个，其中相对 per-channel 独有 `69` 个。代表性样例显示 G128 更容易从自然语言分析开始、反复自问自答、很晚或不能进入要求的 `### CUDA_KERNELS` 结构，部分直接 `finish_reason=length`，因此后续 turn 为空；这解释了为什么 per-channel 没有同样 wall 膨胀。
- “G128 量化损失应该小于 per-channel”这个前提在当前实现里不成立。全 263 个量化矩阵对 BF16 反量化 rel-L2：per-channel mean/median/max 为 `0.01037/0.01010/0.01804`，G128 为 `0.01348/0.01245/0.02628`。G128 在 `attn.k/o/v` 和部分 `mlp.down/gate` 上明显更差。原因是当前 SGLang `blockwise_int8` 的 scale 是 `128x128` block 共用（例如 `attn.k` scale `(8,40)`），不是每输出通道一条独立 scale；per-channel 压缩格式则是 row scale（例如 `attn.k` scale `(1024,1)`）。所以 G128 的长尾不是“更低量化损失下反而更差”的悖论，而是一个更粗 scale 方案在输出格式/停止倾向上放大误差的可疑失败。
- 2026-06-01 按要求重做 SmoothQuant + W8A8-G64：checkpoint 为 `checkpoints/quantized/SmoothQuant/Qwen3.6-27B-SQ-W8A8-G64-RTN-nonla-mtp-a0p5-ultrachat`，`linear_attn` 和 `mtp.fc` 保持 BF16，EAGLE 开启，`.22` 上补齐并验证 G64 SGLang blockwise-int8 configs。第一份 G64 eval 因缺 G64 configs 被停掉，只作 diagnostic；有效 tuned run 成功但不胜出：score `0.37500`，wall `1:35:25`，srv tok/s `2597`，accept `3.244`。这不是 EAGLE collapse，也不是 missing-config/OOM；它仍有明显长输出病态，且 high-concurrency decode 本身低于 BF16/per-channel/G128。
- G64+SmoothQuant 比 G128 改善了总生成量但没有解决 wall：G64 全 turn `27.23M` tokens，低于 G128 `30.27M`，但仍高于 per-channel `26.47M` 和 BF16 `26.56M`；turn1 `>32768` 为 `104`，低于 G128 `168`，但仍有 `14` 个 `>60k`。更关键的是 G64 full-run `req>=80` median decode 只有 `2597 tok/s`，低于 G128 `2767`、BF16 `2994`、per-channel `3162`。所以 G64+SQ 不是可接受替代，后续若继续看 blockwise，需要单独做 fixed-output kernel/long-context audit，而不能只看 full eval wall。
- W4A16 AWQ 已拆到独立 handoff：`handoffs/in_progress/handoff_w4a16_awq.md`。这里不再维护 W4 细节，避免两份文档分叉。

缩写：`SQ` = SmoothQuant；`MLP` = 只量化 MLP；`Non-LA` = `MLP+self_attn`，不量化 `linear_attn`；`all` = 全 Linear。

## 精度

单位为 `%`；每列是单 turn 独立统计；`Fast@x` 分母是全部 800 个样本。复算脚本：
`scripts/analysis/per_turn_acc.py <eval_0.pt>`。

| Target | 方法 | 范围 | Compile T1/T2/T3 | Correct T1/T2/T3 | Fast@1.0 T1/T2/T3 | Fast@1.2 T1/T2/T3 |
|---|---|---|---:|---:|---:|---:|
| BF16 | - | - | 39.8 / 62.1 / 65.4 | 23.2 / 34.6 / 38.8 | 9.6 / 12.6 / 14.6 | 4.9 / 7.5 / 9.9 |
| W8A8 | RTN | Non-LA | 34.0 / 60.5 / 62.5 | 19.6 / 32.6 / 40.8 | 8.2 / 15.0 / 17.0 | 4.0 / 8.1 / 9.6 |
| W8A8-G128 | RTN | Non-LA | 46.5 / 65.6 / 67.8 | 27.0 / 34.9 / 39.2 | 11.2 / 15.1 / 15.4 | 6.4 / 8.0 / 7.2 |
| W8A8-G64 | SQ+RTN | Non-LA | 39.6 / 61.6 / 67.4 | 24.2 / 33.1 / 38.2 | 9.8 / 14.8 / 17.2 | 4.5 / 8.0 / 9.9 |
| W8A8 | QuaRot+RTN | Non-LA | 36.4 / 66.4 / 65.8 | 21.9 / 33.2 / 37.5 | 9.5 / 13.9 / 16.6 | 3.5 / 6.8 / 8.5 |
| W8A8 | SQ+RTN | Non-LA | 34.2 / 61.5 / 65.0 | 18.8 / 28.9 / 35.4 | 7.6 / 12.4 / 15.8 | 3.1 / 7.1 / 8.9 |
| W8A8 | SQ+RTN | MLP | 36.4 / 63.2 / 65.6 | 21.2 / 32.4 / 37.8 | 8.5 / 13.5 / 17.2 | 4.2 / 7.0 / 11.0 |

当前读法：

- RTN Non-LA 的 T3 Correct / Fast 不差，但 T1/T2 低于 BF16。
- G128 的 T1/T2 指标仍偏高，但修复后最终 score `0.38875` 仍低于 per-channel RTN Non-LA 的 `0.40750`，且 wall 仍更慢；只有在明确优先 G128 质量信号时才值得继续看。
- G64+SmoothQuant 的 per-turn 指标不差，但最终 score 只有 `0.37500`，wall `1:35:25`，比 BF16/per-channel 都慢；不能用 per-turn 表掩盖端到端长尾和吞吐问题。
- QuaRot+RTN Non-LA 修复后 T3 Correct `37.5%`，没有旧 accept-collapse，但低于 RTN Non-LA 的 `40.8%`；T3 Fast@1.0/1.2 为 `16.6%/8.5%`，也低于 RTN 的 `17.0%/9.6%`。
- SQ+RTN MLP 明显好于 SQ+RTN Non-LA，但仍没有超过 RTN Non-LA；当前不值得继续扩 SmoothQuant。
- W4A16 AWQ 旧结果来自坏 checkpoint，已移出有效精度表；后续状态只看独立 W4A16 handoff。

## Fixed-Request 效率

同形状 probe：`.22`，TP4，`max-running=96`，96 并发，每个请求 2048 output tokens。

| 模型 | EAGLE | wall | compl tok/s | gain | srv tok/s | accept |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 否 | 87.732s | 2241.011 | 1.000x | 3404 | - |
| BF16 | 是 | 70.073s | 2805.776 | 1.252x | 3594 | 2.96 |
| W8A8 all | 否 | 71.443s | 2751.964 | 1.000x | 4071 | - |
| W8A8 all | 是 | 63.865s | 3078.495 | 1.119x | 3858 | 2.89 |
| W8A8 Non-LA | 否 | 74.432s | 2641.428 | 1.000x | 3947 | - |
| W8A8 Non-LA | 是 | 64.150s | 3064.837 | 1.160x | 3838 | 2.95 |

为什么不是 BF16 上的同等倍数：

- BF16 no-spec 慢，EAGLE 省 target decode 的空间大，所以 wall 是 `1.252x`。
- W8A8 no-spec 已经快，EAGLE 的 round/verify/draft 开销占比变大，所以只剩 `1.119-1.160x`。
- accept 没坏：BF16 和 W8A8 都在 `~2.9-3.2`，所以问题不是 draft 质量塌了。

## 100x8 生产结果

| Target | 方法 | 范围 | draft tokens | RM worker | wall | srv tok/s | accept len | 接收率 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| BF16 | - | - | 0 | 8 | 1:58:37 | 2320 | - | - |
| BF16 | - | - | 4 | 16 | 1:24:37 | 2994 | 3.24 | 74.6% |
| W8A8 | RTN | Non-LA | 4 | 16 | 1:16:12 | 3162 | 3.22 | 74.2% |
| W8A8-G128 | RTN | Non-LA | 4 | 16 | 1:39:56 | 2767 | 3.23 | 74.2% |
| W8A8-G64 | SQ+RTN | Non-LA | 4 | 16 | 1:35:25 | 2597 | 3.24 | 74.8% |
| W8A8 | SmoothQuant | Non-LA | 4 | 16 | 1:16:47 | 3169 | 3.24 | 74.6% |
| W8A8 | SmoothQuant | MLP | 4 | 16 | 1:17:55 | 3160 | 3.23 | 74.4% |
| W8A8 | QuaRot | Non-LA | 4 | 16 | 1:24:12 | 3086 | 3.15 | 71.5% |
| W8A8 | QuaRot | Non-LA | 0 | 16 | 1:45:44 | 2560 | - | - |

`num draft tokens` = SGLang `--sglang-speculative-num-draft-tokens`；`0` 表示没有开启 EAGLE/speculative。
`RM worker` = KernelGym reward server 端 worker 数；BF16 no-EAGLE 是旧 8-worker 基线，其余行使用 `.40:20111` 的 16-worker 服务。
`srv tok/s` = decode log 中 `req>=80` 的 median throughput，只代表高并发 decode 局部，不包含 prefill、prefix cache 差异、reward/eval、排队和低并发 tail。
`accept len` = `eval_0` 的 full-run `spec_accept_length`；接收率按 SGLang EAGLE 口径由 `(accept_len - 1) / 3` 折算；`num draft tokens=0` 行为 `-`。
G128 使用 SGLang `blockwise_int8`，`weight_block_size=[128,128]`，动态 activation；`linear_attn` 和 `mtp.fc` 不量化。旧 G128 full eval 的 `srv tok/s` 高但 wall 长，不代表 G128 算子本身必然慢：`srv tok/s` 只取高并发 decode 局部，不包括 prefill、reward/eval、排队、tail，也不反映 engine OOM 后的单边运行。

G128 诊断结论：`.22` 的 SGLang 最初缺四个 A800 blockwise-int8 config (`8704x5120`, `5120x4352`, `3584x5120`, `5120x1536`)，pure SGLang fixed decode 因此从 BF16 的 `50.10s/1962 tok/s` 退到旧 G128 的 `54.41s/1807 tok/s`。补齐 config 后，G128 fixed decode 变为 `46.02s/2136 tok/s`；startup log 已不再报 missing config。2026-06-01 又补了当前态小型 fixed-output audit，G128 no-spec/EAGLE 都快于 BF16，确认短上下文固定输出的 SGLang 侧已修复。之后 full eval 又暴露内存高水位问题：旧默认 token pool `1,524,367`，available GPU mem `6.28GB`，在大 prefill/logits temp 时有 162 个 OOM/router error；新跑法用 `SGLANG_MEM_FRACTION_STATIC=0.82`、`SGLANG_CHUNKED_PREFILL_SIZE=4096`、`SGLANG_MAX_PREFILL_TOKENS=4096`，token pool `1,443,543`、available GPU mem `8.70GB`，full eval OOM 为 0，max queue req `48`。

G128 fixed 的 `1:39:56` 仍慢于 BF16+EAGLE 的 `1:24:37`，这不能被忽略。拆解后主要原因是生成量不同：G128 全 turn response tokens `30.27M`，BF16 `26.56M`，多 `14.0%`；其中 turn1 为 `16.02M` vs `12.81M`，多 `25.1%`。按全 turn 粗算，G128 端到端 token/s 为 `30.27M / 5996s = 5048 tok/s`，BF16 为 `26.56M / 5077s = 5231 tok/s`，token-normalized 只慢约 `3.5%`。剩余差异来自长上下文/高水位时的 G128+EAGLE decode：`req>=80` median `2767 tok/s` vs BF16 `2994 tok/s`，以及 G128 高 token usage decode 行更多 (`usage>=0.95` 为 49 行 vs BF16 1 行)。所以 raw wall 更长是可解释的，但不代表 G128 是更好的生产基线；raw wall 对比必须同时报告 response-token volume。

G128 相对 per-channel 也不是同样的长度分布：per-channel 全 turn `26.47M`，G128 `30.27M`，多 `3.80M`；turn1 `16.02M` vs `13.21M`，多 `2.82M`，解释了绝大多数差异。逐样本看，G128 比 per-channel 总量更长的样本 `490/800`，更短 `310/800`，但长尾非常重：`g128 - per_channel` 总量 p90 `32.0k`、p99 `51.3k`，最大 `81.0k`。turn1 `>32768` 的 G128 样本 `168`，per-channel `101`，其中 `133` 个是 G128 相对 per-channel 独有；turn1 `>45000` 的 G128 样本 `71`，per-channel `47`，但重合只有 `2` 个。样例文件 `top_g128_turn1_longer_examples.md` 和 `length_pathology_examples.md` 中，sample 65/59/184/412/365/605 等是 G128 turn1 直接打满 `~64k` 且 `finish_reason=length`，最终后两 turn 为空；sample 65 的 G128 第一轮约 `199k` chars，开头是自然语言分析，尾部仍在自我校验 tile 逻辑，而 per-channel 同题第一轮是 `3.9k` chars 并直接从 `### CUDA_KERNELS` 开始。当前判断：G128 blockwise-int8 不是 SGLang kernel 慢，而是 checkpoint/量化格式导致 target logits/停止倾向/格式服从被扰动后更容易走向 verbose 长推理/长代码长尾；per-channel W8A8 没有同等长尾，所以 wall 没有同样膨胀。这个现象不应被合理化为正常 production behavior，G128 不应替换基线。

G64+SmoothQuant 的有效 tuned run 不含 missing-config/OOM：G64 configs 已为 `8704x5120`, `5120x4352`, `3584x5120`, `5120x1536` 调好，startup 使用 tuned config，没有 `Using default W8A8` 警告。第一份未调 config 的 run 在约 43% 被停掉，只能说明 G64+SQ 也有长度病态，不能作生产数。有效 run 的全 turn tokens `27.23M`，比 G128 少 `3.03M`，但 wall 仍 `1:35:25`，只比 G128 快 `4:31`，比 BF16 慢 `10:48`，比 per-channel 慢 `19:13`。原因不是 EAGLE 坏掉：accept len `3.244`。更像是两件事叠加：一是仍有长输出长尾（turn1 `>32768=104`, `>60000=14`，turn2/turn3 也有 `>32k`），二是 G64 full-run high-concurrency decode median 只有 `2597 tok/s`，低于 G128 的 `2767` 和 per-channel 的 `3162`。所以 G64+SQ 不能替代 RTN Non-LA；如果继续追，需要单独固定输出/长上下文 microbench 判断 G64 blockwise kernel 是否本身慢。

对比 QuaRot no-EAGLE 时，不能直接用 G128+EAGLE 的 `srv tok/s` 判定端到端 wall：QuaRot no-draft 没有 draft/MTP 显存和 speculative verify 开销，token pool 也更大 (`2,000,958`)；G128+EAGLE 修复后仍是更小 token pool、三轮 multi-turn、reward 和 tail 的组合指标。修复后 G128 wall 从旧 diagnostic 的 `1:55:24` 降到 `1:39:56`，但仍慢于 per-channel RTN Non-LA+EAGLE 的 `1:16:12`。
SQ 两行的 wrapper `rc=1` 来自 Ray 状态收尾，`eval 0` 和 `eval_0.pt` 都完整，按完成记录。
G128 fixed run 的 Ray job 成功；外层 wrapper 因终端状态查询返回 `unknown` 退出 `rc=1`，但 `Job 'raysubmit_gFWnazLJUabafytt' succeeded`、`eval 0` 和 `eval_0.pt` 都完整。旧 G128 `20260531_110823` 仅保留为 diagnostic：它有 OOM/router error，不再作为生产对比数。
旧 QuaRot 20260530 Non-LA/MLP 是 early-stop 失败摘要：`accept=1.00`，0/800；对应旧量化目录和 eval 目录已按要求删除，不再作为可复跑对象。

## W4A16 AWQ 异常记录

已拆到独立 handoff：`handoffs/in_progress/handoff_w4a16_awq.md`。

本文件只保留这个入口，不重复维护 W4A16 的 checkpoint、probe、gate 和下一步。

## MTP / EAGLE 状态

- 已修旧 bug：SGLang MTP 模块名是 `mtp.layers...`，旧 ignore 正则漏掉开头的 `mtp`，会把 draft head 错量化。
- 已修 checkpoint 问题：HF `save_pretrained` 会丢 `mtp.*` tensors，现在恢复为 `model-mtp.safetensors` 后再跑 RTN。
- 已修 QuaRot MTP basis 问题：`mtp_checkpoint.py` 可按 R1 旋转 MTP，并把 `mtp.norm` 精确融合到独立 `mtp.lm_head.weight`；`.22` 的 SGLang 已应用 `sglang_qwen3_5_mtp_separate_lm_head.patch`，只共享 embedding，不再覆盖 checkpoint 提供的 draft head。
- 已修 QuaRot Gemma RMSNorm 问题：body 和 MTP 的 norm fusion 统一使用 `1 + weight`，融合后 norm weight 置 `0`。旧单测只测 MTP 自洽，不足以发现 body `lm_head`/norm 留错；现在已加负向单测，坏 checkpoint 会在 `check_mtp_rotation.py` 阶段失败。
- 静态校验：Non-LA 共 263 个 INT8 Linear（body 256 + MTP 7）；MLP 共 195 个 INT8 Linear（body 192 + MTP 3）；`linear_attn` 和 `mtp.fc` 保持 BF16。
- G128 静态校验同样是 263 个 `weight_scale_inv`，其中 MTP 7 个；`linear_attn_scale_inv_count=0`，`mtp.fc.weight` 保持 BF16。
- 新 gate：`check_mtp_rotation.py` 会在 full eval 前拦住缺 `mtp.lm_head.weight`、MTP 未旋转、body `lm_head` 未按 `1+norm` 融合、fused norm identity 非 0、或 W8A8 MTP int8 反量化偏差过大的 checkpoint。旧短路径 W8A8 被该 gate 拦住；重建后的短路径 Non-LA W8A8 已通过。
- 当前 QuaRot Non-LA full eval 说明：结构 gate 只能证明 checkpoint 没有明显融合/旋转错误，不能保证 EAGLE 或下游 score 更好。EAGLE/no-EAGLE 对照说明 EAGLE 本身仍有正收益，但该 checkpoint 的 score、accept、srv tok/s、T3 Correct 都低于 RTN；下一步要看 logit/KL 和 draft-target agreement，而不是继续直接 full eval。

## 证据路径

| 类型 | 路径 |
|---|---|
| handoff | `handoffs/in_progress/handoff_drkernel_w8a8_rollout.md` |
| W4A16 handoff | `handoffs/in_progress/handoff_w4a16_awq.md` |
| RTN producer | `scripts/quantize/producers/rtn_w8a8.py` |
| RTN G128 producer | `scripts/quantize/producers/rtn_w8a8_g128.py` |
| SQ producer | `scripts/quantize/producers/smoothquant_w8a8.py` |
| MTP checkpoint 工具 | `scripts/quantize/utils/mtp_checkpoint.py` |
| QuaRot MTP repair | `scripts/quantize/rotation/repair_mtp_rotation.py` |
| QuaRot MTP sanity | `scripts/quantize/rotation/check_mtp_rotation.py` |
| SGLang MTP patch | `scripts/quantize/patches/sglang_qwen3_5_mtp_separate_lm_head.patch` |
| BF16 no-EAGLE eval | `checkpoints/Qwen3.6-27B/20260528_234734_ctx65536_n8_summ1600_nonla_emfrac09_newSlimeKG_tp4` |
| BF16 EAGLE eval | `checkpoints/Qwen3.6-27B/20260529_132454_newSlimeKG.tp4.eagle.rm16_ctx65536_n8_summ1600` |
| QuaRot BF16 short ckpt | `checkpoints/quantized/QuaRot/Qwen3.6-27B-QR-BF16-mtp` |
| QuaRot Non-LA W8A8 short ckpt | `checkpoints/quantized/QuaRot/Qwen3.6-27B-QR-W8A8-nonla-mtp` |
| QuaRot Non-LA completed eval | `checkpoints/Qwen3.6-27B-QR-W8A8-nonla-mtp/20260531_002353_quarot.nonla_mtp.gemmafix_ctx65536_n8_summ1600` |
| QuaRot Non-LA no-EAGLE eval | `checkpoints/Qwen3.6-27B-QR-W8A8-nonla-mtp/20260531_023637_quarot.nonla_mtp.gemmafix.nospec_ctx65536_n8_summ1600` |
| QuaRot rebuild logs | `checkpoints/quantized/QuaRot/logs/` |
| fixed-request probe | `checkpoints/Qwen3.6-27B/w8a8_eagle_probe_20260530/` |
| G128 W8A8 ckpt | `checkpoints/quantized/RTN/Qwen3.6-27B-W8A8-G128-RTN-nonla-mtp` |
| G128 static scope | `checkpoints/quantized/RTN/Qwen3.6-27B-W8A8-G128-RTN-nonla-mtp/static_scope_summary.txt` |
| G128 reference validation | `checkpoints/quantized/RTN/Qwen3.6-27B-W8A8-G128-RTN-nonla-mtp/validation_32tensor.txt` |
| G128 pure SGLang audit | `checkpoints/Qwen3.6-27B/g128_sglang_audit_20260601/` |
| G128 length analysis | `checkpoints/Qwen3.6-27B/g128_length_analysis_20260601/` |
| G64 SQ length analysis | `checkpoints/Qwen3.6-27B/g64_smooth_length_analysis_20260601/` |
| G128 length script | `scripts/debug/analyze_g128_lengths.py` |
| G128 weight rel-L2 compare | `checkpoints/Qwen3.6-27B/g128_length_analysis_20260601/weight_rel_l2_compare.txt` |
| G128 pathology examples | `checkpoints/Qwen3.6-27B/g128_length_analysis_20260601/length_pathology_examples.md` |
| G128 old diagnostic eval | `checkpoints/Qwen3.6-27B-W8A8-G128-RTN-nonla-mtp/20260531_110823_w8a8.g128.nonla_mtp.100x8.eagle_ctx65536_n8_summ1600` |
| G128 fixed eval | `checkpoints/Qwen3.6-27B-W8A8-G128-RTN-nonla-mtp/20260531_150558_w8a8.g128.nonla_mtp.sglcfg.mem82.cp4096.100x8.eagle_ctx65536_n8_summ1600` |
| SQ G64 W8A8 ckpt | `checkpoints/quantized/SmoothQuant/Qwen3.6-27B-SQ-W8A8-G64-RTN-nonla-mtp-a0p5-ultrachat` |
| SQ G64 static scope | `checkpoints/quantized/SmoothQuant/Qwen3.6-27B-SQ-W8A8-G64-RTN-nonla-mtp-a0p5-ultrachat/static_scope_summary.txt` |
| SQ G64 tuned eval | `checkpoints/Qwen3.6-27B-SQ-W8A8-G64-RTN-nonla-mtp-a0p5-ultrachat/20260601_013236_smooth.g64.nonla_mtp.sglcfg64.mem82.cp4096.100x8.eagle.rm16_ctx65536_n8_summ1600` |
| SQ G64 quant/tune logs | `checkpoints/quantized/SmoothQuant/logs/sq_w8a8_g64_nonla_mtp_20260601_081350.log`, `checkpoints/quantized/SmoothQuant/logs/g64_sglang_tune_20260601_011540.log` |
| 新实验 root | `checkpoints/Qwen3.6-27B/quarot_smoothquant_100x8_20260530/` |
| SQ Non-LA run | `checkpoints/Qwen3.6-27B-SQ-W8A8-RTN-nonla-mtp-a0p5-ultrachat/20260530_153143_smooth.nonla_mtp.100x8.eagle_ctx65536_n8_summ1600` |
| SQ MLP run | `checkpoints/Qwen3.6-27B-SQ-W8A8-RTN-mlp-mtp-a0p5-ultrachat/20260530_165501_smooth.mlp_mtp.100x8.eagle_ctx65536_n8_summ1600` |

## 下一步

1. 当前可用基线继续用 W8A8 RTN Non-LA + EAGLE；G128 fixed 已无 OOM，score `0.38875`，但 wall `1:39:56` 仍慢于 per-channel RTN Non-LA 的 `1:16:12`，所以不替换基线。后续如再跑 G128，必须沿用 tuned SGLang configs + `mem82/cp4096`。
2. SmoothQuant 暂停扩展：MLP 精度接近 BF16，但没有超过 RTN Non-LA；G64+SQ 也没有解决 wall，score `0.37500`、wall `1:35:25`，且 high-concurrency decode median `2597 tok/s` 偏低。
3. QuaRot Non-LA EAGLE/no-EAGLE 都已完整跑完但不胜出；先做 logit/KL 与 draft-target agreement 小样本 sanity，再决定是否补 MLP。旧 20260530 QuaRot W8A8 不要再跑 full eval。
4. W4A16 AWQ 后续只维护在 `handoffs/in_progress/handoff_w4a16_awq.md`；不要在本文继续追加 W4 细节。
