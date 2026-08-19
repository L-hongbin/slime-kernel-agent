# DeepSeek-V4 MTP/speculative decoding：根因、结论与配置指引

本文说明 BF16 KV-cache、MTP/DSpark speculative decoding 与 LoRA serving 的性能关系，
给出可复现的根因、配置边界和诊断方法

## 工程结论

1. **`28b095c` serving stack（raw FP8 ckpt、8×H20、EP8/DeepEP、DP8、bf16 KV、16 req/rank）
   的 "MTP 无加速"（F1/NS = 0.878× client / 0.951× server）根因是
   `--kv-cache-dtype bfloat16`。** bf16 KV 使 decode 走 Triton sparse-decode
   fallback（`is_bf16_kv` → `flash_mla_with_kvcache_sm120` →
   `flash_mla_sparse_decode_triton`），每 step 比 fp8 KV 的编译版 flash-MLA 慢
   3.3×（101.6 vs 30.4 ms @16 req/rank），step 变为 attention 主导；F1 verify 每
   请求 2 个 query token 使该主导项近似翻倍（+71 ms/iter），iteration cost 1.78×
   ≥ 接受长度 1.70，于是没有收益。同一 stack 换 fp8 KV 后 F1/NS = **1.32×**
2. **已排除**：fp8 routed-expert 格式（fp8 KV 下 raw 的 verify 成本 1.30×，与
   mixed MXFP4 的 1.33× 相同）；spec 实现；EP8/8 卡拓扑（4 卡定量复现
   8 卡端点）；LoRA（B/C = 1.005，LoRA×MTP 交互 ≥ 1）；LoRA/spec 引起生成
   loop（matched-seed N=256：spec on/off 同为 1–2%，是 adapter 的 temp=1 行为）
3. **配置指引**：
   - 在 `28b095c` stack 保持 bf16 KV 时，bulk 并发下 **MTP 关闭是正确配置**；调 draft
     数量/steps 无意义（主导项是 attention 随 verify token 线性放大，
     D2(2/1/3) = 0.89× 与此一致）。低并发 tail（≤8 req/rank）spec 仍为正收益
     （+11%–+52%），如做 batch-size 感知开关可只在低并发启用
   - 重新权衡质量与吞吐时：fp8 KV + MTP 在同一 stack 上为 bf16-no-spec 的
     4.4×（4 卡 @16 req/rank 测量值，非 8 卡预测；fp8 KV 3.3× × MTP 1.32×）
   - dev-dspark `692c5f7d` + standard mixed checkpoint：MTP best
     1.30–1.89×（batch 1–256，官方复现口径），DSpark 1.31–2.72×（accept 约 4.6–4.9）
     该 runtime 的 DSV4 pool 断言 uint8，bf16 KV 无法配置，不存在此陷阱；但其
     KV pool 尺寸有 starvation bug（见陷阱 1）

## 机制：为什么 bf16 KV 杀死 MTP

因果链（node62 GPU0-3 单因素 A/B，runtime `28b095c`+patches，
raw FP8 ckpt，TP4/DP4 dp-attention，EP4/DeepEP，production engine env，CUDA graph
约 99.8%，全部 arm 0 retraction；`raw_oldshape` 与 `raw_ep4` profile 除
`--kv-cache-dtype` 外逐字节相同）：

| KV | global batch | NS tok/s | F1 tok/s | F1/NS | accept | iter cost |
|---|---|---:|---:|---:|---:|---:|
| bf16 | 8 | 259.2 | 284.1 | 1.096× | 1.701 | 1.55× |
| bf16 | 32 | 477.5 | 484.4 | 1.014× | 1.680 | 1.66× |
| bf16 | 64 (16/rank) | 629.9 | 605.2 | **0.961×** | 1.709 | **1.78×** |
| fp8 | 8 | 443.9 | 568.5 | 1.281× | 1.694 | 1.32× |
| fp8 | 32 | 1261.8 | 1620.7 | 1.284× | 1.708 | 1.33× |
| fp8 | 64 (16/rank) | 2107.0 | 2772.9 | **1.316×** | 1.706 | **1.30×** |

与 8 卡 production 端点定量对齐：plain step 101.6 vs 96.0 ms；F1 iteration 180.8 vs
171–183 ms；cost 1.78× vs 1.777×；F1/NS 0.961× vs 0.951×。device-timer 分解
（`forward_execution_seconds_total`，b64 级）：bf16 NS decode 约 105 ms/step、F1
target_verify 约 176 ms/iter（+71 ms，与 8 卡端点的 +74 ms 一致）；fp8 对应约 31 / 36 ms

吞吐恒等式（所有结论用它闭合）：`speedup = accept_len / (spec_iter / plain_step)`
spec 在某 batch 有收益当且仅当 iteration cost 增幅 < accept；attention 主导且
attention 随 query token 线性增长时，verify 翻倍 ≈ cost →1.8×，spec 必败

## Triton sparse-decode kernel 为什么慢（microbench 已量化）

`flash_mla_sm120_triton.py` 的 `_tiled_sparse_decode_kernel` 在 H20 上于
production shape（B=16、H=64、topk=512）实测 **2.06 ms/次调用**：

- 有效带宽（按唯一 KV 字节）**4.1 GB/s = H20 4 TB/s 的 0.1%**；fp32 算力
  约 0.5 TFLOPS，相当于峰值 1%。kernel 的瓶颈位于指令和延迟，字节带宽利用率很低
- **无 MQA head 复用**：grid=(B,H)，64 个 query head 各自重新 gather 同一份
  KV（MLA 只有 1 个 KV head）。实测时间随 H 近线性增长（H=8→64：0.30→2.06 ms，
  8× head → 6.8× 时间）；编译版 flash-MLA 用一次 KV 加载喂 [64,512]×[512,T]
  tensor-core GEMM。计入 64× 冗余流量后实际 load 吞吐也只有约 260 GB/s（6.5%）
- **不用 tensor core**：QK 与 PV 都是 `tl.sum(q*kv)` 的 fp32 CUDA-core
  mul-reduce；tile 只有 16/32 token（autotune 三档），对 topk 做串行循环
- **不因 `topk_length` 提前退出**：循环遍历满 topk，短序列同样付满价
- **fp8 gather 分支比 bf16 更慢**（2.55 vs 2.06 ms）：scale gather + exp2 dequant
  的指令开销超过省下的字节，再次证明瓶颈是指令而非带宽
- 每层 2 次调用（SWA + c4/c128）+ eager LSE merge/sink；约 2.6 ms/层 × 43 层 ≈
  112 ms，与实测 bf16 step 105 ms 闭合，bf16 step 几乎全部时间在这个 kernel 里
- 该 kernel 注释明写为 RTX PRO 6000（SM120）编写，从未为 H20 优化。若需要
  高效 bf16-KV decode，方向是：单 CTA 处理全部 64 head（KV 只读一次 +
  `tl.dot` tensor-core）、按 valid_topk 提前退出、小 B 时 split-KV，预期可到
  带宽约束（约 10–30 µs/调用），或直接使用 fp8 KV 的编译 kernel
- **来源归属**：慢的 kernel 结构是 **upstream sglang** 的（base `28b095c` 与
  dev-dspark 镜像中逐字节存在；upstream 只在真 SM120 硬件上用它作 stopgap）
  本 fork 的修改是 +163/−36 行（`IS_BF16` gather 分支、FP32_PARTIAL_MERGE
  路径），不引入慢因（纯 upstream 的 fp8 分支更慢）；但**把 H20 bf16-KV decode
  路由到该 kernel 的 dispatch（`_is_sm120 or is_bf16_kv`）是本 fork 的 bf16-KV
  patch 加的**，编译版 fp8 kernel 拒绝 bf16 key，当时没有其他 decode 路径

复现：`scripts/dsv4/diagnostics/mtp/triton_decode_microbench.py`（node62 GPU0-3，csl_slime_0614
容器）；原始数字 `local_artifacts/deepseek-v4/fp8_mtp_kv_dtype_results/triton_microbench.json`

## 补充测量

- **低并发 tail（`28b095c` stack，8 卡，bf16 KV，LoRA on）**：F1 在 0.25–8 req/rank
  全程为正（+52% → +11%），无 crossover；crossover 位于 8–16 req/rank。rollout
  drain tail 占 decode wall 约 16–20%
- **dev-dspark + mixed MXFP4（node62 8 卡，官方 DSpark blog 口径）**：MTP best
  1.30×–1.89×（F3 低/中 batch 更快，≥160 起 F1 更快）；DSpark 1.31×–2.72×
  （accept 4.62–4.85）。同一 checkpoint 的 non-spec 曲线与 base checkpoint 相差
  <0.9%。DSpark ckpt 的 `mtp.0/1/2.*` 属于 DSpark stage，与标准 NextN 权重不同
- **dev-dspark 4 卡 mixed**：F1/NS = 1.285×（真实 prompt、t1.0），custom
  all-reduce 与 DP LM head 均非因素（关闭后 1.276×/1.280×）

## LoRA × MTP 共存

Upstream sglang 硬性禁止 LoRA + neural-draft spec；本 fork 的
`sglang_dsv4_mtp_lora_phase1.patch`（8 文件）实现 **adapter-on-target-only**：
adapter 只作用于 target/verify forward，draft（`DeepseekV4ForCausalLMNextN`，
从同一 ckpt 的 `mtp.0.*` 加载）跑 base 权重。正确性由 verify 兜底；唯一可能的
代价是 accept-rate（实测 base 2.213 → adapter 2.292，无退化；greedy
spec-vs-nonspec 0/8 diverge，字节一致）

关键实现点（危险处）：`server_args` 在 target/draft ModelRunner 间共享，所有
runtime LoRA gate 必须改为 per-runner 的
`lora_enabled = enable_lora and not is_draft_worker`，并加 fail-CLOSED 审计
（draft 有 LoRA-wrapped module 时 launch 即 RuntimeError）。V4 hook 只接受算法字面串
`EAGLE`，`NEXTN` alias 会被拒绝。launch guard 同时禁止与 shared-expert LoRA serving
不兼容的 `--enable-deepep-waterfill`。测试 `tests/deepseek-v4/test_dsv4_mtp_lora_phase1.py`；
A/B 已证 LoRA 对 spec serving 增量约 0%（B/C=1.005）

## 诊断陷阱

1. **KV pool starvation 会伪装成 "spec kernel 慢"**。pinned `692c5f7d` 镜像的
   `DSV4PoolConfigurator` 把 C128 state 按 token 比例分配（upstream PR #28612 已
   改为按 request 固定分配）；raw FP8 权重（74.4 GB/rank @TP4）+ draft + spec
   graph 把 pool 压到 swa 6144 tokens/rank 后，SWA 池打满 → retract →
   re-prefill → DP IDLE，吞吐崩到 0.22–0.29×。按
   `accept/speedup` 恒等式得到的 "iteration cost 7.8×/5.9×" 是 workload 级
   有效成本，不能解释成 kernel 成本。**诊断 spec 性能前必须先看 decode log 的
   retract/swa usage/#running-req**；恒等式会把并发损失折进 "成本"
2. **对照实验的每个辅助路径都要拉平**。"bf16 KV 质量更差" 与 "raw FP8
   verify 慢" 的误判都源于对照 arm 携带了未拉平的侧路径（sparse-prefill 强制、
   pool 尺寸）。本次单因素 A/B 用逐字节 profile diff + 0 retract 门槛避免
3. **V3.2-vs-V4 kernel ABI 陷阱**：安装的 `sgl_kernel` sparse-fp8 decode 对外
   Python 入口只接受 V3.2 的 656 B/token 布局且不支持动态 `topk_length`；V4 的
   584 B pool 在 serving 内可用但离线合成调用会被拒，离线复现编译 kernel 需要
   live input capture
4. **测 acc/spec 的 baseline 要用完整 metric 序列**；bench 里约 2% 的 verbatim loop 是 adapter 在
   temp=1 的自身行为，与 spec/graph/overlap 无关（matched-seed N=256 四 arm
   全部 1.17–2.34%，互在噪声内）

## 边界与未决

- bf16-KV Triton kernel 的指令级 stall 分解未做 nsight-compute（方向性结论不
  依赖它）；离线未能直接调用编译版 fp8 kernel（ABI 陷阱 3）
- bf16 KV 的 mismatch 质量收益（约 5%，token-adjusted）与 3.3× decode 吞吐成本
  只适用于支持 full-bf16 KV 的 `28b095c` stack；dev-dspark 的 packed uint8 pool 不支持该配置
- batch-size 感知的 MTP 开关尚未实现；它只对保留 bf16 KV 且 decode tail 占比较高的部署有价值

## 证据与复现

- 最终根因证据：`local_artifacts/deepseek-v4/fp8_mtp_kv_dtype_results/`（bench JSON 逐请求记录、
  manifests、b64 metrics 快照、逐字节 profile diff、triton microbench、
  summary.json）
- DSpark/MTP frontier 证据：`local_artifacts/deepseek-v4/dspark_results/`
- 主复现入口：`scripts/dsv4/diagnostics/mtp/triton_decode_microbench.py` 和
  `scripts/dsv4/diagnostics/mtp/analyze_spec_frontier.py`。节点专用 launch、KV-dtype A/B
  和阶段分析脚本位于 `local_artifacts/deepseek-v4/retired_scripts/diagnostics/mtp/`
