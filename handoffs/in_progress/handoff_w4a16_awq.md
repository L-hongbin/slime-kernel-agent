# W4A16 AWQ 单独交接

最后更新：2026-06-01 10:50 CST

本文只记录 Qwen3.6-27B 的 W4A16/AWQ 方向；W8A8、QuaRot、SmoothQuant 结果不要继续塞进这里。

## 结论

- 当前可用证据最完整的是 symmetric `W4A16` MLP-only v2 和 `W4A16_ASYM` MLP-only：量化、静态 gate、离线 probe、`.16` rollout + `.39` reward 的 100x8 full eval 都跑完了。
- v2 runtime 没有明显坏掉：MTP 能加载，EAGLE accept length 正常，server throughput 正常；但模型质量不合格，100x8 score 只有 `0.31750`。
- `W4A16_ASYM` 的 runtime bug 是真实 bug：`.16` 本地 SGLang 之前放行 ASYM checkpoint，却没把 `weight_quant.symmetric=false` 传进 WNA16 scheme，导致 zero-point 被按 symmetric 路径忽略。patch 后 full eval score 提到 `0.35375`，但仍不是合格生产结果。
- 主要问题不是 EAGLE、MTP 或 reward，而是 W4 MLP 量化损失过大。real hidden MLP rel-L2 mean 是 `0.115086`，W8A8 RTN 只有 `0.010690`。
- 旧 `Qwen3.6-27B-AWQ-W4A16-mlp` 的 full eval 结果无效：checkpoint 有非目标 BF16 tensor drift，只能作为坏样本证据。
- RMSNorm `y = norm(x) * (1 + weight)` 已基本排除为主根因：`mlp-rmsfix` 已完成，静态 gate、real hidden probe 全通过；probe 与 v2 只有噪声级差异。

## 当前状态

| 项 | 状态 |
|---|---|
| 主线 checkpoint | `checkpoints/quantized/AWQ/Qwen3.6-27B-AWQ-W4A16-mlp-v2` |
| 主线结果 | full eval 完成，runtime 正常但质量不合格 |
| RMSNorm 对照 | `checkpoints/quantized/AWQ/Qwen3.6-27B-AWQ-W4A16-mlp-rmsfix` |
| RMSNorm 对照结果 | gate/probe 通过；与 v2 等价，不建议为同一配置重复烧 full eval |
| RMSNorm 对照日志 | `checkpoints/quantized/AWQ/logs/awq_w4a16_mlp_rmsfix.log` |
| ASYM runtime patch | `scripts/quantize/patches/sglang_compressed_tensors_wna16_asym.patch` 已应用到 `.16` 的 `/sgl-workspace/sglang` |
| ASYM full eval | Ray job `raysubmit_3RpzfvxksL2Lnmae` 成功；score `0.35375`，run dir `checkpoints/Qwen3.6-27B-AWQ-W4A16-asym-mlp/20260531_233411_awq.w4a16.asym_mlp.sglzpfix.100x8.eagle.rm39_ctx65536_n8_summ1600` |
| full eval 机器 | rollout `.16`：`ssh root@192.168.16.16 -p 23422` |
| reward | `.39`：`http://192.168.16.39:20111`，full eval 时 8 GPU worker + 24 CPU worker online |

## 量化范围

源模型：`/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B`

环境：`.16` 的 `.venv_llmcompressor/`

| 模块 | 状态 |
|---|---|
| `model.language_model.layers.*.mlp.gate_proj` | W4A16 |
| `model.language_model.layers.*.mlp.up_proj` | W4A16 |
| `model.language_model.layers.*.mlp.down_proj` | W4A16 |
| SGLang runtime `gate_up_proj` | 写入 target regex，便于 fused MLP load |
| `self_attn` | BF16 |
| `linear_attn` | BF16 |
| `lm_head` | BF16 |
| `mtp.*` | BF16 |
| 视觉/音频相关模块 | BF16 |

这不是 all-linear W4，也没有量化 linear-attn。结果表里必须写成 `W4A16 AWQ MLP-only`。

## 量化配置

| checkpoint | scheme | quant method | activation | group size | symmetric | AWQ search | 状态 |
|---|---|---|---|---:|---|---|---|
| `mlp-preservefix` | `W4A16` | AWQ | A16 | 128 | true | `duo_scaling=true`, `n_grid=20` | 对照 |
| `asym-mlp` | `W4A16_ASYM` | AWQ | A16 | 128 | false | `duo_scaling=both`, `n_grid=40` | SGLang zero-point patch 后 full eval 完成 |
| `mlp-v2` | `W4A16` | AWQ | A16 | 128 | true | `duo_scaling=both`, `n_grid=40` | full eval 完成 |
| `mlp-rmsfix` | `W4A16` | AWQ | A16 | 128 | true | `duo_scaling=both`, `n_grid=40` | gate/probe 完成 |

校准集：

| 项 | 值 |
|---|---|
| 路径 | `/tmp/calib_ultrachat_200k_train_sft_256_qwen36.jsonl` |
| 样本数 | 256 |
| 来源 | UltraChat |
| max seq length | 2048 |
| 是否 DrKernel/KernelBench eval dump | 否 |

不要用 DrKernel/KernelBench eval dump 当通用质量结论的校准集；如果后续要做代码域校准，必须使用训练或非评测来源并写清楚。

## 已修问题

| 问题 | 证据 | 修复 | 防线 |
|---|---|---|---|
| 非目标 BF16 drift | 旧 checkpoint 的 `input_layernorm`、`self_attn.{q,k}_norm` 等 raw rel-L2 约 `0.006-0.033` | producer 保存后恢复所有应保持 BF16 的非目标 tensor；`post_attention_layernorm.weight` 作为 AWQ smoothing 例外 | `check_awq_w4a16.py --reference-checkpoint` |
| AWQModifier target/ignore 初始化错 | `duo_scaling=both` 下触发 `AttributeError: 'NoneType' object has no attribute 'strategy'` | `AWQModifier` 只负责 smoothing mapping；最终 MLP-only 量化范围放在 `QuantizationModifier` | recipe 单测 |
| 旧 AWQ search 没有真正 identity baseline | 旧 `duo_scaling=True` 的 ratio=0 仍含 `1 / w_mean` 归一化 | 默认改为 `--awq-duo-scaling=both --awq-n-grid=40`，activation-only 分支包含 identity | 单测检查默认 search |
| 大 tensor gate 不够 | 旧 gate 为省内存跳过 `lm_head`/embedding 级别大 tensor 的精确比较 | 改成固定抽样首/中/尾切片比较 | reference gate |
| ASYM 静态通过但 runtime 乱码 | ASYM gate 和离线 dequant probe 通过，但 SGLang 真实加载路径输出异常；上游 vLLM dispatcher 会传 `symmetric=weight_quant.symmetric`，而 `.16` 本地 SGLang 没传。SGLang upstream 主线更保守，直接要求 WNA16 symmetric，避免 ASYM 被静默误载 | 补 `sglang_compressed_tensors_wna16_asym.patch`，让 WNA16 构造拿到 checkpoint 的 `symmetric=false` 并注册/使用 `weight_zero_point`；eval wrapper 改为可选 symmetric-only，并新增 SGLang ASYM runtime gate | 单测覆盖缺失 `symmetric` 会失败；patch 前真实 ASYM gate 失败，patch 后真实 ASYM gate 通过；full eval 成功 |
| RMSNorm offset 风险没有单测 | Qwen/Gemma norm 参数语义是 `1 + weight` | 新增 AWQ smoothing patch 单测；同时确认 llmcompressor oneshot 已有 offset-norm context | `.16` 单测 `16 passed` |

`.16` 单测命令：

```bash
ssh -p 23422 root@192.168.16.16 \
  'cd /nfs/FM/chenshuailin/projects/kernel_agents/slime && \
   .venv_llmcompressor/bin/python -m pytest tests/utils/test_awq_w4a16_producer.py -q'
```

结果：`17 passed in 25.19s`。

## Sanity Gate

v2 full-eval 前静态 gate 已通过：

```text
OK: AWQ W4A16 checkpoint ... has 192 quantized MLP weights,
192 quantized weights total, 15 mtp tensor(s),
820 preserved tensor(s) checked (66 sampled large tensor(s)).
```

`mlp-rmsfix` 静态 gate 也已通过：

```text
OK: AWQ W4A16 checkpoint .../Qwen3.6-27B-AWQ-W4A16-mlp-rmsfix
has 192 quantized MLP weights, 192 quantized weights total,
15 mtp tensor(s), 820 preserved tensor(s) checked
(66 sampled large tensor(s)).
```

含义：64 层 body MLP 的 `gate/up/down` 共 192 个 weight 被量化；`self_attn`、`linear_attn`、`lm_head`、`mtp.*` 没有 W4 量化；非目标 BF16 tensor 与源模型一致。小 tensor 全量比较，`lm_head`/embedding 这类大 tensor 做固定切片抽样比较。

## 离线量化损失

Real hidden MLP probe：先用 BF16 源模型跑真实文本前向，抓选定层 `post_attention_layernorm` 的真实输入，再比较同层 MLP 输出。

| layer | tokens | W8A8 RTN | W4 AWQ sym v2 | W4 AWQ asym |
|---:|---:|---:|---:|---:|
| 0 | 70 | 0.006227 | 0.071166 | 0.071545 |
| 1 | 70 | 0.010507 | 0.117739 | 0.108387 |
| 11 | 70 | 0.012288 | 0.144460 | 0.130810 |
| 31 | 70 | 0.015730 | 0.168508 | 0.151482 |
| 63 | 70 | 0.008697 | 0.073554 | 0.054853 |
| mean | - | 0.010690 | 0.115086 | 0.103415 |

RMSNorm offset 对照：`mlp-rmsfix` 与 `mlp-v2` 使用同一量化范围和同一 AWQ search，差异只用于审计 Qwen/Gemma offset norm 处理是否影响结果。

Real hidden：

| layer | tokens | W8A8 RTN | W4 AWQ rmsfix | W4 AWQ v2 |
|---:|---:|---:|---:|---:|
| 0 | 70 | 0.006227 | 0.071167 | 0.071166 |
| 1 | 70 | 0.010507 | 0.117742 | 0.117739 |
| 11 | 70 | 0.012288 | 0.144451 | 0.144460 |
| 31 | 70 | 0.015730 | 0.168486 | 0.168508 |
| 63 | 70 | 0.008697 | 0.074180 | 0.073554 |
| mean | - | 0.010690 | 0.115205 | 0.115086 |

结论：`rmsfix` 没有改善 W4 MLP loss，说明 `y = norm(x) * (1 + weight)` 不是 v2 低分主因；主要损失仍来自 INT4 MLP 本身。

## Full Eval 结果

运行：`.16` rollout，`.39` KernelGym reward，EAGLE 开启，`--sglang-speculative-num-draft-tokens 4`。

### 与 BF16 / W8A8 基线对比

精度表的分母都是 `in_all=800`；`Fast@*` 也是 in_all，不是 correct-only。BF16/W8A8 数字来自 `handoffs/in_progress/handoff_drkernel_w8a8_rollout.md`，W4A16 数字来自本文对应 `eval_0.pt`。

| Target | 方法 | 范围 | Compile T1/T2/T3 | Correct T1/T2/T3 | Fast@1.0 T1/T2/T3 | Fast@1.2 T1/T2/T3 |
|---|---|---|---:|---:|---:|---:|
| BF16 | - | - | 39.8 / 62.1 / 65.4 | 23.2 / 34.6 / 38.8 | 9.6 / 12.6 / 14.6 | 4.9 / 7.5 / 9.9 |
| W8A8 | RTN | Non-LA | 34.0 / 60.5 / 62.5 | 19.6 / 32.6 / 40.8 | 8.2 / 15.0 / 17.0 | 4.0 / 8.1 / 9.6 |
| W4A16 | AWQ symmetric | MLP-only | 22.8 / 59.9 / 65.6 | 12.1 / 20.5 / 31.9 | 5.5 / 9.2 / 12.5 | 3.6 / 6.0 / 7.9 |
| W4A16 | AWQ ASYM | MLP-only | 20.0 / 57.8 / 65.2 | 10.9 / 21.0 / 35.5 | 3.0 / 7.5 / 14.1 | 1.0 / 4.2 / 8.2 |

100x8 端到端表不是严格同机同 RM A/B：BF16/W8A8 是 `.22/.40` 生产基线，W4A16 是 `.16/.39` 当前排查 run。这里用于定位量级和质量差距；不要只按 wall 排名。

| Target | 方法 | 范围 | draft tokens | RM worker | score | wall | response len mean/median | srv tok/s | accept len | 接收率 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BF16 | - | - | 0 | 8 | 0.37500 | 1:58:37 | - | 2320 | - | - |
| BF16 | - | - | 4 | 16 | 0.38375 | 1:24:37 | 8366.2 / 7968.5 | 2994 | 3.24 | 74.6% |
| W8A8 | RTN | Non-LA | 4 | 16 | 0.40750 | 1:16:12 | 8262.8 / 7720.0 | 3162 | 3.22 | 74.2% |
| W4A16 | AWQ symmetric | MLP-only | 4 | 8 GPU + 24 CPU | 0.31750 | 1:10:55 | 6970.6 / 6675.0 | 2659 | 3.28 | 76.0% |
| W4A16 | AWQ ASYM | MLP-only | 4 | 8 GPU + 24 CPU | 0.35375 | 1:25:49 | 7878.5 / 7185.0 | 2607 | 3.25 | 75.0% |

读法：

- W4A16 ASYM 修掉 zero-point runtime 后确实比 symmetric 好：score `0.31750 -> 0.35375`。
- 但 ASYM 仍低于 BF16 EAGLE `0.38375` 和 W8A8 RTN Non-LA `0.40750`；它不是生产候选。
- symmetric W4A16 wall 看起来最快，但这是低质量/短输出带来的假象；不能把 `1:10:55` 解读成有效加速。
- ASYM 的 accept len/rate 与 BF16/W8A8 同量级，说明 EAGLE 没塌；问题仍在 target checkpoint 的 W4 MLP 量化质量和输出分布。

这里没有 `req>=80` 的 server tput 行，因为这两次 W4A16 full eval 都使用 `SGLANG_MAX_RUNNING_REQUESTS=64`，decode log 里最大 running request 不会到 80。

## 为什么 W4A16 低分

1. 低分在离线 probe 阶段已经能看到，不需要等 full eval。real hidden MLP rel-L2 mean `0.115086` 比 W8A8 RTN `0.010690` 大一个数量级，方向和 full eval 的 `0.31750` / `0.35375` 一致。
2. 这不是 all-linear W4，而是每层 MLP 的 `gate/up/down` 全量 W4。SwiGLU 的 `silu(gate_proj(x)) * up_proj(x) -> down_proj` 会把 `gate` 和 `up` 的误差做乘法耦合，再经过 `down` 放大，比单个 linear 更容易损坏。
3. `group_size=128` 下 W4 每组只有 16 个离散值；W8 有 256 个离散值。AWQ smoothing 只能在校准样本上找局部重参数化平衡，不能把 INT4 信息损失变成无损变换。
4. “AWQ 为什么不搜到 1”不能单独解释问题。v2 已经把默认 search 改成 `duo_scaling=both, n_grid=40`，activation-only 分支包含 identity；但优化目标是局部 MSE，不保证每层或每个任务分布都选 identity，也不保证选到后 full eval 质量就不掉。
5. RMSNorm `1 + weight` 语义需要严肃处理，但当前已基本排除为主根因：llmcompressor 的 `norm_calibration_context` 会在 calibration 期间把 offset norm 转成有效权重，保存时再转回；`mlp-rmsfix` 与 v2 的 real hidden probe 基本一致。
6. ASYM 的 mean loss 较低，修复 SGLang zero-point runtime 后 full eval 也从 v2 的 `0.31750` 提到 `0.35375`。但这只说明 ASYM loader bug 被修掉了，不说明 W4A16 已经可用：`missing_response` 变成 `14/800`，turn1 mean response length 达到 `16112`，整体仍是低分长输出状态。

## 公开 HF 对照

对照来源：

- `https://huggingface.co/mattbucci/Qwen3.6-27B-AWQ`
- `https://huggingface.co/QuantTrio/Qwen3.6-27B-AWQ`

已下载少量 config/index/header 到 `checkpoints/quantized/AWQ/hf_compare/`。

| 来源 | 格式 | group size | zero point / symmetric | 量化范围 |
|---|---|---:|---|---|
| `mattbucci/Qwen3.6-27B-AWQ` | native AWQ GEMM | 128 | `zero_point=true` | 接近所有 Linear，跳过 DeltaNet `in_proj_a/b`、visual、`lm_head`；index/header 中没有 `mtp.*` tensor |
| `mattbucci/Qwen3.6-27B-AWQ-CT` | compressed-tensors | 128 | symmetric | 接近所有 Linear，跳过 DeltaNet `in_proj_a/b`、`lm_head`；index/header 中没有 `mtp.*` tensor |
| `QuantTrio/Qwen3.6-27B-AWQ` | native AWQ GEMM | 128 | `zero_point=true` | 跳过 visual、`linear_attn.in_proj_a/b`、self-attn q/k/v、layer0、MTP |

当前 HF/local metadata 关键计数：

| 来源 | quantized module 口径 | MLP | linear-attn 非 `in_proj_a/b` | self-attn q/k/v | MTP |
|---|---|---:|---:|---:|---|
| 本地 `mlp-v2` | `weight_packed` | 192 | 0 | 0 | 15 个 BF16 tensor |
| `mattbucci` native | `qweight` | 192 | 144 | 48 | index/header 无 `mtp.*` |
| `mattbucci` CT | `weight_packed` | 192 | 144 | 48 | index/header 无 `mtp.*` |
| `QuantTrio` native | `qweight` | 189 | 141 | 0 | 15 个 BF16 tensor |

这个对照不能直接证明本地 MLP-only producer 没问题，因为它们的量化范围、格式、zero-point/symmetric 都不同。但它说明两点：

- 开源 AWQ 没有要求 MTP 必须 W4：`QuantTrio` 明确跳过 MTP；`mattbucci` 两个 artifact 的 index/header 里没有 `mtp.*` tensor。
- 本地 MLP-only scope 比开源 AWQ 更窄；因此“开源 AWQ 能发布”不能推出“本地 MLP-only W4A16 full eval 应该好”。
- `mattbucci` native AWQ 的 config 在 3 天前更新过 ignore list，当前本地 `hf_compare/` 元数据和 HF 页面一致；不要沿用旧版“ignore 为空”的判断。

## 下一步

1. 不建议为 `mlp-rmsfix` 直接重复 full eval：它的 gate/probe 已经和 v2 等价，full eval 大概率只复现 v2 的低分。
2. 后续应转向校准集、group size、只量化 `down_proj`、只量化 `gate/up`、或代码域校准的小样本对照；每个分支先跑 offline probe 和静态 gate，再决定是否 full eval。
3. ASYM 路线可以作为“zero-point runtime 已修”的参考，但不建议继续直接烧同配置 full eval；下一步应该先做更小的 offline probe 对照，尤其比较 `down_proj-only`、`gate/up-only`、不同 group size 和代码域校准。
4. 如果仍需要把 `mlp-rmsfix` 做成正式生产表行，再用 `.16` rollout + `.39` reward 启动 full eval，但这不是当前最有效的排查路径。

full eval wrapper：

`scripts/debug/debug.27b.tp4.eagle.awq_w4a16.sh`

## 证据路径

| 类型 | 路径 |
|---|---|
| 本 handoff | `handoffs/in_progress/handoff_w4a16_awq.md` |
| producer | `scripts/quantize/producers/awq_w4a16.py` |
| RMSNorm AWQ patch | `scripts/quantize/patches/llmcompressor_qwen3_5_awq.py` |
| checkpoint gate | `scripts/quantize/utils/check_awq_w4a16.py` |
| real hidden probe | `scripts/quantize/utils/compare_real_hidden_mlp_quant_loss.py` |
| SGLang ASYM runtime patch | `scripts/quantize/patches/sglang_compressed_tensors_wna16_asym.patch` |
| producer/gate tests | `tests/utils/test_awq_w4a16_producer.py` |
| eval wrapper | `scripts/debug/debug.27b.tp4.eagle.awq_w4a16.sh` |
| HF 对照元数据 | `checkpoints/quantized/AWQ/hf_compare/` |
| `mlp-rmsfix` quant log | `checkpoints/quantized/AWQ/logs/awq_w4a16_mlp_rmsfix.log` |
| `mlp-rmsfix` static gate log | `checkpoints/quantized/AWQ/logs/check_awq_w4a16_mlp_rmsfix.log` |
| `mlp-rmsfix` real hidden probe log/json | `checkpoints/quantized/AWQ/logs/real_hidden_mlp_probe_w4a16_rmsfix_vs_v2.{log,json}` |
| ASYM quant log | `checkpoints/quantized/AWQ/logs/awq_w4a16_asym_mlp_fixscope.log` |
| symmetric v2 quant log | `checkpoints/quantized/AWQ/logs/awq_w4a16_sym_mlp_v2.log` |
| v2 static gate log | `checkpoints/quantized/AWQ/logs/check_awq_w4a16_mlp_v2.log` |
| v2 real hidden probe log/json | `checkpoints/quantized/AWQ/logs/real_hidden_mlp_probe_w4a16_v2.{log,json}` |
| ASYM real hidden probe log/json | `checkpoints/quantized/AWQ/logs/real_hidden_mlp_probe_w4a16_asym.{log,json}` |
| ASYM full eval driver log | `checkpoints/quantized/AWQ/logs/eval_awq_w4a16_asym_sglzpfix_100x8_eagle_driver.log` |
| ASYM full eval run dir | `checkpoints/Qwen3.6-27B-AWQ-W4A16-asym-mlp/20260531_233411_awq.w4a16.asym_mlp.sglzpfix.100x8.eagle.rm39_ctx65536_n8_summ1600` |
| ASYM full eval artifact | `checkpoints/Qwen3.6-27B-AWQ-W4A16-asym-mlp/20260531_233411_awq.w4a16.asym_mlp.sglzpfix.100x8.eagle.rm39_ctx65536_n8_summ1600/dumps/rollout_data/eval_0.pt` |
| ASYM full eval summary/review | `checkpoints/Qwen3.6-27B-AWQ-W4A16-asym-mlp/20260531_233411_awq.w4a16.asym_mlp.sglzpfix.100x8.eagle.rm39_ctx65536_n8_summ1600/dumps/rollout_data/SUMMARY.json` and `sample_*.txt` |
| v2 full eval run dir | `checkpoints/Qwen3.6-27B-AWQ-W4A16-mlp-v2/20260531_150856_awq.w4a16.mlp_v2.100x8.eagle.rm39_ctx65536_n8_summ1600` |
| v2 full eval artifact | `checkpoints/Qwen3.6-27B-AWQ-W4A16-mlp-v2/20260531_150856_awq.w4a16.mlp_v2.100x8.eagle.rm39_ctx65536_n8_summ1600/dumps/rollout_data/eval_0.pt` |
| v2 review samples | `checkpoints/Qwen3.6-27B-AWQ-W4A16-mlp-v2/20260531_150856_awq.w4a16.mlp_v2.100x8.eagle.rm39_ctx65536_n8_summ1600/dumps/rollout_data/sample_*.txt` |
| 旧坏 full eval | `checkpoints/Qwen3.6-27B-AWQ-W4A16-mlp/20260531_093445_awq.w4a16.mlp.100x8.eagle_ctx65536_n8_summ1600` |
