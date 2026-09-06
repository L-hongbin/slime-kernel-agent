# Qwen3.8 MTP rollout：负载、容量与正确性

当前真实负载回放中，MTP3 的性能最好：64 并发持续补充请求时，FA3 候选修复版本相对 no-spec 提升约 65% 吞吐，MTP1/2 分别约为 40%/54%。原栈 MTP3 在满长三轮 canary 中发生过 CUDA illegal memory access；本文记录冻结权重 benchmark 的环境与边界，后续补丁、单步 MTP 训练和当前 launcher 用法见[在线训练说明](mtp_training.md)

训练配置和 checkpoint lineage 见[训练配置](training.md)。本文拥有 MTP 的测试口径、容量和社区问题证据；原始产物位于 [mtp_training_workload_20260905](../../local_artifacts/qwen38/mtp_training_workload_20260905/)

## 实验口径

使用实际训练的 DataV4、shuffle seed 42 的前 8 个 prompt groups，每组 16 个 samples。8 个 UUID 都能在已有三轮训练日志中找到。采集调用维护代码的 prompt、精确 token history、precheck、KernelGym 和反馈构造函数，三轮 cap 为 24,576 / 32,768 / 40,960，thinking 为 medium，temperature=1、top-p=1、top-k=-1

隔离 serving 使用原始 `Qwen3.8-27B-FP8`，每引擎 4 张 H20，FA3 attention、Triton GDN/Mamba、FP32 SSM、extra_buffer、memory fraction 0.85、request/graph cap 128，CUDA Graph 开启。每个请求返回 sampled logprob 和 top-20，使用训练的 predictive-support 提取函数核对行数和有限值

MTP head 只有一层。本文的 MTP1/2/3 指重复使用该 head 的草稿深度，对应 `NEXTN`、`steps=1/2/3`、`topk=1`、`draft_tokens=2/3/4`；SGLang 的有效配置会把 NEXTN 显示为 EAGLE

性能对照保留相同输入 token 和相同输出工作量，用 `ignore_eos=True` 固定输出 token 数；因此它测 serving 对给定训练负载的处理速度。自然 EOS 的采集数据用于构造负载，采集墙钟不用于速度比较。固定长度切片先填充相同前缀，命中长度按 64-token 边界对齐；它们仅用于解释长度效应，不能当作训练整体加速比

客户端墙钟包含 HTTP、JSON、sampled/top-20 解析和 predictive-support 检查。SGLang 端时间使用同批请求的最早 `request_received_ts` 到最晚 `request_finished_ts`，包含服务端排队与生成。两种口径都不包含 actor、权重同步和 reward 对完整训练关键路径的影响

## 真实混合请求回放

完整采集了 128 条三轮 trajectory，共 384 个 turn。固定工作量对照从每条 trajectory 取一轮，按 `trajectory_index % 3` 选择，得到首轮/第二轮/第三轮 43/43/42 条请求；包含全部 8 个 prompt groups。128 个输入中有 93 个不同 token 序列，重复项主要来自训练本身的同 prompt 多采样

输入合计 1,657,097 tokens，input p50/p90/max 为 7,409 / 31,254 / 33,060；输出合计 1,050,016 tokens，output p50/p90/max 为 5,665 / 18,839 / 22,935。为所有 arm 统一预留 4 个 draft slots，一共只从自然输出长度中减去了 12 个 token。固定请求见 [mixed_workload.json](../../local_artifacts/qwen38/mtp_training_workload_20260905/mixed_workload.json)，统计与 SHA256 见[负载摘要](../../local_artifacts/qwen38/mtp_training_workload_20260905/mixed_workload_summary.json)

### 持续 64 并发

使用同一冻结请求池和相同确定性顺序，请求完成后立即补入下一条。先进行 30 秒 ramp，再跨约 240 秒的 decoder 日志计数更新边界，计算 `sglang:realtime_tokens_total{mode="decode"}` 的增量；因此未完成请求中已经生成的 token 也会计入，排除了批次排空时并发下降带来的收益放大。窗口结束后只中止本次测试的 request IDs，所有 arm 均正常结束且没有错误

| 配置 | decode tok/s | 相对 node53 no-spec | 实际 running requests 均值 |
| --- | ---: | ---: | ---: |
| no-spec，node53 | 1,973.4 | 1.000× | 63.62 |
| no-spec，node69 对照 | 1,993.0 | 1.010× | 63.60 |
| MTP1 | 2,756.5 | 1.397× | 63.55 |
| MTP2 | 3,033.4 | 1.537× | 63.39 |
| MTP3，原栈 | 3,346.7 | 1.696× | 63.36 |
| MTP3，仅 FA3 页表候选修复 | 3,282.8 | 1.664× | 63.27 |

修复版 MTP3 与同节点 no-spec 对比为 1.647×；两个 no-spec 节点基线相差约 1%，因此结论取约 65%。窗口内 running requests 的 p10 为 63，p90 为 64。该结果说明长尾排空不是主要收益来源；相同生成量对应约 39%–40% 的 serving 时间减少，不等于整个异步训练 step 减少相同比例

这些是单窗口结果，不提供重复运行的置信区间，也不把原栈与修复版 MTP3 约 2% 的差异解释为补丁性能回退。统计和原始时序见 [steady_summary.json](../../local_artifacts/qwen38/mtp_training_workload_20260905/steady_summary.json) 及各 arm 的 `steady_c64.json`

### 128 请求整批完成

下表为同一批 128 个请求全部完成的单次回放，包含批次尾部并发下降的阶段；不能直接当作 fully-async 稳态吞吐

| 配置 | 完成时间 s | output tok/s | 相对 no-spec 吞吐 | 接受长度 | 稳定性边界 |
| --- | ---: | ---: | ---: | ---: | --- |
| no-spec，node53 | 609.05 | 1,724.0 | 1.000× | — | 本次通过 |
| no-spec，node69 对照 | 603.04 | 1,741.2 | 1.010× | — | 基线跨节点差约 1% |
| MTP1 | 434.81 | 2,414.9 | 1.401× | 1.899 | 本次通过 |
| MTP2 | 380.21 | 2,761.6 | 1.602× | 2.659 | 本次通过 |
| MTP3，原栈重启后 | 360.68 | 2,911.2 | 1.689× | 3.299 | 此回放通过，独立三轮 canary 曾崩溃 |
| MTP3，仅 FA3 页表候选修复 | 371.61 | 2,825.6 | 1.639× | 3.292 | 此回放通过，不代表全部 GDN/cache 问题已修复 |

各行输出 token 数完全相同，未发生 retraction。原栈 MTP3 的成功回放发生在引擎重启后，不能用它抵消此前的 CUDA illegal memory access。FA3 候选修复实验只改变页表 headroom，不包含 #35821；其包副本在独立 `fa3_overlay` 中，原安装和正式 launcher 没有修改

### 长度切片

这些切片取自实际生成 token 的精确前缀，每个切片包含 7–8 个不同输入，并重复到指定并发。它们用于解释长度效应；前缀共享比混合后续轮更高，不用于估计整个训练的收益。64 并发组每请求生成 4,096 tokens，128 并发组每请求生成 2,048 tokens，两组之间不是只改变并发的 A/B

| 输入切片 | 并发 | no-spec tok/s | MTP1 增益 | MTP2 增益 | MTP3 增益 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 首轮 1.3K–3.3K | 128 | 3,918 | +5% | +5% | +3% |
| 首轮 1.3K–3.3K | 64 | 3,337 | +13% | +13% | +19% |
| 14K | 128 | 2,120 | +31% | +43% | +42% |
| 14K | 64 | 1,916 | +34% | +51% | +63% |
| 21K | 128 | 1,636 | +41% | +58% | +63% |
| 21K | 64 | 1,552 | +43% | +64% | +85% |
| 32K | 128 | 1,221 | +48% | +63% | +68% |

切片使用未修复栈；在这些有界请求上通过不等于长时间或满 context 安全。先完成的切片仅使用相同前缀 priming；后续 21K/64、32K/128 和混合回放显式先 flush cache，原始 JSON 记录口径。短切片中 MTP1/2 墙钟接近到毫秒级不作为统计等价结论；真实混合回放中二者有明确差距

## 运行时容量

下表来自各引擎 `get_server_info().internal_states[0]`。设置 request cap=128 不代表运行时一定能调度 128 条请求

| 配置 | Mamba slots | 有效 request cap | KV token capacity | target graph GiB |
| --- | ---: | ---: | ---: | ---: |
| no-spec | 952 | 128 | 2,487,147 | 0.41 |
| MTP1 | 667 | 128 | 2,465,540 | 0.79 |
| MTP2 | 583 | 116 | 2,446,748 | 0.85 |
| MTP3 | 519 | 103 | 2,446,748 | 0.92 |

MTP2/3 在 128 个并发请求下包含排队，这是沿用当前 memory 配置启用 MTP 的实际成本。64 并发对照用于分开观察草稿深度和容量影响。不能把减少并发或改变状态池配比后的结果直接与当前 launcher 的 128 上限混为一组

已有三轮日志中，decode running requests 的 p10/p50/p90 为 62/90/109，最大 128；这些是日志采样分位数，受 Ray 日志去重影响，并非按时间加权的完整负载分布。日志中首轮、第二轮、第三轮的 prompt token 中位数约为 1.5K、14K、21K，后两轮 p90 约为 24K、33K

## GDN + MTP 的已知问题

### 已复现的 FA3 context 边界越界

原栈 MTP3 在三轮 canary 的第三轮出现 CUDA illegal memory access，引擎退出，3 条请求未完成。首个错误在 `event_loop_overlap -> process_batch_result_decode -> copy_done.synchronize()` 被报告；由于 CUDA 异步执行，该栈不能单独定位出错 kernel。当时 KV/Mamba 使用率很低，日志没有 OOM

[SGLang #34239](https://github.com/sgl-project/sglang/issues/34239) 报告了同版本 FA3 + NEXTN 在 context 上限附近的相同异常路径。[候选修复 #35985](https://github.com/sgl-project/sglang/pull/35985) 指出，draft-extend/target-verify 的长度会额外包含 draft tokens，而 FA3 静态 page table 仍只按模型 context 分配。本地 `flashattention_backend.py` 确实缺少 headroom

隔离对照只把 FA3 的 `max_context_len` 扩大 `speculative_num_draft_tokens` 并重新计算 page 数，不改变模型 context、采样、Graph、overlap 或 GDN/radix 代码。随后从真实满长生成中截取 40,920-token 前缀，固定生成 36 tokens，分别测试 batch 1/3/4/8，所有请求都使用相同输入文件

| 配置 | 1 / 3 / 4 请求 | 8 请求 |
| --- | --- | --- |
| no-spec，两节点 | 通过 | 通过 |
| MTP1 | 通过 | 通过 |
| MTP2 | 通过 | 通过 |
| 原栈 MTP3，干净重启并完成回放后 | 通过 | 引擎退出，客户端断连 |
| MTP3，仅 FA3 页表修复 | 通过 | 通过 |

原栈 MTP3 在独立三轮和近边界对照中均失败，候选修复通过了相同边界输入、完整混合回放和稳态窗口。代码缺口、触发位置和补丁对照共同支持 FA3 页表越界归因。原栈 MTP3 与候选运行在不同但软件指纹一致的 H20 节点，未完成同一 GPU 的交叉重启重复试验；本结论不声称所有潜在 GDN/cache 缺陷都来自这一处

证据：[boundary_inputs.json](../../local_artifacts/qwen38/mtp_training_workload_20260905/boundary_inputs.json)、各 arm 的 `boundary.json`、原栈 `mtp3/server.log` / `mtp3_retry/server.log`、`fa3_overlay_manifest.json`。#35985 在查询时仍是 open PR，隔离通过不等于已进入当前发布版本

### 已在真实第三轮复现的 context 预算错误

直接沿用训练生成器的 `max_new_tokens = turn_cap - prompt_tokens`，MTP3 的第三轮 8/8 请求被 SGLang 拒绝为 HTTP 400：请求的输入加输出原本恰好为 40,960，SGLang 校验还会计入 4 个 speculative draft slots，实际按 40,964 判断

这是已闭合因果链的接入问题，与 GDN 数值精度问题分开处理。隔离测试在全局 serving context 边界预留 `speculative_num_draft_tokens`；前两轮 cap 低于 serving 上限，不额外截短。真实工作量回放对所有 arm 统一预留 4 个位置，避免 MTP2/3 在回放 no-spec 的满长样本时超界。该 benchmark 当时只在测试 driver 中调整；维护生成器的实现与训练验证见[在线训练说明](mtp_training.md)

复现请求和 HTTP 返回保存在实验目录 `mtp3/api_error_*.json`。接入需要同时处理预算与运行时补丁，单独添加 NEXTN flags 无法完成多轮训练验证

### benchmark 原环境缺少的状态缓存修复

[SGLang #35821](https://github.com/sgl-project/sglang/pull/35821) 于 2026-08-21 合入 `qwen_optimize` 分支，包含两个修复：没有可缓存 token 时直接释放 KV/Mamba slots，避免插入长度为零的 radix 节点；修正 speculative verify 后的 track-step 计算并限制索引上界

该 benchmark 的 node53、node70、node69 源码均保留旧分支：`cache_len is None` 时置零后继续插入，track-step 为 `clamp(tracking_point - seq_lens_pre_verify - 1, min=0)`。它们与实验使用的 extra_buffer + NEXTN 路径相关。已确认的是补丁缺失和路径适用；本次尚未通过补丁 A/B 把某次数值分叉或性能下降归因到这两个缺陷

[SGLang #37326](https://github.com/sgl-project/sglang/issues/37326) 报告 Qwen3.8-Flash-Next 的 MTP 接受率随运行时间衰减，重启后恢复，并怀疑上述 ghost-node 路径。该报告使用 176B MoE、QSA、GB10，不能直接作为本次 27B dense/H20 的故障复现；短 benchmark 也不能证明长时间状态稳定

### 不应混在一起的其它报告

- [#20791](https://github.com/sgl-project/sglang/issues/20791)：FlashInfer GDN 在 no_buffer/in-place state 路径上的精度问题，是维护配置保留 Triton GDN 的背景；本次没有切到 FlashInfer GDN
- [#34786](https://github.com/sgl-project/sglang/issues/34786)：lazy buffer 与 NEXTN 的 track-index 空值错误。本次使用 extra_buffer，当前源码也拒绝 extra_buffer_lazy 与 speculative 的组合
- [#37111](https://github.com/sgl-project/sglang/issues/37111)：Flash-Next 的 QSA + NEXTN 在 GB10 TP2 的 decode graph 上出现静默输出损坏。这里的 27B 使用不同 attention 架构和 GPU，不能直接套用其结论
- [#22311](https://github.com/sgl-project/sglang/issues/22311)：Qwen3.5-27B 的非连续 GDN split view 回归，报告针对更早的 0.5.10；与当前缺失的 radix/verify 修复是不同问题

### Greedy 分叉的证据边界

四条 512-token greedy 检查均成功返回 sampled logprob 和 top-20，人工检查输出是与题目相关的推理。MTP 与 no-spec 在第 9–293 个 token 之间分叉；共同前缀的 chosen-logprob 平均绝对差约为 0.003–0.012。分叉处包含 top-1/top-2 近并列或相同 logprob，MTP2 与 MTP3 的四条输出 token 完全一致

这些观测不足以认定 GDN 状态 bug，也不能证明数学上的 lossless decoding。FP8/BF16 logits 的数值变化、batch shape 和不同 forward 路径都还混在其中。正式启用前应针对同一前缀、同一 token 比较 logprob，并覆盖缓存命中、verify 跨 track boundary、retract/refit 和长时间运行；仅检查 HTTP 200、接受率或短 greedy 输出不够

## 与训练权重同步的边界

所查三轮训练运行参数为 `enable_mtp_training=False`、`mtp_num_layers=None`。当时的 SGLang engine 默认打开 `enable_draft_weights_cpu_backup`，以支持 actor 不训练 MTP 权重的 serving。隔离服务使用冻结的 HF target 和 MTP head，本次没有进行 actor optimizer update 或 refit

因此，测得的接受率只对应这份冻结 checkpoint。启用训练 rollout 的 MTP 不等于必须同时训练 MTP head，但 target 更新后的接受率、draft/shared embedding 的恢复与同步、每 token behavior logprob 的保留，需要另外验证。不能把本次结果外推为训练后期恒定收益

## 复现与审查

- [workload.py](../../local_artifacts/qwen38/mtp_training_workload_20260905/workload.py)：真实请求采集、固定负载回放、top-20 检查和计时
- [serve.py](../../local_artifacts/qwen38/mtp_training_workload_20260905/serve.py)：独立 TP4 服务；各 arm 的 `provenance.json` 保存实际执行命令及文件 hash，复现历史结果以 manifest 为准
- [training_metrics.json](../../local_artifacts/qwen38/mtp_training_workload_20260905/training_metrics.json)、[client_provenance.json](../../local_artifacts/qwen38/mtp_training_workload_20260905/client_provenance.json)：训练统计、数据与客户端代码来源
- [capture_audit.json](../../local_artifacts/qwen38/mtp_training_workload_20260905/capture_audit.json)、[manual_samples.txt](../../local_artifacts/qwen38/mtp_training_workload_20260905/manual_samples.txt)：最终采集维度审计、实际 prompt/反馈/输出及成功样例；不是 MTP 的配对质量评测
- 未修复引擎的 493 个源码/配置指纹一致，另核对 209 个 kernel 源码/动态库指纹一致；FA3 候选副本仅 `flashattention_backend.py` 不同。软件为 torch 2.11.0+cu129、SGLang 标识 0.5.16、transformers 5.12.1、FlashInfer 0.6.14、Triton 3.6.0。包版本号不足以代替源码指纹
- 按 cross-model-review 技能完成只读复核，复核发现有效 request cap、长历史覆盖、跨节点和客户端计时等问题，后续对照分别记录这些变量

采集阶段的适配器曾把 label 当字符串，导致 reference code 缺失和 KernelGym 400；已修正为训练使用的 label dict。另有连接超时中断了部分采集，恢复时复用已保存的精确 token 与原反馈并校验 history 一致。上述采集尝试不用于端到端计时。实验采用独立 rollout 服务和真实请求回放，没有启动完整训练环，也没有改变正式 launcher。收尾时停止了全部测试引擎和专用 SSH tunnel，node53/node69/node70 的 24 张 H20 均确认显存回到 0 MiB
