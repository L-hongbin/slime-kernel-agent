# KernelBench L1 模型精度登记

统一存放各模型在 **KernelBench Level 1**(tvm_ffi backend,KernelGym 评测)上的精度,供横向对比与复现。新增模型在「汇总」表加一行,并补一节「逐模型明细」。

## 汇总

指标定义:compile=编译通过;correct=数值正确;fast@1.0 / 1.2=正确且相对 torch 参考 speedup ≥ 1.0 / 1.2。口径均为 `summarize_eval.py`(逐 trajectory)。

| 模型 | checkpoint | backend / ctx=resp / turns / n | compile | correct | fast@1.0 | fast@1.2 |
|---|---|---|---:|---:|---:|---:|
| Qwen3.6-27B(slime RL,kernel-agent full-async) | iter_39(HF) | tvm_ffi / 16384 / **1** / 8 | **59.75%** | **35.00%** | **5.62%** | **4.38%** |
| **Step-3.7-Flash**(198B MoE VLM,language-only) | base(未训练) | tvm_ffi / 32768 / **3(best)** / 8 / high + MTP | **31.25%** | **18.88%** | **5.12%** | **4.12%** |
| **gpt-oss-120b**(117B MoE,base 未训练,mxfp4) | base | tvm_ffi / 32768 / **3(best)** / 8 | **49.75%** | **39.62%** | **12.75%** | **7.75%** |
| DeepSeek-V4-Flash(MoE,fp8) | base | tvm_ffi / 32768 / 3 / 8 | — | — | — | — |

> **不可直接横比**:Qwen iter_39 是 **单轮**(max-turns=1);Step-3.7-Flash 与 gpt-oss-120b 是 **3 轮取最好**。turns/ctx 不同,只看绝对水平。
> **gpt-oss-120b 口径注意**:表内为 n=800(含 144 个 wall-clock 超时 trajectory 记 0)的**保守**口径;剔除超时(分母≈656)为 compile 60.7 / correct 48.3 / fast@1.0 15.5 / fast@1.2 9.5。超时是 mxfp4 在 Ampere 上解码慢所致,非质量问题(见明细)。
> **DeepSeek-V4-Flash 无结果**:sglang `deepseek_v4` 是 Hopper(sm90)专用,A800 跑不起来;脚本/模板/投机已就绪,待上 H20(见明细)。

---

## Qwen3.6-27B — iter_39(slime RL,kernel-agent full-async)

### Takeaway
- 单轮(turn1)即达 **compile 59.75% / correct 35.00% / fast@1.0 5.62% / fast@1.2 4.38%**,正确率与编译率显著高于未训练基座类模型,说明 RL 训练已让模型掌握 tvm_ffi 绑定格式。
- 瓶颈在「快」而非「对」:correct 35% 但 fast@1.0 仅 5.62% —— 多数正确 kernel 慢于 torch 参考(speedup 均值 0.32)。

### 精度(summarize_eval.py,800 trajectories = 100 题 × 8 samples,max-turns=1)

| n | compile | correct | fast@1.0 | fast@1.2 |
|---:|---:|---:|---:|---:|
| 800 | 59.75% | 35.00% | 5.62% | 4.38% |

RolloutManager 内部指标(口径与 summarize 略不同,供参考):综合分 `eval/kb_l1_val`=**0.464**;precheck_pass_rate=0.91;speedup mean=0.316(max 18.0);response_len mean=8944;truncated_ratio=0.05。

### 已知异常
- **`spec_accept_rate=0.0`**:EAGLE 投机解码在本次 eval 完全没命中(训练时约 0.6–0.7)。只拖慢生成、**不影响** compile/correct/fast 等正确性指标。怀疑转出的 HF 在 SGLang eval 路径下 EAGLE 草稿(MTP)未生效,待排查。

### 来历与复现
- **checkpoint 来历**:full-async 训练(`examples/kernel_agent/run.t1.qwen3.6.27B.full-async.sh`,EXP=`FAsync.tvm_ffi.Qwen3.6-27B.CTX16384`)在 iter≈89 因**磁盘写满**崩溃(异步存盘 iter_79 时 3 个 shard 被截断而损坏);iter_39 是最后一个完整 checkpoint(iter_59 目录已丢、iter_79 损坏)。
- torch_dist → HF:32 个 `.distcp` shard 原本分裂在 node64(`__0`–`__7`)与 node69(`__8`–`__15`),gather 到一处后用 `tools/convert_torch_dist_to_hf_parallel.py --vocab-size 248320` 转换,产出 11 safetensors / 866 tensors / 54.6GB(含 lm_head + MTP)。torch_dist 原始 ckpt 已删,仅保留 HF。
- **评测**:`examples/kernel_agent/eval.t1.qwen3.6.27B.sh`(`EVAL_HF_CKPT=.../hf/iter_39`),node69(10.11.2.169)8×H20,2 引擎 TP4,`--debug-rollout-only`,KernelGym `127.0.0.1:20211`。关键参数:`ctx=resp=16384`、`max-turns=1`、`n-samples-per-eval-prompt=8`、`enable_thinking`、level1 验证集、EAGLE/linear-attn/mamba(Qwen 专属 flag)。

<!-- HF: experiments/FAsync.tvm_ffi.Qwen3.6-27B.CTX16384/hf/iter_39 (node69)
     评测产物: experiments/EvalFAsync.tvm_ffi.Qwen3.6-27B.CTX16384/iter_39/{summary.20260622.122238.txt, 20260622.122238.log, dumps/rollout_data/eval_0.pt} (node69) -->

---

## Step-3.7-Flash(198B MoE VLM,language-only)

> ⚠️ 前置:本 checkpoint 必须用 `fix_mistral_regex=True` 加载 tokenizer(脚本已加 `--tokenizer-load-kwargs '{"fix_mistral_regex": true}'`),否则 transformers v5 默认 tokenizer 会丢掉全部空格/换行、喂给模型乱码。原理与判别见 memory `transformers-v5-tokenizer-whitespace-bug`。

### 精度(summarize_eval.py;**所有列(Tk 与 best)统一用 800 traj = 100 题 × 8 做分母**;单位 %;best = 同 trajectory 跨 3 轮取最好)

单位 %;每个指标 4 列 = T1 / T2 / T3 / best。Tk = 该轮满足条件的 trajectory 数 ÷ 800;best = 任意轮满足的 trajectory 数 ÷ 800,Tk 与 best 同母体可直接横比:

| effort | compile T1 | compile T2 | compile T3 | compile best | correct T1 | correct T2 | correct T3 | correct best | fast@1.0 T1 | fast@1.0 T2 | fast@1.0 T3 | fast@1.0 best | fast@1.2 T1 | fast@1.2 T2 | fast@1.2 T3 | fast@1.2 best |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| high | 1.00 | 14.62 | 22.50 | **31.25** | 0.62 | 7.38 | 13.12 | **18.88** | 0.12 | 1.88 | 4.00 | **5.12** | 0.12 | 1.62 | 3.25 | **4.12** |
| medium | 2.38 | 15.62 | 21.88 | **30.38** | 1.12 | 8.75 | 11.62 | **17.88** | 0.00 | 1.75 | 3.50 | **4.88** | 0.00 | 1.38 | 2.75 | **3.75** |
| low | 2.50 | 15.25 | 23.62 | **32.00** | 0.62 | 8.75 | 11.62 | **18.12** | 0.38 | 1.88 | 3.62 | **5.25** | 0.12 | 1.62 | 2.50 | **3.62** |

- **分母口径(2026-06-24 更新)**:Tk 与 best 现统一以 800 traj 为分母(旧版 Tk 各用「达到该轮的 trajectory 数」做分母,数值偏高;best 列不变)。各轮实际产出的 trajectory 数(present,仅信息展示,summarize_eval.py 仍打印)high 800/792/725、medium 800/779/702、low 800/783/744。重算脚本 `examples/kernel_agent/fix_old_summarize_eval.py` 已把新口径表 append 到各 `summary*.txt`。非截断口径 high best:compile 34.34 / correct 21.10。

### 解读
- **三档成绩都在噪声内**(compile ~31% / correct ~18% / fast@1.0 ~5%):reasoning_effort 不改变最终成绩。
- **reasoning_effort = soft prompt(真实但非单调)**:turn-1 思考长度 low(21.5k 字符)< high(28.4k)< medium(38.2k),非 low<med<high。统计极显著(codex:配对 t=9.36,p≈1e-15,d=0.94,86/100 prompt medium 比 high 长)但方向无意义——base 权重没把档位词和算力对齐,只是预训练联想。
- **瓶颈 = TVM-FFI 绑定 API 用错**(`tvm_ffi_api_misuse` 主导)——真实能力上限(没在这套绑定格式上训练),非配置问题(codex + 自查双确认:解析不丢码、工具链正常、容差不误杀、summarizer 计数正确)。
- **fast 低**:正确 kernel 多数慢于 cuBLAS-backed torch(median speedup 0.89×,28% 快过 torch)。
- **最大可恢复杠杆**:ctx=resp=32768 多轮上下文耗尽(~9% trajectory 截断);放宽到 65536 codex 估最多 +9pp。

<!-- 评测目录: ...n8.rehigh.mtpON / ...n8.remedium.mtpON / ...n8.relow.mtpON (各 800 traj) -->
### 复现配置
- 脚本:`examples/kernel_agent/eval.step37flash.l1.sh`(基于 `eval.t3.qwen3.6.27B.sh` 改)
- 运行:node64(10.11.2.164),8×H20,单引擎 **TP=8**;KernelGym 在 `127.0.0.1:20211`(编译/bench 实际在 A800)
- 默认即全量:`bash examples/kernel_agent/eval.step37flash.l1.sh`;smoke 用 `EVAL_NUM_PROMPTS=1/8`
- 关键参数:`ctx=resp=32768`、`reasoning_effort=high`、`max-turns=3`、`n-samples-per-eval-prompt=8`、level1 验证集

针对 Step3p7(非 Qwen)的改动要点:
- `MODEL_ARGS` 置空——`--debug-rollout-only` 下不建 Megatron、跳过 hf 校验(已验证)。
- **language-only**:不传 `--sglang-enable-multimodal`,KernelBench 纯文本,视觉塔不参与;但 sglang 把 step3p7 归类多模态,**视觉权重仍会加载**(~4GB 死重,显存够)。无法对 step3p7 走真·纯文本加载。
- 去掉 Qwen 专属 sglang flag(EAGLE / linear-attn / mamba);step3 思考由 `reasoning_effort`(low|medium|high)控制,非 Qwen 的 `enable_thinking`。
- `RAY_agent_register_timeout_ms=180000`:绕过 raylet 等 dashboard-agent 30s 超时崩溃(KernelGym 打满 GPU 时 nvidia-smi 探测变慢)。

<!-- 评测目录: experiments/Eval.TVMFFI.Step3.7-Flash.LangOnly.Step-3.7-Flash.l1.ctx32768.resp32768.turn3.n8.rehigh/Step-3.7-Flash/
     summary.20260622.133502.txt、review_results.md(拆分+归因,含 codex PASS 复核)、dumps/rollout_data/eval_0.pt
     已知可改进项:脚本结尾不自动 ray stop;相同配置重跑会覆盖 dump;ray stop --force 会无差别停掉本机任何 ray。 -->

---

## gpt-oss-120b(117B MoE,base 未训练,mxfp4)

### Takeaway
- best-of-3:**compile 49.75% / correct 39.62% / fast@1.0 12.75% / fast@1.2 7.75%**(n=800 含 abort 的保守口径)。逐轮单调上升(correct 7.0→23.0→28.75%),**多轮显著有效**。
- 未训练基座里成绩偏高(correct best 39.62% > Step-3.7-Flash 18.88%),能产出高质量 kernel(单样本 speedup 达 **7.07×**);harmony 响应解析无问题(env 正确抽 `### CUDA_KERNELS`/`### MODEL_NEW`)。
- **本次跑得极慢,但不影响质量数**:mxfp4 在 A800(sm80)无硬件支持,解码 ~15 token/s,致 18% trajectory 撞 50min 看门狗被记 0(压低 800 口径)。修复见下。

### 精度(summarize_eval.py,800 traj = 100 题 × 8,max-turns=3;Tk 与 best 均以 800 为分母)

| 指标 | T1 | T2 | T3 | best |
|---|---:|---:|---:|---:|
| compile | 12.88 | 33.50 | 38.88 | **49.75** |
| correct | 7.00 | 23.00 | 28.75 | **39.62** |
| fast@1.0 | 1.88 | 5.62 | 10.75 | **12.75** |
| fast@1.2 | 1.62 | 3.62 | 6.50 | **7.75** |

- **非截断口径(剔除 144 个 abort trajectory,分母≈656)**:compile 60.7 / correct 48.3 / fast@1.0 15.5 / fast@1.2 9.5。
- RolloutManager 综合分 `eval/kb_l1_val`=**0.3225**;response_len mean 2798;reward max 2.5;470/2106 turn-sample reward>0。

### 已知问题与修复
- **慢的根因 = mxfp4 在 Ampere 无优化(非配置问题)**:gpt-oss 是 mxfp4(4-bit),官方要求 compute capability ≥ 9.0(H100/B100);A800=sm80 无 fp4 硬件,sglang 用未优化的 `triton_kernels` mxfp4 MoE(启动印 `mxfp4 quantization is not fully optimized yet`),单序列 ~15 tok/s(应 100+)。**Marlin 救不了**——sglang 的 MXFP4 Marlin 同样要 sm90(`mxfp4.py`:`raise RuntimeError("MXFP4 Marlin requires Hopper/SM90 or above.")`)。
  - **修复 = 转 bf16**:`tools/preprocess_gpt_oss.py --input <ckpt> --output <ckpt>-bf16`(反量化,数值等价、质量不变),用 **TP=8**(bf16≈240GB)跑,native bf16 张量核快 5–10×、abort 基本消失。可叠加 **EAGLE3 投机**(公开草稿 `lmsys/EAGLE3-gpt-oss-120b-bf16`,`--speculative-algorithm EAGLE3 --speculative-num-steps 3 --speculative-eagle-topk 1`)再 ~2–3×。
- **144 个 abort 全是 wall_clock_timeout**:guard=`KERNEL_AGENT_GENERATE_GUARD_SEC`=client_timeout(2400)+task(300)+300=**3000s(50min)**;143 个跑完 2 轮卡在第 3 轮。时间 99% 花在生成(total_model_time 均值 2272s),KernelGym 仅 ~16s(env_time 中位 1.8s,p99 196s)——慢在生成,不在评测。
- **harmony 多轮修复(behavior-sensitive,已加单测)**:gpt-oss 响应带字面 `<|channel|>`(skip_special_tokens=False),直接回填 assistant 历史会让下一轮 `apply_chat_template` 抛 `TemplateError`、每个样本第 2 轮崩、结果全 0。修复 `generate_with_cuda_agent.py:_sanitize_assistant_history_content`(回填前抽 final-channel、剥 eos,非 harmony 为严格 no-op);单测 `examples/kernel_agent/test/test_sanitize_harmony_history.py`(10 例,过 codex review)。
- **EAGLE3 本次未启用(疏漏)**:误判 gpt-oss 无草稿而删了投机;实际有公开 EAGLE3 草稿(见上),应启用。

### 复现配置
- 脚本:`examples/kernel_agent/eval.gpt-oss-120b.sh`;模型 args `scripts/models/gpt-oss-120b.sh`(`--debug-rollout-only` 下仅解析、不建 Megatron)。
- 运行:**.22(192.168.16.22)**,8×A800-80G,本次**单引擎 TP=8**(脚本现默认 TP=4——mxfp4 仅 ~63GB;**若改 bf16 必须回 TP=8**);KernelGym 在 **.21(192.168.16.21:20111,tvm_ffi)**;gloo iface `ens22f0np0`;loopback notify 网关在 .53。
- 关键参数:`ctx=resp=32768`、`max-turns=3`、`n=8`、`enable_thinking`(gpt-oss 忽略 → 默认 medium reasoning);去掉 Qwen 专属 sglang flag(EAGLE/linear-attn/mamba)。

<!-- 评测产物(.22, root): experiments/Eval.TVMFFI.gpt-oss-120b.gpt-oss-120b.ctx32768.resp32768.turn3.n8/gpt-oss-120b/{summary.manual.txt, 20260627.040957.log, dumps/rollout_data/eval_0.pt}
     分析快照(.53): /nfs/FM/chenshuailin/tmp_eval_preflight/{gptoss_summary.txt, gptoss_samples.txt, abort_*.py} -->

---

## DeepSeek-V4-Flash(MoE,fp8)— 无结果:.22(A800)跑不了,需 Hopper

### 结论
sglang 的 `deepseek_v4` 实现是 **Hopper(sm90)专用**,在 A800(sm80)上无法启动。两道硬件墙:

| 墙 | 位置 | 能否在 Ampere 绕过 |
|---|---|---|
| MoE top-k 簇 kernel(`__cluster_dims__`/`this_cluster`) | `jit_kernel/csrc/deepseek_v4/topk_v2.cuh` | **能**:env `SGLANG_OPT_USE_JIT_KERNEL_FUSED_TOPK=0` / `SGLANG_OPT_USE_TOPK_V2=0` 走非簇回退 |
| DeepGEMM HC-prenorm GEMM(V4 hash-compress 层) | `deepseek_v4.py` hc_pre→`mhc.py` mhc_pre→`tf32_hc_prenorm_gemm`(DeepGEMM,sm90-only) | **不能**:cuda graph 捕获即崩 |

> mxfp4/fp8 的 Hopper 依赖是 V4 的设计内核,非配置问题。`deep_gemm` 包能 import,但其 kernel 仅 sm90。

### 已就绪(只差 Hopper 硬件)
- **chat template**:V4 checkpoint 不带 jinja chat_template(只有 `encoding/encoding_dsv4.py`),手写模板与官方 `encode_messages` **逐字节一致**,装到 `<ckpt>/chat_template.jinja`(AutoTokenizer 自动加载);仓库副本 `examples/kernel_agent/prompt_config/deepseek_v4_chat_template.jinja`,单测 `test/test_deepseek_v4_chat_template.py`(15 例,过 codex review)。多轮历史由模板自剥 reasoning+eos,不动 harness、不影响其他模型。
- **eval 脚本** `examples/kernel_agent/eval.deepseek-v4-flash.sh`:TP=4(fp8 权重约 147GB,实测 ~37GB/卡,fp8 不解包);**MTP 投机已开**(EAGLE 3/1/4,源码核实:V4 hook 要求 algo==EAGLE 字面、topk==1、num-steps 显式,draft 从主 ckpt 自动加载、**无需 draft path**);`AMPERE_TOPK_FALLBACK=1` 控制 topk env 回退(上 H20 设 0)。
- **下一步(待用户定)**:在 H20(Hopper,如 `10.11.2.x-H20`)节点重跑;确认 KernelGym(.21:20111)从 H20 可达,或在 H20 侧另起 KernelGym。

<!-- 失败证据(.53): /nfs/FM/chenshuailin/tmp_eval_preflight/deepseek.launch{,.v2}.log(topk_v2 sm90 报错 / DeepGEMM NameError 全栈) -->

<!-- 名词:trajectory = 一个 (题, sample) 的完整多轮轨迹;turn = agent 的一轮生成+评测。 -->
