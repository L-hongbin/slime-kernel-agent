# KernelBench L1 模型精度登记

统一存放各模型在 **KernelBench Level 1**(tvm_ffi backend,KernelGym 评测)上的精度,供横向对比与复现。新增模型在「汇总」表加一行,并补一节「逐模型明细」。

## 汇总

指标定义:compile=编译通过;correct=数值正确;fast@1.0 / 1.2=正确且相对 torch 参考 speedup ≥ 1.0 / 1.2。口径均为 `summarize_eval.py`(逐 trajectory)。

| 模型 | checkpoint | backend / ctx=resp / turns / n | compile | correct | fast@1.0 | fast@1.2 |
|---|---|---|---:|---:|---:|---:|
| Qwen3.6-27B(slime RL,kernel-agent full-async) | iter_39(HF) | tvm_ffi / 16384 / **1** / 8 | **59.75%** | **35.00%** | **5.62%** | **4.38%** |
| **Step-3.7-Flash**(198B MoE VLM,language-only) | base(未训练) | tvm_ffi / 32768 / **3(best)** / 8 / high + MTP | **31.25%** | **18.88%** | **5.12%** | **4.12%** |
| **gpt-oss-120b**(117B MoE,base 未训练,mxfp4) | base @H20(TP8) | tvm_ffi / 32768 / **3(best)** / 8 | **53.87%** | **42.88%** | **14.12%** | **8.00%** |
| **DeepSeek-V4-Flash**(MoE,fp8) | base @H20(TP4+MTP+marlin) | tvm_ffi / 32768 / **3(best)** / 8 | **86.00%** | **65.12%** | **29.12%** | **14.25%** |

> **不可直接横比**:Qwen iter_39 是 **单轮**(max-turns=1);Step-3.7-Flash 与 gpt-oss-120b 是 **3 轮取最好**。turns/ctx 不同,只看绝对水平。
> **gpt-oss / DeepSeek 均为 H20 干净口径**:两者在 A800 都跑不动(mxfp4 / `deepseek_v4` 均需 sm90,见各自明细);DeepSeek 靠 `--moe-runner-backend marlin` 跑通。
> **复核**:gpt-oss 与 DeepSeek 的 H20 结果经 codex(gpt-5.5,xhigh)对抗式复核,均判 **TRUSTWORTHY**(从 `eval_0.pt` dump 逐 trajectory 重算 = summary;无 correct-but-not-compiled、无空 kernel 误判、无 reference 抄答、decoy 正确剔除)。

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

### Takeaway(H20,best-of-3)
- **compile 53.87 / correct 42.88 / fast@1.0 14.12 / fast@1.2 8.00**(node164 8×H20,TP8 单引擎,800 traj;codex gpt-5.5/xhigh 复核 **TRUSTWORTHY**)。逐轮单调上升(correct 11.1→23.8→29.8→best 42.9),多轮显著有效。
- 未训练基座里偏高(correct best 42.9% > Step-3.7-Flash 18.88%),能产出高质量 kernel(单样本 speedup 达 7×+);harmony 响应解析无问题(env 正确抽 `### CUDA_KERNELS`/`### MODEL_NEW`)。

### 精度(summarize_eval.py,800 traj = 100 题 × 8,max-turns=3;Tk/best 均以 800 为分母)

| 指标 | T1 | T2 | T3 | **best** |
|---|---:|---:|---:|---:|
| compile | 17.75 | 35.50 | 41.88 | **53.87** |
| correct | 11.12 | 23.75 | 29.75 | **42.88** |
| fast@1.0 | 3.25 | 6.88 | 11.25 | **14.12** |
| fast@1.2 | 2.12 | 4.00 | 6.25 | **8.00** |

### 关键坑与修复
- **A800 跑不动,只能上 H20**:mxfp4(4-bit)要 compute capability ≥ 9.0;A800=sm80 无 fp4 硬件,sglang 走未优化 triton mxfp4 MoE(~15 tok/s,应 100+),18% trajectory 撞 50min 看门狗记 0、拿不到干净数(Marlin 在 A800 也不行——MXFP4 Marlin 同样要 sm90)。H20 上 mxfp4 原生快、无超时,故只保留 H20 结果。
- **必须 TP=8 单引擎**(`GPUS_PER_ENGINE=8`):脚本默认 TP=4=双引擎会触发 **harmony 编码并发加载 race**(`pyo3_runtime.PanicException: Encoder and decoder must be of equal length`,`harmony_utils.py:get_encoding`)→ 一引擎崩 → 整个 eval **挂起零产出**(不报错,易误判"在跑")。**验证 eval 真在跑要看 GPU util + decode 日志 + KernelGym total_processed**,别只看"无报错"。
- **harmony 多轮修复(behavior-sensitive,有单测)**:响应带字面 `<|channel|>`(skip_special_tokens=False),直接回填 assistant 历史会让下轮 `apply_chat_template` 抛 `TemplateError`、结果全 0。修复 `generate_with_cuda_agent.py:_sanitize_assistant_history_content`(回填前抽 final-channel、剥 eos,非 harmony 严格 no-op);单测 `examples/kernel_agent/test/test_sanitize_harmony_history.py`。
- **EAGLE3 可选提速**:草稿 `lmsys/EAGLE3-gpt-oss-120b-bf16`,`USE_EAGLE3=1 SPEC_DRAFT_PATH=<dir>`(脚本已加 env 门控);本次未带(求稳先拿结果)。

### 复现(H20)
- 脚本 `examples/kernel_agent/eval.gpt-oss-120b.sh` + 模型 args `scripts/models/gpt-oss-120b.sh`(`--debug-rollout-only` 仅解析、不建 Megatron)。
- node164 8×H20,TP8 单引擎;env `MASTER_ADDR=10.11.2.164 KERNEL_ENV_URL=http://127.0.0.1:20211 LOCAL_GLOO_SOCKET_IFNAME=bond0 GPUS_PER_ENGINE=8`;`ctx=resp=32768`、`max-turns=3`、`n=8`。
- **reasoning_effort = medium(默认)**:脚本传的 `enable_thinking=true` 被 gpt-oss harmony 模板**忽略**(它只读 `reasoning_effort`,未设则默认 `medium`,见 `chat_template.jinja:203-206`)。即本结果是 **medium effort**,非 high;跑 high 需 `--apply-chat-template-kwargs '{"reasoning_effort":"high"}'`,数值大概率更高。
- 产物 `experiments/Eval.TVMFFI.gpt-oss-120b.gpt-oss-120b.ctx32768.resp32768.turn3.n8/gpt-oss-120b/summary.20260628.111822.txt`(node164)。

---

## DeepSeek-V4-Flash(MoE,fp8)— H20 marlin 跑通(2026-06-28,node164,8×H20/sm90)

### Takeaway
**KernelBench-L1 最强未训练基座**:best-of-3 **compile 86.0 / correct 65.1 / fast@1.0 29.1 / fast@1.2 14.25**,大幅领先 gpt-oss-120b(53.9/42.9/14.1/8.0)。逐轮单调上升(correct 27→37.5→48.75→best 65.1),多轮极有效。MTP 投机生效(accept rate ~0.5,gen ~1600 tok/s)。

### 精度(summarize_eval.py,800 traj = 100 题 × 8,max-turns=3;Tk 与 best 均以 800 为分母)

| 指标 | T1 | T2 | T3 | **best** |
|---|---:|---:|---:|---:|
| compile | 39.00 | 64.75 | 75.88 | **86.00** |
| correct | 27.00 | 37.50 | 48.75 | **65.12** |
| fast@1.0 | 8.25 | 14.25 | 23.25 | **29.12** |
| fast@1.2 | 5.50 | 6.62 | 12.12 | **14.25** |

### 修复:默认 triton MoE 后端 → marlin(关键)
- **症状**:默认(triton fused_moe)在 V4 fp8 上 sglang 启动即崩 `AssertionError: Hidden size mismatch`(`triton_utils/fused_moe.py:fused_experts_impl` 的 `hidden_states.shape[1] == w1.shape[2] - padded_size` 断言,在 cuda-graph 捕获/forward 处)。**MTP-off / TP=4 / TP=8 / eager(--disable-cuda-graph)全崩**——非 TP/投机/graph 问题,是默认 triton MoE 与 V4 fp8 权重形状不兼容。
- **解**:`--sglang-moe-runner-backend marlin`(sglang cookbook 官方 DeepSeek-V4 部署参数;Hopper 上 V4 原始 FP4 走 W4A16 Marlin MoE kernel)。脚本 `eval.deepseek-v4-flash.sh` 已加 `MOE_RUNNER_BACKEND` env 门控(`MOE_RUNNER_BACKEND=marlin`)。
- **复现**:`MASTER_ADDR=10.11.2.164 KERNEL_ENV_URL=http://127.0.0.1:20211 LOCAL_GLOO_SOCKET_IFNAME=bond0 AMPERE_TOPK_FALLBACK=0 MOE_RUNNER_BACKEND=marlin bash examples/kernel_agent/eval.deepseek-v4-flash.sh`(TP4+MTP)。产物 `experiments/Eval.TVMFFI.deepseek-v4-flash.DeepSeek-V4-Flash.ctx32768.resp32768.turn3.n8/DeepSeek-V4-Flash/summary.20260628.123236.txt`(node164)。
- **DeepGEMM JIT 慢**:V4 hash-compress 的 `TF32_HC_PRENORM_GEMM` 每个新 GEMM shape 现编(10-20min),首跑 ETA ~2h、shape 缓存后 ~1h。可选 `sglang.compile_deep_gemm` 预编译。
- **多 eval 互斥坑(已踩)**:Qwen iter-eval 流水线 `eval.t1.*.sh` 启动时 `ray stop --force` 会**误杀**同机正在跑的 deepseek eval(首跑被 iter139 杀在 7/800)。新模型 eval 必须与 Qwen eval **串行**,用 `/tmp/eval164.busy` 锁(Qwen 监控见锁让步),且全程持锁防中途插入。

### A800 跑不了(只能 H20)
sglang 的 `deepseek_v4` 是 **Hopper(sm90)专用**:MoE top-k 簇 kernel(`topk_v2.cuh` 的 `__cluster_dims__`)+ DeepGEMM HC-prenorm GEMM(`tf32_hc_prenorm_gemm`,V4 hash-compress 层)都要 sm90,在 A800(sm80)上 cuda graph 捕获即崩。(topk 可用 `AMPERE_TOPK_FALLBACK=1` 走非簇回退,但 DeepGEMM 那道墙绕不过。)

### 配置要点
- **chat template**:V4 不带 jinja chat_template,手写模板与官方 `encode_messages` **逐字节一致**,需装到 `<ckpt>/chat_template.jinja`(AutoTokenizer 自动加载);仓库副本 `examples/kernel_agent/prompt_config/deepseek_v4_chat_template.jinja`,单测 `test/test_deepseek_v4_chat_template.py`(过 codex review;本次 eval 经 codex 复核确认字节匹配)。
- **MTP 投机**:`USE_MTP_SPEC=1`(默认开,EAGLE 3/1/4,draft 从主 ckpt 自动加载、无需 draft path);lossless,只提速。
- **thinking = ON**:`enable_thinking=true` → 模板 emit `<think>`(`chat_template.jinja:20,34`);DeepSeek 无 low/med/high 分级,只 on/off,本结果是 thinking-on。与 gpt-oss 的 medium effort **不在同一轴,不可直接横比 effort**。
- **H20 覆盖**:`AMPERE_TOPK_FALLBACK=0`(非 Ampere)、TP=4(fp8 ~37GB/卡)。

<!-- 名词:trajectory = 一个 (题, sample) 的完整多轮轨迹;turn = agent 的一轮生成+评测。 -->
