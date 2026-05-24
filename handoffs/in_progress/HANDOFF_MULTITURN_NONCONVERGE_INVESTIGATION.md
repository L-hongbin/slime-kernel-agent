# DrKernel 多轮 eval 不收敛调研 handoff

调研日期：2026-05-23
模型：Qwen3.6-27B
数据集：KernelBench Level 1 validation (100 prompts × n_samples_per_eval_prompt=8 = 800 samples)
配置：TP=2，max-turns=3，eval ctx 32k (见"side discovery #1")，tvm_ffi backend

## 问题

LHB DIR1 参考跑（`/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent/drkernel/logs/cuda-qwen36-27b-l1fullset-tvmffi-n2-tp2-seqs16-32k-24-r2.run.20260508-085544`）的多轮指标随 turn 单调上升；我们当前 slime/drkernel 版本几乎平。

| metric | LHB DIR1 (T1→T3) | 我们 summ800 (T1→T3) |
|---|---|---|
| compile | 41% → 43% → **54%** | 44% → 45% → 45% |
| correct | 9.5% → 9.9% → 14.3% | 27.6% → 26.2% → 26.2% |
| fast@1.0_in_all | 3.5% → 5.5% → 7.7% | 13.8% → 10.9% → 11.0% |
| fast@1.2_in_all | 2.5% → 4.9% → **6.6%** | 0.9% → 1.2% → 1.0% |

注意：denominator 不同。LHB n=2 → T1=200, T2/T3=182（18 个 sample T1 后早停）；我们 n=8 → 全部 T1/T2/T3=800（无早停）。

Per-problem 任意 sample 任意 turn fast@1.2 命中：LHB 8/100=8% vs ours 6/100=6%。**raw 数字差 6× 主要是 denominator 差，但"多轮上升趋势"差异是真实的**。

## 做了什么

### 1. 三向 truncate-length sweep（n=800，Qwen3.6-27B，KernelBench L1）

测试 `KERNELGYM_ERROR_SUMMARY_CHARS` 取 800/1600/3200 三个值是否影响多轮收敛。

run dirs：
- `checkpoints/Qwen3.6-27B/20260523_054450_ctx65536_n8_summ800/`
- `checkpoints/Qwen3.6-27B/20260523_072056_ctx65536_n8_summ1600/`
- `checkpoints/Qwen3.6-27B/20260523_090038_ctx65536_n8_summ3200/`

| metric | T1 | T2 | T3 |
|---|---|---|---|
| compile | 44/44/41 | 45/49/44 | 45/48/46 |
| correct | 28/28/27 | 26/30/26 | 26/28/27 |
| fast@1.0 | 14/14/14 | 11/10/10 | 11/10/10 |
| fast@1.2 | 0.9/0.8/0.6 | 1.2/0.8/0.8 | 1.0/1.0/1.6 |

**结论：feedback 长度（800→3200，4×）不影响多轮收敛曲线。** 不是 feedback 信息不够多导致 model 不能改进。

### 2. Per-turn 转移分析（fix vs regression rate）

| | T1→T2 | T2→T3 |
|---|---|---|
| LHB compile fix/regression | 14% / 24% | **23% / 6%** |
| ours compile | 23% / 28% | 24% / 28% |
| LHB correct | 6% / 63% | 9% / 33% |
| ours correct | 13% / 38% | 13% / 37% |
| LHB fast@1.0 | 3% / 29% | 4% / 20% |
| ours fast@1.0 | 4% / **46%** | 5% / 37% |

**关键发现**：我们的 fix rate 不输甚至超过 LHB，但 **regression rate 极高**——working kernel 在下一轮被打破的概率 38-46%。LHB T2→T3 regression 已经下降到 6%（compile）/20%（fast@1.0），我们一直平在 28%/37%。

LHB 的 T2→T3 收敛部分得益于 **early termination**（18 个 sample 在 T2 后停了，去除了高风险的"已成功 sample"）。我们没早停。

### 3. Tokenizer thinking-strip 调研（用户假设：我们 strip history thinking 而 LHB 不 strip）

**假设被证伪。** Codex 用 token-count 匹配验证：

Qwen3.6 chat_template 默认行为：历史 assistant turn 的 `<think>...</think>` 全部被剥掉，只保留最后一个 user query 之后的 thinking。Escape hatch：传 `preserve_thinking=True` kwarg。

| LHB problem_87 sample_0 | actual prefill | re-render default | re-render preserve_thinking=True |
|---|---|---|---|
| T2 进入 | 2854 | **2854** ✅ | 7995 |
| T3 进入 | 4073 | **4073** ✅ | 11769 |

LHB 也是 strip 历史 thinking 的。`full_conversation.txt` 看起来含 thinking 是因为它从 `multiturn_messages` 重建的人类可读 transcript，不是实际送 vLLM 的 prompt。

**两边都 strip，不是差异源**。但 `preserve_thinking=True` 是 LHB 没做、我们也没做的事——如果开了能改善多轮收敛，是绝对的净改进（用户决定**不做** preserve_thinking 实验）。

我们的 `--preserve-history-thinking` arg 定义了但**没接到 `apply_chat_template_kwargs`**，是 dead arg（[args.py:46](slime_plugins/drkernel/args.py#L46)）。

### 4. 七维 systemic 差异调研（codex）

| # | dimension | verdict |
|---|---|---|
| Q1 | 采样参数 (T/top_p/top_k/min_p/repetition/presence) | matches（均 T=1.0, top_p=0.95, top_k=20, 其他默认） |
| Q2 | 引擎 (SGLang vs vLLM 0.18) | differs but no Qwen3 specific quirk found |
| Q3 | response_truncation 是否影响 history append | LHB 的 `python_code, answer_code` 仅影响 reward/action 抽取，history 写入仍是 full response，matches |
| Q4 | stop_token_ids | LHB `[872, 77091, 151645, 151644]` 是 Qwen 2.5/3.0 IDs，对 Qwen3.6 是**死 token**，两边都靠 EOS 停 |
| Q5 | max response length | 都 32k（见 side discovery #1） |
| Q6 | tools field in apply_chat_template | 我们 metadata.tools=None，no-op |
| Q7 | 多轮 plumbing（apply_initial_user_prompt_template / apply_turn_prompt_template） | LHB hooks 是 idempotent，无额外 mutation，matches |

**唯一真实非 template 差异**：并发。LHB 4×16=64 active seqs vs 我们 4×64=256。SGLang `enable_deterministic_inference=False` 下高并发可能影响动态 batching 路径，但 codex 没找到 mechanism 关联到多轮收敛曲线。

## 结论

**LHB 多轮改善优于我们，主要驱动来自 prompt template**（first_turn 模板 + tool_response 模板的措辞和结构）。其他维度已基本排除。

具体 template 差异（codex 早期调研）：
- LHB first turn: `"Optimize... preserving correctness"`（强调保持正确）
- 我们: `accelerate_best_perf`（推动激进优化）/ `optimize_correctness`（修正）双 role
- LHB tool_response: `"The server feedback..."`（中立陈述）
- 我们: `"Use this feedback to improve..."`（暗示必须改）

这些差异让我们的 model 拿到 working kernel 时仍被推动"继续优化"，对应 per-turn 转移分析里高 regression rate。

## 5. Prompt template ablation (E0-E4, 100×1×3turn, 64k context)

调研日期 2026-05-23/24。CTX_LEN=65536, n_samples_per_eval_prompt=1, KERNELGYM_ERROR_SUMMARY_CHARS=1600, KernelGym reference_backend=pytorch。

每个 E = 在 baseline 之上**只动一个变量**，layer-on 前一轮的胜出配置。每 run 100 samples，运行约 25 min。

### 配置 diff

| variant | tool_response | first_turn role | first_turn layout |
|---|---|---|---|
| **E0** baseline | 我们原版 tvm_ffi_short.jinja（"Use this feedback to revise your previous CUDA + TVM-FFI implementation:" + "Return ... `APPLY_BINDINGS` (TVM-FFI, no pybind), and `MODEL_NEW` (using `tvm_ffi_extension`)"）—带 TVM-FFI 特定提示 | dual cycle (accelerate_best_perf / optimize_correctness) | "You are given the following PyTorch model:" + ```python fence``` |
| E1 | LHB short.jinja 内容（"Use this feedback to revise your previous CUDA answer:"）—**generic backend-agnostic** | dual cycle | wrapper |
| **E2** ⭐ | LHB default.jinja 内容（"The server feedback for your previous CUDA implementation is below.\n\nServer feedback:" + "Fix the relevant issue or improve the implementation" + 显式三段 skeleton）—**generic backend-agnostic，byte-identical 于我们既有 pybind_default.jinja** | dual cycle | wrapper |
| E3 | E2 content | LHB **single role**（"You are a PyTorch and CUDA expert. Optimize the given PyTorch \`Model\` by returning a TVM-FFI CUDA extension implementation that is faster than the original while preserving correctness."） | wrapper |
| E4 | E2 content | dual cycle | **去 wrapper**（直接 `{{ problem }}`） |

**重要修正**：LHB DIR1 的 `multi_turn_cuda_kernel_tvm_ffi_only.yaml` 配置 `tool_response: template_dir: cuda_templates/tool_response`，加载该目录下**全部** `.jinja`（`default.jinja` + `short.jinja`），然后 `select_prompt_template` 每次 turn 调用时 `random.choice(candidates)`。**LHB 实际是 ~50/50 随机用 default 或 short**，不是固定一个。

所以 E1（always short）和 E2（always default）都不是 LHB DIR1 的真实行为。E2 测的核心变量是"**丢掉 TVM-FFI 特定提示，换成 generic backend-agnostic 模板**"（这一变量本身有效），尚未单独测过"random mix of default + short" 这一更真实的 LHB 行为。值得加 E5（见下方建议）。

### 结果（per-turn metrics）

| variant | T1 compile/correct/fast@1.0 | T2 compile/correct/fast@1.0 | T3 compile/correct/fast@1.0 | reward hits | T1→T3 monotonic? |
|---|---|---|---|---|---|
| E0 | 43% / 28% / 17% | 50% / 31% / 7% | 46% / 26% / 10% | 26 | ✗ |
| E1 | 49% / 27% / 16% | 51% / 28% / 12% | 44% / 27% / 12% | 27 | ✗ |
| **E2** | 48% / 30% / 15% | 50% / 32% / **15%** | **53% / 32%** / 11% | **32** | **✓ compile & correct** |
| E3 | 45% / 28% / 17% | 40% / 21% / 7% | 49% / 30% / 11% | 30 | ✗ (V-shape) |
| E4 | 38% / 26% / 12% | 44% / 29% / 12% | 44% / 27% / 6% | 27 | ✗ |

### 结论

**E2 唯一胜出**：T1→T3 单调 compile (48→50→53) + 单调 correct (30→32→32)；reward 32 > E0 26 (+23%)。

其他 ablation 都没改善：
- E1 (LHB short 文案)：跟 E0 在 100 sample 噪声内分不出来
- E3 (LHB 单一 role)：T2 dip 大（compile -10pp）— dual role cycle 的 model behavior diversity 比单一 role 好
- E4 (去 layout wrapper)：T1 compile -10pp 退化—"You are given the following PyTorch model:" 包装**非冗余**

**当前代码状态 = E2**：tool_response 已替换为 LHB default 风格；first_turn role 和 layout 保持原样。E3/E4 的改动已 revert。

### E2 vs LHB DIR1 直接对比（要点）

LHB DIR1: n=2×100=200 samples, 32k context, pytorch ref backend  
E2: n=1×100=100 samples, 64k context, pytorch ref backend

| metric | T1 | T2 | T3 |
|---|---|---|---|
| compile | E2 48 / LHB 41 | E2 50 / LHB 43 | E2 53 / LHB 54 |
| correct | E2 **30** / LHB 9.5 | E2 **32** / LHB 9.9 | E2 **32** / LHB 14.3 |
| fast@1.0 | E2 **15** / LHB 3.5 | E2 **15** / LHB 5.5 | E2 11 / LHB 7.7 |
| fast@1.2 | E2 0 / **LHB 2.5** | E2 1 / **LHB 4.9** | E2 0 / **LHB 6.6** |

**E2 比 LHB compile 平、correct 高 2-3 倍、fast@1.0 高，但 fast@1.2 全面输**。

E2 speedup 分布全部挤在 1.0 附近——T1 max 1.08, T3 max 1.08, T2 才偶然出 1.36。LHB 在 MinGPTNewGelu (8.6×), KLDivLoss (20.7×), CrossEntropy (1.82×) 上有真正大赢家。

**Trade-off**：E2 的 prompt template 让 model 更"保守"（保持 correct），LHB 的让 model 更"激进"（多失败但少数大命中）。两边各有优势，**不是单一最优解**。

注意 sample size 不对等（n=1 vs n=2）。fast@1.2 比较受 100-sample 噪声影响大，需要 100×8 (n=8) 验证。

## 下一步建议

| 优先级 | 内容 |
|---|---|
| **P0** | E2 配置（tool_response = generic default）跑一次 100×8×3turn，跟 summ800/summ1600/summ3200 baseline 直接可比，验证 reward / compile / correct 改善能 reproduce |
| **P0b** | E5（待跑）：random.choice between default + short（真实对齐 LHB DIR1 行为）。需要在 renderer 加 random.choice 支持，或在 yaml 让 tool_response_text_path 支持 list + cycle/random |
| P1 | 调研 model 为什么 turn 1 不写激进 kernel：对比 E2 高 speedup 的几个 kernel 与 LHB hit fast@1.2 的 kernel 风格 |
| P2 | 把 LHB feedback 字段（`task_id` / `reward` / `success` / 更广 metrics）加回我们的 build_prompt_feedback_payload，看 fast@1.2 是否能提升 |
| P3 | 加 early termination（hit fast@1.0 或 correct 即停），用 LHB 的早停语义 |
| P4 | 接通 `--preserve-history-thinking` arg 到 `apply_chat_template_kwargs`，验证是否是净改进 |
| 杂项 | 修 YAML/CLI override bug（用户已 commented out，见 side #1）；stale 注释清理 |

## Side discoveries

### #1: YAML defaults 压制 CLI args（已用户 commented out 修复）

`scripts/eval_kernelbench_level1.yaml` 的 `eval.defaults.max_*_len: 32768` 优先级高于 `--eval-max-*-len 65536` CLI args（[eval_config.py:201](slime/utils/eval_config.py#L201) `_apply_dataset_field_overrides` 顺序：dataset_cfg → defaults → args）。

之前以为 eval 跑在 64k context，实际是 **32k**。用户已把 YAML 里 max_*_len 行 comment out，CLI args 现在生效。

Stale 注释（同文件）：
- L3 `DrKernel Qwen3-14B` → 实际 27B
- L31 `reference_backend=torch_compile` → 已改 pytorch（kernelgym_rm.py:33）

### #2: LHB stop_token_ids 对 Qwen3.6 是死 token

LHB 显式设置 `stop_token_ids=[872, 77091, 151645, 151644]`，对应 Qwen 2.5/3.0 tokenizer 的 `<|im_start|>` / `<|im_end|>`。Qwen3.6 tokenizer 这些 ID 对应普通文本 token（`nder`, `[hash`），不会触发停止。LHB 实际靠 model 默认 EOS 停。我们没设 stop_token_ids 也靠 EOS 停。两边等价。

### #3: KernelGym reference_backend 切到 pytorch

旧 codex 调研发现 LHB DIR1 用 `reference_backend=pytorch, num_warmup=5, task_timeout=30`，我们之前是 `torch_compile, 30, 90`。**ref_backend 只影响 speedup 数值（fast@x 指标）**，不影响 compile/correct。已改 `KERNELGYM_REFERENCE_BACKEND = "pytorch"`（kernelgym_rm.py:33）。num_warmup 和 task_timeout 暂未改回 LHB 值。

### #4: 没改的：feedback 字段集 + 措辞

LHB feedback 包含 `task_id` / `reward` / `success` / 更广的 metrics（`time_coverage`, `num_coverage`, `num_custom_kernel`, total runtime fields）/ metadata（`gpu_name`, `device`, `backend`）。我们migrate 时去掉了这些，只保留诊断错误。LHB feedback ~958 chars vs ours 558-752 chars（对同样 problem_0/sample_0）。是否加回是 P1 任务。

## 关键文件参考

- 调研代码（temp scripts，已在 .22 留存）：
  - `/tmp/quick_think_probe.py` (chat template thinking-strip 验证)
  - `/tmp/turn_transitions.py` (per-turn 转移分析)
  - `/tmp/lhb_transitions.py` (LHB 对应的转移分析)
  - `/tmp/codex_thinking_output.log` (codex thinking-strip 调研报告)
  - `/tmp/codex_nonprompt_output.log` (codex 七维 systemic 差异调研报告)
- 三次 sweep run 的 eval_0.pt 在 NFS：`/nfs/FM/chenshuailin/projects/kernel_agents/slime/checkpoints/Qwen3.6-27B/20260523_*_summ{800,1600,3200}/dumps/rollout_data/`
- 5 个 E ablation run：`/nfs/FM/chenshuailin/projects/kernel_agents/slime/checkpoints/Qwen3.6-27B/20260523_*_ctx65536_n1_summ1600_E{0,1,2,3,4}/dumps/rollout_data/`
- LHB DIR1 参考：`/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent/drkernel/logs/cuda-qwen36-27b-l1fullset-tvmffi-n2-tp2-seqs16-32k-24-r2.run.20260508-085544/`
- 之前的 plan handoff：`handoffs/in_progress/HANDOFF_DRKERNEL_SLIME_PLAN.md`
