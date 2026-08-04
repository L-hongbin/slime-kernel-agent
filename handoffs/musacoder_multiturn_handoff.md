# MusaCoder 多轮（3-round）评测 handoff（含 KernelGym load_inline 并发 bug）

> 合并自原 `musacoder_multiturn.md`（多轮基建/里程碑）+ `kernelgym_load_inline_concurrency_handoff.md`（并发假失败 bug），并更新了过时内容（架构整合到 .22 单机、bug 的最终修复与实证、大 shape 最终结果）。**最终 L1/L2/L3 结果表及复现步骤见同目录 [`musacoder_mt3_reproduction.md`](musacoder_mt3_reproduction.md)**。

## ✅ 最终状态（TL;DR，已实证 + codex xhigh 复核）
- **多轮评测完成（大 shape）**：clean **T1 correct 86.50% / best-by-turn 96.50%**（compile best 99.12%，fast@1.0 best 25.00%，cannot-open=0）。final-turn 会回落（T3 75.25% < T1）→ 真实使用须按 reward **选 best turn**，不能盲取末轮。best-by-turn 仍失败的 28 条集中在 9 道硬题（conv_transposed_3D、HingeLoss 等）。最终结果见 [`musacoder_mt3_reproduction.md`](musacoder_mt3_reproduction.md)。
- **KernelGym load_inline 并发假失败 bug 已根治**：根因 = PyTorch JIT versioner 在 **compile 进程**与 **execute 进程**间对复用的 `load_inline(name=…)` 算出不同 `_vN` → 假 `<name>_vN.so: cannot open shared object`。修复 = **客户端 `config.py:29` + 服务端 profile 都设 `split_compile_and_execute=false`**。实证：重跑 179 条污染 trajectory → **cannot-open 193→0**。
- **架构整合到 .22 单机**：KernelGym 在 GPU 0-3，rollout 在 GPU 4-7。**不再用 .21**（已释放，用户另作他用，勿动）。

---

## 1. 目标
在 baseline（load_inline 单块格式）之上做 **3 轮 multi-turn** 评测：每轮把 reward 环境的结构化 feedback（compile/correctness/speedup/error）拼进上下文，让模型迭代修正/优化。复刻论文 MusaCoder 的多轮形式。

## 2. 架构（已更新：.22 单机 4+4）
全部在 `ssh -p 24167 root@192.168.16.22`（ai-16-22，8×A800-80G）：
- **KernelGym（reward）**：GPU **0-3**。worktree `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-load-inline`（分支 `feature/musacoder-load-inline`）。profile v1：`gpu_devices=(0,1,2,3)`、`WORKER_POOL_SIZE=4`、`SPLIT_COMPILE_AND_EXECUTE=false`、`MAX_TASKS_PER_WORKER=1`。API `http://127.0.0.1:20111`，redis 本地（`apt-get install redis-server`）。部署：`bash deploy_node.sh --nnodes 1`。
- **rollout（slime）**：GPU **4-7**。worktree `/nfs/FM/chenshuailin/projects/kernel_agents/slime-musacoder-mt`（分支 `feature/musacoder-multiturn`）。`CUDA_VISIBLE_DEVICES=4,5,6,7`、`NUM_GPUS=4`、`ACTOR_NUM_GPUS=4`、1 engine TP=4。
- 启动脚本：`/nfs/FM/chenshuailin/staging_oneshot_conv1x1/launch_mt3_*.sh`（fire-and-forget，内含 `ray stop --force`）。
- **历史**：早期是两节点（.22 rollout + .21 reward `ssh -p 23729`）；大 shape repair 时整合到 .22 单机。两端 GPU 用 `CUDA_VISIBLE_DEVICES`/`GPU_DEVICES` 物理隔离（0-3 vs 4-7），互不抢占。

## 3. 多轮为什么必须改代码（核心发现）
单轮 baseline 用「cuda_agent precheck 拒 load_inline → reward=0 → 事后 rescore dump」绕过；**多轮不行**，第 N+1 轮 prompt 依赖第 N 轮的**实时**评测。改动（slime 侧）：
1. `slime/utils/arguments.py`：`--kernel-backend` 加 `load_inline` 选项。
2. `examples/kernel_agent/kernel_response.py`：load_inline 时 `kernel_code` 传**原始 response**（让 server 端抽块，不走三段式 `extract_cuda_agent_kernel_code`）；precheck 对 load_inline 返回 None 自动放行。
3. KernelGym 专用分支提供完整 `load_inline` 路径：`Backend.LOAD_INLINE`、dispatcher 路由、原始单块代码抽取、复用 CUDA 编译执行路径以及 load_inline decoy 检查。KernelGym `main` 不能替代该分支。
4. 已测试的 KernelGym 代码基线为 `feature/musacoder-load-inline@55018290`；它已合入 `dev_csl@2625505`，包含 no-grad profiling、true-FP32 correctness、CUPTI profiler 和 worker supervision 等近期更新。

## 4. feedback 拼接模板（论文 Appendix G）
下一轮 user message = `=== System Feedback for Round N ===` + curated feedback，二分支：**Correct**→优化（保持 correctness、只提速）；**Wrong**→先修正确性。slime 侧：`generate_with_cuda_agent.py` 每轮 `_apply_feedback_template` 按 `prompt_config/multi_turn_load_inline.yaml`（新写，Appendix G 风格、要求单块 load_inline 输出）拼 user turn。
- **feedback curation**：`_curate_env_feedback` 只留模型相关字段（compiled/correctness/decoy/speedup/runtimes/status/error + metadata{backend, correctness_issue, max/avg_difference, atol/rtol, tf32_disabled}），砍掉 `kg_stage_*` 遥测。
- **历史轮 thinking 自动剥离**：每轮 `apply_chat_template` 重渲染，Qwen3.6 模板自动去历史 `<think>`，对齐论文 §4.2.3，无需手动 strip。
- context：`MAX_CONTEXT_LEN=40960`（8K prompt + 32K response，对齐论文）。

## 5. KernelGym load_inline 并发假失败 bug（完整根因 + 修复 + 实证）

### 5.1 现象
load_inline live reward 并发编译时出现假编译失败：`Compilation failed … /tmp/kernelgym_cuda_XXXX/<name>/<name>_vN.so: cannot open shared object file`。`.so` 在该任务**自己唯一**的 mkdtemp 目录里，期望名有时带 `_v1/_v3` 后缀、有时不带。污染率随并发单调上升（128-wide→~11%、16-wide→~3.6%、32-wide split-on→8%、1/离线→0）。

### 5.2 根因（codex xhigh 复核确认）
PyTorch JIT 扩展 versioner 是**进程级全局**、按扩展 `name` 计数（`torch/utils/_cpp_extension_versioner.py:29/46`、`cpp_extension.py:367`）。同一长寿命进程里 `name="matmul_cuda"` 被不同 source/build_dir 反复编译 → PyTorch 改名 `matmul_cuda_v1/_v2/_v3…`（`cpp_extension.py:2217`）。**split-compile 把 compile 与 execute 放在两个不同进程**（`workflow/kernelbench.py:93`）：compile 进程写出 `…_vN.so`，execute 进程（versioner 计数不同）去找 `…_vM.so` → cannot open。**铁证**：实际报错带 `_v3` 后缀，正是 versioner 改名产物（早期"os.environ race"假设已被 codex REFUTE——执行是串行/跨进程的，不存在同进程并发编译）。

### 5.3 为什么我一开始没修好（改错了开关侧）
先在**服务端** profile 设 `SPLIT_COMPILE_AND_EXECUTE=false`，但 **slime 客户端** `examples/kernel_agent/config.py:29` 默认 `split_compile_and_execute=True`，每请求都带 True；服务端 `request_defaults.py` 只**升级**、从不降级 → 客户端 True **覆盖**服务端 false → split 从没真正关掉（大 shape run 仍 193/2400=8% cannot-open，且 task_manager 日志确证仍在提交 `_compile`(cpu)+`_kernel`(gpu) 分离任务）。

### 5.4 正确修复（两侧都 false）+ 实证
- 客户端：`slime-musacoder-mt/examples/kernel_agent/config.py:29` → `"split_compile_and_execute": False`。
- 服务端：profile `SPLIT_COMPILE_AND_EXECUTE=false`（`deployment_profiles.py:49`，**保持 false，非临时**）。
- ⚠️ 两侧都必须 false：服务端若 True 可经 runtime defaults 强制 split（有测试 `tests/test_request_defaults.py`），客户端 args/payload 也可能发 True（`kernel_response.py:195/274`）。
- **实证**（codex 独立复算）：把大 shape run 中 179 条 cannot-open 污染 trajectory 用 fix 重跑 → **cannot-open 193→0**（rerun 0/537，仅 3 条真实 illegal-memory）。合并回 800 → cannot-open=0。

### 5.5 只影响 load_inline，三段式/tvm_ffi 不受影响（codex 复核）
- **cuda_agent（三段式）**：用内容 hash 唯一 ext name（`kernelgym_cuda_agent_<cache_key>`）、显式跑 ninja 传 build_dir → name 不复用、无 versioner bump。
- **tvm_ffi**：独立编译路径（`tvm_ffi.cpp`）、每任务 work_dir、内容键控 cache → 不走 torch versioner。
- 唯独 load_inline 的 `name=` 由**模型自己写**（大量复用 `matmul_cuda` 等）+ split-compile 跨进程 → 独中此 bug。（redis 编译 cache 是 cuda_agent-only，与此 bug 无关。）

### 5.6 代价 + 长期方案
split-off 让编译失去 24-CPU-worker 流水线、且 `MAX_TASKS=1` 每任务冷启重编 → reward 编译变慢（~100s/条复杂大 shape kernel），是吞吐瓶颈（大 shape 全量 ~5h）。**长期 = codex 方案 C**：保留 split-on（恢复编译流水线速度）+ monkeypatch 让 `load_inline(name=…)` **每任务唯一**（不复用就不 bump、跨进程也对得上）→ 兼顾快与正确。尚未实现。

### 5.7 多轮污染必须"重跑"而非"rescore"
cannot-open 假失败会经 feedback 传染后续轮（模型去修不存在的编译错）。所以修复污染 trajectory 必须**重跑整条**（同 problem_id 全新随机样本，引入少量额外采样方差），不能只对单轮 rescore。merge 时按 problem_id 匹配替换、normalize group_id/index。

## 6. 里程碑（历史，已全部完成）
- **M0–M2**：recon + worktree；reward 侧（Backend 枚举 +LOAD_INLINE、no_grad 修复）；slime 侧（arguments/kernel_response/curation/yaml/eval 脚本）。codex(xhigh) M1+M2 review = GO。
- **M3 sanity = GO**：3 题×2×3 轮，无 400、零遥测泄漏、context 受控、多轮真起作用（repair ❌→✅ + optimize），codex feedback review faithful。
- **M4 小 shape 全量**：v1（128-wide）被并发污染作废；v2（16-wide）仍 3.6% 污染 → repair（重跑 84 条 `.so cannot open` 污染组）。**注：M4 时期记的"os.environ race"根因已被 codex REFUTE，正解是 §5.2 的 versioner 机制。** 小 shape repaired 结果仅作历史，不属于最终三轮复现口径。
- **M5 大 shape 全量 + 根治（最终）**：split-off（先服务端→发现客户端覆盖→改客户端 config.py:29）；32-wide 跑大 shape；重跑 179 条污染 trajectory 实证 cannot-open=0；merge + 算指标 + codex milestone review（3 VERIFIED / 1 ISSUE：merged .pt 的 group_id/index 已 normalize 修复）。

## 7. 已发布改动文件（slime `69566460` / KernelGym 代码基线 `55018290`）
- **slime**：`utils/arguments.py`（+load_inline）、`examples/kernel_agent/kernel_response.py`（raw response 路由）、`generate_with_cuda_agent.py`（`_curate_env_feedback`）、`config.py:29`（**split=False，bug 修复**）、`eval.mt3.musacoder.27B.sh`（`NUM_GPUS`/`ACTOR_NUM_GPUS` 参数化）、新 `prompt_config/multi_turn_load_inline.yaml`。
- **KernelGym**：`kernelgym/common.py`（Backend +LOAD_INLINE）、`kernelgym/backend/kernelbench/load_inline_backend.py`（单块代码抽取/加载）、`kernelgym/toolkit/kernelbench/load_inline_decoy.py`（compiled-but-unused 检查）、`deployment_profiles.py`（split=false、gpu_devices=(0,1,2,3)、pool=4）。

## 8. 关键事实/坑（沿用单轮经验）
- 评测 harness 跑 **train 模式**（不调 `.eval()`），BatchNorm 须现算 batch 统计量（见 [`musacoder_mt3_reproduction.md`](musacoder_mt3_reproduction.md)）。
- correctness 用 **TF32-off + 1e-4**（`KERNELGYM_CORRECTNESS_DISABLE_TF32=1`，默认 on）。
- EVAL_DATA 必须**绝对路径**（Data/ 被 gitignore，ray 上传 working-dir 会丢相对路径）。
- ray 结束不自动 `ray stop`；重跑前 `ray stop --force` + 清残留 GCS/raylet（否则 GCS 启动超时）。fire-and-forget 启动（`setsid … </dev/null &`）以防 ssh 掉线杀进程。
- **大 shape 数据集**：`Data/kernelbench-level1-validation-musa-coder-load-inline/`（median max-dim 4096）；小 shape：`…-oldsize-rand/`（256）。
- **dump 分析**：按 `turn_idx==0` reset 重建 trajectory（dump 内 group_id 可能 None/重复，除非已 normalize）。

## 9. 证据 / 辅助脚本（staging）
- merged clean 大 shape dump：`…/MusaCoder-27B.mt3_largeshape_nosplit.load_inline/dumps/rollout_data/eval_0_merged_clean.pt`（cannot-open=0，group_id/index 已 normalize，800 组）。
- 重跑实证 dump：`…/MusaCoder-27B.mt3_largeshape_rerun179.load_inline/…/eval_0.pt`（cannot-open=0）。
- 多轮离线 rescore wrapper：`/nfs/FM/chenshuailin/staging_oneshot_conv1x1/rescore_mt_dump.py`（逐样本调 `score_one_sample.py`，保留 idx/group_id/turn_idx，出 T1/T2/T3 + best-by-turn）。
- Telegram 通知：`/nfs/FM/chenshuailin/staging_oneshot_conv1x1/notify.sh "<msg>"`（走 notify-mcp 网关）。
- codex 复核记录均在 `/nfs/FM/chenshuailin/staging_oneshot_conv1x1/codex_*_out.txt`。
