# 用 slime 重新实现 drkernel 的计划

## 目标

用 slime 承担训练、rollout、数据组织、checkpoint 和日志；KernelGYM 保持为独立 HTTP reward/eval 服务，复用现有实现，不在 slime 内重写 kernel 编译、运行、计时和 worker 调度。

终点是**多轮 GRPO 训练**。截至 2026-06-02，多轮 **eval** 闭环已经打通（Qwen3.6-27B / KernelBench L1）；多轮**训练** rollout 和首次 GRPO 训练仍未做——这是当前的主线缺口。

## 关键路径

- slime 项目：`/nfs/FM/chenshuailin/projects/kernel_agents/slime`
- KernelGYM 项目：`/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent`
- KernelGYM HTTP 服务 / 请求响应模型：`kernelgym/server/api/server.py`、`kernelgym/server/api/models.py`（`EvaluationRequest` / `EvaluationResponse`）

## 当前状态（2026-06-02）

已从「单轮 eval」推进到「多轮 eval 闭环」：

- DrKernel custom rollout（`slime_plugins.drkernel.rollout.generate_rollout`）同时提供单轮训练 rollout 和多轮 eval rollout。
- 多轮 eval driver `generate_multi_turn_eval_sample` 固定跑 `--max-turns` 轮（无早停）：每轮抽取 kernel → 调 KernelGym `/evaluate` → 把编译/正确性/性能反馈摘要成下一轮 user message → 再生成；逐轮 audit 落在 `sample.metadata["turns"]`，`--dump-details` 可看完整轨迹。
- 反馈走白名单摘要（`build_prompt_feedback_payload` / `summarize_diagnostic_text`），错误文本压到 ~1600 字符；`<|im_end|>` 双重包裹已修复（`_strip_chat_stop_markers`）。
- 动态 prompt 模板系统成型：可组合 YAML（role × backend × layout × tool_response），active profile `drkernel_v1_tvm_ffi`（默认 backend 模板 `tvm_ffi_module_v2_3.jinja`）。
- eval 并发节流、per-turn 日志、KernelGym 健康检查、结果汇总工具齐备；有中文 tutorial 和示例脚本。
- 主线模型/节点已切到 Qwen3.6-27B、A800 `.22` / A100 `.16`（不再是 5 月的 Qwen3.5-9B / `.67`）。

**仍未做（训练主线）**：训练路径 `generate_rollout_async` 目前**只跑单轮**；多轮训练 rollout、训练 `loss_mask`、跨轮 reward 聚合 / advantage、首次 GRPO 训练都还没接。多轮相关 CLI 参数（`--use-multi-turn` / `--max-turns` / `--padding-turns` / `--multi-turn-gamma` / `--filter-by-last-turn`）在 `args.py` 已声明，但训练侧尚未消费 `padding-turns` / `multi-turn-gamma` / `filter-by-last-turn` 这几个。

## 已完成组件清单

| 组件 | 文件 | 状态 / 要点 |
|---|---|---|
| 数据转换 + Dataset 接入 | `scripts/data/convert_verl_to_slime.py`、`tests/utils/test_convert_verl_to_slime_data.py` | 完成。`--input-key/--label-key ground_truth --metadata-key extra_info`；`Sample.prompt`=动态 prompt 输入，`Sample.label`=KernelGym `reference_code`，`extra_info`→`Sample.metadata` |
| 动态 prompt 渲染 | `slime_plugins/drkernel/rollout.py::DrKernelPromptRenderer`、`slime_plugins/drkernel/prompt_templates/prompts_v1.yaml`、`slime_plugins/drkernel/design-docs/dynamic_prompt_templates.md` | 完成。fragment 先经 jinja 渲染避免 `{# #}`/`{{ }}` 泄漏；模板选择只受 `metadata["template_allowed"]` 限制 |
| 模板单测 | `tests/utils/test_drkernel_prompt_templates.py`、`test_drkernel_no_jinja_leak.py`、`test_drkernel_cache_template.py` | 完成 |
| kernel 抽取 | `slime_plugins/drkernel/extract.py`、`tests/utils/test_drkernel_extract.py` | 完成。CUDA_KERNELS→APPLY_BINDINGS→MODEL_NEW 三段；去 think；auto-backend；失败返回 `None` |
| KernelGym client + 单轮 RM | `slime_plugins/drkernel/kernelgym_rm.py`、`tests/utils/test_drkernel_kernelgym_rm.py` | 完成。async client、`evaluate_sample`、`custom_rm`、健康检查 |
| 多轮 eval driver + 并发节流 | `rollout.py::generate_multi_turn_eval_sample` / `eval_rollout_single_dataset`、`eval_throttle.py`、`tests/utils/test_drkernel_multiturn_render.py`、`test_drkernel_eval_throttle.py` | 完成。单样本异常隔离；`DRKERNEL_EVAL_MAX_CONCURRENCY` 或按 `max_running×engines×2` 自动 |
| 反馈摘要 | `rollout.py::format_kernelgym_feedback` / `build_prompt_feedback_payload`、`slime_plugins/drkernel/design-docs/feedback_summarization.md`、`tests/utils/test_drkernel_kernelgym_feedback.py` | 完成 |
| eval 上下文/配置 | `slime/utils/eval_config.py`、`scripts/eval_kernelbench_level1.yaml`、`tests/utils/test_eval_config.py` | 完成。`max_prompt_len`/`max_context_len`；SGLang 侧留 32 token reserve 防 400 |
| DrKernel CLI 参数 | `slime_plugins/drkernel/args.py` | 完成（训练侧多轮参数已声明，未全部消费） |
| 结果汇总 | `scripts/drkernel/summarize_kernelgym_eval.py` | 完成。compile / correct / fast@1.0 / fast@1.2 + fast@*_correct（无独立单测） |
| 文档 / 示例 | `scripts/eval_drkernel_example.sh`、`docs/zh/get_started/eval_tutorial.md` | 完成 |

注：原计划提议的 `slime_plugins/drkernel/data.py`、`scripts/drkernel/smoke_slime_data_loading.py`、`smoke_kernelgym_reward.py` 这几个 smoke 脚本**未建**，被真实 eval run + tutorial + 单测取代。

## 运行 / 环境事实（指针）

不在本文档重复完整参数，以下为权威来源：

- 节点、共享路径、reward endpoint、数据路径：`RUNTIME.md`
- eval 跑法（含多轮、可覆盖环境变量）：`scripts/eval_drkernel_example.sh`、`scripts/debug.sh`
- 完整中文教程（依赖、流程图、结果分析、排错）：`docs/zh/get_started/eval_tutorial.md`
- 当前主线速记：Qwen3.6-27B，A800 `.22`（`-p 16834`）/ A100 `.16`（`-p 23422`），reward `.40:20111` 或 `.39:8111`，多轮 eval `--use-multi-turn --max-turns N`，context 65536，n_samples 8

仍需注意的工程点：

- eval-only 时 `slime/backends/megatron_utils/model.py:165` 保留 `scheduler_train_iters = max(args.train_iters, 1)` 最小保护。
- 客户端并发须 ≤ `--sglang-max-running-requests`，否则触发 router `503 / no_available_workers / all circuits open`。
- `expandable_segments:True` + TP≥4 自定义 all-reduce 会在 cuda-graph capture 崩，需 `--sglang-disable-custom-all-reduce`（见 memory）。

## 剩余工作（下一步：多轮训练）

1. **多轮训练 rollout**：把多轮轨迹接进训练采样路径（目前 `generate_rollout_async` 仍单轮，只复用默认 `generate_and_rm_group`）。确定 turn→`Sample` 形状，落地 `--padding-turns`（短轨迹补 placeholder 并下游 mask）。
2. **训练 `loss_mask`**：每轮只训练 assistant 生成 token；首轮 prompt、tool_response/feedback token、padded turn 全部 mask 掉。custom rollout 必须显式维护 `tokens` / `response_length` / `loss_mask` / `status` / `reward`。
3. **跨轮 reward 聚合 / advantage**：确定 last-turn / best-of-turn / per-turn 折扣（`--multi-turn-gamma`）语义，以及 GRPO 分组（同 prompt 的 n_samples 如何配对）。`--filter-by-last-turn` 决定 dynamic filter 作用在哪一轮。
4. **train-data 转换**：确认 slime 默认 `convert_samples_to_train_data` 能表达多轮字段；不行再补 custom converter。
5. **首次 GRPO 训练 run**：小规模先验证 loss / advantage / ckpt 保存；启动前 sanity check、跑完后 codex review（xhigh）再下结论（CLAUDE.md 政策）。
6. **reward 吞吐**：KernelGym `/evaluate` 同步、长尾会拖慢 rollout；训练规模下评估是否需要 batch / async / 限流。

## reward 映射（第一版，固定不在训练中临时变）

- 抽取失败：`0.0`（记 `extract_error`，不发 KernelGym）
- HTTP / timeout / 系统失败：`0.0`（记 `error_type`）
- 编译失败：`0.0`
- correctness false：`0.0`
- decoy kernel 检出（`decoy_kernel=True`）：`0.0`
- compiled + correctness + 非 decoy：`1.0`（`kernelgym_result_to_reward`：`1.0 if compiled and correctness and not decoy_kernel else 0.0`）

speedup shaping 和多轮 best-of-turn 聚合暂不接入，避免 reward 语义漂移、保证与旧 drkernel 可比。KernelGym 请求固定参数见 `kernelgym_rm.py`（client timeout `1800s`、`timeout` 默认 `90s`、`detect_decoy_kernel=true`、`enable_profiling=true`、`verbose_errors=true`；采样白名单 `kernelgym_num_correct_trials=5` / `num_perf_trials=50` / `num_warmup=30` / `perf_trim_count=5` / `reference_backend=pytorch`，不允许样本 metadata 覆盖；注意 `is_valid` 仍可被样本 metadata 覆盖）。

## 相关 handoff 索引

本文档只管「drkernel-on-slime 计划与状态」。以下专题各自有独立 handoff（早期版本把模板 ablation 和 W8A8 实验日志内嵌在本文档，现已迁出）：

- 模板 ablation（v1 / v2_3 / v2_4，27B 适合精简、9B 上 v2_3 反而退化的反向结论）：`handoffs/in_progress/handoff_prompt_templates.md`
- W8A8 rollout 量化（取代旧 Phase 2「W8A8 abandoned」叙述）：`handoffs/rollout_speedup/handoff_drkernel_w8a8_rollout.md`
- W4A16 AWQ：`handoffs/rollout_speedup/handoff_w4a16_awq.md`
- rollout 加速总览（hub）+ 子专题（prefix cache / reward 并发 / specdec / 设备效率 / 低精度 / W8A8 survey）：`handoffs/rollout_speedup/handoff_rollout_speedup.md` 及同目录
- origin/main 合并：`handoffs/in_progress/handoff_merge_origin_main.md`
- per-turn 精度证据（原 `handoff_drkernel_eval_accuracy.md` 已删）：`scripts/analysis/per_turn_acc_nopt.py`、`handoffs/in_progress/evidence_per_turn_quality.txt`；精度结论现并入模板 handoff

## 关键风险

- KernelGYM `/evaluate` 同步，长尾直接拖慢 rollout；训练规模可能要 batch/group reward、限流或异步提交/轮询。
- reward shaping 过早引入 speedup 容易和旧 drkernel 语义不一致；第一版只做 correctness reward。
- 多轮训练必须显式维护 `tokens` / `response_length` / `status` / `reward` / `loss_mask`，否则默认 train-data 转换不可靠。
- 训练容器与 KernelGYM endpoint 的可达性必须在容器内验证。
- 模板结论不能跨模型规模直接套用（27B 适合精简 v2_3，9B 上 v2_3 反而退化）；见模板 handoff。
