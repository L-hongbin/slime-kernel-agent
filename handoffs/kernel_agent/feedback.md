# Kernel agent 模型反馈与诊断

## 当前代码合同

入口是 [`build_model_feedback`](../../examples/kernel_agent/generate_with_cuda_agent.py) 与 [`normalize_env_feedback`](../../examples/kernel_agent/utils.py)，测试在 [`test_cuda_agent_model_feedback.py`](../../tests/test_cuda_agent_model_feedback.py)

- 在清理 compile artifact 前提取服务端静态预检的 code、位置和 snippet；最多显示 16 处，文本不超过 2,000 字符，并保留省略数量
- 预检未通过时省略未执行阶段的性能/decoy 字段；已有结构化错误时消除重复错误正文
- Sanitizer 保留检查状态、应用是否失败、issue 类型、源码位置和线程/地址范围，移除 raw tool output 和 replay payload
- 诊断只提供观察事实，不提供最近符号、编辑距离、修复建议或 checker 内部推导
- 当前客户端默认开启 `error_based` Sanitizer；请求设置与反馈压缩是两个独立合同

原始数据中只存在于 Sanitizer raw output 的 native-library 错误可能在压缩时丢失。`clean` 且 `target_application_failed=true` 也不能解释为程序正确；对应真实样例见[L3 多轮分析](../evaluation/kernelbench_l3_multiturn.md)

## 已验证的反馈压缩

2026-08-28 的审计把 KernelGym 的原始 `env_state` 与送给第二轮模型的 feedback 分成两个 contract：原始/normalized 结果继续供 reward、日志和审计使用；新增 `build_model_feedback` 只构造模型可行动的结构化摘要。当前 8,000 字符预算下，448 条真实 iter39 rollout 从 57/448 原始 feedback 会被截断，降为 0/448；compact JSON 最大 3,084 字符，没有触发最终 budget reduction

独立人工审计逐类检查了 validation、precheck、compilation、runtime/correctness、timeout、correct 和 decoy 样本。真实样本中的 compiler primary diagnostics、caret、notes、model-load detail、forbidden operators、profiling summary、backend probe、policy warnings 和 decoy reasons 均完整保留；原始 payload 无 mutation，冗余 metadata 没有进入模型 payload

## 实现 contract

模型侧 payload 明确保留：

- outcome：`status/error/precheck/compiled/correctness/decoy_kernel`；
- actionable error：清洗后的 `error_message`，以及 model-load/compiler/runtime error type/detail；
- performance：`speedup/kernel_runtime/reference_runtime`；
- correctness：issue、max/avg difference、failed/correct trial summary、forward/output-mismatch 状态；
- policy：ATen detection validity、forbidden operator `{name,count}`、enforced decoy reason、diagnostic-only policy warning、精简 backend probe；
- profiling：custom kernel names/count/coverage 和 missing-kernel summary

模型侧明确删除：

- `aten_detection_trials` 和完整/allowed ATen operator 明细；
- full profiler kernels、memory/event/TF32 state；
- backend probe 内嵌的第二份 profiler；
- device/host/node/worker/task/cache/compile artifact 信息；
- correctness 各 substage 的逐 trial timing arrays；
- 固定 backend/precision/execution-policy 配置；
- 顶层 `task_id/processing_time/success` 和任何未审核的新增字段

未知字段不自动透传。新增 KernelGym schema 若需进入下一轮 prompt，必须显式加入 model-feedback contract 和行为测试，避免未来 debug blob 重新污染上下文

## Compiler/runtime diagnostics 规则

Compiler 清洗只做高置信转换：

1. 完整 NVCC/C++ command 改成 `compile source -> object` 或 `link -> target` 摘要；
2. 删除重复 command、stdout/stderr wrapper 和固定 Ninja 收尾；
3. compile-cache/机器绝对路径只缩短，不删除 source file、line、column；
4. 保留所有能放入预算的 `error/fatal/undefined reference` block、source、caret、关联 `note` 和 terminal exception；
5. 未识别格式使用 head+tail fallback，不做猜测式删除；
6. primary blocks 本身超预算时，均匀抽取代表 block，并显式输出 `omitted X of Y primary diagnostic blocks`，不静默丢失

Runtime traceback 不删除 frame，只把机器路径归一化；最终异常、`model_new.py` frame、FFI/generated-binding frame 和源码行保留

## 多层预算与旁路防护

- 普通 detail 单字段最多 2,048 字符，列表/字典最多 32 项；任何缩减带 omission marker
- 嵌套深度最多 12，循环引用以 `cyclic reference omitted` 标记，异常 payload 不会令 rollout 递归崩溃
- 如果结构化 payload 仍超过 8K，按 16/8/4/2/1 项与 1,024/512/256/128/64 字符逐级结构化缩减；输出始终是合法 JSON，不再截断 JSON 字符串中段
- `max_feedback_chars=0` 只关闭最终总字符预算；结构化 schema、compiler command 清洗和单字段/list 安全上限仍然生效。这是新的 model-feedback contract，不再表示把原始 `env_state` 无损送入 prompt
- Format/Jinja template 必须恰好出现一次 plain `{feedback}` / `{{ feedback }}`。`feedback_dict`、重复插值、format spec 和 Jinja 运算均 fail-fast，不能绕过 8K 注入预算
- runtime 日志记录 `original_chars -> compacted_chars -> final_chars`、是否 structured reduction；原始 `env_state` 不被覆盖

## 448 条真实数据审计

数据：`local_artifacts/qwen38/ctx24_dynamic_filter_20260825/fixed_step40_medium_temp10_n16/dumps/rollout_data/eval_0.pt`。该文件位于 `slime-dev-csl-2` 主 worktree 的 ignored `local_artifacts`，是 iter39、28 UUID × 16 的真实 Qwen3.8 rollout。审计使用正式 Qwen3.8-FP8 tokenizer

| 指标 | 原 feedback | model feedback |
|---|---:|---:|
| 总字符 | 1,346,017 | 约 219K |
| 字符 mean | 3,004.50 | 约 490 |
| 字符 p50 / p90 / p95 | 371 / 9,736 / 13,840 | 236 / 1,218 / 1,390 |
| 字符 max | 61,105 | 3,084 |
| 超过 8K | 57/448（12.72%） | 0/448 |
| tokenizer tokens 总量 | 473,796 | 约 68.6K |
| tokenizer tokens mean | 1,057.58 | 约 153 |
| token 节省 | — | 约 405K（85.5%） |

按 outcome 的字符中位数/最大值：

| outcome | n | raw p50 / max | compact p50 / max |
|---|---:|---:|---:|
| validation | 301 | 371 / 371 | 236 / 236 |
| precheck | 20 | 1,347 / 1,351 | 438 / 438 |
| compilation | 30 | 1,131 / 8,025 | 547 / 3,084 |
| kernel eval/runtime/correctness failure | 57 | 7,381 / 13,840 | 约 1,245 / <2K |
| timeout | 1 | 947 / 947 | 414 / 414 |
| correct | 32 | 19,942 / 29,933 | 约 1,043 / 约 1.1K |
| decoy | 7 | 17,437 / 61,105 | 1,074 / 1,821 |

## 语义与泄漏检查

最终 448-sample 逐条检查结果：

- raw input mutation：0/448；
- invalid compact JSON：0/448；最终 >8K：0/448；
- compiler primary diagnostics：25/25；caret：全部保留；notes：7/7；
- model-load detail：17/17；
- forbidden ATen summaries：7/7；
- profiling summaries：32/32；
- backend probes：51/51；
- diagnostic-only policy warnings：23/23，且不再误命名为 decoy reason；
- enforced decoy reasons：7/7；
- 冗余 key 泄漏：0。检查 key 包括 `aten_detection_trials/aten_ops/allowed_aten_ops/profiling/device_info/compile_artifact/correctness_trial_s/compile_hostname/cpu_worker_id/task_id/processing_time/additional_details`

人工代表样本：

- compilation：index 49、54、55、88、320、341、366、373、378；覆盖 type mismatch、missing header、TVM FFI API misuse、undefined launcher、CUDNN symbol 等；
- runtime/FFI：218、220、287；terminal `AttributeError/TypeError/RuntimeError` 和定位 frame 保留；
- correct/profile：210、219、323、377；speedup、kernel names、count/time coverage 保留；
- decoy：289、335、375；forbidden operator name/count 与 enforced reason 保留；
- 其余 outcome 每类至少两条人工对照，timeout 类只有一条

## 自动化检查

CPU 行为测试 `tests/test_cuda_agent_model_feedback.py`，不测试 launcher 参数。覆盖：

- keep/drop/summarize schema 和 raw input immutability；
- compiler source/caret/note/multiple-error/fallback/omission；
- Unicode 与严格字符预算；
- format/Jinja template budget bypass；
- per-field/list omission、unknown fields、cyclic/deep metadata；
- extreme valid JSON 的结构化总预算；
- runtime path normalization；
- normalize 阶段不再消费 raw `compile_artifact.error`

该次审计运行了 33 个 CPU cases。联合原 `tests/test_cuda_kernel_eval.py`、`tests/test_kernel_agent_fully_async_rollout.py`、`tests/test_predictive_mask_rollout_plumbing.py` 和 `tests/test_kernel_agent_partial_reward.py` 的 targeted suite 共 106 passed、2 个原有 opt-in integration skipped，确认 normalize、环境结果、multiturn/predictive feedback 和日志行为没有回归。真实 KernelGym integration case 仍由原测试的 opt-in gate 控制；本轮核心证据来自保留的真实 KernelGym rollout，而不是合成结果

额外 property/fuzz 使用 seeds `0,1,7,42,20260828`，覆盖 50 个 build payload、80 个 compiler diagnostics、55 个 template/budget cases，以及 Unicode、100 error blocks、循环、1,200 层嵌套和百万项超长 warning list。最终结果为 0 exception、0 mutation、0 positive-cap violation、0 invalid JSON、0 omission-marker failure、0 template bypass；百万项 warning 只扫描/保留有界前缀，输出 31 项加明确的 999,969 项省略标记

## 正式两轮 feedback 样本

2026-08-29 正式 rollout 0 完成前记录了 911 条逐 turn feedback，包含之后被动态过滤的尝试：

| 指标 | 原 feedback | compact/final feedback |
|---|---:|---:|
| 字符 p50 / p95 | 371 / 15,210 | 236 / 5,013 |
| 字符 max | 40,602 | 6,483 |
| 最终触发 8K 截断 | — | 0/911 |

这补充了单轮重建样本之外的真实两轮长度分布；观测中的余量不能作为后续训练分布的上限

## 训练链路验证

两轮和三轮部署的 context、token-history 与 full-loop 证据见[训练验证](../qwen38/training_validation.md)。反馈字符数和 tokenizer 审计仍由本文维护

## 正式 rollout 1–6 的 compiler note 补充（2026-08-29）

后续正式分布暴露了一个早于最终 8K budget 的子级压缩边界：`feedback_truncated=False` 只表示最终结构化 JSON 没有 reduction，不表示 6K compiler diagnostics 内没有省略信息。正式 `train_rollout_capture/rollout_1.pt` 至 `rollout_6.pt` 共 1,536 条两轮轨迹，其中 583 条第一轮编译失败；121 条进入 oversized diagnostic excerpt，100 条至少丢失一个原始 note occurrence，共 2,819 行。去除重复和 macro definition/expansion 后，96 条轨迹仍至少缺一条可行动 note，主要是 `candidate:` 与 `no known conversion`

修复只改变超过 6K 且完整 primary/note excerpt 仍放不下的分支，6K compiler cap 和最终 8K feedback cap 均保持不变。新优先级为：

1. 完整 primary error 行；只有这些行本身放不下时才均匀采样，并保留 primary omission marker；
2. 最终 terminal exception；
3. 去重后的高价值 note，优先保留两组 `candidate + rejection reason`，最多四行；
4. `[omitted N additional unique actionable diagnostic notes]`，其中 `N` 只统计去重、去 macro 后仍未显示的可行动 note；
5. source/caret 和前置上下文使用剩余空间，单行最多 384 字符；oversized 分支中的 macro definition/expansion note 不再占预算。未超预算 diagnostics 仍保持原来的近似无损行为

583 条真实编译失败轨迹逐条新旧回放结果：

| 指标 | 旧实现 | 修复后 |
|---|---:|---:|
| 原本缺 note 且得到有效补充的轨迹 | 0/96 | 96/96 至少新增 1 条；超过四条时显式省略 |
| 完整新增 note 数 | — | 1/2/3/4 条分别为 1/74/1/20 条轨迹 |
| normalized primary error 完整行缺失数 | 126 | 0 |
| 旧版可见 primary error 回归 | — | 0 |
| terminal exception 保留 | 1/1 | 1/1 |
| compiler diagnostics 超过 6K | 0 | 0 |
| 最终 feedback 触发 8K reduction | 0 | 0 |
| 最终 feedback p95 / max | — | 6,360 / 6,669 chars |

102/583 条 compiler feedback 文本发生变化；相对旧实现的字符变化 mean/p50/p95/min/max 为 `+181.6/+161/+461/-150/+557`。变化不仅来自补 note，也来自删除 oversized 分支中的 macro 噪声和更紧凑的 primary/context 打包。受影响第二轮当前 prompt 最大 25,575 tokens，离 32,768 总上限仍有明显余量；exact token history 不变，因为修改只发生在第一轮生成历史之后的 feedback suffix

额外使用 seed `20260829` 生成 1,000 组含 1–100 个 primary errors、长 source、macro/candidate/rejection notes 和可选 terminal exception 的 diagnostics，并覆盖 128/129/256/511/512/600/1,000/2,048/6,000 字符预算；结果为 0 exception、0 strict-cap violation

人工复核发现 terminal 紧邻 sampled primary 时会在 supplement 和 context 中重复；实现排除 `context_index == terminal_index`，并用长 primary 后接 terminal 的样例确认 terminal 只出现一次

人工检查了三类真实反例：

- rollout 1 `parallel_task_000738_a5f292b4`：原始 26,935 chars、20 个 primary、12 条 unique actionable notes；修复后 5,999 chars，20/20 primary 完整，保留两组 candidate/rejection，并显示 omitted 8；
- rollout 4 `parallel_task_002392_48424f24`：原始 28,420 chars、40 个 primary；旧版 block 内截断导致 40 条 normalized primary 均不能完整匹配，修复后 40/40 完整，同时保留 3 条 actionable notes；
- rollout 3 `parallel_task_001879_972ddd3c`：唯一 terminal `NameError: name 'Tensor' is not defined` 在新旧实现中均完整保留

这些数据只证明旧 feedback 存在信息损失以及修复成本很低；丢 note 组第二轮正确率低约 10 个百分点仍有日志长度、错误数量和任务难度混杂，不能解释为该修复必然带来 10 个百分点收益。这项比较不构成训练质量提升的因果证明；复现实验需绑定实际 feedback-policy 版本

## 剩余边界

- 固定预算不可能在任意多 primary errors 时完整保留全部文本；当前策略明确报告 total/shown/omitted，并优先保证不静默丢失
- 正式配置预算是 8K；若调用方刻意把预算设到约 18 字符以下，合法 JSON 与完整 omission marker 无法同时容纳，最小 fallback 只能退化为 `{}`。直接调用 compiler sanitizer 且给出小于 128 字符的非正式预算时，也可能只能做通用中间截断；正式 `build_model_feedback` 路径对该分项设有 512 字符下限
- 未审核的新 KernelGym metadata 默认不会进入模型 prompt；这是安全默认值，但 schema 演进时需要显式维护 allowlist
- 448 dump 是 iter39 的 24K 单轮结果重建；正式两轮 rollout 0 已把新分布的 runtime gap 补齐，当前观测最大 6,483 字符，距离 8K 仍有 1,517 字符余量。后续模型分布可能随训练变化，仍需按现有 runtime 指标持续观察


## 本地结构化预检的 token 成本审计

For the saved Qwen3.8 `group=7`, T2 response, the existing error remains:

```text
Code precheck failed: TVM-FFI model calls are not exported: ml_forward_ops
```

The additional payload is:

```json
{
  "code": "TVM_FFI_UNRESOLVED_CALL",
  "phase": "binding_contract",
  "evidence": [
    {
      "kind": "extension_call",
      "value": "ml_forward_ops",
      "section": "MODEL_NEW",
      "line": 32,
      "column": 9,
      "snippet": "tvm_ffi_extension.ml_forward_ops(x.contiguous(), W1, b1, W2, b2, W3, b3, buf1, buf2, output)"
    },
    {
      "kind": "exported_symbol",
      "value": "mlp_forward_ops",
      "section": "APPLY_BINDINGS",
      "line": 59,
      "column": 31
    }
  ]
}
```

Feedback is serialized with `json.dumps(feedback_dict, ensure_ascii=False, default=str)` and no indentation. The Qwen3.8 tokenizer counts 115 tokens for the old rendered tool-response prompt and 265 for the new one, an increase of 150 tokens.

### 2,000 条响应的离线回放

Input:

- Official DeepSeek-V4-Flash-0731 KernelBench L3 evaluation
- 400 trajectories, five turns each, 2,000 saved responses
- Dump: `local_artifacts/qwen38/official_l3_5turn_eval_20260901/final_inputs/dsv4/dumps/rollout_data/eval_0.pt` in the source worktree
- Original prompt template: `multi_turn_tvm_ffi_short.yaml`
- Tokenizers loaded from the exact official Qwen3.8 and DeepSeek-V4-Flash checkpoint directories

The dump contains 244 recorded precheck failures. Local replay identifies 45 failures handled by the in-repository precheck; the other 199 are remote KernelGym static-check results and therefore receive no new payload. Four of the 45 local failures occur at T5 and would not be inserted into another model turn, leaving 41 actual new feedback messages.

#### Token increase for the 41 feedback messages actually inserted into a next turn

| tokenizer | min | median | mean | p95 | max | total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.8-27B-FP8 | 25 | 72 | 78.71 | 105 | 220 | 3,227 |
| DeepSeek-V4-Flash-0731 | 33 | 87 | 92.22 | 126 | 241 | 3,781 |

The corresponding character increase has median 250, mean 260.24, p95 364, and maximum 680.

#### Mean increase by local diagnostic code

| code | next-turn feedbacks | Qwen3.8 tokens | DeepSeek tokens | maximum DeepSeek tokens |
| --- | ---: | ---: | ---: | ---: |
| `TVM_FFI_HOST_CUDA_MARKER_FORBIDDEN` | 23 | 82.83 | 98.43 | 126 |
| `MODEL_NEW_INVALID` | 6 | 68.00 | 80.00 | 80 |
| `TVM_FFI_EXTENSION_CALL_MISSING` | 6 | 25.00 | 33.00 | 33 |
| `MODEL_NEW_PYTHON_SYNTAX` | 4 | 84.50 | 96.50 | 114 |
| `TVM_FFI_UNRESOLVED_CALL` | 2 | 213.00 | 226.50 | 241 |

Across all 244 stored precheck failures, including remote-string failures and terminal T5 failures that add no next-turn prompt, the actual additions average 13.23 Qwen3.8 tokens or 15.50 DeepSeek tokens and the median is zero. Across affected trajectories, the cumulative DeepSeek increase has median 95, p95 179.60, and maximum 241 tokens.

### CPU 成本

A single-process replay over the same 2,000 responses took 3.92 seconds with the old precheck and 4.60 seconds with the structured diagnostics enabled. Per response this is 1.96 ms versus 2.30 ms, an increase of 0.34 ms. This is CPU-only parsing work and does not add a compiler, GPU, or KernelGym request.



该成本审计比较的是 2026-09-03 仅扩展本地 precheck 的版本，base commit 为 `ee230db1fff62295b14b9c0d9990ecd3a6342b6a`；244 个旧失败中有 199 个远端字符串错误当时没有新增字段。服务端结构化 findings 现已进入反馈，不能把旧版本的 token 成本或覆盖率当作当前上限
