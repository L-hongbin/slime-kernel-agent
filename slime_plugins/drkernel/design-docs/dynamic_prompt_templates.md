# DrKernel Dynamic Prompt Templates

## 目标

支持 first-turn 模板的动态可组合选择，同时保留与旧 DrKernel first-turn 模板的内容等价（空白归一化后逐字一致，不含新插入的运行环境信息）。在不修改协议骨架的前提下，同一份 profile 也可被多轮 rollout 复用——每个 backend candidate 同时挂着 `first_turn_text_path` 和 `tool_response_text_path`，turn-0 取前者、turn≥1 取后者，输出格式天然一致。

当前结构：

- `role`：角色定义，可在多个候选里选择。
- `backend`：旧模板中 role 之后、`{{ problem }}` 之前的任务规则主体。
- `environment`：由 rollout 参数注入的可选运行环境信息。
- `problem`：样本里的原始题目。
- `tool_response`：每个 backend candidate 自带的 turn≥1 反馈模板，确保输出格式与 turn-0 backend 的三段式（CUDA_KERNELS / APPLY_BINDINGS / MODEL_NEW）保持一致。

layout（`layouts/first_turn.jinja`，单/多轮共用 turn-0 user 内容）：

```jinja
{{ role }}

<optional environment block>

{{ backend }}

{{ problem }}
```

## 非目标

- 不把最终 prompt 展开写入 parquet。
- 不在 slime 通用 `Dataset` 里硬编码 DrKernel 模板逻辑。
- profile 不限制轮数上限：协议层只描述"turn-0 用什么模板、turn≥1 反馈用什么模板"，**总轮数由 CLI `--max-turns` 控制**。
- 不把 NVCC/GPU 作为模板文件；它们只是运行配置字符串。

## 当前模板来源

旧模板原文来源（用于 legacy 等价测试）：

```text
/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent/drkernel/kernel/config/prompt_config/cuda_templates/first_turn/csl_cuda_agent.jinja
.../first_turn/lhb_v3.jinja
.../first_turn/lhb_v4.jinja
.../first_turn/pybind11_module.jinja
.../first_turn_tvm_ffi/tvm_ffi_module.jinja
```

## 目录布局

```text
slime_plugins/drkernel/
  prompt_templates/
    prompts_v1.yaml                  # 协议描述，包含 profiles: drkernel_v1, drkernel_v1_tvm_ffi
    layouts/
      first_turn.jinja               # turn-0 user 消息骨架（单/多轮共用）
    roles/
      accelerate_best_perf.jinja
      optimize_correctness.jinja
    backends/
      csl_cuda_agent.jinja
      lhb_v3.jinja
      lhb_v4.jinja
      pybind11_module.jinja
      tvm_ffi_module.jinja
    tool_response/                   # turn≥1 用户消息模板，按 backend 系列分组
      pybind_short.jinja             # 通用 pybind 系（csl_cuda_agent / lhb_v3 / pybind11_module）
      pybind_default.jinja           # pybind 系详细版（暂未启用）
      tvm_ffi_short.jinja            # tvm_ffi 系
    legacy/
      first_turn/
      first_turn_tvm_ffi/
      tool_response/
```

`legacy/` 保存迁移到 slime 后的 legacy 参考副本，便于审查和测试。`roles/` 是可选 role。`backends/` 是 active backend 候选。`tool_response/` 是 turn≥1 模板池——通过 YAML 字段 `tool_response_text_path` 与各 backend candidate 耦合，避免 turn-0 用 pybind 输出格式、turn-1 又要求 tvm_ffi 输出的错配。

## 等价拆分规则

对每个旧 first-turn 模板：

1. 抽出首句里的通用身份。
2. 将首句剩余任务描述和后续规则放入对应 backend。
3. layout 插入 role、可选环境信息、backend 和 problem。

内容等价判断：

```text
normalize_whitespace(render(old_template, problem=x))
==
normalize_whitespace(render(new_layout_without_environment, role=matching_role, backend=split_backend, problem=x))
```

只有 legacy 中存在相同 role 句子的组合才做等价测试。选择另一个 role 或注入环境信息是有意变体，不纳入旧模板等价判断。

## 环境信息

layout 支持这些可选变量：

- `compiler_name`：例如 `nvcc_12_4`。
- `gpu_name`：例如 `H100`。
- `extra_environment`：其他需要补充的自由文本。

`nvcc` 和 `cuda_version` 通常重复，因为 NVCC 版本基本代表 CUDA toolkit 版本。除非后续明确要区分 runtime/driver CUDA，否则先只保留 `compiler_name`。

## YAML 配置

`prompts_v1.yaml` 描述协议层的可组合空间。当前包含两个 profile：

```yaml
profiles:
  drkernel_v1:
    layout: layouts/first_turn.jinja
    role:
      select: cycle
      candidates:
        - id: accelerate_best_perf
          text_path: roles/accelerate_best_perf.jinja
        - id: optimize_correctness
          text_path: roles/optimize_correctness.jinja
    backend:
      select: cycle
      candidates:
        - id: csl_cuda_agent
          first_turn_text_path: backends/csl_cuda_agent.jinja
          tool_response_text_path: tool_response/pybind_short.jinja
        - id: lhb_v3
          first_turn_text_path: backends/lhb_v3.jinja
          tool_response_text_path: tool_response/pybind_short.jinja
        - id: pybind11_module
          first_turn_text_path: backends/pybind11_module.jinja
          tool_response_text_path: tool_response/pybind_short.jinja
        - id: tvm_ffi_module
          first_turn_text_path: backends/tvm_ffi_module.jinja
          tool_response_text_path: tool_response/tvm_ffi_short.jinja
  drkernel_v1_tvm_ffi:
    layout: layouts/first_turn.jinja
    role:
      select: cycle
      candidates: [...]                # 与上一致
    backend:
      select: cycle
      candidates:
        - id: tvm_ffi_module
          first_turn_text_path: backends/tvm_ffi_module.jinja
          tool_response_text_path: tool_response/tvm_ffi_short.jinja
```

当前 active profile 由 `slime_plugins/drkernel/rollout.py` 硬编码为 `drkernel_v1_tvm_ffi`。需要切换时改源码常量；不引入 CLI flag。

legacy 等价性不靠 YAML 固化。测试里指定 legacy 对应 role/backend 组合，验证组合结果和本地 `legacy/` 参考副本内容一致。`lhb_v4` 保留为 legacy 文件，但不在 active 候选中。

## 数据格式

parquet/jsonl 只保存任务内容和唯一的可选模板 allowed list：

```json
{
  "ground_truth": "原始 KernelBench/DrKernel problem 文本",
  "extra_info": {
    "task_id": "kernelbench_l1_0001",
    "template_allowed": {
      "role": ["accelerate_best_perf", "optimize_correctness"],
      "backend": ["csl_cuda_agent", "pybind11_module"]
    }
  }
}
```

运行参数仍使用：

```bash
--prompt-data data/drkernel-rl-data-0513/train.parquet
--input-key ground_truth
--metadata-key extra_info
```

custom rollout 负责最终 prompt 渲染和 chat template，因此不要在 Dataset 阶段启用 `--apply-chat-template`。DrKernel rollout 会在选完 first-turn template 后调用 tokenizer 的 `apply_chat_template(..., add_generation_prompt=True)`，再把结果发给 SGLang。

## Renderer 接口

`DrKernelPromptRenderer` 暴露四个公共方法 + 一个内部 helper：

| 方法 | 用途 |
|---|---|
| `render_first_turn_messages(args, sample, rollout_id) -> list[dict]` | 渲染 turn-0 user 消息内容，写入 `sample.metadata["messages"]`/`["chosen_prompt_slots"]`/`["drkernel_user_prompt"]`/`["raw_problem"]`，返回 `[{"role":"user","content":...}]` |
| `render_tool_response_message(args, sample, feedback) -> dict` | 根据已选 backend candidate 查 `tool_response_text_path`，渲染并返回 `{"role":"user","content":...}` |
| `materialize_prompt(args, sample, messages)` | `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **apply_chat_template_kwargs)`，结果写回 `sample.prompt` 并清空 `sample.tokens` |
| `apply_to_sample(args, sample, rollout_id)` | 单轮便捷封装：`render_first_turn_messages` + `materialize_prompt` |
| `_lookup_candidate(slot_name, candidate_id)` | 反查 YAML candidate；供 `render_tool_response_message` 拿耦合字段使用 |

`sample.metadata["messages"]` 在多轮中是状态载体——单一事实源，每轮调用方往里追加 `{"role":"assistant", "content": resp}` 和 `render_tool_response_message(...)` 的结果，然后再 `materialize_prompt` 一次即可形成下一轮的 `sample.prompt`。

## Rollout 渲染流程

### 单轮（eval/train，turn 0 即终止）

1. 从 `data_source.get_samples(args.rollout_batch_size)` 取样本组。
2. 对每个 `Sample` 读取 `sample.prompt` 作为原始 problem。
3. 使用固定 profile `drkernel_v1_tvm_ffi`。
4. `apply_to_sample(args, sample, rollout_id)` —— 内部依次调 `render_first_turn_messages` 和 `materialize_prompt`。
5. 记录选择结果到 `sample.metadata["chosen_prompt_slots"]`。
6. 把 first-turn user prompt 保存到 `sample.metadata["drkernel_user_prompt"]`。
7. 调用 `slime.rollout.sglang_rollout.generate_and_rm_group()` 继续生成和打分。

选择结果示例：

```json
{
  "chosen_prompt_slots": {
    "profile": "drkernel_v1_tvm_ffi",
    "role": "accelerate_best_perf",
    "backend": "tvm_ffi_module",
    "compiler_name": "nvcc_12_4",
    "gpu_name": "H100"
  }
}
```

### 多轮（`--use-multi-turn`，turn 1…N）

每轮：

1. **首轮初始化**：`render_first_turn_messages(args, sample, rollout_id)` 写入 `sample.metadata["messages"]`；`materialize_prompt(...)` 得到 turn-0 的 `sample.prompt`；提交 SGLang。
2. **拿到 assistant response 后**：追加 `{"role":"assistant", "content": response}` 到 `sample.metadata["messages"]`。
3. **环境反馈**：调用 KernelGym/RM 拿到 `feedback` 字符串。
4. **追加 user 反馈**：`tool_msg = render_tool_response_message(args, sample, feedback)`；`sample.metadata["messages"].append(tool_msg)`。
5. **重渲染**：`materialize_prompt(args, sample, sample.metadata["messages"])`；得到 turn-(t+1) 的 `sample.prompt`；提交 SGLang。
6. 直到达到 `args.max_turns` 或终止条件（拿到合格 reward / 上下文超限）。

注意事项：

- **跨轮 `apply_chat_template_kwargs` 必须保持一致**，否则 token 化前缀漂移，SGLang prefix-cache 失效。
- assistant 历史里可能带 `<think>...</think>`；是否在下轮被 tokenizer 剥除，由 `--preserve-history-thinking` 翻译成的 `apply_chat_template_kwargs` 决定。不要在 Python 端手动剥。
- 当 sample 是 `--padding-turns` 模式补出来的占位 turn 时，需要把 `metadata["is_padding"]=True` 标好，下游训练阶段据此打 loss mask。

## CLI 参数

DrKernel plugin 端参数（`slime_plugins/drkernel/args.py`）：

```bash
# 环境注入
--drkernel-compiler-name nvcc_12_4
--drkernel-gpu-name H100

# 多轮控制
--use-multi-turn
--max-turns 3
--padding-turns
--preserve-history-thinking
--multi-turn-gamma 1.0
--filter-by-last-turn
```

`--max-turns` 不在 YAML profile 里设上限。同一份 `drkernel_v1_tvm_ffi` profile 既支持单轮（不开 `--use-multi-turn`）也支持任意多轮。

## 最小可复现示例（`drkernel_v1_tvm_ffi`）

> 需要可用的 tokenizer 路径；下例用 Qwen3.5-9B。

```python
# minimal_repro_drkernel_v1_tvm_ffi.py
from argparse import Namespace
from types import SimpleNamespace

from slime_plugins.drkernel.rollout import DrKernelPromptRenderer

renderer = DrKernelPromptRenderer(
    hf_checkpoint="/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.5-9B",
)

args = Namespace(
    apply_chat_template_kwargs=None,
    drkernel_compiler_name="nvcc_12_4",
    drkernel_gpu_name="H100",
    drkernel_extra_environment=None,
    rollout_seed=0,
)

sample = SimpleNamespace(
    index=0,
    prompt=(
        "import torch\n"
        "import torch.nn as nn\n\n"
        "class Model(nn.Module):\n"
        "    def forward(self, x):\n"
        "        return torch.relu(x)\n"
    ),
    metadata={},
    tokens=[],
)

# --- turn 0: render first-turn user message ---
messages = renderer.render_first_turn_messages(args, sample, rollout_id=0)
print("[turn-0] chosen_prompt_slots =", sample.metadata["chosen_prompt_slots"])
print("[turn-0] user content head:\n", messages[0]["content"][:240], "...\n")

# --- simulate model response + KernelGym feedback ---
messages.append({
    "role": "assistant",
    "content": "<think>draft kernel...</think>\n### CUDA_KERNELS\n```cpp\n// kernel code\n```\n...",
})
feedback = "compile error: undefined symbol my_kernel_launcher in apply_bindings.cpp:24"

# --- turn 1: render tool_response user message ---
tool_msg = renderer.render_tool_response_message(args, sample, feedback)
messages.append(tool_msg)
print("[turn-1] tool_response content:\n", tool_msg["content"], "\n")

# --- materialize prompt for turn 1 (apply chat template) ---
renderer.materialize_prompt(args, sample, messages)
print("[turn-1] sample.prompt tail:\n", sample.prompt[-260:])
```

预期输出（截断；具体 chat-template 前后缀以你的 tokenizer 为准）：

```text
[turn-0] chosen_prompt_slots = {
  'profile': 'drkernel_v1_tvm_ffi',
  'role': 'accelerate_best_perf',
  'backend': 'tvm_ffi_module',
  'compiler_name': 'nvcc_12_4',
  'gpu_name': 'H100',
}

[turn-0] user content head:
You are a PyTorch and CUDA expert. Accelerate the given PyTorch Model by creating a high-performance CUDA C++ extension, targeting the best possible performance faster than baseline.

Target environment:
- NVCC: nvcc_12_4
- GPU: H100

Do not use inline CUDA strings or call `tvm_ffi.cpp.build`, `tvm_ffi.cpp.load`, ...

[turn-1] tool_response content:
Use this feedback to revise your previous CUDA + TVM-FFI implementation:

compile error: undefined symbol my_kernel_launcher in apply_bindings.cpp:24

Return a full improved implementation with `CUDA_KERNELS`, `APPLY_BINDINGS` (TVM-FFI, no pybind), and `MODEL_NEW` (using `tvm_ffi_extension`).

[turn-1] sample.prompt tail:
...Return a full improved implementation with `CUDA_KERNELS`, `APPLY_BINDINGS` (TVM-FFI, no pybind), and `MODEL_NEW` (using `tvm_ffi_extension`).<|im_end|>
<|im_start|>assistant
```

要点解读：

- `drkernel_v1_tvm_ffi` profile 的 `backend` 只有 `tvm_ffi_module` 一个 candidate，所以 turn-0 backend 固定。role 走 cycle，basis=`sample.index=0` → `accelerate_best_perf`。
- turn-1 的 `tool_response_text_path` 通过 `_lookup_candidate("backend", "tvm_ffi_module")` 反查到 `tool_response/tvm_ffi_short.jinja`，自动与 turn-0 的输出格式保持一致。
- `materialize_prompt` 输出的尾部一定是 `<|im_start|>assistant\n`（`add_generation_prompt=True`）。多轮循环下次拿到这个 `sample.prompt` 直接送 sglang。

## 后续扩展

后续可新增非等价 profile（例如 `drkernel_v2_experimental`），进一步拆分 backend 内部规则。`tool_response_text_path` 字段是协议层入口，新增 backend 时必填，缺失时 `render_tool_response_message` 会立即 `KeyError`，保证多轮场景不会因为漏配置而静默退化。

## 验证要求

实现后至少保留一个人工可审查 dump 文件，包含原始 problem、选择的 role/template id、环境信息和最终 prompt 前后若干行，例如：

```text
checkpoints/drkernel_prompt_debug/rollout_000001.sample.txt
```

最小单元测试应覆盖：

- YAML profile 加载（`tests/utils/test_drkernel_prompt_templates.py::test_drkernel_prompts_yaml_is_composable`）。
- role/backend/tool_response 候选路径存在。
- legacy 等价：固定可精确还原 legacy 的 role/backend 组合；不带环境信息时，组合渲染结果和旧模板渲染结果在空白归一化后相同。
- 通过 `sample.metadata["template_allowed"]` 固定 role/backend 后选择正确候选；不再提供第二套模板选择入口。
- 多轮：`render_tool_response_message` 在缺失 `chosen_prompt_slots` 或缺失 `tool_response_text_path` 时报错；同 sample 的 turn-1 渲染走的是与 turn-0 backend 相同系列的 tool_response 模板。
