# DrKernel Dynamic Prompt Templates

## 目标

先支持单轮 rollout 中的动态 first-turn 模板选择，同时保持和旧 DrKernel first-turn 模板的内容等价。这里的等价不要求空行数量逐字一致，也不包含新插入的运行环境信息。

当前结构：

- `role`：角色定义，可在两个候选里选择。
- `backend`：旧模板中 role 之后、`{{ problem }}` 之前的任务规则主体。
- `environment`：由 rollout 参数注入的可选运行环境信息。
- `problem`：样本里的原始题目。

layout：

```jinja
{{ role }}

<optional environment block>

{{ backend }}

{{ problem }}
```

## 非目标

- 不把最终 prompt 展开写入 parquet。
- 不在 slime 通用 `Dataset` 里硬编码 DrKernel 模板逻辑。
- 单轮阶段不实现 tool response、多轮 memory、环境反馈模板。
- 不把 NVCC/GPU 作为模板文件；它们只是运行配置字符串。

## 当前模板来源

旧模板原文来源：

```text
/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent/drkernel/kernel/config/prompt_config/cuda_templates/first_turn/csl_cuda_agent.jinja
/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent/drkernel/kernel/config/prompt_config/cuda_templates/first_turn/lhb_v3.jinja
/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent/drkernel/kernel/config/prompt_config/cuda_templates/first_turn/lhb_v4.jinja
/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent/drkernel/kernel/config/prompt_config/cuda_templates/first_turn/pybind11_module.jinja
/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent/drkernel/kernel/config/prompt_config/cuda_templates/first_turn_tvm_ffi/tvm_ffi_module.jinja
```

## 目录布局

```text
slime_plugins/drkernel/
  prompt_templates/
    single_turn_v1.yaml
    layouts/
      single_turn.jinja
    roles/
      accelerate_best_perf.jinja
      optimize_correctness.jinja
    backends/
      csl_cuda_agent.jinja
      lhb_v3.jinja
      lhb_v4.jinja
      pybind11_module.jinja
      tvm_ffi_module.jinja
    tool_response/
      default.jinja
      short.jinja
    legacy/
      first_turn/
      first_turn_tvm_ffi/
      tool_response/
```

`legacy/` 保存迁移到 slime 后的 legacy 参考副本，便于审查和测试。`roles/` 是可选 role。`backends/` 是 active backend 候选。

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

`single_turn_v1.yaml` 只描述可组合空间，不为 legacy 等价性硬编码多个 profile：

```yaml
profiles:
  drkernel_single_turn_v1:
    layout: layouts/single_turn.jinja
    role:
      select: cycle
      candidates:
        - id: accelerate_best_perf
          text_path: roles/accelerate_best_perf.jinja
        - id: optimize_correctness
          text_path: roles/optimize_correctness.jinja
    first_turn_template:
      select: cycle
      candidates:
        - id: csl_cuda_agent
          backend_text_path: backends/csl_cuda_agent.jinja
        - id: lhb_v3
          backend_text_path: backends/lhb_v3.jinja
        - id: pybind11_module
          backend_text_path: backends/pybind11_module.jinja
        - id: tvm_ffi_module
          backend_text_path: backends/tvm_ffi_module.jinja
```

legacy 等价性不靠 YAML 固化。测试里指定 legacy 对应 role/backend 组合，验证组合结果和本地 `legacy/` 参考副本内容一致。`lhb_v4` 保留为 legacy 文件，但不在 `drkernel_single_turn_v1` 的 active backend 候选中。目前 active profile 覆盖的可精确还原组合包括 `csl_cuda_agent + optimize_correctness`、`pybind11_module + optimize_correctness`、`tvm_ffi_module + optimize_correctness`。

## 数据格式

parquet/jsonl 只保存任务内容和唯一的可选模板 allowed list：

```json
{
  "ground_truth": "原始 KernelBench/DrKernel problem 文本",
  "extra_info": {
    "task_id": "kernelbench_l1_0001",
    "template_allowed": {
      "role": ["accelerate_best_perf", "optimize_correctness"],
      "first_turn_template": ["csl_cuda_agent", "pybind11_module"]
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

## Rollout 渲染流程

单轮 custom rollout 的处理顺序：

1. 从 `data_source.get_samples(args.rollout_batch_size)` 取样本组。
2. 对每个 `Sample` 读取 `sample.prompt` 作为原始 problem。
3. 使用固定 profile `drkernel_single_turn_v1`。
4. 从 role 和 backend 候选中选择片段；唯一的限制入口是 `sample.metadata["template_allowed"]`。如果它限制了某个 slot，就只在 allowed list 内按 profile 策略选择。allowed list 长度为 1 时等价于固定该 slot。
5. 从 args 读取 `compiler_name`、`gpu_name`、`extra_environment`。
6. 渲染 layout，得到 first-turn user prompt。
7. 记录选择结果到 `sample.metadata["chosen_prompt_slots"]`。
8. 将 first-turn user prompt 保存到 `sample.metadata["drkernel_user_prompt"]`。
9. 使用 tokenizer 对 `[{"role": "user", "content": user_prompt}]` 调用 `apply_chat_template(..., add_generation_prompt=True)`。
10. 将 chat-template 后的最终 prompt 写回 `sample.prompt`。
11. 调用 `slime.rollout.sglang_rollout.generate_and_rm_group()` 继续生成和打分。

选择结果示例：

```json
{
  "chosen_prompt_slots": {
    "profile": "drkernel_single_turn_v1",
    "role": "accelerate_best_perf",
    "first_turn_template": "lhb_v3",
    "compiler_name": "nvcc_12_4",
    "gpu_name": "H100"
  }
}
```

## CLI 参数

建议新增 DrKernel plugin 参数：

```bash
--drkernel-compiler-name nvcc_12_4
--drkernel-gpu-name H100
```

## 后续扩展

后续如果要做真正可自由组合的模板，可以新增非等价 profile，例如 `drkernel_composable_experimental_v1`。它可以进一步拆分 backend 内部规则，但不能声称和旧模板内容等价。

多轮阶段再加入 `tool_response` profile、environment feedback、loss mask 规则。

## 验证要求

实现后至少保留一个人工可审查 dump 文件，包含原始 problem、选择的 role/template id、环境信息和最终 prompt 前后若干行，例如：

```text
checkpoints/drkernel_prompt_debug/rollout_000001.sample.txt
```

最小单元测试应覆盖：

- YAML profile 加载。
- role/backend 候选路径存在。
- 测试中固定可精确还原 legacy 的 role/backend 组合；不带环境信息时，组合渲染结果和旧模板渲染结果在空白归一化后相同。
- 通过 `sample.metadata["template_allowed"]` 固定 role/backend 后选择正确候选；不再提供第二套模板选择入口。
