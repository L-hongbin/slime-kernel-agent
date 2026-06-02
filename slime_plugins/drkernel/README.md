# slime_plugins/drkernel

DrKernel custom rollout plugin 入口。这里是 rollout、prompt 渲染、KernelGym reward
和输出抽取的实现位置；实验状态和结论不要写进本目录，放到 handoff。

## Runtime Code

- `args.py`: DrKernel custom rollout / reward 相关参数。
- `rollout.py`: 多轮 rollout、prompt 渲染、template selection 和样本输出组织。
- `kernelgym_rm.py`: async KernelGym HTTP client 和 optional custom RM wrapper。
- `eval_throttle.py`: KernelGym eval coroutine fan-out 限流工具。
- `extract.py`: 从 assistant response 中抽取 KernelGym-ready kernel submission。

## Prompt Templates

- `prompt_templates/prompts_v1.yaml`: 当前动态 prompt profile 配置。
- `prompt_templates/layouts/`: turn/layout 模板。
- `prompt_templates/backends/`: backend-specific first-turn 模板。
- `prompt_templates/roles/`: correctness/performance 等 role 模板。
- `prompt_templates/tool_response/`: tool feedback 压缩模板。
- `prompt_templates/legacy/`: 旧版 prompt reference，仅用于对照和迁移。

## Design Docs

- `design-docs/dynamic_prompt_templates.md`: fragment-based dynamic prompt template
  设计。
- `design-docs/feedback_summarization.md`: KernelGym feedback summarization 设计。
