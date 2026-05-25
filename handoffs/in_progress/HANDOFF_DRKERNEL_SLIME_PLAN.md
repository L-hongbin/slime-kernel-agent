# 用 slime 重新实现 drkernel 的计划

## 目标

用 slime 承担训练、rollout、数据组织、checkpoint 和日志；KernelGYM 保持为独立 HTTP reward/eval 服务，复用现有实现，不在 slime 内重写 kernel 编译、运行、计时和 worker 调度。

计划分 7 步推进。只有第 1 步需要先做细，因为它决定后续接口、数据格式和失败语义。

## 关键路径

- slime 项目路径：`/nfs/FM/chenshuailin/projects/kernel_agents/slime`
- KernelGYM 项目路径：`/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent`
- KernelGYM HTTP 服务代码：`/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent/kernelgym/server/api/server.py`
- KernelGYM 请求/响应模型：`/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-vllm018-cuda-agent/kernelgym/server/api/models.py`

## 当前状态

截至 2026-05-16，当前已经完成的是单轮 DrKernel eval 闭环，不是完整多轮 DrKernel RL：

- 已完成已转换 parquet 的 slime Dataset 接入方案：`--input-key ground_truth --label-key ground_truth --metadata-key extra_info`；`Sample.prompt` 用于动态 prompt，`Sample.label` 用作 KernelGYM `reference_code`。
- 已完成 DrKernel 动态首轮 prompt scaffold：`slime_plugins/drkernel/prompt_templates/single_turn_v1.yaml`、role/backend 片段、固定 `drkernel_single_turn_v1` profile。
- 已完成 `slime_plugins.drkernel.rollout.generate_rollout` 基础入口：从默认 data source 取样，渲染 DrKernel user prompt，在 rollout 内部调用 tokenizer `apply_chat_template(..., add_generation_prompt=True)`，再交给 SGLang 生成。
- 已完成单轮 CUDA-Agent submission 抽取：`slime_plugins/drkernel/extract.py`，严格要求 `CUDA_KERNELS -> APPLY_BINDINGS -> MODEL_NEW` 三段顺序完整。
- 已完成 KernelGYM async HTTP client 与单轮 reward wrapper：`slime_plugins/drkernel/kernelgym_rm.py`；通过 `--rm-url` 指向 KernelGYM 服务。
- 已完成可人工 review 的真实数据 prompt dump：`checkpoints/drkernel_prompt_debug/formatted_prompt_examples.txt`。
- 已在 `.67` 容器内完成一次 Qwen3.5-9B KernelBench L1 eval 跑通验证：`checkpoints/Qwen3.5-9B/run_20260515_122920.log`，Ray job `raysubmit_v7whRX3AfMxtL3YY` succeeded，`800/800` 完成，`eval/kernelbench_level1 = 0.035`。
- 尚未完成多轮 feedback、turn 控制、多轮停止条件、best-of-turn reward 聚合和最终多轮 `loss_mask` 语义。

下一步是把当前单轮 eval 闭环推进到多轮：每轮生成后抽取 kernel submission，调用 KernelGYM `/evaluate`，把编译/正确性/性能反馈转成下一轮 user message，并明确训练 token mask 与最终 reward 聚合。

## 2026-05-15 调试问题与修复记录

这批改动尚未 commit。当前工作区仍有多处未提交修改，提交前需要重新 review diff。

### 已验证环境与结果

- 训练/rollout 容器：`ssh -p 23452 root@192.168.16.67`
- KernelGYM endpoint：`--rm-url http://192.168.16.39:8111`
- 启动脚本：`scripts/debug.sh`
- Eval config：`scripts/eval_kernelbench_level1.yaml`
- 成功日志：`checkpoints/Qwen3.5-9B/run_20260515_122920.log`
- 成功 Ray job：`raysubmit_v7whRX3AfMxtL3YY`
- 结果：`eval/kernelbench_level1 = 0.035`，`response_len/mean = 9750.64`，`response_len/max = 31814`，`truncated_ratio = 0.03125`

### 遇到的问题和对应改动

| 问题表现 | 原因判断 | 已做改动 | 当前状态 |
| --- | --- | --- | --- |
| `ray job submit` 后终端和 `checkpoints/.../run_*.log` 不稳定同步，Ctrl-C 不一定能正确反映 Ray job 状态 | `ray job submit` 默认等待/日志流和外层 shell 日志处理混在一起，job agent 未 ready 时还可能直接失败 | `scripts/ray/start_cluster.sh::submit_ray_job()` 改为 `ray job submit --no-wait`，解析 submission id 后用 `ray job logs --follow` 跟随日志，并用 `ray job status` 返回真实状态；保留 agent 未 ready 重试 | 成功 run 的日志完整进入 `run_20260515_122920.log` |
| Ray job agent 启动早期报 `No available agent to submit job` | Ray dashboard/job agent 比 `ray start` 返回更晚 ready | `submit_ray_job()` 对该错误做限时重试 | 不再阻塞当前 eval |
| 数据加载时报 `prompt must be a list when processor is not None` | Qwen3.5 checkpoint 会加载 processor；非多模态 string prompt 被 Dataset 按 processor 路径校验 | `slime/rollout/data_source.py` 和 eval rollout 中仅在 `args.multimodal_keys is not None` 时加载 processor | converted parquet 的 string prompt 正常加载 |
| Eval path 需要 prompt、response、prompt+response 都限制到 32768 | 原 eval config 只覆盖 response 长度，不够表达统一 context 上限 | `slime/utils/eval_config.py` 增加 `max_prompt_len`、`max_context_len`；`slime/rollout/sglang_rollout.py` 在发请求前根据 `_slime_max_context_len` cap `max_new_tokens`；`scripts/eval_kernelbench_level1.yaml` 设置三者为 `32768` | SGLang 日志显示 `context_len=32768` |
| SGLang ready check 看起来卡住 | 需要用生成路径健康检查，且缺少周期日志时不清楚在等什么 | `slime/backends/sglang_utils/sglang_engine.py::_wait_server_healthy()` 使用 `/health_generate`，设置 HTTP timeout，周期性打印等待信息 | 保持 `/health_generate`，启动可观测 |
| SGLang router 出现 `503/no_available_workers/all circuits open` | 服务端 `--sglang-max-running-requests 64` 生效，但客户端 semaphore 仍可能按更高 `sglang_server_concurrency` 发太多并发 | `slime/rollout/sglang_rollout.py::GenerateState` 和 `slime/utils/http_utils.py` 当前将客户端并发限制为 `int(min(args.sglang_server_concurrency * 1.5, args.sglang_max_running_requests)) * num_engines`；未设置 max-running 时退化为 `int(args.sglang_server_concurrency * 1.5) * num_engines` | 待用该更小改动重跑确认；上一版更严格 cap 已跑通 |
| `--sglang-context-length 32768` 后 KV/Mamba 预分配仍很大 | Qwen3.5/GDN 模型的 SGLang Mamba cache 与 KV pool 预分配不只由普通 context len 线性决定 | 已记录现象到 `SPEC.md`；未继续改 SGLang 内部 | 当前通过限制并发跑通，显存行为仍需后续专门分析 |
| Eval-only 初始化时 `train_iters=0` 可能影响 scheduler | slime `num_rollout=0` eval-only 仍会初始化 actor 和 optimizer scheduler，以便加载权重并推给 rollout | 只保留 `model.py` 中 `scheduler_train_iters = max(args.train_iters, 1)` 的最小保护；不再跳过 optimizer、不关闭训练 actor、不特殊禁用 ref | 对源代码保持更小改动；正式是否加载 ref 仍由 `--use-kl-loss` / `kl_coef` 控制 |
| KernelBench eval metadata 没有统一 `uuid` / `entry_point` | converted train/eval parquet metadata 字段不完全一致 | `slime_plugins/drkernel/kernelgym_rm.py::_get_uuid()` 读取 `uuid`，否则 `problem_id/name`；`_get_entry_point()` 默认 `"Model"`；eval config 注入 `entry_point: Model`、`is_valid: true` | `KeyError: entry_point` 未复现 |
| 模型输出可能包含 reasoning 或不完整 submission | KernelGym CUDA-Agent 后端只应吃完整三段 markdown submission | `slime_plugins/drkernel/extract.py` 去掉 think 区域，只接受最后一个完整 `CUDA_KERNELS -> APPLY_BINDINGS -> MODEL_NEW` 组；缺失时返回 `None`，reward 为 `0.0` | 单轮 eval 可跑通，后续多轮要基于该失败语义生成反馈 |

### 当前关键参数

```bash
--custom-rm-path slime_plugins.drkernel.kernelgym_rm.custom_rm
--rollout-function-path slime_plugins.drkernel.rollout.generate_rollout
--prompt-data data/drkernel-rl-data-0513/train.parquet
--input-key ground_truth
--label-key ground_truth
--metadata-key extra_info
--num-rollout 0
--eval-config scripts/eval_kernelbench_level1.yaml
--eval-max-prompt-len 32768
--eval-max-response-len 32768
--eval-max-context-len 32768
--rm-url http://192.168.16.39:8111
--sglang-context-length 32768
--sglang-max-running-requests 64
--sglang-mem-fraction-static 0.7
```

当前 `scripts/debug.sh` 里 `--use-kl-loss` 已注释；eval-only 当前只验证 rollout/eval/reward 链路，不代表正式 GRPO 训练配置已经最终确定。

## 第 1 步：冻结接口与最小闭环 smoke

### 目的

先把 slime 与 KernelGYM 的边界钉死，避免后面在 rollout、reward、prompt、多轮反馈里同时改太多东西。第 1 步不做正式训练，只做一个最小闭环：

```text
KernelBench 样本
  -> slime 侧 prompt / metadata 表示
  -> 模型生成 kernel answer
  -> slime 侧提取 kernel_code
  -> KernelGYM HTTP /evaluate
  -> slime Sample.reward / metadata
  -> 可人工 review 的 jsonl 证据
```

### 现有能力与待补充逻辑

结论：drkernel 迁移应该按 slime customization/plugin 方式实现。`docs/en/get_started/customization.md` 里的扩展点都是通过 import path 加载函数，适合把 drkernel 专用逻辑放在 `slime_plugins/drkernel/`，再由 run 脚本通过参数接入；不要把任务专用逻辑写进 `slime/` core。

建议 plugin 结构：

```text
slime_plugins/drkernel/
  __init__.py
  data.py              # 读取已转换 schema、模板变量校验、prompt/message helper
  extract.py           # 从模型回答中抽取 kernel_code
  kernelgym_rm.py      # KernelGYM HTTP client/reward helper，供 rollout 调用
  rollout.py           # --rollout-function-path: drkernel 多轮 rollout 主入口
  logging.py           # 可选: --custom-rollout-log-function-path
```

当前计划已经确定最终要做多轮，因此主路线直接实现 `slime_plugins.drkernel.rollout.generate_rollout`。默认 data source 仍负责读取已离线转换好的 parquet；custom rollout 已先完成单轮首轮 prompt 与 chat-template scaffold，后续继续在同一入口中补 KernelGYM 反馈、停止条件、reward、loss mask 和 metadata。

| 模块 | 对应 slime custom arg | slime 已有/只需参数 | drkernel/KernelGYM 对应代码位置 | 需要补充的 drkernel 逻辑 | 建议位置 |
| --- | --- | --- | --- | --- | --- |
| 数据文件读取 | `--prompt-data`、`--input-key ground_truth`、`--label-key ground_truth`、`--metadata-key extra_info`；通常不需要 `--data-source-path` | `slime/utils/data.py::read_file()` 已支持 `.jsonl` / `.parquet` | 已有离线转换脚本：`scripts/data/convert_verl_to_slime.py`；旧训练数据是 parquet：`drkernel/README.md` 的 `TRAIN_DATASET` / `VALID_DATASET`；训练加载入口：`drkernel/kernel/kernel_trainer.py` 用 `RLHFDataset(parquet_files=self.config.data.train_files)` | 数据格式已离线转换成 slime 可读 schema：顶层 `ground_truth` 同时作为输入字段和 reference label，`extra_info` 作为 metadata 字段；后续只需要校验字段完整性 | 已有 `scripts/data/convert_verl_to_slime.py`；测试 `tests/utils/test_convert_verl_to_slime_data.py` |
| prompt 字段选择 | `--prompt-data`、`--input-key ground_truth`、`--label-key ground_truth`、`--metadata-key extra_info`；custom rollout 可选择是否自行 apply chat template | `Dataset` 能读取 string prompt 或 OpenAI messages，并可调用 tokenizer chat template | 转换脚本把 `reward_model.ground_truth` 提到顶层 `ground_truth`，把 `data_source` / `ability` 合并进 `extra_info` | custom rollout 从 `sample.prompt` 构造第一轮 messages；KernelGym `reference_code` 从 `sample.label` 读取 | `slime_plugins/drkernel/rollout.py`、`slime_plugins/drkernel/kernelgym_rm.py` |
| DrKernel 任务模板 | `--rollout-function-path slime_plugins.drkernel.rollout.generate_rollout` | slime 原生 `--apply-chat-template` 是 tokenizer chat template，不是 drkernel 的 Jinja 任务模板系统 | 模板读取/渲染：`drkernel/kernel/workers/rollout/prompt_templates.py`；legacy 模板文件包括 `drkernel/kernel/config/prompt_config/cuda_templates/first_turn/lhb_v4.jinja`；当前 slime active profile 是 `slime_plugins/drkernel/prompt_templates/single_turn_v1.yaml` | custom rollout 运行时渲染首轮模板和后续 feedback/tool response 模板，并记录选中模板 | `slime_plugins/drkernel/rollout.py`，模板 helper 可放 `slime_plugins/drkernel/data.py` |
| metadata 传递 | `--metadata-key extra_info` | `Dataset` 会写入 `Sample.metadata` | `scripts/data/convert_verl_to_slime.py::merge_into_extra_info()`；旧字段映射参考：`drkernel/run_cuda_reward_dir.py::_build_remote_task()` | 以 `extra_info` 作为 `Sample.metadata`；custom rollout/reward 从中读取 `entry_point`、`uuid`、`data_source`、`ability` 等，并补齐 KernelGYM request 所需默认值 | 已有转换脚本；运行时校验在 `slime_plugins/drkernel/data.py` |
| rollout 取样 | 默认不设；继续用默认 data source | `slime/rollout/data_source.py::RolloutDataSource` 已处理 shuffle、epoch、`n_samples_per_prompt` deepcopy | 旧 drkernel 离线结果按样本文件处理；slime online 取样由默认 DataSource 接管 | 不改 `--data-source-path`；custom rollout 调 `data_source.get_samples(args.rollout_batch_size)` 后接管每条样本的多轮轨迹 | `slime_plugins/drkernel/rollout.py` |
| 模型生成 | 由 `--rollout-function-path` 内部调用 SGLang；不单独用 `--custom-generate-function-path` | 默认 generate 可参考 `slime.rollout.sglang_rollout` 复用 token/response 填充方式 | 旧离线流程从 `messages` 取最后一条 assistant 内容；online 生成由 custom rollout/SGLang 接管 | custom rollout 每轮构造 messages、调用 rollout engine、更新 `tokens`、`response`、`response_length`、`loss_mask` | `slime_plugins/drkernel/rollout.py` |
| 代码抽取 | 由 `--rollout-function-path` 内部调用；可复用 `extract.py` | slime 没有 drkernel 专用 `kernel_code` 抽取 | 旧抽取参考：`drkernel/run_cuda_reward_dir.py::_build_remote_task()` 调 `extract_kernel_submission(response, kernel_backend=\"cuda_agent\")` | 从每轮 `sample.response` 中抽取 kernel submission；失败时生成反馈或终止，并记录 `extract_error` | `slime_plugins/drkernel/extract.py`，由 `rollout.py` 调用 |
| reward 接入 | 由 `--rollout-function-path` 内部调用 KernelGYM client；可选保留 `--custom-rm-path` 做单轮 smoke | `slime/rollout/rm_hub` 支持 custom RM，但多轮主线不靠它硬塞完整行为 | 旧 reward request 构造参考：`drkernel/run_cuda_reward_dir.py::_build_remote_task()`；reward 结果归一化参考：`_normalize_result()` / `_summarize()` | 实现 KernelGYM HTTP client，custom rollout 每轮调用 `/evaluate`，并决定反馈、停止条件和最终 reward | `slime_plugins/drkernel/kernelgym_rm.py` 作为 client/helper，`slime_plugins/drkernel/rollout.py` 负责编排 |
| reward 后处理 | 可选 `--custom-reward-post-process-path` | 默认 GRPO/RLOO reward 处理通常够用 | 旧 drkernel reward 权重/penalty 参考：`drkernel/kernel/rewards/kernel_reward.py`、`drkernel/kernel/rewards/reward_client.py` | 第一版不做 speedup shaping；后续如果要复现旧 reward 权重，再补 post-process | 可选 `slime_plugins/drkernel/reward_postprocess.py` |
| KernelGYM 服务 | 通过 slime 已有 `--rm-url` 指定 KernelGYM endpoint | KernelGYM 已有 `/evaluate`、`/evaluate/batch`、`/health`、`/workers/status` | HTTP 路由：`kernelgym/server/api/server.py`；请求/响应模型：`kernelgym/server/api/models.py::EvaluationRequest` / `EvaluationResponse` | 不在 slime 里重写编译/运行/计时；只补 HTTP 调用、timeout、错误分类、结果摘要 | `slime_plugins/drkernel/kernelgym_rm.py`、`scripts/drkernel/smoke_kernelgym_reward.py` |
| Sample 到训练 batch | 默认不设；只有默认转换不够时用 `--custom-convert-samples-to-train-data-path` | 默认转换会消费 `tokens`、`response_length`、`reward`、`loss_mask` 等 | 旧多轮训练字段和 turn 统计参考：`drkernel/kernel/main_grading.py` 的 `global_turn_indices`、`turn_token_stats` 相关逻辑 | custom rollout 先直接填好 `loss_mask` / `metadata.round_number`，尽量走默认转换；如果默认转换无法表达多轮训练字段，再补 custom converter | 可选 `slime_plugins/drkernel/train_data.py` |
| 数据加载验证 | 无训练 arg；smoke 脚本直接调用 | slime 没有 drkernel 专用 smoke | 旧 parquet 数据路径解析参考：`drkernel/kernel/scripts/rl/train_rl_common.sh::format_dataset_paths()`；训练加载参考：`drkernel/kernel/kernel_trainer.py` | 加载 3 条已转换 parquet/jsonl，dump 实际 `Sample.prompt` 和 `Sample.metadata`，确认字段没有丢 | `scripts/drkernel/smoke_slime_data_loading.py` |
| reward 验证 | 无训练 arg；smoke 脚本模拟 `--custom-rm-path` | slime 没有 KernelGYM reward smoke | 旧离线 smoke/reward 批处理参考：`drkernel/run_cuda_reward_dir.py`；KernelGYM API contract：`kernelgym/server/api/models.py` | 用 2-4 条样本和手写回答跑 `/evaluate`，产出可人工 review 的 request/response/reward | `scripts/drkernel/smoke_kernelgym_reward.py` |
| checkpoint/log/Ray | run 脚本参数；可选 `--custom-rollout-log-function-path` | slime 现有训练脚本模式和 `scripts/ray/start_cluster.sh` 可复用 | KernelGYM 作为独立 HTTP 服务；历史 endpoint 线索来自 handoff：直连 `http://192.168.16.39:8111`，relay `http://127.0.0.1:18111` | 增加 drkernel 专用 run 脚本参数，记录 `--rm-url`，小规模验证 checkpoint | `scripts/run-drkernel-*.sh`、`SPEC.md` |
| 多轮反馈 | `--rollout-function-path slime_plugins.drkernel.rollout.generate_rollout` | `--rollout-function-path` 支持自定义 rollout | 旧多轮/反馈语义需要从 drkernel 历史 prompt、messages、KernelGYM response artifacts 对齐；HTTP workflow 可参考 `kernelgym/server/api/server.py` 的 `/workflow/*` 路由 | 实现编译/正确性/性能反馈进入下一轮 user message、turn 控制、stop reason、loss mask | `slime_plugins/drkernel/rollout.py` |

建议第一版 run 参数：

```bash
--prompt-data data/drkernel-rl-data-0513/train.parquet
--input-key ground_truth
--label-key ground_truth
--metadata-key extra_info
--rollout-function-path slime_plugins.drkernel.rollout.generate_rollout
```

不要在 Dataset 阶段启用 `--apply-chat-template`。DrKernel custom rollout 需要先完成动态 prompt 选择，再在 `rollout.py` 内部调用 tokenizer chat template。

`--custom-rm-path` 可保留给单轮 reward smoke，但多轮主线不依赖它。

### 要做的具体事情

1. 对齐旧 drkernel 数据字段和 KernelGYM 请求字段。

   先不要凭空设计字段，直接以旧 drkernel 的实际请求构造为准：

   ```text
   KernelGYM-vllm018-cuda-agent/drkernel/run_cuda_reward_dir.py
     _build_remote_task(sample_path, sample)
   ```

   这个函数是第一版迁移的字段来源：

   - 旧数据字段 `original_python_code` / 已转换后的 `ground_truth` -> `Sample.label` -> KernelGYM `reference_code`
   - 旧数据字段 `entry_point` -> KernelGYM `entry_point`，默认 `"Model"`
   - 旧数据字段 `uuid` -> KernelGYM `uuid`；slime 侧必须由 `extra_info.uuid` 提供
   - KernelGym HTTP API 额外要求 `task_id`，slime 侧按旧 `KernelRewardClient` 语义生成 `parallel_task_{counter}_{uuid4}`，不支持 metadata override
   - assistant 最后一条回答 -> 抽取 `kernel_code`
   - 固定 `use_reference_cache=True`
   - 固定 `is_valid=False`

   KernelGYM 接口字段以这个文件为准：

   ```text
   KernelGYM-vllm018-cuda-agent/kernelgym/server/api/models.py
     EvaluationRequest
     EvaluationResponse
   ```

2. 记录当前已完成的数据 schema 转换。

   当前已有转换脚本：

   ```text
   scripts/data/convert_verl_to_slime.py
   ```

   它已经把 VERL/drkernel parquet 离线转换为 slime 可读 schema。核心规则是：

   - `reward_model.ground_truth` -> 顶层 `ground_truth`
   - 删除原 `prompt` 和 `reward_model`
   - `data_source` / `ability` -> `extra_info.data_source` / `extra_info.ability`
   - 删除 `extra_info.original_prompt`
   - 保留 `extra_info.entry_point`、`extra_info.uuid` 等字段

   已有测试：

   ```text
   tests/utils/test_convert_verl_to_slime_data.py
   ```

   对应 slime 参数：

   ```bash
   --prompt-data data/drkernel-rl-data-0513/train.parquet
   --input-key ground_truth
   --label-key ground_truth
   --metadata-key extra_info
   ```

   转换后的每条记录形如：

   ```json
   {
     "ground_truth": "...",
     "extra_info": {
       "entry_point": "Model",
       "uuid": "...",
       "data_source": "...",
       "ability": "..."
     }
   }
   ```

   后续 plugin 运行时把 `Sample.label` 和 `extra_info` 组合成 KernelGYM request 所需字段，例如 `reference_code`、`entry_point`、`uuid`、`task_id=parallel_task_{counter}_{uuid4}`、`backend=cuda`、`workflow=kernelbench`、`use_reference_cache=true`、`is_valid=false`。

3. 明确 slime 数据加载逻辑写在哪里。

   第一版分工如下：

   - 不改：`slime/rollout/data_source.py`
   - 不改：`slime/utils/data.py`
   - 已有：`scripts/data/convert_verl_to_slime.py`
   - 新增：`slime_plugins/drkernel/data.py`，只放运行时 schema 校验、模板变量提取等 helper

   原因是当前 `Dataset` 已经满足需要：

   - `read_file()` 支持 jsonl/parquet
   - `prompt_key=args.input_key`
   - `metadata_key=args.metadata_key`
   - `Sample(metadata=metadata)` 会保留 reward/rollout 所需字段
   - `RolloutDataSource.get_samples()` 会 deepcopy prompt sample，保证同一 prompt 的多条采样共享 metadata

   不再新增 `scripts/drkernel/normalize_kernelbench_for_slime.py`。

4. 实现 custom rollout scaffold。

   因为最终目标是 drkernel 多轮，这一步直接新增：

   ```text
   slime_plugins/drkernel/rollout.py
   ```

   接口：

   ```python
   def generate_rollout(args, rollout_id, data_source, evaluation=False):
       ...
   ```

   当前已完成的职责：

   - `data_source.get_samples(args.rollout_batch_size)`
   - 从 `sample.prompt` 取得 `ground_truth`
   - 从 `sample.metadata` 取得 `extra_info`
   - 渲染首轮 DrKernel 任务模板
   - 在 rollout 内部调用 tokenizer chat template
   - 复用 SGLang generate 路径生成单轮回答

   待补充的职责：

   - 抽取 kernel submission
   - 调 KernelGYM `/evaluate`
   - 把 KernelGYM feedback 渲染成下一轮 user message
   - 控制最大 turn、失败停止、正确性/性能停止
   - 填充或修正训练需要的 `tokens`、`response_length`、`reward`、`status`、`loss_mask`、`metadata`

5. 写数据加载 smoke，先验证 metadata 没丢。

   推荐新增：

   ```text
   scripts/drkernel/smoke_slime_data_loading.py
   ```

   它只做一件事：用 slime 的 `Dataset` 加载 3 条已转换 parquet/jsonl，dump 出实际 `Sample` 摘要。

   输出路径：

   ```text
   handoffs/in_progress/assets/drkernel_slime_smoke/data_samples.jsonl
   ```

   每条至少检查：

   - `sample.prompt` 对应 `ground_truth`
   - `sample.label` 对应 `ground_truth`，作为 KernelGYM `reference_code`
   - `sample.metadata` 对应 `extra_info`
   - `sample.metadata["entry_point"] == "Model"` 或旧数据显式值
   - `sample.metadata["uuid"]` 存在
   - `sample.metadata["data_source"]` 存在
   - `sample.metadata["ability"]` 存在

6. 定义模型输出到 `kernel_code` 的提取规则。（已完成单轮版本）

   当前只支持 CUDA-Agent 三段 markdown submission，避免训练早期 reward 语义漂移：

   ````text
   ### CUDA_KERNELS
   ```cpp
   ...
   ```

   ### APPLY_BINDINGS
   ```cpp
   ...
   ```

   ### MODEL_NEW
   ```python
   ...
   ```
   ````

   `slime_plugins/drkernel/extract.py` 会去掉 think 区域，并选择最后一个完整三段组。如果没有完整 submission，`extract_kernel_submission()` 返回 `None`，当前 custom RM 直接给 `reward=0.0`，不发给 KernelGYM。

7. 写一个 slime 侧 KernelGYM client 原型。（已完成单轮版本）

   当前实现放在：

   ```text
   slime_plugins/drkernel/kernelgym_rm.py
   ```

   现有入口：

   ```python
   async def evaluate_sample(args, sample, *, kernel_code=None, client=None)
   async def custom_rm(args, sample_or_samples, **_)
   ```

   它已经负责：

   - 从 `sample.label` 取 `reference_code`；缺失时直接报错
   - 从 `sample.metadata` 取 `entry_point`、`uuid/problem_id/name` 等
   - 按旧 `KernelRewardClient` 风格生成 `parallel_task_{counter}_{uuid4}` task id
   - `POST {args.rm_url}/evaluate`
   - 把 KernelGYM request/response/reward 写入 `sample.metadata["kernelgym"]`

   后续多轮 rollout 仍需要把 KernelGYM response 转成下一轮 feedback，并决定最终 reward 聚合。

8. 固定 reward 映射，不在训练中临时变。

   初始建议：

   - 提取失败：`0.0`
   - HTTP/timeout/系统失败：`0.0`，并记录 `error_type`
   - 编译失败：`0.0`
   - correctness false：`0.0`
   - correctness true：基础 `1.0`，后续再加 speedup shaping

   speedup shaping 不放进第一版，先保证正确性语义和 drkernel 历史结果可比。

9. 做一个离线 reward smoke 脚本。

   推荐路径：

   ```text
   scripts/drkernel/smoke_kernelgym_reward.py
   ```

   输入 2-4 条 KernelBench 样本和 2-4 条手写模型输出，输出：

   ```text
   handoffs/in_progress/assets/drkernel_slime_smoke/results.jsonl
   handoffs/in_progress/assets/drkernel_slime_smoke/summary.json
   ```

   `results.jsonl` 每行保留：

   - 原始 prompt id
   - 提取出的 `kernel_code`
   - KernelGYM request 摘要
   - KernelGYM response 摘要
   - slime reward
   - failure bucket

10. 验证 KernelGYM 服务可达性。

   在训练节点上检查：

   ```bash
   curl --max-time 8 ${RM_URL}/health
   curl --max-time 8 ${RM_URL}/workers/status
   ```

   如果训练跑在容器里，必须在容器内检查同一个 URL，而不是只在宿主机检查。

### 第 1 步完成标准

- 有一个固定的 `--rm-url` 配置方式。
- 已转换 parquet 能用 `--input-key ground_truth --label-key ground_truth --metadata-key extra_info` 被 slime `Dataset` 加载。
- 单轮 eval 已能通过 custom rollout/custom RM 构造 request 调到 KernelGYM，并已在 `.67` 跑完 `800/800` KernelBench L1。
- 仍需要补成功、编译失败、提取失败至少各一个的可 review 小样本 artifact。
- `results.jsonl` 可以人工确认 ground_truth、DrKernel prompt、response、kernel_code、KernelGYM feedback、reward 是一致的。
- 不启动 PPO 训练，不改 KernelGYM 服务实现。

## 第 2 步：确认已转换数据与 Dataset 入口

数据离线转换已经由 `scripts/data/convert_verl_to_slime.py` 完成，输出为 slime 可读 parquet。当前要做的是固定 Dataset 参数、补 smoke，并确认字段在 `Sample` 中的位置：

```bash
--prompt-data data/drkernel-rl-data-0513/train.parquet
--input-key ground_truth
--label-key ground_truth
--metadata-key extra_info
```

`ground_truth` 进入 `Sample.prompt` 和 `Sample.label`；前者用于动态 prompt，后者用于 KernelGYM `reference_code`。`extra_info` 进入 `Sample.metadata`。正式多轮 rollout 中，DrKernel Jinja prompt 和 tokenizer chat template 的调用时机由 `slime_plugins/drkernel/rollout.py` 控制。

交付物：保留 `scripts/data/convert_verl_to_slime.py`、`tests/utils/test_convert_verl_to_slime_data.py`，新增或完善 `scripts/drkernel/smoke_slime_data_loading.py`。

## 第 3 步：实现 drkernel custom rollout scaffold（当前单轮已完成）

直接接入：

```bash
--rollout-function-path slime_plugins.drkernel.rollout.generate_rollout
```

当前 custom rollout 已完成单轮 scaffold：取样、首轮模板选择、首轮 user prompt 渲染、tokenizer chat template、SGLang 生成。单轮 eval path 已能接入 KernelGYM reward；它还不是完整多轮 agent loop。

已完成交付物：`slime_plugins/drkernel/rollout.py`、`slime_plugins/drkernel/prompt_templates/single_turn_v1.yaml`、`tests/utils/test_drkernel_prompt_templates.py`、`checkpoints/drkernel_prompt_debug/formatted_prompt_examples.txt`。

已完成单轮 reward 相关交付物：`slime_plugins/drkernel/extract.py`、`slime_plugins/drkernel/kernelgym_rm.py`、`tests/utils/test_drkernel_extract.py`、`tests/utils/test_drkernel_kernelgym_rm.py`。

剩余交付物：多轮 rollout loop、反馈模板、turn 控制、最终 reward 聚合、训练样本 `loss_mask` 语义，以及 10-20 prompt 的多轮 rollout+reward review artifact。

## 第 4 步：接入 KernelGYM reward/evaluate（单轮已完成）

先不要直接做完整多轮。当前单轮 reward/eval 已完成这些内容：

1. 新增 `slime_plugins/drkernel/extract.py`，从 assistant response 中抽取 CUDA-Agent kernel submission。
2. 新增 `slime_plugins/drkernel/kernelgym_rm.py`，实现 async KernelGYM HTTP client 和 `custom_rm`。
3. 单轮 eval 通过 `generate_and_rm(..., evaluation=True)` 调用 custom RM，custom RM 内部执行 extract + KernelGYM `/evaluate`。
4. 将 KernelGYM 结果映射成 `sample.reward`；原始 request/response/reward 写入 `sample.metadata["kernelgym"]`。
5. 在 `.67` 上用 Qwen3.5-9B 跑完 KernelBench L1 eval：`800/800`，`eval/kernelbench_level1 = 0.035`。

仍未完成：

1. 多轮 rollout 内部直接调用 KernelGYM，并根据 response 生成下一轮 feedback。
2. `sample.metadata["failure_bucket"]`、`sample.metadata["stop_reason"]` 的多轮语义。
3. 成功、提取失败、编译失败、correctness false 的小规模人工 review artifact。
4. 多轮 best-of-turn 或 final-turn reward 聚合。

第一版 reward 映射保持简单：

- 提取失败：`0.0`
- HTTP/timeout/系统失败：`0.0`
- 编译失败：`0.0`
- correctness false：`0.0`
- correctness true：`1.0`

KernelGym 请求第一版固定这些 DrKernel 兼容参数：client timeout `1800s`；请求 `timeout`（等价旧 `reward_model.task_timeout`）默认 `90s`；`detect_decoy_kernel=true`；`enable_profiling=true`；`verbose_errors=true`。其中 KernelGym API 字段名是 `verbose_errors`，不是单独的 `verbose`。评测采样参数保留显式可配白名单：`kernelgym_num_correct_trials` 默认 `5`、`kernelgym_num_perf_trials` 默认 `50`、`kernelgym_num_warmup` 默认 `30`、`kernelgym_perf_trim_count` 默认 `5`、`kernelgym_reference_backend` 默认 `torch_compile`；这些值不允许从样本 metadata 覆盖。

speedup shaping 和多轮 best-of-turn 聚合暂不接入，避免 reward 语义漂移。

已完成交付物：`slime_plugins/drkernel/extract.py`、`slime_plugins/drkernel/kernelgym_rm.py`、`tests/utils/test_drkernel_extract.py`、`tests/utils/test_drkernel_kernelgym_rm.py`。

剩余交付物：`scripts/drkernel/smoke_kernelgym_reward.py`、`handoffs/in_progress/assets/drkernel_slime_smoke/results.jsonl`、`summary.json`。

## 第 5 步：对齐 drkernel 多轮反馈语义

将 custom rollout 产出的 messages、feedback、turn 数、stop reason、reward 分布和旧 drkernel artifacts 做人工对照。重点确认：

- 首轮 prompt 和选中的模板组合是否一致
- 编译失败、正确性失败、性能结果如何变成下一轮 user message
- 哪些 token 参与训练，`loss_mask` 是否符合预期
- 最终 reward 是取最后一轮、最好一轮，还是按旧 drkernel 规则聚合

交付物：多轮样本 artifact、与旧 drkernel prompt/feedback 的人工对照、失败分类统计。

## 第 6 步：训练脚本与资源编排

新增专门的 drkernel run 脚本，复用现有 `scripts/ray/start_cluster.sh`。训练节点只跑 slime/Ray/SGLang/Megatron；KernelGYM reward worker 独立部署，通过 slime `--rm-url` 访问。run 脚本应使用已转换数据和 custom rollout：

```bash
--prompt-data data/drkernel-rl-data-0513/train.parquet
--input-key ground_truth
--label-key ground_truth
--metadata-key extra_info
--rollout-function-path slime_plugins.drkernel.rollout.generate_rollout
```

交付物：`scripts/run-drkernel-*.sh`、SPEC 中记录当前 KernelGYM endpoint、一次小规模 checkpoint 保存验证。

## 第 7 步：评估、对齐和扩展

把 slime 产出的样本、reward、metrics 转成可与旧 drkernel offline eval 对比的 artifacts。先比较 correctness、compile rate、speedup、reward 分布，再扩大 batch、样本数和多机配置。当前新增 `scripts/drkernel/summarize_kernelgym_eval.py`，可从包含 `sample.metadata["kernelgym"]["response"]` 或扁平 KernelGym response 的 JSON/JSONL/PT/PTH/PKL artifact 汇总 `compile_rate`、`correctness_rate`、`fast@1.0_rate`、`fast@1.2_rate`；同时输出 `fast@*_correct_rate` 以对齐旧 DrKernel 只在正确样本上统计 fast@ 的口径。

交付物：对齐报告、失败分类统计、与旧 drkernel 结果的差异表。

## 关键风险

- KernelGYM `/evaluate` 是同步接口，长尾会直接拖慢 rollout。后续可能需要 batch/group reward、限流或异步提交/轮询。
- reward shaping 如果过早引入 speedup，容易和旧 drkernel 语义不一致。第一版只做 correctness reward。
- 多轮反馈已经确定走自定义 rollout；不要再把主线拆到 `--custom-rm-path` 或 `--data-source-path`。
- custom rollout 必须显式维护 `tokens`、`response_length`、`status`、`reward`、`loss_mask`，否则后续默认 train-data conversion 会不可靠。
- 训练容器与 KernelGYM endpoint 的网络可达性必须在容器内验证。

## 2026-05-24/25 Phase 1+2 实验日志（27B / KernelBench Level1 / 多轮 eval）

### Phase 1：v2_3 模板 + env-block ablation — 已结论

**结论：ADOPT v2_3 cleanup 作为默认；DROP "Target environment" env-block 配置（卸下相关 plumbing 或保留为 off-by-default）。**

背景：之前 v1（与 LHB DIR1 byte-identical）出现 fast@1.2 显著低于 LHB DIR1 的 gap（1.4% vs 6.6%）。逐步 ablation 后落地 v2_3：相对于 v1 只增加"轻量 cleanup"（移除模板中泄漏的 jinja 注释、去除 prompt 末尾 cargo-cult markers），其它行为不变。

两个 100×8（800 samples） paired eval：
- v2_3_env_n8：在 first-turn prompt 顶部注入 `Target environment:` 块（GPU="RTX 4090 SM 8.9"，NVCC="12.9 sm_89"）
- v2_3_noenv_n8：无 env 块

| metric | v1 (n=4) | env_n8 | noenv_n8 |
|---|---|---|---|
| T1 compile | 39.0% | 35.0% | 35.4% |
| T3 compile | 45.8% | 51.4% | 50.1% |
| T1 correct | 26.5% | 16.6% | 20.1% |
| T3 correct | 26.8% | 28.0% | 29.4% |
| T1 fast@1.0 | 15.8% | 6.0% | 9.4% |
| T3 fast@1.0 | 11.8% | 9.0% | 11.0% |
| T3 fast@1.2 | 2.2% | 1.4% | 1.6% |

Paired McNemar (n=800, env vs noenv)：
- T1 fast@1.0：env 显著差，p=0.0045
- T3 reward：env 略差，但 n=800 时不显著（n=400 时 p=0.0029，n=800 收窄）
- 其它 metrics 差异不显著

机制层面 cleanup 收益：
- `.shape(0)` 错误：env 58 vs noenv 3
- pybind11/REGISTER_BINDING 残漏：env 68 vs noenv 1
- env 模式会触发 hardware cargo-culting（692/800 responses 提到 `sm_89`），但并不转化为更高 fast@x

fast@1.2 gap 解释：模型在 tvm_ffi 后端从未写出激进优化（cluster gap 而非 cargo-cult gap）；training data limitation。

#### Phase 1 交付物
- 模板：`slime_plugins/drkernel/prompt_templates/backends/tvm_ffi_module_v2_3.jinja`（默认）
- 渲染修复：`slime_plugins/drkernel/rollout.py` 中 `_load_fragment` 全部经 `_render_template`（修复 jinja `{# #}` 注释泄漏）
- 启动 sanity check：`scripts/debug/render_prompt_check.py`（在 submit_ray_job 前 render first-turn prompt，遇 `{{`/`{%`/`{#` 残漏 或 expected GPU words 缺失则 exit 非 0）
- 启动脚本：`scripts/debug/debug.27b.sh` 已硬编码 GPU="NVIDIA GeForce RTX 4090 (SM 8.9, Ada Lovelace)" + NVCC="CUDA 12.9 (nvcc, targeting sm_89)"，`DRKERNEL_NO_TARGET_ENV=1` 关掉 env 块
- 数据：v2_3_env_n8 / v2_3_noenv_n8 的 eval_0.pt 在 `checkpoints/Qwen3.6-27B/20260524_*_v2_3_*_n8/dumps/rollout_data/`
- Codex review 报告：`/tmp/codex_phase1_complete_output.log`

#### Phase 1 后续动作（推荐）
1. 把 env-block plumbing 从 `debug.27b.sh` + `rollout.py` + 模板 里彻底卸掉（或者改为默认 off），减少 cognitive load
2. v2_3 模板 normalize 命名（删 `_v2_3` 后缀），让 default 路径直接走 cleanup 版本
3. 把 `render_prompt_check.py` 集成进 `scripts/run-*.sh` 而非只 debug 路径

### Phase 2：W8A8-INT rollout 加速 — 已放弃（abandoned, blocked）

**结论：ABANDON。量化本身成功（28GB compressed-tensors INT8 ckpt），但 sglang 0.5.10.post1 加载 `Qwen3_5ForCausalLM` 失败，需要 sglang-side 改动，超出 ROI。**

#### 详细路径

1. ✅ Calibration 集：从 v2_3_noenv n=4 dump 提取 1200 个 T0/T1/T2 prompt 写入 `/tmp/calibration_drkernel_v2_3_noenv_n4.jsonl`（512 用作 calibration sample）
2. ✅ 量化脚本：`/tmp/quantize_qwen36_w8a8.py`
    - 路径：llmcompressor（git main）+ compressed-tensors（git main）+ transformers 5.3.0
    - `GPTQModifier(targets="Linear", scheme="W8A8", ignore=["lm_head"])`，single-modifier（SmoothQuant 无 Qwen3_5 mapping，drop）
    - 跑了 ~3h 在 .22 八卡 A800
3. ✅ Checkpoint 写入：`/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B-W8A8-ct/`（28GB single safetensors，quant_method=compressed-tensors, format=int-quantized, token-dynamic activations + per-channel static weights）
4. ❌ SGLang load smoke：两连失败
    - **Blocker 1**：`ValueError: Qwen3_5ForCausalLM has no SGlang implementation`。原因：llmcompressor 在量化时将 architecture 从 `Qwen3_5ForConditionalGeneration`（multimodal，sglang 已注册）改成 `Qwen3_5ForCausalLM`（dense LM only，sglang 类存在但**不在 EntryClass**）
    - 临时 patch：把 `Qwen3_5ForCausalLM` 加入 `/sgl-workspace/sglang/python/sglang/srt/models/qwen3_5.py` 的 `EntryClass`（已 revert）
    - **Blocker 2**：`AttributeError: 'Qwen3_5TextConfig' object has no attribute 'num_experts'`。原因：`Qwen3_5ForCausalLM.get_model_config_for_expert_location` 实现里硬编码访问 `config.num_experts`，假设 MoE config。dense ckpt 没有该字段。要修需要修改 sglang 类层多个方法，影响面大。
5. ✅ Patch 已回滚：`/sgl-workspace/sglang/python/sglang/srt/models/qwen3_5.py` 已恢复为 EntryClass = [Qwen3_5MoeForConditionalGeneration, Qwen3_5ForConditionalGeneration]

#### 为什么放弃

- W8A8 预期收益：rollout 阶段 ~1.5–2× 加速；Phase 1 一次 100×8 eval ≈ 70 min，最多能省 ~35 min/run
- 解锁成本：要么 (a) 修改 sglang 类层（多个 method 都 hardcode MoE 假设；维护 fork），要么 (b) 重新量化保留 ForConditionalGeneration 架构（保留 vision tower 权重，量化只覆盖 LM linear 层；HF AutoModelForConditionalGeneration 路径 + GPTQ 配置目标只 hit `model.language_model.*` 层；可行但又是 3-6h 实验且不保证 sglang 加载成功）
- Phase 1 已经提供了清晰可 ship 的结论（DROP env block），无 wall-time 紧急性

#### 留下来的 artifacts（保留可复用）

- `/tmp/quantize_qwen36_w8a8.py` — 量化脚本
- `/tmp/build_w8a8_calibration.py` — calibration dataset 抽取
- `/tmp/calibration_drkernel_v2_3_noenv_n4.jsonl` — 1200 个 prompts
- `/tmp/Qwen3.6-27B-W8A8-ct/` 和 `/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B-W8A8-ct/` — 28GB W8A8 ckpt
- `/tmp/w8a8-venv/` on .22 — llmcompressor + compressed-tensors 环境

#### 关键细节 — 现有 W8A8 ckpt 的权重命名（codex review 后核实）

`/nfs/.../Qwen3.6-27B-W8A8-ct/model.safetensors` 的权重 key 仍然带 `model.language_model.*` 前缀（共 1347 keys），符合多模态 architecture 的语言模型部分命名。但 `model.visual.*` 完全缺失（llmcompressor 在 `AutoModelForCausalLM.from_pretrained` 时丢弃了 vision tower）。

#### Path D（codex 推荐路径）— 经核实仍受阻

Codex 给出的"低成本 retry"：拿原 BF16 full multimodal config.json + 把 W8A8 的 quantization_config 移植进去 + 保留 `architectures=["Qwen3_5ForConditionalGeneration"]` + 指向同一份 W8A8 safetensors。验证后：

- ✅ 权重 key 名匹配（W8A8 weights 已是 `model.language_model.*` 前缀）
- ❌ sglang 的 `Qwen3VLForConditionalGeneration.__init__` 无条件实例化 `self.visual = Qwen3VLMoeVisionModel(...)`（`/sgl-workspace/sglang/python/sglang/srt/models/qwen3_vl.py:1085`），不受 `language_only` 影响
- ❌ sglang 的 `--language-only` server arg 只影响 mooncake transfer engine 初始化（`model_runner.py:1054`），不会跳过 vision 权重加载
- ❌ 加载时 sglang weight loader 会找 `model.visual.*` keys → 缺失 → 加载失败

Path D 要可行还需 weight surgery：把原 BF16 `model.visual.*` 权重合并进 W8A8 safetensors，并且 ensure sglang quant_config 不会试图量化 vision tower。预计 0.5-1 day。

#### 重启 Phase 2 的建议路径（按推荐优先级）

如果未来要 retry，按 ROI 排序：
- **路径 D'（修订版，最便宜）**：先拿 BF16 vision tower 的权重（用 safetensors 工具从 `/nfs/.../Qwen3.6-27B/` 抽取 `model.visual.*` 全部 keys），再 concat 到 W8A8 safetensors，写一个 overlay config（原 full config + W8A8 `quantization_config` 块，但 `ignore` 列表必须包含 vision tower 各 module 路径以免 sglang quant 加载器试图把它们当 INT8 处理）。预计 4-6h，主要是 weight concat + ignore 配置 + smoke debug。
- **路径 A（最干净但贵）**：重做量化，用 `Qwen3_5ForConditionalGeneration.from_pretrained` 加载原 multimodal，GPTQModifier `targets` 限制为 `["re:.*model\\.language_model\\..*"]`，vision tower 保留 BF16。save 时保留原 architecture。完整重跑 3-5h calibration。
- **路径 B（不推荐）**：fork sglang，给 `Qwen3_5ForCausalLM` 写 dense-aware `get_model_config_for_expert_location` 并加入 EntryClass。维护成本高，碰升级回归风险大。
- **路径 C**：等 sglang 上游支持 dense Qwen3_5 entry。

#### Codex 警告（来自 Phase 2 review）

- **FP8 KV cache** 在 Qwen3.5 上默认 scale=1.0 会伤 reasoning-heavy 精度（sglang docs warn），不要轻易加 `--kv-cache-dtype fp8` 到 sweep
- llmcompressor 没有"保留原 multimodal architecture 但只 save AutoModelForCausalLM 抽取"的 save 选项；要么 retry path A 走多模态加载，要么 path D' 手动 weight surgery

#### Phase 2 的副产物 — 推荐 sglang flag sweep（不依赖 W8A8）

Codex 给出 top-3 候选（按预期 wall-time win × 命中概率）：
1. Speculative decoding NEXTN：`--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`（如果模型有 NEXTN draft，前文已看到 sglang 注册了 `qwen3_5_mtp.py`，可能可用）
2. `--mamba-scheduler-strategy extra_buffer --page-size 64`（Qwen3.5 hybrid attn 含 mamba layers）
3. 显存压榨：`--max-running-requests 96/128 + --mem-fraction-static 0.92/0.94 + --schedule-policy lpm`（如果 n=8 同 prompt 共享前缀）

每个建议都是 1 次 100×8 eval 即可验证（~70 min），未跑过。

### Phase 1+2 总览交付物

- 模板 + 渲染修复（Phase 1）：见上
- W8A8 残留产物（Phase 2）：见上

### Phase 3 后续 — 9B 跨规模验证 + v2_4 模板

#### 9B v2_3 noenv 100×8 — 与 27B 结论方向相反，**v2_3 全线退化**

| 指标 | T3 9B v1 (5/22, n=8) | T3 9B v2_3 noenv (今, n=8) | Δ |
|---|---|---|---|
| compile | 19.5% | 11.4% | **-8.1pp** |
| correct | 5.5% | 2.0% | **-3.5pp** |
| fast@1.0 | 4.6% | 1.1% | **-3.5pp** |

同 profile、同 backend id、同 KG server、同模型、同 noenv，只换 first-turn 模板文件。CTX 65k → 32k 不构成因素（两边都 0 truncation）。

**退化机制**（通过错误分类 + 响应模式分析验证）：
- v2_3 的 C++ minimal example 比 v1 的削去了 6 个 ICHECK 行 + 把 size 参数改成 1D-only `input.shape()[0]`（v1 是 rank-agnostic `input.numel()`）
- 9B 倾向 verbatim 复制 example：拿到 matmul/conv 等多维 op 时，复制出来的 `input.shape()[0]` 只是 M 维 → 传错 size → 编译/运行失败
- error category 数据支撑：v2_3 `icheck_use_issue` 比 v1 多 +7.5pp（9B 缺乏可复制的 ICHECK pattern，自己乱写）；v1 `no_matching_function` 比 v2_3 多 +19pp（v1 verbose 30 条指导让 9B 调用错误函数）—— 不同失败模式，但 v2_3 的更致命

**含义**：Phase 1 "ADOPT v2_3 default" **不能直接推广到 9B**。小模型需要 redundancy（verbose example + 多个 ICHECK pattern 可复制），大模型需要 conciseness（减噪音）。

#### v2_4 模板（未测试）

基于 v2_3 的纯精简（**reword/merge/trim only, 零新增 information**）：净 -728 chars / -4 行。改了 11 处：
- CUDA_KERNELS 段 4→3 bullets（merge "pure CUDA + don't call X" + "real CUDA C++ code"，删 "for APPLY_BINDINGS" 隐含词等）
- APPLY_BINDINGS 段 7→6 bullets（删 "binding's job ..." duplicate of L22，删 rationale "This keeps shape derivation in Python ..."，inline stream-getting call，shorten ICHECK bullet）
- MODEL_NEW 段 5→4 bullets（merge "import ext + don't bypass + allocate output + pass into wrapper" 成单条）
- example 块**完全不动**（C++ + Python 两段 example 与 v2_3 完全一致）

文件：`slime_plugins/drkernel/prompt_templates/backends/tvm_ffi_module_v2_4.jinja`，未 wire 到 yaml profile，未跑 eval。

**注意**：v2_4 没有针对 9B 退化机制做特别修复（没有恢复 v1 example 的 7 ICHECK + numel）。所以预期 9B 在 v2_4 上**仍会退化**（与 v2_3 相近，因为 example 部分没变）。要修 9B 需要单独的 v2_5 / per-model template 选择。

#### 推荐下一步（按 ROI）

1. **v2_4 上 27B 100×8 验证** — 确认精简没引入 regression（context 短 ~728 chars，27B 上预期 fast@1.x 持平甚至略升）。1×70min
2. **决定 9B 策略**：
   - (a) `prompts_v1.yaml` 的 `drkernel_v1_tvm_ffi` profile 改回 `tvm_ffi_module.jinja` (v1)，把 v2_3/v2_4 留给 27B 显式 opt-in
   - (b) `rollout.py` 引入 model-size-aware profile 选择（小模型走 v1，大模型走 v2_4）。维护成本高
   - (c) v2_5：在 v2_4 基础上把 C++ example 恢复成 v1 的 7-ICHECK + numel 形式，看能否两者都满足
3. **不再考虑 W8A8** — 见 Phase 2 abandon decision

### 评估精度参考

所有 100×8 (n_samples_per_eval_prompt=8) 运行的精度汇总见 `handoffs/in_progress/HANDOFF_DRKERNEL_EVAL_ACCURACY.md`，含 9B + 27B 全部 8 个有效运行的 per-turn compile / correct / fast@1.0 / fast@1.2 (in_all) 表格 + 模板演化 + 元数据说明。
