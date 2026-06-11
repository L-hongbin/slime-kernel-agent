# DrKernel Entropy Collapse 排查手记

## 结论先行

固定 `--entropy-coef 0.00` 排查。目标不是通过熵正则把曲线拉回去，而是闭合
这条因果链：

```text
当前分支相对 main 的实现/配置差异
  -> train/entropy_loss 下降
  -> 同组 rollout 变同质或 reward pattern 变坏
  -> dynamic filter 丢弃更多 zero-std / low-variance group
  -> 为凑满 rollout_batch_size 生成更多候选
  -> rollout time 上升
```

| 已知事实 | 含义 |
| :--- | :--- |
| main 分支脚本 `examples/kernel_agent/run_qwen3.6_27B_async.sh` 未观察到 entropy collapse | 第一优先级是 current vs main 对照，不是泛化调参 |
| 曲线是 `train/entropy_loss` | 必须按训练侧代码口径解释，不是 rollout entropy，也不是 `correct_entropy` |
| rollout time 同时上升 | 优先验证是否由 filter drop / oversampling 增多造成 |
| sampling 参数需要看代码 | 当前路径传 `temperature/top_p/top_k`，没有 `min_p` 接线 |

## Main 对照

已检查 main 分支脚本：`examples/kernel_agent/run_qwen3.6_27B_async.sh`。它是完整
async 多轮 kernel_agent 配置，不是单轮 smoke。

| 维度 | main 脚本配置 |
| :--- | :--- |
| 启动入口 | `train_async.py` |
| 资源形态 | 4 个 actor node；rollout GPU 与 actor GPU 分离 |
| 数据 | `Drkernel-rl-thinking-PV4`，`prompt_v4/drkernel_rl_thinking.parquet` |
| rollout batch | `rollout-batch-size=16`，`n-samples-per-prompt=16`，`global-batch-size=96` |
| 生成长度/上下文 | `max_response_len=8192`，`rollout_max_context_len=32768` |
| 采样 | `rollout-temperature=1`，默认 `top_p=1.0`，`top_k=-1` |
| 并行 | TP4 / PP2 / CP4 |
| loss 归约 | `--calculate-per-token-loss` |
| RL | `--advantage-estimator trloo`，`--multi-turn-gamma 1.0` |
| entropy | `--entropy-coef 0.00` |
| PPO clip | `--eps-clip 0.2 --eps-clip-high 0.28` |
| TIS | `--use-tis` 在脚本中注释，未启用 |
| 多轮 | `--use-multi-turn --max-turns 3 --padding-turns --filter-by-last-turn` |
| Sequence MIS | `--rollout-data-postprocess-path examples.kernel_agent.kernel_filter.sequence_mis`，`aggregation=turns_geometric`，`lower=0.999`，`upper=1.001`，`token_veto_threshold=1e-4` |
| Coverage RS | `--use-coverage-rs --coverage-rs-key time_coverage --coverage-rs-threshold 0.3 --coverage-rs-factor 0.1` |

| 自定义 hook | main 路径 |
| :--- | :--- |
| generate | `examples.kernel_agent.generate_with_cuda_agent.generate` |
| reward | `examples.kernel_agent.generate_with_cuda_agent.reward_func` |
| reward post-process | `examples.kernel_agent.kernel_reward.reward_post_process_by_group` |
| dynamic filter | `examples.kernel_agent.kernel_filter.filter_cuda_kernel_group` |
| rollout data postprocess | `examples.kernel_agent.kernel_filter.sequence_mis` |

| main 多轮 generate / reward 行为 | 代码含义 |
| :--- | :--- |
| 每 turn 产一个 `Sample` | 每轮 response 都可成为训练样本 |
| `loss_mask=[1] * response_len` | 训练 assistant response token |
| pad turn `loss_mask=[0]` 且 `remove_sample=True` | padding 不进 loss / reward postprocess |
| `group_id = base_sample.index` | 同轨迹 turn 共享 group |
| `trloo` 读取 `metadata["multi_turn_reward"]` | reward postprocess 用折叠后的多轮 return |
| `multi_turn_reward = r_t + gamma * future_return` | `gamma=1.0` 时未来 turn 成功会回传到前面 turn |
| `reward_post_process_by_group` 按 `(group_index, turn_idx)` 分组 | 多轮 RLOO 不把不同 turn 混在同一组 |
| `finalize_mode=positive` | 出现正 reward 后，后续非正 turn 可被 `remove_sample` |
| `coverage_rs` | 可能移除低 coverage 的正确样本 |
| sequence MIS | off-policy sequence 可被置零 loss_mask |

因此，“main 不 collapse”不能直接证明当前单轮配置应该不 collapse。它说明当前分支必须
优先对照这些差异。

## 当前差异

| 维度 | main | 当前 27B debug 单轮 | 风险 |
| :--- | :--- | :--- | :--- |
| 训练形态 | async 多轮，`max_turns=3` | 单轮 smoke | 多轮反馈 / gamma / MIS 保护缺失 |
| advantage | `trloo` | `rloo` | 稀有成功样本的梯度形态不同 |
| reward | speedup / coverage / penalty shaping | phase-1 binary 0/1 | binary 更容易把概率推向少数成功模板 |
| reward postprocess | custom `reward_post_process_by_group` | core `group_normalize_rewards` / rloo | 多轮 `(prompt, turn)` 分组和 `multi_turn_reward` 不存在 |
| dynamic filter | `filter_cuda_kernel_group` | slime 内置 `check_reward_nonzero_std` | 当前缺少 remove_sample 排除、小组保护、阈值差异 |
| sequence MIS | 开启 `turns_geometric` | 未实现 / 未开启 | off-policy 或异常序列无法被 sequence 级置零 |
| TIS | 注释，未启用 | `--use-tis` | current-only 变量，需优先短消融 |
| CP | CP4 | CP1 | 指标和吞吐不可直接对比 |
| rollout 实现 | `examples.kernel_agent` cuda_agent | `slime_plugins.drkernel` | prompt/render/reward 服务路径不同 |
| filter-by-last-turn / padding | 开启 | 单轮无 | filter 语义不同 |

## `train/entropy_loss` 口径

| 步骤 | 代码路径 / 语义 |
| :--- | :--- |
| 入口 | `policy_loss_function` 调 `get_log_probs_and_entropy(..., with_entropy=True)` |
| temperature scaling | `get_log_probs_and_entropy` 中先做 `logits = logits / args.rollout_temperature`；当前 temperature=1，数值等价于裸 logits |
| entropy 公式 | `_VocabParallelEntropy`: `H = logsumexp(logits) - sum_v softmax(logits)_v * logits_v` |
| token 范围 | `_extract_per_sample` 只抽 response token 对应位置 |
| mask | `sum_of_sample_mean(entropy)` 会乘 `loss_mask` |
| 当前归约 | 当前脚本开 `--calculate-per-token-loss`，最终 train metric 除以全局有效 token 数 |
| objective | `loss = pg_loss - args.entropy_coef * entropy_loss`；`entropy_coef=0` 时指标仍计算但不进目标 |
| 注意 | `correct_entropy` 更接近 correct 样本的 `-log_probs` / NLL，不是 full-vocab policy entropy |

需要单独确认：current 使用 `rollout_mask_sums`，main 对应路径使用 `group_mask_sums`。
若两者在单轮 group 场景不等价，必须先解释指标和 loss 归一化口径差异。

## SGLang Sampling 口径

| 项 | 代码结论 |
| :--- | :--- |
| 参数来源 | `GenerateState(args).sampling_params` |
| 已传参数 | `temperature`、`top_p`、`top_k`、`max_new_tokens`、`stop`、`stop_token_ids`、`skip_special_tokens`、`no_stop_trim=True` |
| current 训练路径 | DrKernel rollout 调 `state.submit_generate_tasks(samples)`，走默认 `generate_and_rm_group -> generate`，POST payload 带 `sampling_params` 和 `return_logprob=True` |
| main custom generate | 每 turn 也把传入的 `sampling_params` 直接放入 SGLang `/generate` payload |
| `min_p` | 当前没有 `--rollout-min-p` / `min_p` 参数，也没有写入 `sampling_params` |
| 当前实际采样 | 只显式设置 `--rollout-temperature 1`；默认 `top_p=1.0`、`top_k=-1`，基本不截断 |

## 排查优先级

| 优先级 | 问题 | 必查证据 | 通过/失败解释 |
| :--- | :--- | :--- | :--- |
| P0 | current vs main 差异是否解释 collapse | `evidence_entropy_collapse_main_diff.txt`：argv、reward/filter/advantage/TIS/MIS/multi-turn 对照 | 若差异集中在 TIS/filter/MIS/reward，则后续消融围绕这些项 |
| P1 | `train/entropy_loss` 是否代表真实 collapse | rollout dump、unique response/kernel、pass@k、eval、repetition、length | 若多样性和 eval 正常，先判 policy sharpening |
| P2 | rollout time 上升是否由 filter/oversampling 导致 | submitted / accepted / filtered groups、drop reason、reward pattern、drained tokens、wall time | 若 filter drop 随 entropy 下降上升，链条成立 |
| P3 | reward/filter/advantage 是否放大少数模式 | reward pattern histogram、advantage stats、positive token mass、top successful templates、`pg_clipfrac/ppo_kl/tis*` | 若 `1/16`、`2/16` 组主导，RLOO + binary 是强嫌疑 |
| P4 | loss mask / response surface 是否偏 | token+mask 可视化、boilerplate token mass、remove_sample mask | 若固定格式 token 主导，entropy 下降可能是目标表面问题 |

## Rollout Time 链条指标

| 指标 | 用途 |
| :--- | :--- |
| `submitted_groups` | 实际生成了多少候选组 |
| `accepted_groups` | 进入训练 batch 的组数 |
| `filtered_groups` | 被 dynamic filter 丢弃的组数 |
| `filter_drop_by_reason` | 区分 zero-std、low-variance、小组等原因 |
| `reward_pattern_histogram` | 统计 `0/16, 1/16, ..., 16/16` |
| `drained_inflight_groups/tokens` | 估计 oversampling 浪费 |
| `rollout_time` | 验证 wall time 是否由过滤增多解释 |
| `train/entropy_loss` | 判断是否领先 filter drop 变化 |
| `rollout/kernel/correctness` | 防止把质量变化误判成采样效率问题 |

只有当 `entropy_loss` 下降领先 group 同质化 / filter drop 上升，并且 filter drop
上升解释了 rollout time 增长，才能说这条链成立。

## 证据产物

| 文件 | 内容 |
| :--- | :--- |
| `handoffs/in_progress/evidence_entropy_collapse_main_diff.txt` | main 配置摘要、current argv、entropy 公式和 reducer 口径、reward/filter/advantage/TIS/MIS/multi-turn 差异、sampling params 路径 |
| `handoffs/in_progress/evidence_entropy_collapse_curves.txt` | 每 rollout 的 `entropy_loss`、`pg_clipfrac`、`ppo_kl`、TIS、rollout time、filter 计数、reward pattern、correctness、repetition、response length |
| `handoffs/in_progress/evidence_entropy_collapse_rollout_review.txt` | early/mid/late 同 prompt group；response；extracted/normalized kernel；reward；失败类别；filter 状态；unique response/kernel 数 |

建议 `curves` 表头：

| 列 | 说明 |
| :--- | :--- |
| `rollout_id` | rollout 编号 |
| `train/entropy_loss` | 训练侧 entropy 指标 |
| `train/pg_clipfrac`, `train/ppo_kl` | PPO 更新强度 |
| `train/tis`, `train/tis_abs`, `train/tis_clipfrac` | TIS/off-policy 影响 |
| `rollout_time` | rollout wall time |
| `submitted_groups`, `accepted_groups`, `filtered_groups` | filter/oversampling 计数 |
| `filter_drop_by_reason` | filter 原因分布 |
| `reward_pattern_histogram` | group reward pattern |
| `rollout/kernel/correctness` | reward 质量信号 |
| `rollout/repetition_frac` | 文本重复度 |
| `response_len_mean/p50/p95` | 长度分布 |

## 短消融矩阵

固定 `--entropy-coef 0.00`，按顺序做：

| 顺序 | 消融 | 目的 | 判据 |
| :--- | :--- | :--- | :--- |
| 1 | baseline + dump + filter 统计 | 建立事实 | collapse 是否真实；rollout time 是否由 filter drop 解释 |
| 2 | main-like 对照 | 对齐 main 无 collapse 条件 | TIS off、sequence MIS、RLOO/trloo、dynamic filter、reward shaping、多轮/单轮逐项区分 |
| 3 | 关闭 dynamic filter | 验证 filter 是否在因果链上 | entropy 下降变慢或 rollout time 恢复 |
| 4 | 记录 pre-filter reward pattern | 判断 policy 是否已导致组内同质 | 候选组大量变成 `0/16` 或 `16/16` |
| 5 | TIS off | current 启用 TIS，而 main 未启用 | `tis_clipfrac`、entropy、rollout time 明显变化则 TIS 是嫌疑 |
| 6 | advantage / reward 对照 | 检查 binary + RLOO 是否放大稀有成功 | 稀有成功组 positive advantage mass 是否下降 |
| 7 | `eps_clip_high 0.28 -> 0.2` | 检查正 advantage token 是否被过快推高 | `pg_clipfrac/ppo_kl/entropy` 变化 |

学习率是后置全局 update-size 控制，不作为第一解释。

## 关闭标准

| 必须回答 | 关闭要求 |
| :--- | :--- |
| current 与 main 的关键差异是什么 | 有明确配置/代码对照 |
| `train/entropy_loss` 口径是什么 | 已确认公式、temperature scaling、mask/reducer |
| rollout 文本是否真的坍缩 | 有人工 review 和 unique response/kernel 证据 |
| rollout time 上升原因 | 能解释 filter drop / oversampling 是否负责 |
| reward pattern 是否恶化 | 有 `0/16..16/16` 分布趋势 |
| RLOO / TIS / clip-high 是否放大少数模式 | 有 advantage/off-policy/clip 指标 |
| 根因区分 | 至少一个短消融区分两个候选根因 |

