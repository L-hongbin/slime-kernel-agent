# DeepSeek-V4 fresh RL 全零 reward group 审计（2026-08-16）

## 1. Handoff 状态

- **全零 group / dynamic filter 的原因分析已完成。** 大量 drop 不是计数器或标准差实现错误；它来自大量 prompt 的 16 个候选在 pre-overlong task reward 上全部为 0。
- **Slime 与 KernelGym 现已支持任务级 dtype/precision。** Slime 解析 `fp32`、`fp16`、`bf16` 并随请求发送；KernelGym 将该字段传到 CUDA-Agent 和 TVM-FFI 的静态 precheck。
- **修复后的 formal 训练已完成在线影响量化。** 在两条 fresh lineage 共同可比的 step 0--26，低方差 drop 从 576 降到 404（`-29.9%`），工作口径下的 drop 概率从 57.1% 降到 48.3%。真实请求中的 FP16/BF16 precision 与静态检查均正确，不再出现旧的 `required FP32` 门禁误判。
- **收益不是“全部低精度题恢复”。** 同一窗口 low-precision drop 从 171 降到 110，但 FP32 drop 也从 405 降到 294；剩余失败已进入真实 compile/correctness 路径。由于 refill 提前停止后两跑访问的 prompt 不同，不能把 172 个总降幅逐个归因给 dtype。
- **修复后 entropy 确实下降更快，但 dtype 不直接改 logits。** 共同 step 0--25 的 rollout/train entropy OLS（Ordinary Least Squares，普通最小二乘） 斜率约为修复前的 2.3 倍；而 grad norm、PPO KL、train-rollout MAE 和平均长度基本不变。当前最符合证据的是 dtype 通过 reward/filter 改写 accepted population 和 TRLOO 梯度方向，再由无 entropy/reference 恢复项的目标放大；尚无冻结 replay 能给出 dtype 对降熵加速的独立因果效应。

## 2. 结论

本轮现象不是“模型问题”与“reward 环境问题”的二选一。最符合证据的因果链是：

> 原始 V4 较弱的 first-shot 能力 × `max_turns=1` × 失败 reward 全部折叠为 0 × 当时默认 FP32 的隐藏 precision 合同。

各因素的边界如下：

1. **模型候选的真实失败是主触发源。** 日志中有未定义标识符、缺头文件、CUDA/cuDNN 参数错误、host function 被当 kernel launch、非法显存访问、不存在的 PyTorch API，以及编译通过但结果明显错误。
2. **单轮配置放大 first-shot 弱点。** formal 配置由 `scripts/dsv4/_dsv4_task_args.sh:163` 设为 `max_turns=1`，没有利用环境反馈修复候选。
3. **reward 与 filter 放大了可见现象和训练选择偏差。** precheck、compile 和普通失败当前都得 0；一个 group 内即使失败阶段不同，filter 仍只看到 `[0, ..., 0]`。关闭 filter 也不会让完全同分 group 获得 TRLOO 相对 advantage。
4. **旧 precision 路径造成了一个可信的 false-zero 子集。** 低精度扩增题此前没有把真实 dtype 送给 KernelGym，合法 FP16/BF16 实现可能被默认 FP32 precheck 判为 `precision_downgrade`。该传递缺口现已修复并做过端到端 A/B 验证。
5. **没有证据支持 KernelGym 整体失效。** 新训练所有观测到的 HTTP POST 均返回 200，task terminal timeout 约 0.27%；同一窗口持续有成功任务，错误与候选代码吻合。但历史 drop 中 reference/liveness 为 pending/None 的约 30% 子集没有逐条重验，仍是独立未闭环项。

## 3. 指标到底在数什么

`examples/kernel_agent/kernel_filter.py:99-143` 对未被移除的样本计算 **population std**：

\[
\operatorname{std}(r_1,\ldots,r_n),\quad \texttt{unbiased=False}
\]

当标准差小于 `0.001` 时丢弃整个 prompt group。过滤使用 `metadata["task_reward"]`（若存在），即 overlong penalty 之前的 task reward；因此日志里某些最终 reward 为负，并不反驳该组的 `filter_reward` 全为 0。

`rollout/dynamic_filter/drop_reward_std_lt_0.001` 是**单个 rollout step 丢弃的 prompt-group 数量**：

- 它不是标准差值；
- 它不是样本条数；
- 它不是跨 step 累计值；
- 本轮每个 group 有 16 个 completion（launcher `271-278` 行）。

截至完成 step 0–18，低方差 drop 计数为：

```text
28, 34, 17, 22, 14, 34, 25, 16, 22, 36,
15, 29, 12, 23, 14, 23, 20, 12, 18
```

合计 414 个全零 group；同一 19-step 窗口接收 `19 × 16 = 304` 个 group。在只比较“低方差 drop”和“accepted”这两个出口的工作口径下，drop 占：

\[
q=\frac{414}{414+304}=57.7\%
\]

若以该经验概率近似 refill 流程，收满 16 个有效 group 前的期望 drop 数为：

\[
16\frac{q}{1-q}=21.8
\]

正好等于实测 `414/19=21.8`。所以每步看到 16、22、30 左右是 refill 几何的自然结果，不是计数 bug。这个口径未包含 small-group、取消等其他候选出口，不应解释成完整候选总体的无偏失败率。

另一个反事实能排除“只是独立采样运气差”：若所有 prompt 的单样本成功率都同为 27%，16 次都失败的概率只有

\[
(1-0.27)^{16}\approx0.65\%.
\]

观察到的 57.7% 说明成功率高度依赖 prompt：一部分题容易成功，另一部分题存在近乎系统性的能力或合同失败。

## 4. 真实失败与模型能力证据

最近失败样本同时覆盖语法、编译、运行时和 correctness 阶段。错误输出也不是普遍卡在 `1e-4` 容差边缘：审计窗口内错误候选的 `max_difference` 中位数约 8.76，很多超过 1。

该 rollout 使用原始 `DeepSeek-V4-Flash-DSpark`，不是 0731 endpoint。历史固定 KernelBench-L1 评估为：

| 模型 | 首轮 compile | 首轮 correct | 三轮 Best |
|---|---:|---:|---:|
| 原始 V4 | 39.00% | 27.00% | 65.12% |
| V4-0731 | 53.62% | 39.88% | 85.12% |

原始证据见 `handoffs/in_progress/handoff_kernelbench_l1_model_accuracy.md` 和 `handoffs/deepseek-v4/deepseek_v4_flash_3turn_eval_20260801.md`。首轮与三轮的差距说明环境反馈本来能修复大量错误，而 formal 的单轮设置关闭了这条路径。

## 5. Reward/filter 如何放大问题

`examples/kernel_agent/config.py` 当前把 `penalty_score`、`compilation_fail_penalty`、`precheck_fail_penalty` 都设为 0；`examples/kernel_agent/kernel_reward.py:74-88,122-155` 将相应失败映射为这些值。

因此一个 16-sample group 即使包含不同进度的失败——例如语法错、编译错、runtime 崩溃和数值错误——也会坍缩为同一个 reward 向量。结果有两层：

1. dynamic filter 丢弃该 group；
2. 即使不丢弃，完全同分组的 TRLOO advantage 也全部为 0。

所以 hard prompt 得不到“离成功更近”的相对信号，训练会偏向至少偶尔能成功的 prompt。要改变这一点需要 graded failure reward、curriculum 或多轮反馈等单独的 policy 设计；dtype 传递本身不会解决它。

## 6. Dtype 缺口、当前实现与验证

### 6.1 修复前证据的正确读法

一个与近期 drop 重叠的 66 条 compilation-class failure 审计中：

- 30 条为 `required FP32 but code uses FP16`；
- 4 条为 framework compute；
- 20 条为未定义/缺失标识符；
- 3 条为 API/type mismatch；
- 其余为少量其他错误。

`30/66` 只说明 FP32 门禁在该编译失败窗口中很突出，**不是** 414 个 drop 中的占比，也不是修复后可恢复比例。对应 prompt 只要求 preserve correctness，没有显式声明“FP32 输入不得使用任何内部 FP16”，所以这些拒绝在旧链路下是合同误配风险，不能直接算作模型数值错误。

数据分布也提供了方向性证据：414 个全零 group 中 dtype 变体占 24.6%，训练全集中占 16.4%，约 1.5 倍富集；CSP-DAG 也约 1.5 倍。414 个 group 来自不同根问题，并非 augmentation siblings 重复计数。

### 6.2 Slime 当前行为

`examples/kernel_agent/generate_with_cuda_agent.py:401-469,535-572` 的解析顺序是：

1. `metadata.augmentation.dtype_after`；
2. `metadata.precision`；
3. 只检查 reference `get_inputs()` 的 AST，恢复 dtype→layout 串行扩增继承的低精度；
4. 缺失、未知或混合/歧义 dtype fail closed 到 `fp32`。

`examples/kernel_agent/kernel_response.py:232-242` 将规范化后的 `precision` 加入 KernelGym 请求。正式 39,636 行数据的实际解析分布是：

| precision | 数量 |
|---|---:|
| FP16 | 4,215 |
| BF16 | 3,995 |
| FP32 | 31,426 |

其中 6,488 个显式 dtype augmentation（3,353 FP16、3,135 BF16）的 metadata 与 reference 一致；另从 dtype→layout 子代恢复 1,720 个低精度样本（FP16/BF16 各 860）。

本 handoff 写作时，Slime 侧这些实现和测试仍是工作区中的未提交修改；现已由
`3c93077`（`fix(kernel-agent): propagate task precision to precheck`）进入仓库历史。

### 6.3 KernelGym 当前行为

KernelGym 仓库 `/nfs/FM/chenshuailin/projects/kernel_agents/KernelGYM-reward-only` 的提交 `737542bb8527`（`feat: pass client precision to kernel checks`）已实现：

- API 接收并规范化 precision，缺省为 `fp32`，未知外部值拒绝；
- 字段穿过 `EvaluationTask`、paired kernel task、split compile、toolkit 和 pipeline；
- CUDA-Agent 与 TVM-FFI backend 都把 precision 传给静态 precheck；
- request/result-cache hash 区分不同 precision，避免 FP32/FP16 结果碰撞。

### 6.4 验证结果

本地定向测试：

```text
Slime:     python -m pytest -q tests/test_kernel_reference_cache.py
           15 passed
KernelGym: python -m pytest -q tests/test_precision_passthrough.py
           14 passed
```

更宽的 KernelGym 组合测试此前得到 29 passed、1 failed；唯一失败是既有的 `test_cuda_agent_object_reuse_skips_module_bound_sources` 对象复用断言，与 precision 传递无关，未在本任务中顺手修改。

线上 `20211` 做过同代码 A/B：

| backend | `precision=fp32` | `precision=fp16` |
|---|---|---|
| CUDA-Agent | `codex_precision_ab_20260816_210915_fp32`：被 precision precheck 拒绝 | `codex_precision_ab_20260816_210915_fp16`：completed、compiled |
| TVM-FFI | `codex_precision_tvm_ab_20260816_211255_fp32`：被 precision precheck 拒绝 | `codex_precision_tvm_ab_20260816_211255_fp16`：completed、compiled |

这证明了 API → worker/task → backend precheck 的端到端传递。它是 compile-only 合同测试，不是 414 个历史 group 的 correctness 重放。

### 6.5 修复后 formal 训练复盘（2026-08-17）

对比对象均从同一 base checkpoint、同一 v4 release parquet 和 fresh state 启动：

- 修复前：`formal_dsv4r21_fp4_pp1cp2_12k_prompt_tvm_v4_release_20260816_fresh`；
- 修复后：`formal_dsv4r21_fp4_pp1cp2_12k_prompt_tvm_v4_release_dtypefix_verified_20260816_fresh`。

两跑不是逐候选冻结重放：生成、KernelGym 完成顺序和 refill 均为异步；一旦某组的 filter 决策改变，
本步停止位置、后续 dataset cursor 和模型更新都会分叉。因此下表是最接近的 matched-window 在线证据，
不是单变量随机实验。

| step 0--26 | 修复前 | 修复后 | 变化 |
|---|---:|---:|---:|
| low-variance drop groups | 576 | 404 | -172（-29.9%） |
| 每步平均 drop | 21.33 | 14.96 | -6.37 |
| 工作口径 drop 概率 `drop/(drop+accepted)` | 57.14% | 48.33% | -8.82 pp |
| drop 中 FP16+BF16 | 171（29.69%） | 110（27.23%） | -61 |
| accepted precheck rate | 82.76% | 83.97% | +1.21 pp |
| accepted compile rate | 71.85% | 67.63% | -4.22 pp |
| accepted correctness rate | 52.67% | 47.86% | -4.81 pp |

27 个 step 中修复后有 19 个 step 的 drop 更少、1 个相同、7 个更多。新 run step 27--35 的
low-precision drop 占 20.59%，已接近数据全集的 20.71%；这是“precision 误判富集消退”的一致证据，
但同时包含学习进展和 cursor 分叉，不能单独解释为 dtype 的纯效应。

accepted compile/correctness 反而降低，不是 KernelGym 回归的证据。rollout 指标只对最终接受的样本
聚合；修复使一部分此前 16/16 全零的困难组变成有 reward 方差的可训练组，harder population 被纳入后，
accepted-pool 的平均成功率自然可能下降。与此同时 rollout 平均耗时下降约 12%，吞吐提高约 9%，
与 refill 浪费减少方向一致。

实际训练请求进一步闭合了 contract：step 0 的 14 个 FP16 和 21 个 BF16 KernelGym 任务全部带有
正确的 request/result/static-check precision，静态检查均通过，未再出现 `precision_downgrade` 或
`required FP32`。失败已转移到真实编译、运行时和 correctness 阶段。具体例子：

- `layout_b24e86ef6ca36fc7aa34e9be` 是继承 FP16 的 dtype→layout 组；修复前 16/16 为 0，修复后同一
  首轮候选序列中不再被 low-variance 或 small-group filter 丢弃，是一个直接的 rescued group。
- `dtype_7b2df0d61e5b396a01413e2b` 和 `dtype_23784819b5f4a5f34e60a25d` 在两跑中仍为全零 BF16 组，
  说明正确 precision 只移除了错误门禁，不会自动修好模型代码。

### 6.6 Entropy 影响边界

修复后的共同窗口中，rollout/train entropy 下降斜率约为修复前的 2.3 倍；但 precision
在 response 生成后才进入 evaluator，只能通过 reward/filter 改变 accepted population 和后续
梯度，不能直接改变同一 checkpoint/prefix 的 logits。§6.5 保留本报告负责的 drop、precision
分层和 rescued-group 证据；完整 entropy 曲线、优化器/mismatch 控制、TRLOO × global-token
中介链和冻结 replay 缺口统一见
`handoffs/deepseek-v4/entropy_0731_v4_fresh_audit_20260816.md`。

## 7. 已解决和未解决的边界

已解决：

- FP16/BF16 任务不再因请求丢失 dtype 而默认套用 FP32 静态合同；
- 普通 FP32 任务仍保留禁止静默降精度的门禁；
- dtype→layout 串行扩增不会因当前子层 `dtype_after=None` 而漏判；
- 两个 KernelGym backend 与 result-cache 身份都覆盖 precision。

未解决：

- 已量化 aggregate drop 改善，但没有固定 response 配对重放，无法把 172 个净减少逐组归因给 dtype；
- 语法、API、runtime、非法内存和大幅 correctness error 仍是真失败；
- `max_turns=1` 与失败 reward 全折叠为 0 的训练信号问题未变；
- 历史 pending/None reference 子集尚未完成 reference-only 复验；
- filter audit 的 `rollout_step` 仍为 `null`，KernelGym 的部分 runtime/timeout error result 也缺 precision
  metadata；本次靠日志时序和 task 查询离线关联，观测性缺口仍在；
- 没有 accepted/all-candidate 的 `precision × reward × advantage × length × entropy` 样本级联表，
  因而 entropy 加速只能定位到 selection-mediated 机制，不能量化其中 dtype 的独立份额。

## 8. 下一步最小验证

不需要先启动另一轮完整训练。建议按以下顺序闭环：

1. 用 `--debug-rollout-only --save-debug-rollout-data` 保存全部候选，而不启动训练 backend；记录每个
   response 的 prompt id、resolved precision、KernelGym task id、task/effective reward、长度、sampled
   surprisal 和 filter decision。
2. 对同一批已生成 response 分别按旧 FP32 默认和正确 precision 重放环境，直接得到
   `reward/filter decision` 的 paired dtype effect；另对 pending/None 子集执行 reference-only 验证。
3. 固定同一 accepted batch 和 optimizer/RNG clone，用 `--debug-train-only --load-debug-rollout-data`
   比较两套 filter/reward population 的单步 `ΔH`；同时在固定 prompt/prefix 上测更新前后 full-vocab H，
   避免在线 accepted-population 指标混入选择偏差。
4. 在 filter audit 写入真实 rollout id 与 resolved precision，并让 KernelGym error result 保留 request
   precision，消除当前离线时序关联和 `precision=None` 的观测歧义。

## 9. 可复核证据

- 修复前 rollout 日志：node69 `/tmp/dsv4_formal.out.prev.122926`；共同窗口 step 0--26 的
  low-variance drop 合计 576。
- dtype 修复后 rollout 日志：node69 `/tmp/dsv4_formal.out`；step 0--26 drop 合计 404，修复后
  step 0--35 合计 540。
- Codex CLI 原始审计 session：`train-analysis-0816`，session id `01a00961-b9db-7033-ad6c-d4fc37cbf079`；本机记录 `/root/.codex/sessions/2026/08/16/rollout-2026-08-16T16-03-14-01a00961-b9db-7033-ad6c-d4fc37cbf079.jsonl`。
- Slime dtype 解析与请求：`examples/kernel_agent/generate_with_cuda_agent.py`、`examples/kernel_agent/kernel_response.py`。
- Slime filter/reward：`examples/kernel_agent/kernel_filter.py`、`examples/kernel_agent/config.py`、`examples/kernel_agent/kernel_reward.py`。
- KernelGym precision 实现与测试：提交 `737542bb8527`、`kernelgym/schema/precision.py`、`tests/test_precision_passthrough.py`。
- entropy 目标、reducer 与固定探针方案：`handoffs/deepseek-v4/entropy_0731_v4_fresh_audit_20260816.md`。
- Kimi 对抗式复核同意主要边界：服务管道整体可用；不能把 precision policy 与未验 reference 子集直接归到模型失败，也不能从 `30/66` 外推 414 个 group 的恢复率。
