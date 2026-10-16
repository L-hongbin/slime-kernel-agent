# 使用文档

## slime 参数简介

在使用 slime 时，传参主要是为了如下几件事：

1. 把集群中一部分 GPU 分配做训练，一部分分配做推理；
2. 训练的部分加载 megatron；
3. 推理部分加载 sglang；
4. 配置 RL 训练需要的超参。

按照这个顺序，我们需要配置这些参数：

### 集群资源分配

集群资源分配主要有这样的 4 个参数：

- `--actor-num-nodes`：RL 的 actor 训练需要多少节点；

- `--actor-num-gpus-per-node`：RL 的 actor 训练的每个节点有卡；

- `--rollout-num-gpus`：rollout （inference）一共需要多少卡。设置为 `0` 时，slime 仍会解析 SGLang 参数并启动 router，但不会启动本地 SGLang server；

- `--rollout-num-gpus-per-engine`：每个 inference engine 有多少卡，这个参数会比较像 sglang 的 `tp_size`，也就是在进行多机 serving 的时候，这个数值应该是总卡数，例如 2 机 16 卡 serving 一个模型，这里的值应该是 16。

  这里不像其他的 sglang 参数那样引入 `--sglang-tp-size` 是因为未来也许会考虑支持 sglang 的 dp_size 参数，也就是一个 engine 里面其实是有多个 sglang server 的（目前只支持 `--sglang-enable-dp-attention` 情况下的 `--sglang-dp-size`）。

在默认的配置下，我们会根据这些参数，通过 ray 给训练部分分配 `actor_num_nodes * actor_num_gpus_per_node` 张 GPU，给推理分配 `rollout_num_gpus` 张 GPU，也就是实现了训推分离。

当需要训推一体的时候，还需要配置上：

- `--colocate`：开启训推一体。开启后默认会让训练和推理的卡数相等；也可以显式设置一个不同的正数，例如让 rollout 卡数多于 actor，多出的 GPU 会作为 rollout-only 资源使用。如果显式设置 `--rollout-num-gpus 0`，则只启动 router，不启动本地 SGLang server。

此外，slime 支持 Prefill 和 Decode 的分离部署 (PD Disaggregation)，可以通过设置 `--prefill-num-servers` 参数来指定用于 Prefill 的服务器数量。

### 选择训练后端

slime 当前使用 Megatron-LM 作为训练后端。为了兼容已有脚本，仍然可以显式传入
`--train-backend megatron`。

### 加载 megatron

megatron 与 sglang, vllm 或者 huggingface trainer 之类的工具不同，他不能直接读取 huggingface ckpt，而是需要用户配置好要训练的模型的参数，并且加载 megatron 自己的 ckpt。

一般来说，我们需要做 3 点准备：

- 配置模型参数
- 配置并行以及一些优化
- 配置需要加载的 ckpt

对于一些 megatron 的自定义以及 slime 引入 megatron 的原理，请见 megatron 使用方法一节。

#### 配置模型参数

这里以 qwen3 4B 为例，我们需要这些参数：

```bash
MODEL_ARGS=(
   --num-layers 36
   --hidden-size 2560
   --ffn-hidden-size 9728
   --swiglu
   --vocab-size 151936
   --disable-bias-linear
   # attn head
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   --kv-channels 128
   --qk-layernorm
   # norm
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   # rope
   --use-rotary-position-embeddings
   --rotary-base 1000000
)
```

我们在 [scripts/models](../../../scripts/models) 提供了常用模型的配置，可以直接复用。如果你也在使用 megatron 进行 pretrain/sft 的话，可以直接复用 pretrain/sft 中的模型配置。

注意：

- slime 会加载 `PYTHONPATH` 中的 megatron 的所有参数，所以可以在环境中的 megatron 里找参数以及参数的说明；
- slime 会使用 data packing (或称 varlen 或 thd) 进行训练，无需配置 `--seq-length` 或 `--max-positional-embedding`，这两个参数不会影响训练模型的最大 context length。

#### 设置各种并行与重计算

megatron 是目前优化最为齐全的训练框架，大家使用 megatron 的一个主要目的就是追求其卓越的性能，这里简单介绍一些 megatron 的并行和重计算的配置方法。

- 这里我们简单陈列 megatron 的并行策略，关于这些并行策略之间的 trade-off 请参考更专业的一些讨论：
  - `--tensor-model-parallel-size`：tp
  - `--sequence-parallel`：megatron 的 sp 是 tp 的一种优化，推荐在使用 tp 的时候一直开启 sp。
  - `--pipeline-model-parallel-size`: pp
  - `--context-parallel-size`：megatron 的 cp，也就是序列并行，一般对应 ring attention；
  - `--expert-model-parallel-size`：moe 的 ep，每张卡上有 `num_experts / ep_size` 个 expert；
  - `--expert-tensor-parallel-size`：megatron 支持 moe 的 expert 与其他部分采用不同的 tp_size，我们一般称为 etp。
- 对于重计算，megatron 中一般是配置如下的几个 flag：
  - `--recompute-granularity` 这个值可以选 full 或者 selective，full 就是完全重计算，selective 会少重计算一些，不配置就是不重算；
  - `--recompute-method`：一般用 uniform 就行；
  - `--recompute-num-layers`：多少层分一组来做重算，一般 1 就行。
  

#### 加载 megatron ckpt

megatron 支持多种其自定义的 ckpt 格式，这里介绍 2 种比较主流的格式，

- 曾经比较主流的 torch 格式（对应 `--ckpt-format torch`）；
- 现在推荐使用的 torch_dist 格式（对应  `--ckpt-format torch_dist`）

torch 格式是 megatron 的老存储格式，里面的结构大约是一些 `mp_rank_xxx` 的文件夹，每个文件夹对应了在对应的并行划分下，每个 rank 存储的 ckpt。也是因为如此，在加载 torch 格式的 ckpt 的时候，需要保证 ckpt 的并行策略和训练任务的并行策略是相同的。

我们推荐使用 torch_dist 格式 ckpt，因为 torch_dist 格式可以支持自动并行切分，也就是不同并行的训练任务都可以共用同一个 ckpt，会方便很多。torch_dist 这也是开源 megatron 目前的默认格式。torch_dist 格式的 ckpt 中一般是一堆 `.distcp` 文件。在使用 torch_dist 时，可以使用 [README](../../../README_zh.md) 中介绍的 ckpt 转化方法从 huggingface 转化为 torch_dist，反之亦然。

在存储结构上，megatron 的 ckpt 一般是这样的结构，这里假设存储的路径为 `/ckpt/`：

```bash
--/ckpt/
    |-- latest_checkpointed_iteration.txt
    |-- iter_0000100/
         |-- _0_0.distcp
         |-- _0_1.distcp
         |-- ...
    |-- iter_0000200/
    |-- iter_0000300/
    |-- ...
```

其中 `latest_checkpointed_iteration.txt` 中记录了训练最新的训练步。在加载模型时，不能直接传入 `/ckpt/iter_xxxxxxx`，而是要传入 `/ckpt/`，并用 `--ckpt-step` 来选取对应的训练步（如果不使用 `--ckpt-step`，则会通过 `latest_checkpointed_iteration.txt` 读取对应的训练步。）

在使用 slime 的时候，有 3 个参数用来加载和保存 ckpt：

- `--ref-load`：reference model 用的 megatron ckpt；
- `--load`：actor 用的 megatron ckpt，如果没有设置 `--load`，或者设置的目录不存在，目录中没有 `latest_checkpointed_iteration.txt`，都会直接从 `--ref-load` 的 ckpt 进行初始化；
- `--save`：actor 保存的路径。

注意：

- 不管进行何种方式存储 ckpt，即无论如何设置 `--ckpt-format`，megatron 都可以加载 torch 或 torch_dist 格式

### 加载 sglang

sglang 的加载非常简单，只需要：

- `--hf-checkpoint`：初始化 sglang 用的 huggingface ckpt；

注意：

- 在第一个训练步之前，slime 会把 megatron 里的参数同步给 sglang，所以 `--hf-checkpoint` 中不需要有最新的训练参数，在续训得时候也不需要更换 hf ckpt；
- sglang 默认会从 huggingface ckpt 中 `config.json` 读取模型的最大 context length，可以使用 `--sglang-context-length` 参数来对这个值进行覆盖，从而支持进行更长的推理；
- 在训推一体的训练过程中，虽然 megatron 和 sglang 会先后 offload，但是还是需要为对方留有一些空间，需要通过减小 `--sglang-mem-fraction-static` 来调整 sglang 的显存占用总量。
- slime 支持透传 sgl-router 的参数，方式是在原参数名前加上 `router` 前缀。例如，sgl-router 的 `--balance-abs-threshold` 参数需要设置为 `--router-balance-abs-threshold`。由于 sgl-router 默认使用 cache-aware routing，可能会导致请求分配不均衡的问题。可以通过设置 `--router-balance-abs-threshold 0` 来强制均衡分配，但这可能会影响多轮对话场景下 prefix cache 的命中率。
- 如果 SGLang engine 已经由外部系统预启动，可以通过 `--rollout-external-engine-addrs host1:port host2:port` 连接。此时如果训练器和 engine 无法建立 NCCL 权重同步 group，可以使用 `--update-weight-mode full --update-weight-transport disk --update-weight-disk-dir /shared/fs/updates`，slime 会写完整 HF checkpoint 并调用 SGLang 的 `update_weights_from_disk` 热加载；大模型或跨集群场景可进一步使用 `--update-weight-mode delta --update-weight-transport disk`。详见 [External Rollout Engines 配置路线图](../advanced/external-rollout-engines.md) 和 [Delta 权重同步](../advanced/delta-weight-sync.md)。

对于一些 sglang 的自定义以及 slime 引入 sglang 的原理，请见 sglang 使用方法一节。

### 数据格式

slime 支持加载 `.jsonl` 和 `.parquet` 格式文件；读取 Parquet 需要安装 `pyarrow`。两种格式中的每条记录都应包含 `--input-key` 和 `--label-key` 指定的字段。下面是一条 JSONL 数据展开后的示例：

```json
{
  "prompt": [
    {
      "content": "Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{$Answer} where $Answer is the answer to the problem.\n\nIn triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$ be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$ and $\\angle BDC = 90^\\circ$. Suppose that $AD = 1$ and that $\\frac{BD}{CD} = \\frac{3}{2}$. If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$ where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$.\n\nRemember to put your answer on its own line after \"Answer:\".",
      "role": "user",
      "step_loss_mask": 1,
    }
  ],
  "label": "34"
}
```

对应的配置为：

```bash
  --input-key prompt
  --label-key label
  --apply-chat-template
```

请注意，这里的 `step_loss_mask`（默认值为 1）字段为 SFT 阶段提供，若设置为 0，则会将该轮 `loss_mask` 设置为 0；若设置为 1，则使用正常 `loss_mask`。
另外我们还提供了一个 metadata_key，默认为 `"metadata"`，读取后我们会把数据中的 metadata 加载进 slime，可能会对自定义数据生成或者自定义 reward model 有帮助。

如果同一次训练混合了多个数据 source，可以在 metadata 中写入 `source_name`：

```json
{
  "prompt": "...",
  "label": "...",
  "metadata": {
    "source_name": "math"
  }
}
```

推荐把 source 标识放在 `metadata["source_name"]` 中；自定义 data source 如果已经动态设置了 `sample.source`，slime 也会识别。rollout 转换成训练数据时，slime 会为每个样本生成 `source_names` 并传到训练侧。source 的读取优先级为动态 `sample.source`、`metadata["source_name"]`，都不存在时为 `"unknown"`。这可以用于自定义 reward、filter、日志统计，以及后续按 source 路由 OPD teacher 等需要分 source 处理的场景。

### RL 训练需要的超参

- `--advantage-estimator`: 当前训练需要的 RL 算法，目前支持：
  - `grpo`（https://arxiv.org/abs/2402.03300）；
  - `gspo`（https://arxiv.org/abs/2507.18071）；
  - `cispo`（https://arxiv.org/abs/2506.13585）；
  - `reinforce_plus_plus` 与 `reinforce_plus_plus_baseline`（https://arxiv.org/abs/2501.03262）；
  - `ppo`（https://arxiv.org/abs/1707.06347）。

  注意：在策略蒸馏 (OPD) 现在与 advantage estimator 正交，使用 `--use-opd` 和 `--opd-kl-coef` 可以在任意 estimator 之上启用 OPD。
- `--calculate-per-token-loss`：slime 中默认的方案是 per sample loss，即 `mean(sum(sample_i) / len(sample_i))`，如果需要计算 per token loss，即 `sum(sum(sample_i)) / sum(len(sample_i))`，可以开启 `--calculate-per-token-loss`；
- `--use-tis`：如果需要开启 tis（https://fengyao.notion.site/off-policy-rl），可以开启这一设置；

`--calculate-token-sum-loss` 使用 [MiniRL 公式 (7)](https://arxiv.org/html/2512.01374v1#S4.SS1) 的 PG 聚合方式：`sum(有效 token 的 PG loss) / 本训练步的 rollout 数`，不除以实际回复长度。多轮时同一轨迹的所有 turn 累加，外层仍按轨迹数平均；padding 和被 mask 的 token 不贡献 loss。

该开关默认关闭，不能与 `--calculate-per-token-loss` 或 `--custom-pg-loss-reducer-function-path` 同时启用，仅支持 `--loss-type policy_loss`。它不改变 advantage、重要性采样和 clipping，也不改变 entropy/KL 项及其他诊断指标的归约。若使用 GRPO 并希望只减组内 reward 均值，还需单独传入 `--disable-grpo-std-normalization`。因此这不是完整 MiniRL 算法开关。

去掉长度分母会明显增大 PG 梯度尺度，切换时应检查学习率、梯度范数及梯度裁剪比例。KernelAgent 的 `run_qwen3.6_27B_full_async_dppo.sh` 可用 `CALC_LOSS_MODE=TokenSum` 启用；脚本默认仍是 `PerToken`。

`--calculate-per-prompt-loss` 启用 prompt-mean PG 聚合，参考
[slime PR #2090](https://github.com/THUDM/slime/pull/2090)：同一 `Sample.group_index` 下所有
candidate、所有 turn 的有效 token loss 相加，除以该 prompt 的有效 token 总数，再对当前
optimizer step 的非空 prompt 求平均。默认关闭，仅支持 Megatron 的 `--loss-type policy_loss`；
不能与 per-token、token-sum 或自定义 PG reducer 同时启用。不改变 reward、return、advantage
和 entropy/KL 的聚合方式。

分母在 DP/microbatch 切分前计算，TIS/RS 后续拒绝 token 时不重新计算分母。缩放使用当前 step
的实际轨迹数 / 非空 prompt 数，支持不等大的 group。全 mask 的 prompt 不计入 prompt 数；
整个 step 无有效 token 时明确报错。同一 prompt 可以跨 DP rank/microbatch，但跨 optimizer step
或尾部只保留部分 prompt 时会报错。请保持 group 连续，并选择合适的 global batch size
（固定 group size 时可取其整数倍）。自定义训练数据转换器需提供 `group_indices`。
KernelAgent 的 Qwen 启动脚本支持 `CALC_LOSS_MODE=PerPrompt`，默认仍为 `PerToken`。

#### ArgMaxRL advantage

```bash
--advantage-estimator argmaxrl
# 若完整 reward 的已知下界为 -1，再加：
--argmaxrl-reward-offset 1
```

实现参考 [ArgMaxRL](https://www.doubleai.com/research/argmaxrl-generalizing-maxrl-to-continuous-rewards)。
直接使用已结算的聚合 reward（包括已启用的动态调权、失败评分及长度分数），不拆分 correctness、
performance、coverage，也不额外应用 correctness gate。默认 offset 为 0；只在计算权重时加上固定
offset，不改写 `sample.reward`、`task_reward`或组件。偏移后仍为负或非有限值时报错；
不会按组减最小值或将负数截成零。

同组 reward 降序排列 `r[1] >= ... >= r[N] >= 0`，并令 `r[N+1] = 0`：

```text
w[j] = sum((r[m] - r[m+1]) / m, m=j..N)
A[j] = N * w[j]
```

`N` 是该组有效候选数，不是配置的固定 group size 或 turn 总数。`N*w` 适配按候选平均的约定：
`mean(A * score) = sum(w * score)`。二值奖励退化为成功样本 `A=N/K`、失败样本 `A=0`。
不减组均值、不除标准差，`--disable-grpo-std-normalization` 无需额外设置；
`--normalize-advantages` 和自定义 advantage 函数会在启动时被拒绝。

权重在 `RolloutManager._post_process_rewards` 中、DP 切分前按完整组计算，训练侧只广播到本地 token。
使用 `Sample.group_index`，多轮使用 `(group_index, turn_idx)`，不累加 TRLOO `return_reward`。
移除和全 mask 样本不参与排序且 advantage 为零；同组内同一轨迹的 fan-out 片段只计一个候选，
要求它们携带相同 reward。自定义 reward hook 应返回未中心化的聚合分数，随后只执行一次 ArgMaxRL。

现有 loss 聚合、PPO/DPPO clipping、动态过滤、CTM 和 OPD 不自动改变，因此组合后的训练不应宣称
严格复现原文的无偏 REINFORCE 梯度。`TokenSum` 更接近原文完整 response 的 log-prob 求和；
PerSample/PerToken/PerPrompt 会施加各自的长度归一化。相同的正 reward 仍产生非零权重，现有低方差
过滤若开启仍可能过滤这些组。动态调权等依赖当前组的 reward 变换也属于额外变体。
`--kl-coef` 不支持；需要 KL 正则时使用独立 `--use-kl-loss`。

#### TailRL advantage

使用 `--advantage-estimator tailrl`。复用上述 ArgMaxRL 的权重和分组管线，仅在每组有效独立候选之间
增加中心化：`A = N * (w - mean(w))`，不除标准差。对齐
[TailRL 官方 code optimization 实现](https://github.com/Zanette-Labs/TailRL/blob/5682c6ac03387355e017ce966693266bb148fa10/experiments/code_optimization/code_opt/advantages.py#L22)，
不是官网简化伪代码中省略 N 的版本。

例如 reward `[0, 1, 3]`，ArgMaxRL 的 advantage 为 `[0, 1.5, 7.5]`，TailRL 为 `[-3, -1.5, 4.5]`。
全组同分、全零和单候选组均为零；组内有成功时，二值失败样本为 -1，成功样本为 N/K-1。
padding、移除样本不参与均值，fan-out 片段共享同一候选的结果，不重复计数。

TailRL 对 reward 的共同平移不变，支持有限负 reward，不需要也不接受非零
`--argmaxrl-reward-offset`。内部先减组最小值再复用非负 tail 权重计算，中心化后的结果与官方公式
等价；这个步骤只用于 TailRL，不改变原 ArgMaxRL。不修改 `sample.reward` 或 reward components。

多轮仍按同题同 turn 计算，DP 切分前中心化；不折算 TRLOO 未来 reward。与 ArgMaxRL 相同，禁用
`--normalize-advantages`；其它 loss/过滤/CTM/OPD 设置保持独立。
组内减均值不是跨 batch whitening，也不应把这种依赖同组样本的 baseline 直接等同于原始未中心化
ArgMaxRL 的有限样本无偏估计器。

#### Kernel-agent 上下文预算提醒

Kernel-agent 多轮 rollout 可通过 `--use-context-budget-nudge 0.2` 开启上下文预算提醒，参考 [Mercor 的 context nudge](https://www.mercor.com/blog/training-frontier-knowledge-work-agents-a-397b-rl-training-guide-with-skyrl/)。参数为 `(0, 1]` 内的有限比例；不传或显式传 `None` 均关闭。要求 `--rollout-max-context-len > 0`。`0.2` 表示剩余上下文大于 0 且不超过上限的 20% 时，提醒下一轮优先提交完整、正确的 kernel，避免探索性优化。

`GenerateState` 加载模板时，用 rollout tokenizer 统计模板原文的 token 数，缓存为模板的 `template_tokens` 属性；内置 fallback 模板也在首次使用时缓存。`_apply_feedback_template` 每轮统计序列化、截断后的 feedback，得到 `feedback_tokens`，即使关闭 nudge 也统计。两项均使用 `add_special_tokens=False`，与格式化后的 feedback 一起返回，写入 turn log 并打印到 turn stats。nudge 使用 `prompt_tokens + response_tokens + template_tokens + feedback_tokens` 估计上下文长度，在模板渲染前构建提醒，以 `context_budget_nudge` 传入模板（未触发时为空字符串）。这是估计值：模板原文包含占位符/Jinja 语法，分段 tokenize 的边界也可能有误差，且未计入新增 chat framing 和提醒文本。不为此重复 tokenize 整段上下文，下一轮已有的上下文检查负责实际长度限制。

内置及仓库中的 response 模板已包含提醒字段；自定义 format/YAML 模板需要加入 `{context_budget_nudge}`，Jinja 模板需要加入 `{{ context_budget_nudge }}` 才会展示提醒。每轮满足条件都可以追加，不是整条轨迹只提醒一次。提醒属于下一轮 user 输入，不是 assistant 输出，不直接修改 reward 或 response loss mask；只在轮次之间生效，不能中断单轮长思考。若评测也打开此参数，评测同样生效。

#### Kernel Agent 有理函数速度奖励

速度得分默认使用 `legacy`：`f(x) = min(x, M)`。设置以下环境变量可启用平滑有界的有理函数：

```bash
export CUDA_AGENT_SPEEDUP_SCORE_MODE=rational
export CUDA_AGENT_SPEEDUP_REWARD_UPPER=2
```

`x` 是原始加速比；`M` 复用 `CUDA_AGENT_SPEEDUP_REWARD_UPPER`，默认 `2`，在 `rational` 模式下必须有限且大于 `1`。
得分为 `f(x) = M*x/(x+M-1)`，满足 `f(0)=0`、`f(1)=1`，随加速比增加而趋近 `M`，不会在 `x=M` 处截断。
例如 `M=2` 时，`0.5x / 1x / 2x / 5x` 对应 `0.667 / 1 / 1.333 / 1.667`。
非有限或非正的加速比不获得速度得分；错误程序仍走原有失败奖励路径。

该模式只替换 `kernel_score.performance` 的映射，随后仍乘 performance 权重和 speedup 动态系数；correctness、coverage 的独立调权和原始 speedup 指标保持原有计算方式。
Qwen3.8 WarmUp/MultiTurn 启动脚本会将模式透传给 Ray worker；非 `legacy` 模式的实验名追加 `-SSM模式名`，例如 `-SRUP2-SSMrational`。
自定义启动脚本也需要将 `CUDA_AGENT_SPEEDUP_SCORE_MODE` 和 `CUDA_AGENT_SPEEDUP_REWARD_UPPER` 放入 Ray runtime environment。

#### KernelGYM 详细正确性诊断

`CUDA_AGENT_RETURN_DETAIL_CORRECTNESS` 默认 `0`（`False`），训练启动前设为 `1` 即可在 KernelGYM 评测请求中传入 `return_detail_correctness=true`。Qwen3.8 的 WarmUp/MultiTurn 脚本已通过 Ray runtime environment 透传；自定义启动脚本也需要将此环境变量传给 rollout worker。

该开关请求详细正确性诊断，不修改 reward 计算，与 `CUDA_AGENT_ENABLE_COMPUTE_SANITIZER` 独立：两者可以各自单独开启，也可以同时开启。sanitizer 是否实际执行仍取决于服务端触发规则。显式诊断命令 `run_request_env.py --mode sanitizer` 只开启 sanitizer，详细正确性诊断仍由 `CUDA_AGENT_RETURN_DETAIL_CORRECTNESS` 控制，默认关闭。

#### KernelGYM 详细编译诊断

`CUDA_AGENT_RETURN_DETAIL_COMPILATION` 默认 `0`（`False`）。设为 `1` 后，在 KernelGYM 请求中传入
`return_detail_compilation=true`，启用编译错误分类，返回 `metadata.compilation_error_detail` 和摘要形式的
`error_message`；关闭时服务端跳过分类，在 `error_message` 中返回完整编译错误文本。
该开关与详细正确性诊断、Compute Sanitizer 独立，不改变 reward 计算。

```bash
export CUDA_AGENT_RETURN_DETAIL_COMPILATION=1
```

Qwen3.8 WarmUp/MultiTurn 脚本已通过 Ray runtime environment 透传，自定义启动脚本也需要传给 rollout worker。
训练请求、`run_request_env.py` 和 `run_response_pipeline.py` 都沿用此配置；`--mode compile` 不会自动开启详细编译诊断。

#### Kernel rollout reward 后处理

`examples.kernel_agent.kernel_reward.post_process_rollout_rewards(args, samples)` 接收一批 samples，返回逐 turn 的处理后 reward，并同步更新 `sample.reward`。动态权重和长度惩罚在这里统一管理；同题同轮次的 group 只作为动态权重的内部统计范围。接口不计算累计 return、baseline 或归一化 advantage。

现有训练 hook 已调用此接口，启动时仍使用：

```bash
--custom-reward-post-process-path examples.kernel_agent.kernel_reward.reward_post_process_by_group \
--dynamic-reward-speedup correct-rate-sqrt \
--dynamic-reward-coverage correct-rate-sqrt \
--overlong-penalty dapo \
--overlong-buffer-len 2048 \
--overlong-penalty-factor 0.2
```

三个独立入口 `--dynamic-reward-correctness`、`--dynamic-reward-speedup`、`--dynamic-reward-coverage` 均默认为 Python `None`；未设置或显式传 `None` 时，对应分量不调权。`--overlong-penalty` 独立选择长度奖惩，总是在动态调权之后执行。旧 `--dynamic-reward-gate`、`--dynamic-reward-gate-range`、`sqrt` 和 `inverse-gate` 已移除，三个入口统一使用 `--dynamic-reward-amplitude`，`--dynamic-reward-range` 已移除。后处理接口返回单个 reward 列表，不能直接替换返回 `(raw_rewards, rewards)` 的训练 hook。

##### 三个 reward 分量独立加性调权

每个入口仍支持 `piecewise-sqrt`、`piecewise`、`correct-rate-sqrt`、`correct-rate` 及对应的 `inverse-*`。三个入口都额外支持 `global-piecewise`、`global-piecewise-sqrt` 及对应的 `inverse-*`。对于三个分量，正向模式在配置基础权重上 **加上** `x(p)`，inverse 模式 **减去** `x(p)`。这里 inverse 表示反向偏移，不再表示倒数。省略或显式传 `None` 时，对应贡献保持不变。

使用匹配曲线时，选择 correctness inverse、speedup 正向：

```bash
--dynamic-reward-correctness inverse-piecewise-sqrt \
--dynamic-reward-speedup piecewise-sqrt \
--dynamic-reward-amplitude 0.1 \
--dynamic-reward-coverage None
```

默认基础权重为 `0.5 / 0.5`，最终得到 `0.5-x(p)` 和 `0.5+x(p)`，总和为 1。若通过 `CUDA_AGENT_CORRECT_REWARD_WEIGHT`、`CUDA_AGENT_SPEEDUP_REWARD_WEIGHT` 改写基础权重，则分别为 `base_correctness-x(p)`、`base_speedup+x(p)`，仍保持原来的权重总和。选择不同曲线、相同方向或只启用一项时，不保证总和不变。固定总权重不等于严格固定 reward 期望。

令 `G` 为同一个 `(group_index, turn_idx)` 的有效样本数，`k` 为正确数，`p=k/G`；remove、aborted、pad 和非 kernel 样本不计入。`--dynamic-reward-amplitude a` 默认 `0.1`，控制 `x` 的值域 `[-a,a]`。要求 `a` 有限且位于 `[0,0.5]`，并且不超过每个已启用分量的有限基础权重，确保最终权重非负。

- `correct-rate`：`x = a*(2*p-1)`，在 `p=0.5` 时平衡。
- `correct-rate-sqrt`：`x = a*(2*sqrt(p)-1)`，在 `p=0.25` 时平衡。
- `piecewise`：使用下方分段公式，`f(t)=t`。
- `piecewise-sqrt`：使用相同分段公式，`f(t)=sqrt(t)`，两端调整更强。
- `global-piecewise`：在 `[h,e]` 内令 `x=0`，区间外使用 `x=a*(2*p-1)`，不按阈值重新缩放。
- `global-piecewise-sqrt`：在 `[h,e]` 内令 `x=0`，区间外使用 `x=a*(2*sqrt(p)-1)`；详见下方。
- 所有 `inverse-*` 都改为减去对应的 `x`；包括 `p=0` 在内，不再执行除法。

共享 `--difficulty-thresholds` 默认为 `[1/3,2/3]`。分段模式和 global-piecewise 模式要求恰好两个有限、严格递增且位于 `(0,1)` 的阈值 `h,e`：

```python
if G <= 1:
    x = 0
elif p < h:
    x = -a * f((h - p) / h)
elif p > e:
    x = a * f((p - e) / (1 - e))
else:
    x = 0
```

空组在所有模式下均不偏移。分段模式和 global-piecewise 模式沿用 singleton 不调整的规则；`correct-rate`、`correct-rate-sqrt` 按 singleton 的真实 `p` 计算。阈值点属于中间区间。`--dynamic-reward-range` 已由 amplitude 替代：基础权重为 0.5 时，旧对称乘数范围 `[0.8,1.2]` 对应 `a=0.1`；设置 `a=0.2` 可得到 `[0.3,0.7]` 的最终权重范围。

已开启分量每次都从基础评分重建，重复处理不会累积偏移。speedup 的 correctness 前置条件仍在选择权重后应用；失败分、长度奖惩、原始任务奖励和冻结的未来 credit 保持原有行为。日志中的权重是按 group 选择的权重，尚未乘单样本 correctness 条件，因此不表示该样本必然获得非零贡献。

Qwen3.8 WarmUp/MultiTurn 脚本仍使用 `DYNAMIC_CORRECTNESS_REWARD`、`DYNAMIC_SPEEDUP_REWARD`、`DYNAMIC_COVERAGE_REWARD`，均默认 `None`。新增 `DYNAMIC_REWARD_AMPLITUDE`，默认 `0.1`；开启任一分量调整时，幅度也会写入实验名。

##### Group=16 的最终权重

取 a=0.2、默认基础权重和阈值 [1/3,2/3]，correctness 使用 inverse、speedup 使用正向模式。每格为 correctness / speedup。

| k | p | piecewise | piecewise-sqrt |
|---:|---:|---:|---:|
| 0 | 0.0000 | 0.7000 / 0.3000 | 0.7000 / 0.3000 |
| 1 | 0.0625 | 0.6625 / 0.3375 | 0.6803 / 0.3197 |
| 2 | 0.1250 | 0.6250 / 0.3750 | 0.6581 / 0.3419 |
| 3 | 0.1875 | 0.5875 / 0.4125 | 0.6323 / 0.3677 |
| 4 | 0.2500 | 0.5500 / 0.4500 | 0.6000 / 0.4000 |
| 5 | 0.3125 | 0.5125 / 0.4875 | 0.5500 / 0.4500 |
| 6 | 0.3750 | 0.5000 / 0.5000 | 0.5000 / 0.5000 |
| 7 | 0.4375 | 0.5000 / 0.5000 | 0.5000 / 0.5000 |
| 8 | 0.5000 | 0.5000 / 0.5000 | 0.5000 / 0.5000 |
| 9 | 0.5625 | 0.5000 / 0.5000 | 0.5000 / 0.5000 |
| 10 | 0.6250 | 0.5000 / 0.5000 | 0.5000 / 0.5000 |
| 11 | 0.6875 | 0.4875 / 0.5125 | 0.4500 / 0.5500 |
| 12 | 0.7500 | 0.4500 / 0.5500 | 0.4000 / 0.6000 |
| 13 | 0.8125 | 0.4125 / 0.5875 | 0.3677 / 0.6323 |
| 14 | 0.8750 | 0.3750 / 0.6250 | 0.3419 / 0.6581 |
| 15 | 0.9375 | 0.3375 / 0.6625 | 0.3197 / 0.6803 |
| 16 | 1.0000 | 0.3000 / 0.7000 | 0.3000 / 0.7000 |

##### 全局线性曲线加中间不调整区间

新增 `global-piecewise`、`global-piecewise-sqrt`，现有模式不变。区间两端 h、e 都属于不调整区间，空组和 singleton 也保留基础权重。三个分量均支持这些模式及其 inverse，使用现有 amplitude 和 thresholds，不使用 `--dynamic-reward-range`：

```python
if G <= 1 or h <= p <= e:
    x = 0
else:
    x = a * (2 * f(p) - 1)
```

`global-piecewise` 使用 `f(p)=p`，`global-piecewise-sqrt` 使用 `f(p)=sqrt(p)`。

h、e 只决定是否开启调整，不参与两端幅度的缩放，因此阈值处通常存在跳变。区间外线性版本在 p=0.5 变号，平方根版本在 p=0.25 变号。默认阈值下，global 平方根版本在 p=5/16 时 x 为正，到 p=6/16 时置零，所以并非全局单调。不额外增加 piecewise-mode 参数。

```bash
--dynamic-reward-correctness inverse-global-piecewise \
--dynamic-reward-speedup global-piecewise \
--dynamic-reward-amplitude 0.1 \
--dynamic-reward-coverage None
```

Qwen3.8 启动脚本对应设置 `DYNAMIC_CORRECTNESS_REWARD=inverse-global-piecewise`、`DYNAMIC_SPEEDUP_REWARD=global-piecewise`。以下取 G=16、a=0.1、默认阈值和基础权重，每格为 correctness / speedup：

| k | piecewise | global-piecewise |
|---:|---:|---:|
| 0 | 0.60000 / 0.40000 | 0.60000 / 0.40000 |
| 1 | 0.58125 / 0.41875 | 0.58750 / 0.41250 |
| 2 | 0.56250 / 0.43750 | 0.57500 / 0.42500 |
| 3 | 0.54375 / 0.45625 | 0.56250 / 0.43750 |
| 4 | 0.52500 / 0.47500 | 0.55000 / 0.45000 |
| 5 | 0.50625 / 0.49375 | 0.53750 / 0.46250 |
| 6 | 0.50000 / 0.50000 | 0.50000 / 0.50000 |
| 7 | 0.50000 / 0.50000 | 0.50000 / 0.50000 |
| 8 | 0.50000 / 0.50000 | 0.50000 / 0.50000 |
| 9 | 0.50000 / 0.50000 | 0.50000 / 0.50000 |
| 10 | 0.50000 / 0.50000 | 0.50000 / 0.50000 |
| 11 | 0.49375 / 0.50625 | 0.46250 / 0.53750 |
| 12 | 0.47500 / 0.52500 | 0.45000 / 0.55000 |
| 13 | 0.45625 / 0.54375 | 0.43750 / 0.56250 |
| 14 | 0.43750 / 0.56250 | 0.42500 / 0.57500 |
| 15 | 0.41875 / 0.58125 | 0.41250 / 0.58750 |
| 16 | 0.40000 / 0.60000 | 0.40000 / 0.60000 |

##### coverage 使用相同的加性公式

`--dynamic-reward-coverage` 支持完全相同的模式，包括 `global-piecewise*` 和所有 inverse 选项。最终权重为 `coverage_reward_weight + x(p)` 或 `coverage_reward_weight - x(p)`；基础权重来自 `CUDA_AGENT_COVERAGE_REWARD_WEIGHT`，默认 0.5。它共用 `--dynamic-reward-amplitude` 和 `--difficulty-thresholds`。省略入口或传 `None` 保持原有 coverage 贡献；coverage 总开关仍决定该项是否贡献奖励。

`--dynamic-reward-range` 已移除，旧命令需要删除该参数，改用 `--dynamic-reward-amplitude`。基础权重为 0.5 时，旧正向分段范围 `[0.8,1.2]` 对应幅度 0.1。coverage 的 rate 和 inverse 模式也统一为加减偏移，不再使用旧乘法或倒数公式。开启 coverage 调整不保证三项总权重不变；匹配的 correctness inverse / speedup 正向配置只保持这两项的权重总和。

`kernel_score` 保留未乘 reward 权重的评分；`reward_component` 保存实际加权贡献，长度贡献记为 `length`：DAPO 惩罚为负数，LASER-D bonus 为正数，未启用或未触发时为 0。组件指标统一为 `reward/component/length`，不再分别记录 `overlong_penalty` 和 `length_bonus` 组件。顶层 `metadata.length_score` 使用相同有符号值，相关指标为 `kernel/length_score/mean` 和 `exp/rollout/reward/length_score/*`；现有 CLI 参数不变。`task_reward` 不含长度奖惩，`sample.reward` 包含长度奖惩。再次执行从评分/分量重建，避免重复扣分。

TRLOO 在完整轨迹收尾时、group reward 后处理和过滤之前，用原始 `task_reward` 固定未来折算项 `metadata.return_reward`：`gamma * 原始task_reward[t+1] + gamma² * 原始task_reward[t+2] + ...`。该项不含动态调权、失败组替换或长度奖惩；收尾时已标记移除的 turn 贡献为零。训练时只计算 `return[t] = 当前sample.reward + return_reward[t]`，所以当前 turn 保留全部后处理效果，未来 turn 不携带这些调整。之后的 group 过滤不会重算或删除已固定的未来贡献，单 turn reward 不被 return 覆盖。旧 rollout dump 若缺少此字段，需重新生成，不能直接用可能包含后处理结果的旧 `multi_turn_reward` 代替。

TRLOO 同时把固定的未来折算项记录到 `metadata.reward_component.return_reward`，用于观测。该组件不计入 `sample.reward`；对有效普通 TRLOO 样本，全部组件求和现在包含未来贡献，对应减 baseline 前的 return。动态调权和失败组替换均保留该项。所有支持的奖励组件（`correctness`、`performance`、`coverage`、`failed`、`length`、`return_reward`）均作为常规指标记录到 `rollout/reward/component/{字段}/{mean,min,max}`，无需 `--log-exp-metrics`；开启 `--use-tensorboard` 即可写入 TensorBoard。padding、缺失值和非有限值不进入组件统计，旧的 `exp/rollout/reward/component/*` 指标不再输出。

Megatron 训练侧在 advantage 计算及 rollout 后处理后、训练 forward 前，默认记录以下指标，无需 `--log-exp-metrics`；开启 `--use-tensorboard` 即可写入 TensorBoard：

- `rollout/advantage/{positive,negative,zero}_sample_fraction`：按每条样本有效 token 的平均 advantage 判定符号，统计样本占比。
- `rollout/advantage/{positive,negative}_abs_mass`：对应符号样本的 advantage 绝对值之和除以全部样本数，不是该符号内部的条件均值。
- `rollout/sequence/{positive,negative}_response_length`：对应符号样本的完整 response token 数均值，多轮中每个拆分 turn 算一条样本。

统计沿用当前 loss mask；全 mask 样本的平均 advantage 为零，计入零样本占比。DP/CP 汇总使用总和/计数；全局没有某符号样本时，其平均长度记为 0，应结合样本占比解读。这些指标不反映随后训练 forward 内 seq-MIS 新增的 mask。原 `exp/rollout/train_batch/advantage/*` 和 `exp/rollout/train_batch/sequence/*` 不再重复输出；按 turn 的细分占比和异步诊断仍由 `--log-exp-metrics` 控制。

`metadata.raw_task_reward` 保存按配置基础权重计算的原始单 turn 任务奖励，不含动态调权、失败组替换和长度奖惩。基础评分时写入，后续 reward 后处理不覆盖；它与可变的 `task_reward` 分开，可作为后续历史统计的数据源。TRLOO 的未来折算优先读取此字段；缺少字段的旧样本，在后处理前的轨迹收尾阶段仍回退到 `task_reward`，再回退到单 turn reward。该字段不是额外可加的 reward component。

`calculate_kernel_reward()` 只计算基础评分。普通 kernel 的处理顺序为：基础评分 → 完整 prompt/turn group 的 reward 后处理（动态调权 → 长度奖励/惩罚 → 失败组替换）→ 写回 `sample.reward` 和 `metadata.reward_component` → group 过滤 → return/advantage。默认 SGLang rollout 在配置 kernel reward hook 时自动接入 `examples.kernel_agent.kernel_reward.generate_rollout`；fully-async collector 在取出完整 group 后、过滤前执行相同后处理。

开启 `apply_failed_group_reward` 时，先按过滤器相同的奖励口径判断：有效样本全部 failed，且每个分数都精确等于默认失败分 `failed_score * init_correct_weight`，才尝试用各样本的 `kernel_failed_score * init_correct_weight` 替换基础奖励。失败组判定不使用方差阈值。普通模式的判定使用 `task_reward`，不计入 DAPO 长度惩罚；LASER-D 使用含 bonus 的 reward（失败样本无 bonus）。缺少任一样本的失败阶段分则保持原值。替换同步更新 `task_reward`、`sample.reward` 和组件贡献，保留长度惩罚且不重复扣分；随后只执行一次正式过滤，再计算 return。替换后仍低方差或样本数不足的组不会被强制放行。

过滤器只读已结算结果：LASER-D 使用包含长度 bonus 的 `sample.reward`；其它模式优先使用 `metadata.task_reward`，字段不存在时才回退到 `sample.reward`。训练侧的 `reward_post_process_by_group` 不再调权或加减分，只从写回的单 turn reward 计算 return/advantage，避免过滤后重算 gate。完整 group 后处理跳过无效样本。两项长度/调权开关均未启用也不影响独立的失败评分或 CTM。

group 正确率分档默认输出到普通 rollout 日志和 TensorBoard/W&B，不依赖动态调权或 `--log-exp-metrics`。`rollout/group_correct_rate_{low,medium,high}_count` 和 `rollout/group_correct_rate_{low,medium,high}_fraction` 分别记录 `rate < 1/3`、`1/3 <= rate <= 2/3`、`rate > 2/3` 三档的 group 数和比例。每个保留的 `(group_index, turn_idx)` 只计一次，使用过滤前的 `metadata.group_correct_rate`；排除 remove、aborted、pad 和非 kernel 样本。阈值固定，不随 `--difficulty-thresholds` 改变。`rollout/group_correct_rate_count` 是比例分母；缺少正确率或值不在有限的 `[0, 1]` 范围内的组不计入，没有符合条件的组时不输出这些指标。

任一动态分量开启时，训练数据转换在最终 reward 后处理后输出 `reward post-process` 日志，并通过 TensorBoard/W&B 记录 `rollout/dynamic_reward/*`，不需要 `--log-exp-metrics`：

- `{correctness,speedup,coverage}_{weight,offset}_{mean,min,max,p25,p50,p75}`：按 group 汇总最终权重及有符号偏移，仅记录已开启分量。
- `{correctness,speedup,coverage}_scale_{mean,min,max,p25,p50,p75}`：按 `(group_index, turn_idx)` 去重的缩放系数统计，每组计一次。
- `{correctness,speedup,coverage}_scale_{zero,scaled,one,boosted}_fraction`：系数为 0、在 0 和 1 之间、等于 1、大于 1 的组比例。
- `{correctness,speedup,coverage}_reward_delta_{mean,min,max,p25,p50,p75}`：逐样本相对未调权基础贡献的变化，包含未发生变化的零值。speedup delta 是 performance 奖励的变化，不是实测加速比变化。
- `group_count`、`sample_count`：有最终记录的有效组数和样本数，排除 remove、aborted、pad 和非 kernel 样本。

仅开启的分量输出动态指标。最终逐样本记录位于 `metadata.dynamic_reward`，过滤前生成，训练数据转换只汇总保留样本。原统一 `gate_*` 指标改为三个分量各自的 scale 指标。普通 `rollout/group_correct_rate_*` 指标仍默认记录，不受动态调权开关影响。生成阶段的早期日志可能早于完整 group 调权。

#### LASER-D 自适应长度奖励

`--overlong-penalty` 可选 `None`（默认关闭）、`dapo`（现有线性扣分）、`laser-d`。LASER-D 在这个统一入口下是**加分**：正确且完整 response token 数不超过预算时，加一个固定 bonus；错误或超预算不加分，也不额外扣分。不改变生成长度上限，不往 prompt 注入预算，不加入 speedup 系数。参见[论文 Table 2 / 第 5 节](https://arxiv.org/html/2505.15612v1#S5)和[官方奖励实现](https://github.com/hkust-nlp/Laser/blob/main/verl/utils/reward_score/length_penalty.py)。

```bash
--custom-reward-post-process-path examples.kernel_agent.kernel_reward.reward_post_process_by_group \
--overlong-penalty laser-d \
--laser-d-length-score 0.5 \
--laser-d-min-length 1024 \
--laser-d-length-interval 1024 \
--laser-d-update-interval 20 \
--laser-d-monitor-groups 500
```

上面是各参数默认值；预算搜索上限直接取 `--rollout-max-response-len`，下限不得超过上限。DAPO 的 `--overlong-buffer-len`、`--overlong-penalty-factor`、`--overlong-use-effective-response-cap` 不参与 LASER-D。三个动态 reward 入口独立控制 correctness、speedup、coverage 权重，不缩放长度 bonus。

按同题、同 turn 的有效样本计算正确率 `s`，复用 `--difficulty-thresholds`（必须恰好两个，默认 `1/3 2/3`）：`s < h` 为 hard，`h <= s < e` 为 medium，`s >= e` 为 easy。监测和奖励使用同一套边界。移除、aborted、pad 轨迹不参与监测或 LASER-D 加分。

预算计算（普通文本公式）：

```text
K = 当前有效 group size
C_min = 此难度桶在 K 下可达到的最小正整数正确数
        默认阈值、K=8:  hard=1, medium=3, easy=6
        默认阈值、K=16: hard=1, medium=6, easy=11
coverage_g(L) = group g 中 response_length <= L 的比例（包含错误回答）
ECR_bucket(L) = mean_g[C_min(g) * coverage_g(L)]
B_bucket = 网格中使 ECR >= 1 的最小 L
           缺少观测或未找到时使用 response 上限
reward = task_reward + bonus * I(correct and response_length <= B_bucket)
```

固定 K 时等价于论文的 `C_min * coverage`；多轮导致 K 不等时，按 group 等权平均其估计值。网格总是包含上限，即使上限不能整除步长。ECR 是长度覆盖代理，不保证实际保留下来的回答一定正确。本实现采用论文中的最小可达正确数，而非官方代码可选的 `floor` 近似。

**与论文的区别及异步语义：**按当前训练方案，不新增独立 monitoring rollout，而是复用动态过滤前的训练 group。一个有界 reservoir 在更新窗口中最多保留 `monitor-groups` 个 prompt/turn group 的长度列表，不保存 response 文本；被动态过滤的组也参与，warm queue 中尚未取出的组不提前统计。多轮每个 turn 是一次观测。初始各桶预算为 `min-length`；首个有监测数据的 rollout 完成后更新一次，之后每隔 `update-interval` 个 rollout 批次更新，**不是 actor optimizer step**。窗口更新后清空监测池。

每批 collector 固定使用进入该批时的预算快照，打标后才过滤；本批监测只更新后续批次的预算。加分通过现有 `post_process_rollout_rewards` 写回，`reward_component.length` 记录正贡献，`task_reward` 不包含它。重复后处理不重复加分；TRLOO 仅在当前 turn 保留该 bonus，不计入未来折算项。

LASER-D 的低方差过滤使用包含 bonus 的 reward，保留“全部正确，但短回答有加分”的有效组；DAPO 仍按原来的 pre-penalty `task_reward` 过滤。

普通 SGLang rollout 在参数校验时自动接入 LASER-D collector；kernel-agent fully-async 直接支持。要求保持默认启用的数据源管理（不要传 `--disable-rollout-global-dataset`），并使用上面的 reward hook；其它自定义 rollout 会启动报错，防止静默漏接。预算、监测池、更新位置、reservoir RNG 状态随数据源 `metadata["laser_d"]` 保存在现有 `rollout/global_dataset_state_dict_<rollout_id>.pt` 中；用对应 checkpoint 恢复即可，预算/监测参数不匹配会报错。没有完成的批次不会提交新的预算状态。

TensorBoard/W&B 沿用现有通道：`rollout/laser_d/{hard,medium,easy}/budget_used`、`budget_next`、`budget_updated`、`observed_groups`、`monitor_groups_retained/seen` 记录收集/更新；最终训练样本记录 `length_score_mean`、`bonus_fraction`、`sample_count` 及各桶 `budget_mean`/`sample_count`。逐样本预算和 bonus 在 `metadata.laser_d`。这些指标不需要 `--log-exp-metrics`。

#### GRPO 算法

GRPO（Group Relative Policy Optimization）是 DeepSeek-Math 中提出的一种 RL 算法，其核心思想是通过组内相对比较来计算 advantage，而不需要额外的 critic 模型。

使用 GRPO 时，需要设置：

```bash
--advantage-estimator grpo
```

GRPO 的主要特点：

- **无需 Critic 模型**：GRPO 通过对同一 prompt 采样多个 response，然后在组内计算相对 reward 来估计 advantage，避免了训练和维护 critic 模型的开销；
- **资源高效**：由于不需要 critic 模型，GPU 资源可以完全用于 actor 训练和推理；
- **简单易用**：配置简单，只需要设置 `--advantage-estimator grpo` 即可。

相关参数：

- `--n-samples-per-prompt`：每个 prompt 采样的 response 数量，用于组内比较；
- `--normalize-advantages`：是否对 advantage 进行归一化；
- `--use-conditional-truncation-mask`：Kernel Agent 在 rollout reward 后处理时只记录 CTM 抽样标记；Megatron 训练端在 OPD 和 advantage 归一化之后，才将选中样本的 advantage 置零。未启用归一化时仍在 advantage 计算末尾置零。原 reward、returns、loss mask 和归一化统计保持不变；runtime error 及之后的执行阶段错误不参与 CTM 屏蔽。
- `--eps-clip`：PPO 风格的 clip 范围。

#### PPO 算法

PPO（Proximal Policy Optimization）是经典的 RL 算法，使用 critic 模型来估计 value function，从而计算 advantage。

使用 PPO 时，需要设置：

```bash
--advantage-estimator ppo
```

**注意：当前 PPO 下 Critic 和 Actor 共享同一组训练 GPU**，资源分配时不需要为 critic 额外预留一组独立 GPU。具体来说：

- PPO 会创建 actor 和 critic 两套训练进程组，但它们会被放到同一组 train placement group 上；
- critic 的训练规模跟随 actor 配置，当前 actor / critic 的 Megatron 并行拓扑必须保持一致；
- PPO 会强制开启 train 侧 offload，使 actor 和 critic 在同一批 GPU 上轮流唤醒和释放显存；
- 当前没有单独配置 critic 训练资源的 CLI 参数，critic 的节点数和每节点 GPU 数会由 actor 配置派生。


PPO 相关参数：

- `--megatron-config-path`：通过 YAML 对 actor / critic 分别覆盖 Megatron 参数，例如为 critic 单独设置 `load`、`save`、`lr` 或 warmup 参数；
- `--num-critic-only-steps`：训练开始时只训练 critic 的步数；
- `--eps-clip`：PPO clip 范围；
- `--value-clip`：value loss 的 clip 范围；
- `--kl-coef`：KL penalty 系数，用于 reward shaping。

### 高级 Megatron 配置（--megatron-config-path）

对于 PPO 场景，可以使用 `--megatron-config-path` 指定一个 YAML 文件，对 actor / critic 分别覆盖 Megatron 参数。常见用途包括给 critic 设置不同的 `lr`，或者分别指定 `load` / `save` 等路径。

```yaml
megatron:
  - name: default
    role: actor
    overrides:
      lr: 1e-6
  - name: default
    role: critic
    overrides:
      lr: 1e-5
```

> **注意：** 当前该配置只支持 PPO；并且当前 PPO 下 actor 和 critic 的 Megatron 并行配置必须保持一致。建议把并行相关参数继续写在公共 CLI 中，只把角色差异项放在 YAML 里。详见 [Megatron Config：按角色覆盖训练参数](../advanced/megatron-config.md)。

## 自定义 rollout 函数

slime 支持不同程度的自定义数据生成（rollout）。

- 默认会使用 [slime/rollout/sglang_rollout.py](https://github.com/THUDM/slime/blob/main/slime/rollout/sglang_rollout.py) 中的 `generate_rollout` 函数进行数据生成。这个文件中实现了基于 sglang 的异步（asyncio）数据生成流程，并支持了例如 dynamic sampling，partial rollout 等功能；

- 可以通过 `--rollout-function-path` 参数，完全替换 sglang_rollout.py 中的 `generate_rollout`，只需要保证 `--rollout-function-path` 传入的函数签名满足：

  ```python
  def generate_rollout(args, rollout_id, data_source, evaluation=False) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
      """
      Args:
          args: the whole args
          rollout_id: int, the id of the rollout, used for deterministic data generation
          data_source: the data source to get and store samples
          evaluation: bool, whether the rollout is for evaluation or not
      
      Returns:
          RolloutFnTrainOutput | RolloutFnEvalOutput: the output of the rollout
      """
          ...
          return output
  ```

  其中：

  -  `args` 为整个 slime 运行使用的 args；
  - `rollout_id` 对应的是当前是第几次数据生成，用作保证续训时的数据顺序；
  - `data_source` 是 slime 中全局唯一的数据源，可以用来获取初始 prompt，数据 id，将生成至一半的 sample 存储下来下次留作下次使用等；
  - `evaluation` 是否是当做 evaluation 使用。可以通过 `--eval-function-path` 单独配置 eval 的函数；
  -  返回的 `Sample` 类型见 [slime/utils/types.py](https://github.com/THUDM/slime/blob/main/slime/utils/types.py)，在实现时，需要保证
     -   `tokens`：prompt + response 的 token；
     -  `response_length`：response 的总长。对于多轮任务，则是除去第一轮 prompt，剩余的 token 长度；
     -  `reward`：这条数据的 reward；
     -  `status`：这条数据的状态（如 `Sample.Status.COMPLETED`、`Sample.Status.TRUNCATED`、`Sample.Status.ABORTED`、`Sample.Status.FAILED`）。
     
     这几个参数被正确配置了。以及如果有工具调用或者多轮使用等场景，确保 `loss_mask` 是正确的：
     
     - `loss_mask` 应该和 `response_length` 一样长，其中需要算 loss 的 token 为 1，mask 掉的为 0
  
- 在一些情况下，可能只需要替换数据生成的逻辑，那么使用 `--custom-generate-function-path` 进行替换即可，这个函数一个简化版实现如下：

  ```python
  async def generate(args, sample: Sample, sampling_params) -> Sample:
      global TOKENIZER
      if TOKENIZER is None:
          TOKENIZER = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
  
      # send request to router
      output = await post(
          f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate",
          {
              "text": sample.prompt,
              "sampling_params": sampling_params,
          }
      )
  
      prompt_tokens_ids = TOKENIZER(sample.prompt, add_special_tokens=False)["input_ids"]
      response_token_ids = TOKENIZER(output["text"], add_special_tokens=False)["input_ids"]
  
      # set sample
      sample.tokens = prompt_tokens_ids + response_token_ids
      sample.response_length = len(response_token_ids)
      finish_reason = output["meta_info"]["finish_reason"]["type"]
      if finish_reason == "length":
          sample.status = Sample.Status.TRUNCATED
      elif finish_reason == "abort":
          sample.status = Sample.Status.ABORTED
      else:
          sample.status = Sample.Status.COMPLETED
      sample.response = output["text"]
  
      return sample
  ```

   更完备的版本请查看 [slime/rollout/sglang_rollout.py](https://github.com/THUDM/slime/blob/main/slime/rollout/sglang_rollout.py)。

- 有的时候，我们还需要支持自定义的 reward model，可以通过配置 `--custom-rm-path` 来进行配置。

## sglang 使用方法

slime 通过 `HttpServerEngineAdapter` 作为中介，实现了基于 sglang 的 server based engine。

### 参数配置

slime 通过引入 sglang 的 `ServerArgs.add_cli_args`，从而引入了几乎所有的 sglang 参数，在设置一个 sglang 参数的时候，需要在参数前加上 `--sglang` 的前缀，例如：

- 在训推一体的训练时，往往需要限制 `--mem-fraction-static`，这个参数需要转变为 `--sglang-mem-fraction-static`；
- 在训练中，希望 sglang 能推理超过 huggingface checkpoint 的 `config.json` 中标识的最长 context length，需要使用 `--context-length`，那么在 slime 中需要使用 `--sglang-context-length`；
- 在进行多机大 ep 推理的时候，需要 `--ep-size`、`--enable-dp-attention`、`--dp-size`、`--moe-a2a-backend deepep` 等，则可以对应地传入 `--sglang-ep-size`、`--sglang-enable-dp-attention`、`--sglang-dp-size`、`--sglang-moe-a2a-backend deepep` 。

有部分参与和 slime 的资源调度相关，会由 slime 自行配置，例如：

- `--tp-size` 在 slime 中会使用 `--rollout-num-gpus-per-engine`
- `--model-path` 在 slime 中会使用 `--hf-checkpoint`

sglang 参数引入 slime 的方式可以参考 [slime/backends/sglang_utils/arguments.py](https://github.com/THUDM/slime/blob/main/slime/backends/sglang_utils/arguments.py)。

### router 使用方法

slime 会用 [sglang-router](https://github.com/sgl-project/sglang/tree/main/sgl-router) 来管理训练过程中的 sglang server。可以通过 `--sglang-router-ip` 与 `--sglang-router-port` 来配置 [sglang-router](https://github.com/sgl-project/sglang/tree/main/sgl-router) 的地址。如果不进行配置，则会在集群中默认启动一个 router。

所有的 sglang server 在启动后，会通过 `/add_worker` 申请加入 router。在实际进行数据生成的时候，只需要向 router 发送 http 请求，router 会进行 load balancing 操作，将请求转发给 server 们。

当通过 `--sglang-router-ip` 与 `--sglang-router-port` 来配置传入一个外部的 router，此时 slime 不再会在内部启动一个 router，而是会把所有的 server 都注册在这个外部 router 上。这时可以利用这个外部的 router 地址来实现更复杂的数据生成流程。注意 router 是支持 openai compatible api 的。

### 高级引擎配置（--sglang-config）

对于高级部署场景，可以使用 `--sglang-config` 指定一个 YAML 文件，来配置服务器组、多模型部署以及选择性权重更新。

**多模型部署**允许同时服务多个模型（例如一个接收权重更新的 actor 模型和一个冻结的 reference/reward 模型）：

```yaml
sglang:
  - name: actor
    update_weights: true          # 接收训练的权重更新（默认）
    server_groups:
      - worker_type: regular
        num_gpus: 8
        num_gpus_per_engine: 4
  - name: ref
    model_path: /path/to/ref_model
    update_weights: false          # 冻结，不更新权重
    server_groups:
      - worker_type: regular
        num_gpus: 4
        num_gpus_per_engine: 2
```

每个模型都有自己独立的 router。每个模型的 router 信息可通过 `args.sglang_model_routers`（一个将模型名映射到 `(ip, port)` 元组的字典）访问。自定义 rollout 函数可以使用 `slime.rollout.sglang_rollout` 中的 `get_model_url(args, "ref")` 来将请求路由到指定模型。

**服务器组功能：**
- `worker_type`：`regular`、`prefill`、`decode` 或 `placeholder`（预留 GPU 位置但不创建引擎）
- `overrides`：SGLang `ServerArgs` 字段覆盖字典，会叠加在 `--sglang-*` CLI 参数之上
- `num_gpus_per_engine`：每组的 TP 大小覆盖

## megatron 使用方法

slime 通过复用 `megatron.training` 目录下的常规函数，如 `parse_args`， `save_checkpoint`，`load_checkpoint`，从而实现对不同版本以及轻度魔改的 megatron 的支持。所以在使用时，需要保证 `PYTHONPATH` 中能访问到 megatron，例如在运行时加入 `export PYTHONPATH=/root/Megatron-LM`。

### 参数配置

slime 通过直接引入 `from megatron.training.arguments import parse_args` 引入了当前环境中 megatron 的所有参数。如果当前使用的 megatron 有在 `parse_args` 之外的参数，可以通过像 [train.py](https://github.com/THUDM/slime/blob/main/train.py) 中传入参数来进行配置，例如：

```python
if __name__ == "__main__":
    try:
        from pretrain_gpt import extra_args_provider
    except:
        extra_args_provider = None
    args = parse_args(extra_args_provider)
    train(args)
```

### 自定义参数

在一些定制版 megatron 的实现中，需要在初始化，或者训练步的前后进行特殊的操作。目前我们加入如下的插件：

- `--custom-megatron-init-path`：会增加一些 init 的调用；
- `--custom-megatron-before-log-prob-hook-path`：会在计算 log prob 之前调用；
- `--custom-megatron-before-train-step-hook-path`：会在每个训练步之前调用。可以考虑用这种方式混入特殊的训练 loss 之类的。


### 多个 KernelENV 服务

单个服务继续使用 `--kernel-env-url URL`。多个服务使用互斥参数：

```bash
--kernel-env-urls http://host-a:20111 http://host-b:20111
```

Qwen3.8 WarmUp 和 MultiTurn 脚本支持 `KERNEL_ENV_URLS`，使用空格或逗号分隔；设置后优先于原来的 `KERNEL_ENV_URL`。例如：

```bash
KERNEL_ENV_URLS="http://host-a:20111,http://host-b:20111" \
bash examples/kernel_agent/run_qwen3.8_27B_WarmUp.sh
```

启动脚本会一次传入全部 URL，并发检查各服务的 `/health`；全部健康后，多 URL 场景校验所有 GPU 的型号和总显存一致。设备编号、GPU 数量和实时负载可以不同；GPU 信息缺失或不可用则失败。只有所有服务都提供有效 `kernelgym_version` 时，才比较完整版本信息；任意服务未返回版本信息时，输出 `SKIP` 并跳过版本比较。单 URL 保持原来的健康检查行为。每个 rollout 进程对每组 URL 独立维护轮询位置，首次使用时随机选择起点，后续新任务按配置顺序循环分配，不受任务完成顺序影响。提交、重试、状态查询与取消固定使用选中的 URL。样本 metadata 的 `kernel_eval_server_url` 和 `event=assigned` 日志记录所选服务。

轮询均分的是任务数量，不读取 ENV 实时负载，也不保证耗时不同的任务均衡占用 GPU；多个进程的轮询位置独立。`kernel_eval_rate_limit` 限制同一 Ray namespace 下同一 URL 的并发提交请求，同 URL 的 worker 共享限流器；不同 URL 独立限流。该限额不等同于服务端 GPU 执行并发数，worker 的任务并发上限也独立生效。初始轮询位置使用独立随机源，不消耗 coverage 筛选使用的全局随机数状态。运行中沿用原有单服务重试和超时处理，不将已提交的任务自动转发到其他服务。建议服务使用相同 GPU 型号、KernelGym 版本和评测配置，以保持速度奖励可比。


也可以直接运行健康检查：

```bash
python scripts/check_kernelgym_health.py \
  --urls http://host-a:20111 http://host-b:20111 \
  --attempts 3 --timeout 5
```

原来的 `--url URL` 用法保持兼容；也可通过 `KERNELGYM_URLS` 配置逗号或空格分隔的 URL。显式命令行参数优先于环境变量。GPU 或版本不一致时退出码为 1，训练脚本随即退出。
