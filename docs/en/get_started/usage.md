# Usage Guide


## Introduction to slime Parameters

When using slime, parameters are primarily passed for the following purposes:

1.  To allocate a portion of the GPUs in the cluster for training and another portion for inference.
2.  To load Megatron for the training portion.
3.  To load SGLang for the inference portion.
4.  To configure the hyperparameters required for RL training.

Following this order, we need to configure these parameters:

### Cluster Resource Allocation

There are four main parameters for cluster resource allocation:

  - `--actor-num-nodes`: The number of nodes required for RL actor training.
  - `--actor-num-gpus-per-node`: The number of GPUs per node for RL actor training.
  - `--rollout-num-gpus`: The total number of GPUs required for rollout (inference). Set it to `0` to still parse SGLang arguments and launch the router without launching local SGLang servers.
  - `--rollout-num-gpus-per-engine`: The number of GPUs per inference engine. This parameter is similar to SGLang's `tp_size`. When performing multi-node serving, this value should be the total number of GPUs. For example, if serving one model with 2 nodes and 16 GPUs, this value should be 16.
    The reason for not using a parameter like `--sglang-tp-size` is that we might consider supporting SGLang's `dp_size` parameter in the future, which means an engine could contain multiple SGLang servers (currently, only `--sglang-dp-size` under the `--sglang-enable-dp-attention` condition is supported).

With the default configuration, we use these parameters to allocate `actor_num_nodes * actor_num_gpus_per_node` GPUs for training and `rollout_num_gpus` GPUs for inference via Ray, thus achieving a separation of training and inference resources.

For co-located training and inference, you also need to configure:

  - `--colocate`: Enables co-located training and inference. By default, this makes the number of GPUs for training and inference equal. You can explicitly set a different positive `--rollout-num-gpus`, for example to use more rollout GPUs than actor GPUs; the extra GPUs are used as rollout-only resources. If `--rollout-num-gpus 0` is set explicitly, slime launches only the router and no local SGLang servers.

Additionally, slime supports Prefill and Decode disaggregation (PD Disaggregation). You can set the number of servers used for Prefill by setting the `--prefill-num-servers` argument.

### Choosing Training Backend

slime currently uses Megatron-LM as its training backend. The compatibility option
`--train-backend megatron` may still be supplied explicitly.

### Loading Megatron

Unlike tools such as SGLang, vLLM, or Hugging Face Trainer, Megatron cannot directly read Hugging Face checkpoints. Instead, the user must configure the parameters for the model to be trained and load Megatron's own checkpoint format.

Generally, we need to perform three preparatory steps:

  - Configure model parameters.
  - Configure parallelism and other optimizations.
  - Configure the checkpoint to be loaded.

For details on some of Megatron's customizations and the principles behind how slime incorporates Megatron, please see the "How to Use Megatron" section.

#### Configuring Model Parameters

Taking qwen3 4B as an example, we need these parameters:

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

We provide configurations for common models in [scripts/models](../../../scripts/models), which you can reuse directly. If you are also using Megatron for pre-training/SFT, you can directly reuse the model configurations from your pre-training/SFT setup.

Note:

  - slime will load all parameters of Megatron found in the `PYTHONPATH`, so you can find parameters and their descriptions within the Megatron in your environment.
  - slime uses data packing (also known as varlen or thd) for training. There is no need to configure `--seq-length` or `--max-positional-embedding`, as these parameters do not affect the maximum context length of the trained model.

#### Setting Up Parallelism and Recomputation

Megatron is currently the most comprehensively optimized training framework. A major reason for using Megatron is to pursue its excellent performance. Here is a brief introduction to configuring Megatron's parallelism and recomputation.

  - Here we list Megatron's parallelism strategies. For a more detailed discussion on the trade-offs between these strategies, please refer to more specialized discussions:
      - `--tensor-model-parallel-size`: TP
      - `--sequence-parallel`: Megatron's SP is an optimization for TP. It is recommended to always enable SP when using TP.
      - `--pipeline-model-parallel-size`: PP
      - `--context-parallel-size`: Megatron's CP, also known as sequence parallelism, generally corresponds to ring attention.
      - `--expert-model-parallel-size`: EP for MoE, where each GPU has `num_experts / ep_size` experts.
      - `--expert-tensor-parallel-size`: Megatron supports using a different `tp_size` for the MoE experts than for other parts of the model, which we generally call ETP.
  - For recomputation, the following flags are commonly configured in Megatron:
      - `--recompute-granularity`: This can be set to `full` or `selective`. `full` means complete recomputation, while `selective` recomputes less. If not configured, no recomputation is done.
      - `--recompute-method`: `uniform` is generally sufficient.
      - `--recompute-num-layers`: The number of layers per group for recomputation. A value of 1 is usually fine.

#### Loading Megatron Checkpoints

Megatron supports several of its custom checkpoint formats. Here are two of the more common ones:

  - The once mainstream `torch` format (corresponding to `--ckpt-format torch`).
  - The currently recommended `torch_dist` format (corresponding to `--ckpt-format torch_dist`).

The `torch` format is Megatron's older storage format. Its structure consists of directories like `mp_rank_xxx`, where each directory corresponds to the checkpoint stored by each rank under a specific parallel partitioning. Because of this, when loading a `torch` format checkpoint, you must ensure that the checkpoint's parallelism strategy matches that of the training task.

We recommend using the `torch_dist` format because it supports automatic parallel sharding, meaning that training tasks with different parallelism settings can share the same checkpoint, which is much more convenient. `torch_dist` is also the default format in the open-source Megatron. A `torch_dist` format checkpoint typically contains a set of `.distcp` files. When using `torch_dist`, you can convert from Hugging Face to `torch_dist` and vice versa using the checkpoint conversion method described in the [README](../../../README.md).

In terms of storage structure, a Megatron checkpoint typically looks like this, assuming the storage path is `/ckpt/`:

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

The `latest_checkpointed_iteration.txt` file records the latest training step. When loading a model, you should not directly pass `/ckpt/iter_xxxxxxx`, but rather pass `/ckpt/` and use `--ckpt-step` to select the corresponding training step (if `--ckpt-step` is not used, the step will be read from `latest_checkpointed_iteration.txt`).

When using slime, there are three parameters for loading and saving checkpoints:

  - `--ref-load`: The Megatron checkpoint for the reference model.
  - `--load`: The Megatron checkpoint for the actor. If `--load` is not set, or if the specified directory does not exist or does not contain `latest_checkpointed_iteration.txt`, the actor will be initialized from the `--ref-load` checkpoint.
  - `--save`: The path where the actor's checkpoints are saved.

Note:

  - Regardless of the checkpoint storage method (i.e., however `--ckpt-format` is set), Megatron can load both `torch` and `torch_dist` formats.

### Loading SGLang

Loading SGLang is very simple. You only need:

  - `--hf-checkpoint`: The Hugging Face checkpoint used to initialize SGLang.

Note:

  - Before the first training step, slime will synchronize the parameters from Megatron to SGLang. Therefore, the `--hf-checkpoint` does not need to contain the latest training parameters, and you do not need to change the HF checkpoint when resuming training.
  - By default, SGLang reads the maximum context length from the `config.json` in the Hugging Face checkpoint. You can use the `--sglang-context-length` parameter to override this value to support longer inference.
  - During co-located training and inference, although Megatron and SGLang will offload sequentially, they still need to leave some memory for each other. You need to adjust SGLang's total VRAM usage by reducing `--sglang-mem-fraction-static`.
  - slime supports passing through sgl-router parameters by adding a `router` prefix to the original parameter name. For example, sgl-router's `--balance-abs-threshold` parameter should be set as `--router-balance-abs-threshold`. Since sgl-router uses cache-aware routing by default, it may cause uneven request distribution. You can set `--router-balance-abs-threshold 0` to force balanced distribution, but this may affect prefix cache hit rate in multi-turn conversation scenarios.
  - If SGLang engines are pre-launched by an external system, connect to them with `--rollout-external-engine-addrs host1:port host2:port`. When the trainer and engines cannot form an NCCL weight-update group, use `--update-weight-mode full --update-weight-transport disk --update-weight-disk-dir /shared/fs/updates`; slime writes a complete HF checkpoint and asks SGLang to hot-load it through `update_weights_from_disk`. For large models or cross-cluster deployments, use `--update-weight-mode delta --update-weight-transport disk` instead. See [External Rollout Engines Roadmap](../advanced/external-rollout-engines.md) and [Delta Weight Sync](../advanced/delta-weight-sync.md).

For details on some of SGLang's customizations and the principles behind how slime incorporates SGLang, please see the "How to Use SGLang" section.

### Data Format

slime supports `.jsonl` and `.parquet` files; reading Parquet requires `pyarrow`. Each record in either format should contain the fields selected by `--input-key` and `--label-key`. An expanded JSONL record looks like this:

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

This corresponds to the following configuration:

```bash
  --input-key prompt
  --label-key label
  --apply-chat-template
```

Please note that the `step_loss_mask` (default=1) here is for SFT phase. If it is set to 0, the turn will not contibute to the final loss; if it is set to 1, slime will use the normal `loss_mask`.
Additionally, we provide a `metadata_key`, which defaults to `"metadata"`. When read, slime will load the metadata from the data, which can be helpful for custom data generation or creating custom reward models.

If one run mixes multiple data sources, put `source_name` in the sample metadata:

```json
{
  "prompt": "...",
  "label": "...",
  "metadata": {
    "source_name": "math"
  }
}
```

The recommended contract is to put the source identifier in `metadata["source_name"]`; slime also recognizes a dynamically set `sample.source` from custom data sources. When rollout samples are converted to training data, slime carries one `source_names` entry per sample to the training side. The source lookup order is dynamic `sample.source`, then `metadata["source_name"]`; if neither is set, the source is `"unknown"`. This is useful for custom rewards, filters, logging, and future per-source routing such as OPD teacher selection.

### Hyperparameters for RL Training

- `--advantage-estimator`: Specifies the RL algorithm for the training process. Currently supported algorithms include:
    - `grpo` ([https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300))
    - `gspo` ([https://arxiv.org/abs/2507.18071](https://arxiv.org/abs/2507.18071))
    - `cispo` ([https://arxiv.org/abs/2506.13585](https://arxiv.org/abs/2506.13585))
    - `reinforce_plus_plus` and `reinforce_plus_plus_baseline` ([https://arxiv.org/abs/2501.03262](https://arxiv.org/abs/2501.03262))
    - `ppo` ([https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347))

  Note: On-policy distillation (OPD) is now orthogonal to the advantage estimator. Use `--use-opd` and `--opd-kl-coef` to enable OPD on top of any estimator.
- `--calculate-per-token-loss`: By default, slime calculates loss on a per-sample basis, i.e., `mean(sum(sample_i) / len(sample_i))`. Enable this flag to calculate loss on a per-token basis, i.e., `sum(sum(sample_i)) / sum(len(sample_i))`.
- `--use-tis`: Enable this setting to use TIS (Truncated Importance Sampling) (https://fengyao.notion.site/off-policy-rl).

`--calculate-token-sum-loss` uses the PG reduction from [MiniRL Eq. (7)](https://arxiv.org/html/2512.01374v1#S4.SS1): `sum(valid token PG losses) / rollouts in the training step`, without dividing by actual response length. For multi-turn rollouts, all turns contribute to one trajectory sum and the outer average counts trajectories. Padding and masked tokens contribute no loss.

This flag defaults to off, requires `--loss-type policy_loss`, and is incompatible with `--calculate-per-token-loss` and `--custom-pg-loss-reducer-function-path`. It does not change advantages, importance sampling, clipping, entropy/KL reductions, or other diagnostic reductions. For GRPO with mean-only reward centering, separately pass `--disable-grpo-std-normalization`. This is not a switch for the complete MiniRL algorithm.

Removing length division substantially increases the PG gradient scale: recheck learning rate, gradient norms and gradient clipping when switching. KernelAgent's `run_qwen3.6_27B_full_async_dppo.sh` accepts `CALC_LOSS_MODE=TokenSum`; its default remains `PerToken`.

`--calculate-per-prompt-loss` enables prompt-mean PG aggregation, following the approach in
[slime PR #2090](https://github.com/THUDM/slime/pull/2090): pool valid token losses across all
trajectories and turns sharing `Sample.group_index`, divide by that prompt's valid token count,
then average nonempty prompts within each optimizer step. It defaults to off and requires Megatron
with `--loss-type policy_loss`. It is mutually exclusive with per-token loss, token-sum loss, and
custom PG reducers. Rewards, returns, advantages, and entropy/KL reductions are unchanged.

Denominators are computed before DP/microbatch slicing and are not recomputed after TIS/RS rejection.
The step scale uses the actual rollout count divided by the number of nonempty prompts, supporting
uneven groups. Fully masked prompts are excluded from that prompt count; an entirely masked step
fails explicitly. Prompts may span DP ranks/microbatches, but spanning optimizer steps or partially
dropping a prompt at the batch tail raises an error. Keep groups contiguous and choose an appropriate
global batch size (a multiple of group size for fixed-size groups). Custom train-data converters must
provide `group_indices`. The KernelAgent Qwen launch script accepts `CALC_LOSS_MODE=PerPrompt`;
its default remains `PerToken`.

#### ArgMaxRL advantages

Use `--advantage-estimator argmaxrl`, optionally with `--argmaxrl-reward-offset 1` if the known
aggregate-reward lower bound is -1. The offset defaults to zero and is used only in weight calculation;
it never rewrites sample rewards, components, or task rewards. Negative/nonfinite
shifted rewards fail explicitly: there is no observed-group-minimum shift or zero clipping.

Following [ArgMaxRL](https://www.doubleai.com/research/argmaxrl-generalizing-maxrl-to-continuous-rewards),
sort nonnegative rewards descending and set `r[N+1]=0`. Compute
`w[j] = sum((r[m]-r[m+1])/m, m=j..N)` and `A[j]=N*w[j]`.
Here N counts valid candidates in the actual group. Scaling by N adapts the usual candidate-mean
convention: `mean(A*score)=sum(w*score)`. Binary rewards give N/K for successes and zero for failures.

Complete-group advantages are computed in `RolloutManager._post_process_rewards` before DP splitting,
then broadcast to local tokens by the training backend. Group by `Sample.group_index`, or
`(group_index, turn_idx)` for multi-turn kernel training. Use the settled aggregate reward, including
enabled shaping, without separate correctness/performance/coverage objectives or TRLOO future-return
folding. Removed/fully masked samples receive zero and are excluded. Fan-out segments sharing a
trajectory in one group count once and must carry identical rewards. Custom reward hooks must provide
uncentered aggregate scores; ArgMaxRL is applied once afterwards.

No group centering or standard-deviation normalization is applied; the GRPO std flag is irrelevant.
`--normalize-advantages`, custom advantage functions, and reward KL (`--kl-coef`) are rejected. Independent KL loss (`--use-kl-loss`) remains available.

Loss reduction, PPO/DPPO clipping, filtering, CTM, and OPD remain independently configured. Thus the
combined trainer is not necessarily the article's unbiased REINFORCE estimator. TokenSum is closest
to its sequence log-prob sum; PerSample/PerToken/PerPrompt introduce their own length normalization.
Equal positive rewards still produce nonzero advantages, although existing low-variance filtering
may discard these groups. Group-dependent reward shaping is also an additional variant.

#### TailRL advantages

Use `--advantage-estimator tailrl`. This reuses ArgMaxRL's group/weight pipeline and adds
centering over unique valid candidates: `A=N*(w-mean(w))`, without std normalization. It matches
the [official code optimization implementation](https://github.com/Zanette-Labs/TailRL/blob/5682c6ac03387355e017ce966693266bb148fa10/experiments/code_optimization/code_opt/advantages.py#L22),
including its N factor, rather than the website's simplified unscaled pseudocode.

For rewards `[0,1,3]`, ArgMaxRL gives `[0,1.5,7.5]`; TailRL gives `[-3,-1.5,4.5]`.
Equal-reward and singleton groups yield zero. Binary rewards give N/K-1 for successes and -1
for failures when K>0; all-zero groups yield zero. Padding and removed samples do not enter
the mean; repeated fan-out segments count as one candidate and share its advantage.

TailRL is invariant to a common reward shift and accepts finite signed rewards. It rejects
nonzero `--argmaxrl-reward-offset`. Internally subtracting the group minimum before computing
nonnegative weights is algebraically equivalent after centering and improves numerical stability;
this is never applied to uncentered ArgMaxRL. Sample rewards/components remain unchanged.

Group by prompt/turn and center before DP slicing, without TRLOO future-return folding.
The same restriction on whitening applies as for ArgMaxRL.
Other loss/filter/CTM/OPD options remain independent. Group centering is not batch whitening;
its sample-dependent baseline should not be conflated with the original uncentered ArgMaxRL's
finite-sample unbiased estimator.

#### Kernel-agent context budget nudge

Kernel-agent multi-turn rollouts optionally support `--use-context-budget-nudge 0.2`, inspired by [Mercor's context nudge](https://www.mercor.com/blog/training-frontier-knowledge-work-agents-a-397b-rl-training-guide-with-skyrl/). The value is a finite remaining-budget fraction in `(0, 1]`; omitted or explicit `None` disables it. It requires a positive `--rollout-max-context-len`. With `0.2`, a reminder is inserted when the remaining context is positive and at most 20% of that limit, asking for a complete, correct kernel on the next turn rather than speculative optimization.

`GenerateState` counts template source tokens once on load with the rollout tokenizer and caches `template_tokens` on the template; the built-in fallback is also cached on first use. `_apply_feedback_template` counts the serialized, truncated feedback as `feedback_tokens` each turn, even when nudge is disabled. Both counts use `add_special_tokens=False` and are returned alongside the formatted feedback, stored in the turn log, and printed in turn stats. The nudge estimates context use as `prompt_tokens + response_tokens + template_tokens + feedback_tokens`, builds the reminder before rendering, and passes `context_budget_nudge` to the template (empty when inactive). This is an estimate: static template tokens include placeholders/Jinja syntax, separate tokenization can differ at boundaries, and new chat framing and the reminder are not counted. The conversation is not re-tokenized for this estimate; the next turn's existing context guard enforces the actual limit.

Built-in and repository response templates include the nudge field. Custom format/YAML templates need `{context_budget_nudge}`, and custom Jinja templates need `{{ context_budget_nudge }}` to display it. Each qualifying feedback may receive a reminder (not one-shot per trajectory). This is next-turn user input, not assistant output, and does not directly change rewards or response loss masks. It acts between turns, not during a single response, and the flag also applies to evaluation if enabled there.

#### KernelGYM detailed correctness diagnostics

`CUDA_AGENT_RETURN_DETAIL_CORRECTNESS` defaults to `0` (`False`). Set it to `1` before starting training to send `return_detail_correctness=true` in KernelGYM evaluation requests. The Qwen3.8 WarmUp/MultiTurn scripts forward this setting through Ray's runtime environment; custom launchers must also forward it to rollout workers.

This requests detailed correctness diagnostics, not a reward change. It is independent of `CUDA_AGENT_ENABLE_COMPUTE_SANITIZER`: either feature can be enabled alone or both can be enabled together. Sanitizer execution remains subject to the server's trigger rules. The explicit `run_request_env.py --mode sanitizer` diagnostic enables only the sanitizer; detailed correctness still follows `CUDA_AGENT_RETURN_DETAIL_CORRECTNESS` (default off).

#### KernelGYM detailed compilation diagnostics

`CUDA_AGENT_RETURN_DETAIL_COMPILATION` independently defaults to `0` (`False`). Set it to `1` to request
compilation-error classification (`return_detail_compilation=true`): the server returns
`metadata.compilation_error_detail` and a summarized `error_message`. When disabled, classification is skipped
and `error_message` contains the full compilation error. This does not change reward computation.
The Qwen3.8 WarmUp/MultiTurn launchers forward it through Ray's runtime environment; custom launchers must do the same.
Training requests, `run_request_env.py`, and `run_response_pipeline.py` use this setting; `--mode compile` does not enable it automatically.

#### Kernel rollout reward post-processing

`examples.kernel_agent.kernel_reward.post_process_rollout_rewards(args, samples)` returns shaped single-turn rewards and writes them back to `sample.reward`. It manages dynamic weighting and overlong penalties; same-prompt/same-turn groups are only an internal statistical scope. It does not compute trajectory returns, baselines, or normalized advantages.

The existing training hook calls this interface:

```bash
--custom-reward-post-process-path examples.kernel_agent.kernel_reward.reward_post_process_by_group \
--dynamic-reward-speedup correct-rate-sqrt \
--dynamic-reward-coverage correct-rate-sqrt \
--overlong-penalty dapo \
--overlong-buffer-len 2048 \
--overlong-penalty-factor 0.2
```

The three independent selectors `--dynamic-reward-correctness`, `--dynamic-reward-speedup`, and `--dynamic-reward-coverage` each default to Python `None`. Omitting a selector or explicitly passing `None` leaves that component unchanged. `--overlong-penalty` independently selects length shaping, applied after dynamic weights. The old `--dynamic-reward-gate`, `--dynamic-reward-gate-range`, `sqrt`, and `inverse-gate` selectors are removed. All three selectors use `--dynamic-reward-amplitude`; `--dynamic-reward-range` is removed. The reward interface returns one list; it must not directly replace the training hook returning `(raw_rewards, rewards)`.

##### Independent additive reward weights

Each selector accepts `piecewise-sqrt`, `piecewise`, `correct-rate-sqrt`, `correct-rate`, and their `inverse-*` variants. All three selectors additionally accept `global-piecewise`, `global-piecewise-sqrt`, and their `inverse-*` variants. For all three components, forward modes **add** `x(p)` to the configured base weight and inverse modes **subtract** it. Here `inverse` means the opposite offset, not a reciprocal. Omitted selectors and explicit `None` leave the corresponding contribution unchanged.

For matched curves, choose inverse correctness and forward speedup:

```bash
--dynamic-reward-correctness inverse-piecewise-sqrt \
--dynamic-reward-speedup piecewise-sqrt \
--dynamic-reward-amplitude 0.1 \
--dynamic-reward-coverage None
```

With default base weights `0.5 / 0.5`, the final weights are `0.5-x(p)` and `0.5+x(p)`. Matching curves preserve their sum of 1. If `CUDA_AGENT_CORRECT_REWARD_WEIGHT` and `CUDA_AGENT_SPEEDUP_REWARD_WEIGHT` override the bases, the weights become `base_correctness-x(p)` and `base_speedup+x(p)`, preserving the configured total. Unmatched curves or enabling only one selector do not generally preserve the sum. This fixes the total weight, not the expected reward.

Let `G` be the valid sample count for one `(group_index, turn_idx)`, `k` its correct count, and `p=k/G`. Removed, aborted, padded, and non-kernel samples are excluded. `--dynamic-reward-amplitude a` defaults to `0.1` and bounds `x` to `[-a,a]`. It must be finite, between 0 and 0.5, and no larger than each enabled component's finite base weight, so adjusted weights remain nonnegative.

- `correct-rate`: `x = a*(2*p-1)`; balanced at `p=0.5`.
- `correct-rate-sqrt`: `x = a*(2*sqrt(p)-1)`; balanced at `p=0.25`.
- `piecewise`: the formula below with `f(t)=t`.
- `piecewise-sqrt`: the same formula with `f(t)=sqrt(t)`, strengthening both tails.
- `global-piecewise`: use `x=0` inside `[h,e]`, otherwise `x=a*(2*p-1)`, without rescaling by the thresholds.
- `global-piecewise-sqrt`: use `x=0` inside `[h,e]`, otherwise `x=a*(2*sqrt(p)-1)`; see below.
- Every `inverse-*` variant subtracts the corresponding `x`; there is no division by `p`, including at `p=0`.

Shared `--difficulty-thresholds` default to `[1/3,2/3]`. Piecewise and global-piecewise modes require exactly two finite, increasing thresholds `h,e` in `(0,1)`:

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

Empty groups are neutral for all modes. Piecewise and global-piecewise singleton groups remain neutral; `correct-rate` and `correct-rate-sqrt` use the singleton's actual `p`. Threshold equality belongs to the neutral region. The removed `--dynamic-reward-range` is replaced by amplitude: the former symmetric multiplier range `[0.8,1.2]` corresponds to `a=0.1` at base weight 0.5; use `a=0.2` for final weights in `[0.3,0.7]`.

Enabled contributions are rebuilt from their base scores on every invocation, making processing idempotent. The speedup correctness prerequisite still applies after weight selection. Failure scores, length shaping, raw task rewards, and frozen future credit are preserved. Logging weights describe the configured group weights before sample correctness gating, not guaranteed nonzero reward contributions.

The Qwen3.8 WarmUp/MultiTurn launchers expose `DYNAMIC_CORRECTNESS_REWARD`, `DYNAMIC_SPEEDUP_REWARD`, and `DYNAMIC_COVERAGE_REWARD`, all defaulting to `None`. `DYNAMIC_REWARD_AMPLITUDE` defaults to `0.1` and is included in the experiment name when any reward adjustment is enabled.

##### Final weights for Group=16

With a=0.2, inverse correctness and forward speedup, default base weights, and thresholds [1/3,2/3]. Each cell is correctness / speedup.

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

##### Global linear curve with a neutral interval

`global-piecewise` and `global-piecewise-sqrt` are additional options, leaving existing modes unchanged. Both interval endpoints are neutral; empty and singleton groups retain their base weights. All three components support these modes and their inverse variants. They use the existing amplitude and thresholds, not `--dynamic-reward-range`:

```python
if G <= 1 or h <= p <= e:
    x = 0
else:
    x = a * (2 * f(p) - 1)
```

Here `f(p)=p` for `global-piecewise`, and `f(p)=sqrt(p)` for `global-piecewise-sqrt`.

Thresholds only switch adjustment on or off; they do not rescale the tails. This generally introduces jumps at h and e. Outside the neutral interval, the linear curve changes sign at p=0.5 and the square-root curve at p=0.25. With default thresholds, global square root has a positive x at p=5/16, then x=0 at p=6/16; it is not globally monotonic. There is no separate piecewise-mode argument.

```bash
--dynamic-reward-correctness inverse-global-piecewise \
--dynamic-reward-speedup global-piecewise \
--dynamic-reward-amplitude 0.1 \
--dynamic-reward-coverage None
```

For the Qwen3.8 launchers, set `DYNAMIC_CORRECTNESS_REWARD=inverse-global-piecewise` and `DYNAMIC_SPEEDUP_REWARD=global-piecewise`. With G=16, a=0.1, default thresholds and base weights, each cell below is correctness / speedup:

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

##### Coverage uses the same additive formula

`--dynamic-reward-coverage` accepts all the same modes, including `global-piecewise*` and inverse variants. Its final weight is `coverage_reward_weight + x(p)` or `coverage_reward_weight - x(p)`; the base comes from `CUDA_AGENT_COVERAGE_REWARD_WEIGHT` (default 0.5). It shares `--dynamic-reward-amplitude` and `--difficulty-thresholds`. Omitting the selector or using `None` leaves the coverage contribution unchanged. The coverage enable switch still controls whether the reward contributes at all.

`--dynamic-reward-range` has been removed. Remove it from old commands and set `--dynamic-reward-amplitude` instead. At base weight 0.5, the former forward piecewise range `[0.8,1.2]` corresponds to amplitude 0.1. Coverage rate modes and inverse modes now use additive offsets rather than the old multiplicative or reciprocal formulas. Enabling coverage adjustment does not preserve the sum of all three weights; paired inverse correctness/forward speedup preserves only their own total.

`kernel_score` keeps scores before reward weighting, while `reward_component` contains weighted contributions. Its `length` contribution is negative for DAPO penalties, positive for LASER-D bonuses, and zero when disabled or inactive. Component metrics use `reward/component/length`, replacing the separate `overlong_penalty` and `length_bonus` components. Top-level `metadata.length_score` uses the same signed value; related metrics are `kernel/length_score/mean` and `exp/rollout/reward/length_score/*`. Existing CLI arguments are unchanged. `task_reward` excludes length shaping and `sample.reward` includes it. Reprocessing rebuilds from scores/components without compounding penalties.

At complete-trajectory finalization, before group reward processing and filtering, TRLOO freezes `metadata.return_reward = gamma * original_task_reward[t+1] + gamma² * original_task_reward[t+2] + ...`. This excludes dynamic weighting, failed-group replacement, and length shaping; turns already marked removed contribute zero. Training computes only `return[t] = current sample.reward + return_reward[t]`, retaining all shaping for the current turn but none in future credit. Later group filtering does not recompute or erase frozen future credit. Single-turn rewards are not overwritten. Legacy rollout dumps missing this field must be regenerated rather than silently falling back to potentially shaped `multi_turn_reward` values.

TRLOO also mirrors frozen future credit into `metadata.reward_component.return_reward` for diagnostics. This component is not part of `sample.reward`; summing all components now includes future credit, giving the pre-baseline return for valid ordinary TRLOO samples. Dynamic weighting and failed-group replacement preserve it. All supported reward components (`correctness`, `performance`, `coverage`, `failed`, `length`, `return_reward`) are logged as regular metrics under `rollout/reward/component/{field}/{mean,min,max}`, without requiring `--log-exp-metrics`. Enable `--use-tensorboard` to write them to TensorBoard. Padding and missing/nonfinite component values are excluded. The old `exp/rollout/reward/component/*` metrics are no longer emitted.

Megatron logs these regular metrics after advantage calculation and rollout postprocessing, before the training forward. They do not require `--log-exp-metrics`; enable `--use-tensorboard` to write them to TensorBoard:

- `rollout/advantage/{positive,negative,zero}_sample_fraction`: sample fractions classified by the sign of each sample's masked mean token advantage.
- `rollout/advantage/{positive,negative}_abs_mass`: the sum of absolute sample advantages of that sign divided by the total sample count, not a within-sign conditional mean.
- `rollout/sequence/{positive,negative}_response_length`: mean full response token count among samples of that sign; each split turn counts as one sample.

These metrics use the current loss masks. Fully masked samples have zero mean advantage and contribute to the zero fraction. DP/CP aggregation uses sums/counts; if a sign has no samples globally, its mean length is reported as 0 and should be interpreted alongside its sample fraction. Masks subsequently added by live-forward seq-MIS are not reflected. The old `exp/rollout/train_batch/advantage/*` and `exp/rollout/train_batch/sequence/*` metrics are no longer duplicated; per-turn sign fractions and async diagnostics still require `--log-exp-metrics`.

`metadata.raw_task_reward` preserves the initial single-turn task reward with the configured base weights, before dynamic weighting, failed-group replacement, and length shaping. It is written at base scoring and is not overwritten by reward postprocessing. Unlike mutable `task_reward`, it is a stable source for historical statistics. TRLOO prefers this field for future credit, falling back to `task_reward` (then the turn reward) for legacy samples at pre-shaping trajectory finalization. It is not an extra additive reward component.

Ordinary kernels follow: base scoring → complete prompt/turn group reward processing (dynamic weights → length bonus/penalty → failed-group replacement) → write back `sample.reward` and `metadata.reward_component` → group filtering → return/advantage calculation. With the kernel reward hook configured, the default SGLang rollout automatically uses `examples.kernel_agent.kernel_reward.generate_rollout`. The fully-async collector settles rewards after draining complete groups and before filtering.

With `apply_failed_group_reward` enabled, detect all-failed groups whose scores all exactly equal the weighted default failure score (`failed_score * init_correct_weight`), using the same reward basis as filtering. Failure-group detection does not use a variance threshold. Ordinary modes use `task_reward` without DAPO penalties; LASER-D uses reward including its bonus (failed samples receive no bonus). If every sample has a saved `kernel_failed_score`, replace task contributions with that score times `init_correct_weight`, updating `task_reward`, `sample.reward`, and components while retaining length penalties exactly once. Then run formal filtering once, followed by return calculation. Replacement does not bypass low-variance or minimum-size filtering.

The filter is read-only: LASER-D reads `sample.reward` including its length bonus; other modes prefer `metadata.task_reward`, falling back to `sample.reward` only when absent. The training hook `reward_post_process_by_group` computes returns/advantages from written-back turn rewards without reshaping a filtered subset. Group processing excludes invalid samples. Disabling length and gate shaping does not disable separate failure-scoring policies or CTM.

Group correctness buckets are logged by default in the regular rollout log and TensorBoard/W&B, regardless of dynamic weighting or `--log-exp-metrics`. `rollout/group_correct_rate_{low,medium,high}_count` and `rollout/group_correct_rate_{low,medium,high}_fraction` report group counts and fractions for `rate < 1/3`, `1/3 <= rate <= 2/3`, and `rate > 2/3`. Each retained `(group_index, turn_idx)` counts once, using its pre-filter `metadata.group_correct_rate`; removed, aborted, padded, and non-kernel samples are excluded. These fixed thresholds do not follow `--difficulty-thresholds`. `rollout/group_correct_rate_count` is the fraction denominator. Groups without a finite rate in `[0, 1]` are excluded; no bucket metrics are emitted when none qualify.

When any dynamic component is enabled, train-data conversion emits a `reward post-process` log and sends `rollout/dynamic_reward/*` scalars to TensorBoard/W&B after final reward processing, without `--log-exp-metrics`:

- `{correctness,speedup,coverage}_{weight,offset}_{mean,min,max,p25,p50,p75}`: group-weighted final weights and signed adjustments; emitted only for enabled components.
- `{correctness,speedup,coverage}_scale_{mean,min,max,p25,p50,p75}`: group-weighted scale statistics, deduplicated by `(group_index, turn_idx)`.
- `{correctness,speedup,coverage}_scale_{zero,scaled,one,boosted}_fraction`: group fractions with scale 0, between 0 and 1, equal to 1, and above 1.
- `{correctness,speedup,coverage}_reward_delta_{mean,min,max,p25,p50,p75}`: sample-weighted changes from unscaled base contributions, including unchanged zeros. Speedup deltas refer to performance reward, not measured speedup.
- `group_count`, `sample_count`: valid groups and samples with final records. Removed, aborted, padded, and non-kernel samples are excluded.

Only enabled components emit dynamic metrics. Final per-sample records live in `metadata.dynamic_reward` and are created before filtering; train-data conversion aggregates retained samples only. The former shared `gate_*` metrics are replaced by separate component scales. Regular `rollout/group_correct_rate_*` metrics remain independent of all dynamic switches. Early generation logs can precede complete-group weighting.

#### LASER-D adaptive length rewards

`--overlong-penalty` accepts `None` (default/off), `dapo` (the existing linear subtraction), or `laser-d`. Despite the shared selector name, LASER-D is a **bonus**: add a constant only to correct responses whose full response token count is within budget. Incorrect or over-budget responses are unchanged. No decoding cap, prompt budget, or speedup multiplier is introduced. See [the paper, Table 2 / Section 5](https://arxiv.org/html/2505.15612v1#S5) and [official reward code](https://github.com/hkust-nlp/Laser/blob/main/verl/utils/reward_score/length_penalty.py).

```bash
--custom-reward-post-process-path examples.kernel_agent.kernel_reward.reward_post_process_by_group \
--overlong-penalty laser-d \
--laser-d-length-score 0.5 \
--laser-d-min-length 1024 \
--laser-d-length-interval 1024 \
--laser-d-update-interval 20 \
--laser-d-monitor-groups 500
```

These are the defaults. The search upper bound is `--rollout-max-response-len`; the lower bound must not exceed it. DAPO's buffer, penalty factor, and effective-context-cap options do not affect LASER-D. Dynamic reward weighting remains independent and does not scale the bonus.

Reuse the two `--difficulty-thresholds` (default thirds): hard if `s < h`, medium if `h <= s < e`, easy if `s >= e`, where `s` is the valid prompt/turn group's correctness rate. Monitoring and reward assignment share these boundaries. Removed, aborted, and padded samples are excluded.

```text
C_min(g) = smallest attainable positive correct count in g's bucket
           K=8: hard=1, medium=3, easy=6; K=16: hard=1, medium=6, easy=11
coverage_g(L) = fraction of ALL responses in g with response_length <= L
ECR_bucket(L) = mean_g[C_min(g) * coverage_g(L)]
B_bucket = smallest grid L with ECR >= 1; use response cap if unavailable
reward = task_reward + bonus * I(correct and response_length <= B_bucket)
```

At fixed group size this reduces to `C_min * coverage`; variable-size turn groups receive equal group weight. The grid always includes the cap, even when not divisible by its interval. ECR is a coverage proxy, not a correctness guarantee. We use the paper's minimum attainable counts rather than the official code's optional floor approximation.

Our monitoring adaptation reuses **pre-dynamic-filter training rollouts**, not a separate monitoring dataset/rollout. A bounded reservoir keeps at most `monitor-groups` prompt/turn length lists per update window, without response bodies. Rejected groups contribute; unconsumed warm-queue groups do not. Each turn contributes one observation. All budgets start at `min-length`. Bootstrap after the first rollout with monitoring data; then update every `update-interval` rollout batches (not actor optimizer steps), clearing the reservoir after each update.

Each collector batch uses a fixed budget snapshot; its observations only change future batches. The shared reward postprocessor writes `sample.reward` and positive `reward_component.length`; `task_reward` excludes the bonus. Reprocessing is idempotent; TRLOO retains the bonus only for the current turn, not in future credit.

LASER-D's low-variance filter includes the bonus, retaining all-correct groups with informative length differences. DAPO retains its existing pre-penalty `task_reward` filter policy.

Argument validation automatically routes the default SGLang rollout through the LASER-D collector; kernel-agent fully-async supports it directly. Keep dataset management enabled by default (do not pass `--disable-rollout-global-dataset`) and use the reward hook above; other custom rollout paths fail validation. Budgets, reservoir, update position, and RNG state use data-source `metadata["laser_d"]`, saved in the existing `rollout/global_dataset_state_dict_<rollout_id>.pt`. Resume with the matching checkpoint; changed budget/monitor settings fail explicitly. Unfinished batches do not commit budget state.

Existing TensorBoard/W&B channels log `rollout/laser_d/{hard,medium,easy}/budget_used` and `budget_next`, plus `budget_updated`, `observed_groups`, and `monitor_groups_retained/seen`. Final training records provide `length_score_mean`, `bonus_fraction`, `sample_count`, and each bucket's `budget_mean`/`sample_count`. Per-sample snapshots live in `metadata.laser_d`. No `--log-exp-metrics` flag is required.

#### GRPO Algorithm

GRPO (Group Relative Policy Optimization) is an RL algorithm proposed in DeepSeek-Math. Its core idea is to compute advantage through intra-group relative comparisons, eliminating the need for a separate critic model.

To use GRPO, set:

```bash
--advantage-estimator grpo
```

Key features of GRPO:

- **No Critic Model Required**: GRPO samples multiple responses for the same prompt and estimates advantage by computing relative rewards within the group, avoiding the overhead of training and maintaining a critic model.
- **Resource Efficient**: Since no critic model is needed, GPU resources can be fully utilized for actor training and inference.
- **Simple to Use**: Easy configuration - just set `--advantage-estimator grpo`.

Related parameters:

- `--n-samples-per-prompt`: Number of responses sampled per prompt for intra-group comparison.
- `--normalize-advantages`: Whether to normalize advantages.
- `--use-conditional-truncation-mask`: Kernel Agent reward post-processing records only the CTM selection. The Megatron backend zeros selected advantages after OPD and advantage normalization, or at the end of advantage computation when normalization is disabled. Rewards, returns, loss masks, and normalization statistics are unchanged. Runtime and later execution-stage failures are excluded from CTM masking.
- `--eps-clip`: PPO-style clip range.

#### PPO Algorithm

PPO (Proximal Policy Optimization) is a classic RL algorithm that uses a critic model to estimate the value function for computing advantages.

To use PPO, set:

```bash
--advantage-estimator ppo
```

**Note: In PPO, the critic and actor share the same training GPU group.** You do not need to reserve a separate set of GPUs for the critic. Specifically:

- PPO creates separate actor and critic training process groups, but places them on the same train placement group.
- The critic training scale follows the actor configuration, and the actor / critic Megatron parallel topology must currently stay identical.
- PPO forces train-side offload so that actor and critic can wake up and release memory on the same GPUs in turn.
- There are currently no separate CLI arguments for configuring critic training resources; the critic node count and GPUs per node are derived from the actor configuration.


PPO-related parameters:

- `--megatron-config-path`: YAML config for role-specific Megatron overrides, such as setting critic-specific `load`, `save`, `lr`, or warmup parameters.
- `--num-critic-only-steps`: Number of steps to train only the critic at the beginning of training.
- `--eps-clip`: PPO clip range.
- `--value-clip`: Clip range for value loss.
- `--kl-coef`: KL penalty coefficient for reward shaping.

### Advanced Megatron Configuration (--megatron-config-path)

For PPO workflows, you can use `--megatron-config-path` with a YAML file to override Megatron arguments separately for actor and critic. Common use cases include setting a different critic `lr`, or giving actor and critic different `load` / `save` paths.

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

> **Note:** This configuration currently only supports PPO, and in current PPO the actor and critic must use the same Megatron parallel topology. The recommended pattern is to keep parallelism-related settings in the shared CLI arguments and put only role-specific differences in YAML. See [Megatron Config: Role-Based Training Overrides](../advanced/megatron-config.md) for details.

## Custom Rollout Function

slime supports customizing data generation (rollout) to various degrees.

  - By default, it uses the `generate_rollout` function from [slime/rollout/sglang_rollout.py](https://github.com/THUDM/slime/blob/main/slime/rollout/sglang_rollout.py) for data generation. This file implements an asynchronous (asyncio) data generation flow based on SGLang and supports features like dynamic sampling and partial rollout.

  - You can completely replace the `generate_rollout` in sglang\_example.py by using the `--rollout-function-path` parameter. You just need to ensure that the function signature passed via `--rollout-function-path` is as follows:

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

    Where:

      - `args`: The complete arguments used for the slime run.

      - `rollout_id`: The ID of the current data generation round, used to ensure data order when resuming training.

      - `data_source`: A globally unique data source in slime, which can be used to get initial prompts, data IDs, and store partially generated samples for later use.

      - `evaluation`: A boolean indicating if the rollout is for evaluation. You can configure a separate evaluation function using `--eval-function-path`.

      - The returned `Sample` type is defined in [slime/utils/types.py](https://github.com/THUDM/slime/blob/main/slime/utils/types.py). When implementing, you need to ensure the following fields are correctly set:

          - `tokens`: The tokens for the prompt + response.
          - `response_length`: The total length of the response. For multi-turn tasks, this is the length of the tokens remaining after the first-turn prompt.
          - `reward`: The reward for this data sample.
        - `status`: The status of this data sample (e.g., `Sample.Status.COMPLETED`, `Sample.Status.TRUNCATED`, `Sample.Status.ABORTED`, `Sample.Status.FAILED`).
          - `loss_mask` should be the same length as `response_length`, with `1` for tokens that should be included in the loss calculation and `0` for those that should be masked out.

  - In some cases, you may only need to replace the data generation logic. You can do this using `--custom-generate-function-path`. A simplified implementation of this function is as follows:

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

    For a more complete version, please refer to [slime/rollout/sglang_rollout.py](https://github.com/THUDM/slime/blob/main/slime/rollout/sglang_rollout.py).

  - Sometimes, you may also need to support a custom reward model. This can be configured by setting `--custom-rm-path`.

## How to Use SGLang

slime implements a server-based engine using SGLang via the `HttpServerEngineAdapter` as an intermediary.

### Parameter Configuration

slime incorporates almost all SGLang parameters by using SGLang's `ServerArgs.add_cli_args`. When setting an SGLang parameter, you need to add the `--sglang-` prefix. For example:

  - In co-located training and inference, you often need to limit `--mem-fraction-static`. This parameter should be changed to `--sglang-mem-fraction-static`.
  - During training, if you want SGLang to infer beyond the maximum context length specified in the Hugging Face checkpoint's `config.json`, you need to use `--context-length`, which becomes `--sglang-context-length` in slime.
  - For multi-node large EP inference, you might need `--ep-size`, `--enable-dp-attention`, `--dp-size`, `--moe-a2a-backend deepep`, etc. These can be passed as `--sglang-ep-size`, `--sglang-enable-dp-attention`, `--sglang-dp-size`, and `--sglang-moe-a2a-backend deepep` respectively.

Some parameters related to slime's resource scheduling are configured by slime itself, for example:

  - `--tp-size` in slime is set using `--rollout-num-gpus-per-engine`.
  - `--model-path` in slime is set using `--hf-checkpoint`.

The way SGLang parameters are integrated into slime can be found in [slime/backends/sglang_utils/arguments.py](https://github.com/THUDM/slime/blob/main/slime/backends/sglang_utils/arguments.py).

### How to Use the Router

slime uses [sglang-router](https://github.com/sgl-project/sglang/tree/main/sgl-model-gateway) to manage the SGLang servers during the training process. You can configure the address of the [sglang-router](https://github.com/sgl-project/sglang/tree/main/sgl-model-gateway) using `--sglang-router-ip` and `--sglang-router-port`. If not configured, a router will be started by default within the cluster.

After starting, all SGLang servers will register with the router via the `/add_worker` endpoint. When actually generating data, you only need to send HTTP requests to the router, which will perform load balancing and forward the requests to the servers.

When you configure an external router using `--sglang-router-ip` and `--sglang-router-port`, slime will not start an internal router. Instead, it will register all its servers with this external router. You can then use this external router's address to implement more complex data generation workflows. Note that the router supports OpenAI-compatible APIs.

### Advanced Engine Configuration (--sglang-config)

For advanced deployments, you can use `--sglang-config` with a YAML file to configure server groups, multi-model serving, and selective weight updates.

**Multi-model deployment** allows serving multiple models simultaneously (e.g., an actor model that receives weight updates and a frozen reference/reward model):

```yaml
sglang:
  - name: actor
    update_weights: true          # receives weight updates from training (default)
    server_groups:
      - worker_type: regular
        num_gpus: 8
        num_gpus_per_engine: 4
  - name: ref
    model_path: /path/to/ref_model
    update_weights: false          # frozen, no weight updates
    server_groups:
      - worker_type: regular
        num_gpus: 4
        num_gpus_per_engine: 2
```

Each model gets its own router. The per-model router info is accessible via `args.sglang_model_routers` (a dict mapping model name to `(ip, port)` tuples). Custom rollout functions can use `get_model_url(args, "ref")` from `slime.rollout.sglang_rollout` to route requests to a specific model.

**Server group features:**
- `worker_type`: `regular`, `prefill`, `decode`, or `placeholder` (reserves GPU slots without creating engines)
- `overrides`: Dict of SGLang `ServerArgs` field overrides applied on top of `--sglang-*` CLI args
- `num_gpus_per_engine`: Per-group TP size override

## How to Use Megatron

slime supports different and lightly modified versions of Megatron by reusing common functions from the `megatron.training` directory, such as `parse_args`, `save_checkpoint`, and `load_checkpoint`. Therefore, when using it, you must ensure that Megatron is accessible in the `PYTHONPATH`, for example, by adding `export PYTHONPATH=/root/Megatron-LM` at runtime.

### Parameter Configuration

slime directly imports all parameters of the Megatron in the current environment by using `from megatron.training.arguments import parse_args`. If the version of Megatron you are using has parameters defined outside of `parse_args`, you can configure them by passing them in, similar to how it's done in [train.py](https://github.com/THUDM/slime/blob/main/train.py), for example:

```python
if __name__ == "__main__":
    try:
        from pretrain_gpt import extra_args_provider
    except:
        extra_args_provider = None
    args = parse_args(extra_args_provider)
    train(args)
```

### Custom Parameters

In some customized Megatron implementations, special operations need to be performed during initialization or before/after a training step. We have added the following plugins for this purpose:

  - `--custom-megatron-init-path`: Adds some initialization calls.
  - `--custom-megatron-before-log-prob-hook-path`: Is called before calculating the log probability.
  - `--custom-megatron-before-train-step-hook-path`: Is called before each training step. You could use this to mix in special training losses, for example.
