# Coverage-only integration

Base: `9cd8b634998cff609e362fb64073a4209b5e74b8`. Source: the coverage-related portion of `ff67bacbcb1024d4a73c9ae8d8b6f6340d74b679` (time-coverge PR #4). Branch: `time-coverage-only`.

This is a selective source integration, not a merge of the original branch history: unrelated optimization commits must not become ancestors of this branch.

Included: shared coverage computation; configurable coverage type and weight; reference-time coverage, efficiency-discounted reference coverage, gated time coverage and capped-speed auxiliary alternatives; invalid reference-runtime filtering; reference-coverage feedback and rejection sampling; associated CPU contract tests. Defaults remain `time_coverage` and weight `0.5`. Select alternatives with `CUDA_AGENT_COVERAGE_REWARD_TYPE` and `CUDA_AGENT_COVERAGE_REWARD_WEIGHT`; `--coverage-rs-key reference_time_coverage` is available.

Excluded from the initial coverage-only integration: FP32 LM-head GEMM/MTP changes, native SGLang sampling-mask/logprob switching, top-p replay chunking, loss/response-extraction changes, GDN CPU-boundary changes, communication polling, checkpoint changes, Ray placement, runtime/deployment patches, hardware launchers, eval suites, unrelated verbosity controls, CI and AGENTS changes. Existing behavior already present at the base is intentionally preserved, including its gated-attention compatibility and GDN optimizations.

Verification on 2026-10-08:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python -m pytest \
  tests/test_kernel_agent_coverage.py \
  tests/test_kernel_agent_reference_coverage_rs.py \
  tests/test_kernel_agent_capped_speed_auxiliary.py \
  tests/test_kernel_agent_gated_time_coverage.py \
  tests/test_kernel_agent_partial_reward.py -q -o addopts=''
```

Result: **152 passed**, 25 warnings, 11.10 seconds. No training, GPU workload or service restart performed.

At the initial coverage-only integration, the following scope check produced no differences:

```bash
git diff 9cd8b634 -- slime/backends slime/rollout slime_plugins \
  slime/utils/ppo_utils.py slime/utils/types.py
```

At that initial integration, the only core `slime/` change was adding `reference_time_coverage` to `--coverage-rs-key` choices. At that point, `generate_with_cuda_agent.py` imported only the math import and invalid-reference reward filtering; rollout request/logprob code was unchanged.

## Additional selected integrations

The branch now also includes these user-requested changes from `main`:

- GDN CPU sequence-boundary caching, including packed-batch metadata construction, cache invalidation, and boundary validation. The optional FlashQLA CPU-metadata patch script is included with its contract test; copying the script does not patch the installed package.
- Configurable `sqrt` reward gate bounds and support for `inverse-gate` when the minimum is positive, from `f87db0e7`. All enabled gate modes now require an explicit range; no mode-specific default is supplied.

The earlier scope check describes the initial integration only. The selected GDN changes now also modify `slime/backends/megatron_utils/data.py`, `slime/utils/sequence_metadata.py`, and `slime_plugins/models/distributed_gdn.py`. The sqrt integration additionally updates argument validation and reward processing. Training launcher parallelism and token-budget defaults remain those of the pre-merge base; the speedup reward setting below is an intentional additional change.

Validation of the additional integrations:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python -m pytest \
  tests/test_sequence_metadata.py tests/test_packed_sequence_metadata.py \
  tests/test_distributed_qwen_gdn.py tests/test_rollout_reward_post_process.py \
  -q -o addopts=''
```

Result: **474 passed, 7 skipped** (CPU-only run). Black, isort, and whitespace checks passed. The two new metadata test files are registered in the CPU CI matrix. No GPU training or installed-package patching was performed.

## Speedup reward upper bound

`CUDA_AGENT_SPEEDUP_REWARD_UPPER` configures `speedup_reward_upper_bound` and defaults to `2.0` (previously fixed at `5.0`). Both Qwen3.8 WarmUp and MultiTurn launchers forward it through the Ray runtime environment and append `-SRUP${CUDA_AGENT_SPEEDUP_REWARD_UPPER}` to the experiment name. For example:

```bash
CUDA_AGENT_SPEEDUP_REWARD_UPPER=3 bash examples/kernel_agent/run_qwen3.8_27B_WarmUp.sh
```

In legacy mode, speedup scores are capped at this value; improvement modes use it as the speedup corresponding to score 1.

## Sequence MIS integration

The sequence MIS changes from `5835beda` are also integrated. Static pre-training filtering and dynamic live-forward filtering now share `slime/utils/sequence_mis.py`. The `actor_logprob` option selects `static` or `dynamic`; defaults remain static for existing aggregations and dynamic for `binary_kl`. The `turns_*` aggregations require static filtering. Actor/old-actor recomputation, policy-loss masks, validation, documentation, and regression tests are included.

This selective integration modifies the MIS portions of `loss.py` and `ppo_utils.py`; it does not import the top-p mask chunking, native SGLang sampling-mask path, or predictive-support extraction changes. Existing launcher MIS enablement remains unchanged.

Sequence MIS validation: `CUDA_VISIBLE_DEVICES='' PYTHONPATH=. python -m pytest tests/test_token_sum_loss.py tests/deepseek-v4/test_dsv4_lora_old_actor.py tests/test_rollout_temperature_zero_guard.py tests/test_rollout_reward_post_process.py -q -o addopts=''` — **528 passed**. Black, isort, and whitespace checks passed; no GPU training was run.


After requiring explicit gate ranges and adding the speedup environment setting, the combined CPU regression run covered metadata/GDN, reward processing, partial rewards, sequence MIS, old-actor recomputation, and temperature validation: **703 passed, 7 skipped**. The earlier counts above describe their respective integration stages.

## Multiple KernelGym endpoints

Use the same option for one or multiple endpoints. Space- or comma-separated URLs are automatically split and deduplicated:

```bash
python scripts/check_kernelgym_health.py --url "http://host-a:20111 http://host-b:20111"
```

Training accepts `--kernel-env-url "http://host-a:20111 http://host-b:20111"`. The Qwen3.8 WarmUp and MultiTurn launchers also accept multiple addresses in `KERNEL_ENV_URL`. The older plural options and environment variables remain supported; `KERNEL_ENV_URLS` takes precedence when set. Health checks require matching GPU hardware and, when every endpoint reports it, matching KernelGym versions.
