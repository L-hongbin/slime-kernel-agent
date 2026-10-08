# Coverage-only integration

Base: `9cd8b634998cff609e362fb64073a4209b5e74b8`. Source: the coverage-related portion of `ff67bacbcb1024d4a73c9ae8d8b6f6340d74b679` (time-coverge PR #4). Branch: `time-coverage-only`.

This is a selective source integration, not a merge of the original branch history: unrelated optimization commits must not become ancestors of this branch.

Included: shared coverage computation; configurable coverage type and weight; reference-time coverage, efficiency-discounted reference coverage, gated time coverage and capped-speed auxiliary alternatives; invalid reference-runtime filtering; reference-coverage feedback and rejection sampling; associated CPU contract tests. Defaults remain `time_coverage` and weight `0.5`. Select alternatives with `CUDA_AGENT_COVERAGE_REWARD_TYPE` and `CUDA_AGENT_COVERAGE_REWARD_WEIGHT`; `--coverage-rs-key reference_time_coverage` is available.

Excluded: FP32 LM-head GEMM/MTP changes, native SGLang sampling-mask/logprob switching, top-p replay chunking, loss/response-extraction changes, GDN CPU-boundary changes, communication polling, checkpoint changes, Ray placement, runtime/deployment patches, hardware launchers, eval suites, unrelated verbosity controls, CI and AGENTS changes. Existing behavior already present at the base is intentionally preserved, including its gated-attention compatibility and GDN optimizations.

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

The following scope check produced no differences:

```bash
git diff 9cd8b634 -- slime/backends slime/rollout slime_plugins \
  slime/utils/ppo_utils.py slime/utils/types.py
```

The only core `slime/` change is adding `reference_time_coverage` to `--coverage-rs-key` choices. In `generate_with_cuda_agent.py`, only the math import and invalid-reference reward filtering are imported; rollout request/logprob code remains unchanged.
