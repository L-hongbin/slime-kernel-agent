# BF16 Baseline Jump Root Cause

## Summary

The May 24 to May 28 BF16 baseline jump was caused by an SGLang Qwen3.5 GDN
kernel bug fixed between the two environments, not by SmoothQuant, rotation,
prompt changes, tokenizer changes, `extra_buffer`, overlap scheduling, or
thinking-parser behavior.

The likely root cause is SGLang `sgl-project/sglang#21019` introducing a fused
Qwen3.5 GDN projection path whose `a/b` outputs can be non-contiguous, while the
old GDN Triton kernels still assumed contiguous layout. SGLang
`sgl-project/sglang#22312` fixed this by making GDN kernels stride-aware.
`#22312` first appears in SGLang `v0.5.11`; it is absent from `v0.5.10.post1`
and present in `v0.5.12.post1`.

## Observed Effect

Same BF16 checkpoint, prompt data, and KernelGym reward server:

| run | SGLang/env | correctness | missing_response | final sections | final `</think>` | final mean response_len |
|---|---|---:|---:|---:|---:|---:|
| `20260524_142451_*v2_3_env_n8` BF16 | old | 0.27875 | 79 | 730 | 684 | 6436 |
| `20260528_043455_*newSlimeKG` BF16 | new `0.5.12.post1` | 0.37875 | 7 | 793 | 793 | 8453 |

T1 also changed sharply: old BF16 mean response length was `9163` tokens; new
BF16 mean response length was `15698` tokens. The checkpoint mtime was Apr 30
and did not change.

The visible symptom was premature/incomplete generation: old outputs often
ended with or triggered `<|im_end|>` before `</think>` and before the required
CUDA sections. This was a downstream symptom of corrupted logits, not a direct
stop-parser bug.

## Version And Config Evidence

Known server-arg differences:

| key | May24 old BF16 | May28 new BF16 |
|---|---|---|
| `disable_overlap_schedule` | `True` | `False` |
| `mamba_scheduler_strategy` | `no_buffer` | `extra_buffer` |
| `max_running_requests` | 64 | 48 |
| `random_seed` | 1237 | 1234 |
| `hicache_storage_prefetch_policy` | `best_effort` | `timeout` |

Version evidence:

- May24 old SGLang was before `v0.5.12`; based on removed/new `server_args`,
  likely around `0.5.10.post1` or a nearby dev build. One concrete marker:
  `enable_double_sparsity` was removed by SGLang PR `#23009` / commit
  `44e67c683`, which is included in `v0.5.12`; May24 still had the older
  server-args shape.
- May28 had 42 new `server_args` and 8 removed `server_args` relative to May24,
  including removed `enable_double_sparsity`, `ds_heavy_*`,
  `multi_item_scoring_delimiter`, and `collect_tokens_histogram`.
- May28 new SGLang was measured as `0.5.12.post1` on `.64` with
  `python -c "import sglang; print(sglang.__version__)"`.
- `sgl-project/sglang#22312` first appears in `v0.5.11`.

## Exclusions

- Prompt and tokenizer are not the cause: representative prompts and token IDs
  matched across old/new runs.
- `extra_buffer` is not the cause: old SGLang with `extra_buffer` and overlap
  still showed old output shape and low accuracy. The counterexample is
  `checkpoints/Qwen3.6-27B/20260526_085412_ctx65536_n8_summ1600_w8a8-rtn-local-rotated-mlp-mambasched-v2_3_env_n8`,
  with `disable_overlap_schedule=False`, `mamba_scheduler_strategy=extra_buffer`,
  `max_running_requests=64`, `random_seed=1237`; final correctness was `0.23375`,
  `missing_response=80`, final sections `728`, final `</think>` `685`, final
  mean response length `6329`.
- Overlap scheduler alone is unlikely: the same old-SGLang `extra_buffer` run
  had `disable_overlap_schedule=False` but still produced old behavior.
- Strict/thinking parser is not the cause: May28 had `enable_strict_thinking=False`
  and `strip_thinking_cache=False`; those code paths were gated off.
- Random seed cannot explain the jump: it may move +/-1-2pp, not
  `missing_response` 79 to 7 and final `</think>` 684 to 793.

## Root Cause Details

Relevant SGLang PRs:

- `sgl-project/sglang#21019`
  `[Qwen3.5] Fuse split/reshape/cat ops in GDN projection with Triton kernel`
  - commit `5bdc07d97`
  - present in both `v0.5.10.post1` and `v0.5.12.post1`
  - creates the fused Qwen3.5 GDN projection path where the fallback BA path can
    return non-contiguous split views.

- `sgl-project/sglang#22312`
  `Make GDN support non-continuous B/A Tensor input to fix the accuracy regression of Qwen3.5-27B`
  - commit `8ba964604`
  - first release: `v0.5.11`
  - absent from `v0.5.10.post1`, present in `v0.5.12.post1`
  - changes `fused_gdn_gating.py` and `fused_sigmoid_gating_recurrent.py` from
    hardcoded contiguous pointer arithmetic to explicit `stride_a` / `stride_b`
    reads and token-axis stride updates.
  - PR report says the Qwen3.5-27B regression test improved from `3/50` to
    `49/50` after the fix.

Interpretation: the May24 environment was likely in the bad window after
`#21019` but before `#22312`; the May28 environment includes `#22312`.

Earlier planned ablations with May24 server args on `0.5.12.post1` were not run.
They are no longer required to explain the jump because the version-diff audit
found the direct Qwen3.5-27B accuracy-regression fix above.

## vLLM Comparison

vLLM does not appear to have the same bug. Its Qwen3.5 implementation made
`a/b` contiguous before entering the GDN core from initial support onward:

- `vllm-project/vllm#34110`: initial Qwen3.5 support; uses separate
  `in_proj_b` / `in_proj_a` and explicitly calls `b.contiguous()` /
  `a.contiguous()`.
- `vllm-project/vllm#34492`: Qwen3.5 GDN projector fusion.
- `vllm-project/vllm#34683`: reverted `#34492` due to Qwen3-Next random-symbol
  output.
- `vllm-project/vllm#34697`: redid the projector fusion.
- `vllm-project/vllm#37975`: extracted shared GatedDeltaNetAttention for
  Qwen3-Next and Qwen3.5; current code still does
  `ba.chunk(...); b/a.contiguous()`.

vLLM has its own Qwen3.5/GDN bugfix history, but not the SGLang pattern where
non-contiguous `a/b` split views were passed directly into kernels that assumed
contiguous layout.

## Key References

- SGLang root-cause fix: https://github.com/sgl-project/sglang/pull/22312
- SGLang inducing fused projection: https://github.com/sgl-project/sglang/pull/21019
- Related premature-stop symptom report: https://github.com/sgl-project/sglang/issues/20550
- vLLM initial Qwen3.5 support: https://github.com/vllm-project/vllm/pull/34110
- vLLM Qwen3.5 GDN fusion/revert/redo: https://github.com/vllm-project/vllm/pull/34492, https://github.com/vllm-project/vllm/pull/34683, https://github.com/vllm-project/vllm/pull/34697
- vLLM shared GDN layer: https://github.com/vllm-project/vllm/pull/37975
- Main investigation context: `handoffs/complete/handoff_drkernel_w8a8_rollout.md`
