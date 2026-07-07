# DeepSeek-V4-Flash RL — training-not-learning investigation + fix plan

**Status (2026-07-07):** the MIS-off DrKernel run `misoff2_0706_2222` (25 steps) did NOT learn.
`rollout/kernel/turn0/correctness` flat ~0.45, `compilation` ~0.63, `raw_reward` ~0.5,
`train/grad_norm` flat ~0.018. Run STOPPED, cluster idle. This doc is the current state so work
can resume cleanly. Nothing is being changed yet (user: "update handoff, do not do anything else").

## Concern 1 — training not learning (OPEN, ROOT CAUSE UNRESOLVED, TOP PRIORITY)

- Symptoms: `grad_norm` ~0.018 flat every step; reward/correctness/compilation flat over 25 steps.
- **NOT the advantage estimator.** `trloo` (leave-one-out, no std-norm, `ppo_utils.py:234-235`)
  is the SAME estimator that WORKS for Qwen3.6-27B on the same DrKernel task → estimator is fine.
  **Do NOT modify the advantage function.** (An earlier "grad_norm 0.19 random-reward vs 0.018
  real-reward" comparison is INVALID — it conflated random vs real reward, not a fair baseline.
  Discard it as evidence.)
- Cause is **V4-SPECIFIC**. Candidates to investigate (in priority order):
  1. **Gradient flow to the attention LoRA through the FROZEN fp8 MoE** (`V4_FP8_FROZEN_EXPERTS=1`,
     `custom_kernels/deepseek_v4/megatron/mcore_model.py`) — fp8 quantization / the swiglu-clamp STE
     in the backward may be dampening the grad that must pass through the MoE to reach the attn LoRA.
  2. **mHC (hyper-connection) gradient flow** — detached Sinkhorn "mixing oracle" + the B1 kernel
     backward (`custom_kernels/deepseek_v4/mhc/`, `decoder.py`).
  3. **LoRA on ATTENTION ONLY** (MoE frozen) — is attention-only capacity enough to move this task?
  4. **Custom-kernel backward correctness** (A1 attention `custom_kernels/deepseek_v4/attention/`,
     mHC B1) — verify grads vs the torch reference.
- Diagnostic idea: compare V4 real-reward grad_norm/LoRA-grad against a Qwen3.6-27B run on the same
  task (Qwen learns) to localize what V4 does differently in the backward.

## Concern 2 — mHC prenorm fp32 (DECIDED: REVERT)

Forced sglang mHC prenorm to fp32 via `V4_ALIGN_MHC_FP32=1` (`SGLANG_OPT_USE_TILELANG_MHC_PRE=false`
+ `SGLANG_OPT_DEEPGEMM_HC_PRENORM=False`). DeepSeek-V4 uses **shared bitwise batch-invariant kernels
train↔inference** (paper §High-Performance Batch-Invariant... kernels); sglang's TF32
(`SGLANG_OPT_DEEPGEMM_HC_PRENORM`) is an sglang OPT, not self-evidently DeepSeek's design. Effect is
≤1.7e-3 (measured) and forcing the torch fp32 path costs rollout throughput (disables the TileLang
mHC-pre kernel). → **Revert**: let sglang use its TF32 default; do not special-case.

## Concern 3 — wandb `train/` metrics not visible (OPEN, TO FIX)

`train/loss`, `train/grad_norm`, `train/entropy_loss` reach the console (`model.py:923`) but not the
wandb run. `wandb_utils.py:224-225` defines `train/*` with `step_metric=train/step`, but `train/step`
is only populated in `log_rollout_data` (`data.py:211`) when `wandb_always_use_train_step` is set,
and the train-actor metrics themselves may not be routed to slime's centralized wandb run. Fix:
wire `train/step` (`--wandb-always-use-train-step`) + verify the train-actor → centralized-run path.
(Prior `--wandb-centralized` did NOT fix it.)

## Concern 4 — indexer 16k mismatch (OPEN, MEASURE FIRST)

Train attends DENSE over compressed KV (indexer top-k dropped); rollout attends SPARSE top-512.
Measured ≤0.0007 at **ctx-8192 only**; UNMEASURED at 16k, where the indexer keeps top-512 of ~4096
compressed (~12.5% vs ~50% at 8k) → dense-train sees ~8× more entries. **Measure at 16k first**;
port the top-k sparse mask into the A1 tilelang kernel (fwd+bwd) ONLY if material. `V4_SPARSE_ATTENTION`
reference path exists but is O(S²), too slow for a real 16k run.

## Concern 5 (task #14) — routing-replay None-crash (OPEN, FIX PROPERLY)

Failed/aborted rollouts have no `routed_experts` in meta_info → `sample.rollout_routed_experts=None`
→ `torch.from_numpy(None)` crash (`actor.py:307`); replay consumer asserts `len==len(tokens)`.
Worked around for the experiment by `USE_ROLLOUT_ROUTING_REPLAY=0` (reintroduces routing mismatch).
Proper fix: filter None-routed_experts samples **globally BEFORE the DP shard** (where the per-DP
`rollout_data_ref` Box list is built — `_get_rollout_data`/`process_rollout_data` is already
per-rank, so filtering there deadlocks PP on uneven counts). Then keep routing replay ON.

## Restart config (DECIDED)

- **GBS=256, N_SAMPLES=16, 16k ctx** — the validated config, NOT the fast GBS-64 experiment config
  (the small batch was only for faster entropy-trend observation and is not for real training).

## Fixes already applied this session (train→rollout alignment; Codex-reviewed; 94 tests pass)

- **KEEP:** router fp32 scoring + removed `+1e-20` renorm (`mcore_model.py`); RMSNorm sglang
  cast-order (`rope.py`, `V4_RMSNORM_HF=1` reverts). Both Codex-verified.
- **REVERT:** mHC fp32 forcing (concern 2).

## Next steps (do AFTER user approval — currently paused)

1. **Investigate concern 1** (V4-specific gradient flow to the attention LoRA) — the priority; the
   run cannot restart usefully until the model actually learns.
2. Apply fixes: concern 2 (revert mHC fp32), concern 3 (wandb), concern 5 (routing-replay global filter).
3. Measure concern 4 at 16k.
4. Codex re-review the fixes.
5. Restart with GBS=256 / 16k.

## Notes / open items

- A Codex diagnostic review timed out (>31 min, idle at 0.4% CPU); this doc is the self-review
  substitute. The stuck Codex process is still alive but idle.
- The MIS-off entropy result (entropy stable ~0.40, no collapse) is INVALID as a conclusion because
  the underlying run was not learning — see [[v4-mis-off-no-entropy-collapse]] (needs revisiting once
  the model learns).
