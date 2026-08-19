# DeepSeek-V4-Flash LoRA on slime/Megatron

## Engineering State

The R6 full loop runs **end-to-end** and both historical blockers are
**root-caused**:

1. **Driver "external SIGKILL"** = concurrent-run fratricide (other agent
   sessions' `ray stop --force` on shared nodes). Resolved — see "Root Cause of
   R6 Driver Death" below.
2. **First-backward NaN** (`found NaN in local grad norm for bucket #0`, PP
   stage 1) = **division by rollout temperature 0 in the train-side log-prob
   path**. Fixed in `slime/backends/megatron_utils/loss.py` — see next section.
   **Verified: attempt36_tempfix PASSED the full R6 gate** (finite loss 0.0743,
   grad_norm 0.344, routing-replay dump verified). The R6 smoke's post-hoc dump
   verifier also needed `--expected-samples` (it defaulted to R4's 2; R6 has 8)
   — fixed in `scripts/dsv4/_dsv4_launch_core.sh` (both call sites) via `N_SAMPLES_PER_PROMPT`.

## Root Cause of the First-Backward NaN: rollout-temperature 0

The R6 smoke passes `--rollout-temperature 0` (greedy rollout). The train-side
log-prob computation divides logits by `args.rollout_temperature` to match
rollout-time log-probs (`loss.py` `get_responses`, and the fused path at the
second temperature-scaling site). Division by 0 gives Inf logits →
`log_softmax` NaN → **NaN loss** → every grad NaN → Megatron's grad check
raises "NaN in local grad norm for bucket #0" on the loss-computing stage
(node69 = PP stage 1). The passing debug-train-only replay (run.r3) never
passes `--rollout-temperature`, so it defaults to 1.0 and skips the division —
that config difference, not the full-loop execution, was the real
pass/fail discriminator.

Decisive evidence (attempt35_synced, first run with diagnostics actually
deployed on stage 1): `[SLIME_DEBUG_LOSS] loss=nan finite=False
logits_absmax≈55 num_tokens=16 mask_sums=[16]` on every stage-1 rank — the
logits entering the loss are healthy and the masks are correct, yet the loss is
NaN, so the NaN is born inside the loss computation; the only zero-guarded
division there is the temperature scaling.

Fix (landed) — fail loudly, and use on-policy sampling:
- `slime_validate_args` (slime/utils/arguments.py) now **rejects
  `--rollout-temperature <= 0` at launch** for training (allowed only with
  `--debug-rollout-only`, which has no train side) — misconfiguration fails in
  seconds instead of NaN-ing 15 minutes in.
- Both temperature-scaling sites in `loss.py` **assert `rollout_temperature >
  0`** as defense in depth before dividing.
- The R6/R4 smokes now default to **`--rollout-temperature 1 --rollout-top-p 1`**
  (proper on-policy sampling; parameterized via `ROLLOUT_TEMPERATURE`/
  `ROLLOUT_TOP_P`). The historical temp-0 setting was a smoke-determinism choice
  that is invalid for training.
- Unit tests: `tests/test_rollout_temperature_zero_guard.py` (6 tests: the
  loss-path assertion fires at 0, validate_args rejects 0 for training and
  allows it for rollout-only debug, positive temperatures still scale).
  Note: `tests/test_megatron_argument_validation.py` fails pre-existing
  (ImportError on relative import, unrelated — verified by stash-check).

Codex (xhigh) review status: the review job verified the failing-log NaN
pattern, confirmed the two patched sites are the only train-side
`rollout_temperature` divisions (swept sglang_rollout, streaming rollout,
ppo_utils, on_policy_distillation, eval_config), and validated the evidence
trail — then the codex broker wedged mid-synthesis (twice; a known failure
mode), so no final formatted verdict was returned. Combined with the
definitive on-policy PASS (attempt37, fresh sampled rollouts through the fixed
path), the fix is treated as reviewed-with-caveat rather than formally
sign-off'd.

What was ruled out along the way (all with evidence, all still true): the V4
TileLang kernels (torch fallbacks NaN'd identically), param corruption from
`update_weights` (params finite; grad buffers zeroed in `train_one_step` right
before the backward), the mHC Sinkhorn backward (DeepSeek's mixing-oracle
detach — implemented as `V4_MHC_MIXING_ORACLE`, default on, in
`custom_kernels/deepseek_v4/megatron/mcore_model.py` — is a sound stability measure per
DeepSeek's recipe but was not the cause), and forward overflow (stage-1
activations grow smoothly to ~7e4 absmax at layer 42 — bounded and finite; the
growth pattern is intrinsic to the checkpoint's mHC gates).

DeepSeek stability recipe (from the LMSYS "DeepSeek-V4 Day-0 RL" blog; recorded
for the production run even though it was not this NaN): freeze the mHC
Sinkhorn as a forward-only mixing oracle (done, `V4_MHC_MIXING_ORACLE=1`),
freeze the MoE router gate + per-expert score-correction bias + hash-routed
early layers (LoRA already freezes them as params), switch the compressor
backward all-reduce to FP32 (TODO), pin deterministic ops (~10-15% throughput,
optional). Sources: lmsys.org/blog/2026-04-25-deepseek-v4/, arXiv 2512.24880
(mHC), arXiv 2606.19348 (V4 report), arXiv 2409.19606 (Hyper-Connections).
Caveat: the mixing-oracle detach fully skips the Sinkhorn VJP only on the torch
mHC path; the TileLang `HyperConnectionFn.backward` still computes the full VJP
unconditionally (`_middle_bwd_c`), so a kernel-level oracle mode remains a
follow-up for training-at-speed.

## CRITICAL: Code Edits Do Not Propagate Across Nodes

`/nfs/FM` is per-node-local storage: editing the repo on node64 does NOT change
node69/62/70's copies; a multi-node run executes each node's LOCAL copy. This
silently invalidated several diagnostics (they only ran on node64/PP stage 0)
until caught via an md5 probe. Env vars DO propagate (`--train-env-vars`, Ray
runtime_env); only code does not.

**Rule: after ANY code edit, `rsync -a slime custom_kernels scripts tests
train.py node6X_slime:$REPO/` for X in 69,62,70, and verify md5 before
running.**

Diagnostics left in place (all env-gated, default off unless noted):
`SLIME_DEBUG_CHECK_PARAMS` (per-rank param finiteness + value checksum, actor),
`SLIME_DEBUG_LOSS` (per-microbatch loss/logits/mask print, loss.py),
`SLIME_DEBUG_ANOMALY` (autograd anomaly mode, actor), `V4_DEBUG_FWD_CHECK`
(decoder activation absmax probes, mcore_model), `V4_MHC_MIXING_ORACLE`
(default ON, mcore_model). A former `SLIME_DEBUG_SKIP_UPDATE_WEIGHTS` diagnostic
was removed — skipping update_weights breaks the gloo group.

## Hard Constraints

- Megatron `--load` must use the converted `torch_dist` checkpoint, not the raw
  HF checkpoint.
- The optimizer must be Megatron Muon. The active command path uses
  `--optimizer muon` and Megatron's `/root/Megatron-LM/megatron/core/optimizer/muon.py`.
- Do not write a new MoE communication stack. Use Megatron flex dispatch with
  DeepEP for train-side expert parallel communication.
- With only four H20 nodes currently available, reserve at least one node for
  rollout. Do not use PP2/EP16 across all four nodes.
- If any kernel changes, rerun forward and backward efficiency measurements and
  update `handoffs/deepseek-v4/dsv4_kernel_inventory.md`.

## Current Runtime Target

| Role | Nodes | Checkpoint / base | Status |
| --- | --- | --- | --- |
| Train | node64 + node69 | `DeepSeek-V4-Flash-FP8-r2-pp2-ep8-torch_dist` | R6 full loop PASSED (attempt37, on-policy) |
| Rollout | node62 | raw HF `DeepSeek-V4-Flash-FP8` | TP4 rollout in the passing loop (TileLang MHC + cuda-graph on) |
| Rollout fallback only | node70 | raw HF `DeepSeek-V4-Flash-FP8` | Works ONLY with torch MHC fallback + `SGLANG_DISABLE_CUDA_GRAPH=1` (sm90a TileLang load issue); not for kernel-on train |
| Future scale-out | up to 6 H20 | PP3/EP8 ckpt converted + verified | Revisit EP/PP split when extra nodes land |

## Proven Gates

| Gate | Result | Evidence |
| --- | --- | --- |
| V4 mcore model, LoRA wrapping, SFT sanity | Passed | `custom_kernels/deepseek_v4/megatron/*`, `tests/test_v4_model_provider.py` |
| HF-native FP8 to Megatron conversion | Passed | audit evidence removed in cleanup; re-derivable via `native_checkpoint.py --output-json` |
| PP2/EP8 Megatron checkpoint conversion | Passed | verify evidence removed in cleanup; re-verify any checkpoint via `scripts/dsv4/convert_torch_dist.sh` chained verify or `verify_torch_dist.py` standalone |
| Real-weight HF vs mcore parity samples | Passed | parity evidence removed in cleanup; re-run `real_weight_parity.py` (layers 0/2/3 all passed) |
| Kernel-on PP2/EP8 debug-train smoke | Passed | `r2_logs/r4_pp2_ep8_kernel_on_train_smoke_attempt2.log` |
| Node62 SGLang rollout smoke with routed experts | Passed | `r2_logs/r4_node62_rollout_smoke_attempt16.log` |
| R5 PP2/EP8 replay train smoke with Muon | Passed | `r2_logs/r5_pp2_ep8_replay_train_smoke_attempt10.log`, `r2_logs/r5_attempt10_train_debug_audit.txt` |
| R6 full loop (rollout → weight sync → train, Muon, routing replay) | **Passed** | Definitive: `r2_logs/r6_pp2_ep8_full_loop_attempt37_onpolicy.log` — on-policy sampling (`temperature=1, top_p=1`), `train/loss=0.180, grad_norm=0.689` (finite; per-rank probe 2.95/16 tokens ≈ 0.18, consistent), in-script dump verify `ROLLOUT_SMOKE_PASS routed_experts_present=true`, and the script's own `=== R6 V4 full-loop smoke PASS ===` marker. Earlier greedy-config pass: attempt36_tempfix (`loss=0.0743, grad_norm=0.344`, dump verified manually with `--expected-samples 8`). Per-rank train dumps `/nfs/FM/csl_v4r6_full_loop/debug/train_0_*.pt` |

The R4 rollout audit shows hash-routed layers 0, 1, and 2 produce zero routed
expert captures while learned top-k MoE layers 3 through 42 are nonzero. That is
the expected replay contract.

## Root Cause of R6 Driver Death: Concurrent-Run Fratricide

The R6 driver dies because **another agent session runs the same R6 script
concurrently on overlapping nodes**, and its `ray stop --force` kills this run's
Ray processes cluster-wide. `ray stop --force` sends SIGTERM (then SIGKILL) to
every Ray process on a node, including a connected driver, so the driver and its
rollout actors die together with no application traceback.

Failure chain (identical across attempts 13, 21, 22, 23, 24):

1. This run reaches `train: waiting for rollout metrics router address` and the
   driver blocks in `ray.get(...)`.
2. Another R6 run (observed: `RUN_ID=attempt32` launched from a different codex
   session on node69/node62/node70) starts or cleans up and runs
   `ray stop --force` on the shared physical hosts, arriving over ssh
   (`sshd: root@notty`).
3. The driver process receives SIGTERM (glog: `*** SIGTERM received ***` in
   `ray::core::CoreWorker::GetObjects`) and aborts; the node62 rollout actors go
   ALIVE→DEAD in the same instant.

Evidence (all ruled out as *not* the cause): host/cgroup/kernel OOM (node64 has
~1.6 TB free, `memory.max` unlimited, `oom_kill 0`, dmesg empty), Ray
memory-monitor (threshold never tripped), slime self-kill (the only kill in
slime is `kill_process_tree(self.process.pid)` scoped to sglang's own child),
GPU-usage trigger (a bare torch GPU holder survived 120 s), and process-name
trigger (sentinels named `train.py`/`sglang`/`ray::` survived 60 s). The
positive proof is a 0.2 s process snapshot of node64 during attempt24: an
external `/usr/bin/python3 /usr/local/bin/ray stop --force` runs on node64 at
the exact SIGTERM instant, and a concurrent `attempt32` R6 run is visible in the
same snapshot (preserved:
`r2_logs/r6_attempt24_fratricide_proc_snapshot.txt`).

Fix is coordination, not code: run only one R6 at a time cluster-wide (exclusive
nodes, or a cross-session lock). A per-node `flock` does not help — `/nfs` is
per-node-local, so each container's lock file and repo copy are independent.

## Fixes Already Landed

- SGLang LoRA-only weight update now keeps each DeepSeek-V4 compressor `wkv` /
  `wgate` pair in the same distributed update chunk.
- Update-weight failures now include group name, tensor count, and sample tensor
  names before releasing rollout engine locks.
- Ray dashboard agent startup is patched by
  `scripts/patch_ray_dashboard_agent_early_port.py`; it writes the listen-port
  file early enough to avoid Raylet dashboard-agent port wait timeouts.
- R6 launch uses direct-driver mode, fixed Ray dashboard agent ports, a run lock,
  converted Megatron checkpoint load, Megatron Muon, and PP2/EP8 train with a
  separate TP4 rollout node.
- Train-actor `megatron.training` import fix: the R6 script passes
  `--train-env-vars` carrying `PYTHONPATH=${REPO}:/root/Megatron-LM` (plus the
  `V4_*`/TILELANG vars) into the actor runtime_env. Root cause of attempt20's
  `ModuleNotFoundError: No module named 'megatron.training'`: `megatron.core` is
  an editable install exposing only `megatron.core`; `megatron.training` lives
  only in `/root/Megatron-LM`, and a Ray actor whose raylet PYTHONPATH lacked it
  imported `megatron.core` but not `megatron.training`. Confirmed with a
  standalone Ray actor test. `slime` is editable-installed pointing at a
  different repo and `custom_kernels/deepseek_v4` exists only in the current repo, so the
  actor PYTHONPATH must list the current repo first, then `/root/Megatron-LM`.
- Cleanup safety: the EXIT-trap orphan pkill (`CLEANUP_KILL_ORPHANS_ON_EXIT`)
  now defaults OFF, because `kill_old_processes` uses broad unscoped
  `pkill -9 -f` (sglang/ray::/train.py) that kills other tenants on the shared
  box. It must be scoped to this run (PGID/env marker) before defaulting on.

Validation already run for these fixes:

- `python3 -m pytest -q tests/test_deepseekv4_megatron_to_hf.py`
- Standalone Ray actor test: env_vars `PYTHONPATH=/root/Megatron-LM` →
  `megatron.training` imports; without it → `ModuleNotFoundError` (reproduced).
- `bash -n scripts/dsv4/_dsv4_launch_core.sh`
- Codex (xhigh) reviewed the diagnosis and the fixes; it verified the PYTHONPATH
  fix and flagged the cleanup pkills as unsafe on the shared box (now defaulted off).

## Operational Notes

- Attempt16 was manually terminated after the SGLang crash. Its Ray cluster,
  orphan `train.py`, and node70 orphan SGLang children were cleaned up.
- The attempt13 node62 SGLang scheduler/detokenizer leftovers were also killed.
  After cleanup, node62/node64/node69/node70 showed no active GPU compute apps.
- The R6 script's EXIT cleanup currently stops Ray but can leave orphan
  `train.py` or SGLang child processes after hard failures. Patch this before
  long unattended runs.
- The full-loop log from attempt16 was produced inside the node69 container and
  copied back as `r6_pp2_ep8_full_loop_attempt16.remote.log`.
- The worktree is still intentionally dirty with implementation, tests, scripts,
  current evidence, and the V4 kernel package. A separate cleanup pass is still
  needed before commit/review; it was not performed in this handoff-only update.

## Formal RL Training (2026-07-03)

Goal: DeepSeek-V4-Flash LoRA RL, hyperparameters borrowed from
`examples/kernel_agent/run.t1.qwen3.6.27B.fasync.sh`. MoE is FROZEN (attention/
compressor LoRA only) — Megatron-Bridge *does* support expert/router LoRA
(`GroupedExpertLinearAdapter`, `LoRATopKRouter`), but our experts are raw
`nn.Parameter` grouped GEMMs (`V4GroupedExperts`), not `linear_fc1/2` submodules,
so `is_expert_linear` never matches them — freezing is the validated path.

Launcher: `scripts/dsv4/run.t1.deepseek_v4_flash.rl.sh` -> the R6 smoke in
`TASK_MODE=rl`. Task-arg assembly extracted to `scripts/dsv4/_dsv4_task_args.sh`
(`build_dsv4_task_args`), unit-tested in `tests/deepseek-v4/test_dsv4_rl_task_args.py`.
Borrowed hyperparams: advantage-estimator trloo, eps-clip 0.2/0.28, entropy 0,
gbs 256 (rollout 16 x n-samples 16), constant LR, wd 0.01; V4-mandatory kept:
Muon, PP2/EP8, custom provider, torch_dist load, routing replay.

`REWARD_MODE`: `random` isolates the RL math from KernelGym (Gate A);
`drkernel` wires the real multi-turn CUDA-agent task (KernelGym healthy on
node62/64). Staging: Gate A (RL-math finite-grad) -> Gate B (scale + ckpt/resume)
-> formal.

Codex (xhigh) review findings on the launcher (2026-07-03), status:
- FIXED: SGLang context vs response — V4 SGLang strictly rejects
  prompt+new_tokens > context (qwen's SGLang silently clamps). Now MAX_CONTEXT_LEN
  is the total window, MAX_RESPONSE_LEN defaults to half, and
  `--rollout-max-prompt-len = MAX_CONTEXT_LEN - MAX_RESPONSE_LEN` is pinned.
- OK-for-gates: trloo+random skips `_post_process_rewards` group-norm (that path
  is grpo/rloo only; trloo does leave-one-out at advantage time) — random reward
  still yields finite non-degenerate advantages for the Gate-A finite-grad check;
  the real trloo normalization runs via drkernel's `reward_post_process_by_group`.
- TODO before formal run: (a) forward `WANDB_API_KEY` into RUNTIME_ENV_JSON +
  TRAIN_ENV_VARS_JSON (USE_WANDB=1 else logs silently fail auth); (b) for Gate-B
  resume, drop `--no-save-optim/--no-load-optim` only after confirming Muon
  optimizer-state save/resume works; (c) data-key defaults couple to REWARD_MODE
  (fine for toy-jsonl-random / parquet-drkernel, fragile otherwise) — pass keys
  explicitly for off-nominal combos; (d) strengthen `test_smoke_sft_unchanged`
  to an exact TASK_ARGS array match.
- OK: temperature-0 NaN guard intact (defaults > 0, validator rejects <= 0).

## Gate A PASSED — RL math works on V4 (2026-07-03)

`REWARD_MODE=random` RL-math smoke (policy_loss + trloo, ctx 1024, 2 rollouts)
ran end-to-end: step0 loss -0.030 grad_norm 0.106, step1 loss -0.558 grad_norm
0.236, weight sync between them, rollout↔train logprob abs-diff ~0.03 (the fixed
log-prob PP forward is numerically consistent). Evidence:
`r2_logs/gateA_rl_math_PASS_evidence.txt`.

Three V4+PP RL-path bugs found+fixed getting here (all only exercised by the RL
log-prob/train forward, never the SFT smoke; rollout worked every attempt):
1. `forward_only`/`compute_log_prob` didn't pass `adjust_tensor_shapes_fn` -> mHC
   got 3-D not 4-D. Fixed in `model.py` (both forward paths now wire it;
   `test_v4_pp_shape_adapter.py` guards both).
2. SGLang context vs response length rejection -> total-budget windows +
   `--rollout-max-prompt-len` (`_dsv4_launch_core.sh`, `_dsv4_task_args.sh`).
3. V4 PP shape adapter hardcoded S from static `args.seq_length`, but slime uses
   `variable_seq_lengths` and pads each rollout to a per-step `max_seq_len` (~384)
   while RoPE used the real length -> RoPE shape clash. Fixed: `actor.py` stashes
   `max_seq_len` on `args._v4_pp_current_seq_len`; `model.py`
   `_v4_current_pp_seq_length` reads it for the adapter AND both
   `forward_backward_func` seq_length args (gated to V4+PP). Codex (xhigh) audit
   verdict: **fix correct for the bshd RL path; no remaining PP shape/seq-length
   crash bug**, multiple-microbatch safe.

Known non-crash item (codex): `trloo` is excluded from the built-in reward
normalization (`ray/rollout.py:640`) and maps to `get_grpo_returns`
(`loss.py:650`), so `trloo` WITHOUT a custom reward postprocess yields raw
(un-leave-one-out) advantages. Fine for Gate A (finite-grad check only). The
formal drkernel run sets `--custom-reward-post-process-path
reward_post_process_by_group` (same as the qwen ref) which bypasses that branch
and normalizes — VERIFY that postprocess does the intended group normalization
before the formal launch.

## Gate B scale PASSED — RL trains at 4x context (2026-07-03)

`REWARD_MODE=random`, ctx 4096 / response 2048, rollout-batch 4 x n-samples 4 =
gbs 16, 2 rollouts. Both RL steps finite (step0 loss -0.438 grad_norm 0.086,
step1 loss -0.576 grad_norm 0.272), weight sync between, ~17s/step, GPU 45/95 GB
(comfortable headroom), train↔infer logprob abs-diff 0.024-0.033 held at scale.
Evidence: `r2_logs/gateB_scale_PASS_evidence.txt`.

4th RL-path bug (pre-existing, first REACHED here because earlier fixes cleared
the forward): `fill_routing_replay` (actor.py) padded each microbatch's recorded
routing only to `pad_size`, but `get_batch` pads tokens to the rollout-wide
`max_seq_lens[0]`; short samples then had too few routing rows vs the padded
tokens ("replayed indices [128,6] vs scores rows 256"). Fixed: the bshd branch
now pads each sample's routing to `max_seq_lens[0]` via
`slice_with_cp(..., "bshd", max_seqlen)` (matching get_batch). Guard:
`test_v4_routing_replay.py::test_fill_routing_replay_pads_bshd_to_max_seqlen`.

Gate B checkpoint/resume is SPLIT OUT to the adapter-only-save task: a full V4
`--save` is ~259 GB (frozen 284B base) and `/nfs/FM` runs near-full, so periodic
full checkpoints are unsustainable. Megatron-Bridge PEFT has adapter-only save
(`peft/base.py` `adapter_key_filter`) but slime uses the standard full-model
`megatron.training.checkpointing.save_checkpoint`; wiring adapter-only save +
split resume (base from `--load` torch_dist, adapters from `--save`) is the fix.
Disk was unblocked by reclaiming my own stale R3 smoke checkpoint (csl_v4r3,
~531 GB across node64+69); node64 97 GB / node69 253 GB free now.

## Next Actions

0. FORMAL-RUN BLOCKERS (do first): (a) wire adapter-only checkpoint save/load
   (259 GB full saves are unsustainable) — MAIN blocker; OR reclaim my own unused
   pp3/ep8 converted ckpt (~260 GB) for infrequent full checkpoints.
   (b) DONE/VERIFIED: drkernel `reward_post_process_by_group`
   (examples/kernel_agent/kernel_reward.py:16) DOES implement the trloo
   leave-one-out baseline (arXiv 2402.14740) — the formal run gets correct
   advantages despite the built-in `_post_process_rewards` skipping `trloo`.
   (c) confirm KernelGym-on-H20 reward wiring with a small `REWARD_MODE=drkernel`
   smoke before the full run (KernelGym healthy on node62/64; custom fns present).
1. Confirm attempt36_tempfix passes (finite loss, `R6 V4 full-loop smoke PASS`,
   routing-replay dump verified), then dispatch a codex (xhigh) review of the
   loop result + the temperature fix before declaring the R6 gate passed.
2. Fratricide guard for future runs: only one R6 at a time cluster-wide (kill
   any leftover `codex --yolo` sessions first: `pgrep -x codex`); longer-term,
   scope `kill_old_processes` to this run's PGID and add a shared-resource lock
   (TCP port mutex — NOT an `/nfs` file, it is per-node-local).
3. Production hardening from DeepSeek's recipe: port the mHC mixing-oracle into
   the TileLang `HyperConnectionFn.backward` (skip the Sinkhorn VJP when
   detached), and switch the compressor backward all-reduce to FP32.
4. Preferred full-loop target: node64+node69 PP2/EP8 train, node62 TP4 rollout
   (native TileLang kernels + cuda-graph). node70 rollout only with the torch
   MHC fallback + `SGLANG_DISABLE_CUDA_GRAPH=1`.
5. Remember the sync rule: rsync code to all `_slime` nodes + md5 verify after
   every edit (see the CRITICAL section).

## FORMAL RUN TRAINING (2026-07-05) — v7 healthy at step 0

`formal_v4_lora_rl_v7_20260705_003133`: ctx 8192, gbs 256 (16x16), trloo,
lr 1e-5, LoRA dim 16, adapter-only 22MB ckpts q20 steps, wandb
`FAsync.tvm_ffi.DeepSeek-V4-Flash.CTX8192`... step 0: loss 0.0443, entropy
0.381, pg_clipfrac 9.5e-05, ppo_kl 2.7e-06, logprob_diff 0.0271, grad_norm
0.0034. Iteration ~33 min (rollout 28 min + step ~5 min, ~44 iters/day).
The v1-v6 fix stack that got here:
1. `NCCL_IB_DISABLE=1` everywhere (fabric workaround, below) — also made the
   pipeline ~100x faster (log-prob 1.81s/mb vs 120-250s/mb on the sick fabric).
   Env must reach engines via the per-node `ray start` env (direct mode ignores
   RUNTIME_ENV_JSON) — v4/v5 died on mixed-transport bootstrap until that fix.
2. `--distributed-timeout-minutes 120` (COMMON_ARGS).
3. `V4_ACT_CKPT=1` — activation checkpointing wired into the V4 decoder loop
   (mcore_model.py, use_reentrant=False, training+grad-gated): v6 OOM'd stage-0
   in the train backward (1F1B holds pp_size mbs of activations). Compatible
   with routing replay BY DESIGN (recompute runs under replay_backward ->
   pop_backward; actor/model stage management verified).

## ROOT-CAUSED: formal-run NCCL deaths = flaky RoCE fabric (2026-07-04)

Formal v1 (ctx16k), v2 (ctx8k), v3 (ctx8k + 120min PG timeout) all died in the
log-prob phase with NCCL "remote process exited or network error". v3's
NCCL_DEBUG=WARN exposed the primary error:
`NET/IB: Got completion from peer 10.11.2.169<50673> with status=
IBV_WC_RETRY_EXC_ERR(12) ... req_type=Send hca mlx5_5` — an RDMA retry-exceeded
hard failure on the RoCE link node64->node69. Formal-scale sustained PP
transfers (~256MB per microbatch per stage boundary) trip the flaky fabric;
the small smokes never generated enough traffic. This also retroactively
explains the "ctx-16384 deadlock" below (same transport failure, generic
error text without NCCL_DEBUG).

WORKAROUND in force (v4+): `NCCL_IB_DISABLE=1` — inter-node NCCL over TCP/bond0.
Cost is negligible for this workload: inter-node traffic is only PP p2p
(0.25s/transfer at TCP speeds vs 60-250s/mb compute) + tiny LoRA-grad
allreduces; the EP all-to-all is intra-node NVLink and unaffected.
ROOT CAUSE (open, infra-level): the RoCE fabric (hca mlx5_5 at least,
node64<->node69 path) drops RDMA sends under sustained load — needs fabric
diagnosis (PFC/ECN config, port errors `ibstat`/`ethtool -S`, cable). Keep
`--distributed-timeout-minutes 120` + `NCCL_DEBUG=WARN` until stable.

## (superseded by the above) OPEN: ctx-16384 PP p2p deadlock (2026-07-04)

The first formal attempt (ctx 16384, gbs 256) rolled out fine (256 samples,
22 min, correctness 0.52) but DEADLOCKED in the first `compute_log_prob`
forward: all 3 PP stages simultaneously stuck in p2p (`send_forward` on stages
0+1, recv `cuda.synchronize` on stage 2) until the NCCL watchdog (600 s) fired
SIGABRT on multiple ranks. No OOM, no CUDA error. All prior end-to-end passes
were ctx <= 8192; 16k is outside the validated envelope. Ruled out: DP
microbatch-count desync (`build_dp_schedule` equalizes counts via
`get_seqlen_balanced_partitions(equal_size=True)`), TileLang compile stall
(all train-actor compiles completed 26 min before the hang). Evidence:
`r2_logs/formal_v4_lora_rl_20260704_200050.driver.log` (~line 221189+, watchdog
at 12:03:24). LIKELY explanation (from the ctx-8k v2 run): first-microbatch
latency, not a true deadlock — at 8k the first log-prob microbatches take
125-250s each (TileLang JIT warmup for the new padded shape); at 16k the first
microbatch would exceed the 600s NCCL watchdog on stage-2's parked recv ->
SIGABRT. Fix candidates for 16k: raise the ProcessGroup/NCCL timeout for the
log-prob phase, or pre-warm the 16k-shape kernels before the pipeline starts. WORKAROUND in force:
formal runs pinned to ctx 8192 (`MAX_CONTEXT_LEN=8192`), which covers DrKernel
response lengths; 16k needs this investigation before use.

## PP3 topology + node70 fix + full drkernel pass (2026-07-04)

ALL formal-run blockers cleared. Full drkernel RL step end-to-end on the new
topology: step0 loss 1.664, grad_norm 0.031, **logprob_diff 0.0172** (tighter
than the random smokes — routing replay + kernel-on rollout works), run
finished OK. Evidence: `r2_logs/pp3_drkernel_full_PASS_evidence.txt`.

1. **PP3 topology (user's plan)**: TRAIN node62+64+69 PP3/EP8 = 24 GPU (fixes
   the real-seq activation OOM: base 31GB/GPU vs 40 at PP2), ROLLOUT node70.
   slime sorts actor ranks by node IP (`placement_group.py sort_key`:
   162<164<169), so the PP3 shards were chain-moved (byte-verified) to
   node62=__0-7/stage0, node64=__8-15, node69=__16-23; node70 keeps a backup.
   `TOPOLOGY=pp3` is the run.t1 launcher default. Random-smoke evidence:
   `r2_logs/pp3_topology_PASS_evidence.txt`.
2. **node70 "sm90a TileLang issue" ROOT-CAUSED — poisoned JIT cache, not
   hardware**: node70 alone has pip `nvidia/cu13` nvcc shadowing
   `/usr/local/cuda` without a PATH pin; cubins it compiled can't load on the
   570 driver (`cuModuleLoadData` failures) and poisoned `/root/.tilelang`
   persistently (cache keyed by source hash, not compiler). Proof: fresh-cache
   compile+load of the exact crashing kernel (mhc `_build_norm_gemm`) succeeds.
   Cache quarantined (`/root/.tilelang.poisoned.bak`); kernel-on rollout now
   runs on node70 at 600-5900 tok/s (vs 43 fallback). Residual risk: any
   process without the PATH pin can re-poison — launchers pin PATH; consider
   removing the cu13 package or adding the qwen launcher's compiler preflight.
3. **All-zeros train step — CORRECTED after codex review (my first explanation
   was WRONG)**: I initially blamed `--enable-turns-dp-partitions` at
   max_turns=1 and gated it off — but (a) slime's `slime_validate_args` RAISES
   when `sequence_mis_aggregation=turns_geometric` lacks that flag (the gate was
   launch-breaking, never exercised), and (b) a later HEALTHY run (loss 1.664,
   grad 0.031) ran with the same flag on and max_turns=1, refuting the theory.
   The observed rank16-real/ranks17-23-pad distribution is the flag's normal
   layout, not a pathology. The zeros were a batch-composition edge (most
   likely exact-degenerate advantages in that batch); treat rare zero steps as
   a monitoring item, not a config bug. The gate was REVERTED
   (`--enable-turns-dp-partitions` is unconditional in drkernel mode + test
   pins it present alongside turns_geometric).

## DrKernel reward integration (2026-07-04)

The full DrKernel/KernelGym task works on the V4 rollout (REWARD_MODE=drkernel):
real kernel generation, compilation 0.75, correctness 0.67, speedup mean 1.27 /
max 3.35, fast@1 0.375, precheck 1.0, dynamic-sampling filter + coverage + TIS
all functional. Evidence: `r2_logs/drkernel_reward_works_evidence.txt`.

Real-scale memory gap (found with routing-replay OFF): the drkernel train step
OOMs at ctx 8192 (GPU 94.9/95 GiB). Megatron's `--recompute-*` args are a NO-OP
for V4 — its `mcore_model.py` uses a hand-written `for layer in self.layers`
forward loop, not a Megatron TransformerBlock, so activation checkpointing is not
wired. Real-scale RL needs activation checkpointing added to the V4 decoder loop
(wrap each `V4DecoderLayer` in `torch.utils.checkpoint`) OR reduced batch/context.
Also worth checking: whether the DDP grad buffer / DistributedOptimizer allocates
for the frozen base (it seems to — the 276GB optim save), which would waste
~tens of GB/GPU that a LoRA run does not need. Both are pre-formal-run follow-ups.

Open gap for the formal run: V4 routing-replay needs `rollout_routed_experts`,
which slime's DEFAULT sglang rollout captures (`sglang_rollout.py:288`
`return_routed_experts=True`, `:332` decode) but the DrKernel custom multi-turn
generate (`examples/kernel_agent/generate_with_cuda_agent.py`) does NOT →
`ValueError: rollout_routed_experts is required ... use_rollout_routing_replay`
in `fill_routing_replay`. RESOLVED via option (A), guided by upstream THUDM/slime
`1b73ddc1` (routed-experts-in-meta_info shape contract, rows = len(tokens)-1):
`generate_with_cuda_agent.py` now (1) requests `return_routed_experts` in the
per-turn payload when `--use-rollout-routing-replay` is set, (2) decodes
`meta_info["routed_experts"]` into each turn sample via the standard
`_decode_routed_experts`, (3) gives pad turns an empty 0-row routed array (their
`len(tokens)-1 == 0`). Tests: `tests/deepseek-v4/test_v4_drkernel_routing_replay.py`.
Option (B) `USE_ROLLOUT_ROUTING_REPLAY=0` remains available as a fallback toggle.
Upstream note: `5c530c15` (allgather_cp refactor of fill_routing_replay) is NOT
ported — we run CP=1 and it lacks our bshd max-seq-len padding fix; reconcile on
the next upstream rebase.

## Task #11 DONE — adapter-only save + resume validated (2026-07-04)

Full save→resume loop validated at MB scale. A full V4 save is ~259GB; the
adapter-only save is **22 MB** (0.0265% — 244 adapter keys / 10.86 MB per rank of
the 41 GB full model dict). Resume RE-LOADS the adapters and CONTINUES training:
`V4 LoRA adapter resume: loaded adapters ... at iteration 1` → step0 loss -0.037
grad_norm 0.119 → `run finished OK`.
Positive `--lora-dim` writes MB-scale LoRA checkpoints;
`--lora-adapter-resume-load <adapter ckpt dir>` overlays adapters on the
cold-loaded base and continues from the checkpoint iteration.

Three per-node-`/nfs` issues fixed along the way (torch_dist writes shared files
from global rank 0 only, but /nfs is per-node): (1) filter was on the DDP wrapper
not the unwrapped model (below); (2) `latest_checkpointed_iteration.txt` missing
on non-rank-0 nodes → resume routed to the HF loader — fixed by
`write_latest_marker_per_node`; (3) `.metadata`/`common.pt`/`metadata.json`
missing on non-rank-0 nodes → "unknown checkpoint format" — fixed by
`replicate_ckpt_metadata_per_node` (broadcasts the small metadata from rank 0 to
all nodes). Both run after every adapter save in `model.py::save`.

Root cause of the earlier 259GB failures (each filled the disk): the filter was
wrapping the wrong object. `megatron.training.checkpointing.save_checkpoint`
calls `unwrap_model(model)` BEFORE `generate_state_dict`, so it invokes the
UNWRAPPED module's `sharded_state_dict`; wrapping the DDP-wrapped `model[i]` was
bypassed and the full base was written. Fix: `adapter_ckpt.py` wraps
`unwrap_model(chunk).sharded_state_dict` (both save and resume paths). The
"flat-buffer view" theory was WRONG — the adapter ShardedTensors are genuinely
tiny once the filter actually runs. Implementation:
`slime/backends/megatron_utils/adapter_ckpt.py` (`adapter_only_model_save/load`,
requires_grad-keyed filter, with size validation that aborts
before write), wired in `model.py::save` and `actor.py` (`load_adapter_resume`,
init overlay). Muon optimizer-save stub crashes fixed en route
(`checkpoint.py` patches `ChainedOptimizer._synchronize_steps` +
`Float16OptimizerWithFloat16Params.{state_dict,sharded_state_dict}` for
param-less stub sub-optimizers). Tests: `tests/deepseek-v4/test_v4_adapter_ckpt.py`.
Note: SAVE_OPTIM=1 (Muon momentum) still writes the full ~276GB DistributedOptimizer
buffer — adapter checkpoints are weights-only (cold Muon momentum on resume,
re-warms in a few steps); filtering the optimizer state is a follow-up.

## (superseded) Task #11 implementation plan — adapter-only checkpoint save/resume

Execution-ready (injection points located):
- **Adapter filter**: trainable params = LoRA adapters (base frozen). Build the
  filter like Megatron-Bridge `peft/base.py:230 adapter_key_filter` /
  `set_params_to_save` (key in trainable set, or `.adapter.`/`.adapters`).
- **Save (tiny)**: slime `checkpoint.py` wraps `megatron.training.checkpointing.
  save_checkpoint` -> `generate_state_dict` (checkpointing.py:877) ->
  `model[i].sharded_state_dict()` (:903). Inject by filtering the model's
  `sharded_state_dict` to adapter keys during save (drop `_extra_state`, per
  bridge `apply_peft_adapter_filter_to_state_dict`, checkpointing.py:2594). The
  implemented contract now enables this automatically whenever `--lora-dim > 0`.
- **Resume (two-source)**: base still loads cold from the torch_dist `--load`
  (existing path). Then overlay adapters via the `load_other_checkpoint`
  (`actor.py:740`) pattern (no_load_optim/rng, finetune=True) pointed at the
  adapter `--save`, with the model's `sharded_state_dict` filtered to adapter
  keys so Megatron's load tolerates the missing base keys. Muon optim state for
  the (tiny) trainable params saves/loads normally (`SAVE_OPTIM`/`LOAD_OPTIM`
  already wired in `_dsv4_launch_core.sh`).
- **Validate**: adapter-only checkpoints are MB-scale so a save+resume smoke
  fits the tight disk. Confirm: (1) `--save` writes MB not 259GB; (2) resume
  restores adapters + Muon momentum and continues finite training.
- Risk: checkpoint integrity — do NOT ship without the save+resume validation
  run. Keep the full-save path as the default until validated.
