# Verify RL training

`generate()` owns the shared timeout/error handling and dispatches by sample role:

- Ordinary kernel (including missing role) → `_generate_kernel_impl`.
- Verify → `_generate_with_verify_impl`, which owns the verify/kernel turn loop
  and calls `_generate_kernel_impl` for each revised kernel.
- Anchor → `generate_anchor` builds the direct prompt, then calls `generate()`
  as a kernel sample with a one-turn budget.

`_generate_kernel_impl` contains the actual model-request, token/logprob and
environment logic. The verify controller also calls it with a verify-role sample
for a single diagnosis response, which bypasses KernelEnv. Ordinary kernels never
enter the verify controller. There is no separate forwarding layer.

`KernelAgentDataSource` draws a failed kernel candidate and expands it into
`n_samples_per_prompt` independent verify samples. Each sample generates a
diagnosis, then generates and evaluates one kernel to score that diagnosis.
`--kernel-verify-max-turns` is a positive even total generation-turn budget (default 2):
2 means `verify → kernel`; 4 means `verify → kernel → verify → kernel`, and so on.
Each next diagnosis receives the previous conversation, kernel response and fresh
execution feedback, using the configured verify prompt. An environment completion
signal or invalid diagnosis/scoring pair stops the sequence early.
For the next kernel request, extract exactly one non-empty `<VERIFY>...</VERIFY>`
block outside `<think>` reasoning. Insert it into the existing response template's
`feedback` (also available as `feedback_dict.verification`) without modifying the
verify sample's response, tokens or logprobs. Preserve the current user verification
request (P1), append the full verify response as an assistant message (V1), then
append the kernel repair request (P2) with the extracted block in its feedback:
`P1 → full V1 → P2 → kernel`. Extraction affects only P2's feedback, not V1 in the
conversation. Later pairs reuse this complete kernel prompt history.
Missing, empty, repeated or malformed VERIFY blocks mask the verify sample with
`remove_reason=invalid_verify_format` and do not launch verified kernel scoring.
A shared anchor is kept for the other valid candidates; if no candidate has a
trainable diagnosis left, the group cancels and awaits any unfinished anchor.
Both diagnosis and verified kernel responses enter the actor training batch as
separate samples with their own generated tokens, logprobs and loss masks.
`--use-multi-turn` is required even for a budget of 2. At most
`n_samples_per_prompt * kernel_verify_max_turns` samples enter training per source
group. `metadata.turn_idx` follows generation order: verify 0, kernel 1, verify 2,
kernel 3. Their roles remain `verify` and `kernel`; both carry
`metadata.verify_trajectory = true`.

Online failed-kernel capture is enabled only by `--capture-verify-data`, regardless
of the verify rollout ratio. A positive `--verify-rollout-ratio` without that flag
emits a startup warning but remains valid for fixed-data experiments using
`--load-verify-data`. When the pool has no eligible candidates, sampling falls back
to ordinary kernel prompts. `--save-verify-data` requires explicit capture too.

Verify buffer sampling controls:

- `--verify-samples-per-group` (default 1): maximum failed kernel candidates selected
  from each original source group per model version, before expansion into N
  verify candidates. This limits selection, not capture.
- `--verify-version-lag` (default 2): maximum source model version lag for online
  candidates. Loaded fixed candidates are exempt from this age-based pruning.

The launcher exposes these as `VERIFY_SAMPLES_PER_GROUP` and `VERIFY_VERSION_LAG`.

## Advantage baseline selection

Use `--verify-advantage-baseline {group,history,anchor}` (default `group`). Only
`anchor` creates an extra direct-repair sample. The launcher exposes the same
selection as `VERIFY_ADVANTAGE_BASELINE`; no separate anchor-enable flag is needed.

| Mode | Verify advantage before token-level processing | Extra rollout |
| --- | --- | --- |
| `group` | Configured GRPO/RLOO/TRLOO group-relative estimator on R2 | None |
| `history` | R2 - R1, without group centering or group standardization | None |
| `anchor` | R2 - Ra, without group centering or group standardization | One direct kernel per group |

Here R1 is the initial source kernel's single-turn reward, R2 is the kernel reward
following this diagnosis, and Ra is the fixed shared direct-repair reward. Kernel
turns retain their own execution rewards and existing per-turn group estimator in
all modes. Existing loss masking and token-level training processing still apply.

## Dedicated rollout metrics

Dedicated metrics are automatically included in the existing rollout logger and
configured TensorBoard/W&B sinks when `--verify-rollout-ratio > 0` or the collected
batch contains verify trajectories. No additional logging flag is required, and
`--log-exp-metrics` is not required. Ordinary-only runs with verify disabled keep
their existing metrics unchanged.

The three prefixes separate diagnosis turns (`rollout/verify/`), their generated
kernels (`rollout/verify/kernel/`), and ordinary kernel turns
(`rollout/kernel/ordinary/`). Each reports:

- `reward/{count,mean,median,min,max}`: finite rewards of non-removed, non-aborted
  samples, before reward postprocessing. Verify rewards are R2 in group/history
  mode and R2 - Ra in anchor mode; these are not group-normalized advantages.
- `response_len/{count,mean,median,min,max}`: generated response token lengths,
  including discarded responses. `trainable_response_len/...` uses the effective
  response length (loss mask) of non-removed, non-aborted samples.
- `sample_count`, `valid_sample_count`, `removed_fraction`, `aborted_fraction`,
  and `truncated_fraction`. Kernel buckets additionally report `correctness_rate`
  and `correctness_count` over samples with correctness feedback, excluding decoys
  from successful results.

Additional `rollout/verify/` metrics:

- `group_fraction`: unique verify prompt groups / all unique prompt groups, not
  diagnosis turns / all turns. `group_count` and `total_group_count` give the
  denominator; samples without a group index do not contribute. Empty-pool
  fallback to ordinary data records zero verify groups.
- `format_valid_fraction`: fraction of diagnosis outputs with an extracted VERIFY
  block. `invalid_verify_format/{count,fraction}`,
  `verify_diagnosis_incomplete/{count,fraction}`, and
  `verify_scoring_version_mismatch/{count,fraction}` count the corresponding
  removal reasons. Their denominator is all collected non-pad diagnosis turns.
- `kernel_reward/...` and `scored_fraction`: finite R2 and scoring coverage among
  non-removed, non-aborted diagnoses (coverage denominator: all diagnosis turns).
- `baseline_reward/...` and `improvement/...`: available in history/anchor mode,
  computed directly as R2 - R1 / R2 - Ra. R1 and Ra remain fixed across rounds.
  Improvement also reports `win_rate`, `tie_rate`, and `loss_rate` according to
  whether the difference is positive, zero, or negative. `missing_baseline_count`
  counts scored diagnoses without a finite baseline. Group mode does not report
  an improvement metric because it has no explicit counterfactual baseline.
- `anchor/reward/...`: available settled anchor rewards, deduplicated by the
  shared anchor key across candidates and rounds. In contrast, baseline and
  improvement statistics are weighted per valid diagnosis/kernel pair.

All these metrics describe the batch returned for training, not every attempted
rollout: cancelled, retried or filtered-away groups are outside their scope.
Padding and standalone anchor samples never enter the three turn buckets. Empty
statistics emit a zero `count` but no mean/min/max/median or win/tie/loss rates;
missing and non-finite rewards are omitted. Existing aggregate/turn-index metrics
are preserved; use the dedicated prefixes when comparing the three roles.

## Group baseline (default)

Each verify reward is its immediately following kernel's reward, computed by the
existing kernel reward function. The kernel turn uses that same execution reward
for its own training. The group reward postprocessor computes GRPO/RLOO/TRLOO
advantages separately for each turn index within the source group, never mixing
verify and kernel responses. Neither turn accumulates future rewards, including
under TRLOO, so a kernel reward is not counted twice in a trajectory return.

Assigned rewards on this paired trajectory bypass later dynamic-weight and
failed-group reward rewriting to preserve the pair's utility. Ordinary kernel RL
retains its existing dynamic rewards and trajectory-return behavior. Kernel-only
dynamic filters do not filter either role in a verify trajectory. Verified kernel
outputs are not recursively captured into the verify buffer; ordinary kernel
capture remains unchanged.

```bash
VERIFY_ROLLOUT_RATIO=0.2 \
KERNEL_VERIFY_MAX_TURNS=2 \
CAPTURE_VERIFY_DATA=true \
bash examples/kernel_agent/run_qwen3.6_27B_full_async_dppo.sh
```

For fixed-data ablations, set `LOAD_VERIFY_DATA=/path/to/verify_data` and leave
`CAPTURE_VERIFY_DATA=false` (the default). Loading fixed data does not enable or
disable capture; only the explicit flag controls it.

## History kernel baseline

```bash
VERIFY_ROLLOUT_RATIO=0.2 \
VERIFY_ADVANTAGE_BASELINE=history \
KERNEL_VERIFY_MAX_TURNS=4 \
CAPTURE_VERIFY_DATA=true \
bash examples/kernel_agent/run_qwen3.6_27B_full_async_dppo.sh
```

Capture copies `source.reward` into `metadata.verify_source_reward` and carries it
through all verify rounds. This is the fixed initial R1, not the previous round's
reward, an accumulated trajectory return, or a group-normalized training reward.
Missing/non-finite source rewards are rejected, including when loading fixed data.
Use the same reward configuration for data collection and training; stored scores
are not automatically recalibrated when reward settings change.

Verify samples keep raw reward R2; the reward postprocessor supplies R2 - R1 to
the actor without group centering/standardization. Negative improvements remain
negative and singleton groups retain their signal. No extra kernel is generated.
Version checks still compare the new diagnosis and its kernel; the historical
source kernel is allowed to have an older version under the buffer's age policy.

## Shared fixed direct-repair anchor

```bash
VERIFY_ROLLOUT_RATIO=0.2 \
KERNEL_VERIFY_MAX_TURNS=2 \
VERIFY_ADVANTAGE_BASELINE=anchor \
CAPTURE_VERIFY_DATA=true \
bash examples/kernel_agent/run_qwen3.6_27B_full_async_dppo.sh
```

The corresponding CLI flags are `--verify-rollout-ratio`,
`--kernel-verify-max-turns`, and `--verify-advantage-baseline anchor`.
Use the kernel-agent full-async rollout and
`--custom-reward-post-process-path examples.kernel_agent.kernel_reward.reward_post_process_by_group`
as configured in the launcher.

Every new verify group receives one `metadata.verify_anchor_key`, shared by its
N candidate samples. `KernelAgentDataSource.anchor_kv` holds the separate anchor
sample under that key. It has its own reserved sample index and uses the source
conversation prefix, original reference,
execution feedback, response prompt, sampling parameters, and one kernel generation
turn. Its prompt uses the ordinary response template with the original execution
feedback. It does not include this round's verification request (P1), response (V1),
extracted VERIFY block, NULL text, or verification-analysis heading.

The two initial repair inputs have the form:

```text
verified: shared task/kernel/history → user P1 → assistant full V1 → user P2 (feedback + extracted VERIFY)
anchor:   shared task/kernel/history → user ordinary response prompt (original execution feedback only)
```

The worker atomically claims the anchor input once and submits it before the N
candidate tasks. Generation uses the same concurrency limiter; candidates do not
wait for the anchor while holding generation slots. Submission order does not
guarantee backend execution or completion priority. The group settles rewards after
both the anchor and candidates finish. With deterministic inference the anchor uses
an independent seed, leaving the N candidate seeds unchanged.

The anchor runs exactly one kernel generation/evaluation from the initial source.
All candidates and all later verify rounds reuse that fixed reward. No turn-specific
T1/T2 anchor keys or follow-up anchor prompts are created. Only each candidate's
verified kernel result feeds its next verify round. Both branches use the configured
token limits and existing context-length clipping. The extra anchor generation does
not count toward `--kernel-verify-max-turns` or the N trainable candidates.

KV lifecycle is `pending input → claimed/running → ready result → released`.
Claiming consumes the input, but reading the result is non-consuming so all
candidates can share it. Results are released when the group finishes or aborts;
retrying the group creates a fresh attempt key and regenerates the anchor under
the current policy. These live records are not persisted with captured verify data;
loading that data constructs new groups and anchors. Even a verify-only dataset
mixture consumes queued retry groups before drawing new sources.

The verify reward and pre-token-normalization advantage are
`R_kernel_at_this_pair - R_fixed_source_anchor`. The difference is not group-centered
again, and negative differences remain negative. The verified kernel still trains
on its own `R_verified_final`, with ordinary per-turn group normalization, not on
the anchor difference. Only anchor tokens are excluded from actor training.
There is one extra anchor kernel generation per group, regardless of N or the
number of verify rounds. Later-round rewards measure improvement over the initial
direct-repair baseline, not the isolated contribution of that round's diagnosis.
Pending anchor rewards cannot enter training. `--group-rm` is not supported with
`--verify-advantage-baseline anchor`.

## Asynchronous versions and failures

Inference and environment requests reuse the current rollout process and model
router. A scoring trajectory may cross an asynchronous weight update. The engine
versions reported for the diagnosis and all helper turns are checked; a detected
mismatch marks both the verify and its verified kernel sample removed and zeros
their loss masks. Each sample's rollout/train weight version remains its own
generation version. `verify_scoring_versions_complete` records whether every generation
reported a version; missing version reports cannot establish same-version scoring.

Kernel compilation/runtime/correctness failures receive ordinary kernel rewards.
Missing/aborted scoring trajectories, non-finite anchor rewards or transport
exceptions abort the entire shared-anchor group for retry instead of substituting
a zero-reward anchor. Errors and external cancellation cancel and await all
unfinished group tasks, including their active KernelEnv evaluations. An incomplete
diagnosis is masked and does not launch verified kernel scoring, but does not
cancel the shared anchor needed by other valid candidates.

Useful metadata: `verify_reward_mode`, `verify_kernel_reward`,
`verify_kernel_rollout`, `verify_anchor_key`, `verify_anchor_index`,
`verify_anchor_baseline` (`fixed_source`), `verify_anchor_reward`, `verify_anchor_rollout`,
`verify_scoring_weight_versions`, and `verify_scoring_version_mismatch`.

With `VERIFY_ROLLOUT_RATIO=0`, ordinary kernel RL and optional verify-data capture
continue without diagnosis or helper scoring requests.

### Kernel evaluation lifecycle

Ordinary, verified and anchor kernels share the same `run_kernel_eval()` lifecycle.
`kernel_eval_client_timeout` now covers Ray queueing, token acquisition, HTTP
submission/retries and result polling. The deadline is stamped before the Ray
submission and is capped by the enclosing `generate()` trajectory deadline, including verify-scoring and anchor branches. A locally expired or invalidated call is rejected before creating a Ray RPC, and the actor checks again when execution begins, after token acquisition, before POST/retries, and while polling. Cross-node deadlines use wall time (nodes should have synchronized
clocks); the caller and executing actor also enforce local monotonic timeouts.
The separate `kernel_eval_task_timeout` remains the server execution budget.

HTTP operations and Ray result waits are asynchronous. Heartbeat queries and
invalidation RPCs use distinct finite Ray concurrency groups (`heartbeat=4`, `cancellation=4`), neither sharing the long submit/poll group's slots. HTTP status/results use a separate eight-connection pool; HTTP cancellation has a separate eight-request-per-event-loop bound, with queue wait included in the cleanup deadline. Each status/control request is bounded; a stalled
heartbeat cannot delay a ready result. No default executor thread is occupied by
the evaluation wait or HTTP cancellation.

On timeout, cancellation or a result-wait exception, the owner invalidates the
task ID, cancels the matching Ray call without recursively cancelling cleanup,
and independently requests server cancellation even for calls still queued in Ray. An invalidation reply saying HTTP has not started does not bypass the server tombstone; a `ray.cancel()` exception does not skip invalidation acknowledgement or DELETE. A timed-out invalidation RPC is preserved so it can still fence a late-starting call. The actor
checks invalidation/deadline before submission and retries, and cleans up on exit.
Rate-limit tokens use expiring, idempotent lease IDs so a late grant or duplicate
release cannot leak or inflate capacity. A separate limiter actor name ending in
`-v2` avoids reusing an actor with the old blocking-acquire interface.

Cleanup has its own bounded budget (`kernel_eval_cancel_timeout`, default 5 seconds)
and may extend beyond the evaluation deadline. An initial DELETE 404 is retried
within that budget, not reported as confirmed cancellation. Unconfirmed server
cancellation emits a warning. Logs distinguish `event=cancel_requested`, `event=cancel_acknowledged`, `event=cancel_not_found`, and `event=cancel_failed`, with the call/invalidation/server scope; a 404 is never acknowledged as cancellation. A server 409 explicitly reporting a cancelled ID stops polling immediately. This client-side fencing does not replace the
server's responsibility to terminate running workflows/child tasks; HTTP
cancellation acknowledgements still depend on the deployed KernelGYM API.
Custom `kernel_eval_function_path` hooks retain their existing lifecycle contract.

The first POST passes the remaining evaluation budget as `workflow_timeout`, capped by an explicitly supplied smaller value. That payload is reused unchanged across retries so the server request hash remains stable. This requires the parent lifecycle / cancellation-tombstone server implementation for full end-to-end protection. Clock synchronization is still required; neither a cancellation ACK nor a business failure proves GPU process containment.

### Bounded group retries

`--rollout-max-retries 10` defaults to at most ten additional attempts of an aborted/failed prompt group (eleven attempts total); `0` disables retries, and negative/unlimited values are rejected. The retry count travels with the original group metadata as `kernel_agent_group_retry`, independent of newly stamped model versions. The generic rollout argument is currently enforced by the kernel-agent full-async worker for ordinary groups and shared-anchor groups; it does not change HTTP retry settings or add retries to other rollout implementations. Exhaustion stops new submissions, cancels remaining active groups, and fails the collector promptly rather than waiting for the global no-progress timeout or silently dropping the failed input. Shutdown does not requeue cancelled groups.

The ordinary `generate_and_rm_group()` entrypoint now owns sibling cleanup: if one trajectory raises, all unfinished siblings are cancelled and awaited before group exit. This closes the `asyncio.gather()` exception path that previously left sibling kernel evaluations running. Shared-anchor groups retain their existing cleanup path. Runtime worker statistics and experimental async metrics include `retried_groups` and `retry_exhausted_groups`.

### Rollout no-progress guard

The kernel-agent full-async collector enables these defaults for both ordinary
and verify RL, independently of `--use-fault-tolerance`:

```bash
--rollout-no-progress-warn-seconds 900 \
--rollout-no-progress-timeout-seconds 7200
```

Progress means an increase in the current batch's accepted unique prompt groups,
using the collector's existing filtering rules. Submitted/completed requests,
aborted/retried groups, filter rejections and additional turns of the same group
do not count as new progress. This does not change the existing filtering rules.
The monotonic timer resets on accepted-group growth and starts fresh for each
collection call; time spent training between collection calls does not count.

After 15 minutes without progress, a warning includes accepted/target counts,
filter-drop reasons, worker counters, and bounded heads of the active/completed
queues (up to 8 groups and 8 sample identifiers per group). Active entries include
their age, task IDs and generation versions. Snapshots use local metadata, not
Ray/HTTP queries, and never include prompt or kernel bodies. Warnings repeat every
15 minutes until progress resumes.

After 120 minutes without progress, the worker stops admitting new groups,
cancels in-flight tasks immediately without the normal 10-second drain grace,
and the collector raises `RuntimeError` instead of returning a partial batch.
Shutdown waits are bounded and log whether the worker thread stopped. A failed
worker is not automatically recreated by the next collection call: inspect the
snapshot and restart the rollout process to recover. No automatic checkpoint or
recovery policy is introduced.

Both thresholds accept finite non-negative seconds; 0 disables that action.
When both are enabled, the warning threshold must be smaller than the timeout.
The launcher exposes `ROLLOUT_NO_PROGRESS_WARN_SECONDS` and
`ROLLOUT_NO_PROGRESS_TIMEOUT_SECONDS`. Successful batches additionally report
`fully_async_no_progress_warnings` and `fully_async_max_no_progress_seconds`.
