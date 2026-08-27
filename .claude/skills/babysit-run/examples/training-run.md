# Verifying a slime training run

Handle: launcher under `examples/kernel_agent/`. Evidence: the run's own log. `run.t1.qwen3.6.27B.fasync.sh` prints `Logging to <path>` before `exec >> "$LOG_PATH"`; capture that path, not a cwd `run.log`.

## Pattern

1. Check iter/paths/flags, inputs, resources; isolate first with train-only or rollout-only when useful.
2. Launch; poll tight for expected resume iter, live config, and sane first-step metrics.
3. Loosen once steady, but keep polling; auto-return is not verification.

## Worked example

**Diff/claim:** rollout filtering prevents empty-step trajectories from becoming all-reject training steps.
**Expected:** reject-rate < 1.0, real samples survive, intended `iter_N` resumes.

```bash
HEAD=/tmp/slime-launch.$$.head
bash examples/kernel_agent/run.t1.qwen3.6.27B.fasync.sh 2>&1 | tee "$HEAD" &

for _ in {1..60}; do
  LOG=$(sed -n 's/^Logging to //p' "$HEAD" | tail -1)
  [ -n "${LOG:-}" ] && break
  sleep 1
done
test -n "${LOG:-}" && test -f "$LOG"

grep -E "load.*iter_|resume" "$LOG" | tail -5
# -> loaded checkpoint ... iter_39    # expected, not iter_0/base

grep -E "reject_rate|filtered|empty.*step" "$LOG" | tail -5
# -> reject_rate 0.18                # bug signature moved off ~1.0

grep -E "entropy|grad_norm|reward" "$LOG" | head -6
# -> step 40 reward 0.31 entropy 1.92 grad_norm 0.7
```

**PASS requires:** resume iter from the run log, changed behavior in metrics, and first-step health without collapse/spike. 🔍 Probe: inspect saved rollout samples, when available, to confirm accepted steps hold real trajectories.
