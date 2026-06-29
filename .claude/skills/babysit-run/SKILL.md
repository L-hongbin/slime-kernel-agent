---
name: babysit-run
description: Verify long-running ML/RL jobs from their own logs, metrics, dumps, artifacts, or service responses — never infer success from launches, tests, configs, or imports. Use after starting training, eval, checkpoint convert, serving, or multi-step jobs to confirm intended checkpoint, config, progress, and outputs before reporting success.
---

Verification means runtime observation: drive changed code to its real surface and capture what the job emits. Logs, metrics, dumps, tables, artifacts, and responses are evidence.

Not surfaces: `pytest` and import-and-call snippets. Use them as prechecks/unit evidence, then verify the real loop at a surface below.

## Trigger

Use immediately after launching a training run, eval, checkpoint convert, service, or multi-step job. Launch is unverified until the job's own output proves:

- **checkpoint/resume** — expected iter/path, not base or fallback;
- **live config** — intended flags, model/data paths, resources, and no unintended eager mode;
- **real progress** — first step/request/artifact/metric appears and is sane for the claim.

Before launch, sanity-check config, edits, inputs, resources, and whether one pipeline side can be isolated. After launch, poll tight (~1 min) until all criteria are observed, then loosen once steady. The monitor wakes you to look; it does not look for you. Review final outputs too.

## Surface

Observe where the change executes and emits something readable.

| Change | Surface | Verify |
|---|---|---|
| Training | logs + metrics | resume, config, first metrics — [example](examples/training-run.md) |
| Rollout / reward / data | rollout dump | rollout-only; inspect trajectories |
| Eval / scoring | dumps + summary | table and samples — [example](examples/eval.md) |
| Serving / router | HTTP endpoint | readiness, request, response + log |
| Dependency service | health + canary | health plus real exercise; cached health can lie |
| Convert / resume | artifact + load log | complete artifact and expected iter |
| Offline script | output | run on real input; read output |
| Docs / config-only | - | **SKIP — no runtime surface: <reason>.** |

Internal functions are not surfaces; follow the caller. Tests in the diff are author evidence, not this verification.

## Get a Handle

Use recorded entrypoints (`RUNTIME.md`, `INDEX.md`, launchers, scripts). Cold-start max ~15 min; if stuck, report BLOCKED with the exact stop point. Note stable new recipes for docs.

## Drive It

Smallest real path: config -> launch/process state; rollout/reward/data -> dump; train step -> train-only replay; eval/scoring -> small eval + dumps; convert/resume -> one iter + load log.

Read your plan back. If every step is pytest, import-and-call, or "it should log X", find a real surface or report BLOCKED. Inspect real examples, not only rates; dump prompts, trajectories, scored samples, or tables when useful.

For shared/destructive paths, use dry-run or safe targets where practical. Before `ray stop`, `pkill`, service restart, or node-wide cleanup, confirm what jobs/processes it would kill. Say what path you did not exercise.

## Push on It

Probe around the happy path at the same surface: empty/conflicting flag, malformed data, missing field, wrong target, non-regression path, non-finalized checkpoint, repeated convert/resume. Follow odd logs/metrics/dumps. A clean probe still gets one report line.

## Capture

Capture log tails, metric values, response bodies, artifact paths, sample excerpts, and tables. Treat surprising output as a bug until evidence says otherwise; unrelated breakage is still a finding.

## Report

```
## Verification: <what changed / what you ran>

**Verdict:** PASS | FAIL | BLOCKED | SKIP
**Claim:** <diff/stated intent; note mismatch>
**Method:** <what you launched, where, with which handle>

### Steps
Each step = running-job action + observed evidence. Setup is not evidence. pytest/import do not belong here.
1. ✅/❌/⚠️/🔍 <action> -> <observed> <evidence: log tail, metric, dump excerpt, table>
🔍 marks a probe; expected when practical. All-✅/no-🔍 is only a happy-path replay.

**Sample / table:** <reviewer-facing artifact; omit for config-only>

### Findings
⚠️ Interrupt-worthy lines first; plain bullets are context. Include oddities and each probe result.
```

Verdicts: PASS — real surface did what it should. FAIL — it did not, broke something else, or claim/diff disagree. BLOCKED — no observable state reached; say where. SKIP — no runtime surface. No partial pass. When in doubt, FAIL with raw evidence.
