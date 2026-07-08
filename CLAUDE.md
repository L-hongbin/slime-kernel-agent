# Agents Notes

## Harness Files

The repository's top-level harness files are the stable operational documents that guide work in this repo.

- `AGENTS.md` records repository-level principles and stable collaboration rules.
- `RUNTIME.md` records specific run details such as environments, service endpoints, model paths, dataset paths, hyperparameters, and other experiment-specific configuration.
- `INDEX.md` is the index for important documents, logs, scripts, and code entry points so key references are easy to locate.
- `WRITING.md` records writing requirements for handoffs, debugging conclusions, experiment summaries, and long-lived engineering docs; follow it when producing those documents.
- `CONFIRMATION_GATES.md` records additional confirmation gates that the user explicitly asked to write down. It is not an exhaustive list of every situation that may require confirmation; the agent may still ask the user to confirm other actions when judgment, risk, or ambiguity makes confirmation necessary. After confirmation, delete the item or mark the confirmed result as appropriate.
- `AGENTS.md` is not a routine scratchpad; do not update it proactively. Change it only when the user explicitly asks for an instruction or policy update.

## Documentation Policy

1. Update `RUNTIME.md` proactively when run-specific facts materially change, including environments, service endpoints, model paths, dataset paths, hyperparameters, and other experiment configuration.
2. Update `INDEX.md` proactively when important references materially change, including documents, logs, scripts, code entry points, generated artifacts, and reviewable evidence files.
3. Do not create or maintain a separate running-status log unless the user explicitly asks for one.
4. `CONFIRMATION_GATES.md` should not be populated proactively; add items there only when the user explicitly asks to record an additional confirmation gate.
5. Keep `RUNTIME.md` concise and under 100 lines.
6. Keep `INDEX.md` concise and under 100 lines.

## Working Conventions

1. For any artifact, behavior, or result that can be practically manually reviewed, inspect representative real examples in addition to automated checks. When useful, dump reviewable prompts, data samples, logs, model outputs, scored examples, or other concrete artifacts to text files so the user can inspect the same evidence.
2. Do not use eager mode when deploying models unless it is for temporary debugging.
3. Treat `/ms` as shared filesystems.
4. Before launching any long-running job, run a sanity check that verifies the critical configuration and recently modifications. After the run finishes, dispatch a codex review (xhigh effort) of the results before drawing conclusions.
5. For substantial code or design changes, dispatch a codex review (xhigh effort) before treating the change as final.
6. When changing behavior-sensitive logic, add a unit test. If a unit test is impractical, replace it with an explicit sanity check that exercises the changed path on a real input.
7. Do not blindly trust that a dispatched job will auto-return. A progress-based job notification is not enough — periodically poll status to catch unexpected situations (broker wedges, sandbox failures, silent stalls, etc).
8. Avoid heavy `find` invocations. Prefer (a) checking known paths directly, (b) `ls` / `glob` of specific directories, (c) `locate` / `mlocate` if available. If a broad `find` is genuinely necessary, ask the user for approval first.
9. When debugging slime runs, isolate one side of the pipeline first instead of launching the full loop: `--debug-rollout-only` runs rollout without the training backend (pair with `--save-debug-rollout-data` to dump rollout data), and `--debug-train-only` runs training without sglang (pair with `--load-debug-rollout-data` to replay saved rollouts; setting it implies debug-train-only). The two flags are mutually exclusive.

## Execution Policy

1. If user instructions need adjustment during execution, it is acceptable to adapt pragmatically instead of stalling.
2. Any such adjustment must be reported back at the end, including:
   - why the original instruction could not be executed as-is
   - what was changed
   - the result after the change
   - the current status and remaining gaps
3. If execution encounters a situation that the relevant skill did not anticipate, report that explicitly afterwards.
   - state what the skill did not cover
   - state what workaround, judgment call, or temporary procedure was used instead
   - state whether the gap still exists in the skill after the task
4. In final summaries, prefer a combined "problem and resolution" structure instead of separating "problems" and "adjustments".
5. Don’t fight errors. Whenever you encounter the same error twice, research the web and find 3-5 possible ways to fix it. Then choose the most efficient solution and implement it.
6. Treat surprising results as bugs until the evidence says otherwise. Do not start by explaining why an anomaly may be reasonable. First ask: what would make this result impossible, misleading, or caused by a bad measurement?
7. Close the causal chain before closing the task. Do not wait for the user to name the next obvious check.
8. Separate workaround from root cause. A workaround can unblock progress, but it must be recorded as such. Do not convert "use the safer path" into "the unsafe path is naturally bad" without proving it.
