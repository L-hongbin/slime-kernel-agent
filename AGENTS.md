# Agents Notes

## Harness Files

The repository's top-level harness files are the stable operational documents that guide work in this repo.

- `AGENTS.md` records repository-level principles and stable collaboration rules.
- `RUNTIME.md` records stable runtime facts such as environments, service endpoints, shared paths, data paths, and common run entrypoints.
- `INDEX.md` is the index for important documents, logs, scripts, and code entry points so key references are easy to locate.
- `CONFIRMATION_GATES.md` records additional confirmation gates that the user explicitly asked to write down. It is not an exhaustive list of every situation that may require confirmation; the agent may still ask the user to confirm other actions when judgment, risk, or ambiguity makes confirmation necessary. After confirmation, delete the item or mark the confirmed result as appropriate.
- `AGENTS.md` is not a routine scratchpad; do not update it proactively. Change it only when the user explicitly asks for an instruction or policy update.

## Documentation Policy

1. Update `RUNTIME.md` proactively when stable runtime facts materially change, including environments, service endpoints, shared paths, data paths, and common run entrypoints. Keep experiment-specific settings and results in handoffs.
2. Update `INDEX.md` proactively when important references materially change, including documents, logs, scripts, code entry points, generated artifacts, and reviewable evidence files.
3. Do not create or maintain a separate running-status log unless the user explicitly asks for one.
4. `CONFIRMATION_GATES.md` should not be populated proactively; add items there only when the user explicitly asks to record an additional confirmation gate.
5. Keep `RUNTIME.md` concise and under 100 lines.
6. Keep `INDEX.md` concise and under 100 lines.

## Working Conventions

1. For any artifact, behavior, or result that can be practically manually reviewed, inspect representative real examples in addition to automated checks. When useful, dump reviewable prompts, data samples, logs, model outputs, scored examples, or other concrete artifacts to text files so the user can inspect the same evidence.
2. Do not use eager mode when deploying models unless it is for temporary debugging.
3. Treat `/nfs` and `/ms` as shared filesystems.
4. Before launching any long-running job, run a sanity check that verifies the critical configuration and recently modifications.
5. When changing behavior-sensitive logic, add a unit test. If a unit test is impractical, replace it with an explicit sanity check that exercises the changed path on a real input.
6. Do not blindly trust that a dispatched job will auto-return. A progress-based job notification is not enough — periodically poll status to catch unexpected situations (broker wedges, sandbox failures, silent stalls, processes stuck in D-state on NFS).
7. Avoid heavy `find` invocations (full-FS scans like `find / -name X 2>/dev/null`). They can take hours on shared NFS and pin filesystem caches. Prefer (a) checking known paths directly, (b) `ls` / `glob` of specific directories, (c) `locate` / `mlocate` if available. If a broad `find` is genuinely necessary, ask the user for approval first.

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
