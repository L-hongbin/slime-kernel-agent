---
name: unattended-user-decides
description: Unattended mode where decisions belong to the user. When a problem needs a user decision (ambiguous requirements, risky/irreversible actions, scope changes, conflicting evidence), page the user via the page_user MCP tool with the situation and options, then block on that decision. Use when the user wants to stay in control of judgment calls during an unattended run, including when switching from unattended-self-decide and handing off its progress.
---

# Unattended Mode — User Decides

You are running unattended, but decision authority stays with the user. Do not guess on judgment calls; escalate them.

## On entry

When this mode starts, send one page confirming: the active mode (user-decides), the heartbeat cadence (every 3 hours), and the immediate next objective. This confirms the run actually entered the intended unattended mode. Mark it STARTED.

If entering by switching from `unattended-self-decide`, include a handoff of the self-decide period in the STARTED page: progress completed, work currently in flight, material decisions made and why, relevant evidence or artifacts, unresolved problems or risks, and the next action. Do not report only the mode change; give the user enough context to understand and review what happened while decisions were delegated to the agent.

## Independent review

Use the `cross-model-review` skill as the required gate before each MILESTONE page. You may also use it for a second opinion while diagnosing a problem or weighing options; when it informs your recommendation, include its verified take in the page. Follow that skill's timeout and fallback rules without using them to bypass a decision that belongs to the user.

## When to page

Page the user (via `mcp__page-user__page_user`) when any of these holds:

- **You need the user's assistance** — a problem needs their decision (categories below) or something only they can do (credentials, approvals, physical access).
- **You reach a milestone** — after the cross-model review passes (or its documented timeout fallback completes), page with the milestone result and what comes next.
- **Every 3 hours** — a heartbeat with current status, progress since the last page, and anything queued for their attention.

Mark each page as STARTED, DECISION NEEDED, MILESTONE, or HEARTBEAT.

### Decisions that need the user

Escalate whenever you hit a problem that genuinely needs their decision:

- **Ambiguity** — the request or evidence supports multiple reasonable interpretations and the choice changes the outcome.
- **Risk** — destructive, irreversible, or outward-facing actions (deleting checkpoints, killing shared jobs, force-pushing, spending significant compute), or anything listed in `CONFIRMATION_GATES.md`.
- **Scope change** — completing the task would require doing something materially beyond what was asked.
- **Conflicting evidence** — results contradict the user's stated assumptions and proceeding would bake in one interpretation.
- **Repeated failure** — the same error twice after researching fixes, and the candidate fixes have materially different trade-offs.

Do NOT page for problems you can resolve from the request, the code, project docs, or sensible defaults — handle those and continue.

## How to page

Each page must be decision-ready. Include:

1. **Context** — one or two sentences on what you were doing and where you are.
2. **The problem** — what happened, with the key evidence (exact error, conflicting numbers, file paths).
3. **Options** — 2–4 concrete choices, each with its trade-off in one line.
4. **Recommendation** — which option you would pick and why, so the user can reply with a single word.
5. **What is blocked vs. what continues** — state which work waits on this decision.

## While waiting

- Do not proceed with the blocked decision or any action that presumes an answer.
- Continue any independent work that does not depend on the decision.
- If nothing else can proceed, checkpoint your state (notes, partial results, logs) so the run resumes cleanly once the user answers.
- Keep sending the MILESTONE and 3-hour HEARTBEAT pages on schedule; note in them which decision is still pending.

## Guardrails

- **Workarounds stay labeled as workarounds.** Falling back to a different kernel, disabling a feature, changing precision, skipping a gate, or narrowing scope to make progress is a workaround, not a root-cause fix. The root cause remains an open item until separately validated; never let a workaround silently become a conclusion.
- **Sync code before trusting results.** After any code edit in an environment where paths may be node-local or inconsistently mounted (e.g., per-node `/nfs`), sync the change to all relevant nodes and verify it — for example with checksums — before drawing conclusions from a run. Any result produced by stale or partially synced code is invalidated: mark it as such and rerun.
