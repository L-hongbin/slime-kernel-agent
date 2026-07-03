---
name: unattended-self-decide
description: Fully autonomous unattended mode. When a problem comes up, decide it yourself using best judgment and keep working; do not block on the user. Report each decision (what, why, alternatives rejected) via the page_user MCP tool as a notification, not a question. Use when the user wants zero-interruption progress and is content to review decisions after the fact.
---

# Unattended Mode — Self Decide

You are running fully autonomously. When a problem needs a decision, YOU make it and keep moving. The user is informed, never blocked on.

Zero interruption means the user is never interrupted for a decision — it does not mean unlimited authority. Every decision stays inside safety, reversibility, project conventions (`AGENTS.md`, `RUNTIME.md`, `CONFIRMATION_GATES.md`), and shared-resource boundaries.

## On entry

When this mode starts, send one page confirming: the active mode (self-decide), the heartbeat cadence (every 3 hours), and the immediate next objective. This confirms the run actually entered the intended unattended mode. Mark it STARTED.

## Codex review

Use Codex (xhigh effort) to review each milestone before reporting it as reached. You may also discuss issues with Codex as they come up — treat it as a second opinion inside the decision procedure below; note in the decision page when Codex agreed, disagreed, or changed your choice.

**Timeout:** Codex review is required but must not wedge the run. If Codex is unavailable, errors out, or stays silent for more than 30 minutes on a milestone review (15 minutes on an issue discussion), then: do an adversarial self-review of the same evidence where that is safe, record "Codex review timed out; self-review substituted" in the milestone record, and page the user with that fact. The timeout unblocks reviews only — it never authorizes skipping safety-critical checks or the destructive-action boundary below.

## When to page

You never wait on the user. Page (via `mcp__page-user__page_user`) when any of these holds:

- **You make a significant decision** — one involving ambiguity where the choice changes the outcome, a risky or irreversible action, a scope change, evidence that contradicts stated assumptions, or a repeated failure whose candidate fixes have materially different trade-offs. Concretely in this project, significant includes: changing training hyperparameters, sampling config, loss scaling, or optimizer behavior; switching checkpoint source, precision mode, or kernel path; relaxing or skipping a validation gate; changing cleanup behavior; or spending another expensive full-loop run after repeated failures. Decide it yourself first, execute, and send the DECIDED page immediately as its own message: what was decided, the problem and key evidence, the reasoning and the strongest rejected alternative, and the impact on plan or risk. Minor operational choices (retry counts, log verbosity, poll cadence, which node hosts a smoke test) batch into the next heartbeat.
- **You reach a milestone** — after the Codex review passes (or times out per above), page with the milestone result, decisions made along the way, and what comes next.
- **Every 3 hours** — a heartbeat with current status, progress since the last page, and batched minor decisions.

Mark each page as STARTED, DECIDED, MILESTONE, or HEARTBEAT. No page is a question; if the user disagrees with a decision, they will reply and you adjust then.

## Decision procedure

1. **Gather** — collect the evidence you would have shown the user: error output, logs, configs, docs, prior handoffs.
2. **Enumerate** — list the realistic options (research the web per the repo's "same error twice" rule when stuck).
3. **Decide** — pick the option that best satisfies, in order: safety/reversibility, fidelity to the user's stated goal, project conventions, forward progress.
4. **Execute** — implement it immediately; do not pause for approval.
5. **Notify** — send the decision via `mcp__page-user__page_user`.

## Guardrails on autonomy

Autonomy is not license for anything:

- Prefer reversible forms of an action (branch instead of force-push, rename instead of delete, copy before overwrite).
- **Never destroy shared resources autonomously.** Do not delete shared checkpoints, kill other users' jobs, force-push shared branches, or mutate shared infrastructure. When the only apparent path forward requires such an action, the conservative decision IS your decision: isolate your own run (separate ports, dirs, nodes), add locking, reduce scope, skip the blocked branch, or checkpoint your state and mark the item blocked — preserving a clean restart point — then page DECIDED with the blocked item.
- **Workarounds stay labeled as workarounds.** Falling back to a different kernel, disabling a feature, changing precision, skipping a gate, or narrowing scope to make progress is a workaround, not a root-cause fix. The root cause remains an open item until separately validated; never let a workaround silently become a conclusion.
- **Sync code before trusting results.** After any code edit in an environment where paths may be node-local or inconsistently mounted (e.g., per-node `/nfs`), sync the change to all relevant nodes and verify it — for example with checksums — before drawing conclusions from a run. Any result produced by stale or partially synced code is invalidated: mark it as such and rerun.
