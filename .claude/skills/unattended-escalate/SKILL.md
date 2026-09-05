---
name: unattended-escalate
description: Unattended work with phone escalation for decisions reserved to the user. Use when the user wants to retain decision authority, including handoffs from unattended-autonomous.
---

# Unattended Mode — Escalate

Continue work covered by the user's instructions and prior approvals. Escalate unresolved choices that materially change the outcome or need authority or access the user has not supplied.

## On entry

Send a STARTED page with the active mode, the 3-hour heartbeat cadence, and the immediate objective.

When switching from `unattended-autonomous`, include completed and in-flight work, material decisions and reasons, evidence, unresolved issues, and the next action.

## When to page

Use the `page_user` MCP tool for these notifications:

- **DECISION NEEDED**: a material choice remains unresolved after checking the request, code, and evidence; the next action exceeds authorized scope or resources; the user explicitly reserved the decision; or credentials or access require the user.
- **MILESTONE**: report the verified result and next work.
- **HEARTBEAT**: every 3 hours, report progress and any pending decision.

Prior authorization remains valid. Error counts, surprising evidence, or expensive work already within the agreed budget do not alone require another decision; investigate first and escalate the unresolved trade-off, if any.

Use `cross-model-review` for requested reviews or when important experiment conclusions or complex high-risk changes need independent review; include relevant verified findings in the recommendation.

## How to page

For a decision request, give the context, key evidence, concrete options and trade-offs, your recommendation, and which work depends on the answer.

## While waiting

- Wait for the answer before taking dependent actions; continue independent work.
- Preserve a restart point when blocked and continue the 3-hour heartbeat with the pending decision.

## Guardrails

- Stay within the user's authorized task and shared-resource boundaries. Label workarounds and any reduced scope; keep the original outcome open until verified.
