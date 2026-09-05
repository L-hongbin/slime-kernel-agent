---
name: unattended-autonomous
description: Unattended work with delegated decisions and phone notifications. Use when the user wants uninterrupted progress within the authorized task and will review decisions afterward.
---

# Unattended Mode — Autonomous

Make decisions within the user's authorized task and resource boundaries. Notify the user of material decisions without asking them to choose.

## On entry

Send a STARTED page with the active mode, the 3-hour heartbeat cadence, and the immediate objective.

## When to page

Use the `page_user` MCP tool for these notifications:

- **DECIDED**: immediately report a material choice, its key evidence and rationale, the strongest rejected alternative, and impact. Examples include authorized changes to hyperparameters, sampling, loss or optimizer behavior, checkpoint source, precision, kernel path, validation or cleanup, and additional expensive runs. Batch minor operational choices into the next heartbeat.
- **MILESTONE**: report the verified result, material decisions, and next work.
- **HEARTBEAT**: every 3 hours, report progress and batched minor decisions.

Use `cross-model-review` for requested reviews or when important experiment conclusions or complex high-risk changes need independent review; report how verified findings affected the decision.

## Guardrails on autonomy

- This mode grants no authority to delete shared checkpoints, kill other users' jobs, force-push shared branches, or mutate shared infrastructure. If further authorization or a user-owned decision is needed, preserve a restart point, page DECIDED with the blocked item, and continue independent work within scope.
- Label workarounds and any reduced scope explicitly; the original outcome remains open until verified.
