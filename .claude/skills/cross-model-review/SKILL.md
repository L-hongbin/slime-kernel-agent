---
name: cross-model-review
description: Use a configurable external model to review milestones or provide a second opinion.
---

# Cross-Model Review

## Reviewer configuration

Try reviewers in this order, moving to the next when the command, model, or service is unavailable or the call fails:

1. `agentp --print --model cursor-grok-4.6-xhigh --workspace <repo-root> <prompt>`
2. `agentp --print --model kimi-k3-high --workspace <repo-root> <prompt>`
3. `kimi --model kimi-code/k3 --prompt <prompt>` with `thinking.effort = "high"` in `~/.kimi-code/config.toml`

- Milestone timeout: 30 minutes
- Issue-discussion timeout: 30 minutes

A reviewer or model specified by the user for the current task overrides this default order. The applicable timeout covers the whole fallback sequence and does not restart for each candidate.

## Review

Use the configured model to review each milestone before reporting it as reached. It may also provide a second opinion while diagnosing a problem or weighing options. Give it the task, relevant code and evidence, and the result or decision being reviewed.

The reviewer may inspect the repository, run focused tests, make small targeted changes, or add test files. It must not make broad changes or start long-running, expensive, or externally mutating work.

If the reviewer is unavailable, errors out, or exceeds the applicable timeout, do an adversarial self-review where safe and record `Cross-model review timed out; self-review substituted.` This fallback never bypasses a user-owned decision or safety-critical confirmation.
