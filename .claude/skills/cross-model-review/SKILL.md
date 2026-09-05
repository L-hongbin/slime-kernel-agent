---
name: cross-model-review
description: Read-only review with a configurable external model. Use when requested or when important experiment conclusions or complex high-risk changes need independent review.
---

# Cross-Model Review

## Review scope

Give the reviewer the task, relevant code and evidence, and the conclusion or change to assess. Ask for findings with supporting evidence and unresolved questions. Routine milestone reporting does not require a review.

The reviewer is read-only: it may inspect code and evidence and run focused diagnostics within the task's resource limits. Explicitly prohibit source edits, new tests, and expensive or externally mutating work in the review prompt. The primary agent evaluates findings and implements any fixes.

While a review runs, continue independent work and treat work that relies on the reviewed conclusion as provisional until the review completes.

## Reviewer configuration

Try reviewers in this order, moving to the next when the command, model, or service is unavailable or the call fails:

1. `agentp --print --output-format stream-json --model cursor-grok-4.6-xhigh --workspace <repo-root> <prompt>`
2. `agentp --print --output-format stream-json --model kimi-k3-high --workspace <repo-root> <prompt>`
3. `kimi --model kimi-code/k3 --prompt <prompt>` with `thinking.effort = "high"` in `~/.kimi-code/config.toml`

Keep draining the `agentp` JSON event stream until the process exits, and take the review verdict from its final result event. Use event-level `stream-json` for progress visibility; do not add `--stream-partial-output` for routine reviews because wrapping every text delta as a JSON object wastes context. Default `--print` text mode may remain silent until completion, so lack of stdout from a non-streaming invocation is not evidence that the reviewer stalled.

Tolerate reconnect and checkpoint-resume events while `agentp` remains alive. A finite number of reconnects, replayed events after resume, or a temporary quiet interval is not by itself reviewer failure; let `agentp` use its own bounded reconnect policy and continue draining until it exits or the shared review timeout expires. Do not manually terminate a candidate solely because reconnects repeat.

Treat the candidate as failed when `agentp` exits unsuccessfully, reports that its reconnect limit was exhausted, the event stream terminates without a final result, an unrecoverable TLS/proxy/rate/resource error occurs, or the shared review timeout expires. Do not manually retry or resume a candidate after such a failure during the same review; continue with the next viable model while the overall timeout remains. After checkpoint replay, use only the final result event as the verdict rather than intermediate or duplicated reasoning.

A reviewer, model, or time budget specified by the user overrides the defaults. The default timeout is 30 minutes for the whole fallback sequence, without restarting the clock for each candidate.

If all candidates fail or the shared timeout expires, perform a self-review where useful and report the actual failure reason and substitution. Do not represent self-review as independent review or as satisfying an explicitly required external review or user confirmation.
