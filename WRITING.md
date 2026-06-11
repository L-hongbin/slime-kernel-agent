# Writing Requirements

This guide applies to handoffs, debugging conclusions, experiment summaries, and long-lived engineering docs. The target reader is a colleague who did not participate in the investigation but must make engineering decisions based on the document.

## Principles

- **Write for the reader, not for yourself.** Readers want to know: how do I decide, how do I configure, what is the risk, what is the fix. Lead with the engineering conclusion, then the causal chain, then the evidence — the reader should know what action to take within the first 1–2 screens.
- **A document is not a chronological log.** Do not organize the body by timeline or by "how I found it"; the main thread is what the failure chain is, how to fix it, and how to verify the fix.
- **Omit what the reader does not need.** Debugging detours that proved irrelevant, parameters that only mattered during a detour, and adjacent flows outside the topic are left out by default. Internal artifacts appear only when necessary; if something must be preserved for later review, put it in a Markdown comment.
- **State conclusions, not hedges.** With evidence, state the conclusion directly; without evidence, state the boundary of what is known. Do not substitute "maybe / possibly / I suspect" for a causal chain.

## Style

- Direct, concrete, actionable. Prefer "the root cause is", "the fix is", "the risk is" over "we tried", "later we also discovered".
- One concept per paragraph; explain every term on first use, e.g. TMS = `torch_memory_saver`.
- Prefer tables for comparisons instead of describing differences in prose.
- Headings express topics, not historical order.

## Checklist

- Can a colleague who did not participate know how to configure things within 2 minutes?
- Are detours, timestamps, run ids, and artifact paths out of the body (or in comments)?
- Do the correctness checks and performance results directly support the proposed fix?