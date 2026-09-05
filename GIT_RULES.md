# Git Rules

## Data and generated artifacts

1. Do not commit files under `Data/`
2. Keep formal datasets, retained intermediates, logs, caches, and machine evidence (host information) out of Git tracking. They may stay on disk, but only inside a dedicated directory such as `local_artifacts/` — not scattered across the repo
3. Important local intermediates may remain under an ignored, clearly named directory such as `<experiment>/intermediate_artifacts/`

## Maintained source

1. Keep one maintained entry point for each workflow
2. Delete superseded versioned copies, one-off debugging programs, temporary migration helpers, and evolution-only trace scripts
3. Do not keep compatibility shims unless an active caller requires them
4. Keep implementations simple and remove redundant validation or defensive branches that do not protect a demonstrated contract

## Tests and launchers

1. Do not commit unit tests that assert launcher parameters, launcher-specific defaults, or assembled launcher command lines. Cover the underlying argument validation and runtime safety contracts in Python tests; validate launchers with syntax checks and manual dry runs.

## Simplicity and canonical ownership

1. Each workflow, document, and evidence set has exactly one canonical owner; extend it instead of creating parallel versions, and store details once — other places keep only the conclusion plus a link
2. Delete obsolete code and documents instead of labeling them legacy, deprecated, v2, or backup; Git history already preserves the past
3. Split documents that are reviewed or reverted independently, even when they share a topic

## Data workflow checks

1. Do not put code related to offline data processing — including its checks — under `tests/`; `tests/` is for slime tests
2. Put focused executable checks next to the data-processing code they validate (e.g. under `tools/data/`)

## Before committing

1. Group task-related changes, including relevant untracked files, into independently revertible responsibilities; plan the whole dirty worktree only when the user asks to organize all pending changes
2. Resolve semantic duplication in documentation being committed
3. Use `--amend` for review feedback only when the latest unpushed commit belongs to your current task and the feedback has the same responsibility; preserve unrelated commits
