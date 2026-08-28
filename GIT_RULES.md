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

## Simplicity and canonical ownership

1. Each workflow, document, and evidence set has exactly one canonical owner; extend it instead of creating parallel versions, and store details once — other places keep only the conclusion plus a link
2. Delete obsolete code and documents instead of labeling them legacy, deprecated, v2, or backup; Git history already preserves the past
3. Split documents that are reviewed or reverted independently, even when they share a topic

## Data workflow checks

1. Do not put code related to offline data processing — including its checks — under `tests/`; `tests/` is for slime tests
2. Put focused executable checks next to the data-processing code they validate (e.g. under `tools/data/`)

## Before committing

1. Group the entire dirty worktree (including untracked files) into commits upfront; define each commit as one independently revertible responsibility instead of choosing boundaries file-by-file or by a broad label like train, eval, or docs
2. Before a commit adding 3+ documents or 500+ documentation lines, do a semantic-duplication review and resolve it first
3. Use `--amend` on the latest unpushed commit for review feedback instead of adding fix-up commits
