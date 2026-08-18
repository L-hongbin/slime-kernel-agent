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

## Data workflow checks

1. Do not put code related to offline data processing — including its checks — under `tests/`; `tests/` is for slime tests
2. Put focused executable checks next to the data-processing code they validate (e.g. under `tools/data/`)

## Before committing

1. Use `--amend` when review feedback applies to the latest unpushed commit rather than adding a corrective follow-up commit
