# E1.1 candidate screening and source audit

Screen the existing output of `tools/data/trajectory_structure` without running
generated code. This is a diagnostic front end for candidate selection, not an
implementation of causal credit, training reward, or formal natural-pool collection.

```bash
python -m tools.data.trajectory_candidates.screen \
  --input-dir local_artifacts/paper/trajectory_structure/release \
  --output-dir local_artifacts/paper/trajectory_candidates/new_screen \
  --review-selection local_artifacts/paper/trajectory_candidates/release/review_selection.json

python -m tools.data.trajectory_candidates.audit \
  --input-dir local_artifacts/paper/trajectory_candidates/new_screen \
  --annotations local_artifacts/paper/trajectory_candidates/manual_annotations.json \
  --output-dir local_artifacts/paper/trajectory_candidates/new_audit
```

Both output directories must be new. If fixing a screener after reviewing a
sample, pass `--review-selection <previous review_selection.json>` to keep that
cohort fixed; initial flags and current flags are recorded separately. Manual
annotations must cover that exact cohort without duplicates. The audit renderer
does not infer verdicts from automatic flags.
The example reproduces the existing fixed audit cohort. For a new cohort, omit
`--review-selection`, inspect the selected cases, and supply matching new manual
annotations rather than reusing the current forty annotations.

The annotation JSON contains a `scope` string and a `cases` list. Each case needs
`model`, `group`, `verdict` (`candidate`, `control`, or `indeterminate`), `note`,
`evidence`, `priority`, and `decision_class`. Notes and evidence must be nonempty;
priority and decision class are human-supplied descriptions, not inferred labels.

## Frozen diagnostic defaults

- Score proxy: positive raw speedup, only on explicitly correct, compiled,
  non-decoy, error-free results; this does not reconstruct training reward
- Observed breakthrough margin: strictly more than 5% above the prior observed
  best; a repeated full component-syntax version with a faster clock is flagged
  for remeasurement instead of accepted as a new-code breakthrough
- Large rewrite: CUDA token-trigram similarity below 0.50 in a comparable
  adjacent transition, followed by a strictly later observed breakthrough
- Sensitivity: margins 0/5/10%, rewrite cutoffs 0.35/0.50/0.65; report all settings,
  never choose the most favorable setting using these historical test results
- Labels overlap; counts use trajectories as denominator, not number of hunks
- Audit selection: deterministic task-diverse stratified sample, initially ten
  candidates and ten controls per model when available; not a probability sample

`decline_then_breakthrough` requires an earlier correct baseline, an observed
failure or a >5% slowdown, and a later observed new best. `nonadjacent_reuse`
requires a syntactic edit still present at a breakthrough at least two turns
after introduction. `rewrite_then_breakthrough` is temporal association only:
the output separates improvement at the rewrite turn from additional later gain.

Unobserved responses, uncertain resource errors, unattributed execution timeouts
and worker crashes do not become model regressions. Opportunities hidden by a
missing adjacent response remain in `uncertain_windows`, including when no
adjacent rewrite comparison or retained hunk exists. An evaluator policy rejection
is distinct from a contradictory measurement and never enters the timed best.

## Evidence and limitations

Each candidate JSON contains the timeline, baseline/outcome turns, matched edit
IDs, grouped component edits, uncertainty and literal decision call-site facts.
Math mode, launch configuration, named tiling constants and selected GEMM/layout
arguments are supported. Comment nodes are removed from historical argument lists.
One call may have both math-mode and transpose facts. These facts and lexical
`decision_cues` are not semantic equivalence or executable optimization decisions.
Names, casts, error text and checks can satisfy the raw reuse rule.

For the audit sample, `review/*.md` contains diffs and error excerpts, while
`review_sources/<id>/T*.{cu,cpp,py}` contains complete selected source sections.
`audit` rejoins the original extracted `g*.json` inputs when their paths and hashes
are available in the structural manifest. It verifies the structural manifest
and source hashes against the screening run, response identity, unique turn
coverage and timing ratios, and preserves original runtime, trial count, CV and profiling
kernel records. A profiling call confirms that profiling invocation, not every
timing trial or a causal effect. Within-evaluation repetitions are not independent
controlled reruns.

The historical KernelBench pool is used only to expose tooling failure modes.
Formal E1.1 must use independently sampled train/development tasks, preserve the
actual model context/feedback and freeze its protocol separately. The current
audit is non-blind source review, not a population precision/recall estimate.

## Checks

```bash
PYTHONPATH=local_artifacts/paper/trajectory_structure/deps \
  python -m tools.data.trajectory_candidates.check_candidates
```

Screening/audit use the Python standard library. CPU checks also use the existing
structural parser fixtures and their pinned dependencies. No GPU or CI changes.

[E1.1 findings and current experiment plan](../../../handoffs/paper/plan.md#e11-留下的有用线索)
