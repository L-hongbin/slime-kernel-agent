# Offline trajectory structure analysis

Extract the CUDA/binding/Python sections selected by the actual rollout response
parser, align syntactic components across turns, and trace anchored local edits
and component-version recurrences. This tool never executes generated code or
contacts KernelGym. Its output is structural evidence, not semantic equivalence,
optimization-decision attribution, or causal credit.

## Run

Install the pinned parsers into the analysis environment:

```bash
pip install -r tools/data/trajectory_structure/requirements.txt
python -m tools.data.trajectory_structure.analyze \
  --input model_a=/path/to/eval_0.pt \
  --input model_b=/path/to/extracted_groups \
  --review-group 0 \
  --output-dir local_artifacts/paper/trajectory_structure/new_run
```

Inputs can be trusted local `.pt` dumps containing `samples`, JSON/JSONL rows,
or a directory of extracted `g*.json` lists. `.pt` uses pickle through
`torch.load(weights_only=False)`; never supply untrusted files. Rows need full
`response`, `group_id` (or `trajectory_id`), and `turn_idx` (also accepted in
`metadata`). Reference identity and dataset are included in the grouping key
when available. Supply distinct input names for distinct runs/models. Missing
task identity cannot be reconstructed from a reused group ID.

`--group-id` restricts processing to selected IDs; repeat it as needed.
`--max-trajectories` is a global cap across inputs, not a per-model balanced
sample. `--review-group` adds Markdown source-diff views without restricting
the analyzed pool. Output directories must be new.

## Output contract

- `manifest.json`: input and implementation hashes, parser versions, runtime,
  selection and a final code/input stability check
- `summary.json`, `trajectory_index.json`: coverage and evidence locations
- `trajectories/*.json`: selected source, parser errors/adaptations, components,
  call-site candidates, adjacent diffs, edit observations and version recurrences
- `trajectories/*.md`: review views for requested groups, including source diffs

JSON turn indices are zero-based; Markdown turns are one-based. Component IDs
are turn-local locators; lineages and edit-event IDs are local to a trajectory.
Native source locations refer to the extracted section, not the full response.
Python file-context source is an AST rendering, explicitly marked in the output.

`syntax_valid` means accepted by the local grammar, not compiled by NVCC or
accepted by precheck. Native pragmas are lexically masked for parsing and retained
in source and token hashes; every adaptation is recorded. Macros/templates are
not expanded and conditional compilation is not evaluated.

Qualified names together with exact signatures establish syntactic alignment.
Native signatures ignore formatting, but retain parameter names as well as types.
Signature changes start a separate lineage; this deliberately misses some valid
cross-signature continuations in exchange for not merging different overloads.
Exact syntax after replacing a declaration's own name produces a rename
*candidate*, in `rename_candidates`, without sharing a lineage. Call targets are name-based
candidates without type checking, dynamic dispatch or reachability proof.

Token-trigram similarity is reported separately for CUDA, binding and Python.
No automatic large-rewrite threshold is imposed. Local edit hunks retain five
tokens of unchanged context; large components retain source diffs even when
local token alignment is skipped. Nearby edits whose anchors cannot be isolated
may not produce trackable events. An event is a token edit, not necessarily one
complete optimization decision.

Observation states:

| State | Meaning |
|---|---|
| `present` | Unique edited token pattern is observed, or added component version matches exactly |
| `reverted` | Unique pre-edit token pattern is observed and edited pattern is absent |
| `component_absent` | Complete parseable snapshot has no matching component identity or exact-syntax rename candidate; syntactic absence only |
| `unknown` | Neither pattern is uniquely identifiable, including context drift |
| `unknown_component_identity` | No reliable matching component in this turn |
| `unobserved` | Missing/incomplete protocol, missing ModelNew, or relevant syntax cannot be parsed |
| `modified_version` | Added component is aligned but its syntax has changed |

`nonadjacent_retained_turns` means the edit is present at least two turn indices
after introduction; it does not require uninterrupted survival.
`reintroduced_turns` requires an observed `reverted` or `component_absent` followed
by `present`; `reintroductions[].after_state` distinguishes these two paths.
Component-version recurrence is a separate whole-component observation.
An intervening unobserved response alone does not establish disappearance.

`unmatched_after` excludes components reconnected to historical identities;
`reconnected_after` lists those separately. Rename candidates remain unmatched.
`complete_selected_turns` includes an executable last-of-each-section fallback;
`complete_protocol_turns` requires the contiguous complete group. Selection mode
is preserved per turn; the analyzer follows the executor's fallback behavior.

`saved_model_feedback_available` only checks explicit saved model-feedback fields.
`saved_environment_feedback_available` is separate, and raw evaluator errors are
retained as `environment_error_message`. Evaluator output is not evidence of the
exact text ultimately included in the model's context.

## Checks and evidence

```bash
python -m tools.data.trajectory_structure.check_structure
```

These CPU-only workflow checks stay alongside the offline tool, outside slime CI.
The optional `.pt` check needs PyTorch. Development-pool results, manual examples
and remaining limits are documented in
[the E1.1 findings in the experiment plan](../../../handoffs/paper/plan.md#e11-留下的有用线索).

## Single-pass optimization strategy coverage

The canonical registry in `optimization_strategies.py` contains 58 templates
across memory access, reuse, parallel work, pipelines, computation, fusion,
launch scheduling and library configuration. A single analysis records kernel
and device-helper source, native host/configuration calls, and adjacent launch
site changes. It has no retrieval/filter split and never executes candidate code.

```bash
PYTHONPATH=.:local_artifacts/paper/trajectory_structure/deps \
  python -m tools.data.trajectory_structure.check_optimization_strategies
PYTHONPATH=.:local_artifacts/paper/trajectory_structure/deps \
  python -m tools.data.trajectory_structure.optimization_coverage \
  --input train121=local_artifacts/paper/optimization_speedup_pilot/train121.2LyPXF/exported \
  --output local_artifacts/paper/optimization_speedup_pilot/new_coverage
```

Repeat `--input name=path` for other pools. Directory inputs accept `g*.json`
and `trajectory_*.json`; file inputs use the existing JSON/JSONL/trusted `.pt`
adapter. Full saved source is required; padding is excluded. Failed, untimed
and evaluator-rejected turns are analyzed with their original status retained.
Profiling is supplementary evidence, not an inclusion gate.

Outputs are `catalog.json`, `summary.json`, `trajectory_index.json`,
`examples.json`, reviewable per-trajectory source/evidence, and a provenance
`manifest.json`. Observations carry a strategy, component, section, source
location and concrete parameters. Each component/strategy retains up to four
evidence excerpts and the total occurrence count; complete selected source is
also saved. Counts deduplicate by turn, trajectory and input/reference task.
Transition templates count source launch-site changes, not runtime launches.

Coverage reports explicit API/directive observations separately from broader
structural patterns. A plain contiguous copy can match a memory-access pattern;
this is not proof of a new optimization, speed benefit or valid component reward.
All registered templates have constructive CPU examples. Real-pool hits and
remaining scope are documented in the
[strategy report](../../../handoffs/paper/component_reward_training.md).

The opt-in training backend `--component-reward --component-reward-backend source`
uses `examples/kernel_agent/source_components.py` and `source_component_reward.py`.
Install this directory's pinned `requirements.txt` in every rollout environment.

FastCredit adds source credit to baseline TRLOO with
`--component-reward-mode trloo-credit-additive --component-reward-scale 0.25
--component-reward-min-speedup 1.0`. The gate selects correct, measured-fast
anchors; only earlier retained source turns receive the extra credit.
The baseline dynamic filter remains active. See the
[method and limitations](../../../handoffs/paper/component_reward_training.md).
