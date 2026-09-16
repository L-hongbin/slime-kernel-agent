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

## First-correct repair-diff validation

The current whole-turn training-snapshot workflow is `repair_credit_training.py`:

```bash
PYTHONPATH=.:local_artifacts/paper/trajectory_structure/deps \
  python -m tools.data.trajectory_structure.check_repair_credit_training
PYTHONPATH=.:local_artifacts/paper/trajectory_structure/deps \
  python -m tools.data.trajectory_structure.repair_credit_training prepare \
  --plan /path/to/training_snapshot_plan.json --output /path/to/new_audit
PYTHONPATH=.:local_artifacts/paper/trajectory_structure/deps \
  python -m tools.data.trajectory_structure.repair_credit_training shadow \
  --audit /path/to/audit --labels /path/to/replay_labels.json \
  --output /path/to/new_shadow
```

The plan and an adjacent `input_sync.json` identify already synchronized trusted
training dumps and frozen runtimes. The scoped pilot requires full 256x3 batches
and full 16-trajectory prompt groups; it is not a general training-set loader.
It verifies saved gamma=1 returns, executes the archived scalar TRLOO postprocess,
and independently checks leave-one-out arithmetic without modifying masks or
training data. A whole adjacent turn is one bundle: transport blocks are not
credit units. All B changes must transfer exactly and uniquely from T2 to T1.
Conflicts, missing sections and unchanged-source outcome flips remain unknown.

Training cases carry the recorded per-task precision and entry point. Replay
plans may contain source-hash-bound historical client prechecks; rejected variants
are recorded locally without POST. Explicit precision overrides must agree with
the recorded case. A failed correct control stops interpretation; an ambiguous
submission is never automatically retried. The shadow coefficients are diagnostic
settings only, with no optimizer update or policy-gradient inference.

The earlier evaluation-pool local-diff workflow remains available below.

`repair_credit_audit.py` audits correctness repairs on original TRLOO, independently
of the optimization registry and FastCredit. It selects the first correct,
compiled, non-decoy completed turn, compares the actual executor-selected source
to preceding turns, and records retained local-edit candidates. T1 is the base
version, not an automatically rewarded origin. All-failed trajectories have no
correct anchor. No reward is assigned, and unknown alignment is not a negative
contribution label.

```bash
PYTHONPATH=.:local_artifacts/paper/trajectory_structure/deps \
  python -m tools.data.trajectory_structure.check_repair_credit_audit
PYTHONPATH=.:local_artifacts/paper/trajectory_structure/deps \
  python -m tools.data.trajectory_structure.repair_credit_audit \
  --input /path/to/trusted/baseline/eval_0.pt \
  --runtime-root /path/to/evaluated/runtime_repo \
  --output local_artifacts/paper/repair_credit_validation/new_audit
```

The CLI currently validates the complete three-turn KernelBench eval layout:
800/800/400 trajectories across L1/L2/L3. It is a scoped validation driver, not
yet a general training-set loader. It checks the response-selection parser
against the frozen evaluated code and records input/code hashes. Outputs include
`summary.json`, per-case source/diffs/feedback, a seeded review selection, and a
manifest. Selection for review does not itself mean completed manual annotation.

Strict categories stay separate from `syntax_fallback_candidates`: the latter
allows malformed native functions with exact identity and unique old/new token
contexts to supply lexical evidence, while keeping incomplete programs and
Python AST failures unknown. Neither branch proves a contribution to correctness.
To supplement a completed strict audit without overwriting it:

```bash
PYTHONPATH=.:local_artifacts/paper/trajectory_structure/deps \
  python -m tools.data.trajectory_structure.repair_credit_audit \
  --existing-audit /path/to/completed/audit \
  --output /path/to/completed/audit/new_syntax_supplement.json
```

`repair_credit_replay.py` prepares explicitly planned, exact-source edits. Its
default is prepare-only; `--submit` sends one case at a time to an already deployed
KernelGym. A plan names the saved cases, base turns, exact scoped replacements,
evaluation settings and required correct controls. The helper is not an automatic
semantic patch generator. It records payloads and IDs before POST and does not
automatically retry ambiguous submissions or redeploy services.

```bash
python -m tools.data.trajectory_structure.repair_credit_replay \
  --plan /path/to/reviewed/replay_plan.json \
  --output /path/to/new_preflight
# Add --submit --url <existing-service> only for authorized execution.
```

The current method, coverage, manually inspected counterexamples and controlled
replays are in [the correctness diff-credit report](../../../handoffs/paper/correctness_diff_credit.md).

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
`source_credit_pilot.py` audits historical source allocations with explicitly
reconstructed eligibility; its output is never training data. It uses the
historical allocation rule without the additive mode's anchor speedup gate and
may restore soft-finalization removals. It cannot reproduce current FastCredit
training targets or masks.
`source_credit_rollout_audit.py` instead checks a newly saved real rollout dump
without reconstructing identity, rewards, tokens, masks or predictive support:

```bash
PYTHONPATH=. python -m tools.data.trajectory_structure.source_credit_rollout_audit \
  --input /path/to/real/rollout_0.pt \
  --output local_artifacts/source_credit_rollout_audit
python -m tools.data.trajectory_structure.source_credit_pilot \
  --input /path/to/exported_trajectories --output /path/to/new_pilot
python -m tools.data.trajectory_structure.check_source_credit_audits
```

Both tools require new output directories. Only load trusted local `.pt` files;
they use `torch.load(weights_only=False)`. The rollout audit checks serialized
replacement-mode targets or additive-mode baseline rewards and targets, plus
the predictive top-k support required by DPPO. An absent predictive payload on
an active turn is an audit failure, even if another training configuration does
not use predictive DPPO. Saved additive scale and gate values are checked for
internal consistency; the audit does not verify a separate launch configuration
or execute trajectory packing. Historical reconstruction is explicitly marked
in the pilot output and does not satisfy the real-rollout audit.

FastCredit adds source credit to baseline TRLOO with
`--component-reward-mode trloo-credit-additive --component-reward-scale 0.25
--component-reward-min-speedup 1.0`. The gate selects correct, measured-fast
anchors; only earlier retained source turns receive the extra credit.
The baseline dynamic filter remains active. See the
[method and limitations](../../../handoffs/paper/component_reward_training.md).

## Correct-only optimization-feature speedup pilot

`optimization_pilot.py` describes optimization features in correctly evaluated
kernels, links them to saved profiling events and compares successive eligible
correct turns. It uses the shared registry; program-level configuration and
cross-turn strategy coverage are maintained by `optimization_coverage.py`

```bash
python -m tools.data.trajectory_structure.check_optimization_pilot
python -m tools.data.trajectory_structure.optimization_pilot \
  --structure-root /path/to/trajectory_structure \
  --output /path/to/new_speedup_pilot
```

Install the pinned parser dependencies as described above, and run from the
repository root. The input structure archive must retain its original raw-data
manifest and extracted `g*.json` files. The output directory must be new

The workflow verifies raw-response hashes and correct-only timing ratios, emits
per-kernel source evidence and searches earlier correct versions of the best
answer for unique normalized kernel structures. It records input/code hashes
and checks stability at completion. The earlier correct turns can be
nonadjacent in the original trajectory

Outputs describe source features and retention candidates. They do not establish
execution of every branch, general fusion/coalescing recognition or per-feature
speed contributions. Launch arguments, helper bodies and semantic call roles
are outside this matcher. Reduction postprocessing is reported separately because
it can also describe required arithmetic. The 5% timing screen is not a confidence
interval; profiling is a separate invocation from timing trials. Historical
findings are in the [component-reward report](../../../handoffs/paper/component_reward_training.md#附录历史试验)

## Local revision / partial-output matching validation

`implementation_match.py` builds bounded symbolic value descriptions for native
kernels. It normalizes local names and exchanges addition/multiplication operands
without reassociation, while preserving guards, formal input roles, output
addresses, source context and pragmas. Loops remain opaque structured regions

A partial candidate requires an after-kernel output value to occur in a
before-kernel stage with a compatible output port. Unknown helpers, repeated
local declarations and unsupported side effects produce unknown results rather
than matches. These descriptions do not establish mathematical equivalence,
automatic fusion counts or training credit

```bash
python -m tools.data.trajectory_structure.check_implementation_match
python -m tools.data.trajectory_structure.match_validation \
  --input /path/to/exported_training_trajectories \
  --freeze /path/to/current_source_freeze.json \
  --legacy-manifest /path/to/earlier_structure/manifest.json \
  --previous-pool /path/to/earlier_training_exports \
  --output /path/to/new_validation
```

The freeze JSON contains a `code_sha256` mapping from source paths to SHA-256
hashes. Freeze the matcher, validation driver, parser and feature-analysis
dependencies before examining candidate outcomes; a changed file stops the run

The driver excludes raw/Python-AST reference overlap with the earlier pools,
then compares exact and normalized matching on the same correct-timed records.
Each method resolves one-to-one matches independently. Outputs retain selection,
source sections, matched regions, timing evidence and input/code hashes. The
input adapter and run identity are scoped to the retained rollout121 experiment;
this is not a general sampling pipeline, and no generated CUDA code is executed

The historical frozen validation found no additional natural matches. Its
original source hashes are required to reproduce that result; a run with current
code is a separate validation. See the [method and manual findings](../../../handoffs/paper/component_reward_training.md)

## Best-answer source membership

`best_answer_credit.py` selects the earliest maximum among consistent correct
recorded evaluations, then traces exact component versions and unique historical
token blocks to their observed source turns

```bash
python -m tools.data.trajectory_structure.best_answer_credit \
  --structure-root /path/to/trajectory_structure \
  --coverage component \
  --output /path/to/new_membership
python -m tools.data.trajectory_structure.check_best_answer_credit
```

`--coverage component` keeps locally valid components from incomplete or partly
malformed intermediate submissions. `--coverage program` requires whole-program
coverage. If the winning answer's source is unavailable, the result remains
unknown; the tool does not choose a lower-scoring answer instead

Outputs contain component/token origins, exact-version first sightings,
possible origins for unknown tokens, and per-turn membership (`1`, `0`, or
`null`). A copied version retains its earlier origins. Short or repeated token
matches remain uncertain; strict signature identity can miss cross-signature
reuse. These are normalized syntax tokens, not model tokens

The CLI uses diagnostic raw speedup for winner selection; the Python API accepts
an explicit score mapping for `q(K)`. All-failed trajectories emit no membership
labels. Output directories must be new, and the manifest records input/code
hashes and their final stability checks. This is an offline retention heuristic:
it neither establishes causal contribution nor modifies training rewards. The
proposed retention-ratio return and current training method are discussed in
[the component-reward report](../../../handoffs/paper/component_reward_training.md)

## Component state observations

`prepare_component_replays` exports twelve fixed, manually selected submissions
as source-hashed JSON payloads for a separate KernelGym tracing process. These
cases cover launch patterns, descriptor caches, workspace and interface repairs;
they are a diagnostic validation set, not a random prevalence sample

```bash
python -m tools.data.trajectory_structure.prepare_component_replays \
  --raw-root /path/to/extracted_trajectories \
  --structure-root /path/to/trajectory_structure \
  --output /path/to/new_replay_inputs
python -m tools.data.trajectory_structure.component_state \
  --structure-root /path/to/trajectory_structure \
  --runtime-root /path/to/saved_runtime_observations \
  --payload-root /path/to/replay_inputs \
  --output /path/to/new_state_inventory
python -m tools.data.trajectory_structure.native_state \
  --runtime-root /path/to/saved_runtime_observations \
  --output /path/to/new_native_state
python -m tools.data.trajectory_structure.check_component_state
python -m tools.data.trajectory_structure.check_native_state
```

All output directories must be new. The exporter checks raw response identity
against the selected source. `component_state` verifies candidate-code and
structural-source hashes before joining source declarations, FFI calls, CUDA
activities and tensor snapshots. Shared storage establishes argument presence;
read/write effects and output dependence remain unknown

`native_state` interprets selected successful cuBLAS/cuDNN API events from
`native_state.jsonl`, scoped by the observer ID in `observation.json` and probe
loss/error counts in `result.json`. It records object generations and workspace
resets. Missing hook coverage or dropped events cannot establish effective
workspace state. API arguments record requested policy, not proof of particular
hardware instructions; missing destroys do not establish memory leaks

These commands process saved data without executing generated code or contacting
a service. Collecting new observations requires a separate tracing adapter with
a matching source manifest. Raw traces, argument snapshots and host API events
remain separate from evaluator feedback and scored timing

## State aggregation

`state_aggregation.py` builds reference-aligned operation graphs with source
evidence, bounded address checks, adjacent-turn differences and pairwise
comparisons. Its adapters are maintained in `structural_state_pilot.py`; the
address and comparison rules live in `semantic_state.py`

```bash
python -m tools.data.trajectory_structure.check_state_aggregation
python -m tools.data.trajectory_structure.state_aggregation \
  --structure-root /path/to/trajectory_structure \
  --output /path/to/new_structural_state
```

This is a manually reviewed two-task adapter pilot: MLP and MiniGPTBlock, two
trajectories per task/model, snapshots 1–4 (32 cards total). It does not claim
automatic semantic mapping for arbitrary generated programs. Source locators
were manually aligned to reference operations. Each snapshot describes the
state before the *next* decision; future answers/results are excluded

Cards distinguish `reference_inputs/reference_output` from `candidate_wiring`
Initial parameter roles and MLP loop/activation mappings are manually reviewed
The attention adapter tracks buffer value roles through selected driver calls;
unsupported paths and untracked reads stay unknown. Operation mapping coverage
means that the selected reference slots have source anchors, not that all
execution paths or numerical contracts are validated

The CPU address checker uses actual Sgemm/StridedBatchedGemm call arguments,
driver-to-helper scalar argument binding, integer declarations read from the
helper's source, and reviewed outer shape environments. It checks leading dimensions, selected factor addresses, batch
strides/counts. A mismatch carries a concrete witness; a pass only means the
checked address constraints agree. Scalar coefficients, complete numerical
behavior, runtime aliasing, library heuristics, GemmEx/Lt descriptors and custom
kernel correctness remain outside that check

`same_computation_plan` compares operation implementations and tracked activation,
weight/bias roles (including explicit transposes), not source text. It is a candidate relation for further inspection, not state
or policy equivalence. `same_recorded_state` is an audit of equality of currently
recorded facts, **not** a training merge rule. Unknown wiring cannot establish
a match. Shared checked-address slots are reported separately, so whole-program
differences do not erase local common structure. Whole-program Correct never
sets an untested operator's local correctness to true

Outputs include Markdown/JSON `cards/`, `pairs.json`, `transitions.json`,
`summary.json` and a before/after source-stability `manifest.json`. The output
directory must be new. No generated CUDA code is executed; no GPU, live
KernelGym endpoint, reward change or RL update is involved
Read the [method and manually inspected examples](../../../handoffs/paper/state_aggregation.md)
