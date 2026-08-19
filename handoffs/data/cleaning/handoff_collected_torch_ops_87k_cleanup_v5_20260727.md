# Collected Torch Ops 87k cleanup v5

Date: 2026-07-27
Status: CPU cleanup complete; target-GPU and provenance approval pending

## Decision

The colleague delivery is useful, but it is not an 87k-row independent augmentation set. Of 87,446 source rows, 81,743 converted rows are exact semantic duplicates of existing DrKernel, CUDA-Agent, or KernelBook references. The final strict CPU candidate contains 3,502 rows (4.0048% of the source): 2,065 with no train/eval difference observed and 1,437 with an observed mode/RNG/state boundary.

Do not merge the 3,502-row file directly into production training yet. The delivery lacks row-level upstream URL, revision, and license, and target-H20 eager/compile validation has not run. The safest next input to GPU validation is the 2,065-row `mode_same_observed` partition. The 1,437 mode-variant rows remain useful if the generation/evaluation contract explicitly fixes `ModelNew.eval()` semantics.

## Data lineage and funnel

| Stage | Rows | Retention from prior stage | What happened |
| --- | ---: | ---: | --- |
| Colleague source | 87,446 | — | Immutable input delivery |
| Converted to current schema | 87,440 | 99.9931% | Six KernelBook rows lacked a top-level `Model` |
| Static v5 survivors | 4,893 | 5.5958% | Canonical dedup, AST/dataflow/effect checks, archive exclusion, KernelBench L1/L2/L3 near-overlap gate |
| Initial robust runtime survivors | 3,599 | 73.5541% | Three seeds, up to 20 natural draws, fixed-input confirmation, output-activity floor |
| Final per-argument survivors | 3,502 | 97.3048% of initial runtime survivors | Sixty argument-sensitivity quarantines and 37 extra CPU resource timeouts |
| Conservative mode-same partition | 2,065 | 58.9663% of final survivors | Full-corpus train/eval audit observed no difference |
| Explicit mode-variant partition | 1,437 | 41.0337% of final survivors | Output, state, stochasticity, or train-only failure differs |

Converted-source composition and final yield:

| Coarse source label | Converted | Static survivors | Final CPU candidates |
| --- | ---: | ---: | ---: |
| `drkernel_level0` | 71,912 | 1,739 | 1,162 |
| `drkernel_sft` | 2,682 | 2,523 | 1,865 |
| `kernel_book` | 11,433 | 591 | 475 |
| CUDA-Agent family | 1,342 | 40 | 0 |
| CUDA archive levels 2/3 | 71 | 0 | 0 |
| **Total** | **87,440** | **4,893** | **3,502** |

The final lack of CUDA-Agent rows does not prove that source is bad. Its last survivor moved into the resource-timeout pool when the stricter per-argument pass added more forwards.

## Static findings

Primary reasons are mutually exclusive; nonexclusive counts show the full incidence of each condition.

| Static condition | Primary rows | Nonexclusive rows | Policy |
| --- | ---: | ---: | --- |
| Exact semantic duplicate against existing corpora | 74,522 | 81,743 | Reject |
| Dead forward computation | 2,812 | 2,812 | Reject |
| Random forward | 2,770 | 3,272 | Reject unless mode-controlled |
| Global used instead of same-named constructor argument | 909 | 989 | Reject |
| Forward output independent of inputs | 634 | 1,198 | Reject |
| Unused initialized module/parameter | 272 | 4,014 | Reject |
| Forward has no valid inputs | 215 | 227 | Reject; `**kwargs` remains a valid input contract |
| Unused forward argument | 78 | 1,726 | Reject |
| Excluded archive level | 68 | 71 | Quarantine by project policy |
| Token Jaccard `> 0.8` against official KernelBench L1/L2/L3 | 5 | 50 | Reject benchmark-near rows |
| Stateful/RNG/unknown dead effects | 109 | 276 | Quarantine rather than call pure dead computation |

The 64-row increase from the earlier v4 exact-duplicate result came from applying entry-point repair and shadowed-contract/random-scalar canonicalization to baseline corpora as well as candidates. Raw helper metadata such as `entry_point=Mish` can no longer hide an effective `Model` duplicate.

## Runtime findings

The final runtime table partitions all 4,893 static survivors.

| Runtime disposition | Rows | Rule |
| --- | ---: | --- |
| Passed final CPU contract | 3,502 | All three validation seeds passed |
| CPU execution failed | 536 | 438 timeout; 98 Python/runtime errors |
| Fixed `get_inputs()` values | 363 | All 40 consecutive input calls exactly equal |
| Synthetic sensitivity only | 244 | Natural draws did not change output; only hidden-distribution probes did |
| Natural sensitivity inconclusive | 144 | Static value flow exists, but all runtime probes matched |
| Per-forward-argument sensitivity inconclusive | 60 | At least one logical positional/keyword argument never changed output under natural replacement plus up to three domain-conscious probes |
| Natural output activity below `1e-4` | 15 | Output changed, but no output leaf changed enough elements |
| Different input, unchanged output | 11 | Natural and all synthetic probes matched with no confirming static ambiguity |
| Non-finite output | 10 | Baseline or fresh output contained NaN/Inf |
| Same-input output mismatch | 7 | Three identical-input forwards were not stable in eval mode |
| Unstable input structure | 1 | `get_inputs()` changed container/shape/type structure |

Failure categories within the 536 execution failures:

| Failure category | Rows | Treatment |
| --- | ---: | --- |
| Timeout | 438 | Separate resource quarantine; not evidence of dirt |
| `NameError` | 66 | Reject |
| `TypeError` | 26 | Reject |
| `RuntimeError` | 2 | Reject pending case review |
| `AttributeError` | 2 | Reject |
| `KeyError` | 2 | Reject |

The original robust pass took 1,502.9 seconds. Full train/eval completion took 622.6 seconds, the retained-row per-argument pass took 788.1 seconds, and the weight-norm repair pass took 11.7 seconds. Runtime work used 112 concurrent disposable processes with one Torch thread each. It is still slow because each row is untrusted Python, three independent seeds must pass, up to 20 natural draws and additional probes may execute, and each resource-heavy row can occupy a worker for 60 seconds. Concurrency was enabled throughout; serial execution would have been much slower.

## Train/eval behavior

The initial audit scheduled mode checks only for statically discovered Dropout/BatchNorm candidates. Manual review found nested custom modules that hid mode-sensitive children. The later audit covered every retained row and found 45 additional output differences, 43 additional train-only stochastic rows, and one additional state difference. The current `CleanupPolicy` makes full train/eval coverage mandatory rather than exposing it as a CLI choice.

Final kept-row mode evidence is:

| Nonexclusive mode evidence | Kept rows |
| --- | ---: |
| `train_eval_same` | 2,065 |
| `train_eval_output_diff` | 1,429 |
| `train_only_stochastic` | 1,387 |
| `train_eval_state_diff` | 84 |
| `train_only_failure` | 1 |
| Any explicit mode variant | 1,437 |
| Mode audit unresolved | 0 |

Four weight-normalized KernelBook models initially failed because current PyTorch cannot deepcopy their parametrized tensors. The mode validator now falls back to rebuilding the model and loading the same cloned state. All four resolved to `train_eval_same`; a regression test covers this path.

## Manual review

The second review read 140 unique rows: 59 final survivors and 81 exclusions/quarantines. Selection was deterministic and stratified, not an unbiased prevalence estimate.

| Cohort | Read | Result |
| --- | ---: | --- |
| Final `mode_same_observed` survivors | 31 | No hard miss observed |
| Final mode-variant survivors | 28 | Mode classification supported; keep partitioned |
| Per-argument sensitivity quarantine | 42 of 60 | All 42 supported under the strict anti-memorization contract |
| Low natural-output activity | 15 of 15 | All 15 exclusions supported |
| CPU timeout/resource boundary | 24 | 20 likely valid but resource-heavy; four also had semantic concerns |
| **Unique rows read** | **140** | **140 dispositions supported; zero final-survivor hard misses** |

Notable cases and tradeoffs:

- A forward branch computed from `x1` was later overwritten; only `x1.size(...)` affected output shape. Static value-flow considered it dependent, while the new per-argument runtime check correctly quarantined it.
- Several equality, `isin`, `isinf`, `isnan`, and Boolean-reduction tasks were formally connected to an input but effectively constant under continuous random inputs.
- Fixed shape tuples, `None` masks, einsum strings, and scalar controls are policy quarantines, not necessarily incorrect programs. They are excluded because the current contract rejects forward arguments whose values the generated kernel can safely hardcode or ignore.
- All 15 low-activity rows were reviewed. Examples included a single accidental equality bit in a million-element tensor and a large fixed count that hid smaller variation under evaluator tolerance.
- Timeout rows mix valid giant tensors with bad semantics. Examples included 2+ GiB inputs for reductions/convolutions, but also `isnan(BatchNorm(random))` and `isinf(random)` outputs that are almost always constant. They must be revalidated, not automatically recovered.

The row-level review manifest is reproducible:

- `local_artifacts/data_handoffs/collected_torch_ops_87k_v5_manual_review_20260727.json`

## Same-domain assessment

The retained rows are broadly same-domain with KernelBench levels 1/2/3: each asks for a CUDA kernel that matches an executable PyTorch `Model` plus `get_inputs()` contract. The source mix is not identical to KernelBench:

- `drkernel_sft` is closest to KernelBench level 1/2 operator composition.
- `drkernel_level0` is synthetic and often combines unrelated Boolean, loss, dropout, and reduction operations; it is same-task-format but has a stronger synthetic distribution shift.
- `kernel_book` contains real repository-derived modules, custom nested classes, attention, and model blocks. It extends compositional diversity beyond many KernelBench tasks but carries much weaker provenance and more framework-specific behavior.
- CUDA-Agent rows are same-domain, but none survived the strict final CPU budget in this delivery.

All 250 official KernelBench L1/L2/L3 references were held out. Fifty collected rows exceeded Python-token Jaccard 0.8 against that holdout and were removed; no official reference is added to training. MultiKernelBench remains excluded by explicit project decision.

## Artifacts

Paths are repository-relative.

| Artifact | Rows | SHA-256 | Intended use |
| --- | ---: | --- | --- |
| `Data/collected_torch_ops_DedupKernelBench_87k_fixed_validated.parquet` | 87,446 | `d0a873f81846f262f06204c2f0709dc0ec9eeae21ed5b83935abe26c4b5f2c11` | Immutable source |
| `Data/external/converted/collected_torch_ops_87k.converted.parquet` | 87,440 | `6f6261b9eecb3599e61d740cf91d3d3ee225e97fd8df7eb951dc78ca7a11be86` | Schema-normalized source |
| `Data/external/converted/collected_torch_ops_87k.static.v5.parquet` | 4,893 | `ee1b2cb381de29226977114125113f2070373d19ec440bb3b0673a553ce27c6b` | Static survivors |
| `Data/external/converted/collected_torch_ops_87k.clean.v5.final.candidate.parquet` | 3,502 | `3d7f31a19f513c8732cdfc2f7f513f1c15c9084229cf9b037f02d68f3c471b07` | Complete CPU candidate, not yet production-approved |
| `Data/external/converted/collected_torch_ops_87k.clean.v5.mode_same_observed.candidate.parquet` | 2,065 | `46613d4910c467bad24f1b810f3e535342d964af2a7439e6c04730cc05fec531` | Conservative next GPU-validation pool |
| `Data/external/converted/collected_torch_ops_87k.clean.v5.mode_variant.candidate.parquet` | 1,437 | `0f6f4298deda312635a9ff99990906a78492cd27f16c408ab7ebcaf1a01bcac1` | Conditional pool requiring explicit eval-mode policy |
| `Data/external/converted/collected_torch_ops_87k.final.quarantine.v5.parquet` | 463 | `8728d335086eb648582de882839e2db93a24265ff1c1653cbcf2f4a73ebb9f9c` | Sensitivity/activity quarantine |
| `Data/external/converted/collected_torch_ops_87k.resource_timeout.v5.final.parquet` | 438 | `682d9e623fe95166f65f54dd566744d1eb0b2f5cb792857093e962e45057b571` | Resource recovery pool |
| `Data/external/converted/collected_torch_ops_87k.clean.v5.final.candidate.audit.jsonl` | 4,893 records | `7ac1a3936809081ff27060c96ecb60fd837b81327c92d9cc171e5adb47937df0` | Row-level final decision ledger |
| `Data/external/converted/collected_torch_ops_87k.clean.v5.final.candidate.summary.json` | — | `88ea06c9726b1139ba29019b779289135f0d4dbd5bda6f930f5bddfa4f05bc5c` | Aggregate settings/counts/hashes |

The converter provenance sidecar is `Data/external/converted/collected_torch_ops_87k.converted.provenance.jsonl` with SHA-256 `3dca03aeaf48913cf590f5ca7609e7f3e8b23d19eb34eb37cde4864d2907f561`. Its `provenance_status` is deliberately incomplete.

## Code and verification

V5 changes are in:

- `tools/data/cleaning/runtime_validation.py`: multi-seed output activity, every-forward-argument probes, full mode audit, weight-norm rebuild fallback.
- `tools/data/cleaning/pipeline.py`: repaired-baseline deduplication, mandatory all-retained mode scheduling, fixed quarantine policy, and concurrent runtime orchestration.
- `tools/data/cleaning/external.py`: streaming colleague-delivery and official KernelBench conversion.
- `tools/data/cleaning/subsets.py`: audit-aligned policy/resource partitions and their module CLI.
- `tests/tools/data/test_ops_data_cleaning.py`: 79 focused regression tests.

The complete focused suite passed: `79 passed` in 10.32 seconds. Warnings were dependency deprecations only.

## Remaining blockers and next actions

1. Recover upstream URL/revision/license per row or per source group. Until then, even the 2,065 conservative rows are not cleared for production training.
2. Run eager and compile/correctness validation on an isolated target H20. At handoff time all eight local H20s were 97-98% utilized with 77-81 GB allocated, so using them would interfere with active work.
3. Re-run the 438 timeout rows on the target GPU with a larger but bounded budget, then pass recovered rows through the same output-activity and per-argument checks. Do not bulk-admit them.
4. Decide whether the evaluator always forces eval mode. If yes, the 1,437 mode-variant rows can be considered after GPU validation; otherwise keep them out of the default mixture.
5. Group-split by upstream repository/task family before sampling and cap the new corpus mixture weight. The 3,502 survivors add diversity, but they are not independent enough to justify an unweighted append.
6. Add semantic static support for shape-only dataflow and probability/Boolean saturation. Runtime probes catch many cases, but a static explanation would reduce CPU cost and make the rule easier to audit.
