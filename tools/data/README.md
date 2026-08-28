# Data tooling

`tools/data` contains reproducible dataset build tools. Generated data belongs
under `Data/`; review evidence and large audits belong under
`local_artifacts/`.

## Current cleanup interface

The production cleaner has one fixed acceptance policy. Callers choose input,
output, resource limits, and optional cross-corpus baselines; they cannot
silently weaken semantic checks from the command line.

Run a complete cleanup on GPU:

```bash
python -m tools.data.cleaning.pipeline clean INPUT.parquet OUTPUT.parquet \
  --workers 4 --timeout 60 --overwrite
```

Run only the global static and within-corpus deduplication pass before splitting
work across GPUs:

```bash
python -m tools.data.cleaning.pipeline static INPUT.parquet STATIC.parquet \
  --overwrite
```

Reuse a complete source-row audit and rerun selected rows. The indices file
contains one source row index per line; recovery defaults to a 300-second
per-row budget:

```bash
python -m tools.data.cleaning.pipeline recover INPUT.parquet RECOVERED.parquet \
  --prior-audit PRIOR.audit.jsonl \
  --indices timeout_indices.txt \
  --workers 4 --overwrite
```

Every command derives four sidecars from `OUTPUT.parquet`:

- `OUTPUT.audit.jsonl`: one decision record per source row;
- `OUTPUT.summary.json`: counts, hashes, resource settings, and the full policy;
- `OUTPUT.rejected_samples.txt`: representative rejected references;
- `OUTPUT.quarantine.parquet`: every row isolated by the quarantine policy.

Passing `--dedup-against BASELINE.parquet` enables exact semantic, token
Jaccard, and AST structural checks against that baseline. The option is
repeatable. It is deliberately absent by default because benchmark
decontamination is a separate policy decision.

## Fixed acceptance policy

`CleanupPolicy` in `cleaning/pipeline.py` is the source of truth. The current
policy fixes these values:

| Check | Fixed value |
| --- | --- |
| Execution device | GPU |
| Seeds | 3 |
| Same-input forwards per seed | 3 |
| Natural `get_inputs()` draws per seed | 20 |
| Calls used to confirm a fixed input generator | 40 |
| Minimum changed fraction in one output leaf | `1e-4` |
| Numeric tolerance | `rtol=1e-4`, `atol=1e-5` |
| Per-forward-argument sensitivity | Required |
| Train/eval audit | All statically admissible rows |
| Ops metadata normalization | Enabled |
| Random forward and unused forward arguments | Rejected |
| Token-Jaccard threshold | Strictly greater than `0.8` |
| AST structural threshold | Strictly greater than `0.9` |

The policy quarantines inconclusive sensitivity verdicts, low natural-output
activity, and dead computation with RNG, stateful, or unresolved effects. It
also contains the three UUIDs that manual review confirmed should never be
used. The summary records the complete resolved policy and its SHA-256 hash.

CPU execution and reduced probe budgets are available only through the typed
Python API for focused tests and diagnostics. They are not production CLI
options.

## Sharded execution

Shard helpers share one module entry point:

```bash
python -m tools.data.cleaning.shards split ...
python -m tools.data.cleaning.shards merge ...
python -m tools.data.cleaning.shards overlay ...
```

Run `static` once on the complete source before splitting. This preserves
global within-corpus semantic deduplication. `merge` restores source order for
the GPU candidate artifacts; `overlay` places shard runtime evidence back onto
the complete static audit so static rejects remain in the final denominator.

## Other commands

| Module command | Responsibility |
| --- | --- |
| `python -m tools.data.cleaning.runtime_validation` | Diagnose one executable PyTorch reference. |
| `python -m tools.data.cleaning.external` | Convert supported source datasets with provenance. |
| `python -m tools.data.cleaning.subsets` | Materialize an audit-defined subset. |
| `python -m tools.data.cleaning.audit_summary` | Aggregate a complete audit without executing references. |
| `python -m tools.data.cleaning.profile_ops_reference_smoke` | Profile import, construction, transfer, and one forward without assigning a verdict. |
| `python -m tools.data.cleaning.deduplicate_ops_candidates` | Near-deduplicate candidate pools. |
| `python -m tools.data.cleaning.stratify_ops_complexity` | Assign structural complexity labels and cap Level1-like rows. |

Reusable logic remains separated by responsibility:

- `static_analysis.py`: AST contract, dataflow, RNG, dead-computation, and
  reachable-module checks;
- `runtime_validation.py`: deterministic execution, input/output sensitivity,
  finite-output, and train/eval checks;
- `pipeline.py`: the fixed policy, orchestration, Parquet rewrite, and audit;
- `similarity.py` and `ast_similarity.py`: cross-corpus similarity primitives;
- `external.py`: source conversion and provenance;
- `subsets.py`: audit-aligned partitions;
- `shards.py`: split, merge, and overlay operations;
- `complexity.py`: KernelBench-calibrated structural classification;
- `audit_summary.py`: deterministic audit aggregation.

Historical cleanup results and the exact artifact hashes used for training are
preserved in `handoffs/data/cleaning/`. The removed incremental migration flags
were specific to superseded intermediate audits; recovery now accepts an
explicit prior audit and an auditable row-index file.
