# Shape expansion

This directory is the maintained implementation boundary for shape-only data
expansion. It contains the static solver, the DSV4F fallback, target-GPU
validation, final resampling, and distribution/semantic audits.

The method, contracts, results, and known limitations are documented in
[`handoffs/data/synthesize/SHAPE_EXPANSION.md`](../../../../handoffs/data/synthesize/SHAPE_EXPANSION.md).

## Components

- `solve_shape_coverage.py`, `solve_multidim_shape_coverage.py`, and
  `solve_variable_shape_delta.py`: AST analysis, integer shape solving, and
  FakeTensor gates.
- `pipeline.py` and `prompt.py`: DSV4F selection, user-only prompts,
  generation, shape-only materialization, and proposal/acceptance bias.
- `validate_shape_region_liveness.py` and
  `verify_shape_solver_runtime_shards.py`: changed-region H20 validation and
  fail-closed evidence verification.
- `resample_shape_coverage.py`: combine runtime-valid static/model lanes and
  choose at most one child per canonical parent.
- `plot_shape_child_vs_kernelbench.py` and
  `sample_shape_semantic_audit.py`: final distribution and semantic-audit
  evidence. Their JSON/TSV details default to `local_artifacts/`.
- `deploy_four.sh` and `deploy_one.sh`: TP8 SGLang deployment with DSPARK and
  the repository's reasoning-effort/SWA fixes. Eager mode is not used.

Cross-method utilities remain one level above: `augment_prompt_tasks.py`,
`profile_prompt_tvm_distribution.py`, `launch_reference_validation_shards.sh`,
and `validate_train_mode_contract.py` are also used by value, dtype, layout, or
general dataset construction.

## DSV4F pipeline

Use exactly one selection command for a new run, followed by the common
generation and materialization stages:

```bash
python -m tools.data.synthesize.model_shape.pipeline --run-dir RUN select
python -m tools.data.synthesize.model_shape.pipeline --run-dir RUN select-full-residual
python -m tools.data.synthesize.model_shape.pipeline --run-dir RUN select-relaxed-residual
python -m tools.data.synthesize.model_shape.pipeline --run-dir RUN select-unprofiled-residual

python -m tools.data.synthesize.model_shape.pipeline --run-dir RUN preview
python -m tools.data.synthesize.model_shape.pipeline --run-dir RUN generate
python -m tools.data.synthesize.model_shape.pipeline --run-dir RUN materialize
python -m tools.data.synthesize.model_shape.pipeline --run-dir RUN analyze-bias
```

`preview-candidate` renders the current candidate template without changing a
run pinned to an older prompt version. Generation uses a user message only,
`reasoning_effort=low`, a 128K maximum completion budget, and up to 64
concurrent requests per endpoint.

Materialization reconstructs each child from the immutable parent using only
approved integer-span replacements. Static/FakeTensor acceptance is not final:
paired parent/child H20 reference validation and changed-region validation are
required before a child enters the final census.

## Deployment

```bash
bash tools/data/synthesize/model_shape/deploy_four.sh preflight
bash tools/data/synthesize/model_shape/deploy_four.sh start
bash tools/data/synthesize/model_shape/deploy_four.sh wait
bash tools/data/synthesize/model_shape/deploy_four.sh smoke
```

`preflight` verifies the configured H20 hosts, official model shards,
container image, SGLang patches, TP8, DSPARK, and reasoning-effort mapping.
