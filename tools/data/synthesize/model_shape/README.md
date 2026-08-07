# Model-assisted shape hard-tail lane

This is the only supported model-generation path for shape augmentation. It
targets parents that the three historical static-solver rounds could not solve;
those solver implementations remain in place only so their existing artifacts
can be reproduced.

The workflow is deliberately staged:

1. `pipeline.py select` chooses a deterministic, source/operator-stratified
   canary from upstream `no_variable_multislot_product_solution` decisions.
2. `pipeline.py preview` writes the exact user-only prompts pinned by an
   existing run. `preview-candidate` separately renders the current candidate
   prompt without changing an existing run. Targets are sampled at byte
   granularity rather than integer MiB granularity.
3. `deploy_four.sh preflight` verifies node53/node64/node69/node70 as four idle
   8-H20 hosts, official 48-shard weights, the frozen image, both source
   patches, pure TP8, DSPARK, and the low/high/max effort mapping. `start` is a
   separate action.
4. `pipeline.py generate` sends explicit `reasoning_effort=low` requests, with
   no system message, at at most 64 concurrent requests per endpoint.
5. `pipeline.py materialize` rejects non-shape edits, reconstructs every child
   from the original source using only approved integer-span replacements, and
   applies storage, target, balance, dead-tail, and FakeTensor gates.
6. `pipeline.py analyze-bias` reports proposal and accepted anchor bias,
   concentration, slot cardinality, target snapping, source/operator/endpoint
   acceptance, and the projected effect on the known 64,315-row distribution.

No stage mutates the immutable parent parquet. Static acceptance is review-only;
paired target-GPU reference and changed-region validation remain mandatory.

## Why this lane exists

The latest analysis-only 64,315-row substitution still leaves 40,957 parents
(63.68%) unchanged. Current versus KernelBench per-tensor numel P50/P90/P99 is
0.79M/399.5M/1.02B versus 33.6M/1.61B/2.15B. Changed occurrences are also
concentrated: 44.45% are powers of two and the top ten values account for
41.04%. Replacement coverage is 12.85% for CUDA-Agent, 31.95% for DrKernel,
63.91% for KernelBook, and 31.28% for Oubo; operator-dense parents are harder.

The model lane therefore samples from the 17,864 upstream no-product-solution
decisions, uses byte-granularity targets, avoids source/operator-to-endpoint
confounding, and measures proposal bias separately from accepted-child bias.
The 4 GiB aggregate cap remains intentional, so this lane can reduce but cannot
eliminate the extreme KernelBench right-tail gap.

Recommended first run size is 1,000 parents. Do not run `deploy_four.sh start`
or `pipeline.py generate` until the code and rendered prompt have been reviewed.

After review, the staged commands are:

```bash
python -m tools.data.synthesize.model_shape.pipeline select
python -m tools.data.synthesize.model_shape.pipeline preview
python -m tools.data.synthesize.model_shape.pipeline preview-candidate
bash tools/data/synthesize/model_shape/deploy_four.sh preflight
bash tools/data/synthesize/model_shape/deploy_four.sh start
bash tools/data/synthesize/model_shape/deploy_four.sh wait
bash tools/data/synthesize/model_shape/deploy_four.sh smoke
python -m tools.data.synthesize.model_shape.pipeline generate
python -m tools.data.synthesize.model_shape.pipeline materialize
python -m tools.data.synthesize.model_shape.pipeline analyze-bias
```

Each command consumes the fixed run-directory layout from the prior stage.
Transport failures are append-retryable; a successful generation is never
silently overwritten. Use `--run-dir` only when creating a separate experiment.
Generation records carry the prompt version and prompt hash. Preview,
generation, and materialization fail closed on mixed or unknown versions, so a
new candidate template cannot silently alter replay of an existing run.
