# Coverage reward and PRS ablations (Qwen3.8 B300)

These launchers collect the B300 coverage-reward and probabilistic rejection
sampling (PRS) experiments. Shared environment setup remains in
`examples/kernel_agent/qwen38_b300_env.sh`. Runtime preflight and GPU diagnostic
tools are documented in [`scripts/b300/README.md`](../../../scripts/b300/README.md).

## Experiment defaults

All filenames below end in `.sh`. Reward weight refers only to the coverage or
speed auxiliary reward; zero weight does not disable the base reward or other
validity filters. Enabled PRS uses threshold `0.3` and factor `0.1`.

| Launcher | Auxiliary reward type | Weight | PRS coverage key | Recompute layers |
| --- | --- | --- | --- | --- |
| `qwen38_b300_baseline_t1` | `time_coverage` | 0.5 | `time_coverage` | 24 |
| `qwen38_b300_baseline_t1_refcov` | `reference_time_coverage` | 0.5 | `time_coverage` | 8 |
| `qwen38_b300_baseline_t1_effrefcov` | `efficiency_reference_time_coverage` | 0.5 | `reference_time_coverage` | 32 |
| `qwen38_b300_baseline_t1_nocov_noprs` | `efficiency_reference_time_coverage` | 0 | Disabled | 32 |
| `qwen38_b300_baseline_t1_nocov_prs` | `capped_speed_auxiliary` | 0 | `reference_time_coverage` | 32 |
| `qwen38_b300_baseline_t1_refcov_prs` | `reference_time_coverage` | 0.5 | `reference_time_coverage` | 32 |
| `qwen38_b300_baseline_t1_speedaux_prs` | `capped_speed_auxiliary` | 0.5 | `reference_time_coverage` | 32 |
| `qwen38_b300_baseline_t1_timecov025_prs` | `gated_time_coverage` | 0.25 | `reference_time_coverage` | 32 |

These are preserved experiment configurations, not a guarantee that only the
ablation variable differs. In particular, the completed time-coverage baseline
used an explicit `RECOMPUTE_NUM_LAYERS=32` override. Its current default of 24 is
a candidate for future launches, not a validated replacement for that run; the
earlier 8-layer configuration OOMed. The historical refcov launcher still
defaults to 8. Review the full scripts and recorded launch overrides before
claiming a controlled comparison. Directory reorganization does not normalize
these settings.

## Running

These scripts require the prepared B300 deployment, not a generic local Python
environment. Model/data paths, local checkpoint mounts, runtime packages,
KernelGym endpoint, and Ray resources have machine-specific defaults. Check
those settings and active jobs before launching; do not submit duplicate jobs.
The train and rollout roles require separate compatible containers, with
`slime_actor` and `slime_rollout` placement resources. See the shared environment
script and your deployment records for setup.

From the repository root, after preparing the environment and checking resources:

```bash
bash examples/kernel_agent/coverage_ablate/qwen38_b300_baseline_t1_effrefcov.sh
```

Existing environment overrides remain supported. Experiment output paths,
checkpoint locations, Ray submission IDs, and TensorBoard names are unchanged
by the move. Launch automation must use the new script paths; no compatibility
wrappers are retained at the old paths.

The machine-local GEPA-V2 evaluation launchers and artifact validators are kept
separately under `local_artifacts/qwen38_gepav2_eval/` and are intentionally not
versioned. Training scripts here do not launch that evaluation suite.
