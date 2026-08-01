# scripts/dsv4 — DS-V4 run and tooling index

Keep the top level limited to operational entry points and the helpers they
source directly. Reproducers and experiment-specific analysis belong in the
subdirectories below.

## Operational entry points

| Path | Purpose |
|---|---|
| `run.t1.deepseek_v4_flash.rl.sh` → `full_loop_smoke.sh` | Formal/smoke launch chain |
| `_dsv4_task_args.sh` | Shared task-argument assembly |
| `train_smoke.sh` / `rollout_smoke.sh` | Isolated train and rollout bring-up |
| `convert_torch_dist.sh` | HF-to-`torch_dist` conversion and chained verification |
| `verify_rollout_dump.py` | Saved-rollout inspection |
| `patch_sglang_dense_attn.py` / `patch_sglang_dsv4_mhc_sinkhorn_torch.py` | Dedicated rollout parity patchers |

## Reusable tooling

| Directory | Scope |
|---|---|
| `diagnostics/parity/` | Primary byte-parity, dtype adjudication, mHC, and activation-memory probes |
| `diagnostics/mtp/` | Speculative frontier analysis and the Triton decode microbenchmark |
| `diagnostics/mismatch/` | Paired SGLang/Megatron full-vocabulary probes |
| `diagnostics/lora/` | Standalone LoRA reload reproducer and driver |
| `studies/entropy/` | Fixed-rollout entropy and predictive-tail experiment drivers |
| `tools/` | Offline conversion, checkpoint assembly, dataset-state recovery, and CPU inspection |

Closed one-off probes are retained only under ignored
`local_artifacts/deepseek-v4/retired_scripts/diagnostics/`. Design and evidence start at
`handoffs/deepseek-v4/release/00_overview.md`; reviewable logs belong under
`local_artifacts/deepseek-v4/`.
