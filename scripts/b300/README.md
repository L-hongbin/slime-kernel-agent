# Qwen3.8 B300 runtime checks and benchmarks

These tools validate the prepared Qwen3.8 B300 training/rollout environment.
They are deployment-specific checks, not a general environment installer.
Run commands from the repository root so existing model and patch-tool imports
resolve correctly. Shared environment configuration remains at
`examples/kernel_agent/qwen38_b300_env.sh`.

| Tool | Purpose | Resources |
| --- | --- | --- |
| `check_qwen38_b300_runtime.py` | Validate model/data, dependency versions, patches and train/rollout NCCL library contents | Prepared runtime; may initialize CUDA through dependency checks |
| `check_qwen38_b300_attention.py` | Check FA4 backend; compare attention outputs/gradients | CUDA GPU for numerical checks |
| `check_qwen38_b300_flashqla.py` | Compare packed BF16 FlashQLA forward/backward against FLA | CUDA GPU |
| `check_qwen38_b300_nccl.py` | Validate communication for eight GPUs and supported subgroups | Eight GPUs, launched with `torchrun` |
| `benchmark_qwen38_training_hotpaths.py` | Measure replay-mask and TP log-prob chunking costs without loading the model | CUDA GPU; mode-specific inputs |

The coverage-ablation launchers invoke `scripts/b300/check_qwen38_b300_runtime.py`
automatically before submission. Keep this preflight check: moving the tools
does not relax version, checkpoint, or NCCL validation.

Environment paths can be supplied through the existing variables, including
`SLIME_MEGATRON_LM_PATH`, `B300_RUNTIME`, `SLIME_TRAIN_PACKAGES`,
`SLIME_ROLLOUT_PYTHONPATH`, `HF_MODEL_PATH`, `RL_DATA`, and checkpoint paths.
The pinned dependency versions and historical shape/topology assumptions are
intentional; review them before using another deployment. Relocation changes
only paths, not benchmark settings or validation criteria.

CPU-only tests (no Ray service or training submission):

```bash
python tests/test_qwen38_b300_runtime.py
```

For GPU diagnostics, first source the shared environment and select the proper
training package overlay as required by your deployment. Run only on explicitly
available GPUs, never concurrently with an active training/evaluation job.
For example, with all eight GPUs available:

```bash
torchrun --standalone --nproc-per-node=8 scripts/b300/check_qwen38_b300_nccl.py
```

Historical piecewise training, Ray/TensorBoard service launchers, and the script
that mutates installed dependencies are intentionally kept outside Git at
`local_artifacts/qwen38_b300_deployment/`, together with their launcher tests.
They are not required files in this versioned tool directory. Provision the
runtime separately; do not restart services or apply historical patches blindly.
