# Repository Guidelines

## Project Structure & Module Organization

`slime/` contains the core RL training framework, including rollout logic, Ray orchestration, backends, reward/filter hubs, and shared utilities. `slime_plugins/` contains optional integration modules for Megatron Bridge, model-specific support, and rollout buffer extensions. Entrypoints live at `train.py` and `train_async.py`; use the async entrypoint when training should overlap with rollout generation. `tests/` holds unit, integration, and CI-oriented tests, with reusable utility tests under `tests/utils/` and plugin contract tests under `tests/plugin_contracts/`. `examples/`, `scripts/`, and `docs/` provide runnable configurations, model launch scripts, and user/developer documentation. Static images are in `imgs/`.

## Build, Test, and Development Commands

- `pip install -e . --no-deps`: install slime from the repo checkout after preparing the project-specific CUDA/Megatron/SGLang environment.
- `bash build_conda.sh`: build the conda-based development environment when Docker is not suitable.
- `python train.py ...`: run the standard training loop using arguments from the selected example or script.
- `python train_async.py ...`: run the asynchronous training loop.
- `python -m pytest`: run the configured test suite under `tests/`.
- `python -m pytest tests/plugin_contracts`: run plugin API contract tests.
- `pre-commit run --all-files --show-diff-on-failure --color=always`: run formatting, import cleanup, and lint checks locally.

## Coding Style & Naming Conventions

Target Python 3.10+. Format Python with Black at 119 columns and sort imports with isort using the Black profile. Ruff enforces core `E`, `F`, `B`, and `UP` checks; do not bypass these without a clear reason. Prefer existing module patterns in `slime/` and `slime_plugins/` over new abstractions. Use `snake_case` for functions, modules, and test files; use `PascalCase` for classes.

## Testing Guidelines

Pytest is configured in `pyproject.toml` with strict markers and verbose output. Name Python tests `test_*.py` and mark broader scenarios with existing markers such as `unit`, `integration`, `system`, or `docs`. For plugin behavior, update or add tests in `tests/plugin_contracts/`. For training behavior, include a focused test or a reproducible command from `tests/` or `scripts/`.

## Commit & Pull Request Guidelines

Recent history uses short imperative subjects, often with scopes or prefixes such as `fix(qwen3_next): ...`, `[Fix] ...`, or `Add ...`. Keep commits focused and mention affected models, backends, or flags when useful. Pull requests should describe the bug or optimization, list verification commands, link related issues, and include benchmarks for performance changes. The project welcomes bug fixes and general large-scale RL optimizations with clear verification; avoid broad refactors or unverifiable features.

## Historical deployment records (September 2026)

The following records preserve prior deployment decisions. References to active jobs or authorization are historical; consult the current task handoff and latest user authorization before acting.

## Qwen3.8 B300 train and rollout containers

For `examples/kernel_agent/qwen38_b300_baseline_t1_refcov.sh` on 99.151, run training in `csl_slime_qwen38_b300_r9` and rollout in `csl_slime_qwen38_b300_r9_sgl0520`. Rollout requires features from newer SGLang (0.5.20), which is incompatible with the existing training container; that container retains SGLang 0.5.15.post1. Keep the roles in separate containers and use Ray placement resources `slime_actor` and `slime_rollout` to keep training on GPUs 0-3 and rollout on GPUs 4-7.

The two containers must use the same Ray and NCCL versions. Before starting or restarting this experiment, check Ray in both containers and compare the SHA-256 of the actual NCCL `libnccl.so.2` loaded through their respective Python environments; a matching reported NCCL version alone is insufficient. The current setup uses Ray 2.58.0 and a byte-identical loaded NCCL 2.29.7 library in both roles (Torch may report its build-time NCCL 2.28.9). Also confirm CUDA initializes in both containers and that Ray places each role in its assigned container before submitting the training job.

The Ray job driver and RolloutManager must run in the rollout container with SGLang 0.5.20, even when the Ray head runs in the training container. Pin the job entrypoint and RolloutManager to the rollout Ray node; keep Megatron training actors on the training node. The driver still needs `SLIME_TRAIN_PACKAGES` first in `PYTHONPATH` for Megatron and CUTLASS imports. SGLang 0.5.15 silently ignores the newer sampling-mask CLI flag, so a driver or manager on that version can send requests without `return_sampling_mask` and abort every top-p rollout. Before a full run, verify the driver and manager versions and run `scripts/check_sglang_top_p_replay.py --url` against a live 0.5.20 engine.

## Final KernelBench GEPA-V2 evaluation

The user authorized hourly progress checks and final evaluation of `experiments/qwen38_b300_baseline_t1_refcov` on KernelBench Levels 1, 2, and 3 after training succeeds. Evaluation must use the complete initial prompt messages from `KernelBench-TVMFFI-GEPA-V2/kernelbench_level{1,2,3}_val.parquet`; changing only the response template is insufficient. Verified copies and SHA-256 provenance are under `experiments/qwen38_b300_baseline_t1_refcov/eval_kernelbench_gepav2/data` and `provenance`. The files contain 100/100/50 initial user prompts, with a shared 10,006-character GEPA-V2 instruction prefix and no separate system-role message.

Use `scripts/qwen38_refcov_eval_artifacts.py check-training` to verify the current Ray job succeeded and the final iter79 DCP shards are complete. Export HF, restore only missing frozen visual weights from the original model using `complete-hf`, and verify all tensor shapes/dtypes before evaluation. The prepared launcher is `examples/kernel_agent/eval/qwen38_b300_baseline_t1_refcov_gepav2_all.sh`, using the rollout container's eight-GPU eval Ray cluster at `http://127.0.0.3:8271` only after training releases the GPUs. It evaluates Levels 1/2/3 sequentially, with 8 candidates per prompt (800/800/400 trajectories), T=0.7, top-p=0.7, context80k/response60k, and KernelGym warmup/perf/reference trials10/100/150; NCU and sanitizer are disabled. Each level must pass result-count and env_result completeness checks before its scores are accepted. Preserve failed logs and never resubmit an already running suite. Detailed instructions are in the evaluation provenance directory's `automation_handoff.md`.

## Local checkpoint staging and independent upload

`examples/kernel_agent/qwen38_b300_baseline_t1.sh` now defaults `TRAIN_SAVE_PATH` to `/data2/chenshuailin/slime_checkpoints/${EXP_ROOT##*/}/checkpoints`. Logs and TensorBoard stay in EXP_ROOT on NFS. Both train and rollout containers need a bind mount at `/data2/chenshuailin/slime_checkpoints`; the launcher refuses a missing mount or system-disk fallback. Both containers were recreated from preserved environment snapshots on 2026-09-24 and now have this bind mount. Prior containers remain stopped with suffix _pre_nvme_20260924T083535.

The uploader lives OUTSIDE this repository at `/nfs/hw-data/ms/FM/chenshuailin/projects/checkpoint_transfer/`. See its README and `run_qwen38_transfer.sh`. It copies only committed checkpoints and their matching rollout dataset cursor, verifies SHA-256, atomically publishes to the original EXP_ROOT/checkpoints, and only then advances the archive tracker. The user approved keeping only the newest one verified local checkpoint while retaining all NFS history. Never prune unverified data. Local and archive latest paths are both valid explicit resume inputs; resuming behind an existing output checkpoint is rejected.

The user subsequently authorized training launch on 2026-09-24. The standalone uploader runs in csl_checkpoint_upload_qwen38_b300_t1 with restart=unless-stopped, no GPU, and no network. Do not also start a duplicate systemd uploader. This launch uses the original submission ID qwen38_b300_baseline_t1 after clearing its failed job record; RECOMPUTE_NUM_LAYERS=32 is an explicit launch override to reduce activation memory after the prior OOM.

## TensorBoard service

TensorBoard port 6006 runs independently in Docker container csl_slime_tensorboard_6006 with restart=unless-stopped and a read-only mount of this repository's experiments directory. Its --logdir is REPO_ROOT/experiments. Do not launch another listener on this port or couple it to training container restarts. Check with docker logs or docker inspect. The service was restored on 2026-09-24 after training-container recreation stopped the former TensorBoard process.

On 2026-09-24, the launcher default RECOMPUTE_NUM_LAYERS was changed to 24 for future launches only. The active qwen38_b300_baseline_t1 job was launched with an explicit 32-layer override and must remain unchanged. The 24-layer setting is a conservative tuning candidate, not yet validated by a training run or complete peak-memory measurements; the earlier 8-layer run OOMed.

## Time coverage final evaluation (authorized 2026-09-24)
The user explicitly requested KernelBench Levels 1, 2, and 3 for completed
experiments/qwen38_b300_baseline_t1, final iter79. The dedicated suite is
experiments/qwen38_b300_baseline_t1/eval_kernelbench_gepav2. Use
examples/kernel_agent/eval/qwen38_b300_baseline_t1_gepav2_all.sh and
scripts/qwen38_timecov_eval_artifacts.py for this experiment. Ray submission IDs
are qwen38-timecov-final-kb-l{1,2,3}-gepav2 on dashboard127.0.0.3:8271.
Keep the prior refcov results intact; check suite/status.txt and active jobs
before any submission. The suite runs levels sequentially and validates
800/800/400 results with zero missing env_result. GEPA-V2 source prompt messages
and previously approved evaluation settings are unchanged. Latest training and
HF export checks are under this suite's provenance directory.
