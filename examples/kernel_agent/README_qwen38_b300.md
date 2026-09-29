# Qwen3.8 piecewise r9 on one B300 node

> Historical runtime and tuning notes: the configurations and launch commands below
> describe the September 20–21 piecewise r9 setup, including failed attempts. They
> are retained for reproducibility, not as instructions to restart a live service.
> Later baseline/coverage experiments use their dedicated `qwen38_b300_baseline_t1*.sh`
> launchers and `eval/*_gepav2_all.sh` suites. In particular, Timecov025 PRS uses an
> absolute coverage weight of 0.25 with the existing reference-coverage PRS and
> common validity gates. Current process state and execution authorization remain
> in the task handoff and experiment provenance. Do not infer permission for a new
> training run, export, or evaluation from this historical document.


This configuration uses the dedicated `csl_slime_qwen38_b300_r9` container,
created from `slimerl/slime:nightly-dev-20260827a-cu130-b300`. It does not change
the existing `slime_lhb` container or stop other Ray clusters.

## Approved configuration

| Setting | Value |
| --- | --- |
| Training GPUs | 0–3, TP4 / CP1 / PP1 / DP1 |
| Rollout GPUs | 4–7, four TP1 engines |
| Prompts / samples | 16 prompts × 16 samples |
| Global batch | 128; two optimizer steps per rollout |
| Training microbatch token budget | 16,384; dynamic packing, not a sequence-length cap |
| Training duration | 80 rollouts, starting from the original HF BF16 model |
| Optimizer | Adam, constant LR 1e-6 |
| Loss / reward | TRLOO, DPPO binary-TV, piecewise reward |
| Model / inference | BF16, FP32 LM head, NEXTN3, one trained MTP layer |
| Length limits | 120,000 context / 32,000 response tokens |
| Recompute | 8 transformer layers, full/block; `RECOMPUTE_NUM_LAYERS=0` disables it for explicit comparisons |
| Attention | Training `flash` via FlashAttention 4; rollout `trtllm_mha` |
| Rollout requests / graph batch | 64 / 64 per engine |
| Rollout memory | Static fraction 0.85; explicit Mamba state pool of 320 slots |
| Policy log-prob chunks | 2048 tokens; `LOG_PROBS_CHUNK_SIZE` can override for comparisons |
| Saving | Every 10 rollouts; all Megatron checkpoints retained, no HF export |
| Logging | TensorBoard, no W&B run or credentials |

The user-approved CP1 revision uses all four training GPUs as TP4/CP1/PP1/DP1.
FlashAttention 4 `4.0.0b15`, already included in the container, is selected via
Megatron's `--attention-backend flash`. No TE, cuDNN or SGLang upgrade is
required. FA4 cannot use CP2 through the current TE integration. FlashQLA
continues to handle GDN linear attention separately from FA4's dense attention.

The ordinary non-colocated Ray placement group allocates eight GPUs and sorts
their physical IDs, giving the first four to training and the last four to
rollout. The separate cluster uses GCS port 6389 and dashboard port 8269.

`SAVE_INTERVAL_STEPS=20` is converted to rollout units from the configured
sample count and global batch. Invalid divisibility is rejected before a job
is submitted. Checkpoints retain optimizer state; the launcher deliberately
does not inherit WarmUp's `--no-save-optim` option.

The legacy-named `--wandb-centralized` and `--wandb-always-use-train-step`
options also control TensorBoard aggregation and step numbering. They do not
enable W&B; `--use-wandb` is absent.

With four engines the per-engine request limit remains 64, giving a client
concurrency of 256 and 16 simultaneous prompt groups. The earlier two-TP2
configuration had client concurrency 128 and eight groups. This does not
change the accepted rollout batch of 16 prompts × 16 samples.

## Inputs and outputs

- Model: `/nfs/hw-data/ms/FM/checkpoints/Qwen-Zoo/Qwen3.8-27B`.
- Data: `/nfs/hw-data/ms/FM/lihongbin/dataset/CUDA_RL/cuda_rl/prompt_tvm_GEPA4o_v2/torch_ops_difficulty_lt18.parquet`.
- KernelGym: `http://192.168.112.55:20111`, on eight A800 GPUs.
- Default experiment directory: `local_artifacts/qwen38_b300_r9` in this checkout.
- Output subdirectories: `checkpoints`, `tensorboard`, `logs`, `rollout`, `provenance`.
- Isolated training packages and caches: `local_artifacts/qwen38_b300_r9/runtime`.

Ephemeral sockets and compiler scratch space use `/tmp/slime_qwen38_b300`
inside the container, not the long NFS path. Unix socket paths are limited to
107 characters. Persistent logs, caches and checkpoints remain in the checkout.

The approved data contains 27,525 records. Its SHA256 is
`47c5ad75d127ee3647d99b485202bf7624a8d366c24a184d28f4dd1215357217`.
This intentionally differs from the original r9 handoff file and is checked
explicitly rather than bypassing checksum validation.

The repository is mounted read/write in the container at the same absolute
path as the host. The model and data directories are mounted read-only.

## Runtime preparation

Keep SGLang at **0.5.15.post1** and rollout TileLang at **0.1.11**. Training
FlashQLA requires a separate TileLang 0.1.9 installation; do not downgrade
the container-wide TileLang installation.

Run inside the dedicated container, from this checkout:

```bash
source examples/kernel_agent/qwen38_b300_env.sh
python -m pip config --user set global.index-url http://192.168.99.216:8081/repository/python/simple
python -m pip config --user set global.trusted-host 192.168.99.216
python -m pip install -e . --no-deps --no-build-isolation
python -m pip install --target "$SLIME_TRAIN_PACKAGES" --no-deps --no-build-isolation \
  flash-qla==0.1.2 tilelang==0.1.9 apache-tvm-ffi==0.1.9
bash scripts/prepare_qwen38_b300_runtime.sh
```

The internal mirror supplies `flash-qla` (not `flashqla`). Version 0.1.2
requires exactly TileLang 0.1.9 and TVM-FFI 0.1.9. All three are installed
only in the training overlay; rollout retains its original TileLang 0.1.11
and TVM-FFI 0.1.11. The same mirror is configured in the host user's
`~/.config/pip/pip.conf` without changing other users or containers.

FlashQLA 0.1.2 needs an SM103 dispatch patch in two files to select its
Blackwell kernels on B300. `patch_flashqla_b300.py` backs up and checks those
changes; it does not force SM100 compilation. TileLang still compiles for
the native `sm_103a` target. Training alone receives the existing Z3 shared
library directory in `LD_LIBRARY_PATH`.

The preparation script applies the version-specific top-p replay, one-sort
reuse, and FP32 head-cache patches, installs hidden-state MTP detach, and
verifies output-weight MTP detach. It is safe to rerun. The 0.5.15 cache patch
targets `ModelRunner` weight updates, not the 0.5.16 `WeightUpdater` layout.
GPU validation is separate from installing missing FlashQLA dependencies.

## Validate before training

```bash
source examples/kernel_agent/qwen38_b300_env.sh
torchrun --standalone --nproc-per-node=8 scripts/check_qwen38_b300_nccl.py
python scripts/check_sglang_top_p_replay.py --check-sort-reuse --device cuda
python scripts/check_sglang_fp32_lm_head_cache.py --device cuda
LD_LIBRARY_PATH="$SLIME_TRAIN_LD_LIBRARY_PATH" PYTHONPATH="$SLIME_TRAIN_PACKAGES:$PYTHONPATH" \
  python scripts/check_qwen38_b300_flashqla.py
LD_LIBRARY_PATH="$SLIME_TRAIN_LD_LIBRARY_PATH" PYTHONPATH="$SLIME_TRAIN_PACKAGES:$PYTHONPATH" \
  python scripts/check_qwen38_b300_attention.py
CONFIG_DRY_RUN=1 bash examples/kernel_agent/run_qwen38_b300_piecewise.sh
PREFLIGHT_ONLY=1 bash examples/kernel_agent/run_qwen38_b300_piecewise.sh
python -m pytest tests/test_qwen38_b300_launcher.py
```

The NCCL check covers FP32/BF16 all-reduce, all-gather, reduce-scatter,
all-to-all and paired send/receive on the TP4 training and TP2 rollout groups,
as well as the previous TP2/CP2 groups. The node's
active interface is `ens16f1np1`, not the old clusters' `bond0` or `front1`.
No blanket P2P, NVLS or IB disable flag is applied. A successful collective
check does not replace validation of real model forward/backward and weight
synchronization, especially for the 120k length limit.

## Start and resume

Only proceed after full runtime, model execution, and checkpoint
save/restore validation:

```bash
bash examples/kernel_agent/start_qwen38_b300_ray.sh
bash examples/kernel_agent/run_qwen38_b300_piecewise.sh
```

TensorBoard runs independently of Ray jobs in this dedicated container:

```bash
docker exec -d csl_slime_qwen38_b300_r9 \
  bash examples/kernel_agent/start_qwen38_b300_tensorboard.sh
```

The URL is **http://192.168.99.151:6006/**. The `training` and `validation`
run groups are separate. Its PID and output are in `logs/tensorboard.pid`
and `logs/tensorboard_server.log`. The server survives client disconnects
and training-job exits, but must be restarted after the container restarts.
Do not run the command again while port 6006 is already occupied.

On 2026-09-20, the old `validation/` experiment outputs were removed at the
user's request, including their checkpoints, replay data, and TensorBoard
events. Current formal outputs, runtime dependencies, and diagnostic logs
are retained. Historical validation paths below must be regenerated before
rerunning those validation commands; only the `training` run remains in
TensorBoard.

To benchmark training without another 305-GiB checkpoint write, use an
explicit saved-rollout replay and a separate experiment directory:

```bash
EXP_ROOT="$PWD/local_artifacts/qwen38_b300_r9/validation/cp1_rc0_benchmark" \
  LOAD_DEBUG_ROLLOUT_DATA="$PWD/local_artifacts/qwen38_b300_r9/validation/save_resume/rollout/rollout_0.pt" \
  NUM_ROLLOUT=1 VALIDATION_NO_SAVE=1 RECOMPUTE_NUM_LAYERS=0 \
  RAY_SUBMISSION_ID=qwen38-b300-r9-cp1-rc0-benchmark \
  bash examples/kernel_agent/run_qwen38_b300_piecewise.sh
```

`VALIDATION_ROLLOUT_ONLY=1` instead tests live rollout without allocating
training actors. These validation switches are off by default and cannot
combine rollout-only mode with saved-data replay. Formal training always
starts from the HF weights unless `TRAIN_LOAD_PATH` is explicitly supplied.

For a separate one-rollout save/restore validation, run inside the container
after starting Ray:

```bash
source examples/kernel_agent/qwen38_b300_env.sh
VALIDATION_ROOT="$SLIME_REPO/local_artifacts/qwen38_b300_r9/validation/save_resume"
EXP_ROOT="$VALIDATION_ROOT" NUM_ROLLOUT=1 SAVE_INTERVAL_STEPS=2 \
  RAY_SUBMISSION_ID=qwen38-b300-r9-validate-save \
  bash examples/kernel_agent/run_qwen38_b300_piecewise.sh
EXP_ROOT="$VALIDATION_ROOT" TRAIN_LOAD_PATH="$VALIDATION_ROOT/checkpoints" \
  NUM_ROLLOUT=1 SAVE_INTERVAL_STEPS=2 \
  RAY_SUBMISSION_ID=qwen38-b300-r9-validate-resume \
  bash examples/kernel_agent/run_qwen38_b300_piecewise.sh
```

Use unused submission IDs and a fresh validation directory when repeating
the save test. The first command trains rollout 0 and saves its two optimizer
updates as Megatron iteration 0. The second restores that checkpoint and
publishes its weights to rollout engines, then exits without generating a
new rollout because the restored start rollout is 1. These temporary
overrides do not change the formal 80-rollout run or its save interval.

To reuse a previously evaluated rollout for a training-only save test, set
`LOAD_DEBUG_ROLLOUT_DATA` to its `rollout_0.pt` file and use a fresh
`EXP_ROOT`. This skips SGLang allocation and KernelGym calls only for that
validation job. Leave it unset for full restore/weight-sync validation and
formal training. The current CP1 save test uses
`validation/cp1_save_resume` and replays `validation/save_resume/rollout/rollout_0.pt`.

For an intentional resume, set `TRAIN_LOAD_PATH` to the Megatron checkpoint
directory and use a new Ray submission ID. An existing checkpoint output
directory is rejected unless that exact directory is explicitly selected as
the load path. No checkpoint retention/deletion service is installed.

The checkout's original `.git` file points to an unavailable Windows
worktree. It has not been rewritten or reinitialized. The model argument
script has been normalized to LF so the new Linux launcher can source it.

## Validation status on 2026-09-20

- Eight-GPU NCCL 2.28.9 collective and TP2/CP2 P2P checks passed.
- Top-p replay numerical checks and one-sort checks passed on CPU and CUDA.
- FP32 head-cache update and shared-storage checks passed on CPU and CUDA,
  including CUDA graph replay after weight replacement.
- MTP detach verification and repeated patch preparation passed.
- Distributed GDN, training FP32 head, launcher and patch regression tests:
  94 passed in the isolated training runtime.
- Full launch preflight and complete Megatron/SGLang argument validation pass.
- The internal mirror resolves the earlier public-DNS download failure.
  FlashQLA 0.1.2, TileLang 0.1.9 and TVM-FFI 0.1.9 are installed separately
  from the unchanged rollout dependencies.
- Native SM103 FlashQLA forward/backward agrees with FLA on packed lengths
  `[256, 256]` and `[4096, 512]`; all relative L2 errors are below 0.6%.
  The comparison applies the same GVA head expansion as the training model.
- Both TP2 rollout engines initialize successfully, including CUDA graphs
  through batch 64. Initial Megatron-to-SGLang weight synchronization passed
  on the approved 4+4 GPU topology. Rollout 0 completed 16 accepted groups
  with KernelGym evaluation in approximately 900 seconds.
- Full-model validation uses `validation/save_resume` below the experiment
  directory, with one rollout and a validation-only save interval of one.
  Formal 80-rollout training must wait for complete save/restore validation.
- Validation job `qwen38-b300-r9-validate-save-r2` failed before its first
  optimizer update: TE reported no available dot-product attention backend.
  The saved `rollout/rollout_0.pt` can be reused for training-only diagnosis;
  no checkpoint was produced and formal training has not started.
- TE defaults to system cuDNN 9.13 while PyTorch uses the installed 9.19.
  An isolated diagnostic with both `CUDNN_HOME` and `LD_LIBRARY_PATH` pointing
  to 9.19 makes both libraries report 9.19, but does not fix backend selection.
  TE 2.16.1 on this B300 selects fused attention for head dimensions 64/128
  and for dimension-256 inference, but not dimension-256 training, even
  without context parallelism. No backend or permanent cuDNN setting has
  been changed to bypass this restriction. See `te_backend_support_matrix.log`
  and `te_backend_cudnn_home_diagnostic.log` in the logs directory.
- The subsequent CP1 revision passes FA4 output and Q/K/V gradient comparison
  against an FP32 reference at packed lengths `[256, 128]` and `[2048, 512]`;
  all relative L2 errors are below 0.27%. Forward/backward also produces finite
  outputs and gradients at `[32768, 512]` and `[120000]`. This long-context
  check covers attention in isolation, not the full model's peak memory.
  Evidence: `attention_fa4_cp1.log`. CP1 launch preflight and 18 launcher/patch
  regression tests pass.
- CP1 replay job `qwen38-b300-r9-cp1-validate-save` completed both optimizer
  updates with finite losses and gradient norms 0.2108 / 0.2017. TensorBoard
  contains both training steps. The first run took approximately 24.7 minutes
  including kernel compilation; this is not a steady-state throughput result.
- Its complete Megatron checkpoint has eight data shards totaling 305.37 GiB,
  2,895 metadata entries including 1,584 optimizer entries, and scheduler
  state covering 256 samples. Shard byte ranges and the final iteration-0
  tracker were verified. The first NFS save took approximately 485 seconds.
  Evidence: `checkpoint_cp1_validation.log`.
- `qwen38-b300-r9-cp1-validate-resume` succeeded: it restored iteration 0,
  including optimizer state, synchronized the weights from TP4 training to
  the original two TP2 rollout engines, and exited without a training step.
  Low GPU utilization during this load/sync-only run is not training
  utilization. Evidence: `validation_cp1_resume_driver.log`.

The no-recompute benchmark uses the same 256 cached samples, totaling
3,151,356 tokens (median 11,161.5; maximum 30,720). Passing this replay is
not a full-model memory guarantee for the configured 120k context limit.
The 16,384-token dynamic batching target is not a hard sequence-length cap:
an individual longer sample remains a full microbatch. GPU utilization
and memory are sampled every two seconds in `logs/gpu_efficiency.csv`;
sampled maxima are lower bounds, not allocator high-water marks.

## Efficiency findings and remaining gates

The following benchmarks preceded the formal launch documented below.
Training stays on GPUs 0–3;
the four TP1 engines stay on GPUs 4–7. Four additional SGLang detokenizer
processes are CPU workers, not four additional GPU engines. The physical
CUDA process mapping is recorded in `logs/gpu_placement_tp1_validation.log`.

- `qwen38-b300-r9-cp1-rc0-benchmark` succeeded without recompute or checkpoint
  I/O. Its two optimizer updates took 653.67 seconds, approximately 4821
  training tokens/second. Gradient norms were 0.21083 and 0.20166. The earlier
  recompute-40 test took 1482 seconds, but also incurred more first-run
  compilation; the difference is not a controlled recompute-only speedup.
- Two-second GPU sampling during the no-recompute training interval measured
  mean GPU utilization of 62–72% across training ranks, with individual
  samples reaching 100%. Loading and compilation must not be interpreted as
  steady training utilization.
- Sampled GPU memory reached **255.97 GiB per training GPU**, versus about
  267.68 GiB usable capacity. This includes allocator reservations; it is not
  a measurement of peak live tensor allocation. A low reading earlier in
  the rollout does not demonstrate safe headroom for larger microbatches,
  lower TP, or all sequences allowed by the 120k/32k limits. Measure both
  `max_memory_allocated` and `max_memory_reserved` before approving those
  changes. No full-model 120k no-recompute guarantee is claimed.
- Offline packing of this same replay gives 256 total microbatches at an
  8192-token target, 213 at 16384, 149 at 24576, and 106 at 32768 for TP4/CP1.
  These are schedule counts, not throughput measurements. Global batch 128
  and two optimizer updates remain unchanged. Larger packing targets need
  a GPU memory/performance validation.
- TP2/CP1/DP2 is a candidate to reduce tensor-parallel communication and
  enlarge local GEMMs, at higher per-GPU model/activation memory. TP1/CP1/DP4
  has still greater memory pressure. Neither has been validated for this
  no-recompute configuration. TP2/CP2 and TP1/CP4 cannot use the currently
  selected TE/FA4 path: TE explicitly disables FA4 for CP>1, while fused
  dimension-256 training is unavailable in this runtime.
- The TP1 rollout benchmark requests 64 concurrent requests per engine but
  SGLang reports `effective_max_running_requests_per_dp=52`. The automatic
  Mamba state pool has 263 slots; this NEXTN/overlap configuration budgets
  five slots per request, so `263 // 5 = 52`. This is a real admission cap,
  not merely a GPU-memory display issue.
- `qwen38-b300-r9-tp1-rollout-benchmark` succeeded with four physical rollout
  GPUs and 16 accepted groups in 668.7 seconds. Its accepted responses total
  2,150,825 tokens versus 2,472,140 in the old 899.9-second TP2 run, so raw
  rollout times are not a fixed-output throughput comparison.
- During this rollout, full KV token usage peaked around
  31%, Mamba usage around 66%, and no memory-driven request retraction/OOM
  was reported. Merely increasing ordinary KV capacity is not the first
  optimization. A candidate is `--sglang-max-mamba-cache-size 320` while
  retaining `--sglang-mem-fraction-static 0.75`: it should permit the intended
  64 requests by reallocating cache budget toward Mamba states. This needs
  validation of effective concurrency, actual decode throughput, long
  requests, and weight-update peak memory. This was not applied during the
  baseline; see the subsequent approved 0.80/320 configuration below.
- Increasing `mem-fraction-static` is another valid way to grow the automatic
  Mamba pool: the current ratio allocates `0.9 / 1.9` of the post-weight
  cache budget to Mamba main and speculative intermediate states. Increasing
  the fraction also grows ordinary KV capacity and reduces runtime headroom,
  unlike explicitly reallocating the existing budget toward Mamba. Neither
  SGLang setting has been changed without a confirmed target value.
- The old TP2 rollout's accepted samples spent a mean 171.17 seconds in model
  generation and 4.90 seconds in KernelGym evaluation (2.8% of summed request
  time). This is not a wall-clock critical-path fraction and excludes dropped
  groups. The earlier 305.37-GiB checkpoint write took about 485 seconds;
  at one save per ten rollouts this is about 48.5 seconds per rollout averaged
  over time, before accounting for any overlap.

TensorBoard at `http://192.168.99.151:6006/` remains running independently of
the validation jobs. All 105 focused GDN, FP32-head, launcher, and patch tests
pass; see `logs/test_training_rc0_tp1.log`.

The user subsequently approved increasing the training microbatch token
budget from 8192 to 16384. TP4/CP1, global batch 128, two optimizer updates,
no recompute, response/context limits, and SGLang's 0.75 memory fraction are
unchanged. The new comparison uses the same original 256-sample replay,
not the shorter TP1 rollout. With `--log-device-memory-used`, the actor now
records CUDA allocator peak/current allocated and reserved GiB, allocation
retry deltas, and OOM deltas around the full training phase. All ranks log
their own measurements; the primary rank also publishes them under
`memory/actor_train_*` in TensorBoard. Allocator statistics do not include
all non-PyTorch allocations such as NCCL buffers.

The 16384-token replay `qwen38-b300-r9-cp1-rc0-mb16k-benchmark` succeeded:

| Metric | 8192-token target | 16384-token target |
| --- | --- | --- |
| Samples / optimizer updates | 256 / 2 | 256 / 2 |
| Microbatches | 256 | 213 |
| Actor training time | 653.67 s | 678.83 s |
| Training tokens/s | 4820.99 | 4642.31 |
| Gradient norms | 0.21083 / 0.20166 | 0.18433 / 0.17653 |
| Peak live CUDA allocation | Not instrumented | 250.03 GiB |
| Peak CUDA allocator reservation | Not instrumented | 250.77 GiB |
| End-of-phase live CUDA allocation | Not instrumented | 103.12 GiB |
| Allocation retries / OOMs | Not instrumented | 0 / 0 |

This one-rollout comparison does not demonstrate a speedup (elapsed time
increased about 3.8%); process-local JIT compilation and changed packed
shapes are included, so it is not a steady-state benchmark. The configured
16384 target remains as explicitly requested. The measured live allocation
confirms the earlier high GPU usage was not merely unused allocator cache.
Longer samples allowed by the unchanged length limits remain an OOM risk;
the formal trial was subsequently launched with user approval.

Packing also is not guaranteed to preserve identical optimizer updates:
the existing Megatron MTP path computes a token-mean auxiliary loss inside
each microbatch, then scales it by `1 / num_microbatches`. Combining samples
with different lengths changes their relative MTP weighting. The main policy
loss retains its global-batch normalization. MTP's coefficient remains 0.2
and its implementation has not been changed; the observed gradient-norm
differences must not be characterized as purely roundoff without further
isolation. Changing MTP normalization would require a separate decision.

Evidence: `logs/validation_cp1_rc0_mb16k_driver.log`,
`logs/mb16k_comparison.json`, and
`validation/cp1_rc0_mb16k_benchmark/tensorboard`.

### Approved 0.80 / 320 rollout configuration

A standalone TP1 engine on physical GPU 4 confirmed that increasing only
`mem-fraction-static` to 0.80 yields 287 Mamba slots and an effective request
limit of 57, not the configured 64. Its full-attention KV pool held 1,360,512
tokens. Evidence: `logs/sglang_mem080_capacity.json` and
`logs/sglang_mem080_probe.log`. The probe is stopped before the full 4+4 job.

The user then authorized reallocating self-attention cache budget to Mamba
and starting training. That formal launch used the static fraction 0.80
and explicitly sets `--sglang-max-mamba-cache-size 320`. The existing SGLang
allocator reserves the main Mamba and speculative intermediate states first,
then uses the remaining static budget for full-attention KV. Thus the change
does not independently add another memory pool on top of the static budget.
The target is 64 effective requests per engine (`320 // 5`), with the same
NEXTN3, FP32 head, 120k/32k limits, four TP1 engines, and four TP4/CP1 training
GPUs. All four live engines report 320 Mamba slots, effective concurrency 64,
and decode CUDA graph maximum batch 64. Their full-attention KV capacity is
1,217,152 tokens each, down from the standalone 0.80 automatic-allocation
probe's 1,360,512. The initial TP4-to-four-TP1 weight update completed in 9.2
seconds without an observed NCCL error. Evidence: `logs/formal_engine_capacity.json`.

### Historical no-restart memory change

The initial change to `--sglang-mem-fraction-static 0.85` only changed the
next-launch configuration. The `mem80-m320-r9` job continued at 0.80 without
being restarted. Static pools are allocated at engine startup, so those
original measurements do not validate 0.85. The user subsequently authorized
stopping and restarting training after the hot-path optimizations below.

The explicit Mamba pool remains 320 slots and the effective concurrency target
remains 64. Raising the static fraction primarily adds full-attention KV
capacity, not additional Mamba slots. Never launch a second job over the same
GPU allocation to apply a configuration change.

### Original formal job (failed)

- Ray submission ID: `qwen38-piecewise-b300-4train4rollout-tp4-cp1-fa4-rc0-mb16k-rtp1-mem80-m320-r9`.
- Submitted on 2026-09-20 at 19:12:10 China time; rollout 0 began at 19:17:33.
- Starts from the original HF BF16 model, not a validation checkpoint or replay.
- Runs 80 rollouts, with training TP4/CP1 on GPUs 0–3 and four TP1 engines on GPUs 4–7.
- Keeps the approved 16384 training microbatch target, no recompute, global batch 128,
  LR 1e-6, piecewise/TRLOO/DPPO, NEXTN3, and original length limits.
- Retains all Megatron checkpoints including optimizer state every 10 rollouts;
  no HF checkpoint export and no W&B.
- Driver log: `logs/formal_mem080_m320_driver.log`.
- TensorBoard: **http://192.168.99.151:6006/**, under the `training` run group.
- Launch configuration audit: `logs/formal_configuration_verified.json`.

At 19:38 China time, the first formal rollout had completed both optimizer
updates with finite losses and gradient norms 0.17368 / 0.17361. Generation
took 640.9 seconds and actor training took 601.48 seconds. The subsequent
weight update completed in 4.4 seconds, and training on rollout 1 began while
the next rollout was generated asynchronously. No NCCL error, OOM, or CUDA
allocator retry was observed through this first full cycle. The job remains
running at that observation; this was startup validation, not completion
of all 80 rollouts. It later failed at 20:01 China time during training of
rollout 2, with a NaN gradient reported for bucket 0. Five optimizer updates
had completed and no scheduled checkpoint existed, so this job cannot be
resumed at its last update. Its first saved rollout is retained for comparisons.

The first formal rollout's maximum sample length was only 21,756 tokens;
its measured peak live allocation was 208.42 GiB and peak reservation was
209.02 GiB per primary training rank. These lower peaks do not supersede
the 250-GiB result on the earlier longer-sample replay. Evidence:
`logs/formal_first_rollout_verified.json` and `logs/formal_rollout0_lengths.json`.

The driver is launched with detached `docker exec -d`, and the Ray submission
is independent of the client terminal. To inspect or stop this job specifically,
run inside the dedicated container:

```bash
ray job status --address http://192.168.99.151:8269 \
  qwen38-piecewise-b300-4train4rollout-tp4-cp1-fa4-rc0-mb16k-rtp1-mem80-m320-r9
ray job stop --address http://192.168.99.151:8269 \
  qwen38-piecewise-b300-4train4rollout-tp4-cp1-fa4-rc0-mb16k-rtp1-mem80-m320-r9
```

Evidence is recorded under `local_artifacts/qwen38_b300_r9/logs` and
`local_artifacts/qwen38_b300_r9/provenance`.

### September 20 hot-path optimization and restart (failed)

The optimized submission ID was
`qwen38-piecewise-b300-4train4rollout-tp4-cp1-fa4-rc0-mb16k-rtp1-mem85-m320-opt1-r9`.
It restarts from the original HF model, not from a checkpoint of the failed
job or from a replay benchmark's parameters. Training remains TP4/CP1 on
GPUs 0–3, with four TP1 engines on GPUs 4–7 and all original RL hyperparameters.

- Top-p replay constructs masks with batched index transfers and scatter,
  preserving the exact rollout support, missing metadata, and TP/CP mapping.
- Packed sequence boundaries and maximum lengths are computed on the CPU.
  Versioned metadata is reused by GDN validation and FlashQLA's internal
  CP planner; in-place boundary mutations invalidate the metadata.
- Target-token restoration no longer uses a dynamic GPU `nonzero`.
  Policy log-prob chunks default to 2048. Training actors check free device
  memory every 64 eligible collectives or 0.25 seconds, whichever comes first;
  low-memory observations restore per-call checks and exception diagnostics
  remain enabled. Other launchers retain the original per-call default.
- Megatron MTP clamps an empty supervision denominator to one, producing
  zero loss and gradient for an all-zero mask. Nonempty normalization is
  unchanged. This fixes a reproducible zero-token division path; the failed
  rollout-2 data was not saved, so it does not prove the original failure's
  sole cause. The new launcher saves rollout debug data through ID 2.

Runtime preparation and preflight now apply/check the FlashQLA CPU metadata
and MTP empty-mask patches without changing package versions. The old source
baseline is under `optimization/baseline`; benchmark logs and JSON results
are under `logs/benchmark_top_p_mask.*` and `logs/benchmark_loss_chunks.*`.
The mask benchmark on a real 19,135-token response reduced TP-rank-0 mask
construction from 828 ms to 0.76 ms with bitwise-identical masks. The TP4
synthetic 16k-token loss forward/backward measured 41.3 ms at chunk 512 and
31.4 ms at chunk 2048, with matching log-probs and gradients. These are
component measurements, not an end-to-end training speedup claim.

The complete optimized replay (`qwen38-b300-r9-hotpaths-opt1-replay`) succeeded
on the original formal rollout's 256 samples and 208 microbatches. Actor
training took 385.15 s, versus the historical 601.48 s (36.0% shorter).
Both updates matched the logged policy loss, MTP loss, PPO KL, and clip fraction
exactly. The first gradient norm matched exactly; the second differed by
1.69e-6 absolute / 9.73e-6 relative. Peak live CUDA allocation was 208.85 GiB,
versus 208.42 GiB previously, with no allocator retries or OOMs. Evidence:
`logs/optimization_replay_verified.json`. The historical baseline overlapped
rollout, and JIT/cache states can differ, so this is not a controlled steady-state
end-to-end throughput comparison. Generation can still limit total RL throughput.

The failed original job's `tensorboard`, `rollout`, and `provenance` directories
are archived under `optimization/failed_mem080/`. Current formal outputs reuse
the normal experiment-root paths without mixing TensorBoard steps. To reproduce
the comparison after this archive move, point `LOAD_DEBUG_ROLLOUT_DATA` at
`optimization/failed_mem080/rollout/rollout_0.pt` under the experiment root.
The new detached launch writes `logs/formal_mem085_opt1_driver.log` and uses
the same persistent TensorBoard URL, **http://192.168.99.151:6006/**.

At 20:52 China time on 2026-09-20, the optimized formal job was RUNNING,
with training actors on physical GPUs 0–3 and TP1 schedulers on GPUs 4–7.
All four live server configurations reported static fraction 0.85 and
320 Mamba slots. Scheduler startup reported 64 effective requests and
1,435,840 full-attention KV tokens per engine. Initial weight synchronization
completed in 7.5 s, and generation plus KernelGym evaluation were active.
This observation was still in rollout 0 generation, before the first formal
optimizer update; the two verified updates above belong to the replay test.
The live `/server_info` internal-state queries timed out after two seconds,
so capacity was cross-checked against startup logs rather than inferred from
empty API internals. Evidence: `logs/formal_opt1_engine_verified.json`.
TensorBoard's run list can remain empty until the first metrics are emitted.

Run CPU-stub tests separately from tests importing the real Megatron model
provider; combining those test groups can replace `megatron.core` with an
incompatible stub during collection. Dedicated groups validate top-p masks,
CPU metadata, memory polling, MTP empty masks, CUDA log-prob gradients, TP2
collectives, GDN layouts, and packed-sequence padding.

### September 21: long-sample top-p mask OOM

The `opt1` formal run stopped at 21:48 China time on September 20, during
rollout 4 after eight completed optimizer updates. A full-sequence boolean
top-p mask requested 2.01 GiB with only 1.85 GiB free on trainer GPU 0.
The saved batch contains 256 samples, with a maximum length of 35,079 tokens;
114 exceed the 16,384-token packing budget. That budget does not truncate
individual long samples: they run in singleton microbatches.

The loss now builds top-p masks lazily for each log-prob chunk and releases
each mask before constructing the next. The saved rollout support, target
restoration, unmasked entropy, and TP/CP row mapping are unchanged. A
2048-row mask at TP4 occupies at most 121.25 MiB instead of about 2 GiB for
the failing long sequence. This removes the full-mask allocation, but does
not by itself bound transformer activations or saved softmax tensors.

CPU tests compare dense and chunked masks across CP1, zigzag CP, allgather
CP, TP vocabulary partitions, empty nuclei, and missing replay rows.
Additional tests verify identical outputs and gradients, mask lifetime,
and CUDA/NCCL TP2 parity. Full replay evidence is kept under
`local_artifacts/qwen38_b300_r9/optimization/oom_fix/` separately from the
formal TensorBoard and checkpoint directories.

The first full replay with recompute disabled completed one optimizer update,
but still OOMed during the second update (progress 144/229 microbatches).
The remaining failure requested a 472 MiB masked-logit scratch tensor with
only 159 MiB free. This is why passing mask tests or the first optimizer
update is not sufficient: the complete batch must fit after optimizer state
has been initialized. The follow-up replay uses eight block-recomputed
transformer layers, retaining the 16,384-token packing budget, FP32 head,
TP4/CP1 topology, and original loss and rollout settings.

The eight-layer replay `qwen38-b300-r9-chunk-mask-replay-rc8` succeeded on
all 256 samples and 229 microbatches, completing both optimizer updates.
Actor training took 588.80 seconds, including startup compilation. The
maximum logged CUDA allocation was 253.95 GiB (254.93 GiB reserved), with
zero allocator retries and zero OOMs. The first update's policy loss, PPO
KL, clipping fraction, MTP loss, and gradient norm match the no-recompute
replay exactly. Both updates have finite losses and gradient norms.
Evidence: `optimization/oom_fix/logs/replay_rc8_verified.json`.
This validates the saved 35,079-token maximum batch, not all possible
120k-context inputs or a steady-state end-to-end speedup.

To repeat the full failing-batch test inside the prepared container:

```bash
EXP_ROOT="$PWD/local_artifacts/qwen38_b300_r9/optimization/oom_fix/replay_repeat" \
  LOAD_DEBUG_ROLLOUT_DATA="$PWD/local_artifacts/qwen38_b300_r9/optimization/oom_fix/input/rollout_4.pt" \
  NUM_ROLLOUT=1 VALIDATION_NO_SAVE=1 RECOMPUTE_NUM_LAYERS=8 \
  RAY_SUBMISSION_ID=qwen38-b300-r9-chunk-mask-replay-repeat \
  bash examples/kernel_agent/run_qwen38_b300_piecewise.sh
```

Do not run that replay concurrently with the formal training job on the same
GPUs. The current default is eight block-recomputed layers; all other
approved topology, sampling, optimizer, and checkpoint settings are retained.
The new formal submission is
`qwen38-piecewise-b300-4train4rollout-tp4-cp1-fa4-rc8-mb16k-rtp1-mem85-m320-opt2-r9`,
with driver output in `logs/formal_mem085_opt2_rc8_driver.log`. It starts
from original HF weights at step zero, not from replay-updated parameters;
the failed formal run never reached its first scheduled checkpoint.
Old formal rollout, provenance, and TensorBoard directories are preserved
under `optimization/failed_opt1/`. The persistent TensorBoard was restarted
against the fresh formal directory, keeping the URL
**http://192.168.99.151:6006/** without mixing old step numbers.
