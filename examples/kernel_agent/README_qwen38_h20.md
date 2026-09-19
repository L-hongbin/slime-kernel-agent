# Qwen3.8 piecewise on four H20 nodes

Use the `csl_slime_qwen38_rl_0819` containers (root SSH on port 23538).

| Node | IP | Role |
| --- | --- | --- |
| Node70 | 10.11.2.170 | Ray head, 8 training GPUs |
| Node69 | 10.11.2.169 | 8 training GPUs |
| Node53 | 10.11.2.153 | 8 rollout GPUs, two TP4 engines |
| Node64 | 10.11.2.164 | 8 rollout GPUs, two TP4 engines |

Training uses TP4/CP2/PP2, sequence parallelism, distributed FlashQLA GDN,
BF16, FlashAttention for training attention, an FP32 LM head projection, and one
MTP layer. Rollout uses BF16 TP4,
FA3, Triton linear attention, and NEXTN with three speculative steps. The
maximum running requests and CUDA graph maximum batch are both 64 per engine.
The four engines allow 256 client requests and 16 concurrent prompt groups.
The 120,000 context limit, 32,000 response limit, batch sizes, and reward settings
are inherited from the B200 experiment. Block recomputation is 32 layers per
pipeline stage. `KernelGym` remains at `http://127.0.0.1:20211`, forwarding to
the existing A800 evaluation service.

The H20 scripts use the container's installed Python packages and
`/root/Megatron-LM`. They do not require the B200 runtime directories.
`qwen38_h20_env.sh` provides node-local caches and the `bond0` network interface;
NCCL uses `mlx5_0,mlx5_3,mlx5_4`, excluding the historically unstable `mlx5_5`.

## Prepare and verify

Run in each container, from the checkout:

```bash
bash scripts/prepare_qwen38_h20_runtime.sh
```

This checks the installed SGLang version before applying its matching replay
patch, preserves backups, installs the MTP hidden-state detach when needed,
checks the shared output weight detach, and runs the replay numerical check.
It does not install missing output weight isolation automatically.

The preparation script also installs the separate FP32 LM head cache patch
for the verified SGLang 0.5.16 weight-updater layout. Other versions fail rather
than silently skipping the optimization. `git apply --check` must succeed.
The cache preserves BF16 parameters and FP32 projection arithmetic. Target and
MTP draft share the Qwen LM head module, hence share its cache: approximately
1.18 GiB per GPU at TP4. Eager graph warmup creates the cache; disk, distributed,
tensor (including flattened buckets), and IPC weight updates refresh it in
place, preserving the address referenced by CUDA graphs. It is excluded from
the checkpoint state dictionary.

Validate the installed cache on a GPU with:

```bash
python scripts/check_sglang_fp32_lm_head_cache.py --device cuda
```

This checks projection equivalence, weight refresh, shared storage, and CUDA
graph replay after updates. It does not benchmark full rollout throughput.

The preparation script applies `sglang-top_p-reuse.patch` after the base replay
patch. Ordinary and speculative replay share one complete vocabulary sort for
the nucleus IDs and renormalized logprobs. The sampled-token force-keep rule is
unchanged; the shared support is local to the current step. There is no top-32
candidate shortcut. Validate numerical behavior and the one-sort invariant with:

```bash
python scripts/check_sglang_top_p_replay.py --check-sort-reuse --device cuda
```

Both optimizations were installed on all four containers. CPU/CUDA checks passed, including cache refresh after
weight updates and replay of an already captured graph. A real TP4/FA3/NEXTN3
engine with FP32 head and graph batch 64 passed 119,000 input tokens plus 16
generated tokens, and 64 concurrent requests with 128 generated tokens each.
All requests returned finite logprobs and valid top-p metadata.

On an idle H20, median helper/projection timings over six interleaved trials:

| Operation | Before | After |
| --- | ---: | ---: |
| Replay, 42 x 4 rows, vocabulary 248320 | 30.856 ms | 22.659 ms |
| FP32 head, 42 rows, TP4 shard 62080 x 5120 | 3.239 ms | 1.730 ms |
| FP32 head, 168 rows, same shard | 7.036 ms | 5.522 ms |

These are isolated measurements, not full-rollout speedups. Evidence is in
`local_artifacts/h20_replay_reuse/` on the validation workspace (not tracked in Git).
The combined validation run was `qwen38-h20-fa3-tp4-bs64-p1p2-20260918-r4`;
its final status is recorded below.

Defaults:

- Model: `/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.8-27B`
- Data: `Data/prompt_tvm_GEPA4o_v2/torch_ops_difficulty_lt18.parquet` in this checkout
- Runtime/cache: `/nfs/FM/chenshuailin/runtime/qwen38_piecewise_h20`
- W&B: `WANDB_API_KEY` or `WANDB_KEY_FILE`, default `/root/.config/wandb/slime.key`

The parquet was copied from
`/nfs/FM/chenshuailin/projects/kernel_agents/slime-dev-csl-2/Data/prompt_tvm_GEPA4o_v2`
and must have SHA256
`ca5cd825d33406de2f73245274be63617ebccf8160c46f34ffcafffca8d03f94`.
These `/nfs/FM` paths are node-local container mounts, so each node needs its own copy.

On Node70, validate without submitting a job:

```bash
PREFLIGHT_ONLY=1 bash examples/kernel_agent/run_qwen38_h20_piecewise.sh
CONFIG_DRY_RUN=1 bash examples/kernel_agent/run_qwen38_h20_piecewise.sh
```

## Start

Run the first command on Node70, then the matching worker command on each other node:

```bash
bash examples/kernel_agent/start_qwen38_h20_ray.sh node70
bash examples/kernel_agent/start_qwen38_h20_ray.sh node69
bash examples/kernel_agent/start_qwen38_h20_ray.sh node53
bash examples/kernel_agent/start_qwen38_h20_ray.sh node64
```

The scripts do not stop existing Ray processes. The new cluster uses port 6588,
dashboard 8468, service ports 25901–25905, and worker ports 26000–26999.
Object store allocation is 16 GiB per node; the actual `/dev/shm` mount is
checked and the node-local SSD is used only if shared memory is insufficient.

Then submit from Node70:

```bash
bash examples/kernel_agent/run_qwen38_h20_piecewise.sh
```

Set `EXP_ROOT` and `RAY_SUBMISSION_ID` for a separate run. `NUM_ROLLOUT` defaults
to 80. `HF_MODEL_PATH`, `RL_DATA`, `RAY_DASHBOARD`, `WANDB_MODE`, and
`SGLANG_ATTENTION_BACKEND` can also be overridden through the environment.

The save interval currently equals `NUM_ROLLOUT`, so the default run saves only
after 80 rollouts. This interval counts rollouts, not W&B training steps. Each
rollout supplies 256 samples and global batch 128 produces two optimizer steps;
the first scheduled save is therefore after training step 159 (zero-based).
The launcher loads the original BF16 checkpoint and requires a fresh output
directory; it does not automatically resume an earlier training run.

## Validation on 2026-09-18

- All four containers: PyTorch 2.11.0+cu129, SGLang 0.5.16, Ray 2.56.1,
  TileLang 0.1.9; matching patched SGLang and MTP source hashes.
- All four nodes: full SHA256 checks agree for all 18 model shards, config,
  shard index, tokenizer and tokenizer config.
- All four containers: top-p replay numerical checks passed.
- Node70: complete training/SGLang argument parsing and launch preflight passed.
- Node70 + Node69: 16-GPU NCCL all-reduce and TP4/CP2/PP2 groups passed.
- Node70 and Node69: FlashQLA packed BF16 forward/backward passed with the
  local GDN head counts used by TP4/CP2; outputs and all input gradients were finite.
- Node53: the initial TP2 configuration, with 16 maximum running requests and
  CUDA graph batch 16, passed live 128-token top-p replay and a synthetic
  119,000-token input plus 16 generated tokens. This does not establish
  throughput or memory safety for simultaneous maximum-length requests.
- The final TP4 configuration started all four engines with 64 maximum running
  requests and CUDA graph batch 64, completed initial weight synchronization,
  and began sampling with 256 client requests and 16 concurrent prompt groups.

Per-node preparation evidence is under `$H20_RUNTIME/provenance/`.

## Training launch and fixes on 2026-09-18

The first complete launch reused the idle four-node Ray cluster at
`http://10.11.2.170:8268` (GCS port 6388), whose actor/rollout resources already
matched the table above. Initial weight synchronization and rollout completed.
Rollout 0 collected 16 accepted groups in 822.5 seconds, with mean response
length 9,610 tokens and maximum 23,586 tokens.

Two training issues were found and addressed:

- The FP32 LM head returned a view from its custom autograd function. Megatron's
  MTP cross entropy modifies FP32 logits in place, making backward invalid.
  `fp32_lm_head.py` now writes GEMM into an owned output tensor without a logits
  copy. The regression test reproduces the failure before the fix; all 15 tests
  in `tests/test_fp32_lm_head.py` pass after it.
- Training with `--attention-backend fused` produced NaN gradients. Anomaly
  detection localized them to `AttnFuncWithCPAndKVP2PBackward`. Using
  `--attention-backend flash`, as in the existing H20 launch scripts, passed the
  same 32-sample replay with TP4/CP2/PP2 and normal NaN checks enabled. The
  optimizer step completed in about 102 seconds with grad norm `0.3848601`,
  policy loss `-0.00153993`, and MTP loss `0.2730218`.

The successful replay is `qwen38-h20-fa3-replay-20260918-flash`; its artifacts are
in `/nfs/FM/chenshuailin/experiments/qwen38_h20_fa3_replay_20260918_flash` on Node70.
The replay used global batch 32 for validation. The full launcher retains global
batch 128, 16 prompts x 16 samples, and 80 rollouts.

## Full-run outcome checked on 2026-09-19

The r4 run completed training for rollouts 0–49: 100 optimizer updates, ending
at step 99 with finite grad norm `0.1018852` and MTP loss `0.1980834`. It failed
at 2026-09-19 04:52 JST while collecting rollout 50 because group 1864 exhausted
the abort retry limit (11 failed attempts, maximum 10). The underlying abort
cause has not yet been diagnosed. No checkpoint or HF export was saved before
the failure because the interval was 80 rollouts. Checkpoint saving remains
unvalidated in this run.
