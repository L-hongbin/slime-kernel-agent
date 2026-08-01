#!/bin/bash
# Formal DeepSeek-V4-Flash LoRA RL training launcher.
#
# Reuses the validated R6 full-loop infrastructure (scripts/dsv4/full_loop_smoke.sh:
# fratricide guard, ray bring-up, external-sglang guard, PP3/EP8 Megatron actor +
# SGLang EP8/dp-attention rollout, Muon, torch_dist checkpoint, routing replay) via
# TASK_MODE=rl, and layers on the RL task.
#
# V4-mandatory settings are kept (Muon optimizer, PP3/EP8, custom v4_model_provider,
# attention/compressor LoRA with the MoE frozen). Only the portable hyperparameters
# are borrowed from examples/kernel_agent/run.t1.qwen3.6.27B.fasync.sh:
#   - RL_ARGS: advantage-estimator trloo, eps-clip 0.2 / high 0.28, entropy-coef 0
#   - batch: rollout-batch-size 16 x n-samples 16 = global-batch-size 256
#   - MAX_CONTEXT_LEN 16384, constant LR schedule, weight-decay 0.01
#   - DrKernel task: same custom generate/reward/filter/multi-turn CUDA-agent wiring
#
# REWARD_MODE selects the reward source:
#   drkernel (default) -> the real KernelGym CUDA-agent task (needs KernelGym on the
#                         rollout node at KERNEL_ENV_URL)
#   random             -> reward-free RL-math isolation (Gate A; no KernelGym)
#
# Staging (self-decided; see handoffs/deepseek-v4):
#   Gate A: REWARD_MODE=random NUM_ROLLOUT=2 MAX_CONTEXT_LEN=1024 ROLLOUT_BATCH_SIZE=2
#           N_SAMPLES_PER_PROMPT=4   (validates policy_loss + trloo advantages on V4)
#   Gate B: REWARD_MODE=random NUM_ROLLOUT=3 SAVE_MODEL=1 SAVE_INTERVAL=1  (ckpt/resume)
#   Formal: defaults below.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# --- Topology -----------------------------------------------------------------
# pp3 (default): train node54+64+69 (PP3/EP8 = 24 GPUs, ~67% per-GPU
#   activation memory vs PP2 — needed for real drkernel sequences), rollout
#   node70+53 (two EP8 engines). Head = node54_slime. node62 is excluded from
#   the run after its GPU7 exhausted HBM row remapping on 2026-07-15.
#   IMPORTANT: slime sorts actor ranks by node IP (placement_group.py sort_key):
#   154(node54) < 164(node64) < 169(node69) -> node54=stage0(__0-7),
#   node64=stage1(__8-15), node69=stage2(__16-23). The PP3 torch_dist shards
#   MUST be laid out that way per node (re-verified 2026-07-11 after the
#   node64->node54 swap; adding/removing a train node RESHUFFLES this).
# pp2 defaults below are the LEGACY node54-head layout. The formal DS-V4
#   launcher (run.deepseek_v4_flash.fp4.formal.rl.sh) pre-exports its own
#   node69-head topology (node64 returned to the train pool 2026-07-17) and
#   routes around every default here. The old "node64 belongs to zlc's GLM"
#   warning is OBSOLETE.
TOPOLOGY="${TOPOLOGY:-pp3}"
if [[ "${TOPOLOGY}" != "pp3" && "${TOPOLOGY}" != "pp2" ]]; then
  echo "FATAL: TOPOLOGY='${TOPOLOGY}' is not supported (pp3 or pp2)." >&2
  exit 1
fi
if [[ "${TOPOLOGY}" == "pp2" ]]; then
  # PP2 x EP8 (2026-07-14, memory experiment): train = node54(stage0 ranks 0-7,
  # shards copied from node64's pp2 set) + node69(stage1 ranks 8-15, original
  # pp2-era set). The pp2 conversion was built with FIRST_LAYERS=21 LAST_LAYERS=22
  # (convert_torch_dist.sh defaults) — these MUST match or torch_dist load fails.
  # Rollout = 3 engines (node70/53/64). node62 excluded (tenant).
  # PP2@16k FITS via LOG_PROBS_CHUNK_SIZE=2048 (r16 step-1 validated; see
  # pp2_oom_gate.md). PP3 history: ~67% per-GPU activation memory vs PP2.
  export TRAIN_WORKER_HOSTS="${TRAIN_WORKER_HOSTS:-node69_slime}"
  export TRAIN_WORKER_IPS="${TRAIN_WORKER_IPS:-10.11.2.169}"
  export ROLLOUT_WORKER_HOSTS="${ROLLOUT_WORKER_HOSTS:-node70_slime node53_slime node64_slime}"
  export ROLLOUT_WORKER_IPS="${ROLLOUT_WORKER_IPS:-10.11.2.170 10.11.2.153 10.11.2.164}"
  export HEAD_HOST="${HEAD_HOST:-node54_slime}"
  export HEAD_IP="${HEAD_IP:-10.11.2.154}"
  export ACTOR_PHYSICAL_HOSTS="${ACTOR_PHYSICAL_HOSTS:-node54 node69}"
  export ROLLOUT_PHYSICAL_HOSTS="${ROLLOUT_PHYSICAL_HOSTS:-node70 node53 node64}"
  export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-2}"
  export PP_SIZE="${PP_SIZE:-2}"
  export FIRST_LAYERS="${FIRST_LAYERS:-21}"
  export LAST_LAYERS="${LAST_LAYERS:-22}"
  export LOAD="${LOAD:-/nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8-r2-pp2-ep8-torch_dist}"
  export ROLLOUT_GPUS="${ROLLOUT_GPUS:-24}"
  # Per-engine concurrency 96 (user order 2026-07-14): with 3 engines and
  # gbs 256, round-robin gives each engine <=86 in-flight, so 96 admits the
  # whole batch while cutting the CUDA-graph footprint vs 128 (graph max-bs
  # follows SGLANG_MAX_RUNNING_REQUESTS below).
  export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-96}"
  # PP2 gate v4 OOM fix (user order 2026-07-14: keep expandable_segments
  # False, shrink the chunk): the failing alloc was an 8192x129280 fp32
  # logits transient (3.95GB) in the log-probs path on the LAST stage with
  # 3.43GB free. 2048 shrinks it to ~1GB. See pp2_oom_gate.md.
  export LOG_PROBS_CHUNK_SIZE="${LOG_PROBS_CHUNK_SIZE:-2048}"
  # Implemented for the custom V4 loop on 2026-07-15: with full + uniform,
  # RECOMPUTE_NUM_LAYERS=2 checkpoints consecutive 2-layer segments (and a
  # 1-layer tail when needed), matching Megatron's uniform semantics. This
  # reduces the number of saved segment-boundary tensors versus per-layer
  # segments, but the resulting PP2 memory margin still needs a real gate run.
  export RECOMPUTE_NUM_LAYERS="${RECOMPUTE_NUM_LAYERS:-2}"
fi
if [[ "${TOPOLOGY}" == "pp3" ]]; then
  export TRAIN_WORKER_HOSTS="${TRAIN_WORKER_HOSTS:-node64_slime node69_slime}"
  export TRAIN_WORKER_IPS="${TRAIN_WORKER_IPS:-10.11.2.164 10.11.2.169}"
  # Two 8-GPU EP8 rollout engines, one each on node70 and node53.
  export ROLLOUT_WORKER_HOSTS="${ROLLOUT_WORKER_HOSTS:-node70_slime node53_slime}"
  export ROLLOUT_WORKER_IPS="${ROLLOUT_WORKER_IPS:-10.11.2.170 10.11.2.153}"
  # Head + first PP stage stay on node54 (container node54_slime, IP .154).
  # The KernelGYM reward proxy lives on node64 bound to 127.0.0.1:20211 — the
  # head reaches it through a persistent ssh -L tunnel (see
  # /tmp/kgym_tunnel.log on node54_slime).
  # WORKAROUND: root fix is rebinding the service to 0.0.0.0.
  export HEAD_HOST="${HEAD_HOST:-node54_slime}"
  export HEAD_IP="${HEAD_IP:-10.11.2.154}"
  export ACTOR_PHYSICAL_HOSTS="${ACTOR_PHYSICAL_HOSTS:-node54 node64 node69}"
  export ROLLOUT_PHYSICAL_HOSTS="${ROLLOUT_PHYSICAL_HOSTS:-node70 node53}"
  export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-3}"
  export PP_SIZE="${PP_SIZE:-3}"
  export FIRST_LAYERS="${FIRST_LAYERS:-15}"
  export LAST_LAYERS="${LAST_LAYERS:-14}"
  export LOAD="${LOAD:-/nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8-r2-pp3-ep8-torch_dist}"
  export ROLLOUT_GPUS="${ROLLOUT_GPUS:-16}"
  # node70 rollout: KERNEL-ON by default (600-5900 tok/s). Its former sm90a
  # TileLang issue was a poisoned /root/.tilelang JIT cache (cubins from the pip
  # cu13 nvcc), quarantined 2026-07-04 and validated by kernel-on debug runs.
  export SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-0}"
  export SGLANG_OPT_USE_TILELANG_MHC_PRE="${SGLANG_OPT_USE_TILELANG_MHC_PRE:-true}"
  export SGLANG_OPT_USE_TILELANG_MHC_POST="${SGLANG_OPT_USE_TILELANG_MHC_POST:-true}"
  # Train (official TileKernels) and rollout both use TF32 for the mHC prenorm
  # GEMM. Keep the optimized rollout path fixed; precision experiments belong
  # in a dedicated harness rather than the production launcher.
fi

# --- RL hyperparameters (borrowed from the qwen fasync reference) ----------
export TASK_MODE="${TASK_MODE:-rl}"
export REWARD_MODE="${REWARD_MODE:-drkernel}"
export ADVANTAGE_ESTIMATOR="${ADVANTAGE_ESTIMATOR:-trloo}"
export EPS_CLIP="${EPS_CLIP:-0.2}"
export EPS_CLIP_HIGH="${EPS_CLIP_HIGH:-0.28}"
export ENTROPY_COEF="${ENTROPY_COEF:-0.00}"
export LR_DECAY_STYLE="${LR_DECAY_STYLE:-constant}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
# Muon is mandatory for V4; the qwen ref used adam 1e-6. Muon needs a larger LR.
# 1e-5 was the root cause of the "25 steps, zero learning" incident (grad_norm
# flat ~0.018); 1e-4 is the validated working value (same-batch overfit + the
# r7 formal reward climb 0.42->0.58). Do NOT lower without rerunning that gate.
export LR="${LR:-1e-4}"

# --- Scale (qwen ref: ctx 16384, rollout 16 x 16 = 256) --------------------
# MAX_CONTEXT_LEN is the total prompt+response window. V4's SGLang strictly
# rejects prompt+new_tokens > context (the qwen ref's SGLang silently clamps
# instead), so response gets half the window and rollout_max_prompt_len is
# pinned to the remainder in full_loop_smoke.sh.
export MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-16384}"
# Full-window responses (2026-07-11): drkernel's custom generate clamps
# max_new_tokens to (ctx - prompt_len) per request, so response may equal ctx —
# kills the 17% truncation at the old ctx/2 cap. Only the random-reward mode
# needs response < ctx (unclamped path; full_loop_smoke guards it).
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-${MAX_CONTEXT_LEN}}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-3000}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1}"
export USE_ROLLOUT_ROUTING_REPLAY="${USE_ROLLOUT_ROUTING_REPLAY:-1}"
# keep-old-actor + TIS ON by default per user directive 2026-07-13: LoRA
# old-actor = adapter-only snapshot (no 2nd model, ~18ms/step); makes PPO
# ratios truly behavioral (previously ratio==1 -> effectively REINFORCE) and
# TIS = exp(megatron_V - sglang_V) corrects the pure same-version cross-
# engine mismatch in-loss. Validated V1-V3 (tests/deepseek-v4/
# test_dsv4_lora_old_actor.py; handoffs/deepseek-v4/lora_old_actor_tis.md).
# Rollback: USE_KEEP_OLD_ACTOR=0 USE_TIS=0.
export USE_KEEP_OLD_ACTOR="${USE_KEEP_OLD_ACTOR:-1}"
export USE_TIS="${USE_TIS:-1}"
# Sequence-MIS band, CALIBRATED from measured train/rollout mismatch (mean
# |log ratio| ~0.04 -> per-sequence geometric-mean ratios land ~0.99-1.00).
# The _dsv4_task_args.sh default [0.999,1.001] predates the calibration and
# rejects ~100% of healthy sequences -> zero effective reward, silent
# no-learning (bit formal r8e 2026-07-11: reject_rate 0.992 while ratios
# were 0.992-0.998). This was another operator-env-only setting; bake it in.
# MIS band tightened to [0.99,1.01] (user order 2026-07-14; was [0.85,1.15]).
# Token veto unchanged at 1e-4. NOTE: same-version sequence geometric ratios
# have bias ~0.99 under keep-old-actor — expect a materially higher reject
# rate than the ~7% seen at [0.85,1.15]; watch rollout/mis_reject_rate.
# CAUTION: never inline a {...} JSON default inside ${VAR:-...} — bash closes
# the expansion at the wrong brace and appends the tail as literal garbage
# EVEN WHEN VAR IS ALREADY SET (killed the first r21 formal launch twice).
_t1_mis_default='{"aggregation":"turns_geometric","token_veto_threshold":1e-4,"lower":0.99,"upper":1.01,"use_advantage":false}'
readonly T1_SEQUENCE_MIS_CONFIG="${_t1_mis_default}"

# --- LoRA-adapter serving (the validated r7/smoke23g production shape) ------
# These lived only in the operators' launch env before (formal r8c launched
# without them and inherited the smoke's 4-GPU/no-deepep rollout defaults ->
# the V4 model hard-raises on enable_lora + attn_tp>1). Bake them in: adapter
# serving REQUIRES attn-tp==1, i.e. dp-attention with dp == engine tp (EP8),
# and the r7-validated engine sizing.
export ROLLOUT_GPUS="${ROLLOUT_GPUS:-8}"
export ROLLOUT_GPUS_PER_ENGINE="${ROLLOUT_GPUS_PER_ENGINE:-8}"
export USE_SGLANG_DEEPEP="${USE_SGLANG_DEEPEP:-1}"
export SGLANG_DP_SIZE="${SGLANG_DP_SIZE:-8}"
# Per-engine concurrency for the node70/node53 EP8 engines.
export SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-128}"
# Chunked prefill MUST stay at the validated 2048 (dp8 -> 256 tokens/rank ==
# SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK). The smoke's rl default
# tracks ctx (16384 -> 2048 tokens/rank into a 256-cap DeepEP dispatch), which
# is the shape-dependent illegal-memory-access that killed r9n at cycle ~10
# (crash stack: token_dispatcher/deepep.py _dispatch_core during extend).
export SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-2048}"
export SGLANG_CUDA_GRAPH_MAX_BS="${SGLANG_CUDA_GRAPH_MAX_BS:-${SGLANG_MAX_RUNNING_REQUESTS}}"
# Fully-async one-step-ahead train loop (train.py = legacy synchronous loop:
# train_wait ~= the whole generate phase; async hides train inside rollout).
export TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_async.py}"

# --- Speculative decode (user order 2026-07-14: production = EAGLE 1/1/2) ---
# +3% net wall-clock from the bulk+tail A/B studies (tail +52%..+12% at <=8
# req/rank, ~neutral at bulk with LoRA); deeper fixed drafts lose at bulk.
# These were operator-env-only before: the pp2 gate v3/v4 relaunches and the
# first r16 launch silently ran spec-OFF (echo 'sglang_speculative=off') —
# the launch-echo line is the check, the actor env audit does not cover
# SGLANG_* keys. Bake them in.
export SGLANG_SPECULATIVE_ALGORITHM="${SGLANG_SPECULATIVE_ALGORITHM:-EAGLE}"
export SGLANG_SPECULATIVE_NUM_STEPS="${SGLANG_SPECULATIVE_NUM_STEPS:-1}"
export SGLANG_SPECULATIVE_EAGLE_TOPK="${SGLANG_SPECULATIVE_EAGLE_TOPK:-1}"
export SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS="${SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS:-2}"

# Kernel-alignment behavior is part of the DS-V4 implementation and is no
# longer exposed as a family of environment switches.

# --- LoRA (attention/compressor + shared expert; routed MoE frozen) --------
readonly T1_LORA_DIM=16
readonly T1_LORA_ALPHA=32
readonly T1_LORA_DROPOUT=0.0
# rsLoRA (default OFF): trainer scale alpha/sqrt(r) instead of alpha/r — at
# r16/alpha32 a 4x stronger adapter multiplier; co-adjust LR when enabling and
# NEVER flip it across a resume (the adapter ckpt records the scaling and the
# resume fails loud on mismatch). Serving equivalence is automatic (the sync
# ships lora_alpha = alpha*sqrt(r) = 128 so sglang's lora_alpha/r == the trainer
# scale exactly). See handoffs/deepseek-v4/lora_rslora_loraplus.md.
# LoRA+ (default OFF = unset/empty/1.0): eta_B = lambda * eta_A via a separate
# Muon param group for the LoRA B (linear_out) matrices. Start with 4-8.
# Shared-expert LoRA (FFN-direction capacity, +~12.7M params at r16): frozen fp8
# base + bf16 adapter on mlp.shared_experts gate/up/down of all 43 MoE layers.

# --- Data + task -----------------------------------------------------------
export PROMPT_DATA="${PROMPT_DATA:-/ms/FM/lihongbin/dataset/CUDA_RL/cuda_rl/prompt_tvm_v2/drkernel_rl_thinking.parquet}"
export KERNEL_ENV_URL="${KERNEL_ENV_URL:-http://127.0.0.1:20211}"
export KERNEL_BACKEND="${KERNEL_BACKEND:-tvm_ffi}"

# --- Checkpointing + logging -----------------------------------------------
export SAVE_MODEL="${SAVE_MODEL:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
# DS-V4 model checkpoints always avoid writing the frozen base. Optimizer and
# RNG state have independent switches; their RNG defaults preserve the legacy
# behavior by following the corresponding optimizer switch when not overridden.
# Resume: pass --lora-adapter-resume-load <save dir>.
export SAVE_OPTIM="${SAVE_OPTIM:-0}"
export SAVE_RNG="${SAVE_RNG:-${SAVE_OPTIM}}"
export LOAD_OPTIM="${LOAD_OPTIM:-0}"
export LOAD_RNG="${LOAD_RNG:-${LOAD_OPTIM}}"
export USE_WANDB="${USE_WANDB:-1}"
export WANDB_PROJECT="${WANDB_PROJECT:-slime}"
export WANDB_GROUP="${WANDB_GROUP:-FAsync.${KERNEL_BACKEND}.DeepSeek-V4-Flash.CTX${MAX_CONTEXT_LEN}}"
# Populate WANDB_API_KEY from the standard key file (as the qwen ref does) so
# full_loop_smoke.sh can forward it into the Ray runtime env; else wandb logging
# silently fails to authenticate on the actors.
WANDB_KEY_FILE="${WANDB_KEY_FILE:-${HOME}/.config/wandb/slime.key}"
if [[ "${USE_WANDB}" == "1" && -z "${WANDB_API_KEY:-}" && -f "${WANDB_KEY_FILE}" ]]; then
  WANDB_API_KEY="$(tr -d '[:space:]' < "${WANDB_KEY_FILE}")"
  export WANDB_API_KEY
fi
if [[ "${USE_WANDB}" == "1" && -z "${WANDB_API_KEY:-}" ]]; then
  echo "warning: USE_WANDB=1 but no WANDB_API_KEY and no key file at ${WANDB_KEY_FILE}; wandb logging will be disabled" >&2
  export USE_WANDB=0
fi

# Longer-run guards than the smoke default (formal run, many iterations).
export EXTERNAL_SGLANG_GUARD_SECS="${EXTERNAL_SGLANG_GUARD_SECS:-604800}"

# --- NCCL transport ----------------------------------------------------------
# InfiniBand on the HEALTHY HCAs only. Fabric map (ib_write_bw, 2026-07-05):
# mlx5_0/mlx5_3/mlx5_4 = ~364 Gb/s on every train pair (mlx5_4's 33M retrans were
# historical — re-benched clean); mlx5_5 = 0.48 Gb/s (BROKEN: PortXmitWait 1.18T,
# transport retry-exhaustion, but link trains at 400 with zero symbol/rcv errors
# => credit-starvation/congestion, not a bad cable; killed formal v1-v3 with
# IBV_WC_RETRY_EXC_ERR). Full-TCP fallback: NCCL_IB_DISABLE=1. mlx5_5 repair is an
# open infra item (likely switch-port/SM-congestion, needs admin access).
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_3,mlx5_4}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

echo "Launching V4-Flash LoRA RL: TASK_MODE=${TASK_MODE} REWARD_MODE=${REWARD_MODE} \
ctx=${MAX_CONTEXT_LEN} gbs=${GLOBAL_BATCH_SIZE} num_rollout=${NUM_ROLLOUT} \
adv=${ADVANTAGE_ESTIMATOR} lr=${LR} \
lora_defaults=r${T1_LORA_DIM}/alpha${T1_LORA_ALPHA}/rslora0/plusoff/shared1 \
use_rollout_routing_replay=${USE_ROLLOUT_ROUTING_REPLAY} \
checkpoint=save_optim${SAVE_OPTIM}/save_rng${SAVE_RNG}/load_optim${LOAD_OPTIM}/load_rng${LOAD_RNG}"

exec "${SCRIPT_DIR}/full_loop_smoke.sh" \
  --sequence-mis-config "${T1_SEQUENCE_MIS_CONFIG}" \
  --lora-dim "${T1_LORA_DIM}" \
  --lora-alpha "${T1_LORA_ALPHA}" \
  --lora-dropout "${T1_LORA_DROPOUT}" \
  --no-lora-rslora \
  --dsv4-lora-shared-expert \
  --sglang-enable-lora \
  --use-lora-weight-sync \
  "$@"
