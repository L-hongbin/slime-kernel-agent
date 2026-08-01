#!/bin/bash
# Formal DeepSeek-V4-Flash LoRA RL training launcher.
#
# Reuses the validated R6 full-loop infrastructure (scripts/dsv4/full_loop_smoke.sh:
# fratricide guard, ray bring-up, external-sglang guard, PP2/EP8 Megatron actor +
# SGLang TP4 rollout, Muon, torch_dist checkpoint, rollout routing replay) via
# TASK_MODE=rl, and layers on the RL task.
#
# V4-mandatory settings are kept (Muon optimizer, PP2/EP8, custom v4_model_provider,
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
# pp3 (default): train node64+69+62 (PP3/EP8 = 24 GPUs, ~67% per-GPU activation
#   memory vs PP2 — needed for real drkernel sequences), rollout node70 with the
#   torch-MHC sglang fallback (its sm90a TileLang load issue makes it the right
#   rollout node, not a train node).
#   IMPORTANT: slime sorts actor ranks by node IP (placement_group.py sort_key):
#   162(node62) < 164(node64) < 169(node69) -> node62=stage0(__0-7),
#   node64=stage1(__8-15), node69=stage2(__16-23). The PP3 torch_dist shards
#   MUST be laid out that way per node (they were chain-moved to match).
# pp2: the R6-validated 16-GPU layout (train 64+69, rollout node62).
TOPOLOGY="${TOPOLOGY:-pp3}"
if [[ "${TOPOLOGY}" == "pp3" ]]; then
  export TRAIN_WORKER_HOSTS="${TRAIN_WORKER_HOSTS:-node69_slime node62_slime}"
  export TRAIN_WORKER_IPS="${TRAIN_WORKER_IPS:-10.11.2.169 10.11.2.162}"
  export ROLLOUT_WORKER_HOSTS="${ROLLOUT_WORKER_HOSTS:-node70_slime}"
  export ROLLOUT_WORKER_IPS="${ROLLOUT_WORKER_IPS:-10.11.2.170}"
  export ACTOR_PHYSICAL_HOSTS="${ACTOR_PHYSICAL_HOSTS:-node64 node69 node62}"
  export ROLLOUT_PHYSICAL_HOSTS="${ROLLOUT_PHYSICAL_HOSTS:-node70}"
  export ACTOR_NUM_NODES="${ACTOR_NUM_NODES:-3}"
  export PP_SIZE="${PP_SIZE:-3}"
  export FIRST_LAYERS="${FIRST_LAYERS:-15}"
  export LAST_LAYERS="${LAST_LAYERS:-14}"
  export LOAD="${LOAD:-/nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8-r2-pp3-ep8-torch_dist}"
  # node70 rollout: KERNEL-ON by default (600-5900 tok/s). Its "sm90a TileLang
  # issue" was a poisoned /root/.tilelang JIT cache (cubins from the pip cu13
  # nvcc), quarantined 2026-07-04 — validated by the pp3_drkernel_kernelon +
  # debug runs. Fallback if it ever recurs: SGLANG_DISABLE_CUDA_GRAPH=1 +
  # SGLANG_OPT_USE_TILELANG_MHC_{PRE,POST,SPLIT_SINKHORN}=false (the sinkhorn
  # patch below is env-gated and harmless when the TileLang opts are true).
  export SLIME_PATCH_SGLANG_DSV4_MHC_SINKHORN_TORCH="${SLIME_PATCH_SGLANG_DSV4_MHC_SINKHORN_TORCH:-1}"
  export SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-0}"
  export SGLANG_OPT_USE_TILELANG_MHC_PRE="${SGLANG_OPT_USE_TILELANG_MHC_PRE:-true}"
  export SGLANG_OPT_USE_TILELANG_MHC_POST="${SGLANG_OPT_USE_TILELANG_MHC_POST:-true}"
  export SGLANG_OPT_USE_TILELANG_MHC_SPLIT_SINKHORN="${SGLANG_OPT_USE_TILELANG_MHC_SPLIT_SINKHORN:-true}"
  # train/rollout alignment: mHC prenorm precision. Train computes it in FP32 (B1
  # kernel); sglang defaults to TF32 (`tf32_hc_prenorm_gemm`, mhc.py:730). Codex
  # verified DEEPGEMM_HC_PRENORM=False ALONE is insufficient — with
  # USE_TILELANG_MHC_PRE=true it lands on a different TileLang prenorm kernel, not the
  # torch fp32 path. Force BOTH flags off → sglang's `hc_pre_torch_impl` (fp32,
  # deepseek_v4.py:1290). Both sides fp32 (the CORRECT value; we do NOT degrade train
  # to TF32). Removes the measured ~1.7e-3 TF32 gap; cost = torch (vs kernel) mHC-pre
  # in the rollout, small per-layer gemm. Set V4_ALIGN_MHC_FP32=0 to keep sglang TF32.
  if [[ "${V4_ALIGN_MHC_FP32:-1}" == "1" ]]; then
    export SGLANG_OPT_USE_TILELANG_MHC_PRE=false
    export SGLANG_OPT_DEEPGEMM_HC_PRENORM=False
  fi
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
# Muon is mandatory for V4; the qwen ref used adam 1e-6. Muon needs a larger LR;
# default conservatively and let gates tune it.
export LR="${LR:-1e-5}"

# --- Scale (qwen ref: ctx 16384, rollout 16 x 16 = 256) --------------------
# MAX_CONTEXT_LEN is the total prompt+response window. V4's SGLang strictly
# rejects prompt+new_tokens > context (the qwen ref's SGLang silently clamps
# instead), so response gets half the window and rollout_max_prompt_len is
# pinned to the remainder in full_loop_smoke.sh.
export MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-16384}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-$((MAX_CONTEXT_LEN / 2))}"
export ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-16}"
export N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}"
export NUM_ROLLOUT="${NUM_ROLLOUT:-3000}"
export ROLLOUT_TEMPERATURE="${ROLLOUT_TEMPERATURE:-1}"
export ROLLOUT_TOP_P="${ROLLOUT_TOP_P:-1}"

# --- LoRA (attention/compressor only; MoE frozen) --------------------------
export V4_LORA_DIM="${V4_LORA_DIM:-16}"
export V4_LORA_ALPHA="${V4_LORA_ALPHA:-32}"
export V4_LORA_DROPOUT="${V4_LORA_DROPOUT:-0.0}"

# --- Data + task -----------------------------------------------------------
export PROMPT_DATA="${PROMPT_DATA:-/ms/FM/lihongbin/dataset/CUDA_RL/cuda_rl/prompt_tvm_v2/drkernel_rl_thinking.parquet}"
export KERNEL_ENV_URL="${KERNEL_ENV_URL:-http://127.0.0.1:20211}"
export KERNEL_BACKEND="${KERNEL_BACKEND:-tvm_ffi}"

# --- Checkpointing + logging -----------------------------------------------
export SAVE_MODEL="${SAVE_MODEL:-1}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-20}"
# Adapter-only checkpoints (22 MB) — a FULL V4 save is ~259GB and would fill the
# near-full /nfs within a few save intervals (codex milestone review flagged this
# as the top formal-run risk). Resume: V4_LORA_ADAPTER_RESUME_LOAD=<save dir>.
export V4_LORA_ADAPTER_ONLY_CKPT="${V4_LORA_ADAPTER_ONLY_CKPT:-1}"
export SAVE_OPTIM="${SAVE_OPTIM:-0}"
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
export RAY_JOB_STATUS_TIMEOUT="${RAY_JOB_STATUS_TIMEOUT:-604800}"

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
adv=${ADVANTAGE_ESTIMATOR} lr=${LR} lora_dim=${V4_LORA_DIM}"

exec "${SCRIPT_DIR}/full_loop_smoke.sh"
