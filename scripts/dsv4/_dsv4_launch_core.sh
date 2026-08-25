#!/bin/bash
# Shared DeepSeek-V4 full-loop launch core
#
# TASK_MODE=smoke_sft runs the small integration smoke; TASK_MODE=rl runs the
# configured training recipe. Public wrappers own topology and experiment values.
set -euo pipefail

OVERLONG_PENALTY=0
OVERLONG_BUFFER_LEN=2048
OVERLONG_PENALTY_FACTOR=1.0
ENTROPY_COMMON_PROBE=0
ASSERT_ZERO_LORA_OUT=0
DATA_PAD_SIZE_MULTIPLIER=128
SEQUENCE_MIS_CONFIG=
LORA_DIM=4
LORA_ALPHA=8
LORA_DROPOUT=0.0
LORA_RSLORA=0
LORA_PLUS_LAMBDA=
LORA_SHARED_EXPERT=0
LORA_ADAPTER_RESUME_LOAD=
LORA_CHECKPOINT_MAX_NODE_BYTES=2147483648
SGLANG_ENABLE_LORA=0
USE_LORA_WEIGHT_SYNC=0
while (( "$#" > 0 )); do
  case "$1" in
    --overlong-penalty)
      OVERLONG_PENALTY=1
      shift
      ;;
    --overlong-buffer-len)
      [[ "$#" -ge 2 ]] || { echo "FATAL: --overlong-buffer-len requires a value" >&2; exit 2; }
      OVERLONG_BUFFER_LEN=$2
      shift 2
      ;;
    --overlong-penalty-factor)
      [[ "$#" -ge 2 ]] || { echo "FATAL: --overlong-penalty-factor requires a value" >&2; exit 2; }
      OVERLONG_PENALTY_FACTOR=$2
      shift 2
      ;;
    --entropy-common-probe)
      ENTROPY_COMMON_PROBE=1
      shift
      ;;
    --assert-zero-lora-out)
      ASSERT_ZERO_LORA_OUT=1
      shift
      ;;
    --data-pad-size-multiplier)
      [[ "$#" -ge 2 ]] || { echo "FATAL: --data-pad-size-multiplier requires a value" >&2; exit 2; }
      DATA_PAD_SIZE_MULTIPLIER=$2
      shift 2
      ;;
    --sequence-mis-config)
      [[ "$#" -ge 2 ]] || { echo "FATAL: --sequence-mis-config requires a value" >&2; exit 2; }
      SEQUENCE_MIS_CONFIG=$2
      shift 2
      ;;
    --lora-dim)
      [[ "$#" -ge 2 ]] || { echo "FATAL: --lora-dim requires a value" >&2; exit 2; }
      LORA_DIM=$2
      shift 2
      ;;
    --lora-alpha)
      [[ "$#" -ge 2 ]] || { echo "FATAL: --lora-alpha requires a value" >&2; exit 2; }
      LORA_ALPHA=$2
      shift 2
      ;;
    --lora-dropout)
      [[ "$#" -ge 2 ]] || { echo "FATAL: --lora-dropout requires a value" >&2; exit 2; }
      LORA_DROPOUT=$2
      shift 2
      ;;
    --lora-rslora)
      LORA_RSLORA=1
      shift
      ;;
    --no-lora-rslora)
      LORA_RSLORA=0
      shift
      ;;
    --lora-plus-lambda)
      [[ "$#" -ge 2 ]] || { echo "FATAL: --lora-plus-lambda requires a value" >&2; exit 2; }
      LORA_PLUS_LAMBDA=$2
      shift 2
      ;;
    --dsv4-lora-shared-expert)
      LORA_SHARED_EXPERT=1
      shift
      ;;
    --no-dsv4-lora-shared-expert)
      LORA_SHARED_EXPERT=0
      shift
      ;;
    --lora-adapter-resume-load)
      [[ "$#" -ge 2 ]] || { echo "FATAL: --lora-adapter-resume-load requires a value" >&2; exit 2; }
      LORA_ADAPTER_RESUME_LOAD=$2
      shift 2
      ;;
    --lora-checkpoint-max-node-bytes)
      [[ "$#" -ge 2 ]] || { echo "FATAL: --lora-checkpoint-max-node-bytes requires a value" >&2; exit 2; }
      LORA_CHECKPOINT_MAX_NODE_BYTES=$2
      shift 2
      ;;
    --sglang-enable-lora)
      SGLANG_ENABLE_LORA=1
      shift
      ;;
    --no-sglang-enable-lora)
      SGLANG_ENABLE_LORA=0
      shift
      ;;
    --use-lora-weight-sync)
      USE_LORA_WEIGHT_SYNC=1
      shift
      ;;
    --no-use-lora-weight-sync)
      USE_LORA_WEIGHT_SYNC=0
      shift
      ;;
    *)
      echo "FATAL: unknown launcher argument '$1'" >&2
      exit 2
      ;;
  esac
done

REPO=${REPO:-/nfs/FM/chenshuailin/projects/kernel_agents/slime-v4flash-lora}
HF_CKPT=${HF_CKPT:-/nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8}
LOAD=${LOAD:-/nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8-r2-pp2-ep8-torch_dist}
PROMPT_DATA=${PROMPT_DATA:-${REPO}/Data/dsv4_full_loop_smoke.jsonl}
SCRATCH=${SCRATCH:-/nfs/FM/chenshuailin/projects/kernel_agents/slime-v4flash-lora/experiments/csl_v4r6_full_loop}
SAVE=${SAVE:-${SCRATCH}/out}
DEBUG_DIR=${DEBUG_DIR:-${SCRATCH}/debug}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
LOG=${LOG:-${REPO}/local_artifacts/deepseek-v4/r2_logs/r6_pp2_ep8_full_loop_${RUN_ID}.log}

HEAD_HOST=${HEAD_HOST:-node64_slime}
HEAD_IP=${HEAD_IP:-10.11.2.164}
TRAIN_WORKER_HOSTS=(${TRAIN_WORKER_HOSTS:-node69_slime})
TRAIN_WORKER_IPS=(${TRAIN_WORKER_IPS:-10.11.2.169})
ROLLOUT_WORKER_HOSTS=(${ROLLOUT_WORKER_HOSTS:-node62_slime})
ROLLOUT_WORKER_IPS=(${ROLLOUT_WORKER_IPS:-10.11.2.162})
WORKER_HOSTS=("${TRAIN_WORKER_HOSTS[@]}" "${ROLLOUT_WORKER_HOSTS[@]}")
WORKER_IPS=("${TRAIN_WORKER_IPS[@]}" "${ROLLOUT_WORKER_IPS[@]}")
ACTOR_PHYSICAL_HOSTS=(${ACTOR_PHYSICAL_HOSTS:-node64 node69})
ROLLOUT_PHYSICAL_HOSTS=(${ROLLOUT_PHYSICAL_HOSTS:-node62})
# Hosts swept by kill_old_processes + the external sglang guard. Default to the
# hosts THIS run actually uses — a fixed all-nodes list SIGKILLed an unrelated
# LoRA repro server on node70 while a smoke ran on 64/69/62 (fratricide variant
# 3, 2026-07-10; the guard loops for hours and kills with zero log output).
PHYSICAL_CLEAN_HOSTS=(${PHYSICAL_CLEAN_HOSTS:-${ACTOR_PHYSICAL_HOSTS[@]} ${ROLLOUT_PHYSICAL_HOSTS[@]}})

ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-2}
ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-8}
ACTOR_CPUS_PER_NODE=${ACTOR_CPUS_PER_NODE:-64}
ROLLOUT_GPUS=${ROLLOUT_GPUS:-4}
ROLLOUT_GPUS_PER_ENGINE=${ROLLOUT_GPUS_PER_ENGINE:-4}
ROLLOUT_CPUS_PER_NODE=${ROLLOUT_CPUS_PER_NODE:-64}
ACTOR_PLACEMENT_RESOURCE=${ACTOR_PLACEMENT_RESOURCE:-slime_actor}
ROLLOUT_PLACEMENT_RESOURCE=${ROLLOUT_PLACEMENT_RESOURCE:-slime_rollout}

PP_SIZE=${PP_SIZE:-2}
EP_SIZE=${EP_SIZE:-8}
# CP_SIZE>1 enables contiguous context parallelism (V4-Flash). Megatron derives
# DP from world/(TP*PP*CP): formal DS-V4 is PP1 x CP2 x DP8, while a legacy PP2
# CP2 variant is DP4. EP stays 8 and divides DP*CP in either topology.
CP_SIZE=${CP_SIZE:-1}
FIRST_LAYERS=${FIRST_LAYERS:-21}
LAST_LAYERS=${LAST_LAYERS:-22}
SAVE_MODEL=${SAVE_MODEL:-0}
SAVE_INTERVAL=${SAVE_INTERVAL:-100000}
ASYNC_SAVE=${ASYNC_SAVE:-0}
# Megatron's generic default remains 128. The formal DS-V4 recipe passes 1024
# explicitly to bound the static TileLang shape set.
if [[ ! "${DATA_PAD_SIZE_MULTIPLIER}" =~ ^[1-9][0-9]*$ ]]; then
  echo "FATAL: --data-pad-size-multiplier must be a positive integer, got '${DATA_PAD_SIZE_MULTIPLIER}'." >&2
  exit 2
fi
# Checkpoint state controls are independently overridable.  The RNG defaults
# follow their optimizer counterparts to preserve the historical launcher
# behavior when SAVE_RNG/LOAD_RNG are not explicitly set.
SAVE_OPTIM=${SAVE_OPTIM:-0}
SAVE_RNG=${SAVE_RNG:-${SAVE_OPTIM}}
LOAD_OPTIM=${LOAD_OPTIM:-0}
LOAD_RNG=${LOAD_RNG:-${LOAD_OPTIM}}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
# Number of prompt groups kept in the rollout candidate pool.  This is
# deliberately independent of N_SAMPLES_PER_PROMPT: increasing it improves
# refill/low-variance-filter headroom without changing the accepted training
# batch (ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT).
OVER_SAMPLING_BATCH_SIZE=${OVER_SAMPLING_BATCH_SIZE:-${ROLLOUT_BATCH_SIZE}}
# Optional adaptive tail refill.  Empty preserves slime's fixed batch-size
# behavior; the formal DS-V4 recipe sets 2 so a deficit of k accepted prompt groups submits 2*k
# candidates, capped by OVER_SAMPLING_BATCH_SIZE.
OVER_SAMPLING_REFILL_FACTOR=${OVER_SAMPLING_REFILL_FACTOR:-}
if [[ -n "${OVER_SAMPLING_REFILL_FACTOR}" && ! "${OVER_SAMPLING_REFILL_FACTOR}" =~ ^[1-9][0-9]*$ ]]; then
  echo "FATAL: OVER_SAMPLING_REFILL_FACTOR must be empty or a positive integer, got '${OVER_SAMPLING_REFILL_FACTOR}'." >&2
  exit 2
fi
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-1}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-1}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-1}
NUM_ROLLOUT=${NUM_ROLLOUT:-1}
START_ROLLOUT_ID=${START_ROLLOUT_ID:-0}
USE_ROLLOUT_ROUTING_REPLAY=${USE_ROLLOUT_ROUTING_REPLAY:-1}
CLEAN_DEBUG_DIR=${CLEAN_DEBUG_DIR:-1}

# --- Task mode -------------------------------------------------------------
# TASK_MODE=smoke_sft (default): the original R6 SFT smoke (random reward,
#   sft_loss, no advantages) — behavior is byte-identical to the validated run.
# TASK_MODE=rl: formal RL training (policy_loss + advantages). REWARD_MODE picks
#   the reward source:
#     random   -> --rm-type random (isolates the RL math from KernelGym; Gate A)
#     drkernel -> full DrKernel custom generate/reward/filter/multi-turn (real task)
TASK_MODE=${TASK_MODE:-smoke_sft}
REWARD_MODE=${REWARD_MODE:-$([[ "${TASK_MODE}" == "rl" ]] && echo drkernel || echo random)}
# Optimizer LR/schedule (COMMON_ARGS). Muon is mandatory for V4; only the LR and
# schedule shape are borrowed from the qwen reference (constant, wd).
LR=${LR:-$([[ "${TASK_MODE}" == "rl" ]] && echo 1e-5 || echo 1e-4)}
LR_DECAY_STYLE=${LR_DECAY_STYLE:-constant}
WEIGHT_DECAY=${WEIGHT_DECAY:-$([[ "${TASK_MODE}" == "rl" ]] && echo 0.01 || echo 0.0)}
MUON_MOMENTUM=${MUON_MOMENTUM:-0.9}
MUON_USE_NESTEROV=${MUON_USE_NESTEROV:-0}
MUON_NUM_NS_STEPS=${MUON_NUM_NS_STEPS:-5}
MUON_COEFFICIENT_TYPE=${MUON_COEFFICIENT_TYPE:-}
MUON_SCALE_MODE=${MUON_SCALE_MODE:-spectral}
MUON_EXTRA_SCALE_FACTOR=${MUON_EXTRA_SCALE_FACTOR:-1.0}
MUON_FP32_MATMUL_PREC=${MUON_FP32_MATMUL_PREC:-medium}
MUON_TP_MODE=${MUON_TP_MODE:-blockwise}
MUON_ARGS=(
  --muon-momentum "${MUON_MOMENTUM}"
  --muon-num-ns-steps "${MUON_NUM_NS_STEPS}"
  --muon-scale-mode "${MUON_SCALE_MODE}"
  --muon-extra-scale-factor "${MUON_EXTRA_SCALE_FACTOR}"
  --muon-fp32-matmul-prec "${MUON_FP32_MATMUL_PREC}"
  --muon-tp-mode "${MUON_TP_MODE}"
)
if [[ "${MUON_USE_NESTEROV}" == "1" ]]; then
  MUON_ARGS+=(--muon-use-nesterov)
fi
if [[ -n "${MUON_COEFFICIENT_TYPE}" ]]; then
  MUON_ARGS+=(--muon-coefficient-type "${MUON_COEFFICIENT_TYPE}")
fi
# RL hyperparameters (borrowed from run.t1.qwen3.6.27B.fasync.sh RL_ARGS).
ADVANTAGE_ESTIMATOR=${ADVANTAGE_ESTIMATOR:-trloo}
EPS_CLIP=${EPS_CLIP:-0.2}
EPS_CLIP_HIGH=${EPS_CLIP_HIGH:-0.28}
ENTROPY_COEF=${ENTROPY_COEF:-0.00}
# Real-task context/response windows (smoke_sft keeps its tiny 512/16 windows).
# MAX_CONTEXT_LEN is the TOTAL serving window (prompt + response).
# - REWARD_MODE=drkernel: the custom generate clamps max_new_tokens to
#   (rollout_max_context_len - prompt_len) per turn (_sampling_params_for_
#   prompt_context), so MAX_RESPONSE_LEN may equal MAX_CONTEXT_LEN (long
#   prompts simply get a smaller generation budget).
# - REWARD_MODE=random uses slime's default rollout, which sends
#   max_new_tokens un-clamped and V4's SGLang strictly rejects
#   prompt+new_tokens > context — so response must stay < context there.
MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN:-16384}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-$((MAX_CONTEXT_LEN / 2))}
if [[ "${TASK_MODE}" == "rl" && "${REWARD_MODE:-drkernel}" != "drkernel" \
      && "${MAX_RESPONSE_LEN}" -ge "${MAX_CONTEXT_LEN}" ]]; then
  echo "MAX_RESPONSE_LEN (${MAX_RESPONSE_LEN}) must be < MAX_CONTEXT_LEN (${MAX_CONTEXT_LEN}) for the unclamped random-reward rollout" >&2
  exit 2
fi
# Prompt budget: the remaining window, floored to a sane minimum when
# response==context (slime requires prompt_len <= context-1; DrKernel prompts
# are ~1-3k tokens, 8192 is a generous ceiling).
_prompt_budget=$((MAX_CONTEXT_LEN - MAX_RESPONSE_LEN))
if [[ "${_prompt_budget}" -le 0 ]]; then _prompt_budget=8192; fi
ROLLOUT_MAX_PROMPT_LEN=${ROLLOUT_MAX_PROMPT_LEN:-${_prompt_budget}}
# Data keys: smoke toy jsonl uses input/label/metadata; DrKernel parquet uses
# prompt/reward_model/extra_info (matches the qwen reference).
INPUT_KEY=${INPUT_KEY:-$([[ "${REWARD_MODE}" == "drkernel" ]] && echo prompt || echo input)}
LABEL_KEY=${LABEL_KEY:-$([[ "${REWARD_MODE}" == "drkernel" ]] && echo reward_model || echo label)}
METADATA_KEY=${METADATA_KEY:-$([[ "${REWARD_MODE}" == "drkernel" ]] && echo extra_info || echo metadata)}
# wandb (off by default; on for formal RL when a key is available).
USE_WANDB=${USE_WANDB:-0}
WANDB_PROJECT=${WANDB_PROJECT:-slime}
WANDB_GROUP=${WANDB_GROUP:-v4flash_lora_rl}
# KernelGym / DrKernel wiring (only used when REWARD_MODE=drkernel).
KERNEL_ENV_URL=${KERNEL_ENV_URL:-http://127.0.0.1:20211}
KERNEL_BACKEND=${KERNEL_BACKEND:-tvm_ffi}
# SGLang serving window must cover the rollout context: the smoke's tiny 512 is
# only valid for smoke_sft; RL uses the real MAX_CONTEXT_LEN. Running-requests
# also scale up for RL throughput.
SGLANG_CONTEXT_LENGTH=${SGLANG_CONTEXT_LENGTH:-$([[ "${TASK_MODE}" == "rl" ]] && echo "${MAX_CONTEXT_LEN}" || echo 512)}
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-$([[ "${TASK_MODE}" == "rl" ]] && echo 16 || echo 2)}

# Fixed Ray runtime for the single supported launch path: one direct driver and
# one cluster per physical fleet. The run lock and destructive Ray cleanup make
# alternate per-run ports/paths an unsupported form of concurrency.
readonly RAY_PORT=6396
readonly RAY_DASHBOARD_PORT=8280
readonly RAY_DASHBOARD_AGENT_LISTEN_PORT=52365
readonly RAY_DASHBOARD_AGENT_GRPC_PORT=52366
readonly RAY_RUNTIME_ENV_AGENT_PORT=52367
readonly RAY_HEAD_ADDR="${HEAD_IP}:${RAY_PORT}"
readonly RAY_WAIT_TIMEOUT=300
readonly RAY_TMP_ROOT=/dev/shm/v4r6_full_loop_ray
readonly RUNTIME_CACHE_ROOT=/dev/shm/v4r6_full_loop_cache
readonly RAY_OBJECT_STORE_MEMORY=20000000000
readonly RAY_DASHBOARD_AGENT_PATCHER="${REPO}/scripts/patch_ray_dashboard_agent_early_port.py"
SLIME_RAY_DASHBOARD_AGENT_EARLY_PORT=${SLIME_RAY_DASHBOARD_AGENT_EARLY_PORT:-1}
SSH_OPTS=${SSH_OPTS:--o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null}
PHYSICAL_SSH_OPTS=${PHYSICAL_SSH_OPTS:-${SSH_OPTS}}
CLEANUP_RAY_ON_EXIT=${CLEANUP_RAY_ON_EXIT:-1}
EXTERNAL_SGLANG_GUARD=${EXTERNAL_SGLANG_GUARD:-1}
EXTERNAL_SGLANG_GUARD_SECS=${EXTERNAL_SGLANG_GUARD_SECS:-14400}
EXTERNAL_SGLANG_GUARD_POLL_SECS=${EXTERNAL_SGLANG_GUARD_POLL_SECS:-10}
EXTERNAL_SGLANG_GUARD_FILE=${EXTERNAL_SGLANG_GUARD_FILE:-/tmp/slime_external_sglang_guard_${RUN_ID}.alive}
RAY_STARTED=0
EXTERNAL_SGLANG_GUARD_PIDS=()

# 0.85 per user directive 2026-07-13 (was 0.8): more KV-pool headroom on the
# fp8 production engines. Rollback: export SGLANG_MEM_FRACTION_STATIC=0.8.
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.85}
# DS-V4 training uses the single DSpark SGLang stack. Its packed DSV4-CUDA
# memory pool requires uint8 storage, so fp8_e4m3 is the only supported KV type.
SGLANG_KV_CACHE_DTYPE=${SGLANG_KV_CACHE_DTYPE:-fp8_e4m3}
if [[ "${SGLANG_KV_CACHE_DTYPE}" == bf* ]]; then
  echo "FATAL: the DS-V4 DSpark rollout stack cannot serve ${SGLANG_KV_CACHE_DTYPE} KV (packed-pool uint8 assert; G-M1)." >&2
  exit 1
fi
# Prefill sizing must cover the longest prompt; smoke's 1024 is fine for the
# tiny SFT smoke and for RL prefill chunks up to the context window.
SGLANG_CHUNKED_PREFILL_SIZE=${SGLANG_CHUNKED_PREFILL_SIZE:-$([[ "${TASK_MODE}" == "rl" ]] && echo "${SGLANG_CONTEXT_LENGTH}" || echo 1024)}
SGLANG_MAX_PREFILL_TOKENS=${SGLANG_MAX_PREFILL_TOKENS:-$([[ "${TASK_MODE}" == "rl" ]] && echo "${SGLANG_CONTEXT_LENGTH}" || echo 1024)}
# cuda-graph batch must cover the concurrent request count (qwen ref pins it to
# max-running-requests).
SGLANG_CUDA_GRAPH_MAX_BS=${SGLANG_CUDA_GRAPH_MAX_BS:-$([[ "${TASK_MODE}" == "rl" ]] && echo "${SGLANG_MAX_RUNNING_REQUESTS}" || echo 2)}
SGLANG_DISABLE_CUDA_GRAPH=${SGLANG_DISABLE_CUDA_GRAPH:-0}
SGLANG_ROUTER_REGISTRATION_TIMEOUT_SECS=${SGLANG_ROUTER_REGISTRATION_TIMEOUT_SECS:-120}
USE_SGLANG_DEEPEP=${USE_SGLANG_DEEPEP:-0}
SGLANG_DP_SIZE=${SGLANG_DP_SIZE:-4}
# dp-attention divides chunked_prefill_size by dp_size and dsv4 forces
# page_size=256, so the post-division chunk must stay a multiple of 256.
# The smoke_sft default (1024) breaks at dp8 (1024/8=128 -> hard assert at
# server init); clamp to dp*256 (bit smoke23c 2026-07-10).
if [[ "${SGLANG_CHUNKED_PREFILL_SIZE}" -lt $((SGLANG_DP_SIZE * 256)) ]]; then
  SGLANG_CHUNKED_PREFILL_SIZE=$((SGLANG_DP_SIZE * 256))
fi
# Same dp division applies to cuda-graph capture sizes: the smoke_sft default
# (2) yields capture_bs=[0] at dp8 -> "AssertionError: capture_bs=[0]" at
# engine init (bit smoke23e 2026-07-10, again fp4 smoke takes 4-5 2026-07-16:
# the per-DP-rank REQUEST POOL also floors to 0 below dp_size and feeds
# get_batch_sizes_to_capture). Enforce the validated >=2-per-dp-rank floor on
# BOTH knobs (codex launcher review finding 3); warn when raising an explicit
# value rather than silently running a different config.
if [[ "${USE_SGLANG_DEEPEP}" == "1" || "${SGLANG_ENABLE_DP_ATTENTION:-0}" == "1" || "${V4_FP4_FROZEN_EXPERTS:-0}" == "1" ]]; then
  if [[ "${SGLANG_CUDA_GRAPH_MAX_BS}" -lt $((SGLANG_DP_SIZE * 2)) ]]; then
    echo "WARN: raising SGLANG_CUDA_GRAPH_MAX_BS ${SGLANG_CUDA_GRAPH_MAX_BS} -> $((SGLANG_DP_SIZE * 2)) (>=2 per DP rank required)" >&2
    SGLANG_CUDA_GRAPH_MAX_BS=$((SGLANG_DP_SIZE * 2))
  fi
  if [[ "${SGLANG_MAX_RUNNING_REQUESTS}" -lt $((SGLANG_DP_SIZE * 2)) ]]; then
    echo "WARN: raising SGLANG_MAX_RUNNING_REQUESTS ${SGLANG_MAX_RUNNING_REQUESTS} -> $((SGLANG_DP_SIZE * 2)) (>=2 per DP rank required)" >&2
    SGLANG_MAX_RUNNING_REQUESTS=$((SGLANG_DP_SIZE * 2))
  fi
fi
# H20 has 78 SMs; DeepEP cooperative launch needs all blocks resident, so the
# old H800-derived default of 96 ALWAYS fails here ("too many blocks in
# cooperative launch" — bit formal r7b 2026-07-09). 64 is the validated value.
SGLANG_DEEPEP_CONFIG=${SGLANG_DEEPEP_CONFIG:-'{"normal_dispatch":{"num_sms":64},"normal_combine":{"num_sms":64}}'}

export PYTHONUNBUFFERED=1
export SLIME_RAY_DASHBOARD_AGENT_EARLY_PORT
export RAY_raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s:-120}
# The dashboard agent can spend tens of seconds in GPU probing on these nodes.
# Raylet otherwise aborts in WaitForDashboardAgentPorts before the agent writes
# its listen-port file.
export RAY_agent_register_timeout_ms=${RAY_agent_register_timeout_ms:-${RAY_AGENT_REGISTER_TIMEOUT_MS:-180000}}
RAY_SYSTEM_CONFIG_JSON=${RAY_SYSTEM_CONFIG_JSON:-"{\"agent_register_timeout_ms\": ${RAY_agent_register_timeout_ms}}"}
export PATH="/usr/local/cuda/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
# TileKernels (DeepSeek official mHC kernels; always used outside torch-reference diagnostics).
TILEKERNELS_DIR=${TILEKERNELS_DIR:-/nfs/FM/chenshuailin/projects/kernel_agents/TileKernels}
export PYTHONPATH="${REPO}:/root/Megatron-LM:${TILEKERNELS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
# rsLoRA (default OFF): adapter forward scale alpha/sqrt(r) instead of alpha/r.
# At r=16/alpha=32 that is a 4x stronger multiplier — co-adjust LR (see
# handoffs/deepseek-v4/lora_training_features.md). Serving follows automatically
# (the sync exports the effective lora_alpha = alpha*sqrt(r)).
# LoRA+ (default OFF = unset/empty/1.0): eta_B = lambda * eta_A via a separate
# optimizer param group for the LoRA B (linear_out) matrices.
# Shared-expert LoRA (adds *.mlp.shared_experts.{gate,up,down}_proj to the train
# target set; serving side gains gate_up_proj/down_proj targets — see below).
# Every LoRA-trained model keeps frozen base weights out of saves.
# --lora-adapter-resume-load overlays the saved adapter and,
# when enabled independently below, Muon optimizer/RNG state on the cold base.
# LoRA-adapter serving: sglang serves the frozen fp8 base + a hot-swapped LoRA
# adapter, and the train loop syncs ONLY the trainable adapter params each step
# (eliminates the growing fp8-requant drift of the full-weight merge sync).
# Default OFF (full-weight merge sync unchanged). When ON:
#   * engines launch with --sglang-enable-lora (SGLANG_ARGS block below),
#   * the train loop uses --use-lora-weight-sync (LORA_SYNC_ARGS below),
#   * SGLANG_OPT_FUSE_WQA_WKV is forced 0 so wq_a/wkv are separate LoRA targets,
#   * DSPARK speculative decoding is supported by the pinned DSpark stack, and
#   * attention must be replicated (attn-tp==1): use the dp-attention config
#     (USE_SGLANG_DEEPEP=1 with SGLANG_DP_SIZE == rollout TP). CUDA graph stays on.
SGLANG_MAX_LORA_RANK=${SGLANG_MAX_LORA_RANK:-${LORA_DIM}}
# The alternating-adapter sync keeps the OLD adapter resident while it loads the
# NEW one (load-new -> switch -> unload-old double buffer), so it needs >= 2
# mem-pool slots. Default to 2 when the weight sync is on, else 1.
if [[ "${USE_LORA_WEIGHT_SYNC}" == "1" ]]; then
  SGLANG_MAX_LORAS_PER_BATCH=${SGLANG_MAX_LORAS_PER_BATCH:-2}
else
  SGLANG_MAX_LORAS_PER_BATCH=${SGLANG_MAX_LORAS_PER_BATCH:-1}
fi
if [[ "${LORA_SHARED_EXPERT}" == "1" ]]; then
  # Shared-expert targets: exported native leaves w1/w3 stack into the served
  # tp1-replicated gate_up_proj; w2 -> down_proj. Pass the NORMALIZED names.
  SGLANG_LORA_TARGET_MODULES=${SGLANG_LORA_TARGET_MODULES:-"wq_a wkv wq_b wo_b wkv_gate gate_up_proj down_proj"}
else
  SGLANG_LORA_TARGET_MODULES=${SGLANG_LORA_TARGET_MODULES:-"wq_a wkv wq_b wo_b wkv_gate"}
fi
if [[ "${SGLANG_ENABLE_LORA}" == "1" ]]; then
  # wq_a and wkv must stay separate ReplicatedLinear LoRA targets; the default
  # fusion (=1) merges them into wqkv_a and the targets disappear.
  export SGLANG_OPT_FUSE_WQA_WKV=0
  if [[ "${USE_LORA_WEIGHT_SYNC}" != "1" ]]; then
    echo "WARN: SGLANG_ENABLE_LORA=1 but USE_LORA_WEIGHT_SYNC=0 (engine serves LoRA but train still full-weight syncs)" >&2
  fi
  # LoRA+DSPARK is GPU-validated on the single supported runtime. EAGLE/NEXTN
  # remain allowlisted in the pinned source but have not been re-validated.
  if [[ -n "${SGLANG_SPECULATIVE_ALGORITHM:-}" && "${SGLANG_SPECULATIVE_ALGORITHM}" != "DSPARK" ]]; then
    echo "WARN: DS-V4 LoRA rollout is validated only with DSPARK speculative decoding; ${SGLANG_SPECULATIVE_ALGORITHM} is unvalidated." >&2
  fi
fi
if [[ "${USE_LORA_WEIGHT_SYNC}" == "1" && "${SGLANG_ENABLE_LORA}" != "1" ]]; then
  echo "FATAL: USE_LORA_WEIGHT_SYNC=1 requires SGLANG_ENABLE_LORA=1 (adapter load would hit a LoRA-disabled server)." >&2
  exit 1
fi
if [[ "${USE_LORA_WEIGHT_SYNC}" == "1" && "${SGLANG_MAX_LORAS_PER_BATCH}" -lt 2 ]]; then
  echo "FATAL: USE_LORA_WEIGHT_SYNC=1 needs SGLANG_MAX_LORAS_PER_BATCH>=2 (alternating load-new-before-unload-old keeps two adapters resident during the swap); got ${SGLANG_MAX_LORAS_PER_BATCH}." >&2
  exit 1
fi
# Activation checkpointing in the V4 decoder loop (recompute in backward).
# Required at formal scale: without it the train backward OOMs (1F1B holds
# pp_size microbatches of activations). Default ON for RL, OFF for the tiny smoke.
# Configured via Megatron's own --recompute-* flags (RECOMPUTE_ARGS below); the
# V4 provider bridges --recompute-granularity full into the custom decoder loop.
RECOMPUTE=${RECOMPUTE:-$([[ "${TASK_MODE}" == "rl" ]] && echo 1 || echo 0)}
# Packed-MXFP4 backbone mode (OFFICIAL deepseek-ai/DeepSeek-V4-Flash checkpoint):
# V4_FP4_FROZEN_EXPERTS=1 keeps routed experts packed-FP4-resident on BOTH sides
# and selects W4A16 MoE compute (trainer: transient-unpack bf16 GEMM; rollout:
# flashinfer_mxfp4 cutlass, a2a=none). Requires HF_CKPT/LOAD to point at the
# official checkpoint + its packed torch_dist conversion (FP4_EXPERTS=1
# convert_torch_dist.sh). Design: handoffs/deepseek-v4/fp4_w4a16_design.md.
export V4_FP4_FROZEN_EXPERTS=${V4_FP4_FROZEN_EXPERTS:-0}
if [[ "${V4_FP4_FROZEN_EXPERTS}" == "1" ]]; then
  # a2a=none is NOT in the fork's shared-expert-TP1 predicate (deepseek_v2.py:694-701),
  # so without this the shared expert TP-shards (per-rank gate_up 4096/8=512) and the
  # UNSHARDED shared-expert LoRA adapters fail at engine init ("LoRA B output dim 4096
  # does not match base partition prefix dim 512 for 2 slices"). TP1 replication is
  # also the production (DeepEP-era) semantics the train side is aligned against.
  export SGLANG_SHARED_EXPERT_TP1=${SGLANG_SHARED_EXPERT_TP1:-1}
fi
# DS-V4 training uses the official TileKernels mHC. Diagnostic harnesses call
# or inject their torch references explicitly.
export TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR:-/dev/shm/tilelang_cache_v4_r6_full_loop}
export TILELANG_TMP_DIR=${TILELANG_TMP_DIR:-${TILELANG_CACHE_DIR}/tmp}
export TMPDIR=${TMPDIR:-${RUNTIME_CACHE_ROOT}/tmp}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-${RUNTIME_CACHE_ROOT}/xdg}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/torchinductor}
export CUDA_CACHE_PATH=${CUDA_CACHE_PATH:-${RUNTIME_CACHE_ROOT}/cuda}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}
export SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=${SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK:-false}
export SGLANG_MEMORY_SAVER_CUDA_GRAPH=${SGLANG_MEMORY_SAVER_CUDA_GRAPH:-true}
export SGLANG_DSV4_FP4_EXPERTS=${SGLANG_DSV4_FP4_EXPERTS:-${V4_FP4_FROZEN_EXPERTS}}
if [[ "${V4_FP4_FROZEN_EXPERTS}" == "1" && "${SGLANG_DSV4_FP4_EXPERTS}" != "1" ]]; then
  echo "FATAL: V4_FP4_FROZEN_EXPERTS=1 requires SGLANG_DSV4_FP4_EXPERTS=1 (rollout must serve the packed experts)." >&2
  exit 1
fi
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-256}
export SGLANG_OPT_USE_TILELANG_MHC_PRE=${SGLANG_OPT_USE_TILELANG_MHC_PRE:-true}
export SGLANG_OPT_USE_TILELANG_MHC_POST=${SGLANG_OPT_USE_TILELANG_MHC_POST:-true}
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=${SGLANG_OPT_DEEPGEMM_HC_PRENORM:-true}
export SGLANG_ROUTER_REGISTRATION_TIMEOUT_SECS
ulimit -n 1048576 || true

CLUSTER_NO_PROXY="127.0.0.1,localhost,0.0.0.0,::1,${HEAD_IP},${WORKER_IPS[*]},node64,node69,node70,node62,node53,node54,node64_slime,node69_slime,node70_slime,node62_slime,node53_slime,node54_slime"
CLUSTER_NO_PROXY=${CLUSTER_NO_PROXY// /,}
export no_proxy="${no_proxy:-},${CLUSTER_NO_PROXY}"
export NO_PROXY="${NO_PROXY:-},${CLUSTER_NO_PROXY}"

mkdir -p "${SCRATCH}" "${DEBUG_DIR}" "$(dirname "${LOG}")"
if [[ "${CLEAN_RUNTIME_CACHE:-0}" == "1" ]]; then
  rm -rf "${RUNTIME_CACHE_ROOT}"
fi
mkdir -p "${TMPDIR}" "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${CUDA_CACHE_PATH}"
if [[ "${CLEAN_DEBUG_DIR}" == "1" ]]; then
  rm -f "${DEBUG_DIR}"/rollout_*.pt "${DEBUG_DIR}"/train_*.pt
fi
cd "${REPO}"
RUN_LOCK=${RUN_LOCK:-${REPO}/local_artifacts/deepseek-v4/r2_logs/r6_full_loop.lock}
exec 9>"${RUN_LOCK}"
if ! flock -n 9; then
  echo "Another full-loop smoke is already running; lock=${RUN_LOCK}" | tee -a "${LOG}"
  exit 75
fi

# Stable, rarely-changing cluster infrastructure (preflight helpers, cleanup,
# unified env transport, ray bring-up). Functions only — sourced after the
# config above so the functions see this launcher's variables at call time.
# shellcheck source=scripts/dsv4/_dsv4_cluster_lib.sh
source "${REPO}/scripts/dsv4/_dsv4_cluster_lib.sh"
trap cleanup_ray_cluster EXIT

echo "=== R6 V4 full-loop smoke sanity ===" | tee "${LOG}"
echo "repo=${REPO}" | tee -a "${LOG}"
if [[ -n "${ENTROPY_AB_ARM:-}" ]]; then
  # Fixed-batch experiment provenance.  The arm launcher computes these on
  # every selected container before entering this common script; emit them
  # only after LOG has been opened because the first tee above truncates it.
  echo "entropy_ab_arm=${ENTROPY_AB_ARM}" | tee -a "${LOG}"
  echo "source_debug_data=${ENTROPY_AB_SOURCE_DEBUG_DATA}" | tee -a "${LOG}"
  echo "source_debug_data_sha256=${ENTROPY_AB_SOURCE_DEBUG_DATA_SHA256}" | tee -a "${LOG}"
  echo "debug_data=${ENTROPY_AB_DEBUG_DATA} subsample=${LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE}" | tee -a "${LOG}"
  echo "debug_data_sha256=${ENTROPY_AB_DEBUG_DATA_SHA256}" | tee -a "${LOG}"
  echo "git_revision=${ENTROPY_AB_GIT_REVISION}" | tee -a "${LOG}"
  echo "critical_code_sha256=${ENTROPY_AB_CRITICAL_CODE_SHA256}" | tee -a "${LOG}"
  echo "base_manifest_sha256=${ENTROPY_AB_BASE_MANIFEST_SHA256}" | tee -a "${LOG}"
  echo "megatron_revision=${ENTROPY_AB_MEGATRON_REVISION}" | tee -a "${LOG}"
  echo "tilekernels_revision=${ENTROPY_AB_TILEKERNELS_REVISION}" | tee -a "${LOG}"
  echo "resolved_contract=${ENTROPY_AB_CONTRACT}" | tee -a "${LOG}"
  echo "reduction=${ENTROPY_AB_REDUCTION}" | tee -a "${LOG}"
fi
echo "hf=${HF_CKPT}" | tee -a "${LOG}"
echo "load=${LOAD}" | tee -a "${LOG}"
echo "prompt_data=${PROMPT_DATA}" | tee -a "${LOG}"
echo "save=${SAVE} save_model=${SAVE_MODEL} async_save=${ASYNC_SAVE}" | tee -a "${LOG}"
echo "debug_dir=${DEBUG_DIR}" | tee -a "${LOG}"
echo "actor=${ACTOR_PHYSICAL_HOSTS[*]} rollout=${ROLLOUT_PHYSICAL_HOSTS[*]} rollout_gpus=${ROLLOUT_GPUS}" | tee -a "${LOG}"
echo "placement actor=${ACTOR_PLACEMENT_RESOURCE} rollout=${ROLLOUT_PLACEMENT_RESOURCE}" | tee -a "${LOG}"
echo "pp=${PP_SIZE} ep=${EP_SIZE} first_layers=${FIRST_LAYERS} last_layers=${LAST_LAYERS} data_pad_size_multiplier=${DATA_PAD_SIZE_MULTIPLIER}" | tee -a "${LOG}"
echo "global_batch_size=${GLOBAL_BATCH_SIZE} rollout_batch_size=${ROLLOUT_BATCH_SIZE} over_sampling_batch_size=${OVER_SAMPLING_BATCH_SIZE} over_sampling_refill_factor=${OVER_SAMPLING_REFILL_FACTOR:-legacy_fixed} num_rollout=${NUM_ROLLOUT}" | tee -a "${LOG}"
echo "start_rollout_id=${START_ROLLOUT_ID} clean_debug_dir=${CLEAN_DEBUG_DIR}" | tee -a "${LOG}"
echo "use_rollout_routing_replay=${USE_ROLLOUT_ROUTING_REPLAY}" | tee -a "${LOG}"
echo "use_sglang_deepep=${USE_SGLANG_DEEPEP} sglang_dp_size=${SGLANG_DP_SIZE}" | tee -a "${LOG}"
echo "v4_fp4_frozen_experts=${V4_FP4_FROZEN_EXPERTS} sglang_dsv4_fp4_experts=${SGLANG_DSV4_FP4_EXPERTS} moe_runner=$([[ "${V4_FP4_FROZEN_EXPERTS}" == "1" ]] && echo "${SGLANG_MOE_RUNNER_BACKEND:-flashinfer_mxfp4}" || echo auto)" | tee -a "${LOG}"
echo "sglang_enable_lora=${SGLANG_ENABLE_LORA} use_lora_weight_sync=${USE_LORA_WEIGHT_SYNC} max_lora_rank=${SGLANG_MAX_LORA_RANK} fuse_wqa_wkv=${SGLANG_OPT_FUSE_WQA_WKV:-1} lora_targets='${SGLANG_LORA_TARGET_MODULES}'" | tee -a "${LOG}"
echo "lora=r${LORA_DIM}/alpha${LORA_ALPHA}/dropout${LORA_DROPOUT}/rslora${LORA_RSLORA}/plus${LORA_PLUS_LAMBDA:-off}/shared${LORA_SHARED_EXPERT} adapter_resume=${LORA_ADAPTER_RESUME_LOAD:-none}" | tee -a "${LOG}"
echo "sglang_cuda_graph_max_bs=${SGLANG_CUDA_GRAPH_MAX_BS} sglang_disable_cuda_graph=${SGLANG_DISABLE_CUDA_GRAPH}" | tee -a "${LOG}"
echo "kernel_alignment=ds-v4-fixed" | tee -a "${LOG}"
echo "sglang_mhc_env pre=${SGLANG_OPT_USE_TILELANG_MHC_PRE} post=${SGLANG_OPT_USE_TILELANG_MHC_POST} deepgemm_prenorm=${SGLANG_OPT_DEEPGEMM_HC_PRENORM}" | tee -a "${LOG}"
echo "sglang_speculative=${SGLANG_SPECULATIVE_ALGORITHM:-off} steps=${SGLANG_SPECULATIVE_NUM_STEPS:-2} topk=${SGLANG_SPECULATIVE_EAGLE_TOPK:-1} draft=${SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS:-3}" | tee -a "${LOG}"
echo "gpu_idle_max_mib=1024 gpu_idle_wait_secs=120 gpu_idle_poll_secs=5 (fixed)" | tee -a "${LOG}"
echo "external_sglang_guard=${EXTERNAL_SGLANG_GUARD} seconds=${EXTERNAL_SGLANG_GUARD_SECS} poll=${EXTERNAL_SGLANG_GUARD_POLL_SECS}" | tee -a "${LOG}"
echo "ray_agent_register_timeout_ms=${RAY_agent_register_timeout_ms}" | tee -a "${LOG}"
echo "ray_system_config=${RAY_SYSTEM_CONFIG_JSON}" | tee -a "${LOG}"
echo "ray_agent_ports=http:${RAY_DASHBOARD_AGENT_LISTEN_PORT} grpc:${RAY_DASHBOARD_AGENT_GRPC_PORT} runtime_env:${RAY_RUNTIME_ENV_AGENT_PORT}" | tee -a "${LOG}"
echo "ray_run_mode=direct (fixed)" | tee -a "${LOG}"
echo "runtime_cache_root=${RUNTIME_CACHE_ROOT} tmpdir=${TMPDIR}" | tee -a "${LOG}"
echo "ray_dashboard_agent_early_port=${SLIME_RAY_DASHBOARD_AGENT_EARLY_PORT} patcher=${RAY_DASHBOARD_AGENT_PATCHER}" | tee -a "${LOG}"
echo "sequence_mis_config=${SEQUENCE_MIS_CONFIG:-<default>}" | tee -a "${LOG}"
# Fail fast on corrupted JSON (an inline ${VAR:-{...}} default upstream appends
# a stray brace even when VAR is set; killed the first two r21 launches at
# train_async arg-parse, AFTER the expensive Ray bring-up).
if [[ -n "${SEQUENCE_MIS_CONFIG}" ]]; then
  python3 -c 'import json,sys
v = sys.argv[1]
try:
    assert isinstance(json.loads(v), dict)
except Exception as e:
    sys.exit(f"FATAL: --sequence-mis-config is not a JSON object: {e}\nvalue={v!r}")' "${SEQUENCE_MIS_CONFIG}" 2>&1 | tee -a "${LOG}"
fi

check_actor_node "${HEAD_HOST}"
for host in "${TRAIN_WORKER_HOSTS[@]}"; do
  check_actor_node "${host}"
done
for host in "${ROLLOUT_WORKER_HOSTS[@]}"; do
  check_node "${host}"
done
prompt_lines=$(wc -l <"${PROMPT_DATA}")
if (( prompt_lines < ROLLOUT_BATCH_SIZE )); then
  echo "prompt_data has ${prompt_lines} lines, need at least rollout_batch_size=${ROLLOUT_BATCH_SIZE}" | tee -a "${LOG}"
  exit 1
fi

# Checkpoint-FAMILY preflight: the official mixed ckpt and the secondary FP8 ckpt
# have byte-identical config.json/index.json — only per-tensor dtypes distinguish
# them. Probe a routed-expert weight header and hard-fail on a mismatch with the
# selected expert mode (both directions).
expert_dtype=$(python3 "${REPO}/scripts/dsv4/probe_expert_dtype.py" "${HF_CKPT}") || {
  echo "FATAL: expert dtype probe failed for ${HF_CKPT} (see stderr above)" | tee -a "${LOG}" >&2
  exit 1
}
if [[ "${V4_FP4_FROZEN_EXPERTS}" == "1" && "${expert_dtype}" != "I8" && "${expert_dtype}" != "U8" ]]; then
  echo "FATAL: V4_FP4_FROZEN_EXPERTS=1 but HF_CKPT routed experts are ${expert_dtype} (need packed I8):" >&2
  echo "  ${HF_CKPT} is not the official mixed checkpoint." >&2
  exit 1
fi
if [[ "${V4_FP4_FROZEN_EXPERTS}" != "1" && ( "${expert_dtype}" == "I8" || "${expert_dtype}" == "U8" ) ]]; then
  echo "FATAL: HF_CKPT routed experts are packed ${expert_dtype} (official mixed checkpoint)" >&2
  echo "  but V4_FP4_FROZEN_EXPERTS!=1 — the fp8 stack would misread them." >&2
  exit 1
fi
echo "expert_dtype_probe=${expert_dtype} v4_fp4_frozen_experts=${V4_FP4_FROZEN_EXPERTS}" | tee -a "${LOG}"

DSV4_CHAT_TEMPLATE="${REPO}/examples/kernel_agent/prompt_config/deepseek_v4_chat_template.jinja"
if [[ ! -s "${HF_CKPT}/chat_template.jinja" && -f "${DSV4_CHAT_TEMPLATE}" ]]; then
  cp "${DSV4_CHAT_TEMPLATE}" "${HF_CKPT}/chat_template.jinja"
  echo "installed chat_template.jinja -> ${HF_CKPT}/chat_template.jinja" | tee -a "${LOG}"
fi

# A prepare-only launch performs all launcher, remote-node, dataset, checkpoint,
# frozen-dump, and code-identity sanity checks above, then exits before any Ray
# cleanup/start, process killing, GPU polling, or driver dispatch.  Remove the
# EXIT trap as an additional guarantee that even cleanup_ray_cluster is not run.
if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "=== V4 prepare-only sanity PASS (no Ray cleanup/start, GPU work, or driver dispatch) ===" | tee -a "${LOG}"
  trap - EXIT
  exit 0
fi

if [[ "${CLEAN_RAY:-1}" == "1" ]]; then
  echo "=== stopping old Ray on selected nodes ===" | tee -a "${LOG}"
  ray stop --force >/dev/null 2>&1 || true
  for host in "${WORKER_HOSTS[@]}"; do
    ssh ${SSH_OPTS} "${host}" "ray stop --force >/dev/null 2>&1 || true" &
  done
  wait
fi
if [[ "${KILL_OLD_PROCESSES:-1}" == "1" ]]; then
  echo "=== killing old sglang/Ray actors on selected nodes ===" | tee -a "${LOG}"
  kill_old_processes
fi
if [[ "${EXTERNAL_SGLANG_GUARD}" == "1" ]]; then
  echo "=== starting external SGLang serve guard on physical clean hosts ===" | tee -a "${LOG}"
  start_external_sglang_guard
fi
if [[ "${CLEAN_TILELANG_CACHE:-0}" == "1" ]]; then
  echo "=== cleaning TileLang cache on actor nodes: ${TILELANG_CACHE_DIR} ===" | tee -a "${LOG}"
  rm -rf "${TILELANG_CACHE_DIR}" && mkdir -p "${TILELANG_TMP_DIR}"
  cache_clean_pids=()
  for host in "${TRAIN_WORKER_HOSTS[@]}"; do
    ssh ${SSH_OPTS} "${host}" "rm -rf ${TILELANG_CACHE_DIR} && mkdir -p ${TILELANG_TMP_DIR}" &
    cache_clean_pids+=("$!")
  done
  for pid in "${cache_clean_pids[@]}"; do
    wait "${pid}"
  done
else
  mkdir -p "${TILELANG_TMP_DIR}"
fi

echo "=== checking selected GPUs are idle ===" | tee -a "${LOG}"
wait_gpu_idle | tee -a "${LOG}"

if [[ "${SLIME_RAY_DASHBOARD_AGENT_EARLY_PORT}" == "1" ]]; then
  echo "=== patching Ray dashboard agent early-port hook ===" | tee -a "${LOG}"
  patch_ray_dashboard_agent "${HEAD_HOST}" | tee -a "${LOG}"
  patch_pids=()
  for host in "${WORKER_HOSTS[@]}"; do
    patch_ray_dashboard_agent "${host}" | tee -a "${LOG}" &
    patch_pids+=("$!")
  done
  patch_rc=0
  for pid in "${patch_pids[@]}"; do
    if ! wait "${pid}"; then
      patch_rc=1
    fi
  done
  if (( patch_rc != 0 )); then
    echo "Ray dashboard agent patch failed on at least one worker" | tee -a "${LOG}"
    exit 1
  fi
fi

ACTOR_RESOURCE_JSON="{\"${ACTOR_PLACEMENT_RESOURCE}\": ${ACTOR_GPUS_PER_NODE}}"
# PER-NODE rollout GPU count: ROLLOUT_GPUS is the TOTAL across all rollout
# hosts. Passing the total as each worker's --num-gpus made ray believe one
# node had 16 GPUs — all 16 bundles landed on node70 and engine rank-1 died
# with "CUDA error: invalid device ordinal" at base_gpu_id=8 (bit formal r9j
# 2026-07-11, first 2-rollout-node run).
ROLLOUT_GPUS_PER_NODE=$(( ROLLOUT_GPUS / ${#ROLLOUT_WORKER_HOSTS[@]} ))
if (( ROLLOUT_GPUS_PER_NODE * ${#ROLLOUT_WORKER_HOSTS[@]} != ROLLOUT_GPUS )); then
  echo "FATAL: ROLLOUT_GPUS=${ROLLOUT_GPUS} not divisible by ${#ROLLOUT_WORKER_HOSTS[@]} rollout hosts" >&2
  exit 1
fi
if (( ROLLOUT_GPUS_PER_NODE > 8 )); then
  echo "FATAL: ROLLOUT_GPUS_PER_NODE=${ROLLOUT_GPUS_PER_NODE} exceeds the 8 GPUs per H20 node" >&2
  exit 1
fi
ROLLOUT_RESOURCE_JSON="{\"${ROLLOUT_PLACEMENT_RESOURCE}\": ${ROLLOUT_GPUS_PER_NODE}}"

# Ray head + worker bring-up (lib). Env transport to the head/worker ray
# processes comes from dsv4_transport_env_pairs (see _dsv4_cluster_lib.sh).
start_ray_head
start_ray_workers

echo "=== waiting for Ray cluster ===" | tee -a "${LOG}"
deadline=$((SECONDS + RAY_WAIT_TIMEOUT))
while true; do
  if python - "${RAY_HEAD_ADDR}" "$((1 + ${#WORKER_HOSTS[@]}))" "${ACTOR_PLACEMENT_RESOURCE}" "$((ACTOR_NUM_NODES * ACTOR_GPUS_PER_NODE))" "${ROLLOUT_PLACEMENT_RESOURCE}" "${ROLLOUT_GPUS}" <<'PY'
import sys
import ray

addr = sys.argv[1]
nodes = int(sys.argv[2])
actor_resource = sys.argv[3]
actor_need = float(sys.argv[4])
rollout_resource = sys.argv[5]
rollout_need = float(sys.argv[6])
ray.init(address=addr, ignore_reinit_error=True)
alive = [n for n in ray.nodes() if n.get("Alive")]
resources = {}
for node in alive:
    for key, value in node.get("Resources", {}).items():
        resources[key] = resources.get(key, 0.0) + float(value)
ok = (
    len(alive) >= nodes
    and resources.get(actor_resource, 0.0) >= actor_need
    and resources.get(rollout_resource, 0.0) >= rollout_need
)
ray.shutdown()
raise SystemExit(0 if ok else 1)
PY
  then
    break
  fi
  if (( SECONDS > deadline )); then
    ray status --address "${RAY_HEAD_ADDR}" | tee -a "${LOG}" || true
    echo "Ray cluster did not become ready within ${RAY_WAIT_TIMEOUT}s" | tee -a "${LOG}"
    exit 1
  fi
  sleep 5
done
ray status --address "${RAY_HEAD_ADDR}" | tee -a "${LOG}"

if [[ "${RECHECK_GPU_IDLE_BEFORE_SUBMIT:-1}" == "1" ]]; then
  echo "=== rechecking GPUs before job submit ===" | tee -a "${LOG}"
  if [[ "${KILL_OLD_PROCESSES_BEFORE_SUBMIT:-0}" == "1" ]]; then
    kill_old_processes
  fi
  wait_gpu_idle | tee -a "${LOG}"
fi

MODEL_ARGS=(
  --num-layers 43
  --hidden-size 4096
  --ffn-hidden-size 2048
  --moe-ffn-hidden-size 2048
  --num-experts 256
  --num-attention-heads 64
  --kv-channels 512
  --vocab-size 129280
  --seq-length 64
  --max-position-embeddings 1048576
  --untie-embeddings-and-output-weights
  --rotary-base 10000
  --disable-bias-linear
  --normalization RMSNorm
  --norm-epsilon 1e-6
)

# Uneven-PP layer split only applies at PP>1; megatron rejects the
# decoder-first/last args at PP1 (the DP2 variant).
UNEVEN_PP_ARGS=()
if [[ "${PP_SIZE}" -gt 1 ]]; then
  UNEVEN_PP_ARGS=(
    --decoder-first-pipeline-num-layers "${FIRST_LAYERS}"
    --decoder-last-pipeline-num-layers "${LAST_LAYERS}"
  )
fi

# Contiguous CP (V4-Flash) only when CP_SIZE>1; strict no-op / bit-identical at CP_SIZE=1.
CP_ARGS=()
if [[ "${CP_SIZE}" -gt 1 ]]; then
  CP_ARGS=(--cp-partition-mode contiguous)
fi

COMMON_ARGS=(
  "${UNEVEN_PP_ARGS[@]}"
  "${CP_ARGS[@]}"
  --custom-model-provider-path custom_kernels.deepseek_v4.megatron.model_provider.v4_model_provider
  --hf-checkpoint "${HF_CKPT}"
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size "${PP_SIZE}"
  --context-parallel-size "${CP_SIZE}"
  --expert-model-parallel-size "${EP_SIZE}"
  --expert-tensor-parallel-size 1
  --moe-token-dispatcher-type flex
  --moe-flex-dispatcher-backend deepep
  --moe-router-dtype fp32
  --moe-deepep-num-sms 20
  --bf16
  --qkv-format bshd
  --lora-dim "${LORA_DIM}"
  --lora-alpha "${LORA_ALPHA}"
  --lora-dropout "${LORA_DROPOUT}"
  --lora-checkpoint-max-node-bytes "${LORA_CHECKPOINT_MAX_NODE_BYTES}"
  --data-pad-size-multiplier "${DATA_PAD_SIZE_MULTIPLIER}"
  --micro-batch-size 1
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --optimizer muon
  --lr "${LR}"
  --lr-decay-style "${LR_DECAY_STYLE}"
  --weight-decay "${WEIGHT_DECAY}"
  "${MUON_ARGS[@]}"
  --accumulate-allreduce-grads-in-fp32
  --ckpt-format torch_dist
  # PG timeout: default 10min killed formal runs — at formal scale the slowest
  # DP replica's first log-prob microbatches (TileLang JIT warmup at the new
  # padded shape, 120-250s/mb observed) can exceed 600s, watchdog SIGABRTs a
  # peer, and survivors see "remote process exited" (2026-07-04 v1+v2 failures).
  --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES:-120}"
)
if [[ "${LORA_RSLORA}" == "1" ]]; then
  COMMON_ARGS+=(--lora-rslora)
else
  COMMON_ARGS+=(--no-lora-rslora)
fi
if [[ -n "${LORA_PLUS_LAMBDA}" ]]; then
  COMMON_ARGS+=(--lora-plus-lambda "${LORA_PLUS_LAMBDA}")
fi
if [[ "${LORA_SHARED_EXPERT}" == "1" ]]; then
  COMMON_ARGS+=(--dsv4-lora-shared-expert)
else
  COMMON_ARGS+=(--no-dsv4-lora-shared-expert)
fi
if [[ -n "${LORA_ADAPTER_RESUME_LOAD}" ]]; then
  COMMON_ARGS+=(--lora-adapter-resume-load "${LORA_ADAPTER_RESUME_LOAD}")
fi
# Adapter resumes that EXTEND the training horizon (NUM_ROLLOUT beyond the
# saved run's) need Megatron's scheduler override, else:
# "OptimizerParamScheduler: class input value X and checkpoint value Y ...
# do not match" (same as train_smoke.sh; RUNTIME.md scheduler-horizon note).
if [[ "${OVERRIDE_OPT_PARAM_SCHEDULER:-0}" == "1" ]]; then
  COMMON_ARGS+=(--override-opt-param-scheduler)
fi

# Activation checkpointing via Megatron's standard flags. Megatron's recompute
# IMPLEMENTATION still can't run for the V4 custom model (hand-written layer
# loop, no TransformerBlock) — instead v4_model_provider reads
# --recompute-granularity full and applies the requested V4 decoder method in
# torch.utils.checkpoint: uniform uses N-layer segments; block checkpoints the
# first K local layers. Reentrant checkpointing is the sole validated path; see
# mcore_model.py for the pytorch#147449 rationale.
RECOMPUTE_ARGS=()
if [[ "${RECOMPUTE}" == "1" ]]; then
  # RECOMPUTE_METHOD=block checkpoints only the FIRST RECOMPUTE_NUM_LAYERS
  # layers per stage and stores the rest (memory-for-speed dial; V4 decoder
  # loop implements both methods — mcore_model._forward_v4_decoder_layers).
  RECOMPUTE_ARGS=(--recompute-granularity full --recompute-method "${RECOMPUTE_METHOD:-uniform}" --recompute-num-layers "${RECOMPUTE_NUM_LAYERS:-1}")
fi

SGLANG_ARGS=(
  --rollout-num-gpus "${ROLLOUT_GPUS}"
  --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}"
  --sglang-context-length "${SGLANG_CONTEXT_LENGTH}"
  --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
  --sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --sglang-kv-cache-dtype "${SGLANG_KV_CACHE_DTYPE}"
  --sglang-chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE}"
  --sglang-max-prefill-tokens "${SGLANG_MAX_PREFILL_TOKENS}"
  --sglang-disable-custom-all-reduce
  --sglang-watchdog-timeout 2400
  --sglang-decode-log-interval 40
  --router-policy round_robin
  --router-queue-timeout-secs 2400
)
if [[ "${SGLANG_DISABLE_CUDA_GRAPH}" == "1" ]]; then
  SGLANG_ARGS+=(--sglang-disable-cuda-graph)
fi
# NEXTN speculative decoding with the checkpoint's MTP head (rollout throughput).
# Chain mode (eagle-topk=1) requires num-draft-tokens == num-steps + 1. Requires
# the routed-experts capturer spec-decode fix when routing replay is on.
if [[ "${SGLANG_SPECULATIVE_ALGORITHM:-}" == "none" ]]; then
  # 'none' = explicit disable that survives run.t1's `:-EAGLE` default expansion.
  SGLANG_SPECULATIVE_ALGORITHM=""
fi
if [[ "${SGLANG_SPECULATIVE_ALGORITHM:-}" == "DSPARK" ]]; then
  # DSPARK (new runtime): draft config auto-infers from the -DSpark ckpt unless
  # SGLANG_SPECULATIVE_DSPARK_BLOCK_SIZE overrides gamma. Do NOT pass
  # EAGLE-style steps/topk/draft-tokens. dp-lm-head is a hard requirement under
  # dp-attention (arg_groups/speculative_hook.py:_handle_dspark).
  SGLANG_ARGS+=(
    --sglang-speculative-algorithm DSPARK
    --sglang-enable-dp-lm-head
  )
  if [[ -n "${SGLANG_SPECULATIVE_DSPARK_BLOCK_SIZE:-}" ]]; then
    if [[ ! "${SGLANG_SPECULATIVE_DSPARK_BLOCK_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
      echo "FATAL: SGLANG_SPECULATIVE_DSPARK_BLOCK_SIZE must be a positive integer, got '${SGLANG_SPECULATIVE_DSPARK_BLOCK_SIZE}'." >&2
      exit 2
    fi
    SGLANG_ARGS+=(
      --sglang-speculative-dspark-block-size "${SGLANG_SPECULATIVE_DSPARK_BLOCK_SIZE}"
    )
  fi
  if [[ -z "${V4_ROLLOUT_MODEL_PATH:-}" ]]; then
    echo "FATAL: SGLANG_SPECULATIVE_ALGORITHM=DSPARK requires V4_ROLLOUT_MODEL_PATH (the -DSpark ckpt variant with mtp.* draft stages)." >&2
    exit 1
  fi
elif [[ -n "${SGLANG_SPECULATIVE_ALGORITHM:-}" ]]; then
  SGLANG_ARGS+=(
    --sglang-speculative-algorithm "${SGLANG_SPECULATIVE_ALGORITHM}"
    --sglang-speculative-num-steps "${SGLANG_SPECULATIVE_NUM_STEPS:-2}"
    --sglang-speculative-eagle-topk "${SGLANG_SPECULATIVE_EAGLE_TOPK:-1}"
    --sglang-speculative-num-draft-tokens "${SGLANG_SPECULATIVE_NUM_DRAFT_TOKENS:-3}"
  )
fi
if [[ -n "${V4_ROLLOUT_MODEL_PATH:-}" ]]; then
  # Rollout serves a different ckpt than the trainer (e.g. -DSpark variant);
  # weight-sync/tokenizer stay on --hf-checkpoint.
  SGLANG_ARGS+=(--rollout-model-path "${V4_ROLLOUT_MODEL_PATH}")
fi
if [[ "${V4_FP4_FROZEN_EXPERTS}" == "1" ]]; then
  # W4A16 serving of the official packed experts: the SM90 runners register only
  # the standard a2a=none dispatch (deepep+fp4 would silently select the DeepGEMM
  # W4A8 path — wrong compute class), and 'auto' runner falls into Fp8MoEMethod.
  # Keep dp-attention (LoRA needs attn-tp==1); MoE runs TP-per-engine.
  if [[ "${USE_SGLANG_DEEPEP}" == "1" ]]; then
    echo "FATAL: V4_FP4_FROZEN_EXPERTS=1 requires USE_SGLANG_DEEPEP=0 (W4A16 runners are a2a=none only)." >&2
    exit 1
  fi
  SGLANG_ARGS+=(
    --sglang-data-parallel-size "${SGLANG_DP_SIZE}"
    --sglang-enable-dp-attention
    --sglang-moe-a2a-backend none
    --sglang-moe-runner-backend "${SGLANG_MOE_RUNNER_BACKEND:-flashinfer_mxfp4}"
    --sglang-disable-flashinfer-autotune
  )
elif [[ "${USE_SGLANG_DEEPEP}" == "1" ]]; then
  SGLANG_ARGS+=(
    --sglang-data-parallel-size "${SGLANG_DP_SIZE}"
    --sglang-enable-dp-attention
    --sglang-moe-a2a-backend deepep
    --sglang-deepep-config "${SGLANG_DEEPEP_CONFIG}"
  )
elif [[ "${SGLANG_ENABLE_DP_ATTENTION:-0}" == "1" ]]; then
  # dp-attention WITHOUT DeepEP: LoRA serving needs attn-tp==1. NOTE the old
  # 'two instances' theory was falsified — DeepEP dies on H20 whenever
  # num_sms > 78 (H20 SM count; default 96 is an H800 value). With
  # SGLANG_DEEPEP_CONFIG num_sms<=64, DeepEP works and this branch is optional.
  SGLANG_ARGS+=(
    --sglang-data-parallel-size "${SGLANG_DP_SIZE}"
    --sglang-enable-dp-attention
  )
fi
# LoRA-adapter serving (base fp8 + hot-swapped adapter). CUDA graph stays on
# (this fork supports LoRA + cuda-graph; only piecewise/torch.compile graph is
# auto-disabled). --sglang-lora-target-modules is nargs="*" → pass unquoted.
if [[ "${SGLANG_ENABLE_LORA}" == "1" ]]; then
  # shellcheck disable=SC2206
  SGLANG_ARGS+=(
    --sglang-enable-lora
    --sglang-max-lora-rank "${SGLANG_MAX_LORA_RANK}"
    --sglang-max-loras-per-batch "${SGLANG_MAX_LORAS_PER_BATCH}"
    --sglang-lora-target-modules ${SGLANG_LORA_TARGET_MODULES}
    # WORKAROUND (2026-07-08): the default csgmv LoRA backend has a cuda-graph-
    # visible metadata mutation bug on adapter load/reload (illegal access on
    # replay; bisected via scripts/dsv4/diagnostics/lora/lora_reload_repro.sh — triton PASSES,
    # csgmv crashes). Keep triton until csgmv is fixed. NOT a root-cause fix.
    --sglang-lora-backend "${SGLANG_LORA_BACKEND:-triton}"
  )
fi

SAVE_ARGS=()
if [[ "${ASYNC_SAVE}" != "0" && "${ASYNC_SAVE}" != "1" ]]; then
  echo "FATAL: ASYNC_SAVE must be 0 or 1, got '${ASYNC_SAVE}'." >&2
  exit 2
fi
if [[ "${SAVE_MODEL}" == "1" ]]; then
  SAVE_ARGS=(
    --save "${SAVE}"
    --save-interval "${SAVE_INTERVAL}"
  )
  # Optimizer and RNG persistence are separate: adapter-only checkpoints can
  # retain RNG state without Muon momentum, while exact resume enables both.
  if [[ "${SAVE_OPTIM}" != "1" ]]; then
    SAVE_ARGS+=(--no-save-optim)
  fi
  if [[ "${SAVE_RNG}" != "1" ]]; then
    SAVE_ARGS+=(--no-save-rng)
  fi
  if [[ "${ASYNC_SAVE}" == "1" ]]; then
    SAVE_ARGS+=(--async-save --use-persistent-ckpt-worker)
  fi
fi

# The base torch_dist checkpoint has no optimizer/RNG state, so a cold start
# skips both. A resume can independently restore Muon momentum and RNG state.
LOAD_ARGS=(--load "${LOAD}")
if [[ "${LOAD_OPTIM}" != "1" ]]; then
  LOAD_ARGS+=(--no-load-optim)
fi
if [[ "${LOAD_RNG}" != "1" ]]; then
  LOAD_ARGS+=(--no-load-rng)
fi

# Debug flags for fast train-side iteration (skip the ~30min rollout):
#   capture once: SAVE_DEBUG_ROLLOUT_DATA=<path/{rollout_id}.pt> DEBUG_ROLLOUT_ONLY=1
#   then iterate: DEBUG_TRAIN_ONLY=1 LOAD_DEBUG_ROLLOUT_DATA=<path/{rollout_id}.pt>
# (slime: --debug-train-only / --load-debug-rollout-data imply skip_sglang.)
# To replay a captured batch while retaining SGLang and real LoRA hot-swap, set
# LOAD_FORGE_ROLLOUT_DATA=<path>. _dsv4_task_args.sh wires the dedicated forge
# rollout function and rejects combinations with the train/rollout-only modes.
DEBUG_ARGS=()
[[ -n "${SAVE_DEBUG_ROLLOUT_DATA:-}" ]] && DEBUG_ARGS+=(--save-debug-rollout-data "${SAVE_DEBUG_ROLLOUT_DATA}")
[[ -n "${LOAD_DEBUG_ROLLOUT_DATA:-}" ]] && DEBUG_ARGS+=(--load-debug-rollout-data "${LOAD_DEBUG_ROLLOUT_DATA}")
[[ -n "${LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE:-}" ]] && DEBUG_ARGS+=(
  --load-debug-rollout-data-subsample "${LOAD_DEBUG_ROLLOUT_DATA_SUBSAMPLE}"
)
[[ "${DEBUG_FREEZE_OLD_ACTOR_SNAPSHOT:-0}" == "1" ]] && DEBUG_ARGS+=(
  --debug-freeze-old-actor-snapshot
)
[[ "${DEBUG_FORCE_OLD_ACTOR_LOGPROB_RECOMPUTE:-0}" == "1" ]] && DEBUG_ARGS+=(
  --debug-force-old-actor-logprob-recompute
)
[[ "${DEBUG_ROLLOUT_ONLY:-0}" == "1" ]] && DEBUG_ARGS+=(--debug-rollout-only)
[[ "${DEBUG_TRAIN_ONLY:-0}" == "1" ]] && DEBUG_ARGS+=(--debug-train-only)

# --- Task-specific arg assembly (smoke_sft vs rl) --------------------------
# Assembly lives in a sourceable helper so it can be unit-tested without the
# cluster bring-up (tests/deepseek-v4/test_dsv4_rl_task_args.py). Sets TASK_ARGS.
# shellcheck source=scripts/dsv4/_dsv4_task_args.sh
source "${REPO}/scripts/dsv4/_dsv4_task_args.sh"
build_dsv4_task_args

# The train-actor environment comes from the unified transport in
# _dsv4_cluster_lib.sh
# (dsv4_transport_env_pairs: explicit core list + V4_*/SGLANG_*/SLIME_DEBUG_*/
# SLIME_PATCH_* prefix sweep over exported vars) — a value-identical superset
# of the four hand-maintained key lists this replaced.
TRAIN_ENV_VARS_JSON=$(dsv4_build_train_env_vars_json)

# Train-side LoRA-adapter sync flag (slime arg, not an sglang server arg). When
# set, the weight-sync pushes only the trainable adapter to the engines instead
# of the merged full weight. Gated + validated against SGLANG_ENABLE_LORA above.
LORA_SYNC_ARGS=()
if [[ "${USE_LORA_WEIGHT_SYNC}" == "1" ]]; then
  LORA_SYNC_ARGS+=(--use-lora-weight-sync)
fi

OVER_SAMPLING_REFILL_ARGS=()
if [[ -n "${OVER_SAMPLING_REFILL_FACTOR}" ]]; then
  OVER_SAMPLING_REFILL_ARGS+=(--over-sampling-refill-factor "${OVER_SAMPLING_REFILL_FACTOR}")
fi

# TRAIN_SCRIPT=train_async.py enables one-step-ahead pipelining: generate(N+1)
# overlaps train(N) (1-step off-policy; the MIS band absorbs the staleness, and
# the alternating LoRA adapter swap already tolerates in-flight requests).
# update_weights_interval=1 drains the pipeline at each weight update, so
# staleness never exceeds one step.
TRAIN_CMD=(
  python3 "${TRAIN_SCRIPT:-train.py}"
  --actor-num-nodes "${ACTOR_NUM_NODES}"
  --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}"
  --num-gpus-per-node 8
  --actor-placement-resource "${ACTOR_PLACEMENT_RESOURCE}"
  --rollout-placement-resource "${ROLLOUT_PLACEMENT_RESOURCE}"
  --train-env-vars "${TRAIN_ENV_VARS_JSON}"
  "${MODEL_ARGS[@]}" "${COMMON_ARGS[@]}" "${RECOMPUTE_ARGS[@]}"
  "${LOAD_ARGS[@]}"
  "${SAVE_ARGS[@]}"
  "${SGLANG_ARGS[@]}"
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --over-sampling-batch-size "${OVER_SAMPLING_BATCH_SIZE}"
  "${OVER_SAMPLING_REFILL_ARGS[@]}"
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --num-rollout "${NUM_ROLLOUT}"
  --start-rollout-id "${START_ROLLOUT_ID}"
  "${DEBUG_ARGS[@]}"
  "${TASK_ARGS[@]}"
  "${LORA_SYNC_ARGS[@]}"
)

run_direct_driver() {
  echo "=== running full-loop direct driver ===" | tee -a "${LOG}"
  RAY_ADDRESS="${RAY_HEAD_ADDR}" "${TRAIN_CMD[@]}" 2>&1 | tee -a "${LOG}"
  return "${PIPESTATUS[0]}"
}

if ! run_direct_driver; then
  echo "Direct driver full-loop smoke failed" | tee -a "${LOG}"
  exit 1
fi
if [[ "${TASK_MODE}" == "smoke_sft" ]]; then
  python3 scripts/dsv4/verify_rollout_dump.py "${DEBUG_DIR}/rollout_0.pt" --expected-samples "$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))" | tee -a "${LOG}"
  echo "=== R6 V4 full-loop smoke PASS ===" | tee -a "${LOG}"
else
  echo "=== V4 ${TASK_MODE} (${REWARD_MODE}) run finished OK ===" | tee -a "${LOG}"
fi
