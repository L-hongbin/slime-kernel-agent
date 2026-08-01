#!/bin/bash
# R6 full-loop smoke for DeepSeek-V4-Flash:
# node64+node69 Megatron PP2/EP8 actor train, node62 SGLang TP4 rollout.
#
# This runs one small rollout+SFT train iteration with rollout routing replay,
# Megatron DeepEP, Megatron Muon, and a converted torch_dist checkpoint.
set -euo pipefail

REPO=${REPO:-/nfs/FM/chenshuailin/projects/kernel_agents/slime-v4flash-lora}
HF_CKPT=${HF_CKPT:-/nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8}
LOAD=${LOAD:-/nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8-r2-pp2-ep8-torch_dist}
PROMPT_DATA=${PROMPT_DATA:-${REPO}/Data/v4_full_loop_smoke.jsonl}
SCRATCH=${SCRATCH:-/nfs/FM/csl_v4r6_full_loop}
SAVE=${SAVE:-${SCRATCH}/out}
DEBUG_DIR=${DEBUG_DIR:-${SCRATCH}/debug}
RUN_ID=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
LOG=${LOG:-${REPO}/handoffs/deepseek-v4/r2_logs/r6_pp2_ep8_full_loop_${RUN_ID}.log}

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
PHYSICAL_CLEAN_HOSTS=(${PHYSICAL_CLEAN_HOSTS:-node64 node69 node70 node62})

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
MOE_ROUTER_TOPK=${MOE_ROUTER_TOPK:-6}
FIRST_LAYERS=${FIRST_LAYERS:-21}
LAST_LAYERS=${LAST_LAYERS:-22}
SAVE_MODEL=${SAVE_MODEL:-0}
SAVE_INTERVAL=${SAVE_INTERVAL:-100000}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
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

RAY_PORT=${RAY_PORT:-6396}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8280}
RAY_DASHBOARD_AGENT_LISTEN_PORT=${RAY_DASHBOARD_AGENT_LISTEN_PORT:-52365}
RAY_DASHBOARD_AGENT_GRPC_PORT=${RAY_DASHBOARD_AGENT_GRPC_PORT:-52366}
RAY_RUNTIME_ENV_AGENT_PORT=${RAY_RUNTIME_ENV_AGENT_PORT:-52367}
RAY_HEAD_ADDR="${HEAD_IP}:${RAY_PORT}"
RAY_WAIT_TIMEOUT=${RAY_WAIT_TIMEOUT:-300}
RAY_DASHBOARD_WAIT_TIMEOUT=${RAY_DASHBOARD_WAIT_TIMEOUT:-180}
RAY_RUN_MODE=${RAY_RUN_MODE:-direct}
RAY_JOB_STATUS_TIMEOUT=${RAY_JOB_STATUS_TIMEOUT:-14400}
RAY_JOB_STATUS_POLL_SECS=${RAY_JOB_STATUS_POLL_SECS:-15}
RAY_JOB_STATUS_MAX_FAILURES=${RAY_JOB_STATUS_MAX_FAILURES:-8}
RAY_JOB_SUBMIT_RETRIES=${RAY_JOB_SUBMIT_RETRIES:-5}
RAY_JOB_SUBMIT_RETRY_SECS=${RAY_JOB_SUBMIT_RETRY_SECS:-10}
RAY_JOB_LOG_POLL_SECS=${RAY_JOB_LOG_POLL_SECS:-60}
RAY_JOB_LOG_TAIL_LINES=${RAY_JOB_LOG_TAIL_LINES:-200}
RAY_TMP_ROOT=${RAY_TMP_ROOT:-/dev/shm/v4r6_full_loop_ray}
RUNTIME_CACHE_ROOT=${RUNTIME_CACHE_ROOT:-/dev/shm/v4r6_full_loop_cache}
RAY_OBJECT_STORE_MEMORY=${RAY_OBJECT_STORE_MEMORY:-20000000000}
RAY_DASHBOARD_AGENT_PATCHER=${RAY_DASHBOARD_AGENT_PATCHER:-${REPO}/scripts/patch_ray_dashboard_agent_early_port.py}
SLIME_RAY_DASHBOARD_AGENT_EARLY_PORT=${SLIME_RAY_DASHBOARD_AGENT_EARLY_PORT:-1}
SGLANG_MHC_PATCHER=${SGLANG_MHC_PATCHER:-${REPO}/scripts/v4/patch_sglang_dsv4_mhc_sinkhorn_torch.py}
SLIME_PATCH_SGLANG_DSV4_MHC_SINKHORN_TORCH=${SLIME_PATCH_SGLANG_DSV4_MHC_SINKHORN_TORCH:-0}
SSH_OPTS=${SSH_OPTS:--o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null}
PHYSICAL_SSH_OPTS=${PHYSICAL_SSH_OPTS:-${SSH_OPTS}}
GPU_IDLE_MAX_MIB=${GPU_IDLE_MAX_MIB:-1024}
GPU_IDLE_WAIT_SECS=${GPU_IDLE_WAIT_SECS:-120}
GPU_IDLE_POLL_SECS=${GPU_IDLE_POLL_SECS:-5}
CLEANUP_RAY_ON_EXIT=${CLEANUP_RAY_ON_EXIT:-1}
EXTERNAL_SGLANG_GUARD=${EXTERNAL_SGLANG_GUARD:-1}
EXTERNAL_SGLANG_GUARD_SECS=${EXTERNAL_SGLANG_GUARD_SECS:-14400}
EXTERNAL_SGLANG_GUARD_POLL_SECS=${EXTERNAL_SGLANG_GUARD_POLL_SECS:-10}
EXTERNAL_SGLANG_GUARD_FILE=${EXTERNAL_SGLANG_GUARD_FILE:-/tmp/slime_external_sglang_guard_${RUN_ID}.alive}
RAY_STARTED=0
EXTERNAL_SGLANG_GUARD_PIDS=()

SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.8}
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
SGLANG_DEEPEP_CONFIG=${SGLANG_DEEPEP_CONFIG:-'{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'}

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
export V4_LORA_DIM=${V4_LORA_DIM:-4}
export V4_LORA_ALPHA=${V4_LORA_ALPHA:-8}
export V4_LORA_DROPOUT=${V4_LORA_DROPOUT:-0.0}
# Adapter-only (LoRA) checkpointing: save/load only the tiny adapter params
# (frozen base reloads cold from --load). V4_LORA_ADAPTER_RESUME_LOAD=<adapter
# ckpt dir> overlays saved adapters + Muon optim on top of the base at startup.
export V4_LORA_ADAPTER_ONLY_CKPT=${V4_LORA_ADAPTER_ONLY_CKPT:-0}
export V4_LORA_ADAPTER_RESUME_LOAD=${V4_LORA_ADAPTER_RESUME_LOAD:-}
# Activation checkpointing in the V4 decoder loop (recompute in backward).
# Required at formal scale: without it the train backward OOMs (1F1B holds
# pp_size microbatches of activations). Default ON for RL, OFF for the tiny smoke.
export V4_ACT_CKPT=${V4_ACT_CKPT:-$([[ "${TASK_MODE}" == "rl" ]] && echo 1 || echo 0)}
export V4_FP8_FROZEN_EXPERTS=${V4_FP8_FROZEN_EXPERTS:-$([[ "${TASK_MODE}" == "rl" ]] && echo 1 || echo 0)}
export V4_FP8_EXPERT_GEMM=${V4_FP8_EXPERT_GEMM:-$([[ "${TASK_MODE}" == "rl" ]] && echo 1 || echo 0)}
export V4_FP8_SHARED_EXPERT=${V4_FP8_SHARED_EXPERT:-$([[ "${TASK_MODE}" == "rl" ]] && echo 1 || echo 0)}
export V4_FP8_ATTENTION=${V4_FP8_ATTENTION:-$([[ "${TASK_MODE}" == "rl" ]] && echo 1 || echo 0)}
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
export SGLANG_DSV4_FP4_EXPERTS=${SGLANG_DSV4_FP4_EXPERTS:-0}
export SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK=${SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-256}
export SGLANG_OPT_USE_TILELANG_MHC_PRE=${SGLANG_OPT_USE_TILELANG_MHC_PRE:-true}
export SGLANG_OPT_USE_TILELANG_MHC_POST=${SGLANG_OPT_USE_TILELANG_MHC_POST:-true}
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=${SGLANG_OPT_DEEPGEMM_HC_PRENORM:-true}
export SGLANG_OPT_USE_TILELANG_MHC_SPLIT_SINKHORN=${SGLANG_OPT_USE_TILELANG_MHC_SPLIT_SINKHORN:-true}
export SGLANG_ROUTER_REGISTRATION_TIMEOUT_SECS
ulimit -n 1048576 || true

CLUSTER_NO_PROXY="127.0.0.1,localhost,0.0.0.0,::1,${HEAD_IP},${WORKER_IPS[*]},node64,node69,node70,node62,node64_slime,node69_slime,node70_slime,node62_slime"
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
RUN_LOCK=${RUN_LOCK:-${REPO}/handoffs/deepseek-v4/r2_logs/r6_full_loop.lock}
exec 9>"${RUN_LOCK}"
if ! flock -n 9; then
  echo "Another full-loop smoke is already running; lock=${RUN_LOCK}" | tee -a "${LOG}"
  exit 75
fi

remote() {
  local host=$1
  shift
  if [[ "${host}" == "${HEAD_HOST}" ]]; then
    "$@"
  else
    ssh ${SSH_OPTS} "${host}" "$*"
  fi
}

check_node() {
  local host=$1
  remote "${host}" test -d "${REPO}"
  remote "${host}" test -f "${RAY_DASHBOARD_AGENT_PATCHER}"
  remote "${host}" test -f "${HF_CKPT}/config.json"
  remote "${host}" test -f "${PROMPT_DATA}"
}

check_actor_node() {
  local host=$1
  check_node "${host}"
  remote "${host}" test -f "${LOAD}/latest_checkpointed_iteration.txt"
  remote "${host}" test -f "${LOAD}/release/.metadata"
}

kill_old_processes() {
  local patterns=(
    "[1]d\\.sh"
    "[1]p\\.sh"
    "[s]glang"
    "[V]LLM::EngineCore"
    "[r]ay::SGLangEngine"
    "[r]ay::RolloutManager"
    "[r]ay::MegatronTrainRayActor"
    "[t]rain\\.py"
  )
  local pattern
  for pattern in "${patterns[@]}"; do
    pkill -9 -f "${pattern}" >/dev/null 2>&1 || true
  done
  local host
  for host in "${WORKER_HOSTS[@]}"; do
    ssh ${SSH_OPTS} "${host}" "$(printf "pkill -9 -f '%s' >/dev/null 2>&1 || true; " "${patterns[@]}")" &
  done
  for host in "${PHYSICAL_CLEAN_HOSTS[@]}"; do
    ssh ${PHYSICAL_SSH_OPTS} "${host}" "$(printf "pkill -9 -f '%s' >/dev/null 2>&1 || true; " "${patterns[@]}")" &
  done
  wait
}

external_sglang_kill_cmd() {
  local patterns=(
    "[1]d\\.sh"
    "[1]p\\.sh"
    "/usr/local/bin/[s]glang serve"
  )
  printf "pkill -9 -f '%s' >/dev/null 2>&1 || true; " "${patterns[@]}"
}

start_external_sglang_guard() {
  local kill_cmd
  kill_cmd=$(external_sglang_kill_cmd)
  local host
  for host in "${PHYSICAL_CLEAN_HOSTS[@]}"; do
    ssh ${PHYSICAL_SSH_OPTS} "${host}" \
      "guard_file='${EXTERNAL_SGLANG_GUARD_FILE}'; touch \"\${guard_file}\"; trap 'rm -f \"\${guard_file}\"; exit 0' TERM INT HUP EXIT; end=\$((\$(date +%s) + ${EXTERNAL_SGLANG_GUARD_SECS})); while [ -e \"\${guard_file}\" ] && [ \$(date +%s) -lt \${end} ]; do ${kill_cmd} sleep ${EXTERNAL_SGLANG_GUARD_POLL_SECS}; done" &
    EXTERNAL_SGLANG_GUARD_PIDS+=("$!")
  done
}

stop_external_sglang_guard() {
  local stop_pids=()
  local host
  for host in "${PHYSICAL_CLEAN_HOSTS[@]}"; do
    ssh ${PHYSICAL_SSH_OPTS} "${host}" "rm -f '${EXTERNAL_SGLANG_GUARD_FILE}'" >/dev/null 2>&1 &
    stop_pids+=("$!")
  done
  local stop_pid
  for stop_pid in "${stop_pids[@]}"; do
    wait "${stop_pid}" >/dev/null 2>&1 || true
  done

  local pid
  for pid in "${EXTERNAL_SGLANG_GUARD_PIDS[@]}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
  for pid in "${EXTERNAL_SGLANG_GUARD_PIDS[@]}"; do
    wait "${pid}" >/dev/null 2>&1 || true
  done
  EXTERNAL_SGLANG_GUARD_PIDS=()
}

check_gpu_idle() {
  local host
  for host in "${ACTOR_PHYSICAL_HOSTS[@]}" "${ROLLOUT_PHYSICAL_HOSTS[@]}"; do
    ssh ${PHYSICAL_SSH_OPTS} "${host}" "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v host=${host} -v max=${GPU_IDLE_MAX_MIB} '{gsub(/[^0-9]/, \"\", \$1); gsub(/[^0-9.]/, \"\", \$2); if (\$2 + 0 > max) {printf(\"%s gpu %s uses %s MiB > %s MiB\\n\", host, \$1, \$2, max); bad=1}} END {exit bad ? 1 : 0}'"
  done
}

wait_gpu_idle() {
  local deadline=$((SECONDS + GPU_IDLE_WAIT_SECS))
  while true; do
    if check_gpu_idle; then
      return 0
    fi
    if (( SECONDS >= deadline )); then
      echo "GPUs did not become idle within ${GPU_IDLE_WAIT_SECS}s"
      check_gpu_idle
      return 1
    fi
    sleep "${GPU_IDLE_POLL_SECS}"
  done
}

cleanup_ray_cluster() {
  local rc=$?
  trap - EXIT
  stop_external_sglang_guard || true
  if [[ "${CLEANUP_RAY_ON_EXIT}" == "1" && "${RAY_STARTED}" == "1" ]]; then
    echo "=== stopping Ray cluster on exit (rc=${rc}) ===" | tee -a "${LOG}" || true
    ray stop --force >/dev/null 2>&1 || true
    local host
    for host in "${WORKER_HOSTS[@]}"; do
      ssh ${SSH_OPTS} "${host}" "ray stop --force >/dev/null 2>&1 || true" &
    done
    wait || true
    # ray stop reaps Ray actors, but sglang spawns detached scheduler/detokenizer/
    # EngineCore children and the direct driver leaves train.py; pkill those on all
    # physical hosts so a hard failure does not strand orphan compute processes.
    # NOTE: kill_old_processes uses broad `pkill -9 -f` (sglang/ray::/train.py) with no
    # run-id/PGID scoping. On this shared multi-tenant box that can kill OTHER sessions'
    # processes, so it defaults OFF. Enable only when you own the nodes.
    # TODO: scope kills to this run (PGID / env marker) before defaulting on.
    if [[ "${CLEANUP_KILL_ORPHANS_ON_EXIT:-0}" == "1" ]]; then
      echo "=== killing orphan sglang/train processes on exit ===" | tee -a "${LOG}" || true
      kill_old_processes >/dev/null 2>&1 || true
    fi
  fi
  exit "${rc}"
}
trap cleanup_ray_cluster EXIT

echo "=== R6 V4 full-loop smoke sanity ===" | tee "${LOG}"
echo "repo=${REPO}" | tee -a "${LOG}"
echo "hf=${HF_CKPT}" | tee -a "${LOG}"
echo "load=${LOAD}" | tee -a "${LOG}"
echo "prompt_data=${PROMPT_DATA}" | tee -a "${LOG}"
echo "save=${SAVE} save_model=${SAVE_MODEL}" | tee -a "${LOG}"
echo "debug_dir=${DEBUG_DIR}" | tee -a "${LOG}"
echo "actor=${ACTOR_PHYSICAL_HOSTS[*]} rollout=${ROLLOUT_PHYSICAL_HOSTS[*]} rollout_gpus=${ROLLOUT_GPUS}" | tee -a "${LOG}"
echo "placement actor=${ACTOR_PLACEMENT_RESOURCE} rollout=${ROLLOUT_PLACEMENT_RESOURCE}" | tee -a "${LOG}"
echo "pp=${PP_SIZE} ep=${EP_SIZE} moe_router_topk=${MOE_ROUTER_TOPK} first_layers=${FIRST_LAYERS} last_layers=${LAST_LAYERS}" | tee -a "${LOG}"
echo "global_batch_size=${GLOBAL_BATCH_SIZE} rollout_batch_size=${ROLLOUT_BATCH_SIZE} num_rollout=${NUM_ROLLOUT}" | tee -a "${LOG}"
echo "start_rollout_id=${START_ROLLOUT_ID} clean_debug_dir=${CLEAN_DEBUG_DIR}" | tee -a "${LOG}"
echo "use_rollout_routing_replay=${USE_ROLLOUT_ROUTING_REPLAY}" | tee -a "${LOG}"
echo "use_sglang_deepep=${USE_SGLANG_DEEPEP} sglang_dp_size=${SGLANG_DP_SIZE}" | tee -a "${LOG}"
echo "sglang_cuda_graph_max_bs=${SGLANG_CUDA_GRAPH_MAX_BS} sglang_disable_cuda_graph=${SGLANG_DISABLE_CUDA_GRAPH}" | tee -a "${LOG}"
echo "sglang_mhc_env pre=${SGLANG_OPT_USE_TILELANG_MHC_PRE} post=${SGLANG_OPT_USE_TILELANG_MHC_POST} split_sinkhorn=${SGLANG_OPT_USE_TILELANG_MHC_SPLIT_SINKHORN} deepgemm_prenorm=${SGLANG_OPT_DEEPGEMM_HC_PRENORM}" | tee -a "${LOG}"
echo "gpu_idle_max_mib=${GPU_IDLE_MAX_MIB} gpu_idle_wait_secs=${GPU_IDLE_WAIT_SECS}" | tee -a "${LOG}"
echo "external_sglang_guard=${EXTERNAL_SGLANG_GUARD} seconds=${EXTERNAL_SGLANG_GUARD_SECS} poll=${EXTERNAL_SGLANG_GUARD_POLL_SECS}" | tee -a "${LOG}"
echo "ray_agent_register_timeout_ms=${RAY_agent_register_timeout_ms}" | tee -a "${LOG}"
echo "ray_system_config=${RAY_SYSTEM_CONFIG_JSON}" | tee -a "${LOG}"
echo "ray_agent_ports=http:${RAY_DASHBOARD_AGENT_LISTEN_PORT} grpc:${RAY_DASHBOARD_AGENT_GRPC_PORT} runtime_env:${RAY_RUNTIME_ENV_AGENT_PORT}" | tee -a "${LOG}"
echo "ray_run_mode=${RAY_RUN_MODE}" | tee -a "${LOG}"
echo "runtime_cache_root=${RUNTIME_CACHE_ROOT} tmpdir=${TMPDIR}" | tee -a "${LOG}"
echo "ray_dashboard_agent_early_port=${SLIME_RAY_DASHBOARD_AGENT_EARLY_PORT} patcher=${RAY_DASHBOARD_AGENT_PATCHER}" | tee -a "${LOG}"
echo "sglang_mhc_patcher=${SLIME_PATCH_SGLANG_DSV4_MHC_SINKHORN_TORCH} patcher=${SGLANG_MHC_PATCHER}" | tee -a "${LOG}"

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

DSV4_CHAT_TEMPLATE="${REPO}/examples/kernel_agent/prompt_config/deepseek_v4_chat_template.jinja"
if [[ ! -s "${HF_CKPT}/chat_template.jinja" && -f "${DSV4_CHAT_TEMPLATE}" ]]; then
  cp "${DSV4_CHAT_TEMPLATE}" "${HF_CKPT}/chat_template.jinja"
  echo "installed chat_template.jinja -> ${HF_CKPT}/chat_template.jinja" | tee -a "${LOG}"
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
  for host in "${TRAIN_WORKER_HOSTS[@]}"; do
    ssh ${SSH_OPTS} "${host}" "rm -rf ${TILELANG_CACHE_DIR} && mkdir -p ${TILELANG_TMP_DIR}" &
  done
  wait
else
  mkdir -p "${TILELANG_TMP_DIR}"
fi

echo "=== checking selected GPUs are idle ===" | tee -a "${LOG}"
wait_gpu_idle | tee -a "${LOG}"

patch_ray_dashboard_agent() {
  local host=$1
  remote "${host}" python3 "${RAY_DASHBOARD_AGENT_PATCHER}"
}

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

if [[ "${SLIME_PATCH_SGLANG_DSV4_MHC_SINKHORN_TORCH}" == "1" ]]; then
  echo "=== patching SGLang DSV4 MHC split-sinkhorn torch fallback ===" | tee -a "${LOG}"
  python3 "${SGLANG_MHC_PATCHER}" | tee -a "${LOG}"
  patch_pids=()
  for host in "${WORKER_HOSTS[@]}"; do
    remote "${host}" python3 "${SGLANG_MHC_PATCHER}" | tee -a "${LOG}" &
    patch_pids+=("$!")
  done
  patch_rc=0
  for pid in "${patch_pids[@]}"; do
    if ! wait "${pid}"; then
      patch_rc=1
    fi
  done
  if (( patch_rc != 0 )); then
    echo "SGLang MHC patch failed on at least one worker" | tee -a "${LOG}"
    exit 1
  fi
fi

ACTOR_RESOURCE_JSON="{\"${ACTOR_PLACEMENT_RESOURCE}\": ${ACTOR_GPUS_PER_NODE}}"
ROLLOUT_RESOURCE_JSON="{\"${ROLLOUT_PLACEMENT_RESOURCE}\": ${ROLLOUT_GPUS}}"
rm -rf "${RAY_TMP_ROOT}/head"
mkdir -p "${RAY_TMP_ROOT}/head"

echo "=== starting Ray head ${HEAD_IP} ===" | tee -a "${LOG}"
SLIME_RAY_DASHBOARD_AGENT_EARLY_PORT="${SLIME_RAY_DASHBOARD_AGENT_EARLY_PORT}" \
SGLANG_OPT_USE_TILELANG_MHC_PRE="${SGLANG_OPT_USE_TILELANG_MHC_PRE}" \
SGLANG_OPT_USE_TILELANG_MHC_POST="${SGLANG_OPT_USE_TILELANG_MHC_POST}" \
SGLANG_OPT_USE_TILELANG_MHC_SPLIT_SINKHORN="${SGLANG_OPT_USE_TILELANG_MHC_SPLIT_SINKHORN}" \
SGLANG_OPT_DEEPGEMM_HC_PRENORM="${SGLANG_OPT_DEEPGEMM_HC_PRENORM}" \
TMPDIR="${TMPDIR}" \
XDG_CACHE_HOME="${XDG_CACHE_HOME}" \
TRITON_CACHE_DIR="${TRITON_CACHE_DIR}" \
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR}" \
CUDA_CACHE_PATH="${CUDA_CACHE_PATH}" \
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}" \
NCCL_DEBUG="${NCCL_DEBUG:-}" \
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-}" \
NCCL_IB_HCA="${NCCL_IB_HCA:-}" ray start \
  --head \
  --node-ip-address "${HEAD_IP}" \
  --port "${RAY_PORT}" \
  --dashboard-host=0.0.0.0 \
  --dashboard-port "${RAY_DASHBOARD_PORT}" \
  --dashboard-agent-listen-port "${RAY_DASHBOARD_AGENT_LISTEN_PORT}" \
  --dashboard-agent-grpc-port "${RAY_DASHBOARD_AGENT_GRPC_PORT}" \
  --runtime-env-agent-port "${RAY_RUNTIME_ENV_AGENT_PORT}" \
  --system-config "${RAY_SYSTEM_CONFIG_JSON}" \
  --num-gpus "${ACTOR_GPUS_PER_NODE}" \
  --num-cpus "${ACTOR_CPUS_PER_NODE}" \
  --resources "${ACTOR_RESOURCE_JSON}" \
  --object-store-memory "${RAY_OBJECT_STORE_MEMORY}" \
  --disable-usage-stats \
	  --temp-dir "${RAY_TMP_ROOT}/head" | tee -a "${LOG}"
RAY_STARTED=1

start_ray_worker() {
  local host=$1
  local ip=$2
  local num_gpus=$3
  local num_cpus=$4
  local resource_json=$5
  local temp_dir="${RAY_TMP_ROOT}/${host}"

  echo "ray_worker_start host=${host} ip=${ip} gpus=${num_gpus} cpus=${num_cpus} resources=${resource_json}"
  ssh ${SSH_OPTS} "${host}" \
    "ulimit -n 1048576 || true; rm -rf ${temp_dir}; mkdir -p ${temp_dir} ${TMPDIR} ${XDG_CACHE_HOME} ${TRITON_CACHE_DIR} ${TORCHINDUCTOR_CACHE_DIR} ${CUDA_CACHE_PATH}; cd ${REPO} && PYTHONPATH=${PYTHONPATH} PATH=${PATH} SLIME_RAY_DASHBOARD_AGENT_EARLY_PORT=${SLIME_RAY_DASHBOARD_AGENT_EARLY_PORT} SGLANG_OPT_USE_TILELANG_MHC_PRE=${SGLANG_OPT_USE_TILELANG_MHC_PRE} SGLANG_OPT_USE_TILELANG_MHC_POST=${SGLANG_OPT_USE_TILELANG_MHC_POST} SGLANG_OPT_USE_TILELANG_MHC_SPLIT_SINKHORN=${SGLANG_OPT_USE_TILELANG_MHC_SPLIT_SINKHORN} SGLANG_OPT_DEEPGEMM_HC_PRENORM=${SGLANG_OPT_DEEPGEMM_HC_PRENORM} TMPDIR=${TMPDIR} XDG_CACHE_HOME=${XDG_CACHE_HOME} TRITON_CACHE_DIR=${TRITON_CACHE_DIR} TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR} CUDA_CACHE_PATH=${CUDA_CACHE_PATH} RAY_raylet_start_wait_time_s=${RAY_raylet_start_wait_time_s} RAY_agent_register_timeout_ms=${RAY_agent_register_timeout_ms} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} NCCL_DEBUG=${NCCL_DEBUG:-} NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-} NCCL_IB_HCA=${NCCL_IB_HCA:-} ray start --address ${RAY_HEAD_ADDR} --node-ip-address ${ip} --num-gpus ${num_gpus} --num-cpus ${num_cpus} --resources '${resource_json}' --object-store-memory ${RAY_OBJECT_STORE_MEMORY} --dashboard-agent-listen-port ${RAY_DASHBOARD_AGENT_LISTEN_PORT} --dashboard-agent-grpc-port ${RAY_DASHBOARD_AGENT_GRPC_PORT} --runtime-env-agent-port ${RAY_RUNTIME_ENV_AGENT_PORT} --disable-usage-stats --temp-dir ${temp_dir}" 2>&1
}

wait_for_worker_starts() {
  local rc=0
  local pid
  for pid in "$@"; do
    if ! wait "${pid}"; then
      rc=1
    fi
  done
  return "${rc}"
}

echo "=== starting Ray train workers ===" | tee -a "${LOG}"
worker_pids=()
for i in "${!TRAIN_WORKER_HOSTS[@]}"; do
  host=${TRAIN_WORKER_HOSTS[$i]}
  ip=${TRAIN_WORKER_IPS[$i]}
  start_ray_worker "${host}" "${ip}" "${ACTOR_GPUS_PER_NODE}" "${ACTOR_CPUS_PER_NODE}" "${ACTOR_RESOURCE_JSON}" | tee -a "${LOG}" &
  worker_pids+=("$!")
done
wait_for_worker_starts "${worker_pids[@]}"

echo "=== starting Ray rollout workers ===" | tee -a "${LOG}"
worker_pids=()
for i in "${!ROLLOUT_WORKER_HOSTS[@]}"; do
  host=${ROLLOUT_WORKER_HOSTS[$i]}
  ip=${ROLLOUT_WORKER_IPS[$i]}
  start_ray_worker "${host}" "${ip}" "${ROLLOUT_GPUS}" "${ROLLOUT_CPUS_PER_NODE}" "${ROLLOUT_RESOURCE_JSON}" | tee -a "${LOG}" &
  worker_pids+=("$!")
done
wait_for_worker_starts "${worker_pids[@]}"

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

if [[ "${RAY_RUN_MODE}" == "jobapi" ]]; then
  echo "=== waiting for Ray dashboard jobs API ===" | tee -a "${LOG}"
  deadline=$((SECONDS + RAY_DASHBOARD_WAIT_TIMEOUT))
  while true; do
    if python - "http://${HEAD_IP}:${RAY_DASHBOARD_PORT}/api/version" <<'PY'
import sys
import urllib.request

opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
try:
    with opener.open(sys.argv[1], timeout=2) as response:
        raise SystemExit(0 if response.status == 200 else 1)
except Exception:
    raise SystemExit(1)
PY
    then
      break
    fi
    if (( SECONDS > deadline )); then
      echo "Ray dashboard jobs API did not become ready within ${RAY_DASHBOARD_WAIT_TIMEOUT}s" | tee -a "${LOG}"
      exit 1
    fi
    sleep 2
  done
elif [[ "${RAY_RUN_MODE}" != "direct" ]]; then
  echo "Unsupported RAY_RUN_MODE=${RAY_RUN_MODE}; expected direct or jobapi" | tee -a "${LOG}"
  exit 2
fi

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

COMMON_ARGS=(
  --custom-model-provider-path custom_kernels.deepseek_v4.megatron.model_provider.v4_model_provider
  --hf-checkpoint "${HF_CKPT}"
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size "${PP_SIZE}"
  --decoder-first-pipeline-num-layers "${FIRST_LAYERS}"
  --decoder-last-pipeline-num-layers "${LAST_LAYERS}"
  --context-parallel-size 1
  --expert-model-parallel-size "${EP_SIZE}"
  --expert-tensor-parallel-size 1
  --moe-token-dispatcher-type flex
  --moe-flex-dispatcher-backend deepep
  --moe-router-topk "${MOE_ROUTER_TOPK}"
  --moe-router-dtype fp32
  --moe-deepep-num-sms 20
  --bf16
  --qkv-format bshd
  --micro-batch-size 1
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --optimizer muon
  --lr "${LR}"
  --lr-decay-style "${LR_DECAY_STYLE}"
  --weight-decay "${WEIGHT_DECAY}"
  --muon-momentum 0.9
  --muon-num-ns-steps 5
  --muon-tp-mode blockwise
  --accumulate-allreduce-grads-in-fp32
  --ckpt-format torch_dist
  # PG timeout: default 10min killed formal runs — at formal scale the slowest
  # DP replica's first log-prob microbatches (TileLang JIT warmup at the new
  # padded shape, 120-250s/mb observed) can exceed 600s, watchdog SIGABRTs a
  # peer, and survivors see "remote process exited" (2026-07-04 v1+v2 failures).
  --distributed-timeout-minutes "${DISTRIBUTED_TIMEOUT_MINUTES:-120}"
)

# NOTE: Megatron's --recompute-* args are a NO-OP for the V4 custom model (it
# uses a hand-written `for layer in self.layers` forward loop in
# custom_kernels/deepseek_v4/megatron/mcore_model.py, not a Megatron
# TransformerBlock). Real-task long sequences (ctx 8192) OOM the train step;
# fixing that needs activation checkpointing wired into the V4 decoder loop (or a
# reduced batch/context). RECOMPUTE_ARGS is kept empty until that lands.
RECOMPUTE_ARGS=()

SGLANG_ARGS=(
  --rollout-num-gpus "${ROLLOUT_GPUS}"
  --rollout-num-gpus-per-engine "${ROLLOUT_GPUS_PER_ENGINE}"
  --sglang-context-length "${SGLANG_CONTEXT_LENGTH}"
  --sglang-max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
  --sglang-cuda-graph-max-bs "${SGLANG_CUDA_GRAPH_MAX_BS}"
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --sglang-chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE}"
  --sglang-max-prefill-tokens "${SGLANG_MAX_PREFILL_TOKENS}"
  --sglang-disable-custom-all-reduce
  --sglang-watchdog-timeout 2400
  --sglang-decode-log-interval 1
  --router-policy round_robin
  --router-queue-timeout-secs 2400
)
if [[ "${SGLANG_DISABLE_CUDA_GRAPH}" == "1" ]]; then
  SGLANG_ARGS+=(--sglang-disable-cuda-graph)
fi
if [[ "${USE_SGLANG_DEEPEP}" == "1" ]]; then
  SGLANG_ARGS+=(
    --sglang-data-parallel-size "${SGLANG_DP_SIZE}"
    --sglang-enable-dp-attention
    --sglang-moe-a2a-backend deepep
    --sglang-deepep-config "${SGLANG_DEEPEP_CONFIG}"
  )
fi

SAVE_ARGS=()
if [[ "${SAVE_MODEL}" == "1" ]]; then
  SAVE_ARGS=(
    --save "${SAVE}"
    --save-interval "${SAVE_INTERVAL}"
  )
  # SAVE_OPTIM=1 (Gate B / formal resume) persists Muon optimizer + RNG state so
  # a restart continues training instead of resetting momentum. Default 0 keeps
  # the smoke's model-only checkpoint (fast, no resume).
  if [[ "${SAVE_OPTIM:-0}" != "1" ]]; then
    SAVE_ARGS+=(--no-save-optim --no-save-rng)
  fi
fi

# The base torch_dist checkpoint has no optimizer/RNG state, so a cold start must
# skip loading them. LOAD_OPTIM=1 (Gate B resume, with LOAD pointed at a prior
# --save dir that has optimizer state) loads Muon momentum + RNG to continue.
LOAD_ARGS=(--load "${LOAD}")
if [[ "${LOAD_OPTIM:-0}" != "1" ]]; then
  LOAD_ARGS+=(--no-load-optim --no-load-rng)
fi

# Debug flags for fast train-side iteration (skip the ~30min rollout):
#   capture once: SAVE_DEBUG_ROLLOUT_DATA=<path/{rollout_id}.pt> DEBUG_ROLLOUT_ONLY=1
#   then iterate: DEBUG_TRAIN_ONLY=1 LOAD_DEBUG_ROLLOUT_DATA=<path/{rollout_id}.pt>
# (slime: --debug-train-only / --load-debug-rollout-data imply skip_sglang.)
DEBUG_ARGS=()
[[ -n "${SAVE_DEBUG_ROLLOUT_DATA:-}" ]] && DEBUG_ARGS+=(--save-debug-rollout-data "${SAVE_DEBUG_ROLLOUT_DATA}")
[[ -n "${LOAD_DEBUG_ROLLOUT_DATA:-}" ]] && DEBUG_ARGS+=(--load-debug-rollout-data "${LOAD_DEBUG_ROLLOUT_DATA}")
[[ "${DEBUG_ROLLOUT_ONLY:-0}" == "1" ]] && DEBUG_ARGS+=(--debug-rollout-only)
[[ "${DEBUG_TRAIN_ONLY:-0}" == "1" ]] && DEBUG_ARGS+=(--debug-train-only)

ROUTING_REPLAY_ARGS=()
if [[ "${USE_ROLLOUT_ROUTING_REPLAY}" == "1" ]]; then
  ROUTING_REPLAY_ARGS=(--use-rollout-routing-replay)
fi

# --- Task-specific arg assembly (smoke_sft vs rl) --------------------------
# Assembly lives in a sourceable helper so it can be unit-tested without the
# cluster bring-up (tests/deepseek-v4/test_v4_rl_task_args.py). Sets TASK_ARGS.
# shellcheck source=scripts/v4/_v4_task_args.sh
source "${REPO}/scripts/v4/_v4_task_args.sh"
build_v4_task_args

RUNTIME_ENV_JSON=$(python - <<PY
import json, os
env = {
    "PYTHONPATH": "${REPO}:/root/Megatron-LM",
    "CUDA_DEVICE_MAX_CONNECTIONS": os.environ["CUDA_DEVICE_MAX_CONNECTIONS"],
    "PYTORCH_CUDA_ALLOC_CONF": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
    "V4_LORA_DIM": os.environ["V4_LORA_DIM"],
    "V4_LORA_ALPHA": os.environ["V4_LORA_ALPHA"],
    "V4_LORA_DROPOUT": os.environ["V4_LORA_DROPOUT"],
    "PATH": os.environ["PATH"],
    "TILELANG_CACHE_DIR": os.environ["TILELANG_CACHE_DIR"],
    "TILELANG_TMP_DIR": os.environ["TILELANG_TMP_DIR"],
    "TMPDIR": os.environ["TMPDIR"],
    "XDG_CACHE_HOME": os.environ["XDG_CACHE_HOME"],
    "TRITON_CACHE_DIR": os.environ["TRITON_CACHE_DIR"],
    "TORCHINDUCTOR_CACHE_DIR": os.environ["TORCHINDUCTOR_CACHE_DIR"],
    "CUDA_CACHE_PATH": os.environ["CUDA_CACHE_PATH"],
    "GLOO_SOCKET_IFNAME": os.environ["GLOO_SOCKET_IFNAME"],
    "NCCL_SOCKET_IFNAME": os.environ["NCCL_SOCKET_IFNAME"],
    "NO_PROXY": os.environ["NO_PROXY"],
    "no_proxy": os.environ["no_proxy"],
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": os.environ["SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK"],
    "SGLANG_MEMORY_SAVER_CUDA_GRAPH": os.environ["SGLANG_MEMORY_SAVER_CUDA_GRAPH"],
    "SGLANG_DSV4_FP4_EXPERTS": os.environ["SGLANG_DSV4_FP4_EXPERTS"],
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": os.environ["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"],
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": os.environ["SGLANG_OPT_USE_TILELANG_MHC_PRE"],
    "SGLANG_OPT_USE_TILELANG_MHC_POST": os.environ["SGLANG_OPT_USE_TILELANG_MHC_POST"],
    "SGLANG_OPT_USE_TILELANG_MHC_SPLIT_SINKHORN": os.environ["SGLANG_OPT_USE_TILELANG_MHC_SPLIT_SINKHORN"],
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": os.environ["SGLANG_OPT_DEEPGEMM_HC_PRENORM"],
    "SGLANG_ROUTER_REGISTRATION_TIMEOUT_SECS": os.environ["SGLANG_ROUTER_REGISTRATION_TIMEOUT_SECS"],
}
# Forward wandb auth only when a key is exported (formal RL runs), else wandb
# logging silently fails to authenticate on the actors. NCCL transport settings
# MUST also reach the sglang engines: the Megatron->SGLang weight-sync NCCL
# group spans train ranks AND engine ranks, and mixed transports (train on TCP
# via NCCL_IB_DISABLE=1, engines on IB) fail instantly with ncclRemoteError.
for _opt in ("WANDB_API_KEY", "NCCL_DEBUG", "NCCL_IB_DISABLE", "NCCL_IB_HCA", "V4_DENSE_ATTENTION"):
    if os.environ.get(_opt):
        env[_opt] = os.environ[_opt]
print(json.dumps({"env_vars": env}))
PY
)

TRAIN_ENV_VARS_JSON=$(python - <<PY
import json, os
keys = [
    "PYTHONPATH",
    "PATH",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "PYTORCH_CUDA_ALLOC_CONF",
    "V4_LORA_DIM",
    "V4_LORA_ALPHA",
    "V4_LORA_DROPOUT",
    "V4_LORA_ADAPTER_ONLY_CKPT",
    "V4_LORA_ADAPTER_RESUME_LOAD",
    "V4_ACT_CKPT",
    "V4_FP8_FROZEN_EXPERTS",
    "V4_FP8_EXPERT_GEMM",
    "V4_FP8_SHARED_EXPERT",
    "V4_FP8_ATTENTION",
    "V4_ACT_CKPT_REENTRANT",
    "V4_EMPTY_CACHE_BETWEEN_PHASES",
    "V4_MHC_MIXING_ORACLE",
    "TILELANG_CACHE_DIR",
    "TILELANG_TMP_DIR",
    "TMPDIR",
    "XDG_CACHE_HOME",
    "TRITON_CACHE_DIR",
    "TORCHINDUCTOR_CACHE_DIR",
    "CUDA_CACHE_PATH",
    "GLOO_SOCKET_IFNAME",
    "NCCL_SOCKET_IFNAME",
    "NCCL_DEBUG",
    "NCCL_IB_DISABLE",
    "NCCL_IB_HCA",
    "NO_PROXY",
    "no_proxy",
    "SLIME_DEBUG_CHECK_PARAMS",
    "SLIME_DEBUG_ANOMALY",
    "SLIME_DEBUG_LOSS",
    "WANDB_API_KEY",
]
print(json.dumps({key: os.environ[key] for key in keys if key in os.environ}))
PY
)

TRAIN_CMD=(
  python3 train.py
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
  --n-samples-per-prompt "${N_SAMPLES_PER_PROMPT}"
  --global-batch-size "${GLOBAL_BATCH_SIZE}"
  --num-rollout "${NUM_ROLLOUT}"
  --start-rollout-id "${START_ROLLOUT_ID}"
  "${DEBUG_ARGS[@]}"
  "${TASK_ARGS[@]}"
)

run_direct_driver() {
  echo "=== running full-loop direct driver ===" | tee -a "${LOG}"
  RAY_ADDRESS="${RAY_HEAD_ADDR}" "${TRAIN_CMD[@]}" 2>&1 | tee -a "${LOG}"
  return "${PIPESTATUS[0]}"
}

if [[ "${RAY_RUN_MODE}" == "direct" ]]; then
  if ! run_direct_driver; then
    echo "Direct driver full-loop smoke failed" | tee -a "${LOG}"
    exit 1
  fi
  if [[ "${TASK_MODE}" == "smoke_sft" ]]; then
    python3 scripts/v4/verify_rollout_dump.py "${DEBUG_DIR}/rollout_0.pt" --expected-samples "$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))" | tee -a "${LOG}"
    echo "=== R6 V4 full-loop smoke PASS ===" | tee -a "${LOG}"
  else
    echo "=== V4 ${TASK_MODE} (${REWARD_MODE}) run finished OK ===" | tee -a "${LOG}"
  fi
  exit 0
fi

echo "=== submitting full-loop job ===" | tee -a "${LOG}"
JOB_ID=${RAY_JOB_ID:-r6_v4_full_loop_$(date -u +%Y%m%d_%H%M%S)_$$}
JOB_LOG_CAPTURE="${SCRATCH}/ray_job_${JOB_ID}.log"
JOB_LOG_CAPTURE_TMP="${JOB_LOG_CAPTURE}.tmp"
LAST_JOB_LOG_LINES=0
echo "ray_job_id=${JOB_ID}" | tee -a "${LOG}"

append_new_job_logs() {
  ray job logs --address="http://${HEAD_IP}:${RAY_DASHBOARD_PORT}" "${JOB_ID}" >"${JOB_LOG_CAPTURE_TMP}" 2>&1 || return 0
  local total_lines
  total_lines=$(wc -l <"${JOB_LOG_CAPTURE_TMP}")
  if (( total_lines > LAST_JOB_LOG_LINES )); then
    sed -n "$((LAST_JOB_LOG_LINES + 1)),${total_lines}p" "${JOB_LOG_CAPTURE_TMP}" | tee -a "${LOG}"
    LAST_JOB_LOG_LINES="${total_lines}"
  fi
  mv -f "${JOB_LOG_CAPTURE_TMP}" "${JOB_LOG_CAPTURE}"
}

submit_ray_job() {
  ray job submit --address="http://${HEAD_IP}:${RAY_DASHBOARD_PORT}" \
    --submission-id="${JOB_ID}" \
    --no-wait \
    --working-dir="${REPO}" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- "${TRAIN_CMD[@]}"
}

submit_rc=1
for ((submit_attempt = 1; submit_attempt <= RAY_JOB_SUBMIT_RETRIES; submit_attempt++)); do
  echo "ray_job_submit_attempt=${submit_attempt}/${RAY_JOB_SUBMIT_RETRIES}" | tee -a "${LOG}"
  if submit_ray_job 2>&1 | tee -a "${LOG}"; then
    submit_rc=0
    break
  fi
  submit_rc=$?
  echo "ray job submit failed rc=${submit_rc}" | tee -a "${LOG}"
  ray status --address "${RAY_HEAD_ADDR}" | tee -a "${LOG}" || true
  if (( submit_attempt < RAY_JOB_SUBMIT_RETRIES )); then
    sleep "${RAY_JOB_SUBMIT_RETRY_SECS}"
  fi
done
if (( submit_rc != 0 )); then
  exit "${submit_rc}"
fi

deadline=$((SECONDS + RAY_JOB_STATUS_TIMEOUT))
status_failures=0
while true; do
  echo "=== ray job status ${JOB_ID} $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "${LOG}"
  status_rc=0
  status_out=$(ray job status "${JOB_ID}" --address="http://${HEAD_IP}:${RAY_DASHBOARD_PORT}" 2>&1) || status_rc=$?
  if [[ "${status_rc}" == "0" ]]; then
    status_failures=0
    echo "${status_out}" | tee -a "${LOG}"
    append_new_job_logs
    if grep -Eiq "SUCCEEDED|succeeded" <<<"${status_out}"; then
      break
    fi
    if grep -Eiq "FAILED|STOPPED|failed|stopped" <<<"${status_out}"; then
      echo "Ray job ${JOB_ID} ended unsuccessfully" | tee -a "${LOG}"
      append_new_job_logs
      exit 1
    fi
  else
    status_failures=$((status_failures + 1))
    echo "ray job status failed rc=${status_rc} consecutive=${status_failures}/${RAY_JOB_STATUS_MAX_FAILURES}" | tee -a "${LOG}"
    echo "${status_out}" | tail -n "${RAY_JOB_LOG_TAIL_LINES}" | tee -a "${LOG}"
    append_new_job_logs
    if (( status_failures >= RAY_JOB_STATUS_MAX_FAILURES )); then
      echo "Ray job ${JOB_ID} status polling failed ${status_failures} consecutive times" | tee -a "${LOG}"
      exit 1
    fi
  fi
  if (( SECONDS >= deadline )); then
    echo "Ray job ${JOB_ID} did not finish within ${RAY_JOB_STATUS_TIMEOUT}s" | tee -a "${LOG}"
    append_new_job_logs
    ray job stop "${JOB_ID}" --address="http://${HEAD_IP}:${RAY_DASHBOARD_PORT}" >/dev/null 2>&1 || true
    exit 1
  fi
  sleep "${RAY_JOB_STATUS_POLL_SECS}"
done

append_new_job_logs
python3 scripts/v4/verify_rollout_dump.py "${DEBUG_DIR}/rollout_0.pt" --expected-samples "$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))" | tee -a "${LOG}"

echo "=== R6 V4 full-loop smoke PASS ===" | tee -a "${LOG}"
