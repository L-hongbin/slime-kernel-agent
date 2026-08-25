#!/bin/bash
# Parameterized DeepSeek-V4-Flash actor train smoke: converted Megatron
# checkpoint, V4 LoRA, Megatron DeepEP, Megatron Muon, and slime
# train.py --debug-train-only.
#
# Defaults reproduce the earlier R3 PP3/EP8 smoke. Override LOAD, PP_SIZE,
# WORKER_HOSTS, ACTOR_NUM_NODES, and LOAD_DEBUG_ROLLOUT_DATA for R5 PP2/EP8
# replay smokes. It runs one synthetic/replay SFT-LoRA train step; it is not an
# RL run.
set -euo pipefail

DATA_PAD_SIZE_MULTIPLIER=128
LORA_DIM=4
LORA_ALPHA=8
LORA_DROPOUT=0.0
LORA_RSLORA=0
LORA_PLUS_LAMBDA=
LORA_SHARED_EXPERT=0
LORA_ADAPTER_RESUME_LOAD=
LORA_CHECKPOINT_MAX_NODE_BYTES=2147483648
while (( "$#" > 0 )); do
  case "$1" in
    --data-pad-size-multiplier)
      [[ "$#" -ge 2 ]] || { echo "FATAL: --data-pad-size-multiplier requires a value" >&2; exit 2; }
      DATA_PAD_SIZE_MULTIPLIER=$2
      shift 2
      ;;
    --lora-dim)
      [[ "$#" -ge 2 ]] || { echo "FATAL: --lora-dim requires a value" >&2; exit 2; }
      LORA_DIM=$2; shift 2 ;;
    --lora-alpha)
      [[ "$#" -ge 2 ]] || { echo "FATAL: --lora-alpha requires a value" >&2; exit 2; }
      LORA_ALPHA=$2; shift 2 ;;
    --lora-dropout)
      [[ "$#" -ge 2 ]] || { echo "FATAL: --lora-dropout requires a value" >&2; exit 2; }
      LORA_DROPOUT=$2; shift 2 ;;
    --lora-rslora) LORA_RSLORA=1; shift ;;
    --no-lora-rslora) LORA_RSLORA=0; shift ;;
    --lora-plus-lambda)
      [[ "$#" -ge 2 ]] || { echo "FATAL: --lora-plus-lambda requires a value" >&2; exit 2; }
      LORA_PLUS_LAMBDA=$2; shift 2 ;;
    --dsv4-lora-shared-expert) LORA_SHARED_EXPERT=1; shift ;;
    --no-dsv4-lora-shared-expert) LORA_SHARED_EXPERT=0; shift ;;
    --lora-adapter-resume-load)
      [[ "$#" -ge 2 ]] || { echo "FATAL: --lora-adapter-resume-load requires a value" >&2; exit 2; }
      LORA_ADAPTER_RESUME_LOAD=$2; shift 2 ;;
    --lora-checkpoint-max-node-bytes)
      [[ "$#" -ge 2 ]] || { echo "FATAL: --lora-checkpoint-max-node-bytes requires a value" >&2; exit 2; }
      LORA_CHECKPOINT_MAX_NODE_BYTES=$2; shift 2 ;;
    *)
      echo "FATAL: unknown launcher argument '$1'" >&2
      exit 2
      ;;
  esac
done
if [[ ! "${DATA_PAD_SIZE_MULTIPLIER}" =~ ^[1-9][0-9]*$ ]]; then
  echo "FATAL: --data-pad-size-multiplier must be a positive integer, got '${DATA_PAD_SIZE_MULTIPLIER}'." >&2
  exit 2
fi

REPO=${REPO:-/nfs/FM/chenshuailin/projects/kernel_agents/slime-v4flash-lora}
TILEKERNELS_DIR=${TILEKERNELS_DIR:-/nfs/FM/chenshuailin/projects/kernel_agents/TileKernels}
HF_CKPT=${HF_CKPT:-/nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8}
LOAD=${LOAD:-/nfs/FM/chenshuailin/checkpoints/sgl-project/DeepSeek-V4-Flash-FP8-r2-pp3-ep8-torch_dist}
SCRATCH=${SCRATCH:-/nfs/FM/chenshuailin/projects/kernel_agents/slime-v4flash-lora/experiments/csl_v4r3}
SAVE=${SAVE:-${SCRATCH}/out}
DEBUG_DIR=${DEBUG_DIR:-${SCRATCH}/debug}
LOG=${LOG:-${REPO}/local_artifacts/deepseek-v4/r2_logs/train_smoke_$(date +%Y%m%d_%H%M%S).log}
LOAD_DEBUG_ROLLOUT_DATA=${LOAD_DEBUG_ROLLOUT_DATA:-}

HEAD_HOST=${HEAD_HOST:-node64_slime}
HEAD_IP=${HEAD_IP:-10.11.2.164}
WORKER_HOSTS=(${WORKER_HOSTS:-node69_slime node70_slime})
WORKER_IPS=(${WORKER_IPS:-10.11.2.169 10.11.2.170})
PHYSICAL_CLEAN_HOSTS=(${PHYSICAL_CLEAN_HOSTS:-node64 node69 node70 node62})
ACTOR_PHYSICAL_HOSTS=(${ACTOR_PHYSICAL_HOSTS:-node64 node69 node70})
ACTOR_NUM_NODES=${ACTOR_NUM_NODES:-3}
ACTOR_GPUS_PER_NODE=${ACTOR_GPUS_PER_NODE:-8}
ACTOR_CPUS_PER_NODE=${ACTOR_CPUS_PER_NODE:-64}
PP_SIZE=${PP_SIZE:-3}
EP_SIZE=${EP_SIZE:-8}
FIRST_LAYERS=${FIRST_LAYERS:-15}
LAST_LAYERS=${LAST_LAYERS:-14}
SAVE_MODEL=${SAVE_MODEL:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-100000}
ASYNC_SAVE=${ASYNC_SAVE:-0}
SAVE_OPTIM=${SAVE_OPTIM:-0}
SAVE_RNG=${SAVE_RNG:-${SAVE_OPTIM}}
LOAD_OPTIM=${LOAD_OPTIM:-0}
LOAD_RNG=${LOAD_RNG:-${LOAD_OPTIM}}
OVERRIDE_OPT_PARAM_SCHEDULER=${OVERRIDE_OPT_PARAM_SCHEDULER:-0}
LR=${LR:-1e-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-8}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-8}
NUM_ROLLOUT=${NUM_ROLLOUT:-2}
START_ROLLOUT_ID=${START_ROLLOUT_ID:-}
USE_ROLLOUT_ROUTING_REPLAY=${USE_ROLLOUT_ROUTING_REPLAY:-0}
if [[ -n "${LOAD_DEBUG_ROLLOUT_DATA}" ]]; then
  USE_ROLLOUT_ROUTING_REPLAY=1
fi
RAY_PORT=${RAY_PORT:-6379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
RAY_TMP_ROOT=${RAY_TMP_ROOT:-/tmp/v4_train_smoke_${RAY_PORT}}
RAY_HEAD_ADDR="${HEAD_IP}:${RAY_PORT}"
RAY_WAIT_TIMEOUT=${RAY_WAIT_TIMEOUT:-300}
RAY_DASHBOARD_WAIT_TIMEOUT=${RAY_DASHBOARD_WAIT_TIMEOUT:-120}
RAY_JOB_STATUS_TIMEOUT=${RAY_JOB_STATUS_TIMEOUT:-7200}
RAY_JOB_STATUS_POLL_SECS=${RAY_JOB_STATUS_POLL_SECS:-15}
RAY_JOB_STATUS_MAX_FAILURES=${RAY_JOB_STATUS_MAX_FAILURES:-5}
RAY_JOB_LOG_POLL_SECS=${RAY_JOB_LOG_POLL_SECS:-60}
RAY_JOB_LOG_TAIL_LINES=${RAY_JOB_LOG_TAIL_LINES:-160}
CLEANUP_RAY_ON_EXIT=${CLEANUP_RAY_ON_EXIT:-1}
RAY_STARTED=0

export PYTHONUNBUFFERED=1
export PATH="/usr/local/cuda/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond0}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_0,mlx5_3,mlx5_4}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export PYTHONPATH="${REPO}:/root/Megatron-LM:${TILEKERNELS_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export TILELANG_CACHE_DIR=${TILELANG_CACHE_DIR:-/tmp/tilelang_cache_v4_r3_smoke}
export TILELANG_TMP_DIR=${TILELANG_TMP_DIR:-${TILELANG_CACHE_DIR}/tmp}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

CLUSTER_NO_PROXY="127.0.0.1,localhost,${HEAD_IP},${WORKER_IPS[*]},node64,node69,node70,node62,node64_slime,node69_slime,node70_slime,node62_slime"
CLUSTER_NO_PROXY=${CLUSTER_NO_PROXY// /,}
export no_proxy="${no_proxy:-},${CLUSTER_NO_PROXY}"
export NO_PROXY="${NO_PROXY:-},${CLUSTER_NO_PROXY}"

mkdir -p "${SCRATCH}" "${DEBUG_DIR}" "$(dirname "${LOG}")"
cd "${REPO}"
RUN_LOCK=${RUN_LOCK:-${REPO}/local_artifacts/deepseek-v4/r2_logs/train_smoke.lock}
exec 9>"${RUN_LOCK}"
if ! flock -n 9; then
  echo "Another train smoke is already running; lock=${RUN_LOCK}" | tee -a "${LOG}"
  exit 75
fi

remote() {
  local host=$1
  shift
  if [[ "${host}" == "${HEAD_HOST}" ]]; then
    "$@"
  else
    ssh "${host}" "$*"
  fi
}

check_node() {
  local host=$1
  remote "${host}" test -d "${REPO}"
  remote "${host}" test -f "${HF_CKPT}/config.json"
  remote "${host}" test -f "${LOAD}/latest_checkpointed_iteration.txt"
  remote "${host}" test -f "${LOAD}/release/.metadata"
}

kill_sglang_processes() {
  pkill -9 -f "[s]glang" >/dev/null 2>&1 || true
  for host in "${WORKER_HOSTS[@]}"; do
    ssh "${host}" "pkill -9 -f '[s]glang' >/dev/null 2>&1 || true" &
  done
  for host in "${PHYSICAL_CLEAN_HOSTS[@]}"; do
    ssh "${host}" "pkill -9 -f '[s]glang' >/dev/null 2>&1 || true" &
  done
  wait
}

check_actor_gpus_idle() {
  local -r max_mib=1024
  local host
  for host in "${ACTOR_PHYSICAL_HOSTS[@]}"; do
    ssh "${host}" "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F, -v host=${host} -v max=${max_mib} '{gsub(/[^0-9]/, \"\", \$1); gsub(/[^0-9.]/, \"\", \$2); if (\$2 + 0 > max) {printf(\"%s gpu %s uses %s MiB > %s MiB\\n\", host, \$1, \$2, max); bad=1}} END {exit bad ? 1 : 0}'"
  done
}

cleanup_ray_cluster() {
  local rc=$?
  trap - EXIT
  if [[ "${CLEANUP_RAY_ON_EXIT}" == "1" && "${RAY_STARTED}" == "1" ]]; then
    echo "=== stopping Ray cluster on exit (rc=${rc}) ===" | tee -a "${LOG}" || true
    ray stop --force >/dev/null 2>&1 || true
    for host in "${WORKER_HOSTS[@]}"; do
      ssh "${host}" "ray stop --force >/dev/null 2>&1 || true" &
    done
    wait || true
  fi
  exit "${rc}"
}
trap cleanup_ray_cluster EXIT

echo "=== V4 train smoke sanity (pp=${PP_SIZE}, ep=${EP_SIZE}, nodes=${ACTOR_NUM_NODES}) ===" | tee "${LOG}"
echo "repo=${REPO}" | tee -a "${LOG}"
echo "hf=${HF_CKPT}" | tee -a "${LOG}"
echo "load=${LOAD}" | tee -a "${LOG}"
echo "save=${SAVE}" | tee -a "${LOG}"
echo "pp=${PP_SIZE} ep=${EP_SIZE} first_layers=${FIRST_LAYERS} last_layers=${LAST_LAYERS} save_model=${SAVE_MODEL} async_save=${ASYNC_SAVE} data_pad_size_multiplier=${DATA_PAD_SIZE_MULTIPLIER}" | tee -a "${LOG}"
echo "lr=${LR} save_optim=${SAVE_OPTIM} save_rng=${SAVE_RNG} load_optim=${LOAD_OPTIM} load_rng=${LOAD_RNG}" | tee -a "${LOG}"
echo "override_opt_param_scheduler=${OVERRIDE_OPT_PARAM_SCHEDULER}" | tee -a "${LOG}"
if [[ "${LORA_DIM}" == "0" ]]; then
  _adapter_only=false
else
  _adapter_only=true
fi
echo "adapter_only=${_adapter_only} adapter_resume=${LORA_ADAPTER_RESUME_LOAD:-none} lora=r${LORA_DIM}/alpha${LORA_ALPHA}/rslora${LORA_RSLORA}/plus${LORA_PLUS_LAMBDA:-off}/shared${LORA_SHARED_EXPERT}" | tee -a "${LOG}"
echo "actor_nodes=${ACTOR_NUM_NODES} gpus_per_node=${ACTOR_GPUS_PER_NODE} cpus_per_node=${ACTOR_CPUS_PER_NODE}" | tee -a "${LOG}"
echo "global_batch_size=${GLOBAL_BATCH_SIZE} rollout_batch_size=${ROLLOUT_BATCH_SIZE} num_rollout=${NUM_ROLLOUT} start_rollout_id=${START_ROLLOUT_ID:-auto}" | tee -a "${LOG}"
echo "load_debug_rollout_data=${LOAD_DEBUG_ROLLOUT_DATA:-none} use_rollout_routing_replay=${USE_ROLLOUT_ROUTING_REPLAY}" | tee -a "${LOG}"
check_node "${HEAD_HOST}"
for host in "${WORKER_HOSTS[@]}"; do
  check_node "${host}"
done

if [[ "${PREPARE_ONLY:-0}" == "1" ]]; then
  echo "PREPARE_ONLY=1: two-node debug-train configuration is ready; Ray/training was not started." | tee -a "${LOG}"
  exit 0
fi

if [[ "${CLEAN_RAY:-1}" == "1" ]]; then
  echo "=== stopping old Ray on actor nodes ===" | tee -a "${LOG}"
  ray stop --force >/dev/null 2>&1 || true
  for host in "${WORKER_HOSTS[@]}"; do
    ssh "${host}" "ray stop --force >/dev/null 2>&1 || true" &
  done
  wait
fi
if [[ "${KILL_SGLANG:-1}" == "1" ]]; then
  echo "=== killing old sglang processes on actor nodes ===" | tee -a "${LOG}"
  kill_sglang_processes
fi
if [[ "${CLEAN_TILELANG_CACHE:-1}" == "1" ]]; then
  echo "=== cleaning TileLang cache on actor nodes: ${TILELANG_CACHE_DIR} ===" | tee -a "${LOG}"
  rm -rf "${TILELANG_CACHE_DIR}" && mkdir -p "${TILELANG_TMP_DIR}"
  for host in "${WORKER_HOSTS[@]}"; do
    ssh "${host}" "rm -rf ${TILELANG_CACHE_DIR} && mkdir -p ${TILELANG_TMP_DIR}" &
  done
  wait
fi

echo "=== starting Ray head ${HEAD_IP} ===" | tee -a "${LOG}"
rm -rf "${RAY_TMP_ROOT}/head"
mkdir -p "${RAY_TMP_ROOT}/head"
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}" ray start \
  --head \
  --node-ip-address "${HEAD_IP}" \
  --port "${RAY_PORT}" \
  --dashboard-host=0.0.0.0 \
  --dashboard-port "${RAY_DASHBOARD_PORT}" \
  --num-gpus "${ACTOR_GPUS_PER_NODE}" \
  --num-cpus "${ACTOR_CPUS_PER_NODE}" \
  --disable-usage-stats \
  --temp-dir "${RAY_TMP_ROOT}/head" | tee -a "${LOG}"
RAY_STARTED=1

echo "=== starting Ray workers ===" | tee -a "${LOG}"
for i in "${!WORKER_HOSTS[@]}"; do
  host=${WORKER_HOSTS[$i]}
  ip=${WORKER_IPS[$i]}
  ssh "${host}" \
    "rm -rf ${RAY_TMP_ROOT}/${host} && mkdir -p ${RAY_TMP_ROOT}/${host} && cd ${REPO} && GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME} ray start --address ${RAY_HEAD_ADDR} --node-ip-address ${ip} --num-gpus ${ACTOR_GPUS_PER_NODE} --num-cpus ${ACTOR_CPUS_PER_NODE} --disable-usage-stats --temp-dir ${RAY_TMP_ROOT}/${host}" | tee -a "${LOG}" &
done
wait

echo "=== waiting for Ray cluster ===" | tee -a "${LOG}"
deadline=$((SECONDS + RAY_WAIT_TIMEOUT))
while true; do
  if python - "${RAY_HEAD_ADDR}" "${ACTOR_NUM_NODES}" "${ACTOR_GPUS_PER_NODE}" <<'PY'
import sys
import ray
addr, nodes, gpus = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
ray.init(address=addr, ignore_reinit_error=True)
alive = [n for n in ray.nodes() if n.get("Alive")]
total_gpus = sum(n.get("Resources", {}).get("GPU", 0) for n in alive)
ok = len(alive) >= nodes and total_gpus >= nodes * gpus
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

if [[ "${KILL_SGLANG:-1}" == "1" ]]; then
  echo "=== rechecking actor GPUs before job submit ===" | tee -a "${LOG}"
  kill_sglang_processes
  check_actor_gpus_idle | tee -a "${LOG}"
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
  --lr-decay-style constant
  --weight-decay 0.0
  --muon-momentum 0.9
  --muon-num-ns-steps 5
  --muon-tp-mode blockwise
  --accumulate-allreduce-grads-in-fp32
  --ckpt-format torch_dist
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
if [[ "${OVERRIDE_OPT_PARAM_SCHEDULER}" == "1" ]]; then
  COMMON_ARGS+=(--override-opt-param-scheduler)
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
LOAD_ARGS=(--load "${LOAD}")
if [[ "${LOAD_OPTIM}" != "1" ]]; then
  LOAD_ARGS+=(--no-load-optim)
fi
if [[ "${LOAD_RNG}" != "1" ]]; then
  LOAD_ARGS+=(--no-load-rng)
fi
START_ARGS=()
if [[ -n "${START_ROLLOUT_ID}" ]]; then
  START_ARGS=(--start-rollout-id "${START_ROLLOUT_ID}")
fi
ROLLOUT_DATA_ARGS=(
  --rollout-function-path custom_kernels.deepseek_v4.megatron.sft_rollout.v4_sft_rollout
  --save-debug-rollout-data "${DEBUG_DIR}/rollout_{rollout_id}.pt"
)
if [[ -n "${LOAD_DEBUG_ROLLOUT_DATA}" ]]; then
  ROLLOUT_DATA_ARGS=(--load-debug-rollout-data "${LOAD_DEBUG_ROLLOUT_DATA}")
fi
ROUTING_REPLAY_ARGS=()
if [[ "${USE_ROLLOUT_ROUTING_REPLAY}" == "1" ]]; then
  ROUTING_REPLAY_ARGS=(--use-rollout-routing-replay)
fi

RUNTIME_ENV_JSON=$(python - <<PY
import json, os
env = {
    "PYTHONPATH": "${REPO}:/root/Megatron-LM:${TILEKERNELS_DIR}",
    "CUDA_DEVICE_MAX_CONNECTIONS": os.environ["CUDA_DEVICE_MAX_CONNECTIONS"],
    "PYTORCH_CUDA_ALLOC_CONF": os.environ["PYTORCH_CUDA_ALLOC_CONF"],
    "PATH": os.environ["PATH"],
    "TILELANG_CACHE_DIR": os.environ["TILELANG_CACHE_DIR"],
    "TILELANG_TMP_DIR": os.environ["TILELANG_TMP_DIR"],
    "GLOO_SOCKET_IFNAME": os.environ["GLOO_SOCKET_IFNAME"],
    "NCCL_SOCKET_IFNAME": os.environ["NCCL_SOCKET_IFNAME"],
    "NCCL_IB_HCA": os.environ["NCCL_IB_HCA"],
    "NCCL_DEBUG": os.environ["NCCL_DEBUG"],
    "NO_PROXY": os.environ["NO_PROXY"],
    "no_proxy": os.environ["no_proxy"],
    "SLIME_DEBUG_CHECK_PARAMS": os.environ.get("SLIME_DEBUG_CHECK_PARAMS", "0"),
    "SLIME_DEBUG_LOSS": os.environ.get("SLIME_DEBUG_LOSS", "0"),
}
print(json.dumps({"env_vars": env}))
PY
)

echo "=== submitting debug-train-only job ===" | tee -a "${LOG}"
# The caller can set START_ROLLOUT_ID/NUM_ROLLOUT explicitly when replaying a
# saved rollout dump. The defaults still run a one-iteration train smoke.
SUBMIT_LOG=$(mktemp)
ray job submit --no-wait --address="http://${HEAD_IP}:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  --actor-num-nodes "${ACTOR_NUM_NODES}" \
  --actor-num-gpus-per-node "${ACTOR_GPUS_PER_NODE}" \
  "${MODEL_ARGS[@]}" "${COMMON_ARGS[@]}" \
  "${LOAD_ARGS[@]}" \
  "${SAVE_ARGS[@]}" \
  "${START_ARGS[@]}" \
  "${ROLLOUT_DATA_ARGS[@]}" \
  "${ROUTING_REPLAY_ARGS[@]}" \
  --save-debug-train-data "${DEBUG_DIR}/train_{rollout_id}_{rank}.pt" \
  --disable-rollout-global-dataset \
  --loss-type sft_loss \
  --calculate-per-token-loss \
  --disable-compute-advantages-and-returns \
  --debug-train-only \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}" \
  --num-rollout "${NUM_ROLLOUT}" 2>&1 | tee -a "${LOG}" | tee "${SUBMIT_LOG}" >/dev/null

JOB_ID=$(python - "${SUBMIT_LOG}" <<'PY'
import re
import sys

text = open(sys.argv[1]).read()
match = re.search(r"Job '([^']+)' submitted successfully", text)
if not match:
    raise SystemExit("Could not parse Ray job id from submit output")
print(match.group(1))
PY
)
rm -f "${SUBMIT_LOG}"
echo "=== submitted Ray job ${JOB_ID}; polling status ===" | tee -a "${LOG}"

deadline=$((SECONDS + RAY_JOB_STATUS_TIMEOUT))
next_log=$SECONDS
status_failures=0
while true; do
  status_rc=0
  status_out=$(ray job status "${JOB_ID}" --address="http://${HEAD_IP}:${RAY_DASHBOARD_PORT}" 2>&1) || status_rc=$?
  if [[ "${status_rc}" == "0" ]]; then
    status_failures=0
    echo "${status_out}" | tee -a "${LOG}"
    status=$(STATUS_OUT="${status_out}" python - <<'PY'
import re
import os

text = os.environ["STATUS_OUT"]
matches = re.findall(r"Status for job '[^']+': ([A-Z_]+)", text)
if matches:
    print(matches[-1])
elif re.search(r"Job '[^']+' succeeded", text, re.IGNORECASE):
    print("SUCCEEDED")
elif re.search(r"Job '[^']+' failed", text, re.IGNORECASE):
    print("FAILED")
elif re.search(r"Job '[^']+' stopped", text, re.IGNORECASE):
    print("STOPPED")
else:
    print("")
PY
)
    case "${status}" in
      SUCCEEDED)
        echo "=== Ray job ${JOB_ID} SUCCEEDED ===" | tee -a "${LOG}"
        ray job logs "${JOB_ID}" --address="http://${HEAD_IP}:${RAY_DASHBOARD_PORT}" 2>&1 \
          | tail -n "${RAY_JOB_LOG_TAIL_LINES}" | tee -a "${LOG}" || true
        break
        ;;
      FAILED|STOPPED)
        echo "=== Ray job ${JOB_ID} terminal status ${status}; last logs ===" | tee -a "${LOG}"
        ray job logs "${JOB_ID}" --address="http://${HEAD_IP}:${RAY_DASHBOARD_PORT}" 2>&1 \
          | tail -n "${RAY_JOB_LOG_TAIL_LINES}" | tee -a "${LOG}" || true
        exit 1
        ;;
    esac
  else
    status_failures=$((status_failures + 1))
    echo "ray job status failed rc=${status_rc} consecutive=${status_failures}/${RAY_JOB_STATUS_MAX_FAILURES}" | tee -a "${LOG}"
    echo "${status_out}" | tail -n "${RAY_JOB_LOG_TAIL_LINES}" | tee -a "${LOG}"
    if (( status_failures >= RAY_JOB_STATUS_MAX_FAILURES )); then
      echo "Ray job ${JOB_ID} status polling failed ${status_failures} consecutive times; last logs if available" | tee -a "${LOG}"
      ray job logs "${JOB_ID}" --address="http://${HEAD_IP}:${RAY_DASHBOARD_PORT}" 2>&1 \
        | tail -n "${RAY_JOB_LOG_TAIL_LINES}" | tee -a "${LOG}" || true
      exit 1
    fi
  fi

  if (( SECONDS >= deadline )); then
    echo "Ray job ${JOB_ID} did not finish within ${RAY_JOB_STATUS_TIMEOUT}s" | tee -a "${LOG}"
    ray job logs "${JOB_ID}" --address="http://${HEAD_IP}:${RAY_DASHBOARD_PORT}" 2>&1 \
      | tail -n "${RAY_JOB_LOG_TAIL_LINES}" | tee -a "${LOG}" || true
    ray job stop "${JOB_ID}" --address="http://${HEAD_IP}:${RAY_DASHBOARD_PORT}" >/dev/null 2>&1 || true
    exit 1
  fi

  if (( SECONDS >= next_log )); then
    ray job logs "${JOB_ID}" --address="http://${HEAD_IP}:${RAY_DASHBOARD_PORT}" 2>&1 \
      | tail -n "${RAY_JOB_LOG_TAIL_LINES}" | tee -a "${LOG}" || true
    next_log=$((SECONDS + RAY_JOB_LOG_POLL_SECS))
  fi
  sleep "${RAY_JOB_STATUS_POLL_SECS}"
done
