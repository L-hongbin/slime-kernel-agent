#!/bin/bash

set -Eex
trap 'status=$?; echo "Script exiting with status ${status} at line ${LINENO}: ${BASH_COMMAND}"' EXIT
trap 'status=$?; echo "ERROR status ${status} at line ${LINENO}: ${BASH_COMMAND}" >&2' ERR

# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# MODEL CONFIG
source "${SCRIPT_DIR}/../../scripts/models/qwen3.5-27B.sh"
# NODE CONFIG
MASTER_ADDR="${MASTER_ADDR:-10.11.2.164}"
REMOTE_HOSTS=(
   "10.11.2.169"
   "10.11.2.170"
)
REMOTE_PORTS=(
   "23422"
   "23422"
)
if [ "${#REMOTE_PORTS[@]}" -ne "${#REMOTE_HOSTS[@]}" ]; then
   echo "REMOTE_PORTS length (${#REMOTE_PORTS[@]}) must match REMOTE_HOSTS length (${#REMOTE_HOSTS[@]})."
   exit 1
fi
NUM_NODES=$((1 + ${#REMOTE_HOSTS[@]}))
GPUS_PER_NODE=8
NUM_GPUS=$((NUM_NODES * GPUS_PER_NODE))
ACTOR_NUM_NODES=2
ACTOR_NUM_GPUS_PER_NODE=8
ACTOR_GPUS=$((ACTOR_NUM_NODES*ACTOR_NUM_GPUS_PER_NODE))
ROLLOUT_GPUS=$((NUM_GPUS-ACTOR_GPUS))
echo "ACTOR_GPUS ${ACTOR_GPUS} ROLLOUT_GPUS ${ROLLOUT_GPUS}"
# EXP CONFIG
MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN:-16384}
EAGLE_DRAFT_TOKENS=${EAGLE_DRAFT_TOKENS:-4}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-${MAX_CONTEXT_LEN}}
SGLANG_MAX_RUNNING_REQUESTS=${SGLANG_MAX_RUNNING_REQUESTS:-32}
SGLANG_WATCHDOG_TIMEOUT=${SGLANG_WATCHDOG_TIMEOUT:-1200}
ROUTER_QUEUE_TIMEOUT_SECS=${ROUTER_QUEUE_TIMEOUT_SECS:-1200}
DEBUG_ROLLOUT_ONLY=${DEBUG_ROLLOUT_ONLY:-0}
if [[ "${DEBUG_ROLLOUT_ONLY}" == "1" ]]; then
   NUM_ROLLOUT=${NUM_ROLLOUT:-1}
else
   NUM_ROLLOUT=${NUM_ROLLOUT:-3000}
fi
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-16}
N_SAMPLES_PER_PROMPT=${N_SAMPLES_PER_PROMPT:-16}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * N_SAMPLES_PER_PROMPT))}
MODEL_NAME="Qwen3.6-27B"
KERNEL_BACKEND="tvm_ffi"
KERNEL_ENV_URL="http://127.0.0.1:20211"


HF_MODEL_PATH="/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B"
MEGATRON_MODEL_PATH="/nfs/FM/chenshuailin/checkpoints/Qwen/Qwen3.6-27B/torch_dist_tp4_pp2"
      
echo "HF_MODEL_PATH=${HF_MODEL_PATH}"
echo "MEGATRON_MODEL_PATH=${MEGATRON_MODEL_PATH}"


RL_DATA="/ms/FM/lihongbin/dataset/CUDA_RL/cuda_rl/prompt_tvm_v2/drkernel_rl_thinking.parquet"

case "${KERNEL_BACKEND}" in
   tvm_ffi)
      KERNEL_BACKEND_LABEL="TVM"
      ;;
   cuda_agent)
      KERNEL_BACKEND_LABEL="CUDA"
      ;;
   *)
      KERNEL_BACKEND_LABEL="${KERNEL_BACKEND^^}"
      KERNEL_BACKEND_LABEL="${KERNEL_BACKEND_LABEL//_/.}"
      ;;
esac
EXP_NAME="FAsync.${KERNEL_BACKEND_LABEL}.${MODEL_NAME}.CTX${MAX_CONTEXT_LEN}"
EXP_ROOT="${REPO_ROOT}/experiments/${EXP_NAME}"

export TENSORBOARD_DIR="${EXP_ROOT}/tensorboard"
DEBUG_DUMP_DIR="${EXP_ROOT}/debug"
echo "TENSORBOARD_DIR ${TENSORBOARD_DIR}"
echo "DEBUG_DUMP_DIR ${DEBUG_DUMP_DIR}"

NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-^lo,docker0}"
LOCAL_GLOO_SOCKET_IFNAME="${LOCAL_GLOO_SOCKET_IFNAME:-bond0}"
REMOTE_GLOO_SOCKET_IFNAMES=(
   "enp76s0f0np0"
   "bond0"
)
if [ "${#REMOTE_GLOO_SOCKET_IFNAMES[@]}" -ne "${#REMOTE_HOSTS[@]}" ]; then
   echo "REMOTE_GLOO_SOCKET_IFNAMES length (${#REMOTE_GLOO_SOCKET_IFNAMES[@]}) must match REMOTE_HOSTS length (${#REMOTE_HOSTS[@]})."
   exit 1
fi

RAY_DASHBOARD_PORT=8265
RAY_PORT=6379
RAY_HEAD_ADDR="${MASTER_ADDR}:${RAY_PORT}"
RAY_TEMP_DIR="/tmp/ray"

PYTHON_BIN=${PYTHON_BIN:-python3}
RAY_WAIT_TIMEOUT=${RAY_WAIT_TIMEOUT:-300}
KERNELGYM_HEALTH_TIMEOUT=${KERNELGYM_HEALTH_TIMEOUT:-5}
KERNELGYM_HEALTH_ATTEMPTS=${KERNELGYM_HEALTH_ATTEMPTS:-3}
KERNELGYM_HEALTH_INTERVAL=${KERNELGYM_HEALTH_INTERVAL:-2}
MIN_CPUS_PER_NODE=${MIN_CPUS_PER_NODE:-64}
CPU_HEALTH_CHECK=${CPU_HEALTH_CHECK:-1}
GPU_OCCUPANCY_CHECK=${GPU_OCCUPANCY_CHECK:-1}
CPU_LOAD_MAX_RATIO=${CPU_LOAD_MAX_RATIO:-0.7}
CPU_IDLE_MIN_PERCENT=${CPU_IDLE_MIN_PERCENT:-50}
CPU_IDLE_WINDOW_SECONDS=${CPU_IDLE_WINDOW_SECONDS:-1}

# LOG CONFIG
LOG_DATE="$(date +%Y%m%d)"
LOG_STAMP="$(date +%Y%m%d.%H%M%S)"
LOG_DIR="${EXP_ROOT}/logs"
PREFLIGHT_DIR="${EXP_ROOT}/preflight"
LOG_PATH="${LOG_DIR}/${LOG_STAMP}.log"
echo "Logging to ${LOG_PATH}"

mkdir -p "${LOG_DIR}" "${TENSORBOARD_DIR}" "${DEBUG_DUMP_DIR}" "${PREFLIGHT_DIR}"
exec >> "${LOG_PATH}" 2>&1

HAS_NVLINK="${HAS_NVLINK:-1}"
echo "HAS_NVLINK: $HAS_NVLINK"
echo "DEBUG_ROLLOUT_ONLY ${DEBUG_ROLLOUT_ONLY}"
echo "NUM_ROLLOUT ${NUM_ROLLOUT}"
echo "ROLLOUT_BATCH_SIZE ${ROLLOUT_BATCH_SIZE}"
echo "N_SAMPLES_PER_PROMPT ${N_SAMPLES_PER_PROMPT}"
echo "GLOBAL_BATCH_SIZE ${GLOBAL_BATCH_SIZE}"
echo "MAX_CONTEXT_LEN ${MAX_CONTEXT_LEN}"
echo "MAX_RESPONSE_LEN ${MAX_RESPONSE_LEN}"

if [[ -n "${SGLANG_WATCHDOG_TIMEOUT}" ]] && (( SGLANG_WATCHDOG_TIMEOUT < 600 )); then
   echo "SGLANG_WATCHDOG_TIMEOUT=${SGLANG_WATCHDOG_TIMEOUT} is too low for 16k kernel-agent rollouts." >&2
   exit 1
fi
if (( ROUTER_QUEUE_TIMEOUT_SECS < 600 )); then
   echo "ROUTER_QUEUE_TIMEOUT_SECS=${ROUTER_QUEUE_TIMEOUT_SECS} is too low for saturated 16k kernel-agent rollouts." >&2
   exit 1
fi
echo "SGLANG_WATCHDOG_TIMEOUT ${SGLANG_WATCHDOG_TIMEOUT}"
echo "ROUTER_QUEUE_TIMEOUT_SECS ${ROUTER_QUEUE_TIMEOUT_SECS}"

echo "Checking KernelGym health at ${KERNEL_ENV_URL}"
KERNELGYM_HEALTH_LOG="${PREFLIGHT_DIR}/kernelgym_health.${LOG_DATE}.$(date +%H%M%S).log"
if ! "${PYTHON_BIN}" "${REPO_ROOT}/scripts/check_kernelgym_health.py" \
   --url "${KERNEL_ENV_URL}" \
   --timeout "${KERNELGYM_HEALTH_TIMEOUT}" \
   --attempts "${KERNELGYM_HEALTH_ATTEMPTS}" \
   --interval "${KERNELGYM_HEALTH_INTERVAL}" > "${KERNELGYM_HEALTH_LOG}" 2>&1; then
   echo "KernelGym health check failed. Output from ${KERNELGYM_HEALTH_LOG}:"
   cat "${KERNELGYM_HEALTH_LOG}"
   exit 1
fi
cat "${KERNELGYM_HEALTH_LOG}"

run_ssh() {
   local host="$1"
   local port="$2"
   shift 2
   ssh -p "${port}" "${host}" "$@"
}

shell_quote() {
   printf "%q" "$1"
}

host_resource_check_args() {
   local -n args_ref=$1
   args_ref=(
      --expected-gpus "${GPUS_PER_NODE}"
      --min-cpus "${MIN_CPUS_PER_NODE}"
      --load-max-ratio "${CPU_LOAD_MAX_RATIO}"
      --idle-min-percent "${CPU_IDLE_MIN_PERCENT}"
      --cpu-window "${CPU_IDLE_WINDOW_SECONDS}"
   )

   if [[ "${CPU_HEALTH_CHECK}" != "1" ]]; then
      args_ref+=(--skip-cpu-health)
   fi
   if [[ "${GPU_OCCUPANCY_CHECK}" != "1" ]]; then
      args_ref+=(--skip-gpu-occupancy)
   fi
}

check_local_host_resources() {
   local label="$1"
   local args=()
   host_resource_check_args args

   "${REPO_ROOT}/scripts/check_host_resources.sh" \
      --label "${label}" \
      "${args[@]}"
}

check_remote_host_resources() {
   local host="$1"
   local port="$2"
   local label="$3"
   local args=()

   host_resource_check_args args
   run_ssh "${host}" "${port}" bash -s -- --label "${label}" "${args[@]}" \
      < "${REPO_ROOT}/scripts/check_host_resources.sh"
}

check_all_host_resources() {
   echo "Checking CPU/GPU resources before Ray start"
   check_local_host_resources "head-${MASTER_ADDR}"
   for i in "${!REMOTE_HOSTS[@]}"; do
      check_remote_host_resources \
         "${REMOTE_HOSTS[$i]}" \
         "${REMOTE_PORTS[$i]}" \
         "worker-${REMOTE_HOSTS[$i]}"
   done
}

wait_for_cluster() {
   echo "Waiting for Ray cluster: expected nodes=${NUM_NODES}, expected GPUs=${NUM_GPUS}"
   local deadline=$((SECONDS + RAY_WAIT_TIMEOUT))
   local ready=0

   while ((SECONDS < deadline)); do
      if "${PYTHON_BIN}" - "${RAY_HEAD_ADDR}" "${NUM_NODES}" "${NUM_GPUS}" <<'PY'
import sys

import ray

ray_address = sys.argv[1]
expected_nodes = int(sys.argv[2])
expected_gpus = float(sys.argv[3])

ray.init(address=ray_address, ignore_reinit_error=True, logging_level="ERROR")
alive_nodes = [node for node in ray.nodes() if node.get("Alive")]
gpu_count = sum(float(node.get("Resources", {}).get("GPU", 0)) for node in alive_nodes)

print(f"Ray alive nodes={len(alive_nodes)}, GPUs={gpu_count:g}")
sys.exit(0 if len(alive_nodes) >= expected_nodes and gpu_count >= expected_gpus else 1)
PY
      then
         ready=1
         break
      fi
      sleep 5
   done

   ray status --address "${RAY_HEAD_ADDR}" || true

   if [[ "${ready}" != "1" ]]; then
      echo "Ray cluster did not become ready within ${RAY_WAIT_TIMEOUT}s." >&2
      exit 1
   fi
}

# for rerun the task
pkill -9 sglang || true
ray stop --force || true
pkill -9 -x raylet || true
pkill -9 -x gcs_server || true
pkill -9 -f "python3? (train|train_async)\\.py" || true
for i in "${!REMOTE_HOSTS[@]}"; do
   run_ssh "${REMOTE_HOSTS[$i]}" "${REMOTE_PORTS[$i]}" "pkill -9 sglang || true; ray stop --force || true; pkill -9 -x raylet || true; pkill -9 -x gcs_server || true; pkill -9 -f 'python3? (train|train_async)\\.py' || true"
done
sleep 3

check_all_host_resources

TENSORBOARD_ARGS=(
   --use-tensorboard
)

LOGGING_ARGS=(
   --log-multi-turn
   --log-memory-to-tensorboard
   --log-timers-to-tensorboard
   --log-throughput
   --log-throughput-to-tensorboard
   --log-progress
   --log-device-memory-used
)

CKPT_ARGS=(
   --hf-checkpoint ${HF_MODEL_PATH}
   --ref-load ${MEGATRON_MODEL_PATH}
   # --load /root/Qwen2.5-3B_slime/
   # --save /root/Qwen2.5-3B_slime/
   # --save-interval 20
)

ROLLOUT_ARGS=(
   --rollout-function-path examples.kernel_agent.fully_async_rollout.generate_rollout_fully_async
   --update-weights-interval 1
   --prompt-data ${RL_DATA}
   --input-key prompt
   --label-key reward_model
   --metadata-key extra_info
   --rollout-shuffle
   --num-rollout ${NUM_ROLLOUT}
   --rollout-batch-size ${ROLLOUT_BATCH_SIZE}
   --n-samples-per-prompt ${N_SAMPLES_PER_PROMPT}
   --rollout-max-response-len $MAX_RESPONSE_LEN
   --rollout-max-context-len $MAX_CONTEXT_LEN
   --apply-chat-template-kwargs '{"enable_thinking":true}'
   --rollout-temperature 1

   # eval args
   # --eval-interval 25
   # --eval-prompt-data nq_test /root/Search-R1/data/nq_hotpotqa_train/test.parquet@[0:3000]
   # # --eval-prompt-data nq_test /root/nq_search/test.parquet
   # --eval-input-key prompt
   # --eval-label-key reward_model
   # --n-samples-per-eval-prompt 1

   --global-batch-size ${GLOBAL_BATCH_SIZE}
   --balance-data
)

# CURRICULUM_ARGS=(
#    --use-dynamic-curriculum
#    --difficulty-level-key difficulty_level
#    --difficulty-score-key difficulty_score
# )
# --num-layers-per-virtual-pipeline-stage 16
# --num-virtual-stages-per-pipeline-rank 2
# --decoder-last-pipeline-num-layers 30

PERF_ARGS=(
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 2
   --decoder-last-pipeline-num-layers 31
   --context-parallel-size 2
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --qwen-gdn-backend flashqla

   --recompute-granularity full
   --recompute-method block
   --recompute-num-layers 25

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --calculate-per-token-loss
   --max-tokens-per-gpu 8192
   --log-probs-max-tokens-per-gpu 16384
   # --init-model-with-meta-device
   --mtp-num-layers 1
   --enable-mtp-training
   --mtp-loss-scaling-factor 0.2
)

RL_ARGS=(
   --advantage-estimator trloo
   --multi-turn-gamma 1.0
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   # whether enabling TIS
   # --use-tis
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.01
   --adam-beta1 0.9
   --adam-beta2 0.98
   --use-distributed-optimizer
   --overlap-grad-reduce
   --overlap-param-gather
   --use-precision-aware-optimizer
   # --optimizer-cpu-offload
   # --overlap-cpu-optimizer-d2h-h2d
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 4
   --sglang-context-length ${MAX_CONTEXT_LEN}
   --sglang-max-running-requests ${SGLANG_MAX_RUNNING_REQUESTS}
   --sglang-mem-fraction-static 0.7
   --sglang-decode-log-interval 400
   --router-policy round_robin
   --router-queue-timeout-secs ${ROUTER_QUEUE_TIMEOUT_SECS}
   --sglang-cuda-graph-max-bs ${SGLANG_MAX_RUNNING_REQUESTS}
   --sglang-disable-custom-all-reduce
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 3
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens ${EAGLE_DRAFT_TOKENS}
   --sglang-linear-attn-backend flashinfer
   --sglang-mamba-scheduler-strategy extra_buffer
)
SGLANG_ARGS+=(--sglang-watchdog-timeout ${SGLANG_WATCHDOG_TIMEOUT})

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --log-probs-chunk-size 10000
   --update-weight-buffer-size 1073741824
   # need to comment this when using model with MLA
   --attention-backend flash
   --no-pin-cpu-grads
   --no-pin-cpu-params
)

CUSTOM_ARGS=(
   --custom-generate-function-path examples.kernel_agent.generate_with_cuda_agent.generate
   --custom-rm-path examples.kernel_agent.generate_with_cuda_agent.reward_func
   --custom-reward-post-process-path examples.kernel_agent.kernel_reward.reward_post_process_by_group
   --dynamic-sampling-filter-path examples.kernel_agent.kernel_filter.filter_cuda_kernel_group
   --multi-turn-prompt-config-path "${SCRIPT_DIR}/prompt_config/multi_turn_cuda_kernel.yaml"
   --rollout-data-postprocess-path examples.kernel_agent.kernel_filter.sequence_mis

   # TIS-related args, recommended to enable when using TIS
   # --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   # --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

DEBUG_ARGS=(
   --dump-details "${DEBUG_DUMP_DIR}"
)
if [[ "${DEBUG_ROLLOUT_ONLY}" == "1" ]]; then
   DEBUG_ARGS+=(--debug-rollout-only)
fi

KERNEL_AGENT_ARGS=(
   --kernel-env-url ${KERNEL_ENV_URL}
   --kernel-backend $KERNEL_BACKEND
   --reference-backend torch
   --do-precheck
   --finalize-mode positive
   --use-multi-turn
   --filter-by-last-turn
   --padding-turns
   --max-turns 1
   --sequence-mis-config '{"aggregation":"turns_geometric","token_veto_threshold":1e-4,"lower":0.999,"upper":1.001,"use_advantage":true}'
   --enable-turns-dp-partitions
   --use-coverage-rs
   --coverage-rs-key time_coverage
   --coverage-rs-threshold 0.3
   --coverage-rs-factor 0.1
)

# launch the master node of ray in container
export MASTER_ADDR
GLOO_SOCKET_IFNAME="${LOCAL_GLOO_SOCKET_IFNAME}" ray start \
   --head \
   --node-ip-address ${MASTER_ADDR} \
   --port ${RAY_PORT} \
   --dashboard-host 0.0.0.0 \
   --dashboard-port $RAY_DASHBOARD_PORT \
   --num-gpus ${GPUS_PER_NODE} \
   --disable-usage-stats \
   --temp-dir=$RAY_TEMP_DIR

for i in "${!REMOTE_HOSTS[@]}"; do
   echo "Starting Ray worker on ${REMOTE_HOSTS[$i]}"
   run_ssh "${REMOTE_HOSTS[$i]}" "${REMOTE_PORTS[$i]}" \
      "GLOO_SOCKET_IFNAME=${REMOTE_GLOO_SOCKET_IFNAMES[$i]} ray start --address ${RAY_HEAD_ADDR} --num-gpus ${GPUS_PER_NODE} --disable-usage-stats"
done

wait_for_cluster

# Cover every node IP in BOTH no_proxy spellings: reqwest (sglang router) and
# other HTTP clients must never reach in-cluster engines through the egress
# proxy — it kills connections that stay silent for ~60s, aborting long
# non-streaming /generate requests.
NO_PROXY_LIST="localhost,127.0.0.1,0.0.0.0,::1,${MASTER_ADDR}$(printf ',%s' "${REMOTE_HOSTS[@]}")"

RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
    "no_proxy": "${NO_PROXY_LIST}",
    "NO_PROXY": "${NO_PROXY_LIST}",
    "NCCL_SOCKET_IFNAME": "${NCCL_SOCKET_IFNAME}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "PYTHONPATH": ".:/root/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "NCCL_DEBUG": "WARN",
    "TENSORBOARD_DIR": "${TENSORBOARD_DIR}"
  }
}
EOF_JSON
)
ray job submit --address="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}" \
   --working-dir="${REPO_ROOT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes ${ACTOR_NUM_NODES} \
   --actor-num-gpus-per-node ${ACTOR_NUM_GPUS_PER_NODE} \
   --rollout-num-gpus "${ROLLOUT_GPUS}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${ROLLOUT_ARGS[@]}" \
   "${CURRICULUM_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${RL_ARGS[@]}" \
   "${TENSORBOARD_ARGS[@]}" \
   "${LOGGING_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   "${DEBUG_ARGS[@]}" \
   "${KERNEL_AGENT_ARGS[@]}" \
   "${CUSTOM_ARGS[@]}"
