#!/bin/bash

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# MODEL CONFIG
source "${SCRIPT_DIR}/../../scripts/models/qwen3.5-27B.sh"
# NODE CONFIG
MASTER_ADDR=""
REMOTE_HOSTS=(
   ""
   ""
   ""
)
REMOTE_PORTS=(
   ""
   ""
   ""
)
if [ "${#REMOTE_PORTS[@]}" -ne "${#REMOTE_HOSTS[@]}" ]; then
   echo "REMOTE_PORTS length (${#REMOTE_PORTS[@]}) must match REMOTE_HOSTS length (${#REMOTE_HOSTS[@]})."
   exit 1
fi
NUM_NODES=$((1 + ${#REMOTE_HOSTS[@]}))
GPUS_PER_NODE=8
NUM_GPUS=$((NUM_NODES * GPUS_PER_NODE))
ACTOR_NUM_NODES=4
ACTOR_NUM_GPUS_PER_NODE=8
ACTOR_GPUS=$((ACTOR_NUM_NODES*ACTOR_NUM_GPUS_PER_NODE))
ROLLOUT_GPUS=$((NUM_GPUS-ACTOR_GPUS))
echo "ACTOR_GPUS ${ACTOR_GPUS} ROLLOUT_GPUS ${ROLLOUT_GPUS}"
# EXP CONFIG
MAX_RESPONSE_LEN=8192
MODEL_NAME="Qwen3.6-27B"
DATASET="Drkernel-rl-thinking-PV4"
KERNEL_BACKEND="cuda_agent"
KERNEL_ENV_URL="http://192.168.116.97:20111"

case "${MODEL_NAME}" in
    Qwen3.6-27B)
      HF_MODEL_PATH="/ms/FM/checkpoints/Qwen-Zoo/Qwen3.6-27B/"
      MEGATRON_MODEL_PATH="/ms/FM/lihongbin/dataset/CUDA_RL/megatron_ckpt/Qwen3.6-27B-TP4-PP2-Torch-Dist"
      ;;
    *)
      echo "Unknown MODEL_NAME: ${MODEL_NAME}" >&2
      exit 1
      ;;
esac
echo "HF_MODEL_PATH=${HF_MODEL_PATH}"
echo "MEGATRON_MODEL_PATH=${MEGATRON_MODEL_PATH}"

case "${DATASET}" in
    Drkernel-rl-thinking-PV4)
      RL_DATA="/ms/FM/lihongbin/dataset/CUDA_RL/cuda_rl/prompt_v4/drkernel_rl_thinking.parquet"
      ;;
    *)
      echo "Unknown DATASET: ${DATASET}" >&2
      exit 1
      ;;
esac

EXP_NAME="Kernel-Async-RL-${KERNEL_BACKEND^^}"
EXP_NAME="${EXP_NAME//_/-}-$MODEL_NAME-$DATASET-Response$MAX_RESPONSE_LEN"

export TENSORBOARD_DIR="/data/FM/lhb/slime-kernel-agent-v0.3.0/tensorboard_log/kernel_agent/${EXP_NAME}"
echo "TENSORBOARD_DIR ${TENSORBOARD_DIR}"

NCCL_SOCKET_IFNAME="front1"
GLOO_SOCKET_IFNAME="front1"

RAY_DASHBOARD_PORT=8265
RAY_PORT=6379
RAY_HEAD_ADDR="${MASTER_ADDR}:${RAY_PORT}"
RAY_TEMP_DIR="/tmp/ray"

PYTHON_BIN=${PYTHON_BIN:-python3}
RAY_WAIT_TIMEOUT=${RAY_WAIT_TIMEOUT:-300}

# LOG CONFIG
LOG_DATE="$(date +%Y%m%d)"
LOG_TIME="$(date +%H%M%S)"
LOG_DIR="${SCRIPT_DIR}/logs/${EXP_NAME}/${LOG_DATE}"
LOG_PATH="${LOG_DIR}/${LOG_TIME}.log"
echo "Logging to ${LOG_PATH}"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_PATH}") 2>&1

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

run_ssh() {
   local host="$1"
   local port="$2"
   shift 2
   ssh -p "${port}" "${host}" "$@"
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
pkill -9 -f ray || true
pkill -9 python || true
for i in "${!REMOTE_HOSTS[@]}"; do
   run_ssh "${REMOTE_HOSTS[$i]}" "${REMOTE_PORTS[$i]}" "pkill -9 sglang || true; ray stop --force || true; pkill -9 ray || true; pkill -9 python || true"
done
sleep 3

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
   --update-weights-interval 1
   --prompt-data ${RL_DATA}
   --input-key prompt
   --label-key reward_model
   --metadata-key extra_info
   --rollout-shuffle
   --num-rollout 3000
   --rollout-batch-size 16
   --n-samples-per-prompt 16
   --rollout-max-response-len $MAX_RESPONSE_LEN
   --rollout-max-context-len 32768
   --rollout-temperature 1

   # eval args
   # --eval-interval 25
   # --eval-prompt-data nq_test /root/Search-R1/data/nq_hotpotqa_train/test.parquet@[0:3000]
   # # --eval-prompt-data nq_test /root/nq_search/test.parquet
   # --eval-input-key prompt
   # --eval-label-key reward_model
   # --n-samples-per-eval-prompt 1

   --global-batch-size 96
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
   --num-virtual-stages-per-pipeline-rank 2
   --context-parallel-size 4
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --calculate-per-token-loss
   --max-tokens-per-gpu 8192
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
)
# --optimizer-cpu-offload
# --overlap-cpu-optimizer-d2h-h2d
# --use-precision-aware-optimizer

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.75
   --sglang-speculative-algorithm EAGLE
   --sglang-speculative-num-steps 3
   --sglang-speculative-eagle-topk 1
   --sglang-speculative-num-draft-tokens 4
   --sglang-mamba-scheduler-strategy extra_buffer
   --sglang-server-concurrency 8
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --log-probs-chunk-size 10000
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
   --multi-turn-prompt-config-path "${SCRIPT_DIR}/prompt_config/initial_prompt/multi_turn_cuda_kernel.yaml"
   --rollout-data-postprocess-path examples.kernel_agent.kernel_filter.sequence_mis

   # TIS-related args, recommended to enable when using TIS
   # --custom-config-path examples/train_infer_mismatch_helper/mis.yaml
   # --custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp
)

KERNEL_AGENT_ARGS=(
   --kernel-env-url ${KERNEL_ENV_URL}
   --kernel-backend $KERNEL_BACKEND
   --reference-backend torch
   --do-precheck
   --finalize-mode positive
   --use-multi-turn
   --filter-by-last-turn
   --padding-turns
   --max-turns 3
   --sequence-mis-config '{"aggregation":"turns_geometric","token_veto_threshold":1e-4,"lower":0.999,"upper":1.001,"use_advantage":true}'
   --enable-turns-dp-partitions
   --use-coverage-rs
   --coverage-rs-key time_coverage
   --coverage-rs-threshold 0.3
   --coverage-rs-factor 0.1
)

# launch the master node of ray in container
export MASTER_ADDR
# NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME}" GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME}" \
ray start \
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
      "ray start --address ${RAY_HEAD_ADDR} --num-gpus ${GPUS_PER_NODE} --disable-usage-stats"
done

wait_for_cluster

RUNTIME_ENV_JSON=$(cat <<EOF_JSON
{
  "env_vars": {
    "no_proxy": "localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}",
    "GLOO_SOCKET_IFNAME": "${GLOO_SOCKET_IFNAME}",
    "TP_SOCKET_IFNAME": "${NCCL_SOCKET_IFNAME}",
    "NCCL_SOCKET_IFNAME": "${NCCL_SOCKET_IFNAME}",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "PYTHONPATH": "/root/Megatron-LM/",
    "CUDA_DEVICE_MAX_CONNECTIONS": "1",
    "NCCL_NVLS_ENABLE": "${HAS_NVLINK}",
    "NCCL_DEBUG": "INFO",
    "NCCL_DEBUG_SUBSYS": "INIT,GRAPH",
    "TENSORBOARD_DIR": "${TENSORBOARD_DIR}"
  }
}
EOF_JSON
)
ray job submit --address="http://${MASTER_ADDR}:${RAY_DASHBOARD_PORT}" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes ${ACTOR_NUM_NODES} \
   --actor-num-gpus-per-node ${ACTOR_NUM_GPUS_PER_NODE} \
   --rollout-num-gpus "${ROLLOUT_GPUS}" \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${CURRICULUM_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${RL_ARGS[@]} \
   ${TENSORBOARD_ARGS[@]} \
   ${LOGGING_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]} \
   ${KERNEL_AGENT_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
